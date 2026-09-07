/* LLVM tier-2 runtime interface & sampling profiler.
 *
 * Hot loops detected at run time are collected as traces of TranslationBlocks,
 * enqueued for asynchronous compilation by a background compiler thread, and
 * executed in place of TCG code until invalidated.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef TCG_LLVM_TIER2_H
#define TCG_LLVM_TIER2_H

#include "qemu/osdep.h"
#include <stdbool.h>
#include <stdint.h>
#include "exec/translation-block.h"

#include "tcg/tcg.h"

#define TIER2_MAX_TRACE_TBS  16
#define TIER2_HOT_THRESHOLD  10000
#define TIER2_SAMPLE_MASK    0xff  /* sample 1 in 256 dispatches */

/* Trace of TranslationBlocks forming a closed loop body */
typedef struct Tier2Trace {
    TranslationBlock *header;
    uint32_t num_tbs;
    TranslationBlock *tbs[TIER2_MAX_TRACE_TBS];
    uint64_t total_exec_count;
} Tier2Trace;

/* Initialize/shutdown the background compiler worker thread & queue */
void tier2_init(void);
void tier2_shutdown(void);
bool tier2_is_enabled(void);

/* Record TB visit in per-vCPU ring buffer for dynamic trace discovery */
void tier2_record_tb_visit(TranslationBlock *tb);

/* Inspect control flow graph / history to extract closed loop trace */
bool tier2_find_loop_trace(TranslationBlock *header, Tier2Trace *out_trace);

/* Enqueue closed loop trace for asynchronous compilation */
void tier2_enqueue_trace(const Tier2Trace *trace);

/* Handler called when a TB crosses the hotness threshold */
void tier2_on_hot_tb(CPUState *cpu, TranslationBlock *tb);

/*
 * Snapshot one TB's post-optimization TCG ops into tier-2 records.
 * Called from tcg_gen_code() on the translating vCPU thread while the
 * op list is still alive. Cheap bounded walk; stores a compact copy for
 * the background compiler thread. Safe to call for every TB.
 */
struct TCGContext;
void tier2_capture_tb_ops(struct TCGContext *s, TranslationBlock *tb);

/*
 * Look up tier-2 code for @tb. Returns the executable entry point, or
 * NULL to keep executing TCG code.
 */
static inline void *tier2_lookup(const TranslationBlock *tb)
{
    return tb ? tb->tier2_code : NULL;
}

/*
 * Drop all tier-2 code derived from @tb. Called from existing TB
 * invalidation paths. Also drops any installed trace that merely
 * CONTAINS @tb (a stale loop body is a correctness bug, not a perf bug).
 */
void tier2_invalidate(TranslationBlock *tb);

/*
 * Drop ALL installed tier-2 code and snapshots. Called from tb_flush
 * (exclusive context, no vCPU running) and from breakpoint insert/remove
 * (tier-2 dispatch bypasses QEMU's breakpoint-page checks, so any
 * breakpoint change must force everything back through TCG).
 * Retire-only: use tier2_reclaim() to release ORC resources once no
 * vCPU can be inside retired code.
 */
void tier2_invalidate_all(void);

/* Release retired ORC modules. Exclusive context only (see above). */
void tier2_reclaim(void);

/* True while op capture stores are still being taken (snapshot budget). */
bool tier2_capture_enabled(void);

/*
 * Fast inline sampling profiler:
 * - Runs on the vCPU thread without mutex or atomic operations.
 * - Increments thread-local counter; bumps tb->exec_count only every 256th hit.
 * - TB structs are always accessed in the RW view here (the view stored in
 *   the hash and handed to dispatch); qemu_thread_jit_write() handles W^X
 *   permission toggling where the host needs it. Do NOT apply
 *   tcg_splitwx_to_rw to TB struct pointers (that conversion is only for
 *   RX code addresses such as exit_tb values).
 * - Triggers trace collection and asynchronous compile enqueue at threshold.
 */
static inline void tier2_profile_sample(CPUState *cpu, TranslationBlock *tb)
{
    static __thread uint32_t sample_tick = 0;
    if (unlikely((++sample_tick & TIER2_SAMPLE_MASK) == 0)) {
        tier2_record_tb_visit(tb);
        if (!tb->tier2_enqueued) {
            qemu_thread_jit_write();
            tb->exec_count += 256;
            if (unlikely(tb->exec_count >= TIER2_HOT_THRESHOLD && !tb->tier2_enqueued)) {
                tier2_on_hot_tb(cpu, tb);
            }
            qemu_thread_jit_execute();
        }
    }
}

#endif /* TCG_LLVM_TIER2_H */
