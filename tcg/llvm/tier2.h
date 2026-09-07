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
#include "qemu/rcu.h"
#include <stdbool.h>
#include <stdint.h>
#include "exec/vaddr.h"
#include "exec/translation-block.h"
#include "tcg/llvm/tier2-jit.h"

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
    /*
     * Provenance for fusion. next[i] = index into tbs[] of the TB that
     * tbs[i] provably jumps to (observed linked via jmp_dest at
     * discovery), or -1. next_slot[i] is the goto_tb slot (0/1) the edge
     * was seen on, or -1. Only proven edges may become internal LLVM
     * branches; everything else side-exits to the dispatcher.
     */
    int32_t next[TIER2_MAX_TRACE_TBS];
    int32_t next_slot[TIER2_MAX_TRACE_TBS];
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
 * Async sample profiler (P6): SIGPROF-driven hotspot requests.
 *
 * A sampler thread wakes at QEMU_TIER2_PROF_HZ (default 500) and SIGPROFs
 * every vCPU. The handler records get_tb_cpu_state()->pc (tb->pc space)
 * into a lock-free SPSC ring; the sampler aggregates per-PC counts with
 * periodic decay and publishes hot PCs here. cpu_tb_exec range-matches
 * each dispatch against the table (one predictable branch when empty)
 * and force-triggers tier2_on_hot_tb on a cover, reusing the entire
 * existing find/enqueue/compile/install pipeline.
 *
 * All cross-thread state is plain data with benign races: a torn read
 * can only cause a spurious compile (valid trace, install guard dedupes)
 * or a dropped sample (statistical profiler tolerates loss). pc==0 is
 * the empty sentinel on both sides (a hot loop with top at guest 0 is
 * not a real program shape).
 */
#define TIER2_PROF_REQUESTS 8
typedef struct Tier2ProfRequest {
    vaddr pc;
    uint32_t age; /* sampler seconds since insertion; stale slots recycled */
} Tier2ProfRequest;
extern Tier2ProfRequest tier2_prof_requests[TIER2_PROF_REQUESTS];
extern bool tier2_has_prof_requests;
void tier2_check_prof_requests(CPUState *cpu, TranslationBlock *tb);

/*
 * Exact-state TB lookup for the P6 sampler (no codegen on miss).
 * Implemented in accel/tcg/cpu-exec.c. Caller must hold rcu_read_lock().
 */
TranslationBlock *tier2_tb_lookup(CPUState *cpu, vaddr pc, uint64_t cs_base,
                                  uint32_t flags, uint32_t cflags);

/*
 * Unblock SIGPROF on the calling vCPU thread (P6 profiler). vCPU threads
 * inherit an all-blocked signal mask from qemu_thread_create; without
 * this the sampler's SIGPROF pends forever and no sample is ever taken.
 * Called once per TCG vCPU thread at startup (rr + mttcg thread fns),
 * so there is zero per-dispatch cost. No-op on Windows.
 */
void tier2_unblock_profiler_signal(void);

/* Installed-trace record for exit resolution. Immutable after publish
 * except via RCU replacement; freed with call_rcu. Protocol constants
 * (TIER2_EXIT_*) live in tier2-jit.h, shared with the JIT side. */
typedef struct Tier2Installed {
    struct rcu_head rcu; /* must be first (call_rcu requirement) */
    uint64_t gen;        /* tier2_snap_gen at install */
    uint32_t num_members;
    void *rx[TIER2_MAX_TRACE_TBS]; /* current rx per trace index */
} Tier2Installed;

/* Current invalidation generation (for the exit-protocol validator). */
uint64_t tier2_snap_gen_current(void);

/* Dispatch-mix accounting for diagnosis (QEMU_TIER2_COUNT=1). */
extern bool tier2_counting;
void tier2_note_dispatch(bool compiled);

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
