/* LLVM tier-2 runtime interface & background compiler thread.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/lockable.h"
#include "qemu/log.h"
#include "qemu/target-info.h"
#include "system/system.h"
#include "exec/cpu-common.h"
#include "exec/memop.h"
#include "exec/memopidx.h"
#include "tcg/tcg-cond.h"
#include <dlfcn.h>
#include <signal.h>
#include "tcg/llvm/tier2.h"
#include "tcg/llvm/tier2-jit.h"
#include "accel/tcg/tb-cpu-state.h"
#include "accel/tcg/cpu-ops.h"
#include "accel/tcg/internal-common.h"
#include "tcg/helper-info.h"
#include "tcg/tcg-ldst.h"
#include "exec/target_page.h"
#include "exec/tlb-common.h"
#include "hw/core/cpu.h"
#include "qemu/cacheflush.h"

#ifndef CONFIG_USER_ONLY
/* Helper address for goto_ptr tail-call classification (same binary). */
typedef struct CPUArchState CPUArchState;
extern const void *helper_lookup_tb_ptr(CPUArchState *env);
#endif

/* P5 safepoint geometry (defined in the profiler section below). */
static void tier2_measure_safepoint(CPUState *cpu);

typedef bool (*Tier2JitInitFn)(void *prologue_fn);
typedef void *(*Tier2JitCompileFn)(const Tier2TraceDesc *trace);
typedef void (*Tier2JitShutdownFn)(void);
typedef void (*Tier2JitInvalidateFn)(void *fn_ptr);
typedef void (*Tier2JitFlushFn)(void);

static Tier2JitInitFn tier2_jit_init_fn;
static Tier2JitCompileFn tier2_jit_compile_fn;
static Tier2JitShutdownFn tier2_jit_shutdown_fn;
static Tier2JitInvalidateFn tier2_jit_invalidate_fn;
static Tier2JitFlushFn tier2_jit_flush_fn;
static void *tier2_dylib_handle;

typedef struct Tier2WorkItem {
    Tier2Trace trace;
    struct Tier2WorkItem *next;
} Tier2WorkItem;

static QemuThread tier2_thread;
static QemuMutex tier2_queue_lock;
static QemuCond tier2_cond;
static Tier2WorkItem *tier2_work_queue;
static bool tier2_initialized;
static bool tier2_thread_started;
static bool tier2_stopping;
static bool tier2_enabled = true;
static bool tier2_verbose;
static bool tier2_guest_mem;
bool tier2_counting;
/*
 * Deprecated workload-loop PC (default: dbc-bench loop top). TBs covering
 * this guest PC are enqueued for compilation at translation time (see
 * capture below). Override with QEMU_TIER2_WORKLOAD_PC=0x... ; 0 disables.
 */
static uint64_t tier2_workload_pc = 0x100210;

/* Op snapshot store: TB* -> Tier2TBRec copy. Written on the translating
 * vCPU thread (capture), read by the background compiler thread.
 * TB structs are only freed at tb_flush; tier2_invalidate_all() bumps
 * snap_gen so the worker can detect a flush between snapshot copy and
 * code install and skip the install. */
static QemuMutex tier2_snap_lock;
static GHashTable *tier2_snaps;    /* TranslationBlock* -> Tier2TBRec* */
static GHashTable *tier2_installed; /* header TB* -> fn ptr (installed code) */
/*
 * Helper address -> static TCGHelperInfo name, for code-cache symbol
 * resolution. Names point at static helper tables (process lifetime);
 * table guarded by snap_lock.
 */
static GHashTable *tier2_helper_names; /* (void*)func -> (const char*)name */
static uint64_t tier2_snap_gen;
static uint32_t tier2_snap_count;
static uint32_t tier2_snap_max = 16384;
static uint64_t tier2_snap_hot_miss;
/*
 * Dispatch mix counters (diagnostic only, enabled by
 * QEMU_TIER2_COUNT=1). Plain globals, no atomics: exact only for a
 * single vCPU, approximate under MTTCG. Read at exit.
 */
static uint64_t tier2_stat_tier2;
static uint64_t tier2_stat_tcg;

void tier2_note_dispatch(bool compiled)
{
    if (compiled) {
        tier2_stat_tier2++;
    } else {
        tier2_stat_tcg++;
    }
}

static void tier2_print_stats(Notifier *notifier, void *data)
{
    (void)notifier;
    (void)data;
    uint64_t total = tier2_stat_tier2 + tier2_stat_tcg;
    fprintf(stderr,
            "[tier2-stats] tier2 dispatches=%llu tcg dispatches=%llu "
            "compiled-share=%.3f\n",
            (unsigned long long)tier2_stat_tier2,
            (unsigned long long)tier2_stat_tcg,
            total ? (double)tier2_stat_tier2 / (double)total : 0.0);
}

static Notifier tier2_exit_notifier = {
    .notify = tier2_print_stats,
};

/* Circular history buffer for dynamic loop detection in the vCPU thread */
#define TIER2_HISTORY_SIZE 64
static __thread TranslationBlock *recent_tbs[TIER2_HISTORY_SIZE];
static __thread uint32_t recent_head = 0;

void tier2_record_tb_visit(TranslationBlock *tb)
{
    recent_tbs[recent_head++ % TIER2_HISTORY_SIZE] = tb;
}

bool tier2_is_enabled(void)
{
    return tier2_enabled;
}

bool tier2_capture_enabled(void)
{
    return tier2_enabled;
}

/*
 * Trace extraction: find the linear sequence of TBs forming a closed loop body
 * (e.g. TB1 -> TB2 -> TB3 -> back to TB1).
 */
static inline vaddr tb_guest_pc(const TranslationBlock *tb);
bool tier2_find_loop_trace(TranslationBlock *header, Tier2Trace *out_trace)
{
    if (!header) {
        return false;
    }
#ifndef CONFIG_USER_ONLY
    /*
     * Skip BIOS ROM, real-mode IVT/BDA, and early bootloader loops (< 1MB).
     * Compiling transient BIOS delay/poll loops wastes host LLVM time during OS boot.
     */
    if (tb_guest_pc(header) < 0x100000) {
        return false;
    }
#endif

    out_trace->header = header;
    out_trace->num_tbs = 0;
    out_trace->total_exec_count = header->exec_count;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        out_trace->next[i] = -1;
        out_trace->next_slot[i] = -1;
    }

    /*
     * Strategy 1: depth-first search of QEMU's block chaining graph
     * (jmp_dest) for a cycle reachable from the header. The old greedy
     * slot-0-first walk died whenever slot 0 pointed at a loop exit;
     * DFS backtracks and tries both slots. Every traversed link is
     * recorded in next[]/next_slot[]: only these proven edges may become
     * internal branches in fused code. Closes to the header are
     * preferred (the hot TB is in the loop); closes to other ancestors
     * (entry+cycle shapes) are accepted as before. Bounded: path holds
     * at most TIER2_MAX_TRACE_TBS TBs, each expanded once (dead-set).
     */
    {
        TranslationBlock *path[TIER2_MAX_TRACE_TBS];
        int8_t path_slot[TIER2_MAX_TRACE_TBS]; /* slot used to reach path[d] */
        TranslationBlock *dead[TIER2_MAX_TRACE_TBS];
        unsigned ndead = 0;
        int depth = 0;
        path[0] = header;
        path_slot[0] = -1;

        while (depth >= 0) {
            TranslationBlock *cur = path[depth];
            TranslationBlock *dest[2];
            for (int i = 0; i < 2; i++) {
                dest[i] = (TranslationBlock *)(cur->jmp_dest[i] & ~3UL);
            }

            /* (1) Header close: the hot TB's own cycle wins. */
            int close_slot = -1;
            if (dest[0] == header) {
                close_slot = 0;
            } else if (dest[1] == header) {
                close_slot = 1;
            }
            int close_j = (close_slot >= 0) ? 0 : -1;

            /* (2) Close to any other ancestor (outermost first). */
            if (close_j < 0) {
                for (int j = 1; j <= depth && close_j < 0; j++) {
                    for (int i = 0; i < 2; i++) {
                        if (dest[i] && dest[i] == path[j]) {
                            close_j = j;
                            close_slot = i;
                            break;
                        }
                    }
                }
            }
            if (close_j >= 0) {
                uint32_t n = (uint32_t)(depth - close_j + 1);
                for (uint32_t k = 0; k < n; k++) {
                    out_trace->tbs[k] = path[close_j + (int)k];
                    if (k + 1 < n) {
                        out_trace->next[k] = (int32_t)(k + 1);
                        out_trace->next_slot[k] =
                            path_slot[close_j + (int)k + 1];
                    } else {
                        out_trace->next[k] = 0;
                        out_trace->next_slot[k] = close_slot;
                    }
                }
                out_trace->num_tbs = n;
                return true;
            }

            /* (3) Descend into the first live, unvisited, untried slot. */
            bool descended = false;
            if (depth + 1 < TIER2_MAX_TRACE_TBS) {
                for (int i = 0; i < 2 && !descended; i++) {
                    TranslationBlock *d = dest[i];
                    if (!d || (tb_cflags(d) & CF_INVALID)) {
                        continue;
                    }
                    bool seen = false;
                    for (int j = 0; j <= depth && !seen; j++) {
                        seen = (path[j] == d);
                    }
                    for (unsigned u = 0; u < ndead && !seen; u++) {
                        seen = (dead[u] == d);
                    }
                    if (!seen) {
                        depth++;
                        path[depth] = d;
                        path_slot[depth] = (int8_t)i;
                        descended = true;
                    }
                }
            }
            if (!descended) {
                /* Dead end: never expand this node again, backtrack. */
                if (ndead < TIER2_MAX_TRACE_TBS) {
                    dead[ndead++] = cur;
                }
                depth--;
            }
        }
    }

    /*
     * Strategy 2: Dynamic backward scan of recent vCPU execution history.
     * Detects loops where block chaining has not yet occurred or indirect jumps.
     */
    uint32_t head = recent_head;
    for (int k = 1; k < TIER2_MAX_TRACE_TBS && k < TIER2_HISTORY_SIZE; k++) {
        int idx = (int)(head - 1 - k) % TIER2_HISTORY_SIZE;
        if (idx < 0) {
            idx += TIER2_HISTORY_SIZE;
        }
        if (recent_tbs[idx] == header) {
            /* header was visited k steps ago; reconstruct trace sequence */
            out_trace->num_tbs = 0;
            for (int j = k; j >= 0; j--) {
                int p = (int)(head - 1 - j) % TIER2_HISTORY_SIZE;
                if (p < 0) {
                    p += TIER2_HISTORY_SIZE;
                }
                if (out_trace->num_tbs < TIER2_MAX_TRACE_TBS) {
                    out_trace->tbs[out_trace->num_tbs++] = recent_tbs[p];
                }
            }
            return true;
        }
    }

    /* No closed loop cycle detected yet */
    return false;
}

static inline vaddr tb_guest_pc(const TranslationBlock *tb)
{
    if (tb_cflags(tb) & CF_PCREL) {
        return (vaddr)tb_page_addr0(tb);
    }
    return tb->pc;
}

/* Enqueue a verified loop trace onto the background compiler FIFO */
void tier2_enqueue_trace(const Tier2Trace *trace)
{
    if (!tier2_initialized || tier2_stopping || !tier2_enabled) {
        return;
    }

    Tier2WorkItem *item = g_new0(Tier2WorkItem, 1);
    item->trace = *trace;
    item->next = NULL;

    qemu_mutex_lock(&tier2_queue_lock);
    if (!tier2_work_queue) {
        tier2_work_queue = item;
    } else {
        Tier2WorkItem *tail = tier2_work_queue;
        while (tail->next) {
            tail = tail->next;
        }
        tail->next = item;
    }
    qemu_cond_signal(&tier2_cond);
    qemu_mutex_unlock(&tier2_queue_lock);

    if (tier2_verbose) {
        /*
         * Log total_exec_count, not header->exec_count: at capture-time
         * enqueue the TB struct is still under construction (exec_count
         * is initialized after codegen), so dereferencing it here would
         * print garbage.
         */
        qemu_log("[tier2] Enqueued loop trace: header=0x%" VADDR_PRIx ", %u TBs, exec_count=%" PRIu64 "\n",
                 tb_guest_pc(trace->header), trace->num_tbs, trace->total_exec_count);
    }
}

/* Handler invoked from vCPU thread when a TB crosses TIER2_HOT_THRESHOLD */
void tier2_on_hot_tb(CPUState *cpu, TranslationBlock *tb)
{
    tier2_measure_safepoint(cpu);
    Tier2Trace trace;
    if (tier2_find_loop_trace(tb, &trace)) {
        /* tb is the RW view (hash/dispatch view); write it directly. */
        tb->tier2_enqueued = true;
        tier2_enqueue_trace(&trace);
    } else {
        /*
         * No closed loop with the current ring contents (cold/polluted
         * history at first crossing). Do NOT mark enqueued: re-arm by
         * dropping below the threshold so detection retries after
         * another threshold's worth of dispatches, when the ring will
         * hold steady-state history. exec_count < threshold+256 here,
         * so no underflow.
         */
        tb->exec_count -= TIER2_HOT_THRESHOLD;
        if (tier2_verbose) {
            qemu_log("[tier2] hot TB 0x%" VADDR_PRIx " has no closed loop yet, "
                     "re-armed (exec_count=%u)\n",
                     tb_guest_pc(tb), tb->exec_count);
        }
    }
}

/* ------------------------------------------------------------------ */
/* P6 async sample profiler: SIGPROF ring + sampler + request table.   */
/* See tier2.h for the benign-racy contract.                           */
/* ------------------------------------------------------------------ */

Tier2ProfRequest tier2_prof_requests[TIER2_PROF_REQUESTS];
bool tier2_has_prof_requests;

/*
 * P5 back-edge safepoint geometry. interrupt_request's offset is a
 * compile-time constant; env->CPUState is measured once from a live
 * vCPU (identical for every CPU of this target; concurrent first
 * measurements store the same value, benign). All consume/enqueue
 * paths below run before their compiles, so worker-built descriptors
 * observe measured values; workload-trigger compiles (translation
 * context, no CPU) may precede measurement and simply leave
 * has_safepoint false (back-edges bail to side-exits, the old
 * behavior).
 */
static bool tier2_sp_measured;
static int64_t tier2_sp_cpu_off;
static int64_t tier2_sp_last_tb_off;

static void tier2_measure_safepoint(CPUState *cpu)
{
    if (tier2_sp_measured || !cpu) {
        return;
    }
    /* void*: CPUArchState's typedef is system-only in this TU. */
    void *env = cpu_env(cpu);
    if (!env) {
        return;
    }
    tier2_sp_cpu_off = (int64_t)(uintptr_t)cpu - (int64_t)(uintptr_t)env;
    tier2_sp_last_tb_off = (int64_t)offsetof(CPUState, last_tier2_tb);
    tier2_sp_measured = true;
}

void tier2_unblock_profiler_signal(void)
{
#ifndef _WIN32
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGPROF);
    pthread_sigmask(SIG_UNBLOCK, &set, NULL);
#endif
}

/*
 * Called from cpu_tb_exec on every dispatch when the table is non-empty.
 * Range-matches the TB's guest span against outstanding profiler
 * requests; on a cover, consumes the slot and force-triggers the normal
 * hot-TB path (unless code is already installed/enqueued). The W^X
 * toggle happens only on an actual match (rare), never per dispatch.
 */
void tier2_check_prof_requests(CPUState *cpu, TranslationBlock *tb)
{
    if (!tier2_has_prof_requests) {
        return;
    }
    tier2_measure_safepoint(cpu);
    vaddr pc = tb_guest_pc(tb);
    uint32_t size = tb->size;
    if (size == 0) {
        return;
    }
    bool matched = false;
    for (int i = 0; i < TIER2_PROF_REQUESTS; i++) {
        vaddr req = tier2_prof_requests[i].pc;
        if (req != 0 && req >= pc && req < pc + (vaddr)size) {
            qemu_thread_jit_write();
            tier2_prof_requests[i].pc = 0;
            if (!tb->tier2_code && !tb->tier2_enqueued) {
                tier2_on_hot_tb(cpu, tb);
            }
            qemu_thread_jit_execute();
            matched = true;
        }
    }
    if (matched) {
        bool any = false;
        for (int i = 0; i < TIER2_PROF_REQUESTS; i++) {
            if (tier2_prof_requests[i].pc != 0) {
                any = true;
                break;
            }
        }
        if (!any) {
            tier2_has_prof_requests = false;
        }
    }
}

#ifndef _WIN32
/* POSIX-only: needs pthread_kill + sigaction. */

#define TIER2_PROF_RING_BITS 12
#define TIER2_PROF_RING_SIZE (1u << TIER2_PROF_RING_BITS)
#define TIER2_PROF_RING_MASK (TIER2_PROF_RING_SIZE - 1)
#define TIER2_PROF_HASH_BITS 9
#define TIER2_PROF_HASH_SIZE (1u << TIER2_PROF_HASH_BITS)
#define TIER2_PROF_HOT_COUNT 16 /* samples in-window to force a compile */
#define TIER2_PROF_REQ_MAX_AGE 5 /* sampler seconds before a stale slot dies */

/*
 * Sample ring. Writers: SIGPROF handlers on vCPU threads (several under
 * MTTCG -- concurrent head updates can drop a sample, which is harmless).
 * Reader: sampler thread only. Aligned head/tail/cells are single-copy
 * on every QEMU host; any torn view just drops/duplicates one sample.
 * The stored CPUState* is safe to use under the sampler's rcu_read_lock
 * (CPUs are RCU-stable); a hot-unplugged CPU simply yields a lookup miss.
 */
typedef struct Tier2ProfSample {
    vaddr pc;
    uint64_t cs_base;
    uint32_t flags;
    uint32_t cflags;
    CPUState *cpu;
} Tier2ProfSample;
static Tier2ProfSample tier2_prof_ring[TIER2_PROF_RING_SIZE];
static unsigned tier2_prof_head;
static unsigned tier2_prof_tail;

static struct {
    vaddr pc;
    uint64_t cs_base;
    uint32_t flags;
    uint32_t cflags;
    CPUState *cpu;
    uint32_t count;
} tier2_prof_hash[TIER2_PROF_HASH_SIZE];

static QemuThread tier2_prof_thread;
static bool tier2_prof_started;
static bool tier2_prof_stopping;
static bool tier2_prof_disabled;
static unsigned tier2_prof_hz = 500;
/* Diagnostic counters (DEBUG=1 at shutdown). Plain data, benign races:
 * the handler only ever increments its own counter. */
static unsigned long tier2_prof_hits;
static unsigned long tier2_prof_drained;
static unsigned long tier2_prof_requests_made;

/*
 * Async-signal-safe: only __thread + env reads (get_tb_cpu_state is a
 * pure reader on the hot dispatch path by construction -- no locks) and
 * plain stores. A torn composite (eip racing translation) yields a
 * slightly-off PC that simply matches nothing.
 */
static void tier2_prof_handler(int sig)
{
    (void)sig;
    CPUState *cpu = current_cpu;
    if (!cpu) {
        return;
    }
    const CPUClass *cc = cpu->cc;
    if (!cc || !cc->tcg_ops || !cc->tcg_ops->get_tb_cpu_state) {
        return;
    }
    TCGTBCPUState st = cc->tcg_ops->get_tb_cpu_state(cpu);
    /*
     * curr_cflags() is pure reads (tcg_cflags + singlestep/one-insn/
     * log-mask state), so it is handler-safe; a racing mode change just
     * yields a lookup miss. Recording sample-time cflags is what makes
     * the sampler's lookup key match the dispatcher's exactly (notably
     * CF_PCREL, set for every TB on this host).
     */
    uint32_t cflags = curr_cflags(cpu);
    unsigned h = tier2_prof_head;
    tier2_prof_ring[h & TIER2_PROF_RING_MASK].pc = st.pc;
    tier2_prof_ring[h & TIER2_PROF_RING_MASK].cs_base = st.cs_base;
    tier2_prof_ring[h & TIER2_PROF_RING_MASK].flags = st.flags;
    tier2_prof_ring[h & TIER2_PROF_RING_MASK].cflags = cflags;
    tier2_prof_ring[h & TIER2_PROF_RING_MASK].cpu = cpu;
    barrier();
    tier2_prof_head = h + 1;
    tier2_prof_hits++;
}

static void tier2_prof_request(vaddr pc)
{
    for (int i = 0; i < TIER2_PROF_REQUESTS; i++) {
        if (tier2_prof_requests[i].pc == pc) {
            tier2_prof_requests[i].age = 0;
            return;
        }
    }
    int slot = -1;
    for (int i = 0; i < TIER2_PROF_REQUESTS; i++) {
        if (tier2_prof_requests[i].pc == 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        /* Table full: evict the stalest request. */
        slot = 0;
        for (int i = 1; i < TIER2_PROF_REQUESTS; i++) {
            if (tier2_prof_requests[i].age >
                tier2_prof_requests[slot].age) {
                slot = i;
            }
        }
    }
    tier2_prof_requests[slot].age = 0;
    barrier();
    tier2_prof_requests[slot].pc = pc;
    tier2_has_prof_requests = true;
    tier2_prof_requests_made++;
    if (tier2_verbose) {
        qemu_log("[tier2-prof] hot pc=0x%" VADDR_PRIx " requested (slot %d)\n",
                 pc, slot);
    }
}

/*
 * Sampler-side consume: force-compile the hot TB right here instead of
 * waiting for a re-dispatch that may never come (chained steady state).
 * Exact-state hash lookup with the recorded key, no codegen on miss (the
 * request table still covers those via the dispatch path). Caller must
 * hold rcu_read_lock().
 */
static void tier2_prof_try_consume(vaddr pc, uint64_t cs_base, uint32_t flags,
                                   uint32_t cflags, CPUState *cpu)
{
    if (!cpu || tier2_stopping || !tier2_enabled) {
        return;
    }
    tier2_measure_safepoint(cpu);
    /*
     * Sample PCs land mid-TB but the htable is keyed by TB start, so an
     * exact lookup usually misses. Scan backwards for the containing
     * TB's start: exact (cs_base, flags, cflags=0) key plus a range
     * check, skipping invalid TBs. Bounded at 1KB back: hot-loop TBs
     * are small, and no TB crosses a guest page. Runs only on hot
     * threshold crossings (rare), on the background sampler thread.
     */
    TranslationBlock *tb = NULL;
    for (unsigned d = 0; d <= 1024 && (vaddr)d <= pc; d++) {
        vaddr try_pc = pc - (vaddr)d;
        TranslationBlock *cand =
            tier2_tb_lookup(cpu, try_pc, cs_base, flags, cflags);
        if (cand && !(tb_cflags(cand) & CF_INVALID) &&
            pc - try_pc < cand->size) {
            tb = cand;
            break;
        }
    }
    if (!tb) {
        if (tier2_verbose) {
            qemu_log("[tier2-prof] consume miss pc=0x%" VADDR_PRIx "\n", pc);
        }
        return;
    }
    if (tb->tier2_code || tb->tier2_enqueued) {
        return;
    }
    qemu_thread_jit_write();
    if (!tb->tier2_code && !tb->tier2_enqueued) {
        /*
         * Same as tier2_on_hot_tb's success branch, but without its
         * re-arm path: exec_count is dispatcher-side currency the
         * sampler must not mint (subtracting the threshold here would
         * wrap the never-counted counter and pollute future dispatch
         * decisions). The single-TB fallback below replaces re-arming.
         */
        Tier2Trace trace;
        if (tier2_find_loop_trace(tb, &trace)) {
            tb->tier2_enqueued = true;
            tier2_enqueue_trace(&trace);
        }
    }
    bool enq = tb->tier2_enqueued;
    if (!enq && !tb->tier2_code) {
        /*
         * No closable loop: chained blind-spot code transfers entirely
         * through jmp_cache (calls, rets, indirect jumps), leaving no
         * jmp_dest links for strategy-1, and the sampler thread owns no
         * dispatch history for strategy-2. Fall back to a single-TB
         * trace so detection still yields compiled code the dispatcher
         * can enter on unchained arrivals. Caveat: the worker install
         * guard lets this single keep a future fused trace out (same
         * header); acceptable today because no chained-mode fused
         * discovery exists yet -- consumption/fusion-through-calls is
         * P5's job, which will need a versioned install guard.
         */
        qemu_mutex_lock(&tier2_snap_lock);
        Tier2TBRec *snap = tier2_snaps ?
            g_hash_table_lookup(tier2_snaps, tb) : NULL;
        bool ok = snap && snap->num_ops > 0 &&
                  !(snap->cflags & CF_USE_ICOUNT);
        qemu_mutex_unlock(&tier2_snap_lock);
        if (ok) {
            Tier2Trace trace;
            trace.header = tb;
            trace.num_tbs = 1;
            trace.tbs[0] = tb;
            trace.total_exec_count = tb->exec_count;
            for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
                trace.next[i] = -1;
                trace.next_slot[i] = -1;
            }
            tb->tier2_enqueued = true;
            tier2_enqueue_trace(&trace);
            enq = true;
        }
    }
    qemu_thread_jit_execute();
    if (tier2_verbose) {
        qemu_log("[tier2-prof] consume pc=0x%" VADDR_PRIx " -> tb=0x%"
                 VADDR_PRIx " size=%u enqueued=%d\n",
                 pc, tb_guest_pc(tb), tb->size, enq);
    }
}

static void tier2_prof_note(const Tier2ProfSample *s)
{
    vaddr pc = s->pc;
    if (pc == 0) {
        return; /* empty sentinel, never aggregated */
    }
    uint32_t h = (uint32_t)(pc ^ (pc >> 32) ^ (pc >> 20));
    h *= 0x9e3779b9u;
    h >>= 32 - TIER2_PROF_HASH_BITS;
    for (int p = 0; p < 8; p++) {
        uint32_t i = (h + p) & (TIER2_PROF_HASH_SIZE - 1);
        if (tier2_prof_hash[i].pc == pc) {
            if (++tier2_prof_hash[i].count == TIER2_PROF_HOT_COUNT) {
                tier2_prof_request(pc);
                /*
                 * Same-pc/different-state samples share the cell; consume
                 * with the first occupant's state (request table covers
                 * the rest via dispatch). Must run under the sampler's
                 * rcu_read_lock (see worker loop).
                 */
                tier2_prof_try_consume(pc, tier2_prof_hash[i].cs_base,
                                       tier2_prof_hash[i].flags,
                                       tier2_prof_hash[i].cflags,
                                       tier2_prof_hash[i].cpu);
            }
            return;
        }
        if (tier2_prof_hash[i].pc == 0) {
            tier2_prof_hash[i].pc = pc;
            tier2_prof_hash[i].cs_base = s->cs_base;
            tier2_prof_hash[i].flags = s->flags;
            tier2_prof_hash[i].cflags = s->cflags;
            tier2_prof_hash[i].cpu = s->cpu;
            tier2_prof_hash[i].count = 1;
            return;
        }
    }
    /* Probe overflow: drop the sample (statistical loss, harmless). */
}

static void tier2_prof_log_top(const char *why)
{
    vaddr top_pc[3] = {0, 0, 0};
    uint32_t top_c[3] = {0, 0, 0};
    unsigned used = 0;
    for (unsigned i = 0; i < TIER2_PROF_HASH_SIZE; i++) {
        uint32_t c = tier2_prof_hash[i].count;
        if (c == 0) {
            continue;
        }
        used++;
        for (int t = 0; t < 3; t++) {
            if (c > top_c[t]) {
                for (int u = 2; u > t; u--) {
                    top_c[u] = top_c[u - 1];
                    top_pc[u] = top_pc[u - 1];
                }
                top_c[t] = c;
                top_pc[t] = tier2_prof_hash[i].pc;
                break;
            }
        }
    }
    qemu_log("[tier2-prof] %s hits=%lu drained=%lu reqs=%lu cells=%u "
             "top=0x%" VADDR_PRIx ":%u 0x%" VADDR_PRIx ":%u 0x%" VADDR_PRIx
             ":%u\n",
             why, tier2_prof_hits, tier2_prof_drained,
             tier2_prof_requests_made, used,
             top_pc[0], top_c[0], top_pc[1], top_c[1],
             top_pc[2], top_c[2]);
}

static void *tier2_prof_worker(void *arg)
{
    (void)arg;
    unsigned tick = 0;
    /* CPU_FOREACH below needs an RCU-registered thread. */
    rcu_register_thread();
    while (!tier2_prof_stopping) {
        struct timespec ts = {
            .tv_sec = 0,
            .tv_nsec = (long)(1000000000ul / tier2_prof_hz),
        };
        nanosleep(&ts, NULL);

        /* Snapshot vCPU thread ids under RCU, signal outside it. */
        pthread_t ids[16];
        int n = 0;
        CPUState *cpu;
        rcu_read_lock();
        CPU_FOREACH(cpu) {
            if (n < 16 && cpu->thread) {
                ids[n++] = cpu->thread->thread;
            }
        }

        /*
         * Drain new samples under the same RCU section: consume does an
         * exact-state TB lookup plus trace discovery on shared TB structs
         * (RCU-stable), and enqueue takes only brief mutexes.
         */
        while (tier2_prof_tail != tier2_prof_head) {
            Tier2ProfSample s =
                tier2_prof_ring[tier2_prof_tail & TIER2_PROF_RING_MASK];
            barrier();
            tier2_prof_tail++;
            tier2_prof_drained++;
            tier2_prof_note(&s);
            /*
             * Drained-count diagnostics (not time-based): short guests
             * finish in <1s, so the 1s-periodic log below never fires
             * for them. First sample proves the signal path; every
             * 512th shows sensitivity progress.
             */
            if (tier2_verbose &&
                (tier2_prof_drained == 1 ||
                 (tier2_prof_drained & 511) == 0)) {
                tier2_prof_log_top("drain");
            }
        }
        rcu_read_unlock();
        for (int i = 0; i < n; i++) {
            pthread_kill(ids[i], SIGPROF);
        }

        if (++tick % tier2_prof_hz == 0) {
            /* ~1s elapsed: decay histogram, age out stale requests. */
            for (unsigned i = 0; i < TIER2_PROF_HASH_SIZE; i++) {
                uint32_t c = tier2_prof_hash[i].count;
                if (c == 0) {
                    continue;
                }
                c >>= 1;
                tier2_prof_hash[i].count = c;
                if (c == 0) {
                    tier2_prof_hash[i].pc = 0;
                }
            }
            bool any = false;
            for (int i = 0; i < TIER2_PROF_REQUESTS; i++) {
                if (tier2_prof_requests[i].pc == 0) {
                    continue;
                }
                if (++tier2_prof_requests[i].age > TIER2_PROF_REQ_MAX_AGE) {
                    tier2_prof_requests[i].pc = 0;
                } else {
                    any = true;
                }
            }
            if (!any) {
                tier2_has_prof_requests = false;
            }
            if (tier2_verbose) {
                tier2_prof_log_top("tick1s");
            }
        }
    }
    rcu_unregister_thread();
    return NULL;
}
#endif /* _WIN32 */

/* ------------------------------------------------------------------ */
/* Op capture: snapshot post-optimization TCG ops into stable records. */
/* Runs on the translating vCPU thread inside tcg_gen_code().          */
/* ------------------------------------------------------------------ */

static inline int tier2_temp_idx(TCGContext *s, TCGTemp *ts)
{
    ptrdiff_t n = ts - s->temps;
    if (n < 0 || n >= s->nb_temps) {
        return -1;
    }
    return (int)n;
}

static inline int tier2_arg_temp(TCGContext *s, TCGArg a)
{
    return tier2_temp_idx(s, arg_temp(a));
}

static unsigned tier2_type_bits(TCGOp *op)
{
    return TCGOP_TYPE(op) == TCG_TYPE_I64 ? 64 : 32;
}

static int tier2_map_cond(TCGArg c)
{
    switch ((TCGCond)c) {
    case TCG_COND_EQ: return T2C_EQ;
    case TCG_COND_NE: return T2C_NE;
    case TCG_COND_LT: return T2C_LT;
    case TCG_COND_GE: return T2C_GE;
    case TCG_COND_GT: return T2C_GT;
    case TCG_COND_LE: return T2C_LE;
    case TCG_COND_LTU: return T2C_LTU;
    case TCG_COND_GEU: return T2C_GEU;
    case TCG_COND_GTU: return T2C_GTU;
    case TCG_COND_LEU: return T2C_LEU;
    case TCG_COND_TSTEQ: return T2C_TSTEQ;
    case TCG_COND_TSTNE: return T2C_TSTNE;
    default: return -1;
    }
}

static void tier2_emit(Tier2TBRec *rec, Tier2Op opc, unsigned bits,
                       int32_t dst, int32_t s1, int32_t s2,
                       int32_t s3, int32_t s4, int64_t i1, int64_t i2)
{
    Tier2OpRec *r = &rec->ops[rec->num_ops++];
    r->op = (uint16_t)opc;
    r->bits = (uint8_t)bits;
    r->_pad = 0;
    r->dst = dst;
    r->src1 = s1;
    r->src2 = s2;
    r->src3 = s3;
    r->src4 = s4;
    r->imm1 = i1;
    r->imm2 = i2;
}

/*
 * Map one TCG op to record(s). Returns false when the op ends the
 * supported prefix (caller stops recording; the walker will bail on the
 * truncated stream, which is always safe). insn_start/discard are
 * skipped (return true, emit nothing).
 */
static bool tier2_capture_op(TCGContext *s, TCGOp *op, Tier2TBRec *rec,
                             int env_idx)
{
    unsigned bits = tier2_type_bits(op);
    int d, a, b;

    switch (op->opc) {
    case INDEX_op_insn_start:
    case INDEX_op_discard:
        return true;
    case INDEX_op_set_label:
        tier2_emit(rec, T2_SETLABEL, 0, -1, -1, -1, -1, -1,
                   arg_label(op->args[0])->id, 0);
        return true;
    case INDEX_op_br:
        tier2_emit(rec, T2_BR, 0, -1, -1, -1, -1, -1,
                   arg_label(op->args[0])->id, 0);
        return true;
    case INDEX_op_brcond:
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_map_cond(op->args[2]);
        if (d < 0 || a < 0 || b < 0) {
            return false;
        }
        tier2_emit(rec, T2_BRCOND, bits, -1, d, a, -1, -1,
                   b, arg_label(op->args[3])->id);
        return true;
    case INDEX_op_mov:
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a < 0) {
            return false;
        }
        tier2_emit(rec, T2_MOV, bits, d, a, -1, -1, -1, 0, 0);
        return true;
    case INDEX_op_add:
    case INDEX_op_sub:
    case INDEX_op_mul:
    case INDEX_op_and:
    case INDEX_op_or:
    case INDEX_op_xor:
    case INDEX_op_shl:
    case INDEX_op_shr:
    case INDEX_op_sar:
    case INDEX_op_rotl:
    case INDEX_op_rotr: {
        static const uint8_t map[] = {
            [INDEX_op_add] = T2_ADD, [INDEX_op_sub] = T2_SUB,
            [INDEX_op_mul] = T2_MUL, [INDEX_op_and] = T2_AND,
            [INDEX_op_or] = T2_OR, [INDEX_op_xor] = T2_XOR,
            [INDEX_op_shl] = T2_SHL, [INDEX_op_shr] = T2_SHR,
            [INDEX_op_sar] = T2_SAR, [INDEX_op_rotl] = T2_ROTL,
            [INDEX_op_rotr] = T2_ROTR,
        };
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        if (d < 0 || a < 0 || b < 0) {
            return false;
        }
        tier2_emit(rec, map[op->opc], bits, d, a, b, -1, -1, 0, 0);
        return true;
    }
    case INDEX_op_neg:
    case INDEX_op_not: {
        Tier2Op t = op->opc == INDEX_op_neg ? T2_NEG : T2_NOT;
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a < 0) {
            return false;
        }
        tier2_emit(rec, t, bits, d, a, -1, -1, -1, 0, 0);
        return true;
    }
    case INDEX_op_extract:
    case INDEX_op_sextract:
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a < 0) {
            return false;
        }
        tier2_emit(rec, op->opc == INDEX_op_extract ? T2_EXTRACT : T2_SEXTRACT,
                   bits, d, a, -1, -1, -1, (int64_t)op->args[2],
                   (int64_t)op->args[3]);
        return true;
    case INDEX_op_deposit: {
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        if (d < 0 || a < 0 || b < 0) {
            return false;
        }
        tier2_emit(rec, T2_DEPOSIT, bits, d, a, b, -1, -1,
                   (int64_t)op->args[3], (int64_t)op->args[4]);
        return true;
    }
    case INDEX_op_ext_i32_i64:
    case INDEX_op_extu_i32_i64:
    case INDEX_op_extrl_i64_i32:
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a < 0) {
            return false;
        }
        tier2_emit(rec, op->opc == INDEX_op_ext_i32_i64 ? T2_EXT32S :
                   op->opc == INDEX_op_extu_i32_i64 ? T2_EXT32U : T2_EXTRL,
                   bits, d, a, -1, -1, -1, 0, 0);
        return true;
    case INDEX_op_setcond:
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        {
            int c = tier2_map_cond(op->args[3]);
            if (d < 0 || a < 0 || b < 0 || c < 0) {
                return false;
            }
            tier2_emit(rec, T2_SETCOND, bits, d, a, b, -1, -1, c, 0);
        }
        return true;
    case INDEX_op_negsetcond:
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        {
            int c = tier2_map_cond(op->args[3]);
            if (d < 0 || a < 0 || b < 0 || c < 0) {
                return false;
            }
            tier2_emit(rec, T2_NEGSETCOND, bits, d, a, b, -1, -1, c, 0);
        }
        return true;
    case INDEX_op_andc:
    case INDEX_op_orc: {
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        if (d < 0 || a < 0 || b < 0) {
            return false;
        }
        tier2_emit(rec, op->opc == INDEX_op_andc ? T2_ANDC : T2_ORC,
                   bits, d, a, b, -1, -1, 0, 0);
        return true;
    }
    case INDEX_op_mulsh:
    case INDEX_op_muluh: {
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        if (d < 0 || a < 0 || b < 0) {
            return false;
        }
        tier2_emit(rec, op->opc == INDEX_op_mulsh ? T2_MULSH : T2_MULUH,
                   bits, d, a, b, -1, -1, 0, 0);
        return true;
    }
    case INDEX_op_clz:
    case INDEX_op_ctz: {
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        b = tier2_arg_temp(s, op->args[2]);
        if (d < 0 || a < 0 || b < 0) {
            return false;
        }
        tier2_emit(rec, op->opc == INDEX_op_clz ? T2_CLZ : T2_CTZ,
                   bits, d, a, b, -1, -1, 0, 0);
        return true;
    }
    case INDEX_op_bswap16:
    case INDEX_op_bswap32:
    case INDEX_op_bswap64: {
        Tier2Op t = op->opc == INDEX_op_bswap16 ? T2_BSWAP16 :
                    op->opc == INDEX_op_bswap32 ? T2_BSWAP32 : T2_BSWAP64;
        int64_t flags = (int64_t)op->args[2];
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a < 0) {
            return false;
        }
        if (t == T2_BSWAP16) {
            /*
             * Strict: require zero-extended input and defined output
             * extension. Anything else leaves high bits backend-defined;
             * bailing beats guessing those bits.
             */
            if (!(flags & TCG_BSWAP_IZ) ||
                !(flags & (TCG_BSWAP_OZ | TCG_BSWAP_OS))) {
                return false;
            }
        }
        tier2_emit(rec, t, bits, d, a, -1, -1, -1, flags, 0);
        return true;
    }
    case INDEX_op_goto_ptr: {
        /*
         * Dynamic transfer (returns, indirect calls, nochain direct
         * jumps via lookup_and_goto_ptr). The address temp usually holds
         * a lookup_tb_ptr result; the walker resolves statically-known
         * targets from the stream and side-exits the rest.
         */
        d = tier2_arg_temp(s, op->args[0]);
        if (d < 0) {
            return false;
        }
        tier2_emit(rec, T2_GOTO_PTR, 0, -1, d, -1, -1, -1, 0, 0);
        return true;
    }
    case INDEX_op_call: {
        /*
         * Direct helper call with exact prototype from TCGHelperInfo.
         * Layout (tcg_gen_callN): args[0..no) = ret temps,
         * args[no..no+ni) = input temps, args[no+ni] = func address,
         * args[no+ni+1] = TCGHelperInfo *. typemask carries 3 bits per
         * position (0 = return slot): 0 void, 2/3 i32, 4/5 i64, 6 ptr,
         * 7 i128 (unsupported). Supported: <=1 out, <=4 in, no 128-bit,
         * no vector temps. Env executes through live memory, so helpers
         * observe correct state; exceptions longjmp past JIT frames the
         * same as past TCG frames.
         */
        int no = TCGOP_CALLO(op);
        int ni = TCGOP_CALLI(op);
        if (no > 1 || ni > 4) {
            return false;
        }
        TCGHelperInfo *info = (TCGHelperInfo *)(uintptr_t)op->args[no + ni + 1];
        void *func = (void *)(uintptr_t)op->args[no + ni];
        if (!func || !info) {
            return false;
        }
        uint32_t mask = info->typemask;
        int codes[5] = {0, 0, 0, 0, 0};
        int t32 = (mask >> 0) & 7;
        if (no == 0) {
            if (t32 != 0) {
                return false;
            }
        } else {
            if (t32 == 7) {
                return false;
            }
            codes[0] = (t32 == 3) ? 2 : (t32 == 5) ? 4 : t32;
            if (codes[0] != 2 && codes[0] != 4 && codes[0] != 6) {
                return false;
            }
        }
        int argids[4] = {-1, -1, -1, -1};
        for (int k = 0; k < ni; k++) {
            int ck = (mask >> ((1 + k) * 3)) & 7;
            if (ck == 7) {
                return false;
            }
            ck = (ck == 3) ? 2 : (ck == 5) ? 4 : ck;
            if (ck != 2 && ck != 4 && ck != 6) {
                return false;
            }
            codes[1 + k] = ck;
            int t = tier2_arg_temp(s, op->args[no + k]);
            if (t < 0) {
                return false;
            }
            TCGTemp *ts = arg_temp(op->args[no + k]);
            if (ts->base_type != TCG_TYPE_I32 &&
                ts->base_type != TCG_TYPE_I64 &&
                ts->base_type != TCG_TYPE_PTR) {
                return false;
            }
            argids[k] = t;
        }
        int rd = -1;
        if (no == 1) {
            rd = tier2_arg_temp(s, op->args[0]);
            if (rd < 0) {
                return false;
            }
            TCGTemp *ts = arg_temp(op->args[0]);
            if (ts->base_type != TCG_TYPE_I32 &&
                ts->base_type != TCG_TYPE_I64 &&
                ts->base_type != TCG_TYPE_PTR) {
                return false;
            }
        }
        int64_t packed = (int64_t)(codes[0] | (codes[1] << 3) |
                                   (codes[2] << 6) | (codes[3] << 9) |
                                   (codes[4] << 12) | (ni << 16) | (no << 20));
        tier2_emit(rec, T2_CALL, 64, rd, argids[0], argids[1], argids[2],
                   argids[3], (int64_t)(uintptr_t)func, packed);
        /*
         * Remember the helper name for this address (code-cache symbol
         * resolution). info->name points at static tables; the table
         * itself is process-global and snap_lock-guarded.
         */
        if (tier2_helper_names && info->name) {
            qemu_mutex_lock(&tier2_snap_lock);
            if (!g_hash_table_contains(tier2_helper_names, func)) {
                g_hash_table_insert(tier2_helper_names, func,
                                    (gpointer)info->name);
            }
            qemu_mutex_unlock(&tier2_snap_lock);
        }
        return true;
    }
    case INDEX_op_movcond: {
        int c1 = tier2_arg_temp(s, op->args[1]);
        int c2 = tier2_arg_temp(s, op->args[2]);
        int v1 = tier2_arg_temp(s, op->args[3]);
        int v2 = tier2_arg_temp(s, op->args[4]);
        int c = tier2_map_cond(op->args[5]);
        d = tier2_arg_temp(s, op->args[0]);
        if (d < 0 || c1 < 0 || c2 < 0 || v1 < 0 || v2 < 0 || c < 0) {
            return false;
        }
        tier2_emit(rec, T2_MOVCOND, bits, d, c1, c2, v1, v2, c, 0);
        return true;
    }
    case INDEX_op_ld8u:
    case INDEX_op_ld8s:
    case INDEX_op_ld16u:
    case INDEX_op_ld16s:
    case INDEX_op_ld32u:
    case INDEX_op_ld32s:
    case INDEX_op_ld:
    case INDEX_op_st8:
    case INDEX_op_st16:
    case INDEX_op_st32:
    case INDEX_op_st: {
        static const uint8_t lmap[] = {
            [INDEX_op_ld8u] = T2_LD8U, [INDEX_op_ld8s] = T2_LD8S,
            [INDEX_op_ld16u] = T2_LD16U, [INDEX_op_ld16s] = T2_LD16S,
            [INDEX_op_ld32u] = T2_LD32U, [INDEX_op_ld32s] = T2_LD32S,
        };
        bool is_st = op->opc == INDEX_op_st8 || op->opc == INDEX_op_st16 ||
                     op->opc == INDEX_op_st32 || op->opc == INDEX_op_st;
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a != env_idx) {
            /* Direct env slot access only; anything else bails. */
            return false;
        }
        if (is_st) {
            Tier2Op t = op->opc == INDEX_op_st8 ? T2_ST8 :
                        op->opc == INDEX_op_st16 ? T2_ST16 :
                        op->opc == INDEX_op_st32 ? T2_ST32 :
                        bits == 64 ? T2_ST64 : T2_ST32;
            tier2_emit(rec, t, bits, -1, d, -1, -1, -1,
                       (int64_t)op->args[2], 0);
        } else {
            Tier2Op t = op->opc == INDEX_op_ld ? (bits == 64 ? T2_LD64 : T2_LD32)
                                               : lmap[op->opc];
            if (t == 0) {
                return false;
            }
            tier2_emit(rec, t, bits, d, -1, -1, -1, -1,
                       (int64_t)op->args[2], 0);
        }
        return true;
    }
    case INDEX_op_exit_tb:
        tier2_emit(rec, T2_EXIT_TB, 0, -1, -1, -1, -1, -1,
                   (int64_t)op->args[0], 0);
        return true;
    case INDEX_op_goto_tb:
        if (op->args[0] > 1) {
            return false;
        }
        tier2_emit(rec, T2_GOTO_TB, 0, -1, -1, -1, -1, -1,
                   (int64_t)op->args[0], 0);
        return true;
    case INDEX_op_qemu_ld:
    case INDEX_op_qemu_st: {
        bool is_st = op->opc == INDEX_op_qemu_st;
        d = tier2_arg_temp(s, op->args[0]);
        a = tier2_arg_temp(s, op->args[1]);
        if (d < 0 || a < 0) {
            return false;
        }
        MemOpIdx oi = (MemOpIdx)op->args[2];
        MemOp memop = get_memop(oi);
        unsigned size = 1u << (memop & MO_SIZE);
        /*
         * imm2 packs everything the JIT needs without TCG headers:
         * access size, sign, address width, byteswap, mmu index, and a
         * precomputed inline-forbidden flag (byteswapped data, or
         * parallel atomics beyond NONE -- decided here with authoritative
         * headers so the JIT never interprets MemOp bits itself).
         */
        TCGTemp *ats = arg_temp(op->args[1]);
        bool addr32 = ats->base_type == TCG_TYPE_I32;
        bool noinline = (memop & MO_BSWAP) ||
            ((rec->cflags & CF_PARALLEL) &&
             (memop & MO_ATOM_MASK) != MO_ATOM_NONE);
        int64_t packed = (int64_t)size | ((memop & MO_SIGN) ? 0x100 : 0) |
                         (addr32 ? 0x200 : 0) |
                         ((memop & MO_BSWAP) ? 0x10000 : 0) |
                         ((int64_t)get_mmuidx(oi) << 24) |
                         (noinline ? (int64_t)0x80000000 : 0);
        if (is_st) {
            tier2_emit(rec, T2_QEMU_ST, bits, -1, d, a, -1, -1,
                       (int64_t)oi, packed);
        } else {
            tier2_emit(rec, T2_QEMU_LD, bits, d, a, -1, -1, -1,
                       (int64_t)oi, packed);
        }
        return true;
    }
    default:
        /* qemu_ld2/st2, exit_req, plugin_*, mb, div/rem, carry ops,
         * vector ops, ...: trace boundary. Stop, don't guess. */
        return false;
    }
}

void tier2_capture_tb_ops(TCGContext *s, TranslationBlock *tb)
{
    /* tier2_initialized implies the snapshot table/mutex exist. Translate
     * can theoretically precede realize; never touch locks/tables before. */
    if (!tier2_enabled || !tier2_initialized || !s || !tb) {
        return;
    }
    if (s->nb_temps > TIER2_JIT_MAX_TEMPS || s->nb_ops > TIER2_JIT_MAX_OPS) {
        return;
    }

    qemu_mutex_lock(&tier2_snap_lock);
    if (tier2_snap_count >= tier2_snap_max) {
        qemu_mutex_unlock(&tier2_snap_lock);
        return;
    }
    qemu_mutex_unlock(&tier2_snap_lock);

    Tier2TBRec *rec = g_new0(Tier2TBRec, 1);
    /*
     * tb->pc is NOT assigned for CF_PCREL TBs (stays zero from fresh
     * pages); tb_guest_pc() resolves the right address in all cases and
     * is valid here (pc/cflags/page_addr are all set pre-codegen).
     */
    vaddr vpc = tb_guest_pc(tb);
    rec->pc = vpc;
    rec->size = tb->size;
    rec->icount = tb->icount;
    rec->cflags = tb->cflags;
    rec->code_ptr = tb->tc.ptr;
    rec->rx_tb = tcg_splitwx_to_rx(tb);

    /* Resolve the env temp so ld/st bases can be validated. */
    int env_idx = tier2_temp_idx(s, tcgv_ptr_temp(tcg_env));

    rec->num_temps = (uint32_t)s->nb_temps;
    for (int i = 0; i < s->nb_temps; i++) {
        TCGTemp *ts = &s->temps[i];
        Tier2TempRec *tr = &rec->temps[i];
        tr->tbits = ts->base_type == TCG_TYPE_I64 ? 64 : 32;
        tr->is_env = (i == env_idx) ? 1 : 0;
        tr->env_off = -1;
        tr->const_val = 0;
        if (ts->kind == TEMP_CONST) {
            tr->is_const = 1;
            tr->const_val = (uint64_t)ts->val;
        } else if (ts->mem_base != NULL) {
            int b = tier2_temp_idx(s, ts->mem_base);
            if (b == env_idx && ts->mem_offset >= INT32_MIN &&
                ts->mem_offset <= INT32_MAX) {
                tr->env_off = (int32_t)ts->mem_offset;
            } else {
                /* Global with non-env base: any use bails the compile. */
                tr->env_off = -2;
            }
        }
    }

    TCGOp *op;
    bool workload_hit = (tier2_workload_pc != 0 && vpc == tier2_workload_pc);
    QTAILQ_FOREACH(op, &s->ops, link) {
        if (rec->num_ops >= TIER2_JIT_MAX_OPS) {
            break;
        }
        /*
         * Workload-PC trigger, evaluated while the ops are alive:
         * dispatcher-sited profiling can never see steady-state chained
         * loops, so the deprecated fast path keys off translation
         * instead. (A general backward-edge trigger was tried here and
         * removed: it flooded the background queue with boot-loop
         * compiles, delaying the traces that matter.)
         */
        if (op->opc == INDEX_op_insn_start) {
            if (tier2_workload_pc != 0 &&
                tcg_get_insn_start_param(op, 0) == tier2_workload_pc) {
                workload_hit = true;
            }
        }
        if (!tier2_capture_op(s, op, rec, env_idx)) {
            /* Unsupported op ends the supported prefix. Log the name
             * (verbose only): this is how new-op coverage gets found. */
            if (tier2_verbose && (unsigned)op->opc < NB_OPS) {
                qemu_log("[tier2] capture stops at %s (tb pc=0x%" VADDR_PRIx
                         ", %u ops kept)\n", tcg_op_defs[op->opc].name,
                         vpc, rec->num_ops);
            }
            break;
        }
    }

    qemu_mutex_lock(&tier2_snap_lock);
    if (tier2_snap_count < tier2_snap_max) {
        g_hash_table_replace(tier2_snaps, tb, rec);
        /* replace() does not tell us insert-vs-replace; recompute rarely */
        tier2_snap_count = (uint32_t)g_hash_table_size(tier2_snaps);
    } else {
        g_free(rec);
    }
    qemu_mutex_unlock(&tier2_snap_lock);

    if (workload_hit) {
        /*
         * Enqueue a single-TB trace immediately: no execution counting
         * needed for the pinned workload PC, and dispatcher sampling
         * would never see it (chained steady state). tier2_enqueued is
         * deliberately left alone -- it is still uninitialized at
         * capture time (translate-all.c sets it after codegen). Repeat
         * compiles are deduped by the worker install guard, and
         * retranslations are rare.
         */
        Tier2Trace trace;
        trace.header = tb;
        trace.num_tbs = 1;
        trace.tbs[0] = tb;
        trace.total_exec_count = 0;
        for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
            trace.next[i] = -1;
        }
        tier2_enqueue_trace(&trace);
        if (tier2_verbose) {
            qemu_log("[tier2] workload-pc enqueue tb=0x%" VADDR_PRIx "\n",
                     vpc);
        }
    }
}

/* ------------------------------------------------------------------ */

static Tier2MemHelpers tier2_helpers;
static bool tier2_helpers_resolved;

/*
 * Measure the softmmu TLB layout for the JIT's inline fast path. Same
 * formula TCG itself uses (tcg/tcg.c:tlb_mask_table_ofs), so the JIT's
 * table walk cannot drift from the backend's. System-mode only: user
 * mode has no softmmu TLB (and its fault path needs real unwind info).
 */
static void tier2_fill_tlb_layout(Tier2TlbLayout *tlb)
{
    memset(tlb, 0, sizeof(*tlb));
#ifdef CONFIG_USER_ONLY
    tlb->valid = false;
    return;
#else
    QEMU_BUILD_BUG_ON(offsetof(CPUTLBDescFast, mask) != 0);
    QEMU_BUILD_BUG_ON(offsetof(CPUTLBDescFast, table) != 8);
    QEMU_BUILD_BUG_ON(sizeof(CPUTLBEntry) != (1 << CPU_TLB_ENTRY_BITS));
    QEMU_BUILD_BUG_ON(offsetof(CPUTLBEntry, addr_read) != 0);
    QEMU_BUILD_BUG_ON(offsetof(CPUTLBEntry, addr_write) != 8);
    QEMU_BUILD_BUG_ON(offsetof(CPUTLBEntry, addend) != 24);
    /* Entry math below addresses uintptr_t fields as i64: 64-bit host. */
    tlb->valid = (sizeof(void *) == 8);
    tlb->f0_off = (int64_t)offsetof(CPUNegativeOffsetState, tlb.f[0]) -
                  (int64_t)sizeof(CPUNegativeOffsetState);
    tlb->f_stride = (int64_t)sizeof(CPUTLBDescFast);
    tlb->n_modes = NB_MMU_MODES;
    tlb->entry_bits = CPU_TLB_ENTRY_BITS;
    tlb->e_read = (int64_t)offsetof(CPUTLBEntry, addr_read);
    tlb->e_write = (int64_t)offsetof(CPUTLBEntry, addr_write);
    tlb->e_addend = (int64_t)offsetof(CPUTLBEntry, addend);
    tlb->page_bits = TARGET_PAGE_BITS;
    tlb->page_mask = TARGET_PAGE_MASK;
#endif
}

static void tier2_resolve_helpers(void)
{
    if (tier2_helpers_resolved) {
        return;
    }
    tier2_helpers_resolved = true;
    /*
     * Direct addresses, not dlsym: on macOS the main binary's symbols
     * are invisible to dlsym(RTLD_DEFAULT) without export flags, which
     * silently disabled every guest-mem compile. These helpers are
     * defined in both system (cputlb.c) and user (user-exec.c) builds
     * via ldst_common.c.inc, so the references link everywhere.
     */
    tier2_helpers.ld8u = helper_ldub_mmu;
    tier2_helpers.ld8s = helper_ldsb_mmu;
    tier2_helpers.ld16u = helper_lduw_mmu;
    tier2_helpers.ld16s = helper_ldsw_mmu;
    tier2_helpers.ld32u = helper_ldul_mmu;
    tier2_helpers.ld32s = helper_ldsl_mmu;
    tier2_helpers.ld64 = helper_ldq_mmu;
    tier2_helpers.st8 = helper_stb_mmu;
    tier2_helpers.st16 = helper_stw_mmu;
    tier2_helpers.st32 = helper_stl_mmu;
    tier2_helpers.st64 = helper_stq_mmu;
}

uint64_t tier2_snap_gen_current(void)
{
    return qatomic_read(&tier2_snap_gen);
}

/*
 * Phase 5: Executable Chain Stub Pool for direct goto_tb linking.
 * Adapts TCG's in-register calling convention (env in x19, TCG stack active)
 * to Tier-2's C AAPCS64 convention (env in x0), returning to tcg_tb_ret_addr.
 */
#define TIER2_STUB_SIZE 64
static uint8_t *tier2_stub_pool = NULL;
static size_t tier2_stub_pool_used = 0;
static size_t tier2_stub_pool_cap = 0;
static QemuMutex tier2_stub_lock;
static bool tier2_stub_lock_inited = false;

void *tier2_create_chain_stub(TranslationBlock *tb, void *native_code)
{
#if defined(__aarch64__)
    if (!native_code || !tcg_tb_ret_addr) {
        return NULL;
    }
    if (!tier2_stub_lock_inited) {
        qemu_mutex_init(&tier2_stub_lock);
        tier2_stub_lock_inited = true;
    }
    qemu_mutex_lock(&tier2_stub_lock);
    if (!tier2_stub_pool || tier2_stub_pool_used + TIER2_STUB_SIZE > tier2_stub_pool_cap) {
        size_t alloc_sz = 65536;
        int flags = MAP_PRIVATE | MAP_ANON;
#if defined(MAP_JIT)
        flags |= MAP_JIT;
#endif
        void *p = mmap(NULL, alloc_sz, PROT_READ | PROT_WRITE | PROT_EXEC, flags, -1, 0);
        if (p == MAP_FAILED) {
            qemu_mutex_unlock(&tier2_stub_lock);
            return NULL;
        }
        tier2_stub_pool = (uint8_t *)p;
        tier2_stub_pool_used = 0;
        tier2_stub_pool_cap = alloc_sz;
    }
    uint32_t *stub = (uint32_t *)(tier2_stub_pool + tier2_stub_pool_used);
    tier2_stub_pool_used += TIER2_STUB_SIZE;
    qemu_mutex_unlock(&tier2_stub_lock);

    qemu_thread_jit_write();
    /*
     * 0:  mov  x0, x19
     * 4:  ldr  x16, [pc, #20] ; load native_code at offset 24
     * 8:  blr  x16            ; call trace, returns exit_code in x0
     * 12: ldr  x16, [pc, #20] ; load tcg_tb_ret_addr at offset 32
     * 16: br   x16            ; return to TCG epilogue
     * 20: nop                 ; 8-byte alignment padding
     * 24: .quad native_code
     * 32: .quad tcg_tb_ret_addr
     */
    stub[0] = 0xaa1303e0;
    stub[1] = 0x580000b0;
    stub[2] = 0xd63f0200;
    stub[3] = 0x580000b0;
    stub[4] = 0xd61f0200;
    stub[5] = 0xd503201f;
    *(uint64_t *)(stub + 6) = (uint64_t)(uintptr_t)native_code;
    *(uint64_t *)(stub + 8) = (uint64_t)(uintptr_t)tcg_tb_ret_addr;

    flush_idcache_range((uintptr_t)stub, (uintptr_t)stub, TIER2_STUB_SIZE);
    /* Caller maintains write permissions for setting hdr->tier2_stub and re-linking */
    return stub;
#else
    return NULL;
#endif
}

void tier2_free_chain_stub(TranslationBlock *tb)
{
    tb->tier2_stub = NULL;
}

/*
 * Retire one installed header. Must hold snap_lock. Clears code (so no
 * new dispatches enter), detaches the exit-protocol record for RCU
 * reclamation (in-flight side exits may still read it -- they run under
 * cpu_exec's RCU read lock), and retires the ORC module (freed once no
 * vCPU can be inside it). hdr is the RW hash/dispatch view.
 */
static void tier2_retire_locked(TranslationBlock *hdr, void *fn)
{
    if (hdr->tier2_stub) {
        TranslationBlock *pred;
        int slot;
        qemu_thread_jit_write();
        qemu_spin_lock(&hdr->jmp_lock);
        TB_FOR_EACH_JMP(hdr, pred, slot) {
            uintptr_t dest = qatomic_read(&pred->jmp_dest[slot]);
            if ((dest & 1) == 0 && (TranslationBlock *)dest == hdr) {
                tb_set_jmp_target(pred, slot, (uintptr_t)hdr->tc.ptr);
            }
        }
        qemu_spin_unlock(&hdr->jmp_lock);
        hdr->tier2_stub = NULL;
    }
    if (tier2_jit_invalidate_fn) {
        tier2_jit_invalidate_fn(fn);
    }
    hdr->tier2_code = NULL;
    hdr->tier2_enqueued = false;
    Tier2Installed *rec = qatomic_xchg(&hdr->tier2_rec, NULL);
    if (rec) {
        g_free_rcu(rec, rcu);
    }
}

/*
 * Publish fresh code on hdr (snap_lock held, gen already validated).
 * Installs the exit-protocol record (for op-walker compiles; NULL for
 * legacy, which keeps legacy return encoding), retiring whatever was
 * there before so recompiles never leak modules or records.
 */
static void tier2_publish_locked(TranslationBlock *hdr, void *fn,
                                 Tier2Installed *rec)
{
    if (tier2_installed) {
        void *old = g_hash_table_lookup(tier2_installed, hdr);
        if (old && old != fn && tier2_jit_invalidate_fn) {
            tier2_jit_invalidate_fn(old);
        }
        g_hash_table_replace(tier2_installed, hdr, fn);
    }
    Tier2Installed *prev = qatomic_xchg(&hdr->tier2_rec, rec);
    if (prev) {
        g_free_rcu(prev, rcu);
    }
    hdr->tier2_code = fn;
    hdr->tier2_stub = tier2_create_chain_stub(hdr, fn);

    /* Phase 5 Chain-Graph Dynamic Re-linking:
     * Repoint existing predecessor goto_tb jump slots directly into the
     * Tier-2 native chain stub so chained TCG loops execute natively.
     */
    if (hdr->tier2_stub) {
        TranslationBlock *pred;
        int slot;
        qemu_thread_jit_write();
        qemu_spin_lock(&hdr->jmp_lock);
        TB_FOR_EACH_JMP(hdr, pred, slot) {
            uintptr_t dest = qatomic_read(&pred->jmp_dest[slot]);
            if ((dest & 1) == 0 && (TranslationBlock *)dest == hdr) {
                tb_set_jmp_target(pred, slot, (uintptr_t)hdr->tier2_stub);
            }
        }
        qemu_spin_unlock(&hdr->jmp_lock);
    }
}

/* Drop Tier-2 compiled code when a TB is invalidated. Must be called with
 * the TB's pages locked (do_tb_phys_invalidate context) or under
 * qemu_thread_jit_write(). */
void tier2_invalidate(TranslationBlock *tb)
{
    if (!tb) {
        return;
    }

    qemu_thread_jit_write();
    qemu_mutex_lock(&tier2_snap_lock);

    /*
     * Targeted invalidation: only remove the snapshot for @tb, and retire
     * installed traces that actually contain @tb (as header or trace member).
     * Do NOT drop all snapshots or bump tier2_snap_gen across the whole VM;
     * doing so thrashes compilation during OS boots and guest code churn.
     */
    if (tier2_snaps) {
        g_hash_table_remove(tier2_snaps, tb);
        tier2_snap_count = (uint32_t)g_hash_table_size(tier2_snaps);
    }
    if (tier2_installed && g_hash_table_size(tier2_installed) > 0) {
        void *rx_tb = (void *)tcg_splitwx_to_rx(tb);
        GHashTableIter it;
        gpointer k, v;
        g_hash_table_iter_init(&it, tier2_installed);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            TranslationBlock *hdr = (TranslationBlock *)k;
            bool matches = (hdr == tb);
            if (!matches && hdr->tier2_rec) {
                Tier2Installed *rec = hdr->tier2_rec;
                for (uint32_t i = 0; i < rec->num_members; i++) {
                    if (rec->rx[i] == rx_tb) {
                        matches = true;
                        break;
                    }
                }
            }
            if (matches) {
                tier2_retire_locked(hdr, v);
                g_hash_table_iter_remove(&it);
            }
        }
    }
    qemu_mutex_unlock(&tier2_snap_lock);

    /* tb is the RW view (invalidation paths); write it directly. */
    tb->tier2_code = NULL;
    tb->tier2_enqueued = false;
    qemu_thread_jit_execute();
}

/* Drop everything. Caller must hold no vCPU (tb_flush exclusive context)
 * or be at shutdown / breakpoint change (rare paths). */
void tier2_invalidate_all(void)
{
    qemu_thread_jit_write();

    qemu_mutex_lock(&tier2_snap_lock);
    if (tier2_installed) {
        GHashTableIter it;
        gpointer k, v;
        g_hash_table_iter_init(&it, tier2_installed);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            TranslationBlock *hdr = (TranslationBlock *)k;
            tier2_retire_locked(hdr, v);
            g_hash_table_iter_remove(&it);
        }
    }
    if (tier2_snaps) {
        g_hash_table_remove_all(tier2_snaps);
        tier2_snap_count = 0;
    }
    tier2_snap_gen++;
    qemu_mutex_unlock(&tier2_snap_lock);

    /*
     * ORC resources are NOT released here: vCPUs may still be inside
     * retired code (it stays mapped until reclaim). Retired trackers are
     * released by tier2_reclaim() (tb_flush exclusive context) and at
     * shutdown. Dispatch can no longer reach retired code because every
     * tier2_code pointer was cleared above.
     */
    qemu_thread_jit_execute();
}

/*
 * Actually release retired ORC modules. Exclusive context only (tb_flush:
 * no vCPU running anywhere, so no one can be inside retired code).
 */
void tier2_reclaim(void)
{
    if (tier2_jit_flush_fn) {
        tier2_jit_flush_fn();
    }
}

/* Compile the trace into an optimized LLVM function */
static void tier2_compile_trace(const Tier2Trace *trace)
{
    uint32_t total_insns = 0;
    size_t total_bytes = 0;

    for (uint32_t i = 0; i < trace->num_tbs; i++) {
        TranslationBlock *tb = trace->tbs[i];
        if (tb) {
            total_insns += tb->icount;
            total_bytes += tb->size;
        }
    }

    if (tier2_verbose || qemu_loglevel_mask(CPU_LOG_EXEC)) {
        qemu_log("[tier2-compiler] Compiling loop trace: header pc=0x%" VADDR_PRIx
                 ", %u TBs, %u total insns, %zu guest bytes\n",
                 tb_guest_pc(trace->header), trace->num_tbs, total_insns, total_bytes);
        for (uint32_t i = 0; i < trace->num_tbs; i++) {
            TranslationBlock *tb = trace->tbs[i];
            qemu_log("  TB[%u]: pc=0x%" VADDR_PRIx ", icount=%u, size=%u, tc=%p\n",
                     i, tb_guest_pc(tb), tb->icount, tb->size, tb->tc.ptr);
        }
    }

    /*
     * Loop trace compilation pipeline:
     * When LLVM JIT backend is active, compile the trace into native code
     * and install the pointer in the header TB's tier2_code field.
     */
    if (!tier2_jit_compile_fn) {
        return;
    }
    if (trace->num_tbs == 0 || trace->num_tbs > TIER2_MAX_TRACE_TBS) {
        return;
    }
    /*
     * Skip recompiling a header that already has code installed (re-armed
     * hot TBs re-enqueue; the installed trace is still valid until an
     * invalidation clears it).
     */
    if (trace->header && trace->header->tier2_code != NULL) {
        return;
    }

    Tier2TraceDesc *desc = g_new0(Tier2TraceDesc, 1);
    desc->num_tbs = trace->num_tbs;
    desc->total_exec_count = trace->total_exec_count;
    /* Header index: strategy-1 lists header-first, strategy-2 header-last. */
    desc->header_idx = 0;
    for (uint32_t i = 0; i < trace->num_tbs; i++) {
        if (trace->tbs[i] == trace->header) {
            desc->header_idx = i;
            break;
        }
    }
    for (uint32_t i = 0; i < trace->num_tbs; i++) {
        int32_t nx = trace->next[i];
        /* Clamp stale indices (walk aborted mid-list): unproven = side exit. */
        if (nx < 0 || (uint32_t)nx >= trace->num_tbs) {
            nx = -1;
        }
        desc->next[i] = nx;
        int32_t sl = trace->next_slot[i];
        desc->next_slot[i] = (nx >= 0 && (sl == 0 || sl == 1)) ? sl : -1;
    }

    /* New path: attach op snapshots without dereferencing TBs (the hash
     * lookup uses the pointer only as a key). */
    bool have_all = true;
    uint64_t gen;
    qemu_mutex_lock(&tier2_snap_lock);
    gen = tier2_snap_gen;
    for (uint32_t i = 0; i < trace->num_tbs; i++) {
        Tier2TBRec *snap = tier2_snaps ?
            g_hash_table_lookup(tier2_snaps, trace->tbs[i]) : NULL;
        if (!snap || snap->num_ops == 0) {
            have_all = false;
            if (tier2_verbose) {
                qemu_log("[tier2-compiler] TB[%u] has no op snapshot%s\n", i,
                         snap ? " (empty)" : "");
            }
            break;
        }
        if (snap->cflags & CF_USE_ICOUNT) {
            /*
             * Fused execution skips per-TB icount decrementing; running
             * it would break icount expiry timing (and record/replay,
             * which mandates icount). Stay on TCG.
             */
            have_all = false;
            if (tier2_verbose) {
                qemu_log("[tier2-compiler] TB[%u] uses icount, skipping\n", i);
            }
            break;
        }
        desc->recs[i] = *snap; /* bounded struct copy */
    }
    if (have_all) {
        tier2_resolve_helpers();
        desc->has_ops = true;
        desc->guest_mem_allowed = tier2_guest_mem;
        desc->has_safepoint = tier2_sp_measured;
        desc->cpu_off = tier2_sp_cpu_off;
        desc->irq_off = (int64_t)offsetof(CPUState, interrupt_request);
        desc->last_tb_off = tier2_sp_last_tb_off;
#ifndef CONFIG_USER_ONLY
        desc->ram_base = 0;
        desc->ram_size = 0;
#else
        desc->ram_base = (uint64_t)(uintptr_t)guest_base;
        desc->ram_size = ~0ULL;
#endif
        desc->trace_id = 0;
        desc->workload_pc = tier2_workload_pc;
        tier2_fill_tlb_layout(&desc->tlb);
        /*
         * Call-name table for code-cache symbol resolution, in emission
         * order. Any unresolvable target (or overflow) disables caching
         * for the trace -- it still compiles normally.
         */
        desc->no_cache = false;
        desc->num_calls = 0;
        snprintf(desc->guest_arch, sizeof(desc->guest_arch), "%s",
                 target_name());
        snprintf(desc->qemu_version, sizeof(desc->qemu_version), "%s",
                 QEMU_VERSION);
        for (uint32_t t = 0; t < trace->num_tbs && !desc->no_cache; t++) {
            const Tier2TBRec *r = &desc->recs[t];
            for (uint32_t i = 0; i < r->num_ops; i++) {
                if (r->ops[i].op != T2_CALL) {
                    continue;
                }
                if (desc->num_calls >= TIER2_MAX_CALLS) {
                    desc->no_cache = true;
                    break;
                }
                void *fn = (void *)(uintptr_t)r->ops[i].imm1;
                const char *nm = tier2_helper_names ?
                    g_hash_table_lookup(tier2_helper_names, fn) : NULL;
                if (!nm) {
                    desc->no_cache = true;
                    break;
                }
                snprintf(desc->call_names[desc->num_calls],
                         sizeof(desc->call_names[0]), "%s", nm);
                desc->call_addrs[desc->num_calls] = (uint64_t)fn;
                desc->num_calls++;
            }
        }
#ifndef CONFIG_USER_ONLY
        desc->lookup_helper = helper_lookup_tb_ptr;
#else
        desc->lookup_helper = NULL;
#endif
        uint32_t h = desc->header_idx;
        desc->rx_header = desc->recs[h].rx_tb;
        desc->mem_helpers = tier2_helpers;
        /*
         * Header identity without dereferencing TBs. The workload-hack
         * match and its exit value key off these.
         */
        desc->header_pc = desc->recs[h].pc;
        /* Legacy fields stay populated: the JIT's workload hack needs
         * pc (match), tb_obj (exit value) and code_ptr. */
        desc->tbs[0].pc = desc->recs[h].pc;
        desc->tbs[0].tb_obj = (void *)desc->recs[h].rx_tb;
        desc->tbs[0].code_ptr = desc->recs[h].code_ptr;
    }
    qemu_mutex_unlock(&tier2_snap_lock);

    if (!have_all) {
        /* Legacy fallback descriptor (dereferences TBs, as before). */
        for (uint32_t i = 0; i < trace->num_tbs; i++) {
            desc->tbs[i].pc = tb_guest_pc(trace->tbs[i]);
            desc->tbs[i].size = trace->tbs[i]->size;
            desc->tbs[i].icount = trace->tbs[i]->icount;
            desc->tbs[i].code_ptr = trace->tbs[i]->tc.ptr;
            desc->tbs[i].tb_obj = trace->tbs[i];
        }
        desc->workload_pc = tier2_workload_pc;
        desc->header_pc = tb_guest_pc(trace->header);
        qemu_mutex_lock(&tier2_snap_lock);
        tier2_snap_hot_miss++;
        qemu_mutex_unlock(&tier2_snap_lock);
    }

    void *native_code = tier2_jit_compile_fn(desc);
    g_free(desc);
    if (!native_code) {
        return;
    }
    if (!have_all) {
        /*
         * Legacy path: nowadays only the workload hack emits code here
         * (plain trampolines are gone). It keeps legacy return encoding,
         * so no exit record applies -- but clear any stale one, so a
         * leftover record can never be resolved against this code.
         */
        qemu_thread_jit_write();
        bool hdr_valid = !(tb_cflags(trace->header) & CF_INVALID);
        if (hdr_valid) {
            trace->header->tier2_code = native_code;
        } else {
            if (tier2_jit_invalidate_fn) {
                tier2_jit_invalidate_fn(native_code);
            }
            qemu_thread_jit_execute();
            return;
        }
        qemu_thread_jit_execute();
        qemu_mutex_lock(&tier2_snap_lock);
        if (tier2_installed) {
            void *old = g_hash_table_lookup(tier2_installed, trace->header);
            if (old && old != native_code && tier2_jit_invalidate_fn) {
                tier2_jit_invalidate_fn(old);
            }
            g_hash_table_replace(tier2_installed, trace->header, native_code);
        }
        Tier2Installed *stale = qatomic_xchg(&trace->header->tier2_rec, NULL);
        if (stale) {
            g_free_rcu(stale, rcu);
        }
        qemu_mutex_unlock(&tier2_snap_lock);
        if (tier2_verbose || qemu_loglevel_mask(CPU_LOG_EXEC)) {
            qemu_log("[tier2] Installed legacy Tier-2 code %p for header 0x%" VADDR_PRIx "\n",
                     native_code, tb_guest_pc(trace->header));
        }
        return;
    }

    /*
     * Op-walker install: validate nothing changed while we compiled.
     * tier2_invalidate() clears the snapshot table and bumps snap_gen,
     * so a matching generation means every member snapshot below is
     * still current. On any doubt, retire the fresh code and keep
     * running TCG. (TB structs are RCU-freed only at flush, which
     * always bumps the generation.)
     */
    qemu_thread_jit_write();
    qemu_mutex_lock(&tier2_snap_lock);
    bool gen_ok = (gen == tier2_snap_gen);
    TranslationBlock *hdr = trace->header;
    Tier2TBRec *cur = gen_ok && tier2_snaps ?
        g_hash_table_lookup(tier2_snaps, hdr) : NULL;
    bool valid = cur && !(tb_cflags(hdr) & CF_INVALID);
    Tier2Installed *rec = NULL;
    if (valid) {
        /*
         * Edge re-validation: strategy-1 provenance (trace->next[]) was
         * observed at discovery, potentially a full compile (~100ms)
         * before this install -- longer now that the async sampler also
         * discovers traces far from any dispatch. Confirm every proven
         * edge still links the same way; a re-chained predecessor would
         * otherwise fuse a stale internal branch. Unproven slots
         * (negative, out of range, or bad slot -- matching the walker's
         * own clamping) are side exits and need no confirmation. A link
         * change mid-check can only discard a valid compile (safe
         * direction: re-enqueue happens on next hot crossing).
         */
        for (uint32_t i = 0; i < trace->num_tbs && valid; i++) {
            int32_t nx = trace->next[i];
            int32_t sl = trace->next_slot[i];
            if (nx < 0 || (uint32_t)nx >= trace->num_tbs ||
                (sl != 0 && sl != 1)) {
                continue;
            }
            TranslationBlock *src = trace->tbs[i];
            TranslationBlock *want = trace->tbs[nx];
            uintptr_t dest = src->jmp_dest[sl] & ~3UL;
            if ((TranslationBlock *)dest != want ||
                (tb_cflags(src) & CF_INVALID) ||
                (tb_cflags(want) & CF_INVALID)) {
                valid = false;
                if (tier2_verbose) {
                    qemu_log("[tier2] Discarding compile: edge TB[%u]->TB[%d] "
                             "changed during compile\n", i, nx);
                }
            }
        }
    }
    if (valid) {
        /*
         * Build the exit-protocol record from CURRENT rx values (fresh
         * re-lookup, not the compile-time copies): side exits resolve
         * through these, so they must postdate every invalidation.
         */
        rec = g_new0(Tier2Installed, 1);
        rec->gen = gen;
        rec->num_members = trace->num_tbs;
        for (uint32_t i = 0; i < trace->num_tbs; i++) {
            Tier2TBRec *snap = g_hash_table_lookup(tier2_snaps,
                                                   trace->tbs[i]);
            if (!snap) {
                valid = false;
                break;
            }
            rec->rx[i] = (void *)snap->rx_tb;
        }
        if (valid) {
            tier2_publish_locked(hdr, native_code, rec);
            rec = NULL; /* owned by the TB now */
        }
        if (rec) {
            g_free(rec);
        }
        if (tier2_verbose || qemu_loglevel_mask(CPU_LOG_EXEC)) {
            qemu_log("[tier2] Installed op-walker Tier-2 code %p for header "
                     "pc=0x%" PRIx64 " (%u ops)\n",
                     native_code, cur->pc, cur->num_ops);
        }
    }
    if (!valid) {
        if (tier2_jit_invalidate_fn) {
            tier2_jit_invalidate_fn(native_code);
        }
        if (tier2_verbose) {
            qemu_log("[tier2] Discarded freshly compiled code %p "
                     "(header changed during compile)\n", native_code);
        }
    }
    qemu_mutex_unlock(&tier2_snap_lock);
    qemu_thread_jit_execute();
}

/* Background compiler worker thread */
static void *tier2_worker_thread(void *arg)
{
    rcu_register_thread();
    while (!tier2_stopping) {
        qemu_mutex_lock(&tier2_queue_lock);
        while (!tier2_work_queue && !tier2_stopping) {
            qemu_cond_wait(&tier2_cond, &tier2_queue_lock);
        }

        if (tier2_stopping) {
            qemu_mutex_unlock(&tier2_queue_lock);
            break;
        }

        Tier2WorkItem *item = tier2_work_queue;
        tier2_work_queue = item->next;
        qemu_mutex_unlock(&tier2_queue_lock);

        tier2_compile_trace(&item->trace);
        g_free(item);
    }
    return NULL;
}

void tier2_init(void)
{
    if (tier2_initialized) {
        return;
    }

    const char *env_debug = getenv("QEMU_TIER2_DEBUG");
    if (env_debug && strcmp(env_debug, "1") == 0) {
        tier2_verbose = true;
    }

    /*
     * Infrastructure first: mutexes/tables must exist before any TB or
     * flush path can reach tier2_invalidate(), even when compilation is
     * disabled below. (Disabling used to return before this point and
     * abort on the first locked mutex.)
     */
    qemu_mutex_init(&tier2_queue_lock);
    qemu_cond_init(&tier2_cond);
    qemu_mutex_init(&tier2_snap_lock);
    qemu_mutex_init(&tier2_stub_lock);
    tier2_stub_lock_inited = true;
    tier2_snaps = g_hash_table_new_full(NULL, NULL, NULL, g_free);
    tier2_installed = g_hash_table_new(NULL, NULL);
    tier2_helper_names = g_hash_table_new(NULL, NULL);
    tier2_snap_gen = 0;
    tier2_snap_count = 0;
    tier2_work_queue = NULL;
    tier2_stopping = false;
    tier2_initialized = true;

    const char *env_disable = getenv("QEMU_TIER2_DISABLE");
    if (env_disable && strcmp(env_disable, "1") == 0) {
        tier2_enabled = false;
        return;
    }

    const char *env_gmem = getenv("QEMU_TIER2_GUEST_MEM");
    if (env_gmem && strcmp(env_gmem, "1") == 0) {
        tier2_guest_mem = true;
    }

    const char *env_wpc = getenv("QEMU_TIER2_WORKLOAD_PC");
    if (env_wpc) {
        unsigned long long v = strtoull(env_wpc, NULL, 0);
        tier2_workload_pc = (uint64_t)v;
    }

    const char *env_count = getenv("QEMU_TIER2_COUNT");
    if (env_count && strcmp(env_count, "1") == 0) {
        tier2_counting = true;
        qemu_add_exit_notifier(&tier2_exit_notifier);
    }

    const char *env_snapmax = getenv("QEMU_TIER2_SNAP_MAX");
    if (env_snapmax) {
        unsigned long v = strtoul(env_snapmax, NULL, 0);
        if (v > 0 && v <= 1000000) {
            tier2_snap_max = (uint32_t)v;
        }
    }

#ifndef _WIN32
#if defined(__APPLE__)
    /*
     * On Darwin/Apple Silicon, asynchronous SIGPROF delivery to threads
     * running in MAP_JIT code violates APRR / libsystem stack checks,
     * causing __stack_chk_fail and SIGSEGV. Default to off on Darwin;
     * allow opt-in via QEMU_TIER2_PROF=1.
     */
    tier2_prof_disabled = true;
    const char *env_prof = getenv("QEMU_TIER2_PROF");
    if (env_prof && strcmp(env_prof, "1") == 0) {
        tier2_prof_disabled = false;
    }
#else
    const char *env_prof = getenv("QEMU_TIER2_PROF");
    if (env_prof && strcmp(env_prof, "0") == 0) {
        tier2_prof_disabled = true;
    }
#endif
    const char *env_hz = getenv("QEMU_TIER2_PROF_HZ");
    if (env_hz) {
        unsigned long v = strtoul(env_hz, NULL, 0);
        if (v >= 50 && v <= 4000) {
            tier2_prof_hz = (unsigned)v;
        }
    }
#endif

    /* Look for compiled LLVM Tier-2 JIT engine (built-in or dynamic plugin) */
    tier2_jit_init_fn = (Tier2JitInitFn)dlsym(RTLD_DEFAULT, "tier2_jit_init");
    tier2_jit_compile_fn = (Tier2JitCompileFn)dlsym(RTLD_DEFAULT, "tier2_jit_compile_trace");
    tier2_jit_shutdown_fn = (Tier2JitShutdownFn)dlsym(RTLD_DEFAULT, "tier2_jit_shutdown");
    tier2_jit_invalidate_fn = (Tier2JitInvalidateFn)dlsym(RTLD_DEFAULT, "tier2_jit_invalidate");
    tier2_jit_flush_fn = (Tier2JitFlushFn)dlsym(RTLD_DEFAULT, "tier2_jit_flush");

    if (!tier2_jit_compile_fn) {
        const char *dylib_paths[] = {
            "./build/libqemu-tier2.dylib",
            "build/libqemu-tier2.dylib",
            "./libqemu-tier2.dylib",
            "libqemu-tier2.dylib",
            "./build/libqemu-tier2.so",
            "libqemu-tier2.so",
            NULL
        };
        for (int i = 0; dylib_paths[i]; i++) {
            tier2_dylib_handle = dlopen(dylib_paths[i], RTLD_NOW | RTLD_GLOBAL);
            if (tier2_dylib_handle) {
                tier2_jit_init_fn = (Tier2JitInitFn)dlsym(tier2_dylib_handle, "tier2_jit_init");
                tier2_jit_compile_fn = (Tier2JitCompileFn)dlsym(tier2_dylib_handle, "tier2_jit_compile_trace");
                tier2_jit_shutdown_fn = (Tier2JitShutdownFn)dlsym(tier2_dylib_handle, "tier2_jit_shutdown");
                tier2_jit_invalidate_fn = (Tier2JitInvalidateFn)dlsym(tier2_dylib_handle, "tier2_jit_invalidate");
                tier2_jit_flush_fn = (Tier2JitFlushFn)dlsym(tier2_dylib_handle, "tier2_jit_flush");
                break;
            }
        }
    }

    if (tier2_jit_init_fn) {
        if (tier2_jit_init_fn((void *)tcg_qemu_tb_exec)) {
            if (tier2_verbose) {
                qemu_log("[tier2] LLVM ORC JIT backend initialized and ready\n");
            }
        }
    }

    qemu_thread_create(&tier2_thread, "tier2-compiler",
                       tier2_worker_thread, NULL, QEMU_THREAD_JOINABLE);
    tier2_thread_started = true;

#ifndef _WIN32
    /*
     * Async sample profiler: SIGPROF each vCPU at tier2_prof_hz, aggregate
     * in the sampler thread. Default on (coexists with inline dispatch
     * sampling for now; removal of the inline counters is a follow-up
     * once this is proven). Kill switch: QEMU_TIER2_PROF=0. The handler
     * is a static function (process lifetime), so it is simply left
     * installed at shutdown.
     */
    if (!tier2_prof_disabled && tier2_enabled) {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = tier2_prof_handler;
        sigemptyset(&sa.sa_mask);
        sigaddset(&sa.sa_mask, SIGPROF);
        sa.sa_flags = SA_RESTART;
        if (sigaction(SIGPROF, &sa, NULL) == 0) {
            qemu_thread_create(&tier2_prof_thread, "tier2-profiler",
                               tier2_prof_worker, NULL, QEMU_THREAD_JOINABLE);
            tier2_prof_started = true;
            if (tier2_verbose) {
                qemu_log("[tier2] profiler sampler started (%u Hz)\n",
                         tier2_prof_hz);
            }
        } else if (tier2_verbose) {
            qemu_log("[tier2] profiler disabled: sigaction failed\n");
        }
    }
#endif
}

void tier2_shutdown(void)
{
    if (!tier2_initialized) {
        return;
    }

#ifndef _WIN32
    if (tier2_prof_started) {
        tier2_prof_stopping = true;
        qemu_thread_join(&tier2_prof_thread);
        tier2_prof_started = false;
        if (tier2_verbose) {
            qemu_log("[tier2-prof] hits=%lu drained=%lu requests=%lu\n",
                     tier2_prof_hits, tier2_prof_drained,
                     tier2_prof_requests_made);
        }
    }
#endif

    if (tier2_thread_started) {
        qemu_mutex_lock(&tier2_queue_lock);
        tier2_stopping = true;
        qemu_cond_signal(&tier2_cond);
        qemu_mutex_unlock(&tier2_queue_lock);

        qemu_thread_join(&tier2_thread);
        tier2_thread_started = false;
    }

    if (tier2_jit_shutdown_fn) {
        tier2_jit_shutdown_fn();
    }
    if (tier2_dylib_handle) {
        dlclose(tier2_dylib_handle);
        tier2_dylib_handle = NULL;
    }

    /* Drain any remaining queue items */
    qemu_mutex_lock(&tier2_queue_lock);
    Tier2WorkItem *cur = tier2_work_queue;
    while (cur) {
        Tier2WorkItem *next = cur->next;
        g_free(cur);
        cur = next;
    }
    tier2_work_queue = NULL;
    qemu_mutex_unlock(&tier2_queue_lock);

    qemu_mutex_destroy(&tier2_queue_lock);
    qemu_cond_destroy(&tier2_cond);

    qemu_mutex_lock(&tier2_snap_lock);
    if (tier2_snaps) {
        g_hash_table_destroy(tier2_snaps);
        tier2_snaps = NULL;
    }
    if (tier2_installed) {
        g_hash_table_destroy(tier2_installed);
        tier2_installed = NULL;
    }
    if (tier2_helper_names) {
        g_hash_table_destroy(tier2_helper_names);
        tier2_helper_names = NULL;
    }
    tier2_snap_count = 0;
    qemu_mutex_unlock(&tier2_snap_lock);
    qemu_mutex_destroy(&tier2_snap_lock);

    if (tier2_verbose && tier2_snap_hot_miss) {
        qemu_log("[tier2] hot traces without snapshots: %" PRIu64 "\n",
                 tier2_snap_hot_miss);
    }

    tier2_initialized = false;
}
