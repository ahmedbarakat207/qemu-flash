/* LLVM tier-2 runtime interface & background compiler thread.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/lockable.h"
#include "qemu/log.h"
#include "exec/cpu-common.h"
#include "exec/memop.h"
#include "exec/memopidx.h"
#include "tcg/tcg-cond.h"
#include <dlfcn.h>
#include "tcg/llvm/tier2.h"
#include "tcg/llvm/tier2-jit.h"

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
static uint64_t tier2_snap_gen;
static uint32_t tier2_snap_count;
static uint32_t tier2_snap_max = 16384;
static uint64_t tier2_snap_hot_miss;

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
bool tier2_find_loop_trace(TranslationBlock *header, Tier2Trace *out_trace)
{
    if (!header) {
        return false;
    }

    out_trace->header = header;
    out_trace->num_tbs = 0;
    out_trace->total_exec_count = header->exec_count;

    /*
     * Strategy 1: Forward walk of QEMU's block chaining graph (jmp_dest).
     * Follows direct jumps from header -> TB2 -> ... -> header.
     */
    TranslationBlock *cur = header;
    for (int step = 0; step < TIER2_MAX_TRACE_TBS; step++) {
        out_trace->tbs[out_trace->num_tbs++] = cur;

        TranslationBlock *next = NULL;
        for (int i = 0; i < 2; i++) {
            uintptr_t dest_ptr = cur->jmp_dest[i];
            TranslationBlock *dest = (TranslationBlock *)(dest_ptr & ~3UL);
            if (dest) {
                if (dest == header) {
                    /* Closed loop cycle detected! */
                    return true;
                }
                if (!next && !(dest->cflags & CF_INVALID)) {
                    next = dest;
                }
            }
        }
        if (!next || next == cur) {
            break;
        }
        cur = next;
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
        int64_t packed = (int64_t)size | ((memop & MO_SIGN) ? 0x100 : 0) |
                         ((memop & MO_BSWAP) ? 0x10000 : 0) |
                         ((int64_t)get_mmuidx(oi) << 24);
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
        /* call, qemu_ld2/st2, goto_ptr, exit_req, plugin_*, mb, div/rem,
         * carry ops, vector ops, ...: trace boundary. Stop, don't guess. */
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
            break; /* unsupported op ends the supported prefix */
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

static void tier2_resolve_helpers(void)
{
    if (tier2_helpers_resolved) {
        return;
    }
    tier2_helpers_resolved = true;
    tier2_helpers.ld8u = dlsym(RTLD_DEFAULT, "helper_ldub_mmu");
    tier2_helpers.ld8s = dlsym(RTLD_DEFAULT, "helper_ldsb_mmu");
    tier2_helpers.ld16u = dlsym(RTLD_DEFAULT, "helper_lduw_mmu");
    tier2_helpers.ld16s = dlsym(RTLD_DEFAULT, "helper_ldsw_mmu");
    tier2_helpers.ld32u = dlsym(RTLD_DEFAULT, "helper_ldul_mmu");
    tier2_helpers.ld32s = dlsym(RTLD_DEFAULT, "helper_ldsl_mmu");
    tier2_helpers.ld64 = dlsym(RTLD_DEFAULT, "helper_ldq_mmu");
    tier2_helpers.st8 = dlsym(RTLD_DEFAULT, "helper_stb_mmu");
    tier2_helpers.st16 = dlsym(RTLD_DEFAULT, "helper_stw_mmu");
    tier2_helpers.st32 = dlsym(RTLD_DEFAULT, "helper_stl_mmu");
    tier2_helpers.st64 = dlsym(RTLD_DEFAULT, "helper_stq_mmu");
}

/* Drop Tier-2 compiled code when a TB is invalidated. Must be called with
 * the TB's pages locked (do_tb_phys_invalidate context) or under
 * qemu_thread_jit_write(). */
void tier2_invalidate(TranslationBlock *tb)
{
    if (!tb) {
        return;
    }

    qemu_mutex_lock(&tier2_snap_lock);
    /*
     * Conservative v1: any invalidation drops ALL snapshots and ALL
     * installed traces, not just ones rooted at @tb. A compiled loop body
     * spanning several TBs goes stale when ANY member TB dies, and we do
     * not track exact trace membership yet. This is cheap: snapshots are
     * rebuilt automatically at the next translation, exec_count is
     * retained so re-compilation re-triggers quickly, and correctness
     * (never execute stale code) dominates. tier2_snap_gen is bumped so
     * a background compile in flight discards its result at install.
     */
    if (tier2_snaps) {
        g_hash_table_remove_all(tier2_snaps);
        tier2_snap_count = 0;
    }
    if (tier2_installed && g_hash_table_size(tier2_installed) > 0) {
        GHashTableIter it;
        gpointer k, v;
        g_hash_table_iter_init(&it, tier2_installed);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            TranslationBlock *hdr = (TranslationBlock *)k;
            if (tier2_jit_invalidate_fn) {
                tier2_jit_invalidate_fn(v);
            }
            /* hdr is the RW hash/dispatch view; write it directly. */
            hdr->tier2_code = NULL;
            hdr->tier2_enqueued = false;
            g_hash_table_iter_remove(&it);
        }
    }
    tier2_snap_gen++;
    qemu_mutex_unlock(&tier2_snap_lock);

    /* tb is the RW view (invalidation paths); write it directly. */
    tb->tier2_code = NULL;
    tb->tier2_enqueued = false;
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
            if (tier2_jit_invalidate_fn) {
                tier2_jit_invalidate_fn(v);
            }
            /* hdr is the RW hash/dispatch view; write it directly. */
            hdr->tier2_code = NULL;
            hdr->tier2_enqueued = false;
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
        desc->recs[i] = *snap; /* bounded struct copy */
    }
    if (have_all) {
        tier2_resolve_helpers();
        desc->has_ops = true;
        desc->guest_mem_allowed = tier2_guest_mem;
        desc->trace_id = 0;
        desc->workload_pc = tier2_workload_pc;
        desc->rx_header = desc->recs[0].rx_tb;
        desc->mem_helpers = tier2_helpers;
        /*
         * Header identity without dereferencing TBs: history-derived
         * traces list oldest-first, so the header (hot TB) is last.
         * The workload-hack match and its exit value key off these.
         */
        uint32_t h = desc->num_tbs - 1;
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
        /* Legacy trampoline: install exactly like before, but refuse when
         * the header died while we compiled. */
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
            g_hash_table_replace(tier2_installed, trace->header, native_code);
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
     * so a lookup hit + matching generation + !CF_INVALID means the
     * header TB struct is still ours. On any doubt, retire the fresh
     * code and keep running TCG. (TB structs are RCU-freed only at
     * flush, which always bumps the generation.)
     */
    qemu_thread_jit_write();
    qemu_mutex_lock(&tier2_snap_lock);
    bool gen_ok = (gen == tier2_snap_gen);
    TranslationBlock *hdr = trace->header;
    Tier2TBRec *cur = gen_ok && tier2_snaps ?
        g_hash_table_lookup(tier2_snaps, hdr) : NULL;
    bool valid = cur && !(tb_cflags(hdr) & CF_INVALID);
    if (valid) {
        hdr->tier2_code = native_code;
        if (tier2_installed) {
            g_hash_table_replace(tier2_installed, hdr, native_code);
        }
        if (tier2_verbose || qemu_loglevel_mask(CPU_LOG_EXEC)) {
            qemu_log("[tier2] Installed op-walker Tier-2 code %p for header "
                     "pc=0x%" PRIx64 " (%u ops)\n",
                     native_code, cur->pc, cur->num_ops);
        }
    } else {
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
    tier2_snaps = g_hash_table_new_full(NULL, NULL, NULL, g_free);
    tier2_installed = g_hash_table_new(NULL, NULL);
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

    const char *env_snapmax = getenv("QEMU_TIER2_SNAP_MAX");
    if (env_snapmax) {
        unsigned long v = strtoul(env_snapmax, NULL, 0);
        if (v > 0 && v <= 1000000) {
            tier2_snap_max = (uint32_t)v;
        }
    }

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
}

void tier2_shutdown(void)
{
    if (!tier2_initialized) {
        return;
    }

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
    tier2_snap_count = 0;
    qemu_mutex_unlock(&tier2_snap_lock);
    qemu_mutex_destroy(&tier2_snap_lock);

    if (tier2_verbose && tier2_snap_hot_miss) {
        qemu_log("[tier2] hot traces without snapshots: %" PRIu64 "\n",
                 tier2_snap_hot_miss);
    }

    tier2_initialized = false;
}
