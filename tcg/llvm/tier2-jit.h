/* LLVM Tier-2 JIT Compiler C/C++ Interface.
 * Clean POD boundary avoiding C++ keyword conflicts (class/typename in QOM).
 *
 * The trace descriptor below is a STABLE ABI decoupled from TCG internals:
 * tier2.c (which sees real TCG headers) snapshots live TCG ops into these
 * records at translation time; tier2-jit.cpp (standalone, LLVM-only) lowers
 * them to LLVM IR. Nothing here includes tcg.h or target cpu.h, so the JIT
 * never hardcodes guest struct layouts: env accesses always carry explicit
 * byte offsets captured from TCGTemp.mem_offset.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TCG_LLVM_TIER2_JIT_H
#define TCG_LLVM_TIER2_JIT_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define TIER2_MAX_TRACE_TBS 16

/* Cap on recorded helper calls per trace for cache name tables. */
#define TIER2_MAX_CALLS 32

/*
 * On-disk cache epoch. Bump on ANY change to walker emission semantics,
 * exit encoding, commit discipline, or anything else that alters what a
 * given op stream compiles to. The content key cannot see those (it
 * hashes inputs, not the emitter), so without this, objects written by
 * an older build load as false hits -- silent wrong-code execution.
 *
 * Epoch 2: T2_CALL helpers go through named externals instead of baked
 * addresses (epoch-1 objects jump to the recording boot's ASLR addresses
 * when loaded -- flaky SIGSEGV on warm runs).
 *
 * Epoch 4: Phase 5 goto_tb chain-stub linking with active TB tracking,
 * Phase 4 vector SIMD opcodes (T2_VEC_*), and Phase 3 flat RAM base.
 *
 * Epoch 5: Phase 5 bounded self-loop safepoint and split-WX RW/RX active TB.
 *
 * Epoch 6: Eliminate baked host pointers for active TB tracking to keep
 * cached traces completely relocatable across ASLR runs.
 *
 * Epoch 7: Add bitwise and arithmetic opcodes (mulsh, muluh, andc, orc, clz, ctz).
 */
#define TIER2_CACHE_EPOCH 7

/*
 * Tier-2 side-exit protocol. Compiled traces never return raw TB
 * pointers (they are ASLR-unstable across processes, which would poison
 * an on-disk code cache, and lifetime-unstable across retranslations).
 * Instead a side exit returns TIER2_EXIT_PROTOCOL | (k << 2) | idx,
 * where k is the trace TB index that is exiting and idx the TCG exit
 * index (0/1, or 3 for exit_tb REQUESTED). cpu_tb_exec resolves k
 * against the installed record below into a CURRENT rx pointer, or
 * falls back to a clean dispatcher lookup when no valid record exists.
 */
#define TIER2_EXIT_PROTOCOL (1ULL << 63)
#define TIER2_EXIT_TB_SHIFT 2
#define TIER2_EXIT_TB_MASK  0x3c

/* Bounds for one TB's snapshot. Larger TBs fall back to the TCG path. */
#define TIER2_JIT_MAX_TEMPS 512
#define TIER2_JIT_MAX_OPS 1024

/* CF_PARALLEL value mirrored from include/exec/translation-block.h.
 * Carried in Tier2TBRec.cflags so the JIT can refuse parallel-only
 * constructs (atomics/fences) without including QEMU headers. */
#define TIER2_CF_PARALLEL 0x00008000u

/* Stable op enum. Mapped from TCGOpcode by tier2.c at capture time. */
typedef enum Tier2Op {
    T2_UNSUPPORTED = 0,
    T2_MOV = 1,
    T2_ADD = 2,
    T2_SUB = 3,
    T2_MUL = 4,
    T2_AND = 5,
    T2_OR = 6,
    T2_XOR = 7,
    T2_NEG = 8,
    T2_NOT = 9,
    T2_SHL = 10,
    T2_SHR = 11,   /* logical */
    T2_SAR = 12,   /* arithmetic */
    T2_ROTL = 13,
    T2_ROTR = 14,
    T2_EXTRACT = 15,
    T2_SEXTRACT = 16,
    T2_DEPOSIT = 17,
    T2_EXT32U = 18,  /* extu_i32_i64: zero-extend low 32 */
    T2_EXT32S = 19,  /* ext_i32_i64: sign-extend low 32 */
    T2_EXTRL = 20,   /* extrl_i64_i32: low 32 -> 32-bit cell */
    T2_SETCOND = 21,
    T2_MOVCOND = 22,
    T2_BR = 23,
    T2_BRCOND = 24,
    T2_SETLABEL = 25,
    T2_EXIT_TB = 26, /* imm1 = dispatcher return value verbatim */
    T2_GOTO_TB = 27, /* imm1 = idx; internal edge iff next[] proves it,
                       * else ret (rx_origin | idx) like unlinked TCG */
    T2_GOTO_PTR = 28, /* src1 = addr temp. Static const target in trace =>
                       * internal edge; else side exit ret 0 (dispatcher
                       * re-derives from env; matches TCI null behavior) */
    /* Direct env accesses (TCG ld/st with byte offset from capture). */
    T2_LD8U = 30,
    T2_LD8S = 31,
    T2_LD16U = 32,
    T2_LD16S = 33,
    T2_LD32U = 34,
    T2_LD32S = 35,
    T2_LD32 = 36,
    T2_LD64 = 37,
    T2_ST8 = 38,
    T2_ST16 = 39,
    T2_ST32 = 40,
    T2_ST64 = 41,
    /* Guest memory (TCG qemu_ld/st). Only emitted when the caller sets
     * guest_mem_allowed; lowered to helper_*_mmu calls (tier-2 plan 3a). */
    T2_QEMU_LD = 50,
    T2_QEMU_ST = 51,
    /* Direct helper call (TCG call op). Args/return marshaled by exact
     * prototype decoded at capture (see T2TC_* below). */
    T2_CALL = 52,
    /* Byte swaps. imm1 = TCG_BSWAP_* flags (bswap16 only; 32/64 ignore). */
    T2_BSWAP16 = 53,
    T2_BSWAP32 = 54,
    T2_BSWAP64 = 55,
    /* negsetcond: dst = -(c1 cond c2 ? 1 : 0). imm1=Tier2Cond. */
    T2_NEGSETCOND = 56,
    /* Vector / SIMD opcodes (Phase 4: SSE/AVX -> ARM64 NEON) */
    T2_VEC_ADD = 60,
    T2_VEC_SUB = 61,
    T2_VEC_MUL = 62,
    T2_VEC_AND = 63,
    T2_VEC_OR  = 64,
    T2_VEC_XOR = 65,
    T2_VEC_NOT = 66,
    T2_VEC_SHL = 67,
    T2_VEC_SHR = 68,
    T2_VEC_SAR = 69,
    T2_VEC_DUP = 70,
    T2_VEC_LD  = 71,
    T2_VEC_ST  = 72,
    /* High multiply and bitwise operations */
    T2_MULSH   = 75,
    T2_MULUH   = 76,
    T2_ANDC    = 77,
    T2_ORC     = 78,
    T2_CLZ     = 79,
    T2_CTZ     = 80,
} Tier2Op;

/* Stable condition enum. Mapped from TCGCond by tier2.c at capture time. */
typedef enum Tier2Cond {    T2C_EQ = 0,
    T2C_NE = 1,
    T2C_LT = 2,
    T2C_GE = 3,
    T2C_GT = 4,
    T2C_LE = 5,
    T2C_LTU = 6,
    T2C_GEU = 7,
    T2C_GTU = 8,
    T2C_LEU = 9,
    T2C_TSTEQ = 10,
    T2C_TSTNE = 11,
} Tier2Cond;

/* Stable call-argument type codes for T2_CALL (see Tier2OpRec.imm2).
 * Values mirror dh_typecode() classes, decoupled from TCG headers. */
typedef enum Tier2CallType {
    T2T_VOID = 0,
    T2T_I32 = 2,
    T2T_I64 = 4,
    T2T_PTR = 6,
} Tier2CallType;

/*
 * One TCG temp, resolved at capture time. Temps are modelled as untyped
 * 64-bit cells (matching backend behavior: i32 defs zero-extend).
 * - is_const: cell holds const_val (masked to tbits at init).
 * - is_env: this temp IS the env pointer (ld/st base check).
 * - env_off >= 0: direct CPUArchState slot at this byte offset.
 * - env_off == -1: ordinary temp (zero-initialized SSA-promotable alloca).
 * - env_off == -2: global with non-env base; any use bails the compile.
 */
typedef struct Tier2TempRec {
    uint8_t is_const;
    uint8_t is_env;
    uint8_t tbits; /* 32 or 64 */
    uint8_t _pad;
    int32_t env_off;
    uint64_t const_val;
} Tier2TempRec;

/*
 * One op. Temp operands are compact indices (>= 0) or -1 when unused.
 * Field use per op:
 *   ALU3 (ADD/SUB/MUL/AND/OR/XOR/SHL/SHR/SAR/ROTL/ROTR): dst,src1,src2
 *   ALU2 (MOV/NEG/NOT/EXT32U/EXT32S/EXTRL): dst,src1
 *   EXTRACT/SEXTRACT: dst,src1, imm1=ofs, imm2=len
 *   DEPOSIT: dst,src1,src2, imm1=ofs, imm2=len
 *   LD_ENV: dst, imm1=env byte offset (also cross-checked via temp init)
 *   ST_ENV: src1, imm1=env byte offset
 *   SETCOND: dst,src1,src2, imm1=Tier2Cond
 *   MOVCOND: dst,src1(c1),src2(c2),src3(v1),src4(v2), imm1=Tier2Cond
 *   BR: imm1=label id.  BRCOND: src1,src2, imm1=cond, imm2=label
 *   SETLABEL: imm1=label id
 *   EXIT_TB: imm1=return value.  GOTO_TB: imm1=idx
 *   GOTO_PTR: src1=addr temp
 *   QEMU_LD: dst,value-temp src1=addr-temp, imm1=oi verbatim,
 *            imm2=size_bytes | sign<<8 | addr32<<9 | bswap<<16 |
 *            mmuidx<<24 | inline_forbidden<<31. addr32 set when the
 *            address temp is 32-bit (compare in 32 bits); the JIT takes
 *            the inline TLB fast path unless inline_forbidden.
 *   QEMU_ST: src1=value-temp, src2=addr-temp, imm1/imm2 as above
 *   CALL: dst=ret-temp or -1, src1..src4=first input temps, imm1=func
 *            address, imm2=retcode | a0<<3 | a1<<6 | a2<<9 | a3<<12 |
 *            nr_in<<16 | nr_out<<20 (codes: Tier2CallType; max 4 inputs,
 *            1 output, no 128-bit). Bit-exact C call to that address.
 *   BSWAP16/32/64: dst,src1, imm1=flags (16 only)
 *   NEGSETCOND: dst,src1,src2, imm1=Tier2Cond
 * bits: effective width, 32 or 64 (from TCGOP_TYPE at capture).
 */
typedef struct Tier2OpRec {
    uint16_t op;
    uint8_t bits;
    uint8_t _pad;
    int32_t dst;
    int32_t src1;
    int32_t src2;
    int32_t src3;
    int32_t src4;
    int64_t imm1;
    int64_t imm2;
} Tier2OpRec;

/* Snapshot of one TB's post-optimization TCG ops. */
typedef struct Tier2TBRec {
    uint64_t pc;
    uint32_t size;
    uint32_t icount;
    uint32_t cflags;
    uint32_t num_temps;
    uint32_t num_ops;
    uint32_t _pad;
    const void *code_ptr; /* TB rx code pointer (prologue fallback) */
    const void *rx_tb;    /* TB rx pointer (goto_tb lowering, validation) */
    Tier2TempRec temps[TIER2_JIT_MAX_TEMPS];
    Tier2OpRec ops[TIER2_JIT_MAX_OPS];
} Tier2TBRec;

/* Guest-memory helper table (option 3a: call helpers, no inline TLB yet).
 * Resolved by tier2.c via dlsym; NULL entries must never be called. */
typedef struct Tier2MemHelpers {
    void *ld8u;  /* helper_ldub_mmu */
    void *ld8s;  /* helper_ldsb_mmu */
    void *ld16u; /* helper_lduw_mmu */
    void *ld16s; /* helper_ldsw_mmu */
    void *ld32u; /* helper_ldul_mmu */
    void *ld32s; /* helper_ldsl_mmu */
    void *ld64;  /* helper_ldq_mmu */
    void *st8;   /* helper_stb_mmu */
    void *st16;  /* helper_stw_mmu */
    void *st32;  /* helper_stl_mmu */
    void *st64;  /* helper_stq_mmu */
} Tier2MemHelpers;

/*
 * SoftMMU TLB layout for the inline fast path. All values are measured
 * by tier2.c with offsetof/sizeof (never hardcoded here), mirroring
 * tcg/tcg.c:tlb_mask_table_ofs() exactly:
 *   f_desc = env + f0_off + (n_modes - 1 - mmuidx) * f_stride
 * Fast-path algorithm mirrors the TCG backend's prepare_host_addr():
 * same table, same comparator, same addend -- so hits, misses, fills,
 * MMIO, watchpoints, dirty tracking and large pages behave identically.
 * Anything the fast path declines goes to the helper slow path.
 */
typedef struct Tier2TlbLayout {
    bool valid;            /* false in user-mode (no softmmu TLB) */
    uint8_t _pad[7];
    int64_t f0_off;        /* byte offset env -> f[0] (negative) */
    int64_t f_stride;      /* sizeof(CPUTLBDescFast) */
    uint32_t n_modes;      /* NB_MMU_MODES */
    uint32_t entry_bits;   /* CPU_TLB_ENTRY_BITS */
    int64_t e_read;        /* offsetof entry.addr_read */
    int64_t e_write;       /* offsetof entry.addr_write */
    int64_t e_addend;      /* offsetof entry.addend */
    uint32_t page_bits;    /* TARGET_PAGE_BITS */
    uint64_t page_mask;    /* TARGET_PAGE_MASK */
} Tier2TlbLayout;

/* Generic TB descriptor passed to the JIT compiler (legacy fields kept so
 * old callers still build). num_ops == 0 selects the legacy fallback path
 * (deprecated dbc special-case, else prologue trampoline). */
typedef struct Tier2TBDesc {
    uint64_t pc;
    uint32_t size;
    uint32_t icount;
    const void *code_ptr;
    void *tb_obj;
} Tier2TBDesc;

/* Complete closed loop trace descriptor */
typedef struct Tier2TraceDesc {
    uint32_t num_tbs;
    Tier2TBDesc tbs[TIER2_MAX_TRACE_TBS];
    uint64_t total_exec_count;
    /* Guest PC of the trace header (the hot TB that rooted collection).
     * NOTE: this is NOT necessarily tbs[0].pc -- history-derived traces
     * list oldest-first with the header last. */
    uint64_t header_pc;
    /*
     * Deprecated workload-loop PC (0 = disabled). When nonzero and any
     * TB in the trace covers it, the whole-loop native body is emitted
     * instead of walking ops. Set by tier2.c from QEMU_TIER2_WORKLOAD_PC.
     */
    uint64_t workload_pc;
    /* Generic walker input. Valid iff has_ops; tbs[i] mirrors recs[i]. */
    bool has_ops;
    bool guest_mem_allowed;
    uint32_t trace_id;
    const void *rx_header;
    /*
     * Fusion map. next[i] = trace index of the TB that TB i's goto_tb
     * chain provably reaches (jmp_dest-linked at discovery), or -1 for
     * "no proven edge: side-exit". next_slot[i] says WHICH goto_tb slot
     * (0/1) that edge was observed on; a goto_tb on any other slot
     * side-exits even when next[i] is set (the other slot may lead
     * elsewhere). Only proven edges become internal LLVM branches;
     * history adjacency is never trusted for code layout.
     * header_idx = index of the dispatch-entry TB (tbs[0] for chained
     * discovery, tbs[num_tbs-1] for history discovery).
     */
    int32_t next[TIER2_MAX_TRACE_TBS];
    int32_t next_slot[TIER2_MAX_TRACE_TBS];
    uint32_t header_idx;
    Tier2MemHelpers mem_helpers;
    Tier2TlbLayout tlb;
    /*
     * Address of helper_lookup_tb_ptr (filled by tier2.c; NULL in
     * user-mode builds). A goto_ptr whose address temp was most
     * recently defined by a call to exactly this helper tail-calls the
     * TCG prologue with the result instead of side-exiting: identical
     * functions, identical order, minus one dispatcher round trip.
     */
    const void *lookup_helper;
    /*
     * On-disk code-cache support. Call targets are recorded two ways:
     * call_addrs[] holds this process's addresses (link-time only, never
     * hashed), call_names[] the stable helper names in emission order
     * (hashed into the key; resolved to current addresses at load).
     * num_calls > TIER2_MAX_CALLS, any unresolvable target, or no_cache
     * skips caching for the trace (it still compiles normally).
     * guest_arch identifies the guest target for the key ("x86_64",
     * "i386", ...; empty disables caching -- better safe than colliding).
     */
    uint32_t num_calls;
    bool no_cache;
    uint8_t _pad2[3];
    char call_names[TIER2_MAX_CALLS][64];
    uint64_t call_addrs[TIER2_MAX_CALLS];
    /*
     * Back-edge safepoint (P5 native loops). has_safepoint: tier2.c
     * measured env->CPUState and interrupt_request offsets on a live
     * vCPU (identical for every CPU of this target). cpu_off =
     * (int64_t)cpu - (int64_t)env (negative); irq_off =
     * offsetof(CPUState, interrupt_request). The walker polls *irq on
     * self back-edges and side-exits when nonzero, so a native loop
     * can never starve interrupts. All three are hashed into the cache
     * key (build-derived). Unset = bail back-edges (side-exit, the old
     * behavior).
     */
    bool has_safepoint;
    uint8_t _pad3[7];
    int64_t cpu_off;
    int64_t irq_off;
    int64_t last_tb_off;
    uint64_t ram_base;
    uint64_t ram_size;
    char guest_arch[16];
    /* QEMU version string (upgrade safety: env layouts and helper
     * semantics follow the QEMU build). Empty disables caching. */
    char qemu_version[32];
    Tier2TBRec recs[TIER2_MAX_TRACE_TBS];
} Tier2TraceDesc;

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize LLVM native targets and ORC JIT execution engine */
bool tier2_jit_init(void *prologue_fn);

/* Compile a closed loop trace into native executable code */
void *tier2_jit_compile_trace(const Tier2TraceDesc *trace);

/* Release ORC resources for one compiled function (deferred-safe: the
 * entry is dropped; reclamation happens at the next flush/shutdown when
 * no vCPU can be inside it). Unknown pointers are ignored. */
void tier2_jit_invalidate(void *fn_ptr);

/* Release all ORC resources. Must be called with no vCPU running
 * (tb_flush exclusive context) or at shutdown. */
void tier2_jit_flush(void);

/* Shutdown JIT execution engine */
void tier2_jit_shutdown(void);

/* On-disk cache hits served so far (diagnostic for tests). */
uint64_t tier2_jit_cache_hits(void);

/* Offline self-test: compiles hand-built op streams (counting loop with
 * env ld/st, ALU, brcond; plus negative cases) and executes them against
 * a fake env buffer. Returns true iff checksums match. No QEMU needed. */
bool tier2_jit_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* TCG_LLVM_TIER2_JIT_H */
