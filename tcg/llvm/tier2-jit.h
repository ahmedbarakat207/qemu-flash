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
    T2_GOTO_TB = 27, /* imm1 = idx; lowered to ret (rx_header | idx),
                       * i.e. exactly what unlinked TCG goto_tb returns */
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
} Tier2Op;

/* Stable condition enum. Mapped from TCGCond by tier2.c at capture time. */
typedef enum Tier2Cond {
    T2C_EQ = 0,
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
 *   QEMU_LD: dst,value-temp src1=addr-temp, imm1=oi verbatim,
 *            imm2=size_bytes | sign<<8 | bswap<<16 | mmuidx<<24
 *   QEMU_ST: src1=value-temp, src2=addr-temp, imm1/imm2 as above
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
    Tier2MemHelpers mem_helpers;
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

/* Offline self-test: compiles hand-built op streams (counting loop with
 * env ld/st, ALU, brcond; plus negative cases) and executes them against
 * a fake env buffer. Returns true iff checksums match. No QEMU needed. */
bool tier2_jit_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* TCG_LLVM_TIER2_JIT_H */
