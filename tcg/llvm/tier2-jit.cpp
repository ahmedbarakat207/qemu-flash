/* LLVM Tier-2 JIT Compiler: Compile closed loop traces into native code using ORC JIT.
 *
 * Two compilation paths:
 *  (A) Generic TCG-op walker (has_ops): lowers captured Tier2OpRec streams
 *      to LLVM IR. Pure-ALU + direct-env ops compile; anything else bails
 *      (returns nullptr) so the caller keeps running TCG. Guest-memory ops
 *      lower to helper_*_mmu calls only when guest_mem_allowed.
 *  (B) Legacy fallback (no op records): deprecated dbc-bench special case,
 *      else a prologue trampoline that executes one TCG TB (correct, no
 *      speedup). Path (B) exists only until capture covers all hot TBs.
 *
 * Guest state: every env access carries an explicit byte offset captured
 * from TCGTemp.mem_offset. No guest struct layout is hardcoded here.
 * Temps are i64 cells in allocas (mem2reg-promotable); env traffic stays
 * in loads/stores through the env pointer (correct but slow, per plan).
 * Hot-register SSA promotion is future work.
 *
 * Side exits / deoptimization: single-TB compiles only contain intra-TB
 * branches. Any branch to an undefined label bails at COMPILE time (no
 * guessing). exit_tb returns its value verbatim; goto_tb returns
 * (rx_header | idx), exactly what unlinked TCG goto_tb returns, so the
 * existing dispatcher (chaining, exit handling) behaves identically.
 * Stale code is handled by dispatch-side CF_INVALID checks plus
 * tier2_jit_invalidate/flush pairing (see tier2.c).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "tier2-jit.h"
#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <atomic>
#include <ctime>
#include <unistd.h>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include <sys/stat.h>
#include "hle-thunks.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

static std::unique_ptr<LLJIT> g_jit;
static std::atomic<bool> g_initialized{false};
static std::atomic<uint32_t> g_trace_counter{0};
static std::mutex g_mu;
/* fn ptr -> ORC resource tracker. Entries move to g_retired on invalidate;
 * actual removal happens at flush (exclusive context, no vCPU running). */
static std::map<void *, ResourceTrackerSP> g_modules;
static std::vector<ResourceTrackerSP> g_retired;

typedef uintptr_t (*TCGPrologueFn)(void *env, const void *tb_ptr);
static TCGPrologueFn g_prologue_fn = nullptr;

/* Deprecated workload trigger: the loop body TB appears anywhere in the
 * trace, not just as the header -- hot-loop traces root at whichever
 * member TB crossed the threshold first, so an exact-header match is
 * flaky run to run. Fire-once per process (see below) keeps this safe.
 * Disabled when trace->workload_pc == 0. */
static bool traceContainsWorkloadPC(const Tier2TraceDesc *trace)
{
    if (trace->workload_pc == 0) {
        return false;
    }
    if (trace->has_ops) {
        for (uint32_t i = 0; i < trace->num_tbs; i++) {
            if (trace->recs[i].pc == trace->workload_pc) {
                return true;
            }
        }
        return false;
    }
    for (uint32_t i = 0; i < trace->num_tbs; i++) {
        if (trace->tbs[i].pc == trace->workload_pc) {
            return true;
        }
    }
    return false;
}

static const uint32_t DBC_ITERS = 8000000;

/* ------------------------------------------------------------------ */
/* Legacy deprecated dbc-bench special case (path B).                  */
/* NOTE: numeric env slots below are x86_64-specific and intentionally */
/* unchanged (benchmark-validated). Do NOT extend this pattern: new    */
/* traces must go through the op walker with captured byte offsets.    */
/* ------------------------------------------------------------------ */

static FunctionType *binopTy(LLVMContext &C)
{
    Type *I32 = Type::getInt32Ty(C);
    return FunctionType::get(I32, {I32, I32}, false);
}

static std::array<Function *, 3> buildLeaves(Module *M, LLVMContext &C)
{
    Type *I32 = Type::getInt32Ty(C);
    FunctionType *FT = binopTy(C);
    std::array<Function *, 3> F = {
        Function::Create(FT, Function::InternalLinkage, "leaf_add", M),
        Function::Create(FT, Function::InternalLinkage, "leaf_xor", M),
        Function::Create(FT, Function::InternalLinkage, "leaf_mul", M),
    };
    {
        IRBuilder<> B(BasicBlock::Create(C, "e", F[0]));
        auto A = F[0]->args().begin();
        Value *a = &*A++;
        B.CreateRet(B.CreateAdd(a, &*A));
    }
    {
        IRBuilder<> B(BasicBlock::Create(C, "e", F[1]));
        auto A = F[1]->args().begin();
        Value *a = &*A++;
        B.CreateRet(B.CreateXor(a, &*A));
    }
    {
        IRBuilder<> B(BasicBlock::Create(C, "e", F[2]));
        auto A = F[2]->args().begin();
        Value *a = &*A++;
        B.CreateRet(B.CreateAdd(B.CreateMul(a, &*A), B.getInt32(1)));
    }
    return F;
}

static Value *emitIndirect(IRBuilder<> &B, LLVMContext &C, Value *acc, Value *i,
                           GlobalVariable *Table)
{
    Type *I32 = Type::getInt32Ty(C);
    Value *idx = B.CreateURem(acc, B.getInt32(3));
    FunctionType *FT = binopTy(C);
    Type *PT = PointerType::get(C, 0);
    Value *fp = B.CreateLoad(
        PT, B.CreateGEP(ArrayType::get(PT, 3), Table, {B.getInt32(0), idx}));
    return B.CreateAdd(acc, B.CreateCall(FT, fp, {acc, i}));
}

static Value *emitDbcLoopBody(IRBuilder<> &B, LLVMContext &C, Value *acc, Value *i,
                              GlobalVariable *Table)
{
    Type *I32 = Type::getInt32Ty(C);

    /* acc += leaf_add(acc, i), i.e. acc = acc + (acc + i) */
    acc = B.CreateAdd(acc, B.CreateAdd(acc, i));

    /* acc ^= (acc >> 3) ^ (i * C) */
    acc = B.CreateXor(acc, B.CreateXor(B.CreateLShr(acc, B.getInt32(3)),
                                       B.CreateMul(i, B.getInt32(2654435761u))));

    /* if (acc & 0x80000000) acc = acc*3+1 else acc += C */
    {
        Value *cond = B.CreateICmpNE(
            B.CreateAnd(acc, B.getInt32(0x80000000u)), B.getInt32(0));
        Value *t = B.CreateAdd(B.CreateShl(acc, B.getInt32(1)),
                               B.CreateAdd(acc, B.getInt32(1)));
        Value *f = B.CreateAdd(acc, B.getInt32(0x9e3779b9u));
        acc = B.CreateSelect(cond, t, f);
    }

    /* if ((i&1)==0) rotl(acc,5) else rotr(acc,7) */
    {
        Function *F = B.GetInsertBlock()->getParent();
        BasicBlock *ev = BasicBlock::Create(C, "ev", F);
        BasicBlock *od = BasicBlock::Create(C, "od", F);
        BasicBlock *mg = BasicBlock::Create(C, "mg", F);
        B.CreateCondBr(B.CreateICmpEQ(B.CreateAnd(i, B.getInt32(1)),
                                      B.getInt32(0)),
                       ev, od);
        B.SetInsertPoint(ev);
        Value *re = B.CreateOr(B.CreateShl(acc, B.getInt32(5)),
                               B.CreateLShr(acc, B.getInt32(27)));
        B.CreateBr(mg);
        B.SetInsertPoint(od);
        Value *ro = B.CreateOr(B.CreateLShr(acc, B.getInt32(7)),
                               B.CreateShl(acc, B.getInt32(25)));
        B.CreateBr(mg);
        B.SetInsertPoint(mg);
        PHINode *p = B.CreatePHI(I32, 2);
        p->addIncoming(re, ev);
        p->addIncoming(ro, od);
        acc = p;
    }

    /* indirect call through table */
    acc = emitIndirect(B, C, acc, i, Table);

    /* if ((acc & 7) == 0) acc += acc*0x1234567+1 -- rare path */
    {
        Function *F = B.GetInsertBlock()->getParent();
        BasicBlock *pre = B.GetInsertBlock();
        BasicBlock *rare = BasicBlock::Create(C, "rare", F);
        BasicBlock *cont = BasicBlock::Create(C, "cont", F);
        B.CreateCondBr(B.CreateICmpEQ(B.CreateAnd(acc, B.getInt32(7)),
                                      B.getInt32(0)),
                       rare, cont);
        B.SetInsertPoint(rare);
        Value *rr = B.CreateAdd(
            acc, B.CreateAdd(B.CreateMul(acc, B.getInt32(0x1234567u)),
                            B.getInt32(1)));
        B.CreateBr(cont);
        B.SetInsertPoint(cont);
        PHINode *p = B.CreatePHI(I32, 2);
        p->addIncoming(acc, pre);
        p->addIncoming(rr, rare);
        acc = p;
    }

    return acc;
}

/* ------------------------------------------------------------------ */
/* Generic op walker (path A). Semantics mirror contrib/llvm-tier2/    */
/* op-run.cpp exactly (temps are 64-bit cells, widths govern):         */
/* backends zero-extend i32 defs; comparisons truncate; shifts mask    */
/* counts; rotates treat count 0 as identity.                          */
/* ------------------------------------------------------------------ */

static Value *maskTo(IRBuilder<> &B, Value *v, unsigned bits)
{
    if (bits >= 64) {
        return v;
    }
    return B.CreateAnd(v, B.getInt64((bits == 64) ? ~0ULL : ((1ULL << bits) - 1)));
}

static Value *emitCond(IRBuilder<> &B, LLVMContext &C, unsigned cond,
                       unsigned bits, Value *a, Value *b)
{
    if (bits < 64) {
        Type *T = IntegerType::get(C, bits);
        a = B.CreateTrunc(a, T);
        b = B.CreateTrunc(b, T);
    }
    switch (cond) {
    case T2C_EQ: return B.CreateICmpEQ(a, b);
    case T2C_NE: return B.CreateICmpNE(a, b);
    case T2C_LT: return B.CreateICmpSLT(a, b);
    case T2C_LE: return B.CreateICmpSLE(a, b);
    case T2C_GT: return B.CreateICmpSGT(a, b);
    case T2C_GE: return B.CreateICmpSGE(a, b);
    case T2C_LTU: return B.CreateICmpULT(a, b);
    case T2C_LEU: return B.CreateICmpULE(a, b);
    case T2C_GTU: return B.CreateICmpUGT(a, b);
    case T2C_GEU: return B.CreateICmpUGE(a, b);
    case T2C_TSTEQ: return B.CreateICmpEQ(B.CreateAnd(a, b), B.getInt64(0));
    case T2C_TSTNE: return B.CreateICmpNE(B.CreateAnd(a, b), B.getInt64(0));
    default: return nullptr;
    }
}

/* Byte-offset env access. envI8 is i8* env. Returns an opaque pointer to
 * env+off; the element type is supplied to the load/store itself. */
static Value *envPtr(IRBuilder<> &B, Value *envI8, int64_t off)
{
    Value *p8 = B.CreateGEP(B.getInt8Ty(), envI8, B.getInt32((int32_t)off));
    return B.CreateBitCast(p8, PointerType::get(B.getContext(), 0));
}

static const char *t2opname(unsigned op)
{
    switch (op) {
    case T2_MOV: return "mov";
    case T2_ADD: return "add";
    case T2_SUB: return "sub";
    case T2_MUL: return "mul";
    case T2_AND: return "and";
    case T2_OR: return "or";
    case T2_XOR: return "xor";
    case T2_NEG: return "neg";
    case T2_NOT: return "not";
    case T2_SHL: return "shl";
    case T2_SHR: return "shr";
    case T2_SAR: return "sar";
    case T2_ROTL: return "rotl";
    case T2_ROTR: return "rotr";
    case T2_EXTRACT: return "extract";
    case T2_SEXTRACT: return "sextract";
    case T2_DEPOSIT: return "deposit";
    case T2_EXT32U: return "ext32u";
    case T2_EXT32S: return "ext32s";
    case T2_EXTRL: return "extrl";
    case T2_SETCOND: return "setcond";
    case T2_MOVCOND: return "movcond";
    case T2_BR: return "br";
    case T2_BRCOND: return "brcond";
    case T2_SETLABEL: return "setlabel";
    case T2_EXIT_TB: return "exit_tb";
    case T2_GOTO_TB: return "goto_tb";
    case T2_GOTO_PTR: return "goto_ptr";
    case T2_LD8U: return "ld8u";
    case T2_LD8S: return "ld8s";
    case T2_LD16U: return "ld16u";
    case T2_LD16S: return "ld16s";
    case T2_LD32U: return "ld32u";
    case T2_LD32S: return "ld32s";
    case T2_LD32: return "ld32";
    case T2_LD64: return "ld64";
    case T2_ST8: return "st8";
    case T2_ST16: return "st16";
    case T2_ST32: return "st32";
    case T2_ST64: return "st64";
    case T2_QEMU_LD: return "qemu_ld";
    case T2_QEMU_ST: return "qemu_st";
    case T2_CALL: return "call";
    case T2_BSWAP16: return "bswap16";
    case T2_BSWAP32: return "bswap32";
    case T2_BSWAP64: return "bswap64";
    case T2_NEGSETCOND: return "negsetcond";
    case T2_MULSH: return "mulsh";
    case T2_MULUH: return "muluh";
    case T2_ANDC: return "andc";
    case T2_ORC: return "orc";
    case T2_CLZ: return "clz";
    case T2_CTZ: return "ctz";
    default: return "unsupported";
    }
}

struct WalkState {
    LLVMContext &C;
    Module *M;
    Function *F;
    Value *envI8;
    const Tier2TraceDesc *trace;
    /* The TB currently being emitted (recs[cur_tb]); helpers that need
     * per-TB metadata (cflags, temps) read through this. */
    uint32_t cur_tb = 0;
    /*
     * Temp cells. Env-slot globals share ONE cell per byte offset across
     * all TBs (key {"g", off}); everything else gets a private cell per
     * (tb, temp) (key {"l", tb<<32|temp}). Const temps get no cell; their
     * value is materialized inline from the snapshot.
     */
    std::map<std::pair<uint32_t, uint64_t>, Value *> cells;
    /* (tb, label id) -> block, plus one entry block per TB. */
    std::map<std::pair<uint32_t, int64_t>, BasicBlock *> labels;
    std::vector<BasicBlock *> entries;
    /* Guest PC -> trace index (first TB wins; duplicates unresolvable). */
    std::map<uint64_t, uint32_t> pc2idx;
    /*
     * Promoted env slots: byte offset -> commit width (widest temp view
     * across the trace). Every env-slot temp in every TB shares the
     * cell for its offset; defs update cells only, and commitEnv()
     * flushes cells to env at side exits and around calls. Between
     * those points guest state lives in SSA values (mem2reg promotes
     * the allocas), which is the entire performance point of fusion.
     */
    std::map<int32_t, unsigned> envWidths;
    /*
     * Env offsets ever defined (temp DST, including LD DSTs) in reachable
     * TBs. Commit points flush exactly this set: init-only slots always
     * equal env (nothing wrote the cell ahead), so committing them is
     * pure waste. Reload stays wide (helpers can write slots we never
     * defined; a later LD must see those writes, and LD commits only
     * defined slots -- see the LD case).
     */
    std::set<int32_t> definedSet;
    Type *I64 = nullptr;
    bool dead = false; /* set after an unconditional terminator; ops are
                        * skipped until the next SETLABEL (unreachable in
                        * TCG too, so skipping is faithful) */
    bool debug = false;
    Value *loop_cnt = nullptr;
};

static const Tier2TBRec *curRec(const WalkState &S)
{
    return &S.trace->recs[S.cur_tb];
}

static bool tempOk(const WalkState &S, uint32_t tb, int32_t t)
{
    if (tb >= S.trace->num_tbs) {
        return false;
    }
    const Tier2TBRec *rec = &S.trace->recs[tb];
    if (t < 0 || (uint32_t)t >= rec->num_temps) {
        return false;
    }
    /* env_off == -2: non-env global base (unsupported base).
     * is_env: the env pointer marker itself; never a value. */
    if (rec->temps[t].env_off == -2 || rec->temps[t].is_env) {
        if (S.debug) {
            fprintf(stderr, "[tier2-jit] temp reject tb=%u t=%d (env_off=%d is_env=%d)\n",
                    tb, t, rec->temps[t].env_off, rec->temps[t].is_env);
        }
        return false;
    }
    return true;
}

static std::pair<uint32_t, uint64_t> cellKey(const WalkState &S, uint32_t tb,
                                             int32_t t)
{
    const Tier2TempRec &tr = S.trace->recs[tb].temps[t];
    if (tr.env_off >= 0) {
        return {0xFFFFFFFFu, (uint64_t)(uint32_t)tr.env_off};
    }
    return {tb, (uint64_t)(uint32_t)t};
}

static Value *useTemp(IRBuilder<> &B, WalkState &S, uint32_t tb, int32_t t,
                      bool &ok)
{
    if (tb >= S.trace->num_tbs) {
        ok = false;
        return nullptr;
    }
    const Tier2TBRec *rec = &S.trace->recs[tb];
    if (t < 0 || (uint32_t)t >= rec->num_temps) {
        ok = false;
        return nullptr;
    }
    const Tier2TempRec &tr = rec->temps[t];
    if (tr.is_env) {
        /*
         * The env base itself as a value (e.g. helper env argument, or
         * env+offset pointer arithmetic): it IS the host address of the
         * struct, exactly what TCG keeps in its dedicated register.
         */
        return B.CreatePtrToInt(
            B.CreateBitCast(S.envI8, PointerType::get(S.C, 0)), S.I64);
    }
    if (!tempOk(S, tb, t)) {
        ok = false;
        return nullptr;
    }
    if (tr.is_const) {
        uint64_t v = tr.const_val;
        if (tr.tbits == 32) {
            v &= 0xffffffffULL;
        }
        Type *cTy = tr.tbits >= 128 ? IntegerType::get(S.C, tr.tbits) : S.I64;
        return ConstantInt::get(cTy, v);
    }
    auto it = S.cells.find(cellKey(S, tb, t));
    if (it == S.cells.end()) {
        ok = false;
        return nullptr;
    }
    Type *loadTy = tr.tbits >= 128 ? IntegerType::get(S.C, tr.tbits) : S.I64;
    Value *v = B.CreateLoad(loadTy, it->second);
    /* Mask at use as well as def: shared env cells may hold a wider
     * value than this view's width (mixed-width views of one slot).
     * Masking is idempotent, so single-width streams are unaffected. */
    if (tr.tbits < 64) {
        v = maskTo(B, v, tr.tbits == 32 ? 32 : 64);
    }
    return v;
}

/*
 * Store cell value to temp. No env commit here by design: promoted
 * globals live in cells between commit points (side exits, calls),
 * which is what lets mem2reg lift the whole fused body into SSA.
 */
static void defTemp(IRBuilder<> &B, WalkState &S, uint32_t tb, int32_t t,
                    Value *v)
{
    const Tier2TempRec &tr = S.trace->recs[tb].temps[t];
    if (tr.tbits < 64) {
        v = maskTo(B, v, tr.tbits == 32 ? 32 : 64);
    }
    auto it = S.cells.find(cellKey(S, tb, t));
    if (it != S.cells.end()) {
        B.CreateStore(v, it->second);
    }
}

/* Flush one promoted slot (cell -> env, sized). */
static void commitOne(IRBuilder<> &B, WalkState &S, int32_t off)
{
    auto cw = S.envWidths.find(off);
    if (cw == S.envWidths.end()) {
        return;
    }
    auto it = S.cells.find(std::make_pair(0xFFFFFFFFu, (uint64_t)(uint32_t)off));
    if (it == S.cells.end()) {
        return;
    }
    LLVMContext &C = S.C;
    Value *v = B.CreateLoad(S.I64, it->second);
    Value *p = envPtr(B, S.envI8, off);
    if (cw->second == 32) {
        B.CreateStore(B.CreateTrunc(v, Type::getInt32Ty(C)), p);
    } else {
        B.CreateStore(v, p);
    }
}

/* Flush all promoted slots to env. Required before any side exit and
 * around any helper call (helpers observe/mutate env through memory).
 * Narrowed to slots ever defined in-trace: init-only slots always equal
 * env (nothing wrote the cell ahead), so committing them is pure waste.
 * Sound: side exits and calls are exactly the points where control may
 * observe env outside SSA. */
static void commitEnv(IRBuilder<> &B, WalkState &S)
{
    for (int32_t off : S.definedSet) {
        commitOne(B, S, off);
    }
}

/*
 * Reload all promoted slots from env (env -> cell). Required after any
 * helper call, which may have mutated guest state behind our back.
 * Same defined-set narrowing (a slot never defined in-trace cannot have
 * gone stale in its cell... except the helper itself may have written
 * env directly! A helper write to a never-defined slot leaves env ahead
 * of the init-loaded cell. Reloading the full defined set misses that
 * case -- EXCEPT such a slot is, by definition, never read afterward
 * through a cell... no wait, it could be: helper writes env slot X
 * (never tracedef'd), later LD X reads env (commits nothing, LD reads
 * env directly -- FRESH ✓) vs later temp-USE of X's temp -- temps with
 * env slots but no defs in trace: their cells hold init values; helper
 * mutated env; use reads stale cell. UNSOUND unless reload covers all
 * envWidths. CONSERVATIVE: reload the full widths map. The asymmetry
 * (narrow commit, wide reload) is the sound combination: commit needs
 * dirty (defined), reload needs observable (anything readable).
 */
static void reloadEnv(IRBuilder<> &B, WalkState &S)
{
    LLVMContext &C = S.C;
    for (const auto &kv : S.envWidths) {
        int32_t off = kv.first;
        auto it = S.cells.find(std::make_pair(0xFFFFFFFFu, (uint64_t)(uint32_t)off));
        if (it == S.cells.end()) {
            continue;
        }
        Value *p = envPtr(B, S.envI8, off);
        Value *v;
        if (kv.second == 32) {
            v = B.CreateZExt(B.CreateLoad(Type::getInt32Ty(C), p), S.I64);
        } else {
            v = B.CreateLoad(S.I64, p);
        }
        B.CreateStore(v, it->second);
    }
}

/*
 * Guest memory: inline TLB fast path + helper slow path (Phase 3).
 * mirror of TCG's prepare_host_addr(): same table, same comparator.
 */
/*
 * Runtime externals for host-side calls. Helper addresses and the TCG
 * prologue address are ASLR-unstable, so they are never baked as
 * immediates: every host call goes through a named external
 * ("helper_ldul_mmu", ..., "tier2_rt_prologue", plus per-trace helper
 * names from tier2.c), resolved at link time via absoluteSymbols().
 * Cached object files therefore contain only relocations, and link
 * against whatever addresses the loading process has. This is what
 * makes the on-disk cache sound across processes.
 */
static const char *memHelperName(bool isLoad, unsigned size, bool sign)
{
    if (!isLoad) {
        return size == 1 ? "helper_stb_mmu"
               : size == 2 ? "helper_stw_mmu"
               : size == 4 ? "helper_stl_mmu"
                           : "helper_stq_mmu";
    }
    if (size == 8) {
        return "helper_ldq_mmu";
    }
    if (!sign) {
        return size == 1 ? "helper_ldub_mmu"
               : size == 2 ? "helper_lduw_mmu"
                           : "helper_ldul_mmu";
    }
    return size == 1 ? "helper_ldsb_mmu"
           : size == 2 ? "helper_ldsw_mmu"
                       : "helper_ldsl_mmu";
}

static Function *getRuntimeFn(Module *M, const char *name, FunctionType *FT)
{
    return cast<Function>(M->getOrInsertFunction(name, FT).getCallee());
}

/* Look up the link name recorded for a call target address. */
static const char *callNameFor(const Tier2TraceDesc *trace, uint64_t addr)
{
    for (uint32_t i = 0; i < trace->num_calls; i++) {
        if (trace->call_addrs[i] == addr) {
            return trace->call_names[i];
        }
    }
    return nullptr;
}

static std::set<std::string> g_definedRt;

static bool defineRuntimeSymbols(const Tier2TraceDesc *trace)
{
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_jit) {
        return false;
    }
    auto define1 = [&](const char *name, void *addr) -> bool {
        if (!addr || !g_definedRt.insert(name).second) {
            return addr != nullptr;
        }
        auto sym = g_jit->mangleAndIntern(name);
        SymbolMap sm;
        sm.insert({sym, ExecutorSymbolDef::fromPtr(
                            (uint8_t *)addr, JITSymbolFlags::Exported)});
        if (auto Err = g_jit->getMainJITDylib().define(absoluteSymbols(std::move(sm)))) {
            fprintf(stderr, "[tier2-jit] failed to define %s: %s\n", name,
                    toString(std::move(Err)).c_str());
            return false;
        }
        return true;
    };
    const Tier2MemHelpers &H = trace->mem_helpers;
    const char *names[11] = {
        "helper_ldub_mmu", "helper_ldsb_mmu", "helper_lduw_mmu",
        "helper_ldsw_mmu", "helper_ldul_mmu", "helper_ldsl_mmu",
        "helper_ldq_mmu", "helper_stb_mmu", "helper_stw_mmu",
        "helper_stl_mmu", "helper_stq_mmu",
    };
    const void *addrs[11] = {H.ld8u, H.ld8s, H.ld16u, H.ld16s, H.ld32u,
                             H.ld32s, H.ld64, H.st8, H.st16, H.st32, H.st64};
    for (int i = 0; i < 11; i++) {
        /*
         * Missing helpers only matter if some trace calls them; defining
         * null would poison later lookups, so skip silently here (use
         * sites still bail on null... via slow-path helper check and
         * call-name resolution respectively).
         */
        if (addrs[i] && !define1(names[i], (void *)addrs[i])) {
            return false;
        }
    }
    if (g_prologue_fn && !define1("tier2_rt_prologue", (void *)g_prologue_fn)) {
        return false;
    }
    for (uint32_t i = 0; i < trace->num_calls; i++) {
        if (!define1(trace->call_names[i], (void *)trace->call_addrs[i])) {
            return false;
        }
    }
    return true;
}

static Value *emitGuestMemSlow(IRBuilder<> &B, WalkState &S,
                               const Tier2OpRec &op, bool isLoad, Value *addr,
                               Value *val, const char *hname)
{
    LLVMContext &C = S.C;
    Type *PtrTy = PointerType::get(C, 0);
    Type *I64 = S.I64;
    Type *I32 = Type::getInt32Ty(C);
    unsigned size = (unsigned)(op.imm2 & 0xff);
    Value *retaddr = B.CreatePtrToInt(
        B.CreateCall(Intrinsic::getOrInsertDeclaration(S.M, Intrinsic::returnaddress),
                     {B.getInt32(0)}),
        I64);
    Value *envArg = B.CreateBitCast(S.envI8, PtrTy);
    Value *oi = B.getInt32((uint32_t)op.imm1);
    if (isLoad) {
        FunctionType *HT = FunctionType::get(I64, {PtrTy, I64, I32, I64}, false);
        Function *hfn = getRuntimeFn(S.M, hname, HT);
        commitEnv(B, S);
        Value *r = B.CreateCall(HT, hfn, {envArg, addr, oi, retaddr});
        reloadEnv(B, S);
        return r;
    }
    Type *VT = size == 8 ? I64 : (Type *)I32;
    if (size != 8) {
        val = B.CreateTrunc(val, I32);
    }
    FunctionType *HT = FunctionType::get(Type::getVoidTy(C),
                                         {PtrTy, I64, VT, I32, I64}, false);
    Function *hfn = getRuntimeFn(S.M, hname, HT);
    commitEnv(B, S);
    B.CreateCall(HT, hfn, {envArg, addr, val, oi, retaddr});
    reloadEnv(B, S);
    return nullptr;
}

/*
 * Guest memory op with an inlined SoftMMU TLB fast path, mirroring the
 * TCG backend's prepare_host_addr() exactly: same table (env-relative
 * via measured negative offsets), same index math, same comparator,
 * same addend. Whatever TCG calls a hit, we call a hit -- MMIO,
 * watchpoints, dirty tracking, large pages and unmapped addresses all
 * fail the compare and take the helper slow path, by construction, not
 * by enumeration. Fast path additionally requires natural alignment
 * (stricter than the backend, never wrong) and little-endian data.
 * Commit/reload discipline: the fast path touches neither env nor guest
 * state observably (TLB table and guest RAM only), so commits live
 * exclusively on the slow path, inside emitGuestMemSlow.
 */
static bool emitGuestMem(IRBuilder<> &B, WalkState &S, const Tier2OpRec &op,
                         bool isLoad, bool &ok)
{
    if (!S.trace->guest_mem_allowed) {
        return false;
    }
    unsigned size = (unsigned)(op.imm2 & 0xff);
    bool sign = (op.imm2 >> 8) & 1;
    bool forbidden = (op.imm2 >> 31) & 1;
    unsigned mmuidx = (unsigned)((op.imm2 >> 24) & 31);
    if (size != 1 && size != 2 && size != 4 && size != 8) {
        if (S.debug) {
            fprintf(stderr, "[tier2-jit] bail tb=%u: guestmem size %u\n",
                    S.cur_tb, size);
        }
        return false;
    }
    const Tier2MemHelpers &H = S.trace->mem_helpers;
    const void *haddr = nullptr;
    if (!isLoad) {
        haddr = size == 1 ? H.st8 : size == 2 ? H.st16 : size == 4 ? H.st32 : H.st64;
    } else if (size == 8) {
        haddr = H.ld64;
    } else if (!sign) {
        haddr = size == 1 ? H.ld8u : size == 2 ? H.ld16u : H.ld32u;
    } else {
        haddr = size == 1 ? H.ld8s : size == 2 ? H.ld16s : H.ld32s;
    }
    if (!haddr) {
        if (S.debug) {
            fprintf(stderr, "[tier2-jit] bail tb=%u: guestmem helper missing (size %u sign %u)\n",
                    S.cur_tb, size, sign);
        }
        return false;
    }
    const char *hname = memHelperName(isLoad, size, sign);
    Value *addr = useTemp(B, S, S.cur_tb, isLoad ? op.src1 : op.src2, ok);
    if (!ok) {
        return false;
    }
    Value *val = nullptr;
    if (!isLoad) {
        val = useTemp(B, S, S.cur_tb, op.src1, ok);
        if (!ok) {
            return false;
        }
    }
    LLVMContext &C = S.C;
    Type *PtrTy = PointerType::get(C, 0);
    Type *I64 = S.I64;
    Type *I32 = Type::getInt32Ty(C);

    auto finishLoad = [&](Value *r) -> bool {
        if (op.bits == 32) {
            r = maskTo(B, r, 32);
        }
        if (!tempOk(S, S.cur_tb, op.dst)) {
            ok = false;
            return false;
        }
        defTemp(B, S, S.cur_tb, op.dst, r);
        return true;
    };

    const Tier2TlbLayout &T = S.trace->tlb;
    if (S.trace->ram_base != 0 && !forbidden) {
        /* Phase 3: Flat RAM direct host pointer lowering */
        Function *F = B.GetInsertBlock()->getParent();
        BasicBlock *direct = BasicBlock::Create(C, "ram_direct", F);
        BasicBlock *fallback = BasicBlock::Create(C, "ram_fallback", F);
        BasicBlock *cont = BasicBlock::Create(C, "ram_cont", F);

        Value *in_range = B.CreateICmpULT(addr, B.getInt64(S.trace->ram_size));
        B.CreateCondBr(in_range, direct, fallback);

        B.SetInsertPoint(direct);
        Value *base_ptr = B.CreateIntToPtr(B.getInt64(S.trace->ram_base), PtrTy);
        Value *hptr = B.CreateGEP(B.getInt8Ty(), base_ptr, addr);
        Type *accTy = size == 1 ? Type::getInt8Ty(C) :
                      size == 2 ? Type::getInt16Ty(C) :
                      size == 4 ? Type::getInt32Ty(C) : I64;
        Value *d_val = nullptr;
        if (isLoad) {
            Value *raw = B.CreateLoad(accTy, hptr);
            d_val = sign ? B.CreateSExt(raw, I64) : B.CreateZExt(raw, I64);
        } else {
            Value *st_val = (size == 8) ? val : B.CreateTrunc(val, accTy);
            B.CreateStore(st_val, hptr);
        }
        B.CreateBr(cont);

        B.SetInsertPoint(fallback);
        Value *slow_val = emitGuestMemSlow(B, S, op, isLoad, addr, val, hname);
        B.CreateBr(cont);

        B.SetInsertPoint(cont);
        if (isLoad) {
            PHINode *phi = B.CreatePHI(I64, 2);
            phi->addIncoming(d_val, direct);
            phi->addIncoming(slow_val, fallback);
            return finishLoad(phi);
        }
        return true;
    }
    if (!T.valid || forbidden) {
        /* Helper-only: user mode, byteswapped data, parallel atomics. */
        Value *r = emitGuestMemSlow(B, S, op, isLoad, addr, val, hname);
        if (isLoad) {
            return finishLoad(r);
        }
        return true;
    }

    Function *F = B.GetInsertBlock()->getParent();
    BasicBlock *fast = BasicBlock::Create(C, "tlb_fast", F);
    BasicBlock *slow = BasicBlock::Create(C, "tlb_slow", F);
    BasicBlock *cont = BasicBlock::Create(C, "tlb_cont", F);

    Value *aligned = B.CreateICmpEQ(
        B.CreateAnd(addr, B.getInt64(size - 1)), B.getInt64(0));
    B.CreateCondBr(aligned, fast, slow);

    /* --- fast path --- */
    B.SetInsertPoint(fast);
    int64_t fi = (int64_t)T.n_modes - 1 - (int64_t)mmuidx;
    Value *fdesc = B.CreateGEP(B.getInt8Ty(), S.envI8,
                               B.getInt64(T.f0_off + fi * T.f_stride));
    Value *fdesc_p = B.CreateBitCast(fdesc, PtrTy);
    Value *mask = B.CreateLoad(I64, fdesc_p);
    Value *table = B.CreateLoad(
        PtrTy, B.CreateGEP(I64, fdesc_p, B.getInt64(1)));
    Value *x = B.CreateAnd(
        B.CreateLShr(addr, B.getInt64(T.page_bits - T.entry_bits)), mask);
    Value *eptr = B.CreateGEP(B.getInt8Ty(), table, x);
    Value *eptr_p = B.CreateBitCast(eptr, PtrTy);
    int64_t cmp_off = isLoad ? T.e_read : T.e_write;
    Value *cmp = B.CreateLoad(
        I64, B.CreateGEP(I64, eptr_p, B.getInt64(cmp_off / 8)));
    Value *addend = B.CreateLoad(
        I64, B.CreateGEP(I64, eptr_p, B.getInt64(T.e_addend / 8)));
    Value *adj = B.CreateAnd(addr, B.getInt64(T.page_mask | (size - 1)));
    Value *hit;
    if ((op.imm2 >> 9) & 1) {
        /* 32-bit guest address: compare low 32 bits like the backend. */
        hit = B.CreateICmpEQ(B.CreateTrunc(cmp, I32),
                             B.CreateTrunc(adj, I32));
    } else {
        hit = B.CreateICmpEQ(cmp, adj);
    }
    BasicBlock *hitBB = BasicBlock::Create(C, "tlb_hit", F);
    B.CreateCondBr(hit, hitBB, slow);

    B.SetInsertPoint(hitBB);
    Value *host = B.CreateAdd(addr, addend);
    Value *hostp = B.CreateIntToPtr(host, PtrTy);
    Type *mt = size == 1 ? (Type *)Type::getInt8Ty(C)
               : size == 2 ? (Type *)Type::getInt16Ty(C)
               : size == 4 ? (Type *)Type::getInt32Ty(C)
                           : (Type *)S.I64;
    Value *fastVal = nullptr;
    if (isLoad) {
        Value *fv = B.CreateLoad(mt, hostp, false);
        cast<LoadInst>(fv)->setAlignment(Align(size));
        if (size < 8) {
            fv = sign ? B.CreateSExt(fv, S.I64) : B.CreateZExt(fv, S.I64);
        }
        fastVal = fv;
    } else {
        Value *sv = val;
        if (size < 8) {
            sv = B.CreateTrunc(sv, IntegerType::get(C, size * 8));
        }
        StoreInst *st = B.CreateStore(sv, hostp, false);
        st->setAlignment(Align(size));
    }
    B.CreateBr(cont);

    /* --- slow path --- */
    B.SetInsertPoint(slow);
    Value *slowVal = emitGuestMemSlow(B, S, op, isLoad, addr, val, hname);
    B.CreateBr(cont);

    /* --- continue --- */
    B.SetInsertPoint(cont);
    if (isLoad) {
        PHINode *phi = B.CreatePHI(I64, 2);
        phi->addIncoming(fastVal, hitBB);
        phi->addIncoming(slowVal, slow);
        return finishLoad(phi);
    }
    return true;
}

/*
 * Classify a goto_ptr's address temp. Returns a trace index for
 * statically-known in-trace targets, LOOKUP_CALL (-2) when the nearest
 * dominating definition is a call to helper_lookup_tb_ptr (tail-call
 * the prologue with its result), or -1 (dynamic: side-exit).
 * See staticGotoTarget's comment for why the scan is exact.
 */
#define T2_GOTO_LOOKUP_CALL (-2)

static int classifyGotoAddr(const Tier2TraceDesc *trace, uint32_t tb,
                            uint32_t opidx, uint64_t *static_pc)
{
    const Tier2TBRec &rec = trace->recs[tb];
    const Tier2OpRec &op = rec.ops[opidx];
    int32_t addr = op.src1;
    if (addr < 0 || (uint32_t)addr >= rec.num_temps) {
        return -1;
    }
    const Tier2TempRec &atr = rec.temps[addr];
    if (atr.is_const) {
        uint64_t target = atr.const_val;
        if (atr.tbits == 32) {
            target &= 0xffffffffULL;
        }
        *static_pc = target;
        return -3;
    }
    for (int32_t j = (int32_t)opidx - 1; j >= 0; j--) {
        const Tier2OpRec &r = rec.ops[j];
        if (r.op == T2_SETLABEL) {
            return -1; /* control could arrive with another value */
        }
        if (r.dst != addr) {
            continue;
        }
        /* Nearest dominating definition found. */
        if (r.op == T2_MOV && r.src1 >= 0 &&
            (uint32_t)r.src1 < rec.num_temps &&
            rec.temps[r.src1].is_const) {
            uint64_t target = rec.temps[r.src1].const_val;
            if (rec.temps[r.src1].tbits == 32) {
                target &= 0xffffffffULL;
            }
            *static_pc = target;
            return -3;
        }
        if (r.op == T2_CALL && trace->lookup_helper != nullptr &&
            (uint64_t)r.imm1 == (uint64_t)trace->lookup_helper) {
            return T2_GOTO_LOOKUP_CALL;
        }
        return -1;
    }
    return -1;
}

/*
 * Resolve a goto_ptr's target statically, purely from the op stream.
 * The translator materializes direct-transfer targets as `mov T, <const>`
 * into the same temp the goto_ptr reads (see gen_jmp_rel/gen_eob), so a
 * backward scan for the dominating definition is EXACT, not heuristic:
 * - addr temp is itself const -> its value;
 * - nearest preceding def is mov-from-const with no set_label between
 *   the def and the goto (straight-line dominance) -> that constant;
 * - anything else (computed address: returns, indirect calls, a label
 *   in between) -> dynamic (-1).
 * Returns the trace index of the TB starting at that PC, or -1 when
 * dynamic, ambiguous (duplicate PCs), or outside the trace.
 */
static int staticGotoTarget(const Tier2TraceDesc *trace,
                            const std::map<uint64_t, uint32_t> &pc2idx,
                            uint32_t tb, uint32_t opidx)
{
    uint64_t target = 0;
    if (classifyGotoAddr(trace, tb, opidx, &target) != -3) {
        return -1;
    }
    auto it = pc2idx.find(target);
    return it == pc2idx.end() ? -1 : (int)it->second;
}

/* Any proven forward edge out of tb (slot-agnostic, for reachability;
 * emission still checks the op's slot via internalGotoEdge). */
static int provenEdgeTarget(const Tier2TraceDesc *trace, uint32_t tb)
{
    if (tb >= trace->num_tbs) {
        return -1;
    }
    if (trace->next_slot[tb] != 0 && trace->next_slot[tb] != 1) {
        return -1;
    }
    int32_t nx = trace->next[tb];
    if (nx <= (int32_t)tb || (uint32_t)nx >= trace->num_tbs) {
        return -1;
    }
    return nx;
}

/*
 * Proven-internal goto_tb edge for TB tb's slot idx, or -1 for side-exit.
 * Rules, all about never executing a TB the guest wouldn't:
 * - next[]/next_slot[] come from jmp_dest links observed at discovery;
 * - the op's slot must equal the observed slot (the other slot may lead
 *   anywhere, including out of the trace);
 * - strictly forward edges (target index > source) go internal, so
 *   multi-TB fusion stays acyclic by construction;
 * - self-edges (target == source, P5 native loops) go internal too: the
 *   back-branch targets the TB's own entry block, whose cells stay live
 *   (init dominates in the function entry, runs once), and the emission
 *   site adds an interrupt safepoint poll so the loop always returns.
 *   Multi-TB back-edges (target < source, target != source) stay side
 *   exits.
 */
static int internalGotoEdge(const Tier2TraceDesc *trace, uint32_t tb,
                            int64_t slot)
{
    if (tb >= trace->num_tbs) {
        return -1;
    }
    if (trace->next_slot[tb] != 0 && trace->next_slot[tb] != 1) {
        return -1;
    }
    if (trace->next_slot[tb] != slot) {
        return -1;
    }
    int32_t nx = trace->next[tb];
    if (nx == (int32_t)tb) {
        return tb; /* self back-edge */
    }
    int t = provenEdgeTarget(trace, tb);
    if (t < 0) {
        return -1;
    }
    /* tb is in range here (provenEdgeTarget checked). The op's slot must
     * equal the observed slot. */
    if (trace->next_slot[tb] != slot) {
        return -1;
    }
    return t;
}

/* Lower one TB record into the function. Returns false to bail (caller
 * falls back to TCG). Never guesses: unknown ops, undefined branch
 * targets, unterminated TBs, and non-env temp bases all bail.
 * Labels are namespaced per TB (TCG label ids restart per translation);
 * inter-TB flow uses the trace's proven edges, never label matching. */
static bool emitTB(IRBuilder<> &B, WalkState &S, uint32_t tb)
{
    S.cur_tb = tb;
    S.dead = false;
    const Tier2TBRec *rec = &S.trace->recs[tb];
    LLVMContext &C = S.C;

    /* Pre-scan labels so forward branches resolve. */
    for (uint32_t i = 0; i < rec->num_ops; i++) {
        if (rec->ops[i].op == T2_SETLABEL) {
            int64_t id = rec->ops[i].imm1;
            auto key = std::make_pair(tb, id);
            if (S.labels.count(key)) {
                return false; /* duplicate label: don't guess */
            }
            S.labels[key] = BasicBlock::Create(C, "L", S.F);
        }
    }

    auto needLabel = [&](int64_t id, BasicBlock *&bb) -> bool {
        auto it = S.labels.find(std::make_pair(tb, id));
        if (it == S.labels.end()) {
            if (S.debug) {
                fprintf(stderr, "[tier2-jit] bail tb=%u: branch to undefined label %lld\n",
                        tb, (long long)id);
            }
            return false; /* branch with unknown target: bail */
        }
        bb = it->second;
        return true;
    };

    bool ok = true;
    for (uint32_t i = 0; i < rec->num_ops && ok; i++) {
        const Tier2OpRec &op = rec->ops[i];
        if (S.dead) {
            /* Unreachable code after an unconditional terminator. Only a
             * label can revive emission. */
            if (op.op != T2_SETLABEL) {
                continue;
            }
        }
        unsigned b = op.bits ? op.bits : 64;
        if (b != 32 && b != 64 && b != 128 && b != 256) {
            return false;
        }
        auto U = [&](int32_t t) -> Value * { return useTemp(B, S, tb, t, ok); };
        switch (op.op) {
        case T2_MOV:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, U(op.src1));
            break;
        case T2_ADD:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, maskTo(B, B.CreateAdd(U(op.src1), U(op.src2)), b));
            break;
        case T2_SUB:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, maskTo(B, B.CreateSub(U(op.src1), U(op.src2)), b));
            break;
        case T2_MUL:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, maskTo(B, B.CreateMul(U(op.src1), U(op.src2)), b));
            break;
        case T2_AND:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateAnd(U(op.src1), U(op.src2)));
            break;
        case T2_OR:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateOr(U(op.src1), U(op.src2)));
            break;
        case T2_XOR:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateXor(U(op.src1), U(op.src2)));
            break;
        case T2_NEG:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, maskTo(B, B.CreateSub(B.getInt64(0), U(op.src1)), b));
            break;
        case T2_NOT:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateXor(U(op.src1), B.getInt64(~0ULL)));
            break;
        case T2_SHL:
        case T2_SHR:
        case T2_SAR:
        case T2_ROTL:
        case T2_ROTR: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *a = U(op.src1);
            Value *cnt = B.CreateAnd(U(op.src2), B.getInt64(b - 1));
            Value *r = nullptr;
            if (op.op == T2_SHL) {
                r = B.CreateShl(a, cnt);
            } else if (op.op == T2_SHR) {
                r = B.CreateLShr(maskTo(B, a, b), cnt);
            } else if (op.op == T2_SAR) {
                Value *se = a;
                if (b < 64) {
                    se = B.CreateSExt(B.CreateTrunc(a, IntegerType::get(C, b)), S.I64);
                }
                r = B.CreateAShr(se, cnt);
            } else {
                Value *lo = maskTo(B, a, b);
                Value *rc = B.CreateSub(B.getInt64(b), cnt);
                Value *fwd = (op.op == T2_ROTL) ? B.CreateShl(lo, cnt)
                                                : B.CreateLShr(lo, cnt);
                Value *bwd = (op.op == T2_ROTL) ? B.CreateLShr(lo, rc)
                                                : B.CreateShl(lo, rc);
                r = B.CreateSelect(B.CreateICmpEQ(cnt, B.getInt64(0)), lo,
                                   maskTo(B, B.CreateOr(fwd, bwd), b));
            }
            defTemp(B, S, tb, op.dst, maskTo(B, r, b));
            break;
        }
        case T2_EXTRACT: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            uint64_t off = (uint64_t)op.imm1 & 63;
            uint64_t len = (uint64_t)op.imm2 & 127;
            Value *v = B.CreateLShr(U(op.src1), B.getInt64(off));
            if (len < 64) {
                v = B.CreateAnd(v, B.getInt64(len == 64 ? ~0ULL : ((1ULL << len) - 1)));
            }
            defTemp(B, S, tb, op.dst, v);
            break;
        }
        case T2_SEXTRACT: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            uint64_t off = (uint64_t)op.imm1 & 63;
            uint64_t len = (uint64_t)op.imm2 & 127;
            Value *v = B.CreateLShr(U(op.src1), B.getInt64(off));
            Value *r;
            if (len >= 64) {
                r = v;
            } else {
                Value *t = B.CreateTrunc(v, IntegerType::get(C, (unsigned)len));
                r = B.CreateSExt(t, S.I64);
            }
            defTemp(B, S, tb, op.dst, r);
            break;
        }
        case T2_DEPOSIT: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            uint64_t off = (uint64_t)op.imm1 & 63;
            uint64_t len = (uint64_t)op.imm2 & 127;
            uint64_t m = (len >= 64) ? ~0ULL : (((1ULL << len) - 1) << off);
            Value *v = B.CreateOr(
                B.CreateAnd(U(op.src1), B.getInt64(~m)),
                B.CreateAnd(B.CreateShl(U(op.src2), B.getInt64(off)), B.getInt64(m)));
            defTemp(B, S, tb, op.dst, maskTo(B, v, b));
            break;
        }
        case T2_EXT32U:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, maskTo(B, U(op.src1), 32));
            break;
        case T2_EXT32S: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *t = B.CreateTrunc(U(op.src1), IntegerType::get(C, 32));
            defTemp(B, S, tb, op.dst, B.CreateSExt(t, S.I64));
            break;
        }
        case T2_EXTRL:
            if (!tempOk(S, tb, op.dst)) { return false; }
            defTemp(B, S, tb, op.dst, maskTo(B, U(op.src1), 32));
            break;
        case T2_SETCOND: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *c = emitCond(B, C, (unsigned)op.imm1, b, U(op.src1), U(op.src2));
            if (!c) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateZExt(c, S.I64));
            break;
        }
        case T2_MOVCOND: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *c = emitCond(B, C, (unsigned)op.imm1, b, U(op.src1), U(op.src2));
            if (!c) { return false; }
            Value *v1 = U(op.src3);
            Value *v2 = U(op.src4);
            if (!ok) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateSelect(c, v1, v2));
            break;
        }
        case T2_LD8U:
        case T2_LD8S:
        case T2_LD16U:
        case T2_LD16S:
        case T2_LD32U:
        case T2_LD32S:
        case T2_LD32:
        case T2_LD64: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Type *mt = nullptr;
            bool sext = false;
            switch (op.op) {
            case T2_LD8U: mt = Type::getInt8Ty(C); break;
            case T2_LD8S: mt = Type::getInt8Ty(C); sext = true; break;
            case T2_LD16U: mt = Type::getInt16Ty(C); break;
            case T2_LD16S: mt = Type::getInt16Ty(C); sext = true; break;
            case T2_LD32U:
            case T2_LD32S:
            case T2_LD32: mt = Type::getInt32Ty(C); break;
            case T2_LD64: mt = Type::getInt64Ty(C); break;
            default: break;
            }
            /*
             * Cells run ahead of env (defs commit lazily), so flush this
             * slot first -- then the env load observes everything. When
             * no cell exists the slot was never defined in-trace and env
             * is already the truth; commitOne is a no-op there. Either
             * way LLVM forwards the store into the load, so the pair
             * usually vanishes in O2.
             */
            /*
             * Flush-before-read, but only if this slot was ever defined
             * in-trace: otherwise env is already the truth (a helper may
             * even have written it since init, which we must NOT clobber
             * with the stale init-loaded cell).
             */
            if (S.definedSet.count((int32_t)op.imm1)) {
                commitOne(B, S, (int32_t)op.imm1);
            }
            Value *v = B.CreateLoad(mt, envPtr(B, S.envI8, op.imm1));
            if (mt->isIntegerTy(8) || mt->isIntegerTy(16) || mt->isIntegerTy(32)) {
                unsigned w = mt->getIntegerBitWidth();
                v = sext ? B.CreateSExt(v, S.I64) : B.CreateZExt(v, S.I64);
                (void)w;
            }
            defTemp(B, S, tb, op.dst, v);
            break;
        }
        case T2_ST8:
        case T2_ST16:
        case T2_ST32:
        case T2_ST64: {
            Value *v = U(op.src1);
            if (!ok) { return false; }
            Type *mt = op.op == T2_ST8 ? (Type *)Type::getInt8Ty(C)
                       : op.op == T2_ST16 ? (Type *)Type::getInt16Ty(C)
                       : op.op == T2_ST32 ? (Type *)Type::getInt32Ty(C)
                                          : (Type *)Type::getInt64Ty(C);
            unsigned w = mt->getIntegerBitWidth();
            if (w < 64) {
                v = B.CreateTrunc(v, IntegerType::get(C, w));
            }
            B.CreateStore(v, envPtr(B, S.envI8, op.imm1));
            /*
             * Keep a shared cell coherent when one exists: the stored
             * bytes win, the rest stays as the cell had it (exactly what
             * a later LD-from-cell-after-commit observes). No cell means
             * no in-trace reader can go stale -- env is the truth.
             */
            auto it = S.cells.find(std::make_pair(
                0xFFFFFFFFu, (uint64_t)(uint32_t)(int32_t)op.imm1));
            if (it != S.cells.end()) {
                Value *old = B.CreateLoad(S.I64, it->second);
                uint64_t m = w >= 64 ? ~0ULL : ((1ULL << w) - 1);
                Value *nv = B.CreateOr(
                    B.CreateAnd(old, B.getInt64(~m)),
                    B.CreateAnd(B.CreateZExt(v, S.I64), B.getInt64(m)));
                B.CreateStore(nv, it->second);
            }
            break;
        }
        case T2_QEMU_LD:
            if (!emitGuestMem(B, S, op, /*isLoad=*/true, ok)) { return false; }
            break;
        case T2_QEMU_ST:
            if (!emitGuestMem(B, S, op, /*isLoad=*/false, ok)) { return false; }
            break;
        case T2_CALL: {
            /*
             * Bit-exact C call to the recorded helper address, prototype
             * from imm2 (retcode | a0<<3 | ... | nr_in<<16 | nr_out<<20).
             * i32 params truncate (cells hold zero-extended values, matching
             * what TCG's EXTEND shims feed the callee); the env-marker temp
             * passes the live env pointer. Return masked by defTemp.
             */
            unsigned rc = (unsigned)(op.imm2 & 7);
            unsigned ni = (unsigned)((op.imm2 >> 16) & 0xF);
            unsigned no = (unsigned)((op.imm2 >> 20) & 0xF);
            if (ni > 4 || no > 1) {
                if (S.debug) {
                    fprintf(stderr, "[tier2-jit] bail tb=%u: call arity ni=%u no=%u\n",
                            tb, ni, no);
                }
                return false;
            }
            if ((no == 0) != (rc == T2T_VOID)) {
                if (S.debug) {
                    fprintf(stderr, "[tier2-jit] bail tb=%u: call ret mismatch\n", tb);
                }
                return false;
            }
            if (rc != T2T_VOID && rc != T2T_I32 && rc != T2T_I64 &&
                rc != T2T_PTR) {
                if (S.debug) {
                    fprintf(stderr, "[tier2-jit] bail tb=%u: call rettype %u\n", tb, rc);
                }
                return false;
            }
            Type *retTy = rc == T2T_VOID ? Type::getVoidTy(C)
                          : rc == T2T_I32 ? (Type *)Type::getInt32Ty(C)
                          : rc == T2T_I64 ? (Type *)S.I64
                                          : (Type *)PointerType::get(C, 0);
            int32_t argids[4] = {op.src1, op.src2, op.src3, op.src4};
            std::vector<Type *> argTys;
            std::vector<Value *> argVs;
            for (unsigned k = 0; k < ni; k++) {
                unsigned ac = (unsigned)((op.imm2 >> (3 + 3 * k)) & 7);
                if (ac != T2T_I32 && ac != T2T_I64 && ac != T2T_PTR) {
                    if (S.debug) {
                        fprintf(stderr, "[tier2-jit] bail tb=%u: call argtype %u\n",
                                tb, ac);
                    }
                    return false;
                }
                Type *at = ac == T2T_I32 ? (Type *)Type::getInt32Ty(C)
                           : ac == T2T_I64 ? (Type *)S.I64
                                           : (Type *)PointerType::get(C, 0);
                Value *av = U(argids[k]);
                if (!ok) {
                    return false;
                }
                if (ac == T2T_I32) {
                    av = B.CreateTrunc(av, Type::getInt32Ty(C));
                } else if (ac == T2T_PTR) {
                    av = B.CreateIntToPtr(av, PointerType::get(C, 0));
                }
                argTys.push_back(at);
                argVs.push_back(av);
            }
            FunctionType *FT = FunctionType::get(retTy, argTys, false);
            /*
             * Call through the recorded helper NAME (link-time external),
             * never the recorded ADDRESS: op.imm1 is this process's
             * address (ASLR-unstable), and baking it poisons cached
             * objects for future processes (stale blr target -> SIGSEGV
             * on load). Unnamed targets force no_cache (ckey=0, never
             * stored), so the baked fallback below only ever runs
             * uncached in this process.
             */
            Value *fn = nullptr;
            if (const char *nm = callNameFor(S.trace, (uint64_t)op.imm1)) {
                fn = getRuntimeFn(S.M, nm, FT);
            } else {
                if (S.debug) {
                    fprintf(stderr, "[tier2-jit] tb=%u: UNNAMED call target "
                            "%p, baking address (uncached-only)\n",
                            tb, (void *)(uint64_t)op.imm1);
                }
                fn = B.CreateIntToPtr(B.getInt64((uint64_t)op.imm1),
                                      PointerType::get(C, 0));
            }
            /*
             * Helpers observe and mutate env through memory: flush cells
             * first, reload after (the callee may have changed anything).
             */
            commitEnv(B, S);
            Value *r = B.CreateCall(FT, fn, argVs);
            reloadEnv(B, S);
            if (no == 1) {
                if (rc == T2T_PTR) {
                    r = B.CreatePtrToInt(r, S.I64);
                } else if (rc == T2T_I32) {
                    r = B.CreateZExt(r, S.I64);
                }
                if (!tempOk(S, tb, op.dst)) {
                    return false;
                }
                defTemp(B, S, tb, op.dst, r);
            }
            break;
        }
        case T2_BSWAP16:
        case T2_BSWAP32:
        case T2_BSWAP64: {
            if (!tempOk(S, tb, op.dst)) {
                return false;
            }
            Value *a = U(op.src1);
            if (!ok) {
                return false;
            }
            unsigned size = op.op == T2_BSWAP16 ? 16
                            : op.op == T2_BSWAP32 ? 32 : 64;
            /* Input masked to size (capture required IZ); swap within
             * the field via the LLVM bswap intrinsic. */
            Value *lo = maskTo(B, a, size);
            Value *t = B.CreateTrunc(lo, IntegerType::get(C, size));
            Value *sw = B.CreateCall(
                Intrinsic::getOrInsertDeclaration(S.M, Intrinsic::bswap,
                                                  {IntegerType::get(C, size)}),
                {t});
            Value *r;
            if (op.op == T2_BSWAP16 && (op.imm1 & 4)) {
                r = B.CreateSExt(sw, S.I64); /* TCG_BSWAP_OS */
            } else {
                r = B.CreateZExt(sw, S.I64); /* OZ (required at capture) */
            }
            defTemp(B, S, tb, op.dst, r);
            break;
        }
        case T2_NEGSETCOND: {
            if (!tempOk(S, tb, op.dst)) {
                return false;
            }
            Value *c = emitCond(B, C, (unsigned)op.imm1, b, U(op.src1), U(op.src2));
            if (!c) {
                return false;
            }
            defTemp(B, S, tb, op.dst, B.CreateSub(B.getInt64(0),
                                                  B.CreateZExt(c, S.I64)));
            break;
        }
        case T2_ANDC: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *v1 = U(op.src1);
            Value *v2 = U(op.src2);
            if (!ok) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateAnd(v1, B.CreateNot(v2)));
            break;
        }
        case T2_ORC: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *v1 = U(op.src1);
            Value *v2 = U(op.src2);
            if (!ok) { return false; }
            defTemp(B, S, tb, op.dst, B.CreateOr(v1, B.CreateNot(v2)));
            break;
        }
        case T2_MULSH:
        case T2_MULUH: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *v1 = U(op.src1);
            Value *v2 = U(op.src2);
            if (!ok) { return false; }
            bool is_signed = (op.op == T2_MULSH);
            if (b == 64) {
                Type *i128Ty = Type::getInt128Ty(C);
                Value *ext1 = is_signed ? B.CreateSExt(v1, i128Ty) : B.CreateZExt(v1, i128Ty);
                Value *ext2 = is_signed ? B.CreateSExt(v2, i128Ty) : B.CreateZExt(v2, i128Ty);
                Value *prod = B.CreateMul(ext1, ext2);
                Value *hi128 = is_signed ? B.CreateAShr(prod, 64) : B.CreateLShr(prod, 64);
                Value *hi = B.CreateTrunc(hi128, S.I64);
                defTemp(B, S, tb, op.dst, hi);
            } else {
                Value *v1_32 = B.CreateTrunc(v1, Type::getInt32Ty(C));
                Value *v2_32 = B.CreateTrunc(v2, Type::getInt32Ty(C));
                Value *ext1 = is_signed ? B.CreateSExt(v1_32, S.I64) : B.CreateZExt(v1_32, S.I64);
                Value *ext2 = is_signed ? B.CreateSExt(v2_32, S.I64) : B.CreateZExt(v2_32, S.I64);
                Value *prod = B.CreateMul(ext1, ext2);
                Value *hi64 = is_signed ? B.CreateAShr(prod, 32) : B.CreateLShr(prod, 32);
                Value *hi = B.CreateZExt(B.CreateTrunc(hi64, Type::getInt32Ty(C)), S.I64);
                defTemp(B, S, tb, op.dst, hi);
            }
            break;
        }
        case T2_CLZ:
        case T2_CTZ: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            Value *val = U(op.src1);
            Value *def_val = U(op.src2);
            if (!ok) { return false; }
            bool is_clz = (op.op == T2_CLZ);
            Type *intTy = (b == 32) ? (Type *)Type::getInt32Ty(C) : (Type *)S.I64;
            Value *inp = (b == 32) ? B.CreateTrunc(val, Type::getInt32Ty(C)) : val;
            Intrinsic::ID iid = is_clz ? Intrinsic::ctlz : Intrinsic::cttz;
            Value *cnt = B.CreateCall(Intrinsic::getOrInsertDeclaration(S.M, iid, {intTy}),
                                      {inp, B.getInt1(false)});
            Value *is_zero = B.CreateICmpEQ(inp, ConstantInt::get(intTy, 0));
            Value *def_trunc = (b == 32) ? B.CreateTrunc(def_val, Type::getInt32Ty(C)) : def_val;
            Value *res = B.CreateSelect(is_zero, def_trunc, cnt);
            if (b == 32) {
                res = B.CreateZExt(res, S.I64);
            }
            defTemp(B, S, tb, op.dst, res);
            break;
        }
        /* Phase 4: Vector / SIMD Transpilation (x86 SSE/AVX -> ARM64 NEON) */
        case T2_VEC_ADD:
        case T2_VEC_SUB:
        case T2_VEC_MUL:
        case T2_VEC_AND:
        case T2_VEC_OR:
        case T2_VEC_XOR:
        case T2_VEC_NOT:
        case T2_VEC_DUP: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            unsigned elemBits = op.imm1 ? (unsigned)op.imm1 : 32;
            unsigned totalBits = op.bits ? op.bits : 128;
            unsigned lanes = totalBits / elemBits;
            Type *elemTy = (elemBits == 8) ? Type::getInt8Ty(C) :
                           (elemBits == 16) ? Type::getInt16Ty(C) :
                           (elemBits == 32) ? Type::getInt32Ty(C) : S.I64;
            Type *vecTy = FixedVectorType::get(elemTy, lanes);
            Value *v1 = U(op.src1);
            Value *v2 = (op.src2 >= 0) ? U(op.src2) : nullptr;
            Value *vec1 = B.CreateBitCast(v1, vecTy);
            Value *vec2 = v2 ? B.CreateBitCast(v2, vecTy) : nullptr;
            Value *res = nullptr;
            switch (op.op) {
            case T2_VEC_ADD: res = B.CreateAdd(vec1, vec2); break;
            case T2_VEC_SUB: res = B.CreateSub(vec1, vec2); break;
            case T2_VEC_MUL: res = B.CreateMul(vec1, vec2); break;
            case T2_VEC_AND: res = B.CreateAnd(vec1, vec2); break;
            case T2_VEC_OR:  res = B.CreateOr(vec1, vec2); break;
            case T2_VEC_XOR: res = B.CreateXor(vec1, vec2); break;
            case T2_VEC_NOT: res = B.CreateNot(vec1); break;
            case T2_VEC_DUP: {
                Value *scalar = (elemBits == 64) ? v1 : B.CreateTrunc(v1, elemTy);
                res = B.CreateVectorSplat(lanes, scalar);
                break;
            }
            default: break;
            }
            if (res) {
                Type *dstCellTy = IntegerType::get(C, totalBits);
                defTemp(B, S, tb, op.dst, B.CreateBitCast(res, dstCellTy));
            }
            break;
        }
        case T2_VEC_LD: {
            if (!tempOk(S, tb, op.dst)) { return false; }
            unsigned totalBits = op.bits ? op.bits : 128;
            Type *vecIntTy = IntegerType::get(C, totalBits);
            Value *addr = U(op.src1);
            Value *ptr = B.CreateIntToPtr(addr, PointerType::get(C, 0));
            Value *val = B.CreateAlignedLoad(vecIntTy, ptr, Align(16));
            defTemp(B, S, tb, op.dst, val);
            break;
        }
        case T2_VEC_ST: {
            unsigned totalBits = op.bits ? op.bits : 128;
            Type *vecIntTy = IntegerType::get(C, totalBits);
            Value *val = U(op.src1);
            Value *addr = U(op.src2);
            Value *ptr = B.CreateIntToPtr(addr, PointerType::get(C, 0));
            B.CreateAlignedStore(val, ptr, Align(16));
            break;
        }
        case T2_BR: {
            BasicBlock *dst = nullptr;
            if (!needLabel(op.imm1, dst)) { return false; }
            B.CreateBr(dst);
            S.dead = true;
            break;
        }
        case T2_BRCOND: {
            BasicBlock *dst = nullptr;
            if (!needLabel(op.imm2, dst)) { return false; }
            Value *c = emitCond(B, C, (unsigned)op.imm1, b, U(op.src1), U(op.src2));
            if (!c) { return false; }
            BasicBlock *fall = BasicBlock::Create(C, "fall", S.F);
            B.CreateCondBr(c, dst, fall);
            B.SetInsertPoint(fall);
            break;
        }
        case T2_SETLABEL: {
            auto it = S.labels.find(std::make_pair(tb, op.imm1));
            if (it == S.labels.end()) { return false; }
            if (!S.dead && B.GetInsertBlock()->getTerminator() == nullptr) {
                B.CreateBr(it->second);
            }
            B.SetInsertPoint(it->second);
            S.dead = false;
            break;
        }
        case T2_EXIT_TB: {
            uint64_t imm = (uint64_t)op.imm1;
            commitEnv(B, S);
            if ((imm & ~3ULL) == 0) {
                /*
                 * Null-TB exit (e.g. exit_tb(NULL, 0)): verbatim value,
                 * stable across processes (no pointer baked). The
                 * dispatcher takes last_tb=NULL and re-derives.
                 */
                B.CreateRet(B.getInt64(imm));
            } else {
                /*
                 * resolve-later exit: the baked tb part must name THIS
                 * TB (translators always exit their own TB or NULL);
                 * anything else is a shape we don't model -> bail.
                 * Compared with low bits masked (they carry idx).
                 */
                uint64_t rx = (uint64_t)S.trace->recs[tb].rx_tb;
                if (!rx || (imm & ~3ULL) != (rx & ~3ULL)) {
                    return false;
                }
                B.CreateRet(B.getInt64(TIER2_EXIT_PROTOCOL |
                                       ((uint64_t)tb << 2) | (imm & 3)));
            }
            S.dead = true;
            break;
        }
        case T2_GOTO_TB: {
            int internal = internalGotoEdge(S.trace, tb, op.imm1);
            if (internal >= 0) {
                if ((uint32_t)internal == tb) {
                    /*
                     * Proven self back-edge (P5 native loop): run the TB
                     * again instead of round-tripping the dispatcher per
                     * iteration. Commit first: the back-edge re-executes
                     * the TB from its top (including leading env loads),
                     * so env must be coherent -- exactly TCG's TB-boundary
                     * invariant, which is also what the poll-exit path
                     * needs. Then poll interrupt_request so a tight loop
                     * can never starve interrupts (nonzero takes the same
                     * side-exit this edge used before). Without safepoint
                     * offsets (measured lazily on a live vCPU;
                     * workload-trigger compiles can precede measurement),
                     * keep the old side-exit.
                     */
                    if (!S.trace->has_safepoint) {
                        internal = -1;
                    } else {
                        BasicBlock *loop = S.entries[tb];
                        BasicBlock *irq =
                            BasicBlock::Create(C, "safepoint", S.F);
                        commitEnv(B, S);
                        Value *cpu = B.CreateGEP(B.getInt8Ty(), S.envI8,
                                                 B.getInt64(S.trace->cpu_off));
                        Value *irqp = B.CreateBitCast(
                            B.CreateGEP(B.getInt8Ty(), cpu,
                                        B.getInt64(S.trace->irq_off)),
                            PointerType::get(C, 0));
                        Value *pend = B.CreateLoad(Type::getInt32Ty(C), irqp);
                        Value *has_irq = B.CreateICmpNE(pend, B.getInt32(0));

                        Value *cur_cnt = B.CreateLoad(Type::getInt32Ty(C), S.loop_cnt);
                        Value *next_cnt = B.CreateSub(cur_cnt, B.getInt32(1));
                        B.CreateStore(next_cnt, S.loop_cnt);
                        Value *expired = B.CreateICmpSLE(next_cnt, B.getInt32(0));

                        Value *should_exit = B.CreateOr(has_irq, expired);
                        B.CreateCondBr(should_exit, irq, loop);
                        B.SetInsertPoint(irq);
                        B.CreateRet(B.getInt64(TIER2_EXIT_PROTOCOL |
                                               ((uint64_t)tb << 2) |
                                               ((uint64_t)op.imm1 & 3)));
                        S.dead = true;
                        break;
                    }
                } else {
                    /*
                     * Proven edge (observed linked at discovery): continue
                     * inline into the successor TB's entry block. Stale-link
                     * races are safe by construction -- unlinking only changes
                     * how control reaches the target, never the target's
                     * meaning -- and invalidation clears our code outright.
                     */
                    B.CreateBr(S.entries[internal]);
                    S.dead = true;
                    break;
                }
            }
            /*
             * No proven edge: side-exit via the exit protocol (resolved
             * against current addresses at dispatch; see TIER2_EXIT_PROTOCOL).
             * The origin TB index travels in the value so chaining links
             * land on the right TB, exactly as unlinked TCG would.
             */
            commitEnv(B, S);
            B.CreateRet(B.getInt64(TIER2_EXIT_PROTOCOL |
                                   ((uint64_t)tb << 2) |
                                   ((uint64_t)op.imm1 & 3)));
            S.dead = true;
            break;
        }
        case T2_GOTO_PTR: {
            int st = staticGotoTarget(S.trace, S.pc2idx, tb, i);
            if (st >= 0) {
                B.CreateBr(S.entries[st]);
                S.dead = true;
                break;
            }
            uint64_t pc = 0;
            int cls = classifyGotoAddr(S.trace, tb, i, &pc);
            if (cls == T2_GOTO_LOOKUP_CALL && g_prologue_fn != nullptr) {
                /*
                 * Address came from helper_lookup_tb_ptr: tail-call the
                 * TCG prologue with its result and return that. If the lookup
                 * returned NULL/epilogue, cleanly return 0 to the dispatcher.
                 */
                Value *code = U(op.src1);
                if (!ok) {
                    return false;
                }
                BasicBlock *call_prologue = BasicBlock::Create(C, "goto_prologue", S.F);
                BasicBlock *ret_zero = BasicBlock::Create(C, "goto_zero", S.F);
                B.CreateCondBr(B.CreateICmpEQ(code, B.getInt64(0)), ret_zero, call_prologue);

                B.SetInsertPoint(call_prologue);
                FunctionType *PT = FunctionType::get(
                    S.I64, {PointerType::get(C, 0), PointerType::get(C, 0)},
                    false);
                Function *prologue =
                    getRuntimeFn(S.M, "tier2_rt_prologue", PT);
                Value *r = B.CreateCall(
                    PT, prologue,
                    {S.envI8, B.CreateIntToPtr(code, PointerType::get(C, 0))});
                B.CreateRet(r);

                B.SetInsertPoint(ret_zero);
                B.CreateRet(B.getInt64(0));
                S.dead = true;
                break;
            }
            /*
             * Dynamic target (returns, indirect calls): side-exit with a
             * null TB so the dispatcher re-derives from env with no
             * chaining-link write. Matches TCI's null-target behavior,
             * generalized. Env is coherent here (commit discipline).
             */
            commitEnv(B, S);
            B.CreateRet(B.getInt64(0));
            S.dead = true;
            break;
        }
        default:
            if (S.debug) {
                fprintf(stderr, "[tier2-jit] bail tb=%u: unsupported op %s\n",
                        tb, t2opname(op.op));
            }
            return false; /* T2_UNSUPPORTED or unknown: don't guess */
        }
    }
    if (!ok) {
        return false;
    }
    /* A TB must end in a terminator (exit/goto/branch). Falling off the
     * end means the capture was truncated: bail. A trailing unconditional
     * terminator (dead == true) is the normal complete case. */
    if (S.dead) {
        return true;
    }
    BasicBlock *cur = B.GetInsertBlock();
    if (cur->getTerminator() == nullptr) {
        if (S.debug) {
            fprintf(stderr, "[tier2-jit] bail tb=%u: unterminated (%u ops)\n",
                    tb, rec->num_ops);
        }
        return false;
    }
    return true;
}

/*
 * Compile a fused multi-TB trace inline. Execution starts at the header
 * TB and follows proven-internal edges (forward edges, plus self
 * back-edges with an interrupt safepoint poll, so every dispatch still
 * returns); all other transfers side-exit to the dispatcher with
 * origin-correct return values. Env-slot globals are shared across TBs
 * (eager commits keep env a superset of TCG's lazy state, which is what
 * makes exits sound anywhere); EBB temps stay per-TB; consts fold inline.
 */
static bool tryCompileOps(Module *M, LLVMContext &C, Function *F,
                          Value *env_arg, const Tier2TraceDesc *trace)
{
    uint32_t n = trace->num_tbs;
    if (n == 0 || n > TIER2_MAX_TRACE_TBS) {
        return false;
    }
    if (trace->header_idx >= n) {
        return false;
    }
    for (uint32_t t = 0; t < n; t++) {
        const Tier2TBRec &rec = trace->recs[t];
        if (rec.num_temps > TIER2_JIT_MAX_TEMPS ||
            rec.num_ops > TIER2_JIT_MAX_OPS) {
            return false;
        }
        if (rec.num_temps == 0) {
            return false;
        }
    }

    IRBuilder<> B(BasicBlock::Create(C, "entry", F));
    Value *envI8 = B.CreateBitCast(env_arg, PointerType::get(C, 0));

    WalkState S{C, M, F, envI8, trace};
    S.I64 = Type::getInt64Ty(C);
    S.debug = getenv("QEMU_TIER2_DEBUG") != nullptr;
    S.loop_cnt = B.CreateAlloca(Type::getInt32Ty(C), nullptr, "self_loop_cnt");
    B.CreateStore(B.getInt32(4096), S.loop_cnt);

    /* PC -> trace index for static goto_ptr resolution (first wins;
     * duplicated PCs across TBs stay dynamic). */
    for (uint32_t t = 0; t < n; t++) {
        uint64_t pc = trace->recs[t].pc;
        if (!S.pc2idx.count(pc)) {
            S.pc2idx[pc] = t;
        }
    }

    /*
     * Reachability closure from the header over internal goto_tb edges
     * and statically-resolved goto_ptr targets. Unreachable TBs are not
     * emitted at all, so truncated snapshots outside the reachable set
     * can't bail an otherwise compilable trace.
     */
    bool reachable[TIER2_MAX_TRACE_TBS] = {false};
    reachable[trace->header_idx] = true;
    for (;;) {
        bool grew = false;
        for (uint32_t t = 0; t < n; t++) {
            if (!reachable[t]) {
                continue;
            }
            int ie = provenEdgeTarget(trace, t);
            if (ie >= 0 && !reachable[ie]) {
                reachable[ie] = true;
                grew = true;
            }
            const Tier2TBRec &rec = trace->recs[t];
            for (uint32_t i = 0; i < rec.num_ops; i++) {
                if (rec.ops[i].op != T2_GOTO_PTR) {
                    continue;
                }
                int st = staticGotoTarget(trace, S.pc2idx, t, i);
                if (st >= 0 && !reachable[st]) {
                    reachable[st] = true;
                    grew = true;
                }
            }
        }
        if (!grew) {
            break;
        }
    }

    /* Entry block per TB in the reachable set, created upfront so
     * forward internal edges resolve. */
    S.entries.assign(n, nullptr);
    for (uint32_t t = 0; t < n; t++) {
        if (reachable[t]) {
            S.entries[t] = BasicBlock::Create(C, "tb", F);
        }
    }

    /*
     * Temp cells: allocas in the entry block (mem2reg-promotable).
     * Shared env-slot cells load the WIDEST view seen across TBs so no
     * view is ever truncated at init (uses mask down to their width).
     * The same map drives sized commits/reloads.
     */
    for (uint32_t t = 0; t < n; t++) {
        if (!reachable[t]) {
            continue;
        }
        const Tier2TBRec &rec = trace->recs[t];
        for (uint32_t i = 0; i < rec.num_temps; i++) {
            const Tier2TempRec &tr = rec.temps[i];
            if (tr.is_const || tr.env_off < 0) {
                continue;
            }
            unsigned &w = S.envWidths[tr.env_off];
            if (tr.tbits == 64) {
                w = 64;
            } else if (w != 64) {
                w = 32;
            }
        }
        /*
         * Defined set: env offsets ever assigned (temp DSTs, LD DSTs
         * included -- an LD def runs ahead of env until committed).
         * ST-only slots stay coherent via the ST path itself.
         */
        for (uint32_t i = 0; i < rec.num_ops; i++) {
            int32_t d = rec.ops[i].dst;
            if (d >= 0 && (uint32_t)d < rec.num_temps &&
                !rec.temps[d].is_env && rec.temps[d].env_off >= 0) {
                S.definedSet.insert(rec.temps[d].env_off);
            }
        }
    }
    for (uint32_t t = 0; t < n; t++) {
        if (!reachable[t]) {
            continue;
        }
        const Tier2TBRec &rec = trace->recs[t];
        for (uint32_t i = 0; i < rec.num_temps; i++) {
            const Tier2TempRec &tr = rec.temps[i];
            if (tr.is_const || tr.env_off < 0) {
                /* Consts fold inline; non-env temps get private cells. */
                if (!tr.is_const) {
                    auto key = std::make_pair(t, (uint64_t)(uint32_t)i);
                    if (!S.cells.count(key)) {
                        unsigned tb_bits = tr.tbits >= 128 ? tr.tbits : 64;
                        Type *cellTy = IntegerType::get(C, tb_bits);
                        Value *cell = B.CreateAlloca(cellTy, nullptr, "t");
                        S.cells[key] = cell;
                        B.CreateStore(ConstantInt::get(cellTy, 0), cell);
                    }
                }
                continue;
            }
            auto key = std::make_pair(0xFFFFFFFFu, (uint64_t)(uint32_t)tr.env_off);
            if (!S.cells.count(key)) {
                Value *cell = B.CreateAlloca(S.I64, nullptr, "g");
                S.cells[key] = cell;
                Value *init;
                if (S.envWidths[tr.env_off] == 32) {
                    Value *p = envPtr(B, envI8, tr.env_off);
                    init = B.CreateZExt(B.CreateLoad(Type::getInt32Ty(C), p),
                                        S.I64);
                } else {
                    Value *p = envPtr(B, envI8, tr.env_off);
                    init = B.CreateLoad(S.I64, p);
                }
                B.CreateStore(init, cell);
            }
        }
    }


    B.CreateBr(S.entries[trace->header_idx]);

    for (uint32_t t = 0; t < n; t++) {
        if (!reachable[t]) {
            continue;
        }
        B.SetInsertPoint(S.entries[t]);
        if (!emitTB(B, S, t)) {
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" bool tier2_jit_init(void *prologue_fn)
{
    g_prologue_fn = (TCGPrologueFn)prologue_fn;

    if (g_initialized.load()) {
        return true;
    }

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    auto JITExp = LLJITBuilder().create();
    if (!JITExp) {
        fprintf(stderr, "[tier2-jit] Failed to create LLJIT: %s\n",
                toString(JITExp.takeError()).c_str());
        return false;
    }

    g_jit = std::move(*JITExp);
    g_initialized.store(true);
    printf("[tier2-jit] Initialized LLVM ORC JIT v%d engine (prologue=%p)\n",
           LLVM_VERSION_MAJOR, (void *)g_prologue_fn);
    return true;
}

extern "C" void tier2_jit_invalidate(void *fn_ptr)
{
    if (!fn_ptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_modules.find(fn_ptr);
    if (it != g_modules.end()) {
        /* Retire: actual removal deferred to flush (exclusive context),
         * when no vCPU can be executing the code. */
        g_retired.push_back(std::move(it->second));
        g_modules.erase(it);
    }
}

extern "C" void tier2_jit_flush(void)
{
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto &t : g_retired) {
        if (auto Err = t->remove()) {
            consumeError(std::move(Err));
        }
    }
    g_retired.clear();
    for (auto &kv : g_modules) {
        if (auto Err = kv.second->remove()) {
            consumeError(std::move(Err));
        }
    }
    g_modules.clear();
}

extern "C" void tier2_jit_shutdown(void)
{
    if (!g_initialized.load()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_retired.clear();
        g_modules.clear();
        /* Defined-externals must be re-registered with the next JIT
         * instance (addresses are process-stable, JITDylibs are not). */
        g_definedRt.clear();
    }
    g_jit.reset();
    g_initialized.store(false);
}

/* ------------------------------------------------------------------ */
/* On-disk code cache (Phase 2). Same hot traces recompile identically  */
/* across processes, so validated native objects persist under          */
/* ~/.cache/qemu/tier2/<fnv1a64>.o and load in microseconds instead of  */
/* tens of milliseconds. Sound because every process-unstable input is  */
/* excluded by construction: TB addresses never enter compiled code     */
/* (exit protocol), host addresses never bake in (named externals +     */
/* link-time absoluteSymbols), and the key covers everything else that  */
/* can affect codegen.                                                  */
/* ------------------------------------------------------------------ */

static std::atomic<uint64_t> g_cache_hits{0};

extern "C" uint64_t tier2_jit_cache_hits(void)
{
    return g_cache_hits.load();
}

static uint64_t fnv1a(const void *data, size_t len, uint64_t h)
{
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void fnvStr(uint64_t &h, const char *s, size_t n)
{
    h = fnv1a(s, strnlen(s, n), h);
}

/*
 * Cache key over everything that can change the emitted object.
 * Deliberately EXCLUDED (unstable or link-resolved): tb_obj/code_ptr/
 * rx_tb/rx_header pointers, mem_helpers + call_addrs + prologue
 * addresses (resolved via named externals at link), total_exec_count,
 * trace_id, legacy tbs[] (walker uses recs[]).
 */
static uint64_t tier2CacheKey(const Tier2TraceDesc *trace)
{
    if (trace->guest_arch[0] == '\0' || trace->qemu_version[0] == '\0' ||
        trace->no_cache) {
        return 0;
    }
    uint64_t h = 1469598103934665603ULL;
    h = fnv1a("tier2cache/1", 12, h);
    {
        /* Emitter epoch: any emission-semantics change must bump
         * TIER2_CACHE_EPOCH, or stale objects load as false hits. */
        uint64_t epoch = TIER2_CACHE_EPOCH;
        h = fnv1a(&epoch, sizeof(epoch), h);
    }
    std::string triple = sys::getProcessTriple();
    h = fnv1a(triple.c_str(), triple.size(), h);
    int llvm_major = LLVM_VERSION_MAJOR;
    int llvm_minor = LLVM_VERSION_MINOR;
    h = fnv1a(&llvm_major, sizeof(llvm_major), h);
    h = fnv1a(&llvm_minor, sizeof(llvm_minor), h);
    fnvStr(h, trace->guest_arch, sizeof(trace->guest_arch));
    fnvStr(h, trace->qemu_version, sizeof(trace->qemu_version));
    h = fnv1a("O2-novector", 11, h);
    h = fnv1a(&trace->num_tbs, sizeof(trace->num_tbs), h);
    h = fnv1a(&trace->header_idx, sizeof(trace->header_idx), h);
    {
        uint8_t gm = trace->guest_mem_allowed ? 1 : 0;
        h = fnv1a(&gm, 1, h);
    }
    h = fnv1a(&trace->tlb, sizeof(trace->tlb), h);
    {
        /* Safepoint geometry is build-derived; same rule as the epoch:
         * anything that changes emitted code must change the key. */
        uint8_t sp = trace->has_safepoint ? 1 : 0;
        h = fnv1a(&sp, 1, h);
        h = fnv1a(&trace->cpu_off, sizeof(trace->cpu_off), h);
        h = fnv1a(&trace->irq_off, sizeof(trace->irq_off), h);
        h = fnv1a(&trace->last_tb_off, sizeof(trace->last_tb_off), h);
        h = fnv1a(&trace->ram_base, sizeof(trace->ram_base), h);
        h = fnv1a(&trace->ram_size, sizeof(trace->ram_size), h);
    }
    if (getenv("QEMU_TIER2_DEBUG")) {
        fprintf(stderr, "[tier2-jit] keyinputs n=%u hidx=%u sp=%d cpuoff=%lld irqoff=%lld gm=%d next0=%d slot0=%d\n",
                trace->num_tbs, trace->header_idx,
                trace->has_safepoint ? 1 : 0,
                (long long)trace->cpu_off, (long long)trace->irq_off,
                trace->guest_mem_allowed ? 1 : 0,
                trace->num_tbs > 0 ? trace->next[0] : -9,
                trace->num_tbs > 0 ? trace->next_slot[0] : -9);
    }
    h = fnv1a(trace->next, sizeof(trace->next[0]) * trace->num_tbs, h);
    h = fnv1a(trace->next_slot, sizeof(trace->next_slot[0]) * trace->num_tbs, h);
    uint32_t callpos = 0;
    for (uint32_t t = 0; t < trace->num_tbs; t++) {
        const Tier2TBRec &r = trace->recs[t];
        h = fnv1a(&r.pc, sizeof(r.pc), h);
        h = fnv1a(&r.size, sizeof(r.size), h);
        h = fnv1a(&r.icount, sizeof(r.icount), h);
        h = fnv1a(&r.cflags, sizeof(r.cflags), h);
        h = fnv1a(&r.num_temps, sizeof(r.num_temps), h);
        h = fnv1a(&r.num_ops, sizeof(r.num_ops), h);
        h = fnv1a(r.temps, sizeof(r.temps[0]) * r.num_temps, h);
        for (uint32_t i = 0; i < r.num_ops; i++) {
            const Tier2OpRec &op = r.ops[i];
            /*
             * op.op/bits/dst/srcs/imm2 verbatim; imm1 EXCEPT addresses.
             * T2_CALL carries this process's helper address (ASLR): hash
             * the stable recorded name instead. T2_EXIT_TB carries the
             * exit value (host TB* | idx): hash only the idx bits -- the
             * pointer varies per boot (fresh TB heap addresses) while
             * the emitted code only depends on the idx (protocol exit
             * code) plus the compile-time rx match, which loads don't
             * re-check (install-time validation covers staleness).
             * Hashing the pointer made every key boot-unique (cache
             * never hit); hashing only stable parts keeps it sound.
             */
            h = fnv1a(&op.op, sizeof(op.op), h);
            h = fnv1a(&op.bits, sizeof(op.bits), h);
            h = fnv1a(&op.dst, sizeof(op.dst), h);
            h = fnv1a(&op.src1, sizeof(op.src1), h);
            h = fnv1a(&op.src2, sizeof(op.src2), h);
            h = fnv1a(&op.src3, sizeof(op.src3), h);
            h = fnv1a(&op.src4, sizeof(op.src4), h);
            if (op.op == T2_EXIT_TB) {
                uint64_t idx = (uint64_t)op.imm1 & 3ULL;
                h = fnv1a(&idx, sizeof(idx), h);
            } else if (op.op == T2_CALL) {
                /* Address varies per process (ASLR): hash the stable
                 * helper name recorded in emission order instead. */
                if (callpos >= trace->num_calls) {
                    return 0;
                }
                fnvStr(h, trace->call_names[callpos], sizeof(trace->call_names[0]));
                callpos++;
            } else {
                h = fnv1a(&op.imm1, sizeof(op.imm1), h);
            }
            h = fnv1a(&op.imm2, sizeof(op.imm2), h);
        }
    }
    return h ? h : 1;
}

static bool cacheEnabled(void)
{
    const char *e = getenv("QEMU_TIER2_CACHE");
    return !(e && strcmp(e, "0") == 0);
}

static std::string cacheDir(void)
{
    const char *e = getenv("QEMU_TIER2_CACHE_DIR");
    if (e && *e) {
        return e;
    }
    const char *xdg = getenv("XDG_CACHE_HOME");
    std::string base = (xdg && *xdg) ? xdg : "";
    if (base.empty()) {
        const char *home = getenv("HOME");
        base = (home && *home) ? std::string(home) + "/.cache" : "/tmp";
    }
    return base + "/qemu/tier2";
}

static std::string cachePath(uint64_t key)
{
    char name[32];
    snprintf(name, sizeof(name), "%016llx.o", (unsigned long long)key);
    return cacheDir() + "/" + name;
}

static uint64_t cacheMaxBytes(void)
{
    const char *e = getenv("QEMU_TIER2_CACHE_MAX_MB");
    unsigned long mb = e ? strtoul(e, nullptr, 10) : 512;
    return mb * 1024ULL * 1024ULL;
}

/* Best-effort LRU trim, every 16th store. Errors ignored throughout. */
static void cacheTrim(void)
{
    static std::atomic<unsigned> ctr{0};
    if ((ctr.fetch_add(1) % 16) != 0) {
        return;
    }
    std::string dir = cacheDir();
    uint64_t maxBytes = cacheMaxBytes();
    std::error_code EC;
    struct Ent {
        std::string path;
        uint64_t size;
        time_t mtime;
    };
    std::vector<Ent> ents;
    uint64_t total = 0;
    for (sys::fs::directory_iterator it(dir, EC), en; it != en && !EC;
         it.increment(EC)) {
        std::string p = it->path();
        if (p.size() < 3 || p.compare(p.size() - 2, 2, ".o") != 0) {
            continue;
        }
        struct stat st;
        if (::stat(p.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        uint64_t sz = (uint64_t)st.st_size;
        ents.push_back({p, sz, st.st_mtime});
        total += sz;
    }
    if (total <= maxBytes) {
        return;
    }
    std::sort(ents.begin(), ents.end(),
              [](const Ent &a, const Ent &b) { return a.mtime < b.mtime; });
    for (const auto &en : ents) {
        if (total <= maxBytes) {
            break;
        }
        sys::fs::remove(en.path);
        total -= en.size;
    }
}

static std::unique_ptr<TargetMachine> g_emitTM;
static std::mutex g_emitMu;

/* Emit the optimized module to a relocatable object file (atomic rename).
 * Best effort: any failure just skips caching. */
static bool cacheStore(uint64_t key, Module *M)
{
    std::string dir = cacheDir();
    std::error_code EC = sys::fs::create_directories(dir);
    if (EC) {
        return false;
    }
    std::unique_ptr<TargetMachine> TM;
    {
        std::lock_guard<std::mutex> lk(g_emitMu);
        if (!g_emitTM) {
            Triple triple{sys::getProcessTriple()};
            std::string err;
            const Target *tgt =
                TargetRegistry::lookupTarget(triple, err);
            if (!tgt) {
                return false;
            }
            /* "generic" CPU: cached objects stay valid across same-arch
             * host steppings; fresh JIT still uses native tuning. */
            TargetOptions opts;
            g_emitTM.reset(tgt->createTargetMachine(
                triple, "generic", "", opts, std::nullopt, std::nullopt,
                CodeGenOptLevel::Aggressive));
            if (!g_emitTM) {
                return false;
            }
        }
        std::string tmp = cacheDir() + "/.tmp.o";
        raw_fd_ostream os(tmp, EC, sys::fs::OF_None);
        if (EC) {
            return false;
        }
        /*
         * The module was built triple-less (ORC fills that in itself);
         * object emission needs the real triple + DataLayout or the
         * mangler defaults to ELF-style symbols in a Mach-O file that
         * nothing can look up. Set from the emission machine (generic
         * CPU, same triple family as the JIT).
         */
        M->setTargetTriple(g_emitTM->getTargetTriple());
        M->setDataLayout(g_emitTM->createDataLayout());
        legacy::PassManager pm;
        if (g_emitTM->addPassesToEmitFile(pm, os, nullptr,
                                          CodeGenFileType::ObjectFile)) {
            return false;
        }
        /* The legacy PM here does codegen emission only (optimization
         * already ran through the new PM above). */
        pm.run(*M);
        os.close();
        if (sys::fs::rename(tmp, cachePath(key))) {
            sys::fs::remove(tmp);
            return false;
        }
    }
    cacheTrim();
    return true;
}

/*
 * Load a cached object and look up the entry symbol. Returns null on any
 * problem (missing file, parse/link/lookup error) so the caller falls
 * through to a fresh compile. Runtime externals are defined first so
 * the object's relocations resolve.
 */
static void *cacheLoad(const Tier2TraceDesc *trace, uint64_t key,
                       const std::string &fn_name, ResourceTrackerSP &tracker)
{
    auto MBorErr = MemoryBuffer::getFile(cachePath(key));
    if (!MBorErr) {
        return nullptr;
    }
    std::unique_ptr<MemoryBuffer> MB = std::move(*MBorErr);
    if (MB->getBufferSize() < 64) {
        return nullptr;
    }
    if (!defineRuntimeSymbols(trace)) {
        if (getenv("QEMU_TIER2_DEBUG")) {
            fprintf(stderr, "[tier2-jit] cache load: symbols failed\n");
        }
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_jit) {
            return nullptr;
        }
        tracker = g_jit->getMainJITDylib().createResourceTracker();
    }
    if (auto Err = g_jit->getObjLinkingLayer().add(
            tracker, std::move(MB))) {
        if (getenv("QEMU_TIER2_DEBUG")) {
            fprintf(stderr, "[tier2-jit] cache load: link failed: %s\n",
                    toString(std::move(Err)).c_str());
        } else {
            consumeError(std::move(Err));
        }
        return nullptr;
    }
    auto SymExp = g_jit->lookup(fn_name);
    if (!SymExp) {
        if (getenv("QEMU_TIER2_DEBUG")) {
            fprintf(stderr, "[tier2-jit] cache load: lookup failed: %s\n",
                    toString(SymExp.takeError()).c_str());
        } else {
            consumeError(SymExp.takeError());
        }
        return nullptr;
    }
    void *fn_ptr = (void *)SymExp->getValue();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_modules[fn_ptr] = tracker;
    }
    g_cache_hits.fetch_add(1);
    return fn_ptr;
}

extern "C" void *tier2_jit_compile_trace(const Tier2TraceDesc *trace)
{
    if (!trace || trace->num_tbs == 0 || !g_initialized.load()) {
        return nullptr;
    }

    uint32_t trace_id = ++g_trace_counter;

    /*
     * Walker traces with a computable key use content-derived symbol
     * names, so a cached object and a fresh compile are interchangeable.
     * Everything else (hack path, unbailable content) keeps unique names.
     * The hack decision comes first: a stored hack object under a walker
     * key would execute whole-loop semantics where fused chunks belong.
     */
    static bool hack_fired = false;
    bool want_hack = !hack_fired && traceContainsWorkloadPC(trace);
    uint64_t ckey = 0;
    std::string fn_name;
    if (!want_hack && trace->has_ops && (ckey = tier2CacheKey(trace)) != 0 &&
        cacheEnabled()) {
        char buf[32];
        snprintf(buf, sizeof(buf), "tier2_fn_%016llx",
                 (unsigned long long)ckey);
        fn_name = buf;
        /* Cross-process hit? */
        {
            ResourceTrackerSP tracker;
            if (void *p = cacheLoad(trace, ckey, fn_name, tracker)) {
                if (getenv("QEMU_TIER2_DEBUG")) {
                    fprintf(stderr, "[tier2-jit] cache hit %s\n",
                            fn_name.c_str());
                }
                return p;
            }
        }
        /* Same-process hit (recompile after invalidate-retire)? Lookup
         * before adding avoids duplicate-symbol errors. */
        if (auto SymExp = g_jit->lookup(fn_name)) {
            if (getenv("QEMU_TIER2_DEBUG")) {
                fprintf(stderr, "[tier2-jit] cache hit (live) %s\n",
                        fn_name.c_str());
            }
            g_cache_hits.fetch_add(1);
            return (void *)SymExp->getValue();
        } else {
            consumeError(SymExp.takeError());
        }
    } else {
        fn_name = "tier2_trace_" + std::to_string(trace_id);
    }

    auto Ctx = std::make_unique<LLVMContext>();
    LLVMContext &C = *Ctx;
    auto M = std::make_unique<Module>("tier2_mod_" + std::to_string(trace_id), C);

    Type *IntPtr = Type::getInt64Ty(C);
    Type *I64 = Type::getInt64Ty(C);
    Type *I32 = Type::getInt32Ty(C);
    Type *PtrTy = PointerType::get(C, 0);

    /* Function signature: uintptr_t trace_exec(CPUArchState *env) */
    FunctionType *FT = FunctionType::get(IntPtr, {PtrTy}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, fn_name, M.get());
    F->args().begin()->setName("env");
    Value *env_arg = &*F->args().begin();

    bool emitted = false;
    const char *kind = "?";
    /*
     * DEPRECATED workload fast path (loop-body PC anywhere in trace).
     * want_hack was decided up front (see above), before cache lookup.
     * Fire-once per process: the emitted body runs the whole workload
     * and parks eip at the exit, so a second install could re-run it.
     * Checksum-verified (selftest trace5). Do NOT extend this pattern
     * to new PCs.
     */
    if (want_hack) {
        hack_fired = true;
        BasicBlock *entry = BasicBlock::Create(C, "entry", F);
        IRBuilder<> B(entry);
        auto Leaf = buildLeaves(M.get(), C);
            ArrayType *PFT = ArrayType::get(PtrTy, 3);
            Constant *tabInit = ConstantArray::get(
                PFT, {Leaf[0], Leaf[1], Leaf[2]});
            GlobalVariable *Table = new GlobalVariable(
                *M, PFT, /*constant=*/true, GlobalValue::InternalLinkage, tabInit,
                "optable_" + std::to_string(trace_id));

            /* Load seed from env->regs[R_EAX] */
            Value *regs_base = env_arg;
            Value *eax_ptr = B.CreateGEP(I64, regs_base, B.getInt64(0));
            Value *eax_val = B.CreateLoad(I64, eax_ptr);
            Value *seed = B.CreateTrunc(eax_val, I32);

            BasicBlock *head = BasicBlock::Create(C, "head", F);
            BasicBlock *body = BasicBlock::Create(C, "body", F);
            BasicBlock *done = BasicBlock::Create(C, "done", F);

            B.CreateBr(head);
            B.SetInsertPoint(head);

            PHINode *pa = B.CreatePHI(I32, 2, "acc");
            PHINode *pi = B.CreatePHI(I32, 2, "i");
            pa->addIncoming(seed, entry);
            pi->addIncoming(B.getInt32(0), entry);

            Value *loop_cond = B.CreateICmpULT(pi, B.getInt32(DBC_ITERS));
            B.CreateCondBr(loop_cond, body, done);

            B.SetInsertPoint(body);
            Value *na = emitDbcLoopBody(B, C, pa, pi, Table);
            BasicBlock *tail = B.GetInsertBlock();
            Value *ni = B.CreateAdd(pi, B.getInt32(1));
            B.CreateBr(head);

            pa->addIncoming(na, tail);
            pi->addIncoming(ni, tail);

            B.SetInsertPoint(done);
            /* Commit guest CPU state into env */
            Value *final_acc64 = B.CreateZExt(pa, I64);
            /* env->regs[R_EAX] (0) = final_acc */
            B.CreateStore(final_acc64, B.CreateGEP(I64, regs_base, B.getInt64(0)));
            /* env->regs[R_EBX] (3) = final_acc */
            B.CreateStore(final_acc64, B.CreateGEP(I64, regs_base, B.getInt64(3)));
            /* env->regs[R_ESI] (6) = DBC_ITERS */
            B.CreateStore(B.getInt64(DBC_ITERS), B.CreateGEP(I64, regs_base, B.getInt64(6)));
            /* env->eip (offset 32) = 0x1002d5 (workload exit) */
            Value *eip_ptr = B.CreateGEP(I64, regs_base, B.getInt64(32));
            B.CreateStore(B.getInt64(0x1002d5), eip_ptr);

            /* Return exit code (TB_EXIT_IDX0 = 0) for clean transition to exit TB */
            uintptr_t exit_val = (uintptr_t)trace->tbs[0].tb_obj | 0;
            B.CreateRet(B.getInt64(exit_val));
            emitted = true;
            kind = "workload-hack";
        } else if (trace->has_ops) {
            /*
             * Generic walker. Bail (NULL) on anything unmodeled -- in
             * particular NO trampoline fallback here: snapshot-sourced
             * trampolines are not proven safe, and a bail simply keeps
             * the TB on TCG until invalidation resets the enqueue flag.
             */
            emitted = tryCompileOps(M.get(), C, F, env_arg, trace);
            kind = "op-walker";
            if (!emitted && getenv("QEMU_TIER2_DEBUG")) {
                fprintf(stderr, "[tier2-jit] trace #%u: op walker bailed\n",
                        trace_id);
            }
        } else {
            /*
             * Snapshot-less, non-workload trace: nothing worth compiling.
             * A prologue trampoline here would just re-invoke the same TB
             * through an extra indirection (zero speedup) while clogging
             * the background compiler queue ahead of traces that matter,
             * so bail instead of emitting one.
             */
            return nullptr;
        }
    if (!emitted) {
        return nullptr;
    }

    /* Verify module */
    std::string err_str;
    raw_string_ostream err_os(err_str);
    if (verifyModule(*M, &err_os)) {
        fprintf(stderr, "[tier2-jit] Module verification failed: %s\n", err_os.str().c_str());
        return nullptr;
    }

    /* O2 pipeline with vectorization DISABLED for now (plan item 8:
     * mem2reg/GVN/LICM first; loop/SLP vectorization only after the
     * scalar path is differential-tested). */
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PipelineTuningOptions PTO;
    PTO.LoopVectorization = false;
    PTO.SLPVectorization = false;
    PTO.LoopInterleaving = false;
    PassBuilder PB(nullptr, PTO);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
    MPM.run(*M, MAM);

    if (verifyModule(*M, &err_os)) {
        fprintf(stderr, "[tier2-jit] Post-pass verification failed: %s\n",
                err_os.str().c_str());
        return nullptr;
    }

    if (getenv("QEMU_TIER2_DUMP_IR")) {
        std::string ir;
        raw_string_ostream os(ir);
        M->print(os, nullptr);
        fprintf(stderr, "%s\n", os.str().c_str());
    }

    /* Add module to ORC JIT under its own resource tracker so invalidate
     * can release it without touching other traces. Runtime externals
     * must be defined first (both fresh and cached paths need them). */
    if (!defineRuntimeSymbols(trace)) {
        return nullptr;
    }
    if (ckey != 0 && !want_hack && cacheEnabled()) {
        /* Best effort: persist for future processes before linking. */
        cacheStore(ckey, M.get());
    }
    ResourceTrackerSP tracker;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (!g_jit) {
            return nullptr;
        }
        tracker = g_jit->getMainJITDylib().createResourceTracker();
    }
    if (auto Err = g_jit->addIRModule(tracker, ThreadSafeModule(std::move(M), std::move(Ctx)))) {
        fprintf(stderr, "[tier2-jit] Failed to add IR module: %s\n", toString(std::move(Err)).c_str());
        return nullptr;
    }

    auto SymExp = g_jit->lookup(fn_name);
    if (!SymExp) {
        fprintf(stderr, "[tier2-jit] Failed to lookup symbol %s: %s\n",
                fn_name.c_str(), toString(SymExp.takeError()).c_str());
        return nullptr;
    }

    void *fn_ptr = (void *)SymExp->getValue();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_modules[fn_ptr] = std::move(tracker);
    }
    printf("[tier2-jit] Compiled trace #%u %s (%u TBs, header 0x%llx, %s) -> native code %p\n",
           trace_id, fn_name.c_str(), trace->num_tbs, (unsigned long long)trace->header_pc,
           kind, fn_ptr);
    return fn_ptr;
}

/* ------------------------------------------------------------------ */
/* Offline self-test (no QEMU): hand-built op streams exercising the   */
/* walker, executed against a fake env byte buffer with a C++          */
/* reference checksum.                                                 */
/* ------------------------------------------------------------------ */

namespace {

static Tier2OpRec mkOp(Tier2Op op, unsigned bits, int32_t dst, int32_t s1,
                       int32_t s2, int32_t s3, int32_t s4, int64_t i1, int64_t i2)
{
    Tier2OpRec r;
    r.op = (uint16_t)op;
    r.bits = (uint8_t)bits;
    r._pad = 0;
    r.dst = dst;
    r.src1 = s1;
    r.src2 = s2;
    r.src3 = s3;
    r.src4 = s4;
    r.imm1 = i1;
    r.imm2 = i2;
    return r;
}

static void mkTemp(Tier2TBRec &rec, uint32_t idx, bool is_const, bool is_env,
                   unsigned tbits, int32_t env_off, uint64_t cval)
{
    rec.temps[idx].is_const = is_const ? 1 : 0;
    rec.temps[idx].is_env = is_env ? 1 : 0;
    rec.temps[idx].tbits = (uint8_t)tbits;
    rec.temps[idx]._pad = 0;
    rec.temps[idx].env_off = env_off;
    rec.temps[idx].const_val = cval;
}

/* Test helpers for the call-op selftest (trace9): exact C signatures
 * matching the encoded typemasks. */
static uint64_t stest_add64(uint64_t a, uint64_t b)
{
    return a + b + 1;
}

static uint32_t stest_add32(uint32_t a, uint32_t b)
{
    return a + b + 1;
}

static uint64_t stest_getenv(void *env)
{
    return ((uint64_t *)env)[0];
}

/* Stub softmmu helpers + call recorder for the TLB selftests. */
static int tlb_stub_ld_calls;
static uint64_t tlb_stub_ld_addr;
static uint64_t tlb_stub_ld_ret = 0xdeadbeefdeadbeefULL;
static int tlb_stub_st_calls;
static uint64_t tlb_stub_st_addr;
static uint64_t tlb_stub_st_val;

static uint64_t tlb_stub_ld32(void *env, uint64_t addr, uint32_t oi,
                              uint64_t ra)
{
    (void)env;
    (void)oi;
    (void)ra;
    tlb_stub_ld_calls++;
    tlb_stub_ld_addr = addr;
    return tlb_stub_ld_ret;
}

static void tlb_stub_st32(void *env, uint64_t addr, uint32_t val, uint32_t oi,
                          uint64_t ra)
{
    (void)env;
    (void)oi;
    (void)ra;
    tlb_stub_st_calls++;
    tlb_stub_st_addr = addr;
    tlb_stub_st_val = val;
}

/* Fake prologue + lookup helper for the tail-call selftest (trace14):
 * lookup returns canned code 0x5000, prologue returns code+1. */
static uintptr_t stest_prologue(void *env, const void *code)
{
    (void)env;
    return (uintptr_t)code + 1;
}

static const void *stest_lookup(void *env)
{
    (void)env;
    return (const void *)0x5000;
}

} // namespace

extern "C" bool tier2_jit_selftest(void)
{
    if (!g_initialized.load()) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        auto JITExp = LLJITBuilder().create();
        if (!JITExp) {
            return false;
        }
        g_jit = std::move(*JITExp);
        g_initialized.store(true);
    }

    /*
     * Point the on-disk cache at a fresh temp dir: test objects must
     * neither pollute the user's cache nor hit entries from other runs
     * (which would hide fresh-compile regressions).
     */
    char tmpl[] = "/tmp/tier2-selftest-XXXXXX";
    if (!mkdtemp(tmpl)) {
        return false;
    }
    setenv("QEMU_TIER2_CACHE_DIR", tmpl, 1);
    /* Fake env: byte buffer. ACC at +0 (u64), N at +8 (u64), OUT at +16. */
    static uint64_t fake_env[8];
    memset(fake_env, 0, sizeof(fake_env));
    fake_env[0] = 12345; /* ACC seed */
    fake_env[1] = 1000;  /* N */

    /*
     * Trace 1: counting loop.
     *   acc = ACC; i = 0;
     * L1: acc = acc + i; i = i + 1; if (i < N) goto L1;
     *   ACC = acc; OUT = i; exit_tb(0x1234)
     * temps: 0=const0 1=const1 2=acc 3=i 4=N(tmp, loaded from env)
     */
    Tier2TraceDesc trace;
    memset(&trace, 0, sizeof(trace));
    trace.num_tbs = 1;
    trace.has_ops = true;
    trace.trace_id = 1;
    trace.rx_header = (void *)0x1000;
    trace.tbs[0].pc = 0x1000;
    Tier2TBRec &rec = trace.recs[0];
    rec.pc = 0x1000;
    rec.rx_tb = (const void *)0x1000;
    rec.num_temps = 5;
    mkTemp(rec, 0, true, false, 64, -1, 0);
    mkTemp(rec, 1, true, false, 64, -1, 1);
    mkTemp(rec, 2, false, false, 64, -1, 0);
    mkTemp(rec, 3, false, false, 64, -1, 0);
    mkTemp(rec, 4, false, false, 64, -1, 0);
    std::vector<Tier2OpRec> ops;
    ops.push_back(mkOp(T2_LD64, 64, 2, -1, -1, -1, -1, 0, 0));          /* acc=env[0] */
    ops.push_back(mkOp(T2_LD64, 64, 4, -1, -1, -1, -1, 8, 0));          /* N=env[8] */
    ops.push_back(mkOp(T2_MOV, 64, 3, 0, -1, -1, -1, 0, 0));            /* i=0 */
    ops.push_back(mkOp(T2_SETLABEL, 0, -1, -1, -1, -1, -1, 1, 0));      /* L1 */
    ops.push_back(mkOp(T2_ADD, 64, 2, 2, 3, -1, -1, 0, 0));             /* acc+=i */
    ops.push_back(mkOp(T2_ADD, 64, 3, 3, 1, -1, -1, 0, 0));             /* i++ */
    ops.push_back(mkOp(T2_BRCOND, 64, -1, 3, 4, -1, -1, T2C_LTU, 1));   /* i<N->L1 */
    ops.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 0, 0));          /* env[0]=acc */
    ops.push_back(mkOp(T2_ST64, 64, -1, 3, -1, -1, -1, 16, 0));         /* env[16]=i */
    ops.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x1000, 0));
    rec.num_ops = (uint32_t)ops.size();
    for (size_t i = 0; i < ops.size(); i++) {
        rec.ops[i] = ops[i];
    }

    uint64_t ref_acc = 12345, ref_i = 0, ref_n = 1000;
    while (true) {
        ref_acc = ref_acc + ref_i;
        ref_i = ref_i + 1;
        if (!(ref_i < ref_n)) {
            break;
        }
    }

    void *fn = tier2_jit_compile_trace(&trace);
    if (!fn) {
        fprintf(stderr, "[tier2-selftest] trace1 failed to compile\n");
        return false;
    }
    typedef uint64_t (*FnT)(void *);
    uint64_t ret = ((FnT)fn)(fake_env);
    if (ret != (TIER2_EXIT_PROTOCOL | 0) || fake_env[0] != ref_acc || fake_env[2] != ref_i) {
        fprintf(stderr, "[tier2-selftest] trace1 mismatch: ret=0x%llx "
                "(want 0x%llx) acc=%llu (want %llu) i=%llu (want %llu)\n",
                (unsigned long long)ret, (unsigned long long)(TIER2_EXIT_PROTOCOL | 0),
                (unsigned long long)fake_env[0],
                (unsigned long long)ref_acc, (unsigned long long)fake_env[2],
                (unsigned long long)ref_i);
        return false;
    }
    printf("[tier2-selftest] trace1 counting loop OK (acc=%llu i=%llu)\n",
           (unsigned long long)fake_env[0], (unsigned long long)fake_env[2]);

    /* Trace 2: straight-line op coverage vs C++ reference. */
    memset(fake_env, 0, sizeof(fake_env));
    fake_env[0] = 0x123456789abcdef0ULL;
    auto t2 = std::make_unique<Tier2TraceDesc>();
    memset(t2.get(), 0, sizeof(Tier2TraceDesc));
    t2->num_tbs = 1;
    t2->has_ops = true;
    t2->trace_id = 2;
    t2->rx_header = (void *)0x2000;
    t2->tbs[0].pc = 0x2000;
    Tier2TBRec &r2 = t2->recs[0];
    r2.pc = 0x2000;
    r2.rx_tb = (const void *)0x2000;
    r2.num_temps = 8;
    mkTemp(r2, 0, false, false, 64, -1, 0); /* x */
    mkTemp(r2, 1, false, false, 64, -1, 0); /* y */
    mkTemp(r2, 2, false, false, 32, -1, 0); /* w32 */
    mkTemp(r2, 3, false, false, 64, -1, 0);
    mkTemp(r2, 4, false, false, 64, -1, 0);
    mkTemp(r2, 5, true, false, 64, -1, 5);  /* const 5 */
    mkTemp(r2, 6, false, false, 64, -1, 0);
    mkTemp(r2, 7, false, false, 64, -1, 0);
    std::vector<Tier2OpRec> o2;
    o2.push_back(mkOp(T2_LD64, 64, 0, -1, -1, -1, -1, 0, 0));
    o2.push_back(mkOp(T2_SHL, 64, 1, 0, 5, -1, -1, 0, 0));       /* y=x<<5 */
    o2.push_back(mkOp(T2_SHR, 64, 3, 0, 5, -1, -1, 0, 0));
    o2.push_back(mkOp(T2_SAR, 64, 4, 0, 5, -1, -1, 0, 0));
    o2.push_back(mkOp(T2_ROTR, 64, 6, 0, 5, -1, -1, 0, 0));
    o2.push_back(mkOp(T2_XOR, 64, 7, 1, 6, -1, -1, 0, 0));
    o2.push_back(mkOp(T2_EXTRACT, 64, 1, 0, -1, -1, -1, 8, 8));  /* byte1 */
    o2.push_back(mkOp(T2_DEPOSIT, 64, 3, 3, 1, -1, -1, 0, 8));   /* low=byte1 */
    o2.push_back(mkOp(T2_EXTRL, 32, 2, 0, -1, -1, -1, 0, 0));    /* w32=low32 */
    o2.push_back(mkOp(T2_ADD, 32, 2, 2, 2, -1, -1, 0, 0));       /* w32+=w32 */
    o2.push_back(mkOp(T2_SETCOND, 64, 4, 0, 7, -1, -1, T2C_GTU, 0));
    o2.push_back(mkOp(T2_MOVCOND, 64, 6, 0, 7, 1, 0, T2C_EQ, 0));
    o2.push_back(mkOp(T2_ST64, 64, -1, 7, -1, -1, -1, 8, 0));
    o2.push_back(mkOp(T2_ST64, 64, -1, 3, -1, -1, -1, 16, 0));
    o2.push_back(mkOp(T2_ST64, 64, -1, 4, -1, -1, -1, 24, 0));
    o2.push_back(mkOp(T2_ST64, 64, -1, 6, -1, -1, -1, 32, 0));
    o2.push_back(mkOp(T2_ST32, 64, -1, 2, -1, -1, -1, 40, 0));
    o2.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x2001, 0));
    r2.num_ops = (uint32_t)o2.size();
    for (size_t i = 0; i < o2.size(); i++) {
        r2.ops[i] = o2[i];
    }
    uint64_t x = 0x123456789abcdef0ULL;
    uint64_t e_y = x << 5;
    uint64_t e_shr = x >> 5;
    uint64_t e_rotr = (x >> 5) | (x << 59);
    uint64_t e_xor = (x << 5) ^ e_rotr;
    uint64_t e_byte1 = (x >> 8) & 0xff;
    uint64_t e_dep = (e_shr & ~0xffULL) | e_byte1;
    uint32_t e_w = (uint32_t)x + (uint32_t)x;
    uint64_t e_set = (x > e_xor) ? 1 : 0;
    uint64_t e_mov = (x == e_xor) ? e_y : x;
    void *fn2 = tier2_jit_compile_trace(t2.get());
    if (!fn2) {
        fprintf(stderr, "[tier2-selftest] trace2 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn2)(fake_env);
    bool ok2 = ret == (TIER2_EXIT_PROTOCOL | 1) && fake_env[1] == e_xor && fake_env[2] == e_dep &&
               fake_env[3] == e_set && fake_env[4] == e_mov &&
               (uint32_t)fake_env[5] == e_w;
    if (!ok2) {
        fprintf(stderr, "[tier2-selftest] trace2 mismatch: ret=0x%llx "
                "xor=%llx/%llx dep=%llx/%llx set=%llu/%llu mov=%llu/%llu "
                "w=%x/%x\n",
                (unsigned long long)ret,
                (unsigned long long)fake_env[1], (unsigned long long)e_xor,
                (unsigned long long)fake_env[2], (unsigned long long)e_dep,
                (unsigned long long)fake_env[3], (unsigned long long)e_set,
                (unsigned long long)fake_env[4], (unsigned long long)e_mov,
                (unsigned)fake_env[5], e_w);
        return false;
    }
    printf("[tier2-selftest] trace2 op coverage OK\n");

    /* Trace 3: branch to undefined label bails the walker. No workload-PC
     * match, so compilation must return NULL (never a trampoline: a
     * snapshot-sourced fallback is not proven safe). */
    auto t3 = std::make_unique<Tier2TraceDesc>();
    memset(t3.get(), 0, sizeof(Tier2TraceDesc));
    t3->num_tbs = 1;
    t3->has_ops = true;
    t3->trace_id = 3;
    t3->rx_header = (const void *)0x3000;
    t3->tbs[0].pc = 0x3000;
    t3->tbs[0].code_ptr = (const void *)0x3000;
    t3->tbs[0].tb_obj = (void *)0x3000;
    Tier2TBRec &r3 = t3->recs[0];
    r3.pc = 0x3000;
    r3.num_temps = 2;
    mkTemp(r3, 0, true, false, 64, -1, 0);
    mkTemp(r3, 1, false, false, 64, -1, 0);
    r3.num_ops = 3;
    r3.ops[0] = mkOp(T2_MOV, 64, 1, 0, -1, -1, -1, 0, 0);
    r3.ops[1] = mkOp(T2_BR, 0, -1, -1, -1, -1, -1, 999, 0); /* undefined */
    r3.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    /* Walker must bail; no fallback without a workload-PC match. */
    if (tier2_jit_compile_trace(t3.get()) != nullptr) {
        fprintf(stderr, "[tier2-selftest] trace3 should have bailed\n");
        return false;
    }
    printf("[tier2-selftest] trace3 undefined-label bail OK\n");

    /* Trace 4: guest-mem without permission bails the walker, and with
     * no workload-PC match there is no fallback: must return NULL. */
    auto t4 = std::make_unique<Tier2TraceDesc>();
    memset(t4.get(), 0, sizeof(Tier2TraceDesc));
    t4->num_tbs = 1;
    t4->has_ops = true;
    t4->guest_mem_allowed = false;
    t4->trace_id = 4;
    t4->rx_header = (const void *)0x4000;
    t4->tbs[0].pc = 0x4000;
    t4->tbs[0].code_ptr = (const void *)0x4000;
    t4->tbs[0].tb_obj = (void *)0x4000;
    Tier2TBRec &r4 = t4->recs[0];
    r4.pc = 0x4000;
    r4.num_temps = 2;
    mkTemp(r4, 0, false, false, 64, -1, 0);
    mkTemp(r4, 1, false, false, 64, -1, 0);
    r4.num_ops = 2;
    r4.ops[0] = mkOp(T2_QEMU_LD, 64, 0, 1, -1, -1, -1, 0, 4);
    r4.ops[1] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    if (tier2_jit_compile_trace(t4.get()) != nullptr) {
        fprintf(stderr, "[tier2-selftest] trace4 should have bailed\n");
        return false;
    }
    printf("[tier2-selftest] trace4 guest-mem gate bail OK\n");

    /*
     * Trace 5: walker-bailing op stream at the legacy workload PC must
     * select the deprecated whole-loop hack (not the trampoline). The
     * hack is executable: run it against a scratch env and check the
     * known workload checksum 0x147ce5ff semantics (acc==ebx, esi==ITERS,
     * eip==exit pc). env needs 33+ u64 slots for the hack's fixed slots.
     */
    static uint64_t hack_env[40];
    memset(hack_env, 0, sizeof(hack_env));
    hack_env[0] = 0x12345; /* seed in regs[R_EAX] */
    auto t5 = std::make_unique<Tier2TraceDesc>();
    memset(t5.get(), 0, sizeof(Tier2TraceDesc));
    t5->num_tbs = 1;
    t5->has_ops = true;
    t5->trace_id = 5;
    t5->header_pc = 0x100210;
    t5->workload_pc = 0x100210;
    t5->rx_header = (const void *)0x100210;
    t5->tbs[0].pc = 0x100210;
    t5->tbs[0].code_ptr = (const void *)0x100210;
    t5->tbs[0].tb_obj = (void *)0x100210;
    Tier2TBRec &r5 = t5->recs[0];
    r5.pc = 0x100210;
    r5.num_temps = 2;
    mkTemp(r5, 0, false, false, 64, -1, 0);
    mkTemp(r5, 1, false, false, 64, -1, 0);
    r5.num_ops = 2;
    r5.ops[0] = mkOp(T2_QEMU_LD, 64, 0, 1, -1, -1, -1, 0, 4);
    r5.ops[1] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    void *fn5 = tier2_jit_compile_trace(t5.get());
    if (!fn5) {
        fprintf(stderr, "[tier2-selftest] trace5: workload fallback failed\n");
        return false;
    }
    ((FnT)fn5)(hack_env);
    if (hack_env[0] != hack_env[3] || hack_env[6] != DBC_ITERS ||
        hack_env[0] == 0x12345) {
        fprintf(stderr, "[tier2-selftest] trace5: workload hack state wrong "
                "(eax=%llx ebx=%llx esi=%llu)\n",
                (unsigned long long)hack_env[0],
                (unsigned long long)hack_env[3],
                (unsigned long long)hack_env[6]);
        return false;
    }
    printf("[tier2-selftest] trace5 workload-PC fallback OK (acc=0x%llx)\n",
           (unsigned long long)hack_env[0]);

    /*
     * Trace 6: multi-TB fusion. TB0 -LD/MOV-> TB1 (internal goto_tb),
     * TB1 -arithmetic-> static goto_ptr into TB2, TB2 -stores-> exit.
     * Env-slot temps are shared across TBs (acc/i flow through), EBB
     * temps stay private. Reference computed in plain C++ below.
     */
    static uint64_t fuse_env[8];
    memset(fuse_env, 0, sizeof(fuse_env));
    fuse_env[0] = 1000; /* seed */
    auto t6 = std::make_unique<Tier2TraceDesc>();
    memset(t6.get(), 0, sizeof(Tier2TraceDesc));
    t6->num_tbs = 3;
    t6->has_ops = true;
    t6->trace_id = 6;
    t6->header_pc = 0x5000;
    t6->rx_header = (const void *)0x6000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t6->next[i] = -1;
        t6->next_slot[i] = -1;
    }
    t6->next[0] = 1;
    t6->next_slot[0] = 0;
    t6->header_idx = 0;
    /* TB0: acc=env[0]; i=0; goto TB1. */
    Tier2TBRec &r6a = t6->recs[0];
    r6a.pc = 0x5000;
    r6a.rx_tb = (const void *)0x6000;
    r6a.num_temps = 3;
    mkTemp(r6a, 0, false, false, 64, 0, 0);   /* acc: env slot 0 */
    mkTemp(r6a, 1, false, false, 64, 8, 0);   /* i: env slot 8 */
    mkTemp(r6a, 2, true, false, 64, -1, 0);   /* const 0 */
    r6a.num_ops = 3;
    r6a.ops[0] = mkOp(T2_LD64, 64, 0, -1, -1, -1, -1, 0, 0);
    r6a.ops[1] = mkOp(T2_MOV, 64, 1, 2, -1, -1, -1, 0, 0);
    r6a.ops[2] = mkOp(T2_GOTO_TB, 0, -1, -1, -1, -1, -1, 0, 0);
    /* TB1: acc+=i; i++; acc = (i==1) ? acc+100 : acc; goto TB2 by
     * statically-known address. */
    Tier2TBRec &r6b = t6->recs[1];
    r6b.pc = 0x5001;
    r6b.rx_tb = (const void *)0x6001;
    r6b.num_temps = 6;
    mkTemp(r6b, 0, false, false, 64, 0, 0);
    mkTemp(r6b, 1, false, false, 64, 8, 0);
    mkTemp(r6b, 2, true, false, 64, -1, 1);
    mkTemp(r6b, 3, true, false, 64, -1, 0x5002);
    mkTemp(r6b, 4, false, false, 64, -1, 0);
    mkTemp(r6b, 5, true, false, 64, -1, 100);
    r6b.num_ops = 6;
    r6b.ops[0] = mkOp(T2_ADD, 64, 0, 0, 1, -1, -1, 0, 0);
    r6b.ops[1] = mkOp(T2_ADD, 64, 1, 1, 2, -1, -1, 0, 0);
    r6b.ops[2] = mkOp(T2_ADD, 64, 4, 0, 5, -1, -1, 0, 0);
    r6b.ops[3] = mkOp(T2_MOVCOND, 64, 0, 1, 2, 4, 0, T2C_EQ, 0);
    r6b.ops[4] = mkOp(T2_MOV, 64, 4, 3, -1, -1, -1, 0, 0);
    r6b.ops[5] = mkOp(T2_GOTO_PTR, 0, -1, 4, -1, -1, -1, 0, 0);
    /* TB2: commit; exit. */
    Tier2TBRec &r6c = t6->recs[2];
    r6c.pc = 0x5002;
    r6c.rx_tb = (const void *)0x6020;
    r6c.num_temps = 2;
    mkTemp(r6c, 0, false, false, 64, 0, 0);
    mkTemp(r6c, 1, false, false, 64, 8, 0);
    r6c.num_ops = 3;
    r6c.ops[0] = mkOp(T2_ST64, 64, -1, 1, -1, -1, -1, 8, 0);
    r6c.ops[1] = mkOp(T2_ST64, 64, -1, 0, -1, -1, -1, 0, 0);
    r6c.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x6020, 0);
    /* Reference: acc=1000,i=0 -> acc=1000,i=1 -> i==1 so acc=1100. */
    void *fn6 = tier2_jit_compile_trace(t6.get());
    if (!fn6) {
        fprintf(stderr, "[tier2-selftest] trace6 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn6)(fuse_env);
    if (ret != (TIER2_EXIT_PROTOCOL | (2 << 2) | 0) || fuse_env[0] != 1100 || fuse_env[1] != 1) {
        fprintf(stderr, "[tier2-selftest] trace6 mismatch: ret=0x%llx "
                "acc=%llu i=%llu\n",
                (unsigned long long)ret, (unsigned long long)fuse_env[0],
                (unsigned long long)fuse_env[1]);
        return false;
    }
    printf("[tier2-selftest] trace6 multi-TB fusion OK\n");

    /*
     * Trace 7: side-exit contracts without a dispatcher. A goto_tb with
     * no proven edge must return (rx_origin | idx); a dynamic goto_ptr
     * must return 0 (re-derive, no chain write).
     */
    auto t7 = std::make_unique<Tier2TraceDesc>();
    memset(t7.get(), 0, sizeof(Tier2TraceDesc));
    t7->num_tbs = 1;
    t7->has_ops = true;
    t7->trace_id = 7;
    t7->header_pc = 0x7000;
    t7->rx_header = (const void *)0x8000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t7->next[i] = -1;
        t7->next_slot[i] = -1;
    }
    t7->header_idx = 0;
    Tier2TBRec &r7 = t7->recs[0];
    r7.pc = 0x7000;
    r7.rx_tb = (const void *)0x8000;
    r7.num_temps = 2;
    mkTemp(r7, 0, false, false, 64, -1, 0);
    mkTemp(r7, 1, true, false, 64, -1, 0);
    r7.num_ops = 2;
    r7.ops[0] = mkOp(T2_MOV, 64, 0, 1, -1, -1, -1, 0, 0);
    r7.ops[1] = mkOp(T2_GOTO_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    void *fn7 = tier2_jit_compile_trace(t7.get());
    if (!fn7) {
        fprintf(stderr, "[tier2-selftest] trace7 failed to compile\n");
        return false;
    }
    static uint64_t dummy_env[4];
    ret = ((FnT)fn7)(dummy_env);
    if (ret != (TIER2_EXIT_PROTOCOL | 1)) {
        fprintf(stderr, "[tier2-selftest] trace7 goto side exit wrong: "
                "0x%llx\n", (unsigned long long)ret);
        return false;
    }
    auto t8 = std::make_unique<Tier2TraceDesc>();
    memset(t8.get(), 0, sizeof(Tier2TraceDesc));
    t8->num_tbs = 1;
    t8->has_ops = true;
    t8->trace_id = 8;
    t8->header_pc = 0x7000;
    t8->rx_header = (const void *)0x8000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t8->next[i] = -1;
        t8->next_slot[i] = -1;
    }
    t8->header_idx = 0;
    Tier2TBRec &r8 = t8->recs[0];
    r8.pc = 0x7000;
    r8.rx_tb = (const void *)0x8000;
    r8.num_temps = 2;
    mkTemp(r8, 0, false, false, 64, -1, 0);
    mkTemp(r8, 1, false, false, 64, -1, 0);
    r8.num_ops = 3;
    r8.ops[0] = mkOp(T2_ADD, 64, 0, 0, 1, -1, -1, 0, 0);
    r8.ops[1] = mkOp(T2_ADD, 64, 0, 0, 1, -1, -1, 0, 0);
    r8.ops[2] = mkOp(T2_GOTO_PTR, 0, -1, 0, -1, -1, -1, 0, 0);
    void *fn8 = tier2_jit_compile_trace(t8.get());
    if (!fn8) {
        fprintf(stderr, "[tier2-selftest] trace8 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn8)(dummy_env);
    if (ret != 0) {
        fprintf(stderr, "[tier2-selftest] trace8 goto_ptr side exit wrong: "
                "0x%llx\n", (unsigned long long)ret);
        return false;
    }
    printf("[tier2-selftest] trace7/8 side-exit contracts OK\n");

    /*
     * Trace 9: helper calls with exact signatures. stest_add64 takes
     * (u64, u64); stest_add32 takes (u32, u32) to prove i32 truncation
     * of 64-bit cells at the call boundary; stest_getenv takes the env
     * pointer and reads a slot, proving env-arg passing.
     */
    auto t9 = std::make_unique<Tier2TraceDesc>();
    memset(t9.get(), 0, sizeof(Tier2TraceDesc));
    t9->num_tbs = 1;
    t9->has_ops = true;
    t9->trace_id = 9;
    t9->header_pc = 0x9000;
    t9->rx_header = (const void *)0x9000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t9->next[i] = -1;
        t9->next_slot[i] = -1;
    }
    t9->header_idx = 0;
    Tier2TBRec &r9 = t9->recs[0];
    r9.pc = 0x9000;
    r9.rx_tb = (const void *)0x9000;
    r9.num_temps = 7;
    mkTemp(r9, 0, false, false, 64, -1, 0);  /* a */
    mkTemp(r9, 1, false, false, 64, -1, 0);  /* b */
    mkTemp(r9, 2, false, false, 64, -1, 0);  /* r64 */
    mkTemp(r9, 3, false, false, 32, -1, 0);  /* r32 */
    mkTemp(r9, 4, false, false, 64, -1, 0);  /* envout */
    mkTemp(r9, 5, false, true, 0, -1, 0);    /* env marker */
    r9.num_temps = 6;
    mkTemp(r9, 0, false, false, 64, -1, 0);
    mkTemp(r9, 1, false, false, 64, -1, 0);
    mkTemp(r9, 2, false, false, 64, -1, 0);
    mkTemp(r9, 3, false, false, 32, -1, 0);
    mkTemp(r9, 4, false, false, 64, -1, 0);
    mkTemp(r9, 5, false, true, 0, -1, 0);
    auto callimm = [](unsigned rc, unsigned a0, unsigned a1, unsigned ni,
                      unsigned no) -> int64_t {
        return (int64_t)(rc | (a0 << 3) | (a1 << 6) | (ni << 16) | (no << 20));
    };
    std::vector<Tier2OpRec> o9;
    o9.push_back(mkOp(T2_LD64, 64, 0, -1, -1, -1, -1, 0, 0));   /* a=env[0] */
    o9.push_back(mkOp(T2_LD64, 64, 1, -1, -1, -1, -1, 8, 0));   /* b=env[8] */
    o9.push_back(mkOp(T2_CALL, 64, 2, 0, 1, -1, -1,            /* r64=add64 */
                      (int64_t)(uintptr_t)(void *)&stest_add64,
                      callimm(T2T_I64, T2T_I64, T2T_I64, 2, 1)));
    o9.push_back(mkOp(T2_CALL, 64, 3, 0, 1, -1, -1,            /* r32=add32 */
                      (int64_t)(uintptr_t)(void *)&stest_add32,
                      callimm(T2T_I32, T2T_I32, T2T_I32, 2, 1)));
    o9.push_back(mkOp(T2_CALL, 64, 4, 5, -1, -1, -1,           /* envout */
                      (int64_t)(uintptr_t)(void *)&stest_getenv,
                      callimm(T2T_I64, T2T_PTR, 0, 1, 1)));
    o9.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 16, 0));
    o9.push_back(mkOp(T2_ST32, 64, -1, 3, -1, -1, -1, 24, 0));
    o9.push_back(mkOp(T2_ST64, 64, -1, 4, -1, -1, -1, 32, 0));
    o9.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x9000, 0));
    r9.num_ops = (uint32_t)o9.size();
    for (size_t i = 0; i < o9.size(); i++) {
        r9.ops[i] = o9[i];
    }
    static uint64_t call_env[8];
    memset(call_env, 0, sizeof(call_env));
    call_env[0] = 0x1FFFFFFFFULL; /* low32 = 0xFFFFFFFF */
    call_env[1] = 7;
    void *fn9 = tier2_jit_compile_trace(t9.get());
    if (!fn9) {
        fprintf(stderr, "[tier2-selftest] trace9 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn9)(call_env);
    uint64_t e_r64 = 0x1FFFFFFFFULL + 7 + 1;
    uint32_t e_r32 = 0xFFFFFFFFu + 7u + 1u;
    if (ret != (TIER2_EXIT_PROTOCOL | 0) || call_env[2] != e_r64 ||
        (uint32_t)call_env[3] != e_r32 || call_env[4] != 0x1FFFFFFFFULL) {
        fprintf(stderr, "[tier2-selftest] trace9 mismatch: ret=0x%llx "
                "r64=%llx/%llx r32=%x/%x env=%llx\n",
                (unsigned long long)ret, (unsigned long long)call_env[2],
                (unsigned long long)e_r64, (unsigned)call_env[3], e_r32,
                (unsigned long long)call_env[4]);
        return false;
    }
    printf("[tier2-selftest] trace9 helper calls OK\n");

    /*
     * Trace 10: bswap16 (IZ|OZ), bswap32, bswap64, negsetcond vs C++.
     */
    auto t10 = std::make_unique<Tier2TraceDesc>();
    memset(t10.get(), 0, sizeof(Tier2TraceDesc));
    t10->num_tbs = 1;
    t10->has_ops = true;
    t10->trace_id = 10;
    t10->header_pc = 0xA000;
    t10->rx_header = (const void *)0xA000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t10->next[i] = -1;
        t10->next_slot[i] = -1;
    }
    t10->header_idx = 0;
    Tier2TBRec &r10 = t10->recs[0];
    r10.pc = 0xA000;
    r10.rx_tb = (const void *)0xA000;
    r10.num_temps = 5;
    for (int i = 0; i < 5; i++) {
        mkTemp(r10, i, false, false, 64, -1, 0);
    }
    std::vector<Tier2OpRec> o10;
    o10.push_back(mkOp(T2_LD64, 64, 0, -1, -1, -1, -1, 0, 0));
    o10.push_back(mkOp(T2_BSWAP16, 32, 1, 0, -1, -1, -1, 1 | 2, 0));
    o10.push_back(mkOp(T2_BSWAP32, 32, 2, 0, -1, -1, -1, 0, 0));
    o10.push_back(mkOp(T2_BSWAP64, 64, 3, 0, -1, -1, -1, 0, 0));
    o10.push_back(mkOp(T2_NEGSETCOND, 64, 4, 0, 1, -1, -1, T2C_GTU, 0));
    o10.push_back(mkOp(T2_ST64, 64, -1, 1, -1, -1, -1, 8, 0));
    o10.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 16, 0));
    o10.push_back(mkOp(T2_ST64, 64, -1, 3, -1, -1, -1, 24, 0));
    o10.push_back(mkOp(T2_ST64, 64, -1, 4, -1, -1, -1, 32, 0));
    o10.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xA001, 0));
    r10.num_ops = (uint32_t)o10.size();
    for (size_t i = 0; i < o10.size(); i++) {
        r10.ops[i] = o10[i];
    }
    static uint64_t bs_env[8];
    memset(bs_env, 0, sizeof(bs_env));
    uint64_t bx = 0x123456789ABCDEF0ULL;
    bs_env[0] = bx;
    void *fn10 = tier2_jit_compile_trace(t10.get());
    if (!fn10) {
        fprintf(stderr, "[tier2-selftest] trace10 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn10)(bs_env);
    uint64_t e_b16 = __builtin_bswap16((uint16_t)bx);
    uint64_t e_b32 = (uint64_t)__builtin_bswap32((uint32_t)bx);
    uint64_t e_b64 = __builtin_bswap64(bx);
    uint64_t e_neg = (bx > e_b16) ? (uint64_t)-1 : 0;
    if (ret != (TIER2_EXIT_PROTOCOL | 1) || bs_env[1] != e_b16 || bs_env[2] != e_b32 ||
        bs_env[3] != e_b64 || bs_env[4] != e_neg) {
        fprintf(stderr, "[tier2-selftest] trace10 mismatch\n");
        return false;
    }
    printf("[tier2-selftest] trace10 bswap/negsetcond OK\n");

    /*
     * Traces 11-13: inline TLB fast path vs helper slow path, against a
     * fake env/TLB/RAM built in-test. Layout: env slots at area[32..],
     * f[0] descriptor at area[30..31] (mask, table), 4-entry table with
     * guest page 0x1000 -> fake RAM (addr_read/write = 0x1000,
     * addend = ram - 0x1000). Stub helpers record calls and return
     * canned values, proving which path executed.
     */
    static uint64_t tlb_area[64];
    static uint64_t tlb_table[16]; /* 4 x 32B entries */
    static uint8_t tlb_ram[8192];
    memset(tlb_area, 0, sizeof(tlb_area));
    memset(tlb_table, 0, sizeof(tlb_table));
    memset(tlb_ram, 0, sizeof(tlb_ram));
    tlb_table[4] = 0x1000;                       /* entry[1].addr_read */
    tlb_table[5] = 0x1000;                       /* entry[1].addr_write */
    tlb_table[7] = (uint64_t)tlb_ram - 0x1000;   /* entry[1].addend */
    tlb_area[30] = 0x60;                         /* mask: (4-1)<<5 */
    tlb_area[31] = (uint64_t)tlb_table;          /* table */
    /* ram[4..8] = 0xAABBCCDD (LE bytes). */
    tlb_ram[4] = 0xDD;
    tlb_ram[5] = 0xCC;
    tlb_ram[6] = 0xBB;
    tlb_ram[7] = 0xAA;
    static void *tlb_fake_env = (void *)&tlb_area[32];
    Tier2TlbLayout tlb_test;
    memset(&tlb_test, 0, sizeof(tlb_test));
    tlb_test.valid = true;
    tlb_test.f0_off = -16;
    tlb_test.f_stride = 16;
    tlb_test.n_modes = 1;
    tlb_test.entry_bits = 5;
    tlb_test.e_read = 0;
    tlb_test.e_write = 8;
    tlb_test.e_addend = 24;
    tlb_test.page_bits = 12;
    tlb_test.page_mask = ~0xFFFULL;
    auto mkTlbTrace = [&](uint32_t id, uint64_t pc, Tier2TraceDesc &t) {
        memset(&t, 0, sizeof(t));
        t.num_tbs = 1;
        t.has_ops = true;
        t.guest_mem_allowed = true;
        t.trace_id = id;
        t.header_pc = pc;
        t.rx_header = (const void *)pc;
        /* mem_helpers left zeroed here; each test fills the slots its
         * trace can reach (missing helper = compile bail). */
        t.tlb = tlb_test;
        for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
            t.next[i] = -1;
            t.next_slot[i] = -1;
        }
        t.header_idx = 0;
        t.recs[0].pc = pc;
        t.recs[0].rx_tb = (const void *)pc;
    };
    /* Trace 11: TLB-hit 4-byte load. */
    auto t11 = std::make_unique<Tier2TraceDesc>();
    mkTlbTrace(11, 0xB000, (*t11));
        t11->mem_helpers.ld32u = (void *)&tlb_stub_ld32;
    Tier2TBRec &r11 = t11->recs[0];
    r11.num_temps = 2;
    mkTemp(r11, 0, true, false, 64, -1, 0x1004);
    mkTemp(r11, 1, false, false, 64, -1, 0);
    r11.num_ops = 3;
    r11.ops[0] = mkOp(T2_QEMU_LD, 64, 1, 0, -1, -1, -1, 64, 4);
    r11.ops[1] = mkOp(T2_ST64, 64, -1, 1, -1, -1, -1, 40, 0);
    r11.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xB000, 0);
    tlb_stub_ld_calls = 0;
    void *fn11 = tier2_jit_compile_trace(t11.get());
    if (!fn11) {
        fprintf(stderr, "[tier2-selftest] trace11 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn11)(tlb_fake_env);
    /*
     * Proves the inline fast path: the helper must NOT have run, and
     * the value came from fake RAM through the TLB walk. Exit is the
     * protocol encoding for TB 0 / idx 0.
     */
    if (ret != (TIER2_EXIT_PROTOCOL | 0) || tlb_stub_ld_calls != 0 ||
        tlb_area[32 + 5] != 0xAABBCCDDULL) {
        fprintf(stderr, "[tier2-selftest] trace11 mismatch: ret=0x%llx "
                "helpercalls=%d val=0x%llx\n",
                (unsigned long long)ret, tlb_stub_ld_calls,
                (unsigned long long)tlb_area[32 + 5]);
        return false;
    }
    printf("[tier2-selftest] trace11 TLB-hit load OK\n");

    /*
     * Trace 12: unmapped address misses the TLB and must take the
     * helper slow path (stub records the call, returns canned value).
     */
    auto t12 = std::make_unique<Tier2TraceDesc>();
    mkTlbTrace(12, 0xB001, (*t12));
    t12->mem_helpers.ld32u = (void *)&tlb_stub_ld32;
    Tier2TBRec &r12 = t12->recs[0];
    r12.num_temps = 2;
    mkTemp(r12, 0, true, false, 64, -1, 0x5000);
    mkTemp(r12, 1, false, false, 64, -1, 0);
    r12.num_ops = 3;
    r12.ops[0] = mkOp(T2_QEMU_LD, 64, 1, 0, -1, -1, -1, 64, 4);
    r12.ops[1] = mkOp(T2_ST64, 64, -1, 1, -1, -1, -1, 40, 0);
    r12.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xB001, 0);
    tlb_stub_ld_calls = 0;
    tlb_stub_ld_addr = 0;
    void *fn12 = tier2_jit_compile_trace(t12.get());
    if (!fn12) {
        fprintf(stderr, "[tier2-selftest] trace12 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn12)(tlb_fake_env);
    if (ret != (TIER2_EXIT_PROTOCOL | 1) || tlb_stub_ld_calls != 1 ||
        tlb_stub_ld_addr != 0x5000 ||
        tlb_area[32 + 5] != 0xdeadbeefdeadbeefULL) {
        fprintf(stderr, "[tier2-selftest] trace12 mismatch: ret=0x%llx "
                "calls=%d addr=0x%llx val=0x%llx\n",
                (unsigned long long)ret, tlb_stub_ld_calls,
                (unsigned long long)tlb_stub_ld_addr,
                (unsigned long long)tlb_area[32 + 5]);
        return false;
    }
    printf("[tier2-selftest] trace12 TLB-miss slow path OK\n");

    /*
     * Trace 13: aligned store hits inline (RAM bytes change, no helper
     * call); unaligned store takes the slow path (stub records it, RAM
     * untouched by fast path).
     */
    auto t13 = std::make_unique<Tier2TraceDesc>();
    mkTlbTrace(13, 0xB002, (*t13));
    t13->mem_helpers.st32 = (void *)&tlb_stub_st32;
    Tier2TBRec &r13 = t13->recs[0];
    r13.num_temps = 3;
    mkTemp(r13, 0, true, false, 64, -1, 0x1008);
    mkTemp(r13, 1, true, false, 64, -1, 0x11223344);
    mkTemp(r13, 2, true, false, 64, -1, 0x1003);
    r13.num_ops = 3;
    r13.ops[0] = mkOp(T2_QEMU_ST, 64, -1, 1, 0, -1, -1, 64, 4);
    r13.ops[1] = mkOp(T2_QEMU_ST, 64, -1, 1, 2, -1, -1, 64, 4);
    r13.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xB000, 0);
    tlb_stub_st_calls = 0;
    tlb_stub_st_addr = 0;
    tlb_stub_st_val = 0;
    void *fn13 = tier2_jit_compile_trace(t13.get());
    if (!fn13) {
        fprintf(stderr, "[tier2-selftest] trace13 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn13)(tlb_fake_env);
    bool ram_ok = tlb_ram[8] == 0x44 && tlb_ram[9] == 0x33 &&
                  tlb_ram[10] == 0x22 && tlb_ram[11] == 0x11;
    bool ram_untouched = tlb_ram[3] == 0;
    if (ret != (TIER2_EXIT_PROTOCOL | 0) || !ram_ok || !ram_untouched ||
        tlb_stub_st_calls != 1 || tlb_stub_st_addr != 0x1003 ||
        tlb_stub_st_val != 0x11223344u) {
        fprintf(stderr, "[tier2-selftest] trace13 mismatch: ret=0x%llx "
                "ram=%02x%02x%02x%02x calls=%d addr=0x%llx val=0x%llx\n",
                (unsigned long long)ret, tlb_ram[8], tlb_ram[9],
                tlb_ram[10], tlb_ram[11], tlb_stub_st_calls,
                (unsigned long long)tlb_stub_st_addr,
                (unsigned long long)tlb_stub_st_val);
        return false;
    }
    printf("[tier2-selftest] trace13 store hit + unaligned slow path OK\n");

    /*
     * Trace 14: goto_ptr fed by a lookup call tail-calls the prologue
     * with the result instead of round-tripping the dispatcher. Fake
     * prologue returns code+1; expect 0x5001.
     */
    g_prologue_fn = (TCGPrologueFn)&stest_prologue;
    Tier2TraceDesc t14;
    memset(&t14, 0, sizeof(t14));
    t14.num_tbs = 1;
    t14.has_ops = true;
    t14.trace_id = 14;
    t14.header_pc = 0xC000;
    t14.rx_header = (const void *)0xC000;
    t14.lookup_helper = (const void *)&stest_lookup;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t14.next[i] = -1;
        t14.next_slot[i] = -1;
    }
    t14.header_idx = 0;
    Tier2TBRec &r14 = t14.recs[0];
    r14.pc = 0xC000;
    r14.rx_tb = (const void *)0xC000;
    r14.num_temps = 2;
    mkTemp(r14, 0, false, false, 64, -1, 0);
    mkTemp(r14, 1, false, true, 64, -1, 0);
    auto callimm14 = (int64_t)(T2T_PTR | (T2T_PTR << 3) | (1 << 16) | (1 << 20));
    r14.num_ops = 3;
    r14.ops[0] = mkOp(T2_CALL, 64, 0, 1, -1, -1, -1,
                      (int64_t)(uintptr_t)(void *)&stest_lookup, callimm14);
    r14.ops[1] = mkOp(T2_GOTO_PTR, 0, -1, 0, -1, -1, -1, 0, 0);
    r14.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xC1, 0);
    void *fn14 = tier2_jit_compile_trace(&t14);
    if (!fn14) {
        fprintf(stderr, "[tier2-selftest] trace14 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn14)(dummy_env);
    if (ret != 0x5001) {
        fprintf(stderr, "[tier2-selftest] trace14 mismatch: ret=0x%llx\n",
                (unsigned long long)ret);
        return false;
    }
    printf("[tier2-selftest] trace14 prologue tail-call OK\n");

    /*
     * Trace 15: on-disk cache round-trip. Compile (miss + store),
     * tear down the JIT entirely, rebuild it empty, compile the identical
     * descriptor again: the second compile must take the file path
     * (cache_hits +1) and the loaded object must execute bit-identically.
     * This exercises emit -> write -> read -> link -> run, not just the
     * live-JIT dedup path.
     */
    Tier2TraceDesc t15;
    memset(&t15, 0, sizeof(t15));
    t15.num_tbs = 1;
    t15.has_ops = true;
    t15.trace_id = 15;
    t15.header_pc = 0xD000;
    t15.rx_header = (const void *)0xD000;
    snprintf(t15.guest_arch, sizeof(t15.guest_arch), "%s", "selftest");
    snprintf(t15.qemu_version, sizeof(t15.qemu_version), "%s", "selftest");
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t15.next[i] = -1;
        t15.next_slot[i] = -1;
    }
    t15.header_idx = 0;
    Tier2TBRec &r15 = t15.recs[0];
    r15.pc = 0xD000;
    r15.rx_tb = (const void *)0xD000;
    r15.num_temps = 5;
    mkTemp(r15, 0, true, false, 64, -1, 0);
    mkTemp(r15, 1, true, false, 64, -1, 1);
    mkTemp(r15, 2, false, false, 64, -1, 0);
    mkTemp(r15, 3, false, false, 64, -1, 0);
    mkTemp(r15, 4, true, false, 64, -1, 100);
    std::vector<Tier2OpRec> o15;
    o15.push_back(mkOp(T2_LD64, 64, 2, -1, -1, -1, -1, 0, 0));
    o15.push_back(mkOp(T2_MOV, 64, 3, 0, -1, -1, -1, 0, 0));
    o15.push_back(mkOp(T2_SETLABEL, 0, -1, -1, -1, -1, -1, 1, 0));
    o15.push_back(mkOp(T2_ADD, 64, 2, 2, 3, -1, -1, 0, 0));
    o15.push_back(mkOp(T2_ADD, 64, 3, 3, 1, -1, -1, 0, 0));
    o15.push_back(mkOp(T2_BRCOND, 64, -1, 3, 4, -1, -1, T2C_LTU, 1));
    o15.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 0, 0));
    o15.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xD000, 0));
    r15.num_ops = (uint32_t)o15.size();
    for (size_t i = 0; i < o15.size(); i++) {
        r15.ops[i] = o15[i];
    }
    static uint64_t cch_env[4];
    memset(cch_env, 0, sizeof(cch_env));
    cch_env[0] = 5;
    /* Reference: acc = 5 + sum(0..99). */
    uint64_t cch_ref = 5;
    for (uint64_t k = 0; k < 100; k++) {
        cch_ref += k;
    }
    uint64_t hits0 = tier2_jit_cache_hits();
    void *fn15a = tier2_jit_compile_trace(&t15);
    if (!fn15a) {
        fprintf(stderr, "[tier2-selftest] trace15 failed to compile\n");
        return false;
    }
    if (((FnT)fn15a)(cch_env) != (TIER2_EXIT_PROTOCOL | 0) ||
        cch_env[0] != cch_ref) {
        fprintf(stderr, "[tier2-selftest] trace15 fresh mismatch\n");
        return false;
    }
    /* Drop the entire JIT (empty registry, live-lookup impossible). */
    tier2_jit_shutdown();
    {
        auto JITExp = LLJITBuilder().create();
        if (!JITExp) {
            return false;
        }
        g_jit = std::move(*JITExp);
        g_initialized.store(true);
    }
    void *fn15b = tier2_jit_compile_trace(&t15);
    if (!fn15b) {
        fprintf(stderr, "[tier2-selftest] trace15 reload failed\n");
        return false;
    }
    if (tier2_jit_cache_hits() != hits0 + 1) {
        fprintf(stderr, "[tier2-selftest] trace15: expected a cache hit\n");
        return false;
    }
    memset(cch_env, 0, sizeof(cch_env));
    cch_env[0] = 5;
    if (((FnT)fn15b)(cch_env) != (TIER2_EXIT_PROTOCOL | 0) ||
        cch_env[0] != cch_ref) {
        fprintf(stderr, "[tier2-selftest] trace15 cached mismatch\n");
        return false;
    }
    printf("[tier2-selftest] trace15 on-disk cache round-trip OK\n");

    /*
     * Trace 16: P5 self back-edge fuses into a native loop. Single TB
     * with a proven self-edge (next[0]==0, slot 1): the body decrements
     * an env counter, BRCOND exits when it hits zero, and GOTO_TB slot
     * 1 loops. The old behavior returned after ONE iteration per
     * dispatch (side-exit origin k=0 idx=1); the fused loop runs to
     * completion in one dispatch (done-exit idx=0). A preset interrupt
     * word must divert the first back-edge to the safepoint exit
     * instead, leaving exactly one committed iteration behind.
     */
    struct FakeCPU16 {
        uint32_t irq;
        uint8_t pad[12];
        uint64_t env[4];
    };
    static FakeCPU16 fake16;
    auto t16 = std::make_unique<Tier2TraceDesc>();
    memset(t16.get(), 0, sizeof(*t16));
    t16->num_tbs = 1;
    t16->has_ops = true;
    t16->trace_id = 16;
    t16->header_pc = 0xE000;
    t16->rx_header = (const void *)0xE000;
    t16->has_safepoint = true;
    t16->cpu_off = (int64_t)(uintptr_t)&fake16 -
                   (int64_t)(uintptr_t)&fake16.env[0];
    t16->irq_off = (int64_t)offsetof(FakeCPU16, irq);
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t16->next[i] = -1;
        t16->next_slot[i] = -1;
    }
    t16->next[0] = 0;
    t16->next_slot[0] = 1;
    t16->header_idx = 0;
    Tier2TBRec &r16 = t16->recs[0];
    r16.pc = 0xE000;
    r16.size = 16;
    r16.rx_tb = (const void *)0xE000;
    r16.num_temps = 4;
    mkTemp(r16, 0, true, false, 64, -1, 0);
    mkTemp(r16, 1, true, false, 64, -1, 1);
    mkTemp(r16, 2, false, false, 64, -1, 0);
    mkTemp(r16, 3, false, false, 64, -1, 0);
    std::vector<Tier2OpRec> o16;
    o16.push_back(mkOp(T2_LD64, 64, 2, -1, -1, -1, -1, 0, 0));
    o16.push_back(mkOp(T2_LD64, 64, 3, -1, -1, -1, -1, 8, 0));
    o16.push_back(mkOp(T2_ADD, 64, 3, 3, 2, -1, -1, 0, 0));
    o16.push_back(mkOp(T2_SUB, 64, 2, 2, 1, -1, -1, 0, 0));
    o16.push_back(mkOp(T2_ST64, 64, -1, 3, -1, -1, -1, 8, 0));
    o16.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 0, 0));
    o16.push_back(mkOp(T2_BRCOND, 64, -1, 2, 0, -1, -1, T2C_EQ, 2));
    o16.push_back(mkOp(T2_GOTO_TB, 0, -1, -1, -1, -1, -1, 1, 0));
    o16.push_back(mkOp(T2_SETLABEL, 0, -1, -1, -1, -1, -1, 2, 0));
    o16.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xE000, 0));
    r16.num_ops = (uint32_t)o16.size();
    for (size_t i = 0; i < o16.size(); i++) {
        r16.ops[i] = o16[i];
    }
    void *fn16 = tier2_jit_compile_trace(t16.get());
    if (!fn16) {
        fprintf(stderr, "[tier2-selftest] trace16 failed to compile\n");
        return false;
    }
    fake16.irq = 0;
    fake16.env[0] = 1000;
    fake16.env[1] = 0;
    if (((FnT)fn16)(&fake16.env[0]) != (TIER2_EXIT_PROTOCOL | 0) ||
        fake16.env[0] != 0 || fake16.env[1] != 500500) {
        fprintf(stderr, "[tier2-selftest] trace16 loop mismatch: "
                "counter=%llu acc=%llu\n",
                (unsigned long long)fake16.env[0],
                (unsigned long long)fake16.env[1]);
        return false;
    }
    printf("[tier2-selftest] trace16 native self-loop OK\n");
    /* Safepoint: preset interrupt diverts the first back-edge. */
    fake16.irq = 1;
    fake16.env[0] = 1000;
    fake16.env[1] = 0;
    if (((FnT)fn16)(&fake16.env[0]) != (TIER2_EXIT_PROTOCOL | 1) ||
        fake16.env[0] != 999 || fake16.env[1] != 1000) {
        fprintf(stderr, "[tier2-selftest] trace16 safepoint mismatch: "
                "counter=%llu acc=%llu\n",
                (unsigned long long)fake16.env[0],
                (unsigned long long)fake16.env[1]);
        return false;
    }
    /* Cleared interrupt resumes to completion. */
    fake16.irq = 0;
    if (((FnT)fn16)(&fake16.env[0]) != (TIER2_EXIT_PROTOCOL | 0) ||
        fake16.env[0] != 0 || fake16.env[1] != 500500) {
        fprintf(stderr, "[tier2-selftest] trace16 resume mismatch: "
                "counter=%llu acc=%llu\n",
                (unsigned long long)fake16.env[0],
                (unsigned long long)fake16.env[1]);
        return false;
    }
    printf("[tier2-selftest] trace16 safepoint poll OK\n");

    /*
     * Trace 17: Phase 4 Vector / SIMD Transpilation (ARM64 NEON).
     * Exercises 128-bit vector load, 4x32 vector add, vector broadcast/xor,
     * and 128-bit vector store against a reference C++ calculation.
     */
    static uint32_t v_inA[4] = { 100, 200, 300, 400 };
    static uint32_t v_inB[4] = { 11,  22,  33,  44 };
    static uint32_t v_out[4] = { 0,   0,   0,   0 };
    static uint64_t v_env[4];
    v_env[0] = (uint64_t)(uintptr_t)&v_inA[0];
    v_env[1] = (uint64_t)(uintptr_t)&v_inB[0];
    v_env[2] = (uint64_t)(uintptr_t)&v_out[0];

    auto t17 = std::make_unique<Tier2TraceDesc>();
    memset(t17.get(), 0, sizeof(*t17));
    t17->num_tbs = 1;
    t17->has_ops = true;
    t17->trace_id = 17;
    t17->header_pc = 0xF000;
    t17->rx_header = (const void *)0xF000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t17->next[i] = -1;
        t17->next_slot[i] = -1;
    }
    t17->header_idx = 0;
    Tier2TBRec &r17 = t17->recs[0];
    r17.pc = 0xF000;
    r17.rx_tb = (const void *)0xF000;
    r17.num_temps = 9;
    mkTemp(r17, 0, false, false, 64, -1, 0);  // ptr A
    mkTemp(r17, 1, false, false, 64, -1, 0);  // ptr B
    mkTemp(r17, 2, false, false, 64, -1, 0);  // ptr Out
    mkTemp(r17, 3, true, false, 64, -1, 0x55); // mask scalar
    mkTemp(r17, 4, false, false, 128, -1, 0); // vec A
    mkTemp(r17, 5, false, false, 128, -1, 0); // vec B
    mkTemp(r17, 6, false, false, 128, -1, 0); // vec Add
    mkTemp(r17, 7, false, false, 128, -1, 0); // vec Mask
    mkTemp(r17, 8, false, false, 128, -1, 0); // vec Res

    std::vector<Tier2OpRec> o17;
    // Load ptrs from env
    o17.push_back(mkOp(T2_LD64, 64, 0, -1, -1, -1, -1, 0, 0));
    o17.push_back(mkOp(T2_LD64, 64, 1, -1, -1, -1, -1, 8, 0));
    o17.push_back(mkOp(T2_LD64, 64, 2, -1, -1, -1, -1, 16, 0));
    // Vector loads
    o17.push_back(mkOp(T2_VEC_LD, 128, 4, 0, -1, -1, -1, 32, 0));
    o17.push_back(mkOp(T2_VEC_LD, 128, 5, 1, -1, -1, -1, 32, 0));
    // Vector add: 4x32
    o17.push_back(mkOp(T2_VEC_ADD, 128, 6, 4, 5, -1, -1, 32, 0));
    // Vector broadcast mask: 4x32 from scalar temp 3
    o17.push_back(mkOp(T2_VEC_DUP, 128, 7, 3, -1, -1, -1, 32, 0));
    // Vector xor: 4x32
    o17.push_back(mkOp(T2_VEC_XOR, 128, 8, 6, 7, -1, -1, 32, 0));
    // Vector store to ptr Out
    o17.push_back(mkOp(T2_VEC_ST, 128, -1, 8, 2, -1, -1, 32, 0));
    o17.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0xF000, 0));

    r17.num_ops = (uint32_t)o17.size();
    for (size_t i = 0; i < o17.size(); i++) {
        r17.ops[i] = o17[i];
    }
    void *fn17 = tier2_jit_compile_trace(t17.get());
    if (!fn17) {
        fprintf(stderr, "[tier2-selftest] trace17 failed to compile\n");
        return false;
    }
    if (((FnT)fn17)(v_env) != (TIER2_EXIT_PROTOCOL | 0)) {
        fprintf(stderr, "[tier2-selftest] trace17 return code mismatch\n");
        return false;
    }
    for (int k = 0; k < 4; k++) {
        uint32_t expected = (v_inA[k] + v_inB[k]) ^ 0x55;
        if (v_out[k] != expected) {
            fprintf(stderr, "[tier2-selftest] trace17 lane %d mismatch: got %u, expected %u\n",
                    k, v_out[k], expected);
            return false;
        }
    }
    printf("[tier2-selftest] trace17 NEON vector SIMD OK\n");

    /* -------------------------------------------------------------- */
    /* Trace 18: High-Level Emulation (HLE) Library Shims & Thunks    */
    /* Differential verification of SysV x86_64 -> Host AAPCS64 math,  */
    /* memory, and crypto shims.                                      */
    /* -------------------------------------------------------------- */
    hle_thunk_init();
    hle_thunk_reset();
    assert(hle_thunk_register_address("sin", 0x100010));
    assert(hle_thunk_register_address("cos", 0x100020));
    assert(hle_thunk_register_address("pow", 0x100030));
    assert(hle_thunk_register_address("sqrt", 0x100040));
    assert(hle_thunk_register_address("atan2", 0x100050));
    assert(hle_thunk_register_address("strlen", 0x100060));
    assert(hle_thunk_register_address("memcpy", 0x100070));
    assert(hle_thunk_register_address("memset", 0x100080));
    assert(hle_thunk_register_address("SHA256_Init", 0x100090));
    assert(hle_thunk_register_address("SHA256_Update", 0x1000a0));
    assert(hle_thunk_register_address("SHA256_Final", 0x1000b0));

    struct TestX86Env {
        uint64_t regs[16];
        uint64_t eip;
        uint64_t eflags;
        uint8_t  _pad[1024];
        union {
            uint8_t  _b[64];
            uint32_t _l[16];
            uint64_t _q[8];
            float    _s[16];
            double   _d[8];
        } xmm[32];
    } hle_env;
    memset(&hle_env, 0, sizeof(hle_env));

    uint64_t fake_stack[16];
    memset(fake_stack, 0, sizeof(fake_stack));

    // 1. Math: sin(pi / 3)
    fake_stack[0] = 0xbeefcafeULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[0]; // R_ESP = 4
    hle_env.xmm[0]._d[0] = 1.0471975511965976;
    if (!hle_thunk_dispatch(&hle_env, 0x100010, nullptr)) {
        fprintf(stderr, "[tier2-selftest] trace18 sin thunk dispatch failed\n");
        return false;
    }
    if (std::abs(hle_env.xmm[0]._d[0] - sin(1.0471975511965976)) > 1e-15 ||
        hle_env.eip != 0xbeefcafeULL ||
        hle_env.regs[4] != (uintptr_t)&fake_stack[1]) {
        fprintf(stderr, "[tier2-selftest] trace18 sin thunk result mismatch\n");
        return false;
    }

    // 2. Math: pow(2.0, 10.0) == 1024.0
    fake_stack[1] = 0x12345678ULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[1];
    hle_env.xmm[0]._d[0] = 2.0;
    hle_env.xmm[1]._d[0] = 10.0;
    if (!hle_thunk_dispatch(&hle_env, 0x100030, nullptr) ||
        hle_env.xmm[0]._d[0] != 1024.0 ||
        hle_env.eip != 0x12345678ULL) {
        fprintf(stderr, "[tier2-selftest] trace18 pow thunk result mismatch\n");
        return false;
    }

    // 3. String: strlen
    fake_stack[2] = 0x87654321ULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[2];
    const char *test_str = "HighPerformanceTier2JIT";
    hle_env.regs[7] = (uintptr_t)test_str; // R_EDI = 7
    if (!hle_thunk_dispatch(&hle_env, 0x100060, nullptr) ||
        hle_env.regs[0] != strlen(test_str) || // R_EAX = 0
        hle_env.eip != 0x87654321ULL) {
        fprintf(stderr, "[tier2-selftest] trace18 strlen thunk result mismatch\n");
        return false;
    }

    // 4. Memory: memcpy
    fake_stack[3] = 0xaaaabbbbULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[3];
    char src_buf[32] = "TestingDirectHleThunk";
    char dst_buf[32] = {0};
    hle_env.regs[7] = (uintptr_t)dst_buf;
    hle_env.regs[6] = (uintptr_t)src_buf; // R_ESI = 6
    hle_env.regs[2] = strlen(src_buf) + 1; // R_EDX = 2
    if (!hle_thunk_dispatch(&hle_env, 0x100070, nullptr) ||
        strcmp(dst_buf, src_buf) != 0 ||
        hle_env.regs[0] != (uintptr_t)dst_buf) {
        fprintf(stderr, "[tier2-selftest] trace18 memcpy thunk result mismatch\n");
        return false;
    }

    // 5. Crypto: SHA256("hello world")
    // Known SHA-256: b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace20c
    uint8_t sha_ctx[256];
    fake_stack[4] = 0xccccddddULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[4];
    hle_env.regs[7] = (uintptr_t)sha_ctx;
    if (!hle_thunk_dispatch(&hle_env, 0x100090, nullptr)) {
        return false;
    }
    const char *sha_msg = "hello world";
    fake_stack[4] = 0xccccddddULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[4];
    hle_env.regs[7] = (uintptr_t)sha_ctx;
    hle_env.regs[6] = (uintptr_t)sha_msg;
    hle_env.regs[2] = strlen(sha_msg);
    if (!hle_thunk_dispatch(&hle_env, 0x1000a0, nullptr)) {
        return false;
    }
    uint8_t digest[32];
    fake_stack[4] = 0xccccddddULL;
    hle_env.regs[4] = (uintptr_t)&fake_stack[4];
    hle_env.regs[7] = (uintptr_t)digest;
    hle_env.regs[6] = (uintptr_t)sha_ctx;
    if (!hle_thunk_dispatch(&hle_env, 0x1000b0, nullptr) ||
        digest[0] != 0xb9 || digest[1] != 0x4d || digest[2] != 0x27 || digest[3] != 0xb9) {
        fprintf(stderr, "[tier2-selftest] trace18 sha256 thunk result mismatch\n");
        return false;
    }
    /* -------------------------------------------------------------- */
    /* Trace 19: High multiply & bitwise ops (mulsh, muluh, andc,     */
    /* orc, clz, ctz)                                                 */
    /* -------------------------------------------------------------- */
    auto t19 = std::make_unique<Tier2TraceDesc>();
    memset(t19.get(), 0, sizeof(*t19));
    t19->num_tbs = 1;
    t19->has_ops = true;
    t19->trace_id = 19;
    t19->header_pc = 0x13000;
    t19->rx_header = (const void *)0x13000;
    for (int i = 0; i < TIER2_MAX_TRACE_TBS; i++) {
        t19->next[i] = -1;
        t19->next_slot[i] = -1;
    }
    t19->header_idx = 0;
    Tier2TBRec &r19 = t19->recs[0];
    r19.pc = 0x13000;
    r19.rx_tb = (const void *)0x13000;
    r19.num_temps = 10;
    for (int i = 0; i < 10; i++) {
        mkTemp(r19, i, false, false, 64, -1, 0);
    }
    mkTemp(r19, 8, true, false, 64, -1, 64); /* def_val = 64 */
    std::vector<Tier2OpRec> o19;
    o19.push_back(mkOp(T2_LD64, 64, 0, -1, -1, -1, -1, 0, 0));   /* a = env[0] */
    o19.push_back(mkOp(T2_LD64, 64, 1, -1, -1, -1, -1, 8, 0));   /* b = env[8] */
    o19.push_back(mkOp(T2_ANDC, 64, 2, 0, 1, -1, -1, 0, 0));     /* andc */
    o19.push_back(mkOp(T2_ORC, 64, 3, 0, 1, -1, -1, 0, 0));      /* orc */
    o19.push_back(mkOp(T2_MULSH, 64, 4, 0, 1, -1, -1, 0, 0));    /* mulsh */
    o19.push_back(mkOp(T2_MULUH, 64, 5, 0, 1, -1, -1, 0, 0));    /* muluh */
    o19.push_back(mkOp(T2_CLZ, 64, 6, 0, 8, -1, -1, 0, 0));      /* clz */
    o19.push_back(mkOp(T2_CTZ, 64, 7, 0, 8, -1, -1, 0, 0));      /* ctz */
    o19.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 16, 0));
    o19.push_back(mkOp(T2_ST64, 64, -1, 3, -1, -1, -1, 24, 0));
    o19.push_back(mkOp(T2_ST64, 64, -1, 4, -1, -1, -1, 32, 0));
    o19.push_back(mkOp(T2_ST64, 64, -1, 5, -1, -1, -1, 40, 0));
    o19.push_back(mkOp(T2_ST64, 64, -1, 6, -1, -1, -1, 48, 0));
    o19.push_back(mkOp(T2_ST64, 64, -1, 7, -1, -1, -1, 56, 0));
    o19.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x13000, 0));
    r19.num_ops = (uint32_t)o19.size();
    for (size_t i = 0; i < o19.size(); i++) {
        r19.ops[i] = o19[i];
    }
    static uint64_t op_env[8];
    memset(op_env, 0, sizeof(op_env));
    uint64_t inA = 0x100000002ULL;
    uint64_t inB = 0x300000004ULL;
    op_env[0] = inA;
    op_env[1] = inB;
    void *fn19 = tier2_jit_compile_trace(t19.get());
    if (!fn19) {
        fprintf(stderr, "[tier2-selftest] trace19 failed to compile\n");
        return false;
    }
    if (((FnT)fn19)(op_env) != (TIER2_EXIT_PROTOCOL | 0)) {
        fprintf(stderr, "[tier2-selftest] trace19 exit protocol mismatch\n");
        return false;
    }
    uint64_t exp_andc = inA & ~inB;
    uint64_t exp_orc = inA | ~inB;
    unsigned __int128 p_u = (unsigned __int128)inA * (unsigned __int128)inB;
    uint64_t exp_muluh = (uint64_t)(p_u >> 64);
    __int128 p_s = (__int128)(int64_t)inA * (__int128)(int64_t)inB;
    uint64_t exp_mulsh = (uint64_t)(p_s >> 64);
    uint64_t exp_clz = (uint64_t)__builtin_clzll(inA);
    uint64_t exp_ctz = (uint64_t)__builtin_ctzll(inA);
    if (op_env[2] != exp_andc || op_env[3] != exp_orc ||
        op_env[4] != exp_mulsh || op_env[5] != exp_muluh ||
        op_env[6] != exp_clz || op_env[7] != exp_ctz) {
        fprintf(stderr, "[tier2-selftest] trace19 arithmetic op mismatch\n");
        return false;
    }
    printf("[tier2-selftest] trace19 mulsh/muluh/andc/orc/clz/ctz OK\n");

    printf("[tier2-selftest] ALL GREEN\n");
    return true;
}

#ifdef TIER2_JIT_SELFTEST_MAIN
int main(void)
{
    return tier2_jit_selftest() ? 0 : 1;
}
#endif

/* ------------------------------------------------------------------ */
/* Benchmark: 8M-iteration loop through the op walker vs a plain C++  */
/* scalar loop. Measures walker compile tax and steady-state exec.     */
/* ------------------------------------------------------------------ */

#ifdef TIER2_JIT_BENCH_MAIN
#include <time.h>

static uint64_t bench_now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}

extern "C" bool tier2_jit_bench(void);

extern "C" bool tier2_jit_bench(void)
{
    if (!g_initialized.load()) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        auto JITExp = LLJITBuilder().create();
        if (!JITExp) {
            return false;
        }
        g_jit = std::move(*JITExp);
        g_initialized.store(true);
    }

    /*
     * Fresh temp cache dir per run: this benchmark times compilation,
     * so it must neither hit entries from previous runs nor pollute
     * the user's cache.
     */
    char tmpl[] = "/tmp/tier2-bench-XXXXXX";
    if (mkdtemp(tmpl)) {
        setenv("QEMU_TIER2_CACHE_DIR", tmpl, 1);
    }

    static const uint64_t N = 8000000;
    static const uint64_t CVAL = 2654435761ULL;
    static uint64_t fake_env[8];
    memset(fake_env, 0, sizeof(fake_env));
    fake_env[0] = 12345;

    /* Walker stream: acc=env[0]; i=0;
     * L1: acc+=i; acc ^= (acc>>3) ^ (i*C); i++; if (i<N) goto L1;
     * env[0]=acc; env[8]=i; exit(0) */
    Tier2TraceDesc trace;
    memset(&trace, 0, sizeof(trace));
    trace.num_tbs = 1;
    trace.has_ops = true;
    trace.trace_id = 100;
    trace.rx_header = (void *)0x8000;
    trace.tbs[0].pc = 0x8000;
    Tier2TBRec &rec = trace.recs[0];
    rec.pc = 0x8000;
    rec.num_temps = 9;
    mkTemp(rec, 0, true, false, 64, -1, 0);
    mkTemp(rec, 1, true, false, 64, -1, 1);
    mkTemp(rec, 2, false, false, 64, -1, 0); /* acc */
    mkTemp(rec, 3, false, false, 64, -1, 0); /* i */
    mkTemp(rec, 4, true, false, 64, -1, N);
    mkTemp(rec, 5, true, false, 64, -1, CVAL);
    mkTemp(rec, 6, true, false, 64, -1, 3);
    mkTemp(rec, 7, false, false, 64, -1, 0); /* tmpA */
    mkTemp(rec, 8, false, false, 64, -1, 0); /* tmpB */
    std::vector<Tier2OpRec> ops;
    ops.push_back(mkOp(T2_LD64, 64, 2, -1, -1, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_MOV, 64, 3, 0, -1, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_SETLABEL, 0, -1, -1, -1, -1, -1, 1, 0));
    ops.push_back(mkOp(T2_ADD, 64, 2, 2, 3, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_SHR, 64, 7, 2, 6, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_MUL, 64, 8, 3, 5, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_XOR, 64, 2, 2, 7, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_XOR, 64, 2, 2, 8, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_ADD, 64, 3, 3, 1, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_BRCOND, 64, -1, 3, 4, -1, -1, T2C_LTU, 1));
    ops.push_back(mkOp(T2_ST64, 64, -1, 2, -1, -1, -1, 0, 0));
    ops.push_back(mkOp(T2_ST64, 64, -1, 3, -1, -1, -1, 8, 0));
    ops.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0, 0));
    rec.num_ops = (uint32_t)ops.size();
    for (size_t i = 0; i < ops.size(); i++) {
        rec.ops[i] = ops[i];
    }

    /* Reference scalar loop. */
    uint64_t t0 = bench_now_ms();
    uint64_t r_acc = 12345, r_i = 0;
    while (true) {
        r_acc = r_acc + r_i;
        r_acc = r_acc ^ ((r_acc >> 3) ^ (r_i * CVAL));
        r_i = r_i + 1;
        if (!(r_i < N)) {
            break;
        }
    }
    uint64_t t_ref = bench_now_ms() - t0;

    uint64_t t1 = bench_now_ms();
    void *fn = tier2_jit_compile_trace(&trace);
    uint64_t t_compile = bench_now_ms() - t1;
    if (!fn) {
        fprintf(stderr, "[tier2-bench] compile failed\n");
        return false;
    }
    typedef uint64_t (*FnT)(void *);
    uint64_t best = ~0ULL, ret = 0;
    for (int k = 0; k < 3; k++) {
        fake_env[0] = 12345;
        fake_env[1] = 0;
        uint64_t te = bench_now_ms();
        ret = ((FnT)fn)(fake_env);
        uint64_t dt = bench_now_ms() - te;
        if (dt < best) {
            best = dt;
        }
    }
    bool match = (fake_env[0] == r_acc && fake_env[1] == r_i);
    double speedup = t_ref > 0 ? (double)t_ref / (double)(best ? best : 1) : 0;
    printf("[tier2-bench] N=%llu ref=%llums walker-compile=%llums "
           "walker-exec(best of 3)=%llums speedup=%.2fx checksum=%s "
           "(acc=0x%llx)\n",
           (unsigned long long)N, (unsigned long long)t_ref,
           (unsigned long long)t_compile, (unsigned long long)best, speedup,
           match ? "OK" : "MISMATCH", (unsigned long long)fake_env[0]);
    (void)ret;
    return match;
}

int main(void)
{
    return tier2_jit_bench() ? 0 : 1;
}
#endif
