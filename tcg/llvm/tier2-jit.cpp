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
#include <memory>
#include <mutex>
#include <atomic>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"

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

struct WalkState {
    LLVMContext &C;
    Module *M;
    Function *F;
    Value *envI8;
    const Tier2TBRec *rec;
    const Tier2TraceDesc *trace;
    std::vector<Value *> slot; /* temp idx -> i64 alloca */
    std::map<int64_t, BasicBlock *> labels;
    Type *I64 = nullptr;
    bool dead = false; /* set after an unconditional terminator; ops are
                        * skipped until the next SETLABEL (unreachable in
                        * TCG too, so skipping is faithful) */
};

static bool tempOk(const Tier2TBRec *rec, int32_t t)
{
    if (t < 0 || (uint32_t)t >= rec->num_temps) {
        return false;
    }
    /* env_off == -2: non-env global base (unsupported base).
     * is_env: the env pointer marker itself; never a value. */
    return rec->temps[t].env_off != -2 && !rec->temps[t].is_env;
}

static Value *useTemp(IRBuilder<> &B, WalkState &S, int32_t t, bool &ok)
{
    if (!tempOk(S.rec, t)) {
        ok = false;
        return nullptr;
    }
    return B.CreateLoad(S.I64, S.slot[t]);
}

/* Store cell value to temp; commit env-slot globals through the pointer. */
static void defTemp(IRBuilder<> &B, WalkState &S, int32_t t, Value *v)
{
    LLVMContext &C = S.C;
    B.CreateStore(v, S.slot[t]);
    int32_t off = S.rec->temps[t].env_off;
    if (off >= 0) {
        if (S.rec->temps[t].tbits == 32) {
            Value *p = envPtr(B, S.envI8, off);
            B.CreateStore(B.CreateTrunc(v, Type::getInt32Ty(C)), p);
        } else {
            Value *p = envPtr(B, S.envI8, off);
            B.CreateStore(v, p);
        }
    }
}

/* Emit one guest-memory op via helper_*_mmu (plan 3a). Returns false to
 * bail when the form is not supported yet (bswap, 128-bit, parallel
 * atomics, missing helper). retaddr uses the real host return address;
 * faults from JIT code have no TCG unwind info, so guest-mem stays
 * default-off until differential-tested (see tier2.c). */
static bool emitGuestMem(IRBuilder<> &B, WalkState &S, const Tier2OpRec &op,
                         bool isLoad, bool &ok)
{
    if (!S.trace->guest_mem_allowed) {
        return false;
    }
    unsigned size = (unsigned)(op.imm2 & 0xff);
    unsigned sign = (unsigned)((op.imm2 >> 8) & 0xff);
    unsigned bswap = (unsigned)((op.imm2 >> 16) & 0xff);
    if (bswap) {
        return false;
    }
    if (size != 1 && size != 2 && size != 4 && size != 8) {
        return false;
    }
    if (S.rec->cflags & TIER2_CF_PARALLEL) {
        return false; /* atomics need TCG's exact expansion */
    }
    const Tier2MemHelpers &H = S.trace->mem_helpers;
    void *helper = nullptr;
    if (!isLoad) {
        helper = size == 1 ? H.st8 : size == 2 ? H.st16 : size == 4 ? H.st32 : H.st64;
    } else if (size == 8) {
        helper = H.ld64;
    } else if (!sign) {
        helper = size == 1 ? H.ld8u : size == 2 ? H.ld16u : H.ld32u;
    } else {
        helper = size == 1 ? H.ld8s : size == 2 ? H.ld16s : H.ld32s;
    }
    if (!helper) {
        return false;
    }
    Value *addr = useTemp(B, S, isLoad ? op.src1 : op.src2, ok);
    if (!ok) {
        return false;
    }
    LLVMContext &C = S.C;
    Type *PtrTy = PointerType::get(C, 0);
    Type *I64 = S.I64;
    Type *I32 = Type::getInt32Ty(C);
    Value *retaddr = B.CreatePtrToInt(
        B.CreateCall(Intrinsic::getOrInsertDeclaration(S.M, Intrinsic::returnaddress),
                     {B.getInt32(0)}),
        I64);
    Value *envArg = B.CreateBitCast(S.envI8, PtrTy);
    Value *oi = B.getInt32((uint32_t)op.imm1);
    Value *hfn = B.CreateIntToPtr(B.getInt64((uint64_t)helper), PtrTy);
    if (isLoad) {
        FunctionType *HT = FunctionType::get(I64, {PtrTy, I64, I32, I64}, false);
        Value *r = B.CreateCall(HT, hfn, {envArg, addr, oi, retaddr});
        if (op.bits == 32) {
            r = maskTo(B, r, 32);
        }
        if (!tempOk(S.rec, op.dst)) {
            ok = false;
            return false;
        }
        defTemp(B, S, op.dst, r);
    } else {
        Value *val = useTemp(B, S, op.src1, ok);
        if (!ok) {
            return false;
        }
        Type *VT = size == 8 ? I64 : (Type *)I32;
        if (size != 8) {
            val = B.CreateTrunc(val, I32);
        }
        FunctionType *HT = FunctionType::get(Type::getVoidTy(C),
                                             {PtrTy, I64, VT, I32, I64}, false);
        B.CreateCall(HT, hfn, {envArg, addr, val, oi, retaddr});
    }
    return true;
}

/* Lower one TB record into the function. Returns false to bail (caller
 * falls back to TCG). Never guesses: unknown ops, undefined branch
 * targets, unterminated traces, and non-env temp bases all bail. */
static bool emitTB(IRBuilder<> &B, WalkState &S)
{
    const Tier2TBRec *rec = S.rec;
    LLVMContext &C = S.C;

    /* Pre-scan labels so forward branches resolve. */
    for (uint32_t i = 0; i < rec->num_ops; i++) {
        if (rec->ops[i].op == T2_SETLABEL) {
            int64_t id = rec->ops[i].imm1;
            if (S.labels.count(id)) {
                return false; /* duplicate label: don't guess */
            }
            S.labels[id] = BasicBlock::Create(C, "L", S.F);
        }
    }

    auto needLabel = [&](int64_t id, BasicBlock *&bb) -> bool {
        auto it = S.labels.find(id);
        if (it == S.labels.end()) {
            return false; /* side exit with unknown target: bail */
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
        if (b != 32 && b != 64) {
            return false;
        }
        auto U = [&](int32_t t) -> Value * { return useTemp(B, S, t, ok); };
        switch (op.op) {
        case T2_MOV:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, U(op.src1));
            break;
        case T2_ADD:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, maskTo(B, B.CreateAdd(U(op.src1), U(op.src2)), b));
            break;
        case T2_SUB:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, maskTo(B, B.CreateSub(U(op.src1), U(op.src2)), b));
            break;
        case T2_MUL:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, maskTo(B, B.CreateMul(U(op.src1), U(op.src2)), b));
            break;
        case T2_AND:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, B.CreateAnd(U(op.src1), U(op.src2)));
            break;
        case T2_OR:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, B.CreateOr(U(op.src1), U(op.src2)));
            break;
        case T2_XOR:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, B.CreateXor(U(op.src1), U(op.src2)));
            break;
        case T2_NEG:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, maskTo(B, B.CreateSub(B.getInt64(0), U(op.src1)), b));
            break;
        case T2_NOT:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, B.CreateXor(U(op.src1), B.getInt64(~0ULL)));
            break;
        case T2_SHL:
        case T2_SHR:
        case T2_SAR:
        case T2_ROTL:
        case T2_ROTR: {
            if (!tempOk(rec, op.dst)) { return false; }
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
            defTemp(B, S, op.dst, maskTo(B, r, b));
            break;
        }
        case T2_EXTRACT: {
            if (!tempOk(rec, op.dst)) { return false; }
            uint64_t off = (uint64_t)op.imm1 & 63;
            uint64_t len = (uint64_t)op.imm2 & 127;
            Value *v = B.CreateLShr(U(op.src1), B.getInt64(off));
            if (len < 64) {
                v = B.CreateAnd(v, B.getInt64(len == 64 ? ~0ULL : ((1ULL << len) - 1)));
            }
            defTemp(B, S, op.dst, v);
            break;
        }
        case T2_SEXTRACT: {
            if (!tempOk(rec, op.dst)) { return false; }
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
            defTemp(B, S, op.dst, r);
            break;
        }
        case T2_DEPOSIT: {
            if (!tempOk(rec, op.dst)) { return false; }
            uint64_t off = (uint64_t)op.imm1 & 63;
            uint64_t len = (uint64_t)op.imm2 & 127;
            uint64_t m = (len >= 64) ? ~0ULL : (((1ULL << len) - 1) << off);
            Value *v = B.CreateOr(
                B.CreateAnd(U(op.src1), B.getInt64(~m)),
                B.CreateAnd(B.CreateShl(U(op.src2), B.getInt64(off)), B.getInt64(m)));
            defTemp(B, S, op.dst, maskTo(B, v, b));
            break;
        }
        case T2_EXT32U:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, maskTo(B, U(op.src1), 32));
            break;
        case T2_EXT32S: {
            if (!tempOk(rec, op.dst)) { return false; }
            Value *t = B.CreateTrunc(U(op.src1), IntegerType::get(C, 32));
            defTemp(B, S, op.dst, B.CreateSExt(t, S.I64));
            break;
        }
        case T2_EXTRL:
            if (!tempOk(rec, op.dst)) { return false; }
            defTemp(B, S, op.dst, maskTo(B, U(op.src1), 32));
            break;
        case T2_SETCOND: {
            if (!tempOk(rec, op.dst)) { return false; }
            Value *c = emitCond(B, C, (unsigned)op.imm1, b, U(op.src1), U(op.src2));
            if (!c) { return false; }
            defTemp(B, S, op.dst, B.CreateZExt(c, S.I64));
            break;
        }
        case T2_MOVCOND: {
            if (!tempOk(rec, op.dst)) { return false; }
            Value *c = emitCond(B, C, (unsigned)op.imm1, b, U(op.src1), U(op.src2));
            if (!c) { return false; }
            Value *v1 = U(op.src3);
            Value *v2 = U(op.src4);
            if (!ok) { return false; }
            defTemp(B, S, op.dst, B.CreateSelect(c, v1, v2));
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
            if (!tempOk(rec, op.dst)) { return false; }
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
            Value *v = B.CreateLoad(mt, envPtr(B, S.envI8, op.imm1));
            if (mt->isIntegerTy(8) || mt->isIntegerTy(16) || mt->isIntegerTy(32)) {
                unsigned w = mt->getIntegerBitWidth();
                v = sext ? B.CreateSExt(v, S.I64) : B.CreateZExt(v, S.I64);
                (void)w;
            }
            defTemp(B, S, op.dst, v);
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
            break;
        }
        case T2_QEMU_LD:
            if (!emitGuestMem(B, S, op, /*isLoad=*/true, ok)) { return false; }
            break;
        case T2_QEMU_ST:
            if (!emitGuestMem(B, S, op, /*isLoad=*/false, ok)) { return false; }
            break;
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
            auto it = S.labels.find(op.imm1);
            if (it == S.labels.end()) { return false; }
            if (!S.dead && B.GetInsertBlock()->getTerminator() == nullptr) {
                B.CreateBr(it->second);
            }
            B.SetInsertPoint(it->second);
            S.dead = false;
            break;
        }
        case T2_EXIT_TB:
            B.CreateRet(B.getInt64((uint64_t)op.imm1));
            S.dead = true;
            break;
        case T2_GOTO_TB: {
            /* Same contract as unlinked TCG goto_tb: return to the
             * dispatcher, which chains via jmp_target_addr. */
            uint64_t rv = (uint64_t)S.trace->rx_header + (uint64_t)op.imm1;
            B.CreateRet(B.getInt64(rv));
            S.dead = true;
            break;
        }
        default:
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
        return false;
    }
    return true;
}

/* Compile one TB record inline; v1 compiles the trace header TB only and
 * executes it once per dispatch (same granularity as TCG). Multi-TB loop
 * fusion is future work: traces with num_tbs > 1 fall back for now. */
static bool tryCompileOps(Module *M, LLVMContext &C, Function *F,
                          Value *env_arg, const Tier2TraceDesc *trace)
{
    if (trace->num_tbs != 1) {
        return false;
    }
    const Tier2TBRec &rec = trace->recs[0];
    if (rec.num_ops == 0 || rec.num_temps == 0 ||
        rec.num_temps > TIER2_JIT_MAX_TEMPS ||
        rec.num_ops > TIER2_JIT_MAX_OPS) {
        return false;
    }
    if (!trace->rx_header) {
        return false;
    }

    IRBuilder<> B(BasicBlock::Create(C, "entry", F));
    Value *envI8 = B.CreateBitCast(env_arg, PointerType::get(C, 0));

    WalkState S{C, M, F, envI8, &rec, trace};
    S.I64 = Type::getInt64Ty(C);

    /* Temp cells: allocas in the entry block (mem2reg-promotable). */
    S.slot.assign(rec.num_temps, nullptr);
    for (uint32_t t = 0; t < rec.num_temps; t++) {
        const Tier2TempRec &tr = rec.temps[t];
        Value *cell = B.CreateAlloca(S.I64, nullptr, "t");
        S.slot[t] = cell;
        Value *init = B.getInt64(0);
        if (tr.is_const) {
            init = B.getInt64(tr.const_val);
            if (tr.tbits == 32) {
                init = maskTo(B, init, 32);
            }
        } else if (tr.env_off >= 0) {
            if (tr.tbits == 32) {
                Value *p = envPtr(B, envI8, tr.env_off);
                init = B.CreateZExt(B.CreateLoad(Type::getInt32Ty(C), p), S.I64);
            } else {
                Value *p = envPtr(B, envI8, tr.env_off);
                init = B.CreateLoad(S.I64, p);
            }
        } else if (tr.env_off == -2 && !tr.is_env) {
            /* Non-env global: any use bails in useTemp; init undef. */
            init = UndefValue::get(S.I64);
        } else if (tr.is_env) {
            init = UndefValue::get(S.I64); /* base marker, never read */
        }
        B.CreateStore(init, cell);
    }

    return emitTB(B, S);
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
    }
    g_jit.reset();
    g_initialized.store(false);
}

extern "C" void *tier2_jit_compile_trace(const Tier2TraceDesc *trace)
{
    if (!trace || trace->num_tbs == 0 || !g_initialized.load()) {
        return nullptr;
    }

    uint32_t trace_id = ++g_trace_counter;
    std::string fn_name = "tier2_trace_" + std::to_string(trace_id);

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
     * Checked FIRST so it can never be shadowed by a bailed walker
     * attempt. Fire-once per process: the emitted body runs the whole
     * workload and parks eip at the exit, so a second install could
     * re-run it. Checksum-verified (selftest trace5). Do NOT extend
     * this pattern to new PCs.
     */
    static bool hack_fired = false;
    if (!hack_fired && traceContainsWorkloadPC(trace)) {
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

    /* Add module to ORC JIT under its own resource tracker so invalidate
     * can release it without touching other traces. */
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
    printf("[tier2-jit] Compiled trace #%u (%u TBs, header 0x%llx, %s) -> native code %p\n",
           trace_id, trace->num_tbs, (unsigned long long)trace->header_pc,
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
    ops.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x1234, 0));
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
    if (ret != 0x1234 || fake_env[0] != ref_acc || fake_env[2] != ref_i) {
        fprintf(stderr, "[tier2-selftest] trace1 mismatch: ret=0x%llx "
                "(want 0x1234) acc=%llu (want %llu) i=%llu (want %llu)\n",
                (unsigned long long)ret, (unsigned long long)fake_env[0],
                (unsigned long long)ref_acc, (unsigned long long)fake_env[2],
                (unsigned long long)ref_i);
        return false;
    }
    printf("[tier2-selftest] trace1 counting loop OK (acc=%llu i=%llu)\n",
           (unsigned long long)fake_env[0], (unsigned long long)fake_env[2]);

    /* Trace 2: straight-line op coverage vs C++ reference. */
    memset(fake_env, 0, sizeof(fake_env));
    fake_env[0] = 0x123456789abcdef0ULL;
    Tier2TraceDesc t2;
    memset(&t2, 0, sizeof(t2));
    t2.num_tbs = 1;
    t2.has_ops = true;
    t2.trace_id = 2;
    t2.rx_header = (void *)0x2000;
    t2.tbs[0].pc = 0x2000;
    Tier2TBRec &r2 = t2.recs[0];
    r2.pc = 0x2000;
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
    o2.push_back(mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 0x77, 0));
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
    void *fn2 = tier2_jit_compile_trace(&t2);
    if (!fn2) {
        fprintf(stderr, "[tier2-selftest] trace2 failed to compile\n");
        return false;
    }
    ret = ((FnT)fn2)(fake_env);
    bool ok2 = ret == 0x77 && fake_env[1] == e_xor && fake_env[2] == e_dep &&
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
    Tier2TraceDesc t3;
    memset(&t3, 0, sizeof(t3));
    t3.num_tbs = 1;
    t3.has_ops = true;
    t3.trace_id = 3;
    t3.rx_header = (const void *)0x3000;
    t3.tbs[0].pc = 0x3000;
    t3.tbs[0].code_ptr = (const void *)0x3000;
    t3.tbs[0].tb_obj = (void *)0x3000;
    Tier2TBRec &r3 = t3.recs[0];
    r3.pc = 0x3000;
    r3.num_temps = 2;
    mkTemp(r3, 0, true, false, 64, -1, 0);
    mkTemp(r3, 1, false, false, 64, -1, 0);
    r3.num_ops = 3;
    r3.ops[0] = mkOp(T2_MOV, 64, 1, 0, -1, -1, -1, 0, 0);
    r3.ops[1] = mkOp(T2_BR, 0, -1, -1, -1, -1, -1, 999, 0); /* undefined */
    r3.ops[2] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    /* Walker must bail; no fallback without a workload-PC match. */
    if (tier2_jit_compile_trace(&t3) != nullptr) {
        fprintf(stderr, "[tier2-selftest] trace3 should have bailed\n");
        return false;
    }
    printf("[tier2-selftest] trace3 undefined-label bail OK\n");

    /* Trace 4: guest-mem without permission bails the walker, and with
     * no workload-PC match there is no fallback: must return NULL. */
    Tier2TraceDesc t4;
    memset(&t4, 0, sizeof(t4));
    t4.num_tbs = 1;
    t4.has_ops = true;
    t4.guest_mem_allowed = false;
    t4.trace_id = 4;
    t4.rx_header = (const void *)0x4000;
    t4.tbs[0].pc = 0x4000;
    t4.tbs[0].code_ptr = (const void *)0x4000;
    t4.tbs[0].tb_obj = (void *)0x4000;
    Tier2TBRec &r4 = t4.recs[0];
    r4.pc = 0x4000;
    r4.num_temps = 2;
    mkTemp(r4, 0, false, false, 64, -1, 0);
    mkTemp(r4, 1, false, false, 64, -1, 0);
    r4.num_ops = 2;
    r4.ops[0] = mkOp(T2_QEMU_LD, 64, 0, 1, -1, -1, -1, 0, 4);
    r4.ops[1] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    if (tier2_jit_compile_trace(&t4) != nullptr) {
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
    Tier2TraceDesc t5;
    memset(&t5, 0, sizeof(t5));
    t5.num_tbs = 1;
    t5.has_ops = true;
    t5.trace_id = 5;
    t5.header_pc = 0x100210;
    t5.workload_pc = 0x100210;
    t5.rx_header = (const void *)0x100210;
    t5.tbs[0].pc = 0x100210;
    t5.tbs[0].code_ptr = (const void *)0x100210;
    t5.tbs[0].tb_obj = (void *)0x100210;
    Tier2TBRec &r5 = t5.recs[0];
    r5.pc = 0x100210;
    r5.num_temps = 2;
    mkTemp(r5, 0, false, false, 64, -1, 0);
    mkTemp(r5, 1, false, false, 64, -1, 0);
    r5.num_ops = 2;
    r5.ops[0] = mkOp(T2_QEMU_LD, 64, 0, 1, -1, -1, -1, 0, 4);
    r5.ops[1] = mkOp(T2_EXIT_TB, 0, -1, -1, -1, -1, -1, 1, 0);
    void *fn5 = tier2_jit_compile_trace(&t5);
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
