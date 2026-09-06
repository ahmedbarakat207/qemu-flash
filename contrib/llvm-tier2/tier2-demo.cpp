// Tier-2 prototype: the dbc-bench hot loop as LLVM IR, two codegen styles.
//
// Variant A (trace style, HQEMU-like): guest state lives in SSA values,
// committed nowhere inside the loop -- valid because a trace has a single
// entry and state is only observable at trace exits.
//
// Variant B (TCG style): the accumulator is committed to an "env" slot on
// every op, like TCG's per-instruction state sync. LLVM's mem2reg/GVN/LICM
// then clean it up -- the optimizer dividend.
//
// Both keep faithful control flow (alternating diamond, rare path) and a
// true indirect call per iteration. Checksum must equal the TCG run
// (0x147ce5ff) or the model is wrong.

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <array>
#include <string>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace llvm::orc;

static const uint32_t ITERS = 8000000;

static FunctionType *binopTy(LLVMContext &C)
{
    Type *I32 = Type::getInt32Ty(C);
    return FunctionType::get(I32, {I32, I32}, false);
}

// acc += table[acc % 3](acc, i), with acc in SSA or env per ENV.
static Value *emitIndirect(IRBuilder<> &B, LLVMContext &C, Value *acc, Value *i,
                           GlobalVariable *Table, GlobalVariable *Env)
{
    Type *I32 = Type::getInt32Ty(C);
    auto accAddr = [&]() {
        return B.CreateConstGEP2_32(ArrayType::get(I32, 64), Env, 0, 0);
    };
    Value *a = Env ? B.CreateLoad(I32, accAddr()) : acc;
    Value *idx = B.CreateURem(a, B.getInt32(3));
    FunctionType *FT = binopTy(C);
    Type *PT = PointerType::get(C, 0);
    Value *fp = B.CreateLoad(
        PT, B.CreateGEP(ArrayType::get(PT, 3), Table,
                         {B.getInt32(0), idx}));
    Value *r = B.CreateAdd(a, B.CreateCall(FT, fp, {a, i}));
    if (Env)
        B.CreateStore(r, accAddr());
    return r;
}

// One loop body. ENV==null: pure SSA (trace style). Else: commit acc to
// Env[0] after every op (TCG style). Returns final acc value.
static Value *emitBody(IRBuilder<> &B, LLVMContext &C, Value *acc, Value *i,
                       GlobalVariable *Table, GlobalVariable *Env)
{
    Type *I32 = Type::getInt32Ty(C);
    auto accAddr = [&]() {
        return B.CreateConstGEP2_32(ArrayType::get(I32, 64), Env, 0, 0);
    };
    auto LA = [&]() -> Value * {
        return Env ? B.CreateLoad(I32, accAddr()) : acc;
    };
    auto SA = [&](Value *v) {
        acc = v;
        if (Env)
            B.CreateStore(v, accAddr());
    };

    // acc += leaf_add(acc, i), i.e. acc = acc + (acc + i)
    {
        Value *a = LA();
        SA(B.CreateAdd(a, B.CreateAdd(a, i)));
    }
    // acc ^= (acc >> 3) ^ (i * C)  (leaf_xor inlined)
    {
        Value *a = LA();
        SA(B.CreateXor(a, B.CreateXor(B.CreateLShr(a, B.getInt32(3)),
                                      B.CreateMul(i, B.getInt32(2654435761u)))));
    }
    // if (acc & 0x80000000) acc = acc*3+1 else acc += C
    {
        Value *a = LA();
        Value *cond = B.CreateICmpNE(
            B.CreateAnd(a, B.getInt32(0x80000000u)), B.getInt32(0));
        Value *t = B.CreateAdd(B.CreateShl(a, B.getInt32(1)),
                               B.CreateAdd(a, B.getInt32(1)));
        Value *f = B.CreateAdd(a, B.getInt32(0x9e3779b9u));
        SA(B.CreateSelect(cond, t, f));
    }
    // if ((i&1)==0) rotl(acc,5) else rotr(acc,7) -- real diamond
    {
        Function *F = B.GetInsertBlock()->getParent();
        BasicBlock *pre = B.GetInsertBlock();
        BasicBlock *ev = BasicBlock::Create(C, "ev", F);
        BasicBlock *od = BasicBlock::Create(C, "od", F);
        BasicBlock *mg = BasicBlock::Create(C, "mg", F);
        B.CreateCondBr(B.CreateICmpEQ(B.CreateAnd(i, B.getInt32(1)),
                                      B.getInt32(0)),
                       ev, od);
        B.SetInsertPoint(ev);
        Value *ae = LA();
        Value *re = B.CreateOr(B.CreateShl(ae, B.getInt32(5)),
                               B.CreateLShr(ae, B.getInt32(27)));
        B.CreateBr(mg);
        B.SetInsertPoint(od);
        Value *ao = LA();
        Value *ro = B.CreateOr(B.CreateLShr(ao, B.getInt32(7)),
                               B.CreateShl(ao, B.getInt32(25)));
        B.CreateBr(mg);
        B.SetInsertPoint(mg);
        PHINode *p = B.CreatePHI(I32, 2);
        p->addIncoming(re, ev);
        p->addIncoming(ro, od);
        SA(p);
        (void)pre;
    }
    // indirect call through table
    SA(emitIndirect(B, C, acc, i, Table, Env));
    // if ((acc & 7) == 0) acc += acc*0x1234567+1 -- rare path
    {
        Function *F = B.GetInsertBlock()->getParent();
        BasicBlock *pre = B.GetInsertBlock();
        BasicBlock *rare = BasicBlock::Create(C, "rare", F);
        BasicBlock *cont = BasicBlock::Create(C, "cont", F);
        Value *a = LA();
        B.CreateCondBr(B.CreateICmpEQ(B.CreateAnd(a, B.getInt32(7)),
                                      B.getInt32(0)),
                       rare, cont);
        B.SetInsertPoint(rare);
        Value *ar = LA();
        Value *rr = B.CreateAdd(
            ar, B.CreateAdd(B.CreateMul(ar, B.getInt32(0x1234567u)),
                            B.getInt32(1)));
        B.CreateBr(cont);
        B.SetInsertPoint(cont);
        PHINode *p = B.CreatePHI(I32, 2);
        p->addIncoming(a, pre);
        p->addIncoming(rr, rare);
        SA(p);
    }
    return acc;
}

// Build workload(seed)->i32 with or without env commits. Table holds the
// three leaf function pointers; leaves are emitted once per module.
static Function *buildWorkload(Module *M, LLVMContext &C, bool useEnv,
                               GlobalVariable *Table, GlobalVariable *Env,
                               std::array<Function *, 3> &Leaf)
{
    Type *I32 = Type::getInt32Ty(C);
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *W = Function::Create(
        FT, Function::ExternalLinkage,
        useEnv ? "workload_env" : "workload_ssa", M);
    W->args().begin()->setName("seed");
    IRBuilder<> B(BasicBlock::Create(C, "entry", W));
    Value *acc = &*W->args().begin();
    if (useEnv)
        B.CreateStore(acc, B.CreateConstGEP2_32(ArrayType::get(I32, 64), Env,
                                                0, 0));
    BasicBlock *head = BasicBlock::Create(C, "head", W);
    BasicBlock *body = BasicBlock::Create(C, "body", W);
    BasicBlock *done = BasicBlock::Create(C, "done", W);
    B.CreateBr(head);
    B.SetInsertPoint(head);
    PHINode *pa = B.CreatePHI(I32, 2);
    PHINode *pi = B.CreatePHI(I32, 2);
    pa->addIncoming(acc, B.GetInsertBlock()->getSinglePredecessor());
    pi->addIncoming(B.getInt32(0), pa->getIncomingBlock(0));
    B.CreateCondBr(B.CreateICmpULT(pi, B.getInt32(ITERS)), body, done);
    B.SetInsertPoint(body);
    Value *na = emitBody(B, C, pa, pi, Table, Env);
    BasicBlock *tail = B.GetInsertBlock(); // diamond exit, not 'body'
    Value *ni = B.CreateAdd(pi, B.getInt32(1));
    B.CreateBr(head);
    pa->addIncoming(na, tail);
    pi->addIncoming(ni, tail);
    B.SetInsertPoint(done);
    Value *r = pa;
    if (useEnv)
        r = B.CreateLoad(I32, B.CreateConstGEP2_32(ArrayType::get(I32, 64),
                                                   Env, 0, 0));
    B.CreateRet(r);
    (void)Leaf;
    return W;
}

static std::array<Function *, 3> buildLeaves(Module *M, LLVMContext &C)
{
    Type *I32 = Type::getInt32Ty(C);
    FunctionType *FT = binopTy(C);
    std::array<Function *, 3> F = {
        Function::Create(FT, Function::ExternalLinkage, "leaf_add", M),
        Function::Create(FT, Function::ExternalLinkage, "leaf_xor", M),
        Function::Create(FT, Function::ExternalLinkage, "leaf_mul", M),
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

static uint64_t nsNow(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int main(int argc, char **argv)
{
    bool useEnv = argc > 1 && std::string(argv[1]) == "env";
    printf("tier2 demo: %s mode\n", useEnv ? "env-commit (TCG-like)" : "ssa (trace-like)");

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    auto Ctx = std::make_unique<LLVMContext>();
    LLVMContext &C = *Ctx;
    auto M = std::make_unique<Module>("tier2", C);
    Type *I32 = Type::getInt32Ty(C);

    auto Leaf = buildLeaves(M.get(), C);

    // Table of leaf pointers + env array, with initializers.
    ArrayType *PFT = ArrayType::get(PointerType::get(C, 0), 3);
    Constant *tabInit = ConstantArray::get(
        PFT, {Leaf[0], Leaf[1], Leaf[2]});
    GlobalVariable *Table = new GlobalVariable(
        *M, PFT, /*constant=*/true, GlobalValue::InternalLinkage, tabInit,
        "optable");
    ArrayType *EnvT = ArrayType::get(I32, 64);
    GlobalVariable *Env = new GlobalVariable(
        *M, EnvT, /*constant=*/false, GlobalValue::InternalLinkage,
        ConstantAggregateZero::get(EnvT), "env");

    Function *W = buildWorkload(M.get(), C, useEnv, Table, Env, Leaf);
    (void)W;

    std::string Err;
    raw_string_ostream OS(Err);
    if (verifyModule(*M, &OS)) {
        fprintf(stderr, "verify failed:\n%s\n", OS.str().c_str());
        return 1;
    }

    // Optimize with the default O2 pipeline.
    uint64_t t0 = nsNow();
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
    MPM.run(*M, MAM);
    uint64_t t1 = nsNow();

    auto JIT = cantFail(LLJITBuilder().create());
    cantFail(JIT->addIRModule(ThreadSafeModule(std::move(M), std::move(Ctx))));
    auto Sym = cantFail(JIT->lookup(useEnv ? "workload_env" : "workload_ssa"));
    auto *Fn = Sym.toPtr<uint32_t(uint32_t)>();
    uint64_t t2 = nsNow();
    uint32_t sum = Fn(0x12345u);
    uint64_t t3 = nsNow();
    printf("opt time: %llums  jit-link: %llums  exec: %llums  sum=0x%x %s\n",
           (unsigned long long)(t1 - t0) / 1000000,
           (unsigned long long)(t2 - t1) / 1000000,
           (unsigned long long)(t3 - t2) / 1000000, sum,
           sum == 0x147ce5ff ? "OK" : "MISMATCH");
    return sum == 0x147ce5ff ? 0 : 1;
}
