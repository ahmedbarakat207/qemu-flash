// op-run: execute `-d op` records (see opparse.py) as LLVM IR.
//
// Usage: op-run <trace.rec> [slot=value ...]
//   Initializes all slots (default 0), runs the trace once through ORC,
//   prints final `slot=value` lines (decimal, unsigned 64-bit).
//   Branch to an undefined label = trace side exit (early return).
//   All state lives in the state[] array (TCG-style commits); LLVM's
//   optimizer removes the traffic, so no PHI bookkeeping is needed.

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>

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

namespace {

struct Arg {
    enum { REG, SLOT, IMM, LABEL, COND } kind;
    int slot;          // REG/SLOT: state[] index
    unsigned bits;     // REG: value width
    int64_t imm;       // IMM (low-bits-correct for its use width)
    std::string text;  // LABEL id or COND name
};

struct Op {
    std::string name;
    unsigned bits;
    std::vector<Arg> args;
};

Type *T64(LLVMContext &C) { return Type::getInt64Ty(C); }

Value *maskVal(IRBuilder<> &B, Value *v, unsigned bits)
{
    if (bits >= 64)
        return v;
    return B.CreateAnd(v, B.getInt64((1ULL << bits) - 1));
}

uint64_t maskOf(unsigned bits)
{
    return bits >= 64 ? ~0ULL : ((1ULL << bits) - 1);
}

Value *evalCond(IRBuilder<> &B, LLVMContext &C, const std::string &c,
                 unsigned bits, Value *a, Value *b)
{
    // Comparisons observe the operand width (w-form backend semantics):
    // truncate i32 operands before comparing.
    if (bits < 64) {
        Type *T = IntegerType::get(C, bits);
        a = B.CreateTrunc(a, T);
        b = B.CreateTrunc(b, T);
    }
    if (c == "eq")
        return B.CreateICmpEQ(a, b);
    if (c == "ne")
        return B.CreateICmpNE(a, b);
    if (c == "lt")
        return B.CreateICmpSLT(a, b);
    if (c == "le")
        return B.CreateICmpSLE(a, b);
    if (c == "gt")
        return B.CreateICmpSGT(a, b);
    if (c == "ge")
        return B.CreateICmpSGE(a, b);
    if (c == "ult")
        return B.CreateICmpULT(a, b);
    if (c == "ule")
        return B.CreateICmpULE(a, b);
    if (c == "ugt")
        return B.CreateICmpUGT(a, b);
    if (c == "uge")
        return B.CreateICmpUGE(a, b);
    if (c == "tsteq")
        return B.CreateICmpEQ(B.CreateAnd(a, b), B.getInt64(0));
    if (c == "tstne")
        return B.CreateICmpNE(B.CreateAnd(a, b), B.getInt64(0));
    fprintf(stderr, "bad cond %s\n", c.c_str());
    exit(2);
    return nullptr;
}

} // namespace

static Arg parseArg(const std::string &t)
{
    Arg a{};
    if (t[0] == 'r') {
        a.kind = Arg::REG;
        auto c = t.find(':');
        a.slot = std::stoi(t.substr(1, c - 1));
        a.bits = std::stoi(t.substr(c + 1));
    } else if (t[0] == 's') {
        a.kind = Arg::SLOT;
        a.slot = std::stoi(t.substr(1));
    } else if (t[0] == 'i') {
        a.kind = Arg::IMM;
        a.imm = std::stoll(t.substr(1));
    } else if (t[0] == 'L') {
        a.kind = Arg::LABEL;
        a.text = t;
    } else {
        a.kind = Arg::COND;
        a.text = t;
    }
    return a;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: op-run <trace.rec> [slot=value ...]\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("open");
        return 2;
    }
    int nslots = 0;
    std::vector<Op> ops;
    std::map<int, uint64_t> init;
    for (int i = 2; i < argc; i++) {
        int s;
        unsigned long long v;
        if (sscanf(argv[i], "%d=%llu", &s, &v) == 2)
            init[s] = v;
    }
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        std::vector<std::string> toks;
        std::string l = line;
        size_t p = 0;
        while (p < l.size()) {
            while (p < l.size() && isspace((unsigned char)l[p]))
                p++;
            size_t q = p;
            while (q < l.size() && !isspace((unsigned char)l[q]))
                q++;
            if (q > p)
                toks.push_back(l.substr(p, q - p));
            p = q;
        }
        if (toks.empty())
            continue;
        if (toks[0] == "slots") {
            nslots = std::stoi(toks[1]);
            continue;
        }
        Op op;
        op.name = toks[0];
        if (op.name == "label" || op.name == "br") {
            op.args.push_back(parseArg("L" + toks[1]));
        } else if (op.name == "brcond" || op.name == "movcond") {
            // brcond bits cond a1 a2 label | movcond bits cond c1 c2 v1 v2
            op.bits = std::stoi(toks[1]);
            Arg c;
            c.kind = Arg::COND;
            c.text = toks[2];
            op.args.push_back(c);
            for (size_t i = 3; i < toks.size(); i++)
                op.args.push_back(parseArg(toks[i]));
        } else {
            op.bits = std::stoi(toks[1]);
            for (size_t i = 2; i < toks.size(); i++)
                op.args.push_back(parseArg(toks[i]));
        }
        ops.push_back(op);
    }
    fclose(f);
    if (nslots <= 0 || nslots > 4096) {
        fprintf(stderr, "bad slots %d\n", nslots);
        return 2;
    }

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    auto Ctx = std::make_unique<LLVMContext>();
    LLVMContext &C = *Ctx;
    auto M = std::make_unique<Module>("oprun", C);
    FunctionType *FT =
        FunctionType::get(Type::getVoidTy(C), {PointerType::get(C, 0)}, false);
    Function *Run =
        Function::Create(FT, Function::ExternalLinkage, "run", M.get());
    Value *State = &*Run->args().begin();
    State->setName("state");

    // Split into blocks: a block ends at br/brcond (inclusive);
    // a label starts a new block. Fallthrough of a terminator is the
    // next block; a trace ending mid-air returns void.
    struct Block {
        std::string label; // "" for anonymous
        std::vector<Op> ops;
    };
    std::vector<Block> blocks(1);
    std::map<std::string, size_t> labelDef;
    for (auto &op : ops) {
        if (op.name == "label") {
            if (!blocks.back().ops.empty() || !blocks.back().label.empty()) {
                blocks.push_back(Block());
            }
            blocks.back().label = op.args[0].text;
            if (labelDef.count(op.args[0].text)) {
                fprintf(stderr, "duplicate label\n");
                return 2;
            }
            labelDef[op.args[0].text] = blocks.size() - 1;
        } else {
            blocks.back().ops.push_back(op);
            if (op.name == "br" || op.name == "brcond")
                blocks.push_back(Block());
        }
    }
    while (!blocks.empty() && blocks.back().ops.empty() &&
           blocks.back().label.empty())
        blocks.pop_back();
    std::map<size_t, BasicBlock *> bb;
    for (size_t i = 0; i < blocks.size(); i++)
        bb[i] = BasicBlock::Create(
            C, blocks[i].label.empty() ? "entry" : "L" + blocks[i].label, Run);

    auto statePtr = [&](IRBuilder<> &B, int s) {
        return B.CreateGEP(T64(C), State, B.getInt64(s));
    };
    auto getR = [&](IRBuilder<> &B, const Arg &a) -> Value * {
        if (a.kind == Arg::IMM)
            return B.getInt64((uint64_t)a.imm);
        return B.CreateLoad(T64(C), statePtr(B, a.slot));
    };
    auto setR = [&](IRBuilder<> &B, const Arg &a, Value *v) {
        B.CreateStore(maskVal(B, v, a.bits), statePtr(B, a.slot));
    };
    auto getImm = [&](const Arg &a) -> uint64_t {
        return (uint64_t)a.imm;
    };

    for (size_t bi = 0; bi < blocks.size(); bi++) {
        IRBuilder<> B(bb[bi]);
        bool term = false;
        for (auto &op : blocks[bi].ops) {
            const std::string &n = op.name;
            unsigned b = op.bits ? op.bits : 64;
            auto A = [&](size_t i) -> Value * { return getR(B, op.args[i]); };
            if (n == "mov") {
                setR(B, op.args[0], A(1));
            } else if (n == "add") {
                setR(B, op.args[0], B.CreateAdd(A(1), A(2)));
            } else if (n == "sub") {
                setR(B, op.args[0], B.CreateSub(A(1), A(2)));
            } else if (n == "mul") {
                setR(B, op.args[0], B.CreateMul(A(1), A(2)));
            } else if (n == "and") {
                setR(B, op.args[0], B.CreateAnd(A(1), A(2)));
            } else if (n == "or") {
                setR(B, op.args[0], B.CreateOr(A(1), A(2)));
            } else if (n == "xor") {
                setR(B, op.args[0], B.CreateXor(A(1), A(2)));
            } else if (n == "neg") {
                setR(B, op.args[0], B.CreateSub(B.getInt64(0), A(1)));
            } else if (n == "not") {
                setR(B, op.args[0], B.CreateXor(A(1), B.getInt64(~0ULL)));
            } else if (n == "shl" || n == "shr" || n == "sar" ||
                       n == "rotl" || n == "rotr") {
                Value *cnt = B.CreateAnd(A(2), B.getInt64(b - 1));
                Value *r = nullptr;
                if (n == "shl")
                    r = B.CreateShl(A(1), cnt);
                else if (n == "shr")
                    // Logical shift within width b: mask input first so
                    // high bits cannot shift down into the field.
                    r = B.CreateLShr(maskVal(B, A(1), b), cnt);
                else if (n == "sar") {
                    // Arithmetic shift within width b: sign-extend the
                    // low b bits first (matches w-form backend behavior).
                    Value *se = A(1);
                    if (b < 64)
                        se = B.CreateSExt(
                            B.CreateTrunc(A(1), IntegerType::get(C, b)),
                            T64(C));
                    r = B.CreateAShr(se, cnt);
                } else {
                    // Rotate the low-b field; count 0 is identity (a
                    // full-width shift by b would be poison).
                    Value *lo = maskVal(B, A(1), b);
                    Value *rc = B.CreateSub(B.getInt64(b), cnt);
                    Value *fwd = (n == "rotl")
                                     ? B.CreateShl(lo, cnt)
                                     : B.CreateLShr(lo, cnt);
                    Value *bwd = (n == "rotl")
                                     ? B.CreateLShr(lo, rc)
                                     : B.CreateShl(lo, rc);
                    r = B.CreateSelect(
                        B.CreateICmpEQ(cnt, B.getInt64(0)), lo,
                        maskVal(B, B.CreateOr(fwd, bwd), b));
                }
                setR(B, op.args[0], r);
            } else if (n == "extract") {
                // extract bits dst src off len (off/len are immediates)
                uint64_t off = getImm(op.args[2]) & 63;
                uint64_t len = getImm(op.args[3]) & 127;
                Value *v = B.CreateLShr(A(1), B.getInt64(off));
                if (len < 64)
                    v = B.CreateAnd(v, B.getInt64((1ULL << len) - 1));
                setR(B, op.args[0], v);
            } else if (n == "sextract") {
                // sextract bits dst src off len: sign-extend field
                uint64_t off = getImm(op.args[2]) & 63;
                uint64_t len = getImm(op.args[3]) & 127;
                Value *v = B.CreateLShr(A(1), B.getInt64(off));
                Value *r;
                if (len >= 64) {
                    r = v;
                } else {
                    Value *t = B.CreateTrunc(v, IntegerType::get(C, (unsigned)len));
                    r = B.CreateSExt(t, T64(C));
                }
                setR(B, op.args[0], r);
            } else if (n == "deposit") {
                uint64_t off = getImm(op.args[3]) & 63;
                uint64_t len = getImm(op.args[4]) & 127;
                uint64_t m = (len >= 64) ? ~0ULL : (((1ULL << len) - 1) << off);
                Value *v = B.CreateOr(
                    B.CreateAnd(A(1), B.getInt64(~m)),
                    B.CreateAnd(B.CreateShl(A(2), B.getInt64(off)),
                                B.getInt64(m)));
                setR(B, op.args[0], v);
            } else if (n == "zext") {
                // dst64 = zero-extend(low32(src64))
                setR(B, op.args[0],
                     B.CreateAnd(A(1), B.getInt64(0xffffffffULL)));
            } else if (n == "ld") {
                // ld width dst envslot
                unsigned w = op.bits;
                Value *v = B.CreateLoad(
                    T64(C), statePtr(B, op.args[1].slot));
                setR(B, op.args[0], maskVal(B, v, w));
            } else if (n == "st") {
                // st width envslot src (read-modify-write for sub-width)
                unsigned w = op.bits;
                Value *old = B.CreateLoad(T64(C),
                                          statePtr(B, op.args[0].slot));
                Value *nv = B.CreateOr(
                    B.CreateAnd(old, B.getInt64(~maskOf(w))),
                    B.CreateAnd(A(1), B.getInt64(maskOf(w))));
                B.CreateStore(nv, statePtr(B, op.args[0].slot));
            } else if (n == "movcond") {
                // movcond bits | cond dst c1 c2 v1 v2
                Value *c = evalCond(B, C, op.args[0].text, op.bits, A(2), A(3));
                setR(B, op.args[1], B.CreateSelect(c, A(4), A(5)));
            } else if (n == "br") {
                auto it = labelDef.find(op.args[0].text);
                if (it == labelDef.end())
                    B.CreateRetVoid();
                else
                    B.CreateBr(bb[it->second]);
                term = true;
                break;
            } else if (n == "brcond") {
                // brcond bits cond a1 a2 label
                Value *c = evalCond(B, C, op.args[0].text, op.bits, A(1), A(2));
                auto it = labelDef.find(op.args[3].text);
                BasicBlock *fall =
                    (bi + 1 < blocks.size()) ? bb[bi + 1] : nullptr;
                if (it == labelDef.end()) {
                    if (!fall) {
                        B.CreateRetVoid(); // taken would exit; no code after
                    } else {
                        BasicBlock *retB =
                            BasicBlock::Create(C, "sideexit", Run);
                        IRBuilder<> RB(retB);
                        RB.CreateRetVoid();
                        B.CreateCondBr(c, retB, fall);
                    }
                } else if (fall) {
                    B.CreateCondBr(c, bb[it->second], fall);
                } else {
                    BasicBlock *retB =
                        BasicBlock::Create(C, "sideexit", Run);
                    IRBuilder<> RB(retB);
                    RB.CreateRetVoid();
                    B.CreateCondBr(c, bb[it->second], retB);
                }
                term = true;
                break;
            } else {
                fprintf(stderr, "unhandled op %s\n", n.c_str());
                return 2;
            }
        }
        if (!term) {
            if (bi + 1 < blocks.size())
                IRBuilder<>(bb[bi]).CreateBr(bb[bi + 1]);
            else
                IRBuilder<>(bb[bi]).CreateRetVoid();
        }
    }

    std::string Err;
    raw_string_ostream OS(Err);
    if (verifyModule(*M, &OS)) {
        fprintf(stderr, "verify failed:\n%s\n", OS.str().c_str());
        return 1;
    }
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
    PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2).run(*M, MAM);

    auto JIT = cantFail(LLJITBuilder().create());
    cantFail(JIT->addIRModule(ThreadSafeModule(std::move(M), std::move(Ctx))));
    auto Sym = cantFail(JIT->lookup("run"));
    auto *Fn = Sym.toPtr<void(uint64_t *)>();

    static uint64_t state[4096];
    for (auto &kv : init)
        if (kv.first >= 0 && kv.first < (int)(sizeof(state) / sizeof(state[0])))
            state[kv.first] = kv.second;
    Fn(state);
    for (int s = 0; s < nslots; s++)
        printf("%d=%llu\n", s, (unsigned long long)state[s]);
    return 0;
}
