# LLVM tier-2 prototype (modern ORC, LLVM >= 16 APIs)

Answers one question: how much headroom does an LLVM tier-2 backend have
over TCG on our actual hot loop? (HQEMU's idea, re-implemented against
current LLVM instead of porting its decade-old TCG/LLVM glue.)

## Result (Apple M2, 8M loop iterations, re-measured Sep 2026, LLVM 22.1.6)

| implementation | exec time | vs TCG |
|---|---|---|
| TCG (this repo, patched; prior measurement, same machine class) | ~510ms | 1x |
| tier2-demo ssa (trace-style) | ~80-83ms | ~6x |
| tier2-demo env (TCG-style commits) | ~78-79ms | ~6.5x |
| native clang -O2 (ceiling) | 77-96ms | ~6x |

Checksum `0x147ce5ff` matches the TCG run in both modes or the model is
wrong. Tier-2 reaches the native ceiling here.

Compile tax (same machine): cold first process ~36ms opt + ~47ms JIT
link; warm steady state ~1-2ms opt + ~4ms link. It amortizes only
for hot code -- cold code like OS boot must never enter tier-2.
Hot detection gating is mandatory, as in HQEMU.

## Why this is representative, and where it is optimistic

- The loop keeps real control flow (alternating diamond, rare path) and a
  true indirect call per iteration -- the optimizer cannot cheat it away.
- The `env` variant commits guest state per op exactly like TCG; LLVM's
  mem2reg/GVN/LICM delete all of it (both variants compile to the same
  code). That erasure is precisely the dividend TCG's RTL design leaves
  on the table.
- Optimistic parts (upper bound, not a promise): no exception-state or
  TB prologue costs, no softmmu/TLB traffic in this loop, and the
  indirect call is direct (a real tier-2 needs guards or a lookup for
  polymorphic sites). Realistic tier-2 on this loop: 3-5x, not 6.5x.
- Compile tax is real (see numbers above). Hot detection gating is
  mandatory, as in HQEMU.

## op-walker bench (in-tree `tier2-jit.cpp` walker, Sep 2026)

`build/tier2-bench` (from `tcg/llvm/Makefile`) compiles an 8M-iteration
loop expressed as captured-style op records (`LD/ST env`, `ADD/SUB`,
`SHR`, `MUL`, `XOR`, `BRCOND`) through the real in-tree walker + O2
(vectorization off) + ORC, and times it against the identical plain C++
scalar loop:

| implementation | exec time | vs clang scalar |
|---|---|---|
| clang -O2 scalar reference | ~9-10ms | 1x |
| op-walker compiled trace | ~10ms (best of 3) | ~0.9-1.0x |
| walker compile tax | ~4ms warm / ~71ms cold first process | -- |

Checksum matches the C++ reference or the bench fails. ~1.0x is the
honest result here: the loop carries a true data dependency (`acc`),
so neither clang nor the walker can vectorize it -- both run at the
native scalar ceiling. The tier-2 dividend on real guest code comes
from deleting TCG dispatch and env-commit traffic around such loops,
not from beating clang at straight-line scalar code.

## Why not port HQEMU's backend

`llvm-opc.cpp` there iterates `tcg_ctx.gen_op_buf[]`, an op array that no
longer exists in QEMU 11 (intrusive `QTAILQ` TCGOps now), defines its own
parallel op table/extensions, and targets the legacy JIT/MCJIT APIs that
current LLVM deleted (ORC only). Porting = rewriting ~8-10k lines across
two API breaks. This prototype validates the *idea* on current APIs
instead; a production tier-2 would map the full TCG op set (incl.
softmmu helpers, exits, atomics), add hot-region detection, ORC code
replacement with invalidation, and MTTCG safety -- weeks to months.

## op-run: real TCG ops in, verified execution out (v1: pure-ALU traces)

`opparse.py` converts QEMU `-d op` text for one TB into op records;
`op-run` compiles them with LLVM O2 + ORC and executes once. Pure-ALU
traces only (calls, exits, `goto_*`, guest-memory ops are trace
boundaries and stop the parse with rc=2). Temps are modeled as untyped
64-bit cells with width-governed ops -- matching backend behavior
(backends zero-extend i32 defs; comparisons truncate; shifts mask
counts; stores are read-modify-write).

Verification: `interp.py` is an independent interpreter of the same
record format. 12/12 parseable hot TBs agree, plus 144 randomized
differential runs with edge values (0/1/2^31-1/2^31/2^32-1/2^63-1/2^63/
2^64-1), all green (harness re-verified Sep 2026: 30/30 fresh
randomized ALU runs agree after the LLVM 22 rebuild).

    ./build/qemu-system-x86_64 ... -d op -D ops.log -dfilter <range>
    python3 opparse.py ops.log <tb-index> > tb.rec
    ./op-run tb.rec 0=5 4=100 ...

## Build/run

    make          # needs llvm-config (Homebrew: /opt/homebrew/opt/llvm)
    make run      # runs both modes, verifies checksum

In-tree walker checks (from repo root, needs the same llvm-config):

    make -C tcg/llvm selftest   # build/tier2-selftest: counting loop +
                                # op coverage vs C++ references, plus
                                # bail-out negative cases; ALL GREEN
    make -C tcg/llvm bench      # build/tier2-bench: 8M-iteration loop
                                # through the walker vs scalar C++ (see above)
    ./build/tier2-selftest && ./build/tier2-bench
