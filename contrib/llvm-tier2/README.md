# LLVM tier-2 prototype (modern ORC, LLVM >= 16 APIs)

Answers one question: how much headroom does an LLVM tier-2 backend have
over TCG on our actual hot loop? (HQEMU's idea, re-implemented against
current LLVM instead of porting its decade-old TCG/LLVM glue.)

## Result (Apple M2, 8M loop iterations)

| implementation | exec time | vs TCG |
|---|---|---|
| TCG (this repo, patched) | ~510ms | 1x |
| tier2-demo ssa (trace-style) | 78ms | 6.5x |
| tier2-demo env (TCG-style commits) | 78ms | 6.5x |
| native clang -O2 (ceiling) | 77-96ms | ~6x |

Checksum `0x147ce5ff` matches the TCG run in both modes or the model is
wrong. Tier-2 reaches the native ceiling here.

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
- Compile tax is real: ~28ms opt + ~43ms JIT link here. It amortizes only
  for hot code (this run saves ~430ms) -- cold code like OS boot must
  never enter tier-2. Hot detection gating is mandatory, as in HQEMU.

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
2^64-1), all green.

    ./build/qemu-system-x86_64 ... -d op -D ops.log -dfilter <range>
    python3 opparse.py ops.log <tb-index> > tb.rec
    ./op-run tb.rec 0=5 4=100 ...

## Build/run

    make          # needs llvm-config (Homebrew: /opt/homebrew/opt/llvm)
    make run      # runs both modes, verifies checksum
