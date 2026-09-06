# LLVM tier-2 backend home (in-tree interface + roadmap)

Long-term goal: an LLVM tier-2 translator for hot traces, in the spirit
of HQEMU but against current TCG (linked-list TCGOps) and current LLVM
(ORC v2), living here instead of as an unportable fork.

## Why not port HQEMU's backend

Its op mapper iterates `tcg_ctx.gen_op_buf[]`, an op array removed in
modern TCG (intrusive `QTAILQ` now), ships its own parallel op table and
trace ops, and targets the legacy JIT/MCJIT APIs that current LLVM
deleted. Porting means rewriting ~8-10k lines across two API breaks.
Instead this directory grows a native implementation, validated
off-line first (see `contrib/llvm-tier2/`).

## Roadmap (each step shippable and measured)

1. DONE (`contrib/llvm-tier2/tier2-demo`): hot loop as LLVM IR, SSA vs
   env-commit styles, checksum-verified, 6.5x over TCG on the model.
2. DONE (`contrib/llvm-tier2/opparse.py` + `op-run` + `interp.py`):
   real `-d op` text for a hot TB -> op records -> LLVM IR -> ORC
   execute, differentially tested (12/12 hot TBs + 144 randomized runs
   with edge values, all green). Pure-ALU traces only so far; calls,
   exits, `goto_*` and guest-memory ops are trace boundaries.
3. NEXT: feed live TB op streams (structured op dump in `tcg_dump_ops`),
   hot-region detection (execution counters), ORC code replacement with
   invalidation hooks, MTTCG safety. `tier2.h` below sketches the
   runtime interface; it is header-only and unused until then.

## Runtime interface sketch (`tier2.h`)

- `tier2_init()` / `tier2_shutdown()`: ORC context lifetime.
- Hot detection and code replacement plug into `tb_gen_code` /
  `tb_phys_invalidate` later; until wired, everything here is inert.
