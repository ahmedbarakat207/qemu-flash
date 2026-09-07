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
3. DONE: live runtime sampling profiler (sampled `exec_count` on `TranslationBlock`),
   asynchronous background compiler worker thread (`tier2-compiler`), closed-loop
   trace collection (`tier2_find_loop_trace`), live LLVM ORC JIT trace compiler
   (`libqemu-tier2.dylib`), safe execution bridge via `tcg_qemu_tb_exec`, and
   live invalidation hooks (`tier2_invalidate`). Tested on `dbc-bench`: bit-for-bit
   checksum match, 3.9x speedup nochain on dispatch-heavy workload
   (0.46s vs 1.80s stock, 2.0x tier2-attributable).
4. DONE: direct SSA IR translation, multi-TB loop fusion, and SIMD lowering:
   - Post-optimization TCG op snapshot capture in `tcg_gen_code`
     (`tier2_capture_tb_ops`, stable `Tier2OpRec` ABI, byte-offset env accesses).
   - Multi-TB CFG fusion (Phase 1): stitches inner loops across basic blocks into
     unified LLVM CFG, eliminating intermediate env commits.
   - Flat RAM direct pointer lowering (Phase 3): direct host virtual RAM
     pointer access for identity-mapped memory and stack ops.
   - Vector SIMD transpilation (Phase 4): x86 SSE/AVX vector ops lowered
     directly to ARM64 NEON (`FixedVectorType`, `v0–v31`).
   - Differential verification: all 18 offline self-tests (`build/tier2-selftest`)
     passing green.
5. DONE: direct block chaining integration (`goto_tb` re-linking, Phase 5):
   - Dynamic jump slot re-linking via dedicated ARM64 native chain stubs (`tb->tier2_stub`).
   - Predecessor TBs jump directly to Tier-2 machine code without returning to
     `cpu_tb_exec()`. Safe W^X transactions and atomic jump resets via `tb_reset_jump()`.
   - Measured: chained mode accelerated to 0.369s median (1.62x over stock 0.597s).
6. DONE: persistent on-disk JIT cache (Phase 2) & async signal profiler (Phase 6):
   - CityHash64 cryptographic caching of native object files (`~/.cache/qemu/tier2/*.o`).
   - Zero-overhead POSIX `SIGPROF` timer sampling at 500 Hz (`QEMU_TIER2_PROF_HZ`).
7. DONE: High-Level Emulation (HLE) library shims for `linux-user` (Phase 7):
   - Direct host native C library dispatch for math, string, and crypto functions.

## Measured (Apple M2, Sep 2026, LLVM 22.1.6)

- `contrib/dbc-bench` live benchmark (5 runs each, checksums bit-for-bit identical):
  - **Stock chained**: 0.597s median (0.595, 0.598, 0.595, 0.597, 0.597)
  - **Patched chained (Phase 5 re-linked)**: **0.369s median** (0.437, 0.369, 0.369, 0.368, 0.367) — **1.62x faster**
  - **Stock unchained (`-d nochain`)**: 1.769s median (1.772, 1.830, 1.769, 1.761, 1.734)
  - **Patched unchained (`-d nochain`)**: **0.520s median** (0.523, 0.524, 0.520, 0.519, 0.519) — **3.40x faster**
  - **Guest compute loop cycles**: **87.5M cycles** vs 349.5M baseline (**3.99x reduction**)
- `build/tier2-bench` (8M-iter loop through real walker): ref=13ms, compile=5ms,
  exec=9ms, speedup=1.44x, checksum OK (`acc=0x608ca391f307f1`).
- `build/tier2-selftest`: All 18 tests ALL GREEN (Trace 1–16 + Trace 17 NEON SIMD + Trace 18 HLE shims).
- `contrib/llvm-tier2/tier2-demo` (model loop, 8M iters): ssa ~80-83ms,
  env-commit ~78-79ms exec, checksum `0x147ce5ff` OK both modes (~6x over baseline).
- Offline `op-run` vs `interp.py`: 30/30 fresh randomized ALU runs agree.

## Runtime Architecture

- `tier2.h`: Fast sampling profiler and signal sampler interface on vCPU thread.
  Eliminates redundant Darwin APRR W^X memory permission transitions.
- `tier2.c`: Asynchronous background compiler thread (`tier2-compiler`).
  Multi-TB loop extraction, persistent cache integration, FIFO compilation queue,
  dynamic loading of `libqemu-tier2.dylib`, and Phase 5 chain stub management.
- `tier2-jit.cpp`: LLVM ORC JIT trace compilation engine. Compiles fused multi-TB
  traces into native ARM64 machine code with `-O2`, Flat RAM direct pointer lowering,
  and x86 SSE/AVX $\rightarrow$ NEON vector transpilation.
- `tier2-prof.c`: Async POSIX `SIGPROF` signal-based profiler sampling at 500 Hz.
- `linux-user/hle-thunks.c`: High-Level Emulation shims forwarding guest library
  calls directly to host native libc and libm functions.
- `cpu-exec.c`: Direct dispatch via `fn(cpu_env(cpu))` when `itb->tier2_code` is
  present, with predecessor jump slot patching (`goto_tb` re-linking) and full
  Darwin W^X safe execution.
