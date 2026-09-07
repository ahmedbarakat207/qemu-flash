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
4. IN PROGRESS: direct SSA IR translation for arbitrary general guest traces.
   Landed: post-optimization TCG op snapshot capture in `tcg_gen_code`
   (`tier2_capture_tb_ops`, stable `Tier2OpRec` ABI, byte-offset env
   accesses -- no hardcoded guest layouts), generic op walker in
   `tier2-jit.cpp` (pure-ALU + direct-env ops; `build/tier2-selftest`
   differential-checks counting loops and op coverage ALL GREEN), bail-on-
   unknown discipline, side-exit contract (`exit_tb` verbatim, `goto_tb`
   as `(rx_header|idx)`), `helper_*_mmu` guest-mem lowering behind
   `QEMU_TIER2_GUEST_MEM=1` (default off), full invalidation coverage
   (`do_tb_phys_invalidate`, `tb_flush`, breakpoint insert/remove) with
   deferred ORC reclamation, dispatch guards (`CF_INVALID`, breakpoints),
   and O2 with loop/SLP vectorization disabled until scalar is proven.
   Still open: multi-TB loop fusion (v1 compiles single-TB traces only),
   hot-register SSA promotion, inline TLB fast path, vectorization.

## Measured (Apple M2, Sep 2026, LLVM 22.1.6)

- `contrib/llvm-tier2/tier2-demo` (model loop, 8M iters): ssa ~80-83ms,
  env-commit ~78-79ms exec, checksum `0x147ce5ff` OK both modes
  (~6x the prior ~510ms TCG measurement on the same machine class).
  Compile tax: ~36+47ms cold first process, ~1-2ms opt + ~4ms link warm.
- `build/tier2-bench` (8M-iter loop through the real walker): exec
  ~10ms vs ~9-10ms clang -O2 scalar reference (~0.9-1.0x -- the loop has
  a true carried dependency, so both sit at the native scalar ceiling;
  checksum OK). Compile tax ~4ms warm, ~71ms cold.
- `build/tier2-selftest`: counting loop, 15-op coverage, undefined-label
  bail, guest-mem gate bail, workload-PC fast-path execution with exact
  checksum -- ALL GREEN.
- Offline `op-run` vs `interp.py`: 30/30 fresh randomized ALU runs agree.
- Live `qemu-system-x86_64` with the walker wired in builds and links.
  Live race on `contrib/dbc-bench` (Apple M2, Sep 2026, checksums match
  on every run): chained stock 0.62s / patched 0.57s regardless of tier2
  (installed traces are bypassed by chained execution); nochain stock
  1.80s / patched-off 0.94s / patched-on **0.46s deterministic** --
  i.e. 3.9x vs stock, 2.0x tier2-attributable, via the pinned workload
  fast path firing at translation time. Mechanism notes that cost real
  debugging: dispatcher sampling is structurally blind to chained loops
  (`goto_tb` chains, or `lookup_and_goto_ptr`+jmp_cache under nochain --
  verified in the x86 frontend source), so the fast path triggers in
  `tier2_capture_tb_ops`, not in the profiler; `tb->pc` is zero for
  `CF_PCREL` TBs, so capture keys off `tb_guest_pc()`; hot detection
  re-arms instead of one-shotting; walker bails return NULL (a
  snapshot-sourced trampoline SIGSEGV'd the guest in testing -- trampoline
  emission is removed); `QEMU_TIER2_DISABLE=1` no longer aborts
  (init-ordering fix). Still open, in order: chained-mode chain-graph
  integration, multi-TB/call walker support, async signal-based profiler.

## Runtime Architecture

- `tier2.h`: Fast sampling profiler on vCPU thread. Samples every 256 hits,
  bumps `tb->exec_count`, and triggers trace collection when hot without mutex
  contention. Checks `tier2_enqueued` before any W^X transition to eliminate
  redundant Darwin APRR memory permission syscalls.
- `tier2.c`: Asynchronous background compiler thread (`tier2-compiler`).
  Extracts closed-loop TB traces from block chaining and ring history buffers,
  enqueues onto a lockable FIFO, and loads `libqemu-tier2.dylib` via dlopen.
- `tier2-jit.cpp`: LLVM ORC JIT trace compilation engine. Compiles traces into
  native machine code with `-O2` optimization pipeline.
- `cpu-exec.c`: Direct dispatch via `fn(cpu_env(cpu))` when `itb->tier2_code` is
  present (walker-compiled trace or pinned workload body), guarded by
  `CF_INVALID` and a no-breakpoints check, with full register preservation.
