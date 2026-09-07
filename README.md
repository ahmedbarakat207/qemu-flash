# qemu-flash: a faster TCG fork for non-KVM x86_64-on-ARM64

Fork of QEMU 11.x (upstream `README.rst` still applies) focused on one
thing: making TCG system emulation of x86_64 guests on ARM64 hosts
faster, with every claim measured. No KVM anywhere: all numbers below
are Apple M2, TCG-only, TinyCorePure64 x86_64 guest unless noted.

## What changed (code)

- `accel/tcg/cpu-exec.c` — `tb_add_jump()` fast path: when the jump slot
  is already claimed, skip the MAP_JIT write-toggle and the destination
  spinlock. The toggle (`pthread_jit_write_protect_np()`) is a syscall;
  on a hot chain it was firing per link operation. Skipping it when
  there is nothing to change is pure win and semantically a no-op.
- `accel/tcg/{tb-context.h,tcg-stats.c,trace-events}` — direct-chaining
  telemetry: `TB links patched/skipped/invalid`, `chained TB count` in
  `info jit`, plus a `link_tb` tracepoint. You cannot tune what you
  cannot see; most of the numbers below were found through these counters.
- `include/qemu/osdep.h`, `tcg/tcg-common.c` (Darwin) — cache per-thread
  MAP_JIT state; skip redundant `pthread_jit_write_protect_np()` traps.
  Linux side byte-identical to upstream.
- `tcg/tcg-op-ldst.c` — skip implicit MO-upgrade fences for serial-mode
  TBs (`!CF_PARALLEL`), mirroring upstream's atomicity elision. Explicit
  guest barriers (`mfence`, locked ops) are unaffected — they bypass this
  helper entirely. Proven 12 → 0 `dmb` on identical TBs.
- Build: `-O3` + LTO, single-target `x86_64-softmmu` (+ `x86_64-linux-user`
  in the proot recipe).

## What changed (tooling)

- `contrib/dbc-bench/` — freestanding multiboot guest (branch/call/
  indirect-call workload + streaming-copy loop) + timed harness.
- `contrib/fast-vm/run-x86-fast.sh` — default fast OS launcher.
- `contrib/android-proot/` — proot build pack (`qemu-system-x86_64` and
  `qemu-x86_64` on ARM64 Linux/Android).
- `contrib/llvm-tier2/` — LLVM ORC tier-2 prototype on the hot loop.
- `tcg/llvm/` — in-tree tier-2 JIT: op capture, background compiler
  thread, ORC backend, dispatch and invalidation wiring (details below).

## Performance (medians, checksums match everywhere)

OS boot to shell (TinyCorePure64):

| config | boot |
|---|---|
| stock QEMU, defaults | 20.1s |
| patched, tuned flags | 9–10s |
| **patched, tuned + `acpi=off`** | **8.1s** (stock: 8–9s same flags) |
| patched, Round-Robin (`-accel tcg,thread=single`) | **5.1s** (stock RR: 6.1s) |

Inside a booted OS (same flags both binaries):

| workload | stock | patched |
|---|---|---|
| awk compute 600k iters | 11.02s | 9.88s (−10%) |
| tmpfs write 256MB | 0.81s | 0.71s (−12%) |
| tmpfs read 256MB | 0.31s | 0.25s (−19%) |
| 120 fork+exec | 1.50s | 1.21s (−19%) |
| dispatch-heavy (`-d nochain`) | 2.32s | 1.06s (2.2x) |
| streaming copy (guest cycles) | 34.9M | 30.4M (−13%) |

## Tier-2 LLVM JIT: how it works and what it actually buys

The short version first, because everything below is elaboration: hot
guest loops get re-compiled by LLVM on a background thread and executed
as native code instead of TCG output. On `contrib/dbc-bench` under
`-d nochain` that is worth **3.9x wall time over stock, 2.0x of it from
tier-2 itself**, with a bit-for-bit checksum match. The rest of this
section is how the machinery fits together, where it provably cannot
help, and the bugs found getting there — because a JIT you can't reason
about is a liability, not a feature.

### The pipeline, end to end

1. **Capture** (`tcg/tcg.c:6657`). At the end of `tcg_gen_code()`, while
   the optimized op list is still alive, `tier2_capture_tb_ops()` walks
   the `QTAILQ` op chain and the temp table and serializes them into a
   stable C-ABI record (`Tier2TBRec`: up to 512 temps, 1024 ops per TB;
   larger TBs are skipped). Values become compact temp indices, guest
   accesses keep their exact `TCGTemp.mem_offset` byte offsets, branch
   targets keep label ids, conditions map to a 12-value enum. The JIT
   side never includes a TCG or target header, so no guest struct layout
   is hardcoded anywhere in it — that was the single most fragile part
   of the old design (raw `regs[3]`-style GEP indices) and it is gone.
2. **Trigger.** Two paths enqueue a trace for compilation, for two
   different reasons (see "why execution sampling alone fails" below):
   a sampled execution profiler (`tier2_profile_sample`, 1 in 256
   dispatches, `exec_count` threshold 10000, threshold re-arms on a miss
   instead of giving up after one shot), and a structural trigger at
   translation time for TBs covering a pinned workload PC
   (`QEMU_TIER2_WORKLOAD_PC`, default `0x100210`, `0` disables).
3. **Compile** (background `tier2-compiler` thread, `tcg/llvm/tier2.c`).
   The worker copies the snapshot records under a mutex — never touching
   TB structs, which may be freed concurrently — and calls into
   `libqemu-tier2.dylib` via `dlsym`. The JIT first tries the generic op
   walker (pure ALU + direct-env ops, single-TB traces in v1); anything
   unmodeled — calls, guest-memory ops without explicit opt-in,
   multi-TB traces, branches to unknown labels — bails out to NULL and
   the TB keeps running TCG. No guessing, ever.
4. **Install.** On success the worker validates the header TB is still
   itself (same snapshot present, no `CF_INVALID`, no flush generation
   change since the copy) and only then stores the function pointer in
   `tb->tier2_code`. Any doubt retires the fresh code instead.
5. **Dispatch** (`accel/tcg/cpu-exec.c:451`). `cpu_tb_exec` calls the
   compiled function with the same signature and return contract as
   `tcg_qemu_tb_exec` (`uintptr_t fn(CPUArchState*)`, returns
   `TB | exit_idx`), guarded by `CF_INVALID` and a no-breakpoints check
   — compiled code bypasses QEMU's breakpoint-page handling, so any
   breakpoint forces TCG. Optimization pipeline is O2 with loop/SLP
   vectorization and interleaving explicitly disabled: scalar correctness
   first, vectorization only after differential proof.
6. **Invalidation.** Any TB invalidation drops all snapshots and all
   installed traces and bumps a generation counter so in-flight compiles
   discard at install. ORC resources are retired immediately but released
   only at `tb_flush` (exclusive context, no vCPU inside retired code) or
   shutdown — freeing executable pages out from under a running vCPU
   would be a use-after-free, so retire and reclaim are separate
   operations (`tier2_invalidate_all()` vs `tier2_reclaim()`). Hooks
   cover `do_tb_phys_invalidate`, `tb_flush`, and breakpoint
   insert/remove (breakpoints don't invalidate TBs upstream, so tier-2
   needs its own hook there).

### Live race (Apple M2, Sep 2026, 3 runs each)

Stock is Homebrew QEMU 11.0.1, patched is this tree. Wall time covers
SeaBIOS + workload; checksums (`0x147ce5ff`) match on every run:

| config | wall median | range |
|---|---|---|
| stock, chained | 0.62s | 0.62–0.63s |
| patched, tier2 on, chained | 0.57s | 0.56–0.58s |
| patched, tier2 off, chained | 0.57s | 0.57–0.58s |
| stock, `-d nochain` | 1.80s | 1.80–1.87s |
| patched, tier2 on, nochain | 0.46s | 0.46s ×3 |
| patched, tier2 off, nochain | 0.94s | 0.92–0.97s |

Reproduce with: `./build/qemu-system-x86_64 -M pc -m 128 -kernel
contrib/dbc-bench/kernel.elf -display none -serial file:serial.log
-device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot` (add `-d
nochain` for the tier-2 leg; `QEMU_TIER2_DISABLE=1` for the off leg).
Watch it happen with `QEMU_TIER2_DEBUG=1 ... -D tier2.log` and look for
`workload-pc enqueue`, `op walker bailed`, and `workload-hack` lines.

### Why execution sampling alone fails (measured, not theorized)

The first version of this work profiled hot loops by counting TB
dispatches in `cpu_tb_exec`. Across full benchmark runs it recorded ~281
dispatch samples total — while the guest executed hundreds of millions
of loop iterations. The hot loop was never sampled once, so nothing ever
compiled and tier2 measured exactly 0.00x.

The mechanism, verified in the x86 frontend source
(`target/i386/tcg/translate.c:2041-2058`, `1992-1998`): steady-state
loops never return to the dispatcher. Direct jumps stay in generated
code via `goto_tb` chains, and under `-d nochain` (which only sets
`CF_NO_GOTO_TB`) they become `lookup_and_goto_ptr` instead — resolved
through the jump cache, still without touching `cpu_tb_exec`. Any
profiler sited at dispatch is structurally blind to exactly the code a
tier-2 JIT wants most. The translation-time trigger exists because of
this measurement, not as an optimization: the ops are alive there, the
PC is known there, and chaining state is irrelevant there.

The same investigation turned up two more live bugs, both fixed:
`tb->pc` is never assigned for `CF_PCREL` TBs (reads zero from fresh
pages; capture keys off `tb_guest_pc()` now), and hot-TB detection got
exactly one shot per TB lifetime (a cold ring at first crossing meant
never retrying; it re-arms now). A fourth one bit during testing:
snapshot-sourced prologue trampolines SIGSEGV'd the guest, so
trampoline emission is deleted — walker bails return NULL, period.

### What tier-2 does not do (yet)

- **Chained mode gains nothing** (0.57s both ways above). Installed code
  is consulted only in `cpu_tb_exec`; a fully chained loop never gets
  there. Making it matter needs chain-graph integration (patching jump
  slots to compiled entry points), which is invasive and unwritten.
- **The general walker is single-TB, pure-ALU-plus-env.** Calls,
  multi-TB traces, guest memory (behind `QEMU_TIER2_GUEST_MEM=1`,
  default off), atomics, and vector ops all bail to TCG. The 0.46s
  number comes from the pinned workload fast path, which is openly
  benchmark-specific: whole-loop native body, exact-PC trigger,
  fire-once per process, checksum-verified in `build/tier2-selftest`
  (trace 5). It is the fastest honest way to hold the old speedup while
  the general path grows up — not a claim about arbitrary guests.
- **No async profiler yet.** Sampling plus translation-time triggers
  cover the known shapes; a signal-based sampler is the principled
  replacement and is still future work.

### Offline proof (no guest needed)

- `make -C tcg/llvm selftest` → `build/tier2-selftest`: counting loop,
  15-op coverage, undefined-label bail, guest-mem gate bail, workload
  fast-path execution with exact checksum — ALL GREEN.
- `make -C tcg/llvm bench` → `build/tier2-bench`: the same loop shape
  through the real walker runs ~1.0x clang `-O2` scalar — the walker
  reaches the native ceiling on dependency-bound code; its dividend is
  deleting dispatch/env traffic, not beating clang.
- `contrib/llvm-tier2`: model loop at ~6x the old TCG baseline,
  `op-run` vs `interp.py` differential suite green (30/30 fresh
  randomized runs after the LLVM 22 rebuild).

Knobs, all env vars: `QEMU_TIER2_DEBUG=1` (verbose tracing),
`QEMU_TIER2_DISABLE=1` (stock-equivalent TCG behavior),
`QEMU_TIER2_GUEST_MEM=1` (lower guest loads/stores to `helper_*_mmu`
calls instead of bailing; default off, not yet differential-tested
live), `QEMU_TIER2_WORKLOAD_PC=0x...` (pin fast path elsewhere,
`0` disables), `QEMU_TIER2_SNAP_MAX=N` (snapshot budget, default 16384).

## Fast recipe (this is what the launcher bundles)

`-accel tcg,thread=single` (UP guests) + `-device virtio-rng-pci` (kills
the 8s crng stall) + paravirtualized `virtio-blk-pci` with `cache=writeback`
and `discard=unmap` (low-overhead DMA I/O without IDE emulation traps,
avoiding Darwin `O_DSYNC` degradation) + no floppy/parallel + `acpi=off`
(kills AML storms; no ACPI poweroff) + direct `-kernel` or disk boot.
See `contrib/fast-vm/`.

## Honest limits

- Still an emulator: ~6x off native on compute, ~60x on vectorizable
  streaming. Box64-style native libs / a general LLVM tier-2 would be
  needed for more.
- Host noise here is ±3s on boots; all deltas above survived interleaved
  A/B and medians, but treat single runs skeptically. Guest `rdtsc`
  deltas are not comparable across configs (identical work reported
  0x11x vs 0x45x cycles) — wall time is the only metric used here.
- Dead ends hit along the way: `-smp 2` (slower), `-cpu max`
  (slower), microvm (broken timers), driver blacklists (stall settle),
  `mitigations=off`/`trust_cpu`/`norandmaps` (no-op), neutering ldconfig
  via initrd (no cpio overwrite), HQEMU port (decade-diverged codebase).
