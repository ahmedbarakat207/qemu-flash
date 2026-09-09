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
- `accel/tcg/tlb-bounds.h` — SoftMMU dynamic TLB bounds: optimized default and
  minimum bounds (256/64 entries) to prevent cache thrashing and memset storms during
  frequent 32-bit CR3 page-table switches (e.g. Windows XP boot), while retaining
  automatic dynamic expansion under sustained multi-process workloads.
- `tcg/llvm/tier2.c`, `tier2.h` — Boot churn & transient loop filtering: skips
  real-mode / sub-1MB SeaBIOS delay loops and tunes `TIER2_HOT_THRESHOLD` to 50,000,
  eliminating LLVM compile storms and thread contention during OS boot while preserving
  maximum acceleration for sustained workloads.
- `system/vl.c` — High-performance default I/O options for `-hda`: automatically
  configures `-hda`/`-hdb`/`-hdc`/`-hdd` with `cache=unsafe,aio=threads` (and `format=raw`
  for `.img`/`.raw` disk images), eliminating host `fsync` stalls, thread blocking, and
  probing warnings during OS boot.
- `include/ui/console.h`, `ui/console.c`, `ui/sdl2.c`, `ui/sdl2-2d.c` — Full 60 FPS
  display pipeline & zero-stutter presentation engine:
  - Native Cocoa priority on macOS matching stock QEMU smoothness by default, with full
    support for `-display sdl` powered by the non-blocking 60 FPS engine.
  - 60 FPS refresh rate: reduced `GUI_REFRESH_INTERVAL_DEFAULT` from 30ms to 16ms.
  - Dirty rect batching: decoupled scanline slice updates from presentation; presents
    once per frame instead of 5–20 blocking presents per refresh.
  - Non-blocking OpenGL renderer on macOS: drops presentation time from 7.92ms
    (Metal blocking vsync) down to 0.43ms (asynchronous buffer swap), freeing the
    main thread and eliminating vCPU starvation.
  - Guaranteed inter-frame spacing in `ui/console.c` preventing 1ms timer starvation bursts.
  - Autonomous guest rendering detection: guest updates keep refresh active at 16ms.
- `tcg/llvm/` — in-tree Tier-2 JIT (Phases 1–7): Phase 5 chain-graph dynamic re-linking
  (`goto_tb` patching directly to native Tier-2 stubs), targeted invalidation
  (prevents invalidation avalanches during guest boot), and complete integer ALU op
  coverage (`mulsh`, `muluh`, `andc`, `orc`, `clz`, `ctz`).
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

## Performance (medians; re-measured Sep 8 2026, Apple M2, checksums match everywhere)

Method: stock is Homebrew QEMU 11.0.1, patched is this tree (11.1.50).
OS boots: TinyCorePure64 15.x (`vmlinuz64`+`corepure64.gz`, md5-verified),
`-kernel`/`-initrd` direct boot, host wall time to the `login[` getty
marker, 5 runs per config round-robin interleaved (RR rows: separate
cool-host alternating A/B, 4 runs each). In-guest workloads run from
`/opt/bootlocal.sh` inside the guest at boot; times are guest `time`
reals, awk checksum `688238872` identical on both binaries.
Microbench: `contrib/dbc-bench`, 5 runs per config round-robin.

OS boot to shell (TinyCorePure64):

| config | boot (median, 5 runs) |
|---|---|
| stock QEMU, defaults | 5.3s (5.09, 5.11, 5.30, 19.42, 26.82 — last two heat-polluted, see limits) |
| patched, tuned flags | 4.5s (4.08, 4.28, 4.48, 8.78, 12.12) |
| patched, tuned + `acpi=off` | 4.5s (3.88, 4.08, 4.51, 6.51, 7.16; stock: 5.1s same flags) |
| **patched, Round-Robin (`-accel tcg,thread=single`)** | **4.0s** (4 runs: 3.88, 3.88, 4.08, 4.09; stock RR: 4.9s) |

Paired result, robust to the noise: in every one of 15 interleaved
same-flags boot rounds, patched beat stock (typical gap 0.5–1.5s).
The Sep 6 `20.1s` stock-defaults figure does **not** reproduce under
controlled interleaved runs — it was most likely a loaded-host /
cold-start artifact (our own matrix shows +10–20s under sustained
load). Current story is `5.3s → 4.0s` (−25%), not `20s → 4s`.

Inside a booted OS (same tuned flags both binaries unless noted; 4 runs each):

| workload | stock defaults | stock tuned | patched tuned | delta (patched vs stock defaults) |
|---|---|---|---|---|
| awk compute 600k iters | 6.64s | 5.47s | 4.14s | **−38%** (1.32x vs tuned stock) |
| tmpfs write 256MB | 0.46s | 0.46s | 0.35s | **−24%** (1.31x vs tuned stock) |
| tmpfs read 256MB | 0.14s | 0.15s | 0.08s | **−43%** (1.9x vs tuned stock) |
| 120 fork+exec | 0.76s | 0.83s | 0.67s | **−12%** (1.24x vs tuned stock) |
| microbench dispatch-heavy (`-d nochain`) | 1.77s | — | 0.46s | **3.8x faster** |
| microbench chained execution | 0.62s | — | 0.40s | **1.5x faster** |

Dropped vs the previous revision: the `streaming copy (guest cycles)`
row. Fresh data proves guest `rdtsc` deltas track wall time, not work —
identical guest work reports 305M cycles chained vs 1147M under
`-d nochain` on the same stock binary — so cycle deltas are not a
speedup metric. Wall time is the only metric used here.

## Tier-2 LLVM JIT: how it works and what it actually buys

The short version first, because everything below is elaboration: hot
guest loops get re-compiled by LLVM on a background thread and executed
as native code instead of TCG output. With direct block chaining integration
(Phase 5 `goto_tb` re-linking) and LLVM ORC JIT optimization, `contrib/dbc-bench`
achieves **1.55x wall time over stock in chained mode (0.400s vs 0.619s)** and
**3.83x wall time over stock under `-d nochain` (0.461s vs 1.766s)**, with
bit-for-bit checksum correctness (`sum=0x0000000000001768`). A
`QEMU_TIER2_DISABLE=1` control (patched binary, tier2 off) shows the base TCG
patches alone give 1.09x chained / 1.98x nochain over stock; tier2 adds
1.41x / 1.77x on top of that. The rest of this
section explains how the machinery fits together, the architecture across
all roadmap phases, and the engineering details that make it fast and safe.

### The pipeline, end to end

1. **Capture** (`tcg/tcg.c:6657`). At the end of `tcg_gen_code()`, while
   the optimized op list is still alive, `tier2_capture_tb_ops()` walks
   the `QTAILQ` op chain and the temp table and serializes them into a
   stable C-ABI record (`Tier2TBRec`: up to 512 temps, 1024 ops per TB;
   larger TBs are skipped). Values become compact temp indices, guest
   accesses keep their exact `TCGTemp.mem_offset` byte offsets, branch
   targets keep label ids, conditions map to a 12-value enum. The JIT
   side never includes a TCG or target header, so no guest struct layout
   is hardcoded anywhere in it.
2. **Trigger & Profiling** (`tcg/llvm/tier2-prof.c`, `tier2.c`). Two profilers
   feed the background compiler without blind spots:
   - *Async Signal-Based Sampler (Phase 6):* A POSIX `SIGPROF` host timer
     samples the vCPU thread's guest PC at 500 Hz (`QEMU_TIER2_PROF_HZ`),
     detecting tight steady-state loops that execute entirely inside chained
     code with zero dispatch-path overhead.
   - *Sampled Execution Profiler:* Samples dispatch frequency (`tier2_profile_sample`,
     1 in 256 dispatches, `exec_count` threshold 10000).
   - *Structural Workload Trigger:* Translation-time trigger for target PCs
     (`QEMU_TIER2_WORKLOAD_PC`).
3. **Compile** (background `tier2-compiler` thread, `tcg/llvm/tier2.c`).
   The worker copies snapshot records under mutex protection and invokes
   `libqemu-tier2.dylib`. It leverages:
   - *Multi-TB CFG Fusion (Phase 1):* Connects extended basic blocks into
     a unified LLVM CFG, allowing LLVM's `mem2reg`, `EarlyCSE`, `LICM`,
     and `GVN` passes to promote guest registers to native host registers
     across block boundaries.
   - *Persistent On-Disk Object Cache (Phase 2):* CityHash64 cryptographic
     keys cache native object files to `~/.cache/qemu/tier2/*.o`. Repeated
     guest executions bypass LLVM compilation in <100μs.
   - *Flat RAM Direct Pointer Lowering (Phase 3):* Translates guest loads and
     stores within identity-mapped RAM directly into host pointer offsets,
     bypassing SoftMMU TLB lookups.
   - *NEON Vector Lowering (Phase 4):* Transpiles x86 SSE/AVX vector operations
     directly to ARM64 NEON instructions (`v0–v31`) via LLVM `FixedVectorType`.
4. **Install & Chain Re-linking** (`accel/tcg/cpu-exec.c`, `tcg/llvm/tier2.c`).
   On successful compilation, the worker validates that the header TB is still
   valid, stores the native function pointer in `tb->tier2_code`, and emits a
   dedicated ARM64 chain stub (`tb->tier2_stub`).
   - *Phase 5 Chain Re-linking:* Scans all predecessor TBs that jump to this
     header (`tb->jmp_list_head`) and dynamically patches their `goto_tb` jump
     slots to target the native chain stub instead of the TCG block. Chained
     guest loops now jump directly into native machine code without touching
     the dispatcher.
5. **Dispatch & HLE Shims** (`accel/tcg/cpu-exec.c`, `linux-user/hle-thunks.c`).
   `cpu_tb_exec` executes native code via `fn(cpu_env(cpu))` with full
   register preservation, guarded by `CF_INVALID` and breakpoint checks.
   For `linux-user` execution (Phase 7), High-Level Emulation (HLE) shims
   intercept guest standard C library and math functions (`sin`, `cos`, `pow`,
   `memcpy`, crypto) and execute host-native ARM64 implementations directly.
6. **Invalidation.** Any TB invalidation safely unlinks predecessor jump slots
   via `tb_reset_jump`, drops snapshots, retires ORC JIT resources, and frees
   stubs at `tb_flush` or shutdown under strict Darwin W^X safety.

### Live race (Apple M2, Sep 8 2026, 5 runs each, round-robin interleaved)

Stock is Homebrew QEMU 11.0.1, patched is this tree. Wall time covers
SeaBIOS + workload; guest checksums (`sum=0x0000000000001768` mem,
`sum=0x00000000147ce5ff` compute) match on every run:

| config | wall median | 5-run samples | vs stock |
|---|---|---|---|
| stock, chained | 0.619s | 0.612, 0.617, 0.619, 0.620, 0.622 | 1.00x baseline |
| **patched, tier2 on, chained** | **0.400s** | **0.391, 0.392, 0.400, 0.402, 0.407** | **1.55x faster** |
| patched, tier2 off, chained (`QEMU_TIER2_DISABLE=1`) | 0.566s | 0.559, 0.562, 0.566, 0.617, 0.620 | 1.09x faster |
| stock, `-d nochain` | 1.766s | 1.752, 1.761, 1.766, 1.769, 1.832 | 1.00x baseline |
| **patched, tier2 on, nochain** | **0.461s** | **0.457, 0.457, 0.461, 0.461, 0.461** | **3.83x faster** |
| patched, tier2 off, nochain | 0.891s | 0.883, 0.888, 0.891, 0.936, 0.942 | 1.98x faster |

Note: the Sep 7 revision claimed 0.201s / 0.296s for patched tier2-on.
That does not reproduce on current HEAD (stable 0.39–0.41s / 0.46s
across 10+ runs in two harnesses, tier2 verified engaged via
`QEMU_TIER2_DEBUG=1`); the old figures are replaced, not averaged.
Guest `rdtsc` cycle deltas are deliberately not reported: identical
work measures 305M chained vs 1147M nochain on stock, i.e. the
virtual TSC tracks wall time, not retired work.

Reproduce with: `./contrib/dbc-bench/run-bench.sh build/qemu-system-x86_64 /opt/homebrew/bin/qemu-system-x86_64 5`
(or standalone: `./build/qemu-system-x86_64 -M pc -m 128 -kernel contrib/dbc-bench/kernel.elf -display none -serial stdio -device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot`).
Watch live JIT activity with `QEMU_TIER2_DEBUG=1 ... -D tier2.log`.

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

### What tier-2 delivers (Phases 1–7 complete)

- **Chained mode re-linking (Phase 5):** Re-links `goto_tb` jump slots to
  dedicated native chain stubs (`tier2_stub`). Chained loops execute
  in native ARM64 machine code across block boundaries without
  bouncing back to `cpu_tb_exec` (0.400s median, 1.55x over stock;
  tier2-off control at 0.566s shows 1.41x of that is tier2, the rest
  is the base TCG patch set).
- **Multi-TB CFG fusion (Phase 1):** Fuses inner loops across multiple basic
  blocks, eliminating intermediate env register commits via LLVM SSA optimization.
- **Direct Flat RAM pointer lowering (Phase 3):** Bypasses SoftMMU TLB lookups
  for direct-mapped guest memory and stack operations, lowering to direct host
  virtual pointer arithmetic.
- **x86 SSE/AVX to ARM64 NEON transpilation (Phase 4):** Maps vector operations
  directly to LLVM vector types (`<4 x i32>`, `<2 x i64>`, etc.), compiling to
  native 128-bit NEON instructions (`v0–v31`).
- **Persistent on-disk JIT cache (Phase 2):** Caches compiled trace objects to
  `~/.cache/qemu/tier2/*.o` via cryptographic hashing (CityHash64), reducing
  compilation time from 25–50ms to <100μs on warm boots.
- **Async signal-based profiler (Phase 6):** POSIX `SIGPROF` timer sampling
  guest PCs at 500 Hz, eliminating inline counter overhead and detecting
  tight steady-state loops that bypass the dispatcher.
- **High-Level Emulation (HLE) shims (Phase 7):** Symbol interception for
  `linux-user` guest libraries, dispatching math, string, and crypto operations
  directly to host native ARM64 libraries.

### Offline proof (no guest needed)

- `make -C tcg/llvm selftest` → `build/tier2-selftest`: All 19 tests ALL GREEN:
  - Trace 1: Counting loop
  - Trace 2: Op coverage (15+ core ALU ops)
  - Trace 3: Undefined-label bail
  - Trace 4: Guest-mem gate bail
  - Trace 5: Workload-PC fallback
  - Trace 6: Multi-TB loop fusion
  - Trace 7 & 8: Side-exit contracts
  - Trace 9: Helper calls
  - Trace 10: bswap / negsetcond
  - Trace 11: TLB-hit load
  - Trace 12: TLB-miss slow path
  - Trace 13: Store hit + unaligned slow path
  - Trace 14: Prologue tail-call
  - Trace 15: On-disk cache round-trip
  - Trace 16: Native self-loop & safepoint poll
  - Trace 17: NEON vector SIMD
  - Trace 18: HLE library shims
  - Trace 19: mulsh / muluh / andc / orc / clz / ctz (full ALU coverage)
- `make -C tcg/llvm bench` → `build/tier2-bench`: 8M-iter loop running at 1.80x
  speedup (10ms exec best-of-3, 6ms warm compile, 18ms scalar reference,
  checksum OK `acc=0x608ca391f307f1`).
- `contrib/llvm-tier2`: model loop at ~6x the old TCG baseline
  (re-measured Sep 8: `ssa` 79ms, `env` 79ms, checksum `0x147ce5ff` OK),
  `op-run` vs `interp.py` differential suite green (30/30 fresh
  randomized runs with edge values, Sep 8 2026).

Knobs, all env vars: `QEMU_TIER2_DEBUG=1` (verbose tracing),
`QEMU_TIER2_DISABLE=1` (stock-equivalent TCG behavior),
`QEMU_TIER2_GUEST_MEM=1` (lower guest loads/stores to direct RAM or `helper_*_mmu`),
`QEMU_TIER2_WORKLOAD_PC=0x...` (pin fast path elsewhere, `0` disables),
`QEMU_TIER2_SNAP_MAX=N` (snapshot budget, default 16384),
`QEMU_TIER2_PROF_HZ=N` (profiler frequency, default 500 Hz).

## Full 60 FPS SDL Display Subsystem & Zero-Stutter Architecture

Stock QEMU's graphical UI feels sluggish (running under ~20 FPS) due to hardcoded 30ms refresh timers, autonomous rendering throttling, and per-dirty-rectangle present serialization stalls. This fork completely overhauls the display pipeline:

1. **Full 60 FPS Refresh Rate**: `GUI_REFRESH_INTERVAL_DEFAULT` in `include/ui/console.h` is reduced from 30ms to 16ms (62.5 Hz), perfectly covering 60 Hz display refresh cycles.
2. **Batch 2D Presentation**: In stock QEMU, `sdl2_2d_update()` called `SDL_RenderPresent()` for every single dirty scanline slice (5–20 presents per frame). In `qemu-flash`, dirty rectangles are uploaded via `SDL_UpdateTexture()`, and `sdl2_2d_refresh()` batches and presents the entire frame exactly once.
3. **Non-Blocking OpenGL Engine on macOS**: On macOS, SDL's default Metal backend blocks inside `[CAMetalLayer nextDrawable]` for ~7.92ms per present call, freezing the QEMU main thread for ~50% of every frame. Explicitly configuring `SDL_SetHint(SDL_HINT_RENDER_DRIVER, "opengl")` and `SDL_SetHint(SDL_HINT_RENDER_VSYNC, "0")` drops presentation time to **0.43ms (18x faster)**.
4. **Zero-Stutter Inter-Frame Timing**: The GUI refresh timer reschedules from the current completion timestamp (`timer_mod(ds->gui_timer, ds->last_update + interval)`), ensuring the guest vCPU always has a clean, uninterrupted 16ms execution window between display updates, eliminating timer bunching and mouse stutter.
5. **Autonomous Activity Detection**: Screen updates from video playback, animations, or guest games (`scon->updates > 0`) automatically keep the refresh interval active at 16ms without requiring continuous host mouse or keyboard activity.

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
- Host noise dominates boots, far beyond the old ±3s estimate: in the
  Sep 8 matrix, rounds 3–4 (after ~25 back-to-back boots on a fanless
  M2 Air) ran +5–20s slower across *all* configs simultaneously
  (e.g. stock-defaults hit 26.8s). Interleaved A/B with medians and
  full sample lists is load-bearing — treat any single boot run,
  including the old `20.1s` stock figure, skeptically.
- Guest `rdtsc` deltas are not comparable across configs (identical
  work reported 305M cycles chained vs 1147M under `-d nochain` on
  stock) — wall time is the only metric used here, and the old
  cycle-reduction rows are deleted for that reason.
- Stock baseline is Homebrew QEMU 11.0.1 vs tree 11.1.50: part of the
  "base patch" delta may be upstream drift between those versions,
  not just this fork's patches. The `QEMU_TIER2_DISABLE=1` control
  isolates tier2 from everything else, but not this fork from upstream.
- Dead ends hit along the way: `-smp 2` (slower), `-cpu max`
  (slower), microvm (broken timers), driver blacklists (stall settle),
  `mitigations=off`/`trust_cpu`/`norandmaps` (no-op), neutering ldconfig
  via initrd (no cpio overwrite), HQEMU port (decade-diverged codebase).
