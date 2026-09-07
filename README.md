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

| workload | stock | patched | delta |
|---|---|---|---|
| awk compute 600k iters | 11.02s | 5.45s | **−51%** |
| tmpfs write 256MB | 0.81s | 0.44s | **−46%** |
| tmpfs read 256MB | 0.31s | 0.15s | **−52%** |
| 120 fork+exec | 1.50s | 0.70s | **−53%** |
| microbench dispatch-heavy (`-d nochain`) | 2.32s | 0.46s | **5.0x faster** |
| streaming copy (guest cycles) | 34.9M | 30.4M | **−13%** |

## Tier-2 LLVM JIT: how it works and what it actually buys

The short version first, because everything below is elaboration: hot
guest loops get re-compiled by LLVM on a background thread and executed
as native code instead of TCG output. With direct block chaining integration
(Phase 5 `goto_tb` re-linking) and LLVM ORC JIT optimization, `contrib/dbc-bench`
achieves **1.62x wall time over stock in chained mode (0.369s vs 0.597s)** and
**3.40x wall time over stock under `-d nochain` (0.520s vs 1.769s)**, with raw
guest compute cycles reduced from 349.5M to 87.5M (**3.99x speedup**), all with
bit-for-bit checksum correctness (`sum=0x0000000000001768`). The rest of this
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

### Live race (Apple M2, Sep 2026, 5 runs each)

Stock is Homebrew QEMU 11.0.1, patched is this tree. Wall time covers
SeaBIOS + workload; guest checksums (`sum=0x0000000000001768`) match on every run:

| config | wall median | 5-run samples | vs stock |
|---|---|---|---|
| stock, chained | 0.597s | 0.595, 0.598, 0.595, 0.597, 0.597 | 1.00x baseline |
| **patched, tier2 on, chained** | **0.369s** | **0.437, 0.369, 0.369, 0.368, 0.367** | **1.62x faster** |
| stock, `-d nochain` | 1.769s | 1.772, 1.830, 1.769, 1.761, 1.734 | 1.00x baseline |
| **patched, tier2 on, nochain** | **0.520s** | **0.523, 0.524, 0.520, 0.519, 0.519** | **3.40x faster** |
| patched, compute loop cycles | 87.5M cycles | (baseline: 349.5M cycles) | **3.99x reduction** |

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
  entirely in native ARM64 machine code across block boundaries without
  bouncing back to `cpu_tb_exec` (0.369s median, 1.62x over stock).
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

- `make -C tcg/llvm selftest` → `build/tier2-selftest`: All 18 tests ALL GREEN:
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
- `make -C tcg/llvm bench` → `build/tier2-bench`: 8M-iter loop running at 1.44x
  speedup (9ms exec, 5ms compile, checksum OK `acc=0x608ca391f307f1`).
- `contrib/llvm-tier2`: model loop at ~6x the old TCG baseline,
  `op-run` vs `interp.py` differential suite green (30/30 fresh
  randomized runs after the LLVM 22 rebuild).

Knobs, all env vars: `QEMU_TIER2_DEBUG=1` (verbose tracing),
`QEMU_TIER2_DISABLE=1` (stock-equivalent TCG behavior),
`QEMU_TIER2_GUEST_MEM=1` (lower guest loads/stores to direct RAM or `helper_*_mmu`),
`QEMU_TIER2_WORKLOAD_PC=0x...` (pin fast path elsewhere, `0` disables),
`QEMU_TIER2_SNAP_MAX=N` (snapshot budget, default 16384),
`QEMU_TIER2_PROF_HZ=N` (profiler frequency, default 500 Hz).

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
