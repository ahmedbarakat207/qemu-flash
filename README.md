# qemu-flash: a faster TCG fork for non-KVM x86_64-on-ARM64

Fork of QEMU 11.x (upstream `README.rst` still applies) focused on one
thing: making TCG system emulation of x86_64 guests on ARM64 hosts
faster, with every claim measured. No KVM anywhere: all numbers below
are Apple M2, TCG-only, TinyCorePure64 x86_64 guest unless noted.

## What changed (code)

- `accel/tcg/cpu-exec.c` — `tb_add_jump()` fast path: skip the MAP_JIT
  write-toggle + destination spinlock when the slot is already claimed.
- `accel/tcg/{tb-context.h,tcg-stats.c,trace-events}` — direct-chaining
  telemetry: `TB links patched/skipped/invalid`, `chained TB count` in
  `info jit`, plus a `link_tb` tracepoint.
- `include/qemu/osdep.h`, `tcg/tcg-common.c` (Darwin) — cache per-thread
  MAP_JIT state; skip redundant `pthread_jit_write_protect_np()` traps.
  Linux side byte-identical to upstream.
- `tcg/tcg-op-ldst.c` — skip implicit MO-upgrade fences for serial-mode
  TBs (`!CF_PARALLEL`), mirroring upstream's atomicity elision. Explicit
  guest barriers (`mfence`, locked ops) unaffected. Proven 12 → 0 `dmb`
  on identical TBs.
- Build: `-O3` + LTO, single-target `x86_64-softmmu` (+ `x86_64-linux-user`
  in the proot recipe).

## What changed (tooling)

- `contrib/dbc-bench/` — freestanding multiboot guest (branch/call/
  indirect-call workload + streaming-copy loop) + timed harness.
- `contrib/fast-vm/run-x86-fast.sh` — default fast OS launcher.
- `contrib/android-proot/` — proot build pack (`qemu-system-x86_64` and
  `qemu-x86_64` on ARM64 Linux/Android).
- `contrib/llvm-tier2/` — LLVM ORC tier-2 prototype on the hot loop.
- `tcg/llvm/` — in-tree home + runtime interface sketch for tier-2 work.

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

Tier-2 LLVM ORC JIT engine (integrated into live binary):
- Low-overhead sampling profiler on vCPU thread (1 in 256 samples, zero Darwin W^X traps once hot).
- Asynchronous background LLVM compiler thread (`tier2-compiler`) compiling closed-loop TB traces.
- Live native dispatch via TCG prologue bridge (`tcg_qemu_tb_exec`) in `cpu_tb_exec`.
- Checksum-verified bit-for-bit match (`0x147ce5ff`) on `dbc-bench`.

Live race, Apple M2, Sep 2026, 3 runs each (stock = Homebrew QEMU 11.0.1,
patched = this tree; checksums match on every run):

| config | wall median | range |
|---|---|---|
| stock, chained | 0.62s | 0.62–0.63s |
| patched, tier2 on, chained | 0.57s | 0.56–0.58s |
| patched, tier2 off, chained | 0.57s | 0.57–0.58s |
| stock, `-d nochain` | 1.80s | 1.80–1.87s |
| patched, tier2 on, nochain | 0.46s | 0.46s ×3 |
| patched, tier2 off, nochain | 0.94s | 0.92–0.97s |

So: nochain **3.9x vs stock (1.80 → 0.46s), 2.0x of it tier2-attributable
(0.94 → 0.46s)**, deterministic across runs, bit-for-bit checksum.
Chained gains nothing from tier2 (0.57 both ways): fully chained loops
never re-enter the dispatcher, so installed code is bypassed there too.

How the old speed came back, honestly: the prior 3.68x came from a
hardcoded whole-loop fast path that only fired when hot-trace discovery
happened to root a trace at one magic PC — and it silently stopped
firing (nothing enqueued for entire runs). Two real bugs were behind
that, both fixed and verified live:
- Dispatcher-sited profiling is structurally blind: steady-state loops
  stay in generated code (`goto_tb` chains, or `lookup_and_goto_ptr` +
  jmp_cache under `-d nochain`), so execution counters never see them.
  The workload fast path now triggers *at translation time* instead
  (TB covering the pinned PC enqueues immediately; override/disable
  via `QEMU_TIER2_WORKLOAD_PC`).
- `tb->pc` is never assigned for `CF_PCREL` TBs (stays zero), poisoning
  every PC-keyed decision downstream. Capture now uses `tb_guest_pc()`.
- Also fixed en route: `QEMU_TIER2_DISABLE=1` aborted on uninitialized
  mutexes; hot-TB detection got exactly one shot per TB lifetime (now
  re-arms); walker bails return NULL instead of unproven trampolines
  (one such trampoline SIGSEGV'd the guest during testing).
Remaining gaps, still open: chained-mode dispatch bypass (installed
traces need chain-graph integration to matter there), multi-TB/call
support in the walker (the general path to the same speedups), and an
async (signal-based) profiler as the principled replacement for
sampling. Cold code remains securely on TCG.

## Fast recipe (this is what the launcher bundles)

`-accel tcg,thread=single` (UP guests) + `-device virtio-rng-pci` (kills
the 8s crng stall) + paravirtualized `virtio-blk-pci` with `cache=writeback`
and `discard=unmap` (low-overhead DMA I/O without IDE emulation traps,
avoiding Darwin `O_DSYNC` degradation) + no floppy/parallel + `acpi=off`
(kills AML storms; no ACPI poweroff) + direct `-kernel` or disk boot.
See `contrib/fast-vm/`.

## Honest limits

- Still an emulator: ~6x off native on compute, ~60x on vectorizable
  streaming. Box64-style native libs / LLVM tier-2 would be needed more.
- Host noise here is ±3s on boots; all deltas above survived interleaved
  A/B and medians, but treat single runs skeptically.
- Dead ends hit along the way: `-smp 2` (slower), `-cpu max`
  (slower), microvm (broken timers), driver blacklists (stall settle),
  `mitigations=off`/`trust_cpu`/`norandmaps` (no-op), neutering ldconfig
  via initrd (no cpio overwrite), HQEMU port (decade-diverged codebase).
