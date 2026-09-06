# qemu-dbc-jit: a faster TCG fork for non-KVM x86_64-on-ARM64

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

Tier-2 ceiling (offline prototype, checksum-verified): 6.5x over TCG on
the hot loop, ≈ native speed. Consistent with HQEMU's published 2.6x
(their baseline was 2012 QEMU; their FP win is already in modern TCG).
Not integrated — cold code must stay on TCG.

## Fast recipe (this is what the launcher bundles)

`-accel tcg,thread=single` (UP guests) + `-device virtio-rng-pci` (kills
the 8s crng stall) + no floppy/parallel + `acpi=off` (kills AML storms;
no ACPI poweroff) + direct `-kernel` boot. See `contrib/fast-vm/`.

## Honest limits

- Still an emulator: ~6x off native on compute, ~60x on vectorizable
  streaming. Box64-style native libs / LLVM tier-2 would be needed more.
- Host noise here is ±3s on boots; all deltas above survived interleaved
  A/B and medians, but treat single runs skeptically.
- Dead ends hit along the way: `-smp 2` (slower), `-cpu max`
  (slower), microvm (broken timers), driver blacklists (stall settle),
  `mitigations=off`/`trust_cpu`/`norandmaps` (no-op), neutering ldconfig
  via initrd (no cpio overwrite), HQEMU port (decade-diverged codebase).
