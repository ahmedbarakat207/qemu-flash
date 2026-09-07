# Patched QEMU on Android (proot)

Runs on: Termux + `proot-distro` Ubuntu/Debian (aarch64), or any aarch64
Linux. No KVM on Android, so TCG only — same as this repo's Mac setup.

## 1. Get the source onto the phone

From your Mac (this repo), push to a remote first:

    git push <your-fork> master

Then inside proot (needs ~3GB free):

    proot-distro install ubuntu
    proot-distro login ubuntu
    apt update && apt install -y git
    git clone --depth 1 <your-fork-url> qemu && cd qemu

## 2. Build

    sh contrib/android-proot/build-proot.sh

Low-RAM phones: add nothing (LTO stays off). With 8GB+ RAM you can pass
`--lto` for a slightly faster binary. Build takes ~10-25 min on modern SoCs.

## 3. Run

Full x86_64 VM (needs an x86_64 kernel + initrd, e.g. TinyCorePure64):

    sh contrib/android-proot/run-x86-vm.sh -kernel vmlinuz64 -initrd corepure64.gz

Single x86_64 app, much faster (no OS emulation — the Box64-style path):

    build/qemu-x86_64 ./hello-x86_64

For dynamic apps point at an x86_64 sysroot: `build/qemu-x86_64 -L /x86root ./app`.

## Notes

- The repo's patches are portable: chaining fast-path + `info jit`
  telemetry work on Linux as-is; the MAP_JIT toggle batching is
  Darwin-only (on Linux those calls are already no-ops).
- Prefer `qemu-x86_64` (linux-user) over system mode whenever you only
  need apps: no softMMU, no device emulation, roughly an order of
  magnitude faster. For games/apps with heavy libc use, Box64 (native
  Termux package) may still win via wrapped libs.
- proot adds ptrace overhead per host syscall; steady-state JIT code is
  unaffected. If possible, run `qemu-x86_64` directly in Termux instead
  of inside proot.
- **Automatic Big-Core Pinning (Built-in)**: The QEMU binary automatically
  detects heterogeneous CPU clusters via sysfs (`cpuinfo_max_freq` and
  `cpu_capacity`) at startup and pins itself to the big/prime cores via
  `sched_setaffinity()`. All vCPU and worker threads inherit this affinity.
  Override with `QEMU_PIN_CORES=4-7` or disable with `QEMU_PIN_CORES=off`.
- The launcher defaults to `quiet loglevel=3` and `-nic none` because every
  serial `write()` and socket syscall triggers a `ptrace` context switch
  through PRoot. Pass `-net` only when network access is required.
