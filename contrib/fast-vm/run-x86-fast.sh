#!/bin/sh
# Default fast way to boot an x86_64 Linux VM with this patched QEMU.
# Bundles every measured win: tuned machine (virtio-rng, no legacy
# floppy/parallel), direct -kernel boot, and the acpi=off guest flag.
# Code opts (chaining fast-path, JIT-toggle batching, fence elision)
# are compiled in and always on -- nothing to enable.
#
# Usage: sh contrib/fast-vm/run-x86-fast.sh [-kernel K -initrd I] [-drive D] [-append EXTRA] [-m MB] [-smp N] [-net]
# Env overrides: QEMU_BIN, CACHE (default: writeback), THREAD (single|multi).
set -u
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
QEMU_BIN=${QEMU_BIN:-$REPO_ROOT/build/qemu-system-x86_64}
KERNEL=""; INITRD=""; DRIVE=""; EXTRA=""; MEM=1024
CACHE=${CACHE:-writeback}
NET=0
SMP=1
THREAD=""
while [ $# -gt 0 ]; do
    case $1 in
        -kernel) KERNEL=$2; shift 2 ;;
        -initrd) INITRD=$2; shift 2 ;;
        -drive|-disk) DRIVE=$2; shift 2 ;;
        -append) EXTRA=$2; shift 2 ;;
        -m) MEM=$2; shift 2 ;;
        -smp) SMP=$2; shift 2 ;;
        -thread) THREAD=$2; shift 2 ;;
        -net) NET=1; shift ;;
        *) echo "unknown: $1" >&2; exit 1 ;;
    esac
done
[ -n "$KERNEL" ] || [ -n "$DRIVE" ] || { echo "need -kernel or -drive/-disk" >&2; exit 1; }

# Fast kernel cmdline:
# - quiet loglevel=3: eliminates thousands of serial printk I/O port traps during boot.
# - tsc=reliable no_timer_check nowatchdog: skips TSC calibration loops and timer IRQ stalls.
APPEND="console=ttyS0,115200n8 acpi=off quiet loglevel=3 tsc=reliable no_timer_check nomce nowatchdog $EXTRA"

# Auto-select thread mode:
# - SMP=1: thread=single (Round-Robin) is fastest for UP guests (elided barriers, 0 sync overhead).
# - SMP>1: thread=multi (MTTCG) is required so multiple vCPUs run on dedicated host threads.
if [ -z "$THREAD" ]; then
    if [ "$SMP" -gt 1 ] 2>/dev/null; then
        THREAD="multi"
    else
        THREAD="single"
    fi
fi

set -- "$QEMU_BIN" -accel tcg,thread="$THREAD" -M pc,smm=off,vmport=off -m "$MEM"
if [ "$SMP" -gt 1 ] 2>/dev/null; then
    set -- "$@" -smp "$SMP"
fi
if [ -n "$KERNEL" ]; then
    set -- "$@" -kernel "$KERNEL"
    [ -n "$INITRD" ] && set -- "$@" -initrd "$INITRD"
    set -- "$@" -append "$APPEND"
fi
if [ -n "$DRIVE" ]; then
    # Paravirtualized virtio-blk with host-tuned caching and thin-provisioning:
    # - cache=writeback: uses host page cache; avoids macOS O_DSYNC crawl (cache=none on Darwin).
    # - discard/detect-zeroes: passes TRIM/unmap through to host storage (APFS).
    # - thread=single: keeps I/O on the main loop, avoiding cross-thread lock ping-pong.
    set -- "$@" -drive "file=$DRIVE,if=none,id=hd0,cache=$CACHE,discard=unmap,detect-zeroes=unmap" \
               -device virtio-blk-pci,drive=hd0
fi
set -- "$@" \
    -vga none -display none -serial stdio \
    -boot menu=off,strict=on \
    -parallel none \
    -device virtio-rng-pci \
    -global isa-fdc.fdtypeA=none -global isa-fdc.fdtypeB=none \
    -no-reboot

if [ "$NET" = 1 ]; then
    set -- "$@" -device virtio-net-pci,netdev=n0 -netdev user,id=n0
else
    set -- "$@" -nic none
fi

exec "$@"

# Notes:
# - thread=single (Round-Robin): one thread runs everything, so there is
#   no vCPU<->IO lock ping-pong or cross-core state bouncing. Fastest for
#   single-vCPU guests (measured -37% vs MTTCG on boot); do not use with
#   -smp > 1 (SMP needs MTTCG to use the extra vCPUs).
# - -vga none + smm=off + vmport=off + boot menu=off: removes unused emulated
#   hardware (PCI VGA, VGA ROM real-mode execution, SMRAM relocation, VMware ports).
# - virtio-blk-pci: paravirtualized I/O avoids costly IDE/AHCI register-level
#   emulation in TCG. Combined with cache=writeback (or CACHE=unsafe for
#   ephemeral bench runs) for maximum throughput on macOS hosts.
# - quiet loglevel=3: under TCG, serial output is byte-by-byte port I/O (outb 0x3f8),
#   causing tens of thousands of TB exits. Silencing boot logs cuts 1-2s of boot time.
# - acpi=off saves ~1s of AML storms. Cost: no ACPI poweroff; stop the VM
#   with quit in the monitor (add -monitor stdio, conflicts with serial
#   stdio -- use -serial file:boot.log in that case) or kill the process.
# - virtio-rng-pci counters guest entropy starvation (crng init stalls).
# - Single vCPU is fastest for boot (MTTCG sync overhead beats -smp 2).
# - Needs an x86_64 kernel+initrd with serial console (e.g. TinyCorePure64
#   /boot/vmlinuz64 + /boot/corepure64.gz) or a bootable disk image.
