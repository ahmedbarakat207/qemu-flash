#!/bin/sh
# Default fast way to boot an x86_64 Linux VM with this patched QEMU.
# Bundles every measured win: tuned machine (virtio-rng, no legacy
# floppy/parallel), direct -kernel boot, and the acpi=off guest flag.
# Code opts (chaining fast-path, JIT-toggle batching, fence elision)
# are compiled in and always on -- nothing to enable.
#
# Usage: sh contrib/fast-vm/run-x86-fast.sh -kernel K -initrd I [-append EXTRA] [-m MB]
# Env overrides: QEMU_BIN.
set -u
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
QEMU_BIN=${QEMU_BIN:-$REPO_ROOT/build/qemu-system-x86_64}
KERNEL=""; INITRD=""; EXTRA=""; MEM=1024
while [ $# -gt 0 ]; do
    case $1 in
        -kernel) KERNEL=$2; shift 2 ;;
        -initrd) INITRD=$2; shift 2 ;;
        -append) EXTRA=$2; shift 2 ;;
        -m) MEM=$2; shift 2 ;;
        *) echo "unknown: $1" >&2; exit 1 ;;
    esac
done
[ -n "$KERNEL" ] || { echo "need -kernel" >&2; exit 1; }

APPEND="console=ttyS0,115200n8 acpi=off $EXTRA"

exec "$QEMU_BIN" -accel tcg,thread=single -M pc -m "$MEM" \
    -kernel "$KERNEL" ${INITRD:+-initrd "$INITRD"} -append "$APPEND" \
    -display none -serial stdio \
    -parallel none \
    -device virtio-rng-pci \
    -global isa-fdc.fdtypeA=none -global isa-fdc.fdtypeB=none \
    -no-reboot

# Notes:
# - thread=single (Round-Robin): one thread runs everything, so there is
#   no vCPU<->IO lock ping-pong or cross-core state bouncing. Fastest for
#   single-vCPU guests (measured -37% vs MTTCG on boot); do not use with
#   -smp > 1 (SMP needs MTTCG to use the extra vCPUs).
# - acpi=off saves ~1s of AML storms. Cost: no ACPI poweroff; stop the VM
#   with quit in the monitor (add -monitor stdio, conflicts with serial
#   stdio -- use -serial file:boot.log in that case) or kill the process.
# - virtio-rng-pci counters guest entropy starvation (crng init stalls).
#   Default e1000 NIC is kept: virtio-net has no driver in minimal
#   initrds (no virtio_pci core) and an unbound NIC stalls udev settle.
# - Single vCPU is fastest for boot (MTTCG sync overhead beats -smp 2).
# - Needs an x86_64 kernel+initrd with serial console (e.g. TinyCorePure64
#   /boot/vmlinuz64 + /boot/corepure64.gz).
