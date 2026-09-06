#!/bin/sh
# Boot an x86_64 Linux VM under the proot-built QEMU with the fast flags
# found by profiling (virtio-rng for entropy, no legacy floppy/parallel).
#
# Usage: sh contrib/android-proot/run-x86-vm.sh -kernel K -initrd I [-append A] [-m MB]
set -u
QEMU=${QEMU_BIN:-build/qemu-system-x86_64}
KERNEL=""; INITRD=""; APPEND="console=ttyS0,115200n8 acpi=off"; MEM=1024
while [ $# -gt 0 ]; do
    case $1 in
        -kernel) KERNEL=$2; shift 2 ;;
        -initrd) INITRD=$2; shift 2 ;;
        -append) APPEND=$2; shift 2 ;;
        -m) MEM=$2; shift 2 ;;
        *) echo "unknown: $1" >&2; exit 1 ;;
    esac
done
[ -n "$KERNEL" ] || { echo "need -kernel" >&2; exit 1; }

exec "$QEMU" -M pc -m "$MEM" \
    -kernel "$KERNEL" ${INITRD:+-initrd "$INITRD"} -append "$APPEND" \
    -display none -serial stdio \
    -parallel none \
    -device virtio-rng-pci \
    -global isa-fdc.fdtypeA=none -global isa-fdc.fdtypeB=none \
    -device virtio-net-pci,netdev=n0 -netdev user,id=n0 \
    -no-reboot

# Notes:
# - x86_64 Linux APPS (not a full OS) run far faster via linux-user, no VM:
#     build/qemu-x86_64 ./your-x86_64-binary
#   (needs x86_64 dynamic loader + libs reachable via -L /path/to/sysroot)
# - No KVM exists on Android: TCG only. Single vCPU is fastest for boot
#   (MTTCG sync overhead beats parallel gains on small guests).
