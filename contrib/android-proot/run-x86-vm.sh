#!/bin/sh
# Boot an x86_64 Linux VM under the proot-built QEMU with the fast flags
# found by profiling (virtio-rng for entropy, no legacy floppy/parallel).
#
# Usage: sh contrib/android-proot/run-x86-vm.sh [-kernel K -initrd I] [-drive D] [-append EXTRA] [-m MB] [-smp N] [-net]
# Env overrides: QEMU_BIN, CACHE (default: writeback), THREAD (single|multi).
set -u
QEMU=${QEMU_BIN:-build/qemu-system-x86_64}
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

# In PRoot, every host syscall (e.g. write() to serial) incurs a ptrace trap.
# - quiet loglevel=3 cuts thousands of serial write traps during boot.
# - tsc=reliable no_timer_check nowatchdog avoids delay loops and timer IRQs.
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

set -- "$QEMU" -accel tcg,thread="$THREAD" -M pc,smm=off,vmport=off -m "$MEM"
if [ "$SMP" -gt 1 ] 2>/dev/null; then
    set -- "$@" -smp "$SMP"
fi
if [ -n "$KERNEL" ]; then
    set -- "$@" -kernel "$KERNEL"
    [ -n "$INITRD" ] && set -- "$@" -initrd "$INITRD"
    set -- "$@" -append "$APPEND"
fi
if [ -n "$DRIVE" ]; then
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
# - PRoot ptrace overhead: PRoot intercepts every host syscall via ptrace.
#   Silencing kernel serial output (quiet loglevel=3) and avoiding unused
#   devices (VGA, SMM, floppy, unneeded NIC) cuts thousands of ptrace traps.
# - thread=single (Round-Robin) removes all vCPU<->IO lock contention and futex
#   storms; fastest for single-vCPU guests.
# - Big-core pinning on Android: phones use big.LITTLE / DynamIQ CPU topologies.
#   Run with 'taskset -c 4-7 sh run-x86-vm.sh ...' (or your SoC's big cores)
#   to prevent Android from scheduling QEMU on slow Cortex-A55 efficiency cores.
# - x86_64 Linux APPS (not a full OS) run far faster via linux-user, no VM:
#     build/qemu-x86_64 ./your-x86_64-binary
# - No KVM exists on Android: TCG only. Single vCPU is fastest for boot.
