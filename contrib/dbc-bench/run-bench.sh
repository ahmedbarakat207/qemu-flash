#!/bin/sh
# Benchmark TCG direct block chaining: chained (default) vs -d nochain,
# stock QEMU vs patched QEMU. Times whole-guest wall time (SeaBIOS boot
# overhead is constant across configs) and checks the guest checksum.
#
# Usage: ./run-bench.sh [patched-qemu] [stock-qemu] [runs]
set -u

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PATCHED=${1:-$REPO_ROOT/build/qemu-system-x86_64}
STOCK=${2:-/opt/homebrew/bin/qemu-system-x86_64}
RUNS=${3:-5}
KERNEL=$(dirname "$0")/kernel.elf

one_run() {
    # $1=qemu $2=extra-flags... ; prints "<wall_seconds> <sum>"
    qemu=$1; shift
    out=$(/usr/bin/time -p "$qemu" -M pc -m 128 -kernel "$KERNEL" \
        -display none -serial stdio \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot "$@" \
        2>&1 >/tmp/dbc-serial.log; echo "TIME:$?")
    # /usr/bin/time -p prints real/user/sys to stderr; guest serial went to file
    wall=$(printf '%s\n' "$out" | awk '/^real /{print $2}')
    # serial log holds the guest lines; time output went to $out mixed?
    echo "$wall"
}

run_cfg() {
    # $1=label $2=qemu $3...=extra qemu flags
    label=$1; qemu=$2; shift 2
    # warmup (discarded, fills host page cache / tb cache cold path)
    "$qemu" -M pc -m 128 -kernel "$KERNEL" -display none \
        -serial file:/tmp/dbc-warm.log \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot "$@" \
        >/dev/null 2>&1
    i=0; results=""
    sums=""
    while [ "$i" -lt "$RUNS" ]; do
        start=$(python3 -c 'import time; print(time.time())')
        "$qemu" -M pc -m 128 -kernel "$KERNEL" -display none \
            -serial file:/tmp/dbc-serial.log \
            -device isa-debug-exit,iobase=0xf4,iosize=0x04 -no-reboot "$@" \
            >/dev/null 2>/tmp/dbc-qemu-err.log
        end=$(python3 -c 'import time; print(time.time())')
        wall=$(python3 -c "print('%.3f' % ($end - $start))")
        sum=$(grep -a -o 'sum=0x[0-9a-f]*' /tmp/dbc-serial.log | tail -1)
        done_line=$(grep -a -c 'DBC-BENCH done' /tmp/dbc-serial.log)
        if [ "$done_line" -ne 1 ]; then
            echo "FAIL: $label run $i did not finish (see /tmp/dbc-qemu-err.log)"
            cat /tmp/dbc-qemu-err.log | head -5
            return 1
        fi
        results="$results $wall"
        sums="$sums $sum"
        i=$((i + 1))
    done
    median=$(python3 -c "import sys; v=sorted(float(x) for x in sys.argv[1:]); print('%.3f' % v[len(v)//2])" $results)
    echo "$label: runs:$results median=${median}s sums:$sums"
}

echo "kernel: $KERNEL"
run_cfg "stock-chained " "$STOCK"
run_cfg "stock-nochain " "$STOCK" -d nochain -D /tmp/dbc-nochain.log
run_cfg "patched-chained" "$PATCHED"
run_cfg "patched-nochain" "$PATCHED" -d nochain -D /tmp/dbc-nochain.log
