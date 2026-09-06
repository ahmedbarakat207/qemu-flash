#!/bin/sh
# Build this patched QEMU inside Android proot (proot-distro Ubuntu/Debian).
# Run INSIDE proot, from the repo root. Produces:
#   build/qemu-system-x86_64   (full x86_64 VM, TCG only - no KVM on Android)
#   build/qemu-x86_64          (x86_64 Linux apps directly - much faster than system)
#
# Usage: sh contrib/android-proot/build-proot.sh [--lto] [-j N]
#   --lto   enable LTO (faster binary, needs lots of RAM - skip on phones)
set -u

LTO=no
JOBS=$(nproc 2>/dev/null || echo 4)
for a in "$@"; do
    case $a in
        --lto) LTO=yes ;;
        -j*) JOBS=$(printf '%s' "$a" | sed 's/-j//') ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1; }

# --- distro deps -----------------------------------------------------------
if need apt-get; then
    echo "[deps] apt (Debian/Ubuntu)"
    apt-get update
    apt-get install -y git python3 python3-pip ninja-build pkg-config \
        build-essential libglib2.0-dev libpixman-1-dev zlib1g-dev \
        libffi-dev libslirp-dev
elif need dnf; then
    echo "[deps] dnf (Fedora)"
    dnf install -y git python3 python3-pip ninja-build pkg-config gcc \
        glib2-devel pixman-devel zlib-devel libffi-devel libslirp-devel
elif need pacman; then
    echo "[deps] pacman (Arch)"
    pacman -Sy --noconfirm git python python-pip ninja pkgconf base-devel \
        glib2 pixman zlib libffi libslirp
elif need apk; then
    echo "[deps] apk (Alpine)"
    apk add git python3 py3-pip ninja meson bash build-base pkgconfig \
        glib-dev pixman-dev zlib-dev libffi-dev slirp-dev
else
    echo "no supported package manager (need apt/dnf/pacman/apk)" >&2
    exit 1
fi

# --- meson >= 1.6 (repo requirement; distro ones are usually too old) ------
MESON_OK=no
if need meson; then
    if python3 -c "import sys
from mesonbuild import coredata
" 2>/dev/null; then :; fi
    MV=$(meson --version 2>/dev/null || echo 0)
    MAJ=$(printf '%s' "$MV" | cut -d. -f1); MIN=$(printf '%s' "$MV" | cut -d. -f2)
    if [ "${MAJ:-0}" -gt 1 ] || { [ "${MAJ:-0}" = 1 ] && [ "${MIN:-0}" -ge 6 ]; }; then
        MESON_OK=yes
    fi
fi
if [ "$MESON_OK" = no ]; then
    echo "[meson] system meson too old/missing, installing via pip"
    python3 -m pip install --upgrade --break-system-packages meson ninja 2>/dev/null || \
        python3 -m pip install --upgrade meson ninja
    export PATH="$HOME/.local/bin:$PATH"
fi
meson --version
ninja --version

# --- configure (minimal, headless, no -Werror) ------------------------------
CFG="./configure --target-list=x86_64-softmmu,x86_64-linux-user"
CFG="$CFG --disable-tools --disable-docs --disable-gtk --disable-sdl"
CFG="$CFG --disable-vte --disable-curses --disable-vnc --disable-werror"
if [ "$LTO" = yes ]; then
    CFG="$CFG --enable-lto"
fi
echo "[configure] $CFG"
# shellcheck disable=SC2086
$CFG || { echo "configure failed" >&2; exit 1; }

# --- build ------------------------------------------------------------------
echo "[build] jobs=$JOBS"
ninja -C build qemu-system-x86_64 qemu-x86_64 || exit 1
ls -la build/qemu-system-x86_64 build/qemu-x86_64
echo OK
