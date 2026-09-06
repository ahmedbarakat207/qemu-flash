#!/usr/bin/env python3
"""Independent interpreter for opparse.py records. Differential-tests op-run.

Usage: interp.py <trace.rec> [slot=value ...]  -> same `slot=value` output.
Semantics mirror TCG ops as modeled (i64-carried, defs masked to width,
shift counts masked, read-modify-write sub-width stores). Shares NO code
with op-run.cpp; agreement validates the emitter + LLVM pipeline.
"""
import sys

M64 = (1 << 64) - 1


def mask(v, b):
    return v & (M64 if b >= 64 else (1 << b) - 1)


def scmp(a, b):
    a = a if a < 1 << 63 else a - (1 << 64)
    b = b if b < 1 << 63 else b - (1 << 64)
    return (a > b) - (a < b)


def cond(c, a, b, bits=64):
    if bits < 64:
        m = (1 << bits) - 1
        a &= m
        b &= m
    if c == "eq":
        return a == b
    if c == "ne":
        return a != b
    r = scmp(a, b) if bits == 64 else scmp(a if a < 1 << (bits - 1)
                                          else a - (1 << bits),
                                          b if b < 1 << (bits - 1)
                                          else b - (1 << bits))
    if c == "lt":
        return r < 0
    if c == "le":
        return r <= 0
    if c == "gt":
        return r > 0
    if c == "ge":
        return r >= 0
    if c == "ult":
        return a < b
    if c == "ule":
        return a <= b
    if c == "ugt":
        return a > b
    if c == "uge":
        return a >= b
    if c == "tsteq":
        return (a & b) == 0
    if c == "tstne":
        return (a & b) != 0
    raise ValueError(c)


def parse_arg(t, state):
    if t[0] == "r" or t[0] == "s":
        s = int(t[1:].split(":")[0])
        return state.get(s, 0)
    return int(t[1:]) & M64


def main():
    lines = open(sys.argv[1]).read().split("\n")
    nslots = 0
    ops = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        t = ln.split()
        if t[0] == "slots":
            nslots = int(t[1])
            continue
        ops.append(t)
    state = {}
    for a in sys.argv[2:]:
        s, v = a.split("=")
        state[int(s)] = int(v) & M64

    # blocks
    blocks = [[]]
    labels = {}
    for op in ops:
        if op[0] == "label":
            blocks.append([])
            labels[op[1]] = len(blocks) - 1
        else:
            blocks[-1].append(op)

    def width_of_reg(tok):
        return int(tok.split(":")[1]) if ":" in tok else 64

    def dst_slot(tok):
        return int(tok[1:].split(":")[0])

    bi = 0
    while True:
        blk = blocks[bi]
        nxt = bi + 1 if bi + 1 < len(blocks) else None
        jumped = False
        for op in blk:
            n = op[0]
            if n in ("mov", "add", "sub", "mul", "and", "or", "xor"):
                b = int(op[1])
                d = dst_slot(op[2])
                vals = [parse_arg(t, state) for t in op[3:]]
                if n == "mov":
                    r = vals[0]
                elif n == "add":
                    r = vals[0] + vals[1]
                elif n == "sub":
                    r = vals[0] - vals[1]
                elif n == "mul":
                    r = vals[0] * vals[1]
                elif n == "and":
                    r = vals[0] & vals[1]
                elif n == "or":
                    r = vals[0] | vals[1]
                else:
                    r = vals[0] ^ vals[1]
                state[d] = mask(r, b)
            elif n in ("neg", "not"):
                b = int(op[1])
                d = dst_slot(op[2])
                v = parse_arg(op[3], state)
                state[d] = mask(-v if n == "neg" else ~v, b)
            elif n in ("shl", "shr", "sar", "rotl", "rotr"):
                b = int(op[1])
                d = dst_slot(op[2])
                a = parse_arg(op[3], state)
                c = parse_arg(op[4], state) & (b - 1)
                if n == "shl":
                    r = (a << c)
                elif n == "shr":
                    r = (a & (M64 if b >= 64 else (1 << b) - 1)) >> c
                elif n == "sar":
                    av = a if a < 1 << 63 else a - (1 << 64)
                    # arithmetic shift within width b of the low bits
                    lo = a & mask(a, b)
                    lo = lo if lo < 1 << (b - 1) else lo - (1 << b)
                    r = (lo >> c) & M64
                else:
                    lo = a & mask(a, b)
                    if n == "rotl":
                        r = ((lo << c) | (lo >> (b - c))) if c else lo
                    else:
                        r = ((lo >> c) | (lo << (b - c))) if c else lo
                state[d] = mask(r, b)
            elif n == "extract":
                d = dst_slot(op[2])
                a = parse_arg(op[3], state)
                off = parse_arg(op[4], state) & 63
                ln_ = parse_arg(op[5], state) & 127
                m = M64 if ln_ >= 64 else (1 << ln_) - 1
                state[d] = (a >> off) & m
            elif n == "sextract":
                d = dst_slot(op[2])
                a = parse_arg(op[3], state)
                off = parse_arg(op[4], state) & 63
                ln_ = parse_arg(op[5], state) & 127
                if ln_ >= 64:
                    r = a
                else:
                    lo = (a >> off) & ((1 << ln_) - 1)
                    r = lo - (1 << ln_) if lo >= 1 << (ln_ - 1) else lo
                state[d] = r & M64
            elif n == "deposit":
                b = int(op[1])
                d = dst_slot(op[2])
                base = parse_arg(op[3], state)
                val = parse_arg(op[4], state)
                off = parse_arg(op[5], state) & 63
                ln_ = parse_arg(op[6], state) & 127
                m = M64 if ln_ >= 64 else (((1 << ln_) - 1) << off)
                state[d] = mask((base & (m ^ M64)) | ((val << off) & m), b)
            elif n == "zext":
                d = dst_slot(op[2])
                state[d] = parse_arg(op[3], state) & 0xFFFFFFFF
            elif n == "ld":
                w = int(op[1])
                d = dst_slot(op[2])
                s = int(op[3][1:])
                state[d] = state.get(s, 0) & (M64 if w >= 64
                                              else (1 << w) - 1)
            elif n == "st":
                w = int(op[1])
                s = int(op[2][1:])
                v = parse_arg(op[3], state)
                m = M64 if w >= 64 else (1 << w) - 1
                state[s] = (state.get(s, 0) & (m ^ M64)) | (v & m)
            elif n == "movcond":
                b = int(op[1])
                d = dst_slot(op[3])
                c = parse_arg(op[4], state)
                c2 = parse_arg(op[5], state)
                v1 = parse_arg(op[6], state)
                v2 = parse_arg(op[7], state)
                state[d] = mask(v1 if cond(op[2], c, c2, b) else v2, b)
            elif n == "br":
                if op[1] in labels:
                    bi = labels[op[1]]
                else:
                    break
                jumped = True
                break
            elif n == "brcond":
                b = int(op[1])
                c = parse_arg(op[3], state)
                c2 = parse_arg(op[4], state)
                if cond(op[2], c, c2, b if op[0] == 'brcond' else 64):
                    if op[5] in labels:
                        bi = labels[op[5]]
                    else:
                        break
                    jumped = True
                    break
                # fallthrough
            else:
                raise ValueError("unhandled " + n)
        if not jumped:
            if nxt is None:
                break
            bi = nxt
    for s in range(nslots):
        print("%d=%d" % (s, state.get(s, 0) & M64))


if __name__ == "__main__":
    main()
