#!/usr/bin/env python3
"""Parse QEMU `-d op` text for ONE TB into op records for op-run.

Usage: opparse.py <oplog> <tb-index> > tb.rec
  tb-index: 0-based index of the OP: block in the log.

Record format (one per line):
  slots <n>
  <op> <args...>            # op in {mov,add,sub,mul,neg,and,or,xor,shl,shr,
                           #          sar,rotr,extract,deposit,extrl,extu,
                           #          sextract,ld,st,movcond,brcond,br,label}
  args: r<slot>:<bits> | i<value> | L<labelid> | c<cond>
  ld/st: ld <bits> <dstslot> <envoff> | st <bits> <envoff> <src>
  brcond: brcond <bits> <cond> <a1> <a2> <labelid>

Exits nonzero if the TB contains an unsupported op (call, exit_tb,
goto_tb, goto_ptr, qemu_ld/st, ...): those are trace boundaries v1
does not cross.
"""
import re
import sys

PURE = {
    # name -> (kind, n_args kinds)
    "mov": ("alu", 2), "add": ("alu", 3), "sub": ("alu", 3),
    "mul": ("alu", 3), "neg": ("alu", 2), "and": ("alu", 3),
    "or": ("alu", 3), "xor": ("alu", 3), "shl": ("alu", 3),
    "shr": ("alu", 3), "sar": ("alu", 3), "rotr": ("alu", 3),
    "rotl": ("alu", 3), "not": ("alu", 2),
    "extract": ("ext", 4), "deposit": ("dep", 5),
    "extrl": ("cv", 2), "extu": ("cv", 2), "sextract": ("ext", 4),
    "movcond": ("mc", 6), "brcond": ("bc", 4), "br": ("br", 1),
    "set_label": ("lb", 1),
}

BOUNDARY = {"call", "exit_tb", "goto_tb", "goto_ptr", "qemu_ld", "qemu_st",
            "qemu_ld2", "qemu_st2", "exit_req", "lookup_and_goto_ptr"}


def parse_imm(tok):
    v = int(tok[1:], 16)
    if v >= 1 << 63:
        v -= 1 << 64
    return v


def main():
    log, want = sys.argv[1], int(sys.argv[2])
    data = open(log, "rb").read().replace(b"\x00", b"").decode("utf-8",
                                                               "replace")
    blocks = [b for b in data.split("OP:") if b.strip()]
    tb = blocks[want]
    slots = {}
    envmap = {}

    def slot(name, bits):
        # Temps are untyped 64-bit cells in practice (op widths govern
        # which bits are meaningful, as backends zero-extend i32 defs).
        # Keep first-seen width only for the slot map comment.
        if name not in slots:
            slots[name] = (len(slots), bits)
        return slots[name][0]

    def arg(tok, bits):
        if tok.startswith("$0x") or re.match(r"^\$0$", tok) or \
                re.match(r"^\$-?0x", tok):
            v = parse_imm(tok)
            return "i%d" % (v & ((1 << bits) - 1) if bits < 64 else v)
        if tok.startswith("$L"):
            return "L" + tok[2:]
        if tok.startswith("$"):
            v = int(tok[1:], 0)
            return "i%d" % v
        if tok == "env":
            return "env"
        return "r%d:%d" % (slot(tok, bits), bits)

    labels = {}
    out = []
    for line in tb.split("\n"):
        line = line.strip()
        if not line or line.startswith("----") or line.startswith("OP"):
            if line.startswith("OP") and ("after" in line or "before" in line):
                pass
            continue
        if line.startswith("IN:") or line.startswith("OUT:"):
            break
        m = re.match(r"([a-z0-9_]+)\s*(.*)", line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        toks = [t.strip() for t in rest.split(",") if t.strip() != ""]
        mm = re.match(r"([a-z]+?)(?:_(i32|i64))?$", name)
        base = mm.group(1) if mm else name
        if (base in BOUNDARY or name in BOUNDARY
                or re.match(r"^(call|exit_tb|goto_tb|goto_ptr|qemu_ld2?|"
                            r"qemu_st2?|exit_req|lookup_and_goto_ptr)\b",
                            name)):
            print("trace ends at: %s" % line.strip(), file=sys.stderr)
            break
        if base == "discard":
            continue
        if name in ("extrl_i64_i32", "extu_i32_i64"):
            # dst64 = zero-extend(low32(src64)); src stays 64-bit typed
            d = arg(toks[0], 64)
            s = arg(toks[1], 64)
            out.append(("zext", ["64", d, s]))
            continue
        m2 = re.match(r"^(ld|st)(\d*)_(i32|i64)$", name)
        if m2:
            is_st = m2.group(1) == "st"
            w = int(m2.group(2)) if m2.group(2) else (
                32 if m2.group(3) == "i32" else 64)
            if not is_st:
                assert toks[1] == "env", line
                off = parse_imm(toks[2])
                key = (off, w)
                if key not in envmap:
                    envmap[key] = len(slots)
                    slots["env@%d:%d" % key] = (envmap[key], 32)
                d = arg(toks[0], 64 if w == 64 else 32)
                out.append(("ld", [str(w), d, "s%d" % envmap[key]]))
            else:
                assert toks[1] == "env", line
                off = parse_imm(toks[2])
                key = (off, 32)
                if key not in envmap:
                    envmap[key] = len(slots)
                    slots["env@%d:%d" % key] = (envmap[key], 32)
                s = arg(toks[0], 32)
                out.append(("st", [str(w), "s%d" % envmap[key], s]))
            continue
        if base not in PURE:
            print("unsupported op: %s" % line, file=sys.stderr)
            return 2
        bits = 64 if not mm or mm.group(2) != "i32" else 32
        kind = PURE[base][0]
        if kind == "lb":
            lab = toks[0][2:] if toks[0].startswith("$L") else toks[0]
            labels[lab] = lab
            out.append(("label", [lab]))
        elif kind == "br":
            out.append(("br", [toks[0][2:]]))
        elif kind == "bc":
            a1 = arg(toks[0], bits)
            a2 = arg(toks[1], bits)
            out.append(("brcond",
                        [str(bits), toks[2], a1, a2, toks[3][2:]]))
        elif kind == "mc":
            c = [arg(t, bits) for t in toks[:5]]
            out.append(("movcond", [str(bits), toks[5]] + c))
        else:
            aa = [arg(t, bits) for t in toks]
            out.append((base, [str(bits)] + aa))

    print("slots %d" % len(slots))
    inv = {v[0]: (k, v[1]) for k, v in slots.items()}
    print("# " + " ".join("%d=%s" % (i, inv[i][0]) for i in sorted(inv)),
          file=sys.stderr)
    for op, aa in out:
        print(op + " " + " ".join(aa))


if __name__ == "__main__":
    sys.exit(main())
