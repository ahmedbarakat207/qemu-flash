#!/usr/bin/env python3
"""Query `info jit` from a running QEMU via QMP and print it.

Usage: qmp_info_jit.py /tmp/qmp.sock
"""
import json
import socket
import sys


def qmp(sock_path, cmd):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    f = s.makefile("rwb")
    f.readline()  # greeting
    f.write(json.dumps({"execute": "qmp_capabilities"}).encode() + b"\n")
    f.flush()
    f.readline()  # ack
    f.write(json.dumps(cmd).encode() + b"\n")
    f.flush()
    resp = json.loads(f.readline().decode())
    # 'quit' closes the connection; tolerate EOF afterwards.
    try:
        s.close()
    except OSError:
        pass
    return resp


def main():
    sock = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dbc-qmp.sock"
    resp = qmp(sock, {"execute": "human-monitor-command",
                      "arguments": {"command-line": "info jit"}})
    print(resp.get("return", resp))
    qmp(sock, {"execute": "quit"})


if __name__ == "__main__":
    main()
