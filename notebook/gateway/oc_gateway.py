#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

"""oc_gateway.py -- the read-only notebook gateway (hecke-extensions slot 19, NB-3).

Stdlib-only Python 3 (>= 3.8; no pip dependency), outside the C++ build, runs
unchanged on x86-64 and aarch64 (Jetson Orin). It serves the notebook page
(notebook/page/) and the vendored KaTeX (notebook/vendor/katex/), and maps one
WebSocket per browser tab onto one `hacs --serve` child speaking
`oc-serve/1` (NDJSON on stdio, [P] v3.10 Sec. 18).

Normative sources: docs/console/00-console-interface-plan.md Sec. 18.9 - 18.12
(the gateway requirements) and docs/implementation/slot18/WP-19d_notebook_design_note.md
Sec. 0 (owner decisions 2026-10-03). The binary's own checks (env gate,
read_only dispatch, RSS watchdog) are authoritative; everything here is the
second, independent line.

Exposure (owner decision 2026-10-03; [P] Sec. 18.12):
  * `--bind` is a literal IP address (default 127.0.0.1). The gateway REFUSES TO
    START on a public bind address, and on the unspecified address (0.0.0.0 /
    ::) unless every address of the host is loopback / private / link-local
    (Linux: /proc/net/fib_trie + /proc/net/if_inet6; elsewhere unknown ->
    refused).
  * Peer filter at accept: loopback, RFC 1918 (10/8, 172.16/12, 192.168/16),
    169.254/16, fe80::/10, fc00::/7 (IPv4-mapped IPv6 peers are unwrapped).
    Everything else -- 100.64/10 (RFC 6598 shared/CGNAT space) included -- is
    dropped: that range is carrier or overlay-VPN space (e.g. Tailscale), i.e.
    by definition not "the local network".
  * A per-process random access token (256 bit, secrets.token_urlsafe(32)),
    printed once at startup; every WebSocket must present it (first frame
    `{"op":"auth","token":...}` or an `Authorization: Bearer` header on the
    upgrade), compared with hmac.compare_digest. Static files carry no secret
    and are served without it; nothing reaches a binary without it.
  * No TLS in v1: LAN traffic, token included, is cleartext.

Transport choice: a minimal RFC 6455 WebSocket (text frames, fragmentation on
receive, ping/pong, close; no extensions) rather than long-polling. A cell
streams `out` / `progress` / `value` frames and must be interruptible while it
runs; WebSocket gives one ordered, full-duplex channel per tab with a natural
lifetime (= the session's), whereas long-polling would need its own session
table, sequence acknowledgement and reconnection logic -- more code, not less.
The codec is ~150 lines and is exercised end to end by test_gateway.py.

Child process ([P] Sec. 18.9): environment built from the whitelist only
(PATH, HOME, LANG, LC_ALL, TMPDIR + ENGINE_THREADS, ENGINE_CUDA_DISABLE=1 and,
if configured, ENGINE_CUDA_DEVICE / ENGINE_HOST_THREADS); working directory a
fresh empty scratch directory owned by the gateway; own process group. Resource
guard: the binary gets `--rss-cap-mib N`; the gateway's watchdog SIGKILLs at
N + margin, at the per-cell wall and CPU limits, and closes idle sessions.
Rule 19 (predict first): a session is refused if MemAvailable < N.

Cost confirmation (NBCONFIRM S4, [P] Sec. 18.4 / 18.6): an `eval` may carry `confirm`, a list of at most 16
tokens (1-64 printable ASCII characters each), forwarded unchanged; the child alone decides what a token
confirms. The per-cell wall limit of a CONFIRMED run is max(--cell-wall-s, 3 * wall_hi + 60), at most
--cell-wall-max-s (default 21600 = 6 h; owner ruling 2026-11-06 Q4), where wall_hi is the child's own
estimate, taken ONLY from the child's own `progress` frame of a confirmed run (its `confirmed.wall_hi`), which
the child emits at the start of every confirmed run, before any of the call's work, for the request it is
working on (this also covers a lazy verb whose work starts in a later cell). Presenting a token raises nothing:
the child alone decides whether a token confirms a call of this cell, and only its `progress.confirmed` frame
says that it did (NBCONFIRM S4 audit M1). A value sent by the page is never used. The RSS cap is unchanged: a confirmed
cell lives under the session's --rss-cap-mib like any other, and nothing a page sends can raise it.

Atlas channel (slot-20 NB-4, [P] v3.11 Sec. 18.16 (f)): with `--atlas-store <store.json>`
a session's first `{"op":"atlas","id","query",...}` starts ONE `hacs --serve-atlas`
child for that session (protocol `oc-serve-atlas/1`; same inert whitelist environment, own
empty working directory and process group, RSS / wall limits); the query and its fields are
type-checked here, the store path is fixed by the command line, frames come back unmodified.

Run:  python3 oc_gateway.py --binary <path>/hacs [--bind 192.168.1.20]
      then open the printed URL (the token travels in the URL fragment, which
      the browser never sends to the server).
"""

import argparse
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATEWAY_PROTO = "oc-gateway/1"
SERVE_PROTO = "oc-serve/1"
HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------
ALLOWED_NETS = tuple(ipaddress.ip_network(n) for n in (
    "127.0.0.0/8",      # loopback
    "10.0.0.0/8",       # RFC 1918
    "172.16.0.0/12",    # RFC 1918
    "192.168.0.0/16",   # RFC 1918
    "169.254.0.0/16",   # IPv4 link-local
    "::1/128",          # loopback
    "fc00::/7",         # unique-local
    "fe80::/10",        # link-local
))


def parse_ip(text):
    """A literal IP address (zone id dropped; IPv4-mapped IPv6 unwrapped), else None."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    t = t.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(t)
    except ValueError:
        return None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip


def is_allowed_address(text):
    """True iff `text` is loopback, RFC 1918, link-local or unique-local."""
    ip = parse_ip(text)
    if ip is None:
        return False
    return any(ip.version == n.version and ip in n for n in ALLOWED_NETS)


def host_addresses():
    """Every address configured on this host, or None when it cannot be enumerated."""
    out = []
    try:
        with open("/proc/net/fib_trie") as f:
            prev = None
            for line in f:
                s = line.strip()
                if s.startswith("|--"):
                    prev = s[3:].strip()
                elif s.startswith("/32 host LOCAL") and prev:
                    out.append(prev)
    except OSError:
        return None
    try:
        with open("/proc/net/if_inet6") as f:
            for line in f:
                h = line.split()[0]
                out.append(str(ipaddress.ip_address(int(h, 16))) if len(h) == 32 else h)
    except OSError:
        pass  # no IPv6 on this host
    return out


def bind_verdict(bind, addrs=None):
    """(ok, reason) for a bind address. `addrs` overrides the host enumeration (tests)."""
    ip = parse_ip(bind)
    if ip is None:
        return False, "bind address %r is not a literal IP address" % bind
    if ip.is_unspecified:
        if addrs is None:
            addrs = host_addresses()
        if addrs is None:
            return False, "unspecified bind %s refused: the host's addresses cannot be enumerated" % bind
        public = [a for a in addrs if not is_allowed_address(a)]
        if public:
            return False, ("unspecified bind %s refused: the host has public address(es) %s; "
                           "bind a private interface address instead" % (bind, ", ".join(public)))
        return True, "unspecified bind, host has only private addresses"
    if not is_allowed_address(str(ip)):
        return False, "bind address %s is public: refused (the notebook never leaves the local network)" % bind
    return True, "private bind"


# --------------------------------------------------------------------------
# RFC 6455 WebSocket (minimal: text/binary, fragmentation on receive, ping/pong, close)
# --------------------------------------------------------------------------
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


def ws_accept_key(key):
    return base64.b64encode(hashlib.sha1((key + WS_GUID).encode("ascii")).digest()).decode("ascii")


class WSClosed(Exception):
    pass


class WebSocket:
    """One WebSocket endpoint over a connected socket. `client=True` masks outgoing frames."""

    def __init__(self, sock, rfile=None, client=False, max_message=9 * 1024 * 1024):
        self.sock = sock
        self.rfile = rfile if rfile is not None else sock.makefile("rb")
        self.client = client
        self.max_message = max_message
        self.lock = threading.Lock()
        self.closed = False

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.rfile.read(n - len(buf))
            if not chunk:
                raise WSClosed("eof")
            buf += chunk
        return buf

    def send(self, opcode, payload):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        n = len(payload)
        head = bytes([0x80 | opcode])
        mbit = 0x80 if self.client else 0
        if n < 126:
            head += bytes([mbit | n])
        elif n < 65536:
            head += bytes([mbit | 126]) + struct.pack("!H", n)
        else:
            head += bytes([mbit | 127]) + struct.pack("!Q", n)
        if self.client:
            mask = os.urandom(4)
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            head += mask
        with self.lock:
            if self.closed and opcode != OP_CLOSE:
                raise WSClosed("closed")
            self.sock.sendall(head + payload)

    def send_text(self, text):
        self.send(OP_TEXT, text)

    def close(self, code=1000, reason=""):
        try:
            self.send(OP_CLOSE, struct.pack("!H", code) + reason.encode("utf-8")[:120])
        except (OSError, WSClosed):
            pass
        self.closed = True

    def recv(self):
        """Next complete text message (str); None on close. Control frames are handled here."""
        parts, first_op = [], None
        total = 0
        while True:
            b0, b1 = self._read(2)
            fin, op = b0 & 0x80, b0 & 0x0F
            if b0 & 0x70:
                raise WSClosed("reserved bits set (no extension negotiated)")
            masked, n = b1 & 0x80, b1 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read(8))[0]
            if not self.client and not masked:
                raise WSClosed("unmasked client frame")
            if op >= 0x8 and (n > 125 or not fin):
                raise WSClosed("bad control frame")
            if n > self.max_message or total + n > self.max_message:
                self.close(1009, "message too big")
                raise WSClosed("message too big")
            mask = self._read(4) if masked else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == OP_PING:
                self.send(OP_PONG, data)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                if not self.closed:
                    self.close(1000)
                return None
            if op in (OP_TEXT, OP_BIN):
                if first_op is not None:
                    raise WSClosed("new data frame inside a fragmented message")
                first_op = op
            elif op == OP_CONT:
                if first_op is None:
                    raise WSClosed("continuation without a start")
            else:
                raise WSClosed("unknown opcode %d" % op)
            parts.append(data)
            total += n
            if fin:
                raw = b"".join(parts)
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    raise WSClosed("message is not UTF-8")


# --------------------------------------------------------------------------
# The read-only list (the JSON twin, [P] Sec. 18.8)
# --------------------------------------------------------------------------
class ReadOnlyList:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") != "oc_read_only_verbs/1":
            raise ValueError("%s: schema %r, expected oc_read_only_verbs/1" % (path, d.get("schema")))
        self.path = path
        self.read_only = int(d["read_only"])
        self.read_only_verbs = int(d["read_only_verbs"])
        served, all_verbs = set(), set()
        for e in d["entries"]:
            all_verbs.add(e["verb"])
            if e["read_only"]:
                served.add(e["verb"])
        if len(served) != self.read_only_verbs or sum(1 for e in d["entries"] if e["read_only"]) != self.read_only:
            raise ValueError("%s: its counts disagree with its entries" % path)
        self.served = served
        self.unserved = all_verbs - served   # verbs with NO read-only overload

    def matches_hello(self, hello):
        c = hello.get("counts", {})
        return c.get("verbs") == self.read_only_verbs and c.get("overloads") == self.read_only


IDENT0 = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
IDENT1 = IDENT0 | set("0123456789")


def first_unserved_call(src, names):
    """(name, line, col) of the first call `name(` of a name in `names` outside strings and
    `#` comments, not a member call (`.name(`); None if there is none. Lexical only: the
    binary's dispatch-time check stays authoritative ([P] Sec. 18.8)."""
    i, n, line, col = 0, len(src), 1, 1
    last_sig = ""
    while i < n:
        c = src[i]
        if c == "\n":
            i, line, col = i + 1, line + 1, 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
                col += 1
            continue
        if c == '"':
            i += 1
            col += 1
            while i < n and src[i] != '"':
                if src[i] == "\\" and i + 1 < n:
                    i += 1
                    col += 1
                if src[i] == "\n":
                    line, col = line + 1, 0
                i += 1
                col += 1
            i += 1
            col += 1
            last_sig = '"'
            continue
        if c in IDENT0:
            j = i
            while j < n and src[j] in IDENT1:
                j += 1
            word = src[i:j]
            k = j
            while k < n and src[k] in " \t":
                k += 1
            if word in names and last_sig != "." and k < n and src[k] == "(":
                return word, line, col
            col += j - i
            i = j
            last_sig = "a"
            continue
        if not c.isspace():
            last_sig = c
        i += 1
        col += 1
    return None


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------
ENV_WHITELIST_INERT = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
ID_CHARS = set(chr(c) for c in range(0x21, 0x7F))


def mem_available_mib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def proc_rss_mib(pid):
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def proc_cpu_s(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            s = f.read()
        fields = s[s.rindex(")") + 2:].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# The atlas channel (slot-20 NB-4; [P] v3.11 Sec. 18.16 (f))
# --------------------------------------------------------------------------
ATLAS_PROTO = "oc-serve-atlas/1"
# query -> {field: kind}; kind "int" (a non-negative int, never a bool), "filter".
ATLAS_QUERIES = {
    "info": {},
    "cells": {"offset": "int", "limit": "int"},
    "cell": {"n": "int"},
    "system": {"sid": "int"},
    "systems": {"filter": "filter", "offset": "int", "limit": "int"},
}
ATLAS_FILTER = {"r": "int", "type": "type", "collected": "bool", "base": "int", "twist": "int", "n": "int"}


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))


def _is_uint(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 2 ** 32


def atlas_forward(req):
    """(forwarded request dict, None) or (None, error message) for a page message
    {"op":"atlas","id","query",...}. Only the fields of ATLAS_QUERIES pass, with their JSON
    types checked; no field can carry a path (the store comes from --atlas-store only)."""
    q = req.get("query")
    if q not in ATLAS_QUERIES:
        return None, "gateway: atlas query %r is not offered (one of %s)" % (q, ", ".join(sorted(ATLAS_QUERIES)))
    spec = ATLAS_QUERIES[q]
    fwd = {"op": q, "id": req.get("id")}
    for k, v in req.items():
        if k in ("op", "id", "query"):
            continue
        kind = spec.get(k)
        if kind is None:
            return None, "gateway: atlas %s takes no field %r" % (q, k)
        if kind == "int":
            if not _is_uint(v):
                return None, "gateway: atlas %s.%s must be a non-negative integer" % (q, k)
        elif kind == "filter":
            if not isinstance(v, dict):
                return None, "gateway: atlas filter must be an object"
            for fk, fv in v.items():
                fkind = ATLAS_FILTER.get(fk)
                if fkind is None:
                    return None, "gateway: unknown atlas filter key %r" % (fk,)
                if fkind == "int" and not _is_uint(fv):
                    return None, "gateway: filter.%s must be a non-negative integer" % fk
                if fkind == "bool" and not isinstance(fv, bool):
                    return None, "gateway: filter.%s must be a Bool" % fk
                if fkind == "type" and fv not in ("eisenstein", "cuspidal"):
                    return None, "gateway: filter.type must be eisenstein or cuspidal"
        fwd[k] = v
    return fwd, None


class AtlasChild:
    """One `hacs --serve-atlas` child of a session, started lazily at the first atlas request.
    Same whitelist environment rule as the console child (inert variables only), its own empty working
    directory and process group; frames are relayed unmodified."""

    def __init__(self, sess):
        self.sess, self.cfg = sess, sess.cfg
        self.proc = None
        self.workdir = None
        self.inflight = None
        self.inflight_since = 0.0
        self.kill_reason = None
        self.stderr_tail = []
        self.lock = threading.Lock()

    def kill_cap(self):
        cap = self.cfg.atlas_rss_cap_mib
        return cap + max(64, cap // 10)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def env(self):
        env = {}
        for k in ENV_WHITELIST_INERT:
            if k in os.environ:
                env[k] = os.environ[k]
        env["TMPDIR"] = self.workdir
        return env

    def start(self):
        cap = self.cfg.atlas_rss_cap_mib
        avail = mem_available_mib()
        if avail is not None and avail < cap:
            return "MemAvailable %d MiB < atlas RSS cap %d MiB (rule 19: predict first)" % (avail, cap)
        self.workdir = tempfile.mkdtemp(prefix="oc-atlas-", dir=self.cfg.scratch)
        argv = [self.cfg.binary, "--serve-atlas", "--store", self.cfg.atlas_store, "--rss-cap-mib", str(cap)]
        if self.cfg.atlas_index:
            argv += ["--index", self.cfg.atlas_index]
        if self.cfg.atlas_expect_fnv:
            argv += ["--expect-fnv", self.cfg.atlas_expect_fnv]
        self.kill_reason = None
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         cwd=self.workdir, env=self.env(), start_new_session=True, close_fds=True)
        except OSError as e:
            self._cleanup_dir()
            return "spawn: %s" % e
        threading.Thread(target=self._stderr_pump, args=(self.proc,), daemon=True).start()
        line = self.sess._read_line(self.proc, timeout=self.cfg.atlas_hello_timeout_s)
        try:
            hello = json.loads(line) if line else None
        except ValueError:
            hello = None
        if not hello or hello.get("kind") != "atlas_hello" or hello.get("proto") != ATLAS_PROTO:
            self.stop()
            return "the binary did not say an %s hello: %s" % (ATLAS_PROTO, " | ".join(self.stderr_tail[-3:]))
        self.sess.ws.send_text(line)   # the hello, unmodified
        threading.Thread(target=self._stdout_pump, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._watchdog, args=(self.proc,), daemon=True).start()
        return None

    def _stderr_pump(self, proc):
        for raw in iter(proc.stderr.readline, b""):
            s = raw.decode("utf-8", "replace").rstrip("\n")
            self.stderr_tail.append(s)
            del self.stderr_tail[:-50]
            if self.cfg.verbose:
                sys.stderr.write("[atlas %d] %s\n" % (proc.pid, s))

    def _stdout_pump(self, proc):
        cut, fatal = False, None
        limit = self.cfg.max_frame_bytes
        while True:
            raw = proc.stdout.readline(limit + 2)
            if not raw:
                break
            if not raw.endswith(b"\n"):
                cut = True
                break
            text = raw[:-1].decode("utf-8", "replace")
            try:
                fr = json.loads(text)
            except ValueError:
                cut = True
                break
            if fr.get("kind") == "atlas_reply":
                with self.lock:
                    if fr.get("id") == self.inflight:
                        self.inflight = None
            elif fr.get("kind") == "atlas_fatal":
                fatal = fr.get("reason")
            try:
                self.sess.ws.send_text(text)   # relayed unmodified
            except (OSError, WSClosed):
                break
        code = proc.wait()
        with self.lock:
            rid, self.inflight = self.inflight, None
        if not self.sess.closed:
            reason = self.kill_reason or ("fatal_" + fatal if fatal else ("stream_cut" if cut else "exited"))
            self.sess.gw_event("atlas_exited", code=code, reason=reason, id=rid, stderr=self.stderr_tail[-5:])
        self._cleanup_dir()

    def _watchdog(self, proc):
        while proc.poll() is None and not self.sess.closed:
            time.sleep(0.25)
            rss = proc_rss_mib(proc.pid)
            if rss is not None and rss > self.kill_cap():
                self._kill("rss_cap", "resident %d MiB > gateway atlas cap %d MiB" % (rss, self.kill_cap()))
                return
            with self.lock:
                inflight, since = self.inflight, self.inflight_since
            if inflight is not None and self.cfg.atlas_wall_s > 0 and time.monotonic() - since > self.cfg.atlas_wall_s:
                self._kill("atlas_wall", "atlas request wall time > %g s" % self.cfg.atlas_wall_s)
                return

    def _kill(self, reason, detail=""):
        p = self.proc
        if p is None or p.poll() is not None:
            return
        self.kill_reason = reason
        if detail:
            self.sess.gw_event("killing", reason=reason, detail=detail, channel="atlas")
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            try:
                p.kill()
            except OSError:
                pass

    def _cleanup_dir(self):
        d, self.workdir = self.workdir, None
        if d and os.path.isdir(d) and os.path.dirname(os.path.abspath(d)) == os.path.abspath(self.cfg.scratch):
            shutil.rmtree(d, ignore_errors=True)

    def stop(self):
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.stdin.write(b'{"op":"close"}\n')
                p.stdin.flush()
                p.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            if p.poll() is None:
                self.kill_reason = self.kill_reason or "closed"
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self._cleanup_dir()

    def request(self, rid, fwd):
        """Forward one checked request; returns an error (kind, message) or None."""
        if not self.alive():
            why = self.start()
            if why is not None:
                return "atlas_unavailable", "gateway: the atlas server did not start: " + why
        with self.lock:
            if self.inflight is not None:
                return "busy", "gateway: atlas request %r is still running" % self.inflight
            self.inflight = rid
            self.inflight_since = time.monotonic()
        try:
            self.proc.stdin.write((json.dumps(fwd, separators=(",", ":")) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (OSError, ValueError):
            with self.lock:
                self.inflight = None
            return "atlas_unavailable", "gateway: the atlas server went away"
        return None


class Session:
    """One browser tab <-> one `hacs --serve` child (+ at most one `--serve-atlas` child)."""

    def __init__(self, gw, ws, tab):
        self.gw, self.ws, self.tab = gw, ws, tab
        self.cfg = gw.cfg
        self.atlas = None             # AtlasChild, started at the first atlas request
        self.proc = None
        self.workdir = None
        self.lock = threading.Lock()
        self.inflight = None          # id of the request the child is working on
        self.inflight_since = 0.0
        self.inflight_cpu0 = 0.0
        self.inflight_wall_s = self.cfg.cell_wall_s   # (NBCONFIRM S4) this request's wall limit
        self.last_activity = time.monotonic()
        self.kill_reason = None
        self.closed = False
        self.generation = 0
        self.stderr_tail = []

    # ---- helpers ---------------------------------------------------------
    def gw_event(self, event, **fields):
        msg = {"kind": "gw", "event": event}
        msg.update(fields)
        try:
            self.ws.send_text(json.dumps(msg, separators=(",", ":"), ensure_ascii=False))
        except (OSError, WSClosed):
            pass

    def child_env(self):
        env = {}
        for k in ENV_WHITELIST_INERT:
            if k in os.environ:
                env[k] = os.environ[k]
        env["TMPDIR"] = self.workdir
        env["ENGINE_THREADS"] = str(self.cfg.threads)
        env["ENGINE_CUDA_DISABLE"] = "1"
        if self.cfg.cuda_device is not None:
            env["ENGINE_CUDA_DEVICE"] = str(self.cfg.cuda_device)
        if self.cfg.host_threads is not None:
            env["ENGINE_HOST_THREADS"] = str(self.cfg.host_threads)
        return env

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        """Spawn a child, read and check its hello, relay it. Returns True on success."""
        cap = self.cfg.rss_cap_mib
        avail = mem_available_mib()
        if avail is not None and avail < cap:
            self.gw_event("refused", reason="memory",
                          detail="MemAvailable %d MiB < session RSS cap %d MiB (rule 19: predict first)" % (avail, cap))
            return False
        self.workdir = tempfile.mkdtemp(prefix="oc-session-", dir=self.cfg.scratch)
        self.kill_reason = None
        self.inflight = None
        self.generation += 1
        argv = [self.cfg.binary, "--serve", "--rss-cap-mib", str(cap)]
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, cwd=self.workdir, env=self.child_env(),
                                         start_new_session=True, close_fds=True)
        except OSError as e:
            self.gw_event("refused", reason="spawn", detail=str(e))
            self._cleanup_dir()
            return False
        threading.Thread(target=self._stderr_pump, args=(self.proc,), daemon=True).start()
        line = self._read_line(self.proc, timeout=self.cfg.hello_timeout_s)
        hello = None
        try:
            hello = json.loads(line) if line else None
        except ValueError:
            hello = None
        if not hello or hello.get("kind") != "hello" or hello.get("proto") != SERVE_PROTO:
            self._kill("bad_hello")
            self.gw_event("refused", reason="hello", detail="the binary did not say an oc-serve/1 hello",
                          stderr=self.stderr_tail[-5:])
            return False
        if not self.gw.ro.matches_hello(hello):
            self._kill("read_only_mismatch")
            self.gw_event("refused", reason="read_only_mismatch",
                          detail="hello.counts %s != %s (read_only=%d, read_only_verbs=%d)" % (
                              hello.get("counts"), self.gw.ro.path, self.gw.ro.read_only, self.gw.ro.read_only_verbs))
            return False
        self.pid = self.proc.pid
        self.ws.send_text(line)   # the hello, unmodified
        self.gw_event("session", generation=self.generation,
                      limits={"rss_cap_mib": cap, "kill_rss_mib": self.kill_cap(),
                              "cell_wall_s": self.cfg.cell_wall_s, "cell_wall_max_s": self.cfg.cell_wall_max_s,
                              "cell_cpu_s": self.cfg.cell_cpu_s,
                              "idle_timeout_s": self.cfg.idle_timeout_s},
                      unserved=sorted(self.gw.ro.unserved),
                      atlas={"configured": bool(self.cfg.atlas_store),
                             "store": os.path.basename(self.cfg.atlas_store) if self.cfg.atlas_store else None})
        threading.Thread(target=self._stdout_pump, args=(self.proc, self.generation), daemon=True).start()
        threading.Thread(target=self._watchdog, args=(self.proc, self.generation), daemon=True).start()
        self.last_activity = time.monotonic()
        return True

    def kill_cap(self):
        cap = self.cfg.rss_cap_mib
        return cap + max(64, cap // 10)

    def confirmed_wall(self, wall_hi):
        """(NBCONFIRM S4) the wall limit of a confirmed run: max(cell_wall_s, 3 * wall_hi + 60), at most cell_wall_max_s."""
        base = self.cfg.cell_wall_s
        if base <= 0:
            return base               # no wall limit at all
        if self.cfg.cell_wall_max_s <= 0 or not _is_num(wall_hi) or wall_hi <= 0:
            return base
        return max(base, min(3.0 * wall_hi + 60.0, self.cfg.cell_wall_max_s))

    def _raise_wall(self, rid, wall_hi, source):
        """With self.lock held: raise the in-flight request's wall limit (never lowers it)."""
        new = self.confirmed_wall(wall_hi)
        if self.inflight == rid and new > self.inflight_wall_s:
            self.inflight_wall_s = new
            return new
        return None

    def _read_line(self, proc, timeout):
        box = {}

        def rd():
            box["l"] = proc.stdout.readline()
        t = threading.Thread(target=rd, daemon=True)
        t.start()
        t.join(timeout)
        raw = box.get("l")
        if not raw or not raw.endswith(b"\n"):
            return None
        return raw[:-1].decode("utf-8", "replace")

    def _stderr_pump(self, proc):
        for raw in iter(proc.stderr.readline, b""):
            s = raw.decode("utf-8", "replace").rstrip("\n")
            self.stderr_tail.append(s)
            del self.stderr_tail[:-50]
            if self.cfg.verbose:
                sys.stderr.write("[child %d] %s\n" % (proc.pid, s))

    def _stdout_pump(self, proc, gen):
        cut = False
        fatal = None
        limit = self.cfg.max_frame_bytes
        while True:
            raw = proc.stdout.readline(limit + 2)
            if not raw:
                break
            if not raw.endswith(b"\n"):
                cut = True          # a stream cut mid-line (or an oversize frame): treated as killed
                break
            text = raw[:-1].decode("utf-8", "replace")
            try:
                fr = json.loads(text)
            except ValueError:
                fr = None
            if fr is None:
                cut = True
                break
            kind = fr.get("kind")
            raised = None
            if kind == "done":
                with self.lock:
                    if self.inflight is not None and fr.get("id") == self.inflight:
                        self.inflight = None
                        self.last_activity = time.monotonic()
            elif kind == "progress" and isinstance(fr.get("confirmed"), dict):
                # the ONLY source of a wall raise: the child's own frame that it is running a confirmed call
                # of the in-flight request (a presented token raises nothing; NBCONFIRM S4 audit M1)
                with self.lock:
                    raised = self._raise_wall(fr.get("id"), fr["confirmed"].get("wall_hi"), "progress")
            elif kind == "fatal":
                fatal = fr.get("reason")
            try:
                self.ws.send_text(text)   # relayed unmodified
            except (OSError, WSClosed):
                break
            if raised is not None:
                self.gw_event("cell_wall", id=fr.get("id"), cell_wall_s=raised, reason="confirmed")
        code = proc.wait()
        if gen != self.generation or self.closed:
            return
        with self.lock:
            self.inflight = None
        reason = self.kill_reason or ("fatal_" + fatal if fatal else ("stream_cut" if cut else "exited"))
        self.gw_event("exited", code=code, reason=reason, stderr=self.stderr_tail[-5:])
        self._cleanup_dir()

    def _watchdog(self, proc, gen):
        cfg = self.cfg
        while proc.poll() is None and gen == self.generation and not self.closed:
            time.sleep(0.25)
            rss = proc_rss_mib(proc.pid)
            if rss is not None and rss > self.kill_cap():
                self._kill("rss_cap", "resident %d MiB > gateway cap %d MiB" % (rss, self.kill_cap()))
                return
            now = time.monotonic()
            with self.lock:
                inflight, since, cpu0, wall = self.inflight, self.inflight_since, self.inflight_cpu0, self.inflight_wall_s
            if inflight is not None:
                if wall > 0 and now - since > wall:
                    self._kill("cell_wall", "cell wall time > %g s" % wall)
                    return
                cpu = proc_cpu_s(proc.pid)
                if cfg.cell_cpu_s > 0 and cpu is not None and cpu - cpu0 > cfg.cell_cpu_s:
                    self._kill("cell_cpu", "cell CPU time > %g s" % cfg.cell_cpu_s)
                    return
            elif cfg.idle_timeout_s > 0 and now - self.last_activity > cfg.idle_timeout_s:
                self._kill("idle", "idle > %g s" % cfg.idle_timeout_s)
                return

    def _kill(self, reason, detail=""):
        p = self.proc
        if p is None or p.poll() is not None:
            return
        self.kill_reason = reason
        if detail:
            self.gw_event("killing", reason=reason, detail=detail)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                p.kill()
            except OSError:
                pass

    def _cleanup_dir(self):
        d, self.workdir = self.workdir, None
        if d and os.path.isdir(d) and os.path.dirname(os.path.abspath(d)) == os.path.abspath(self.cfg.scratch):
            shutil.rmtree(d, ignore_errors=True)

    def stop(self, graceful=True):
        p = self.proc
        if p is not None and p.poll() is None:
            if graceful:
                try:
                    p.stdin.write(b'{"op":"close"}\n')
                    p.stdin.flush()
                    p.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    pass
            if p.poll() is None:
                self.kill_reason = self.kill_reason or "closed"
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    def close(self):
        self.closed = True
        self.stop()
        self._cleanup_dir()
        if self.atlas is not None:
            self.atlas.stop()

    def restart(self):
        self.generation += 1     # silences the old pumps' exit event
        self.stop(graceful=False)
        self._cleanup_dir()
        return self.start()

    # ---- requests from the page -----------------------------------------
    def reject(self, rid, kind, message, at=None):
        self.gw_event("rejected", id=rid, error={"kind": kind, "message": message, "at": at})

    def handle(self, text):
        self.last_activity = time.monotonic()
        try:
            req = json.loads(text)
        except ValueError:
            return self.reject(None, "protocol", "gateway: the message is not JSON")
        if not isinstance(req, dict):
            return self.reject(None, "protocol", "gateway: the message is not a JSON object")
        op = req.get("op")
        rid = req.get("id")
        if op == "restart":
            if not self.restart():
                self.gw_event("exited", code=None, reason="restart_failed")
            return
        if op == "interrupt":
            return self._to_child({"op": "interrupt"})
        if op == "atlas":
            return self.handle_atlas(req, rid)
        if op not in ("eval", "complete", "help", "describe"):
            return self.reject(rid if isinstance(rid, str) else None, "protocol",
                               "gateway: op %r is not offered by the notebook" % (op,))
        if not (isinstance(rid, str) and 1 <= len(rid) <= 64 and all(ch in ID_CHARS for ch in rid)):
            return self.reject(None, "protocol", "gateway: id must be 1-64 printable ASCII characters")
        fwd = {"op": op, "id": rid}
        if op == "eval":
            src = req.get("src")
            if not isinstance(src, str) or len(src.encode("utf-8")) > 8 * 1024 * 1024:
                return self.reject(rid, "protocol", "gateway: src must be a string of at most 8 MiB")
            hit = first_unserved_call(src, self.gw.ro.unserved)
            if hit:
                name, line, col = hit
                return self.reject(rid, "forbidden",
                                   "<cell>:%d:%d: gateway: verb '%s' has no read-only overload (%s)" % (
                                       line, col, name, os.path.basename(self.gw.ro.path)),
                                   at={"line": line, "col": col})
            fwd["src"] = src
            if "args" in req:
                if not isinstance(req["args"], dict):
                    return self.reject(rid, "protocol", "gateway: args must be an object")
                fwd["args"] = req["args"]
            if "confirm" in req:   # (NBCONFIRM S4) forwarded unchanged; the child decides what a token confirms
                conf = req["confirm"]
                if not (isinstance(conf, list) and len(conf) <= 16 and
                        all(isinstance(t, str) and 1 <= len(t) <= 64 and all(ch in ID_CHARS for ch in t) for t in conf)):
                    return self.reject(rid, "protocol",
                                       "gateway: confirm must be a list of at most 16 tokens of 1-64 printable ASCII characters")
                fwd["confirm"] = conf
        elif op == "complete":
            if not isinstance(req.get("prefix"), str):
                return self.reject(rid, "protocol", "gateway: prefix must be a string")
            fwd["prefix"] = req["prefix"]
        elif op == "help":
            if not isinstance(req.get("topic"), str):
                return self.reject(rid, "protocol", "gateway: topic must be a string")
            fwd["topic"] = req["topic"]
        elif op == "describe":
            if not isinstance(req.get("name"), str):
                return self.reject(rid, "protocol", "gateway: name must be a string")
            fwd["name"] = req["name"]
        with self.lock:
            if self.inflight is not None:
                return self.reject(rid, "busy", "gateway: request %r is still running" % self.inflight)
            if self.proc is None or self.proc.poll() is not None:
                return self.reject(rid, "no_session", "gateway: no running session (restart it)")
            self.inflight = rid
            self.inflight_since = time.monotonic()
            self.inflight_cpu0 = proc_cpu_s(self.proc.pid) or 0.0
            self.inflight_wall_s = self.cfg.cell_wall_s   # raised only by the child's progress.confirmed frame
        self._to_child(fwd)

    def handle_atlas(self, req, rid):
        """[P] v3.11 Sec. 18.16 (f): one checked atlas query to the session's --serve-atlas child."""
        if not (isinstance(rid, str) and 1 <= len(rid) <= 64 and all(ch in ID_CHARS for ch in rid)):
            return self.reject(None, "protocol", "gateway: id must be 1-64 printable ASCII characters")
        if not self.cfg.atlas_store:
            return self.reject(rid, "no_atlas", "gateway: no atlas store is configured (--atlas-store)")
        fwd, why = atlas_forward(req)
        if fwd is None:
            return self.reject(rid, "protocol", why)
        if self.atlas is None:
            self.atlas = AtlasChild(self)
        err = self.atlas.request(rid, fwd)
        if err is not None:
            self.reject(rid, err[0], err[1])

    def _to_child(self, obj):
        p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            p.stdin.write((json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))
            p.stdin.flush()
        except (OSError, ValueError):
            pass


# --------------------------------------------------------------------------
# HTTP + WebSocket server
# --------------------------------------------------------------------------
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".woff2": "font/woff2", ".json": "application/json",
    ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml",
}


class Gateway:
    def __init__(self, cfg, ro):
        self.cfg, self.ro = cfg, ro
        self.token = secrets.token_urlsafe(32)
        self.sessions = {}            # tab -> Session
        self.lock = threading.Lock()

    def token_ok(self, presented):
        if not isinstance(presented, str):
            return False
        return hmac.compare_digest(presented.encode("utf-8"), self.token.encode("utf-8"))

    # (slot-24 NB-FILE) The example notebooks: ONE fixed directory, listed flat, read-only. The URL names an entry of
    # that listing and is never joined to a path: a request is served only when its name equals a listed file name
    # (regular file, not a symlink, <= EXAMPLE_MAX_BYTES, name matching EXAMPLE_NAME), so no `..`, no separator, no
    # absolute path and no other extension can select a file. No directory structure is exposed.
    def example_names(self):
        d = getattr(self.cfg, "examples_dir", None)
        if not d or not os.path.isdir(d):
            return []
        out = []
        for n in sorted(os.listdir(d)):
            f = os.path.join(d, n)
            if (EXAMPLE_NAME.fullmatch(n) and os.path.isfile(f) and not os.path.islink(f)
                    and os.path.getsize(f) <= EXAMPLE_MAX_BYTES):
                out.append(n)
        return out

    def resolve_example(self, path):
        """/examples/<name>.oc -> the file's path (text/plain), or None. /examples/index.json is handled by the caller."""
        if not path.startswith("/examples/"):
            return None
        name = path[len("/examples/"):]
        if name not in self.example_names():      # membership in the listing, not path arithmetic
            return None
        return os.path.join(self.cfg.examples_dir, name)

    def resolve_static(self, path):
        """Map a URL path to a file under the page or KaTeX roots; None if not served."""
        if path in ("/", "/index.html"):
            root, rel = self.cfg.page_dir, "index.html"
        elif path.startswith("/static/"):
            root, rel = self.cfg.page_dir, path[len("/static/"):]
        elif path.startswith("/katex/"):
            root, rel = self.cfg.katex_dir, path[len("/katex/"):]
        else:
            return None
        if not rel or "\\" in rel or "\0" in rel or any(p in ("", ".", "..") for p in rel.split("/")):
            return None
        full = os.path.realpath(os.path.join(root, rel))
        if not full.startswith(os.path.realpath(root) + os.sep) or not os.path.isfile(full):
            return None
        if os.path.splitext(full)[1] not in CONTENT_TYPES:
            return None
        return full


EXAMPLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\.oc")
EXAMPLE_MAX_BYTES = 262144


def make_handler(gw):
    class Handler(BaseHTTPRequestHandler):
        server_version = "oc-gateway"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError):
                pass   # the peer went away mid-response: nothing to report

        def log_message(self, fmt, *args):
            if gw.cfg.verbose:
                sys.stderr.write("[gateway] %s %s\n" % (self.client_address[0], fmt % args))

        def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            # Defence in depth: the server already filtered at accept.
            if not is_allowed_address(self.client_address[0]):
                self.close_connection = True
                return
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path == "/ws":
                return self._websocket()
            if path == "/examples/index.json":
                names = gw.example_names()
                if not names:
                    return self._send(404, b"not found\n")
                return self._send(200, (json.dumps({"schema": "oc-notebook-examples/1", "files": names}) + "\n").encode(),
                                  "application/json", {"Content-Security-Policy": "default-src 'none'"})
            ex = gw.resolve_example(path)
            if ex is not None:
                with open(ex, "rb") as f:
                    body = f.read()
                return self._send(200, body, "text/plain; charset=utf-8", {"Content-Security-Policy": "default-src 'none'"})
            full = gw.resolve_static(path)
            if full is None:
                return self._send(404, b"not found\n")
            with open(full, "rb") as f:
                body = f.read()
            csp = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                   "font-src 'self'; img-src 'self' data:; connect-src 'self' ws://%s; "
                   "base-uri 'none'; form-action 'none'; frame-ancestors 'none'" % (self.headers.get("Host") or ""))
            self._send(200, body, CONTENT_TYPES[os.path.splitext(full)[1]], {"Content-Security-Policy": csp})

        def _websocket(self):
            h = self.headers
            if (h.get("Upgrade", "").lower() != "websocket" or "upgrade" not in h.get("Connection", "").lower()
                    or h.get("Sec-WebSocket-Version") != "13" or not h.get("Sec-WebSocket-Key")):
                return self._send(400, b"websocket upgrade expected\n")
            origin = h.get("Origin")
            if origin is not None:
                host = h.get("Host", "")
                o = origin.split("://", 1)[-1]
                if o != host:
                    return self._send(403, b"cross-origin websocket refused\n")
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", ws_accept_key(h["Sec-WebSocket-Key"]))
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True
            ws = WebSocket(self.connection, rfile=self.rfile)
            auth = h.get("Authorization", "")
            pre_auth = auth.startswith("Bearer ") and gw.token_ok(auth[7:].strip())
            try:
                self.connection.settimeout(gw.cfg.auth_timeout_s)
                first = ws.recv()
                if first is None:
                    return
                try:
                    msg = json.loads(first)
                except ValueError:
                    msg = None
                if not isinstance(msg, dict) or msg.get("op") != "auth":
                    ws.send_text('{"kind":"gw","event":"unauthorized","detail":"the first message must be {\\"op\\":\\"auth\\"}"}')
                    ws.close(1008, "unauthorized")
                    return
                if not (pre_auth or gw.token_ok(msg.get("token"))):
                    ws.send_text('{"kind":"gw","event":"unauthorized","detail":"missing or wrong token"}')
                    ws.close(1008, "unauthorized")
                    sys.stderr.write("[gateway] %s: websocket with a missing/wrong token refused\n" % self.client_address[0])
                    return
                self.connection.settimeout(None)
                tab = msg.get("tab")
                if not (isinstance(tab, str) and 1 <= len(tab) <= 64 and all(ch in ID_CHARS for ch in tab)):
                    tab = secrets.token_hex(8)
                sess = Session(gw, ws, tab)
                with gw.lock:
                    old = gw.sessions.get(tab)
                    others = [s for t, s in gw.sessions.items() if t != tab]
                    if len(others) >= gw.cfg.max_sessions:
                        ws.send_text('{"kind":"gw","event":"refused","reason":"max_sessions"}')
                        ws.close(1013, "too many sessions")
                        return
                    gw.sessions[tab] = sess
                if old is not None:      # one session per tab: a reconnecting tab replaces its old session
                    old.gw_event("replaced")
                    old.close()
                    old.ws.close(1000, "replaced")
                ws.send_text(json.dumps({"kind": "gw", "event": "authorized", "gateway": GATEWAY_PROTO, "tab": tab}))
                try:
                    if sess.start():
                        while True:
                            text = ws.recv()
                            if text is None:
                                break
                            sess.handle(text)
                finally:
                    sess.close()
                    with gw.lock:
                        if gw.sessions.get(tab) is sess:
                            del gw.sessions[tab]
                    ws.close(1000)
            except (WSClosed, OSError, socket.timeout):
                ws.closed = True

    return Handler


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, family):
        self.address_family = family
        super().__init__(addr, handler)

    def server_bind(self):
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        super().server_bind()

    def verify_request(self, request, client_address):
        ok = is_allowed_address(client_address[0])
        if not ok:
            sys.stderr.write("[gateway] dropped connection from non-private address %s\n" % client_address[0])
        return ok


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def default_path(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def build_parser():
    ap = argparse.ArgumentParser(description="Read-only notebook gateway for hacs --serve (oc-serve/1).")
    ap.add_argument("--binary", default=default_path(os.path.join(HERE, "..", "..", "build", "hacs"),
                                         os.path.join(HERE, "..", "..", "build", "cohomology_app")),
                    help="hacs built with -DOC_BUILD_NOTEBOOK=ON")
    ap.add_argument("--bind", default="127.0.0.1", help="literal IP to bind (default 127.0.0.1); public addresses are refused")
    ap.add_argument("--port", type=int, default=8765, help="TCP port (0 = any free port)")
    ap.add_argument("--verbs-json", default=default_path(
        os.path.join(HERE, "..", "..", "docs", "latex", "cli_manual", "read_only_verbs.json"),
        os.path.join(HERE, "..", "read_only_verbs.json")), help="read_only_verbs.json (the JSON twin)")
    ap.add_argument("--page-dir", default=os.path.join(HERE, "..", "page"))
    ap.add_argument("--examples-dir", default=default_path(os.path.join(HERE, "..", "examples", "notebook"),
                                                           os.path.join(HERE, "..", "examples")),
                    help="a flat directory of example notebooks (*.oc) offered read-only under /examples/ (default: the tour)")
    ap.add_argument("--katex-dir", default=default_path(os.path.join(HERE, "..", "vendor", "katex"),
                                                        os.path.join(HERE, "..", "katex")))
    ap.add_argument("--scratch", default=os.path.join(os.path.expanduser("~"), ".cache", "oc-notebook"),
                    help="parent of the per-session empty working directories")
    ap.add_argument("--rss-cap-mib", type=int, default=4096, help="per-session RSS cap passed to the binary")
    ap.add_argument("--cell-wall-s", type=float, default=1800.0, help="per-request wall limit (0 = off)")
    ap.add_argument("--cell-wall-max-s", type=float, default=21600.0,
                    help="(NBCONFIRM S4) the largest wall limit of a confirmed cell, max(--cell-wall-s, 3*wall_hi+60) capped "
                         "here (0 = a confirmed cell keeps --cell-wall-s)")
    ap.add_argument("--cell-cpu-s", type=float, default=0.0, help="per-request CPU-time limit (0 = off)")
    ap.add_argument("--idle-timeout-s", type=float, default=3600.0, help="close an idle session (0 = off)")
    ap.add_argument("--max-sessions", type=int, default=4)
    ap.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)), help="ENGINE_THREADS of a child")
    ap.add_argument("--cuda-device", default=None, help="ENGINE_CUDA_DEVICE (the device stays disabled; [P] 18.9)")
    ap.add_argument("--host-threads", default=None, help="ENGINE_HOST_THREADS of a child")
    ap.add_argument("--hello-timeout-s", type=float, default=30.0)
    ap.add_argument("--auth-timeout-s", type=float, default=15.0)
    ap.add_argument("--max-frame-bytes", type=int, default=1048576 + 4096)
    ap.add_argument("--verbose", action="store_true", help="log requests and child stderr")
    # (slot-20 NB-4, [P] v3.11 Sec. 18.16 (f)) the atlas channel; the store path is fixed here, never by a page.
    ap.add_argument("--atlas-store", default=None, help="a hecke_atlas/v1 store served read-only by --serve-atlas")
    ap.add_argument("--atlas-index", default=None, help="its oc-atlas-index/1 sidecar (default <store>.ocidx if present)")
    ap.add_argument("--atlas-expect-fnv", default=None, help="refuse a store whose FNV-1a-64 differs (0x<hex>)")
    ap.add_argument("--atlas-rss-cap-mib", type=int, default=1024, help="RSS cap of an atlas child")
    ap.add_argument("--atlas-wall-s", type=float, default=60.0, help="per-request wall limit of an atlas child (0 = off)")
    ap.add_argument("--atlas-hello-timeout-s", type=float, default=300.0,
                    help="time allowed for the atlas child's open (FNV + index check of the store)")
    return ap


def main(argv=None):
    cfg = build_parser().parse_args(argv)
    ok, why = bind_verdict(cfg.bind)
    if not ok:
        sys.stderr.write("oc_gateway: refused: %s\n" % why)
        return 2
    for name, p in (("--binary", cfg.binary), ("--verbs-json", cfg.verbs_json),
                    ("--page-dir", cfg.page_dir), ("--katex-dir", cfg.katex_dir)):
        if not os.path.exists(p):
            sys.stderr.write("oc_gateway: %s %s does not exist\n" % (name, p))
            return 2
    if cfg.rss_cap_mib <= 0:
        sys.stderr.write("oc_gateway: --rss-cap-mib must be > 0 (rule 19)\n")
        return 2
    cfg.binary = os.path.abspath(cfg.binary)
    if cfg.atlas_store is not None:
        if not os.path.isfile(cfg.atlas_store):
            sys.stderr.write("oc_gateway: --atlas-store %s is not a file\n" % cfg.atlas_store)
            return 2
        cfg.atlas_store = os.path.abspath(cfg.atlas_store)
        if cfg.atlas_index is not None:
            if not os.path.isfile(cfg.atlas_index):
                sys.stderr.write("oc_gateway: --atlas-index %s is not a file\n" % cfg.atlas_index)
                return 2
            cfg.atlas_index = os.path.abspath(cfg.atlas_index)
        if cfg.atlas_expect_fnv is not None:
            t = cfg.atlas_expect_fnv
            if not (t.startswith("0x") and 3 <= len(t) <= 18 and all(c in "0123456789abcdefABCDEF" for c in t[2:])):
                sys.stderr.write("oc_gateway: --atlas-expect-fnv takes 0x<1-16 hex digits>\n")
                return 2
        if cfg.atlas_rss_cap_mib <= 0:
            sys.stderr.write("oc_gateway: --atlas-rss-cap-mib must be > 0 (rule 19)\n")
            return 2
    elif cfg.atlas_index is not None or cfg.atlas_expect_fnv is not None:
        sys.stderr.write("oc_gateway: --atlas-index / --atlas-expect-fnv need --atlas-store\n")
        return 2
    try:
        ro = ReadOnlyList(cfg.verbs_json)
    except (OSError, ValueError, KeyError) as e:
        sys.stderr.write("oc_gateway: cannot read the read-only list: %s\n" % e)
        return 2
    os.makedirs(cfg.scratch, mode=0o700, exist_ok=True)
    gw = Gateway(cfg, ro)
    ip = parse_ip(cfg.bind)
    family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
    try:
        srv = GatewayServer((str(ip), cfg.port), make_handler(gw), family)
    except OSError as e:
        sys.stderr.write("oc_gateway: cannot bind %s:%d: %s\n" % (cfg.bind, cfg.port, e))
        return 2
    port = srv.server_address[1]
    host = "[%s]" % ip if ip.version == 6 else str(ip)
    sys.stdout.write("OC_GATEWAY listening=%s:%d proto=%s bind_check=\"%s\" read_only=%d/%d\n" % (
        host, port, GATEWAY_PROTO, why, ro.read_only, ro.read_only_verbs))
    sys.stdout.write("OC_GATEWAY atlas=%s\n" % (os.path.basename(cfg.atlas_store) if cfg.atlas_store else "none"))
    sys.stdout.write("OC_GATEWAY token=%s\n" % gw.token)
    sys.stdout.write("OC_GATEWAY url=http://%s:%d/#token=%s\n" % (host, port, gw.token))
    sys.stdout.write("OC_GATEWAY note: no TLS -- LAN traffic, the token included, is cleartext.\n")
    sys.stdout.flush()

    def on_term(signum, frame):
        threading.Thread(target=srv.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, on_term)
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        with gw.lock:
            sessions = list(gw.sessions.values())
        for s in sessions:
            s.close()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
