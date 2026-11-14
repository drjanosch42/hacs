#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

"""test_gateway.py -- the NB-3 gateway self-test ([P] Sec. 18.11 "NB-3 gateway conformance").

Stdlib only (unittest). Starts oc_gateway.py against a REAL `hacs --serve`
(built with -DOC_BUILD_NOTEBOOK=ON) and talks to it over WebSocket from loopback.

    python3 notebook/gateway/test_gateway.py --binary <build>/hacs \
            [--scratch <dir>] [--scripts 001_dimensions,005_linear_algebra | all] [-v]

`--scratch` (default ~/.cache/oc-notebook-test) receives the per-session working
directories and a few fixture files; it is created if missing and never inside the tree.
`--scripts all` serves the 34 runnable numbered scripts (the exempt four of [P] 18.14 (g)
excluded). Cost (default scripts): ~10 s, < 0.5 GB (the RSS-cap arm allocates ~300 MiB under a 128 MiB cap).
Exit code 0 iff every test passed; the last line is `GATEWAY_TEST ran=<k> failures=<f> errors=<e> verdict=OK|FAIL`.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import oc_gateway as G  # noqa: E402

OPTS = argparse.Namespace(binary=None, scratch=None, scripts="001_dimensions,005_linear_algebra")
SCRIPTS_DIR = os.path.join(ROOT, "src/hacs", "Console", "scripts")
WHITELIST_KEYS = {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "ENGINE_THREADS", "ENGINE_CUDA_DISABLE",
                  "ENGINE_CUDA_DEVICE", "ENGINE_HOST_THREADS"}
# [P] Sec. 18.14 (g): the scripts that call an unserved verb / overload (refused, not served).
EXEMPT_PREFIXES = {"014", "021", "022", "048"}
LOOP = "for i in range(0, 20000) { for j in range(0, 20000) { x = abs(j); } }"
PAGE_DIR = os.path.join(ROOT, "notebook", "page")


# ---------------------------------------------------------------- helpers
def documented_tour():
    # (slot-25 EXZ2-P3 fix round 1, audit F1) the tour files as documented in notebook/examples/README.md's table ("| k | `NN_*.oc` |"),
    # in order: the tests count against this list instead of a hard-coded number
    with open(os.path.join(ROOT, "notebook", "examples", "README.md"), encoding="utf-8") as f:
        return [m.group(1) for m in re.finditer(r"^\| *\d+ *\| *`(\d\d_[A-Za-z0-9_]+\.oc)` *\|", f.read(), re.M)]


def flatten(name, stack=()):
    """`include("x.oc");` lines replaced by the file's text (as ConsoleServeValidation's Flatten)."""
    assert name not in stack, "include cycle"
    with open(os.path.join(SCRIPTS_DIR, name), encoding="utf-8") as f:
        text = f.read()
    out = []
    for line in text.split("\n")[:-1] if text.endswith("\n") else text.split("\n"):
        s = line.lstrip(" \t")
        if s.startswith('include("'):
            out.append(flatten(s[9:s.index('"', 9)], stack + (name,)))
            continue
        out.append(line + "\n")
    return "".join(out)


def transcript_of(raw):
    out = []
    lines = raw.split("\n")
    if raw.endswith("\n"):
        lines = lines[:-1]
    for line in lines:
        if line.startswith("#~"):
            continue
        out.append(line.rstrip(" \r\t") + "\n")
    return "".join(out)


def expected_body(text):
    nl = text.find("\n")
    return "" if nl < 0 else text[nl + 1:]


class GatewayProc:
    def __init__(self, *extra, env_extra=None):
        env = dict(os.environ)
        env.update(env_extra or {})
        argv = [sys.executable, os.path.join(HERE, "oc_gateway.py"), "--binary", OPTS.binary, "--port", "0",
                "--scratch", os.path.join(OPTS.scratch, "sessions"), "--threads", "2"] + list(extra)
        self.p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
        self.port = self.token = None
        deadline = time.time() + 20
        while time.time() < deadline and self.token is None:
            line = self.p.stdout.readline()
            if not line:
                break
            if line.startswith("OC_GATEWAY listening="):
                self.port = int(line.split()[1].split("=")[1].rsplit(":", 1)[1])
            elif line.startswith("OC_GATEWAY token="):
                self.token = line.strip().split("=", 1)[1]
        if self.token is None:
            self.stop()
            raise RuntimeError("gateway did not start: " + self.p.stderr.read())

    def children(self):
        pids = []
        try:
            for t in os.listdir("/proc/%d/task" % self.p.pid):
                with open("/proc/%d/task/%s/children" % (self.p.pid, t)) as f:
                    pids += [int(x) for x in f.read().split()]
        except OSError:
            pass
        return pids

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        for s in (self.p.stdout, self.p.stderr):
            try:
                s.close()
            except Exception:
                pass


def ws_connect(port, headers=None, timeout=60):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    hdr = {"Host": "127.0.0.1:%d" % port, "Upgrade": "websocket", "Connection": "Upgrade",
           "Sec-WebSocket-Key": key, "Sec-WebSocket-Version": "13"}
    hdr.update(headers or {})
    req = "GET /ws HTTP/1.1\r\n" + "".join("%s: %s\r\n" % kv for kv in hdr.items()) + "\r\n"
    s.sendall(req.encode())
    rfile = s.makefile("rb")
    status = rfile.readline().decode()
    while rfile.readline() not in (b"\r\n", b""):
        pass
    if " 101 " not in status:
        s.close()
        return status.strip()
    return G.WebSocket(s, rfile=rfile, client=True)


class Client:
    def __init__(self, gp, token=None, tab="t-test", headers=None):
        self.ws = ws_connect(gp.port, headers)
        assert not isinstance(self.ws, str), self.ws
        self.ws.send_text(json.dumps({"op": "auth", "token": gp.token if token is None else token, "tab": tab}))
        self.n = 0

    def recv(self):
        t = self.ws.recv()
        return None if t is None else json.loads(t)

    def until(self, pred, limit=500):
        seen = []
        for _ in range(limit):
            fr = self.recv()
            if fr is None:
                break
            seen.append(fr)
            if pred(fr):
                return fr, seen
        return None, seen

    def hello(self):
        fr, seen = self.until(lambda f: f.get("kind") == "hello" or (f.get("kind") == "gw" and f.get("event") in ("refused", "unauthorized")))
        return fr

    def request(self, obj):
        self.n += 1
        obj["id"] = "q%d" % self.n
        self.ws.send_text(json.dumps(obj))
        return obj["id"]

    def run(self, op, **kw):
        """Send a request; return (terminal frame, all frames of it)."""
        kw["op"] = op
        rid = self.request(kw)
        frames = []
        while True:
            fr = self.recv()
            if fr is None:
                return None, frames
            if fr.get("id") == rid or fr.get("kind") == "gw":
                frames.append(fr)
            if fr.get("id") == rid and (fr.get("kind") in ("done", "atlas_reply") or
                                        (fr.get("kind") == "gw" and fr.get("event") == "rejected")):
                return fr, frames
            if fr.get("kind") == "gw" and fr.get("event") == "exited":
                return fr, frames

    def close(self):
        try:
            self.ws.close()
            self.ws.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------- unit tests (no binary)
class AddressRules(unittest.TestCase):
    def test_peer_filter(self):
        for a in ("127.0.0.1", "127.8.9.10", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.0.1",
                  "169.254.1.1", "::1", "fd00::1", "fc12::5", "fe80::1%eth0", "::ffff:192.168.1.1", "[::1]"):
            self.assertTrue(G.is_allowed_address(a), a)
        for a in ("8.8.8.8", "172.32.0.1", "172.15.255.255", "100.64.0.1", "100.127.255.254", "192.0.2.1",
                  "2001:db8::1", "2a00:1450::1", "::ffff:8.8.8.8", "0.0.0.0", "::", "localhost", "", None, "1.2.3"):
            self.assertFalse(G.is_allowed_address(a), a)

    def test_bind_verdict(self):
        self.assertTrue(G.bind_verdict("127.0.0.1")[0])
        self.assertTrue(G.bind_verdict("192.168.1.20")[0])
        self.assertTrue(G.bind_verdict("::1")[0])
        self.assertFalse(G.bind_verdict("8.8.8.8")[0])
        self.assertFalse(G.bind_verdict("100.64.0.1")[0])
        self.assertFalse(G.bind_verdict("localhost")[0])
        self.assertTrue(G.bind_verdict("0.0.0.0", addrs=["127.0.0.1", "192.168.1.5", "fe80::1"])[0])
        self.assertFalse(G.bind_verdict("0.0.0.0", addrs=["127.0.0.1", "203.0.113.7"])[0])
        self.assertFalse(G.bind_verdict("::", addrs=["::1", "2a00:1450::1"])[0])

    def test_server_drops_non_private_peer(self):
        cfg = G.build_parser().parse_args(["--binary", sys.executable])
        gw = G.Gateway(cfg, None)
        srv = G.GatewayServer(("127.0.0.1", 0), G.make_handler(gw), socket.AF_INET)
        try:
            self.assertFalse(srv.verify_request(None, ("203.0.113.5", 4444)))
            self.assertFalse(srv.verify_request(None, ("100.64.1.1", 4444)))
            self.assertTrue(srv.verify_request(None, ("127.0.0.1", 4444)))
            self.assertTrue(srv.verify_request(None, ("192.168.7.7", 4444)))
        finally:
            srv.server_close()

    def test_lexical_unserved(self):
        names = {"run_campaign", "exit", "bench"}
        self.assertEqual(G.first_unserved_call("x = 1;\n  run_campaign(s);", names), ("run_campaign", 2, 3))
        self.assertIsNone(G.first_unserved_call('s = "run_campaign(";\n# exit(1)\nprint(s);', names))
        self.assertIsNone(G.first_unserved_call("r.bench(1); my_exit(2); exit_code = 1;", names))
        self.assertEqual(G.first_unserved_call("exit (3)", names), ("exit", 1, 1))


# ---------------------------------------------------------------- page themes (slot-22 WP-NBT, no binary)
def theme_data():
    """The strict-JSON block of themes.js between the OC-THEME-DATA markers."""
    with open(os.path.join(PAGE_DIR, "themes.js"), encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"/\* OC-THEME-DATA-BEGIN \*/(.*?)/\* OC-THEME-DATA-END \*/", src, re.S)
    if not m:
        raise AssertionError("themes.js: no OC-THEME-DATA block")
    return json.loads(m.group(1))


def wcag_contrast(a, b):
    """WCAG 2.x contrast ratio of two #rrggbb colours."""
    def lum(h):
        def ch(i):
            c = int(h[i:i + 2], 16) / 255.0
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * ch(1) + 0.7152 * ch(3) + 0.0722 * ch(5)
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def css_without_comments():
    with open(os.path.join(PAGE_DIR, "style.css"), encoding="utf-8") as f:
        return re.sub(r"/\*.*?\*/", "", f.read(), flags=re.S)


ROOT_BLOCK = re.compile(r":root[^{]*\{([^{}]*)\}")
LITERAL_COLOUR = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch)\(|"
                            r":\s*[^;{}]*\b(?:black|white|red|green|blue|gray|grey|silver|yellow|orange|navy|maroon)\b")


class Themes(unittest.TestCase):
    def test_theme_presets_wcag_aa(self):
        d = theme_data()
        self.assertEqual(d["schema"], "oc-notebook-theme/1")
        toks = d["tokens"]
        self.assertEqual(len(toks), 13)
        self.assertTrue({"bg", "fg", "muted", "line", "panel", "accent", "ok", "err", "warn", "code",
                         "katex", "sel", "focus"} == set(toks))
        self.assertEqual(set(d["labels"]), set(toks))
        self.assertEqual(set(d["presets"]), {"light", "dark", "high-contrast-light", "high-contrast-dark"})
        floors = {}
        for f, b, floor in d["pairs"]:
            self.assertIn(f, toks)
            self.assertIn(b, toks)
            self.assertIn(floor, (3.0, 4.5))
            floors[(f, b)] = floor
        for pair in (("fg", "bg"), ("muted", "bg"), ("accent", "bg"), ("ok", "panel"), ("err", "panel"),
                     ("warn", "panel"), ("fg", "code")):   # the plan's item 7, at least
            self.assertEqual(floors.get(pair), 4.5, pair)
        n = 0
        for pid, p in d["presets"].items():
            self.assertIn(p["base"], ("light", "dark"), pid)
            self.assertEqual(set(p["tokens"]), set(toks), pid)
            for k, v in p["tokens"].items():
                self.assertRegex(v, r"^#[0-9a-f]{6}$", "%s.%s" % (pid, k))
            for (f, b), floor in floors.items():
                r = wcag_contrast(p["tokens"][f], p["tokens"][b])
                self.assertGreaterEqual(r, floor, "%s: %s/%s = %.2f" % (pid, f, b, r))
                n += 1
        print("THEME_CONTRAST presets=%d pairs=%d checked=%d below_aa=0" % (len(d["presets"]), len(floors), n),
              file=sys.stderr)

    def test_css_tokens_only(self):
        d = theme_data()
        css = css_without_comments()
        outside = ROOT_BLOCK.sub("", css)
        self.assertEqual(LITERAL_COLOUR.findall(outside), [], "literal colour in style.css outside :root")
        for name in ("render.js", "app.js"):
            with open(os.path.join(PAGE_DIR, name), encoding="utf-8") as f:
                self.assertIsNone(re.search(r"#[0-9a-fA-F]{6}\b|\brgba?\(", f.read()), name)
        blocks = ROOT_BLOCK.findall(css)
        self.assertEqual(len(blocks), 3)            # light; dark under prefers-color-scheme; [data-theme=dark]
        decl = [dict(re.findall(r"--([a-z]+)\s*:\s*(#[0-9a-fA-F]{6})", b)) for b in blocks]
        light, dark = d["presets"]["light"]["tokens"], d["presets"]["dark"]["tokens"]
        self.assertEqual(decl[0], light, "style.css :root != themes.js light")
        self.assertEqual(decl[1], dark, "style.css prefers-color-scheme dark != themes.js dark")
        self.assertEqual(decl[2], dark, "style.css [data-theme=dark] != themes.js dark")
        used = set(re.findall(r"var\(--([a-z]+)\)", css))
        self.assertLessEqual(used, set(d["tokens"]) | {"shadow"})
        self.assertIn("katex", used)

    def test_theme_js_pure(self):
        node = shutil.which("node") or shutil.which("nodejs")
        if not node:
            self.skipTest("no node on PATH: the pure functions of themes.js are not exercised")
        js = r"""
const T = require(process.argv[1]); const a = require("assert");
for (const id of Object.keys(T.DATA.presets)) {
  const t = T.presetTheme(id); a.strictEqual(T.checkContrast(t.tokens).filter(r => !r.ok).length, 0, id);
  const s = T.serializeTheme(t); const r = T.parseTheme(s); a.ok(r.theme, id);
  a.strictEqual(T.serializeTheme(r.theme), s, "round trip " + id);
}
a.ok(Math.abs(T.contrast("#000000", "#ffffff") - 21) < 1e-9);
a.ok(Math.abs(T.contrast("#777777", "#ffffff") - 4.478) < 1e-3);
for (const bad of ["{", "[]", '{"schema":"x"}', '{"schema":"oc-notebook-theme/1","base":"sepia"}',
                   '{"schema":"oc-notebook-theme/1","tokens":{"bg":"red"}}',
                   '{"schema":"oc-notebook-theme/1","tokens":{"bg":"#fff"}}',
                   '{"schema":"oc-notebook-theme/1","tokens":{"fg":"#000000;}body{x"}}',
                   '{"schema":"oc-notebook-theme/1","tokens":[]}', '{"schema":"oc-notebook-theme/1","name":7}'])
  a.ok(T.parseTheme(bad).error, bad);
const r = T.parseTheme({schema: T.SCHEMA, base: "dark", name: "  mine ", tokens: {bg: "#ABCDEF", zz: "#000000"}, x: 1});
a.strictEqual(r.theme.name, "mine"); a.strictEqual(r.theme.tokens.bg, "#abcdef");
a.strictEqual(r.theme.tokens.fg, T.DATA.presets.dark.tokens.fg); a.strictEqual(r.notes.length, 3);
const low = T.parseTheme({schema: T.SCHEMA, tokens: {fg: "#bbbbbb", bg: "#ffffff"}}).theme;
a.ok(T.checkContrast(low.tokens).some(p => p.fg === "fg" && p.bg === "bg" && !p.ok), "low contrast flagged");
console.log("THEME_JS ok");
"""
        p = subprocess.run([node, "-e", js, os.path.join(PAGE_DIR, "themes.js")], capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("THEME_JS ok", p.stdout)

    def test_render_fixtures(self):
        # (slot-24 NB-RENDER R2) render.js against the frozen envelopes of fixtures/envelopes.ndjson (test_render.mjs)
        node = shutil.which("node") or shutil.which("nodejs")
        if not node:
            self.skipTest("no node on PATH: the render fixtures are not exercised")
        p = subprocess.run([node, os.path.join(HERE, "test_render.mjs")], capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stdout[-4000:] + p.stderr[-2000:])
        last = p.stdout.strip().splitlines()[-1]
        self.assertTrue(last.startswith("RENDER_TEST ") and last.endswith("verdict=OK"), last)
        print(last, file=sys.stderr)


# ---------------------------------------------------------------- against the real binary
class Refusals(unittest.TestCase):
    def test_public_bind_refused(self):
        for b in ("8.8.8.8", "100.64.0.1", "localhost", "2a00:1450::1"):
            p = subprocess.run([sys.executable, os.path.join(HERE, "oc_gateway.py"), "--binary", OPTS.binary,
                                "--bind", b, "--port", "0"], capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, 2, b)
            self.assertIn("refused", p.stderr, b)
            self.assertNotIn("token=", p.stdout, b)


class Live(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The gateway's OWN environment carries emitter / hook variables: the child must not see them.
        cls.gp = GatewayProc(env_extra={"GOLDEN_EMIT": "1", "CONSOLE_EMIT_EXPECTED": "1",
                                        "HECKE_NEBENTYPUS_NB3_FULL": "1", "ENGINE_DEVICE": "gpu", "OC_FOO": "1"})

    @classmethod
    def tearDownClass(cls):
        cls.gp.stop()

    def test_static_files(self):
        base = "http://127.0.0.1:%d" % self.gp.port
        with urllib.request.urlopen(base + "/") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("default-src 'none'", r.headers["Content-Security-Policy"])
            self.assertIn(b"katex.min.js", r.read())
        for path in ("/katex/katex.min.js", "/katex/katex.min.css",
                     "/static/app.js", "/static/render.js", "/katex/fonts/KaTeX_Main-Regular.woff2"):
            with urllib.request.urlopen(base + path) as r:
                self.assertEqual(r.status, 200, path)
        for path in ("/static/../gateway/oc_gateway.py", "/katex/../../gateway/oc_gateway.py",
                     "/static/%2e%2e/gateway/oc_gateway.py", "/etc/passwd", "/static/", "/katex/SHA256SUMS"):
            with self.assertRaises(urllib.error.HTTPError, msg=path) as cm:
                urllib.request.urlopen(base + path)
            self.assertEqual(cm.exception.code, 404, path)

    def test_examples_files(self):
        # (slot-24 NB-FILE) the example notebooks: one fixed directory, flat, read-only; no path can leave it.
        base = "http://127.0.0.1:%d" % self.gp.port
        ex_dir = os.path.join(ROOT, "notebook", "examples", "notebook")
        with urllib.request.urlopen(base + "/examples/index.json") as r:
            self.assertEqual(r.status, 200)
            ix = json.loads(r.read())
        self.assertEqual(ix["schema"], "oc-notebook-examples/1")
        want = sorted(n for n in os.listdir(ex_dir) if n.endswith(".oc"))
        self.assertEqual(ix["files"], want)
        self.assertIn("tour_all.oc", ix["files"])
        tour = documented_tour()
        self.assertGreaterEqual(len(tour), 9, tour)
        self.assertEqual([n for n in ix["files"] if n[:2].isdigit()], tour)
        for n in ix["files"]:
            with urllib.request.urlopen(base + "/examples/" + n) as r:
                self.assertEqual(r.status, 200, n)
                self.assertTrue(r.headers["Content-Type"].startswith("text/plain"), n)
                with open(os.path.join(ex_dir, n), "rb") as f:
                    self.assertEqual(r.read(), f.read(), n)
        for path in ("/examples/", "/examples/README.md", "/examples/../page/app.js", "/examples/%2e%2e/page/app.js",
                     "/examples/..%2fpage%2fapp.js", "/examples/01_model_and_local_system.oc/", "/examples/../examples/tour_all.oc",
                     "/examples//etc/passwd", "/examples/%2fetc%2fpasswd", "/examples/01_model_and_local_system.oc%00.txt",
                     "/examples/dickson_theta_16_7.oc", "/examples/index.json/x"):
            with self.assertRaises(urllib.error.HTTPError, msg=path) as cm:
                urllib.request.urlopen(base + path)
            self.assertEqual(cm.exception.code, 404, path)

    def test_theme_assets(self):
        base = "http://127.0.0.1:%d" % self.gp.port
        with urllib.request.urlopen(base + "/") as r:
            page = r.read().decode("utf-8")
        # themes.js is a blocking <head> script (no first paint in the wrong theme), before app.js
        self.assertIn('<script src="/static/themes.js"></script>', page)
        self.assertLess(page.index("/static/themes.js"), page.index("/static/app.js"))
        for i in ("theme-select", "btn-theme", "theme-editor", "theme-warn"):
            self.assertIn('id="%s"' % i, page)
        for path, ctype in (("/static/themes.js", "text/javascript"), ("/static/style.css", "text/css")):
            with urllib.request.urlopen(base + path) as r:
                self.assertEqual(r.status, 200, path)
                self.assertTrue(r.headers["Content-Type"].startswith(ctype), path)
                csp = r.headers["Content-Security-Policy"]
                self.assertIn("script-src 'self';", csp)
                self.assertIn("style-src 'self' 'unsafe-inline';", csp)
                with open(os.path.join(PAGE_DIR, path[len("/static/"):]), "rb") as f:
                    self.assertEqual(r.read(), f.read(), path)

    def test_token_required(self):
        before = set(self.gp.children())
        for tok in ("", "wrong-" + self.gp.token[6:], self.gp.token[:-1]):
            c = Client(self.gp, token=tok, tab="t-bad")
            fr = c.recv()
            self.assertEqual((fr.get("kind"), fr.get("event")), ("gw", "unauthorized"), tok)
            self.assertIsNone(c.recv())       # the gateway closes
            c.close()
        # a first frame that is not auth
        ws = ws_connect(self.gp.port)
        ws.send_text(json.dumps({"op": "eval", "id": "x", "src": "print(1);"}))
        fr = json.loads(ws.recv())
        self.assertEqual(fr.get("event"), "unauthorized")
        ws.sock.close()
        # cross-origin upgrade refused before any frame
        st = ws_connect(self.gp.port, headers={"Origin": "http://evil.example"})
        self.assertIsInstance(st, str)
        self.assertIn("403", st)
        time.sleep(0.3)
        self.assertEqual(set(self.gp.children()) - before, set(), "no child was spawned without a token")

    def test_script_roundtrip_env_and_cwd(self):
        c = Client(self.gp, tab="t-script")
        hello = c.hello()
        self.assertEqual(hello.get("proto"), "oc-serve/1")
        ro = G.ReadOnlyList(os.path.join(ROOT, "docs", "latex", "cli_manual", "read_only_verbs.json"))
        self.assertTrue(ro.matches_hello(hello))
        pid = hello["build"]["pid"]
        with open("/proc/%d/environ" % pid, "rb") as f:
            keys = {kv.split(b"=", 1)[0].decode() for kv in f.read().split(b"\0") if kv}
        self.assertTrue(keys <= WHITELIST_KEYS, keys - WHITELIST_KEYS)
        self.assertIn("ENGINE_CUDA_DISABLE", keys)
        cwd = os.readlink("/proc/%d/cwd" % pid)
        self.assertTrue(cwd.startswith(os.path.join(OPTS.scratch, "sessions")), cwd)
        if OPTS.scripts == "all":
            names = sorted(n[:-3] for n in os.listdir(SCRIPTS_DIR)
                           if n[:3].isdigit() and n.endswith(".oc") and n[:3] not in EXEMPT_PREFIXES)
        else:
            names = [n.strip() for n in OPTS.scripts.split(",") if n.strip()]
        print("EFFECTIVE test_gateway arm=scripts cells=%d" % len(names), file=sys.stderr)
        for name in names:
            src = flatten(name + ".oc")
            done, frames = c.run("eval", src=src)
            self.assertIsNotNone(done, name)
            self.assertEqual((done.get("status"), done.get("exit")), ("ok", 0), (name, done.get("error")))
            out = "".join(f["text"] for f in frames if f.get("kind") == "out" and f.get("stream") == "stdout")
            with open(os.path.join(SCRIPTS_DIR, name + ".expected"), encoding="utf-8") as f:
                exp = f.read()
            self.assertTrue(exp.startswith("#! oc-format 1\n"))
            self.assertEqual(transcript_of(out), expected_body(exp), name)
            seqs = [f["seq"] for f in frames if "seq" in f]
            self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(os.listdir("/proc/%d/cwd" % pid), [], "the session's working directory stays empty")
        c.close()

    def test_tour_any_order(self):
        # (slot-24 NB-TOURFIX) the notebook tour runs in one session in order 01..NN and then NN..01 (NN = the last documented
        # file; 10 since slot-26 TOUR-Z): every file states the coefficient_ring it needs (01..07 set F_ell, 08 and later set Z; none restores),
        # so no setting leaks from one file to the next.
        tour = os.path.join(ROOT, "notebook", "examples", "notebook")
        names = sorted(n for n in os.listdir(tour) if n[:2].isdigit() and n.endswith(".oc"))
        self.assertEqual(names, documented_tour(), "the tour directory = the README's table (audit F1: no hard-coded count)")
        c = Client(self.gp, tab="t-tour")
        c.hello()
        print("EFFECTIVE test_gateway arm=tour cells=%d" % (2 * len(names)), file=sys.stderr)
        for name in names + names[::-1]:
            with open(os.path.join(tour, name), encoding="utf-8") as f:
                src = f.read()
            done, frames = c.run("eval", src=src)
            self.assertIsNotNone(done, name)
            self.assertEqual((done.get("status"), done.get("exit")), ("ok", 0), (name, done.get("error")))
            if name.startswith("08"):
                self.assertTrue(any("(X + 24)^2 (X - 2049)" in json.dumps(f) for f in frames if f.get("kind") == "value"),
                                "08: char_Z(T_2) factored in a value frame")
        c.close()

    def test_showcases(self):
        # (slot-26 NBMATH A3) the showcase notebooks (showcase_*.oc, not part of the numbered tour) run in one served session,
        # each ok / exit 0 (every verb read_only; each states the coefficient_ring it needs); Ramanujan shows C_T in a value frame.
        tour = os.path.join(ROOT, "notebook", "examples", "notebook")
        names = sorted(n for n in os.listdir(tour) if n.startswith("showcase_") and n.endswith(".oc"))
        self.assertGreaterEqual(len(names), 2, names)
        c = Client(self.gp, tab="t-showcase")
        c.hello()
        print("EFFECTIVE test_gateway arm=showcases cells=%d" % len(names), file=sys.stderr)
        for name in names:
            with open(os.path.join(tour, name), encoding="utf-8") as f:
                src = f.read()
            done, frames = c.run("eval", src=src)
            self.assertIsNotNone(done, name)
            self.assertEqual((done.get("status"), done.get("exit")), ("ok", 0), (name, done.get("error")))
            if name == "showcase_ramanujan_691.oc":
                self.assertTrue(any("certified:0x2b3" in json.dumps(f) for f in frames if f.get("kind") == "value"),
                                "showcase_ramanujan_691: C_T in a value frame")
        c.close()

    def test_non_read_only_refused(self):
        c = Client(self.gp, tab="t-ro")
        c.hello()
        fr, _ = c.run("eval", src='x = 1;\nrun_campaign(campaign_spec_text("x"), "out");')
        self.assertEqual((fr.get("kind"), fr.get("event")), ("gw", "rejected"))
        self.assertEqual(fr["error"]["kind"], "forbidden")
        self.assertEqual(fr["error"]["at"], {"line": 2, "col": 1})
        fr, _ = c.run("eval", src='save_state(1, "nb3_sentinel");')     # tagged name, untagged overload: the binary refuses
        self.assertEqual((fr.get("status"), fr.get("exit"), fr["error"]["kind"]), ("error", 3, "forbidden_overload"))
        fr, frames = c.run("eval", src='s = "run_campaign(";\n# exit(1)\nprint(1 + 1);')
        self.assertEqual(fr.get("status"), "ok")
        self.assertEqual([f["text"] for f in frames if f.get("kind") == "out"], ["2\n"])
        fr, _ = c.run("bogus")
        self.assertEqual(fr.get("event"), "rejected")
        self.assertEqual(fr["error"]["kind"], "protocol")
        c.close()

    def test_interrupt_and_busy(self):
        c = Client(self.gp, tab="t-int")
        c.hello()
        fr, _ = c.run("eval", src="k = 41;")
        self.assertEqual(fr.get("status"), "ok")
        rid = c.request({"op": "eval", "src": LOOP})
        time.sleep(1.0)
        busy = c.request({"op": "eval", "src": "print(0);"})
        c.ws.send_text(json.dumps({"op": "interrupt"}))
        t0 = time.time()
        got_busy, done = False, None
        while done is None:
            f = c.recv()
            self.assertIsNotNone(f)
            if f.get("kind") == "gw" and f.get("event") == "rejected" and f.get("id") == busy:
                got_busy = f["error"]["kind"] == "busy"
            if f.get("kind") == "done" and f.get("id") == rid:
                done = f
        self.assertTrue(got_busy, "a second request while one runs is refused busy")
        self.assertEqual((done["status"], done["exit"], done["error"]["kind"]), ("interrupted", 3, "interrupted"))
        self.assertLess(time.time() - t0, 30)
        fr, frames = c.run("eval", src="print(k + 1);")
        self.assertEqual([f["text"] for f in frames if f.get("kind") == "out"], ["42\n"])
        c.close()

    def test_interrupt_inside_verb(self):
        # (NBCONFIRM S5) the Interrupt reaches the running verb at its next cancellation point (hida_tower: per Hecke
        # operator), not only the next statement: one ~5 s statement ends `interrupted` (before S5 it finished `ok`), binds
        # nothing, and the session goes on.
        c = Client(self.gp, tab="t-int-verb")
        c.hello()
        rid = c.request({"op": "eval", "src": "x = hida_tower(160, 5, 1)"})
        time.sleep(1.0)
        c.ws.send_text(json.dumps({"op": "interrupt"}))
        t0 = time.time()
        done = None
        while done is None:
            f = c.recv()
            self.assertIsNotNone(f)
            if f.get("kind") == "done" and f.get("id") == rid:
                done = f
        self.assertEqual((done["status"], done["exit"], done["error"]["kind"]), ("interrupted", 3, "interrupted"))
        self.assertLess(time.time() - t0, 30)
        fr, _ = c.run("eval", src="print(x);")
        self.assertEqual(fr.get("status"), "error", "the interrupted statement bound nothing")
        fr, frames = c.run("eval", src="print(1 + 1);")
        self.assertEqual([f["text"] for f in frames if f.get("kind") == "out"], ["2\n"])
        c.close()

    def test_one_session_per_tab(self):
        a = Client(self.gp, tab="t-same")
        h1 = a.hello()
        b = Client(self.gp, tab="t-same")
        h2 = b.hello()
        self.assertNotEqual(h1["build"]["pid"], h2["build"]["pid"])
        fr, _ = a.until(lambda f: f.get("kind") == "gw" and f.get("event") == "replaced")
        self.assertIsNotNone(fr)
        time.sleep(0.5)
        self.assertFalse(os.path.exists("/proc/%d/cmdline" % h1["build"]["pid"])
                         and open("/proc/%d/stat" % h1["build"]["pid"]).read().split()[2] not in ("Z", "X"),
                         "the replaced session's child is gone")
        a.close()
        b.close()


class Limits(unittest.TestCase):
    def test_cell_wall_kill_then_restart(self):
        gp = GatewayProc("--cell-wall-s", "2")
        try:
            c = Client(gp, tab="t-wall")
            h1 = c.hello()
            fr, frames = c.run("eval", src=LOOP)
            self.assertEqual((fr.get("kind"), fr.get("event"), fr.get("reason")), ("gw", "exited", "cell_wall"))
            c.ws.send_text(json.dumps({"op": "restart"}))
            h2 = c.hello()
            self.assertEqual(h2.get("kind"), "hello")
            self.assertNotEqual(h1["build"]["pid"], h2["build"]["pid"])
            fr, frames = c.run("eval", src="print(6 * 7);")
            self.assertEqual([f["text"] for f in frames if f.get("kind") == "out"], ["42\n"])
            c.close()
        finally:
            gp.stop()

    def test_rss_cap(self):
        gp = GatewayProc("--rss-cap-mib", "128")
        try:
            c = Client(gp, tab="t-rss")
            c.hello()
            fr, frames = c.run("eval", src="M = random_matrix(7, 9000, 9000, 1);")
            kinds = [(f.get("kind"), f.get("event") or f.get("reason")) for f in frames]
            fatal = any(f.get("kind") == "fatal" and f.get("reason") == "rss_cap" for f in frames) or \
                (fr.get("kind") == "gw" and fr.get("reason") in ("rss_cap", "fatal_rss_cap"))
            bad_alloc = fr.get("kind") == "done" and fr.get("status") == "error" and fr.get("exit") == 3
            self.assertTrue(fatal or bad_alloc, kinds)
            c.close()
        finally:
            gp.stop()

    def test_read_only_list_mismatch_refuses_session(self):
        src = os.path.join(ROOT, "docs", "latex", "cli_manual", "read_only_verbs.json")
        with open(src, encoding="utf-8") as f:
            d = json.load(f)
        for e in d["entries"]:
            if e["verb"] == "print":
                e["read_only"] = False
        d["read_only"] -= 1
        d["read_only_verbs"] -= 1
        path = os.path.join(OPTS.scratch, "read_only_verbs.mismatch.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
        gp = GatewayProc("--verbs-json", path)
        try:
            c = Client(gp, tab="t-mis")
            fr = c.hello()
            self.assertEqual((fr.get("kind"), fr.get("event"), fr.get("reason")), ("gw", "refused", "read_only_mismatch"))
            c.close()
        finally:
            gp.stop()


class Confirm(unittest.TestCase):
    """(NBCONFIRM S4, [P] Sec. 18.4 / 18.6): `confirm` forwarded; a confirmed cell's wall limit; nothing a page sends
    raises a limit. descent(1, 62) is just over its envelope (n <= 60): ~1-3 s, < 1 GiB; confirm_wall_s = 0 makes it a
    normal confirmation."""

    CELL = 'set coefficient_ring = "Z"; set confirm_wall_s = 0; r = descent(1, 62);'

    def test_confirmed_wall_formula(self):
        cfg = argparse.Namespace(cell_wall_s=1800.0, cell_wall_max_s=21600.0)
        s = G.Session.__new__(G.Session)
        s.cfg = cfg
        self.assertEqual(s.confirmed_wall(10.0), 1800.0)                 # max(1800, 90)
        self.assertEqual(s.confirmed_wall(1000.0), 3060.0)               # 3 * 1000 + 60
        self.assertEqual(s.confirmed_wall(1e9), 21600.0)                 # capped at 6 h
        self.assertEqual(s.confirmed_wall(float("nan")), 1800.0)         # not a number: unchanged
        self.assertEqual(s.confirmed_wall(True), 1800.0)
        cfg.cell_wall_max_s = 0.0
        self.assertEqual(s.confirmed_wall(1e9), 1800.0)                  # 0: a confirmed cell keeps --cell-wall-s
        cfg.cell_wall_s, cfg.cell_wall_max_s = 0.0, 21600.0
        self.assertEqual(s.confirmed_wall(1e9), 0.0)                     # no wall limit at all stays off

    def test_confirm_roundtrip_and_wall(self):
        gp = GatewayProc("--cell-wall-s", "5", "--cell-wall-max-s", "30")
        try:
            c = Client(gp, tab="t-confirm")
            h = c.hello()
            self.assertEqual(h["limits"]["confirm_wall_s"], 60)
            self.assertIn("confirm_wall_s", h["settable"])
            fr, frames = c.run("eval", src=self.CELL)
            self.assertEqual((fr.get("status"), fr.get("exit")), ("confirm_required", 3), fr)
            tok = fr["confirm"]["token"]
            self.assertRegex(tok, r"^0x[0-9a-f]{16}$")
            self.assertGreater(fr["confirm"]["wall_hi"], 0)
            # a forged token, extra fields: forwarded, refused by the child, no wall change
            fr, frames = c.run("eval", src=self.CELL, confirm=["0x0123456789abcdef"], wall_hi=1e9, rss_cap_mib=999999)
            self.assertEqual(fr.get("status"), "confirm_required")
            self.assertEqual(fr["confirm"]["token"], tok)
            self.assertFalse([f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"])
            # malformed lists are refused by the gateway itself
            for bad in ("0x1", [1], [""], ["x" * 65], ["a b"], [tok] * 17):
                fr, _ = c.run("eval", src="print(1);", confirm=bad)
                self.assertEqual((fr.get("kind"), fr.get("event"), fr["error"]["kind"]), ("gw", "rejected", "protocol"), bad)
            # the token confirms: the wall limit is raised to min(3 * wall_hi + 60, --cell-wall-max-s) = 30
            fr, frames = c.run("eval", src=self.CELL, confirm=[tok])
            self.assertEqual(fr.get("status"), "ok", fr)
            walls = [f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"]
            self.assertTrue(walls, frames)
            self.assertEqual(walls[0]["cell_wall_s"], 30)
            self.assertTrue(any(f.get("kind") == "progress" and isinstance(f.get("confirmed"), dict) for f in frames))
            # the lazy work starts in a LATER cell without a token: the child's progress frame raises that cell's limit
            fr, frames = c.run("eval", src="print(r.status);")
            self.assertEqual(fr.get("status"), "ok", fr)
            self.assertEqual([f["text"] for f in frames if f.get("kind") == "out"], ["ok\n"])
            walls = [f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"]
            self.assertTrue(walls and walls[0]["cell_wall_s"] == 30, frames)
            # an ordinary cell afterwards: no raise
            fr, frames = c.run("eval", src="print(6 * 7);")
            self.assertFalse([f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"])
            # after a restart the old token is stale (a new session, a new nonce)
            c.ws.send_text(json.dumps({"op": "restart"}))
            c.hello()
            fr, frames = c.run("eval", src=self.CELL, confirm=[tok])
            self.assertEqual(fr.get("status"), "confirm_required")
            self.assertNotEqual(fr["confirm"]["token"], tok)
            self.assertFalse([f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"])
            c.close()
        finally:
            gp.stop()

    def test_stale_token_does_not_raise_wall(self):
        """(fix round 1, audit M1) a token the child offered for ANOTHER call, presented with an unconfirmed cell, raises
        nothing: only the child's own progress.confirmed frame does. The audit's cell (t7.log) is killed at --cell-wall-s."""
        gp = GatewayProc("--cell-wall-s", "3", "--cell-wall-max-s", "100")
        try:
            c = Client(gp, tab="t-stale")
            c.hello()
            fr, _ = c.run("eval", src=self.CELL)
            self.assertEqual(fr.get("status"), "confirm_required", fr)
            tok = fr["confirm"]["token"]
            slow = 'set coefficient_ring = "F_ell"; h = hida_tower(200, 5, 1); print(h.n);'
            t0 = time.monotonic()
            fr, frames = c.run("eval", src=slow, confirm=[tok])
            took = time.monotonic() - t0
            self.assertFalse([f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"], frames)
            self.assertFalse([f for f in frames if f.get("kind") == "progress" and isinstance(f.get("confirmed"), dict)])
            self.assertEqual((fr.get("kind"), fr.get("event"), fr.get("reason")), ("gw", "exited", "cell_wall"), fr)
            self.assertLess(took, 30.0)   # killed at the base 3 s (+ watchdog poll), not at min(3 wall_hi + 60, 100)
            c.close()
        finally:
            gp.stop()

    def test_non_idempotent_cell_stops_after_one_retry(self):
        """(fix round 1, audit M2) `k = k + 2; r = descent(1, 60 + k)`: the re-send re-executes `k = k + 2`, the call's
        argument changes, the child asks again at the same site with a new token; the page's rule (render.js
        confirmStep) stops there instead of showing a second dialog. The earlier statement has run twice (k = 4)."""
        gp = GatewayProc("--cell-wall-s", "60")
        try:
            c = Client(gp, tab="t-idem")
            c.hello()
            fr, _ = c.run("eval", src='set coefficient_ring = "Z"; set confirm_wall_s = 0; k = 0;')
            self.assertEqual(fr.get("status"), "ok", fr)
            cell = "k = k + 2; r = descent(1, 60 + k);"
            fr1, _ = c.run("eval", src=cell)
            self.assertEqual(fr1.get("status"), "confirm_required", fr1)
            fr2, frames = c.run("eval", src=cell, confirm=[fr1["confirm"]["token"]])
            self.assertEqual(fr2.get("status"), "confirm_required", fr2)
            self.assertEqual(fr2["confirm"]["site"], fr1["confirm"]["site"])
            self.assertNotEqual(fr2["confirm"]["token"], fr1["confirm"]["token"])
            self.assertFalse([f for f in frames if f.get("kind") == "gw" and f.get("event") == "cell_wall"])
            fr, frames = c.run("eval", src="print(k);")
            self.assertEqual([f["text"] for f in frames if f.get("kind") == "out"], ["4\n"])
            c.close()
        finally:
            gp.stop()
        node = shutil.which("node") or shutil.which("nodejs")
        if not node:
            self.skipTest("no node on PATH: the page rule confirmStep is not exercised (the gateway part passed)")
        js = ("const R = require(process.argv[1]); const [f1, f2, f3] = JSON.parse(process.argv[2]);"
              "const asked = []; const out = [];"
              "let s = R.confirmStep(asked, [], f1); out.push(s.action); if (s.action === 'ask') asked.push(s.site);"
              "s = R.confirmStep(asked, [f1.confirm.token], f2); out.push(s.action, typeof s.message === 'string' && /idempotent/.test(s.message));"
              "out.push(R.confirmStep(asked, [f1.confirm.token], f3).action);"
              "console.log(JSON.stringify(out));")
        other = json.loads(json.dumps(fr2))
        other["confirm"]["site"] = {"line": 7, "col": 1}   # a different call site still gets its dialog
        p = subprocess.run([node, "-e", js, os.path.join(PAGE_DIR, "render.js"), json.dumps([fr1, fr2, other])],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout.strip().splitlines()[-1]), ["ask", "stop", True, "ask"])


class AtlasForward(unittest.TestCase):
    """The gateway's own check of atlas queries (slot-20 NB-4, [P] v3.11 Sec. 18.16 (f)); no binary."""

    def test_forward(self):
        fwd, err = G.atlas_forward({"op": "atlas", "id": "a1", "query": "systems",
                                    "filter": {"r": 2, "type": "cuspidal", "collected": False, "n": 22}, "limit": 5})
        self.assertIsNone(err)
        self.assertEqual(fwd, {"op": "systems", "id": "a1", "filter": {"r": 2, "type": "cuspidal", "collected": False,
                                                                       "n": 22}, "limit": 5})
        for bad in ({"query": "close"}, {"query": "bogus"}, {"query": "system", "sid": -1},
                    {"query": "system", "sid": True}, {"query": "system", "sid": "3"},
                    {"query": "system", "sid": 0, "path": "/etc/passwd"}, {"query": "info", "store": "x.json"},
                    {"query": "systems", "filter": {"type": "x"}}, {"query": "systems", "filter": {"zzz": 1}},
                    {"query": "systems", "filter": {"collected": 1}}, {"query": "systems", "filter": []},
                    {"query": "cells", "limit": 2 ** 40}):
            msg = dict(bad, op="atlas", id="a")
            fwd, err = G.atlas_forward(msg)
            self.assertIsNone(fwd, bad)
            self.assertIn("gateway:", err)


def atlas_fixture():
    """A small hecke_atlas/v1 store (Sigma(13), built by the binary into the scratch) and its sidecar."""
    d = os.path.join(OPTS.scratch, "atlas13")
    store = os.path.join(d, "hecke_atlas_ell13.json")
    if not os.path.exists(store + ".ocidx"):
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, HECKE_ATLAS_OUT=d, ENGINE_CUDA_DISABLE="1")
        p = subprocess.run([OPTS.binary, "--atlas", "13", "--quiet"], env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0 and os.path.exists(store), p.stdout + p.stderr
        p = subprocess.run([OPTS.binary, "--atlas", "--index", store], capture_output=True, text=True, timeout=300)
        assert p.returncode == 0 and "ATLAS_INDEX" in p.stdout, p.stdout + p.stderr
    return store


class Atlas(unittest.TestCase):
    """The atlas channel end to end: gateway -> --serve-atlas child (slot-20 NB-4)."""

    def test_atlas_channel(self):
        store = atlas_fixture()
        with open(store, "rb") as f:
            before = f.read()
        mtime = os.stat(store).st_mtime_ns
        gp = GatewayProc("--atlas-store", store, "--atlas-expect-fnv", "0x" + "%016x" % fnv1a64(before),
                         env_extra={"GOLDEN_EMIT": "1", "HECKE_ATLAS_OUT": os.path.join(OPTS.scratch, "nope")})
        try:
            c = Client(gp, tab="t-atlas")
            hello, seen = c.until(lambda f: f.get("kind") == "hello")
            self.assertIsNotNone(hello)
            sess = [f for f in seen if f.get("kind") == "gw" and f.get("event") == "session"]
            sess = sess or [c.until(lambda f: f.get("kind") == "gw" and f.get("event") == "session")[0]]
            self.assertEqual(sess[0].get("atlas"), {"configured": True, "store": "hecke_atlas_ell13.json"})
            fr, frames = c.run("atlas", query="info")
            self.assertEqual((fr.get("kind"), fr.get("status")), ("atlas_reply", "ok"), fr)
            self.assertEqual(fr["result"]["header"]["ell"], 13)
            self.assertEqual(fr["store"], "0x%016x" % fnv1a64(before))
            fr, _ = c.run("atlas", query="system", sid=0)
            self.assertEqual((fr.get("status"), fr["result"]["system"]["id"]), ("ok", 0))
            fr, _ = c.run("atlas", query="systems", filter={"type": "cuspidal"}, limit=3)
            self.assertEqual(fr.get("status"), "ok")
            self.assertTrue(all(s["type"] == "cuspidal" for s in fr["result"]["items"]))
            fr, _ = c.run("atlas", query="system", sid=10 ** 6)
            self.assertEqual((fr.get("status"), fr["error"]["kind"]), ("error", "not_found"))
            for bad in ({"query": "system", "sid": 0, "path": "/etc/passwd"}, {"query": "close"}, {"query": "bogus"}):
                fr, _ = c.run("atlas", **bad)
                self.assertEqual((fr.get("kind"), fr.get("event"), fr["error"]["kind"]), ("gw", "rejected", "protocol"), bad)
            # the atlas child: whitelist environment, its own empty scratch working directory
            pids = [p for p in gp.children()]
            atlas_pids = []
            for p in pids:
                try:
                    with open("/proc/%d/cmdline" % p, "rb") as f:
                        if b"--serve-atlas" in f.read():
                            atlas_pids.append(p)
                except OSError:
                    pass
            self.assertEqual(len(atlas_pids), 1, "one --serve-atlas child per session")
            pid = atlas_pids[0]
            with open("/proc/%d/environ" % pid, "rb") as f:
                keys = {kv.split(b"=", 1)[0].decode() for kv in f.read().split(b"\0") if kv}
            self.assertTrue(keys <= set(G.ENV_WHITELIST_INERT), keys)
            cwd = os.readlink("/proc/%d/cwd" % pid)
            self.assertTrue(cwd.startswith(os.path.join(OPTS.scratch, "sessions")), cwd)
            self.assertEqual(os.listdir("/proc/%d/cwd" % pid), [])
            c.close()
            time.sleep(1.0)
            self.assertFalse(os.path.exists("/proc/%d/cmdline" % pid) and
                             open("/proc/%d/stat" % pid).read().split()[2] not in ("Z", "X"),
                             "the atlas child ends with its session")
        finally:
            gp.stop()
        with open(store, "rb") as f:
            self.assertEqual(f.read(), before, "the store is never written")
        self.assertEqual(os.stat(store).st_mtime_ns, mtime)
        self.assertFalse(os.path.exists(os.path.join(OPTS.scratch, "nope")))

    def test_atlas_not_configured(self):
        gp = GatewayProc()
        try:
            c = Client(gp, tab="t-noatlas")
            c.hello()
            fr, _ = c.run("atlas", query="info")
            self.assertEqual((fr.get("kind"), fr.get("event"), fr["error"]["kind"]), ("gw", "rejected", "no_atlas"))
            c.close()
        finally:
            gp.stop()

    def test_atlas_bad_config_refused(self):
        for extra in (["--atlas-store", os.path.join(OPTS.scratch, "missing.json")],
                      ["--atlas-index", "x.ocidx"],
                      ["--atlas-store", atlas_fixture(), "--atlas-expect-fnv", "zz"]):
            p = subprocess.run([sys.executable, os.path.join(HERE, "oc_gateway.py"), "--binary", OPTS.binary,
                                "--port", "0"] + extra, capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, 2, extra)
            self.assertNotIn("token=", p.stdout, extra)


def fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--scratch", default=os.path.join(os.path.expanduser("~"), ".cache", "oc-notebook-test"))
    ap.add_argument("--scripts", default=OPTS.scripts)
    args, rest = ap.parse_known_args()
    OPTS.binary = os.path.abspath(args.binary)
    OPTS.scratch = os.path.abspath(args.scratch)
    OPTS.scripts = args.scripts
    if os.path.commonpath([OPTS.scratch, ROOT]) == ROOT:
        sys.stderr.write("test_gateway: --scratch must lie outside the source tree\n")
        return 2
    os.makedirs(os.path.join(OPTS.scratch, "sessions"), exist_ok=True)
    prog = unittest.main(argv=[sys.argv[0]] + rest, exit=False)
    r = prog.result
    ok = r.wasSuccessful()
    print("GATEWAY_TEST ran=%d failures=%d errors=%d verdict=%s" % (r.testsRun, len(r.failures), len(r.errors), "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
