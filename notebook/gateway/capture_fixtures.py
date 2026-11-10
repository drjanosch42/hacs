#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
"""Capture the render fixtures of test_render.mjs from a real `--serve` session (slot-24 NB-RENDER R2).

    python3 notebook/gateway/capture_fixtures.py --binary <build>/hacs [--out <file>]

Runs ONE `hacs --serve` child (scrubbed environment, CPU only, 2 threads, RSS cap 2048 MiB), evaluates
the small cells of CELLS below (each well under a second and 1 GiB), and writes one line per value frame:
{"name", "expect", "env"[, "contains"]}. Default output: notebook/gateway/fixtures/envelopes.ndjson (frozen;
re-capture only when an envelope legitimately changes, and read the diff). Read-only: the binary only evaluates.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, source, expect, contains) -- the LAST value frame of the cell is kept.
CELLS = [
    ("int", "42", "tex", ["42"]),
    ("fp", "Fp(7, 3)", "tex", ["\\mathbb{F}_{7}"]),
    ("poly", "poly(691, [4, 346, 72, 1])", "tex", ["X^{3}"]),
    ("matrix", "matrix(7, [[1, 2], [3, 4]])", "tex", ["pmatrix"]),
    ("gf", "GF(7, 2)", "tex", ["\\mathbb{F}_{7^{2}}"]),
    ("int_list", "[0, 3, 0]", "tex", []),
    ("string", "\"hello\"", "text", []),
    ("cohomology_field", "M = sl2z_model(); H = cohomology(symmetric_power(M, 10, 691)); H.dims; H", "tex", ["\\dim"]),
    ("hecke_operator", "M = sl2z_model(); H = cohomology(symmetric_power(M, 10, 691)); H.dims; T = hecke(H, 2); T.A(1); T", "text", []),
    ("hecke_A1", "M = sl2z_model(); H = cohomology(symmetric_power(M, 10, 691)); T = hecke(H, 2); T.A(1)", "tex", ["pmatrix"]),
    ("charpoly", "M = sl2z_model(); H = cohomology(symmetric_power(M, 10, 691)); T = hecke(H, 2); charpoly(T.A(1))", "tex", ["\\mathbb{F}_{691}[X]"]),
    ("local_system_ring", "set coefficient_ring = \"Z/ell^2\"; M = sl2z_model(); L = symmetric_power(M, 10, 5); L", "tex", ["\\mathbb{Z}/5^{2}"]),
    ("cohomology_ring", "set coefficient_ring = \"Z/ell^2\"; M = sl2z_model(); H = cohomology(symmetric_power(M, 10, 5)); H.lengths; H", "tex", ["\\cong"]),
    ("ext_algebra", "R = ext_record(10, 691); T = local_algebra(R.ideal(0)); T.dim; T", "text", []),
    ("ext_ideal", "R = ext_record(10, 691); m = R.ideal(0); m.dim; m", "text", []),
    ("ext_loewy", "R = ext_record(10, 691); m = R.ideal(0); P = m.modules[0]; P.D; P", "text", []),
    ("ext_hilbert","R = ext_record(10, 691); T = local_algebra(R.ideal(0)); T.hilbert", "tex", []),
    ("level_n", "level_n(11, 0, 7)", "text", []),
    ("level_n_charpoly", "r = level_n(11, 0, 7); r", "text", []),
    ("forms_algebra", "forms_algebra(24, 7, [2], 6)", "text", []),
    ("hida_lambda", "hida_lambda(5, 12, 2, [2])", "text", []),
    ("descent", "set coefficient_ring = \"Z\"; D = descent(1, 10); D.status; D", "tex", ["\\Gamma_0(1)"]),
    ("descent_groups_inv", "set coefficient_ring = \"Z\"; D = descent(1, 10); D.groups_inv", "tex", ["\\oplus"]),
    ("descent_charpoly", "set coefficient_ring = \"Z\"; D = descent(1, 10); D.charpoly", "text", []),
    ("descent_declared", "set coefficient_ring = \"Z\"; D = descent(1, 10); D", "text", []),
    # (slot-24 NB-TOURFIX) char_Z(T_p) factored over Z: a grid; its `factors` column is hinted (kind factored), checked by the walk
    ("descent_charpoly_factored", "set coefficient_ring = \"Z\"; D = descent(1, 10); D.charpoly_factored", "text", []),
    # (slot-24 NB-RENDER R3) display(x): the frame of a forced handle; the hints of hecke_ring_algebra/v1, of
    # MaximalIdeal.character(p) and of the DicksonLES maps (checked at depth by the hinted walk); a shape-only Matrix.
    ("display_forced", "H = cohomology(symmetric_power(sl2z_model(), 10, 691)); display(H)", "tex", ["\\dim"]),
    ("display_ring_algebra", "R = hecke_ring_algebra(48, 7, 2); display(R)", "text", []),
    ("display_ideal", "R = ext_record(10, 691); display(R.ideal(0))", "text", []),
    ("display_dickson_les", "display(dickson_les(16, 7))", "text", []),
    ("matrix_shape_only", "zero(7, 30, 30)", "text", []),
]


def child_env(workdir):
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "HOME") if k in os.environ}
    env.update({"TMPDIR": workdir, "ENGINE_THREADS": "2", "ENGINE_CUDA_DISABLE": "1"})
    return env


def run_cell(binary, src, workdir):
    """One fresh session per cell (no state leaks between fixtures); returns the value frames."""
    p = subprocess.Popen([binary, "--serve", "--rss-cap-mib", "2048"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, env=child_env(workdir), cwd=workdir, text=True)
    hello = json.loads(p.stdout.readline())
    if hello.get("kind") != "hello":
        raise RuntimeError("no hello: %r" % hello)
    p.stdin.write(json.dumps({"op": "eval", "id": "c1", "src": src}) + "\n")
    p.stdin.flush()
    values, done = [], None
    for line in p.stdout:
        fr = json.loads(line)
        if fr.get("kind") == "value":
            values.append(fr["value"])
        if fr.get("kind") == "done":
            done = fr
            break
    p.stdin.close()
    p.wait(timeout=30)
    return values, done


# (NBCONFIRM S4 fix round 1, audit L4) members of a record's `json` string that vary from run to run (wall times, peak
# RSS): removed before the envelope is frozen, so a re-capture does not churn. The envelope keeps its shape; its `v` and
# `canonical` are re-serialised exactly as the binary does (compact JSON, then the string's JSON quoting) and its `hash`
# is recomputed as the binary hashes a String (FNV-1a-64 of the UTF-8 text), so the envelope stays self-consistent.
VOLATILE_JSON_KEYS = ("timings_s", "rss_peak_mib")


def fnv1a64(data):
    h = 0xcbf29ce484222325
    for c in data:
        h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def scrub(env):
    if isinstance(env, dict):
        v = env.get("v")
        members = v.get("members") if isinstance(v, dict) else None
        if isinstance(members, dict):
            js = members.get("json")
            if isinstance(js, dict) and js.get("t") == "String" and isinstance(js.get("v"), str):
                try:
                    obj = json.loads(js["v"])
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and any(k in obj for k in VOLATILE_JSON_KEYS):
                    for k in VOLATILE_JSON_KEYS:
                        obj.pop(k, None)
                    js["v"] = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
                    js["canonical"] = json.dumps(js["v"], ensure_ascii=False)
                    js["hash"] = "0x%016x" % fnv1a64(js["v"].encode("utf-8"))
            for m in members.values():
                scrub(m)
        if env.get("t") == "List" and isinstance(v, list):
            for x in v:
                scrub(x)
        elif env.get("t") == "Record" and isinstance(v, list):
            for pr in v:
                if isinstance(pr, list) and len(pr) == 2:
                    scrub(pr[1])
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures", "envelopes.ndjson"))
    a = ap.parse_args()
    out = []
    with tempfile.TemporaryDirectory(prefix="oc-fixtures-") as wd:
        for name, src, expect, contains in CELLS:
            values, done = run_cell(a.binary, src, wd)
            if not values or not done or done.get("status") != "ok":
                print("CAPTURE %s: no value (done=%r)" % (name, done), file=sys.stderr)
                return 1
            rec = {"name": name, "expect": expect, "env": scrub(values[-1])}
            if contains:
                rec["contains"] = contains
            out.append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# frozen oc-serve/1 envelopes for test_render.mjs; regenerate with capture_fixtures.py\n")
        f.write("# volatile members of a record's json string removed: %s\n" % ", ".join(VOLATILE_JSON_KEYS))
        f.write("\n".join(out) + "\n")
    print("CAPTURE fixtures=%d out=%s" % (len(out), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
