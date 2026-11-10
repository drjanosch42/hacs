#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""Untrusted generator of the K10 / K10' certificates of the Mazur cells (slot 25, L-K10; 07-lean-plan.md §5.10, A7-M1).

Reads the H3 run's per-cell outputs `<out_dir>/cell_<N>_<ell>.out` (one `MAZUR_EIS_CELL mazur_eis/v1 ...` line each; the
`HeckeMazurEisValidation` of `Integral/HeckeMazurEis.cpp`, `slot25/H3_report.md` §2): N, ell, v_nu, K, the accepted q, e,
F (the Hensel factor in y = x - 1 - q, residues mod ell^K, lowest first) and c = c_q (exact, hex, lowest first).

It computes (all re-checked by the Lean kernel, `GL2/HeckeZ/Exec/K10.lean`): F in the variable x (a Taylor shift mod ell^K),
the cofactor H = c_q / F mod ell^K, and Merel's residue P. Before writing anything it runs the checker's conditions itself
and refuses a cell that fails. Only c_q (TB-chi: the record's c_q with chi_Z(T_q) = (x - 1 - q) c_q^2) enters as data the
kernel does not re-derive; F and H are certificates.

Writes the Mathlib-free data modules `MazurData<X>.lean` (<= 30 cells and <= ~160 kB each), `MerelData{A,B,C}.lean` (60 cells each), and
`manifest_lk10.tsv` (cell, out-file sha256, module).

Usage: hz_mazur_to_lean.py <h3_out_dir> <lean_cells_dir> <manifest_out>
"""
import hashlib, math, os, re, sys

CELL_RE = re.compile(r"MAZUR_EIS_CELL mazur_eis/v1 (.*)$")
MAXCELLS = 30
MAXBYTES = 160_000


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse(line):
    d = {}
    for m in re.finditer(r"(\w+)=(\[[^\]]*\]|\S+)", line):
        d[m.group(1)] = m.group(2)
    return d


def ilist(s, base):
    s = s.strip("[]")
    return [int(t, base) for t in s.split(",")] if s else []


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, int(math.isqrt(n)) + 1))


def pmul(a, b):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r


def peval(p, t):
    r = 0
    for c in reversed(p):
        r = r * t + c
    return r


def plin(a, e):
    r = [1]
    for _ in range(e):
        r = pmul([-a, 1], r)
    return r


def peqmod(m, a, b):
    n = max(len(a), len(b))
    return all(((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % m == 0 for i in range(n))


def shift(fy, a, m):
    """F_x(x) = F_y(x - a) mod m."""
    r = [0]
    for c in reversed(fy):
        # r = r * (x - a) + c
        r = [(x % m) for x in [0] + r]
        for i in range(len(r) - 1):
            r[i] = (r[i] - a * r[i + 1]) % m
        r[0] = (r[0] + c) % m
    while len(r) > 1 and r[-1] == 0:
        r.pop()
    return r


def divmonic(h, f, m):
    """(quotient, remainder) of h by monic f, mod m."""
    h = [x % m for x in h]
    df = len(f) - 1
    if len(h) - 1 < df:
        return [], h
    q = [0] * (len(h) - df)
    for k in range(len(h) - 1 - df, -1, -1):
        coef = h[k + df] % m
        q[k] = coef
        for i, fi in enumerate(f):
            h[k + i] = (h[k + i] - coef * fi) % m
    return q, h[:df]


def merel_p(N):
    P = 1 % N
    for i in range(1, (N - 1) // 2 + 1):
        P = P * pow(i, i, N) % N
    return P


def nu_nat(N):
    return (N - 1) // math.gcd(N - 1, 12)


def good(N, ell, q):
    return q % ell != 1 and pow(q, (N - 1) // ell, N) != 1


def build(rec):
    N, ell, q, K, v, e = (int(rec[k]) for k in ("N", "ell", "q", "K", "v_nu", "e"))
    m = ell ** K
    fy = ilist(rec["F"], 10)
    h = ilist(rec["c"], 16)
    a = 1 + q
    assert len(fy) == e + 1 and fy[-1] == 1, (N, ell, "F shape")
    assert h and h[-1] == 1 and len(h) - 1 == int(rec["g"]), (N, ell, "c shape")
    fx = shift(fy, a, m)
    assert len(fx) == e + 1 and fx[-1] == 1
    H, rem = divmonic(h, fx, m)
    assert all(x % m == 0 for x in rem), (N, ell, "F does not divide c_q mod ell^K")
    # the checker's conditions (Exec/K10.lean k10pCheck)
    nu = nu_nat(N)
    assert is_prime(N) and is_prime(ell) and is_prime(q) and N >= 5 and ell >= 5 and q != N
    assert (N - 1) % ell == 0 and nu % ell ** v == 0 and nu % ell ** (v + 1) != 0 and 1 <= v < K
    assert good(N, ell, q), (N, ell, q, "not good")
    assert peqmod(ell, fx, plin(a, e))
    assert peqmod(m, h, pmul(fx, H))
    assert peval(H, a) % ell != 0
    fa = peval(fx, a)
    assert fa % ell ** v == 0 and fa % ell ** (v + 1) != 0, (N, ell, "valuation check")
    P = merel_p(N)
    merel = pow(P, (N - 1) // ell, N) == 1
    assert merel == (e >= 2) == (rec["merel"] == "1"), (N, ell, "Merel")
    return dict(N=N, ell=ell, q=q, K=K, v=v, e=e, h=h, flo=fx[:-1], H=H, P=P)


def lint(x):
    return f"({x})" if x < 0 else str(x)


def llist(xs):
    return "[" + ", ".join(lint(x) for x in xs) + "]"


def cname(c):
    return f"c{c['N']}_{c['ell']}"


def cert_text(c):
    return (f"set_option maxRecDepth 8192 in\n"
            f"/-- The cell `({c['N']}, {c['ell']})`, `q = {c['q']}`, `e = {c['e']}`. -/\n"
            f"def {cname(c)} : K10pCert where\n  N := {c['N']}\n  ell := {c['ell']}\n  q := {c['q']}\n  K := {c['K']}\n  v := {c['v']}\n"
            f"  h := {llist(c['h'])}\n  flo := {llist(c['flo'])}\n  H := {llist(c['H'])}\n")


HEADER = """/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K10
"""


def write_data(path, tag, cells, provenance):
    body = "\n".join(cert_text(c) for c in cells)
    lst = ", ".join(cname(c) for c in cells)
    names = ", ".join(f"({c['N']},{c['ell']})" for c in cells)
    with open(path, "w") as f:
        f.write(HEADER)
        f.write(f"""
/-!
# L-K10: K10′ certificates of the Mazur cells, part {tag} (generated; do not edit)

Generated by `lean/tools/hz_mazur_to_lean.py` from the H3 run (`slot25/H3_report.md` §2, `HeckeMazurEisValidation`, binary sha256
`{provenance}`); the per-cell out-file sha256s are in `docs/implementation/slot25/lk10-data/manifest_lk10.tsv`.
`h` is the record's `c_q` verbatim (TB-χ); `flo`, `H` are certificates the kernel checks. Cells `(N, ℓ)`: {names}.
-/

namespace GL2.HeckeZ.Cells.MazurData

open GL2.HeckeZ.Exec.K10

-- Long coefficient-list literals (up to 163 entries) nest deeper than the default elaborator recursion limit.
{body}
/-- The K10′ certificates, part {tag} ({len(cells)} cells). -/
def cells{tag} : List K10pCert := [{lst}]

/-- Every certificate of part {tag} passes K10′ (goodness, Hensel factorisation, valuation check). -/
theorem cells{tag}_check : cells{tag}.all k10pCheck = true := by decide +kernel

end GL2.HeckeZ.Cells.MazurData
""")


def write_merel(path, tag, cells, provenance):
    body = ",\n".join(f"  {{ N := {c['N']}, ell := {c['ell']}, P := {c['P']}, e := {c['e']} }}" for c in cells)
    with open(path, "w") as f:
        f.write(HEADER)
        f.write(f"""
/-!
# L-K10: K10 (Merel's criterion, [C] LN5), part {tag}: {len(cells)} Mazur cells (generated; do not edit)

Generated by `lean/tools/hz_mazur_to_lean.py` (H3 binary sha256 `{provenance}`). `P = ∏_{{i ≤ (N−1)/2}} i^i mod N` is recomputed
by the kernel; `e` is the cell's `deg F` (the K10′ certificate's `flo.length`). An `orc:` consistency check: no soundness theorem.
-/

namespace GL2.HeckeZ.Cells.MazurData

open GL2.HeckeZ.Exec.K10

/-- `(N, ℓ, P, e)` for every Mazur cell. -/
def merelCells{tag} : List K10Cert := [
{body}
]

/-- K10 agrees with `e` on every cell: `e ≥ 2` exactly when `P` is an `ℓ`-th power mod `N`. -/
theorem merelCells{tag}_check : merelCells{tag}.all merelCheck = true := by decide +kernel

end GL2.HeckeZ.Cells.MazurData
""")


def main():
    out_dir, cells_dir, manifest = sys.argv[1:4]
    prov_path = os.path.join(os.path.dirname(os.path.abspath(out_dir.rstrip("/"))), "bin.sha256")
    provenance = open(prov_path).read().split()[0] if os.path.exists(prov_path) else "unknown"
    files = sorted(os.listdir(out_dir), key=lambda s: tuple(int(x) for x in re.findall(r"\d+", s)))
    cells, rows = [], []
    for fn in files:
        p = os.path.join(out_dir, fn)
        recs = [CELL_RE.search(l) for l in open(p)]
        recs = [r for r in recs if r]
        assert len(recs) == 1, fn
        c = build(parse(recs[0].group(1)))
        c["file"], c["sha"] = fn, sha256(p)
        cells.append(c)
    # split by count and bytes
    parts, cur, size = [], [], 0
    for c in cells:
        t = len(cert_text(c))
        if cur and (len(cur) >= MAXCELLS or size + t > MAXBYTES):
            parts.append(cur); cur, size = [], 0
        cur.append(c); size += t
    parts.append(cur)
    tags = [chr(ord("A") + i) for i in range(len(parts))]
    for tag, part in zip(tags, parts):
        write_data(os.path.join(cells_dir, f"MazurData{tag}.lean"), tag, part, provenance)
        for c in part:
            rows.append((c["N"], c["ell"], c["q"], c["e"], c["v"], c["file"], c["sha"], f"MazurData{tag}"))
    for i, tag in enumerate("ABC"):
        write_merel(os.path.join(cells_dir, f"MerelData{tag}.lean"), tag, cells[60 * i:60 * (i + 1)], provenance)
    with open(manifest, "w") as f:
        f.write("# L-K10 manifest: H3 out files -> Lean data modules (hz_mazur_to_lean.py)\n")
        f.write(f"# h3_binary_sha256\t{provenance}\n")
        f.write("N\tell\tq\te\tv\tfile\tsha256\tmodule\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"HZ_MAZUR_GEN cells={len(cells)} parts={len(parts)} tags={''.join(tags)} merel_e2={sum(c['e'] >= 2 for c in cells)}")


if __name__ == "__main__":
    main()
