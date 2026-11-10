#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""LV-2 Part C (slot 22): generate the ell = 3 cell modules (even n; odd n is the theorem V3_dims_odd).

Per even n four ranks over F_3: s (stacked [1 - rho(S) ; 1 - rho(U)]), p (1 + rho(S)), r (1 + rho(U) + rho(U^2)),
u (1 - rho(U)); dim H^0 = d - s, dim H^1 = 2d - s - p - r, dim H^{q+2} = d - r - u (GL2.L23.TreeF3.V3_dims_of_ranks).

Expected values: an INDEPENDENT reference (rho_n(A) mod 3 from the full double sum of [C] eq:rhondef, U^2 = U.U asserted,
Gaussian elimination mod 3; r_S+ + r_S- = d asserted; for n > 64 the second reference ranks_fast, equal to it on every
n <= 64), checked against the closed forms of [R] (h0 = #{4a + 6b = n},
h1 = 2 dim M_{n+2} - 1 + h0 + [n = 0 mod 6], h^{q>=2} = [n = 0, 4 mod 6]) and, with --census, against census rows (ell = 3,
status ok, formal 0).  A disagreement aborts generation.  The kernel recomputes everything from n and refuses a wrong value.

  gen_lv3_cells.py [--large 100,128,...] [--ref-cache FILE] [--census GLOB ...] [--manifest PATH]
  gen_lv3_cells.py --range-max 700 [--budget-s 120] [--max-cells 30] ...   (also: every even n in (64, N] not in --large,
      packed into cells/L23/LV3R<lo>_<hi>.lean by the time model 1 s + 20.3 s (n/400)^2 and at most --max-cells cells,
      cell theorems in cells/L23/LV3RDims<b>.lean, one per block of 100 n; manifest lv3r_cells_manifest.tsv)

writes GL2/L23/CellsF3.lean + CellsF3Dims.lean (every even n <= 64, default target GL2), cells/L23/LV3K<n>.lean +
cells/L23/LV3Dims.lean (the --large cells, lib L23Cells) and the manifest (docs/implementation/slot22/lv/lv3_cells_manifest.tsv).
"""
import argparse
import csv
import glob
import sys
from math import comb
from pathlib import Path

from gen_lv1_cells import HDR

P = 3
SMALL = list(range(0, 65, 2))
NEG = 46


def rho(n, a, b, c, d):
    M = [[0] * (n + 1) for _ in range(n + 1)]
    for j in range(n + 1):
        for s in range(n - j + 1):
            for r in range(j + 1):
                M[s + r][j] = (M[s + r][j] + comb(n - j, s) * pow(a, n - j - s, P) * pow(c, s, P) * comb(j, r)
                               * pow(b, j - r, P) * pow(d, r, P)) % P
    return M


def rank(M):
    M = [row[:] for row in M]
    r = 0
    rows, cols = len(M), (len(M[0]) if M else 0)
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], P - 2, P)
        M[r] = [x * inv % P for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(x - f * y) % P for x, y in zip(M[i], M[r])]
        r += 1
    return r


def ranks(n):
    d = n + 1
    S, U, U2 = rho(n, 0, P - 1, 1, 0), rho(n, 0, P - 1, 1, 1), rho(n, P - 1, P - 1, 1, 0)
    assert [[sum(U[i][k] * U[k][j] for k in range(d)) % P for j in range(d)] for i in range(d)] == U2
    I = [[int(i == j) for j in range(d)] for i in range(d)]

    def lin(A, B, sa, sb):
        return [[(sa * x + sb * y) % P for x, y in zip(ra, rb)] for ra, rb in zip(A, B)]
    mS, pS, mU = lin(I, S, 1, -1), lin(I, S, 1, 1), lin(I, U, 1, -1)
    n3 = lin(lin(I, U, 1, 1), U2, 1, 1)
    out = dict(s=rank(mS + mU), p=rank(pS), m=rank(mS), r=rank(n3), u=rank(mU))
    assert out['p'] + out['m'] == d
    return out


def ranks_fast(n):
    """Second reference: column j of rho_n(A) = coefficients of (a + cY)^(n-j) (b + dY)^j mod 3 (def:rhon as a polynomial
    product), numpy elimination mod 3."""
    import numpy as np

    def powers(a, c):
        out = [np.array([1], dtype=np.int64)]
        for _ in range(n):
            out.append(np.convolve(out[-1], np.array([a % P, c % P], dtype=np.int64)) % P)
        return out

    def rho_f(a, b, c, d):
        A, B = powers(a, c), powers(b, d)
        M = np.zeros((n + 1, n + 1), dtype=np.int64)
        for j in range(n + 1):
            col = np.convolve(A[n - j], B[j]) % P
            M[:len(col), j] = col
        return M

    def rk(M):
        M = M.copy() % P
        r = 0
        rows, cols = M.shape
        for c in range(cols):
            nz = np.nonzero(M[r:, c])[0]
            if len(nz) == 0:
                continue
            q = r + nz[0]
            M[[r, q]] = M[[q, r]]
            M[r] = (M[r] * (1 if M[r, c] == 1 else 2)) % P
            f = M[:, c].copy()
            f[r] = 0
            M = (M - np.outer(f, M[r])) % P
            r += 1
            if r == rows:
                break
        return r
    S, U, U2 = rho_f(0, -1, 1, 0), rho_f(0, -1, 1, 1), rho_f(-1, -1, 1, 0)
    assert ((U @ U) % P == U2).all()
    I = np.eye(n + 1, dtype=np.int64)
    out = dict(s=rk(np.vstack([(I - S) % P, (I - U) % P])), p=rk((I + S) % P), m=rk((I - S) % P),
               r=rk((I + U + U2) % P), u=rk((I - U) % P))
    assert out['p'] + out['m'] == n + 1
    return out


def dimM(k):
    return 0 if k < 0 or k % 2 else k // 12 + (0 if k % 12 == 2 else 1)


def closed3(n):
    if n % 2:
        return (0, 0, 0, 0)
    h0 = sum(1 for a in range(n // 4 + 1) if (n - 4 * a) % 6 == 0)
    h2 = int(n % 6 in (0, 4))
    return (h0, 2 * dimM(n + 2) - 1 + h0 + (n % 6 == 0), h2, h2)


def cell(n, cache):
    if n in cache:
        R = cache[n]
    elif n > 64:
        R = ranks_fast(n)
        cache[n] = R
    else:
        R = ranks(n)
        assert R == ranks_fast(n)
        cache[n] = R
    d = n + 1
    h = (d - R['s'], 2 * d - R['s'] - R['p'] - R['r'], d - R['r'] - R['u'])
    return dict(n=n, d=d, h0=h[0], h1=h[1], h2=h[2], **{k: R[k] for k in 'spru'})


def kernel_facts(c):
    n = c['n']
    return (f"theorem rk3S_{n} : rankF3 (colsSig3 {n}) = {c['s']} := by decide +kernel\n"
            f"theorem rk3P_{n} : rankF3 (colsPS {n}) = {c['p']} := by decide +kernel\n"
            f"theorem rk3N_{n} : rankF3 (colsN3 {n}) = {c['r']} := by decide +kernel\n"
            f"theorem rk3U_{n} : rankF3 (colsMU {n}) = {c['u']} := by decide +kernel\n")


def dims_fact(c):
    n = c['n']
    return (f"/-- `n = {n}`: `dim H⁰ = {c['h0']}`, `dim H¹ = {c['h1']}`, `dim H^(q+2) = {c['h2']}`. -/\n"
            f"theorem dims3_{n} : finrank (ZMod 3) (groupCohomology (V3 {n}) 0) = {c['h0']} ∧\n"
            f"    finrank (ZMod 3) (groupCohomology (V3 {n}) 1) = {c['h1']} ∧\n"
            f"    ∀ q, finrank (ZMod 3) (groupCohomology (V3 {n}) (q + 2)) = {c['h2']} :=\n"
            f"  V3_dims_of_ranks {n} {c['s']} {c['p']} {c['r']} {c['u']} {c['h0']} {c['h1']} {c['h2']} rfl rk3S_{n} rk3P_{n}\n"
            f"    rk3N_{n} rk3U_{n} (by norm_num) (by norm_num) (by norm_num)\n")


def kernel_module(cells, doc, neg=None):
    out = HDR + "import GL2.L23.RankF3Exec\n\n/-!\n" + doc + "\n-/\n\nnamespace GL2.L23.Cells\n\nopen GL2.L23.RankF3\n\n"
    for c in cells:
        out += kernel_facts(c) + "\n"
    if neg:
        out += ("/-! Negative control: the kernel refuses a rank one too small (the refutation path works). -/\n"
                f"theorem rk3N_{neg['n']}_ne : rankF3 (colsN3 {neg['n']}) ≠ {neg['r'] - 1} := by decide +kernel\n\n")
    return out + "end GL2.L23.Cells\n"


def dims_module(cells, imports, doc):
    out = HDR + "".join(f"import {m}\n" for m in imports) + "import GL2.L23.TreeF3V3\n\n/-!\n" + doc + "\n-/\n\n"
    out += "namespace GL2.L23.Cells\n\nopen GL2.L23.RankF3 GL2.L23.TreeF3 Module\n\n"
    for c in cells:
        out += dims_fact(c) + "\n"
    return out + "end GL2.L23.Cells\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--large', default='100,128,200,256,300,400,500,600,700')
    ap.add_argument('--ref-cache', default=None)
    ap.add_argument('--census', action='append', default=[])
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--range-max', type=int, default=0)
    ap.add_argument('--budget-s', type=float, default=120.0)
    ap.add_argument('--max-cells', type=int, default=30)
    args = ap.parse_args()
    lean = Path(__file__).resolve().parent.parent
    repo = lean.parent
    manifest = Path(args.manifest) if args.manifest else repo / 'docs/implementation/slot22/lv/lv3_cells_manifest.tsv'
    large = [int(x) for x in args.large.split(',') if x]
    cache = {}
    if args.ref_cache and Path(args.ref_cache).exists():
        for L in open(args.ref_cache):
            f = L.split()
            if len(f) == 6:
                cache[int(f[0])] = dict(s=int(f[1]), p=int(f[2]), m=int(f[3]), r=int(f[4]), u=int(f[5]))
    census = {}
    for g in args.census:
        for f in sorted(glob.glob(g)):
            with open(f) as fh:
                for r in csv.DictReader(fh, delimiter='\t'):
                    if r.get('ell') == '3' and r['status'] == 'ok' and r['formal'] == '0':
                        census.setdefault(int(r['n']), []).append(r)
    rng = [n for n in range(66, args.range_max + 1, 2) if n not in large]
    cells, bad = [], 0
    for n in SMALL + large + rng:
        c = cell(n, cache)
        if (c['h0'], c['h1'], c['h2'], c['h2']) != closed3(n):
            print(f'DISAGREE n={n} reference={(c["h0"], c["h1"], c["h2"])} closed={closed3(n)}')
            bad += 1
        c['census'] = '-'
        for r in census.get(n, []):
            h = tuple(int(r[k]) for k in ('h0', 'h1', 'h2', 'h3'))
            ok = h == (c['h0'], c['h1'], c['h2'], c['h2'])
            c['census'] = f"{r['model']}:{'match' if ok else 'MISMATCH'}"
            bad += not ok
        cells.append(c)
    if args.ref_cache:
        with open(args.ref_cache, 'w') as fh:
            for n in sorted(cache):
                R = cache[n]
                fh.write(f"{n} {R['s']} {R['p']} {R['m']} {R['r']} {R['u']}\n")
    if bad:
        print(f'GEN_LV3 refused: {bad} disagreement(s); nothing written')
        return 1
    sm = [c for c in cells if c['n'] in SMALL]
    lg = [c for c in cells if c['n'] in large]
    rg = [c for c in cells if c['n'] in rng]
    neg = next(c for c in sm if c['n'] == NEG)
    doc = ("# LV-2 ℓ = 3 cells (kernel facts, Mathlib-free): every even `n ≤ 64`\n\n"
           "Generated by `lean/tools/gen_lv3_cells.py` (do not edit).  For each even `n` the four ranks over `F₃` of\n"
           "`[1 − ρ_n(S) ; 1 − ρ_n(U)]`, `1 + ρ_n(S)`, `1 + ρ_n(U) + ρ_n(U²)`, `1 − ρ_n(U)` (two bit-planes per row), by\n"
           "`decide +kernel` (no `native_decide`, no literal matrix data).  `CellsF3Dims.lean` turns them into\n"
           "`dim H^q(SL₂(ℤ); Sym^n F₃²)`; odd `n` is the theorem `GL2.L23.TreeF3.V3_dims_odd`.")
    (lean / 'GL2/L23/CellsF3.lean').write_text(kernel_module(sm, doc, neg))
    doc = ("# LV-2 ℓ = 3 cells: `dim_{F₃} H^q(SL₂(ℤ); Sym^n F₃²)`, every `q`, every even `n ≤ 64`\n\n"
           "Generated by `lean/tools/gen_lv3_cells.py` (do not edit).  Each `dims3_n` is `V3_dims_of_ranks` applied to the\n"
           "four kernel facts of `CellsF3.lean`; the values agree with the closed forms of [R] and an independent Python\n"
           "reference (and with the C++ `C_I` records for n ≤ 60).")
    (lean / 'GL2/L23/CellsF3Dims.lean').write_text(dims_module(sm, ['GL2.L23.CellsF3'], doc))
    for c in lg:
        doc = (f"# LV-2 ℓ = 3 large cell n = {c['n']} (kernel facts, Mathlib-free; lib `L23Cells`, not a default target)\n\n"
               "Generated by `lean/tools/gen_lv3_cells.py` (do not edit).")
        (lean / f"cells/L23/LV3K{c['n']}.lean").write_text(kernel_module([c], doc))
    if lg:
        doc = ("# LV-2 ℓ = 3 large cells: `dim_{F₃} H^q(SL₂(ℤ); Sym^n F₃²)` for n = " + ", ".join(str(c['n']) for c in lg) +
               " (lib `L23Cells`, not a default target)\n\nGenerated by `lean/tools/gen_lv3_cells.py` (do not edit).")
        (lean / 'cells/L23/LV3Dims.lean').write_text(dims_module(lg, [f"L23.LV3K{c['n']}" for c in lg], doc))
    if rg:
        mods, cur, t = [], [], 0.0
        for c in rg:
            k = 1.0 + 20.3 * (c['n'] / 400) ** 2
            if cur and (t + k > args.budget_s or len(cur) >= args.max_cells):
                mods.append(cur)
                cur, t = [], 0.0
            cur.append(c)
            t += k
        if cur:
            mods.append(cur)
        keep = set()
        for m in mods:
            name = f"LV3R{m[0]['n']:03d}_{m[-1]['n']:03d}"
            keep.add(name + '.lean')
            for c in m:
                c['kmod'], c['dmod'] = 'L23.' + name, f"L23.LV3RDims{c['n'] // 100}"
            doc = (f"# LV-2 ℓ = 3 cells, even n = {m[0]['n']} … {m[-1]['n']} (kernel facts, Mathlib-free; lib `L23Cells`)\n\n"
                   "Generated by `lean/tools/gen_lv3_cells.py --range-max` (do not edit).")
            (lean / f'cells/L23/{name}.lean').write_text(kernel_module(m, doc))
        for f in (lean / 'cells/L23').glob('LV3R[0-9]*.lean'):
            if f.name not in keep:
                f.unlink()
        for b in sorted({c['n'] // 100 for c in rg}):
            cs = [c for c in rg if c['n'] // 100 == b]
            doc = (f"# LV-2 ℓ = 3 cells: `dim_{{F₃}} H^q(SL₂(ℤ); Sym^n F₃²)` for the even n in [{100 * b}, {100 * b + 99}] of the "
                   "range run (lib `L23Cells`)\n\nGenerated by `lean/tools/gen_lv3_cells.py --range-max` (do not edit).")
            (lean / f'cells/L23/LV3RDims{b}.lean').write_text(dims_module(cs, sorted({c['kmod'] for c in cs}), doc))
        with open(manifest.parent / 'lv3r_cells_manifest.tsv', 'w') as fh:
            fh.write('# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\n# SPDX-License-Identifier: CC-BY-4.0\n')
            fh.write('ell\tn\ts\tp\tr\tu\th0\th1\th2\th3\tclosed\tcensus\tkernel_module\tdims_module\n')
            for c in rg:
                fh.write(f"3\t{c['n']}\t{c['s']}\t{c['p']}\t{c['r']}\t{c['u']}\t{c['h0']}\t{c['h1']}\t{c['h2']}\t{c['h2']}\t"
                         f"match\t{c['census']}\t{c['kmod']}\t{c['dmod']}\n")
        print(f'GEN_LV3 range cells={len(rg)} kernel_modules={len(mods)}')
    with open(manifest, 'w') as fh:
        fh.write('# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\n# SPDX-License-Identifier: CC-BY-4.0\n')
        fh.write('ell\tn\ts\tp\tr\tu\th0\th1\th2\th3\tclosed\tcensus\tkernel_module\tdims_module\n')
        for c in sm + lg:
            km, dm = (('GL2.L23.CellsF3', 'GL2.L23.CellsF3Dims') if c['n'] in SMALL
                      else (f"L23.LV3K{c['n']}", 'L23.LV3Dims'))
            fh.write(f"3\t{c['n']}\t{c['s']}\t{c['p']}\t{c['r']}\t{c['u']}\t{c['h0']}\t{c['h1']}\t{c['h2']}\t{c['h2']}\t"
                     f"match\t{c['census']}\t{km}\t{dm}\n")
    print(f'GEN_LV3 cells={len(cells)} small={len(sm)} large={len(lg)} census_compared='
          f'{sum(c["census"] != "-" for c in cells)} closed_mismatch=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
