#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""LV-2 (slot 22): generate the ell = 2 cell modules for every n <= N not certified by LV-1 (lib L23Cells).

Expected values, three sources (LV1_audit.md §5.1(3), MINOR-3):
  * the census summary.tsv rows (ell = 2, keyed on (n, model, formal, status); only status = ok, formal = 0 feed a value),
  * the independent Python reference of gen_lv1_cells.py (rho_n mod 2 via math.comb, Python elimination),
  * the closed forms: odd n [C] prop:oddnltwo (h0 = floor((floor(n/3)+1)/2), h^q = (n+1)/2), even n [R] Thm A (Dickson count
    #{3a+2b = n}, h^q = floor(n/2)+1).
The value written into the module is the census value when a computed record exists, else the reference value; the kernel
refuses a wrong one (the module then fails to build, and the census column reads `refused`).  A disagreement between the
three sources aborts generation (exit 1): nothing is written.

  gen_lv2_cells.py --census 'GLOB' [--census 'GLOB' ...] [--max-n 700] [--budget-s 120] [--manifest PATH]

writes cells/L23/LV2K<lo>_<hi>.lean (kernel facts, Mathlib-free; at most --max-cells cells and --budget-s modelled kernel
seconds per module: the peak RSS of a module grows with its number of cells, LV2_report.md), cells/L23/LV2Dims<b>.lean (the cell theorems, one per block
of 100 n) and the manifest TSV (default docs/implementation/slot22/lv/lv2_cells_manifest.tsv).
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

from math import comb

from gen_lv1_cells import HDR, SMALL, MEDIUM, LARGE, rank, rho

DONE_GL2 = set(SMALL) | set(MEDIUM)          # GL2.L23.CellsF2 / CellsF2Dims
DONE_LV1 = {n for n in LARGE}                 # L23.LV1K<n> / L23.LV1Dims


_PAR = []


def _par(m, t):
    """C(m, t) mod 2 from math.comb, tabulated once (rows grow on demand)."""
    while len(_PAR) <= m:
        k = len(_PAR)
        _PAR.append([comb(k, i) % 2 for i in range(k + 1)])
    return _PAR[m][t] if 0 <= t <= m else 0


def cell(n):
    """The LV-1 reference (gen_lv1_cells.cell), with the binomial parities tabulated: rho_n(S), rho_n(U), rho_n(U^2)
    mod 2 entrywise from [C] eq:rhondef's closed forms, then a Python elimination."""
    d = n + 1
    S = [1 << (n - j) for j in range(d)]
    U = [sum(_par(j, t - (n - j)) << t for t in range(n - j, n + 1)) for j in range(d)]
    U2 = [sum(_par(n - j, t) << t for t in range(n - j + 1)) for j in range(d)]
    oneS = [(1 << j) ^ S[j] for j in range(d)]
    oneU = [(1 << j) ^ U[j] for j in range(d)]
    n3 = [(1 << j) ^ U[j] ^ U2[j] for j in range(d)]
    a = rank(oneS + n3)
    s = rank([oneS[j] ^ (oneU[j] << d) for j in range(d)])
    return dict(n=n, d=d, a=a, s=s, h0=d - s, h1=2 * d - a - s)


for _n in range(65):  # the tabulated reference is the LV-1 reference
    assert cell(_n)['a'] == __import__('gen_lv1_cells').cell(_n)['a']
    assert cell(_n)['s'] == __import__('gen_lv1_cells').cell(_n)['s']
    assert [1 << (_n - j) for j in range(_n + 1)] == rho(_n, 'S')


def closed(n):
    if n % 2:
        return (n // 3 + 1) // 2, (n + 1) // 2
    return sum(1 for a in range(n // 3 + 1) if (n - 3 * a) % 2 == 0), n // 2 + 1


def kernel_s(n):
    """LV-1 measured model: 108 s at n = 1000 for both ranks, exponent 2.3; plus 0.05 s per cell."""
    return 108.0 * (n / 1000.0) ** 2.3 + 0.05


def read_census(globs):
    rows = {}
    for g in globs:
        for f in sorted(glob.glob(g)):
            with open(f) as fh:
                for r in csv.DictReader(fh, delimiter='\t'):
                    if r.get('ell') != '2':
                        continue
                    rows.setdefault(int(r['n']), []).append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--census', action='append', default=[])
    ap.add_argument('--max-n', type=int, default=700)
    ap.add_argument('--budget-s', type=float, default=120.0)
    ap.add_argument('--max-cells', type=int, default=30)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    lean = Path(__file__).resolve().parent.parent
    repo = lean.parent
    manifest = Path(args.manifest) if args.manifest else repo / 'docs/implementation/slot22/lv/lv2_cells_manifest.tsv'
    census = read_census(args.census)

    todo = [n for n in range(args.max_n + 1) if n not in DONE_GL2 and n not in DONE_LV1]
    out, bad = [], 0
    for n in todo:
        c = cell(n)
        cf = closed(n)
        d = n + 1
        src, crow = 'reference', None
        ok_rows = [r for r in census.get(n, []) if r['status'] == 'ok' and r['formal'] == '0']
        for r in ok_rows:
            h = tuple(int(r[k]) for k in ('h0', 'h1', 'h2', 'h3'))
            if h != (c['h0'], c['h1'], c['h1'], c['h1']):
                print(f'DISAGREE n={n} census={h} model={r["model"]} reference={(c["h0"], c["h1"])}')
                bad += 1
            crow = r
        if crow is not None:
            src = 'census'
            h0, h1 = int(crow['h0']), int(crow['h1'])
        else:
            h0, h1 = c['h0'], c['h1']
        if (c['h0'], c['h1']) != cf:
            print(f'DISAGREE n={n} reference={(c["h0"], c["h1"])} closed={cf}')
            bad += 1
        a, s = 2 * d - h1 - (d - h0), d - h0
        out.append(dict(n=n, a=a, s=s, h0=h0, h1=h1, src=src,
                        model=crow['model'] if crow else '-', formal=crow['formal'] if crow else '-', formal_row=n % 2,
                        status=crow['status'] if crow else '-', fnv=crow['record_fnv1a64'] if crow else '-',
                        ref='ok' if (c['a'], c['s']) == (a, s) else 'DIFF', closed='ok' if cf == (h0, h1) else 'DIFF'))
    if bad:
        print(f'GEN_LV2 refused: {bad} disagreement(s); nothing written')
        return 1

    # pack consecutive n into kernel modules by the time model
    mods, cur, t = [], [], 0.0
    for c in out:
        k = kernel_s(c['n'])
        if cur and t + k > args.budget_s:
            mods.append(cur)
            cur, t = [], 0.0
        cur.append(c)
        t += k
    if cur:
        mods.append(cur)
    # then split a module with more than --max-cells cells into equal consecutive chunks
    split = []
    for m in mods:
        k = -(-len(m) // args.max_cells)
        q, r = divmod(len(m), k)
        i = 0
        for c in range(k):
            sz = q + (1 if c < r else 0)
            split.append(m[i:i + sz])
            i += sz
    mods = split
    for m in mods:
        name = f"LV2K{m[0]['n']:03d}_{m[-1]['n']:03d}"
        for c in m:
            c['kmod'] = 'L23.' + name
            c['dmod'] = f"L23.LV2Dims{c['n'] // 100}"
    blocks = sorted({c['n'] // 100 for c in out})
    print(f'GEN_LV2 cells={len(out)} kernel_modules={len(mods)} dims_modules={len(blocks)} '
          f'census_fed={sum(c["src"] == "census" for c in out)} reference_fed={sum(c["src"] == "reference" for c in out)} '
          f'model_kernel_s={sum(kernel_s(c["n"]) for c in out):.0f} max_module_s={max(sum(kernel_s(c["n"]) for c in m) for m in mods):.0f}')
    if args.dry_run:
        for m in mods:
            print(f"PLAN {m[0]['kmod']} cells={len(m)} n={m[0]['n']}..{m[-1]['n']} "
                  f"model_s={sum(kernel_s(c['n']) for c in m):.0f}")
        return 0

    (lean / 'cells/L23').mkdir(parents=True, exist_ok=True)
    keep = {m[0]['kmod'].split('.')[1] + '.lean' for m in mods}
    for f in (lean / 'cells/L23').glob('LV2K*.lean'):
        if f.name not in keep:
            f.unlink()
    for m in mods:
        name = m[0]['kmod'].split('.')[1]
        body = (HDR + "import GL2.L23.TreeF2Exec\n\n/-!\n"
                f"# LV-2 cells n = {m[0]['n']} … {m[-1]['n']} (kernel facts, Mathlib-free; lib `L23Cells`, not a default target)\n\n"
                "Generated by `lean/tools/gen_lv2_cells.py` (do not edit).  Per cell: the two ranks over `F₂` by `decide +kernel`\n"
                "(no `native_decide`, no literal matrix data).  Value sources (census row `model/formal/status/record_fnv1a64`, or\n"
                "`reference` = the Python elimination; cross-checks `ref`, `closed`):\n\n")
        for c in m:
            prov = f"{c['model']}, {c['formal']}, {c['status']}, {c['fnv']}" if c['src'] == 'census' else 'no census record'
            body += (f"* n = {c['n']}: a = {c['a']}, s = {c['s']}; source {c['src']} ({prov}); "
                     f"ref {c['ref']}, closed {c['closed']}\n")
        body += "-/\n\nnamespace GL2.L23.Cells\n\nopen GL2.L23.RankF2\n\n"
        for c in m:
            body += (f"theorem rkA_{c['n']} : rankF2 (colsA {c['n']}) = {c['a']} := by decide +kernel\n"
                     f"theorem rkS_{c['n']} : rankF2 (colsSig {c['n']}) = {c['s']} := by decide +kernel\n")
        body += "\nend GL2.L23.Cells\n"
        (lean / f'cells/L23/{name}.lean').write_text(body)
    for b in blocks:
        cs = [c for c in out if c['n'] // 100 == b]
        kms = sorted({c['kmod'] for c in cs})
        body = HDR + "".join(f"import {k}\n" for k in kms) + "import GL2.L23.TreeF2\n\n/-!\n"
        body += (f"# LV-2 cells: `dim_{{F₂}} H^q(SL₂(ℤ); Sym^n F₂²)`, every `q`, for the n in [{100 * b}, {100 * b + 99}] not in LV-1\n\n"
                 "Generated by `lean/tools/gen_lv2_cells.py` (do not edit).  Each `dims_n` is `V2_dims_of_ranks` (the width\n"
                 "bounds are theorems, `rowsLt_colsA` / `rowsLt_colsSig`) applied to the two kernel facts.\n-/\n\n")
        body += "namespace GL2.L23.Cells\n\nopen GL2.L23.RankF2 GL2.L23.TreeF2 Module\n\n"
        for c in cs:
            n = c['n']
            body += (f"/-- `n = {n}`: `dim H⁰ = {c['h0']}`, `dim H^q = {c['h1']}` (q ≥ 1). -/\n"
                     f"theorem dims_{n} : finrank (ZMod 2) (groupCohomology (V2 {n}) 0) = {c['h0']} ∧\n"
                     f"    ∀ q, finrank (ZMod 2) (groupCohomology (V2 {n}) (q + 1)) = {c['h1']} :=\n"
                     f"  V2_dims_of_ranks {n} {c['a']} {c['s']} {c['h0']} {c['h1']} rkA_{n} rkS_{n} (by norm_num) (by norm_num)\n\n")
        body += "end GL2.L23.Cells\n"
        (lean / f'cells/L23/LV2Dims{b}.lean').write_text(body)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, 'w') as fh:
        fh.write('# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\n# SPDX-License-Identifier: CC-BY-4.0\n')
        fh.write('ell\tn\ta\ts\th0\th1\tvalue_source\tcensus_model\tcensus_formal\tcensus_status\tcensus_fnv\tref\tclosed\t'
                 'formal_row\tkernel_module\tdims_module\n')
        for c in out:
            fh.write(f"2\t{c['n']}\t{c['a']}\t{c['s']}\t{c['h0']}\t{c['h1']}\t{c['src']}\t{c['model']}\t{c['formal']}\t"
                     f"{c['status']}\t{c['fnv']}\t{c['ref']}\t{c['closed']}\t{c['formal_row']}\t{c['kmod']}\t{c['dmod']}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
