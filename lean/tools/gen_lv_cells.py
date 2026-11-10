#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""LV (slot 23): generate the per-cell modules of the O-6 bounds that remain after LG-3
(docs/implementation/slot23/LV_plan_and_predictions.md).

  --part h0     ell = 2, every n in [702, 998] (n = 701, 999, 1000 are LV-1 cells): ONE rank per cell, rankF2 (colsSig n),
                h0 = n + 1 - s (GL2.L23.TreeF2.V2_h0_of_rankS); h^{q+1} = floor(n/2) + 1 is LG-3's Split.V2_dims_all.
                cells/L23/LV4S<lo>_<hi>.lean (kernel, Mathlib-free) + LV4SDims<b>.lean (cell theorems dims_n).
  --part dims3  ell = 3, even n in [702, 1000] (odd n is the theorem V3_dims_odd): the four rankF3 of LV-2
                (gen_lv3_cells.cell, the numpy reference ranks_fast; checked against the [R] closed forms).
                cells/L23/LV4T<lo>_<hi>.lean + LV4TDims<b>.lean (dims3_n, V3_dims_of_ranks).
  --part smith  ell in {2, 3}, even n <= 150, q in {0, 1, 2}: the Smith signature sig K (cellPiv l K H n q) of d^q
                (GL2.L23.SmithPK; K = 10 at ell = 2, K = 6 at ell = 3; H minimal with l^K + R l^2K < 2^H).  n <= 30 go to
                GL2/L23/CellsSmith.lean + CellsSmithCard.lean (default target GL2), the rest to cells/L23/LV4M<l>_<lo>_<hi>.lean +
                LV4MCard<l>_<b>.lean (cell_card0/1/2).  Values: the step mirror of the kernel algorithm (lv_smith_ref.smithpk),
                cross-checked against the independent full-pivot reference (lv_smith_ref.smith_vals) for n <= --ref-max, and
                'certified' = every valuation < K - 1 and #valuations = rank over Q (rank mod two primes, numpy).
  --dry-run     print the batch plan (modules, cells, predicted kernel s and peak RSS per module) and write nothing.

Packing (rule 19, LV2_report.md sec. 2.3): a kernel module holds at most --max-cells cells and --budget-s modelled seconds;
its predicted peak is (largest single cell) + 0.075 GiB per further declaration; the generator refuses a module predicted
above --max-gib.  A disagreement between sources aborts generation: nothing is written.
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_lv1_cells import HDR  # noqa: E402

# ------------------------------------------------------------------ models (calibrated, see the plan doc sec. 4)


def model_h0(n):
    """rankF2F (colsSig n) (GL2.L23.RankFast, no Nat.log2): measured at n = 1000 in 21 s / 2.97 GiB (rankF2: 46 s / 5.34);
    LV-1's exponents 2.3 (time) and 1.5 (RSS), floor 2 GiB (LV-1: 2.1 GiB at n = 400)."""
    return 21.0 * (n / 1000) ** 2.3 + 0.05, max(2.0, 3.0 * (n / 1000) ** 1.5)


def model_dims3(n):
    """Four rankF3F (no Nat.log2): LV-2 measured the four rankF3 at n = 700 in 74.4 s / 4.84 GiB; rankF3F on colsSig3 700
    took 20.6 s / 2.74 GiB against 33 s / 3.86 GiB for rankF3 (x 0.62), so 46 s at 700; then measured the four rankF3F at
    n = 1000 in one file: 156 s / 4.98 GiB (box carrying the census), refit to 64 s at 700, exponent 2.5, RSS 3.0 (n/700)^1.5
    (5.12 GiB at 1000 predicted, 4.98 measured)."""
    return 64.0 * (n / 700) ** 2.5 + 0.05, max(2.0, 3.0 * (n / 700) ** 1.5)


def model_smith(n, q, ell):
    """Slot 23 measured d^2 at n = 150: 24 s / 4.56 GiB (ell = 2, K = 10), 21 s / 4.27 GiB (ell = 3, K = 6); n = 80: 5 s /
    1.38 GiB; n = 50: 2 s / 0.79 GiB.  d^0, d^1 have 2d columns: x 0.5 time, x 0.6 RSS above the floor (to be measured)."""
    t150 = 24.0 if ell == 2 else 21.0
    r150 = 4.56 if ell == 2 else 4.27
    f_t, f_r = (1.0, 1.0) if q == 2 else (0.5, 0.6)
    t = f_t * t150 * (n / 150) ** 2.5 + 0.05
    rss = 0.6 + f_r * (r150 - 0.6) * (n / 150) ** 2.3
    return t, rss


DIMS_THEOREM_GIB = 8.5  # a Mathlib cell-theorem module: LV-2 measured 7.48-7.53 GiB (8.05 with rebuilt dependencies)


def pack(cells, model, budget, max_cells, max_gib):
    mods, cur, t = [], [], 0.0
    for c in cells:
        k = model(c)[0]
        if cur and (t + k > budget or len(cur) >= max_cells):
            mods.append(cur)
            cur, t = [], 0.0
        cur.append(c)
        t += k
    if cur:
        mods.append(cur)
    plan = []
    for m in mods:
        t = sum(model(c)[0] for c in m)
        decls = sum(c.get('decls', 1) for c in m)
        rss = max(model(c)[1] for c in m) + 0.075 * (decls - 1)
        if rss > max_gib:
            raise SystemExit(f'GEN_LV refused: module {m[0]} predicted {rss:.2f} GiB > {max_gib}')
        plan.append((m, t, rss))
    return plan


def write_manifest(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\n# SPDX-License-Identifier: CC-BY-4.0\n')
        fh.write('\t'.join(header) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r[h]) for h in header) + '\n')


def print_plan(tag, plan):
    tot = sum(p[1] for p in plan)
    print(f'GEN_LV_PLAN part={tag} modules={len(plan)} cells={sum(len(p[0]) for p in plan)} kernel_s={tot:.0f} '
          f'max_module_s={max(p[1] for p in plan):.0f} max_module_gib={max(p[2] for p in plan):.2f}')
    for m, t, rss in plan:
        print(f"PLAN {m[0]['kmod']} cells={len(m)} n={m[0]['n']}..{m[-1]['n']} model_s={t:.0f} model_gib={rss:.2f}")


# ------------------------------------------------------------------ part h0 (ell = 2)


def closed2(n):
    if n % 2:
        return (n // 3 + 1) // 2
    return sum(1 for a in range(n // 3 + 1) if (n - 3 * a) % 2 == 0)


def part_h0(args, lean, docs):
    from gen_lv2_cells import cell as cell2
    cells, bad = [], 0
    for n in range(args.h0_min, args.h0_max + 1):
        if n in (701, 999, 1000):
            continue
        c = cell2(n)
        if c['h0'] != closed2(n) or c['h1'] != n // 2 + 1:
            print(f'DISAGREE n={n} reference=({c["h0"]}, {c["h1"]}) closed=({closed2(n)}, {n // 2 + 1})')
            bad += 1
        cells.append(dict(n=n, s=c['s'], h0=c['h0'], h1=n // 2 + 1))
    if bad:
        print(f'GEN_LV refused: {bad} disagreement(s); nothing written')
        return 1
    plan = pack(cells, lambda c: model_h0(c['n']), args.budget_s, args.max_cells, args.max_gib)
    for m, _, _ in plan:
        name = f"LV4S{m[0]['n']:03d}_{m[-1]['n']:03d}"
        for c in m:
            c['kmod'], c['dmod'] = 'L23.' + name, f"L23.LV4SDims{c['n'] // 100}"
    print_plan('h0', plan)
    if args.dry_run:
        return 0
    for m, t, rss in plan:
        name = m[0]['kmod'].split('.')[1]
        body = (HDR + "import GL2.L23.TreeF2Exec\nimport GL2.L23.RankFast\n\n/-!\n"
                f"# LV (slot 23) ℓ = 2 cells n = {m[0]['n']} … {m[-1]['n']}: the rank of `σ` (kernel facts, Mathlib-free; lib "
                "`L23Cells`)\n\nGenerated by `lean/tools/gen_lv_cells.py --part h0` (do not edit).  One rank per cell, `rankF2F` "
                "(= `rankF2`, `rankF2F_eq`) "
                "(`h⁰`; `h^{q+1}` is LG-3's\n`Split.V2_dims_all`).  Values: the Python reference of `gen_lv2_cells.py`, equal to the "
                f"closed forms.  Model: {t:.0f} s, {rss:.2f} GiB.\n-/\n\nnamespace GL2.L23.Cells\n\nopen GL2.L23.RankF2\n\n")
        for c in m:
            body += f"theorem rkS_{c['n']} : rankF2F (colsSig {c['n']}) = {c['s']} := by decide +kernel\n"
        body += "\nend GL2.L23.Cells\n"
        (lean / f'cells/L23/{name}.lean').write_text(body)
    for b in sorted({c['n'] // 100 for c in cells}):
        cs = [c for c in cells if c['n'] // 100 == b]
        body = HDR + "".join(f"import {k}\n" for k in sorted({c['kmod'] for c in cs}))
        body += ("import GL2.L23.LVH0\nimport GL2.L23.Splitting\n\n/-!\n"
                 f"# LV (slot 23) ℓ = 2 cells: `dim_{{F₂}} H^q(SL₂(ℤ); Sym^n F₂²)`, every `q`, n in [{100 * b}, {100 * b + 99}]\n\n"
                 "Generated by `lean/tools/gen_lv_cells.py --part h0` (do not edit).  `h⁰` by `V2_h0_of_rankS` from one kernel "
                 "fact;\n`h^{q+1} = ⌊n/2⌋ + 1` by `Split.V2_dims_all` (LG-3, every `n`).\n-/\n\n"
                 "namespace GL2.L23.Cells\n\nopen GL2.L23.RankF2 GL2.L23.TreeF2 Module\n\n")
        for c in cs:
            n = c['n']
            body += (f"/-- `n = {n}`: `dim H⁰ = {c['h0']}`, `dim H^q = {c['h1']}` (q ≥ 1). -/\n"
                     f"theorem dims_{n} : finrank (ZMod 2) (groupCohomology (V2 {n}) 0) = {c['h0']} ∧\n"
                     f"    ∀ q, finrank (ZMod 2) (groupCohomology (V2 {n}) (q + 1)) = {n} / 2 + 1 :=\n"
                     f"  ⟨V2_h0_of_rankS {n} {c['s']} {c['h0']} ((rankF2F_eq _).symm.trans rkS_{n}) (by norm_num),\n"
                     f"    GL2.L23.Split.V2_dims_all {n}⟩\n\n")
        body += "end GL2.L23.Cells\n"
        (lean / f'cells/L23/LV4SDims{b}.lean').write_text(body)
    write_manifest(docs / 'lv4_h0_manifest.tsv', ['ell', 'n', 's', 'h0', 'h1', 'closed', 'kmod', 'dmod'],
                   [dict(c, ell=2, closed='match') for c in cells])
    print(f'GEN_LV part=h0 cells={len(cells)} kernel_modules={len(plan)} written')
    return 0


# ------------------------------------------------------------------ part dims3 (ell = 3)


def part_dims3(args, lean, docs):
    from gen_lv3_cells import cell as cell3, closed3

    def kernel_module(cells, doc):
        out = (HDR + "import GL2.L23.RankFast\n\n/-!\n" + doc + "\n-/\n\nnamespace GL2.L23.Cells\n\n"
               "open GL2.L23.RankF3\n\n")
        for c in cells:
            n = c['n']
            out += (f"theorem rk3S_{n} : rankF3F (colsSig3 {n}) = {c['s']} := by decide +kernel\n"
                    f"theorem rk3P_{n} : rankF3F (colsPS {n}) = {c['p']} := by decide +kernel\n"
                    f"theorem rk3N_{n} : rankF3F (colsN3 {n}) = {c['r']} := by decide +kernel\n"
                    f"theorem rk3U_{n} : rankF3F (colsMU {n}) = {c['u']} := by decide +kernel\n\n")
        return out + "end GL2.L23.Cells\n"

    def dims_module(cells, imports, doc):
        out = HDR + "".join(f"import {m}\n" for m in imports) + "import GL2.L23.TreeF3V3\n\n/-!\n" + doc + "\n-/\n\n"
        out += "namespace GL2.L23.Cells\n\nopen GL2.L23.RankF3 GL2.L23.TreeF3 Module\n\n"
        for c in cells:
            n = c['n']
            out += (f"/-- `n = {n}`: `dim H⁰ = {c['h0']}`, `dim H¹ = {c['h1']}`, `dim H^(q+2) = {c['h2']}`. -/\n"
                    f"theorem dims3_{n} : finrank (ZMod 3) (groupCohomology (V3 {n}) 0) = {c['h0']} ∧\n"
                    f"    finrank (ZMod 3) (groupCohomology (V3 {n}) 1) = {c['h1']} ∧\n"
                    f"    ∀ q, finrank (ZMod 3) (groupCohomology (V3 {n}) (q + 2)) = {c['h2']} :=\n"
                    f"  V3_dims_of_ranks {n} {c['s']} {c['p']} {c['r']} {c['u']} {c['h0']} {c['h1']} {c['h2']} rfl\n"
                    f"    ((rankF3F_eq _).symm.trans rk3S_{n}) ((rankF3F_eq _).symm.trans rk3P_{n})\n"
                    f"    ((rankF3F_eq _).symm.trans rk3N_{n}) ((rankF3F_eq _).symm.trans rk3U_{n})\n"
                    f"    (by norm_num) (by norm_num) (by norm_num)\n\n")
        return out + "end GL2.L23.Cells\n"
    cache, cells, bad = {}, [], 0
    if args.ref_cache and Path(args.ref_cache).exists():
        for L in open(args.ref_cache):
            f = L.split()
            if len(f) == 6:
                cache[int(f[0])] = dict(s=int(f[1]), p=int(f[2]), m=int(f[3]), r=int(f[4]), u=int(f[5]))
    for n in range(args.d3_min, args.d3_max + 1, 2):
        t0 = time.time()
        c = cell3(n, cache)
        if (c['h0'], c['h1'], c['h2'], c['h2']) != closed3(n):
            print(f'DISAGREE n={n} reference={(c["h0"], c["h1"], c["h2"])} closed={closed3(n)}')
            bad += 1
        cells.append(c)
        if args.ref_cache:
            with open(args.ref_cache, 'a') as fh:
                if n not in cache or time.time() - t0 > 0.5:
                    R = cache[n]
                    fh.write(f"{n} {R['s']} {R['p']} {R['m']} {R['r']} {R['u']}\n")
    if bad:
        print(f'GEN_LV refused: {bad} disagreement(s); nothing written')
        return 1
    for c in cells:
        c['decls'] = 4
    plan = pack(cells, lambda c: model_dims3(c['n']), args.budget_s, args.max_cells, args.max_gib)
    for m, _, _ in plan:
        name = f"LV4T{m[0]['n']:03d}_{m[-1]['n']:03d}"
        for c in m:
            c['kmod'], c['dmod'] = 'L23.' + name, f"L23.LV4TDims{c['n'] // 100}"
    print_plan('dims3', plan)
    if args.dry_run:
        return 0
    for m, t, rss in plan:
        name = m[0]['kmod'].split('.')[1]
        doc = (f"# LV (slot 23) ℓ = 3 cells, even n = {m[0]['n']} … {m[-1]['n']} (kernel facts, Mathlib-free; lib `L23Cells`)\n\n"
               f"Generated by `lean/tools/gen_lv_cells.py --part dims3` (do not edit).  Four `rankF3F` (= `rankF3`, `rankF3F_eq`) per "
               f"cell.  Model: {t:.0f} s, {rss:.2f} GiB.")
        (lean / f'cells/L23/{name}.lean').write_text(kernel_module(m, doc))
    for b in sorted({c['n'] // 100 for c in cells}):
        cs = [c for c in cells if c['n'] // 100 == b]
        doc = (f"# LV (slot 23) ℓ = 3 cells: `dim_{{F₃}} H^q(SL₂(ℤ); Sym^n F₃²)` for the even n in [{100 * b}, {100 * b + 99}] "
               "(lib `L23Cells`)\n\nGenerated by `lean/tools/gen_lv_cells.py --part dims3` (do not edit).")
        (lean / f'cells/L23/LV4TDims{b}.lean').write_text(dims_module(cs, sorted({c['kmod'] for c in cs}), doc))
    write_manifest(docs / 'lv4_dims3_manifest.tsv',
                   ['ell', 'n', 's', 'p', 'r', 'u', 'h0', 'h1', 'h2', 'closed', 'kmod', 'dmod'],
                   [dict(c, ell=3, closed='match') for c in cells])
    print(f'GEN_LV part=dims3 cells={len(cells)} kernel_modules={len(plan)} written')
    return 0


# ------------------------------------------------------------------ part smith


def rank_mod_np(M, p):
    import numpy as np
    A = np.array(M, dtype=object)
    A = np.array([[int(x) % p for x in row] for row in M], dtype=np.int64)
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        q = r + nz[0]
        A[[r, q]] = A[[q, r]]
        A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        f = A[:, c].copy()
        f[r] = 0
        A = (A - np.outer(f, A[r]) % p) % p
        r += 1
        if r == rows:
            break
    return r


def part_smith(args, lean, docs):
    import lv_smith_ref as ref
    KS = {2: 10, 3: 6}
    P1, P2 = 2147483629, 2147483587
    cells, bad = [], 0
    for ell in (2, 3):
        K = KS[ell]
        N = ell ** K
        for n in range(0, args.smith_max + 1, 2):
            D = ref.tree_mats(n) if (n <= args.ref_max or args.certify) else None
            for q in range(3):
                R = 2 * (n + 1) if q < 2 else 3 * (n + 1)
                H = 1
                while N + R * N * N >= 2 ** H:
                    H += 1
                cols = ref.cols_lean(n, q, N, 2 * H)
                st = {}
                vs = [p[0] for p in ref.smithpk(ell, K, H, 3 * (n + 1), cols, st)]
                if D is not None and n <= args.ref_max:
                    vr = ref.smith_vals(D[q], ell, K)
                    if vr != vs:
                        print(f'DISAGREE ell={ell} n={n} q={q} mirror={vs} reference={vr}')
                        bad += 1
                cert = '-'
                if D is not None and args.certify:
                    rq = max(rank_mod_np(D[q], P1), rank_mod_np(D[q], P2))
                    cert = 'yes' if (len(vs) == rq and (not vs or max(vs) < K - 1)) else 'no'
                s = sum(K - v for v in vs)
                cnt = [sum(1 for v in vs if v == u) for u in range(K)]
                cells.append(dict(ell=ell, n=n, q=q, K=K, H=H, R=R, s=s, cnt=cnt, nv=len(vs),
                                  vmax=max(vs) if vs else -1, cert=cert, decls=1,
                                  vals=','.join(str(v) for v in vs if v > 0) or '-'))
    if bad:
        print(f'GEN_LV refused: {bad} disagreement(s); nothing written')
        return 1
    small = [c for c in cells if c['n'] <= args.smith_gl2]
    large = [c for c in cells if c['n'] > args.smith_gl2]
    plans = {}
    for ell in (2, 3):
        cs = [c for c in large if c['ell'] == ell]
        plans[ell] = pack(cs, lambda c: model_smith(c['n'], c['q'], c['ell']), args.budget_s, args.max_cells, args.max_gib)
        for m, _, _ in plans[ell]:
            name = f"LV4M{ell}_{m[0]['n']:03d}_{m[-1]['n']:03d}"
            for c in m:
                c['kmod'], c['dmod'] = 'L23.' + name, f"L23.LV4MCard{ell}_{c['n'] // 50}"
        print_plan(f'smith{ell}', plans[ell])
    for c in small:
        c['kmod'], c['dmod'] = 'GL2.L23.CellsSmith', 'GL2.L23.CellsSmithCard'
    t_small = sum(model_smith(c['n'], c['q'], c['ell'])[0] for c in small)
    print(f"GEN_LV_PLAN part=smith_gl2 cells={len(small)} kernel_s={t_small:.0f} (GL2/L23/CellsSmith.lean)")
    if args.dry_run:
        return 0

    def fact(c):
        cnt = '[' + ', '.join(str(x) for x in c['cnt']) + ']'
        return (f"theorem sig{c['ell']}_{c['n']}_{c['q']} : sig {c['K']} (cellPiv {c['ell']} {c['K']} {c['H']} {c['n']} "
                f"{c['q']}) = ({c['s']}, {cnt}) := by decide +kernel\n")

    def card(c):
        n, q, ell, K, H = c['n'], c['q'], c['ell'], c['K'], c['H']
        cnt = '[' + ', '.join(str(x) for x in c['cnt']) + ']'
        R = '2 * ({} + 1)'.format(n) if q < 2 else '3 * ({} + 1)'.format(n)
        return (f"/-- `ℓ = {ell}`, `n = {n}`, `d^{q}`: order of the span; valuations `> 0` (their counts certified by `type{ell}_{n}_{q}`): "
                f"{c['vals']}; reference K-certification (Python): {c['cert']}. -/\n"
                f"theorem card{ell}_{n}_{q} : Nat.card (spanK {ell} {K} {H} (3 * ({n} + 1)) (colsD{q} (2 * {H}) ({ell} ^ {K}) {n}))"
                f" = {ell} ^ {c['s']} :=\n"
                f"  cell_card{q} (n := {n}) ⟨by norm_num, by norm_num, by norm_num, by norm_num⟩ sig{ell}_{n}_{q}\n" +
                f"\n/-- The type (LV audit R1(a)): `#(ℓ^t · span)` for every `t`, from the certified counts. -/\n"
                f"theorem type{ell}_{n}_{q} (t : ℕ) : Nat.card ((spanK {ell} {K} {H} (3 * ({n} + 1)) (colsD{q} (2 * {H}) "
                f"({ell} ^ {K}) {n})).map ((({ell} : ℕ) : ZMod ({ell} ^ {K})) ^ t • LinearMap.id)) =\n"
                f"    {ell} ^ ∑ v ∈ Finset.range {K}, {cnt}.getD v 0 * ({K} - (v + t)) :=\n"
                f"  cell_type{q} (n := {n}) ⟨by norm_num, by norm_num, by norm_num, by norm_num⟩ sig{ell}_{n}_{q} t\n")

    kdoc = ("Generated by `lean/tools/gen_lv_cells.py --part smith` (do not edit).  Per cell the Smith signature\n"
            "`sig K (cellPiv ℓ K H n q) = (Σ (K − v), counts)` of `d^q` on `Sym^n (ℤ/ℓ^K)²` by `decide +kernel` (no `native_decide`,\n"
            "no literal matrix data; the matrices are generated from `n`).  Values: the step mirror of the kernel algorithm,\n"
            f"equal to the independent full-pivot reference for n ≤ {args.ref_max}.")
    head = HDR + "import GL2.L23.SmithPKExec\n\n/-!\n"
    body = head + "# LV (slot 23) Smith cells, even n ≤ " + str(args.smith_gl2) + " (kernel facts, Mathlib-free)\n\n" + kdoc
    body += "\n-/\n\nnamespace GL2.L23.Cells\n\nopen GL2.L23.SmithPK\n\n" + "".join(fact(c) for c in small)
    neg = next(c for c in small if c['ell'] == 2 and c['n'] == 10 and c['q'] == 0)
    body += ("\n/-! Negative control: the kernel refuses a signature one too small (the refutation path works). -/\n"
             f"theorem sig2_10_0_ne : sig {neg['K']} (cellPiv 2 {neg['K']} {neg['H']} 10 0) ≠ ({neg['s'] - 1}, "
             f"[{', '.join(str(x) for x in neg['cnt'])}]) := by decide +kernel\n")
    body += "\nend GL2.L23.Cells\n"
    (lean / 'GL2/L23/CellsSmith.lean').write_text(body)
    body = (HDR + "import GL2.L23.CellsSmith\nimport GL2.L23.SmithPKCellType\n\n/-!\n"
            f"# LV (slot 23) Smith cells: `#span_{{ℤ/ℓ^K}}(d^q) = ℓ^s`, even n ≤ {args.smith_gl2}\n\n"
            "Generated by `lean/tools/gen_lv_cells.py --part smith` (do not edit).  Each `card` is `cell_card0/1/2`, each `type` is\n"
            "`cell_type0/1/2` (the counts certified, LV audit R1(a)), applied to the kernel fact of `CellsSmith.lean`.\n-/\n\nnamespace GL2.L23.Cells\n\nopen GL2.L23.SmithPK\n\n")
    body += "\n".join(card(c) for c in small) + "\nend GL2.L23.Cells\n"
    (lean / 'GL2/L23/CellsSmithCard.lean').write_text(body)
    for ell in (2, 3):
        for m, t, rss in plans[ell]:
            name = m[0]['kmod'].split('.')[1]
            b = head + (f"# LV (slot 23) Smith cells ℓ = {ell}, even n = {m[0]['n']} … {m[-1]['n']} (kernel facts, Mathlib-free; "
                        f"lib `L23Cells`)\n\n{kdoc}  Model: {t:.0f} s, {rss:.2f} GiB.\n-/\n\nnamespace GL2.L23.Cells\n\n"
                        "open GL2.L23.SmithPK\n\n") + "".join(fact(c) for c in m) + "\nend GL2.L23.Cells\n"
            (lean / f'cells/L23/{name}.lean').write_text(b)
        for blk in sorted({c['n'] // 50 for c in large if c['ell'] == ell}):
            cs = [c for c in large if c['ell'] == ell and c['n'] // 50 == blk]
            b = HDR + "".join(f"import {k}\n" for k in sorted({c['kmod'] for c in cs})) + "import GL2.L23.SmithPKCellType\n\n/-!\n"
            b += (f"# LV (slot 23) Smith cells ℓ = {ell}: `#span_{{ℤ/ℓ^K}}(d^q) = ℓ^s`, even n in [{50 * blk}, {50 * blk + 49}] "
                  "(lib `L23Cells`)\n\nGenerated by `lean/tools/gen_lv_cells.py --part smith` (do not edit).\n-/\n\n"
                  "namespace GL2.L23.Cells\n\nopen GL2.L23.SmithPK\n\n")
            b += "\n".join(card(c) for c in cs) + "\nend GL2.L23.Cells\n"
            (lean / f'cells/L23/LV4MCard{ell}_{blk}.lean').write_text(b)
    write_manifest(docs / 'lv4_smith_manifest.tsv',
                   ['ell', 'n', 'q', 'K', 'H', 'R', 's', 'nv', 'vmax', 'vals', 'cert', 'kmod', 'dmod'], cells)
    print(f'GEN_LV part=smith cells={len(cells)} gl2={len(small)} written')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['h0', 'dims3', 'smith'], required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--budget-s', type=float, default=150.0)
    ap.add_argument('--max-cells', type=int, default=30)
    ap.add_argument('--max-gib', type=float, default=12.0)
    ap.add_argument('--h0-min', type=int, default=702)
    ap.add_argument('--h0-max', type=int, default=998)
    ap.add_argument('--d3-min', type=int, default=702)
    ap.add_argument('--d3-max', type=int, default=1000)
    ap.add_argument('--ref-cache', default=None)
    ap.add_argument('--smith-max', type=int, default=150)
    ap.add_argument('--smith-gl2', type=int, default=30)
    ap.add_argument('--ref-max', type=int, default=40)
    ap.add_argument('--certify', action='store_true')
    args = ap.parse_args()
    lean = HERE.parent
    docs = lean.parent / 'docs/implementation/slot23/lv'
    (lean / 'cells/L23').mkdir(parents=True, exist_ok=True)
    return {'h0': part_h0, 'dims3': part_dims3, 'smith': part_smith}[args.part](args, lean, docs)


if __name__ == '__main__':
    sys.exit(main())
