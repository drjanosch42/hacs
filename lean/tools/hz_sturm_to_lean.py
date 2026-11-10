#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""Untrusted generator of the LC-11 (01 Prop. 5.6) certificates for the S1 cells (slot 25; 07-lean-plan.md §2.2 LC-11, §10).

Reads
  * the committed L-HZ1 data (`docs/implementation/slot25/lhz1-data/`): the ed.1 `char_Z` extracts of T_2, T_3, T_5 (TB-chi);
  * the cell certificates' cofactors from `lean/GL2/HeckeZ/Cells/S1Data{A,B,C}.lean` (01 Thm. 4.1, L-HZ1);
  * the slot-25 EXPORT runs `ex_n<n>_ell<l>_K<K>.txt` (console `descent(1, n, [2, 3, 5]).module(l, K)`; TB-mat for the reads:
    the free block of T_p on H^1 (x) Z/l^K = L/l^K (+ torsion), column convention),
and writes the Mathlib-free data module `S1SturmData.lean` (`SturmCert` values with `decide +kernel` checks), the theorem
module `S1Sturm.lean` and `manifest_lc11.tsv`. Everything it computes (c = f/(x - lambda), r = chi/(x - lambda), g, the reads'
s and v, the primality bounds, cof56, cof) is re-checked by the kernel; only chi and the exported matrices are trusted.

Usage:
  hz_sturm_to_lean.py plan <lhz1_data_dir> [<export_dir>]          -> the (n, ell, K) exports to run
  hz_sturm_to_lean.py emit <lhz1_data_dir> <export_dir> <lean_cells_dir> <out_dir>
"""
import ast, hashlib, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hz_descent_to_lean as G  # noqa: E402

CELLS = list(range(2, 63, 2))
PS = (2, 3, 5)
WORD = 2 ** 31  # EXPORT: ell prime < 2^31, ell^K < 2^31 at ell >= 5


def sha256(path):
    return G.sha256(path)


def ev(le, x):
    r = 0
    for c in reversed(le):
        r = r * x + c
    return r


def div_lin(le, lam):
    q, r = G.pdivmod(le, [-lam, 1])
    assert not r and all(x.denominator == 1 for x in q)
    return [int(x) for x in q]


def vl(x, l):
    x = abs(x)
    if x == 0:
        return 10 ** 9
    v = 0
    while x % l == 0:
        x //= l
        v += 1
    return v


def cell_data(data_dir):
    chis = G.read_charz_extract(data_dir, set(CELLS))
    out = {}
    for n in CELLS:
        k = n + 2
        b1 = len(chis[(n, 2)]) - 1
        d = (b1 + 1) // 2
        items, g = [], 0
        for p in PS:
            it = G.k2_item(chis[(n, p)], p, d)
            lam = 1 + p ** (k - 1)
            c = div_lin(it['f'], lam)
            r = div_lin(it['chi'], lam)
            assert ev(r, lam) != 0, ('lambda not a simple root', n, p)
            items.append(dict(p=p, c=c, r=r, chi=it['chi'], val=ev(c, lam)))
            g = math.gcd(g, abs(ev(c, lam)))
        out[n] = dict(n=n, k=k, b1=b1, d=d, items=items, g=g)
    return out


def read_cofs(cells_dir):
    cofs = {}
    for name in ('S1DataA', 'S1DataB', 'S1DataC'):
        s = open(os.path.join(cells_dir, name + '.lean')).read()
        for m in re.finditer(r'def cert_n(\d+) : CellCert where.*?\n  cof := (\d+)\n', s, re.S):
            cofs[int(m.group(1))] = int(m.group(2))
    assert sorted(cofs) == CELLS, sorted(cofs)
    return cofs


def candidates(c):
    fac, rest = G.factor_small(c['g'], 2 * 10 ** 6) if c['g'] > 1 else ({}, 1)
    return sorted(l for l in fac if l < WORD), fac, rest


def read_export(path):
    meta, ops = {}, []
    for line in open(path):
        if line.startswith('status'):
            w = line.split()
            meta = dict(zip(w[0::2], w[1::2]))
            meta['torsion'] = ast.literal_eval(line.split('torsion', 1)[1].split('refusal')[0].strip())
        elif line.startswith('op '):
            w = line.split()
            ops.append([int(w[1]), ast.literal_eval(line.split('charpoly_free', 1)[1].strip()), None])
        elif line.startswith('matrix'):
            ops[-1][2] = ast.literal_eval(line[len('matrix'):].strip())
    return meta, ops


def peval_mat(c, M, q):
    b = len(M)
    R = [[0] * b for _ in range(b)]
    for a in reversed(c):
        R = G.matmul(M, R, q)
        for i in range(b):
            R[i][i] = (R[i][i] + a) % q
    return R


def free_block(M, t, b1):
    assert len(M) == t + b1
    for i in range(t, t + b1):
        for j in range(t):
            assert M[i][j] == 0, 'torsion generators must map into torsion (column convention)'
    return [row[t:] for row in M[t:]]


def try_read(c, ell, export_dir):
    """The best read at ell from the exports present: (read dict, file, sha) or (None, reason)."""
    v = vl(c['g'], ell)
    best = None
    for K in range(1, 8):
        fn = os.path.join(export_dir, 'ex_n%d_ell%d_K%d.txt' % (c['n'], ell, K))
        if not os.path.exists(fn):
            continue
        meta, ops = read_export(fn)
        if meta.get('status') != 'ok' or int(meta['K']) != K:
            continue
        q = ell ** K
        t = len(meta['torsion'])
        for p, cp, M in ops:
            it = next(i for i in c['items'] if i['p'] == p)
            F = free_block(M, t, c['b1'])
            assert all(x % q == 0 for r in peval_mat(it['chi'], F, q) for x in r), ('chi(M) != 0', c['n'], ell, K, p)
            s = K - 1
            C = peval_mat(it['c'], F, q)
            if any(x % (ell ** s) for r in C for x in r):
                continue
            if all(x % q == 0 for r in C for x in r):
                continue
            if vl(it['val'], ell) < s + v:
                continue
            rd = dict(ell=ell, sq=math.isqrt(ell), s=s, v=v, p=p, mat=F, file=os.path.basename(fn), sha=sha256(fn))
            if best is None or s < best['s']:
                best = rd
        if best is not None:
            return best, None
    if not any(os.path.exists(os.path.join(export_dir, 'ex_n%d_ell%d_K%d.txt' % (c['n'], ell, K))) and
               read_export(os.path.join(export_dir, 'ex_n%d_ell%d_K%d.txt' % (c['n'], ell, K)))[0].get('status') == 'ok'
               for K in range(1, 8)):
        return None, 'no_export'
    return None, 'no_read'


def plan(data_dir, export_dir=None):
    cd = cell_data(data_dir)
    for n in CELLS:
        c = cd[n]
        cands, _, _ = candidates(c)
        for ell in cands:
            print(n, ell, 1)
            if export_dir is None:
                continue
            rd, _ = try_read(c, ell, export_dir)
            if rd is None:
                slack = max(vl(it['val'], ell) for it in c['items']) - vl(c['g'], ell)
                for K in range(2, slack + 2):
                    if ell ** K < WORD:
                        print(n, ell, K)


def lean_list(xs):
    return '[' + ', '.join(str(x) for x in xs) + ']'


def lean_mat(M):
    return '[' + ', '.join(lean_list(r) for r in M) + ']'


DATA_HEADER = """/-! GENERATED by `lean/tools/hz_sturm_to_lean.py` (LC-11, slot 25) — do not edit by hand.

01 Prop. 5.6 certificates (`GL2.HeckeZ.Exec.SturmCert`) for the 31 S1 cells, on top of the L-HZ1 cell certificates
`cert_n<n>`; Mathlib-free, checked by `decide +kernel`. Trusted inputs (named in the theorems of `S1Sturm`): TB-χ (the ed.1
`char_Z` of `T_2, T_3, T_5`, shared with `cert_n<n>`) and, for each read, TB-mat: the free block of `T_{p₀}` on
`H¹ ⊗ ℤ/ℓ^{s+1} = L/ℓ^{s+1}` from the slot-25 EXPORT (`DescentRecord.module(ℓ, s + 1)`; file names and sha256 in the docstrings
and in `docs/implementation/slot25/lc11-data/SHA256SUMS`). Everything else (`c = f/(x − λ)`, `r = χ/(x − λ)`, `g`, `s`, `v`, the
primality bounds, `cof56`, `cof`) is computed by the untrusted generator and re-checked by the kernel. -/
"""


def emit(data_dir, export_dir, cells_dir, out_dir):
    cd = cell_data(data_dir)
    cofs = read_cofs(cells_dir)
    rows, certs = [], {}
    for n in CELLS:
        c = cd[n]
        cands, fac, rest = candidates(c)
        reads, unread = [], {}
        for ell in cands:
            rd, why = try_read(c, ell, export_dir)
            if rd is None:
                unread[ell] = why
            else:
                reads.append(rd)
        big = {l: e for l, e in fac.items() if l >= WORD}
        cof56 = c['g']
        for rd in reads:
            while cof56 % rd['ell'] == 0:
                cof56 //= rd['ell']
        ex = max([rd['v'] for rd in reads] + [0])
        rprod = 1
        for rd in reads:
            rprod *= rd['ell']
        assert (cof56 * rprod ** ex) % c['g'] == 0
        cof = math.gcd(cofs[n], cof56)
        c.update(reads=reads, cof56=cof56, ex=ex, cof=cof, cc_cof=cofs[n], unread=unread, rest=rest, big=big)
        certs[n] = c
        cf, crest = G.factor_small(cof, 2 * 10 ** 6) if cof > 1 else ({}, 1)
        rows.append((n, c['b1'], c['d'], c['g'].bit_length(), ','.join(str(r['ell']) for r in reads) or '-',
                     ';'.join('%d:%s' % (l, unread[l]) for l in sorted(unread)) or '-',
                     cof56 == 1, cofs[n] == 1, cof == 1,
                     ';'.join('%d^%d' % (l, e) for l, e in sorted(cf.items())) or '-',
                     '%d' % crest.bit_length() if crest > 1 else '-'))
    # data module
    path = os.path.join(out_dir, 'S1SturmData.lean')
    with open(path, 'w') as o:
        o.write('/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/\n')
        o.write('import GL2.HeckeZ.Exec.Sturm\nimport GL2.HeckeZ.Cells.S1DataA\nimport GL2.HeckeZ.Cells.S1DataB\n'
                'import GL2.HeckeZ.Cells.S1DataC\n\n')
        o.write(DATA_HEADER)
        o.write('\nnamespace GL2.HeckeZ.Cells.S1SturmData\n\nopen GL2.HeckeZ.Exec GL2.HeckeZ.Cells.S1DataA '
                'GL2.HeckeZ.Cells.S1DataB GL2.HeckeZ.Cells.S1DataC\n\n')
        for n in CELLS:
            c = certs[n]
            o.write('/-- Cell `(1, %d)` (weight %d): `g = gcd_p c_p(λ_p)` has %d bits; reads at %s; `cof56 = %s`; with Thm. 4.1 '
                    '`cof = gcd(cert_n%d.cof, cof56) = %s`. ' % (
                        n, c['k'], c['g'].bit_length(), ', '.join(str(r['ell']) for r in c['reads']) or 'none',
                        c['cof56'] if c['cof56'].bit_length() < 64 else '(%d bits)' % c['cof56'].bit_length(), n,
                        c['cof'] if c['cof'].bit_length() < 64 else '(%d bits)' % c['cof'].bit_length()))
            for r in c['reads']:
                o.write('Read at %d (`s = %d`, `v = %d`, `T_%d`): export `%s` (sha256 `%s`). ' % (
                    r['ell'], r['s'], r['v'], r['p'], r['file'], r['sha']))
            if c['unread']:
                o.write('No closing read at %s. ' % ', '.join(str(l) for l in sorted(c['unread'])))
            o.write('-/\n')
            o.write('def sturm_n%d : SturmCert where\n  k := %d\n' % (n, c['k']))
            o.write('  items := [\n' + ',\n'.join('    { p := %d,\n      c := %s,\n      r := %s }' % (
                it['p'], lean_list(it['c']), lean_list(it['r'])) for it in c['items']) + ' ]\n')
            o.write('  reads := [' + ','.join('\n    { ell := %d, sq := %d, s := %d, v := %d, p := %d,\n      mat := %s }' % (
                r['ell'], r['sq'], r['s'], r['v'], r['p'], lean_mat(r['mat'])) for r in c['reads']) + ' ]\n')
            o.write('  cof56 := %d\n  ex := %d\n  cof := %d\n\n' % (c['cof56'], c['ex'], c['cof']))
            o.write('theorem sturm_n%d_check : sturmCheck cert_n%d sturm_n%d = true := by decide +kernel\n\n' % (n, n, n))
        o.write('end GL2.HeckeZ.Cells.S1SturmData\n')
    print('wrote', path, os.path.getsize(path))
    # theorem module
    tpath = os.path.join(out_dir, 'S1Sturm.lean')
    with open(tpath, 'w') as o:
        o.write('/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/\n')
        o.write('import Mathlib\nimport GL2.HeckeZ.Check.Sturm\nimport GL2.HeckeZ.Cells.S1SturmData\n')
        o.write(THEOREM_HEADER)
        o.write('namespace GL2.HeckeZ.Cells.S1Sturm\n\nopen GL2.Literature GL2.HeckeZ.Exec GL2.HeckeZ.Check GL2.HeckeZ.Cert '
                'GL2.HeckeZ.Bridge TensorProduct\nopen GL2.HeckeZ.Cells.S1DataA GL2.HeckeZ.Cells.S1DataB '
                'GL2.HeckeZ.Cells.S1DataC GL2.HeckeZ.Cells.S1SturmData\n\n')
        o.write('variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L] {p : ℕ} [Fact p.Prime]\n')
        o.write('  {E : Type*} [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]\n\n')
        for n in CELLS:
            c = certs[n]
            b1 = c['b1']
            hyp_common = ('(X : ℕ → Module.End ℤ L) (hrk : Module.finrank ℤ L = %d)\n'
                          '    (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E) (hS : SturmCuspLevelOne X %d)\n'
                          '    (hχ : ∀ it ∈ cert_n%d.items, (X it.p).charpoly = toPoly it.chi)\n' % (b1, c['k'], n))
            hyp_read = ('    (hread : ∀ rd ∈ sturm_n%d.reads, ∃ (b : Module.Basis (Fin %d) ℤ L)\n'
                        '      (U : Matrix (Fin %d) (Fin %d) (ZMod (rd.ell ^ (rd.s + 1)))), IsUnit U ∧\n'
                        '        U * (LinearMap.toMatrix b b (X rd.p)).map (Int.cast) = lmat %d (rd.ell ^ (rd.s + 1)) rd.mat * U)'
                        % (n, b1, b1, b1, b1))
            hyp_mat = ('    (hmat : ∀ k ∈ cert_n%d.k1, ∃ (b : Module.Basis (Fin %d) ℤ L) (U : Matrix (Fin %d) (Fin %d) (ZMod k.q)), IsUnit U ∧\n'
                       '      ∀ j < k.gens.length,\n'
                       '        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) = lmat %d k.q (k.mats.getD j []) * U)\n'
                       '    (hmatm : ∀ k ∈ cert_n%d.k1m, ∃ (b : Module.Basis (Fin k.fB ⊕ Fin (%d - k.fB)) ℤ L)\n'
                       '      (U : Matrix (Fin k.fB ⊕ Fin (%d - k.fB)) (Fin k.fB ⊕ Fin (%d - k.fB)) (ZMod k.q)), IsUnit U ∧\n'
                       '      ∀ j < k.gens.length, ∃ Y : Matrix (Fin (%d - k.fB)) (Fin (%d - k.fB)) (ZMod k.q),\n'
                       '        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) =\n'
                       '          Matrix.fromBlocks (lmat k.fB k.q (k.mats.getD j [])) 0 0 Y * U)\n'
                       % (n, b1, b1, b1, b1, n, b1, b1, b1, b1, b1))
            reads = ', '.join(str(r['ell']) for r in c['reads']) or 'none'
            # Prop. 5.6 alone
            o.write('/-- **Cell `(1, %d)`, weight %d, by 01 Prop. 5.6 alone** (LC-11): `g_P` has %d bits, reads at %s. ' % (
                n, c['k'], c['g'].bit_length(), reads))
            if c['cof56'] == 1:
                o.write('**`𝕋^fr = ℤ[T_2, T_3, T_5]`**, with no saturation, K1 or K1m certificate. ')
            else:
                o.write('`𝕋\'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof56` (%d bits), in particular at `2`, `3`, `5`. ' %
                        c['cof56'].bit_length())
            o.write('Trusted: `hχ` (ed.1 `char_Z`), `hread` (the EXPORT matrices named in `sturm_n%d`); literature: `hES`, '
                    '`hS` (`hSturm`). -/\n' % n)
            o.write('theorem prop56_n%d %s%s :\n' % (n, hyp_common, hyp_read))
            if c['cof56'] == 1:
                o.write('    Tfr X = Tp X [2, 3, 5] :=\n')
                o.write('  (cell56_sound cert_n%d cert_n%d_check sturm_n%d sturm_n%d_check X hrk hES hS hχ hread).2 rfl\n\n' % (n, n, n, n))
            else:
                o.write('    ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ %d → Loc (Tfr X) (Tp X [2, 3, 5]) ℓ :=\n' % c['cof56'])
                o.write('  (cell56_sound cert_n%d cert_n%d_check sturm_n%d sturm_n%d_check X hrk hES hS hχ hread).1\n\n' % (n, n, n, n))
            # with Thm. 4.1
            o.write('/-- **Cell `(1, %d)`, weight %d, by 01 Thm. 4.1 and Prop. 5.6** (M-HZ2 after LC-11). ' % (n, c['k']))
            if c['cof'] == 1:
                o.write('**`𝕋^fr = ℤ[T_2, T_3, T_5]`**. ')
            else:
                o.write('`𝕋\'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof = %s`. ' % (
                    c['cof'] if c['cof'].bit_length() < 64 else '(%d bits)' % c['cof'].bit_length()))
            o.write('Trusted: `hχ`, `hmat` / `hmatm` (L-HZ1 dumps of `cert_n%d`), `hread` (EXPORT, `sturm_n%d`); literature: '
                    '`hES`, `hS`. -/\n' % (n, n))
            o.write('theorem cellZ_n%d %s%s%s :\n' % (n, hyp_common, hyp_mat, hyp_read))
            if c['cof'] == 1:
                o.write('    Tfr X = Tp X [2, 3, 5] :=\n')
                o.write('  (cell_sturm_sound cert_n%d cert_n%d_check sturm_n%d sturm_n%d_check X hrk hES hS hχ hmat hmatm hread).2 rfl\n\n' % (n, n, n, n))
            else:
                o.write('    ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ %d → Loc (Tfr X) (Tp X [2, 3, 5]) ℓ :=\n' % c['cof'])
                o.write('  (cell_sturm_sound cert_n%d cert_n%d_check sturm_n%d sturm_n%d_check X hrk hES hS hχ hmat hmatm hread).1\n\n' % (n, n, n, n))
        o.write('end GL2.HeckeZ.Cells.S1Sturm\n')
    print('wrote', tpath, os.path.getsize(tpath))
    with open(os.path.join(out_dir, 'manifest_lc11.tsv'), 'w') as o:
        o.write('n\tb1\td\tg_bits\treads\tunread\tprop56_complete\tthm41_complete\tcomplete\tcof_factors\tcof_unfactored_bits\n')
        for r in rows:
            o.write('\t'.join(str(x) for x in r) + '\n')
    for r in rows:
        print(*r)


THEOREM_HEADER = """
/-!
# HeckeZ S1 after LC-11: 01 Prop. 5.6, alone and with Thm. 4.1, on the 31 S1 cells (slot 25, M-HZ2)

GENERATED by `lean/tools/hz_sturm_to_lean.py` — do not edit by hand. `07-lean-plan.md` §2.2 LC-11, §10 (M-HZ2: "LC-11
reproduces T1, T2"). `L` is `H¹(SL₂(ℤ); Sym^n ℤ)/tors` and `X m` the Hecke operator `T_m` (interface data, as in
`GL2.HeckeZ.Cells.S1`); `𝕋' = ℤ[T_2, T_3, T_5]`. Per cell two theorems:
* `prop56_n<n>` — 01 Prop. 5.6 alone (`GL2.HeckeZ.Check.cell56_sound`): `hSturm` gives `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime
  `ℓ ∤ g_P` (so at `2`, `3`, `5` in every cell), and a read at a root (01 Prop. 5.4) closes a prime `ℓ ∣ g_P`;
* `cellZ_n<n>` — with 01 Thm. 4.1 (`cell_sturm_sound`): at every prime not dividing `gcd(cof, cof56)`.
**Trusted C++ outputs ([LP] v1.9, Q10(a)):** `hχ` (TB-χ, ed.1 `char_Z`), `hmat` / `hmatm` (TB-mat, the L-HZ1
`HECKE_DUMP_SAT_CERT` blocks), `hread` (TB-mat, the slot-25 EXPORT matrices of the reads). **Literature:** `hES`
(`lit:eichler-shimura-levelone`) and `hS` (`hSturm`, `lit:hz-sturm-integral`, `GL2.Literature.SturmCuspLevelOne`).
-/

"""


if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        plan(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == 'emit':
        emit(*sys.argv[2:6])
    else:
        sys.exit(__doc__)
