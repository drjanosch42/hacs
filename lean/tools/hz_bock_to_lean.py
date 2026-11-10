#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
"""hz_bock_to_lean.py -- the L-HZ2 (M-HZ3) data generator: K5, K6, K7, K8 certificates for GL2.HeckeZ.Cells.BockData.

UNTRUSTED: every number it computes (the torsion polynomial, the nilpotency exponent, the Bezout pairs, sections, pivot
minors and their inverses, the Dickson exponents) is re-checked by the Lean kernel (decide +kernel on the Mathlib-free
checkers of GL2/HeckeZ/Exec/Bock.lean).  The trusted inputs it only copies:
  tb_mat  the export matrices of DescentRecord.module(ell, K) (slot-25 EXPORT; JSON dumps m<n>_l<ell>.json);
  tb_chi  the integral characteristic polynomials char_Z of the integral census ed.1 (committed level-1 extract
          docs/implementation/slot25/lhz1-data/ed1_charz_level1_n_le_62.tsv).
The K5 antecedent (the descent groups, d<n>.json) and the emitted h^q (docs/implementation/slot22/lv/lean_dims_l{2,3}.tsv, the
Lean-kernel dimensions of the L23 lane) are compared, not trusted.

usage: hz_bock_to_lean.py <dump_dir> <repo_root> <out.lean> <manifest.tsv>
"""
import json, sys, os, hashlib
from fractions import Fraction

dump, repo, out, manifest = sys.argv[1:5]

def J(tag):
    return json.load(open(os.path.join(dump, tag + '.json')))

def sha(tag):
    return hashlib.sha256(open(os.path.join(dump, tag + '.json'), 'rb').read()).hexdigest()

def lst(v):
    return '[' + ', '.join(str(x) for x in v) + ']'

def mat(M):
    return '[' + ', '.join(lst(r) for r in M) + ']'

# ---------------- polynomial helpers (little-endian, mod a prime) ----------------
def ptrim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p

def pmod(p, m):
    return ptrim([x % m for x in p])

def pmulm(a, b, m):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] = (r[i + j] + x * y) % m
    return ptrim(r)

def psubm(a, b, m):
    n = max(len(a), len(b))
    return ptrim([((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % m for i in range(n)])

def pdivmod(a, b, m):
    a = pmod(a, m); b = pmod(b, m)
    q = [0] * max(0, len(a) - len(b) + 1)
    inv = pow(b[-1], -1, m)
    while len(a) >= len(b) and a:
        c = a[-1] * inv % m
        d = len(a) - len(b)
        q[d] = c
        a = psubm(a, [0] * d + [c * x % m for x in b], m)
    return ptrim(q), a

def pegcd(a, b, m):
    """u, v with u a + v b = gcd (monic) over F_m."""
    r0, r1 = pmod(a, m), pmod(b, m)
    u0, u1, v0, v1 = [1], [], [], [1]
    while r1:
        q, r = pdivmod(r0, r1, m)
        r0, r1 = r1, r
        u0, u1 = u1, psubm(u0, pmulm(q, u1, m), m)
        v0, v1 = v1, psubm(v0, pmulm(q, v1, m), m)
    inv = pow(r0[-1], -1, m)
    return [x * inv % m for x in u0], [x * inv % m for x in v0], [x * inv % m for x in r0]

def charpoly_int(C):
    """det(x - C), little-endian, by Faddeev-LeVerrier over Q."""
    n = len(C)
    M = [[Fraction(0)] * n for _ in range(n)]
    coeffs = [Fraction(1)]
    I = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    Mk = [[Fraction(0)] * n for _ in range(n)]
    c = Fraction(1)
    cs = [Fraction(1)]
    for k in range(1, n + 1):
        Mk = [[sum(Fraction(C[i][l]) * Mk[l][j] for l in range(n)) + (cs[-1] * I[i][j]) for j in range(n)] for i in range(n)]
        tr = sum(sum(Fraction(C[i][l]) * Mk[l][i] for l in range(n)) for i in range(n))
        cs.append(-tr / k)
    # cs[k] is the coefficient of x^{n-k}
    return [int(cs[n - i]) for i in range(n + 1)]

# ---------------- K5 ----------------
def lean_dims(ell):
    rows = {}
    for line in open(os.path.join(repo, 'docs/implementation/slot22/lv/lean_dims_l%d.tsv' % ell)):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        if f[0] == 'ell' or f[2] != '1':
            continue
        rows[int(f[1])] = ([int(x) for x in f[5:9]], f[4], f[10])
    return rows

k5 = []
LD = {2: lean_dims(2), 3: lean_dims(3)}
for n in range(8, 61, 2):
    st, g = J('d%d' % n)
    assert st == 'ok'
    for ell in (2, 3):
        b = [g[q]['free_rank'] for q in range(4)]
        mu = []
        for q in range(4):
            mx = max([e for (l, e, m) in g[q]['torsion'] if l == ell], default=0)
            mu.append([sum(t[2] for t in g[q]['torsion'] if t[0] == ell and t[1] == j) for j in range(1, mx + 1)])
        h, decl, mod = LD[ell][n]
        t = lambda q: sum(mu[2 if q == 4 else q])
        pred = [b[q] + t(q) + t(q + 1) for q in range(4)]
        k5.append((ell, n, b, mu, h, decl, mod, pred == h, sha('d%d' % n)))

# ---------------- tb_chi ----------------
charz = {}
for line in open(os.path.join(repo, 'docs/implementation/slot25/lhz1-data/ed1_charz_level1_n_le_62.tsv')):
    if line.startswith('#') or line.startswith('N\t'):
        continue
    f = line.rstrip('\n').split('\t')
    if f[0] == '1' and f[7] == 'no' and all(x.lstrip('-').isdigit() for x in f[11].split(',')):  # inline coefficients only
        charz[(int(f[1]), int(f[2]))] = [int(x) for x in f[11].split(',')][::-1]  # big-endian -> little-endian

# ---------------- K6 ----------------
def k6cert(n, ell, p, tag):
    M = J(tag)
    nt = len(M['torsion'])
    op = [o for o in M['ops'] if o['p'] == p][0]['matrix']
    C = [r[:nt] for r in op[:nt]]
    sig = M['torsion']
    chi = charz[(n, p)]
    P = charpoly_int(C)
    u, v, gd = pegcd(chi, P, ell)
    assert gd == [1], ('not separated', n, ell, p, gd)
    # nilpotency exponent: smallest m with P(C)^m e_j == 0 (row i mod ell^sig_i)
    E = max(sig)
    def apply(Pp, vec):
        # Horner, mod ell^E
        res = [0] * nt
        for a in reversed(Pp):
            res = [(a * vec[i] + sum(C[i][j] * res[j] for j in range(nt))) for i in range(nt)]
            res = [res[i] % ell ** sig[i] for i in range(nt)]
        return res
    for m in range(0, 40):
        ok = True
        for j in range(nt):
            vec = [int(i == j) for i in range(nt)]
            for _ in range(m):
                vec = apply(P, vec)
            if any(vec[i] % ell ** sig[i] for i in range(nt)):
                ok = False
        if ok:
            break
    return dict(ell=ell, p=p, chi=chi, sig=sig, C=C, P=P, m=m, u=u, v=v, n=n, tag=tag)

k6 = [k6cert(8, 2, 2, 'm8_l2')]

# ---------------- K7 ----------------
def inv_mod_matrix(A, q):
    n = len(A)
    M = [[A[i][j] % q for j in range(n)] + [int(i == j) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] % q and __import__('math').gcd(M[r][c], q) == 1)
        M[c], M[piv] = M[piv], M[c]
        iv = pow(M[c][c], -1, q)
        M[c] = [x * iv % q for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % q for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]

def k7cert(n, ell, tag, P):
    M = J(tag)
    assert len(M['torsion']) == 1
    e = M['torsion'][0]
    q = ell ** e
    r = M['free_rank']
    ops = {o['p']: o['matrix'] for o in M['ops']}
    B = [[[ops[p][1 + i][1 + j] % q for j in range(r)] for i in range(r)] for p in P]
    beta = [[ops[p][0][1 + j] % q for j in range(r)] for p in P]
    c = [ops[p][0][0] % q for p in P]
    for p in P:  # the torsion column is torsion
        assert all(ops[p][1 + i][0] % ell ** M['K'] == 0 for i in range(r))
    E = r * len(P)
    A = [[(B[col // r][i][col % r] - (c[col // r] if i == col % r else 0)) % q for col in range(E)] for i in range(r)]
    # Smith form over Z/ell^e with transforms:  Pm A Qm = D (D diagonal r x E), so A Qm[:, :r] = Pm^{-1} D[:, :r].
    def val(x):
        x %= q
        if x == 0:
            return e
        v = 0
        while x % ell == 0:
            x //= ell; v += 1
        return v
    D = [row[:] for row in A]
    Pm = [[int(i == j) for j in range(r)] for i in range(r)]
    Qm = [[int(i == j) for j in range(E)] for i in range(E)]
    for t in range(min(r, E)):
        best = None
        for i in range(t, r):
            for j in range(t, E):
                if D[i][j] % q and (best is None or val(D[i][j]) < best[0]):
                    best = (val(D[i][j]), i, j)
        if best is None:
            break
        _, i0, j0 = best
        D[t], D[i0] = D[i0], D[t]; Pm[t], Pm[i0] = Pm[i0], Pm[t]
        for row in D: row[t], row[j0] = row[j0], row[t]
        for row in Qm: row[t], row[j0] = row[j0], row[t]
        v = val(D[t][t]); u = (D[t][t] // ell ** v) % q; ui = pow(u, -1, q)
        for i in range(r):  # clear column t (row ops)
            if i != t and D[i][t] % q:
                f = (D[i][t] // ell ** v) * ui % q
                D[i] = [(x - f * y) % q for x, y in zip(D[i], D[t])]
                Pm[i] = [(x - f * y) % q for x, y in zip(Pm[i], Pm[t])]
        for j in range(E):  # clear row t (column ops)
            if j != t and D[t][j] % q:
                f = (D[t][j] // ell ** v) * ui % q
                for row in D: row[j] = (row[j] - f * row[t]) % q
                for row in Qm: row[j] = (row[j] - f * row[t]) % q
        # normalise the pivot to ell^v
        for row in D: row[t] = row[t] * ui % q
        for row in Qm: row[t] = row[t] * ui % q
    vals = [val(D[k][k]) if k < E else e for k in range(r)]
    W = [[Qm[col][k] for k in range(r)] for col in range(E)]
    Gi = [row[:] for row in Pm]
    G = inv_mod_matrix(Pm, q)
    for i in range(r):
        for k in range(r):
            assert (sum(A[i][col] * W[col][k] for col in range(E)) - G[i][k] * ell ** vals[k]) % q == 0, ('smith', n, i, k)
    rhs = [beta[col // r][col % r] for col in range(E)]
    import itertools
    sols = [h for h in itertools.product(range(q), repeat=r) if all(sum(h[i] * A[i][col] for i in range(r)) % q == 0 for col in range(E))] if q ** r <= 10 ** 7 else None
    hom = len(sols) if sols is not None else -1
    hsec, isSplit = None, False
    # particular solution of h A = rhs: h G D = rhs Qm  ->  y = hG with y_k ell^v_k = (rhs Qm)_k for k < r, (rhs Qm)_k = 0 for k >= r
    rq = [sum(rhs[col] * Qm[col][k] for col in range(E)) % q for k in range(E)]
    if all(rq[k] == 0 for k in range(r, E)) and all(rq[k] % ell ** vals[k] == 0 for k in range(r)):
        y = [(rq[k] // ell ** vals[k]) % q if vals[k] < e else 0 for k in range(r)]
        h = [sum(y[k] * Gi[k][i] for k in range(r)) % q for i in range(r)]
        if all((sum(h[i] * A[i][col] for i in range(r)) - rhs[col]) % q == 0 for col in range(E)):
            hsec, isSplit = h, True
    bound = ell ** sum(vals)
    return dict(n=n, ell=ell, e=e, r=r, P=P, B=B, beta=beta, c=c, isSplit=isSplit, h=hsec or [0] * r, W=W, G=G, Gi=Gi,
                vals=vals, bound=bound, hom=hom, tag=tag, K=M['K'])

P5 = [2, 3, 5, 7]
k7 = [k7cert(n, 5, 'm%d_l5' % n, P5) for n in (18, 20, 26, 30)]

# ---------------- K8 ----------------
def k8items(n, ell, tag):
    M = J(tag)
    nt = len(M['torsion'])
    items = []
    for i in range(nt):
        sig = M['torsion'][i]
        j = sig - 1
        res = []
        for o in M['ops']:
            if o['p'] == ell:
                continue
            row = o['matrix'][i][:nt]
            assert all(row[k] % ell ** sig == 0 for k in range(nt) if k != i), ('not diagonal', n, ell)
            res.append((o['p'], o['matrix'][i][i] % ell ** sig))
        if n % ell ** j:
            items.append(None); continue
        m = n // ell ** j
        found = None
        for a in range(0, m // (ell + 1) + 1):
            rem = m - a * (ell + 1)
            if rem % (ell * ell - ell):
                continue
            b = rem // (ell * ell - ell)
            if all(((1 + p) * pow(p, a * ell ** j, ell ** (j + 1))) % ell ** (j + 1) == rv for p, rv in res):
                found = (a, b); break
        items.append(dict(ell=ell, n=n, j=j, a=found[0] if found else 0, b=found[1] if found else 0, res=res, ok=found is not None, tag=tag, i=i))
    return items

k8 = []
for tag, n, ell in [('m8_l2', 8, 2), ('m6_l5', 6, 5), ('m18_l5', 18, 5), ('m20_l5', 20, 5), ('m26_l5', 26, 5), ('m30_l5', 30, 5),
                    ('m12_l3', 12, 3), ('m30_l3', 30, 3), ('m36_l3', 36, 3)]:
    k8 += [x for x in k8items(n, ell, tag) if x]

# ---------------- emit ----------------
L = []
w = L.append
w('/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/')
w('import GL2.HeckeZ.Exec.Bock\n')
w('/-!\n# HeckeZ M-HZ3: the K5, K6, K7, K8 certificates at level one (L-HZ2) — Mathlib-free, generated\n')
w('Generated by `lean/tools/hz_bock_to_lean.py` (untrusted; every claim below is re-checked by `decide +kernel`). Inputs, with the')
w('sha256 of every dump in the docstrings and in `docs/implementation/slot25/lhz2-data/SHA256SUMS`:')
w('* K5: the descent groups `descent(1, n, [2,3,5,7]).groups_inv` (the antecedent of B2) against the emitted `h^q` of the')
w('  Lean-kernel tables `docs/implementation/slot22/lv/lean_dims_l{2,3}.tsv` (the `dims_n` declarations of the L23 lane);')
w('* K6, K7, K8: **TB-mat**, the matrices of `descent(1, n, [2,3,5,7]).module(ell, K)` (slot-25 EXPORT, column convention, torsion')
w('  generators first); K6 also **TB-χ**, `char_Z` of the integral census ed.1.')
w('-/\n')
w('namespace GL2.HeckeZ.Cells.BockData\n')
w('open GL2.HeckeZ.Exec\n')
w('/-! ### K5: universal coefficients at `ℓ ∈ {2, 3}`, `n ∈ {8, 10, …, 60}` (54 cells, A3 §4.1) -/\n')
for (ell, n, b, mu, h, decl, mod, ok, dsha) in k5:
    nm = 'k5_l%d_n%d' % (ell, n)
    w('/-- K5 at `(1, %d)`, `ℓ = %d`; `h` from `%s.%s`; descent dump sha256 `%s`. -/' % (n, ell, mod, decl, dsha))
    w('def %s : K5Cert := { ell := %d, n := %d, b := %s, mu := %s, h := %s }\n' % (nm, ell, n, lst(b), mat(mu), lst(h)))
w('/-- The 54 K5 certificates. -/')
w('def k5All : List K5Cert := [' + ', '.join('k5_l%d_n%d' % (x[0], x[1]) for x in k5) + ']\n')
w("/-- **K5 on the 54 cells** (A3's re-check, 07 §10 M-HZ3: 54/54). -/")
w('theorem k5All_check : k5All.all k5Check = true := by decide +kernel\n')
w('/-! ### K6: separation at `(1, 8)`, `ℓ = 2`, `θ = T_2` -/\n')
for c in k6:
    w('/-- K6 at `(1, %d)`, `ℓ = %d`, `p = %d`: `χ_ℤ(T_%d | L)` (TB-χ, ed.1 `char_Z`), `tors_%d = %s` with `T_%d` on the export\'s torsion' % (c['n'], c['ell'], c['p'], c['p'], c['ell'], ' ⊕ '.join('ℤ/%d' % (c['ell'] ** s) for s in c['sig']), c['p']))
    w('generators (TB-mat, dump `%s.json` sha256 `%s`), `∏ P_j = %s` (computed, untrusted), Bezout over `𝔽_%d`. -/' % (c['tag'], sha(c['tag']), lst(c['P']), c['ell']))
    w('def k6_n%d : K6Cert :=' % c['n'])
    w('  { ell := %d, p := %d, chi := %s, sig := %s, C := %s, P := %s, m := %d, u := %s, v := %s }\n'
      % (c['ell'], c['p'], lst(c['chi']), lst(c['sig']), mat(c['C']), lst(c['P']), c['m'], lst(c['u']), lst(c['v'])))
    w('theorem k6_n%d_check : k6Check k6_n%d = true := by decide +kernel\n' % (c['n'], c['n']))
w('/-! ### K7: split and Hom bound at `ℓ = 5`, `𝕋\' = ℤ[T_2, T_3, T_5, T_7]` -/\n')
for c in k7:
    st = 'zero' if sum(c['vals']) == 0 else 'bounds'
    w('/-- K7 at `(1, %d)`, `ℓ = 5`: `tors_5 = ℤ/%d`, `r = %d`; export at `K = %d` (dump `%s.json` sha256 `%s`) reduced mod `5^%d`; split for `𝕋\'`: %s; `|Hom_𝕋\'(L/5^%d, tors)| ≤ %d` (exact count by the generator: %d); status `%s`. -/'
      % (c['n'], 5 ** c['e'], c['r'], c['K'], c['tag'], sha(c['tag']), c['e'], 'yes' if c['isSplit'] else 'no', c['e'], c['bound'], c['hom'], st))
    w('def k7_n%d : K7Cert :=' % c['n'])
    w('  { ell := 5, e := %d, r := %d, P := %s,' % (c['e'], c['r'], lst(c['P'])))
    w('    B := [%s],' % ', '.join(mat(x) for x in c['B']))
    w('    beta := %s, c := %s, isSplit := %s, h := %s,' % (mat(c['beta']), lst(c['c']), 'true' if c['isSplit'] else 'false', lst(c['h'])))
    w('    W := %s,' % mat(c['W']))
    w('    G := %s, Gi := %s, vals := %s }\n' % (mat(c['G']), mat(c['Gi']), lst(c['vals'])))
    w('theorem k7_n%d_check : k7Check k7_n%d = true := by decide +kernel\n' % (c['n'], c['n']))
w('/-! ### K8: Frobenius residues (B10(2); `orc:`) -/\n')
for x in k8:
    w('/-- K8 at `(1, %d)`, `ℓ = %d`, torsion summand %d of order `%d^%d` (dump `%s.json`): `a = %d`, `b = %d`%s. -/'
      % (x['n'], x['ell'], x['i'], x['ell'], x['j'] + 1, x['tag'], x['a'], x['b'], '' if x['ok'] else ' (NO MATCH)'))
    w('def k8_l%d_n%d_s%d : K8Item := { ell := %d, n := %d, j := %d, a := %d, b := %d, res := %s }\n'
      % (x['ell'], x['n'], x['i'], x['ell'], x['n'], x['j'], x['a'], x['b'], '[' + ', '.join('(%d, %d)' % pr for pr in x['res']) + ']'))
w('/-- The K8 items. -/')
w('def k8All : List K8Item := [' + ', '.join('k8_l%d_n%d_s%d' % (x['ell'], x['n'], x['i']) for x in k8) + ']\n')
w('theorem k8All_check : k8All.all k8Check = true := by decide +kernel\n')
w('end GL2.HeckeZ.Cells.BockData')
open(out, 'w').write('\n'.join(L) + '\n')

with open(manifest, 'w') as f:
    f.write('# L-HZ2 (M-HZ3) manifest, generated by lean/tools/hz_bock_to_lean.py; one row per certificate\n')
    f.write('checker\tcell\tell\tdetail\tverdict_generator\n')
    for (ell, n, b, mu, h, decl, mod, ok, dsha) in k5:
        f.write('K5\t(1,%d)\t%d\tb=%s mu=%s h=%s\t%s\n' % (n, ell, b, mu, h, 'match' if ok else 'MISMATCH'))
    for c in k6:
        f.write('K6\t(1,%d)\t%d\tp=%d chi=%s P=%s m=%d\tseparated\n' % (c['n'], c['ell'], c['p'], c['chi'], c['P'], c['m']))
    for c in k7:
        f.write('K7\t(1,%d)\t5\te=%d r=%d split=%s |Hom|<=%d (exact %d)\t%s\n' % (c['n'], c['e'], c['r'], c['isSplit'], c['bound'], c['hom'], 'zero' if sum(c['vals']) == 0 else 'bounds'))
    for x in k8:
        f.write('K8\t(1,%d)\t%d\tsummand %d j=%d a=%d b=%d res=%s\t%s\n' % (x['n'], x['ell'], x['i'], x['j'], x['a'], x['b'], x['res'], 'match' if x['ok'] else 'NO MATCH'))
print('K5', sum(1 for x in k5 if x[7]), '/', len(k5), 'K6', len(k6), 'K7', [(c['n'], c['isSplit'], c['bound'], c['hom']) for c in k7], 'K8', sum(1 for x in k8 if x['ok']), '/', len(k8))
