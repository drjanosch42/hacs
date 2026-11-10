#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""Untrusted generator of the M-HZ2 cell certificates (L-HZ1; 07-lean-plan.md §5.1, §5.2, §10; [LP] v1.9 data route).

Reads
  * the integral census ed.1 data set (`integral_census_dataset/v1`): `tables/charz.tsv.gz` and
    `external/charz_big.tsv.gz` (TB-chi: the integral characteristic polynomials of T_2, T_3, T_5 on H^1/tors),
    `tables/torsion.tsv` (the torsion primes of H^1: global K1 is refused there, K1m is marked "blockwise");
    or, instead of the data set, the committed level-1 extracts in `docs/implementation/slot25/lhz1-data/`
    (`ed1_charz_level1_n_le_62.tsv`, `ed1_charz_big_level1_n_le_62.tsv`, `ed1_torsion_level1_n_le_62.tsv`);
  * the `HECKE_DUMP_SAT_CERT` dumps `satcert_n<n>_ell<l>_b<k>.txt` (TB-mat), assembled block-diagonally when the
    free ranks of the stable blocks add up to b1;
and writes Mathlib-free Lean data modules (`GL2.HeckeZ.Exec.CellCert` values with `decide +kernel` checks) and a
manifest TSV. Everything it computes (f, the cofactor q, the Bezout pair, words, pivots, inverses, cofactor) is
re-checked by the kernel; only the TB-chi coefficients and the dumped matrices are trusted (named in the theorems).

Usage: hz_descent_to_lean.py <ed1_dataset_dir> <dump_dir> <out_dir> <cells n,...> <module_spec>
  module_spec: Name:n1,n2,...;Name2:...   (one Lean module per name)
"""
import gzip, hashlib, os, sys, itertools
from fractions import Fraction

sys.setrecursionlimit(10000)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_charz(ds, ns):
    big = {}
    with gzip.open(os.path.join(ds, 'external/charz_big.tsv.gz'), 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('N\t'):
                continue
            N, n, p, i, sign, bits, dd, h = line.rstrip('\n').split('\t')
            if N != '1' or int(n) not in ns:
                continue
            v = int(h.replace('-', ''), 16) * (-1 if h.startswith('-') else 1)
            big[(int(n), int(p), int(i))] = v
    chis = {}
    with gzip.open(os.path.join(ds, 'tables/charz.tsv.gz'), 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('N\t'):
                continue
            fl = line.rstrip('\n').split('\t')
            if fl[0] != '1' or int(fl[1]) not in ns:
                continue
            n, p = int(fl[1]), int(fl[2])
            assert fl[4] == 'yes' and fl[7] == 'no', ('undetermined or refused', n, p)
            vals = []
            for i, c in enumerate(fl[11].split(',')):
                vals.append(big[(n, p, i)] if c.endswith('d') else int(c))
            chis[(n, p)] = vals  # big-endian, monic
    return chis


def read_charz_extract(dd, ns):
    """The same TB-chi from the committed level-1 extracts of `docs/implementation/slot25/lhz1-data/` (M-HZ2 audit F4)."""
    big = {}
    for line in open(os.path.join(dd, 'ed1_charz_big_level1_n_le_62.tsv')):
        if line.startswith('#') or line.startswith('N\t'):
            continue
        N, n, p, i, sign, bits, d_, h = line.rstrip('\n').split('\t')
        if N == '1' and int(n) in ns:
            big[(int(n), int(p), int(i))] = int(h.replace('-', ''), 16) * (-1 if h.startswith('-') else 1)
    chis = {}
    for line in open(os.path.join(dd, 'ed1_charz_level1_n_le_62.tsv')):
        if line.startswith('#') or line.startswith('N\t'):
            continue
        fl = line.rstrip('\n').split('\t')
        if fl[0] != '1' or int(fl[1]) not in ns:
            continue
        n, p = int(fl[1]), int(fl[2])
        assert fl[4] == 'yes' and fl[7] == 'no', ('undetermined or refused', n, p)
        chis[(n, p)] = [big[(n, p, i)] if c.endswith('d') else int(c) for i, c in enumerate(fl[11].split(','))]
    return chis


def is_extract(ds):
    return os.path.exists(os.path.join(ds, 'ed1_charz_level1_n_le_62.tsv'))


def read_torsion(ds):
    tors = {}
    path = os.path.join(ds, 'ed1_torsion_level1_n_le_62.tsv') if is_extract(ds) else os.path.join(ds, 'tables/torsion.tsv')
    for line in open(path):
        if line.startswith('#') or line.startswith('N\t'):
            continue
        f = line.split('\t')
        if f[0] == '1' and f[2] == '1' and f[4] != '-':
            tors.setdefault(int(f[1]), set()).add(int(f[3]))
    return tors

# ---------- polynomials (little-endian lists of ints / Fractions) ----------


def trim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def padd(a, b):
    n = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def pmul(a, b):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] += x * y
    return trim(r)


def pscale(c, a):
    return trim([c * x for x in a])


def pderiv(a):
    return trim([i * a[i] for i in range(1, len(a))])


def pdivmod(a, b):
    a = [Fraction(x) for x in a]
    b = trim([Fraction(x) for x in b])
    q = [Fraction(0)] * max(len(a) - len(b) + 1, 1)
    a = trim(a)
    while len(a) >= len(b) and a:
        c = a[-1] / b[-1]
        k = len(a) - len(b)
        q[k] = c
        a = trim([a[i] - (c * b[i - k] if 0 <= i - k < len(b) else 0) for i in range(len(a))])
    return trim(q), a


def pgcd(a, b):
    a, b = trim([Fraction(x) for x in a]), trim([Fraction(x) for x in b])
    while b:
        _, r = pdivmod(a, b)
        a, b = b, r
    return [x / a[-1] for x in a]


def gcdex(a, b):
    """s a + t b = g (monic gcd) over Q."""
    r0, r1 = trim([Fraction(x) for x in a]), trim([Fraction(x) for x in b])
    s0, s1, t0, t1 = [Fraction(1)], [], [], [Fraction(1)]
    while r1:
        q, r = pdivmod(r0, r1)
        r0, r1 = r1, r
        s0, s1 = s1, padd(s0, pscale(-1, pmul(q, s1)))
        t0, t1 = t1, padd(t0, pscale(-1, pmul(q, t1)))
    lc = r0[-1]
    return [x / lc for x in s0], [x / lc for x in t0], [x / lc for x in r0]


def lcm(a, b):
    from math import gcd
    return a // gcd(a, b) * b


def k2_item(chi_be, p, d):
    chi = list(reversed(chi_be))  # little-endian, monic
    g = pgcd(chi, pderiv(chi))
    f, r = pdivmod(chi, g)
    assert not r
    assert all(x.denominator == 1 for x in f) and f[-1] == 1, 'squarefree part not integral monic'
    f = [int(x) for x in f]
    assert len(f) == d + 1, ('deg f', p, len(f) - 1, d)
    q, r = pdivmod(chi, f)
    assert not r and all(x.denominator == 1 for x in q)
    q = [int(x) for x in q]
    s, t, one = gcdex(f, pderiv(f))
    assert one == [1], 'f not separable over Q'
    D = 1
    for x in s + t:
        D = lcm(D, x.denominator)
    u = [int(x * D) for x in s]
    v = [int(x * D) for x in t]
    assert padd(pmul(u, f), pmul(v, pderiv(f))) == [D]
    assert pmul(f, q) == chi
    return dict(p=p, chi=chi, f=f, q=q, u=u, v=v, D=D)

# ---------- dumps ----------


def read_dump(path):
    gens, cur, meta = [], None, {}
    for line in open(path):
        w = line.split()
        if not w or w[0].startswith('#'):
            continue
        if w[0] == 'gen':
            cur = (int(w[1]), [])
            gens.append(cur)
        elif w[0] == 'row':
            cur[1].append([int(x) for x in w[1:]])
        elif w[0] in ('n', 'ell', 'K', 'f'):
            meta[w[0]] = int(w[1])
        elif w[0] == 'verdict':
            meta['verdict'] = w[1]
    return meta, gens


def assemble(dump_dir, n, ell, b1):
    files = sorted(f for f in os.listdir(dump_dir) if f.startswith('satcert_n%d_ell%d_b' % (n, ell)))
    if not files:
        return None, [], 'no_dump'
    blocks = []
    for fn in files:
        meta, gens = read_dump(os.path.join(dump_dir, fn))
        assert meta['verdict'] == 'stable'
        blocks.append((fn, meta, gens))
    if sum(m['f'] for _, m, _ in blocks) != b1:
        return None, files, 'blocks_do_not_cover(sum_f=%d,b1=%d)' % (sum(m['f'] for _, m, _ in blocks), b1)
    primes = [g for g, _ in blocks[0][2]]
    for _, _, gens in blocks:
        assert [g for g, _ in gens] == primes
    mats = []
    for gi, g in enumerate(primes):
        M = [[0] * b1 for _ in range(b1)]
        off = 0
        for _, meta, gens in blocks:
            f = meta['f']
            A = gens[gi][1]
            for i in range(f):
                for j in range(f):
                    M[off + i][off + j] = A[i][j] % ell
            off += f
        mats.append(M)
    return (primes, mats), files, 'ok'


def matmul(A, B, q):
    n = len(A)
    return [[sum(A[i][l] * B[l][j] for l in range(n)) % q for j in range(n)] for i in range(n)]


def rank_extend(basis_rows, v, q):
    """Gaussian elimination mod prime q: return reduced basis if v is independent, else None."""
    v = list(v)
    for (piv, row) in basis_rows:
        if v[piv]:
            c = v[piv]
            v = [(x - c * y) % q for x, y in zip(v, row)]
    for i, x in enumerate(v):
        if x:
            inv = pow(x, q - 2, q)
            v = [(y * inv) % q for y in v]
            new = []
            for (piv, row) in basis_rows:
                if row[i]:
                    c = row[i]
                    row = [(a - c * b) % q for a, b in zip(row, v)]
                new.append((piv, row))
            return new + [(i, v)]
    return None


def matinv(C, q):
    d = len(C)
    A = [list(r) + [1 if i == j else 0 for j in range(d)] for i, r in enumerate(C)]
    for col in range(d):
        piv = next(r for r in range(col, d) if A[r][col] % q)
        A[col], A[piv] = A[piv], A[col]
        inv = pow(A[col][col], q - 2, q)
        A[col] = [(x * inv) % q for x in A[col]]
        for r in range(d):
            if r != col and A[r][col]:
                c = A[r][col]
                A[r] = [(x - c * y) % q for x, y in zip(A[r], A[col])]
    return [row[d:] for row in A]


def k1_cert(primes, mats, q, b1, d, prefer=(2, 3, 5)):
    for allowed in (prefer, tuple(primes)):
        idx = [i for i, g in enumerate(primes) if g in allowed]
        I = [[1 if i == j else 0 for j in range(b1)] for i in range(b1)]
        basis, words, wmats = [], [], []
        frontier = [((), I)]
        seen = set()
        deg = 0
        while frontier and len(words) < d and deg <= 2 * d + 2:
            nxt = []
            for w, M in frontier:
                flat = [x for r in M for x in r]
                nb = rank_extend(basis, flat, q)
                if nb is not None:
                    basis = nb
                    words.append(list(w))
                    wmats.append(M)
                    if len(words) == d:
                        break
                for j in idx:
                    w2 = tuple(sorted(w + (j,)))
                    if w2 in seen:
                        continue
                    seen.add(w2)
                    nxt.append((w2, matmul(mats[j], M, q)))
            frontier = nxt
            deg += 1
        if len(words) == d:
            # words as products in list order: M_{w0} M_{w1} ... I ; recompute in that order
            ws = []
            for w in words:
                M = I
                for j in reversed(w):
                    M = matmul(mats[j], M, q)
                ws.append(M)
            # pivot columns: row-reduce the d x b1^2 matrix
            rows = [[x for r in M for x in r] for M in ws]
            cols, red = [], [list(r) for r in rows]
            # choose d independent columns greedily
            colbasis = []
            for c in range(b1 * b1):
                colv = [red[k][c] for k in range(d)]
                nb = rank_extend(colbasis, colv, q)
                if nb is not None:
                    colbasis = nb
                    cols.append(c)
                    if len(cols) == d:
                        break
            assert len(cols) == d
            pos = [(c // b1, c % b1) for c in cols]
            C = [[ws[k][i][j] for (i, j) in pos] for k in range(d)]
            Ci = matinv(C, q)
            used = sorted(set(j for w in words for j in w))
            return dict(q=q, gens=primes, mats=mats, words=words, pos=pos, cinv=Ci, used=[primes[j] for j in used])
    return None


def is_probable_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def rho(n, budget=200000):
    from math import gcd
    if n % 2 == 0:
        return 2
    for c in range(1, 20):
        y, g, q, x, it = 2, 1, 1, 2, 0
        m = 128
        r = 1
        while g == 1 and it < budget:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += m
                it += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if 1 < g < n:
            return g
    return None


# ---------- polynomials mod a prime (little-endian lists of residues) ----------


def mtrim(a):
    a = list(a)
    while a and a[-1] == 0:
        a.pop()
    return a


def mpadd(a, b, q):
    n = max(len(a), len(b))
    return mtrim([((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % q for i in range(n)])


def mpmul(a, b, q):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = (r[i + j] + x * y) % q
    return mtrim(r)


def mpdivmod(a, b, q):
    a, b = mtrim([x % q for x in a]), mtrim([x % q for x in b])
    inv = pow(b[-1], q - 2, q)
    out = [0] * max(len(a) - len(b) + 1, 1)
    while len(a) >= len(b) and a:
        c = a[-1] * inv % q
        k = len(a) - len(b)
        out[k] = c
        a = mtrim([(a[i] - c * (b[i - k] if 0 <= i - k < len(b) else 0)) % q for i in range(len(a))])
    return mtrim(out), a


def mpgcdex(a, b, q):
    r0, r1 = mtrim([x % q for x in a]), mtrim([x % q for x in b])
    s0, s1, t0, t1 = [1], [], [], [1]
    while r1:
        qq, r = mpdivmod(r0, r1, q)
        r0, r1 = r1, r
        s0, s1 = s1, mpadd(s0, [(-x) % q for x in mpmul(qq, s1, q)], q)
        t0, t1 = t1, mpadd(t0, [(-x) % q for x in mpmul(qq, t1, q)], q)
    inv = pow(r0[-1], q - 2, q)
    return [x * inv % q for x in s0], [x * inv % q for x in t0], [x * inv % q for x in r0]


def mpderiv(a, q):
    return mtrim([i * a[i] % q for i in range(1, len(a))])


def mprad(a, q):
    g = mpgcdex(a, mpderiv(a, q), q)[2]
    r, rem = mpdivmod(a, g, q)
    assert not rem
    return r


def charpoly_mod(M, q):
    """det(x I - M) mod q by interpolation at n+1 points (q > n + 1)."""
    n = len(M)
    def det(A):
        A = [list(r) for r in A]
        d = 1
        for c in range(n):
            piv = next((r for r in range(c, n) if A[r][c] % q), None)
            if piv is None:
                return 0
            if piv != c:
                A[c], A[piv] = A[piv], A[c]
                d = -d
            d = d * A[c][c] % q
            inv = pow(A[c][c], q - 2, q)
            for r in range(c + 1, n):
                if A[r][c]:
                    f = A[r][c] * inv % q
                    A[r] = [(x - f * y) % q for x, y in zip(A[r], A[c])]
        return d % q
    xs = list(range(n + 1))
    ys = [det([[((t if i == j else 0) - M[i][j]) % q for j in range(n)] for i in range(n)]) for t in xs]
    # Lagrange
    res = []
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        num, den = [1], 1
        for j, xj in enumerate(xs):
            if j != i:
                num = mpmul(num, [(-xj) % q, 1], q)
                den = den * (xi - xj) % q
        res = mpadd(res, [c * yi * pow(den, q - 2, q) % q for c in num], q)
    return res


def max_words(primes, mats, q, f, prefer=(2, 3, 5), maxdeg=None):
    idx = [i for i, g in enumerate(primes) if g in prefer] or list(range(len(primes)))
    I = [[1 if i == j else 0 for j in range(f)] for i in range(f)]
    basis, words = [], []
    frontier, seen = [((), I)], set()
    deg = 0
    maxdeg = maxdeg if maxdeg is not None else f + 1
    while frontier and deg <= maxdeg:
        nxt = []
        for w, M in frontier:
            nb = rank_extend(basis, [x for r in M for x in r], q)
            if nb is not None:
                basis = nb
                words.append(list(w))
            for j in idx:
                w2 = tuple(sorted(w + (j,)))
                if w2 in seen:
                    continue
                seen.add(w2)
                nxt.append((w2, matmul(mats[j], M, q)))
        frontier = nxt
        deg += 1
    return words


def k1m_cert(dump_dir, n, ell, b1, d, items):
    files = sorted(f for f in os.listdir(dump_dir) if f.startswith('satcert_n%d_ell%d_b' % (n, ell)))
    if not files:
        return None, 'no_dump'
    blocks = []
    for fn in files:
        meta, gens = read_dump(os.path.join(dump_dir, fn))
        assert meta['verdict'] == 'stable', (fn, meta.get('verdict'))  # M-HZ2 audit F6
        blocks.append((fn, meta, gens))
    primes = [g for g, _ in blocks[0][2]]
    fB = sum(m['f'] for _, m, _ in blocks)
    mats = []
    for gi in range(len(primes)):
        M = [[0] * fB for _ in range(fB)]
        off = 0
        for _, meta, gens in blocks:
            f = meta['f']
            for i in range(f):
                for j in range(f):
                    M[off + i][off + j] = gens[gi][1][i][j] % ell
            off += f
        mats.append(M)
    if ell <= fB + 1:
        return None, 'ell_too_small_for_interpolation'
    words = max_words(primes, mats, ell, fB)
    rB = len(words)
    I = [[1 if i == j else 0 for j in range(fB)] for i in range(fB)]
    ws = []
    for w in words:
        M = I
        for j in reversed(w):
            M = matmul(mats[j], M, ell)
        ws.append(M)
    cols, colbasis = [], []
    for c in range(fB * fB):
        nb = rank_extend(colbasis, [ws[k][c // fB][c % fB] for k in range(rB)], ell)
        if nb is not None:
            colbasis = nb
            cols.append(c)
            if len(cols) == rB:
                break
    pos = [(c // fB, c % fB) for c in cols]
    C = [[ws[k][i][j] for (i, j) in pos] for k in range(rB)]
    Ci = matinv(C, ell) if rB else []
    for th, g in enumerate(primes):
        it = next((it for it in items if it['p'] == g), None)
        if it is None:
            continue
        chi = [x % ell for x in it['chi']]
        cpB = charpoly_mod(mats[th], ell)
        f1 = mprad(cpB, ell)
        radchi = mprad(chi, ell)
        gg = mpgcdex(radchi, f1, ell)[2]
        f2, rem = mpdivmod(radchi, gg, ell)
        assert not rem
        if rB + len(f2) - 1 != d:
            continue
        c2, rem = mpdivmod(chi, f2, ell)
        if rem:
            continue
        u2, v2, one = mpgcdex(f2, mpderiv(f2, ell), ell)
        s1, t1, one2 = mpgcdex(f2, f1, ell)
        if one != [1] or one2 != [1]:
            continue
        return dict(q=ell, fB=fB, gens=primes, mats=mats, words=words, pos=pos, cinv=Ci, th=th, chi=it['chi'],
                    f1=f1, m=fB, f2=f2, c2=c2, u2=u2, v2=v2, s1=s1, t1=t1,
                    files=[(fn, sha256(os.path.join(dump_dir, fn))) for fn in files],
                    used=sorted(set(primes[j] for w in words for j in w) | {g})), None
    return None, 'k1m_count(rB=%d,fB=%d)' % (rB, fB)


def factor_small(G, limit):
    """Untrusted finder: trial division to `limit`, then Pollard rho (Brent) on word-size cofactors; returns
    (factors of primes < 2^31 with exponents, unfactored rest)."""
    fac = {}
    for p in [2, 3, 5, 7]:
        while G % p == 0:
            fac[p] = fac.get(p, 0) + 1
            G //= p
    p = 11
    while p <= limit and p * p <= G:
        while G % p == 0:
            fac[p] = fac.get(p, 0) + 1
            G //= p
        p += 2
    stack, rest = [G], 1
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_probable_prime(m):
            if m < 2 ** 31:
                fac[m] = fac.get(m, 0) + 1
            else:
                rest *= m
            continue
        if m.bit_length() > 128:
            rest *= m
            continue
        g = rho(m)
        if g is None:
            rest *= m
        else:
            stack += [g, m // g]
    return fac, rest


def lean_list(xs):
    return '[' + ', '.join(str(x) for x in xs) + ']'


def lean_mat(M):
    return '[' + ', '.join(lean_list(r) for r in M) + ']'


THEOREM_HEADER = """
/-!
# HeckeZ S1: 01 Thm. 4.1 at level 1 on T1, T2 and the 29 cells of C1, `n ≤ 62` (L-HZ1, M-HZ2)

GENERATED by `lean/tools/hz_descent_to_lean.py` — do not edit by hand. `07-lean-plan.md` §6 (S1 = {T1, T2} ∪ C1), §10 (M-HZ2).
One theorem per cell, each an instance of `GL2.HeckeZ.Check.cell_sound_ES` on the kernel-checked certificate `cert_n<n>`:
`L` is `H¹(SL₂(ℤ); Sym^n ℤ)/tors` and `X m` the Hecke operator `T_m` on it (interface data, as in `GL2.HeckeZ.T1`); the
conclusion is `𝕋^fr = ℤ[T_p : p ∈ P]` (cofactor `1`) or `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime not dividing the stated cofactor.
**Trusted C++ outputs ([LP] v1.9, Q10(a)), in every statement:** `hχ` (tb_chi: the integral characteristic polynomials of
`T_2, T_3, T_5` of the integral census ed.1) and `hmat` (tb_mat: the `HECKE_DUMP_SAT_CERT` stable blocks of each K1
certificate, block-diagonally assembled); the literature hypothesis is `hES` (`lit:eichler-shimura-levelone`, any `p`).
The primes not covered (`2`, `3`, the torsion primes of `H¹`, primes whose blocks were not all dumped as `stable`, and
factors beyond the trial-division finder) are listed per cell; they divide the cofactor, so the statement says nothing there.
-/

"""


def main():
    ds, dump_dirs_s, out_dir, cells_s, modspec = sys.argv[1:6]
    dump_dirs = dump_dirs_s.split(',')
    ns = [int(x) for x in cells_s.split(',')]
    chis = read_charz_extract(ds, set(ns)) if is_extract(ds) else read_charz(ds, set(ns))
    tors = read_torsion(ds)
    manifest = []
    certs = {}
    for n in ns:
        b1 = len(chis[(n, 2)]) - 1
        d = (b1 + 1) // 2
        items = [k2_item(chis[(n, p)], p, d) for p in (2, 3, 5)]
        from math import gcd
        G = 0
        for it in items:
            G = gcd(G, abs(it['D']))
        # candidate primes for K1: the prime factors of G found by trial division (an untrusted finder)
        fac, rest = factor_small(G, 2 * 10 ** 6) if G > 1 else ({}, 1)
        k1s, k1ms, uncovered = [], [], {}
        for ell, e in sorted(fac.items()):
            why = None
            ddir = next((dd for dd in dump_dirs if any(f.startswith('satcert_n%d_ell%d_b' % (n, ell))
                                                         for f in os.listdir(dd))), dump_dirs[0])
            if ell in (2, 3):
                why = 'ell_in_{2,3}(no dump)'
            else:
                # M-HZ2 audit F6: global K1 reads H^1(V/ell) as L/ell, which fails at a torsion prime; refused there.
                # K1m stays allowed (a torsion-free block, 01 §4 "blockwise"); its docstring says so.
                A, files, st = assemble(ddir, n, ell, b1)
                if A is not None and ell in tors.get(n, set()):
                    A = None
                if A is not None:
                    cert = k1_cert(A[0], A[1], ell, b1, d)
                    if cert is not None:
                        cert['files'] = [(fn, sha256(os.path.join(ddir, fn))) for fn in files]
                        cert['e'] = e
                        k1s.append(cert)
                        continue
                if st == 'no_dump':
                    why = 'no_dump'
                else:
                    cm, whym = k1m_cert(ddir, n, ell, b1, d, items)
                    if cm is None:
                        why = whym
                    else:
                        cm['e'] = e
                        cm['torsion_prime'] = ell in tors.get(n, set())
                        k1ms.append(cm)
            if why:
                uncovered[ell] = (e, why + ('; torsion prime' if ell in tors.get(n, set()) else ''))
        cof = G
        ex = 0
        for c in k1s + k1ms:
            while cof % c['q'] == 0:
                cof //= c['q']
            ex = max(ex, c['e'])
        qprod = 1
        for c in k1s + k1ms:
            qprod *= c['q']
        assert (cof * qprod ** ex) % G == 0
        # restrict every certificate to the generators it uses (TB-mat is then stated for those only)
        for c in k1s + k1ms:
            keep = [j for j, g in enumerate(c['gens']) if g in c['used']]
            remap = {j: t for t, j in enumerate(keep)}
            c['gens'] = [c['gens'][j] for j in keep]
            c['mats'] = [c['mats'][j] for j in keep]
            c['words'] = [[remap[j] for j in w] for w in c['words']]
            if 'th' in c:
                c['th'] = remap[c['th']]
        P = sorted(set([2, 3, 5] + [g for c in k1s + k1ms for g in c['gens']]))
        # restrict each K1 to the generators actually used? keep the full dump for TB-mat fidelity
        certs[n] = dict(n=n, b1=b1, d=d, P=P, items=items, k1=k1s, k1m=k1ms, cof=cof, ex=ex, G=G,
                        uncovered=uncovered, rest=rest)
        manifest.append((n, b1, d, G.bit_length(), ','.join(str(c['q']) for c in k1s) or '-',
                         ','.join(str(c['q']) for c in k1ms) or '-',
                         ';'.join('%s^%d:%s' % (l, e, w) for l, (e, w) in sorted(uncovered.items())) or '-',
                         '%d' % rest.bit_length() if rest > 1 else '-', cof == 1))
    # modules
    for part in modspec.split(';'):
        name, cs = part.split(':')
        cns = [int(x) for x in cs.split(',')]
        path = os.path.join(out_dir, name + '.lean')
        with open(path, 'w') as o:
            o.write('/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/\n')
            o.write('import GL2.HeckeZ.Exec.Cert\n\n')
            o.write('/-! GENERATED by `lean/tools/hz_descent_to_lean.py` (L-HZ1, M-HZ2) — do not edit by hand.\n\n')
            o.write('Level-1 cell certificates for 01 Thm. 4.1 (`GL2.HeckeZ.Exec.CellCert`), Mathlib-free, checked by `decide +kernel`.\n')
            o.write('Trusted inputs (named in `GL2.HeckeZ.Check.TBchi` / `TBmat`): the `chi` lists are the integral characteristic\n')
            o.write('polynomials `char_Z` of the integral census ed.1 (`charz.tsv.gz` + `charz_big.tsv.gz`, sha256 in\n')
            o.write('`docs/implementation/slot25/lhz1-data/SHA256SUMS`); the `mats` of each K1 certificate are the `HECKE_DUMP_SAT_CERT`\n')
            o.write('stable blocks (file names and sha256 in the docstrings), assembled block-diagonally and reduced mod `q`.\n')
            o.write('Everything else (`f`, `q`, `u`, `v`, `D`, words, pivots, inverses, `cof`, `ex`) is computed by the untrusted\n')
            o.write('generator and re-checked by the kernel. -/\n\n')
            o.write('namespace GL2.HeckeZ.Cells.%s\n\nopen GL2.HeckeZ.Exec\n\n' % name)
            for n in cns:
                c = certs[n]
                unc = ', '.join('%s^%d (%s)' % (l, e, w) for l, (e, w) in sorted(c['uncovered'].items()))
                o.write('/-- Cell `(1, %d)` (weight %d): `b1 = %d`, `d = %d`; `gcd D` has %d bits; K1 at %s; K1m at %s; ' %
                        (n, n + 2, c['b1'], c['d'], c['G'].bit_length(), ', '.join(str(k['q']) for k in c['k1']) or 'none',
                         ', '.join(str(k['q']) for k in c['k1m']) or 'none'))
                o.write('not covered: %s%s; `cof = 1`: %s.\n' % (unc or 'none',
                        (', unfactored part of %d bits' % c['rest'].bit_length()) if c['rest'] > 1 else '', c['cof'] == 1))
                for k in c['k1']:
                    o.write('K1 at %d: dumps %s. ' % (k['q'], ', '.join('`%s` (sha256 `%s`)' % (fn, h) for fn, h in k['files'])))
                for k in c['k1m']:
                    o.write('K1m at %d (dumped rank %d of %d%s): dumps %s. ' % (k['q'], k['fB'], c['b1'],
                            '; blockwise at a torsion prime' if k.get('torsion_prime') else '',
                            ', '.join('`%s` (sha256 `%s`)' % (fn, h) for fn, h in k['files'])))
                o.write('-/\n')
                o.write('def cert_n%d : CellCert where\n' % n)
                o.write('  n := %d\n  b1 := %d\n  d := %d\n  P := %s\n' % (n, c['b1'], c['d'], lean_list(c['P'])))
                o.write('  items := [\n')
                o.write(',\n'.join('    { p := %d, chi := %s,\n      f := %s,\n      q := %s,\n      u := %s,\n      v := %s,\n      D := %d }'
                                   % (it['p'], lean_list(it['chi']), lean_list(it['f']), lean_list(it['q']),
                                      lean_list(it['u']), lean_list(it['v']), it['D']) for it in c['items']))
                o.write(' ]\n')
                o.write('  k1 := [')
                o.write(',\n'.join('\n    { q := %d, gens := %s,\n      mats := %s,\n      words := %s, pos := %s,\n      cinv := %s }'
                                   % (k['q'], lean_list(k['gens']), '[' + ', '.join(lean_mat(M) for M in k['mats']) + ']',
                                      '[' + ', '.join(lean_list(w) for w in k['words']) + ']',
                                      '[' + ', '.join('(%d, %d)' % ij for ij in k['pos']) + ']', lean_mat(k['cinv']))
                                   for k in c['k1']))
                o.write(' ]\n')
                o.write('  k1m := [')
                o.write(',\n'.join('\n    { q := %d, fB := %d, gens := %s,\n      mats := %s,\n      words := %s, pos := %s,\n      cinv := %s,\n      th := %d, chi := %s,\n      f1 := %s, m := %d,\n      f2 := %s,\n      c2 := %s,\n      u2 := %s, v2 := %s,\n      s1 := %s, t1 := %s }'
                                   % (k['q'], k['fB'], lean_list(k['gens']), '[' + ', '.join(lean_mat(M) for M in k['mats']) + ']',
                                      '[' + ', '.join(lean_list(w) for w in k['words']) + ']',
                                      '[' + ', '.join('(%d, %d)' % ij for ij in k['pos']) + ']', lean_mat(k['cinv']),
                                      k['th'], lean_list(k['chi']), lean_list(k['f1']), k['m'], lean_list(k['f2']),
                                      lean_list(k['c2']), lean_list(k['u2']), lean_list(k['v2']), lean_list(k['s1']),
                                      lean_list(k['t1']))
                                   for k in c['k1m']))
                o.write(' ]\n')
                o.write('  cof := %d\n  ex := %d\n\n' % (c['cof'], c['ex']))
                o.write('theorem cert_n%d_check : cellCheck cert_n%d = true := by decide +kernel\n\n' % (n, n))
            o.write('end GL2.HeckeZ.Cells.%s\n' % name)
        print('wrote', path, os.path.getsize(path))
    # the theorem module (Mathlib): one theorem per cell
    mods = [(part.split(':')[0], [int(x) for x in part.split(':')[1].split(',')]) for part in modspec.split(';')]
    tpath = os.path.join(out_dir, 'S1.lean')
    with open(tpath, 'w') as o:
        o.write('/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/\n')
        o.write('import Mathlib\nimport GL2.HeckeZ.Check.Cert\n')
        for name, _ in mods:
            o.write('import GL2.HeckeZ.Cells.%s\n' % name)
        o.write(THEOREM_HEADER)
        o.write('namespace GL2.HeckeZ.Cells.S1\n\nopen GL2.Literature GL2.HeckeZ.Exec GL2.HeckeZ.Check GL2.HeckeZ.Cert GL2.HeckeZ.Bridge TensorProduct\n')
        for name, cns in mods:
            o.write('open GL2.HeckeZ.Cells.%s\n' % name)
        o.write('\nvariable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L] {p : ℕ} [Fact p.Prime]\n')
        o.write('  {E : Type*} [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]\n\n')
        for name, cns in mods:
            for n in cns:
                c = certs[n]
                b1 = c['b1']
                P = lean_list(c['P'])
                unc = ', '.join('%s^%d (%s)' % (l, e, w) for l, (e, w) in sorted(c['uncovered'].items()))
                o.write('/-- **Cell `(1, %d)`, weight %d** (01 Thm. 4.1, M-HZ2): `rank L = %d`, `d = %d`, `𝕋\' = ℤ[T_p : p ∈ %s]`. ' % (n, n + 2, b1, c['d'], P))
                if c['cof'] == 1:
                    o.write('**`𝕋^fr = 𝕋\'`** (cofactor `1`). ')
                else:
                    o.write('`𝕋\'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof = %d`; not covered: %s%s. ' % (
                        c['cof'], unc, (', an unfactored part of %d bits' % c['rest'].bit_length()) if c['rest'] > 1 else ''))
                o.write('Trusted: `hχ` (ed.1 `char_Z` of `T_2, T_3, T_5`), `hmat` / `hmatm` (the `HECKE_DUMP_SAT_CERT` blocks named in `cert_n%d`); literature: `hES`. -/\n' % n)
                o.write('theorem cell_n%d (X : ℕ → Module.End ℤ L) (hrk : Module.finrank ℤ L = %d)\n' % (n, b1))
                o.write('    (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E)\n')
                o.write('    (hχ : ∀ it ∈ cert_n%d.items, (X it.p).charpoly = toPoly it.chi)\n' % n)
                o.write('    (hmat : ∀ k ∈ cert_n%d.k1, ∃ (b : Module.Basis (Fin %d) ℤ L) (U : Matrix (Fin %d) (Fin %d) (ZMod k.q)), IsUnit U ∧\n' % (n, b1, b1, b1))
                o.write('      ∀ j < k.gens.length,\n')
                o.write('        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) = lmat %d k.q (k.mats.getD j []) * U)\n' % b1)
                o.write('    (hmatm : ∀ k ∈ cert_n%d.k1m, ∃ (b : Module.Basis (Fin k.fB ⊕ Fin (%d - k.fB)) ℤ L)\n' % (n, b1))
                o.write('      (U : Matrix (Fin k.fB ⊕ Fin (%d - k.fB)) (Fin k.fB ⊕ Fin (%d - k.fB)) (ZMod k.q)), IsUnit U ∧\n' % (b1, b1))
                o.write('      ∀ j < k.gens.length, ∃ Y : Matrix (Fin (%d - k.fB)) (Fin (%d - k.fB)) (ZMod k.q),\n' % (b1, b1))
                o.write('        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) =\n')
                o.write('          Matrix.fromBlocks (lmat k.fB k.q (k.mats.getD j [])) 0 0 Y * U) :\n')
                if c['cof'] == 1:
                    o.write('    Tfr X = Tp X %s :=\n' % P)
                    o.write('  (cell_sound_ES cert_n%d cert_n%d_check X hrk hES hχ hmat hmatm).2 rfl\n\n' % (n, n))
                else:
                    o.write('    ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ %d → Loc (Tfr X) (Tp X %s) ℓ :=\n' % (c['cof'], P))
                    o.write('  (cell_sound_ES cert_n%d cert_n%d_check X hrk hES hχ hmat hmatm).1\n\n' % (n, n))
        o.write('end GL2.HeckeZ.Cells.S1\n')
    print('wrote', tpath, os.path.getsize(tpath))
    with open(os.path.join(out_dir, 'manifest_lhz1.tsv'), 'w') as o:
        o.write('n\tb1\td\tG_bits\tK1_moduli\tK1m_moduli\tuncovered\tunfactored_bits\tTfr_eq_Tprime\n')
        for r in manifest:
            o.write('\t'.join(str(x) for x in r) + '\n')
    for r in manifest:
        print(*r)


if __name__ == '__main__':
    main()
