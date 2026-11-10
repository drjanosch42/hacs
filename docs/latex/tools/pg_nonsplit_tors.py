#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
"""pg_nonsplit_tors.py -- the exact computations behind [PG] sec:pg-nonsplit (non-split torsion in H^1 of Gamma_0(N) over Z).

Input: the certified presentation H^1(Gamma_0(N); Sym^n Z^2) = tors (+) Z^r, T_p = [[tau, M], [0, F]] (F over Z, tau and M modulo the
torsion orders), printed by the console verb `descent(N, n, primes).module_Z()` ([C] prop:torsionhecke).  This script runs the shipped
binary, parses that record and does integer linear algebra only (no floating point, no external CAS).

Output lines (deterministic, one per cell and ell), used verbatim by [PG]:

  CELL N= n= ell= tors=[orders] r= P=[primes] scal= split(L)= split(Eis1)= split(Eisx)= split(cusp)= S=(order, generators)
      P        = the primes not dividing ell*N that were computed.  scal = every T_p (p in P) acts on tors_ell by the scalar 1+p.
      split(X) = the test of [C] prop:heckeZglue(2)(b) for T'_P = Z[T_p, p in P] on the saturated T'-stable sublattice X of L=H^1/tors
                 (the word `split` printed here is [C] sec:recordschema's `split_on_P`: non_split is a certificate for all of T, split is a statement on P only):
                 L, the Eisenstein trivial-character lattice Eis1, the twisted Eisenstein lattice Eisx, the cuspidal lattice cusp;
                 '-' when X = 0.  ([C]'s test is for tors_ell against L/ell^K; the script solves h F_p - c_p h = M_p over Z/ell^e, F_p, M_p
                 taken on X.)
      S        = the subgroup tors_ell cap Sum_p (T_p - 1 - p) H_ell of tors_ell, its order (only when scal); S_inv = its invariant factors.
  ELINE       one row per prime: the (value of M_p) on the rank-one twisted Eisenstein line, against the logarithm prediction
              (printed for the rank-one lines of (16,2), (16,4), (9,4), and for the first Eisx column at (27,2)); ELINE_CHECK verifies the formula on p <= 61.
  TBT         torsion-by-torsion: whether tors_ell is a direct sum of T'-stable cyclic groups, for T' = prime-to-ell*N operators and for T = all
              operators computed (including T_ell when ell does not divide N), by exhaustive search for |tors_ell| <= the bound;
              TBT_SUMMARY counts lines = searched + skipped (a line above the bound is skipped, never decided).
  SIGMA       M_e = (V/ell^e)^Gamma_0(N) and sigma_a on it, computed directly from Sym^n (independent of the descent record); `agree` compares with the
              record's torsion; `sigma` lists the summands by the character of the Teichmueller units (ell odd), or whether sigma_a = 1 for all a (ell = 2).
  EIS1M       M_p on the trivial-character Eisenstein lattice, for the primes <= 61 prime to ell*N.

Exit 0 iff every cell ran.  Usage:
  pg_nonsplit_tors.py <hacs> [--scan DIR] census | scan | tbt | k0 | modcheck | modl | sigma | extra | law
  (k0: the least K with eps_K non-split and dim(A cap mM); modcheck: the same verdict read from the shipped `.module(ell, K)`; modl: the census 4.0 search of [PG] sec:pg-nonsplit;
   sigma: the SIGMA and EIS1M rows; extra: five further confirming cells, prefixed X; law: the empirical law of sec:pg-nonsplit against family and extra cells.)
With `--scan DIR` the records are cached as one JSON file per cell in DIR (resumable).  Read-only on the source tree.
"""
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile

BOUND = 1024
CENSUS = [(8, 4), (8, 10), (16, 2), (16, 4), (27, 2), (27, 4)]
EXTRA = [(4, 4), (9, 4), (9, 6), (3, 4)]
ELINES = [(16, 2), (16, 4), (9, 4), (27, 2)]
XTRA = [(56, 2), (8, 14), (9, 14), (12, 8), (5, 16)]
SIGMA_CELLS = [(8, 4, 2), (8, 10, 2), (16, 2, 2), (16, 4, 2), (32, 2, 2), (4, 4, 2), (1, 6, 2), (27, 2, 3), (27, 4, 3), (9, 2, 3), (9, 4, 3), (9, 6, 3), (9, 10, 3), (3, 4, 3), (25, 4, 5)]
EIS1M_CELLS = [(27, 4, 3), (9, 4, 3), (9, 6, 3), (27, 2, 3), (16, 2, 2)]


# ---------------------------------------------------------------------------------------------------- the data
def family():
    """the cells of [PG] sec:pg-nonsplit: prime-power levels and 36 to n <= 12, and small composite levels at n <= 6, psi(N)(n+1) <= 300, n even."""
    return [(N, n) for N in (2, 3, 4, 5, 7, 8, 9, 16, 25, 27, 32, 36, 49, 64) for n in range(2, 14, 2) if psi(N) * (n + 1) <= 300] + \
           [(N, n) for N in (12, 18, 20, 24, 28, 40, 45, 48, 50, 54, 63, 72) for n in (2, 4, 6) if psi(N) * (n + 1) <= 300]


def psi(N):
    r, m, p = N, N, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            r = r * (p + 1) // p
        p += 1
    if m > 1:
        r = r * (m + 1) // m
    return r


def primes_for(N):
    ps, s = [], 0
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23):
        if N % q and s + q <= 100:
            ps.append(q)
            s += q
        if len(ps) >= 6:
            break
    return ps


def parse(text):
    # the console prints {key: value, ...}; quote the keys, outside string literals only
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*:)', r'\1"\2"\3', parts[i])
    return json.loads(''.join(parts))


def run_record(binary, N, n, primes, cache):
    if cache:
        path = os.path.join(cache, '%d_%d_%s.json' % (N, n, '_'.join(str(p) for p in primes)))
        if os.path.exists(path):
            with open(path) as f:
                return parse(f.read())
    script = 'set coefficient_ring = "Z";\nD = descent(%d, %d, %s);\nZ = D.module_Z();\nprint(Z);\n' % (N, n, json.dumps(primes))
    with tempfile.NamedTemporaryFile('w', suffix='.oc', dir=cache or '.', delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        out = subprocess.run([binary, '--script', tmp], capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    if out.returncode:
        raise RuntimeError('descent(%d,%d) failed: %s %s' % (N, n, out.stdout[-200:], out.stderr[-200:]))
    line = out.stdout.strip().splitlines()[-1]
    if cache:
        with open(path, 'w') as f:
            f.write(line)
    return parse(line)


# --------------------------------------------------------------------------------------------- exact linear algebra
def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def transpose(A):
    return [list(c) for c in zip(*A)]


def kernel_basis(A, ncols):
    """a Z-basis of ker A in Z^ncols, by a unimodular column transformation; the kernel of an integer matrix is saturated."""
    A = [list(r) for r in A]
    U = [[int(i == j) for j in range(ncols)] for i in range(ncols)]
    r = 0
    for row in range(len(A)):
        while True:
            nz = [j for j in range(r, ncols) if A[row][j]]
            if len(nz) <= 1:
                break
            j0 = min(nz, key=lambda j: abs(A[row][j]))
            for j in nz:
                if j != j0:
                    q = A[row][j] // A[row][j0]
                    for R in A:
                        R[j] -= q * R[j0]
                    for R in U:
                        R[j] -= q * R[j0]
        nz = [j for j in range(r, ncols) if A[row][j]]
        if nz:
            j0 = nz[0]
            if j0 != r:
                for R in A:
                    R[r], R[j0] = R[j0], R[r]
                for R in U:
                    R[r], R[j0] = R[j0], R[r]
            r += 1
    return [[U[i][j] for i in range(ncols)] for j in range(r, ncols)]


def v_ell(x, ell, cap):
    if x == 0:
        return cap
    v = 0
    while v < cap and x % ell == 0:
        x //= ell
        v += 1
    return v


def solve_mod(rows, rhs, ell, e):
    """solve A x = b over Z/ell^e (a chain ring), elimination on pivots of minimal valuation; returns x or None."""
    m = ell ** e
    if not rows:
        return [] if all(b % m == 0 for b in rhs) else None
    A = [[a % m for a in r] + [b % m] for r, b in zip(rows, rhs)]
    ncols = len(rows[0])
    piv, used = [], [False] * len(A)
    while True:
        best, bestv = None, e
        for i, R in enumerate(A):
            if used[i]:
                continue
            for j in range(ncols):
                v = v_ell(R[j], ell, e)
                if v < bestv:
                    bestv, best = v, (i, j)
            if bestv == 0:
                break
        if best is None:
            break
        i0, j0 = best
        used[i0] = True
        inv = pow(A[i0][j0] // ell ** bestv, -1, m)
        A[i0] = [(a * inv) % m for a in A[i0]]
        for i, R in enumerate(A):
            if i != i0 and R[j0]:
                q = (R[j0] // ell ** bestv) % m
                A[i] = [(a - q * b) % m for a, b in zip(R, A[i0])]
        piv.append((i0, j0, bestv))
    if any(not used[i] and R[ncols] % m for i, R in enumerate(A)):
        return None
    x = [0] * ncols
    for i0, j0, v in piv:
        if A[i0][ncols] % ell ** v:
            return None
        x[j0] = (A[i0][ncols] // ell ** v) % m
    return x


class Ell:
    """the ell-part of a module_Z record, with the operators T_p for p prime to ell*N (or all computed p with all_p)."""

    def __init__(self, rec, ell, all_p=False):
        gens = rec['torsion_generators']
        idx = [i for i, g in enumerate(gens) if g['ell'] == ell]
        self.ell, self.N, self.n, self.r = ell, rec['N'], rec['n'], rec['free_rank']
        self.exps = [gens[i]['exponent'] for i in idx]
        self.ds = [ell ** a for a in self.exps]
        self.e = max(self.exps)
        self.ps = [o['p'] for o in rec['ops'] if all_p or (o['p'] % ell and self.N % o['p'])]
        self.c, self.M, self.F = {}, {}, {}
        for o in rec['ops']:
            if o['p'] in self.ps:
                self.c[o['p']] = [[o['tau'][i][j] for j in idx] for i in idx]
                self.M[o['p']] = [x for x in o['M'] if x['ell'] == ell][0]['matrix']
                self.F[o['p']] = [[int(v) for v in row] for row in o['F']]

    def scalar(self):
        return all(self.c[p][a][b] % self.ds[a] == ((1 + p) % self.ds[a] if a == b else 0) for p in self.ps for a in range(len(self.ds))
                   for b in range(len(self.ds)))

    def restrict(self, basis):
        if basis is None:
            basis = [[int(i == j) for i in range(self.r)] for j in range(self.r)]
        B = transpose(basis)
        G = self.left_inverse(B)
        return basis, B, G

    def left_inverse(self, B):
        r, s, k = len(B), len(B[0]), self.e + 16
        return [solve_mod(transpose(B), [int(i == j) for i in range(s)], self.ell, k) for j in range(s)]

    def split(self, basis=None):
        if basis is not None and not basis:
            return '-'
        basis, B, G = self.restrict(basis)
        s = len(B[0])
        t, ell, e = len(self.exps), self.ell, self.e
        m = ell ** e
        rows, rhs = [], []
        for p in self.ps:
            Fs = [[x % m for x in R] for R in matmul(G, matmul(self.F[p], B))]
            Ms = matmul(self.M[p], B)
            c = self.c[p]
            for a in range(t):
                scale = ell ** (e - self.exps[a])
                for j in range(s):
                    row = [0] * (t * s)
                    for j2 in range(s):
                        row[a * s + j2] = (row[a * s + j2] + scale * Fs[j2][j]) % m
                    for a2 in range(t):
                        row[a2 * s + j] = (row[a2 * s + j] - scale * c[a][a2]) % m
                    rows.append(row)
                    rhs.append((scale * Ms[a][j]) % m)
        return 'split' if solve_mod(rows, rhs, ell, e) is not None else 'non_split'

    def S(self, basis=None):
        """the subgroup tors_ell cap Sum_p (T_p - 1 - p) H, as (order, elements of a generating set), when the action is the scalar 1+p."""
        if basis is not None and not basis:
            return None
        basis, B, G = self.restrict(basis)
        s, ell = len(B[0]), self.ell
        big = ell ** 40
        ps = self.ps
        DS = {p: [[((self.signed(matmul(G, matmul(self.F[p], B))[i][j], big) - (1 + p if i == j else 0))) for j in range(s)] for i in range(s)] for p in ps}
        MS = {p: matmul(self.M[p], B) for p in ps}
        A = [[DS[p][i][j] for p in ps for j in range(s)] for i in range(s)]
        K = kernel_basis(A, s * len(ps))
        t = len(self.ds)
        gens = {tuple(sum(kv[ip * s + j] * MS[p][a][j] for ip, p in enumerate(ps) for j in range(s)) % self.ds[a] for a in range(t)) for kv in K}
        gens.discard(tuple([0] * t))
        sub = {tuple([0] * t)}
        for g in gens:
            while True:
                add = {tuple((x + y) % d for x, y, d in zip(v, g, self.ds)) for v in sub}
                if add <= sub:
                    break
                sub |= add
        return len(sub), sorted(gens)[:3], self.group_invariants(sub)

    def group_invariants(self, sub):
        """the invariant factors (ascending) of a finite ell-group given as the set of its elements in (+) Z/d_i."""
        ell, ds = self.ell, self.ds
        def times(x, m):
            return tuple((a * m) % d for a, d in zip(x, ds))
        zero = tuple([0] * len(ds))
        cnt, k = [1], 1                          # cnt[k] = #sub[ell^k]
        while True:
            c = sum(1 for x in sub if times(x, ell ** k) == zero)
            cnt.append(c)
            if c == len(sub):
                break
            k += 1
        # r_j = number of cyclic factors of order >= ell^j = log_ell(#[ell^j] / #[ell^(j-1)])
        r = []
        for j in range(1, len(cnt)):
            q, t = cnt[j] // cnt[j - 1], 0
            while q > 1:
                q //= ell
                t += 1
            r.append(t)
        out = []
        for j in range(len(r), 0, -1):
            m = r[j - 1] - (r[j] if j < len(r) else 0)
            out = [ell ** j] * m + out
        return out

    @staticmethod
    def signed(x, big):
        return ((x + big // 2) % big) - big // 2

    # Eisenstein lattices.  The twisted part is the part with eigenvalue chi(p) (1 + p^(n+1)), chi of order 2 with conductor 4 (16 | N) or 3 (9 | N);
    # a prime with chi(p) = -1 is used so that the two systems are distinct.  (Complex or conductor-8 characters are not treated: at levels 25, 49,
    # 50, 64, the column cusp is the complement of Eis1 + the listed twists only, and the text uses cusp only where these are all the characters.)
    def _p0_vals(self):
        N, n = self.N, self.n
        p0 = min(self.ps)
        if N % 16 == 0:
            p0 = min(p for p in self.ps if p % 4 == 3)
        if N % 9 == 0:
            p0 = min(p for p in self.ps if p % 3 == 2)
        vals = {1 + p0 ** (n + 1)}
        twist = set()
        if N % 16 == 0 and p0 % 2:
            twist.add((1 if p0 % 4 == 1 else -1) * (1 + p0 ** (n + 1)))
        if N % 9 == 0 and p0 % 3:
            twist.add((1 if p0 % 3 == 1 else -1) * (1 + p0 ** (n + 1)))
        return p0, vals, twist

    def eis_lattices(self):
        p0, vals, twist = self._p0_vals()
        r, Fp = self.r, self.F[p0]
        def ker_val(vs):
            P = [[int(i == j) for j in range(r)] for i in range(r)]
            for v in vs:
                P = matmul(P, [[Fp[i][j] - (v if i == j else 0) for j in range(r)] for i in range(r)])
            return kernel_basis(P, r)
        return ker_val(sorted(vals)), ker_val(sorted(twist - vals)) if (twist - vals) else [], ker_val(sorted(vals | twist))

    def cusp_lattice(self, keis_all):
        """the saturation of the image of the Eisenstein product"""
        p0, vals, twist = self._p0_vals()
        r, Fp = self.r, self.F[p0]
        P = [[int(i == j) for j in range(r)] for i in range(r)]
        for v in sorted(vals | twist):
            P = matmul(P, [[Fp[i][j] - (v if i == j else 0) for j in range(r)] for i in range(r)])
        left = kernel_basis(transpose(P), r)
        return kernel_basis(left, r) if left else [[int(i == j) for i in range(r)] for j in range(r)]


def report_cell(rec, N, n):
    lines = []
    for ell in sorted({g['ell'] for g in rec['torsion_generators']}):
        d = Ell(rec, ell)
        if not d.ps:
            continue
        keis1, keisx, keisall = d.eis_lattices()
        kcusp = d.cusp_lattice(keisall)
        sc = d.scalar()
        s = d.S() if sc else None
        lines.append('CELL N=%d n=%d ell=%d tors=%s r=%d P=%s scal=%s split(L)=%s split(Eis1)=%s split(Eisx)=%s split(cusp)=%s S=%s' % (
            N, n, ell, d.ds, d.r, d.ps, sc, d.split(), d.split(keis1), d.split(keisx), d.split(kcusp),
            ('%d S_inv=%s' % (s[0], str(s[2]).replace(' ', '')) if s else 'n/a')))
    return lines


# ------------------------------------------------------------------------------- sigma_a on M_e, directly from Sym^n
def sym_rep(g, n, m):
    """rho(g) on Sym^n (Z/m)^2 in the basis x^(n-j) y^j, with g.x = a x + c y, g.y = b x + d y ([C]'s convention); columns = images."""
    (a, b), (c, d) = g
    def pmul(u, v):
        w = [0] * (len(u) + len(v) - 1)
        for i, x in enumerate(u):
            for j, y in enumerate(v):
                w[i + j] = (w[i + j] + x * y) % m
        return w
    def ppow(u, k):
        r = [1]
        for _ in range(k):
            r = pmul(r, u)
        return r
    cols = []
    for j in range(n + 1):                      # x^(n-j) y^j, polynomial in t = y/x: coefficient of t^i = coefficient of x^(n-i) y^i
        cols.append(pmul(ppow([a % m, c % m], n - j), ppow([b % m, d % m], j)))
    return [[cols[j][i] for j in range(n + 1)] for i in range(n + 1)]


def smith_kernel(A, ncols, ell, e):
    """over Z/ell^e: generators of ker A with their orders, by a Smith reduction on pivots of minimal valuation."""
    m = ell ** e
    A = [[x % m for x in r] for r in A]
    U = [[int(i == j) for j in range(ncols)] for i in range(ncols)]
    t = 0
    piv = []
    while True:
        best, bestv = None, e
        for i in range(t, len(A)):
            for j in range(t, ncols):
                v = v_ell(A[i][j], ell, e)
                if v < bestv:
                    bestv, best = v, (i, j)
        if best is None:
            break
        i0, j0 = best
        A[t], A[i0] = A[i0], A[t]
        for R in A:
            R[t], R[j0] = R[j0], R[t]
        for R in U:
            R[t], R[j0] = R[j0], R[t]
        inv = pow(A[t][t] // ell ** bestv, -1, m)
        A[t] = [x * inv % m for x in A[t]]
        for i in range(len(A)):
            if i != t and A[i][t]:
                q = A[i][t] // ell ** bestv
                A[i] = [(x - q * y) % m for x, y in zip(A[i], A[t])]
        for j in range(t + 1, ncols):
            if A[t][j]:
                q = A[t][j] // ell ** bestv
                for R in A:
                    R[j] = (R[j] - q * R[t]) % m
                for R in U:
                    R[j] = (R[j] - q * R[t]) % m
        piv.append(bestv)
        t += 1
    gens = []
    for j, v in enumerate(piv):
        if v > 0:
            gens.append(([(ell ** (e - v) * U[i][j]) % m for i in range(ncols)], ell ** v))
    for j in range(t, ncols):
        gens.append(([U[i][j] for i in range(ncols)], m))
    return gens


def span(gens, m):
    zero = tuple([0] * len(gens[0][0])) if gens else ()
    S = {zero}
    for g, _ in gens:
        x = tuple(g)
        while True:
            add = {tuple((a + b) % m for a, b in zip(v, x)) for v in S}
            if add <= S:
                break
            S |= add
    return S


def orders_of(S, ell, m):
    """the invariant factors (descending) of a finite subgroup S of (Z/m)^r, given as a set."""
    if len(S) == 1:
        return []
    def tors(k):
        return sum(1 for x in S if all((a * ell ** k) % m == 0 for a in x))
    cnt, k = [1], 1
    while True:
        cnt.append(tors(k))
        if cnt[-1] == len(S):
            break
        k += 1
    r = []
    for j in range(1, len(cnt)):
        q, t = cnt[j] // cnt[j - 1], 0
        while q > 1:
            q //= ell
            t += 1
        r.append(t)
    out = []
    for j in range(len(r), 0, -1):
        c = r[j - 1] - (r[j] if j < len(r) else 0)
        out += [ell ** j] * c
    return out


def unit_gens(ell, e):
    m = ell ** e
    if ell == 2:
        return [x % m for x in (-1, 5) if x % m != 1]
    for g in range(2, m):
        if g % ell and all(pow(g, (ell - 1) * ell ** (e - 1) // q, m) != 1 for q in {*prime_factors((ell - 1) * ell ** (e - 1))}):
            return [g]
    return [1]


def prime_factors(x):
    r, p = [], 2
    while p * p <= x:
        if x % p == 0:
            r.append(p)
            while x % p == 0:
                x //= p
        p += 1
    if x > 1:
        r.append(x)
    return r


def report_sigma(N, n, ell, rec):
    """M_e = (V/ell^e)^Gamma_0(N), e = 1 + exponent of the record's tors_ell; sigma_a = rho(diag(1, a)) on it."""
    idx = [g['exponent'] for g in rec['torsion_generators'] if g['ell'] == ell]
    e = max(idx) + 1
    m = ell ** e
    v = 0
    while N % ell ** (v + 1) == 0:
        v += 1
    gens = [((1, 1), (0, 1)), ((1, 0), (N, 1))] + [((u, 0), (0, pow(u, -1, m))) for u in unit_gens(ell, e)]
    A = []
    for g in gens:
        R = sym_rep(g, n, m)
        A += [[(R[i][j] - int(i == j)) % m for j in range(n + 1)] for i in range(n + 1)]
    K = smith_kernel(A, n + 1, ell, e)
    S = span(K, m)
    ords = sorted(orders_of(S, ell, m), reverse=False)
    agree = ords == sorted(ell ** a for a in idx)
    def sig(a, x):
        R = sym_rep(((1, 0), (0, a)), n, m)
        return tuple(sum(R[i][j] * x[j] for j in range(n + 1)) % m for i in range(n + 1))
    trivial = all(sig(a, x) == x for a in unit_gens(ell, e) for x in S)
    parts = []
    if ell > 2:
        g = unit_gens(ell, e)[0]
        tau = pow(g, ell ** (e - 1), m)          # a Teichmueller generator of mu_(ell-1) modulo ell^e
        for j in range(ell - 1):
            ev = pow(tau, j, m)
            part = {x for x in S if sig(tau, x) == tuple(ev * c % m for c in x)}
            if len(part) > 1:
                o = sorted(orders_of(part, ell, m))
                wild = pow(1 + ell, 1, m)
                wild_trivial = all(sig(wild, x) == x for x in part)
                parts.append('omega^%d:%s:1+ell_trivial=%s' % (j, str(o).replace(' ', ''), wild_trivial))
    wstr = ' parts=' + ','.join(parts) if parts else ''
    xn = tuple([1 if i == 0 else 0 for i in range(n + 1)])
    return 'SIGMA N=%d n=%d ell=%d e_used=%d M=%s agree=%s sigma_trivial=%s xn_in_M1=%s%s' % (
        N, n, ell, e, str(ords).replace(' ', ''), agree, trivial, tuple(c * ell ** (e - 1) % m for c in xn) in S, wstr)


def prime_batches(N, ell):
    chi_p = 3 if ell == 2 else 2
    out = []
    for ps in ([3, 5, 7, 11, 13, 17], [19, 23, 29], [31, 37], [41, 43], [47], [53], [59], [61]):
        ps2 = sorted(set(ps + [chi_p]) - ({2} if ell == 2 else {3}))
        out.append([p for p in ps2 if N % p])
    return out


def report_eis1m(binary, N, n, ell, cache):
    """M_p (mod the torsion orders) on the lattice Eis1: the columns, for every p <= 61 prime to ell*N."""
    rows = {}
    rk, ds = None, None
    for ps in prime_batches(N, ell):
        d = Ell(run_record(binary, N, n, ps, cache), ell)
        k = d.eis_lattices()[0]
        rk, ds = len(k), d.ds
        B = transpose(k)
        for p in d.ps:
            rows[p] = [[x % q for x in R] for R, q in zip(matmul(d.M[p], B), d.ds)]
    nz = sorted(p for p, m in rows.items() if any(any(R) for R in m))
    return 'EIS1M N=%d n=%d ell=%d rank=%d tors=%s primes=%d nonzero_M_at=%s' % (N, n, ell, rk, str(ds).replace(' ', ''), len(rows), str(nz).replace(' ', ''))


# ------------------------------------------------------------------------------------------- torsion-by-torsion
def stable_cyclics(ds, ops):
    zero = tuple([0] * len(ds))
    def mul(t, v):
        return tuple(sum(t[a][b] * v[b] for b in range(len(v))) % ds[a] for a in range(len(v)))
    seen, res = set(), []
    for g in itertools.product(*[range(d) for d in ds]):
        if g == zero:
            continue
        c, x = [zero], g
        while x != zero:
            c.append(x)
            x = tuple((a + b) % d for a, b, d in zip(x, g, ds))
        fs = frozenset(c)
        if fs in seen:
            continue
        seen.add(fs)
        if all(mul(t, g) in fs for t in ops):
            res.append(fs)
    return res


def decomposes(ds, ops):
    """is the group (+) Z/d_i a direct sum of cyclic subgroups stable under every matrix of ops?  (exhaustive)"""
    size = 1
    for d in ds:
        size *= d
    zero = frozenset([tuple([0] * len(ds))])
    cyc = sorted(stable_cyclics(ds, ops), key=len, reverse=True)
    dead = set()
    def add(A, B):
        return frozenset(tuple((a + b) % d for a, b, d in zip(x, y, ds)) for x in A for y in B)
    def rec(S):
        if len(S) == size:
            return True
        if S in dead:
            return False
        for c in cyc:
            if len(S & c) == 1 and rec(add(S, c)):
                return True
        dead.add(S)
        return False
    return rec(zero)


def report_tbt(rec, N, n, bound):
    out = []
    for ell in sorted({g['ell'] for g in rec['torsion_generators']}):
        d = Ell(rec, ell, all_p=True)
        gens = rec['torsion_generators']
        idx = [i for i, g in enumerate(gens) if g['ell'] == ell]
        ds = d.ds
        size = 1
        for q in ds:
            size *= q
        Tp = [[[o['tau'][i][j] for j in idx] for i in idx] for o in rec['ops'] if o['p'] % ell and N % o['p']]
        Tall = [[[o['tau'][i][j] for j in idx] for i in idx] for o in rec['ops']]
        if size > bound:
            out.append((ell, ds, 'skip', 'skip'))
            continue
        out.append((ell, ds, decomposes(ds, Tp) if Tp else True, decomposes(ds, Tall) if Tall else True))
    return out


def report_eline(rec):
    """M_p on the rank-one twisted Eisenstein line (or the first column at rank 2), against (1 - chi(p)) * t * log-character."""
    N, n = rec['N'], rec['n']
    ell = 2 if N % 2 == 0 else 3
    d = Ell(rec, ell)
    chi = (lambda p: 1 if p % 4 == 1 else -1) if ell == 2 else (lambda p: 1 if p % 3 == 1 else -1)
    pchi = [p for p in d.ps if chi(p) == -1][0]
    v = -(1 + pchi ** (n + 1))
    k = kernel_basis([[d.F[pchi][i][j] - (v if i == j else 0) for j in range(d.r)] for i in range(d.r)], d.r)
    B = transpose(k)
    rows = []
    for p in d.ps:
        rows.append((p, chi(p), [(R[0]) % ds for R, ds in zip(matmul(d.M[p], B), d.ds)]))
    return ell, len(k), rows, d.ds


# ------------------------------------------------------------------------------------- visibility modulo ell^K
def split_at_K(d, K):
    """the split test for eps_K: 0 -> tors/ell^K -> H/ell^K -> L/ell^K -> 0, i.e. T' acting on H (x) Z/ell^K, K <= e."""
    ell, r = d.ell, d.r
    exps = [min(x, K) for x in d.exps]
    t, e = len(exps), max(exps)
    m = ell ** e
    rows, rhs = [], []
    for p in d.ps:
        F = [[x % m for x in R] for R in d.F[p]]
        for a in range(t):
            scale = ell ** (e - exps[a])
            for j in range(r):
                row = [0] * (t * r)
                for j2 in range(r):
                    row[a * r + j2] = (row[a * r + j2] + scale * F[j2][j]) % m
                for a2 in range(t):
                    row[a2 * r + j] = (row[a2 * r + j] - scale * d.c[p][a][a2]) % m
                rows.append(row)
                rhs.append((scale * d.M[p][a][j]) % m)
    return solve_mod(rows, rhs, ell, e) is not None


def rank_mod(A, ell):
    A = [[x % ell for x in r] for r in A]
    rk = 0
    for c in range(len(A[0]) if A else 0):
        piv = next((i for i in range(rk, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[rk], A[piv] = A[piv], A[rk]
        inv = pow(A[rk][c], -1, ell)
        A[rk] = [x * inv % ell for x in A[rk]]
        for i in range(len(A)):
            if i != rk and A[i][c]:
                q = A[i][c]
                A[i] = [(x - q * y) % ell for x, y in zip(A[i], A[rk])]
        rk += 1
    return rk


def dim_A_cap_mM(d):
    """over F_ell, with T' acting on A = tors/ell by the scalar 1+p: dim(A cap Sum_p (T_p - 1 - p) M), M = H/ell."""
    ell, t, r = d.ell, len(d.exps), d.r
    W = []                                      # columns of T_p - 1 - p on M = A (+) L/ell, as rows here
    for p in d.ps:
        for j in range(t):
            col = [((1 if a == j else 0) * 0 + (d.c[p][a][j] - ((1 + p) if a == j else 0))) % ell for a in range(t)] + [0] * r
            W.append(col)
        for j in range(r):
            col = [d.M[p][a][j] % ell for a in range(t)] + [(d.F[p][i][j] - ((1 + p) if i == j else 0)) % ell for i in range(r)]
            W.append(col)
    rkW = rank_mod(W, ell)
    A = [[int(i == j) for i in range(t + r)] for j in range(t)]
    return t + rkW - rank_mod(A + W, ell)


def report_k0(rec, N, n):
    out = []
    for ell in sorted({g['ell'] for g in rec['torsion_generators']}):
        d = Ell(rec, ell)
        if not d.ps or d.split() != 'non_split':
            continue
        k0 = next((K for K in range(1, d.e + 1) if not split_at_K(d, K)), None)
        x = dim_A_cap_mM(d) if d.scalar() else 'n/a'
        out.append('KZERO N=%d n=%d ell=%d tors=%s e=%d K0=%s dim(A cap mM)=%s' % (N, n, ell, d.ds, d.e, k0, x))
    return out


def run_module(binary, N, n, ps, ell, K, cache):
    """the shipped mod-ell^K route, independent of module_Z: the split verdict of H (x) Z/ell^K read from `.module(ell, K)`."""
    src = 'set coefficient_ring = "Z";\nD = descent(%d, %d, %s);\nprint(D.module(%d, %d));\n' % (N, n, json.dumps(ps), ell, K)
    with tempfile.NamedTemporaryFile('w', suffix='.oc', dir=cache or '.', delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        out = subprocess.run([binary, '--script', tmp], capture_output=True, text=True)
    finally:
        os.unlink(tmp)
    lines = out.stdout.strip().splitlines()
    if not lines or not lines[-1].startswith('{'):
        raise RuntimeError('module(%d,%d,%d): %s %s' % (N, n, ell, out.stdout[-300:], out.stderr[-300:]))
    return parse(lines[-1])


def split_from_module(m, ell):
    gens = m['generators']
    tidx = [i for i, g in enumerate(gens) if g['kind'] == 'torsion']
    fidx = [i for i, g in enumerate(gens) if g['kind'] != 'torsion']
    K = m['K']
    ops = [o for o in m['ops'] if o['p'] % ell and m['N'] % o['p']]
    mod = [ell ** g['exponent'] for g in gens]
    e = max([g['exponent'] for g in gens if g['kind'] == 'torsion'] or [1])
    M = ell ** e
    rows, rhs = [], []
    for o in ops:
        T = o['matrix']
        for a in tidx:
            scale = ell ** (e - gens[a]['exponent'])
            for j in fidx:
                row = [0] * (len(tidx) * len(fidx))
                for jj, j2 in enumerate(fidx):
                    row[tidx.index(a) * len(fidx) + jj] = scale * T[j2][j] % M
                for a2i, a2 in enumerate(tidx):
                    row[a2i * len(fidx) + fidx.index(j)] = (row[a2i * len(fidx) + fidx.index(j)] - scale * T[a][a2]) % M
                rows.append(row)
                rhs.append(scale * T[a][j] % M)
    return solve_mod(rows, rhs, ell, e) is not None


def report_modl_census(tree):
    """census 4.0 main + b1 at level 1: the collisions between an H^1 ideal and an H^0 ideal, and the non-semisimple H^0 blocks."""
    import gzip
    root = os.path.join(tree, 'docs/matrix-engine/results/congruence_census/edition3/4.0/')
    R = []
    for fn in ('rows40.tsv.gz', 'rows40_b1.tsv.gz'):
        with gzip.open(root + fn, 'rt') as f:
            hdr = None
            for l in f:
                if l.startswith('#'):
                    continue
                c = l.rstrip('\n').split('\t')
                if hdr is None:
                    hdr = c
                    continue
                R.append(dict(zip(hdr, c)))
    def dick(m, l):
        return m >= 0 and any(m - b * l * (l - 1) >= 0 and (m - b * l * (l - 1)) % (l + 1) == 0 for b in range(m // (l * (l - 1)) + 2))
    def beta(n, l):
        b, j, q = 0, 0, 1
        while q <= n:
            if n % q == 0 and dick(n // q, l):
                b = j + 1
            q *= l
            j += 1
        return b
    coll = [r for r in R if r['own_group'] == 'H1' and any(x['group'] == 'H0' and x.get('matched') for x in json.loads(r['groups']))]
    semis = [r for r in coll if r['module_loewy_length'] == '1']
    e1 = [r for r in semis if beta(int(r['n']), int(r['ell'])) == 1]
    eg = [r for r in semis if beta(int(r['n']), int(r['ell'])) > 1]
    h0 = [r for r in R if r['own_group'] == 'H0']
    ns = [r for r in h0 if r['module_loewy_length'] != '1']
    cells = sorted({(int(r['n']), int(r['ell'])) for r in ns})
    print('MODL_COLLISIONS rows=%d n_max=%d primes=%d H0_matched_rows=%d semisimple=%d e1_rows=%d e_gt1_rows=%d e_gt1_cells=%d' % (
        len(R), max(int(r['n']) for r in R), len({r['ell'] for r in R}), len(coll), len(semis), len(e1), len(eg), len({(r['n'], r['ell']) for r in eg})))
    def cert(rs):
        return sum(1 for r in rs if r['certified'] == 'true'), sum(1 for r in rs if r['certified'] != 'true')
    hl = sorted({int(r['ell']) for r in coll if r['certified'] != 'true'})
    print('MODL_CERT certified=%d heuristic=%d heuristic_ells=%s heuristic_final=%s e1_certified=%d e1_heuristic=%d e_gt1_certified=%d e_gt1_heuristic=%d loewy_A1=%d free_on_block=%d heuristic_n_min=%s' % (
        *cert(coll), str(hl).replace(' ', ''), sorted({r['final'] for r in coll if r['certified'] != 'true'}), *cert(e1), *cert(eg),
        sum(1 for r in coll if r['loewy_A'] == '1'), sum(1 for r in coll if r['sturm_outcome'] != 'not_applicable'),
        min([int(r['n']) for r in coll if r['certified'] != 'true'] or [0])))
    print('MODL_H0 rows=%d nonsemisimple_rows=%d cells=%s' % (len(h0), len(ns), ' '.join('(%d,%d)' % c for c in cells)))
    print('MODL_H0_SPLIT rows_n_pos=%d rows_n0=%d nonsemisimple_certified=%d nonsemisimple_heuristic=%d' % (
        sum(1 for r in h0 if int(r['n']) > 0), sum(1 for r in h0 if int(r['n']) == 0), *cert(ns)))


def eline_check(ell, rows, ds):
    """m_p = (1 - chi(p)) * t * k(p) modulo the order of the one generator carrying the class, k = the logarithm of p ([PG] sec:pg-nonsplit)."""
    def k_of(p):
        if ell == 2:
            q = p % 16
            for k in range(4):
                if pow(5, k, 16) in (q, (-q) % 16):
                    return k
        x = 1
        for k in range(6):
            if x in (p % 9, (-p) % 9):
                return k % 3
            x = x * 2 % 9
        raise AssertionError(p)
    carriers = [a for a in range(len(ds)) if any(m[a] % ds[a] for _, _, m in rows)]
    if len(carriers) != 1:
        return False, None
    a = carriers[0]
    for t in ((2,) if ell == 2 else (1, 2)):
        if all(((1 - chi) * t * k_of(p) - m[a]) % ds[a] == 0 for p, chi, m in rows):
            return True, t
    return False, None


def main(argv):
    args, cache = [], None
    a = argv[1:]
    while a:
        x = a.pop(0)
        if x == '--scan':
            cache = a.pop(0)
        else:
            args.append(x)
    binary, mode = args[0], args[1]
    if cache:
        os.makedirs(cache, exist_ok=True)
    if mode == 'census':
        for N, n in CENSUS + EXTRA:
            rec = run_record(binary, N, n, primes_for(N), cache)
            for line in report_cell(rec, N, n):
                print(line)
        for N, n in ELINES:
            ell = 2 if N % 2 == 0 else 3
            chi_p = 3 if ell == 2 else 2
            rows, rk = [], 0
            for ps in ([3, 5, 7, 11, 13, 17], [19, 23, 29], [31, 37], [41, 43], [47], [53], [59], [61]):
                ps2 = sorted(set(ps + [chi_p]) - ({2} if ell == 2 else {3}))
                ps2 = [p for p in ps2 if N % p]
                ell2, rk, r, ds = report_eline(run_record(binary, N, n, ps2, cache))
                rows += [x for x in r if x[0] not in [y[0] for y in rows]]
            rows.sort()
            print('ELINE N=%d n=%d ell=%d rank=%d ' % (N, n, ell, rk) + ' '.join('p=%d:chi=%d:M=%s' % (p, c, m) for p, c, m in rows[:6]))
            ok, t = eline_check(ell, rows, ds)
            print('ELINE_CHECK N=%d n=%d ell=%d primes=%d t=%s formula=%s' % (N, n, ell, len(rows), t, 'OK' if ok else 'FAIL'))
            if (N, n) == (16, 2):
                # the (mod 2) statement: m_p / 4 = 1 exactly when chi(p) = -1 and 2 is a non-residue mod p, on the generator carrying the class
                a = [i for i in range(len(ds)) if any(m[i] % ds[i] for _, _, m in rows)][0]
                good = all(m[a] % 4 == 0 and (m[a] // 4) % 2 == int(c == -1 and p % 8 in (3, 5)) for p, c, m in rows)
                print('KUMMER2 N=16 n=2 primes=%d m_over_4_eq_[chi=-1 and (2/p)=-1]=%s' % (len(rows), good))
    elif mode == 'modl':
        tree = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..'))
        report_modl_census(tree)
    elif mode == 'k0':
        fam = family()
        for N, n in list(dict.fromkeys(CENSUS + EXTRA + fam)):
            for line in report_k0(run_record(binary, N, n, primes_for(N), cache), N, n):
                print(line, flush=True)
    elif mode == 'modcheck':
        for N, n, ps, ell, Ks in [(8, 4, [3, 5, 7, 11], 2, (1, 2, 4)), (8, 2, [3, 5, 7, 11], 2, (1, 2, 3)), (16, 2, [3, 5, 7, 11], 2, (1, 2, 3)), (27, 2, [2, 5, 7, 11], 3, (1,)),
                                  (25, 4, [2, 3, 7, 11], 5, (1,)), (4, 4, [3, 5, 7, 11], 2, (1, 4))]:
            for K in Ks:
                m = run_module(binary, N, n, ps, ell, K, cache)
                print('MODK N=%d n=%d ell=%d K=%d torsion=%s verdict=%s' % (N, n, ell, K, m['torsion'], 'split' if split_from_module(m, ell) else 'non_split'), flush=True)
    elif mode == 'scan':
        # the family of [PG] sec:pg-nonsplit: prime and prime-power levels and 36, even n <= 12, psi(N)(n+1) <= 300
        for N, n in family():
            rec = run_record(binary, N, n, primes_for(N), cache)
            for line in report_cell(rec, N, n):
                print(line, flush=True)
    elif mode == 'tbt':
        tot = nd_p = nd_all = skipped = 0
        for N, n in CENSUS + EXTRA + [(1, m) for m in range(2, 61, 2)]:
            rec = run_record(binary, N, n, primes_for(N), cache)
            for ell, ds, dp, da in report_tbt(rec, N, n, BOUND):
                tot += 1
                skipped += dp == 'skip'
                nd_p += dp is False
                nd_all += da is False
                if dp is False or da is False:
                    print('TBT N=%d n=%d ell=%d tors=%s Tprime_decomposes=%s Tall_decomposes=%s' % (N, n, ell, ds, dp, da), flush=True)
        print('TBT_SUMMARY lines=%d searched=%d skipped=%d Tprime_nondecomposable=%d Tall_nondecomposable=%d bound=%d' % (tot, tot - skipped, skipped, nd_p, nd_all, BOUND))
    elif mode == 'sigma':
        for N, n, ell in SIGMA_CELLS:
            print(report_sigma(N, n, ell, run_record(binary, N, n, primes_for(N), cache)), flush=True)
        for N, n, ell in EIS1M_CELLS:
            print(report_eis1m(binary, N, n, ell, cache), flush=True)
    elif mode == 'extra':
        for N, n in XTRA:
            rec = run_record(binary, N, n, primes_for(N), cache)
            for line in report_cell(rec, N, n) + report_k0(rec, N, n):
                print('X' + line, flush=True)
    elif mode == 'law':
        def verdict(cells):
            lines = nonsp = bad = 0
            odd = []
            for N, n in cells:
                rec = run_record(binary, N, n, primes_for(N), cache)
                for line in report_cell(rec, N, n):
                    m = re.match(r'CELL N=(\d+) n=(\d+) ell=(\d+) .* split\(L\)=(\w+) ', line)
                    a, b, ell, v = int(m[1]), int(m[2]), int(m[3]), m[4]
                    if a % ell:
                        continue
                    lines += 1
                    nonsp += v == 'non_split'
                    law = (ell == 2 and a % 8 == 0) or (ell > 2 and a % ell ** 2 == 0 and b % (ell - 1) == 0)
                    bad += law != (v == 'non_split')
                    if ell > 2 and a % ell ** 2 == 0 and b % (ell - 1):
                        odd.append('(%d,%d,%d)' % (a, b, ell))
            return lines, nonsp, bad, odd
        for name, cells in (('family', list(dict.fromkeys(family()))), ('extra', XTRA)):
            lines, nonsp, bad, odd = verdict(cells)
            print('LAW set=%s lines=%d non_split=%d split_on_P=%d violations=%d odd_clause_lines=%s' % (name, lines, nonsp, lines - nonsp, bad, ','.join(odd) or '-'), flush=True)
    else:
        raise SystemExit('unknown mode')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
