#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""LV (slot 23): the independent Python references of the Smith cells, and a step-for-step mirror of the Lean kernel
algorithm GL2.L23.SmithPK.smithPK (for kernel-cost prediction and generator-time cross-checks).

Reference (independent of the Lean generator):
  * rho_int(n, a, b, c, d): the integer matrix rho_n(A) from the full double sum of [C] eq:rhondef (u_t = x^(n-t) y^t);
  * tree_mats(n): LR-Y's d^0 = treeD0, d^1 = treeDS 0, d^2 = treeDS 1 (GL2/MP/TreeCochains.lean) on Sym^n Z^2, built
    from the DEFINITIONS (N_S = sum_{j<4} S^j, N_U = sum_{j<6} U^j, edgA with rho(-I)), not from the even-n
    simplification the Lean generator uses;
  * smith_vals(M, l, K): elementary divisors over Z/l^K by full pivoting (minimal valuation over the whole remaining
    matrix, row AND column operations) -- a different algorithm from the kernel's (level-wise, row operations only);
  * rank_q(M): the rank over Q (Gaussian elimination over Fraction), so a cell is *certified* when its valuations are all
    < K and their number equals rank_Q (no elementary divisor is 0 mod l^K by accident).
Mirror: smithpk(l, K, H, w, rows) with rows packed exactly as the Lean generators pack them (cols_lean(n, q, N, F)).
"""
from fractions import Fraction
from math import comb


def rho_int(n, a, b, c, d):
    """Integer rho_n(A), entry [t][j] = coefficient of u_t in A.u_j (eq:rhondef double sum)."""
    M = [[0] * (n + 1) for _ in range(n + 1)]
    for j in range(n + 1):
        for s in range(n - j + 1):
            for r in range(j + 1):
                M[s + r][j] += comb(n - j, s) * a ** (n - j - s) * c ** s * comb(j, r) * b ** (j - r) * d ** r
    return M


def mmul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def madd(*Ms):
    return [[sum(M[i][j] for M in Ms) for j in range(len(Ms[0][0]))] for i in range(len(Ms[0]))]


def mscale(c, A):
    return [[c * x for x in row] for row in A]


def eye(d):
    return [[int(i == j) for j in range(d)] for i in range(d)]


def zeros(r, c):
    return [[0] * c for _ in range(r)]


def block(rows_of_blocks):
    """Block matrix from a list of block rows."""
    out = []
    for br in rows_of_blocks:
        for i in range(len(br[0])):
            out.append(sum((B[i] for B in br), []))
    return out


def tree_mats(n):
    """d^0 (3d x 2d), d^1 (3d x 3d), d^2 (3d x 3d) of GL2.MP.treeD0 / treeDS 0 / treeDS 1 on Sym^n Z^2, as integer
    matrices acting on column vectors ((x, y), z)."""
    d = n + 1
    S, U = rho_int(n, 0, -1, 1, 0), rho_int(n, 0, -1, 1, 1)
    mI = rho_int(n, -1, 0, 0, -1)
    I = eye(d)
    S2, S3 = mmul(S, S), mmul(mmul(S, S), S)
    assert S2 == mI
    U2 = mmul(U, U)
    assert U2 == rho_int(n, -1, -1, 1, 0)
    Up = [I, U, U2]
    for _ in range(3):
        Up.append(mmul(Up[-1], U))
    NS4, NU6, NS2, NU3 = madd(I, S, S2, S3), madd(*Up[:6]), madd(I, S), madd(I, U, U2)
    neg = lambda A: mscale(-1, A)
    Z = zeros(d, d)
    # treeD0 (x, y) = ((1 - S)x, (1 - U)y, x - y)
    D0 = block([[madd(I, neg(S)), Z], [Z, madd(I, neg(U))], [I, neg(I)]])
    # treeDS m ((x,y),z) = (vtxA (m+1) (x,y), phiA (m+1) (x,y) - edgA m z)
    # m = 0: vtxA 1 = (N_S4, N_U6), phiA 1 = N_S2 x - N_U3 y, edgA 0 = 1 - (-I)
    D1 = block([[NS4, Z, Z], [Z, NU6, Z], [NS2, neg(NU3), neg(madd(I, neg(mI)))]])
    # m = 1: vtxA 2 = (1 - S, 1 - U), phiA 2 = x - y, edgA 1 = 1 + (-I)
    D2 = block([[madd(I, neg(S)), Z, Z], [Z, madd(I, neg(U)), Z], [I, neg(I), neg(madd(I, mI))]])
    return D0, D1, D2


def vl(x, l):
    v = 0
    while x % l == 0:
        x //= l
        v += 1
    return v


def smith_vals(M, l, K):
    """Valuations (< K) of the elementary divisors over Z/l^K of the integer matrix M, by full pivoting; non-increasing."""
    N = l ** K
    A = [[x % N for x in row] for row in M]
    rows, cols = len(A), len(A[0]) if A else 0
    out = []
    ri, ci = list(range(rows)), list(range(cols))
    while True:
        best = None
        for i in ri:
            for j in ci:
                if A[i][j]:
                    v = vl(A[i][j], l)
                    if best is None or v < best[0]:
                        best = (v, i, j)
                        if v == 0:
                            break
            if best and best[0] == 0:
                break
        if best is None:
            break
        v, i, j = best
        u = A[i][j] // l ** v
        ui = pow(u, -1, N)
        for i2 in ri:
            if i2 != i and A[i2][j]:
                f = (A[i2][j] // l ** v) * ui % N
                A[i2] = [(x - f * y) % N for x, y in zip(A[i2], A[i])]
        for j2 in ci:
            if j2 != j and A[i][j2]:
                f = (A[i][j2] // l ** v) * ui % N
                for i2 in ri:
                    A[i2][j2] = (A[i2][j2] - f * A[i2][j]) % N
        out.append(v)
        ri.remove(i)
        ci.remove(j)
    return sorted(out, reverse=True)


def rank_q(M):
    A = [[Fraction(x) for x in row] for row in M]
    r = 0
    rows, cols = len(A), len(A[0]) if A else 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c] / A[r][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        r += 1
    return r


def rank_mod(M, p):
    A = [[x % p for x in row] for row in M]
    r = 0
    rows, cols = len(A), len(A[0]) if A else 0
    for c in range(cols):
        q = next((i for i in range(r, rows) if A[i][c]), None)
        if q is None:
            continue
        A[r], A[q] = A[q], A[r]
        inv = pow(A[r][c], -1, p)
        A[r] = [x * inv % p for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[r])]
        r += 1
    return r


# ---------------------------------------------------------------- the mirror of GL2/L23/SmithPKExec.lean

def fld(F, x, i):
    return (x >> (F * i)) & ((1 << F) - 1)


def red_row(F, N, w, r):
    return sum((fld(F, r, i) % N) << (F * i) for i in range(w))


def rep(F, a, w):
    return sum(a << (F * i) for i in range(w))


def lvc(l, H, v):
    if l == 2:
        return 1, v + 1, 0
    m = l ** (v + 1)
    return pow(m, 2 ** (H - 1) - 1, 2 ** H), H, (2 ** H - 1) // m


def smithpk(l, K, H, w, rows, stats=None):
    """Step-for-step mirror of smithPK: returns the pivots (v, j, p), latest first."""
    F, N = 2 * H, l ** K
    e = l ** (K - 1) * (l - 1) - 1
    piv = []
    st = dict(tests=0, pivots=0, elims=0, redfields=0)
    for v in range(K):
        lv = l ** v
        c, b, T = lvc(l, H, v)
        LOW, ADD, TOP = rep(F, 2 ** b - 1, w), rep(F, 2 ** H - 1 - T, w), rep(F, 2 ** H, w)
        pend, done = list(rows), []
        while pend:
            r = pend.pop(0)
            st['tests'] += 1
            t = (((r * c) & LOW) + ADD) & TOP
            if t == 0:
                done.insert(0, r)
                continue
            j = next(i for i in range(w) if fld(F, t, i))   # firstNZ
            p = red_row(F, N, w, r)
            st['pivots'] += 1
            st['redfields'] += w
            u = fld(F, p, j) // lv
            ui = pow(u, e, N)

            def el(x):
                bb = (fld(F, x, j) % N) // lv
                cc = (N - (bb * ui) % N) % N
                return x + cc * p
            st['elims'] += len(pend) + len(done)
            pend = [el(x) for x in pend][::-1]   # mapA: reversed
            done = [el(x) for x in done][::-1]
            piv.insert(0, (v, j, p))
        rows = done
    for r in rows:  # every remaining row is 0 mod N
        assert all(fld(F, r, i) % N == 0 for i in range(w))
    if stats is not None:
        stats.update(st)
    return piv


def q_row(F, N, m):
    return sum(((comb(m, t) * (-1) ** (m + t)) % N) << (F * t) for t in range(m + 1))


def cols_lean(n, q, N, F):
    """The packed columns exactly as GL2.L23.SmithPK.colsD0 / colsD1 / colsD2 build them (reduced)."""
    d, w = n + 1, 3 * (n + 1)
    e = lambda k: 1 << (F * k)
    sh = lambda x, k: x << (F * k)
    neg = lambda x: (N - 1) * x
    sgn = lambda j, x: neg(x) if j % 2 else x
    colS = lambda j: sgn(j, e(n - j))
    colU = lambda j: sh(q_row(F, N, j), n - j)
    colUU = lambda j: sgn(j, q_row(F, N, n - j))
    n3 = lambda j: e(j) + colU(j) + colUU(j)
    red = lambda x: red_row(F, N, w, x)
    d0x = [red(e(j) + neg(colS(j)) + sh(e(j), 2 * d)) for j in range(d)]
    d0y = [red(sh(e(j) + neg(colU(j)), d) + sh(neg(e(j)), 2 * d)) for j in range(d)]
    d1x = [red(2 * (e(j) + colS(j)) + sh(e(j) + colS(j), 2 * d)) for j in range(d)]
    d1y = [red(sh(2 * n3(j), d) + sh(neg(n3(j)), 2 * d)) for j in range(d)]
    d2z = [red(sh(neg(2 * e(j)), 2 * d)) for j in range(d)]
    return [d0x + d0y, d1x + d1y, d0x + d0y + d2z][q]


def unpack(F, N, w, x):
    return [fld(F, x, i) % N for i in range(w)]


def min_H(l, K, R):
    N = l ** K
    H = 1
    while N + R * N * N > 2 ** H:
        H += 1
    return H


if __name__ == '__main__':
    import sys
    # self-test: the Lean-style packed columns are the columns of the reference matrices mod N; the mirror agrees with
    # the full-pivot reference.
    for n in (0, 2, 4, 6, 10, 12):
        D = tree_mats(n)
        for l, K in ((2, 6), (3, 4)):
            N = l ** K
            for q in range(3):
                R = len(D[q][0])
                H = min_H(l, K, R)
                F = 2 * H
                cols = cols_lean(n, q, N, F)
                ref_cols = [[D[q][i][j] % N for i in range(3 * (n + 1))] for j in range(len(D[q][0]))]
                nz = [c for c in ref_cols if any(c)]
                got = [unpack(F, N, 3 * (n + 1), c) for c in cols]
                assert [g for g in got if any(g)] == nz, (n, l, q)
                a = smith_vals(D[q], l, K)
                b = [p[0] for p in smithpk(l, K, H, 3 * (n + 1), cols)]
                assert a == b, (n, l, K, q, a, b)
    print('LV_SMITH_REF selftest ok')
