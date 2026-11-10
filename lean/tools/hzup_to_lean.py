#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""hzup_to_lean.py: the TB-int reader of M-HZ5 (slot 26 Z5).

Reads the exact integer matrices of U_p and T_p on H^1/tors for a toy cell -- <record id>.Up<p>.mat (HECKE_DUMP_HZ_CERT, `:full`,
Z1) and the `module_Z()` transcript of the console (one file per cell, `mz_<N>_<n>.out` with lines
`BASIS <fnv>`, `OP <p>`, `F [[..]]`, `CHI [..]`), joined by basis_fnv -- and emits the Mathlib-free data module
GL2/HeckeZ/Cells/FullNData.lean (matrices + `decide +kernel` checks).

Usage: hzup_to_lean.py --dumps <dir> --mz <dir> --binary-sha256 <hex> --out <file>      generate
       hzup_to_lean.py --selftest                                                    negative controls (each must print REJECTED)

Every input is re-verified here: the file's own sha256 line, basis_fnv equal on both routes, the shapes, each CHI line equal to the
characteristic polynomial of its F and, at the cusp cells, to the factorisation the Lean `hχ` states (TB-chi), U_p and T_p commuting, the eigenvector
data, f(U) = 0, and at the non-reduced cell a rank factorisation F^2 = Q R with integer Q, R (no field inverse is written).
Exits 1 on the first failure; never writes partial output on failure.
"""
import argparse
import ast
import glob
import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from fractions import Fraction

CELLS = [
    {"name": "c2_6", "N": 2, "n": 6, "p": 2, "a": 1, "b": 128, "c": -8, "i0": 3, "e": 2188, "g": 12, "ell": 17},
    {"name": "c5_4", "N": 5, "n": 4, "p": 5, "a": 1, "b": 3125, "c": 25, "i0": 2, "e": 33, "g": 2, "ell": 31},
    {"name": "c8_10", "N": 8, "n": 10, "p": 2, "count": 6},
    {"name": "c11_0", "N": 11, "n": 0, "p": 11, "identity": True, "cusp": "x+2", "ell": 5},
    {"name": "c23_0", "N": 23, "n": 0, "p": 23, "cusp": "x2+x-1", "ell": 11},
]


def fail(msg):
    print("HZUP_TO_LEAN FAIL: " + msg)
    raise SystemExit(1)


def parse_up(path):
    text = open(path, encoding="utf-8", newline="").read()
    if not text.endswith("\n") or "sha256=" not in text:
        fail("no sha256 line: " + path)
    at = text.rfind("sha256=")
    if hashlib.sha256(text[:at].encode()).hexdigest() != text[at + 7:].strip():
        fail("sha256 mismatch (tampered?): " + path)
    head, body = text[:at].rsplit("\nF\n", 1)
    F = [[int(x) for x in ln.split()] for ln in body.strip().split("\n")]
    p = int(re.search(r"^p=(\d+)$", head, re.M).group(1))
    basis = re.search(r"^basis_fnv=(\S+)$", head, re.M).group(1)
    r = int(re.search(r"^r=(\d+)$", head, re.M).group(1))
    if r != len(F) or any(len(row) != r for row in F):
        fail("shape: " + path)
    return {"p": p, "basis": basis, "F": F, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def parse_mz(path):
    basis = None
    ops = {}
    cur = None
    for line in open(path, encoding="utf-8"):
        if line.startswith("BASIS "):
            basis = line.split()[1]
        elif line.startswith("OP "):
            cur = int(line.split()[1])
            ops[cur] = {}
        elif line.startswith("F ") and cur is not None:
            ops[cur]["F"] = [[int(x) for x in r] for r in ast.literal_eval(line[2:])]
        elif line.startswith("CHI ") and cur is not None:
            ops[cur]["chi"] = [int(x) for x in ast.literal_eval(line[4:])]
    if basis is None or not ops or any("F" not in o for o in ops.values()):
        fail("module_Z transcript incomplete: " + path)
    for q, o in ops.items():
        F = o["F"]
        if not F or any(len(row) != len(F) for row in F):
            fail("shape of module_Z T_%d: %s" % (q, path))
    return {"basis": basis, "ops": ops, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def matvec(A, v):
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def shift(A, c):
    return [[A[i][j] - (c if i == j else 0) for j in range(len(A))] for i in range(len(A))]


def nullspace(M):
    M = [[Fraction(x) for x in row] for row in M]
    m = len(M[0])
    r = 0
    piv = []
    for c in range(m):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        d = M[r][c]
        M[r] = [x / d for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    basis = []
    for fc in [c for c in range(m) if c not in piv]:
        v = [Fraction(0)] * m
        v[fc] = Fraction(1)
        for i, c in enumerate(piv):
            v[c] = -M[i][fc]
        lcm = 1
        for x in v:
            lcm = lcm * x.denominator // math.gcd(lcm, x.denominator)
        w = [int(x * lcm) for x in v]
        g = 0
        for x in w:
            g = math.gcd(g, abs(x))
        basis.append([x // g for x in w])
    return basis


def charpoly(A):
    """Coefficients of det(x I - A), leading first (Faddeev--LeVerrier over Q; exact)."""
    n = len(A)
    I = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    Af = [[Fraction(x) for x in row] for row in A]
    c = [Fraction(1)]
    M = [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        M = [[sum(Af[i][l] * M[l][j] for l in range(n)) + c[-1] * I[i][j] for j in range(n)] for i in range(n)]
        AM = [[sum(Af[i][l] * M[l][j] for l in range(n)) for j in range(n)] for i in range(n)]
        c.append(-sum(AM[i][i] for i in range(n)) / k)
    if any(x.denominator != 1 for x in c):
        fail("non-integral characteristic polynomial")
    return [int(x) for x in c]


def polymul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


# TB-chi: the factorisation the cell theorem's `hχ` states for the characteristic polynomial of T_2 (leading coefficient first).
HCHI = {
    "x+2": polymul([1, -3], polymul([1, 2], [1, 2])),               # (x - 3)(x + 2)^2
    "x2+x-1": polymul([1, -3], polymul([1, 1, -1], [1, 1, -1])),   # (x - 3)(x^2 + x - 1)^2
}


def rank(M):
    return len(M[0]) - len(nullspace(M))


def rank_factor(M):
    """M = Q R with Q (n x k), R (k x m), integer, k = rank M: column HNF of M by unimodular integer column operations."""
    n, m = len(M), len(M[0])
    A = [[M[i][j] for j in range(m)] for i in range(n)]
    Vinv = [[int(i == j) for j in range(m)] for i in range(m)]  # A = M V^-1 tracked as M = A * W with W
    W = [[int(i == j) for j in range(m)] for i in range(m)]
    col = 0
    for row in range(n):
        while True:
            pivots = [j for j in range(col, m) if A[row][j] != 0]
            if len(pivots) <= 1:
                break
            j0 = min(pivots, key=lambda j: abs(A[row][j]))
            for j in pivots:
                if j != j0:
                    q = A[row][j] // A[row][j0]
                    for i in range(n):
                        A[i][j] -= q * A[i][j0]
                    # column j <- column j - q column j0 means M e_j' ...; track: W row j0 += q * W row j
                    for t in range(m):
                        W[j0][t] += q * W[j][t]
        pivots = [j for j in range(col, m) if A[row][j] != 0]
        if pivots:
            j0 = pivots[0]
            if j0 != col:
                for i in range(n):
                    A[i][j0], A[i][col] = A[i][col], A[i][j0]
                W[j0], W[col] = W[col], W[j0]
            col += 1
    k = col
    Q = [[A[i][j] for j in range(k)] for i in range(n)]
    R = [W[j] for j in range(k)]
    if mul(Q, R) != M:
        fail("rank factorisation failed")
    return k, Q, R


def lean_mat(name, M):
    return "def %s : IMat := [%s]\n" % (name, ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in M))


def lean_vec(name, v):
    return "def %s : List Int := [%s]\n" % (name, ", ".join(str(x) for x in v))


def gen_cell(c, ups, mzd):
    N, n = c["N"], c["n"]
    up = ups.get((N, n, c["p"]))
    if up is None:
        fail("missing Up file for cell %r" % (c,))
    mz = mzd.get((N, n))
    if mz is None:
        fail("missing module_Z transcript for cell (%d,%d)" % (N, n))
    if mz["basis"] != up["basis"]:
        fail("basis_fnv mismatch at (%d,%d): %s vs %s" % (N, n, mz["basis"], up["basis"]))
    U = up["F"]
    r = len(U)
    for q, o in mz["ops"].items():
        if len(o["F"]) != r:
            fail("shape: module_Z T_%d is %dx%d, U_%d is %dx%d at (%d,%d)" % (q, len(o["F"]), len(o["F"]), c["p"], r, r, N, n))
        # TB-chi (07 sec. 5): the emitter's CHI line is the characteristic polynomial of its own F
        if "chi" not in o:
            fail("no CHI line for T_%d at (%d,%d)" % (q, N, n))
        if charpoly(o["F"]) != o["chi"]:
            fail("CHI of T_%d is not the characteristic polynomial of F at (%d,%d)" % (q, N, n))
    if c.get("cusp"):
        if 2 not in mz["ops"]:
            fail("no T_2 at the cusp cell (%d,%d)" % (N, n))
        if mz["ops"][2]["chi"] != HCHI[c["cusp"]]:
            fail("TB-chi: CHI of T_2 is not the factorisation h-chi states at (%d,%d)" % (N, n))
    for q, o in mz["ops"].items():
        if mul(U, o["F"]) != mul(o["F"], U):
            fail("U_%d and T_%d do not commute at (%d,%d)" % (c["p"], q, N, n))
    name = c["name"]
    out = []
    out.append("-- cell (N, n) = (%d, %d): U_%d (%s), module_Z (%s), basis %s\n" % (N, n, c["p"], up["sha256"], mz["sha256"], up["basis"]))
    out.append(lean_mat(name + "_U", U))
    if c.get("identity"):
        I = [[int(i == j) for j in range(r)] for i in range(r)]
        if U != I:
            fail("U is not the identity at the identity cell")
        out.append(lean_mat(name + "_I", I))
        out.append("theorem %s_U_eq : ieq %d %d %s_U %s_I = true := by decide +kernel\n" % (name, r, r, name, name))
    if c.get("cusp"):
        T2 = mz["ops"][2]["F"]
        out.append(lean_mat(name + "_T2", T2))
        if c["cusp"] == "x+2":
            C = shift(T2, -2)
            Cexpr = "ishift %d %s_T2 (-2)" % (r, name)
        else:
            T22 = mul(T2, T2)
            s1 = shift(T2, 1)
            C = [[T22[i][j] + s1[i][j] for j in range(r)] for i in range(r)]
            Cexpr = "iadd %d %d (imul %d %d %d %s_T2 %s_T2) (ishift %d %s_T2 1)" % (r, r, r, r, r, name, name, r, name)
        if mul(shift(T2, 3), C) != [[0] * r for _ in range(r)]:
            fail("(T_2 - 3) c(T_2) != 0 at (%d,%d)" % (N, n))
        out.append("def %s_C : IMat := %s\n" % (name, Cexpr))
        out.append("theorem %s_hc : ieq %d %d (imul %d %d %d (ishift %d %s_T2 3) %s_C) [] = true := by decide +kernel\n" % (name, r, r, r, r, r, r, name, name))
        ell = c["ell"]
        found = None
        for i in range(r):
            for j in range(r):
                if C[i][j] % ell != 0:
                    found = (i, j, C[i][j])
                    break
            if found:
                break
        if found is None:
            fail("no %d-unit entry of c(T_2) at (%d,%d)" % (ell, N, n))
        i, j, x = found
        out.append("theorem %s_Cent : ient %s_C %d %d = (%d) := by decide +kernel\n" % (name, name, i, j, x))
        out.append("def %s_i : Fin %d := ⟨%d, by decide⟩\n" % (name, r, i))
        out.append("def %s_j : Fin %d := ⟨%d, by decide⟩\n" % (name, r, j))
    if c.get("identity") or c.get("cusp"):
        return out
    if c.get("count"):
        k = c["count"]
        U2 = mul(U, U)
        if rank(U2) != k:
            fail("rank U^2 != %d at (%d,%d)" % (k, N, n))
        k2, Q, R = rank_factor(U2)
        if k2 != k:
            fail("rank_factor k mismatch")
        out.append(lean_mat(name + "_Q", Q))
        out.append(lean_mat(name + "_R", R))
        out.append("theorem %s_factor : ieq %d %d (imul %d %d %d %s_U %s_U) (imul %d %d %d %s_Q %s_R) = true := by decide +kernel\n"
                   % (name, r, r, r, r, r, name, name, r, r, k, name, name))
        return out
    if c.get("read"):
        out.append("-- LC-11 cell (data of T_2 only; c = x^2 + x - 1 at T_2)\n")
        out.append(lean_mat(name + "_T2", mz["ops"][2]["F"]))
        return out
    # the three-system shape
    a, b, cv, i0, e, g, ell = c["a"], c["b"], c["c"], c["i0"], c["e"], c["g"], c["ell"]
    T = mz["ops"][i0]["F"]
    out.append(lean_mat(name + "_T", T))
    def pick(ev, tv):
        for v in nullspace(shift(U, ev)):
            if matvec(T, v) == [tv * x for x in v]:
                return v
        fail("no eigenvector U = %d, T = %d at (%d,%d)" % (ev, tv, N, n))
    va, vb, vc = pick(a, e), pick(b, e), pick(cv, g)
    for nm, v, ev, tv in (("va", va, a, e), ("vb", vb, b, e), ("vc", vc, cv, g)):
        out.append(lean_vec(name + "_" + nm, v))
        out.append("theorem %s_U%s : ieigen %d %s_U %s_%s (%d) = true := by decide +kernel\n" % (name, nm, r, name, name, nm, ev))
        out.append("theorem %s_T%s : ieigen %d %s_T %s_%s (%d) = true := by decide +kernel\n" % (name, nm, r, name, name, nm, tv))
        out.append("theorem %s_nz%s : ((List.range %d).any fun i => ient (icol %s_%s) i 0 != 0) = true := by decide +kernel\n" % (name, nm, r, name, nm))
    fU = mul(mul(shift(U, a), shift(U, b)), shift(U, cv))
    if any(x != 0 for row in fU for x in row):
        fail("f(U) != 0 at (%d,%d)" % (N, n))
    out.append("theorem %s_f : ieq %d %d (imul %d %d %d (imul %d %d %d (ishift %d %s_U (%d)) (ishift %d %s_U (%d))) (ishift %d %s_U (%d))) [] = true := by decide +kernel\n"
               % (name, r, r, r, r, r, r, r, r, r, name, a, r, name, b, r, name, cv))
    P = mul(shift(U, a), shift(U, cv))
    best = None
    for i in range(r):
        for j in range(r):
            x = P[i][j]
            if x != 0 and x % ell != 0:
                best = (i, j, x)
                break
        if best:
            break
    if best is None:
        fail("no unit entry of P at ell = %d, (%d,%d)" % (ell, N, n))
    i, j, x = best
    out.append("/-- entry (%d, %d) of P = (U - a)(U - c), `ℓ = %d`-unit -/\n" % (i, j, ell))
    out.append("theorem %s_Pent : ient (imul %d %d %d (ishift %d %s_U (%d)) (ishift %d %s_U (%d))) %d %d = (%d) := by decide +kernel\n"
               % (name, r, r, r, r, name, a, r, name, cv, i, j, x))
    out.append("def %s_i : Fin %d := ⟨%d, by decide⟩\n" % (name, r, i))
    out.append("def %s_j : Fin %d := ⟨%d, by decide⟩\n" % (name, r, j))
    return out


def run(dumps, mzdir, sha, outpath):
    ups = {}
    for f in sorted(glob.glob(os.path.join(dumps, "hz_N*_n*_full_*.Up*.mat"))):
        m = re.match(r".*hz_N(\d+)_n(\d+)_full_.*\.Up(\d+)\.mat$", f)
        ups[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = parse_up(f)
    mzd = {}
    for f in sorted(glob.glob(os.path.join(mzdir, "mz_*_*.out"))):
        m = re.match(r".*mz_(\d+)_(\d+)\.out$", f)
        mzd[(int(m.group(1)), int(m.group(2)))] = parse_mz(f)
    out = ["/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/\n"
           "import GL2.HeckeZ.Exec.TBInt\n\n/-!\n# M-HZ5 toy-cell data (generated by lean/tools/hzup_to_lean.py; do not edit)\n\n"
           "Binary sha256 `%s`. Exact integer matrices (TB-int) of `U_p` (`*.Up<p>.mat`) and `T_p` (`module_Z()`, same `basis_fnv`)\n"
           "on `L = H¹/tors` in one `ℤ`-basis, and the kernel checks of the cell theorems.\n-/\n\n"
           "namespace GL2.HeckeZ.Cells.FullNData\n\nopen GL2.HeckeZ.Exec.TBInt\n\n" % sha]
    for c in CELLS:
        out.extend(gen_cell(c, ups, mzd))
        out.append("\n")
    out.append("end GL2.HeckeZ.Cells.FullNData\n")
    tmp = outpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    os.replace(tmp, outpath)
    print("HZUP_TO_LEAN cells=%d out=%s verdict=OK" % (len(CELLS), outpath))


def selftest():
    """Negative controls: each must REJECT (exit via SystemExit), then the good fixture must pass."""
    d = tempfile.mkdtemp(prefix="hzup_selftest_", dir=os.environ.get("HZUP_SELFTEST_DIR"))
    try:
        def mkcell(F, T, chi=None):
            text = ("schema=hz_up_matrix/v1\nN=2\nn=6\np=2\nrecord_id=x\nrecord_fnv1a64=0x1\noperator=U_2\nbasis=x\nbasis_fnv=0xABC\nr=%d\nproof=apriori\nF_fnv=0x0\n"
                    "tau_M=blocked\nF\n" % len(F) + "\n".join(" ".join(str(x) for x in r) for r in F) + "\n")
            text += "sha256=" + hashlib.sha256(text.encode()).hexdigest() + "\n"
            open(os.path.join(d, "hz_N2_n6_full_x.Up2.mat"), "w", encoding="utf-8", newline="").write(text)
            open(os.path.join(d, "mz_2_6.out"), "w", encoding="utf-8").write("BASIS 0xABC\nOP 3\nF %s\nCHI %s\n" % (str([[str(x) for x in r] for r in T]), str([str(x) for x in (chi if chi is not None else charpoly(T))])))
            return text
        ok = 0
        def expect_reject(label, fn):
            nonlocal ok
            try:
                fn()
            except SystemExit:
                print("CONTROL %s REJECTED" % label)
                ok += 1
                return
            print("CONTROL %s NOT REJECTED: FAIL" % label)
            raise SystemExit(1)
        # tampered: sha256 no longer matches
        t = mkcell([[1, 0], [0, 2]], [[3, 0], [0, 4]])
        tampered = t.replace("1 0", "9 0", 1)
        open(os.path.join(d, "hz_N2_n6_full_x.Up2.mat"), "w", encoding="utf-8", newline="").write(tampered)
        expect_reject("tampered_entry", lambda: parse_up(os.path.join(d, "hz_N2_n6_full_x.Up2.mat")))
        # basis mismatch
        mkcell([[1, 0], [0, 2]], [[3, 0], [0, 4]])
        open(os.path.join(d, "mz_2_6.out"), "w", encoding="utf-8").write("BASIS 0xDEF\nOP 3\nF [[\"3\",\"0\"],[\"0\",\"4\"]]\nCHI [\"1\",\"-7\",\"12\"]\n")
        expect_reject("basis_fnv_mismatch", lambda: gen_cell({"name": "x", "N": 2, "n": 6, "p": 2, "a": 1, "b": 2, "c": 3, "i0": 3, "e": 3, "g": 4, "ell": 5},
                                                             {(2, 6, 2): parse_up(os.path.join(d, "hz_N2_n6_full_x.Up2.mat"))},
                                                             {(2, 6): parse_mz(os.path.join(d, "mz_2_6.out"))}))
        # non-commuting pair
        mkcell([[1, 1], [0, 2]], [[3, 0], [0, 4]])
        expect_reject("non_commuting", lambda: gen_cell({"name": "x", "N": 2, "n": 6, "p": 2, "a": 1, "b": 2, "c": 3, "i0": 3, "e": 3, "g": 4, "ell": 5},
                                                        {(2, 6, 2): parse_up(os.path.join(d, "hz_N2_n6_full_x.Up2.mat"))},
                                                        {(2, 6): parse_mz(os.path.join(d, "mz_2_6.out"))}))
        # wrong eigenvalue expected (cell description disagrees with the matrix)
        mkcell([[1, 0], [0, 2]], [[3, 0], [0, 4]])
        expect_reject("bad_eigen", lambda: gen_cell({"name": "x", "N": 2, "n": 6, "p": 2, "a": 5, "b": 2, "c": 3, "i0": 3, "e": 3, "g": 4, "ell": 5},
                                                    {(2, 6, 2): parse_up(os.path.join(d, "hz_N2_n6_full_x.Up2.mat"))},
                                                    {(2, 6): parse_mz(os.path.join(d, "mz_2_6.out"))}))
        cell = {"name": "x", "N": 2, "n": 6, "p": 2, "a": 1, "b": 2, "c": 3, "i0": 3, "e": 3, "g": 4, "ell": 5}
        def run_cell(cc=cell):
            return gen_cell(cc, {(2, 6, 2): parse_up(os.path.join(d, "hz_N2_n6_full_x.Up2.mat"))},
                            {(2, 6): parse_mz(os.path.join(d, "mz_2_6.out"))})
        # TB-chi: a CHI line that is not the characteristic polynomial of F
        mkcell([[1, 0], [0, 2]], [[3, 0], [0, 4]], chi=[1, -7, 13])
        expect_reject("chi_mismatch", run_cell)
        # TB-chi: a correct CHI that is not the factorisation h-chi states (cusp cell)
        mkcell([[1, 0], [0, 1]], [[3, 0], [0, 4]])
        open(os.path.join(d, "mz_2_6.out"), "w", encoding="utf-8").write("BASIS 0xABC\nOP 2\nF [[\"3\",\"0\"],[\"0\",\"4\"]]\nCHI [\"1\",\"-7\",\"12\"]\n")
        expect_reject("hchi_mismatch", lambda: run_cell({"name": "x", "N": 2, "n": 6, "p": 2, "identity": True, "cusp": "x+2", "ell": 5}))
        # shape: module_Z F of a different size than U_p
        mkcell([[1, 0], [0, 2]], [[3, 0, 0], [0, 4, 0], [0, 0, 5]])
        expect_reject("shape_mismatch", run_cell)
        # the good fixture passes (the controls above are not vacuous)
        mkcell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-2, 0, 0], [0, -2, 1], [0, 0, 3]])
        open(os.path.join(d, "mz_2_6.out"), "w", encoding="utf-8").write(
            "BASIS 0xABC\nOP 2\nF [[\"-2\",\"0\",\"0\"],[\"0\",\"-2\",\"1\"],[\"0\",\"0\",\"3\"]]\nCHI [\"1\",\"1\",\"-8\",\"-12\"]\n")
        run_cell({"name": "x", "N": 2, "n": 6, "p": 2, "identity": True, "cusp": "x+2", "ell": 5})
        print("CONTROL good_fixture ACCEPTED")
        print("HZUP_SELFTEST controls=%d verdict=OK" % ok)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps")
    ap.add_argument("--mz")
    ap.add_argument("--binary-sha256", default="-")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.dumps and a.mz and a.out):
        ap.error("need --dumps --mz --out (or --selftest)")
    run(a.dumps, a.mz, a.binary_sha256, a.out)


if __name__ == "__main__":
    main()
