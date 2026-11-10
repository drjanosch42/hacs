#!/usr/bin/env python3
"""satcert_to_lean.py (L17-2): HECKE_DUMP_SAT_CERT dumps -> lean/GL2/Algorithms/SatCertData.lean.

UNTRUSTED witness generator for the Lean stable-certificate checker (GL2/Algorithms/SatCert.lean,
[GT] chk:stablecert).  For each dumped block (Z/ell^K)^f with the matrices of the T_q it
  1. builds a straight-line program spanning A'_K = Z-span of the monomials in the T_q (closure: a
     product g * P is kept iff it enlarges the Z/ell^K-span, measured by the Smith valuations);
  2. computes A'_K[ell] by the Smith form over Z/ell^K with row-transform tracking: for each invariant
     factor ell^v (v < K) of the span, ell^{K-1-v} times the corresponding row of U G lies in A'_K[ell];
  3. puts those elements in pivot form (RREF over F_ell of W / ell^{K-1}).
Nothing here is trusted: the kernel re-checks every program step, every witness combination, the
ell-torsion and the pivot pattern (satCheck, decide +kernel), and stable_of_spec turns that into
[C] cor:stablecert(1).

usage: satcert_to_lean.py OUT.lean DUMP1.txt [DUMP2.txt ...]
"""
import hashlib
import os
import sys


def parse(path):
    d = {"gens": [], "gen_primes": []}
    cur = None
    with open(path) as fh:
        lines = fh.read().splitlines()
    if not lines or lines[0] != "SATCERT 1":
        raise SystemExit(f"{path}: not a SATCERT 1 dump")
    for ln in lines[1:]:
        if not ln or ln.startswith("#"):
            continue
        t = ln.split()
        k = t[0]
        if k in ("n", "ell", "K", "f", "rank_T", "D"):
            d[k] = int(t[1])
        elif k in ("primes", "system"):
            d[k] = [int(x) for x in t[1:]]
        elif k in ("A1_hilbert", "AK_hilbert", "verdict", "stable_cert"):
            d[k] = " ".join(t[1:])
        elif k == "gen":
            cur = []
            d["gens"].append(cur)
            d["gen_primes"].append(int(t[1]))
        elif k == "row":
            cur.append([int(x) for x in t[1:]])
        elif k == "end":
            break
        else:
            raise SystemExit(f"{path}: unknown line {ln!r}")
    for g in d["gens"]:
        assert len(g) == d["f"] and all(len(r) == d["f"] for r in g), "shape"
    d["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return d


def val(x, p):
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def snf(rows, p, K):
    """Smith form over Z/p^K: returns (valuations, U) with (U G V) = diag(p^v_i), U tracked."""
    m = p ** K
    A = [[x % m for x in r] for r in rows]
    R = len(A)
    C = len(A[0]) if R else 0
    U = [[int(i == j) for j in range(R)] for i in range(R)]
    vals = []
    t = 0
    while t < min(R, C):
        best = None
        for i in range(t, R):
            for j in range(t, C):
                x = A[i][j]
                if x:
                    v = val(x, p)
                    if best is None or v < best[0]:
                        best = (v, i, j)
        if best is None:
            break
        v, i, j = best
        A[t], A[i] = A[i], A[t]
        U[t], U[i] = U[i], U[t]
        for r in A:
            r[t], r[j] = r[j], r[t]
        pv = p ** v
        uinv = pow(A[t][t] // pv, -1, m)
        for i2 in range(t + 1, R):
            x = A[i2][t]
            if x:
                c = (x // pv) * uinv % m
                A[i2] = [(a - c * b) % m for a, b in zip(A[i2], A[t])]
                U[i2] = [(a - c * b) % m for a, b in zip(U[i2], U[t])]
        for j2 in range(t + 1, C):
            x = A[t][j2]
            if x:
                c = (x // pv) * uinv % m
                for r in A:
                    r[j2] = (r[j2] - c * r[t]) % m
        vals.append(v)
        t += 1
    return vals, U


def span_length(rows, p, K):
    if not rows:
        return 0
    vals, _ = snf(rows, p, K)
    return sum(K - v for v in vals if v < K)


def matmul(A, B, m):
    f = len(A)
    return [[sum(A[i][l] * B[l][j] for l in range(f)) % m for j in range(f)] for i in range(f)]


def flat(M):
    return [x for r in M for x in r]


def build(d):
    p, K, f = d["ell"], d["K"], d["f"]
    m = p ** K
    gens = [[[x % m for x in r] for r in g] for g in d["gens"]]
    prog = [(("one",), [[int(i == j) for j in range(f)] for i in range(f)])]
    gidx = []
    for j, g in enumerate(gens):
        gidx.append(len(prog))
        prog.append((("gen", j), g))
    sel = [0]                      # program indices spanning A'_K
    cur = span_length([flat(prog[0][1])], p, K)
    q = 0
    while q < len(sel):
        s = sel[q]
        q += 1
        for j, g in enumerate(gens):
            cand = matmul(g, prog[s][1], m)
            ln = span_length([flat(prog[k][1]) for k in sel] + [flat(cand)], p, K)
            if ln > cur:
                prog.append((("mul", gidx[j], s), cand))
                sel.append(len(prog) - 1)
                cur = ln
    G = [flat(prog[k][1]) for k in sel]
    vals, U = snf(G, p, K)
    W, CO = [], []
    for i, v in enumerate(vals):
        if v >= K:
            continue
        s = p ** (K - 1 - v)
        co = [0] * len(prog)
        for k, pk in enumerate(sel):
            co[pk] = (co[pk] + s * U[i][k]) % m
        w = [sum(co[pk] * G[k][e] for k, pk in enumerate(sel)) % m for e in range(f * f)]
        assert all((p * x) % m == 0 for x in w)
        W.append(w)
        CO.append(co)
    r = (f + 1) // 2
    if len(W) != r:
        raise SystemExit(f"n={d['n']} ell={p}: dim A_K/ell = {len(W)} != ceil(f/2) = {r} -- not a stable block")
    # pivot form: RREF over F_p of W / p^{K-1}
    pk1 = p ** (K - 1)
    Y = [[(x // pk1) % p for x in w] for w in W]
    E = [[int(i == j) for j in range(r)] for i in range(r)]
    piv = []
    row = 0
    for col in range(f * f):
        if row == r:
            break
        sr = next((i for i in range(row, r) if Y[i][col]), None)
        if sr is None:
            continue
        Y[row], Y[sr] = Y[sr], Y[row]
        E[row], E[sr] = E[sr], E[row]
        inv = pow(Y[row][col], -1, p)
        Y[row] = [x * inv % p for x in Y[row]]
        E[row] = [x * inv % p for x in E[row]]
        for i in range(r):
            if i != row and Y[i][col]:
                c = Y[i][col]
                Y[i] = [(a - c * b) % p for a, b in zip(Y[i], Y[row])]
                E[i] = [(a - c * b) % p for a, b in zip(E[i], E[row])]
        piv.append(col)
        row += 1
    assert row == r, "witnesses dependent mod ell"
    wits = []
    for i in range(r):
        co = [sum(E[i][j] * CO[j][k] for j in range(r)) % m for k in range(len(prog))]
        w = [sum(co[k] * flat(prog[k][1])[e] for k in range(len(prog))) % m for e in range(f * f)]
        rows = [w[a * f:(a + 1) * f] for a in range(f)]
        wits.append((co, rows, (piv[i] // f, piv[i] % f)))
    return prog, wits


def lmat(M):
    return "[" + ", ".join("[" + ", ".join(str(x) for x in r) + "]" for r in M) + "]"


def lstep(s):
    if s[0] == "one":
        return ".one"
    if s[0] == "gen":
        return f"(.gen {s[1]})"
    return f"(.mul {s[1]} {s[2]})"


def emit(dumps, out):
    names, L = [], []
    L.append("import GL2.Algorithms.SatCert\n")
    L.append("/-! GENERATED by lean/tools/satcert_to_lean.py from the HECKE_DUMP_SAT_CERT dumps of")
    L.append("`HeckeSatCertDumpValidation` (L17-2): the matrices `gens` are real C++ output, re-typeset only; the program")
    L.append("and the witnesses are untrusted and re-checked by the kernel. Do not edit by hand. -/\n")
    L.append("namespace GL2.SatChk.Data")
    L.append("open GL2.SatChk\n")
    for path, d in dumps:
        prog, wits = build(d)
        nm = f"cert_n{d['n']}_ell{d['ell']}_b{os.path.basename(path).rsplit('_b', 1)[1].split('.')[0]}"
        names.append(nm)
        L.append(f"/-- `{os.path.basename(path)}` (sha256 `{d['sha256'][:16]}…`): `(n, ℓ) = ({d['n']}, {d['ell']})`, system "
                 f"`({', '.join(map(str, d['system']))})` at primes `{d['primes']}`, `K = {d['K']}`, `f = {d['f']}`, "
                 f"`A_1` Hilbert `{d.get('A1_hilbert', '?')}`, `A_K` Hilbert `{d.get('AK_hilbert', '?')}`. -/")
        L.append(f"def {nm} : SatCert where")
        L.append(f"  n := {d['n']}")
        L.append(f"  p := {d['ell']}")
        L.append(f"  K := {d['K']}")
        L.append(f"  f := {d['f']}")
        L.append(f"  system := {d['system']}")
        L.append(f"  primes := {d['primes']}")
        L.append("  gens := [" + ",\n    ".join(lmat(g) for g in d["gens"]) + "]")
        L.append("  prog := [" + ",\n    ".join(f"({lstep(s)}, {lmat(M)})" for s, M in prog) + "]")
        L.append("  wits := [" + ",\n    ".join(f"⟨{co}, {lmat(rows)}, ({pos[0]}, {pos[1]})⟩" for co, rows, pos in wits) + "]\n")
        L.append(f"theorem {nm}_check : satCheck {nm} = true := by decide +kernel\n")
        L.append(f"theorem {nm}_spec : SatSpec {nm} := satCheck_sound _ {nm}_check\n")
    L.append("/-- The real certificates, one per dumped block. -/")
    L.append("def allCerts : List SatCert := [" + ", ".join(names) + "]\n")
    L.append("/-- **`chk:stablecert-real`**: every dumped certificate is accepted. -/")
    L.append("theorem allCerts_check : allCerts.all satCheck = true := by decide +kernel\n")
    # tamper: the first witness's pivot entry set to 0 -> rejected
    nm0 = names[0]
    L.append("/-- A tampered certificate (the first witness's pivot entry zeroed) is rejected. -/")
    L.append(f"def tampered : SatCert :=")
    L.append(f"  {{ {nm0} with wits := match {nm0}.wits with")
    L.append(f"    | w :: ws => {{ w with rows := w.rows.set w.pos.1 ((w.rows.getD w.pos.1 []).set w.pos.2 0) }} :: ws")
    L.append(f"    | [] => [] }}\n")
    L.append("theorem tampered_rejected : satCheck tampered = false := by decide +kernel\n")
    # L18-4 (L17-2 audit MINOR-1): one tamper control per rejected check
    L.append(TAMPER_LEAN.replace("NM0", nm0))
    L.append("end GL2.SatChk.Data")
    open(out, "w").write("\n".join(L) + "\n")
    return names


TAMPER_LEAN = """
/-! L18-4 (L17-2 audit MINOR-1): one tamper control per rejected check of `satCheck`. -/

/-- `1 ≤ K` fails. -/
def tampered_K0 : SatCert := { NM0 with K := 0 }

theorem tampered_K0_rejected : satCheck tampered_K0 = false := by decide +kernel

/-- `r = ⌈f/2⌉` fails: the last witness dropped. -/
def tampered_short : SatCert := { NM0 with wits := NM0.wits.take (NM0.wits.length - 1) }

theorem tampered_short_rejected : satCheck tampered_short = false := by decide +kernel

/-- `stepOK` fails at a `.gen` step: the generator index out of range. -/
def tampered_gen : SatCert := { NM0 with prog := NM0.prog.set 1 (.gen 99, NM0.P 1) }

theorem tampered_gen_rejected : satCheck tampered_gen = false := by decide +kernel

/-- `stepOK` fails at a `.mul` step: the last step relabelled `P_1 · P_1`, its stored matrix kept. -/
def tampered_mul : SatCert :=
  { NM0 with prog := NM0.prog.set (NM0.prog.length - 1) (.mul 1 1, NM0.P (NM0.prog.length - 1)) }

theorem tampered_mul_rejected : satCheck tampered_mul = false := by decide +kernel

/-- `witOK`'s combination test fails: the first witness's first coefficient shifted by one, its matrix kept. -/
def tampered_comb : SatCert :=
  { NM0 with wits := match NM0.wits with
    | w :: ws => { w with coeffs := (w.coeffs.getD 0 0 + 1) :: w.coeffs.drop 1 } :: ws
    | [] => [] }

theorem tampered_comb_rejected : satCheck tampered_comb = false := by decide +kernel

/-- `witOK`'s torsion test fails: the first witness replaced by the identity `P_0` (consistent, not killed by `p`). -/
def tampered_tors : SatCert :=
  { NM0 with wits := match NM0.wits with
    | w :: ws => { w with coeffs := [1], rows := NM0.P 0, pos := (0, 0) } :: ws
    | [] => [] }

theorem tampered_tors_rejected : satCheck tampered_tors = false := by decide +kernel

/-- `pivotOK` fails with consistent rows: the first witness's pivot moved to an entry that is not `p^{K-1}`. -/
def tampered_pivot : SatCert :=
  { NM0 with wits := match NM0.wits with
    | w :: ws => { w with pos := (1, 0) } :: ws
    | [] => [] }

theorem tampered_pivot_rejected : satCheck tampered_pivot = false := by decide +kernel
"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    dumps = [(p, parse(p)) for p in sys.argv[2:]]
    names = emit(dumps, sys.argv[1])
    print(f"SATCERT_TO_LEAN certs={len(names)} out={sys.argv[1]}")
