#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""L-HZ3 (slot 25, M-HZ4b partly): K3N certificates of the level-N two-system cells T3, T4 -> Lean data module.

UNTRUSTED witness generator (07-lean-plan.md §5 "Emission"): it reads the committed console outputs of the slot-25 shared
binary (descent record char_Z(T_p | H^1/tors) = TB-chi; DescentRecord.module(ell, K) = TB-mat, the EXPORT), extracts the free
block of T_p on L/ell^K, computes lambda = 1 + p^(k-1), mu (the other root of char_Z(T_p)), and per read the valuation v of
c(T_p) = T_p - mu, a witness entry and the exponent e = v_ell(lambda - mu) - v.  It re-runs every check of the Lean checker
GL2.HeckeZ.Exec.K3N.k3nCheck (and the cross-check chiReadsOK) and refuses on a failure.  The Lean kernel re-checks everything
by `decide +kernel`; nothing computed here is trusted.

usage: python3 -I lean/tools/hz_levelN_to_lean.py <data dir> <out .lean> <manifest .tsv>
"""
import hashlib, json, os, re, sys

# (name, N, n, p, [(file, ell, K, role)]) -- the cells of the brief (LHZ3_brief.md §3).  role "cert": a read of the K3N
# certificate; role "aside": a read checked on its own (readOK) and recorded, not part of the certificate (T4 at ell = 2,
# where the read reaches nothing: e = 0).
CELLS = [
    ("certT3", 11, 0, 2, [("t3_N11_n0_ell5_K2", 5, 2, "cert")]),
    ("certT4", 2, 6, 3, [("t4_N2_n6_ell17_K2", 17, 2, "cert"), ("t4_N2_n6_ell2_K9", 2, 9, "aside")]),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parse(path):
    out = {"charpoly": {}, "factored": {}, "gens": [], "ops": {}}
    cur = None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("charpoly "):
            _, p, rest = line.split(" ", 2)
            out["charpoly"][int(p)] = [int(x) for x in json.loads(rest)]
        elif line.startswith("factored "):
            _, p, rest = line.split(" ", 2)
            out["factored"][int(p)] = rest
        elif line.startswith("status "):
            toks = line.split()
            out["status"] = toks[1]
            out["route"] = toks[3]
            out["K"] = int(toks[5])
        elif line.startswith("gen "):
            _, kind, ex = line.split()
            out["gens"].append((kind, int(ex)))
        elif line.startswith("op "):
            cur = int(line.split()[1])
        elif line.startswith("matrix "):
            out["ops"][cur] = json.loads(line[len("matrix "):])
    return out


def roots_of_factored(s):
    """'(X + 2)^2 (X - 3)' -> [(-2, 2), (3, 1)] (root, multiplicity); refuses any non-linear factor."""
    res = []
    for m in re.finditer(r"\(X ([+-]) (\d+)\)(?:\^(\d+))?", s):
        sign, c, mult = m.group(1), int(m.group(2)), int(m.group(3) or 1)
        res.append((-c if sign == "+" else c, mult))
    if re.sub(r"\(X [+-] \d+\)(\^\d+)?", "", s).strip():
        raise SystemExit("REFUSED: non-linear factor in %r" % s)
    return res


def pmul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            r[i + j] += x * y
    return r


def val(x, ell, cap):
    if x == 0:
        return cap
    v = 0
    while x % ell == 0:
        x //= ell
        v += 1
    return v


def matmul(A, B, m):
    f = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(f)) % m for j in range(f)] for i in range(f)]


def peval_mat(chi_lo, D, m):
    f = len(D)
    R = [[0] * f for _ in range(f)]
    for a in reversed(chi_lo):  # Horner, as Exec.Cert.pevalMat
        R = matmul(D, R, m)
        for i in range(f):
            R[i][i] = (R[i][i] + a) % m
    return R


def is_prime(q):
    return q >= 2 and all(q % d for d in range(2, q))


def build(datadir):
    cells, inputs = [], []
    for name, N, n, p, reads in CELLS:
        k = n + 2
        lam = 1 + p ** (k - 1)
        rd, chi_hi, mu = [], None, None
        for fname, ell, K, role in reads:
            for ext in (".oc", ".txt"):
                inputs.append(os.path.join(datadir, "export", fname + ext))
            d = parse(os.path.join(datadir, "export", fname + ".txt"))
            if d.get("status") != "ok" or d["K"] != K:
                raise SystemExit("REFUSED: %s status %s K %s" % (fname, d.get("status"), d.get("K")))
            ch = d["charpoly"][p]
            rts = roots_of_factored(d["factored"][p])
            prod = [1]
            for r0, mlt in rts:
                for _ in range(mlt):
                    prod = pmul(prod, [1, -r0])  # high -> low
            if prod != ch:
                raise SystemExit("REFUSED: factored != exact at %s" % fname)
            if len(rts) != 2 or lam not in [r0 for r0, _ in rts]:
                raise SystemExit("REFUSED: %s: not two roots with lambda = %d: %s" % (fname, lam, rts))
            mu_here = [r0 for r0, _ in rts if r0 != lam][0]
            if chi_hi is None:
                chi_hi, mu = ch, mu_here
            elif chi_hi != ch or mu != mu_here:
                raise SystemExit("REFUSED: char_Z differs between the runs of %s" % name)
            free = [i for i, (kind, _) in enumerate(d["gens"]) if kind == "free"]
            M = d["ops"][p]
            mod = ell ** K
            Dm = [[M[i][j] % mod for j in free] for i in free]
            f = len(free)
            C = [[(Dm[i][j] - (mu if i == j else 0)) % mod for j in range(f)] for i in range(f)]
            v = min(val(C[i][j], ell, K) for i in range(f) for j in range(f))
            if v + 1 > K:
                raise SystemExit("REFUSED: %s: c(T_p) = 0 mod ell^K, raise K" % fname)
            wit = next((i, j) for i in range(f) for j in range(f) if val(C[i][j], ell, K) == v)
            vlm = val(lam - mu, ell, 10 ** 9)
            e = max(vlm - v, 0)
            chi_lo = list(reversed(ch))
            if any(x % mod for row in peval_mat(chi_lo, Dm, mod) for x in row):
                raise SystemExit("REFUSED: %s: chi(D) != 0 mod ell^K (TB-chi vs TB-mat)" % fname)
            if not is_prime(ell) or (lam - mu) % (ell ** (v + e)):
                raise SystemExit("REFUSED: %s: read check" % fname)
            rd.append(dict(ell=ell, K=K, f=f, dump=Dm, v=v, wit=wit, e=e, file=fname, route=d["route"], role=role, vlm=vlm,
                           sha=sha(os.path.join(datadir, "export", fname + ".txt"))))
        if not is_prime(p) or p % N == 0 or lam == mu:
            raise SystemExit("REFUSED: operator guard at %s" % name)
        aside = [r for r in rd if r["role"] == "aside"]
        rd = [r for r in rd if r["role"] == "cert"]
        ells = [r["ell"] for r in rd]
        if len(set(ells)) != len(ells):
            raise SystemExit("REFUSED: repeated ell at %s" % name)
        G = 1
        for r in rd:
            G *= r["ell"] ** r["e"]
        cells.append(dict(name=name, N=N, n=n, k=k, p=p, lam=lam, mu=mu, reads=rd, chi=list(reversed(chi_hi)),
                          G=G, gUp=abs(lam - mu), aside=aside))
    return cells, inputs


def lint(x):
    return "(%d)" % x if x < 0 else "%d" % x


def rlit(r):
    dump = "[" + ", ".join("[" + ", ".join(str(x) for x in row) + "]" for row in r["dump"]) + "]"
    return ("{ ell := %d, K := %d, f := %d,\n      dump := %s,\n      v := %d, wit := (%d, %d), e := %d }"
            % (r["ell"], r["K"], r["f"], dump, r["v"], r["wit"][0], r["wit"][1], r["e"]))


def dumpref(r):
    """LP v1.10 EXPORT route: every use of a read names the dump file and its sha256."""
    return "dump `export/%s.txt` sha256 `%s`" % (r["file"], r["sha"])


def emit(cells, binsha):
    o = []
    o.append("/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/")
    o.append("import GL2.HeckeZ.Exec.K3N\n")
    o.append("/-!\n# L-HZ3: K3N certificates of the level-`N` cells T3, T4 (generated; do not edit)\n")
    o.append("Generated by `lean/tools/hz_levelN_to_lean.py` (untrusted) from the console outputs in "
             "`docs/implementation/slot25/lhz3-data/export/` (slot-25 shared binary, sha256 `%s`; input and output sha256s in "
             "`docs/implementation/slot25/lhz3-data/manifest_lhz3.tsv`). `dump` is the free block of the EXPORT's `T_p` on `L/ℓ^K` "
             "(TB-mat), `mu` and `chi` the descent record's `char_Z(T_p | H¹/tors)` (TB-χ); `v`, `wit`, `e` are witnesses the kernel "
             "checks." % binsha)
    for c in cells:
        o.append("* `%s`: `(N, n) = (%d, %d)`, `k = %d`, `T_%d`, `λ = %d`, `μ = %d`, `|λ − μ| = %d`, lower bound `G = %d`; reads %s."
                 % (c["name"], c["N"], c["n"], c["k"], c["p"], c["lam"], c["mu"], c["gUp"], c["G"],
                    ", ".join("`ℓ = %d, K = %d, v = %d, e = %d` (%s route; %s)"
                              % (r["ell"], r["K"], r["v"], r["e"], r["route"], dumpref(r))
                              for r in c["reads"])
                    + "".join("; aside (not in the certificate) `ℓ = %d, K = %d, v = %d, e = %d` (%s route; %s)"
                              % (r["ell"], r["K"], r["v"], r["e"], r["route"], dumpref(r)) for r in c["aside"])))
    o.append("-/\n")
    o.append("namespace GL2.HeckeZ.Cells.LevelNData\n\nopen GL2.HeckeZ.Exec.K3N\n")
    for c in cells:
        rs = ["    " + rlit(r) for r in c["reads"]]
        o.append("/-- The cell `(N, n) = (%d, %d)`; %s. -/\ndef %s : K3NCert where\n  N := %d\n  k := %d\n  p := %d\n  lam := %s\n"
                 "  mu := %s\n  reads := [\n%s]\n  chi := [%s]\n"
                 % (c["N"], c["n"], "; ".join(dumpref(r) for r in c["reads"]), c["name"], c["N"], c["k"], c["p"], lint(c["lam"]), lint(c["mu"]), ",\n".join(rs),
                    ", ".join(lint(x) for x in c["chi"])))
        nm = c["name"]
        o.append("/-- `%s` passes K3N (kernel). -/\ntheorem %s_check : k3nCheck %s = true := by decide +kernel\n" % (nm, nm, nm))
        o.append("/-- TB-χ against TB-mat: `χ(D) ≡ 0 mod ℓ^K` on every read of `%s` (a guard). -/\n"
                 "theorem %s_chi : chiReadsOK %s = true := by decide +kernel\n" % (nm, nm, nm))
        o.append("/-- The bounds of `%s`: `G = %d`, `|λ − μ| = %d`. -/\ntheorem %s_bounds : %s.G = %d ∧ %s.gUp = %d := by decide +kernel\n"
                 % (nm, c["G"], c["gUp"], nm, nm, c["G"], nm, c["gUp"]))
        comp = "true" if c["G"] == c["gUp"] else "false"
        o.append("/-- `%s` is %scomplete. -/\ntheorem %s_complete : %s.complete = %s := by decide +kernel\n"
                 % (nm, "" if comp == "true" else "not ", nm, nm, comp))
        for r in c["aside"]:
            an = "read%s_%d" % (nm[len("cert"):], r["ell"])
            o.append("/-- The read of `%s`'s operator `T_%d` at `ℓ = %d` (`K = %d`, %s route), checked on its own and **not** part of the "
                     "certificate: its valuation `v = %d` equals `v_ℓ(λ − μ) = %d`, so it gives the lower bound `ℓ^0` (`e = 0`); %s. -/\n"
                     "def %s : K3Read :=\n  %s\n" % (nm, c["p"], r["ell"], r["K"], r["route"], r["v"], r["vlm"], dumpref(r), an, rlit(r)))
            o.append("/-- `%s` passes the read check at `(λ, μ) = (%d, %d)` (kernel). -/\ntheorem %s_check : readOK %s %s %s = true := by "
                     "decide +kernel\n" % (an, c["lam"], c["mu"], an, an, lint(c["lam"]), lint(c["mu"])))
            o.append("/-- TB-χ against TB-mat on `%s` (a guard). -/\ntheorem %s_chi : chiReadsOK { %s with reads := [%s] } = true := by "
                     "decide +kernel\n" % (an, an, nm, an))
            o.append("/-- `v = v_ℓ(λ − μ)`: `ℓ^v ∣ λ − μ` and `ℓ^{v+1} ∤ λ − μ`, so the read reaches nothing at `ℓ`. -/\n"
                     "theorem %s_exact : %s.v = %d ∧ %s.e = 0 ∧ (%s - %s) %% ((%d : Int) ^ %d) = 0 ∧ (%s - %s) %% ((%d : Int) ^ %d) ≠ 0 := by "
                     "decide +kernel\n" % (an, an, r["v"], an, lint(c["lam"]), lint(c["mu"]), r["ell"], r["v"], lint(c["lam"]),
                                           lint(c["mu"]), r["ell"], r["v"] + 1))
    o.append("end GL2.HeckeZ.Cells.LevelNData\n")
    return "\n".join(o)


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    datadir, out, man = sys.argv[1:]
    binsha = open(os.path.join(datadir, "binary.sha256")).read().split()[0]
    cells, inputs = build(datadir)
    text = emit(cells, binsha)
    with open(out, "w") as fh:
        fh.write(text)
    with open(man, "w") as fh:
        fh.write("binary.sha256\t%s\n" % binsha)
        for pth in inputs:
            fh.write("input\t%s\t%s\n" % (os.path.relpath(pth, datadir), sha(pth)))
        fh.write("output\t%s\t%s\n" % (os.path.basename(out), sha(out)))
    for c in cells:
        print("HZ_LEVELN_GEN cell=%s N=%d n=%d p=%d lam=%d mu=%d G=%d gUp=%d complete=%s reads=%s"
              % (c["name"], c["N"], c["n"], c["p"], c["lam"], c["mu"], c["G"], c["gUp"], c["G"] == c["gUp"],
                 ";".join("%d:K%d:v%d:e%d" % (r["ell"], r["K"], r["v"], r["e"]) for r in c["reads"])))


if __name__ == "__main__":
    main()
