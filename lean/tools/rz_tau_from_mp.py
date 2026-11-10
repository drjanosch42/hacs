#!/usr/bin/env python3
"""MP-L2 (slot 22): the X_bulk descriptor of R_Z, and the subdivision tables M7 <-> R_Z, as Lean data.

usage: rz_tau_from_mp.py <lean/GL2/MP/RZDump.lean> <lean/GL2/Checkers/TauP2Data.lean> <out.lean>

Reads the R_Z descriptor re-typeset from the C++ dump (GL2/MP/RZDump.lean: orbits with stabiliser generator and order,
points (orbit, g), maximal simplices) and the seven-vertex dump tauP2 (GL2/Checkers/TauP2Data.lean: simplices), and writes
one Lean module GL2.MP.RZTau with

* `tauRZ : TauData` -- the simplicial Gamma-complex of R_Z on the same point labels: every face of every maximal simplex
  (vertex lists increasing in the label), the orbit representatives, and EVERY transporter gamma with
  gamma . vtx(s2) = vtx(s1) (with its vertex correspondence), searched in the cosets g_p Gamma_o g_w^{-1};
* the subdivision tables (formal sums of simplices of the lifted domains, all with transporter 1):
  `sdTab` (M7 -> R_Z: an edge 4-6 / 5-6 and the triangles on it split at the new vertex 7 / 8), `cTab` (R_Z -> M7: the
  simplicial collapse 7 |-> 4, 8 |-> 5), `h0Tab`, `h1Tab` (the chain homotopy id - sd c = d h + h d on R_Z).

Nothing here is trusted: GL2/MP/RZSubdiv.lean decides the X_bulk axioms of `tauRZ`, the (AC)-bridge checks against
`rzMP`, and every identity of the tables (`decide +kernel`).  Pure: no Lean, no C++.
"""
import itertools
import re
import sys


def parse_mats(s):
    return [tuple(int(x) for x in m) for m in re.findall(r"⟨\(?(-?\d+)\)?, \(?(-?\d+)\)?, \(?(-?\d+)\)?, \(?(-?\d+)\)?⟩", s)]


def block(text, name):
    m = re.search(r"def " + name + r" : [^\n]*:= \[(.*?)\]\n\n", text, re.S)
    if not m:
        raise SystemExit("cannot find " + name)
    return m.group(1)


def mul(a, b):
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3])


def adj(a):
    return (a[3], -a[1], -a[2], a[0])


I = (1, 0, 0, 1)


def main():
    dump, taup2, out = sys.argv[1], sys.argv[2], sys.argv[3]
    text = open(dump).read()
    orb_rows = re.findall(r"\((\d+), (⟨[^⟩]*⟩), (\d+)\)", block(text, "orbits"))
    orbits = [(int(l), parse_mats(g)[0], int(o)) for (l, g, o) in orb_rows]
    pt_rows = re.findall(r"\((\d+), (⟨[^⟩]*⟩), (true|false)\)", block(text, "points"))
    points = [(int(o), parse_mats(g)[0]) for (o, g, _) in pt_rows]
    top = [[int(x) for x in t.split(",")] for t in re.findall(r"\[([\d, ]+)\]", re.search(r"def top : List \(List ℕ\) := \[(.*)\]\n", text).group(1))]

    stab = []
    for (_, gen, order) in orbits:
        l, x = [], I
        for _ in range(order):
            l.append(x)
            x = mul(gen, x)
        stab.append(l)

    def maps(g, w, p):
        ow, gw = points[w]
        op, gp = points[p]
        return ow == op and mul(adj(gp), mul(g, gw)) in stab[op]

    simp = {0: set(), 1: set(), 2: set()}
    for t in top:
        for r in (1, 2, 3):
            for c in itertools.combinations(sorted(t), r):
                simp[r - 1].add(c)
    simp = {q: sorted(v) for q, v in simp.items()}
    idx = {q: {s: i for i, s in enumerate(simp[q])} for q in simp}

    trs, rep = [], {}
    for q in (0, 1, 2):
        for a in simp[q]:
            for b in simp[q]:
                p0 = a[0]
                cands = []
                for w in b:
                    if points[w][0] == points[p0][0]:
                        for h in stab[points[p0][0]]:
                            g = mul(mul(points[p0][1], h), adj(points[w][1]))
                            if g not in cands:
                                cands.append(g)
                for g in cands:
                    perm = []
                    for j in range(q + 1):
                        ii = [i for i in range(q + 1) if maps(g, b[i], a[j])]
                        if len(ii) != 1:
                            break
                        perm.append(ii[0])
                    if len(perm) == q + 1 and sorted(perm) == list(range(q + 1)):
                        trs.append((q, idx[q][a], idx[q][b], g, perm))
        for a in simp[q]:
            ia = idx[q][a]
            rep[(q, ia)] = min(t[2] for t in trs if t[0] == q and t[1] == ia)

    # the seven-vertex model's simplices (labels 0..6 are the same points)
    tp = open(taup2).read()
    m7 = [(int(q), int(i), [int(x) for x in vs.split(",")]) for (q, i, vs) in
          re.findall(r"\((\d), (\d+), \[([\d, ]+)\]\)", re.search(r"simplices := \[(.*?)\]\n", tp).group(1))]
    m7idx = {(q, tuple(vs)): i for (q, i, vs) in m7}

    def sign_to(tup, stored):
        perm = [stored.index(v) for v in tup]
        inv = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j])
        return -1 if inv % 2 else 1

    def r_term(tup):  # an oriented R_Z simplex as +-(stored simplex)
        s = tuple(sorted(tup))
        return (sign_to(tup, list(s)), idx[len(s) - 1][s])

    def m_term(tup):
        s = tuple(sorted(tup))
        return (sign_to(tup, list(s)), m7idx[(len(s) - 1, s)])

    # sd : M7 -> R_Z (split the edges 4-6, 5-6 at 7, 8)
    split = {(4, 6): 7, (5, 6): 8}
    sd = {}
    for (q, i, vs) in m7:
        terms = []
        e = [s for s in split if set(s) <= set(vs)]
        if not e:
            terms = [r_term(tuple(vs))]
        else:
            (a, b), m = e[0], split[e[0]]
            # orientation: [.., a, .., b, ..] -> [.., a, .., m, ..] + [.., m, .., b, ..]
            t1 = tuple(m if v == b else v for v in vs)
            t2 = tuple(m if v == a else v for v in vs)
            terms = [r_term(t1), r_term(t2)]
        sd[(q, i)] = terms
    # c : R_Z -> M7 (7 |-> 4, 8 |-> 5)
    phi = {v: v for v in range(7)}
    phi[7], phi[8] = 4, 5
    cm = {}
    for q in (0, 1, 2):
        for s in simp[q]:
            im = tuple(phi[v] for v in s)
            cm[(q, idx[q][s])] = [] if len(set(im)) < len(im) else [m_term(im)]
    # h0, h1 on R_Z
    h0 = {(0, idx[0][(v,)]): [] for (v,) in simp[0]}
    h0[(0, idx[0][(7,)])] = [r_term((4, 7))]
    h0[(0, idx[0][(8,)])] = [r_term((5, 8))]
    h1 = {(1, idx[1][s]): [] for s in simp[1]}
    for (m, a, x) in ((7, 4, 1), (8, 5, 2)):  # the edge x-m: h1 = -[x, a, m]
        e = tuple(sorted((x, m)))
        sg, _ = r_term((x, m))
        s3, i3 = r_term((x, a, m))
        h1[(1, idx[1][e])] = [(-sg * s3, i3)]

    def lst(terms):
        return "[" + ", ".join(f"(({c} : ℤ), M22.I, {i})" for (c, i) in terms) + "]"

    def tab(name, doc, d, dims):
        lines = [f"/-- {doc} -/", f"def {name} : ℕ → ℕ → List (ℤ × M22 × ℕ)"]
        for q in dims:
            for (qq, i) in sorted(k for k in d if k[0] == q):
                lines.append(f"  | {q}, {i} => {lst(d[(qq, i)])}")
        lines.append("  | _, _ => []")
        return "\n".join(lines)

    def li(x):
        return f"({x})" if x < 0 else str(x)

    def mat(g):
        return "⟨" + ", ".join(li(x) for x in g) + "⟩"

    simps = ", ".join(f"({q}, {idx[q][s]}, [{', '.join(map(str, s))}])" for q in (0, 1, 2) for s in simp[q])
    reps = ", ".join(f"({q}, {i}, {rep[(q, i)]})" for q in (0, 1, 2) for i in range(len(simp[q])))
    trl = ",\n    ".join(f"⟨{q}, {a}, {b}, {mat(g)}, [{', '.join(map(str, p))}]⟩" for (q, a, b, g, p) in trs)
    body = f"""import GL2.MP.Descriptor

/-! GENERATED by lean/tools/rz_tau_from_mp.py from GL2/MP/RZDump.lean (the C++ MP dump) and the simplex list of
GL2/Checkers/TauP2Data.lean.  Do not edit by hand.  Nothing here is trusted: GL2/MP/RZSubdiv.lean decides the `X_bulk`
axioms of `tauRZ`, the (AC)-bridge checks against `rzMP` and every identity of the tables (`decide +kernel`).

`tauRZ`: {len(simp[0])} vertices, {len(simp[1])} edges, {len(simp[2])} triangles of the lifted domain of `R_ℤ`
(point labels of the dump), {len(trs)} transporters. -/

namespace GL2.MP.RZTau
open GL2.TauChk GL2.MP

/-- **`R_ℤ` as a `τ`-dump-shaped `X_bulk` descriptor** (no `τ`-table: `slots` is empty). -/
def tauRZ : TauData where
  N := 1
  m := 0
  g := M22.I
  s := []
  simplices := [{simps}]
  orbitRep := [{reps}]
  trs := [
    {trl}]
  slots := []
  regime := "MP R_Z (generated)"
  fnv := 0

{tab("sdTab", "The subdivision `sd : C(M₇) → C(R_ℤ)` on the simplices of `M₇` (the edges `4-6`, `5-6` split at `7`, `8`).", sd, (0, 1, 2))}

{tab("cTab", "The collapse `c : C(R_ℤ) → C(M₇)` (`7 ↦ 4`, `8 ↦ 5`; degenerate images are `0`).", cm, (0, 1, 2))}

{tab("h0Tab", "The homotopy `h₀ : C_0(R_ℤ) → C_1(R_ℤ)` (`7 ↦ [4, 7]`, `8 ↦ [5, 8]`).", h0, (0,))}

{tab("h1Tab", "The homotopy `h₁ : C_1(R_ℤ) → C_2(R_ℤ)` (`[1, 7] ↦ -[1, 4, 7]`, `[2, 8] ↦ -[2, 5, 8]`).", h1, (1,))}

end GL2.MP.RZTau
"""
    open(out, "w").write(body)
    print(f"RZ_TAU vertices={len(simp[0])} edges={len(simp[1])} triangles={len(simp[2])} transporters={len(trs)} out={out}")


if __name__ == "__main__":
    main()
