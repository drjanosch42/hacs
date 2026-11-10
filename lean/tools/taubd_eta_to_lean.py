#!/usr/bin/env python3
"""L18-3: turn the HECKE_DUMP_TAU_PAIR dump (HeckeTauDumpValidation) into Lean data.

usage: taubd_eta_to_lean.py <dump-dir> <lean-root> --golden <GoldenReference.inc>

For every (p, nu) with <dump-dir>/{tau,taubd,eta}_p<p>_r<nu>.txt, writes
  <lean-root>/GL2/Checkers/PairData/P<p>R<nu>.lean
holding (namespace GL2.TauChk.Data)
  tauP<p>R<nu>    : TauData     the bulk tau table (for (2, 1): `tauP2` of TauP2Data, after checking
                                that tau_p2_r1.txt is tau_p2.txt line for line);
  tauBdP<p>R<nu>  : TauBdData   the tau_bd table over it;
  etaP<p>R<nu>    : EtaData     the eta table over that;
  goldenTau / goldenTauBd / goldenEta hashes copied verbatim from the golden .inc where an object
  exists (tau / tau_bd at every (p, nu): `n=2 ell=5 ext cone_action` for (2, 1), else
  `n=2 ell=<ell> ext tau`; eta only at (2, 1)), or the dump's own HashScalar64 pinned (option (a))
  where none does -- the file says which.
Nothing is computed here except HashScalar64 for the pinned eta hashes beyond (2, 1); every table
value is re-typeset only.  The kernel checks live in GL2/Checkers/TauBdP2.lean, EtaP2.lean, PairGrid.lean.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_dump_to_lean import lean_int, mat, parse_tau  # noqa: E402

FNV_OFF = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
GRID_K = 4  # tauCheck pieces per grid table (kernel memory, see GL2/Checkers/TauSplit.lean)


def hash_scalar64(v):
    h = FNV_OFF
    for i in range(8):
        h = ((h ^ ((v >> (8 * i)) & 0xff)) * FNV_PRIME) % (1 << 64)
    return h


def parse_slots(path, magic):
    d = {"s": [], "bdry": [], "slots": []}
    cur = None
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.startswith("#")]
    assert lines[0] == [magic, "1"], lines[0]
    for t in lines[1:]:
        k = t[0]
        if k in ("p", "nu", "ell", "nsym", "N", "m", "fnv", "tau_fnv", "taubd_fnv"):
            d[k] = int(t[1])
        elif k == "g":
            d["g"] = t[1:5]
        elif k == "s":
            assert int(t[1]) == len(d["s"]) + 1
            d["s"].append(t[2:6])
        elif k == "bsimplex":
            d["bdry"].append((int(t[1]), int(t[2])))
        elif k == "slot":
            cur = [int(t[1]), int(t[2]), int(t[3]), int(t[4]), []]
            d["slots"].append(cur)
        elif k == "term":
            cur[4].append((int(t[1]), t[2:6], int(t[6])))
        elif k == "regime":
            d["regime"] = t[1]
        elif k == "end":
            pass
        else:
            raise SystemExit(f"{path}: unknown line {t}")
    for sl in d["slots"]:
        assert sl[3] == len(sl[4]), sl
    return d


def golden(inc, point, obj):
    pat = re.compile(r'\{"' + re.escape(point) + r'", "' + re.escape(obj) + r'", \'B\', 1, 1, (0x[0-9a-f]+)ull\}')
    with open(inc) as f:
        for line in f:
            m = pat.search(line)
            if m:
                return m.group(1)
    return None


def slots_lean(slots):
    out = []
    for i, q, s, _, terms in slots:
        ts = ", ".join(f"⟨{lean_int(c)}, {mat([lean_int(x) for x in g])}, {sg}⟩" for c, g, sg in terms)
        out.append(f"    ⟨{i}, {q}, {s}, [{ts}]⟩")
    return ",\n".join(out)


def tau_lean(name, d, what):
    L = [f"/-- The bulk τ-table of {what}, as dumped (`tau_*.txt`). -/", f"def {name} : TauData where"]
    L.append(f"  N := {d['N']}")
    L.append(f"  m := {d['m']}")
    L.append(f"  g := {mat([lean_int(x) for x in d['g']])}")
    L.append("  s := [" + ", ".join(mat([lean_int(x) for x in s]) for s in d["s"]) + "]")
    L.append("  simplices := [" + ", ".join(f"({q}, {i}, {vs})" for q, i, vs in d["simplices"]) + "]")
    L.append("  orbitRep := [" + ", ".join(f"({q}, {a}, {b})" for q, a, b in d["orbitrep"]) + "]")
    L.append("  trs := [")
    L.append(",\n".join(f"    ⟨{q}, {a}, {b}, {mat([lean_int(x) for x in g])}, {p}⟩" for q, a, b, g, p in d["trs"]))
    L.append("  ]")
    L.append("  slots := [")
    L.append(slots_lean([[i, q, s, n, [(c, g, sg) for c, g, sg in ts]] for i, q, s, n, ts in d["slots"]]))
    L.append("  ]")
    L.append(f"  regime := \"{d['regime']}\"")
    L.append(f"  fnv := {d['fnv']}")
    L.append("")
    return L


def same_coset_header(t, b, path):
    assert t["N"] == b["N"] and t["m"] == b["m"], path
    assert [int(x) for x in t["g"]] == [int(x) for x in b["g"]], path
    assert [[int(x) for x in s] for s in t["s"]] == [[int(x) for x in s] for s in b["s"]], path


def main():
    args = sys.argv[1:]
    assert "--golden" in args, __doc__
    i = args.index("--golden")
    inc = args[i + 1]
    del args[i:i + 2]
    dump_dir, root = args
    outdir = os.path.join(root, "GL2", "Checkers", "PairData")
    os.makedirs(outdir, exist_ok=True)
    pairs = []
    for path in sorted(glob.glob(os.path.join(dump_dir, "taubd_p*_r*.txt"))):
        m = re.search(r"taubd_p(\d+)_r(\d+)\.txt$", path)
        pairs.append((int(m.group(1)), int(m.group(2))))
    for p, nu in pairs:
        suf = f"_p{p}_r{nu}"
        tau = parse_tau(os.path.join(dump_dir, f"tau{suf}.txt"))
        bd = parse_slots(os.path.join(dump_dir, f"taubd{suf}.txt"), "TAUBDDUMP")
        eta = parse_slots(os.path.join(dump_dir, f"eta{suf}.txt"), "ETADUMP")
        same_coset_header(tau, bd, "taubd" + suf)
        same_coset_header(tau, eta, "eta" + suf)
        assert eta["bdry"] == bd["bdry"], "eta / taubd boundary lists differ"
        assert eta["tau_fnv"] == tau["fnv"] and eta["taubd_fnv"] == bd["fnv"], "base-table hashes differ"
        assert (bd["p"], bd["nu"]) == (p, nu)
        tag = f"P{p}R{nu}"
        ell = bd["ell"]
        what = f"`HeckeCosetTables({p}, {nu}, {ell}, {bd['nsym']})` (N = {bd['N']}, m = {bd['m']})"
        L = ["import GL2.Checkers.EtaCheck"]
        if (p, nu) == (2, 1):
            def strip(fn):
                with open(os.path.join(dump_dir, fn)) as f:
                    return [l for l in f if not l.startswith("#")]
            assert strip("tau_p2.txt") == strip("tau_p2_r1.txt"), "tau_p2_r1.txt is not tau_p2.txt"
            L.append("import GL2.Checkers.TauP2Data")
        L += ["", "/-! GENERATED by lean/tools/taubd_eta_to_lean.py from the HECKE_DUMP_TAU_PAIR dump of",
              "`HeckeTauDumpValidation` (L18-3). Real C++ output, re-typeset only. Do not edit by hand. -/", "",
              "namespace GL2.TauChk.Data", "open GL2.TauChk", ""]
        if (p, nu) == (2, 1):
            base = "tauP2"
            L.append(f"/-- The bulk τ-table of {what}: `tauP2` (its dump `tau_p2_r1.txt` is `tau_p2.txt` line for line). -/")
            L.append(f"abbrev tau{tag} : TauData := tauP2")
            L.append("")
        else:
            base = f"tau{tag}"
            L += tau_lean(base, tau, what)
        L.append(f"/-- The `τ_∂`-table of {what}, as dumped (`taubd{suf}.txt`). -/")
        L.append(f"def tauBd{tag} : TauBdData where")
        L.append(f"  base := {base}")
        L.append("  bdry := [" + ", ".join(f"({q}, {s})" for q, s in bd["bdry"]) + "]")
        L.append("  slots := [")
        L.append(slots_lean(bd["slots"]))
        L.append("  ]")
        L.append(f"  regime := \"{bd['regime']}\"")
        L.append(f"  fnv := {bd['fnv']}")
        L.append("")
        L.append(f"/-- The pair homotopy η of {what}, as dumped (`eta{suf}.txt`). -/")
        L.append(f"def eta{tag} : EtaData where")
        L.append(f"  bd := tauBd{tag}")
        L.append("  slots := [")
        L.append(slots_lean(eta["slots"]))
        L.append("  ]")
        L.append(f"  regime := \"{eta['regime']}\"")
        L.append(f"  fnv := {eta['fnv']}")
        L.append("")
        sfx = f"_p{p}_r{nu}"
        point = "n=2 ell=5 ext cone_action" if (p, nu) == (2, 1) else f"n=2 ell={ell} ext tau"
        for obj, nm in ((f"tau_table_fnv64{sfx}", "goldenTau"), (f"tau_bd_table_fnv64{sfx}", "goldenTauBd"),
                        (f"eta_table_fnv64{sfx}", "goldenEta")):
            g = golden(inc, point, obj)
            if g is not None:
                L.append(f"/-- The golden object `{obj}` (`{point}`, tier B) of")
                L.append("`Orbifold_Cohomology_of_GL2/Golden/GoldenReference.inc`, copied verbatim. -/")
                L.append(f"def {nm}{tag} : ℕ := {g}")
            else:
                assert nm == "goldenEta", f"golden {obj} at {point} not found"
                pin = hash_scalar64(eta["fnv"])
                L.append(f"/-- **Not a golden object** (no `{obj}` exists): `HashScalar64` of the dump's own printed")
                L.append("FNV, pinned (option (a) of `L17-4_report.md` §7, MINOR-4). -/")
                L.append(f"def pinnedEta{tag} : ℕ := 0x{pin:016x}")
            L.append("")
        L.append("end GL2.TauChk.Data")
        out = os.path.join(outdir, f"{tag}.lean")
        with open(out, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"TAUBD_ETA_TO_LEAN p={p} nu={nu} tau_slots={len(tau['slots'])} taubd_slots={len(bd['slots'])} "
              f"eta_slots={len(eta['slots'])} out={out}")
    emit_grid(root, [pn for pn in pairs if pn != (2, 1)])


def emit_grid(root, pairs):
    """GL2/Checkers/PairGrid/<tag>.lean (one import chain, so a plain `lake build` runs one kernel check
    at a time) and the aggregator GL2/Checkers/PairGrid.lean, for every (p, nu) != (2, 1)."""
    gdir = os.path.join(root, "GL2", "Checkers", "PairGrid")
    os.makedirs(gdir, exist_ok=True)
    tags = []
    prev = None
    for p, nu in pairs:
        tag = f"P{p}R{nu}"
        L = [f"import GL2.Checkers.PairData.{tag}", "import GL2.Checkers.TauSplit"]
        if prev:
            L.append(f"import GL2.Checkers.PairGrid.{prev} -- serialisation only (one kernel check at a time)")
        L += ["", f"/-! GENERATED by lean/tools/taubd_eta_to_lean.py. `(p, ν) = ({p}, {nu})`: τ (`obl:tau`), `τ_∂`",
              "(`obl:taubdry`) and η (`obl:eta`) on the real C++ tables by `decide +kernel`; τ and `τ_∂` tied to their",
              "golden objects (slot-18 R1, `ext tau` points), η to the dump's pinned hash (no golden η object).",
              f"`tauCheck` runs in {GRID_K + 1} declarations (`tauCheck_of_mod`, `s.i % {GRID_K}`): one declaration",
              "on the `(3, 2)` table peaks at ≈ 23 GB, above the 20 GB `lake` budget. -/", "",
              "namespace GL2.TauChk.Data", "open GL2.TauChk", "",
              f"theorem tau{tag}_global : tauGlobalCheck tau{tag} = true := by decide +kernel"]
        for r in range(GRID_K):
            L.append(f"theorem tau{tag}_part{r} :")
            L.append(f"    tauSlotsCheck tau{tag} (tau{tag}.slots.filter fun s => s.i % {GRID_K} = {r}) = true := by")
            L.append("  decide +kernel")
        rs = " | ".join(["rfl"] * GRID_K)
        L += [f"theorem tau{tag}_check : tauCheck tau{tag} = true :=",
              f"  tauCheck_of_mod _ {GRID_K} (by decide) tau{tag}_global fun r hr => by",
              f"    rcases (by omega : {' ∨ '.join(f'r = {r}' for r in range(GRID_K))}) with {rs}",
              f"    exacts [{', '.join(f'tau{tag}_part{r}' for r in range(GRID_K))}]",
              f"theorem tau{tag}_fnv_golden : hashScalar64 (tableFnv tau{tag}) = goldenTau{tag} := by decide +kernel",
              f"theorem tauBd{tag}_check : taubdCheck tauBd{tag} = true := by decide +kernel",
              f"theorem tauBd{tag}_fnv_golden : hashScalar64 (bdTableFnv tauBd{tag}) = goldenTauBd{tag} := by",
              "  decide +kernel",
              f"theorem eta{tag}_core : etaCoreCheck eta{tag} = true := by decide +kernel",
              f"theorem eta{tag}_check : etaCheck eta{tag} = true := etaCheck_of tau{tag}_check tauBd{tag}_check eta{tag}_core",
              f"theorem eta{tag}_fnv_pinned : hashScalar64 (etaTableFnv eta{tag}) = pinnedEta{tag} := by decide +kernel",
              "", "end GL2.TauChk.Data"]
        with open(os.path.join(gdir, f"{tag}.lean"), "w") as f:
            f.write("\n".join(L) + "\n")
        tags.append(tag)
        prev = tag
    L = [f"import GL2.Checkers.PairGrid.{prev}", "",
         "/-! GENERATED by lean/tools/taubd_eta_to_lean.py: the `(p, ν) ≠ (2, 1)` grid of [C] `rmk:iotatausym`",
         "(the slot-18 R1 `ext tau` objects). -/", "", "namespace GL2.TauChk.Data", "open GL2.TauChk", "",
         "/-- **τ, `τ_∂` and η are certified at every `(p, ν)` of the grid** (`obl:tau` (i)–(v), (vii);",
         "`obl:taubdry` (i)–(iv); `obl:eta` (i)–(iv)), and the τ / `τ_∂` tables checked are the golden ones. -/",
         "theorem pair_grid_specs :"]
    parts = []
    for t in tags:
        parts.append(f"    (EtaSpec eta{t} ∧ hashScalar64 (tableFnv tau{t}) = goldenTau{t} ∧\n"
                     f"      hashScalar64 (bdTableFnv tauBd{t}) = goldenTauBd{t})")
    L.append(" ∧\n".join(parts) + " :=")
    proofs = [f"⟨etaCheck_sound _ eta{t}_check, tau{t}_fnv_golden, tauBd{t}_fnv_golden⟩" for t in tags]
    L.append("  ⟨" + ",\n   ".join(proofs) + "⟩")
    L += ["", "end GL2.TauChk.Data"]
    with open(os.path.join(root, "GL2", "Checkers", "PairGrid.lean"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"TAUBD_ETA_TO_LEAN grid={','.join(tags)}")


if __name__ == "__main__":
    main()
