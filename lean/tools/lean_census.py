#!/usr/bin/env python3
"""lean_census.py -- the Lean census gate of the formal ground truth [GT].

Reads  <root>/blueprint/src/gl2_groundtruth.tex + gt_part_*.tex  (the statements, their \\lean{} names, \\leanok marks)
and    <root>/GL2/**/*.lean, <root>/GL2.lean                      (the sources),
asks Lean (through `lake env lean`) for `#print axioms` of every \\lean{} name, runs the per-constant scan
<root>/tools/CensusScan.lean over EVERY constant defined in a GL2 module, and prints one line

  LEAN_CENSUS decls=N proved=P proved_mod_lit=M stated=S planned=Q sorries=K global_axioms=A unresolved=U stale=T
              lit_hyp=H native=V gl2_consts=C escapes=E modlit_bad=B hyp_bad=Y verdict=OK|FAIL

(one line; the fields up to `stale=` are those of the L16-0 line, in the same order; `lit_hyp` .. `modlit_bad` were
added by the L16-5 repairs, `hyp_bad` by the F3 repairs, 2026-10-02).  READ THE TEXT of that line, never only the exit status (the slot-11 lesson).
Exit 0 iff verdict=OK.

Definitions
  decls          \\lean{} names found in [GT]
  proved         statement \\leanok and proof \\leanok  (axioms must be a subset of the three standard ones)
  proved_mod_lit as proved, and the statement carries \\gtmodlit{...}: the Lean declarations are sorry-free and
                 the literature enters only as named hypotheses in their types (L16-2); the hypotheses are
                 DECLARED by \\gtlithyp{name} inside \\gtmodlit and checked (modlit_bad)
  stated         statement \\leanok, proof not marked, not a `hypothesis` environment   (sorryAx allowed)
  lit_hyp        a `hypothesis` environment (the literature register) with \\leanok; must be sorry-free
                 (L16-5: these used to count as `stated`, which gave `sorries <= stated` slack)
  planned        no \\leanok
  sorries        occurrences of the tokens `sorry`, `admit`, `stop` in the sources, comments and strings stripped
  global_axioms  `axiom` declarations in the sources, plus any non-standard axiom printed for a linked name, plus
                 every unlinked GL2 constant depending on a non-standard axiom other than sorryAx / the native ones.
                 A native axiom on a LINKED name is non-standard here, whatever the row (chk: / thm: / hypothesis):
                 `proved` means kernel-checked (L17-4 repair, 2026-10-02; selftest controls linked_native_*)
  unresolved     \\lean{} names Lean does not resolve; for a name claimed proved / lit_hyp, also a sorryAx / extra axiom
  stale          rows of <root>/census.tsv that disagree with the table computed from [GT] (either direction)
  lit_hyp        number of lit_hyp rows
  native         GL2 constants whose own proof uses `native_decide` (mentions Lean.ofReduceBool / ofReduceNat /
                 trustCompiler, or -- Lean v4.33 -- the auxiliary axiom `<decl>._native.native_decide.ax_…`)
  gl2_consts     constants defined in GL2 modules that the per-constant scan saw (auxiliaries included)
  escapes        (a) a GL2 constant depending on sorryAx that is neither the decl of a stated / planned row nor an
                 auxiliary of one (`<decl>.…`) -- an undeclared sorry / admit, linked or not;
                 (b) a GL2 constant depending on a native axiom, in ANY module (F3 finding 8, owner direction
                 2026-10-02: the GL2.Algorithms.* exemption for unlinked native constants is gone);
                 (c) [source scan, also under --no-lean, so the committed gate sees it] a `set_option` in the GL2
                 sources, or an option / extra Lean argument in lakefile.toml, outside OPTION_ALLOW (F3 finding 4:
                 `debug.skipKernelTC` makes a false `decide +kernel` theorem axiom-free); a `native_decide`,
                 `implemented_by`, `extern`, `ofReduceBool` / `ofReduceNat` / `trustCompiler` token in the sources
  modlit_bad     violations of the \\gtmodlit check: a row declaring no \\gtlithyp; a declared name matching no Prop
                 binder of the row's declarations; a Prop hypothesis of type False; an UNDECLARED Prop hypothesis
                 that is closed (mentions no earlier binder), headed by a GL2-defined constant outside
                 GL2.Literature, or headed by a local variable
  hyp_bad        (F3 finding 3) the hypotheses of a PROVED row, which declares none: a Prop binder of type False; a
                 closed Prop binder (mentions no earlier binder); a Prop binder headed by a GL2.Literature constant
                 (that row is proved_mod_lit and must say so).  Live: from the per-constant HYP scan, which now runs
                 on every proved and proved_mod_lit declaration.  Under --no-lean (the committed gate): a source scan
                 of the linked proved / proved_mod_lit / lit_hyp declarations' signatures for a binder of type False.
                 A hypothesis that is a real mathematical premise (e.g. `hrank`) cannot be told from a definition by
                 any scan: [GT] declares it with \\gtmodlit{... \\gtlithyp{hrank}} (the row is then proved_mod_lit
                 and modlit_bad checks it).
FAIL also when sorries > stated (a sorry nobody declared).

census.tsv columns: label, decl, lean_status, axioms, lit_hyps (for proved_mod_lit rows the `binder:Head` list of
the literature hypotheses found in the declaration's type, `-` if none / other rows).

Usage: lean_census.py [--root DIR] [--emit] [--no-lean]
  --emit     rewrite <root>/census.tsv from the computed table (the committed census), then check it
  --no-lean  skip Lean; statuses and names only (axioms column '-'); never used by the gate itself
"""
import os, re, subprocess, sys, tempfile

STD = {"propext", "Classical.choice", "Quot.sound"}
NATIVE = {"Lean.ofReduceBool", "Lean.ofReduceNat", "Lean.trustCompiler"}


def is_native(a):
    """A native-evaluation axiom: the classical ones, or the auxiliary axiom `<decl>._native.native_decide.ax_…`
    that `native_decide` declares since Lean 4.2x (found on v4.33 by the L16-5 repair selftest)."""
    return a in NATIVE or "._native." in a
# F3 finding 4: options that cannot affect what the kernel checks.  Anything else (debug.skipKernelTC, ...) is an escape.
OPTION_ALLOW = {"maxRecDepth", "maxHeartbeats", "synthInstance.maxHeartbeats", "autoImplicit", "relaxedAutoImplicit"}
OPTION_ALLOW_PREFIX = ("linter.",)
# S21 L21-V: the vendored FLT lib needs FLT's own elaborator option (Meta-level defeq transparency, not read by the
# kernel; every vendored module is kernel-replayed by lean_replay.sh).  Allowed in that lib's leanOptions only, never
# in [leanOptions], GL2's lib or a GL2 source.
VENDOR_OPTION_ALLOW = {"AnthropicFLT": {"backward.isDefEq.respectTransparency.types"}}
LIB_OPTS = re.compile(r"^leanOptions\s*=\s*\{([^{}]*)\}$")
SRC_BANNED = re.compile(r"(?<![\w.'])(native_decide|implemented_by|extern|ofReduceBool|ofReduceNat|trustCompiler)(?![\w'])")
LAKE_ARGS = {"moreLeanArgs", "weakLeanArgs", "moreServerArgs", "moreServerOptions", "moreGlobalServerArgs",
             "moreLinkArgs", "weakLinkArgs"}


def option_ok(name):
    return name in OPTION_ALLOW or name.startswith(OPTION_ALLOW_PREFIX)


def source_escapes(root, srcs):
    """F3 findings 4, 8 (lake-free): -> list of (where, what)."""
    out = []
    for f in srcs:
        s = strip_lean(open(f, encoding="utf-8").read())
        rel = os.path.relpath(f, root)
        for m in re.finditer(r"(?<![\w.'])set_option\s+([\w.]+)", s):
            if not option_ok(m.group(1)):
                out.append((rel, "set_option:" + m.group(1)))
        for m in SRC_BANNED.finditer(s):
            out.append((rel, "token:" + m.group(1)))
    lf = os.path.join(root, "lakefile.toml")
    if os.path.exists(lf):
        section, lib = "", ""
        for ln in open(lf, encoding="utf-8").read().splitlines():
            t = ln.split("#", 1)[0].strip()
            if not t:
                continue
            if t.startswith("["):
                section = t.strip("[] ")
                lib = lib if section == "lean_lib.leanOptions" else ""
                continue
            key = t.split("=", 1)[0].strip().strip('"')
            if section == "lean_lib" and key == "name":
                lib = t.split("=", 1)[1].strip().strip('"')
            if section == "leanOptions" and not option_ok(key):
                out.append(("lakefile.toml", "leanOption:" + key))
            if section.endswith(".leanOptions") and not (option_ok(key) or key in VENDOR_OPTION_ALLOW.get(lib, ())):
                out.append(("lakefile.toml", "leanOption:%s:%s" % (lib or section, key)))
            m = LIB_OPTS.match(t) if section == "lean_lib" and key == "leanOptions" else None
            if m:  # S21 L21-V: a vendored lib's one-line inline table, each key checked
                for k in (x.split("=", 1)[0].strip() for x in m.group(1).split(",")):
                    if not (option_ok(k) or k in VENDOR_OPTION_ALLOW.get(lib, ())):
                        out.append(("lakefile.toml", "leanOption:%s:%s" % (lib, k)))
            elif key in LAKE_ARGS or (key == "leanOptions" and section != "leanOptions"):
                out.append(("lakefile.toml", "lake_arg:" + key))
    if os.path.exists(os.path.join(root, "lakefile.lean")):
        out.append(("lakefile.lean", "lakefile.lean_present(only_lakefile.toml_is_scanned)"))
    return out


FALSE_BINDER = re.compile(r"[(\[{\u2983]\s*[^:()\[\]{}]*:\s*False\s*[)\]}\u2984]")


def false_binder_decls(srcs):
    """Lake-free (the committed gate): -> set of declaration names (as written after `theorem` / `lemma` / `def`)
    whose signature has a binder of type False."""
    out = set()
    for f in srcs:
        s = strip_lean(open(f, encoding="utf-8").read())
        for m in re.finditer(r"(?<![\w.'])(?:theorem|lemma|def|abbrev|instance)\s+(\S+)", s):
            sig = s[m.end():m.end() + 4000]
            end = sig.find(":=")
            sig = sig[:end] if end >= 0 else sig
            if FALSE_BINDER.search(sig):
                out.add(m.group(1))
    return out


ENVS = r"(?:theorem|lemma|proposition|corollary|definition|remark|conjecture|example|axiom|hypothesis)"
HEADER = "label\tdecl\tlean_status\taxioms\tlit_hyps"


def strip_lean(src):
    """Remove comments and string literals from Lean source (enough for token counting)."""
    out, i, n, depth = [], 0, len(src), 0
    while i < n:
        if depth == 0 and src.startswith("--", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/-", i):
            depth += 1; i += 2
        elif depth > 0 and src.startswith("-/", i):
            depth -= 1; i += 2
        elif depth > 0:
            i += 1
        elif src[i] == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
            i += 1
        else:
            out.append(src[i]); i += 1
    return "".join(out)


def lean_sources(root):
    files = []
    top = os.path.join(root, "GL2.lean")
    if os.path.exists(top):
        files.append(top)
    for d, _, fs in os.walk(os.path.join(root, "GL2")):
        files += [os.path.join(d, f) for f in sorted(fs) if f.endswith(".lean")]
    return sorted(files)


SRC_HASH_PREFIX = "# sources_sha256="


def sources_sha256(root):
    """sha256 over the GL2 Lean sources (GL2.lean + GL2/**.lean, sorted, as lean_sources lists them): for each file
    `<path relative to root>\\0<bytes>\\0`.  Written by --emit as the second line of census.tsv (L18-5, F3 13)."""
    import hashlib
    h = hashlib.sha256()
    for f in lean_sources(root):
        h.update(os.path.relpath(f, root).replace(os.sep, "/").encode() + b"\0")
        h.update(open(f, "rb").read() + b"\0")
    return h.hexdigest()


def read_sources_hash(path):
    """-> the committed hex value, or None when the line is absent."""
    if not os.path.exists(path):
        return None
    for ln in open(path, encoding="utf-8").read().splitlines():
        if ln.startswith(SRC_HASH_PREFIX):
            return ln[len(SRC_HASH_PREFIX):].strip()
    return None


def parse_gt(tex):
    """-> list of dict(label, decl, status, lithyps)"""
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    rows = []
    pat = re.compile(r"\\begin\{(" + ENVS + r")\*?\}(.*?)\\end\{\1\*?\}(\s*\\begin\{proof\}(.*?)\\end\{proof\})?", re.S)
    for m in pat.finditer(tex):
        env, body, proof = m.group(1), m.group(2), m.group(4) or ""
        names = re.findall(r"\\lean\{([^}]*)\}", body)
        decls = [x.strip() for s in names for x in s.split(",") if x.strip()]
        if not decls:
            continue
        lab = re.search(r"\\label\{([^}]*)\}", body)
        label = lab.group(1) if lab else "?"
        stmt_ok = "\\leanok" in body
        proof_ok = "\\leanok" in proof
        mathlib = "\\mathlibok" in body
        modlit = "\\gtmodlit" in body
        if mathlib:
            status = "lit_mathlib"
        elif env == "hypothesis":
            status = "lit_hyp" if stmt_ok else "planned"
        elif stmt_ok and proof_ok:
            status = "proved_mod_lit" if modlit else "proved"
        else:
            status = "stated" if stmt_ok else "planned"
        lithyps = [x.strip() for x in re.findall(r"\\gtlithyp\{([^}]*)\}", body)] if modlit else []
        for d in decls:
            rows.append(dict(label=label, decl=d, status=status, lithyps=lithyps))
    return rows


def run_lean(root, decls):
    """-> dict decl -> (resolved: bool, axioms: list[str])"""
    res = {}
    if not decls:
        return res
    lines = ["import GL2", ""]
    for d in decls:
        lines += ['#eval IO.println "CENSUS_BEGIN %s"' % d, "#print axioms %s" % d]
    lines.append('#eval IO.println "CENSUS_END"')
    with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=root, prefix="_census_", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name
    try:
        p = subprocess.run(["lake", "env", "lean", path], cwd=root, capture_output=True, text=True)
    finally:
        os.unlink(path)
    out = p.stdout + p.stderr
    blocks = re.split(r"CENSUS_BEGIN (\S+)\n", out)
    # blocks: [pre, name1, text1, name2, text2, ...]
    seen = {blocks[i]: blocks[i + 1] for i in range(1, len(blocks) - 1, 2)}
    for d in decls:
        text = seen.get(d)
        if text is None or "error" in text or "unknown" in text:
            res[d] = (False, [])
            continue
        m = re.search(r"depends on axioms: \[(.*?)\]", text, re.S)
        if m:
            res[d] = (True, sorted(a.strip() for a in m.group(1).replace("\n", " ").split(",") if a.strip()))
        elif "does not depend on any axioms" in text:
            res[d] = (True, [])
        else:
            res[d] = (False, [])
    return res


def run_scan(root, hyp_decls):
    """The per-constant scan (tools/CensusScan.lean).
    -> (consts: dict name -> (module, axioms list, direct list), axdecls: list, hyps: dict decl -> list, ok: bool)"""
    scan = os.path.join(root, "tools", "CensusScan.lean")
    env = dict(os.environ, LEAN_CENSUS_HYPS=",".join(hyp_decls))
    p = subprocess.run(["lake", "env", "lean", scan], cwd=root, capture_output=True, text=True, env=env)
    consts, axdecls, hyps, ended = {}, [], {}, False
    for ln in p.stdout.splitlines():
        c = ln.split("\t")
        if c[0] == "CONST" and len(c) == 5:
            sp = lambda s: [] if s == "none" else s.split(",")
            consts[c[1]] = (c[2], sp(c[3]), sp(c[4]))
        elif c[0] == "AXDECL" and len(c) >= 2:
            axdecls.append(c[1])
        elif c[0] == "HYP" and len(c) == 7:
            hyps.setdefault(c[1], []).append(dict(binder=c[2], head=c[3], closed=c[4] == "1",
                                                  gl2=c[5] == "1", lit=c[6] == "1"))
        elif c[0] == "SCAN_END":
            ended = True
    if not ended or p.returncode != 0:
        sys.stderr.write((p.stdout + p.stderr)[-4000:])
    return consts, axdecls, hyps, ended and p.returncode == 0


def read_census(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for ln in open(path, encoding="utf-8").read().splitlines():
        if not ln or ln.startswith("#") or ln.startswith("label\t"):
            continue
        c = ln.split("\t")
        if len(c) >= 4:
            rows[(c[0], c[1])] = (c[2], c[3], c[4] if len(c) >= 5 else "<absent>")
    return rows


def short(head):
    return head.split(".")[-1]


def modlit_check(rows, hyps):
    """-> (bad list of (label, decl, reason), lit_hyps dict decl -> str)"""
    bad, lit = [], {}
    by_label = {}
    for r in rows:
        if r["status"] == "proved_mod_lit":
            by_label.setdefault(r["label"], []).append(r)
    for label, rs in sorted(by_label.items()):
        declared = set(rs[0]["lithyps"])
        if not declared:
            bad.append((label, "-", "no_declared_hypothesis(\\gtlithyp)"))
        matched = set()
        for r in rs:
            found = []
            for h in hyps.get(r["decl"], []):
                hd = h["head"]
                if hd == "UNRESOLVED":
                    bad.append((label, r["decl"], "unresolved")); continue
                names = {h["binder"]} | ({short(hd)} if hd not in ("-", "?fvar") else set())
                hit = names & declared
                matched |= hit
                if hd == "False":
                    bad.append((label, r["decl"], "false_hypothesis:%s" % h["binder"]))
                elif hit or h["lit"]:
                    found.append("%s:%s" % (h["binder"], short(hd) if hd not in ("-", "?fvar") else hd))
                elif h["closed"]:
                    bad.append((label, r["decl"], "undeclared_closed_hypothesis:%s:%s" % (h["binder"], hd)))
                elif h["gl2"]:
                    bad.append((label, r["decl"], "undeclared_project_hypothesis:%s:%s" % (h["binder"], hd)))
                elif hd == "?fvar":
                    bad.append((label, r["decl"], "undeclared_variable_hypothesis:%s" % h["binder"]))
            lit[r["decl"]] = ",".join(found) if found else "-"
        for d in sorted(declared - matched):
            bad.append((label, "-", "declared_but_absent:%s" % d))
    return bad, lit


def hyp_check(rows, hyps):
    """F3 finding 3: the hypotheses of a proved row (live; from the HYP scan). -> list of (label, decl, reason)."""
    bad = []
    for r in rows:
        if r["status"] != "proved":
            continue
        for h in hyps.get(r["decl"], []):
            hd = h["head"]
            if hd == "False":
                bad.append((r["label"], r["decl"], "false_hypothesis:%s" % h["binder"]))
            elif h["closed"]:
                bad.append((r["label"], r["decl"], "closed_hypothesis_on_proved_row:%s:%s" % (h["binder"], hd)))
            elif h["lit"]:
                bad.append((r["label"], r["decl"], "literature_hypothesis_on_proved_row:%s:%s" % (h["binder"], hd)))
    return bad


def main(argv):
    root, emit, nolean = None, False, False
    it = iter(argv)
    for a in it:
        if a == "--root": root = next(it)
        elif a == "--emit": emit = True
        elif a == "--no-lean": nolean = True
        else:
            print("usage: lean_census.py [--root DIR] [--emit] [--no-lean]", file=sys.stderr); return 2
    root = os.path.abspath(root or os.path.join(os.path.dirname(__file__), ".."))
    texp = os.path.join(root, "blueprint", "src", "gl2_groundtruth.tex")
    if not os.path.exists(texp):
        print("LEAN_CENSUS error: no %s" % texp, file=sys.stderr); return 2
    import glob
    texts = [texp] + sorted(glob.glob(os.path.join(root, "blueprint", "src", "gt_part_*.tex")))
    gt = parse_gt("\n".join(open(t, encoding="utf-8").read() for t in texts))

    srcs = lean_sources(root)
    sorries = axiom_decls = 0
    for f in srcs:
        s = strip_lean(open(f, encoding="utf-8").read())
        sorries += len(re.findall(r"(?<![\w.'])(?:sorry|admit|stop)(?![\w'])", s))
        axiom_decls += len(re.findall(r"(?m)^\s*(?:private\s+|protected\s+|unsafe\s+|noncomputable\s+)*axiom\s", s))

    unresolved = 0
    extra_axioms = 0
    table = []
    ax = run_lean(root, [r["decl"] for r in gt]) if (gt and not nolean) else {}
    hyp_decls = sorted({r["decl"] for r in gt if r["status"] in ("proved_mod_lit", "proved")})
    consts, axdecls, hyps, scan_ok = ({}, [], {}, True) if nolean else run_scan(root, hyp_decls)
    if not scan_ok:
        unresolved += 1
        print("SCAN_FAILED tools/CensusScan.lean did not complete")
    bad, lit = ([], {}) if nolean else modlit_check(gt, hyps)
    if nolean:
        fb = false_binder_decls(srcs)
        hbad = [(r["label"], r["decl"], "false_hypothesis(source)") for r in gt
                if r["status"] in ("proved", "proved_mod_lit", "lit_hyp")
                and any(r["decl"] == d or r["decl"].endswith("." + d) for d in fb)]
    else:
        hbad = hyp_check(gt, hyps)
    for r in gt:
        if nolean:
            axs = "-"
        else:
            ok, al = ax.get(r["decl"], (False, []))
            if not ok:
                unresolved += 1; axs = "UNRESOLVED"
            else:
                badax = [a for a in al if a not in STD and a != "sorryAx"]
                extra_axioms += len(badax)
                if "sorryAx" in al and r["status"] not in ("stated", "planned"):
                    unresolved += 1  # claimed proved / a literature hypothesis, depends on sorry
                    print("UNRESOLVED_SORRY %s %s (%s)" % (r["label"], r["decl"], r["status"]))
                axs = ",".join(al) if al else "none"
        lh = "-" if nolean or r["status"] != "proved_mod_lit" else lit.get(r["decl"], "-")
        table.append((r["label"], r["decl"], r["status"], axs, lh))

    # The per-constant checks (finding 2 (c)).
    linked = {r["decl"] for r in gt}
    sorry_ok = {r["decl"] for r in gt if r["status"] in ("stated", "planned")}
    escapes = native = unlinked_bad = 0
    for n, (mod, al, direct) in sorted(consts.items()):
        if any(is_native(a) for a in direct):
            native += 1
        if "sorryAx" in al and n not in sorry_ok and not any(n.startswith(d + ".") for d in sorry_ok):
            if n not in linked:  # a linked non-stated name is already counted as unresolved
                escapes += 1
                print("ESCAPE sorry %s (%s; %s)" % (n, mod, "direct" if "sorryAx" in direct else "inherited"))
        if any(is_native(a) for a in al):  # F3 finding 8: no module exemption
            escapes += 1
            print("ESCAPE native %s (%s)" % (n, mod))
        if n not in linked and any(a not in STD and a != "sorryAx" and not is_native(a) for a in al):
            unlinked_bad += 1
            print("NONSTD_AXIOM %s (%s): %s" % (n, mod, ",".join(al)))
    for where, what in source_escapes(root, srcs):
        escapes += 1
        print("ESCAPE source %s (%s)" % (what, where))
    for a in axdecls:
        print("AXIOM_DECL %s" % a)
    for b in hbad:
        print("HYP_BAD %s %s %s" % b)
    for b in bad:
        print("MODLIT_BAD %s %s %s" % b)

    cpath = os.path.join(root, "census.tsv")
    if emit:
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(HEADER + "\n")
            f.write(SRC_HASH_PREFIX + sources_sha256(root) + "\n")
            for t in sorted(table):
                f.write("\t".join(t) + "\n")
    committed = read_census(cpath)
    computed = {(t[0], t[1]): (t[2], t[3], t[4]) for t in table}
    stale = 0
    for k in sorted(set(committed) | set(computed)):
        if committed.get(k) != computed.get(k):
            stale += 1
            print("STALE %s %s: committed=%s computed=%s" % (k[0], k[1], committed.get(k), computed.get(k)))

    chash, nhash = read_sources_hash(cpath), sources_sha256(root)
    hstate = "MISSING" if chash is None else ("OK" if chash == nhash else "STALE")
    if hstate != "OK":
        print("SOURCES_HASH %s: committed=%s computed=%s" % (hstate, chash, nhash))
    cnt = lambda s: sum(1 for t in table if t[2] == s)
    stated = cnt("stated")
    gax = max(axiom_decls, len([a for a in axdecls if not is_native(a)])) + extra_axioms + unlinked_bad
    ok = (gax == 0 and unresolved == 0 and stale == 0 and sorries <= stated and escapes == 0 and not bad and not hbad
          and hstate == "OK")
    print("LEAN_CENSUS decls=%d proved=%d proved_mod_lit=%d stated=%d planned=%d sorries=%d global_axioms=%d "
          "unresolved=%d stale=%d lit_hyp=%d native=%d gl2_consts=%d escapes=%d modlit_bad=%d hyp_bad=%d sources_hash=%s verdict=%s"
          % (len(table), cnt("proved"), cnt("proved_mod_lit"), stated, cnt("planned"), sorries, gax, unresolved,
             stale, cnt("lit_hyp"), native, len(consts), escapes, len(bad), len(hbad), hstate, "OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
