#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
"""add_spdx.py -- put SPDX headers on the repository's first-party source files (slot-21 WP-LIC).

  add_spdx.py            dry run: list what would change (default)
  add_spdx.py --apply    write the headers (idempotent: a file that already carries an SPDX-License-Identifier
                         in its first 20 or last 6 lines is left alone)
  add_spdx.py --check    verify: every candidate file has a header or is listed in
                         docs/licence/pending_tex_headers.txt; a pending entry that already has a header
                         (stale) or that is no longer a candidate fails too.  Exit 0 = clean, 1 = findings.

Candidates are the tracked files (git ls-files) of the header-able kinds below, minus the exempt paths.
Licence by kind: .tex / .sty / .bib -> CC-BY-4.0; code (C/C++/CUDA, cmake, shell, python, js, css, sbatch)
-> LGPL-3.0-only WITH LicenseRef-CUDA-exception.  Everything else (data, markdown, json, tsv, console
scripts and their frozen transcripts, generated includes) and lean/ are covered by REUSE.toml, not by headers.

Exempt (never touched, never required): lean/** (census.tsv hashes the Lean sources; Apache-2.0 by REUSE.toml;
lean/Vendor/** must stay byte-identical), notebook/vendor/**, (docs/literature/ was third party: removed from HEAD in slot 22 WP-LIC2),
.claude/**, generated files (docs/latex/cli_manual/*.tex, docs/latex/phenomena_gallery/costs.tex, *.inc).

Placement: after a shebang (and a Python coding line); for shell scripts that print their own header with
`sed -n 'a,bp' "$0"` the header goes at the END of the file instead, so those line ranges stay valid.
With --apply, files with uncommitted changes in the working tree are skipped and reported (another agent's
work in progress); a later run picks them up.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PENDING = "docs/licence/pending_tex_headers.txt"

LGPL = "LGPL-3.0-only WITH LicenseRef-CUDA-exception"
CCBY = "CC-BY-4.0"
COPYRIGHT = "2024-2026 Fabian Januszewski"

SLASH = {".cpp", ".h", ".hpp", ".cu", ".cuh", ".js", ".mjs"}
HASH = {".cmake", ".py", ".sh", ".sbatch"}
PERCENT = {".tex", ".sty", ".bib"}
CSS = {".css"}

EXEMPT_PREFIX = ("lean/", "notebook/vendor/", ".claude/",
                 "docs/latex/cli_manual/", "docs/latex/phenomena_gallery/costs.tex")
SELF_SED = re.compile(rb'sed\s+-n\s+[^\n]*("\$0"|"\$\{BASH_SOURCE)')
# REUSE-IgnoreStart (the tag literals below are code, not this file's licence)
SPDX = b"SPDX-License-Identifier:"
# REUSE-IgnoreEnd


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def dirty():
    out = subprocess.run(["git", "status", "--porcelain", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    names = set()
    for ent in out.decode().split("\0"):
        if len(ent) > 3:
            names.add(ent[3:])
    return names


def kind(path):
    """(comment style, licence) for a header-able path, else None."""
    if path.startswith(EXEMPT_PREFIX) or path.endswith(".inc"):
        return None
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1]
    if base == "CMakeLists.txt":
        return ("#", LGPL)
    if ext in SLASH:
        return ("//", LGPL)
    if ext in HASH:
        return ("#", LGPL)
    if ext in CSS:
        return ("/*", LGPL)
    if ext in PERCENT:
        return ("%", CCBY)
    return None


def has_header(data):
    lines = data.split(b"\n")
    head = b"\n".join(lines[:20])
    tail = b"\n".join(lines[-7:])
    return SPDX in head or SPDX in tail


# REUSE-IgnoreStart
def header_lines(style, lic):
    a = "SPDX-FileCopyrightText: " + COPYRIGHT
    b = "SPDX-License-Identifier: " + lic
    # REUSE-IgnoreEnd
    if style == "/*":
        return ["/* " + a, "   " + b + " */"]
    return [style + " " + a, style + " " + b]


def insert(path, data, style, lic):
    if data.startswith(b"\xef\xbb\xbf"):          # keep a UTF-8 BOM at offset 0 (nvcc rejects one mid-file)
        return b"\xef\xbb\xbf" + insert(path, data[3:], style, lic)
    hdr = [h.encode() for h in header_lines(style, lic)]
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines = lines[:-1]
    if SELF_SED.search(data) and style == "#":
        out = lines + [b""] + hdr
    else:
        i = 0
        if lines and lines[0].startswith(b"#!"):
            i = 1
        if path.endswith(".py") and len(lines) > i and re.match(rb"#.*coding[:=]", lines[i]):
            i += 1
        block = list(hdr)
        if i < len(lines) and lines[i].strip() != b"":
            block.append(b"")
        out = lines[:i] + block + lines[i:]
    return b"\n".join(out) + b"\n"


def read_pending():
    p = os.path.join(ROOT, PENDING)
    if not os.path.exists(p):
        return []
    res = []
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            res.append(ln)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--write-pending", action="store_true",
                    help="with --apply: also append the skipped dirty files to the pending list")
    args = ap.parse_args()

    cands = []
    for p in tracked():
        k = kind(p)
        if k and os.path.isfile(os.path.join(ROOT, p)):
            cands.append((p, k))
    pending = read_pending()
    pend = set(pending)

    if args.check:
        missing, stale, unknown = [], [], []
        by_path = dict(cands)
        for p, k in cands:
            data = open(os.path.join(ROOT, p), "rb").read()
            if len(data) == 0:
                continue
            h = has_header(data)
            if p in pend:
                if h:
                    stale.append(p)
            elif not h:
                missing.append(p)
        for p in pending:
            if p not in by_path:
                unknown.append(p)
        for p in missing:
            print("MISSING_HEADER " + p)
        for p in stale:
            print("STALE_PENDING (already has a header; remove from %s) %s" % (PENDING, p))
        for p in unknown:
            print("UNKNOWN_PENDING (not a candidate file) " + p)
        print("ADD_SPDX_CHECK candidates=%d pending=%d missing=%d stale=%d unknown=%d verdict=%s" % (
            len(cands), len(pending), len(missing), len(stale), len(unknown),
            "OK" if not (missing or stale or unknown) else "FAIL"))
        return 0 if not (missing or stale or unknown) else 1

    dirt = dirty() if args.apply else set()
    todo, skipped_dirty, done = [], [], 0
    for p, (style, lic) in cands:
        if p in pend:
            continue
        full = os.path.join(ROOT, p)
        data = open(full, "rb").read()
        if len(data) == 0 or has_header(data):
            continue
        if p in dirt:
            skipped_dirty.append(p)
            continue
        todo.append(p)
        if args.apply:
            new = insert(p, data, style, lic)
            with open(full, "wb") as f:
                f.write(new)
            done += 1
    for p in todo:
        print(("ADDED " if args.apply else "WOULD_ADD ") + p)
    for p in skipped_dirty:
        print("SKIPPED_DIRTY " + p)
    print("ADD_SPDX %s candidates=%d pending=%d %s=%d skipped_dirty=%d" % (
        "apply" if args.apply else "dry-run", len(cands), len(pending),
        "added" if args.apply else "would_add", len(todo), len(skipped_dirty)))
    if args.apply and args.write_pending and skipped_dirty:
        with open(os.path.join(ROOT, PENDING), "a", encoding="utf-8") as f:
            for p in skipped_dirty:
                f.write(p + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
