#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# golden_ref_diff.py -- hecke-extensions slot 7, WP-1 (scope 0; audit [S7A] B-1).
#
# THE REFERENCE-LEVEL DIFF of two GoldenReference.inc files.  It parses the
# data lines
#
#     {"<point>", "<object>", '<tier>', <rows>, <cols>, 0x<16 hex>ull},
#
# ignores every comment line (`// ...`, the provenance header and the
# `// Entries: N` line included) and blank lines, keys every entry by
# (point, object) -- the key GoldenLookup / GoldenRegressionValidation use --
# and prints three sets of (point, object, tier):
#
#   added    key present only in AFTER;
#   changed  key present in both with a different tier, rows, cols or hash;
#   removed  key present only in BEFORE;
#
# each with its size split by tier and by OBJECT KIND (the object name with
# every `^<digits>` written `^q` and every `T_<digits>` written `T_p`).
#
# WHY IT EXISTS.  golden_explain.py reads only the GoldenDump texts and
# decompositions; it is BLIND to every raw-hash object (ext_ideals,
# ext_certify, sturm_bound, every PushGolden family, every E7 object).  This
# tool sees every data line.  The two are used side by side: this one proves
# WHICH keys moved, golden_explain.py explains HOW a changed text changed.
# golden_explain.py is not modified.
#
# THE GATE.  With `--expect FILE` it exits 0 iff the three observed sets are
# EQUAL to the expected ones (set equality, not inclusion).  FILE has one
# line per expected key, tab-separated:
#
#     +<TAB><point><TAB><object><TAB><tier>     an expected addition
#     ~<TAB><point><TAB><object><TAB><tier>     an expected change (AFTER's tier)
#     -<TAB><point><TAB><object><TAB><tier>     an expected removal (BEFORE's tier)
#
# `#` lines and blank lines are ignored.  A generating rule is expanded into
# this form by the package that states it (the report quotes the rule and
# the count); `--write-observed FILE` writes the observed sets in exactly
# this form, which is how an expectation is compared by `diff` as well.
#
# Usage:
#   golden_ref_diff.py BEFORE.inc AFTER.inc [--expect FILE]
#                      [--write-observed FILE] [--list] [--quiet]
#
#   --list   also print every key of the three sets (one line each).
#   --quiet  print only the summary lines.
#
# Exit codes:
#   0  parsed; with --expect, the observed sets equal the expected ones;
#   1  --expect given and the sets differ (the mismatches are printed);
#   2  usage error, unreadable file, a MALFORMED data line, a DUPLICATE
#      (point, object) key in either file, or a malformed/duplicate --expect
#      line.
#
# Stdlib only (the repo's "no third-party library" convention).
# ---------------------------------------------------------------------------

import re
import sys

USAGE = (
    "usage: golden_ref_diff.py BEFORE.inc AFTER.inc [--expect FILE]\n"
    "                          [--write-observed FILE] [--list] [--quiet]\n"
)

# A C string literal as WriteReference emits it: no escapes are ever written,
# but a backslash escape is accepted so the parser is total on valid C.
_STR = r'"((?:[^"\\]|\\.)*)"'
_DATA = re.compile(
    r"^\{" + _STR + r", " + _STR + r", '([AB])', (\d+), (\d+), (0x[0-9a-f]{16})ull\},$"
)


class ParseError(Exception):
    pass


def read_reference(path):
    """-> {(point, object): (tier, rows, cols, hash)}; raises ParseError."""
    out = {}
    try:
        fh = open(path, "r", encoding="utf-8", errors="surrogateescape")
    except OSError as e:
        raise ParseError("%s: cannot open: %s" % (path, e))
    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").rstrip("\r")
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            m = _DATA.match(line)
            if m is None:
                raise ParseError("%s:%d: malformed data line: %r" % (path, lineno, line[:200]))
            point, obj, tier, rows, cols, h = m.groups()
            key = (point, obj)
            if key in out:
                raise ParseError("%s:%d: duplicate key (point=%r, object=%r)" % (path, lineno, point, obj))
            out[key] = (tier, int(rows), int(cols), h)
    return out


def read_expect(path):
    """-> {'+': set, '~': set, '-': set} of (point, object, tier)."""
    exp = {"+": set(), "~": set(), "-": set()}
    seen = set()
    try:
        fh = open(path, "r", encoding="utf-8", errors="surrogateescape")
    except OSError as e:
        raise ParseError("%s: cannot open: %s" % (path, e))
    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4 or parts[0] not in exp or parts[3] not in ("A", "B"):
                raise ParseError("%s:%d: malformed expect line (want 'op<TAB>point<TAB>object<TAB>tier'): %r"
                                 % (path, lineno, line[:200]))
            key = (parts[1], parts[2])
            if key in seen:
                raise ParseError("%s:%d: duplicate expect key (point=%r, object=%r)" % (path, lineno, key[0], key[1]))
            seen.add(key)
            exp[parts[0]].add((parts[1], parts[2], parts[3]))
    return exp


_KIND_POW = re.compile(r"\^\d+")
_KIND_TP = re.compile(r"T_\d+")


def kind_of(obj):
    return _KIND_TP.sub("T_p", _KIND_POW.sub("^q", obj))


def diff(before, after):
    added = {(p, o, after[(p, o)][0]) for (p, o) in after if (p, o) not in before}
    removed = {(p, o, before[(p, o)][0]) for (p, o) in before if (p, o) not in after}
    changed = {(p, o, after[(p, o)][0]) for (p, o) in after if (p, o) in before and before[(p, o)] != after[(p, o)]}
    return {"+": added, "~": changed, "-": removed}


def tier_counts(s):
    a = sum(1 for k in s if k[2] == "A")
    return a, len(s) - a


def print_set(label, s, show_list):
    a, b = tier_counts(s)
    print("%-8s %6d  (tier A %d, tier B %d)" % (label, len(s), a, b))
    if not s:
        return
    kinds = {}
    for (_, o, t) in s:
        k = kinds.setdefault(kind_of(o), [0, 0])
        k[0 if t == "A" else 1] += 1
    for name in sorted(kinds):
        ka, kb = kinds[name]
        print("    %-40s %6d  (A %d, B %d)" % (name, ka + kb, ka, kb))
    if show_list:
        for (p, o, t) in sorted(s):
            print("    [%s] %s | %s" % (t, p, o))


def write_observed(path, sets):
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as fh:
        for op in ("+", "~", "-"):
            for (p, o, t) in sorted(sets[op]):
                fh.write("%s\t%s\t%s\t%s\n" % (op, p, o, t))


def main(argv):
    pos, expect, observed, show_list, quiet = [], None, None, False, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--expect", "--write-observed"):
            if i + 1 >= len(argv):
                sys.stderr.write(USAGE)
                return 2
            if a == "--expect":
                expect = argv[i + 1]
            else:
                observed = argv[i + 1]
            i += 2
            continue
        if a == "--list":
            show_list = True
        elif a == "--quiet":
            quiet = True
        elif a in ("-h", "--help"):
            sys.stdout.write(USAGE)
            return 0
        elif a.startswith("--"):
            sys.stderr.write("unknown option %s\n%s" % (a, USAGE))
            return 2
        else:
            pos.append(a)
        i += 1
    if len(pos) != 2:
        sys.stderr.write(USAGE)
        return 2

    try:
        before = read_reference(pos[0])
        after = read_reference(pos[1])
        exp = read_expect(expect) if expect is not None else None
    except ParseError as e:
        sys.stderr.write("golden_ref_diff: %s\n" % e)
        return 2

    sets = diff(before, after)
    if observed is not None:
        write_observed(observed, sets)

    ba, bb = tier_counts({(p, o, v[0]) for (p, o), v in before.items()})
    aa, ab = tier_counts({(p, o, v[0]) for (p, o), v in after.items()})
    print("---- golden_ref_diff ----")
    print("before: %d entries (tier A %d, tier B %d)  %s" % (len(before), ba, bb, pos[0]))
    print("after:  %d entries (tier A %d, tier B %d)  %s" % (len(after), aa, ab, pos[1]))
    if quiet:
        for label, op in (("added", "+"), ("changed", "~"), ("removed", "-")):
            a, b = tier_counts(sets[op])
            print("%-8s %6d  (tier A %d, tier B %d)" % (label, len(sets[op]), a, b))
    else:
        print_set("added", sets["+"], show_list)
        print_set("changed", sets["~"], show_list)
        print_set("removed", sets["-"], show_list)

    if exp is None:
        print("EXIT=0 (no --expect)")
        return 0

    ok = True
    for label, op in (("added", "+"), ("changed", "~"), ("removed", "-")):
        missing = exp[op] - sets[op]
        extra = sets[op] - exp[op]
        print("expect %-8s expected %d, observed %d, expected-not-observed %d, observed-not-expected %d"
              % (label, len(exp[op]), len(sets[op]), len(missing), len(extra)))
        for (p, o, t) in sorted(missing)[:50]:
            print("    MISSING %s [%s] %s | %s" % (op, t, p, o))
        for (p, o, t) in sorted(extra)[:50]:
            print("    UNEXPECTED %s [%s] %s | %s" % (op, t, p, o))
        if missing or extra:
            ok = False
    print("PREDICATE %s" % ("HOLDS (set equality)" if ok else "FAILS"))
    print("EXIT=%d" % (0 if ok else 1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
