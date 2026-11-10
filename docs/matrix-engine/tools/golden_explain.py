#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# golden_explain.py -- hecke-extensions phase E0a.
#
# The object-by-object explanation of a golden re-baseline, as required by
# the re-baseline rule ("Every changed line of a re-baseline must be explained object by
# object", CONTRIBUTING.md) and specified by hecke_extensions_implementation_plan.md Sec. 1.6,
# step 3.  GoldenReference.inc stores HASHES, so the explanation needs the
# TEXTS: `GOLDEN_DUMP=<dir> ./hacs GoldenDump` writes them, once on
# the "before" binary and once on the "after" one, and this script diffs the
# two dumps.  It does two things, and FAILS on either:
#
#   (i)  TEXT objects (report_full^q, report_canonical^q, op_json_full, ...):
#        for every object whose text changed it prints the JSON KEY PATHS that
#        differ, and fails unless every one of them is in the allow-list for
#        this re-baseline (--allow-keys FILE; without it NOTHING is allowed, so
#        any text change fails).
#
#   (ii) DECOMPOSITION block lists: it diffs them and fails on ANY difference
#        in any input of the four golden hashes _invariants / _certify /
#        _witness / _U, unless that decomposition is named in an owner-accepted
#        split list (--accept-split "<point> <object>"), which the re-baseline
#        commit then quotes.
#
# An object present only in `after` is an ADDITION (the trivially explained
# case: E1-E5 only add object names) -- reported, never a failure.  An object
# that DISAPPEARED is a failure.
#
# Usage:
#   golden_explain.py BEFORE AFTER [--allow-keys FILE]
#                     [--accept-split "<point> <object>"]... [--quiet]
# Exit: 0 = every difference explained, 1 = an unexplained difference,
#       2 = usage / environment error.
#
# Hand-rolled except for the stdlib `json` module (the repo's "no third-party
# library" convention).
# ---------------------------------------------------------------------------

import json
import os
import sys

USAGE = (
    "usage: golden_explain.py BEFORE AFTER [--allow-keys FILE]\n"
    "                        [--accept-split \"<point> <object>\"]... [--quiet]\n"
)


# --- dump readers ---------------------------------------------------------

def read_texts(root):
    """<root>/texts.tsv -> {(point, object): payload}."""
    path = os.path.join(root, "texts.tsv")
    out = {}
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise ValueError("%s: malformed text record: %.80s" % (path, line))
            out[(parts[0], parts[1])] = parts[2]
    return out


def read_decomps(root):
    """<root>/decompositions.txt -> {(point, object): [line, ...]}."""
    path = os.path.join(root, "decompositions.txt")
    out = {}
    key = None
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("## "):  # checked BEFORE the '#' comment rule
                parts = line[3:].split("\t")
                if len(parts) != 2:
                    raise ValueError("%s: malformed header: %.80s" % (path, line))
                key = (parts[0], parts[1])
                out[key] = []
                continue
            if line.startswith("#"):
                continue
            if key is None:
                raise ValueError("%s: record line before any header" % path)
            out[key].append(line)
    return out


def read_allow_keys(path):
    """One JSON key-path pattern per line; '#' comments; '[]' matches any index."""
    pats = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                pats.add(line)
    return pats


# --- JSON key-path diff ---------------------------------------------------

def erase_indices(path):
    """'level_c.blocks[3].local' -> 'level_c.blocks[].local'."""
    out = []
    skipping = False
    for ch in path:
        if ch == "[":
            skipping = True
            out.append(ch)
        elif ch == "]":
            skipping = False
            out.append(ch)
        elif not skipping:
            out.append(ch)
    return "".join(out)


def join(prefix, key):
    return key if not prefix else prefix + "." + key


def diff_json(a, b, prefix=""):
    """Yields the key paths at which the two decoded JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield join(prefix, k)
            else:
                for p in diff_json(a[k], b[k], join(prefix, k)):
                    yield p
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield prefix + "[#len]"
            return
        for i, (x, y) in enumerate(zip(a, b)):
            for p in diff_json(x, y, "%s[%d]" % (prefix, i)):
                yield p
        return
    if a != b:
        yield prefix if prefix else "<root>"


def text_diff_paths(before, after):
    """Key paths that differ, or ['<unparseable-json>'] if either side is not JSON."""
    try:
        a = json.loads(before)
        b = json.loads(after)
    except Exception:
        return ["<unparseable-json>"]
    return list(diff_json(a, b))


# --- main -----------------------------------------------------------------

def main(argv):
    if len(argv) < 3:
        sys.stderr.write(USAGE)
        return 2
    before_dir, after_dir = argv[1], argv[2]
    allow_keys = set()
    accept_split = set()
    quiet = False
    i = 3
    while i < len(argv):
        arg = argv[i]
        if arg == "--allow-keys":
            if i + 1 >= len(argv):
                sys.stderr.write(USAGE)
                return 2
            allow_keys |= read_allow_keys(argv[i + 1])
            i += 2
        elif arg == "--accept-split":
            if i + 1 >= len(argv):
                sys.stderr.write(USAGE)
                return 2
            accept_split.add(argv[i + 1].strip())
            i += 2
        elif arg == "--quiet":
            quiet = True
            i += 1
        else:
            sys.stderr.write("golden_explain.py: unknown argument %r\n" % arg)
            sys.stderr.write(USAGE)
            return 2

    for d in (before_dir, after_dir):
        if not os.path.isdir(d):
            sys.stderr.write("golden_explain.py: %s is not a directory\n" % d)
            return 2
        for f in ("texts.tsv", "decompositions.txt"):
            if not os.path.isfile(os.path.join(d, f)):
                sys.stderr.write("golden_explain.py: %s/%s is missing "
                                 "(run GOLDEN_DUMP=%s ./hacs GoldenDump)\n"
                                 % (d, f, d))
                return 2

    try:
        tb, ta = read_texts(before_dir), read_texts(after_dir)
        db, da = read_decomps(before_dir), read_decomps(after_dir)
    except (OSError, ValueError) as exc:
        sys.stderr.write("golden_explain.py: %s\n" % exc)
        return 2

    failures = 0
    path_counts = {}
    changed_texts = 0

    # --- (i) texts -------------------------------------------------------
    for key in sorted(set(tb) | set(ta)):
        point, obj = key
        if key not in ta:
            failures += 1
            print("FAIL  text object DISAPPEARED: %s / %s" % (point, obj))
            continue
        if key not in tb:
            if not quiet:
                print("ADDED text object: %s / %s" % (point, obj))
            continue
        if tb[key] == ta[key]:
            continue
        changed_texts += 1
        paths = text_diff_paths(tb[key], ta[key])
        if not paths:
            paths = ["<byte-difference-outside-json>"]
        for p in paths:
            path_counts[erase_indices(p)] = path_counts.get(erase_indices(p), 0) + 1
            if p in allow_keys or erase_indices(p) in allow_keys:
                continue
            failures += 1
            print("FAIL  text key path NOT allow-listed: %s / %s : %s" % (point, obj, p))

    # --- (ii) decompositions --------------------------------------------
    changed_decomps = 0
    for key in sorted(set(db) | set(da)):
        point, obj = key
        if key not in da:
            failures += 1
            print("FAIL  decomposition DISAPPEARED: %s / %s" % (point, obj))
            continue
        if key not in db:
            if not quiet:
                print("ADDED decomposition: %s / %s" % (point, obj))
            continue
        if db[key] == da[key]:
            continue
        changed_decomps += 1
        name = "%s %s" % (point, obj)
        accepted = name in accept_split
        lb, la = db[key], da[key]
        detail = []
        for idx in range(max(len(lb), len(la))):
            x = lb[idx] if idx < len(lb) else "<absent>"
            y = la[idx] if idx < len(la) else "<absent>"
            if x != y:
                detail.append("      - %.160s\n      + %.160s" % (x, y))
            if len(detail) >= 4:
                detail.append("      ... (further lines differ)")
                break
        if accepted:
            print("ACCEPTED SPLIT (owner): %s / %s" % (point, obj))
        else:
            failures += 1
            print("FAIL  decomposition CHANGED (not in --accept-split): %s / %s" % (point, obj))
        if not quiet:
            print("\n".join(detail))

    # --- summary ---------------------------------------------------------
    print("---- golden_explain summary ----")
    print("text objects: %d before, %d after, %d changed" % (len(tb), len(ta), changed_texts))
    if path_counts:
        print("changed JSON key paths (count per path):")
        for p in sorted(path_counts):
            mark = "allowed" if (p in allow_keys) else "NOT ALLOWED"
            print("  %-60s %6d  [%s]" % (p, path_counts[p], mark))
    print("decompositions: %d before, %d after, %d changed, %d accepted splits"
          % (len(db), len(da), changed_decomps, len(accept_split)))
    print("unexplained differences: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
