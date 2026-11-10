#!/usr/bin/env bash
# ===========================================================================
# gen_console_md.sh -- regenerate docs/console/console.md, the console's
# reference manual (console phase C4, hecke-extensions slot 8, WP-10).
#
# [P] docs/console/00-console-interface-plan.md Sec. 4.4: the manual is
# GENERATED from the registry, never written by hand.  The generator is the
# binding verb `help_markdown()` (Console/Bindings/Reference.cpp; [P]'s
# `help --markdown`, spelled as a verb because Console/Core/ is frozen), so
# this script only runs it and writes the result:
#
#     docs/console/gen_console_md.sh [path/to/hacs] [--check]
#
# The binary defaults to ./build/hacs under the repository root --
# the out-of-source build every hecke-extensions package's `$WT/build` uses
# (plan Sec. 3.0 step 0) -- and ONLY that path (hecke-extensions slot 9,
# WP-9): a bare invocation deliberately does NOT fall back to a root-level
# `./hacs`.  That in-source binary is git-ignored and can be an
# ARBITRARILY OLD leftover (a worktree can be created with one already sitting
# on disk from whatever state the tree was in when the worktree was made; a
# checkout can carry one from months ago), so silently picking it up risks
# writing this doc from a stale registry with no warning at all. With no
# ./build/hacs the script REQUIRES the path as an explicit argument,
# with a clear message naming the binary it looked for and why it did not
# fall back further.
#
# With --check nothing is written: the script exits 1 (and prints a diff)
# when the committed file differs from the generated text, 0 when it is
# current.  Regenerate whenever a verb, a type or a setting is registered.
# The text is a pure function of the registry: no environment value, timing
# or machine property enters it.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/docs/console/console.md"
APP=""
CHECK=0
for a in "$@"; do
    case "$a" in
        --check) CHECK=1 ;;
        -h|--help) sed -n '2,29p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) APP="$a" ;;
    esac
done
if [[ -z "$APP" ]]; then
    DEFAULT_APP="$ROOT/build/hacs"
    if [[ -x "$DEFAULT_APP" ]]; then
        APP="$DEFAULT_APP"
    else
        echo "gen_console_md.sh: no $DEFAULT_APP (the out-of-source build)." >&2
        echo "  Not falling back to a root-level ./hacs: that binary is" >&2
        echo "  git-ignored and may be an arbitrarily stale leftover (e.g. copied into a" >&2
        echo "  freshly created worktree), which would write this doc from an outdated" >&2
        echo "  registry with no warning." >&2
        echo "  Build under ./build (cmake -B build -S . && make -C build -j6) or pass the" >&2
        echo "  binary's path explicitly: gen_console_md.sh /path/to/hacs" >&2
        exit 2
    fi
fi
[[ -n "$APP" && -x "$APP" ]] || { echo "gen_console_md.sh: '$APP' is not an executable file" >&2; exit 2; }

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
"$APP" -c 'print(help_markdown())' > "$TMP"

if [[ "$CHECK" -eq 1 ]]; then
    if diff -u "$OUT" "$TMP"; then
        echo "gen_console_md.sh: docs/console/console.md is current"
        exit 0
    fi
    echo "gen_console_md.sh: docs/console/console.md is STALE -- regenerate it" >&2
    exit 1
fi
cp "$TMP" "$OUT"
echo "gen_console_md.sh: wrote $OUT ($(wc -l < "$OUT") lines) from $APP"

# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
