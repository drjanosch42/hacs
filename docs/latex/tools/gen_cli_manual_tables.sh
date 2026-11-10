#!/usr/bin/env bash
# ===========================================================================
# gen_cli_manual_tables.sh -- regenerate the generated include files of the
# CLI / console user manual, docs/latex/cli_manual/*.tex (hecke-extensions
# slot 18, WP-M; modelled on docs/console/gen_console_md.sh).
#
#     docs/latex/tools/gen_cli_manual_tables.sh <path/to/hacs> [--check]
#
# The tables (verbs, types, settings, the command-line token tables, the usage
# texts, the validation roster, the counts) are a pure function of the
# binary: the script runs
#
#     CLI_MANUAL_EMIT=<scratch> <binary> CliManualCoverageValidation
#
# which writes them into a scratch directory OUTSIDE the source tree (the
# validation refuses an in-tree directory), and then copies them into
# docs/latex/cli_manual/ -- or, with --check, only diffs and exits 1 when the
# committed files are stale.  The binary is REQUIRED (no default: a stale
# leftover binary would silently write an outdated manual; see
# gen_console_md.sh's note).  Scratch: $CLI_MANUAL_SCRATCH, else
# ${XDG_CACHE_HOME:-~/.cache}/hacs/scratch (persistent, never /tmp); the
# temporary directory is removed on exit.
#
# Exit: 0 written / current, 1 stale (--check) or emit failure, 2 usage.
# ===========================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/docs/latex/cli_manual"
APP=""
CHECK=0
for a in "$@"; do
    case "$a" in
        --check) CHECK=1 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) APP="$a" ;;
    esac
done
if [[ -z "$APP" ]]; then
    echo "gen_cli_manual_tables.sh: pass the binary: gen_cli_manual_tables.sh <path/to/hacs> [--check]" >&2
    exit 2
fi
[[ -x "$APP" ]] || { echo "gen_cli_manual_tables.sh: '$APP' is not an executable file" >&2; exit 2; }

PARENT="${CLI_MANUAL_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/hacs/scratch}"
mkdir -p "$PARENT"
TMP="$(mktemp -d "$PARENT/cli_manual_gen.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

if ! CLI_MANUAL_EMIT="$TMP" "$APP" CliManualCoverageValidation > "$TMP.log" 2>&1; then
    cat "$TMP.log" >&2
    rm -f "$TMP.log"
    echo "gen_cli_manual_tables.sh: the emit run failed" >&2
    exit 1
fi
grep -q '^EFFECTIVE CliManualCoverageValidation arm=EMIT' "$TMP.log" || {
    cat "$TMP.log" >&2; rm -f "$TMP.log"
    echo "gen_cli_manual_tables.sh: no EFFECTIVE arm=EMIT line -- does this binary register CliManualCoverageValidation?" >&2
    exit 1
}
rm -f "$TMP.log"

shopt -s nullglob
GENERATED=("$TMP"/*.tex "$TMP"/*.json)
[[ ${#GENERATED[@]} -gt 0 ]] || { echo "gen_cli_manual_tables.sh: nothing was emitted" >&2; exit 1; }

if [[ "$CHECK" -eq 1 ]]; then
    stale=0
    for f in "${GENERATED[@]}"; do
        b="$(basename "$f")"
        if ! diff -u "$OUT/$b" "$f" > /dev/null 2>&1; then
            echo "STALE docs/latex/cli_manual/$b"
            diff -u "$OUT/$b" "$f" | head -20 || true
            stale=1
        fi
    done
    for f in "$OUT"/*.tex "$OUT"/*.json; do
        [[ -e "$TMP/$(basename "$f")" ]] || { echo "ORPHAN docs/latex/cli_manual/$(basename "$f") (the binary no longer generates it)"; stale=1; }
    done
    if [[ "$stale" -eq 0 ]]; then
        echo "gen_cli_manual_tables.sh: docs/latex/cli_manual/ is current (${#GENERATED[@]} files)"
        exit 0
    fi
    echo "gen_cli_manual_tables.sh: docs/latex/cli_manual/ is STALE -- regenerate it" >&2
    exit 1
fi

mkdir -p "$OUT"
for f in "$OUT"/*.tex "$OUT"/*.json; do
    [[ -e "$TMP/$(basename "$f")" ]] || { rm -f "$f"; echo "removed orphan $(basename "$f")"; }
done
cp "${GENERATED[@]}" "$OUT/"
echo "gen_cli_manual_tables.sh: wrote ${#GENERATED[@]} files to $OUT from $APP"

# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
