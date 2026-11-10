#!/usr/bin/env bash
# ===========================================================================
# raw_mod_calls.sh -- build-time census of raw ModArithmetic `mod_*` call sites
# under src/hacs/, gated by an explicit allow-list.
#
# WHY. 00-implementation-plan.md, Phase 3 "Validations", requires
# `ConsumerMigrationValidation` to "assert the raw-`mod_*`-loop count by a
# build-time grep script in docs/matrix-engine/tools/", and states that
# **the grep target is not zero**: the largest concentrations sit in files the
# plan deliberately leaves alone (FpPoly.cpp, HeckeModuleDecomposition.cpp,
# HeckeTauTable.cpp's exact-Z solver, every *Validation.cpp). The assertion is
# therefore "zero outside the allow-list", and REMOVING A FILE FROM THE
# ALLOW-LIST IS THE UNIT OF PROGRESS -- see raw_mod_calls.allow.
#
# USAGE
#   docs/matrix-engine/tools/raw_mod_calls.sh [--root DIR] [--allow FILE]
#                                             [--census] [--quiet]
#   --root DIR    repository root (default: two levels above this script)
#   --allow FILE  allow-list (default: raw_mod_calls.allow beside this script)
#   --census      also print the per-file table for allow-listed files
#   --quiet       print only the final one-line verdict
#
# EXIT CODE = number of call sites OUTSIDE the allow-list (0 == clean),
# clamped to 125 so it stays a valid process exit status; 126 signals a usage
# or environment error. The count itself is always printed, unclamped.
#
# WHAT COUNTS AS A CALL SITE. One occurrence of `NAME(` (optionally qualified
# `ModArithmetic::NAME(`) where NAME is one of the 21 exported primitives of
# ModArithmetic.h -- mod_{mul,add,sub,minus,inv,pow,nCr}, their `64` variants
# and their width-generic `_w` wrappers. `//` line comments and `/* */` block
# comments are stripped first, so the file-header prose does not inflate the
# count. Declarations are not distinguished from calls, but ModArithmetic.h
# itself is outside the scanned tree, so under src/hacs/
# every hit is a use.
#
# CENSUS OF RECORD (branch phase3/morph, 2026-09-06, after Phase 3 Stage B's
# morphism/presets/SimplicialCohomology/OperatorFamily migrations; `--census`)
#   TOTAL 163 sites in 20 files, ALL allow-listed. The Stage B backlog is down
#   to five files (was nine): HeckeCosetTables, HeckeOperatorOnK, BarCocycleH1,
#   HeckeCrossChecks, EquivariantCohomologyOnK -- owned by the other Stage B
#   engineers. Cleared here: EquivariantCochainComplexMorphismOnK.cpp (2),
#   LocalSystemPresets.cpp (6), SimplicialCohomology.cpp (3),
#   Hecke/HeckeOperatorFamily.cpp (3).
#
#     25  Hecke/FpPoly.cpp                              permanent
#     21  Hecke/HeckeBlockFormValidation.cpp            permanent (*Validation)
#     20  Hecke/HeckeSimultaneousValidation.cpp         permanent (*Validation)
#     16  Hecke/HeckeCosetTables.cpp                    STAGE B
#     15  Hecke/HeckeOperatorOnK.cpp                    STAGE B
#     15  Hecke/BarCocycleH1.cpp                        STAGE B
#     10  Hecke/HeckeCrossChecks.cpp                    STAGE B
#      7  Hecke/HeckeTauTable.cpp                       permanent
#      6  EquivariantCohomologyOnK.cpp                  STAGE B
#      5  Hecke/ThetaEqSolveValidation.cpp              permanent (*Validation)
#      5  Hecke/HeckeOracleValidation.cpp               permanent (*Validation)
#      4  Hecke/HeckeEndToEndValidation.cpp             permanent (*Validation)
#      3  Hecke/HeckeRelationsValidation.cpp            permanent (*Validation)
#      2  Hecke/HeckeExtendedRegimeValidation.cpp       permanent (*Validation)
#      2  Golden/GoldenHarness.cpp                      permanent (Golden/)
#      2  DenseElimPipelineValidation.cpp               permanent (*Validation)
#      2  BlockModelPipelineValidation.cpp              permanent (*Validation)
#      1  Hecke/HeckeOperatorFamilySmokeValidation.cpp  permanent (*Validation)
#      1  Hecke/HeckeModuleDecomposition.cpp            permanent
#      1  Hecke/HeckeDecompositionSmokeValidation.cpp   permanent (*Validation)
#
# STAGE B UPDATE (branch phase3/xchk, 2026-09-06). The cross-check / coset /
# bar-route files cleared their raw sites and left the backlog:
#     Hecke/HeckeCrossChecks.cpp    10 -> 0   (block assembly + block GEMMs)
#     Hecke/BarCocycleH1.cpp        15 -> 0   (DenseElim solve + ModCtx)
#     Hecke/HeckeCosetTables.cpp    16 -> 0   (one ModCtx for the level N)
#   TOTAL 143 sites in 22 files, 0 outside the allow-list; the Stage B backlog
#   is down to 6 files (HeckeOperatorOnK.cpp, LocalSystemPresets.cpp,
#   EquivariantCohomologyOnK.cpp, SimplicialCohomology.cpp,
#   HeckeOperatorFamily.cpp, EquivariantCochainComplexMorphismOnK.cpp) plus the
#   new permanent-by-category *Validation entries.
#
# ON THE PLAN'S FIGURE OF 146. The plan (Phase 3, "Validations") quotes "146
# ModArithmetic::mod_* call sites under src/hacs/". That
# figure counts only the FULLY QUALIFIED `ModArithmetic::mod_*(` spelling; on
# the same tree that count is 149 today (the +3 is post-audit churn: the Phase
# 1/2 validations ThetaEqSolveValidation.cpp and DenseElimPipelineValidation.cpp
# did not exist when 00/02 were written). This script counts the qualified and
# the unqualified spelling alike -- 175 -- because a `using namespace
# ModArithmetic;` in a TU must not be able to hide a raw loop from the gate.
# ===========================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ALLOW="$SCRIPT_DIR/raw_mod_calls.allow"
CENSUS=0
QUIET=0

while [ $# -gt 0 ]; do
    case "$1" in
        --root)   ROOT="${2:-}"; shift 2 || exit 126 ;;
        --allow)  ALLOW="${2:-}"; shift 2 || exit 126 ;;
        --census) CENSUS=1; shift ;;
        --quiet)  QUIET=1; shift ;;
        -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "raw_mod_calls.sh: unknown argument '$1'" >&2; exit 126 ;;
    esac
done

SCAN="$ROOT/src/hacs"
if [ ! -d "$SCAN" ]; then
    echo "raw_mod_calls.sh: not a repository root (no src/hacs/): $ROOT" >&2
    exit 126
fi
if [ ! -f "$ALLOW" ]; then
    echo "raw_mod_calls.sh: allow-list not found: $ALLOW" >&2
    exit 126
fi

# The 21 exported primitives of ModArithmetic.h.
NAMES='mod_(mul|add|sub|minus|inv|pow|nCr)(64|_w)?'

# --- allow-list matching -------------------------------------------------
# `path` is relative to src/hacs/. An entry matches if it is
# the exact path, a directory prefix (trailing '/'), or a glob on the basename.
allowed() {
    local path="$1" base entry
    base="${path##*/}"
    while IFS= read -r entry; do
        entry="${entry%%#*}"                       # strip comments
        entry="$(printf '%s' "$entry" | tr -d '[:space:]')"
        [ -z "$entry" ] && continue
        case "$entry" in
            */) case "$path" in "$entry"*) return 0 ;; esac ;;
            *)  [ "$path" = "$entry" ] && return 0
                # shellcheck disable=SC2254
                case "$base" in $entry) return 0 ;; esac ;;
        esac
    done < "$ALLOW"
    return 1
}

# --- per-file count ------------------------------------------------------
# Strip /* */ block comments and // line comments, then count occurrences.
count_file() {
    awk '
        BEGIN { inblk = 0 }
        {
            line = $0; out = ""
            while (length(line) > 0) {
                if (inblk) {
                    i = index(line, "*/")
                    if (i == 0) { line = ""; break }
                    line = substr(line, i + 2); inblk = 0
                } else {
                    i = index(line, "/*"); j = index(line, "//")
                    if (j > 0 && (i == 0 || j < i)) { out = out substr(line, 1, j - 1); line = ""; break }
                    if (i == 0) { out = out line; line = ""; break }
                    out = out substr(line, 1, i - 1)
                    line = substr(line, i + 2); inblk = 1
                }
            }
            print out
        }
    ' "$1" | grep -oE "(ModArithmetic::)?\b${NAMES}[[:space:]]*\(" | wc -l
}

total=0
outside=0
outside_files=0
report=""

while IFS= read -r f; do
    rel="${f#"$SCAN"/}"
    n="$(count_file "$f")"
    [ "$n" -eq 0 ] && continue
    total=$((total + n))
    if allowed "$rel"; then
        [ "$CENSUS" -eq 1 ] && report="${report}$(printf '%6d  %-46s allow\n' "$n" "$rel")"$'\n'
    else
        outside=$((outside + n))
        outside_files=$((outside_files + 1))
        report="${report}$(printf '%6d  %-46s NOT ALLOWED\n' "$n" "$rel")"$'\n'
    fi
done < <(find "$SCAN" \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' \) -type f | LC_ALL=C sort)

if [ "$QUIET" -eq 0 ]; then
    echo "raw_mod_calls: scanning ${SCAN#"$ROOT"/} against ${ALLOW##*/}"
    printf '%s' "$report" | LC_ALL=C sort -rn
    echo "raw_mod_calls: TOTAL $total site(s); OUTSIDE allow-list $outside site(s) in $outside_files file(s)"
fi

if [ "$outside" -eq 0 ]; then
    echo "raw_mod_calls: PASS (0 sites outside the allow-list; $total allow-listed)"
    exit 0
fi
echo "raw_mod_calls: FAIL ($outside site(s) outside the allow-list)"
[ "$outside" -gt 125 ] && exit 125
exit "$outside"

# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
