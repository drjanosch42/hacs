#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# check_matrix_includes.sh -- matrix-engine Phase 3 acceptance guard.
#
# Asserts that NO translation unit outside the allow-lists below includes a
# DEPRECATED substrate header DIRECTLY.  Two headers are guarded:
#
#   Matrix.h           the deprecated dense matrix (Phase 3 exit criterion).
#   BlockFlatMatrix.h  the deprecated block container: LEGACY ARM ONLY since
#                      Phase 3 (the live model is engine::BlockMat /
#                      engine::blk).  Since S24 CL-L2 it lives in
#                      src/ModArithmetic/legacy/ and is admitted only
#                      from files under a legacy/ directory, so that a new
#                      consumer cannot appear without this guard failing.
#
# Historically this script guarded Matrix.h only.  Phase 3's exit criterion is
# "zero `Matrix` includes outside MatrixValidation / HeckeModuleDecomposition"
# (docs/matrix-engine/00-implementation-plan.md, Phase 3); this script is what
# makes that criterion checkable, and *removing a file from the allow-list is
# the unit of progress* -- the same convention as the raw-mod_* grep script.
#
# It checks DIRECT includes only.  `Matrix.h` is still reachable from every TU
# transitively, because FlatMatrix.h includes it for FlatMatrixT's
# MatrixT-conversion constructor; that is by design and is not what this guard
# is about.  What it forbids is a file reaching for MatrixT as its own working
# type.
#
# Usage:  check_matrix_includes.sh [repo-root]
# Exit:   0 = clean, 1 = an offending include, 2 = usage/environment error.
# Output: one "FAIL <path>:<line>: <text>" line per offender, then a summary.
# ---------------------------------------------------------------------------
set -u

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
    # Default: the repository root, two levels above docs/matrix-engine/tools.
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
if [ ! -d "$ROOT/src/hacs" ] || [ ! -d "$ROOT/src/ModArithmetic" ]; then
    echo "check_matrix_includes.sh: '$ROOT' is not the repository root." >&2
    exit 2
fi

# --- The allow-list ------------------------------------------------------
# Every entry is a repo-relative path (exact match) or a shell glob, and every
# entry carries a reason.  Nothing may be added without one.
#
#   ModArithmetic/*                     the substrate itself defines Matrix.
#   *Validation.cpp                     the test harness; re-pointed later
#                                       (plan Phase 3, "compatibility accessors").
#   Hecke/HeckeModuleDecomposition.*    d_q <= ~20, 0.004 % of runtime --
#   Hecke/FpPoly.*                      explicitly left alone (audit 03 Sec. 4.3).
#   Hecke/SubspaceOps.*                 (hecke-extensions E0a) SolveColumnsExact /
#                                       HStack / KernelAsMatrix / PrimaryRCFBasis
#                                       moved VERBATIM out of
#                                       HeckeModuleDecomposition.cpp, whose
#                                       MatrixT typing they keep by construction
#                                       (the move must not change a value; it is
#                                       gated by GOLDEN_STRICT tier B's
#                                       decomp_*_U). Same scope decision as the
#                                       two lines above: d_q <= ~20, level-(c)
#                                       only, explicitly left alone.
#   HeckeExt/HeckeSturmBound.*          (hecke-extensions E1s / WP-3) the ext
#                                       layer is MatrixT<uint32_t>-typed by
#                                       [X] Sec. 1.1's decision ("MatrixT<uint32_t>
#                                       for everything in HeckeExt/ except the
#                                       cone"): every object is D-sized
#                                       (D <= dim M_k <= ~84 at n = 1000, and
#                                       D <= ~20 for the decomposition blocks),
#                                       so the block model would buy nothing and
#                                       the MatrixT RREF/kernel/inverse surface
#                                       is exactly what the certificates
#                                       re-derive from.  Same scope decision as
#                                       HeckeModuleDecomposition / SubspaceOps
#                                       above.  (HeckeExt/*Validation.* are
#                                       already covered by the *Validation
#                                       globs.)
#   Golden/GoldenHarness.cpp            the frozen-hash harness; it hashes the
#                                       MatrixT-typed A^{(q)} that the
#                                       decomposition above consumes, and calls
#                                       its MatrixT-only CharPoly/Rank/Trace.
#   Bench/BenchDriver.cpp               same A^{(q)} boundary, read-only.
#   HeckeExt/HeckeIdentityCertificate.* (hecke-extensions slot 1, WP-2) the
#                                       [X Sec.1.1] placement decision
#                                       "MatrixT<uint32_t> for everything in
#                                       HeckeExt/" -- its stored operator
#                                       matrices (A^{(1)}(T_p) etc.) arrive
#                                       from HeckeOperatorFamily, itself on
#                                       MatrixT (the level-(b) A^{(q)}
#                                       boundary this whole allow-list is
#                                       about).
#   HeckeExt/HeckeLocalAlgebra.h        (hecke-extensions slot 2, WP-4 / E1a)
#                                       the same [X Sec.1.1] decision, one
#                                       layer on: the local algebra A_S of a
#                                       decomposition block.  Its inputs are a
#                                       hecke::PrimaryBlock and a
#                                       hecke::LocalityVerdict, BOTH declared
#                                       in the MatrixT-typed
#                                       HeckeModuleDecomposition.h above, and
#                                       its whole surface (RREF, kernel,
#                                       inverse, entrywise comparison of
#                                       certificate bases) is what the
#                                       certificate re-derives from; D <= ~200,
#                                       so the block model would buy nothing.
#                                       Only the HEADER is listed: the .cpp
#                                       reaches Matrix.h through it and needs
#                                       no entry of its own, and
#                                       HeckeExtAlgebraValidation.* is already
#                                       covered by the *Validation globs.
#   HeckeExt/HeckeBoundary.*            (hecke-extensions slot 2, WP-7 / E4b)
#                                       the same [X Sec.1.1] decision: its
#                                       stored objects -- r^q, the interior
#                                       basis of H^1_! = ker r^1 (the RREF
#                                       kernel of a dim H^1(bd) x dim H^1
#                                       matrix), A_!, the Eisenstein vector,
#                                       A_bd | im r^1 -- are D-sized level-(b)
#                                       matrices (D = dim H^1 <= ~200) arriving
#                                       from HeckeOperatorOnK / HeckeBoundary-
#                                       Operator's MatrixT A^{(q)}, and its
#                                       certificate re-derives everything with
#                                       the MatrixT rank / kernel / exact-solve
#                                       surface (SubspaceOps). The block model
#                                       would buy nothing here.
#   HeckeExt/HeckeSignSplit.h           (hecke-extensions slot 4, WP-3 / E3)
#                                       the same [X Sec.1.1] decision once
#                                       more: the sign split's stored objects
#                                       -- iota^* on H^q, the two eigenbases,
#                                       A^{(q)}_+- by an exact solve, and the
#                                       per-sign block data -- are D-sized
#                                       level-(b) matrices (D = dim H^q <=
#                                       ~200) arriving from Hecke/Hecke-
#                                       Involution, and the unit consumes
#                                       hecke::HeckeModuleDecomposition and
#                                       HeckeExt/HeckeLocalAlgebra, both of
#                                       which are MatrixT already.  Only the
#                                       HEADER is listed (the HeckeLocal-
#                                       Algebra.h precedent): HeckeSignSplit.cpp
#                                       reaches Matrix.h through it, and
#                                       HeckeInvolutionValidation.* is covered
#                                       by the *Validation globs.
ALLOW=(
    'src/ModArithmetic/*'
    'src/Engine/*'
    'src/Cuda/*'
    '*Validation.cpp'
    '*Validation.h'
    'src/hacs/Hecke/HeckeModuleDecomposition.h'
    'src/hacs/Hecke/HeckeModuleDecomposition.cpp'
    'src/hacs/Hecke/FpPoly.h'
    'src/hacs/Hecke/FpPoly.cpp'
    'src/hacs/Hecke/SubspaceOps.h'
    'src/hacs/Hecke/SubspaceOps.cpp'
    'src/hacs/HeckeExt/HeckeSturmBound.h'
    'src/hacs/HeckeExt/HeckeSturmBound.cpp'
    'src/hacs/Golden/GoldenHarness.cpp'
    'src/hacs/Bench/BenchDriver.cpp'
    'src/hacs/HeckeExt/HeckeIdentityCertificate.h'
    'src/hacs/HeckeExt/HeckeIdentityCertificate.cpp'
    'src/hacs/HeckeExt/HeckeLocalAlgebra.h'
    'src/hacs/HeckeExt/HeckeBoundary.h'
    'src/hacs/HeckeExt/HeckeBoundary.cpp'
    'src/hacs/HeckeExt/HeckeSignSplit.h'
)

# --- The BlockFlatMatrix.h allow-list ------------------------------------
#   */legacy/*                          (S24 CL-L2, CLEANUP plan Sec. 2.2) since the
#                                       legacy backend moved behind OC_ENABLE_LEGACY,
#                                       BlockFlatMatrix.h lives in
#                                       src/ModArithmetic/legacy/ and may be
#                                       included ONLY from a file under a legacy/
#                                       directory (the .cpp / .inc sources compiled
#                                       only with OC_ENABLE_LEGACY=ON).  The former
#                                       entries (ModArithmetic/*, *Validation.*,
#                                       EquivariantCohomologyOnK.h,
#                                       SimplicialCohomology.h) are gone: those files
#                                       reach it through their legacy/*.inc halves.
ALLOW_BFM=(
    '*/legacy/*'
)

is_allowed_in() {
    local path="$1" pat
    shift
    for pat in "$@"; do
        # shellcheck disable=SC2053
        case "$path" in $pat) return 0 ;; esac
    done
    return 1
}

is_allowed() { is_allowed_in "$1" "${ALLOW[@]}"; }
is_allowed_bfm() { is_allowed_in "$1" "${ALLOW_BFM[@]}"; }

fails=0
scanned=0
# Match a real preprocessor include of Matrix.h (any relative prefix), not a
# commented-out one and not FlatMatrix.h / BlockFlatMatrix.h.
pattern='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?Matrix\.h"'
pattern_bfm='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?BlockFlatMatrix\.h"'

while IFS= read -r file; do
    rel="${file#"$ROOT"/}"
    scanned=$((scanned + 1))
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        if is_allowed "$rel"; then continue; fi
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//')"
        fails=$((fails + 1))
    done < <(grep -nE "$pattern" "$file" 2>/dev/null)
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        if is_allowed_bfm "$rel"; then continue; fi
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//')"
        fails=$((fails + 1))
    done < <(grep -nE "$pattern_bfm" "$file" 2>/dev/null)
done < <(find "$ROOT/src" \
             \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.inc' \) -type f | sort)

# --- Census (slot-24 CL-M, CODEAUDIT A9; informational, never fails) -------
# HeckeExt's `ExtMatrix` (= MatrixT<uint32_t>, HeckeExt/HeckeLocalAlgebra.h) is
# the one SANCTIONED MatrixT working type (the small dense F_ell algebra, [X]
# Sec. 2.3); Matrix.h carries the same note.  Its spread is printed so that a
# later migration (a FlatMatrix-backed ExtMatrix) has a baseline to measure.
ext_files=0
ext_sites=0
while IFS= read -r file; do
    n=$(grep -ow 'ExtMatrix' "$file" 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then ext_files=$((ext_files + 1)); ext_sites=$((ext_sites + n)); fi
done < <(find "$ROOT/src/hacs" \( -name '*.h' -o -name '*.cpp' \) -type f | sort)
echo "check_matrix_includes: census -- ExtMatrix (= MatrixT<uint32_t>, the sanctioned MatrixT user) $ext_sites site(s) in $ext_files file(s)."

if [ "$fails" -eq 0 ]; then
    echo "check_matrix_includes: OK -- $scanned files scanned, no direct Matrix.h / BlockFlatMatrix.h include outside the allow-lists."
    exit 0
fi
echo "check_matrix_includes: FAILED -- $fails direct deprecated-header include(s) outside the allow-lists (of $scanned files scanned)."
echo "  Either migrate the file (FlatMatrix for Matrix.h, engine::BlockMat for BlockFlatMatrix.h),"
echo "  or add it to ALLOW / ALLOW_BFM in $0 WITH A REASON."
exit 1
