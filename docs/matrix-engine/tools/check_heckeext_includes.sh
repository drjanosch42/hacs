#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# check_heckeext_includes.sh -- the hecke-extensions HeckeExt/ include guard.
#
# WHY A GREP GUARD.  `src/hacs/HeckeExt/` is compiled straight
# into `hacs`, not into a static library
# (hecke_extensions_implementation_plan.md v1.5 Sec. 1.1: the layer consumes
# MatrixT, FpPolyT, HeckeModuleDecomposition, HeckeOperatorFamily and
# EquivariantCohomologyOnK, all of which are executable TUs, and a HeckeExt
# library would first need the Pipeline-library refactor the console plan
# records as an unscheduled follow-up).  There is therefore no link-time barrier
# to enforce the layer's ONE-WAY dependency, and this script is that barrier --
# the fourth grep-level guard next to check_matrix_includes.sh,
# raw_mod_calls.sh and check_console_includes.sh.  Like them it owns its
# allow-lists WITH A REASON PER ENTRY, and removing a line is the unit of
# progress.
#
#   RULE 1 (the forward rule).  No translation unit may `#include` a
#   `HeckeExt/` header except HeckeExt/ itself, Campaign/, Golden/, Bench/,
#   main.cpp, any *Validation.* and ValidationRegistry.cpp.  The pipeline never
#   depends on the extension layer.
#
#   RULE 2 (the reverse rule).  Inside HeckeExt/, no include of Console/, of the
#   CUDA layer (src/Cuda/), of the tape's internals
#   (src/Engine/) or of the deprecated BlockFlatMatrix.h.  The ext
#   layer is Matrix-sized ([X] Sec. 2.3) and does not use the tape; a few
#   index-disjoint loops (the atlas, the LValue measures) run on std::thread
#   through Util/ParallelFor.h (S24 CL-2).  It must not reach for the device or
#   the tape, and it must not acquire a console dependency.
#
#   RULE 2's ONE EXCEPTION (ALLOW_EXT_CONSOLE below, console phase C1b / slot-4
#   WP-5).  [OD] hecke-extensions slot-3 close-out 1 decided that the extension
#   layer registers its OWN console verbs, because rule 1 forbids
#   Console/Bindings/ from including a HeckeExt/ header.  The listed paths may
#   therefore include Console/ headers -- and Console/ ONLY: Cuda/, Engine/ and
#   BlockFlatMatrix.h stay banned in them too, and rules 1, 3 and 4 are
#   unchanged.  Each entry carries its reason; removing one is progress.
#
#   RULE 2's SECOND EXCEPTION (ALLOW_EXT_ENGINE below, S23 fix).  The listed
#   exact paths may include src/Engine/ headers -- and Engine/ ONLY:
#   Cuda/ and BlockFlatMatrix.h stay banned in them too.  Reason per entry below.
#
#   RULE 3 (the pipeline-free Sturm unit).  HeckeSturmBound.{h,cpp} may include
#   ONLY Hecke/SubspaceOps.h, Hecke/FpPoly.h, ModArithmetic's Matrix.h,
#   ModArithmetic.h and EngineCheck.h, its own header, and C++ standard headers
#   ([X] Sec. 1.1: "HeckeSturmBound is the one unit that consumes NOTHING of the
#   pipeline ... it is pure modular-form arithmetic on the Miller basis and can
#   be built and validated in isolation").  EngineCheck.h is not in [X]'s
#   four-item list because that list predates the policy that all new code uses
#   ENGINE_CHECK rather than a bare assert (the assertion policy:
#   ENGINE_CHECK survives NDEBUG); it is dependency-free (<stdexcept> + <string>) and is therefore
#   admitted here with that reason -- see docs/implementation/slot1/WP-3_report.md.
#   Fnv.h is admitted on the same grounds: it is dependency-free (<cstdint> +
#   <string>) and is [X] Sec. 1.1's "One src/ModArithmetic/Fnv.h",
#   which absorbed HeckeSturmBound's former local FNV copy (WP-2/WP-3 integration).
#   HeckeExtFinal.h (hecke-extensions slot 4, WP-1 / E2a) is admitted on the
#   same grounds again: it is dependency-free (<cstdint> + <string>), entirely
#   inline/constexpr, and holds THE ONE TRI-STATE `Final` that [X] Sec. 1.3.1
#   R3.2 requires HeckeSturmBound, HeckeGenerationCertificates, HeckeSturmSweep
#   and the E2 record layer to share (`SturmFinal` is now an alias of it).
#   Admitting it is what keeps that type from being duplicated.
#
#   RULE 4 (determinism).  No `unordered_` anywhere in HeckeExt/: the ext
#   objects are hashed and compared, and a hash container on a result path is
#   determinism hazard 02 Sec. 6.5(5).
#
# Usage:  check_heckeext_includes.sh [repo-root]
# Exit:   0 = clean, 1 = an offending line, 2 = usage/environment error.
# Output: one "FAIL <path>:<line>: <text>" line per offender, then a summary.
# ---------------------------------------------------------------------------
set -u

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
    # Default: the repository root, two levels above docs/matrix-engine/tools.
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
if [ ! -d "$ROOT/src/hacs" ] || [ ! -d "$ROOT/src/ModArithmetic" ]; then
    echo "check_heckeext_includes.sh: '$ROOT' is not the repository root." >&2
    exit 2
fi

EXT_DIR="$ROOT/src/hacs/HeckeExt"
if [ ! -d "$EXT_DIR" ]; then
    echo "check_heckeext_includes: OK -- no HeckeExt/ directory in this tree (nothing to check)."
    exit 0
fi

# --- Rule 1's allow-list --------------------------------------------------
# Repo-relative exact paths or shell globs; every entry carries a reason.
#
#   src/hacs/HeckeExt/*   the layer itself.
#   .../Campaign/*                          the campaign driver runs the ext
#                                           engine per (n, ell) cell ([X] Sec. 1.5).
#   .../Golden/*                            the golden harness hashes the ext
#                                           objects ([X] Sec. 1.6).
#   .../Bench/*                             `--bench ext` ([X] Sec. 1.5).
#   .../main.cpp                           the entry point / CLI.
#   *Validation.cpp / *Validation.h         the test harness.
#   .../Registry/Slot<S>.cpp               registration TUs: one per closed slot 9-23 (the
#                                           folded per-package fragments, S24 CL-10); they
#                                           include validation headers only.
#   .../Registry/Slot24/*.cpp               per-package registration TUs (slot 24,
#                                           same pattern); validation headers only
#                                           (MT1: HeckeModuleStreamValidation.h).
#   .../ValidationRegistry.cpp              it includes HeckeExt/*Validation.h
#                                           and matches no other pattern ([XA m9]).
ALLOW_FORWARD=(
    'src/hacs/HeckeExt/*'
    'src/hacs/Campaign/*'
    'src/hacs/Golden/*'
    'src/hacs/Bench/*'
    'src/hacs/main.cpp'
    '*Validation.cpp'
    '*Validation.h'
    'src/hacs/ValidationRegistry.cpp'
    'src/hacs/Registry/Slot[0-9]*.cpp'
    'src/hacs/Registry/Slot24/*.cpp'
)

# --- Rule 2's exception list -----------------------------------------------
# Repo-relative EXACT paths that may include Console/ headers (and nothing else
# rule 2 bans); every entry carries a reason.
#
#   src/hacs/HeckeExt/RegisterHeckeExtensionVerbs.cpp
#       [OD] slot-3 close-out 1: the extension layer's console registration TU.
#       The .cpp ALONE -- RegisterHeckeExtensionVerbs.h forward-declares
#       `console::Registry` and includes no Console/ header, so the header is
#       not an offending path and must never become one (slot-4 plan Sec. 2.3
#       item 4).
#   src/hacs/HeckeExt/RegisterHeckeExtensionVerbs{Detail.h,
#       Sturm,Algebra,Records,Values,Complex,Gallery}.cpp
#       S24 CL-8 (CODEAUDIT A10): the same TU split by area, same reason; the
#       private Detail.h is included by these seven TUs only.
#   src/hacs/HeckeExt/RegisterHeckeZRecordVerbs.cpp
#       S25 HZ-REC (HZ-LOC audit F8): the verb hecke_structure needs the record builder's
#       HeckeExt/ sources (HeckeZRecordSource.h), which Console/Bindings/ may not include
#       (rule 1); same reason as [OD] slot-3 close-out 1, the .cpp alone.
ALLOW_EXT_CONSOLE=(
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbs.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsDetail.h'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsSturm.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsAlgebra.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsRecords.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsValues.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsComplex.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsGallery.cpp'
    'src/hacs/HeckeExt/RegisterHeckeZRecordVerbs.cpp'
)

# --- Rule 2's Engine/ exception list ---------------------------------------
# Repo-relative EXACT paths that may include src/Engine/ headers (and
# nothing else rule 2 bans); every entry carries a reason.
#
#   src/hacs/HeckeExt/HeckeModuleStreamValidation.cpp
#       S23 G2-P: a test-harness TU.  Its thread-count arms must pin the engine
#       pool's consumer size (engine::ScopedConsumerPool, Engine/EngineRuntime.h,
#       and the pool type from Engine/ThreadPool.h) to run the same objects at
#       1 / 7 / 16 workers and compare them.  The extension layer proper
#       (non-validation HeckeExt/ files) stays Engine-free, with ONE exception:
#       HeckeSaturationDefect.cpp (S24 SATPERF, reason below; SATPERF audit sec. 4).
#       (S24 MT-1 audit: the forms-overlap exception for HeckeSaturationDefect.cpp
#       was removed together with HECKE_MODULE_MT_FORMS; MT1_audit.md F1.)
#   * HeckeSaturationDefect.cpp (S24 SATPERF, slot24/SATPERF_report.md): includes
#       Engine/EngineRuntime.h for engine::tape::ScopedHostElimHook ONLY, installed
#       (under HECKE_SAT_PERF, never around a module run) around the sat cell's
#       post-module stages so its whole-module End(N) products (EndCtx::mul_fast's
#       disjoint rows) and eliminations split over the consumer pool; a no-op at
#       one worker.  No tape, no executor, no device.
ALLOW_EXT_ENGINE=(
    'src/hacs/HeckeExt/HeckeModuleStreamValidation.cpp'
    'src/hacs/HeckeExt/HeckeSaturationDefect.cpp'
)

# --- Rule 3's allow-list ---------------------------------------------------
# The ONLY quoted includes HeckeSturmBound.{h,cpp} may carry (basenames).
ALLOW_STURM_HEADERS=(
    'HeckeSturmBound.h'
    'SubspaceOps.h'
    'FpPoly.h'
    'Matrix.h'
    'ModArithmetic.h'
    'EngineCheck.h'
    'Fnv.h'
    'HeckeExtFinal.h'   # slot-4 WP-1 (E2a): the one tri-state, dependency-free
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

fails=0
scanned=0

# --- Rule 1 ----------------------------------------------------------------
pattern_ext='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?HeckeExt/[^"]*"'
while IFS= read -r file; do
    rel="${file#"$ROOT"/}"
    scanned=$((scanned + 1))
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        if is_allowed_in "$rel" "${ALLOW_FORWARD[@]}"; then continue; fi
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//') [rule 1: HeckeExt/ is consumed one-way]"
        fails=$((fails + 1))
    done < <(grep -nE "$pattern_ext" "$file" 2>/dev/null)
done < <(find "$ROOT/src" \
             \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) -type f | sort)

# --- Rules 2, 3 and 4, inside HeckeExt/ ------------------------------------
# Rule 2's banned set, split so that the ONE exception can admit Console/ only.
pattern_banned_console='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?Console/[^"]*"'
pattern_banned_other='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?(Cuda/[^"]*|Engine/[^"]*|BlockFlatMatrix\.h)"'
pattern_banned_noengine='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?(Cuda/[^"]*|BlockFlatMatrix\.h)"'
pattern_quoted='^[[:space:]]*#[[:space:]]*include[[:space:]]*"'
while IFS= read -r file; do
    rel="${file#"$ROOT"/}"
    base="$(basename "$file")"

    # Rule 2 -- the Console/ half, subject to the one exception list.
    if ! is_allowed_in "$rel" "${ALLOW_EXT_CONSOLE[@]}"; then
        while IFS=: read -r lineno text; do
            [ -n "$lineno" ] || continue
            echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//') [rule 2: HeckeExt/ may not include Console/ -- see ALLOW_EXT_CONSOLE]"
            fails=$((fails + 1))
        done < <(grep -nE "$pattern_banned_console" "$file" 2>/dev/null)
    fi

    # Rule 2 -- Cuda/, Engine/ and BlockFlatMatrix.h; Engine/ alone is excepted
    # for ALLOW_EXT_ENGINE, Cuda/ and BlockFlatMatrix.h with NO exception.
    pattern_other="$pattern_banned_other"
    if is_allowed_in "$rel" "${ALLOW_EXT_ENGINE[@]}"; then pattern_other="$pattern_banned_noengine"; fi
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//') [rule 2: HeckeExt/ may not include Cuda/, Engine/ or BlockFlatMatrix.h]"
        fails=$((fails + 1))
    done < <(grep -nE "$pattern_other" "$file" 2>/dev/null)

    # Rule 3.
    case "$base" in
        HeckeSturmBound.h|HeckeSturmBound.cpp)
            while IFS=: read -r lineno text; do
                [ -n "$lineno" ] || continue
                hdr="$(echo "$text" | sed -E 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*)".*/\1/')"
                hdr="${hdr##*/}"
                if ! is_allowed_in "$hdr" "${ALLOW_STURM_HEADERS[@]}"; then
                    echo "FAIL $rel:$lineno: \"$hdr\" [rule 3: HeckeSturmBound is pipeline-free]"
                    fails=$((fails + 1))
                fi
            done < <(grep -nE "$pattern_quoted" "$file" 2>/dev/null)
            ;;
    esac

    # Rule 4.
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//') [rule 4: no hash container in the ext layer]"
        fails=$((fails + 1))
    done < <(grep -nE 'unordered_' "$file" 2>/dev/null | grep -vE '^[0-9]+:[[:space:]]*(//|\*)' )
done < <(find "$EXT_DIR" \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -type f | sort)

if [ "$fails" -eq 0 ]; then
    echo "check_heckeext_includes: OK -- $scanned files scanned, the HeckeExt/ dependency is one-way, HeckeSturmBound is pipeline-free, no hash container."
    exit 0
fi
echo "check_heckeext_includes: FAILED -- $fails offending line(s)."
echo "  Either remove the dependency, or add the file to the matching allow-list in $0 WITH A REASON."
exit 1
