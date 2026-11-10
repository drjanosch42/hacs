#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# check_console_includes.sh -- console C0 acceptance guard.
#
# The console plan's one-way dependency (docs/console/00-console-interface-plan.md
# v2, Sec. 2 C8 and Sec. 9) has TWO halves, and this script checks both.  It is
# the third grep-level guard next to check_matrix_includes.sh and
# raw_mod_calls.sh, and like them it owns its allow-lists WITH A REASON PER
# ENTRY; removing a line from one is the unit of progress.
#
#   RULE 1 (the reverse rule).  src/hacs/Console/Core/ is the
#   ConsoleCore static library, "the analogue of HeckeCore: a pure static
#   library that the executable links ... unit-tested by the language
#   validation without constructing a single matrix".  So every `#include <...>`
#   there must be a C++ standard header, and every `#include "..."` must name a
#   file that lives in Console/Core/ itself.  A POSIX header, a CUDA header or
#   anything from ModArithmetic/ or the pipeline is a FAILURE -- that is what
#   makes "ConsoleCore has no include outside the STL" (plan Sec. 10, C0
#   acceptance (b)) a checkable property rather than a promise.
#
#   RULE 2 (the forward rule).  No translation unit OUTSIDE Console/ may
#   `#include` a Console header, except the entry point and the validation
#   harness.  The pipeline never depends on the console.
#
# Usage:  check_console_includes.sh [repo-root]
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
    echo "check_console_includes.sh: '$ROOT' is not the repository root." >&2
    exit 2
fi

CORE_DIR="$ROOT/src/hacs/Console/Core"
CONSOLE_DIR="$ROOT/src/hacs/Console"

if [ ! -d "$CONSOLE_DIR" ]; then
    echo "check_console_includes: OK -- no Console/ directory in this tree (nothing to check)."
    exit 0
fi

# --- Rule 1: the C++ standard headers ConsoleCore may include --------------
# The C++17 standard library, plus the <c*> C wrappers.  Deliberately a CLOSED
# list: a header that is not here is either non-standard (POSIX, CUDA, a
# vendor's) or new enough to deserve a conscious decision.
STD_HEADERS=(
    algorithm any array atomic bitset cassert ccomplex cctype cerrno cfenv cfloat
    charconv chrono cinttypes ciso646 climits clocale cmath codecvt complex
    condition_variable csetjmp csignal cstdalign cstdarg cstdbool cstddef cstdint
    cstdio cstdlib cstring ctgmath ctime cuchar cwchar cwctype deque exception
    execution filesystem forward_list fstream functional future initializer_list
    iomanip ios iosfwd iostream istream iterator limits list locale map memory
    memory_resource mutex new numeric optional ostream queue random ratio regex
    scoped_allocator set shared_mutex sstream stack stdexcept streambuf string
    string_view strstream system_error thread tuple type_traits typeindex typeinfo
    unordered_map unordered_set utility valarray variant vector
)

is_std_header() {
    local h="$1" s
    for s in "${STD_HEADERS[@]}"; do
        [ "$h" = "$s" ] && return 0
    done
    return 1
}

fails=0
scanned=0

if [ -d "$CORE_DIR" ]; then
    while IFS= read -r file; do
        rel="${file#"$ROOT"/}"
        scanned=$((scanned + 1))
        # <...> includes: must be standard headers.
        while IFS=: read -r lineno text; do
            [ -n "$lineno" ] || continue
            hdr="$(echo "$text" | sed -E 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*<([^>]*)>.*/\1/')"
            if ! is_std_header "$hdr"; then
                echo "FAIL $rel:$lineno: <$hdr> is not a C++ standard header (ConsoleCore is STL-only)"
                fails=$((fails + 1))
            fi
        done < <(grep -nE '^[[:space:]]*#[[:space:]]*include[[:space:]]*<' "$file" 2>/dev/null)
        # "..." includes: must name a file inside Console/Core/.
        while IFS=: read -r lineno text; do
            [ -n "$lineno" ] || continue
            hdr="$(echo "$text" | sed -E 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*)".*/\1/')"
            base="$(basename "$hdr")"
            if [ "$hdr" != "$base" ] || [ ! -f "$CORE_DIR/$base" ]; then
                echo "FAIL $rel:$lineno: \"$hdr\" is not a Console/Core header (ConsoleCore depends on nothing in the tree)"
                fails=$((fails + 1))
            fi
        done < <(grep -nE '^[[:space:]]*#[[:space:]]*include[[:space:]]*"' "$file" 2>/dev/null)
    done < <(find "$CORE_DIR" \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -type f | sort)
fi

# --- Rule 2: who may include a Console header ------------------------------
# Every entry is a repo-relative path (exact match) or a shell glob, with a
# reason.  Nothing may be added without one.
#
#   src/hacs/Console/*     the console itself.
#   src/hacs/main.cpp     the one CLI branch (phase C1;
#                                            C0 adds no branch).
#   *Validation.cpp / *Validation.h          the test harness.
#   src/hacs/Registry/Slot9.cpp
#                                            per-package registration TUs (slot-9
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot10.cpp
#                                            per-package registration TUs (slot-10
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot11.cpp
#                                            per-package registration TUs (slot-11
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot12.cpp
#                                            per-package registration TUs (slot-12
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot13.cpp
#                                            per-package registration TUs (slot-13
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot14.cpp
#                                            per-package registration TUs (slot-14
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot15.cpp
#                                            per-package registration TUs (slot-15
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot16.cpp
#                                            per-package registration TUs (slot-16
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot17.cpp
#                                            per-package registration TUs (slot-17
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot18.cpp
#                                            per-package registration TUs (slot-18
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/Registry/Slot22.cpp
#                                            per-package registration TUs (slot-22
#                                            WP-1); validation headers only.
#   src/hacs/Registry/Slot21.cpp
#                                            per-package registration TUs (slot-21
#                                            WP-1); validation headers only.
#   src/hacs/Registry/Slot20.cpp
#                                            per-package registration TUs (slot-20
#                                            WP-1); validation headers only.
#   src/hacs/Registry/Slot19.cpp
#                                            per-package registration TUs (slot-19
#                                            WP-1); they include validation
#                                            headers only.
#   src/hacs/ValidationRegistry.cpp
#                                            the explicit registration table --
#                                            it includes every validation's
#                                            header, ConsoleC0Validation.h
#                                            among them.
#   src/hacs/HeckeExt/RegisterHeckeExtensionVerbs.cpp
#                                            [OD] slot-3 close-out 1: the
#                                            extension layer registers its OWN
#                                            console verbs, because
#                                            check_heckeext_includes.sh rule 1
#                                            forbids Console/Bindings/ from
#                                            including a HeckeExt/ header.  The
#                                            entry is the .cpp ALONE -- its
#                                            header forward-declares
#                                            console::Registry and includes no
#                                            Console/ header (slot-4 plan
#                                            Sec. 2.3 item 4).  Rule 1
#                                            (ConsoleCore is STL-only) is
#                                            untouched: this TU is compiled into
#                                            hacs, never into the
#                                            library.
#   src/hacs/HeckeExt/RegisterHeckeExtensionVerbs{Detail.h,
#   Sturm,Algebra,Records,Values,Complex,Gallery}.cpp
#                                            S24 CL-8 (CODEAUDIT A10): the same
#                                            TU split by area, same reason; the
#                                            private Detail.h has only these
#                                            seven includers.
#   src/hacs/HeckeExt/RegisterHeckeZRecordVerbs.cpp
#                                            S25 HZ-REC: the verb hecke_structure
#                                            calls the record builder with the
#                                            HeckeExt/ sources (HZ-LOC audit F8);
#                                            same reason as above, the .cpp alone
#                                            (its header forward-declares
#                                            console::Registry).
#   src/hacs/Registry/Slot25.cpp
#                                            S25 NBCONFIRM S3: the package's
#                                            registration fragment (formerly
#                                            Registry/Slot25/NbConfirm.cpp,
#                                            folded into Slot25.cpp at the
#                                            slot-25 close, S25 FOLD); it
#                                            includes validation headers only
#                                            (ConsoleCostModelValidation.h,
#                                            ConsoleCancelValidation.h), like
#                                            Slot9..22.cpp.
ALLOW_CONSOLE=(
    'src/hacs/Console/*'
    'src/hacs/main.cpp'
    '*Validation.cpp'
    '*Validation.h'
    'src/hacs/ValidationRegistry.cpp'
    'src/hacs/Registry/Slot9.cpp'
    'src/hacs/Registry/Slot10.cpp'
    'src/hacs/Registry/Slot11.cpp'
    'src/hacs/Registry/Slot12.cpp'
    'src/hacs/Registry/Slot13.cpp'
    'src/hacs/Registry/Slot14.cpp'
    'src/hacs/Registry/Slot15.cpp'
    'src/hacs/Registry/Slot16.cpp'
    'src/hacs/Registry/Slot17.cpp'
    'src/hacs/Registry/Slot18.cpp'
    'src/hacs/Registry/Slot19.cpp'
    'src/hacs/Registry/Slot20.cpp'
    'src/hacs/Registry/Slot21.cpp'
    'src/hacs/Registry/Slot22.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbs.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsDetail.h'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsSturm.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsAlgebra.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsRecords.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsValues.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsComplex.cpp'
    'src/hacs/HeckeExt/RegisterHeckeExtensionVerbsGallery.cpp'
    'src/hacs/HeckeExt/RegisterHeckeZRecordVerbs.cpp'
    'src/hacs/Registry/Slot25.cpp'
)

is_allowed_console() {
    local path="$1" pat
    for pat in "${ALLOW_CONSOLE[@]}"; do
        # shellcheck disable=SC2053
        case "$path" in $pat) return 0 ;; esac
    done
    return 1
}

pattern_console='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?Console/'

while IFS= read -r file; do
    rel="${file#"$ROOT"/}"
    case "$rel" in src/hacs/Console/*) ;; *) scanned=$((scanned + 1)) ;; esac
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        if is_allowed_console "$rel"; then continue; fi
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//')"
        fails=$((fails + 1))
    done < <(grep -nE "$pattern_console" "$file" 2>/dev/null)
done < <(find "$ROOT/src" \
             \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) -type f | sort)

if [ "$fails" -eq 0 ]; then
    echo "check_console_includes: OK -- $scanned files scanned; ConsoleCore is STL-only and nothing outside the allow-list includes a Console header."
    exit 0
fi
echo "check_console_includes: FAILED -- $fails offending include(s) (of $scanned files scanned)."
echo "  Rule 1: move the dependency into a BINDING (Console/Bindings/, compiled into hacs)."
echo "  Rule 2: or add the file to ALLOW_CONSOLE in $0 WITH A REASON."
exit 1
