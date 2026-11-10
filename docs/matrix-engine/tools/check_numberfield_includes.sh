#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# check_numberfield_includes.sh -- the NumberField library's include guard
# (hecke-extensions slot 19, WP-NF; [E] sec:lvnf code map, [C] sec:codemap /
# sec:notimplemented "WP-NF: a seventh static target NumberField/, STL only,
# guarded like HeckeCore"). Modelled on check_console_includes.sh.
#
#   RULE 1 (STL only).  Every library source under
#   src/hacs/NumberField/ (all *.h / *.hpp / *.cpp except the
#   *Validation.* test harness, which is compiled into hacs and not
#   into the library) may `#include <...>` only a C++ standard header and
#   `#include "..."` only a file that lives in NumberField/ itself.  No GMP, no
#   CAS, no POSIX, no CUDA, nothing from ModArithmetic/ or the pipeline (owner
#   ruling O3 and the no-CAS rule of 2026-10-03; HPC / aarch64 portability).
#
#   RULE 2 (the reverse direction).  NumberField sits beside the pipeline, not
#   under it: no file of the lower libraries -- ModArithmetic/ (ModArithmetic,
#   MatrixEngine, EngineCuda), HeckeCore's Hecke/{ExactInt,Predicates,QuadPoint}
#   and ConsoleCore's Console/Core/ -- may include a NumberField header.
#   HeckeExt/, research drivers and the validation harness may (plan WP-NF).
#
# Usage:  check_numberfield_includes.sh [repo-root]
# Exit:   0 = clean, 1 = an offending include, 2 = usage/environment error.
# Output: one "FAIL <path>:<line>: <text>" line per offender, then a summary.
# ---------------------------------------------------------------------------
set -u

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
if [ ! -d "$ROOT/src/hacs" ] || [ ! -d "$ROOT/src/ModArithmetic" ]; then
    echo "check_numberfield_includes.sh: '$ROOT' is not the repository root." >&2
    exit 2
fi

NF_DIR="$ROOT/src/hacs/NumberField"
if [ ! -d "$NF_DIR" ]; then
    echo "check_numberfield_includes: OK -- no NumberField/ directory in this tree (nothing to check)."
    exit 0
fi

# The C++17 standard headers (closed list, as check_console_includes.sh).
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

# --- Rule 1 -------------------------------------------------------------------
while IFS= read -r file; do
    rel="${file#"$ROOT"/}"
    case "$(basename "$file")" in *Validation.cpp|*Validation.h) continue ;; esac
    scanned=$((scanned + 1))
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        hdr="$(echo "$text" | sed -E 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*<([^>]*)>.*/\1/')"
        if ! is_std_header "$hdr"; then
            echo "FAIL $rel:$lineno: <$hdr> is not a C++ standard header (NumberField is STL-only)"
            fails=$((fails + 1))
        fi
    done < <(grep -nE '^[[:space:]]*#[[:space:]]*include[[:space:]]*<' "$file" 2>/dev/null)
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        hdr="$(echo "$text" | sed -E 's/^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*)".*/\1/')"
        base="$(basename "$hdr")"
        if [ "$hdr" != "$base" ] || [ ! -f "$NF_DIR/$base" ]; then
            echo "FAIL $rel:$lineno: \"$hdr\" is not a NumberField header (NumberField depends on nothing in the tree)"
            fails=$((fails + 1))
        fi
    done < <(grep -nE '^[[:space:]]*#[[:space:]]*include[[:space:]]*"' "$file" 2>/dev/null)
done < <(find "$NF_DIR" \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \) -type f | sort)

# --- Rule 2 -------------------------------------------------------------------
pattern_nf='^[[:space:]]*#[[:space:]]*include[[:space:]]*"([^"]*/)?NumberField/'
LOWER=("$ROOT/src/ModArithmetic" "$ROOT/src/Engine" "$ROOT/src/Cuda" "$ROOT/src/hacs/Console/Core")
for f in ExactInt Predicates QuadPoint; do
    for ext in h cpp; do
        [ -f "$ROOT/src/hacs/Hecke/$f.$ext" ] && LOWER+=("$ROOT/src/hacs/Hecke/$f.$ext")
    done
done
while IFS= read -r file; do
    rel="${file#"$ROOT"/}"
    scanned=$((scanned + 1))
    while IFS=: read -r lineno text; do
        [ -n "$lineno" ] || continue
        echo "FAIL $rel:$lineno: $(echo "$text" | sed 's/^[[:space:]]*//') (a lower library may not include NumberField)"
        fails=$((fails + 1))
    done < <(grep -nE "$pattern_nf" "$file" 2>/dev/null)
done < <(find "${LOWER[@]}" \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -type f 2>/dev/null | sort)

if [ "$fails" -eq 0 ]; then
    echo "check_numberfield_includes: OK -- $scanned files scanned; NumberField is STL-only and no lower library includes it."
    exit 0
fi
echo "check_numberfield_includes: FAILED -- $fails offending include(s) (of $scanned files scanned)."
exit 1
