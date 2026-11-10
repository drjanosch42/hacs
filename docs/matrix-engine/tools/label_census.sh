#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# label_census.sh -- the fifth grep-level acceptance guard next to
# raw_mod_calls.sh, check_matrix_includes.sh, check_console_includes.sh and
# check_heckeext_includes.sh. Unlike them it needs more than a line-level
# grep pattern (a `[C, label]` citation is a comma-separated, sometimes
# clause-suffixed, sometimes multi-label payload that has to be resolved
# against the live `\label{...}` set of docs/latex/gl2_contract.tex), so the
# parsing lives in the co-located label_census.py; this script is the
# std::system-compatible entry point ContractCensusValidation calls, in
# EXACTLY the shape MatrixRetirementValidation / HeckeExtArchitectureValidation
# already use for their scripts.
#
# RULE L1 ([C] `sec:machinery` item 2): every `[C, <label>]` citation --
# docs, C++ comments, CMakeLists.txt, READMEs, plans -- and every
# `\HA{<label>}` cross-reference in docs/latex/hecke_extensions.tex (`\HA` is
# DEFINED as `[C,~\texttt{#1}]`, gl2_preamble.tex / hecke_extensions.tex's own
# copy) must resolve to a `\label{...}` gl2_contract.tex actually defines.
#
# SITE COUNT IS NOT GATED (it grows every slot -- hecke_extensions_slot3_plan.md
# WP-10). Only the UNRESOLVED count, after label_census.allow, must be 0.
#
# ALLOW-LIST FORMAT (content-keyed since hecke-extensions slot 4 WP-6):
# label_census.allow entries are `relpath|<verbatim fragment of the site's
# source line>`, not `relpath:line` -- so inserting or deleting lines above a
# cited passage cannot silently move an entry off its site. See
# label_census.py's module docstring for the full rule (matching, staleness,
# and the named rejection of the retired `relpath:line` form).
#
# Usage:  label_census.sh [repo-root]
# Exit:   0 = 0 unresolved citations and 0 stale allow-list entries, 1 = at
#         least one of either (or a malformed allow-list entry), 2 =
#         usage/environment error (no python3, or root not found).
# Output: one "FAIL path:line: ..." line per unresolved citation, one
#         "ALLOWED path:line: ..." line per allow-listed one, one
#         "STALE path|fragment: ..." line per allow-list entry that matched
#         nothing, a machine "LABEL_CENSUS ... []" summary line, and the
#         verdict. Regenerates docs/latex/label_census.tsv (one row per
#         citing file, plus TOTAL) unconditionally -- unlike GoldenEmit this
#         is not owner-gated: the census is a report on the tree as it
#         stands, not a frozen baseline.
# ---------------------------------------------------------------------------
set -u

ROOT="${1:-}"
if [ -z "$ROOT" ]; then
    # Default: the repository root, two levels above docs/matrix-engine/tools.
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
if [ ! -d "$ROOT/src/hacs" ] || [ ! -f "$ROOT/docs/latex/gl2_contract.tex" ]; then
    echo "label_census.sh: '$ROOT' is not the repository root." >&2
    exit 2
fi

PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
    echo "label_census.sh: python3 not found on PATH; cannot run the census." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PY" "$SCRIPT_DIR/label_census.py" "$ROOT" \
    --tsv "$ROOT/docs/latex/label_census.tsv" \
    --allow "$SCRIPT_DIR/label_census.allow"
