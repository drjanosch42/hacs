#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Build [U] (hida_hecke_algebra.tex, slot-14 WP-8) and report the two acceptance numbers the design
# pass is graded on: page count, and undefined references / citations.
#
#   usage: docs/latex/build_hida_hecke_algebra.sh [outdir]
#
# Runs pdflatex, bibtex, pdflatex, pdflatex (so \ref, \eqref, \cite and the ToC
# all settle) and then greps the log.  Exits non-zero on any LaTeX error, any
# undefined reference, or any undefined citation -- the label namespace is part
# of the contract, so a dangling \ref must fail the build, not warn.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC=hida_hecke_algebra
OUT="${1:-$HERE}"
mkdir -p "$OUT"
cd "$HERE" || exit 1

run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" "$DOC.tex" >/dev/null 2>&1; }

run || { echo "FAIL: pdflatex pass 1"; tail -40 "$OUT/$DOC.log"; exit 1; }
# BIBINPUTS: an out-of-tree outdir must still find gl2_refs.bib (the one change beyond DOC).
(cd "$OUT" && BIBINPUTS="$HERE:${BIBINPUTS:-}" bibtex "$DOC" >/dev/null 2>&1)
run || { echo "FAIL: pdflatex pass 2"; tail -40 "$OUT/$DOC.log"; exit 1; }
run || { echo "FAIL: pdflatex pass 3"; tail -40 "$OUT/$DOC.log"; exit 1; }

LOG="$OUT/$DOC.log"
PAGES=$(pdfinfo "$OUT/$DOC.pdf" 2>/dev/null | sed -n 's/^Pages: *//p')
[ -n "$PAGES" ] || PAGES=$(sed -n 's/.*Output written.*(\([0-9][0-9]*\) page.*/\1/p' "$LOG" | tail -1)
UNDEF_REF=$(grep -c 'Reference .* undefined' "$LOG" || true)
UNDEF_CIT=$(grep -c 'Citation .* undefined' "$LOG" || true)
MULTI=$(grep -c 'multiply defined' "$LOG" || true)

echo "pages                : ${PAGES:-?}"
echo "undefined references : $UNDEF_REF"
echo "undefined citations  : $UNDEF_CIT"
echo "multiply-defined     : $MULTI"

[ "$UNDEF_REF" -eq 0 ] && [ "$UNDEF_CIT" -eq 0 ] && [ "$MULTI" -eq 0 ] || {
  echo "FAIL: label namespace is not clean"; exit 1; }
echo "OK: $OUT/$DOC.pdf"
