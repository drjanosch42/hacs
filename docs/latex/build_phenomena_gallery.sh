#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Build the phenomena gallery [PG] phenomena_gallery.tex (slot-20 S5; not binding)
# and report the acceptance numbers: page count, undefined references / citations, multiply-defined
# labels and overfull boxes.
#
#   usage: docs/latex/build_phenomena_gallery.sh [outdir]
#
# Same shape as build_hecke_examples.sh: pdflatex, bibtex, pdflatex, pdflatex; exits non-zero on
# any LaTeX error, undefined reference / citation, multiply-defined label, or overfull box.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC=phenomena_gallery
OUT="${1:-$HERE}"
mkdir -p "$OUT"
cd "$HERE" || exit 1

# A stale .bbl from an earlier failed run must not poison pass 1 (-halt-on-error).
rm -f "$OUT/$DOC.bbl"

run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" "$DOC.tex" >/dev/null 2>&1; }

run || { echo "FAIL: pdflatex pass 1"; tail -40 "$OUT/$DOC.log"; exit 1; }
(cd "$OUT" && BIBINPUTS="$HERE:${BIBINPUTS:-}" bibtex "$DOC" >/dev/null 2>&1)
run || { echo "FAIL: pdflatex pass 2"; tail -40 "$OUT/$DOC.log"; exit 1; }
run || { echo "FAIL: pdflatex pass 3"; tail -40 "$OUT/$DOC.log"; exit 1; }

LOG="$OUT/$DOC.log"
PAGES=$(pdfinfo "$OUT/$DOC.pdf" 2>/dev/null | sed -n 's/^Pages: *//p')
[ -n "$PAGES" ] || PAGES=$(sed -n 's/.*Output written.*(\([0-9][0-9]*\) page.*/\1/p' "$LOG" | tail -1)
UNDEF_REF=$(grep -c 'Reference .* undefined' "$LOG" || true)
UNDEF_CIT=$(grep -c 'Citation .* undefined' "$LOG" || true)
MULTI=$(grep -c 'multiply defined' "$LOG" || true)
OVERFULL=$(grep -c '^Overfull' "$LOG" || true)

echo "pages                : ${PAGES:-?}"
echo "undefined references : $UNDEF_REF"
echo "undefined citations  : $UNDEF_CIT"
echo "multiply-defined     : $MULTI"
echo "overfull boxes       : $OVERFULL"

[ "$UNDEF_REF" -eq 0 ] && [ "$UNDEF_CIT" -eq 0 ] && [ "$MULTI" -eq 0 ] && [ "$OVERFULL" -eq 0 ] || {
  echo "FAIL: label namespace or layout is not clean"; exit 1; }
echo "OK: $OUT/$DOC.pdf"
