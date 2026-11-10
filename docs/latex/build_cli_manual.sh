#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Build the CLI / console user manual [CLI] (cli_manual.tex) into an output
# directory -- scratch, never the tree -- and report page count and undefined
# references.
#
#   usage: docs/latex/build_cli_manual.sh <outdir>
#
# Runs pdflatex, makeindex, pdflatex, pdflatex from docs/latex/ (the manual
# \input's docs/latex/cli_manual/*.tex and \VerbatimInput's the frozen console
# transcripts by relative path).  Exits non-zero on a LaTeX error, an undefined
# reference or a multiply-defined label.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC=cli_manual
OUT="${1:-}"
[ -n "$OUT" ] || { echo "usage: $0 <outdir>  (scratch, never the tree)"; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$HERE" || exit 1

rm -f "$OUT/$DOC".{aux,toc,idx,ind,ilg,out,log}
run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" "$DOC.tex" >/dev/null 2>&1; }

run || { echo "FAIL: pdflatex pass 1"; tail -40 "$OUT/$DOC.log"; exit 1; }
(cd "$OUT" && makeindex -q "$DOC.idx" >/dev/null 2>&1) || { echo "FAIL: makeindex"; cat "$OUT/$DOC.ilg"; exit 1; }
run || { echo "FAIL: pdflatex pass 2"; tail -40 "$OUT/$DOC.log"; exit 1; }
(cd "$OUT" && makeindex -q "$DOC.idx" >/dev/null 2>&1)
run || { echo "FAIL: pdflatex pass 3"; tail -40 "$OUT/$DOC.log"; exit 1; }

LOG="$OUT/$DOC.log"
PAGES=$(pdfinfo "$OUT/$DOC.pdf" 2>/dev/null | sed -n 's/^Pages: *//p')
UNDEF_REF=$(grep -c 'Reference .* undefined' "$LOG" || true)
MULTI=$(grep -c 'multiply defined' "$LOG" || true)
OVERFULL=$(grep -c '^Overfull \\hbox' "$LOG" || true)

echo "pages                : ${PAGES:-?}"
echo "undefined references : $UNDEF_REF"
echo "multiply-defined     : $MULTI"
echo "overfull hboxes      : $OVERFULL"

[ "$UNDEF_REF" -eq 0 ] && [ "$MULTI" -eq 0 ] || { echo "FAIL: label namespace is not clean"; exit 1; }
echo "OK: $OUT/$DOC.pdf"
