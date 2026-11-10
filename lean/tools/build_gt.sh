#!/usr/bin/env bash
# Build [GT] (web via plasTeX/leanblueprint, PDF via latexmk+xelatex). Outputs are git-ignored:
#   lean/blueprint/web/   lean/blueprint/print/print.pdf
# The leanblueprint venv lives outside the tree: $LEAN_BP_VENV (default ${XDG_DATA_HOME:-$HOME/.local/share}/hacs/leanblueprint-venv).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
venv="${LEAN_BP_VENV:-${XDG_DATA_HOME:-$HOME/.local/share}/hacs/leanblueprint-venv}"
what="${1:-all}"
cd "$here/blueprint/src"
if [ "$what" = all ] || [ "$what" = pdf ]; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../print print.tex >../print.log 2>&1 \
    || { tail -30 ../print.log; echo "GT_BUILD pdf FAIL" >&2; exit 1; }
  echo "GT_BUILD pdf OK $(ls -l ../print/print.pdf | awk '{print $5}') bytes"
fi
if [ "$what" = all ] || [ "$what" = web ]; then
  "$venv/bin/plastex" -c plastex.cfg web.tex >../web.log 2>&1 \
    || { tail -30 ../web.log; echo "GT_BUILD web FAIL" >&2; exit 1; }
  echo "GT_BUILD web OK $(find ../web -type f | wc -l) files"
fi
