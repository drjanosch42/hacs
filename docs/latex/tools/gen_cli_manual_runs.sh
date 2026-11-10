#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# gen_cli_manual_runs.sh -- the real command-line outputs quoted by the user manual [CLI]
# (docs/latex/cli_manual.tex, macro \runoutput{<name>}): slot-24 DOCS-CLI.
#
#   usage: docs/latex/tools/gen_cli_manual_runs.sh <hacs> [--check]
#
# Runs each command of the list below (small: every run takes seconds and well under 1 GiB) from a scratch
# directory OUTSIDE the tree (default ${XDG_CACHE_HOME:-~/.cache}/hacs/scratch, or $CLI_MANUAL_SCRATCH), and
#   * (default)  writes "$ <command>" plus its stdout and stderr to docs/latex/cli_manual_runs/<name>.txt;
#   * --check    writes nothing in the tree: it compares each fresh output with the committed file after
#                masking every decimal fraction (timings, RSS, rates) and every 0x hash of the binary
#                provenance, and exits 1 on the first other difference.
# The outputs are a function of the binary and of the machine (CPU flags, the GPU); they were produced on the
# reference machine. Spec files the runs read live beside the outputs (docs/latex/cli_manual_runs/*.spec).
# CliManualCoverageValidation checks that every \runoutput{<name>} of the manual has its <name>.txt.
# Exit: 0 written / unchanged, 1 a run failed or (--check) an output changed, 2 usage.
set -u
BIN="${1:-}"
MODE="${2:-}"
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "usage: $0 <hacs> [--check]" >&2; exit 2; }
[ -z "$MODE" ] || [ "$MODE" = "--check" ] || { echo "usage: $0 <hacs> [--check]" >&2; exit 2; }
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # docs/latex
RUNS="$HERE/cli_manual_runs"
PARENT="${CLI_MANUAL_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/hacs/scratch}"
mkdir -p "$PARENT"
TMP="$(mktemp -d "$PARENT/cli_manual_runs.XXXXXX")" || exit 2
trap 'rm -rf "$TMP"' EXIT
cp "$RUNS"/*.spec "$TMP/"
fail=0

# run <name> <expected exit> <env assignments or -> <args...>: the displayed command names the binary hacs.
run() {
    local name="$1" want="$2" envs="$3"; shift 3
    local out="$TMP/$name.txt" shown="hacs"
    for a in "$@"; do
        case "$a" in *" "*|*"'"*|*'"'*|*'('*) shown="$shown '$a'" ;; *) shown="$shown $a" ;; esac
    done
    [ "$envs" = "-" ] || shown="$envs $shown"
    echo "\$ $shown" > "$out"
    if [ "$envs" = "-" ]; then
        ( cd "$TMP" && "$BIN" "$@" ) >> "$out" 2>&1
    else
        # shellcheck disable=SC2086
        ( cd "$TMP" && env $envs "$BIN" "$@" ) >> "$out" 2>&1
    fi
    local rc=$?
    [ "$rc" = "$want" ] || { echo "FAIL $name: exit $rc, expected $want"; tail -5 "$out"; fail=1; }
    echo "[exit $rc]" >> "$out"
}
# show <name> <file relative to the scratch dir> [max lines]: append a file the previous run wrote.
show() {
    local name="$1" f="$2" n="${3:-40}"
    { echo "\$ head -$n $f"; head -"$n" "$TMP/$f"; } >> "$TMP/$name.txt"
}

# --- the integral descent from the command line ([CLI] ch:integral) --------------------------------------------
run descent_cli 0 - -c 'print(descent(1, 12).groups)' --set coefficient_ring=Z
run descent_refused 5 - -c 'print(cohomology(symmetric_power(sl2z_model(), 12, 5)).dims)' --set coefficient_ring=Z

# --- campaigns over Z ([CLI] sec:campaign-z) ---------------------------------------------------------------------
run campaign_z_dry 0 - --campaign descent_small.spec --dry-run
run campaign_z_run 0 - --campaign descent_small.spec --output descent_small_out --quiet
show campaign_z_run descent_small_out/summary.tsv
{ echo "\$ cut -c1-400 descent_small_out/cells/desc_N11_n2_s*.json"; cut -c1-400 "$TMP"/descent_small_out/cells/desc_N11_n2_s*.json; echo; } >> "$TMP/campaign_z_run.txt"

# --- the orbifold census at ell = 2, 3 ([CLI] sec:orbifold) ------------------------------------------------------
run orbifold_dry 0 - --campaign orbifold_small.spec --dry-run
run orbifold_run 0 - --campaign orbifold_small.spec --output orbifold_small_out
{ echo "\$ cut -f1-16 orbifold_small_out/summary.tsv"; cut -f1-16 "$TMP"/orbifold_small_out/summary.tsv; } >> "$TMP/orbifold_run.txt"

# --- benchmarks and engine switches ([CLI] ch:perf) --------------------------------------------------------------
run bench_hecke 0 - --bench hecke 48 134217757 2
run env_kernel 0 "ENGINE_KERNEL=avx2 ENGINE_AVX2_PAIR=0" --bench micro gemm 256
run env_descent_panel 0 "HECKE_DESCENT_SMITH_PANEL=128" -c 'print(descent(1, 30).groups[1])' --set coefficient_ring=Z
run env_descent_gpu 0 "ENGINE_DEVICE=gpu ENGINE_GPU_SMITH2K_MIN=0" -c 'print(descent(1, 30).groups[1])' --set coefficient_ring=Z

mask() { sed -E 's/[0-9]+\.[0-9]+/#.#/g; s/[0-9]+ MB/# MB/g; s/sha256=[0-9a-f]+/sha256=#/g; s/(commit|binary)[=:" ]+[0-9a-f]{7,}/\1=#/g' "$1"; }
for f in "$TMP"/*.txt; do
    b="$(basename "$f")"
    if [ "$MODE" = "--check" ]; then
        if ! diff <(mask "$RUNS/$b") <(mask "$f") > "$TMP/$b.diff" 2>&1; then
            echo "CHANGED docs/latex/cli_manual_runs/$b"; head -20 "$TMP/$b.diff"; fail=1
        fi
    else
        cp "$f" "$RUNS/$b"
    fi
done
if [ "$MODE" = "--check" ]; then
    [ "$fail" = 0 ] && echo "gen_cli_manual_runs.sh: current ($(ls "$TMP"/*.txt | wc -l) outputs)"
else
    echo "gen_cli_manual_runs.sh: wrote $(ls "$TMP"/*.txt | wc -l) outputs to docs/latex/cli_manual_runs/"
fi
exit "$fail"
