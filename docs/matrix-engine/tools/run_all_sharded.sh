#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
#
# run_all_sharded.sh <binary> [N] [--full] [--logs <dir>] -- `--all` (or `--all-full`) as N parallel shards (S24 SHARD,
# VALSTREAM item 5).  Launches `<binary> --all[-full] --shard i/N` for i = 1..N at once, one log per shard
# (<dir>/shard_<i>of<N>.log; default dir ${XDG_CACHE_HOME:-~/.cache}/hacs/sharded/<UTC stamp>, never inside the source
# tree), waits for all, prints each shard's SHARD header and summary line, then ONE merged line
#     SHARDED_ALL shards=N passed=x/y failed=z verdict=OK|FAIL
# and exits with the total failure count.  verdict=OK needs: z = 0, every shard exited with its summary's failure count,
# the shards' planned names (`--shard i/N --list`) add up to the unsharded roster (plain `--list`, filtered by its tier column), each name
# exactly once, and every shard ran exactly its planned number of names (its summary's y).
# Default N = 4: peak RSS is ~4.3 GiB unsharded, < 3 GiB per shard at N = 4 (RELFIX, 2026-11-09).
set -u
BIN=${1:?usage: run_all_sharded.sh <binary> [N] [--full] [--logs <dir>]}
shift
N=4; FULL=0; LOGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --full) FULL=1 ;;
    --logs) LOGS=${2:?--logs needs a directory}; shift ;;
    *[!0-9]*|'') echo "run_all_sharded.sh: bad argument '$1'" >&2; exit 1 ;;
    *) N=$1 ;;
  esac
  shift
done
if [ "$N" -lt 1 ] || [ "$N" -gt 64 ]; then echo "run_all_sharded.sh: N must be in [1, 64]" >&2; exit 1; fi
BIN=$(readlink -f "$BIN"); [ -x "$BIN" ] || { echo "run_all_sharded.sh: $BIN is not executable" >&2; exit 1; }
MODE=--all; [ $FULL = 1 ] && MODE=--all-full
[ -n "$LOGS" ] || LOGS=${XDG_CACHE_HOME:-$HOME/.cache}/hacs/sharded/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$LOGS"; LOGS=$(readlink -f "$LOGS")
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
case "$LOGS/" in "$REPO"/*) echo "REFUSED: $LOGS is inside the source tree $REPO" >&2; exit 1 ;; esac

# The roster this binary runs unsharded, and each shard's planned names (cheap: --list registers, runs nothing).
# The roster is read from plain `--list` (its tier column), independently of the shard code.
if [ $FULL = 1 ]; then "$BIN" --list | cut -f1 | sort > "$LOGS/roster.txt"
else "$BIN" --list | grep -v $'\t\[slow\]$' | cut -f1 | sort > "$LOGS/roster.txt"; fi
: > "$LOGS/planned.txt"
for i in $(seq 1 "$N"); do
  "$BIN" $MODE --shard "$i/$N" --list | grep -v '^SHARD ' | cut -f1 > "$LOGS/planned_${i}of${N}.txt"
  cat "$LOGS/planned_${i}of${N}.txt" >> "$LOGS/planned.txt"
done
sort "$LOGS/planned.txt" > "$LOGS/planned.sorted.txt"
T0=$(date +%s.%N)
pids=()
for i in $(seq 1 "$N"); do
  "$BIN" $MODE --shard "$i/$N" > "$LOGS/shard_${i}of${N}.log" 2>&1 &
  pids+=($!)
done
rcs=()
for i in $(seq 1 "$N"); do
  wait "${pids[$((i-1))]}"; rcs+=($?)
done
T1=$(date +%s.%N)

passed=0; ran=0; failed=0; bad=0
for i in $(seq 1 "$N"); do
  L="$LOGS/shard_${i}of${N}.log"; rc=${rcs[$((i-1))]}
  head_line=$(grep -m1 '^SHARD ' "$L" || echo "SHARD $i/$N (no header)")
  sum_line=$(grep -E '^(All validations PASSED|Some validations FAILED) \(' "$L" | tail -1)
  echo "$head_line"
  echo "  ${sum_line:-(no summary line)}  exit=$rc"
  if [[ "$sum_line" =~ \(([0-9]+)/([0-9]+)\ passed ]]; then
    p=${BASH_REMATCH[1]}; r=${BASH_REMATCH[2]}
    passed=$((passed + p)); ran=$((ran + r)); failed=$((failed + r - p))
    [ "$rc" -eq $((r - p)) ] || { echo "  shard $i: exit $rc differs from its summary's failure count $((r - p))"; bad=1; }
    k=$(wc -l < "$LOGS/planned_${i}of${N}.txt")
    [ "$r" -eq "$k" ] || { echo "  shard $i: ran $r names, planned $k"; bad=1; }
  else
    echo "  shard $i: no summary line (crashed?), exit $rc"; bad=1; failed=$((failed + (rc > 0 ? rc : 1)))
  fi
  grep -E '^\[FAIL\] |^\[SHARD-DEVICE\] ' "$L" | sed 's/^/    /'
done
if ! cmp -s "$LOGS/roster.txt" "$LOGS/planned.sorted.txt"; then
  echo "  the shards' names differ from the roster ($LOGS/roster.txt vs planned.sorted.txt):"
  diff "$LOGS/roster.txt" "$LOGS/planned.sorted.txt" | sed 's/^/    /' | head -20
  bad=1
fi
echo "  union of the shards = roster: $(cmp -s "$LOGS/roster.txt" "$LOGS/planned.sorted.txt" && echo yes || echo NO) ($(wc -l < "$LOGS/planned.sorted.txt") planned, $(wc -l < "$LOGS/roster.txt") in the roster, $(uniq -d "$LOGS/planned.sorted.txt" | wc -l) duplicated)"
verdict=OK; { [ "$failed" -ne 0 ] || [ $bad -ne 0 ]; } && verdict=FAIL
printf 'SHARDED_ALL shards=%d passed=%d/%d failed=%d mode=%s wall_s=%.1f logs=%s verdict=%s\n' \
  "$N" "$passed" "$(wc -l < "$LOGS/roster.txt")" "$failed" "$MODE" "$(echo "$T1 - $T0" | bc)" "$LOGS" "$verdict"
[ $bad -ne 0 ] && [ "$failed" -eq 0 ] && failed=1
exit "$failed"
