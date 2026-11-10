#!/usr/bin/env bash
# Point lean/.lake/packages at the ONE shared dependency tree. Run once per checkout.
# Never create a per-worktree mathlib: this script refuses to replace a real directory.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# The shared tree: $LEAN_DEPS, default ${XDG_DATA_HOME:-$HOME/.local/share}/hacs/lean-deps (set LEAN_DEPS to keep an older location).
deps="${LEAN_DEPS:-${XDG_DATA_HOME:-$HOME/.local/share}/hacs/lean-deps}"
mkdir -p "$deps" "$here/.lake"
if [ -e "$here/.lake/packages" ] && [ ! -L "$here/.lake/packages" ]; then
  echo "refusing: $here/.lake/packages is a real directory (a per-worktree copy); remove it first" >&2
  exit 1
fi
ln -sfn "$deps" "$here/.lake/packages"
echo "LEAN_DEPS $here/.lake/packages -> $deps"
