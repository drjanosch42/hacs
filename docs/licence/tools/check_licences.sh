#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
# ---------------------------------------------------------------------------
# check_licences.sh -- the licence guard (slot-21 WP-LIC; run by LicenceGuardValidation).
#
# Read-only.  Checks, in order:
#   1. the licence files exist (LICENSE, LICENSE.GPL, NOTICE, REUSE.toml, COPYRIGHT.md, lean/LICENSE,
#      docs/LICENSE, docs/licence/THIRD_PARTY.md) and REUSE.toml carries the default annotation path "**";
#   2. every header-able first-party file has an SPDX header or is in docs/licence/pending_tex_headers.txt
#      (add_spdx.py --check; a stale or unknown pending entry fails too);
#   3. every SPDX-License-Identifier in a tracked file is on the allow-list (no GPL-only, AGPL, proprietary,
#      NonCommercial, ... anywhere), and LICENSES/<id>.txt exists for each id in use;
#   4. the vendored paths are untouched: no SPDX header inside lean/Vendor/**, every file of
#      lean/Vendor/AnthropicFLT/MANIFEST.tsv matches its sha256, notebook/vendor/katex/SHA256SUMS verifies;
#   5. no PDF outside docs/latex, docs/talk (our own documents); docs/literature was removed from HEAD in slot 22 WP-LIC2 (history keeps it: THIRD_PARTY.md section 3; was a KNOWN, reported item:
#      docs/licence/THIRD_PARTY.md section 3).
#
# Usage:  check_licences.sh [repo-root]
# Exit:   0 = clean, 1 = findings, 2 = environment error (no git / python3).
# Output: "FAIL <what>" lines, then one LICENCE_GUARD summary line -- read the text, not only the status.
# ---------------------------------------------------------------------------
set -u
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT" || exit 2
command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1 || { echo "LICENCE_GUARD: not a git tree"; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "LICENCE_GUARD: python3 missing"; exit 2; }
fails=0
fail() { echo "FAIL $*"; fails=$((fails + 1)); }

# 1. required files
for f in LICENSE LICENSE.GPL NOTICE REUSE.toml COPYRIGHT.md lean/LICENSE docs/LICENSE docs/licence/THIRD_PARTY.md \
         docs/licence/tools/add_spdx.py docs/licence/pending_tex_headers.txt; do
    [ -f "$f" ] || fail "missing required file: $f"
done
grep -q '^path = "\*\*"' REUSE.toml 2>/dev/null || fail 'REUSE.toml has no default annotation path = "**"'

# 2. headers
spdx_out=$(python3 docs/licence/tools/add_spdx.py --check 2>&1)
spdx_rc=$?
printf '%s\n' "$spdx_out" | grep -v '^ADD_SPDX_CHECK' | head -20 | sed 's/^/FAIL /'
spdx_line=$(printf '%s\n' "$spdx_out" | grep '^ADD_SPDX_CHECK' | tail -1)
spdx_line=${spdx_line% verdict=*}
[ "$spdx_rc" -eq 0 ] || fails=$((fails + 1))

# 3. identifiers: allow-list + LICENSES/<id>.txt.  Only header-position lines count (a comment marker, then the tag).
allowed='LGPL-3.0-only WITH LicenseRef-CUDA-exception|CC-BY-4.0|Apache-2.0|MIT|OFL-1.1|LicenseRef-third-party-literature'
# REUSE-IgnoreStart (a pattern, not this file's licence)
ids=$(git grep -I -h -E '^[[:space:]]*(//|#|%|/\*)[[:space:]]*SPDX-License-Identifier:' -- . \
        ':(exclude)lean/Vendor' ':(exclude)notebook/vendor' 2>/dev/null |
      sed -E 's/^[^:]*SPDX-License-Identifier:[[:space:]]*//; s/[[:space:]]*\*\/[[:space:]]*$//; s/[[:space:]]+$//' | sort | uniq -c)
# REUSE-IgnoreEnd
nbad=0
while read -r cnt id; do
    [ -n "${id:-}" ] || continue
    if ! printf '%s\n' "$id" | grep -q -x -E "$allowed"; then
        fail "SPDX id not on the allow-list: '$id' ($cnt files)"; nbad=$((nbad + 1))
    fi
done <<< "$ids"
nids=$(printf '%s\n' "$ids" | grep -c .)
for id in LGPL-3.0-only LicenseRef-CUDA-exception CC-BY-4.0 Apache-2.0 MIT OFL-1.1; do
    [ -f "LICENSES/$id.txt" ] || fail "LICENSES/$id.txt missing"
done
# The GPL-3.0 text must ship with the LGPL-3.0 (which supplements it): LICENSE.GPL or LICENSES/GPL-3.0-only.txt.  A release
# omits the two LICENSES/ texts no SPDX tag uses (GPL-3.0-only, LicenseRef-third-party-literature; `reuse lint` rejects an
# unused licence text, release audit S9); the development tree keeps them.
[ -f LICENSE.GPL ] || [ -f LICENSES/GPL-3.0-only.txt ] || fail "no GPL-3.0 text (LICENSE.GPL or LICENSES/GPL-3.0-only.txt)"

# 4. vendored paths
n_vendor_spdx=$(git grep -I -l -E 'SPDX-(License-Identifier|FileCopyrightText):' -- lean/Vendor 2>/dev/null | wc -l)
[ "$n_vendor_spdx" -eq 0 ] || fail "SPDX header inside lean/Vendor/** ($n_vendor_spdx files): vendored files must stay byte-identical"
man=lean/Vendor/AnthropicFLT/MANIFEST.tsv
nman=0; nman_bad=0
if [ -f "$man" ]; then
    while IFS=$'\t' read -r mod path sha rest; do
        case "$mod" in '#'*|module) continue;; esac
        nman=$((nman + 1))
        got=$(sha256sum "lean/Vendor/AnthropicFLT/$path" 2>/dev/null | cut -d' ' -f1)
        [ "$got" = "$sha" ] || { fail "vendored file differs from MANIFEST.tsv: $path"; nman_bad=$((nman_bad + 1)); }
    done < "$man"
    [ "$nman" -gt 0 ] || fail "MANIFEST.tsv lists no file"
else
    fail "missing $man"
fi
if [ -f notebook/vendor/katex/SHA256SUMS ]; then
    (cd notebook/vendor/katex && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1) || fail "notebook/vendor/katex/SHA256SUMS does not verify"
else
    fail "missing notebook/vendor/katex/SHA256SUMS"
fi

# 5. PDFs
stray=$(git ls-files '*.pdf' | grep -v -E '^(docs/latex/|docs/talk/)')
[ -z "$stray" ] || fail "tracked PDF outside docs/latex, docs/talk: $(echo "$stray" | head -3 | tr '\n' ' ')"
lit=$(git ls-files 'docs/literature/*.pdf' | wc -l)   # 0 since slot 22 WP-LIC2; kept so a re-added third-party PDF shows up (and fails check 5 anyway)

verdict=OK; [ "$fails" -eq 0 ] || verdict=FAIL
echo "LICENCE_GUARD ${spdx_line#ADD_SPDX_CHECK } spdx_ids_in_use=$nids vendored_files=$nman vendored_mismatch=$nman_bad known_third_party_pdfs=$lit failures=$fails verdict=$verdict"
[ "$fails" -eq 0 ]
