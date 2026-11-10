#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# ---------------------------------------------------------------------------
# label_census.py -- the citation-resolution engine behind label_census.sh.
#
# WHY A SEPARATE PYTHON FILE. The other four grep-level guards
# (raw_mod_calls.sh, check_matrix_includes.sh, check_console_includes.sh,
# check_heckeext_includes.sh) are pure bash: they classify one LINE at a time
# against a fixed pattern. This guard cannot: a `[C, label]` citation carries
# an arbitrary-length, comma-separated, sometimes multi-label, sometimes
# clause-suffixed payload (`[C, prop:cosets(1)]`, `[C, thm:relations,
# cor:ulrecursion, sec:example]`, `[C, obl:oracle, test 9]`, the historical
# `\texttt{[C,~thm:...]}` and the [C, <label>] / [C, \ldots] prose that merely
# DESCRIBES the citation syntax) and every payload has to be resolved against
# the live set of `\label{...}` names in docs/latex/gl2_contract.tex. That is
# a small parser, not a grep pattern, so it is written once here rather than
# fought out in sed. `label_census.sh` is still the thing a validation
# `std::system`s (the `golden_explain.py` precedent already makes python3 a
# tooling dependency of this tree, never of the guarded build itself).
#
# RULE L1 (contract `sec:machinery` item 2): every `[C, <label>]` citation --
# in C++ comments, CMakeLists.txt, docs, READMEs, plans -- and every
# `\HA{<label>}` cross-reference in docs/latex/hecke_extensions.tex (the `\HA`
# macro is *defined as* `[C,~\texttt{#1}]`, i.e. a citation into
# gl2_contract.tex by label; see gl2_preamble.tex and hecke_extensions.tex's
# own local copy) must name a label that a `\label{...}` in gl2_contract.tex
# actually defines. Site count is explicitly NOT gated (
# hecke_extensions_slot3_plan.md WP-10: it grows every slot); only the
# UNRESOLVED count, after the allow-list, must be 0.
#
# TOKENISATION. A `[C, ...]` (or `\HA{...}`) payload is split on top-level
# commas, then each token is normalised:
#   * a trailing "(...)" clause reference is stripped ("thm:xi(2)" -> "thm:xi",
#     "prop:cosets(1)" -> "prop:cosets");
#   * a trailing " test N" annotation is stripped (from payloads like
#     "[C, obl:oracle, test 9]", where "test 9" names a numbered test inside
#     the obligation, not a second label);
#   * "eq. " is normalised to "eq:" (the one historical typo found in
#     hecke_action_implementation_plan.md, a frozen v1.0.0 document);
#   * LaTeX escapes (`\_`) and the non-breaking-space separator (`~`, as in
#     `\texttt{[C,~cl:pd]}`) are unescaped.
# A token that still does not look like an identifier after normalisation
# (contains a space, a backslash, an angle bracket, "..."/"\ldots"/"…", or is
# the macro's own `#1` placeholder or `$\cdot$`) is not a citation at all --
# it is prose ABOUT the citation syntax, or descriptive filler inside a real
# citation's brackets -- and is skipped: not counted as a site, never flagged
# unresolved. See label_census.allow for the exceptions that DO look like an
# identifier: the bare word "label" used as the generic placeholder in
# `[C, label]` (never a real label: gl2_contract.tex has no `\label{label}`).
#
# ALLOW-LIST FORMAT (content-keyed, hecke-extensions slot 4 WP-6). Each entry
# is `relpath|<verbatim fragment of the site's source line>`, preceded by a
# `#`-comment reason. An entry matches every UNRESOLVED citation site in
# `relpath` whose full source line contains the fragment as a substring --
# keyed by the line's own text, not by its line number, so inserting or
# deleting lines above a cited passage cannot silently move the entry off its
# site (which a `relpath:line` entry, WP-10's original format, could). An
# entry that matches zero sites over a run is STALE and fails the census --
# removing a stale entry (because the passage was rewritten, or the citation
# now resolves) is the unit of progress, the same rule raw_mod_calls.allow /
# check_matrix_includes.sh use for their own allow-lists. The retired
# `relpath:line` form is REJECTED with a named AllowListFormatError -- never
# silently skipped or silently reinterpreted -- so a leftover line-number
# entry cannot quietly stop gating anything.
# ---------------------------------------------------------------------------

import argparse
import os
import re
import sys

CITATION_RE = re.compile(r'\[C,[~ ]*([^\]]+)\]')
HA_RE = re.compile(r'\\HA\{([^}]*)\}')
LABEL_DEF_RE = re.compile(r'\\label\{([^}]+)\}')
IDENT_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_:]*$')
# The retired `relpath:line` form, so a leftover entry can be named precisely
# in the rejection message rather than only being told "no '|' found".
OLD_FORM_RE = re.compile(r'^[^|]+:\d+$')

CODE_EXTS = ('.cpp', '.h', '.hpp', '.cc', '.cu')
EXCLUDE_DIR_NAMES = {'.git', 'CMakeFiles', '.claude'}


class AllowListFormatError(RuntimeError):
    """Raised on a malformed label_census.allow entry. Never caught silently:
    main() reports it and the census fails (exit 1) rather than running with
    an allow-list it could not fully parse."""
    pass


def should_descend(dirname):
    return dirname not in EXCLUDE_DIR_NAMES and not dirname.startswith('build')


def find_label_set(root):
    contract = os.path.join(root, 'docs', 'latex', 'gl2_contract.tex')
    with open(contract, encoding='utf-8', errors='replace') as f:
        text = f.read()
    return set(LABEL_DEF_RE.findall(text)), contract


def candidate_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if should_descend(d)]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if rel == 'CMakeLists.txt' or fn.endswith(CODE_EXTS) or fn.endswith('.md'):
                files.append(rel)
            elif rel.startswith('docs' + os.sep + 'latex' + os.sep) and fn.endswith('.tex'):
                files.append(rel)
    return sorted(set(files))


def normalise_token(tok):
    tok = tok.strip()
    tok = re.sub(r'\s+test\s+\d+$', '', tok)
    tok = re.sub(r'\([^()]*\)$', '', tok).strip()
    tok = re.sub(r'^eq\.\s*', 'eq:', tok)
    tok = tok.replace('\\_', '_').replace('\\-', '')
    tok = tok.replace('~', ' ').strip()
    return tok


def load_allow_list(path):
    """Returns a list of {'relpath', 'fragment', 'reason', 'hits'} entries, in
    file order. Non-comment, non-blank lines must be the content-keyed
    `relpath|<verbatim fragment>` form; the reason is the nearest preceding
    '#' comment block (joined). The retired `relpath:line` form raises
    AllowListFormatError -- see the module docstring."""
    entries = []
    if not path or not os.path.isfile(path):
        return entries
    pending_reason = []
    with open(path, encoding='utf-8') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip('\n')
            stripped = line.strip()
            if not stripped:
                pending_reason = []
                continue
            if stripped.startswith('#'):
                pending_reason.append(stripped.lstrip('#').strip())
                continue
            if '|' not in stripped:
                if OLD_FORM_RE.match(stripped):
                    raise AllowListFormatError(
                        f'{path}:{lineno}: entry "{stripped}" uses the retired "relpath:line" '
                        f'form, which is no longer accepted (hecke-extensions slot 4 WP-6: the '
                        f'allow-list is content-keyed). Rewrite it as '
                        f'"relpath|<fragment>" (a verbatim substring of the site\'s source '
                        f'line), or delete the entry if it no longer applies.')
                raise AllowListFormatError(
                    f'{path}:{lineno}: entry "{stripped}" is not in the "relpath|<fragment>" '
                    f'form (no "|" found).')
            relpath, _, fragment = stripped.partition('|')
            relpath = relpath.strip()
            fragment = fragment.strip()
            if not relpath or not fragment:
                raise AllowListFormatError(
                    f'{path}:{lineno}: entry "{stripped}" has an empty relpath or fragment.')
            entries.append({
                'relpath': relpath,
                'fragment': fragment,
                'reason': ' '.join(pending_reason) or '(no reason given)',
                'hits': 0,
            })
            pending_reason = []
    return entries


def read_committed_unresolved(tsv_path):
    """Best-effort: the TOTAL row's 'unresolved' column of a previously
    committed census, read ONLY for the printed comparison line -- never for
    gating (site counts, and therefore every other column, are expected to
    drift release to release)."""
    if not os.path.isfile(tsv_path):
        return None
    try:
        with open(tsv_path, encoding='utf-8') as f:
            header = None
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if header is None:
                    header = parts
                    continue
                if parts and parts[0] == 'TOTAL':
                    idx = header.index('unresolved')
                    return int(parts[idx])
    except Exception:
        return None
    return None


def run_census(root, allow_path):
    labels, contract_path = find_label_set(root)
    allow = load_allow_list(allow_path)  # may raise AllowListFormatError -- not caught here
    allow_by_relpath = {}
    for a in allow:
        allow_by_relpath.setdefault(a['relpath'], []).append(a)

    per_file = {}  # rel -> dict(sites, distinct set, unresolved list, allowed list)
    all_unresolved = []   # (rel, line, raw, token) -- genuinely bad, not allow-listed
    all_allowed = []       # (rel, line, raw, token, reason)
    distinct_all = set()
    total_sites = 0

    def record(rel, lineno, raw, token, line_text):
        nonlocal total_sites
        entry = per_file.setdefault(rel, {'sites': 0, 'labels': set(), 'unresolved': [], 'allowed': []})
        entry['sites'] += 1
        entry['labels'].add(token)
        total_sites += 1
        distinct_all.add(token)
        if token in labels:
            return
        matched = None
        for a in allow_by_relpath.get(rel, []):
            if a['fragment'] in line_text:
                matched = a
                break
        if matched is not None:
            matched['hits'] += 1
            entry['allowed'].append((lineno, token))
            all_allowed.append((rel, lineno, raw, token, matched['reason']))
        else:
            entry['unresolved'].append((lineno, token))
            all_unresolved.append((rel, lineno, raw, token))

    files = candidate_files(root)
    contract_rel = os.path.relpath(contract_path, root)

    for rel in files:
        full = os.path.join(root, rel)
        try:
            with open(full, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except OSError:
            continue
        is_preamble = rel.endswith(os.path.join('docs', 'latex', 'gl2_preamble.tex'))
        for i, line in enumerate(lines, 1):
            if is_preamble and '\\newcommand{\\HA}' in line:
                continue  # the macro DEFINITION, not a use
            if '\\newcommand{\\HA}' in line:
                continue  # hecke_extensions.tex's own local copy of the macro
            for m in CITATION_RE.finditer(line):
                for raw_tok in m.group(1).split(','):
                    tok = normalise_token(raw_tok)
                    if not IDENT_RE.match(tok):
                        continue  # prose about the syntax, or a clause annotation
                    record(rel, i, m.group(0), tok, line)

    # \HA{<label>} is scanned in [E] and, since slot-14 WP-8, in [U] (hida_hecke_algebra.tex), which uses the
    # same macro from gl2_preamble.tex.
    for ha_rel in (os.path.join('docs', 'latex', 'hecke_extensions.tex'),
                   os.path.join('docs', 'latex', 'hida_hecke_algebra.tex')):
        ha_full = os.path.join(root, ha_rel)
        if not os.path.isfile(ha_full):
            continue
        with open(ha_full, encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f, 1):
                if '\\newcommand{\\HA}' in line:
                    continue
                for m in HA_RE.finditer(line):
                    tok = normalise_token(m.group(1))
                    if not IDENT_RE.match(tok):
                        continue
                    record(ha_rel, i, m.group(0), tok, line)

    stale = [a for a in allow if a['hits'] == 0]

    return {
        'labels_defined': len(labels),
        'contract_rel': contract_rel,
        'files': files,
        'per_file': per_file,
        'total_sites': total_sites,
        'distinct_cited': len(distinct_all),
        'unresolved': all_unresolved,
        'allowed': all_allowed,
        'stale': stale,
    }


def write_tsv(result, tsv_path):
    header = ['file', 'sites', 'distinct_labels', 'resolved', 'allowed', 'unresolved', 'unresolved_detail']
    rows = []
    tot_sites = tot_allowed = tot_unresolved = 0
    tot_labels = set()
    for rel in sorted(result['per_file']):
        entry = result['per_file'][rel]
        n_unresolved = len(entry['unresolved'])
        n_allowed = len(entry['allowed'])
        n_resolved = entry['sites'] - n_unresolved - n_allowed
        detail = ';'.join(f'{ln}:{tok}' for ln, tok in entry['unresolved'])
        rows.append([rel, str(entry['sites']), str(len(entry['labels'])),
                     str(n_resolved), str(n_allowed), str(n_unresolved), detail])
        tot_sites += entry['sites']
        tot_allowed += n_allowed
        tot_unresolved += n_unresolved
        tot_labels |= entry['labels']
    rows.append(['TOTAL', str(tot_sites), str(len(tot_labels)),
                 str(tot_sites - tot_allowed - tot_unresolved), str(tot_allowed),
                 str(tot_unresolved), ''])
    os.makedirs(os.path.dirname(tsv_path) or '.', exist_ok=True)
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root')
    ap.add_argument('--tsv', default=None, help='output TSV path (default: <root>/docs/latex/label_census.tsv)')
    ap.add_argument('--allow', default=None, help='allow-list path (default: <root>/docs/matrix-engine/tools/label_census.allow)')
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    tsv_path = args.tsv or os.path.join(root, 'docs', 'latex', 'label_census.tsv')
    allow_path = args.allow or os.path.join(root, 'docs', 'matrix-engine', 'tools', 'label_census.allow')

    committed_unresolved = read_committed_unresolved(tsv_path)

    try:
        result = run_census(root, allow_path)
    except AllowListFormatError as e:
        print(f'ALLOW_LIST_ERROR: {e}')
        print('label_census: FAILED -- the allow-list could not be parsed (see ALLOW_LIST_ERROR '
              'above). Nothing was asserted about citation sites; fix the allow-list entry and '
              're-run.')
        return 1

    write_tsv(result, tsv_path)

    for rel, lineno, raw, tok in result['unresolved']:
        print(f'FAIL {rel}:{lineno}: {raw} -- label "{tok}" has no \\label in {result["contract_rel"]} [rule L1]')

    for rel, lineno, raw, tok, reason in result['allowed']:
        print(f'ALLOWED {rel}:{lineno}: {raw} -- label "{tok}" not in {result["contract_rel"]}, allow-listed: {reason}')

    for a in result['stale']:
        print(f'STALE {a["relpath"]}|{a["fragment"]}: matched no citation site -- '
              f'allow-listed for: {a["reason"]}')

    n_unresolved = len(result['unresolved'])
    n_stale = len(result['stale'])
    print(f'LABEL_CENSUS labels_defined={result["labels_defined"]} files_scanned={len(result["files"])} '
          f'files_with_citations={len(result["per_file"])} sites={result["total_sites"]} '
          f'distinct_labels_cited={result["distinct_cited"]} allowed={len(result["allowed"])} '
          f'unresolved={n_unresolved} stale={n_stale} committed_unresolved={committed_unresolved} []')

    if n_unresolved == 0 and n_stale == 0:
        print(f'label_census: OK -- {result["total_sites"]} citation sites in {len(result["files"])} files, '
              f'0 unresolved (after {len(result["allowed"])} allow-listed).')
        return 0
    if n_unresolved:
        print(f'label_census: FAILED -- {n_unresolved} unresolved citation(s). '
              f'Either fix the citing document, or add the site to {os.path.relpath(allow_path, root)} WITH A REASON.')
    if n_stale:
        print(f'label_census: FAILED -- {n_stale} stale allow-list entr{"y" if n_stale == 1 else "ies"} '
              f'(matched no citation site). Remove the entry -- that is the unit of progress.')
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
