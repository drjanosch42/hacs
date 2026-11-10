#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
"""pg_census_check.py -- every census number quoted by the phenomena gallery [PG], derived from the census files.

[PG] (docs/latex/phenomena_gallery.tex, since 1.4.0) quotes census edition 4.0 and its L23 addendum. This script recomputes each
quoted number from the data files of docs/matrix-engine/results/congruence_census/edition3/ and compares it with the value [PG]
prints (the table EXPECT below, which is the gallery's text, and the shape table parsed out of the .tex itself).

Scopes (named in every output line):
  signed  = census 4.0 as signed by the owner 2026-10-27: main (rows40.tsv.gz) + section b1 (rows40_b1.tsv.gz)
  main, b1  the two signed sections separately
  b2      = section b2 (rows40_b2.tsv.gz), 4.0-pre.3, a PARTIAL pre-release addendum, not signed
  l23     = the L23 addendum of 4.0 (README_l23.md; DEEP-1 / L23 tables), a pre-release addendum, not signed
  3.4     = edition 3.4 (rows34.tsv.gz), only to show the old value next to the new one (--old)

Ordinarity (U_ell a unit on the block) is not a census column; it is read from the hecke_ext records (out of tree, under
$HACS_DATA, default ~/.local/share/hacs) as docs/implementation/slot22/ordloewy/ord_join.py does. --ord-cache FILE stores that join
(scratch, outside the tree); without records and cache the ordinary block is skipped and reported as such (never as a pass).

Usage:
  pg_census_check.py [--old] [--ord-cache FILE] [--tex docs/latex/phenomena_gallery.tex]
Exit 0 iff every compared value agrees; last line `PG_CENSUS_CHECK compared=<k> mismatches=<m> skipped=<s> verdict=OK|FAIL`.
Read-only on the tree; deterministic.
"""
import argparse
import collections
import csv
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, '../../..'))
ED3 = os.path.join(TREE, 'docs/matrix-engine/results/congruence_census/edition3')
DATA = os.environ.get('HACS_DATA') or os.path.expanduser('~/.local/share/hacs')
RECORD_ROOTS = [os.path.join(DATA, p) for p in ('s16/wp4/view', 's22/census33/view', 's15/c3/view', 's24/census40/a/view',
                                                's24/census40/b1/view', 's22/ingest_lowl/view', 's22/ingest_l547/view')] + \
               [os.path.join(TREE, 'docs/matrix-engine/results')]

KEEP = ('source_set', 'record_path', 'n', 'ell', 'ideal_id', 'own_group', 'residue_degree', 'reduced', 'loewy_A', 'embdim',
        'gorenstein', 'hilbert', 'module_D', 'module_loewy', 'residual_type', 'residual_reducible', 'hidden_congruence',
        'sat_defect_dim', 'sat_defect_loewy', 'sat_length', 'sat_kmax')
SECTIONS = {'main': '4.0/rows40.tsv.gz', 'b1': '4.0/rows40_b1.tsv.gz', 'b2': '4.0/rows40_b2.tsv.gz', '3.4': '3.4/rows34.tsv.gz'}


def read_rows(rel):
    hdr, out = None, []
    with gzip.open(os.path.join(ED3, rel), 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            if hdr is None:
                hdr = c
                ix = [(k, hdr.index(k)) for k in KEEP if k in hdr]
                continue
            out.append({k: sys.intern(c[i]) for k, i in ix})
    return out


def read_tsv(rel):
    with open(os.path.join(ED3, rel)) as f:
        return list(csv.DictReader((l for l in f if not l.startswith('#')), delimiter='\t'))


def onsets(rows, pred):
    d = collections.defaultdict(list)
    for r in rows:
        if pred(r):
            d[int(r['ell'])].append(int(r['n']))
    return {l: (min(v), len(v)) for l, v in sorted(d.items())}


def fmt(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':')) if not isinstance(v, str) else v


# ------------------------------------------------------------------------------------------------ the row statistics
def heckez_stats():
    """the torsion-by-free verdicts of HeckeZ edition 1 quoted in [PG] sec:pg-nonsplit."""
    path = os.path.join(TREE, 'docs/integral_census/results/hecke_ed1/torsion_points.tsv')
    cells = set()
    lines = 0
    with open(path) as f:
        rows = list(csv.DictReader((l for l in f if not l.startswith('#')), delimiter='\t'))
    for r in rows:
        if r['split'] == 'non_split':
            lines += 1
            m = re.match(r'N(\d+)_n(\d+)_(full|anemic)', r['key'])
            cells.add((int(m.group(1)), int(m.group(2)), int(r['ell'])))
    return {'non_split_lines': lines, 'non_split_cells': sorted(cells)}


def row_stats(R):
    S = {}
    cells = {(int(r['n']), int(r['ell'])) for r in R}
    S['rows'] = len(R)
    S['cells'] = len(cells)
    S['primes'] = len({c[1] for c in cells})
    S['n_max'] = max(c[0] for c in cells)
    nr = [r for r in R if r['reduced'] == 'false']
    S['nonreduced'] = len(nr)
    S['shapes'] = {'%s/%s' % (r['hilbert'], 'gor' if r['gorenstein'] == 'true' else 'nongor'): 0 for r in nr}
    for r in nr:
        S['shapes']['%s/%s' % (r['hilbert'], 'gor' if r['gorenstein'] == 'true' else 'nongor')] += 1
    S['viol_gorenstein_iff_curvilinear'] = sum((r['gorenstein'] == 'true') != (r['embdim'] == '1') for r in nr)
    S['noncurvilinear_loewy_ge3'] = sum(r['embdim'] != '1' and int(r['loewy_A']) >= 3 for r in nr)
    S['max_embdim'] = max(int(r['embdim']) for r in R if r['embdim'].isdigit())
    L3 = [r for r in R if int(r['loewy_A']) >= 3]
    S['loewy_ge3_rows'] = len(L3)
    fam = collections.Counter('%s:L%s:%s' % (r['ell'], r['loewy_A'], r['residual_type'].split('(')[0]) for r in L3)
    S['loewy_ge3_by_family'] = dict(sorted(fam.items()))
    S['loewy_ge3_primes'] = sorted({int(r['ell']) for r in L3})
    S['loewy_ge3_onset'] = onsets(L3, lambda r: True)
    dih = [r for r in L3 if r['residual_type'].startswith('dihedral')]
    S['dihedral_loewy3'] = {l: {'rows': c, 'onset': o} for l, (o, c) in onsets(dih, lambda r: True).items()}
    S['dihedral_loewy3_D'] = sorted({int(r['module_D']) for r in dih})
    S['dihedral_loewy3_module_loewy'] = sorted({r['module_loewy'] for r in dih})
    S['nongorenstein_onset_rows'] = onsets(R, lambda r: r['gorenstein'] == 'false')
    S['embdim3_onset_rows'] = onsets(R, lambda r: r['embdim'].isdigit() and int(r['embdim']) >= 3)
    S['hidden_true'] = dict(sorted(collections.Counter(int(r['ell']) for r in R if r['hidden_congruence'] == 'true').items()))
    S['hidden_true_rows'] = sum(S['hidden_true'].values())
    S['hidden_false_rows'] = sum(r['hidden_congruence'] == 'false' for r in R)
    S['hidden_torsion_block_rows'] = sum(r['hidden_congruence'] == 'not_stable:torsion_block' for r in R)
    S['hidden_not_stable_rows'] = sum(r['hidden_congruence'].startswith('not_stable') for r in R)
    S['hidden_not_stable_ktop_rows'] = sum(r['hidden_congruence'] == 'not_stable:not_stable_at_KTOP' for r in R)
    S['hidden_computed_rows'] = S['hidden_true_rows'] + S['hidden_false_rows']
    S['cell_604_7_id5'] = [(r['module_D'], r['sat_length']) for r in R if (r['n'], r['ell'], r['ideal_id']) == ('604', '7', '5')]
    S['cell_700_7_id5_D'] = [r['module_D'] for r in R if (r['n'], r['ell'], r['ideal_id']) == ('700', '7', '5')]
    sat = [r for r in R if r['sat_length'].isdigit()]
    S['sat_rows'] = len(sat)
    S['sat_ge_prefix_rows'] = sum(r['sat_length'].startswith('>=') for r in R)
    S['sat_torsion_block_rows'] = sum(r['sat_length'] == 'not_computed:torsion_block' for r in R)
    pos = [r for r in sat if int(r['sat_length']) > 0]
    S['sat_pos_rows'] = len(pos)
    S['sat_pos_primes'] = sorted({int(r['ell']) for r in pos})
    S['sat_ge2_primes'] = sorted({int(r['ell']) for r in pos if int(r['sat_length']) >= 2})
    if sat:
        mx = max(sat, key=lambda r: (int(r['sat_length']), -int(r['n'])))
        S['sat_max'] = {'delta': int(mx['sat_length']), 'cell': [int(mx['n']), int(mx['ell'])], 'ideal': int(mx['ideal_id']),
                        'dim_tor': int(mx['sat_defect_dim']), 'loewy_tor': int(mx['sat_defect_loewy'])}
        for r in sat:
            if (r['n'], r['ell'], r['ideal_id']) == ('700', '7', '5'):
                S['sat_700_7_id5'] = {'delta': int(r['sat_length']), 'dim_tor': int(r['sat_defect_dim']),
                                      'loewy_tor': int(r['sat_defect_loewy']), 'kmax': r.get('sat_kmax', '')}
        S['sat_pos_by_prime_ge547'] = [[int(r['n']), int(r['ell']), int(r['ideal_id']), int(r['sat_length'])]
                                       for r in pos if int(r['ell']) >= 547]
        # the second-machine run sat_run2 on the new cells of 3.3: source_set c1a_local
        sub = [r for r in pos if r['source_set'] == 'c1a_local']
        sc = collections.defaultdict(set)
        for r in sub:
            sc[int(r['ell'])].add(int(r['n']))
        S['sat_c1a_pos_rows'] = len(sub)
        S['sat_c1a_pos_cells'] = sum(len(v) for v in sc.values())
        S['sat_c1a_pos_cells_by_prime'] = {l: len(v) for l, v in sorted(sc.items())}
        S['sat_c1a_max_dim_loewy_delta'] = list(max((int(r['sat_defect_dim']), int(r['sat_defect_loewy']), int(r['sat_length']))
                                                    for r in sub)) if sub else None
    return S


# ------------------------------------------------------------------------------------------------ ordinarity (records)
def rank_mod(M, p):
    M = [[x % p for x in row] for row in M]
    r = 0
    for c in range(len(M[0]) if M else 0):
        piv = next((i for i in range(r, len(M)) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [x * inv % p for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [(a - f * b) % p for a, b in zip(M[i], M[r])]
        r += 1
    return r


def ordinarity(rec, iid, ell):
    ide = next((i for i in rec['ideals'] if i['id'] == iid), None)
    if ide is None:
        return 'unknown:no_ideal'
    s = ide.get('split') or {}
    if s.get('ok') and ell in (s.get('primes') or []):
        ev = s['eigenvalues'][s['primes'].index(ell)]
        return 'ordinary' if any(x % ell for x in ev) else 'nonordinary'
    tp = (ide.get('t_p') or {}).get(str(ell))
    if tp:
        return 'ordinary' if rank_mod(tp, ell) == len(tp) else 'nonordinary'
    return 'unknown:' + str((rec.get('u_ell') or {}).get('reason', 'no_u_ell'))


def ord_join(sections, cache):
    if cache and os.path.exists(cache):
        out = {}
        with open(cache) as f:
            for line in f:
                s, rp, iid, o = line.rstrip('\n').split('\t')
                out[(rp, iid)] = o
        return out
    if not any(os.path.isdir(r) for r in RECORD_ROOTS[:-1]):
        return None
    out = {}
    byrec = collections.defaultdict(list)
    for name, R in sections.items():
        for r in R:
            byrec[r['record_path']].append((int(r['ell']), r['ideal_id']))
    for rp in sorted(byrec):
        p = next((os.path.join(root, rp) for root in RECORD_ROOTS if os.path.exists(os.path.join(root, rp))), None)
        rec = json.load(open(p)) if p else None
        for ell, iid in byrec[rp]:
            out[(rp, iid)] = 'unknown:no_record' if rec is None else ordinarity(rec, int(iid), ell)
    if cache:
        if os.path.abspath(cache).startswith(TREE + os.sep):
            sys.exit('pg_census_check: REFUSED: --ord-cache must lie outside the source tree')
        with open(cache, 'w') as f:
            for (rp, iid), o in sorted(out.items()):
                f.write('%s\t%s\t%s\t%s\n' % ('x', rp, iid, o))
    return out


def ord_stats(R, O):
    S = {}
    tag = [O.get((r['record_path'], r['ideal_id']), 'unknown:no_record') for r in R]
    S['rows_unknown'] = dict(sorted(collections.Counter(t for t in tag if t.startswith('unknown')).items()))
    ordr = [r for r, t in zip(R, tag) if t == 'ordinary']
    S['ordinary_rows'] = len(ordr)
    S['ordinary_loewy_ge3'] = [[int(r['n']), int(r['ell']), int(r['ideal_id'])] for r in ordr if int(r['loewy_A']) >= 3]
    l2 = [r for r in ordr if r['loewy_A'] == '2']
    S['ordinary_loewy2_rows'] = len(l2)
    S['ordinary_loewy2_primes'] = sorted({int(r['ell']) for r in l2})
    S['ordinary_loewy2_reducible'] = sum(r['residual_reducible'] == 'true' for r in l2)
    S['ordinary_loewy2_irreducible'] = sum(r['residual_reducible'] != 'true' for r in l2)
    # periodicity: the multiset of (f, loewy_A) over the ordinary rows of (n, ell) against (n + ell - 1, ell), on pairs of cells
    # whose rows all have a known ordinarity
    cell = collections.defaultdict(list)
    known = collections.defaultdict(lambda: True)
    for r, t in zip(R, tag):
        k = (int(r['n']), int(r['ell']))
        if t.startswith('unknown'):
            known[k] = False
        elif t == 'ordinary' and r['own_group'] == 'H1':
            cell[k].append((r['residue_degree'], r['loewy_A']))
        cell.setdefault(k, cell[k])
    per = {}
    cells = set(cell)
    for (n, ell) in sorted(cells):
        if ell < 17:
            continue
        m = (n + ell - 1, ell)
        if m in cells and known[(n, ell)] and known[m] and n >= ell:
            d = per.setdefault(ell, [0, 0])
            d[0] += 1
            d[1] += sorted(cell[(n, ell)]) != sorted(cell[m])
    S['periodic_pairs'] = {l: {'pairs': v[0], 'differing': v[1]} for l, v in sorted(per.items())}
    S['periodic_pairs_range'] = [min(v[0] for v in per.values()), max(v[0] for v in per.values())] if per else None
    S['periodic_differing_total'] = sum(v[1] for v in per.values())
    return S


# ------------------------------------------------------------------------------------------------ the L23 addendum
def beta(n, ell):
    if n <= 0:
        return None
    best, j, m = 0, 0, n
    while True:
        dick = m >= 2 if ell == 2 else (m > 0 and m % 2 == 0 and m != 2)
        if dick:
            best = j + 1
        if m % ell:
            break
        m //= ell
        j += 1
    return best


def v(n, p):
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


def l23_stats():
    S = {}
    sec = read_tsv('4.0/l23_section40_v2.tsv')
    S['l23_cells'] = len(sec)
    T = read_tsv('4.0/deep1_l23_torsion40.tsv')
    side = read_tsv('4.0/deep1_l23_side40.tsv')
    S['deep1_cells'] = len(side)
    for ell in ('2', '3'):
        rec = [r for r in T if r['ell'] == ell and r['mode'] == 'l23_record' and int(r['n']) % 2 == 0 and int(r['n']) >= 4]
        S['torsion_law_l23_%s' % ell] = '%d/%d' % (sum(int(r['max_tors']) == beta(int(r['n']), int(ell)) for r in rec), len(rec))
        # DEEP-1 only (source deep1; the two slot-24 cells t3cells are the table of \S pg-torsexp): distinct even n, and the
        # number of computations (modes r3 and full can both hold an n)
        d, runs = {}, 0
        for r in T:
            if r['ell'] == ell and r['source'].startswith('deep1') and r['mode'] in ('r3', 'full') and int(r['n']) % 2 == 0:
                runs += 1
                d.setdefault(int(r['n']), set()).add(int(r['max_tors']) == beta(int(r['n']), int(ell)))
        S['torsion_law_deep1_%s' % ell] = '%d/%d' % (sum(all(x) for x in d.values()), len(d))
        S['torsion_law_deep1_%s_runs' % ell] = runs
        S['torsion_law_deep1_%s_n' % ell] = [min(d), max(d)]
    H = read_tsv('4.0/deep1_l23_h01_40.tsv')
    for ell in ('2', '3'):
        h = [r for r in H if r['ell'] == ell]
        S['tell_rank_one_%s' % ell] = '%d/%d' % (sum(r['rank_tell'] == '1' and int(r['rank_tell_minus_1']) == int(r['h1']) - 1
                                                    for r in h), len(h))
        S['tell_rank_one_%s_nmax' % ell] = max(int(r['n']) for r in h)
    for r in H:
        if (r['ell'], r['n']) == ('2', '10000'):
            S['h1_10000_2'] = [int(r['h1']), int(r['rank_tell']), int(r['rank_tell_minus_1'])]
    full = {(r['ell'], r['n']): r for r in side if r['mode'] == 'full'}
    S['tell_max_finite_1000'] = [int(full[('2', '1000')]['tell_max_finite']), int(full[('3', '1000')]['tell_max_finite'])]
    mt = {(r['ell'], int(r['n'])): int(r['max_tors']) for r in T}
    S['e1_examples'] = [mt[('2', 8192)], mt[('3', 972)], mt[('3', 486)], mt.get(('3', 1458)), mt.get(('3', 2916))]
    # the constant floors c_3 = min(v_2(n)+2, 7), c_5 = min(v_2(n)+2, 9), c_7 = min(v_2(n)+3, 8) at ell = 2, even n
    bad, tot = 0, 0
    for r in side:
        if r['ell'] != '2' or r['mode'] not in ('r3', 'full') or not r['c_3']:
            continue
        n = int(r['n'])
        if n % 2:
            continue
        tot += 1
        bad += (int(r['c_3']), int(r['c_5']), int(r['c_7'])) != (min(v(n, 2) + 2, 7), min(v(n, 2) + 2, 9), min(v(n, 2) + 3, 8))
    S['floors_ell2'] = '%d/%d' % (tot - bad, tot)
    return S


# ------------------------------------------------------------------------------------------------ what [PG] 1.4.0 prints
EXPECT = {
    # \S pg-nonsplit (HeckeZ edition 1, torsion_points.tsv)
    ('heckez', 'non_split_lines'): 12,
    ('heckez', 'non_split_cells'): [[8, 4, 2], [8, 10, 2], [16, 2, 2], [16, 4, 2], [27, 2, 3], [27, 4, 3]],
    # \S pg-runs and the abstract (signed 4.0 = main + b1)
    ('signed', 'rows'): 448978, ('signed', 'cells'): 23005, ('signed', 'primes'): 46, ('signed', 'n_max'): 1000,
    ('main', 'rows'): 279093, ('main', 'cells'): 13505, ('b1', 'rows'): 169885, ('b1', 'cells'): 9500,
    ('b2', 'rows'): 152760, ('b2', 'cells'): 11911, ('b2', 'primes'): 35,
    # \S pg-census34 (now: census 4.0 at a glance)
    ('signed', 'nonreduced'): 49176,
    ('signed', 'viol_gorenstein_iff_curvilinear'): 0, ('signed', 'noncurvilinear_loewy_ge3'): 10, ('signed', 'max_embdim'): 3,
    ('signed', 'loewy_ge3_rows'): 672, ('signed', 'loewy_ge3_primes'): [23, 31, 37, 59, 67, 139, 547],
    ('signed', 'loewy_ge3_by_family'): {'139:L4:theta^2': 5, '139:L4:theta^3': 4, '139:L4:theta^4': 3, '139:L4:theta^5': 2,
                                        '139:L4:theta^6': 1, '23:L3:dihedral': 286, '31:L3:dihedral': 132, '37:L4:eis': 222,
                                        '547:L3:eis': 2, '59:L3:theta^0': 3, '59:L3:theta^1': 1, '67:L4:eis': 11},
    ('signed', 'loewy_ge3_onset'): {23: [298, 286], 31: [526, 132], 37: [258, 222], 59: [884, 4], 67: [736, 11],
                                    139: [416, 15], 547: [484, 2]},
    ('signed', 'dihedral_loewy3'): {23: {'rows': 286, 'onset': 298}, 31: {'rows': 132, 'onset': 526}},
    ('signed', 'dihedral_loewy3_D'): [8, 12, 16],
    ('signed', 'nongorenstein_onset_rows'): {7: [82, 421], 11: [174, 680], 13: [206, 569], 17: [338, 282], 19: [416, 222],
                                             23: [596, 148], 29: [926, 5], 59: [942, 2]},
    ('signed', 'embdim3_onset_rows'): {7: [142, 119], 11: [354, 114], 13: [496, 61], 17: [852, 9]},
    ('signed', 'hidden_true'): {7: 196, 11: 338, 13: 164, 17: 144, 19: 31, 23: 10},
    ('b2', 'nonreduced'): 825, ('b2', 'loewy_ge3_rows'): 0, ('b2', 'max_embdim'): 1,
    # \S pg-satdef
    ('signed', 'sat_rows'): 62021, ('signed', 'sat_pos_rows'): 8954, ('signed', 'sat_ge_prefix_rows'): 0,
    ('signed', 'sat_torsion_block_rows'): 1593,
    ('signed', 'sat_pos_primes'): [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 547],
    ('signed', 'sat_ge2_primes'): [7, 11, 13, 17, 19, 23],
    ('signed', 'sat_max'): {'cell': [698, 7], 'delta': 190, 'dim_tor': 26, 'ideal': 4, 'loewy_tor': 6},
    ('signed', 'sat_700_7_id5'): {'delta': 182, 'dim_tor': 26, 'kmax': '16', 'loewy_tor': 6},
    ('signed', 'sat_pos_by_prime_ge547'): [[272, 547, 0, 1]],
    ('signed', 'sat_c1a_pos_rows'): 5060, ('signed', 'sat_c1a_pos_cells'): 723,
    ('signed', 'sat_c1a_pos_cells_by_prime'): {17: 219, 19: 195, 23: 158, 29: 75, 31: 55, 37: 15, 41: 4, 43: 2},
    ('signed', 'sat_c1a_max_dim_loewy_delta'): [4, 3, 6],
    # \S pg-l23, \S pg-torsexp (the L23 addendum)
    ('l23', 'l23_cells'): 1052, ('l23', 'deep1_cells'): 12220,
    ('l23', 'torsion_law_l23_2'): '349/349', ('l23', 'torsion_law_l23_3'): '349/349',
    ('l23', 'torsion_law_deep1_2'): '2158/2158', ('l23', 'torsion_law_deep1_3'): '128/128',
    ('l23', 'torsion_law_deep1_2_runs'): 2572, ('l23', 'torsion_law_deep1_3_runs'): 146, ('l23', 'floors_ell2'): '2572/2572',
    ('l23', 'torsion_law_deep1_2_n'): [600, 10000], ('l23', 'torsion_law_deep1_3_n'): [600, 3840],
    ('l23', 'tell_rank_one_2'): '9303/9303', ('l23', 'tell_rank_one_3'): '196/196',
    ('l23', 'tell_rank_one_2_nmax'): 10000, ('l23', 'tell_rank_one_3_nmax'): 7500,
    ('l23', 'h1_10000_2'): [5001, 1, 5000], ('l23', 'tell_max_finite_1000'): [334, 249],
    ('l23', 'e1_examples'): [13, 6, 5, 6, 7],
}
EXPECT_ORD = {
    ('signed', 'rows_unknown'): {'unknown:u_ell_over_budget': 18124},
    ('signed', 'ordinary_rows'): 57728, ('main', 'ordinary_rows'): 27539, ('b1', 'ordinary_rows'): 30189,
    ('signed', 'ordinary_loewy_ge3'): [[484, 547, 1]],
    ('signed', 'ordinary_loewy2_rows'): 240, ('signed', 'ordinary_loewy2_reducible'): 106,
    ('signed', 'ordinary_loewy2_irreducible'): 134,
    ('signed', 'ordinary_loewy2_primes'): [37, 59, 67, 71, 73, 89, 101, 103, 113, 131, 139, 149, 157, 163, 173, 179, 181, 547, 691],
    ('signed', 'periodic_pairs_range'): [302, 484], ('signed', 'periodic_differing_total'): 0,
}


# ------------------------------------------------------------------------------------------------ the printed literals
def _lit(n):
    """How [PG] prints an integer: plain below 10000, with thin spaces (\\,) from 10000 on."""
    t = '{:,}'.format(n).replace(',', '\\,')
    return t if n >= 10000 else str(n)


def _ints(v):
    if isinstance(v, bool):
        return
    if isinstance(v, int):
        yield v
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _ints(x)
    elif isinstance(v, dict):
        for k, x in v.items():
            if isinstance(k, int):
                yield k
            yield from _ints(x)
    elif isinstance(v, str):
        for m in re.finditer(r'\d+', v):
            yield int(m.group(0))


# (scope, key) of EXPECT whose integers are printed in the text (ints >= 10; the small ones are not distinctive)
# plus numbers the text prints that EXPECT does not compare: (literal-bearing value computed from the data).
EXTRA = [  # (description, data value (a function of stats), the literal it must print: None = _lit(value))
    ('b1 (1,1) rows', lambda S: S['b1']['shapes']['[1,1]/gor'], None),
    ('b1 (1,1,1,1) rows', lambda S: S['b1']['shapes']['[1,1,1,1]/gor'], None),
    ('hidden computed rows', lambda S: S['signed']['hidden_computed_rows'], None),
    ('hidden false rows in b1', lambda S: S['b1']['hidden_false_rows'], None),
    ('hidden not_stable rows', lambda S: S['signed']['hidden_not_stable_rows'], None),
    ('hidden torsion-block rows', lambda S: S['signed']['hidden_torsion_block_rows'], None),
    ('hidden not_stable_at_KTOP rows', lambda S: S['signed']['hidden_not_stable_ktop_rows'], None),
    ('b2 nonreduced', lambda S: S['b2']['nonreduced'], None),
    ('(604,7) id 5 D', lambda S: int(S['signed']['cell_604_7_id5'][0][0]), None),
    ('(604,7) id 5 delta', lambda S: int(S['signed']['cell_604_7_id5'][0][1]), None),
    ('(700,7) id 5 D', lambda S: int(S['signed']['cell_700_7_id5_D'][0]), None),
]


def check_literals(tex, stats):
    """Every integer >= 10 in EXPECT / EXPECT_ORD (plus EXTRA) must occur in the .tex exactly as printed."""
    bad = []
    n = 0
    seen = set()
    for (scope, key), want in list(EXPECT.items()) + list(EXPECT_ORD.items()):
        for i in _ints(want):
            if i >= 100 and key != 'embdim3_onset_rows' and (key, i) not in seen:  # embdim3 row counts are not printed
                seen.add((key, i))
                n += 1
                if _lit(i) not in tex:
                    bad.append('%s %s: literal %s not in the .tex' % (scope, key, _lit(i)))
    for name, f, _ in EXTRA:
        try:
            v = f(stats)
        except KeyError:
            continue
        n += 1
        if _lit(v) not in tex:
            bad.append('%s: literal %s not in the .tex' % (name, _lit(v)))
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', action='store_true', help='also print the edition-3.4 values')
    ap.add_argument('--ord-cache', default=None)
    ap.add_argument('--tex', default=os.path.join(TREE, 'docs/latex/phenomena_gallery.tex'))
    ap.add_argument('--no-ord', action='store_true')
    a = ap.parse_args()

    R = {k: read_rows(v) for k, v in SECTIONS.items() if k != '3.4' or a.old}
    R['signed'] = R['main'] + R['b1']
    stats = {k: row_stats(v) for k, v in R.items()}
    stats['l23'] = l23_stats()
    stats['heckez'] = heckez_stats()
    O = None if a.no_ord else ord_join({k: R[k] for k in ('main', 'b1')} if not a.old else
                                       {k: R[k] for k in ('main', 'b1', '3.4')}, a.ord_cache)
    if O is not None:
        for k in ('signed', 'main', 'b1') + (('3.4',) if a.old else ()):
            stats[k].update(ord_stats(R[k], O))

    for scope in ('signed', 'main', 'b1', 'b2', 'l23', 'heckez') + (('3.4',) if a.old else ()):
        for key, val in stats[scope].items():
            print('PG_CENSUS %s %s=%s' % (scope, key, fmt(val)))

    compared = mism = skipped = 0
    for (scope, key), want in list(EXPECT.items()) + list(EXPECT_ORD.items()):
        if (scope, key) in EXPECT_ORD and O is None:
            skipped += 1
            print('PG_CENSUS_SKIP %s %s (no records and no --ord-cache)' % (scope, key))
            continue
        got = json.loads(json.dumps(stats[scope].get(key), sort_keys=True))
        want = json.loads(json.dumps(want, sort_keys=True))
        compared += 1
        if got != want:
            mism += 1
            print('PG_CENSUS_MISMATCH %s %s gallery=%s data=%s' % (scope, key, fmt(want), fmt(got)))

    # the shape table of the .tex (between the markers PG-CENSUS-SHAPES-BEGIN / -END): rows `$A$ & count & $(h)$ & yes|no \\`
    tex = open(a.tex).read()
    m = re.search(r'PG-CENSUS-SHAPES-BEGIN(.*?)PG-CENSUS-SHAPES-END', tex, re.S)
    if not m:
        mism += 1
        print('PG_CENSUS_MISMATCH tex shape table not found (markers PG-CENSUS-SHAPES-BEGIN / -END)')
    else:
        table = {}
        for line in m.group(1).splitlines():
            if line.lstrip().startswith('%'):
                continue
            t = re.search(r'&\s*\$([\d\\,]+)\$\s*&\s*\$\(([\d,]+)\)\$\s*&\s*(yes|no)\s*\\\\', line)
            if t:
                table['[%s]/%s' % (t.group(2), 'gor' if t.group(3) == 'yes' else 'nongor')] = int(t.group(1).replace('\\,', ''))
        compared += 1
        if table != stats['signed']['shapes']:
            mism += 1
            print('PG_CENSUS_MISMATCH tex shape table %s data %s' % (fmt(table), fmt(stats['signed']['shapes'])))
        compared += 1
        if sum(table.values()) != stats['signed']['nonreduced']:
            mism += 1
            print('PG_CENSUS_MISMATCH tex shape table sum %d nonreduced %d' % (sum(table.values()), stats['signed']['nonreduced']))
    nl, bad = check_literals(tex, stats)
    compared += nl
    for b in bad:
        mism += 1
        print('PG_CENSUS_MISMATCH ' + b)
    print('PG_CENSUS_CHECK compared=%d mismatches=%d skipped=%d verdict=%s' % (compared, mism, skipped,
                                                                               'OK' if mism == 0 else 'FAIL'))
    return 0 if mism == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
