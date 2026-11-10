#!/usr/bin/env python3
"""mkinv2_postbuild.py <repo> -- regenerate docs/implementation/lean/statement_inventory_v2.tsv from inventory v1 and
lean/census.tsv (L17-5; committed by the F3 repairs, which added the suffix-split join of finding 5).  Run after every
lean_census.sh --emit."""
import csv, collections
import sys
R = sys.argv[1].rstrip('/') + '/'
inv = list(csv.DictReader(open(R + 'docs/implementation/lean/statement_inventory_v1.tsv'), delimiter='\t'))
cen = list(csv.DictReader(open(R + 'lean/census.tsv'), delimiter='\t'))
rank = {'planned': 0, 'stated': 1, 'lit_mathlib': 2, 'lit_hyp': 3, 'proved_mod_lit': 4, 'proved': 5}
bylab = collections.defaultdict(list)
for r in cen:
    bylab[r['label']].append(r)
ilabs = collections.Counter(r['label'] for r in inv)
matched = 0
unmatched = []
before = collections.Counter(r['lean_status'] for r in inv)
invlabs = set(ilabs)
for r in inv:
    # F3 finding 5: join a label with its [GT] suffix splits (`<label>-<suffix>`, not themselves inventory labels),
    # so that e.g. prop:bdrychar takes the weakest status over prop:bdrychar and prop:bdrychar-inv
    rows = list(bylab.get(r['label'], []))
    for l2, rs2 in bylab.items():
        if l2.startswith(r['label'] + '-') and l2 not in invlabs:
            rows += rs2
    if not rows:
        continue
    matched += 1
    sts = [x['lean_status'] for x in rows]
    # weakest status over the label's declarations
    r['lean_status'] = min(sts, key=lambda s: rank[s])
    lits = sorted({h for x in rows for h in x['lit_hyps'].split(',') if h not in ('-', '')})
    r['lit_structure'] = ','.join(lits) if lits else r['lit_structure']
extra = [l for l in bylab if l not in ilabs]
# census labels with no inventory row: keep as appended rows (doc '?')
fields = list(inv[0].keys())
out = list(inv)
for l in sorted(extra):
    rows = bylab[l]
    sts = [x['lean_status'] for x in rows]
    kind = 'hyp' if l.startswith('lit:') else 'lean'
    d = {f: '' for f in fields}
    d.update(doc='GT', label=l, kind=kind, section='-', tags='-', cites='-', layer='L1' if l.startswith('lit:') else 'L2',
             lean_status_planned='-', lean_status=min(sts, key=lambda s: rank[s]),
             adjudication='added by L17-5 from census.tsv: label has no [C]/[E]/[U] inventory row (Lean-side or register entry)')
    out.append(d)
w = csv.DictWriter(open(R + 'docs/implementation/lean/statement_inventory_v2.tsv', 'w'), fieldnames=fields, delimiter='\t', lineterminator='\n')
w.writeheader()
w.writerows(out)
after = collections.Counter(r['lean_status'] for r in out)
print('inventory v1 rows', len(inv), 'before', dict(before))
print('census labels', len(bylab), 'matched to inventory rows', matched, 'added rows', len(extra))
print('v2 rows', len(out), 'after', dict(after))
print('extra labels', sorted(extra))
