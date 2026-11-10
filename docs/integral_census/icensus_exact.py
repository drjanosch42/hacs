#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
"""icensus_exact.py -- the reference decoder of the faithful char_Z coefficients of `integral_descent_record/v1` (slot-24 FMTMP;
[C] gl2_contract.tex sec:recordschema item 7 (l)).  Python 3 standard library only; read-only; deterministic.

THE ENCODING.  A charpoly entry lists its coefficients c_0 .. c_r (high -> low, c_0 = 1) in `coefficients`: a decimal integer when
|c_i| < 2^63, else the token "[+,D digits]" / "[-,D digits]" (the sign and the number D of mixed-radix digits D3 used; not the value).
Since FMTMP every such token has its exact value in the entry's list `big` (written only when the entry has one):

    "big": [{"i": <index into coefficients>, "hex": "<signed hex>"}, ...]          (ascending i)

<signed hex> is canonical: "-" iff c_i < 0, then "0x", then |c_i| in lowercase hexadecimal without a leading zero; regex
^-?0x(0|[1-9a-f][0-9a-f]*)$, "-0x0" excluded.  Python: int(s, 16).  A cell written before FMTMP has tokens without `big` (the
"old format"): its big coefficients are not recoverable from the file, and every reader here reports it as such, never guesses.

WHAT --check VERIFIES per entry (every coefficient known): each hex text is canonical; its sign is the token's sign and |c| >= 2^63;
|c| < 2^(30 D) (D primes below 2^30, prop:crtcharpolyN); c_0 = 1; and output rule (5)'s digest recomputed from the integers --
FNV-1a-64 over unsigned 64-bit words (8 bytes, least significant first): r, then per i: i, 1 + [c_i < 0], c_i mod 2^64, c_i mod 3^40
-- equals the entry's `digest`.

  icensus_exact.py decode [--check] [--quiet] [--mod Q ...] PATH ...
        PATH: a cell file (integral_descent_cell/v1), a bare record JSON, or a directory (its *.json and cells/*.json, recursively
        under out/*/cells too).  Prints `EXACT p=<p> i=<i> value=<decimal>` per big coefficient (not with --quiet), with --mod
        `RESIDUE N=<N> n=<n> p=<p> q=<Q> poly=<c_0 mod Q>,...` per entry, and one line
        ICENSUS_EXACT files= entries= big= old_format_entries= digest_ok= digest_bad= bad= verdict=OK|FAIL
        (FAIL iff some check failed; an old-format entry is counted, not failed).
  icensus_exact.py selftest
"""
import argparse
import glob
import json
import os
import re
import sys

if hasattr(sys, 'set_int_max_str_digits'):   # Python >= 3.11 limits int <-> decimal str; the values can have > 4300 digits
    sys.set_int_max_str_digits(0)

FNV_OFF = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
MASK64 = (1 << 64) - 1
M3_40 = 3 ** 40
HEX_RE = re.compile(r'^-?0x(0|[1-9a-f][0-9a-f]*)$')
TOK_RE = re.compile(r'^\[([+-]),(\d+) digits\]$')


def decode_hex(s):
    """canonical signed hex -> int; ValueError if not canonical."""
    if not isinstance(s, str) or not HEX_RE.match(s) or s == '-0x0':
        raise ValueError('not a canonical signed hex integer: %r' % (s,))
    return int(s, 16)


def encode_hex(v):
    """int -> canonical signed hex (the inverse of decode_hex)."""
    return ('-' if v < 0 else '') + '0x' + format(abs(v), 'x')


def fnv_u64(h, v):
    v &= MASK64
    for k in range(8):
        h ^= (v >> (8 * k)) & 0xff
        h = (h * FNV_PRIME) & MASK64
    return h


def charpoly_digest(vals):
    """output rule (5)'s digest of a fully certified polynomial c_0 .. c_r (integers)."""
    r = len(vals) - 1
    h = fnv_u64(FNV_OFF, r)
    for i, c in enumerate(vals):
        h = fnv_u64(h, i)
        h = fnv_u64(h, 2 if c < 0 else 1)
        h = fnv_u64(h, c % (1 << 64))
        h = fnv_u64(h, c % M3_40)
    return h


def entry_integers(entry):
    """-> (values, problems, old_format).  values: the integers c_0 .. c_r, or None when some coefficient is not known (undetermined
    '?', or an old-format token without `big`).  problems: a list of strings (empty when every check passes)."""
    toks = entry.get('coefficients', [])
    big = {}
    problems = []
    for b in entry.get('big', []):
        try:
            big[int(b['i'])] = decode_hex(b['hex'])
        except (KeyError, ValueError, TypeError) as e:
            problems.append('p=%s big item %r: %s' % (entry.get('p'), b, e))
    vals, old = [], False
    for i, t in enumerate(toks):
        m = TOK_RE.match(t)
        if m:
            if i not in big:
                old = True
                vals.append(None)
                continue
            v = big.pop(i)
            if (v < 0) != (m.group(1) == '-'):
                problems.append('p=%s i=%d: hex sign disagrees with token %s' % (entry.get('p'), i, t))
            if abs(v) < (1 << 63):
                problems.append('p=%s i=%d: |c| < 2^63 but elided' % (entry.get('p'), i))
            if abs(v) >= 1 << (30 * int(m.group(2))):
                problems.append('p=%s i=%d: |c| >= 2^(30 D), D = %s' % (entry.get('p'), i, m.group(2)))
            vals.append(v)
        elif t == '?':
            vals.append(None)
        else:
            try:
                vals.append(int(t))
            except ValueError:
                problems.append('p=%s i=%d: unparsable token %r' % (entry.get('p'), i, t))
                vals.append(None)
    for i in sorted(big):
        problems.append('p=%s: big item i=%d has no elided token' % (entry.get('p'), i))
    if vals and vals[0] is not None and vals[0] != 1:
        problems.append('p=%s: c_0 = %d, not 1' % (entry.get('p'), vals[0]))
    if any(v is None for v in vals):
        return None, problems, old
    return vals, problems, old


def records_in(path):
    """yield (file, record dict) for a cell file, a record JSON or a directory."""
    if os.path.isdir(path):
        files = set(glob.glob(os.path.join(path, '*.json')))
        files |= set(glob.glob(os.path.join(path, 'cells', '*.json')))
        files |= set(glob.glob(os.path.join(path, 'out', '*', 'cells', '*.json')))
        for f in sorted(files):
            yield from records_in(f)
        return
    with open(path) as fh:
        d = json.load(fh)
    rec = d.get('record', d)
    if isinstance(rec, dict) and 'charpoly' in rec:
        yield path, rec


def cmd_decode(args):
    files = entries = nbig = old = dok = dbad = bad = 0
    for path in args.paths:
        for f, rec in records_in(path):
            files += 1
            for e in rec.get('charpoly', []):
                entries += 1
                vals, problems, is_old = entry_integers(e)
                nbig += len(e.get('big', []))
                old += 1 if is_old else 0
                if args.check:
                    for pr in problems:
                        print('EXACT_BAD file=%s %s' % (f, pr))
                    bad += len(problems)
                    if vals is not None and e.get('determined', False) and not e.get('refused', False):
                        if charpoly_digest(vals) == int(e['digest'], 16):
                            dok += 1
                        else:
                            dbad += 1
                            print('EXACT_BAD file=%s p=%s: recomputed digest 0x%016x != %s' % (f, e.get('p'), charpoly_digest(vals), e['digest']))
                if not args.quiet:
                    for b in e.get('big', []):
                        try:
                            print('EXACT N=%s n=%s p=%s i=%d value=%d' % (rec.get('N'), rec.get('n'), e.get('p'), int(b['i']), decode_hex(b['hex'])))
                        except (KeyError, ValueError, TypeError):
                            pass
                for q in args.mod or []:
                    if vals is not None:
                        print('RESIDUE N=%s n=%s p=%s q=%d poly=%s' % (rec.get('N'), rec.get('n'), e.get('p'), q, ','.join(str(v % q) for v in vals)))
    ok = bad == 0 and dbad == 0
    print('ICENSUS_EXACT files=%d entries=%d big=%d old_format_entries=%d digest_ok=%d digest_bad=%d bad=%d verdict=%s' % (
        files, entries, nbig, old, dok, dbad, bad, 'OK' if ok else 'FAIL'))
    return 0 if ok else 1


def cmd_selftest(_args):
    fails = 0

    def check(cond, what):
        nonlocal fails
        if not cond:
            fails += 1
            print('SELFTEST_FAIL', what)

    for v in (0, 1, -1, 255, -256, 2 ** 63, -(2 ** 63) - 5, 7 ** 300, -(13 ** 1000)):
        check(decode_hex(encode_hex(v)) == v, 'round trip %d' % v)
    for s in ('0x01', '0X1', '0x1F', '-0x0', '+0x1', '1f', '0x', '', ' 0x1'):
        try:
            decode_hex(s)
            check(False, 'accepted %r' % s)
        except ValueError:
            pass
    # the (11, 0) T_2 polynomial of the contract, x^3 + x^2 - 8x - 12, digest of output rule (5)
    vals = [1, 1, -8, -12]
    e = {'p': 2, 'r': 3, 'coefficients': [str(v) for v in vals], 'determined': True, 'digest': '0x%016x' % charpoly_digest(vals)}
    got, pr, old = entry_integers(e)
    check(got == vals and not pr and not old, 'small entry')
    big = [1, -(3 ** 50), 5 ** 40 + 1, -(2 ** 64) - 3]
    toks = ['1', '[-,3 digits]', '[+,4 digits]', '[-,3 digits]']
    e = {'p': 3, 'r': 3, 'coefficients': toks, 'determined': True, 'digest': '0x%016x' % charpoly_digest(big),
         'big': [{'i': i, 'hex': encode_hex(big[i])} for i in (1, 2, 3)]}
    got, pr, old = entry_integers(e)
    check(got == big and not pr and not old, 'big entry %r %r' % (pr, old))
    e2 = dict(e, big=e['big'][:2])
    got, pr, old = entry_integers(e2)
    check(got is None and old, 'old format detected')
    e3 = dict(e, big=[{'i': 1, 'hex': encode_hex(-big[1])}] + e['big'][1:])
    got, pr, old = entry_integers(e3)
    check(any('sign' in x for x in pr), 'sign mismatch caught')
    e4 = dict(e, big=[{'i': 1, 'hex': encode_hex(big[1] + 1)}] + e['big'][1:])
    got, _, _ = entry_integers(e4)
    check(charpoly_digest(got) != int(e['digest'], 16), 'perturbed value moves the digest')
    print('ICENSUS_EXACT_SELFTEST checks_failed=%d verdict=%s' % (fails, 'OK' if fails == 0 else 'FAIL'))
    return 0 if fails == 0 else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('decode')
    d.add_argument('--check', action='store_true')
    d.add_argument('--quiet', action='store_true')
    d.add_argument('--mod', type=int, action='append')
    d.add_argument('paths', nargs='+')
    sub.add_parser('selftest')
    args = ap.parse_args(argv)
    return cmd_decode(args) if args.cmd == 'decode' else cmd_selftest(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
