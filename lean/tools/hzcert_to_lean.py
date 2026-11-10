#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: Apache-2.0
"""Untrusted generator: HZ-EMIT certificate files -> Lean K3 data modules (slot-25 HZ-EMIT; 07-lean-plan.md §5 "Emission", §5.3).

Reads the files `hzcert_N<N>_n<n>_<checker>_l<ell>.txt` written by `HECKE_DUMP_HZ_CERT=<dir> ./cohomology_app HeckeZCertDumpValidation`
(or by a `structure = hecke` campaign with `hz_cert = on`, <output>/hzcert/) and writes

  * one Mathlib-free Lean module (`import GL2.HeckeZ.Exec.K3`) with a `K3Cert` value and the theorems
    `k3Check cert = true` and `chiCheck cert = true` (`by decide +kernel`) per cell of the T1 SHAPE: level 1, d = 2 (one rational
    cusp system, chi_Z(T_p) = (x - mu_p)^2 (x - lambda_p)), g_P = ell^e a prime power, the at-root read at ell with r = 1;
  * the side manifest rows (07 §11.1 Lean status, outside both record hashes): cell, checker, ell, lean_cert, lean_module,
    dump_sha256, record_fnv1a64, file sha256.  lean_cert is `emitted` for a generated certificate, `absent` (with the reason in
    the `note` column) otherwise; `checked` is written by the caller after a kernel check (--checked).

Everything is recomputed here from the dumps and re-checked in Python before emission (the K3 conditions of 07 §5.3: lambda_p =
1 + p^(k-1), g_P = gcd_p (lambda_p - mu_p) = ell^e, every entry of c(T_p) = T_p - mu_p divisible by ell^v, the witness of valuation
exactly v < K, ell^(v+e) | lambda_p - mu_p); the kernel re-checks all of it.  The trusted inputs are the dumped matrix (TB-mat) and
the descent's chi_Z (TB-chi, via the K2 file's f for each unit T_p), named in the module docstring with their sha256s.

Usage: hzcert_to_lean.py <dump_dir> <out_dir> <ModuleName> [--manifest <path>] [--checked]
  (writes <out_dir>/<ModuleName>.lean; the module's namespace is GL2.HeckeZ.HZGen.<ModuleName>)
"""
import hashlib
import math
import os
import re
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def hexint(s):
    s = s.strip()
    neg = s.startswith('-')
    v = int(s[1:] if neg else s, 16)
    return -v if neg else v


def parse(path):
    kv = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if '\t' in line:
                k, v = line.split('\t', 1)
                kv.setdefault(k, v)
    return kv


def matrix(text):
    return [[int(x) for x in row.split(',')] for row in text.split(';')]


def val(x, ell):
    if x == 0:
        return 10 ** 9
    v = 0
    while x % ell == 0:
        x //= ell
        v += 1
    return v


def cell_k3(dump_dir, stem, k3file):
    """(cert dict, note) for one K3 file; cert None when the T1 shape does not apply."""
    K3 = parse(os.path.join(dump_dir, k3file))
    K2p = os.path.join(dump_dir, stem + '_K2_l0.txt')
    if not os.path.exists(K2p):
        return None, 'no K2 file'
    K2 = parse(K2p)
    N, n = int(K2['N']), int(K2['n'])
    if N != 1:
        return None, 'shape: level N > 1 (K3Cert is the level-1 two-system case)'
    if int(K2['G1.d']) != 2:
        return None, 'shape: d != 2 (K3Cert is the two-system rational case, r = 1)'
    k = n + 2
    ell, K = int(K3['ell']), int(K3['K'])
    P = [int(p) for p in K2['G1.P'].split(',')]
    # TB-chi: the unit theta = T_p of K2 with its exact f = (x - lambda_p)(x - mu_p)
    sys_ = []
    for i, p in enumerate(P):
        key = 'G9.K2.theta=(' + ' '.join('1' if j == i else '0' for j in range(len(P))) + ').f'
        if key not in K2:
            return None, 'no primitive unit theta for T_%d' % p
        f = [hexint(c) for c in K2[key].strip('[]').split(',')]
        lam = 1 + p ** (k - 1)
        if len(f) != 3 or f[2] != 1:
            return None, 'f of T_%d is not monic of degree 2' % p
        mu = -f[1] - lam
        assert f[0] == lam * mu, ('f(lambda) != 0', p)
        sys_.append((p, lam, mu))
    gP = 0
    for (_, lam, mu) in sys_:
        gP = math.gcd(gP, abs(lam - mu))
    e = val(gP, ell)
    if gP != ell ** e or e < 1:
        return None, 'shape: g_P = %d is not a power of ell = %d' % (gP, ell)
    theta = [int(c) for c in K3['theta'].split(',')]
    if sorted(theta) != [0] * (len(theta) - 1) + [1]:
        return None, 'shape: the read operator is a combination, not a unit T_p'
    readP = P[theta.index(1)]
    D = matrix(K3['T_%d' % readP])
    f = len(D)
    m = ell ** K
    mu = [s for s in sys_ if s[0] == readP][0][2]
    C = [[(D[i][j] + (-mu if i == j else 0)) % m for j in range(f)] for i in range(f)]
    v = min(val(x, ell) for row in C for x in row)
    wit = next((i, j) for i in range(f) for j in range(f) if val(C[i][j], ell) == v)
    lam = [s for s in sys_ if s[0] == readP][0][1]
    assert v + 1 <= K and (lam - mu) % (ell ** (v + e)) == 0, 'the at-root read does not close (generator check)'
    assert ('(%d,%d):v=%d' % (wit[0], wit[1], v)) == K3['witness'], ('witness differs from the record', K3['witness'])
    cert = dict(k=k, ell=ell, K=K, f=f, sys=sys_, gP=gP, e=e, readP=readP, dump=D, v=v, wit=wit,
                dump_sha256=K3['dump_sha256'], record=K3['record_fnv1a64'])
    return cert, ''


def lean_cert(name, c):
    sys_ = ', '.join('(%d, %d, %d)' % s for s in c['sys'])
    dump = ', '.join('[' + ', '.join(str(x) for x in row) + ']' for row in c['dump'])
    return ('/-- The K3 certificate of %s (generated). -/\n' % name +
            'def %s : K3Cert where\n' % name +
            '  k := %d\n  ell := %d\n  K := %d\n  f := %d\n  sys := [%s]\n  gP := %d\n  e := %d\n  readP := %d\n'
            '  dump := [%s]\n  v := %d\n  wit := (%d, %d)\n\n' % (c['k'], c['ell'], c['K'], c['f'], sys_, c['gP'], c['e'],
                                                                  c['readP'], dump, c['v'], c['wit'][0], c['wit'][1]) +
            'theorem %s_check : k3Check %s = true := by decide +kernel\n\n' % (name, name) +
            ('theorem %s_chi : chiCheck %s = true := by decide +kernel\n\n' % (name, name) if c['f'] == 3 else ''))


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    dump_dir, out_dir, mod = argv[1], argv[2], argv[3]
    manifest = None
    checked = '--checked' in argv
    if '--manifest' in argv:
        manifest = argv[argv.index('--manifest') + 1]
    files = sorted(os.listdir(dump_dir))
    k3s = [f for f in files if re.match(r'hzcert_N\d+_n\d+_K3_l\d+\.txt$', f)]
    body, rows, srcs = [], [], []
    for f in k3s:
        m = re.match(r'(hzcert_N(\d+)_n(\d+))_K3_l(\d+)\.txt$', f)
        stem, N, n, ell = m.group(1), m.group(2), m.group(3), m.group(4)
        cert, note = cell_k3(dump_dir, stem, f)
        cell = 'N%s_n%s' % (N, n)
        fsha = sha256(os.path.join(dump_dir, f))
        if cert is None:
            rows.append((cell, 'K3', ell, 'absent', '-', '-', parse(os.path.join(dump_dir, f)).get('record_fnv1a64', '-'), fsha, note))
            continue
        name = 'cert_%s_l%s' % (cell, ell)
        body.append(lean_cert(name, cert))
        srcs.append('* `%s`: sha256 `%s` (TB-mat dump sha256 `%s`; record `%s`)' % (f, fsha, cert['dump_sha256'], cert['record']))
        srcs.append('* `%s_K2_l0.txt`: sha256 `%s` (TB-chi: the exact f of each unit T_p)' % (stem, sha256(os.path.join(dump_dir, stem + '_K2_l0.txt'))))
        rows.append((cell, 'K3', ell, 'checked' if checked else 'emitted', 'GL2.HeckeZ.HZGen.' + mod, cert['dump_sha256'], cert['record'], fsha, ''))
    os.makedirs(out_dir, exist_ok=True)
    if body:
        text = ('/-\nSPDX-FileCopyrightText: 2024-2026 Fabian Januszewski\nSPDX-License-Identifier: Apache-2.0\n-/\n'
                'import GL2.HeckeZ.Exec.K3\n\n/-!\n# HeckeZ K3 certificates generated by `lean/tools/hzcert_to_lean.py` (HZ-EMIT) — Mathlib-free\n\n'
                'Untrusted generator output; the kernel re-checks every condition (`k3Check`, and `chiCheck` for `f = 3`).\nSources:\n'
                + '\n'.join(srcs) + '\n-/\n\nnamespace GL2.HeckeZ.HZGen.%s\n\nopen GL2.HeckeZ.Exec\n\n' % mod
                + ''.join(body) + 'end GL2.HeckeZ.HZGen.%s\n' % mod)
        with open(os.path.join(out_dir, mod + '.lean'), 'w') as o:
            o.write(text)
    if manifest:
        head = 'cell\tchecker\tell\tlean_cert\tlean_module\tdump_sha256\trecord_fnv1a64\tfile_sha256\tnote\n'
        with open(manifest, 'w') as o:
            o.write('# schema=hz_lean_manifest/v1 (07-lean-plan.md §11.1: the Lean status per (cell, checker), outside both record hashes)\n')
            o.write(head)
            for r in rows:
                o.write('\t'.join(str(x) for x in r) + '\n')
    for r in rows:
        print('HZCERT_TO_LEAN', ' '.join('%s=%s' % kv for kv in zip(('cell', 'checker', 'ell', 'lean_cert', 'module'), r[:5])),
              ('note=' + r[8].replace(' ', '_')) if r[8] else '')
    print('HZCERT_TO_LEAN_SUMMARY k3_files=%d emitted=%d absent=%d' % (len(k3s), len(body), len(k3s) - len(body)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
