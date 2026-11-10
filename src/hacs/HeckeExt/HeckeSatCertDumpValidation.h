// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SAT_CERT_DUMP_VALIDATION_H
#define HECKE_SAT_CERT_DUMP_VALIDATION_H

#include <cstdint>
#include <string>

// (hecke-extensions slot 17, L17-2; owner ruling Lean Q14 of 2026-10-01)
// HeckeSatCertDumpValidation (`fast`): the validation-only, env-interlocked
// CERTIFICATE-DUMP hook for the Lean stable-certificate checker
// lean/GL2/Algorithms/SatCert.lean ([GT] chk:stablecert, mirroring
// [C, cor:stablecert](1)).  Modelled on HeckeTauDumpValidation (L16-4).
//
//   * HECKE_DUMP_SAT_CERT unset or empty: prints one [SKIP] line and returns 0
//     -- a no-op, so `--all` / `--all-full` stay read-only.
//   * HECKE_DUMP_SAT_CERT=<dir>: <dir> must already exist, must be a directory
//     and must NOT lie inside the source tree (OC_SOURCE_DIR); otherwise the
//     run is refused (returns 1).  Over the cell list (default
//     "168:13:2,220:17:2,258:37:2"; HECKE_DUMP_SAT_CERT_CELLS="n:ell:kmax,..."
//     replaces it, a malformed cell refused by position) it runs
//     ringalg::ComputeRingLoewyCell (certify = true, so CertifyStable has
//     passed on every dumped block) and, for every TORSION-FREE block with
//     verdict `stable` and stable_cert `certified`, re-runs the module at
//     K = K_stab (ringalg::RunModule, the same shipped route), localises it
//     (ringalg::Localize), checks the block is (Z/ell^K)^f, and writes one file
//     satcert_n<n>_ell<ell>_b<k>.txt into <dir>: n, ell, K, f, rank_T, the
//     system, the primes S and the f x f matrices of the T_q (q in S) on the
//     block, column convention, entries mod ell^K.  Other blocks are counted as
//     skipped (torsion / reduced_rank1 / not stable), never failed.
//     It prints `EFFECTIVE HeckeSatCertDumpValidation arm=DUMP cells=<k>` and
//     one SAT_CERT_DUMP line per file.
// No record, golden, digest or setting is touched; writes into scratch only.
uint32_t HeckeSatCertDumpValidation();

// (slot-17 WP-8) The grammar of HECKE_DUMP_SAT_CERT_CELLS ("n:ell:kmax,..."): false with `*why` naming the cell by position.
bool HeckeSatCertDumpParseCells(const std::string& spec, std::string* why);

#endif // HECKE_SAT_CERT_DUMP_VALIDATION_H
