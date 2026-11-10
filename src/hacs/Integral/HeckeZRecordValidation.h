// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_RECORD_VALIDATION_H
#define INTEGRAL_HECKEZ_RECORD_VALIDATION_H

#include <cstdint>

// HeckeZRecordValidation (standard; slot-25 HZ-REC): the record integral_hecke_structure/v1 on the five toy cells T1-T5
// (seven records: T4 and T5 in both algebras) against the frozen status table (04-record-census.md Sec. 2 as of the HZ-REC build,
// every deviation explained in docs/implementation/slot25/HZREC_report.md Sec. 3); the scans (no value outside the closed lists, no
// exact value under a bound, the overall-status rule); record_fnv1a64 and cert_sha256 at 1, 4, 16 threads; four tamper arms; the ed.1
// join (descent record hash and char_Z digests); the word mapping of the Z0 merge report Sec. 7 through forged sources (the
// export's K_max -> refused(export_word), the level-1 K = 1 route -> refused(k1_route), the step cap -> open(precision));
// the refusals of item 8 (a).  Gate line:
// HECKEZ_RECORD cells=5 status_table=exact exact_under_bound=0 values_outside_lists=0 threads_equal=yes tampered_detected=4
// ed1_digest_mismatch=0 verdict=OK
uint32_t HeckeZRecordValidation();

// HeckeZCertDumpValidation (slow; slot-25 HZ-EMIT): HECKE_DUMP_HZ_CERT=<dir> (must exist, outside the source tree, else REFUSED,
// exit 1) writes the certificate files of WriteCertFiles for HECKE_DUMP_HZ_CERT_CELLS=<N:n:ell:K,...> (default: T1, T2 at their
// Eisenstein primes) twice and compares them byte for byte.  Without the variable a no-op that prints [SKIP]-style text.
// Gate lines: EFFECTIVE HeckeZCertDumpValidation cells=<k>; HECKEZ_CERT_DUMP cells=<k> files=<f> byte_stable=yes verdict=OK
uint32_t HeckeZCertDumpValidation();

#endif
