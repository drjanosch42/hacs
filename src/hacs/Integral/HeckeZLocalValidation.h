// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_LOCAL_VALIDATION_H
#define INTEGRAL_HECKEZ_LOCAL_VALIDATION_H

#include <cstdint>

// HeckeZLocalValidation (standard) -- slot-25 HZ-LOC, the local reads of Integral/HeckeZLocal.h on the named cells of
// slot24/heckez/06-work-packages.md "HZ-LOC" (expected values from 02-toy-cases.md):
//  (A) T1 (1, 10) at 691, K = 2: dim A'_1 = 2 = d ((C3) certified), theta-bar generates, e = 0, both components through the
//      Eisenstein block, the at-root read closes: v_691(exp eta_L) = 1 = v_691(g_P).
//  (B) T2 (1, 22) at 103 and at 2294797 (the u64 regime: 2294797^2 > 2^31): dim A'_1 = 3 = d, e = 0, the reads close.
//  (C) THE u64 ITEM (06 "HZ-LOC" acceptance item; HZ-GOLD k = 112 vs 107, 06 Q3): every certified prime ell >= 5 of g_P at (1, 24),
//      (1, 26), (1, 28), (1, 30), (1, 42) read at K = v + 1; the named primes with ell^2 > 2^31 must close on the u64 regimes.
//  (D) (1, 246) at 37 (the Z/37^2 matrices, local-only: no HZ-RAT at k = 248).
//  (E) T3 (11, 0) at 5 on the EXPORT: eta_L = Z/5, the read closes; (C3) certified at 5.
//  (F) T4 (2, 6), anemic, rank L_S = 2, on the EXPORT: at 17 and at 2 (the tree route, K >= 8).
//  (G) the torsion-prime routing at level 1: the H^1 module is recognised as not L / ell^K, read only as A1_of_computed (no certified
//      status), the EXPORT's free block read instead.
//  (H) refusals: ell^K >= 2^63 -> refused(word_modulus); level N, ell >= 46341, K = 2 -> refused(export_word) (EXPORT O1, the scope
//      limit); the (C3) read at K = 1 there still runs; (C3) at level 1, ell = 3, d >= 4 -> refused(by_theory).
//  (I) the Z/ell^K layer: Howell length = Smith length, membership, on random spans (u32 and u64 moduli); determinism.
// Gate line: HECKEZ_LOCAL cells=<k> mismatched=0 torsion_prime_certified_from_H1=0 verdict=OK
// Returns the number of failing checks.
uint32_t HeckeZLocalValidation();

// HeckeZLocalCensusXcheck (slow) -- on every SATURATED census-4.0 row of the level-1 HeckeZ list
// (docs/implementation/slot25/hzloc/census40_level1_saturated.tsv, from mk_census_rows.py) the local invariants of A'_1 at the block
// (dim_E, length_A, Hilbert function, embedding dimension, Gorenstein, Loewy length) re-read by HZ-LOC's spans on the shipped module
// route equal the census columns (each census row matched to a distinct block). Gate line:
// HECKEZ_LOCAL_XCHECK rows=<r> disagree=0 verdict=OK
uint32_t HeckeZLocalCensusXcheck();

#endif
