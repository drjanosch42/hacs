// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INTEGRAL_CHARPOLY_VALIDATION_H
#define HECKE_INTEGRAL_CHARPOLY_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 8, WP-7 (Line 2, milestone M-0; [IB] =
// docs/implementation/integral_line_design_brief.md Sec. 6.2 M-0, Sec. 5.2, Sec. 3.4).
// `standard` tier. The library is Integral/IntegralCharpolyCrt.{h,cpp}.
//
// (0) O5 transcription checks: for every frozen row, deg P = rank Lambda^1 =
//     2 d_k - 1 (O3, [C, cl:dimH1]) = the probe's frozen H^1 rank; the cusp
//     root obeys Deligne (a_p^2 <= 4 p^{k-1}); P's coefficient bound fits
//     inside the Deligne bound.
// (1) CERTIFIED EQUALITY OVER Z, M-0 item 1: for n in {10, 12, 14, 16, 18, 20},
//     p = 2, and n = 10, p = 3: charpoly(A^{(1)}(T_p)) of the shipped pipeline
//     at good primes ell (descending from 2^30), as many as the product needs to
//     exceed 2^{max(L, L_P) + 1} (Deligne bound L, oracle bound L_P, both as
//     bit lengths), each equal to O5 reduced digit-wise mod ell. Hence
//     char_Z(T_p | Lambda^1) = O5 over Z. Every cell prints its prime list,
//     its bound and its held-out prime.
// (2) One HELD-OUT prime per cell (X2): the next prime of the sequence, not
//     used by the certificate, must agree as well.
// (3) Campaign n (M-0 item 3), recorded, never compared with an oracle:
//     n = 64 always, n = 128 only under HECKE_INTEGRAL_CAMPAIGN=1 (read-only;
//     ~a minute of F_ell runs), p in {2, 3}: the residue vector (one `RES`
//     line per prime) and the certified bound; the oracle-free checks at every
//     prime -- deg = 2 d_k - 1 (O3), (x - 1 - p^{k-1}) divides, the cofactor is
//     a square -- are theorems (Eichler-Shimura) and count as failures.
//     The oracle-free checks also run at every prime of (1) and (2).
// (4) The F_ell path is untouched: nothing in the pipeline changes (this
//     layer only calls it); the standing golden gate is the evidence.
//
// Returns the number of failing checks (0 on success).
uint32_t HeckeIntegralCharpolyValidation();

#endif // HECKE_INTEGRAL_CHARPOLY_VALIDATION_H
