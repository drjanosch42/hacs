// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LOCALITY_VALIDATION_H
#define HECKE_LOCALITY_VALIDATION_H

#include <cstdint>

// hecke-extensions phase E0a -- validation of the READ-ONLY locality
// certificate hecke::CertifyLocality ([E] Cert. 5.2 / Claims 3.1, 3.3;
// hecke_extensions_implementation_plan.md Sec. 3.1-3.3).  Returns 0 on success.
//
// ARMS (all constructed truths, fixed seeds -- nothing here is an oracle value
// copied out of another validation):
//
//  (1) THE DEFECT, exhibited.  The [E] Claim 3.1 families
//      t_1 = (alpha, alpha^ell), t_2 = (alpha, alpha) over F_{ell^f}, with
//      multiplicities in {1, 2}, with and without nilpotent parts, conjugated
//      by a seeded invertible U.  Each operator has a SINGLE irreducible
//      char-poly factor, so HeckeModuleDecomposition emits ONE block; a FROZEN
//      copy of the PRE-FIX acceptance predicate (kept inside this validation,
//      the BarCocycleSolverValidation pattern) ACCEPTS it, while CertifyLocality
//      REJECTS it -- so this arm proves both that the defect exists and that
//      the certificate catches it.
//
//  (2) GENUINELY LOCAL blocks are accepted: F_{ell^f}[eps]/(eps^2) and
//      F_{ell^f}[x, y]/(x, y)^2, with the expected (pi_w, N, P_p).
//
//  (3) THE [E] CLAIM 3.3 TRAP.  A_S = F_{ell^f}[eps]/(eps^2) presented by the
//      SINGLE operator t = alpha + eps, so the trivial witness w = t is
//      accepted and min_V(w) = pi^2.  Asserts w_hat = alpha (pi_w(w_hat) = 0
//      and w_hat != w), P_t = X mod pi_w, n_t = t - w_hat = eps != 0 and
//      nilpotent, Jordan partition (2) -- and, as the NEGATIVE CONTROL, that
//      the v1.0 recipe t - P_t(w) returns 0, i.e. would have reported a
//      semisimple block.
//
//  (4) The k = 1 ANALYTIC path (which never forms w_hat) equals the general
//      matrix computation, verdict for verdict, on every constructed block.
//
//  (5) THE GRID ARM ([X] Sec. 3.3).  Through GoldenLocalitySweep: every
//      decomposition the golden harness builds -- decomp_A(T_p)^q over the
//      whole 32-bit grid and decomp_sim{2,3,5}^1, ell in {2, 3} and n = 1
//      included -- is recomputed, CertifyLocality is run on every block, and
//      all FOUR golden hashes (_invariants, _certify, _witness, _U) are
//      asserted equal to GoldenReference.inc through a harness lookup,
//      together with local == true on every block.  Prints
//          LOCALITY_GRID blocks=<k> nonlocal=<count> [<cell list>]
//      A non-local cell FAILS this validation and is a finding for the owner:
//      it must not be worked around, because it decides whether E0b's
//      re-baseline can move a decomposition hash.
uint32_t HeckeLocalityValidation();

// (S24 VAL-IMPL, VALSTREAM_audit_1030.md row 9) Arm 7 (c) of the above, the strict-mode GRID IDENTITY: the whole
// golden-grid sweep re-run under `locality = strict` (all four golden decomposition hashes, every block local, linear
// phase).  Slow tier; HeckeLocalityValidation + HeckeLocalitySlowValidation = the pre-split check set.
uint32_t HeckeLocalitySlowValidation();

// hecke-extensions phase E0a -- the (160, 691) probe ([Q17], [X] Sec. 3.4).
// ComputeMultiOp on {T_2, T_3, T_5} at n_sym = 160, ell = 691, with
// CertifyLocality on every block.  RECORDED, NOT ASSERTED (its verdict becomes
// the asserted outcome of E2's HeckeExtSeparationValidation), so it always
// returns 0.  Registered `slow` AND gated on HECKE_LOCALITY_PROBE=1, so it
// runs neither in `--all` nor in `--all-full` nor under the sanitizer
// configurations nor in cross_config.sh -- it is run once, by hand, on an
// optimised binary, for the slot-1 verdict.  Prints
//     LOCALITY_PROBE n=160 ell=691 blocks=... verdict=...
uint32_t HeckeLocalityProbe();

#endif // HECKE_LOCALITY_VALIDATION_H
