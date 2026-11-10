// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_U_ELL_REACH_VALIDATION_H
#define HECKE_U_ELL_REACH_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 3, WP-1 (phase E1b-0): the U_ell exact-integer reach.
//
// Both functions return 0 on success (invariant I9) and are registered in
// ValidationRegistry::RegisterAllValidations().
//
// HeckeUEllReachValidation (STANDARD) -- everything that fits the standard
// tier's budget:
//   (1) QuadPointW, the __int128 escalation tier of [C, thm:backend], against
//       QuadPoint on the shipped vertex positions and on seeded transforms:
//       moebius, cmp_abs_re_half, cmp_norm_one identical wherever int64
//       completes, and correct past int64 where it does not;
//   (2) Gamma_0(N)\Gamma coset tables built and self-checked (self_check() +
//       self_check_upper()) at ell = 3617 (the old limit) and at (p, r) =
//       (71, 2), whose N = 5041 is past it -- the vacuity check of the
//       slot-3 audit's F13: the coset construction alone was NEVER the
//       overflowing site, so this arm is context, not the reach proof;
//   (3) the forced-wide-tier arm at ell in {5, 691} and at (p, r) = (71, 2):
//       the tau-table's chains, its certified bound B and A^{(q)}(U_ell)
//       entrywise equal under ExactTier::ForceWide and under the default;
//   (4) escalation accounting: Auto NEVER escalates below the measured
//       threshold, so every golden cell runs the pre-WP-1 code path;
//   (5) the negative control: under ExactTier::ForceNarrow the tau table at
//       ell = 5003 still throws hecke::HeckeOverflow -- i.e. the wide tier is
//       what buys the reach, not an unrelated change.
//
// HeckeUEllReachSlow (SLOW) -- the arms past ~10 s:
//   (6) the reach proof itself: the tau table at ell = 5003 and ell = 10007,
//       certified, and U_5003 = T_p(5003) at n = 2 computed END TO END with
//       HeckeBadPrimeValidation's oracle rows (well-typedness, rank(U^r)
//       non-increasing and stationary by r = d_q);
//   (7) the coset tables at ell = 43867, built and self-checked (U_43867 end
//       to end is NOT asserted: u_ell = auto refuses it, and
//       HeckeUEllBudgetValidation pins that refusal);
//   (8) the forced-wide arm at ell = 3617;
//   (9) THE REACH LIMIT of certify() clause (e) -- [C] lem:bounds(5) --
//       which is a different object from step 1's: the NARROW tier needs
//       B = max(B_tab, 4 N^2) <= 2^31 - 1, i.e. ell <= 23170. Until slot 24 the
//       table at ell = 43867 BUILT and was REFUSED with code 51; since the
//       slot-24 ELL-W wide tier it is CERTIFIED (wide_tier(), 2 N B <= 2^63 - 1).
//       Under the rollback HECKE_TAU_WIDE=0 the arm asserts code 51 as before;
//  (10) the narrow limit is EXACT: ell = 23167 (the largest prime with
//       4 N^2 <= 2^31 - 1) certifies narrow, ell = 23173 is the first wide
//       table (refused with code 51 under HECKE_TAU_WIDE=0).
uint32_t HeckeUEllReachValidation();
uint32_t HeckeUEllReachSlow();

#endif // HECKE_U_ELL_REACH_VALIDATION_H
