// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_GROUPS_VALIDATION_H
#define INTEGRAL_DESCENT_GROUPS_VALIDATION_H

// DESCENT D0 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.2 Sec. 4 "D0", report
// DESCENT0_report.md): the level-1 integral groups by the owner's descent (Integral/Descent/DescentAssembly).
//
// IntegralDescentGroupsValidation (standard, <= 20 s).  Arms:
//   (A) n = 0..20, q = 0..5: the descent == [IB] Appendix A (the frozen O1 table, 126 cells);
//   (T) the independent oracle of record docs/implementation/slot24/descent_oracle_n128.tsv (integral_line_probe.py
//       extended to n <= 128: the tree, exact integer SNF, independent Python), on a subset of n here, every cell in
//       the slow name; and A/T agreement on n <= 20 (the oracle checked against [IB] Appendix A on 90 cells only);
//   (D) the descent == comparator (a) (IntegralGroups on C_I, to degree 4) at n <= 60 (an independent route, shared
//       model family: labelled so);
//   (P) Iso^(2) built to degree 6 at n <= 60: H^4, H^5 read directly == H^2, H^3 and == the production values;
//   (W) the O_2 -> Z/2^64 route (ring2k uint64_t) == the Z/2^30 route at sample n ([C, prop:certifiedeell](6));
//   (C) consistency (not an oracle): universal coefficients against the shipped F_ell pipeline at the Dickson primes,
//       against Iso^(ell) (x) F_ell at 2 and 3, and [C, prop:ellthreeclosed](3) against the shipped Bredon F_3 value;
//   (Z) every zero-count prime 5 <= ell <= n + 1: the shipped F_ell dims == the good-prime dims (no ell-torsion);
//   (B) every Dickson prime's number of cyclic summands == [C, prop:badprimes](1)'s count;
//   (N) negative control: a width below the certified exponent is REFUSED (r_K < rank_Q, nothing read off);
//   (Q) build-time denominators of Iso^(2), Iso^(3) divide q_ell = 3, 4 ([C, prop:certifiedeell]);
//   (c) consistency: the tree comparator (c) and the tree Bredon comparator (c') reproduce the descent's groups.
// IntegralDescentGroupsSlowValidation (slow): the DESCENT_COST lines at n in {100, 200, 300} (HECKE_DESCENT_CELLS
// replaces the list; EFFECTIVE line), the oracle (T) on every cell n <= 128, and the level-N cost / RSS probe
// (HECKE_DESCENT_LEVELN_CELLS=N:n[,N:n...] replaces its default cells; EFFECTIVE line; RSS predicted before each
// cell, a cell above the cap is not started).  Both env hooks are READ-ONLY: they choose cells, never a value.

#include <cstdint>

uint32_t IntegralDescentGroupsValidation();
uint32_t IntegralDescentGroupsSlowValidation();

#endif
