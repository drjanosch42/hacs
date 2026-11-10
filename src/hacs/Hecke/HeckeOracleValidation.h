// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_ORACLE_VALIDATION_H
#define HECKE_ORACLE_VALIDATION_H

#include <cstdint>

// Stage S7V -- INDEPENDENT primary oracle battery for HeckeOperatorOnK
// (Stage S7's fused Theta_g pipeline / [C, alg:assembly]). Written by the
// S7V validation agent directly from docs/latex/hecke_action.tex ([C]) and
// docs/implementation/hecke_action_implementation_plan.md, WITHOUT reading
// HeckeOperatorSmokeValidation.cpp's expected values (only the PUBLIC
// headers HeckeOperatorOnK.h / HeckeCosetTables.h / HeckeTauTable.h /
// EquivariantCohomologyOnK.h / LocalSystemSL2Z.h were consulted for API
// signatures). All expected eigenvalues/char-polys below are re-derived
// from classical Eichler-Shimura / modular-forms theory (dimension formula
// for M_k(SL(2,Z)), the unique level-1 cusp forms of weight 12/16/24, and
// the Eisenstein eigenvalue 1+p^{n+1}) -- see the long derivation comment at
// the top of HeckeOracleValidation.cpp for the full independent re-check.
//
// Gating (binding, per plan amendment A-F4 in Sec 0.1, which OVERRIDES the
// plan's originally-stated "ell > n+2" gate for the two invariant classes
// dimension (O14) and char-poly (O5/O7), because both equal char-0 data
// reduced mod ell with no Tor obstruction for ell >= 5): O1/O2/O3/O4/O5/O14
// below are asserted at every listed ell >= 5 with NO ell > n+2 restriction
// (in particular O5 is asserted at ell in {5,7}, both <= n+2 = 12, and O14's
// n=10/n=14 rows are asserted at ell in {5,7,101}, all <= their own n+2).
// O7/O8 are the plan's own ADDED extensions beyond the contract's worked
// weight-12 example (weight 16/24); their listed ell in {101,10007} already
// satisfy ell > n+2 (16 and 24 respectively), so the "surviving gate" for
// these two plan-added rows is automatically met and no case at ell <= n+2
// is asserted for them (per the task's explicit instruction). O6 (ell=691)
// implements only the HARD part of the Ramanujan-congruence oracle (the
// char-poly (X+24)^3 identity and A != -24*I); the exact Jordan partition
// (2,1) vs (3) is amendment A-F2's job, deferred to Stage S10V.
//
// Pipeline exercised, per (p_hecke, ell, n_sym), exactly as specified by the
// task (no shortcuts, no reading S7's own internal expected values):
//   K = CreateSL2ZFundamentalDomainTriangulation();
//   base L = Sym^n(F_ell^2) via GenerateSL2ZRelationsWithGroupData(K,n,ell);
//   EquivariantCohomologyOnK::compute_cohomology();
//   HeckeCosetTables(p_hecke, 1, ell, n_sym);
//   HeckeTauTable(K, model, cosets, ell, n_sym);
//   HeckeOperatorOnK(&cohom, &cosets, &tau).compute();
//   read get_H_matrices().
//
// A LOCAL, division-free characteristic-polynomial helper is implemented in
// the .cpp (Leibniz/cofactor expansion of det(X*I - A) over F_ell, entries
// represented as low->high coefficient-list polynomials; only uses +, -, *
// of F_ell elements -- NEVER a division, so it is valid for every ell >= 5
// used here, including ell = 5 <= d_q = 5 where Faddeev-LeVerrier's
// division-by-k step would be ill-defined). d_q <= 5 throughout (dim H^1 is
// at most 5, for n=22), so the O(d!) cost of cofactor expansion is trivial.
// This helper does NOT depend on Stage S10 (which has not landed yet).
//
// Oracles implemented (plan Sec 2 numerical-oracles table + the "S7V --
// Primary oracle battery" stage; [C, sec:example]):
//   O1  -- n=0: A^(0)(T_p) = (p+1 mod ell), p in {2,3,5,7}, ell in {5,7,101};
//          plus the structural fact n>0 => dim H^0 = 0 (checked throughout
//          every n>0 case below).
//   O2  -- odd n in {1,3,5}, ell in {5,7} (ell != 2): all H^q = 0 and the
//          pipeline returns 0x0 matrices in every degree.
//   O3  -- n=2, ell=5 (weight 4, no cusp forms): dim H^1=1,
//          A^(1)(T_2)=(4), A^(1)(T_3)=(3).
//   O4  -- n=8, ell in {5,7,101} (weight 10, no cusp forms): dim H^1=1,
//          A^(1)(T_2) = 513 mod ell.
//   O5  -- n=10 (weight 12, S_12 = C*Delta), ell in {5,7,13,101,10007}:
//          dim H^1=3; charpoly(A^(1)(T_2)) = (X+24)^2(X-2049) mod ell
//          (a_2(Delta)=-24, Eisenstein 1+2^11=2049); charpoly(A^(1)(T_3)) =
//          (X-252)^2(X-177148) mod ell (a_3(Delta)=252, Eis 1+3^11=177148).
//   O6  -- n=10, ell=691 (Ramanujan congruence, HARD PART ONLY):
//          charpoly(A^(1)(T_2)) = (X+24)^3 mod 691 (since
//          2049 = -24 + 3*691); HARD: A^(1)(T_2) != -24*I. The generalized
//          Jordan partition (rank(A+24I), expected 1 i.e. partition (2,1))
//          is RECORDED (printed) only, per amendment A-F2 / Stage S10V.
//   O7  -- n=14 (weight 16, S_16 = C*f_16), ell in {101,10007}: dim H^1=3;
//          charpoly(A^(1)(T_2)) = (X-216)^2(X-32769) mod ell (a_2=216,
//          Eisenstein 1+2^15=32769).
//   O8  -- n=22 (weight 24, dim S_24=2, Galois-conjugate pair), ell in
//          {101,10007}: dim H^1=5; charpoly(A^(1)(T_2)) =
//          (X^2-1080X-20468736)^2*(X-8388609) mod ell (cuspidal T_2 trace
//          1080 / norm -20468736 on S_24; Eisenstein 1+2^23=8388609).
//          Char-poly identity ONLY -- irreducibility/block structure of the
//          quadratic factor is Stage S10V's job (Legendre symbol of
//          144169), not asserted here.
//   O14 -- Eichler-Shimura dimension table: dim H^1(Sym^n) = 2*dim S_{n+2}+1
//          for even n, at ell in {5,7,101} for n in
//          {0,2,4,6,8,10,12,14} (n=0 is the H^0=1,H^1=0 special case),
//          PLUS one large-ell anchor ell=134217757 at n=10 (A-F4: no
//          ell > n+2 gate for this invariant class); dim H^2 = 0 throughout
//          (ell >= 5).
//
// No tolerances anywhere (I9): every comparison is an exact identity over
// F_ell. On ANY failure the offending (q, n, ell, p) is printed together
// with the computed matrix, its computed char-poly, and the expected
// char-poly (or expected scalar, for the 1x1 oracles). Returns the number
// of failing cases (0 = success, matching the codebase's "functions
// returning 0" convention, I9).
//
// KNOWN NONZERO RESULT (documented at length in the .cpp file header):
// O2 (odd n => all H^q=0) is EXPECTED TO FAIL, honestly reporting a
// confirmed, PRE-EXISTING gap in the base (K,R) 7-vertex model that
// predates the entire Hecke branch (reproduced bit-for-bit on the
// pre-Hecke merge-base commit 0a931e1 using only the pre-existing
// TestSL2ZCohomologySymmetricPower / EquivariantCohomologyOnK machinery --
// Stage S7's own new code is not involved). Root cause: the model's
// SimplexRelation list never imposes the "-I stabilizes every point of H"
// relation on the "free" (non-elliptic) simplices v0..v3 and dependents --
// only implicitly at the elliptic points v4/v5/v6 -- which is harmless for
// even n (rho_n(-I)=I) but leaves a spurious surviving n+1-dimensional
// piece of H^1 for odd n. This is deliberately kept as a hard assertion
// (not suppressed) per the task's explicit "fail loudly, do not patch S7
// to force a pass" instruction; O1/O3/O4/O5/O6/O7/O8/O14 all pass cleanly,
// demonstrating Stage S7's OWN Hecke-operator code is correct wherever the
// underlying base cohomology is well-defined. See also the Dickson-
// invariant exceptional-locus finding (n === 0 mod ell+1) recorded, not
// asserted, within O4/O14.
uint32_t HeckeOracleValidation();

#endif // HECKE_ORACLE_VALIDATION_H
