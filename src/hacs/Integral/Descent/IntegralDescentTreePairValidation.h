// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_TREE_PAIR_VALIDATION_H
#define INTEGRAL_DESCENT_TREE_PAIR_VALIDATION_H

// DESCENT milestone D5c (DESCENT_plan.md v0.4 Sec. 4 "D5c"; [C] v1.23.0-pre.34 sec:treepair, clause C-D8, its code map (a)-(g);
// the C-D8 referee's m3, m4, N5, N6 taken into the arms; the D5c code audit's m1, m2, m3 likewise).
//
// IntegralDescentTreePairValidation (standard):
//   (M)  the chain-level certificates of prop:treepair(1)(b), (3)(a)-(c) exactly in Q[Gamma] on the code's Iso, ell = 2, 3 (production)
//        and ell = 5 (the Bredon tree pair, a check): the tables, B_T / B^(ell) complexes, a^(ell) a chain map over f_a, Phi^Z_d, Psi^Z_d
//        chain maps over r_d, i_d (Psi^Z_d in Isob), Phi^d_ell Psi^d_ell = id, the two square homotopies Y_Phi, Y_Psi;
//   (X)  the closed form of prop:treepair(2) (BuildTreePairComplex, the production cochains) = the pullbacks of the chain-level a^(ell)
//        and of the cusp differential, entry by entry, and the tree's J-coordinate d = D3's TreeTransport::differential;
//   (T)  the tree pair = the dense pair (HECKE_DESCENT_TREE_PAIR rollback, scoped) = the oracle d4_pair_oracle.tsv on all 32 cells: every
//        group and map type of eq:descentpairles, q <= 4; the route of the 2- and 3-adic pieces is "tree";
//   (J)  j_Phi, j_Psi (prop:treepair(3)(d)): cochain maps entrywise, the strict coneinc / coneproj blocks, MUTUALLY INVERSE on H^q_c
//        (q <= 4) and Phi_d^*, Psi_d^* on H^q(d), at cells incl. one with 2-torsion in H^{>=3}_c (the five-lemma caveat, the referee);
//   (H)  T_p by transport (ReadTreePairHecke, the production route): on H^q_c, H^q, H^q(d) equal to D4's operator on (Iso, Isob) through
//        H(j_Phi), H(Phi_T^*), H(Phi_d^*) (q <= 4, torsion included), the oracle's HECKE rows on H^1_c, the Eisenstein value on H^0(d);
//        the eta identity and the APPLIED operator's blocks (= T'_T, eta_T, T'_d, the D5c audit's m1) entrywise (D <= 1300), no cone
//        degree skipped at q_max = 4; the D > 1300 mode (generators + kFreivaldsVectors seeded random vectors, the C-D8b referee's
//        Arm (G)) forced at 4 small cells: certified, the same read; and (T) extra_primes = {3} with n odd at (1,1), (1,5): tree = dense
//        (the audit's m3);
//   (Q5) q_max = 5 (the audit's m2): H^q_c, H^q, H^q(d), q <= 5, against D4's BuildConeHecke(.., 6) at (1,10), (11,0) at 2 and (1,10)
//        at 3; the eta identity's top cone degree 4 counted as skipped on the read (TreePairCertCounts::eta_skipped);
//   (O)  a homotopy-type pair operator theta^* = 1 + K^* d + d K^* (K(Isob) in Isob, not a fibre map): T^c a cochain map with eta_T != 0,
//        inducing the identity on H_c; a K leaving Isob is refused (IsobClosure);
//   (N)  negative controls through the production route, exact caught patterns: S for S^{-1} in a (ell = 2 caught; ell = 3 the SAME
//        map, the referee's m3), the S^{-1} e_U v^U term dropped (ell = 2, 3 caught), 1 - T^2 in B, + Y_Phi^* in j_Phi (T_p at (1,4)
//        at 2; (1,4), (1,10), (11,0) at 3), eta_T dropped (theta^* at (1,4), (2,2) at 2 and (1,4), (1,0) at 3; T_p at (1,10) at 2, 3) --
//        planted into the operator only, never into eta_T (the D5c audit's m1), the catching certificate printed; Arm (G) of the C-D8b
//        referee: a d_B-cocycle-valued error shared by the operator and eta_T (ScopedPlantedCocycleEta), clean in the generators-only mode
//        with a changed read, caught with the random vectors and entrywise; every refusal code of
//        prop:treepair(6) that is reachable without a planted lift (NotPrime, PDividesN, DegreeRange x2, RssCeiling x2, Lift[RssCeiling],
//        IsobClosure); detection counts printed;
//   (RB) the rollback keeps value-level coverage in --all: IntegralDescentPairValidation re-run in-process under the dense route.
// IntegralDescentTreePairSlowValidation (slow): the cost of the tree pair (DESCENT5c_COST lines; DESCENT5c_predictions.md, filed before
//   the runs) along a ladder past D4's wall, the dense pair at two cells for the ratio, the eta identity's cost (the referee's N5), the
//   forced generators + random-vectors mode's cost (DESCENT5c_FREIVALDS), the
//   transport against D4 at (23, 2); read-only hook HECKE_DESCENT_TREEPAIR_CELLS=<N>:<n>[,...] replaces the ladder (EFFECTIVE line).

#include <cstdint>

uint32_t IntegralDescentTreePairValidation();
uint32_t IntegralDescentTreePairSlowValidation();

#endif
