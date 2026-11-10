// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_ISOTROPY_MODEL_TREE_VALIDATION_H
#define INTEGRAL_DESCENT_HECKE_ISOTROPY_MODEL_TREE_VALIDATION_H

// DESCENT milestone D5 (hecke-extensions slot 24; DESCENT_plan.md v0.4 Sec. 4 "D5"): the tree bulk, [C] def:isomodeltree,
// prop:isomodeltree (C-D6).  Arms (standard name):
//   (L1)  level 1, every n of the D0 standard grid: AssembleDescent on the tree route == on the dense route (every H^q, and
//         the 2- and 3-primary pieces q = 1..5), both == descent_oracle_n128.tsv; the tree form over F_ell' (the good prime) has
//         dim H^q == the shipped C_7 Betti numbers, q = 0..3, ranks computed in the arm (D5 audit m3);
//   (LN)  level N, every D1 standard cell (D <= 100): AssembleDescentN tree == dense, both == descentN_oracle.tsv;
//   (E)   the production form TreeLocalForm against the independent chain model TreeModelChain over F_p (p = ell and two primes
//         prime to q_ell), Sym^n and Ind, q <= 7: d^{q+1} G^{-1} d^q = 0, c_q = rank of the chain model's projector, dim H^q
//         equal (D5 audit m3; replaces "explicit d^q == the (2)(d) copy", which ran the same closed-form branch twice);
//   (W)   the W = 61 route on the tree == the K = 30 route (Sym^n at 2);
//   (I)   the two rank sources of (1)(d) agree on the tree (F_ell' reduction of THIS complex == the Betti recursion);
//   (B)   the Bredon tree at ell >= 5 ((2)(c)) == the direct Smith of C_eq (measured, not switched);
//   (M)   the chain-level models of def:isomodeltree: d o d = 0, well defined, the canonical lifts chain maps ((1)(a));
//         Phi^T_ell, Psi^T_ell of (3) chain maps with Phi Psi = id strictly; Phi^Z_T, Psi^Z_T of (4)(c) built by (L) on the
//         code's IsotropyModel: chain maps over r_T, i_T; the tables of (4)(a); H_T, H_X, H_ell, K, K' homotopy identities;
//   (C)   cohomology: Phi^*, Psi^* mutually inverse on H^q mod p (the free tree and the ell-local tree against C_I);
//   (N)   negative controls: a mutated table entry of r_T / G_T is caught;
//   (NT)  planted faults through the PRODUCTION tree route (D5 audit m4): each integral_model::TreeFault (TreeCochains.h,
//         validation-only, ScopedTreeFault) through AssembleDescent, AssembleDescentN, AssemblePairMaps, (E) and (L1); the
//         caught / clean pattern per fault asserted exactly;
//   (G)   D0's guards on the tree route (wrong ring, a field, a wrong rank module / prime, a wrong b_1, a non-prime Bredon ell),
//         the uncertified K = 4 / Z/3^2 pieces, the K = 4 escalation, the tree descent REFUSED as a whole;
//   (RB)  the rollback in --all: D0 / D1 / D2's standard names rerun in-process under ScopedDescentTree(false);
//   (R)   the RSS of every tree piece below its prediction (PredictTreeLocalRss);
//   cost: one DESCENT_COST ... route=tree line per default cell (and route=dense beside it).
// The slow companion runs the tree == dense comparison on every oracle cell (774 at level 1, 204 at level N), the
// chain-level arms to degree 8, and the controlled cost / reach cells (serial pinned pool, HECKE_DESCENT_REPEATS rounds;
// HECKE_DESCENT_TREE_CELLS = "N:n[:both|tree],..." read-only, N = 1 for level 1; EFFECTIVE line).
#include <cstdint>

uint32_t HeckeIsotropyModelTreeValidation();
uint32_t HeckeIsotropyModelTreeSlowValidation();

#endif
