// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_HECKE_VALIDATION_H
#define INTEGRAL_DESCENT_HECKE_VALIDATION_H

// DESCENT milestone D3 (slot 24): T_p, p !| N, on H^q(Gamma_0(N); Sym^n Z^2) over Z, as invariants ([C] sec:descenthecke;
// Integral/Descent/DescentHecke.h).
//
// IntegralDescentHeckeValidation (standard):
//   (O)  the oracle d3_hecke_oracle.tsv (group cocycles of Gamma_0(N) over Z, the C-D4 referee's modules; independent of cohomology_app):
//        char_Z(T_p | H^1/tors) equal coefficient by coefficient (every coefficient certified); the l-primary torsion of H^1 and H^2
//        with T_p -- the tree read at 2 and 3, the cokernel read at l >= 5 -- equal in every choice-free invariant of
//        prop:torsionhecke(1) (type, P_j, the layer blocks mod l^{e_j}, ker / coker (T_p - c) for every c, the Fitting types), and no
//        torsion the oracle does not have; the output-rule digest (CharpolyZ::digest) = the digest of the oracle's integers; the record's
//        torsion primes (RecordTorsionPrimes) contain the oracle's and equal them at l >= 5;
//   (S)  the residues against the shipped mod-p Hecke: the production residue (H_{N,p} on the cokernel presentation) = the shipped
//        F_p pipeline (level 1: RunPipelineModEll, the tau-table route; level N: eq:levelcorr on the 7-vertex model, HFrame) = T~ on
//        the free model over F_l' (the new operator, on H^1 of C_I), at several good primes;
//   (X)  the transported operators agree on cohomology (D5's arm "the transported T'_p equals the one on Iso^(2)"): the tree read =
//        the dense Iso^(2) / Iso^(3) read (T'^(l) = Psi_l^* T~ Phi_l^*) = the free model read (T~ itself), every invariant, q = 1, 2, 3;
//        and the tree groups = D5's production groups;
//   (M)  T~ applied sparsely = D4's dense bulk operator entrywise; T~ a cochain map on C_I;
//   (E)  periodicity: H^4 = H^2 with T_p on the tree (thm:descentcert(6));
//   (C)  commutation of the T_p on every torsion group read (one system of generators) and on H^1 over F_l';
//   (W)  the word arithmetic of prop:crtcharpolyN(4) (Garner, the sign, mod 2^64 / 3^40 with the word mulmod, the negative case), also
//        on 300 primes (~ 8680-bit integers);
//   (B)  the bound and the Eisenstein root; the free quotient of thm:descentcert(4) at l >= 5;
//   (K)  the precision rule: a read at K = 2 sigma_max - 1 refuses the module and keeps P_j and the Fitting types;
//   (N)  negative controls through the production route, each with its exact caught pattern (a held-out mismatch, int64 and digits
//        branches, is a HeckeInternalError "[HeldOutMismatch]", a defect, not a refusal);
//   (R)  refusals by code (the l >= 5 cokernel read's RssCeiling included, at (1, 70000) under the default ceiling); the record's refused
//        flag; a genuine internal error is not a refusal;
//   (L)  (ICENSUS-P1) the ker / coker width ladder at l = 2, q = 1, level 1, n = 48, 64, 80, 200, P = {2, 3, 5} (cells where the read at
//        k_0 = min(30, 2 a_max + 3) falls short, the ICENSUS probes' [KerCokerPrecision]): every read now reads, at least one per cell
//        escalated to k = 30; the module = the groups' 2-primary part (RunTreeLocalPieceN); an escalated read forced to k_0 refuses as a
//        precision shortfall (kc_precision), a read at k_0 forced to k = 30 gives the same text; per c, the full types of ker / coker
//        (T_p - c) against an independent 2-adic Smith form (coker [F | diag 2^{a_i}], and the Pontryagin dual's cokernel for ker).
// Each arm runs in its own try: an exception fails that arm and the later arms still run.
// IntegralDescentHeckeSlowValidation (slow): (O) on every oracle cell, (S) at three good primes on 17 cells, (X) on 17 cells, and the cost
//   cells (1, 60), (1, 100), (37, 2), (11, 10), (101, 0), (23, 4), (1, 200), (1009, 0), (101, 10), (211, 10) (DESCENT3_COST lines); read-only
//   hook HECKE_DESCENT_HECKE_CELLS=<N>:<n>[:<p>/<p>..][:chi|tors|all][:<max primes>],... replaces the cells (EFFECTIVE line).

#include <cstdint>

uint32_t IntegralDescentHeckeValidation();
uint32_t IntegralDescentHeckeSlowValidation();

#endif
