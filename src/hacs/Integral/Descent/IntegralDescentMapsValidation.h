// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_MAPS_VALIDATION_H
#define INTEGRAL_DESCENT_MAPS_VALIDATION_H

// DESCENT D2 (hecke-extensions slot 24; docs/implementation/slot24/DESCENT_plan.md v0.3 Sec. 4 "D2", report
// DESCENT2_report.md): res / cor for Gamma_0(MN) < Gamma_0(N), the level cone and its long exact sequence over Z
// (Integral/Descent/DescentMaps; [C, prop:descentmaps], clause C-D3, and its precision rule (2)(g), v1.23.0-pre.22).
//
// IntegralDescentMapsValidation (standard, <= 20 s).  Arms:
//   (T) the oracle table of record docs/implementation/slot24/d2_maps_oracle.tsv (d2_maps_oracle.py: exact integers over Z on the
//       free amalgam resolution, no code shared with cohomology_app): H^q(A), H^q(B), H^q(A, B) and kernel / image / cokernel
//       of res^q, cor^q, j^q, dl^q, q <= 4, on every cell with psi(MN)(n + 1) <= 40 (every cell in the slow name); 87 cells, among
//       them five with a prime ell >= 5 dividing [A : B] (D2 audit F2: (1,13,0), (1,25,0), (5,25,0) here, (5,25,2), (7,49,0) slow).
//       The table has a SECOND, fully independent oracle, the D2 code audit's docs/implementation/slot24/d2audit/d2audit_oracle.py
//       (own resolution, modules and Smith forms; F6), which reproduces every row (DESCENT2_report.md, "Audit fixes");
//   (E) the glued long sequence is exact at every spot through H^4(A) AS SUBGROUPS, in every local piece (counted; a failure
//       refuses the cell, so every compared cell carries it);
//   (I) cor o res = [A : B] on cohomology in every degree and piece; the split case at ell !| [A : B];
//   (X) the examples of (2)(b): H^3(Gamma_0(4), Gamma_0(8)) = Z/2 at n = 0, H^3(Gamma, Gamma_0(11)) = Z/3 at n = 4;
//       24 H^q(A, B) = 0 for q >= 3 on every compared cell;
//   (F) normal pairs ((4)(a), (b)): c_sigma o res = res, res o cor = Nm_G, and the five-term statement in degree 1;
//   (P) periodicity: Iso^(2) built to degree 6 at small cells, so H^5 and res^5 / cor^5 are computed, against q = 3;
//   (D) the end groups against D1's AssembleDescentN (an independent assembly route of the same pieces);
//   (N) negative controls, each must REFUSE with empty results: alpha perturbed in one entry (not a cochain map), a wrong
//       rational rank for res^1 (+1: the free-rank guard; -1: the Smith certification), rank + 1 on the invariants map of a normal
//       pair (admissible but false: the Smith certification, D2 audit F5), and an RSS ceiling below the prediction (refused before
//       anything is built).
// IntegralDescentMapsSlowValidation (slow): (T) on every oracle cell; DESCENT2_COST lines (serial pinned pool,
// HECKE_DESCENT_REPEATS rotated repeats, default 5) on default cells; HECKE_DESCENT_PAIR_CELLS=N:M:n[,...] replaces them
// (EFFECTIVE line; the RSS prediction is printed first and a cell above 0.9 x the 40 GiB ceiling is refused, never started).
// The hooks are READ-ONLY: they choose cells, never a value.

#include <cstdint>

uint32_t IntegralDescentMapsValidation();
uint32_t IntegralDescentMapsSlowValidation();

#endif
