// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_PERIODICITY_CLASS_VALIDATION_H
#define HECKE_PERIODICITY_CLASS_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 10, WP-13 (Line 2, L2-1b (2)): the class
// e in H^2(Gamma; Z) of [C, lem:bulkperiodic](1) as a 2-cocycle in
// Hom_{Z[Gamma]}(R_2, Z) on the isotropy-resolution model R = Iso_. of
// [C, def:isomodel], and E_q = e u - as a chain map R_{.+2} -> R_. over
// Z[Gamma] and as a matrix on C^._I over any ModCtx
// (Integral/Model/PeriodicityClass).  [C] = docs/latex/gl2_contract.tex,
// cited by label.
//
// Arms:
//   (A) the construction certificate: e a cocycle, 12 e = delta^1(Phi) over
//       Z, res_C e = chi_C at the three kinds of vertex stabiliser (the zeta_+
//       characters), chi == zeta_+ on the stabilisers' powers and conjugates,
//       aug o E_0 = e, d E_q = E_{q-1} d for q = 1 .. 4 exactly over Z[Gamma],
//       the tree map C_.(Xbulk) -> C_.(T) a chain map; integer sizes printed;
//   (O) [e] has order 12 and generates H^2(Gamma; Z) = Z/12 (O2 of M0):
//       6 e is not a coboundary (2 e mod 4 is not one over Z/4), 4 e is not
//       (e mod 3 is not one over F_3), 12 e is (Phi);
//   (C) E_q^* is a cochain map at word size: delta^{q+2} E_q^* =
//       E_{q+1}^* delta^q, q = 0 .. 3, on every ring below;
//   (P) [C, lem:bulkperiodic](2), clause by clause AS WRITTEN, on every M0
//       cell (n <= 20, ell in {2, 3}, over F_ell and over Z/ell^K with K
//       M0's certified exponent) and over F_ell at good ell in {5, 7}:
//         q = 2, 3: E_q : H^q -> H^{q+2} an isomorphism (the image of H^q in
//                   H^{q+2} has the length of both), source and target
//                   lengths equal to M0's groups by the UCT
//                   ([C, prop:certifiede](3)) at 2, 3 and to
//                   IsotropyModel::cohomology_lengths everywhere;
//         q = 1:    E_1 : H^1 -> H^3 surjective; ker E_1 = partial(N_{-I} V)
//                   as submodules of H^1 (partial(N_{-I}V) is in the kernel,
//                   and the lengths agree), partial realised on the model by
//                   the Mayer-Vietoris connecting cochains of the Bass-Serre
//                   tree, and its length also computed in V alone as
//                   length((N V + V^S + V^U) / (V^S + V^U)); at odd n
//                   N_{-I} V = 0 and E_1 injective.  No "zero" clause is
//                   asserted (H^3 != 0 at the M0 cells);
//         q = 0:    nothing (the lemma states nothing there).
//       Degree bookkeeping: the reads are H^1..H^5 (q_max = 5); degree 6 is
//       only the target of delta^5 and is never read as cohomology.
//   (N) negative controls: a perturbed E entry fails the chain-map
//       certificate; a perturbed e fails the class certificate; 2 E_2^* over
//       Z/2^K is refused as an isomorphism where H^2 has 2-torsion; the zero
//       cochain in place of partial(N_{-I} V) is refused where ker E_1 != 0.
// Writes nothing; no golden object, no record key, no settings key.
// Returns the number of failed checks.
uint32_t HeckePeriodicityClassValidation();

#endif
