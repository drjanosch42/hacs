// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_PAIR_HOMOTOPY_VALIDATION_H
#define HECKE_PAIR_HOMOTOPY_VALIDATION_H

// hecke-extensions slot 2, WP-8 (phase E5-1) -- validation of the certified
// pair homotopy eta (Hecke/HeckePairHomotopy) against docs/latex/gl2_contract.tex
// ([C], cited by label) chapter sec:cone: [C, def:pairhomotopy],
// [C, lem:etaexists], [C, lem:etaunique], [C, obl:eta](i)-(iv), and the
// degree bookkeeping of [C, open:ell2] rider (i). Registered as `standard`
// (hecke_extensions_slot2_plan.md Sec. 3 WP-8 scope 3; the validation name
// deviates from [X] Sec. 5's HeckeConeValidation, which WP-13 records).
//
// INDEPENDENCE. Every predicate is re-derived here from the contract text
// and evaluated through public accessors only: an OWN canonical
// accumulation over SL2ZTessellationTables::canonical_key (the S2 primitive
// every Hecke validation shares), an OWN boundary operator (faces computed
// from K's vertex lists), an OWN (T1) translation (the (P1) split
// gamma = h s_j through HeckeCosetTables::class_index_of / s_inv and the
// conjugate g^{-1} h g = [[h.a, N h.b],[h.c / N, h.d]] written out here --
// never HeckeCosetTables::twisted_conjugate), an OWN transporter parity, an
// OWN slot / term / bound recount. The library's helpers are additionally
// cross-checked against the own ones (eta_on_translate, defect,
// defect_on_translate).
//
// GATES (a failure of any is a returned failure count > 0):
//   G1  certify() == 0 re-asserted on every table of the grid.
//   G2  [C, obl:eta](i) structural: every term det 1, a (q+1)-simplex of K,
//       canonical, coefficient != 0; the table refers to the SAME coset datum
//       and model as tau and tau_bd (identity), tau integral; the record
//       fields solve_regime == "Z", integral == true; num_degrees() == 2,
//       degrees() == [0, 1], q_max() == 1 derived, truncated == false
//       ([C, open:ell2] rider (i) on the finite model).
//   G3  [C, obl:eta](ii) on every stored slot, own boundary / own faces:
//       degree 0  boundary(eta(s_j v)) == tau_bd(s_j v) - tau(s_j v);
//       degree 1  boundary(eta(s_j e)) + eta(s_j v_b) - eta(s_j v_a) ==
//                 tau_bd(s_j e) - tau(s_j e), e = {v_a < v_b}.
//   G4  [C, obl:eta](ii) on 12 fixed-seed random Gamma-translates gamma . sigma
//       per (table, slot), all three tables own-translated: boundary(eta(gamma
//       sigma)) + eta(gamma boundary sigma) == (j tau_bd - tau j)(gamma sigma);
//       and the library helper eta_on_translate / defect_on_translate equal to
//       the own translation.
//   G5  [C, obl:eta](iii): for EVERY stored transporter t in S(sigma, sigma')
//       between boundary simplices and every j, slot (j, sigma) equals the
//       own sign-tracked (T1)-translate through s_j t; and the entry bound:
//       B_tab recounted, B = max(B_tab, 4 N^2), 2 B^2 < 2^63.
//   G6  [C, obl:eta](iv): solve regime Z on every table for EVERY ell
//       (ell in {2, 3} and the u64 primes included); no mod-m path exists.
//   G7  ell / n / width independence: the whole table (every term) is
//       identical across ell, n and the element width at fixed (N, policy).
//   G8  Tamper arm ([C, obl:eta] "a failure aborts"): eight tampered chains
//       (a term dropped, a coefficient changed, a nonzero vertex chain zeroed
//       -> 30, a nonzero edge chain zeroed -> 31, a transporter moved by T, a
//       det != 1 element -> 11, a non-canonical sign -> 12, a wrong-dimension
//       simplex -> 14, a chain moved to a Gamma''-related slot without
//       conjugation) each rejected by certify_with_override; the untampered
//       table certifies.
//   G9  eta_zero ([C, obl:eta](vi), RECORDED) agrees with WP-7's cusp-support
//       walk (Hecke/HeckeBoundaryDiagnostics): the verdict, the count of
//       differing vertex slots, and slot by slot (a vertex slot's eta is
//       nonzero iff tau and tau_bd differ there). Both false on the whole grid:
//       the eta-table is genuinely needed (recorded, asserted only as agreement).
//   G10 [C, lem:etaunique]: the two vertex-chain policies' tables eta, eta'
//       (both certified) differ by boundary L - L boundary with L the UNIQUE
//       degree-2 map -- L(x) := unique_fill(1, (eta' - eta)(x)) on every vertex
//       slot, own boundary(L(x)) == (eta' - eta)(x), and on every edge slot
//       (eta' - eta)(e) == -L(boundary e) exactly; the two tables differ on at
//       least one slot of the grid (the arm is not vacuous).
//   G11 Counts: num_slots == m * #(boundary simplices), constructed + derived,
//       nonzero / terms / max |coeff| recounted; the p = 2 table printed.
//
// Prints the p = 2, nu = 1 table and its JSON record once, and one line
//   PAIR_HOMOTOPY tables=... checks=... failures=... slots=... nonzero=...
//   eta_zero=0/... fills=... max_rings=... max_coeff=... policies_differ=... (secs)
// Returns the failure count (0 on success).

#include <cstdint>

uint32_t HeckePairHomotopyValidation();

#endif // HECKE_PAIR_HOMOTOPY_VALIDATION_H
