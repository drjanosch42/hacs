// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_CANONICAL_FIELD_VALIDATION_H
#define HECKE_EXT_HECKE_CANONICAL_FIELD_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 5, WP-7 (phase E6).  Returns 0 on success (invariant
// I9); registered in ValidationRegistry::RegisterAllValidations() as a `fast`
// entry.  It builds no cohomology, no coset table and no matrix: the unit
// under test is pure F_ell polynomial arithmetic, so the whole validation is
// `Hecke/FpPoly` + `Hecke/HeckeSettings` + the unit itself.
//
// ARMS.
//
//   (A) C_{ell,r} AGAINST BRUTE-FORCE MINIMALITY.  For every (ell, r) whose
//       full search space fits, EVERY monic polynomial of degree r over F_ell
//       is enumerated in CanonicalLess order and tested for irreducibility by
//       an INDEPENDENT oracle -- trial division by every monic polynomial of
//       degree 1 .. floor(r/2), which shares no line with
//       hecke::CertifyIrreducible's gcd/Frobenius certificate.  The arm asserts
//       that `CanonicalFieldPolynomial(ell, r)` is the FIRST such polynomial,
//       that it is CanonicalLess-below every other irreducible of that degree,
//       and that the two irreducibility oracles agree on every one of the
//       ell^r candidates (not only on the winner).  It also pins the classical
//       small answers (Y, Y^2 + Y + 1, Y^3 + Y + 1, ...) and re-derives the
//       count of monic irreducibles against the Moebius formula
//       (1/r) sum_{d | r} mu(d) ell^{r/d}.
//
//   (B) THE FIELD ARITHMETIC.  `CanonicalField`'s add / sub / mul / inv / pow /
//       frobenius against an independent model: the SAME element as an
//       `hecke::FpPoly` reduced modulo C with `FpPoly::mod`, which is the
//       repository's own polynomial arithmetic and knows nothing of this
//       unit's reduction table.  Plus the field axioms (associativity,
//       distributivity, a * a^{-1} = 1), Fermat (a^{ell^r} = a for every a,
//       a^{ell^r - 1} = 1 for every a != 0), the Frobenius fixed field
//       (frob(a) = a iff a in F_ell) and the enumeration order
//       (`element_at` is a bijection onto F and `less` is exactly
//       `hecke::CanonicalLess` on the reduced representatives).
//
//   (C) POLYNOMIALS OVER F.  `FqPoly`'s ring laws, the division identity
//       a = q b + rem with deg rem < deg b re-multiplied out, gcd against the
//       definition (it divides both and every common divisor divides it, on
//       constructed products), `FqPowMod` against naive repeated multiplication,
//       and `FqEval` as a ring homomorphism.
//
//   (D) ROOT FINDING, AGAINST EXHAUSTION.  For every (ell, r) with |F| small
//       enough, EVERY element of F is tested against pi_w by direct evaluation,
//       giving the true root set; `FindCanonicalRoot` must return a member of
//       it, its r conjugates must be exactly that set, and `rho_canonical` must
//       be its CanonicalLess-least element.  Run for EVERY monic irreducible
//       pi_w of degree r, not just the canonical one -- that is the case a
//       report layer actually meets, where pi_w is the block's witness and C is
//       the project's.
//
//   (E) THE SPLIT AND ITS CERTIFICATE.  `SplitInCanonicalField` on constructed
//       character data: the eigenvalues re-evaluated independently, the
//       Frobenius orbit closed and of length r, `orbit[0] == eigenvalues`,
//       `CertifyCanonicalSplit == 0`, determinism (two runs byte-identical in
//       `to_json()`), independence of the caller's prime ORDER, and a
//       NINE-WAY TAMPER arm in which each corruption fails exactly its own
//       clause of the certificate.
//
//   (F) THE (2, 30) CASE.  `HeckeLocalityValidation` arm 7's covering example
//       produces a certified degree-30 irreducible over F_2 as the block's
//       witness polynomial; this arm builds the field that witness would be
//       reported in, F = F_2[Y]/(C_{2,30}), and splits degree-30 irreducibles
//       over F_2 in it.  It is the unit's stress case and the only one that
//       exercises the char-2 trace-map branch at the largest residue degree the
//       repository produces.  `C_{2,30}` itself is pinned and its
//       irreducibility independently confirmed.
//
//   (G) THE TWO SETTINGS KEYS.  `field_polynomial` and `split_field_report`:
//       their defaults, the registry ORDER (appended at the tail, so the
//       all-defaults fingerprint is still 0), parsing and its named refusals,
//       `MathSettings::text()` as `set()`'s inverse on both, the canonical
//       text and the non-zero fingerprint off the default, `to_json()` carrying
//       both with their source, and -- the point of `field_polynomial` --
//       `FieldPolynomialFromSettings` returning C_{ell,r} under `least` and
//       raising `hecke::NotImplementedInThisBuild` (with its literal tag) under
//       `conway`, never a parse error and never a silent fall back.
//
//   (H) THE OPERATION COUNT (slot-7 WP-4).  FindCanonicalRoot on a constructed
//       witness at r = 2..20, ell in {5, 691, 134217757}, under the
//       deterministic counter (`CANONFIELD_OPS` lines: candidates, F_ell
//       multiplications, F multiplications, F.inv calls, polynomial products,
//       divisions, Frobenius applications), the fitted exponent in r, and the
//       PRE-WP-4 ROUTE kept verbatim as an oracle: at r <= 10 on the three
//       ell, and on every monic irreducible of seven small (ell, r) with ell = 2
//       and 3 included, the two routes agree on candidates_used, rho,
//       rho_canonical and the conjugates (`CANONFIELD_OPS_OLD` prints the old
//       route's counts beside the new).
//
//   (I) THE `conway` ROUTING (slot-7 WP-4).  Under `field_polynomial =
//       conway`, one arm per production entry point raises
//       hecke::NotImplementedInThisBuild: CanonicalField(ell, r), the `--bench
//       ext` path SplitInCanonicalField(ell, ...), CertifyCanonicalSplit's
//       clause 3, BuildAtlas (one cell at ell = 5 -- the only place this
//       validation runs the driver), and FieldPolynomialFromSettings; under
//       `least` the routing is byte-identical.  The console's exit-5 arm is
//       ConsoleSettingsValidation (g).
uint32_t HeckeCanonicalFieldValidation();

#endif // HECKE_EXT_HECKE_CANONICAL_FIELD_VALIDATION_H
