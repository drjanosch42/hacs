// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_ATLAS_VALIDATION_H
#define HECKE_EXT_HECKE_ATLAS_VALIDATION_H

// ===========================================================================
// (hecke-extensions slot 6, WP-8 / phase E6) the two atlas gates of
// HeckeExt/HeckeAtlas ([X] Sec. 5 `HeckeAtlasValidation`; the slot-6 plan
// Sec. 3 WP-8 scope 5 and 6).
//
// HeckeAtlasSkeletonValidation  (fast; ell in {5, 7})
//   (A) Sigma(ell) is CLOSED under the theta-twist, checked two ways: on the
//       canonical keys (every twist of every system is a system) and on the
//       POLYNOMIAL data -- every collected ideal's (pi_w, {P_p}) twisted to
//       (pi_w, {p^j P_p}) and re-split in the canonical field keys to the
//       scalar twist of its own key, and CertifyCanonicalSplit passes on
//       every split, twisted or not;
//   (B) the greedy S_sep separates Sigma u chi, and is minimal in the weak
//       sense that dropping any one of its primes does not;
//   (C) the JSON round-trips: to_json(parse(to_json(x))) == to_json(x), plus
//       field-by-field equality and a refusal on a corrupted schema string;
//   (D) THE STORE IS BYTE-DETERMINISTIC: two independent builds give the same
//       bytes; and the write gate is CLOSED without HECKE_ATLAS_OUT;
//   (J) the r >= 2 path on constructed data (every system of Sigma(ell),
//       ell <= 13, is F_ell-rational, so the atlas's own cells never reach
//       it): orbit keys, the twist on keys vs on polynomial data at
//       r in {2, 3}, the twist group law, and AtlasSeparationBy's three
//       decision paths (rational, same r, the conservative `undecided`).
//
// HeckeAtlasValidation  (slow; the full slot-6 scope, owner ruling 4)
//   Sigma(5), Sigma(7), Sigma(11), Sigma(13), each with (A)-(C) above, and
//   (E) the Eisenstein systems 1 + q^{n+1} occur in H^1 at every even
//       2 <= n <= ell - 1, and Sigma contains every chi_{a,b} (a + b odd);
//   (F) the cuspidal count: 0 base systems at ell in {5, 7}, exactly one --
//       Delta -- at ell in {11, 13}, with a_q = tau(q) mod ell on the whole
//       pool, at n = 10 only, of residue degree 1; at ell = 11 (weight
//       ell + 1) it is NOT a theta-twist of any lower-weight system;
//   (G) bernoulli_mod against the Kummer table (691 | B_12, 3617 | B_16,
//       103 * 2294797 | B_24, 657931 | num(B_26/26)) and, on the atlas, the
//       Kummer/Ribet dichotomy: the Eisenstein block of H^1 at weight
//       k = n + 2 <= ell - 1 has dimension 1 iff ell does not divide
//       num(B_k/k) (regular primes: always 1);
//   (H) Glover factors against the Brauer character of rho_n for n <= 3 ell
//       at ell in {5, 7} (the SAME oracle as HeckePredictionsValidation arm
//       [H]), and every Glover factor's weight n_i + 2 is an atlas cell;
//   (I) the per-cell column canonical_field_s / total_s (WP-8 acceptance (d)),
//       printed for every cell -- reported, never asserted;
//   and writes each store under HECKE_ATLAS_OUT=<dir> when (and only when)
//   that is set.
//
// (slot-7 WP-5) added arms:
//   (L) AtlasSeparationBy == the slot-6 pairwise loop (kept verbatim in the
//       .cpp) on every subset S of the pool (ell in {5, 7} in the skeleton,
//       {11, 13} here) and on arm (J)'s constructed store (all three
//       decision paths, the `undecided` one included);
//   (M) CollectAtlas + CloseAtlas == BuildAtlas, and a parsed store
//       re-closes to itself byte for byte;
//   (N) the per-rho-bar tangent pair (HeckeTangentPair.h) at ell in
//       {5, 7, 11, 13}: none at {5, 7}, Delta computed at n = 10 + ell^2 - 1
//       with multiplicity >= 2 and every twist inherited at {11, 13}; (N')
//       in the skeleton: the deferral path at ell = 691 on a constructed store;
//   (K) THE ell = 691 STORE'S INTEGRITY without rebuilding it: read from
//       HECKE_ATLAS_IN=<dir> or its publication path under results/ext_first/
//       (SKIP when neither exists); it parses, round-trips byte for byte, has
//       the 346 cells and the pool, Delta mod 691 is chi(0, 11) with an H1
//       occurrence at n = 10, a_691 is known exactly where U_691 was
//       admitted, S_sep separates, and every cuspidal tangent pair is
//       deferred.
// ===========================================================================

#include <cstdint>

uint32_t HeckeAtlasSkeletonValidation();
uint32_t HeckeAtlasValidation();

#endif // HECKE_EXT_HECKE_ATLAS_VALIDATION_H
