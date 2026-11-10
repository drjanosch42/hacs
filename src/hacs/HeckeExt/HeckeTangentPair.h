// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_TANGENT_PAIR_H
#define HECKE_EXT_HECKE_TANGENT_PAIR_H

// ===========================================================================
// (hecke-extensions slot 7, WP-5 / phase E6 cont.) THE PER-rho-bar TANGENT
// PAIR -- [E] Stage 6 ("for each cuspidal rho-bar in Sigma(ell), from a weight
// with multiplicity >= 2 (n = w - 2 + ell^2 - 1) compute dim m/m^2 and a
// generating pair of anemic operators; store it as the default S_m (Stage 4
// still certifies)"), `cl:bk`; [X] Sec. 2.2 Stage 6 and Sec. 6.2 E6 ("the
// per-rho-bar tangent pair for ell <= 31"; [XA M7]).
//
// WHAT IS COMPUTED, stated before the code (slot-7 plan WP-5 scope 2):
//
//   For each CUSPIDAL system of a store Sigma(ell) (type "cuspidal": equal to
//   no chi_{a,b} on the pool):
//
//   * a COLLECTED system (twist 0) has a weight: w = the least n + 2 of its
//     H1 occurrences.  Its multiplicity->=2 weight is n = w - 2 + ell^2 - 1
//     (theta^{ell-1} rho-bar and rho-bar E_{ell-1}^{ell+1} agree at every
//     q != ell there, `cl:Tl`).
//       - n > 1000 (the model's reach, [X] Sec. 2.2): status `deferred`,
//         reason "n = <n> > 1000" -- every cuspidal system at ell = 691;
//       - otherwise HeckeExtDriver runs at (n, ell) on the ANEMIC pool of the
//         store, with u_ell = off (scoped): the multiplicity is a property of
//         the ANEMIC maximal ideal -- U_ell is what separates the two
//         systems, so admitting it would split the block and hide the
//         multiplicity the weight was chosen for.  The H1 ideal whose anemic
//         key (AtlasOrbitKey over the pool) is the system's is located; if
//         none is, status `not_found`; if it is not certified local,
//         `not_local`.  Else status `computed` with
//           D            dim_{F_ell} of the block,
//           r            its residue degree,
//           multiplicity D / (2 r) (the H1 block of one system is 2r-dim),
//           embdim       dim_K m/m^2 (LocalAlgebra::embdim, the K-span test),
//           active       the anemic primes with n_p-bar != 0 in m/m^2,
//           generating   the greedy K-spanning subset of the ANEMIC primes,
//                        in ascending p (LocalAlgebra::minimal_active is
//                        the same greedy; here the pool is anemic by
//                        construction, so the two agree -- asserted),
//           pair         generating.size() <= 2 (a generating PAIR exists;
//                        embdim <= 2 is necessary).
//   * a theta-TWISTED system (twist j != 0) is `inherited` from its base:
//     theta^j maps T_q to q^j T_q, a unit multiple, on the same module, so
//     the local algebra, m/m^2 and every K-span are the same and the
//     generating subset is the same primes.  Its weight is not recorded by
//     the atlas; the base's n and data are quoted.
//
//   The pair is a DEFAULT S_m, never a certificate (Stage 4 certifies), and
//   it enters neither the `hecke_atlas/v1` store nor any record nor golden:
//   it is printed by HeckeAtlasValidation and by `--atlas` (the store schema
//   stays v1; a persisted form is WP-7's / slot 8's to decide).
//
// At the atlas ell of the E6 campaign: ell in {5, 7} have NO cuspidal
// system (Delta mod 5 and mod 7 are Eisenstein-type: chi_{1,2} resp. its
// analogue), so their tangent list is empty; ell = 11 and 13 each have one
// collected cuspidal system (Delta, w = 12: n = 130 resp. 178) plus its
// theta-twists; ell = 691 is deferred throughout (n >= 477490).
// ===========================================================================

#include "HeckeAtlas.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

inline constexpr const char* kTangentPairSchema = "hecke_tangent_pair/v1";
inline constexpr uint32_t kTangentPairReach = 1000;   // [X] Sec. 2.2 Stage 6

struct TangentPair {
    uint32_t    ell = 0;
    uint32_t    system = 0;          // id in Sigma(ell)
    uint32_t    base = 0;
    uint32_t    twist = 0;
    uint32_t    weight = 0;          // w of the base (collected) system
    uint32_t    n = 0;               // w - 2 + ell^2 - 1
    std::string status;              // computed | inherited | deferred | not_found | not_local
    std::string reason;              // why deferred / not_found / not_local
    uint32_t    D = 0, r = 0, multiplicity = 0, embdim = 0;
    std::vector<uint64_t> active;
    std::vector<uint64_t> generating;
    bool        pair = false;

    std::string to_json() const;     // one `hecke_tangent_pair/v1` object
};

// The tangent pair of every cuspidal system of `s`, in system-id order.
// `n_reach` is the model's reach (default 1000); a weight beyond it is
// `deferred` without running anything.
std::vector<TangentPair> ComputeTangentPairs(const AtlasStore& s,
                                             uint32_t n_reach = kTangentPairReach);

} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_TANGENT_PAIR_H
