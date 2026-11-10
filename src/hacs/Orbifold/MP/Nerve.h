// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_NERVE_H
#define ORBIFOLD_MP_NERVE_H

// ===========================================================================
// (hecke-extensions slot 22, MP-1b) The MP set S(X; T) of [MP] Def. 1.5
// (MP_math_plan.md v0.2) in INTERNED form, and its twisted normalised cochain
// complex C^.(S; A_{Sym^n}) over Z/ell^k ([MP] Def. 1.5, Thm 2.1;
// MP_impl_plan.md v0.2 Sec. 2.3).
//
// INTERNING.  Every group element occurring in a string lies in the finite
// union U of the vertex transporters A(s, s') (orbifold_mp::Arrows): gamma_i
// in A(s_{i-1}, s_i) by construction, and every cumulative product
// g_j = gamma_1 ... gamma_j and every face product gamma_i gamma_{i+1} lies in
// some A(., .) because the string's intersection is non-empty.  U is interned
// once (sorted by GKey; 32-bit indices) with its partial multiplication table
// (UINT32_MAX where a product leaves U), so a string of S_k is k + 1 simplex
// indices and k element indices -- |S_4| ~ 1.1 M strings fit in ~50 MB.
//
// ENUMERATION.  S_k from S_{k-1} by appending (s, g), g in A(s_{k-1}, s), in
// (parent, s, g) order, so the level keys (parent * F + s) * |U| + g are
// increasing and a string is found by k binary searches.  Each string carries
// its cumulative product g_k and the mask of the vertices p of iota(s_0) with
// g_j^{-1} p in iota(s_j) for all j <= k (<= 3 bits): the common-vertex form
// of "non-empty intersection" under (SC), exactly MPSet's test, updated in
// O(1) per extension from a precomputed table.  MPSet (MP-1a) is the
// independent reference: the validation checks the two agree string for
// string to k = 3.
//
// COCHAINS.  C^k = { f : S_k^nd -> M }, M = Sym^n (Z/m)^2, flat index
// (nd index) * (n + 1) + component, and
//   (df)(s) = rho_n(gamma_1) f(d_0 s) + sum_{i >= 1} (-1)^i f(d_i s),
// f := 0 on degenerate faces, rho_n = integral_model::rho_n (the shipped
// Sym^n convention, [C, def:rhon]).  Coboundary() returns d^k : C^k -> C^{k+1}
// as CSR rows (one row per (s, a), s in S_{k+1}^nd), entries reduced mod m.
// `twisted = false` drops rho_n(gamma_1) (the R5 negative control).
//
// FormalDSquared() checks d o d = 0 in Z[U] (x) Z[S_q^nd] -- before any
// coefficient module is chosen, so for every M and every ring at once --
// exhaustively on S_{q+2}^nd.  No modular arithmetic beyond `%` on residues
// < 2^8 * 2^8; no golden object; no setting.  Pure host C++.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "Descriptor.h"
#include "MPSet.h"
#include "SparseLength.h"

namespace orbifold_mp {

class InternedMPSet {
public:
    // kNone: the 32-bit "no element" of the multiplication table; kNoIndex: the 64-bit "no string" of the
    // level indices (find, nd_index).  (MP-2, F-MP1 F6: MP-1b used the 32-bit kNone for both.)
    static constexpr uint32_t kNone = 0xFFFFFFFFu;
    static constexpr uint64_t kNoIndex = UINT64_MAX;

    InternedMPSet(const Descriptor& D, uint32_t k_max);

    uint32_t k_max() const { return static_cast<uint32_t>(sig_.size()) - 1; }
    uint32_t top_count() const { return F_; }
    // The interned elements U (sorted by GKey) and the identity's index.
    const std::vector<Mat22>& elements() const { return elems_; }
    uint32_t identity() const { return id_; }
    uint32_t mul(uint32_t a, uint32_t b) const { return mult_[static_cast<size_t>(a) * T_ + b]; }

    uint64_t count_all(uint32_t k) const { return sig_.at(k).size() / (k + 1); }
    uint64_t count_nondegenerate(uint32_t k) const { return nd_.at(k).size(); }
    // String i of level k (all strings): simplex indices (k + 1) and element indices (k).
    const uint32_t* sigma(uint32_t k, uint64_t i) const { return &sig_[k][i * (k + 1)]; }
    const uint32_t* gamma(uint32_t k, uint64_t i) const { return k ? &gam_[k][i * k] : nullptr; }
    bool degenerate(uint32_t k, uint64_t i) const;
    // The non-degenerate strings: nd(k)[j] = index in the full level; nd_index(k, i) the inverse (kNoIndex if degenerate).
    const std::vector<uint64_t>& nd(uint32_t k) const { return nd_.at(k); }
    uint64_t nd_index(uint32_t k, uint64_t i) const { return ndi_.at(k)[i]; }

    // Index (in the full level k) of the string (sigma; gamma), kNoIndex if it is not in S_k.
    uint64_t find(uint32_t k, const uint32_t* sigma, const uint32_t* gamma) const;
    // d_i of string idx at level k >= 1, as an index into level k - 1 (ENGINE_CHECK: the face is in S_{k-1}).
    uint64_t face(uint32_t k, uint64_t idx, uint32_t i) const;
    // The same as an MPString (for the comparison with MPSet).
    MPString to_string(uint32_t k, uint64_t i) const;

    // Bytes held by the string tables (the RSS prediction's main term).
    uint64_t table_bytes() const;

private:
    uint32_t F_ = 0, T_ = 0, id_ = 0;
    std::vector<Mat22> elems_;
    std::vector<uint32_t> mult_;
    std::vector<std::vector<uint32_t>> sig_, gam_, cum_;
    std::vector<std::vector<uint8_t>> mask_;
    std::vector<std::vector<uint64_t>> key_;   // per level: (parent * F + s) * T + g, increasing
    std::vector<std::vector<uint64_t>> nd_, ndi_;
    std::vector<std::vector<uint64_t>> facetab_;   // levels 1 .. min(k_max, 3): all faces, precomputed
    uint64_t face_uncached(uint32_t k, uint64_t idx, uint32_t i) const;
};

// The twist on d_0: Left = rho_n(gamma_1) ([MP] Def. 1.5, the shipped complex); Untwisted = none (MP-1b's NC);
// Inverse = rho_n(gamma_1^{-1}), an anti-homomorphism (MP-2, F-MP1 F1: the NC that sees the side of the action,
// expected to break d o d = 0).
enum class Twist : uint8_t { Left = 0, Untwisted = 1, Inverse = 2 };

// d^k : C^k -> C^{k+1} over Z/m (m = ell^e <= 255), Sym^n, CSR rows (S_{k+1}^nd x (n+1)) by
// (S_k^nd x (n+1)).  ENGINE_CHECK: k + 1 <= S.k_max().
SparseRows Coboundary(const InternedMPSet& S, uint32_t k, uint32_t n, uint32_t m, Twist twist);
inline SparseRows Coboundary(const InternedMPSet& S, uint32_t k, uint32_t n, uint32_t m, bool twisted = true) {
    return Coboundary(S, k, n, m, twisted ? Twist::Left : Twist::Untwisted);
}

// The product B * A of two CSR matrices over Z/m is zero (A: r_A x c_A, B: r_B x r_A).
bool ProductIsZero(const SparseRows& B, const SparseRows& A, uint32_t m);

// d^{q+1} d^q = 0 formally (in Z[U] (x) Z[S_q^nd]) on every string of S_{q+2}^nd; *checked = strings checked.
bool FormalDSquared(const InternedMPSet& S, uint32_t q, uint64_t* checked);

}  // namespace orbifold_mp

#endif
