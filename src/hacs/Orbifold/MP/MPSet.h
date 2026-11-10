// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MPSET_H
#define ORBIFOLD_MP_MPSET_H

// ===========================================================================
// (hecke-extensions slot 22, MP-1a) The choice-free Moerdijk--Pronk simplicial
// set S(X; T) of [MP] Def. 1.5 (MP_math_plan.md v0.2), enumerated as finite
// data on a Descriptor -- the strings only (MP-1a: the counts |S_k|, the face
// maps, the simplicial identities).  The twisted cochains, ranks and the nerve
// cohomology are MP-1b's (MP_impl_plan.md v0.2 Sec. 2.3), built on this API.
//
//   S_k = { (s_0, ..., s_k ; g_1, ..., g_k) : s_i in Sigma_top, g_i in Gamma,
//           iota(s_0) cap g_1 iota(s_1) cap ... cap g_1...g_k iota(s_k) != empty }
//
// Enumeration as in mp_sl2z.py build(): S_k from S_{k-1} by appending (s, g),
// g in the vertex transporters A(s_{k-1}, s) (Arrows), kept iff some vertex p
// of iota(s_0) has (g_1...g_j)^{-1} p in iota(s_j) for every j (the
// common-vertex form of "non-empty", valid under (SC)).  Each string arises
// exactly once (A(.,.) is a set), so no de-duplication is needed; the
// validation re-checks that by key.  Faces: d_0 drops (s_0, g_1); d_i
// (0 < i < k) merges g_i g_{i+1} and drops s_i; d_k drops (s_k, g_k).
// A string is degenerate iff s_i = s_{i+1} and g_{i+1} = 1 for some i.
// ===========================================================================

#include <cstdint>
#include <set>
#include <vector>

#include "Descriptor.h"

namespace orbifold_mp {

struct MPString {
    std::vector<uint32_t> sigma;   // k + 1 top-simplex indices
    std::vector<Mat22> gamma;      // k group elements
    uint32_t k() const { return static_cast<uint32_t>(gamma.size()); }
    // An exact, ordered encoding (for set membership; no hash on a result path).
    std::vector<int64_t> encode() const;
};

class MPSet {
public:
    MPSet(const Descriptor& D, uint32_t k_max);

    uint32_t k_max() const { return static_cast<uint32_t>(S_.size()) - 1; }
    const std::vector<MPString>& strings(uint32_t k) const { return S_.at(k); }   // all, degenerate included
    uint64_t count_all(uint32_t k) const { return S_.at(k).size(); }
    uint64_t count_nondegenerate(uint32_t k) const;

    static bool degenerate(const MPString& s);
    static MPString face(const MPString& s, uint32_t i);   // d_i, 0 <= i <= k
    // Does the string satisfy the defining condition (independent re-test)?
    bool member(const MPString& s) const;

private:
    Descriptor D_;   // a copy: an MPSet never dangles
    std::vector<std::set<PointKey>> keys_;   // vertex keys of iota(s), per top simplex
    std::vector<std::vector<std::vector<Mat22>>> arrows_;   // [s0][s1]
    std::vector<std::vector<MPString>> S_;
};

// [MP] Sec. 1.2 / audit adapted_form_check.py: MP's mu-form count sum over (k+1)-tuples of
// quotient top simplices with non-empty orbit intersection tau of |Gamma_{v(tau)}|^k,
// v(tau) the vertex of tau of largest stabiliser order.
uint64_t MuFormCount(const Descriptor& D, uint32_t k);

}  // namespace orbifold_mp

#endif
