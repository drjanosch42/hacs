// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PeriodicColumn.h -- the periodic-resolution column helpers shared by
// StrictLift.cpp, PeriodicityClass.cpp and IsotropyModel.cpp.
//
// (hecke-extensions slot-14 WP-14b, review finding slot-10 m-2): since slot
// 10 the three functions below existed as three independent copies (one per
// consumer -- PeriodicityClass.cpp's comment even explained why: "IsotropyModel
// keeps it private and WP-13 uses the public API only"). Folded here,
// append-only, VALUE-PRESERVING: every body below is byte-identical to the
// ones it replaces (diffed before folding), so no computed object moves.
//
// decompose / dF_coeff were literally identical in all three files.
// column_homotopy was StrictLift.cpp's named function; PeriodicityClass.cpp's
// PeriodicityClass::fill() inlined the same loop body over a single column
// residue (now calls this function instead); IsotropyModel.cpp's batched
// `homotopy(col, z, t)` applies the same per-representative loop over a
// ColumnChain (now delegates to this function per representative -- col.m[r]
// == col.pow[r].size() by construction, see BuildResolution(), so using
// powg.size() for m changes nothing).
#pragma once
#ifndef INTEGRAL_MODEL_PERIODIC_COLUMN_H
#define INTEGRAL_MODEL_PERIODIC_COLUMN_H

#include <cstdint>
#include <utility>
#include <vector>

#include "IsotropyModel.h"

namespace integral_model {

// gamma = c * g^j with c the lexicographically least element of gamma <g>.
inline std::pair<GKey, uint32_t> decompose(const GKey& gamma, const std::vector<hecke::Mat22>& powg) {
    const uint32_t m = static_cast<uint32_t>(powg.size());
    const hecke::Mat22 G = mat_of(gamma);
    GKey best{};
    uint32_t i0 = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const GKey k = key_of(G * powg[i]);
        if (i == 0 || k < best) { best = k; i0 = i; }
    }
    return {best, (m - i0) % m};
}

// x_t: g - 1 from odd t, the norm from even t >= 2.
inline GroupRingElt dF_coeff(const std::vector<hecke::Mat22>& powg, uint32_t t) {
    GroupRingElt x;
    if (t % 2 == 1) {
        x.add(key_of(powg[1 % powg.size()]), 1);
        x.add(key_of(powg[0]), -1);
    } else {
        for (const auto& g : powg) x.add(key_of(g), 1);
    }
    return x;
}

// The Z-linear contracting homotopy h_t : F_t -> F_{t+1} of the periodic
// resolution, on one coefficient, coset by coset:
//   t even: h(c g^j) = sum_{i<j} c g^i;   t odd: h(c g^j) = c if j = m - 1, else 0.
inline GroupRingElt column_homotopy(const GroupRingElt& a, const std::vector<hecke::Mat22>& powg, uint32_t t) {
    GroupRingElt out;
    const uint32_t m = static_cast<uint32_t>(powg.size());
    for (const auto& [g, n] : a.terms) {
        const auto [c, j] = decompose(g, powg);
        const hecke::Mat22 C = mat_of(c);
        if (t % 2 == 0) {
            for (uint32_t i = 0; i < j; ++i) out.add(key_of(C * powg[i]), n);
        } else if (j == m - 1) {
            out.add(c, n);
        }
    }
    return out;
}

}  // namespace integral_model

#endif // INTEGRAL_MODEL_PERIODIC_COLUMN_H
