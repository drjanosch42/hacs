// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3) The boundary MP set and the labels of j -- see Boundary.h.
#include "Boundary.h"

#include <map>
#include <set>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

using integral_model::key_of;

Descriptor BoundaryDescriptor(const integral_model::ModelData& d, const Descriptor& D, bool planted) {
    Descriptor B = D;
    B.name = D.name + ":dX" + (planted ? "(planted trivial stabiliser)" : "");
    B.top.clear();
    for (uint32_t r = 0; r < d.reps[1].size(); ++r)
        if (d.bdry[1][r]) B.top.push_back(d.reps[1][r]);
    ENGINE_CHECK_MSG(!B.top.empty(), "orbifold_mp::BoundaryDescriptor: no boundary edge representative");
    for (const auto& e : B.top)
        for (uint32_t v : e) ENGINE_CHECK_MSG(D.boundary.count(v) == 1, "orbifold_mp::BoundaryDescriptor: a boundary edge leaves the horocycle");
    if (planted) {
        std::set<uint32_t> orbits;
        for (const auto& e : B.top)
            for (uint32_t v : e) orbits.insert(D.point.at(v).orbit);
        for (uint32_t o : orbits) B.orbit_stab[o] = CyclicGroup{Mat22::I(), 1};
    }
    return B;
}

bool ContainedInSimplex(const Descriptor& Dsub, uint32_t t, const Descriptor& Dbig, uint32_t sigma, const Mat22& h) {
    std::set<PointKey> big;
    for (const Point& w : Dbig.lifted(sigma)) big.insert(Dbig.key(Point{w.orbit, h * w.g}));
    for (const Point& p : Dsub.lifted(t))
        if (!big.count(Dbig.key(p))) return false;
    return true;
}

std::vector<std::pair<uint32_t, Mat22>> ContainingLabels(const Descriptor& Dsub, uint32_t t, const Descriptor& Dbig) {
    std::vector<std::pair<uint32_t, Mat22>> out;
    const std::vector<Point> pts = Dsub.lifted(t);
    ENGINE_CHECK_MSG(!pts.empty(), "orbifold_mp::ContainingLabels: empty simplex");
    const Point& p = pts[0];
    for (uint32_t s = 0; s < Dbig.top.size(); ++s) {
        std::map<GKey, Mat22> cand;
        for (const Point& w : Dbig.lifted(s)) {
            if (w.orbit != p.orbit) continue;
            const Mat22 winv = w.g.inv_unimodular();
            for (const Mat22& u : Elements(Dbig.orbit_stab.at(p.orbit))) {
                const Mat22 h = p.g * u * winv;   // h * w = p
                cand.emplace(key_of(h), h);
            }
        }
        for (const auto& kv : cand)
            if (ContainedInSimplex(Dsub, t, Dbig, s, kv.second)) out.emplace_back(s, kv.second);
    }
    return out;
}

RefinementLabels BoundaryLabels(const Descriptor& Dbdry, const Descriptor& D) {
    RefinementLabels L;
    for (uint32_t t = 0; t < Dbdry.top.size(); ++t) {
        const auto all = ContainingLabels(Dbdry, t, D);
        ENGINE_CHECK_MSG(!all.empty(), "orbifold_mp::BoundaryLabels: a boundary edge lies in no top simplex");
        for (const auto& x : all)
            ENGINE_CHECK_MSG(x.first == all.front().first, "orbifold_mp::BoundaryLabels: a boundary edge in two top simplices");
        L.coarse.push_back(all.front().first);   // candidates come sorted by GKey within one simplex
        L.h.push_back(all.front().second);
    }
    return L;
}

}  // namespace orbifold_mp
