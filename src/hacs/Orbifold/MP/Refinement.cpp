// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) The refinement map of MP sets -- see Refinement.h.
#include "Refinement.h"

#include <algorithm>
#include <map>
#include <set>

#include "CochainMaps.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

using integral_model::GKey;
using integral_model::key_of;
using integral_model::ModelData;
using integral_model::Simplex;

namespace {

Simplex sorted_of(Simplex s) {
    std::sort(s.begin(), s.end());
    return s;
}

// The leaves of the subdivision record of a coarse simplex (empty if it was not split).
std::vector<Simplex> Leaves(const ModelData& fine, const Simplex& coarse_simplex) {
    std::vector<Simplex> out;
    const Simplex key = sorted_of(coarse_simplex);
    for (const auto& [root, chain] : fine.refines)
        if (sorted_of(root) == key)
            for (const auto& [sg, leaf] : chain) {
                (void)sg;
                out.push_back(leaf);
            }
    return out;
}

}  // namespace

bool ContainedIn(const ModelData& fine, const Descriptor& Dfine, const Descriptor& Dcoarse, uint32_t fine_t,
                 uint32_t coarse_t, const Mat22& h) {
    // The vertex labels of the subdivided closed coarse simplex (labels of the shared fundamental complex K').
    std::set<uint32_t> allowed;
    for (uint32_t v : Dcoarse.top.at(coarse_t)) allowed.insert(v);
    for (const Simplex& leaf : Leaves(fine, Dcoarse.top.at(coarse_t)))
        for (uint32_t v : leaf) allowed.insert(v);
    std::set<PointKey> keys;
    for (uint32_t v : allowed) {
        const auto it = Dfine.point.find(v);
        ENGINE_CHECK_MSG(it != Dfine.point.end(), "orbifold_mp::ContainedIn: a coarse vertex label is not a point of X~");
        keys.insert(Dfine.key(Point{it->second.orbit, h * it->second.g}));
    }
    for (const Point& p : Dfine.lifted(fine_t))
        if (!keys.count(Dfine.key(p))) return false;
    return true;
}

RefinementLabels LabelsFromRecord(const ModelData& fine, const Descriptor& Dfine, const ModelData& coarse,
                                  const Descriptor& Dcoarse) {
    RefinementLabels L;
    for (uint32_t t = 0; t < Dfine.top.size(); ++t) {
        const Simplex ft = sorted_of(Dfine.top[t]);
        int found = -1, hits = 0;
        for (uint32_t c = 0; c < Dcoarse.top.size(); ++c) {
            bool in = sorted_of(Dcoarse.top[c]) == ft;
            for (const Simplex& leaf : Leaves(fine, Dcoarse.top[c])) in = in || sorted_of(leaf) == ft;
            if (in) {
                found = static_cast<int>(c);
                ++hits;
            }
        }
        ENGINE_CHECK_MSG(hits == 1, "orbifold_mp::LabelsFromRecord: a fine top simplex is not in exactly one coarse one");
        const uint32_t c = static_cast<uint32_t>(found);
        // h_0 = I (the record lives in one fundamental complex K'); the coset h_0 Gamma_s, Gamma_s = <gen> of the
        // coarse triangle representative; the label = its least GKey among the elements that contain s~.
        ENGINE_CHECK_MSG(coarse.reps[2].at(c) == Dcoarse.top[c], "orbifold_mp::LabelsFromRecord: Sigma_top != reps[2]");
        const CyclicGroup G{coarse.gen[2][c], coarse.order[2][c]};
        bool have = false;
        Mat22 best = Mat22::I();
        for (const Mat22& g : Elements(G)) {
            if (!ContainedIn(fine, Dfine, Dcoarse, t, c, g)) continue;
            if (!have || key_of(g) < key_of(best)) best = g;
            have = true;
        }
        ENGINE_CHECK_MSG(have, "orbifold_mp::LabelsFromRecord: no label of the coset contains the fine simplex");
        L.coarse.push_back(c);
        L.h.push_back(best);
    }
    return L;
}

StringMap MapStrings(const InternedMPSet& Sf, const InternedMPSet& Sc, const RefinementLabels& L, uint32_t k_top) {
    ENGINE_CHECK_MSG(k_top <= Sf.k_max() && k_top <= Sc.k_max(), "orbifold_mp::MapStrings: level not built");
    std::map<GKey, uint32_t> cidx;
    for (uint32_t i = 0; i < Sc.elements().size(); ++i) cidx.emplace(key_of(Sc.elements()[i]), i);
    std::vector<Mat22> hinv;
    for (const Mat22& h : L.h) hinv.push_back(h.inv_unimodular());
    StringMap U;
    U.image.resize(k_top + 1);
    std::vector<uint32_t> sg(k_top + 1), gm(k_top + 1);
    for (uint32_t k = 0; k <= k_top; ++k) {
        const uint64_t N = Sf.count_all(k);
        U.image[k].assign(N, InternedMPSet::kNoIndex);
        for (uint64_t i = 0; i < N; ++i) {
            const uint32_t* s = Sf.sigma(k, i);
            const uint32_t* g = Sf.gamma(k, i);
            bool ok = true;
            for (uint32_t t = 0; t <= k; ++t) sg[t] = L.coarse.at(s[t]);
            for (uint32_t t = 0; t < k && ok; ++t) {
                const Mat22 x = hinv[s[t]] * Sf.elements()[g[t]] * L.h[s[t + 1]];
                const auto it = cidx.find(key_of(x));
                if (it == cidx.end()) ok = false;
                else gm[t] = it->second;
            }
            const uint64_t img = ok ? Sc.find(k, sg.data(), gm.data()) : InternedMPSet::kNoIndex;
            U.image[k][i] = img;
            if (img == InternedMPSet::kNoIndex) {
                ++U.outside;
            } else if (Sf.nd_index(k, i) != InternedMPSet::kNoIndex && Sc.nd_index(k, img) == InternedMPSet::kNoIndex) {
                ++U.degenerate_images;
            }
        }
    }
    return U;
}

SparseRows RefinementPullback(const InternedMPSet& Sf, const InternedMPSet& Sc, const RefinementLabels& L,
                              const StringMap& U, uint32_t q, uint32_t n, uint32_t ell, uint32_t e, bool twist) {
    const uint32_t m = PowU(ell, e), d = n + 1;
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    std::vector<Vec> rho(L.h.size());
    for (uint32_t t = 0; t < L.h.size(); ++t) {
        const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx, twist ? L.h[t] : Mat22::I(), n);
        rho[t].resize(static_cast<size_t>(d) * d);
        for (uint32_t a = 0; a < d; ++a)
            for (uint32_t b = 0; b < d; ++b) rho[t][a * d + b] = R.at(a, b) % m;
    }
    SparseRows A;
    ENGINE_CHECK_MSG(Sc.count_nondegenerate(q) * d < (uint64_t(1) << 32), "orbifold_mp::RefinementPullback: too many columns");
    A.ncols = static_cast<uint32_t>(Sc.count_nondegenerate(q) * d);
    for (uint64_t i : Sf.nd(q)) {
        const uint64_t img = U.image.at(q).at(i);
        ENGINE_CHECK_MSG(img != InternedMPSet::kNoIndex, "orbifold_mp::RefinementPullback: a string leaves S(X)");
        const uint64_t c = Sc.nd_index(q, img);
        const uint32_t t0 = Sf.sigma(q, i)[0];
        for (uint32_t a = 0; a < d; ++a) {
            if (c != InternedMPSet::kNoIndex)
                for (uint32_t b = 0; b < d; ++b) {
                    const uint32_t v = rho[t0][a * d + b];
                    if (v) {
                        A.col.push_back(static_cast<uint32_t>(c * d + b));
                        A.val.push_back(static_cast<uint8_t>(v));
                    }
                }
            A.start.push_back(A.col.size());
        }
    }
    return A;
}

}  // namespace orbifold_mp
