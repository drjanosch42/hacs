// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1a) The choice-free MP simplicial set -- see MPSet.h.
#include "MPSet.h"

#include <algorithm>
#include <iterator>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

std::vector<int64_t> MPString::encode() const {
    std::vector<int64_t> e;
    e.reserve(1 + sigma.size() + 4 * gamma.size());
    e.push_back(static_cast<int64_t>(gamma.size()));
    for (uint32_t s : sigma) e.push_back(s);
    for (const Mat22& g : gamma) {
        e.push_back(g.a);
        e.push_back(g.b);
        e.push_back(g.c);
        e.push_back(g.d);
    }
    return e;
}

MPSet::MPSet(const Descriptor& D, uint32_t k_max) : D_(D) {
    const uint32_t F = static_cast<uint32_t>(D_.top.size());
    keys_.resize(F);
    for (uint32_t t = 0; t < F; ++t)
        for (const Point& p : D_.lifted(t)) keys_[t].insert(D_.key(p));
    arrows_.assign(F, std::vector<std::vector<Mat22>>(F));
    for (uint32_t a = 0; a < F; ++a)
        for (uint32_t b = 0; b < F; ++b) arrows_[a][b] = Arrows(D_, a, b);

    S_.resize(1);
    for (uint32_t a = 0; a < F; ++a) S_[0].push_back(MPString{{a}, {}});
    for (uint32_t k = 1; k <= k_max; ++k) {
        std::vector<MPString> next;
        for (const MPString& s : S_[k - 1]) {
            const uint32_t last = s.sigma.back();
            // Cumulative products c_j = g_1 ... g_j, j < k, and their inverses.
            std::vector<Mat22> cinv(k);
            Mat22 c = Mat22::I();
            cinv[0] = c;
            for (uint32_t j = 1; j < k; ++j) {
                c = c * s.gamma[j - 1];
                cinv[j] = c.inv_unimodular();
            }
            const std::vector<Point> base = D_.lifted(s.sigma[0]);
            for (uint32_t b = 0; b < F; ++b)
                for (const Mat22& g : arrows_[last][b]) {
                    const Mat22 ck_inv = (c * g).inv_unimodular();
                    bool ok = false;
                    for (const Point& p : base) {
                        bool all = true;
                        for (uint32_t j = 1; j < k && all; ++j)
                            all = keys_[s.sigma[j]].count(D_.key(Point{p.orbit, cinv[j] * p.g})) == 1;
                        all = all && keys_[b].count(D_.key(Point{p.orbit, ck_inv * p.g})) == 1;
                        if (all) {
                            ok = true;
                            break;
                        }
                    }
                    if (!ok) continue;
                    MPString t = s;
                    t.sigma.push_back(b);
                    t.gamma.push_back(g);
                    next.push_back(std::move(t));
                }
        }
        S_.push_back(std::move(next));
    }
}

bool MPSet::degenerate(const MPString& s) {
    for (uint32_t i = 0; i < s.k(); ++i)
        if (s.sigma[i] == s.sigma[i + 1] && s.gamma[i] == Mat22::I()) return true;
    return false;
}

uint64_t MPSet::count_nondegenerate(uint32_t k) const {
    uint64_t n = 0;
    for (const MPString& s : S_.at(k)) n += degenerate(s) ? 0 : 1;
    return n;
}

MPString MPSet::face(const MPString& s, uint32_t i) {
    const uint32_t k = s.k();
    ENGINE_CHECK_MSG(k >= 1 && i <= k, "orbifold_mp::MPSet::face: face index out of range");
    MPString f;
    if (i == 0) {
        f.sigma.assign(s.sigma.begin() + 1, s.sigma.end());
        f.gamma.assign(s.gamma.begin() + 1, s.gamma.end());
    } else if (i == k) {
        f.sigma.assign(s.sigma.begin(), s.sigma.end() - 1);
        f.gamma.assign(s.gamma.begin(), s.gamma.end() - 1);
    } else {
        for (uint32_t j = 0; j <= k; ++j)
            if (j != i) f.sigma.push_back(s.sigma[j]);
        for (uint32_t j = 0; j < k; ++j) {
            if (j == i - 1) {
                f.gamma.push_back(s.gamma[i - 1] * s.gamma[i]);
                ++j;
            } else {
                f.gamma.push_back(s.gamma[j]);
            }
        }
    }
    return f;
}

bool MPSet::member(const MPString& s) const {
    // Forward form: the intersection of the vertex sets of the translates g_1...g_j iota(s_j), j = 0..k.
    std::set<PointKey> inter = keys_.at(s.sigma.at(0));
    Mat22 c = Mat22::I();
    for (uint32_t j = 1; j <= s.k() && !inter.empty(); ++j) {
        c = c * s.gamma[j - 1];
        std::set<PointKey> tj, out;
        for (const Point& p : D_.lifted(s.sigma[j])) tj.insert(D_.key(Point{p.orbit, c * p.g}));
        std::set_intersection(inter.begin(), inter.end(), tj.begin(), tj.end(), std::inserter(out, out.begin()));
        inter = std::move(out);
    }
    return !inter.empty();
}

uint64_t MuFormCount(const Descriptor& D, uint32_t k) {
    const uint32_t F = static_cast<uint32_t>(D.top.size());
    std::vector<std::set<uint32_t>> orb(F);
    for (uint32_t t = 0; t < F; ++t)
        for (const Point& p : D.lifted(t)) orb[t].insert(p.orbit);
    uint64_t total = 0;
    std::vector<uint32_t> idx(k + 1, 0);
    while (true) {
        std::set<uint32_t> tau = orb[idx[0]];
        for (uint32_t j = 1; j <= k; ++j) {
            std::set<uint32_t> out;
            std::set_intersection(tau.begin(), tau.end(), orb[idx[j]].begin(), orb[idx[j]].end(),
                                  std::inserter(out, out.begin()));
            tau = std::move(out);
        }
        if (!tau.empty()) {
            uint64_t mx = 0;
            for (uint32_t o : tau) mx = std::max<uint64_t>(mx, D.orbit_stab[o].order);
            uint64_t p = 1;
            for (uint32_t j = 0; j < k; ++j) p *= mx;
            total += p;
        }
        uint32_t pos = 0;
        while (pos <= k && ++idx[pos] == F) idx[pos++] = 0;
        if (pos > k) break;
    }
    return total;
}

}  // namespace orbifold_mp
