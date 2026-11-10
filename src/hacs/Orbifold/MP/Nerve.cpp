// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-1b) The interned MP set and its twisted cochains -- see Nerve.h.
#include "Nerve.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "../../Integral/Model/IsotropyModel.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

InternedMPSet::InternedMPSet(const Descriptor& D, uint32_t k_max) {
    F_ = static_cast<uint32_t>(D.top.size());
    ENGINE_CHECK_MSG(F_ >= 1 && F_ < 256, "InternedMPSet: 1 <= |Sigma_top| < 256");
    // The interned elements: the union of all vertex transporters, sorted by GKey.
    std::vector<std::vector<std::vector<Mat22>>> arrows(F_, std::vector<std::vector<Mat22>>(F_));
    std::map<GKey, uint32_t> index;
    for (uint32_t a = 0; a < F_; ++a)
        for (uint32_t b = 0; b < F_; ++b) {
            arrows[a][b] = Arrows(D, a, b);
            for (const Mat22& g : arrows[a][b]) index.emplace(integral_model::key_of(g), 0);
        }
    for (auto& [k, v] : index) {
        v = static_cast<uint32_t>(elems_.size());
        elems_.push_back(integral_model::mat_of(k));
    }
    T_ = static_cast<uint32_t>(elems_.size());
    ENGINE_CHECK_MSG(index.count(integral_model::key_of(Mat22::I())) == 1, "InternedMPSet: the identity is not a transporter");
    id_ = index.at(integral_model::key_of(Mat22::I()));
    mult_.assign(static_cast<size_t>(T_) * T_, kNone);
    for (uint32_t a = 0; a < T_; ++a)
        for (uint32_t b = 0; b < T_; ++b) {
            const auto it = index.find(integral_model::key_of(elems_[a] * elems_[b]));
            if (it != index.end()) mult_[static_cast<size_t>(a) * T_ + b] = it->second;
        }
    std::vector<std::vector<std::vector<uint32_t>>> arr(F_, std::vector<std::vector<uint32_t>>(F_));
    for (uint32_t a = 0; a < F_; ++a)
        for (uint32_t b = 0; b < F_; ++b) {
            for (const Mat22& g : arrows[a][b]) arr[a][b].push_back(index.at(integral_model::key_of(g)));
            std::sort(arr[a][b].begin(), arr[a][b].end());
        }
    // ok[c][s0][j][s] : c^{-1} p_j in iota(s), p_j the j-th vertex of iota(s0).
    std::vector<std::set<PointKey>> keys(F_);
    std::vector<std::vector<Point>> lifted(F_);
    for (uint32_t t = 0; t < F_; ++t) {
        lifted[t] = D.lifted(t);
        ENGINE_CHECK_MSG(lifted[t].size() <= 8, "InternedMPSet: a top simplex with more than 8 vertices");
        for (const Point& p : lifted[t]) keys[t].insert(D.key(p));
    }
    std::vector<uint8_t> ok(static_cast<size_t>(T_) * F_ * 8 * F_, 0);
    auto okat = [&](uint32_t c, uint32_t s0, uint32_t j, uint32_t s) -> uint8_t& {
        return ok[((static_cast<size_t>(c) * F_ + s0) * 8 + j) * F_ + s];
    };
    for (uint32_t c = 0; c < T_; ++c) {
        const Mat22 ci = elems_[c].inv_unimodular();
        for (uint32_t s0 = 0; s0 < F_; ++s0)
            for (uint32_t j = 0; j < lifted[s0].size(); ++j) {
                const Point& p = lifted[s0][j];
                const PointKey key = D.key(Point{p.orbit, ci * p.g});
                for (uint32_t s = 0; s < F_; ++s) okat(c, s0, j, s) = keys[s].count(key) ? 1 : 0;
            }
    }

    sig_.resize(1);
    gam_.resize(1);
    cum_.resize(1);
    mask_.resize(1);
    key_.resize(1);
    for (uint32_t a = 0; a < F_; ++a) {
        sig_[0].push_back(a);
        cum_[0].push_back(id_);
        mask_[0].push_back(static_cast<uint8_t>((1u << lifted[a].size()) - 1));
        key_[0].push_back(a);
    }
    for (uint32_t k = 1; k <= k_max; ++k) {
        std::vector<uint32_t> sg, gm, cm;
        std::vector<uint8_t> mk;
        std::vector<uint64_t> ky;
        const uint64_t N = sig_[k - 1].size() / k;
        for (uint64_t i = 0; i < N; ++i) {
            const uint32_t* ps = &sig_[k - 1][i * k];
            const uint32_t s0 = ps[0], last = ps[k - 1];
            const uint32_t c = cum_[k - 1][i];
            const uint8_t msk = mask_[k - 1][i];
            for (uint32_t b = 0; b < F_; ++b)
                for (uint32_t g : arr[last][b]) {
                    const uint32_t c2 = mul(c, g);
                    if (c2 == kNone) continue;
                    uint8_t m2 = 0;
                    for (uint32_t j = 0; j < 8; ++j)
                        if ((msk >> j) & 1u)
                            if (okat(c2, s0, j, b)) m2 = static_cast<uint8_t>(m2 | (1u << j));
                    if (m2 == 0) continue;
                    for (uint32_t t = 0; t < k; ++t) sg.push_back(ps[t]);
                    sg.push_back(b);
                    for (uint32_t t = 0; t + 1 < k; ++t) gm.push_back(gam_[k - 1][i * (k - 1) + t]);
                    gm.push_back(g);
                    cm.push_back(c2);
                    mk.push_back(m2);
                    ky.push_back((i * F_ + b) * static_cast<uint64_t>(T_) + g);
                }
        }
        sig_.push_back(std::move(sg));
        gam_.push_back(std::move(gm));
        cum_.push_back(std::move(cm));
        mask_.push_back(std::move(mk));
        key_.push_back(std::move(ky));
    }
    for (uint32_t k = 0; k <= k_max; ++k)
        ENGINE_CHECK_MSG(std::is_sorted(key_[k].begin(), key_[k].end()), "InternedMPSet: level keys not increasing");
    nd_.resize(k_max + 1);
    ndi_.resize(k_max + 1);
    for (uint32_t k = 0; k <= k_max; ++k) {
        const uint64_t N = count_all(k);
        ENGINE_CHECK_MSG(N < kNoIndex, "InternedMPSet: level too large for a 64-bit index");
        ndi_[k].assign(N, kNoIndex);
        for (uint64_t i = 0; i < N; ++i)
            if (!degenerate(k, i)) {
                ndi_[k][i] = nd_[k].size();
                nd_[k].push_back(i);
            }
    }
    // The face table of the low levels (the top level's faces are computed on demand).
    facetab_.resize(std::min<uint32_t>(k_max, 3) + 1);
    for (uint32_t k = 1; k < facetab_.size(); ++k) {
        const uint64_t N = count_all(k);
        facetab_[k].resize(N * (k + 1));
        for (uint64_t i = 0; i < N; ++i)
            for (uint32_t j = 0; j <= k; ++j) facetab_[k][i * (k + 1) + j] = face_uncached(k, i, j);
    }
}

bool InternedMPSet::degenerate(uint32_t k, uint64_t i) const {
    const uint32_t* s = sigma(k, i);
    const uint32_t* g = gamma(k, i);
    for (uint32_t t = 0; t < k; ++t)
        if (s[t] == s[t + 1] && g[t] == id_) return true;
    return false;
}

uint64_t InternedMPSet::find(uint32_t k, const uint32_t* s, const uint32_t* g) const {
    if (k > k_max() || s[0] >= F_) return kNoIndex;
    uint64_t idx = s[0];
    for (uint32_t j = 1; j <= k; ++j) {
        if (s[j] >= F_ || g[j - 1] >= T_) return kNoIndex;
        const uint64_t key = (idx * F_ + s[j]) * static_cast<uint64_t>(T_) + g[j - 1];
        const auto& K = key_[j];
        const auto it = std::lower_bound(K.begin(), K.end(), key);
        if (it == K.end() || *it != key) return kNoIndex;
        idx = static_cast<uint64_t>(it - K.begin());
    }
    return idx;
}

uint64_t InternedMPSet::face(uint32_t k, uint64_t idx, uint32_t i) const {
    ENGINE_CHECK_MSG(k >= 1 && i <= k, "InternedMPSet::face: index out of range");
    if (k < facetab_.size()) return facetab_[k][idx * (k + 1) + i];
    return face_uncached(k, idx, i);
}

uint64_t InternedMPSet::face_uncached(uint32_t k, uint64_t idx, uint32_t i) const {
    const uint32_t* s = sigma(k, idx);
    const uint32_t* g = gamma(k, idx);
    uint32_t fs[16], fg[16];
    ENGINE_CHECK_MSG(k < 16, "InternedMPSet::face: k < 16");
    uint32_t ns = 0, ng = 0;
    for (uint32_t j = 0; j <= k; ++j)
        if (j != i) fs[ns++] = s[j];
    if (i == 0) {
        for (uint32_t j = 1; j < k; ++j) fg[ng++] = g[j];
    } else if (i == k) {
        for (uint32_t j = 0; j + 1 < k; ++j) fg[ng++] = g[j];
    } else {
        for (uint32_t j = 0; j < k; ++j) {
            if (j == i - 1) {
                const uint32_t p = mul(g[i - 1], g[i]);
                ENGINE_CHECK_MSG(p != kNone, "InternedMPSet::face: a face product leaves the transporters");
                fg[ng++] = p;
                ++j;
            } else {
                fg[ng++] = g[j];
            }
        }
    }
    const uint64_t f = find(k - 1, fs, fg);
    ENGINE_CHECK_MSG(f != kNoIndex, "InternedMPSet::face: a face is not in S_{k-1}");
    return f;
}

MPString InternedMPSet::to_string(uint32_t k, uint64_t i) const {
    MPString out;
    const uint32_t* s = sigma(k, i);
    const uint32_t* g = gamma(k, i);
    for (uint32_t t = 0; t <= k; ++t) out.sigma.push_back(s[t]);
    for (uint32_t t = 0; t < k; ++t) out.gamma.push_back(elems_[g[t]]);
    return out;
}

uint64_t InternedMPSet::table_bytes() const {
    uint64_t b = 0;
    for (uint32_t k = 0; k < sig_.size(); ++k)
        b += 8 * (k < facetab_.size() ? facetab_[k].size() : 0) + 4 * (sig_[k].size() + gam_[k].size() + cum_[k].size()) + mask_[k].size() + 8 * (key_[k].size() + nd_[k].size() + ndi_[k].size());
    return b + 4 * mult_.size();
}

SparseRows Coboundary(const InternedMPSet& S, uint32_t k, uint32_t n, uint32_t m, Twist twist) {
    const bool twisted = (twist != Twist::Untwisted);
    ENGINE_CHECK_MSG(k + 1 <= S.k_max(), "orbifold_mp::Coboundary: S_{k+1} not built");
    ENGINE_CHECK_MSG(m >= 2 && m <= 255, "orbifold_mp::Coboundary: 2 <= m <= 255");
    const uint32_t d = n + 1;
    // rho_n of every interned element, over Z/m.  The context: a prime m is a field, else m = ell^e.
    uint32_t ell = 2;
    while (m % ell) ++ell;
    uint32_t e = 0, mm = 1;
    while (mm < m) {
        mm *= ell;
        ++e;
    }
    ENGINE_CHECK_MSG(mm == m, "orbifold_mp::Coboundary: m must be a prime power");
    const engine::ModCtx<uint32_t> ctx = (e == 1) ? engine::ModCtx<uint32_t>(m) : engine::ModCtx<uint32_t>::prime_power(ell, e);
    const auto& E = S.elements();
    std::vector<std::vector<uint32_t>> rho(E.size());
    // Only the elements occurring as gamma_1 of a non-degenerate S_{k+1} string are needed.
    std::vector<uint8_t> need(E.size(), 0);
    for (uint64_t i : S.nd(k + 1)) need[S.gamma(k + 1, i)[0]] = 1;
    for (size_t x = 0; x < E.size(); ++x)
        if (need[x]) {
            const engine::DenseBuf<uint32_t> R =
                integral_model::rho_n(ctx, twist == Twist::Inverse ? E[x].inv_unimodular() : E[x], n);
            rho[x].resize(static_cast<size_t>(d) * d);
            for (uint32_t a = 0; a < d; ++a)
                for (uint32_t b = 0; b < d; ++b) rho[x][a * d + b] = R.at(a, b) % m;
        }
    SparseRows A;
    A.ncols = static_cast<uint32_t>(S.count_nondegenerate(k) * d);
    const auto& nd = S.nd(k + 1);
    A.start.reserve(nd.size() * d + 1);
    std::vector<std::pair<uint32_t, uint32_t>> row;   // (col, value)
    std::vector<uint64_t> fidx(k + 2);
    for (uint64_t r = 0; r < nd.size(); ++r) {
        const uint64_t idx = nd[r];
        for (uint32_t i = 0; i <= k + 1; ++i) fidx[i] = S.nd_index(k, S.face(k + 1, idx, i));
        const uint32_t g1 = S.gamma(k + 1, idx)[0];
        for (uint32_t a = 0; a < d; ++a) {
            row.clear();
            for (uint32_t i = 0; i <= k + 1; ++i) {
                if (fidx[i] == InternedMPSet::kNoIndex) continue;   // f := 0 on degenerate faces
                const uint64_t base = fidx[i] * d;
                if (i == 0 && twisted) {
                    for (uint32_t b = 0; b < d; ++b) {
                        const uint32_t v = rho[g1][a * d + b];
                        if (v) row.emplace_back(static_cast<uint32_t>(base + b), v);
                    }
                } else {
                    const uint32_t v = (i % 2 == 0) ? 1u : m - 1u;
                    row.emplace_back(static_cast<uint32_t>(base + a), v);
                }
            }
            std::sort(row.begin(), row.end());
            // merge equal columns (two faces may coincide)
            uint32_t lastc = UINT32_MAX, acc = 0;
            for (size_t t = 0; t <= row.size(); ++t) {
                if (t < row.size() && row[t].first == lastc) {
                    acc = (acc + row[t].second) % m;
                    continue;
                }
                if (lastc != UINT32_MAX && acc != 0) {
                    A.col.push_back(lastc);
                    A.val.push_back(static_cast<uint8_t>(acc));
                }
                if (t < row.size()) {
                    lastc = row[t].first;
                    acc = row[t].second % m;
                }
            }
            A.start.push_back(A.col.size());
        }
    }
    return A;
}

bool ProductIsZero(const SparseRows& B, const SparseRows& A, uint32_t m) {
    ENGINE_CHECK_MSG(B.ncols == A.rows(), "orbifold_mp::ProductIsZero: shapes");
    std::vector<uint32_t> acc(A.ncols, 0);
    std::vector<uint32_t> touched;
    for (uint64_t r = 0; r < B.rows(); ++r) {
        touched.clear();
        for (uint64_t t = B.start[r]; t < B.start[r + 1]; ++t) {
            const uint32_t c = B.col[t], v = B.val[t];
            for (uint64_t u = A.start[c]; u < A.start[c + 1]; ++u) {
                const uint32_t cc = A.col[u];
                if (acc[cc] == 0) touched.push_back(cc);
                acc[cc] = (acc[cc] + v * A.val[u]) % m;
                if (acc[cc] == 0) acc[cc] = m;   // keep "touched" distinct from zero; reduced below
            }
        }
        bool zero = true;
        for (uint32_t cc : touched) {
            if (acc[cc] % m != 0) zero = false;
            acc[cc] = 0;
        }
        if (!zero) return false;
    }
    return true;
}

bool FormalDSquared(const InternedMPSet& S, uint32_t q, uint64_t* checked) {
    ENGINE_CHECK_MSG(q + 2 <= S.k_max(), "orbifold_mp::FormalDSquared: S_{q+2} not built");
    *checked = 0;
    std::vector<std::pair<std::pair<uint32_t, uint64_t>, int64_t>> terms;
    for (uint64_t idx : S.nd(q + 2)) {
        terms.clear();
        const uint32_t g1 = S.gamma(q + 2, idx)[0];
        for (uint32_t i = 0; i <= q + 2; ++i) {
            const uint64_t t1 = S.face(q + 2, idx, i);
            if (S.nd_index(q + 1, t1) == InternedMPSet::kNoIndex) continue;
            const uint32_t a = (i == 0) ? g1 : S.identity();
            const int64_t si = (i % 2 == 0) ? 1 : -1;
            const uint32_t h1 = S.gamma(q + 1, t1)[0];
            for (uint32_t j = 0; j <= q + 1; ++j) {
                const uint64_t t2 = S.face(q + 1, t1, j);
                const uint64_t t2nd = S.nd_index(q, t2);
                if (t2nd == InternedMPSet::kNoIndex) continue;
                const uint32_t b = (j == 0) ? h1 : S.identity();
                const uint32_t g = S.mul(a, b);
                ENGINE_CHECK_MSG(g != InternedMPSet::kNone, "FormalDSquared: a twist product leaves the transporters");
                const int64_t sj = (j % 2 == 0) ? 1 : -1;
                terms.push_back({{g, t2nd}, si * sj});
            }
        }
        std::sort(terms.begin(), terms.end());
        for (size_t t = 0; t < terms.size();) {
            size_t u = t;
            int64_t sum = 0;
            while (u < terms.size() && terms[u].first == terms[t].first) sum += terms[u++].second;
            if (sum != 0) return false;
            t = u;
        }
        ++*checked;
    }
    return true;
}

}  // namespace orbifold_mp
