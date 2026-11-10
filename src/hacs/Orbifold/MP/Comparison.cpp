// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) Phi : Z[F'_.] -> Iso_. -- see Comparison.h.
#include "Comparison.h"

#include <algorithm>
#include <chrono>

#include "CochainMaps.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

using integral_model::Chain;
using integral_model::GKey;
using integral_model::GroupRingElt;
using integral_model::key_of;
using integral_model::mat_of;

namespace {

void chain_add(Chain& a, const Chain& b, int64_t scale) {
    ENGINE_CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (!b[i].is_zero()) a[i].add(b[i], scale);
}

Chain left_mul(const Mat22& g, const Chain& x) {
    const GroupRingElt u = GroupRingElt::unit(g);
    Chain out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        if (!x[i].is_zero()) out[i] = u * x[i];
    return out;
}

}  // namespace

uint32_t PhiLift::intern(const GKey& g) {
    const auto it = index_.find(g);
    if (it != index_.end()) return it->second;
    const uint32_t i = static_cast<uint32_t>(elems_.size());
    elems_.push_back(g);
    index_.emplace(g, i);
    return i;
}

void PhiLift::store(uint32_t k, const Chain& c) {
    for (uint32_t cell = 0; cell < c.size(); ++cell)
        for (const auto& [g, n] : c[cell].terms) terms_[k].push_back(Term{cell, intern(g), n});
    start_[k].push_back(terms_[k].size());
}

Chain PhiLift::value(uint32_t k, uint64_t j) const {
    Chain c(P_->model().rank(k));
    for (uint64_t t = start_.at(k).at(j); t < start_[k][j + 1]; ++t) {
        const Term& x = terms_[k][t];
        c[x.cell].add(elems_[x.g], x.c);
    }
    return c;
}

Chain PhiLift::boundary_image(uint32_t k, uint64_t j) const {
    const InternedMPSet& S = *S_;
    const uint64_t s = S.nd(k).at(j);
    Chain z(P_->model().rank(k - 1));
    for (uint32_t i = 0; i <= k; ++i) {
        const uint64_t f = S.nd_index(k - 1, S.face(k, s, i));
        if (f == InternedMPSet::kNoIndex) continue;   // degenerate face: 0 in the normalised complex
        const Chain v = value(k - 1, f);
        if (i == 0) {
            chain_add(z, left_mul(S.elements()[S.gamma(k, s)[0]], v), 1);
        } else {
            chain_add(z, v, (i % 2 == 0) ? 1 : -1);
        }
    }
    return z;
}

PhiLift::PhiLift(const InternedMPSet& S, const Descriptor& D, const integral_model::PeriodicityClass& P, uint32_t k_top)
    : S_(&S), P_(&P) {
    const auto t0 = std::chrono::steady_clock::now();
    const integral_model::IsotropyModel& M = P.model();
    ENGINE_CHECK_MSG(k_top <= S.k_max(), "PhiLift: S_{k_top} not built");
    ENGINE_CHECK_MSG(k_top <= M.q_top(), "PhiLift: the target model is not built to degree k_top");
    ENGINE_CHECK_MSG(D.top.size() == S.top_count(), "PhiLift: the descriptor is not the MP set's");
    start_.assign(k_top + 1, std::vector<uint64_t>{0});
    terms_.assign(k_top + 1, {});
    // Phi_0.
    for (uint64_t j = 0; j < S.count_nondegenerate(0); ++j) {
        const uint32_t t = S.sigma(0, S.nd(0)[j])[0];
        Chain c(M.rank(0));
        bool done = false;
        for (const Point& p : D.lifted(t)) {
            const uint32_t label = D.orbit_label.at(p.orbit);
            for (uint32_t r = 0; r < M.reps(0).size() && !done; ++r)
                if (M.reps(0)[r] == integral_model::Simplex{label}) {
                    c[M.cell_index(0, r, 0)].add(key_of(p.g), 1);
                    done = true;
                }
            if (done) break;
        }
        ENGINE_CHECK_MSG(done, "PhiLift: a top simplex has no vertex in an orbit of the target model");
        store(0, c);
    }
    // Phi_k, k >= 1.
    for (uint32_t k = 1; k <= k_top; ++k)
        for (uint64_t j = 0; j < S.count_nondegenerate(k); ++j) store(k, P.fill(k - 1, boundary_image(k, j)));
    build_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

PhiCertificate PhiLift::certify() const {
    PhiCertificate C;
    C.aug = true;
    C.chain_map = true;
    for (uint64_t j = 0; j < S_->count_nondegenerate(0); ++j) {
        ++C.generators_checked;
        if (P_->augmentation(value(0, j)) != 1) {
            if (C.aug && C.first_failure.empty()) C.first_failure = "aug Phi_0(e_s) != 1 at s = " + std::to_string(j);
            C.aug = false;
        }
    }
    for (uint32_t k = 1; k <= k_top(); ++k)
        for (uint64_t j = 0; j < S_->count_nondegenerate(k); ++j) {
            ++C.generators_checked;
            if (!(P_->apply_d(k, value(k, j)) == boundary_image(k, j))) {
                if (C.first_failure.empty())
                    C.first_failure = "d Phi_" + std::to_string(k) + "(e_s) != Phi(d e_s) at s = " + std::to_string(j);
                C.chain_map = false;
            }
        }
    return C;
}

void PhiLift::perturb_for_negative_control(uint32_t k, uint64_t j) {
    // Insert one term (cell 0, identity, +1) into Phi_k(e_s): shift the later offsets.
    const uint32_t id = intern(key_of(Mat22::I()));
    const uint64_t at = start_.at(k).at(j + 1);
    terms_[k].insert(terms_[k].begin() + static_cast<std::ptrdiff_t>(at), Term{0, id, 1});
    for (uint64_t t = j + 1; t < start_[k].size(); ++t) ++start_[k][t];
}

SparseRows PhiLift::Pullback(uint32_t q, uint32_t n, uint32_t ell, uint32_t e) const {
    ENGINE_CHECK_MSG(q <= k_top(), "PhiLift::Pullback: Phi_q not built");
    const uint32_t m = PowU(ell, e), d = n + 1;
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    const uint32_t w = P_->model().rank(q);
    std::vector<Vec> rho(elems_.size());
    SparseRows A;
    A.ncols = w * d;
    std::vector<uint32_t> acc(static_cast<size_t>(w) * d);
    const uint64_t N = S_->count_nondegenerate(q);
    for (uint64_t j = 0; j < N; ++j) {
        for (uint32_t a = 0; a < d; ++a) {
            std::fill(acc.begin(), acc.end(), 0u);
            for (uint64_t t = start_[q][j]; t < start_[q][j + 1]; ++t) {
                const Term& x = terms_[q][t];
                if (rho[x.g].empty()) {
                    const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx, mat_of(elems_[x.g]), n);
                    rho[x.g].resize(static_cast<size_t>(d) * d);
                    for (uint32_t r = 0; r < d; ++r)
                        for (uint32_t c = 0; c < d; ++c) rho[x.g][r * d + c] = R.at(r, c) % m;
                }
                int64_t cm = x.c % static_cast<int64_t>(m);
                if (cm < 0) cm += m;
                if (cm == 0) continue;
                for (uint32_t b = 0; b < d; ++b) {
                    uint32_t& v = acc[static_cast<size_t>(x.cell) * d + b];
                    v = static_cast<uint32_t>((v + static_cast<uint64_t>(cm) * rho[x.g][a * d + b]) % m);
                }
            }
            for (uint32_t c = 0; c < acc.size(); ++c)
                if (acc[c]) {
                    A.col.push_back(c);
                    A.val.push_back(static_cast<uint8_t>(acc[c]));
                }
            A.start.push_back(A.col.size());
        }
    }
    return A;
}

uint64_t PhiLift::total_terms() const {
    uint64_t t = 0;
    for (const auto& v : terms_) t += v.size();
    return t;
}

int64_t PhiLift::max_abs_coefficient() const {
    int64_t mx = 0;
    for (const auto& v : terms_)
        for (const Term& x : v) mx = std::max<int64_t>(mx, x.c < 0 ? -x.c : x.c);
    return mx;
}

}  // namespace orbifold_mp
