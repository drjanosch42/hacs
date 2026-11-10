// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-4) Hecke operators on the MP model -- see MPHecke.h.
#include "MPHecke.h"

#include <algorithm>
#include <chrono>
#include <functional>

#include "CochainMaps.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

namespace orbifold_mp {

using integral_model::key_of;
using integral_model::mat_of;

namespace {

Vec RhoMod(const Mat22& g, uint32_t n, uint32_t ell, uint32_t e) {
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    const uint32_t m = PowU(ell, e), d = n + 1;
    const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx, g, n);
    Vec v(static_cast<size_t>(d) * d);
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c) v[static_cast<size_t>(r) * d + c] = R.at(r, c) % m;
    return v;
}

void AddScaled(StrChain& out, const StrChain& x, int64_t scale) {
    for (const auto& [s, n] : x) {
        auto it = out.find(s);
        const int64_t v = hecke::checked_mul<int64_t>(n, scale);
        if (it == out.end()) {
            if (v) out.emplace(s, v);
            continue;
        }
        it->second = hecke::checked_add<int64_t>(it->second, v);
        if (it->second == 0) out.erase(it);
    }
}

StrChain LeftMul(const Mat22& u, const StrChain& x) {
    StrChain out;
    for (const auto& [s, n] : x) {
        Str t = s;
        for (Tr& tr : t) tr.h = key_of(u * mat_of(tr.h));
        out.emplace(std::move(t), n);
    }
    return out;
}

}  // namespace

uint32_t MPHeckeDatum::intern(const GKey& g) {
    const auto it = index_.find(g);
    if (it != index_.end()) return it->second;
    const uint32_t i = static_cast<uint32_t>(elems_.size());
    elems_.push_back(g);
    index_.emplace(g, i);
    return i;
}

void MPHeckeDatum::store(uint32_t k, const StrChain& c) {
    for (const auto& [s, n] : c) {
        const auto ff = FreeForm(*S_, s);
        terms_[k].push_back(Term{ff.second, intern(ff.first), n});
    }
    start_[k].push_back(terms_[k].size());
}

StrChain MPHeckeDatum::value(uint32_t k, uint32_t j, uint64_t s) const {
    ENGINE_CHECK(k <= k_top() && j >= 1 && j <= m_);
    const uint64_t N = S_->count_nondegenerate(k);
    ENGINE_CHECK(s < N);
    const uint64_t slot = static_cast<uint64_t>(j - 1) * N + s;
    StrChain out;
    for (uint64_t t = start_[k][slot]; t < start_[k][slot + 1]; ++t) {
        const Term& x = terms_[k][t];
        const Str str = ConcreteString(*S_, k, x.s, mat_of(elems_[x.g]));
        auto it = out.find(str);
        if (it == out.end()) out.emplace(str, x.c);
        else {
            it->second = hecke::checked_add<int64_t>(it->second, x.c);
            if (it->second == 0) out.erase(it);
        }
    }
    return out;
}

StrChain MPHeckeDatum::on_translate(const Mat22& gamma, uint32_t k, uint64_t s) const {
    uint32_t j = 0;
    const Mat22 u = cosets_->twisted_conjugate(gamma, j);   // gamma = h s_j, u = g^{-1} h g
    return LeftMul(u, value(k, j, s));
}

StrChain MPHeckeDatum::boundary_image(uint32_t k, uint32_t j, uint64_t s) const {
    const InternedMPSet& S = *S_;
    const uint64_t idx = S.nd(k).at(s);
    const Mat22 sj = cosets_->s(j);
    StrChain z;
    for (uint32_t i = 0; i <= k; ++i) {
        const uint64_t f = S.nd_index(k - 1, S.face(k, idx, i));
        if (f == InternedMPSet::kNoIndex) continue;
        if (i == 0) {
            AddScaled(z, on_translate(sj * S.elements()[S.gamma(k, idx)[0]], k - 1, f), 1);
        } else {
            AddScaled(z, on_translate(sj, k - 1, f), (i % 2 == 0) ? 1 : -1);
        }
    }
    return z;
}

MPHeckeDatum::MPHeckeDatum(const InternedMPSet& S, const Descriptor& D, const hecke::HeckeCosetTablesT<uint32_t>& cosets,
                           uint64_t p, uint32_t k_top)
    : S_(&S), cosets_(&cosets), p_(p), m_(cosets.num_cosets()), fill_(D) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(cosets.N() == p && cosets.r() == 1, "MPHeckeDatum: the coset tables are not those of diag(1, p)");
    ENGINE_CHECK_MSG(k_top <= S.k_max(), "MPHeckeDatum: S_{k_top} not built");
    ENGINE_CHECK_MSG(D.top.size() == S.top_count(), "MPHeckeDatum: the descriptor is not the MP set's");
    start_.assign(k_top + 1, std::vector<uint64_t>{0});
    terms_.assign(k_top + 1, {});
    // tau_0(s_j e_s) = e_s.
    for (uint32_t j = 1; j <= m_; ++j)
        for (uint64_t s = 0; s < S.count_nondegenerate(0); ++s) {
            StrChain c;
            c.emplace(ConcreteString(S, 0, s, Mat22::I()), 1);
            store(0, c);
        }
    for (uint32_t k = 1; k <= k_top; ++k)
        for (uint32_t j = 1; j <= m_; ++j)
            for (uint64_t s = 0; s < S.count_nondegenerate(k); ++s) store(k, fill_.fill(k - 1, boundary_image(k, j, s)));
    build_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

MPHeckeCertificate MPHeckeDatum::certify() const {
    MPHeckeCertificate C;
    C.aug = C.chain_map = C.schreier = true;
    auto fail = [&](bool& flag, const std::string& what) {
        flag = false;
        if (C.first_failure.empty()) C.first_failure = what;
    };
    for (uint32_t j = 1; j <= m_; ++j)
        for (uint64_t s = 0; s < S_->count_nondegenerate(0); ++s) {
            ++C.generators;
            if (MPFill::augmentation(value(0, j, s)) != 1) fail(C.aug, "aug tau_0 != 1 at j=" + std::to_string(j));
        }
    for (uint32_t k = 1; k <= k_top(); ++k)
        for (uint32_t j = 1; j <= m_; ++j)
            for (uint64_t s = 0; s < S_->count_nondegenerate(k); ++s) {
                ++C.generators;
                if (!(fill_.boundary(value(k, j, s)) == boundary_image(k, j, s)))
                    fail(C.chain_map, "d tau_" + std::to_string(k) + " != tau d at j=" + std::to_string(j) + " s=" + std::to_string(s));
            }
    // (T1): the split gamma = h s_{j'} on the Schreier words s_j x, x in {T, S}, formed independently.
    const int64_t P = static_cast<int64_t>(p_);
    for (uint32_t j = 1; j <= m_; ++j)
        for (const Mat22& x : {Mat22::T(), Mat22::S()}) {
            ++C.schreier_checks;
            const Mat22 gamma = cosets_->s(j) * x;
            uint32_t jj = 0;
            const Mat22 u = cosets_->twisted_conjugate(gamma, jj);
            const Mat22 h = gamma * cosets_->s(jj).inv_unimodular();
            bool ok = h.det() == 1 && h.c % P == 0;
            if (ok) {
                const Mat22 want{h.a, h.b * P, h.c / P, h.d};
                ok = (u == want);
            }
            if (!ok) fail(C.schreier, "Schreier split at j=" + std::to_string(j));
        }
    return C;
}

void MPHeckeDatum::perturb_for_negative_control(uint32_t k, uint32_t j, uint64_t s) {
    const uint64_t N = S_->count_nondegenerate(k);
    const uint64_t slot = static_cast<uint64_t>(j - 1) * N + s;
    const uint32_t id = intern(key_of(Mat22::I()));
    const uint64_t at = start_.at(k).at(slot + 1);
    terms_[k].insert(terms_[k].begin() + static_cast<std::ptrdiff_t>(at), Term{s, id, 1});
    for (uint64_t t = slot + 1; t < start_[k].size(); ++t) ++start_[k][t];
}

SparseRows MPHeckeDatum::Assemble(uint32_t q, uint32_t n, uint32_t ell, uint32_t e, uint32_t d_out,
                                  const std::function<Mat22(uint32_t, const Mat22&)>& left, bool coind_rows) const {
    ENGINE_CHECK_MSG(q <= k_top(), "MPHeckeDatum: tau_q not built");
    const uint32_t m = PowU(ell, e), d = n + 1;
    ENGINE_CHECK(m <= 255);
    const uint64_t N = S_->count_nondegenerate(q);
    std::map<GKey, Vec> memo;
    SparseRows A;
    A.ncols = static_cast<uint32_t>(N * d);
    const uint32_t blocks = coind_rows ? m_ : 1;
    ENGINE_CHECK(d_out == d);
    for (uint64_t s = 0; s < N; ++s)
        for (uint32_t blk = 0; blk < blocks; ++blk) {
            // rows (s, blk, a), a = 0 .. d - 1
            std::vector<std::map<uint32_t, uint32_t>> rows(d);
            const uint32_t jlo = coind_rows ? blk + 1 : 1, jhi = coind_rows ? blk + 1 : m_;
            for (uint32_t j = jlo; j <= jhi; ++j) {
                const uint64_t slot = static_cast<uint64_t>(j - 1) * N + s;
                for (uint64_t t = start_[q][slot]; t < start_[q][slot + 1]; ++t) {
                    const Term& x = terms_[q][t];
                    const Mat22 L = left(j, mat_of(elems_[x.g]));
                    const GKey key = key_of(L);
                    auto it = memo.find(key);
                    if (it == memo.end()) it = memo.emplace(key, RhoMod(L, n, ell, e)).first;
                    int64_t cm = x.c % static_cast<int64_t>(m);
                    if (cm < 0) cm += m;
                    if (cm == 0) continue;
                    const Vec& R = it->second;
                    for (uint32_t a = 0; a < d; ++a)
                        for (uint32_t b = 0; b < d; ++b) {
                            const uint32_t r = R[static_cast<size_t>(a) * d + b];
                            if (!r) continue;
                            uint32_t& v = rows[a][static_cast<uint32_t>(x.s * d + b)];
                            v = static_cast<uint32_t>((v + static_cast<uint64_t>(cm) * r) % m);
                        }
                }
            }
            for (uint32_t a = 0; a < d; ++a) {
                for (const auto& [c, v] : rows[a])
                    if (v) {
                        A.col.push_back(c);
                        A.val.push_back(static_cast<uint8_t>(v));
                    }
                A.start.push_back(A.col.size());
            }
        }
    return A;
}

SparseRows MPHeckeDatum::Operator(uint32_t q, uint32_t n, uint32_t ell, uint32_t e, Factor f) const {
    const Mat22 g = cosets_->g();
    const Mat22 gT = g * Mat22::T();
    return Assemble(
        q, n, ell, e, n + 1,
        [&](uint32_t j, const Mat22& gamma) {
            switch (f) {
            case Factor::Label: return cosets_->delta(j) * gamma;               // s_j^{-1} g gamma
            case Factor::Dropped: return cosets_->s_inv(j) * gamma;             // s_j^{-1} gamma
            case Factor::WrongRep: return cosets_->s_inv(j) * gT * gamma;       // s_j^{-1} (g T) gamma
            }
            return cosets_->delta(j) * gamma;
        },
        false);
}

SparseRows MPHeckeDatum::CgRes(uint32_t q, uint32_t n, uint32_t ell, uint32_t e, bool drop_factor) const {
    const Mat22 g = cosets_->g();
    return Assemble(
        q, n, ell, e, n + 1, [&](uint32_t, const Mat22& gamma) { return drop_factor ? gamma : g * gamma; }, true);
}

ModuleRep MPHeckeDatum::Coind(uint32_t n, uint32_t ell, uint32_t e) const {
    const uint32_t d = n + 1, D = m_ * d;
    const hecke::HeckeCosetTablesT<uint32_t>* cs = cosets_;
    const uint32_t mm = m_;
    return ModuleRep("Coind_G0(" + std::to_string(p_) + ") Sym^" + std::to_string(n), D, ell, e,
                     [cs, mm, d, D, n, ell, e](const Mat22& gamma) {
                         Vec out(static_cast<size_t>(D) * D, 0);
                         for (uint32_t j = 1; j <= mm; ++j) {
                             const Mat22 x = cs->s(j) * gamma;
                             uint32_t jj = 0;
                             (void)cs->twisted_conjugate(x, jj);
                             const Mat22 h = x * cs->s_inv(jj);   // s_j gamma = h s_jj, h in Gamma_0(p)
                             const Vec R = RhoMod(h, n, ell, e);
                             for (uint32_t a = 0; a < d; ++a)
                                 for (uint32_t b = 0; b < d; ++b)
                                     out[static_cast<size_t>((j - 1) * d + a) * D + (jj - 1) * d + b] = R[a * d + b];
                         }
                         return out;
                     });
}

SparseRows MPHeckeDatum::Cor(uint32_t q, uint32_t n, uint32_t ell, uint32_t e) const {
    const uint32_t d = n + 1, D = m_ * d, m = PowU(ell, e);
    Vec phi(static_cast<size_t>(d) * D, 0);
    for (uint32_t j = 1; j <= m_; ++j) {
        const Vec R = RhoMod(cosets_->s_inv(j), n, ell, e);
        for (uint32_t a = 0; a < d; ++a)
            for (uint32_t b = 0; b < d; ++b) phi[static_cast<size_t>(a) * D + (j - 1) * d + b] = R[a * d + b];
    }
    return CoefficientMap(*S_, q, phi, d, D, m);
}

uint64_t MPHeckeDatum::total_terms() const {
    uint64_t t = 0;
    for (const auto& v : terms_) t += v.size();
    return t;
}

int64_t MPHeckeDatum::max_abs_coefficient() const {
    int64_t mx = 0;
    for (const auto& v : terms_)
        for (const Term& x : v) mx = std::max<int64_t>(mx, x.c < 0 ? -x.c : x.c);
    return mx;
}

}  // namespace orbifold_mp
