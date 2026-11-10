// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Line 2, M0 (hecke-extensions slot 9, WP-11): the integral groups from the
// isotropy-resolution model by the certified-exponent rule.  See the header;
// [C] = docs/latex/gl2_contract.tex (v1.8.1), cited by label.
#include "IntegralGroups.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_model {

// ---------------------------------------------------------------------------
// AbelianGroupType
// ---------------------------------------------------------------------------
void AbelianGroupType::add_torsion(uint32_t ell, uint32_t a) {
    ENGINE_CHECK_MSG(ell >= 2 && a >= 1, "AbelianGroupType::add_torsion: needs ell >= 2, a >= 1");
    auto& v = torsion[ell];
    v.push_back(a);
    std::sort(v.begin(), v.end(), std::greater<uint32_t>());
}

AbelianGroupType AbelianGroupType::primary_part(uint32_t ell) const {
    AbelianGroupType out;
    const auto it = torsion.find(ell);
    if (it != torsion.end()) out.torsion[ell] = it->second;
    return out;
}

std::string AbelianGroupType::to_string() const {
    if (is_zero()) return "0";
    std::ostringstream os;
    bool first = true;
    if (free_rank > 0) {
        os << "Z";
        if (free_rank > 1) os << "^" << free_rank;
        first = false;
    }
    for (const auto& [ell, exps] : torsion) {
        std::vector<uint32_t> asc(exps.rbegin(), exps.rend());
        for (uint32_t a : asc) {
            os << (first ? "" : " + ") << "Z/" << ell;
            if (a > 1) os << "^" << a;
            first = false;
        }
    }
    return os.str();
}

std::vector<uint64_t> AbelianGroupType::invariant_factors() const {
    size_t m = 0;
    for (const auto& kv : torsion) m = std::max(m, kv.second.size());
    std::vector<uint64_t> out;   // descending first, reversed below
    for (size_t j = 0; j < m; ++j) {
        uint64_t d = 1;
        for (const auto& [ell, exps] : torsion) {
            if (j >= exps.size()) continue;
            for (uint32_t i = 0; i < exps[j]; ++i) {
                ENGINE_CHECK_MSG(d <= (std::numeric_limits<uint64_t>::max() >> 1) / ell,
                                 "AbelianGroupType::invariant_factors: a factor exceeds 2^63");
                d *= ell;
            }
        }
        out.push_back(d);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::string AbelianGroupType::invariant_factor_string() const {
    if (is_zero()) return "0";
    std::ostringstream os;
    bool first = true;
    if (free_rank > 0) {
        os << "Z";
        if (free_rank > 1) os << "^" << free_rank;
        first = false;
    }
    for (uint64_t d : invariant_factors()) {
        os << (first ? "" : " + ") << "Z/" << d;
        first = false;
    }
    return os.str();
}

uint32_t AbelianGroupType::tor_length(uint32_t ell, uint32_t f) const {
    uint32_t s = 0;
    const auto it = torsion.find(ell);
    if (it != torsion.end())
        for (uint32_t a : it->second) s += std::min(a, f);
    return s;
}

uint32_t AbelianGroupType::tensor_length(uint32_t ell, uint32_t f) const {
    return f * free_rank + tor_length(ell, f);
}

// ---------------------------------------------------------------------------
// Certification
// ---------------------------------------------------------------------------
bool PrimeCertification::all_certified() const {
    for (const auto& x : d)
        if (!x.certified) return false;
    return true;
}

uint32_t PrimeCertification::K_max() const {
    uint32_t k = 0;
    for (const auto& x : d) k = std::max(k, x.K);
    return k;
}

bool ModelIntegralCohomology::all_certified() const {
    for (const auto& p : primes)
        if (!p.all_certified()) return false;
    return true;
}

std::vector<uint32_t> RationalRanksAtGoodPrime(const IsotropyModel& M, uint32_t n, uint32_t good_prime) {
    ENGINE_CHECK_MSG(good_prime >= 5 && good_prime > n + 1,
                     "RationalRanksAtGoodPrime: [C, prop:certifiede](4) needs ell' >= 5 and ell' > n + 1");
    const engine::ModCtx<uint32_t> ctx(good_prime);
    std::vector<uint32_t> r(M.q_max() + 1, 0);
    for (uint32_t q = 0; q <= M.q_max(); ++q) {
        engine::DenseBuf<uint32_t> D = M.coboundary(ctx, n, q);
        if (D.rows == 0 || D.cols == 0) continue;
        r[q] = engine::elim::rref(ctx, engine::view(D)).rank;
    }
    return r;
}

uint32_t DefaultKBudget(uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "DefaultKBudget: ell >= 2");
    uint32_t K = 0;
    uint64_t v = 1;
    while (v * ell < (uint64_t(1) << 31)) { v *= ell; ++K; }
    return K;
}

PrimeCertification CertifyAtPrime(const IsotropyModel& M, uint32_t n, uint32_t ell,
                                  const std::vector<uint32_t>& rank_Q, uint32_t K_budget) {
    ENGINE_CHECK_MSG(rank_Q.size() == M.q_max() + 1u, "CertifyAtPrime: rank_Q must cover q = 0 .. q_max");
    if (K_budget == 0) K_budget = DefaultKBudget(ell);
    ENGINE_CHECK_MSG(K_budget >= 2 && K_budget <= DefaultKBudget(ell), "CertifyAtPrime: K budget out of range");
    PrimeCertification out;
    out.ell = ell;
    out.K_budget = K_budget;
    for (uint32_t q = 0; q <= M.q_max(); ++q) {
        CertifiedDifferential c;
        c.q = q;
        c.ell = ell;
        c.rank_Q = rank_Q[q];
        for (uint32_t K = 2; K <= K_budget; ++K) {
            const auto ctx = engine::ModCtx<uint32_t>::prime_power(ell, K);
            const engine::DenseBuf<uint32_t> D = M.coboundary(ctx, n, q);
            std::vector<uint32_t> e;
            if (D.rows > 0 && D.cols > 0) e = engine::elim::invariant_factors<uint32_t>(ctx, engine::cview(D));
            const uint32_t rK = static_cast<uint32_t>(e.size());
            c.r_K_tried.push_back(rK);
            // [C, prop:certifiede](1): r_K(D) <= rank_Q D, always.
            ENGINE_CHECK_MSG(rK <= c.rank_Q, "CertifyAtPrime: r_K(d^q) > rank_Q(d^q) -- the good prime was not good");
            if (rK == c.rank_Q) {
                c.K = K;
                c.certified = true;
                for (uint32_t ei : e) {
                    ENGINE_CHECK(ei >= 1 && ei <= K);
                    if (ei < K) c.nonunit.push_back(K - ei);
                }
                std::sort(c.nonunit.begin(), c.nonunit.end(), std::greater<uint32_t>());
                break;
            }
        }
        out.d.push_back(std::move(c));
    }
    return out;
}

ModelIntegralCohomology AssembleFromModel(const IsotropyModel& M, uint32_t n, const std::vector<uint32_t>& primes,
                                          uint32_t good_prime) {
    ModelIntegralCohomology G;
    G.n = n;
    G.q_max = M.q_max();
    G.good_prime = good_prime;
    G.rank_Q = RationalRanksAtGoodPrime(M, n, good_prime);
    for (uint32_t q = 0; q <= M.q_top(); ++q) G.cochain_rank.push_back(M.rank(q) * (n + 1));
    G.H.assign(G.q_max + 1, AbelianGroupType{});
    for (uint32_t q = 0; q <= G.q_max; ++q) {
        const uint32_t sub = G.rank_Q[q] + (q > 0 ? G.rank_Q[q - 1] : 0u);
        ENGINE_CHECK_MSG(sub <= G.cochain_rank[q], "AssembleFromModel: negative rank");
        G.H[q].free_rank = G.cochain_rank[q] - sub;
    }
    for (uint32_t ell : primes) {
        PrimeCertification pc = CertifyAtPrime(M, n, ell, G.rank_Q);
        for (uint32_t q = 0; q <= G.q_max; ++q) {
            if (!pc.d[q].certified) continue;
            // tors H^{q+1} = the non-unit elementary divisors of d^q.
            AbelianGroupType& target = (q + 1 <= G.q_max) ? G.H[q + 1] : G.tors_H_top;
            for (uint32_t a : pc.d[q].nonunit) target.add_torsion(ell, a);
        }
        G.primes.push_back(std::move(pc));
    }
    return G;
}

std::vector<uint32_t> PredictedLengths(const ModelIntegralCohomology& G, uint32_t ell, uint32_t f) {
    std::vector<uint32_t> out(G.q_max + 1, 0);
    for (uint32_t q = 0; q <= G.q_max; ++q) {
        const AbelianGroupType& next = (q + 1 <= G.q_max) ? G.H[q + 1] : G.tors_H_top;
        out[q] = G.H[q].tensor_length(ell, f) + next.tor_length(ell, f);
    }
    return out;
}

std::vector<std::pair<uint32_t, uint32_t>> DicksonPrimes(uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (n == 0 || n % 2 != 0) return out;
    auto is_prime = [](uint32_t v) {
        if (v < 2) return false;
        for (uint32_t d = 2; d * d <= v; ++d)
            if (v % d == 0) return false;
        return true;
    };
    for (uint32_t ell = 5; ell + 1 <= n; ++ell) {
        if (!is_prime(ell)) continue;
        const uint64_t A = ell + 1, B = uint64_t(ell) * ell - ell;
        uint32_t count = 0;
        for (uint64_t b = 0; b * B <= n; ++b)
            if ((n - b * B) % A == 0) ++count;
        if (count > 0) out.emplace_back(ell, count);
    }
    return out;
}

}  // namespace integral_model
