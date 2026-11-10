// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeMazurEis -- slot-25 H3 (see the header for the mathematics and the trust boundary).

#include "HeckeMazurEis.h"

#include "HeckeZRational.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeLevelNR3.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace heckez {

using u64 = uint64_t;
using u128 = unsigned __int128;

namespace {

u64 MulQ(u64 a, u64 b, u64 Q) { return static_cast<u64>(static_cast<u128>(a) * b % Q); }

uint32_t ValQ(u64 x, u64 ell, uint32_t K) {   // v_ell(x) for x mod ell^K (K when x = 0)
    if (x == 0) return K;
    uint32_t v = 0;
    while (x % ell == 0) { x /= ell; ++v; }
    return v;
}

using Poly = std::vector<u64>;   // low to high

Poly MulPoly(const Poly& a, const Poly& b, u64 Q) {
    if (a.empty() || b.empty()) return {};
    Poly c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        if (!a[i]) continue;
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + MulQ(a[i], b[j], Q)) % Q;
    }
    return c;
}

// Per word prime and candidate: the residues of c_q (g + 1 coefficients, low to high) and whether the structure held.
struct PassAtPrime {
    std::vector<std::vector<uint32_t>> c;   // per candidate
    uint32_t structure_failures = 0;
};

PassAtPrime OnePrime(uint32_t N, uint32_t g, const std::vector<uint32_t>& qs, uint32_t ellp) {
    const PrimeRun r = RunAtPrime(N, 0, qs, ellp);
    ENGINE_CHECK_MSG(r.h1 == 2 * g + 1, "MazurEisStep: dim H^1(Gamma_0(N); F_ell') = 2g + 1 at every word prime");
    PassAtPrime out;
    const u64 P = ellp;
    for (size_t k = 0; k < qs.size(); ++k) {
        std::vector<uint32_t> coeffs(qs.size(), 0);
        coeffs[k] = 1;
        const std::vector<u64> chi = CharpolyOfCombination(r, coeffs);   // low to high, degree 2g + 1
        ENGINE_CHECK_MSG(chi.size() == size_t(2 * g + 2) && chi.back() % P == 1, "MazurEisStep: chi(T_q) monic of degree 2g + 1");
        // divide by (x - a), a = 1 + q
        const u64 a = (1 + static_cast<u64>(qs[k])) % P;
        const size_t d = chi.size() - 1;   // 2g + 1
        Poly G(d, 0);
        G[d - 1] = chi[d] % P;
        for (size_t i = d - 1; i >= 1; --i) G[i - 1] = (chi[i] % P + a * G[i]) % P;
        const u64 rem = (chi[0] % P + a * G[0]) % P;
        bool ok = rem == 0;
        // the monic square root, top-down (2 invertible): h_g = 1, h_{g-k} = (G_{2g-k} - sum_{g-k<i,j<g} h_i h_j) / 2
        std::vector<u64> h(g + 1, 0);
        h[g] = 1;
        const u64 inv2 = (P + 1) / 2;
        for (uint32_t kk = 1; kk <= g; ++kk) {
            const uint32_t deg = 2 * g - kk;
            u64 S = 0;
            for (uint32_t i = g - kk + 1; i <= g - 1; ++i) {
                const uint32_t j = deg - i;
                if (j <= g - kk || j >= g) continue;
                S = (S + h[i] * h[j]) % P;
            }
            h[g - kk] = (G[deg] + P - S) % P * inv2 % P;
        }
        // check h^2 = G
        if (ok) {
            std::vector<u64> sq(2 * g + 1, 0);
            for (uint32_t i = 0; i <= g; ++i)
                for (uint32_t j = 0; j <= g; ++j) sq[i + j] = (sq[i + j] + h[i] * h[j]) % P;
            for (size_t i = 0; i < sq.size(); ++i)
                if (sq[i] != G[i]) { ok = false; break; }
        }
        if (!ok) ++out.structure_failures;
        out.c.emplace_back(h.begin(), h.end());
    }
    return out;
}

bool GoodIndependent(uint32_t N, uint32_t ell, uint32_t q) {   // two modular powers (05 Sec. 3.5, P2a-16)
    if (q == N) return false;
    return q % ell != 1 && PowModU64(q % N, (N - 1) / ell, N) != 1;
}

}  // namespace

std::string MazurEisSettings::canonical_text() const {
    std::ostringstream o;
    o << "batch=" << batch << " max_candidates=" << max_candidates << " q_max=" << q_max;
    if (!forced_q.empty()) {
        o << " forced_q=";
        for (size_t i = 0; i < forced_q.size(); ++i) o << (i ? "," : "") << forced_q[i];
    }
    return o.str();
}

std::string MazurEisResult::text() const {
    std::ostringstream o;
    o << "mazur_eis/v1 N=" << N << " ell=" << ell << " nu=" << nu << " v_nu=" << v_nu << " g=" << g << " b1=" << b1 << " K=" << K;
    o << " candidates=";
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        o << (i ? "," : "") << c.q << (c.good ? "g" : "b") << ":e" << c.e << ":v" << c.v_f << (c.accepted ? ":A" : "");
    }
    o << " q=" << q << " e=" << e << " F=[";
    for (size_t i = 0; i < F.size(); ++i) o << (i ? "," : "") << F[i];
    o << "] c=[";
    for (size_t i = 0; i < c_hex.size(); ++i) o << (i ? "," : "") << c_hex[i];
    o << "] dimW=" << dimW << " loewy=" << loewy << " e_W=" << e_W << " ln6_qs=";
    for (size_t i = 0; i < ln6_qs.size(); ++i) o << (i ? "," : "") << ln6_qs[i];
    o << " merel=" << (merel ? 1 : 0) << " newton=" << newton << " embdim=" << embdim << " irreducible=" << irreducible
      << " structure_failures=" << structure_failures << " first_good_rejected=" << (first_good_rejected ? 1 : 0)
      << " a_ok=" << a_ok << " b=" << (b_applicable ? (b_ok ? "ok" : "FAIL") : "n/a") << " c_ok=" << c_ok
      << " good_q_ok=" << good_q_ok << " settings{" << settings << "}";
    return o.str();
}

uint64_t MazurEisResult::digest() const { return engine::FnvBytes(engine::kFnvOffset64, text()); }

MazurEisResult MazurEisStep(uint32_t N, uint32_t ell, const MazurEisSettings& s) {
    if (N < 11 || !hecke::IsPrimeU64(N)) throw std::invalid_argument("MazurEisStep: N prime >= 11");
    if (ell < 5 || !hecke::IsPrimeU64(ell)) throw std::invalid_argument("MazurEisStep: ell >= 5 prime");
    MazurEisResult R;
    R.N = N;
    R.ell = ell;
    R.settings = s.canonical_text();
    {
        const u64 g12 = std::gcd<u64>(N - 1, 12);
        R.nu = (N - 1) / g12;
    }
    R.v_nu = hecke::level::NumConditionValuation(N, ell);
    if (R.v_nu == 0) throw std::invalid_argument("MazurEisStep: ell | num((N - 1)/12)");
    const int64_t gs = DimSk(2, N);
    ENGINE_CHECK(gs >= 1);
    R.g = static_cast<uint32_t>(gs);
    R.b1 = 2 * R.g + 1;
    // precision: ell^K < 2^62
    u64 Q = 1;
    R.K = 0;
    while (static_cast<u128>(Q) * ell < (static_cast<u128>(1) << 62)) { Q *= ell; ++R.K; }
    const uint32_t g = R.g;

    // ---- [C, LN6]'s reading (check (a)) and Merel [C, LN5]
    {
        const auto qs = hecke::level::R3HeckePrimes(N, 7);
        const auto rd = hecke::level::ReadEisensteinR3Scan(N, ell, qs);
        R.dimW = rd.dimW;
        R.loewy = rd.loewy;
        R.ln6_qs = rd.qs;
        R.e_W = rd.dimW >= 1 ? (rd.dimW - 1) / 2 : 0;
        R.merel = hecke::level::MerelCriterionLN5(N, ell);
    }

    // ---- candidates
    std::vector<uint32_t> pool;
    if (!s.forced_q.empty()) {
        pool = s.forced_q;
    } else {
        for (uint32_t q = 2; q < s.q_max && pool.size() < s.max_candidates; ++q)
            if (hecke::IsPrimeU64(q) && q != N && hecke::level::MazurGoodPrime(N, ell, q)) pool.push_back(q);
    }
    bool first_good_seen = false;
    std::vector<std::vector<uint32_t>> acc_res;   // residues of the accepted c
    std::vector<uint32_t> acc_primes;
    uint64_t acc_bits = 0;
    for (size_t from = 0; from < pool.size() && R.accepted_index < 0; from += s.batch) {
        const size_t to = std::min(pool.size(), from + std::max<uint32_t>(s.batch, 1));
        const std::vector<uint32_t> qs(pool.begin() + from, pool.begin() + to);
        const uint32_t qmax = *std::max_element(qs.begin(), qs.end());
        // |c_i| <= binom(g, i) (2 sqrt q)^i <= 2^g (2 sqrt q)^g
        const uint64_t bound_bits = g + static_cast<uint64_t>(std::ceil(g * (1.0 + 0.5 * std::log2(double(qmax))))) + 2;
        uint32_t count = static_cast<uint32_t>(bound_bits / 29 + 2);
        std::vector<uint32_t> primes;
        for (;;) {
            primes = WordPrimes(count, N);
            if (CoveredBits(primes, primes.size()) >= bound_bits + 1) break;
            ++count;
        }
        // word primes in parallel (index-ordered storage)
        std::vector<PassAtPrime> runs(primes.size());
        {
            std::atomic<size_t> next(0);
            const uint32_t T = std::max<uint32_t>(s.threads, 1);
            std::vector<std::exception_ptr> errs(T);
            auto work = [&](uint32_t w) {
                try {
                    for (;;) {
                        const size_t i = next.fetch_add(1);
                        if (i >= primes.size()) break;
                        runs[i] = OnePrime(N, g, qs, primes[i]);
                    }
                } catch (...) {
                    errs[w] = std::current_exception();
                }
            };
            std::vector<std::thread> pool_t;
            for (uint32_t w = 1; w < T; ++w) pool_t.emplace_back(work, w);
            work(0);
            for (auto& t : pool_t) t.join();
            for (auto& e : errs)
                if (e) std::rethrow_exception(e);
        }
        R.primes_used += static_cast<uint32_t>(primes.size());
        for (const auto& r : runs) R.structure_failures += r.structure_failures;

        for (size_t k = 0; k < qs.size(); ++k) {
            const uint32_t q = qs[k];
            MazurEisCandidate cand;
            cand.q = q;
            cand.good = hecke::level::MazurGoodPrime(N, ell, q);
            // c_q mod ell^K
            Poly c(g + 1, 0);
            std::vector<std::vector<uint32_t>> res_all(g + 1);
            for (uint32_t i = 0; i <= g; ++i) {
                std::vector<uint32_t> res(primes.size());
                for (size_t j = 0; j < primes.size(); ++j) res[j] = runs[j].c[k][i];
                const ZExact z = ZFromResidues(primes, res, bound_bits);
                const u64 am = z.abs_mod(Q);
                c[i] = z.negative ? (Q - am) % Q : am;
                res_all[i] = std::move(res);
            }
            ENGINE_CHECK_MSG(c[g] == 1, "MazurEisStep: c_q monic");
            // C(y) = c(y + 1 + q) mod Q (Horner)
            Poly C(g + 1, 0);
            const u64 a = (1 + static_cast<u64>(q)) % Q;
            for (int i = static_cast<int>(g); i >= 0; --i) {
                // C <- C * (y + a) + c_i
                for (int j = static_cast<int>(g); j >= 1; --j) C[j] = (C[j - 1] + MulQ(C[j], a, Q)) % Q;
                C[0] = (MulQ(C[0], a, Q) + c[i]) % Q;
            }
            ENGINE_CHECK(C[g] == 1);
            uint32_t e = 0;
            while (e < g && C[e] % ell == 0) ++e;
            cand.e = e;
            Poly F;
            if (e == 0) {
                F = {1};
            } else {
                // H0 = (C mod ell) / y^e over F_ell; t = H0^{-1} mod y^e; s = (1 - t H0)/y^e
                const u64 L = ell;
                Poly H0(g - e + 1);
                for (uint32_t i = 0; i + e <= g; ++i) H0[i] = C[i + e] % L;
                ENGINE_CHECK(H0[0] != 0);
                Poly t(e, 0);
                const u64 inv0 = PowModU64(H0[0], L - 2, L);
                for (uint32_t i = 0; i < e; ++i) {
                    u64 acc = (i == 0) ? 1 : 0;
                    for (uint32_t j = 1; j <= i && j < H0.size(); ++j) acc = (acc + L * L - H0[j] * t[i - j] % L) % L;
                    t[i] = acc * inv0 % L;
                }
                Poly tH = MulPoly(t, H0, L);
                Poly one_m(tH.size(), 0);
                for (size_t i = 0; i < tH.size(); ++i) one_m[i] = (L - tH[i]) % L;
                one_m[0] = (one_m[0] + 1) % L;
                for (uint32_t i = 0; i < e; ++i) ENGINE_CHECK(one_m[i] == 0);
                Poly sP(one_m.begin() + e, one_m.end());
                // lifting: F = y^e, Hh = H0 (mod ell), j = 1 .. K - 1
                F.assign(e + 1, 0);
                F[e] = 1;
                Poly Hh(H0.begin(), H0.end());
                u64 lj = 1;
                for (uint32_t j = 1; j < R.K; ++j) {
                    lj *= L;   // ell^j
                    const Poly FH = MulPoly(F, Hh, Q);
                    Poly E(g + 1, 0);
                    for (uint32_t i = 0; i <= g; ++i) {
                        const u64 D = (C[i] + Q - (i < FH.size() ? FH[i] : 0)) % Q;
                        ENGINE_CHECK_MSG(D % lj == 0, "MazurEisStep: Hensel step (C = F H mod ell^j)");
                        E[i] = (D / lj) % L;
                    }
                    const Poly tE = MulPoly(t, E, L);
                    Poly Rr(e, 0), Qp;
                    for (size_t i = 0; i < tE.size(); ++i) {
                        if (i < e) Rr[i] = tE[i];
                        else Qp.push_back(tE[i]);
                    }
                    Poly dh = MulPoly(sP, E, L);
                    const Poly QH = MulPoly(Qp, H0, L);
                    if (QH.size() > dh.size()) dh.resize(QH.size(), 0);
                    for (size_t i = 0; i < QH.size(); ++i) dh[i] = (dh[i] + QH[i]) % L;
                    for (size_t i = g - e; i < dh.size(); ++i) ENGINE_CHECK_MSG(dh[i] == 0, "MazurEisStep: Hensel degree bound");
                    for (uint32_t i = 0; i < e; ++i) F[i] = (F[i] + MulQ(Rr[i], lj, Q)) % Q;
                    for (size_t i = 0; i < g - e && i < dh.size(); ++i) Hh[i] = (Hh[i] + MulQ(dh[i], lj, Q)) % Q;
                }
                // re-check: F monic, F = y^e mod ell, F Hh = C mod ell^K, Hh(0) a unit
                const Poly FH = MulPoly(F, Hh, Q);
                for (uint32_t i = 0; i <= g; ++i) ENGINE_CHECK_MSG(FH[i] == C[i], "MazurEisStep: F * H = C mod ell^K");
                for (uint32_t i = 0; i < e; ++i) ENGINE_CHECK(F[i] % L == 0);
                ENGINE_CHECK(F[e] == 1 && Hh[0] % L != 0);
            }
            cand.v_f = e == 0 ? 0 : ValQ(F[0], ell, R.K);
            cand.accepted = cand.good && e >= 1 && cand.v_f == R.v_nu;
            if (cand.good && !first_good_seen) {
                first_good_seen = true;
                if (!cand.accepted) R.first_good_rejected = true;
            }
            R.candidates.push_back(cand);
            if (cand.accepted && R.accepted_index < 0) {
                R.accepted_index = static_cast<int32_t>(R.candidates.size() - 1);
                R.q = q;
                R.F = F;
                R.e = e;
                acc_res = res_all;
                acc_primes = primes;
                acc_bits = bound_bits;
            }
            if (!s.forced_q.empty() && R.accepted_index < 0 && R.F.empty()) {
                // the negative control reports the first forced candidate's F (an accepted one replaces it)
                R.F = F;
                R.e = e;
            }
        }
    }
    if (R.accepted_index >= 0) {
        for (uint32_t i = 0; i <= g; ++i) R.c_hex.push_back(ZFromResidues(acc_primes, acc_res[i], acc_bits).hex());
    }

    // ---- the checks
    R.c_ok = R.accepted_index >= 0;
    if (R.c_ok) {
        R.good_q_ok = GoodIndependent(N, ell, R.q) == hecke::level::MazurGoodPrime(N, ell, R.q) && GoodIndependent(N, ell, R.q);
        R.a_ok = R.e == R.e_W && R.dimW == 2 * R.e_W + 1 && ((R.e >= 2) == R.merel);
    }
    // (d) Newton polygon, embedding dimension, irreducibility -- of the reported F
    if (!R.F.empty() && R.e >= 1) {
        const uint32_t e = R.e;
        R.newton_v.resize(e + 1);
        for (uint32_t i = 0; i <= e; ++i) R.newton_v[i] = ValQ(R.F[i], ell, R.K);
        R.b_applicable = R.v_nu == 1;
        bool eis = R.newton_v[0] == 1;
        for (uint32_t i = 1; i < e; ++i) eis = eis && R.newton_v[i] >= 1;
        if (R.b_applicable) R.b_ok = R.c_ok && eis;
        R.embdim = (e == 1 || R.newton_v[0] == 1) ? 1 : 2;
        // lower convex hull from (0, v0) to (e, 0)
        std::vector<uint32_t> hull{0};
        uint32_t cur = 0;
        while (cur < e) {
            uint32_t best = cur + 1;
            for (uint32_t j = cur + 1; j <= e; ++j) {
                // slope (v_j - v_cur)/(j - cur) minimal; ties -> farthest
                const int64_t lhs = (int64_t(R.newton_v[j]) - R.newton_v[cur]) * int64_t(best - cur);
                const int64_t rhs = (int64_t(R.newton_v[best]) - R.newton_v[cur]) * int64_t(j - cur);
                if (lhs < rhs || (lhs == rhs && j > best)) best = j;
            }
            hull.push_back(best);
            cur = best;
        }
        std::ostringstream o;
        for (size_t i = 0; i < hull.size(); ++i) o << (i ? "-" : "") << "(" << hull[i] << "," << R.newton_v[hull[i]] << ")";
        R.newton = o.str();
        if (e == 1) R.irreducible = "yes";
        else if (hull.size() > 2) R.irreducible = "no";
        else R.irreducible = std::gcd<uint32_t>(R.newton_v[0], e) == 1 ? "yes" : "undecided";
    } else {
        R.newton = "-";
        R.irreducible = "-";
    }
    return R;
}

std::vector<MazurEisCell> MazurEisCellList() {
    std::vector<MazurEisCell> out;
    const uint32_t ells_ln6[] = {5, 7, 11, 13};
    for (uint32_t N = 11; N <= 2000; ++N) {
        if (!hecke::IsPrimeU64(N)) continue;
        const u64 nu = (N - 1) / std::gcd<u64>(N - 1, 12);
        bool sq = false;
        std::vector<uint32_t> ells;
        for (uint32_t l = 5; l <= nu; ++l) {
            if (!hecke::IsPrimeU64(l) || nu % l) continue;
            ells.push_back(l);
            if (nu % (u64(l) * l) == 0) sq = true;
        }
        const bool c4N = !ells.empty() && (N < 200 || (N < 900 && sq));
        for (uint32_t l : ells) {
            const bool ln6 = std::find(std::begin(ells_ln6), std::end(ells_ln6), l) != std::end(ells_ln6);
            if (c4N || ln6) out.push_back({N, l, c4N, ln6});
        }
    }
    return out;
}

}  // namespace heckez
