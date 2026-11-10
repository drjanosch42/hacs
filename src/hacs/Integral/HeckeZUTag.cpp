// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZUTag -- slot-25 HZ-UTAG (see the header).

#include "HeckeZUTag.h"

#include "IntegralCharpolyCrt.h"
#include "Descent/DescentModuleExport.h"
#include "../Hecke/HeckeLevelNHeilbronn.h"
#include "../Hecke/HeckeLevelNSparse.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace heckez {

namespace {
using u64 = uint64_t;
using u128 = unsigned __int128;

u64 Gcd(u64 a, u64 b) { while (b) { const u64 t = a % b; a = b; b = t; } return a; }
u64 ResMod(int64_t r, u64 ell) {
    const int64_t m = r % static_cast<int64_t>(ell);
    return static_cast<u64>(m < 0 ? m + static_cast<int64_t>(ell) : m);
}
// synthetic division by (x - r) over F_ell: quotient (low to high) and the remainder q(r)
std::vector<u64> DivLinear(const std::vector<u64>& a, u64 r, u64 ell, u64* rem) {
    if (a.empty()) { *rem = 0; return {}; }
    std::vector<u64> q(a.size() - 1);
    u64 acc = 0;
    for (size_t i = a.size(); i-- > 0;) {
        acc = (u128(acc) * r + a[i]) % ell;
        if (i > 0) q[i - 1] = acc;
    }
    *rem = acc;
    return q;
}

// fn(i) for i in [from, to) on up to T workers (dynamic schedule, index-owned outputs: no value depends on T); the first exception of
// any worker is rethrown after the join; a worker that cannot start leaves its share to the others
template <class F>
void ParallelForEach(size_t from, size_t to, uint32_t T, F fn) {
    if (to <= from) return;
    T = std::max<uint32_t>(1, std::min<uint32_t>(T, static_cast<uint32_t>(std::min<size_t>(to - from, size_t(1) << 16))));
    std::atomic<size_t> next(from);
    std::vector<std::exception_ptr> errs(T);
    auto work = [&](uint32_t w) {
        try {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= to) break;
                fn(i);
            }
        } catch (...) {
            errs[w] = std::current_exception();
            next.store(to);
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) {
        try {
            pool.emplace_back(work, w);
        } catch (const std::system_error&) {
            break;
        }
    }
    work(0);
    for (auto& t : pool) t.join();
    for (auto& e : errs)
        if (e) std::rethrow_exception(e);
}

// y = M x over F_ell for a column-major D x D matrix Mc (entries < ell) and a canonical sparse x; the result canonical (ascending
// columns, no zero entry) -- the same vector, in the same form, as the term route's ApplyWith (sparse::Acc::take)
hecke::level::SparseVec ApplyDenseCols(const std::vector<uint32_t>& Mc, uint32_t D, const hecke::level::SparseVec& x, u64 ell,
                                       std::vector<u64>& acc) {
    std::fill(acc.begin(), acc.end(), 0);
    for (size_t k = 0; k < x.col.size(); ++k) {
        const u64 a = x.val[k];
        const uint32_t* col = &Mc[size_t(x.col[k]) * D];
        for (uint32_t r = 0; r < D; ++r)
            if (col[r]) acc[r] = (acc[r] + a * col[r]) % ell;
    }
    hecke::level::SparseVec y;
    for (uint32_t r = 0; r < D; ++r)
        if (acc[r]) {
            y.col.push_back(r);
            y.val.push_back(static_cast<uint32_t>(acc[r]));
        }
    return y;
}
}  // namespace

// ---- (HZ-RAT-PERF fix round 1, audit F1) U_p at the word primes (header note on UTagRoute) ------------------------------------------
class UTagOperator {
public:
    UTagOperator(uint32_t N, uint32_t n, uint32_t p, uint32_t ell0, uint32_t threads, UTagRoute route, size_t count_slice_terms = 0)
        : N_(N), n_(n), p_(p), lvl_(std::make_unique<hecke::level::Gamma0LevelN>(N)) {
        const uint32_t m = lvl_->m(), d = n + 1;
        D_ = m * d;
        if (route == UTagRoute::kTermTable) return;
        mer_ = hecke::level::MerelSet(p);
        // the dense routes pay D^2 per prime and generator column; the term route |Mer_p| m_N d^2: dense once |Mer_p| > m_N
        const bool many = mer_.size() > m;
        const uint64_t DD4 = uint64_t(D_) * D_ * 4;
        const uint32_t T = std::max<uint32_t>(threads, 1);
        if ((route == UTagRoute::kAuto || route == UTagRoute::kCountOnce) && n == 0 && many && DD4 <= (512ull << 20)) {
            route_ = UTagRoute::kCountOnce;
            BuildCounts(ell0, T, count_slice_terms ? count_slice_terms : (size_t(1) << 20));
        } else if ((route == UTagRoute::kAuto || route == UTagRoute::kDensePerPrime) && n > 0 && many &&
                   DD4 * 2 * T <= (512ull << 20)) {
            route_ = UTagRoute::kDensePerPrime;
            const uint64_t per_mat = uint64_t(m) * (448 + 12ull * d * d);
            slice_ = static_cast<size_t>(std::max<uint64_t>(1, (32ull << 20) / per_mat));
        }
    }
    UTagRoute route() const { return route_; }
    UTagCountInfo count_info() const { return info_; }
    // the worker count for the primes: the term route holds one whole term table per worker, so beyond 256 MiB of terms (the export's
    // kExportTermBudget) its primes run one at a time, as before the fix
    uint32_t prime_threads(uint32_t threads) const {
        const uint64_t d = n_ + 1, bytes = uint64_t(mer_.size()) * (D_ / d) * (448 + 12 * d * d);
        return route_ == UTagRoute::kTermTable && (mer_.empty() || bytes > (256ull << 20)) ? 1u : std::max<uint32_t>(threads, 1);
    }

    // U_p on H^1(Gamma_0(N); Sym^n F_ell) in the free-column basis of the sparse presentation: RunAtPrime(N, n, {p}, ell) entrywise
    PrimeRun run(uint32_t ell) const {
        if (route_ == UTagRoute::kTermTable) return RunAtPrime(N_, n_, {p_}, ell);
        ENGINE_CHECK_MSG(ell > n_ + 1 && ell >= 5 && N_ % ell != 0, "UTagOperator: ell' > n + 1, ell' >= 5, ell' prime to N");
        const engine::ModCtx<uint32_t> ctx(ell);
        std::vector<uint32_t> Mc;
        const std::vector<uint32_t>* M = &counts_;   // kCountOnce: the integer counts, each < 2^29 < ell (checked at the build)
        if (route_ == UTagRoute::kDensePerPrime) {
            const std::vector<uint32_t> R = integral_descent::modexport::HeilbronnDense(*lvl_, p_, n_, ctx, mer_, slice_, 1);
            Mc.resize(R.size());
            for (size_t r = 0; r < D_; ++r)
                for (size_t c = 0; c < D_; ++c) Mc[c * D_ + r] = R[r * D_ + c];
            M = &Mc;
        }
        const hecke::level::SparseCokernel Q = hecke::level::SparseCokernel::Build(*lvl_, n_, ctx, true);
        ENGINE_CHECK_MSG(Q.D() == D_, "UTagOperator: presentation and operator of different size");
        std::vector<u64> acc(D_, 0);
        // prop:heilbronn (1) on every generator, as CokernelHeckeMatrices checks it on the term route
        uint32_t bad = 0;
        for (const auto& g : Q.generators()) bad += Q.in_rel(ApplyDenseCols(*M, D_, g, ell, acc)) ? 0 : 1;
        ENGINE_CHECK_MSG(bad == 0, "UTagOperator: H(Rel) not in Rel (prop:heilbronn (1))");
        const uint32_t h = Q.h1();
        PrimeRun r;
        r.ell = ell;
        r.h1 = h;
        std::vector<u64> A(size_t(h) * h, 0);
        for (uint32_t j = 0; j < h; ++j) {
            hecke::level::SparseVec e;
            e.col.push_back(Q.free_columns()[j]);
            e.val.push_back(1);
            const std::vector<uint32_t> c = Q.coords(ApplyDenseCols(*M, D_, e, ell, acc));
            for (uint32_t i = 0; i < h; ++i) A[size_t(i) * h + j] = c[i] % ell;
        }
        r.mats.push_back(std::move(A));
        return r;
    }

private:
    // n = 0: rho_0 = (1), so H_{N,p} is the integer matrix counting the admissible pairs (eps, j) with j.eps = i -- prime-independent.
    // Built ONCE from the partial operators over slices of Mer_p (about 2^20 terms each) on up to T workers (one count matrix each),
    // every block checked to be (1); stored column-major.
    void BuildCounts(uint32_t ell0, uint32_t threads, size_t slice_terms) {
        const engine::ModCtx<uint32_t> ctx0(ell0);
        const size_t slice = std::max<size_t>(1, slice_terms / std::max<uint32_t>(lvl_->m(), 1));
        const size_t nsl = (mer_.size() + slice - 1) / slice;
        uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, static_cast<uint32_t>(std::max<size_t>(nsl, 1))));
        while (T > 1 && uint64_t(T) * D_ * D_ * 4 > (512ull << 20)) --T;
        std::vector<std::vector<uint32_t>> part(T);
        std::atomic<size_t> next(0);
        std::vector<std::exception_ptr> errs(T);
        auto work = [&](uint32_t w) {
            try {
                std::vector<uint32_t>& C = part[w];
                C.assign(size_t(D_) * D_, 0);
                for (;;) {
                    const size_t si = next.fetch_add(1);
                    if (si >= nsl) break;
                    const size_t a = si * slice, b = std::min(mer_.size(), a + slice);
                    const hecke::level::HeilbronnOperator Hc(*lvl_, p_, 0, ctx0,
                                                             std::vector<hecke::Mat22>(mer_.begin() + a, mer_.begin() + b));
                    for (const auto& B : Hc.blocks())
                        ENGINE_CHECK_MSG(B.dense.size() == 1 && B.dense[0] == 1, "UTagOperator: rho_0 is not (1)");
                    for (const hecke::level::HeilbronnTerm& tm : Hc.terms()) ++C[size_t(tm.src - 1) * D_ + (tm.dst - 1)];
                }
            } catch (...) {
                errs[w] = std::current_exception();
            }
        };
        std::vector<std::thread> pool;
        for (uint32_t w = 1; w < T; ++w) {
            try {
                pool.emplace_back(work, w);
            } catch (const std::system_error&) {
                break;
            }
        }
        work(0);
        for (auto& t : pool) t.join();
        for (auto& e : errs)
            if (e) std::rethrow_exception(e);
        counts_ = std::move(part[0]);
        info_.slices = nsl;
        info_.workers = 1;
        for (uint32_t w = 1; w < T; ++w) {
            if (part[w].empty()) continue;   // a worker that did not start
            ++info_.workers;
            for (size_t i = 0; i < counts_.size(); ++i) counts_[i] += part[w][i];
            std::vector<uint32_t>().swap(part[w]);
        }
        // an entry counts matrices eps at a fixed (i, j): at most |Mer_p| < 2^29 < every word prime, so it is its own residue
        ENGINE_CHECK_MSG(mer_.size() < (1u << 29), "UTagOperator: |Mer_p| < 2^29");
        for (uint32_t v : counts_) ENGINE_CHECK_MSG(v <= mer_.size(), "UTagOperator: a count above |Mer_p|");
    }

    uint32_t N_, n_, p_;
    uint32_t D_ = 0;
    std::unique_ptr<hecke::level::Gamma0LevelN> lvl_;
    UTagRoute route_ = UTagRoute::kTermTable;
    std::vector<hecke::Mat22> mer_;
    size_t slice_ = 1;
    std::vector<uint32_t> counts_;   // kCountOnce: column-major D x D
    UTagCountInfo info_;             // kCountOnce: the slices and the workers that built counts_
};

std::vector<PrimeRun> UTagRuns(uint32_t N, uint32_t n, uint32_t p, const std::vector<uint32_t>& ells, uint32_t threads, UTagRoute route,
                               UTagRoute* used, size_t count_slice_terms, UTagCountInfo* count_info) {
    std::vector<PrimeRun> out(ells.size());
    if (ells.empty()) return out;
    const UTagOperator op(N, n, p, ells[0], std::max<uint32_t>(threads, 1), route, count_slice_terms);
    if (used) *used = op.route();
    if (count_info) *count_info = op.count_info();
    ParallelForEach(0, ells.size(), op.prime_threads(threads), [&](size_t i) { out[i] = op.run(ells[i]); });
    return out;
}

const char* UTagRouteName(UTagRoute r) {
    switch (r) {
        case UTagRoute::kAuto: return "auto";
        case UTagRoute::kTermTable: return "term_table";
        case UTagRoute::kCountOnce: return "count_once";
        case UTagRoute::kDensePerPrime: return "dense_per_prime";
    }
    return "?";
}

std::vector<std::string> UTagResult::tags() const {
    std::vector<std::string> t;
    if (upper_bound_certified) t.push_back("upper_bound_certified");
    if (non_reduced_certified) t.push_back("non_reduced_certified");
    return t;
}

std::string UTagResult::text() const {
    std::ostringstream o;
    o << "hz_utag/v1 N=" << N << " n=" << n << " p=" << p << " b1=" << b1 << " d=" << d << " d_an=" << d_an << " e_k=" << e_k
      << " chi=[";
    for (size_t i = 0; i < chi.size(); ++i) o << (i ? "," : "") << chi[i];
    o << "] m0=" << m0 << " roots=[";
    for (size_t i = 0; i < roots.size(); ++i) o << (i ? "," : "") << roots[i].r << "^" << roots[i].mult << ":" << roots[i].type;
    o << "] split_degree=" << split_degree << " distinct=" << distinct << " gP_an=" << gP_an << " bounds=[";
    for (size_t i = 0; i < bounds.size(); ++i) o << (i ? "," : "") << "C(" << bounds[i].a << "," << bounds[i].b << ")|" << bounds[i].bound;
    o << "] tags=[";
    const auto t = tags();
    for (size_t i = 0; i < t.size(); ++i) o << (i ? "," : "") << t[i];
    o << "]";
    if (!n_is_p_power) o << " nr_scope=N_not_p_power(criterion=" << (non_reduced_criterion ? 1 : 0) << ")";
    o << "\n";
    return o.str();
}
uint64_t UTagResult::digest() const { return engine::FnvBytes(engine::kFnvOffset64, text()); }

UTagResult UTagStep(uint32_t N, uint32_t n, uint32_t p, const std::vector<uint32_t>& anemic_P, const RationalSettings& s) {
    if (N < 2) throw std::invalid_argument("UTagStep: N >= 2");
    if (n % 2) throw std::invalid_argument("UTagStep: n even");
    if (!integral::IsPrimeU32(p) || N % p != 0) throw std::invalid_argument("UTagStep: p a prime dividing N");
    UTagResult R;
    R.N = N;
    R.n = n;
    R.p = p;
    const uint32_t k = n + 2;
    R.d = DimMk(k, N);
    R.d_an = DimAnemic(k, N);
    R.e_k = static_cast<int64_t>(Gamma0Cusps(N)) - (k == 2 ? 1 : 0);

    // the bound: |root| <= p^{n+1} < 2^X, X = (n + 1) bitlen(p); B = 1 + p^{n+1} <= 2^X
    const u64 X = u64(n + 1) * BitLen64(p);
    // a first run fixes b1
    // (HZ-RAT-PERF (a)) grown on demand from 2048 (prefix-stable: the same primes in the same order), capped like HZ-RAT's list:
    // beyond the cap the named refusal refused(word_prime_cap); before, a need above 2048 * 29 bits looped forever (CoveredBits clamps).
    // (fix round 1, audit F7) the cap is honoured at every size: the list starts at min(2048, cap) primes and the walk refuses as soon
    // as it would take a prime beyond the cap (at the default cap 2^21: the same prefix, the same primes, the same values).
    if (s.word_prime_cap == 0) throw WordPrimeCapRefusal(s.word_prime_cap);
    std::vector<uint32_t> primes = WordPrimes(std::min<uint32_t>(2048, s.word_prime_cap), N);
    // (fix round 1, audit F1) U_p built once for every word prime where it can be (UTagRoute, header): at n = 0 the prime-independent
    // count matrix of eq:heilbronn, at n > 0 with many Merel matrices the D x D matrix per prime from slices, else the shipped term
    // route -- each equal to RunAtPrime(N, n, {p}, ell') entrywise (HeckeZUTagValidation arm RT); the primes run in parallel below
    const uint32_t threads = std::max<uint32_t>(s.threads, 1);
    const UTagOperator op(N, n, p, primes[0], threads, UTagRoute::kAuto);
    const PrimeRun r0 = op.run(primes[0]);
    R.b1 = r0.h1;
    ENGINE_CHECK_MSG(static_cast<int64_t>(R.b1) + R.e_k == 2 * R.d, "UTagStep: d != (b1 + e_k) / 2");
    const uint32_t b1 = R.b1;
    // integer-root candidates, |r| < 2^62 (proposals; each is certified below)
    std::vector<int64_t> cand;
    cand.push_back(1);
    cand.push_back(-1);
    {
        u128 pj = 1;
        for (uint32_t j = 1; j <= n + 1; ++j) {
            pj *= p;
            if (pj >= (u128(1) << 62)) break;
            cand.push_back(static_cast<int64_t>(pj));
            cand.push_back(-static_cast<int64_t>(pj));
        }
    }
    // the cover: coefficients (b1 + b1 X bits) and every candidate evaluation (|q(r)| <= (|r| + B)^{b1}, bits <= b1 (X + 2) + b1)
    const u64 b_coef = u64(b1) + u64(b1) * X;
    const u64 b_eval = u64(b1) * (X + 2) + 1;
    const u64 need = std::max(b_coef, b_eval) + 1;
    size_t kk = 0;
    for (u64 cov = 0; cov < need; ++kk) {
        if (kk >= s.word_prime_cap) throw WordPrimeCapRefusal(s.word_prime_cap);
        if (kk >= primes.size()) ExtendWordPrimes(primes, std::min<size_t>(s.word_prime_cap, 2 * primes.size()), N);
        cov += BitLen64(primes[kk]) - 1u;
    }
    R.primes_used = static_cast<uint32_t>(kk);
    std::vector<uint32_t> used(primes.begin(), primes.begin() + kk);
    std::vector<std::vector<u64>> chis(kk);
    // (fix round 1, audit F1) the word primes in parallel, each its own index (no value depends on the thread count)
    ParallelForEach(0, kk, op.prime_threads(threads), [&](size_t i) {
        const PrimeRun r = i == 0 ? r0 : op.run(used[i]);
        ENGINE_CHECK_MSG(r.h1 == b1, "UTagStep: dim H^1 differs between word primes");
        chis[i] = CharpolyOfCombination(r, {1});
        ENGINE_CHECK_MSG(chis[i].size() == b1 + 1u, "UTagStep: charpoly degree");
    });
    // chi exact
    for (uint32_t j = 0; j <= b1; ++j) {
        std::vector<uint32_t> res(kk);
        for (size_t i = 0; i < kk; ++i) res[i] = static_cast<uint32_t>(chis[i][j] % used[i]);
        R.chi.push_back(ZFromResidues(used, res, b_coef).hex());
    }
    // m0: the lowest nonzero coefficient (exact: a coefficient is 0 iff its residues all are)
    {
        uint32_t j = 0;
        for (; j <= b1; ++j) {
            bool zero = true;
            for (size_t i = 0; i < kk && zero; ++i) zero = chis[i][j] % used[i] == 0;
            if (!zero) break;
        }
        R.m0 = j;
    }
    // strip x^{m0}, then certified integer roots with multiplicities
    std::vector<std::vector<u64>> cur(kk);
    for (size_t i = 0; i < kk; ++i) cur[i].assign(chis[i].begin() + R.m0, chis[i].end());
    for (int64_t r : cand) {
        uint32_t mult = 0;
        for (;;) {
            if (cur[0].size() <= 1) break;
            std::vector<std::vector<u64>> nxt(kk);
            bool all_zero = true;
            for (size_t i = 0; i < kk && all_zero; ++i) {
                u64 rem = 0;
                nxt[i] = DivLinear(cur[i], ResMod(r, used[i]), used[i], &rem);
                all_zero = rem == 0;
            }
            // |q(r)| < 2^{b_eval} and the primes cover it: q(r) = 0 over Z iff every residue is 0
            if (!all_zero) break;
            cur.swap(nxt);
            ++mult;
        }
        if (mult) R.roots.push_back({r, mult, ""});
    }
    std::sort(R.roots.begin(), R.roots.end(), [](const UTagRoot& a, const UTagRoot& b) { return a.r < b.r; });
    R.split_degree = R.m0;
    for (const auto& rt : R.roots) R.split_degree += rt.mult;
    R.distinct = static_cast<uint32_t>(R.roots.size()) + (R.m0 > 0 ? 1u : 0u);

    // non_reduced_certified: b1 - m0 < d - d^an, proved only for N a power of p (02 Sec. 3.3; audit F1)
    {
        uint32_t M = N;
        while (M % p == 0) M /= p;
        R.n_is_p_power = M == 1;
    }
    R.non_reduced_criterion = static_cast<int64_t>(b1) - static_cast<int64_t>(R.m0) < R.d - R.d_an;
    R.non_reduced_certified = R.n_is_p_power && R.non_reduced_criterion;

    // upper_bound_certified: split, (C1) for theta = U_p (distinct = d), multiplicities 1 (Eisenstein, e_k of them) and 2 (cusp)
    if (R.split_degree == b1 && static_cast<int64_t>(R.distinct) == R.d && R.m0 == 0) {
        uint32_t ones = 0;
        bool pattern = true;
        for (const auto& rt : R.roots) {
            if (rt.mult == 1) ++ones;
            else if (rt.mult != 2) pattern = false;
        }
        if (pattern && static_cast<int64_t>(ones) == R.e_k) {
            for (auto& rt : R.roots) rt.type = rt.mult == 1 ? "eis" : "cusp";
            // the anemic g_P (word size, complete), else the U_p bounds alone
            u64 gp = 0;
            if (!anemic_P.empty()) {
                const RationalResult A = RationalStep(N, n, anemic_P, s);
                if (A.gP_status == "upper_bound_certified") {
                    u128 v = 1;
                    bool fits = A.gP.cofactor_test == "one";
                    for (const auto& pp : A.gP.factors) {
                        if (!fits || pp.proof == "probable" || pp.q == 0) { fits = false; break; }
                        for (uint32_t e = 0; e < pp.e && fits; ++e) { v *= pp.q; fits = v < (u128(1) << 63); }
                    }
                    if (fits) {
                        gp = static_cast<u64>(v);
                        R.gP_an = FactoredText(A.gP);
                    }
                }
            }
            for (size_t i = 0; i < R.roots.size(); ++i)
                for (size_t j = 0; j < R.roots.size(); ++j) {
                    const UTagRoot &a = R.roots[i], &b = R.roots[j];
                    if (a.type != "eis") continue;
                    if (b.type == "eis" && j <= i) continue;
                    const u64 diff = static_cast<u64>(a.r > b.r ? a.r - b.r : b.r - a.r);
                    UTagBound ub;
                    ub.a = std::string(a.type) + "(" + std::to_string(a.r) + ")";
                    ub.b = std::string(b.type) + "(" + std::to_string(b.r) + ")";
                    ub.bound = (b.type == "cusp" && gp) ? Gcd(gp, diff) : diff;
                    R.bounds.push_back(ub);
                }
            R.upper_bound_certified = true;
        }
    }
    return R;
}

}  // namespace heckez
