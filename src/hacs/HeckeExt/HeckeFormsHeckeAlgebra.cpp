// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-7) HeckeExt/HeckeFormsHeckeAlgebra -- see the header.
#include "HeckeFormsHeckeAlgebra.h"

#include "../Hecke/SubspaceOps.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>   // (slot-24 FNTT) the AVX2 butterflies (runtime dispatch, target attributes)
#endif

namespace hecke_ext {
namespace forms {

// ---------------------------------------------------------------------------
// (slot-23 G2-X) the q-series product and the divisor sums: the fast route and the naive oracle (see the header).
namespace {

// HECKE_FORMS_FASTMUL (default on; exactly "0" = naive; anything else warns and stays fast): one engine::lever::EnvLever
constexpr engine::lever::Policy kFastMulPolicy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING HeckeFormsHeckeAlgebra: HECKE_FORMS_FASTMUL='%s' is neither 0 nor 1; the fast route stays ON "
                                    "(only exactly 0 selects the naive route)\n"};
engine::lever::EnvArmRetiredLever g_fastmul("HECKE_FORMS_FASTMUL", kFastMulPolicy);
std::atomic<uint64_t> g_cnt_ntt{0}, g_cnt_school{0}, g_cnt_naive{0}, g_cnt_fallback{0}, g_cnt_chunked{0}, g_cnt_sieve{0}, g_cnt_divloop{0};

void SayEffectiveOnce() {
    // the EFFECTIVE convention: when the variable is set (any value), the first forms product of the process says what
    // the variable selects (its reading, not a scoped override), so a silently ignored setting can be seen
    if (g_fastmul.is_set()) g_fastmul.say_once(std::cout, g_fastmul.short_line("mul", "fast", "naive"));
}

// --- the NTT over two NTT primes; CRT to the exact integer coefficient, then mod ell ------------------------------
// P1 = 5 * 2^25 + 1, P2 = 7 * 2^26 + 1, 3 a primitive root of both; P1 * P2 = 7.88e16 (> 2^56).
constexpr uint32_t kP1 = 167772161u, kP2 = 469762049u;
constexpr uint32_t kMaxLogLen = 25;   // the transform length limit of P1 (the 2-adic valuation of P1 - 1)
constexpr uint32_t kPlanCacheMaxLog = 20;   // (G2-X audit N1) plans up to 2^20 are cached per prime (<= 32 MiB each prime)

template <uint32_t MOD>
inline uint32_t MulM(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>(static_cast<uint64_t>(a) * b % MOD);
}
template <uint32_t MOD>
uint32_t PowM(uint32_t b, uint64_t e) {
    uint32_t r = 1;
    while (e) {
        if (e & 1) r = MulM<MOD>(r, b);
        b = MulM<MOD>(b, b);
        e >>= 1;
    }
    return r;
}

// The NTT plan of length L = 2^lg over Z/MOD: per stage of half-length h the roots w_{2h}^k (k < h) at tab[h + k] (and
// the inverse roots at itab[h + k]), each with its Shoup companion floor(w 2^32 / MOD).  Lazy butterflies (Harvey): values
// live in [0, 2 MOD) between stages; 4 MOD < 2^31 for both primes, so no uint32_t sum wraps.
template <uint32_t MOD>
struct NttPlan {
    static_assert(static_cast<uint64_t>(MOD) * 4 < (uint64_t{1} << 31), "NttPlan: 4 MOD < 2^31");
    uint32_t lg = 0;
    size_t L = 0;
    std::vector<uint32_t> tw, twp, itw, itwp;
    explicit NttPlan(uint32_t lg_) : lg(lg_), L(size_t{1} << lg_), tw(L), twp(L), itw(L), itwp(L) {
        for (size_t h = 1; h < L; h <<= 1) {
            const uint32_t wl = PowM<MOD>(3u, (MOD - 1) / (2 * h)), iwl = PowM<MOD>(wl, MOD - 2);
            uint32_t w = 1, iw = 1;
            for (size_t k = 0; k < h; ++k) {
                tw[h + k] = w;
                twp[h + k] = static_cast<uint32_t>((static_cast<uint64_t>(w) << 32) / MOD);
                itw[h + k] = iw;
                itwp[h + k] = static_cast<uint32_t>((static_cast<uint64_t>(iw) << 32) / MOD);
                w = MulM<MOD>(w, wl);
                iw = MulM<MOD>(iw, iwl);
            }
        }
    }
    // y w mod MOD in [0, 2 MOD) for any y < 2^32 (Shoup)
    static inline uint32_t Shoup(uint32_t y, uint32_t w, uint32_t wp) {
        const uint32_t q = static_cast<uint32_t>((static_cast<uint64_t>(y) * wp) >> 32);
        return y * w - q * MOD;   // exact mod 2^32, and the true value lies in [0, 2 MOD)
    }
    // forward, decimation in frequency: natural order in ([0, 2 MOD)), bit-reversed order out ([0, 2 MOD))
    void Forward(uint32_t* a) const {
        constexpr uint32_t m2 = 2 * MOD;
        for (size_t h = L >> 1; h >= 1; h >>= 1) {
            const uint32_t* W = tw.data() + h;
            const uint32_t* Wp = twp.data() + h;
            for (size_t i = 0; i < L; i += 2 * h) {
                uint32_t* x = a + i;
                uint32_t* y = x + h;
                for (size_t k = 0; k < h; ++k) {
                    const uint32_t u = x[k], v = y[k];
                    uint32_t s = u + v;
                    s = s >= m2 ? s - m2 : s;
                    x[k] = s;
                    y[k] = Shoup(u + m2 - v, W[k], Wp[k]);
                }
            }
        }
    }
    // inverse, decimation in time: bit-reversed order in ([0, 2 MOD)), natural order out, scaled by 1/L, in [0, MOD)
    void Inverse(uint32_t* a) const {
        constexpr uint32_t m2 = 2 * MOD;
        for (size_t h = 1; h < L; h <<= 1) {
            const uint32_t* W = itw.data() + h;
            const uint32_t* Wp = itwp.data() + h;
            for (size_t i = 0; i < L; i += 2 * h) {
                uint32_t* x = a + i;
                uint32_t* y = x + h;
                for (size_t k = 0; k < h; ++k) {
                    const uint32_t u = x[k], v = Shoup(y[k], W[k], Wp[k]);
                    uint32_t s = u + v, t = u + m2 - v;
                    x[k] = s >= m2 ? s - m2 : s;
                    y[k] = t >= m2 ? t - m2 : t;
                }
            }
        }
        const uint32_t li = PowM<MOD>(static_cast<uint32_t>(L % MOD), MOD - 2);
        const uint32_t lip = static_cast<uint32_t>((static_cast<uint64_t>(li) << 32) / MOD);
        for (size_t i = 0; i < L; ++i) {
            const uint32_t r = Shoup(a[i], li, lip);
            a[i] = r >= MOD ? r - MOD : r;
        }
    }
};

// (G2-X audit N1) the plan of length 2^lg, built once per (MOD, lg) for lg <= kPlanCacheMaxLog (std::call_once: thread
// safe; a plan is immutable once built); longer plans are built per call, so no large table stays resident.
template <uint32_t MOD>
const NttPlan<MOD>& CachedPlan(uint32_t lg) {
    static std::array<std::once_flag, kPlanCacheMaxLog + 1> once;
    static std::array<std::unique_ptr<const NttPlan<MOD>>, kPlanCacheMaxLog + 1> plans;
    std::call_once(once[lg], [lg] { plans[lg] = std::make_unique<const NttPlan<MOD>>(lg); });
    return *plans[lg];
}

// The first n coefficients of the length-2^lg cyclic convolution of (a[0, la) mod ell) and (b[0, lb) mod ell), mod MOD.
template <uint32_t MOD>
std::vector<uint32_t> ConvMod(const QSeries& a, size_t la, const QSeries& b, size_t lb, bool square, uint32_t lg, uint32_t ell,
                              size_t n) {
    std::unique_ptr<const NttPlan<MOD>> own;
    if (lg > kPlanCacheMaxLog) own = std::make_unique<const NttPlan<MOD>>(lg);
    const NttPlan<MOD>& plan = own ? *own : CachedPlan<MOD>(lg);
    const size_t L = plan.L;
    std::vector<uint32_t> fa(L, 0);
    for (size_t i = 0; i < la; ++i) fa[i] = a[i] % ell;   // < ell < 2^16 < MOD
    plan.Forward(fa.data());
    if (square) {
        for (size_t i = 0; i < L; ++i) fa[i] = MulM<MOD>(fa[i], fa[i]);   // (2 MOD)^2 < 2^64; the product lands in [0, MOD)
    } else {
        std::vector<uint32_t> fb(L, 0);
        for (size_t i = 0; i < lb; ++i) fb[i] = b[i] % ell;
        plan.Forward(fb.data());
        for (size_t i = 0; i < L; ++i) fa[i] = MulM<MOD>(fa[i], fb[i]);
    }
    plan.Inverse(fa.data());
    fa.resize(n);
    return fa;
}

}  // namespace

QSeries MulSeriesNaive(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    // The product through slot 23, unchanged: the oracle and the rollback arm HECKE_FORMS_FASTMUL=0.
    QSeries c(N, 0);
    for (uint32_t i = 0; i < N && i < a.size(); ++i) {
        if (!a[i]) continue;
        const uint64_t x = a[i];
        for (uint32_t j = 0; i + j < N && j < b.size(); ++j)
            c[i + j] = static_cast<uint32_t>((c[i + j] + x * b[j]) % ell);
    }
    return c;
}

QSeries MulSeriesSchoolbook(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    // The same sum with the reduction delayed: every term (a_i mod ell)(b_j mod ell) < 2^32 (ell < 2^16), and a
    // coefficient has at most min(|a|, |b|, N) < 2^32 terms, so the uint64_t accumulator cannot wrap.
    ENGINE_CHECK_MSG(ell >= 1 && ell < (1u << 16), "MulSeriesSchoolbook: 1 <= ell < 2^16");
    const size_t la = std::min<size_t>(a.size(), N), lb = std::min<size_t>(b.size(), N);
    std::vector<uint64_t> acc(N, 0);
    std::vector<uint64_t> bb(lb);
    for (size_t j = 0; j < lb; ++j) bb[j] = b[j] % ell;
    for (size_t i = 0; i < la; ++i) {
        const uint64_t x = a[i] % ell;
        if (!x) continue;
        const size_t m = std::min(lb, static_cast<size_t>(N) - i);
        uint64_t* out = acc.data() + i;
        for (size_t j = 0; j < m; ++j) out[j] += x * bb[j];
    }
    QSeries c(N, 0);
    for (uint32_t i = 0; i < N; ++i) c[i] = static_cast<uint32_t>(acc[i] % ell);
    return c;
}

QSeries MulSeriesFast(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    // Exact over Z, then mod ell: c_k = sum_{i+j=k} (a_i mod ell)(b_j mod ell) is an integer in [0, min(la, lb) (ell-1)^2],
    // below P1 P2 (checked), so its residues mod P1 and P2 determine it (CRT; a cyclic length L >= la + lb - 1 does not
    // alias), and c_k mod ell is the naive route's value: the product mod ell is unique, so no route can change a byte.
    if (ell < 2 || ell >= (1u << 16)) {   // outside the forms grid (BuildCuspSpace: 5 <= ell < 2^16): the oracle
        g_cnt_fallback.fetch_add(1, std::memory_order_relaxed);
        return MulSeriesNaive(a, b, ell, N);
    }
    if (N == 0) return QSeries();
    const size_t la = std::min<size_t>(a.size(), N), lb = std::min<size_t>(b.size(), N);
    if (la == 0 || lb == 0) {
        g_cnt_school.fetch_add(1, std::memory_order_relaxed);
        return QSeries(N, 0);
    }
    if (std::min(la, lb) < kFastMulSchoolbookMin || N < kFastMulNttMinN) {
        g_cnt_school.fetch_add(1, std::memory_order_relaxed);
        return MulSeriesSchoolbook(a, b, ell, N);
    }
    const unsigned __int128 bound = static_cast<unsigned __int128>(std::min(la, lb)) * (ell - 1) * (ell - 1);
    const size_t need = la + lb - 1;
    uint32_t lg = 0;
    while ((size_t{1} << lg) < need) ++lg;
    if (lg > kMaxLogLen || bound >= static_cast<unsigned __int128>(kP1) * kP2) {
        // (G2-X audit M1) past the transform length (|a| + |b| - 1 > 2^25, i.e. N > 2^24 for balanced products; the
        // CRT bound can only fail there too, as min(la, lb) <= 2^24 forces it for ell < 2^16): a short operand goes to the
        // schoolbook (cost <= kFastMulChunkSchoolbookMax N), anything else to the chunked product, whose blocks are NTT
        // products within both limits -- no quadratic cliff, and the same bytes.
        g_cnt_fallback.fetch_add(1, std::memory_order_relaxed);
        if (std::min(la, lb) <= kFastMulChunkSchoolbookMax) return MulSeriesSchoolbook(a, b, ell, N);
        return MulSeriesChunked(a, b, ell, N, kMaxLogLen - 1);
    }
    g_cnt_ntt.fetch_add(1, std::memory_order_relaxed);
    const bool square = (&a == &b) || (la == lb && std::equal(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(la), b.begin()));
    const size_t n = std::min<size_t>(N, need);
    const std::vector<uint32_t> r1 = ConvMod<kP1>(a, la, b, lb, square, lg, ell, n);
    const std::vector<uint32_t> r2 = ConvMod<kP2>(a, la, b, lb, square, lg, ell, n);
    const uint32_t inv_p1 = PowM<kP2>(kP1 % kP2, kP2 - 2);   // P1^{-1} mod P2
    QSeries c(N, 0);
    for (size_t i = 0; i < n; ++i) {
        // x = r1 + P1 t with t = (r2 - r1) P1^{-1} mod P2: the unique x in [0, P1 P2) (r1 < P1 < P2)
        const uint32_t d = r2[i] >= r1[i] ? r2[i] - r1[i] : r2[i] + kP2 - r1[i];
        const uint32_t t = MulM<kP2>(d, inv_p1);
        const uint64_t x = r1[i] + static_cast<uint64_t>(kP1) * t;
        c[i] = static_cast<uint32_t>(x % ell);
    }
    return c;
}

QSeries MulSeriesChunked(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N, uint32_t chunk_lg) {
    // a = sum_i A_i q^{iC}, b = sum_j B_j q^{jC} with blocks of length C = 2^chunk_lg; c = sum_{i,j} A_i B_j q^{(i+j)C}
    // truncated to N, accumulated mod ell.  Each block product has |A_i| + |B_j| - 1 < 2C <= 2^kMaxLogLen and
    // min(|A_i|, |B_j|) (ell-1)^2 <= 2^24 (ell-1)^2 < P1 P2, so MulSeriesFast takes it within both limits (never this
    // function again); the sum mod ell is the product mod ell, so the bytes are the naive route's.
    ENGINE_CHECK_MSG(ell >= 2 && ell < (1u << 16), "MulSeriesChunked: 2 <= ell < 2^16");
    ENGINE_CHECK_MSG(chunk_lg >= 1 && chunk_lg <= kMaxLogLen - 1, "MulSeriesChunked: 1 <= chunk_lg <= kMaxLogLen - 1");
    g_cnt_chunked.fetch_add(1, std::memory_order_relaxed);
    QSeries c(N, 0);
    const size_t C = size_t{1} << chunk_lg;
    const size_t la = std::min<size_t>(a.size(), N), lb = std::min<size_t>(b.size(), N);
    for (size_t i0 = 0; i0 < la; i0 += C) {
        const QSeries A(a.begin() + static_cast<std::ptrdiff_t>(i0), a.begin() + static_cast<std::ptrdiff_t>(std::min(la, i0 + C)));
        for (size_t j0 = 0; j0 < lb && i0 + j0 < N; j0 += C) {
            const QSeries B(b.begin() + static_cast<std::ptrdiff_t>(j0), b.begin() + static_cast<std::ptrdiff_t>(std::min(lb, j0 + C)));
            const size_t off = i0 + j0;
            const uint32_t Nb = static_cast<uint32_t>(std::min<size_t>(N - off, A.size() + B.size() - 1));
            const QSeries p = MulSeriesFast(A, B, ell, Nb);
            for (size_t k = 0; k < Nb; ++k) {
                const uint32_t s = c[off + k] + p[k];   // both < ell < 2^16
                c[off + k] = s >= ell ? s - ell : s;
            }
        }
    }
    return c;
}

bool FormsFastMulEnabled() { return g_fastmul.enabled(); }

QSeries MulSeries(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    SayEffectiveOnce();
    if (!FormsFastMulEnabled()) {
        g_cnt_naive.fetch_add(1, std::memory_order_relaxed);
        return MulSeriesNaive(a, b, ell, N);
    }
    return MulSeriesFast(a, b, ell, N);
}

FormsFastMulCounters FormsFastMulCountersNow() {
    FormsFastMulCounters c;
    c.ntt = g_cnt_ntt.load();
    c.schoolbook = g_cnt_school.load();
    c.naive = g_cnt_naive.load();
    c.fallback = g_cnt_fallback.load();
    c.chunked = g_cnt_chunked.load();
    c.sieve = g_cnt_sieve.load();
    c.divisor_loop = g_cnt_divloop.load();
    return c;
}

std::string FormsFastMulEffectiveLine() {
    g_fastmul.mark_said();   // a caller printing the full line: the short line would repeat it
    const FormsFastMulCounters c = FormsFastMulCountersNow();
    return g_fastmul.effective_line("mul", "fast", "naive") + " ntt=" + std::to_string(c.ntt) + " schoolbook=" +
           std::to_string(c.schoolbook) + " naive=" + std::to_string(c.naive) + " fallback=" + std::to_string(c.fallback) + " chunked=" + std::to_string(c.chunked) +
           " sieve=" + std::to_string(c.sieve) + " divisor_loop=" + std::to_string(c.divisor_loop) +
           " (forms cross-route q-series products and E_4 / E_6 divisor sums; naive = the O(N^2) route, the oracle and rollback arm)";
}

ScopedFormsFastMul::ScopedFormsFastMul(bool on) : guard_(g_fastmul, on) {}

namespace {

uint32_t PowMod(uint64_t b, uint64_t e, uint32_t m) { return engine::scalar::powmod<uint32_t>(static_cast<uint32_t>(b % m), e, m); }

QSeries Mul(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    return MulSeries(a, b, ell, N);   // (slot-23 G2-X) the fast route, or the naive one under HECKE_FORMS_FASTMUL=0
}

QSeries One(uint32_t N) {
    QSeries s(N, 0);
    s[0] = 1;
    return s;
}

ExtMatrix Sub(const ExtMatrix& a, const ExtMatrix& b) {
    ExtMatrix c(a.rows, a.cols, a.p);
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j) c.data[i][j] = (a.data[i][j] + a.p - b.data[i][j]) % a.p;
    return c;
}

ExtMatrix Scale(const ExtMatrix& a, uint32_t s) {
    ExtMatrix c(a.rows, a.cols, a.p);
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            c.data[i][j] = static_cast<uint32_t>(static_cast<uint64_t>(a.data[i][j]) * s % a.p);
    return c;
}

ExtMatrix ShiftByScalar(const ExtMatrix& a, uint32_t lambda) {
    ExtMatrix c = a;
    for (uint32_t i = 0; i < a.rows; ++i) c.data[i][i] = (c.data[i][i] + a.p - lambda % a.p) % a.p;
    return c;
}

ExtMatrix MatPow(const ExtMatrix& a, uint32_t e) {
    ExtMatrix r(a.rows, a.p), b = a;
    while (e) {
        if (e & 1) r = r * b;
        b = b * b;
        e >>= 1;
    }
    return r;
}

bool IsZero(const ExtMatrix& a) {
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j]) return false;
    return true;
}

} // namespace

QSeries DeltaSeries(uint32_t ell, uint32_t N) {
    ENGINE_CHECK_MSG(N >= 2, "DeltaSeries: precision N >= 2");
    // prod (1 - q^n) by Euler's pentagonal theorem, to precision N.
    QSeries P(N, 0);
    for (int64_t m = -static_cast<int64_t>(N); m <= static_cast<int64_t>(N); ++m) {
        const int64_t e = m * (3 * m - 1) / 2;
        if (e < 0 || e >= static_cast<int64_t>(N)) continue;
        P[e] = (m % 2 == 0) ? 1u : ell - 1u;
    }
    const QSeries P2 = Mul(P, P, ell, N), P4 = Mul(P2, P2, ell, N), P8 = Mul(P4, P4, ell, N);
    const QSeries P16 = Mul(P8, P8, ell, N), P24 = Mul(P16, P8, ell, N);
    QSeries D(N, 0);
    for (uint32_t i = 1; i < N; ++i) D[i] = P24[i - 1];
    return D;
}

QSeries EisensteinSeries(uint32_t weight, uint32_t ell, uint32_t N) {
    ENGINE_CHECK_MSG(weight == 4 || weight == 6, "EisensteinSeries: weight 4 or 6 only");
    const uint32_t c = weight == 4 ? 240u % ell : (ell - 504u % ell) % ell;
    QSeries E(N, 0);
    if (N == 0) return E;
    E[0] = 1 % ell;
    SayEffectiveOnce();
    if (!FormsFastMulEnabled()) {
        // the route through slot 23 (O(N^2) trial division): the oracle and the rollback arm HECKE_FORMS_FASTMUL=0
        g_cnt_divloop.fetch_add(1, std::memory_order_relaxed);
        for (uint32_t n = 1; n < N; ++n) {
            uint64_t s = 0;
            for (uint32_t t = 1; t <= n; ++t)
                if (n % t == 0) s += PowMod(t, weight - 1, ell);
            E[n] = static_cast<uint32_t>(s % ell * c % ell);
        }
        return E;
    }
    // (slot-23 G2-X) the same sums s_n = sum_{t | n} (t^{w-1} mod ell) by a divisor sieve, O(N log N): the same uint64_t
    // summands (s_n <= sigma_0(n) (ell - 1) cannot wrap), so s_n and E[n] are the same integers.
    g_cnt_sieve.fetch_add(1, std::memory_order_relaxed);
    std::vector<uint64_t> s(N, 0);
    for (uint32_t t = 1; t < N; ++t) {
        const uint64_t pt = PowMod(t, weight - 1, ell);
        if (!pt) continue;
        for (uint64_t m = t; m < N; m += t) s[m] += pt;
    }
    for (uint32_t n = 1; n < N; ++n) E[n] = static_cast<uint32_t>(s[n] % ell * c % ell);
    return E;
}

namespace {

// (slot-24 FNTT) the G2-X construction, verbatim: the rollback arm HECKE_FORMS_NTT_CACHE=0 and the route wherever the
// FNTT route does not apply (FnttApplicable), and under HECKE_FORMS_FASTMUL=0.
CuspSpace BuildCuspSpaceLegacy(uint32_t k, uint32_t ell, uint32_t N) {
    ENGINE_CHECK_MSG(k % 2 == 0 && k >= 12, "BuildCuspSpace: even weight k >= 12");
    ENGINE_CHECK_MSG(ell >= 5 && ell < kFormsMaxEllExclusive, "BuildCuspSpace: 5 <= ell < 2^27");
    ENGINE_CHECK_MSG(N >= 2, "BuildCuspSpace: precision N >= 2");
    CuspSpace S;
    S.k = k;
    S.ell = ell;
    S.precision = N;
    uint32_t kp = k % 12;
    if (kp == 2) kp = 14;
    S.k_prime = kp;
    S.d = (k - kp) / 12;
    ENGINE_CHECK_MSG(N > S.d, "BuildCuspSpace: precision must exceed dim S_k");

    const QSeries E4 = EisensteinSeries(4, ell, N), E6 = EisensteinSeries(6, ell, N);
    const QSeries D = DeltaSeries(ell, N);
    QSeries base = One(N);
    if (kp == 4) base = E4;
    else if (kp == 6) base = E6;
    else if (kp == 8) base = Mul(E4, E4, ell, N);
    else if (kp == 10) base = Mul(E4, E6, ell, N);
    else if (kp == 14) base = Mul(Mul(E4, E4, ell, N), E6, ell, N);

    const QSeries E6sq = Mul(E6, E6, ell, N);
    std::vector<QSeries> Dpow(S.d + 1), Epow(S.d + 1);
    Dpow[0] = One(N);
    Epow[0] = One(N);
    for (uint32_t j = 1; j <= S.d; ++j) {
        Dpow[j] = Mul(Dpow[j - 1], D, ell, N);
        Epow[j] = Mul(Epow[j - 1], E6sq, ell, N);
    }
    for (uint32_t j = 1; j <= S.d; ++j) {
        QSeries f = Mul(Mul(base, Dpow[j], ell, N), Epow[S.d - j], ell, N);
        for (uint32_t i = 0; i < j; ++i) ENGINE_CHECK_MSG(f[i] == 0, "BuildCuspSpace: f_j not O(q^j)");
        ENGINE_CHECK_MSG(f[j] == 1 % ell, "BuildCuspSpace: f_j's leading coefficient is not 1");
        S.basis.push_back(std::move(f));
    }
    return S;
}

CuspSpace BuildModularSpaceLegacy(uint32_t k, uint32_t ell, uint32_t N) {
    // (slot-12 WP-12) M_k = F_ell f_0 (+) S_k with f_0 = E_6^{2d} E_{k'}: every
    // f_j = Delta^j E_6^{2(d-j)} E_{k'}, j = 0 .. d, has leading term q^j, so the
    // d + 1 forms are a Z-basis of M_k(Z) (Miller) read from a_0 .. a_d.
    CuspSpace S = BuildCuspSpaceLegacy(k, ell, N);
    const QSeries E4 = EisensteinSeries(4, ell, N), E6 = EisensteinSeries(6, ell, N);
    QSeries base = One(N);
    const uint32_t kp = S.k_prime;
    if (kp == 4) base = E4;
    else if (kp == 6) base = E6;
    else if (kp == 8) base = Mul(E4, E4, ell, N);
    else if (kp == 10) base = Mul(E4, E6, ell, N);
    else if (kp == 14) base = Mul(Mul(E4, E4, ell, N), E6, ell, N);
    const QSeries E6sq = Mul(E6, E6, ell, N);
    QSeries f0 = base;
    for (uint32_t j = 0; j < S.d; ++j) f0 = Mul(f0, E6sq, ell, N);
    ENGINE_CHECK_MSG(f0[0] == 1 % ell, "BuildModularSpace: f_0's constant term is not 1");
    S.basis.insert(S.basis.begin(), std::move(f0));
    S.with_constant = true;
    return S;
}

}  // namespace

// ---------------------------------------------------------------------------
// (slot-24 FNTT) the FNTT route of BuildCuspSpace / BuildModularSpace (see the header and FNTT_report.md).
namespace {

// HECKE_FORMS_NTT_CACHE (default on; exactly "0" = the G2-X route; anything else warns and stays on)
constexpr engine::lever::Policy kNttCachePolicy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING HeckeFormsHeckeAlgebra: HECKE_FORMS_NTT_CACHE='%s' is neither 0 nor 1; the FNTT route stays ON "
                                    "(only exactly 0 selects the G2-X route)\n"};
engine::lever::EnvLever g_nttcache("HECKE_FORMS_NTT_CACHE", kNttCachePolicy);
std::atomic<int> g_simd_override{-1};    // -1 = AVX2 when the CPU has it, 0 = scalar butterflies
std::atomic<uint64_t> g_cnt_fntt_spaces{0}, g_cnt_legacy_spaces{0}, g_cnt_fwd{0}, g_cnt_inv{0}, g_cnt_simd_tf{0}, g_cnt_scalar_tf{0},
    g_cnt_hoc_delayed{0}, g_cnt_hoc_legacy{0};

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  define FNTT_HAVE_X86 1
#  define FNTT_AVX2_FN __attribute__((target("avx2")))
#  define FNTT_AVX2_INLINE inline __attribute__((target("avx2"), always_inline))
#else
#  define FNTT_HAVE_X86 0
#endif

bool CpuHasAvx2() {
#if FNTT_HAVE_X86
    static const bool h = [] { __builtin_cpu_init(); return __builtin_cpu_supports("avx2") != 0; }();
    return h;
#else
    return false;
#endif
}

bool SimdActive() { return CpuHasAvx2() && g_simd_override.load(std::memory_order_relaxed) != 0; }

void SayCacheEffectiveOnce() {
    if (g_nttcache.is_set())
        g_nttcache.say_once(std::cout, std::string("EFFECTIVE HECKE_FORMS_NTT_CACHE route=") + (g_nttcache.env_enabled() ? "fntt" : "legacy") +
                                           " simd=" + (CpuHasAvx2() ? "avx2" : "scalar") + " (HECKE_FORMS_NTT_CACHE=" + g_nttcache.raw() + ")");
}

// Butterfly blocks: stages whose span 2h is at most kBlock2 elements (512 KiB, L2) run depth first inside each such block,
// and inside it the spans of at most kBlock1 elements (16 KiB, L1) sub-block by sub-block; the longer ones breadth first
// over the whole array.  Each butterfly is the G2-X one (Shoup, lazy in [0, 2 MOD)); the order of the stages on each span is
// unchanged and different spans are independent, so every value is the same integer.
constexpr size_t kBlock1 = size_t{1} << 12, kBlock2 = size_t{1} << 17;

template <uint32_t MOD>
inline uint32_t ShoupS(uint32_t y, uint32_t w, uint32_t wp) {
    const uint32_t q = static_cast<uint32_t>((static_cast<uint64_t>(y) * wp) >> 32);
    return y * w - q * MOD;
}

template <uint32_t MOD>
void DifStageScalar(uint32_t* a, size_t len, size_t h, const uint32_t* W, const uint32_t* Wp) {
    constexpr uint32_t m2 = 2 * MOD;
    for (size_t i = 0; i < len; i += 2 * h) {
        uint32_t* x = a + i;
        uint32_t* y = x + h;
        for (size_t k = 0; k < h; ++k) {
            const uint32_t u = x[k], v = y[k];
            uint32_t s = u + v;
            s = s >= m2 ? s - m2 : s;
            x[k] = s;
            y[k] = ShoupS<MOD>(u + m2 - v, W[k], Wp[k]);
        }
    }
}

template <uint32_t MOD>
void DitStageScalar(uint32_t* a, size_t len, size_t h, const uint32_t* W, const uint32_t* Wp) {
    constexpr uint32_t m2 = 2 * MOD;
    for (size_t i = 0; i < len; i += 2 * h) {
        uint32_t* x = a + i;
        uint32_t* y = x + h;
        for (size_t k = 0; k < h; ++k) {
            const uint32_t u = x[k], v = ShoupS<MOD>(y[k], W[k], Wp[k]);
            const uint32_t s = u + v, t = u + m2 - v;
            x[k] = s >= m2 ? s - m2 : s;
            y[k] = t >= m2 ? t - m2 : t;
        }
    }
}

#if FNTT_HAVE_X86
FNTT_AVX2_INLINE __m256i MulHi32x8(__m256i a, __m256i b) {
    const __m256i e = _mm256_mul_epu32(a, b);
    const __m256i o = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
    return _mm256_blend_epi32(_mm256_srli_epi64(e, 32), o, 0xAA);
}

// The scalar butterflies lane by lane: s = u + v, and min(s, s - 2MOD) is the conditional subtraction (s < 2^31, and a
// wrapped s - 2MOD is >= 2^32 - 2MOD > s); Shoup y w - floor(y wp / 2^32) MOD (mod 2^32) with the high product by mul_epu32.
template <uint32_t MOD>
FNTT_AVX2_INLINE void DifBfly8(__m256i& u, __m256i& v, __m256i w, __m256i wp) {
    const __m256i m2 = _mm256_set1_epi32(static_cast<int>(2 * MOD)), md = _mm256_set1_epi32(static_cast<int>(MOD));
    __m256i s = _mm256_add_epi32(u, v);
    s = _mm256_min_epu32(s, _mm256_sub_epi32(s, m2));
    const __m256i t = _mm256_sub_epi32(_mm256_add_epi32(u, m2), v);
    const __m256i q = MulHi32x8(t, wp);
    u = s;
    v = _mm256_sub_epi32(_mm256_mullo_epi32(t, w), _mm256_mullo_epi32(q, md));
}
template <uint32_t MOD>
FNTT_AVX2_INLINE void DitBfly8(__m256i& u, __m256i& v, __m256i w, __m256i wp) {
    const __m256i m2 = _mm256_set1_epi32(static_cast<int>(2 * MOD)), md = _mm256_set1_epi32(static_cast<int>(MOD));
    const __m256i q = MulHi32x8(v, wp);
    const __m256i x = _mm256_sub_epi32(_mm256_mullo_epi32(v, w), _mm256_mullo_epi32(q, md));
    __m256i s = _mm256_add_epi32(u, x), t = _mm256_sub_epi32(_mm256_add_epi32(u, m2), x);
    u = _mm256_min_epu32(s, _mm256_sub_epi32(s, m2));
    v = _mm256_min_epu32(t, _mm256_sub_epi32(t, m2));
}
template <uint32_t MOD, bool DIF>
FNTT_AVX2_INLINE void Bfly8(__m256i& u, __m256i& v, __m256i w, __m256i wp) {
    if (DIF) DifBfly8<MOD>(u, v, w, wp);
    else DitBfly8<MOD>(u, v, w, wp);
}

// h >= 8: the spans' halves are whole vectors.  h in {1, 2, 4}: 16 consecutive elements (X0, X1) hold whole spans; they
// are de-interleaved into u (the first halves) and v (the second halves) with the twiddle of each lane's offset k, then
// interleaved back -- the same butterfly on the same pair (x[k], x[k + h]) as the scalar stage.
template <uint32_t MOD, bool DIF>
FNTT_AVX2_FN void StageAvx2(uint32_t* a, size_t len, size_t h, const uint32_t* W, const uint32_t* Wp) {
    if (len < 16) return DIF ? DifStageScalar<MOD>(a, len, h, W, Wp) : DitStageScalar<MOD>(a, len, h, W, Wp);
    if (h >= 8) {
        for (size_t i = 0; i < len; i += 2 * h) {
            uint32_t* x = a + i;
            uint32_t* y = x + h;
            for (size_t k = 0; k < h; k += 8) {
                __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k));
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + k));
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(W + k));
                const __m256i wp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Wp + k));
                Bfly8<MOD, DIF>(u, v, w, wp);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(x + k), u);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + k), v);
            }
        }
        return;
    }
    __m256i w, wp;
    if (h == 4) {
        w = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(W)));
        wp = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(Wp)));
    } else if (h == 2) {
        const int w0 = static_cast<int>(W[0]), w1 = static_cast<int>(W[1]), p0 = static_cast<int>(Wp[0]), p1 = static_cast<int>(Wp[1]);
        w = _mm256_setr_epi32(w0, w1, w0, w1, w0, w1, w0, w1);
        wp = _mm256_setr_epi32(p0, p1, p0, p1, p0, p1, p0, p1);
    } else {
        w = _mm256_set1_epi32(static_cast<int>(W[0]));
        wp = _mm256_set1_epi32(static_cast<int>(Wp[0]));
    }
    for (size_t i = 0; i < len; i += 16) {
        const __m256i X0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i X1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 8));
        __m256i u, v, Y0, Y1;
        if (h == 4) {          // u = a0..a3 | a8..a11, v = a4..a7 | a12..a15
            u = _mm256_permute2x128_si256(X0, X1, 0x20);
            v = _mm256_permute2x128_si256(X0, X1, 0x31);
            Bfly8<MOD, DIF>(u, v, w, wp);
            Y0 = _mm256_permute2x128_si256(u, v, 0x20);
            Y1 = _mm256_permute2x128_si256(u, v, 0x31);
        } else if (h == 2) {   // u = a0 a1 a8 a9 | a4 a5 a12 a13, v = a2 a3 a10 a11 | a6 a7 a14 a15
            u = _mm256_unpacklo_epi64(X0, X1);
            v = _mm256_unpackhi_epi64(X0, X1);
            Bfly8<MOD, DIF>(u, v, w, wp);
            Y0 = _mm256_unpacklo_epi64(u, v);
            Y1 = _mm256_unpackhi_epi64(u, v);
        } else {               // u = a0 a2 a8 a10 | a4 a6 a12 a14, v = a1 a3 a9 a11 | a5 a7 a13 a15
            u = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(X0), _mm256_castsi256_ps(X1), _MM_SHUFFLE(2, 0, 2, 0)));
            v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(X0), _mm256_castsi256_ps(X1), _MM_SHUFFLE(3, 1, 3, 1)));
            Bfly8<MOD, DIF>(u, v, w, wp);
            Y0 = _mm256_unpacklo_epi32(u, v);
            Y1 = _mm256_unpackhi_epi32(u, v);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i), Y0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + i + 8), Y1);
    }
}

// The pointwise product x <- x y / L: Montgomery x y 2^{-32} in [0, 2 MOD) (x y < 4 MOD^2 < MOD 2^32), then Shoup by
// c = L^{-1} 2^32 mod MOD; congruent to the scalar route's MulM-then-Shoup value (a representative in [0, 2 MOD) of the
// same residue -- the CRT step reduces it).
template <uint32_t MOD>
FNTT_AVX2_FN void PointwiseAvx2(uint32_t* x, const uint32_t* y, size_t L, uint32_t ninv, uint32_t c, uint32_t cp) {
    const __m256i md = _mm256_set1_epi32(static_cast<int>(MOD)), ni = _mm256_set1_epi32(static_cast<int>(ninv));
    const __m256i cv = _mm256_set1_epi32(static_cast<int>(c)), cpv = _mm256_set1_epi32(static_cast<int>(cp));
    for (size_t i = 0; i < L; i += 8) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + i));
        const __m256i te = _mm256_mul_epu32(a, b);
        const __m256i to = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
        const __m256i me = _mm256_mul_epu32(te, ni), mo = _mm256_mul_epu32(to, ni);   // m = t * (-MOD^{-1}) mod 2^32
        const __m256i ue = _mm256_srli_epi64(_mm256_add_epi64(te, _mm256_mul_epu32(me, md)), 32);
        const __m256i uo = _mm256_add_epi64(to, _mm256_mul_epu32(mo, md));
        const __m256i z = _mm256_blend_epi32(ue, uo, 0xAA);   // (t + m MOD) / 2^32 in [0, 2 MOD)
        const __m256i q = MulHi32x8(z, cpv);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(x + i), _mm256_sub_epi32(_mm256_mullo_epi32(z, cv), _mm256_mullo_epi32(q, md)));
    }
}

// g[n] += mc b[n] for n in [from, to) (uint64_t accumulators, mc and b[n] < 2^32): HeckeOnCusp's delayed elimination
FNTT_AVX2_FN void AxpyU64Avx2(uint64_t* g, const uint32_t* b, uint64_t mc, size_t from, size_t to) {
    size_t n = from;
    const __m256i m = _mm256_set1_epi64x(static_cast<long long>(mc));
    for (; n + 4 <= to; n += 4) {
        const __m256i bv = _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + n)));
        const __m256i gv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g + n));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(g + n), _mm256_add_epi64(gv, _mm256_mul_epu32(bv, m)));
    }
    for (; n < to; ++n) g[n] += mc * b[n];
}
#endif

// The FNTT plan of length L = 2^lg over Z/MOD: the forward roots only (tw[h + k] = w_{2h}^k and the Shoup companion, the
// G2-X NttPlan's tables).  Forward: decimation in frequency, natural in, bit-reversed out.  "Inverse": decimation in time
// with the FORWARD roots, bit-reversed in, natural out -- that is the DFT with root w of the natural-order spectrum, so
// the inverse transform's value at n is (that output at (L - n) mod L) / L; the 1/L is folded into the pointwise product.
// (slot-24 FNTT3) G is the generator whose powers give the roots: 3 for P1, P2 (the FNTT default, unchanged), 7 for P3 (3 is
// a square mod P3 = 45 * 2^23 + 1, so its powers never reach a primitive 2^23-th root; 7 is a primitive root of P3).
template <uint32_t MOD, uint32_t G = 3>
struct FPlan {
    static_assert(static_cast<uint64_t>(MOD) * 4 < (uint64_t{1} << 31), "FPlan: 4 MOD < 2^31");
    uint32_t lg = 0;
    size_t L = 0;
    std::vector<uint32_t> tw, twp;
    uint32_t li = 0, lip = 0;   // L^{-1} mod MOD and its Shoup companion
    uint32_t ninv = 0;          // -MOD^{-1} mod 2^32 (the AVX2 pointwise product's Montgomery step)
    uint32_t lr = 0, lrp = 0;   // L^{-1} 2^32 mod MOD and its Shoup companion (ditto)
    explicit FPlan(uint32_t lg_) : lg(lg_), L(size_t{1} << lg_), tw(L), twp(L) {
        for (size_t h = 1; h < L; h <<= 1) {
            const uint32_t wl = PowM<MOD>(G, (MOD - 1) / (2 * h));
            uint32_t w = 1;
            for (size_t k = 0; k < h; ++k) {
                tw[h + k] = w;
                twp[h + k] = static_cast<uint32_t>((static_cast<uint64_t>(w) << 32) / MOD);
                w = MulM<MOD>(w, wl);
            }
        }
        li = PowM<MOD>(static_cast<uint32_t>(L % MOD), MOD - 2);
        lip = static_cast<uint32_t>((static_cast<uint64_t>(li) << 32) / MOD);
        uint32_t inv = 1;   // MOD^{-1} mod 2^32 by Newton (MOD odd): each step doubles the correct low bits
        for (int it = 0; it < 5; ++it) inv *= 2u - MOD * inv;
        ninv = 0u - inv;
        lr = static_cast<uint32_t>((static_cast<uint64_t>(li) << 32) % MOD);
        lrp = static_cast<uint32_t>((static_cast<uint64_t>(lr) << 32) / MOD);
    }
    void Stage(bool dif, bool simd, uint32_t* a, size_t len, size_t h) const {
#if FNTT_HAVE_X86
        if (simd) {
            if (dif) StageAvx2<MOD, true>(a, len, h, tw.data() + h, twp.data() + h);
            else StageAvx2<MOD, false>(a, len, h, tw.data() + h, twp.data() + h);
            return;
        }
#endif
        (void)simd;
        if (dif) DifStageScalar<MOD>(a, len, h, tw.data() + h, twp.data() + h);
        else DitStageScalar<MOD>(a, len, h, tw.data() + h, twp.data() + h);
    }
    // Forward: the stages h = L/2 .. 1 in that order on every span; the spans of 2h <= kBlock2 are finished block by
    // block, and inside such a block the spans of 2h <= kBlock1 sub-block by sub-block (the order on each span is kept).
    void Forward(uint32_t* a, bool simd) const {   // [0, 2 MOD) in and out
        size_t h = L >> 1;
        for (; h >= 1 && 2 * h > kBlock2; h >>= 1) Stage(true, simd, a, L, h);
        if (h == 0) return;
        const size_t S2 = 2 * h;
        for (size_t i2 = 0; i2 < L; i2 += S2) {
            uint32_t* b2 = a + i2;
            size_t hh = h;
            for (; hh >= 1 && 2 * hh > kBlock1; hh >>= 1) Stage(true, simd, b2, S2, hh);
            if (hh == 0) continue;
            const size_t S1 = 2 * hh;
            for (size_t i1 = 0; i1 < S2; i1 += S1)
                for (size_t h3 = hh; h3 >= 1; h3 >>= 1) Stage(true, simd, b2 + i1, S1, h3);
        }
    }
    void Backward(uint32_t* a, bool simd) const {   // [0, 2 MOD) in and out; see above for the index reversal and 1/L
        const size_t S1 = std::min(L, kBlock1), S2 = std::min(L, kBlock2);
        for (size_t i2 = 0; i2 < L; i2 += S2) {
            uint32_t* b2 = a + i2;
            for (size_t i1 = 0; i1 < S2; i1 += S1)
                for (size_t h = 1; h < S1; h <<= 1) Stage(false, simd, b2 + i1, S1, h);
            for (size_t h = S1; h < S2; h <<= 1) Stage(false, simd, b2, S2, h);
        }
        for (size_t h = S2; h < L; h <<= 1) Stage(false, simd, a, L, h);
    }
};

// One spectrum: the transforms mod P1 and P2 of a series of length <= N, zero-padded to L.
struct Spec {
    std::vector<uint32_t> s1, s2;
};

// The engine of one space: ell, N, the plans of length L = 2^lg >= 2N - 1, Barrett for ell.
class FormsNtt {
public:
    FormsNtt(uint32_t ell, uint32_t N)
        : ell_(ell), N_(N), lg_(LogLen(N)), p1_(lg_), p2_(lg_), simd_(SimdActive()),
          bar_(~uint64_t{0} / ell), inv_p1_(PowM<kP2>(kP1 % kP2, kP2 - 2)) {}
    static uint32_t LogLen(uint32_t N) {
        uint32_t lg = 0;
        while ((size_t{1} << lg) < 2 * static_cast<size_t>(N) - 1) ++lg;
        return lg;
    }
    size_t L() const { return p1_.L; }
    // a holds at most N residues < ell (every series of the route is reduced: E_4, E_6, Delta and every CRT output)
    void Forward(const QSeries& a, Spec& s) const {
        const size_t la = std::min<size_t>(a.size(), N_), L = p1_.L;
        s.s1.assign(L, 0);
        s.s2.assign(L, 0);
        std::copy(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(la), s.s1.begin());
        std::copy(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(la), s.s2.begin());
        p1_.Forward(s.s1.data(), simd_);
        p2_.Forward(s.s2.data(), simd_);
        Count(1, 0);
    }
    // x <- x * y / L pointwise (both primes), in [0, 2 MOD)
    void MulInto(Spec& x, const Spec& y) const {
        PointwiseScaled<kP1>(x.s1.data(), y.s1.data(), p1_);
        PointwiseScaled<kP2>(x.s2.data(), y.s2.data(), p2_);
    }
    void SquareScaled(Spec& x) const {
        PointwiseScaled<kP1>(x.s1.data(), x.s1.data(), p1_);
        PointwiseScaled<kP2>(x.s2.data(), x.s2.data(), p2_);
    }
    // the backward transform of x in place, then out = the first N coefficients, CRT to the exact integer, mod ell
    void InverseTo(Spec& x, QSeries& out) const {
        p1_.Backward(x.s1.data(), simd_);
        p2_.Backward(x.s2.data(), simd_);
        Count(0, 1);
        const size_t L = p1_.L, mask = L - 1;
        out.resize(N_);
        const uint32_t* r1p = x.s1.data();
        const uint32_t* r2p = x.s2.data();
        for (size_t n = 0; n < N_; ++n) {
            const size_t idx = (L - n) & mask;
            uint32_t r1 = r1p[idx], r2 = r2p[idx];
            r1 = r1 >= kP1 ? r1 - kP1 : r1;
            r2 = r2 >= kP2 ? r2 - kP2 : r2;
            const uint32_t d = r2 >= r1 ? r2 - r1 : r2 + kP2 - r1;   // r1 < P1 < P2
            const uint32_t t = MulM<kP2>(d, inv_p1_);
            const uint64_t v = r1 + static_cast<uint64_t>(kP1) * t;   // the unique integer in [0, P1 P2)
            out[n] = ModEll(v);
        }
    }
    QSeries Mul(const QSeries& a, const QSeries& b) const {
        Spec A, B;
        Forward(a, A);
        Forward(b, B);
        MulInto(A, B);
        QSeries c;
        InverseTo(A, c);
        return c;
    }
    QSeries Square(const QSeries& a) const {
        Spec A;
        Forward(a, A);
        SquareScaled(A);
        QSeries c;
        InverseTo(A, c);
        return c;
    }

private:
    template <uint32_t MOD>
    void PointwiseScaled(uint32_t* x, const uint32_t* y, const FPlan<MOD>& P) const {
        const size_t L = P.L;
#if FNTT_HAVE_X86
        if (simd_) return PointwiseAvx2<MOD>(x, y, L, P.ninv, P.lr, P.lrp);
#endif
        const uint32_t li = P.li, lip = P.lip;
        for (size_t i = 0; i < L; ++i) x[i] = ShoupS<MOD>(MulM<MOD>(x[i], y[i]), li, lip);   // (2 MOD)^2 < 2^64
    }
    uint32_t ModEll(uint64_t v) const {   // Barrett: m = floor((2^64 - 1) / ell), q in {floor(v / ell) - 1, floor(v / ell)}
        const uint64_t q = static_cast<uint64_t>((static_cast<unsigned __int128>(v) * bar_) >> 64);
        uint64_t r = v - q * ell_;
        if (r >= ell_) r -= ell_;
        return static_cast<uint32_t>(r);
    }
    void Count(uint64_t f, uint64_t i) const {
        g_cnt_fwd.fetch_add(f, std::memory_order_relaxed);
        g_cnt_inv.fetch_add(i, std::memory_order_relaxed);
        (simd_ ? g_cnt_simd_tf : g_cnt_scalar_tf).fetch_add(f + i, std::memory_order_relaxed);
    }
    uint32_t ell_, N_, lg_;
    FPlan<kP1> p1_;
    FPlan<kP2> p2_;
    bool simd_;
    uint64_t bar_;
    uint32_t inv_p1_;
};

void Release(QSeries& s) { QSeries().swap(s); }

// ---------------------------------------------------------------------------
// (slot-24 FNTT3, docs/implementation/slot24/FNTT3_predictions.md) the three-prime, chunked engine: P1, P2 and
// P3 = 45 * 2^23 + 1 (root 7), Garner CRT to the exact integer in [0, P1 P2 P3), then mod ell.  P3's 2-adic order caps the
// transform at 2^23, so a series of length <= N is split into nb blocks of length B (nb = 1, B = N when 2N - 1 <= 2^23;
// else B = 2^kNtt3BlockLg); a product sums the block spectra pointwise per output block s, Z_s = sum_{i+j=s} F(A_i) F(B_j) / L,
// and adds the inverse of Z_s at offset s B.  Exact while nb B (ell-1)^2 < P1 P2 P3 (Fntt3Applicable): every coefficient
// of Z_s is a sum of at most (s+1) B products of residues < ell, with no cyclic wrap (2B - 1 <= L).
constexpr uint32_t kP3 = 377487361u, kG3 = 7u;
constexpr uint32_t kNtt3MaxLog = 23;      // the transform length limit of P3 (the 2-adic valuation of P3 - 1)
constexpr uint32_t kNtt3BlockLg = 22;     // the production block length 2^22 (L = 2^23)
static_assert(kP3 < kP2 && kFormsMaxEllExclusive < kP3, "FNTT3: P3 < P2 (the FNTT audit's range bounds hold), residues < P3");
// Garner's constants: P1^{-1} mod P2, P1 P2 mod P3 and its inverse mod P3
constexpr uint32_t kInvP1ModP2 = 104391568u, kP12ModP3 = 219035876u, kInvP12ModP3 = 308853288u;
static_assert(static_cast<uint64_t>(kP1) * kInvP1ModP2 % kP2 == 1, "FNTT3: P1^{-1} mod P2");
static_assert(static_cast<uint64_t>(kP1) * kP2 % kP3 == kP12ModP3, "FNTT3: P1 P2 mod P3");
static_assert(static_cast<uint64_t>(kP12ModP3) * kInvP12ModP3 % kP3 == 1, "FNTT3: (P1 P2)^{-1} mod P3");

// Barrett mod ell: m = floor((2^64 - 1) / ell), q in {floor(v / ell) - 1, floor(v / ell)} (as FormsNtt::ModEll)
inline uint32_t BarrettModEll(uint64_t v, uint32_t ell, uint64_t bar) {
    const uint64_t q = static_cast<uint64_t>((static_cast<unsigned __int128>(v) * bar) >> 64);
    uint64_t r = v - q * ell;
    if (r >= ell) r -= ell;
    return static_cast<uint32_t>(r);
}

// Three-limb Garner: the unique x in [0, P1 P2 P3) with x = r_i mod P_i (r_i in [0, 2 P_i)), reduced mod ell
// (p12_ell = P1 P2 mod ell, bar = ell's Barrett constant).  x = x12 + P1 P2 t3, x12 = r1 + P1 t2 in [0, P1 P2) < 2^57; then
// x mod ell = (x12 mod ell + (P1 P2 mod ell) t3) mod ell with (P1 P2 mod ell) t3 < 2^27 2^29: no 128-bit value.
inline uint32_t Garner3(uint32_t r1, uint32_t r2, uint32_t r3, uint32_t ell, uint64_t bar, uint32_t p12_ell) {
    r1 = r1 >= kP1 ? r1 - kP1 : r1;
    r2 = r2 >= kP2 ? r2 - kP2 : r2;
    r3 = r3 >= kP3 ? r3 - kP3 : r3;
    const uint32_t d2 = r2 >= r1 ? r2 - r1 : r2 + kP2 - r1;   // r1 < P1 < P2
    const uint32_t t2 = MulM<kP2>(d2, kInvP1ModP2);
    const uint64_t x12 = r1 + static_cast<uint64_t>(kP1) * t2;
    const uint32_t x12p3 = static_cast<uint32_t>(x12 % kP3);
    const uint32_t d3 = r3 >= x12p3 ? r3 - x12p3 : r3 + kP3 - x12p3;
    const uint32_t t3 = MulM<kP3>(d3, kInvP12ModP3);
    return BarrettModEll(BarrettModEll(x12, ell, bar) + static_cast<uint64_t>(p12_ell) * t3, ell, bar);
}

// HECKE_FORMS_NTT3 (default on; exactly "0" restores the two-prime refusal; anything else warns and stays on)
constexpr engine::lever::Policy kNtt3Policy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING HeckeFormsHeckeAlgebra: HECKE_FORMS_NTT3='%s' is neither 0 nor 1; the three-prime route stays ON "
                                    "(only exactly 0 restores the two-prime refusal)\n"};
engine::lever::EnvLever g_ntt3("HECKE_FORMS_NTT3", kNtt3Policy);
std::atomic<uint32_t> g_ntt3_block_lg{kNtt3BlockLg};    // validation probe only: a smaller block length
std::atomic<bool> g_ntt3_force{false};                  // validation probe only: the 3-prime engine wherever it is exact
std::atomic<uint64_t> g_cnt_fntt3_spaces{0}, g_cnt_fntt3_fwd{0}, g_cnt_fntt3_inv{0}, g_cnt_fntt3_pw{0}, g_cnt_fntt3_multiblock{0};

void SayNtt3EffectiveOnce() {
    if (g_ntt3.is_set()) g_ntt3.say_once(std::cout, g_ntt3.short_line("three_prime", "on", "off"));
}

uint32_t CheckedNtt3BlockLg(uint32_t lg) {   // checked before the probe scope swaps it in (nothing to restore on a throw)
    ENGINE_CHECK_MSG(lg >= 4 && lg <= kNtt3BlockLg, "ScopedFormsNtt3Probe: block_lg in [4, 22]");
    return lg;
}

struct Ntt3Layout {
    uint32_t B = 0, nb = 0, lg = 0;
};
Ntt3Layout Ntt3LayoutFor(uint32_t N) {
    Ntt3Layout y;
    uint32_t lgN = 0;
    while ((size_t{1} << lgN) < 2 * static_cast<size_t>(N) - 1) ++lgN;
    const uint32_t cap = g_ntt3_block_lg.load(std::memory_order_relaxed);
    if (lgN <= cap + 1) {
        y.B = N;
        y.nb = 1;
        y.lg = lgN;
    } else {
        y.B = 1u << cap;
        y.nb = static_cast<uint32_t>((static_cast<uint64_t>(N) + y.B - 1) / y.B);
        y.lg = cap + 1;
    }
    return y;
}

// x[i] <- x[i] y[i] / L in [0, 2 MOD) (the FNTT kernels: AVX2 Montgomery + Shoup, or MulM + Shoup)
template <uint32_t MOD, uint32_t G>
void PointwiseScaledP(uint32_t* x, const uint32_t* y, const FPlan<MOD, G>& P, bool simd) {
    const size_t L = P.L;
#if FNTT_HAVE_X86
    if (simd) return PointwiseAvx2<MOD>(x, y, L, P.ninv, P.lr, P.lrp);
#endif
    (void)simd;
    const uint32_t li = P.li, lip = P.lip;
    for (size_t i = 0; i < L; ++i) x[i] = ShoupS<MOD>(MulM<MOD>(x[i], y[i]), li, lip);
}

// One three-prime spectrum: block i of prime p at s_p[i L, (i+1) L).  `state`: 0 empty or consumed, 1 the spectrum of a
// series (Forward; MulInto's operands), 2 the spectral sums Z_s of a product (MulInto; only InverseTo reads them).
struct Spec3 {
    std::vector<uint32_t> s1, s2, s3;
    int state = 0;
};

class FormsNtt3 {
public:
    FormsNtt3(uint32_t ell, uint32_t N)
        : ell_(ell), N_(N), lay_(Ntt3LayoutFor(N)), p1_(lay_.lg), p2_(lay_.lg), p3_(lay_.lg), simd_(SimdActive()),
          bar_(~uint64_t{0} / ell), p12_ell_(static_cast<uint32_t>(static_cast<uint64_t>(kP1) * kP2 % ell)) {
        ENGINE_CHECK_MSG(lay_.lg <= kNtt3MaxLog && lay_.lg >= 4, "FormsNtt3: transform length 2^4 .. 2^23");
        ENGINE_CHECK_MSG(2 * static_cast<uint64_t>(lay_.B) - 1 <= (uint64_t{1} << lay_.lg), "FormsNtt3: 2B - 1 <= L (no cyclic wrap)");
        if (lay_.nb > 1) g_cnt_fntt3_multiblock.fetch_add(1, std::memory_order_relaxed);
    }
    size_t L() const { return p1_.L; }
    uint32_t Blocks() const { return lay_.nb; }
    // (FNTT3 audit F2) the engine's own constants and Garner step, read-only, for the validation
    uint32_t GarnerAt(uint32_t r1, uint32_t r2, uint32_t r3) const { return Garner3(r1, r2, r3, ell_, bar_, p12_ell_); }
    Fntt3EngineConstants Constants() const {
        Fntt3EngineConstants c;
        c.p1 = kP1;
        c.p2 = kP2;
        c.p3 = kP3;
        c.g3 = kG3;
        c.inv_p1_mod_p2 = kInvP1ModP2;
        c.p12_mod_p3 = kP12ModP3;
        c.inv_p12_mod_p3 = kInvP12ModP3;
        c.p12_mod_ell = p12_ell_;
        c.barrett = bar_;
        c.B = lay_.B;
        c.nb = lay_.nb;
        c.lg = lay_.lg;
        const size_t h = p1_.L >> 1;   // tw[h + 1] = w_{2h} = a primitive L-th root of unity
        c.w1 = p1_.tw[h + 1];
        c.w2 = p2_.tw[h + 1];
        c.w3 = p3_.tw[h + 1];
        return c;
    }
    // a holds at most N residues < ell
    void Forward(const QSeries& a, Spec3& s) const {
        const size_t L = p1_.L, nb = lay_.nb, B = lay_.B, la = std::min<size_t>(a.size(), N_);
        s.s1.assign(nb * L, 0);
        s.s2.assign(nb * L, 0);
        s.s3.assign(nb * L, 0);
        s.state = 1;
        for (size_t i = 0; i < nb; ++i) {
            const size_t from = std::min(i * B, la), to = std::min(from + B, la);
            const auto b0 = a.begin() + static_cast<std::ptrdiff_t>(from), b1 = a.begin() + static_cast<std::ptrdiff_t>(to);
            std::copy(b0, b1, s.s1.begin() + static_cast<std::ptrdiff_t>(i * L));
            std::copy(b0, b1, s.s2.begin() + static_cast<std::ptrdiff_t>(i * L));
            std::copy(b0, b1, s.s3.begin() + static_cast<std::ptrdiff_t>(i * L));
            p1_.Forward(s.s1.data() + i * L, simd_);
            p2_.Forward(s.s2.data() + i * L, simd_);
            p3_.Forward(s.s3.data() + i * L, simd_);
        }
        g_cnt_fntt3_fwd.fetch_add(1, std::memory_order_relaxed);
    }
    // x <- the spectral sums Z_s (s < nb) of x * y (x and y spectra of series; y may be x)
    void MulInto(Spec3& x, const Spec3& y) const {
        ENGINE_CHECK_MSG(x.state == 1 && y.state == 1, "FormsNtt3::MulInto: operands must be spectra of series");
        SumsP(x.s1.data(), y.s1.data(), p1_);
        SumsP(x.s2.data(), y.s2.data(), p2_);
        SumsP(x.s3.data(), y.s3.data(), p3_);
        g_cnt_fntt3_pw.fetch_add(static_cast<uint64_t>(lay_.nb) * (lay_.nb + 1) / 2, std::memory_order_relaxed);
        x.state = 2;
    }
    void SquareScaled(Spec3& x) const { MulInto(x, x); }
    // the inverse of every Z_s, Garner CRT, mod ell, added at offset s B; out = the first N coefficients
    void InverseTo(Spec3& x, QSeries& out) const {
        ENGINE_CHECK_MSG(x.state == 2, "FormsNtt3::InverseTo: the operand must be a product's spectral sums");
        const size_t L = p1_.L, mask = L - 1, nb = lay_.nb, B = lay_.B;
        out.assign(N_, 0);
        for (size_t s = 0; s < nb; ++s) {
            uint32_t* r1p = x.s1.data() + s * L;
            uint32_t* r2p = x.s2.data() + s * L;
            uint32_t* r3p = x.s3.data() + s * L;
            p1_.Backward(r1p, simd_);
            p2_.Backward(r2p, simd_);
            p3_.Backward(r3p, simd_);
            const size_t base = s * B, tmax = std::min<size_t>(2 * B - 1, N_ - base);
            for (size_t t = 0; t < tmax; ++t) {
                const size_t idx = (L - t) & mask;
                const uint32_t v = GarnerAt(r1p[idx], r2p[idx], r3p[idx]);
                const uint32_t o = out[base + t] + v;   // both < ell < 2^27
                out[base + t] = o >= ell_ ? o - ell_ : o;
            }
        }
        g_cnt_fntt3_inv.fetch_add(1, std::memory_order_relaxed);
        x.state = 0;   // the sums are consumed (the buffers now hold their inverses); Forward re-fills them
    }
    QSeries Mul(const QSeries& a, const QSeries& b) const {
        Spec3 A, Bs;
        Forward(a, A);
        Forward(b, Bs);
        MulInto(A, Bs);
        QSeries c;
        InverseTo(A, c);
        return c;
    }
    QSeries Square(const QSeries& a) const {
        Spec3 A;
        Forward(a, A);
        SquareScaled(A);
        QSeries c;
        InverseTo(A, c);
        return c;
    }

private:
    // Z_s = sum_{i=0}^{s} X_i Y_{s-i} / L for s = nb-1 .. 0, written over X_s: Z_s reads X_0 .. X_s only, so the
    // descending order leaves every X_i it still needs intact (also when Y is X).  Each term in [0, 2 MOD), the sum
    // kept in [0, 2 MOD) by one conditional subtraction per term (4 MOD < 2^31).
    template <uint32_t MOD, uint32_t G>
    void SumsP(uint32_t* X, const uint32_t* Y, const FPlan<MOD, G>& P) const {
        const size_t L = P.L, nb = lay_.nb;
        constexpr uint32_t m2 = 2 * MOD;
        std::vector<uint32_t> acc(L), tmp(nb > 1 ? L : 0);
        for (size_t s = nb; s-- > 0;) {
            std::copy(X + s * L, X + (s + 1) * L, acc.begin());
            PointwiseScaledP<MOD, G>(acc.data(), Y, P, simd_);   // X_s Y_0
            for (size_t i = 0; i < s; ++i) {
                std::copy(X + i * L, X + (i + 1) * L, tmp.begin());
                PointwiseScaledP<MOD, G>(tmp.data(), Y + (s - i) * L, P, simd_);
                for (size_t k = 0; k < L; ++k) {
                    const uint32_t v = acc[k] + tmp[k];
                    acc[k] = v >= m2 ? v - m2 : v;
                }
            }
            std::copy(acc.begin(), acc.end(), X + s * L);
        }
    }
    uint32_t ell_, N_;
    Ntt3Layout lay_;
    FPlan<kP1> p1_;
    FPlan<kP2> p2_;
    FPlan<kP3, kG3> p3_;
    bool simd_;
    uint64_t bar_;
    uint32_t p12_ell_;
};

// Delta = q (prod (1 - q^n))^24 as DeltaSeries, the powers through the engine (the same product mod (ell, q^N)).
template <class Engine>
QSeries DeltaFntt(const Engine& E, uint32_t ell, uint32_t N) {
    QSeries P(N, 0);
    for (int64_t m = -static_cast<int64_t>(N); m <= static_cast<int64_t>(N); ++m) {
        const int64_t e = m * (3 * m - 1) / 2;
        if (e < 0 || e >= static_cast<int64_t>(N)) continue;
        P[e] = (m % 2 == 0) ? 1u : ell - 1u;
    }
    const QSeries P2 = E.Square(P), P4 = E.Square(P2), P8 = E.Square(P4);
    const QSeries P16 = E.Square(P8), P24 = E.Mul(P16, P8);
    QSeries D(N, 0);
    for (uint32_t i = 1; i < N; ++i) D[i] = P24[i - 1];
    return D;
}

// The FNTT route.  f_j = Delta^j G_{d-j} with G_i = E_{k'} E_6^{2i}; f_0 = G_d.  The G_i are computed first and stored in
// the basis slots (G_{d-j} in f_j's slot), then the Delta^j chain multiplies each slot in place: the live series are the
// basis, Delta and one Delta^j, the live spectra F(E_6^2) (G phase) or F(Delta), F(Delta^j) (Delta phase) and one work
// spectrum.  Every slot ends as the product mod (ell, q^N) the legacy route forms, in another order of the factors.
// (slot-24 FNTT3) Engine / SpecT: FormsNtt / Spec (two primes, the FNTT route as before) or FormsNtt3 / Spec3 (three primes,
// chunked; the same calls: every MulInto / SquareScaled is followed by its InverseTo).
template <class Engine, class SpecT>
CuspSpace BuildSpaceFntt(uint32_t k, uint32_t ell, uint32_t N, bool with_constant) {
    ENGINE_CHECK_MSG(k % 2 == 0 && k >= 12, "BuildCuspSpace: even weight k >= 12");
    ENGINE_CHECK_MSG(ell >= 5 && ell < kFormsMaxEllExclusive, "BuildCuspSpace: 5 <= ell < 2^27");
    ENGINE_CHECK_MSG(N >= 2, "BuildCuspSpace: precision N >= 2");
    CuspSpace S;
    S.k = k;
    S.ell = ell;
    S.precision = N;
    uint32_t kp = k % 12;
    if (kp == 2) kp = 14;
    S.k_prime = kp;
    S.d = (k - kp) / 12;
    ENGINE_CHECK_MSG(N > S.d, "BuildCuspSpace: precision must exceed dim S_k");
    const uint32_t d = S.d, o = with_constant ? 1u : 0u;
    (std::is_same<Engine, FormsNtt>::value ? g_cnt_fntt_spaces : g_cnt_fntt3_spaces).fetch_add(1, std::memory_order_relaxed);

    const Engine E(ell, N);
    QSeries base, E6sq;
    {
        QSeries E4 = EisensteinSeries(4, ell, N), E6 = EisensteinSeries(6, ell, N);
        if (kp == 0) base = One(N);
        else if (kp == 4) base = E4;
        else if (kp == 6) base = E6;
        else if (kp == 8) base = E.Square(E4);
        else if (kp == 10) base = E.Mul(E4, E6);
        else if (kp == 14) base = E.Mul(E.Square(E4), E6);
        E6sq = E.Square(E6);
    }
    S.basis.resize(d + o);
    // the slot of G_i: f_{d-i}'s (i < d), f_0's (i = d, M_k only)
    auto slot = [&](uint32_t i) -> QSeries& { return i < d ? S.basis[o + (d - i) - 1] : S.basis[0]; };
    const uint32_t top = with_constant ? d : (d == 0 ? 0 : d - 1);   // the last G_i needed
    if (d > 0 || with_constant) {
        SpecT SE, FA;
        slot(0) = std::move(base);
        if (top >= 1) {
            E.Forward(E6sq, SE);
            for (uint32_t i = 1; i <= top; ++i) {
                E.Forward(slot(i - 1), FA);
                E.MulInto(FA, SE);
                E.InverseTo(FA, slot(i));
            }
        }
    }
    Release(E6sq);
    if (d > 0) {
        SpecT SD, SDp, FA;
        QSeries Dp = DeltaFntt(E, ell, N);
        E.Forward(Dp, SD);
        SDp = SD;
        for (uint32_t j = 1; j <= d; ++j) {
            QSeries& f = S.basis[o + j - 1];   // G_{d-j}
            E.Forward(f, FA);
            E.MulInto(FA, SDp);
            E.InverseTo(FA, f);              // f_j = Delta^j G_{d-j}
            for (uint32_t i = 0; i < j; ++i) ENGINE_CHECK_MSG(f[i] == 0, "BuildCuspSpace: f_j not O(q^j)");
            ENGINE_CHECK_MSG(f[j] == 1 % ell, "BuildCuspSpace: f_j's leading coefficient is not 1");
            if (j < d) {                     // Delta^{j+1}, its spectrum
                E.MulInto(SDp, SD);
                E.InverseTo(SDp, Dp);
                E.Forward(Dp, SDp);
            }
        }
    }
    if (with_constant) {
        ENGINE_CHECK_MSG(S.basis[0][0] == 1 % ell, "BuildModularSpace: f_0's constant term is not 1");
        S.with_constant = true;
    }
    return S;
}

// The route of a space: 0 the legacy (G2-X) code, 2 the two-prime FNTT engine, 3 (slot-24 FNTT3) the three-prime engine --
// only where the two-prime one is not exact and ell >= 2^16 (where the space was refused before FNTT3), or under the
// validation's force probe.  Every space admitted before FNTT3 keeps its route.
int FormsRoute(uint32_t ell, uint32_t N) {
    SayCacheEffectiveOnce();
    SayNtt3EffectiveOnce();   // (FNTT3 audit F6) also when every space is two-prime (prints only when HECKE_FORMS_NTT3 is set)
    if (!FormsFastMulEnabled() || !FormsNttCacheEnabled()) return 0;
    if (g_ntt3_force.load(std::memory_order_relaxed) && Fntt3Applicable(ell, N)) return 3;
    if (FnttApplicable(ell, N)) return 2;
    if (ell >= kFormsWordEll && FormsNtt3Enabled() && Fntt3Applicable(ell, N)) return 3;
    return 0;
}

}  // namespace

bool FormsNttCacheEnabled() { return g_nttcache.enabled(); }
bool FormsNttSimdAvailable() { return CpuHasAvx2(); }

bool FnttApplicable(uint32_t ell, uint32_t N) {
    // the G2-X exactness bounds for two length-N operands (min(|a|, |b|) (ell-1)^2 < P1 P2, |a| + |b| - 1 <= 2^25), and
    // the NTT threshold; ell in the forms range (ell < 2^27 < P1: every residue is a valid NTT input)
    if (ell < 5 || ell >= kFormsMaxEllExclusive || N < kFastMulNttMinN) return false;
    if (2 * static_cast<uint64_t>(N) - 1 > (uint64_t{1} << kMaxLogLen)) return false;
    return static_cast<unsigned __int128>(N) * (ell - 1) * (ell - 1) < static_cast<unsigned __int128>(kP1) * kP2;
}

QSeries MulSeriesFntt(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    if (!FnttApplicable(ell, N)) return MulSeriesFast(a, b, ell, N);
    const FormsNtt E(ell, N);
    QSeries ra(std::min<size_t>(a.size(), N)), rb(std::min<size_t>(b.size(), N));   // the engine takes reduced residues
    for (size_t i = 0; i < ra.size(); ++i) ra[i] = a[i] % ell;
    for (size_t i = 0; i < rb.size(); ++i) rb[i] = b[i] % ell;
    return (&a == &b) ? E.Square(ra) : E.Mul(ra, rb);
}

// (slot-24 FNTT3) the three-prime route's public surface
bool FormsNtt3Enabled() {
    SayNtt3EffectiveOnce();
    return g_ntt3.enabled();
}

bool Fntt3Applicable(uint32_t ell, uint32_t N) {
    if (ell < 5 || ell >= kFormsMaxEllExclusive || N < kFastMulNttMinN) return false;
    const Ntt3Layout y = Ntt3LayoutFor(N);
    const unsigned __int128 P123 = static_cast<unsigned __int128>(static_cast<uint64_t>(kP1) * kP2) * kP3;
    return static_cast<unsigned __int128>(static_cast<uint64_t>(y.nb) * y.B) * (ell - 1) * (ell - 1) < P123;
}

uint32_t Fntt3Blocks(uint32_t N) { return N == 0 ? 0 : Ntt3LayoutFor(N).nb; }

uint32_t Fntt3GarnerForValidation(uint32_t r1, uint32_t r2, uint32_t r3, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2 && ell < kFormsMaxEllExclusive && r1 < 2 * kP1 && r2 < 2 * kP2 && r3 < 2 * kP3,
                     "Fntt3GarnerForValidation: r_i in [0, 2 P_i), 2 <= ell < 2^27");
    return Garner3(r1, r2, r3, ell, ~uint64_t{0} / ell, static_cast<uint32_t>(static_cast<uint64_t>(kP1) * kP2 % ell));
}

Fntt3EngineConstants Fntt3EngineConstantsForValidation(uint32_t ell, uint32_t N) {
    ENGINE_CHECK_MSG(Fntt3Applicable(ell, N), "Fntt3EngineConstantsForValidation: outside Fntt3Applicable");
    return FormsNtt3(ell, N).Constants();
}

std::vector<uint32_t> Fntt3EngineGarnerForValidation(uint32_t ell, uint32_t N, const std::vector<uint32_t>& r) {
    ENGINE_CHECK_MSG(Fntt3Applicable(ell, N) && r.size() % 3 == 0, "Fntt3EngineGarnerForValidation: Fntt3Applicable, triples");
    const FormsNtt3 E(ell, N);
    std::vector<uint32_t> out(r.size() / 3);
    for (size_t i = 0; i < out.size(); ++i) {
        ENGINE_CHECK_MSG(r[3 * i] < 2 * kP1 && r[3 * i + 1] < 2 * kP2 && r[3 * i + 2] < 2 * kP3,
                         "Fntt3EngineGarnerForValidation: r_i in [0, 2 P_i)");
        out[i] = E.GarnerAt(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    }
    return out;
}

QSeries MulSeriesFntt3(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N) {
    ENGINE_CHECK_MSG(Fntt3Applicable(ell, N), "MulSeriesFntt3: outside the three-prime route's range (Fntt3Applicable)");
    const FormsNtt3 E(ell, N);
    QSeries ra(std::min<size_t>(a.size(), N)), rb(std::min<size_t>(b.size(), N));
    for (size_t i = 0; i < ra.size(); ++i) ra[i] = a[i] % ell;
    for (size_t i = 0; i < rb.size(); ++i) rb[i] = b[i] % ell;
    return (&a == &b) ? E.Square(ra) : E.Mul(ra, rb);
}

FormsNtt3Counters FormsNtt3CountersNow() {
    FormsNtt3Counters c;
    c.spaces = g_cnt_fntt3_spaces.load();
    c.forward = g_cnt_fntt3_fwd.load();
    c.inverse = g_cnt_fntt3_inv.load();
    c.pointwise_blocks = g_cnt_fntt3_pw.load();
    c.multiblock_engines = g_cnt_fntt3_multiblock.load();
    return c;
}

std::string FormsNtt3EffectiveLine() {
    g_ntt3.mark_said();
    const FormsNtt3Counters c = FormsNtt3CountersNow();
    return g_ntt3.effective_line("three_prime", "on", "off") + " spaces=" + std::to_string(c.spaces) +
           " forward=" + std::to_string(c.forward) + " inverse=" + std::to_string(c.inverse) +
           " pointwise_blocks=" + std::to_string(c.pointwise_blocks) + " multiblock_engines=" + std::to_string(c.multiblock_engines) +
           " (P3 = 45*2^23+1; on: spaces past the two-prime CRT bound at ell >= 2^16 take the three-prime chunked route; off: refused)";
}

ScopedFormsNtt3::ScopedFormsNtt3(bool on) : guard_(g_ntt3, on) {}
ScopedFormsNtt3Probe::ScopedFormsNtt3Probe(uint32_t block_lg, bool force)
    : saved_lg_(g_ntt3_block_lg.exchange(CheckedNtt3BlockLg(block_lg))), saved_force_(g_ntt3_force.exchange(force)) {}
ScopedFormsNtt3Probe::~ScopedFormsNtt3Probe() {
    g_ntt3_block_lg.store(saved_lg_);
    g_ntt3_force.store(saved_force_);
}

FormsNttCounters FormsNttCountersNow() {
    FormsNttCounters c;
    c.fntt_spaces = g_cnt_fntt_spaces.load();
    c.legacy_spaces = g_cnt_legacy_spaces.load();
    c.forward = g_cnt_fwd.load();
    c.inverse = g_cnt_inv.load();
    c.simd_transforms = g_cnt_simd_tf.load();
    c.scalar_transforms = g_cnt_scalar_tf.load();
    c.hoc_delayed = g_cnt_hoc_delayed.load();
    c.hoc_legacy = g_cnt_hoc_legacy.load();
    return c;
}

std::string FormsNttEffectiveLine() {
    g_nttcache.mark_said();
    const FormsNttCounters c = FormsNttCountersNow();
    return std::string("EFFECTIVE HECKE_FORMS_NTT_CACHE route=") + (FormsNttCacheEnabled() ? "fntt" : "legacy") +
           " simd=" + (SimdActive() ? "avx2" : "scalar") + " source=" + g_nttcache.source() +
           " fntt_spaces=" + std::to_string(c.fntt_spaces) + " legacy_spaces=" + std::to_string(c.legacy_spaces) +
           " forward=" + std::to_string(c.forward) + " inverse=" + std::to_string(c.inverse) + " simd_transforms=" +
           std::to_string(c.simd_transforms) + " scalar_transforms=" + std::to_string(c.scalar_transforms) +
           " hoc_delayed=" + std::to_string(c.hoc_delayed) + " hoc_legacy=" + std::to_string(c.hoc_legacy) +
           " (forms spaces: fntt = cached spectra / one pass for M_k / no Dpow-Epow tables; legacy = the G2-X route, the rollback arm)";
}

ScopedFormsNttCache::ScopedFormsNttCache(bool on) : guard_(g_nttcache, on) {}
ScopedFormsNttSimd::ScopedFormsNttSimd(bool on) : saved_(g_simd_override.exchange(on ? 1 : 0)) {}
ScopedFormsNttSimd::~ScopedFormsNttSimd() { g_simd_override.store(saved_); }

bool FormsSpaceSupported(uint32_t ell, uint32_t N) {
    SayNtt3EffectiveOnce();   // (FNTT3 audit F6) prints only when HECKE_FORMS_NTT3 is set
    if (ell < 5 || ell >= kFormsMaxEllExclusive) return false;
    if (ell < kFormsWordEll || N < kFastMulNttMinN) return true;
    return FormsFastMulEnabled() && FormsNttCacheEnabled() && (FnttApplicable(ell, N) || (FormsNtt3Enabled() && Fntt3Applicable(ell, N)));
}

uint32_t SatFormsPrecision(uint32_t k, uint32_t ell) {
    uint32_t pmax = ell;   // the largest prime <= max(31, k/12), if larger (HeckeSaturationDefect BuildForms' generator list)
    for (uint32_t q = std::max(31u, k / 12); q > pmax; --q) {
        bool pr = q >= 2;
        for (uint32_t r = 2; pr && r * r <= q; ++r)
            if (q % r == 0) pr = false;
        if (pr) {
            pmax = q;
            break;
        }
    }
    const uint64_t N = 2 * static_cast<uint64_t>(k / 12 + 1) * pmax + 2;
    return N > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(N);
}

namespace {
void CheckSupported(uint32_t ell, uint32_t N, const char* who) {
    if (ell < kFormsWordEll || ell >= kFormsMaxEllExclusive) return;   // the routes' own range checks speak there
    ENGINE_CHECK_MSG(FormsSpaceSupported(ell, N),
                     (std::string(who) + ": ell = " + std::to_string(ell) + " >= 2^16 at N = " + std::to_string(N) +
                      " needs the FNTT route (HECKE_FORMS_FASTMUL and HECKE_FORMS_NTT_CACHE not 0) within its exactness bounds "
                      "2N - 1 <= 2^25 and N (ell-1)^2 < P1 P2, or the three-prime route (HECKE_FORMS_NTT3 not 0; "
                      "nb B (ell-1)^2 < P1 P2 P3) (FormsSpaceSupported); the naive route would be O(N^2)").c_str());
}
}  // namespace

CuspSpace BuildCuspSpace(uint32_t k, uint32_t ell, uint32_t N) {
    CheckSupported(ell, N, "BuildCuspSpace");
    const int route = FormsRoute(ell, N);
    if (route == 2) return BuildSpaceFntt<FormsNtt, Spec>(k, ell, N, false);
    if (route == 3) return BuildSpaceFntt<FormsNtt3, Spec3>(k, ell, N, false);
    g_cnt_legacy_spaces.fetch_add(1, std::memory_order_relaxed);
    return BuildCuspSpaceLegacy(k, ell, N);
}

CuspSpace BuildModularSpace(uint32_t k, uint32_t ell, uint32_t N) {
    CheckSupported(ell, N, "BuildModularSpace");
    const int route = FormsRoute(ell, N);
    if (route == 2) return BuildSpaceFntt<FormsNtt, Spec>(k, ell, N, true);
    if (route == 3) return BuildSpaceFntt<FormsNtt3, Spec3>(k, ell, N, true);
    g_cnt_legacy_spaces.fetch_add(1, std::memory_order_relaxed);
    return BuildModularSpaceLegacy(k, ell, N);
}

ExtMatrix HeckeOnCusp(const CuspSpace& S, uint32_t p) {
    ENGINE_CHECK_MSG(p >= 2, "HeckeOnCusp: p >= 2");
    // (slot-12 WP-12) m = dim of the space (d, or d + 1 on M_k); the coordinates
    // are read from a_o .. a_{o + m - 1}, o = 1 on S_k and 0 on M_k (with
    // a_0(T_p f) = (1 + p^{k-1}) a_0(f)); on S_k every value is as before.
    const uint32_t m = static_cast<uint32_t>(S.basis.size());
    const uint32_t o = S.with_constant ? 0u : 1u;
    ENGINE_CHECK_MSG(static_cast<uint64_t>(S.d) * p < S.precision,
                     "HeckeOnCusp: precision must exceed d * p (a_{dp} is needed)");
    const uint32_t ell = S.ell;
    const uint32_t M = (S.precision - 1) / p;   // a_n(T_p f) known for n <= M
    const uint32_t pk1 = PowMod(p, S.k - 1, ell);
    ExtMatrix T(m, m, ell);
    if (static_cast<unsigned __int128>(m + 1) * ell * ell < (static_cast<unsigned __int128>(1) << 64) && FormsFastMulEnabled() &&
        FormsNttCacheEnabled()) {
        // (slot-24 FNTT) the same elimination in two passes.  (1) The coordinates: the legacy loop reads c = g[i + o]
        // after the subtractions of the earlier i' < i, and those reads only touch a_o .. a_{o+m-1}, so the loop restricted
        // to those m indices gives the same T.  (2) The residual g[n] = a_n(T_p f_j) - sum_i c_ij a_n(f_i) for every n <= M,
        // in chunks of n (all columns per chunk, so the basis rows stay in cache), with the reduction mod ell delayed: g[n]
        // starts < ell and gains at most m terms (ell - c) b[n] < ell^2, and (m + 1) ell^2 < 2^64 (checked: any ell < 2^16 with
        // m < 2^31; ell = 10^5 with m < 1.8e9), so the uint64_t cannot wrap.  The
        // residual is zero mod ell iff the legacy one is; on a non-zero residual the error names the legacy loop's
        // (first column, then first n), so the message is the same.
        g_cnt_hoc_delayed.fetch_add(1, std::memory_order_relaxed);
        const bool simd = SimdActive();   // the AVX2 axpy (the same uint64_t sums)
        auto a_n = [&](const QSeries& f, uint32_t n) -> uint64_t {   // a_n(T_p f) mod ell
            uint64_t v = f[static_cast<uint64_t>(n) * p];
            if (n % p == 0) v += static_cast<uint64_t>(pk1) * f[n / p];
            return v % ell;
        };
        // (1) coordinates, column by column, on the indices o .. o + m - 1
        {
            std::vector<uint64_t> h(m);
            for (uint32_t j = 0; j < m; ++j) {
                const QSeries& f = S.basis[j];
                for (uint32_t t = 0; t < m; ++t) h[t] = a_n(f, t + o);
                for (uint32_t i = 0; i < m; ++i) {
                    const uint32_t c = static_cast<uint32_t>(h[i]);
                    T.data[i][j] = c;
                    if (!c) continue;
                    const QSeries& bb = S.basis[i];
                    for (uint32_t t = i; t < m; ++t) h[t] = (h[t] + static_cast<uint64_t>(ell - c) * bb[t + o]) % ell;
                }
            }
        }
        // (2) the residual on n = o .. M, chunk by chunk
        constexpr uint32_t kChunk = 1024;
        std::vector<uint64_t> g(static_cast<size_t>(m) * kChunk);
        std::vector<uint32_t> bad(m, UINT32_MAX);   // per column, the first n with a non-zero residual
        for (uint32_t n0 = o; n0 <= M; n0 += kChunk) {
            const uint32_t n1 = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(M) + 1, static_cast<uint64_t>(n0) + kChunk));
            const uint32_t len = n1 - n0;
            for (uint32_t j = 0; j < m; ++j) {
                uint64_t* gj = g.data() + static_cast<size_t>(j) * kChunk;
                for (uint32_t t = 0; t < len; ++t) gj[t] = a_n(S.basis[j], n0 + t);
            }
            for (uint32_t i = 0; i < m; ++i) {
                const uint32_t* bb = S.basis[i].data() + n0;
                for (uint32_t j = 0; j < m; ++j) {
                    const uint32_t c = T.data[i][j];
                    if (!c) continue;
                    const uint64_t mc = ell - c;
                    uint64_t* gj = g.data() + static_cast<size_t>(j) * kChunk;
#if FNTT_HAVE_X86
                    if (simd) {
                        AxpyU64Avx2(gj, bb, mc, 0, len);
                        continue;
                    }
#endif
                    for (uint32_t t = 0; t < len; ++t) gj[t] += mc * bb[t];
                }
            }
            for (uint32_t j = 0; j < m; ++j) {
                if (bad[j] != UINT32_MAX) continue;
                const uint64_t* gj = g.data() + static_cast<size_t>(j) * kChunk;
                for (uint32_t t = 0; t < len; ++t)
                    if (gj[t] % ell != 0) {
                        bad[j] = n0 + t;
                        break;
                    }
            }
        }
        for (uint32_t j = 0; j < m; ++j)
            if (bad[j] != UINT32_MAX)
                throw std::runtime_error("HeckeOnCusp: T_" + std::to_string(p) + " f_" + std::to_string(j + o)
                                         + " not in the space to the available precision (a_" + std::to_string(bad[j]) + ")");
        return T;
    }
    g_cnt_hoc_legacy.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t j = 0; j < m; ++j) {
        const QSeries& f = S.basis[j];
        std::vector<uint64_t> g(M + 1, 0);
        for (uint32_t n = o; n <= M; ++n) {
            uint64_t v = f[static_cast<uint64_t>(n) * p];
            if (n % p == 0) v += static_cast<uint64_t>(pk1) * f[n / p];
            g[n] = v % ell;
        }
        // Unitriangular coordinates from a_o .. a_{o+m-1}, then the residual to n <= M.
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t c = static_cast<uint32_t>(g[i + o]);
            T.data[i][j] = c;
            if (!c) continue;
            const QSeries& bb = S.basis[i];
            for (uint32_t n = o; n <= M; ++n)
                g[n] = (g[n] + static_cast<uint64_t>(ell - c) * bb[n]) % ell;
        }
        for (uint32_t n = o; n <= M; ++n)
            if (g[n] != 0)
                throw std::runtime_error("HeckeOnCusp: T_" + std::to_string(p) + " f_" + std::to_string(j + o)
                                         + " not in the space to the available precision (a_" + std::to_string(n) + ")");
    }
    return T;
}

std::vector<ExtMatrix> SturmRangeOperators(const CuspSpace& S) {
    ENGINE_CHECK_MSG(!S.with_constant, "SturmRangeOperators: S_k only");
    const uint32_t d = S.d, ell = S.ell;
    std::vector<ExtMatrix> T(d + 1, ExtMatrix(d, ell));   // T[0] unused, T[1] = I
    std::vector<bool> done(d + 1, false);
    if (d >= 1) done[1] = true;
    for (uint32_t p = 2; p <= d; ++p) {
        bool prime = true;
        for (uint32_t q = 2; q * q <= p; ++q)
            if (p % q == 0) prime = false;
        if (!prime) continue;
        const ExtMatrix Tp = HeckeOnCusp(S, p);
        const uint32_t pk1 = PowMod(p, S.k - 1, ell);
        uint32_t prev = 1, cur = p;
        T[p] = Tp;
        done[p] = true;
        while (static_cast<uint64_t>(cur) * p <= d) {
            const uint32_t next = cur * p;
            T[next] = Sub(Tp * T[cur], Scale(T[prev], pk1));
            done[next] = true;
            prev = cur;
            cur = next;
        }
    }
    for (uint32_t n = 2; n <= d; ++n) {
        if (done[n]) continue;
        // n = q^a * m with q the smallest prime factor, gcd(q^a, m) = 1, both < n.
        uint32_t q = 2;
        while (n % q) ++q;
        uint32_t qa = 1;
        while (n % (qa * q) == 0) qa *= q;
        const uint32_t m = n / qa;
        ENGINE_CHECK_MSG(done[qa] && done[m], "SturmRangeOperators: factor not ready");
        T[n] = T[qa] * T[m];
        done[n] = true;
    }
    T.erase(T.begin());
    return T;
}

FormsBlock LocaliseAtSystem(const CuspSpace& S, const std::vector<uint64_t>& primes,
                            const std::vector<ExtMatrix>& T,
                            const std::vector<uint64_t>& system_primes,
                            const std::vector<uint32_t>& system) {
    ENGINE_CHECK_MSG(primes.size() == T.size(), "LocaliseAtSystem: primes and T not aligned");
    ENGINE_CHECK_MSG(system_primes.size() == system.size(), "LocaliseAtSystem: system not aligned");
    FormsBlock B;
    B.primes = primes;
    B.system_primes = system_primes;
    B.system = system;
    const uint32_t d = static_cast<uint32_t>(S.basis.size()), ell = S.ell;   // (WP-12) dim of the space
    if (d == 0) return B;
    ExtMatrix stack(1, d, ell);
    bool first = true;
    for (size_t s = 0; s < system_primes.size(); ++s) {
        const auto it = std::find(primes.begin(), primes.end(), system_primes[s]);
        ENGINE_CHECK_MSG(it != primes.end(), "LocaliseAtSystem: a system prime is not among the operators");
        const ExtMatrix N = MatPow(ShiftByScalar(T[it - primes.begin()], system[s]), d);
        stack = first ? N : hecke::VStack(stack, N);
        first = false;
    }
    B.V = hecke::KernelAsMatrix(stack);
    B.dim = B.V.cols;
    if (B.dim == 0) return B;
    for (size_t k = 0; k < primes.size(); ++k)
        B.t.push_back(hecke::SolveColumnsExact(B.V, T[k] * B.V, "LocaliseAtSystem"));
    B.A = ComputeAlgebraInvariants(B.t, ell);
    return B;
}

std::vector<uint32_t> NilpotentJordanType(const ExtMatrix& N) {
    const uint32_t D = N.rows;
    std::vector<uint32_t> ranks{D};
    ExtMatrix P = N;
    for (uint32_t i = 1; ranks.back() != 0; ++i) {
        ENGINE_CHECK_MSG(i <= D + 1, "NilpotentJordanType: not nilpotent");
        ranks.push_back(hecke::RankOf(P));
        ENGINE_CHECK_MSG(ranks.back() < ranks[ranks.size() - 2], "NilpotentJordanType: not nilpotent");
        P = P * N;
    }
    // c_i = #blocks of size >= i = ranks[i-1] - ranks[i]
    std::vector<uint32_t> ge;
    for (size_t i = 1; i < ranks.size(); ++i) ge.push_back(ranks[i - 1] - ranks[i]);
    std::vector<uint32_t> sizes;
    for (size_t i = 0; i < ge.size(); ++i) {
        const uint32_t exact = ge[i] - (i + 1 < ge.size() ? ge[i + 1] : 0);
        for (uint32_t c = 0; c < exact; ++c) sizes.push_back(static_cast<uint32_t>(i + 1));
    }
    std::sort(sizes.rbegin(), sizes.rend());
    return sizes;
}

int64_t RationalEigenvalue(const ExtMatrix& t) {
    ENGINE_CHECK_MSG(t.rows == t.cols && t.p < (1u << 16), "RationalEigenvalue: square, ell < 2^16");
    if (t.rows == 0) return -1;
    for (uint32_t lambda = 0; lambda < t.p; ++lambda)
        if (IsZero(MatPow(ShiftByScalar(t, lambda), t.rows))) return lambda;
    return -1;
}

} // namespace forms
} // namespace hecke_ext
