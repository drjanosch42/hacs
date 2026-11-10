// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// RibetField.cpp -- see RibetField.h and T2_claim_for_F4.md.
#include "RibetField.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "Cyclotomic.h"
#include "ModUtil.h"
#include "Mont.h"
#include "NfCheck.h"
#include "Parallel.h"

namespace nf {

namespace {

double now_secs() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint32_t umod(int64_t a, uint32_t m) {
    const int64_t r = a % static_cast<int64_t>(m);
    return static_cast<uint32_t>(r < 0 ? r + m : r);
}

// log |2 sin(pi m / ell)| = log |1 - z^m| in double precision, m mod ell != 0.
double log_s_double(uint32_t m, uint32_t ell) {
    return std::log(2.0 * std::sin(3.14159265358979323846 * static_cast<double>(m % ell) / static_cast<double>(ell)));
}

}  // namespace

RibetFieldSpec ribet_field_spec(uint32_t ell, uint32_t i) {
    NF_CHECK_MSG(ell >= 5 && is_prime_u32(ell), "ribet_field_spec: ell must be a prime >= 5");
    NF_CHECK_MSG(i % 2 == 0 && i % (ell - 1) != 0, "ribet_field_spec: i must be even and != 0 mod ell - 1");
    i %= (ell - 1);
    NF_CHECK_MSG(std::gcd<uint64_t>((i + ell - 2) % (ell - 1), ell - 1) == 1,
                 "ribet_field_spec: omega^{1-i} is not faithful (gcd(i - 1, ell - 1) != 1)");
    RibetFieldSpec S;
    S.ell = ell;
    S.i = i;
    S.f = kummer_exponents(ell, 1, i);
    for (const auto& kv : S.f) S.F += kv.second;
    S.g = primitive_root_prime(ell);
    S.e = powmod32(S.g, i, ell);
    S.spow.assign(ell, 0);
    for (uint32_t s = 1; s < ell; ++s) S.spow[s] = powmod32(s, i, ell);

    // sigma_g(eta) = prod_c u_{gc}^{f_c} u_g^{-F} (u_1 = 1): the exponent of u_d is
    // f_{d/g} for d != g and -F for d = g. gamma = (sigma_g eta / eta^e)^{1/ell}.
    const uint32_t ginv = invmod32(S.g, ell);
    S.gamma.assign(ell, 0);
    for (uint32_t d = 2; d < ell; ++d) {
        const int64_t Ed = (d == S.g) ? -S.F : S.f.at(mulmod32(d, ginv, ell));
        const int64_t num = Ed - static_cast<int64_t>(S.e) * S.f.at(d);
        NF_CHECK_MSG(num % static_cast<int64_t>(ell) == 0, "ribet_field_spec: sigma_g(eta)/eta^e is not an ell-th power of the u_d");
        S.gamma[d] = num / static_cast<int64_t>(ell);
    }
    // arg gamma(z) = (pi/ell) M, M = sum_d c_d (d - 1); gamma_R = zeta^m gamma real <=> M + 2m = 0 mod ell.
    int64_t M = 0;
    for (uint32_t d = 2; d < ell; ++d) M += S.gamma[d] * static_cast<int64_t>(d - 1);
    S.twist = mulmod32(umod(-M, ell), invmod32(2, ell), ell);
    // gamma_R real in every embedding s: sum_d c_d ((s d mod ell) - s) + 2 s m = 0 mod ell.
    for (uint32_t s = 1; s < ell; ++s) {
        int64_t A = 2 * static_cast<int64_t>(s) * S.twist;
        for (uint32_t d = 2; d < ell; ++d) A += S.gamma[d] * (static_cast<int64_t>(mulmod32(s, d, ell)) - s);
        NF_CHECK_MSG(A % static_cast<int64_t>(ell) == 0, "ribet_field_spec: gamma_R not real");
    }
    // signs of sigma_s(eta): arg = (pi/ell) N_s, N_s = sum_c f_c ((s c mod ell) - s).
    S.sign.assign(ell, 0);
    S.log_b.assign(ell, 0.0);
    std::vector<double> Ld(ell, 0.0);
    for (uint32_t m = 1; m < ell; ++m) Ld[m] = log_s_double(m, ell);
    double lmax = -1e300;
    for (uint32_t s = 1; s < ell; ++s) {
        int64_t N = 0;
        double lg = -static_cast<double>(S.F) * Ld[s];
        for (const auto& kv : S.f) {
            N += kv.second * (static_cast<int64_t>(mulmod32(s, kv.first, ell)) - s);
            lg += static_cast<double>(kv.second) * Ld[mulmod32(s, kv.first, ell)];
        }
        NF_CHECK_MSG(N % static_cast<int64_t>(ell) == 0, "ribet_field_spec: sigma_s(eta) not real");
        const int64_t h = N / static_cast<int64_t>(ell);
        S.sign[s] = (h % 2 == 0) ? 1 : -1;
        S.log_b[s] = lg / static_cast<double>(ell);
        lmax = std::max(lmax, S.log_b[s]);
    }
    // n_0 = eta mod pi (u_c = c mod pi), lifted to (-ell/2, ell/2).
    {
        uint32_t r = 1;
        for (const auto& kv : S.f) r = mulmod32(r, powmod32(kv.first, umod(kv.second, ell - 1), ell), ell);
        S.n0 = r > ell / 2 ? static_cast<int64_t>(r) - ell : static_cast<int64_t>(r);
    }
    // log2 (1 + sum_s |b_s|) by log-sum-exp; max |coeff of P| <= prod_j (1 + |theta_j|) <= (1 + sum |b_s|)^ell.
    double acc = std::exp(-lmax);
    for (uint32_t s = 1; s < ell; ++s) acc += std::exp(S.log_b[s] - lmax);
    const double l2 = (lmax + std::log(acc)) / std::log(2.0);
    S.bound_bits = static_cast<double>(ell) * std::max(l2, 0.0) * (1.0 + 1e-9) + 2.0;
    return S;
}

double ribet_complement_check(const RibetFieldSpec& S, bool* signs_ok) {
    const uint32_t ell = S.ell;
    std::vector<double> Ld(ell, 0.0);
    for (uint32_t m = 1; m < ell; ++m) Ld[m] = log_s_double(m, ell);
    int64_t Csum = 0;
    for (uint32_t d = 2; d < ell; ++d) Csum += S.gamma[d];
    double worst = 0;
    bool ok = true;
    for (uint32_t s = 1; s < ell; ++s) {
        // log |sigma_s(gamma_R)| and its sign.
        double lg = -static_cast<double>(Csum) * Ld[s];
        int64_t A = 2 * static_cast<int64_t>(s) * S.twist;
        for (uint32_t d = 2; d < ell; ++d) {
            lg += static_cast<double>(S.gamma[d]) * Ld[mulmod32(s, d, ell)];
            A += S.gamma[d] * (static_cast<int64_t>(mulmod32(s, d, ell)) - s);
        }
        const int sg = ((A / static_cast<int64_t>(ell)) % 2 == 0) ? 1 : -1;
        const uint32_t sg_s = mulmod32(s, S.g, ell);
        const double diff = static_cast<double>(S.e) * S.log_b[s] + lg - S.log_b[sg_s];
        worst = std::max(worst, std::fabs(diff));
        const int lhs_sign = ((S.e % 2 == 1) ? S.sign[s] : 1) * sg;
        if (lhs_sign != S.sign[sg_s]) ok = false;
    }
    if (signs_ok) *signs_ok = ok;
    return worst;
}

RibetNumeric ribet_numeric(const RibetFieldSpec& S, unsigned prec, unsigned threads) {
    const uint32_t ell = S.ell, h = (ell - 1) / 2;
    const unsigned wp = prec + 48;
    const MpContext C(wp);
    RibetNumeric N;
    N.prec = prec;
    double t0 = now_secs();
    // w = e^{i pi/ell}; 2 sin(pi m/ell) = 2 Im w^m; z = w^2.
    const MpComplex w = C.expi(mp_div_u32(C.pi(), ell, wp));
    std::vector<MpReal> Sm(h + 1);
    {
        MpComplex p = w;
        for (uint32_t m = 1; m <= h; ++m) {
            Sm[m] = mp_ldexp(p.im, 1);
            p = mp_cmul(p, w, wp);
        }
    }
    const MpComplex z = mp_cmul(w, w, wp);
    N.zpow.resize(ell);
    N.zpow[0] = MpComplex{mp_from_i64(1), MpReal{}};
    for (uint32_t k = 1; k < ell; ++k) N.zpow[k] = mp_cmul(N.zpow[k - 1], z, wp);
    std::vector<MpReal> L(h + 1);
    parallel_ranges(h, threads, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t m = b; m < e; ++m) L[m + 1] = C.log(Sm[m + 1]);
    });
    N.secs_logs = now_secs() - t0;
    t0 = now_secs();
    // log b_s = (sum_c f_c L(sc) - F L(s)) / ell, L(x) = L(min(x, ell - x)); s = 1 .. h.
    auto fold = [ell](uint32_t x) { return std::min(x, ell - x); };
    N.b.assign(ell, MpReal{});
    parallel_ranges(h, threads, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t si = b; si < e; ++si) {
            const uint32_t s = static_cast<uint32_t>(si + 1);
            std::vector<int64_t> A(h + 1, 0);
            for (const auto& kv : S.f) A[fold(mulmod32(s, kv.first, ell))] += kv.second;
            A[fold(s)] -= S.F;
            MpReal acc;
            for (uint32_t m = 1; m <= h; ++m)
                if (A[m] != 0) acc = mp_add(acc, mp_mul_i64(L[m], A[m], wp + 64), wp + 32);
            const MpReal lb = mp_div_u32(acc, ell, wp);
            MpReal bs = C.exp(lb);
            if (S.sign[s] < 0) bs = mp_neg(bs);
            N.b[s] = bs;
        }
    });
    for (uint32_t s = 1; s <= h; ++s) N.b[ell - s] = N.b[s];
    N.secs_exps = now_secs() - t0;
    t0 = now_secs();
    // theta_j = sum_t B_t z^{j t}, B_t = sum_{s^i = t} b_s.
    std::vector<MpReal> B(ell);
    std::vector<uint32_t> T;
    for (uint32_t s = 1; s < ell; ++s) B[S.spow[s]] = mp_add(B[S.spow[s]], N.b[s], wp);
    for (uint32_t t = 1; t < ell; ++t)
        if (!mp_is_zero(B[t])) T.push_back(t);
    N.theta.assign(ell, MpComplex{});
    parallel_ranges(h + 1, threads, [&](std::size_t b, std::size_t e, unsigned) {
        for (std::size_t j = b; j < e; ++j) {
            MpReal re, im;
            for (uint32_t t : T) {
                const MpComplex& zz = N.zpow[(j * t) % ell];
                re = mp_add(re, mp_mul(B[t], zz.re, wp), wp);
                im = mp_add(im, mp_mul(B[t], zz.im, wp), wp);
            }
            N.theta[j] = MpComplex{re, im};
        }
    });
    for (uint32_t j = 1; j <= h; ++j) N.theta[ell - j] = MpComplex{N.theta[j].re, mp_neg(N.theta[j].im)};
    N.secs_theta = now_secs() - t0;
    return N;
}

std::vector<MpComplex> ribet_psi_conjugates(const RibetFieldSpec& S, const RibetNumeric& N, const std::vector<int64_t>& y) {
    const uint32_t ell = S.ell;
    const unsigned wp = N.prec + 48;
    std::vector<MpComplex> out(ell);
    for (uint32_t j = 0; j < ell; ++j) {
        std::vector<MpReal> c(ell);  // coefficient of z^k
        for (uint32_t s = 1; s < ell; ++s) {
            for (std::size_t a = 0; a < y.size(); ++a) {
                if (y[a] == 0) continue;
                const uint32_t k = static_cast<uint32_t>((static_cast<uint64_t>(a) * s + static_cast<uint64_t>(j) * S.spow[s]) % ell);
                c[k] = mp_add(c[k], mp_mul_i64(N.b[s], y[a], wp), wp);
            }
        }
        MpReal re, im;
        for (uint32_t k = 0; k < ell; ++k) {
            if (mp_is_zero(c[k])) continue;
            re = mp_add(re, mp_mul(c[k], N.zpow[k].re, wp), wp);
            im = mp_add(im, mp_mul(c[k], N.zpow[k].im, wp), wp);
        }
        out[j] = MpComplex{re, im};
    }
    return out;
}

namespace {
using RPoly = std::vector<MpReal>;  // constant term first

RPoly rpoly_mul(const RPoly& a, const RPoly& b, unsigned prec, unsigned threads) {
    RPoly c(a.size() + b.size() - 1);
    parallel_ranges(c.size(), threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t k = lo; k < hi; ++k) {
            MpReal acc;
            const std::size_t i0 = k >= b.size() - 1 ? k - (b.size() - 1) : 0;
            const std::size_t i1 = std::min(k, a.size() - 1);
            for (std::size_t i = i0; i <= i1; ++i) acc = mp_add(acc, mp_mul(a[i], b[k - i], prec), prec);
            c[k] = acc;
        }
    });
    return c;
}
}  // namespace

IntPoly poly_from_conjugates(const std::vector<MpComplex>& x, unsigned prec, unsigned threads, double* max_log2_dist) {
    const std::size_t n = x.size();
    NF_CHECK_MSG(n % 2 == 1, "poly_from_conjugates: odd degree (one real conjugate) expected");
    const unsigned wp = prec + 48;
    std::vector<RPoly> level;
    level.push_back(RPoly{mp_neg(x[0].re), mp_from_i64(1)});
    for (std::size_t j = 1; j <= n / 2; ++j)
        level.push_back(RPoly{mp_cnorm(x[j], wp), mp_neg(mp_ldexp(x[j].re, 1)), mp_from_i64(1)});
    while (level.size() > 1) {
        std::vector<RPoly> next((level.size() + 1) / 2);
        const std::size_t pairs = level.size() / 2;
        if (pairs >= threads) {
            parallel_ranges(pairs, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
                for (std::size_t p = lo; p < hi; ++p) next[p] = rpoly_mul(level[2 * p], level[2 * p + 1], wp, 1);
            });
        } else {
            for (std::size_t p = 0; p < pairs; ++p) next[p] = rpoly_mul(level[2 * p], level[2 * p + 1], wp, threads);
        }
        if (level.size() % 2 == 1) next.back() = level.back();
        level.swap(next);
    }
    IntPoly P(n + 1);
    double worst = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k <= n; ++k) {
        double d;
        P[k] = mp_round(level[0][k], &d);
        worst = std::max(worst, d);
    }
    if (max_log2_dist) *max_log2_dist = worst;
    return P;
}

// ---------------------------------------------------------------------------
// The exact modular route.
namespace {

struct Mod {
    Mont64 M;
    uint64_t one;
    explicit Mod(uint64_t q) : M(q), one(M.to(1)) {}
    uint64_t mul(uint64_t a, uint64_t b) const { return M.mul(a, b); }
    uint64_t pw(uint64_t a, uint64_t e) const {  // Montgomery in, Montgomery out
        uint64_t r = one, b = a;
        while (e) {
            if (e & 1u) r = M.mul(r, b);
            e >>= 1;
            if (e) b = M.mul(b, b);
        }
        return r;
    }
    uint64_t inv(uint64_t a) const { return pw(a, M.p - 2); }
};

bool is_prime_mont(uint64_t n) {
    for (uint64_t p : {3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull, 41ull, 43ull, 47ull, 53ull}) {
        if (n % p == 0) return n == p;
    }
    const Mod R(n);
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        ++s;
    }
    const uint64_t mone = R.M.to(n - 1);
    for (uint64_t a : {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull}) {
        const uint64_t am = a % n;
        if (am == 0) continue;
        uint64_t x = R.pw(R.M.to(am), d);
        if (x == R.one || x == mone) continue;
        bool comp = true;
        for (unsigned r = 1; r < s; ++r) {
            x = R.mul(x, x);
            if (x == mone) {
                comp = false;
                break;
            }
        }
        if (comp) return false;
    }
    return true;
}

// a^{-1} mod m (gcd = 1), m < 2^62.
uint64_t inv_mod_u64(uint64_t a, uint64_t m) {
    int64_t t = 0, nt = 1;
    int64_t r = static_cast<int64_t>(m), nr = static_cast<int64_t>(a % m);
    while (nr != 0) {
        const int64_t qq = r / nr;
        int64_t tmp = t - qq * nt;
        t = nt;
        nt = tmp;
        tmp = r - qq * nr;
        r = nr;
        nr = tmp;
    }
    NF_CHECK(r == 1);
    if (t < 0) t += static_cast<int64_t>(m);
    return static_cast<uint64_t>(t);
}

// Bit-plane plan for prod_d lambda[d t]^{E_d}, d = 1 .. ell - 1.
struct ExpPlan {
    unsigned bits = 0;
    std::vector<std::vector<uint32_t>> pos, neg;  // pos[b] = {d : E_d > 0, bit b of E_d}
    explicit ExpPlan(const std::vector<int64_t>& E) {
        int64_t mx = 0;
        for (int64_t v : E) mx = std::max<int64_t>(mx, v < 0 ? -v : v);
        while ((int64_t(1) << bits) <= mx) ++bits;
        pos.assign(bits, {});
        neg.assign(bits, {});
        for (std::size_t d = 1; d < E.size(); ++d) {
            const int64_t v = E[d];
            const uint64_t a = static_cast<uint64_t>(v < 0 ? -v : v);
            for (unsigned b = 0; b < bits; ++b)
                if ((a >> b) & 1u) (v > 0 ? pos : neg)[b].push_back(static_cast<uint32_t>(d));
        }
    }
    // lambda: lambda[m] = 1 - r^m (Montgomery), m = 0 .. ell - 1 (lambda[0] unused).
    uint64_t eval(const Mod& R, const std::vector<uint64_t>& lambda, uint32_t t, uint32_t ell) const {
        uint64_t P = R.one, Nn = R.one;
        for (unsigned b = bits; b-- > 0;) {
            P = R.mul(P, P);
            Nn = R.mul(Nn, Nn);
            for (uint32_t d : pos[b]) P = R.mul(P, lambda[(static_cast<uint64_t>(d) * t) % ell]);
            for (uint32_t d : neg[b]) Nn = R.mul(Nn, lambda[(static_cast<uint64_t>(d) * t) % ell]);
        }
        return R.mul(P, R.inv(Nn));
    }
};

struct PrimeResult {
    uint64_t q = 0;
    bool split = false;
    bool chain_ok = true;
    std::vector<uint64_t> res;  // P mod q, normal form
};

PrimeResult process_prime(const RibetFieldSpec& S, const ExpPlan& eta_plan, const ExpPlan& gam_plan, uint64_t q) {
    const uint32_t ell = S.ell;
    PrimeResult out;
    out.q = q;
    const Mod R(q);
    const uint64_t m1 = (q - 1) / ell;
    // r of order ell.
    uint64_t r = 0;
    for (uint64_t hh = 2;; ++hh) {
        r = R.pw(R.M.to(hh), m1);
        if (r != R.one) break;
    }
    std::vector<uint64_t> rpow(ell), lambda(ell);
    rpow[0] = R.one;
    for (uint32_t k = 1; k < ell; ++k) rpow[k] = R.mul(rpow[k - 1], r);
    for (uint32_t k = 1; k < ell; ++k) lambda[k] = R.M.sub(R.one, rpow[k]);
    const uint64_t eta = eta_plan.eval(R, lambda, 1, ell);
    if (R.pw(eta, m1) != R.one) return out;  // not split completely in H
    out.split = true;
    // beta_0 = eta^{1/ell} = eta^{ell^{-1} mod m1} (gcd(ell, m1) = 1 as q != 1 mod ell^2).
    const uint64_t beta0 = R.pw(eta, inv_mod_u64(ell, m1));
    NF_CHECK(R.pw(beta0, ell) == eta);
    std::vector<uint64_t> beta(ell);
    beta[0] = beta0;
    uint32_t t = 1;  // g^k mod ell
    for (uint32_t k = 0; k + 1 < ell; ++k) {
        // gamma_R(r^t) = r^{t m} gamma(r^t)
        const uint64_t gam = R.mul(rpow[(static_cast<uint64_t>(t) * S.twist) % ell], gam_plan.eval(R, lambda, t, ell));
        beta[k + 1] = R.mul(R.pw(beta[k], S.e), gam);
        t = mulmod32(t, S.g, ell);
    }
    out.chain_ok = (beta[ell - 1] == beta[0]);
    // B_t = sum_k beta_k over e^k = t; theta_j = sum_t B_t r^{j t}.
    std::vector<uint64_t> Bt(ell, 0);
    uint32_t ek = 1;
    for (uint32_t k = 0; k + 1 < ell; ++k) {
        Bt[ek] = R.M.add(Bt[ek], beta[k]);
        ek = mulmod32(ek, S.e, ell);
    }
    std::vector<uint32_t> T;
    for (uint32_t tt = 1; tt < ell; ++tt)
        if (Bt[tt] != 0) T.push_back(tt);
    std::vector<uint64_t> c(ell + 1, 0);  // prod (X - theta_j), Montgomery
    c[0] = R.one;
    std::size_t deg = 0;
    for (uint32_t j = 0; j < ell; ++j) {
        uint64_t th = 0;
        for (uint32_t tt : T) th = R.M.add(th, R.mul(Bt[tt], rpow[(static_cast<uint64_t>(j) * tt) % ell]));
        // c <- c * (X - th)
        c[deg + 1] = c[deg];
        for (std::size_t k = deg; k >= 1; --k) c[k] = R.M.sub(c[k - 1], R.mul(th, c[k]));
        c[0] = R.M.sub(0, R.mul(th, c[0]));
        ++deg;
    }
    out.res.resize(ell + 1);
    for (uint32_t k = 0; k <= ell; ++k) out.res[k] = R.M.from(c[k]);
    return out;
}

}  // namespace

RibetModular ribet_poly_modular(const RibetFieldSpec& S, unsigned threads) {
    const uint32_t ell = S.ell;
    std::vector<int64_t> E(ell, 0), G(ell, 0);
    E[1] = -S.F;
    for (const auto& kv : S.f) E[kv.first] = kv.second;
    int64_t Cs = 0;
    for (uint32_t d = 2; d < ell; ++d) {
        G[d] = S.gamma[d];
        Cs += S.gamma[d];
    }
    G[1] = -Cs;
    const ExpPlan eta_plan(E), gam_plan(G);
    RibetModular out;
    const double need = S.bound_bits + 2.0;  // modulus > 2 * bound
    // Candidates q = 1 + k ell below 2^62, descending, in blocks (deterministic order).
    const uint64_t top = (uint64_t(1) << 62) - 1;
    uint64_t kk = (top - 1) / ell;
    const uint64_t ell2 = static_cast<uint64_t>(ell) * ell;
    std::vector<PrimeResult> got;
    double bits = 0;
    const std::size_t block = 4096 * std::max(1u, threads);
    while (bits < need) {
        std::vector<uint64_t> cand;
        cand.reserve(block);
        for (std::size_t n = 0; n < block && kk > 0; ++n, --kk) {
            const uint64_t q = 1 + kk * ell;
            if (q % 2 == 0 || (q - 1) % ell2 == 0) continue;
            cand.push_back(q);
        }
        NF_CHECK_MSG(!cand.empty(), "ribet_poly_modular: ran out of candidates");
        std::vector<PrimeResult> res(cand.size());
        std::vector<char> isp(cand.size(), 0);
        parallel_ranges(cand.size(), threads, [&](std::size_t lo, std::size_t hi, unsigned) {
            for (std::size_t n = lo; n < hi; ++n) {
                if (!is_prime_mont(cand[n])) continue;
                isp[n] = 1;
                res[n] = process_prime(S, eta_plan, gam_plan, cand[n]);
            }
        });
        for (std::size_t n = 0; n < cand.size() && bits < need; ++n) {
            if (!isp[n]) continue;
            ++out.candidates_tested;
            if (!res[n].split) continue;
            bits += std::log2(static_cast<double>(res[n].q));
            if (!res[n].chain_ok) ++out.chain_failures;
            got.push_back(std::move(res[n]));
        }
    }
    out.primes = got.size();
    out.modulus_bits = bits;
    out.q_max = got.front().q;
    out.q_min = got.back().q;
    NF_CHECK(out.q_min > out.q_max / 2);
    // Garner: x = d_0 + q_0 (d_1 + q_1 (d_2 + ...)), then symmetric lift.
    const std::size_t n = got.size();
    std::vector<Mod> R;
    R.reserve(n);
    for (const auto& g : got) R.emplace_back(g.q);
    std::vector<uint64_t> Cinv(n, 0);              // (q_0 ... q_{i-1})^{-1} mod q_i, Montgomery
    std::vector<std::vector<uint64_t>> Qm(n);      // Qm[i][j] = q_j mod q_i, Montgomery (j < i)
    parallel_ranges(n, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t i = lo; i < hi; ++i) {
            uint64_t p = R[i].one;
            Qm[i].resize(i);
            for (std::size_t j = 0; j < i; ++j) {
                Qm[i][j] = R[i].M.to(got[j].q);
                p = R[i].mul(p, Qm[i][j]);
            }
            Cinv[i] = R[i].inv(p);
        }
    });
    BigInt Mtot(1);
    for (const auto& g : got) Mtot *= BigInt::from_u64(g.q);
    const BigInt half = bigint_shr(Mtot, 1);
    out.poly.assign(ell + 1, BigInt(0));
    parallel_ranges(ell + 1, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        std::vector<uint64_t> d(n);
        for (std::size_t c = lo; c < hi; ++c) {
            for (std::size_t i = 0; i < n; ++i) {
                // the partial mixed-radix value mod q_i by Horner, in normal form
                // (mul(normal, Montgomery) = normal); d_j < q_j < 2 q_i.
                const Mod& Ri = R[i];
                const uint64_t qi = got[i].q;
                uint64_t part = 0;
                for (std::size_t j = i; j-- > 0;) {
                    const uint64_t dj = d[j] >= qi ? d[j] - qi : d[j];
                    part = Ri.M.add(Ri.mul(part, Qm[i][j]), dj);
                }
                d[i] = Ri.mul(Ri.M.sub(got[i].res[c], part), Cinv[i]);
            }
            BigInt x(0);
            for (std::size_t i = n; i-- > 0;) x = x * BigInt::from_u64(got[i].q) + BigInt::from_u64(d[i]);
            if (x > half) x -= Mtot;
            out.poly[c] = x;
        }
    });
    return out;
}

unsigned ribet_predicted_roots(const RibetFieldSpec& S, uint32_t q) {
    NF_CHECK(q != S.ell && is_prime_u32(q));
    if (q % S.ell != 1) return 1;
    const uint32_t r = root_of_order(q, S.ell);
    const uint32_t v = eval_product_unit(S.f, r, q);
    return powmod32(v, (q - 1) / S.ell, q) == 1 ? S.ell : 0;
}

namespace {
using UPoly = std::vector<uint64_t>;  // mod q < 2^31, constant first, trimmed
void utrim(UPoly& a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
}
UPoly umulmod(const UPoly& a, const UPoly& b, const UPoly& f, uint64_t q) {  // f monic
    if (a.empty() || b.empty()) return {};
    UPoly c(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0) continue;
        for (std::size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + a[i] * b[j]) % q;
    }
    const std::size_t n = f.size() - 1;
    for (std::size_t k = c.size(); k-- > n;) {
        const uint64_t t = c[k];
        if (t == 0) continue;
        for (std::size_t j = 0; j <= n; ++j) c[k - n + j] = (c[k - n + j] + (q - t) * f[j]) % q;
    }
    if (c.size() > n) c.resize(n);
    utrim(c);
    return c;
}
UPoly ugcd(UPoly a, UPoly b, uint64_t q) {
    utrim(a);
    utrim(b);
    while (!b.empty()) {
        // a <- a mod b
        const uint64_t inv = powmod32(static_cast<uint32_t>(b.back()), q - 2, static_cast<uint32_t>(q));
        while (a.size() >= b.size()) {
            const uint64_t t = a.back() * inv % q;
            const std::size_t sh = a.size() - b.size();
            for (std::size_t j = 0; j < b.size(); ++j) a[sh + j] = (a[sh + j] + (q - t) * b[j]) % q;
            utrim(a);
            if (a.empty()) break;
        }
        std::swap(a, b);
    }
    return a;
}
}  // namespace

int poly_root_count_mod(const IntPoly& P, uint32_t q) {
    NF_CHECK(is_prime_u32(q) && P.size() >= 2 && P.back() == BigInt(1));
    const std::size_t n = P.size() - 1;
    UPoly f(n + 1);
    for (std::size_t k = 0; k <= n; ++k) f[k] = P[k].mod_u32(q);
    UPoly df(n, 0);
    for (std::size_t k = 1; k <= n; ++k) df[k - 1] = f[k] * (k % q) % q;
    utrim(df);
    if (df.empty()) return -1;
    if (ugcd(f, df, q).size() > 1) return -1;
    // X^q mod f
    UPoly res{1}, base{0, 1};
    if (n == 1) base = UPoly{(q - f[0]) % q};
    for (uint64_t e = q; e; e >>= 1) {
        if (e & 1u) res = umulmod(res, base, f, q);
        if (e > 1) base = umulmod(base, base, f, q);
    }
    res.resize(std::max<std::size_t>(res.size(), 2), 0);
    res[1] = (res[1] + q - 1) % q;
    utrim(res);
    const UPoly g = ugcd(f, res, q);
    return res.empty() ? static_cast<int>(n) : static_cast<int>(g.size()) - 1;
}

std::vector<std::vector<MpComplex>> ribet_trace_beta(const RibetFieldSpec& S, const RibetNumeric& N, uint32_t k,
                                                     unsigned threads) {
    const uint32_t ell = S.ell;
    const unsigned wp = N.prec + 48;
    std::vector<MpReal> bk(ell);  // b_s^k
    for (uint32_t s = 1; s < ell; ++s) {
        MpReal p = mp_from_i64(1);
        for (uint32_t t = 0; t < k; ++t) p = mp_mul(p, N.b[s], wp);
        bk[s] = p;
    }
    std::vector<std::vector<MpComplex>> out(ell - 1, std::vector<MpComplex>(ell));
    parallel_ranges(ell - 1, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t a = lo; a < hi; ++a)
            for (uint32_t j = 0; j < ell; ++j) {
                MpComplex acc;
                for (uint32_t s = 1; s < ell; ++s) {
                    const uint64_t e = (a * s + static_cast<uint64_t>(j) * k % ell * S.spow[s]) % ell;
                    acc = mp_cadd(acc, mp_cmul_real(N.zpow[e], bk[s], wp), wp);
                }
                out[a][j] = acc;
            }
    });
    return out;
}

std::vector<std::vector<MpComplex>> ribet_trace_x(const RibetFieldSpec& S, const RibetNumeric& N, uint32_t k,
                                                  unsigned threads, unsigned prec) {
    const uint32_t ell = S.ell;
    const unsigned wp = (prec == 0 ? N.prec : std::min(prec, N.prec)) + 48;
    const MpReal one = mp_from_i64(1);
    // x_{s,j}^k
    std::vector<std::vector<MpComplex>> xk(ell, std::vector<MpComplex>(ell));
    parallel_ranges(ell - 1, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t si = lo; si < hi; ++si) {
            const uint32_t s = static_cast<uint32_t>(si + 1);
            const MpReal bl = mp_div(mp_from_i64(S.n0), N.b[s], wp);  // n_0 / b_s
            const MpComplex den = mp_csub(MpComplex{one, MpReal{}}, N.zpow[s], wp);
            const MpReal dn = mp_cnorm(den, wp);
            const MpComplex dinv{mp_div(den.re, dn, wp), mp_neg(mp_div(den.im, dn, wp))};
            for (uint32_t j = 0; j < ell; ++j) {
                // n_0 (b_s z^{j s^i})^{-1} = (n_0 / b_s) z^{-j s^i}
                const uint32_t e = (ell - static_cast<uint32_t>(static_cast<uint64_t>(j) * S.spow[s] % ell)) % ell;
                MpComplex v = mp_cmul_real(N.zpow[e], bl, wp);
                v.re = mp_sub(v.re, one, wp);
                const MpComplex x = mp_cmul(v, dinv, wp);
                MpComplex p{one, MpReal{}};
                for (uint32_t t = 0; t < k; ++t) p = mp_cmul(p, x, wp);
                xk[s][j] = p;
            }
        }
    });
    std::vector<std::vector<MpComplex>> out(ell - 1, std::vector<MpComplex>(ell));
    parallel_ranges(ell - 1, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t a = lo; a < hi; ++a)
            for (uint32_t j = 0; j < ell; ++j) {
                MpComplex acc;
                for (uint32_t s = 1; s < ell; ++s) acc = mp_cadd(acc, mp_cmul(N.zpow[(a * s) % ell], xk[s][j], wp), wp);
                out[a][j] = acc;
            }
    });
    return out;
}

IntMat ribet_coords(const IntPoly& P, const std::vector<MpComplex>& theta, const std::vector<std::vector<MpComplex>>& elts,
                    const BigInt& D, unsigned prec, unsigned threads, double* max_log2_dist) {
    const std::size_t n = theta.size();
    NF_CHECK(P.size() == n + 1);
    const unsigned wp = prec + 48;
    const MpReal one = mp_from_i64(1);
    // q_j = P / (X - theta_j) and 1 / P'(theta_j).
    std::vector<std::vector<MpComplex>> quo(n);
    std::vector<MpComplex> dinv(n);
    parallel_ranges(n, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t j = lo; j < hi; ++j) {
            std::vector<MpComplex>& q = quo[j];
            q.assign(n, MpComplex{});
            q[n - 1] = MpComplex{one, MpReal{}};
            for (std::size_t k = n - 1; k-- > 0;)
                q[k] = mp_cadd(MpComplex{mp_from_int(P[k + 1]), MpReal{}}, mp_cmul(theta[j], q[k + 1], wp), wp);
            MpComplex v = q[n - 1];
            for (std::size_t k = n - 1; k-- > 0;) v = mp_cadd(mp_cmul(v, theta[j], wp), q[k], wp);
            const MpReal nv = mp_cnorm(v, wp);
            dinv[j] = MpComplex{mp_div(v.re, nv, wp), mp_neg(mp_div(v.im, nv, wp))};
        }
    });
    const MpReal dd = mp_from_int(D);
    IntMat out(elts.size(), std::vector<BigInt>(n));
    std::vector<double> wst(elts.size(), -std::numeric_limits<double>::infinity());
    parallel_ranges(elts.size(), threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t a = lo; a < hi; ++a) {
            std::vector<MpReal> acc(n);
            for (std::size_t j = 0; j < n; ++j) {
                const MpComplex cj = mp_cmul(elts[a][j], dinv[j], wp);
                for (std::size_t k = 0; k < n; ++k)
                    acc[k] = mp_add(acc[k], mp_sub(mp_mul(cj.re, quo[j][k].re, wp), mp_mul(cj.im, quo[j][k].im, wp), wp), wp);
            }
            for (std::size_t k = 0; k < n; ++k) {
                double d;
                out[a][k] = mp_round(mp_mul(acc[k], dd, wp), &d);
                wst[a] = std::max(wst[a], d);
            }
        }
    });
    double worst = -std::numeric_limits<double>::infinity();
    for (double v : wst) worst = std::max(worst, v);
    if (max_log2_dist) *max_log2_dist = worst;
    return out;
}

BigInt bigint_isqrt(const BigInt& n) {
    NF_CHECK(n.sign() >= 0);
    if (n.is_zero()) return BigInt(0);
    BigInt x = bigint_shl(BigInt(1), (n.bit_length() + 1) / 2 + 1);  // > sqrt(n)
    for (;;) {
        const BigInt y = bigint_shr(x + n / x, 1);
        if (y >= x) return x;
        x = y;
    }
}

namespace {
// g = gcd(a, b) = u a + v b (a, b >= 0 not both 0).
void xgcd(const BigInt& a, const BigInt& b, BigInt& g, BigInt& u, BigInt& v) {
    BigInt r0 = a, r1 = b, s0(1), s1(0), t0(0), t1(1);
    while (!r1.is_zero()) {
        BigInt q, r;
        BigInt::divmod(r0, r1, q, r);
        r0 = std::move(r1);
        r1 = std::move(r);
        BigInt s2 = s0 - q * s1, t2 = t0 - q * t1;
        s0 = std::move(s1);
        s1 = std::move(s2);
        t0 = std::move(t1);
        t1 = std::move(t2);
    }
    g = r0;
    u = s0;
    v = t0;
}
}  // namespace

namespace {
// Insert v (entries mod D, zero before column c0) into the echelon rows B (B[c] has its pivot in
// column c, a divisor of D), Euclid on the pivot column.
void hnf_insert(IntMat& B, std::vector<BigInt> v, const BigInt& D, std::size_t c0) {
    const std::size_t n = B.size();
    for (std::size_t c = c0; c < n; ++c) {
        if (v[c].is_zero()) continue;
        std::vector<BigInt>& p = B[c];
        NF_CHECK(!p[c].is_zero());
        if ((v[c] % p[c]).is_zero()) {
            const BigInt q = v[c] / p[c];
            for (std::size_t t = c; t < n; ++t) v[t] = (v[t] - q * p[t]).floor_mod(D);
            continue;
        }
        BigInt gg, u, w;
        xgcd(p[c], v[c], gg, u, w);
        const BigInt pc = p[c] / gg, vc = v[c] / gg;
        for (std::size_t t = c; t < n; ++t) {
            BigInt np = (u * p[t] + w * v[t]).floor_mod(D);
            v[t] = (pc * v[t] - vc * p[t]).floor_mod(D);
            p[t] = std::move(np);
        }
        p[c] = gg;
    }
}
}  // namespace

IntMat hnf_mod(const IntMat& gens, const BigInt& D, std::size_t n) {
    NF_CHECK(D.sign() > 0);
    // Rows mod D with pivots dividing D; D e_c is the initial row c (D Z^n is in the lattice).
    IntMat B(n, std::vector<BigInt>(n, BigInt(0)));
    for (std::size_t c = 0; c < n; ++c) B[c][c] = D;
    for (const auto& g0 : gens) {
        NF_CHECK(g0.size() == n);
        std::vector<BigInt> v(n);
        for (std::size_t c = 0; c < n; ++c) v[c] = g0[c].floor_mod(D);
        hnf_insert(B, std::move(v), D, 0);
    }
    // Howell completion (Storjohann): (D / p_c) * B[c] mod D vanishes at column c and lies in the
    // module; insert it into the rows below. Then the lifted rows contain D Z^n and form a basis.
    for (std::size_t c = 0; c < n; ++c) {
        const BigInt m = D / B[c][c];
        if (m == BigInt(1)) continue;
        std::vector<BigInt> v(n, BigInt(0));
        bool nz = false;
        for (std::size_t t = c + 1; t < n; ++t) {
            v[t] = (m * B[c][t]).floor_mod(D);
            nz = nz || !v[t].is_zero();
        }
        if (nz) hnf_insert(B, std::move(v), D, c + 1);
    }
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t r = 0; r < c; ++r) {
            const BigInt q = B[r][c] / B[c][c];
            if (q.is_zero()) continue;
            for (std::size_t t = c; t < n; ++t) B[r][t] -= q * B[c][t];
        }
    return B;
}

RibetLattice ribet_lattice(const IntPoly& P, const BigInt& D, const IntMat& num, unsigned threads) {
    const std::size_t n = P.size() - 1;
    NF_CHECK(num.size() == n);
    RibetLattice L;
    L.D = D;
    L.num = num;
    const std::vector<BigInt> ps = power_sums(P, static_cast<unsigned>(n));
    const BigInt D2 = D * D;
    L.trace.assign(n, std::vector<BigInt>(n));
    parallel_ranges(n, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t a = lo; a < hi; ++a)
            for (std::size_t b = a; b < n; ++b) {
                const IntPoly pr = poly_mulmod_monic(num[a], num[b], P);
                BigInt t(0);
                for (std::size_t k = 0; k < pr.size(); ++k)
                    if (!pr[k].is_zero()) t += pr[k] * ps[k];
                BigInt q, r;
                BigInt::divmod(t, D2, q, r);
                NF_CHECK_MSG(r.is_zero(), "ribet_lattice: Tr(w_a w_b) not integral (element not integral?)");
                L.trace[a][b] = q;
            }
    });
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b < a; ++b) L.trace[a][b] = L.trace[b][a];
    L.disc = det_bareiss(L.trace);
    return L;
}

IntMat ribet_t2_gram(const RibetLattice& L, const std::vector<MpComplex>& theta, unsigned shift, unsigned prec,
                     unsigned threads) {
    const std::size_t n = theta.size();
    const unsigned wp = prec + 48;
    // conjugates of w_a = num_a(theta) / D by Horner
    std::vector<std::vector<MpComplex>> w(n, std::vector<MpComplex>(n));
    const MpReal Dr = mp_from_int(L.D);
    parallel_ranges(n, threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t a = lo; a < hi; ++a)
            for (std::size_t j = 0; j < n; ++j) {
                MpComplex v;
                for (std::size_t k = n; k-- > 0;)
                    v = mp_cadd(mp_cmul(v, theta[j], wp), MpComplex{mp_from_int(L.num[a][k]), MpReal{}}, wp);
                w[a][j] = MpComplex{mp_div(v.re, Dr, wp), mp_div(v.im, Dr, wp)};
            }
    });
    IntMat G(n, std::vector<BigInt>(n));
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a; b < n; ++b) {
            MpReal t;
            for (std::size_t j = 0; j < n; ++j)
                t = mp_add(t, mp_add(mp_mul(w[a][j].re, w[b][j].re, wp), mp_mul(w[a][j].im, w[b][j].im, wp), wp), wp);
            G[a][b] = G[b][a] = mp_round(mp_ldexp(t, shift));
        }
    return G;
}


namespace {
// Minkowski coordinates (x_0, Re x_1, Im x_1, ..., Re x_h, Im x_h) of a conjugate vector
// with x_{ell-j} = conj(x_j); linear over Z, Euclidean norm^2 = (x_0^2 + T2(x)) / 2.
std::vector<MpReal> minkowski(const std::vector<MpComplex>& x) {
    const std::size_t n = x.size(), h = n / 2;
    std::vector<MpReal> m(n);
    m[0] = x[0].re;
    for (std::size_t j = 1; j <= h; ++j) {
        m[2 * j - 1] = x[j].re;
        m[2 * j] = x[j].im;
    }
    return m;
}

// Inverse of a real n x n matrix by Gauss--Jordan with partial pivoting.
std::vector<std::vector<MpReal>> mp_inverse(std::vector<std::vector<MpReal>> A, unsigned wp) {
    const std::size_t n = A.size();
    std::vector<std::vector<MpReal>> I(n, std::vector<MpReal>(n));
    for (std::size_t a = 0; a < n; ++a) I[a][a] = mp_from_i64(1);
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t p = c;
        for (std::size_t r = c + 1; r < n; ++r)
            if (mp_log2_abs(A[r][c]) > mp_log2_abs(A[p][c])) p = r;
        NF_CHECK_MSG(!mp_is_zero(A[p][c]), "mp_inverse: singular");
        std::swap(A[p], A[c]);
        std::swap(I[p], I[c]);
        const MpReal piv = A[c][c];
        for (std::size_t t = 0; t < n; ++t) {
            A[c][t] = mp_div(A[c][t], piv, wp);
            I[c][t] = mp_div(I[c][t], piv, wp);
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (r == c || mp_is_zero(A[r][c])) continue;
            const MpReal f = A[r][c];
            for (std::size_t t = 0; t < n; ++t) {
                A[r][t] = mp_sub(A[r][t], mp_mul(f, A[c][t], wp), wp);
                I[r][t] = mp_sub(I[r][t], mp_mul(f, I[c][t], wp), wp);
            }
        }
    }
    return I;
}

bool is_zero_row(const std::vector<BigInt>& r) {
    return std::all_of(r.begin(), r.end(), [](const BigInt& x) { return x.is_zero(); });
}
}  // namespace

RibetMaxOrder ribet_maximal_order(const RibetFieldSpec& S, const RibetNumeric& N, const IntPoly& P, const BigInt& D,
                                  unsigned threads) {
    const uint32_t ell = S.ell;
    const std::size_t n = ell;
    const unsigned wp = N.prec + 48;
    RibetMaxOrder out;
    double t0 = now_secs();
    // class representatives of (Z/ell)^x / <e>
    std::vector<char> seen(ell, 0);
    for (uint32_t k = 1; k < ell; ++k) {
        if (seen[k]) continue;
        out.kappas.push_back(k);
        uint32_t t = k;
        do {
            seen[t] = 1;
            t = mulmod32(t, S.e, ell);
        } while (t != k);
    }
    const std::size_t orbit = (ell - 1) / out.kappas.size();  // |<e>|, the expected rank per class
    // (1) M_beta
    IntMat brows;                            // theta-coordinates over D
    std::vector<std::vector<MpReal>> bmink;  // Minkowski coordinates
    brows.push_back(std::vector<BigInt>(n, BigInt(0)));
    brows[0][0] = D;
    bmink.push_back(minkowski(std::vector<MpComplex>(n, MpComplex{mp_from_i64(1), MpReal{}})));
    for (uint32_t kap : out.kappas) {
        const std::vector<std::vector<MpComplex>> G = ribet_trace_beta(S, N, kap, threads);
        double dist;
        const IntMat C = ribet_coords(P, N.theta, G, D, N.prec, threads, &dist);
        NF_CHECK_MSG(dist < -32, "ribet_maximal_order: theta-coordinates not integral");
        std::vector<std::vector<MpReal>> E;
        for (const auto& g : G) E.push_back(minkowski(g));
        double top = -1e300;
        for (const auto& r : E)
            for (const auto& x : r)
                if (!mp_is_zero(x)) top = std::max(top, mp_log2_abs(x));
        const std::size_t m = G.size();
        bool done = false;
        for (unsigned extra = 64; !done; extra *= 2) {
            NF_CHECK_MSG(extra <= 4096, "ribet_maximal_order: relations not separated");
            const int64_t t = static_cast<int64_t>(std::ceil(top)) - static_cast<int64_t>(extra);
            IntMat rows(m, std::vector<BigInt>(m + n, BigInt(0)));
            for (std::size_t a = 0; a < m; ++a) {
                rows[a][a] = BigInt(1);
                for (std::size_t k = 0; k < n; ++k) rows[a][m + k] = mp_round(mp_ldexp(E[a][k], -t));
            }
            const LllBasisResult lb = lll_basis(rows);
            const IntMat comb = int_mat_mul(lb.lll.H, C);
            std::vector<std::size_t> keep;
            for (std::size_t a = 0; a < m; ++a)
                if (!is_zero_row(comb[a])) keep.push_back(a);
            if (keep.size() != orbit) continue;  // relations and genuine vectors not yet separated
            // the kept rows (after the m - orbit exact relations in LLL order) span the image
            done = true;
            out.beta_shift = extra;
            out.ranks.push_back(keep.size());
            for (std::size_t a : keep) {
                brows.push_back(comb[a]);
                std::vector<MpReal> mk(n);
                for (std::size_t b = 0; b < m; ++b) {
                    if (lb.lll.H[a][b].is_zero()) continue;
                    const MpReal h = mp_from_int(lb.lll.H[a][b]);
                    for (std::size_t k = 0; k < n; ++k) mk[k] = mp_add(mk[k], mp_mul(h, E[b][k], wp), wp);
                }
                bmink.push_back(mk);
            }
        }
    }
    NF_CHECK_MSG(brows.size() == n, "ribet_maximal_order: M_beta does not have rank ell");
    out.beta = ribet_lattice(P, D, brows, threads);
    out.secs.emplace_back("beta", now_secs() - t0);
    t0 = now_secs();
    const unsigned vb = out.beta.disc.valuation(ell);
    NF_CHECK_MSG(vb >= ell - 2 && (vb - (ell - 2)) % 2 == 0, "ribet_maximal_order: unexpected v_ell(disc M_beta)");
    out.v = (vb - (ell - 2)) / 2;
    const BigInt Nn = BigInt::pow(BigInt(ell), out.v);
    // (3) x-generators in M_beta-coordinates, scaled by ell^v
    std::vector<std::vector<MpReal>> Einv = mp_inverse(bmink, wp);
    double topinv = -1e300;
    for (const auto& r : Einv)
        for (const auto& x : r)
            if (!mp_is_zero(x)) topinv = std::max(topinv, mp_log2_abs(x));
    const MpReal Nr = mp_from_int(Nn);
    // |x_{s,j}| <= (|n_0| / b_s + 1) / |1 - z^s|
    double log2_xmax = 0;
    for (uint32_t s2 = 1; s2 < ell; ++s2) {
        const double xs = (std::fabs(static_cast<double>(S.n0)) * std::exp(-S.log_b[s2]) + 1.0) /
                          (2.0 * std::sin(3.14159265358979323846 * static_cast<double>(s2) / ell));
        log2_xmax = std::max(log2_xmax, std::log2(xs) + 1e-6);
    }
    IntMat gens;
    double worst = -std::numeric_limits<double>::infinity();
    for (uint32_t k = 1; k < ell; ++k) {
        // working precision: the coordinates are about |x|^k |E^{-1}| ell^v (|x| from the
        // double-precision log b_s), 160 guard bits
        const unsigned px = std::min<unsigned>(
            wp, static_cast<unsigned>(std::max(0.0, k * log2_xmax + topinv)) + static_cast<unsigned>(Nn.bit_length()) + 160);
        const std::vector<std::vector<MpComplex>> G = ribet_trace_x(S, N, k, threads, px);
        std::vector<std::vector<BigInt>> rows(G.size());
        std::vector<double> wst(G.size(), -std::numeric_limits<double>::infinity());
        parallel_ranges(G.size(), threads, [&](std::size_t lo, std::size_t hi, unsigned) {
            for (std::size_t a = lo; a < hi; ++a) {
                const std::vector<MpReal> mk = minkowski(G[a]);
                rows[a].assign(n, BigInt(0));
                for (std::size_t c = 0; c < n; ++c) {
                    MpReal acc;
                    for (std::size_t k2 = 0; k2 < n; ++k2) acc = mp_add(acc, mp_mul(mk[k2], Einv[k2][c], px), px);
                    double d;
                    rows[a][c] = mp_round(mp_mul(acc, Nr, px), &d);
                    wst[a] = std::max(wst[a], d);
                }
            }
        });
        for (double w : wst) worst = std::max(worst, w);
        for (auto& r : rows) gens.push_back(std::move(r));
    }
    out.x_log2_dist = worst;
    out.secs.emplace_back("x_coords", now_secs() - t0);
    t0 = now_secs();
    const IntMat B = hnf_mod(gens, Nn, n);
    IntMat num = int_mat_mul(B, brows);
    for (auto& r : num)
        for (auto& x : r) {
            BigInt q, rem;
            BigInt::divmod(x, Nn, q, rem);
            NF_CHECK_MSG(rem.is_zero(), "ribet_maximal_order: an O_F row is not in (1/D) Z[theta]");
            x = q;
        }
    out.order = ribet_lattice(P, D, num, threads);
    out.secs.emplace_back("hnf_lattice", now_secs() - t0);
    return out;
}

namespace {
// Conjugates (num_a(theta_j) / D) of lattice rows given over the power basis of theta.
std::vector<std::vector<MpComplex>> lattice_conjugates(const IntMat& num, const BigInt& D, const std::vector<MpComplex>& theta,
                                                       unsigned wp, unsigned threads) {
    const std::size_t n = theta.size();
    std::vector<std::vector<MpComplex>> w(num.size(), std::vector<MpComplex>(n));
    const MpReal Dr = mp_from_int(D);
    parallel_ranges(num.size(), threads, [&](std::size_t lo, std::size_t hi, unsigned) {
        for (std::size_t a = lo; a < hi; ++a)
            for (std::size_t j = 0; j < n; ++j) {
                MpComplex v;
                for (std::size_t k = num[a].size(); k-- > 0;)
                    v = mp_cadd(mp_cmul(v, theta[j], wp), MpComplex{mp_from_int(num[a][k]), MpReal{}}, wp);
                w[a][j] = MpComplex{mp_div(v.re, Dr, wp), mp_div(v.im, Dr, wp)};
            }
    });
    return w;
}
}  // namespace

RibetPolred ribet_polred(const IntPoly& P, const std::vector<MpComplex>& theta, const RibetLattice& L, unsigned prec,
                         unsigned threads) {
    const std::size_t n = theta.size();
    const unsigned wp = prec + 48;
    RibetPolred R;
    double t0 = now_secs();
    // Iterated LLL with truncated embeddings (the identity-column trick): pass rows are
    // [I_n | round(E / 2^t)] with E the Minkowski rows (Re, Im of every conjugate) of the current
    // exact basis and t chosen so that the largest entry has kTopBits bits. The rows are always
    // independent, each pass is an exact LLL (T1) whose unimodular H is applied to the exact
    // basis, and every pass removes about kTopBits bits of the T2 spread of the HNF basis.
    constexpr int kTopBits = 64;
    auto emb_of = [&](const IntMat& nm) {
        std::vector<std::vector<MpComplex>> c = lattice_conjugates(nm, L.D, theta, wp, threads);
        std::vector<std::vector<MpReal>> E(n, std::vector<MpReal>(2 * n));
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t j = 0; j < n; ++j) {
                E[a][2 * j] = c[a][j].re;
                E[a][2 * j + 1] = c[a][j].im;
            }
        return E;
    };
    IntMat num = L.num;
    IntMat Htot(n, std::vector<BigInt>(n, BigInt(0)));
    for (std::size_t a = 0; a < n; ++a) Htot[a][a] = BigInt(1);
    for (R.passes = 0; R.passes < 200; ++R.passes) {
        const std::vector<std::vector<MpReal>> E = emb_of(num);
        double top = -1e300;
        for (const auto& r : E)
            for (const auto& x : r)
                if (!mp_is_zero(x)) top = std::max(top, mp_log2_abs(x));
        const int64_t t = static_cast<int64_t>(std::ceil(top)) - kTopBits;
        IntMat rows(n, std::vector<BigInt>(3 * n, BigInt(0)));
        for (std::size_t a = 0; a < n; ++a) {
            rows[a][a] = BigInt(1);
            for (std::size_t k = 0; k < 2 * n; ++k) rows[a][n + k] = mp_round(mp_ldexp(E[a][k], -t));
        }
        const LllBasisResult lb = lll_basis(rows);
        bool ident = true;
        for (std::size_t a = 0; a < n && ident; ++a)
            for (std::size_t b = 0; b < n && ident; ++b) ident = (lb.lll.H[a][b] == BigInt(a == b ? 1 : 0));
        if (!ident) {
            num = int_mat_mul(lb.lll.H, num);
            Htot = int_mat_mul(lb.lll.H, Htot);
        }
        if (ident) break;  // no progress at this scale (the scale only depends on the basis)
    }
    R.H = Htot;
    R.secs.emplace_back("lll", now_secs() - t0);
    t0 = now_secs();
    auto rows_of = [&](const IntMat& nm, unsigned shift) {
        const std::vector<std::vector<MpReal>> E = emb_of(nm);
        IntMat rows(n, std::vector<BigInt>(2 * n));
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t k = 0; k < 2 * n; ++k) rows[a][k] = mp_round(mp_ldexp(E[a][k], shift));
        return rows;
    };
    const IntMat G = gram_of_rows(rows_of(num, 8));  // the T2 form of POLRED's "size", exact PD
    // conjugates of the reduced basis (and theta itself, for the Q(theta) = Q(alpha) check)
    std::vector<std::vector<MpComplex>> w = lattice_conjugates(num, L.D, theta, wp, threads);
    std::size_t ra = n;
    for (std::size_t r = 0; r < n && ra == n; ++r) {
        double fd;
        IntPoly f = poly_from_conjugates(w[r], prec, threads, &fd);
        if (fd > -32 || poly_discriminant(f).is_zero()) continue;
        R.f = std::move(f);
        R.f_log2_dist = fd;
        ra = r;
    }
    NF_CHECK_MSG(ra < n, "ribet_polred: no reduced basis vector generates F");
    const BigInt df = poly_discriminant(R.f);
    NF_CHECK_MSG((df % L.disc).is_zero(), "ribet_polred: disc(L) does not divide disc(f)");
    const BigInt q2 = df / L.disc;
    R.d = bigint_isqrt(q2);
    NF_CHECK_MSG(R.d * R.d == q2, "ribet_polred: disc(f)/disc(L) is not a square");
    w.emplace_back();
    w[n] = theta;
    R.secs.emplace_back("alpha_charpoly", now_secs() - t0);
    t0 = now_secs();
    const std::vector<MpComplex> alpha = w[ra];
    double dist;
    IntMat hh = ribet_coords(R.f, alpha, w, R.d, prec, threads, &dist);
    R.h_log2_dist = dist;
    const std::vector<BigInt> htheta = hh[n];
    hh.pop_back();
    R.h = hh;
    const RibetLattice L2 = ribet_lattice(R.f, R.d, R.h, threads);
    R.trace_exact = (L2.trace == transform_gram(R.H, L.trace));
    R.theta_exact = (charpoly_element(R.f, htheta, R.d) == P);
    R.secs.emplace_back("coords_checks", now_secs() - t0);
    t0 = now_secs();
    R.pr = polred(R.f, R.h, std::vector<BigInt>(n, R.d), G);
    R.best = R.pr.best_poly();
    if (!R.best.empty()) R.best_disc = poly_discriminant(R.best);
    R.secs.emplace_back("polred", now_secs() - t0);
    return R;
}

}  // namespace nf
