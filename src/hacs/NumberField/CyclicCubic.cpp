// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CyclicCubic.cpp -- the cyclic cubic field of prime conductor ell, h_K = [E_K : C] by saturation, Lambda-bar
// (hecke-extensions slot 23, C3). See CyclicCubic.h for the mathematics and what is proved.
#include "CyclicCubic.h"

#include <cmath>
#include <limits>
#include <sstream>

#include "MpReal.h"
#include "NfCheck.h"

namespace nf {

namespace {

bool is_prime_u32(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t d = 2; static_cast<uint64_t>(d) * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

uint64_t powmod_u64(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    b %= m;
    while (e) {
        if (e & 1) r = r * b % m;  // m < 2^20: no overflow
        b = b * b % m;
        e >>= 1;
    }
    return r;
}

uint32_t least_primitive_root(uint32_t p) {
    std::vector<uint32_t> fac;
    uint32_t n = p - 1;
    for (uint32_t d = 2; d * d <= n; ++d)
        if (n % d == 0) {
            fac.push_back(d);
            while (n % d == 0) n /= d;
        }
    if (n > 1) fac.push_back(n);
    for (uint32_t g = 2; g < p; ++g) {
        bool ok = true;
        for (uint32_t f : fac)
            if (powmod_u64(g, (p - 1) / f, p) == 1) {
                ok = false;
                break;
            }
        if (ok) return g;
    }
    NF_CHECK_MSG(false, "no primitive root");
    return 0;
}

BigInt det3(const BigInt m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

// ---- numerics ------------------------------------------------------------------------------------------------

struct Numerics {
    unsigned prec = 0;
    MpReal eta[3];  // phi_0(eta_i); phi_j(eta_i) = eta[(i + j) % 3]
};

MpReal emb_eta(const Numerics& N, unsigned j, unsigned i) { return N.eta[(i + j) % 3]; }

// Reconstructs the element of O_K with real embeddings e[0..2] (phi_j), if it is one: the traces Tr(x eta_k) are
// integers for x in O_K; round them (rejecting a distance > 2^-20), solve G x = t exactly. log2 of the largest
// trace distance is returned in *maxdist.
bool reconstruct(const CyclicCubicField& K, const Numerics& N, const MpReal e[3], CubicElt& out, double* maxdist) {
    const unsigned P = N.prec;
    BigInt t[3];
    double md = -std::numeric_limits<double>::infinity();
    for (unsigned k = 0; k < 3; ++k) {
        MpReal s;
        for (unsigned j = 0; j < 3; ++j) s = mp_add(s, mp_mul(e[j], emb_eta(N, j, k), P), P);
        double d = 0;
        t[k] = mp_round(s, &d);
        if (d > md) md = d;
    }
    if (maxdist) *maxdist = md;
    if (md > -20.0) return false;
    BigInt g[3][3];
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) g[i][j] = BigInt(K.G[i][j]);
    const BigInt D = det3(g);
    // x = adj(G) t / D (G symmetric).
    for (unsigned i = 0; i < 3; ++i) {
        BigInt acc;
        for (unsigned k = 0; k < 3; ++k) {
            // cofactor C_{k i} of G
            BigInt m[2][2];
            unsigned rr = 0;
            for (unsigned r = 0; r < 3; ++r) {
                if (r == k) continue;
                unsigned cc = 0;
                for (unsigned c = 0; c < 3; ++c) {
                    if (c == i) continue;
                    m[rr][cc++] = g[r][c];
                }
                ++rr;
            }
            BigInt cof = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            if ((k + i) % 2) cof = -cof;
            acc += cof * t[k];
        }
        BigInt q, r;
        BigInt::divmod(acc, D, q, r);
        if (!r.is_zero()) return false;
        out[i] = q;
    }
    return true;
}

// The signs of phi_j(x) for an exact x whose log vector |phi_j(x)| = exp(L_j) is known; NF_CHECK that the
// computed sizes agree with L (so each sign is read off a value computed to full relative accuracy).
void signs_of(const Numerics& N, unsigned W, const CubicElt& x, const MpReal L[3], int out[3]) {
    for (unsigned j = 0; j < 3; ++j) {
        MpReal s;
        for (unsigned i = 0; i < 3; ++i) s = mp_add(s, mp_mul(mp_from_int(x[i]), emb_eta(N, j, i), W), W);
        NF_CHECK_MSG(!mp_is_zero(s) && std::fabs(mp_log2_abs(s) - mp_to_double(L[j]) / std::log(2.0)) < 1e-6,
                     "cyclic cubic: embedding size disagrees with its log vector");
        out[j] = mp_sign(s);
    }
}

double log_abs_bound(const MpReal L[3]) {
    double m = 0;
    for (unsigned j = 0; j < 3; ++j) m = std::max(m, std::fabs(mp_to_double(L[j])));
    return m;
}

}  // namespace

// ---- the field ---------------------------------------------------------------------------------------------------

CyclicCubicField cyclic_cubic_field(uint32_t ell) {
    NF_CHECK_MSG(ell >= 7 && ell < (1u << 20) && ell % 3 == 1 && is_prime_u32(ell), "cyclic_cubic_field: ell prime, 1 mod 3, 7 <= ell < 2^20");
    CyclicCubicField K;
    K.ell = ell;
    K.g = least_primitive_root(ell);
    K.coset.assign(ell, 0);
    uint64_t x = 1;
    for (uint32_t k = 0; k < ell - 1; ++k) {
        K.coset[x] = static_cast<uint8_t>(k % 3);
        x = x * K.g % ell;
    }
    // T[0][j][k] by counting, the rest by sigma-equivariance T[i][j][k] = T[0][j-i][k-i]. eta_0 eta_j =
    // sum_s n_j(s) zeta^s with n_j(s) = #{(a, b) in C_0 x C_j : a + b = s}; n_j is constant on each coset C_k
    // (Galois invariance), so the coefficient of eta_k is cnt[j][k] / |C_k|, and zeta^0 = 1 = -(eta_0 + eta_1 + eta_2).
    {
        int64_t cnt[3][3] = {}, zero[3] = {};
        for (uint32_t a = 1; a < ell; ++a) {
            if (K.coset[a] != 0) continue;
            for (uint32_t b = 1; b < ell; ++b) {
                const uint32_t s = (a + b) % ell;
                if (s == 0)
                    ++zero[K.coset[b]];
                else
                    ++cnt[K.coset[b]][K.coset[s]];
            }
        }
        const int64_t m = (ell - 1) / 3;
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned k = 0; k < 3; ++k) {
                NF_CHECK_MSG(cnt[j][k] % m == 0, "cyclotomic numbers not constant on cosets");
                K.T[0][j][k] = cnt[j][k] / m - zero[j];
            }
    }
    for (unsigned i = 1; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned k = 0; k < 3; ++k) K.T[i][j][k] = K.T[0][(j + 3 - i) % 3][(k + 3 - i) % 3];
    // e2, e3 exactly.
    CubicElt e[3];
    for (unsigned i = 0; i < 3; ++i) e[i] = {BigInt(0), BigInt(0), BigInt(0)}, e[i][i] = BigInt(1);
    CubicElt s2 = cubic_mul(K, e[0], e[1]);
    const CubicElt p12 = cubic_mul(K, e[1], e[2]), p20 = cubic_mul(K, e[2], e[0]);
    for (unsigned k = 0; k < 3; ++k) s2[k] += p12[k] + p20[k];
    const CubicElt s3 = cubic_mul(K, cubic_mul(K, e[0], e[1]), e[2]);
    BigInt v2, v3;
    NF_CHECK_MSG(cubic_is_rational(s2, &v2) && cubic_is_rational(s3, &v3), "periods: e2, e3 not rational");
    K.e2 = v2.to_int64();
    K.e3 = v3.to_int64();
    K.poly[3] = 1;
    K.poly[2] = 1;  // -e1, e1 = -1
    K.poly[1] = K.e2;
    K.poly[0] = -K.e3;
    // 4 ell = a^2 + 27 b^2, a = 1 mod 3.
    for (int64_t b = 1; 27 * b * b < 4 * static_cast<int64_t>(ell); ++b) {
        const int64_t r = 4 * static_cast<int64_t>(ell) - 27 * b * b;
        int64_t s = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(r))));
        while (s * s > r) --s;
        while ((s + 1) * (s + 1) <= r) ++s;
        if (s * s == r) {
            K.b = b;
            K.a = ((s % 3) + 3) % 3 == 1 ? s : -s;
            break;
        }
    }
    NF_CHECK_MSG(K.b > 0, "4 ell = a^2 + 27 b^2 has no solution");
    const BigInt A(K.poly[2]), B(K.poly[1]), C(K.poly[0]);
    K.poly_disc = A * A * B * B - BigInt(4) * B * B * B - BigInt(4) * A * A * A * C - BigInt(27) * C * C + BigInt(18) * A * B * C;
    BigInt g[3][3];
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            int64_t tr = 0;
            for (unsigned k = 0; k < 3; ++k) tr -= K.T[i][j][k];  // Tr(eta_k) = -1
            K.G[i][j] = tr;
            g[i][j] = BigInt(tr);
        }
    K.basis_disc = det3(g);
    return K;
}

CubicElt cubic_one() { return {BigInt(-1), BigInt(-1), BigInt(-1)}; }

CubicElt cubic_mul(const CyclicCubicField& K, const CubicElt& x, const CubicElt& y) {
    CubicElt z = {BigInt(0), BigInt(0), BigInt(0)};
    for (unsigned i = 0; i < 3; ++i) {
        if (x[i].is_zero()) continue;
        for (unsigned j = 0; j < 3; ++j) {
            if (y[j].is_zero()) continue;
            const BigInt p = x[i] * y[j];
            for (unsigned k = 0; k < 3; ++k)
                if (K.T[i][j][k]) z[k] += p * BigInt(K.T[i][j][k]);
        }
    }
    return z;
}

CubicElt cubic_sigma(const CubicElt& x) { return {x[2], x[0], x[1]}; }

CubicElt cubic_pow(const CyclicCubicField& K, CubicElt x, uint64_t e) {
    CubicElt r = cubic_one();
    while (e) {
        if (e & 1) r = cubic_mul(K, r, x);
        e >>= 1;
        if (e) x = cubic_mul(K, x, x);
    }
    return r;
}

bool cubic_is_rational(const CubicElt& x, BigInt* value) {
    if (x[0] != x[1] || x[1] != x[2]) return false;
    if (value) *value = -x[0];
    return true;
}

BigInt cubic_norm(const CyclicCubicField& K, const CubicElt& x) {
    const CubicElt s1 = cubic_sigma(x), s2 = cubic_sigma(s1);
    BigInt v;
    NF_CHECK_MSG(cubic_is_rational(cubic_mul(K, cubic_mul(K, x, s1), s2), &v), "cubic_norm: not rational");
    return v;
}

// ---- the analytic class number formula ----------------------------------------------------------------------

namespace {
long double digamma(long double x) {
    long double acc = 0;
    while (x < 16.0L) {
        acc -= 1.0L / x;
        x += 1.0L;
    }
    const long double r = 1.0L / (x * x);
    // psi(x) ~ ln x - 1/(2x) - sum_k B_{2k} / (2k x^{2k})
    const long double tail =
        r * (1.0L / 12 - r * (1.0L / 120 - r * (1.0L / 252 - r * (1.0L / 240 - r * (1.0L / 132 - r * (691.0L / 32760 - r * (1.0L / 12)))))));
    return acc + std::log(x) - 0.5L / x - tail;
}
}  // namespace

double cubic_hR_analytic(const CyclicCubicField& K) {
    const long double c[3] = {1.0L, -0.5L, -0.5L};
    const long double s3 = std::sqrt(3.0L) / 2;
    const long double s[3] = {0.0L, s3, -s3};
    long double re = 0, im = 0;
    for (uint32_t a = 1; a < K.ell; ++a) {
        const long double ps = digamma(static_cast<long double>(a) / K.ell);
        re += c[K.coset[a]] * ps;
        im += s[K.coset[a]] * ps;
    }
    re /= -static_cast<long double>(K.ell);
    im /= -static_cast<long double>(K.ell);
    return static_cast<double>(K.ell * (re * re + im * im) / 4);
}

// ---- the class number ---------------------------------------------------------------------------------------

CubicClassResult cyclic_cubic_class_number(uint32_t ell, unsigned extra_prec, unsigned start_power) {
    NF_CHECK(start_power >= 1 && start_power <= 4);
    CubicClassResult R;
    R.ell = ell;
    const CyclicCubicField K = cyclic_cubic_field(ell);
    R.a = K.a;
    R.b = K.b;
    {
        const BigInt L(ell);
        R.field_ok = K.poly[1] == -static_cast<int64_t>((ell - 1) / 3) && K.e3 * 27 == static_cast<int64_t>(ell) * (K.a + 3) - 1 &&
                     K.poly_disc == L * L * BigInt(K.b) * BigInt(K.b) && K.basis_disc == L * L;
    }
    // Double-precision estimate of the coset log sums Lc_j = sum_{a in C_j} log|1 - zeta^a| to size the precision.
    double Lcd[3] = {0, 0, 0};
    const double PI = 3.14159265358979323846;
    for (uint32_t a = 1; a < ell; ++a) Lcd[K.coset[a]] += std::log(2.0 * std::fabs(std::sin(PI * a / ell)));
    double lu = 0;
    for (unsigned j = 0; j < 3; ++j) lu = std::max(lu, std::fabs(Lcd[(j + 1) % 3] - Lcd[j]));
    double lx = std::max(std::fabs(Lcd[0]), std::max(std::fabs(Lcd[1]), std::fabs(Lcd[2])));
    // The unit v has |log phi_j(v)| <= lu / 2; a root candidate has |log phi_j| <= M := |L(eps_0)|_inf + |L(eps_1)|_inf
    // (lu at the start). Its coordinates are of size e^M, its smallest embedding >= e^{-2M}: reading that embedding's
    // sign off the coordinates cancels 3M / ln 2 bits. try_root NF_CHECKs the actual M against the precision.
    lu *= start_power;
    const double bits = std::max(3.0 * lu, lx) / std::log(2.0) + std::log2(static_cast<double>(ell)) + 8.0;
    const unsigned P = static_cast<unsigned>(std::max(160.0, bits + 96.0)) + extra_prec;
    R.prec = P;
    const unsigned W = P + 64;  // internal working precision for the period sums
    MpContext ctx(W);

    Numerics N;
    N.prec = P;
    // zeta^a by repeated multiplication; eta_i and the products prod_j = prod_{a in C_j} |1 - zeta^a|^2.
    const MpReal two_pi_over = mp_div_u32(mp_ldexp(ctx.pi(), 1), ell, W);
    const MpComplex w = ctx.expi(two_pi_over);
    MpComplex z = w;
    MpReal eta[3], prodv[3];
    for (unsigned j = 0; j < 3; ++j) prodv[j] = mp_from_i64(1);
    for (uint32_t a = 1; a < ell; ++a) {
        const unsigned j = K.coset[a];
        eta[j] = mp_add(eta[j], z.re, W);
        prodv[j] = mp_mul(prodv[j], mp_sub(mp_from_i64(2), mp_ldexp(z.re, 1), W), W);
        z = mp_cmul(z, w, W);
    }
    for (unsigned j = 0; j < 3; ++j) N.eta[j] = eta[j];  // kept at W bits (signs_of evaluates at W)
    MpReal Lc[3];
    for (unsigned j = 0; j < 3; ++j) Lc[j] = mp_ldexp(ctx.log(prodv[j]), -1);

    // xi = prod_{a in H} (1 - zeta^a): phi_j(xi) = exp(Lc_j) > 0; exact reconstruction and N(xi) = ell.
    MpReal e[3];
    for (unsigned j = 0; j < 3; ++j) e[j] = ctx.exp(Lc[j]);
    CubicElt xi;
    std::string stage = "xi";
    bool ok = reconstruct(K, N, e, xi, nullptr) && cubic_norm(K, xi) == BigInt(ell);
    // u = sigma(xi) / xi: phi_j(u) = exp(Lc_{j+1} - Lc_j).
    MpReal Lu[3];
    for (unsigned j = 0; j < 3; ++j) Lu[j] = mp_sub(Lc[(j + 1) % 3], Lc[j], W);
    for (unsigned j = 0; j < 3; ++j) e[j] = ctx.exp(Lu[j]);
    CubicElt u;
    if (ok) stage = "u";
    ok = ok && reconstruct(K, N, e, u, nullptr);
    ok = ok && cubic_mul(K, u, xi) == cubic_sigma(xi) && cubic_norm(K, u) == BigInt(1);
    ok = ok && cubic_mul(K, cubic_mul(K, u, cubic_sigma(u)), cubic_sigma(cubic_sigma(u))) == cubic_one();
    // The cyclotomic unit group of K is generated by -1 and the conjugates of v = N_{Q(zeta)^+/K}(xi_g),
    // xi_g = zeta^{(1-g)/2} (1 - zeta^g)/(1 - zeta) the real cyclotomic unit: |phi_j(v)| = exp((Lc_{j+1} - Lc_j)/2),
    // so v^2 = u (u is totally positive). The sign pattern of v is unique up to -1: try the four with phi_0(v) > 0.
    MpReal Lv[3];
    for (unsigned j = 0; j < 3; ++j) Lv[j] = mp_ldexp(Lu[j], -1);
    CubicElt v;
    unsigned v_found = 0;  // square roots of u found, counted up to sign (a sign flip on an embedding of size below
                           // 2^-20 does not move the rounded traces, so two patterns may give one element or its negative)
    for (int s1 : {1, -1})
        for (int s2 : {1, -1}) {
            const int ss[3] = {1, s1, s2};
            for (unsigned j = 0; j < 3; ++j) {
                e[j] = ctx.exp(Lv[j]);
                if (ss[j] < 0) e[j] = mp_neg(e[j]);
            }
            CubicElt t;
            if (reconstruct(K, N, e, t, nullptr) && cubic_mul(K, t, t) == u && (v_found == 0 || (t != v && t != CubicElt{-v[0], -v[1], -v[2]}))) {
                ++v_found;
                v = t;
            }
        }
    if (ok) stage = "v (found " + std::to_string(v_found) + ")";
    ok = ok && v_found == 1 && cubic_norm(K, v).abs() == BigInt(1);
    R.units_ok = ok;
    NF_CHECK_MSG(ok, "cyclotomic units of the cubic field of conductor " + std::to_string(ell) + " not reconstructed at " + stage);

    // The saturation state: basis eps[0], eps[1] of E/{+-1} relative to C, their log vectors and sign vectors (the
    // signs are read off the exact element, with its embedding sizes checked against the log vector).
    // start_power k > 1 (a control) starts from <-1, v^k, sigma(v)^k>, of index k^2 h in E.
    const CubicElt vk = cubic_pow(K, v, start_power);
    CubicElt eps[2] = {vk, cubic_sigma(vk)};
    MpReal Lg[2][3];
    int sg[2][3];
    for (unsigned j = 0; j < 3; ++j) {
        Lg[0][j] = mp_mul_i64(Lv[j], start_power, W);
        Lg[1][j] = mp_mul_i64(Lv[(j + 1) % 3], start_power, W);
    }
    signs_of(N, W, eps[0], Lg[0], sg[0]);
    signs_of(N, W, eps[1], Lg[1], sg[1]);
    auto reg = [&]() {
        const double a00 = mp_to_double(Lg[0][0]), a01 = mp_to_double(Lg[0][1]), a10 = mp_to_double(Lg[1][0]), a11 = mp_to_double(Lg[1][1]);
        return std::fabs(a00 * a11 - a01 * a10);
    };
    R.RC = reg();
    const double lg = std::log(static_cast<double>(ell) * ell / 4.0);
    R.R_low = lg * lg / 16.0;  // Cusick 1984: R >= (1/16) log^2(d/4), totally real cubic fields
    R.min_reject_log2 = 0;  // log2 distance of the closest trace-rejection (0 = none closer than 1)
    uint64_t h = 1;

    auto try_root = [&](uint32_t q, unsigned a, unsigned b) -> bool {
        MpReal Lr[3];
        int sc[3];
        for (unsigned j = 0; j < 3; ++j) {
            MpReal s = mp_add(mp_mul_i64(Lg[0][j], a, W), mp_mul_i64(Lg[1][j], b, W), W);
            Lr[j] = mp_div_u32(s, q, W);
            sc[j] = (a % 2 ? sg[0][j] : 1) * (b % 2 ? sg[1][j] : 1);
        }
        // reconstruct() computes the traces (magnitude <= ell e^M) at P bits: keep 80 bits below them. signs_of() reads
        // the smallest embedding (>= e^{-2M}) off coordinates of size e^M at W bits: keep 3M / ln 2 + 16 bits.
        const double Mb = log_abs_bound(Lr) / std::log(2.0);
        NF_CHECK_MSG(Mb + std::log2(static_cast<double>(ell)) + 80.0 <= P && 3.0 * Mb + std::log2(static_cast<double>(ell)) + 16.0 <= W,
                     "cyclic_cubic_class_number: precision too low for a root candidate");
        int target_sign = 1;  // r^q = target_sign * eps0^a eps1^b
        std::vector<std::array<int, 3>> cands;
        if (q == 2) {
            if (!(sc[0] == sc[1] && sc[1] == sc[2])) return false;  // neither +c nor -c totally positive
            target_sign = sc[0];
            for (int s1 : {1, -1})
                for (int s2 : {1, -1}) cands.push_back({1, s1, s2});
        } else {
            cands.push_back({sc[0], sc[1], sc[2]});
        }
        for (const auto& cs : cands) {
            ++R.root_tests;
            MpReal ee[3];
            for (unsigned j = 0; j < 3; ++j) {
                ee[j] = ctx.exp(Lr[j]);
                if (cs[j] < 0) ee[j] = mp_neg(ee[j]);
            }
            CubicElt r;
            double md = 0;
            if (!reconstruct(K, N, ee, r, &md)) {
                if (md > -20.0 && md < R.min_reject_log2) R.min_reject_log2 = md;
                continue;
            }
            CubicElt c = cubic_mul(K, cubic_pow(K, eps[0], a), cubic_pow(K, eps[1], b));
            if (target_sign < 0)
                for (auto& v : c) v = -v;
            if (cubic_pow(K, r, q) != c) continue;
            NF_CHECK(cubic_norm(K, r).abs() == BigInt(1));
            const unsigned slot = a % q ? 0 : 1;
            eps[slot] = r;
            for (unsigned j = 0; j < 3; ++j) {
                Lg[slot][j] = Lr[j];
            }
            signs_of(N, W, eps[slot], Lg[slot], sg[slot]);
            if (R.root_certificate.empty()) {
                std::ostringstream os;
                os << "q=" << q << " class=(" << a << "," << b << ") r=(" << r[0] << "," << r[1] << "," << r[2] << ")";
                R.root_certificate = os.str();
            }
            return true;
        }
        return false;
    };

    for (uint32_t q = 2;; ++q) {
        if (!is_prime_u32(q)) continue;
        if (static_cast<double>(q) > reg() / R.R_low * (1.0 + 1e-9)) break;
        R.sat_bound = q;
        bool found = true;
        while (found) {
            found = false;
            if (q == 2) {
                for (auto ab : {std::pair<unsigned, unsigned>{1, 0}, {0, 1}, {1, 1}})
                    if (try_root(2, ab.first, ab.second)) {
                        found = true;
                        break;
                    }
            } else {
                for (unsigned b = 0; b < q && !found; ++b)
                    if (try_root(q, 1, b)) found = true;
                if (!found && try_root(q, 0, 1)) found = true;
            }
            if (found) {
                h *= q;
                R.roots.push_back(q);
            }
        }
    }
    R.h = h;
    R.R = reg();
    while (h % 2 == 0) {
        ++R.v2;
        h /= 2;
    }
    R.rank2 = R.v2 == 0 ? 0 : (R.v2 == 2 ? 2 : -1);
    R.in_lambda_bar = R.v2 >= 1;
    R.hR_analytic = cubic_hR_analytic(K);
    R.h_analytic = R.hR_analytic / R.R;
    R.rel_err = std::fabs(R.hR_analytic - R.RC) / R.RC;
    return R;
}

}  // namespace nf
