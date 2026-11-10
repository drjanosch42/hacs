// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P2) the Eisenstein measure exactly -- see EisensteinMeasure.h and docs/implementation/slot21/P2_spec.md.
#include "EisensteinMeasure.h"

#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "ModularSymbolMeasure.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "../../Util/ParallelFor.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>

namespace hecke_ext {
namespace lvalue {
namespace eis {
namespace {

using u128 = unsigned __int128;

using engine::scalar::pow_wrap;
inline uint64_t PowU(uint64_t b, uint32_t e) { return pow_wrap(b, e); }
uint32_t Val(uint64_t x, uint64_t ell) {
    uint32_t v = 0;
    while (x != 0 && x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}
// ell-adic inverse of a unit mod NK
uint64_t InvLadic(uint64_t a, uint64_t NK) { return InvModU64(a % NK, NK); }

// E_{k-1}(y + M Z_ell) mod p for y < M (exact rational reduced mod p).
std::vector<uint64_t> BernDistModP(uint32_t r, uint64_t M, uint64_t p, const std::vector<uint64_t>& B) {
    std::vector<uint64_t> coef(r + 1, 0);   // coefficient of y^{r-s}: C(r,s) B_s M^{s-1} / r
    const uint64_t iM = InvModP(M % p, p), ir = InvModP(r % p, p);
    uint64_t C = 1;
    for (uint32_t s = 0; s <= r; ++s) {
        if (s > 0) C = MulModP(MulModP(C, (r - s + 1) % p, p), InvModP(s, p), p);
        if (B[s] == 0) continue;
        const uint64_t Mp = s == 0 ? iM : PowModP(M % p, s - 1, p);
        coef[s] = MulModP(MulModP(MulModP(C, B[s], p), Mp, p), ir, p);
    }
    std::vector<uint64_t> out(M);
    for (uint64_t y = 0; y < M; ++y) {
        // Horner in y: sum_s coef[s] y^{r-s}
        uint64_t acc = 0;
        const uint64_t yy = y % p;
        for (uint32_t s = 0; s <= r; ++s) acc = (MulModP(acc, yy, p) + coef[s]) % p;
        out[y] = acc;
    }
    return out;
}

} // namespace

uint64_t MulModP(uint64_t a, uint64_t b, uint64_t p) { return engine::scalar::mulmod(a, b, p); }
uint64_t PowModP(uint64_t b, uint64_t e, uint64_t p) { return engine::scalar::powmod<uint64_t>(b, e, p); }
uint64_t InvModP(uint64_t a, uint64_t p) {
    ENGINE_CHECK_MSG(a % p != 0, "eis::InvModP: zero");
    return engine::scalar::invmod_fermat<uint64_t>(a, p);
}
uint64_t FromIntModP(int64_t x, uint64_t p) { return engine::scalar::modi(x, p); }

std::vector<uint64_t> BernoulliModP(uint32_t N, uint64_t p) {
    ENGINE_CHECK_MSG(p > N + 1, "eis::BernoulliModP: p too small");
    std::vector<uint64_t> B(N + 1, 0);
    B[0] = 1;
    // B_m = -1/(m+1) sum_{j<m} C(m+1, j) B_j
    std::vector<uint64_t> row{1, 1};   // C(m, j): starts at C(1, .)
    for (uint32_t m = 1; m <= N; ++m) {
        std::vector<uint64_t> nr(m + 2, 1);
        for (uint32_t j = 1; j <= m; ++j) nr[j] = (row[j - 1] + (j < row.size() ? row[j] : 0)) % p;
        row = nr;   // row = C(m+1, .)
        uint64_t s = 0;
        for (uint32_t j = 0; j < m; ++j) s = (s + MulModP(row[j], B[j], p)) % p;
        B[m] = MulModP(p - s % p, InvModP(m + 1, p), p) % p;
    }
    return B;
}

std::vector<uint64_t> ZagierLaurentModP(uint32_t n, uint64_t p, const std::vector<uint64_t>& B) {
    ENGINE_CHECK_MSG(B.size() >= n + 3, "eis::ZagierLaurentModP: Bernoulli table too short");
    std::vector<uint64_t> fact(n + 3, 1);
    for (uint32_t i = 1; i <= n + 2; ++i) fact[i] = MulModP(fact[i - 1], i, p);
    std::vector<uint64_t> Pt(n + 3, 0);
    for (int64_t j = -1; j <= static_cast<int64_t>(n) + 1; j += 2) {
        const uint32_t a = static_cast<uint32_t>(j + 1), b = static_cast<uint32_t>(static_cast<int64_t>(n) - j + 1);
        const uint64_t bj = MulModP(B[a], InvModP(fact[a], p), p), bn = MulModP(B[b], InvModP(fact[b], p), p);
        const int64_t t = static_cast<int64_t>(n) - j;
        Pt[static_cast<size_t>(t + 1)] = MulModP(MulModP(fact[n], bj, p), bn, p);
    }
    return Pt;
}

uint64_t EvalLaurentModP(const std::vector<uint64_t>& Pt, uint32_t n, int64_t c, int64_t d, uint64_t p, bool interior_only,
                         uint32_t* singular) {
    const uint64_t cc = FromIntModP(c, p), dd = FromIntModP(d, p);
    // interior: Horner sum_t P_t c^{n-t} d^t
    uint64_t acc = 0, cp = 1;
    for (int64_t t = n; t >= 0; --t) {
        acc = (MulModP(acc, dd, p) + MulModP(Pt[static_cast<size_t>(t + 1)], cp, p)) % p;
        cp = MulModP(cp, cc, p);
    }
    if (interior_only) return acc;
    // u_{-1} = x^{n+1} / y, u_{n+1} = y^{n+1} / x
    if (Pt[0] != 0) {
        if (d == 0) {
            if (singular) ++*singular;
        } else {
            acc = (acc + MulModP(MulModP(Pt[0], PowModP(cc, n + 1, p), p), InvModP(dd, p), p)) % p;
        }
    }
    if (Pt[n + 2] != 0) {
        if (c == 0) {
            if (singular) ++*singular;
        } else {
            acc = (acc + MulModP(MulModP(Pt[n + 2], PowModP(dd, n + 1, p), p), InvModP(cc, p), p)) % p;
        }
    }
    return acc;
}

uint64_t CuspSymbolModP(const std::vector<uint64_t>& Pt, uint32_t n, uint64_t a, uint64_t M, uint64_t p, bool interior_only) {
    uint64_t s = 0;
    for (const birch::Mat2& g : birch::ManinPath(static_cast<int64_t>(a % M), static_cast<int64_t>(M)))
        s = (s + EvalLaurentModP(Pt, n, g.c, g.d, p, interior_only)) % p;
    return (p - s) % p;
}

uint64_t EvalLaurentAt(const std::vector<uint64_t>& Pt, uint32_t n, int64_t a, int64_t b, int64_t c, int64_t d, uint64_t x, uint64_t y,
                       uint64_t p, bool interior_only) {
    const uint64_t X = (MulModP(FromIntModP(a, p), x, p) + MulModP(FromIntModP(c, p), y, p)) % p;
    const uint64_t Y = (MulModP(FromIntModP(b, p), x, p) + MulModP(FromIntModP(d, p), y, p)) % p;
    uint64_t acc = 0;
    for (uint32_t t = 0; t <= n; ++t)
        acc = (acc + MulModP(Pt[t + 1], MulModP(PowModP(X, n - t, p), PowModP(Y, t, p), p), p)) % p;
    if (!interior_only) {
        acc = (acc + MulModP(MulModP(Pt[0], PowModP(X, n + 1, p), p), InvModP(Y, p), p)) % p;
        acc = (acc + MulModP(MulModP(Pt[n + 2], PowModP(Y, n + 1, p), p), InvModP(X, p), p)) % p;
    }
    return acc;
}

uint64_t EisModP::mu(uint64_t a, uint32_t m) const {
    ENGINE_CHECK_MSG(m >= 1 && m <= M, "EisModP::mu: 1 <= m <= M");
    const uint64_t Mm1 = PowU(ell, m - 1);
    const uint64_t lk2 = PowModP(ell, k - 2, p);
    return (MulModP(lk2, D[m - 1][a % Mm1], p) + p - D[m][a % PowU(ell, m)]) % p;
}

EisModP BuildEisModP(uint32_t n, uint64_t ell, uint32_t M, uint64_t p, unsigned threads) {
    EisModP out;
    out.p = p;
    out.ell = ell;
    out.n = n;
    out.k = n + 2;
    out.M = M;
    const std::vector<uint64_t> B = BernoulliModP(n + 3, p);
    out.Bk_over_k = MulModP(B[n + 2], InvModP(n + 2, p), p);
    out.D.resize(M + 1);
    const uint64_t h = InvModP(2, p);
    for (uint32_t m = 0; m <= M; ++m) {
        const uint64_t Mm = PowU(ell, m);
        const std::vector<uint64_t> ek = BernDistModP(n + 1, Mm, p, B);
        const uint64_t iM = InvModP(Mm % p, p);
        std::vector<uint64_t> b1(Mm);
        for (uint64_t x = 0; x < Mm; ++x) b1[x] = (MulModP(x, iM, p) + p - h) % p;   // Bbar_1(x / M) = x/M - 1/2
        std::vector<uint64_t>& Dm = out.D[m];
        Dm.assign(Mm, 0);
        oc_util::ParallelChunks(Mm, threads, oc_util::kEisensteinChunks, [&](size_t lo, size_t hi) {
            for (size_t a = lo; a < hi; ++a) {
                u128 s = 0;
                uint64_t ay = 0;
                for (uint64_t y = 0; y < Mm; ++y) {
                    s += static_cast<u128>(b1[ay]) * ek[y];
                    if ((y & 7) == 7) s %= p;
                    ay += a;
                    if (ay >= Mm) ay -= Mm;
                }
                Dm[a] = static_cast<uint64_t>(s % p);
            }
        });
    }
    return out;
}

std::vector<uint64_t> ConvolutionModP(uint32_t n, uint64_t ell, uint32_t M, uint64_t p, unsigned threads) {
    const uint64_t Mm = PowU(ell, M);
    const std::vector<uint64_t> B = BernoulliModP(n + 3, p);
    const std::vector<uint64_t> ek = BernDistModP(n + 1, Mm, p, B);
    const uint64_t iM = InvModP(Mm % p, p), h = InvModP(2, p);
    std::vector<uint64_t> units;
    for (uint64_t x = 1; x < Mm; ++x)
        if (x % ell) units.push_back(x);
    std::vector<uint64_t> out(Mm, 0);
    oc_util::ParallelChunks(units.size(), threads, oc_util::kEisensteinChunks, [&](size_t lo, size_t hi) {
        for (size_t ia = lo; ia < hi; ++ia) {
            const uint64_t a = units[ia];
            const uint64_t ainv = InvModU64(a, Mm);
            u128 s = 0;
            uint32_t cnt = 0;
            for (uint64_t x : units) {
                const uint64_t e1 = (MulModP(x, iM, p) + p - h) % p;
                s += static_cast<u128>(e1) * ek[static_cast<uint64_t>(static_cast<u128>(x) * ainv % Mm)];
                if (++cnt == 8) {
                    s %= p;
                    cnt = 0;
                }
            }
            out[a] = (p - static_cast<uint64_t>(s % p)) % p;
        }
    });
    return out;
}

// ---------------------------------------------------------------- ell-adic
uint64_t EisLadic::mul(uint64_t a, uint64_t b) const { return static_cast<uint64_t>(static_cast<u128>(a) * b % NK); }

std::vector<uint64_t> EisLadic::branch(uint32_t br, const std::vector<uint64_t>* v) const {
    const std::vector<uint64_t>& tab = v ? *v : mu;
    const uint64_t MM = PowU(ell, M), G = PowU(ell, M - 1), u = 1 + ell;
    std::vector<uint64_t> deltas, wts;
    for (uint64_t a = 1; a < ell; ++a) {
        deltas.push_back(TeichmullerPowMod(a, 1, ell, M) % MM);
        wts.push_back(TeichmullerPowMod(a, static_cast<uint32_t>(br % (ell - 1)), ell, K) % NK);
    }
    std::vector<uint64_t> g(G, 0);
    uint64_t ui = 1;
    for (uint64_t i = 0; i < G; ++i) {
        u128 s = 0;
        for (size_t t = 0; t < deltas.size(); ++t)
            s = (s + static_cast<u128>(wts[t]) * tab[static_cast<uint64_t>(static_cast<u128>(deltas[t]) * ui % MM)]) % NK;
        g[i] = static_cast<uint64_t>(s);
        ui = static_cast<uint64_t>(static_cast<u128>(ui) * u % MM);
    }
    return g;
}

int32_t EisLadic::vmin(const std::vector<uint64_t>& g) const {
    int32_t best = 99;
    for (uint64_t x : g)
        if (x % NK) best = std::min<int32_t>(best, static_cast<int32_t>(Val(x, ell)) - static_cast<int32_t>(E));
    return best;
}

std::vector<uint64_t> EllBernoulliModEllPower(uint32_t N, uint64_t ell, uint32_t K) {
    const uint64_t NK = PowU(ell, K);
    std::vector<uint64_t> out(N + 1, 0);
    out[0] = ell % NK;
    if (N >= 1) out[1] = static_cast<uint64_t>(static_cast<u128>(NK - ell % NK) * InvModU64(2, NK) % NK);   // ell B_1 = -ell/2
    const std::vector<nf::BigInt> T = TangentNumbers(N / 2 + 1);
    for (uint32_t s = 2; s <= N; s += 2) {
        // B_s / s = sign T_{s-1} / (2^s (2^s - 1)), sign = (-1)^{s/2 - 1}
        const nf::BigInt& num = T[s / 2 - 1];
        const nf::BigInt den = nf::BigInt::pow(nf::BigInt(2), s);
        const nf::BigInt m1 = den - nf::BigInt(1);   // 2^s - 1
        uint64_t un = 0, ud1 = 0, ud2 = 0, us = 0;
        const uint32_t vn = ValuationAndUnitPart(num, ell, NK, un);
        const uint32_t vd1 = ValuationAndUnitPart(den, ell, NK, ud1);
        const uint32_t vd2 = ValuationAndUnitPart(m1, ell, NK, ud2);
        const uint32_t vs = Val(s, ell);
        us = (s / PowU(ell, vs)) % NK;
        const int32_t v = 1 + static_cast<int32_t>(vn + vs) - static_cast<int32_t>(vd1 + vd2);   // v(ell B_s)
        ENGINE_CHECK_MSG(v >= 0, "eis::EllBernoulliModEllPower: v(B_s) < -1");
        if (v >= static_cast<int32_t>(K)) continue;
        u128 x = static_cast<u128>(un) * us % NK;
        x = x * InvModU64(ud1, NK) % NK * InvModU64(ud2, NK) % NK;
        if ((s / 2) % 2 == 0) x = (NK - x) % NK;   // sign (-1)^{s/2-1}
        out[s] = static_cast<uint64_t>(x * PowU(ell, static_cast<uint32_t>(v)) % NK);
    }
    return out;
}

EisLadic BuildEisLadic(uint32_t n, uint64_t ell, uint32_t M, uint32_t R, unsigned threads, bool with_mu) {
    EisLadic L;
    L.ell = ell;
    L.n = n;
    L.k = n + 2;
    L.M = M;
    const uint32_t r = n + 1, vr = Val(r, ell);
    const uint32_t e2 = M + 1 + vr;
    L.E = M + e2;
    L.K = R + L.E;
    ENGINE_CHECK_MSG(static_cast<double>(L.K) * std::log2(static_cast<double>(ell)) < 62.0, "eis::BuildEisLadic: ell^K >= 2^62");
    L.NK = PowU(ell, L.K);
    const uint64_t NK = L.NK, MM = PowU(ell, M);
    const std::vector<uint64_t> Bl = EllBernoulliModEllPower(r, ell, L.K);
    // E_{k-1}(x) ell^{e2} = sum_s C(r,s) (ell B_s) x^{r-s} ell^{M(s-1) + e2 - 1} / r
    const uint64_t ur = InvLadic(r / PowU(ell, vr), NK);
    std::vector<std::pair<uint32_t, uint64_t>> coef;   // (exponent of x, coefficient)
    std::vector<uint64_t> binom(r + 1, 1);
    // C(r, s) mod NK via Pascal (r <= a few hundred)
    {
        std::vector<uint64_t> row{1};
        for (uint32_t i = 1; i <= r; ++i) {
            std::vector<uint64_t> nr(i + 1, 1);
            for (uint32_t j = 1; j < i; ++j) nr[j] = (row[j - 1] + row[j]) % NK;
            row = nr;
        }
        binom = row;
    }
    for (uint32_t s = 0; s <= r; ++s) {
        if (Bl[s] == 0) continue;
        const int64_t ex = static_cast<int64_t>(M) * (static_cast<int64_t>(s) - 1) + e2 - 1 - vr;
        ENGINE_CHECK_MSG(ex >= 0, "eis::BuildEisLadic: negative exponent");
        if (ex >= static_cast<int64_t>(L.K)) continue;
        const uint64_t c = static_cast<uint64_t>(static_cast<u128>(binom[s]) * Bl[s] % NK * PowU(ell, static_cast<uint32_t>(ex)) % NK * ur % NK);
        coef.emplace_back(r - s, c);
    }
    L.e1.assign(MM, 0);
    L.ek.assign(MM, 0);
    const uint64_t h = InvLadic(2, NK);
    std::vector<uint64_t> units;
    for (uint64_t x = 1; x < MM; ++x) {
        if (x % ell == 0) continue;
        units.push_back(x);
        L.e1[x] = static_cast<uint64_t>(static_cast<u128>((2 * x + NK - MM % NK) % NK) * h % NK);   // E_1(x) ell^M = x - M/2
        u128 acc = 0;
        for (const auto& ce : coef) acc = (acc + static_cast<u128>(ce.second) * PowModP(x, ce.first, NK)) % NK;
        L.ek[x] = static_cast<uint64_t>(acc);
    }
    L.mu.assign(MM, 0);
    if (!with_mu) return L;
    oc_util::ParallelChunks(units.size(), threads, oc_util::kEisensteinChunks, [&](size_t lo, size_t hi) {
        for (size_t ia = lo; ia < hi; ++ia) {
            const uint64_t a = units[ia], ainv = InvModU64(a, MM);
            u128 s = 0;
            uint32_t cnt = 0;
            for (uint64_t x : units) {
                s += static_cast<u128>(L.e1[x]) * L.ek[static_cast<uint64_t>(static_cast<u128>(x) * ainv % MM)];
                if (++cnt == 8) {
                    s %= NK;
                    cnt = 0;
                }
            }
            L.mu[a] = (NK - static_cast<uint64_t>(s % NK)) % NK;
        }
    });
    return L;
}

std::vector<uint64_t> PolarElement(const EisLadic& L, uint32_t br, int64_t res_scale_num) {
    const uint64_t ell = L.ell, NK = L.NK;
    const uint64_t G = PowU(ell, L.M - 1), u = 1 + ell;
    const uint32_t k = L.k;
    const uint32_t pm = static_cast<uint32_t>(ell - 2), pk = static_cast<uint32_t>((k - 1) % (ell - 1));
    std::vector<uint64_t> out(G, 0);
    if (br != pm && br != pk) return out;
    // (1 - ell)(1 - ell^{k-1})(B_k/k) ell^{-M} [ / (k-1) on branch k-1 ], scaled by ell^E
    // exact B_k / k (tangent numbers) at precision K
    const BernoulliOverK bk = ExactBernoulliOverK(k);
    uint64_t un = 0, ud = 0;
    const uint32_t vn = ValuationAndUnitPart(bk.num, ell, NK, un), vd = ValuationAndUnitPart(bk.den, ell, NK, ud);
    const int32_t vbk = static_cast<int32_t>(vn) - static_cast<int32_t>(vd);
    u128 unit = static_cast<u128>(un) * InvModU64(ud, NK) % NK;
    if (bk.sign < 0) unit = (NK - unit) % NK;
    unit = unit * ((1 + NK - ell % NK) % NK) % NK;                                      // (1 - ell)
    unit = unit * ((1 + NK - PowModP(ell, k - 1, NK)) % NK) % NK;                       // (1 - ell^{k-1})
    const uint64_t sc = res_scale_num >= 0 ? static_cast<uint64_t>(res_scale_num) % NK : (NK - static_cast<uint64_t>(-res_scale_num) % NK) % NK;
    unit = unit * sc % NK;
    for (int pass = 0; pass < 2; ++pass) {
        const bool on1 = pass == 0 && br == pm, on2 = pass == 1 && br == pk;
        if (!on1 && !on2) continue;
        u128 c = unit;
        int32_t v = vbk - static_cast<int32_t>(L.M) + static_cast<int32_t>(L.E);
        if (on2) {
            const uint32_t vk = Val(k - 1, ell);
            c = c * InvModU64((k - 1) / PowU(ell, vk), NK) % NK;
            v -= static_cast<int32_t>(vk);
        }
        ENGINE_CHECK_MSG(v >= 0, "eis::PolarElement: scale too small");
        if (v >= static_cast<int32_t>(L.K)) continue;
        const uint64_t cs = static_cast<uint64_t>(c * PowU(ell, static_cast<uint32_t>(v)) % NK);
        const uint64_t step = on1 ? u % NK : PowModP(InvModU64(u % NK, NK), k - 1, NK);
        uint64_t w = 1 % NK;
        for (uint64_t i = 0; i < G; ++i) {
            out[i] = static_cast<uint64_t>((out[i] + static_cast<u128>(cs) * w) % NK);
            w = static_cast<uint64_t>(static_cast<u128>(w) * step % NK);
        }
    }
    return out;
}

Lad LadMul(const Lad& a, const Lad& b, uint64_t ell, uint32_t P) {
    Lad r;
    if (a.zero || b.zero) {
        r.zero = true;
        return r;
    }
    const uint64_t NP = PowU(ell, P);
    r.v = a.v + b.v;
    r.unit = static_cast<uint64_t>(static_cast<u128>(a.unit) * b.unit % NP);
    return r;
}

Lad Residue(uint32_t k, uint64_t ell, uint32_t P) {
    const uint64_t NP = PowU(ell, P);
    // log_ell(u)/ell = sum_{m>=1} (-1)^{m+1} ell^{m-1} / m, to ell^P
    const uint32_t W = P + 4;
    const uint64_t NW = PowU(ell, std::min<uint32_t>(W, static_cast<uint32_t>(62.0 / std::log2(static_cast<double>(ell)))));
    u128 lg = 0;
    for (uint32_t m = 1; m < 8 * W + 8; ++m) {
        const uint32_t vm = Val(m, ell);
        const int32_t e = static_cast<int32_t>(m) - 1 - static_cast<int32_t>(vm);
        if (e >= static_cast<int32_t>(P)) continue;
        u128 t = static_cast<u128>(PowU(ell, static_cast<uint32_t>(e))) * InvModU64((m / PowU(ell, vm)) % NW, NW) % NW;
        if (m % 2 == 0) t = (NW - t) % NW;
        lg = (lg + t) % NW;
    }
    const BernoulliOverK bk = ExactBernoulliOverK(k);
    uint64_t un = 0, ud = 0;
    const uint32_t vn = ValuationAndUnitPart(bk.num, ell, NP, un), vd = ValuationAndUnitPart(bk.den, ell, NP, ud);
    Lad r;
    r.v = static_cast<int32_t>(vn) - static_cast<int32_t>(vd);
    u128 x = static_cast<u128>(un) * InvModU64(ud, NP) % NP;
    if (bk.sign < 0) x = (NP - x) % NP;
    x = x * ((1 + NP - ell % NP) % NP) % NP;
    x = x * static_cast<uint64_t>(lg % NP) % NP;
    x = x * ((1 + NP - PowModP(ell, k - 1, NP)) % NP) % NP;
    r.unit = static_cast<uint64_t>(x);
    return r;
}

Lad PolarValue(const Lad& Res, uint64_t ell, uint64_t r, uint32_t P) {
    // u^r - 1 = ell^{1 + v(r)} * unit; compute mod ell^{P + 1 + v(r)}
    const uint32_t vr = Val(r, ell);
    const uint32_t W = P + 1 + vr;
    ENGINE_CHECK_MSG(static_cast<double>(W) * std::log2(static_cast<double>(ell)) < 62.0, "eis::PolarValue: precision");
    const uint64_t NW = PowU(ell, W), NP = PowU(ell, P);
    const uint64_t x = (PowModP(1 + ell, r, NW) + NW - 1) % NW;
    const uint64_t un = (x / PowU(ell, 1 + vr)) % NP;
    Lad d;
    d.v = -static_cast<int32_t>(1 + vr);
    d.unit = InvModU64(un, NP);
    return LadMul(Res, d, ell, P);
}

uint64_t LadScaled(const Lad& x, uint64_t ell, uint32_t S, uint32_t W) {
    if (x.zero) return 0;
    const uint64_t N = PowU(ell, S + W);
    const int32_t e = x.v + static_cast<int32_t>(S);
    ENGINE_CHECK_MSG(e >= 0, "eis::LadScaled: scale too small");
    if (e >= static_cast<int32_t>(S + W)) return 0;
    return static_cast<uint64_t>(static_cast<u128>(x.unit % N) * PowU(ell, static_cast<uint32_t>(e)) % N);
}

} // namespace eis
} // namespace lvalue
} // namespace hecke_ext
