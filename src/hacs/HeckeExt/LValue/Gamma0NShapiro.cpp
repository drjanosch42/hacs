// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P6) the Birch route on Gamma_0(N) by Shapiro -- see Gamma0NShapiro.h.
#include "Gamma0NShapiro.h"

#include "Bernoulli.h"   // TeichmullerPowMod, PrimitiveRootModPrime

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../Util/ParallelFor.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace hecke_ext {
namespace lvalue {
namespace shapiro {

namespace {
using u128 = unsigned __int128;

int64_t Mod(int64_t a, int64_t N) {
    const int64_t r = a % N;
    return r < 0 ? r + N : r;
}
int64_t Gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
// x a + y b = gcd(a, b) >= 0
int64_t ExtGcd(int64_t a, int64_t b, int64_t& x, int64_t& y) {
    int64_t x0 = 1, y0 = 0, x1 = 0, y1 = 1;
    while (b != 0) {
        const int64_t q = a / b;
        int64_t t = a - q * b;
        a = b;
        b = t;
        t = x0 - q * x1;
        x0 = x1;
        x1 = t;
        t = y0 - q * y1;
        y0 = y1;
        y1 = t;
    }
    if (a < 0) {
        a = -a;
        x0 = -x0;
        y0 = -y0;
    }
    x = x0;
    y = y0;
    return a;
}
using zk::Mul;                              // the shared 2x2 kit (../ZellK.h; S24 CL-G1)
inline Mat2 InvSL2(const Mat2& g) { return zk::Adj(g); }   // = adj g for det g = 1

using engine::scalar::pow_wrap;
bool IsPrimeU(uint64_t p) {
    if (p < 2) return false;
    for (uint64_t d = 2; d * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}
Vec Block(const Vec& P, uint32_t x, uint32_t n) {
    return Vec(P.begin() + static_cast<std::ptrdiff_t>(x) * (n + 1), P.begin() + static_cast<std::ptrdiff_t>(x + 1) * (n + 1));
}
void AddInto(const Ring& R, Vec& acc, const Vec& v) {
    for (size_t i = 0; i < acc.size(); ++i) acc[i] = R.add(acc[i], v[i]);
}
// rho(A) as a dense (n+1) x (n+1) matrix (column t = rho(A) u_t), from SymRho::apply.
std::vector<Vec> RhoColumns(const SymRho& rho, const Mat2& A) {
    const uint32_t n = rho.n();
    std::vector<Vec> cols(n + 1);
    for (uint32_t t = 0; t <= n; ++t) {
        Vec e(n + 1, 0);
        e[t] = 1 % rho.ring().q;
        cols[t] = rho.apply(A, e);
    }
    return cols;
}
} // namespace

// ---- P^1(Z/N) -------------------------------------------------------------------------------------------------------
P1N::P1N(uint64_t N) : N_(N) {
    if (N < 1 || N > kMaxLevel) throw std::invalid_argument("shapiro::P1N: 1 <= N <= " + std::to_string(kMaxLevel));
    if (N == 1) {
        pts_.push_back({0, 0});
        table_.assign(1, 0);
        lifts_.push_back(Mat2{1, 0, 0, 1});
        return;
    }
    const int64_t n = static_cast<int64_t>(N);
    std::vector<int64_t> units;
    for (int64_t u = 1; u < n; ++u)
        if (Gcd(u, n) == 1) units.push_back(u);
    const uint32_t NONE = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> orbit(static_cast<size_t>(N * N), NONE);
    std::vector<std::pair<uint64_t, uint64_t>> mins;
    for (int64_t c = 0; c < n; ++c)
        for (int64_t d = 0; d < n; ++d) {
            if (Gcd(Gcd(c, d), n) != 1 || orbit[static_cast<size_t>(c * n + d)] != NONE) continue;
            const uint32_t id = static_cast<uint32_t>(mins.size());
            std::pair<uint64_t, uint64_t> best{static_cast<uint64_t>(c), static_cast<uint64_t>(d)};
            for (int64_t u : units) {
                const int64_t uc = (u * c) % n, ud = (u * d) % n;
                orbit[static_cast<size_t>(uc * n + ud)] = id;
                best = std::min(best, std::pair<uint64_t, uint64_t>{static_cast<uint64_t>(uc), static_cast<uint64_t>(ud)});
            }
            mins.push_back(best);
        }
    std::vector<uint32_t> order(mins.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return mins[a] < mins[b]; });
    std::vector<uint32_t> rank(mins.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        rank[order[i]] = i;
        pts_.push_back(mins[order[i]]);
    }
    table_.assign(orbit.size(), NONE);
    for (size_t k = 0; k < orbit.size(); ++k)
        if (orbit[k] != NONE) table_[k] = rank[orbit[k]];
    lifts_.resize(pts_.size());
    for (uint32_t i = 0; i < pts_.size(); ++i) {
        const int64_t c = static_cast<int64_t>(pts_[i].first), d = static_cast<int64_t>(pts_[i].second);
        const int64_t cc = (c == 0) ? n : c;
        int64_t dd = d;
        while (Gcd(cc, dd) != 1) dd += n;
        int64_t x = 0, y = 0;
        ExtGcd(cc, dd, x, y);   // x cc + y dd = 1
        const Mat2 g{y, -x, cc, dd};
        ENGINE_CHECK_MSG(g.a * g.d - g.b * g.c == 1, "shapiro::P1N: the lift is not in SL_2(Z)");
        ENGINE_CHECK_MSG(idx(g.c, g.d) == i, "shapiro::P1N: the lift's bottom row is not the point");
        lifts_[i] = g;
    }
}

uint32_t P1N::idx(int64_t c, int64_t d) const {
    if (N_ == 1) return 0;
    const int64_t n = static_cast<int64_t>(N_);
    const uint32_t v = table_[static_cast<size_t>(Mod(c, n) * n + Mod(d, n))];
    ENGINE_CHECK_MSG(v != std::numeric_limits<uint32_t>::max(), "shapiro::P1N::idx: gcd(c, d, N) != 1");
    return v;
}
Mat2 P1N::lift(uint32_t i) const { return lifts_.at(i); }
uint32_t P1N::S(uint32_t i) const {
    const auto p = pts_[i];
    return idx(static_cast<int64_t>(p.second), -static_cast<int64_t>(p.first));
}
uint32_t P1N::U(uint32_t i) const {
    const auto p = pts_[i];
    return idx(static_cast<int64_t>(p.second), static_cast<int64_t>(p.second) - static_cast<int64_t>(p.first));
}
uint32_t P1N::iota(uint32_t i) const {
    const auto p = pts_[i];
    return idx(-static_cast<int64_t>(p.first), static_cast<int64_t>(p.second));
}
uint64_t P1N::index_in_sl2z() const { return pts_.size(); }

// ---- the symbol module --------------------------------------------------------------------------------------------
Vec Gamma0NCell::symbol(const std::vector<uint64_t>& coords) const {
    ENGINE_CHECK_MSG(coords.size() == basis.size(), "shapiro::Gamma0NCell::symbol: one coordinate per generator");
    Vec P(static_cast<size_t>(L) * (n + 1), 0);
    for (size_t i = 0; i < basis.size(); ++i) {
        // ell_pow(m) reads as 0 (ell^m == 0 in Z/ell^m): a full-order generator is reduced mod q only
        const uint64_t c = exps[i] >= R.m ? coords[i] % R.q : coords[i] % R.ell_pow(exps[i]);
        if (c == 0) continue;
        for (size_t k = 0; k < P.size(); ++k) P[k] = R.add(P[k], R.mul(c, basis[i][k]));
    }
    return P;
}

namespace {
Gamma0NCell RunCell(uint64_t N, uint32_t n, uint64_t ell, uint32_t m, int sign, bool wrong) {
    const auto t0 = std::chrono::steady_clock::now();
    if (n % 2 != 0) throw std::invalid_argument("shapiro::RunGamma0NCell: n even (-I acts by (-1)^n)");
    if (ell < 5 || !IsPrimeU(ell)) throw std::invalid_argument("shapiro::RunGamma0NCell: ell >= 5 prime");
    if (m < 1) throw std::invalid_argument("shapiro::RunGamma0NCell: m >= 1");
    if (sign < -1 || sign > 1) throw std::invalid_argument("shapiro::RunGamma0NCell: sign in {-1, 0, +1}");
    Gamma0NCell C;
    C.N = N;
    C.n = n;
    C.ell = ell;
    C.m = m;
    C.sign = sign;
    C.R = birch::MakeRing(ell, m);
    ENGINE_CHECK_MSG(C.R.q < (uint64_t(1) << 62), "shapiro::RunGamma0NCell: ell^m < 2^62");
    const P1N p1(N);
    const uint32_t L = p1.size(), B = n + 1;
    C.L = L;
    const Ring& R = C.R;
    const SymRho rho(n, R);
    const Mat2 S{0, -1, 1, 0}, U{0, -1, 1, 1};
    const std::vector<Vec> rS = RhoColumns(rho, S), rU = RhoColumns(rho, U), rU2 = RhoColumns(rho, Mul(U, U));
    auto Sx = [&](uint32_t x) { return wrong ? x : p1.S(x); };
    auto Ux = [&](uint32_t x) { return wrong ? x : p1.U(x); };
    auto Ix = [&](uint32_t x) { return wrong ? x : p1.iota(x); };
    // S-orbit elimination: P_x = M_x y_{blk(x)}, M_x in {I, -rho(S)}
    const uint32_t NONE = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> blk(L, NONE);
    std::vector<uint8_t> twisted(L, 0);
    std::vector<uint32_t> fixed;
    uint32_t nb = 0;
    for (uint32_t x = 0; x < L; ++x) {
        if (blk[x] != NONE) continue;
        const uint32_t xs = Sx(x);
        blk[x] = nb;
        if (xs == x) {
            fixed.push_back(x);
        } else {
            blk[xs] = nb;
            twisted[xs] = 1;
        }
        ++C.s_orbits;
        ++nb;
    }
    C.s_fixed = static_cast<uint32_t>(fixed.size());
    const uint32_t cols = nb * B;
    C.unknowns = cols;
    std::vector<Vec> rows;   // each row has length cols
    // add  coef * rho(A) M_x  (as an (n+1) x (n+1) block) into the rows [r0, r0 + B) at block blk(x); rhoA == nullptr means I
    auto add_block = [&](std::vector<Vec>& rr, size_t r0, const std::vector<Vec>* rhoA, uint32_t x, uint64_t coef) {
        // the block matrix K = rho(A) M_x: column t = rho(A) (M_x u_t)
        for (uint32_t t = 0; t < B; ++t) {
            Vec col(B, 0);
            if (twisted[x]) {
                for (uint32_t s = 0; s < B; ++s) col[s] = R.neg(rS[t][s]);
            } else {
                col[t] = 1 % R.q;
            }
            Vec kc(B, 0);
            if (rhoA) {
                for (uint32_t s = 0; s < B; ++s)
                    if (col[s])
                        for (uint32_t r = 0; r < B; ++r) kc[r] = R.add(kc[r], R.mul((*rhoA)[s][r], col[s]));
            } else {
                kc = col;
            }
            for (uint32_t r = 0; r < B; ++r)
                if (kc[r]) rr[r0 + r][static_cast<size_t>(blk[x]) * B + t] = R.add(rr[r0 + r][static_cast<size_t>(blk[x]) * B + t], R.mul(coef, kc[r]));
        }
    };
    auto new_rows = [&]() {
        const size_t r0 = rows.size();
        for (uint32_t r = 0; r < B; ++r) rows.emplace_back(cols, 0);
        return r0;
    };
    // the S-fixed points: (1 + rho(S)) P_x = 0
    for (uint32_t x : fixed) {
        const size_t r0 = new_rows();
        add_block(rows, r0, nullptr, x, 1 % R.q);
        add_block(rows, r0, &rS, x, 1 % R.q);
    }
    // the U-orbits: P_x + rho(U) P_{xU} + rho(U^2) P_{xU^2} = 0 (once per orbit)
    std::vector<uint8_t> useen(L, 0);
    for (uint32_t x = 0; x < L; ++x) {
        if (useen[x]) continue;
        const uint32_t xu = Ux(x), xu2 = Ux(xu);
        useen[x] = useen[xu] = useen[xu2] = 1;
        ENGINE_CHECK_MSG(Ux(xu2) == x, "shapiro::RunGamma0NCell: U does not act with order 3 on P^1(Z/N)");
        ++C.u_orbits;
        const size_t r0 = new_rows();
        add_block(rows, r0, nullptr, x, 1 % R.q);
        add_block(rows, r0, &rU, xu, 1 % R.q);
        add_block(rows, r0, &rU2, xu2, 1 % R.q);
    }
    // the sign: rho(iota) P_{x'} - s P_x = 0 (once per iota-orbit)
    if (sign != 0) {
        std::vector<Vec> rI(B, Vec(B, 0));
        for (uint32_t t = 0; t < B; ++t) rI[t][t] = (t % 2 == 0) ? 1 % R.q : R.neg(1 % R.q);
        std::vector<uint8_t> iseen(L, 0);
        for (uint32_t x = 0; x < L; ++x) {
            if (iseen[x]) continue;
            const uint32_t xi = Ix(x);
            iseen[x] = iseen[xi] = 1;
            ++C.iota_orbits;
            const size_t r0 = new_rows();
            add_block(rows, r0, &rI, xi, 1 % R.q);
            add_block(rows, r0, nullptr, x, sign > 0 ? R.neg(1 % R.q) : 1 % R.q);   // - s P_x
        }
    }
    C.relation_rows = static_cast<uint32_t>(rows.size());
    birch::Mat A(static_cast<uint32_t>(rows.size()), cols);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < cols; ++j) A.at(i, j) = rows[i][j];
    std::vector<Vec> ygens;
    std::vector<uint32_t> yexps;
    if (A.r == 0) {
        for (uint32_t j = 0; j < cols; ++j) {
            Vec e(cols, 0);
            e[j] = 1 % R.q;
            ygens.push_back(e);
            yexps.push_back(m);
        }
    } else {
        birch::KernelGenerators(R, A, ygens, yexps);
    }
    // back to the full symbol P = (M_x y_{blk(x)})_x
    for (size_t g = 0; g < ygens.size(); ++g) {
        Vec P(static_cast<size_t>(L) * B, 0);
        for (uint32_t x = 0; x < L; ++x) {
            const Vec y(ygens[g].begin() + static_cast<std::ptrdiff_t>(blk[x]) * B, ygens[g].begin() + static_cast<std::ptrdiff_t>(blk[x] + 1) * B);
            Vec px = y;
            if (twisted[x]) {
                px = rho.S(y);
                for (auto& v : px) v = R.neg(v);
            }
            std::copy(px.begin(), px.end(), P.begin() + static_cast<std::ptrdiff_t>(x) * B);
        }
        C.basis.push_back(P);
        C.exps.push_back(yexps[g]);
    }
    C.free_module = std::all_of(C.exps.begin(), C.exps.end(), [&](uint32_t e) { return e == m; });
    C.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return C;
}
} // namespace

Gamma0NCell RunGamma0NCell(uint64_t N, uint32_t n, uint64_t ell, uint32_t m, int sign) { return RunCell(N, n, ell, m, sign, false); }
Gamma0NCell RunGamma0NCellWrongCosets(uint64_t N, uint32_t n, uint64_t ell, uint32_t m, int sign) { return RunCell(N, n, ell, m, sign, true); }

// ---- Phi, Hecke, Res / Cores ----------------------------------------------------------------------------------------
namespace {
Vec PhiOoTo(const SymRho& rho, const P1N& p1, const Vec& P, int64_t a, int64_t M) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    ENGINE_CHECK_MSG(P.size() == static_cast<size_t>(p1.size()) * (n + 1), "shapiro::PhiOoTo: symbol length");
    Vec acc(n + 1, 0);
    if (M == 0) return acc;
    if (M < 0) {
        a = -a;
        M = -M;
    }
    for (const Mat2& g : birch::ManinPath(a, M)) AddInto(R, acc, rho.apply(g, Block(P, p1.idx(g.c, g.d), n)));
    return acc;
}
} // namespace

Vec PhiCuspN(const SymRho& rho, const P1N& p1, const Vec& P, int64_t a, int64_t M) {
    Vec v = PhiOoTo(rho, p1, P, a, M);
    for (auto& x : v) x = rho.ring().neg(x);
    return v;
}

namespace {
Vec PhiPairN(const SymRho& rho, const P1N& p1, const Vec& P, int64_t rn, int64_t rd, int64_t sn, int64_t sd) {
    // Phi({r, s}) = Phi({oo, s}) - Phi({oo, r})
    const Vec A = PhiOoTo(rho, p1, P, sn, sd), Bv = PhiOoTo(rho, p1, P, rn, rd);
    Vec o(A.size());
    for (size_t i = 0; i < A.size(); ++i) o[i] = rho.ring().sub(A[i], Bv[i]);
    return o;
}
} // namespace

bool InWN(const SymRho& rho, const P1N& p1, const Vec& P) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n(), L = p1.size();
    const Mat2 U{0, -1, 1, 1};
    for (uint32_t x = 0; x < L; ++x) {
        const Vec px = Block(P, x, n);
        const Vec s = rho.S(Block(P, p1.S(x), n));
        for (uint32_t t = 0; t <= n; ++t)
            if (R.add(px[t], s[t]) != 0) return false;
        const uint32_t xu = p1.U(x), xu2 = p1.U(xu);
        const Vec u1 = rho.apply(U, Block(P, xu, n));
        const Vec u2 = rho.apply(U, rho.apply(U, Block(P, xu2, n)));
        for (uint32_t t = 0; t <= n; ++t)
            if (R.add(R.add(px[t], u1[t]), u2[t]) != 0) return false;
    }
    return true;
}

namespace {
Vec HeckeImpl(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p, bool wrong) {
    ENGINE_CHECK_MSG(p >= 2 && IsPrimeU(p), "shapiro::HeckeTpN: p prime");
    ENGINE_CHECK_MSG(p1.N() % p != 0, "shapiro::HeckeTpN: p must not divide N (U_p is not this operator)");
    const Ring& R = rho.ring();
    const uint32_t n = rho.n(), L = p1.size();
    const int64_t pp = static_cast<int64_t>(p);
    std::vector<std::pair<Mat2, Mat2>> alphas;   // (alpha, adj alpha)
    alphas.push_back({Mat2{pp, 0, 0, 1}, Mat2{1, 0, 0, pp}});
    for (int64_t b = 0; b < pp; ++b) alphas.push_back({Mat2{1, b, 0, pp}, Mat2{pp, -b, 0, 1}});
    Vec out(P.size(), 0);
    for (uint32_t x = 0; x < L; ++x) {
        const Mat2 g = p1.lift(x);
        Vec acc(n + 1, 0);
        for (const auto& al : alphas) {
            const Mat2 ag = Mul(al.first, g);
            // {ag 0, ag oo} = {b/d, a/c}
            const Vec ph = PhiPairN(rho, p1, P, ag.b, ag.d, ag.a, ag.c);
            AddInto(R, acc, rho.apply(wrong ? al.first : al.second, ph));
        }
        const Vec v = rho.apply(InvSL2(g), acc);
        std::copy(v.begin(), v.end(), out.begin() + static_cast<std::ptrdiff_t>(x) * (n + 1));
    }
    return out;
}
} // namespace

Vec HeckeTpN(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p) { return HeckeImpl(rho, p1, P, p, false); }
Vec HeckeTpNWrongAdjoint(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p) { return HeckeImpl(rho, p1, P, p, true); }

namespace {
Vec IotaN(const SymRho& rho, const P1N& p1, const Vec& P) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n(), L = p1.size();
    Vec out(P.size(), 0);
    for (uint32_t x = 0; x < L; ++x) {
        const uint32_t xi = p1.iota(x);
        for (uint32_t t = 0; t <= n; ++t) {
            const uint64_t v = P[static_cast<size_t>(xi) * (n + 1) + t] % R.q;
            out[static_cast<size_t>(x) * (n + 1) + t] = (t % 2 == 0) ? v : R.neg(v);
        }
    }
    return out;
}
} // namespace

Vec ResFromLevelOne(const Vec& P1v, uint32_t L) {
    Vec out;
    out.reserve(P1v.size() * L);
    for (uint32_t x = 0; x < L; ++x) out.insert(out.end(), P1v.begin(), P1v.end());
    return out;
}

Vec CoresToLevelOne(const Ring& R, const Vec& P, uint32_t L, uint32_t n) {
    ENGINE_CHECK_MSG(P.size() == static_cast<size_t>(L) * (n + 1), "shapiro::CoresToLevelOne: symbol length");
    Vec out(n + 1, 0);
    for (uint32_t x = 0; x < L; ++x)
        for (uint32_t t = 0; t <= n; ++t) out[t] = R.add(out[t], P[static_cast<size_t>(x) * (n + 1) + t]);
    return out;
}

Vec EvAllJN(const SymRho& rho, const P1N& p1, const Vec& P, const birch::DirichletCharacter& chi) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    Vec tot(n + 1, 0);
    const int64_t M = static_cast<int64_t>(chi.M);
    for (int64_t a = 0; a < M; ++a) {
        const uint64_t cv = chi.chibar[static_cast<size_t>(a)];
        if (cv == 0) continue;
        const Vec w = rho.apply(Mat2{M, -a, 0, 1}, PhiCuspN(rho, p1, P, a, M));
        for (uint32_t t = 0; t <= n; ++t) tot[t] = R.add(tot[t], R.mul(cv, w[t]));
    }
    Vec ev(n + 1, 0);
    for (uint32_t j = 0; j <= n; ++j) ev[j] = tot[n - j];
    return ev;
}

bool EigenvalueN(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t p, uint64_t& lambda) {
    const Ring& R = rho.ring();
    size_t t0 = P.size();
    for (size_t i = 0; i < P.size(); ++i)
        if (R.unit(P[i])) {
            t0 = i;
            break;
        }
    if (t0 == P.size()) return false;
    const Vec T = HeckeTpN(rho, p1, P, p);
    lambda = R.mul(T[t0], R.inv(P[t0]));
    for (size_t i = 0; i < P.size(); ++i)
        if (T[i] != R.mul(lambda, P[i] % R.q)) return false;
    return true;
}

void EigenSubmodule(const Gamma0NCell& cell, uint64_t p, uint64_t lambda, std::vector<Vec>& gens, std::vector<uint32_t>& exps) {
    ENGINE_CHECK_MSG(cell.free_module, "shapiro::EigenSubmodule: the cell's module must be free");
    gens.clear();
    exps.clear();
    if (cell.basis.empty()) return;
    const Ring& R = cell.R;
    const SymRho rho(cell.n, R);
    const P1N p1(cell.N);
    const uint32_t len = static_cast<uint32_t>(cell.basis[0].size());
    std::vector<Vec> cols;
    for (const Vec& b : cell.basis) {
        Vec t = HeckeTpN(rho, p1, b, p);
        for (uint32_t i = 0; i < len; ++i) t[i] = R.sub(t[i], R.mul(lambda % R.q, b[i]));
        cols.push_back(t);
    }
    std::vector<Vec> kg;
    std::vector<uint32_t> ke;
    birch::KernelGenerators(R, birch::ColumnsToMat(cols, len), kg, ke);
    for (size_t g = 0; g < kg.size(); ++g) {
        Vec v(len, 0);
        for (size_t j = 0; j < cell.basis.size(); ++j)
            if (kg[g][j])
                for (uint32_t i = 0; i < len; ++i) v[i] = R.add(v[i], R.mul(kg[g][j], cell.basis[j][i]));
        gens.push_back(v);
        exps.push_back(ke[g]);
    }
}

bool NormaliseFirstUnit(const Ring& R, Vec& P) {
    for (size_t i = 0; i < P.size(); ++i)
        if (R.unit(P[i])) {
            const uint64_t iv = R.inv(P[i]);
            for (auto& x : P) x = R.mul(x % R.q, iv);
            return true;
        }
    return false;
}

// ---- measures -----------------------------------------------------------------------------------------------------
namespace {
// c(b, Mod) = - sum_g P_{x(g)}(g_c, g_d) over Manin's matrices of b/Mod; Pc[x] = the coset blocks reduced mod q.
uint64_t CValue(const Ring& R, const P1N& p1, const std::vector<Vec>& Pc, uint32_t n, int64_t b, int64_t Mod) {
    uint64_t sum = 0;
    for (const Mat2& g : birch::ManinPath(b, Mod)) {
        const Vec& Px = Pc[p1.idx(g.c, g.d)];
        const uint64_t c = R.from_int(g.c), d = R.from_int(g.d);
        uint64_t acc = 0, cp = 1 % R.q;
        for (uint32_t t = n + 1; t-- > 0;) {
            acc = R.add(R.mul(acc, d), R.mul(Px[t], cp));
            cp = R.mul(cp, c);
        }
        sum = R.add(sum, acc);
    }
    return R.neg(sum);
}
std::vector<Vec> CosetBlocks(const Ring& R, const P1N& p1, const Vec& P, uint32_t n) {
    ENGINE_CHECK_MSG(P.size() == static_cast<size_t>(p1.size()) * (n + 1), "shapiro: symbol length");
    std::vector<Vec> Pc(p1.size());
    for (uint32_t x = 0; x < p1.size(); ++x) {
        Pc[x] = Block(P, x, n);
        for (auto& v : Pc[x]) v %= R.q;
    }
    return Pc;
}
} // namespace

ModularSymbolMeasure BuildMeasureN(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t a_ell, uint64_t alpha, uint32_t M,
                                   bool ell_n_factor, unsigned threads) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    ENGINE_CHECK_MSG(M >= 1, "shapiro::BuildMeasureN: level M >= 1");
    ENGINE_CHECK_MSG(R.unit(alpha), "shapiro::BuildMeasureN: alpha must be a unit");
    ENGINE_CHECK_MSG(p1.N() % R.ell != 0, "shapiro::BuildMeasureN: ell must not divide N");
    if (threads == 0) threads = birch::MeasureThreads();
    const std::vector<Vec> Pc = CosetBlocks(R, p1, P, n);
    ModularSymbolMeasure out;
    out.ell = R.ell;
    out.R = R.m;
    out.M = M;
    out.n = n;
    out.q = R.q;
    out.a_ell = a_ell % R.q;
    out.alpha = alpha % R.q;
    out.ell_n_factor = ell_n_factor;
    const Vec ip = IotaN(rho, p1, P);
    bool plus = true, minus = true;
    for (size_t i = 0; i < P.size(); ++i) {
        if (ip[i] != P[i] % R.q) plus = false;
        if (ip[i] != R.neg(P[i] % R.q)) minus = false;
    }
    out.geometric_sign = (plus && !minus) ? 1 : (minus && !plus) ? -1 : 0;
    const uint64_t c0 = CValue(R, p1, Pc, n, 0, 1);
    out.c.assign(M + 2, {});
    out.c[0] = {c0};
    out.c[1] = {c0};
    for (uint32_t m = 1; m <= M; ++m) {
        const uint64_t Nm = pow_wrap(R.ell, m);
        ENGINE_CHECK_MSG(Nm < (uint64_t(1) << 62), "shapiro::BuildMeasureN: ell^M too large");
        std::vector<uint64_t>& cm = out.c[m + 1];
        cm.assign(Nm, 0);
        oc_util::ParallelChunks(Nm, threads, oc_util::kShapiroChunks, [&](uint64_t s, uint64_t e) {
            for (uint64_t a = s; a < e; ++a) cm[a] = CValue(R, p1, Pc, n, static_cast<int64_t>(a), static_cast<int64_t>(Nm));
        });
    }
    const uint64_t ia = R.inv(alpha % R.q);
    const uint64_t ln = ell_n_factor ? R.pow(R.ell, n) : 1 % R.q;
    out.mu.assign(M + 1, {});
    uint64_t am = 1 % R.q;
    for (uint32_t m = 0; m <= M; ++m) {
        const uint64_t am1 = R.mul(am, ia);
        const std::vector<uint64_t>& cur = out.c[m + 1];
        const std::vector<uint64_t>& prev = out.c[m];
        std::vector<uint64_t>& mu = out.mu[m];
        mu.assign(cur.size(), 0);
        for (uint64_t a = 0; a < cur.size(); ++a) mu[a] = R.sub(R.mul(am, cur[a]), R.mul(R.mul(am1, ln), prev[a % prev.size()]));
        am = am1;
    }
    return out;
}

TameMeasure BuildTameMeasure(const SymRho& rho, const P1N& p1, const Vec& P, uint64_t T, uint64_t a_ell, uint64_t alpha, uint32_t M,
                             bool ell_n_factor, unsigned threads) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    ENGINE_CHECK_MSG(M >= 1 && T >= 1, "shapiro::BuildTameMeasure: M >= 1, T >= 1");
    ENGINE_CHECK_MSG(T % R.ell != 0 && p1.N() % R.ell != 0, "shapiro::BuildTameMeasure: ell must not divide N T");
    ENGINE_CHECK_MSG(R.unit(alpha), "shapiro::BuildTameMeasure: alpha must be a unit");
    if (threads == 0) threads = birch::MeasureThreads();
    const std::vector<Vec> Pc = CosetBlocks(R, p1, P, n);
    TameMeasure out;
    out.N = p1.N();
    out.T = T;
    out.ell = R.ell;
    out.R = R.m;
    out.M = M;
    out.n = n;
    out.q = R.q;
    out.a_ell = a_ell % R.q;
    out.alpha = alpha % R.q;
    out.ell_n_factor = ell_n_factor;
    out.c.assign(M + 1, {});
    for (uint32_t m = 0; m <= M; ++m) {
        const uint64_t Mod = T * pow_wrap(R.ell, m);
        ENGINE_CHECK_MSG(Mod < (uint64_t(1) << 40), "shapiro::BuildTameMeasure: T ell^M too large");
        std::vector<uint64_t>& cm = out.c[m];
        cm.assign(Mod, 0);
        oc_util::ParallelChunks(Mod, threads, oc_util::kShapiroChunks, [&](uint64_t s, uint64_t e) {
            for (uint64_t b = s; b < e; ++b) cm[b] = CValue(R, p1, Pc, n, static_cast<int64_t>(b), static_cast<int64_t>(Mod));
        });
    }
    const uint64_t ia = R.inv(alpha % R.q);
    const uint64_t ln = ell_n_factor ? R.pow(R.ell, n) : 1 % R.q;
    out.mu.assign(M + 1, {});
    uint64_t am = ia;   // alpha^{-m}, m = 1
    for (uint32_t m = 1; m <= M; ++m) {
        const uint64_t am1 = R.mul(am, ia);
        const std::vector<uint64_t>& cur = out.c[m];
        const std::vector<uint64_t>& prev = out.c[m - 1];
        std::vector<uint64_t>& mu = out.mu[m];
        mu.assign(cur.size(), 0);
        for (uint64_t b = 0; b < cur.size(); ++b) mu[b] = R.sub(R.mul(am, cur[b]), R.mul(R.mul(am1, ln), prev[b % prev.size()]));
        am = am1;
    }
    return out;
}

uint32_t TameMeasure::three_term_failures() const {
    const Ring R = birch::MakeRing(ell, this->R);
    const uint64_t ln = R.pow(ell, n);
    uint32_t bad = 0;
    for (uint32_t m = 1; m + 1 <= M; ++m) {
        const uint64_t Mod = c[m].size(), Mp = c[m - 1].size();
        for (uint64_t b = 0; b < Mod; ++b) {
            uint64_t s = 0;
            for (uint64_t t = 0; t < ell; ++t) s = R.add(s, c[m + 1][b + t * Mod]);
            if (s != R.sub(R.mul(a_ell, c[m][b]), R.mul(ln, c[m - 1][b % Mp]))) ++bad;
        }
    }
    return bad;
}

uint32_t TameMeasure::distribution_failures() const {
    const Ring R = birch::MakeRing(ell, this->R);
    uint32_t bad = 0;
    for (uint32_t m = 1; m + 1 <= M; ++m) {
        const uint64_t Mod = mu[m].size();
        for (uint64_t b = 0; b < Mod; ++b) {
            uint64_t s = 0;
            for (uint64_t t = 0; t < ell; ++t) s = R.add(s, mu[m + 1][b + t * Mod]);
            if (s != mu[m][b]) ++bad;
        }
    }
    return bad;
}

std::vector<std::vector<uint64_t>> TameMeasure::moments(const std::vector<uint64_t>& chibar_mod_T, const std::vector<uint32_t>& chars) const {
    ENGINE_CHECK_MSG(chibar_mod_T.size() == T, "shapiro::TameMeasure::moments: chibar is a table mod T");
    const Ring Rr = birch::MakeRing(ell, R);
    std::vector<std::vector<uint64_t>> wl(chars.size(), std::vector<uint64_t>(ell, 0));
    for (size_t ii = 0; ii < chars.size(); ++ii) {
        const uint32_t e = static_cast<uint32_t>((ell - 1 - chars[ii] % (ell - 1)) % (ell - 1));
        for (uint64_t r = 1; r < ell; ++r) wl[ii][r] = TeichmullerPowMod(r, e, ell, R) % q;
    }
    std::vector<std::vector<uint64_t>> out(chars.size(), std::vector<uint64_t>(n + 1, 0));
    const std::vector<uint64_t>& top = mu[M];
    for (uint64_t b = 0; b < top.size(); ++b) {
        if (b % ell == 0 || chibar_mod_T[b % T] == 0 || top[b] == 0) continue;
        for (size_t ii = 0; ii < chars.size(); ++ii) {
            const uint64_t y = Rr.mul(Rr.mul(chibar_mod_T[b % T], wl[ii][b % ell]), top[b]);
            uint64_t pb = 1 % q;
            const uint64_t bq = b % q;
            for (uint32_t j = 0; j <= n; ++j) {
                out[ii][j] = Rr.add(out[ii][j], Rr.mul(y, pb));
                pb = Rr.mul(pb, bq);
            }
        }
    }
    return out;
}

uint64_t TameMeasure::fnv_units(uint32_t level) const {
    ENGINE_CHECK_MSG(level >= 1 && level <= M, "shapiro::TameMeasure::fnv_units: 1 <= level <= M");
    uint64_t h = engine::kFnvOffset64;
    for (uint64_t b = 1; b < mu[level].size(); ++b)
        if (b % ell != 0 && std::gcd(b, T) == 1) h = engine::FnvU64(h, mu[level][b]);
    return h;
}

MomentIdentityReading TameMomentIdentity(const SymRho& rho, const P1N& p1, const Vec& P, const TameMeasure& me,
                                         const std::vector<uint64_t>& chi, const std::vector<uint32_t>& chars, bool swap_chi) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    const uint64_t T = me.T, ell = me.ell, q = R.q;
    ENGINE_CHECK_MSG(chi.size() == T, "shapiro::TameMomentIdentity: chi is a table mod T");
    std::vector<uint64_t> chibar(T, 0);
    for (uint64_t b = 0; b < T; ++b) chibar[b] = chi[b] ? R.inv(chi[b]) : 0;
    const std::vector<std::vector<uint64_t>> mom = me.moments(chibar, chars);
    const uint64_t prec = pow_wrap(ell, std::min(me.M, me.R));
    const uint64_t ia = R.inv(me.alpha);
    uint64_t chil = chi[ell % T] % q, chibl = chibar[ell % T] % q;
    if (swap_chi) std::swap(chil, chibl);
    MomentIdentityReading out;
    for (size_t ii = 0; ii < chars.size(); ++ii) {
        const uint32_t i = chars[ii];
        birch::DirichletCharacter psi;
        if (i % (ell - 1) != 0) {
            const uint32_t e = static_cast<uint32_t>((ell - 1 - i % (ell - 1)) % (ell - 1));
            psi.M = T * ell;
            psi.chibar.assign(psi.M, 0);
            for (uint64_t b = 0; b < psi.M; ++b)
                if (b % ell != 0 && chibar[b % T] != 0) psi.chibar[b] = R.mul(chibar[b % T], TeichmullerPowMod(b % ell, e, ell, R.m) % q);
        } else {
            psi.M = T;
            psi.chibar = chibar;
        }
        psi.chi = psi.chibar;   // EvAllJN reads chibar only
        const Vec ev = EvAllJN(rho, p1, P, psi);
        for (uint32_t j = 0; j <= n; ++j) {
            uint64_t lhs = R.mul(rho.binom(n, j), mom[ii][j]);
            if (j % 2 == 1) lhs = R.neg(lhs);
            uint64_t E = ia;
            if (i % (ell - 1) == 0)
                E = R.mul(R.sub(1 % q, R.mul(R.mul(chibl, R.pow(ell, j)), ia)), R.sub(1 % q, R.mul(R.mul(chil, R.pow(ell, n - j)), ia)));
            ++out.total;
            if (R.sub(lhs, R.mul(E, ev[j])) % prec != 0) ++out.failures;
        }
    }
    return out;
}

std::vector<uint64_t> TameCharacter(uint64_t T, uint32_t d, uint32_t e, const Ring& R) {
    if (T == 1) return {1 % R.q};
    if (T == 2) {
        ENGINE_CHECK_MSG(d == 1, "shapiro::TameCharacter: mod 2 only the trivial character");
        return {0, 1 % R.q};
    }
    if (d == 1) {
        std::vector<uint64_t> t(T, 1 % R.q);
        t[0] = 0;
        return t;
    }
    return birch::PrimeConductorCharacter(T, d, e, R).chi;
}

// ---- Eichler-Shimura dimensions, eta products -----------------------------------------------------------------------
namespace {
std::vector<std::pair<uint64_t, uint32_t>> Factor(uint64_t N) {
    std::vector<std::pair<uint64_t, uint32_t>> f;
    for (uint64_t p = 2; p * p <= N; ++p)
        if (N % p == 0) {
            uint32_t e = 0;
            while (N % p == 0) {
                N /= p;
                ++e;
            }
            f.push_back({p, e});
        }
    if (N > 1) f.push_back({N, 1});
    return f;
}
uint64_t Phi(uint64_t m) {
    uint64_t r = m;
    for (const auto& pe : Factor(m)) r = r / pe.first * (pe.first - 1);
    return r;
}
struct Gamma0Data {
    int64_t mu = 0, e2 = 0, e3 = 0, c = 0, g = 0;
};
Gamma0Data Data(uint64_t N) {
    Gamma0Data D;
    const auto f = Factor(N);
    D.mu = static_cast<int64_t>(N);
    for (const auto& pe : f) D.mu = D.mu / static_cast<int64_t>(pe.first) * static_cast<int64_t>(pe.first + 1);
    D.e2 = (N % 4 == 0) ? 0 : 1;
    D.e3 = (N % 9 == 0) ? 0 : 1;
    for (const auto& pe : f) {
        const uint64_t p = pe.first;
        if (p == 2) D.e2 *= 1;
        else D.e2 *= (p % 4 == 1) ? 2 : 0;
        if (p == 3) D.e3 *= 1;
        else D.e3 *= (p % 3 == 1) ? 2 : 0;
    }
    for (uint64_t d = 1; d <= N; ++d)
        if (N % d == 0) D.c += static_cast<int64_t>(Phi(std::gcd(d, N / d)));
    const int64_t g12 = 12 + D.mu - 3 * D.e2 - 4 * D.e3 - 6 * D.c;
    ENGINE_CHECK_MSG(g12 % 12 == 0, "shapiro: the genus formula is not integral");
    D.g = g12 / 12;
    return D;
}
} // namespace

uint64_t CuspsGamma0(uint64_t N) { return static_cast<uint64_t>(Data(N).c); }

uint64_t DimCuspFormsGamma0(uint64_t N, uint32_t k) {
    ENGINE_CHECK_MSG(k >= 2 && k % 2 == 0, "shapiro::DimCuspFormsGamma0: k even >= 2");
    const Gamma0Data D = Data(N);
    if (k == 2) return static_cast<uint64_t>(D.g);
    const int64_t kk = k;
    const int64_t d = (kk - 1) * (D.g - 1) + (kk / 2 - 1) * D.c + D.e2 * (kk / 4) + D.e3 * (kk / 3);
    ENGINE_CHECK_MSG(d >= 0, "shapiro::DimCuspFormsGamma0: negative dimension");
    return static_cast<uint64_t>(d);
}

int64_t EtaProductCoefficient(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 2 && 24 % (N + 1) == 0, "shapiro::EtaProductCoefficient: (N + 1) | 24");
    const uint64_t r = 24 / (N + 1);
    const size_t terms = static_cast<size_t>(p + 2);
    auto eta_pow = [&](uint64_t scale) {
        std::vector<int64_t> s(terms, 0);
        s[0] = 1;
        for (uint64_t m = 1; m * scale < terms; ++m)
            for (uint64_t e = 0; e < r; ++e)
                for (size_t i = terms - 1; i >= m * scale; --i) s[i] -= s[i - m * scale];
        return s;
    };
    const std::vector<int64_t> a = eta_pow(1), b = eta_pow(N);
    // eta(z)^r eta(N z)^r = q^{r (N+1) / 24} (prod ...) = q * (a * b)
    std::vector<int64_t> prod(terms, 0);
    for (size_t i = 0; i < terms; ++i)
        if (a[i])
            for (size_t j = 0; i + j < terms; ++j) prod[i + j] += a[i] * b[j];
    return p >= 1 ? prod[static_cast<size_t>(p - 1)] : 0;
}

// (slot 23, G-EVT23) RankOneGenerator -- moved verbatim from HeckeLValueShapiroValidation.cpp -- and the eigen-line reading.
Vec RankOneGenerator(const Gamma0NCell& C) {
    if (C.basis.size() != 1 || !C.free_module) return {};
    Vec P = C.basis[0];
    if (!NormaliseFirstUnit(C.R, P)) return {};
    return P;
}

std::vector<uint64_t> ShapiroLinePrimes(uint64_t N) {
    std::vector<uint64_t> out;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull})
        if (N % p != 0) out.push_back(p);
    ENGINE_CHECK_MSG(out.size() == 5, "shapiro::ShapiroLinePrimes: N must be divisible by exactly one prime p <= 13");
    return out;
}

ShapiroEigenLineReading ShapiroEigenLine(uint64_t N, uint32_t n, uint64_t ell, uint32_t m) {
    ShapiroEigenLineReading r;
    const Gamma0NCell C = RunGamma0NCell(N, n, ell, m, -1);
    r.rankW = static_cast<uint32_t>(C.basis.size());
    r.free_module = C.free_module;
    const Vec P = RankOneGenerator(C);
    r.rank_one = !P.empty();
    if (!r.rank_one) return r;
    const SymRho rho(n, C.R);
    const P1N p1(N);
    r.all_eigen = true;
    for (uint64_t p : ShapiroLinePrimes(N)) {
        uint64_t lam = 0;
        if (!EigenvalueN(rho, p1, P, p, lam)) r.all_eigen = false;
        r.ap.emplace_back(p, lam);
    }
    return r;
}

} // namespace shapiro
} // namespace lvalue
} // namespace hecke_ext
