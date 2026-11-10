// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-10) HeckeNebentypusValidation -- see the header.
// [C, sec:nebentypus]: def:nebdata, lem:teichmuller, lem:inducedchar (eq:MHchi),
// lem:gammaonetable, prop:nebdecomp, prop:nebcorr (eq:nebcorr), thm:nebrelations,
// prop:neborbit, rmk:nebheilbronn, rmk:nebcheckable; [GT, conv:nebentypus],
// [GT, bridge:nebentypus-dictionary].
#include "HeckeNebentypusValidation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "FpPoly.h"
#include "HeckeCosetTables.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"
#include "HeckeNebentypus.h"
#include "HeckeNebentypusGalois.h"   // slot-18 WP-6: the character model over W_K(F_q)
#include "HeckeNebentypusSparse.h"   // slot-17 WP-6: the sparse cokernel with a character, the route
#include "HeckeLevelNSparse.h"       // slot-17 WP-6: CokernelCostModel (the RSS prediction)
#include "../Util/Rss.h"
#include "../../ModArithmetic/Fnv.h"   // slot-17 WP-6: ops_fnv
#include <cstdio>
#include "HeckeOrdinaryPart.h"   // slot-16 WP-11: the torsion pre-pass
#include "HeckeOperatorFamily.h"   // hecke::tau_construction_mutex()
#include "HeckeSettings.h"         // hecke::ScopedCoefficientRing
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../HeckeExt/HeckeCanonicalField.h"
#include "../HeckeExt/HeckeRingHeckeAlgebra.h"   // slot-16 WP-11: the certified Z/ell^K route (validation only)   // [E] def:canon -- the field the chapter fixes (validation only)
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using engine::scalar::gcd;
using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::neb;
using hecke::level::HMat;
using u128 = unsigned __int128;

std::map<std::string, double>& PhaseTimes() { static std::map<std::string, double> t; return t; }
struct PhaseTimer {
    std::string name;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit PhaseTimer(std::string n) : name(std::move(n)) {}
    ~PhaseTimer() { PhaseTimes()[name] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

struct Checker {
    uint32_t checks = 0, fails = 0;
    std::map<std::string, uint32_t> by_arm;
    void check(bool ok, const std::string& what) {
        ++checks;
        by_arm[what.substr(0, what.find(' '))] += 1;
        if (!ok) {
            ++fails;
            if (fails <= 100) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

uint64_t Mod(int64_t x, uint64_t m) { const int64_t r = x % static_cast<int64_t>(m); return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(m) : r); }
bool IsPrime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p = 2; p * p <= n; ++p) if (n % p == 0) return false;
    return true;
}
std::string VecText(const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

// F_q as the chapter fixes it: F_ell[Y]/(C_{ell,d}) of [E] def:canon (rmk:nebfield (1)).
Fq CanonField(uint32_t ell, uint32_t d) {
    if (d == 1) return Fq::Prime(ell);
    return Fq(ell, hecke_ext::CanonicalFieldPolynomial(ell, d));
}
// d(chi): the order of ell modulo ord chi.
uint32_t LeastDegree(uint64_t ell, uint64_t o) {
    if (o <= 1) return 1;
    uint64_t x = ell % o;
    uint32_t d = 1;
    while (x != 1 % o) { x = x * ell % o; ++d; ENGINE_CHECK_MSG(d < 64, "LeastDegree: no degree"); }
    return d;
}

FqMat FromHMat(const HMat& A) {
    ENGINE_CHECK_MSG(A.field, "FromHMat: a field HMat");
    FqMat M(A.rows, A.cols);
    for (size_t i = 0; i < A.a.size(); ++i) M.a[i] = static_cast<uint32_t>(A.a[i]);
    return M;
}
std::vector<uint32_t> ToU32(const std::vector<uint64_t>& v) {
    std::vector<uint32_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = static_cast<uint32_t>(v[i]);
    return o;
}
std::vector<uint32_t> PolyMulFq(const Fq& F, const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.empty() || b.empty()) return {};
    std::vector<uint32_t> c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = F.add(c[i + j], F.mul(a[i], b[j]));
    return c;
}

// ---------------------------------------------------------------------------
// The model and the level-1 data.
// ---------------------------------------------------------------------------
struct Model {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
};
Model& TheModel() {
    static Model m;
    static std::once_flag once;
    std::call_once(once, [] {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        m.K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        m.tess.reset(new hecke::SL2ZTessellationTables(*m.K));
    });
    return m;
}
struct HeckeDatum {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
};
HeckeDatum MakeDatum(uint64_t p, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    HeckeDatum d;
    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
    d.cosets.reset(new hecke::HeckeCosetTables(p, 1, coef, n));
    d.tau.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *d.cosets, coef, n));
    return d;
}

// The character model (or the Gamma_1 model) on the fixed model over a word-sized ring.
struct ModelCell {
    std::unique_ptr<hecke::level::CosetLevel> lvl;
    std::unique_ptr<LocalSystemSimplicialComplex> L;
    std::unique_ptr<EquivariantCohomologyOnK> H;
};
ModelCell BuildModel(std::unique_ptr<hecke::level::CosetLevel> lvl, uint32_t n, const engine::ModCtx<uint32_t>& ctx,
                     const BlockScalar& s) {
    Model& M = TheModel();
    ModelCell c;
    c.lvl = std::move(lvl);
    const uint32_t D = c.lvl->m() * (n + 1);
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    if (!ctx.is_field()) ring.reset(new hecke::ScopedCoefficientRing(ctx.k));
    c.L.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsChi(*M.K, *c.lvl, n, ctx, s), D, ctx.p));
    c.H.reset(new EquivariantCohomologyOnK(c.L.get()));
    c.H->compute_cohomology();
    return c;
}

// T_p (p !| N) / U_p (p | N) on the character model via eq:nebcorr, the pinned r_t of def:cosetrepsN (6).
HMat ModelHeckeChi(const ModelCell& c, const hecke::level::Gamma0LevelN& lvl, uint32_t p, uint32_t n,
                   const engine::ModCtx<uint32_t>& ctx, const BlockScalar& s, const HeckeDatum& dat) {
    const hecke::level::Gamma0LevelN lvlNp(lvl.N() * p);
    hecke::level::CorrespondenceSpec spec{c.lvl.get(), c.lvl.get(), dat.cosets.get(), dat.tau.get(),
                                          hecke::level::Gamma0SubTransversalN(lvl, lvlNp),
                                          std::string(lvl.N() % p ? "T_" : "U_") + std::to_string(p)};
    hecke::level::VerifyTransversal(spec, lvl.N() % p ? p + 1 : p);
    const hecke::level::HFrame f(c.H.get(), 1);
    return hecke::level::InducedOnH(f, f, BuildLevelCorrespondenceChi(ctx, spec, *TheModel().K, n, 1, s), spec.label);
}

// ---------------------------------------------------------------------------
// (G) an independent group-cocycle route WITH A CHARACTER over F_P (P = ell).
// The slot-14 GroupRouteN pattern (HeckeLevelNValidation.cpp: own P^1(Z/N) table by
// the lexicographically least unit multiple, own lifts, own transversals; no tau,
// no model, no def:cosetrepsN code), every rho evaluated as rho_{n,chi}(g) =
// chi(a_g) rho_n(g) (def:nebdata (5)) -- it is only ever evaluated on Delta_0(N).
// ---------------------------------------------------------------------------
class GroupRouteChi {
public:
    GroupRouteChi(uint32_t n, uint64_t N, uint64_t P, std::vector<uint64_t> chi_table)
        : P_(P), N_(N), n_(n), d_(n + 1), chi_(std::move(chi_table)) {
        std::vector<uint64_t> units;
        for (uint64_t u = 1; u <= N; ++u) if (gcd<uint64_t>(u, N) == 1) units.push_back(u % N);
        if (N == 1) units = {0};
        cls_.assign(static_cast<size_t>(N) * N, 0);
        auto norm = [&](uint64_t c, uint64_t d) {
            std::pair<uint64_t, uint64_t> best{N, N};
            for (uint64_t u : units) best = std::min(best, std::make_pair(u * c % N, u * d % N));
            if (N == 1) best = {0, 0};
            return best;
        };
        std::map<std::pair<uint64_t, uint64_t>, uint32_t> idx;
        auto add = [&](uint64_t c, uint64_t d) {
            const auto k = norm(c, d);
            if (idx.count(k)) return;
            idx[k] = static_cast<uint32_t>(reps_.size() + 1);
            const int64_t cc = static_cast<int64_t>(c), dd = static_cast<int64_t>(d);
            bool done = false;
            for (int64_t s = 0; s < 64 && !done; ++s)
                for (int64_t t = 0; t < 64 && !done; ++t) {
                    const int64_t c2 = cc + s * static_cast<int64_t>(N), d2 = dd + t * static_cast<int64_t>(N);
                    if (gcd<uint64_t>(static_cast<uint64_t>(std::llabs(c2)), static_cast<uint64_t>(std::llabs(d2))) != 1) continue;
                    int64_t x0 = 1, y0 = 0, x1 = 0, y1 = 1, a = d2, b = c2;
                    while (b) { const int64_t q = a / b; int64_t tmp = a - q * b; a = b; b = tmp; tmp = x0 - q * x1; x0 = x1; x1 = tmp; tmp = y0 - q * y1; y0 = y1; y1 = tmp; }
                    if (a < 0) { x0 = -x0; y0 = -y0; }
                    reps_.push_back(Mat22{x0, -y0, c2, d2});
                    done = true;
                }
            ENGINE_CHECK_MSG(done, "GroupRouteChi: no lift");
        };
        add(0, 1 % std::max<uint64_t>(N, 1));
        for (uint64_t c = 0; c < N; ++c)
            for (uint64_t d = 0; d < N; ++d)
                if (gcd<uint64_t>(gcd<uint64_t>(c, d), N) == 1) add(c, d);
        for (uint64_t c = 0; c < N; ++c)
            for (uint64_t d = 0; d < N; ++d)
                if (gcd<uint64_t>(gcd<uint64_t>(c, d), N) == 1) cls_[c * N + d] = idx.at(norm(c, d));
        if (N == 1) { reps_ = {Mat22::I()}; cls_ = {1}; }
        m_ = static_cast<uint32_t>(reps_.size());
        D_ = m_ * d_;
        for (auto& r : reps_) inv_.push_back(r.inv_unimodular());
    }
    uint32_t cls(const Mat22& x) const { return N_ == 1 ? 1 : cls_[Mod(x.c, N_) * N_ + Mod(x.d, N_)]; }
    bool inH(const Mat22& x) const { return x.det() == 1 && Mod(x.c, N_) == 0; }
    Mat22 w(uint32_t j) const { return reps_[j - 1]; }
    // rho_{n,chi}(g) = chi(a_g) rho_n(g) mod P, g in Delta_0(N).
    const std::vector<uint64_t>& rho(const Mat22& g) {
        const uint64_t a = N_ == 1 ? 0 : Mod(g.a, N_);
        ENGINE_CHECK_MSG(N_ == 1 || gcd<uint64_t>(a, N_) == 1, "GroupRouteChi: rho evaluated outside Delta_0(N)");
        const std::array<uint32_t, 4> red = g.reduced_mod(static_cast<uint32_t>(P_));
        const std::array<uint64_t, 5> key{red[0], red[1], red[2], red[3], a};
        auto it = rho_.find(key);
        if (it != rho_.end()) return it->second;
        const FlatMatrixT<uint32_t> A(2, 2, static_cast<uint32_t>(P_), std::vector<uint32_t>{red[0], red[1], red[2], red[3]});
        const FlatMatrixT<uint32_t> R = ComputeSymmetricPowerMatrixFM(A, n_);
        const uint64_t s = N_ == 1 ? 1 : chi_[a];
        std::vector<uint64_t> v(static_cast<size_t>(d_) * d_);
        for (uint32_t r = 0; r < d_; ++r)
            for (uint32_t c = 0; c < d_; ++c) v[static_cast<size_t>(r) * d_ + c] = R.get_entry(r, c) * s % P_;
        return rho_.emplace(key, std::move(v)).first->second;
    }
    std::vector<uint64_t> RightMul(const std::vector<uint64_t>& R, const Mat22& gamma) {
        std::vector<uint64_t> out(static_cast<size_t>(d_) * D_, 0);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22 x = w(i) * gamma;
            const uint32_t j = cls(x);
            const std::vector<uint64_t>& B = rho(x * inv_[j - 1]);
            for (uint32_t r = 0; r < d_; ++r)
                for (uint32_t k = 0; k < d_; ++k) {
                    const uint64_t a = R[static_cast<size_t>(r) * D_ + (i - 1) * d_ + k];
                    if (a == 0) continue;
                    for (uint32_t c = 0; c < d_; ++c) {
                        uint64_t& o = out[static_cast<size_t>(r) * D_ + (j - 1) * d_ + c];
                        o = (o + a * B[static_cast<size_t>(k) * d_ + c]) % P_;
                    }
                }
        }
        return out;
    }
    std::vector<uint64_t> Dense(const Mat22& gamma) {
        std::vector<uint64_t> out(static_cast<size_t>(D_) * D_, 0);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22 x = w(i) * gamma;
            const uint32_t j = cls(x);
            const std::vector<uint64_t>& B = rho(x * inv_[j - 1]);
            for (uint32_t r = 0; r < d_; ++r)
                for (uint32_t c = 0; c < d_; ++c) out[static_cast<size_t>((i - 1) * d_ + r) * D_ + (j - 1) * d_ + c] = B[static_cast<size_t>(r) * d_ + c];
        }
        return out;
    }
    struct Letter { bool S; int64_t k; };
    static Mat22 Tpow(int64_t k) { return Mat22{1, k, 0, 1}; }
    std::vector<Letter> Decompose(const Mat22& g0, bool& ok) const {
        std::vector<Letter> L;
        Mat22 x = g0;
        while (x.c != 0) {
            int64_t k = x.a / x.c;
            if ((x.a % x.c != 0) && ((x.a < 0) != (x.c < 0))) --k;
            if (k != 0) { L.push_back({false, k}); x = Tpow(-k) * x; }
            L.push_back({true, 0});
            x = Mat22{0, 1, -1, 0} * x;
        }
        if (x.a == 1) { if (x.b) L.push_back({false, x.b}); }
        else { if (x.b) L.push_back({false, -x.b}); }
        Mat22 prod = Mat22::I();
        for (const Letter& l : L) prod = prod * (l.S ? Mat22::S() : Tpow(l.k));
        ok = (prod == g0) || (prod == g0.neg());
        return L;
    }
    std::vector<uint64_t> SumT(const std::vector<uint64_t>& R, int64_t k) {
        if (k == 0) return std::vector<uint64_t>(R.size(), 0);
        if (k == 1) return R;
        if (k % 2 == 0) {
            std::vector<uint64_t> X = SumT(R, k / 2);
            const std::vector<uint64_t> Y = RightMul(X, Tpow(k / 2));
            for (size_t e = 0; e < X.size(); ++e) X[e] = (X[e] + Y[e]) % P_;
            return X;
        }
        std::vector<uint64_t> X = SumT(R, k - 1);
        const std::vector<uint64_t> Y = RightMul(R, Tpow(k - 1));
        for (size_t e = 0; e < X.size(); ++e) X[e] = (X[e] + Y[e]) % P_;
        return X;
    }
    struct Eval { std::vector<uint64_t> Ea, Eb; };
    const Eval& E(const Mat22& x) {
        const std::array<int64_t, 4> key{x.a, x.b, x.c, x.d};
        auto it = eval_.find(key);
        if (it != eval_.end()) return it->second;
        bool ok = false;
        const std::vector<Letter> L = Decompose(x, ok);
        ENGINE_CHECK_MSG(ok, "GroupRouteChi: word decomposition failed");
        std::vector<uint64_t> R(static_cast<size_t>(d_) * D_, 0);
        for (uint32_t r = 0; r < d_; ++r) R[static_cast<size_t>(r) * D_ + r] = 1;
        Eval ev{std::vector<uint64_t>(R.size(), 0), std::vector<uint64_t>(R.size(), 0)};
        auto add_FT = [&](const std::vector<uint64_t>& Q, bool neg) {
            const std::vector<uint64_t> QS = RightMul(Q, Mat22::S());
            for (size_t e = 0; e < Q.size(); ++e) {
                ev.Ea[e] = neg ? (ev.Ea[e] + P_ - Q[e]) % P_ : (ev.Ea[e] + Q[e]) % P_;
                ev.Eb[e] = neg ? (ev.Eb[e] + P_ - QS[e]) % P_ : (ev.Eb[e] + QS[e]) % P_;
            }
        };
        for (const Letter& l : L) {
            if (l.S) {
                for (size_t e = 0; e < R.size(); ++e) ev.Ea[e] = (ev.Ea[e] + R[e]) % P_;
                R = RightMul(R, Mat22::S());
            } else if (l.k > 0) {
                add_FT(SumT(R, l.k), false);
                R = RightMul(R, Tpow(l.k));
            } else {
                const std::vector<uint64_t> Rk = RightMul(R, Tpow(l.k));
                add_FT(SumT(Rk, -l.k), true);
                R = Rk;
            }
        }
        return eval_.emplace(key, std::move(ev)).first->second;
    }
    std::vector<uint64_t> f(const Mat22& x, const std::vector<uint64_t>& z) {
        const Eval& ev = E(x);
        std::vector<uint64_t> out(d_, 0);
        for (uint32_t r = 0; r < d_; ++r) {
            u128 acc = 0;
            for (uint32_t c = 0; c < D_; ++c)
                acc += static_cast<u128>(ev.Ea[static_cast<size_t>(r) * D_ + c]) * z[c] +
                       static_cast<u128>(ev.Eb[static_cast<size_t>(r) * D_ + c]) * z[D_ + c];
            out[r] = static_cast<uint64_t>(acc % P_);
        }
        return out;
    }
    // Field helpers on dense vectors-of-columns.
    uint32_t RankCols(const std::vector<std::vector<uint64_t>>& cs, uint32_t len) const {
        std::vector<std::vector<uint64_t>> M(len, std::vector<uint64_t>(cs.size(), 0));
        for (size_t j = 0; j < cs.size(); ++j) for (uint32_t i = 0; i < len; ++i) M[i][j] = cs[j][i];
        return RankRows(M);
    }
    uint32_t RankRows(std::vector<std::vector<uint64_t>> M) const {
        uint32_t rk = 0;
        if (M.empty()) return 0;
        const size_t C = M[0].size();
        for (size_t c = 0; c < C && rk < M.size(); ++c) {
            size_t p = rk;
            while (p < M.size() && M[p][c] == 0) ++p;
            if (p == M.size()) continue;
            std::swap(M[p], M[rk]);
            const uint64_t iv = powmod<uint64_t>(M[rk][c], P_ - 2, P_);
            for (uint64_t& x : M[rk]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % P_);
            for (size_t i = 0; i < M.size(); ++i) {
                if (i == rk || M[i][c] == 0) continue;
                const uint64_t fct = M[i][c];
                for (size_t j = 0; j < C; ++j) M[i][j] = (M[i][j] + P_ - static_cast<uint64_t>(static_cast<u128>(fct) * M[rk][j] % P_)) % P_;
            }
            ++rk;
        }
        return rk;
    }
    // Kernel basis of the D x D matrix A (columns).
    std::vector<std::vector<uint64_t>> Kernel(std::vector<std::vector<uint64_t>> M) const {
        const size_t R = M.size(), C = R ? M[0].size() : 0;
        std::vector<int64_t> pivcol;
        size_t rk = 0;
        for (size_t c = 0; c < C && rk < R; ++c) {
            size_t p = rk;
            while (p < R && M[p][c] == 0) ++p;
            if (p == R) continue;
            std::swap(M[p], M[rk]);
            const uint64_t iv = powmod<uint64_t>(M[rk][c], P_ - 2, P_);
            for (uint64_t& x : M[rk]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % P_);
            for (size_t i = 0; i < R; ++i) {
                if (i == rk || M[i][c] == 0) continue;
                const uint64_t fct = M[i][c];
                for (size_t j = 0; j < C; ++j) M[i][j] = (M[i][j] + P_ - static_cast<uint64_t>(static_cast<u128>(fct) * M[rk][j] % P_)) % P_;
            }
            pivcol.push_back(static_cast<int64_t>(c));
            ++rk;
        }
        std::vector<uint8_t> isp(C, 0);
        for (int64_t c : pivcol) isp[c] = 1;
        std::vector<std::vector<uint64_t>> out;
        for (size_t fcol = 0; fcol < C; ++fcol) {
            if (isp[fcol]) continue;
            std::vector<uint64_t> v(C, 0);
            v[fcol] = 1;
            for (size_t r = 0; r < pivcol.size(); ++r) v[pivcol[r]] = (P_ - M[r][fcol]) % P_;
            out.push_back(v);
        }
        return out;
    }
    void Setup() {
        const std::vector<uint64_t> MS = Dense(Mat22::S()), MU = Dense(Mat22::ST());
        // MU^2
        std::vector<uint64_t> MU2(static_cast<size_t>(D_) * D_, 0);
        for (uint32_t i = 0; i < D_; ++i)
            for (uint32_t k = 0; k < D_; ++k) {
                const uint64_t a = MU[static_cast<size_t>(i) * D_ + k];
                if (!a) continue;
                for (uint32_t j = 0; j < D_; ++j) {
                    uint64_t& o = MU2[static_cast<size_t>(i) * D_ + j];
                    o = (o + a * MU[static_cast<size_t>(k) * D_ + j]) % P_;
                }
            }
        std::vector<std::vector<uint64_t>> AS(D_, std::vector<uint64_t>(D_)), AU(D_, std::vector<uint64_t>(D_));
        for (uint32_t i = 0; i < D_; ++i)
            for (uint32_t j = 0; j < D_; ++j) {
                AS[i][j] = (MS[static_cast<size_t>(i) * D_ + j] + (i == j ? 1 : 0)) % P_;
                AU[i][j] = (MU[static_cast<size_t>(i) * D_ + j] + MU2[static_cast<size_t>(i) * D_ + j] + (i == j ? 1 : 0)) % P_;
            }
        for (const auto& v : Kernel(AS)) { std::vector<uint64_t> z(2 * D_, 0); for (uint32_t i = 0; i < D_; ++i) z[i] = v[i]; Z1_.push_back(z); }
        for (const auto& v : Kernel(AU)) { std::vector<uint64_t> z(2 * D_, 0); for (uint32_t i = 0; i < D_; ++i) z[D_ + i] = v[i]; Z1_.push_back(z); }
        std::vector<std::vector<uint64_t>> B1(2 * D_, std::vector<uint64_t>(D_));
        for (uint32_t i = 0; i < D_; ++i)
            for (uint32_t j = 0; j < D_; ++j) {
                B1[i][j] = (MS[static_cast<size_t>(i) * D_ + j] + P_ - (i == j ? 1 : 0)) % P_;
                B1[D_ + i][j] = (MU[static_cast<size_t>(i) * D_ + j] + P_ - (i == j ? 1 : 0)) % P_;
            }
        const uint32_t rankB1 = RankRows(B1);
        h0_ = D_ - rankB1;
        h1_ = static_cast<uint32_t>(Z1_.size()) - rankB1;
        for (uint32_t j = 1; j <= m_; ++j)
            for (const Mat22& x : {Mat22::S(), Mat22::T()}) {
                const Mat22 y = w(j) * x;
                const Mat22 s = y * inv_[cls(y) - 1];
                ENGINE_CHECK(inH(s));
                if (s == Mat22::I() || s == Mat22::I().neg()) continue;
                gens_.push_back(s);
            }
        len_ = static_cast<uint32_t>(gens_.size()) * d_;
        std::vector<std::vector<uint64_t>> cols;
        for (uint32_t r = 0; r < d_; ++r) {
            std::vector<uint64_t> v(len_, 0);
            for (size_t s = 0; s < gens_.size(); ++s) {
                const auto& rs = rho(gens_[s]);
                for (uint32_t a = 0; a < d_; ++a) v[s * d_ + a] = (rs[static_cast<size_t>(a) * d_ + r] + P_ - (a == r ? 1 : 0)) % P_;
            }
            cols.push_back(v);
        }
        for (const auto& v : cols) {
            bh_.push_back(v);
            if (RankCols(bh_, len_) < bh_.size()) bh_.pop_back();
        }
        std::vector<std::vector<uint64_t>> all = bh_;
        uint32_t base = static_cast<uint32_t>(bh_.size());
        for (const auto& z : Z1_) {
            all.push_back(Stack(z));
            if (RankCols(all, len_) > base) { reps_z_.push_back(z); ++base; }
            else all.pop_back();
        }
        basis_cols_ = all;
    }
    uint32_t h1() const { return h1_; }
    uint32_t h0() const { return h0_; }
    uint32_t restricted() const { return static_cast<uint32_t>(reps_z_.size()); }
    std::vector<uint64_t> Stack(const std::vector<uint64_t>& z) {
        std::vector<uint64_t> v(static_cast<size_t>(gens_.size()) * d_, 0);
        for (size_t s = 0; s < gens_.size(); ++s) {
            const auto fs = f(gens_[s], z);
            std::copy(fs.begin(), fs.end(), v.begin() + s * d_);
        }
        return v;
    }
    std::vector<Mat22> Transversal(uint64_t p) const {
        std::vector<Mat22> r;
        for (int64_t t = 0; t < static_cast<int64_t>(p); ++t) r.push_back(Mat22{1, 0, static_cast<int64_t>(N_) * t, 1});
        if (N_ % p != 0)
            for (int64_t x = 1; x < 1000000; ++x)
                if ((x * static_cast<int64_t>(p) - 1) % static_cast<int64_t>(N_) == 0) {
                    r.push_back(Mat22{x, (x * static_cast<int64_t>(p) - 1) / static_cast<int64_t>(N_), static_cast<int64_t>(N_), static_cast<int64_t>(p)});
                    break;
                }
        return r;
    }
    // [H diag(1, p) H] on H^1 in the restricted-cocycle basis (h x h over F_P).
    FqMat Operator(uint64_t p) {
        const Mat22 g{1, 0, 0, static_cast<int64_t>(p)};
        const std::vector<Mat22> r = Transversal(p);
        auto in_Bpp = [&](const Mat22& x) {
            if (!inH(x)) return false;
            const int64_t D = g.det();
            const Mat22 M = g.adj() * x * g;
            if (M.a % D || M.b % D || M.c % D || M.d % D) return false;
            return inH(Mat22{M.a / D, M.b / D, M.c / D, M.d / D});
        };
        const uint32_t nb = static_cast<uint32_t>(bh_.size()), h = static_cast<uint32_t>(reps_z_.size());
        std::vector<std::vector<uint64_t>> rhs;
        for (const auto& z : reps_z_) {
            std::vector<uint64_t> v(len_, 0);
            for (size_t s = 0; s < gens_.size(); ++s)
                for (const Mat22& rt : r) {
                    const Mat22 x = rt * gens_[s];
                    int found = -1;
                    for (size_t u = 0; u < r.size(); ++u)
                        if (in_Bpp(x * r[u].inv_unimodular())) { found = static_cast<int>(u); break; }
                    ENGINE_CHECK_MSG(found >= 0, "GroupRouteChi: t . h not found");
                    const Mat22 bt = x * r[found].inv_unimodular();
                    const int64_t D = g.det();
                    const Mat22 M = g.adj() * bt * g;
                    const auto fy = f(Mat22{M.a / D, M.b / D, M.c / D, M.d / D}, z);
                    const auto& R = rho(rt.inv_unimodular() * g);
                    for (uint32_t a = 0; a < d_; ++a) {
                        u128 acc = 0;
                        for (uint32_t c = 0; c < d_; ++c) acc += static_cast<u128>(R[static_cast<size_t>(a) * d_ + c]) * fy[c];
                        v[s * d_ + a] = (v[s * d_ + a] + static_cast<uint64_t>(acc % P_)) % P_;
                    }
                }
            rhs.push_back(v);
        }
        const uint32_t C = static_cast<uint32_t>(basis_cols_.size()), R = static_cast<uint32_t>(rhs.size());
        std::vector<std::vector<uint64_t>> M(len_, std::vector<uint64_t>(C + R, 0));
        for (uint32_t i = 0; i < len_; ++i) {
            for (uint32_t j = 0; j < C; ++j) M[i][j] = basis_cols_[j][i];
            for (uint32_t j = 0; j < R; ++j) M[i][C + j] = rhs[j][i];
        }
        uint32_t rk = 0;
        for (uint32_t c = 0; c < C; ++c) {
            uint32_t piv = rk;
            while (piv < len_ && M[piv][c] == 0) ++piv;
            ENGINE_CHECK_MSG(piv < len_, "GroupRouteChi: the basis columns are dependent");
            std::swap(M[piv], M[rk]);
            const uint64_t iv = powmod<uint64_t>(M[rk][c], P_ - 2, P_);
            for (uint64_t& x : M[rk]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % P_);
            for (uint32_t i = 0; i < len_; ++i) {
                if (i == rk || M[i][c] == 0) continue;
                const uint64_t fct = M[i][c];
                for (uint32_t j = 0; j < C + R; ++j) M[i][j] = (M[i][j] + P_ - static_cast<uint64_t>(static_cast<u128>(fct) * M[rk][j] % P_)) % P_;
            }
            ++rk;
        }
        for (uint32_t i = C; i < len_; ++i)
            for (uint32_t j = 0; j < R; ++j) ENGINE_CHECK_MSG(M[i][C + j] == 0, "GroupRouteChi: an image is not a cocycle class");
        FqMat Op(h, h);
        for (uint32_t i = 0; i < h; ++i)
            for (uint32_t j = 0; j < h; ++j) Op.at(i, j) = static_cast<uint32_t>(M[nb + i][C + j]);
        return Op;
    }

private:
    uint64_t P_, N_;
    uint32_t n_, d_, m_ = 0, D_ = 0;
    std::vector<uint64_t> chi_;
    std::vector<Mat22> reps_, inv_;
    std::vector<uint32_t> cls_;
    std::map<std::array<uint64_t, 5>, std::vector<uint64_t>> rho_;
    std::map<std::array<int64_t, 4>, Eval> eval_;
    std::vector<std::vector<uint64_t>> Z1_, reps_z_, bh_, basis_cols_;
    std::vector<Mat22> gens_;
    uint32_t h0_ = 0, h1_ = 0, len_ = 0;
};

// ---------------------------------------------------------------------------
// The repository root and WP-9's table.
// ---------------------------------------------------------------------------
bool PathExists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
const char* kTsv = "/docs/implementation/slot16/WP-9_nb_predictions.tsv";
std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    { const std::string r(OC_SOURCE_DIR); if (!r.empty() && PathExists(r + kTsv)) return r; }
#endif
    char buf[4096];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (PathExists(dir + kTsv)) return dir;
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}
std::vector<std::string> SplitStr(const std::string& s, char c) {
    std::vector<std::string> v;
    std::string cur;
    for (char x : s) { if (x == c) { v.push_back(cur); cur.clear(); } else cur.push_back(x); }
    v.push_back(cur);
    return v;
}
std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
struct TsvRow { std::string cell, kind, key, value, ell, note; };
std::vector<TsvRow> ReadTsv(const std::string& path) {
    std::vector<TsvRow> rows;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = SplitStr(line, '\t');
        if (f.size() < 8 || f[0] == "cell") continue;
        rows.push_back(TsvRow{f[0], f[1], f[2], f[3], f[4], f[7]});
    }
    return rows;
}
// "(7,chi_7.2,4)" -> N, k
bool ParseCell(const std::string& cell, uint64_t& N, uint32_t& k) {
    if (cell.size() < 5 || cell.front() != '(' || cell.back() != ')') return false;
    const std::vector<std::string> f = SplitStr(cell.substr(1, cell.size() - 2), ',');
    if (f.size() != 3) return false;
    N = std::stoull(f[0]);
    k = static_cast<uint32_t>(std::stoul(f[2]));
    return true;
}
std::vector<uint32_t> ParseCoeffs(const std::string& v) {
    const size_t eq = v.find('=');
    std::vector<uint32_t> out;
    for (const std::string& t : SplitStr(v.substr(eq == std::string::npos ? 0 : eq + 1), ','))
        if (!Trim(t).empty()) out.push_back(static_cast<uint32_t>(std::stoul(Trim(t))));
    return out;
}

// ---------------------------------------------------------------------------
// (F) the field, the characters, the reduction rule, the Teichmuller lift.
// ---------------------------------------------------------------------------
void ArmField(Checker& g) {
    PhaseTimer pt("(F) field / characters / Teichmuller");
    // F_q against [E] def:canon's field, entrywise on every product (q <= 169).
    for (const auto& ld : std::vector<std::pair<uint32_t, uint32_t>>{{5, 2}, {7, 2}, {13, 2}, {5, 3}, {3, 3}, {13, 1}}) {
        const Fq F = CanonField(ld.first, ld.second);
        const hecke_ext::CanonicalField C(ld.first, ld.second);
        g.check(F.defining_poly() == C.defining_poly() && C.is_canonical(),
                "(F) " + F.name() + ": the defining polynomial is [E] def:canon's C_{ell,d}");
        bool mul_ok = true, add_ok = true, inv_ok = true, less_ok = true;
        auto toE = [&](uint64_t code) {
            hecke_ext::FieldElt e(ld.second, 0);
            const std::vector<uint32_t> dg = F.digits(static_cast<uint32_t>(code));
            for (uint32_t i = 0; i < ld.second; ++i) e[i] = dg[i];
            return e;
        };
        for (uint64_t a = 0; a < F.q(); ++a) {
            const hecke_ext::FieldElt ea = toE(a);
            if (a) inv_ok = inv_ok && F.mul(static_cast<uint32_t>(a), F.inv(static_cast<uint32_t>(a))) == 1 &&
                             toE(F.inv(static_cast<uint32_t>(a))) == C.inv(ea);
            for (uint64_t b = 0; b < F.q(); ++b) {
                const hecke_ext::FieldElt eb = toE(b);
                mul_ok = mul_ok && toE(F.mul(static_cast<uint32_t>(a), static_cast<uint32_t>(b))) == C.mul(ea, eb);
                add_ok = add_ok && toE(F.add(static_cast<uint32_t>(a), static_cast<uint32_t>(b))) == C.add(ea, eb);
                less_ok = less_ok && F.less(static_cast<uint32_t>(a), static_cast<uint32_t>(b)) == hecke_ext::CanonicalField::less(ea, eb);
            }
        }
        g.check(mul_ok && add_ok && inv_ok, "(F) " + F.name() + ": +, x, ^-1 equal HeckeCanonicalField's on all q^2 pairs");
        g.check(less_ok, "(F) " + F.name() + ": the order is CanonicalLess (HeckeCanonicalField::less)");
        // the least element of exact order o, against a direct scan in CanonicalLess order.
        for (uint64_t o = 2; o <= F.q() - 1; ++o) {
            if ((F.q() - 1) % o) continue;
            std::vector<uint32_t> all;
            for (uint64_t x = 1; x < F.q(); ++x) if (F.order_of(static_cast<uint32_t>(x)) == o) all.push_back(static_cast<uint32_t>(x));
            uint32_t best = all[0];
            for (uint32_t x : all) if (F.less(x, best)) best = x;
            if (!(F.least_root_of_unity(o) == best))
                g.check(false, "(F) " + F.name() + ": least root of unity of order " + std::to_string(o));
        }
        g.check(true, "(F) " + F.name() + ": least_root_of_unity = the scan, every o | q - 1");
    }
    // Conrey characters: the WP-9 / WP-9b labels.
    struct Lab { uint64_t N, m, order; int parity; uint64_t g; uint64_t num; };   // chi(g) = e(num / order)
    for (const Lab& l : std::vector<Lab>{{5, 4, 2, 1, 2, 1}, {13, 12, 2, 1, 2, 1}, {7, 6, 2, -1, 3, 1}, {4, 3, 2, -1, 3, 1},
                                         {3, 2, 2, -1, 2, 1}, {7, 2, 3, 1, 3, 1}, {9, 2, 6, -1, 2, 1}, {13, 8, 4, -1, 2, 1},
                                         {7, 4, 3, 1, 3, 2}, {9, 5, 6, -1, 2, 5}, {13, 5, 4, -1, 2, 3}}) {
        const DirichletCharacter c = DirichletCharacter::Conrey(l.N, l.m);
        const int64_t e = c.exponent_at(static_cast<int64_t>(l.g));
        const bool val_ok = static_cast<uint64_t>(e) * l.order == l.num * c.L;
        g.check(c.order() == l.order && c.parity() == l.parity && val_ok,
                "(F) Conrey " + c.label() + ": order " + std::to_string(l.order) + ", parity " + std::to_string(l.parity) +
                    ", chi(" + std::to_string(l.g) + ") = e(" + std::to_string(l.num) + "/" + std::to_string(l.order) + ")");
    }
    g.check(DirichletCharacter::Conrey(7, 2).inverse().label() == "7.4" && DirichletCharacter::Conrey(13, 8).inverse().label() == "13.5" &&
                DirichletCharacter::Conrey(9, 2).inverse().label() == "9.5",
            "(F) chi -> chibar on the WP-9b labels: 7.2 -> 7.4, 9.2 -> 9.5, 13.8 -> 13.5 (the TSV's conj_conrey_label rows)");
    {
        // S16 WP-10/11 repairs (audit finding 10): the Conrey generator at e = 1 is LMFDB's "least positive integer which is
        // a primitive root for all p^e", also where it is not the least primitive root mod p. At p = 40487: 5 is the least
        // primitive root mod p but 5^(p-1) = 1 mod p^2, so g_p = 10; then 40487.10 (10) = e(1/phi), and 40487.5 (5) != e(1/phi).
        const uint64_t P = 40487, P2 = P * P;
        const DirichletCharacter c10 = DirichletCharacter::Conrey(P, 10), c5 = DirichletCharacter::Conrey(P, 5);
        g.check(powmod<uint64_t>(5, P - 1, P2) == 1 && P - 1 == 2 * 31 * 653 && powmod<uint64_t>(5, (P - 1) / 2, P) != 1 &&
                    powmod<uint64_t>(5, (P - 1) / 31, P) != 1 && powmod<uint64_t>(5, (P - 1) / 653, P) != 1 && c10.L == P - 1 && c10.exponent_at(10) == 1 &&
                    c5.exponent_at(5) != 1,
                "(F) Conrey at p = 40487: g_p = 10 (5 is primitive mod p, not mod p^2), the LMFDB convention: 40487.10(10) = e(1/40486)");
    }
    // The character group mod N: phi(N) distinct characters, closed under products, orthogonality over F_q.
    for (uint64_t N : {5ull, 7ull, 8ull, 9ull, 12ull, 13ull, 15ull, 16ull, 20ull, 21ull, 24ull}) {
        const std::vector<DirichletCharacter> all = AllCharacters(N);
        std::set<std::vector<int64_t>> seen;
        uint64_t lam = 1;
        for (const auto& c : all) {
            std::vector<int64_t> norm(c.expo.size());   // normalised to denominator lcm 720720
            for (size_t i = 0; i < c.expo.size(); ++i) norm[i] = c.expo[i] < 0 ? -1 : c.expo[i] * (720720 / static_cast<int64_t>(c.L));
            seen.insert(norm);
            lam = lam / gcd<uint64_t>(lam, c.order()) * c.order();
        }
        bool closed = true;
        for (size_t i = 0; i < all.size(); ++i)
            for (size_t j = 0; j < all.size(); ++j) {
                const DirichletCharacter p = all[i].mul(all[j]);
                const DirichletCharacter q = DirichletCharacter::Conrey(N, p.conrey);
                for (uint64_t u : UnitsMod(N))
                    closed = closed && static_cast<uint64_t>(p.exponent_at(static_cast<int64_t>(u))) * q.L ==
                                           static_cast<uint64_t>(q.exponent_at(static_cast<int64_t>(u))) * p.L;
            }
        g.check(seen.size() == all.size() && all.size() == UnitsMod(N).size() && closed,
                "(F) characters mod " + std::to_string(N) + ": phi(N) distinct, chi_N(m) chi_N(m') = chi_N(m m') (Conrey)");
        // orthogonality over the least F_ell with lambda(N) | ell - 1.
        uint64_t ell = 5;
        while (!(IsPrime(ell) && N % ell != 0 && (ell - 1) % lam == 0)) ++ell;
        const Fq F = Fq::Prime(static_cast<uint32_t>(ell));
        bool orth = true;
        for (const auto& c : all) {
            const CharacterValues v = ReduceCharacter(c, F);
            uint32_t s = 0;
            for (uint64_t u : UnitsMod(N)) s = F.add(s, v.at(static_cast<int64_t>(u)));
            orth = orth && (c.is_trivial() ? s == F.from_int(static_cast<int64_t>(UnitsMod(N).size())) : s == 0);
        }
        g.check(orth, "(F) characters mod " + std::to_string(N) + ": sum_u psi(u) = phi(N) [psi = 1] over F_" + std::to_string(ell));
    }
    // Teichmuller (lem:teichmuller (1)-(3)).
    for (const auto& c : std::vector<std::array<uint32_t, 3>>{{13, 1, 3}, {7, 2, 2}, {5, 2, 3}, {11, 2, 2}}) {
        const Fq F = CanonField(c[0], c[1]);
        const GaloisRing W(F, c[2]);
        bool ok = true;
        std::vector<GaloisRing::W> t(F.q());
        for (uint64_t x = 1; x < F.q(); ++x) {
            t[x] = W.teichmuller(static_cast<uint32_t>(x));
            ok = ok && W.pow(t[x], F.q() - 1) == W.one() && W.reduce(t[x]) == x;
            // independent of the lift: (lift(x) + ell * Y^0) ^ {q^{K-1}}
            GaloisRing::W y = W.add(W.lift(static_cast<uint32_t>(x)), W.scalar(F.ell()));
            for (uint32_t i = 1; i < c[2]; ++i) y = W.pow(y, F.q());
            ok = ok && y == t[x];
        }
        for (uint64_t x = 1; x < F.q(); x += 3)
            for (uint64_t y = 1; y < F.q(); y += 5)
                ok = ok && W.mul(t[x], t[y]) == t[F.mul(static_cast<uint32_t>(x), static_cast<uint32_t>(y))];
        g.check(ok, "(F) lem:teichmuller over W_" + std::to_string(c[2]) + "(" + F.name() +
                        "): t(x)^{q-1} = 1, t(x) = x mod ell, lift-independent, multiplicative");
    }
    {   // the lifted character t o psi: a homomorphism of order ord chi into W_K (lem:teichmuller (3)).
        const Fq F = Fq::Prime(13);
        const GaloisRing W(F, 3);
        const DirichletCharacter c = DirichletCharacter::Conrey(7, 2);
        const CharacterValues v = ReduceCharacter(c, F);
        const std::vector<GaloisRing::W> lt = LiftCharacter(v, W);
        bool ok = true;
        for (uint64_t a : UnitsMod(7))
            for (uint64_t b : UnitsMod(7)) ok = ok && W.mul(lt[a], lt[b]) == lt[a * b % 7];
        for (uint64_t a : UnitsMod(7)) ok = ok && W.pow(lt[a], 3) == W.one() && W.reduce(lt[a]) == v.at(static_cast<int64_t>(a));
        g.check(ok, "(F) t o psi for 7.2 over W_3(F_13) = Z/13^3: a homomorphism of order 3 reducing to psi");
    }
}

// ---------------------------------------------------------------------------
// (B) (NB2): chi = 1 is the shipped level-N model bit for bit.
// ---------------------------------------------------------------------------
void ArmTrivial(Checker& g) {
    PhaseTimer pt("(B) (NB2) chi = 1");
    for (const auto& cell : std::vector<std::array<uint32_t, 3>>{{6, 2, 13}, {11, 2, 7}, {5, 4, 13}, {12, 2, 7}}) {
        const uint64_t N = cell[0];
        const uint32_t n = cell[1], ell = cell[2];
        const std::string tag = "(N=" + std::to_string(N) + ", n=" + std::to_string(n) + ", F_" + std::to_string(ell) + ")";
        const engine::ModCtx<uint32_t> ctx(ell);
        const Fq F = Fq::Prime(ell);
        const hecke::level::Gamma0LevelN lvl(N);
        const CharacterValues one = ReduceCharacter(DirichletCharacter::Trivial(N), F);
        Model& M = TheModel();
        // relations
        const auto rc = InducedRelationsChi(*M.K, lvl, n, ctx, CharacterScalar(one));
        const auto rl = hecke::level::InducedRelationsLevel(*M.K, lvl, n, ctx);
        bool rel_ok = rc.size() == rl.size();
        for (size_t i = 0; rel_ok && i < rc.size(); ++i) {
            rel_ok = rc[i].q_dim == rl[i].q_dim && rc[i].s_lhs_idx_in_Kq == rl[i].s_lhs_idx_in_Kq &&
                     rc[i].s_rhs_idx_in_Kq == rl[i].s_rhs_idx_in_Kq && rc[i].tag == rl[i].tag &&
                     rc[i].M.get_rows() == rl[i].M.get_rows();
            for (uint32_t r = 0; rel_ok && r < rc[i].M.get_rows(); ++r)
                for (uint32_t c = 0; c < rc[i].M.get_cols(); ++c)
                    if (rc[i].M.get_entry(r, c) != rl[i].M.get_entry(r, c)) { rel_ok = false; break; }
        }
        g.check(rel_ok, "(B) " + tag + ": eq:MHchi at chi = 1 = InducedRelationsLevel, every relation matrix entrywise");
        // the cochain maps
        std::map<uint32_t, HeckeDatum> dat;
        for (uint32_t p : {2u, 3u}) dat[p] = MakeDatum(p, ell, n);
        // cohomology on both relation lists
        ModelCell mc = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new hecke::level::Gamma0LevelN(N)), n, ctx,
                                  CharacterScalar(one));
        LocalSystemSimplicialComplex Ls(*M.K, rl, lvl.m() * (n + 1), ell);
        EquivariantCohomologyOnK Hs(&Ls);
        Hs.compute_cohomology();
        g.check(mc.H->get_H_eq_dimensions() == Hs.get_H_eq_dimensions(), "(B) " + tag + ": dim H^q equal (shipped vs chi = 1)");
        const hecke::level::HFrame fc(mc.H.get(), 1), fs(&Hs, 1);
        for (uint32_t p : {2u, 3u}) {
            const hecke::level::Gamma0LevelN lvlNp(N * p);
            hecke::level::CorrespondenceSpec spec{&lvl, &lvl, dat[p].cosets.get(), dat[p].tau.get(),
                                                  hecke::level::Gamma0SubTransversalN(lvl, lvlNp), "T"};
            for (uint32_t q = 0; q <= 2; ++q) {
                const auto Fc = BuildLevelCorrespondenceChi(ctx, spec, *M.K, n, q, CharacterScalar(one));
                const auto Fs = hecke::level::BuildLevelCorrespondence(ctx, spec, *M.K, n, q);
                g.check(Fc.aggregate() == Fs.aggregate() && Fc.terms.size() == Fs.terms.size(),
                        "(B) " + tag + ": eq:nebcorr at chi = 1 = eq:levelcorr, degree " + std::to_string(q) + ", p = " + std::to_string(p));
            }
            const auto Fc = BuildLevelCorrespondenceChi(ctx, spec, *M.K, n, 1, CharacterScalar(one));
            const auto Fs = hecke::level::BuildLevelCorrespondence(ctx, spec, *M.K, n, 1);
            g.check(hecke::level::HEqual(hecke::level::InducedOnH(fc, fc, Fc, "chi"), hecke::level::InducedOnH(fs, fs, Fs, "shipped")),
                    "(B) " + tag + ": the matrix of " + std::string(N % p ? "T_" : "U_") + std::to_string(p) + " on H^1 entrywise");
        }
        // the cokernel and the Heilbronn action
        const NebCokernel ck(lvl, n, F, one);
        const hecke::level::FixedSpace fS = hecke::level::OrbitWiseFixedSpace(lvl, Mat22::S(), 2, n, ctx);
        const hecke::level::FixedSpace fU = hecke::level::OrbitWiseFixedSpace(lvl, Mat22::ST(), 3, n, ctx);
        const hecke::level::CokernelDims cd = hecke::level::CokernelFromFixedSpaces(fS, fU, ell);
        const hecke::level::CokernelPresentation pres(fS, fU, ell);
        g.check(ck.h0() == cd.h0 && ck.h1() == cd.h1 && ck.dim_S() == cd.dim_S && ck.dim_U() == cd.dim_ST &&
                    ck.free_columns() == pres.free_columns() && ck.h1() == Hs.get_H_eq_dimensions()[1],
                "(B) " + tag + ": the cokernel at chi = 1 = prop:levelorbit's (h0, h1, dims, free columns) = the model's h^1");
        for (uint32_t p : {2u, 3u, 5u, 7u}) {
            if (p == ell) continue;
            const HeilbronnChi Hc(lvl, p, n, F, one);
            const hecke::level::HeilbronnOperator Hs2(lvl, p, n, ctx);
            bool app = true;
            std::vector<uint32_t> e(ck.D(), 0);
            for (uint32_t c = 0; c < ck.D() && app; ++c) {
                e[c] = 1;
                app = Hc.apply(e) == Hs2.apply(e);
                e[c] = 0;
            }
            const FqMat A = ck.OnH1(Hc);
            const HMat B = hecke::level::OnCokernel(pres, Hs2);
            g.check(app && A.a == ToU32(B.a), "(B) " + tag + ": H_{N," + std::to_string(p) +
                                                  ",chi} at chi = 1 = H_{N,p} (every column) and its matrix on M/Rel entrywise");
        }
    }
}

// ---------------------------------------------------------------------------
// (C) the character model across three routes (d = 1).
// ---------------------------------------------------------------------------
struct CharCell { uint64_t N, m; uint32_t n, ell; };
void ArmRoutes(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(C) three routes");
    const std::vector<CharCell> cells{{7, 2, 2, 13},  {7, 2, 4, 13}, {9, 2, 1, 7},  {13, 8, 1, 17}, {5, 2, 1, 13},
                                      {12, 11, 2, 7}, {15, 2, 2, 13}, {11, 3, 2, 31}, {8, 5, 2, 5},  {16, 3, 1, 5},
                                      {7, 6, 1, 13},  {21, 5, 2, 7}};
    for (const CharCell& cc : cells) {
        const DirichletCharacter chi = DirichletCharacter::Conrey(cc.N, cc.m);
        const Fq F = Fq::Prime(cc.ell);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const engine::ModCtx<uint32_t> ctx(cc.ell);
        const hecke::level::Gamma0LevelN lvl(cc.N);
        const std::string tag = "(" + chi.label() + ", n=" + std::to_string(cc.n) + ", F_" + std::to_string(cc.ell) + ")";
        const NebCokernel ck(lvl, cc.n, F, psi);
        g.check(!ck.parity_zero(), "(C) " + tag + ": chi(-1) = (-1)^n (a non-zero cell)");
        if (ck.parity_zero()) continue;
        const BlockScalar s = CharacterScalar(psi);
        ModelCell mc = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new hecke::level::Gamma0LevelN(cc.N)), cc.n, ctx, s);
        const std::vector<uint32_t>& dims = mc.H->get_H_eq_dimensions();
        GroupRouteChi gr(cc.n, cc.N, cc.ell, std::vector<uint64_t>(psi.val.begin(), psi.val.end()));
        gr.Setup();
        g.check(dims.size() == 3 && dims[0] == ck.h0() && dims[1] == ck.h1() && dims[2] == 0,
                "(C) " + tag + ": model (h0, h1, h2) = (" + std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + ", " +
                    std::to_string(dims.size() > 2 ? dims[2] : 99) + ") = cokernel (h0, h1, 0)");
        g.check(gr.h0() == ck.h0() && gr.h1() == ck.h1() && gr.restricted() == gr.h1(),
                "(C) " + tag + ": group route (h0, h1) = (" + std::to_string(gr.h0()) + ", " + std::to_string(gr.h1()) +
                    ") = cokernel; Shapiro restriction of full rank");
        std::map<uint32_t, FqMat> ops;
        std::ostringstream os;
        os << "NEB_ROUTES cell=" << chi.label() << " n=" << cc.n << " ell=" << cc.ell << " h0=" << ck.h0() << " h1=" << ck.h1();
        for (uint32_t p : {2u, 3u, 5u}) {
            if (p == cc.ell) continue;
            const HeilbronnChi H(lvl, p, cc.n, F, psi);
            ops[p] = ck.OnH1(H);
            const std::vector<uint32_t> cp = FqCharpoly(F, ops[p]);
            const HeckeDatum dat = MakeDatum(p, cc.ell, cc.n);
            const HMat Am = ModelHeckeChi(mc, lvl, p, cc.n, ctx, s, dat);
            const std::vector<uint32_t> cm = ToU32(hecke::level::HCharpoly(Am));
            const std::vector<uint32_t> cg = FqCharpoly(F, gr.Operator(p));
            const std::string op = std::string(cc.N % p ? "T_" : "U_") + std::to_string(p);
            g.check(cp == cm && cp == cg, "(C) " + tag + ": charpoly of " + op + " -- cokernel (H_{N,p,chi}) " + VecText(cp) +
                                              " = model (eq:nebcorr) " + VecText(cm) + " = group route " + VecText(cg));
            os << " " << op << "=" << VecText(cp);
        }
        bool comm = true;
        for (const auto& a : ops)
            for (const auto& b : ops) comm = comm && FqEqual(FqMul(F, a.second, b.second), FqMul(F, b.second, a.second));
        g.check(comm, "(C) " + tag + ": thm:nebrelations (1) -- the operators commute on H^1 (cokernel route)");
        lines.push_back(os.str());
    }
    // prop:neborbit (2): chi(-1) != (-1)^n gives 0 -- on the model (where nothing is assumed) and reported by the cokernel.
    for (const CharCell& cc : std::vector<CharCell>{{7, 2, 1, 13}, {13, 8, 2, 17}, {9, 2, 2, 7}, {5, 4, 1, 11}}) {
        const DirichletCharacter chi = DirichletCharacter::Conrey(cc.N, cc.m);
        const Fq F = Fq::Prime(cc.ell);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const engine::ModCtx<uint32_t> ctx(cc.ell);
        const hecke::level::Gamma0LevelN lvl(cc.N);
        const NebCokernel ck(lvl, cc.n, F, psi);
        ModelCell mc = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new hecke::level::Gamma0LevelN(cc.N)), cc.n, ctx,
                                  CharacterScalar(psi));
        const std::vector<uint32_t>& dims = mc.H->get_H_eq_dimensions();
        g.check(ck.parity_zero() && dims[0] == 0 && dims[1] == 0 && dims[2] == 0,
                "(C) (" + chi.label() + ", n=" + std::to_string(cc.n) + "): chi(-1) != (-1)^n -> H^* = 0 on the model "
                "(prop:neborbit (2)); the cokernel reports parity_zero");
    }
}

// ---------------------------------------------------------------------------
// (D) (NB1): prop:neborbit (4) in characteristic 0 against the cokernel.
// ---------------------------------------------------------------------------
void ArmEuler(Checker& g) {
    PhaseTimer pt("(D) (NB1) Euler characteristic");
    uint32_t cells = 0, bad = 0;
    for (uint64_t N = 1; N <= 30; ++N) {
        const hecke::level::Gamma0LevelN lvl(N);
        const uint32_t bad_before = bad, cells_before = cells;
        for (const DirichletCharacter& chi : AllCharacters(N)) {
            const uint64_t o = chi.order();
            uint64_t ell = 5;
            while (!(IsPrime(ell) && N % ell != 0 && (ell - 1) % o == 0)) ++ell;
            const Fq F = Fq::Prime(static_cast<uint32_t>(ell));
            const CharacterValues psi = ReduceCharacter(chi, F);
            for (uint32_t n = 0; n <= 5; ++n) {
                const EulerPrediction E = EulerCharacteristicFormula(N, n, chi);
                const NebCokernel ck(lvl, n, F, psi);
                ++cells;
                bool ok = E.parity_ok == !ck.parity_zero();
                if (ok && E.parity_ok)
                    ok = static_cast<int64_t>(ck.h1()) - static_cast<int64_t>(ck.h0()) == E.h1_minus_h0 &&
                         static_cast<int64_t>(ck.dim_S()) == E.dim_S && static_cast<int64_t>(ck.dim_U()) == E.dim_U &&
                         E.nu2 == hecke::level::Nu2(N) && E.nu3 == hecke::level::Nu3(N);
                if (!ok) {
                    ++bad;
                    g.check(false, "(D) (" + chi.label() + ", n=" + std::to_string(n) + "): prop:neborbit (4) h1 - h0 = " +
                                       std::to_string(E.h1_minus_h0) + " vs cokernel " + std::to_string(ck.h1()) + " - " +
                                       std::to_string(ck.h0()));
                }
            }
        }
        g.check(bad == bad_before, "(D) N = " + std::to_string(N) + ": prop:neborbit (4) = the cokernel on all " +
                                       std::to_string(cells - cells_before) + " (chi, n <= 5) cells");
    }
    g.check(bad == 0, "(D) (NB1) prop:neborbit (4) (char 0, exact) = the cokernel's h1 - h0, dim M^S, dim M^U on " +
                          std::to_string(cells) + " cells (every chi mod N <= 30, n <= 5)");
    std::cout << "  NEB_EULER cells=" << cells << " failing=" << bad << std::endl;
}

// ---------------------------------------------------------------------------
// (E) (NB3): the Gamma_1(N) model and its decomposition over F_q.
// ---------------------------------------------------------------------------
void ArmGammaOne(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(E) (NB3) Gamma_1(N)");
    struct G1 { uint64_t N; uint32_t n, ell, d; };
    // rmk:nebcheckable (NB3)'s cells: N in {5, 7, 13}, ell in {5, 7, 11, 13, 29} \ {N}, q = ell^d with lambda(N) | q - 1
    // (d = d(lambda) in {1, 2}), n <= NMAX(N, ell) -- the time budget (standard tier, <= 60 s; Gamma_1(13) has m = 168,
    // D = 168 (n + 1)); the grid is stated below.
    // S16 WP-10/11 repairs (audit finding 4): the default grid is WP-10's grid again (N = 5 n <= 10; N = 7 n <= 10 at
    // ell = 13, n <= 4 otherwise; N = 13 n <= 2 at ell = 5, n = 1 otherwise: 67 cells -- WP-10's report said 81, WP-11's
    // trim left 53, not 67) PLUS n = 0 at each of the 12 (N, ell) pairs: 79 cells; the FULL committed grid of
    // rmk:nebcheckable (NB3) -- n = 0 .. 10 at every (N, ell), 132 cells -- runs under HECKE_NEBENTYPUS_NB3_FULL=1
    // (EFFECTIVE line), once per close.
    bool full = false;
    if (const char* e = std::getenv("HECKE_NEBENTYPUS_NB3_FULL")) {
        if (std::string(e) != "1")
            throw std::invalid_argument("HECKE_NEBENTYPUS_NB3_FULL: unknown value \"" + std::string(e) + "\" (the only value is \"1\")");
        full = true;
    }
    std::vector<G1> grid;
    for (uint64_t N : {5ull, 7ull, 13ull}) {
        const uint64_t lam = N == 5 ? 4 : N == 7 ? 6 : 12;
        for (uint32_t ell : {5u, 7u, 11u, 13u, 29u}) {
            if (ell == N) continue;
            const uint32_t d = LeastDegree(ell, lam);
            const uint32_t nmax = full ? 10 : N == 5 ? 10 : N == 7 ? (ell == 13 ? 10 : 4) : (ell == 5 ? 2 : 1);
            for (uint32_t n = 0; n <= nmax; ++n) grid.push_back(G1{N, n, ell, d});
        }
    }
    if (full) std::cout << "  EFFECTIVE HeckeNebentypusValidation arm=NB3_FULL cells=" << grid.size() << " (HECKE_NEBENTYPUS_NB3_FULL=1)" << std::endl;
    // Slot-17 WP-6: the character models' route (default dense, as shipped; sparse runs the (NB3) verdicts on the sparse
    // cokernel -- "NB3_FULL's 132 cells must read the same verdicts on the sparse route").
    NebRoute nb3_route = NebRoute::kDense;
    if (const char* e = std::getenv("HECKE_NEBENTYPUS_NB3_ROUTE")) {
        nb3_route = ParseNebRoute(e, "HECKE_NEBENTYPUS_NB3_ROUTE");
        std::cout << "  EFFECTIVE HeckeNebentypusValidation arm=NB3_ROUTE cells=" << grid.size() << " route=" << NebRouteName(nb3_route)
                  << " (HECKE_NEBENTYPUS_NB3_ROUTE)" << std::endl;
    }
    g.check(grid.size() == (full ? 132u : 79u), "(E) the (NB3) grid has " + std::to_string(grid.size()) + " cells");
    for (const G1& c : grid) {
        const auto tc0 = std::chrono::steady_clock::now();
        const std::string tag = "(Gamma_1(" + std::to_string(c.N) + "), n=" + std::to_string(c.n) + ", F_" + std::to_string(c.ell) +
                                (c.d > 1 ? "^" + std::to_string(c.d) : std::string()) + ")";
        const engine::ModCtx<uint32_t> ctx(c.ell);
        const Fq F = CanonField(c.ell, c.d);
        const Fq Fl = Fq::Prime(c.ell);
        const std::vector<uint64_t> units = UnitsMod(c.N);
        const uint32_t phi = static_cast<uint32_t>(units.size());
        auto G = std::unique_ptr<Gamma1LevelN>(new Gamma1LevelN(c.N));
        const Gamma1LevelN* Gp = G.get();
        ModelCell mc = BuildModel(std::move(G), c.n, ctx, TableScalar(c.N, std::vector<uint32_t>(c.N, 1)));
        const std::vector<uint32_t> dims = mc.H->get_H_eq_dimensions();
        const hecke::level::HFrame f(mc.H.get(), 1);
        // the diamonds
        std::map<uint64_t, FqMat> dia;
        for (uint64_t u : units)
            dia[u] = FromHMat(hecke::level::InducedOnH(f, f, DiamondOnModel(*Gp, u, *TheModel().K, c.n, ctx, 1), "<d>"));
        bool hom = FqEqual(dia[1], FqIdentity(dims[1]));
        for (uint64_t a : units)
            for (uint64_t b : units) hom = hom && FqEqual(FqMul(Fl, dia[a], dia[b]), dia[a * b % c.N]);
        g.check(hom, "(E) " + tag + ": <1> = 1 and <d><d'> = <dd'> on H^1 (prop:nebdecomp (1))");
        // T_p / U_p on the Gamma_1 model by eq:levelcorr with r_t in Gamma_1(N)
        std::map<uint32_t, FqMat> T;
        for (uint32_t p : {2u, 3u}) {
            if (p == c.ell) continue;
            const HeckeDatum dat = MakeDatum(p, c.ell, c.n);
            hecke::level::CorrespondenceSpec spec{Gp, Gp, dat.cosets.get(), dat.tau.get(), Gamma1SubTransversal(c.N, p),
                                                  std::string(c.N % p ? "T_" : "U_") + std::to_string(p)};
            hecke::level::VerifyTransversal(spec, c.N % p ? p + 1 : p);
            T[p] = FromHMat(hecke::level::InducedOnH(f, f, hecke::level::BuildLevelCorrespondence(ctx, spec, *TheModel().K, c.n, 1), spec.label));
            bool comm = true;
            for (uint64_t u : units) comm = comm && FqEqual(FqMul(Fl, dia[u], T[p]), FqMul(Fl, T[p], dia[u]));
            g.check(comm, "(E) " + tag + ": " + spec.label + " commutes with every <d> (prop:nebcorr (4))");
        }
        // the decomposition over F_q (prop:nebdecomp (3)): e_chi = phi^{-1} sum_u chi(u)^{-1} <u>.
        const hecke::level::Gamma0LevelN lvl(c.N);
        uint32_t sum_h1 = 0, sum_h0 = 0;
        std::map<uint32_t, std::vector<uint32_t>> prod;
        for (const auto& kv : T) prod[kv.first] = {1};
        std::ostringstream os;
        os << "NEB_GAMMA1 N=" << c.N << " n=" << c.n << " field=" << F.name() << " h1(Gamma_1)=" << dims[1] << " parts:";
        const uint32_t phi_inv = F.inv(F.from_int(phi));
        for (const DirichletCharacter& chi : AllCharacters(c.N)) {
            const CharacterValues psi = ReduceCharacter(chi, F);
            FqMat e(dims[1], dims[1]);
            for (uint64_t u : units) e = FqAdd(F, e, FqScale(F, dia[u], F.inv(psi.at(static_cast<int64_t>(u)))));
            e = FqScale(F, e, phi_inv);
            const FqMat W = FqColumnBasis(F, e);
            const NebPresentation ck(lvl, c.n, F, psi, nb3_route);
            sum_h1 += W.cols;
            sum_h0 += ck.h0();
            bool eig = true;
            for (uint64_t u : units) eig = eig && FqEqual(FqMul(F, dia[u], W), FqScale(F, W, psi.at(static_cast<int64_t>(u))));
            g.check(W.cols == ck.h1() && eig, "(E) " + tag + ": the " + chi.label() + "-part has dim " + std::to_string(W.cols) +
                                                  " = h^1 of the character model, <d> = chi(d) on it");
            os << " " << chi.label() << ":" << W.cols;
            for (const auto& kv : T) {
                if (W.cols == 0) continue;
                const FqMat X = FqSolve(F, W, FqMul(F, kv.second, W));
                const std::vector<uint32_t> cpart = FqCharpoly(F, X);
                const HeilbronnChi H(lvl, kv.first, c.n, F, psi);
                const std::vector<uint32_t> cchar = FqCharpoly(F, ck.OnH1(H));
                g.check(cpart == cchar, "(E) " + tag + ": " + std::string(c.N % kv.first ? "T_" : "U_") + std::to_string(kv.first) +
                                            " on the " + chi.label() + "-part of H^1(Gamma_1) = on the character model " +
                                            FqPolyText(F, cchar));
                prod[kv.first] = PolyMulFq(F, prod[kv.first], cchar);
            }
        }
        // h^0(Gamma_1) from the model
        g.check(sum_h1 == dims[1] && sum_h0 == dims[0],
                "(E) " + tag + ": h^q(Gamma_1(N)) = sum_chi h^q(Gamma_0(N), chi), q = 0, 1 (" + std::to_string(dims[0]) + ", " +
                    std::to_string(dims[1]) + ")");
        for (const auto& kv : T)
            g.check(FqCharpoly(F, kv.second) == prod[kv.first],
                    "(E) " + tag + ": charpoly of " + std::string(c.N % kv.first ? "T_" : "U_") + std::to_string(kv.first) +
                        " on H^1(Gamma_1) = the product over chi of the character models' (NB3)");
        os << " s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
        lines.push_back(os.str());
    }
}

// ---------------------------------------------------------------------------
// (N5) / (N6) / NB1's classical half: WP-9's table.
// ---------------------------------------------------------------------------
struct CellKey { uint64_t N; uint64_t m; uint32_t k; };
// The expected tallies of WP-9_nb_predictions.tsv (WP-10's run of record; audit finding 2).
constexpr uint32_t kDistinguishing = 20, kControls = 4, kRealRows = 94, kNb6Rows = 377, kPsiNotes = 24;
void ArmTable(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(N5)/(N6) WP-9 table");
    const std::string root = FindRepoRoot();
    g.check(!root.empty(), "(N5) the repository root holding " + std::string(kTsv) + " is located");
    if (root.empty()) return;
    const std::vector<TsvRow> rows = ReadTsv(root + kTsv);
    g.check(rows.size() >= 700, "(N5) WP-9_nb_predictions.tsv read: " + std::to_string(rows.size()) + " data rows");
    std::map<std::string, CellKey> cells;
    for (const TsvRow& r : rows)
        if (r.kind == "conv" && r.key == "lmfdb_conrey_label") {
            uint64_t N = 0;
            uint32_t k = 0;
            if (!ParseCell(r.cell, N, k)) throw std::runtime_error("(N5) unparsable cell " + r.cell);
            const std::vector<std::string> lab = SplitStr(r.value, '.');
            if (!(lab.size() == 2 && std::stoull(lab[0]) == N)) throw std::runtime_error("(N5) unparsable Conrey label " + r.value);
            cells[r.cell] = CellKey{N, std::stoull(lab[1]), k};
        }
    g.check(cells.size() == 8, "(N5) eight (NB5) cells with Conrey labels (5 real, 3 non-real)");
    // one cokernel per (cell, ell, chi or chibar)
    std::map<std::tuple<std::string, uint32_t, bool>, std::unique_ptr<NebCokernel>> ck;
    std::map<std::tuple<std::string, uint32_t, bool>, std::unique_ptr<Fq>> fields;
    std::map<std::tuple<std::string, uint32_t, bool>, CharacterValues> psis;
    auto get = [&](const std::string& cell, uint32_t ell, bool bar) -> const NebCokernel& {
        const auto key = std::make_tuple(cell, ell, bar);
        auto it = ck.find(key);
        if (it != ck.end()) return *it->second;
        const CellKey& c = cells.at(cell);
        fields[key].reset(new Fq(Fq::Prime(ell)));
        DirichletCharacter chi = DirichletCharacter::Conrey(c.N, c.m);
        if (bar) chi = chi.inverse();
        psis[key] = ReduceCharacter(chi, *fields[key]);
        const hecke::level::Gamma0LevelN lvl(c.N);
        ck[key].reset(new NebCokernel(lvl, c.k - 2, *fields[key], psis[key]));
        return *ck[key];
    };
    auto charpoly = [&](const std::string& cell, uint32_t ell, bool bar, uint32_t p) {
        const NebCokernel& K = get(cell, ell, bar);
        const auto key = std::make_tuple(cell, ell, bar);
        const CellKey& c = cells.at(cell);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const HeilbronnChi H(lvl, p, c.k - 2, *fields[key], psis[key]);
        return FqCharpoly(*fields[key], K.OnH1(H));
    };
    // (NB1)'s classical half: h1 - h0 = 2 dim S + dim E, at every (cell, ell) used.
    std::map<std::string, int64_t> pred_dim;
    for (const TsvRow& r : rows)
        if (r.kind == "dim" && r.key == "dim_H1_predicted_(NB1)") pred_dim[r.cell] = std::stoll(r.value);
    // (NB5)
    uint32_t notes = 0;
    uint32_t n5 = 0, n5_ok = 0, distinguishing = 0, read_psi = 0, read_psibar = 0, read_neither = 0, controls = 0, controls_ok = 0;
    std::set<std::pair<std::string, uint32_t>> dimmed;
    std::map<std::tuple<std::string, uint32_t, uint32_t>, std::array<std::vector<uint32_t>, 2>> nonreal;   // (cell, p, ell) -> psi, psibar lines
    for (const TsvRow& r : rows) {
        if (r.kind != "charpoly_mod_ell") continue;
        const uint32_t ell = static_cast<uint32_t>(std::stoul(r.ell));
        const uint32_t p = static_cast<uint32_t>(std::stoul(r.key.substr(2, r.key.find('|') - 2)));
        const std::vector<uint32_t> want = ParseCoeffs(r.value);
        if (!dimmed.count({r.cell, ell})) {
            dimmed.insert({r.cell, ell});
            const NebCokernel& K = get(r.cell, ell, false);
            g.check(!K.parity_zero() && K.h0() == 0 && static_cast<int64_t>(K.h1()) == pred_dim.at(r.cell),
                    "(N5) " + r.cell + " over F_" + std::to_string(ell) + ": h0 = 0, h1 = " + std::to_string(K.h1()) +
                        " = 2 dim S + dim E (the TSV's dim_H1_predicted_(NB1), NB1's classical half)");
        }
        const bool is_psibar = r.key.find("[psibar]") != std::string::npos;
        const bool is_psi = !is_psibar && r.key.find("[psi]") != std::string::npos;
        if (is_psi || is_psibar) {
            nonreal[std::make_tuple(r.cell, p, ell)][is_psibar ? 1 : 0] = want;
            if (is_psi) {
                // the psi values in the note, against the reduction rule.
                const size_t at = r.note.find("(psi(");
                if (at != std::string::npos) {
                    const size_t close = r.note.find(")=", at);
                    const size_t end = r.note.find(')', close + 2);
                    const std::vector<std::string> args = SplitStr(r.note.substr(at + 5, close - at - 5), ',');
                    const std::vector<std::string> vals = SplitStr(r.note.substr(close + 2, end - close - 2), ',');
                    get(r.cell, ell, false);
                    const CharacterValues& ps = psis.at(std::make_tuple(r.cell, ell, false));
                    bool ok = args.size() == vals.size() && !args.empty();
                    for (size_t i = 0; ok && i < args.size(); ++i)
                        ok = ps.at(std::stoll(args[i])) == static_cast<uint32_t>(std::stoul(vals[i]));
                    ++notes;
                    g.check(ok, "(N5) " + r.cell + " over F_" + std::to_string(ell) + ": psi (e(1/o) -> s, s the least of order " +
                                    std::to_string(ps.order) + " = " + std::to_string(ps.s) + ") gives the note's psi(" +
                                    r.note.substr(at + 5, close - at - 5) + ")");
                }
            }
            continue;
        }
        ++n5;
        const std::vector<uint32_t> got = charpoly(r.cell, ell, false, p);
        const bool ok = got == want;
        n5_ok += ok;
        g.check(ok, "(N5) " + r.cell + " " + r.key + ": computed " + VecText(got) + " = WP-9 " + VecText(want));
    }
    // the dictionary on the non-real cells: the code's character := psi; which line does it reproduce?
    for (const auto& kv : nonreal) {
        const std::string& cell = std::get<0>(kv.first);
        const uint32_t p = std::get<1>(kv.first), ell = std::get<2>(kv.first);
        const std::vector<uint32_t>& lpsi = kv.second[0];
        const std::vector<uint32_t>& lbar = kv.second[1];
        const std::vector<uint32_t> got = charpoly(cell, ell, false, p);
        const std::vector<uint32_t> got_bar = charpoly(cell, ell, true, p);
        const bool m_psi = got == lpsi, m_bar = got == lbar;
        const bool control = lpsi == lbar;
        std::string reading = m_psi && m_bar ? "both" : m_psi ? "psi" : m_bar ? "psibar" : "neither";
        std::ostringstream os;
        os << "NEB_DICT cell=" << cell << " T_" << p << " ell=" << ell << " code(psi)=" << VecText(got) << " [psi]=" << VecText(lpsi)
           << " [psibar]=" << VecText(lbar) << " reading=" << reading << (control ? " control" : " distinguishing")
           << " code(psibar)=" << (got_bar == lbar ? "[psibar]" : got_bar == lpsi ? "[psi]" : "neither");
        lines.push_back(os.str());
        if (control) {
            ++controls;
            controls_ok += m_psi ? 1 : 0;
            g.check(m_psi, "(N5) " + cell + " T_" + std::to_string(p) + " F_" + std::to_string(ell) +
                               ": control (the two lines coincide) reproduced " + VecText(got));
            continue;
        }
        ++distinguishing;
        if (m_psi) ++read_psi; else if (m_bar) ++read_psibar; else ++read_neither;
        g.check(m_psi && !m_bar, "(N5) " + cell + " T_" + std::to_string(p) + " F_" + std::to_string(ell) +
                                     ": the character model at chi := psi reproduces the [psi] line (LMFDB S_k(N, chi)), "
                                     "not [psibar] -- the dictionary of [GT, bridge:nebentypus-dictionary]");
        g.check(got_bar == lbar, "(N5) " + cell + " T_" + std::to_string(p) + " F_" + std::to_string(ell) +
                                     ": and at chi := psibar the [psibar] line (the swap is exactly chi <-> chibar)");
    }
    std::ostringstream sum;
    sum << "NEB_DICT_SUMMARY distinguishing=" << distinguishing << " reading_psi=" << read_psi << " reading_psibar=" << read_psibar
        << " reading_neither=" << read_neither << " controls=" << controls << " controls_held=" << controls_ok
        << " real_rows=" << n5 << " real_held=" << n5_ok << " psi_notes=" << notes;
    lines.push_back(sum.str());
    // S16 WP-10/11 repairs (audit finding 2): the tallies are gated, so a changed TSV key format cannot shrink the
    // distinguishing set (or the psi-note checks) to zero and still pass.
    g.check(distinguishing == kDistinguishing && read_psi == kDistinguishing && read_psibar == 0 && read_neither == 0 &&
                controls == kControls && controls_ok == kControls && n5 == kRealRows && n5_ok == kRealRows && notes == kPsiNotes,
            "(N5) the tallies: " + std::to_string(distinguishing) + " distinguishing (all read [psi]), " + std::to_string(controls) +
                " controls, " + std::to_string(n5) + " real rows, " + std::to_string(notes) + " psi notes (expected " +
                std::to_string(kDistinguishing) + ", " + std::to_string(kControls) + ", " + std::to_string(kRealRows) + ", " +
                std::to_string(kPsiNotes) + ")");
    // (N6): the CM cells, cusp^2 . Eis at every split / inert p of the table.
    uint32_t n6 = 0, n6_ok = 0;
    for (const TsvRow& r : rows) {
        if (r.kind != "nb6_mod_ell") continue;
        const uint32_t ell = static_cast<uint32_t>(std::stoul(r.ell));
        const uint32_t p = static_cast<uint32_t>(std::stoul(r.key.substr(2, r.key.find(' ') - 2)));
        const uint32_t a = static_cast<uint32_t>(std::stoul(Trim(r.value.substr(r.value.find('=') + 1))));
        const CellKey& c = cells.at(r.cell);
        if (c.N % p == 0 || p == ell) continue;
        get(r.cell, ell, false);
        const auto key = std::make_tuple(r.cell, ell, false);
        const Fq& F = *fields.at(key);
        const CharacterValues& ps = psis.at(key);
        const uint32_t chip = ps.at(p);
        const uint32_t pk1 = F.pow(F.from_int(p), c.k - 1);
        const uint32_t e1 = F.add(1, F.mul(chip, pk1)), e2 = F.add(chip, pk1);
        std::vector<uint32_t> want = PolyMulFq(F, PolyMulFq(F, {F.neg(a), 1}, {F.neg(a), 1}), PolyMulFq(F, {F.neg(e1), 1}, {F.neg(e2), 1}));
        const std::vector<uint32_t> got = charpoly(r.cell, ell, false, p);
        ++n6;
        n6_ok += got == want;
        g.check(got == want, "(N6) " + r.cell + " T_" + std::to_string(p) + " F_" + std::to_string(ell) + ": (X - " + std::to_string(a) +
                                 ")^2 (X - (1 + chi(p)p^{k-1}))(X - (chi(p) + p^{k-1})) = " + VecText(got));
    }
    std::ostringstream s6;
    s6 << "NEB_NB6 rows=" << n6 << " held=" << n6_ok;
    lines.push_back(s6.str());
    g.check(n6 == kNb6Rows && n6_ok == kNb6Rows, "(N6) the tally: " + std::to_string(n6) + " rows (expected " + std::to_string(kNb6Rows) + ")");
    // T(p, p) = chi(p) p^n (S16 WP-10/11 repairs, audit finding 8): on every non-real cell of the table and every p !| N
    // of the table's primes, p != ell, the central correspondence (HeilbronnChi::Central, the single Heilbronn matrix pI)
    // is chi(p) p^n on H^1; at a p with chi(p) != chibar(p) the chibar value is the negative control.
    uint32_t tpp = 0, tpp_disc = 0;
    for (const auto& kv : nonreal) {
        const std::string& cell = std::get<0>(kv.first);
        const uint32_t p = std::get<1>(kv.first), ell = std::get<2>(kv.first);
        const CellKey& c = cells.at(cell);
        if (c.N % p == 0 || p == ell) continue;
        const NebCokernel& K = get(cell, ell, false);
        const auto key = std::make_tuple(cell, ell, false);
        const Fq& F = *fields.at(key);
        const CharacterValues& ps = psis.at(key);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const FqMat X = K.OnH1(HeilbronnChi::Central(lvl, p, c.k - 2, F, ps));
        const uint32_t pn = F.pow(F.from_int(p), c.k - 2);
        const uint32_t want = F.mul(ps.at(p), pn), bar = F.mul(F.inv(ps.at(p)), pn);
        const bool ok = FqEqual(X, FqScale(F, FqIdentity(X.rows), want));
        ++tpp;
        g.check(ok && X.rows > 0, "(N6) " + cell + " F_" + std::to_string(ell) + ": T(" + std::to_string(p) + "," + std::to_string(p) +
                                      ") = chi(p) p^n = " + std::to_string(want) + " on H^1 (dim " + std::to_string(X.rows) + ")");
        if (want != bar) {
            ++tpp_disc;
            g.check(!FqEqual(X, FqScale(F, FqIdentity(X.rows), bar)),
                    "(N6) " + cell + " F_" + std::to_string(ell) + ": T(" + std::to_string(p) + "," + std::to_string(p) +
                        ") != chibar(p) p^n (the negative control)");
        }
    }
    g.check(tpp > 0 && tpp_disc > 0, "(N6) T(p, p) was checked (" + std::to_string(tpp) + " cells x p, " + std::to_string(tpp_disc) +
                                         " with chi(p) != chibar(p))");
    lines.push_back("NEB_TPP checked=" + std::to_string(tpp) + " discriminating=" + std::to_string(tpp_disc));
}

// ---------------------------------------------------------------------------
// (W) W_K(F_ell) = Z/ell^K, the Teichmuller-lifted character on the fixed model.
// ---------------------------------------------------------------------------
void ArmGaloisRing(Checker& g) {
    PhaseTimer pt("(W) W_K(F_ell)");
    struct WC { uint64_t N, m; uint32_t n, ell, K; };
    for (const WC& c : std::vector<WC>{{7, 2, 2, 13, 2}, {13, 8, 1, 17, 3}}) {
        const DirichletCharacter chi = DirichletCharacter::Conrey(c.N, c.m);
        const Fq F = Fq::Prime(c.ell);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const GaloisRing W(F, c.K);
        const std::vector<GaloisRing::W> lt = LiftCharacter(psi, W);
        std::vector<uint32_t> table(c.N, 0);
        for (size_t a = 0; a < c.N; ++a) table[a] = static_cast<uint32_t>(lt[a][0]);
        const engine::ModCtx<uint32_t> rctx = engine::ModCtx<uint32_t>::prime_power(c.ell, c.K);
        const engine::ModCtx<uint32_t> fctx(c.ell);
        const std::string tag = "(" + chi.label() + ", n=" + std::to_string(c.n) + ", Z/" + std::to_string(c.ell) + "^" + std::to_string(c.K) + ")";
        Model& M = TheModel();
        const hecke::level::Gamma0LevelN lvl(c.N);
        // cochain level: the relation matrices over W_K reduce to those over F_ell.
        const auto rr = InducedRelationsChi(*M.K, lvl, c.n, rctx, TableScalar(c.N, table));
        const auto rf = InducedRelationsChi(*M.K, lvl, c.n, fctx, CharacterScalar(psi));
        bool red = rr.size() == rf.size();
        for (size_t i = 0; red && i < rr.size(); ++i)
            for (uint32_t r = 0; red && r < rr[i].M.get_rows(); ++r)
                for (uint32_t col = 0; col < rr[i].M.get_cols(); ++col)
                    if (rr[i].M.get_entry(r, col) % c.ell != rf[i].M.get_entry(r, col)) { red = false; break; }
        g.check(red, "(W) " + tag + ": eq:MHchi over W_K with t o psi reduces to eq:MHchi over F_ell, every entry");
        ModelCell mr = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new hecke::level::Gamma0LevelN(c.N)), c.n, rctx,
                                  TableScalar(c.N, table));
        const NebCokernel ck(lvl, c.n, F, psi);
        std::unique_ptr<hecke::ScopedCoefficientRing> ring(new hecke::ScopedCoefficientRing(c.K));
        const std::vector<uint32_t> len = mr.H->H_lengths();
        const std::vector<uint32_t>& inv1 = mr.H->H_invariant_factors(1);
        bool free = inv1.size() == ck.h1();
        for (uint32_t e : inv1) free = free && e == c.K;
        g.check(len.size() >= 2 && len[0] == 0 && len[1] == c.K * ck.h1() && free,
                "(W) " + tag + ": H^1 = (Z/ell^K)^{h1}, h1 = " + std::to_string(ck.h1()) + " (lengths " +
                    std::to_string(len.size() > 1 ? len[1] : 0) + "), H^0 = 0");
        // T_2 over the ring, reduced mod ell: the field charpoly.
        const HeckeDatum dat = MakeDatum(2, c.ell, c.n);
        const HMat A = ModelHeckeChi(mr, lvl, 2, c.n, rctx, TableScalar(c.N, table), dat);
        FqMat Ar(A.rows, A.cols);
        for (size_t i = 0; i < A.a.size(); ++i) Ar.a[i] = static_cast<uint32_t>(A.a[i] % c.ell);
        const HeilbronnChi H(lvl, 2, c.n, F, psi);
        g.check(FqCharpoly(F, Ar) == FqCharpoly(F, ck.OnH1(H)),
                "(W) " + tag + ": T_2 on H^1 over W_K, reduced mod ell, has the field charpoly " + VecText(FqCharpoly(F, Ar)));
    }
}

// ---------------------------------------------------------------------------
// (slot-16 WP-11) Field helpers for the two new arms: kernels, polynomials of
// matrices, the joint primary decomposition over F_ell by a generic combination,
// and the local algebra of a block (its radical by Frobenius, its Hilbert function,
// the number of module generators).
// ---------------------------------------------------------------------------
FqMat FqKernel(const Fq& F, const FqMat& A) {
    FqMat R = A;
    const std::vector<uint32_t> piv = FqRref(F, R);
    std::vector<uint8_t> isp(A.cols, 0);
    for (uint32_t c : piv) isp[c] = 1;
    std::vector<uint32_t> fr;
    for (uint32_t c = 0; c < A.cols; ++c) if (!isp[c]) fr.push_back(c);
    FqMat K(A.cols, static_cast<uint32_t>(fr.size()));
    for (uint32_t j = 0; j < fr.size(); ++j) {
        K.at(fr[j], j) = 1;
        for (uint32_t r = 0; r < piv.size(); ++r) K.at(piv[r], j) = F.neg(R.at(r, fr[j]));
    }
    return K;
}
FqMat PolyOfMat(const Fq& F, const std::vector<uint32_t>& f, const FqMat& X) {   // f low -> high
    FqMat R(X.rows, X.cols);
    for (size_t i = f.size(); i-- > 0;) R = FqAdd(F, FqMul(F, R, X), FqScale(F, FqIdentity(X.rows), f[i]));
    return R;
}
FqMat MatPow(const Fq& F, FqMat A, uint64_t e) {
    FqMat R = FqIdentity(A.rows);
    while (e) { if (e & 1) R = FqMul(F, R, A); A = FqMul(F, A, A); e >>= 1; }
    return R;
}
bool IsZero(const FqMat& A) { return std::all_of(A.a.begin(), A.a.end(), [](uint32_t x) { return x == 0; }); }
// Irreducible factors (with multiplicity) of a charpoly over the prime field F_ell.
std::vector<std::pair<std::vector<uint32_t>, uint32_t>> FactorFl(uint32_t ell, const std::vector<uint32_t>& f) {
    const hecke::FpPolyFactorization fac = hecke::FactorPoly(hecke::FpPoly(ell, f));
    std::vector<std::pair<std::vector<uint32_t>, uint32_t>> out;
    for (const auto& fc : fac.factors) out.push_back({fc.pi.coeffs, fc.multiplicity});
    return out;
}
// Rank of a list of vectors over F (incremental elimination).
struct Span {
    const Fq* F;
    uint32_t len;
    std::vector<std::vector<uint32_t>> rows;
    std::vector<uint32_t> piv;
    Span(const Fq& f, uint32_t l) : F(&f), len(l) {}
    bool add(std::vector<uint32_t> v) {
        for (size_t r = 0; r < rows.size(); ++r) {
            const uint32_t c = v[piv[r]];
            if (!c) continue;
            const uint32_t m = F->neg(c);
            for (uint32_t k = 0; k < len; ++k) if (rows[r][k]) v[k] = F->add(v[k], F->mul(m, rows[r][k]));
        }
        uint32_t p = 0;
        while (p < len && v[p] == 0) ++p;
        if (p == len) return false;
        const uint32_t iv = F->inv(v[p]);
        for (uint32_t& x : v) x = F->mul(x, iv);
        rows.push_back(std::move(v));
        piv.push_back(p);
        return true;
    }
    uint32_t dim() const { return static_cast<uint32_t>(rows.size()); }
};
// A basis (as matrices) of the F_ell-algebra generated by gens, closed under products.
std::vector<FqMat> AlgebraBasis(const Fq& F, const std::vector<FqMat>& gens, uint32_t d) {
    Span sp(F, d * d);
    std::vector<FqMat> basis;
    auto push = [&](const FqMat& M) { if (sp.add(M.a)) basis.push_back(M); };
    push(FqIdentity(d));
    for (size_t i = 0; i < basis.size(); ++i)
        for (const FqMat& g : gens) push(FqMul(F, basis[i], g));
    return basis;
}
// The span of a set of matrices: its basis (as matrices).
std::vector<FqMat> SpanBasis(const Fq& F, const std::vector<FqMat>& ms, uint32_t d) {
    Span sp(F, d * d);
    std::vector<FqMat> b;
    for (const FqMat& M : ms) if (sp.add(M.a)) b.push_back(M);
    return b;
}

struct BlockReading {
    uint32_t D = 0, r = 0;                // F_ell-dimension, residue degree
    bool separated = true;                // every T_p has one irreducible factor on the block
    bool dihedral = false, ordinary = true;
    std::map<uint64_t, std::vector<uint32_t>> pi;   // per p the irreducible factor (low -> high)
    std::vector<uint64_t> nonss;          // primes where T_p is not semisimple on the block
    uint32_t dimA = 0;
    std::vector<uint32_t> hilbert;        // F_ell-dims of rad^j / rad^{j+1}
    uint32_t gens = 0;                    // F_ell-dim of W / rad(A) W
};
// The joint primary blocks of ops (p -> h x h over F_ell) by a generic combination.
std::vector<BlockReading> ReadBlocks(const Fq& F, const std::map<uint64_t, FqMat>& ops, uint64_t ell, int64_t kron_disc) {
    std::vector<BlockReading> out;
    if (ops.empty() || ops.begin()->second.rows == 0) return out;
    const uint32_t h = ops.begin()->second.rows;
    // The joint primary decomposition over F_ell: refine by the generalised eigenspaces of each T_p's irreducible
    // factors in turn, then by a combination sum (i + 1) T_{p_i} (coefficients non-zero mod ell); every final block is
    // then checked LOCAL (A / rad A a field of degree r) -- `separated`.
    std::vector<FqMat> Ws{FqIdentity(h)};
    std::vector<FqMat> gensFull;
    for (const auto& kv : ops) gensFull.push_back(kv.second);
    FqMat comb(h, h);
    {
        uint32_t c = 1;
        for (const auto& kv : ops) { comb = FqAdd(F, comb, FqScale(F, kv.second, F.from_int(c))); ++c; }
    }
    gensFull.push_back(comb);
    for (const FqMat& G : gensFull) {
        std::vector<FqMat> next;
        for (const FqMat& W : Ws) {
            const FqMat X = FqSolve(F, W, FqMul(F, G, W));
            for (const auto& fe : FactorFl(static_cast<uint32_t>(ell), FqCharpoly(F, X))) {
                const FqMat K = FqKernel(F, MatPow(F, PolyOfMat(F, fe.first, X), fe.second));
                next.push_back(FqMul(F, W, K));
            }
        }
        Ws = std::move(next);
    }
    for (const FqMat& W : Ws) {
        BlockReading B;
        B.D = W.cols;
        {
            const auto fc = FactorFl(static_cast<uint32_t>(ell), FqCharpoly(F, FqSolve(F, W, FqMul(F, comb, W))));
            B.r = static_cast<uint32_t>(fc[0].first.size() - 1);
        }
        std::vector<FqMat> X;
        for (const auto& kv : ops) {
            const FqMat Xp = FqSolve(F, W, FqMul(F, kv.second, W));
            X.push_back(Xp);
            const auto fp = FactorFl(static_cast<uint32_t>(ell), FqCharpoly(F, Xp));
            if (fp.size() != 1) { B.separated = false; continue; }
            B.pi[kv.first] = fp[0].first;
            if (!IsZero(PolyOfMat(F, fp[0].first, Xp))) B.nonss.push_back(kv.first);
            const bool nilp = fp[0].first.size() == 2 && fp[0].first[0] == 0;
            if (kv.first == ell) B.ordinary = !nilp;
        }
        // dihedral (for K = Q(sqrt(kron_disc))): T_p nilpotent at every inert p of S, p != ell.
        if (kron_disc != 0) {
            B.dihedral = true;
            for (const auto& kv : B.pi) {
                const uint64_t p = kv.first;
                if (p == ell) continue;
                int kr;
                if (p == 2) { const int64_t m8 = ((kron_disc % 8) + 8) % 8; kr = (m8 == 1) ? 1 : (m8 == 5 ? -1 : 0); }
                else {
                    const uint64_t a = static_cast<uint64_t>(((kron_disc % static_cast<int64_t>(p)) + static_cast<int64_t>(p)) % static_cast<int64_t>(p));
                    kr = a == 0 ? 0 : (powmod<uint64_t>(a, (p - 1) / 2, p) == 1 ? 1 : -1);
                }
                if (kr == -1 && !(kv.second.size() == 2 && kv.second[0] == 0)) B.dihedral = false;
            }
        }
        // the local algebra, its radical (kernel of Frobenius^s, ell^s >= D), Hilbert function, generators
        const std::vector<FqMat> A = AlgebraBasis(F, X, B.D);
        B.dimA = static_cast<uint32_t>(A.size());
        uint64_t es = ell;
        while (es < B.D) es *= ell;
        FqMat Phi(B.D * B.D, B.dimA);   // columns: Frob(b_i) as vectors
        for (uint32_t i = 0; i < B.dimA; ++i) {
            const FqMat Fi = MatPow(F, A[i], es);
            for (uint32_t k = 0; k < B.D * B.D; ++k) Phi.at(k, i) = Fi.a[k];
        }
        const FqMat ker = FqKernel(F, Phi);
        std::vector<FqMat> rad;
        for (uint32_t j = 0; j < ker.cols; ++j) {
            FqMat M(B.D, B.D);
            for (uint32_t i = 0; i < B.dimA; ++i)
                if (ker.at(i, j)) M = FqAdd(F, M, FqScale(F, A[i], ker.at(i, j)));
            rad.push_back(M);
        }
        std::vector<FqMat> cur = SpanBasis(F, rad, B.D);
        uint32_t prev = B.dimA;
        while (true) {
            B.hilbert.push_back(prev - static_cast<uint32_t>(cur.size()));
            if (cur.empty()) break;
            prev = static_cast<uint32_t>(cur.size());
            std::vector<FqMat> nxt;
            for (const FqMat& a : cur) for (const FqMat& b : rad) nxt.push_back(FqMul(F, a, b));
            cur = SpanBasis(F, nxt, B.D);
            ENGINE_CHECK_MSG(cur.size() < prev, "ReadBlocks: the radical is not nilpotent");
        }
        Span mw(F, B.D);
        for (const FqMat& a : rad)
            for (uint32_t col = 0; col < B.D; ++col) {
                std::vector<uint32_t> v(B.D);
                for (uint32_t i = 0; i < B.D; ++i) v[i] = a.at(i, col);
                mw.add(v);
            }
        B.gens = B.D - mw.dim();
        if (B.hilbert.empty() || B.hilbert[0] != B.r) B.separated = false;   // A / rad A is not the field F_{ell^r}: not local
        out.push_back(B);
    }
    return out;
}
std::map<uint64_t, FqMat> LevelOneOps(const Fq& F, uint32_t n, const std::vector<uint64_t>& primes, uint32_t& h1, uint32_t& h0) {
    const hecke::level::Gamma0LevelN one(1);
    const CharacterValues triv = ReduceCharacter(DirichletCharacter::Trivial(1), F);
    const NebCokernel ck(one, n, F, triv);
    h1 = ck.h1();
    h0 = ck.h0();
    std::map<uint64_t, FqMat> ops;
    if (ck.parity_zero() || h1 == 0) return ops;
    for (uint64_t p : primes) ops[p] = ck.OnH1(HeilbronnChi(one, p, n, F, triv));
    return ops;
}

// ---------------------------------------------------------------------------
// (X4) (NB4) the weight shift on cohomology: N = ell, chi = omega^i (omega the
// identity of (Z/ell)^x), against the level-1 route.  At n = 0 the sharp form
// (rmk:nebcheckable (NB4)): the T_p-characteristic polynomial on H^1(Gamma_0(ell);
// F_ell(omega^i)) equals that of T_p on H^1(Gamma; Sym^i) times that of p^i T_p on
// H^1(Gamma; Sym^{ell-1-i}) -- the constituents Sym^i and Sym^{ell-1-i} (x) det^i of
// Ind_B omega^i.  At n = 1, 2 (with (-1)^i = (-1)^n and n <= i <= ell - 1 - n, where
// Clebsch-Gordan holds over F_ell) the first run's reading: Sym^n (x) each constituent.
// ---------------------------------------------------------------------------
std::vector<uint32_t> TwistedCharpoly(const Fq& F, const FqMat& X, uint64_t p, uint32_t e) {
    return FqCharpoly(F, FqScale(F, X, F.pow(F.from_int(static_cast<int64_t>(p)), e)));
}
void ArmWeightShift(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(X4) (NB4) weight shift");
    const std::vector<uint64_t> P{2, 3, 5, 7};
    uint32_t sharp_cells = 0, sharp_held = 0, swap_held = 0, read_cells = 0, read_held = 0, read_h0 = 0;
    std::map<std::pair<uint32_t, uint32_t>, std::map<uint64_t, std::vector<uint32_t>>> lvl1;   // (ell, a) -> p -> charpoly
    std::map<std::pair<uint32_t, uint32_t>, std::map<uint64_t, FqMat>> lvl1ops;
    for (uint32_t ell : {5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u}) {
        const Fq F = Fq::Prime(ell);
        std::vector<uint64_t> primes;
        for (uint64_t p : P) if (p != ell) primes.push_back(p);
        auto level1 = [&](uint32_t a) -> const std::map<uint64_t, FqMat>& {
            auto key = std::make_pair(ell, a);
            auto it = lvl1ops.find(key);
            if (it != lvl1ops.end()) return it->second;
            uint32_t h1 = 0, h0 = 0;
            return lvl1ops[key] = LevelOneOps(F, a, primes, h1, h0);
        };
        auto cp1 = [&](uint32_t a, uint64_t p, uint32_t e) -> std::vector<uint32_t> {
            const auto& ops = level1(a);
            if (ops.empty()) return {1};
            return TwistedCharpoly(F, ops.at(p), p, e);
        };
        const hecke::level::Gamma0LevelN lvl(ell);
        for (uint32_t n = 0; n <= 2; ++n)
            for (uint32_t i = 0; i + 2 <= ell - 1; ++i) {
                if ((i + n) % 2) continue;
                if (n > 0 && !(n <= i && i + n <= ell - 1)) continue;
                CharacterValues om;
                om.N = ell;
                om.order = (ell - 1) / gcd<uint64_t>(i, ell - 1);
                om.val.assign(ell, 0);
                for (uint64_t a = 1; a < ell; ++a) om.val[a] = static_cast<uint32_t>(powmod<uint64_t>(a, i, ell));
                om.s = om.val[1];
                const NebCokernel ck(lvl, n, F, om);
                ENGINE_CHECK_MSG(!ck.parity_zero(), "(X4) omega^i with (-1)^i = (-1)^n is not parity-zero");
                bool held = true, held_swap = true, h0corr = false;
                std::ostringstream os;
                os << "NEB_WEIGHTSHIFT ell=" << ell << " n=" << n << " i=" << i << " h0=" << ck.h0() << " h1=" << ck.h1();
                for (uint64_t p : primes) {
                    const std::vector<uint32_t> got = FqCharpoly(F, ck.OnH1(HeilbronnChi(lvl, p, n, F, om)));
                    std::vector<uint32_t> pred{1}, swp{1};
                    if (n == 0) {
                        pred = PolyMulFq(F, cp1(i, p, 0), cp1(ell - 1 - i, p, i));
                        swp = PolyMulFq(F, cp1(ell - 1 - i, p, 0), cp1(i, p, ell - 1 - i));
                    } else {
                        // Sym^n (x) Sym^i = (+)_j Sym^{n+i-2j} (x) det^j; Sym^n (x) Sym^{ell-1-i} (x) det^i likewise.
                        for (uint32_t j = 0; j <= n; ++j) {
                            pred = PolyMulFq(F, pred, cp1(n + i - 2 * j, p, j));
                            pred = PolyMulFq(F, pred, cp1(n + ell - 1 - i - 2 * j, p, i + j));
                        }
                        swp = pred;
                    }
                    bool ok = got == pred;
                    if (!ok && n > 0 && i == n) {
                        // the constituent Sym^0 (x) det^n has H^0 = F_ell (T_p = p^n (1 + p)); it is one class fewer in H^1.
                        const uint32_t ev = F.mul(F.pow(F.from_int(static_cast<int64_t>(p)), n), F.from_int(static_cast<int64_t>(p) + 1));
                        ok = PolyMulFq(F, got, {F.neg(ev), 1}) == pred;   // the connecting map kills the H^0 class
                        if (ok) h0corr = true;
                    }
                    held = held && ok;
                    held_swap = held_swap && got == swp;
                    os << " T_" << p << (got == pred ? "=pred" : ok ? "=pred/(X-p^n(1+p))" : "!=pred");
                }
                os << (n == 0 ? (held ? " sharp=HELD" : " sharp=FINDING") : (held ? (h0corr ? " reading=HELD_WITH_H0" : " reading=HELD") : " reading=DIFFERS"));
                if (n > 0) read_h0 += h0corr;
                if (n == 0) {
                    os << (held_swap ? " swapped=holds" : " swapped=fails");
                    ++sharp_cells;
                    sharp_held += held;
                    swap_held += held_swap;
                    g.check(held, "(X4) (NB4) n = 0, ell = " + std::to_string(ell) + ", omega^" + std::to_string(i) +
                                      ": charpoly T_p on H^1(Gamma_0(ell); omega^i) = [Sym^i] . [p^i Sym^{ell-1-i}] at level 1, p in {2,3,5,7}");
                } else {
                    ++read_cells;
                    read_held += held;
                }
                lines.push_back(os.str());
            }
    }
    std::ostringstream s;
    // S16 WP-10/11 repairs (audit finding 3): the cell count and the swapped reading's failure are gated -- the run pins
    // which constituent is untwisted only if the swapped reading fails somewhere (it holds exactly where the two coincide).
    g.check(sharp_cells == 132 && swap_held == 18 && swap_held < sharp_cells && read_cells == 240,
            "(X4) the tallies: " + std::to_string(sharp_cells) + " sharp n = 0 cells (132), the swapped reading holds on " +
                std::to_string(swap_held) + " (18, the coincident cells) and fails on the rest; " + std::to_string(read_cells) +
                " n = 1, 2 reading cells (240)");
    // slot-17 WP-17 (F1 m-C2): the (NB4)(b) reading is gated too -- it held on every one of the 240 n = 1, 2 cells at every run
    // of record (NEB_WEIGHTSHIFT_SUMMARY ... n12_held=240), so a regression there is a failure, not a printed number.
    g.check(read_held == 240 && read_held == read_cells,
            "(X4) the (NB4)(b) reading holds on all " + std::to_string(read_cells) + " n = 1, 2 cells (n12_held=" +
                std::to_string(read_held) + ", expected 240)");
    s << "NEB_WEIGHTSHIFT_SUMMARY sharp_n0_cells=" << sharp_cells << " held=" << sharp_held << " swapped_reading_holds=" << swap_held
      << " n12_reading_cells=" << read_cells << " n12_held=" << read_held << " of_which_with_H0_class=" << read_h0;
    lines.push_back(s.str());
}

// ---------------------------------------------------------------------------
// (M2) WP-C16's SPEC items at level 1: m = 2 on the non-ordinary dihedral blocks
// and H^1_m ~= A^2 (A the block's local Hecke algebra), at J-7's fixture cells, the
// ordinary control (24, 47), and the F1 hidden blocks at 17 / 19 (D = 4, a_ell = 0).
// Readings over F_ell on H^1(Gamma; Sym^n F_ell) (the level-1 cokernel), and, where
// the block is F_ell-rational, the certified Z/ell^K route (hecke_ext::ringalg).
// ---------------------------------------------------------------------------
std::string HilbText(const std::vector<uint32_t>& h) { return VecText(h); }
void ArmMultiplicityTwo(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(M2) m = 2, H^1_m = A^2");
    struct MC { uint32_t k, ell; const char* what; };
    const std::vector<MC> cells{{24, 47, "ordinary dihedral control"}, {72, 47, "non-ordinary dihedral (J-7 fixture)"},
                                {118, 47, "non-ordinary dihedral"},
                                {156, 103, "non-ordinary dihedral"}};
    for (const MC& c : cells) {
        const Fq F = Fq::Prime(c.ell);
        std::vector<uint64_t> primes{2, 3, 5, 7, 11, 13, 17, 19, 23};
        primes.push_back(c.ell);
        uint32_t h1 = 0, h0 = 0;
        std::unique_ptr<PhaseTimer> t1(new PhaseTimer("(M2) dihedral: level-1 operators"));
        const std::map<uint64_t, FqMat> ops = LevelOneOps(F, c.k - 2, primes, h1, h0);
        t1.reset(new PhaseTimer("(M2) dihedral: blocks"));
        const std::vector<BlockReading> blocks = ReadBlocks(F, ops, c.ell, -static_cast<int64_t>(c.ell));
        t1.reset();
        const hecke::ordinary::TorsionPrepass tp = hecke::ordinary::TorsionExponentPrepass(c.k - 2, c.ell);
        uint32_t ndih = 0;
        for (const BlockReading& B : blocks) {
            g.check(B.separated, "(M2) (" + std::to_string(c.k) + ", " + std::to_string(c.ell) +
                                     "): the generic combination separates the joint systems (one irreducible factor per T_p)");
            if (!B.dihedral) continue;
            if (B.r == 1) {
                // a rational system with t_p = 0 at the inert p of S is REDUCIBLE here: Ind_K 1 = 1 + eps_K twisted,
                // t_p = p^a + p^b for some a, b (checked on all of S), not one of WP-C16's irreducible Ind_K chi.
                bool red = false;
                for (uint32_t a = 0; a + 1 < c.ell && !red; ++a)
                    for (uint32_t b = a; b + 1 < c.ell && !red; ++b) {
                        bool all = true;
                        for (const auto& kv : B.pi) {
                            const uint64_t p = kv.first;
                            const uint32_t t = kv.second.size() == 2 ? F.neg(kv.second[0]) : 99999;
                            const uint32_t want = p == c.ell ? (a == 0 ? 1u : 0u) + (b == 0 ? 1u : 0u)
                                                             : F.add(F.pow(F.from_int(static_cast<int64_t>(p)), a), F.pow(F.from_int(static_cast<int64_t>(p)), b));
                            if (p == c.ell) continue;   // U-type at ell is not of this shape; S \ {ell} decides
                            (void)want;
                            if (t != want) { all = false; break; }
                        }
                        red = all;
                    }
                g.check(red, "(M2) (" + std::to_string(c.k) + ", " + std::to_string(c.ell) +
                                 "): every rational 'dihedral' system is reducible (t_p = p^a + p^b on S)");
                continue;
            }
            ++ndih;
            const uint32_t m = B.D / (2 * B.r);
            std::ostringstream os;
            os << "NEB_M2 k=" << c.k << " ell=" << c.ell << " " << (B.ordinary ? "ordinary" : "non-ordinary") << " dihedral r=" << B.r
               << " D=" << B.D << " m=" << m << " dimA=" << B.dimA << " hilbert=" << HilbText(B.hilbert) << " gens=" << B.gens
               << " e_max=" << tp.e_max << " nonss=";
            for (uint64_t p : B.nonss) os << p << ",";
            const bool freeA2 = B.gens == 2 * B.r && B.D == 2 * B.dimA;
            os << " H1m_free_rank2_over_A=" << (freeA2 ? "yes" : "no");
            lines.push_back(os.str());
            const std::string tag = "(M2) (" + std::to_string(c.k) + ", " + std::to_string(c.ell) + ") " +
                                    (B.ordinary ? "ordinary" : "non-ordinary") + " dihedral block, r = " + std::to_string(B.r);
            if (B.ordinary) {
                g.check(m == 1 && B.dimA == B.r && B.hilbert == std::vector<uint32_t>{B.r} && freeA2,
                        tag + ": m = 1, A = F_q (reduced), H^1_m = A^2 (the control)");
            } else {
                g.check(m == 2, tag + ": D = 4r, m = 2 (WP-C16 M5, SPEC -> DATA)");
                g.check(B.dimA == 2 * B.r && B.hilbert == std::vector<uint32_t>{B.r, B.r},
                        tag + ": A = F_q[eps]/eps^2 (dim_F_ell A = 2r, Hilbert [r, r]) -- J-2's forms-side A_W");
                g.check(freeA2, tag + ": H^1_m is free of rank 2 over A (generators 2r = " + std::to_string(B.gens) +
                                    ", D = 2 dim A) -- H^1_m ~= A^2");
                g.check(!B.nonss.empty(), tag + ": some T_p is not semisimple (J-7's Jordan (2, 2) reading)");
                // Over Z_ell: Lambda = H^1(Gamma; Sym^n Z_ell) is torsion-free (e_max = 0), so Lambda_m is free of rank D
                // and rank T'_m = ceil(D / 2) = 2r ([C, lem:heckerankone]); dim A_1 = 2r = rank T'_m makes A_1 = T'_m / ell
                // (the [C, cor:stablecert] equality, at K = 1), and two generators over A_1 lift (Nakayama): T'_m^2 ->> Lambda_m
                // between free Z_ell-modules of rank 4r, an isomorphism.
                g.check(tp.e_max == 0 && B.dimA == B.D / 2 && freeA2,
                        tag + ": Lambda_m ~= T'_m^2 over Z_ell (e_max = 0, dim A_1 = rank T'_m = 2r, 2 generators)");
            }
        }
        g.check(ndih == 1, "(M2) (" + std::to_string(c.k) + ", " + std::to_string(c.ell) + "): exactly one irreducible dihedral block (" + c.what + "; predictions.tsv: one ideal)");
    }
    // The F1 blocks (WP-12 (b), law L-F1): the block of rho_Delta (x) omega^j at its onset n = 12 ell + j (ell + 1) - 2,
    // found by its system t_p = p^j tau(p) mod ell (p in {2, 3, 5, 7}); read over F_ell, and on the certified
    // Z/ell^K route (hecke_ext::ringalg, [C, cor:stablecert]).
    //
    // S16 WP-10/11 repairs (audit finding 1): the reading is taken for T' = Z_ell[T_p : p in {2, 3, 5, 7, ell}] (WP-11's
    // algebra, kept and gated as before) AND for the FULL Hecke algebra T, read through every T_p, p <= PMAX (default
    // 23). The generation argument for the full algebra ([standard], the Miller-basis duality): with d = dim S_k,
    // T_Z(S_k) is the Z-span of T_1, ..., T_d (the pairing (T, f) -> a_1(T f) is perfect on T_Z(S_k) x S_k(Z) and T_1 ..
    // T_d are dual to the Miller basis); at a non-Eisenstein maximal ideal m, T(M_k)_m = T(S_k)_m and T^L_m = T^f_m
    // ([C, prop:cohforms] (3)); every T_n, n <= d, is a Z-polynomial in the T_p, p | n. So T_p, p <= d, generate T_m
    // as a Z_ell-algebra, and the images of those T_p generate the image A_1 of T_m in End(Lambda_m / ell): the gate
    // below checks that every prime p <= d is among the operators read and that the block is not Eisenstein.
    // The certified Z/ell^K route (T'_m / ell = F[x]/x^2, [C, cor:stablecert] (1)) costs ~30 s per cell at these n (the
    // module run at K = 2). The repair moved it behind HECKE_NEBENTYPUS_F1_RING (audit findings 11, 12): "first" runs it at
    // (220, 17), "all" at both cells; any other value is refused. The default gates the F_ell reading for the FULL algebra,
    // which alone decides "Lambda_m is not T_m^2".
    const char* ring_env = std::getenv("HECKE_NEBENTYPUS_F1_RING");
    uint32_t ring_cells = 0;
    if (ring_env) {
        const std::string v(ring_env);
        if (v != "all" && v != "first")
            throw std::invalid_argument("HECKE_NEBENTYPUS_F1_RING: unknown value \"" + v + "\" (the values are \"first\", \"all\")");
        ring_cells = v == "all" ? 2 : 1;
        std::cout << "  EFFECTIVE HeckeNebentypusValidation arm=F1_RING cells=" << ring_cells << " (HECKE_NEBENTYPUS_F1_RING=" << v << ")" << std::endl;
    }
    uint64_t pmax = 23;
    if (const char* e = std::getenv("HECKE_NEBENTYPUS_F1_PMAX")) {
        const std::string v(e);
        size_t used = 0;
        unsigned long long x = 0;
        try { x = std::stoull(v, &used); } catch (const std::exception&) { used = 0; }
        if (v.empty() || used != v.size() || x < 23 || x > 199)
            throw std::invalid_argument("HECKE_NEBENTYPUS_F1_PMAX: \"" + v + "\" is not an integer in [23, 199]");
        pmax = x;
        std::cout << "  EFFECTIVE HeckeNebentypusValidation arm=F1_PMAX pmax=" << pmax << " (HECKE_NEBENTYPUS_F1_PMAX)" << std::endl;
    }
    struct FC { uint32_t ell, j; };
    const int64_t tau[4] = {-24, 252, 4830, -16744};
    for (const FC& c : std::vector<FC>{{17, 1}, {19, 1}}) {
        const uint32_t n = 12 * c.ell + c.j * (c.ell + 1) - 2;
        const uint32_t k = n + 2;
        const uint32_t dimS = k / 12 - (k % 12 == 2 ? 1u : 0u);   // dim S_k(SL_2(Z)), k >= 4 even
        const Fq F = Fq::Prime(c.ell);
        std::vector<uint64_t> pfull;
        for (uint64_t p = 2; p <= pmax; ++p) if (IsPrime(p)) pfull.push_back(p);
        if (c.ell > pmax) pfull.push_back(c.ell);
        const std::vector<uint64_t> psub{2, 3, 5, 7, c.ell};
        uint32_t h1 = 0, h0 = 0;
        std::unique_ptr<PhaseTimer> ft(new PhaseTimer("(M2) F1: level-1 operators"));
        const std::map<uint64_t, FqMat> ops_full = LevelOneOps(F, n, pfull, h1, h0);
        ft.reset(new PhaseTimer("(M2) F1: blocks"));
        std::map<uint64_t, FqMat> ops_sub;
        for (uint64_t p : psub) ops_sub[p] = ops_full.at(p);
        const std::vector<BlockReading> blocks_sub = ReadBlocks(F, ops_sub, c.ell, 0);
        const std::vector<BlockReading> blocks_full = ReadBlocks(F, ops_full, c.ell, 0);
        ft.reset();
        std::vector<uint32_t> want;
        for (int t = 0; t < 4; ++t) want.push_back(F.mul(F.pow(F.from_int(static_cast<int64_t>(psub[t])), c.j), F.from_int(tau[t])));
        auto find = [&](const std::vector<BlockReading>& bl) -> const BlockReading* {
            const BlockReading* hit = nullptr;
            for (const BlockReading& B : bl) {
                if (B.r != 1 || !B.separated) continue;
                bool m = true;
                for (int t = 0; t < 4; ++t) m = m && B.pi.at(psub[t]).size() == 2 && F.neg(B.pi.at(psub[t])[0]) == want[t];
                if (m) hit = &B;
            }
            return hit;
        };
        const std::string tag = "(M2) F1 (n=" + std::to_string(n) + ", " + std::to_string(c.ell) + ", Delta (x) omega^" + std::to_string(c.j) + ")";
        const BlockReading* hs = find(blocks_sub);
        const BlockReading* hf = find(blocks_full);
        g.check(hs != nullptr && hf != nullptr, tag + ": the block of the system p^j tau(p) exists (for T' and for T)");
        if (!hs || !hf) continue;
        const BlockReading& B = *hs;
        const BlockReading& BF = *hf;
        std::vector<uint64_t> sys;
        for (uint64_t p : psub) sys.push_back(F.neg(B.pi.at(p)[0]));
        std::ostringstream os;
        os << "NEB_F1 n=" << n << " ell=" << c.ell << " j=" << c.j << " D=" << B.D << " a_ell=" << sys.back()
           << " T'={2,3,5,7,ell}: dimA1=" << B.dimA << " hilbertA1=" << HilbText(B.hilbert) << " gens=" << B.gens
           << " | T (p<=" << pmax << ", dim S_k=" << dimS << "): D=" << BF.D << " dimA1=" << BF.dimA << " hilbertA1=" << HilbText(BF.hilbert)
           << " gens=" << BF.gens << " a_p=";
        for (const auto& kv : BF.pi) os << kv.first << ":" << (kv.second.size() == 2 ? F.neg(kv.second[0]) : 99999u) << ",";
        os << " nonss=";
        for (uint64_t p : BF.nonss) os << p << ",";
        g.check(B.D == 4 && !B.ordinary, tag + ": D = 4 (m = 2, the doubling) and a_ell = 0");
        g.check(B.dimA == 1 && B.gens == 4,
                tag + ": for T' = Z_ell[T_2, T_3, T_5, T_7, T_ell]: A'_1 = F_ell and H^1(F_ell)_m needs 4 generators over A'_1");
        // the generation certificate for the full algebra (above): every prime p <= dim S_k is read, the block is the
        // same block (D unchanged: T' and T have the same block here), and it is not Eisenstein (a_p != 1 + p^{k-1}).
        bool covered = true;
        for (uint64_t p = 2; p <= dimS; ++p) if (IsPrime(p)) covered = covered && ops_full.count(p);
        bool non_eis = false;
        for (const auto& kv : BF.pi) {
            if (kv.first == c.ell || kv.second.size() != 2) continue;
            const uint32_t eis = F.add(1, F.pow(F.from_int(static_cast<int64_t>(kv.first)), k - 1));
            non_eis = non_eis || F.neg(kv.second[0]) != eis;
        }
        g.check(covered && non_eis && BF.D == B.D && BF.separated,
                tag + ": generation certificate for the full Hecke algebra: every prime p <= dim S_k = " + std::to_string(dimS) +
                    " is read (T(S_k) = span_Z(T_1 .. T_d)), the block is non-Eisenstein and is one block of T (D = " +
                    std::to_string(BF.D) + ")");
        g.check(BF.dimA == 1 && BF.gens == 4,
                tag + ": for the FULL Hecke algebra (T_p, p <= " + std::to_string(pmax) + "): A_1 = F_ell (every T_p acts on "
                      "Lambda_m / ell as the scalar a_p) and H^1(F_ell)_m needs 4 generators over A_1 -- Lambda_m is not T_m^2");
        if (!(ring_cells == 2 || (ring_cells == 1 && c.ell == 17))) {
            lines.push_back(os.str() + " ringalg=not_run (HECKE_NEBENTYPUS_F1_RING=first|all) (F1 reading over F_ell: hidden for the full T, 4 generators)");
            continue;
        }
        hecke_ext::ringalg::LoewyOptions o;
        o.kmax = 2;   // the hidden block is stable at K = 2 (T_m / ell of dim rank_T = 2); kmax 3 costs ~30 s per cell
        o.primes = psub;
        std::unique_ptr<PhaseTimer> rt(new PhaseTimer("(M2) F1 Z/ell^K route"));
        const hecke_ext::ringalg::LoewyCell cell = hecke_ext::ringalg::ComputeRingLoewyCell(n, c.ell, o);
        rt.reset();
        const hecke_ext::ringalg::LoewyBlock* lb = nullptr;
        for (const auto& bb : cell.blocks) {
            bool m = bb.system.size() == cell.primes.size();
            for (size_t i = 0; m && i < cell.primes.size(); ++i) m = bb.system[i] == F.neg(B.pi.at(cell.primes[i])[0]);
            if (m) lb = &bb;
        }
        g.check(lb != nullptr && lb->D == B.D && lb->A1.dim_mod_ell == B.dimA && lb->A1.hilbert == B.hilbert,
                tag + ": the Z/ell^K route finds the block with the same D and A_1 (dim, Hilbert function)");
        if (lb) {
            const hecke_ext::ringalg::AlgebraReading* T = lb->certified_T();
            os << " ringalg(T'): verdict=" << lb->verdict << " cert=" << lb->stable_cert << " f=" << lb->free_rank << " rank_T=" << lb->rank_T
               << " T'_m/ell=" << (T ? hecke_ext::ringalg::HilbertText(T->hilbert) : std::string("-"));
            // T'_m / ell certified of dim 2 = rank_T: rank T'_m = rank e T = 2 ([C, cor:stablecert] (1)). Lambda_m ~= T_m^2
            // would need Lambda_m / m_T Lambda_m of dim 2; m_T Lambda_m = ell Lambda_m + sum (T_p - a_p) Lambda_m = ell Lambda_m
            // since A_1 = F_ell for the FULL algebra (above), so 4 generators: not T_m^2, for T as for T'.
            const bool cert2 = T && lb->stable_cert == "certified" && T->dim_mod_ell == 2 && lb->free_rank == 4 && lb->torsion.empty();
            const bool freeT2 = cert2 && BF.gens == 2 && BF.dimA == 2;
            os << " Lambda_m=T_m^2:" << (freeT2 ? "yes" : "no");
            g.check(cert2, tag + ": T'_m / ell certified of dim 2 = rank_T ([C, cor:stablecert] (1)), Lambda_m free of rank 4");
            lines.push_back(os.str() + " (F1 reading: " + (freeT2 ? "H^1_m ~= T_m^2" : BF.dimA == 1 ? "hidden for the full T, not free over T_m" : "not free of rank 2") + ")");
        }
    }
}

// ---------------------------------------------------------------------------
// (R) the refusals, by name.
// ---------------------------------------------------------------------------
template <class Fn>
std::string RefusalCode(Fn fn) {
    try { fn(); } catch (const NebentypusRefusal& e) { return e.code(); } catch (const std::exception&) { return "other"; }
    return "none";
}
void ArmRefusals(Checker& g) {
    PhaseTimer pt("(R) refusals");
    const hecke::level::Gamma0LevelN lvl7(7);
    g.check(RefusalCode([&] {
                const Fq F = Fq::Prime(3);
                const CharacterValues v = ReduceCharacter(DirichletCharacter::Conrey(7, 6), F);
                NebCokernel ck(lvl7, 1, F, v);
            }) == "nebentypus_small_ell",
            "(R) ell = 3: nebentypus_small_ell (no cokernel, no comparison with group cohomology)");
    g.check(RefusalCode([&] { ReduceCharacter(DirichletCharacter::Conrey(11, 3), Fq::Prime(5)); }) == "character_wild",
            "(R) 11.3 (order 5) over F_5: character_wild (rmk:nebfield (2))");
    g.check(RefusalCode([&] { ReduceCharacter(DirichletCharacter::Conrey(7, 2), Fq::Prime(5)); }) == "character_field",
            "(R) 7.2 (order 3) over F_5: character_field (3 !| 4)");
    g.check(RefusalCode([&] { ReduceCharacter(DirichletCharacter::Conrey(7, 2), CanonField(5, 2)); }) == "none",
            "(R) 7.2 over F_25: accepted (3 | 24, d(chi) = 2)");
    g.check(RefusalCode([&] { Fq F(1009, hecke_ext::CanonicalFieldPolynomial(1009, 3)); }) == "field_too_large",
            "(R) F_{1009^3}: field_too_large (log tables)");
    g.check(RefusalCode([&] {
                const hecke::level::Gamma0LevelN lvl(1009);
                const Fq F = Fq::Prime(13);
                NebCokernel ck(lvl, 4, F, ReduceCharacter(DirichletCharacter::Trivial(1009), F));
            }) == "neb_dense_cap",
            "(R) D = 1010 * 5 > " + std::to_string(kNebDenseCap) + ": neb_dense_cap on the dense route (the sparse route of slot-17 WP-6 lifts it: arms (SP), (SL))");
    {
        const Fq F = Fq::Prime(13);
        const NebCokernel ck(lvl7, 1, F, ReduceCharacter(DirichletCharacter::Conrey(7, 2), F));
        g.check(ck.parity_zero() && ck.h1() == 0 && ck.h0() == 0, "(R) chi(-1) != (-1)^n is NOT a refusal: parity_zero, H^* = 0");
    }
}

// ---------------------------------------------------------------------------
// (slot-17 WP-6) NEBENTYPUS AT SCALE: (SP) the sparse cokernel with a character = the dense route entrywise,
// (PV) T_{p^nu}, nu >= 2, by thm:nebrelations (2), (3) against the direct double coset, (SL) one cell beyond the
// dense cap.  [C] sec:nebentypus's slot-17 code map (i), (ii); HeckeNebentypusSparse.h.
// ---------------------------------------------------------------------------
// The first Conrey character mod N of exact order o with chi(-1) = (-1)^n (or, with want_parity_zero, the opposite
// parity); throws if none.
DirichletCharacter PickCharacter(uint64_t N, uint64_t o, uint32_t n, bool want_parity_zero = false) {
    const int want = ((n % 2 == 0) != want_parity_zero) ? 1 : -1;
    for (const DirichletCharacter& c : AllCharacters(N))
        if (c.order() == o && c.parity() == want) return c;
    throw std::logic_error("PickCharacter: no character of order " + std::to_string(o) + " and the wanted parity mod " +
                           std::to_string(N));
}

std::vector<uint32_t> DenseOfSparseRow(uint32_t D, uint32_t piv, const SparseVec& tail) {
    std::vector<uint32_t> v(D, 0);
    v[piv] = 1;
    for (size_t k = 0; k < tail.col.size(); ++k) v[tail.col[k]] = tail.val[k];
    return v;
}

struct SpCell { uint64_t N, o; uint32_t n, ell; bool parity_zero; };
constexpr uint32_t kSpCellsRun = 13;

void ArmSparse(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(SP) sparse = dense");
    // Overlap cells (D <= 600 in the default run; larger overlap cells run under the hook, route=dense against
    // route=sparse): prime and composite N, d(chi) in {1, 2}, U_p at p | N, n up to 11, one parity-zero cell.
    const std::vector<SpCell> cells{
        {7, 6, 1, 13, false},  {7, 6, 3, 5, false},   {7, 6, 2, 13, true},   {13, 12, 3, 5, false}, {13, 4, 3, 17, false},
        {5, 4, 11, 13, false}, {35, 2, 4, 11, false}, {35, 4, 3, 13, false}, {36, 6, 2, 7, false},  {101, 2, 4, 7, false},
        {143, 2, 2, 7, false}, {210, 2, 0, 11, false}, {29, 7, 2, 43, false}, {91, 6, 1, 5, false}};
    uint32_t ran = 0;
    for (const SpCell& c : cells) {
        const DirichletCharacter chi = PickCharacter(c.N, c.o, c.n, c.parity_zero);
        const uint32_t d = LeastDegree(c.ell, chi.order());
        const Fq F = CanonField(c.ell, d);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const std::string tag = "(SP) (" + chi.label() + ", n=" + std::to_string(c.n) + ", " + F.name() + ")";
        const NebCokernel De(lvl, c.n, F, psi);
        const NebSparseCokernel Sp(lvl, c.n, F, psi);
        g.check(De.parity_zero() == Sp.parity_zero() && De.parity_zero() == c.parity_zero,
                tag + ": parity_zero agrees on both routes (" + std::string(c.parity_zero ? "zero" : "non-zero") + ")");
        if (c.parity_zero) {
            g.check(Sp.h0() == 0 && Sp.h1() == 0, tag + ": the sparse route reports H^* = 0 (prop:neborbit (2))");
            continue;
        }
        ++ran;
        g.check(De.D() == Sp.D() && De.dim_S() == Sp.dim_S() && De.dim_U() == Sp.dim_U() && De.orbits_S() == Sp.orbits_S() &&
                    De.orbits_U() == Sp.orbits_U() && De.h0() == Sp.h0() && De.h1() == Sp.h1(),
                tag + ": D, dim M^S, dim M^ST, the orbit counts, h0 = " + std::to_string(Sp.h0()) + ", h1 = " +
                    std::to_string(Sp.h1()) + " agree");
        std::vector<uint32_t> dpiv = De.pivots();
        std::sort(dpiv.begin(), dpiv.end());
        g.check(dpiv == Sp.pivots() && De.free_columns() == Sp.free_columns(), tag + ": the same pivots and free columns");
        bool rows = De.rref_rows().size() == Sp.pivots().size();
        for (size_t r = 0; rows && r < De.rref_rows().size(); ++r) {
            const uint32_t pv = De.pivots()[r];
            rows = Sp.is_pivot(pv) && DenseOfSparseRow(Sp.D(), pv, Sp.tail_of(pv)) == De.rref_rows()[r];
        }
        g.check(rows, tag + ": every RREF row of Rel agrees entrywise (" + std::to_string(De.rref_rows().size()) + " rows)");
        std::ostringstream os;
        os << "NEB_SPARSE_EQ N=" << c.N << " chi=" << chi.label() << " ord=" << chi.order() << " n=" << c.n << " field=" << F.name()
           << " D=" << Sp.D() << " h0=" << Sp.h0() << " h1=" << Sp.h1() << " final_nnz=" << Sp.stats().final_nnz << " ops:";
        for (uint32_t p : {2u, 3u, 5u, 7u}) {
            if (p == c.ell) continue;
            const HeilbronnChi H(lvl, p, c.n, F, psi);
            g.check(Sp.RelEscapes(H) == 0, tag + ": " + H.label() + "(Rel) in Rel on every sparse generator (rmk:nebheilbronn)");
            g.check(FqEqual(De.OnH1(H), Sp.OnH1(H)), tag + ": " + H.label() + " on H^1 agrees entrywise on both routes");
            os << " " << H.label();
            if (p == 2 || (p == 3 && c.ell == 2)) {
                // the __int128 route of HeilbronnChi (taken on an int64 overflow, first at N = 30030) forced on every term:
                // the same terms, scalars and blocks, hence the same matrix
                hecke::level::ScopedForceWideHeilbronn force;
                const HeilbronnChi Hw(lvl, p, c.n, F, psi);
                bool same = Hw.wide_terms() == Hw.terms().size() && Hw.terms().size() == H.terms().size() && Hw.blocks() == H.blocks();
                for (size_t t = 0; same && t < H.terms().size(); ++t)
                    same = Hw.terms()[t].dst == H.terms()[t].dst && Hw.terms()[t].src == H.terms()[t].src &&
                           Hw.terms()[t].scal == H.terms()[t].scal && Hw.terms()[t].block == H.terms()[t].block;
                g.check(same && H.wide_terms() == 0, tag + ": " + H.label() + " on the forced __int128 route = the int64 route term for term");
            }
        }
        // a power operator on both routes (the direct double coset of determinant p^2; U_p^2 when p | N)
        const uint64_t p2 = c.ell == 2 ? 3 : 2;
        const HeilbronnChi H2 = HeilbronnChi::Power(lvl, p2, 2, c.n, F, psi);
        g.check(FqEqual(De.OnH1(H2), Sp.OnH1(H2)), tag + ": " + H2.label() + " (Mer_{p^2}) agrees entrywise on both routes");
        os << " " << H2.label();
        lines.push_back(os.str());
    }
    g.check(ran == kSpCellsRun, "(SP) " + std::to_string(kSpCellsRun) + " non-parity-zero overlap cells ran (" + std::to_string(ran) + ")");
    // The default route (auto) is the shipped dense route at D <= the cap.
    g.check(ResolveNebRoute(NebRoute::kAuto, 1010, 3) == NebRoute::kDense && ResolveNebRoute(NebRoute::kAuto, 1010, 4) == NebRoute::kSparse &&
                ResolveNebRoute(NebRoute::kSparse, 8, 0) == NebRoute::kSparse && ResolveNebRoute(NebRoute::kDense, 1u << 20, 4) == NebRoute::kDense,
            "(SP) route=auto is dense iff D <= " + std::to_string(kNebDenseCap) + " (the shipped route), sparse above");
}

// (PV) thm:nebrelations (2): T_{p^{nu+1}} = T_p T_{p^nu} - chi(p) p^{n+1} T_{p^{nu-1}} (p !| N), and (3) U_{p^nu} = U_p^nu
// (p | N), the left-hand side by the direct double coset (Merel's set of determinant p^nu, HeilbronnChi::Power).
void ArmPowers(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(PV) T_{p^nu}");
    struct PvCell { uint64_t N, o; uint32_t n, ell; };
    // six cells: d = 1 and d = 2, prime and composite N, odd and even n, chi(p) != 1 at several (cell, p).
    const std::vector<PvCell> cells{{7, 6, 1, 13}, {13, 12, 3, 5}, {35, 4, 3, 13}, {101, 2, 4, 7}, {5, 4, 5, 13}, {36, 6, 2, 7}};
    uint32_t chi_nontriv_seen = 0, control_differs = 0, control_cases = 0;
    for (const PvCell& c : cells) {
        const DirichletCharacter chi = PickCharacter(c.N, c.o, c.n);
        const uint32_t d = LeastDegree(c.ell, chi.order());
        const Fq F = CanonField(c.ell, d);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const NebPresentation P(lvl, c.n, F, psi);
        const std::string tag = "(PV) (" + chi.label() + ", n=" + std::to_string(c.n) + ", " + F.name() + ")";
        std::ostringstream os;
        os << "NEB_POWERS N=" << c.N << " chi=" << chi.label() << " n=" << c.n << " field=" << F.name() << " h1=" << P.h1();
        for (uint64_t p : {2ull, 3ull, 5ull}) {
            if (p == c.ell) continue;
            const uint32_t numax = p == 5 ? 2 : 3;
            std::vector<FqMat> T{FqIdentity(P.h1())};
            for (uint32_t nu = 1; nu <= numax; ++nu) T.push_back(P.OnH1(HeilbronnChi::Power(lvl, p, nu, c.n, F, psi)));
            g.check(FqEqual(T[1], P.OnH1(HeilbronnChi(lvl, p, c.n, F, psi))),
                    tag + ": Power(p, 1) = H_{N,p,chi} (p = " + std::to_string(p) + ")");
            if (c.N % p == 0) {
                for (uint32_t nu = 2; nu <= numax; ++nu)
                    g.check(FqEqual(T[nu], FqMul(F, T[nu - 1], T[1])),
                            tag + ": U_" + std::to_string(p) + "^" + std::to_string(nu) + " (Mer_{p^nu}) = (U_p)^nu (thm:nebrelations (3))");
                os << " U_" << p << ":nu<=" << numax;
                continue;
            }
            const uint32_t chip = psi.at(static_cast<int64_t>(p));
            if (chip != 1) ++chi_nontriv_seen;
            const uint32_t pn1 = F.pow(F.from_int(static_cast<int64_t>(p)), c.n + 1);
            const uint32_t c1 = F.mul(chip, pn1);   // chi(p) p^{n+1}
            for (uint32_t nu = 1; nu + 1 <= numax; ++nu) {
                const FqMat rec = FqAdd(F, FqMul(F, T[1], T[nu]), FqScale(F, T[nu - 1], F.neg(c1)));
                g.check(FqEqual(T[nu + 1], rec), tag + ": T_" + std::to_string(p) + "^" + std::to_string(nu + 1) +
                                                     " (Mer_{p^nu}) = T_p T_{p^nu} - chi(p) p^{n+1} T_{p^{nu-1}} (thm:nebrelations (2))");
                if (chip != 1 && P.h1() > 0 && pn1 != 0) {
                    // the negative control: the recursion with chi(p) dropped must fail
                    ++control_cases;
                    const FqMat bad = FqAdd(F, FqMul(F, T[1], T[nu]), FqScale(F, T[nu - 1], F.neg(pn1)));
                    if (!FqEqual(T[nu + 1], bad)) ++control_differs;
                }
            }
            os << " T_" << p << ":nu<=" << numax << ":chi(p)=" << F.text(chip);
        }
        lines.push_back(os.str());
    }
    g.check(chi_nontriv_seen >= 4, "(PV) chi(p) != 1 at " + std::to_string(chi_nontriv_seen) + " (cell, p) pairs (the twist is exercised)");
    g.check(control_cases > 0 && control_differs == control_cases,
            "(PV) negative control: the recursion WITHOUT chi(p) fails at every case with chi(p) != 1 (" + std::to_string(control_differs) +
                " / " + std::to_string(control_cases) + ")");
}

// ---------------------------------------------------------------------------
// (slot-18 WP-6) [C] sec:nebentypus code map (iv)-(vi): the character model over W_K(F_q) (dense and sparse
// Howell routes), its oracles, the model-route T_{p^2}, and the WP-6 audit minors.  Predictions:
// docs/implementation/slot18/WP-6_predictions.md (committed before the code).
// ---------------------------------------------------------------------------
using hecke::neb::NebWkCokernel;
using hecke::neb::WkRoute;
using hecke::neb::Wel;

bool SameHowell(const NebWkCokernel& A, const NebWkCokernel& B) {
    if (A.pivots() != B.pivots() || A.pivot_vals() != B.pivot_vals() || A.rows().size() != B.rows().size()) return false;
    for (size_t i = 0; i < A.rows().size(); ++i)
        if (A.rows()[i].col != B.rows()[i].col || A.rows()[i].val != B.rows()[i].val) return false;
    return true;
}
// The exponents truncated at K' (W/ell^e -> W/ell^min(e, K')), ascending (right exactness of the cokernel).
std::vector<uint32_t> Truncate(const std::vector<uint32_t>& ex, uint32_t Kp) {
    std::vector<uint32_t> o;
    for (uint32_t e : ex) o.push_back(std::min(e, Kp));
    std::sort(o.begin(), o.end());
    return o;
}
// K = 1: the W_1 Howell form against the F_q RREF of NebCokernel (codes: sum c_i ell^i), entrywise.
bool SameAsFieldRref(const NebWkCokernel& W1, const NebCokernel& De) {
    std::vector<size_t> ord(De.pivots().size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return De.pivots()[a] < De.pivots()[b]; });
    if (ord.size() != W1.rows().size()) return false;
    for (size_t r = 0; r < ord.size(); ++r) {
        if (W1.pivots()[r] != De.pivots()[ord[r]] || W1.pivot_vals()[r] != 0) return false;
        std::vector<uint32_t> v(W1.D(), 0);
        const auto& row = W1.rows()[r];
        for (size_t k = 0; k < row.nnz(); ++k) v[row.col[k]] = W1.ring().reduce(row.val[k]);
        if (v != De.rref_rows()[ord[r]]) return false;
    }
    return true;
}

// (GR) W_K(F_q) at d(chi) >= 2 (and two d = 1 controls), K <= 3, dense and sparse: (W1), (W2), (W5).
void ArmWk(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(GR) W_K(F_q)");
    struct WkCell { uint64_t N, o; uint32_t n, ell; };
    // d(chi) = 2 (N = 7, 13, 5, 21), 3 (N = 19: ord 9 at ell = 7), 4 (N = 11: ord 5 at ell = 7); d = 1 controls.
    const std::vector<WkCell> cells{{7, 6, 1, 5},  {13, 12, 3, 5}, {5, 4, 3, 7},   {21, 6, 2, 5},
                                    {19, 9, 2, 7}, {11, 5, 2, 7},  {13, 4, 3, 17}, {7, 2, 1, 13}};
    uint32_t dmax = 1, ran = 0;
    for (const WkCell& c : cells) {
        const DirichletCharacter chi = PickCharacter(c.N, c.o, c.n);
        const uint32_t d = LeastDegree(c.ell, chi.order());
        dmax = std::max(dmax, d);
        const Fq F = CanonField(c.ell, d);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const NebCokernel De(lvl, c.n, F, psi);
        const std::string tag = "(GR) (" + chi.label() + ", n=" + std::to_string(c.n) + ", " + F.name() + ")";
        g.check(!De.parity_zero(), tag + ": a non-zero cell");
        if (De.parity_zero()) continue;
        ++ran;
        std::ostringstream os;
        os << "NEB_WK N=" << c.N << " chi=" << chi.label() << " ord=" << chi.order() << " n=" << c.n << " field=" << F.name()
           << " d=" << d << " D=" << De.D() << " h1(F_q)=" << De.h1();
        std::vector<uint32_t> prev;
        for (uint32_t K = 1; K <= 3; ++K) {
            const NebWkCokernel Wd(lvl, c.n, F, K, psi, WkRoute::kDense);
            const NebWkCokernel Ws(lvl, c.n, F, K, psi, WkRoute::kSparse);
            const std::string tk = tag + " K=" + std::to_string(K);
            const std::string dd = Wd.HowellDefects(), ds = Ws.HowellDefects();
            g.check(dd.empty() && ds.empty(), tk + ": both routes are Howell forms (" + dd + ds + ")");
            g.check(SameHowell(Wd, Ws), tk + ": sparse = dense entrywise (" + std::to_string(Wd.rows().size()) + " rows, (W5))");
            g.check(Wd.rank_S() == De.dim_S() && Wd.rank_U() == De.dim_U() && Ws.rank_S() == De.dim_S() && Ws.rank_U() == De.dim_U(),
                    tk + ": rk M^S, rk M^ST over W_K = dim over F_q (free summands)");
            g.check(Wd.h1_exponents() == Ws.h1_exponents() && Wd.h1_exponents().size() == De.h1(),
                    tk + ": H^1 has h1(F_q) = " + std::to_string(De.h1()) + " invariant factors on both routes ((W2))");
            if (K == 1) g.check(SameAsFieldRref(Wd, De), tk + ": the W_1 Howell form is the F_q RREF entrywise ((W2))");
            else g.check(Truncate(Wd.h1_exponents(), K - 1) == prev, tk + ": truncated at K - 1 it is the K - 1 run ((W2), right exactness)");
            if (!chi.is_trivial())
                g.check(Wd.h1_free_rank() == De.h1() && Wd.h1_length() == static_cast<uint64_t>(K) * De.h1() && Wd.h0_length() == 0,
                        tk + ": chi != 1 -> H^1 = W_K^h1, H^0 = 0 ((W1))");
            prev = Wd.h1_exponents();
            os << " K" << K << ":len_rel=" << Wd.rel_length() << ",h1=" << VecText(Wd.h1_exponents()) << ",nnz=" << Ws.stats().final_nnz
               << ",sat=" << Wd.stats().saturation_rows << "/" << Ws.stats().saturation_rows;
        }
        lines.push_back(os.str());
    }
    g.check(ran == cells.size() && dmax >= 4, "(GR) " + std::to_string(ran) + " cells ran, d(chi) up to " + std::to_string(dmax));
    // The Teichmueller lift agrees with lem:teichmuller's GaloisRing (the slot-16 class) on one d = 2 field.
    {
        const Fq F = CanonField(5, 2);
        const hecke::neb::WkArith W(F, 3);
        const GaloisRing G(F, 3);
        bool same = true;
        for (uint32_t x = 1; x < F.q() && same; ++x) {
            const GaloisRing::W a = G.teichmuller(x);
            const Wel b = W.teichmuller(x);
            for (uint32_t i = 0; i < 2; ++i) same = same && a[i] == b[i];
            same = same && W.reduce(b) == x && W.mul(W.inv(b), b) == W.one();
        }
        g.check(same, "(GR) W_3(F_25): t(x) = GaloisRing::teichmuller(x), t(x) = x mod 5, unit inverse, for every x != 0");
    }
    // A synthetic Howell case with torsion: <(ell, 1)> in W_3(F_25)^2 -> the saturation (0, ell^2); the cokernel is W (free).
    {
        const Fq F = CanonField(5, 2);
        const hecke::neb::WkArith W(F, 3);
        for (WkRoute r : {WkRoute::kDense, WkRoute::kSparse}) {
            std::vector<hecke::neb::WkRow> gens(1);
            gens[0].col = {0, 1};
            gens[0].val = {W.scalar(5), W.one()};
            std::vector<hecke::neb::WkRow> rows;
            std::vector<uint32_t> piv, pval;
            hecke::neb::WkHowellForm(W, 2, gens, r, rows, piv, pval);
            const std::vector<uint32_t> ex = hecke::neb::WkCokernelExponents(W, 2, rows, piv, pval);
            g.check(piv == std::vector<uint32_t>{0, 1} && pval == std::vector<uint32_t>{1, 2} && ex == std::vector<uint32_t>{3},
                    "(GR) synthetic <(5, 1)> in W_3^2 (" + hecke::neb::WkRouteName(r) + "): Howell pivots 5, 25 (saturation); W^2/Rel = W");
        }
    }
}

// (GT) the Gamma_1 oracle at d = 2 ((W4), prop:nebdecomp (3)) and the Z/ell^K model route at d = 1 ((W3), (NB2) over Z/ell^K).
void ArmWkOracles(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(GT) W_K oracles");
    // (W4): H^1(Gamma_1(N); Z/ell^K) exponents = the union over all chi of the W_K(F_{ell^2}) exponents.
    struct G1 { uint64_t N; uint32_t n, ell, K; };
    for (const G1& c : std::vector<G1>{{5, 8, 7, 2}, {7, 2, 5, 3}, {5, 3, 7, 2}}) {
        const Fq F = CanonField(c.ell, 2);
        const hecke::level::Gamma0LevelN lvl(c.N);
        std::vector<uint32_t> uni;
        int64_t h0len = 0;
        uint32_t chars = 0, nonzero = 0;
        for (const DirichletCharacter& chi : AllCharacters(c.N)) {
            ++chars;
            const CharacterValues psi = ReduceCharacter(chi, F);
            const NebWkCokernel W(lvl, c.n, F, c.K, psi, WkRoute::kSparse);
            if (W.parity_zero()) continue;
            ++nonzero;
            uni.insert(uni.end(), W.h1_exponents().begin(), W.h1_exponents().end());
            h0len += W.h0_length();
        }
        std::sort(uni.begin(), uni.end());
        const engine::ModCtx<uint32_t> rctx = engine::ModCtx<uint32_t>::prime_power(c.ell, c.K);
        std::vector<uint32_t> table(c.N, 1);
        ModelCell mc = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new Gamma1LevelN(c.N)), c.n, rctx, TableScalar(c.N, table));
        hecke::ScopedCoefficientRing ring(c.K);
        std::vector<uint32_t> g1 = mc.H->H_invariant_factors(1);
        std::sort(g1.begin(), g1.end());
        const std::vector<uint32_t> len = mc.H->H_lengths();
        // H(Gamma_1; W_K) = H(Gamma_1; Z/ell^K) (x) W_K: as Z/ell^K-modules both sides are d-th powers, so the multisets agree.
        g.check(g1 == uni && len.size() >= 1 && static_cast<int64_t>(len[0]) == h0len,
                "(GT) Gamma_1(" + std::to_string(c.N) + "), n=" + std::to_string(c.n) + ", Z/" + std::to_string(c.ell) + "^" +
                    std::to_string(c.K) + ": H^1 exponents " + VecText(g1) + " = union over " + std::to_string(chars) +
                    " characters of W_K(" + F.name() + ") " + VecText(uni) + "; len H^0 " + std::to_string(len.empty() ? 0 : len[0]) +
                    " = " + std::to_string(h0len) + " ((W4))");
        std::ostringstream os;
        os << "NEB_WK_GAMMA1 N=" << c.N << " n=" << c.n << " ring=Z/" << c.ell << "^" << c.K << " field=" << F.name()
           << " chars=" << chars << " nonzero=" << nonzero << " h1_exps=" << VecText(g1) << " h0_len=" << h0len;
        lines.push_back(os.str());
    }
    // (W3): d = 1, the model route over Z/ell^K with t o chi as its table, against the W_K cokernel route.
    struct M1 { uint64_t N, o; uint32_t n, ell, K; };
    uint32_t torsion_cells = 0;
    for (const M1& c : std::vector<M1>{{1, 1, 6, 5, 3}, {7, 1, 6, 5, 2}, {5, 1, 8, 7, 2}, {13, 4, 3, 17, 3}, {7, 6, 1, 13, 2}}) {
        const DirichletCharacter chi = PickCharacter(c.N, c.o, c.n);
        const Fq F = Fq::Prime(c.ell);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const NebCokernel De(lvl, c.n, F, psi);
        const NebWkCokernel W(lvl, c.n, F, c.K, psi, WkRoute::kDense);
        const NebWkCokernel Ws(lvl, c.n, F, c.K, psi, WkRoute::kSparse);
        const GaloisRing G(F, c.K);
        const std::vector<GaloisRing::W> lt = LiftCharacter(psi, G);
        std::vector<uint32_t> table(std::max<uint64_t>(c.N, 1), 0);
        for (size_t a = 0; a < table.size(); ++a) table[a] = c.N == 1 ? 1u : static_cast<uint32_t>(lt[a][0]);
        const engine::ModCtx<uint32_t> rctx = engine::ModCtx<uint32_t>::prime_power(c.ell, c.K);
        ModelCell mc = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new hecke::level::Gamma0LevelN(c.N)), c.n, rctx,
                                  TableScalar(std::max<uint64_t>(c.N, 1), table));
        hecke::ScopedCoefficientRing ring(c.K);
        std::vector<uint32_t> inv = mc.H->H_invariant_factors(1);
        std::sort(inv.begin(), inv.end());
        const std::vector<uint32_t> len = mc.H->H_lengths();
        const std::string tag = "(GT) (" + chi.label() + ", n=" + std::to_string(c.n) + ", Z/" + std::to_string(c.ell) + "^" + std::to_string(c.K) + ")";
        g.check(inv == W.h1_exponents() && SameHowell(W, Ws) && len.size() >= 2 && static_cast<int64_t>(len[0]) == W.h0_length() &&
                    len[1] == W.h1_length(),
                tag + ": model route over Z/ell^K H^1 " + VecText(inv) + " = cokernel route over W_K " + VecText(W.h1_exponents()) +
                    "; len H^0 " + std::to_string(len.empty() ? 0 : len[0]) + " = " + std::to_string(W.h0_length()) + " ((W3))");
        const uint32_t torsion = static_cast<uint32_t>(W.h1_exponents().size()) - W.h1_free_rank();
        if (torsion > 0) ++torsion_cells;
        g.check(torsion == De.h0(), tag + ": the non-free summands of H^1 number h0(F_ell) = " + std::to_string(De.h0()) + " ((W1)'s mechanism)");
        std::ostringstream os;
        os << "NEB_WK_MODEL N=" << c.N << " chi=" << chi.label() << " n=" << c.n << " ring=Z/" << c.ell << "^" << c.K
           << " h0(F)=" << De.h0() << " h1(F)=" << De.h1() << " h1_exps=" << VecText(W.h1_exponents()) << " h0_len=" << W.h0_length();
        lines.push_back(os.str());
    }
    g.check(torsion_cells >= 2, "(GT) torsion (W/ell^e, 0 < e < K) is exercised on " + std::to_string(torsion_cells) + " cells");
}

// (T2) [C] code map (v): the model-route T_{p^2} = [Gamma_0(N) diag(1, p^2) Gamma_0(N)]_chi + chi(p) p^n through the
// shipped Gamma0LevelN table (Gamma0SubTransversalN to level N p^2) and the level-1 tables of diag(1, p^2); against the
// recursion of thm:nebrelations (2) entrywise on the model's H^1, and against Mer_{p^2} (HeilbronnChi::Power) by charpoly.
HeckeDatum MakeDatumR(uint64_t p, uint32_t r, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    HeckeDatum d;
    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
    d.cosets.reset(new hecke::HeckeCosetTables(p, r, coef, n));
    d.tau.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *d.cosets, coef, n));
    return d;
}

void ArmModelTp2(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(T2) model-route T_{p^2}");
    struct TC { uint64_t N, o; uint32_t n, ell; };
    const std::vector<TC> cells{{7, 6, 1, 13}, {35, 4, 3, 13}, {5, 4, 5, 13}, {36, 6, 2, 7}, {13, 6, 2, 7}, {1, 1, 10, 11}};
    uint32_t pairs = 0, chi_nontriv = 0, control = 0, control_differs = 0;
    for (const TC& c : cells) {
        const DirichletCharacter chi = PickCharacter(c.N, c.o, c.n);
        const Fq F = Fq::Prime(c.ell);
        const CharacterValues psi = ReduceCharacter(chi, F);
        const engine::ModCtx<uint32_t> ctx(c.ell);
        const hecke::level::Gamma0LevelN lvl(c.N);
        const BlockScalar s = CharacterScalar(psi);
        ModelCell mc = BuildModel(std::unique_ptr<hecke::level::CosetLevel>(new hecke::level::Gamma0LevelN(c.N)), c.n, ctx, s);
        const NebCokernel ck(lvl, c.n, F, psi);
        const hecke::level::HFrame f(mc.H.get(), 1);
        const std::string tag = "(T2) (" + chi.label() + ", n=" + std::to_string(c.n) + ", F_" + std::to_string(c.ell) + ")";
        std::ostringstream os;
        os << "NEB_TP2 N=" << c.N << " chi=" << chi.label() << " n=" << c.n << " ell=" << c.ell << " h1=" << ck.h1();
        for (uint64_t p : {2ull, 3ull, 5ull}) {
            if (c.N % p == 0 || p == c.ell) continue;
            if (p == 5 && c.N % 2 != 0 && c.N % 3 != 0) continue;   // p = 5 only where 2 or 3 divides N
            ++pairs;
            const HeckeDatum d1 = MakeDatumR(p, 1, c.ell, c.n);
            const FqMat Tp = FromHMat(ModelHeckeChi(mc, lvl, static_cast<uint32_t>(p), c.n, ctx, s, d1));
            const HeckeDatum d2 = MakeDatumR(p, 2, c.ell, c.n);
            const hecke::level::Gamma0LevelN lvlNp2(std::max<uint64_t>(c.N, 1) * p * p);
            hecke::level::CorrespondenceSpec spec{mc.lvl.get(), mc.lvl.get(), d2.cosets.get(), d2.tau.get(),
                                                  hecke::level::Gamma0SubTransversalN(lvl, lvlNp2), "[diag(1,p^2)]"};
            hecke::level::VerifyTransversal(spec, static_cast<uint32_t>(p * (p + 1)));
            const FqMat P2 = FromHMat(hecke::level::InducedOnH(f, f, BuildLevelCorrespondenceChi(ctx, spec, *TheModel().K, c.n, 1, s), spec.label));
            const uint32_t chip = psi.at(static_cast<int64_t>(p));
            if (chip != 1) ++chi_nontriv;
            const uint32_t pn = F.pow(F.from_int(static_cast<int64_t>(p)), c.n);
            const uint32_t cpn = F.mul(chip, pn);   // T(p, p) = chi(p) p^n (prop:nebcorr (3))
            const FqMat full = FqAdd(F, P2, FqScale(F, FqIdentity(P2.rows), cpn));
            const FqMat rec = FqAdd(F, FqMul(F, Tp, Tp), FqScale(F, FqIdentity(Tp.rows), F.neg(F.mul(cpn, F.from_int(static_cast<int64_t>(p))))));
            const std::string ps = std::to_string(p);
            g.check(FqEqual(full, rec), tag + ": [diag(1," + ps + "^2)]_chi + chi(p) p^n = T_" + ps +
                                            "^2 - chi(p) p^{n+1} on the model's H^1, entrywise (thm:nebrelations (2))");
            const std::vector<uint32_t> cm = FqCharpoly(F, full);
            const std::vector<uint32_t> cc = FqCharpoly(F, ck.OnH1(HeilbronnChi::Power(lvl, p, 2, c.n, F, psi)));
            g.check(cm == cc, tag + ": charpoly of the model-route T_" + ps + "^2 " + VecText(cm) + " = H_{N,p^2,chi} (Mer_{p^2}) " + VecText(cc));
            if (cpn != 0 && ck.h1() > 0) {
                ++control;
                if (!FqEqual(P2, rec)) ++control_differs;
            }
            os << " T_" << p << "^2=" << VecText(cm) << ":chi(p)=" << chip;
        }
        lines.push_back(os.str());
    }
    g.check(pairs >= 6 && chi_nontriv >= 3, "(T2) " + std::to_string(pairs) + " (cell, p) pairs, chi(p) != 1 at " + std::to_string(chi_nontriv));
    g.check(control > 0 && control_differs == control,
            "(T2) negative control: the double coset alone ([diag(1,p^2)] without T(p,p)) fails the recursion at every case (" +
                std::to_string(control_differs) + " / " + std::to_string(control) + ")");
}

// (A1) the slot-17 WP-6 audit MINOR-2: a natural (unforced) __int128 escalation of HeilbronnChi in the default run.
void ArmWideNatural(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(A1) natural __int128");
    const uint64_t N = 30030;
    const hecke::level::Gamma0LevelN lvl(N);
    const Fq F = Fq::Prime(19);
    const CharacterValues psi = ReduceCharacter(DirichletCharacter::Trivial(N), F);
    const HeilbronnChi H(lvl, 3, 0, F, psi);
    hecke::level::ScopedForceWideHeilbronn force;
    const HeilbronnChi Hw(lvl, 3, 0, F, psi);
    bool same = Hw.terms().size() == H.terms().size() && Hw.blocks() == H.blocks() && Hw.wide_terms() == Hw.terms().size();
    for (size_t t = 0; same && t < H.terms().size(); ++t)
        same = Hw.terms()[t].dst == H.terms()[t].dst && Hw.terms()[t].src == H.terms()[t].src && Hw.terms()[t].scal == H.terms()[t].scal &&
               Hw.terms()[t].block == H.terms()[t].block;
    g.check(H.wide_terms() > 0 && H.wide_terms() < H.terms().size() && same,
            "(A1) U_3 at N = 30030, n = 0: " + std::to_string(H.wide_terms()) + " of " + std::to_string(H.terms().size()) +
                " terms escalate to __int128 unforced, and the result equals the forced-wide build term for term");
    lines.push_back("NEB_WIDE_NATURAL N=30030 op=U_3 terms=" + std::to_string(H.terms().size()) + " wide=" + std::to_string(H.wide_terms()));
}

// One cell on a chosen route, with the pre-run RSS prediction (rule 19): the hook's cells and (SL).  Returns 1 if it ran,
// 0 if refused by design (reported, never failed).
uint32_t RouteCell(Checker& g, const std::string& arm, uint64_t N, const DirichletCharacter& chi, uint32_t n, uint32_t ell, uint32_t d_given,
                   NebRoute route, double rss_cap_gib, std::vector<std::string>* lines) {
    const uint64_t m = hecke::level::IndexGamma0(N);
    const uint64_t D64 = m * (static_cast<uint64_t>(n) + 1);
    const NebRoute r = ResolveNebRoute(route, m, n);
    std::ostringstream os;
    os << "NEB_CELL N=" << N << " chi=" << chi.label() << " ord=" << chi.order() << " k=" << n + 2 << " ell=" << ell
       << " route=" << NebRouteName(r) << (route == NebRoute::kAuto ? "(auto)" : "") << " D=" << D64;
    auto report = [&](const std::string& s) {
        std::cout << "  " << s << std::endl;
        if (lines) lines->push_back(s);
    };
    if (r == NebRoute::kDense && D64 > kNebDenseCap) {
        os << " refused=neb_dense_cap";
        report(os.str());
        return 0;
    }
    // the predicted bytes: the sparse fill model of rmk:levelcost (4) (the character costs nothing structurally,
    // rmk:nebcost (1)) plus the operator matrices on H^1 (h1 ~ D / 6, at most four, kept only when h1 <= 1500).
    // Calibrated on the WP-6 ladder: at 4620:k=6 and 2310:k=12 the measured peak was 4.3x and 5.2x the bare
    // sparse_bytes (final fill 4-5x the fitted law at these shapes, ~22 B per stored entry); the factor 10 covers it.
    const hecke::level::CokernelCost cm = hecke::level::CokernelCostModel(N, n);
    const double h1_est = static_cast<double>(D64) / 6.0 + 2.0 * (n + 1);
    const double pred_bytes = (r == NebRoute::kSparse ? 10.0 * cm.sparse_bytes : cm.dense_bytes) +
                              (h1_est <= 1500 ? 4.0 * 4.0 * h1_est * h1_est : 0.0) + 64.0 * D64;
    os << " pred_rss_mib=" << static_cast<uint64_t>(pred_bytes / 1048576.0);
    if (pred_bytes > rss_cap_gib * 1073741824.0) {
        os << " refused=rss_model(>" << rss_cap_gib << "GiB)";
        report(os.str());
        return 0;
    }
    oc_util::ResetPeakRss();
    const double rss_base = oc_util::CurrentRssMiB();
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<Fq> Fp;
    std::unique_ptr<CharacterValues> psip;
    std::unique_ptr<NebPresentation> Pp;
    const hecke::level::Gamma0LevelN lvl(N);
    uint32_t d = 0;
    try {
        d = d_given ? d_given : (chi.order() % ell == 0 ? 1 : LeastDegree(ell, chi.order()));
        Fp.reset(new Fq(CanonField(ell, d)));
        psip.reset(new CharacterValues(ReduceCharacter(chi, *Fp)));
        Pp.reset(new NebPresentation(lvl, n, *Fp, *psip, r));
    } catch (const NebentypusRefusal& e) {
        os << " d=" << d << " refused=" << e.code();
        report(os.str());
        return 0;
    }
    const Fq& F = *Fp;
    const CharacterValues& psi = *psip;
    const NebPresentation& P = *Pp;
    const double t_pres = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const EulerPrediction E = EulerCharacteristicFormula(N, n, chi);
    const std::string tag = arm + " (" + chi.label() + ", k=" + std::to_string(n + 2) + ", " + F.name() + ", " + NebRouteName(r) + ")";
    os << " d=" << d << " s=" << F.text(psi.s);
    if (P.parity_zero()) {
        g.check(!E.parity_ok, tag + ": parity_zero agrees with the formula");
        os << " parity_zero h0=0 h1=0";
        report(os.str());
        return 1;
    }
    g.check(static_cast<int64_t>(P.h1()) - static_cast<int64_t>(P.h0()) == E.h1_minus_h0,
            tag + ": (NB1) h1 - h0 = " + std::to_string(E.h1_minus_h0));
    g.check(static_cast<int64_t>(P.dim_S()) == E.dim_S && static_cast<int64_t>(P.dim_U()) == E.dim_U,
            tag + ": dim M^S, dim M^ST = prop:neborbit (1)'s orbit counts");
    os << " h0=" << P.h0() << " h1=" << P.h1();
    if (P.sparse()) os << " final_nnz=" << P.sparse()->stats().final_nnz << " merge_ops=" << P.sparse()->stats().merge_ops;
    const bool full = P.h1() <= 1500;
    std::map<uint32_t, FqMat> ops;
    std::vector<std::unique_ptr<HeilbronnChi>> Hs;
    for (uint32_t p : {2u, 3u, 5u, 7u}) {
        if (p == ell) continue;
        Hs.emplace_back(new HeilbronnChi(lvl, p, n, F, psi));
        if (Hs.back()->wide_terms()) os << " wide_" << Hs.back()->label() << "=" << Hs.back()->wide_terms();
        if (full) {
            ops[p] = P.OnH1(*Hs.back());   // checks H(Rel) in Rel first (both routes)
            if (P.h1() <= 400) os << " " << Hs.back()->label() << "=" << FqPolyText(F, FqCharpoly(F, ops[p]));
        } else {
            g.check(P.sparse()->RelEscapes(*Hs.back()) == 0, tag + ": " + Hs.back()->label() + "(Rel) in Rel on every generator");
        }
    }
    if (full) {
        // the operators' fingerprint (FNV-1a-64 over p, h1 and every entry): route=dense and route=sparse runs compare by it
        uint64_t fp = engine::kFnvOffset64;
        for (const auto& kv : ops) {
            fp = engine::FnvU64(fp, kv.first);
            fp = engine::FnvU64(fp, kv.second.rows);
            for (uint32_t v : kv.second.a) fp = engine::FnvU64(fp, v);
        }
        char buf[32];
        std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(fp));
        os << " ops_fnv=0x" << buf;
        // A B e_j = B A e_j on every column (h1 <= 200) or on 32 spread columns (O(h1^2) each, no h1^3 product)
        const uint32_t h = P.h1(), k = h <= 200 ? h : 32;
        auto col_times = [&](const FqMat& A, const FqMat& B, uint32_t j) {   // A (B e_j)
            std::vector<uint32_t> y(h, 0);
            for (uint32_t t = 0; t < h; ++t) {
                const uint32_t b = B.at(t, j);
                if (!b) continue;
                for (uint32_t i = 0; i < h; ++i)
                    if (A.at(i, t)) y[i] = F.add(y[i], F.mul(A.at(i, t), b));
            }
            return y;
        };
        bool comm = true;
        for (const auto& a : ops)
            for (const auto& b : ops)
                if (a.first < b.first)
                    for (uint32_t s = 0; comm && s < k; ++s) {
                        const uint32_t j = static_cast<uint32_t>(static_cast<uint64_t>(s) * h / std::max<uint32_t>(k, 1));
                        comm = col_times(a.second, b.second, j) == col_times(b.second, a.second, j);
                    }
        g.check(comm, tag + ": the operators commute (thm:nebrelations (1)) on " + std::to_string(k) + " columns");
    } else {
        const NebSparseCokernel& S = *P.sparse();
        bool comm = true;
        const uint32_t k = std::min<uint32_t>(32, S.h1());
        for (size_t i = 0; i + 1 < Hs.size(); ++i) {
            const HeilbronnChiSparse A(*Hs[i], S.D()), B(*Hs[i + 1], S.D());
            for (uint32_t j = 0; comm && j < k; ++j) {
                SparseVec e;
                e.col.push_back(S.free_columns()[static_cast<size_t>(j) * S.h1() / k]);
                e.val.push_back(1);
                comm = S.coords(A.apply(F, B.apply(F, e))) == S.coords(B.apply(F, A.apply(F, e)));
            }
        }
        g.check(comm, tag + ": the operators commute mod Rel on " + std::to_string(k) + " basis vectors (thm:nebrelations (1))");
        os << " ops=sampled";
    }
    os << " presentation_s=" << t_pres << " wall_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
       << " rss_base_mib=" << rss_base << " peak_rss_mib=" << oc_util::PeakRssMiB();
    report(os.str());
    return 1;
}

// (SL) one cell beyond the dense cap on the sparse route in the default run: (4099, order 3, n = 0) over F_7,
// D = 4100 > 4096 (predicted well under 100 MiB).
void ArmSparseLadder(Checker& g, std::vector<std::string>& lines) {
    PhaseTimer pt("(SL) beyond the dense cap");
    const DirichletCharacter chi = PickCharacter(4099, 3, 0);
    g.check(hecke::level::IndexGamma0(4099) > kNebDenseCap, "(SL) the cell (4099, n = 0) lies beyond the dense cap");
    g.check(RouteCell(g, "(SL)", 4099, chi, 0, 7, 0, NebRoute::kAuto, 1.0, &lines) == 1, "(SL) the cell ran on the sparse route");
}

// ---------------------------------------------------------------------------
// The env hook HECKE_NEBENTYPUS_CELLS="N:chi:k:ell[:d],..."
// ---------------------------------------------------------------------------
struct HookCell { uint64_t N, m; uint32_t k, ell, d; NebRoute route; uint32_t K; };
// S16 WP-10/11 repairs (audit finding 5): every field is parsed as a whole decimal string and range-checked BEFORE any
// narrowing (no silent uint32 truncation, no accepted prefix "13abc"); k - 1 = n + 1 <= kNebDenseCap; an empty list (or
// one of only separators) is refused.  Slot-17 WP-6: an optional last field route=dense|sparse|auto (default auto: the
// shipped dense route at D <= kNebDenseCap, the sparse route above; a dense cell above the cap stays refused by name).
// Slot-18 WP-6: an optional trailing field K=<1..3> (in either order with route=) runs the cell over W_K(F_q) on the
// sparse Howell route instead, against the F_q sparse route (K = 1: entrywise; K >= 2: the generator count and (W1)).
std::vector<HookCell> ParseHook(const std::string& s) {
    std::vector<HookCell> out;
    size_t pos = 0;
    auto whole = [](const std::string& f, uint64_t& v) -> bool {
        if (f.empty() || f.size() > 18) return false;
        for (char ch : f) if (ch < '0' || ch > '9') return false;
        v = std::stoull(f);
        return true;
    };
    for (const std::string& raw : SplitStr(s, ',')) {
        ++pos;
        const std::string t = Trim(raw);
        if (t.empty()) continue;
        std::vector<std::string> f = SplitStr(t, ':');
        auto refuse = [&](const std::string& why) {
            throw std::invalid_argument("HECKE_NEBENTYPUS_CELLS: cell " + std::to_string(pos) + " (\"" + t + "\"): " + why);
        };
        HookCell c{};
        c.route = NebRoute::kAuto;
        c.K = 0;
        bool seen_route = false, seen_K = false;
        while (!f.empty() && (Trim(f.back()).rfind("route=", 0) == 0 || Trim(f.back()).rfind("K=", 0) == 0)) {
            const std::string last = Trim(f.back());
            if (last.rfind("route=", 0) == 0) {
                if (seen_route) refuse("route= given twice");
                seen_route = true;
                const std::string r = last.substr(6);
                if (r != "dense" && r != "sparse" && r != "auto") refuse("route= must be dense, sparse or auto");
                c.route = ParseNebRoute(r, "route");
            } else {
                if (seen_K) refuse("K= given twice");
                seen_K = true;
                uint64_t K = 0;
                if (!whole(last.substr(2), K) || K < 1 || K > 3) refuse("K= must be 1, 2 or 3 (W_K(F_q), slot-18 WP-6)");
                c.K = static_cast<uint32_t>(K);
            }
            f.pop_back();
        }
        if (seen_K && seen_route) refuse("K= runs the W_K sparse route; route= does not apply");
        if (f.size() != 4 && f.size() != 5) refuse("expected N:chi:k:ell[:d][:route=dense|sparse|auto][:K=1|2|3]");
        uint64_t N = 0, m = 0, k = 0, ell = 0, d = 0;
        if (!whole(Trim(f[0]), N) || !whole(Trim(f[1]), m) || !whole(Trim(f[2]), k) || !whole(Trim(f[3]), ell) ||
            (f.size() == 5 && !whole(Trim(f[4]), d)))
            refuse("a field is not a non-negative decimal integer");
        if (N < 1 || N >= (1ull << 20)) refuse("N in [1, 2^20)");
        if (m < 1 || (N > 1 && m >= N) || (N == 1 && m != 1)) refuse("chi: the Conrey index m in [1, N) (m = 1 at N = 1)");
        if (k < 2 || k > static_cast<uint64_t>(kNebDenseCap) + 1) refuse("k in [2, " + std::to_string(kNebDenseCap + 1) + "] (n + 1 <= the dense cap)");
        if (ell >= (1ull << 31)) refuse("ell < 2^31");
        if (f.size() == 5 && (d < 1 || d > 64)) refuse("d in [1, 64]");
        c.N = N;
        c.m = m;
        c.k = static_cast<uint32_t>(k);
        c.ell = static_cast<uint32_t>(ell);
        c.d = static_cast<uint32_t>(d);
        if (c.N > 1 && gcd<uint64_t>(c.m % c.N, c.N) != 1) refuse("chi: the Conrey index m must be a unit mod N");
        if (!IsPrime(c.ell) || c.ell < 5) refuse("ell a prime >= 5");
        if (c.N % c.ell == 0) refuse("ell !| N (rmk:nebcheckable's cells)");
        out.push_back(c);
    }
    if (out.empty()) throw std::invalid_argument("HECKE_NEBENTYPUS_CELLS: the list is empty (\"" + s + "\"); unset the variable for the default arms");
    return out;
}

// A cell above the 8 GiB RSS model is refused before the run (rule 19; no exclusive memory window is assumed).
constexpr double kHookRssCapGiB = 8.0;

// (slot-18 WP-6, (A2)) one hook cell over W_K(F_q) on the sparse Howell route, against the F_q sparse route: the
// second route at a ladder cell beyond the dense reach (the slot-17 WP-6 audit MINOR-2).  RSS predicted first (rule 19).
uint32_t WkHookCell(Checker& g, const HookCell& c, const DirichletCharacter& chi) {
    const uint32_t n = c.k - 2;
    const uint64_t m = hecke::level::IndexGamma0(c.N);
    const uint64_t D64 = m * (static_cast<uint64_t>(n) + 1);
    std::ostringstream os;
    os << "NEB_WK_CELL N=" << c.N << " chi=" << chi.label() << " ord=" << chi.order() << " k=" << c.k << " ell=" << c.ell << " K=" << c.K
       << " route=wk_sparse+fq_sparse D=" << D64;
    // both sparse presentations alive at once: the F_q model (10 x sparse_bytes, the slot-17 calibration) and the W_K rows
    // (36 B per stored entry against ~22 B on the F_q route: x 2 to be safe).
    const hecke::level::CokernelCost cm = hecke::level::CokernelCostModel(c.N, n);
    const double pred_bytes = 30.0 * cm.sparse_bytes + 256.0 * D64;
    os << " pred_rss_mib=" << static_cast<uint64_t>(pred_bytes / 1048576.0);
    if (pred_bytes > kHookRssCapGiB * 1073741824.0) {
        os << " refused=rss_model(>" << kHookRssCapGiB << "GiB)";
        std::cout << "  " << os.str() << std::endl;
        return 0;
    }
    oc_util::ResetPeakRss();
    const double rss_base = oc_util::CurrentRssMiB();
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t d = c.d ? c.d : (chi.order() % c.ell == 0 ? 1 : LeastDegree(c.ell, chi.order()));
    std::unique_ptr<Fq> Fp;
    std::unique_ptr<CharacterValues> psip;
    try {
        Fp.reset(new Fq(CanonField(c.ell, d)));
        psip.reset(new CharacterValues(ReduceCharacter(chi, *Fp)));
    } catch (const NebentypusRefusal& e) {
        os << " d=" << d << " refused=" << e.code();
        std::cout << "  " << os.str() << std::endl;
        return 0;
    }
    const Fq& F = *Fp;
    const hecke::level::Gamma0LevelN lvl(c.N);
    const NebSparseCokernel Sp(lvl, n, F, *psip, false);
    const double t_fq = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    std::unique_ptr<NebWkCokernel> Wp;
    try {
        Wp.reset(new NebWkCokernel(lvl, n, F, c.K, *psip, WkRoute::kSparse, false));
    } catch (const NebentypusRefusal& e) {
        os << " d=" << d << " refused=" << e.code();
        std::cout << "  " << os.str() << std::endl;
        return 0;
    }
    const NebWkCokernel& W = *Wp;
    const double t_wk = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const std::string tag = "(H) (" + chi.label() + ", k=" + std::to_string(c.k) + ", " + F.name() + ", K=" + std::to_string(c.K) + ")";
    os << " d=" << d;
    g.check(W.parity_zero() == Sp.parity_zero(), tag + ": parity_zero agrees with the F_q sparse route");
    if (W.parity_zero()) {
        os << " parity_zero";
        std::cout << "  " << os.str() << std::endl;
        return 1;
    }
    g.check(W.h1_exponents().size() == Sp.h1(), tag + ": H^1 over W_K has h1(F_q) = " + std::to_string(Sp.h1()) + " invariant factors");
    if (!chi.is_trivial())
        g.check(W.h1_free_rank() == Sp.h1(), tag + ": chi != 1 -> H^1 = W_K^h1 ((W1))");
    if (c.K == 1) {
        bool same = W.pivots() == Sp.pivots();
        size_t bad_row = 0;
        for (size_t r = 0; same && r < W.rows().size(); ++r) {
            const auto& row = W.rows()[r];
            const SparseVec& t = Sp.tail_of(W.pivots()[r]);
            same = row.nnz() == t.col.size() + 1 && W.pivot_vals()[r] == 0;
            for (size_t k = 0; same && k < t.col.size(); ++k)
                same = row.col[k + 1] == t.col[k] && W.ring().reduce(row.val[k + 1]) == t.val[k];
            if (!same) bad_row = r;
        }
        g.check(same, tag + ": the W_1 sparse Howell form = the F_q sparse RREF entrywise (" + std::to_string(W.rows().size()) + " rows" +
                          (same ? std::string("") : ", first difference at row " + std::to_string(bad_row)) + ")");
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    os << " h1(F_q)=" << Sp.h1() << " h1_free=" << W.h1_free_rank() << " h1_exps=" << W.h1_exponents().size()
       << " fq_final_nnz=" << Sp.stats().final_nnz << " wk_final_nnz=" << W.stats().final_nnz << " wk_displaced=" << W.stats().displaced
       << " wk_sat=" << W.stats().saturation_rows << " fq_s=" << t_fq << " wk_s=" << t_wk << " wall_s=" << wall
       << " peak_rss_mib=" << static_cast<uint64_t>(oc_util::PeakRssMiB()) << " base_rss_mib=" << static_cast<uint64_t>(rss_base);
    std::cout << "  " << os.str() << std::endl;
    return 1;
}

void ArmHook(Checker& g, const std::vector<HookCell>& cells) {
    PhaseTimer pt("(H) HECKE_NEBENTYPUS_CELLS");
    uint32_t refused = 0, ran = 0;
    for (const HookCell& c : cells) {
        const DirichletCharacter chi = DirichletCharacter::Conrey(c.N, c.m);
        if (c.K > 0) {
            if (WkHookCell(g, c, chi)) ++ran;
            else ++refused;
            continue;
        }
        if (RouteCell(g, "(H)", c.N, chi, c.k - 2, c.ell, c.d, c.route, kHookRssCapGiB, nullptr)) ++ran;
        else ++refused;
    }
    std::cout << "  NEB_CELLS_SUMMARY cells=" << cells.size() << " ran=" << ran << " refused=" << refused << std::endl;
    g.check(ran > 0, "(H) at least one listed cell ran (" + std::to_string(cells.size()) + " listed, all refused: the hook checked nothing)");
}

} // namespace

uint32_t HeckeNebentypusValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    std::cout << "=== HeckeNebentypusValidation (slot-16 WP-10: nebentypus on the fixed model by Shapiro; "
                 "[C] sec:nebentypus, rmk:nebcheckable (NB1)-(NB3), (NB5), (NB6); [GT, conv:nebentypus]) ===" << std::endl;
    Checker g;
    if (const char* env = std::getenv("HECKE_NEBENTYPUS_CELLS")) {
        const std::vector<HookCell> cells = ParseHook(env);
        std::map<std::string, uint32_t> routes;
        for (const HookCell& c : cells)
            ++routes[c.K ? "wk_sparse_K" + std::to_string(c.K)
                         : NebRouteName(ResolveNebRoute(c.route, hecke::level::IndexGamma0(c.N), c.k - 2))];
        std::string rtext;
        for (const auto& kv : routes) rtext += (rtext.empty() ? "" : ",") + kv.first + ":" + std::to_string(kv.second);
        std::cout << "  EFFECTIVE HeckeNebentypusValidation arm=CELLS cells=" << cells.size() << " route=" << rtext
                  << " (HECKE_NEBENTYPUS_CELLS; replaces the default arms)" << std::endl;
        ArmHook(g, cells);
    } else {
        std::vector<std::string> lines;
        ArmField(g);
        std::cout << "  [(F) done at " << secs() << " s]" << std::endl;
        ArmRefusals(g);
        std::cout << "  [(R) done at " << secs() << " s]" << std::endl;
        ArmTrivial(g);
        std::cout << "  [(B) done at " << secs() << " s]" << std::endl;
        ArmRoutes(g, lines);
        std::cout << "  [(C) done at " << secs() << " s]" << std::endl;
        ArmEuler(g);
        std::cout << "  [(D) done at " << secs() << " s]" << std::endl;
        ArmGammaOne(g, lines);
        std::cout << "  [(E) done at " << secs() << " s]" << std::endl;
        ArmTable(g, lines);
        std::cout << "  [(N5)/(N6) done at " << secs() << " s]" << std::endl;
        ArmGaloisRing(g);
        std::cout << "  [(W) done at " << secs() << " s]" << std::endl;
        ArmWeightShift(g, lines);
        std::cout << "  [(X4) done at " << secs() << " s]" << std::endl;
        ArmMultiplicityTwo(g, lines);
        std::cout << "  [(M2) done at " << secs() << " s]" << std::endl;
        ArmSparse(g, lines);
        std::cout << "  [(SP) done at " << secs() << " s]" << std::endl;
        ArmPowers(g, lines);
        std::cout << "  [(PV) done at " << secs() << " s]" << std::endl;
        ArmSparseLadder(g, lines);
        std::cout << "  [(SL) done at " << secs() << " s]" << std::endl;
        ArmWk(g, lines);
        std::cout << "  [(GR) done at " << secs() << " s]" << std::endl;
        ArmWkOracles(g, lines);
        std::cout << "  [(GT) done at " << secs() << " s]" << std::endl;
        ArmModelTp2(g, lines);
        std::cout << "  [(T2) done at " << secs() << " s]" << std::endl;
        ArmWideNatural(g, lines);
        std::cout << "  [(A1) done at " << secs() << " s]" << std::endl;
        for (const std::string& l : lines) std::cout << "  " << l << "\n";
    }
    for (const auto& kv : g.by_arm) std::cout << "  arm " << kv.first << ": " << kv.second << " checks\n";
    for (const auto& kv : PhaseTimes()) std::cout << "  phase " << kv.first << ": " << kv.second << " s\n";
    std::cout << "HeckeNebentypusValidation: " << g.checks << " checks, " << g.fails << " failing, " << secs() << " s\n";
    return g.fails == 0 ? 0 : 1;
}
