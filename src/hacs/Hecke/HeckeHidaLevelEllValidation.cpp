// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-6) HeckeHidaLevelEllValidation -- see the header.
//
// Arms ([C] v1.13.0-pre.2, sec:levelshapiro / sec:stabilisation):
//  (R)  the two model routes of U_ell^{(ell)}: eq:Ulevellmodel (the varpi block row)
//       against the general eq:levelcorr, entrywise as cochain maps, q = 0, 1, 2;
//       and prop:levelcorrespondence (4)(a): the general formula at A = B = Gamma
//       against the SHIPPED HeckeOperatorOnK (Theta entrywise, A^{(1)} entrywise).
//  (X)  prop:levelcorrespondence (2): U^{(ell)}, T_2^{(ell)}, T_3^{(ell)} map
//       C^q_eq(L_{Gamma_0(ell)}) into C^q_eq and commute with d (q = 0, 1).
//  (O)  prop:oldspace (1), (2) as COCHAIN maps on C^q_eq(level 1), q = 0, 1, 2
//       (F_P, F_ell, Z/ell^2); (3), (4) on H^1 over F_P: U Phi = Phi M, Phi
//       injective, det(X - M) = X^r chi_T((X^2 + ell^{n+1})/X), and the new space:
//       (U^2 - ell^n) H^1(level ell) in im Phi.
//  (H)  the Hecke ring at level ell: U T_p = T_p U, T_2 T_3 = T_3 T_2, and the
//       intertwinings dg_1 T_p = T_p^{(ell)} dg_1, dg_ell T_p = T_p^{(ell)} dg_ell,
//       omega T_p^{(ell)} = T_p omega, p in {2, 3} (F_P and F_ell).
//  (G)  THE INDEPENDENT ROUTE: H^1(Gamma_0(ell); V) as group 1-cocycles (Shapiro
//       on <S, T>: Z^1(Gamma; Ind) = ker(1+S) x ker(1+U+U^2), f(h) = F(h)_1 on
//       Schreier generators), the operator [H g H] = cor o conj_g o res,
//       (Uf)(h) = sum_t rho(r_t^{-1} g) f(g^{-1} b_t(h) g), r_t h = b_t r_{t.h}; no
//       tau, no model, its own coset table, class map and linear algebra.
//       Compared by dimensions and characteristic polynomials of U^{(ell)},
//       T_2^{(ell)}, T_3^{(ell)} and of linear combinations, over F_P.
//  (S)  obl:shapiro / prop:levelorbit (3): dim H^1 - dim H^0 of the level-ell model
//       against the orbit formula, H^2 = 0; every (Hi10) cell.
//  (T)  lem:levelelltorsion (1): U^{(ell)} = 0 on H^0 over F_ell, U^e = 0 over
//       Z/ell^e.
//  (Hi10) h^1_ord at level ell = at level 1 for even n in [2, 40], ell in
//       {5, 7, 11, 13}; 0 = 0 at odd n; 2 g(X_0(ell)) + 1 = 1, 1, 3, 1 at n = 0.
//  (St) prop:stabilisation: e_ell dg_1 e_1 = 0 mod ell and U e_ell dg_1 =
//       ell e_ell dg_ell (1); c_st = omega e_ell dg_ell e_1 is an automorphism of
//       e_1 H(1) commuting with T_ell, = T_ell mod ell on it, and EQUAL to
//       (ell+1, T) e_ord(M) (0, 1)^T computed from level 1 alone (2); e_ell dg_ell
//       e_1 has rank h mod ell -- split injective with left inverse c_st^{-1} omega
//       -- and (U^2 - U T' + ell^{n+1}) e_ell dg_ell e_1 = 0 (3); over F_ell at every
//       (Hi10) cell and over Z/ell^m (m = 2, 3) at the ring cells.  Never a trace
//       inverse: omega e_ell dg_1 = 0 mod ell is checked as the NEGATIVE statement.
#include "HeckeHidaLevelEllValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeLevelEll.h"
#include "HeckeOperatorFamily.h"
#include "HeckeOperatorOnK.h"
#include "HeckeOrdinaryPart.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "InducedLocalSystem.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "ValidationSupport.h"

namespace {

using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::level;
using u128 = unsigned __int128;

constexpr uint32_t kP = 1000003;   // the characteristic-0 surrogate field F_P

// Wall time per phase (reported once, at the end).
std::map<std::string, double>& PhaseTimes() { static std::map<std::string, double> t; return t; }
struct PhaseTimer {
    std::string name;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit PhaseTimer(std::string n) : name(std::move(n)) {}
    ~PhaseTimer() { PhaseTimes()[name] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

struct Checker {
    uint32_t checks = 0, fails = 0;
    std::map<std::string, uint32_t> by_arm;   // "(X)" prefix -> checks
    void check(bool ok, const std::string& what) {
        ++checks;
        by_arm[what.substr(0, what.find(' '))] += 1;
        if (!ok) {
            ++fails;
            if (fails <= 60) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

std::string Cell(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + ", " + std::to_string(ell) + ")"; }


// ---------------------------------------------------------------------------
// The model and one cell (level 1 + level ell) over a ring ctx.
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

using hecke::vsupport::HeckeDatum;

struct LevelCell {
    uint32_t n = 0, ell = 0;
    engine::ModCtx<uint32_t> ctx;
    uint32_t coef = 0;                                   // the tables' coefficient prime
    HeckeDatum U;                                        // g = diag(1, ell): the level-ell table AND the U_ell tau
    std::map<uint32_t, HeckeDatum> Tp;                   // g = diag(1, p)
    std::unique_ptr<LocalSystemSimplicialComplex> L1, Ll;
    std::unique_ptr<EquivariantCohomologyOnK> H1, Hl;
    std::unique_ptr<Gamma0Level> lvl;
    FullLevel full;
    double build_s = 0;
};

HeckeDatum MakeDatum(uint64_t p, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    return hecke::vsupport::MakeHeckeDatum(*M.K, *M.tess, p, 1, coef, n);
}

std::unique_ptr<LevelCell> BuildCell(uint32_t n, uint32_t ell, const engine::ModCtx<uint32_t>& ctx,
                                     const std::vector<uint32_t>& tp) {
    const auto t0 = std::chrono::steady_clock::now();
    Model& M = TheModel();
    auto c = std::make_unique<LevelCell>();
    c->n = n;
    c->ell = ell;
    c->ctx = ctx;
    c->coef = ctx.is_field() ? ctx.p : ell;
    {
        PhaseTimer pt("build: coset + tau tables");
        c->U = MakeDatum(ell, c->coef, n);
        for (uint32_t p : tp) c->Tp[p] = MakeDatum(p, c->coef, n);
    }
    PhaseTimer pt("build: local systems + cohomology");
    c->lvl.reset(new Gamma0Level(*c->U.cosets));
    const uint32_t D = (ell + 1) * (n + 1);
    if (ctx.is_field()) {
        c->L1.reset(new LocalSystemSimplicialComplex(*M.K, GenerateSL2ZRelationsElt<uint32_t>(*M.K, n, ctx.p), n + 1, ctx.p));
        c->Ll.reset(new LocalSystemSimplicialComplex(hecke::MakeInducedLocalSystemGamma0(*M.K, *c->U.cosets, ctx.p, n)));
        c->H1.reset(new EquivariantCohomologyOnK(c->L1.get()));
        c->Hl.reset(new EquivariantCohomologyOnK(c->Ll.get()));
        c->H1->compute_cohomology();
        c->Hl->compute_cohomology();
    } else {
        ENGINE_CHECK_MSG(!c->U.tau->used_mod_ell_solve(), "WP-6: the U_ell tau used the mod-ell fallback (not integral)");
        const hecke::ScopedCoefficientRing ring(ctx.k);
        c->L1.reset(new LocalSystemSimplicialComplex(*M.K, hecke::GenerateSL2ZRelationsRing<uint32_t>(*M.K, n, ctx), n + 1, ctx.p));
        c->Ll.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsRing(*M.K, *c->U.cosets, n, ctx), D, ctx.p));
        c->H1.reset(new EquivariantCohomologyOnK(c->L1.get()));
        c->Hl.reset(new EquivariantCohomologyOnK(c->Ll.get()));
        c->H1->compute_cohomology();
        c->Hl->compute_cohomology();
    }
    c->build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return c;
}

// The correspondences of prop:levelellU / prop:levelcorrespondence at N = ell.
CorrespondenceSpec SpecU(const LevelCell& c) {
    CorrespondenceSpec s{c.lvl.get(), c.lvl.get(), c.U.cosets.get(), c.U.tau.get(), Gamma0SubTransversal(c.ell, c.ell), "U_ell^(ell)"};
    VerifyTransversal(s, c.ell);
    return s;
}
CorrespondenceSpec SpecDgEll(const LevelCell& c) {
    CorrespondenceSpec s{&c.full, c.lvl.get(), c.U.cosets.get(), c.U.tau.get(), {Mat22::I()}, "dg_ell"};
    VerifyTransversal(s, 1);
    return s;
}
CorrespondenceSpec SpecDg1(const LevelCell& c) {
    CorrespondenceSpec s{&c.full, c.lvl.get(), nullptr, nullptr, {Mat22::I()}, "dg_1"};
    VerifyTransversal(s, 1);
    return s;
}
CorrespondenceSpec SpecOmega(const LevelCell& c) {
    std::vector<Mat22> r;
    for (uint32_t t = 1; t <= c.lvl->m(); ++t) r.push_back(c.lvl->rep(t));
    CorrespondenceSpec s{c.lvl.get(), &c.full, nullptr, nullptr, r, "omega"};
    VerifyTransversal(s, c.ell + 1);
    return s;
}
CorrespondenceSpec SpecT1(const LevelCell& c) {   // T_ell at level 1, instance (4)(a)
    std::vector<Mat22> r;
    for (uint32_t j = 1; j <= c.U.cosets->num_cosets(); ++j) r.push_back(c.U.cosets->s(j));
    CorrespondenceSpec s{&c.full, &c.full, c.U.cosets.get(), c.U.tau.get(), r, "T_ell (level 1)"};
    VerifyTransversal(s, c.ell + 1);
    return s;
}
CorrespondenceSpec SpecTpLevel(const LevelCell& c, uint32_t p) {
    const HeckeDatum& d = c.Tp.at(p);
    CorrespondenceSpec s{c.lvl.get(), c.lvl.get(), d.cosets.get(), d.tau.get(), Gamma0SubTransversal(c.ell, p),
                         "T_" + std::to_string(p) + "^(ell)"};
    VerifyTransversal(s, p + 1);
    return s;
}
CorrespondenceSpec SpecTp1(const LevelCell& c, uint32_t p) {
    const HeckeDatum& d = c.Tp.at(p);
    std::vector<Mat22> r;
    for (uint32_t j = 1; j <= d.cosets->num_cosets(); ++j) r.push_back(d.cosets->s(j));
    CorrespondenceSpec s{&c.full, &c.full, d.cosets.get(), d.tau.get(), r, "T_" + std::to_string(p) + " (level 1)"};
    VerifyTransversal(s, p + 1);
    return s;
}

BlockCochainMap Build(const LevelCell& c, const CorrespondenceSpec& s, uint32_t q) {
    return BuildLevelCorrespondence(c.ctx, s, *TheModel().K, c.n, q);
}

// ---------------------------------------------------------------------------
// HMat helpers local to the validation.
// ---------------------------------------------------------------------------
HMat Pow(const HMat& A, uint64_t e) {
    HMat R = HIdentity(A), B = A;
    while (e) {
        if (e & 1) R = HMul(R, B);
        B = HMul(B, B);
        e >>= 1;
    }
    return R;
}

HMat FromShippedField(const MatrixT<uint32_t>& A, const HMat& like) {
    HMat M = like;
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) M.a[static_cast<size_t>(i) * M.cols + j] = A.data[i][j];
    return HCanonical(M);
}
HMat FromShippedModule(const engine::DenseBuf<uint32_t>& A, const HMat& like) {
    HMat M = like;
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) M.a[static_cast<size_t>(i) * M.cols + j] = A.at(i, j);
    return HCanonical(M);
}

hecke::ordinary::RingMat ToRing(const HMat& A) {
    ENGINE_CHECK(A.rows == A.cols);
    hecke::ordinary::RingMat R;
    R.ell = A.ell;
    R.row_exp = A.row_exp;
    R.a = A.a;
    return hecke::ordinary::Canonical(R);
}
HMat FromRing(const hecke::ordinary::RingMat& R, const HMat& like) {
    HMat M = like;
    M.a = R.a;
    return HCanonical(M);
}

// Gauss-Jordan inverse over a field (throws if singular).
HMat FieldInverse(const HMat& A) {
    ENGINE_CHECK(A.field && A.rows == A.cols);
    const uint64_t p = A.modulus;
    const uint32_t n = A.rows;
    std::vector<std::vector<uint64_t>> M(n, std::vector<uint64_t>(2 * n, 0));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) M[i][j] = A.at(i, j) % p;
        M[i][n + i] = 1;
    }
    for (uint32_t c = 0; c < n; ++c) {
        uint32_t piv = c;
        while (piv < n && M[piv][c] == 0) ++piv;
        ENGINE_CHECK_MSG(piv < n, "FieldInverse: singular");
        std::swap(M[piv], M[c]);
        const uint64_t iv = powmod<uint64_t>(M[c][c], p - 2, p);
        for (uint64_t& x : M[c]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % p);
        for (uint32_t i = 0; i < n; ++i) {
            if (i == c || M[i][c] == 0) continue;
            const uint64_t f = M[i][c];
            for (uint32_t j = 0; j < 2 * n; ++j) M[i][j] = (M[i][j] + p - static_cast<uint64_t>(static_cast<u128>(f) * M[c][j] % p)) % p;
        }
    }
    HMat R = A;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j) R.a[static_cast<size_t>(i) * n + j] = M[i][n + j];
    return R;
}

// Column space / kernel over a field.
std::vector<std::vector<uint64_t>> FieldColumnBasis(const HMat& A) {
    const uint64_t p = A.modulus;
    std::vector<std::vector<uint64_t>> basis, ech;   // ech: reduced copies with pivots
    std::vector<uint32_t> piv;
    for (uint32_t j = 0; j < A.cols; ++j) {
        std::vector<uint64_t> v(A.rows), w(A.rows);
        for (uint32_t i = 0; i < A.rows; ++i) v[i] = w[i] = A.at(i, j) % p;
        for (size_t b = 0; b < ech.size(); ++b) {
            const uint64_t f = w[piv[b]];
            if (f == 0) continue;
            for (uint32_t i = 0; i < A.rows; ++i) w[i] = (w[i] + p - static_cast<uint64_t>(static_cast<u128>(f) * ech[b][i] % p)) % p;
        }
        uint32_t pr = A.rows;
        for (uint32_t i = 0; i < A.rows; ++i) if (w[i]) { pr = i; break; }
        if (pr == A.rows) continue;
        const uint64_t iv = powmod<uint64_t>(w[pr], p - 2, p);
        for (uint64_t& x : w) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % p);
        for (size_t b = 0; b < ech.size(); ++b) {   // keep ech fully reduced at the new pivot
            const uint64_t f = ech[b][pr];
            if (f == 0) continue;
            for (uint32_t i = 0; i < A.rows; ++i) ech[b][i] = (ech[b][i] + p - static_cast<uint64_t>(static_cast<u128>(f) * w[i] % p)) % p;
        }
        ech.push_back(w);
        piv.push_back(pr);
        basis.push_back(v);
    }
    return basis;
}
std::vector<std::vector<uint64_t>> FieldKernel(const HMat& A) {
    const uint64_t p = A.modulus;
    const uint32_t R = A.rows, C = A.cols;
    std::vector<std::vector<uint64_t>> M(R, std::vector<uint64_t>(C));
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) M[i][j] = A.at(i, j) % p;
    std::vector<int> pivcol_of_row;
    std::vector<int> row_of_col(C, -1);
    uint32_t rk = 0;
    for (uint32_t c = 0; c < C && rk < R; ++c) {
        uint32_t piv = rk;
        while (piv < R && M[piv][c] == 0) ++piv;
        if (piv == R) continue;
        std::swap(M[piv], M[rk]);
        const uint64_t iv = powmod<uint64_t>(M[rk][c], p - 2, p);
        for (uint64_t& x : M[rk]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % p);
        for (uint32_t i = 0; i < R; ++i) {
            if (i == rk || M[i][c] == 0) continue;
            const uint64_t f = M[i][c];
            for (uint32_t j = 0; j < C; ++j) M[i][j] = (M[i][j] + p - static_cast<uint64_t>(static_cast<u128>(f) * M[rk][j] % p)) % p;
        }
        row_of_col[c] = static_cast<int>(rk);
        ++rk;
    }
    std::vector<std::vector<uint64_t>> ker;
    for (uint32_t f = 0; f < C; ++f) {
        if (row_of_col[f] >= 0) continue;
        std::vector<uint64_t> v(C, 0);
        v[f] = 1;
        for (uint32_t c = 0; c < C; ++c)
            if (row_of_col[c] >= 0) v[c] = (p - M[row_of_col[c]][f]) % p;
        ker.push_back(v);
    }
    return ker;
}

// The ordinary projector over a FIELD: projection onto im U^t along ker U^t (Fitting).
HMat FieldOrdinaryProjector(const HMat& U) {
    const uint32_t t = U.rows;
    if (t == 0) return U;
    uint64_t e = 1;
    while (e < t) e <<= 1;
    const HMat A = Pow(U, e);
    const auto V1 = FieldColumnBasis(A);
    const auto V0 = FieldKernel(A);
    ENGINE_CHECK_MSG(V1.size() + V0.size() == t, "FieldOrdinaryProjector: Fitting dimensions do not add up");
    HMat P = U, Dg = U;
    std::fill(Dg.a.begin(), Dg.a.end(), 0);
    for (uint32_t j = 0; j < t; ++j) {
        const auto& col = j < V1.size() ? V1[j] : V0[j - V1.size()];
        for (uint32_t i = 0; i < t; ++i) P.a[static_cast<size_t>(i) * t + j] = col[i];
        if (j < V1.size()) Dg.a[static_cast<size_t>(j) * t + j] = 1;
    }
    return HMul(HMul(P, Dg), FieldInverse(P));
}

HMat OrdinaryProjector(const HMat& U) {
    if (U.field) return FieldOrdinaryProjector(U);
    return FromRing(hecke::ordinary::OrdinaryProjector(ToRing(U)), U);
}

// The 2 x 2 block matrix with blocks A (top left), B (top right), Cb (bottom left), D (bottom right), each t x t.
HMat Block2(const HMat& A, const HMat& B, const HMat& Cb, const HMat& D) {
    HMat M;
    const uint32_t t = A.rows;
    M.rows = M.cols = 2 * t;
    M.modulus = A.modulus;
    M.ell = A.ell;
    M.field = A.field;
    M.row_exp = A.row_exp;
    M.row_exp.insert(M.row_exp.end(), Cb.row_exp.begin(), Cb.row_exp.end());
    M.a.assign(static_cast<size_t>(4) * t * t, 0);
    for (uint32_t i = 0; i < t; ++i)
        for (uint32_t j = 0; j < t; ++j) {
            M.a[static_cast<size_t>(i) * 2 * t + j] = A.at(i, j);
            M.a[static_cast<size_t>(i) * 2 * t + t + j] = B.at(i, j);
            M.a[static_cast<size_t>(t + i) * 2 * t + j] = Cb.at(i, j);
            M.a[static_cast<size_t>(t + i) * 2 * t + t + j] = D.at(i, j);
        }
    return HCanonical(M);
}
HMat SubBlock(const HMat& M, uint32_t bi, uint32_t bj, uint32_t t) {
    HMat A;
    A.rows = A.cols = t;
    A.modulus = M.modulus;
    A.ell = M.ell;
    A.field = M.field;
    A.row_exp.assign(M.row_exp.begin() + bi * t, M.row_exp.begin() + (bi + 1) * t);
    A.a.assign(static_cast<size_t>(t) * t, 0);
    for (uint32_t i = 0; i < t; ++i)
        for (uint32_t j = 0; j < t; ++j) A.a[static_cast<size_t>(i) * t + j] = M.at(bi * t + i, bj * t + j);
    return HCanonical(A);
}
HMat Concat(const HMat& A, const HMat& B) {   // [A | B], same rows
    ENGINE_CHECK(A.rows == B.rows);
    HMat M = A;
    M.cols = A.cols + B.cols;
    M.a.assign(static_cast<size_t>(M.rows) * M.cols, 0);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) M.a[static_cast<size_t>(i) * M.cols + j] = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) M.a[static_cast<size_t>(i) * M.cols + A.cols + j] = B.at(i, j);
    }
    return M;
}

// Polynomials over F_p, low -> high.
using Poly = std::vector<uint64_t>;
Poly PolyMul(const Poly& a, const Poly& b, uint64_t p) {
    Poly c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + static_cast<uint64_t>(static_cast<u128>(a[i]) * b[j] % p)) % p;
    return c;
}
Poly PolyAdd(Poly a, const Poly& b, uint64_t p) {
    if (a.size() < b.size()) a.resize(b.size(), 0);
    for (size_t i = 0; i < b.size(); ++i) a[i] = (a[i] + b[i]) % p;
    return a;
}
// X^r chi((X^2 + c)/X) = sum_k chi_k (X^2 + c)^k X^{r-k}
Poly OldSpacePoly(const Poly& chi, uint64_t c, uint64_t p) {
    const size_t r = chi.size() - 1;
    Poly out{0};
    Poly qk{1};                                          // (X^2 + c)^k
    for (size_t k = 0; k <= r; ++k) {
        Poly term(r - k + 1, 0);
        term[r - k] = chi[k];                            // chi_k X^{r-k}
        out = PolyAdd(out, PolyMul(term, qk, p), p);
        qk = PolyMul(qk, Poly{c % p, 0, 1}, p);
    }
    while (out.size() > 1 && out.back() == 0) out.pop_back();
    return out;
}

std::string PolyText(const Poly& f) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < f.size(); ++i) os << (i ? "," : "") << f[i];
    os << "]";
    return os.str();
}

// ---------------------------------------------------------------------------
// (G) the independent route: group 1-cocycles over F_P, no tau, no model.
// ---------------------------------------------------------------------------
class GroupRoute {
public:
    GroupRoute(uint32_t n, uint32_t ell, uint64_t P) : P_(P), ell_(ell), n_(n), d_(n + 1), m_(ell + 1), D_((ell + 1) * (n + 1)) {}

    uint32_t m() const { return m_; }
    // Own coset table of Gamma_0(ell)\Gamma: w_{1+b} = [[1,0],[b,1]], w_{ell+1} = S.
    Mat22 w(uint32_t j) const {
        if (j <= ell_) return Mat22{1, 0, static_cast<int64_t>(j - 1), 1};
        return Mat22{0, -1, 1, 0};
    }
    uint32_t cls(const Mat22& x) const {
        const int64_t L = ell_;
        const uint64_t c = static_cast<uint64_t>(((x.c % L) + L) % L), dd = static_cast<uint64_t>(((x.d % L) + L) % L);
        if (dd != 0) return 1 + static_cast<uint32_t>(c * powmod<uint64_t>(dd, ell_ - 2, ell_) % ell_);
        return ell_ + 1;
    }
    bool inH(const Mat22& x) const { return x.det() == 1 && x.c % static_cast<int64_t>(ell_) == 0; }

    const std::vector<uint64_t>& rho(const Mat22& g) {
        const std::array<uint32_t, 4> key = g.reduced_mod(static_cast<uint32_t>(P_));
        auto it = rho_.find(key);
        if (it != rho_.end()) return it->second;
        const FlatMatrixT<uint32_t> A(2, 2, static_cast<uint32_t>(P_), std::vector<uint32_t>{key[0], key[1], key[2], key[3]});
        const FlatMatrixT<uint32_t> R = ComputeSymmetricPowerMatrixFM(A, n_);
        std::vector<uint64_t> v(static_cast<size_t>(d_) * d_);
        for (uint32_t r = 0; r < d_; ++r)
            for (uint32_t c = 0; c < d_; ++c) v[static_cast<size_t>(r) * d_ + c] = R.get_entry(r, c);
        return rho_.emplace(key, std::move(v)).first->second;
    }

    // d x D row block R times M(gamma): (R M)[:, i.gamma] = R[:, i] rho(h_i(gamma)).
    std::vector<uint64_t> RightMul(const std::vector<uint64_t>& R, const Mat22& gamma) {
        std::vector<uint64_t> out(static_cast<size_t>(d_) * D_, 0);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22 x = w(i) * gamma;
            const uint32_t j = cls(x);
            const Mat22 h = x * w(j).inv_unimodular();
            const std::vector<uint64_t>& B = rho(h);
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
    // M(gamma) as a dense D x D matrix (rows of blocks).
    std::vector<uint64_t> Dense(const Mat22& gamma) {
        std::vector<uint64_t> out(static_cast<size_t>(D_) * D_, 0);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22 x = w(i) * gamma;
            const uint32_t j = cls(x);
            const std::vector<uint64_t>& B = rho(x * w(j).inv_unimodular());
            for (uint32_t r = 0; r < d_; ++r)
                for (uint32_t c = 0; c < d_; ++c) out[static_cast<size_t>((i - 1) * d_ + r) * D_ + (j - 1) * d_ + c] = B[static_cast<size_t>(r) * d_ + c];
        }
        return out;
    }

    // S / T^k word with gamma = +-prod(letters).
    struct Letter { bool S; int64_t k; };
    static Mat22 Tpow(int64_t k) { return Mat22{1, k, 0, 1}; }
    std::vector<Letter> Decompose(const Mat22& g0, bool& ok) const {
        std::vector<Letter> L;
        Mat22 x = g0;
        while (x.c != 0) {
            int64_t k = x.a / x.c;
            if ((x.a % x.c != 0) && ((x.a < 0) != (x.c < 0))) --k;   // floor
            if (k != 0) { L.push_back({false, k}); x = Tpow(-k) * x; }
            L.push_back({true, 0});
            x = Mat22{0, 1, -1, 0} * x;                                  // S^{-1} x
        }
        if (x.a == 1) { if (x.b) L.push_back({false, x.b}); }
        else { if (x.b) L.push_back({false, -x.b}); }
        Mat22 prod = Mat22::I();
        for (const Letter& l : L) prod = prod * (l.S ? Mat22::S() : Tpow(l.k));
        ok = (prod == g0) || (prod == g0.neg());
        return L;
    }
    // R sum_{i<k} M(T^i), k >= 0.
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
    // f(x) = F(x)_1 = Ea a + Eb b for the Gamma-cocycle F with F(S) = a, F(ST) = b.
    struct Eval { std::vector<uint64_t> Ea, Eb; };
    const Eval& E(const Mat22& x) {
        const std::array<int64_t, 4> key{x.a, x.b, x.c, x.d};
        auto it = eval_.find(key);
        if (it != eval_.end()) return it->second;
        bool ok = false;
        const std::vector<Letter> L = Decompose(x, ok);
        ENGINE_CHECK_MSG(ok, "GroupRoute: word decomposition failed");
        std::vector<uint64_t> R(static_cast<size_t>(d_) * D_, 0);
        for (uint32_t r = 0; r < d_; ++r) R[static_cast<size_t>(r) * D_ + r] = 1;   // block row 1 of the identity
        Eval ev{std::vector<uint64_t>(R.size(), 0), std::vector<uint64_t>(R.size(), 0)};
        auto add_FT = [&](const std::vector<uint64_t>& Q, bool neg) {   // += +-Q F(T), F(T) = a + M(S) b
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
    // f_z(x) for z = (a; b) (length 2D).
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

    // Build Z^1(Gamma; M), the Schreier generators and the quotient basis.
    void Setup(Checker& g, const std::string& tag) {
        HMat like;
        like.modulus = P_;
        like.ell = P_;
        like.field = true;
        auto dense_to = [&](const std::vector<uint64_t>& A, uint32_t R, uint32_t C) {
            HMat M = like;
            M.rows = R;
            M.cols = C;
            M.row_exp.assign(R, 1);
            M.a = A;
            return M;
        };
        const Mat22 U = Mat22::ST();
        const std::vector<uint64_t> MS = Dense(Mat22::S()), MU = Dense(U);
        std::vector<uint64_t> AS(MS), AU(static_cast<size_t>(D_) * D_, 0);
        const HMat MUm = dense_to(MU, D_, D_);
        const HMat MU2 = HMul(MUm, MUm);
        for (uint32_t i = 0; i < D_; ++i) {
            AS[static_cast<size_t>(i) * D_ + i] = (AS[static_cast<size_t>(i) * D_ + i] + 1) % P_;
            for (uint32_t j = 0; j < D_; ++j)
                AU[static_cast<size_t>(i) * D_ + j] = (MU[static_cast<size_t>(i) * D_ + j] + MU2.at(i, j) + (i == j ? 1 : 0)) % P_;
        }
        const auto kS = FieldKernel(dense_to(AS, D_, D_));
        const auto kU = FieldKernel(dense_to(AU, D_, D_));
        for (const auto& a : kS) { std::vector<uint64_t> z(2 * D_, 0); std::copy(a.begin(), a.end(), z.begin()); Z1_.push_back(z); }
        for (const auto& b : kU) { std::vector<uint64_t> z(2 * D_, 0); std::copy(b.begin(), b.end(), z.begin() + D_); Z1_.push_back(z); }
        // B^1(Gamma; M) and M^Gamma
        std::vector<uint64_t> B1(static_cast<size_t>(2) * D_ * D_, 0);
        for (uint32_t i = 0; i < D_; ++i)
            for (uint32_t j = 0; j < D_; ++j) {
                B1[static_cast<size_t>(i) * D_ + j] = (MS[static_cast<size_t>(i) * D_ + j] + P_ - (i == j ? 1 : 0)) % P_;
                B1[static_cast<size_t>(D_ + i) * D_ + j] = (MU[static_cast<size_t>(i) * D_ + j] + P_ - (i == j ? 1 : 0)) % P_;
            }
        const uint32_t rankB1 = HRankModEll(dense_to(B1, 2 * D_, D_));
        h0_ = D_ - rankB1;
        h1_ = static_cast<uint32_t>(Z1_.size()) - rankB1;
        // Schreier generators of Gamma_0(ell)
        for (uint32_t j = 1; j <= m_; ++j)
            for (const Mat22& x : {Mat22::S(), Mat22::T()}) {
                const Mat22 y = w(j) * x;
                const Mat22 s = y * w(cls(y)).inv_unimodular();
                ENGINE_CHECK(inH(s));
                if (s == Mat22::I() || s == Mat22::I().neg()) continue;
                gens_.push_back(s);
            }
        // the H-cocycle identity f(xy) = f(x) + rho(x) f(y) on a few pairs (self-check of the evaluator)
        bool coc = true;
        for (size_t i = 0; i + 1 < gens_.size() && i < 4; ++i)
            for (const auto& z : Z1_) {
                const Mat22 x = gens_[i], y = gens_[i + 1];
                const auto fxy = f(x * y, z), fx = f(x, z), fy = f(y, z);
                const auto& rx = rho(x);
                for (uint32_t r = 0; r < d_; ++r) {
                    u128 acc = fx[r];
                    for (uint32_t c = 0; c < d_; ++c) acc += static_cast<u128>(rx[static_cast<size_t>(r) * d_ + c]) * fy[c];
                    if (static_cast<uint64_t>(acc % P_) != fxy[r]) coc = false;
                }
                break;
            }
        g.check(coc, "(G) " + tag + ": the evaluated Shapiro restriction is an H-cocycle (f(xy) = f(x) + x f(y))");
        // quotient basis: B_H columns, then Phi(z) columns
        const uint32_t len = static_cast<uint32_t>(gens_.size()) * d_;
        std::vector<std::vector<uint64_t>> cols;
        for (uint32_t r = 0; r < d_; ++r) {
            std::vector<uint64_t> v(len, 0);
            for (size_t s = 0; s < gens_.size(); ++s) {
                const auto& rs = rho(gens_[s]);
                for (uint32_t a = 0; a < d_; ++a) v[s * d_ + a] = (rs[static_cast<size_t>(a) * d_ + r] + P_ - (a == r ? 1 : 0)) % P_;
            }
            cols.push_back(v);
        }
        HMat BH = like;
        BH.rows = len;
        BH.cols = static_cast<uint32_t>(cols.size());
        BH.row_exp.assign(len, 1);
        BH.a.assign(static_cast<size_t>(len) * BH.cols, 0);
        for (uint32_t j = 0; j < BH.cols; ++j)
            for (uint32_t i = 0; i < len; ++i) BH.a[static_cast<size_t>(i) * BH.cols + j] = cols[j][i];
        bh_ = FieldColumnBasis(BH);
        std::vector<std::vector<uint64_t>> all = bh_;
        uint32_t base_rank = static_cast<uint32_t>(bh_.size());
        for (const auto& z : Z1_) {
            std::vector<uint64_t> v = Stack(z);
            all.push_back(v);
            HMat T = like;
            T.rows = len;
            T.cols = static_cast<uint32_t>(all.size());
            T.row_exp.assign(len, 1);
            T.a.assign(static_cast<size_t>(len) * T.cols, 0);
            for (uint32_t j = 0; j < T.cols; ++j)
                for (uint32_t i = 0; i < len; ++i) T.a[static_cast<size_t>(i) * T.cols + j] = all[j][i];
            if (HRankModEll(T) > base_rank) { reps_.push_back(z); ++base_rank; }
            else all.pop_back();
        }
        g.check(reps_.size() == h1_, "(G) " + tag + ": Shapiro -- dim Z^1/B^1 (Gamma; Ind) = " + std::to_string(h1_) +
                                         " equals the rank of the restriction to Gamma_0(ell) mod B_H = " +
                                         std::to_string(reps_.size()));
        basis_cols_ = all;
        len_ = len;
    }
    uint32_t h1() const { return h1_; }
    uint32_t h0() const { return h0_; }

    std::vector<uint64_t> Stack(const std::vector<uint64_t>& z) {
        std::vector<uint64_t> v(static_cast<size_t>(gens_.size()) * d_, 0);
        for (size_t s = 0; s < gens_.size(); ++s) {
            const auto fs = f(gens_[s], z);
            std::copy(fs.begin(), fs.end(), v.begin() + s * d_);
        }
        return v;
    }

    // [H g H] on the quotient basis; r = transversal of B''\H, B'' = H cap g H g^{-1}.
    HMat Operator(const Mat22& g, const std::vector<Mat22>& r) {
        auto in_Bpp = [&](const Mat22& x) {
            if (!inH(x)) return false;
            const int64_t D = g.det();
            const Mat22 M = g.adj() * x * g;
            if (M.a % D || M.b % D || M.c % D || M.d % D) return false;
            return inH(Mat22{M.a / D, M.b / D, M.c / D, M.d / D});
        };
        const uint32_t nb = static_cast<uint32_t>(bh_.size()), h = static_cast<uint32_t>(reps_.size());
        // the solve matrix [B_H basis | reps images] augmented with the operator images
        std::vector<std::vector<uint64_t>> rhs;
        for (const auto& z : reps_) {
            std::vector<uint64_t> v(len_, 0);
            for (size_t s = 0; s < gens_.size(); ++s) {
                const Mat22& sig = gens_[s];
                for (const Mat22& rt : r) {
                    const Mat22 x = rt * sig;
                    int found = -1;
                    for (size_t u = 0; u < r.size(); ++u)
                        if (in_Bpp(x * r[u].inv_unimodular())) { found = static_cast<int>(u); break; }
                    ENGINE_CHECK_MSG(found >= 0, "GroupRoute: t . h not found");
                    const Mat22 bt = x * r[found].inv_unimodular();
                    const int64_t D = g.det();
                    const Mat22 M = g.adj() * bt * g;
                    const Mat22 y{M.a / D, M.b / D, M.c / D, M.d / D};   // g^{-1} b_t g, in H
                    const auto fy = f(y, z);
                    const Mat22 rg = rt.inv_unimodular() * g;
                    const auto& R = rho(rg);
                    for (uint32_t a = 0; a < d_; ++a) {
                        u128 acc = 0;
                        for (uint32_t c = 0; c < d_; ++c) acc += static_cast<u128>(R[static_cast<size_t>(a) * d_ + c]) * fy[c];
                        v[s * d_ + a] = (v[s * d_ + a] + static_cast<uint64_t>(acc % P_)) % P_;
                    }
                }
            }
            rhs.push_back(v);
        }
        // RREF of [cols | rhs]
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
            ENGINE_CHECK_MSG(piv < len_, "GroupRoute: the basis columns are dependent");
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
            for (uint32_t j = 0; j < R; ++j) ENGINE_CHECK_MSG(M[i][C + j] == 0, "GroupRoute: an image is not an H-cocycle class");
        HMat Op;
        Op.rows = Op.cols = h;
        Op.modulus = P_;
        Op.ell = P_;
        Op.field = true;
        Op.row_exp.assign(h, 1);
        Op.a.assign(static_cast<size_t>(h) * h, 0);
        for (uint32_t i = 0; i < h; ++i)
            for (uint32_t j = 0; j < h; ++j) Op.a[static_cast<size_t>(i) * h + j] = M[nb + i][C + j];
        return Op;
    }

private:
    uint64_t P_;
    uint32_t ell_, n_, d_, m_, D_;
    std::map<std::array<uint32_t, 4>, std::vector<uint64_t>> rho_;
    std::map<std::array<int64_t, 4>, Eval> eval_;
    std::vector<std::vector<uint64_t>> Z1_, reps_, bh_, basis_cols_;
    std::vector<Mat22> gens_;
    uint32_t h0_ = 0, h1_ = 0, len_ = 0;
};

// Own transversals for the group route (independent of Gamma0SubTransversal).
std::vector<Mat22> GroupTransversal(uint32_t ell, uint32_t p) {
    std::vector<Mat22> r;
    for (int64_t t = 0; t < p; ++t) r.push_back(Mat22{1, 0, static_cast<int64_t>(ell) * t, 1});
    if (p != ell) {
        for (int64_t x = 1; x < 1000; ++x)   // [[x, y], [ell, p]] with x p - y ell = 1
            if ((x * p - 1) % static_cast<int64_t>(ell) == 0) { r.push_back(Mat22{x, (x * p - 1) / static_cast<int64_t>(ell), static_cast<int64_t>(ell), static_cast<int64_t>(p)}); break; }
    }
    return r;
}

// ---------------------------------------------------------------------------
// Operators on H^1 of one cell.
// ---------------------------------------------------------------------------
struct H1Ops {
    std::unique_ptr<HFrame> f1, fl;
    HMat T1, U, dg1, dgl, om;
    std::map<uint32_t, HMat> Tp1, Tpl;
};

H1Ops ReadH1(const LevelCell& c, bool with_tp) {
    PhaseTimer pt("operators on H^1");
    H1Ops o;
    o.f1.reset(new HFrame(c.H1.get(), 1));
    o.fl.reset(new HFrame(c.Hl.get(), 1));
    o.T1 = InducedOnH(*o.f1, *o.f1, Build(c, SpecT1(c), 1), "T_ell level 1");
    o.U = InducedOnH(*o.fl, *o.fl, Build(c, SpecU(c), 1), "U_ell^(ell)");
    o.dg1 = InducedOnH(*o.f1, *o.fl, Build(c, SpecDg1(c), 1), "dg_1");
    o.dgl = InducedOnH(*o.f1, *o.fl, Build(c, SpecDgEll(c), 1), "dg_ell");
    o.om = InducedOnH(*o.fl, *o.f1, Build(c, SpecOmega(c), 1), "omega");
    if (with_tp)
        for (const auto& kv : c.Tp) {
            o.Tp1[kv.first] = InducedOnH(*o.f1, *o.f1, Build(c, SpecTp1(c, kv.first), 1), "T_p level 1");
            o.Tpl[kv.first] = InducedOnH(*o.fl, *o.fl, Build(c, SpecTpLevel(c, kv.first), 1), "T_p^(ell)");
        }
    return o;
}

std::string RingText(const engine::ModCtx<uint32_t>& ctx) {
    if (ctx.is_field()) return "F_" + std::to_string(ctx.p);
    return "Z/" + std::to_string(ctx.ell) + "^" + std::to_string(ctx.k);
}

// ---------------------------------------------------------------------------
// (R) the two model routes of U^{(ell)}; (4)(a) against the shipped operator.
// ---------------------------------------------------------------------------
void ArmRoutes(Checker& g, const LevelCell& c, bool shipped_theta) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const CorrespondenceSpec su = SpecU(c);
    for (uint32_t q = 0; q < 3; ++q) {
        const BlockCochainMap A = Build(c, su, q);
        const BlockCochainMap B = BuildUellLevelEllFused(c.ctx, *c.lvl, *c.U.tau, *TheModel().K, c.n, q);
        g.check(A.aggregate() == B.aggregate(),
                "(R) " + tag + " q=" + std::to_string(q) + ": eq:Ulevellmodel (varpi row) == eq:levelcorr entrywise");
    }
    if (!shipped_theta) return;
    const CorrespondenceSpec st = SpecT1(c);
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    if (!c.ctx.is_field()) ring.reset(new hecke::ScopedCoefficientRing(c.ctx.k));
    hecke::HeckeOperatorOnK op(c.H1.get(), c.U.cosets.get(), c.U.tau.get());
    op.compute();
    for (uint32_t q = 0; q < 3; ++q) {
        const BlockCochainMap A = Build(c, st, q);
        const uint32_t dim = A.src_dim();
        std::vector<uint32_t> I(static_cast<size_t>(dim) * dim, 0);
        for (uint32_t i = 0; i < dim; ++i) I[static_cast<size_t>(i) * dim + i] = 1;
        const std::vector<uint32_t> T = A.apply(I, dim);
        bool eq = true;
        if (c.ctx.is_field()) {
            const FlatMatrixT<uint32_t>& Th = op.get_Theta_flat_matrices()[q];
            for (uint32_t r = 0; r < dim && eq; ++r)
                for (uint32_t s = 0; s < dim; ++s)
                    if (Th.get_entry(r, s) != T[static_cast<size_t>(r) * dim + s]) { eq = false; break; }
        } else {
            const engine::DenseBuf<uint32_t>& Th = op.Theta_module(q);
            for (uint32_t r = 0; r < dim && eq; ++r)
                for (uint32_t s = 0; s < dim; ++s)
                    if (Th.at(r, s) != T[static_cast<size_t>(r) * dim + s]) { eq = false; break; }
        }
        g.check(eq, "(R) " + tag + " q=" + std::to_string(q) +
                        ": prop:levelcorrespondence (4)(a) -- eq:levelcorr at A = B = Gamma == the shipped Theta_g (eq:heckechain)");
    }
    const HFrame f1(c.H1.get(), 1);
    const HMat T1 = InducedOnH(f1, f1, Build(c, st, 1), "T_ell level 1");
    const HMat shipped = c.ctx.is_field() ? FromShippedField(op.get_H_matrices()[1], T1) : FromShippedModule(op.H_matrix_module(1), T1);
    g.check(HEqual(T1, shipped), "(R) " + tag + ": A^{(1)}(T_ell) by eq:levelcorr == the shipped HeckeOperatorOnK");
}

// ---------------------------------------------------------------------------
// (X) cochain maps into C_eq commuting with d;  (O) cochain-level old space.
// ---------------------------------------------------------------------------
std::vector<uint32_t> Coboundary(const LevelCell& c, uint32_t q, uint32_t D, const std::vector<uint32_t>& X, uint32_t ncols) {
    const SimplicialComplex& K = *TheModel().K;
    const FlatMatrixT<uint32_t> D1 = K.coboundary_operator_elt<uint32_t>(q, c.ctx.p);
    const uint32_t N0 = K.get_num_q_simplices(q), N1 = K.get_num_q_simplices(q + 1);
    std::vector<uint32_t> Y(static_cast<size_t>(N1) * D * ncols, 0);
    for (uint32_t s1 = 0; s1 < N1; ++s1)
        for (uint32_t s0 = 0; s0 < N0; ++s0) {
            const uint32_t inc = D1.get_entry(s1, s0);
            if (!inc) continue;
            for (uint32_t a = 0; a < D; ++a)
                for (uint32_t col = 0; col < ncols; ++col) {
                    uint32_t& y = Y[(static_cast<size_t>(s1) * D + a) * ncols + col];
                    y = c.ctx.add(y, c.ctx.mul(inc, X[(static_cast<size_t>(s0) * D + a) * ncols + col]));
                }
        }
    return Y;
}

void ArmCochainMaps(Checker& g, const LevelCell& c) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const uint32_t Dl = (c.ell + 1) * (c.n + 1);
    std::vector<CorrespondenceSpec> specs{SpecU(c)};
    for (const auto& kv : c.Tp) specs.push_back(SpecTpLevel(c, kv.first));
    for (const CorrespondenceSpec& s : specs)
        for (uint32_t q = 0; q < 2; ++q) {
            const HFrame fq(c.Hl.get(), q);
            uint32_t cq = 0;
            const std::vector<uint32_t> C = fq.ceq_basis(cq);
            const BlockCochainMap Fq = Build(c, s, q), Fq1 = Build(c, s, q + 1);
            const std::vector<uint32_t> Y = Fq.apply(C, cq);
            bool in_ceq = true;
            for (const auto& rel : c.Ll->get_user_defined_simplex_relations()) {
                if (rel.q_dim != q || !in_ceq) continue;
                for (uint32_t a = 0; a < Dl && in_ceq; ++a) {
                    std::vector<std::pair<uint32_t, uint32_t>> row;
                    for (uint32_t b = 0; b < Dl; ++b) if (rel.M.get_entry(a, b)) row.emplace_back(b, rel.M.get_entry(a, b));
                    for (uint32_t col = 0; col < cq; ++col) {
                        uint32_t acc = Y[(static_cast<size_t>(rel.s_lhs_idx_in_Kq) * Dl + a) * cq + col];
                        for (const auto& e : row) acc = c.ctx.sub(acc, c.ctx.mul(e.second, Y[(static_cast<size_t>(rel.s_rhs_idx_in_Kq) * Dl + e.first) * cq + col]));
                        if (acc) { in_ceq = false; break; }
                    }
                }
            }
            g.check(in_ceq, "(X) " + tag + " " + s.label + " q=" + std::to_string(q) + ": maps C^q_eq(L_{Gamma_0(ell)}) into C^q_eq");
            const std::vector<uint32_t> lhs = Fq1.apply(Coboundary(c, q, Dl, C, cq), cq);
            const std::vector<uint32_t> rhs = Coboundary(c, q, Dl, Y, cq);
            g.check(lhs == rhs, "(X) " + tag + " " + s.label + " q=" + std::to_string(q) + ": F d = d F on C^q_eq (a cochain map)");
        }
}

void ArmOldSpaceCochain(Checker& g, const LevelCell& c) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const CorrespondenceSpec su = SpecU(c), sg = SpecDgEll(c), s1 = SpecDg1(c), so = SpecOmega(c), st = SpecT1(c);
    for (uint32_t q = 0; q < 3; ++q) {
        const HFrame f(c.H1.get(), q);
        uint32_t cq = 0;
        const std::vector<uint32_t> C = f.ceq_basis(cq);
        if (cq == 0) continue;
        const BlockCochainMap U = Build(c, su, q), G = Build(c, sg, q), G1 = Build(c, s1, q), O = Build(c, so, q), T = Build(c, st, q);
        const std::vector<uint32_t> dg1C = G1.apply(C, cq), dglC = G.apply(C, cq);
        std::vector<uint32_t> ldgl = dglC;
        for (uint32_t& x : ldgl) x = c.ctx.mul(x, c.ell % c.ctx.p);
        g.check(U.apply(dg1C, cq) == ldgl, "(O) " + tag + " q=" + std::to_string(q) + ": prop:oldspace (1) U~ dg~_1 = ell dg~_ell as cochain maps (C^q_eq)");
        std::vector<uint32_t> lp1 = C;
        for (uint32_t& x : lp1) x = c.ctx.mul(x, (c.ell + 1) % c.ctx.p);
        g.check(O.apply(dg1C, cq) == lp1, "(O) " + tag + " q=" + std::to_string(q) + ": prop:oldspace (2) omega dg~_1 = (ell+1) id as cochain maps");
        g.check(O.apply(dglC, cq) == T.apply(C, cq), "(O) " + tag + " q=" + std::to_string(q) + ": prop:oldspace (2) omega dg~_ell = T~_ell as cochain maps");
    }
}

// ---------------------------------------------------------------------------
// (O) cohomology, (H) the Hecke ring at level ell -- over a field.
// ---------------------------------------------------------------------------
struct OldSpaceResult { bool ok = false; uint32_t r1 = 0, rl = 0, rank_phi = 0; };

void ArmOldSpaceH(Checker& g, const LevelCell& c, const H1Ops& o) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const uint64_t p = c.ctx.p;
    const uint64_t ln = powmod<uint64_t>(c.ell, c.n, p), ln1 = powmod<uint64_t>(c.ell, c.n + 1, p);
    // (3) U dg_ell = dg_ell T - ell^n dg_1
    g.check(HEqual(HMul(o.U, o.dgl), HSub(HMul(o.dgl, o.T1), HScale(o.dg1, ln))),
            "(O) " + tag + ": prop:oldspace (3) U dg_ell = dg_ell T_ell - ell^n dg_1 on H^1");
    g.check(HEqual(HMul(o.U, o.dg1), HScale(o.dgl, c.ell)), "(O) " + tag + ": prop:oldspace (1) U dg_1 = ell dg_ell on H^1");
    g.check(HEqual(HMul(o.om, o.dg1), HScale(HIdentity(o.T1), c.ell + 1)), "(O) " + tag + ": omega dg_1 = (ell+1) on H^1");
    g.check(HEqual(HMul(o.om, o.dgl), o.T1), "(O) " + tag + ": omega dg_ell = T_ell on H^1");
    if (!c.ctx.is_field()) return;
    // (4) U Phi = Phi M (every field); Phi injective, the charpoly of the old space
    // and the new space are characteristic-0 statements: checked over F_P only
    // (over F_ell, Phi need not be injective -- its rank is reported).
    const uint32_t r = o.T1.rows;
    const HMat Phi = Concat(o.dg1, o.dgl);
    HMat Z = o.T1;
    std::fill(Z.a.begin(), Z.a.end(), 0);
    const HMat M = Block2(Z, HScale(HIdentity(o.T1), (p - ln) % p), HScale(HIdentity(o.T1), c.ell), o.T1);
    g.check(HEqual(HMul(o.U, Phi), HMul(Phi, M)), "(O) " + tag + ": prop:oldspace (4) U Phi = Phi M");
    const uint32_t rk = HRankModEll(Phi);
    if (c.ctx.p == c.ell) {
        std::cout << "  LEVEL_ELL_OLDSPACE cell=" << Cell(c.n, c.ell) << " ring=" << RingText(c.ctx) << " dim_H1_level1=" << r
                  << " dim_H1_levelell=" << o.U.rows << " rank_Phi_mod_ell=" << rk << " (2r = " << 2 * r << ")\n";
        return;
    }
    g.check(rk == 2 * r, "(O) " + tag + ": Phi = (dg_1, dg_ell) injective on H^1(level 1)^2 (rank " + std::to_string(rk) +
                             " == 2 r = " + std::to_string(2 * r) + ")");
    const Poly chiM = HCharpoly(M), chiT = HCharpoly(o.T1);
    g.check(chiM == OldSpacePoly(chiT, ln1, p), "(O) " + tag + ": det(X - M) = X^r chi_T((X^2 + ell^{n+1})/X) = prod (X^2 - a_i X + ell^{k-1})");
    // new space: (U^2 - ell^n) H^1(level ell) in im Phi
    const HMat N = HSub(HMul(o.U, o.U), HScale(HIdentity(o.U), ln));
    g.check(HRankModEll(Concat(Phi, N)) == rk, "(O) " + tag + ": the new space -- (U^2 - ell^n) H^1(Gamma_0(ell)) lies in the old space im Phi");
    std::cout << "  LEVEL_ELL_OLDSPACE cell=" << Cell(c.n, c.ell) << " ring=" << RingText(c.ctx) << " dim_H1_level1=" << r
              << " dim_H1_levelell=" << o.U.rows << " old=" << rk << " new=" << (o.U.rows - rk) << "\n";
}

void ArmHeckeRing(Checker& g, const LevelCell& c, const H1Ops& o) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    for (const auto& kv : o.Tpl) {
        const uint32_t p = kv.first;
        const HMat& Tl = kv.second;
        const HMat& T1 = o.Tp1.at(p);
        const std::string P = "T_" + std::to_string(p);
        g.check(HEqual(HMul(o.U, Tl), HMul(Tl, o.U)), "(H) " + tag + ": U^(ell) " + P + "^(ell) = " + P + "^(ell) U^(ell)");
        g.check(HEqual(HMul(Tl, o.dg1), HMul(o.dg1, T1)), "(H) " + tag + ": " + P + "^(ell) dg_1 = dg_1 " + P);
        g.check(HEqual(HMul(Tl, o.dgl), HMul(o.dgl, T1)), "(H) " + tag + ": " + P + "^(ell) dg_ell = dg_ell " + P);
        g.check(HEqual(HMul(o.om, Tl), HMul(T1, o.om)), "(H) " + tag + ": omega " + P + "^(ell) = " + P + " omega");
    }
    if (o.Tpl.count(2) && o.Tpl.count(3))
        g.check(HEqual(HMul(o.Tpl.at(2), o.Tpl.at(3)), HMul(o.Tpl.at(3), o.Tpl.at(2))), "(H) " + tag + ": T_2^(ell) T_3^(ell) = T_3^(ell) T_2^(ell)");
}

// ---------------------------------------------------------------------------
// (St) prop:stabilisation.
// ---------------------------------------------------------------------------
void ArmStabilisation(Checker& g, const LevelCell& c, const H1Ops& o, uint32_t& h_out, uint32_t& hl_out) {
    PhaseTimer pt("stabilisation");
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const HMat e1 = OrdinaryProjector(o.T1), el = OrdinaryProjector(o.U);
    const uint32_t h = HRankModEll(e1), hl = HRankModEll(el);
    h_out = h;
    hl_out = hl;
    g.check(HEqual(HMul(e1, e1), e1) && HEqual(HMul(el, el), el), "(St) " + tag + ": e_1, e_ell idempotent");
    if (c.n == 0) return;   // (1)-(3) are stated for n >= 1
    const uint64_t L = c.ell;
    // (1)
    g.check(HEqual(HMul(o.U, HMul(el, o.dg1)), HScale(HMul(el, o.dgl), L)), "(St) " + tag + ": (1) U e_ell dg_1 = ell e_ell dg_ell");
    const HMat e_dg1 = HMul(HMul(el, o.dg1), e1);
    g.check(HRankModEll(e_dg1) == 0, "(St) " + tag + ": (1) e_ell dg_1 e_1 = 0 mod ell (the restriction is NOT the integral stabilisation)");
    g.check(HRankModEll(HMul(o.om, e_dg1)) == 0,
            "(St) " + tag + ": negative control -- omega e_ell dg_1 e_1 = 0 mod ell: the trace is no left inverse of e_ell dg_1");
    // (2) c_st = omega e_ell dg_ell on e_1 H(1)
    const HMat G = HMul(HMul(el, o.dgl), e1);
    const HMat C = HMul(o.om, G);
    g.check(HEqual(HMul(e1, C), C) && HEqual(HMul(C, e1), C), "(St) " + tag + ": (2) c_st = e_1 c_st e_1 (an endomorphism of e_1 H(1))");
    g.check(HEqual(HMul(C, o.T1), HMul(o.T1, C)), "(St) " + tag + ": (2) c_st commutes with T_ell");
    g.check(HRankModEll(C) == h, "(St) " + tag + ": (2) c_st is an automorphism of e_1 H(1) (rank mod ell = h = " + std::to_string(h) + ")");
    g.check(HRankModEll(HMul(HSub(C, o.T1), e1)) == 0, "(St) " + tag + ": (2) c_st = T_ell mod ell on e_1 H(1)");
    // (2)'s formula from level 1 alone: c_st = (ell+1, T) e_ord(M) (0, 1)^T on e_1 H(1)
    {
        HMat Z = o.T1;
        std::fill(Z.a.begin(), Z.a.end(), 0);
        const uint64_t ln = powmod<uint64_t>(L, c.n, o.T1.modulus);
        const HMat M = Block2(Z, HScale(HIdentity(o.T1), (o.T1.modulus - ln) % o.T1.modulus), HScale(HIdentity(o.T1), L), o.T1);
        const HMat E = OrdinaryProjector(M);
        const HMat cf = HAdd(HScale(SubBlock(E, 0, 1, o.T1.rows), L + 1), HMul(o.T1, SubBlock(E, 1, 1, o.T1.rows)));
        g.check(HEqual(HMul(cf, e1), C), "(St) " + tag + ": (2) omega e_ell dg_ell e_1 (the model) == (ell+1, T_ell) e_ord(M) (0,1)^T e_1 (level 1 alone)");
    }
    // (3) split injection, U-stability
    g.check(HRankModEll(G) == h, "(St) " + tag + ": (3) e_ell dg_ell e_1 has rank h mod ell -- split injective, left inverse c_st^{-1} omega");
    const uint64_t ln1 = powmod<uint64_t>(L, c.n + 1, o.T1.modulus);
    const HMat rel = HAdd(HSub(HMul(HMul(o.U, o.U), G), HMul(HMul(o.U, G), o.T1)), HScale(G, ln1));
    g.check(HIsZero(rel), "(St) " + tag + ": (3) (U^2 - U T' + ell^{n+1}) e_ell dg_ell e_1 = 0 (u_st the unit root)");
    g.check(HEqual(HMul(el, G), G), "(St) " + tag + ": (3) the image lies in e_ell H(ell)");
}

// ---------------------------------------------------------------------------
// (S) obl:shapiro / prop:levelorbit (3);  (T) lem:levelelltorsion (1).
// ---------------------------------------------------------------------------
uint32_t CountRoots(uint32_t ell, int kind) {   // kind 2: l^2 + 1; kind 3: l^2 - l + 1
    uint32_t c = 0;
    for (uint64_t x = 0; x < ell; ++x) {
        const uint64_t v = kind == 2 ? (x * x + 1) % ell : (x * x + ell - x + 1) % ell;
        if (v == 0) ++c;
    }
    return c;
}

void ArmShapiroDims(Checker& g, const LevelCell& c, uint32_t& h0, uint32_t& h1, uint32_t& h2) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const auto& dims = c.Hl->get_H_eq_dimensions();
    h0 = dims[0];
    h1 = dims[1];
    h2 = dims[2];
    g.check(h2 == 0, "(S) " + tag + ": H^2(Gamma_0(ell); V) = 0 (6 a unit)");
    if (c.n % 2 == 1) {
        g.check(h0 == 0 && h1 == 0, "(S) " + tag + ": odd n -- H^* = 0 (prop:levelorbit (2))");
        return;
    }
    const uint32_t d = c.n + 1, m = c.ell + 1;
    RhoCache rc(c.ctx, c.n);
    std::vector<Block> st;
    auto fixdim = [&](const Mat22& x) {
        const Block& b = st[rc.block_of(x, st)];
        HMat A;
        A.rows = A.cols = d;
        A.modulus = c.ctx.p;
        A.ell = c.ctx.p;
        A.field = true;
        A.row_exp.assign(d, 1);
        A.a.assign(b.dense.begin(), b.dense.end());
        for (uint32_t i = 0; i < d; ++i) A.a[static_cast<size_t>(i) * d + i] = (A.a[static_cast<size_t>(i) * d + i] + c.ctx.p - 1) % c.ctx.p;
        return d - HRankModEll(A);
    };
    const uint32_t dS = fixdim(Mat22::S()), dST = fixdim(Mat22::ST());
    const uint32_t nu2 = CountRoots(c.ell, 2), nu3 = CountRoots(c.ell, 3);
    const int64_t D = static_cast<int64_t>(m) * d;
    const int64_t formula = D - (static_cast<int64_t>(m - nu2) / 2 * d + nu2 * dS) - (static_cast<int64_t>(m - nu3) / 3 * d + nu3 * dST);
    g.check(static_cast<int64_t>(h1) - static_cast<int64_t>(h0) == formula,
            "(S) " + tag + ": obl:shapiro / prop:levelorbit (3) -- dim H^1 - dim H^0 = " + std::to_string(formula) +
                " (the orbit formula; model " + std::to_string(h1) + " - " + std::to_string(h0) + ")");
}

void ArmH0(Checker& g, const LevelCell& c) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    const HFrame f0(c.Hl.get(), 0);
    if (f0.t() == 0) return;
    const HMat U0 = InducedOnH(f0, f0, Build(c, SpecU(c), 0), "U on H^0");
    g.check(HRankModEll(U0) == 0, "(T) " + tag + ": lem:levelelltorsion (1) -- U^(ell) = 0 mod ell on H^0 (dim " + std::to_string(f0.t()) + ")");
    const uint32_t e = c.ctx.is_field() ? 1 : c.ctx.k;
    g.check(HIsZero(Pow(U0, e)), "(T) " + tag + ": lem:levelelltorsion (1) -- U^e = 0 on H^0 over " + RingText(c.ctx));
}

// ---------------------------------------------------------------------------
// (G) the independent route against the model.
// ---------------------------------------------------------------------------
void ArmGroupRoute(Checker& g, const LevelCell& c, const H1Ops& o) {
    const std::string tag = Cell(c.n, c.ell) + " over " + RingText(c.ctx);
    ENGINE_CHECK(c.ctx.is_field());
    GroupRoute gr(c.n, c.ell, c.ctx.p);
    gr.Setup(g, tag);
    g.check(gr.h1() == o.U.rows, "(G) " + tag + ": dim H^1(Gamma_0(ell)) -- group cocycles " + std::to_string(gr.h1()) +
                                     " == the level-ell model " + std::to_string(o.U.rows) + " (obl:shapiro)");
    if (gr.h1() != o.U.rows) return;
    const Mat22 gU{1, 0, 0, static_cast<int64_t>(c.ell)};
    const HMat Ug = gr.Operator(gU, GroupTransversal(c.ell, c.ell));
    g.check(HCharpoly(Ug) == HCharpoly(o.U),
            "(G) " + tag + ": charpoly of U_ell^(ell) -- group cocycles == eq:Ulevell on the model " + PolyText(HCharpoly(o.U)));
    // Negative control: keeping the (1:0) block of the varpi row gives dg_ell o omega
    // (sum over ALL j of rho(g gamma s_j^{-1})); the comparison must separate it.
    if (o.U.rows > 0)
        g.check(HCharpoly(Ug) != HCharpoly(HMul(o.dgl, o.om)),
                "(G) " + tag + ": negative control -- the group route separates U^(ell) from dg_ell omega (the varpi row with its last block kept)");
    std::map<uint32_t, HMat> Tg;
    for (const auto& kv : o.Tpl) {
        const uint32_t p = kv.first;
        Tg[p] = gr.Operator(Mat22{1, 0, 0, static_cast<int64_t>(p)}, GroupTransversal(c.ell, p));
        g.check(HCharpoly(Tg[p]) == HCharpoly(kv.second), "(G) " + tag + ": charpoly of T_" + std::to_string(p) + "^(ell) -- group == model");
        // joint: U + 3 T_p
        g.check(HCharpoly(HAdd(Ug, HScale(Tg[p], 3))) == HCharpoly(HAdd(o.U, HScale(kv.second, 3))),
                "(G) " + tag + ": charpoly of U + 3 T_" + std::to_string(p) + " -- group == model (joint spectrum)");
        g.check(HEqual(HMul(Ug, Tg[p]), HMul(Tg[p], Ug)), "(G) " + tag + ": U T_" + std::to_string(p) + " = T_" + std::to_string(p) + " U on the group route");
    }
    if (Tg.count(2) && Tg.count(3))
        g.check(HCharpoly(HAdd(HAdd(Ug, HScale(Tg.at(2), 5)), HScale(Tg.at(3), 7))) ==
                    HCharpoly(HAdd(HAdd(o.U, HScale(o.Tpl.at(2), 5)), HScale(o.Tpl.at(3), 7))),
                "(G) " + tag + ": charpoly of U + 5 T_2 + 7 T_3 -- group == model");
    std::cout << "  LEVEL_ELL_GROUP cell=" << Cell(c.n, c.ell) << " ring=" << RingText(c.ctx) << " h1=" << gr.h1()
              << " h0=" << gr.h0() << " charpoly_U=" << PolyText(HCharpoly(Ug)) << "\n";
}

} // namespace

uint32_t HeckeHidaLevelEllValidationBody();

uint32_t HeckeHidaLevelEllValidation() {
    try {
        return HeckeHidaLevelEllValidationBody();
    } catch (const std::exception& e) {
        std::cout << "  FAIL: HeckeHidaLevelEllValidation threw: " << e.what() << std::endl;
        return 1;
    }
}

uint32_t HeckeHidaLevelEllValidationBody() {
    std::cout << "=== HeckeHidaLevelEllValidation (slot-13 WP-6: Gamma_0(ell) on the model by Shapiro; [C] v1.13.0-pre.2 "
                 "sec:levelshapiro, sec:stabilisation, (Hi10)) ===\n";
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    // ---- F_P cells: routes, cochain maps, old space, Hecke ring, the independent route.
    struct PC { uint32_t n, ell; };
    for (const PC& pc : std::vector<PC>{{0, 5}, {2, 5}, {4, 5}, {10, 5}, {2, 7}, {6, 7}, {10, 7}, {0, 11}, {2, 11}, {4, 13}}) {
        const auto c = BuildCell(pc.n, pc.ell, engine::ModCtx<uint32_t>(kP), {2, 3});
        ArmRoutes(g, *c, pc.n <= 10);
        ArmCochainMaps(g, *c);
        ArmOldSpaceCochain(g, *c);
        const H1Ops o = ReadH1(*c, true);
        ArmOldSpaceH(g, *c, o);
        ArmHeckeRing(g, *c, o);
        ArmGroupRoute(g, *c, o);
    }
    std::cout << "  [F_P arms done at " << secs() << " s]" << std::endl;

    // ---- Z/ell^m cells: routes over the ring, cochain-level old space, stabilisation mod ell^m, H^0.
    struct RC { uint32_t n, ell, m; };
    for (const RC& rc : std::vector<RC>{{4, 5, 2}, {10, 5, 2}, {10, 5, 3}, {12, 5, 2}, {6, 7, 2}, {16, 7, 2}, {10, 11, 2}, {12, 13, 2}}) {
        const engine::ModCtx<uint32_t> ctx = hecke::CoefficientRingContext<uint32_t>("HeckeHidaLevelEllValidation", rc.ell, rc.m);
        const auto c = BuildCell(rc.n, rc.ell, ctx, {});
        if (rc.n <= 10) ArmRoutes(g, *c, true);
        if (rc.n <= 10) ArmOldSpaceCochain(g, *c);
        const H1Ops o = ReadH1(*c, false);
        ArmOldSpaceH(g, *c, o);
        uint32_t h = 0, hl = 0;
        ArmStabilisation(g, *c, o, h, hl);
        g.check(h == hl, "(St) " + Cell(rc.n, rc.ell) + " over " + RingText(ctx) + ": rank_Z/ell^m e_ell H = rank e_1 H (" +
                             std::to_string(hl) + " == " + std::to_string(h) + ")");
        ArmH0(g, *c);
        std::cout << "  LEVEL_ELL_STAB cell=" << Cell(rc.n, rc.ell) << " ring=" << RingText(ctx) << " h_ord_level1=" << h
                  << " h_ord_levelell=" << hl << " exps_level1=" << o.T1.rows << " exps_levelell=" << o.U.rows << "\n";
    }
    std::cout << "  [Z/ell^m arms done at " << secs() << " s]" << std::endl;

    // ---- (Hi10) over F_ell: the grid.
    const uint32_t g_of[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};   // genus of X_0(ell), ell in {5, 7, 11, 13}
    uint32_t cells = 0, held = 0;
    for (uint32_t ell : {5u, 7u, 11u, 13u}) {
        std::vector<uint32_t> ns{0, 1, 3};
        for (uint32_t n = 2; n <= 40; n += 2) ns.push_back(n);
        std::sort(ns.begin(), ns.end());
        for (uint32_t n : ns) {
            const bool tp = n <= 12;
            const auto c = BuildCell(n, ell, engine::ModCtx<uint32_t>(ell), tp ? std::vector<uint32_t>{2, 3} : std::vector<uint32_t>{});
            const std::string tag = Cell(n, ell) + " over F_" + std::to_string(ell);
            uint32_t h0 = 0, h1 = 0, h2 = 0;
            ArmShapiroDims(g, *c, h0, h1, h2);
            if (n <= 12 || n % 10 == 0) {
                PhaseTimer pt("routes (aggregates)");
                const CorrespondenceSpec su = SpecU(*c);
                for (uint32_t q = 0; q < 3; ++q)
                    g.check(Build(*c, su, q).aggregate() ==
                                BuildUellLevelEllFused(c->ctx, *c->lvl, *c->U.tau, *TheModel().K, n, q).aggregate(),
                            "(R) " + tag + " q=" + std::to_string(q) + ": eq:Ulevellmodel == eq:levelcorr entrywise");
            }
            if (h0 > 0) ArmH0(g, *c);
            const H1Ops o = ReadH1(*c, tp);
            if (n <= 12) ArmOldSpaceH(g, *c, o);
            if (tp) ArmHeckeRing(g, *c, o);
            uint32_t h = 0, hl = 0;
            ArmStabilisation(g, *c, o, h, hl);
            uint32_t want_l = h;
            std::string verdict;
            if (n == 0) want_l = 2 * g_of[ell] + 1;
            const bool ok = (n % 2 == 1) ? (h == 0 && hl == 0) : (hl == want_l && (n > 0 || h == 0));
            if (n == 0) g.check(HRankModEll(o.U) == o.U.rows, "(Hi10) " + tag + ": n = 0 -- e_ell = id (U^(ell) invertible on H^1)");
            g.check(ok, "(Hi10) " + tag + ": h^1_ord level ell = " + std::to_string(hl) + ", expected " + std::to_string(want_l) +
                            " (level 1: " + std::to_string(h) + ")");
            ++cells;
            held += ok ? 1 : 0;
            std::cout << "  HIDA_LEVEL_ELL n=" << n << " ell=" << ell << " dim_H1_level1=" << o.T1.rows << " h1ord_level1=" << h
                      << " dim_H0_levelell=" << h0 << " dim_H1_levelell=" << h1 << " h1ord_levelell=" << hl
                      << " verdict=" << (ok ? "HELD" : "FAILED") << "\n";
        }
        std::cout << "  [ell = " << ell << " done at " << secs() << " s]" << std::endl;
    }
    for (const auto& kv : g.by_arm) std::cout << "  arm " << kv.first << ": " << kv.second << " checks\n";
    for (const auto& kv : PhaseTimes()) std::cout << "  phase " << kv.first << ": " << kv.second << " s\n";
    std::cout << "  (Hi10): " << held << " / " << cells << " cells held\n";
    std::cout << "HeckeHidaLevelEllValidation: " << g.checks << " checks, " << g.fails << " failing, " << secs() << " s\n";
    return g.fails == 0 ? 0 : 1;
}
