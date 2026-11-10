// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 17, WP-7: the open halves of [C, sec:isomodelell] as arms
// of HeckeIsotropyModelEllValidation.  See the header for the arm list; report
// docs/implementation/slot17/WP-7_report.md.
#include "HeckeIsotropyModelEllOpenHalves.h"

#include "Model/IntegralHecke.h"
#include "Model/IsotropyModel.h"
#include "Model/IsotropyModelEll.h"
#include "Model/StrictLift.h"

#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/HeckeLevelN.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using hecke::Mat22;
using integral_model::CoefficientModule;
using integral_model::ColumnSpec;
using integral_model::ComparisonMap;
using integral_model::GKey;
using integral_model::IsotropyModel;
using integral_model::LiftAveraging;
using integral_model::LocalModel;
using integral_model::ModelData;
using integral_model::Rat;
using integral_model::StrictLiftBundle;
using U32 = uint32_t;
using Mods = CoefficientModule<U32>;

struct Tally {
    IsoEllArmTally* t;
    void check(bool ok, const std::string& what) {
        ++t->checks;
        if (!ok) {
            ++t->failures;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

// ---- dense helpers (as in HeckeIsotropyModelEllValidation.cpp) -------------
DenseBuf<U32> KernelCols(const ModCtx<U32>& ctx, const DenseBuf<U32>& A) {
    if (A.cols == 0) return DenseBuf<U32>(0, 0);
    if (A.rows == 0) {
        DenseBuf<U32> I(A.cols, A.cols);
        for (U32 i = 0; i < A.cols; ++i) I.at(i, i) = 1;
        return I;
    }
    DenseBuf<U32> Kt;
    if (ctx.is_field()) {
        DenseBuf<U32> R = A;
        const auto rr = engine::elim::rref(ctx, engine::view(R));
        Kt = engine::elim::kernel_basis(ctx, engine::cview(R), rr.pivots);
    } else {
        Kt = engine::elim::kernel_module(ctx, engine::cview(A)).H;
    }
    DenseBuf<U32> K(Kt.cols, Kt.rows);
    for (U32 i = 0; i < Kt.rows; ++i)
        for (U32 j = 0; j < Kt.cols; ++j) K.at(j, i) = Kt.at(i, j);
    return K;
}
U32 Len(const ModCtx<U32>& ctx, const DenseBuf<U32>& A) { return integral_model::ColumnLength(ctx, A); }
DenseBuf<U32> Mul(const ModCtx<U32>& ctx, const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    return integral_model::MatMul(ctx, A, B);
}
DenseBuf<U32> Minus(const ModCtx<U32>& ctx, const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    return integral_model::LinComb(ctx, U32(1), A, ctx.neg(1), B);
}
bool Equal(const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}
bool IsZero(const DenseBuf<U32>& A) {
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) != 0) return false;
    return true;
}
bool InSpan(const ModCtx<U32>& ctx, const DenseBuf<U32>& B, const DenseBuf<U32>& Y) {
    if (Y.cols == 0 || IsZero(Y)) return true;
    if (B.cols == 0) return false;
    return Len(ctx, integral_model::HStack(B, Y)) == Len(ctx, B);
}
DenseBuf<U32> Identity(U32 n) {
    DenseBuf<U32> I(n, n);
    for (U32 i = 0; i < n; ++i) I.at(i, i) = 1;
    return I;
}
DenseBuf<U32> Transpose(const DenseBuf<U32>& A) {
    DenseBuf<U32> T(A.cols, A.rows);
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    return T;
}
// The 2 x 2 block matrix with rows (A B) and (C D) (any block may be empty in one dimension).
DenseBuf<U32> Blocks(const DenseBuf<U32>& A, const DenseBuf<U32>& B, const DenseBuf<U32>& C, const DenseBuf<U32>& D) {
    ENGINE_CHECK(A.rows == B.rows && C.rows == D.rows && A.cols == C.cols && B.cols == D.cols);
    DenseBuf<U32> M(A.rows + C.rows, A.cols + B.cols);
    for (U32 i = 0; i < A.rows; ++i) {
        for (U32 j = 0; j < A.cols; ++j) M.at(i, j) = A.at(i, j);
        for (U32 j = 0; j < B.cols; ++j) M.at(i, A.cols + j) = B.at(i, j);
    }
    for (U32 i = 0; i < C.rows; ++i) {
        for (U32 j = 0; j < C.cols; ++j) M.at(A.rows + i, j) = C.at(i, j);
        for (U32 j = 0; j < D.cols; ++j) M.at(A.rows + i, C.cols + j) = D.at(i, j);
    }
    return M;
}
DenseBuf<U32> Diag2(const DenseBuf<U32>& A, const DenseBuf<U32>& D) {
    return Blocks(A, DenseBuf<U32>(A.rows, D.cols), DenseBuf<U32>(D.rows, A.cols), D);
}
DenseBuf<U32> Scale(const ModCtx<U32>& ctx, U32 a, const DenseBuf<U32>& A) {
    return integral_model::LinComb(ctx, a, A, U32(0), A);
}
std::string Str(const std::vector<U32>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

// ---- the cohomology of a cochain complex given by matrices -----------------
// delta[q] : C^q -> C^{q+1} (q = 0 .. Q), P[q] the projector onto the cochain
// group inside the full layout (q = 0 .. Q + 1); delta = P delta P checked.
struct Coh {
    std::vector<DenseBuf<U32>> delta, P, G, B;
    std::vector<U32> lenH;
    std::vector<std::vector<U32>> exps;   // descending
};
Coh CohOf(const ModCtx<U32>& ctx, std::vector<DenseBuf<U32>> delta, std::vector<DenseBuf<U32>> P, bool gens, Tally& T,
          const std::string& tag) {
    Coh c;
    c.delta = std::move(delta);
    c.P = std::move(P);
    const U32 Q = static_cast<U32>(c.delta.size()) - 1;
    ENGINE_CHECK(c.P.size() == Q + 2);
    c.lenH.assign(Q + 1, 0);
    c.exps.assign(Q + 1, {});
    bool pdp = true;
    for (U32 q = 0; q <= Q; ++q) {
        c.B.push_back(q == 0 ? DenseBuf<U32>(c.P[0].rows, 0) : c.delta[q - 1]);
        const U32 lp = Len(ctx, c.P[q]), li = Len(ctx, c.delta[q]), lb = Len(ctx, c.B[q]);
        ENGINE_CHECK_MSG(lp >= li + lb, "CohOf: negative length");
        c.lenH[q] = lp - li - lb;
        pdp = pdp && Equal(Mul(ctx, c.delta[q], c.P[q]), c.delta[q]) && Equal(Mul(ctx, c.P[q + 1], c.delta[q]), c.delta[q]);
        if (q >= 1) pdp = pdp && IsZero(Mul(ctx, c.delta[q], c.delta[q - 1]));
        if (!gens) {
            c.G.push_back(DenseBuf<U32>(c.P[q].rows, 0));
            continue;
        }
        const DenseBuf<U32> Z = KernelCols(ctx, c.delta[q]);
        DenseBuf<U32> G = (Z.cols == 0) ? DenseBuf<U32>(c.P[q].rows, 0) : Mul(ctx, c.P[q], Z);
        const U32 lg = Len(ctx, G);
        T.check(lg >= lb && lg - lb == c.lenH[q], tag + ": len(P Z^q) - len(B^q) = len H^q at q=" + std::to_string(q));
        const auto N = integral_model::Subquotient(ctx, G, c.B[q]);
        c.exps[q] = N.exponents;
        std::sort(c.exps[q].rbegin(), c.exps[q].rend());
        c.G.push_back(std::move(G));
    }
    T.check(pdp, tag + ": delta = P delta P and delta o delta = 0");
    return c;
}
Coh CohOfModel(const LocalModel& M, const ModCtx<U32>& ctx, const Mods& Mod, bool gens, Tally& T, const std::string& tag,
               bool boundary = false) {
    std::vector<DenseBuf<U32>> d, P;
    for (U32 q = 0; q <= M.q_max(); ++q)
        d.push_back(boundary ? M.boundary_coboundary(ctx, Mod, q) : M.coboundary(ctx, Mod, q));
    for (U32 q = 0; q <= M.q_max() + 1; ++q)
        P.push_back(boundary ? M.boundary_projector(ctx, Mod, q) : M.projector(ctx, Mod, q));
    return CohOf(ctx, std::move(d), std::move(P), gens, T, tag);
}

// The restriction C^q -> C^q_d (boundary cells) of a model, cell * D + component.
DenseBuf<U32> Restriction(const LocalModel& M, U32 q, U32 D) {
    U32 nb = 0;
    for (const auto& c : M.cells(q)) nb += c.boundary ? 1 : 0;
    DenseBuf<U32> R(nb * D, M.rank(q) * D);
    U32 b = 0;
    for (U32 ci = 0; ci < M.rank(q); ++ci) {
        if (!M.cells(q)[ci].boundary) continue;
        for (U32 i = 0; i < D; ++i) R.at(b * D + i, ci * D + i) = 1;
        ++b;
    }
    return R;
}

// ---- the closed form H^{>=1}(Z/3; M^{-I}) for any module ------------------
struct ClosedForm {
    U32 h_even = 0, h_odd = 0;
};
ClosedForm ClosedFormZ3(const ModCtx<U32>& ctx, const Mods& M, const ColumnSpec& col) {
    std::map<GKey, DenseBuf<U32>> memo;
    const DenseBuf<U32> Pe = integral_model::RhoQ(ctx, col.e, M, memo);
    const DenseBuf<U32> H = M.rho(col.pow[col.q_order % col.m]);
    const DenseBuf<U32> A = Minus(ctx, Mul(ctx, H, Pe), Pe);
    DenseBuf<U32> N = Pe, Hi = Pe;
    for (U32 i = 1; i < col.p_order; ++i) {
        Hi = Mul(ctx, H, Hi);
        N = integral_model::LinComb(ctx, U32(1), N, U32(1), Hi);
    }
    const U32 lp = Len(ctx, Pe), la = Len(ctx, A), ln = Len(ctx, N);
    ClosedForm cf;
    cf.h_even = (lp - la) - ln;
    cf.h_odd = (lp - ln) - la;
    return cf;
}

// ---- Ind_{Gamma_0(N)}^Gamma Sym^n as a coefficient module (the SHIPPED table) --
Mods InducedModule(std::shared_ptr<const hecke::level::Gamma0LevelN> A, U32 n, const ModCtx<U32>& ctx) {
    Mods M;
    M.name = "Ind_{Gamma_0(" + std::to_string(A->N()) + ")}^Gamma Sym^" + std::to_string(n);
    M.dim = A->m() * (n + 1);
    auto cache = std::make_shared<std::map<GKey, DenseBuf<U32>>>();
    const U32 D = M.dim;
    M.rho = [A, n, ctx, cache, D](const Mat22& g) {
        const GKey k = integral_model::key_of(g);
        auto it = cache->find(k);
        if (it != cache->end()) return it->second;
        const auto mono = hecke::level::MonomialOf(*A, g, n, ctx);
        const std::vector<U32> dense = mono.dense();
        DenseBuf<U32> R(D, D);
        for (U32 i = 0; i < D; ++i)
            for (U32 j = 0; j < D; ++j) R.at(i, j) = dense[static_cast<size_t>(i) * D + j];
        cache->emplace(k, R);
        return R;
    };
    return M;
}

// The shipped level-N pipeline over F_3 (def:levelNmodel (a), the cochain level): dims of H^0..H^2.
std::vector<U32> PipelineDimsLevel(const SimplicialComplex& K, const hecke::level::Gamma0LevelN& A, U32 n,
                                   const ModCtx<U32>& ctx) {
    const U32 D = A.m() * (n + 1);
    LocalSystemSimplicialComplex L(K, hecke::level::InducedRelationsLevel(K, A, n, ctx), D, ctx.p);
    EquivariantCohomologyOnK coh(&L);
    coh.compute_cohomology();
    std::vector<U32> d = coh.get_H_eq_dimensions();
    d.resize(3, 0);
    return d;
}

const ColumnSpec* PeriodicColumn(const LocalModel& M) {
    const ColumnSpec* per = nullptr;
    for (U32 s = 0; s < 3; ++s)
        for (U32 r = 0; r < M.num_reps(s); ++r)
            if (M.column(s, r).periodic()) per = &M.column(s, r);
    return per;
}

bool FullGrid() {
    const char* v = std::getenv("HECKE_ISOMODEL_ELL_FULL");
    if (v == nullptr) return false;
    const std::string s(v);
    if (s == "1") return true;
    throw std::invalid_argument("HECKE_ISOMODEL_ELL_FULL must be 1 or unset (got '" + s + "')");
}

// ===========================================================================
// (IL3N) the level-N clause of [C, IL3]
// ===========================================================================
U32 ArmIL3N(const LocalModel& freeM, const LocalModel& iso3, const ColumnSpec& per, bool full, Tally& T) {
    const U32 QMAX = iso3.q_max();
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    struct Cell {
        U32 e, N, n;
    };
    std::vector<Cell> cells;
    const std::vector<U32> f3N = full ? std::vector<U32>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                                      : std::vector<U32>{1, 2, 3, 4, 5, 7, 11, 13};
    const std::vector<U32> f3n = full ? std::vector<U32>{0, 1, 2, 3, 4, 5, 6} : std::vector<U32>{0, 1, 2, 4, 6};
    for (U32 N : f3N)
        for (U32 n : f3n) cells.push_back({1, N, n});
    for (U32 N : full ? std::vector<U32>{2, 3, 4, 5, 7, 11} : std::vector<U32>{2, 3, 5, 7})
        for (U32 n : full ? std::vector<U32>{0, 2, 4, 6} : std::vector<U32>{0, 2, 4}) cells.push_back({2, N, n});
    if (full)
        for (U32 N : {2u, 5u})
            for (U32 n : {2u, 4u}) cells.push_back({3, N, n});

    U32 done = 0, pipeline = 0, freeChecked = 0;
    for (const Cell& c : cells) {
        const ModCtx<U32> ctx = integral_model::RingContext(3, c.e);
        auto A = std::make_shared<const hecke::level::Gamma0LevelN>(c.N);
        const Mods Mod = InducedModule(A, c.n, ctx);
        const U32 D = Mod.dim;
        const std::string at = " N=" + std::to_string(c.N) + " n=" + std::to_string(c.n) + " e=" + std::to_string(c.e);
        // The module is a left action of Gamma, -I acts by (-1)^n, and at N = 1 it is Sym^n entry for entry.
        {
            const Mat22 S = Mat22::S(), Tm = Mat22::T(), ST = Mat22::ST(), negI = Mat22::I().neg();
            bool hom = Equal(Mul(ctx, Mod.rho(S), Mod.rho(Tm)), Mod.rho(ST)) &&
                       Equal(Mul(ctx, Mod.rho(ST), Mod.rho(S)), Mod.rho(ST * S)) && Equal(Mod.rho(Mat22::I()), Identity(D));
            const DenseBuf<U32> mI = Mod.rho(negI);
            const DenseBuf<U32> want = (c.n % 2 == 0) ? Identity(D) : Scale(ctx, ctx.neg(1), Identity(D));
            hom = hom && Equal(mI, want);
            if (c.N == 1)
                for (const Mat22& g : {S, Tm, ST, negI, ST * Tm * S})
                    hom = hom && Equal(Mod.rho(g), integral_model::rho_n(ctx, g, c.n));
            T.check(hom, "(IL3N) Ind module: a left action, -I by (-1)^n, = Sym^n at N = 1" + at);
        }
        const Coh h = CohOfModel(iso3, ctx, Mod, false, T, "(IL3N) Iso3" + at);
        // Iso^(3) = Iso with the induced module (IL1 at level N), where cheap.
        if (D <= (full ? 80u : 48u)) {
            const auto lf = freeM.cohomology_lengths(ctx, Mod);
            T.check(lf == h.lenH, "(IL3N) len H^q Iso^(3) = Iso with M = Ind" + at + ": " + Str(h.lenH) + " vs " + Str(lf));
            ++freeChecked;
        }
        const ClosedForm cf = ClosedFormZ3(ctx, Mod, per);
        if (c.n % 2 == 1) {
            bool zero = cf.h_even == 0 && cf.h_odd == 0;
            for (U32 q = 0; q <= QMAX; ++q) zero = zero && h.lenH[q] == 0;
            T.check(zero, "(IL3N) parity: -I acts by -1, H^. = 0 (prop:ellthreeclosed(4))" + at);
            if (c.e == 1) {
                const auto pd = PipelineDimsLevel(K, *A, c.n, ctx);
                T.check(pd[0] == 0 && pd[1] == 0 && pd[2] == 0, "(IL3N) the shipped level-N pipeline is 0 at odd n" + at);
                ++pipeline;
            }
            ++done;
            continue;
        }
        // The Bredon row: t = 0 cells with values in M^{Gamma_sigma}.
        std::vector<DenseBuf<U32>> J(QMAX + 2);
        for (U32 q = 0; q <= QMAX + 1; ++q) {
            std::vector<std::pair<U32, DenseBuf<U32>>> blocks;
            U32 cols = 0;
            for (U32 ci = 0; ci < iso3.rank(q); ++ci) {
                const auto& cell = iso3.cells(q)[ci];
                if (cell.t != 0) continue;
                const DenseBuf<U32> Adiff =
                    Minus(ctx, Mod.rho(iso3.descriptor().stabiliser_generator(cell.s, cell.rep)), Identity(D));
                DenseBuf<U32> Kc = KernelCols(ctx, Adiff);
                if (Kc.rows == 0) Kc = DenseBuf<U32>(D, 0);
                cols += Kc.cols;
                blocks.push_back({ci, std::move(Kc)});
            }
            J[q] = DenseBuf<U32>(iso3.rank(q) * D, cols);
            U32 off = 0;
            for (const auto& [ci, Kc] : blocks) {
                for (U32 i = 0; i < D; ++i)
                    for (U32 j = 0; j < Kc.cols; ++j) J[q].at(ci * D + i, off + j) = Kc.at(i, j);
                off += Kc.cols;
            }
        }
        std::vector<U32> hBr(QMAX + 1, 0), img(QMAX + 1, 0);
        bool sub = true;
        for (U32 q = 0; q <= QMAX; ++q) {
            const DenseBuf<U32> dJ = Mul(ctx, h.delta[q], J[q]);
            sub = sub && InSpan(ctx, J[q + 1], dJ);
            const U32 lZ = Len(ctx, J[q]) - Len(ctx, dJ);
            const U32 lB = (q == 0) ? 0 : Len(ctx, Mul(ctx, h.delta[q - 1], J[q - 1]));
            hBr[q] = lZ - lB;
            DenseBuf<U32> kz = KernelCols(ctx, dJ);
            const DenseBuf<U32> Zbr = (kz.cols == 0) ? DenseBuf<U32>(J[q].rows, 0) : Mul(ctx, J[q], kz);
            const U32 lb = Len(ctx, h.B[q]);
            img[q] = (Zbr.cols == 0) ? 0 : Len(ctx, integral_model::HStack(h.B[q], Zbr)) - lb;
        }
        T.check(sub, "(IL3N) the Bredon row is a subcomplex" + at);
        T.check(hBr[0] == h.lenH[0] && img[0] == hBr[0], "(IL3N) H^0: Bredon = group, the map an isomorphism" + at);
        T.check(img[1] == hBr[1], "(IL3N) H^1: the Bredon map is injective" + at);
        T.check(h.lenH[1] - img[1] == cf.h_odd, "(IL3N) H^1: cokernel = H^1(Z/3; M^{-I})" + at);
        T.check(hBr[2] == 0, "(IL3N) H^2_Bredon = 0 at ell = 3" + at);
        for (U32 q = 2; q <= QMAX; ++q)
            T.check(h.lenH[q] == ((q % 2 == 0) ? cf.h_even : cf.h_odd) && img[q] == 0,
                    "(IL3N) H^q = H^q(Z/3; M^{-I}), the Bredon route misses it" + at + " q=" + std::to_string(q));
        if (c.e == 1) {
            const auto pd = PipelineDimsLevel(K, *A, c.n, ctx);
            T.check(pd[0] == h.lenH[0] && pd[0] == hBr[0], "(IL3N) H^0 = the shipped level-N pipeline's" + at);
            T.check(pd[1] == hBr[1], "(IL3N) the Bredon row's H^1 = the shipped level-N pipeline's dim H^1_eq" + at);
            T.check(h.lenH[1] == pd[1] + cf.h_odd,
                    "(IL3N) dim H^1(Gamma; Ind) = dim H^1_eq(level N, pipeline) + dim H^1(Z/3; M^{-I})" + at);
            T.check(pd[2] == 0, "(IL3N) the pipeline's H^2 = 0" + at);
            ++pipeline;
            if (c.n == 2 || c.N == 7)
                std::cout << "ISOMODEL_ELL_IL3N N=" << c.N << " n=" << c.n << " D=" << D << " h0=" << h.lenH[0]
                          << " h1_eq_pipeline=" << pd[1] << " h1_Z3=" << cf.h_odd << " h1_group=" << h.lenH[1]
                          << " h2_group=" << h.lenH[2] << " h3_group=" << h.lenH[3] << "\n";
        }
        ++done;
    }
    std::cout << "ISOMODEL_ELL_IL3N cells=" << done << " pipeline_cells=" << pipeline << " iso_vs_iso3_cells=" << freeChecked
              << " (M = Ind_{Gamma_0(N)}^Gamma Sym^n through the shipped Gamma0LevelN table; F_3 against the level-N "
                 "pipeline, Z/9" << (full ? ", Z/27" : "") << " map-level)\n";
    return done;
}

// ===========================================================================
// (IL4e) [C, IL4] over Z/3^e, and the operator invariants
// ===========================================================================
// The image invariant factors of an operator X (a cochain map preserving Z) on H = G / B.
std::vector<U32> ImageExps(const ModCtx<U32>& ctx, const DenseBuf<U32>& X, const DenseBuf<U32>& G, const DenseBuf<U32>& B) {
    if (G.cols == 0) return {};
    auto e = integral_model::Subquotient(ctx, Mul(ctx, X, G), B).exponents;
    std::sort(e.rbegin(), e.rend());
    return e;
}

U32 ArmIL4e(const LocalModel& freeM, const LocalModel& iso3, const std::vector<std::unique_ptr<StrictLiftBundle>>& bundles,
            const ComparisonMap& Phi, const ComparisonMap& Psi, bool full, Tally& T) {
    const U32 QMAX = iso3.q_max();
    struct Cell {
        U32 e, n;
    };
    std::vector<Cell> cells;
    for (U32 n : full ? std::vector<U32>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20} : std::vector<U32>{0, 2, 4, 6, 8, 10, 12})
        cells.push_back({2, n});
    for (U32 n : full ? std::vector<U32>{0, 2, 4, 6, 8, 10, 12, 14, 16} : std::vector<U32>{0, 2, 4, 6, 8}) cells.push_back({3, n});
    U32 opInv = 0, polys = 0, done = 0;
    for (const Cell& c : cells) {
        const ModCtx<U32> ctx = integral_model::RingContext(3, c.e);
        const Mods Sym = integral_model::SymModule(ctx, c.n);
        const U32 D = c.n + 1;
        const std::string at = " e=" + std::to_string(c.e) + " n=" + std::to_string(c.n);
        const Coh h3 = CohOfModel(iso3, ctx, Sym, true, T, "(IL4e) Iso3" + at);
        const Coh hf = CohOfModel(freeM, ctx, Sym, true, T, "(IL4e) Iso" + at);
        std::vector<DenseBuf<U32>> PhiS, PsiS;
        for (U32 q = 0; q <= QMAX + 1; ++q) {
            PhiS.push_back(integral_model::Pullback(ctx, Sym, q, freeM, iso3, Phi));
            PsiS.push_back(integral_model::Pullback(ctx, Sym, q, iso3, freeM, Psi));
        }
        bool maps = true, qi = true;
        for (U32 q = 0; q <= QMAX; ++q) {
            maps = maps && Equal(Mul(ctx, hf.delta[q], PhiS[q]), Mul(ctx, PhiS[q + 1], h3.delta[q])) &&
                   Equal(Mul(ctx, h3.delta[q], PsiS[q]), Mul(ctx, PsiS[q + 1], hf.delta[q])) &&
                   Equal(Mul(ctx, h3.P[q], PsiS[q]), PsiS[q]);
            const DenseBuf<U32> e1 = Minus(ctx, Mul(ctx, PhiS[q], PsiS[q]), Identity(freeM.rank(q) * D));
            const DenseBuf<U32> e2 = Minus(ctx, Mul(ctx, PsiS[q], PhiS[q]), Identity(iso3.rank(q) * D));
            qi = qi && InSpan(ctx, hf.B[q], Mul(ctx, e1, hf.G[q])) && InSpan(ctx, h3.B[q], Mul(ctx, e2, h3.G[q]));
            T.check(h3.exps[q] == hf.exps[q], "(IL4e) invariant factors of H^q: Iso^(3) = Iso" + at + " q=" +
                                                  std::to_string(q) + ": " + Str(h3.exps[q]) + " vs " + Str(hf.exps[q]));
        }
        T.check(maps, "(IL4e) Phi^*, Psi^* cochain maps, Psi^* lands in C_{Iso^(3)}" + at);
        T.check(qi, "(IL4e) Phi^* Psi^* = id, Psi^* Phi^* = id on cohomology" + at);
        for (const auto& b : bundles) {
            const U32 p = static_cast<U32>(b->p());
            std::vector<DenseBuf<U32>> Th, Tp;
            for (U32 q = 0; q <= QMAX + 1; ++q) {
                Th.push_back(b->lift().theta_hat(ctx, c.n, q));
                Tp.push_back(Mul(ctx, PsiS[q], Mul(ctx, Th[q], PhiS[q])));
            }
            bool cm = true, pp = true;
            for (U32 q = 0; q <= QMAX; ++q) {
                cm = cm && Equal(Mul(ctx, h3.delta[q], Tp[q]), Mul(ctx, Tp[q + 1], h3.delta[q]));
                pp = pp && Equal(Mul(ctx, h3.P[q], Mul(ctx, Tp[q], h3.P[q])), Tp[q]);
            }
            const std::string atp = at + " p=" + std::to_string(p);
            T.check(cm && pp, "(IL4e) T'_p a cochain map of C_{Iso^(3)}, T' = P T' P" + atp);
            for (U32 q = 0; q <= QMAX; ++q) {
                for (U32 a : {U32(0), U32(1), ctx.neg(1)}) {
                    const DenseBuf<U32> Xf = integral_model::AddScalar(ctx, Th[q], ctx.neg(a));
                    const DenseBuf<U32> X3 = Minus(ctx, Tp[q], Scale(ctx, a, h3.P[q]));
                    const auto ef = ImageExps(ctx, Xf, hf.G[q], hf.B[q]);
                    const auto e3 = ImageExps(ctx, X3, h3.G[q], h3.B[q]);
                    T.check(ef == e3, "(IL4e) invariant factors of im(T'_p - a) on H^q(Iso^(3)) = im(T_p - a) on H^q(Iso)" +
                                          atp + " q=" + std::to_string(q) + " a=" + std::to_string(a) + ": " + Str(e3) +
                                          " vs " + Str(ef));
                    ++opInv;
                }
                bool free = !hf.exps[q].empty() && hf.exps[q].size() <= 7;
                for (U32 x : hf.exps[q]) free = free && x == c.e;
                if (free) {
                    const auto Nf = integral_model::Subquotient(ctx, hf.G[q], hf.B[q]);
                    const auto N3 = integral_model::Subquotient(ctx, h3.G[q], h3.B[q]);
                    const auto chiF = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, Nf, hf.B[q], Th[q]));
                    const auto chi3 = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, N3, h3.B[q], Tp[q]));
                    T.check(chiF == chi3, "(IL4e) charpoly of T'_p = T_p on the free H^q" + atp + " q=" + std::to_string(q));
                    ++polys;
                    if (c.n == 6 && q == 1 && c.e == 3)
                        std::cout << "ISOMODEL_ELL_IL4E e=3 n=6 q=1 p=" << p << " H=" << Str(hf.exps[q])
                                  << " charpoly(low->high)=" << Str(chi3) << "\n";
                }
            }
        }
        ++done;
    }
    std::cout << "ISOMODEL_ELL_IL4E cells=" << done << " image_invariants=" << opInv << " free_charpolys=" << polys
              << " (Z/9, Z/27; p in {2,3,5}; a in {0,1,-1}; q<=" << QMAX << ")\n";
    return done;
}

// ===========================================================================
// (PC) the pair / cone half of [C, prop:isomodelell](5)
// ===========================================================================
struct PairData {
    std::vector<DenseBuf<U32>> d, db, P, Pb, res;   // d, P to QMAX+1; db, Pb, res likewise
};
PairData PairOf(const LocalModel& M, const ModCtx<U32>& ctx, const Mods& Mod) {
    PairData pd;
    const U32 Q = M.q_max();
    for (U32 q = 0; q <= Q; ++q) {
        pd.d.push_back(M.coboundary(ctx, Mod, q));
        pd.db.push_back(M.boundary_coboundary(ctx, Mod, q));
    }
    for (U32 q = 0; q <= Q + 1; ++q) {
        pd.P.push_back(M.projector(ctx, Mod, q));
        pd.Pb.push_back(M.boundary_projector(ctx, Mod, q));
        pd.res.push_back(Restriction(M, q, Mod.dim));
    }
    return pd;
}
// Cone(res)^q = C^q (+) C_d^{q-1}, d(a, b) = (delta a, res a - delta_d b); q = 0 .. Q.
Coh ConeOf(const ModCtx<U32>& ctx, const PairData& pd, bool gens, Tally& T, const std::string& tag) {
    const U32 Q = static_cast<U32>(pd.d.size()) - 1;
    auto bdim = [&](int q) { return q < 0 ? U32(0) : pd.Pb[static_cast<U32>(q)].rows; };
    std::vector<DenseBuf<U32>> d, P;
    for (U32 q = 0; q <= Q; ++q) {
        const DenseBuf<U32> nb = (q == 0) ? DenseBuf<U32>(bdim(0), 0) : Scale(ctx, ctx.neg(1), pd.db[q - 1]);
        d.push_back(Blocks(pd.d[q], DenseBuf<U32>(pd.d[q].rows, bdim(static_cast<int>(q) - 1)), pd.res[q], nb));
    }
    for (U32 q = 0; q <= Q + 1; ++q)
        P.push_back(Diag2(pd.P[q], q == 0 ? DenseBuf<U32>(0, 0) : pd.Pb[q - 1]));
    return CohOf(ctx, std::move(d), std::move(P), gens, T, tag);
}

U32 ArmPC(const LocalModel& freeM, const LocalModel& iso3, const std::vector<std::unique_ptr<StrictLiftBundle>>& bundles,
          const ComparisonMap& Phi, const ComparisonMap& Psi, bool full, Tally& T) {
    const U32 QMAX = iso3.q_max();
    struct Cell {
        U32 e, n;
    };
    std::vector<Cell> cells;
    for (U32 n : full ? std::vector<U32>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20} : std::vector<U32>{0, 2, 4, 6, 8, 10, 12})
        cells.push_back({1, n});
    for (U32 n : full ? std::vector<U32>{0, 2, 4, 6, 8, 10, 12} : std::vector<U32>{0, 2, 4, 6, 8}) cells.push_back({2, n});
    U32 polys = 0, skipped = 0, inv = 0, done = 0;
    for (const Cell& c : cells) {
        const ModCtx<U32> ctx = integral_model::RingContext(3, c.e);
        const Mods Sym = integral_model::SymModule(ctx, c.n);
        const std::string at = " e=" + std::to_string(c.e) + " n=" + std::to_string(c.n);
        const PairData pf = PairOf(freeM, ctx, Sym), p3 = PairOf(iso3, ctx, Sym);
        // The restriction is a cochain map on both models.
        bool resOk = true;
        for (U32 q = 0; q <= QMAX; ++q)
            resOk = resOk && Equal(Mul(ctx, pf.db[q], pf.res[q]), Mul(ctx, pf.res[q + 1], pf.d[q])) &&
                    Equal(Mul(ctx, p3.db[q], p3.res[q]), Mul(ctx, p3.res[q + 1], p3.d[q]));
        T.check(resOk, "(PC) res: C -> C_d a cochain map on Iso and Iso^(3)" + at);
        // Phi^*, Psi^* (and T_p) respect the pair: res f^* = f^*_d res, f^*_d := res f^* incl.
        std::vector<DenseBuf<U32>> PhiS, PsiS, PhiB, PsiB;
        bool pairOk = true;
        for (U32 q = 0; q <= QMAX + 1; ++q) {
            PhiS.push_back(integral_model::Pullback(ctx, Sym, q, freeM, iso3, Phi));   // C_3 -> C_I
            PsiS.push_back(integral_model::Pullback(ctx, Sym, q, iso3, freeM, Psi));   // C_I -> C_3
            PhiB.push_back(Mul(ctx, pf.res[q], Mul(ctx, PhiS[q], Transpose(p3.res[q]))));
            PsiB.push_back(Mul(ctx, p3.res[q], Mul(ctx, PsiS[q], Transpose(pf.res[q]))));
            pairOk = pairOk && Equal(Mul(ctx, pf.res[q], PhiS[q]), Mul(ctx, PhiB[q], p3.res[q])) &&
                     Equal(Mul(ctx, p3.res[q], PsiS[q]), Mul(ctx, PsiB[q], pf.res[q]));
        }
        T.check(pairOk, "(PC) Phi^*, Psi^* map the pair to the pair (boundary first, prop:isomodelell(4))" + at);
        const Coh cf = ConeOf(ctx, pf, true, T, "(PC) Cone(Iso)" + at);
        const Coh c3 = ConeOf(ctx, p3, true, T, "(PC) Cone(Iso3)" + at);
        const Coh bf = CohOf(ctx, pf.db, pf.Pb, true, T, "(PC) Iso_d" + at);
        const Coh b3 = CohOf(ctx, p3.db, p3.Pb, true, T, "(PC) Iso3_d" + at);
        for (U32 q = 0; q <= QMAX; ++q) {
            T.check(cf.exps[q] == c3.exps[q], "(PC) invariant factors of H^q(cone): Iso^(3) = Iso" + at + " q=" +
                                                  std::to_string(q) + ": " + Str(c3.exps[q]) + " vs " + Str(cf.exps[q]));
            T.check(bf.exps[q] == b3.exps[q], "(PC) invariant factors of H^q(boundary): Iso^(3) = Iso" + at + " q=" +
                                                  std::to_string(q));
        }
        // Cone-level comparison maps (degree q: C^q (+) C_d^{q-1}).
        auto coneMap = [&](const std::vector<DenseBuf<U32>>& A, const std::vector<DenseBuf<U32>>& Ab, U32 q) {
            return Diag2(A[q], q == 0 ? DenseBuf<U32>(0, 0) : Ab[q - 1]);
        };
        for (const auto& b : bundles) {
            const U32 p = static_cast<U32>(b->p());
            const std::string atp = at + " p=" + std::to_string(p);
            std::vector<DenseBuf<U32>> Th, ThB, Tc, Tc3;
            bool strictPair = true;
            for (U32 q = 0; q <= QMAX + 1; ++q) {
                Th.push_back(b->lift().theta_hat(ctx, c.n, q));
                ThB.push_back(Mul(ctx, pf.res[q], Mul(ctx, Th[q], Transpose(pf.res[q]))));
                strictPair = strictPair && Equal(Mul(ctx, pf.res[q], Th[q]), Mul(ctx, ThB[q], pf.res[q]));
            }
            T.check(strictPair, "(PC) T_p preserves the pair: res T_p = T_p^d res (tau_Iso(R_d) in R_d, (B))" + atp);
            for (U32 q = 0; q <= QMAX + 1; ++q) {
                Tc.push_back(coneMap(Th, ThB, q));
                const DenseBuf<U32> PhiC = coneMap(PhiS, PhiB, q), PsiC = coneMap(PsiS, PsiB, q);
                Tc3.push_back(Mul(ctx, PsiC, Mul(ctx, Tc[q], PhiC)));
            }
            bool cm = true;
            for (U32 q = 0; q <= QMAX; ++q)
                cm = cm && Equal(Mul(ctx, cf.delta[q], Tc[q]), Mul(ctx, Tc[q + 1], cf.delta[q])) &&
                     Equal(Mul(ctx, c3.delta[q], Tc3[q]), Mul(ctx, Tc3[q + 1], c3.delta[q])) &&
                     Equal(Mul(ctx, c3.P[q], Mul(ctx, Tc3[q], c3.P[q])), Tc3[q]);
            T.check(cm, "(PC) T_cone on Cone(Iso) and T'_cone = Psi^*_c T_cone Phi^*_c on Cone(Iso^(3)) are cochain maps" + atp);
            for (U32 q = 0; q <= QMAX; ++q) {
                const std::string atq = atp + " q=" + std::to_string(q);
                if (ctx.is_field()) {
                    for (int which = 0; which < 2; ++which) {   // 0: the cone, 1: the boundary
                        const Coh& F = which ? bf : cf;
                        const Coh& G3 = which ? b3 : c3;
                        const DenseBuf<U32>& TF = which ? ThB[q] : Tc[q];
                        const DenseBuf<U32> T3 =
                            which ? Mul(ctx, PsiB[q], Mul(ctx, ThB[q], PhiB[q])) : Tc3[q];
                        const auto Nf = integral_model::Subquotient(ctx, F.G[q], F.B[q]);
                        const auto N3 = integral_model::Subquotient(ctx, G3.G[q], G3.B[q]);
                        T.check(Nf.rank() == N3.rank(), "(PC) dim H^q equal" + atq);
                        if (Nf.rank() != N3.rank() || Nf.rank() > 7) {
                            ++skipped;   // dim > 7: beyond CharpolyDivisionFree's reach (FULL grid only; reported)
                            continue;
                        }
                        const auto chiF = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, Nf, F.B[q], TF));
                        const auto chi3 = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, N3, G3.B[q], T3));
                        T.check(chiF == chi3, std::string("(PC) charpoly of T'_p = T_p on H^q(") +
                                                  (which ? "boundary" : "cone") + ")" + atq);
                        ++polys;
                        if (!which && c.n == 10 && q == 2)
                            std::cout << "ISOMODEL_ELL_PC n=10 q=2 (cone) p=" << p << " dim=" << Nf.rank()
                                      << " charpoly(low->high)=" << Str(chi3) << "\n";
                    }
                } else {
                    for (U32 a : {U32(0), U32(1)}) {
                        const DenseBuf<U32> Xf = integral_model::AddScalar(ctx, Tc[q], ctx.neg(a));
                        const DenseBuf<U32> X3 = Minus(ctx, Tc3[q], Scale(ctx, a, c3.P[q]));
                        T.check(ImageExps(ctx, Xf, cf.G[q], cf.B[q]) == ImageExps(ctx, X3, c3.G[q], c3.B[q]),
                                "(PC) invariant factors of im(T' - a) on H^q(cone) = T's" + atq + " a=" + std::to_string(a));
                        ++inv;
                    }
                }
            }
        }
        if (c.e == 1 && (c.n == 0 || c.n == 10))
            std::cout << "ISOMODEL_ELL_PC e=1 n=" << c.n << " H(cone)=" << Str(cf.lenH) << " H(boundary)=" << Str(bf.lenH) << "\n";
        ++done;
    }
    if (!full) T.check(skipped == 0, "(PC) coverage: no charpoly skipped on the default grid (" + std::to_string(skipped) + ")");
    std::cout << "ISOMODEL_ELL_PC cells=" << done << " charpolys=" << polys << " skipped=" << skipped
              << " image_invariants_Z9=" << inv << " (Cone(res) and the boundary; F_3 and Z/9; p in {2,3,5})\n";
    return done;
}

// ===========================================================================
// (REF) the synthetic refined descriptor ([C, rmk:isomodellmp](3))
// ===========================================================================
Rat OrbifoldEuler(const IsotropyModel& M) {
    Rat chi = Rat::of(0);
    for (U32 s = 0; s < 3; ++s)
        for (U32 r = 0; r < M.reps(s).size(); ++r)
            chi = chi + Rat::of((s % 2 == 0) ? 1 : -1, static_cast<int64_t>(M.stabiliser_order(s, r)));
    return chi;
}

U32 ArmREF(const IsotropyModel& desc, const LocalModel& iso3, bool full, Tally& T) {
    const U32 QMAX = desc.q_max();
    // The shipped descriptor through the data path is the shipped model.
    {
        const IsotropyModel again(integral_model::SL2Z7VertexModelData(), QMAX);
        bool same = again.is_shipped_model() && desc.is_shipped_model();
        for (U32 q = 1; same && q <= desc.q_top(); ++q) {
            const auto& a = desc.differential(q);
            const auto& b = again.differential(q);
            same = a.size() == b.size();
            for (size_t i = 0; same && i < a.size(); ++i) same = a[i].row == b[i].row && a[i].col == b[i].col && a[i].coeff == b[i].coeff;
        }
        // slot-17 WP-17 (WP-7 audit MINOR-1): both sides run ONE code path (the shipped IsotropyModel(q_max) is
        // IsotropyModel(SL2Z7VertexModelData(), q_max)), so this is a descriptor round-trip, not a comparison with the
        // pre-refactor constructor; that is guarded by HeckeIsotropyModelValidation, HeckeStrictLiftValidation,
        // HeckeFreeModelValidation, IsotropyModelDeltaSquaredValidation and (IL0)-(IL4).
        T.check(same, "(REF) descriptor round-trip: a second IsotropyModel(SL2Z7VertexModelData()) has the same d, entry by entry");
    }
    const ModelData base = integral_model::SL2Z7VertexModelData();
    // The two edges at zeta^2 = v4 (stabiliser <ST>, order 6), found from the inventory, not by name.
    U32 zeta = UINT32_MAX;
    for (U32 r = 0; r < base.reps[0].size(); ++r)
        if (base.order[0][r] % 3 == 0) zeta = base.reps[0][r][0];
    ENGINE_CHECK(zeta != UINT32_MAX);
    std::vector<integral_model::Simplex> atZeta;
    for (const auto& e : base.reps[1])
        if (std::find(e.begin(), e.end(), zeta) != e.end()) atZeta.push_back(e);
    ModelData dA = base;
    for (const auto& e : atZeta) dA = integral_model::SubdivideEdgeOrbit(dA, e);
    // B: A plus a horocycle edge.
    integral_model::Simplex hedge;
    for (U32 r = 0; r < dA.reps[1].size(); ++r)
        if (dA.bdry[1][r]) { hedge = dA.reps[1][r]; break; }
    const ModelData dB = integral_model::SubdivideEdgeOrbit(dA, hedge);
    // Every edge orbit at zeta^2 is subdivided (the edge to i and the edges towards the cusp, [C, rmk:isomodellmp](3)).
    T.check(atZeta.size() == 3, "(REF) three edge orbits at zeta^2, every one subdivided");
    // slot-17 WP-17 (WP-7 audit MINOR-2): SubdivideEdgeOrbit supports only pointwise-central edge stabilisers; the shipped
    // edges all have stabiliser {+-I}, and a planted non-central one (S, order 4) is refused by name.
    {
        bool central = true;
        for (U32 r = 0; r < base.reps[1].size(); ++r) central = central && base.order[1][r] == 2 && base.gen[1][r] == hecke::Mat22::I().neg();
        T.check(central, "(REF) every shipped edge representative has stabiliser {+-I} (pointwise-central)");
        ModelData planted = base;
        planted.gen[1][0] = hecke::Mat22::S();
        planted.order[1][0] = 4;
        bool refused = false;
        try {
            (void)integral_model::SubdivideEdgeOrbit(planted, planted.reps[1][0]);
        } catch (const std::logic_error& ex) {
            refused = std::string(ex.what()).find("not the centre") != std::string::npos;
        }
        T.check(refused, "(NC) SubdivideEdgeOrbit refuses an edge with a non-central stabiliser (planted S, order 4)");
    }

    // The data path's two guards.  (i) Re-orienting a representative (an interior edge with no transported copy) is
    // a different descriptor of the same complex: the derived face signs absorb it, the cohomology does not move.
    // (ii) (NC) one transport corrupted (g -> g T) is caught: a refused construction, a failed certificate, or a
    // different cohomology.
    {
        const ModelData F3data = dA;
        ModelData flipped = dA;
        bool flippedOne = false;
        for (U32 r = 0; r < flipped.reps[1].size() && !flippedOne; ++r) {
            bool hasCopy = false;
            for (const auto& nr : flipped.nonreps) hasCopy = hasCopy || (nr.face.size() == 2 && nr.rep == r);
            if (hasCopy || flipped.bdry[1][r]) continue;
            std::swap(flipped.reps[1][r][0], flipped.reps[1][r][1]);
            flippedOne = true;
        }
        const ModCtx<U32> F3 = integral_model::RingContext(3, 1);
        const IsotropyModel Ra(F3data, QMAX), Rf(flipped, QMAX);
        bool same = flippedOne && Rf.certify().all();
        for (U32 n : {0u, 2u, 4u, 6u}) same = same && Rf.cohomology_lengths(F3, n) == Ra.cohomology_lengths(F3, n);
        T.check(same, "(REF) re-orienting a representative edge: certificates held, cohomology unchanged");
        ModelData bad = dA;
        for (auto& nr : bad.nonreps)
            if (nr.face.size() == 2) {
                nr.g = nr.g * Mat22::T();
                break;
            }
        bool caught = false;
        std::string how;
        try {
            const IsotropyModel Rb(bad, QMAX);
            if (!Rb.certify().all()) {
                caught = true;
                how = "certificate";
            }
            for (U32 n : {0u, 2u, 4u, 6u, 10u, 12u})
                if (!caught && Rb.cohomology_lengths(F3, n) != desc.cohomology_lengths(F3, n)) {
                    caught = true;
                    how = "cohomology at n=" + std::to_string(n);
                }
        } catch (const std::exception& ex) {
            caught = true;
            const std::string w = ex.what();
            const size_t at = w.find("IsotropyModel");
            how = "refused: " + (at == std::string::npos ? w : w.substr(at));
        }
        T.check(caught, "(NC) a corrupted transport in the refined descriptor is caught");
        std::cout << "ISOMODEL_ELL_REF_NC corrupted_transport caught=" << (caught ? 1 : 0) << " by=\"" << how.substr(0, 120)
                  << "\" reoriented_edge_invariant=" << (same ? 1 : 0) << "\n";
    }

    const Rat chi0 = OrbifoldEuler(desc);
    U32 cells = 0;
    int which = 0;
    for (const ModelData* dp : std::vector<const ModelData*>{&dA, &dB}) {
        const ModelData& d = *dp;
        const std::string tag = std::string("(REF ") + (which == 0 ? "A" : "B") + ")";
        ++which;
        const IsotropyModel R(d, QMAX);
        const auto cert = R.certify();
        T.check(cert.all(), tag + " IsotropyModel on Y': d o d = 0, IM1, IM2, eps(R_d) on the horocycle: " + cert.first_failure);
        T.check(!R.is_shipped_model(), tag + " not mistaken for the shipped model");
        const auto Nq = d.simplex_counts();
        T.check(R.data().header.simplex_counts == Nq && Nq[0] - Nq[1] + Nq[2] == 1,
                tag + " the header states K' (N_q), Euler characteristic 1");
        T.check(OrbifoldEuler(R) == chi0 && chi0 == Rat::of(-1, 12), tag + " orbifold Euler characteristic -1/12 unchanged");
        bool refused = false;
        try {
            (void)R.comparison(integral_model::RingContext(3, 1), 0, 0);
        } catch (const std::logic_error&) {
            refused = true;
        }
        T.check(refused, tag + " comparison() (the shipped K's layout) is refused on Y'");
        const LocalModel L0(R, 0);
        const LocalModel L3(R, 3, LiftAveraging::On);
        T.check(L0.certify().all() && L3.certify().all(), tag + " LocalModel ell = 0, 3 on Y': certificates");
        bool idem = true;
        for (U32 q = 1; q <= R.q_top(); ++q) {
            const auto& a = R.differential(q);
            const auto& b = L0.differential(q);
            idem = idem && a.size() == b.size();
            for (size_t i = 0; idem && i < a.size(); ++i)
                idem = a[i].row == b[i].row && a[i].col == b[i].col && integral_model::QGroupRingElt::from(a[i].coeff) == b[i].coeff;
        }
        T.check(idem, tag + " (IL0) on Y': LocalModel at ell = 0 = IsotropyModel entry by entry");
        // Exactly one periodic column at ell = 3, at the vertex of order 6; every new vertex has stabiliser {+-I}.
        U32 periodic = 0, newV = 0, newVcentral = 0;
        for (U32 s = 0; s < 3; ++s)
            for (U32 r = 0; r < L3.num_reps(s); ++r)
                if (L3.column(s, r).periodic())
                    periodic += (s == 0 && L3.column(s, r).p_order == 3 && L3.column(s, r).q_order == 2) ? 1 : 100;
        for (U32 r = 0; r < d.reps[0].size(); ++r)
            if (std::find(base.reps[0].begin(), base.reps[0].end(), d.reps[0][r]) == base.reps[0].end()) {
                ++newV;
                newVcentral += (d.order[0][r] == 2 && d.gen[0][r] == Mat22::I().neg()) ? 1 : 0;
            }
        T.check(periodic == 1 && newV >= 1 && newV == newVcentral,
                tag + " one periodic column (zeta^2) at ell = 3; the new vertices have stabiliser {+-I}");
        U32 wantQ0 = static_cast<U32>(d.reps[0].size()), wantQ1 = static_cast<U32>(d.reps[1].size()) + 1,
            wantQ2 = static_cast<U32>(d.reps[2].size()) + 1;
        T.check(L3.rank(0) == wantQ0 && L3.rank(1) == wantQ1 && L3.rank(2) == wantQ2 && L3.rank(3) == 1,
                tag + " ranks of Iso^(3) on Y'");
        std::cout << "ISOMODEL_ELL_REF " << d.header.name << " N_q=" << Nq[0] << "/" << Nq[1] << "/" << Nq[2]
                  << " Sigma=" << d.reps[0].size() << "/" << d.reps[1].size() << "/" << d.reps[2].size()
                  << " Iso3_ranks=" << L3.rank(0) << "," << L3.rank(1) << "," << L3.rank(2) << "," << L3.rank(3)
                  << " chi_orb=" << OrbifoldEuler(R).num << "/" << OrbifoldEuler(R).den << " new_vertices=" << newV
                  << " certificates=" << (cert.all() ? "HELD" : "FAILED") << "\n";

        // Cohomology only: Iso^(3)(Y') = Iso^(3)(Xbulk) = Iso(Xbulk) in lengths and invariant factors.
        struct C {
            U32 e;
            std::vector<U32> ns;
        };
        const std::vector<C> grid =
            full ? std::vector<C>{{1, {0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20}}, {2, {0, 2, 4, 6, 8, 10, 12}}, {3, {0, 2, 4, 6, 8}}}
                 : std::vector<C>{{1, {0, 1, 2, 4, 6, 8, 12, 20}}, {2, {0, 2, 4, 6, 10, 12}}, {3, {0, 4, 6, 12}}};
        for (const C& g : grid) {
            const ModCtx<U32> ctx = integral_model::RingContext(3, g.e);
            for (U32 n : g.ns) {
                const Mods Sym = integral_model::SymModule(ctx, n);
                const std::string at = " e=" + std::to_string(g.e) + " n=" + std::to_string(n);
                const Coh hY = CohOfModel(L3, ctx, Sym, true, T, tag + " Iso3(Y')" + at);
                const Coh hX = CohOfModel(iso3, ctx, Sym, true, T, tag + " Iso3(X)" + at);
                const auto lI = desc.cohomology_lengths(ctx, n);
                const auto lR = R.cohomology_lengths(ctx, n);
                T.check(hY.exps == hX.exps && hY.lenH == lI && lR == lI,
                        tag + " H^q Iso^(3)(Y') = Iso^(3)(X) (invariant factors) = Iso(X) = Iso(Y') (lengths)" + at);
                T.check(L3.cohomology_lengths(ctx, n, true) == iso3.cohomology_lengths(ctx, n, true),
                        tag + " boundary: H^q(Gamma_d) on Y' = on X" + at);
                if (n % 2 == 1) {
                    bool zero = true;
                    for (U32 x : hY.lenH) zero = zero && x == 0;
                    T.check(zero, tag + " -I |-> -1 is admissible input: H^. = 0 on Y'" + at);
                }
                ++cells;
            }
        }
        // A level-N coefficient module on Y' (Shapiro on the refined model).
        {
            const ModCtx<U32> F3 = integral_model::RingContext(3, 1);
            for (U32 N : {2u, 5u}) {
                auto A = std::make_shared<const hecke::level::Gamma0LevelN>(N);
                const Mods Ind = InducedModule(A, 2, F3);
                T.check(L3.cohomology_lengths(F3, Ind) == iso3.cohomology_lengths(F3, Ind),
                        tag + " H^q(Gamma; Ind_{Gamma_0(" + std::to_string(N) + ")} Sym^2) on Y' = on X");
                ++cells;
            }
        }
        // ell = 2 through the same code on Y': every simplex keeps a Z/2 column; lengths = Iso(Y')'s.
        {
            const LocalModel L2(R, 2, LiftAveraging::On);
            bool allPeriodic = L2.certify().all();
            for (U32 s = 0; s < 3; ++s)
                for (U32 r = 0; r < L2.num_reps(s); ++r) allPeriodic = allPeriodic && L2.column(s, r).periodic();
            T.check(allPeriodic, tag + " (MP) ell = 2 on Y': certificates, every simplex keeps its Z/2 column");
            for (U32 e = 1; e <= 2; ++e) {
                const ModCtx<U32> ctx = integral_model::RingContext(2, e);
                for (U32 n : {0u, 1u, 2u, 4u, 6u}) {
                    T.check(L2.cohomology_lengths(ctx, n) == R.cohomology_lengths(ctx, n) &&
                                R.cohomology_lengths(ctx, n) == desc.cohomology_lengths(ctx, n),
                            tag + " (MP) ell = 2: len H^q Iso^(2)(Y') = Iso(Y') = Iso(X) at e=" + std::to_string(e) +
                                " n=" + std::to_string(n));
                    ++cells;
                }
            }
        }
    }
    std::cout << "ISOMODEL_ELL_REF cells=" << cells
              << " (cohomology only: Iso^(3)(Y') vs Iso^(3)(X) vs Iso; boundary; odd n; Ind_{Gamma_0(N)}; ell = 2)\n";
    return cells;
}

}  // namespace

IsoEllArmTally HeckeIsotropyModelEllOpenHalves(const IsotropyModel& desc, const LocalModel& freeM, const LocalModel& iso3,
                                               const std::vector<std::unique_ptr<StrictLiftBundle>>& bundles) {
    IsoEllArmTally tally;
    Tally T{&tally};
    const bool full = FullGrid();
    const ColumnSpec* per = PeriodicColumn(iso3);
    ENGINE_CHECK_MSG(per != nullptr, "HeckeIsotropyModelEllOpenHalves: no periodic column at ell = 3");
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto lap = [&](const char* arm) {
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        std::cout << "ISOMODEL_ELL_WP7_TIME arm=" << arm << " s=" << s << "\n";
        t0 = clock::now();
    };
    U32 cells = 0;
    cells += ArmIL3N(freeM, iso3, *per, full, T);
    lap("IL3N");
    const ComparisonMap Phi = integral_model::StrictComparison(freeM, iso3, LiftAveraging::On);
    const ComparisonMap Psi = integral_model::StrictComparison(iso3, freeM, LiftAveraging::On);
    cells += ArmIL4e(freeM, iso3, bundles, Phi, Psi, full, T);
    lap("IL4e");
    cells += ArmPC(freeM, iso3, bundles, Phi, Psi, full, T);
    lap("PC");
    cells += ArmREF(desc, iso3, full, T);
    lap("REF");
    if (full) std::cout << "EFFECTIVE HeckeIsotropyModelEllValidation arm=FULL cells=" << cells << "\n";
    std::cout << "ISOMODEL_ELL_WP7 checks=" << tally.checks << " failed=" << tally.failures << " grid=" << (full ? "FULL" : "default")
              << "\n";
    return tally;
}
