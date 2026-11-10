// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 16, WP-16 (stretch): HeckeIsotropyModelEllValidation.
// The ell-local isotropy model at ell = 3 against [C, rmk:isomodellcheckable]
// (IL1)-(IL4).  See the header for the arm list; report
// docs/implementation/slot16/WP-16_report.md.
#include "HeckeIsotropyModelEllValidation.h"
#include "HeckeIsotropyModelEllOpenHalves.h"   // slot-17 WP-7
#include "HeckeIsotropyModelEllStrictRefined.h"   // slot-18 WP-7

#include "Model/IntegralHecke.h"
#include "Model/IsotropyModel.h"
#include "Model/IsotropyModelEll.h"
#include "Model/StrictLift.h"

#include "../EquivariantCohomologyOnK.h"
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
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using integral_model::ColumnSpec;
using integral_model::ComparisonCertificate;
using integral_model::ComparisonMap;
using integral_model::GKey;
using integral_model::IsotropyModel;
using integral_model::LiftAveraging;
using integral_model::LocalCertificates;
using integral_model::LocalModel;
using integral_model::QGroupRingElt;
using integral_model::StrictLiftBundle;
using U32 = uint32_t;

struct Tally {
    uint32_t checks = 0;
    uint32_t failures = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

// ---- dense helpers ---------------------------------------------------------
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

// colspan Y inside colspan B (lengths).
bool InSpan(const ModCtx<U32>& ctx, const DenseBuf<U32>& B, const DenseBuf<U32>& Y) {
    if (Y.cols == 0 || IsZero(Y)) return true;
    if (B.cols == 0) return false;
    return Len(ctx, integral_model::HStack(B, Y)) == Len(ctx, B);
}

// ---- cohomology of a LocalModel at (ctx, n) --------------------------------
struct LocalCoh {
    std::vector<DenseBuf<U32>> delta, P, G, B;   // G = P Z (true cocycles), B = im delta^{q-1}
    std::vector<U32> lenH;
    std::vector<std::vector<U32>> exps;          // invariant-factor exponents of H^q (descending)
};

LocalCoh Cohomology(const LocalModel& M, const ModCtx<U32>& ctx, U32 n, bool generators, Tally& T,
                    const std::string& tag) {
    LocalCoh c;
    const U32 Q = M.q_max();
    const U32 D = n + 1;
    for (U32 q = 0; q <= Q; ++q) {
        c.delta.push_back(M.coboundary(ctx, n, q));
        c.P.push_back(M.projector(ctx, n, q));
    }
    c.lenH.assign(Q + 1, 0);
    c.exps.assign(Q + 1, {});
    for (U32 q = 0; q <= Q; ++q) {
        c.B.push_back(q == 0 ? DenseBuf<U32>(M.rank(0) * D, 0) : c.delta[q - 1]);
        const U32 lp = Len(ctx, c.P[q]), li = Len(ctx, c.delta[q]), lb = Len(ctx, c.B[q]);
        c.lenH[q] = lp - li - lb;
        if (!generators) {
            c.G.push_back(DenseBuf<U32>(M.rank(q) * D, 0));
            continue;
        }
        const DenseBuf<U32> Z = KernelCols(ctx, c.delta[q]);
        DenseBuf<U32> G = (Z.cols == 0) ? DenseBuf<U32>(M.rank(q) * D, 0) : Mul(ctx, c.P[q], Z);
        // The coboundary kills (1 - P) and lands in P (well-definedness at the cochain level).
        // (slot-16 WP-17, audit WP-16 F-7 (a)) P_{q+1} at every q, q = Q included (q + 1 <= q_top always).
        T.check(Equal(Mul(ctx, c.delta[q], c.P[q]), c.delta[q]) &&
                    Equal(Mul(ctx, M.projector(ctx, n, q + 1), c.delta[q]), c.delta[q]),
                tag + ": delta^" + std::to_string(q) + " = P delta P");
        const U32 lg = Len(ctx, G);
        T.check(lg >= lb && lg - lb == c.lenH[q], tag + ": len(P Z^q) - len(B^q) = len H^q at q=" + std::to_string(q));
        const auto N = integral_model::Subquotient(ctx, G, c.B[q]);
        c.exps[q] = N.exponents;
        std::sort(c.exps[q].rbegin(), c.exps[q].rend());
        c.G.push_back(std::move(G));
    }
    return c;
}

// ---- [IB] Appendix A (O1), the 3-primary part and the free rank ------------
// Generated from HeckeFreeModelValidation.cpp's kProbeGroupRows (the frozen O1,
// = [IB] Appendix A on the rows it prints), q = 0..6, n = 0..20: {free rank,
// exponents a of the Z/3^a summands}.
struct Grp {
    U32 fr;
    std::vector<U32> ex;
};
const std::vector<std::vector<Grp>> kO1Three = {
    /* n=0 */ {{1, {}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}},
    /* n=1 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=2 */ {{0, {}}, {1, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=3 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=4 */ {{0, {}}, {1, {1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}},
    /* n=5 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=6 */ {{0, {}}, {1, {1}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}},
    /* n=7 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=8 */ {{0, {}}, {1, {1}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=9 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=10 */ {{0, {}}, {3, {1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}},
    /* n=11 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=12 */ {{0, {}}, {1, {1, 2}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}},
    /* n=13 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=14 */ {{0, {}}, {3, {1}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=15 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=16 */ {{0, {}}, {3, {1, 1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}},
    /* n=17 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=18 */ {{0, {}}, {3, {1, 2}}, {0, {1}}, {0, {}}, {0, {1}}, {0, {}}, {0, {1}}},
    /* n=19 */ {{0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
    /* n=20 */ {{0, {}}, {3, {1, 1}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}, {0, {}}},
};

// UCT: H^q(Gamma; Sym^n (Z/3^e)^2) = H^q (x) Z/3^e + Tor(H^{q+1}, Z/3^e).
std::vector<U32> AppendixExponents(U32 n, U32 q, U32 e) {
    std::vector<U32> out;
    const Grp& g = kO1Three.at(n).at(q);
    for (U32 i = 0; i < g.fr; ++i) out.push_back(e);
    for (U32 a : g.ex) out.push_back(std::min(a, e));
    for (U32 a : kO1Three.at(n).at(q + 1).ex) out.push_back(std::min(a, e));
    std::sort(out.rbegin(), out.rend());
    return out;
}

U32 SumOf(const std::vector<U32>& v) {
    U32 s = 0;
    for (U32 x : v) s += x;
    return s;
}

std::string Str(const std::vector<U32>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

// ---- the closed form H^{>=1}(Z/3; M^{-I}) ----------------------------------
// The unique periodic column (found from the stabiliser inventory, not by
// vertex name): P = <h>, h = g^{|Q|};  M^{Q} = im rho(e).  Returns the common
// length of H^even = ker(h-1)/N and H^odd = ker N/(h-1) on M^Q (Herbrand: equal
// for a finite module; both are computed).
struct ClosedForm {
    U32 h_even = 0, h_odd = 0;
};
ClosedForm ClosedFormZ3(const ModCtx<U32>& ctx, U32 n, const ColumnSpec& col) {
    std::map<GKey, DenseBuf<U32>> memo;
    const DenseBuf<U32> Pe = integral_model::RhoQ(ctx, col.e, n, memo);
    const DenseBuf<U32> H = integral_model::rho_n(ctx, col.pow[col.q_order % col.m], n);
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

// The shipped Bredon route (EquivariantCohomologyOnK, SmallEllPolicy default
// on the 7-vertex model) at (n, ell): dims of H^0, H^1, H^2.
std::vector<U32> PipelineDims(const SimplicialComplex& K, U32 n, U32 ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    std::vector<U32> d = coh.get_H_eq_dimensions();
    d.resize(3, 0);
    return d;
}

}  // namespace

uint32_t HeckeIsotropyModelEllValidation() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    std::cout << "=== HeckeIsotropyModelEllValidation (slot-16 WP-16: the ell-local isotropy model at ell = 3; "
                 "[C, def:isomodelell], prop:isomodelell, prop:ellthreeclosed, rmk:isomodellcheckable (IL1)-(IL4)) ===\n";
    Tally T;
    const U32 QMAX = 5;

    // The three strict lifts (T_2, T_3 = U_3, T_5) and their models.
    std::vector<std::unique_ptr<StrictLiftBundle>> bundles;
    for (U32 p : {2u, 3u, 5u}) bundles.push_back(std::make_unique<StrictLiftBundle>(p, QMAX));
    const IsotropyModel& desc = bundles[0]->model();

    // ---------------------------------------------------------------- (IL0)
    {
        bool same = true;
        for (size_t b = 1; b < bundles.size(); ++b)
            for (U32 q = 1; q <= desc.q_top(); ++q) {
                const auto& d0 = desc.differential(q);
                const auto& d1 = bundles[b]->model().differential(q);
                same = same && d0.size() == d1.size();
                for (size_t i = 0; same && i < d0.size(); ++i)
                    same = d0[i].row == d1[i].row && d0[i].col == d1[i].col && d0[i].coeff == d1[i].coeff;
            }
        T.check(same, "(IL0) the three StrictLiftBundle models have the same differential");
    }
    const LocalModel freeM(desc, 0);
    {
        bool same = true;
        uint64_t entries = 0;
        for (U32 q = 0; q <= desc.q_top(); ++q) same = same && freeM.rank(q) == desc.rank(q);
        for (U32 q = 1; same && q <= desc.q_top(); ++q) {
            const auto& a = desc.differential(q);
            const auto& b = freeM.differential(q);
            same = a.size() == b.size();
            for (size_t i = 0; same && i < a.size(); ++i) {
                same = a[i].row == b[i].row && a[i].col == b[i].col && QGroupRingElt::from(a[i].coeff) == b[i].coeff;
                ++entries;
            }
        }
        T.check(same, "(IL0) LocalModel at ell = 0 reproduces IsotropyModel's d entry by entry");
        const LocalCertificates c = freeM.certify();
        T.check(c.all(), "(IL0) free model certificates: " + c.first_failure);
        std::cout << "ISOMODEL_ELL_IL0 free_rebuild=" << (same ? "IDENTICAL" : "DIFFERENT") << " entries=" << entries
                  << " ranks=" << freeM.rank(0) << "," << freeM.rank(1) << "," << freeM.rank(2) << ","
                  << freeM.rank(3) << "\n";
    }

    // ---------------------------------------------------------------- (IL1) structure
    const LocalModel iso3(desc, 3, LiftAveraging::On);
    const LocalModel iso3raw(desc, 3, LiftAveraging::Off);
    {
        const LocalCertificates c = iso3.certify();
        T.check(c.d_squared_zero, "(IL1) Iso^(3): d o d = 0 over Z_(3)[Gamma]: " + c.first_failure);
        T.check(c.well_defined, "(IL1) Iso^(3): every coefficient in e_col Q[Gamma] e_row");
        T.check(c.boundary_subcomplex, "(IL1) Iso^(3): IM2 (boundary sub-sum a subcomplex)");
        T.check(c.eps_chain_map, "(IL1) Iso^(3): IM1 (eps' a chain map onto C_.(Xbulk))");
        // [C, prop:ellthreeclosed](1): exactly one periodic column, at s = 0, |P| = 3, |Q| = 2.
        U32 periodic = 0;
        for (U32 s = 0; s < 3; ++s)
            for (U32 r = 0; r < iso3.num_reps(s); ++r)
                if (iso3.column(s, r).periodic()) {
                    ++periodic;
                    T.check(s == 0 && iso3.column(s, r).p_order == 3 && iso3.column(s, r).q_order == 2,
                            "(IL1) the periodic column is a vertex with |P| = 3, |Q| = 2");
                }
        T.check(periodic == 1, "(IL1) exactly one periodic column at ell = 3");
        const std::vector<U32> wantRank = {5, 10, 6, 1, 1, 1, 1};
        for (U32 q = 0; q <= iso3.q_top(); ++q)
            T.check(iso3.rank(q) == wantRank.at(q), "(IL1) rank Iso^(3)_" + std::to_string(q));
        // The boundary sub-model is the Bredon boundary complex: P_sigma = 1 on every boundary rep.
        bool bdryTrivial = true;
        for (U32 s = 0; s < 2; ++s)
            for (U32 r = 0; r < iso3.num_reps(s); ++r)
                if (desc.is_boundary_rep(s, r)) bdryTrivial = bdryTrivial && !iso3.column(s, r).periodic();
        T.check(bdryTrivial, "(IL1) P_sigma = 1 on the boundary ([C, B3])");
        const LocalCertificates cr = iso3raw.certify();
        std::cout << "ISOMODEL_ELL_IL1 ell=3 certificates=" << (c.all() ? "HELD" : "FAILED")
                  << " d2_terms=" << c.d_squared_terms_checked << " entries=" << c.entries_checked
                  << " eps_cells=" << c.eps_cells_checked << " ranks=5,10,6,1,1,1,1 lifts=" << iso3.stats().lifts
                  << " changed_by_average=" << iso3.stats().changed_by_average << "\n";
        // (NC, d-side) The average is a no-op on d at ell = 3: every lift of d starts at s >= 1, whose
        // Q_sigma = {+-I} is central and lies in every target Q_tau.  Reported, and checked.
        bool identical = true;
        for (U32 q = 1; q <= iso3.q_top(); ++q) {
            const auto& a = iso3.differential(q);
            const auto& b = iso3raw.differential(q);
            identical = identical && a.size() == b.size();
            for (size_t i = 0; identical && i < a.size(); ++i) identical = a[i].coeff == b[i].coeff;
        }
        T.check(identical && iso3.stats().changed_by_average == 0 && cr.all(),
                "(NC) the e_sigma average is a no-op on d at ell = 3 (s >= 1 stabilisers {+-I}, central)");
        std::cout << "ISOMODEL_ELL_NC_D average_on_d=no-op (sources s>=1 have Q={+-I}, central) identical="
                  << (identical ? 1 : 0) << "\n";
    }

    // ---------------------------------------------------------------- (IL1) lengths, invariant factors
    {
        U32 cells = 0;
        for (U32 e = 1; e <= 3; ++e) {
            const ModCtx<U32> ctx = integral_model::RingContext(3, e);
            for (U32 n = 0; n <= 20; ++n) {
                const bool gens = (e >= 2);
                const LocalCoh h3 = Cohomology(iso3, ctx, n, gens, T, "(IL1) Iso3 n=" + std::to_string(n));
                const LocalCoh hf = Cohomology(freeM, ctx, n, gens, T, "(IL1) Iso n=" + std::to_string(n));
                const auto lI = desc.cohomology_lengths(ctx, n);
                for (U32 q = 0; q <= QMAX; ++q) {
                    const std::vector<U32> app = AppendixExponents(n, q, e);
                    const std::string at = " at e=" + std::to_string(e) + " n=" + std::to_string(n) + " q=" + std::to_string(q);
                    T.check(h3.lenH[q] == hf.lenH[q] && hf.lenH[q] == lI[q], "(IL1) len H^q Iso^(3) = Iso" + at);
                    T.check(h3.lenH[q] == SumOf(app), "(IL1) len H^q Iso^(3) = Appendix A (UCT)" + at);
                    if (gens) {
                        T.check(h3.exps[q] == hf.exps[q], "(IL1) invariant factors Iso^(3) = Iso" + at + ": " +
                                                              Str(h3.exps[q]) + " vs " + Str(hf.exps[q]));
                        T.check(h3.exps[q] == app, "(IL1) invariant factors = Appendix A" + at + ": " +
                                                       Str(h3.exps[q]) + " vs " + Str(app));
                    }
                }
                // The boundary sub-model computes H^q(Gamma_d; M) like Iso's boundary.
                const auto b3 = iso3.cohomology_lengths(ctx, n, true);
                const auto bI = desc.cohomology_lengths(ctx, n, true);
                T.check(b3 == bI, "(IL1) boundary lengths Iso^(3)_d = Iso_d at e=" + std::to_string(e) + " n=" +
                                      std::to_string(n));
                ++cells;
            }
        }
        std::cout << "ISOMODEL_ELL_IL1 lengths cells=" << cells << " (Z/3^e, e<=3, n<=20, q<=5; Iso and Appendix A)\n";
    }

    // Cochain sizes of [C, prop:ellthreeclosed](1) over F_3.
    {
        const ModCtx<U32> F3 = integral_model::RingContext(3, 1);
        for (U32 n : {0u, 2u, 4u, 6u, 7u, 12u, 40u}) {
            const U32 D = n + 1;
            const DenseBuf<U32> S1 = Minus(F3, integral_model::rho_n(F3, hecke::Mat22::S(), n),
                                           integral_model::rho_n(F3, hecke::Mat22::I(), n));
            const U32 dimMS = KernelCols(F3, S1).cols;
            const U32 dimMm = (n % 2 == 0) ? D : 0;   // M^{-I}
            const std::vector<U32> want = {4 * dimMm + dimMS, 10 * dimMm, 6 * dimMm, dimMm, dimMm, dimMm};
            for (U32 q = 0; q <= QMAX; ++q)
                T.check(Len(F3, iso3.projector(F3, n, q)) == want[q],
                        "(IL1) dim C^q_{Iso^(3)} (prop:ellthreeclosed(1)) n=" + std::to_string(n) + " q=" + std::to_string(q));
        }
    }

    // ---------------------------------------------------------------- (IL2) closed form
    const ColumnSpec* per = nullptr;
    for (U32 r = 0; r < iso3.num_reps(0); ++r)
        if (iso3.column(0, r).periodic()) per = &iso3.column(0, r);
    // (slot-16 WP-17, audit WP-16 F-3) (IL1) only RECORDS a missing periodic column; refuse loudly before *per.
    ENGINE_CHECK_MSG(per != nullptr, "HeckeIsotropyModelEllValidation: no periodic s = 0 column at ell = 3 (IL1 failed)");
    {
        U32 cells = 0, pdpCells = 0, isoBeyond = 0;
        struct Arm {
            U32 e;
            std::vector<U32> ns;
        };
        std::vector<U32> f3;
        for (U32 n = 0; n <= 40; ++n) f3.push_back(n);
        for (U32 n : {50u, 51u, 80u, 101u, 120u, 160u, 199u, 200u}) f3.push_back(n);
        std::vector<U32> z9, z27;
        for (U32 n = 0; n <= 40; n += 1) z9.push_back(n);
        for (U32 n = 0; n <= 24; n += 2) z27.push_back(n);
        for (const Arm& arm : {Arm{1, f3}, Arm{2, z9}, Arm{3, z27}}) {
            const ModCtx<U32> ctx = integral_model::RingContext(3, arm.e);
            for (U32 n : arm.ns) {
                const LocalCoh h = Cohomology(iso3, ctx, n, false, T, "(IL2)");
                const ClosedForm cf = ClosedFormZ3(ctx, n, *per);
                const std::string at = " e=" + std::to_string(arm.e) + " n=" + std::to_string(n);
                // (slot-16 WP-17, audit WP-16 F-7 (a)) delta = P delta P on the IL2 cells too (n <= 40; their lenH rests on it).
                if (n <= 40) {
                    bool pdp = true;
                    for (U32 q = 0; q <= QMAX; ++q)
                        pdp = pdp && Equal(Mul(ctx, h.delta[q], h.P[q]), h.delta[q]) &&
                              Equal(Mul(ctx, iso3.projector(ctx, n, q + 1), h.delta[q]), h.delta[q]);
                    T.check(pdp, "(IL2) delta^q = P delta^q P for q <= 5" + at);
                    ++pdpCells;
                }
                // (F-7 (b)) [C] (IL1) Iso^(3) vs Iso beyond the Appendix-A range, n <= 200 over F_3, every q (incl. 0, 1).
                if (arm.e == 1 && n > 20) {
                    const auto lI = desc.cohomology_lengths(ctx, n);
                    bool same = true;
                    for (U32 q = 0; q <= QMAX; ++q) same = same && h.lenH[q] == lI[q];
                    T.check(same, "(IL1) len H^q Iso^(3) = Iso, q <= 5, beyond Appendix A" + at);
                    ++isoBeyond;
                }
                T.check(cf.h_even == cf.h_odd, "(IL2) Herbrand: H^even(Z/3) = H^odd(Z/3)" + at);
                for (U32 q = 2; q <= QMAX; ++q)
                    T.check(h.lenH[q] == ((q % 2 == 0) ? cf.h_even : cf.h_odd),
                            "(IL2) H^q(Gamma; M) = H^q(Z/3; M^{-I})" + at + " q=" + std::to_string(q));
                T.check(h.lenH[2] == h.lenH[4] && h.lenH[3] == h.lenH[5], "(IL2) 2-periodic" + at);
                if (n % 2 == 1) {
                    bool zero = cf.h_even == 0;
                    for (U32 q = 0; q <= QMAX; ++q) zero = zero && h.lenH[q] == 0;
                    T.check(zero, "(IL2) parity: odd n gives H^. = 0 (prop:ellthreeclosed(4))" + at);
                }
                if (arm.e == 1 && (n == 200 || n == 0 || n == 4 || n == 6))
                    std::cout << "ISOMODEL_ELL_IL2 e=1 n=" << n << " H=" << Str(h.lenH) << " H(Z/3;M^-I)=" << cf.h_even << "\n";
                ++cells;
            }
        }
        std::cout << "ISOMODEL_ELL_IL2 cells=" << cells << " (F_3 n<=40 and 8 cells to n=200; Z/9 n<=40; Z/27 even n<=24)\n";
        T.check(pdpCells == 95 && isoBeyond == 28, "(IL2) coverage: delta = P delta P on 95 cells, Iso^(3) = Iso on 28 cells n in (20, 200]");
        std::cout << "ISOMODEL_ELL_IL2 pdp_cells=" << pdpCells << " iso_vs_iso3_beyond_n20=" << isoBeyond << " (WP-17)\n";
    }

    // ---------------------------------------------------------------- (IL3) the Bredon comparison
    {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        U32 cells = 0;
        for (U32 e : {1u, 2u}) {
            const ModCtx<U32> ctx = integral_model::RingContext(3, e);
            const U32 nmax = (e == 1) ? 40 : 20;
            for (U32 n = 0; n <= nmax; ++n) {
                const U32 D = n + 1;
                const LocalCoh h = Cohomology(iso3, ctx, n, true, T, "(IL3)");
                const ClosedForm cf = ClosedFormZ3(ctx, n, *per);
                // The Bredon row: t = 0 cells with values in M^{Gamma_sigma} (= ker(rho(g_sigma) - 1)).
                std::vector<DenseBuf<U32>> J(QMAX + 2);
                for (U32 q = 0; q <= QMAX + 1; ++q) {
                    std::vector<std::pair<U32, DenseBuf<U32>>> blocks;
                    U32 cols = 0;
                    for (U32 ci = 0; ci < iso3.rank(q); ++ci) {
                        const auto& c = iso3.cells(q)[ci];
                        if (c.t != 0) continue;
                        const DenseBuf<U32> A = Minus(ctx, integral_model::rho_n(ctx, desc.stabiliser_generator(c.s, c.rep), n),
                                                      integral_model::rho_n(ctx, hecke::Mat22::I(), n));
                        DenseBuf<U32> Kc = KernelCols(ctx, A);
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
                for (U32 q = 0; q <= QMAX; ++q) {
                    const DenseBuf<U32> dJ = Mul(ctx, h.delta[q], J[q]);
                    T.check(InSpan(ctx, J[q + 1], dJ), "(IL3) the Bredon row is a subcomplex at q=" + std::to_string(q));
                    const U32 lZ = Len(ctx, J[q]) - Len(ctx, dJ);
                    const U32 lB = (q == 0) ? 0 : Len(ctx, Mul(ctx, h.delta[q - 1], J[q - 1]));
                    hBr[q] = lZ - lB;
                    DenseBuf<U32> kz = KernelCols(ctx, dJ);
                    const DenseBuf<U32> Zbr = (kz.cols == 0) ? DenseBuf<U32>(J[q].rows, 0) : Mul(ctx, J[q], kz);
                    const U32 lb = Len(ctx, h.B[q]);
                    img[q] = (Zbr.cols == 0) ? 0 : Len(ctx, integral_model::HStack(h.B[q], Zbr)) - lb;
                }
                const std::string at = " e=" + std::to_string(e) + " n=" + std::to_string(n);
                T.check(hBr[0] == h.lenH[0] && img[0] == hBr[0], "(IL3) H^0: Bredon = group, the map an isomorphism" + at);
                T.check(img[1] == hBr[1], "(IL3) H^1: the Bredon map is injective" + at);
                T.check(h.lenH[1] - img[1] == cf.h_odd, "(IL3) H^1: cokernel = H^1(Z/3; M^{-I})" + at);
                T.check(hBr[2] == 0, "(IL3) H^2_Bredon = 0 at ell = 3" + at);
                for (U32 q = 2; q <= QMAX; ++q)
                    T.check(h.lenH[q] - img[q] == ((q % 2 == 0) ? cf.h_even : cf.h_odd),
                            "(IL3) H^q: the Bredon route misses exactly H^q(Z/3; M^{-I})" + at + " q=" + std::to_string(q));
                if (e == 1) {
                    const std::vector<U32> pd = PipelineDims(K, n, 3);
                    T.check(pd[0] == h.lenH[0] && pd[0] == hBr[0], "(IL3) H^0 = shipped pipeline's" + at);
                    T.check(pd[1] == hBr[1], "(IL3) the Bredon row's H^1 = shipped pipeline's dim H^1_eq" + at);
                    T.check(h.lenH[1] == pd[1] + cf.h_odd,
                            "(IL3) dim H^1(Gamma) = dim H^1_Bredon (pipeline) + dim H^1(Z/3; M^{-I})" + at);
                    T.check(pd[2] == 0, "(IL3) the pipeline's H^2 = 0" + at);
                    std::cout << "ISOMODEL_ELL_IL3 n=" << n << " h0=" << h.lenH[0] << " h1_bredon=" << pd[1]
                              << " h1_Z3=" << cf.h_odd << " h1_group=" << h.lenH[1] << " h2_group=" << h.lenH[2] << "\n";
                }
                ++cells;
            }
        }
        std::cout << "ISOMODEL_ELL_IL3 cells=" << cells << " (F_3 n<=40 against the shipped pipeline; Z/9 n<=20 map-level)\n";
    }

    // ---------------------------------------------------------------- (IL4) comparison + Hecke transport
    {
        const ComparisonMap Phi = integral_model::StrictComparison(freeM, iso3, LiftAveraging::On);
        const ComparisonMap Psi = integral_model::StrictComparison(iso3, freeM, LiftAveraging::On);
        const ComparisonCertificate cPhi = integral_model::CertifyComparison(freeM, iso3, Phi);
        const ComparisonCertificate cPsi = integral_model::CertifyComparison(iso3, freeM, Psi);
        T.check(cPhi.all(), "(IL4) Phi_3 : Iso -> Iso^(3) strict chain map, well defined: " + cPhi.first_failure);
        T.check(cPsi.all(), "(IL4) Psi_3 : Iso^(3) -> Iso strict chain map, well defined: " + cPsi.first_failure);
        // (NC) Psi_3 without the average (C-1): not well defined.
        const ComparisonMap PsiRaw = integral_model::StrictComparison(iso3, freeM, LiftAveraging::Off);
        const ComparisonCertificate cRaw = integral_model::CertifyComparison(iso3, freeM, PsiRaw);
        // (slot-16 WP-17, audit WP-16 F-2) the count pinned (22 of the 25 source generators; every one of the 25 has
        // -I in Q_x at ell = 3, three are invariant already), and the other two certificates asserted, not printed:
        // the un-averaged Psi_3 IS a strict chain map -- only the well-definedness check catches it.
        T.check(!cRaw.well_defined && cRaw.not_well_defined == 22 && cRaw.cells_checked == 25,
                "(NC) the un-averaged Psi_3 is NOT well defined (e_x Psi(x) != Psi(x)) on exactly 22 of 25 generators (" +
                    std::to_string(cRaw.not_well_defined) + " of " + std::to_string(cRaw.cells_checked) + ")");
        T.check(cRaw.chain_map && cRaw.strict, "(NC) the un-averaged Psi_3 still passes the chain-map identity and strictness");
        std::cout << "ISOMODEL_ELL_IL4 Phi=" << (cPhi.all() ? "HELD" : "FAILED") << " (cells " << cPhi.cells_checked
                  << ", averaged " << Phi.stats.changed_by_average << ") Psi=" << (cPsi.all() ? "HELD" : "FAILED")
                  << " (cells " << cPsi.cells_checked << ", averaged " << Psi.stats.changed_by_average << ")\n";
        std::cout << "ISOMODEL_ELL_NC unaveraged_Psi well_defined=" << (cRaw.well_defined ? 1 : 0)
                  << " bad_cells=" << cRaw.not_well_defined << "/" << cRaw.cells_checked
                  << " chain_map_on_generators=" << (cRaw.chain_map ? 1 : 0) << " strict=" << (cRaw.strict ? 1 : 0)
                  << " first=\"" << cRaw.first_failure << "\"\n";

        const ModCtx<U32> F3 = integral_model::RingContext(3, 1);
        U32 polys = 0, skipped = 0, ncLeaves = 0, tcm = 0;
        for (U32 n = 0; n <= 12; n += 2) {
            const U32 D = n + 1;
            const LocalCoh h3 = Cohomology(iso3, F3, n, true, T, "(IL4) Iso3");
            const LocalCoh hf = Cohomology(freeM, F3, n, true, T, "(IL4) Iso");
            std::vector<DenseBuf<U32>> PhiS, PsiS, PsiRawS;
            for (U32 q = 0; q <= QMAX + 1; ++q) {
                PhiS.push_back(integral_model::Pullback(F3, n, q, freeM, iso3, Phi));   // C_{Iso3} -> C_I
                PsiS.push_back(integral_model::Pullback(F3, n, q, iso3, freeM, Psi));   // C_I -> C_{Iso3}
                PsiRawS.push_back(integral_model::Pullback(F3, n, q, iso3, freeM, PsiRaw));
            }
            const std::string at = " n=" + std::to_string(n);
            // (slot-16 WP-17, audit WP-16 F-6 (i)) T'_p = Psi^* Theta^_p Phi^* is a COCHAIN MAP of C_{Iso^(3)} and
            // T' = P T' P -- chain-level statements about Theta^ transported through Phi, Psi that the cohomology
            // identities below do not imply.
            for (size_t b = 0; b < bundles.size(); ++b) {
                std::vector<DenseBuf<U32>> Tq;
                for (U32 q = 0; q <= QMAX + 1; ++q)
                    Tq.push_back(Mul(F3, PsiS[q], Mul(F3, bundles[b]->lift().theta_hat(F3, n, q), PhiS[q])));
                const std::string atp = at + " p=" + std::to_string(bundles[b]->p());
                bool cm = true, pp = true;
                for (U32 q = 0; q <= QMAX; ++q) {
                    cm = cm && Equal(Mul(F3, h3.delta[q], Tq[q]), Mul(F3, Tq[q + 1], h3.delta[q]));
                    pp = pp && Equal(Mul(F3, h3.P[q], Mul(F3, Tq[q], h3.P[q])), Tq[q]);
                }
                T.check(cm, "(IL4) T'_p a cochain map of C_{Iso^(3)}: delta T'_q = T'_{q+1} delta, q <= 5" + atp);
                T.check(pp, "(IL4) T'_p = P T'_p P, q <= 5" + atp);
                ++tcm;
            }
            for (U32 q = 0; q <= QMAX; ++q) {
                const std::string atq = at + " q=" + std::to_string(q);
                T.check(Equal(Mul(F3, hf.delta[q], PhiS[q]), Mul(F3, PhiS[q + 1], h3.delta[q])),
                        "(IL4) Phi^* a cochain map" + atq);
                T.check(Equal(Mul(F3, h3.delta[q], PsiS[q]), Mul(F3, PsiS[q + 1], hf.delta[q])),
                        "(IL4) Psi^* a cochain map" + atq);
                T.check(Equal(Mul(F3, h3.P[q], PsiS[q]), PsiS[q]), "(IL4) Psi^* lands in C^q_{Iso^(3)}" + atq);
                if (!Equal(Mul(F3, h3.P[q], PsiRawS[q]), PsiRawS[q])) ++ncLeaves;
                // quasi-inverse on cohomology
                const U32 wI = freeM.rank(q) * D, w3 = iso3.rank(q) * D;
                DenseBuf<U32> II(wI, wI), I3(w3, w3);
                for (U32 i = 0; i < wI; ++i) II.at(i, i) = 1;
                for (U32 i = 0; i < w3; ++i) I3.at(i, i) = 1;
                const DenseBuf<U32> e1 = Minus(F3, Mul(F3, PhiS[q], PsiS[q]), II);
                const DenseBuf<U32> e2 = Minus(F3, Mul(F3, PsiS[q], PhiS[q]), I3);
                T.check(InSpan(F3, hf.B[q], Mul(F3, e1, hf.G[q])), "(IL4) Phi^* Psi^* = id on H^q(Iso)" + atq);
                T.check(InSpan(F3, h3.B[q], Mul(F3, e2, h3.G[q])), "(IL4) Psi^* Phi^* = id on H^q(Iso^(3))" + atq);
                // Hecke transport
                for (size_t b = 0; b < bundles.size(); ++b) {
                    const U32 p = static_cast<U32>(bundles[b]->p());
                    const DenseBuf<U32> Th = bundles[b]->lift().theta_hat(F3, n, q);
                    const DenseBuf<U32> Tp = Mul(F3, PsiS[q], Mul(F3, Th, PhiS[q]));
                    const auto Nf = integral_model::Subquotient(F3, hf.G[q], hf.B[q]);
                    const auto N3 = integral_model::Subquotient(F3, h3.G[q], h3.B[q]);
                    T.check(Nf.rank() == N3.rank(), "(IL4) dim H^q equal" + atq);
                    if (Nf.rank() > 7 || Nf.rank() != N3.rank()) {
                        ++skipped;
                        continue;
                    }
                    const auto chiF = integral_model::CharpolyDivisionFree(
                        F3, integral_model::OperatorMatrix(F3, Nf, hf.B[q], Th));
                    const auto chi3 = integral_model::CharpolyDivisionFree(
                        F3, integral_model::OperatorMatrix(F3, N3, h3.B[q], Tp));
                    T.check(chiF == chi3, "(IL4) charpoly of T'_" + std::to_string(p) + " on Iso^(3) = T_" +
                                              std::to_string(p) + " on Iso" + atq);
                    ++polys;
                    if (n == 12 && q == 1)
                        std::cout << "ISOMODEL_ELL_IL4 n=12 q=1 p=" << p << " dim=" << Nf.rank() << " charpoly(low->high)="
                                  << Str(chi3) << "\n";
                }
            }
        }
        T.check(ncLeaves > 0, "(NC) the un-averaged Psi^* does not land in C^._{Iso^(3)} = (+) M^{Q_sigma}");
        // (slot-16 WP-17, audit WP-16 F-6 (ii)) no silent skip: 7 n x 6 q x 3 p = 126 charpolys, none skipped.
        T.check(skipped == 0 && polys == 126 && tcm == 21, "(IL4) coverage: 126 charpolys, 0 skipped, 21 cochain-map transports (" +
                                                               std::to_string(polys) + ", " + std::to_string(skipped) + ", " +
                                                               std::to_string(tcm) + ")");
        std::cout << "ISOMODEL_ELL_IL4 charpolys=" << polys << " skipped_dim_gt_7=" << skipped
                  << " (F_3, n<=12 even, q<=5, p in {2,3,5}); NC unaveraged Psi^* leaves C_{Iso3} in " << ncLeaves
                  << " degree-cells\n";
    }

    // ---------------------------------------------------------------- (PHI) the column maps of prop:isomodelell(4)
    // (slot-16 WP-17, audit WP-16 F-9) phi_t = |Q|^{floor(t/2)} pi : F^sigma (x) R -> F'^sigma is a chain map, checked as an
    // identity in Q[Gamma_sigma] on the generator 1 (both columns are cyclic R[Gamma_sigma]-modules, d right multiplication by
    // x_t, pi(x) = x e_sigma):  phi_{t-1}(x_t) = x'_t phi_t(1), i.e. |Q|^{floor((t-1)/2)} x_t e = |Q|^{floor(t/2)} e x'_t,
    // t = 1 .. 6, on every periodic column; for P_sigma = 1, phi_0 (g - 1) = 0 (the target has g-bar = 1).  Degree 0:
    // the augmentations agree (eps(e) = 1).  "Defined over Z": the scalars |Q|^{floor(t/2)} are integers and pi is the
    // projection R[Gamma_sigma] -> R[Gamma_sigma] e = R[Gamma_sigma / Q_sigma], so no inverse of |Q| enters phi.
    {
        using integral_model::Rat;
        auto scaled = [](const QGroupRingElt& a, int64_t k) {
            QGroupRingElt z;
            z.add(a, Rat::of(k));
            return z;
        };
        auto ipow = [](int64_t b, U32 e) {
            int64_t r = 1;
            for (U32 i = 0; i < e; ++i) r *= b;
            return r;
        };
        U32 periodicCols = 0, trivialCols = 0, identities = 0;
        bool ok = true;
        for (U32 s = 0; s < 3; ++s)
            for (U32 r = 0; r < iso3.num_reps(s); ++r) {
                const ColumnSpec& cf = freeM.column(s, r);
                const ColumnSpec& c3 = iso3.column(s, r);
                const QGroupRingElt& e = c3.e;
                // eps(e) = 1: the coefficient sum of e
                Rat sum = Rat::of(0);
                for (const auto& kv : e.terms) sum = sum + kv.second;
                ok = ok && sum == Rat::of(1);
                if (!c3.periodic()) {
                    ok = ok && (integral_model::ColumnCoefficient(cf, 1) * e).is_zero();
                    ++trivialCols;
                    continue;
                }
                const int64_t qo = static_cast<int64_t>(c3.q_order);
                for (U32 t = 1; t <= 6; ++t) {
                    const QGroupRingElt lhs = scaled(integral_model::ColumnCoefficient(cf, t) * e, ipow(qo, (t - 1) / 2));
                    const QGroupRingElt rhs = scaled(e * integral_model::ColumnCoefficient(c3, t), ipow(qo, t / 2));
                    ok = ok && lhs == rhs;
                    ++identities;
                }
                ++periodicCols;
            }
        T.check(ok && periodicCols == 1 && trivialCols == 18 && identities == 6,
                "(PHI) phi_t = |Q|^{floor(t/2)} pi is a chain map F (x) R -> F' on every column (prop:isomodelell(4)): " +
                    std::to_string(periodicCols) + " periodic column(s), " + std::to_string(identities) + " identities, " +
                    std::to_string(trivialCols) + " P = 1 columns");
        std::cout << "ISOMODEL_ELL_PHI ell=3 periodic_columns=" << periodicCols << " identities=" << identities
                  << " p_trivial_columns=" << trivialCols << " result=" << (ok ? "HELD" : "FAILED") << " (WP-17)\n";
    }

    // ---------------------------------------------------------------- (MP) ell = 2 through the same code
    {
        const LocalModel iso2(desc, 2, LiftAveraging::On);
        const LocalCertificates c = iso2.certify();
        T.check(c.all(), "(MP) Iso^(2): d o d = 0, IM1, IM2, well defined: " + c.first_failure);
        bool allPeriodic = true;
        for (U32 s = 0; s < 3; ++s)
            for (U32 r = 0; r < iso2.num_reps(s); ++r) allPeriodic = allPeriodic && iso2.column(s, r).periodic();
        T.check(allPeriodic, "(MP) at ell = 2 every simplex keeps a Z/2 column (the centre is not trivialised)");
        U32 cells = 0;
        for (U32 e = 1; e <= 2; ++e) {
            const ModCtx<U32> ctx = integral_model::RingContext(2, e);
            for (U32 n = 0; n <= 10; ++n) {
                const auto l2 = iso2.cohomology_lengths(ctx, n);
                const auto lI = desc.cohomology_lengths(ctx, n);
                T.check(l2 == lI, "(MP) ell = 2: len H^q Iso^(2) = Iso at e=" + std::to_string(e) + " n=" + std::to_string(n));
                ++cells;
            }
        }
        std::cout << "ISOMODEL_ELL_MP ell=2 certificates=" << (c.all() ? "HELD" : "FAILED") << " ranks=" << iso2.rank(0)
                  << "," << iso2.rank(1) << "," << iso2.rank(2) << "," << iso2.rank(3) << " cells=" << cells
                  << " changed_by_average=" << iso2.stats().changed_by_average << "\n";
    }

    // ---------------------------------------------------------------- slot-17 WP-7: the open halves
    // (IL3N) the level-N clause, (IL4e) over Z/3^e, (PC) the pair / cone half of prop:isomodelell(5), (REF) the
    // synthetic refined descriptor of rmk:isomodellmp(3) -- HeckeIsotropyModelEllOpenHalvesValidationArms.cpp.
    {
        const IsoEllArmTally w = HeckeIsotropyModelEllOpenHalves(desc, freeM, iso3, bundles);
        T.checks += w.checks;
        T.failures += w.failures;
    }
    // (slot-18 WP-7) (IL5) the strict lift on a synthetic refined descriptor, (IL3Q) [C, IL3] over F_9 and Z/9 --
    // HeckeIsotropyModelEllStrictRefinedValidationArms.cpp ([C, sec:isomodelell] code map items (v), (vi)).
    {
        const IsoEllArmTally w = HeckeIsotropyModelEllStrictRefined(desc, freeM, iso3, bundles);
        T.checks += w.checks;
        T.failures += w.failures;
    }

    const double secs = std::chrono::duration<double>(clock::now() - t0).count();
    std::cout << "HeckeIsotropyModelEllValidation: " << T.checks << " checks, " << T.failures << " failed ("
              << secs << " s)\n";
    return T.failures;
}
