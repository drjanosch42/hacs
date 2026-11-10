// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 18, WP-7: (IL5) the strict lift on a synthetic refined
// descriptor, and (IL3Q) [C, IL3] beyond F_3 -- arms of
// HeckeIsotropyModelEllValidation.  See the header; report
// docs/implementation/slot18/WP-7_report.md.
#include "HeckeIsotropyModelEllStrictRefined.h"

#include "Model/IntegralHecke.h"
#include "Model/IsotropyModel.h"
#include "Model/IsotropyModelEll.h"
#include "Model/RefinedStrictLift.h"
#include "Model/StrictLift.h"

#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/HeckeLevelN.h"
#include "../Hecke/HeckeNebentypus.h"
#include "../Hecke/HeckeSettings.h"
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
using integral_model::RefinedStrictLift;
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

// ---- dense helpers (the slot-17 arms' set; HeckeIsotropyModelEllOpenHalvesValidationArms.cpp) ----
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
DenseBuf<U32> Scale(const ModCtx<U32>& ctx, U32 a, const DenseBuf<U32>& A) {
    return integral_model::LinComb(ctx, a, A, U32(0), A);
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
std::string Str(const std::vector<U32>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

struct Coh {
    std::vector<DenseBuf<U32>> delta, P, G, B;
    std::vector<U32> lenH;
    std::vector<std::vector<U32>> exps;   // descending
};
Coh CohOfModel(const LocalModel& M, const ModCtx<U32>& ctx, const Mods& Mod, Tally& T, const std::string& tag) {
    Coh c;
    const U32 Q = M.q_max();
    for (U32 q = 0; q <= Q; ++q) c.delta.push_back(M.coboundary(ctx, Mod, q));
    for (U32 q = 0; q <= Q + 1; ++q) c.P.push_back(M.projector(ctx, Mod, q));
    c.lenH.assign(Q + 1, 0);
    c.exps.assign(Q + 1, {});
    for (U32 q = 0; q <= Q; ++q) {
        c.B.push_back(q == 0 ? DenseBuf<U32>(c.P[0].rows, 0) : c.delta[q - 1]);
        const U32 lp = Len(ctx, c.P[q]), li = Len(ctx, c.delta[q]), lb = Len(ctx, c.B[q]);
        ENGINE_CHECK_MSG(lp >= li + lb, "CohOfModel: negative length");
        c.lenH[q] = lp - li - lb;
        const DenseBuf<U32> Z = KernelCols(ctx, c.delta[q]);
        DenseBuf<U32> G = (Z.cols == 0) ? DenseBuf<U32>(c.P[q].rows, 0) : Mul(ctx, c.P[q], Z);
        const U32 lg = Len(ctx, G);
        T.check(lg >= lb && lg - lb == c.lenH[q], tag + ": len(P Z^q) - len(B^q) = len H^q at q=" + std::to_string(q));
        c.exps[q] = integral_model::Subquotient(ctx, G, c.B[q]).exponents;
        std::sort(c.exps[q].rbegin(), c.exps[q].rend());
        c.G.push_back(std::move(G));
    }
    return c;
}
std::vector<U32> ImageExps(const ModCtx<U32>& ctx, const DenseBuf<U32>& X, const DenseBuf<U32>& G, const DenseBuf<U32>& B) {
    if (G.cols == 0) return {};
    auto e = integral_model::Subquotient(ctx, Mul(ctx, X, G), B).exponents;
    std::sort(e.rbegin(), e.rend());
    return e;
}

bool FullGrid() {
    const char* v = std::getenv("HECKE_ISOMODEL_ELL_FULL");
    if (v == nullptr) return false;
    const std::string s(v);
    if (s == "1") return true;
    throw std::invalid_argument("HECKE_ISOMODEL_ELL_FULL must be 1 or unset (got '" + s + "')");
}

// The two synthetic refined descriptors of the slot-17 (REF) arm: Y'_A = every edge orbit at the vertex whose
// stabiliser order is divisible by 3 (found from the inventory) subdivided; Y'_B = Y'_A plus a horocycle edge.
std::vector<ModelData> RefinedDescriptors() {
    const ModelData base = integral_model::SL2Z7VertexModelData();
    U32 zeta = UINT32_MAX;
    for (U32 r = 0; r < base.reps[0].size(); ++r)
        if (base.order[0][r] % 3 == 0) zeta = base.reps[0][r][0];
    ENGINE_CHECK(zeta != UINT32_MAX);
    ModelData dA = base;
    for (const auto& e : base.reps[1])
        if (std::find(e.begin(), e.end(), zeta) != e.end()) dA = integral_model::SubdivideEdgeOrbit(dA, e);
    integral_model::Simplex hedge;
    for (U32 r = 0; r < dA.reps[1].size(); ++r)
        if (dA.bdry[1][r]) {
            hedge = dA.reps[1][r];
            break;
        }
    return {dA, integral_model::SubdivideEdgeOrbit(dA, hedge)};
}

// ===========================================================================
// (IL5) the strict lift on a refined descriptor
// ===========================================================================
U32 ArmIL5(const IsotropyModel& desc, const LocalModel& freeM, const std::vector<std::unique_ptr<StrictLiftBundle>>& bundles,
           bool full, Tally& T) {
    const U32 QMAX = desc.q_max();
    U32 cells = 0, onH = 0, inv = 0, polys = 0, lifts = 0;
    int which = 0;
    for (const ModelData& d : RefinedDescriptors()) {
        const std::string tagY = std::string("(IL5 ") + (which == 0 ? "A" : "B") + ")";
        ++which;
        const IsotropyModel R(d, QMAX);
        T.check(R.certify().all() && !R.is_shipped_model(), tagY + " IsotropyModel on Y': certificates, not the shipped model");
        // The subdivision record and the two chain maps between C_.(Y') and C_.(Xbulk).
        const auto maps = integral_model::BuildSubdivisionMaps(desc, R);
        const auto mc = integral_model::CertifySubdivisionMaps(desc, R, maps);
        T.check(mc.all(), tagY + " collapse / subdivision: d c = c d, d sd = sd d, c sd = id: " + mc.first_failure);
        U32 newV = 0;
        for (const auto& v : R.data().reps[0]) newV += R.data().collapse.count(v[0]);
        for (const auto& nr : R.data().nonreps) newV += (nr.face.size() == 1) ? R.data().collapse.count(nr.face[0]) : 0;
        T.check(newV == R.data().collapse.size() && !R.data().refines.empty(),
                tagY + " the subdivision record names every new vertex of K'");
        for (U32 ell : full ? std::vector<U32>{3, 2} : std::vector<U32>{3, 2}) {
            const std::string tag = tagY + " ell=" + std::to_string(ell);
            // [C, IL5] read with F1's M-1: the hypothesis "Iso^(ell)(Y') is an ell-local isotropy model over Y'".
            const LocalModel A(R, ell, LiftAveraging::On);
            const auto ac = A.certify();
            T.check(ac.all(), tag + " hypothesis: Iso^(ell)(Y') is an ell-local isotropy model (d d = 0, well defined, IM1, IM2): " +
                                  ac.first_failure);
            const ComparisonMap Phi = integral_model::StrictComparisonOver(freeM, A, maps.subdivide);
            const ComparisonMap Psi = integral_model::StrictComparisonOver(A, freeM, maps.collapse);
            const auto c1 = integral_model::CertifyComparisonOver(freeM, A, maps.subdivide, Phi);
            const auto c2 = integral_model::CertifyComparisonOver(A, freeM, maps.collapse, Psi);
            T.check(c1.all() && c2.all(), tag + " Phi_Y over sd, Psi_Y over c: chain maps, strict, well defined: " +
                                              c1.first_failure + c2.first_failure);
            std::vector<std::unique_ptr<RefinedStrictLift>> tl;
            for (const auto& b : bundles) {
                tl.push_back(std::make_unique<RefinedStrictLift>(A, b->lift(), maps));
                const auto lc = tl.back()->certify();
                const std::string atp = tag + " p=" + std::to_string(b->p());
                T.check(lc.all(), atp + " tau' certificates (T1) (T2) (T3) (S): " + lc.first_failure);
                T.check(lc.schreier_generators >= 1 && lc.t1_checks > 0 && lc.t2_checks > 0 && lc.strict_checks > 0,
                        atp + " tau' certificates are non-vacuous");
                std::ostringstream sz;
                for (size_t q = 0; q < lc.terms.size(); ++q) sz << (q ? "," : "") << lc.terms[q];
                std::cout << "ISOMODEL_ELL_IL5_LIFT " << d.header.name << " ell=" << ell << " p=" << b->p()
                          << " ranks=" << A.rank(0) << "," << A.rank(1) << "," << A.rank(2) << "," << A.rank(3)
                          << " schreier=" << lc.schreier_generators << " t1=" << lc.t1_checks << " t2=" << lc.t2_checks
                          << " strict=" << lc.strict_checks << " changed_by_average=" << lc.changed_by_average
                          << " max_num=" << lc.max_abs_numerator << " max_den=" << lc.max_denominator << " terms=" << sz.str()
                          << "\n";
                ++lifts;
            }
            struct C {
                U32 e;
                std::vector<U32> ns;
            };
            const std::vector<C> grid =
                (ell == 3) ? (full ? std::vector<C>{{1, {0, 1, 2, 4, 6, 8, 10, 12}}, {2, {0, 2, 4, 6, 8}}, {3, {0, 2, 4, 6}}}
                                   : std::vector<C>{{1, {0, 1, 2, 4, 6, 12}}, {2, {0, 4, 6}}, {3, {0, 6}}})
                           : (full ? std::vector<C>{{1, {0, 1, 2, 4, 6, 8}}, {2, {0, 2, 4, 6}}, {3, {0, 2, 4}}}
                                   : std::vector<C>{{1, {0, 1, 2, 4}}, {2, {0, 2}}, {3, {0, 2}}});
            // (NC) the average is load-bearing on tau' at ell = 2 (Q_{v4} = <U^2>, a periodic column): the raw table is not a
            // map on R[Gamma] e_c; at ell = 3 the one non-central Q (<S> at the order-4 vertex) has a t = 0 column only, where
            // the section is canonical -- the average changes nothing (recorded, as the d-side's F-1).  And a perturbed
            // value is caught by the certificates.
            {
                uint64_t changed = 0;
                for (const auto& l : tl) changed += l->certify().changed_by_average;
                if (ell == 2) T.check(changed > 0, tag + " (NC) the raw table is not well defined on R[Gamma] e_c (the average changes it)");
                RefinedStrictLift bad = *tl[0];
                U32 cell = 0;
                while (cell + 1 < A.rank(1) && A.cells(1)[cell].t != 0) ++cell;
                bad.perturb(1, 1, cell);
                const auto bc = bad.certify();
                T.check(!bc.all(), tag + " (NC) one perturbed value of tau' is caught by the certificates");
                std::cout << "ISOMODEL_ELL_IL5_NC " << d.header.name << " ell=" << ell << " changed_by_average=" << changed
                          << " perturbed_caught=" << (bc.all() ? 0 : 1) << " by=\"" << bc.first_failure.substr(0, 60) << "\"\n";
            }
            for (const C& g : grid) {
                const ModCtx<U32> ctx = integral_model::RingContext(ell, g.e);
                for (U32 n : g.ns) {
                    const Mods Sym = integral_model::SymModule(ctx, n);
                    const std::string at = tag + " e=" + std::to_string(g.e) + " n=" + std::to_string(n);
                    const Coh hA = CohOfModel(A, ctx, Sym, T, at + " Iso^(ell)(Y')");
                    const Coh hX = CohOfModel(freeM, ctx, Sym, T, at + " Iso(X)");
                    T.check(hA.exps == hX.exps, at + " H^q(Iso^(ell)(Y')) = H^q(Iso(X)) (invariant factors)");
                    if (n % 2 == 1 && ell != 2) {   // at ell = 2, -1 = 1 mod 2: no control cell there
                        bool zero = true;
                        for (U32 x : hA.lenH) zero = zero && x == 0;
                        for (U32 x : hX.lenH) zero = zero && x == 0;
                        T.check(zero, at + " control: -I |-> -1, every H^q zero on Y' and on Xbulk");
                    }
                    std::vector<DenseBuf<U32>> PhiS, PsiS;
                    for (U32 q = 0; q <= QMAX + 1; ++q) {
                        PhiS.push_back(integral_model::Pullback(ctx, Sym, q, freeM, A, Phi));   // C_A -> C_X
                        PsiS.push_back(integral_model::Pullback(ctx, Sym, q, A, freeM, Psi));   // C_X -> C_A
                    }
                    std::vector<DenseBuf<U32>> Ts1, Th1;   // q = 1, per bundle (the discrimination control)
                    for (size_t bi = 0; bi < bundles.size(); ++bi) {
                        const auto& b = bundles[bi];
                        const std::string atp = at + " p=" + std::to_string(b->p());
                        std::vector<DenseBuf<U32>> Th, Ts, Tx;
                        for (U32 q = 0; q <= QMAX + 1; ++q) {
                            Tx.push_back(b->lift().theta_hat(ctx, n, q));
                            Th.push_back(Mul(ctx, PsiS[q], Mul(ctx, Tx[q], PhiS[q])));   // transported T'_p
                            Ts.push_back(tl[bi]->theta_hat(ctx, Sym, q));                // the strict T_p^{Y'}
                        }
                        bool cm = true, pp = true, same = true;
                        for (U32 q = 0; q <= QMAX; ++q) {
                            cm = cm && Equal(Mul(ctx, hA.delta[q], Ts[q]), Mul(ctx, Ts[q + 1], hA.delta[q])) &&
                                 Equal(Mul(ctx, hA.delta[q], Th[q]), Mul(ctx, Th[q + 1], hA.delta[q]));
                            pp = pp && Equal(Mul(ctx, hA.P[q], Mul(ctx, Ts[q], hA.P[q])), Ts[q]);
                            const bool eq = InSpan(ctx, hA.B[q], Mul(ctx, Minus(ctx, Ts[q], Th[q]), hA.G[q]));
                            same = same && eq;
                            ++onH;
                        }
                        Ts1.push_back(Ts[1]);
                        Th1.push_back(Th[1]);
                        T.check(cm && pp, atp + " T_p^{Y'} and T'_p cochain maps of C_{Iso^(ell)(Y')}, T = P T P");
                        T.check(same, atp + " [C, IL5]: T_p^{Y'} = T'_p on H^q (the difference maps cocycles to coboundaries)");
                        for (U32 q = 0; q <= QMAX; ++q) {
                            for (U32 a : {U32(0), U32(1), ctx.neg(1)}) {
                                const auto es = ImageExps(ctx, Minus(ctx, Ts[q], Scale(ctx, a, hA.P[q])), hA.G[q], hA.B[q]);
                                const auto et = ImageExps(ctx, Minus(ctx, Th[q], Scale(ctx, a, hA.P[q])), hA.G[q], hA.B[q]);
                                const auto ex = ImageExps(ctx, integral_model::AddScalar(ctx, Tx[q], ctx.neg(a)), hX.G[q], hX.B[q]);
                                T.check(es == et && es == ex,
                                        atp + " q=" + std::to_string(q) + " a=" + std::to_string(a) +
                                            ": invariant factors of im(T - a) on H^q, strict = transported = free: " + Str(es) +
                                            " " + Str(et) + " " + Str(ex));
                                ++inv;
                            }
                            if (g.e == 1 && hA.lenH[q] > 0) {
                                const auto Na = integral_model::Subquotient(ctx, hA.G[q], hA.B[q]);
                                const auto Nx = integral_model::Subquotient(ctx, hX.G[q], hX.B[q]);
                                const auto chiS = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, Na, hA.B[q], Ts[q]));
                                const auto chiT = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, Na, hA.B[q], Th[q]));
                                const auto chiX = integral_model::CharpolyDivisionFree(ctx, integral_model::OperatorMatrix(ctx, Nx, hX.B[q], Tx[q]));
                                T.check(chiS == chiT && chiS == chiX, atp + " q=" + std::to_string(q) +
                                                                          ": charpoly of T_p^{Y'} = T'_p = T_p(Iso(X)) over F_ell " + Str(chiS));
                                ++polys;
                                if (which == 1 && q == 1 && n == 12 && ell == 3)
                                    std::cout << "ISOMODEL_ELL_IL5 Y'_A ell=3 n=12 q=1 p=" << b->p() << " H=" << Str(hA.exps[q])
                                              << " charpoly(low->high)=" << Str(chiS) << "\n";
                            }
                        }
                    }
                    // (NC) the identity discriminates: T_2^{Y'} is NOT the transported T'_3 on H^1 (their charpolys differ here).
                    if (ell == 3 && g.e == 1 && n == 12 && Ts1.size() >= 2)
                        T.check(!InSpan(ctx, hA.B[1], Mul(ctx, Minus(ctx, Ts1[0], Th1[1]), hA.G[1])),
                                tag + " (NC) the equal-on-H^q test discriminates: T_2^{Y'} != T'_3 on H^1 at n=12");
                    ++cells;
                }
            }
        }
    }
    std::cout << "ISOMODEL_ELL_IL5 cells=" << cells << " lifts=" << lifts << " equal_on_H=" << onH << " image_invariants=" << inv
              << " charpolys=" << polys << " (ell in {3,2}; Y'_A, Y'_B; p in {2,3,5}; Z/ell^e, e<=3; strict tau' vs "
                 "transported vs free)\n";
    return cells;
}

// ===========================================================================
// (IL3Q) [C, IL3] beyond F_3: F_9 and Z/9
// ===========================================================================
using hecke::neb::Fq;
using hecke::neb::FqMat;

// Ind_{Gamma_0(N)}^Gamma V over F_q through the shipped twisted-monomial route (trivial character), dense.
struct FqModule {
    const Fq* F = nullptr;
    std::shared_ptr<const hecke::level::Gamma0LevelN> A;
    hecke::neb::BlockScalar scal;
    U32 n = 0, D = 0;
    mutable std::map<GKey, FqMat> memo;
    const FqMat& rho(const Mat22& g) const {
        const GKey k = integral_model::key_of(g);
        auto it = memo.find(k);
        if (it != memo.end()) return it->second;
        const auto tm = hecke::neb::TwistedMonomialOf(*A, g, n, *F, scal);
        FqMat R(D, D);
        for (U32 c = 0; c < D; ++c) {
            std::vector<U32> e(D, 0);
            e[c] = 1;
            const auto y = tm.apply(*F, e);
            for (U32 r = 0; r < D; ++r) R.at(r, c) = y[r];
        }
        return memo.emplace(k, std::move(R)).first->second;
    }
    U32 rat(const Rat& r) const { return F->mul(F->from_int(r.num), F->inv(F->from_int(r.den))); }
    FqMat rhoQ(const integral_model::QGroupRingElt& a) const {
        FqMat out(D, D);
        for (const auto& [g, r] : a.terms) out = hecke::neb::FqAdd(*F, out, hecke::neb::FqScale(*F, rho(integral_model::mat_of(g)), rat(r)));
        return out;
    }
};

// The coboundary delta^q and the projector P_q of a model over F_q (layout cell * D + component).
FqMat FqCoboundary(const LocalModel& M, const FqModule& Mod, U32 q) {
    const U32 D = Mod.D;
    FqMat X(M.rank(q + 1) * D, M.rank(q) * D);
    for (const auto& e : M.differential(q + 1)) {
        const FqMat R = Mod.rhoQ(e.coeff);
        for (U32 i = 0; i < D; ++i)
            for (U32 j = 0; j < D; ++j) {
                U32& v = X.at(e.col * D + i, e.row * D + j);
                v = Mod.F->add(v, R.at(i, j));
            }
    }
    return X;
}
FqMat FqProjector(const LocalModel& M, const FqModule& Mod, U32 q) {
    const U32 D = Mod.D;
    FqMat P(M.rank(q) * D, M.rank(q) * D);
    for (U32 c = 0; c < M.rank(q); ++c) {
        const auto& cell = M.cells(q)[c];
        const FqMat R = Mod.rhoQ(M.column(cell.s, cell.rep).e);
        for (U32 i = 0; i < D; ++i)
            for (U32 j = 0; j < D; ++j) P.at(c * D + i, c * D + j) = R.at(i, j);
    }
    return P;
}
std::vector<U32> FqDims(const LocalModel& M, const FqModule& Mod) {
    const Fq& F = *Mod.F;
    std::vector<U32> rk(M.q_max() + 1), rp(M.q_max() + 1), h(M.q_max() + 1);
    for (U32 q = 0; q <= M.q_max(); ++q) {
        rk[q] = hecke::neb::FqRank(F, FqCoboundary(M, Mod, q));
        rp[q] = hecke::neb::FqRank(F, FqProjector(M, Mod, q));
    }
    for (U32 q = 0; q <= M.q_max(); ++q) h[q] = rp[q] - rk[q] - (q == 0 ? 0 : rk[q - 1]);
    return h;
}
// H^{even / odd >= 1}(Z/3; M^{-I}) over F_q (ClosedFormZ3 of the slot-17 arms, in F_q).
std::pair<U32, U32> FqClosedForm(const FqModule& Mod, const ColumnSpec& col) {
    const Fq& F = *Mod.F;
    const FqMat Pe = Mod.rhoQ(col.e);
    const FqMat H = Mod.rho(col.pow[col.q_order % col.m]);
    const FqMat A = hecke::neb::FqAdd(F, hecke::neb::FqMul(F, H, Pe), hecke::neb::FqScale(F, Pe, F.neg(1)));
    FqMat N = Pe, Hi = Pe;
    for (U32 i = 1; i < col.p_order; ++i) {
        Hi = hecke::neb::FqMul(F, H, Hi);
        N = hecke::neb::FqAdd(F, N, Hi);
    }
    const U32 lp = hecke::neb::FqRank(F, Pe), la = hecke::neb::FqRank(F, A), ln = hecke::neb::FqRank(F, N);
    return {(lp - la) - ln, (lp - ln) - la};
}

// Ind_{Gamma_0(N)}^Gamma Sym^n over a ModCtx through the shipped MonomialOf (as the slot-17 arm (IL3N)).
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
std::pair<U32, U32> ClosedFormRing(const ModCtx<U32>& ctx, const Mods& M, const ColumnSpec& col) {
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
    return {(lp - la) - ln, (lp - ln) - la};
}
std::vector<U32> LengthsOf(const LocalModel& M, const ModCtx<U32>& ctx, const Mods& Mod) {
    return M.cohomology_lengths(ctx, Mod);
}

U32 ArmIL3Q(const LocalModel& freeM, const LocalModel& iso3, bool full, Tally& T) {
    const U32 QMAX = iso3.q_max();
    const ColumnSpec* per = nullptr;
    for (U32 s = 0; s < 3; ++s)
        for (U32 r = 0; r < iso3.num_reps(s); ++r)
            if (iso3.column(s, r).periodic()) per = &iso3.column(s, r);
    ENGINE_CHECK(per != nullptr);
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const Fq F9(3, hecke::FpPoly(3, {1, 0, 1}));   // F_3[Y]/(Y^2 + 1), certified irreducible by the constructor
    const ModCtx<U32> F3 = integral_model::RingContext(3, 1);
    const ModCtx<U32> Z9 = integral_model::RingContext(3, 2);
    U32 f9cells = 0, z9cells = 0;
    const std::vector<U32> Ns = full ? std::vector<U32>{1, 2, 3, 4, 5, 7, 11, 13} : std::vector<U32>{1, 2, 5, 7};
    const std::vector<U32> ns = full ? std::vector<U32>{0, 1, 2, 4, 6} : std::vector<U32>{0, 1, 2, 4};
    for (U32 N : Ns)
        for (U32 n : ns) {
            auto A = std::make_shared<const hecke::level::Gamma0LevelN>(N);
            const std::string at = " N=" + std::to_string(N) + " n=" + std::to_string(n);
            // ---- F_9: the F_q code path ----
            FqModule M9;
            M9.F = &F9;
            M9.A = A;
            M9.scal = hecke::neb::CharacterScalar(hecke::neb::ReduceCharacter(hecke::neb::DirichletCharacter::Trivial(N), F9));
            M9.n = n;
            M9.D = A->m() * (n + 1);
            const Mods M3 = InducedModule(A, n, F3);
            {
                bool same = true;
                for (const Mat22& g : {Mat22::S(), Mat22::T(), Mat22::ST(), Mat22::I().neg(), Mat22::ST() * Mat22::T() * Mat22::S()}) {
                    const FqMat& R9 = M9.rho(g);
                    const DenseBuf<U32> R3 = M3.rho(g);
                    for (U32 i = 0; i < M9.D; ++i)
                        for (U32 j = 0; j < M9.D; ++j) same = same && R9.at(i, j) == R3.at(i, j);
                }
                T.check(same, "(IL3Q) F_9: the twisted-monomial Ind (trivial chi) = the F_3 MonomialOf Ind, entry for entry" + at);
            }
            const auto h9 = FqDims(iso3, M9);
            const auto l3 = LengthsOf(iso3, F3, M3);
            const auto cf9 = FqClosedForm(M9, *per);
            T.check(h9 == l3, "(IL3Q) F_9: dim H^q(Gamma; Ind) over F_9 = over F_3 (flat base change)" + at + ": " + Str(h9) +
                                  " vs " + Str(l3));
            if (M9.D <= (full ? 64u : 40u))
                T.check(FqDims(freeM, M9) == h9, "(IL3Q) F_9: Iso^(3) = Iso" + at);
            if (n % 2 == 1) {
                bool zero = cf9.first == 0 && cf9.second == 0;
                for (U32 x : h9) zero = zero && x == 0;
                T.check(zero, "(IL3Q) F_9: -I |-> -1, H^. = 0" + at);
                ++f9cells;
                continue;
            }
            // The level-N pipeline over F_3 (the shipped route does not run over F_9; IL3's first summand by flatness).
            LocalSystemSimplicialComplex L3(K, hecke::level::InducedRelationsLevel(K, *A, n, F3), M9.D, 3);
            EquivariantCohomologyOnK coh3(&L3);
            coh3.compute_cohomology();
            std::vector<U32> pd = coh3.get_H_eq_dimensions();
            pd.resize(3, 0);
            T.check(h9[1] == pd[1] + cf9.second && h9[0] == pd[0],
                    "(IL3Q) F_9: dim H^1 = dim H^1_eq(level N, pipeline) + dim H^1(Z/3; M^{-I}) over F_9" + at);
            for (U32 q = 2; q <= QMAX; ++q)
                T.check(h9[q] == ((q % 2 == 0) ? cf9.first : cf9.second), "(IL3Q) F_9: H^q = H^q(Z/3; M^{-I})" + at +
                                                                                " q=" + std::to_string(q));
            ++f9cells;
            // ---- Z/9: lengths, against the shipped level-N pipeline's flat-coordinate route ----
            if (n > (full ? 6u : 4u)) continue;
            const Mods M9z = InducedModule(A, n, Z9);
            const auto lz = LengthsOf(iso3, Z9, M9z);
            const auto cfz = ClosedFormRing(Z9, M9z, *per);
            std::vector<U32> plen;
            try {
                const hecke::ScopedCoefficientRing ring(2);
                LocalSystemSimplicialComplex L9(K, hecke::level::InducedRelationsLevel(K, *A, n, Z9), M9.D, 9);
                EquivariantCohomologyOnK coh9(&L9);
                coh9.compute_cohomology();
                T.check(coh9.is_flat_route(), "(IL3Q) Z/9: the pipeline takes the flat-coordinate (Bredon) route" + at);
                plen = coh9.H_lengths();
            } catch (const std::exception& ex) {
                T.check(false, std::string("(IL3Q) Z/9: the shipped level-N pipeline over Z/9 threw") + at + ": " + ex.what());
                continue;
            }
            plen.resize(3, 0);
            T.check(lz[0] == plen[0], "(IL3Q) Z/9: length H^0 = the pipeline's" + at);
            T.check(lz[1] == plen[1] + cfz.second,
                    "(IL3Q) Z/9: length H^1(Gamma; Ind) = length H^1_eq(level N, pipeline over Z/9) + length H^1(Z/3; M^{-I})" + at +
                        ": " + std::to_string(lz[1]) + " vs " + std::to_string(plen[1]) + " + " + std::to_string(cfz.second));
            T.check(plen[2] == 0, "(IL3Q) Z/9: the pipeline's H^2 = 0" + at);
            for (U32 q = 2; q <= QMAX; ++q)
                T.check(lz[q] == ((q % 2 == 0) ? cfz.first : cfz.second),
                        "(IL3Q) Z/9: length H^q = length H^q(Z/3; M^{-I})" + at + " q=" + std::to_string(q));
            if (N == 7 && n == 4)
                std::cout << "ISOMODEL_ELL_IL3Q N=7 n=4 D=" << M9.D << " F9_h=" << Str(h9) << " pipeline_F3_h1=" << pd[1]
                          << " Z9_len=" << Str(lz) << " pipeline_Z9_len_h1=" << plen[1] << " len_H1_Z3=" << cfz.second << "\n";
            ++z9cells;
        }
    std::cout << "ISOMODEL_ELL_IL3Q F9_cells=" << f9cells << " Z9_pipeline_cells=" << z9cells
              << " (Ind_{Gamma_0(N)} Sym^n over F_9 by the twisted-monomial route; Z/9 lengths against the shipped "
                 "level-N pipeline's flat route)\n";
    return f9cells + z9cells;
}

}  // namespace

IsoEllArmTally HeckeIsotropyModelEllStrictRefined(const IsotropyModel& desc, const LocalModel& freeM, const LocalModel& iso3,
                                                  const std::vector<std::unique_ptr<StrictLiftBundle>>& bundles) {
    IsoEllArmTally tally;
    Tally T{&tally};
    const bool full = FullGrid();
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    auto lap = [&](const char* arm) {
        const double s = std::chrono::duration<double>(clock::now() - t0).count();
        std::cout << "ISOMODEL_ELL_S18WP7_TIME arm=" << arm << " s=" << s << "\n";
        t0 = clock::now();
    };
    U32 cells = ArmIL5(desc, freeM, bundles, full, T);
    lap("IL5");
    cells += ArmIL3Q(freeM, iso3, full, T);
    lap("IL3Q");
    if (full) std::cout << "EFFECTIVE HeckeIsotropyModelEllValidation arm=S18WP7_FULL cells=" << cells << "\n";
    std::cout << "ISOMODEL_ELL_S18WP7 checks=" << tally.checks << " failed=" << tally.failures
              << " grid=" << (full ? "FULL" : "default") << "\n";
    return tally;
}
