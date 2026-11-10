// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeReductionConeValidation.h"

// hecke-extensions slot 9, WP-4 (E8 item 4b): the boundary and cone arms over
// Z/ell^k -- see the header for the arm list.  The file name ends in
// `Validation` so the include guards and the raw-mod census treat it as the
// test harness it is.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryDiagnostics.h"
#include "HeckeBoundaryOperator.h"
#include "HeckeConeAction.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckePairHomotopy.h"
#include "HeckeReduction.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZBoundaryPair.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSL2Z.h"
#include "../HeckeExt/HeckeBoundary.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/ModCtx.h"

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using hecke::ModuleConeActionT;

uint64_t* g_checks = nullptr;
uint32_t* g_fail = nullptr;
uint32_t g_printed = 0;

void check(bool ok, const std::string& what) {
    ++*g_checks;
    if (!ok) {
        ++*g_fail;
        if (g_printed++ < 60) std::cout << "  [FAIL] " << what << std::endl;
    }
}

template<class F>
bool throws_niitb(F&& f, const std::string& must_contain) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild& e) {
        const std::string m = e.what();
        return m.find(hecke::kNotImplementedInThisBuildTag) == 0 && m.find(must_contain) != std::string::npos;
    } catch (...) { return false; }
    return false;
}
template<class F>
bool throws_logic(F&& f, const std::string& must_contain) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild&) { return false; }
    catch (const std::logic_error& e) { return std::string(e.what()).find(must_contain) != std::string::npos; }
    catch (...) { return false; }
    return false;
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "(";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ")";
    return o.str();
}
uint32_t count_of(const std::vector<uint32_t>& v, uint32_t e) {
    uint32_t c = 0;
    for (uint32_t x : v) c += (x == e) ? 1u : 0u;
    return c;
}

template<class Elt>
DenseBuf<Elt> dense_of(const FlatMatrixT<Elt>& M) {
    DenseBuf<Elt> b(M.get_rows(), M.get_cols());
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) b.at(i, j) = M.get_entry(i, j);
    return b;
}
template<class Elt>
DenseBuf<Elt> dense_of(const MatrixT<Elt>& M) {
    DenseBuf<Elt> b(M.rows, M.cols);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) b.at(i, j) = M.data[i][j];
    return b;
}
template<class Elt>
DenseBuf<Elt> basis_columns(const std::vector<FlatMatrixT<Elt>>& B, uint32_t dim) {
    DenseBuf<Elt> M(dim, static_cast<uint32_t>(B.size()));
    for (uint32_t t = 0; t < B.size(); ++t) {
        const uint32_t r = B[t].get_rows(), c = B[t].get_cols();
        for (uint32_t s = 0; s < r; ++s)
            for (uint32_t v = 0; v < c; ++v) M.at(s * c + v, t) = B[t].get_entry(s, v);
    }
    return M;
}
template<class Elt>
bool dense_equal(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}

// M-W4-1 (WP-14 review): obl:coneexact(vi) over Z/ell^k in (c5) form.  A and B
// are (c5) endomorphism matrices of M = (+) Z/ell^{e_i} (row i defined in
// [0, ell^{e_i})); the products are formed over Z/ell^k and compared row by
// row modulo ell^{e_i} of the target -- well defined because a (c5) matrix of a
// homomorphism has A(i, m) divisible by ell^{e_i - e_m} whenever e_m < e_i.
template<class Elt>
bool c5_commute(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B, const std::vector<uint32_t>& e) {
    const uint32_t t = static_cast<uint32_t>(e.size());
    if (A.rows != t || A.cols != t || B.rows != t || B.cols != t) return false;
    for (uint32_t i = 0; i < t; ++i) {
        const Elt m = (e[i] >= ctx.k) ? Elt(0) : ctx.ell_pow(e[i]);
        for (uint32_t j = 0; j < t; ++j) {
            Elt ab = 0, ba = 0;
            for (uint32_t s = 0; s < t; ++s) {
                ab = ctx.add(ab, ctx.mul(A.at(i, s), B.at(s, j)));
                ba = ctx.add(ba, ctx.mul(B.at(i, s), A.at(s, j)));
            }
            if (m) { ab = static_cast<Elt>(ab % m); ba = static_cast<Elt>(ba % m); }
            if (ab != ba) return false;
        }
    }
    return true;
}

// (b_j mod ell) - sum_i R(i, j) h_i lies in colspan Bf over F_ell (the class map).
template<class Elt>
bool class_map(Elt ell, const DenseBuf<Elt>& Hb, const DenseBuf<Elt>& Hf, const DenseBuf<Elt>& Bf, const DenseBuf<Elt>& R) {
    const ModCtx<Elt> F(ell);
    const uint32_t dim = Hb.rows;
    if (Hf.rows != dim || Bf.rows != dim || Hf.cols != R.rows || Hb.cols != R.cols) return Hb.cols == 0 && R.cols == 0;
    DenseBuf<Elt> D(dim, Hb.cols);
    for (uint32_t s = 0; s < dim; ++s)
        for (uint32_t j = 0; j < Hb.cols; ++j) {
            Elt acc = static_cast<Elt>(Hb.at(s, j) % ell);
            for (uint32_t i = 0; i < Hf.cols; ++i) acc = F.sub(acc, F.mul(Hf.at(s, i), R.at(i, j)));
            D.at(s, j) = acc;
        }
    if (D.cols == 0) return true;
    if (Bf.cols == 0) {
        for (uint32_t s = 0; s < dim; ++s)
            for (uint32_t j = 0; j < D.cols; ++j)
                if (D.at(s, j) != 0) return false;
        return true;
    }
    try {
        (void)engine::elim::solve_multi_rhs(F, engine::cview(Bf), engine::cview(D));
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// red_hi_to_F == red_lo_to_F o red_hi_to_lo (mod ell).
template<class Elt>
bool functorial(Elt ell, const DenseBuf<Elt>& R_hi, const DenseBuf<Elt>& R_lo, const DenseBuf<Elt>& R_hl) {
    if (R_lo.cols != R_hl.rows || R_hi.rows != R_lo.rows || R_hi.cols != R_hl.cols) return false;
    const ModCtx<Elt> F(ell);
    for (uint32_t i = 0; i < R_hi.rows; ++i)
        for (uint32_t j = 0; j < R_hi.cols; ++j) {
            Elt acc = 0;
            for (uint32_t m = 0; m < R_lo.cols; ++m)
                acc = F.add(acc, F.mul(static_cast<Elt>(R_lo.at(i, m) % ell), static_cast<Elt>(R_hl.at(m, j) % ell)));
            if (acc != static_cast<Elt>(R_hi.at(i, j) % ell)) return false;
        }
    return true;
}

// ---- one cell ------------------------------------------------------------------
struct ConeSpec {
    uint32_t n;
    uint64_t ell;
    std::vector<uint32_t> levels;            // must contain 2
    std::vector<uint64_t> primes;
    bool class_route = false;                // (4b-K), uint32_t only
    bool tamper = false;                     // the item-14 negative control
    // [E] (f)'s predicted boundary lists per level (empty: not predicted).
    std::vector<std::vector<uint32_t>> bdry_pred;
    std::vector<uint32_t> L0, L1;            // the landed lengths_H^{0,1} per level ((e)); empty: not checked
};

template<class Elt>
struct FieldTables {
    std::unique_ptr<hecke::HeckeCosetTablesT<Elt>> cosets;
    std::unique_ptr<hecke::HeckeTauTableT<Elt>> tau;
    std::unique_ptr<hecke::HeckeBoundaryTauTableT<Elt>> taub;
    std::unique_ptr<hecke::HeckePairHomotopyT<Elt>> eta;
    std::unique_ptr<hecke::HeckeOperatorOnKT<Elt>> bulk_f;
    std::unique_ptr<hecke::HeckeBoundaryOperatorT<Elt>> bd_f;
    std::unique_ptr<hecke::HeckeConeAction> act_f;   // uint32_t only
};

template<class Elt>
struct RingLevel {
    uint32_t k = 0;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc, lssc_b;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh, coh_b;
    std::unique_ptr<ModuleConeT<Elt>> cone;
    std::vector<std::unique_ptr<hecke::HeckeOperatorOnKT<Elt>>> bulk_r;
    std::vector<std::unique_ptr<hecke::HeckeBoundaryOperatorT<Elt>>> bd_r;
    std::vector<std::unique_ptr<ModuleConeActionT<Elt>>> act_r;
    std::vector<DenseBuf<Elt>> red_bd, red_c, red_bulk;   // to F_ell, per degree
};

struct CellResult {
    bool ran = false;
    std::string excluded;
};

template<class Elt>
CellResult cone_cell(const ConeSpec& cs, bool verbose) {
    CellResult out;
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t n = cs.n;
    const Elt ell = static_cast<Elt>(cs.ell);
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    const SimplicialComplex K_bd = CreateSL2ZBoundaryPathTriangulation();
    const std::string cell = "(" + std::to_string(n) + ", " + std::to_string(cs.ell) + ")";
    constexpr bool u32 = std::is_same_v<Elt, uint32_t>;

    // ---- the F_ell side (defaults) ----
    LocalSystemSimplicialComplexT<Elt> lssc_f(K, GenerateSL2ZRelationsElt<Elt>(K, n, ell), n + 1, ell);
    EquivariantCohomologyOnKT<Elt> coh_f(&lssc_f);
    coh_f.compute_cohomology();
    LocalSystemSimplicialComplexT<Elt> lssc_bf(K_bd, GenerateSL2ZBoundaryRelationsElt<Elt>(K_bd, n, ell), n + 1, ell);
    EquivariantCohomologyOnKT<Elt> coh_bf(&lssc_bf);
    coh_bf.compute_cohomology();
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair_f;
    std::unique_ptr<EquivariantMappingCone> cone_f;
    const EquivariantCohomologyOnKT<Elt>* bd_field = &coh_bf;
    if constexpr (u32) {
        pair_f = std::make_unique<hecke::SL2ZBoundaryPair>(&coh_f, K);
        cone_f = pair_f->make_cone();
        bd_field = &pair_f->boundary_cohomology();
    }
    std::vector<FieldTables<Elt>> ft(cs.primes.size());
    for (size_t i = 0; i < cs.primes.size(); ++i) {
        FieldTables<Elt>& t = ft[i];
        t.cosets = std::make_unique<hecke::HeckeCosetTablesT<Elt>>(cs.primes[i], 1, ell, n);
        t.tau = std::make_unique<hecke::HeckeTauTableT<Elt>>(K, model, *t.cosets, ell, n);
        t.taub = std::make_unique<hecke::HeckeBoundaryTauTableT<Elt>>(K, model, *t.cosets);
        t.eta = std::make_unique<hecke::HeckePairHomotopyT<Elt>>(*t.tau, *t.taub);
        if (t.tau->used_mod_ell_solve()) { out.excluded = "tau used the mod-ell fallback at p = " + std::to_string(cs.primes[i]); return out; }
        t.bulk_f = std::make_unique<hecke::HeckeOperatorOnKT<Elt>>(&coh_f, t.cosets.get(), t.tau.get());
        t.bulk_f->compute();
        t.bd_f = std::make_unique<hecke::HeckeBoundaryOperatorT<Elt>>(bd_field, t.cosets.get(), t.taub.get());
        t.bd_f->compute();
        if constexpr (u32) {
            t.act_f = std::make_unique<hecke::HeckeConeAction>(cone_f.get(), t.bulk_f.get(), t.bd_f.get(), t.eta.get());
            t.act_f->compute();
        }
    }

    // ---- the ring levels ----
    std::vector<std::unique_ptr<RingLevel<Elt>>> lv;
    for (uint32_t k : cs.levels) {
        hecke::ScopedCoefficientRing ring(k);
        auto L = std::make_unique<RingLevel<Elt>>();
        L->k = k;
        const ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeReductionValidation (E8 item 4b)", cs.ell, k);
        L->lssc = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(K, hecke::GenerateSL2ZRelationsRing<Elt>(K, n, ctx), n + 1, ctx.p);
        L->coh = std::make_unique<EquivariantCohomologyOnKT<Elt>>(L->lssc.get());
        L->coh->compute_cohomology();
        L->lssc_b = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(
            K_bd, hecke::GenerateSL2ZBoundaryRelationsRing<Elt>(K_bd, n, ctx), n + 1, ctx.p);
        L->coh_b = std::make_unique<EquivariantCohomologyOnKT<Elt>>(L->lssc_b.get());
        L->coh_b->compute_cohomology();
        std::vector<DenseBuf<Elt>> rf;
        for (uint32_t q = 0; q <= 1; ++q) rf.push_back(RestrictionFlatRing<Elt>(K, K_bd, q, n + 1, ctx));
        L->cone = std::make_unique<ModuleConeT<Elt>>(L->coh.get(), L->coh_b.get(), std::move(rf));
        L->cone->compute();
        for (size_t i = 0; i < cs.primes.size(); ++i) {
            L->bulk_r.push_back(std::make_unique<hecke::HeckeOperatorOnKT<Elt>>(L->coh.get(), ft[i].cosets.get(), ft[i].tau.get()));
            L->bulk_r.back()->compute();
            L->bd_r.push_back(std::make_unique<hecke::HeckeBoundaryOperatorT<Elt>>(L->coh_b.get(), ft[i].cosets.get(), ft[i].taub.get()));
            L->bd_r.back()->compute();
            L->act_r.push_back(std::make_unique<ModuleConeActionT<Elt>>(L->cone.get(), L->bulk_r.back().get(), L->bd_r.back().get(),
                                                                        ft[i].eta.get()));
            L->act_r.back()->compute();
        }
        lv.push_back(std::move(L));
    }
    const RingLevel<Elt>* L2 = nullptr;
    for (const auto& L : lv) if (L->k == 2) L2 = L.get();
    ENGINE_CHECK_MSG(L2 != nullptr, "cone_cell: the level list must contain 2");

    std::ostringstream line;
    line << "  (4b) " << cell;
    for (size_t li = 0; li < lv.size(); ++li) {
        RingLevel<Elt>& L = *lv[li];
        const std::string at = cell + " Z/ell^" + std::to_string(L.k);
        const ModuleConeT<Elt>& C = *L.cone;
        // ---- (4b-B) the boundary ----
        const std::vector<uint32_t>& b0 = L.coh_b->H_invariant_factors(0);
        const std::vector<uint32_t>& b1 = L.coh_b->H_invariant_factors(1);
        // (slot-9 WP-4t) the row "bdry_lengths_H^0 == bdry_lengths_H^1" is dropped: it is
        // implied by the next row, since H_lengths()[q] is by definition the sum of
        // H_invariant_factors(q) (EquivariantCohomologyOnK.cpp, length_H).
        check(b0 == b1, "(4b-B) bdry_invariant_factors_H^0 == bdry_invariant_factors_H^1 at " + at);
        if (li < cs.bdry_pred.size() && !cs.bdry_pred[li].empty())
            check(b0 == cs.bdry_pred[li], "(4b-B) bdry_invariant_factors_H^q = " + vec_str(b0) + " == predicted " +
                                              vec_str(cs.bdry_pred[li]) + " at " + at);
        // ---- (4b-C) the cone ----
        const std::vector<uint32_t> cl = C.H_lengths();
        const std::vector<uint32_t> im = C.les_im_lengths();
        // (slot-9 WP-4t) the flag read coneexact_iv_certified() is dropped: ModuleConeT::compute()
        // throws unless the certificate verifies and sets the flag only after that, so it cannot
        // be false here.  The one explicit re-derivation per level stays: verify() == 0.
        check(C.certificate().verify() == 0, "(4b-C) ConeModuleCertificate::verify() == 0 at " + at);
        check(cl.size() == 3 && cl[0] == 0, "(4b-C) cone_lengths_Hc^0 == 0 at " + at);
        check(im.size() == 8, "(4b-C) eight image lengths at " + at);
        const std::vector<uint32_t> Lb = L.coh->H_lengths();
        if (cl.size() == 3 && im.size() == 8 && Lb.size() == 3) {
            const uint32_t L0 = Lb[0], L1 = Lb[1];
            check(Lb[2] == 0, "(4b-C) H^2 = 0 at " + at);
            // (slot-9 WP-4t) the rows "im == (0, L0, B-L0, L1-B+c2, B-c2, c2, 0, 0)" and
            // "l(H^1_c) == L1 - L0 + c2" (B = l(Hb^0), c2 = l(H^2_c)) are dropped: verify() == 0
            // gives C11 (each term's length = the sum of its two adjacent image lengths, the
            // stored image lengths re-derived; lengths are sums of invariant factors), and with
            // H^0_c = 0, H^2 = 0 (rows above) and l(Hb^0) = l(Hb^1) (the IF row) the eight image
            // lengths and l(H^1_c) are forced by additivity along the sequence.
            if (li < cs.L0.size()) check(L0 == cs.L0[li] && L1 == cs.L1[li], "(4b-C) the landed lengths_H^{0,1} at " + at);
        }
        if (cs.n == 10 && cs.ell == 691 && L.k == 2) {
            check(C.H_invariant_factors(1) == std::vector<uint32_t>({2, 2, 2}) && C.H_invariant_factors(2).empty() &&
                      im == std::vector<uint32_t>({0, 0, 2, 4, 2, 0, 0, 0}),
                  "(4b-C) [E] (f)'s complete row at (10, 691): H^1_c = (2,2,2), H^2_c = (), im (0,0,2,4,2,0,0,0)");
        }
        if (cs.n == 16 && cs.ell == 5 && L.k == 2) {
            const std::vector<uint32_t>& h1 = C.H_invariant_factors(1);
            const std::vector<uint32_t>& h2 = C.H_invariant_factors(2);
            // m-W4-4 (WP-14 review): m pinned to 1 now that R1 has landed -- [E]
            // (f) allowed m in {1, 2}; H^3_c = 0 and dim H^2_c(F_5) = 1 force
            // |H^2_c(Z)_(5)| = 5, i.e. m = 1 (the golden line guards it too).
            const bool ok = h2.size() == 1 && h2[0] == 1 &&
                            h1 == std::vector<uint32_t>({2, 2, 2, h2[0]}) &&
                            im == std::vector<uint32_t>({0, 0, 5, 1 + h2[0], 5 - h2[0], h2[0], 0, 0});
            check(ok, "(4b-C) [E] (f)'s row at (16, 5), m = 1: H^1_c = (2,2,2,1), H^2_c = (1), im (0,0,5,2,4,1,0,0); got H^1_c " +
                          vec_str(h1) + " H^2_c " + vec_str(h2));
            if (ok) line << " [m = " << h2[0] << "]";
        }
        if (cs.tamper && L.k == 2) {
            // m-W4-3 (WP-14 review): tamper the kernel generator x itself, not w1.
            // x[0] + 1 differs from x[0] in Z/ell^{e_0} (e_0 >= 1), and verify()
            // compares in(w1) + D_M w2 with x exactly, so the residual cannot
            // vanish -- unlike w1[0] + 1, which is invisible when column 0 of
            // the incoming map is zero in the target (a false alarm).
            ConeModuleCertificate<Elt> bad = C.certificate();
            bool tampered = false;
            for (auto& w : bad.witnesses)
                if (!w.x.empty()) {
                    w.x[0] = bad.ctx.add(w.x[0], 1);
                    tampered = true;
                    break;
                }
            if (tampered) {
                const std::vector<uint32_t> items = bad.failing_items();
                check(std::find(items.begin(), items.end(), 14u) != items.end(),
                      "(4b-C) negative control: a tampered item-14 witness fails item 14 at " + at);
            }
            ConeModuleCertificate<Elt> bad2 = C.certificate();
            if (!bad2.im_lengths.empty()) {
                bad2.im_lengths[2] += 1;
                const std::vector<uint32_t> items = bad2.failing_items();
                check(std::find(items.begin(), items.end(), 11u) != items.end(),
                      "(4b-C) negative control: a wrong stored image length fails item 11 at " + at);
            }
        }
        line << " K=" << L.k << ": bdry " << vec_str(b0) << " Hc " << vec_str(C.H_invariant_factors(1)) << "/"
             << vec_str(C.H_invariant_factors(2)) << " im " << vec_str(im) << " witnesses " << C.certificate().witnesses.size();
        // ---- (4b-R) the reductions to F_ell ----
        for (uint32_t q = 0; q <= 1; ++q) {
            L.red_bd.push_back(hecke::ReduceHToField(*L.coh_b, *bd_field, q));
            const DenseBuf<Elt> Hf = basis_columns(bd_field->get_H_eq_bases_fm()[q], L.coh_b->H_module_basis(q).rows);
            const DenseBuf<Elt> Bf = basis_columns(bd_field->get_B_eq_bases_fm()[q], L.coh_b->H_module_basis(q).rows);
            check(class_map<Elt>(ell, L.coh_b->H_module_basis(q), Hf, Bf, L.red_bd[q]), "(4b-R) bdry_red_H^" + std::to_string(q) +
                                                                                              " is the class map at " + at);
        }
        for (uint32_t q = 0; q <= 2; ++q) L.red_bulk.push_back(hecke::ReduceHToField(*L.coh, coh_f, q));
        if constexpr (u32) {
            for (uint32_t q = 0; q <= 2; ++q) {
                L.red_c.push_back(hecke::ReduceConeHcToField(C, *cone_f, q));
                const DenseBuf<uint32_t> Hf = cone_f->H_block(q).to_flat(engine::Layout::ColMajor);
                const DenseBuf<uint32_t> Bf = cone_f->B_block(q).to_flat(engine::Layout::ColMajor);
                check(class_map<uint32_t>(ell, C.H_module_basis(q), Hf, Bf, L.red_c[q]),
                      "(4b-R) cone_red_Hc^" + std::to_string(q) + " is the class map at " + at);
            }
            // every map of eq:coneles commutes with red
            for (uint32_t q = 0; q <= 2; ++q) {
                if (cone_f->pi_built(q) && C.pi_built(q))
                    check(hecke::ReductionIntertwines<uint32_t>(ell, dense_of(cone_f->pi(q)), L.red_c[q], C.pi(q), L.red_bulk[q]),
                          "(4b-R) red o pi = pi o red at q = " + std::to_string(q) + ", " + at);
                if (q <= 1 && cone_f->r_built(q))
                    check(hecke::ReductionIntertwines<uint32_t>(ell, dense_of(cone_f->r(q)), L.red_bulk[q], C.r(q), L.red_bd[q]),
                          "(4b-R) red o r = r o red at q = " + std::to_string(q) + ", " + at);
                if (q <= 1 && cone_f->iota_built(q) && C.iota_built(q))
                    check(hecke::ReductionIntertwines<uint32_t>(ell, dense_of(cone_f->iota(q)), L.red_bd[q], C.iota(q), L.red_c[q + 1]),
                          "(4b-R) red o iota = iota o red at q = " + std::to_string(q) + ", " + at);
            }
        }
        // ---- (4b-H) + (4b-R): the Hecke map equalities ----
        for (size_t i = 0; i < cs.primes.size(); ++i) {
            const std::string pp = " p=" + std::to_string(cs.primes[i]) + " " + at;
            for (uint32_t q = 0; q <= 1; ++q)
                check(hecke::ReductionToFieldEquivariant<Elt>(ell, ft[i].bd_f->A(q), L.bd_r[i]->H_matrix_module(q), L.red_bd[q]),
                      "(4b-R) A^bd o red = red o A^bd, q = " + std::to_string(q) + pp);
            // (slot-9 WP-4t) the flag read record().routes_bit_identical is dropped:
            // ModuleConeActionT::compute() throws when the two bracketings of h differ.
            if constexpr (u32) {
                for (uint32_t q = 0; q <= 2; ++q)
                    check(hecke::ReductionToFieldEquivariant<uint32_t>(ell, ft[i].act_f->A_c(q), L.act_r[i]->A_c(q), L.red_c[q]),
                          "(4b-R) A_c o red = red o A_c, q = " + std::to_string(q) + pp);
            }
        }
        // M-W4-1 (WP-14 review): obl:coneexact(vi)'s commutativity over Z/ell^k
        // (6 in R^x at every 4b cell), for every pair of computed primes, on
        // A_c^{(q)} (q <= 2) and A^bd^{(q)} (q <= 1), in (c5) form.  The nu = 2
        // recursion over the ring is not computed (recorded, WP-4 report).
        for (size_t i = 0; i < cs.primes.size(); ++i)
            for (size_t j = i + 1; j < cs.primes.size(); ++j) {
                const std::string pq = " p=" + std::to_string(cs.primes[i]) + ", p'=" + std::to_string(cs.primes[j]) + " " + at;
                for (uint32_t q = 0; q <= 2; ++q)
                    check(c5_commute<Elt>(C.ring_context(), L.act_r[i]->A_c(q), L.act_r[j]->A_c(q), C.H_invariant_factors(q)),
                          "(4b-H) [A_c(T_p), A_c(T_p')] = 0 in (c5) form, q = " + std::to_string(q) + pq);
                for (uint32_t q = 0; q <= 1; ++q)
                    check(c5_commute<Elt>(L.coh_b->ring_context(), L.bd_r[i]->H_matrix_module(q), L.bd_r[j]->H_matrix_module(q),
                                          L.coh_b->H_invariant_factors(q)),
                          "(4b-H) [A^bd(T_p), A^bd(T_p')] = 0 in (c5) form, q = " + std::to_string(q) + pq);
            }
    }
    // ---- the Bocksteins (a mod-ell object: the cell's Z/ell^2 level) ----
    {
        const DenseBuf<Elt> beta = hecke::BocksteinH(*L2->coh_b, *bd_field, 0);
        const uint32_t rk = hecke::RankOverField<Elt>(ell, beta);
        const std::vector<uint32_t>& b1 = L2->coh_b->H_invariant_factors(1);
        check(rk == count_of(b1, 1), "(4b-B) bdry_bockstein_rank^0 = " + std::to_string(rk) + " == #{1 in bdry_invariant_factors_H^1} at " + cell);
        const uint32_t h0 = bd_field->get_H_eq_dimensions()[0], h1 = bd_field->get_H_eq_dimensions()[1];
        const std::vector<uint32_t> bl = L2->coh_b->H_lengths();
        check(bl[0] == 2 * h0 - rk && bl[1] == 2 * h1 - rk, "(4b-B) the coefficient LES bookkeeping of the boundary at Z/ell^2, " + cell);
        for (size_t i = 0; i < cs.primes.size(); ++i)
            check(hecke::BocksteinEquivariant<Elt>(ell, ft[i].bd_f->A(1), ft[i].bd_f->A(0), beta),
                  "(4b-R) beta_bd o A^bd = A^bd o beta_bd at p = " + std::to_string(cs.primes[i]) + ", " + cell);
        line << " | beta_bd " << rk;
        if constexpr (u32) {
            const DenseBuf<uint32_t> bc = hecke::BocksteinConeHc(*L2->cone, *cone_f, 1);
            const uint32_t rc = hecke::RankOverField<uint32_t>(ell, bc);
            const std::vector<uint32_t> cl2 = L2->cone->H_lengths();
            check(cl2[1] == 2 * cone_f->dim_Hc(1) - rc && cl2[2] == 2 * cone_f->dim_Hc(2) - rc,
                  "(4b-B) the coefficient LES bookkeeping of the cone at Z/ell^2 (beta^1_c rank " + std::to_string(rc) + "), " + cell);
            for (size_t i = 0; i < cs.primes.size(); ++i)
                check(hecke::BocksteinEquivariant<uint32_t>(ell, ft[i].act_f->A_c(2), ft[i].act_f->A_c(1), bc),
                      "(4b-R) beta^1_c o A_c = A_c o beta^1_c at p = " + std::to_string(cs.primes[i]) + ", " + cell);
            line << " beta^1_c " << rc;
        }
    }
    // ---- (4b-R) between levels ----
    for (size_t a = 0; a < lv.size(); ++a)
        for (size_t b = 0; b < lv.size(); ++b) {
            if (lv[a]->k != lv[b]->k + 1) continue;
            const RingLevel<Elt>& H = *lv[a];
            const RingLevel<Elt>& Lo = *lv[b];
            const std::string at = cell + " K=" + std::to_string(H.k) + "->" + std::to_string(Lo.k);
            for (uint32_t q = 0; q <= 1; ++q) {
                const DenseBuf<Elt> R = hecke::ReduceHBetweenLevels(*H.coh_b, *Lo.coh_b, q);
                check(functorial<Elt>(ell, H.red_bd[q], Lo.red_bd[q], R), "(4b-R) bdry functoriality, q = " + std::to_string(q) + ", " + at);
                for (size_t i = 0; i < cs.primes.size(); ++i)
                    check(hecke::ReductionBetweenLevelsEquivariant<Elt>(H.coh_b->ring_context(), Lo.coh_b->H_invariant_factors(q),
                                                                       Lo.bd_r[i]->H_matrix_module(q), H.bd_r[i]->H_matrix_module(q), R),
                          "(4b-R) A^bd o red = red o A^bd between levels, q = " + std::to_string(q) + ", " + at);
            }
            for (uint32_t q = 0; q <= 2; ++q) {
                const DenseBuf<Elt> R = hecke::ReduceConeHcBetweenLevels(*H.cone, *Lo.cone, q);
                if constexpr (u32)
                    check(functorial<Elt>(ell, H.red_c[q], Lo.red_c[q], R), "(4b-R) cone functoriality, q = " + std::to_string(q) + ", " + at);
                for (size_t i = 0; i < cs.primes.size(); ++i)
                    check(hecke::ReductionBetweenLevelsEquivariant<Elt>(H.cone->ring_context(), Lo.cone->H_invariant_factors(q),
                                                                       Lo.act_r[i]->A_c(q), H.act_r[i]->A_c(q), R),
                          "(4b-R) A_c o red = red o A_c between levels, q = " + std::to_string(q) + ", " + at);
            }
        }
    // ---- (4b-K) the class route agrees with the template route ----
    if constexpr (u32) {
        if (cs.class_route) {
            for (const auto& Lp : lv) {
                const RingLevel<Elt>& L = *Lp;
                const std::string at = cell + " Z/ell^" + std::to_string(L.k);
                hecke::ScopedCoefficientRing ring(L.k);
                hecke::SL2ZBoundaryPair pair(L.coh.get(), K);
                const std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
                const ModuleConeT<uint32_t>& M = cone->module_cone();
                bool same = M.H_lengths() == L.cone->H_lengths() && M.les_im_lengths() == L.cone->les_im_lengths();
                for (uint32_t q = 0; q <= 2; ++q) same = same && M.H_invariant_factors(q) == L.cone->H_invariant_factors(q);
                for (uint32_t q = 0; q <= 1; ++q) same = same && dense_equal(M.r_flat(q), L.cone->r_flat(q));
                check(same && pair.is_composite() && cone->is_composite() && M.coneexact_iv_certified(),
                      "(4b-K) SL2ZBoundaryPair -> EquivariantMappingCone == the template route (lengths, lists, image lengths, r_flat) at " + at);
                const auto& ind = pair.restriction().get_induced_H_module_maps();
                check(ind.size() >= 2 && dense_equal(ind[0], L.cone->r(0)) && dense_equal(ind[1], L.cone->r(1)),
                      "(4b-K) the composite restriction's induced maps == the cone's r^q at " + at);
                for (size_t i = 0; i < cs.primes.size(); ++i) {
                    hecke::HeckeBoundaryOperatorT<uint32_t> bd(&pair.boundary_cohomology(), ft[i].cosets.get(), ft[i].taub.get());
                    bd.compute();
                    hecke::HeckeConeAction act(cone.get(), L.bulk_r[i].get(), &bd, ft[i].eta.get());
                    act.compute();
                    bool eq = act.is_composite();
                    for (uint32_t q = 0; q <= 2; ++q) eq = eq && dense_equal(act.module_action().A_c(q), L.act_r[i]->A_c(q));
                    check(eq, "(4b-K) HeckeConeAction (route 1) == ModuleConeActionT, A_c at p = " + std::to_string(cs.primes[i]) + ", " + at);
                }
            }
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (verbose) std::cout << line.str() << "  " << secs << " s" << std::endl;
    out.ran = true;
    return out;
}

// ---- (X4b) the closure rows moved by 4b ---------------------------------------------
void closure_arm() {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    const SimplicialComplex K_bd = CreateSL2ZBoundaryPathTriangulation();
    const uint32_t n = 10;
    // (10, 691) over Z/ell^2 -- the honouring objects.
    hecke::HeckeCosetTables cosets(2, 1, 691u, n);
    hecke::HeckeTauTable tau(K, model, cosets, 691u, n);
    hecke::HeckeBoundaryTauTable taub(K, model, cosets);
    hecke::HeckePairHomotopy eta(tau, taub);
    LocalSystemSimplicialComplex lssc_f(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, 691u), n + 1, 691u);
    EquivariantCohomologyOnK coh_f(&lssc_f);
    coh_f.compute_cohomology();
    hecke::SL2ZBoundaryPair pair_f(&coh_f, K);
    uint32_t rows = 0;
    auto row = [&](const std::string& name, bool ok) { ++rows; check(ok, "(X4b) " + name); };
    hecke::ScopedCoefficientRing ring(2);
    const ModCtx<uint32_t> ctx = hecke::CoefficientRingContext<uint32_t>("probe", 691u, 2);
    LocalSystemSimplicialComplex lssc(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    hecke::SL2ZBoundaryPair pair(&coh, K);
    row("SL2ZBoundaryPair HONOURS Z/ell^2 at (10, 691) (changed row, slot-9 WP-4)",
        pair.is_composite() && pair.boundary_cohomology().is_composite() && pair.restriction().is_composite());
    row("SL2ZBoundaryPair: the field accessor dim_H_bd throws a named logic_error over Z/ell^2",
        throws_logic([&] { (void)pair.dim_H_bd(0); }, "dim"));
    auto cone = pair.make_cone();
    row("EquivariantMappingCone HONOURS Z/ell^2 (changed row)", cone->is_composite() && cone->module_cone().coneexact_iv_certified());
    row("EquivariantMappingCone: a field accessor (dim_Hc) throws a named logic_error",
        throws_logic([&] { (void)cone->dim_Hc(1); }, "EquivariantMappingCone::dim_Hc: a field (F_ell) accessor"));
    row("EquivariantMappingCone: a mixed-ring input (F_ell boundary, Z/ell^2 bulk) refuses by name",
        throws_niitb([&] { EquivariantMappingCone x(&coh, &pair_f.boundary_cohomology(), &pair.restriction()); }, "EquivariantMappingCone: "));
    row("EquivariantCochainComplexMorphismOnK (constant fibre) HONOURS (changed row)",
        pair.restriction().are_maps_computed() && pair.restriction().get_induced_H_module_maps().size() == 2);
    row("EquivariantCochainComplexMorphismOnK: a domain / codomain ring mismatch is a named logic_error",
        throws_logic([&] {
            EquivariantCochainComplexMorphismOnK x(&pair_f.boundary_cohomology(), &coh, &pair.inclusion(), FlatMatrix(n + 1, ctx.p));
        }, "are over different rings"));
    row("EquivariantCochainComplexMorphismOnK: get_induced_H_eq_maps on a composite morphism is a named logic_error",
        throws_logic([&] { (void)pair.restriction().get_induced_H_eq_maps(); }, "a field (F_ell) accessor"));
    row("EquivariantCochainComplexMorphismOnK (FiberBlockTable ctor) keeps refusing Z/ell^2",
        throws_niitb([&] { EquivariantCochainComplexMorphismOnK x(&coh, &coh, FiberBlockTable{}, "probe"); },
                     "EquivariantCochainComplexMorphismOnK: "));
    hecke::HeckeBoundaryOperator bd(&pair.boundary_cohomology(), &cosets, &taub);
    bd.compute();
    row("HeckeBoundaryOperator HONOURS Z/ell^2 (changed row)", bd.is_composite() && bd.certificates().routes_bit_identical);
    row("HeckeBoundaryOperator: a field accessor (A) throws a named logic_error",
        throws_logic([&] { (void)bd.A(0); }, "HeckeBoundaryOperator::A: a field (F_ell) accessor"));
    hecke::HeckeOperatorOnK bulk(&coh, &cosets, &tau);
    bulk.compute();
    hecke::HeckeConeAction act(cone.get(), &bulk, &bd, &eta);
    act.compute();
    row("HeckeConeAction (route 1) HONOURS Z/ell^2 (changed row)", act.is_composite() && act.module_action().record().routes_bit_identical);
    row("HeckeConeAction: a field accessor (A_c) throws a named logic_error",
        throws_logic([&] { (void)act.A_c(1); }, "HeckeConeAction::A_c: a field (F_ell) accessor"));
    // CHANGED ROWS (slot-10 WP-3, decision 3 (A); were "keeps refusing" through slot 9):
    // obl:coneexact(vii) over the ring -- both cross-routes HONOUR a composite normative
    // action under its ring and agree with route 1.  The refusals that remain (F_ell
    // normative under the ring, the ring under F_ell, another k, the uint32_t regime above
    // 2^32) are HeckeConeCrossRouteRingValidation's arms (X) / (W).
    row("HeckeConeStrictPairRoute HONOURS a composite normative action and agrees (changed row, slot-10 WP-3)", [&] {
        try { hecke::HeckeConeStrictPairRoute x(&act); x.compute(); return x.record().agrees() && x.record().ring_k == 2; }
        catch (...) { return false; }
    }());
    row("HeckeConeRelativeRoute HONOURS a composite normative action and agrees (changed row, slot-10 WP-3)", [&] {
        try { hecke::HeckeConeRelativeRoute x(&act); x.compute(); return x.record().agrees() && x.is_composite(); }
        catch (...) { return false; }
    }());
    row("HeckeExt/HeckeBoundary keeps refusing (Z/ell^2 input, Z/ell^2 in force)",
        throws_niitb([&] { hecke_ext::HeckeBoundary x(&coh, K, model, hecke_ext::HeckeBoundaryOptions()); }, "HeckeBoundary: "));
    row("cochain_commutes (HeckeBoundaryDiagnostics): now REACHABLE, refuses by name (new arm)",
        throws_niitb([&] { (void)hecke::cochain_commutes(coh, bulk, bd, pair.restriction()); }, "cochain_commutes (HeckeBoundaryDiagnostics): "));
    {   // the cone and the boundary are NOT honoured at ell = 3 (the default SmallEllPolicy)
        const ModCtx<uint32_t> c9 = ModCtx<uint32_t>::prime_power(3u, 2);
        LocalSystemSimplicialComplex l9(K, hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, c9), n + 1, 9u);
        EquivariantCohomologyOnK coh9(&l9);
        coh9.compute_cohomology();
        row("SL2ZBoundaryPair refuses a Bredon bulk at (10, 3) by name", throws_niitb([&] { hecke::SL2ZBoundaryPair x(&coh9, K); }, "at ell = 3"));
        LocalSystemSimplicialComplex lb9(K_bd, hecke::GenerateSL2ZBoundaryRelationsRing<uint32_t>(K_bd, n, c9), n + 1, 9u);
        EquivariantCohomologyOnK cohb9(&lb9);
        cohb9.compute_cohomology();
        std::vector<DenseBuf<uint32_t>> rf;
        for (uint32_t q = 0; q <= 1; ++q) rf.push_back(RestrictionFlatRing<uint32_t>(K, K_bd, q, n + 1, c9));
        row("ModuleConeT (the cone's dense route) refuses at ell = 3 by name",
            throws_niitb([&] { ModuleConeT<uint32_t> x(&coh9, &cohb9, rf); }, "at ell = 3"));
        hecke::HeckeCosetTables cos3(2, 1, 3u, n);
        hecke::HeckeBoundaryTauTable tb3(K, model, cos3);
        row("HeckeBoundaryOperator refuses a Bredon boundary at (10, 3) by name",
            throws_niitb([&] { hecke::HeckeBoundaryOperator x(&cohb9, &cos3, &tb3); }, "at ell = 3"));
        row("EquivariantCochainComplexMorphismOnK refuses at ell = 3 by name",
            throws_niitb([&] {
                SimplicialComplexMorphism scm(&K_bd, &K);
                EquivariantCochainComplexMorphismOnK x(&cohb9, &coh9, &scm, FlatMatrix(n + 1, 9u));
            }, "at ell = 3"));
    }
    std::cout << "  (X4b) closure rows moved or added by 4b: " << rows << " checked" << std::endl;
}

// ---- the U64 regimes: the template route only (the F_ell cone class is uint32_t) ----
template<class Elt>
void regime_cell(uint32_t n, uint64_t ell) {
    ConeSpec cs{n, ell, {2}, {2}, false, false, {}, {}, {}};
    const CellResult r = cone_cell<Elt>(cs, true);
    check(r.ran, "(4b) regime cell (" + std::to_string(n) + ", " + std::to_string(ell) + ") ran: " + r.excluded);
}

} // namespace

void HeckeReductionConeArms(uint64_t& checks, uint32_t& failures) {
    g_checks = &checks;
    g_fail = &failures;
    g_printed = 0;
    const uint64_t c0 = checks;
    const uint32_t f0 = failures;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "  E8 item 4b: the boundary and the cone over Z/ell^k (dense module route; route 1; [E] sec:ring (f))" << std::endl;
    // The E8 points but (60, 5), with [E] (f)'s predicted boundary lists and (e)'s landed lengths.
    // (slot-9 WP-4t) (60, 5) runs in HeckeReductionConeCoverageArms (`slow`), unchanged in
    // content; (30, 5) keeps the full two-level, two-prime shape here.
    const std::vector<ConeSpec> cells = {
        {10, 691, {2}, {2, 3, 5}, true, true, {{2}}, {0}, {6}},
        {6, 5, {2}, {2, 3}, true, false, {{2, 1}}, {1}, {3}},
        {16, 5, {2}, {2, 3}, true, false, {{2, 1, 1, 1}}, {0}, {6}},
        {30, 5, {2, 3}, {2, 3}, true, false, {{2, 2, 1, 1, 1, 1, 1}, {3, 2, 1, 1, 1, 1, 1}}, {2, 2}, {12, 17}},
    };
    for (const ConeSpec& cs : cells) {
        const CellResult r = cone_cell<uint32_t>(cs, true);
        check(r.ran, "(4b) E8 cell (" + std::to_string(cs.n) + ", " + std::to_string(cs.ell) + ") ran: " + r.excluded);
    }
    // Every regime the pipeline honours (m-9): U64_Montgomery and U64_Wide.
    regime_cell<uint64_t>(10, 134217757ull);
    regime_cell<uint64_t>(10, 4294967291ull);
    closure_arm();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  E8 item 4b arms: " << (checks - c0) << " checks, " << (failures - f0) << " failing, " << secs << " s" << std::endl;
}

void HeckeReductionConeCoverageArms(uint64_t& checks, uint32_t& failures, bool large) {
    g_checks = &checks;
    g_fail = &failures;
    g_printed = 0;
    // (slot-9 WP-4t) the E8 point (60, 5), moved here from HeckeReductionConeArms unchanged:
    // K = 2, 3; p in {2, 3}; (4b-K); [E] (f)'s predicted boundary lists and (e)'s landed lengths.
    {
        const auto t60 = std::chrono::steady_clock::now();
        const uint64_t c60 = checks;
        const uint32_t f60 = failures;
        const ConeSpec cs{60, 5, {2, 3}, {2, 3}, true, false,
                          {{2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, {3, 3}, {21, 30}};
        const CellResult r = cone_cell<uint32_t>(cs, true);
        check(r.ran, "(4b) E8 cell (60, 5) ran: " + r.excluded);
        const double s60 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t60).count();
        std::cout << "  E8_4B_POINT (60, 5) checks=" << (checks - c60) << " failing=" << (failures - f60) << " " << s60 << " s"
                  << std::endl;
    }
    const uint64_t c0 = checks;
    const uint32_t f0 = failures;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<uint32_t, uint64_t>> cells;
    for (uint32_t n : {2u, 4u, 10u, 16u, 22u, 24u})
        for (uint64_t ell : {5ull, 13ull, 691ull, 10007ull, 134217757ull}) cells.push_back({n, ell});
    cells.push_back({32, 5});
    const std::vector<std::pair<uint32_t, uint64_t>> m1 = {
        {6, 5},    {8, 7},    {12, 11},  {12, 5},    {34, 5},       {58, 7},     {58, 5},
        {82, 7},   {22, 5},   {26, 7},   {32, 131},  {160, 691},    {198, 691},  {10, 691},
        {14, 3617}, {16, 43867}, {18, 283}, {18, 617}, {20, 131}, {20, 593}, {22, 103},
        {22, 2294797}, {24, 657931}};
    for (const auto& c : m1) {
        bool dup = false;
        for (const auto& d : cells) dup = dup || (d == c);
        if (!dup) cells.push_back(c);
    }
    uint32_t run = 0, left_out = 0, excluded = 0;
    for (const auto& c : cells) {
        if (!large && (c == std::make_pair(160u, uint64_t(691)) || c == std::make_pair(198u, uint64_t(691)))) {
            ++left_out;
            continue;
        }
        const ConeSpec cs{c.first, c.second, {2}, {2}, false, false, {}, {}, {}};
        const unsigned __int128 sq = static_cast<unsigned __int128>(c.second) * c.second;
        const CellResult r = (sq < (static_cast<unsigned __int128>(1) << 32)) ? cone_cell<uint32_t>(cs, false)
                                                                               : cone_cell<uint64_t>(cs, false);
        if (r.ran) ++run;
        else {
            ++excluded;
            std::cout << "  E8_4B_COVERAGE (" << c.first << ", " << c.second << ") excluded: " << r.excluded << std::endl;
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  E8_4B_COVERAGE cells=" << cells.size() << " run=" << run << " left_out=" << left_out << " excluded=" << excluded
              << " checks=" << (checks - c0) << " failing=" << (failures - f0) << " " << secs << " s" << std::endl;
}
