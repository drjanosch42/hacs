// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 11, WP-11 (M-3): HeckeStrictLiftRelationsValidation.
// [C] = docs/latex/gl2_contract.tex, cited by label; [IB] =
// docs/implementation/integral_line_design_brief.md.  See the header for the arms.
#include "HeckeStrictLiftRelationsValidation.h"

#include "Model/IntegralHecke.h"
#include "Model/IsotropyModel.h"
#include "Model/StrictLift.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using integral_model::DoubleCoset;
using integral_model::IsotropyModel;
using integral_model::ModelCohomology;
using integral_model::StrictLift;
using integral_model::StrictLiftBundle;
using integral_model::StrictLiftCertificate;
using U32 = uint32_t;

struct Fails {
    uint32_t n = 0;
    uint64_t checks = 0;
    uint32_t printed = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++n;
            if (printed++ < 60) std::cout << "  [FAIL] " << what << std::endl;
        }
    }
};

std::string S(uint64_t v) { return std::to_string(v); }
std::string ring_name(U32 ell, U32 k) { return k == 1 ? "F_" + S(ell) : "Z/" + S(ell) + "^" + S(k); }

DenseBuf<U32> mul(const ModCtx<U32>& c, const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    return integral_model::MatMul(c, A, B);
}
// a A + b B, a and b signed integers (reduced into the ring).
DenseBuf<U32> lin(const ModCtx<U32>& c, int64_t a, const DenseBuf<U32>& A, int64_t b, const DenseBuf<U32>& B) {
    return integral_model::LinComb(c, integral_model::ReduceSigned(c, a), A, integral_model::ReduceSigned(c, b), B);
}
DenseBuf<U32> scalar(U32 dim, U32 s) {
    DenseBuf<U32> M(dim, dim);
    for (U32 i = 0; i < dim; ++i) M.at(i, i) = s;
    return M;
}
bool equal(const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}
// p^e reduced mod the ring's modulus, exactly (no overflow: stepwise).
U32 pow_mod(const ModCtx<U32>& c, uint64_t p, uint32_t e) {
    uint64_t r = 1 % c.p;
    for (uint32_t i = 0; i < e; ++i) r = (r * (p % c.p)) % c.p;
    return static_cast<U32>(r);
}

// 64-bit helpers (MatMul is instantiated for uint32_t only).
DenseBuf<uint64_t> mul64(const ModCtx<uint64_t>& c, const DenseBuf<uint64_t>& A, const DenseBuf<uint64_t>& B) {
    ENGINE_CHECK(A.cols == B.rows);
    DenseBuf<uint64_t> C(A.rows, B.cols);
    if (A.rows == 0 || B.cols == 0 || A.cols == 0) return C;
    engine::kernels::gemm(c, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}
bool equal64(const DenseBuf<uint64_t>& A, const DenseBuf<uint64_t>& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (U32 i = 0; i < A.rows; ++i)
        for (U32 j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}

DenseBuf<U32> from_flat(const FlatMatrixT<U32>& F) {
    DenseBuf<U32> M(F.get_rows(), F.get_cols());
    for (U32 i = 0; i < F.get_rows(); ++i)
        for (U32 j = 0; j < F.get_cols(); ++j) M.at(i, j) = F.get_entry(i, j);
    return M;
}
DenseBuf<U32> columns_of(const std::vector<FlatMatrixT<U32>>& B, U32 dim) {
    DenseBuf<U32> M(dim, static_cast<U32>(B.size()));
    for (U32 t = 0; t < B.size(); ++t) {
        const U32 r = B[t].get_rows(), c = B[t].get_cols();
        for (U32 s = 0; s < r; ++s)
            for (U32 v = 0; v < c; ++v) M.at(s * c + v, t) = B[t].get_entry(s, v);
    }
    return M;
}

// The lifts, keyed by (p, kind): kind = nu for diag(1, p^nu), 0 for p I.
using LiftKey = std::pair<U32, U32>;
const StrictLift& lift_of(const std::map<LiftKey, std::unique_ptr<StrictLiftBundle>>& B, U32 p, U32 kind) {
    return B.at({p, kind})->lift();
}
std::string kind_name(U32 kind) {
    return kind == 0 ? std::string("diag(p,p)") : (kind == 1 ? std::string("diag(1,p)") : "diag(1,p^" + S(kind) + ")");
}

// One shipped (n, ell, p, nu) cell over F_ell: K's cohomology, the coset tables
// of diag(1, p^nu), and the operator on the BoundaryStrict table (strict) or on
// the canonical tau.
struct ShippedCell {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
    std::unique_ptr<LocalSystemSimplicialComplexT<U32>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<U32>> coh;
    std::unique_ptr<hecke::HeckeCosetTablesT<U32>> cosets;
    std::unique_ptr<hecke::HeckeBoundaryTauTableT<U32>> taub;
    std::unique_ptr<hecke::HeckeTauTableT<U32>> tau;
    std::unique_ptr<hecke::HeckeOperatorOnKT<U32>> op;
};

std::unique_ptr<ShippedCell> build_shipped(U32 n, U32 ell, U32 p, U32 nu, bool strict) {
    auto c = std::make_unique<ShippedCell>();
    c->tess = std::make_unique<hecke::SL2ZTessellationTables>(c->K);
    auto rels = GenerateSL2ZRelationsElt<U32>(c->K, n, ell);
    c->lssc = std::make_unique<LocalSystemSimplicialComplexT<U32>>(c->K, std::move(rels), n + 1, ell);
    c->coh = std::make_unique<EquivariantCohomologyOnKT<U32>>(c->lssc.get());
    c->coh->compute_cohomology();
    c->cosets = std::make_unique<hecke::HeckeCosetTablesT<U32>>(p, nu, ell, n);
    if (strict) {
        c->taub = std::make_unique<hecke::HeckeBoundaryTauTableT<U32>>(c->K, *c->tess, *c->cosets);
        c->tau = std::make_unique<hecke::HeckeTauTableT<U32>>(c->K, *c->tess, *c->cosets, ell, n, *c->taub);
    } else {
        c->tau = std::make_unique<hecke::HeckeTauTableT<U32>>(c->K, *c->tess, *c->cosets, ell, n);
    }
    c->op = std::make_unique<hecke::HeckeOperatorOnKT<U32>>(c->coh.get(), c->cosets.get(), c->tau.get());
    c->op->compute();
    return c;
}

// The matrix of Theta^ (degree 1) on the classes of the columns of W (a basis
// of H^1 modulo B^1 = im delta^0), column convention; the W-coordinates are
// unique because eps^* is injective on H^1 at ell >= 5 (slot-10 WP-14 m-4).
MatrixT<U32> h1_matrix(const StrictLift& lift, const ModCtx<U32>& ctx, U32 n, const DenseBuf<U32>& W) {
    const IsotropyModel& R = lift.model();
    const DenseBuf<U32> d0 = R.coboundary(ctx, n, 0);
    const DenseBuf<U32> Y = mul(ctx, lift.theta_hat(ctx, n, 1), W);
    DenseBuf<U32> Bm(W.rows, W.cols + d0.cols);
    for (U32 i = 0; i < W.rows; ++i) {
        for (U32 j = 0; j < W.cols; ++j) Bm.at(i, j) = W.at(i, j);
        for (U32 j = 0; j < d0.cols; ++j) Bm.at(i, W.cols + j) = d0.at(i, j);
    }
    const auto coords = engine::elim::solve_multi_rhs(ctx, engine::cview(Bm), engine::cview(Y));
    std::vector<std::vector<U32>> A(W.cols, std::vector<U32>(W.cols, 0));
    for (U32 i = 0; i < W.cols; ++i)
        for (U32 j = 0; j < W.cols; ++j) A[j][i] = coords[i][j];
    return MatrixT<U32>(W.cols, W.cols, ctx.p, A);
}

}  // namespace

uint32_t HeckeStrictLiftRelationsValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Fails F;
    std::cout << "=== HeckeStrictLiftRelationsValidation (slot-11 WP-11, M-3: tau_Iso for diag(1, p^2), diag(1, p^3), "
                 "diag(p, p); [C, prop:strictlift], [C, prop:diamond], [C, thm:relations] on the free model) ===\n";

    // ---- (A) the lifts and their certificates ------------------------------------
    std::cout << "--- (A) tau_Iso, degrees 0..6, certified exactly over Z[Gamma''] ---\n";
    std::map<LiftKey, std::unique_ptr<StrictLiftBundle>> B;
    const std::vector<LiftKey> keys = {{2, 1}, {2, 2}, {2, 3}, {2, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 0},
                                       {5, 2}, {5, 0}};
    for (const LiftKey& key : keys) {
        const auto [p, kind] = key;
        const auto tb = std::chrono::steady_clock::now();
        try {
            const DoubleCoset dc = kind == 0 ? DoubleCoset::scalar() : DoubleCoset::diag_one(kind);
            B[key] = std::make_unique<StrictLiftBundle>(p, dc, 5);
        } catch (const std::exception& e) {
            F.check(false, "(A) p=" + S(p) + " " + kind_name(kind) + ": the lift could not be built: " + e.what());
            continue;
        }
        const StrictLift& L = B[key]->lift();
        const StrictLiftCertificate C = L.certify();
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
        const std::string tag = "(A) p=" + S(p) + " " + kind_name(kind);
        F.check(L.double_coset().central == (kind == 0) && (kind == 0 || L.double_coset().nu == kind),
                tag + ": wrong double coset");
        F.check(C.t1_twisted_equivariance, tag + " (T1) " + C.first_failure);
        F.check(C.t2_chain_map, tag + " (T2) " + C.first_failure);
        F.check(C.t3_augmentation, tag + " (T3) " + C.first_failure);
        F.check(C.strict_eps, tag + " (S) " + C.first_failure);
        F.check(C.boundary_preserved, tag + " (B) R_d preserved " + C.first_failure);
        F.check(C.boundary_eps, tag + " (B) eps tau^d = tau_bd eps " + C.first_failure);
        std::cout << "  p=" << p << " " << kind_name(kind) << ": m=" << L.num_cosets() << " det=" << L.N()
                  << " | (T1) " << C.t1_checks << " [" << C.schreier_generators << " Schreier gens] (T2) "
                  << C.t2_checks << " (T3) " << C.t3_checks << " (S) " << C.strict_checks << " (B) "
                  << C.boundary_checks << ": " << (C.all() ? "ALL HOLD" : "FAILED") << " | max|coef| "
                  << C.max_abs_coefficient << " max|entry| " << C.max_abs_entry << " distinct gamma "
                  << C.distinct_group_elements << " | terms q=0..6 (max, max l1):";
        for (U32 q = 0; q < C.terms.size(); ++q)
            std::cout << " " << C.terms[q] << "(" << C.max_terms[q] << "," << C.max_l1[q] << ")";
        std::cout << " | " << secs << " s\n";
    }
    if (B.size() != keys.size()) {
        std::cout << "HeckeStrictLiftRelationsValidation: FAILED (" << F.n << ")\n";
        return F.n;
    }
    const IsotropyModel& M = B.at({2, 1})->model();
    {
        // Every bundle owns a model; they must be one complex (identified below).
        const ModCtx<U32> c(10007);
        bool same = true;
        for (const auto& [key, b] : B) {
            const IsotropyModel& Mk = b->model();
            for (U32 q = 0; q <= M.q_top(); ++q) same = same && (Mk.rank(q) == M.rank(q));
            for (U32 n : {0u, 5u, 10u})
                for (U32 q = 0; q <= M.q_max(); ++q) same = same && equal(Mk.coboundary(c, n, q), M.coboundary(c, n, q));
        }
        F.check(same, "(A) the models of the bundles are not one complex");
        std::cout << "  " << B.size() << " bundles, one model complex: " << (same ? "yes" : "NO") << "\n";
    }
    // Is the central lift the identity on the nose?
    for (U32 p : {2u, 3u, 5u}) {
        const StrictLift& L = lift_of(B, p, 0);
        uint64_t id_cells = 0, cells = 0;
        for (U32 q = 0; q <= L.q_top(); ++q)
            for (U32 c = 0; c < M.rank(q); ++c) {
                integral_model::IsoChain e(M.rank(q));
                e.c[c].add(integral_model::key_of(hecke::Mat22::I()), 1);
                ++cells;
                id_cells += (L.value(q, 1, c) == e) ? 1 : 0;
            }
        std::cout << "  p=" << p << " diag(p,p): tau_Iso(e_c) = e_c on " << id_cells << " / " << cells
                  << " basis cells (the induction lifts id; equality on the nose is recorded, not required)\n";
    }

    // ---- (B) cochain maps -----------------------------------------------------------
    std::cout << "--- (B) Theta^_g a cochain map, q = 0..5 ---\n";
    {
        struct Ring { U32 ell, k; };
        const std::vector<Ring> rings = {{2, 1}, {3, 1}, {5, 1}, {7, 1}, {691, 1}, {2, 3}, {3, 3}, {5, 2}};
        uint32_t ok = 0, tot = 0;
        for (const Ring& r : rings) {
            const ModCtx<U32> ctx = integral_model::RingContext(r.ell, r.k);
            for (U32 n : {0u, 1u, 2u, 4u, 10u}) {
                std::vector<DenseBuf<U32>> d;
                for (U32 q = 0; q <= M.q_max(); ++q) d.push_back(M.coboundary(ctx, n, q));
                for (const auto& [key, b] : B) {
                    if (key.second == 1) continue;   // T_p: slot-10 WP-10 (B)
                    const StrictLift& L = b->lift();
                    bool all = true;
                    DenseBuf<U32> Tq = L.theta_hat(ctx, n, 0);
                    for (U32 q = 0; q <= M.q_max(); ++q) {
                        DenseBuf<U32> Tq1 = L.theta_hat(ctx, n, q + 1);
                        all = all && equal(mul(ctx, d[q], Tq), mul(ctx, Tq1, d[q]));
                        Tq = std::move(Tq1);
                    }
                    F.check(all, "(B) " + ring_name(r.ell, r.k) + " n=" + S(n) + " p=" + S(key.first) + " " +
                                     kind_name(key.second) + ": delta Theta^ != Theta^ delta");
                    ++tot;
                    ok += all ? 1 : 0;
                }
            }
        }
        std::cout << "  delta Theta^q = Theta^{q+1} delta for q = 0..5 on " << ok << " / " << tot
                  << " (ring, n, lift) cells; rings F_2, F_3, F_5, F_7, F_691, Z/8, Z/27, Z/25; n in {0,1,2,4,10}\n";
        // The 64-bit element type (M0's Z/ell^K regime): Z/2^40 and Z/3^25 give
        // cochain maps that reduce entrywise to the 32-bit Theta^ over Z/8, Z/27.
        uint32_t ok64 = 0, tot64 = 0;
        for (U32 ell : {2u, 3u}) {
            const ModCtx<uint64_t> big = ModCtx<uint64_t>::prime_power(ell, ell == 2 ? 40 : 25);
            const ModCtx<U32> small = integral_model::RingContext(ell, 3);
            for (U32 n : {0u, 2u, 10u})
                for (const auto& [key, b] : B) {
                    if (key.second == 1 || key.first == 5) continue;
                    const StrictLift& L = b->lift();
                    bool all = true;
                    for (U32 q = 0; q < M.q_max(); ++q) {
                        const DenseBuf<uint64_t> T = L.theta_hat(big, n, q), T1 = L.theta_hat(big, n, q + 1);
                        const DenseBuf<uint64_t> d = M.coboundary(big, n, q);
                        all = all && equal64(mul64(big, d, T), mul64(big, T1, d));
                        const DenseBuf<U32> Ts = L.theta_hat(small, n, q);
                        for (U32 i = 0; i < T.rows && all; ++i)
                            for (U32 j = 0; j < T.cols; ++j)
                                if (static_cast<U32>(T.at(i, j) % small.p) != Ts.at(i, j)) { all = false; break; }
                    }
                    F.check(all, "(B) 64-bit Z/" + S(ell) + "^K n=" + S(n) + " p=" + S(key.first) + " " +
                                     kind_name(key.second) + ": not a cochain map or not reducing to Z/" + S(ell) + "^3");
                    ++tot64;
                    ok64 += all ? 1 : 0;
                }
        }
        std::cout << "  64-bit: Z/2^40, Z/3^25 cochain maps reducing entrywise to Z/8, Z/27 on " << ok64 << " / " << tot64
                  << " (ring, n, lift) cells\n";
    }

    // ---- (C) strict compatibility for diag(1, p^2) ------------------------------------
    std::cout << "--- (C) eps^* Theta^s_g = Theta^_g eps^* on C^q_eq, q <= 2, g = diag(1, p^2), over F_ell "
                 "(consistency check of the transport code) ---\n";
    {
        uint32_t ok = 0, tot = 0, datum_ok = 0;
        for (U32 ell : {2u, 3u, 5u, 7u})
            for (U32 p : {2u, 3u, 5u})
                for (U32 n : {2u, 4u, 10u}) {
                    if (p == ell) continue;
                    if (p == 5 && n > 4) continue;
                    const ModCtx<U32> ctx(ell);
                    auto c = build_shipped(n, ell, p, 2, true);
                    const StrictLift& L = lift_of(B, p, 2);
                    const bool datum = integral_model::SameIntegralDatum(*c->tau, L.tau_s());
                    F.check(datum, "(C) (n, ell)=(" + S(n) + "," + S(ell) + ") p=" + S(p) +
                                       ": the shipped strict table is not the lift's integral datum");
                    datum_ok += datum ? 1 : 0;
                    bool all = true;
                    for (U32 q = 0; q <= 2; ++q) {
                        const U32 Nq = static_cast<U32>(IsotropyModel::K_simplices()[q].size());
                        const DenseBuf<U32> Ts = from_flat(c->op->get_Theta_flat_matrices()[q]);
                        const DenseBuf<U32> Ceq = columns_of(c->coh->get_C_eq_bases_fm()[q], Nq * (n + 1));
                        const DenseBuf<U32> E = M.comparison(ctx, n, q);
                        all = all && equal(mul(ctx, E, mul(ctx, Ts, Ceq)), mul(ctx, L.theta_hat(ctx, n, q), mul(ctx, E, Ceq)));
                    }
                    F.check(all, "(C) (n, ell)=(" + S(n) + "," + S(ell) + ") p=" + S(p) +
                                     ": eps^* Theta^s != Theta^ eps^* for diag(1,p^2)");
                    ++tot;
                    ok += all ? 1 : 0;
                }
        std::cout << "  " << ok << " / " << tot << " (n, ell, p) cells equal in q = 0,1,2; datum " << datum_ok << " / "
                  << tot << " identical to the lift's\n";
    }

    // ---- (D) T(p, p) on the model; the rings of (E) ------------------------------------
    // One ring cell: the model complex over R at n and, per p, Theta^ of the four
    // lifts in every degree q = 0..5.
    struct RingSpec { U32 ell, k; };
    const std::vector<RingSpec> x_rings = {{2, 8}, {3, 5}, {2, 1}, {3, 1}, {5, 1}, {7, 1}, {691, 1}};
    const std::vector<U32> core_ns = {10, 12, 16};
    std::cout << "--- (D) T(p, p) = Theta^_{pI} induces p^n on H^q(C_I (x) R), q = 0..5 ([C, prop:diamond] on the "
                 "model, every R) ---\n";
    {
        uint32_t ok = 0, tot = 0, nose = 0, nonzero = 0;
        for (const RingSpec& r : x_rings)
            for (U32 n : core_ns) {
                const ModCtx<U32> ctx = integral_model::RingContext(r.ell, r.k);
                const ModelCohomology<U32> mc(M, ctx, n);
                for (U32 p : {2u, 3u, 5u}) {
                    const U32 pn = pow_mod(ctx, p, n);
                    for (U32 q = 0; q <= M.q_max(); ++q) {
                        const DenseBuf<U32> Cq = lift_of(B, p, 0).theta_hat(ctx, n, q);
                        const DenseBuf<U32> I = scalar(Cq.rows, pn);
                        const bool z = mc.induces_zero(q, q, lin(ctx, 1, Cq, -1, I));
                        F.check(z, "(D) " + ring_name(r.ell, r.k) + " n=" + S(n) + " p=" + S(p) + " q=" + S(q) +
                                       ": Theta^_{pI} != p^n on H^q");
                        ++tot;
                        ok += z ? 1 : 0;
                        nose += equal(Cq, I) ? 1 : 0;
                        nonzero += (mc.length_H(q) > 0 && !mc.induces_zero(q, q, I)) ? 1 : 0;
                    }
                }
            }
        std::cout << "  " << ok << " / " << tot << " (ring, n, p, q) cells; p^n acts non-trivially on H^q in " << nonzero
                  << " of them; Theta^_{pI} = p^n I on the nose in " << nose << "\n";
    }

    // ---- (E) X7 in full: the recursion of [C, prop:relationsmodel](2) (the model form of
    // [C, thm:relations](1), every ring) on H^q(C_I (x) R) -----
    std::cout << "--- (E) X7: T_{p^{nu+1}} = T_p T_{p^nu} - p^{n+1} T_{p^{nu-1}}, nu in {1, 2}, on H^q(C_I (x) R), "
                 "q = 0..5, basis-free; FULL operators of [C, eq:fullassembly]; PROVED for every ring, ell in {2, 3} "
                 "included: [C, prop:relationsmodel](2) (v1.11.0-pre.1) ---\n";
    {
        struct Ops { std::vector<DenseBuf<U32>> T1, P2, P3, C; };
        auto ops_for = [&](const ModCtx<U32>& ctx, U32 n, U32 p) {
            Ops o;
            for (U32 q = 0; q <= M.q_max(); ++q) {
                o.T1.push_back(lift_of(B, p, 1).theta_hat(ctx, n, q));
                o.P2.push_back(lift_of(B, p, 2).theta_hat(ctx, n, q));
                o.P3.push_back(lift_of(B, p, 3).theta_hat(ctx, n, q));
                o.C.push_back(lift_of(B, p, 0).theta_hat(ctx, n, q));
            }
            return o;
        };
        // (ring, n) cells: the M1(a) cells on every ring; the sweep n <= 20 on F_2, F_3.
        struct XCell { RingSpec r; U32 n; bool core; };
        std::vector<XCell> xcells;
        for (const RingSpec& r : x_rings)
            for (U32 n : core_ns) xcells.push_back({r, n, true});
        for (U32 n = 0; n <= 20; ++n) {
            if (n == 10 || n == 12 || n == 16) continue;
            for (U32 ell : {2u, 3u}) xcells.push_back({{ell, 1}, n, false});
        }
        uint64_t chain_fail[3] = {0, 0, 0}, nonzero_H = 0;
        uint64_t neg_e1_refused = 0, neg_e1_expected = 0, neg_rc_refused = 0, neg_rc_expected = 0;
        std::map<std::string, std::pair<uint32_t, uint32_t>> per_ring;   // name -> (ok, tot)
        // (slot-14 WP-14b, m-9) chain_fail's C9-style aggregate is global;
        // this per-ring split of it makes the non-vacuity claim inspectable
        // per ring, printed alongside per_ring below -- cosmetic, no F.check
        // added or removed.
        std::map<std::string, std::pair<uint64_t, uint64_t>> chain_fail_per_ring;   // name -> (nu=1, nu=2)
        for (const XCell& xc : xcells) {
            const ModCtx<U32> ctx = integral_model::RingContext(xc.r.ell, xc.r.k);
            const ModelCohomology<U32> mc(M, ctx, xc.n);
            const std::string rn = ring_name(xc.r.ell, xc.r.k);
            auto& pr = per_ring[rn + (xc.core ? " (M1(a) cells)" : " (sweep n <= 20)")];
            for (U32 p : {2u, 3u}) {
                const Ops o = ops_for(ctx, xc.n, p);
                const U32 pn = pow_mod(ctx, p, xc.n);
                const U32 pn1 = pow_mod(ctx, p, xc.n + 1);
                for (U32 q = 0; q <= M.q_max(); ++q) {
                    const U32 D = o.T1[q].rows;
                    nonzero_H += mc.length_H(q) > 0 ? 1 : 0;
                    // The full operators ([C, eq:fullassembly] on the model).
                    const DenseBuf<U32> Tp2 = lin(ctx, 1, o.P2[q], 1, o.C[q]);
                    const DenseBuf<U32> Tp3 = lin(ctx, 1, o.P3[q], static_cast<int64_t>(pn), o.T1[q]);
                    const DenseBuf<U32> TT = mul(ctx, o.T1[q], o.T1[q]);
                    // nu = 1: T_p^2 - T_{p^2} - p^{n+1} I.
                    const DenseBuf<U32> id1 = lin(ctx, 1, lin(ctx, 1, TT, -1, Tp2), -1, scalar(D, pn1));
                    // nu = 2: T_p T_{p^2} - T_{p^3} - p^{n+1} T_p.
                    const DenseBuf<U32> id2 =
                        lin(ctx, 1, lin(ctx, 1, mul(ctx, o.T1[q], Tp2), -1, Tp3), -static_cast<int64_t>(pn1), o.T1[q]);
                    const bool z1 = mc.induces_zero(q, q, id1), z2 = mc.induces_zero(q, q, id2);
                    const std::string tag = rn + " n=" + S(xc.n) + " p=" + S(p) + " q=" + S(q);
                    F.check(z1, "(E) nu=1 " + tag + ": T_p^2 != T_{p^2} + p^{n+1} on H^q");
                    F.check(z2, "(E) nu=2 " + tag + ": T_{p^3} != T_p T_{p^2} - p^{n+1} T_p on H^q");
                    pr.first += (z1 ? 1 : 0) + (z2 ? 1 : 0);
                    pr.second += 2;
                    const bool cf1 = !equal(id1, DenseBuf<U32>(D, D)), cf2 = !equal(id2, DenseBuf<U32>(D, D));
                    chain_fail[1] += cf1 ? 1 : 0;
                    chain_fail[2] += cf2 ? 1 : 0;
                    chain_fail_per_ring[rn + (xc.core ? " (M1(a) cells)" : " (sweep n <= 20)")].first += cf1 ? 1 : 0;
                    chain_fail_per_ring[rn + (xc.core ? " (M1(a) cells)" : " (sweep n <= 20)")].second += cf2 ? 1 : 0;
                    // Negative controls: the PRIMITIVE T_{p^2} ([C, err:e1]) and the
                    // coefficient (p + 1) p^n ([C, rmk:relcoeff]).  Given the identity,
                    // each differs from it by -+ p^n I, so each must be refused exactly
                    // where p^n I does not induce 0 on H^q.
                    const bool pn_zero = mc.induces_zero(q, q, scalar(D, pn));
                    const DenseBuf<U32> e1 = lin(ctx, 1, lin(ctx, 1, TT, -1, o.P2[q]), -1, scalar(D, pn1));
                    const U32 rc = static_cast<U32>((static_cast<uint64_t>(pn1) + pn) % ctx.p);
                    const DenseBuf<U32> relc = lin(ctx, 1, lin(ctx, 1, TT, -1, Tp2), -1, scalar(D, rc));
                    const bool e1_zero = mc.induces_zero(q, q, e1), rc_zero = mc.induces_zero(q, q, relc);
                    F.check(e1_zero == pn_zero, "(E) control err:e1 " + tag + ": refused iff p^n != 0 on H^q");
                    F.check(rc_zero == pn_zero, "(E) control rmk:relcoeff " + tag + ": refused iff p^n != 0 on H^q");
                    neg_e1_expected += pn_zero ? 0 : 1;
                    neg_e1_refused += e1_zero ? 0 : 1;
                    neg_rc_expected += pn_zero ? 0 : 1;
                    neg_rc_refused += rc_zero ? 0 : 1;
                }
            }
        }
        for (const auto& [name, v] : per_ring)
            std::cout << "  " << name << ": " << v.first << " / " << v.second << " (n, p, q, nu) identities hold\n";
        for (const auto& [name, v] : chain_fail_per_ring)
            std::cout << "  " << name << ": cochain-level failure count nu=1 " << v.first << ", nu=2 " << v.second
                       << " (m-9)\n";
        F.check(chain_fail[1] > 0 && chain_fail[2] > 0,
                "(E) X7 is vacuous at the cochain level (the identity holds on the nose everywhere)");
        F.check(neg_e1_refused > 0 && neg_rc_refused > 0, "(E) the negative controls were never refused");
        std::cout << "  non-vacuity: the identity fails at the cochain level in " << chain_fail[1] << " (nu = 1) / "
                  << chain_fail[2] << " (nu = 2) (ring, n, p, q) cases -- it holds on cohomology only; H^q != 0 in "
                  << nonzero_H << " cases\n"
                  << "  negative controls: primitive T_{p^2} ([C, err:e1]) refused in " << neg_e1_refused << " / "
                  << neg_e1_expected << " cases where p^n != 0 on H^q; (p + 1) p^n ([C, rmk:relcoeff]) refused in "
                  << neg_rc_refused << " / " << neg_rc_expected << "; accepted exactly where p^n = 0 on H^q\n";
    }

    // ---- (F) ell >= 5: the shipped primitive operator on H^1 ------------------------
    std::cout << "--- (F) [consistency check] Theta^_{diag(1,p^nu)} on H^1(C_I (x) F_ell) through eps^* = shipped "
                 "primitive A^(1) (canonical tau), nu in {2, 3}, ell >= 5 ---\n";
    {
        struct DC { U32 n, ell; };
        const std::vector<DC> cells = {{10, 691}, {12, 5}, {16, 7}, {12, 13}, {4, 11}, {20, 691}};
        uint32_t ok = 0, tot = 0;
        for (const DC& dc : cells)
            for (U32 p : {2u, 3u, 5u})
                for (U32 nu : {2u, 3u}) {
                    if (p == dc.ell) continue;
                    if (p == 5 && (nu == 3 || dc.n > 12)) continue;
                    const ModCtx<U32> ctx(dc.ell);
                    auto c = build_shipped(dc.n, dc.ell, p, nu, false);
                    const auto& Hb = c->coh->get_H_eq_bases_fm()[1];
                    const U32 N1 = static_cast<U32>(IsotropyModel::K_simplices()[1].size());
                    const DenseBuf<U32> Z = columns_of(Hb, N1 * (dc.n + 1));
                    const StrictLift& L = lift_of(B, p, nu);
                    const DenseBuf<U32> W = mul(ctx, M.comparison(ctx, dc.n, 1), Z);
                    const MatrixT<U32>& A = c->op->get_H_matrices()[1];
                    bool same = false;
                    std::string why;
                    try {
                        const MatrixT<U32> Ah = h1_matrix(L, ctx, dc.n, W);
                        same = (Ah.rows == A.rows && Ah.cols == A.cols && Ah.data == A.data);
                        if (!same) why = " (matrices differ)";
                    } catch (const std::exception& e) {
                        why = std::string(" (") + e.what() + ")";
                    }
                    F.check(same, "(F) (n, ell)=(" + S(dc.n) + "," + S(dc.ell) + ") p=" + S(p) + " nu=" + S(nu) +
                                      ": Theta^ on H^1 != shipped primitive A^(1)" + why);
                    ++tot;
                    ok += same ? 1 : 0;
                }
        std::cout << "  " << ok << " / " << tot << " (n, ell, p, nu) cells EQUAL\n";
    }

    // ---- (G) negative control ----------------------------------------------------------
    std::cout << "--- (G) negative control: one perturbed entry of the diag(1, p^2) lift ---\n";
    {
        const StrictLift bad = lift_of(B, 2, 2).perturbed(1, 2, 5);
        const StrictLiftCertificate C = bad.certify();
        F.check(!C.all(), "(G) a perturbed diag(1,4) lift passed certify()");
        std::cout << "  certify() " << (C.all() ? "ACCEPTED (bad)" : "rejects it: " + C.first_failure) << "\n";
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeStrictLiftRelationsValidation: " << F.checks << " checks, " << F.n << " failing, " << secs
              << " s\n";
    return F.n;
}
