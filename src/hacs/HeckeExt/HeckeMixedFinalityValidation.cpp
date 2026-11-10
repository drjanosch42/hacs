// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-11) HeckeMixedFinalityValidation -- see the header.
//
// The evidence arm of Proposition M of
// docs/implementation/slot13/WP-11_mixed_finality_proposal.md: finality of an
// X_4 block V WITHOUT (d1)'s injectivity hypothesis, by gluing the Sturm law
// on the torsion-free quotient Q (it holds modulo the torsion image S_V,
// [E] cl:sturmmodl) to the boundary law on V / ker r^1 ([C, lem:bdrytorus] with
// [C, prop:bdrytranscription]) along the hypothesis
//      (H4)  K := ker(r^1|_V)  meets  S_V := e_V im(tors H^1(Z) -> H^1(F_ell))  in 0.
// S_V is computed from the Z/ell^E cohomology of the cell (HeckeReduction):
// im beta^0 (the Bockstein of [C, cl:bockstein]) when the torsion at the cell
// has exponent one, else the reductions of the torsion summands of H^1(Z/ell^E)
// at the least E that shows every summand (dim S = dim H^0(F_ell), checked).
//
// Per H^1 ideal: D, rank r^1, dim K, dim S_V, dim (K meet S_V), the H^0 part of
// the same system, (d1)'s verdict, the hypotheses (H2)-(H5) and Proposition M's
// verdict (d2), and -- evidence for the conclusion, not part of the certificate
// -- whether every computed pool operator T_q|_V lies in A_S.  Gated: the proof's
// two laws on the computed operators ((T_q - T_p) V in K for q = p mod ell;
// beta^0 o A^0(T_q) = A^1(T_q) o beta^0), S's decomposition along the blocks,
// dim S_V = dim H^0(F_ell)_V ([C, cl:bockstein](2) localised), d2 => all pool
// operators in A_S, and the pins: (34, 7) ideal 2 (the one known
// not_injective block; S_V = 0: d1 fails, d2 = not_dickson -- M is (c) there,
// Corollary M1(ii)), and the Dickson-type ideals (26, 5) 3, (30, 5) 3
// (ell^2-torsion), (50, 7) 1 (S_V != 0, K = 0: d1 and d2 pass).
//
// (H5) follows Corollary M1 exactly (referee finding M2 of WP-11_referee.md):
// automatic when S_V = 0 (d2 = not_dickson), when K = 0, or when A_S is reduced
// AND K + S_V != V; `complementary` (d2 not passed, M1(iv)) when K (+) S_V = V
// with both nonzero; otherwise (H5) is computed literally -- J as a subspace of
// A_S, pi_S o x for a basis of J, membership in A_S.  The literal path and the
// complementary verdict are exercised on every Dickson-type block with reduced
// A_S by a PLANTED arm (MIXED_FIN_PLANT): K replaced by a chosen complement of
// S_V in V (M1(iv): J = A_S, (H5) fails) and by a hyperplane of it (M1(iii):
// J = 0, (H5) holds) -- abstract data (A_S, S_V, K), no proof step claimed.
// Per cell also dim H^1(bd) and per block dim H^1(bd)_m' (the joint eigenspace
// of the boundary A^bd_1(T_p), p in R(ell), at V's system; rank r^1|_V <= it).
// Read-only; nothing is emitted.
#include "HeckeMixedFinalityValidation.h"

#include "HeckeAlgebraInvariants.h"
#include "HeckeExtDriver.h"
#include "HeckeExtRecord.h"
#include "HeckeLocalAlgebra.h"
#include "HeckeSturmBound.h"

#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeBoundaryOperator.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeReduction.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZBoundaryPair.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/SubspaceOps.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace hecke_ext;

oc_util::Checker Check("  [FAIL] ");

ExtMatrix FlatToExt(const FlatMatrix& M) {
    ExtMatrix out(M.get_rows(), M.get_cols(), M.get_p());
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) out.data[i][j] = M.get_entry(i, j);
    return out;
}

ExtMatrix DenseToExt(const engine::DenseBuf<uint32_t>& B, uint32_t ell, const std::vector<uint32_t>* cols = nullptr) {
    const uint32_t nc = cols ? static_cast<uint32_t>(cols->size()) : B.cols;
    ExtMatrix out(B.rows, nc, ell);
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t j = 0; j < nc; ++j) out.data[i][j] = B.at(i, cols ? (*cols)[j] : j) % ell;
    return out;
}

// rank of [A | B] (either side may have no columns).
uint32_t RankJoin(const ExtMatrix* A, const ExtMatrix* B, uint32_t rows, uint32_t ell) {
    std::vector<ExtMatrix> parts;
    if (A && A->cols) parts.push_back(*A);
    if (B && B->cols) parts.push_back(*B);
    if (parts.empty()) return 0;
    return hecke::RankOf(hecke::HStack(parts, rows, ell));
}

uint64_t LeastPrimeInClass(uint32_t ell, uint32_t cls, const std::vector<uint64_t>& pool) {
    for (uint64_t q : pool)
        if (q % ell == cls) return q;
    return 0;
}

bool IsPrime(uint64_t q) {
    if (q < 2) return false;
    for (uint64_t d = 2; d * d <= q; ++d)
        if (q % d == 0) return false;
    return true;
}

// The F_ell and Z/ell^E cohomology objects of the cell (HeckeReduction's inputs).
struct CohSide {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<LocalSystemSimplicialComplexT<uint32_t>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<uint32_t>> coh;
    std::unique_ptr<hecke::HeckeCosetTablesT<uint32_t>> cosets;
    std::unique_ptr<hecke::HeckeTauTableT<uint32_t>> tau;
    std::unique_ptr<hecke::HeckeOperatorOnKT<uint32_t>> op2;   // T_2 on the field side (the basis check)
};

std::unique_ptr<CohSide> FieldSide(uint32_t n, uint32_t ell) {
    auto s = std::make_unique<CohSide>();
    s->model = std::make_unique<hecke::SL2ZTessellationTables>(s->K);
    auto rels = GenerateSL2ZRelationsElt<uint32_t>(s->K, n, ell);
    s->lssc = std::make_unique<LocalSystemSimplicialComplexT<uint32_t>>(s->K, std::move(rels), n + 1, ell);
    s->coh = std::make_unique<EquivariantCohomologyOnKT<uint32_t>>(s->lssc.get());
    s->coh->compute_cohomology();
    s->cosets = std::make_unique<hecke::HeckeCosetTablesT<uint32_t>>(2, 1, ell, n);
    s->tau = std::make_unique<hecke::HeckeTauTableT<uint32_t>>(s->K, *s->model, *s->cosets, ell, n);
    s->op2 = std::make_unique<hecke::HeckeOperatorOnKT<uint32_t>>(s->coh.get(), s->cosets.get(), s->tau.get());
    s->op2->compute();
    return s;
}

std::unique_ptr<CohSide> RingSide(uint32_t n, uint32_t ell, uint32_t E) {
    hecke::ScopedCoefficientRing ring(E);
    auto s = std::make_unique<CohSide>();
    s->model = std::make_unique<hecke::SL2ZTessellationTables>(s->K);
    const engine::ModCtx<uint32_t> ctx =
        hecke::CoefficientRingContext<uint32_t>("HeckeMixedFinalityValidation (slot-13 WP-11)", ell, E);
    auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(s->K, n, ctx);
    s->lssc = std::make_unique<LocalSystemSimplicialComplexT<uint32_t>>(s->K, std::move(rels), n + 1, ctx.p);
    s->coh = std::make_unique<EquivariantCohomologyOnKT<uint32_t>>(s->lssc.get());
    s->coh->compute_cohomology();
    return s;
}

// The unique lambda with M - lambda nilpotent, or -1 (residue degree > 1).
int64_t UniqueEigenvalue(const ExtMatrix& M, uint32_t ell) {
    const ExtMatrix I(M.rows, ell);
    for (uint32_t lam = 0; lam < ell; ++lam)
        if (hecke::IsNilpotent(M - I * lam, M.rows + 1)) return lam;
    return -1;
}

struct Pin {
    uint32_t n, ell, ideal;
    bool dickson;          // S_V != 0
    bool d1;               // the expected (d1) verdict
    const char* d2;        // the expected (d2) verdict
    int32_t D, rank_r;     // -1: not pinned
};

struct Tally {
    uint32_t ideals = 0, dickson = 0, d1 = 0, d2 = 0, d2_only = 0, not_dickson = 0, complementary = 0,
             meets_torsion = 0, h5_literal = 0, all_pool = 0, h4 = 0, non_injective = 0, non_injective_dickson = 0,
             planted = 0;
};

// Membership of y in span(basis) (D x D matrices over F_ell), by rank.
bool InSpan(const std::vector<ExtMatrix>& basis, const ExtMatrix& y, uint32_t ell) {
    const uint32_t L = y.rows * y.cols, a = static_cast<uint32_t>(basis.size());
    ExtMatrix Mb(L, a, ell), My(L, a + 1, ell);
    for (uint32_t c = 0; c <= a; ++c) {
        const ExtMatrix& X = c < a ? basis[c] : y;
        for (uint32_t i = 0; i < X.rows; ++i)
            for (uint32_t j = 0; j < X.cols; ++j) {
                My.data[i * X.cols + j][c] = X.data[i][j];
                if (c < a) Mb.data[i * X.cols + j][c] = X.data[i][j];
            }
    }
    return hecke::RankOf(My) == (a ? hecke::RankOf(Mb) : 0u);
}

// Columns of `W` (D x w, full column rank) completed by identity columns to an
// invertible D x D matrix; `comp` receives the added columns.
ExtMatrix CompleteBasis(const ExtMatrix& W, uint32_t D, uint32_t ell, ExtMatrix* comp) {
    std::vector<ExtMatrix> cols;
    if (W.cols) cols.push_back(W);
    std::vector<uint32_t> added;
    uint32_t r = W.cols ? hecke::RankOf(W) : 0;
    for (uint32_t i = 0; i < D && r < D; ++i) {
        ExtMatrix e(D, 1, ell);
        e.data[i][0] = 1;
        std::vector<ExtMatrix> trial = cols;
        trial.push_back(e);
        const uint32_t r2 = hecke::RankOf(hecke::HStack(trial, D, ell));
        if (r2 > r) {
            cols = trial;
            r = r2;
            added.push_back(i);
        }
    }
    if (comp) {
        *comp = ExtMatrix(D, static_cast<uint32_t>(added.size()), ell);
        for (uint32_t c = 0; c < added.size(); ++c) comp->data[added[c]][c] = 1;
    }
    return hecke::HStack(cols, D, ell);
}

// (H5) literally (Proposition M): with W = [K_V | S_V] (V coordinates, K (+) S_V
// direct), J = {x in A_S : x(V) in W}; (H5) iff pi_S o x in A_S for a basis of J.
// Returns (H5), and dim J through `dim_J`.
bool H5Literal(const std::vector<ExtMatrix>& AS, const ExtMatrix& KV, const ExtMatrix& SV, uint32_t D, uint32_t ell,
               uint32_t* dim_J) {
    const uint32_t dk = KV.cols, ds = SV.cols, w = dk + ds;
    std::vector<ExtMatrix> parts;
    if (dk) parts.push_back(KV);
    if (ds) parts.push_back(SV);
    const ExtMatrix W = hecke::HStack(parts, D, ell);
    const ExtMatrix Bm = CompleteBasis(W, D, ell, nullptr);
    const ExtMatrix Binv = hecke::SolveColumnsExact(Bm, ExtMatrix(D, ell), "mixed finality: H5 basis");
    const uint32_t a = static_cast<uint32_t>(AS.size());
    std::vector<ExtMatrix> coords;
    for (const ExtMatrix& x : AS) coords.push_back(Binv * x);
    ExtMatrix jsys(std::max<uint32_t>(1, (D - w) * D), a, ell);
    for (uint32_t c = 0; c < a; ++c)
        for (uint32_t i = w; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) jsys.data[(i - w) * D + j][c] = coords[c].data[i][j];
    const ExtMatrix Jc = hecke::KernelAsMatrix(jsys);   // a x dim J
    *dim_J = Jc.cols;
    for (uint32_t k = 0; k < Jc.cols; ++k) {
        ExtMatrix x(D, D, ell);
        for (uint32_t c = 0; c < a; ++c) x = x + AS[c] * Jc.data[c][k];
        const ExtMatrix cx = Binv * x;
        ExtMatrix sc(ds, D, ell);
        for (uint32_t i = 0; i < ds; ++i)
            for (uint32_t j = 0; j < D; ++j) sc.data[i][j] = cx.data[dk + i][j];
        const ExtMatrix pix = ds ? SV * sc : ExtMatrix(D, D, ell);
        if (!InSpan(AS, pix, ell)) return false;
    }
    return true;
}

void Cell(uint32_t n, uint32_t ell, const std::vector<Pin>& pins, Tally& tally) {
    const std::string cell = "(" + std::to_string(n) + ", " + std::to_string(ell) + ")";
    const auto t0 = std::chrono::steady_clock::now();

    // The driver, as HeckeBoundaryFinalityValidation's FinalityCell runs it: every
    // pool prime q <= 31 plus the least prime of each class the pool misses.
    ExtDriverOptions o;
    o.run_commutant = false;
    o.run_sturm = false;
    o.with_cone = true;
    o.good_primes = AnemicPool(ell);
    for (uint32_t cls = 1; cls < ell; ++cls) {
        if (LeastPrimeInClass(ell, cls, o.good_primes) != 0) continue;
        for (uint64_t q = cls;; q += ell)
            if (IsPrime(q)) {
                o.good_primes.push_back(q);
                break;
            }
    }
    std::sort(o.good_primes.begin(), o.good_primes.end());
    HeckeExtDriver drv(n, ell, o);
    drv.run();
    const double t_driver = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double t_ring = 0;
    uint32_t ring_runs = 0;
    const std::vector<uint32_t>& dims = drv.H_dimensions();
    ENGINE_CHECK_MSG(dims.size() >= 2 && dims[1] > 0, "mixed finality: no H^1 at the cell");
    const uint32_t h0 = dims[0], h1 = dims[1];
    const hecke::SL2ZBoundaryPair* pair = drv.boundary_pair();
    ENGINE_CHECK_MSG(pair != nullptr, "mixed finality: no boundary pair");
    const ExtMatrix r1 = FlatToExt(pair->restriction().get_induced_H_eq_maps()[1]);

    auto sorted_ops = [&](uint32_t g) {
        std::vector<std::pair<uint64_t, ExtMatrix>> ops = drv.cell_operators(g);
        std::sort(ops.begin(), ops.end(), [](const std::pair<uint64_t, ExtMatrix>& a,
                                             const std::pair<uint64_t, ExtMatrix>& b) { return a.first < b.first; });
        return ops;
    };
    const std::vector<std::pair<uint64_t, ExtMatrix>> ops1 = sorted_ops(1);
    std::vector<std::pair<uint64_t, ExtMatrix>> ops0;
    if (h0 > 0) ops0 = sorted_ops(0);

    // The boundary: dim H^1(bd), and A^bd_1(T_p) for p in R(ell) (the least pool prime
    // of each class; lem:bdrytorus: the boundary system depends on p mod ell only).
    const auto tb0 = std::chrono::steady_clock::now();
    const uint32_t h1bd = pair->dim_H_bd(1);
    std::vector<std::pair<uint64_t, ExtMatrix>> bd1;
    {
        const hecke::SL2ZTessellationTables bmodel(pair->K());
        for (uint32_t cls = 1; cls < ell && h1bd > 0; ++cls) {
            uint64_t p = 0;
            for (uint64_t q : o.good_primes)
                if (q != ell && q % ell == cls) {
                    p = q;
                    break;
                }
            if (p == 0) continue;
            hecke::HeckeCosetTablesT<uint32_t> cos(p, 1, ell, n);
            hecke::HeckeBoundaryTauTableT<uint32_t> tb(pair->K(), bmodel, cos);
            hecke::HeckeBoundaryOperatorT<uint32_t> bop(&pair->boundary_cohomology(), &cos, &tb);
            bop.compute();
            bd1.push_back({p, bop.A(1)});
        }
    }
    const double t_bd = std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count();

    // The torsion image S in H^1(F_ell) (the driver's D5b basis).
    const auto tf0 = std::chrono::steady_clock::now();
    const std::unique_ptr<CohSide> f = FieldSide(n, ell);
    const double t_field = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
    Check(f->coh->get_H_eq_dimensions()[1] == h1 && f->coh->get_H_eq_dimensions()[0] == h0,
          cell + ": the field side has the driver's dimensions");
    for (const auto& op : ops1)
        if (op.first == 2)
            Check(op.second == f->op2->get_H_matrices()[1],
                  cell + ": A^1(T_2) of the field side == the driver's (one D5b basis)");
    ExtMatrix S(h1, 0, ell);
    uint32_t beta_rank = 0, E_used = 0;
    ExtMatrix beta(h1, 0, ell);
    const auto tr0 = std::chrono::steady_clock::now();
    if (h0 > 0) {
        ++ring_runs;
        const std::unique_ptr<CohSide> r2 = RingSide(n, ell, 2);
        const engine::DenseBuf<uint32_t> B0 = hecke::BocksteinH(*r2->coh, *f->coh, 0);
        beta = DenseToExt(B0, ell);
        beta_rank = hecke::RankOf(beta);
        // [C, cl:bockstein](3): beta^0 is Hecke-equivariant -- on every computed operator.
        for (const auto& op1 : ops1)
            for (const auto& op0 : ops0)
                if (op0.first == op1.first)
                    Check(op1.second * beta == beta * op0.second,
                          cell + ": beta^0 o A^0(T_" + std::to_string(op1.first) + ") = A^1(T_" +
                              std::to_string(op1.first) + ") o beta^0");
        if (beta_rank == h0) {
            S = beta;
            E_used = 2;
        } else {
            // ell^2-torsion: the torsion summands of H^1(Z/ell^E), E the least level at
            // which their number is dim H^0 ([C, cl:bockstein](2); a summand of exponent
            // >= E would read as free and the count would fall short).
            for (uint32_t E = 3; E <= 6 && E_used == 0; ++E) {
                ++ring_runs;
                const std::unique_ptr<CohSide> rE = RingSide(n, ell, E);
                const std::vector<uint32_t>& ex = rE->coh->H_invariant_factors(1);
                std::vector<uint32_t> tors;
                for (uint32_t j = 0; j < ex.size(); ++j)
                    if (ex[j] < E) tors.push_back(j);
                if (tors.size() != h0) continue;
                const engine::DenseBuf<uint32_t> R = hecke::ReduceHToField(*rE->coh, *f->coh, 1);
                Check(R.cols == ex.size(), cell + ": red_H^1 has one column per cyclic summand");
                S = DenseToExt(R, ell, &tors);
                E_used = E;
            }
            ENGINE_CHECK_MSG(E_used != 0, "mixed finality: the torsion image was not reached by E <= 6");
        }
        Check(hecke::RankOf(S) == h0, cell + ": dim S = dim H^0(F_ell) ([C, cl:bockstein](2))");
        if (beta_rank > 0)
            Check(RankJoin(&S, &beta, h1, ell) == h0, cell + ": im beta^0 is inside S");
    }
    t_ring = std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
    const uint32_t dim_S = S.cols ? hecke::RankOf(S) : 0;

    // Stage 2 on S_0 = {2, 3, 5, 7, ell}: the record's H^1 blocks, in record order.
    const std::vector<uint64_t> S0 = {2, 3, 5, 7, static_cast<uint64_t>(ell)};
    auto blocks_of = [&](const std::vector<std::pair<uint64_t, ExtMatrix>>& ops) {
        std::vector<std::pair<uint64_t, ExtMatrix>> s0;
        for (const auto& op : ops)
            if (std::find(S0.begin(), S0.end(), op.first) != S0.end()) s0.push_back(op);
        hecke::HeckeModuleDecomposition dec(s0, ell);
        dec.compute();
        ENGINE_CHECK_MSG(dec.certify() == 0, "mixed finality: the S_0 decomposition does not certify");
        return dec.blocks();
    };
    const uint32_t h0_blocks = h0 > 0 ? static_cast<uint32_t>(blocks_of(ops0).size()) : 0u;
    const std::vector<hecke::PrimaryBlock> blocks = blocks_of(ops1);

    // The Sturm bound B'(k, ell) of (c) (H3's prime range).
    SturmBound sb(n + 2, ell);
    sb.run();
    const uint32_t Bp = sb.has_B_prime() ? sb.B_prime() : 0u;

    uint32_t sum_SV = 0;
    for (uint32_t k = 0; k < blocks.size(); ++k) {
        const hecke::PrimaryBlock& b = blocks[k];
        const uint32_t id = h0_blocks + k, D = b.dim;
        const std::string tag = cell + " ideal " + std::to_string(id);
        const ExtMatrix& V = b.basis_columns;
        std::vector<uint64_t> primes;
        std::vector<ExtMatrix> t;
        for (const auto& op : ops1) {
            primes.push_back(op.first);
            t.push_back(hecke::SolveColumnsExact(V, op.second * V, "mixed finality: block restriction"));
        }
        std::vector<ExtMatrix> ts0;
        bool has_uell = false;
        for (size_t j = 0; j < primes.size(); ++j)
            if (std::find(S0.begin(), S0.end(), primes[j]) != S0.end()) {
                ts0.push_back(t[j]);
                has_uell = has_uell || primes[j] == ell;
            }
        const AlgebraInvariants AS = ComputeAlgebraInvariants(ts0, ell);
        auto member = [&](size_t j) {
            std::vector<ExtMatrix> x = ts0;
            x.push_back(t[j]);
            return ComputeAlgebraInvariants(x, ell).dim_A == AS.dim_A;
        };
        auto index_of = [&](uint64_t q) { return static_cast<size_t>(std::find(primes.begin(), primes.end(), q) - primes.begin()); };

        // K = ker r^1|_V (in H^1 coordinates), S_V = S meet V, K meet S_V.
        const ExtMatrix rV = r1 * V;
        const uint32_t rank_r = hecke::RankOf(rV);
        const uint32_t dim_K = D - rank_r;
        ExtMatrix Kb(h1, 0, ell);
        if (dim_K > 0) Kb = V * hecke::KernelAsMatrix(rV);
        const uint32_t dim_SV = dim_S + D - RankJoin(&S, &V, h1, ell);
        const uint32_t dim_KS = dim_K + dim_S - RankJoin(&Kb, &S, h1, ell);
        sum_SV += dim_SV;

        // The system (residue degree 1) and the H^0 part with the same system.
        std::vector<std::pair<uint64_t, int64_t>> lam;
        bool resdeg1 = true;
        // U_ell included: e_V is a polynomial in S_0 = {2, 3, 5, 7, ell}, and two H^1 blocks
        // can share a Dickson anemic system while differing in U_ell (U_ell is nilpotent on H^0,
        // [C, prop:heckeH0](2)); only the U_ell-nilpotent one meets the torsion image.
        for (size_t j = 0; j < primes.size(); ++j) {
            const int64_t l = UniqueEigenvalue(t[j], ell);
            resdeg1 = resdeg1 && l >= 0;
            lam.push_back({primes[j], l});
        }
        uint32_t h0V = 0;
        if (resdeg1 && h0 > 0) {
            std::vector<ExtMatrix> rows;
            for (const auto& op : ops0)
                for (const auto& e : lam)
                    if (e.first == op.first) {
                        ExtMatrix N = op.second - ExtMatrix(h0, ell) * static_cast<uint32_t>(e.second);
                        ExtMatrix P = N;
                        for (uint32_t i = 1; i < h0; ++i) P = P * N;
                        rows.push_back(P.transpose());
                    }
            // dim of the joint kernel = h0 - rank of the stacked powers.
            h0V = rows.empty() ? h0 : h0 - hecke::RankOf(hecke::HStack(rows, h0, ell));
        }
        if (resdeg1) Check(dim_SV == h0V, tag + ": dim S_V = dim H^0(F_ell)_V = " + std::to_string(h0V));

        // The proof's boundary step on the computed operators: (T_q - T_p) V in K.
        std::vector<uint64_t> pool;
        for (uint64_t q : primes)
            if (q != ell) pool.push_back(q);
        bool bstep = true;
        for (uint64_t q : pool) {
            const uint64_t p = LeastPrimeInClass(ell, static_cast<uint32_t>(q % ell), pool);
            if (p == q) continue;
            bstep = bstep && (rV * (t[index_of(q)] - t[index_of(p)])).is_zero();
        }
        Check(bstep, tag + ": r^1 (T_q - T_p) = 0 on V for q = p mod ell (lem:bdrytorus through r^1)");

        // (d1) and Proposition M's hypotheses.
        bool h2 = true;
        std::vector<uint64_t> reps;
        for (uint32_t cls = 1; cls < ell; ++cls) {
            const uint64_t q = LeastPrimeInClass(ell, cls, pool);
            if (q == 0) {
                h2 = false;
                continue;
            }
            reps.push_back(q);
            h2 = h2 && member(index_of(q));
        }
        const bool injective = rank_r == D;
        const bool d1 = has_uell && injective && h2;
        bool h3 = Bp != 0;
        for (uint64_t q = 2; h3 && q <= Bp; ++q) {
            if (!IsPrime(q)) continue;
            const size_t j = index_of(q);
            h3 = j < primes.size() && member(j);
        }
        const bool h4 = dim_KS == 0;

        // dim H^1(bd)_m': the joint eigenspace of A^bd_1(T_p), p in R(ell), at V's system.
        uint32_t h1bdV = h1bd;
        if (resdeg1 && h1bd > 0 && !bd1.empty()) {
            std::vector<ExtMatrix> rows;
            for (const auto& bp : bd1)
                for (const auto& e : lam)
                    if (e.first == bp.first) {
                        ExtMatrix N = bp.second - ExtMatrix(h1bd, ell) * static_cast<uint32_t>(e.second);
                        rows.push_back(N.transpose());
                    }
            if (!rows.empty()) h1bdV = h1bd - hecke::RankOf(hecke::HStack(rows, h1bd, ell));
            Check(rank_r <= h1bdV, tag + ": rank r^1|_V <= dim H^1(bd)_m' (r^1 is Hecke-equivariant)");
        }

        // K and S_V in V coordinates (for (H5) and the planted arm).
        ExtMatrix KV(D, 0, ell), SVv(D, 0, ell);
        if (dim_K > 0) KV = hecke::KernelAsMatrix(rV);
        if (dim_SV > 0) {
            const ExtMatrix N = hecke::KernelAsMatrix(hecke::HStack({V, S}, h1, ell));
            ExtMatrix top(D, N.cols, ell);
            for (uint32_t r = 0; r < D; ++r)
                for (uint32_t c = 0; c < N.cols; ++c) top.data[r][c] = N.data[r][c];
            // a basis of the column span of `top` (dim S_V columns).
            SVv = ExtMatrix(D, 0, ell);
            std::vector<ExtMatrix> acc;
            uint32_t rk = 0;
            for (uint32_t c = 0; c < top.cols && rk < dim_SV; ++c) {
                ExtMatrix col(D, 1, ell);
                for (uint32_t r = 0; r < D; ++r) col.data[r][0] = top.data[r][c];
                std::vector<ExtMatrix> trial = acc;
                trial.push_back(col);
                const uint32_t r2 = hecke::RankOf(hecke::HStack(trial, D, ell));
                if (r2 > rk) {
                    acc = trial;
                    rk = r2;
                }
            }
            Check(rk == dim_SV, tag + ": S_V in V coordinates has dim S_V columns");
            if (rk) SVv = hecke::HStack(acc, D, ell);
        }

        // (H5) and (d2), by Corollary M1 (referee M2): never the dim_A == 1 shortcut alone.
        std::string h5s, d2s;
        bool h5 = false;
        uint32_t dim_J = 0;
        const bool complementary = h4 && dim_K > 0 && dim_SV > 0 && dim_K + dim_SV == D;
        if (dim_SV == 0) {
            h5 = true;
            h5s = "trivial:S_V=0";
        } else if (dim_K == 0) {
            h5 = true;
            h5s = "trivial:K=0";
        } else if (!h4) {
            h5s = "not_evaluated";
        } else if (complementary) {
            h5s = "complementary";
        } else if (AS.dim_A == 1) {
            h5 = true;
            h5s = "reduced:J=0";
        } else {
            h5 = H5Literal(AS.basis, KV, SVv, D, ell, &dim_J);
            h5s = std::string(h5 ? "literal:held" : "literal:failed") + ":dim_J=" + std::to_string(dim_J);
            ++tally.h5_literal;
        }
        if (!has_uell)
            d2s = "not_applicable:u_ell";
        else if (dim_SV == 0)
            d2s = "not_dickson";   // M is (c) there (Corollary M1(ii)); production already decides it
        else if (!h4)
            d2s = "meets_torsion";
        else if (complementary)
            d2s = "complementary";   // Corollary M1(iv)
        else if (!h2)
            d2s = "failed:H2";
        else if (!h3 && dim_K > 0)   // K = 0: the proof gives t_q = b without (H3) (Corollary M1(i))
            d2s = "failed:H3";
        else if (!h5)
            d2s = "failed:H5";
        else
            d2s = "passed";
        const bool d2 = d2s == "passed";
        Check(!(complementary && d2), tag + ": the complementary case (M1(iv)) never passes");
        bool all_pool = true;
        for (size_t j = 0; j < primes.size(); ++j) all_pool = all_pool && member(j);
        Check(!d2 || all_pool, tag + ": Proposition M passed => every computed T_q|_V lies in A_S");
        Check(!d1 || d2 || d2s == "not_dickson",
              tag + ": (d1) passed => Proposition M passed or not_dickson (M generalises (d1))");

        // The planted arm (referee Sec. 2 point 8(a)): on a Dickson-type block with reduced
        // A_S every subspace is A_S-stable, so K may be replaced by a chosen complement C of
        // S_V (M1(iv): J = A_S, (H5) must fail) or by a hyperplane of C (M1(iii): J = 0, (H5)
        // must hold) -- the literal (H5) against the corollary, on abstract data.
        if (dim_SV > 0 && AS.dim_A == 1 && D > dim_SV) {
            ExtMatrix C(D, 0, ell);
            CompleteBasis(SVv, D, ell, &C);
            uint32_t jc = 0, jh = 0;
            const bool h5c = H5Literal(AS.basis, C, SVv, D, ell, &jc);
            Check(!h5c && jc == AS.dim_A, tag + " (planted K = complement): literal (H5) fails with J = A_S (M1(iv))");
            std::string hyper = "n/a";
            if (C.cols >= 2) {
                ExtMatrix Ch(D, C.cols - 1, ell);
                for (uint32_t r = 0; r < D; ++r)
                    for (uint32_t c = 0; c + 1 < C.cols; ++c) Ch.data[r][c] = C.data[r][c];
                const bool h5h = H5Literal(AS.basis, Ch, SVv, D, ell, &jh);
                Check(h5h && jh == 0, tag + " (planted K = hyperplane of the complement): (H5) holds with J = 0 (M1(iii))");
                hyper = std::string(h5h ? "held" : "failed") + ":dim_J=" + std::to_string(jh);
            }
            ++tally.planted;
            std::cout << "MIXED_FIN_PLANT n=" << n << " ell=" << ell << " ideal=" << id << " D=" << D
                      << " dim_S_V=" << dim_SV << " complement:H5=" << (h5c ? "held" : "failed") << ":dim_J=" << jc
                      << " hyperplane:H5=" << hyper << "\n";
        }

        ++tally.ideals;
        tally.dickson += dim_SV > 0;
        tally.d1 += d1;
        tally.d2 += d2;
        tally.d2_only += d2 && !d1;
        tally.not_dickson += d2s == "not_dickson";
        tally.complementary += d2s == "complementary";
        tally.meets_torsion += d2s == "meets_torsion";
        tally.all_pool += all_pool;
        tally.h4 += h4;
        tally.non_injective += dim_K > 0;
        tally.non_injective_dickson += dim_K > 0 && dim_SV > 0;
        std::cout << "MIXED_FIN n=" << n << " ell=" << ell << " ideal=" << id << " D=" << D << " dim_A=" << AS.dim_A
                  << " rank_r1=" << rank_r << " h1bd_m=" << h1bdV << " dim_K=" << dim_K << " dim_S_V=" << dim_SV
                  << " dim_K_meet_S=" << dim_KS
                  << " h0_V=" << (resdeg1 ? std::to_string(h0V) : std::string("resdeg>1"))
                  << " B'=" << Bp << " d1=" << (d1 ? "passed" : (injective ? "failed" : "not_injective"))
                  << " H2=" << h2 << " H3=" << h3 << " H4=" << h4 << " H5=" << h5s << " d2=" << d2s
                  << " all_pool_in_A=" << all_pool << "\n";
        for (const Pin& pn : pins) {
            if (pn.n != n || pn.ell != ell || pn.ideal != id) continue;
            const std::string pt = tag + " (pinned)";
            Check((dim_SV > 0) == pn.dickson, pt + ": Dickson-type (S_V != 0) as pinned");
            Check(d1 == pn.d1, pt + ": (d1) as pinned");
            Check(d2s == pn.d2, pt + ": Proposition M as pinned (" + d2s + ")");
            if (pn.D >= 0) Check(D == static_cast<uint32_t>(pn.D), pt + ": D as pinned");
            if (pn.rank_r >= 0) Check(rank_r == static_cast<uint32_t>(pn.rank_r), pt + ": rank r^1 as pinned");
        }
    }
    Check(sum_SV == dim_S, cell + ": S = (+)_V S_V (S is Hecke-stable; sum " + std::to_string(sum_SV) + ")");
    std::cout << "MIXED_FIN_CELL n=" << n << " ell=" << ell << " h0=" << h0 << " h1=" << h1 << " h1bd=" << h1bd
              << " dim_S=" << dim_S << " rank_beta0=" << beta_rank << " E=" << E_used << " B'=" << Bp
              << " t_driver=" << t_driver << " t_bd=" << t_bd << " t_field=" << t_field << " t_ring=" << t_ring
              << " ring_runs=" << ring_runs << " seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << std::endl;
}

}  // namespace

uint32_t HeckeMixedFinalityValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeMixedFinalityValidation (slot-13 WP-11): Proposition M on the not_injective block and "
                 "Dickson-type controls\n";
    const std::vector<Pin> pins = {
        {34, 7, 2, false, false, "not_dickson", 4, 2},   // the one known not_injective block (d1 fails; H^0 = 0)
        {50, 7, 3, false, false, "not_dickson", 4, 2},   // a second not_injective block, S_V = 0
        {26, 5, 3, true, true, "passed", 3, 3},          // Dickson-type, exponent-one torsion
        {30, 5, 3, true, true, "passed", 3, 3},          // Dickson-type, ell^2-torsion ([C] rmk:bocksteinexp)
        {50, 7, 1, true, true, "passed", 3, 3},          // Dickson-type at ell = 7
    };
    Tally tally;
    std::vector<std::pair<uint32_t, uint32_t>> cells = {{34, 7}, {26, 5}, {30, 5}, {50, 7}};
    // Read-only scan hook (not an emitter): HECKE_MIXED_FINALITY_CELLS="n:ell,..." replaces the
    // default cells (the pins apply wherever their cell is listed).
    if (const char* env = std::getenv("HECKE_MIXED_FINALITY_CELLS")) {
        cells.clear();
        for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(env, 9)) {
            ENGINE_CHECK_MSG(t.Fits(2, 2), "HECKE_MIXED_FINALITY_CELLS: expected n:ell");
            const uint32_t n = static_cast<uint32_t>(t.fields[0]);
            const uint32_t ell = static_cast<uint32_t>(t.fields[1]);
            ENGINE_CHECK_MSG(ell >= 5 && ell < 256 && n >= 2 && n % 2 == 0,
                             "HECKE_MIXED_FINALITY_CELLS: need n even >= 2 and 5 <= ell < 256");
            cells.push_back({n, ell});
        }
        std::cout << "HECKE_MIXED_FINALITY_CELLS: " << cells.size() << " cell(s) replace the default list\n";
    }
    for (const auto& c : cells) Cell(c.first, c.second, pins, tally);
    std::cout << oc_util::cells::EffectiveLine("HeckeMixedFinalityValidation", std::getenv("HECKE_MIXED_FINALITY_CELLS") ? "scan" : "default",
                                               cells.size())
              << "\n";
    std::cout << "MIXED_FIN_SUMMARY ideals=" << tally.ideals << " dickson_type=" << tally.dickson << " d1=" << tally.d1
              << " d2_passed=" << tally.d2 << " d2_not_d1=" << tally.d2_only << " not_dickson=" << tally.not_dickson
              << " complementary=" << tally.complementary << " meets_torsion=" << tally.meets_torsion
              << " H5_literal=" << tally.h5_literal << " non_injective=" << tally.non_injective
              << " non_injective_dickson=" << tally.non_injective_dickson << " H4=" << tally.h4
              << " all_pool_in_A=" << tally.all_pool << " planted=" << tally.planted << "\n";
    std::cout << "HeckeMixedFinalityValidation: " << Check.checks << " checks, " << Check.fails << " failing, "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    return Check.fails == 0 ? 0u : 1u;
}
