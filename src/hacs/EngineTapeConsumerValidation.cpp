// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineTapeConsumerValidation.cpp -- see the header.
#include "EngineTapeConsumerValidation.h"

#include "../ModArithmetic/EngineConfig.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../Engine/EngineRuntime.h"
#include "../Engine/ProgramCache.h"
#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/Kernels.h"
#include "EquivariantCohomologyOnK.h"
#include "Hecke/HeckeCosetTables.h"
#include "Hecke/HeckeOperatorOnK.h"
#include "Hecke/HeckeTauTable.h"
#include "Hecke/SL2ZTessellationTables.h"
#include "LocalSystemSL2Z.h"
#include "LocalSystemSimplicialComplex.h"
#include "SimplicialCohomology.h"

namespace {

using namespace engine;
using engine::tape::ProgramCache;
using kernels::GemmOp;

struct Checker {
    uint64_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) { ++fails; std::cout << "  FAIL: " << what << std::endl; }
    }
};
Checker g;

bool same_dense(const DenseBuf<uint32_t>& a, const DenseBuf<uint32_t>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}
bool same_block(const BlockMat<uint32_t>& a, const BlockMat<uint32_t>& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    if (a.rows() == 0 || a.cols() == 0) return true;
    return same_dense(a.to_dense(Layout::RowMajor), b.to_dense(Layout::RowMajor));
}
bool same_flat(const FlatMatrix& a, const FlatMatrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}
bool same_matrix(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}

std::vector<LocalSystemSimplicialComplex::SimplexRelation> BuildRelations(const SimplicialComplex& K,
                                                                          uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(datum.rel);
    return rels;
}

// Every point owns its OWN complex and tessellation tables: SimplicialComplex
// caches the trivial cocycle basis Z1 once per object, for the FIRST prime it
// is asked for (SimplicialCohomology.cpp, compute_trivial_cocycle_bases_elt:
// `if (computed) return;`), so a K shared across primes would hand later
// points the first prime's basis entries (a pre-existing hazard of every
// multi-prime driver that shares one K; the pipeline reduces those entries
// mod the new prime silently, the direct check below would not).
struct Point {
    std::unique_ptr<SimplicialComplex> Kown;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    const SimplicialComplex* K;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    uint32_t n, ell;
};

std::unique_ptr<Point> make_point(uint32_t n, uint32_t ell) {
    auto pt = std::make_unique<Point>();
    pt->Kown = std::make_unique<SimplicialComplex>(CreateSL2ZFundamentalDomainTriangulation());
    pt->model = std::make_unique<hecke::SL2ZTessellationTables>(*pt->Kown);
    const SimplicialComplex& K = *pt->Kown;
    pt->K = &K; pt->n = n; pt->ell = ell;
    pt->lssc = std::make_unique<LocalSystemSimplicialComplex>(K, BuildRelations(K, n, ell), n + 1, ell);
    pt->coh = std::make_unique<EquivariantCohomologyOnK>(pt->lssc.get());
    pt->coh->compute_cohomology();
    return pt;
}

// The direct block path of compute_cohomology(): engine::blk on the stored
// constraint matrices, compared entrywise with the tape-produced bases.
void check_cohomology_direct(const Point& pt, const std::string& tag) {
    const EquivariantCohomologyOnK& coh = *pt.coh;
    const uint32_t d = coh.get_representation_dim();
    const uint32_t ell = pt.ell;
    const SimplicialComplex* Kc = coh.get_lssc()->get_base_K_complex();   // the complex the object computed on
    for (uint32_t q = 0; q < coh.get_H_eq_dimensions().size(); ++q) {
        const std::string t = tag + " q=" + std::to_string(q);
        const BlockMat<uint32_t> C = blk::kernel_basis<uint32_t>(coh.A_C_block(q), Layout::ColMajor);
        g.check(same_block(C, coh.C_eq_block(q)), t + " C_eq tape == blk::kernel_basis");
        // Z_eq = (beta (x) I) ker A_Zq
        const auto& Z1 = Kc->get_trivial_cocycle_basis_Z1_elt<uint32_t>(q);
        const uint32_t N_q = Kc->get_num_q_simplices(q), k_q = static_cast<uint32_t>(Z1.size());
        const BlockMat<uint32_t> V = blk::kernel_basis<uint32_t>(coh.A_Z_block(q), Layout::ColMajor);
        auto pool = coh.block_pool();
        BlockBuilder<uint32_t> eb(pool, std::vector<uint32_t>(N_q, d), std::vector<uint32_t>(k_q, d), ell);
        for (uint32_t j = 0; j < k_q; ++j)
            for (uint32_t s = 0; s < N_q; ++s) {
                const uint32_t beta = Z1[j].get_entry(s, 0);
                if (beta != 0) eb.set_scalar(s, j, beta);
            }
        const BlockMat<uint32_t> E = eb.finalize();
        DenseBuf<uint32_t> sv;
        const DenseView<const uint32_t> Vv = blk::multivector_view<uint32_t>(V, sv);
        DenseBuf<uint32_t> Z(E.rows(), Vv.cols(), Layout::ColMajor);
        if (Vv.cols() > 0) blk::gemm_dense<uint32_t>(E, Vv, view(Z));
        g.check(coh.Z_eq_block(q).rows() == Z.rows && coh.Z_eq_block(q).cols() == Z.cols &&
                    (Z.cols == 0 || same_dense(coh.Z_eq_block(q).to_dense(Layout::ColMajor), Z)),
                t + " Z_eq tape == blk gemm_dense of blk::kernel_basis");
        // H, P = factor_basis(B_eq, Z_eq)
        const blk::FactorBasisResult<uint32_t> H = blk::factor_basis<uint32_t>(coh.B_eq_block(q), coh.Z_eq_block(q));
        g.check(static_cast<uint32_t>(H.pivots.size()) == coh.get_H_eq_dimensions()[q], t + " dim H tape == blk::factor_basis");
        g.check(same_block(H.basis, coh.H_eq_block(q)), t + " H_eq tape == blk::factor_basis");
        const BlockMat<uint32_t>& P = coh.P_block(q);
        g.check(P.rows() == H.projector.rows && P.cols() == H.projector.cols &&
                    (P.rows() == 0 || same_dense(P.to_dense(Layout::RowMajor), H.projector)),
                t + " P tape == blk::factor_basis projector");
        if (q > 0) {
            const blk::FactorBasisResult<uint32_t> W = blk::factor_basis<uint32_t>(coh.Z_eq_block(q - 1), coh.C_eq_block(q - 1));
            const FlatMatrix D1 = Kc->coboundary_operator_elt<uint32_t>(q - 1, ell);
            const uint32_t N_src = Kc->get_num_q_simplices(q - 1);
            BlockBuilder<uint32_t> db(pool, std::vector<uint32_t>(N_q, d), std::vector<uint32_t>(N_src, d), ell);
            for (uint32_t i = 0; i < N_q; ++i)
                for (uint32_t j = 0; j < N_src; ++j) {
                    const uint32_t s = D1.get_entry(i, j);
                    if (s != 0) db.set_scalar(i, j, s);
                }
            const BlockMat<uint32_t> D = db.finalize();
            DenseBuf<uint32_t> sw;
            const DenseView<const uint32_t> Wv = blk::multivector_view<uint32_t>(W.basis, sw);
            DenseBuf<uint32_t> B(D.rows(), Wv.cols(), Layout::ColMajor);
            if (Wv.cols() > 0) blk::gemm_dense<uint32_t>(D, Wv, view(B));
            g.check(coh.B_eq_block(q).cols() == B.cols && (B.cols == 0 || same_dense(coh.B_eq_block(q).to_dense(Layout::ColMajor), B)),
                    t + " B_eq tape == blk path");
        }
    }
}

struct HeckeRun {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeOperatorOnK> op;
};
std::unique_ptr<HeckeRun> run_hecke(const Point& pt, uint64_t p, uint32_t r) {
    auto h = std::make_unique<HeckeRun>();
    h->cosets = std::make_unique<hecke::HeckeCosetTables>(p, r, pt.ell, pt.n);
    h->tau = std::make_unique<hecke::HeckeTauTable>(*pt.K, *pt.model, *h->cosets, pt.ell, pt.n);
    h->op = std::make_unique<hecke::HeckeOperatorOnK>(pt.coh.get(), h->cosets.get(), h->tau.get());
    h->op->compute();
    return h;
}

// The direct block path of HeckeOperatorOnK::compute(): apply_route2 column
// by column against Theta_block, then blk::gemm_dense / blk::solve_multi_rhs
// on the stored objects against A^{(q)} and Theta_eq.
void check_hecke_direct(const Point& pt, const HeckeRun& h, const std::string& tag) {
    const hecke::HeckeOperatorOnK& op = *h.op;
    const EquivariantCohomologyOnK& coh = *pt.coh;
    const uint32_t d = pt.n + 1;
    const auto& A = op.get_H_matrices();
    const auto& Te = op.get_Theta_eq_matrices();
    for (uint32_t q = 0; q < 3; ++q) {
        const std::string t = tag + " q=" + std::to_string(q);
        const BlockMat<uint32_t>& Theta = op.Theta_block(q);
        const uint32_t dim = pt.K->get_num_q_simplices(q) * d;
        // route 2 on every standard basis vector == the fused Theta column
        bool cols_ok = true;
        for (uint32_t b = 0; b < dim && cols_ok; ++b) {
            FlatMatrix e(dim, 1, pt.ell);
            e.set_entry(b, 0, 1);
            const FlatMatrix y = op.apply_route2(q, e);
            for (uint32_t i = 0; i < dim; ++i)
                if (y.get_entry(i, 0) != Theta.get_entry(i, b)) { cols_ok = false; break; }
        }
        g.check(cols_ok, t + " Theta_block columns == apply_route2 (matrix-free)");
        // A = P . (Theta . H)
        const BlockMat<uint32_t>& H = coh.H_eq_block(q);
        const uint32_t d_q = H.cols();
        g.check(A[q].rows == d_q && A[q].cols == d_q, t + " A shape");
        if (d_q > 0) {
            DenseBuf<uint32_t> sh;
            const DenseView<const uint32_t> Hv = blk::multivector_view<uint32_t>(H, sh);
            DenseBuf<uint32_t> Y(dim, d_q, Layout::ColMajor);
            blk::gemm_dense<uint32_t>(Theta, Hv, view(Y));
            DenseBuf<uint32_t> Ab(d_q, d_q, Layout::RowMajor);
            blk::gemm_dense<uint32_t>(coh.P_block(q), cview(Y), view(Ab));
            bool ok = true;
            for (uint32_t i = 0; i < d_q && ok; ++i)
                for (uint32_t j = 0; j < d_q; ++j)
                    if (A[q].data[i][j] != Ab.at(i, j)) { ok = false; break; }
            g.check(ok, t + " A^{(q)} tape == blk path");
        }
        // Theta_eq by the direct solve
        const BlockMat<uint32_t>& C = coh.C_eq_block(q);
        const uint32_t dim_C = C.cols();
        g.check(Te[q].get_rows() == dim_C && Te[q].get_cols() == dim_C, t + " Theta_eq shape");
        if (dim_C > 0) {
            DenseBuf<uint32_t> sc;
            const DenseView<const uint32_t> Cv = blk::multivector_view<uint32_t>(C, sc);
            DenseBuf<uint32_t> W(dim, dim_C, Layout::ColMajor);
            blk::gemm_dense<uint32_t>(Theta, Cv, view(W));
            const DenseBuf<uint32_t> coords = blk::solve_multi_rhs<uint32_t>(coh.block_pool(), pt.ell, coh.block_row_dims(q), Cv, cview(W));
            bool ok = true;
            for (uint32_t j = 0; j < dim_C && ok; ++j)
                for (uint32_t tcol = 0; tcol < dim_C; ++tcol)
                    if (Te[q].get_entry(j, tcol) != coords.at(tcol, j)) { ok = false; break; }
            g.check(ok, t + " Theta_eq tape == blk::solve_multi_rhs");
        }
    }
}

bool same_hecke(const HeckeRun& a, const HeckeRun& b) {
    const auto& Ta = a.op->get_Theta_flat_matrices();
    const auto& Tb = b.op->get_Theta_flat_matrices();
    const auto& Ea = a.op->get_Theta_eq_matrices();
    const auto& Eb = b.op->get_Theta_eq_matrices();
    const auto& Aa = a.op->get_H_matrices();
    const auto& Ab = b.op->get_H_matrices();
    for (uint32_t q = 0; q < 3; ++q)
        if (!same_flat(Ta[q], Tb[q]) || !same_flat(Ea[q], Eb[q]) || !same_matrix(Aa[q], Ab[q])) return false;
    return a.op->to_json() == b.op->to_json();
}
bool same_cohomology(const EquivariantCohomologyOnK& a, const EquivariantCohomologyOnK& b) {
    if (a.get_H_eq_dimensions() != b.get_H_eq_dimensions()) return false;
    for (uint32_t q = 0; q < a.get_H_eq_dimensions().size(); ++q)
        if (!same_block(a.C_eq_block(q), b.C_eq_block(q)) || !same_block(a.Z_eq_block(q), b.Z_eq_block(q)) ||
            !same_block(a.B_eq_block(q), b.B_eq_block(q)) || !same_block(a.H_eq_block(q), b.H_eq_block(q)) ||
            !same_block(a.P_block(q), b.P_block(q)))
            return false;
    return true;
}

// (WP-6b) The policy a thread WITHOUT a thread-local override resolves to:
// the process-wide override when one is active, else from_env().
engine::tape::DevicePolicy consumer_pass_options_policy_without_thread() {
    engine::tape::DevicePolicy d = engine::tape::DevicePolicy::Cpu;
    std::thread th([&] { d = engine::tape::consumer_pass_options().effective_policy(); });
    th.join();
    return d;
}

} // namespace

uint32_t EngineTapeConsumerValidation() {
    using namespace engine::tape;
    std::cout << "================= EngineTapeConsumerValidation =================\n";
    if (engine::backend_is_legacy()) {
        // The legacy arms (ENGINE_BACKEND=legacy) do not record programs, so
        // there is no cache behaviour to assert: report SKIP, not a failure.
        std::cout << "EngineTapeConsumerValidation: SKIP (ENGINE_BACKEND=legacy: the consumers' legacy arms record no programs)\n";
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ProgramCache<uint32_t>& cache = ProgramCache<uint32_t>::global();
    cache.clear();

    // ---- 1. tape vs the direct block path, entrywise ------------------------
    const std::vector<std::pair<uint32_t, uint32_t>> points = {{8, 13}, {16, 134217757}, {10, 691}, {24, 5}, {7, 7}};
    std::vector<std::unique_ptr<Point>> pts;
    for (const auto& ne : points) {
        const std::string tag = "cohom n=" + std::to_string(ne.first) + " ell=" + std::to_string(ne.second);
        try {
            pts.push_back(make_point(ne.first, ne.second));
        } catch (const std::exception& e) { std::cout << "  EXCEPTION in " << tag << " (compute): " << e.what() << "\n"; throw; }
        try {
            check_cohomology_direct(*pts.back(), tag);
        } catch (const std::exception& e) { std::cout << "  EXCEPTION in " << tag << " (direct check): " << e.what() << "\n"; throw; }
    }
    std::vector<std::unique_ptr<HeckeRun>> runs;
    for (const auto& pt : pts)
        for (uint64_t p : {2ull, 3ull}) {
            runs.push_back(run_hecke(*pt, p, 1));
            check_hecke_direct(*pt, *runs.back(), "hecke n=" + std::to_string(pt->n) + " ell=" + std::to_string(pt->ell) + " p=" + std::to_string(p));
        }
    {
        const auto& pt = *pts[0];
        runs.push_back(run_hecke(pt, 2, 2));
        check_hecke_direct(pt, *runs.back(), "hecke n=8 ell=13 p=2 r=2");
    }

    // ---- 2. program-cache replay == fresh recording == forced fresh compile --
    {
        const auto s0 = cache.stats();
        auto again = make_point(16, 134217757);                    // second run of the same shape: replays
        const auto s1 = cache.stats();
        g.check(s1.hits > s0.hits, "cohom replay: the second run of the same shape HIT the program cache");
        g.check(same_cohomology(*pts[1]->coh, *again->coh), "cohom replay == first run (all bases, projectors)");
        std::unique_ptr<Point> fresh;
        {
            ScopedProgramCacheBypass bypass(true);
            fresh = make_point(16, 134217757);                     // forced fresh compile
        }
        g.check(same_cohomology(*pts[1]->coh, *fresh->coh), "cohom forced-fresh compile == replay");

        const auto h0 = cache.stats();
        auto hr = run_hecke(*again, 2, 1);
        const auto h1 = cache.stats();
        g.check(h1.hits > h0.hits, "hecke replay: same (n, ell, p, r) HIT the program cache");
        g.check(same_hecke(*runs[2], *hr), "hecke replay == first run (Theta_flat, Theta_eq, A, JSON)");
        std::unique_ptr<HeckeRun> hf;
        {
            ScopedProgramCacheBypass bypass(true);
            hf = run_hecke(*again, 2, 1);
        }
        g.check(same_hecke(*runs[2], *hf), "hecke forced-fresh compile == replay");
        // a replay under a DIFFERENT pool size
        {
            ThreadPool pool(3);
            ScopedConsumerPool sp(&pool);
            auto h3 = run_hecke(*again, 2, 1);
            g.check(same_hecke(*runs[2], *h3), "hecke replay on a 3-worker pool == first run");
            auto c3 = make_point(16, 134217757);
            g.check(same_cohomology(*pts[1]->coh, *c3->coh), "cohom on a 3-worker pool == first run");
        }
    }

    // ---- 3. a changed tau pattern MISSES (never replays a stale program) ------
    {
        const auto& pt = *pts[1];   // n=16 ell=2^27
        const SimplicialComplex& K = *pt.K;
        const hecke::SL2ZTessellationTables& model = *pt.model;
        const uint32_t N1 = K.get_num_q_simplices(1);
        hecke::HeckeCosetTables c2(2, 1, pt.ell, pt.n), c3(3, 1, pt.ell, pt.n);
        hecke::HeckeTauTable t2(K, model, c2, pt.ell, pt.n), t3(K, model, c3, pt.ell, pt.n);
        const uint64_t hp2 = hecke::HeckeOperatorOnK::tau_pattern_hash(t2, c2, 1, N1, c2.num_cosets(), pt.ell);
        const uint64_t hp3 = hecke::HeckeOperatorOnK::tau_pattern_hash(t3, c3, 1, N1, c3.num_cosets(), pt.ell);
        g.check(hp2 != hp3, "tau pattern hash differs between p=2 and p=3 (same n, ell)");
        const uint64_t hp2b = hecke::HeckeOperatorOnK::tau_pattern_hash(t2, c2, 1, N1, c2.num_cosets(), pt.ell);
        g.check(hp2 == hp2b, "tau pattern hash is a pure function of the table");
        // Same (n, ell, r) but another p: the fused program must MISS.
        const auto m0 = cache.stats();
        auto h5 = run_hecke(pt, 5, 1);
        const auto m1 = cache.stats();
        g.check(m1.misses > m0.misses && m1.inserts > m0.inserts, "p=5 at a shape already cached for p=2,3: fused program MISSED and was recorded");
        check_hecke_direct(pt, *h5, "hecke n=16 ell=2^27 p=5 (after miss)");
        // Probe the grid for a mod-ell triangle-fill fallback (audit 02 Sec. 6.5(2)):
        // where one exists, its pattern hash must differ from the Z-solved
        // table of the same (n, p, r) at another ell, and the point must run
        // correctly (a miss, not a stale replay).
        bool found = false;
        for (uint32_t n : {4u, 8u, 10u, 16u, 22u})
            for (uint32_t ell : {2u, 3u, 5u, 7u, 13u})
                for (uint64_t p : {2ull, 3ull, 5ull}) {
                    if (found) break;
                    hecke::HeckeCosetTables cc(p, 1, ell, n);
                    hecke::HeckeTauTable tt(K, model, cc, ell, n);
                    if (!tt.used_mod_ell_solve()) continue;
                    found = true;
                    std::cout << "  mod-ell fallback point found: n=" << n << " ell=" << ell << " p=" << p << "\n";
                    hecke::HeckeCosetTables cz(p, 1, 134217757u, n);
                    hecke::HeckeTauTable tz(K, model, cz, 134217757u, n);
                    const uint64_t hf = hecke::HeckeOperatorOnK::tau_pattern_hash(tt, cc, 1, N1, cc.num_cosets(), ell);
                    const uint64_t hz = hecke::HeckeOperatorOnK::tau_pattern_hash(tz, cz, 1, N1, cz.num_cosets(), 134217757u);
                    std::cout << "    pattern hash (fallback ell) 0x" << std::hex << hf << " vs (Z-solved, ell=2^27) 0x" << hz << std::dec
                              << (hf != hz ? " -- differ" : " -- equal (same support)") << "\n";
                    auto ptf = make_point(n, ell);
                    auto hh = run_hecke(*ptf, p, 1);
                    check_hecke_direct(*ptf, *hh, "hecke mod-ell-fallback point n=" + std::to_string(n) + " ell=" + std::to_string(ell) + " p=" + std::to_string(p));
                }
        if (!found)
            std::cout << "  no mod-ell triangle-fill fallback in the probed cells; the p=5 miss above covers the "
                         "changed-pattern case\n";
    }

    // ---- 4. (slot-8 WP-6b) the THREAD-LOCAL device override ----------------
    // ScopedThreadDevice replaces the device fields on the installing thread
    // (and on the task threads run_concurrently() starts from it) and nowhere
    // else; with none installed consumer_pass_options() / elim_pass_options()
    // are what they were before WP-6b.  The tallies count, from the programs
    // actually run, that a CPU-overridden thread ran ZERO GPU-assigned nodes
    // while a GPU-overridden one ran >= 1 -- concurrently, on real pipeline
    // work -- and the values agree with the reference run either way (with no
    // device the GPU nodes fall back to the CPU inside the executor).
    {
        std::cout << "  -- 4. thread-local device override (WP-6b) --\n";
        const PassOptions env = PassOptions::from_env();
        const bool env_proc = pass_options_overridden();
        g.check(thread_device_override() == ThreadDevice::Inherit, "no thread-local device override by default");
        g.check(env_proc || pass_options_fingerprint(consumer_pass_options()) == pass_options_fingerprint(env),
                "no override: consumer_pass_options() == from_env() (fingerprint)");
        g.check(elim_pass_options().effective_policy() == env.effective_policy() &&
                    pass_options_fingerprint(elim_pass_options()) == pass_options_fingerprint(env),
                "no override: elim_pass_options() == from_env()");
        {
            ScopedThreadDevice sd(ThreadDevice::Gpu);
            g.check(consumer_pass_options().effective_policy() == DevicePolicy::Gpu &&
                        elim_pass_options().effective_policy() == DevicePolicy::Gpu,
                    "Gpu override: consumer and elimination options select the gpu policy");
            g.check((pass_options_fingerprint(consumer_pass_options()) & 128u) != 0,
                    "Gpu override: the cache-key fingerprint carries the device bit");
            DevicePolicy other = DevicePolicy::Gpu;
            std::thread th([&] { other = consumer_pass_options().effective_policy(); });
            th.join();
            g.check(other == consumer_pass_options_policy_without_thread(),
                    "Gpu override: a plain std::thread does NOT see it");
            ThreadPool pool(4);
            ScopedConsumerPool sp(&pool);
            std::vector<DevicePolicy> seen(3, DevicePolicy::Cpu);
            std::vector<std::function<void()>> tasks;
            for (size_t i = 0; i < seen.size(); ++i)
                tasks.push_back([&seen, i] { seen[i] = consumer_pass_options().effective_policy(); });
            run_concurrently(tasks);
            bool all = true;
            for (DevicePolicy d : seen) all = all && d == DevicePolicy::Gpu;
            g.check(all, "Gpu override: run_concurrently()'s task threads inherit it");
        }
        g.check(thread_device_override() == ThreadDevice::Inherit, "the override is gone after its scope");
        {
            // The Cpu override wins over a process-wide gpu override, on its
            // own thread only.
            PassOptions og = PassOptions::from_env();
            og.device_policy = Device::GPU; og.device_model = true; og.policy = DevicePolicy::Gpu;
            ScopedPassOptions so(og);
            ScopedThreadDevice sd(ThreadDevice::Cpu);
            g.check(consumer_pass_options().effective_policy() == DevicePolicy::Cpu &&
                        (pass_options_fingerprint(consumer_pass_options()) & 128u) == 0,
                    "Cpu override beats a process-wide gpu override on its thread");
            DevicePolicy other = DevicePolicy::Cpu;
            std::thread th([&] { other = consumer_pass_options().effective_policy(); });
            th.join();
            g.check(other == DevicePolicy::Gpu, "... while another thread still sees the process-wide gpu override");
        }
        // Concurrent real work: one thread under Gpu, one under Cpu.
        DeviceTally gpu_t, cpu_t;
        std::unique_ptr<HeckeRun> hg, hc;
        std::unique_ptr<Point> pg, pc;
        std::string err;
        std::mutex err_mu;
        auto body = [&](ThreadDevice d, DeviceTally* t, std::unique_ptr<Point>& p, std::unique_ptr<HeckeRun>& h) {
            try {
                ScopedThreadDevice sd(d, t);
                p = make_point(8, 13);
                h = run_hecke(*p, 2, 1);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_mu);
                err = e.what();
            }
        };
        std::thread tg(body, ThreadDevice::Gpu, &gpu_t, std::ref(pg), std::ref(hg));
        std::thread tc(body, ThreadDevice::Cpu, &cpu_t, std::ref(pc), std::ref(hc));
        tg.join();
        tc.join();
        g.check(err.empty(), "concurrent Gpu / Cpu threads ran without an exception" + (err.empty() ? std::string() : ": " + err));
        if (err.empty()) {
            std::cout << "  Gpu thread: " << gpu_t.programs.load() << " programs, " << gpu_t.gpu_programs.load()
                      << " with GPU nodes, " << gpu_t.gpu_nodes.load() << " GPU nodes; Cpu thread: "
                      << cpu_t.programs.load() << " programs, " << cpu_t.gpu_nodes.load() << " GPU nodes\n";
            g.check(cpu_t.programs.load() > 0 && cpu_t.gpu_nodes.load() == 0,
                    "the Cpu thread ran programs, with ZERO GPU-assigned nodes");
            g.check(gpu_t.programs.load() > 0 && gpu_t.gpu_nodes.load() > 0,
                    "the Gpu thread ran programs with >= 1 GPU-assigned node");
            g.check(same_cohomology(*pts[0]->coh, *pg->coh) && same_cohomology(*pts[0]->coh, *pc->coh),
                    "cohomology under the Gpu / Cpu thread override == the reference run");
            g.check(same_hecke(*runs[0], *hg) && same_hecke(*runs[0], *hc),
                    "hecke under the Gpu / Cpu thread override == the reference run");
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "EngineTapeConsumerValidation: " << g.checks << " checks, " << g.fails << " failed, " << std::fixed
              << std::setprecision(2) << secs << " s\n";
    return g.fails == 0 ? 0 : 1;
}
