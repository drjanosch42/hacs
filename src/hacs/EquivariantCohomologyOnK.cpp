// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * EquivariantCohomologyOnK.cpp
 *
 * Implements EquivariantCohomologyOnKT<Elt> (Stage S13 element-width
 * generalization). See the header for the design note.
 *
 * (matrix-engine Phase 3, Stage B) The computation runs on the block model
 * (engine::BlockMat / engine::blk): constraint assembly through BlockBuilder
 * with kinds, kernels through blk::kernel_basis, the Z^q expansion and B^q
 * coboundary as two block GEMMs, W/H/P through blk::factor_basis.  There is
 * no per-cell loop, no raw mod_* arithmetic and no representation conversion
 * on the result path any more; the seven flatten/unflatten round trips of
 * audit 02 Sec. 2.2 are gone because the multi-vector IS the L1 layout.
 *
 * The pre-Stage-B code is preserved verbatim (modulo the scalar arithmetic
 * going through engine::ModCtx, bit-identical to the mod_* primitives it
 * replaces -- ModContextValidation) as the LEGACY arm, since S24 CL-L2 in
 * legacy/EquivariantCohomologyOnKLegacy.inc (OC_ENABLE_LEGACY=ON builds only):
 * it is what ENGINE_BACKEND=legacy runs and what the
 * -DENGINE_SHADOW_LEGACY=ON build compares against.
 */

#include "EquivariantCohomologyOnK.h"

#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockKernels.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/TapeElim.h"
#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/EngineConfig.h"
#include "../ModArithmetic/ModCtx.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/Kernels.h"
#include "Hecke/CoefficientRing.h"   // (slot-8 WP-3) coefficient_ring in force, the refusal rule

#include <chrono>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream> // For cerr

namespace {

using engine::BlockBuilder;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::cview;
using engine::view;

template<class Elt>
using Relation = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;

// ---------------------------------------------------------------------------
// (Phase 1, item 7.4) Vacuous-relation short-circuit -- see the header for the
// mathematical justification and the odd-n argument.
// ---------------------------------------------------------------------------
// True iff the relation constrains one simplex against itself by the identity,
// i.e. f(s) = I f(s).  Its A_Cq block row is then I - I = 0, and every one of
// its A_Zq blocks is (beta_lhs - beta_rhs) * I = 0 because beta_lhs and
// beta_rhs read the SAME simplex.  Nothing else is dropped: a relation with
// M = alpha*I, alpha != 1 (the odd-n rho_n(-I) = -I case) still contributes the
// non-zero scalar block (1 - alpha) I and is kept.
template<class Elt>
bool relation_is_vacuous(const Relation<Elt>& rel, uint32_t d_V) {
    if (rel.s_lhs_idx_in_Kq != rel.s_rhs_idx_in_Kq) return false;
    if (rel.M.get_rows() != d_V || rel.M.get_cols() != d_V) return false;
    Elt sv = 0;
    if (rel.M.is_currently_scalar(sv)) return sv == Elt(1);   // flags are exact (rescanned on mutation)
    for (uint32_t r = 0; r < d_V; ++r) {
        for (uint32_t c = 0; c < d_V; ++c) {
            if (rel.M.get_entry(r, c) != static_cast<Elt>(r == c ? 1 : 0)) return false;
        }
    }
    return true;
}

// The relations of degree q that constrain anything (item 7.4), in input order.
template<class Elt>
std::vector<const Relation<Elt>*> live_relations(const std::vector<Relation<Elt>>& rels,
                                                 uint32_t q, uint32_t d_V) {
    const bool shortcircuit = EquivariantCohomologyOptions::vacuous_relation_shortcircuit_enabled();
    std::vector<const Relation<Elt>*> live;
    for (const auto& rel : rels) {
        if (rel.q_dim != q) continue;
        if (shortcircuit && relation_is_vacuous<Elt>(rel, d_V)) continue;
        live.push_back(&rel);
    }
    return live;
}

// ---------------------------------------------------------------------------
// Block-model assembly (shared by the engine arm and by the legacy arm's
// block re-construction).  One BlockBuilder call per BLOCK, never per cell.
// ---------------------------------------------------------------------------

// Non-owning row-major DenseBuf over a FlatMatrix's dense buffer.  The caller
// keeps `m` alive for as long as the buffer is used (here: one intern call).
template<class Elt>
DenseBuf<Elt> borrow_flat(const FlatMatrixT<Elt>& m) {
    m.populate_flat_data();
    const std::vector<Elt>& fd = m.get_flat_data_unsafe();
    return DenseBuf<Elt>(const_cast<Elt*>(fd.data()), fd.empty() ? 1 : fd.size(),
                         m.get_rows(), m.get_cols(), Layout::RowMajor);
}

// The block ref of a relation matrix rho_n(g): a Scalar ref when the
// FlatMatrix already knows it is alpha*I (no materialisation), else interned
// once by content -- every relation carrying the same rho_n(g) shares the
// pool entry.
template<class Elt>
BlockRef<Elt> relation_ref(BlockBuilder<Elt>& b, const FlatMatrixT<Elt>& M) {
    Elt sv = 0;
    if (M.is_currently_scalar(sv)) return BlockRef<Elt>::scalar(sv);   // Zero when sv == 0
    const DenseBuf<Elt> tmp = borrow_flat<Elt>(M);
    return b.intern_dense(cview(tmp), Layout::RowMajor, true);
}

// A_Cq: block row r = live relation r; I at s_lhs and -M at s_rhs, or I - M on
// the diagonal when s_lhs == s_rhs.  When M = alpha*I the diagonal block is the
// Scalar (1 - alpha) I -- exactly what the legacy DATA constructor detected.
template<class Elt>
BlockMat<Elt> assemble_A_Cq(const std::shared_ptr<BlockPool<Elt>>& pool,
                            const std::vector<const Relation<Elt>*>& live,
                            uint32_t N_q, uint32_t d, Elt p) {
    const std::vector<uint32_t> row_dims(live.size(), d), col_dims(N_q, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    const Elt neg1 = b.ctx().neg(Elt(1));
    uint32_t ri = 0;
    for (const Relation<Elt>* rp : live) {
        const Relation<Elt>& rel = *rp;
        const BlockRef<Elt> M = relation_ref<Elt>(b, rel.M);
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
            b.set(ri, rel.s_lhs_idx_in_Kq, b.compose(M, Elt(1), neg1));           // I - M
        } else {
            b.set(ri, rel.s_lhs_idx_in_Kq, BlockRef<Elt>::identity());            // I
            b.set(ri, rel.s_rhs_idx_in_Kq, b.compose(M, Elt(0), neg1));           // -M
        }
        ++ri;
    }
    return b.finalize();
}

// A_Zq: block (r, j) = beta_j(s_lhs) I - beta_j(s_rhs) M, one pencil ref each
// (Scalar when beta_rhs == 0 or M is scalar; structurally absent when both
// betas vanish or the pencil is identically zero).
template<class Elt>
BlockMat<Elt> assemble_A_Zq(const std::shared_ptr<BlockPool<Elt>>& pool,
                            const std::vector<const Relation<Elt>*>& live,
                            const std::vector<FlatMatrixT<Elt>>& Z1, uint32_t N_q, uint32_t d, Elt p) {
    const uint32_t k_q = static_cast<uint32_t>(Z1.size());
    const std::vector<uint32_t> row_dims(live.size(), d), col_dims(k_q, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    uint32_t ri = 0;
    for (const Relation<Elt>* rp : live) {
        const Relation<Elt>& rel = *rp;
        const BlockRef<Elt> M = relation_ref<Elt>(b, rel.M);
        for (uint32_t j = 0; j < k_q; ++j) {
            const Elt beta_lhs = (N_q > 0) ? Z1[j].get_entry(rel.s_lhs_idx_in_Kq, 0) : Elt(0);
            const Elt beta_rhs = (N_q > 0) ? Z1[j].get_entry(rel.s_rhs_idx_in_Kq, 0) : Elt(0);
            if (beta_lhs == 0 && beta_rhs == 0) continue;
            b.set(ri, j, b.compose(M, beta_lhs, b.ctx().neg(beta_rhs)));
        }
        ++ri;
    }
    return b.finalize();
}

// (beta (x) I): the N_q x k_q block matrix with Scalar block beta_j(s) I at
// (s, j).  Z^q_eq = (beta (x) I) * ker A_Zq  is the legacy
// expand_coeffs_to_cochain_basis as ONE block GEMM.
template<class Elt>
BlockMat<Elt> assemble_beta_expansion(const std::shared_ptr<BlockPool<Elt>>& pool,
                                      const std::vector<FlatMatrixT<Elt>>& Z1, uint32_t N_q, uint32_t d, Elt p) {
    const uint32_t k_q = static_cast<uint32_t>(Z1.size());
    const std::vector<uint32_t> row_dims(N_q, d), col_dims(k_q, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    for (uint32_t j = 0; j < k_q; ++j)
        for (uint32_t s = 0; s < N_q; ++s) {
            const Elt beta = Z1[j].get_entry(s, 0);
            if (beta != 0) b.set_scalar(s, j, beta);
        }
    return b.finalize();
}

// (D_1 (x) I): the coboundary N_{q+1} x N_q with every +-1 a Scalar block.
template<class Elt>
BlockMat<Elt> assemble_coboundary_block(const std::shared_ptr<BlockPool<Elt>>& pool,
                                        const SimplicialComplex* K, uint32_t q, uint32_t d, Elt p) {
    const FlatMatrixT<Elt> D1 = K->template coboundary_operator_elt<Elt>(q, p);   // N_{q+1} x N_q
    const uint32_t N_src = K->get_num_q_simplices(q), N_dst = K->get_num_q_simplices(q + 1);
    const std::vector<uint32_t> row_dims(N_dst, d), col_dims(N_src, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    if (D1.get_rows() == N_dst && D1.get_cols() == N_src) {
        for (uint32_t i = 0; i < N_dst; ++i)
            for (uint32_t j = 0; j < N_src; ++j) {
                const Elt s = D1.get_entry(i, j);
                if (s != 0) b.set_scalar(i, j, s);
            }
    } else {
        ENGINE_CHECK_MSG(D1.get_rows() * D1.get_cols() == 0,
                         "coboundary operator shape disagrees with the simplex counts");
    }
    return b.finalize();
}

// (slot-8 WP-3) The Z/ell^k twins of assemble_A_Cq / assemble_coboundary_block:
// the same blocks, on a BlockBuilder that CARRIES the composite context
// (BlockMat.h's ModCtx overload, ENGINE_API.md Sec. 13.7), so the BlockMat
// declares Z/ell^k and a field-only block elimination on it refuses (Sec.
// 13.3).  The field functions above are untouched.
template<class Elt>
BlockMat<Elt> assemble_A_Cq_ring(const std::shared_ptr<BlockPool<Elt>>& pool,
                                 const std::vector<const Relation<Elt>*>& live,
                                 uint32_t N_q, uint32_t d, const engine::ModCtx<Elt>& ctx) {
    const std::vector<uint32_t> row_dims(live.size(), d), col_dims(N_q, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, ctx);
    const Elt neg1 = b.ctx().neg(Elt(1));
    uint32_t ri = 0;
    for (const Relation<Elt>* rp : live) {
        const Relation<Elt>& rel = *rp;
        ENGINE_CHECK_MSG(rel.M.get_p() == ctx.p, "EquivariantCohomologyOnK (Z/ell^k): relation matrix modulus != ell^k");
        const BlockRef<Elt> M = relation_ref<Elt>(b, rel.M);
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
            b.set(ri, rel.s_lhs_idx_in_Kq, b.compose(M, Elt(1), neg1));           // I - M
        } else {
            b.set(ri, rel.s_lhs_idx_in_Kq, BlockRef<Elt>::identity());            // I
            b.set(ri, rel.s_rhs_idx_in_Kq, b.compose(M, Elt(0), neg1));           // -M
        }
        ++ri;
    }
    return b.finalize();
}
template<class Elt>
BlockMat<Elt> assemble_coboundary_block_ring(const std::shared_ptr<BlockPool<Elt>>& pool,
                                             const SimplicialComplex* K, uint32_t q, uint32_t d,
                                             const engine::ModCtx<Elt>& ctx) {
    const FlatMatrixT<Elt> D1 = K->template coboundary_operator_elt<Elt>(q, ctx.p);   // +-1 incidences mod ell^k
    const uint32_t N_src = K->get_num_q_simplices(q), N_dst = K->get_num_q_simplices(q + 1);
    const std::vector<uint32_t> row_dims(N_dst, d), col_dims(N_src, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, ctx);
    if (D1.get_rows() == N_dst && D1.get_cols() == N_src) {
        for (uint32_t i = 0; i < N_dst; ++i)
            for (uint32_t j = 0; j < N_src; ++j) {
                const Elt s = D1.get_entry(i, j);
                if (s != 0) b.set_scalar(i, j, s);
            }
    } else {
        ENGINE_CHECK_MSG(D1.get_rows() * D1.get_cols() == 0,
                         "coboundary operator shape disagrees with the simplex counts");
    }
    return b.finalize();
}

// Y = A * X for a multi-vector X, as a fresh column-major multi-vector over
// A's row partition (structural-zero detection OFF so the result stays one
// contiguous window -- values are unaffected either way).
template<class Elt>
BlockMat<Elt> block_times_multivector(const BlockMat<Elt>& A, const BlockMat<Elt>& X) {
    DenseBuf<Elt> scratch;
    const DenseView<const Elt> Xv = engine::blk::multivector_view<Elt>(X, scratch);
    DenseBuf<Elt> Y(A.rows(), Xv.cols(), Layout::ColMajor);
    engine::blk::gemm_dense<Elt>(A, Xv, view(Y));
    return BlockMat<Elt>::from_flat(A.pool(), std::move(Y), A.row_dims(), A.p(), /*detect=*/false);
}

// ---------------------------------------------------------------------------
// Materialisation of the legacy FlatMatrix shapes from the block objects
// (the compatibility accessors).  Column t of a multi-vector in L1 order IS
// the row-major N_q x d_V cochain, so the copy is a straight column read.
// ---------------------------------------------------------------------------
template<class Elt>
std::vector<FlatMatrixT<Elt>> columns_as_cochains(const BlockMat<Elt>& mv, uint32_t N_q, uint32_t d, Elt p) {
    std::vector<FlatMatrixT<Elt>> out;
    DenseBuf<Elt> scratch;
    const DenseView<const Elt> V = engine::blk::multivector_view<Elt>(mv, scratch);
    const size_t total = static_cast<size_t>(N_q) * d;
    ENGINE_CHECK(V.rows() == total || V.cols() == 0);
    out.reserve(V.cols());
    const auto st = V.strides();
    for (uint32_t t = 0; t < V.cols(); ++t) {
        if (total == 0) { out.push_back(FlatMatrixT<Elt>::CreateZero(N_q, d, p)); continue; }
        std::vector<Elt> buf(total);
        const Elt* col = st.base + static_cast<size_t>(t) * st.cs;
        if (st.rs == 1) std::memcpy(buf.data(), col, total * sizeof(Elt));
        else for (size_t i = 0; i < total; ++i) buf[i] = col[i * st.rs];
        out.emplace_back(N_q, d, p, buf);   // DATA constructor: same flags as legacy
    }
    return out;
}

// The legacy factor_basis_block projector shape conventions: 0 x 0 when there
// were no generators, CreateZero(dim_H, a) when dim_H or a is 0, else the
// row-major data.
template<class Elt>
FlatMatrixT<Elt> projector_as_flat(const BlockMat<Elt>& P, bool had_generators, uint32_t ambient, Elt p) {
    if (!had_generators) return FlatMatrixT<Elt>(0, 0, p);
    const uint32_t dim_H = P.rows();
    ENGINE_CHECK(P.cols() == ambient);
    if (dim_H == 0 || ambient == 0) return FlatMatrixT<Elt>::CreateZero(dim_H, ambient, p);
    const DenseBuf<Elt> D = P.to_dense(Layout::RowMajor);
    std::vector<Elt> buf(static_cast<size_t>(dim_H) * ambient);
    for (uint32_t i = 0; i < dim_H; ++i)
        std::memcpy(buf.data() + static_cast<size_t>(i) * ambient, D.data() + static_cast<size_t>(i) * D.ld,
                    ambient * sizeof(Elt));
    return FlatMatrixT<Elt>(dim_H, ambient, p, buf);
}

} // namespace

// ===========================================================================
// Options
// ===========================================================================
// (matrix-engine Phase 4, Stage B) Every option is THREAD-LOCAL: pipelines
// running concurrently (HeckeOperatorFamily axis (d), the campaign) cannot
// see each other's scoped settings.  The engine's host fan-out
// (engine::tape::run_concurrently) copies the launching thread's values into
// its task threads through the propagator registered below, so a scope set
// on the caller applies to every degree / pipeline it launches.
namespace EquivariantCohomologyOptions {
    namespace {
        thread_local bool g_retain_intermediates = true;
        thread_local bool g_vacuous_shortcircuit = true;
#if OC_ENABLE_LEGACY
        thread_local bool g_force_legacy_path = false;
        const bool g_registered = [] {
            engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
                const bool retain = g_retain_intermediates, vac = g_vacuous_shortcircuit, legacy = g_force_legacy_path;
                return [retain, vac, legacy]() -> engine::tape::ThreadContextGuard {
                    g_retain_intermediates = retain; g_vacuous_shortcircuit = vac; g_force_legacy_path = legacy;
                    return nullptr;
                };
            });
            return true;
        }();
#else
        const bool g_registered = [] {
            engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
                const bool retain = g_retain_intermediates, vac = g_vacuous_shortcircuit;
                return [retain, vac]() -> engine::tape::ThreadContextGuard {
                    g_retain_intermediates = retain; g_vacuous_shortcircuit = vac;
                    return nullptr;
                };
            });
            return true;
        }();
#endif
    }
    bool retain_intermediates() { return g_retain_intermediates; }
    void set_retain_intermediates(bool retain) { g_retain_intermediates = retain; }
    bool vacuous_relation_shortcircuit_enabled() { return g_vacuous_shortcircuit; }
    void set_vacuous_relation_shortcircuit(bool enabled) { g_vacuous_shortcircuit = enabled; }
#if OC_ENABLE_LEGACY
    bool legacy_cohomology_path_forced() { return g_force_legacy_path; }
    void set_legacy_cohomology_path_forced(bool forced) { g_force_legacy_path = forced; }
#endif
}

// ===========================================================================
// Construction
// ===========================================================================
template<class Elt>
EquivariantCohomologyOnKT<Elt>::EquivariantCohomologyOnKT(const LocalSystemSimplicialComplexT<Elt>* local_system_complex)
    : lssc(local_system_complex), cohomology_computed(false) {
    if (!lssc) {
        throw std::runtime_error("EquivariantCohomologyOnK: LocalSystemSimplicialComplex pointer is null.");
    }
    prime = lssc->get_prime();
    representation_dim = lssc->get_representation_dim();

    // (slot-8 WP-3) The coefficient ring, read ONCE ([E] sec:ring (a)).  At the
    // default nothing below changes; over Z/ell^k the module route needs no
    // trivial-cocycle bases (they feed the field route's A_Zq only).
    init_coefficient_ring();
    if (ring_k_ >= 2) return;

    SimplicialComplex* non_const_K = const_cast<SimplicialComplex*>(lssc->get_base_K_complex());
    non_const_K->template compute_trivial_cocycle_bases_elt<Elt>(prime);
}

// (slot-8 WP-3) See the header.  The composite route is selected by the
// setting alone; the local system must BE over Z/ell^k (its modulus the exact
// k-th power of the cell's ell), else -- as for every existing construction
// site, whose local systems are over F_ell -- the constructor refuses by name.
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::init_coefficient_ring() {
    ring_k_ = hecke::CoefficientRingK("EquivariantCohomologyOnK");
    if (ring_k_ == 1) {
        // (slot-8 WP-3r, review m-6) F_ell in force: a local system whose
        // modulus is a prime power ell^j, j >= 2 (as hecke::GenerateSL2ZRelationsRing
        // builds) would run the FIELD route on ModCtx(ell^j) -- a declared field
        // with a composite modulus (ENGINE_API.md Sec. 13.1).  Refuse it.  A guard
        // never taken on a prime modulus (Sec. 13.8 (2)); O(63 log p).
        for (uint32_t j = 2; j <= 63; ++j) {
            const uint64_t root = hecke::ExactKthRoot(static_cast<uint64_t>(prime), j);
            if (root != 0)
                throw std::logic_error(
                    "EquivariantCohomologyOnK: the local system's modulus " +
                    std::to_string(static_cast<uint64_t>(prime)) + " = " + std::to_string(root) + "^" +
                    std::to_string(j) +
                    " is a prime power, but coefficient_ring = F_ell is in force -- the field route would run on a "
                    "composite modulus (set coefficient_ring = Z/ell^" + std::to_string(j) + "; ENGINE_API.md Sec. 13.1)");
        }
        return;
    }
    // (slot-11 WP-12, F-2) The ring travels as (ell, k) when the producer
    // declared it (ell^k == prime was checked at the declaration); otherwise
    // ell is re-derived from the modulus as before.  Both give the same ell
    // on every declared system (ExactKthRoot's root is unique); a declaration
    // of another k is the same refusal as a modulus that is not ell^k.
    const uint64_t ell = lssc->has_declared_coefficient_ring()
                             ? (lssc->declared_ring_k() == ring_k_ ? lssc->declared_ring_ell() : 0)
                             : hecke::ExactKthRoot(static_cast<uint64_t>(prime), ring_k_);
    ENGINE_DEBUG_CHECK(!lssc->has_declared_coefficient_ring() || ell == 0 ||
                       ell == hecke::ExactKthRoot(static_cast<uint64_t>(prime), ring_k_));
    if (ell == 0) {
        hecke::RefuseCoefficientRing(
            "EquivariantCohomologyOnK", ring_k_,
            " on a local system over modulus " + std::to_string(static_cast<uint64_t>(prime)) +
                " (not ell^" + std::to_string(ring_k_) +
                "; the model over Z/ell^k is built by hecke::GenerateSL2ZRelationsRing)");
    }
    // (slot-9 WP-3; decision 4 (A)) The Bredon policy: this object -- and no
    // other caller of CoefficientRingContext -- honours ell in {2, 3}, by the
    // flat-coordinate route (compute_cohomology_flat_module()).
    ring_ctx_ = hecke::CoefficientRingContext<Elt>("EquivariantCohomologyOnK", ell, ring_k_,
                                                   hecke::SmallEllPolicy::Bredon);
    flat_route_ = hecke::IsBredonPrime(ell);
}

// (hecke-extensions slot 6, WP-5) The relative sub-complex constructor: the
// header note.  Every check is O(total simplices) (ENGINE_CHECK).
template<class Elt>
EquivariantCohomologyOnKT<Elt>::EquivariantCohomologyOnKT(const LocalSystemSimplicialComplexT<Elt>* local_system_complex,
                                                          const std::vector<std::vector<uint32_t>>& vanishing_simplices)
    : EquivariantCohomologyOnKT(local_system_complex) {
    // (slot-8 WP-3) The relative sub-complex is the cone's route 3.  (slot-10
    // WP-3, decision 3 (A)) Honoured over Z/ell^k at ell >= 5 through the
    // MODULE route (compute_cohomology_module() reads effective_relations(),
    // so the zero relations below constrain it exactly as the field arms; the
    // forced-zero set of a pair makes C^q_rel = ker r a direct summand of the
    // free C^q_eq, hence free, which that route asserts).  At ell in {2, 3}
    // (the Bredon object's flat-coordinate route) it keeps refusing by name:
    // the cone is not honoured there.
    if (ring_k_ >= 2 && flat_route_)
        hecke::RefuseCoefficientRing("EquivariantCohomologyOnK (relative sub-complex)", ring_k_,
                                     " at ell = " + std::to_string(static_cast<uint64_t>(ring_ctx_.ell)) +
                                         " (the cone and its route 3 are not honoured at ell in {2, 3})");
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const uint32_t max_dim_K = K->get_dimension();
    const uint32_t nq = (max_dim_K == static_cast<uint32_t>(-1)) ? 0 : max_dim_K + 1;
    ENGINE_CHECK_MSG(vanishing_simplices.size() <= nq,
                     "EquivariantCohomologyOnK (relative): forced-zero sets given above K's top degree");
    relative_ = true;
    vanishing_ = vanishing_simplices;
    relative_relations_ = lssc->get_user_defined_simplex_relations();
    const FlatMatrixT<Elt> zero = FlatMatrixT<Elt>::CreateZero(representation_dim, representation_dim, prime);
    for (uint32_t q = 0; q < vanishing_.size(); ++q) {
        const uint32_t N_q = K->get_num_q_simplices(q);
        for (size_t i = 0; i < vanishing_[q].size(); ++i) {
            const uint32_t s = vanishing_[q][i];
            ENGINE_CHECK_MSG(s < N_q, "EquivariantCohomologyOnK (relative): a forced-zero simplex index is out of range");
            ENGINE_CHECK_MSG(i == 0 || vanishing_[q][i - 1] < s,
                             "EquivariantCohomologyOnK (relative): forced-zero simplices must be sorted and distinct");
            // f(s) = 0 * f(s), i.e. f(s) = 0: A_C block I, A_Z pencil beta(s) I.
            relative_relations_.emplace_back(q, s, s, zero, "relative:vanish");
        }
    }
}

template<class Elt>
const std::vector<typename EquivariantCohomologyOnKT<Elt>::SimplexRelationT>&
EquivariantCohomologyOnKT<Elt>::effective_relations() const {
    return relative_ ? relative_relations_ : lssc->get_user_defined_simplex_relations();
}

// Protected constructor for derived classes
template<class Elt>
EquivariantCohomologyOnKT<Elt>::EquivariantCohomologyOnKT(Elt p, uint32_t rep_d)
    : lssc(nullptr), prime(p), representation_dim(rep_d), cohomology_computed(false) {
    // (slot-8 WP-3) The derived-class base (the mapping cone): refused under a
    // non-default ring, with the cone ([E] sec:ring (b4)).
    hecke::RequireFieldCoefficientRing("EquivariantCohomologyOnK (cone base)");
}

// ===========================================================================
// compute_cohomology(): dispatch between the two arms
// ===========================================================================
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::compute_cohomology() {
    if (cohomology_computed) return;
    if (ring_k_ >= 2) {                     // (slot-8 WP-3) Z/ell^k: the module route
        if (flat_route_) compute_cohomology_flat_module();   // (slot-9 WP-3) ell in {2, 3}
        else compute_cohomology_module();
        cohomology_computed = true;
        return;
    }
    fm_ = FlatResult();
    intermediates_retained_ = EquivariantCohomologyOptions::retain_intermediates();

#if OC_ENABLE_LEGACY
    const bool legacy = engine::backend_is_legacy() ||
                        EquivariantCohomologyOptions::legacy_cohomology_path_forced();
    if (legacy) {
        // The legacy arm is the pre-Phase-4 reference: it retains everything.
        intermediates_retained_ = true;
        FlatResult L;
        compute_cohomology_legacy(L);
        adopt_legacy_result(std::move(L));
        C_eq_dims_.assign(C_eq_block_.size(), 0);
        for (size_t q = 0; q < C_eq_block_.size(); ++q) C_eq_dims_[q] = C_eq_block_[q].cols();
        cohomology_computed = true;
        return;
    }
#endif
    compute_cohomology_block();
#ifdef ENGINE_SHADOW_LEGACY
    shadow_compare_with_legacy();   // legacy/EquivariantCohomologyOnKShadow.inc
#endif
    cohomology_computed = true;
}

// ===========================================================================
// The block-model arm -- on the tape engine (matrix-engine Phase 4, Stage B)
// ===========================================================================
//
// Every heavy step is a tape program executed on the consumer pool
// (EngineRuntime.h): the three kernels and the two factor_basis calls go
// through the panel-driven elimination driver (TapeElim.h, per-panel
// programs, Schur updates tiled over threads), the two block GEMMs
// (Z^q_eq = (beta (x) I) ker A_Zq, B^q_eq = (D_1 (x) I) W^{q-1}) are recorded
// as gemm_dense programs and served from the program cache by shape signature
// (grid, pattern hash of the Scalar-block operator, multi-vector width, d_V,
// regime).  The assembly itself stays BlockBuilder work.
//
// Parallel axis (c): step 1 -- A_Cq, C^q_eq, A_Zq, Z^q_eq -- is independent
// across the degrees q and runs as one host task per q (run_concurrently: a
// thread per task, each with its own Executor, all sharing the pool and the
// now-synchronised BlockPool).  Steps 2 and 3 (W^{q-1}, B^q, H^q, P_q) read
// ONLY step-1 objects -- of q-1 and of q -- so once step 1 has completed for
// every degree the per-degree chains W^{q-1} -> B^q -> (H^q, P_q) are
// independent of each other as well (audit 02 Sec. 4(c)'s dependency is on
// step 1 of q-1, not on step 2/3 of q-1) and run as a second fan-out.  Every
// object is an exact, unique mathematical object (the D5b bases of the
// unique RREF, exact GEMMs), so the schedule cannot move a bit -- golden
// tier B pins exactly that.  ENGINE_HOST_THREADS=0 (or ENGINE_THREADS=1)
// makes both fan-outs sequential in q.
//
// Memory (E10): the step-1 temporaries (ker A_Zq, the beta expansion) die at
// the end of their task; the factor_basis workspaces die inside the driver.
// The seven stored objects are the published accessors' storage.

namespace {

// Y = A * X for a multi-vector X as a tape program (record always, compile
// on a program-cache miss -- EngineRuntime.h run_cached), the result a fresh
// column-major multi-vector over A's row partition (structural-zero
// detection OFF so it stays one contiguous window).
template<class Elt>
DenseBuf<Elt> tape_block_times_multivector_buf(engine::tape::Executor<Elt>& ex, const char* tag, uint32_t q,
                                               const BlockMat<Elt>& A, const BlockMat<Elt>& X) {
    using namespace engine::tape;
    DenseBuf<Elt> scratch;
    const DenseView<const Elt> Xv = engine::blk::multivector_view<Elt>(X, scratch);
    const uint32_t w = Xv.cols();
    DenseBuf<Elt> Y(A.rows(), w, Layout::ColMajor);
    if (A.rows() > 0 && w > 0) {
        Recorder<Elt> rec(A.ctx());
        const BufId bX = rec.input(Xv);
        const BufId bY = rec.output(view(Y));
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), engine::kernels::GemmOp::Set);
        Program<Elt> prog = rec.finish();
        ShapeKey key = make_key<Elt>(tag, A.ctx(), A.row_dim(0), q);
        key.row_dims = A.row_dims();
        key.col_dims = A.col_dims();
        key.pattern_hash = fnv_mix(fnv_mix(pattern_hash_of<Elt>(A), w), A.cols());
        run_cached<Elt>(ex, key, std::move(prog), rec.bindings());
    }
    return Y;
}
template<class Elt>
BlockMat<Elt> tape_block_times_multivector(engine::tape::Executor<Elt>& ex, const char* tag, uint32_t q,
                                           const BlockMat<Elt>& A, const BlockMat<Elt>& X) {
    DenseBuf<Elt> Y = tape_block_times_multivector_buf<Elt>(ex, tag, q, A, X);
    return BlockMat<Elt>::from_flat(A.pool(), std::move(Y), A.row_dims(), A.p(), /*detect=*/false);
}

// ENGINE_TRACE_RSS=1: one line per step of compute_cohomology_block() with the
// wall time since the previous mark and the process's VmRSS / VmHWM (MB).
struct StepTrace {
    bool on = std::getenv("ENGINE_TRACE_RSS") != nullptr;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    static double status_mb(const char* key) {
        FILE* f = std::fopen("/proc/self/status", "r");
        if (!f) return -1.0;
        char line[256]; double out = -1.0;
        while (std::fgets(line, sizeof line, f))
            if (std::strncmp(line, key, std::strlen(key)) == 0) { out = std::atof(line + std::strlen(key)) / 1024.0; break; }
        std::fclose(f);
        return out;
    }
    void mark(const std::string& what) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        std::cout << "      [cohom] " << what << ": " << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(now - t).count() << " s, VmRSS=" << std::setprecision(1)
                  << status_mb("VmRSS:") << " MB VmHWM=" << status_mb("VmHWM:") << " MB\n";
        t = now;
    }
};

} // namespace

template<class Elt>
void EquivariantCohomologyOnKT<Elt>::compute_cohomology_block() {
    using namespace engine::tape;
    StepTrace trace;
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const uint32_t max_dim_K = K->get_dimension();
    const uint32_t nq = (max_dim_K == static_cast<uint32_t>(-1)) ? 0 : max_dim_K + 1;
    const uint32_t d = representation_dim;
    const Elt p = prime;
    const auto& rels = effective_relations();

    pool_ = std::make_shared<BlockPool>();
    A_C_block_.assign(nq, BlockMat());
    A_Z_block_.assign(nq, BlockMat());
    C_eq_block_.assign(nq, BlockMat());
    Z_eq_block_.assign(nq, BlockMat());
    B_eq_block_.assign(nq, BlockMat());
    H_eq_block_.assign(nq, BlockMat());
    P_block_.assign(nq, BlockMat());
    H_eq_dimensions.assign(nq, 0);

    // === Step 1, per degree, concurrently (axis (c)) ======================
    // === C^q_eq = ker A_Cq  (an empty A_Cq -- no live relation -- has rank 0
    // === and its kernel basis is the identity, as before);
    // === Z^q_eq = (beta (x) I) * ker A_Zq.
    std::vector<std::function<void()>> step1;
    for (uint32_t q = 0; q < nq; ++q) {
        step1.emplace_back([this, K, &rels, q, d, p]() {
            Executor<Elt> ex(&consumer_pool());
            const uint32_t N_q = K->get_num_q_simplices(q);
            const std::vector<const Relation<Elt>*> live = live_relations<Elt>(rels, q, d);
            StepTrace tr;
            A_C_block_[q] = assemble_A_Cq<Elt>(pool_, live, N_q, d, p);
            C_eq_block_[q] = kernel_basis<Elt>(ex, A_C_block_[q], Layout::ColMajor);
            tr.mark("  q=" + std::to_string(q) + " A_C " + std::to_string(A_C_block_[q].rows()) + "x" + std::to_string(A_C_block_[q].cols()) +
                    " -> C_eq " + std::to_string(C_eq_block_[q].rows()) + "x" + std::to_string(C_eq_block_[q].cols()));

            const auto& Z1 = K->template get_trivial_cocycle_basis_Z1_elt<Elt>(q);
            A_Z_block_[q] = assemble_A_Zq<Elt>(pool_, live, Z1, N_q, d, p);
            const BlockMat V = kernel_basis<Elt>(ex, A_Z_block_[q], Layout::ColMajor);      // (k_q d) x dim Z
            tr.mark("  q=" + std::to_string(q) + " A_Z " + std::to_string(A_Z_block_[q].rows()) + "x" + std::to_string(A_Z_block_[q].cols()) +
                    " -> V " + std::to_string(V.rows()) + "x" + std::to_string(V.cols()));
            const BlockMat E = assemble_beta_expansion<Elt>(pool_, Z1, N_q, d, p);          // dim_q x (k_q d)
            Z_eq_block_[q] = tape_block_times_multivector<Elt>(ex, "cohom.Zeq", q, E, V);
            tr.mark("  q=" + std::to_string(q) + " Z_eq " + std::to_string(Z_eq_block_[q].rows()) + "x" + std::to_string(Z_eq_block_[q].cols()));
        });
    }
    // Memory-lean mode is memory-first at the consumer level: the degrees run
    // one after the other (each still on the whole pool), so only one
    // executor's workspace is live at a time.  Default mode is speed-first.
    if (intermediates_retained_) run_concurrently(step1);
    else for (auto& task : step1) task();
    trace.mark("step 1 (A_C, C_eq, A_Z, Z_eq; all q)");
    C_eq_dims_.assign(nq, 0);
    for (uint32_t q = 0; q < nq; ++q) C_eq_dims_[q] = C_eq_block_[q].cols();
    if (!intermediates_retained_) {
        for (uint32_t q = 0; q < nq; ++q) { A_C_block_[q] = BlockMat(); A_Z_block_[q] = BlockMat(); }   // no later reader
        if (nq > 0) C_eq_block_[nq - 1] = BlockMat();   // C^{top} feeds only the (off) Theta_eq certification
    }

    // === Steps 2 and 3: one independent chain per degree ==================
    for (uint32_t q = 0; q + 1 < nq; ++q) (void)K->template coboundary_operator_elt<Elt>(q, p);   // prime the lazy index
    std::vector<std::function<void()>> chains;
    for (uint32_t q = 0; q < nq; ++q) chains.emplace_back([this, K, q, d, p]() {
        Executor<Elt> ex(&consumer_pool());
        const uint32_t N_q = K->get_num_q_simplices(q);
        const std::vector<uint32_t> dims_q(N_q, d);

        // === B^q_eq = image of D_1 (x) I on C^{q-1}_eq =========================
        // The pre-Phase-4 route was B^q = (D (x) I) . W^{q-1} with
        // W^{q-1} = C^{q-1}/Z^{q-1} = the C^{q-1} columns at the pivots of
        // [Z^{q-1} | C^{q-1}] (D5b).  A column c_j of C^{q-1} is such a pivot
        // iff c_j is not in span(Z^{q-1} + {c_i : i < j}); since ker(D) on
        // C^{q-1}_eq is exactly Z^{q-1}_eq, that holds iff D c_j is not in
        // span{D c_i : i < j}, i.e. iff column j of D . C^{q-1} is a pivot
        // column of D . C^{q-1}.  So B^q = the pivot columns of D . C^{q-1}
        // -- the D5b image basis of a dim_q x dim C^{q-1} multi-vector --
        // the SAME columns as before, entry for entry (each column of the
        // product depends on its own C column only), from an elimination of
        // dim_q x dim C^{q-1} instead of dim_{q-1} x (dim Z + dim C), and
        // without Z^{q-1} (memory profile 05 Sec. 2: the largest transient).
        // CohomologyBlockModelValidation / golden tier B pin B^q entrywise.
        StepTrace tr;
        if (q > 0) {
            const BlockMat Dblk = assemble_coboundary_block<Elt>(pool_, K, q - 1, d, p);          // dim_q x dim_{q-1}
            const DenseBuf<Elt> DC = tape_block_times_multivector_buf<Elt>(ex, "cohom.DC", q, Dblk, C_eq_block_[q - 1]);
            tr.mark("  q=" + std::to_string(q) + " D.C " + std::to_string(DC.rows) + "x" + std::to_string(DC.cols));
            if (!intermediates_retained_) C_eq_block_[q - 1] = BlockMat();   // its last reader (lean: chains are sequential)
            B_eq_block_[q] = image_basis<Elt>(ex, pool_, p, dims_q, cview(DC), 256);
            tr.mark("  q=" + std::to_string(q) + " B_eq = image_basis -> " + std::to_string(B_eq_block_[q].rows()) + "x" + std::to_string(B_eq_block_[q].cols()));
        } else {
            B_eq_block_[q] = BlockMat::zeros(pool_, dims_q, std::vector<uint32_t>{0}, p);
        }

        // === H^q_eq = Z^q_eq / B^q_eq and the projector P_q ==================
        engine::blk::FactorBasisResult<Elt> H;
        {
            DenseBuf<Elt> sB, sZ;
            const DenseView<const Elt> Bv = engine::blk::multivector_view<Elt>(B_eq_block_[q], sB);
            const DenseView<const Elt> Zv = engine::blk::multivector_view<Elt>(Z_eq_block_[q], sZ);
            H = factor_basis<Elt>(ex, pool_, p, dims_q, Bv, Zv);
        }
        tr.mark("  q=" + std::to_string(q) + " H, P = factor_basis");
        const uint32_t dim_H = static_cast<uint32_t>(H.pivots.size());
        H_eq_dimensions[q] = dim_H;
        H_eq_block_[q] = std::move(H.basis);
        ENGINE_CHECK(H.projector.rows == dim_H && H.projector.cols == N_q * d);
        P_block_[q] = BlockMat::from_dense(pool_, std::move(H.projector), std::vector<uint32_t>{dim_H}, dims_q, p,
                                           /*detect=*/true);
    });
    if (intermediates_retained_) {
        run_concurrently(chains);
        trace.mark("chains (B, H, P; all q concurrently)");
    } else {
        // === Memory-lean mode: the chains run one at a time, in DESCENDING q,
        // === and every object is released at its last reader (memory profile
        // === 05 Sec. 4): chain q reads C^{q-1} (B^q), Z^q and B^q (H^q) --
        // === nothing outside this TU reads C^q (Theta_eq certification is
        // === off with the intermediates, HeckeOperatorOnK guards it), Z^q,
        // === B^q, A_C, A_Z.  Descending order frees the largest object,
        // === C^{q-1}, as early as possible.  The values do not depend on the
        // === order (each chain reads only step-1 objects).
        for (uint32_t qq = nq; qq-- > 0;) {
            chains[qq]();
            trace.mark("chain q=" + std::to_string(qq) + " (B, H, P)");
            Z_eq_block_[qq] = BlockMat();
            B_eq_block_[qq] = BlockMat();
        }
    }
}

// ===========================================================================
// Accessors
// ===========================================================================
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::check_degree(uint32_t q, const char* what) const {
    if (ring_k_ >= 2) require_field(what);
    if (!cohomology_computed) throw std::runtime_error(std::string(what) + ": Cohomology not computed.");
    if (q >= H_eq_dimensions.size())
        throw std::runtime_error(std::string(what) + ": degree " + std::to_string(q) + " out of range.");
}

template<class Elt>
const std::shared_ptr<typename EquivariantCohomologyOnKT<Elt>::BlockPool>&
EquivariantCohomologyOnKT<Elt>::block_pool() const {
    if (!cohomology_computed) throw std::runtime_error("block_pool: Cohomology not computed.");
    return pool_;
}
template<class Elt>
std::vector<uint32_t> EquivariantCohomologyOnKT<Elt>::block_row_dims(uint32_t q) const {
    check_degree(q, "block_row_dims");
    return std::vector<uint32_t>(lssc->get_base_K_complex()->get_num_q_simplices(q), representation_dim);
}
namespace {
void check_retained(bool retained, const char* what) {
    if (!retained)
        throw std::runtime_error(std::string(what) + ": not retained (EquivariantCohomologyOptions::retain_intermediates was off "
                                 "when compute_cohomology() ran).");
}
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::A_C_block(uint32_t q) const {
    check_degree(q, "A_C_block"); check_retained(intermediates_retained_, "A_C_block"); return A_C_block_[q];
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::A_Z_block(uint32_t q) const {
    check_degree(q, "A_Z_block"); check_retained(intermediates_retained_, "A_Z_block"); return A_Z_block_[q];
}
template<class Elt>
uint32_t EquivariantCohomologyOnKT<Elt>::C_eq_dimension(uint32_t q) const {
    check_degree(q, "C_eq_dimension"); return C_eq_dims_[q];
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::C_eq_block(uint32_t q) const {
    check_degree(q, "C_eq_block"); check_retained(intermediates_retained_, "C_eq_block"); return C_eq_block_[q];
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::Z_eq_block(uint32_t q) const {
    check_degree(q, "Z_eq_block"); check_retained(intermediates_retained_, "Z_eq_block"); return Z_eq_block_[q];
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::B_eq_block(uint32_t q) const {
    check_degree(q, "B_eq_block"); check_retained(intermediates_retained_, "B_eq_block"); return B_eq_block_[q];
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::H_eq_block(uint32_t q) const {
    check_degree(q, "H_eq_block"); return H_eq_block_[q];
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::BlockMat& EquivariantCohomologyOnKT<Elt>::P_block(uint32_t q) const {
    check_degree(q, "P_block"); return P_block_[q];
}

// The FlatMatrix shapes, built once from the block objects.
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::ensure_fm_cache() const {
    if (ring_k_ >= 2) require_field("the FlatMatrix basis lists");
    if (fm_.built) return;
    check_retained(intermediates_retained_, "the FlatMatrix basis lists");
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const uint32_t nq = static_cast<uint32_t>(H_eq_dimensions.size());
    const uint32_t d = representation_dim;
    fm_.C.assign(nq, {}); fm_.Z.assign(nq, {}); fm_.B.assign(nq, {}); fm_.H.assign(nq, {});
    fm_.P.assign(nq, FlatMatrixT<Elt>(0, 0, prime));
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t N_q = K->get_num_q_simplices(q);
        fm_.C[q] = columns_as_cochains<Elt>(C_eq_block_[q], N_q, d, prime);
        fm_.Z[q] = columns_as_cochains<Elt>(Z_eq_block_[q], N_q, d, prime);
        fm_.B[q] = columns_as_cochains<Elt>(B_eq_block_[q], N_q, d, prime);
        fm_.H[q] = columns_as_cochains<Elt>(H_eq_block_[q], N_q, d, prime);
        fm_.P[q] = projector_as_flat<Elt>(P_block_[q], Z_eq_block_[q].cols() != 0, N_q * d, prime);
    }
    fm_.dims = H_eq_dimensions;
    fm_.built = true;
}

template<class Elt>
const std::vector<std::vector<FlatMatrixT<Elt>>>& EquivariantCohomologyOnKT<Elt>::get_Z_eq_bases_fm() const {
    if (!cohomology_computed) throw std::runtime_error("Z_eq_bases: Cohomology not computed.");
    ensure_fm_cache();
    return fm_.Z;
}
template<class Elt>
const std::vector<std::vector<FlatMatrixT<Elt>>>& EquivariantCohomologyOnKT<Elt>::get_B_eq_bases_fm() const {
    if (!cohomology_computed) throw std::runtime_error("B_eq_bases: Cohomology not computed.");
    ensure_fm_cache();
    return fm_.B;
}
template<class Elt>
const std::vector<std::vector<FlatMatrixT<Elt>>>& EquivariantCohomologyOnKT<Elt>::get_H_eq_bases_fm() const {
    if (!cohomology_computed) throw std::runtime_error("H_eq_bases: Cohomology not computed.");
    ensure_fm_cache();
    return fm_.H;
}
template<class Elt>
const std::vector<std::vector<FlatMatrixT<Elt>>>& EquivariantCohomologyOnKT<Elt>::get_C_eq_bases_fm() const {
    if (!cohomology_computed) throw std::runtime_error("C_eq_bases: Cohomology not computed.");
    ensure_fm_cache();
    return fm_.C;
}
template<class Elt>
const std::vector<FlatMatrixT<Elt>>& EquivariantCohomologyOnKT<Elt>::get_H_eq_projectors_from_C_eq_flat_fm() const {
    if (!cohomology_computed) {
        throw std::runtime_error("get_H_eq_projectors_from_C_eq_flat: Cohomology not computed.");
    }
    ensure_fm_cache();
    return fm_.P;
}

template<class Elt>
const std::vector<uint32_t>& EquivariantCohomologyOnKT<Elt>::get_H_eq_dimensions() const {
    if (ring_k_ >= 2) require_field("get_H_eq_dimensions");
    if (!cohomology_computed) throw std::runtime_error("H_eq_dimensions: Cohomology not computed.");
    return H_eq_dimensions;
}

// ===========================================================================
// (hecke-extensions slot 8, WP-3; E8 item 3) THE Z/ell^k MODULE ROUTE -- the
// header note.  Dense Sec. 13.6 entry points on the composite context; the
// block model carries the context (the A_Cq and coboundary BlockMats), the
// Howell / Smith work is dense (Sec. 13.7: no block or tape Howell form).
// ===========================================================================
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::require_field(const char* what) const {
    throw std::logic_error(std::string(what) + ": a field (F_ell) accessor, but this EquivariantCohomologyOnK is over " +
                           hecke::CoefficientRingText(ring_k_) +
                           " -- over Z/ell^k there is no dim; use H_lengths() / H_invariant_factors() / "
                           "module_degree() ([E] sec:ring (b3))");
}

template<class Elt>
void EquivariantCohomologyOnKT<Elt>::require_module(const char* what, uint32_t q) const {
    if (ring_k_ < 2)
        throw std::logic_error(std::string(what) + ": a Z/ell^k (module) accessor, but this EquivariantCohomologyOnK "
                               "is over F_ell");
    if (!cohomology_computed) throw std::runtime_error(std::string(what) + ": Cohomology not computed.");
    if (q >= module_.size())
        throw std::runtime_error(std::string(what) + ": degree " + std::to_string(q) + " out of range.");
}

template<class Elt>
void EquivariantCohomologyOnKT<Elt>::compute_cohomology_module() {
    using engine::DenseBuf;
    const engine::ModCtx<Elt>& ctx = ring_ctx_;
    ENGINE_CHECK(!ctx.is_field());
    // (S23 G2-P) HECKE_MODULE_PARALLEL: the elimination hook on consumer_pool() (a no-op at one worker, or when a
    // hook is already in effect); the Howell forms split their disjoint row updates over it.  Value-neutral.
    std::unique_ptr<engine::tape::ScopedHostElimHook> par_hook;
    if (engine::elim::module_parallel_enabled()) par_hook.reset(new engine::tape::ScopedHostElimHook());
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const uint32_t max_dim_K = K->get_dimension();
    const uint32_t nq = (max_dim_K == static_cast<uint32_t>(-1)) ? 0 : max_dim_K + 1;
    const uint32_t d = representation_dim;
    const auto& rels = effective_relations();
    pool_ = std::make_shared<BlockPool>();
    module_.assign(nq, ModuleDegree());
    module_released_ = false;

    // C^q_eq = ker A_Cq, free: its Howell rows are a basis.
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t N_q = K->get_num_q_simplices(q);
        const std::vector<const Relation<Elt>*> live = live_relations<Elt>(rels, q, d);
        const BlockMat A = assemble_A_Cq_ring<Elt>(pool_, live, N_q, d, ctx);
        ENGINE_CHECK(!A.ctx().is_field());
        const DenseBuf<Elt> Ad = A.to_dense(engine::Layout::RowMajor);
        const engine::elim::HowellResult<Elt> Kh = engine::elim::kernel_module<Elt>(ctx, engine::cview(Ad));
        bool unit_pivots = true;
        for (uint32_t v : Kh.pivot_v) unit_pivots = unit_pivots && v == 0;
        DenseBuf<Elt>& C = module_[q].C;
        module_[q].piv.clear();
        if (unit_pivots) {
            C = DenseBuf<Elt>(N_q * d, Kh.rows());
            for (uint32_t c = 0; c < Kh.rows(); ++c)
                for (uint32_t i = 0; i < N_q * d; ++i) C.at(i, c) = Kh.H.at(c, i);
            module_[q].piv = Kh.pivots;   // (S23 G2-A) reduce-above with unit pivots: C[piv, :] = I
        } else {
            // (slot-13 WP-5) A free C^q_eq whose Howell form has a non-unit pivot
            // (the pivot is a property of the coordinate order, not of freeness):
            // Hida's lattice L_n ([C] def:hidalattice variant (H)) at e.g.
            // (8, 5).  Until slot 13 this threw; every object computed before is
            // on the branch above, byte for byte.  Basis by Nakayama: the Howell
            // rows independent mod ell, greedily in row order; they generate a
            // free submodule of rank r, equal to ker A_Cq iff its length is r k
            // (checked -- else C^q_eq is genuinely not free and this throws).
            const uint32_t len = engine::elim::length<Elt>(ctx, Kh);
            const uint32_t ncol = N_q * d;
            const uint64_t ell = static_cast<uint64_t>(ctx.ell);
            std::vector<std::vector<uint64_t>> ech;     // reduced rows mod ell, echelon (pivot = first non-zero)
            std::vector<uint32_t> ech_piv, pick;
            auto inv_mod = [ell](uint64_t a) {
                uint64_t r = 1, b = a % ell, e = ell - 2;
                while (e) { if (e & 1) r = static_cast<uint64_t>(static_cast<unsigned __int128>(r) * b % ell);
                            b = static_cast<uint64_t>(static_cast<unsigned __int128>(b) * b % ell); e >>= 1; }
                return r;
            };
            for (uint32_t c = 0; c < Kh.rows(); ++c) {
                std::vector<uint64_t> v(ncol);
                for (uint32_t i = 0; i < ncol; ++i) v[i] = static_cast<uint64_t>(Kh.H.at(c, i)) % ell;
                for (size_t t = 0; t < ech.size(); ++t) {
                    const uint64_t f = v[ech_piv[t]];
                    if (!f) continue;
                    for (uint32_t i = 0; i < ncol; ++i)
                        v[i] = (v[i] + ell - static_cast<uint64_t>(static_cast<unsigned __int128>(f) * ech[t][i] % ell)) % ell;
                }
                uint32_t p = 0;
                while (p < ncol && v[p] == 0) ++p;
                if (p == ncol) continue;
                const uint64_t iv = inv_mod(v[p]);
                for (uint32_t i = 0; i < ncol; ++i) v[i] = static_cast<uint64_t>(static_cast<unsigned __int128>(v[i]) * iv % ell);
                ech.push_back(std::move(v));
                ech_piv.push_back(p);
                pick.push_back(c);
            }
            ENGINE_CHECK_MSG(static_cast<uint64_t>(pick.size()) * ctx.k == len,
                             "EquivariantCohomologyOnK (Z/ell^k): C^q_eq is not free -- length(ker A_Cq) != k * rank "
                             "mod ell ([C, prop:coneintegral](2) needs ell >= 5)");
            C = DenseBuf<Elt>(ncol, static_cast<uint32_t>(pick.size()));
            for (uint32_t c = 0; c < pick.size(); ++c)
                for (uint32_t i = 0; i < ncol; ++i) C.at(i, c) = Kh.H.at(pick[c], i);
        }
    }
    // X_q: d^q in C_eq coordinates.
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t cq = module_[q].C.cols;
        if (q + 1 >= nq) { module_[q].X = DenseBuf<Elt>(0, cq); continue; }
        const BlockMat D = assemble_coboundary_block_ring<Elt>(pool_, K, q, d, ctx);
        ENGINE_CHECK(!D.ctx().is_field());
        DenseBuf<Elt> DC(D.rows(), cq);
        // (S24 MT-1) HECKE_MODULE_MT with a hook: DC = D C in column tiles of C (each column of DC is its own exact
        // sum; blk::gemm_dense on a column tile makes the same products and axpys on those columns)
        const engine::elim::ParallelFor* mt_par = engine::elim::module_mt_hook();
        if (D.rows() > 0 && cq > 0) {
            if (mt_par != nullptr && cq >= 128 && static_cast<uint64_t>(D.rows()) * D.cols() * cq >= (uint64_t(1) << 24)) {
                const uint32_t tiles = std::min<uint32_t>(16, cq / 64);
                engine::elim::module_mt_count(engine::elim::ModuleMtEvent::Gemm);
                (*mt_par)(tiles, [&](uint32_t x) {
                    const engine::elim::ScopedInHookTask in_task;
                    const uint32_t c0 = static_cast<uint32_t>(static_cast<uint64_t>(cq) * x / tiles);
                    const uint32_t c1 = static_cast<uint32_t>(static_cast<uint64_t>(cq) * (x + 1) / tiles);
                    engine::blk::gemm_dense<Elt>(D, engine::cview(module_[q].C).sub(0, module_[q].C.rows, c0, c1 - c0),
                                                 engine::view(DC).sub(0, DC.rows, c0, c1 - c0));
                });
            } else {
                engine::blk::gemm_dense<Elt>(D, engine::cview(module_[q].C), engine::view(DC));
            }
        }
        const DenseBuf<Elt>& C1 = module_[q + 1].C;
        DenseBuf<Elt>& X = module_[q].X;
        X = DenseBuf<Elt>(C1.cols, cq);
        if (cq > 0 && C1.cols > 0) {
            // (S23 G2-A) C^{q+1}_eq is injective (C1[piv, :] = I): X = DC[piv, :], the residual
            // DC - C1 X recomputed over every row; any other outcome takes the Howell solve,
            // which is the legacy route (and throws its own message when DC is not in im C1).
            const bool stream = engine::elim::module_stream_enabled();
            engine::elim::ReadoffStatus st = engine::elim::ReadoffStatus::NotIdentity;
            const bool tried = stream && !module_[q + 1].piv.empty();
            if (tried) {
                X = DenseBuf<Elt>();   // the read-off allocates X itself (no second c_{q+1} x c_q buffer)
                st = engine::elim::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(C1), module_[q + 1].piv,
                                                                 engine::cview(DC), X);
                if (st != engine::elim::ReadoffStatus::Ok) X = DenseBuf<Elt>(C1.cols, cq);
            }
            engine::elim::module_stream_count_solve(stream, st, tried);
            if (st != engine::elim::ReadoffStatus::Ok) {
                const std::vector<std::vector<Elt>> co =
                    engine::elim::solve_multi_rhs_module<Elt>(ctx, engine::cview(C1), engine::cview(DC));
                for (uint32_t t = 0; t < cq; ++t)
                    for (uint32_t j = 0; j < C1.cols; ++j) X.at(j, t) = co[t][j];
            }
        } else if (cq > 0) {
            for (uint32_t i = 0; i < DC.rows; ++i)
                for (uint32_t t = 0; t < cq; ++t)
                    ENGINE_CHECK_MSG(DC.at(i, t) == 0, "EquivariantCohomologyOnK (Z/ell^k): d C^q_eq not in C^{q+1}_eq = 0");
        }
    }
    // d o d = 0 in coordinates (X_q X_{q-1} == 0): the certificate that B^q lies in Z^q.
    for (uint32_t q = 1; q + 1 < nq; ++q) {
        const DenseBuf<Elt>& Xq = module_[q].X;
        const DenseBuf<Elt>& Xp = module_[q - 1].X;
        // (S23 G2-B) the same predicate by one exact GEMM (Sec. 5 of ENGINE_API.md: bit-identical sums);
        // HECKE_MODULE_STREAM=0: the legacy scalar loop (engine::elim::product_is_zero, audit m1).
        const bool gemm = engine::elim::module_stream_enabled();
        engine::elim::module_stream_count_dd(gemm);
        ENGINE_CHECK_MSG(engine::elim::product_is_zero<Elt>(ctx, engine::cview(Xq), engine::cview(Xp), gemm),
                         "EquivariantCohomologyOnK (Z/ell^k): d o d != 0");
    }
    // Z^q, H^q = Z^q / B^q with the cyclic-summand basis; the K-mode counts.
    // (S24 G2-H) HECKE_MODULE_LEAN_PEAK (d), under set_module_early_release(true) only: first Zc_q = ker X_q and dK_q for
    // every q (ascending), then dec_q and H_q in DESCENDING q; X_q is freed once its last reader has run (ker X_q, dK_q
    // and dec_{q+1}, where it is Rl), Zc_q once dec_q has.  So at the largest kernel (q = 1) no decomposition output is
    // live, and at each decomposition only the X / Zc it still needs.  Each step is the same pure function of the same
    // inputs as on the ascending loop, so no kept object can change.
    const bool early = module_early_release_ && engine::elim::module_lean_peak_enabled() &&
                       engine::elim::module_stream_enabled();
    auto kernel_step = [&](uint32_t q) {
        ModuleDegree& M = module_[q];
        const uint32_t cq = M.C.cols;
        engine::elim::HowellResult<Elt> Zh = engine::elim::kernel_module<Elt>(ctx, engine::cview(M.X));
        M.Zc = DenseBuf<Elt>(cq, Zh.rows());
        for (uint32_t c = 0; c < Zh.rows(); ++c)
            for (uint32_t i = 0; i < cq; ++i) M.Zc.at(i, c) = Zh.H.at(c, i);
        // (S23 G2-G) stream route: Zh's Howell rows are read; drop them before the decomposition (value-neutral).
        if (engine::elim::module_stream_enabled()) Zh = engine::elim::HowellResult<Elt>();
    };
    auto dk_step = [&](uint32_t q) {
        ModuleDegree& M = module_[q];
        const uint32_t cq = M.C.cols;
        M.dK = (M.X.rows > 0 && cq > 0) ? engine::elim::invariant_factors<Elt>(ctx, engine::cview(M.X))
                                        : std::vector<uint32_t>{};
    };
    auto dec_step = [&](uint32_t q) {
        ModuleDegree& M = module_[q];
        const uint32_t cq = M.C.cols;
        const DenseBuf<Elt> none(cq, 0);
        const engine::DenseView<const Elt> Bv = (q > 0) ? engine::cview(module_[q - 1].X) : engine::cview(none);
        M.dec = engine::elim::cyclic_decomposition<Elt>(ctx, engine::cview(M.Zc), Bv);
        M.length_H = 0;
        for (uint32_t e : M.dec.exponents) M.length_H += e;
        const uint32_t t = M.dec.rows();
        M.H = DenseBuf<Elt>(M.C.rows, t);
        // (S24 MT-1) HECKE_MODULE_MT: the same per-entry loop, its rows split over the hook (disjoint rows)
        auto h_rows = [&](uint32_t i0, uint32_t i1) {
            for (uint32_t i = i0; i < i1; ++i)
                for (uint32_t c = 0; c < t; ++c) {
                    Elt acc = 0;
                    for (uint32_t l = 0; l < cq; ++l)
                        if (M.C.at(i, l) != 0) acc = ctx.add(acc, ctx.mul(M.C.at(i, l), M.dec.basis.at(l, c)));
                    M.H.at(i, c) = acc;
                }
        };
        const engine::elim::ParallelFor* mt_par = engine::elim::module_mt_hook();
        const uint32_t hm = M.C.rows;
        if (mt_par != nullptr && hm >= 256 && static_cast<uint64_t>(hm) * t * cq >= (uint64_t(1) << 22)) {
            const uint32_t tiles = std::min<uint32_t>(16, hm / 64);
            engine::elim::module_mt_count(engine::elim::ModuleMtEvent::Gemm);
            (*mt_par)(tiles, [&](uint32_t x) {
                const engine::elim::ScopedInHookTask in_task;
                h_rows(static_cast<uint32_t>(static_cast<uint64_t>(hm) * x / tiles),
                       static_cast<uint32_t>(static_cast<uint64_t>(hm) * (x + 1) / tiles));
            });
        } else {
            h_rows(0, hm);
        }
    };
    if (!early) {
        for (uint32_t q = 0; q < nq; ++q) {   // the order of before G2-H
            kernel_step(q);
            dec_step(q);
            dk_step(q);
        }
    } else {
        for (uint32_t q = 0; q < nq; ++q) {
            kernel_step(q);
            dk_step(q);
        }
        if (nq > 0) module_[nq - 1].X = DenseBuf<Elt>();   // no decomposition reads the top degree's X
        for (uint32_t qi = 0; qi < nq; ++qi) {
            const uint32_t q = nq - 1 - qi;
            dec_step(q);
            module_[q].Zc = DenseBuf<Elt>();
            if (q > 0) module_[q - 1].X = DenseBuf<Elt>();   // its last reader was dec_q (Rl)
        }
    }
    if (early) {   // (S24 G2-H) the object ends released, as after release_module_intermediates()
        module_released_ = true;
        engine::elim::module_stream_count(engine::elim::ModuleStreamEvent::LeanReleased, 1);
        engine::elim::lean_peak_count(engine::elim::LeanPeakEvent::EarlyRelease);
    }
}

// ===========================================================================
// (hecke-extensions slot 9, WP-3; decision 4 (A); plan B-1) THE FLAT-COORDINATE
// MODULE ROUTE, taken at ell in {2, 3} under Z/ell^k ONLY.  There 6 is not a
// unit and C^q_eq = ker A_Cq is not free (odd n: FIX-negI's 2 f = 0; ell = 2,
// n = 2: (V_R)^<S> = R(x^2 + y^2) (+) {b xy : 2b = 0}), so compute_cohomology_
// module()'s C^q_eq coordinates do not exist.  The object is the Bredon one
// over Z/ell^k ([C] def:ringsemantics(2); prop:bredon(2)-(3): the tree complex
// over the ring in force), and everything is done in FLAT cochain coordinates:
//   C^q = kernel_module(A_Cq)                  (Howell generators; lengths only)
//   Z^q = kernel_module([A_Cq ; d^q])          (d^q = D^q (x) I, flat)
//   B^q = d^{q-1} applied to the generators of C^{q-1}
//   H^q = invariant_factors(ctx, Z^q, B^q)     (Sec. 13.6: column spans; no basis)
// with three re-derived certificates (ENGINE_CHECK): B^q lies in Z^q (A_Cq B =
// 0 and d^q B = 0, entrywise); sum e_i = length(Z^q) - length(B^q); and the
// rank-nullity identity of d^q on C^q, length(C^q) = length(Z^q) +
// length(B^{q+1}).  No (c1) basis, no C^q_eq-coordinate solve, no r_K /
// ladic_type / charpoly: those are the K-mode, refused here ([C]
// prop:certifiede(5)).  The ell >= 5 route above is not touched.
// ===========================================================================
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::compute_cohomology_flat_module() {
    using engine::DenseBuf;
    const engine::ModCtx<Elt>& ctx = ring_ctx_;
    ENGINE_CHECK(!ctx.is_field());
    ENGINE_CHECK_MSG(hecke::IsBredonPrime(static_cast<uint64_t>(ctx.ell)),
                     "EquivariantCohomologyOnK: the flat-coordinate route is the ell in {2, 3} branch only");
    const SimplicialComplex* K = lssc->get_base_K_complex();
    const uint32_t max_dim_K = K->get_dimension();
    const uint32_t nq = (max_dim_K == static_cast<uint32_t>(-1)) ? 0 : max_dim_K + 1;
    const uint32_t d = representation_dim;
    const auto& rels = effective_relations();
    pool_ = std::make_shared<BlockPool>();
    flat_.assign(nq, FlatModuleDegree());
    auto columns_of_rows = [](const engine::elim::HowellResult<Elt>& h, uint32_t n) {
        DenseBuf<Elt> out(n, h.rows());
        for (uint32_t c = 0; c < h.rows(); ++c)
            for (uint32_t i = 0; i < n; ++i) out.at(i, c) = h.H.at(c, i);
        return out;
    };
    auto length_of_columns = [&ctx](const DenseBuf<Elt>& G) -> uint32_t {
        if (G.cols == 0 || G.rows == 0) return 0;
        DenseBuf<Elt> Gt(G.cols, G.rows);
        for (uint32_t i = 0; i < G.rows; ++i)
            for (uint32_t j = 0; j < G.cols; ++j) Gt.at(j, i) = G.at(i, j);
        return engine::elim::length<Elt>(ctx, engine::cview(Gt));
    };
    // The dense A_Cq and d^q (N_{q+1} d x N_q d; 0 rows at the top degree).
    std::vector<DenseBuf<Elt>> A(nq), D(nq);
    std::vector<BlockMat> Dblk(nq);
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t N_q = K->get_num_q_simplices(q);
        const std::vector<const Relation<Elt>*> live = live_relations<Elt>(rels, q, d);
        const BlockMat Ab = assemble_A_Cq_ring<Elt>(pool_, live, N_q, d, ctx);
        ENGINE_CHECK(!Ab.ctx().is_field());
        A[q] = Ab.to_dense(engine::Layout::RowMajor);
        if (q + 1 < nq) {
            Dblk[q] = assemble_coboundary_block_ring<Elt>(pool_, K, q, d, ctx);
            ENGINE_CHECK(!Dblk[q].ctx().is_field());
            D[q] = Dblk[q].to_dense(engine::Layout::RowMajor);
        } else {
            D[q] = DenseBuf<Elt>(0, N_q * d);
        }
        ENGINE_CHECK(A[q].cols == N_q * d && D[q].cols == N_q * d);
    }
    // C^q and Z^q.
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t dim_q = K->get_num_q_simplices(q) * d;
        FlatModuleDegree& F = flat_[q];
        const engine::elim::HowellResult<Elt> Ch = engine::elim::kernel_module<Elt>(ctx, engine::cview(A[q]));
        F.C = columns_of_rows(Ch, dim_q);
        F.length_C = engine::elim::length<Elt>(ctx, Ch);
        DenseBuf<Elt> S(A[q].rows + D[q].rows, dim_q);
        for (uint32_t i = 0; i < A[q].rows; ++i)
            for (uint32_t j = 0; j < dim_q; ++j) S.at(i, j) = A[q].at(i, j);
        for (uint32_t i = 0; i < D[q].rows; ++i)
            for (uint32_t j = 0; j < dim_q; ++j) S.at(A[q].rows + i, j) = D[q].at(i, j);
        const engine::elim::HowellResult<Elt> Zh = engine::elim::kernel_module<Elt>(ctx, engine::cview(S));
        F.Z = columns_of_rows(Zh, dim_q);
        F.length_Z = engine::elim::length<Elt>(ctx, Zh);
    }
    // B^q = d^{q-1} C^{q-1}, certified inside Z^q; H^q = Z^q / B^q.
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t dim_q = K->get_num_q_simplices(q) * d;
        FlatModuleDegree& F = flat_[q];
        if (q == 0) {
            F.B = DenseBuf<Elt>(dim_q, 0);
        } else {
            const DenseBuf<Elt>& Cp = flat_[q - 1].C;
            F.B = DenseBuf<Elt>(dim_q, Cp.cols);
            if (dim_q > 0 && Cp.cols > 0)
                engine::blk::gemm_dense<Elt>(Dblk[q - 1], engine::cview(Cp), engine::view(F.B));
        }
        for (const DenseBuf<Elt>* M : {&A[q], &D[q]})
            for (uint32_t i = 0; i < M->rows; ++i)
                for (uint32_t t = 0; t < F.B.cols; ++t) {
                    Elt acc = 0;
                    for (uint32_t l = 0; l < dim_q; ++l)
                        if (M->at(i, l) != 0) acc = ctx.add(acc, ctx.mul(M->at(i, l), F.B.at(l, t)));
                    ENGINE_CHECK_MSG(acc == 0, "EquivariantCohomologyOnK (Z/ell^k, ell in {2, 3}): d C^{q-1}_eq is not "
                                               "inside Z^q (A_Cq B^q or d^q B^q is non-zero)");
                }
        F.length_B = length_of_columns(F.B);
        F.exponents = (F.Z.cols > 0) ? engine::elim::invariant_factors<Elt>(ctx, engine::cview(F.Z), engine::cview(F.B))
                                     : std::vector<uint32_t>{};
        F.length_H = 0;
        for (uint32_t e : F.exponents) F.length_H += e;
        ENGINE_CHECK_MSG(F.length_Z >= F.length_B && F.length_H == F.length_Z - F.length_B,
                         "EquivariantCohomologyOnK (Z/ell^k, ell in {2, 3}): sum e_i != length(Z^q) - length(B^q)");
    }
    // Rank-nullity of d^q on C^q_eq: C^q / Z^q ~= d(C^q_eq) = B^{q+1}.
    for (uint32_t q = 0; q < nq; ++q) {
        const uint32_t next = (q + 1 < nq) ? flat_[q + 1].length_B : 0u;
        ENGINE_CHECK_MSG(flat_[q].length_C == flat_[q].length_Z + next,
                         "EquivariantCohomologyOnK (Z/ell^k, ell in {2, 3}): length(C^q) != length(Z^q) + length(B^{q+1})");
    }
}

template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::FlatModuleDegree&
EquivariantCohomologyOnKT<Elt>::flat_module_degree(uint32_t q) const {
    if (!flat_route_)
        throw std::logic_error("flat_module_degree: the flat-coordinate route is taken at ell in {2, 3} under Z/ell^k only");
    if (!cohomology_computed) throw std::runtime_error("flat_module_degree: Cohomology not computed.");
    if (q >= flat_.size())
        throw std::runtime_error("flat_module_degree: degree " + std::to_string(q) + " out of range.");
    return flat_[q];
}

template<class Elt>
void EquivariantCohomologyOnKT<Elt>::require_free_route(const char* what) const {
    if (!flat_route_) return;
    hecke::RefuseCoefficientRing(
        std::string("EquivariantCohomologyOnK::") + what, ring_k_,
        " at ell = " + std::to_string(static_cast<uint64_t>(ring_ctx_.ell)) +
            " (the Bredon object, [C] def:ringsemantics(2): C^q_eq is not free there, so the flat-coordinate route "
            "reports lengths and invariant factors only -- no stored basis, and no K-mode, [C] prop:certifiede(5))");
}

template<class Elt>
const engine::ModCtx<Elt>& EquivariantCohomologyOnKT<Elt>::ring_context() const {
    if (ring_k_ < 2) throw std::logic_error("ring_context: this EquivariantCohomologyOnK is over F_ell");
    return ring_ctx_;
}
template<class Elt>
uint32_t EquivariantCohomologyOnKT<Elt>::num_degrees() const {
    if (flat_route_) {                                       // (slot-9 WP-3) honours at ell in {2, 3}
        (void)flat_module_degree(0);
        return static_cast<uint32_t>(flat_.size());
    }
    require_module("num_degrees", 0);
    return static_cast<uint32_t>(module_.size());
}
template<class Elt>
const typename EquivariantCohomologyOnKT<Elt>::ModuleDegree& EquivariantCohomologyOnKT<Elt>::module_degree(uint32_t q) const {
    require_free_route("module_degree");                     // (slot-9 WP-3)
    require_module("module_degree", q);
    return module_[q];
}
template<class Elt>
void EquivariantCohomologyOnKT<Elt>::release_module_intermediates() {
    require_free_route("release_module_intermediates");
    require_module("release_module_intermediates", 0);
    for (ModuleDegree& M : module_) { M.X = engine::DenseBuf<Elt>(); M.Zc = engine::DenseBuf<Elt>(); }
    if (!module_released_) engine::elim::module_stream_count(engine::elim::ModuleStreamEvent::LeanReleased, 1);
    module_released_ = true;
}
template<class Elt>
const engine::DenseBuf<Elt>& EquivariantCohomologyOnKT<Elt>::module_X(uint32_t q) const {
    const ModuleDegree& M = module_degree(q);
    ENGINE_CHECK_MSG(!module_released_, "EquivariantCohomologyOnK::module_X: released by release_module_intermediates() "
                                        "(S23 G2-E lean retention); X_q is not kept on this object");
    return M.X;
}
template<class Elt>
const engine::DenseBuf<Elt>& EquivariantCohomologyOnKT<Elt>::module_Zc(uint32_t q) const {
    const ModuleDegree& M = module_degree(q);
    ENGINE_CHECK_MSG(!module_released_, "EquivariantCohomologyOnK::module_Zc: released by release_module_intermediates() "
                                        "(S23 G2-E lean retention); Zc_q is not kept on this object");
    return M.Zc;
}
template<class Elt>
uint32_t EquivariantCohomologyOnKT<Elt>::C_eq_rank(uint32_t q) const {
    require_free_route("C_eq_rank");                         // (slot-9 WP-3)
    require_module("C_eq_rank", q);
    return module_[q].C.cols;
}
template<class Elt>
std::vector<uint32_t> EquivariantCohomologyOnKT<Elt>::H_lengths() const {
    std::vector<uint32_t> out;
    if (flat_route_) {                                       // (slot-9 WP-3) honours at ell in {2, 3}
        (void)flat_module_degree(0);
        for (const FlatModuleDegree& F : flat_) out.push_back(F.length_H);
        return out;
    }
    require_module("H_lengths", 0);
    for (const ModuleDegree& M : module_) out.push_back(M.length_H);
    return out;
}
template<class Elt>
const std::vector<uint32_t>& EquivariantCohomologyOnKT<Elt>::H_invariant_factors(uint32_t q) const {
    if (flat_route_) return flat_module_degree(q).exponents;   // (slot-9 WP-3) honours at ell in {2, 3}
    require_module("H_invariant_factors", q);
    return module_[q].dec.exponents;
}
template<class Elt>
const engine::DenseBuf<Elt>& EquivariantCohomologyOnKT<Elt>::H_module_basis(uint32_t q) const {
    require_free_route("H_module_basis");                    // (slot-9 WP-3)
    require_module("H_module_basis", q);
    return module_[q].H;
}
template<class Elt>
std::vector<std::vector<Elt>> EquivariantCohomologyOnKT<Elt>::H_class_coordinates(uint32_t q,
                                                                                engine::DenseView<const Elt> Y) const {
    require_free_route("H_class_coordinates");               // (slot-9 WP-3)
    require_module("H_class_coordinates", q);
    const ModuleDegree& M = module_[q];
    const uint32_t ny = Y.cols();
    ENGINE_CHECK_MSG(Y.rows() == M.C.rows, "H_class_coordinates: Y must be dim_q x ny (flat cochains)");
    if (ny == 0) return {};
    engine::DenseBuf<Elt> Yc(M.C.cols, ny);
    if (M.C.cols > 0) {
        // (S23 G2-A) the same read-off as for X_q and Theta_eq: C[piv, :] = I, so Yc = Y[piv, :], the
        // residual Y - C Yc recomputed over every row; any other outcome takes the Howell solve (the
        // legacy route, whose throw names a column not in C^q_eq).
        const bool stream = engine::elim::module_stream_enabled();
        engine::elim::ReadoffStatus st = engine::elim::ReadoffStatus::NotIdentity;
        const bool tried = stream && !M.piv.empty();
        if (tried) {
            Yc = engine::DenseBuf<Elt>();
            st = engine::elim::solve_unit_pivot_readoff<Elt>(ring_ctx_, engine::cview(M.C), M.piv, Y, Yc);
            if (st != engine::elim::ReadoffStatus::Ok) Yc = engine::DenseBuf<Elt>(M.C.cols, ny);
        }
        engine::elim::module_stream_count_solve(stream, st, tried);
        if (st != engine::elim::ReadoffStatus::Ok) {
            const std::vector<std::vector<Elt>> co =
                engine::elim::solve_multi_rhs_module<Elt>(ring_ctx_, engine::cview(M.C), Y);   // throws if not in C^q_eq
            for (uint32_t t = 0; t < ny; ++t)
                for (uint32_t j = 0; j < M.C.cols; ++j) Yc.at(j, t) = co[t][j];
        }
    } else {
        for (uint32_t i = 0; i < Y.rows(); ++i)
            for (uint32_t t = 0; t < ny; ++t)
                if (Y.at(i, t) != 0) throw std::runtime_error("H_class_coordinates: a column is not in C^q_eq (= 0)");
    }
    if (M.dec.Gamma.cols == 0) {
        for (uint32_t i = 0; i < Yc.rows; ++i)
            for (uint32_t t = 0; t < ny; ++t)
                if (Yc.at(i, t) != 0) throw std::runtime_error("H_class_coordinates: a column is not a cocycle (Z^q = 0)");
        return std::vector<std::vector<Elt>>(ny);
    }
    return engine::elim::cyclic_coordinates<Elt>(ring_ctx_, M.dec, engine::cview(Yc));   // throws if not a cocycle
}
template<class Elt>
uint32_t EquivariantCohomologyOnKT<Elt>::r_K(uint32_t q) const {
    require_free_route("r_K");                               // (slot-9 WP-3) the K-mode
    require_module("r_K", q);
    return static_cast<uint32_t>(module_[q].dK.size());
}
template<class Elt>
bool EquivariantCohomologyOnKT<Elt>::ladic_applicable() const {
    require_free_route("ladic_applicable");                  // (slot-9 WP-3) the K-mode
    require_module("ladic_applicable", 0);
    const uint32_t n = representation_dim - 1;
    return n > 0 && n % 2 == 0 && module_.size() == 3;
}
template<class Elt>
bool EquivariantCohomologyOnKT<Elt>::ladic_type_certified() const {
    require_free_route("ladic_type_certified");              // (slot-9 WP-3) the K-mode
    if (!ladic_applicable())
        throw std::logic_error("ladic_type_certified: the K-mode is defined on n > 0 even only ([E] sec:ring (d))");
    return r_K(0) == module_[0].C.cols && r_K(1) == module_[2].C.cols;
}
template<class Elt>
typename EquivariantCohomologyOnKT<Elt>::LadicType EquivariantCohomologyOnKT<Elt>::ladic_type(uint32_t q) const {
    require_free_route("ladic_type");                        // (slot-9 WP-3) the K-mode
    require_module("ladic_type", q);
    if (!ladic_type_certified())
        throw std::logic_error("ladic_type: the K-mode is not certified at K = " + std::to_string(ring_k_) +
                               " (r_K(d^0) = " + std::to_string(r_K(0)) + " vs rank C^0_eq = " +
                               std::to_string(module_[0].C.cols) + ", r_K(d^1) = " + std::to_string(r_K(1)) +
                               " vs rank C^2_eq = " + std::to_string(module_[2].C.cols) +
                               "); ladic_type has no value at an uncertified level");
    const uint32_t K = ring_k_;
    LadicType out;
    const uint32_t r_here = r_K(q), r_prev = (q > 0) ? r_K(q - 1) : 0u;
    ENGINE_CHECK(module_[q].C.cols >= r_here + r_prev);
    out.free_rank = module_[q].C.cols - r_here - r_prev;
    if (q > 0)
        for (uint32_t e : module_[q - 1].dK)
            if (e < K) out.torsion.push_back(K - e);
    std::sort(out.torsion.begin(), out.torsion.end(), [](uint32_t a, uint32_t b) { return a > b; });
    // Consistency with the computed module (universal coefficients): at a
    // certified K every e_i < K, so H^q(Z/ell^K) = (Z/ell^K)^{r_q} (+) tors
    // H^q(C) (+) tors H^{q+1}(C) -- the second from Tor(H^{q+1}(C), Z/ell^K),
    // i.e. the non-unit elementary divisors of d^q.
    std::vector<uint32_t> want(out.free_rank, K);
    want.insert(want.end(), out.torsion.begin(), out.torsion.end());
    for (uint32_t e : module_[q].dK)
        if (e < K) want.push_back(K - e);
    std::sort(want.begin(), want.end(), [](uint32_t a, uint32_t b) { return a > b; });
    if (want != module_[q].dec.exponents) {
        std::ostringstream o;
        o << "ladic_type: the certified Z_(ell)-type of H^" << q << " (free " << out.free_rank << ", torsion";
        for (uint32_t e : out.torsion) o << " " << e;
        o << ") disagrees with invariant_factors_H^q (";
        for (uint32_t e : module_[q].dec.exponents) o << " " << e;
        o << " ) [ENGINE_CHECK]";
        throw std::logic_error(o.str());
    }
    return out;
}
template<class Elt>
uint32_t EquivariantCohomologyOnKT<Elt>::ladic_e_max() const {
    require_free_route("ladic_e_max");                       // (slot-9 WP-3) the K-mode
    const LadicType t = ladic_type(1);
    return t.torsion.empty() ? 0u : t.torsion.front();
}
template<class Elt>
typename EquivariantCohomologyOnKT<Elt>::LadicLevelState EquivariantCohomologyOnKT<Elt>::ladic_level_state() const {
    require_free_route("ladic_level_state");                 // (slot-9 WP-3) the K-mode
    require_module("ladic_level_state", 0);
    LadicLevelState st;
    st.budget_K = ring_k_;
    st.reached_r0 = r_K(0);
    st.reached_r1 = module_.size() > 1 ? r_K(1) : 0u;
    st.need_r0 = module_[0].C.cols;
    st.need_r1 = module_.size() > 2 ? module_[2].C.cols : 0u;
    st.complete = ladic_applicable() && ladic_type_certified();
    return st;
}

// --- String Conversion ---
template<class Elt>
std::string EquivariantCohomologyOnKT<Elt>::to_string() const {
    return to_string_offset(0);
}

template<class Elt>
std::string EquivariantCohomologyOnKT<Elt>::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    std::string indent = offset_string(offset);
    std::string indent2 = offset_string(offset + 2);
    std::string indent4 = offset_string(offset + 4);

    oss << indent << "Equivariant Cohomology on K (H^*(K,R;L)) Results:\n";
    if (!lssc) {
        oss << indent2 << "Underlying LSSC is null.\n";
        return oss.str();
    }
    oss << indent2 << "Based on LSSC for K: " << lssc->get_base_K_complex()->get_name() << "\n";
    oss << indent2 << "Representation d_V = " << representation_dim << ", Prime p = " << prime << "\n";

    if (!cohomology_computed) {
        oss << indent2 << "Cohomology has not been computed yet.\n";
        return oss.str();
    }

    oss << indent2 << "Cohomology Dimensions H^q_eq(K,R;L):\n";
    for (uint32_t q = 0; q < H_eq_dimensions.size(); ++q) {
        oss << indent4 << "dim H^" << q << " = " << H_eq_dimensions[q] << "\n";
    }

    bool print_bases_detail = true; // Set to true for debugging
    if (print_bases_detail) {
        for (uint32_t q = 0; q < H_eq_dimensions.size(); ++q) {
            oss << indent2 << "Details for q = " << q << ":\n";
            oss << indent4 << "Basis for Z^" << q << "_eq (size " << Z_eq_block_[q].cols() << "):\n";
            oss << indent4 << "Basis for B^" << q << "_eq (size " << B_eq_block_[q].cols() << "):\n";
            oss << indent4 << "Basis for H^" << q << "_eq (size " << H_eq_block_[q].cols() << "):\n";
        }
    }
    return oss.str();
}

#if OC_ENABLE_LEGACY
#include "legacy/EquivariantCohomologyOnKLegacy.inc"   // (S24 CL-L2) the legacy arm of compute_cohomology()
#endif

#ifdef ENGINE_SHADOW_LEGACY
#include "legacy/EquivariantCohomologyOnKShadow.inc"   // (S24 CL-L1) the shadow comparison
#endif

// Stage S13 explicit instantiations.
template class EquivariantCohomologyOnKT<uint32_t>;
template class EquivariantCohomologyOnKT<uint64_t>;
