// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S7 -- HeckeOperatorOnK. See HeckeOperatorOnK.h for the full
// contract statement ([C, def:xi], [C, eq:xiexplicit], [C, thm:xi],
// [C, eq:heckechain], [C, alg:assembly], [C, prop:symconv](2),
// [C, obl:welldef], [C, obl:agreement]) and the binding conventions
// (flat layout, two routes, runtime assertions, JSON schema).
//
// (matrix-engine Phase 3, Stage B) compute() runs on the block model: the two
// routes are two programs over the tau-table pattern (program_fused /
// program_route2), compared with blk::equal; every later step is one engine
// call (blk::gemm_dense, blk::solve_multi_rhs, kernels::gemm).  There is no
// per-cell accumulation, no raw mod_* arithmetic and no `s*d_V + k` loop on
// the result path any more.  The pre-Stage-B dense assembly is preserved as
// the LEGACY arm (ENGINE_BACKEND=legacy, -DENGINE_SHADOW_LEGACY=ON), since
// S24 CL-L2 in Hecke/legacy/HeckeOperatorOnKLegacy.inc (OC_ENABLE_LEGACY=ON only).

#include "HeckeOperatorOnK.h"

#include "HeckeModelDescriptor.h"
#include "CoefficientRing.h"   // (slot-8 WP-3) the coefficient ring, the composite rho_n
#include "HeckeLowRank.h"      // (slot-13 WP-4) the U_ell lever's factored blocks
#include "../../ModArithmetic/DenseElim.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemSL2Z.h"   // SL2ZRelationSeeds_7VertexModel (slot-6 WP-1 guard)
#include "../SimplicialCohomology.h"
#include "../Util/Rss.h"     // (S23 PERF-P4a) VmRSS on the ENGINE_TRACE_RSS lines
#include "../../ModArithmetic/BlockElim.h"
#include "../../ModArithmetic/BlockKernels.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/EngineConfig.h"
#include "../../ModArithmetic/EnvLever.h"   // (S24 CL-R1) the retired slot-13/14 levers
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/helper.h" // offset_string
#include "../../Engine/EngineRuntime.h"
#include "../../Engine/TapeElim.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <system_error>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include "../Util/FieldDiet.h"
#if defined(__GLIBC__)
#include <malloc.h>   // (slot-24 G6) malloc_trim: HeckeOperatorOptions::release_free_heap
#endif

namespace hecke {

using lowrank::LowRankT;          // (slot-13 WP-4)
using lowrank::factor_low_rank;
using lowrank::dense_rows_of;
using lowrank::sum_of_low_rank;

namespace {

using engine::BlockBuilder;
using engine::BlockMat;
using engine::BlockPool;
using engine::BlockRef;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;
using engine::ModCtx;
using engine::cview;
using engine::view;
using engine::kernels::GemmOp;

// n_k in Z reduced to the canonical representative in [0, ell).
template<class Elt>
Elt coeff_mod(int32_t n_k, Elt ell) {
    int64_t r = static_cast<int64_t>(n_k) % static_cast<int64_t>(ell);
    if (r < 0) r += static_cast<int64_t>(ell);
    return static_cast<Elt>(r);
}

// Non-owning row-major DenseBuf over a FlatMatrix's dense buffer.  The caller
// keeps `m` alive for as long as the buffer is used.  (populate_flat_data()
// materialises the buffer of a scalar-flagged matrix; it is const.)
template<class Elt>
DenseBuf<Elt> borrow_flat(const FlatMatrixT<Elt>& m) {
    m.populate_flat_data();
    const std::vector<Elt>& fd = m.get_flat_data_unsafe();
    return DenseBuf<Elt>(const_cast<Elt*>(fd.data()), fd.empty() ? 1 : fd.size(),
                         m.get_rows(), m.get_cols(), Layout::RowMajor);
}

// Intern a dense view by content, returning a ref of the right kind: Zero for
// an all-zero block, Scalar for a square alpha*I, else Dense with beta = 1.
// (The BlockBuilder offers the same as intern_dense; this is the pool-level
// twin for blocks that are memoised across builders.)
template<class Elt>
BlockRef<Elt> intern_view_detect(BlockPool<Elt>& pool, DenseView<const Elt> v) {
    Elt a;
    if (BlockMat<Elt>::is_zero_view(v)) return BlockRef<Elt>::zero();
    if (BlockMat<Elt>::is_scalar_view(v, a)) return BlockRef<Elt>::scalar(a);
    return BlockRef<Elt>::dense(pool.intern_view(v, Layout::RowMajor));
}

// The ref of a d x d FlatMatrix: Scalar when the FlatMatrix already knows it
// is alpha*I (no materialisation), else interned once by content.
template<class Elt>
BlockRef<Elt> intern_flat(BlockPool<Elt>& pool, const FlatMatrixT<Elt>& M) {
    Elt sv = 0;
    if (M.is_currently_scalar(sv)) return BlockRef<Elt>::scalar(sv);   // Zero when sv == 0
    const DenseBuf<Elt> tmp = borrow_flat<Elt>(M);
    return intern_view_detect<Elt>(pool, cview(tmp));
}

// First non-zero entry of a dense view (row-major scan order); false if none.
template<class Elt>
bool first_nonzero(DenseView<const Elt> v, uint32_t& r_out, uint32_t& c_out) {
    for (uint32_t r = 0; r < v.rows(); ++r)
        for (uint32_t c = 0; c < v.cols(); ++c)
            if (v.at(r, c) != 0) { r_out = r; c_out = c; return true; }
    return false;
}

// Row-major FlatMatrix copy of a dense view (the level-(a) FlatMatrix shapes).
template<class Elt>
FlatMatrixT<Elt> flat_of_view(DenseView<const Elt> v, Elt p) {
    const uint32_t r = v.rows(), c = v.cols();
    if (r == 0 || c == 0) return FlatMatrixT<Elt>(r, c, p);
    // One tiled copy straight into the vector the FlatMatrix adopts (the
    // engine's entries are canonical residues: no re-reduction).
    std::vector<Elt> data(static_cast<size_t>(r) * c);
    DenseBuf<Elt> rm(data.data(), data.size(), r, c, Layout::RowMajor);
    engine::kernels::copy<Elt>(v, view(rm));
    return FlatMatrixT<Elt>::FromCanonical(r, c, p, std::move(data));
}

// The relation block matrix R_q of runtime assertion (i): one block row per
// L-relation of degree q (ALL of them, vacuous ones included -- the count is
// a certificate datum), I at s_lhs and -M at s_rhs (I - M on the diagonal
// when they coincide), so that R_q * y == 0 iff y satisfies every relation.
template<class Elt>
BlockMat<Elt> assemble_relation_block(const std::shared_ptr<BlockPool<Elt>>& pool,
                                      const std::vector<typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation>& rels,
                                      uint32_t q, uint32_t N_q, uint32_t d, Elt p,
                                      std::vector<const typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation*>& used) {
    used.clear();
    for (const auto& rel : rels)
        if (rel.q_dim == q) used.push_back(&rel);
    const std::vector<uint32_t> row_dims(used.size(), d), col_dims(N_q, d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    const Elt neg1 = b.ctx().neg(Elt(1));
    for (uint32_t ri = 0; ri < used.size(); ++ri) {
        const auto& rel = *used[ri];
        ENGINE_CHECK_MSG(rel.M.get_rows() == d && rel.M.get_cols() == d && rel.s_lhs_idx_in_Kq < N_q &&
                             rel.s_rhs_idx_in_Kq < N_q,
                         "HeckeOperatorOnK: relation matrix / simplex index out of shape");
        const BlockRef<Elt> M = intern_flat<Elt>(*pool, rel.M);
        if (rel.s_lhs_idx_in_Kq == rel.s_rhs_idx_in_Kq) {
            b.set(ri, rel.s_lhs_idx_in_Kq, b.compose(M, Elt(1), neg1));          // I - M
        } else {
            b.set(ri, rel.s_lhs_idx_in_Kq, BlockRef<Elt>::identity());            // I
            b.set(ri, rel.s_rhs_idx_in_Kq, b.compose(M, Elt(0), neg1));          // -M
        }
    }
    return b.finalize();
}

// (D_1 (x) I) of runtime assertion (ii): the coboundary N_{q+1} x N_q with
// every +-1 a Scalar block (no data).  Empty (0 x N_q) at the top degree.
template<class Elt>
BlockMat<Elt> assemble_coboundary_block(const std::shared_ptr<BlockPool<Elt>>& pool,
                                        const SimplicialComplex* K, uint32_t q, uint32_t d, Elt p) {
    const FlatMatrixT<Elt> D1 = K->template coboundary_operator_elt<Elt>(q, p);   // N_{q+1} x N_q
    const uint32_t N_src = K->get_num_q_simplices(q);
    const uint32_t N_dst = (q + 1 <= K->get_dimension()) ? K->get_num_q_simplices(q + 1) : 0;
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
                         "HeckeOperatorOnK: coboundary operator shape disagrees with the simplex counts");
    }
    return b.finalize();
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template<class Elt>
HeckeOperatorOnKT<Elt>::HeckeOperatorOnKT(const EquivariantCohomologyOnKT<Elt>* cohomology_of_L,
                                   const HeckeCosetTablesT<Elt>* cosets,
                                   const HeckeTauTableT<Elt>* tau,
                                   FibreRepresentation rep)
    : coh_(cohomology_of_L), cosets_(cosets), tau_(tau), rep_(rep) {
    if (!coh_ || !cosets_ || !tau_) {
        throw std::invalid_argument("HeckeOperatorOnK: null input pointer.");
    }
    if (!coh_->is_cohomology_computed()) {
        throw std::runtime_error(
            "HeckeOperatorOnK: cohomology_of_L must be computed (call "
            "compute_cohomology() first).");
    }
    // (slot-8 WP-3) The ring is the cohomology object's; the ring in force
    // must agree ([E] sec:ring (a)-(b)).  At the default both are F_ell.
    {
        const uint32_t k_set = CoefficientRingK("HeckeOperatorOnK"), k_in = coh_->coefficient_ring_k();
        if (k_set != k_in)
            RefuseCoefficientRing("HeckeOperatorOnK", k_set != 1 ? k_set : k_in,
                                  " (the ring in force, " + CoefficientRingText(k_set) +
                                      ", is not the cohomology object's, " + CoefficientRingText(k_in) + ")");
        if (k_in >= 2) { init_composite(); return; }
    }

    // (slot-13 WP-5) Hida's lattice is a module-route fibre only.
    ENGINE_CHECK_MSG(rep_ != FibreRepresentation::HidaLatticeH,
                     "HeckeOperatorOnK: FibreRepresentation::HidaLatticeH is honoured on the module route "
                     "(coefficient_ring = Z/ell^k, k >= 2) only ([C] def:hidalattice variant (H))");
    ell_ = coh_->get_prime();
    d_V_ = coh_->get_representation_dim();
    if (d_V_ == 0) throw std::runtime_error("HeckeOperatorOnK: d_V == 0.");
    n_sym_ = d_V_ - 1;
    // (slot-11 WP-5) Q_n: the fibre has dimension ell + 1 and the n of Q_n is
    // the tables' n (the coset / tau tables are the ones of Sym^n; only their
    // exact data is read, through rho_of_exact).
    uint32_t table_dim = d_V_;
    if (rep_ == FibreRepresentation::QuotientL) {
        if (static_cast<uint64_t>(d_V_) != static_cast<uint64_t>(ell_) + 1)
            throw std::runtime_error("HeckeOperatorOnK (quotient_L): the cohomology object's representation_dim is "
                                     "not ell + 1.");
        n_sym_ = tau_->n_sym();
        table_dim = n_sym_ + 1;
    }
    m_ = cosets_->num_cosets();
    if (m_ == 0) throw std::runtime_error("HeckeOperatorOnK: no cosets.");

    // Model shape (matrix-engine Phase 6, audit item 7.18): asserted against
    // the MODEL DESCRIPTOR, not against the literals 7/11/5. The predicate is
    // unchanged -- SL2Z7VertexModel() carries dimension 2 and the counts
    // (7, 11, 5) -- so this is value-preserving; what changed is that the
    // shape is now a datum the campaign driver can also read and report.
    // The tau table was built against this model too (its own constructor
    // validates the same shape).
    const SimplicialModelDescriptor& model_desc = SL2Z7VertexModel();
    const SimplicialComplex* Kp = K();
    RequireModel(model_desc, Kp, "HeckeOperatorOnK");
    num_degrees_ = model_desc.num_degrees(); // == 3

    // Cross-check ell / d_V consistency with the S3 tables (the omega blocks
    // rho_n(s_i^{-1}) are (n+1) x (n+1) over F_ell)...
    // (slot-14 WP-10) Through the block accessor: a streamed coset table
    // evaluates the one probe block (the same bytes the table would store).
    const CosetRhoT<Elt> omega1_block = cosets_->rho_s_inv_block(1);
    const FlatMatrixT<Elt>& omega1 = omega1_block.get();
    if (omega1.get_p() != ell_ || omega1.get_rows() != table_dim || omega1.get_cols() != table_dim) {
        throw std::runtime_error(
            "HeckeOperatorOnK: HeckeCosetTables (ell, n_sym) does not match the "
            "cohomology object's (prime, representation_dim).");
    }
    // ... and with the S4 xi-term table (vertex slots always hold exactly one
    // term, per the S4 (T3) normalization, so slot (1, 0, 0) is inhabited).
    const std::vector<XiTermT<Elt>>& probe = tau_->xi_terms(1, 0, 0);
    // (slot-13 WP-4) A tau table built under the U_ell lever keeps its pool
    // factored: probe the slot's factor shape instead of the dense block.
    const bool probe_bad = probe.empty() ||
        (tau_->rho_pool_is_low_rank()
             ? (tau_->rho_pool_low_rank(probe[0].rho_g_gamma.index()).rows != table_dim ||
                tau_->rho_pool_low_rank(probe[0].rho_g_gamma.index()).cols != table_dim)
             : (probe[0].rho_g_gamma.get_p() != ell_ || probe[0].rho_g_gamma.get_rows() != table_dim ||
                probe[0].rho_g_gamma.get_cols() != table_dim));
    if (probe_bad) {
        throw std::runtime_error(
            "HeckeOperatorOnK: HeckeTauTable (ell, n_sym) does not match the "
            "cohomology object's (prime, representation_dim).");
    }
    // (slot-6 WP-1) Nothing below runs for the default SymN operator.
    if (rep_ != FibreRepresentation::SymN) build_contragredient_tables();
}

// (slot-6 WP-1) The contragredient operator's fused-route blocks, rebuilt from
// the EXACT coefficient-free data -- s_i^{-1} of the coset table, g gamma_k of
// the tau table -- through rho_of_exact (i.e. rho_n^vee).  One block per
// rho-pool slot: the tau table dedups its pool by the exact REDUCED key
// (g gamma) mod ell, and rho_n^vee is a function of that key (adj commutes
// with reduction), so the slot determines the block; every revisit of a slot
// is checked to carry the same reduced key.  Also the structural guard that
// the local system IS the contragredient one: its 8 model seed relations
// must carry rho_n^vee(g_r) ([C, def:contragredient]).
template<class Elt>
void HeckeOperatorOnKT<Elt>::build_contragredient_tables() {
    ENGINE_CHECK_MSG(rep_ == FibreRepresentation::ContragredientN || rep_ == FibreRepresentation::QuotientL,
                     "HeckeOperatorOnK: unknown fibre representation");
    const auto& rels = coh_->get_lssc()->get_user_defined_simplex_relations();
    const auto& seeds = SL2ZRelationSeeds_7VertexModel();
    if (rels.size() < seeds.size()) {
        throw std::runtime_error("HeckeOperatorOnK (contragredient): the local system has fewer relations "
                                 "than the 7-vertex model's seed list.");
    }
    for (size_t r = 0; r < seeds.size(); ++r) {
        const FlatMatrixT<Elt> want = rho_of_exact(seeds[r].g_r);
        if (rels[r].tag != seeds[r].tag || !(rels[r].M == want)) {
            if (rep_ == FibreRepresentation::QuotientL)   // (slot-11 WP-5)
                throw std::runtime_error(std::string("HeckeOperatorOnK (quotient_L): relation '") + rels[r].tag +
                                         "' does not carry rho_{Q_n}(g_r) -- the cohomology object is not the "
                                         "quotient local system Q_n (GenerateSL2ZQuotientLRelationsWithGroupData).");
            throw std::runtime_error(std::string("HeckeOperatorOnK (contragredient): relation '") + rels[r].tag +
                                     "' does not carry rho_n^vee(g_r) -- the cohomology object is not the "
                                     "contragredient local system D_n of [C, def:contragredient].");
        }
    }
    omega_dual_.assign(static_cast<size_t>(m_) + 1, FlatMatrixT<Elt>(0, 0, ell_));
    for (uint32_t i = 1; i <= m_; ++i) omega_dual_[i] = rho_of_exact(cosets_->s_inv(i));
    const size_t n_slots = tau_->rho_pool().size();
    rho_pool_dual_.assign(n_slots, FlatMatrixT<Elt>(0, 0, ell_));
    std::vector<char> filled(n_slots, 0);
    std::vector<std::array<Elt, 4>> slot_key(n_slots);
    const SimplicialComplex* Kp = K();
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = Kp->get_num_q_simplices(q);
        for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const std::vector<ChainTerm>& cts = tau_->tau(i, q, sigma);
                const std::vector<XiTermT<Elt>>& xts = tau_->xi_terms(i, q, sigma);
                ENGINE_CHECK_MSG(cts.size() == xts.size(),
                                 "HeckeOperatorOnK: xi_terms and tau term lists disagree in length.");
                for (size_t k = 0; k < cts.size(); ++k) {
                    const uint32_t slot = xts[k].rho_g_gamma.index();
                    ENGINE_CHECK_MSG(slot < n_slots, "HeckeOperatorOnK: xi term names a rho-pool slot out of range");
                    const Mat22 gg = cosets_->g() * cts[k].gamma;   // exact (I6); det N, never inverted (I7)
                    std::array<Elt, 4> key;
                    if constexpr (std::is_same_v<Elt, uint32_t>) key = gg.reduced_mod(ell_);
                    else key = gg.reduced_mod64(ell_);
                    if (!filled[slot]) {
                        rho_pool_dual_[slot] = rho_of_exact(gg);
                        slot_key[slot] = key;
                        filled[slot] = 1;
                    } else {
                        ENGINE_CHECK_MSG(slot_key[slot] == key,
                                         "HeckeOperatorOnK: one rho-pool slot names two reduced keys (g gamma) mod ell");
                    }
                }
            }
        }
    }
    for (size_t t = 0; t < n_slots; ++t)
        ENGINE_CHECK_MSG(filled[t], "HeckeOperatorOnK: a rho-pool slot is named by no tau term");
}

// (slot-8 WP-3) The composite constructor tail: the checks of the field
// constructor, restated for a Z/ell^k cohomology object.
template<class Elt>
void HeckeOperatorOnKT<Elt>::init_composite() {
    ring_k_ = coh_->coefficient_ring_k();
    ring_ctx_ = coh_->ring_context();
    ENGINE_CHECK(!ring_ctx_.is_field());
    // (slot-9 WP-3; decision 4 (A)) The cohomology object honours ell in {2, 3}
    // (the Bredon object, flat-coordinate route); the Hecke operator does NOT.
    if (IsBredonPrime(static_cast<uint64_t>(ring_ctx_.ell)))
        RefuseCoefficientRing("HeckeOperatorOnK", ring_k_,
                              " at ell = " + std::to_string(static_cast<uint64_t>(ring_ctx_.ell)) +
                                  " (6 is not a unit: the Hecke action over Z/ell^k is honoured at ell >= 5 only; "
                                  "[E] sec:ring (b3))");
    // (slot-13 WP-5) Hida's lattice (rho^H_n = rho_n(g^T)^T, [C] def:hidalattice
    // variant (H)) is honoured here, on the module route; every other non-SymN
    // fibre keeps the refusal.
    if (rep_ != FibreRepresentation::SymN && rep_ != FibreRepresentation::HidaLatticeH)
        RefuseCoefficientRing("HeckeOperatorOnK (contragredient)", ring_k_);
    ell_ = ring_ctx_.ell;                 // the cell's prime (the tables are keyed on it)
    d_V_ = coh_->get_representation_dim();
    if (d_V_ == 0) throw std::runtime_error("HeckeOperatorOnK: d_V == 0.");
    n_sym_ = d_V_ - 1;
    m_ = cosets_->num_cosets();
    if (m_ == 0) throw std::runtime_error("HeckeOperatorOnK: no cosets.");
    const SimplicialModelDescriptor& model_desc = SL2Z7VertexModel();
    RequireModel(model_desc, K(), "HeckeOperatorOnK");
    num_degrees_ = model_desc.num_degrees();
    const CosetRhoT<Elt> omega1_block = cosets_->rho_s_inv_block(1);   // (slot-14 WP-10) streamed tables too
    if (omega1_block.get().get_p() != ell_ || omega1_block.get().get_rows() != d_V_ || tau_->ell() != ell_ ||
        tau_->n_sym() != n_sym_) {
        throw std::runtime_error(
            "HeckeOperatorOnK (Z/ell^k): the coset / tau tables' (ell, n_sym) does not match the cohomology "
            "object's (the tables are built at the cell's prime ell; only their exact data is read)");
    }
    if (tau_->used_mod_ell_solve())
        RefuseCoefficientRing("HeckeOperatorOnK", ring_k_,
                              " (the tau datum used the mod-ell triangle fallback, so it is not integral)");
    // (slot-13 WP-5) The structural guard of the contragredient (slot-6 WP-1),
    // for Hida's lattice: the local system's 8 model seed relations must carry
    // rho^H_n(g_r) mod ell^k.
    if (rep_ == FibreRepresentation::HidaLatticeH) {
        const auto& rels = coh_->get_lssc()->get_user_defined_simplex_relations();
        const auto& seeds = SL2ZRelationSeeds_7VertexModel();
        ENGINE_CHECK_MSG(rels.size() >= seeds.size(),
                         "HeckeOperatorOnK (hida_lattice): the local system has fewer relations than the model's seeds");
        for (size_t r = 0; r < seeds.size(); ++r)
            ENGINE_CHECK_MSG(rels[r].tag == seeds[r].tag &&
                                 rels[r].M == RhoHidaOfExactRing<Elt>(ring_ctx_, seeds[r].g_r, n_sym_),
                             "HeckeOperatorOnK (hida_lattice): the local system is not the rho^H_n one "
                             "(build it on GenerateSL2ZRelationsRingHida)");
    }
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::require_field_op(const char* what) const {
    if (ring_k_ >= 2)
        throw std::logic_error(std::string("HeckeOperatorOnK::") + what + ": a field (F_ell) accessor, but this operator "
                               "is over " + CoefficientRingText(ring_k_) +
                               " -- use H_matrix_module() / Theta_module() ([E] sec:ring (b3))");
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::require_module_op(const char* what, uint32_t q) const {
    if (ring_k_ < 2)
        throw std::logic_error(std::string("HeckeOperatorOnK::") + what + ": a Z/ell^k accessor, but this operator is over F_ell");
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    check_degree(q);
}

// (S23 G2; audit m3) VALIDATION ONLY: HeckeOperatorOnK.h, ScopedThetaEqPerturbation.
namespace {
std::atomic<int> g_theta_eq_perturbation{0};
} // namespace
ScopedThetaEqPerturbation::ScopedThetaEqPerturbation(bool on) : saved_(g_theta_eq_perturbation.exchange(on ? 1 : 0)) {}
ScopedThetaEqPerturbation::~ScopedThetaEqPerturbation() { g_theta_eq_perturbation.store(saved_); }
namespace {
std::atomic<int> g_route2_perturbation{0};   // (S23 G2-D) validation only
} // namespace
ScopedRoute2Perturbation::ScopedRoute2Perturbation(Route2Site site)
    : saved_(g_route2_perturbation.exchange(static_cast<int>(site))) {}
ScopedRoute2Perturbation::~ScopedRoute2Perturbation() { g_route2_perturbation.store(saved_); }

// (slot-8 WP-3) The module route -- the note above the class.  (S23 G2-C/D/E) HECKE_MODULE_STREAM (default on)
// selects the stream route; "0" the legacy route below, bit for bit (both compute the same unique objects).
template<class Elt>
void HeckeOperatorOnKT<Elt>::compute_module() {
    if (engine::elim::module_stream_enabled()) compute_module_stream();
    else compute_module_legacy();
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::compute_module_legacy() {
    using engine::DenseBuf;
    const engine::ModCtx<Elt>& ctx = ring_ctx_;
    const Elt p = ctx.p;
    const SimplicialComplex* Kp = K();
    const uint32_t d = d_V_;
    // rho_n mod ell^k of exact matrices, memoised on the key reduced mod ell^k
    // (rho_n(g) mod ell^k is a polynomial in the entries of g mod ell^k).
    std::map<std::array<Elt, 4>, FlatMatrixT<Elt>> rho_memo;
    auto key_of = [&](const Mat22& g) {
        std::array<Elt, 4> r;
        if constexpr (std::is_same_v<Elt, uint32_t>) { const auto a = g.reduced_mod(p); r = {a[0], a[1], a[2], a[3]}; }
        else { const auto a = g.reduced_mod64(p); r = {a[0], a[1], a[2], a[3]}; }
        return r;
    };
    auto rho = [&](const Mat22& g) -> const FlatMatrixT<Elt>& {
        const std::array<Elt, 4> key = key_of(g);
        auto it = rho_memo.find(key);
        if (it == rho_memo.end()) {
            // (slot-13 WP-5) rho^H_n(g) = rho_n(g^T)^T on Hida's lattice; SymN is the shipped call.
            if (rep_ == FibreRepresentation::HidaLatticeH)
                it = rho_memo.emplace(key, ComputeSymmetricPowerMatrixRing<Elt>(ctx, key[0], key[2], key[1], key[3], n_sym_)
                                               .transpose()).first;
            else
                it = rho_memo.emplace(key, ComputeSymmetricPowerMatrixRing<Elt>(ctx, key[0], key[1], key[2], key[3], n_sym_)).first;
        }
        return it->second;
    };
    auto dense_of = [&](const FlatMatrixT<Elt>& M) {
        DenseBuf<Elt> b(M.get_rows(), M.get_cols());
        for (uint32_t i = 0; i < M.get_rows(); ++i)
            for (uint32_t j = 0; j < M.get_cols(); ++j) b.at(i, j) = M.get_entry(i, j);
        return b;
    };
    // n_k mod ell^k (n_k is a small signed integer).  Unsigned throughout
    // (slot-8 WP-3r, review M-2): ell^k may lie in [2^63, 2^64) (U64_Wide),
    // where a signed cast of the modulus would be negative.
    auto coeff_ring = [&](int32_t c) -> Elt {
        if (c >= 0) return static_cast<Elt>(static_cast<Elt>(c) % p);
        const Elt m = static_cast<Elt>(static_cast<Elt>(-static_cast<int64_t>(c)) % p);
        return m == 0 ? Elt(0) : static_cast<Elt>(p - m);
    };
    std::vector<DenseBuf<Elt>> omega(m_ + 1);
    for (uint32_t i = 1; i <= m_; ++i) omega[i] = dense_of(rho(cosets_->s_inv(i)));
    std::map<std::pair<uint32_t, std::array<Elt, 4>>, DenseBuf<Elt>> fused_memo;   // (i, key(g gamma)) -> omega_i rho(g gamma)
    // (slot-13 WP-4) The U_ell lever over Z/ell^k: rho(g gamma) and
    // rho(delta_i gamma) factored into <= k outer products (the rank claim,
    // ENGINE_CHECKed per block), the fused memo holding (omega_i U, V); the
    // dense rho of a singular key is never memoised.
    // (owner decision 2026-09-28, Q8) Auto by default: in effect iff ell | N and SymN.
    const bool low_rank = HeckeOperatorOptions::u_ell_low_rank_in_effect(cosets_->N(), static_cast<uint64_t>(ctx.ell), rep_);
    if (low_rank) {
        ENGINE_CHECK_MSG(rep_ == FibreRepresentation::SymN,
                         "HeckeOperatorOnK: the U_ell low-rank lever is defined for the SymN fibre only.");
        ENGINE_CHECK_MSG(cosets_->N() % static_cast<uint64_t>(ctx.ell) == 0,
                         "HeckeOperatorOnK: the U_ell low-rank lever needs ell | N (U_ell, T_{ell^r}).");
    }
    std::map<std::array<Elt, 4>, LowRankT<Elt>> lr_memo;                           // key -> U V
    std::map<std::pair<uint32_t, std::array<Elt, 4>>, LowRankT<Elt>> fused_lr_memo; // (i, key) -> (omega_i U) V
    auto lr_of = [&](const Mat22& g) -> const LowRankT<Elt>& {
        const std::array<Elt, 4> key = key_of(g);
        auto it = lr_memo.find(key);
        if (it == lr_memo.end()) {
            const FlatMatrixT<Elt> R = ComputeSymmetricPowerMatrixRing<Elt>(ctx, key[0], key[1], key[2], key[3], n_sym_);
            it = lr_memo.emplace(key, factor_low_rank<Elt>(ctx, dense_rows_of<Elt>(R), d, d, std::min<uint32_t>(ctx.k, d))).first;
        }
        return it->second;
    };
    auto add_block_lr = [&](DenseBuf<Elt>& dst, uint32_t sigma, uint32_t sk, Elt c, const LowRankT<Elt>& lr) {
        for (uint32_t a = 0; a < d; ++a)
            for (uint32_t s2 = 0; s2 < lr.t; ++s2) {
                const Elt ua = lr.U[static_cast<size_t>(a) * lr.t + s2];
                if (ua == 0) continue;
                const Elt cu = ctx.mul(c, ua);
                const Elt* vrow = &lr.V[static_cast<size_t>(s2) * d];
                for (uint32_t b = 0; b < d; ++b) {
                    if (vrow[b] == 0) continue;
                    Elt& e = dst.at(sigma * d + a, sk * d + b);
                    e = ctx.add(e, ctx.mul(cu, vrow[b]));
                }
            }
    };

    A_module_.assign(num_degrees_, DenseBuf<Elt>());
    Theta_module_.assign(num_degrees_, DenseBuf<Elt>());
    Theta_eq_module_.assign(num_degrees_, DenseBuf<Elt>());
    released_ = false;   // (S23 G2-E) the legacy route keeps every object
    // (S23 G2-C/D) the EFFECTIVE counters of the legacy assembly: per operator degree
    engine::elim::module_stream_count(engine::elim::ModuleStreamEvent::OmegaTable, num_degrees_);
    engine::elim::module_stream_count(engine::elim::ModuleStreamEvent::Route2Full, num_degrees_);
    if (low_rank) engine::elim::module_stream_count(engine::elim::ModuleStreamEvent::LrScalar, num_degrees_);
    relation_checks_ = cocycle_checks_ = theta_eq_solves_ = 0;
    theta_eq_residual_max_ = 0;
    bool routes_equal = true;
    const auto& rels = coh_->get_lssc()->get_user_defined_simplex_relations();

    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = Kp->get_num_q_simplices(q);
        const uint32_t dim = N_q * d;
        DenseBuf<Elt> T(dim, dim), T2(dim, dim);
        auto add_block = [&](DenseBuf<Elt>& dst, uint32_t sigma, uint32_t sk, Elt c, const DenseBuf<Elt>& B) {
            for (uint32_t a = 0; a < d; ++a)
                for (uint32_t b = 0; b < d; ++b) {
                    const Elt v = B.at(a, b);
                    if (v == 0) continue;
                    Elt& e = dst.at(sigma * d + a, sk * d + b);
                    e = ctx.add(e, ctx.mul(c, v));
                }
        };
        for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const std::vector<ChainTerm>& cts = tau_->tau(i, q, sigma);
                for (const ChainTerm& ct : cts) {
                    const Elt c = coeff_ring(ct.coeff);
                    if (c == 0) continue;
                    ENGINE_CHECK_MSG(ct.sigma_idx < N_q, "HeckeOperatorOnK: tau term names a simplex out of range");
                    // fused: n_k rho(s_i^{-1}) rho(g gamma_k)
                    const Mat22 gg = cosets_->g() * ct.gamma;              // exact (I6); never inverted (I7)
                    const auto fk = std::make_pair(i, key_of(gg));
                    if (low_rank) {
                        auto jt = fused_lr_memo.find(fk);
                        if (jt == fused_lr_memo.end()) {
                            const LowRankT<Elt>& lr = lr_of(gg);
                            LowRankT<Elt> f;
                            f.rows = d; f.cols = d; f.t = lr.t; f.V = lr.V;
                            f.U.assign(static_cast<size_t>(d) * lr.t, Elt(0));
                            if (lr.t > 0) {
                                DenseBuf<Elt> Ub(const_cast<Elt*>(lr.U.data()), lr.U.size(), d, lr.t, engine::Layout::RowMajor);
                                DenseBuf<Elt> Wb(f.U.data(), f.U.size(), d, lr.t, engine::Layout::RowMajor);
                                engine::kernels::gemm<Elt>(ctx, engine::cview(omega[i]), engine::cview(Ub), engine::view(Wb));
                            }
                            jt = fused_lr_memo.emplace(fk, std::move(f)).first;
                        }
                        add_block_lr(T, sigma, ct.sigma_idx, c, jt->second);
                        const Mat22 dg = cosets_->delta(i) * ct.gamma;
                        add_block_lr(T2, sigma, ct.sigma_idx, c, lr_of(dg));
                        continue;
                    }
                    auto it = fused_memo.find(fk);
                    if (it == fused_memo.end()) {
                        const DenseBuf<Elt> R = dense_of(rho(gg));
                        DenseBuf<Elt> P(d, d);
                        engine::kernels::gemm<Elt>(ctx, engine::cview(omega[i]), engine::cview(R), engine::view(P));
                        it = fused_memo.emplace(fk, std::move(P)).first;
                    }
                    add_block(T, sigma, ct.sigma_idx, c, it->second);
                    // route 2: n_k rho(delta_i gamma_k), the exact product first
                    const Mat22 dg = cosets_->delta(i) * ct.gamma;
                    add_block(T2, sigma, ct.sigma_idx, c, dense_of(rho(dg)));
                }
            }
        }
        if (dim > 0) {   // validation only (ScopedRoute2Perturbation): the same entry as the stream route's
            const int site = g_route2_perturbation.load(std::memory_order_relaxed);
            if ((q == 0 && site == 1) || (q == 1 && site == 3))
                T2.at(dim - 1, dim - 1) = ctx.add(T2.at(dim - 1, dim - 1), Elt(1));
            else if (q == 1 && site == 2)
                T2.at(0, 0) = ctx.add(T2.at(0, 0), Elt(1));
            else if (q == 1 && site == 4 && N_q > 3)   // (G2-H audit F2) an interior block row
                T2.at(3 * d + d / 2, dim / 2) = ctx.add(T2.at(3 * d + d / 2, dim / 2), Elt(1));
        }
        for (uint32_t a = 0; a < dim && routes_equal; ++a)
            for (uint32_t b = 0; b < dim; ++b)
                if (T.at(a, b) != T2.at(a, b)) { routes_equal = false; break; }
        if (!routes_equal)
            throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): the fused route and route 2 differ at q=" +
                                     std::to_string(q) + " (O12(a) over Z/ell^k)");

        // Y = Theta . H (the cyclic-summand basis), (i), (ii).
        const DenseBuf<Elt>& Hb = coh_->H_module_basis(q);
        const uint32_t t = Hb.cols;
        DenseBuf<Elt> Y(dim, t);
        if (dim > 0 && t > 0) engine::kernels::gemm<Elt>(ctx, engine::cview(T), engine::cview(Hb), engine::view(Y));
        for (const auto& rel : rels) {
            if (rel.q_dim != q) continue;
            for (uint32_t col = 0; col < t; ++col) {
                for (uint32_t a = 0; a < d; ++a) {
                    Elt acc = Y.at(rel.s_lhs_idx_in_Kq * d + a, col);
                    for (uint32_t b = 0; b < d; ++b) {
                        const Elt mv = rel.M.get_entry(a, b);
                        if (mv != 0) acc = ctx.sub(acc, ctx.mul(mv, Y.at(rel.s_rhs_idx_in_Kq * d + b, col)));
                    }
                    if (acc != 0)
                        throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): runtime assertion (i) failed -- Theta z "
                                                 "violates relation '" + rel.tag + "'");
                }
                ++relation_checks_;
            }
        }
        if (q + 1 < num_degrees_) {
            const FlatMatrixT<Elt> D1 = Kp->template coboundary_operator_elt<Elt>(q, p);
            const uint32_t N1 = Kp->get_num_q_simplices(q + 1);
            for (uint32_t col = 0; col < t; ++col) {
                for (uint32_t s1 = 0; s1 < N1; ++s1)
                    for (uint32_t a = 0; a < d; ++a) {
                        Elt acc = 0;
                        for (uint32_t s0 = 0; s0 < N_q; ++s0) {
                            const Elt inc = D1.get_entry(s1, s0);
                            if (inc != 0) acc = ctx.add(acc, ctx.mul(inc, Y.at(s0 * d + a, col)));
                        }
                        if (acc != 0)
                            throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): runtime assertion (ii) failed -- d(Theta z) != 0");
                    }
                ++cocycle_checks_;
            }
        } else {
            cocycle_checks_ += t;
        }
        // (iii) Theta_eq: Theta C^q_eq in C^q_eq, and the exact residual.
        const DenseBuf<Elt>& C = coh_->module_degree(q).C;
        const uint32_t cq = C.cols;
        DenseBuf<Elt> TE(cq, cq);
        // (S23 G2-A) The read-off on the unit pivots of C (C[piv, :] = I, so C is injective and
        // the unique solution of C TE = W is TE = W[piv, :] -- solve_multi_rhs_module's reduced
        // representative, entry for entry), with the residual W - C TE recomputed exactly over
        // EVERY row ([C, obl:welldef](iii)), in row blocks.  Any other outcome (no unit pivots,
        // the identity check fails, a non-zero residual) takes the Howell solve below: the legacy
        // route, which on a non-zero residual throws the legacy message.  HECKE_MODULE_STREAM=0:
        // the Howell solve always.
        // (audit n3) W = T C once, for the read-off and (on its fallback) the Howell solve alike.
        DenseBuf<Elt> W;
        if (cq > 0) {
            W = DenseBuf<Elt>(dim, cq);
            engine::kernels::gemm<Elt>(ctx, engine::cview(T), engine::cview(C), engine::view(W));
            if (g_theta_eq_perturbation.load(std::memory_order_relaxed) != 0) {   // validation only (audit m3)
                const std::vector<uint32_t>& pv = coh_->module_degree(q).piv;
                uint32_t r = 0;
                for (uint32_t u = 0; u < pv.size() && pv[u] == r; ++u) ++r;   // the first non-pivot row of C
                if (r < dim) W.at(r, 0) = ctx.add(W.at(r, 0), Elt(1));    // C[piv, :] = I: e_r is off im C
            }
        }
        bool readoff_done = false;
        if (cq > 0 && engine::elim::module_stream_enabled()) {
            const std::vector<uint32_t>& piv = coh_->module_degree(q).piv;
            engine::elim::ReadoffStatus st = engine::elim::ReadoffStatus::NotIdentity;
            if (!piv.empty()) {
                TE = DenseBuf<Elt>();   // the read-off allocates TE itself (no second c_q x c_q buffer)
                st = engine::elim::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(C), piv, engine::cview(W), TE);
                if (st == engine::elim::ReadoffStatus::Ok) {
                    theta_eq_solves_ += cq;
                    readoff_done = true;
                } else {
                    TE = DenseBuf<Elt>(cq, cq);
                }
            }
            engine::elim::module_stream_count_solve(true, st, !piv.empty());
        } else if (cq > 0) {
            engine::elim::module_stream_count_solve(false, engine::elim::ReadoffStatus::NotIdentity, false);
        }
        if (cq > 0 && !readoff_done) {
            std::vector<std::vector<Elt>> co;
            try {
                co = engine::elim::solve_multi_rhs_module<Elt>(ctx, engine::cview(C), engine::cview(W));
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(std::string("HeckeOperatorOnK (Z/ell^k): runtime assertion (iii) failed -- ") + e.what());
            }
            for (uint32_t col = 0; col < cq; ++col)
                for (uint32_t j = 0; j < cq; ++j) TE.at(j, col) = co[col][j];
            DenseBuf<Elt> R(dim, cq);
            engine::kernels::gemm<Elt>(ctx, engine::cview(C), engine::cview(TE), engine::view(R));
            for (uint32_t a = 0; a < dim; ++a)
                for (uint32_t col = 0; col < cq; ++col) {
                    const Elt res = ctx.sub(W.at(a, col), R.at(a, col));
                    if (res > theta_eq_residual_max_) theta_eq_residual_max_ = res;
                }
            if (theta_eq_residual_max_ != 0)
                throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): the Theta_eq residual is not identically zero");
            theta_eq_solves_ += cq;
        }
        // A^{(q)} on the cyclic summands (c5).
        DenseBuf<Elt> A(t, t);
        if (t > 0) {
            const std::vector<std::vector<Elt>> co = coh_->H_class_coordinates(q, engine::cview(Y));
            for (uint32_t col = 0; col < t; ++col)
                for (uint32_t i = 0; i < t; ++i) A.at(i, col) = co[col][i];
        }
        A_module_[q] = std::move(A);
        Theta_module_[q] = std::move(T);
        Theta_eq_module_[q] = std::move(TE);
    }
    routes_bit_identical_ = routes_equal;
    theta_eq_certified_ = true;
}

// (S23 G2-C / G2-D / G2-E; slot23/G2_plan.md Sec. 2.3-2.5) THE STREAM ROUTE of compute_module() (HECKE_MODULE_STREAM
// on, the default).  The same objects as compute_module_legacy(), byte for byte, by three value-neutral changes:
//   G2-C  Theta^{(q)} is assembled coset-outer: Omega_i = rho_n(s_i^{-1}) is evaluated into ONE dense buffer per
//         (degree, coset), used for every sigma and dropped (no Omega table, no rho memo copy of it); the fused
//         memos are scoped to (degree, coset) (their key holds i).  On the U_ell low-rank path the terms of one
//         coset that land in one (sigma, sigma_k) block are accumulated by ONE GEMM [c_1 Omega_i U_1 | ...] .
//         [V_1; ...] (lowrank::sum_of_low_rank, GemmOp::Add) instead of the scalar loop.  Every entry of Theta is
//         an exact sum mod ell^k, so neither the order nor the grouping of its terms can change a byte.
//   G2-D  route 2 is built in sigma-row chunks (<= ceil(N_q / 2) block rows when N_q > 7, else all: <= 6 d x 11 d,
//         below the (iii) peak) from rho_n(delta_i gamma_k), memoised per (chunk, coset) only -- the keys depend on i
//         -- compared entrywise with Theta's rows and dropped: [C, obl:agreement](i) on every entry of every degree,
//         as before; routes_bit_identical_ keeps its meaning.  The dense additions go through kernels::axpy (the same
//         canonical products), and factor_low_rank's valuation scan multiplies by ell^{-1} mod 2^W instead of
//         dividing (lowrank::FastValuation, equal to ModCtx::valuation on every residue).
//   G2-E  (iii) is the read-off on the product (engine::elim::solve_unit_pivot_readoff_product): W = Theta C is
//         never materialised, only its non-pivot rows; the residual is recomputed over every row.  Under
//         ModuleRetention::Lean (set by RunModuleT only) Theta^{(q)} and Theta^{(q)}_eq are released after degree
//         q's certificates (their accessors throw).
// A malformed tau table (a term naming a simplex out of range) still throws its ENGINE_CHECK; route 2's throw is
// the legacy text.
template<class Elt>
void HeckeOperatorOnKT<Elt>::compute_module_stream() {
    using engine::DenseBuf;
    namespace el = engine::elim;
    const engine::ModCtx<Elt>& ctx = ring_ctx_;
    const Elt p = ctx.p;
    const SimplicialComplex* Kp = K();
    const uint32_t d = d_V_;
    auto key_of = [&](const Mat22& g) {
        std::array<Elt, 4> r;
        if constexpr (std::is_same_v<Elt, uint32_t>) { const auto a = g.reduced_mod(p); r = {a[0], a[1], a[2], a[3]}; }
        else { const auto a = g.reduced_mod64(p); r = {a[0], a[1], a[2], a[3]}; }
        return r;
    };
    // rho_n mod ell^k of a reduced key (the legacy rho(), without its memo).
    auto rho_eval = [&](const std::array<Elt, 4>& key) -> FlatMatrixT<Elt> {
        if (rep_ == FibreRepresentation::HidaLatticeH)
            return ComputeSymmetricPowerMatrixRing<Elt>(ctx, key[0], key[2], key[1], key[3], n_sym_).transpose();
        return ComputeSymmetricPowerMatrixRing<Elt>(ctx, key[0], key[1], key[2], key[3], n_sym_);
    };
    auto dense_of = [&](const FlatMatrixT<Elt>& M) {
        DenseBuf<Elt> b(M.get_rows(), M.get_cols());
        for (uint32_t i = 0; i < M.get_rows(); ++i)
            for (uint32_t j = 0; j < M.get_cols(); ++j) b.at(i, j) = M.get_entry(i, j);
        return b;
    };
    // (S23 G2-T) HECKE_MODULE_EVAL (DenseElim.h): the term evaluation straight into row-major buffers
    // (ComputeSymmetricPowerRowsRing from one Pascal row, no FlatMatrix carrier) and factor_low_rank's divisibility
    // scan.  Off: the G2-F evaluation above, verbatim.  The same canonical residues either way.
    const bool fast_eval = engine::elim::module_eval_enabled();
    const std::vector<Elt> pascal = fast_eval ? RingPascalRow<Elt>(ctx, n_sym_) : std::vector<Elt>();
    std::vector<Elt> rho_rows, rho_work;
    // rho_n(key) for the SymN fibre into rows (fast_eval only).  (S23 G2-P) The row / work buffers are arguments:
    // the serial route passes rho_rows / rho_work, each pool worker of the parallel route its own pair.
    auto rho_rows_of = [&](const std::array<Elt, 4>& key, std::vector<Elt>& rows, std::vector<Elt>& work) {
        ComputeSymmetricPowerRowsRing<Elt>(ctx, key[0], key[1], key[2], key[3], n_sym_, pascal, rows, work);
    };
    // the dense block of a key: rho_eval + dense_of, or (fast_eval, SymN) straight from the rows
    auto dense_rho = [&](const std::array<Elt, 4>& key, std::vector<Elt>& rho_rows, std::vector<Elt>& rho_work)
        -> DenseBuf<Elt> {
        if (!fast_eval || rep_ != FibreRepresentation::SymN) return dense_of(rho_eval(key));
        rho_rows_of(key, rho_rows, rho_work);
        DenseBuf<Elt> b(d, d);
        ENGINE_CHECK(b.ld == d && b.layout == engine::Layout::RowMajor);
        std::copy(rho_rows.begin(), rho_rows.end(), &b.at(0, 0));
        return b;
    };
    auto coeff_ring = [&](int32_t c) -> Elt {
        if (c >= 0) return static_cast<Elt>(static_cast<Elt>(c) % p);
        const Elt m = static_cast<Elt>(static_cast<Elt>(-static_cast<int64_t>(c)) % p);
        return m == 0 ? Elt(0) : static_cast<Elt>(p - m);
    };
    const bool low_rank = HeckeOperatorOptions::u_ell_low_rank_in_effect(cosets_->N(), static_cast<uint64_t>(ctx.ell), rep_);
    if (low_rank) {
        ENGINE_CHECK_MSG(rep_ == FibreRepresentation::SymN,
                         "HeckeOperatorOnK: the U_ell low-rank lever is defined for the SymN fibre only.");
        ENGINE_CHECK_MSG(cosets_->N() % static_cast<uint64_t>(ctx.ell) == 0,
                         "HeckeOperatorOnK: the U_ell low-rank lever needs ell | N (U_ell, T_{ell^r}).");
    }
    auto lr_eval = [&](const std::array<Elt, 4>& key, std::vector<Elt>& rho_rows, std::vector<Elt>& rho_work) {
        if (fast_eval) {   // (S23 G2-T) the low-rank path is SymN only (checked above)
            rho_rows_of(key, rho_rows, rho_work);
            // factor_low_rank works on its argument in place (an rvalue reference, never moved from): rho_rows
            // keeps its storage for the next key
            return factor_low_rank<Elt>(ctx, std::move(rho_rows), d, d, std::min<uint32_t>(ctx.k, d), /*fast_scan=*/true);
        }
        const FlatMatrixT<Elt> R = ComputeSymmetricPowerMatrixRing<Elt>(ctx, key[0], key[1], key[2], key[3], n_sym_);
        return factor_low_rank<Elt>(ctx, dense_rows_of<Elt>(R), d, d, std::min<uint32_t>(ctx.k, d));
    };
    // dst[r0.., c0..] += c B (kernels::axpy: the same canonical products c v, Shoup-form on u32)
    auto add_block = [&](DenseBuf<Elt>& dst, uint32_t r0, uint32_t c0, Elt c, const DenseBuf<Elt>& B) {
        engine::kernels::axpy<Elt>(ctx, c, engine::cview(B), engine::view(dst).sub(r0, d, c0, d), false);
    };
    using LrTerm = std::tuple<Elt, const std::vector<Elt>*, const std::vector<Elt>*, uint32_t>;
    // the grouped low-rank accumulation of one (sigma, sigma_k) block: dst[r0.., c0..] += sum c U V, one GEMM
    uint64_t grouped = 0, early_flush = 0;
    auto flush_groups = [&](DenseBuf<Elt>& dst, uint32_t r0, std::vector<std::vector<LrTerm>>& groups) {
        for (uint32_t sk = 0; sk < groups.size(); ++sk) {
            if (groups[sk].empty()) continue;
            lowrank::sum_of_low_rank<Elt>(ctx, d, groups[sk], engine::view(dst).sub(r0, d, sk * d, d), GemmOp::Add);
            ++grouped;
            groups[sk].clear();
        }
    };
    // one term into its (sigma, sigma_k) group; a group whose stacked width reaches kGroupCap is flushed at once, so
    // the stacked factors stay <= kGroupCap x d (one coset's chain can hold thousands of terms at ell ~ 10^4)
    constexpr uint32_t kGroupCap = 4096;
    auto push_group = [&](DenseBuf<Elt>& dst, uint32_t r0, std::vector<std::vector<LrTerm>>& groups, uint32_t sk, Elt c,
                          const LowRankT<Elt>& f) {
        if (f.t == 0) return;
        groups[sk].emplace_back(c, &f.U, &f.V, f.t);
        // every t <= min(k, d) (factor_low_rank's rank claim), so size * min(k, d) bounds the stacked width
        if (groups[sk].size() * std::min<uint32_t>(ctx.k, d) >= kGroupCap) {
            lowrank::sum_of_low_rank<Elt>(ctx, d, groups[sk], engine::view(dst).sub(r0, d, sk * d, d), GemmOp::Add);
            ++grouped;
            ++early_flush;   // (audit m2) counted, so the validation can assert the cap is reached
            groups[sk].clear();
        }
    };

    // ---- (S23 G2-P) THE PARALLEL ASSEMBLY (HECKE_MODULE_PARALLEL, DenseElim.h; on when the consumer pool has more
    // than one worker).  Route 1 (Theta^{(q)}) and route 2 (its sigma-row chunks) are assembled in batches of
    // consecutive cosets: (A) the calling thread walks the tau terms of the batch in the serial order and lists
    // the distinct keys per coset (the memo scope of the serial route: (degree, coset), resp. (chunk, coset));
    // (B) the pool evaluates Omega_i and every listed key (pure functions of the key: rho_n, factor_low_rank),
    // each worker into its own row buffers; (C) route 1 only: the fused products Omega_i U (low rank) or
    // Omega_i rho (dense), one task per key; (D) one task per sigma row writes ITS rows of the destination,
    // visiting the batch's cosets and their terms in the serial order with the serial grouping (groups flushed
    // per (coset, sigma); the kGroupCap flush at the same term).  So every GEMM / axpy call of the serial route
    // is made with the same operands on the same block, in the same order per block; the blocks of different
    // sigma rows are disjoint.  The counters are integer sums of per-task counts.  No value depends on the
    // thread count or the schedule (and every entry is an exact sum mod ell^k in any case).  (G2-P audit F3) Every
    // task body below carries el::ScopedInHookTask, so a callee that consults elim::parallel_hook() sees none on
    // the calling thread (worker index 0, which carries par_hook) as on the pool workers (which carry none).
    engine::tape::ThreadPool& par_pool = engine::tape::consumer_pool();
    const bool par = el::module_parallel_enabled() && par_pool.size() > 1;
    // the elimination hook for the Howell forms of (iii)'s fallback and the coordinates (DenseElim howell_rows)
    std::unique_ptr<engine::tape::ScopedHostElimHook> par_hook;
    if (par) par_hook.reset(new engine::tape::ScopedHostElimHook());
    const uint32_t par_threads = par ? par_pool.size() : 1;
    const bool mt_assembly = par && el::module_mt_enabled();   // (S24 MT-1) the (sigma, sigma_k) block tasks of (D)
    el::module_parallel_count(par, num_degrees_, par_pool.size());
    std::vector<std::vector<Elt>> w_rows(par_threads), w_work(par_threads);   // per pool worker of one region
    auto push_group_c = [&](DenseBuf<Elt>& dst, uint32_t r0, std::vector<std::vector<LrTerm>>& groups, uint32_t sk,
                            Elt c, const LowRankT<Elt>& f, uint64_t& g_cnt, uint64_t& ef_cnt) {
        if (f.t == 0) return;
        groups[sk].emplace_back(c, &f.U, &f.V, f.t);
        if (groups[sk].size() * std::min<uint32_t>(ctx.k, d) >= kGroupCap) {
            lowrank::sum_of_low_rank<Elt>(ctx, d, groups[sk], engine::view(dst).sub(r0, d, sk * d, d), GemmOp::Add);
            ++g_cnt;
            ++ef_cnt;
            groups[sk].clear();
        }
    };
    auto flush_groups_c = [&](DenseBuf<Elt>& dst, uint32_t r0, std::vector<std::vector<LrTerm>>& groups, uint64_t& g_cnt) {
        for (uint32_t sk = 0; sk < groups.size(); ++sk) {
            if (groups[sk].empty()) continue;
            lowrank::sum_of_low_rank<Elt>(ctx, d, groups[sk], engine::view(dst).sub(r0, d, sk * d, d), GemmOp::Add);
            ++g_cnt;
            groups[sk].clear();
        }
    };
    // a batch closes at >= 16 tasks per worker, or when its dense buffers (Omega_i, dense keys) reach 256 MiB
    const uint32_t par_target_tasks = 16 * par_threads;
    const size_t par_d2_bytes = std::max<size_t>(1, static_cast<size_t>(d) * d * sizeof(Elt));
    const uint32_t par_max_dense = static_cast<uint32_t>(std::max<size_t>(1, (size_t(256) << 20) / par_d2_bytes));
    uint64_t par_omega_evals = 0, par_grouped = 0, par_early_flush = 0;
    // route1: dst = Theta^{(q)}, rows sigma = s0 + sl at sl * d (s0 = 0, ns = N_q); else dst = route 2's chunk B2
    // (S24 G2-H) dst_s0: the sigma row of dst's row 0 (0 for Theta and for B2; s0 for route 2 IN PLACE on Theta);
    // negate: every term coefficient c enters as -c (route 2 subtracted from Theta, HECKE_MODULE_LEAN_PEAK (c)).
    auto par_assemble = [&](uint32_t q, uint32_t N_q, uint32_t s0, uint32_t ns, bool route1, DenseBuf<Elt>& dst,
                            uint32_t dst_s0 = 0, bool negate = false) {
        struct PTerm { Elt c; uint32_t sk; uint32_t slot; };
        uint32_t i0 = 1;
        while (i0 <= m_) {
            // (A) serial: the batch's terms and distinct keys, per coset in first-appearance order
            std::vector<std::array<Elt, 4>> keys;
            std::vector<uint32_t> key_b;               // slot -> batch-local coset
            std::vector<uint32_t> cos;                 // batch-local coset -> i
            std::vector<std::vector<PTerm>> terms;     // (b, sl) -> the terms, in the serial order
            while (i0 <= m_) {
                const uint32_t i = i0++;
                const uint32_t b = static_cast<uint32_t>(cos.size());
                cos.push_back(i);
                terms.resize(static_cast<size_t>(b + 1) * ns);
                std::map<std::array<Elt, 4>, uint32_t> idx;
                for (uint32_t sl = 0; sl < ns; ++sl) {
                    const std::vector<ChainTerm>& cts = tau_->tau(i, q, s0 + sl);
                    for (const ChainTerm& ct : cts) {
                        const Elt c = coeff_ring(ct.coeff);
                        if (c == 0) continue;
                        ENGINE_CHECK_MSG(ct.sigma_idx < N_q, "HeckeOperatorOnK: tau term names a simplex out of range");
                        const Mat22 gg = route1 ? cosets_->g() * ct.gamma : cosets_->delta(i) * ct.gamma;   // exact
                        const std::array<Elt, 4> key = key_of(gg);
                        const auto ins = idx.emplace(key, static_cast<uint32_t>(keys.size()));
                        if (ins.second) { keys.push_back(key); key_b.push_back(b); }
                        terms[static_cast<size_t>(b) * ns + sl].push_back(
                            PTerm{negate ? ctx.neg(c) : c, ct.sigma_idx, ins.first->second});
                    }
                }
                const size_t ntask = keys.size() + (route1 ? cos.size() : 0);
                const size_t ndense = (route1 ? cos.size() : 0) + (low_rank ? 0 : keys.size());
                if (ntask >= par_target_tasks || ndense >= par_max_dense) break;
            }
            const uint32_t nb = static_cast<uint32_t>(cos.size());
            const uint32_t nk = static_cast<uint32_t>(keys.size());
            const uint32_t nom = route1 ? nb : 0;
            std::vector<DenseBuf<Elt>> om(nom), dn(low_rank ? 0 : nk);
            std::vector<LowRankT<Elt>> lr(low_rank ? nk : 0);
            // (B) Omega_i and the keys
            par_pool.parallel_for(nom + nk, par_threads, [&](uint32_t t, uint32_t w) {
                const el::ScopedInHookTask in_task;   // (G2-P audit F3) no nested hook region from a task, caller or worker
                if (t < nom) { om[t] = dense_rho(key_of(cosets_->s_inv(cos[t])), w_rows[w], w_work[w]); return; }
                const uint32_t s = t - nom;
                if (low_rank) lr[s] = lr_eval(keys[s], w_rows[w], w_work[w]);
                else dn[s] = dense_rho(keys[s], w_rows[w], w_work[w]);
            });
            // (C) route 1: the fused products, as the serial memo forms them
            if (route1) {
                par_omega_evals += nb;
                par_pool.parallel_for(nk, par_threads, [&](uint32_t s, uint32_t) {
                    const el::ScopedInHookTask in_task;   // (G2-P audit F3) no nested hook region from a task, caller or worker
                    const DenseBuf<Elt>& Om = om[key_b[s]];
                    if (low_rank) {
                        const LowRankT<Elt>& l = lr[s];
                        LowRankT<Elt> f;
                        f.rows = d; f.cols = d; f.t = l.t; f.V = l.V;
                        f.U.assign(static_cast<size_t>(d) * l.t, Elt(0));
                        if (l.t > 0) {
                            DenseBuf<Elt> Ub(const_cast<Elt*>(l.U.data()), l.U.size(), d, l.t, engine::Layout::RowMajor);
                            DenseBuf<Elt> Wb(f.U.data(), f.U.size(), d, l.t, engine::Layout::RowMajor);
                            engine::kernels::gemm<Elt>(ctx, engine::cview(Om), engine::cview(Ub), engine::view(Wb));
                        }
                        lr[s] = std::move(f);
                    } else {
                        DenseBuf<Elt> P(d, d);
                        engine::kernels::gemm<Elt>(ctx, engine::cview(Om), engine::cview(dn[s]), engine::view(P));
                        dn[s] = std::move(P);
                    }
                });
                om.clear();
            }
            // (D) one task per sigma row: its rows of dst, the batch's cosets and terms in the serial order
            // (S24 MT-1) HECKE_MODULE_MT: one task per (sigma row, sigma_k block) -- each block is written only by the
            // terms with that sigma_k, and the per-row loop visits them in this order with groups per sigma_k, so a
            // block's task makes the same calls in the same order; the blocks are disjoint.
            const bool mt_blocks = mt_assembly;
            const uint32_t nsk = mt_blocks ? N_q : 1u;
            if (mt_blocks) el::module_mt_count(el::ModuleMtEvent::Assembly);
            std::vector<uint64_t> g_cnt(static_cast<size_t>(ns) * nsk, 0), ef_cnt(static_cast<size_t>(ns) * nsk, 0);
            par_pool.parallel_for(ns * nsk, par_threads, [&](uint32_t task, uint32_t) {
                const el::ScopedInHookTask in_task;   // (G2-P audit F3) no nested hook region from a task, caller or worker
                const uint32_t sl = task / nsk, skb = task % nsk;
                std::vector<std::vector<LrTerm>> groups(low_rank ? N_q : 0);
                const uint32_t r0 = (dst_s0 + sl) * d;
                for (uint32_t b = 0; b < nb; ++b) {
                    for (const PTerm& pt : terms[static_cast<size_t>(b) * ns + sl]) {
                        if (mt_blocks && pt.sk != skb) continue;
                        if (low_rank) push_group_c(dst, r0, groups, pt.sk, pt.c, lr[pt.slot], g_cnt[task], ef_cnt[task]);
                        else add_block(dst, r0, pt.sk * d, pt.c, dn[pt.slot]);
                    }
                    if (low_rank) flush_groups_c(dst, r0, groups, g_cnt[task]);
                }
            });
            for (size_t x = 0; x < g_cnt.size(); ++x) { par_grouped += g_cnt[x]; par_early_flush += ef_cnt[x]; }
        }
    };

    const bool lean = retention_ == ModuleRetention::Lean;
    A_module_.assign(num_degrees_, DenseBuf<Elt>());
    Theta_module_.assign(num_degrees_, DenseBuf<Elt>());
    Theta_eq_module_.assign(num_degrees_, DenseBuf<Elt>());
    released_ = false;
    relation_checks_ = cocycle_checks_ = theta_eq_solves_ = 0;
    theta_eq_residual_max_ = 0;
    bool routes_equal = true;
    const auto& rels = coh_->get_lssc()->get_user_defined_simplex_relations();
    uint64_t omega_evals = 0, route2_blocks = 0, w_np = 0;

    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = Kp->get_num_q_simplices(q);
        const uint32_t dim = N_q * d;
        DenseBuf<Elt> T(dim, dim);
        // ---- G2-C: Theta^{(q)}, coset-outer --------------------------------------------------------------------
        {
            std::vector<std::vector<LrTerm>> groups(low_rank ? N_q : 0);
            if (par) par_assemble(q, N_q, 0, N_q, /*route1=*/true, T);   // (S23 G2-P)
            else for (uint32_t i = 1; i <= m_; ++i) {
                const DenseBuf<Elt> Om = dense_rho(key_of(cosets_->s_inv(i)), rho_rows, rho_work);
                ++omega_evals;
                std::map<std::array<Elt, 4>, DenseBuf<Elt>> fused;      // key(g gamma) -> Omega_i rho(g gamma)
                std::map<std::array<Elt, 4>, LowRankT<Elt>> fused_lr;  // key(g gamma) -> (Omega_i U) V
                for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
                    const std::vector<ChainTerm>& cts = tau_->tau(i, q, sigma);
                    for (const ChainTerm& ct : cts) {
                        const Elt c = coeff_ring(ct.coeff);
                        if (c == 0) continue;
                        ENGINE_CHECK_MSG(ct.sigma_idx < N_q, "HeckeOperatorOnK: tau term names a simplex out of range");
                        const Mat22 gg = cosets_->g() * ct.gamma;              // exact (I6); never inverted (I7)
                        const std::array<Elt, 4> key = key_of(gg);
                        if (low_rank) {
                            auto jt = fused_lr.find(key);
                            if (jt == fused_lr.end()) {
                                const LowRankT<Elt> lr = lr_eval(key, rho_rows, rho_work);
                                LowRankT<Elt> f;
                                f.rows = d; f.cols = d; f.t = lr.t; f.V = lr.V;
                                f.U.assign(static_cast<size_t>(d) * lr.t, Elt(0));
                                if (lr.t > 0) {
                                    DenseBuf<Elt> Ub(const_cast<Elt*>(lr.U.data()), lr.U.size(), d, lr.t, engine::Layout::RowMajor);
                                    DenseBuf<Elt> Wb(f.U.data(), f.U.size(), d, lr.t, engine::Layout::RowMajor);
                                    engine::kernels::gemm<Elt>(ctx, engine::cview(Om), engine::cview(Ub), engine::view(Wb));
                                }
                                jt = fused_lr.emplace(key, std::move(f)).first;
                            }
                            push_group(T, sigma * d, groups, ct.sigma_idx, c, jt->second);
                            continue;
                        }
                        auto it = fused.find(key);
                        if (it == fused.end()) {
                            const DenseBuf<Elt> R = dense_rho(key, rho_rows, rho_work);
                            DenseBuf<Elt> P(d, d);
                            engine::kernels::gemm<Elt>(ctx, engine::cview(Om), engine::cview(R), engine::view(P));
                            it = fused.emplace(key, std::move(P)).first;
                        }
                        add_block(T, sigma * d, ct.sigma_idx * d, c, it->second);
                    }
                    if (low_rank) flush_groups(T, sigma * d, groups);
                }
            }
        }
        // (S24 G2-H) HECKE_MODULE_LEAN_PEAK (b) + (c), under the lean retention only: (iii) by the residual alone (no
        // Theta_eq) and route 2 subtracted from Theta in place AFTER Y, (i)-(iii) (no route-2 buffer).  The certificate
        // "Theta == route 2 entrywise" becomes "Theta - route 2 == 0 entrywise": every entry is the same exact sum mod
        // ell^k, so the two predicates agree on every input.  (A cell failing both O12(a) and (i)-(iii) would now report
        // (i)-(iii) first; no such failure is known, and every single-fault control reports its legacy text.)
        const bool g2h = lean && el::module_lean_peak_enabled();
        // ---- G2-D: route 2 in sigma-row chunks (at most ceil(N_q / 2) rows of blocks when N_q > 7, else all N_q:
        // <= 6 d x 11 d at q = 1, below the (iii) peak), built coset by coset with a memo scoped to (chunk, coset)
        // -- the delta_i gamma_k of one coset repeat across sigma -- compared entrywise with Theta's rows, dropped.
        // (S24 G2-H) in_place: the chunk's terms enter T's own rows with coefficient -c, and the rows must then be 0.
        auto run_route2 = [&](bool in_place) {
            // (S23 G2-T2) HECKE_ROUTE2_SHARED_MEMO (DenseElim.h, default on): one chunk of all N_q rows, so the memo
            // below spans (degree, coset); off: G2-D's two chunks when N_q > 7, verbatim.
            const bool shared_memo = N_q > 7 && el::route2_shared_memo_enabled();
            if (N_q > 7) el::route2_memo_count(shared_memo);
            const uint32_t ns_max = (N_q > 7 && !shared_memo) ? (N_q + 1) / 2 : N_q;
            DenseBuf<Elt> B2 = in_place ? DenseBuf<Elt>() : DenseBuf<Elt>(ns_max * d, dim);
            std::vector<std::vector<LrTerm>> groups(low_rank ? N_q : 0);
            for (uint32_t s0 = 0; s0 < N_q && routes_equal; s0 += ns_max) {
                const uint32_t ns = std::min(ns_max, N_q - s0);
                DenseBuf<Elt>& dst = in_place ? T : B2;
                const uint32_t row0 = in_place ? s0 : 0;   // dst's sigma row of chunk row 0
                if (!in_place)
                    for (uint32_t a = 0; a < ns * d; ++a)
                        for (uint32_t b = 0; b < dim; ++b) B2.at(a, b) = 0;
                if (par) par_assemble(q, N_q, s0, ns, /*route1=*/false, dst, row0, in_place);   // (S23 G2-P)
                else for (uint32_t i = 1; i <= m_; ++i) {
                    std::map<std::array<Elt, 4>, DenseBuf<Elt>> r2;        // key(delta_i gamma) -> rho, this chunk and coset
                    std::map<std::array<Elt, 4>, LowRankT<Elt>> r2_lr;    // key(delta_i gamma) -> U V
                    for (uint32_t sl = 0; sl < ns; ++sl) {
                        const uint32_t sigma = s0 + sl;
                        const std::vector<ChainTerm>& cts = tau_->tau(i, q, sigma);
                        for (const ChainTerm& ct : cts) {
                            const Elt c0 = coeff_ring(ct.coeff);
                            if (c0 == 0) continue;
                            const Elt c = in_place ? ctx.neg(c0) : c0;
                            ENGINE_CHECK_MSG(ct.sigma_idx < N_q, "HeckeOperatorOnK: tau term names a simplex out of range");
                            // route 2: n_k rho(delta_i gamma_k), the exact product first
                            const Mat22 dg = cosets_->delta(i) * ct.gamma;
                            const std::array<Elt, 4> key = key_of(dg);
                            if (low_rank) {
                                auto jt = r2_lr.find(key);
                                if (jt == r2_lr.end()) jt = r2_lr.emplace(key, lr_eval(key, rho_rows, rho_work)).first;
                                push_group(dst, (row0 + sl) * d, groups, ct.sigma_idx, c, jt->second);
                                continue;
                            }
                            auto it = r2.find(key);
                            if (it == r2.end()) it = r2.emplace(key, dense_rho(key, rho_rows, rho_work)).first;
                            add_block(dst, (row0 + sl) * d, ct.sigma_idx * d, c, it->second);
                        }
                        if (low_rank) flush_groups(dst, (row0 + sl) * d, groups);
                    }
                }
                if (ns > 0) {   // validation only (ScopedRoute2Perturbation; audit m1: a site in each chunk of q = 1)
                    // route 2's entry + 1; in place that is Theta - route 2's entry - 1
                    const int site = g_route2_perturbation.load(std::memory_order_relaxed);
                    uint32_t pr = ~0u, pc = 0;
                    if (((q == 0 && site == 1) || (q == 1 && site == 3)) && s0 + ns == N_q) { pr = ns * d - 1; pc = dim - 1; }
                    else if (q == 1 && site == 2 && s0 == 0) { pr = 0; pc = 0; }
                    else if (q == 1 && site == 4 && N_q > 3 && s0 <= 3 && 3 < s0 + ns) {   // (G2-H audit F2) interior row
                        pr = (3 - s0) * d + d / 2; pc = dim / 2;
                    }
                    if (pr != ~0u) {
                        if (in_place) T.at(s0 * d + pr, pc) = ctx.sub(T.at(s0 * d + pr, pc), Elt(1));
                        else B2.at(pr, pc) = ctx.add(B2.at(pr, pc), Elt(1));
                    }
                }
                if (in_place) {
                    for (uint32_t a = 0; a < ns * d && routes_equal; ++a)
                        for (uint32_t b = 0; b < dim; ++b)
                            if (T.at(s0 * d + a, b) != 0) { routes_equal = false; break; }
                } else {
                    for (uint32_t a = 0; a < ns * d && routes_equal; ++a)
                        for (uint32_t b = 0; b < dim; ++b)
                            if (T.at(s0 * d + a, b) != B2.at(a, b)) { routes_equal = false; break; }
                }
                route2_blocks += ns;
            }
            if (in_place) el::lean_peak_count(el::LeanPeakEvent::Route2InPlace);
            if (!routes_equal)
                throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): the fused route and route 2 differ at q=" +
                                         std::to_string(q) + " (O12(a) over Z/ell^k)");
        };
        if (!g2h) run_route2(false);
        // Y = Theta . H (the cyclic-summand basis), (i), (ii) -- as compute_module_legacy().
        const DenseBuf<Elt>& Hb = coh_->H_module_basis(q);
        const uint32_t t = Hb.cols;
        DenseBuf<Elt> Y(dim, t);
        // (S24 MT-1) HECKE_MODULE_MT: Y as row-block products over the hook (bit-identical, ENGINE_API Sec. 5)
        const el::ParallelFor* mt_par = el::module_mt_hook();
        if (dim > 0 && t > 0) {
            if (mt_par != nullptr) el::gemm_rows_hook<Elt>(mt_par, ctx, engine::cview(T), engine::cview(Hb), engine::view(Y));
            else engine::kernels::gemm<Elt>(ctx, engine::cview(T), engine::cview(Hb), engine::view(Y));
        }
        // (S24 MT-1) (i)/(ii) per column on the pool (pure reads of Y, the relations and D^{q+1}): when every column
        // holds, the serial loops below are skipped and the counters advance by exactly what they would have counted;
        // on any failure the serial loops run and throw their own first message.
        bool checks_done = false;
        if (mt_par != nullptr && t >= 2) {
            const bool has_d1 = q + 1 < num_degrees_;
            const FlatMatrixT<Elt> D1p = has_d1 ? Kp->template coboundary_operator_elt<Elt>(q, p) : FlatMatrixT<Elt>();
            const uint32_t N1p = has_d1 ? Kp->get_num_q_simplices(q + 1) : 0;
            uint64_t nrel = 0;
            for (const auto& rel : rels) if (rel.q_dim == q) ++nrel;
            std::vector<uint8_t> col_ok(t, 1);
            el::module_mt_count(el::ModuleMtEvent::Check);
            par_pool.parallel_for(t, par_threads, [&](uint32_t col, uint32_t) {
                const el::ScopedInHookTask in_task;
                for (const auto& rel : rels) {
                    if (rel.q_dim != q) continue;
                    for (uint32_t a = 0; a < d; ++a) {
                        Elt acc = Y.at(rel.s_lhs_idx_in_Kq * d + a, col);
                        for (uint32_t b = 0; b < d; ++b) {
                            const Elt mv = rel.M.get_entry(a, b);
                            if (mv != 0) acc = ctx.sub(acc, ctx.mul(mv, Y.at(rel.s_rhs_idx_in_Kq * d + b, col)));
                        }
                        if (acc != 0) { col_ok[col] = 0; return; }
                    }
                }
                for (uint32_t s1 = 0; s1 < N1p; ++s1)
                    for (uint32_t a = 0; a < d; ++a) {
                        Elt acc = 0;
                        for (uint32_t s0 = 0; s0 < N_q; ++s0) {
                            const Elt inc = D1p.get_entry(s1, s0);
                            if (inc != 0) acc = ctx.add(acc, ctx.mul(inc, Y.at(s0 * d + a, col)));
                        }
                        if (acc != 0) { col_ok[col] = 0; return; }
                    }
            });
            bool all_ok = true;
            for (uint8_t o : col_ok) all_ok = all_ok && o != 0;
            if (all_ok) {
                relation_checks_ += nrel * t;
                cocycle_checks_ += t;
                checks_done = true;
            }
        }
        if (!checks_done) {
        for (const auto& rel : rels) {
            if (rel.q_dim != q) continue;
            for (uint32_t col = 0; col < t; ++col) {
                for (uint32_t a = 0; a < d; ++a) {
                    Elt acc = Y.at(rel.s_lhs_idx_in_Kq * d + a, col);
                    for (uint32_t b = 0; b < d; ++b) {
                        const Elt mv = rel.M.get_entry(a, b);
                        if (mv != 0) acc = ctx.sub(acc, ctx.mul(mv, Y.at(rel.s_rhs_idx_in_Kq * d + b, col)));
                    }
                    if (acc != 0)
                        throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): runtime assertion (i) failed -- Theta z "
                                                 "violates relation '" + rel.tag + "'");
                }
                ++relation_checks_;
            }
        }
        if (q + 1 < num_degrees_) {
            const FlatMatrixT<Elt> D1 = Kp->template coboundary_operator_elt<Elt>(q, p);
            const uint32_t N1 = Kp->get_num_q_simplices(q + 1);
            for (uint32_t col = 0; col < t; ++col) {
                for (uint32_t s1 = 0; s1 < N1; ++s1)
                    for (uint32_t a = 0; a < d; ++a) {
                        Elt acc = 0;
                        for (uint32_t s0 = 0; s0 < N_q; ++s0) {
                            const Elt inc = D1.get_entry(s1, s0);
                            if (inc != 0) acc = ctx.add(acc, ctx.mul(inc, Y.at(s0 * d + a, col)));
                        }
                        if (acc != 0)
                            throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): runtime assertion (ii) failed -- d(Theta z) != 0");
                    }
                ++cocycle_checks_;
            }
        } else {
            cocycle_checks_ += t;
        }
        }   // (S24 MT-1) !checks_done
        // (iii) Theta_eq: Theta C^q_eq in C^q_eq, and the exact residual.  G2-E: the read-off on the product
        // Theta C (W never materialised, only its non-pivot rows; the residual over every row); on any other
        // outcome W = Theta C is formed and the Howell solve below runs -- the legacy route and its texts.
        const DenseBuf<Elt>& C = coh_->module_degree(q).C;
        const std::vector<uint32_t>& piv = coh_->module_degree(q).piv;
        const uint32_t cq = C.cols;
        DenseBuf<Elt> TE = g2h ? DenseBuf<Elt>() : DenseBuf<Elt>(cq, cq);   // (S24 G2-H) never formed on the residual route
        // validation only (audit m3): W gets +1 at the first non-pivot row of C, column 0 (e_r is off im C)
        uint32_t perturb_row = ~0u;
        if (cq > 0 && g_theta_eq_perturbation.load(std::memory_order_relaxed) != 0) {
            uint32_t r = 0;
            for (uint32_t u = 0; u < piv.size() && piv[u] == r; ++u) ++r;
            if (r < dim) perturb_row = r;
        }
        bool readoff_done = false;
        if (cq > 0) {
            el::ReadoffStatus st = el::ReadoffStatus::NotIdentity;
            if (!piv.empty() && g2h) {
                // (S24 G2-H) (b): the read-off's certificate alone; on any other outcome the legacy fallback below
                st = el::unit_pivot_residual_product<Elt>(ctx, engine::cview(C), piv, engine::cview(T), engine::cview(C),
                                                          256, perturb_row);
                if (st == el::ReadoffStatus::Ok) {
                    theta_eq_solves_ += cq;
                    readoff_done = true;
                    el::lean_peak_count(el::LeanPeakEvent::Residual);
                }
            } else if (!piv.empty()) {
                TE = DenseBuf<Elt>();   // the read-off allocates TE itself
                st = el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(C), piv, engine::cview(T),
                                                              engine::cview(C), TE, 256, perturb_row);
                if (st == el::ReadoffStatus::Ok) {
                    theta_eq_solves_ += cq;
                    readoff_done = true;
                    ++w_np;
                } else {
                    TE = DenseBuf<Elt>(cq, cq);
                }
            }
            el::module_stream_count_solve(true, st, !piv.empty());
        }
        if (cq > 0 && !readoff_done) {
            if (g2h) TE = DenseBuf<Elt>(cq, cq);   // (S24 G2-H) the fallback, verbatim
            DenseBuf<Elt> W(dim, cq);
            if (mt_par != nullptr) el::gemm_rows_hook<Elt>(mt_par, ctx, engine::cview(T), engine::cview(C), engine::view(W));   // (S24 MT-1)
            else engine::kernels::gemm<Elt>(ctx, engine::cview(T), engine::cview(C), engine::view(W));
            if (perturb_row != ~0u) W.at(perturb_row, 0) = ctx.add(W.at(perturb_row, 0), Elt(1));
            std::vector<std::vector<Elt>> co;
            try {
                co = el::solve_multi_rhs_module<Elt>(ctx, engine::cview(C), engine::cview(W));
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(std::string("HeckeOperatorOnK (Z/ell^k): runtime assertion (iii) failed -- ") + e.what());
            }
            for (uint32_t col = 0; col < cq; ++col)
                for (uint32_t j = 0; j < cq; ++j) TE.at(j, col) = co[col][j];
            DenseBuf<Elt> R(dim, cq);
            if (mt_par != nullptr) el::gemm_rows_hook<Elt>(mt_par, ctx, engine::cview(C), engine::cview(TE), engine::view(R));   // (S24 MT-1)
            else engine::kernels::gemm<Elt>(ctx, engine::cview(C), engine::cview(TE), engine::view(R));
            for (uint32_t a = 0; a < dim; ++a)
                for (uint32_t col = 0; col < cq; ++col) {
                    const Elt res = ctx.sub(W.at(a, col), R.at(a, col));
                    if (res > theta_eq_residual_max_) theta_eq_residual_max_ = res;
                }
            if (theta_eq_residual_max_ != 0)
                throw std::runtime_error("HeckeOperatorOnK (Z/ell^k): the Theta_eq residual is not identically zero");
            theta_eq_solves_ += cq;
        }
        // (S24 G2-H) (c): route 2 against Theta, in place, now that Y and (iii) have read Theta
        if (g2h) run_route2(true);
        // G2-E: nothing below reads Theta^{(q)} or Theta^{(q)}_eq (Y is formed): release them before the
        // coordinates' solve.
        if (lean) { T = DenseBuf<Elt>(); TE = DenseBuf<Elt>(); }
        // A^{(q)} on the cyclic summands (c5).
        DenseBuf<Elt> A(t, t);
        if (t > 0) {
            const std::vector<std::vector<Elt>> co = coh_->H_class_coordinates(q, engine::cview(Y));
            for (uint32_t col = 0; col < t; ++col)
                for (uint32_t i = 0; i < t; ++i) A.at(i, col) = co[col][i];
        }
        A_module_[q] = std::move(A);
        if (!lean) {
            Theta_module_[q] = std::move(T);
            Theta_eq_module_[q] = std::move(TE);
        }
    }
    routes_bit_identical_ = routes_equal;
    theta_eq_certified_ = true;
    released_ = lean;
    omega_evals += par_omega_evals; grouped += par_grouped; early_flush += par_early_flush;   // (S23 G2-P)
    el::module_stream_count(el::ModuleStreamEvent::OmegaStreamed, omega_evals);
    el::module_stream_count(el::ModuleStreamEvent::Route2Blocked, route2_blocks);
    el::module_stream_count(el::ModuleStreamEvent::LrGrouped, grouped);
    el::module_stream_count(el::ModuleStreamEvent::LrEarlyFlush, early_flush);
    el::module_stream_count(el::ModuleStreamEvent::WNonpivot, w_np);
    if (lean) el::module_stream_count(el::ModuleStreamEvent::LeanReleased, 1);
    el::module_stream_count(fast_eval ? el::ModuleStreamEvent::EvalFast : el::ModuleStreamEvent::EvalLegacy, num_degrees_);
}

template<class Elt>
const engine::DenseBuf<Elt>& HeckeOperatorOnKT<Elt>::H_matrix_module(uint32_t q) const {
    require_module_op("H_matrix_module", q);
    return A_module_[q];
}

template<class Elt>
const engine::DenseBuf<Elt>& HeckeOperatorOnKT<Elt>::Theta_module(uint32_t q) const {
    require_module_op("Theta_module", q);
    ENGINE_CHECK_MSG(!released_, "HeckeOperatorOnK::Theta_module: released under ModuleRetention::Lean (S23 G2-E: "
                                 "the sat runner keeps A^{(q)} and the certificates only; use ModuleRetention::Full)");
    return Theta_module_[q];
}

template<class Elt>
const engine::DenseBuf<Elt>& HeckeOperatorOnKT<Elt>::Theta_eq_module(uint32_t q) const {
    require_module_op("Theta_eq_module", q);
    ENGINE_CHECK_MSG(!released_, "HeckeOperatorOnK::Theta_eq_module: released under ModuleRetention::Lean (S23 G2-E: "
                                 "the sat runner keeps A^{(q)} and the certificates only; use ModuleRetention::Full)");
    return Theta_eq_module_[q];
}

// Berkowitz (division-free) over the ring of `ctx`: det(xI - A), low -> high.
namespace {
template<class Elt>
std::vector<Elt> charpoly_division_free(const engine::ModCtx<Elt>& ctx, const std::vector<std::vector<Elt>>& A) {
    const uint32_t n = static_cast<uint32_t>(A.size());
    std::vector<Elt> pc{Elt(1)};                          // high -> low, for the leading r x r block
    for (uint32_t r = 1; r <= n; ++r) {
        const uint32_t m = r - 1;                         // the new row / column index
        std::vector<Elt> t(r + 1, Elt(0));
        t[0] = 1;
        t[1] = ctx.neg(A[m][m]);
        std::vector<Elt> v(m);                            // v = M^j c, M the leading m x m block
        for (uint32_t i = 0; i < m; ++i) v[i] = A[i][m];
        for (uint32_t kk = 2; kk <= r; ++kk) {
            Elt acc = 0;                                  // R v
            for (uint32_t i = 0; i < m; ++i) acc = ctx.add(acc, ctx.mul(A[m][i], v[i]));
            t[kk] = ctx.neg(acc);
            std::vector<Elt> nv(m, Elt(0));
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < m; ++j) nv[i] = ctx.add(nv[i], ctx.mul(A[i][j], v[j]));
            v.swap(nv);
        }
        std::vector<Elt> np(r + 1, Elt(0));
        for (uint32_t i = 0; i <= r; ++i)
            for (uint32_t j = 0; j < r && j <= i; ++j) np[i] = ctx.add(np[i], ctx.mul(t[i - j], pc[j]));
        pc.swap(np);
    }
    std::reverse(pc.begin(), pc.end());
    return pc;
}
} // namespace

template<class Elt>
typename HeckeOperatorOnKT<Elt>::FreeCharpoly HeckeOperatorOnKT<Elt>::free_charpoly_H1() const {
    require_module_op("free_charpoly_H1", 1);
    if (!coh_->ladic_applicable() || !coh_->ladic_type_certified())
        throw std::logic_error(
            "HeckeOperatorOnK::free_charpoly_H1: charpoly_A(T_p)^1_free is defined only at a certified K-mode "
            "([E] sec:ring (d)); it is absent here and has no value");
    const uint32_t K = ring_k_, e_max = coh_->ladic_e_max();
    ENGINE_CHECK(e_max < K);
    FreeCharpoly out;
    out.precision = K - e_max;
    out.modulus = static_cast<uint64_t>(ring_ctx_.ell_pow(out.precision));
    const std::vector<uint32_t>& ex = coh_->H_invariant_factors(1);
    uint32_t r = 0;
    while (r < ex.size() && ex[r] == K) ++r;              // the free summands come first (descending order)
    out.rank = r;
    const DenseBuf<Elt>& A = A_module_[1];
    std::vector<std::vector<Elt>> B(r, std::vector<Elt>(r));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) B[i][j] = A.at(i, j);
    // Over Z/ell^K, then reduced: reduction to Z/ell^{K - e_max} is a ring map.
    out.coeffs = charpoly_division_free<Elt>(ring_ctx_, B);
    for (Elt& c : out.coeffs) c = static_cast<Elt>(c % static_cast<Elt>(out.modulus));
    return out;
}

template<class Elt>
const FlatMatrixT<Elt>& HeckeOperatorOnKT<Elt>::omega_of(uint32_t i) const {
    if (rep_ == FibreRepresentation::SymN) {
#if OC_ENABLE_LEGACY
        if (!cosets_->streamed()) return cosets_->rho_s_inv(i);   // (S24 CL-R1) a materialised table (ON builds only)
#endif
        // (slot-14 WP-10) A streamed coset table: the operator's own Omega
        // table (ensure_omega_stream(), materialised before any reader).
        ENGINE_CHECK_MSG(i >= 1 && i < omega_stream_.size(),
                         "HeckeOperatorOnK: the streamed Omega table is not materialised (internal error) or the "
                         "coset index is out of range");
        return omega_stream_[i];
    }
    ENGINE_CHECK_MSG(i >= 1 && i < omega_dual_.size(), "HeckeOperatorOnK: coset index out of range");
    return omega_dual_[i];
}

// (slot-14 WP-10) The coset streaming lever: the Omega_i = rho_n(s_i^{-1}) of a
// streamed coset table, materialised ONCE per operator for the field paths
// that borrow them for the whole computation (the tape's program-1 operands,
// the legacy arm); each block is the coset table's own rho_of(s_i^{-1}), i.e.
// the bytes the default table stores.  Under ScopedParallelSetup the blocks
// are evaluated on the consumer pool (pre-assigned slots).
template<class Elt>
void HeckeOperatorOnKT<Elt>::ensure_omega_stream() {
    if (rep_ != FibreRepresentation::SymN || !cosets_->streamed() || omega_stream_.size() == m_ + 1) return;
    std::vector<FlatMatrixT<Elt>> om(static_cast<size_t>(m_) + 1, FlatMatrixT<Elt>(0, 0, ell_));
    HeckeOperatorOptions::for_each_index(m_, HeckeOperatorOptions::parallel_setup(), [&](size_t j) {
        const uint32_t i = static_cast<uint32_t>(j) + 1;
        om[i] = cosets_->rho_s_inv_block(i).take();   // (audit M-4) a streamed block is moved, not copied
    }, static_cast<uint64_t>(d_V_) * d_V_);
    omega_stream_ = std::move(om);
}

template<class Elt>
const FlatMatrixT<Elt>& HeckeOperatorOnKT<Elt>::rho_of_term(const XiTermT<Elt>& xt) const {
    if (rep_ == FibreRepresentation::SymN) return xt.rho_g_gamma.get();
    ENGINE_CHECK_MSG(xt.rho_g_gamma.index() < rho_pool_dual_.size(), "HeckeOperatorOnK: rho-pool slot out of range");
    return rho_pool_dual_[xt.rho_g_gamma.index()];
}

template<class Elt>
const SimplicialComplex* HeckeOperatorOnKT<Elt>::K() const {
    return coh_->get_lssc()->get_base_K_complex();
}

// rho_n(gamma_exact mod ell) via ComputeSymmetricPowerMatrixFM (invariant I2).
// gamma_exact may have determinant N; it is never inverted (invariant I7).
template<class Elt>
FlatMatrixT<Elt> HeckeOperatorOnKT<Elt>::rho_of_exact(const Mat22& gamma_exact) const {
    std::array<Elt, 4> g_mod;
    if constexpr (std::is_same_v<Elt, uint32_t>) {
        g_mod = gamma_exact.reduced_mod(ell_);
    } else {
        g_mod = gamma_exact.reduced_mod64(ell_);
    }
    const FlatMatrixT<Elt> g_fm(2, 2, ell_,
                          std::vector<Elt>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
    // (slot-6 WP-1) SymN: ComputeFibreMatrixFM IS ComputeSymmetricPowerMatrixFM(g_fm, n_sym_).
    return ComputeFibreMatrixFM(rep_, g_fm, n_sym_);
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::check_degree(uint32_t q) const {
    if (q >= num_degrees_) {
        std::ostringstream oss;
        oss << "HeckeOperatorOnK: degree q=" << q << " out of range [0, "
            << num_degrees_ - 1 << "].";
        throw std::out_of_range(oss.str());
    }
}

// ---------------------------------------------------------------------------
// compute(): dispatch between the two arms
// ---------------------------------------------------------------------------

// (matrix-engine Phase 4, Stage B) THREAD-LOCAL options, propagated into the
// engine's host fan-out threads by the registered capture -- see the note on
// EquivariantCohomologyOptions.  memory_lean() is kept for the campaign's
// ScopedMemoryLean: the engine arm now ALWAYS streams route 2 per block row
// (compute_block()), so it changes nothing there; the legacy arm ignores it.
namespace HeckeOperatorOptions {
    namespace {
#if OC_ENABLE_LEGACY
        thread_local bool g_force_legacy_path = false;   // (S24 CL-L2) ON builds only
#endif
        thread_local bool g_theta_eq_certification = true;
        thread_local bool g_memory_lean = false;
        // (owner decision 2026-09-28, Q8) The levers default ON; the environment
        // opt-outs set a thread's INITIAL state (a variable set to "0" turns its
        // lever off; anything else, or unset, leaves it on).  (S25 LEVERS-25: the
        // last env_lever_on reader, HECKE_RHO2_STREAM, moved to EnvArmRetiredLever.)
        // (S24 CL-R1) The four slot-13/14 levers are RETIRED (owner ruling 2 of 2026-10-30): RetiredLever reads
        // exactly "0" as the opt-out (anything else, or unset, leaves the lever on, silently: env_lever_on's rule).
        // OFF build: the getters return the default route and "0" is REFUSED at the first read; ON build: the
        // reading is each thread's initial state, as before.
        constexpr engine::lever::Policy kRetiredLeverPolicy{};
        const engine::lever::RetiredLever g_u_ell_lever("HECKE_U_ELL_LOW_RANK", kRetiredLeverPolicy);       // (slot-13 WP-4)
        const engine::lever::RetiredLever g_coset_stream_lever("HECKE_COSET_STREAM", kRetiredLeverPolicy);  // (slot-14 WP-10)
        const engine::lever::RetiredLever g_parallel_lever("HECKE_PARALLEL_SETUP", kRetiredLeverPolicy);    // (slot-14 WP-10)
        const engine::lever::RetiredLever g_closed_form_lever("HECKE_RANK1_CLOSED_FORM", kRetiredLeverPolicy);  // (slot-14 G0)
#if OC_ENABLE_LEGACY
        thread_local UEllLowRankMode g_u_ell_low_rank =   // (slot-13 WP-4) the U_ell memory lever
            g_u_ell_lever.env_enabled() ? UEllLowRankMode::Auto : UEllLowRankMode::Off;
        thread_local bool g_coset_streaming = g_coset_stream_lever.env_enabled();  // (slot-14 WP-10)
        thread_local bool g_parallel_setup = g_parallel_lever.env_enabled();       // (slot-14 WP-10)
        thread_local uint64_t g_parallel_min_work = kParallelSetupMinWork;         // (slot-14 WP-10) validation hook
        thread_local bool g_rank_one_closed_form = g_closed_form_lever.env_enabled();  // (slot-14 G0)
#else
        constexpr uint64_t g_parallel_min_work = kParallelSetupMinWork;
#endif
        // (S23 P4R; S25 LEVERS-25: env arm retired) OFF build: HECKE_RHO2_STREAM=0 is REFUSED at the read; the
        // materialised memo stays in every build (set_rho2_streaming / ScopedRho2Streaming, the low-rank route).
        const engine::lever::EnvArmRetiredLever g_rho2_stream_lever("HECKE_RHO2_STREAM", kRetiredLeverPolicy);
        thread_local bool g_rho2_streaming = g_rho2_stream_lever.env_enabled();
        thread_local uint64_t g_theta_part_budget = kThetaPartBudgetBytes;                     // (slot-24 G6) validation hook
        thread_local bool g_prewalk_audit = false;                                             // (PERF-VF audit S4) validation only
        std::atomic<uint64_t> g_g6_parted_degrees{0}, g_g6_parts{0}, g_g6_whole_degrees{0}, g_g6_prewalk_skipped{0},
            g_g6_prewalk_audited{0};
        thread_local Route2Capture* g_route2_capture = nullptr;                               // (S23 P4R) validation only
        thread_local Route2TargetMutation g_route2_mutation;                                   // (S23 P4R) validation only
        thread_local uint64_t g_rho2_reduce_every = 0;                                         // (S23 P4R F3) validation only
        std::atomic<uint64_t> g_r2_streamed_degrees{0}, g_r2_materialised_degrees{0}, g_r2_keys{0}, g_r2_evals{0},
            g_r2_blocks{0}, g_r2_mid_reductions{0};
        const bool g_registered = [] {
            engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
                const bool cert = g_theta_eq_certification, lean = g_memory_lean;
#if OC_ENABLE_LEGACY
                const bool legacy = g_force_legacy_path;
                const UEllLowRankMode low_rank = g_u_ell_low_rank;
                const bool streaming = g_coset_streaming, par = g_parallel_setup;
                const uint64_t min_work = g_parallel_min_work;   // (slot-14 WP-16, audit M-1) the hook travels too
                const bool closed = g_rank_one_closed_form;      // (slot-14 G0)
#endif
                const bool rho2s = g_rho2_streaming;             // (S23 P4R)
                const bool diet = oc_util::field_diet_on();      // (slot-24 G6) Util/FieldDiet.h
                const uint64_t part_budget = g_theta_part_budget;
                return [=]() -> engine::tape::ThreadContextGuard {
                    g_theta_eq_certification = cert; g_memory_lean = lean;
#if OC_ENABLE_LEGACY
                    g_force_legacy_path = legacy;
                    g_u_ell_low_rank = low_rank; g_coset_streaming = streaming; g_parallel_setup = par;
                    g_parallel_min_work = min_work; g_rank_one_closed_form = closed;
#endif
                    g_rho2_streaming = rho2s;
                    oc_util::set_field_diet_on(diet); g_theta_part_budget = part_budget;
                    return nullptr;
                };
            });
            return true;
        }();

        // (slot-14 G0, audit M-2) THE ONE HOST FAN-OUT BUDGET (the header, at
        // for_each_index): one process-wide set of helper threads shared by
        // every data-parallel fan-out.  A Job is one fan-out; it lives on its
        // caller's stack until every thread that joined it has left.
        struct Job {
            size_t n = 0;
            size_t max_threads = 1;                          // min(the caller's pool workers, n)
            size_t pool_workers = 1;                         // the caller's pool: at most pool_workers - 1 helpers busy
            const void* pool_id = nullptr;                   // (G0 repairs, audit S1) the caller's consumer pool
            const std::function<void(size_t)>* body = nullptr;
            std::vector<engine::tape::ThreadContextInstaller> context;   // the caller's
            std::atomic<size_t> next{0};                     // the next unclaimed index
            std::atomic<size_t> fail_at{SIZE_MAX};           // the lowest failing index seen
            size_t threads_in = 1;                           // the caller counts (guarded by the pool mutex)
            std::exception_ptr error;                        // of index fail_at (guarded by the pool mutex)
            // Claims and runs indices until none is left.  An index above a
            // known failure is skipped; every index below it runs, so the
            // lowest failing index is exactly the serial loop's first failure.
            void work(std::mutex& mu) {
                for (;;) {
                    const size_t j = next.fetch_add(1, std::memory_order_relaxed);
                    if (j >= n) return;
                    if (j > fail_at.load(std::memory_order_acquire)) continue;
                    try {
                        (*body)(j);
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(mu);
                        if (j < fail_at.load(std::memory_order_relaxed)) {
                            fail_at.store(j, std::memory_order_release);
                            error = std::current_exception();
                        }
                    }
                }
            }
        };
        struct FanoutPool {
            std::mutex mu;
            std::condition_variable cv_work;    // helpers wait for a joinable job
            std::condition_variable cv_done;    // callers wait for their helpers to leave
            std::vector<Job*> jobs;             // the open jobs, oldest first
            size_t helpers = 0;                 // helper threads actually started
            // (G0 repairs, audit S1) THE BUDGET IS PER CONSUMER POOL: busy
            // helpers are counted per pool (the caller's consumer_pool(), which
            // a helper inherits through the captured context), so the per-job
            // pools of a campaign's job loop (jobs > 1, threads > 1) each get
            // their own threads - 1 helpers, as each job's fan-out had its own
            // threads before G0; one pool (a single cell, the WP-11 kit) is
            // exactly the G0 budget.  An entry is erased when its count is 0,
            // so a destroyed per-job pool's address is never read again.
            std::map<const void*, size_t> busy_by_pool;
            // (under mu) Starts helpers until there are enough for every pool
            // with an open job to have its pool_workers - 1 busy at once.
            // (audit m1) a helper is counted only once its thread has started;
            // a failed start (std::system_error at the thread limit) leaves the
            // count exact and the work to the threads that exist -- the caller
            // always drains its own job, so the fan-out still completes.
            void ensure_helpers() {
                std::map<const void*, size_t> want_by_pool;
                for (const Job* j : jobs) {
                    size_t& w = want_by_pool[j->pool_id];
                    w = std::max(w, j->pool_workers - 1);
                }
                size_t want = 0;
                for (const auto& kv : want_by_pool) want += kv.second;
                want = std::min<size_t>(want, kMaxHelpers);
                while (helpers < want) {
                    try {
                        std::thread t([this] { helper_loop(); });
                        t.detach();
                    } catch (const std::system_error&) {
                        return;
                    }
                    ++helpers;
                }
            }
            static constexpr size_t kMaxHelpers = 255;   // a backstop: pools are capped at kMaxWorkers = 32
            Job* joinable() {                   // under mu
                for (Job* j : jobs) {
                    // THE BUDGET: a helper joins only while fewer than
                    // (the job's pool workers - 1) helpers are busy in that pool.
                    const auto b = busy_by_pool.find(j->pool_id);
                    const size_t busy_here = (b == busy_by_pool.end()) ? 0 : b->second;
                    if (busy_here + 1 < j->pool_workers && j->threads_in < j->max_threads &&
                        j->next.load(std::memory_order_relaxed) < j->n) return j;
                }
                return nullptr;
            }
            void helper_loop() {
                std::unique_lock<std::mutex> lk(mu);
                for (;;) {
                    Job* job = joinable();
                    if (job == nullptr) { cv_work.wait(lk); continue; }
                    ++job->threads_in;
                    const void* pool_id = job->pool_id;   // job is gone once threads_in drops to 0
                    ++busy_by_pool[pool_id];
                    lk.unlock();
                    {
                        std::vector<engine::tape::ThreadContextGuard> guards;
                        guards.reserve(job->context.size());
                        bool installed = true;
                        try {
                            for (const auto& install : job->context) guards.push_back(install());
                        } catch (...) {
                            installed = false;   // cannot carry the caller's context: leave the work to the others
                        }
                        if (installed) job->work(mu);
                        guards.clear();
                    }
                    lk.lock();
                    {
                        const auto b = busy_by_pool.find(pool_id);
                        if (--b->second == 0) busy_by_pool.erase(b);
                    }
                    if (--job->threads_in == 0) cv_done.notify_all();
                    cv_work.notify_all();   // a budget slot is free again
                }
            }
        };
        FanoutPool& fanout_pool() {
            static FanoutPool* p = new FanoutPool();   // never destroyed: the detached helpers outlive main()
            return *p;
        }
    }
#if OC_ENABLE_LEGACY
    bool rank_one_closed_form() { return g_rank_one_closed_form; }
#else
    bool rank_one_closed_form() { return g_closed_form_lever.enabled(); }   // (S24 CL-R1) retired: always true
#endif
    bool rho2_streaming() { return g_rho2_streaming; }
    void set_rho2_streaming(bool on) { g_rho2_streaming = on; }
    // (slot-24 G6) The EFFECTIVE line is printed by the flag's first reader
    // (Util/FieldDiet.h, oc_util::field_diet_announce; G6 audit F2).
    bool field_diet() { return oc_util::field_diet_on(); }
    void set_field_diet(bool on) { oc_util::set_field_diet_on(on); }
    uint64_t theta_part_budget() { return g_theta_part_budget; }
    ScopedThetaPartBudget::ScopedThetaPartBudget(uint64_t bytes) : saved_(g_theta_part_budget) { g_theta_part_budget = bytes; }
    ScopedThetaPartBudget::~ScopedThetaPartBudget() { g_theta_part_budget = saved_; }
    bool prewalk_audit() { return g_prewalk_audit; }
    ScopedPrewalkAudit::ScopedPrewalkAudit(bool on) : saved_(g_prewalk_audit) { g_prewalk_audit = on; }
    ScopedPrewalkAudit::~ScopedPrewalkAudit() { g_prewalk_audit = saved_; }
    FieldDietStats field_diet_stats() {
        FieldDietStats st;
        st.parted_degrees = g_g6_parted_degrees.load(); st.parts = g_g6_parts.load(); st.whole_degrees = g_g6_whole_degrees.load();
        st.prewalk_skipped = g_g6_prewalk_skipped.load(); st.prewalk_audited = g_g6_prewalk_audited.load();
        return st;
    }
    void release_free_heap() {
#if defined(__GLIBC__)
        (void)malloc_trim(0);
#endif
    }
    double heap_in_use_mib() {
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
        const struct mallinfo2 mi = mallinfo2();
        return static_cast<double>(mi.uordblks + mi.hblkhd) / 1048576.0;
#else
        return -1.0;
#endif
    }
    void field_diet_count(bool parted, uint64_t parts) {
        (parted ? g_g6_parted_degrees : g_g6_whole_degrees).fetch_add(1);
        g_g6_parts.fetch_add(parts);
    }
    void field_diet_count_prewalk_skipped() { g_g6_prewalk_skipped.fetch_add(1); }
    void field_diet_count_prewalk_audited() { g_g6_prewalk_audited.fetch_add(1); }
    Rho2StreamStats rho2_stream_stats() {
        Rho2StreamStats s;
        s.streamed_degrees = g_r2_streamed_degrees.load(); s.materialised_degrees = g_r2_materialised_degrees.load();
        s.keys = g_r2_keys.load(); s.evals = g_r2_evals.load(); s.blocks = g_r2_blocks.load();
        s.mid_reductions = g_r2_mid_reductions.load();
        return s;
    }
    void rho2_stream_count(bool streamed, uint64_t keys, uint64_t evals, uint64_t blocks, uint64_t mid_reductions) {
        (streamed ? g_r2_streamed_degrees : g_r2_materialised_degrees).fetch_add(1);
        g_r2_keys.fetch_add(keys); g_r2_evals.fetch_add(evals); g_r2_blocks.fetch_add(blocks);
        g_r2_mid_reductions.fetch_add(mid_reductions);
    }
    uint64_t rho2_stream_reduce_every() { return g_rho2_reduce_every; }
    ScopedRho2StreamReduceEvery::ScopedRho2StreamReduceEvery(uint64_t r) : saved_(g_rho2_reduce_every) { g_rho2_reduce_every = r; }
    ScopedRho2StreamReduceEvery::~ScopedRho2StreamReduceEvery() { g_rho2_reduce_every = saved_; }
    Route2Capture* route2_capture() { return g_route2_capture; }
    ScopedRoute2Capture::ScopedRoute2Capture(Route2Capture* c) : saved_(g_route2_capture) { g_route2_capture = c; }
    ScopedRoute2Capture::~ScopedRoute2Capture() { g_route2_capture = saved_; }
    const Route2TargetMutation& route2_target_mutation() { return g_route2_mutation; }
    ScopedRoute2TargetMutation::ScopedRoute2TargetMutation(const Route2TargetMutation& m) : saved_(g_route2_mutation) { g_route2_mutation = m; }
    ScopedRoute2TargetMutation::~ScopedRoute2TargetMutation() { g_route2_mutation = saved_; }
#if OC_ENABLE_LEGACY
    // (S24 CL-R1) the retired levers' setters (legacy/HeckeOperatorLeversLegacy.h).
    void set_rank_one_closed_form(bool on) { g_rank_one_closed_form = on; }
    ScopedParallelSetupMinWork::ScopedParallelSetupMinWork(uint64_t w) : saved_(g_parallel_min_work) { g_parallel_min_work = w; }
    ScopedParallelSetupMinWork::~ScopedParallelSetupMinWork() { g_parallel_min_work = saved_; }
    bool coset_streaming() { return g_coset_streaming; }
    void set_coset_streaming(bool on) { g_coset_streaming = on; }
    bool parallel_setup() { return g_parallel_setup; }
    void set_parallel_setup(bool on) { g_parallel_setup = on; }
#else
    // (S24 CL-R1) retired: always true; the opt-out "=0" is refused at the first read.
    bool coset_streaming() { return g_coset_stream_lever.enabled(); }
    bool parallel_setup() { return g_parallel_lever.enabled(); }
#endif
    void for_each_index(size_t n_items, bool parallel, const std::function<void(size_t)>& body,
                        uint64_t work_per_item) {
        if (n_items == 0) return;
        // n_items * work_per_item >= min_work, without the product (it may overflow: callers pass a huge
        // work_per_item to force the fan-out).
        const uint64_t n64 = static_cast<uint64_t>(n_items);
        const bool big = work_per_item >= (g_parallel_min_work + n64 - 1) / n64;
        if (!parallel || !big || n_items < 2) {
            for (size_t j = 0; j < n_items; ++j) body(j);
            return;
        }
        const size_t workers = std::max<size_t>(1, engine::tape::consumer_pool().size());
        if (workers <= 1 || !engine::tape::host_fanout_enabled()) {
            for (size_t j = 0; j < n_items; ++j) body(j);
            return;
        }
        // (slot-14 G0, audit M-2) on the one process-wide helper set
        FanoutPool& fp = fanout_pool();
        Job job;
        job.n = n_items;
        job.max_threads = std::min(workers, n_items);
        job.pool_workers = workers;
        job.pool_id = &engine::tape::consumer_pool();
        job.body = &body;
        job.context = engine::tape::capture_thread_context();
        {
            std::lock_guard<std::mutex> lk(fp.mu);
            fp.jobs.push_back(&job);
            fp.ensure_helpers();
        }
        fp.cv_work.notify_all();
        job.work(fp.mu);   // the caller always works on its own fan-out
        {
            std::unique_lock<std::mutex> lk(fp.mu);
            fp.jobs.erase(std::find(fp.jobs.begin(), fp.jobs.end(), &job));
            --job.threads_in;   // the caller leaves
            fp.cv_done.wait(lk, [&] { return job.threads_in == 0; });
        }
        if (job.error) std::rethrow_exception(job.error);
    }
#if OC_ENABLE_LEGACY
    bool legacy_hecke_operator_path_forced() { return g_force_legacy_path; }
    void set_legacy_hecke_operator_path_forced(bool forced) { g_force_legacy_path = forced; }
#endif
    bool theta_eq_certification_enabled() { return g_theta_eq_certification; }
    void set_theta_eq_certification(bool enabled) { g_theta_eq_certification = enabled; }
    bool memory_lean() { return g_memory_lean; }
    void set_memory_lean(bool lean) { g_memory_lean = lean; }
#if OC_ENABLE_LEGACY
    UEllLowRankMode u_ell_low_rank_mode() { return g_u_ell_low_rank; }
    void set_u_ell_low_rank_mode(UEllLowRankMode m) { g_u_ell_low_rank = m; }
#else
    // (S24 CL-R1) retired: always Auto; HECKE_U_ELL_LOW_RANK=0 is refused at the first read.
    UEllLowRankMode u_ell_low_rank_mode() { return g_u_ell_lever.enabled() ? UEllLowRankMode::Auto : UEllLowRankMode::Off; }
#endif
    bool u_ell_low_rank_in_effect(uint64_t N, uint64_t ell, FibreRepresentation rep) {
        switch (u_ell_low_rank_mode()) {
            case UEllLowRankMode::On: return true;
            case UEllLowRankMode::Off: return false;
            case UEllLowRankMode::Auto: return rep == FibreRepresentation::SymN && ell != 0 && N % ell == 0;
        }
        return false;
    }
}

// (slot-13 WP-4; S13 WP-16) The U_ell lever's preconditions, ONE definition
// for both field arms (compute_block and the legacy arm) and the module route:
// returns whether the lever is in effect (owner decision 2026-09-28, Q8: Auto
// by default -- in effect iff ell | N and SymN).  A SymN operator over a
// factored tau pool needs the lever in effect (only the non-SymN fibres never
// read the pool); a FORCED lever (mode On) needs SymN and ell | N.
template<class Elt>
bool check_u_ell_lever_preconditions(bool tau_factored, FibreRepresentation rep, uint64_t N, Elt ell) {
    const bool low_rank =
        HeckeOperatorOptions::u_ell_low_rank_in_effect(N, static_cast<uint64_t>(ell), rep);
    ENGINE_CHECK_MSG(low_rank || !tau_factored || rep != FibreRepresentation::SymN,
                     "HeckeOperatorOnK: the tau table was built under the U_ell low-rank lever (factored rho pool); "
                     "compute the operator with the lever in effect as well (not under ScopedUEllLowRank(false) / "
                     "HECKE_U_ELL_LOW_RANK=0), or build the tau table with the lever off.");
    if (low_rank) {
        ENGINE_CHECK_MSG(rep == FibreRepresentation::SymN,
                         "HeckeOperatorOnK: the U_ell low-rank lever is defined for the SymN fibre only.");
        ENGINE_CHECK_MSG(N % static_cast<uint64_t>(ell) == 0,
                         "HeckeOperatorOnK: the U_ell low-rank lever needs ell | N (U_ell, T_{ell^r}): "
                         "only then is every g gamma_k singular mod ell.");
    }
    return low_rank;
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::compute() {
    if (computed_) return;
    if (ring_k_ >= 2) {                      // (slot-8 WP-3) Z/ell^k: the module route
        compute_module();
        computed_ = true;
        return;
    }

    {
        // (slot-14 WP-10) A streamed coset table: every field path but the
        // U_ell lever's host assembly borrows Omega_i for the whole
        // computation, so it gets the operator's own Omega table first.
#if OC_ENABLE_LEGACY
        const bool legacy_arm = engine::backend_is_legacy() || HeckeOperatorOptions::legacy_hecke_operator_path_forced();
#else
        constexpr bool legacy_arm = false;   // (S24 CL-L2) an OFF build has no legacy arm
#endif
        bool needs_omega = legacy_arm || !HeckeOperatorOptions::u_ell_low_rank_in_effect(
                                             cosets_->N(), static_cast<uint64_t>(ell_), rep_);
#ifdef ENGINE_SHADOW_LEGACY
        needs_omega = true;
#endif
        if (needs_omega) ensure_omega_stream();
    }
#if OC_ENABLE_LEGACY
    if (engine::backend_is_legacy() || HeckeOperatorOptions::legacy_hecke_operator_path_forced()) {
        // (S13 WP-16) The legacy arm honours the U_ell lever's refusals exactly
        // as compute_block does, and reads a factored tau pool through
        // U V (assemble_theta_matrix_legacy) -- the same bytes as the dense pool.
        (void)check_u_ell_lever_preconditions(tau_->rho_pool_is_low_rank(), rep_, cosets_->N(), ell_);
        LegacyResult L;
        compute_legacy(L);
        adopt_legacy_result(std::move(L));
        return;
    }
#endif
    compute_block();
#ifdef ENGINE_SHADOW_LEGACY
    shadow_compare_with_legacy();   // Hecke/legacy/HeckeOperatorOnKShadow.inc
#endif
}

// ---------------------------------------------------------------------------
// The engine arm on the TAPE (matrix-engine Phase 4, Stage B)
// ---------------------------------------------------------------------------
//
// Per degree q ONE recorded program carries the fused assembly and every
// dense step that follows it (ENGINE_API.md Sec. 10.8, extended):
//
//   inputs   omega_i = rho_n(s_i^{-1})            (cosets_, borrowed zero-copy)
//            rho_k   = rho_n((g gamma_k) mod ell) (the tau-table's pool, borrowed)
//            H_eq, C_eq (L1 multi-vectors), P_q, the relation blocks R_q
//   nodes    Theta1  block(sigma, sigma_k) += n_k . omega_i . rho_k   as ONE
//                    kind-GEMM per tau term: pass 1 folds the kinds, pass 2
//                    (CSE) computes every DISTINCT product omega_i . rho_k
//                    exactly once -- <= m x #distinct rho, the "m x 8 distinct
//                    products" of parallel axis (a) -- and pass 5 batches the
//                    products of one wavefront into a single BATCHED_GEMM;
//                    the n_k-scaled AXPYs into the pattern blocks are the
//                    scatter.  Block rows sigma write disjoint outputs (axis (b)).
//            Theta5  Y = Theta . H_eq            one GEMM per pattern block
//            Theta6  R_q . Y  == 0               COMPARE against a zero temp
//            Theta7  (D_1 (x) I) . Y == 0        COMPARE against a zero temp
//            Theta8a W = Theta . C_eq
//            Theta10 A^{(q)} = P_q . Y
//   outputs  the pattern blocks of Theta (one d_V x d_V buffer each -- the
//            pool then BORROWS them as Theta_block(q), no copy), Y, W, A.
//
// The program is keyed in the program cache by the shape signature of plan
// D4 INCLUDING the hash of the actual tau pattern of this (n_sym, p, r, q,
// ell) -- the (sigma, i, k) -> (sigma_k, n_k mod ell, rho slot) walk -- plus
// the relation / coboundary / projector patterns and the multi-vector
// widths; a miss records and compiles (the D4 cache-poisoning caveat), a hit
// replays the compiled program on the fresh bindings (EngineRuntime.h).
//
// Theta8b is the multi-RHS solve of TapeElim.h (one elimination for all
// dim C_eq right-hand sides, axis (h)); Theta9 (the A-N3 residual) is a second
// cached program: W -= C_eq . Theta_eq, COMPARE against zero.
//
// ROUTE 2 IS STREAMED (memory profile 05 Sec. 4, "Stream the route-2
// comparison"): per block row sigma one program forms the row's blocks
// n_k . rho_n((delta_i gamma_k) mod ell) as arena TEMPS (exact integer product
// delta_i * gamma_k first, invariant I6, single reduction, one rho_n; memoised
// on the REDUCED 2x2 key, which is a pure function of the exact product) and
// COMPAREs each against the fused block.  Nothing of route 2 is ever stored;
// the whole second Theta never exists.  O12(a) is unchanged: the independence
// is in the bracketing and the separate data, not in the object's lifetime.

namespace {

// The (sigma, i, k)-walk over the tau table that every program of this file
// records against, in ONE place so that the pattern hash, the recording and
// the bindings are the same walk by construction.  Visits every term with a
// non-zero coefficient mod ell as (sigma, i, ct, xt, cmod).
template<class Elt, class Visit>
void walk_tau_terms(const HeckeTauTableT<Elt>& tau, uint32_t q, uint32_t N_q, uint32_t m, Elt ell, Visit visit) {
    for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
        for (uint32_t i = 1; i <= m; ++i) {
            const std::vector<ChainTerm>& cts = tau.tau(i, q, sigma);
            const std::vector<XiTermT<Elt>>& xts = tau.xi_terms(i, q, sigma);
            ENGINE_CHECK_MSG(xts.size() == cts.size(),
                             "HeckeOperatorOnK: xi_terms and tau term lists disagree in length.");
            for (size_t k_term = 0; k_term < cts.size(); ++k_term) {
                const ChainTerm& ct = cts[k_term];
                const XiTermT<Elt>& xt = xts[k_term];
                ENGINE_CHECK_MSG(ct.sigma_idx == xt.sigma_idx && ct.coeff == xt.coeff,
                                 "HeckeOperatorOnK: xi_terms and tau terms disagree.");
                const Elt cmod = coeff_mod(ct.coeff, ell);
                if (cmod == 0) continue;
                ENGINE_CHECK_MSG(ct.sigma_idx < N_q, "HeckeOperatorOnK: tau term names a simplex out of range");
                visit(sigma, i, ct, xt, cmod);
            }
        }
    }
}

// The pattern hash of the fused program of degree q (the D4 shape
// signature's tau-pattern part): the walk's (sigma, sigma_k, cmod, rho slot)
// sequence and the omega kinds.
// (slot-6 WP-1) The blocks are read through two accessors, so that the
// contragredient operator hashes ITS blocks' kinds; the SymN accessors are the
// shared tables, i.e. the pre-slot-6 hash exactly.
// (slot-14 WP-10) The omega accessor is a SCALAR PROBE, omega_scalar(i, sv) ->
// is_currently_scalar(sv) of Omega_i, so that a streamed coset table can be
// hashed through its block accessor (the same block, the same flag).
template<class Elt, class OmegaScalar, class RhoScalar>
uint64_t tau_pattern_hash_with(const HeckeTauTableT<Elt>& tau, uint32_t q, uint32_t N_q, uint32_t m, Elt ell,
                               OmegaScalar omega_scalar, RhoScalar rho_scalar) {
    uint64_t h = engine::tape::fnv_init();
    h = engine::tape::fnv_mix(h, N_q); h = engine::tape::fnv_mix(h, m);
    for (uint32_t i = 1; i <= m; ++i) {
        Elt sv = 0;
        const bool sc = omega_scalar(i, sv);
        h = engine::tape::fnv_mix(h, sc ? 1u : 0u); h = engine::tape::fnv_mix(h, sc ? static_cast<uint64_t>(sv) : 0u);
    }
    walk_tau_terms<Elt>(tau, q, N_q, m, ell, [&](uint32_t sigma, uint32_t i, const ChainTerm& ct, const XiTermT<Elt>& xt, Elt cmod) {
        h = engine::tape::fnv_mix(h, sigma); h = engine::tape::fnv_mix(h, i);
        h = engine::tape::fnv_mix(h, ct.sigma_idx); h = engine::tape::fnv_mix(h, static_cast<uint64_t>(cmod));
        h = engine::tape::fnv_mix(h, xt.rho_g_gamma.index());
        Elt sv = 0;
        const bool sc = rho_scalar(xt, sv);
        h = engine::tape::fnv_mix(h, sc ? 1u : 0u); h = engine::tape::fnv_mix(h, sc ? static_cast<uint64_t>(sv) : 0u);
    });
    return h;
}

// (slot-14 WP-16, audit M-5) VALIDATION-ONLY COST: on a streamed coset table this
// evaluates rho_n(s_i^{-1}) for every coset on every call, and on a factored pool
// rho_pool_block() evaluates rho_n once per TERM of the walk (no memo).  Its callers are
// validations (EngineTapeConsumerValidation, HeckeUEllStreamValidation (B)); the
// production hash goes through the operator's own table (compute()'s pattern-hash
// lambdas).  Do not route production through this overload.
template<class Elt>
uint64_t tau_pattern_hash(const HeckeTauTableT<Elt>& tau, const HeckeCosetTablesT<Elt>& cosets, uint32_t q,
                          uint32_t N_q, uint32_t m, Elt ell) {
    return tau_pattern_hash_with<Elt>(
        tau, q, N_q, m, ell,
        [&](uint32_t i, Elt& sv) -> bool { return cosets.rho_s_inv_block(i).get().is_currently_scalar(sv); },
        // (slot-14 WP-10) through the pool's block accessor: a factored pool
        // (the U_ell lever) evaluates the slot, the same bytes and flag.
        [&](const XiTermT<Elt>& xt, Elt& sv) -> bool {
            return tau.rho_pool_block(xt.rho_g_gamma.index()).get().is_currently_scalar(sv);
        });
}

// A fixed tag mixed into the program-cache keys of a NON-default fibre
// representation only (the SymN keys are untouched).
constexpr uint64_t kContragredientKeyTag = 0x45374144554cULL;   // "E7ADUL"
constexpr uint64_t kQuotientLKeyTag = 0x5331315150554fULL;      // "S11QPUO" (slot-11 WP-5)
inline uint64_t representation_key_tag(FibreRepresentation rep) {
    return rep == FibreRepresentation::QuotientL ? kQuotientLKeyTag : kContragredientKeyTag;
}

// A d x d matrix as a program operand: a Scalar kind (alpha, no data) when the
// FlatMatrix knows it is alpha*I, else a dense input slot over its own
// storage (borrowed, zero-copy; `hold` keeps the borrow alive -- a deque, so
// earlier borrows never move).
template<class Elt>
struct Operand { Elt alpha = 0, beta = 0; engine::tape::ViewRef view; };

template<class Elt>
Operand<Elt> operand_of(engine::tape::Recorder<Elt>& rec, std::deque<DenseBuf<Elt>>& hold, const FlatMatrixT<Elt>& M) {
    Operand<Elt> o;
    Elt sv = 0;
    if (M.is_currently_scalar(sv)) { o.alpha = sv; return o; }
    hold.push_back(borrow_flat<Elt>(M));
    const engine::tape::BufId b = rec.input(cview(hold.back()));
    o.beta = 1;
    o.view = rec.whole(b);
    return o;
}

// (S23 P4R) The streamed route 2 evaluates its distinct keys in batches of this
// many rho_n (live: kRho2StreamBatch d x d matrices, 207 MB at d = 901).
constexpr size_t kRho2StreamBatch = 64;

// The reduced key of route 2: (delta_i gamma_k) mod ell, exact product first.
template<class Elt>
std::array<Elt, 4> reduced_key(const Mat22& delta_i, const Mat22& gamma, Elt ell) {
    const Mat22 prod = delta_i * gamma;   // exact, overflow-checked (I6)
    if constexpr (std::is_same_v<Elt, uint32_t>) return prod.reduced_mod(ell);
    else return prod.reduced_mod64(ell);
}

} // namespace

template<class Elt>
uint64_t HeckeOperatorOnKT<Elt>::tau_pattern_hash(const HeckeTauTableT<Elt>& tau, const HeckeCosetTablesT<Elt>& cosets,
                                                  uint32_t q, uint32_t N_q, uint32_t m, Elt ell) {
    return ::hecke::tau_pattern_hash<Elt>(tau, cosets, q, N_q, m, ell);
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::compute_block() {
    using namespace engine::tape;
    using engine::tape::BufId;
    using engine::tape::ViewRef;
    const SimplicialComplex* Kp = K();
    const ModCtx<Elt> ctx(ell_);
    const uint32_t d = d_V_;
    // (slot-13 WP-4) The U_ell memory lever (HeckeOperatorOptions::u_ell_low_rank,
    // header note): Auto by default since the owner decision of 2026-09-28 (Q8) --
    // in effect iff ell | N and SymN; ScopedUEllLowRank(false) /
    // HECKE_U_ELL_LOW_RANK=0 is the dense opt-out.
    const bool low_rank = check_u_ell_lever_preconditions(tau_->rho_pool_is_low_rank(), rep_, cosets_->N(), ell_);
    // (S23 P4R) Route 2 streamed key by key (HeckeOperatorOptions (5)): the
    // dense route only (the lever's factored rho2_lr is O(d) per key).  The
    // option and the validation-only hooks are read once, here, on the
    // calling thread.
    const bool stream_r2 = HeckeOperatorOptions::rho2_streaming() && !low_rank;
    HeckeOperatorOptions::Route2Capture* const r2_capture = HeckeOperatorOptions::route2_capture();
    const HeckeOperatorOptions::Route2TargetMutation r2_mut = HeckeOperatorOptions::route2_target_mutation();
    const uint64_t r2_reduce_every = HeckeOperatorOptions::rho2_stream_reduce_every();
    {
        static std::once_flag effective_once;
        std::call_once(effective_once, []() {
            if (const char* v = std::getenv("HECKE_RHO2_STREAM"))
                std::cout << "EFFECTIVE HECKE_RHO2_STREAM route2="
                          << (HeckeOperatorOptions::rho2_streaming() ? "streamed" : "materialised")
                          << " (HECKE_RHO2_STREAM=" << v << ")" << std::endl;
        });
    }

    std::shared_ptr<BlockPool> pool = std::make_shared<BlockPool>();        // Theta_block(q) (borrowed blocks) + D_q
    std::vector<BlockMat> theta_block(num_degrees_);
    std::vector<std::deque<DenseBuf<Elt>>> theta_storage(num_degrees_);
    std::vector<FlatMatrixT<Elt>> theta_eq(num_degrees_);
    std::vector<MatrixT<Elt>> A(num_degrees_);

    const auto& H_dims = coh_->get_H_eq_dimensions();
    if (H_dims.size() < num_degrees_) {
        throw std::runtime_error(
            "HeckeOperatorOnK::compute: cohomology object does not report all "
            "3 degrees of the 7-vertex model.");
    }
    const auto& relations = coh_->get_lssc()->get_user_defined_simplex_relations();
    using Relation = typename LocalSystemSimplicialComplexT<Elt>::SimplexRelation;
    // The relation blocks are interned into the COHOMOLOGY object's pool: the
    // distinct rho_n(g_r) are already there (assemble_A_Cq), so this is a
    // content hit per relation, not a copy.
    const std::shared_ptr<BlockPool> coh_pool = coh_->block_pool();
    const bool certify_theta_eq = HeckeOperatorOptions::theta_eq_certification_enabled();

    // ENGINE_TRACE_RSS=1: one line per phase of the operator assembly.
    struct PhaseTrace {
        bool on = std::getenv("ENGINE_TRACE_RSS") != nullptr;
        std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        void mark(uint32_t q, const char* what) {
            if (!on) return;
            const auto now = std::chrono::steady_clock::now();
            // (S23 PERF-P4a) one line per mark, with the process VmRSS (as the
            // [cohom] lines), written in one piece under concurrent degrees.
            std::ostringstream line;
            line << "      [hecke] q=" << q << " " << what << ": " << std::fixed << std::setprecision(3)
                 << std::chrono::duration<double>(now - t).count() << " s, VmRSS=" << std::setprecision(1)
                 << oc_util::CurrentRssMiB() << " MB\n";
            std::cout << line.str() << std::flush;
            t = now;
        }
    };

    // Parallel axis (c) for the operator (matrix-engine Phase 4, performance
    // closure): the degrees are independent -- each reads only the
    // cohomology's stored objects of its own q and writes its own slots of
    // theta_storage / theta_block / theta_eq / A and its own counters -- so
    // they run as one host task per q (run_concurrently: an Executor per
    // task, the pools and the program cache are synchronised).  Every
    // object is an exact function of the inputs, so the schedule cannot
    // move a bit; the certificate counters are merged in q order; the first
    // failing degree's exception, in q order, is what surfaces.  Route 2's
    // rho_n((delta_i gamma_k) mod ell) memo is per degree (a shared memo would
    // need cross-degree waits); the duplicated evaluations are 0.2 ms each.
    std::vector<uint64_t> relation_checks_q(num_degrees_, 0), cocycle_checks_q(num_degrees_, 0), theta_eq_solves_q(num_degrees_, 0);
    std::vector<Elt> residual_max_q(num_degrees_, Elt(0));
    std::vector<std::function<void()>> degree_tasks;
    for (uint32_t q = 0; q < num_degrees_; ++q) degree_tasks.emplace_back([&, q]() {
        // (S23 PERF-P4a) program 1's executor, released at its last reader
        // (below, after Theta6 / Theta7): its arena holds program 1's temps,
        // up to 1.45 GB at (n, ell, p) = (700, 13, 103), and nothing later in
        // the degree reads it.
        std::unique_ptr<Executor<Elt>> ex1 = std::make_unique<Executor<Elt>>(&consumer_pool());
        PhaseTrace ptrace;
        ptrace.mark(q, "degree start");
        uint64_t& relation_checks = relation_checks_q[q];
        uint64_t& cocycle_checks = cocycle_checks_q[q];
        uint64_t& theta_eq_solves = theta_eq_solves_q[q];
        Elt& residual_max = residual_max_q[q];
        // Route 2's rho_n((delta_i gamma_k) mod ell) of this degree: one
        // FlatMatrix per distinct REDUCED key, evaluated once (host fan-out
        // below), borrowed into the per-row programs.
        std::map<std::array<Elt, 4>, uint32_t> rho2_index;
        std::vector<FlatMatrixT<Elt>> rho2;
        std::vector<LowRankT<Elt>> rho2_lr;   // (slot-13 WP-4) the lever's factored rho2

        const uint32_t N_q = Kp->get_num_q_simplices(q);
        const uint32_t dim = N_q * d;
        const std::vector<uint32_t> dims(N_q, d);
        const uint32_t d_q = H_dims[q];
        const BlockMat& H = coh_->H_eq_block(q);
        // C_eq is read ONLY for the Theta_eq certification (Theta8/9); the
        // memory-lean cohomology has released it (05 Sec. 4).
        const BlockMat* Cp = certify_theta_eq ? &coh_->C_eq_block(q) : nullptr;
        const uint32_t dim_C_eq = (Cp != nullptr) ? Cp->cols() : 0u;   // 0: Theta8/9 skipped (level recorded in the JSON)
        if (H.cols() != d_q || H.rows() != dim) {
            throw std::runtime_error("HeckeOperatorOnK::compute: H_eq basis has unexpected shape.");
        }
        if (dim_C_eq > 0 && Cp->rows() != dim) {
            throw std::runtime_error("HeckeOperatorOnK::compute: C_eq basis has unexpected shape.");
        }
        std::vector<const Relation*> used;
        const BlockMat R_q = assemble_relation_block<Elt>(coh_pool, relations, q, N_q, d, ell_, used);
        const BlockMat D_q = assemble_coboundary_block<Elt>(pool, Kp, q, d, ell_);
        const BlockMat* P = nullptr;
        if (d_q > 0) {
            P = &coh_->P_block(q);
            if (P->rows() != d_q || P->cols() != dim) {
                std::ostringstream oss;
                oss << "HeckeOperatorOnK::compute: projector P_" << q << " has shape " << P->rows() << "x"
                    << P->cols() << ", expected " << d_q << "x" << dim << ".";
                throw std::runtime_error(oss.str());
            }
            ENGINE_CHECK_MSG(P->col_off() == H.row_off(),
                             "HeckeOperatorOnK::compute: projector partition disagrees with L1.");
        }

        // ===== Program 1: fused assembly + Theta5 / 6 / 7 / 8a / 10 =========
        std::deque<DenseBuf<Elt>> hold;                    // borrowed omega / rho operands
        std::deque<DenseBuf<Elt>>& blocks = theta_storage[q];
        std::vector<std::pair<uint32_t, uint32_t>> block_keys;   // (sigma, sigma_k) in first-use order
        std::vector<ViewRef> block_views;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> block_index;
        DenseBuf<Elt> Hs, Cs;
        const DenseView<const Elt> Hv = engine::blk::multivector_view<Elt>(H, Hs);
        DenseView<const Elt> Cv;
        if (dim_C_eq > 0) Cv = engine::blk::multivector_view<Elt>(*Cp, Cs);
        DenseBuf<Elt> Y(dim, d_q, Layout::ColMajor);          // column i = Theta z_i in L1 order
        DenseBuf<Elt> W(dim, dim_C_eq, Layout::ColMajor);     // W = Theta . C_eq
        DenseBuf<Elt> Ab(d_q, d_q, Layout::RowMajor);         // A^{(q)}
        uint32_t slot6 = kNoSlot, slot7 = kNoSlot;
        bool ran6 = false, ran7 = false;
        if (low_rank) {
            // (slot-13 WP-4) Host assembly of the fused blocks from the factored
            // pool: block(sigma, sigma_k) = sum cmod (Omega_i U_slot) V_slot.
            // The block list is built in the walk's first-use order, exactly
            // as the tape path's.
            std::map<uint32_t, uint32_t> slot_local;                       // pool slot -> local
            std::vector<uint32_t> slot_pool;                               // local -> pool slot
            const bool tau_factored = tau_->rho_pool_is_low_rank();
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> pair_local;  // (i, local slot) -> pair
            std::vector<std::pair<uint32_t, uint32_t>> pairs;
            std::vector<std::map<uint32_t, Elt>> block_terms;               // per block: pair -> summed cmod
            walk_tau_terms<Elt>(*tau_, q, N_q, m_, ell_, [&](uint32_t sigma, uint32_t i, const ChainTerm& ct,
                                                                const XiTermT<Elt>& xt, Elt cmod) {
                auto sl = slot_local.find(xt.rho_g_gamma.index());
                if (sl == slot_local.end()) {
                    sl = slot_local.emplace(xt.rho_g_gamma.index(), static_cast<uint32_t>(slot_pool.size())).first;
                    slot_pool.push_back(xt.rho_g_gamma.index());
                }
                const std::pair<uint32_t, uint32_t> pk(i, sl->second);
                auto pl = pair_local.find(pk);
                if (pl == pair_local.end()) {
                    pl = pair_local.emplace(pk, static_cast<uint32_t>(pairs.size())).first;
                    pairs.push_back(pk);
                }
                const std::pair<uint32_t, uint32_t> key(sigma, ct.sigma_idx);
                auto bi = block_index.find(key);
                if (bi == block_index.end()) {
                    blocks.emplace_back(d, d, Layout::RowMajor);
                    block_keys.push_back(key);
                    block_terms.emplace_back();
                    bi = block_index.emplace(key, static_cast<uint32_t>(block_keys.size() - 1)).first;
                }
                Elt& acc = block_terms[bi->second][pl->second];
                acc = ctx.add(acc, cmod);
            });
            // (slot-14 G0, audit M-2) on the one process-wide fan-out budget
            // (HeckeOperatorOptions::for_each_index), always parallel as before.
            auto fan_out = [&](size_t n_items, const std::function<void(size_t)>& body) {
                HeckeOperatorOptions::for_each_index(n_items, true, body, UINT64_MAX / 2);
            };
            // (1) every pool slot factored once (the rank claim, per block): read
            //     from a tau table built under the lever, else factored here
            //     from the dense pool -- (G0 repairs, audit m2) by the rank-one
            //     closed form from the slot's reduced key (the same (t, U, V) as
            //     factoring the dense slot, byte for byte; the rank claim
            //     ENGINE_CHECKed per key; re-derived densely under !NDEBUG), or,
            //     under ScopedRankOneClosedForm(false), by the dense factoriser.
            std::vector<LowRankT<Elt>> slot_lr_own(tau_factored ? 0 : slot_pool.size());
            if (!tau_factored) {
                const bool closed_form_own = ctx.is_field() && HeckeOperatorOptions::rank_one_closed_form();
                const std::vector<Elt> binom_own =
                    closed_form_own ? lowrank::binomial_row_mod<Elt>(ctx, n_sym_) : std::vector<Elt>();
                fan_out(slot_pool.size(), [&](size_t j) {
                    if (closed_form_own) {
                        const std::array<Elt, 4>& g = tau_->rho_pool_key(slot_pool[j]);
                        slot_lr_own[j] = lowrank::rank_one_rho_factors<Elt>(ctx, g[0], g[1], g[2], g[3], n_sym_, binom_own);
#ifndef NDEBUG
                        const LowRankT<Elt> ref = factor_low_rank<Elt>(ctx, dense_rows_of<Elt>(tau_->rho_pool()[slot_pool[j]]),
                                                                       d, d, std::min<uint32_t>(ctx.k, d));
                        ENGINE_DEBUG_CHECK_MSG(ref.t == slot_lr_own[j].t && ref.U == slot_lr_own[j].U &&
                                                   ref.V == slot_lr_own[j].V,
                                               "HeckeOperatorOnK (G0 repairs): the rank-one closed form differs from "
                                               "the dense factorisation of a dense tau pool slot");
#endif
                        return;
                    }
                    slot_lr_own[j] = factor_low_rank<Elt>(ctx, dense_rows_of<Elt>(tau_->rho_pool()[slot_pool[j]]), d, d,
                                                          std::min<uint32_t>(ctx.k, d));
                });
            }
            auto slot_lr_at = [&](uint32_t j) -> const LowRankT<Elt>& {
                return tau_factored ? tau_->rho_pool_low_rank(slot_pool[j]) : slot_lr_own[j];
            };
            // (2) W_pair = Omega_i U_slot (d x t).  A scalar-flagged Omega is
            //     applied as alpha U and never borrowed: populate_flat_data()
            //     rewrites a scalar matrix's buffer on EVERY call, which would
            //     race with the concurrent degree tasks (operand_of's rule).
            std::vector<std::vector<Elt>> pair_W(pairs.size());
            auto pair_product = [&](size_t j, const FlatMatrixT<Elt>& OmM) {
                const LowRankT<Elt>& lr = slot_lr_at(pairs[j].second);
                pair_W[j].assign(static_cast<size_t>(d) * lr.t, Elt(0));
                if (lr.t == 0) return;
                Elt alpha = 0;
                if (OmM.is_currently_scalar(alpha)) {
                    for (size_t e = 0; e < lr.U.size(); ++e) pair_W[j][e] = ctx.mul(alpha, lr.U[e]);
                    return;
                }
                const DenseBuf<Elt> Om = borrow_flat<Elt>(OmM);
                DenseBuf<Elt> Ub(const_cast<Elt*>(lr.U.data()), lr.U.size(), d, lr.t, Layout::RowMajor);
                DenseBuf<Elt> Wb(pair_W[j].data(), pair_W[j].size(), d, lr.t, Layout::RowMajor);
                engine::kernels::gemm<Elt>(ctx, cview(Om), cview(Ub), view(Wb), GemmOp::Set);
            };
            if (!cosets_->streamed()) {
                fan_out(pairs.size(), [&](size_t j) { pair_product(j, omega_of(pairs[j].first)); });
            } else {
                // (slot-14 WP-10) The coset streaming lever: the pairs grouped by
                // their coset i (in first-use order); one task per coset
                // evaluates Omega_i = rho_n(s_i^{-1}) ONCE into a task-local
                // block (the bytes the default table stores), forms that coset's
                // W_pair exactly as above, and drops it.  No Omega table exists.
                std::vector<uint32_t> coset_order;
                std::map<uint32_t, std::vector<size_t>> pairs_of_coset;
                for (size_t j = 0; j < pairs.size(); ++j) {
                    auto pc = pairs_of_coset.find(pairs[j].first);
                    if (pc == pairs_of_coset.end()) {
                        coset_order.push_back(pairs[j].first);
                        pc = pairs_of_coset.emplace(pairs[j].first, std::vector<size_t>()).first;
                    }
                    pc->second.push_back(j);
                }
                fan_out(coset_order.size(), [&](size_t c) {
                    const CosetRhoT<Elt> om = cosets_->rho_s_inv_block(coset_order[c]);
                    for (const size_t j : pairs_of_coset.at(coset_order[c])) pair_product(j, om.get());
                });
            }
            // (3) every block as one GEMM over its stacked factors.
            fan_out(block_keys.size(), [&](size_t b) {
                std::vector<std::tuple<Elt, const std::vector<Elt>*, const std::vector<Elt>*, uint32_t>> terms;
                for (const auto& kv : block_terms[b]) {
                    if (kv.second == 0) continue;
                    const LowRankT<Elt>& lr = slot_lr_at(pairs[kv.first].second);
                    terms.emplace_back(kv.second, &pair_W[kv.first], &lr.V, lr.t);
                }
                sum_of_low_rank<Elt>(ctx, d, terms, view(blocks[b]));
            });
            ptrace.mark(q, "low-rank host assembly");
        }
        // (slot-24 G6) PROGRAM 1 IN PARTS (HeckeOperatorOptions (6)).  On the
        // dense route every distinct product Omega_i . Rho that two or more
        // terms use is a CSE temp of program 1, and every such temp is live from
        // level 0 to its last AXPY, so the arena holds them all at once (2.96 GB
        // at (n, ell, p, q) = (1000, 13, 103, 1)).  When that estimate exceeds the
        // part budget, the walk's terms are grouped by their product (omega
        // slot, rho slot) in first-use order and the groups are cut into parts
        // of at most budget / (d^2 sizeof(Elt)) shared products; each part is one
        // program that accumulates its terms -- recorded exactly as program 1
        // records them, in walk order -- into the fused blocks (zero-initialised
        // DenseBufs, created in the walk's first-use order as before), run on
        // ex1 one after the other.  Mod ell the sum of the terms is order-free,
        // so every block is exactly program 1's; program 1 itself then reads the
        // finished blocks as inputs, exactly as under the U_ell lever.  Memory
        // only.  Rollback HECKE_FIELD_DIET=0 / ScopedFieldDiet(false).
        bool parted = false;
        // (slot-25 PERF-VF (b1)) A shared product is a (omega, rho) group used by two or more
        // terms, so shared <= floor(#terms / 2) <= floor(T / 2), T = the walk's term-list lengths
        // summed (cmod == 0 terms included: an upper bound).  When even that bound fits the part
        // budget, the pre-walk below cannot part the program (its test is shared * temp > budget),
        // so it is skipped: the parted decision, the counters and every value are the same by
        // construction, without the walk's two std::map groupings (3-7 % of a small cell,
        // slot24/CLOSE_H2H_diagnosis.md Sec. 3).  Speed only.
        //
        // (PERF-VF audit S4) Under the validation-only ScopedPrewalkAudit the pre-walk runs anyway
        // after a skip, and ENGINE_CHECKs that it finds shared <= floor(T / 2) and does not part:
        // the "whole by construction" claim is then tested, not assumed (HeckeThetaRssDiet G6-P).
        bool prewalk_needed = true;
        bool skipped_by_bound = false;   // the bound decided "whole"; under the audit the walk re-checks it
        uint64_t T = 0;
        if (!low_rank && HeckeOperatorOptions::field_diet()) {
            for (uint32_t sigma = 0; sigma < N_q; ++sigma)
                for (uint32_t i = 1; i <= m_; ++i) T += tau_->tau(i, q, sigma).size();
            const uint64_t temp_bytes_bound = static_cast<uint64_t>(d) * d * sizeof(Elt);
            // (PERF-VF audit N1) floor(T/2) * temp <= budget  <=>  floor(T/2) <= floor(budget / temp): no product to wrap.
            if (temp_bytes_bound == 0 || T / 2 <= HeckeOperatorOptions::theta_part_budget() / temp_bytes_bound) {
                skipped_by_bound = true;
                prewalk_needed = HeckeOperatorOptions::prewalk_audit();
                HeckeOperatorOptions::field_diet_count(false, 0);   // exactly the pre-walk's "whole" outcome
                HeckeOperatorOptions::field_diet_count_prewalk_skipped();
            }
        }
        if (!low_rank && HeckeOperatorOptions::field_diet() && prewalk_needed) {
            struct PTerm { uint32_t block, group, i; Elt cmod; const FlatMatrixT<Elt>* rho; };
            std::vector<PTerm> pterms;
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> group_index;   // (i, rho slot) -> group
            std::vector<uint32_t> group_uses;
            std::vector<char> group_dense;                                   // both factors non-scalar
            std::vector<std::pair<uint32_t, uint32_t>> pkeys;                // block keys, first-use order
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> pindex;
            walk_tau_terms<Elt>(*tau_, q, N_q, m_, ell_, [&](uint32_t sigma, uint32_t i, const ChainTerm& ct,
                                                                const XiTermT<Elt>& xt, Elt cmod) {
                const std::pair<uint32_t, uint32_t> gk(i, xt.rho_g_gamma.index());
                auto gi = group_index.find(gk);
                if (gi == group_index.end()) {
                    gi = group_index.emplace(gk, static_cast<uint32_t>(group_uses.size())).first;
                    group_uses.push_back(0);
                    Elt sv = 0;
                    group_dense.push_back(!omega_of(i).is_currently_scalar(sv) && !rho_of_term(xt).is_currently_scalar(sv));
                }
                ++group_uses[gi->second];
                const std::pair<uint32_t, uint32_t> key(sigma, ct.sigma_idx);
                auto bi = pindex.find(key);
                if (bi == pindex.end()) {
                    bi = pindex.emplace(key, static_cast<uint32_t>(pkeys.size())).first;
                    pkeys.push_back(key);
                }
                pterms.push_back(PTerm{bi->second, gi->second, i, cmod, &rho_of_term(xt)});
            });
            const uint64_t temp_bytes = static_cast<uint64_t>(d) * d * sizeof(Elt);
            uint64_t shared = 0;
            for (size_t g = 0; g < group_uses.size(); ++g) if (group_dense[g] && group_uses[g] >= 2) ++shared;
            const uint64_t budget = HeckeOperatorOptions::theta_part_budget();
            if (skipped_by_bound) {   // (PERF-VF audit S4) only under ScopedPrewalkAudit
                ENGINE_CHECK_MSG(shared <= T / 2, "HeckeOperatorOnK: pre-walk skip bound violated (shared > floor(#terms / 2))");
                ENGINE_CHECK_MSG(!(temp_bytes > 0 && shared * temp_bytes > budget),
                                 "HeckeOperatorOnK: a skipped pre-walk would have parted program 1");
                HeckeOperatorOptions::field_diet_count_prewalk_audited();
            }
            if (!skipped_by_bound && temp_bytes > 0 && shared * temp_bytes > budget) {
                parted = true;
                const uint64_t cap = std::max<uint64_t>(1, budget / temp_bytes);
                std::vector<uint32_t> group_part(group_uses.size(), 0);
                uint32_t n_parts = 1;
                uint64_t in_part = 0;
                for (size_t g = 0; g < group_uses.size(); ++g) {   // groups are numbered in first-use order
                    const bool counts = group_dense[g] && group_uses[g] >= 2;
                    if (counts && in_part == cap) { ++n_parts; in_part = 0; }
                    group_part[g] = n_parts - 1;
                    if (counts) ++in_part;
                }
                for (size_t b = 0; b < pkeys.size(); ++b) {
                    blocks.emplace_back(d, d, Layout::RowMajor);   // zeroed storage (ENGINE_API Sec. 2)
                    block_keys.push_back(pkeys[b]);
                    block_index.emplace(pkeys[b], static_cast<uint32_t>(b));
                }
                size_t part_arena_max = 0;
                for (uint32_t part = 0; part < n_parts; ++part) {
                    Recorder<Elt> rp(ctx);
                    std::deque<DenseBuf<Elt>> hold_p;
                    std::map<uint32_t, std::pair<uint32_t, Operand<Elt>>> om_p;           // i -> (local id, operand)
                    std::map<const FlatMatrixT<Elt>*, std::pair<uint32_t, Operand<Elt>>> rh_p;
                    std::map<uint32_t, std::pair<uint32_t, ViewRef>> out_p;               // block -> (local id, view)
                    uint64_t pph = fnv_mix(fnv_mix(fnv_mix(fnv_init(), 0x4736504152545354ULL), part), n_parts);   // "G6PARTST"
                    for (const PTerm& t : pterms) {
                        if (group_part[t.group] != part) continue;
                        auto oi = om_p.find(t.i);
                        if (oi == om_p.end())
                            oi = om_p.emplace(t.i, std::make_pair(static_cast<uint32_t>(om_p.size()), operand_of<Elt>(rp, hold_p, omega_of(t.i)))).first;
                        auto ri = rh_p.find(t.rho);
                        if (ri == rh_p.end())
                            ri = rh_p.emplace(t.rho, std::make_pair(static_cast<uint32_t>(rh_p.size()), operand_of<Elt>(rp, hold_p, *t.rho))).first;
                        auto bo = out_p.find(t.block);
                        if (bo == out_p.end())
                            bo = out_p.emplace(t.block, std::make_pair(static_cast<uint32_t>(out_p.size()),
                                                                       rp.whole(rp.output(view(blocks[t.block]))))).first;
                        const Operand<Elt>& om = oi->second.second;
                        const Operand<Elt>& rh = ri->second.second;
                        rp.gemm_ref(om.alpha, om.beta, om.view, ctx.mul(t.cmod, rh.alpha), ctx.mul(t.cmod, rh.beta), rh.view,
                                    bo->second.second, d, d, d, GemmOp::Add);
                        // The part's recording is a function of exactly this sequence.
                        pph = fnv_mix(pph, oi->second.first); pph = fnv_mix(pph, ri->second.first); pph = fnv_mix(pph, bo->second.first);
                        pph = fnv_mix(pph, static_cast<uint64_t>(om.alpha)); pph = fnv_mix(pph, static_cast<uint64_t>(om.beta));
                        pph = fnv_mix(pph, static_cast<uint64_t>(rh.alpha)); pph = fnv_mix(pph, static_cast<uint64_t>(rh.beta));
                        pph = fnv_mix(pph, static_cast<uint64_t>(t.cmod));
                    }
                    if (out_p.empty()) continue;
                    Program<Elt> pprog = rp.finish();
                    ShapeKey pkey = make_key<Elt>("hecke.theta.part", ctx, d, q);
                    pkey.row_dims = dims; pkey.col_dims = dims;
                    pkey.pattern_hash = pph;
                    (void)run_cached<Elt>(*ex1, pkey, std::move(pprog), rp.bindings());
                    part_arena_max = std::max(part_arena_max, ex1->arena_capacity());
                }
                HeckeOperatorOptions::field_diet_count(true, n_parts);
                if (ptrace.on) {
                    std::ostringstream line;
                    line << "      [hecke] q=" << q << " program 1 in parts: " << n_parts << " parts, shared products " << shared
                         << " (cap " << cap << " per part), part arena " << std::fixed << std::setprecision(1)
                         << (static_cast<double>(part_arena_max) * sizeof(Elt) / 1048576.0) << " MB, VmRSS="
                         << oc_util::CurrentRssMiB() << " MB\n";
                    std::cout << line.str() << std::flush;
                }
            } else if (!skipped_by_bound) {   // a skip has already counted its "whole"
                HeckeOperatorOptions::field_diet_count(false, 0);
            }
        }
        {
            Recorder<Elt> rec(ctx);
            // Inputs in a fixed order: the multi-vectors first, then the
            // operands in walk order (their slots are memoised per omega_i /
            // rho slot), then the pool inputs of R_q / P_q (record_gemm_dense).
            const BufId bH = (d_q > 0) ? rec.input(Hv) : kNoBuf;
            const BufId bC = (dim_C_eq > 0) ? rec.input(Cv) : kNoBuf;
            std::vector<Operand<Elt>> omega(m_ + 1);
            std::vector<char> omega_done(m_ + 1, 0);
            std::map<uint32_t, Operand<Elt>> rho_ops;
            // (slot-13 WP-4) Under the lever the finished blocks are inputs
            // (slot-24 G6: and after the parts).
            if (low_rank || parted)
                for (uint32_t t = 0; t < block_keys.size(); ++t) block_views.push_back(rec.whole(rec.input(cview(blocks[t]))));
            if (!low_rank && !parted)
            walk_tau_terms<Elt>(*tau_, q, N_q, m_, ell_, [&](uint32_t sigma, uint32_t i, const ChainTerm& ct,
                                                                const XiTermT<Elt>& xt, Elt cmod) {
                if (!omega_done[i]) { omega[i] = operand_of<Elt>(rec, hold, omega_of(i)); omega_done[i] = 1; }
                auto it = rho_ops.find(xt.rho_g_gamma.index());
                if (it == rho_ops.end())
                    it = rho_ops.emplace(xt.rho_g_gamma.index(), operand_of<Elt>(rec, hold, rho_of_term(xt))).first;
                const std::pair<uint32_t, uint32_t> key(sigma, ct.sigma_idx);
                auto bi = block_index.find(key);
                if (bi == block_index.end()) {
                    blocks.emplace_back(d, d, Layout::RowMajor);
                    const BufId bo = rec.output(view(blocks.back()));
                    rec.zero(rec.whole(bo));
                    block_keys.push_back(key);
                    block_views.push_back(rec.whole(bo));
                    bi = block_index.emplace(key, static_cast<uint32_t>(block_keys.size() - 1)).first;
                }
                const Operand<Elt>& om = omega[i];
                const Operand<Elt>& rh = it->second;
                // (om.alpha I + om.beta Omega)(cmod rh.alpha I + cmod rh.beta Rho)
                rec.gemm_ref(om.alpha, om.beta, om.view, ctx.mul(cmod, rh.alpha), ctx.mul(cmod, rh.beta), rh.view,
                             block_views[bi->second], d, d, d, GemmOp::Add);
            });
            const BufId bY = (d_q > 0) ? rec.output(view(Y)) : kNoBuf;
            const BufId bW = (dim_C_eq > 0) ? rec.output(view(W)) : kNoBuf;
            if (d_q > 0) rec.zero(rec.whole(bY));
            if (dim_C_eq > 0) rec.zero(rec.whole(bW));
            for (uint32_t t = 0; t < block_keys.size(); ++t) {
                const uint32_t sigma = block_keys[t].first, sk = block_keys[t].second;
                if (d_q > 0)
                    rec.gemm(block_views[t], rec.view(bH, sk * d, d, 0, d_q), rec.view(bY, sigma * d, d, 0, d_q), GemmOp::Add);
                if (dim_C_eq > 0)
                    rec.gemm(block_views[t], rec.view(bC, sk * d, d, 0, dim_C_eq), rec.view(bW, sigma * d, d, 0, dim_C_eq),
                             GemmOp::Add);
            }
            if (d_q > 0) {
                if (!used.empty() && R_q.rows() > 0) {
                    const BufId t6 = rec.temp(R_q.rows(), d_q, Layout::ColMajor);
                    const BufId z6 = rec.temp(R_q.rows(), d_q, Layout::ColMajor);
                    rec.record_gemm_dense(R_q, rec.whole(bY), rec.whole(t6), GemmOp::Set);
                    rec.zero(rec.whole(z6));
                    slot6 = rec.compare(rec.whole(t6), rec.whole(z6));
                    ran6 = true;
                }
                if (D_q.rows() > 0) {
                    const BufId t7 = rec.temp(D_q.rows(), d_q, Layout::ColMajor);
                    const BufId z7 = rec.temp(D_q.rows(), d_q, Layout::ColMajor);
                    rec.record_gemm_dense(D_q, rec.whole(bY), rec.whole(t7), GemmOp::Set);
                    rec.zero(rec.whole(z7));
                    slot7 = rec.compare(rec.whole(t7), rec.whole(z7));
                    ran7 = true;
                }
                const BufId bA = rec.output(view(Ab));
                rec.record_gemm_dense(*P, rec.whole(bY), rec.whole(bA), GemmOp::Set);
            }
            Program<Elt> prog = rec.finish();
            ptrace.mark(q, "program 1 recorded");
            ShapeKey key = make_key<Elt>(low_rank ? "hecke.theta.lr" : parted ? "hecke.theta.parted" : "hecke.theta", ctx, d, q);
            key.row_dims = dims; key.col_dims = dims;
            // (slot-13 WP-4) Under the lever program 1 reads the finished blocks,
            // so its shape is the block pattern (a factored tau pool has no
            // dense kinds to hash).
            auto lever_pattern_hash = [&]() {
                uint64_t h = fnv_mix(fnv_init(), 0x57503455454c4cULL);   // "WP4UELL"
                h = fnv_mix(h, block_keys.size());
                for (const auto& bk : block_keys) { h = fnv_mix(h, bk.first); h = fnv_mix(h, bk.second); }
                return h;
            };
            uint64_t ph = (low_rank || parted) ? lever_pattern_hash()
                : (rep_ == FibreRepresentation::SymN)
                ? (cosets_->streamed()
                       // (slot-14 WP-10) the same hash, read off the operator's own Omega table
                       ? tau_pattern_hash_with<Elt>(
                             *tau_, q, N_q, m_, ell_,
                             [&](uint32_t i, Elt& sv) -> bool { return omega_of(i).is_currently_scalar(sv); },
                             [](const XiTermT<Elt>& xt, Elt& sv) -> bool { return xt.rho_g_gamma.get().is_currently_scalar(sv); })
                       : ::hecke::tau_pattern_hash<Elt>(*tau_, *cosets_, q, N_q, m_, ell_))
                : fnv_mix(tau_pattern_hash_with<Elt>(
                              *tau_, q, N_q, m_, ell_,
                              [&](uint32_t i, Elt& sv) -> bool { return omega_of(i).is_currently_scalar(sv); },
                              [&](const XiTermT<Elt>& xt, Elt& sv) -> bool { return rho_of_term(xt).is_currently_scalar(sv); }),
                          representation_key_tag(rep_));
            ph = fnv_mix(ph, pattern_hash_of<Elt>(R_q));
            ph = fnv_mix(ph, pattern_hash_of<Elt>(D_q));
            ph = fnv_mix(ph, d_q); ph = fnv_mix(ph, dim_C_eq);
            if (P != nullptr) ph = fnv_mix(ph, pattern_hash_of<Elt>(*P));
            key.pattern_hash = ph;
            const CachedRun run = run_cached<Elt>(*ex1, key, std::move(prog), rec.bindings());
            ptrace.mark(q, run.hit ? "program 1 run (cache hit)" : "program 1 compiled + run");
            if (ptrace.on) {   // one piece under concurrent degrees, as mark()
                std::ostringstream line;
                line << "      [hecke] q=" << q << " program 1 arena " << std::fixed << std::setprecision(1)
                     << (static_cast<double>(ex1->arena_capacity()) * sizeof(Elt) / 1048576.0) << " MB, blocks "
                     << block_keys.size() << ", rho operands " << rho_ops.size() << "\n";
                std::cout << line.str() << std::flush;
            }

            // --- Theta6: runtime assertion (i) -----------------------------------
            if (ran6) {
                const ResultSlot& s6 = run.result.slots[slot6];
                ENGINE_CHECK(s6.written);
                if (!s6.equal) {
                    // Locate the offending entry on the host (failure path only).
                    DenseBuf<Elt> res(R_q.rows(), d_q, Layout::ColMajor);
                    engine::blk::gemm_dense<Elt>(R_q, cview(Y), view(res));
                    uint32_t bad_r = 0, bad_c = 0;
                    first_nonzero<Elt>(cview(res), bad_r, bad_c);
                    const Relation& rel = *used[bad_r / d];
                    const uint32_t r = bad_r % d;
                    const Elt lhs_val = Y.at(rel.s_lhs_idx_in_Kq * d + r, bad_c);
                    const Elt rhs_val = ctx.sub(lhs_val, res.at(bad_r, bad_c));
                    std::ostringstream oss;
                    oss << "HeckeOperatorOnK::compute: runtime assertion (i) "
                           "FAILED -- y_" << bad_c << " = Theta z_" << bad_c
                        << " violates relation '" << rel.tag << "' at q=" << q
                        << ", fiber row " << r << " (lhs " << lhs_val
                        << " != rhs " << rhs_val << ").";
                    throw std::runtime_error(oss.str());
                }
            }
            relation_checks += static_cast<uint64_t>(used.size()) * d_q;
            // --- Theta7: runtime assertion (ii) ----------------------------------
            if (ran7) {
                const ResultSlot& s7 = run.result.slots[slot7];
                ENGINE_CHECK(s7.written);
                if (!s7.equal) {
                    DenseBuf<Elt> res(D_q.rows(), d_q, Layout::ColMajor);
                    engine::blk::gemm_dense<Elt>(D_q, cview(Y), view(res));
                    uint32_t bad_r = 0, bad_c = 0;
                    first_nonzero<Elt>(cview(res), bad_r, bad_c);
                    std::ostringstream oss;
                    oss << "HeckeOperatorOnK::compute: runtime assertion (ii) "
                           "FAILED -- d(Theta z_" << bad_c << ") != 0 at q="
                        << q << ", target simplex " << (bad_r / d) << ", fiber coord "
                        << (bad_r % d) << " (value " << res.at(bad_r, bad_c) << ").";
                    throw std::runtime_error(oss.str());
                }
            }
            cocycle_checks += d_q;
        }
        hold.clear();
        // (S23 PERF-P4a) Program 1's last reader was Theta6 / Theta7 above (Y,
        // the blocks and A^{(q)} are caller-owned outputs, not arena).  Its
        // arena is released here, before route 2 materialises rho2; Theta8 / 9
        // run on a fresh executor.  Memory only: no value is read from it.
        ex1.reset();
        ptrace.mark(q, "program 1 arena released");

        // ===== Theta_block(q): the pool borrows the output blocks ==============
        {
            BlockBuilder<Elt> b(pool, dims, dims, ell_);
            for (uint32_t t = 0; t < block_keys.size(); ++t)
                b.set(block_keys[t].first, block_keys[t].second, BlockRef::dense(pool->borrow(cview(blocks[t]))));
            theta_block[q] = b.finalize();
        }
        ptrace.mark(q, "Theta_block borrowed");

        // ===== Route 2, streamed per block row (O12(a)) =========================
        {
            // (a) the reduced keys of every term of this degree, in walk order;
            //     new keys get a slot, evaluated below.
            std::vector<uint32_t> term_slot;   // per visited term
            // (slot-10 WP-7, L4) the walk visits the rows in sigma order, so
            // row sigma's terms are term_slot[row_start[sigma] .. row_start[sigma + 1]).
            std::vector<size_t> row_start(static_cast<size_t>(N_q) + 1, 0);
            const uint32_t first_new = static_cast<uint32_t>(rho2.size());
            walk_tau_terms<Elt>(*tau_, q, N_q, m_, ell_, [&](uint32_t sigma, uint32_t i, const ChainTerm& ct, const XiTermT<Elt>&, Elt) {
                const std::array<Elt, 4> key = reduced_key<Elt>(cosets_->delta(i), ct.gamma, ell_);
                auto it = rho2_index.find(key);
                if (it == rho2_index.end()) it = rho2_index.emplace(key, static_cast<uint32_t>(rho2_index.size())).first;
                term_slot.push_back(it->second);
                ++row_start[static_cast<size_t>(sigma) + 1];
            });
            for (uint32_t sigma = 0; sigma < N_q; ++sigma) row_start[sigma + 1] += row_start[sigma];
            ENGINE_CHECK(row_start[N_q] == term_slot.size());
            const uint32_t n_new = static_cast<uint32_t>(rho2_index.size()) - first_new;
            // (S23 P4R) streamed: the slot -> reduced key table only (16 bytes
            // per key); every rho_n is evaluated by the rows that use it.
            std::vector<std::array<Elt, 4>> slot_keys;
            if (stream_r2) {
                slot_keys.resize(rho2_index.size());
                for (const auto& kv : rho2_index) slot_keys[kv.second] = kv.first;
            }
            // (S23 P4R) The fused block route 2 is COMPARED against; under the
            // validation-only target mutation a copy with one entry + 1.
            DenseBuf<Elt> mutated_target;
            uint32_t mutated_index = UINT32_MAX;
            if (r2_mut.on && r2_mut.q == q) {
                auto bi = block_index.find(std::pair<uint32_t, uint32_t>(r2_mut.sigma, r2_mut.sigma_k));
                ENGINE_CHECK_MSG(bi != block_index.end() && r2_mut.r < d && r2_mut.c < d,
                                 "HeckeOperatorOnK: the route-2 target mutation names no fused block entry");
                mutated_index = bi->second;
                mutated_target = blocks[bi->second];
                mutated_target.at(r2_mut.r, r2_mut.c) = ctx.add(mutated_target.at(r2_mut.r, r2_mut.c), Elt(1));
            }
            auto target_block = [&](uint32_t b) -> const DenseBuf<Elt>& {
                return b == mutated_index ? mutated_target : blocks[b];
            };
            auto capture_block = [&](uint32_t sigma, uint32_t sk, const DenseBuf<Elt>& blk) {
                std::vector<uint64_t> v(static_cast<size_t>(d) * d);
                for (uint32_t r = 0; r < d; ++r)
                    for (uint32_t cc = 0; cc < d; ++cc) v[static_cast<size_t>(r) * d + cc] = static_cast<uint64_t>(blk.at(r, cc));
                std::lock_guard<std::mutex> lk(r2_capture->mu);
                r2_capture->blocks[std::make_tuple(q, sigma, sk)] = std::move(v);
            };
            std::atomic<uint64_t> r2_evals{0}, r2_blocks{0}, r2_mid_reductions{0};
            double r2_t_eval = 0, r2_t_acc = 0, r2_t_cmp = 0;   // (S23 P4R) trace only
            if (n_new > 0 && !stream_r2) {
                if (low_rank) rho2_lr.resize(rho2_index.size());
                else rho2.resize(rho2_index.size());
                std::vector<std::array<Elt, 4>> new_keys(n_new);
                for (const auto& kv : rho2_index) if (kv.second >= first_new) new_keys[kv.second - first_new] = kv.first;
                // (b) rho_n of every new reduced key: a pure function per key,
                //     evaluated in a host fan-out over fixed chunks (the result
                //     slot is pre-assigned, so the schedule cannot reorder a thing).
                //
                // (slot-14 G0) Under the U_ell lever over F_ell (SymN, every key
                // singular mod ell) the factors come from the rank-one closed
                // form in O(d) per key (lowrank::rank_one_rho_factors: the same
                // (U, V) as factoring the dense block, byte for byte; the rank
                // claim ENGINE_CHECKed per key; under !NDEBUG re-derived by the
                // dense route).  Opt-out: ScopedRankOneClosedForm(false).
                //
                // (slot-14 G0) The fan-out: the old hard cap of 8 threads
                // (std::min({8u, n_new, pool})) is gone; the keys run on the
                // one process-wide fan-out budget (for_each_index, dynamic
                // claiming, at most the consumer pool's workers), so the three
                // degrees share it with every other fan-out (audit M-2).
                // The closed form runs serially below the setup threshold
                // (items x 8d < kParallelSetupMinWork: O(d) per key).
                const bool closed_form = low_rank && rep_ == FibreRepresentation::SymN && ctx.is_field() &&
                                         HeckeOperatorOptions::rank_one_closed_form();
                const std::vector<Elt> binom = closed_form ? lowrank::binomial_row_mod<Elt>(ctx, n_sym_) : std::vector<Elt>();
                // The dense routes keep their unconditional fan-out (as before G0).
                const uint64_t per_key = closed_form ? static_cast<uint64_t>(8) * d : UINT64_MAX / 2;
                HeckeOperatorOptions::for_each_index(n_new, true, [&](size_t j) {
                    const std::array<Elt, 4>& g = new_keys[j];
                    if (closed_form) {
                        rho2_lr[first_new + j] = lowrank::rank_one_rho_factors<Elt>(ctx, g[0], g[1], g[2], g[3], n_sym_, binom);
#ifndef NDEBUG
                        {
                            const FlatMatrixT<Elt> g_fm(2, 2, ell_, std::vector<Elt>{ g[0], g[1], g[2], g[3] });
                            const LowRankT<Elt> ref = factor_low_rank<Elt>(ctx, dense_rows_of<Elt>(ComputeFibreMatrixFM(rep_, g_fm, n_sym_)),
                                                                           d, d, std::min<uint32_t>(ctx.k, d));
                            const LowRankT<Elt>& got = rho2_lr[first_new + j];
                            ENGINE_DEBUG_CHECK_MSG(ref.t == got.t && ref.U == got.U && ref.V == got.V,
                                                   "HeckeOperatorOnK (G0): the rank-one closed form differs from the dense factorisation");
                        }
#endif
                        return;
                    }
                    const FlatMatrixT<Elt> g_fm(2, 2, ell_, std::vector<Elt>{ g[0], g[1], g[2], g[3] });
                    if (low_rank) {
                        // (slot-13 WP-4) factor, keep only U, V (the rank claim, per block)
                        const FlatMatrixT<Elt> dense = ComputeFibreMatrixFM(rep_, g_fm, n_sym_);
                        rho2_lr[first_new + j] = factor_low_rank<Elt>(ctx, dense_rows_of<Elt>(dense), d, d,
                                                                      std::min<uint32_t>(ctx.k, d));
                    } else {
                        rho2[first_new + j] = ComputeFibreMatrixFM(rep_, g_fm, n_sym_);   // SymN: rho_n exactly
                    }
                }, per_key);
            }
            ptrace.mark(q, "route 2 rho evaluation");
            // (c) one program per block row: temps for the row's blocks,
            //     AXPYs of the route-2 blocks, COMPARE against the fused blocks.
            //
            // (slot-8 WP-L1) The row's pattern hash mixes, per term, the RANK of
            // the term's rho2 slot among the slots the row has used so far (its
            // position in the ordered map ops2).  That rank was taken as
            // std::distance(ops2.begin(), op) -- linear in |ops2| = O(ell), so
            // O(ell^2) per row.  It is now read from a Fenwick tree over the
            // slot range holding the row's used slots: rank(slot) = #{used
            // slots < slot}, the same integer (ENGINE_DEBUG_CHECKed against the
            // std::distance form), so the program-cache key is unchanged for
            // every input.  The tree is sized once per row task and starts
            // empty, exactly the state the old per-degree tree was returned to
            // between rows.
            //
            // (slot-10 WP-7, L4) The rows are independent -- each records its
            // own program from read-only inputs (term_slot, rho2, tau, blocks,
            // block_index) and only COMPAREs -- so they run as one host task per
            // row (run_concurrently: an Executor per task; the pools and the
            // program cache are synchronised).  Row sigma reads its own terms
            // term_slot[row_start[sigma] ..), so every recorded program, pattern
            // hash and cache key is the one the serial loop built; nothing is
            // written but the verdicts.  The first failing row's exception, in
            // sigma order, is what surfaces (run_concurrently rethrows in task
            // order; with one worker it runs the rows serially, in order, as
            // before).  rho2's non-scalar entries are materialised here, once,
            // so operand_of's borrow never writes a shared FlatMatrix.
            if (stream_r2) {
                // (S23 P4R) ROUTE 2 STREAMED.  No per-degree memo: the degree's
                // route-2 blocks (one d x d accumulator per (sigma, sigma_k) the
                // walk names -- as many as the fused blocks, 44 at (700, 13, 103)
                // q = 1, against ~760 memo slots) are built key by key.  The
                // distinct reduced keys are taken in ascending slot order in
                // batches of kBatch: the batch's rho_n are evaluated in the host
                // fan-out (each key ONCE, the key of step (a): exact product
                // first, single reduction), then every accumulator adds
                // cmod . rho_n of the batch keys it uses (a fan-out over the
                // blocks: disjoint writes), and the batch is released.  Mod ell
                // the sum is order-free, so each accumulator is exactly the
                // block the materialised "hecke.route2" program forms from the
                // memo.  Live: the accumulators and kBatch rho_n.  Then every
                // entry of every block is COMPAREd with the fused block, rows in
                // sigma order, sigma_k ascending, entries row-major -- the first
                // failure the materialised route would report, with its message.
                std::map<std::pair<uint32_t, uint32_t>, uint32_t> acc_index;          // (sigma, sigma_k) -> accumulator
                std::vector<std::pair<uint32_t, uint32_t>> acc_keys;
                std::vector<std::map<uint32_t, Elt>> acc_terms;                       // accumulator -> (slot -> summed cmod)
                {
                    size_t visited = 0;
                    for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
                        for (uint32_t i = 1; i <= m_; ++i) {
                            for (const ChainTerm& ct : tau_->tau(i, q, sigma)) {
                                const Elt cmod = coeff_mod(ct.coeff, ell_);
                                if (cmod == 0) continue;
                                ENGINE_CHECK(visited < row_start[static_cast<size_t>(sigma) + 1]);
                                const uint32_t slot = term_slot[visited++];
                                ENGINE_CHECK(slot < slot_keys.size());
                                const std::pair<uint32_t, uint32_t> bk(sigma, ct.sigma_idx);
                                auto ai = acc_index.find(bk);
                                if (ai == acc_index.end()) {
                                    ai = acc_index.emplace(bk, static_cast<uint32_t>(acc_keys.size())).first;
                                    acc_keys.push_back(bk);
                                    acc_terms.emplace_back();
                                }
                                Elt& a = acc_terms[ai->second][slot];
                                a = ctx.add(a, cmod);
                            }
                        }
                        ENGINE_CHECK(visited == row_start[static_cast<size_t>(sigma) + 1]);
                    }
                }
                const size_t n_acc = acc_keys.size();
                std::vector<DenseBuf<Elt>> acc;
                acc.reserve(n_acc);
                for (size_t b = 0; b < n_acc; ++b) acc.emplace_back(d, d, Layout::RowMajor);
                std::vector<typename std::map<uint32_t, Elt>::const_iterator> cursor(n_acc);
                for (size_t b = 0; b < n_acc; ++b) cursor[b] = acc_terms[b].begin();
                const size_t n_keys = slot_keys.size();
                const size_t kBatch = kRho2StreamBatch;
                std::vector<FlatMatrixT<Elt>> batch;
                for (size_t s0 = 0; s0 < n_keys; s0 += kBatch) {
                    const size_t s1 = std::min(n_keys, s0 + kBatch);
                    batch.assign(s1 - s0, FlatMatrixT<Elt>());
                    const auto tb0 = std::chrono::steady_clock::now();
                    HeckeOperatorOptions::for_each_index(s1 - s0, true, [&](size_t j) {
                        const std::array<Elt, 4>& g = slot_keys[s0 + j];
                        const FlatMatrixT<Elt> g_fm(2, 2, ell_, std::vector<Elt>{ g[0], g[1], g[2], g[3] });
                        batch[j] = ComputeFibreMatrixFM(rep_, g_fm, n_sym_);   // SymN: rho_n exactly
                        batch[j].populate_flat_data();   // a scalar-flagged rho_n too: read as d x d below
                    }, UINT64_MAX / 2);
                    r2_evals.fetch_add(s1 - s0, std::memory_order_relaxed);
                    const auto tb1 = std::chrono::steady_clock::now();
                    r2_t_eval += std::chrono::duration<double>(tb1 - tb0).count();
                    // Accumulation with DELAYED reduction: per block, the batch's
                    // terms (cmod, rho_n data) are summed entrywise in the exact
                    // accumulator type (ModCtx::Accum) over cache-sized chunks of
                    // the d x d entries, reduced every k_max - 1 products
                    // (ModCtx::k_max: no overflow) and once at the end -- the same
                    // residues as one reduction per product.
                    HeckeOperatorOptions::for_each_index(n_acc, true, [&](size_t b) {
                        using Accum = typename ModCtx<Elt>::Accum;
                        std::vector<std::pair<Elt, const Elt*>> terms;
                        for (auto& it = cursor[b]; it != acc_terms[b].end() && it->first < s1; ++it) {
                            if (it->second == 0) continue;
                            terms.emplace_back(it->second, batch[it->first - s0].get_flat_data_unsafe().data());
                        }
                        if (terms.empty()) return;
                        uint64_t per_reduce = std::max<uint64_t>(1, ctx.k_max - 1);
                        if (r2_reduce_every != 0) per_reduce = std::min(per_reduce, r2_reduce_every);   // validation only (F3)
                        uint64_t mid = 0;
                        const size_t n_entries = static_cast<size_t>(d) * d;
                        constexpr size_t kChunk = 2048;
                        Accum tmp[kChunk];
                        Elt* const dst = acc[b].data();   // owned, row-major, ld == d: entry (r, c) at r * d + c
                        ENGINE_CHECK(&acc[b].at(d - 1, d - 1) == dst + (n_entries - 1));
                        for (size_t e0 = 0; e0 < n_entries; e0 += kChunk) {
                            const size_t len = std::min(kChunk, n_entries - e0);
                            for (size_t e = 0; e < len; ++e) tmp[e] = static_cast<Accum>(dst[e0 + e]);
                            uint64_t since = 0;
                            for (const auto& t : terms) {
                                if (since == per_reduce) {
                                    for (size_t e = 0; e < len; ++e) tmp[e] = static_cast<Accum>(ctx.reduce(tmp[e]));
                                    since = 0;
                                    ++mid;
                                }
                                const Accum c = static_cast<Accum>(t.first);
                                const Elt* src = t.second + e0;
                                for (size_t e = 0; e < len; ++e) tmp[e] += c * static_cast<Accum>(src[e]);
                                ++since;
                            }
                            for (size_t e = 0; e < len; ++e) dst[e0 + e] = ctx.reduce(tmp[e]);
                        }
                        if (mid != 0) r2_mid_reductions.fetch_add(mid, std::memory_order_relaxed);
                    }, UINT64_MAX / 2);
                    r2_t_acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - tb1).count();
                }
                std::vector<FlatMatrixT<Elt>>().swap(batch);
                const auto tc0 = std::chrono::steady_clock::now();
                for (size_t b = 0; b < n_acc; ++b) ENGINE_CHECK(cursor[b] == acc_terms[b].end());
                for (const auto& kv : acc_index) {   // (sigma, sigma_k) ascending
                    const uint32_t sigma = kv.first.first, sk = kv.first.second;
                    auto bi = block_index.find(kv.first);
                    ENGINE_CHECK_MSG(bi != block_index.end(), "HeckeOperatorOnK: route 2 names a block the fused route lacks");
                    r2_blocks.fetch_add(1, std::memory_order_relaxed);
                    const DenseBuf<Elt>& blk2 = acc[kv.second];
                    if (r2_capture != nullptr) capture_block(sigma, sk, blk2);
                    const DenseBuf<Elt>& blk1 = target_block(bi->second);
                    for (uint32_t r = 0; r < d; ++r)
                        for (uint32_t cc = 0; cc < d; ++cc)
                            if (blk1.at(r, cc) != blk2.at(r, cc)) {
                                std::ostringstream oss;
                                oss << "HeckeOperatorOnK::compute: O12(a) FAILED -- the fused "
                                       "route rho_n(s_i^{-1})*rho_n(g gamma_k) and route 2 "
                                       "rho_n((delta_i gamma_k) mod ell) differ at q=" << q
                                    << ", entry (" << (sigma * d + r) << "," << (sk * d + cc) << "): "
                                    << blk1.at(r, cc) << " vs " << blk2.at(r, cc) << " (route 2 streamed).";
                                throw std::runtime_error(oss.str());
                            }
                }
                r2_t_cmp = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
            } else if (!low_rank) {
            for (const FlatMatrixT<Elt>& r : rho2) {
                Elt sv = 0;
                if (!r.is_currently_scalar(sv)) r.populate_flat_data();
            }
            std::vector<std::function<void()>> row_tasks;
            row_tasks.reserve(N_q);
            for (uint32_t sigma = 0; sigma < N_q; ++sigma) row_tasks.emplace_back([&, sigma]() {
                Executor<Elt> ex_row(&consumer_pool());
                std::vector<uint32_t> slot_rank_tree(rho2.size() + 1, 0);   // 1-based Fenwick tree
                auto slot_rank_insert = [&](uint32_t slot) {
                    for (size_t x = static_cast<size_t>(slot) + 1; x < slot_rank_tree.size(); x += x & (~x + 1)) ++slot_rank_tree[x];
                };
                auto slot_rank = [&](uint32_t slot) -> uint64_t {   // #{used slots < slot}
                    uint64_t s = 0;
                    for (size_t x = slot; x > 0; x -= x & (~x + 1)) s += slot_rank_tree[x];
                    return s;
                };
                size_t visited = row_start[sigma];
                Recorder<Elt> r2(ctx);
                std::deque<DenseBuf<Elt>> hold2;
                std::map<uint32_t, Operand<Elt>> ops2;           // rho2 slot -> operand
                std::map<uint32_t, BufId> row_temps;              // sigma_k -> temp
                std::vector<std::pair<uint32_t, uint32_t>> cmp;   // (sigma_k, compare slot)
                uint64_t ph = fnv_init();
                if (rep_ != FibreRepresentation::SymN) ph = fnv_mix(ph, representation_key_tag(rep_));
                ph = fnv_mix(ph, sigma);
                for (uint32_t i = 1; i <= m_; ++i) {
                    const std::vector<ChainTerm>& cts = tau_->tau(i, q, sigma);
                    for (const ChainTerm& ct : cts) {
                        const Elt cmod = coeff_mod(ct.coeff, ell_);
                        if (cmod == 0) continue;
                        ENGINE_CHECK(visited < row_start[static_cast<size_t>(sigma) + 1]);
                        const uint32_t slot = term_slot[visited++];
                        auto op = ops2.find(slot);
                        if (op == ops2.end()) {
                            ENGINE_CHECK(slot < rho2.size());
                            op = ops2.emplace(slot, operand_of<Elt>(r2, hold2, rho2[slot])).first;
                            slot_rank_insert(slot);
                        }
                        auto tp = row_temps.find(ct.sigma_idx);
                        if (tp == row_temps.end()) {
                            const BufId tb = r2.temp(d, d, Layout::RowMajor);
                            r2.zero(r2.whole(tb));
                            tp = row_temps.emplace(ct.sigma_idx, tb).first;
                        }
                        const Operand<Elt>& o = op->second;
                        // temp += cmod * (o.alpha I + o.beta Rho2)
                        if (o.alpha != 0) r2.axpy_identity(ctx.mul(cmod, o.alpha), r2.whole(tp->second));
                        if (o.beta != 0) r2.axpy(ctx.mul(cmod, o.beta), o.view, r2.whole(tp->second));
                        ph = fnv_mix(ph, ct.sigma_idx); ph = fnv_mix(ph, static_cast<uint64_t>(cmod));
                        const uint64_t rank = slot_rank(slot);
                        ENGINE_DEBUG_CHECK_MSG(rank == static_cast<uint64_t>(std::distance(ops2.begin(), op)),
                                               "HeckeOperatorOnK: route 2's slot rank differs from its ops2 position");
                        ph = fnv_mix(ph, rank);
                        ph = fnv_mix(ph, o.alpha != 0 ? 1u : 0u); ph = fnv_mix(ph, static_cast<uint64_t>(o.alpha));
                    }
                }
                ENGINE_CHECK(visited == row_start[static_cast<size_t>(sigma) + 1]);
                for (const auto& kv : row_temps) {
                    const uint32_t sk = kv.first;
                    auto bi = block_index.find(std::pair<uint32_t, uint32_t>(sigma, sk));
                    ENGINE_CHECK_MSG(bi != block_index.end(), "HeckeOperatorOnK: route 2 names a block the fused route lacks");
                    const BufId bf = r2.input(cview(target_block(bi->second)));   // (S23 P4R) the mutation hook
                    cmp.emplace_back(sk, r2.compare(r2.whole(kv.second), r2.whole(bf)));
                }
                if (cmp.empty()) return;
                Program<Elt> p2 = r2.finish();
                ShapeKey key2 = make_key<Elt>("hecke.route2", ctx, d, q);
                key2.row_dims = dims; key2.col_dims = dims;
                key2.pattern_hash = ph;
                const CachedRun run2 = run_cached<Elt>(ex_row, key2, std::move(p2), r2.bindings());
                for (const auto& c : cmp) {
                    const ResultSlot& s = run2.result.slots[c.second];
                    ENGINE_CHECK(s.written);
                    if (!s.equal) {
                        // Locate the entry on the host: rebuild the route-2 block.
                        DenseBuf<Elt> blk2(d, d, Layout::RowMajor);
                        for (uint32_t i = 1; i <= m_; ++i)
                            for (const ChainTerm& ct : tau_->tau(i, q, sigma)) {
                                const Elt cmod = coeff_mod(ct.coeff, ell_);
                                if (cmod == 0 || ct.sigma_idx != c.first) continue;
                                const FlatMatrixT<Elt> rho = rho_of_exact(cosets_->delta(i) * ct.gamma);
                                for (uint32_t r = 0; r < d; ++r)
                                    for (uint32_t cc = 0; cc < d; ++cc)
                                        blk2.at(r, cc) = ctx.add(blk2.at(r, cc), ctx.mul(cmod, rho.get_entry(r, cc)));
                            }
                        const DenseBuf<Elt>& blk1 = target_block(block_index.at(std::pair<uint32_t, uint32_t>(sigma, c.first)));
                        uint32_t bad_r = 0, bad_c = 0;
                        for (uint32_t r = 0; r < d; ++r)
                            for (uint32_t cc = 0; cc < d; ++cc)
                                if (blk1.at(r, cc) != blk2.at(r, cc)) { bad_r = r; bad_c = cc; r = d; break; }
                        std::ostringstream oss;
                        oss << "HeckeOperatorOnK::compute: O12(a) FAILED -- the fused "
                               "route rho_n(s_i^{-1})*rho_n(g gamma_k) and route 2 "
                               "rho_n((delta_i gamma_k) mod ell) differ at q=" << q
                            << ", entry (" << (sigma * d + bad_r) << "," << (c.first * d + bad_c) << "): "
                            << blk1.at(bad_r, bad_c) << " vs " << blk2.at(bad_r, bad_c) << ".";
                        throw std::runtime_error(oss.str());
                    }
                }
                if (r2_capture != nullptr) {
                    // (S23 P4R, validation only) the materialised route-2 blocks
                    // on the host: sum cmod . rho2[slot] over the memo.
                    for (const auto& kv : row_temps) {
                        DenseBuf<Elt> blk(d, d, Layout::RowMajor);
                        size_t vis = row_start[sigma];
                        for (uint32_t i = 1; i <= m_; ++i)
                            for (const ChainTerm& ct : tau_->tau(i, q, sigma)) {
                                const Elt cmod = coeff_mod(ct.coeff, ell_);
                                if (cmod == 0) continue;
                                const uint32_t slot = term_slot[vis++];
                                if (ct.sigma_idx != kv.first) continue;
                                Elt sv = 0;
                                if (rho2[slot].is_currently_scalar(sv)) {
                                    for (uint32_t r = 0; r < d; ++r) blk.at(r, r) = ctx.add(blk.at(r, r), ctx.mul(cmod, sv));
                                } else {
                                    const DenseBuf<Elt> rb = borrow_flat<Elt>(rho2[slot]);
                                    engine::kernels::axpy<Elt>(ctx, cmod, cview(rb), view(blk));
                                }
                            }
                        capture_block(sigma, kv.first, blk);
                    }
                }
            });
            run_concurrently(row_tasks);
            } else {
                // (slot-13 WP-4) Route 2 under the lever: per block row, every
                // route-2 block as one GEMM over its factored rho2 terms,
                // compared entrywise with the fused block on the host.  Same
                // coverage (every entry of every block of every row), same
                // first-failing-row-in-sigma-order rule.
                std::vector<std::function<void()>> row_tasks;
                row_tasks.reserve(N_q);
                for (uint32_t sigma = 0; sigma < N_q; ++sigma) row_tasks.emplace_back([&, sigma]() {
                    size_t visited = row_start[sigma];
                    std::map<uint32_t, std::map<uint32_t, Elt>> row_terms;   // sigma_k -> (rho2 slot -> summed cmod)
                    for (uint32_t i = 1; i <= m_; ++i) {
                        for (const ChainTerm& ct : tau_->tau(i, q, sigma)) {
                            const Elt cmod = coeff_mod(ct.coeff, ell_);
                            if (cmod == 0) continue;
                            ENGINE_CHECK(visited < row_start[static_cast<size_t>(sigma) + 1]);
                            const uint32_t slot = term_slot[visited++];
                            ENGINE_CHECK(slot < rho2_lr.size());
                            Elt& acc = row_terms[ct.sigma_idx][slot];
                            acc = ctx.add(acc, cmod);
                        }
                    }
                    ENGINE_CHECK(visited == row_start[static_cast<size_t>(sigma) + 1]);
                    DenseBuf<Elt> blk2(d, d, Layout::RowMajor);
                    for (const auto& rt : row_terms) {
                        auto bi = block_index.find(std::pair<uint32_t, uint32_t>(sigma, rt.first));
                        ENGINE_CHECK_MSG(bi != block_index.end(), "HeckeOperatorOnK: route 2 names a block the fused route lacks");
                        std::vector<std::tuple<Elt, const std::vector<Elt>*, const std::vector<Elt>*, uint32_t>> terms;
                        for (const auto& kv : rt.second) {
                            if (kv.second == 0) continue;
                            const LowRankT<Elt>& lr = rho2_lr[kv.first];
                            terms.emplace_back(kv.second, &lr.U, &lr.V, lr.t);
                        }
                        sum_of_low_rank<Elt>(ctx, d, terms, view(blk2));
                        const DenseBuf<Elt>& blk1 = blocks[bi->second];
                        for (uint32_t r = 0; r < d; ++r)
                            for (uint32_t cc = 0; cc < d; ++cc)
                                if (blk1.at(r, cc) != blk2.at(r, cc)) {
                                    std::ostringstream oss;
                                    oss << "HeckeOperatorOnK::compute: O12(a) FAILED -- the fused "
                                           "route rho_n(s_i^{-1})*rho_n(g gamma_k) and route 2 "
                                           "rho_n((delta_i gamma_k) mod ell) differ at q=" << q
                                        << ", entry (" << (sigma * d + r) << "," << (rt.first * d + cc) << "): "
                                        << blk1.at(r, cc) << " vs " << blk2.at(r, cc) << " (U_ell low-rank lever).";
                                    throw std::runtime_error(oss.str());
                                }
                    }
                });
                run_concurrently(row_tasks);
            }
            if (!low_rank)
                HeckeOperatorOptions::rho2_stream_count(stream_r2, static_cast<uint64_t>(rho2_index.size()),
                                                        r2_evals.load(), r2_blocks.load(), r2_mid_reductions.load());
            if (stream_r2 && ptrace.on) {
                std::ostringstream line;
                line << "      [hecke] q=" << q << " route 2 streamed: keys " << rho2_index.size() << ", evaluations "
                     << r2_evals.load() << ", blocks " << r2_blocks.load() << ", batch " << kRho2StreamBatch << " (evaluate " << std::fixed << std::setprecision(3)
                     << r2_t_eval << " s, accumulate " << r2_t_acc << " s, compare " << r2_t_cmp << " s), VmRSS="
                     << std::fixed << std::setprecision(1) << oc_util::CurrentRssMiB() << " MB\n";
                std::cout << line.str() << std::flush;
            }
        }
        ptrace.mark(q, "route 2 programs (per block row)");
        // (S23 PERF-P4a) route 2's per-degree memo at its last reader.
        std::vector<FlatMatrixT<Elt>>().swap(rho2);
        std::vector<LowRankT<Elt>>().swap(rho2_lr);

        // ===== Theta8 / Theta9: Theta_eq by ONE multi-RHS solve (axis (h)) + A-N3
        theta_eq[q] = FlatMatrixT<Elt>::CreateZero(dim_C_eq, dim_C_eq, ell_);
        if (dim_C_eq > 0) {
            theta_storage[q].shrink_to_fit();
            // (S23 PERF-P4a) Theta8 / 9 on their own executor (program 1's was
            // released above); its arena only ever holds one residual chunk.
            Executor<Elt> ex9(&consumer_pool());
            DenseBuf<Elt> coords;   // (dim_C_eq x dim_C_eq) row-major: (t, j) = coeff of c_j in w_t
            try {
                // certify = false: A-N3 below IS the exact certificate of this
                // solve (the residual W - C_eq Theta_eq, COMPAREd against zero).
                coords = solve_multi_rhs<Elt>(ex9, pool, ell_, dims, Cv, cview(W), 256, nullptr, /*certify=*/false);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "HeckeOperatorOnK::compute: runtime assertion (iii) "
                       "FAILED -- Theta_flat * (some C_eq basis vector) is not "
                       "in span(C_eq) at q=" << q << " (" << e.what() << ").";
                throw std::runtime_error(oss.str());
            }
            ENGINE_CHECK_MSG(coords.rows == dim_C_eq && coords.cols == dim_C_eq,
                             "HeckeOperatorOnK::compute: multi-RHS solve returned the wrong shape.");
            ptrace.mark(q, "Theta_eq solve");
            // Theta_eq(row j, col t) = coords(t, j): the transposed view.
            const DenseView<const Elt> TE = cview(coords).transposed_view();
            // A-N3 (certificate C5): the EXACT residual W - C_eq * Theta_eq,
            // recomputed independently of the solver on the tape and COMPAREd
            // against zero.
            //
            // (S23 PERF-P4a) Chunked over the right-hand sides: chunk [t0, t0+c)
            // forms P = coords[t0.., :] . C_eq^T (c x dim, a tape temp) and
            // COMPAREs it with W[:, t0..]^T.  P == W^T on every chunk iff
            // W - C_eq Theta_eq == 0, so the verdict is that of the former single
            // program (W^T -= coords . C_eq^T in place, COMPAREd against a
            // dim_C_eq x dim zero temp); only that zero temp is gone, and W is
            // no longer written, so it can be released right after.  The
            // row-major coords stays the A operand as stored and C_eq^T the free
            // transpose, so the GEMM keeps the row-contiguous fast path.  On a
            // failure the full residual is formed on the host and the first
            // non-zero entry is located exactly as before.
            {
                constexpr uint32_t kResidChunk = 512;
                bool resid_ok = true;
                for (uint32_t t0 = 0; t0 < dim_C_eq && resid_ok; t0 += kResidChunk) {
                    const uint32_t c = std::min<uint32_t>(kResidChunk, dim_C_eq - t0);
                    Recorder<Elt> r9(ctx);
                    const BufId bC = r9.input(Cv);
                    const BufId bX = r9.input(cview(coords).rows_range(t0, c));
                    const BufId bW = r9.input(cview(W).cols_range(t0, c));
                    const BufId bP = r9.temp(c, dim, Layout::RowMajor);
                    r9.gemm(r9.whole(bX), r9.whole(bC).transposed_view(), r9.whole(bP), GemmOp::Set);
                    const uint32_t s9 = r9.compare(r9.whole(bP), r9.whole(bW).transposed_view());
                    Program<Elt> p9 = r9.finish();
                    ShapeKey key9 = make_key<Elt>("hecke.resid.chunk", ctx, d, q);
                    key9.row_dims = dims; key9.col_dims = std::vector<uint32_t>{c};
                    key9.pattern_hash = fnv_mix(fnv_mix(fnv_mix(fnv_init(), dim), dim_C_eq), c);
                    const CachedRun run9 = run_cached<Elt>(ex9, key9, std::move(p9), r9.bindings());
                    const ResultSlot& s = run9.result.slots[s9];
                    ENGINE_CHECK(s.written);
                    resid_ok = s.equal;
                }
                if (!resid_ok) {
                    // Failure path only: the full residual W -= C_eq . Theta_eq
                    // on the host, then the same first non-zero entry.
                    engine::kernels::gemm<Elt>(ctx, Cv, TE, view(W), GemmOp::Sub);
                    uint32_t bad_r = 0, bad_c = 0;
                    first_nonzero<Elt>(cview(W), bad_r, bad_c);
                    const Elt res = W.at(bad_r, bad_c);
                    if (res > residual_max) residual_max = res;
                    std::ostringstream oss;
                    oss << "HeckeOperatorOnK::compute: runtime assertion (iii) "
                           "FAILED -- nonzero exact residual " << res
                        << " in the Theta_eq solve at q=" << q << ", basis "
                           "vector " << bad_c << ", flat row " << bad_r << ".";
                    throw std::runtime_error(oss.str());
                }
            }
            ptrace.mark(q, "A-N3 residual");
            // (S23 PERF-P4a) W and the C_eq scratch at their last reader, before
            // the Theta_eq copy allocates.
            W = DenseBuf<Elt>();
            Cs = DenseBuf<Elt>();
            theta_eq_solves += dim_C_eq;
            theta_eq[q] = flat_of_view<Elt>(TE, ell_);
            ptrace.mark(q, "Theta_eq copy");
        }

        // ===== Theta10 / level (b): A^{(q)} (computed in program 1) =============
        A[q] = MatrixT<Elt>(d_q, d_q, ell_);
        for (uint32_t j_row = 0; j_row < d_q; ++j_row)
            for (uint32_t i_col = 0; i_col < d_q; ++i_col)
                A[q].data[j_row][i_col] = Ab.at(j_row, i_col);
        ptrace.mark(q, "degree end");
    });
    // Memory-lean (no Theta_eq certification) keeps the degrees sequential:
    // one executor's workspace live at a time, as compute_cohomology_block.
    if (certify_theta_eq) run_concurrently(degree_tasks);
    else for (auto& task : degree_tasks) task();
    uint64_t relation_checks = 0, cocycle_checks = 0, theta_eq_solves = 0;
    Elt residual_max = 0;
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        relation_checks += relation_checks_q[q];
        cocycle_checks += cocycle_checks_q[q];
        theta_eq_solves += theta_eq_solves_q[q];
        if (residual_max_q[q] > residual_max) residual_max = residual_max_q[q];
    }

    // Publish only after EVERY degree passed EVERY assertion.  The storage
    // deque moves without relocating its buffers (the pool's borrowed entries
    // point at the buffers' heap blocks), and is declared BEFORE Theta_block_
    // so it outlives the BlockMats that borrow from it.
    theta_storage_ = std::move(theta_storage);
    pool_ = std::move(pool);
    Theta_block_ = std::move(theta_block);
    Theta_eq_ = std::move(theta_eq);
    A_ = std::move(A);
    Theta_flat_.clear();
    theta_flat_built_ = false;
    routes_bit_identical_ = true;
    theta_eq_certified_ = certify_theta_eq;
    relation_checks_ = relation_checks;
    cocycle_checks_ = cocycle_checks;
    theta_eq_solves_ = theta_eq_solves;
    theta_eq_residual_max_ = residual_max; // identically 0 (asserted above)
    computed_ = true;
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------

template<class Elt>
const typename HeckeOperatorOnKT<Elt>::BlockMat& HeckeOperatorOnKT<Elt>::Theta_block(uint32_t q) const {
    require_field_op("Theta_block");
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    require_level_a("Theta_block");
    check_degree(q);
    return Theta_block_[q];
}

template<class Elt>
const std::shared_ptr<typename HeckeOperatorOnKT<Elt>::BlockPool>& HeckeOperatorOnKT<Elt>::block_pool() const {
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    require_level_a("block_pool");
    return pool_;
}

// The to_flat shim: Theta_block(q) materialised row-major, once.
template<class Elt>
void HeckeOperatorOnKT<Elt>::ensure_theta_flat() const {
    if (theta_flat_built_) return;
    Theta_flat_.assign(num_degrees_, FlatMatrixT<Elt>(0, 0, ell_));
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const BlockMat& T = Theta_block_[q];
        if (T.rows() == 0 || T.cols() == 0) continue;
        const DenseBuf<Elt> D = T.to_dense(Layout::RowMajor);
        Theta_flat_[q] = flat_of_view<Elt>(cview(D), ell_);
    }
    theta_flat_built_ = true;
}

template<class Elt>
const std::vector<FlatMatrixT<Elt>>& HeckeOperatorOnKT<Elt>::get_Theta_flat_matrices() const {
    require_field_op("get_Theta_flat_matrices");
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    require_level_a("get_Theta_flat_matrices");
    ensure_theta_flat();
    return Theta_flat_;
}

template<class Elt>
const std::vector<FlatMatrixT<Elt>>& HeckeOperatorOnKT<Elt>::get_Theta_eq_matrices() const {
    require_field_op("get_Theta_eq_matrices");
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    require_level_a("get_Theta_eq_matrices");
    return Theta_eq_;
}

template<class Elt>
const std::vector<MatrixT<Elt>>& HeckeOperatorOnKT<Elt>::get_H_matrices() const {
    require_field_op("get_H_matrices");
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    return A_;
}

// (matrix-engine Phase 6) The certificate bundle compute() already produced.
// Read-only; identical to what to_json()'s "certificate" object prints.
template<class Elt>
typename HeckeOperatorOnKT<Elt>::CertificateBundle
HeckeOperatorOnKT<Elt>::certificates() const {
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    CertificateBundle c;
    c.routes_bit_identical = routes_bit_identical_;
    c.relation_checks = relation_checks_;
    c.cocycle_checks = cocycle_checks_;
    c.theta_eq_solves = theta_eq_solves_;
    c.theta_eq_residual_max = theta_eq_residual_max_;
    c.theta_eq_certified = theta_eq_certified_;
    return c;
}

// (slot-24 G6) Release level (a) -- see the header.  Memory only: A_ and the
// certificate counters, the only objects a released operator still serves,
// are untouched, so no value can move.
template<class Elt>
void HeckeOperatorOnKT<Elt>::release_level_a() {
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK::release_level_a: compute() not run.");
    require_field_op("release_level_a");
    if (level_a_released_) return;
    std::vector<BlockMat>().swap(Theta_block_);   // the borrows first, then the pool and the storage
    pool_.reset();
    std::vector<std::deque<engine::DenseBuf<Elt>>>().swap(theta_storage_);
    std::vector<FlatMatrixT<Elt>>().swap(Theta_eq_);
    std::vector<FlatMatrixT<Elt>>().swap(Theta_flat_);
    theta_flat_built_ = false;
    std::vector<FlatMatrixT<Elt>>().swap(omega_stream_);
    std::vector<FlatMatrixT<Elt>>().swap(omega_dual_);
    std::vector<FlatMatrixT<Elt>>().swap(rho_pool_dual_);
    level_a_released_ = true;
}

template<class Elt>
void HeckeOperatorOnKT<Elt>::require_level_a(const char* what) const {
    if (!level_a_released_) return;
    throw std::logic_error(std::string("HeckeOperatorOnK::") + what +
                           ": level (a) was released (release_level_a(), slot-24 G6); only "
                           "get_H_matrices() and certificates() remain");
}

// ---------------------------------------------------------------------------
// The two evaluation routes on a single flat cochain
// ---------------------------------------------------------------------------

template<class Elt>
FlatMatrixT<Elt> HeckeOperatorOnKT<Elt>::apply_Theta(uint32_t q, const FlatMatrixT<Elt>& c_flat) const {
    require_field_op("apply_Theta");
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    require_level_a("apply_Theta");
    check_degree(q);
    const uint32_t dim = K()->get_num_q_simplices(q) * d_V_;
    if (c_flat.get_p() != ell_ || c_flat.get_cols() != 1 || c_flat.get_rows() != dim) {
        throw std::invalid_argument(
            "HeckeOperatorOnK::apply_Theta: c_flat must be an (N_q*d_V) x 1 "
            "column over F_ell.");
    }
    if (dim == 0) return FlatMatrixT<Elt>(0, 1, ell_);
    const DenseBuf<Elt> c = borrow_flat<Elt>(c_flat);
    DenseBuf<Elt> y(dim, 1, Layout::ColMajor);
    engine::blk::gemm_dense<Elt>(Theta_block_[q], cview(c), view(y));
    return FlatMatrixT<Elt>(dim, 1, ell_, std::vector<Elt>(y.data(), y.data() + dim));
}

template<class Elt>
typename HeckeOperatorOnKT<Elt>::RhoMemo::Identity HeckeOperatorOnKT<Elt>::rho_memo_identity() const {
    require_level_a("rho_memo_identity");
    typename RhoMemo::Identity id;
    id.owner = static_cast<const void*>(this);
    id.delta_table = static_cast<const void*>(cosets_);
    id.ell = static_cast<uint64_t>(ell_);
    id.n_sym = static_cast<uint64_t>(n_sym_);
    id.m = static_cast<uint64_t>(m_);
    id.representation = static_cast<uint64_t>(rep_);   // slot-6 WP-1; 0 for SymN
    return id;
}

// Matrix-free direct evaluation of [C, eq:heckechain]: never reads the fused
// Theta tables, so it remains an independent second route (O12(a)).  Per
// term: tmp = rho_n((delta_i gamma_k) mod ell) * c(sigma_k) (one gemv), then
// out(sigma) += n_k * tmp (one axpy); both exact, so the accumulation order
// cannot move a bit.
template<class Elt>
FlatMatrixT<Elt> HeckeOperatorOnKT<Elt>::apply_route2(uint32_t q, const FlatMatrixT<Elt>& c_flat) const {
    // The pre-slot-5 behaviour verbatim: a fresh, call-local table.
    RhoMemo memo;
    return apply_route2(q, c_flat, memo);
}

template<class Elt>
FlatMatrixT<Elt> HeckeOperatorOnKT<Elt>::apply_route2(uint32_t q, const FlatMatrixT<Elt>& c_flat,
                                                      RhoMemo& memo) const {
    require_field_op("apply_route2");
    require_level_a("apply_route2");
    check_degree(q);
    const uint32_t N_q = K()->get_num_q_simplices(q);
    const uint32_t dim = N_q * d_V_;
    if (c_flat.get_p() != ell_ || c_flat.get_cols() != 1 || c_flat.get_rows() != dim) {
        throw std::invalid_argument(
            "HeckeOperatorOnK::apply_route2: c_flat must be an (N_q*d_V) x 1 "
            "column over F_ell.");
    }
    if (dim == 0) return FlatMatrixT<Elt>(0, 1, ell_);

    const ModCtx<Elt> ctx(ell_);
    const DenseBuf<Elt> c = borrow_flat<Elt>(c_flat);
    const DenseView<const Elt> cv = cview(c);
    DenseBuf<Elt> out(dim, 1, Layout::ColMajor);
    DenseBuf<Elt> tmp(d_V_, 1, Layout::ColMajor);
    // Item 7.2: rho_n((delta_i gamma_k) mod ell) depends only on (i, gamma_k)
    // and recurs across the target simplices; memoise it.  (slot-5 WP-6) The
    // table is the caller's, so it can also recur ACROSS calls; the identity
    // stamp is the O(1) guard that (delta table, ell, n_sym, m) -- the part of
    // the value's dependency set that is not in the key -- is the same one the
    // table was filled under.  See Hecke/RhoMemo.h.
    memo.bind(rho_memo_identity());
    typename RhoMemo::Table& rho_cache = memo.table();
    for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
        DenseView<Elt> out_sigma = view(out).sub(sigma * d_V_, d_V_, 0, 1);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22& delta_i = cosets_->delta(i); // exact s_i^{-1} g
            for (const ChainTerm& ct : tau_->tau(i, q, sigma)) {
                const Elt cmod = coeff_mod(ct.coeff, ell_);
                if (cmod == 0) continue;
                // Exact integer product first (I6), single reduction, one rho_n.
                const std::array<int64_t, 5> key{ static_cast<int64_t>(i), ct.gamma.a, ct.gamma.b,
                                                  ct.gamma.c, ct.gamma.d };
                auto it_rho = rho_cache.find(key);
                if (it_rho == rho_cache.end()) {
                    const FlatMatrixT<Elt> rho = rho_of_exact(delta_i * ct.gamma);
                    // Copy from an LVALUE: DenseBuf's copy constructor turns the
                    // externally backed borrow into an OWNED buffer (a prvalue
                    // would be elided and keep pointing into `rho`).
                    const DenseBuf<Elt> borrowed = borrow_flat<Elt>(rho);
                    it_rho = rho_cache.emplace(key, borrowed).first;
                }
                engine::kernels::gemv<Elt>(ctx, cview(it_rho->second),
                                           cv.sub(ct.sigma_idx * d_V_, d_V_, 0, 1), view(tmp), GemmOp::Set);
                engine::kernels::axpy<Elt>(ctx, cmod, cview(tmp), out_sigma, false);
            }
        }
    }
    return FlatMatrixT<Elt>(dim, 1, ell_, std::vector<Elt>(out.data(), out.data() + dim));
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

template<class Elt>
std::string HeckeOperatorOnKT<Elt>::to_string_offset(uint32_t offset) const {
    require_level_a("to_string");
    std::ostringstream oss;
    const std::string indent = offset_string(offset);
    const std::string indent2 = offset_string(offset + 2);
    const std::string indent4 = offset_string(offset + 4);

    oss << indent << "Hecke Operator on K (fused Theta_g = omega . Xi_g, [C, def:xi]):\n";
    oss << indent2 << "ell = " << ell_ << ", n_sym = " << n_sym_
        << " (d_V = " << d_V_ << "), N = " << cosets_->N()
        << ", m = " << m_ << " cosets\n";
    if (!computed_) {
        oss << indent2 << "compute() has not run yet.\n";
        return oss.str();
    }
    oss << indent2 << "Certificate: routes bit-identical (O12(a)) = "
        << (routes_bit_identical_ ? "yes" : "NO") << "; relation checks (i) = "
        << relation_checks_ << "; cocycle checks (ii) = " << cocycle_checks_
        << "; Theta_eq solves (iii) = " << theta_eq_solves_
        << "; exact solve residual max = " << theta_eq_residual_max_ << "\n";
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        oss << indent2 << "Degree q = " << q << ":\n";
        oss << indent4 << "Theta_flat: " << Theta_block_[q].rows() << "x"
            << Theta_block_[q].cols() << " (" << Theta_block_[q].nbr() << "x"
            << Theta_block_[q].nbc() << " blocks, " << Theta_block_[q].nnz()
            << " non-zero), Theta_eq: " << Theta_eq_[q].get_rows()
            << "x" << Theta_eq_[q].get_cols() << ", A^(" << q << "): " << A_[q].rows
            << "x" << A_[q].cols << "\n";
        if (A_[q].rows > 0) {
            oss << indent4 << "A^(" << q << ") (T_p[z_i] = sum_j A_ji [z_j]):\n"
                << A_[q].to_string_offset(offset + 6);
        }
    }
    return oss.str();
}

namespace {

// Minimal hand-rolled JSON emission helpers (no third-party library; plan
// Section 3 "Output formats").
template<class Elt>
void emit_flat_matrix_json(std::ostringstream& oss, const FlatMatrixT<Elt>& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.get_rows(); ++r) {
        if (r > 0) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.get_cols(); ++c) {
            if (c > 0) oss << ",";
            oss << M.get_entry(r, c);
        }
        oss << "]";
    }
    oss << "]";
}

template<class Elt>
void emit_matrix_json(std::ostringstream& oss, const MatrixT<Elt>& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        if (r > 0) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.cols; ++c) {
            if (c > 0) oss << ",";
            oss << M.data[r][c];
        }
        oss << "]";
    }
    oss << "]";
}

} // namespace

template<class Elt>
std::string HeckeOperatorOnKT<Elt>::to_json() const {
    if (!computed_) throw std::runtime_error("HeckeOperatorOnK: compute() not run.");
    require_field_op("to_json");
    require_level_a("to_json");
    ensure_theta_flat();

    // Recover (p_hecke, r) from N = p^r and m = p^r + p^{r-1}:
    // m - N = p^{r-1}, so p = N / (m - N).
    const uint64_t N = cosets_->N();
    const uint64_t p_hecke = N / (static_cast<uint64_t>(m_) - N);
    uint32_t r_exp = 0;
    for (uint64_t t = N; t > 1; t /= p_hecke) ++r_exp;

    std::ostringstream oss;
    oss << "{";
    oss << "\"schema\":\"hecke_operator_on_K/v1\",";
    oss << "\"ell\":" << ell_ << ",";
    oss << "\"n_sym\":" << n_sym_ << ",";
    oss << "\"d_V\":" << d_V_ << ",";
    oss << "\"p_hecke\":" << p_hecke << ",";
    oss << "\"r\":" << r_exp << ",";
    oss << "\"N\":" << N << ",";
    oss << "\"num_cosets\":" << m_ << ",";
    // Documented conventions (amendment A-N4; see also Hecke/README.md):
    oss << "\"matrix_entry_order\":\"arrays of rows (row-major); entries are "
           "canonical residues in [0, ell)\",";
    oss << "\"flat_layout\":\"simplex-major, fiber-minor: flat index = "
           "s*d_V + k (simplex s in stored order, fiber coord k in the basis "
           "u_0 = x^n, ..., u_n = y^n)\",";
    oss << "\"column_convention\":\"T_p[z_i] = sum_j A[j][i] [z_j]\",";
    oss << "\"poly_coeff_order\":\"low->high degree (A-N4; no polynomial data "
           "at levels (a)/(b))\",";
    oss << "\"degrees\":[";
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        if (q > 0) oss << ",";
        oss << "{\"q\":" << q << ",";
        oss << "\"N_q\":" << K()->get_num_q_simplices(q) << ",";
        oss << "\"dim_C_eq\":" << Theta_eq_[q].get_rows() << ",";
        oss << "\"dim_H\":" << A_[q].rows << ",";
        oss << "\"Theta_flat\":";
        emit_flat_matrix_json(oss, Theta_flat_[q]);
        oss << ",\"Theta_eq\":";
        emit_flat_matrix_json(oss, Theta_eq_[q]);
        oss << ",\"A\":";
        emit_matrix_json(oss, A_[q]);
        oss << "}";
    }
    oss << "],";
    oss << "\"certificate\":{";
    oss << "\"routes_bit_identical\":" << (routes_bit_identical_ ? "true" : "false") << ",";
    oss << "\"relation_checks\":" << relation_checks_ << ",";
    oss << "\"cocycle_checks\":" << cocycle_checks_ << ",";
    oss << "\"theta_eq_solves\":" << theta_eq_solves_ << ",";
    oss << "\"theta_eq_residual_max\":" << theta_eq_residual_max_;
    // (Phase 4) The memory-lean level: the omission of Theta8/Theta9 is
    // recorded explicitly.  Emitted ONLY when off, so the default JSON (a
    // tier-A golden object) is byte-identical to before.
    if (!theta_eq_certified_) oss << ",\"theta_eq_certified\":false";
    oss << "}}";
    return oss.str();
}

#if OC_ENABLE_LEGACY
#include "legacy/HeckeOperatorOnKLegacy.inc"   // (S24 CL-L2) the legacy (pre-Phase-3) dense arm
#endif

#ifdef ENGINE_SHADOW_LEGACY
#include "legacy/HeckeOperatorOnKShadow.inc"   // (S24 CL-L1) the shadow comparison
#endif

// Stage S13 explicit instantiations.
template class HeckeOperatorOnKT<uint32_t>;
template class HeckeOperatorOnKT<uint64_t>;

} // namespace hecke
