// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S8 -- see HeckeCrossChecks.h for the contract references and the
// blockwise derivation of beta_bullet.

#include "HeckeCrossChecks.h"

#include "HeckeModelDescriptor.h"
#include "InducedLocalSystem.h"
#include "../BlockModelPipelineValidation.h" // BuildACqBlockMat -- the shared C^q_eq assembly
#include "../../ModArithmetic/BlockElim.h"
#include "../../ModArithmetic/BlockKernels.h"
#include "../../ModArithmetic/Kernels.h"
#include "HeckeSettings.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

// ===========================================================================
// (matrix-engine Phase 3 Stage B) HOW THIS FILE COMPUTES, AND WHY IT IS STILL
// AN INDEPENDENT ROUTE.
//
// Before Stage B every A-F3 identity was evaluated one standard basis vector
// at a time: dim_q applications of a chain of per-simplex GEMVs (audit 02
// Sec. 1.6, items X1-X4). Stage B keeps the MATHEMATICS unchanged -- the same
// tau-derived blocks, the same three maps, the same order of the checks -- and
// changes only how the linear algebra is issued:
//
//   X1  beta's fibre data is block-sparse. Its dense d_ind x d_ind form has at
//       most m nonzero d_V x d_V sub-blocks out of m^2 (one per coset row i,
//       in column j_k); it is now an engine::BlockMat over a content-addressed
//       pool, so each distinct rho_n(g h) is stored once and the m^2 - m zero
//       sub-blocks are structural.
//   X2  the C^q_eq(K,R;L') basis reuses the SHARED block assembly of the C1
//       constraint matrix (BuildACqBlockMat) plus block elimination, instead of
//       a second, dense, hand-written copy of the same I / -M / I-M pattern.
//   X3  the three identities are whole-matrix statements, so they are evaluated
//       as block GEMMs: beta.alpha == Xi and omega.beta.alpha == Theta compare
//       MATHEMATICAL matrices (ENGINE_API.md Sec. 9.3 `equal`), which is what the
//       B3/B4 contract asks for (audit 02 Sec. 6.1, "Engine note").
//   X4  no set_entry accumulation anywhere; results are built in one buffer.
//
// INDEPENDENCE. The cross-check is independent of the fused pipeline in
// METHOD -- it rebuilds the coset representatives t_i from the S3 closed form,
// re-derives j_k and h_k from the exact tau table, and never reads any Theta
// table -- not in its choice of arithmetic kernels. Sharing the engine's GEMM
// with Stage S7 cannot mask a disagreement: modular matrix arithmetic is exact,
// so a wrong block or a wrong index still produces a different matrix.
//
// BIT-IDENTITY of the rewrite is a theorem, not a measurement: every value here
// is a product/sum of canonical residues, matrix multiplication over F_ell is
// exact and associative, so evaluating (omega.beta.alpha) as one block product
// and then reading column b gives exactly what omega(beta(alpha(e_b))) gave.
// The kernel basis of the L'-constraint matrix is the unique D5b basis of the
// RREF (ENGINE_API.md Sec. 8), which does not change when the vacuous
// (all-zero) relation rows are dropped by the shared assembly.
// ===========================================================================

namespace hecke {

namespace {

using engine::BlockMat;
using engine::BlockRef;
using engine::BlockBuilder;
using engine::DenseBuf;
using engine::DenseView;
using engine::Layout;

// n_k mod ell for a signed chain coefficient.
uint32_t coeff_mod(int32_t coeff, uint32_t ell) {
    int64_t r = static_cast<int64_t>(coeff) % static_cast<int64_t>(ell);
    if (r < 0) r += ell;
    return static_cast<uint32_t>(r);
}

// FlatMatrix -> owning row-major DenseBuf. FlatMatrix may store a scalar
// (alpha*I) instead of data, so this must go through get_entry.
DenseBuf<uint32_t> to_buf(const FlatMatrix& m) {
    DenseBuf<uint32_t> b(m.get_rows(), m.get_cols());
    for (uint32_t r = 0; r < b.rows; ++r)
        for (uint32_t c = 0; c < b.cols; ++c) b.at(r, c) = m.get_entry(r, c);
    return b;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

HeckeCrossChecks::HeckeCrossChecks(const EquivariantCohomologyOnK* cohomology_of_L,
                                   const HeckeCosetTables* cosets, const HeckeTauTable* tau)
    : coh_L_(cohomology_of_L), cosets_(cosets), tau_(tau) {
    // (slot-8 WP-3) [E] sec:ring (b): the alpha/beta/omega factorisation is F_ell-only.
    RequireFieldCoefficientRing("HeckeCrossChecks", coh_L_ ? coh_L_->coefficient_ring_k() : 1u);
    if (!coh_L_ || !cosets_ || !tau_) {
        throw std::invalid_argument("HeckeCrossChecks: null input pointer.");
    }
    if (!coh_L_->is_cohomology_computed()) {
        throw std::runtime_error(
            "HeckeCrossChecks: cohomology of the base system L must be computed.");
    }

    ell_ = coh_L_->get_prime();
    d_V_ = coh_L_->get_representation_dim();
    if (d_V_ == 0) throw std::runtime_error("HeckeCrossChecks: d_V == 0.");
    n_sym_ = d_V_ - 1;
    m_ = cosets_->num_cosets();
    d_ind_ = m_ * d_V_;
    if (cosets_->N() > 0xFFFFFFFFull) {
        throw std::runtime_error("HeckeCrossChecks: N does not fit uint32_t.");
    }
    N32_ = static_cast<uint32_t>(cosets_->N());

    // ell / d_V consistency with the S3 tables (same probe as Stage S7).
    const CosetRhoT<uint32_t> probe_omega_block = cosets_->rho_s_inv_block(1);   // (slot-14 WP-10)
    const FlatMatrix& probe_omega = probe_omega_block.get();
    if (probe_omega.get_p() != ell_ || probe_omega.get_rows() != d_V_ ||
        probe_omega.get_cols() != d_V_) {
        throw std::runtime_error(
            "HeckeCrossChecks: HeckeCosetTables (ell, n_sym) does not match the "
            "cohomology object's (prime, representation_dim).");
    }
    const std::vector<XiTerm>& probe_xi = tau_->xi_terms(1, 0, 0);
    // (slot-14 WP-10) through the pool's block accessor: a factored pool (the
    // U_ell lever, the default since 2026-09-28) evaluates the slot.
    const bool probe_bad = probe_xi.empty() || [&] {
        const CosetRhoT<uint32_t> pb = tau_->rho_pool_block(probe_xi[0].rho_g_gamma.index());
        return pb.get().get_p() != ell_ || pb.get().get_rows() != d_V_;
    }();
    if (probe_bad) {
        throw std::runtime_error(
            "HeckeCrossChecks: HeckeTauTable (ell, n_sym) does not match the "
            "cohomology object's (prime, representation_dim).");
    }

    // (matrix-engine Phase 6, audit item 7.18) The model shape is asserted
    // against the descriptor, not the literals 7/11/5; same predicate.
    const SimplicialComplex* Kp = K();
    RequireModel(SL2Z7VertexModel(), Kp, "HeckeCrossChecks");

    build_t_representatives();

    // Induced systems (Stage S5) on the SAME simplex pairs/tags/order as the
    // base relation list ([C, def:inducedls]).
    lssc_Lp_ = std::make_unique<LocalSystemSimplicialComplex>(
        MakeInducedLocalSystemGamma0Upper(*Kp, *cosets_, ell_, n_sym_));
    lssc_Lpp_ = std::make_unique<LocalSystemSimplicialComplex>(
        MakeInducedLocalSystemGamma0(*Kp, *cosets_, ell_, n_sym_));
    coh_Lp_ = std::make_unique<EquivariantCohomologyOnK>(lssc_Lp_.get());
    coh_Lpp_ = std::make_unique<EquivariantCohomologyOnK>(lssc_Lpp_.get());

    // alpha : C(K;L) -> C(K;L'), fiber matrix = block COLUMN (rho_n(t_i))_i
    // ([C, prop:alpha]); contravariant wiring: domain = L' (target of the
    // H*-map), codomain = L (source), F : V -> Ind_{Gamma'}V is (m d_V) x d_V.
    // (matrix-engine Phase 3) built in one row-major buffer, same entries.
    std::vector<uint32_t> F_alpha_buf(static_cast<size_t>(d_ind_) * d_V_, 0);
    for (uint32_t i = 1; i <= m_; ++i) {
        const CosetRhoT<uint32_t> blk_h = cosets_->rho_t_block(i);   // (slot-14 WP-10)
        const FlatMatrix& blk = blk_h.get();
        for (uint32_t r = 0; r < d_V_; ++r)
            for (uint32_t c = 0; c < d_V_; ++c)
                F_alpha_buf[static_cast<size_t>((i - 1) * d_V_ + r) * d_V_ + c] = blk.get_entry(r, c);
    }
    const FlatMatrix F_alpha =
        F_alpha_buf.empty() ? FlatMatrix::CreateZero(d_ind_, d_V_, ell_)
                            : FlatMatrix(d_ind_, d_V_, ell_, F_alpha_buf);
    id_alpha_ = std::make_unique<SimplicialComplexMorphism>(
        lssc_Lp_->get_base_K_complex(), coh_L_->get_lssc()->get_base_K_complex());
    alpha_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(
        coh_Lp_.get(), coh_L_, id_alpha_.get(), F_alpha);

    // omega : C(K;L'') -> C(K;L), fiber matrix = block ROW (rho_n(s_i^{-1}))_i
    // ([C, prop:omega]); domain = L, codomain = L'', F : Ind_{Gamma''}V -> V
    // is d_V x (m d_V).
    std::vector<uint32_t> F_omega_buf(static_cast<size_t>(d_V_) * d_ind_, 0);
    for (uint32_t i = 1; i <= m_; ++i) {
        const CosetRhoT<uint32_t> blk_h = cosets_->rho_s_inv_block(i);   // (slot-14 WP-10)
        const FlatMatrix& blk = blk_h.get();
        for (uint32_t r = 0; r < d_V_; ++r)
            for (uint32_t c = 0; c < d_V_; ++c)
                F_omega_buf[static_cast<size_t>(r) * d_ind_ + ((i - 1) * d_V_ + c)] = blk.get_entry(r, c);
    }
    const FlatMatrix F_omega =
        F_omega_buf.empty() ? FlatMatrix::CreateZero(d_V_, d_ind_, ell_)
                            : FlatMatrix(d_V_, d_ind_, ell_, F_omega_buf);
    id_omega_ = std::make_unique<SimplicialComplexMorphism>(
        coh_L_->get_lssc()->get_base_K_complex(), lssc_Lpp_->get_base_K_complex());
    omega_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(
        coh_L_, coh_Lpp_.get(), id_omega_.get(), F_omega);

    // beta : C(K;L') -> C(K;L''), the per-simplex block rows built from the
    // certified tau-table (header derivation); domain = L'', codomain = L'.
    // Built as a BlockMat here (with alpha, omega and Xi); the dense
    // FiberBlockTable / morphism form is materialised lazily by beta().
    build_block_maps();
}

// Lazily materialises the S6 morphism form of beta_bullet. Nothing in the
// checks needs it; it exists for callers of the published beta() accessor.
const EquivariantCochainComplexMorphismOnK& HeckeCrossChecks::beta() const {
    if (!beta_) {
        beta_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(
            coh_Lpp_.get(), coh_Lp_.get(), build_beta_table(),
            "beta_bullet (per-simplex block rows from tau; [C, rmk:nogo], "
            "\"Recovering the three-map factorization\")");
    }
    return *beta_;
}

const SimplicialComplex* HeckeCrossChecks::K() const {
    return coh_L_->get_lssc()->get_base_K_complex();
}

FlatMatrix HeckeCrossChecks::rho_of_exact(const Mat22& gamma_exact) const {
    const std::array<uint32_t, 4> g_mod = gamma_exact.reduced_mod(ell_);
    const FlatMatrix g_fm(2, 2, ell_,
                          std::vector<uint32_t>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym_);
}

// Reconstructs the exact Gamma^0(N) representatives t_i per the S3
// closed-form rule (HeckeCosetTables.h, "Representatives, r >= 2" note):
//   branch A (i = 1..N):            t_i = [[1, i-1], [0, 1]];
//   branch B (i = N+1..N+p^{r-1}):  t_i = [[-p*c, -1], [1, 0]], c = i-N-1.
// At r = 1 this is exactly the contract's t_1 = I, t_{1+k} = [[1,k],[0,1]],
// t_{p+1} = S. Each t_i is verified against cosets_->rho_t(i) entrywise, so
// a change of the S3 enumeration fails loudly here instead of silently
// skewing beta.
void HeckeCrossChecks::build_t_representatives() {
    const uint64_t N = cosets_->N();
    const uint64_t p_pow_rm1 = static_cast<uint64_t>(m_) - N; // p^{r-1}
    if (p_pow_rm1 == 0 || N % p_pow_rm1 != 0) {
        throw std::runtime_error(
            "HeckeCrossChecks: inconsistent (N, num_cosets) -- cannot derive p.");
    }
    const int64_t p = static_cast<int64_t>(N / p_pow_rm1);

    t_.assign(m_ + 1, Mat22::I());
    t_inv_.assign(m_ + 1, Mat22::I());
    for (uint32_t i = 1; i <= m_; ++i) {
        Mat22 t;
        if (i <= N) {
            const int64_t k = static_cast<int64_t>(i) - 1;
            t = Mat22{ 1, k, 0, 1 };
        } else {
            const int64_t c = static_cast<int64_t>(i) - static_cast<int64_t>(N) - 1;
            t = Mat22{ checked_mul<int64_t>(-p, c), -1, 1, 0 };
        }
        if (t.det() != 1) {
            throw std::runtime_error("HeckeCrossChecks: reconstructed t_i has det != 1.");
        }
        // Verify against the S3 tables (invariant I2: same rho_n convention).
        const FlatMatrix rho_t_mine = rho_of_exact(t);
        const CosetRhoT<uint32_t> rho_t_s3_h = cosets_->rho_t_block(i);   // (slot-14 WP-10)
        const FlatMatrix& rho_t_s3 = rho_t_s3_h.get();
        for (uint32_t r = 0; r < d_V_; ++r) {
            for (uint32_t c = 0; c < d_V_; ++c) {
                if (rho_t_mine.get_entry(r, c) != rho_t_s3.get_entry(r, c)) {
                    std::ostringstream oss;
                    oss << "HeckeCrossChecks: reconstructed t_" << i
                        << " disagrees with cosets.rho_t(" << i
                        << ") -- the S3 Gamma^0(N) enumeration changed.";
                    throw std::runtime_error(oss.str());
                }
            }
        }
        t_[i] = t;
        t_inv_[i] = t.inv_unimodular();
    }
}

uint32_t HeckeCrossChecks::upper_index_and_cocycle(const Mat22& gamma, Mat22& h_out) const {
    uint32_t found = 0;
    uint32_t j_found = 0;
    for (uint32_t j = 1; j <= m_; ++j) {
        const Mat22 h = gamma * t_inv_[j]; // det 1 (both unimodular)
        if (h.reduced_mod(N32_)[1] == 0) { // upper-right == 0 mod N: h in Gamma^0(N)
            ++found;
            j_found = j;
            h_out = h;
        }
    }
    if (found != 1) {
        std::ostringstream oss;
        oss << "HeckeCrossChecks: gamma matches " << found
            << " Gamma^0(N) cosets (expected exactly 1; P^1(Z/N) partition "
               "violated).";
        throw std::runtime_error(oss.str());
    }
    return j_found;
}

// ---------------------------------------------------------------------------
// The block representation (X1/X2/X3/X4; see the file header)
// ---------------------------------------------------------------------------

// alpha, beta, omega and Xi_g, one BlockMat per degree, over one shared
// content-addressed pool. Every rho_n(.) that occurs -- rho_n(t_i),
// rho_n(s_i^{-1}), rho_n(g h_k), rho_n(g gamma_k) -- is interned once, so a
// block that recurs across coset rows, simplices or degrees is stored once and
// the products below reuse it (BlockKernels' CSE keys on the data window).
void HeckeCrossChecks::build_block_maps() {
    const SimplicialComplex* Kp = K();
    const Mat22& g = cosets_->g();
    const uint32_t num_degrees = Kp->get_dimension() + 1;

    pool_ = std::make_shared<engine::BlockPool<uint32_t>>();
    alpha_bm_.assign(num_degrees, BlockMat<uint32_t>());
    beta_bm_.assign(num_degrees, BlockMat<uint32_t>());
    omega_bm_.assign(num_degrees, BlockMat<uint32_t>());
    xi_bm_.assign(num_degrees, BlockMat<uint32_t>());
    beta_alpha_bm_.assign(num_degrees, BlockMat<uint32_t>());
    beta_alpha_built_.assign(num_degrees, false);

    for (uint32_t q = 0; q < num_degrees; ++q) {
        const uint32_t N_q = Kp->get_num_q_simplices(q);
        const std::vector<uint32_t> dims_V(N_q, d_V_);                                // C^q(K;V)
        const std::vector<uint32_t> dims_ind(static_cast<size_t>(N_q) * m_, d_V_);    // C^q(K;Ind V)

        // alpha_bullet: fibre matrix = block COLUMN (rho_n(t_i))_i, applied
        // simplex by simplex ([C, prop:alpha]).
        {
            BlockBuilder<uint32_t> b(pool_, dims_ind, dims_V, ell_);
            for (uint32_t i = 1; i <= m_; ++i) {
                const BlockRef<uint32_t> R =
                    b.intern_dense(cview(to_buf(cosets_->rho_t_block(i).get())), Layout::RowMajor, true);
                for (uint32_t s = 0; s < N_q; ++s) b.set(s * m_ + (i - 1), s, R);
            }
            alpha_bm_[q] = b.finalize();
        }

        // omega_bullet: fibre matrix = block ROW (rho_n(s_i^{-1}))_i
        // ([C, prop:omega]).
        {
            BlockBuilder<uint32_t> b(pool_, dims_V, dims_ind, ell_);
            for (uint32_t i = 1; i <= m_; ++i) {
                const BlockRef<uint32_t> R =
                    b.intern_dense(cview(to_buf(cosets_->rho_s_inv_block(i).get())), Layout::RowMajor, true);
                for (uint32_t s = 0; s < N_q; ++s) b.set(s, s * m_ + (i - 1), R);
            }
            omega_bm_[q] = b.finalize();
        }

        // beta_bullet (the header derivation): for each target (q, sigma),
        // coset row i and ChainTerm (n_k, gamma_k, sigma_k) of
        // tau(s_i . i(sigma)), block ((sigma, i), (sigma_k, j_k)) accumulates
        // n_k * rho_n((g * gamma_k t_{j_k}^{-1}) mod ell). One block per
        // (i, term) -- density <= 1/m by construction (audit 02 Sec. 1.6 X1),
        // the m^2 - m empty sub-blocks of the old dense form are now
        // STRUCTURAL zeros and are never written, stored or multiplied.
        {
            BlockBuilder<uint32_t> b(pool_, dims_ind, dims_ind, ell_);
            for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
                for (uint32_t i = 1; i <= m_; ++i) {
                    for (const ChainTerm& ct : tau_->tau(i, q, sigma)) {
                        const uint32_t cmod = coeff_mod(ct.coeff, ell_);
                        if (cmod == 0) continue;
                        Mat22 h = Mat22::I();
                        const uint32_t j = upper_index_and_cocycle(ct.gamma, h);
                        // rho_n((g h) mod ell); det(g h) = N, NEVER inverted
                        // (invariant I7 / [C, obl:integral]).
                        const BlockRef<uint32_t> R =
                            b.intern_dense(cview(to_buf(rho_of_exact(g * h))), Layout::RowMajor, true);
                        b.accumulate(sigma * m_ + (i - 1), ct.sigma_idx * m_ + (j - 1),
                                     b.compose(R, 0u, cmod));
                    }
                }
            }
            beta_bm_[q] = b.finalize();
        }

        // Xi_g ([C, eq:xiexplicit]) from the S4 xi_terms table: block
        // ((sigma, i), sigma_k) accumulates n_k * rho_n((g gamma_k) mod ell).
        {
            BlockBuilder<uint32_t> b(pool_, dims_ind, dims_V, ell_);
            for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
                for (uint32_t i = 1; i <= m_; ++i) {
                    for (const XiTerm& xt : tau_->xi_terms(i, q, sigma)) {
                        const uint32_t cmod = coeff_mod(xt.coeff, ell_);
                        if (cmod == 0) continue;
                        // (slot-14 WP-10) through the pool's block accessor (a factored
                        // pool evaluates the slot; a dense one is borrowed, as before).
                        const BlockRef<uint32_t> R = b.intern_dense(
                            cview(to_buf(tau_->rho_pool_block(xt.rho_g_gamma.index()).get())), Layout::RowMajor, true);
                        b.accumulate(sigma * m_ + (i - 1), xt.sigma_idx, b.compose(R, 0u, cmod));
                    }
                }
            }
            xi_bm_[q] = b.finalize();
        }
    }
}

const BlockMat<uint32_t>& HeckeCrossChecks::beta_alpha(uint32_t q) const {
    if (!beta_alpha_built_[q]) {
        beta_alpha_bm_[q] = engine::blk::gemm<uint32_t>(beta_bm_[q], alpha_bm_[q]);
        beta_alpha_built_[q] = true;
    }
    return beta_alpha_bm_[q];
}

// The S6 FiberBlockTable form of beta_bullet, materialised from beta_bm_.
// Entry (target sigma, source sigma') is the dense d_ind x d_ind matrix whose
// (i, j) sub-block is beta_bm_[q]((sigma, i), (sigma', j)); the source
// simplices come out in ascending order, exactly as the std::map of the
// pre-Stage-B assembly produced them.
FiberBlockTable HeckeCrossChecks::build_beta_table() const {
    const SimplicialComplex* Kp = K();
    const uint32_t num_degrees = Kp->get_dimension() + 1;

    FiberBlockTable table(num_degrees);
    for (uint32_t q = 0; q < num_degrees; ++q) {
        const uint32_t N_q = Kp->get_num_q_simplices(q);
        const BlockMat<uint32_t>& B = beta_bm_[q];
        table[q].assign(N_q, std::vector<PerSimplexBlock>());
        for (uint32_t sigma = 0; sigma < N_q; ++sigma) {
            // Which source simplices occur in the m block rows of this sigma.
            std::vector<bool> present(N_q, false);
            for (uint32_t i = 0; i < m_; ++i)
                for (uint32_t k = B.row_begin(sigma * m_ + i); k < B.row_end(sigma * m_ + i); ++k)
                    present[B.col_at(k) / m_] = true;
            for (uint32_t src = 0; src < N_q; ++src) {
                if (!present[src]) continue;
                DenseBuf<uint32_t> F(d_ind_, d_ind_);
                for (uint32_t i = 0; i < m_; ++i) {
                    const uint32_t br = sigma * m_ + i;
                    for (uint32_t k = B.row_begin(br); k < B.row_end(br); ++k) {
                        const uint32_t bc = B.col_at(k);
                        if (bc / m_ != src) continue;
                        B.materialise(B.ref_at(k), d_V_, d_V_,
                                      view(F).sub(i * d_V_, d_V_, (bc % m_) * d_V_, d_V_), false);
                    }
                }
                std::vector<uint32_t> data(static_cast<size_t>(d_ind_) * d_ind_, 0);
                for (uint32_t r = 0; r < d_ind_; ++r)
                    for (uint32_t c = 0; c < d_ind_; ++c)
                        data[static_cast<size_t>(r) * d_ind_ + c] = F.at(r, c);
                table[q][sigma].push_back(
                    PerSimplexBlock{ src, FlatMatrix(d_ind_, d_ind_, ell_, data) });
            }
        }
    }
    return table;
}

// ---------------------------------------------------------------------------
// Xi_g (independent re-evaluation of [C, eq:xiexplicit])
// ---------------------------------------------------------------------------

FlatMatrix HeckeCrossChecks::apply_Xi(uint32_t q, const FlatMatrix& c_flat) const {
    const SimplicialComplex* Kp = K();
    const uint32_t N_q = Kp->get_num_q_simplices(q);
    if (c_flat.get_rows() != N_q * d_V_ || c_flat.get_cols() != 1 ||
        c_flat.get_p() != ell_) {
        throw std::invalid_argument("HeckeCrossChecks::apply_Xi: bad input cochain shape.");
    }
    const uint32_t out_rows = N_q * d_ind_;
    DenseBuf<uint32_t> x(N_q * d_V_, 1, Layout::ColMajor);
    for (uint32_t r = 0; r < N_q * d_V_; ++r) x.at(r, 0) = c_flat.get_entry(r, 0);
    DenseBuf<uint32_t> y(out_rows, 1, Layout::ColMajor);
    engine::blk::gemm_dense<uint32_t>(xi_bm_[q], cview(x), view(y));
    // One buffer, one FlatMatrix construction -- no set_entry accumulation
    // (audit 02 Sec. 1.6 X4 / Sec. 2.3).
    std::vector<uint32_t> data(out_rows, 0);
    for (uint32_t r = 0; r < out_rows; ++r) data[r] = y.at(r, 0);
    return out_rows == 0 ? FlatMatrix::CreateZero(0, 1, ell_)
                         : FlatMatrix(out_rows, 1, ell_, data);
}

// ---------------------------------------------------------------------------
// C^q_eq(K,R;L') basis (kernel of the relation constraint matrix)
// ---------------------------------------------------------------------------
//
// X2: the constraint matrix is the SAME I / -M / I-M block pattern
// EquivariantCohomologyOnK assembles for A_Cq, so it is built by the shared
// helper instead of by a second dense copy, and eliminated by the block
// engine. BuildACqBlockMat drops the vacuous relation rows (M == I on a
// self-relation, i.e. an all-zero row of the constraint matrix); an all-zero
// row changes neither the RREF nor its pivot columns, hence neither the D5b
// kernel basis -- the returned basis is entrywise the one FlatMatrix::kernel()
// returned before.
engine::DenseBuf<uint32_t> HeckeCrossChecks::equivariant_cochain_basis_Lp(uint32_t q) const {
    const SimplicialComplex* Kp = K();
    const uint32_t N_q = Kp->get_num_q_simplices(q);
    const uint32_t dim = N_q * d_ind_;

    auto pool = std::make_shared<engine::BlockPool<uint32_t>>();
    const BlockMat<uint32_t> A = BuildACqBlockMat(pool, *lssc_Lp_, q);
    if (dim == 0 || A.rows() == 0) {
        // No constraints: the whole space is equivariant, and the D5b kernel
        // basis of a matrix with no rows is the standard basis in ascending
        // order. (Does not occur for the 7-vertex model, which has relations
        // in every degree since FIX-negI, but keep it total.)
        DenseBuf<uint32_t> I(dim, dim, Layout::ColMajor);
        for (uint32_t b = 0; b < dim; ++b) I.at(b, b) = 1;
        return I;
    }
    if (A.cols() != dim) {
        throw std::runtime_error("HeckeCrossChecks: L' relation matrix dimension mismatch.");
    }
    const engine::blk::BlockRrefResult<uint32_t> res = engine::blk::rref<uint32_t>(A);
    return engine::blk::kernel_basis<uint32_t>(A, res, Layout::ColMajor).to_flat(Layout::ColMajor);
}

// ---------------------------------------------------------------------------
// A-F3 (i): beta maps L'-equivariant cochains to L''-equivariant cochains
// ---------------------------------------------------------------------------

uint32_t HeckeCrossChecks::check_beta_equivariance() const {
    const auto& relations_Lpp = lssc_Lpp_->get_user_defined_simplex_relations();
    const uint32_t num_degrees = K()->get_dimension() + 1;

    uint64_t tested = 0;
    for (uint32_t q = 0; q < num_degrees; ++q) {
        // The whole basis of C^q_eq(K,R;L') as one column-major multi-vector,
        // and beta applied to ALL of it in one block GEMM (X3).
        const DenseBuf<uint32_t> basis = equivariant_cochain_basis_Lp(q);
        const uint32_t nb = basis.cols;
        if (nb == 0 || basis.rows == 0) continue;
        DenseBuf<uint32_t> out(basis.rows, nb, Layout::ColMajor);
        engine::blk::gemm_dense<uint32_t>(beta_bm_[q], cview(basis), view(out));

        DenseBuf<uint32_t> rhs(d_ind_, nb, Layout::ColMajor);
        const std::vector<uint32_t> dims_m(m_, d_V_);
        for (const auto& rel : relations_Lpp) {
            if (rel.q_dim != q) continue;
            if (rel.M.get_rows() != d_ind_ || rel.M.get_cols() != d_ind_) {
                throw std::runtime_error(
                    "HeckeCrossChecks: L'' relation matrix dimension mismatch.");
            }
            // (beta c')(sigma^-) == M_{Gamma''}(g_r) (beta c')(sigma^+), for
            // every basis vector at once. M_{Gamma''}(g_r) = M_H(g_r) has
            // exactly one nonzero d_V x d_V block per block row ([C, eq:MH],
            // audit 02 Sec. 1.4 T4); SimplexRelation::M stores it densely, so
            // the block structure is recovered here with detect = true and the
            // product skips the m^2 - m zero blocks.
            const BlockMat<uint32_t> Mb = BlockMat<uint32_t>::from_dense(
                pool_, cview(to_buf(rel.M)), dims_m, dims_m, ell_, Layout::RowMajor, true);
            engine::blk::gemm_dense<uint32_t>(
                Mb, cview(out).sub(rel.s_rhs_idx_in_Kq * d_ind_, d_ind_, 0, nb), view(rhs));
            if (!engine::same_entries(
                    cview(rhs), cview(out).sub(rel.s_lhs_idx_in_Kq * d_ind_, d_ind_, 0, nb))) {
                return report_equivariance_failure(q, out);
            }
            tested += nb;
        }
    }
    if (tested == 0) {
        std::cout << "HeckeCrossChecks A-F3(i): no (cochain, relation) pair was "
                     "tested -- vacuous check." << std::endl;
        return 2;
    }
    return 0;
}

uint32_t HeckeCrossChecks::report_equivariance_failure(uint32_t q,
                                                       const engine::DenseBuf<uint32_t>& out) const {
    const auto& relations_Lpp = lssc_Lpp_->get_user_defined_simplex_relations();
    const engine::ModCtx<uint32_t> ctx(ell_);
    for (uint32_t b = 0; b < out.cols; ++b) {
        for (const auto& rel : relations_Lpp) {
            if (rel.q_dim != q) continue;
            const uint32_t lhs_base = rel.s_lhs_idx_in_Kq * d_ind_;
            const uint32_t rhs_base = rel.s_rhs_idx_in_Kq * d_ind_;
            for (uint32_t r = 0; r < d_ind_; ++r) {
                uint32_t rhs_val = 0;
                for (uint32_t c = 0; c < d_ind_; ++c) {
                    rhs_val = ctx.add(rhs_val,
                                      ctx.mul(rel.M.get_entry(r, c) % ell_, out.at(rhs_base + c, b)));
                }
                if (out.at(lhs_base + r, b) != rhs_val) {
                    std::cout << "HeckeCrossChecks A-F3(i) FAILED: beta of an "
                                 "L'-equivariant cochain violates the L'' relation '"
                              << rel.tag << "' at q=" << q << ", basis vector " << b
                              << ", fiber row " << r << ": "
                              << out.at(lhs_base + r, b) << " != " << rhs_val
                              << " (ell=" << ell_ << ", n=" << n_sym_
                              << ", N=" << N32_ << ")." << std::endl;
                    return 1;
                }
            }
        }
    }
    std::cout << "HeckeCrossChecks A-F3(i) FAILED at q=" << q
              << " (batched comparison), but the per-vector rescan found no "
                 "differing entry -- report this, it is an engine bug."
              << std::endl;
    return 1;
}

// ---------------------------------------------------------------------------
// A-F3 (ii): beta . alpha == Xi_g (bit-identically on the standard basis)
// ---------------------------------------------------------------------------
//
// The old code applied the composite to each of the dim_q standard basis
// vectors in turn; the statement is exactly "the two matrices are equal", so it
// is now one block product and one comparison AS MATHEMATICAL MATRICES (audit
// 02 Sec. 6.1, "Engine note": structural zeros on one side must compare against
// materialised zeros on the other, which blk::equal does).
uint32_t HeckeCrossChecks::check_beta_alpha_equals_Xi() const {
    const uint32_t num_degrees = K()->get_dimension() + 1;

    for (uint32_t q = 0; q < num_degrees; ++q) {
        const BlockMat<uint32_t>& via_maps = beta_alpha(q);
        const BlockMat<uint32_t>& via_xi = xi_bm_[q];
        if (via_maps.rows() != via_xi.rows() || via_maps.cols() != via_xi.cols()) {
            std::cout << "HeckeCrossChecks A-F3(ii) FAILED: output length mismatch "
                         "at q=" << q << "." << std::endl;
            return 1;
        }
        if (engine::blk::equal<uint32_t>(via_maps, via_xi)) continue;
        // Same first offender the column-at-a-time code would have reported.
        const DenseBuf<uint32_t> A = via_maps.to_dense(Layout::ColMajor);
        const DenseBuf<uint32_t> X = via_xi.to_dense(Layout::ColMajor);
        for (uint32_t b = 0; b < A.cols; ++b) {
            for (uint32_t r = 0; r < A.rows; ++r) {
                if (A.at(r, b) == X.at(r, b)) continue;
                std::cout << "HeckeCrossChecks A-F3(ii) FAILED: beta.alpha != Xi_g "
                             "at q=" << q << ", basis " << b << ", row " << r << ": "
                          << A.at(r, b) << " != " << X.at(r, b) << " (ell=" << ell_
                          << ", n=" << n_sym_ << ", N=" << N32_ << ")." << std::endl;
                return 2;
            }
        }
        return 2;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Acceptance: omega . beta . alpha == Theta_g (as matrices, all of C^q)
// ---------------------------------------------------------------------------

uint32_t HeckeCrossChecks::check_triple_factorization(const HeckeOperatorOnK& op) const {
    const SimplicialComplex* Kp = K();
    const uint32_t num_degrees = Kp->get_dimension() + 1;
    const std::vector<FlatMatrix>& theta = op.get_Theta_flat_matrices();
    if (theta.size() < num_degrees) {
        std::cout << "HeckeCrossChecks (iii): Theta_flat does not cover all degrees."
                  << std::endl;
        return 1;
    }

    for (uint32_t q = 0; q < num_degrees; ++q) {
        const uint32_t dim = Kp->get_num_q_simplices(q) * d_V_;
        if (theta[q].get_rows() != dim || theta[q].get_cols() != dim ||
            theta[q].get_p() != ell_) {
            std::cout << "HeckeCrossChecks (iii): Theta_flat[" << q
                      << "] shape/prime mismatch." << std::endl;
            return 2;
        }
        // omega . (beta . alpha) on ALL of C^q(K;V) in one block GEMM.
        const BlockMat<uint32_t> composite =
            engine::blk::gemm<uint32_t>(omega_bm_[q], beta_alpha(q));
        if (composite.rows() != dim || composite.cols() != dim) {
            std::cout << "HeckeCrossChecks (iii): composite output length "
                      << composite.rows() << " != " << dim << " at q=" << q << "."
                      << std::endl;
            return 3;
        }
        const DenseBuf<uint32_t> W = composite.to_dense(Layout::ColMajor);
        for (uint32_t b = 0; b < dim; ++b) {
            for (uint32_t r = 0; r < dim; ++r) {
                if (W.at(r, b) == theta[q].get_entry(r, b)) continue;
                std::cout << "HeckeCrossChecks ACCEPTANCE FAILED: "
                             "omega.beta.alpha != Theta_g at q=" << q
                          << ", matrix entry (row " << r << ", col " << b << "): "
                          << W.at(r, b) << " != " << theta[q].get_entry(r, b)
                          << " (ell=" << ell_ << ", n=" << n_sym_ << ", N=" << N32_
                          << ")." << std::endl;
                return 4;
            }
        }
    }
    return 0;
}

uint32_t HeckeCrossChecks::run_all(const HeckeOperatorOnK& op) const {
    // A-F3 order: the two localizing assertions FIRST, then the acceptance
    // identity.
    uint32_t code = check_beta_equivariance();
    if (code != 0) return 100 + code;
    code = check_beta_alpha_equals_Xi();
    if (code != 0) return 200 + code;
    code = check_triple_factorization(op);
    if (code != 0) return 300 + code;
    return 0;
}

} // namespace hecke
