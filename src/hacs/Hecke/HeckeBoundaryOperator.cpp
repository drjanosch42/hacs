// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 1, WP-4b -- the boundary Hecke operator Theta^bd_g on
// the circle model. See HeckeBoundaryOperator.h for the contract map, the
// routes and the record fields; [C] = docs/latex/gl2_contract.tex, by label.

#include "HeckeBoundaryOperator.h"

#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "CoefficientRing.h"
#include "HeckeSettings.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/Kernels.h"

#include <array>
#include <map>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace hecke {

namespace {

using engine::ModCtx;

template<class Elt>
std::array<Elt, 4> reduce_mat22(const Mat22& g, Elt ell) {
    if constexpr (std::is_same_v<Elt, uint32_t>) {
        return g.reduced_mod(ell);
    } else {
        return g.reduced_mod64(ell);
    }
}

template<class Elt>
Elt coeff_mod(int64_t c, Elt ell) {
    const int64_t m = static_cast<int64_t>(ell);
    int64_t r = c % m;
    if (r < 0) r += m;
    return static_cast<Elt>(r);
}

template<class Elt>
bool matrices_equal(const MatrixT<Elt>& X, const MatrixT<Elt>& Y) {
    return X.rows == Y.rows && X.cols == Y.cols && X.data == Y.data;
}

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

// y_block[s] += coeff * B * x_block[t] on flat columns (d x d block B).
template<class Elt>
void accumulate_block_gemv(const ModCtx<Elt>& ctx, std::vector<Elt>& y, uint32_t s, const FlatMatrixT<Elt>& B,
                           const std::vector<Elt>& x, uint32_t t, Elt coeff, uint32_t d) {
    for (uint32_t r = 0; r < d; ++r) {
        Elt acc = 0;
        for (uint32_t c = 0; c < d; ++c) {
            acc = ctx.add(acc, ctx.mul(B.get_entry(r, c), x[static_cast<size_t>(t) * d + c]));
        }
        y[static_cast<size_t>(s) * d + r] = ctx.add(y[static_cast<size_t>(s) * d + r], ctx.mul(coeff, acc));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template<class Elt>
HeckeBoundaryOperatorT<Elt>::HeckeBoundaryOperatorT(const EquivariantCohomologyOnKT<Elt>* cohomology_of_bd,
                                                    const HeckeCosetTablesT<Elt>* cosets,
                                                    const HeckeBoundaryTauTableT<Elt>* taub)
    : coh_(cohomology_of_bd), cosets_(cosets), taub_(taub) {
    // (slot-9 WP-4; E8 item 4b) A circle model over the Z/ell^k IN FORCE is
    // honoured; every other combination keeps slot 8's refusal.
    const uint32_t k_in = coh_ ? coh_->coefficient_ring_k() : 1u;
    if (k_in >= 2 && CoefficientRingK() == k_in) {
        ring_k_ = k_in;
    } else {
        // (slot-8 WP-3) [E] sec:ring (b): the boundary side is item 4b's (slot 9).
        RequireFieldCoefficientRing("HeckeBoundaryOperator", k_in);
    }
    ENGINE_CHECK_MSG(coh_ != nullptr && cosets_ != nullptr && taub_ != nullptr,
                     "HeckeBoundaryOperator: null input.");
    if (!coh_->is_cohomology_computed()) {
        throw std::runtime_error("HeckeBoundaryOperator: the circle-model cohomology must be computed first.");
    }
    if (&taub_->cosets() != cosets_) {
        throw std::invalid_argument("HeckeBoundaryOperator: tau_bd must be built against the SAME coset tables.");
    }
    if (ring_k_ >= 2) {
        const engine::ModCtx<Elt>& c = coh_->ring_context();
        // The default SmallEllPolicy: refuses ell in {2, 3} by name (the Bredon
        // object there has no stored basis), and ell^k beyond the element type.
        ring_ctx_ = CoefficientRingContext<Elt>("HeckeBoundaryOperator", static_cast<uint64_t>(c.ell), c.k);
    }
    ell_ = (ring_k_ >= 2) ? ring_ctx_.ell : coh_->get_prime();   // the tables are keyed on the cell's prime
    d_V_ = coh_->get_representation_dim();
    if (d_V_ == 0) throw std::invalid_argument("HeckeBoundaryOperator: d_V = 0.");
    n_sym_ = d_V_ - 1;
    m_ = cosets_->num_cosets();
    num_degrees_ = taub_->num_degrees();
    if (m_ == 0 || num_degrees_ == 0) throw std::invalid_argument("HeckeBoundaryOperator: empty tables.");

    const CosetRhoT<Elt> rs_block = cosets_->rho_s_inv_block(1);   // (slot-14 WP-10) streamed tables too
    const FlatMatrixT<Elt>& rs = rs_block.get();
    if (rs.get_p() != ell_ || rs.get_rows() != d_V_ || rs.get_cols() != d_V_) {
        throw std::invalid_argument("HeckeBoundaryOperator: coset tables and cohomology disagree on ell / d_V.");
    }
    if ((ring_k_ >= 2 ? coh_->num_degrees() : static_cast<uint32_t>(coh_->get_H_eq_dimensions().size())) < num_degrees_) {
        throw std::runtime_error("HeckeBoundaryOperator: the cohomology object does not report every boundary degree.");
    }

    // K_bd's q-simplices must be K's boundary q-simplices in boundary_simplices(q) order.
    const SimplicialComplex* Kbd = coh_->get_lssc()->get_base_K_complex();
    const auto& Ksimp = taub_->K_simplices();
    const auto& bdsimp = Kbd->get_simplices();
    bd_to_K_.assign(num_degrees_, {});
    K_to_bd_.assign(num_degrees_, {});
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const std::vector<uint32_t>& bl = model().boundary_simplices(q);
        if (Kbd->get_num_q_simplices(q) != bl.size()) {
            throw std::invalid_argument("HeckeBoundaryOperator: K_bd has the wrong number of q-simplices for the boundary.");
        }
        for (uint32_t s = 0; s < bl.size(); ++s) {
            if (bdsimp.at(q).at(s) != Ksimp.at(q).at(bl[s])) {
                std::ostringstream oss;
                oss << "HeckeBoundaryOperator: K_bd simplex " << s << " of degree " << q
                    << " is not K's boundary simplex " << bl[s] << " (vertex lists differ).";
                throw std::invalid_argument(oss.str());
            }
            bd_to_K_[q].push_back(bl[s]);
            K_to_bd_[q][bl[s]] = s;
        }
    }
    for (uint32_t q = num_degrees_; q < bdsimp.size(); ++q) {
        if (!bdsimp[q].empty()) {
            throw std::invalid_argument("HeckeBoundaryOperator: K_bd has simplices above the boundary dimension.");
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::rho_of_exact(const Mat22& gamma_exact) const {
    const std::array<Elt, 4> g_mod = reduce_mat22<Elt>(gamma_exact, ell_);
    const FlatMatrixT<Elt> g_fm(2, 2, ell_, std::vector<Elt>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym_);
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::rho_T_pow(int64_t k) const {
    return rho_of_exact(Mat22{ 1, k, 0, 1 });
}

template<class Elt>
void HeckeBoundaryOperatorT<Elt>::check_degree(uint32_t q) const {
    if (q >= num_degrees_) throw std::out_of_range("HeckeBoundaryOperator: degree out of range.");
}

template<class Elt>
void HeckeBoundaryOperatorT<Elt>::require_computed() const {
    if (!computed_) throw std::runtime_error("HeckeBoundaryOperator: compute() not run.");
}

template<class Elt>
uint32_t HeckeBoundaryOperatorT<Elt>::local_index(uint32_t q, uint32_t K_sigma) const {
    const auto it = K_to_bd_.at(q).find(K_sigma);
    if (it == K_to_bd_[q].end()) {
        throw std::runtime_error("HeckeBoundaryOperator: tau_bd names a simplex that is not a boundary simplex.");
    }
    return it->second;
}

template<class Elt>
uint32_t HeckeBoundaryOperatorT<Elt>::N_q(uint32_t q) const {
    check_degree(q);
    return static_cast<uint32_t>(bd_to_K_[q].size());
}

template<class Elt>
uint32_t HeckeBoundaryOperatorT<Elt>::dim_H(uint32_t q) const {
    check_degree(q);
    return coh_->get_H_eq_dimensions().at(q);
}

template<class Elt>
std::vector<uint32_t> HeckeBoundaryOperatorT<Elt>::reported_degrees() const {
    std::vector<uint32_t> out;
    for (uint32_t q = 0; q < num_degrees_; ++q) out.push_back(q);
    return out;
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::flatten(const FlatMatrixT<Elt>& cochain_rows) const {
    const uint32_t total = cochain_rows.get_rows() * cochain_rows.get_cols();
    if (total == 0) return FlatMatrixT<Elt>::CreateZero(total, 1, ell_);
    cochain_rows.populate_flat_data();
    return FlatMatrixT<Elt>(total, 1, ell_, cochain_rows.get_flat_data_unsafe());
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::flat_coboundary(uint32_t q) const {
    check_degree(q);
    const SimplicialComplex* Kbd = coh_->get_lssc()->get_base_K_complex();
    const uint32_t Nq = N_q(q);
    const uint32_t Nq1 = (q + 1 < num_degrees_) ? N_q(q + 1) : 0u;
    std::vector<Elt> data(static_cast<size_t>(Nq1) * d_V_ * Nq * d_V_, 0);
    if (Nq1 > 0) {
        const FlatMatrixT<Elt> D1 = Kbd->template coboundary_operator_elt<Elt>(q, ell_); // N_{q+1} x N_q
        ENGINE_CHECK_MSG(D1.get_rows() == Nq1 && D1.get_cols() == Nq,
                         "HeckeBoundaryOperator: coboundary operator has the wrong shape.");
        const size_t cols = static_cast<size_t>(Nq) * d_V_;
        for (uint32_t t = 0; t < Nq1; ++t) {
            for (uint32_t s = 0; s < Nq; ++s) {
                const Elt v = D1.get_entry(t, s);
                if (v == 0) continue;
                for (uint32_t k = 0; k < d_V_; ++k) {
                    data[(static_cast<size_t>(t) * d_V_ + k) * cols + static_cast<size_t>(s) * d_V_ + k] = v;
                }
            }
        }
    }
    return FlatMatrixT<Elt>(Nq1 * d_V_, Nq * d_V_, ell_, data);
}

// ---------------------------------------------------------------------------
// The normative operator ([C, eq:bdrychain]) in one bracketing
// ---------------------------------------------------------------------------

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::assemble_theta(uint32_t q, bool fused) const {
    const uint32_t Nq = N_q(q);
    const uint32_t dim = Nq * d_V_;
    if (dim == 0) return FlatMatrixT<Elt>(0, 0, ell_);
    const ModCtx<Elt> ctx(ell_);
    std::vector<Elt> data(static_cast<size_t>(dim) * dim, 0);
    std::map<std::array<int64_t, 5>, FlatMatrixT<Elt>> cache; // (j, gamma) -> block
    const Mat22& g = cosets_->g();
    for (uint32_t s = 0; s < Nq; ++s) {
        const uint32_t sigmaK = bd_to_K_[q][s];
        for (uint32_t j = 1; j <= m_; ++j) {
            for (const ChainTerm& ct : taub_->tau(j, q, sigmaK)) {
                const Elt cmod = coeff_mod<Elt>(ct.coeff, ell_);
                if (cmod == 0) continue;
                const uint32_t t = local_index(q, ct.sigma_idx);
                const std::array<int64_t, 5> key{ static_cast<int64_t>(j), ct.gamma.a, ct.gamma.b, ct.gamma.c, ct.gamma.d };
                auto it = cache.find(key);
                if (it == cache.end()) {
                    FlatMatrixT<Elt> block = fused
                        ? cosets_->rho_s_inv_block(j).get() * rho_of_exact(g * ct.gamma)   // rho(s_j^{-1}) * rho((g gamma) mod ell)
                        : rho_of_exact(cosets_->delta(j) * ct.gamma);           // rho((delta_j gamma) mod ell), exact product
                    it = cache.emplace(key, std::move(block)).first;
                }
                const FlatMatrixT<Elt>& B = it->second;
                for (uint32_t r = 0; r < d_V_; ++r) {
                    for (uint32_t c = 0; c < d_V_; ++c) {
                        const Elt v = B.get_entry(r, c);
                        if (v == 0) continue;
                        Elt& cell = data[(static_cast<size_t>(s) * d_V_ + r) * dim + static_cast<size_t>(t) * d_V_ + c];
                        cell = ctx.add(cell, ctx.mul(cmod, v));
                    }
                }
            }
        }
    }
    return FlatMatrixT<Elt>(dim, dim, ell_, data);
}

template<class Elt>
typename HeckeBoundaryOperatorT<Elt>::RhoMemo::Identity
HeckeBoundaryOperatorT<Elt>::rho_memo_identity() const {
    typename RhoMemo::Identity id;
    id.owner = static_cast<const void*>(this);
    id.delta_table = static_cast<const void*>(cosets_);
    id.ell = static_cast<uint64_t>(ell_);
    id.n_sym = static_cast<uint64_t>(n_sym_);
    id.m = static_cast<uint64_t>(m_);
    return id;
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::apply_direct(uint32_t q, const FlatMatrixT<Elt>& c_flat) const {
    // The pre-slot-5 behaviour verbatim: a fresh, call-local table.
    RhoMemo memo;
    return apply_direct(q, c_flat, memo);
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::apply_direct(uint32_t q, const FlatMatrixT<Elt>& c_flat,
                                                           RhoMemo& memo) const {
    require_field_op("apply_direct");
    const uint32_t Nq = N_q(q);
    const uint32_t dim = Nq * d_V_;
    if (c_flat.get_p() != ell_ || c_flat.get_cols() != 1 || c_flat.get_rows() != dim) {
        throw std::invalid_argument("HeckeBoundaryOperator::apply_direct: c_flat must be an (N_q*d_V) x 1 column over F_ell.");
    }
    if (dim == 0) return FlatMatrixT<Elt>(0, 1, ell_);
    const ModCtx<Elt> ctx(ell_);
    std::vector<Elt> x(dim), y(dim, 0);
    for (uint32_t i = 0; i < dim; ++i) x[i] = c_flat.get_entry(i, 0);
    // (slot-5 WP-6) The (j, gamma) table is the caller's, so it can recur
    // across calls; the identity stamp is the O(1) guard on the part of the
    // value's dependency set that is not in the key.  See Hecke/RhoMemo.h.
    memo.bind(rho_memo_identity());
    typename RhoMemo::Table& cache = memo.table();
    for (uint32_t s = 0; s < Nq; ++s) {
        const uint32_t sigmaK = bd_to_K_[q][s];
        for (uint32_t j = 1; j <= m_; ++j) {
            for (const ChainTerm& ct : taub_->tau(j, q, sigmaK)) {
                const Elt cmod = coeff_mod<Elt>(ct.coeff, ell_);
                if (cmod == 0) continue;
                const uint32_t t = local_index(q, ct.sigma_idx);
                const std::array<int64_t, 5> key{ static_cast<int64_t>(j), ct.gamma.a, ct.gamma.b, ct.gamma.c, ct.gamma.d };
                auto it = cache.find(key);
                if (it == cache.end()) it = cache.emplace(key, rho_of_exact(cosets_->delta(j) * ct.gamma)).first;
                accumulate_block_gemv<Elt>(ctx, y, s, it->second, x, t, cmod, d_V_);
            }
        }
    }
    return FlatMatrixT<Elt>(dim, 1, ell_, y);
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::apply_Theta(uint32_t q, const FlatMatrixT<Elt>& c_flat) const {
    require_computed();
    check_degree(q);
    return Theta_flat_[q] * c_flat;
}

// ---------------------------------------------------------------------------
// Runtime assertions on a cochain operator
// ---------------------------------------------------------------------------

template<class Elt>
void HeckeBoundaryOperatorT<Elt>::certify_cochain_operator(uint32_t q, const FlatMatrixT<Elt>& Theta, const char* what,
                                                           uint64_t& relation_checks, uint64_t& cocycle_checks,
                                                           uint64_t& closure_checks, uint64_t& chain_map_checks) const {
    (void)chain_map_checks; // (iv) needs two degrees; done in compute()
    const ModCtx<Elt> ctx(ell_);
    const uint32_t dim = N_q(q) * d_V_;
    ENGINE_CHECK_MSG(Theta.get_rows() == dim && Theta.get_cols() == dim, "HeckeBoundaryOperator: operator shape.");
    const auto& relations = coh_->get_lssc()->get_user_defined_simplex_relations();
    const auto& H_bases = coh_->get_H_eq_bases_fm().at(q);
    const auto& C_bases = coh_->get_C_eq_bases_fm().at(q);
    const bool has_next = (q + 1 < num_degrees_);
    const FlatMatrixT<Elt> D = has_next ? flat_coboundary(q) : FlatMatrixT<Elt>(0, dim, ell_);

    auto check_relations = [&](const FlatMatrixT<Elt>& y, const char* which, uint32_t idx, uint64_t& counter) {
        for (const auto& rel : relations) {
            if (rel.q_dim != q) continue;
            for (uint32_t r = 0; r < d_V_; ++r) {
                Elt rhs = 0;
                for (uint32_t c = 0; c < d_V_; ++c) {
                    rhs = ctx.add(rhs, ctx.mul(rel.M.get_entry(r, c), y.get_entry(rel.s_rhs_idx_in_Kq * d_V_ + c, 0)));
                }
                const Elt lhs = y.get_entry(rel.s_lhs_idx_in_Kq * d_V_ + r, 0);
                if (lhs != rhs) {
                    std::ostringstream oss;
                    oss << "HeckeBoundaryOperator::compute: " << what << " -- runtime assertion FAILED: " << what
                        << " applied to " << which << " " << idx << " violates relation '" << rel.tag << "' at q=" << q
                        << ", fiber row " << r << " (lhs " << lhs << " != rhs " << rhs << ").";
                    throw std::runtime_error(oss.str());
                }
            }
            ++counter;
        }
    };

    // (i) and (ii) on the H_eq basis.
    for (uint32_t i = 0; i < H_bases.size(); ++i) {
        const FlatMatrixT<Elt> y = Theta * flatten(H_bases[i]);
        check_relations(y, "H_eq basis cocycle", i, relation_checks);
        if (has_next) {
            const FlatMatrixT<Elt> dy = D * y;
            if (!dy.is_zero()) {
                std::ostringstream oss;
                oss << "HeckeBoundaryOperator::compute: " << what << " -- runtime assertion (ii) FAILED: d(" << what
                    << " z_" << i << ") != 0 at q=" << q << ".";
                throw std::runtime_error(oss.str());
            }
            ++cocycle_checks;
        }
    }
    // (iii) relation closure on the C_eq basis ([C, prop:bdryop](1): the operator preserves C^*_bd).
    for (uint32_t j = 0; j < C_bases.size(); ++j) {
        const FlatMatrixT<Elt> w = Theta * flatten(C_bases[j]);
        check_relations(w, "C_eq basis vector", j, closure_checks);
    }
}

template<class Elt>
MatrixT<Elt> HeckeBoundaryOperatorT<Elt>::assemble_A_from_flat_images(uint32_t q,
                                                                      const std::vector<FlatMatrixT<Elt>>& y_flat) const {
    const uint32_t d_q = dim_H(q);
    const uint32_t dim = N_q(q) * d_V_;
    MatrixT<Elt> A(d_q, d_q, ell_);
    if (d_q == 0) return A;
    ENGINE_CHECK_MSG(static_cast<uint32_t>(y_flat.size()) == d_q, "HeckeBoundaryOperator: image count != dim H.");
    const FlatMatrixT<Elt>& P = coh_->get_H_eq_projectors_from_C_eq_flat_fm().at(q);
    if (P.get_rows() != d_q || P.get_cols() != dim) {
        std::ostringstream oss;
        oss << "HeckeBoundaryOperator: projector P_" << q << " has shape " << P.get_rows() << "x" << P.get_cols()
            << ", expected " << d_q << "x" << dim << ".";
        throw std::runtime_error(oss.str());
    }
    for (uint32_t i = 0; i < d_q; ++i) {
        const FlatMatrixT<Elt> coords = P * y_flat[i];
        for (uint32_t j = 0; j < d_q; ++j) A.data[j][i] = coords.get_entry(j, 0);
    }
    return A;
}

// ---------------------------------------------------------------------------
// Route A -- the Shapiro transcription on the infinity-line
// ---------------------------------------------------------------------------

template<class Elt>
int64_t HeckeBoundaryOperatorT<Elt>::shapiro_vertex_image(uint32_t which, int64_t P) const {
    int64_t num = 0, den = 1;
    model().boundary_line_real_part(P, num, den);
    const int64_t p = static_cast<int64_t>(p_hecke());
    if (which == 0) return model().boundary_line_nearest(checked_mul<int64_t>(p, num), den); // tau_inf: Re -> p Re
    if (which == 1) return model().boundary_line_nearest(num, checked_mul<int64_t>(p, den)); // tau_0: Re -> Re / p
    throw std::out_of_range("HeckeBoundaryOperator::shapiro_vertex_image: which in {0, 1}.");
}

template<class Elt>
std::vector<typename HeckeBoundaryOperatorT<Elt>::LineChainTerm>
HeckeBoundaryOperatorT<Elt>::line_walk(int64_t from, int64_t to) const {
    std::vector<LineChainTerm> out;
    if (from <= to) {
        for (int64_t Q = from; Q < to; ++Q) out.push_back(LineChainTerm{ 1, Q });
    } else {
        for (int64_t Q = to; Q < from; ++Q) out.push_back(LineChainTerm{ -1, Q });
    }
    return out;
}

template<class Elt>
void HeckeBoundaryOperatorT<Elt>::certify_line_datum() const {
    ENGINE_CHECK_MSG(cosets_->r() == 1, "HeckeBoundaryOperator: route A needs nu = 1.");
    const int64_t p = static_cast<int64_t>(p_hecke());
    const int64_t period = static_cast<int64_t>(model().boundary_period());
    // The twisted equivariances (T1) of the two line data, exactly, on a window.
    for (int64_t P = -period * (p + 1); P <= 2 * period * (p + 1); ++P) {
        ENGINE_CHECK_MSG(shapiro_vertex_image(0, P + period) == shapiro_vertex_image(0, P) + period * p,
                         "route A: tau_inf(T x) != T^p tau_inf(x).");
        ENGINE_CHECK_MSG(shapiro_vertex_image(1, P + period * p) == shapiro_vertex_image(1, P) + period,
                         "route A: tau_0(T^p x) != T tau_0(x).");
    }
    // The canonical representatives delta_inf = diag(1, p), delta_t = T^t delta_0 = [[p, t],[0, 1]].
    const auto& bc = taub_->boundary_cosets();
    ENGINE_CHECK_MSG(bc.size() == static_cast<size_t>(p) + 1, "route A: |B_p / Gamma_inf| != p + 1.");
    ENGINE_CHECK_MSG((bc[0].delta == Mat22{ 1, 0, 0, p }), "route A: delta_inf != diag(1, p).");
    for (int64_t t = 0; t < p; ++t) {
        ENGINE_CHECK_MSG((bc[static_cast<size_t>(1 + t)].delta == Mat22{ p, t, 0, 1 }), "route A: delta_t != [[p, t],[0, 1]].");
    }
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::assemble_theta_shapiro(uint32_t q) const {
    const uint32_t Nq = N_q(q);
    const uint32_t dim = Nq * d_V_;
    if (dim == 0) return FlatMatrixT<Elt>(0, 0, ell_);
    if (q > 1) throw std::runtime_error("HeckeBoundaryOperator: route A is written for a 1-dimensional boundary.");
    const ModCtx<Elt> ctx(ell_);
    const SL2ZTessellationTables& M = model();
    const int64_t p = static_cast<int64_t>(p_hecke());
    const int64_t period = static_cast<int64_t>(M.boundary_period());
    const auto& bc = taub_->boundary_cosets();

    // Line position of every K_bd slot (q = 0: vertex positions 0..period; q = 1:
    // edge positions 0..period-1 with the orientation sign of K's edge).
    std::vector<int64_t> slot_pos(Nq, 0);
    std::vector<int32_t> slot_sign(Nq, 1);
    if (q == 0) {
        for (uint32_t s = 0; s < Nq; ++s) slot_pos[s] = static_cast<int64_t>(M.boundary_line_index(bd_to_K_[0][s]));
    } else {
        std::vector<char> seen(Nq, 0);
        for (int64_t P = 0; P < period; ++P) {
            const auto le = M.boundary_line_edge(P);
            ENGINE_CHECK_MSG(le.gamma == Mat22::I(), "route A: fundamental-path edge with a non-trivial transporter.");
            const uint32_t s = local_index(1, le.edge_idx);
            slot_pos[s] = P;
            slot_sign[s] = le.coeff; // line edge P = coeff * (K edge)  =>  K edge = coeff * (line edge P)
            seen[s] = 1;
        }
        for (uint32_t s = 0; s < Nq; ++s) ENGINE_CHECK_MSG(seen[s], "route A: an edge slot has no line position.");
    }

    std::vector<Elt> data(static_cast<size_t>(dim) * dim, 0);
    std::map<std::array<int64_t, 5>, FlatMatrixT<Elt>> cache; // (coset index, exact gamma) -> rho(delta gamma)
    auto rho_delta_gamma = [&](uint32_t k, const Mat22& gamma) -> const FlatMatrixT<Elt>& {
        const std::array<int64_t, 5> key{ static_cast<int64_t>(k), gamma.a, gamma.b, gamma.c, gamma.d };
        auto it = cache.find(key);
        if (it == cache.end()) it = cache.emplace(key, rho_of_exact(bc[k].delta * gamma)).first;
        return it->second;
    };
    auto add_block = [&](uint32_t s, uint32_t t, const FlatMatrixT<Elt>& B, int64_t coeff) {
        const Elt cmod = coeff_mod<Elt>(coeff, ell_);
        if (cmod == 0) return;
        for (uint32_t r = 0; r < d_V_; ++r) {
            for (uint32_t c = 0; c < d_V_; ++c) {
                const Elt v = B.get_entry(r, c);
                if (v == 0) continue;
                Elt& cell = data[(static_cast<size_t>(s) * d_V_ + r) * dim + static_cast<size_t>(t) * d_V_ + c];
                cell = ctx.add(cell, ctx.mul(cmod, v));
            }
        }
    };
    // phi evaluated on a line vertex / line edge at position Q, through
    // Gamma_inf-equivariance; the block goes to the slot the position reduces to.
    auto add_line_term = [&](uint32_t s, uint32_t k, int64_t coeff, int64_t Q) {
        if (q == 0) {
            const auto vt = M.boundary_line_vertex(Q); // (T^k, K vertex id)
            add_block(s, local_index(0, vt.second), rho_delta_gamma(k, vt.first), coeff);
        } else {
            const auto le = M.boundary_line_edge(Q); // coeff * (gamma . K edge)
            add_block(s, local_index(1, le.edge_idx), rho_delta_gamma(k, le.gamma), coeff * le.coeff);
        }
    };
    // The image chain of the slot at line position P under a vertex map tau
    // (which = 0: tau_inf, 1: tau_0), as (coeff, position) terms.
    auto image_chain = [&](uint32_t which, int64_t P) -> std::vector<LineChainTerm> {
        if (q == 0) return { LineChainTerm{ 1, shapiro_vertex_image(which, P) } };
        return line_walk(shapiro_vertex_image(which, P), shapiro_vertex_image(which, P + 1));
    };

    for (uint32_t s = 0; s < Nq; ++s) {
        const int64_t P = slot_pos[s];
        const int64_t sgn = slot_sign[s];
        // D_inf: one left coset, delta_inf, transporter I.
        for (const LineChainTerm& lt : image_chain(0, P)) add_line_term(s, 0, sgn * lt.coeff, lt.position);
        // D_0: left cosets delta_t = T^t delta_0 = s_t^{-1} delta_0 with s_t = T^{-t}:
        // the term rho(delta_t) phi(tau_0(s_t . x)), s_t shifting positions by -period*t.
        for (int64_t t = 0; t < p; ++t) {
            for (const LineChainTerm& lt : image_chain(1, P - period * t)) {
                add_line_term(s, static_cast<uint32_t>(1 + t), sgn * lt.coeff, lt.position);
            }
        }
    }
    return FlatMatrixT<Elt>(dim, dim, ell_, data);
}

// ---------------------------------------------------------------------------
// The coordinates of [C, prop:circlemodel] and route B
// ---------------------------------------------------------------------------

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::class_representative(uint32_t q, const FlatMatrixT<Elt>& c_flat) const {
    check_degree(q);
    const uint32_t dim = N_q(q) * d_V_;
    if (c_flat.get_p() != ell_ || c_flat.get_cols() != 1 || c_flat.get_rows() != dim) {
        throw std::invalid_argument("HeckeBoundaryOperator::class_representative: bad cochain shape.");
    }
    const ModCtx<Elt> ctx(ell_);
    const SL2ZTessellationTables& M = model();
    std::vector<Elt> m(d_V_, 0);
    if (q == 0) {
        const uint32_t s0 = local_index(0, M.boundary_line_vertex(0).second); // the slot of v0
        for (uint32_t k = 0; k < d_V_; ++k) m[k] = c_flat.get_entry(s0 * d_V_ + k, 0);
    } else if (q == 1) {
        const int64_t period = static_cast<int64_t>(M.boundary_period());
        for (int64_t P = 0; P < period; ++P) {
            const auto le = M.boundary_line_edge(P);
            ENGINE_CHECK_MSG(le.gamma == Mat22::I(), "class_representative: fundamental-path edge with a transporter.");
            const uint32_t s = local_index(1, le.edge_idx);
            for (uint32_t k = 0; k < d_V_; ++k) {
                const Elt v = c_flat.get_entry(s * d_V_ + k, 0);
                m[k] = (le.coeff >= 0) ? ctx.add(m[k], v) : ctx.sub(m[k], v);
            }
        }
    } else {
        throw std::runtime_error("HeckeBoundaryOperator::class_representative: q > 1 has no classes on a line.");
    }
    return FlatMatrixT<Elt>(d_V_, 1, ell_, m);
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::lift_class(uint32_t q, const FlatMatrixT<Elt>& mvec) const {
    check_degree(q);
    if (mvec.get_p() != ell_ || mvec.get_cols() != 1 || mvec.get_rows() != d_V_) {
        throw std::invalid_argument("HeckeBoundaryOperator::lift_class: m must be d_V x 1.");
    }
    const ModCtx<Elt> ctx(ell_);
    const SL2ZTessellationTables& M = model();
    const uint32_t Nq = N_q(q);
    std::vector<Elt> y(static_cast<size_t>(Nq) * d_V_, 0);
    if (q == 0) {
        const int64_t period = static_cast<int64_t>(M.boundary_period());
        for (uint32_t s = 0; s < Nq; ++s) {
            const int64_t idx = static_cast<int64_t>(M.boundary_line_index(bd_to_K_[0][s])); // 0..period
            const FlatMatrixT<Elt> val = (idx / period == 0) ? mvec : rho_T_pow(idx / period) * mvec;
            for (uint32_t k = 0; k < d_V_; ++k) y[static_cast<size_t>(s) * d_V_ + k] = val.get_entry(k, 0);
        }
    } else if (q == 1) {
        const auto le = M.boundary_line_edge(0);
        const uint32_t s = local_index(1, le.edge_idx);
        for (uint32_t k = 0; k < d_V_; ++k) {
            const Elt v = mvec.get_entry(k, 0);
            y[static_cast<size_t>(s) * d_V_ + k] = (le.coeff >= 0) ? v : ctx.neg(v);
        }
    } else {
        throw std::runtime_error("HeckeBoundaryOperator::lift_class: q > 1.");
    }
    return FlatMatrixT<Elt>(Nq * d_V_, 1, ell_, y);
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::class_coordinates(uint32_t q, const FlatMatrixT<Elt>& mvec) const {
    const FlatMatrixT<Elt>& P = coh_->get_H_eq_projectors_from_C_eq_flat_fm().at(q);
    return P * lift_class(q, mvec);
}

template<class Elt>
MatrixT<Elt> HeckeBoundaryOperatorT<Elt>::matrix_on_H_from_fiber_operator(uint32_t q, const FlatMatrixT<Elt>& Mop,
                                                                          bool& descends_out) const {
    check_degree(q);
    if (Mop.get_p() != ell_ || Mop.get_rows() != d_V_ || Mop.get_cols() != d_V_) {
        throw std::invalid_argument("HeckeBoundaryOperator::matrix_on_H_from_fiber_operator: M must be d_V x d_V.");
    }
    const ModCtx<Elt> ctx(ell_);
    const uint32_t d_q = dim_H(q);
    const uint32_t dim = N_q(q) * d_V_;
    const auto& relations = coh_->get_lssc()->get_user_defined_simplex_relations();
    const bool has_next = (q + 1 < num_degrees_);
    const FlatMatrixT<Elt> D = has_next ? flat_coboundary(q) : FlatMatrixT<Elt>(0, dim, ell_);
    auto is_cocycle_in_C_eq = [&](const FlatMatrixT<Elt>& y) -> bool {
        for (const auto& rel : relations) {
            if (rel.q_dim != q) continue;
            for (uint32_t r = 0; r < d_V_; ++r) {
                Elt rhs = 0;
                for (uint32_t c = 0; c < d_V_; ++c) {
                    rhs = ctx.add(rhs, ctx.mul(rel.M.get_entry(r, c), y.get_entry(rel.s_rhs_idx_in_Kq * d_V_ + c, 0)));
                }
                if (y.get_entry(rel.s_lhs_idx_in_Kq * d_V_ + r, 0) != rhs) return false;
            }
        }
        if (has_next && !(D * y).is_zero()) return false;
        return true;
    };
    descends_out = true;
    MatrixT<Elt> A(d_q, d_q, ell_);
    if (d_q == 0) return A;
    const FlatMatrixT<Elt>& P = coh_->get_H_eq_projectors_from_C_eq_flat_fm().at(q);
    const auto& H_bases = coh_->get_H_eq_bases_fm().at(q);
    const auto& B_bases = coh_->get_B_eq_bases_fm().at(q);
    for (uint32_t i = 0; i < d_q; ++i) {
        const FlatMatrixT<Elt> y = lift_class(q, Mop * class_representative(q, flatten(H_bases[i])));
        if (!is_cocycle_in_C_eq(y)) descends_out = false;
        const FlatMatrixT<Elt> coords = P * y;
        for (uint32_t j = 0; j < d_q; ++j) A.data[j][i] = coords.get_entry(j, 0);
    }
    for (const FlatMatrixT<Elt>& b : B_bases) {
        const FlatMatrixT<Elt> y = lift_class(q, Mop * class_representative(q, flatten(b)));
        if (!is_cocycle_in_C_eq(y) || !(P * y).is_zero()) descends_out = false;
    }
    return A;
}

template<class Elt>
bool HeckeBoundaryOperatorT<Elt>::groupform_available(uint32_t q) const {
    check_degree(q);
    if (q == 0) return true;
    if (q == 1) return cosets_->r() == 1;
    return false;
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::groupform_fiber_operator(uint32_t q) const {
    check_degree(q);
    const auto& bc = taub_->boundary_cosets();
    if (q == 0) {
        // [C, prop:bdryop](2): sum over B_N / Gamma_inf, every nu.
        FlatMatrixT<Elt> Mop = FlatMatrixT<Elt>::CreateZero(d_V_, d_V_, ell_);
        for (const BoundaryCoset& c : bc) Mop = Mop + rho_of_exact(c.delta);
        return Mop;
    }
    if (q == 1 && cosets_->r() == 1) {
        // [C, eq:bdrygroupform]: rho(delta_inf) (sum_{a<p} rho(T)^a) + rho(delta_0).
        const int64_t p = static_cast<int64_t>(p_hecke());
        ENGINE_CHECK_MSG(bc.size() >= 2 && bc[0].u == 0 && bc[1].u == 1 && bc[1].b == 0,
                         "groupform: unexpected boundary coset table.");
        FlatMatrixT<Elt> S = FlatMatrixT<Elt>::CreateZero(d_V_, d_V_, ell_);
        for (int64_t a = 0; a < p; ++a) S = S + rho_T_pow(a);
        return rho_of_exact(bc[0].delta) * S + rho_of_exact(bc[1].delta);
    }
    throw std::runtime_error("HeckeBoundaryOperator::groupform_fiber_operator: not available in this degree / level.");
}

template<class Elt>
FlatMatrixT<Elt> HeckeBoundaryOperatorT<Elt>::beta_control_fiber_operator(uint32_t q) const {
    check_degree(q);
    if (cosets_->r() != 1) throw std::runtime_error("HeckeBoundaryOperator::beta_control_fiber_operator: nu = 1 only.");
    const int64_t p = static_cast<int64_t>(p_hecke());
    const Mat22 beta_inf{ p, 0, 0, 1 };
    if (q == 0) {
        FlatMatrixT<Elt> Mop = rho_of_exact(beta_inf);
        for (int64_t b = 0; b < p; ++b) Mop = Mop + rho_of_exact(Mat22{ 1, b, 0, p });
        return Mop;
    }
    if (q == 1) {
        FlatMatrixT<Elt> S = FlatMatrixT<Elt>::CreateZero(d_V_, d_V_, ell_);
        for (int64_t a = 0; a < p; ++a) S = S + rho_T_pow(a);
        return rho_of_exact(beta_inf) * S + rho_of_exact(Mat22{ 1, 0, 0, p });
    }
    throw std::runtime_error("HeckeBoundaryOperator::beta_control_fiber_operator: q > 1.");
}

// ---------------------------------------------------------------------------
// Oracle support
// ---------------------------------------------------------------------------

template<class Elt>
std::vector<FlatMatrixT<Elt>> HeckeBoundaryOperatorT<Elt>::invariant_monomial_basis(uint32_t n_sym, Elt ell) {
    const ModCtx<Elt> ctx(ell);
    const uint32_t d = n_sym + 1;
    const uint64_t ell64 = static_cast<uint64_t>(ell);
    const uint64_t bmax = static_cast<uint64_t>(n_sym) / ell64; // floor(n / ell)
    std::vector<FlatMatrixT<Elt>> out;
    for (uint64_t b = 0; b <= bmax; ++b) {
        // x^{n - b ell} Nm^b = sum_i C(b, i) (-1)^{b-i} x^{n - b ell + (ell-1)(b-i)} y^{ell i + b - i}
        //                    = sum_i C(b, i) (-1)^{b-i} u_{(ell-1) i + b}.
        std::vector<Elt> w(d, 0);
        uint64_t binom = 1; // C(b, i), exact (b <= n / 2 is tiny)
        for (uint64_t i = 0; i <= b; ++i) {
            if (i > 0) binom = binom * (b - i + 1) / i;
            const uint64_t k = (ell64 - 1) * i + b;
            ENGINE_CHECK_MSG(k <= n_sym, "invariant_monomial_basis: index out of range.");
            Elt cval = static_cast<Elt>(binom % ell64);
            if (((b - i) & 1u) != 0) cval = ctx.neg(cval);
            w[k] = ctx.add(w[k], cval);
        }
        out.emplace_back(d, 1, ell, w);
    }
    return out;
}

template<class Elt>
Elt HeckeBoundaryOperatorT<Elt>::chi(Elt ell, uint64_t p, uint32_t n_sym, uint32_t b) {
    ENGINE_CHECK_MSG(b <= n_sym + 1, "chi: b > n + 1.");
    const ModCtx<Elt> ctx(ell);
    const Elt pm = static_cast<Elt>(p % static_cast<uint64_t>(ell));
    return ctx.add(ctx.pow(pm, static_cast<Elt>(b)), ctx.pow(pm, static_cast<Elt>(n_sym + 1 - b)));
}

// ---------------------------------------------------------------------------
// compute()
// ---------------------------------------------------------------------------

template<class Elt>
void HeckeBoundaryOperatorT<Elt>::compute() {
    if (computed_) return;
    if (ring_k_ >= 2) {   // (slot-9 WP-4) the module route; the field route below is untouched
        compute_module();
        computed_ = true;
        return;
    }
    std::vector<FlatMatrixT<Elt>> theta(num_degrees_);
    std::vector<MatrixT<Elt>> A(num_degrees_);
    CertificateBundle cert;

    // Level (a): the two bracketings of [C, eq:bdrychain], bit-identical.
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        FlatMatrixT<Elt> fused = assemble_theta(q, /*fused=*/true);
        const FlatMatrixT<Elt> direct = assemble_theta(q, /*fused=*/false);
        fused.populate_flat_data();
        direct.populate_flat_data();
        const std::vector<Elt>& a = fused.get_flat_data_unsafe();
        const std::vector<Elt>& b = direct.get_flat_data_unsafe();
        const uint32_t dim = N_q(q) * d_V_;
        const size_t n_entries = static_cast<size_t>(dim) * dim;
        ENGINE_CHECK_MSG(a.size() == n_entries && b.size() == n_entries, "HeckeBoundaryOperator: Theta buffer size.");
        if (n_entries != 0 && std::memcmp(a.data(), b.data(), n_entries * sizeof(Elt)) != 0) {
            size_t bad = 0;
            while (bad < n_entries && a[bad] == b[bad]) ++bad;
            std::ostringstream oss;
            oss << "HeckeBoundaryOperator::compute: the fused bracketing rho(s_j^{-1}) rho(g gamma) and the direct "
                   "rho((delta_j gamma) mod ell) of [C, eq:bdrychain] differ at q=" << q << ", entry ("
                << (bad / dim) << "," << (bad % dim) << "): " << a[bad] << " vs " << b[bad] << ".";
            throw std::runtime_error(oss.str());
        }
        theta[q] = std::move(fused);
    }
    cert.routes_bit_identical = true;

    // (i)-(iii) and level (b).
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        certify_cochain_operator(q, theta[q], "Theta^bd", cert.relation_checks, cert.cocycle_checks,
                                 cert.relation_closure_checks, cert.chain_map_checks);
        const auto& H_bases = coh_->get_H_eq_bases_fm().at(q);
        std::vector<FlatMatrixT<Elt>> y_flat;
        y_flat.reserve(H_bases.size());
        for (const FlatMatrixT<Elt>& z : H_bases) y_flat.push_back(theta[q] * flatten(z));
        A[q] = assemble_A_from_flat_images(q, y_flat);
    }
    // (iv) d Theta = Theta d on the C_eq basis ([C, prop:bdryop](1)).
    for (uint32_t q = 0; q + 1 < num_degrees_; ++q) {
        const FlatMatrixT<Elt> D = flat_coboundary(q);
        const auto& C_bases = coh_->get_C_eq_bases_fm().at(q);
        for (uint32_t j = 0; j < C_bases.size(); ++j) {
            const FlatMatrixT<Elt> c = flatten(C_bases[j]);
            const FlatMatrixT<Elt> lhs = D * (theta[q] * c);
            const FlatMatrixT<Elt> rhs = theta[q + 1] * (D * c);
            if (!(lhs == rhs)) {
                std::ostringstream oss;
                oss << "HeckeBoundaryOperator::compute: runtime assertion (iv) FAILED -- d Theta^bd != Theta^bd d on "
                       "C_eq basis vector " << j << " at q=" << q << ".";
                throw std::runtime_error(oss.str());
            }
            ++cert.chain_map_checks;
        }
    }

    // Route A ([C, obl:taubdry](v)), nu = 1.
    std::vector<FlatMatrixT<Elt>> theta_A;
    std::vector<MatrixT<Elt>> A_A;
    if (cosets_->r() == 1) {
        certify_line_datum();
        theta_A.resize(num_degrees_);
        A_A.resize(num_degrees_);
        uint64_t dummy_a = 0, dummy_b = 0, dummy_c = 0, dummy_d = 0;
        for (uint32_t q = 0; q < num_degrees_; ++q) {
            theta_A[q] = assemble_theta_shapiro(q);
            certify_cochain_operator(q, theta_A[q], "route A (Shapiro transcription)", dummy_a, dummy_b, dummy_c, dummy_d);
            const auto& H_bases = coh_->get_H_eq_bases_fm().at(q);
            std::vector<FlatMatrixT<Elt>> y_flat;
            for (const FlatMatrixT<Elt>& z : H_bases) y_flat.push_back(theta_A[q] * flatten(z));
            A_A[q] = assemble_A_from_flat_images(q, y_flat);
            if (!matrices_equal(A_A[q], A[q])) {
                std::ostringstream oss;
                oss << "HeckeBoundaryOperator::compute: [C, obl:taubdry](v) FAILED -- route A (the Shapiro transcription "
                       "on the infinity-line with the left cosets delta_inf, delta_t) differs from [C, eq:bdrychain] on H^"
                    << q << ".";
                throw std::runtime_error(oss.str());
            }
        }
        for (uint32_t q = 0; q + 1 < num_degrees_; ++q) {
            const FlatMatrixT<Elt> D = flat_coboundary(q);
            for (const FlatMatrixT<Elt>& cb : coh_->get_C_eq_bases_fm().at(q)) {
                const FlatMatrixT<Elt> c = flatten(cb);
                if (!(D * (theta_A[q] * c) == theta_A[q + 1] * (D * c))) {
                    throw std::runtime_error("HeckeBoundaryOperator::compute: route A is not a chain map.");
                }
            }
        }
        cert.shapiro_available = true;
        cert.shapiro_agrees = true;
    }

    // Route B (the closed forms), every available degree.
    std::vector<MatrixT<Elt>> A_B(num_degrees_);
    std::vector<char> B_avail(num_degrees_, 0);
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        if (!groupform_available(q)) continue;
        bool descends = false;
        A_B[q] = matrix_on_H_from_fiber_operator(q, groupform_fiber_operator(q), descends);
        if (!descends) {
            std::ostringstream oss;
            oss << "HeckeBoundaryOperator::compute: route B's fiber operator does not descend to H^" << q << ".";
            throw std::runtime_error(oss.str());
        }
        if (!matrices_equal(A_B[q], A[q])) {
            std::ostringstream oss;
            oss << "HeckeBoundaryOperator::compute: [C, obl:taubdry](v) FAILED -- route B ("
                << (q == 0 ? "sum_{delta in B_N/Gamma_inf} rho_n(delta) on V^T, [C, prop:bdryop](2)"
                           : "[C, eq:bdrygroupform] on V_T")
                << ") differs from [C, eq:bdrychain] on H^" << q << ".";
            throw std::runtime_error(oss.str());
        }
        B_avail[q] = 1;
    }
    cert.groupform_agrees = true;

    Theta_flat_ = std::move(theta);
    A_ = std::move(A);
    Theta_shapiro_ = std::move(theta_A);
    A_shapiro_ = std::move(A_A);
    A_groupform_ = std::move(A_B);
    groupform_available_ = std::move(B_avail);
    cert_ = cert;
    computed_ = true;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template<class Elt>
const FlatMatrixT<Elt>& HeckeBoundaryOperatorT<Elt>::Theta_flat(uint32_t q) const {
    require_field_op("Theta_flat");
    require_computed();
    check_degree(q);
    return Theta_flat_[q];
}

template<class Elt>
const MatrixT<Elt>& HeckeBoundaryOperatorT<Elt>::A(uint32_t q) const {
    require_field_op("A");
    require_computed();
    check_degree(q);
    return A_[q];
}

template<class Elt>
const std::vector<MatrixT<Elt>>& HeckeBoundaryOperatorT<Elt>::get_H_matrices() const {
    require_field_op("get_H_matrices");
    require_computed();
    return A_;
}

template<class Elt>
const FlatMatrixT<Elt>& HeckeBoundaryOperatorT<Elt>::Theta_shapiro_flat(uint32_t q) const {
    require_field_op("Theta_shapiro_flat");
    require_computed();
    check_degree(q);
    if (!cert_.shapiro_available) throw std::runtime_error("HeckeBoundaryOperator: route A is available at nu = 1 only.");
    return Theta_shapiro_[q];
}

template<class Elt>
const MatrixT<Elt>& HeckeBoundaryOperatorT<Elt>::A_shapiro(uint32_t q) const {
    require_field_op("A_shapiro");
    require_computed();
    check_degree(q);
    if (!cert_.shapiro_available) throw std::runtime_error("HeckeBoundaryOperator: route A is available at nu = 1 only.");
    return A_shapiro_[q];
}

template<class Elt>
const MatrixT<Elt>& HeckeBoundaryOperatorT<Elt>::A_groupform(uint32_t q) const {
    require_field_op("A_groupform");
    require_computed();
    check_degree(q);
    if (!groupform_available_[q]) throw std::runtime_error("HeckeBoundaryOperator: route B not available in this degree.");
    return A_groupform_[q];
}

template<class Elt>
typename HeckeBoundaryOperatorT<Elt>::CertificateBundle HeckeBoundaryOperatorT<Elt>::certificates() const {
    require_computed();
    return cert_;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

template<class Elt>
std::string HeckeBoundaryOperatorT<Elt>::to_string_offset(uint32_t offset) const {
    require_field_op("to_string");
    const std::string pad(offset, ' ');
    std::ostringstream oss;
    oss << pad << "HeckeBoundaryOperator: ell=" << ell_ << " n_sym=" << n_sym_ << " p=" << p_hecke() << " nu=" << nu()
        << " N=" << N() << " m=" << m_ << " boundary_model=" << boundary_model()
        << " boundary_is_group_cohomology=" << (boundary_is_group_cohomology() ? "true" : "false")
        << " q_max=" << q_max() << " (" << q_max_source() << ") truncated=" << (truncated() ? "true" : "false")
        << (computed_ ? "" : " [not computed]") << "\n";
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        oss << pad << "  q=" << q << ": N_q=" << N_q(q) << " dim C_eq=" << coh_->C_eq_dimension(q) << " dim H=" << dim_H(q);
        if (computed_) {
            oss << " A_bd=" << FlatMatrixT<Elt>(A_[q]).to_string();
        }
        oss << "\n";
    }
    if (computed_) {
        oss << pad << "  certificate: routes_bit_identical=" << cert_.routes_bit_identical
            << " relation_checks=" << cert_.relation_checks << " cocycle_checks=" << cert_.cocycle_checks
            << " relation_closure_checks=" << cert_.relation_closure_checks << " chain_map_checks=" << cert_.chain_map_checks
            << " shapiro_available=" << cert_.shapiro_available << " shapiro_agrees=" << cert_.shapiro_agrees
            << " groupform_agrees=" << cert_.groupform_agrees << "\n";
    }
    return oss.str();
}

template<class Elt>
std::string HeckeBoundaryOperatorT<Elt>::to_json() const {
    require_field_op("to_json");
    require_computed();
    std::ostringstream oss;
    oss << "{";
    oss << "\"schema\":\"hecke_boundary_operator/v1\",";
    oss << "\"ell\":" << ell_ << ",";
    oss << "\"n_sym\":" << n_sym_ << ",";
    oss << "\"d_V\":" << d_V_ << ",";
    oss << "\"p_hecke\":" << p_hecke() << ",";
    oss << "\"nu\":" << nu() << ",";
    oss << "\"N\":" << N() << ",";
    oss << "\"num_cosets\":" << m_ << ",";
    oss << "\"boundary_model\":\"" << boundary_model() << "\",";
    oss << "\"boundary_is_group_cohomology\":" << (boundary_is_group_cohomology() ? "true" : "false") << ",";
    oss << "\"q_max\":" << q_max() << ",";
    oss << "\"q_max_source\":\"" << q_max_source() << "\",";
    oss << "\"truncated\":" << (truncated() ? "true" : "false") << ",";
    oss << "\"reported_degrees\":[";
    for (uint32_t q = 0; q < num_degrees_; ++q) oss << (q > 0 ? "," : "") << q;
    oss << "],";
    oss << "\"matrix_entry_order\":\"arrays of rows (row-major); entries are canonical residues in [0, ell)\",";
    oss << "\"flat_layout\":\"simplex-major, fiber-minor: flat index = s*d_V + k (K_bd simplex s in "
           "boundary_simplices order, fiber coord k in the basis u_0 = x^n, ..., u_n = y^n)\",";
    oss << "\"column_convention\":\"T_p^bd[z_i] = sum_j A[j][i] [z_j]\",";
    oss << "\"degrees\":[";
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        if (q > 0) oss << ",";
        oss << "{\"q\":" << q << ",";
        oss << "\"N_q\":" << N_q(q) << ",";
        oss << "\"dim_C_eq\":" << coh_->C_eq_dimension(q) << ",";
        oss << "\"dim_H\":" << A_[q].rows << ",";
        oss << "\"Theta_flat\":";
        emit_flat_matrix_json(oss, Theta_flat_[q]);
        oss << ",\"A\":";
        emit_matrix_json(oss, A_[q]);
        if (cert_.shapiro_available) {
            oss << ",\"A_shapiro\":";
            emit_matrix_json(oss, A_shapiro_[q]);
        }
        if (groupform_available_[q]) {
            oss << ",\"A_groupform\":";
            emit_matrix_json(oss, A_groupform_[q]);
        }
        oss << "}";
    }
    oss << "],";
    oss << "\"certificate\":{";
    oss << "\"routes_bit_identical\":" << (cert_.routes_bit_identical ? "true" : "false") << ",";
    oss << "\"relation_checks\":" << cert_.relation_checks << ",";
    oss << "\"cocycle_checks\":" << cert_.cocycle_checks << ",";
    oss << "\"relation_closure_checks\":" << cert_.relation_closure_checks << ",";
    oss << "\"chain_map_checks\":" << cert_.chain_map_checks << ",";
    oss << "\"shapiro_available\":" << (cert_.shapiro_available ? "true" : "false") << ",";
    oss << "\"shapiro_agrees\":" << (cert_.shapiro_agrees ? "true" : "false") << ",";
    oss << "\"groupform_agrees\":" << (cert_.groupform_agrees ? "true" : "false");
    oss << "}}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// (slot-9 WP-4; E8 item 4b) The module route over Z/ell^k -- the header note.
// ---------------------------------------------------------------------------
template<class Elt>
void HeckeBoundaryOperatorT<Elt>::require_field_op(const char* what) const {
    if (ring_k_ >= 2)
        throw std::logic_error(std::string("HeckeBoundaryOperator::") + what + ": a field (F_ell) accessor, but this operator is over " +
                               CoefficientRingText(ring_k_) + " -- use Theta_module() / H_matrix_module() ([E] sec:ring (f))");
}

template<class Elt>
void HeckeBoundaryOperatorT<Elt>::compute_module() {
    using engine::DenseBuf;
    const engine::ModCtx<Elt>& ctx = ring_ctx_;
    const Elt p = ctx.p;
    const SimplicialComplex* Kbd = coh_->get_lssc()->get_base_K_complex();
    const uint32_t d = d_V_;
    std::map<std::array<Elt, 4>, DenseBuf<Elt>> rho_memo;
    auto rho = [&](const Mat22& g) -> const DenseBuf<Elt>& {
        std::array<Elt, 4> key;
        if constexpr (std::is_same_v<Elt, uint32_t>) { const auto a = g.reduced_mod(p); key = {a[0], a[1], a[2], a[3]}; }
        else { const auto a = g.reduced_mod64(p); key = {a[0], a[1], a[2], a[3]}; }
        auto it = rho_memo.find(key);
        if (it == rho_memo.end()) {
            const FlatMatrixT<Elt> R = RhoOfExactRing<Elt>(ctx, g, n_sym_);
            DenseBuf<Elt> b(d, d);
            for (uint32_t i = 0; i < d; ++i)
                for (uint32_t j = 0; j < d; ++j) b.at(i, j) = R.get_entry(i, j);
            it = rho_memo.emplace(key, std::move(b)).first;
        }
        return it->second;
    };
    // n_t mod ell^k, unsigned throughout (U64_Wide: ell^k may exceed 2^63).
    auto coeff_ring = [&](int64_t c) -> Elt {
        if (c >= 0) return static_cast<Elt>(static_cast<uint64_t>(c) % static_cast<uint64_t>(p));
        const Elt m = static_cast<Elt>(static_cast<uint64_t>(-c) % static_cast<uint64_t>(p));
        return m == 0 ? Elt(0) : static_cast<Elt>(p - m);
    };
    auto mul = [&](const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
        DenseBuf<Elt> C(A.rows, B.cols);
        if (A.rows > 0 && B.cols > 0 && A.cols > 0) engine::kernels::gemm<Elt>(ctx, engine::cview(A), engine::cview(B), engine::view(C));
        return C;
    };
    auto flat_coboundary_ring = [&](uint32_t q) {
        const FlatMatrixT<Elt> D1 = Kbd->template coboundary_operator_elt<Elt>(q, p);
        DenseBuf<Elt> D(D1.get_rows() * d, D1.get_cols() * d);
        for (uint32_t i = 0; i < D1.get_rows(); ++i)
            for (uint32_t j = 0; j < D1.get_cols(); ++j) {
                const Elt v = D1.get_entry(i, j);
                if (v != 0) for (uint32_t a = 0; a < d; ++a) D.at(i * d + a, j * d + a) = v;
            }
        return D;
    };
    const Mat22& g = cosets_->g();
    std::vector<DenseBuf<Elt>> theta(num_degrees_), A(num_degrees_);
    CertificateBundle cert;
    bool routes_equal = true;
    const auto& rels = coh_->get_lssc()->get_user_defined_simplex_relations();
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t Nq = N_q(q), dim = Nq * d;
        DenseBuf<Elt> T(dim, dim), T2(dim, dim);
        auto add_block = [&](DenseBuf<Elt>& dst, uint32_t s, uint32_t t, Elt c, const DenseBuf<Elt>& B) {
            for (uint32_t a = 0; a < d; ++a)
                for (uint32_t b = 0; b < d; ++b) {
                    const Elt v = B.at(a, b);
                    if (v == 0) continue;
                    Elt& e = dst.at(s * d + a, t * d + b);
                    e = ctx.add(e, ctx.mul(c, v));
                }
        };
        for (uint32_t s = 0; s < Nq; ++s) {
            const uint32_t sigmaK = bd_to_K_[q][s];
            for (uint32_t j = 1; j <= m_; ++j) {
                for (const ChainTerm& ct : taub_->tau(j, q, sigmaK)) {
                    const Elt c = coeff_ring(ct.coeff);
                    if (c == 0) continue;
                    const uint32_t t = local_index(q, ct.sigma_idx);
                    const DenseBuf<Elt> F = mul(rho(cosets_->s_inv(j)), rho(g * ct.gamma));   // fused
                    add_block(T, s, t, c, F);
                    add_block(T2, s, t, c, rho(cosets_->delta(j) * ct.gamma));              // direct, exact product
                }
            }
        }
        for (uint32_t a = 0; a < dim && routes_equal; ++a)
            for (uint32_t b = 0; b < dim; ++b)
                if (T.at(a, b) != T2.at(a, b)) { routes_equal = false; break; }
        if (!routes_equal)
            throw std::runtime_error("HeckeBoundaryOperator (Z/ell^k): the fused and the direct bracketing of [C, eq:bdrychain] "
                                     "differ at q=" + std::to_string(q));
        // (i) relations and (ii) cocycles on Theta b_i; level (b).
        const DenseBuf<Elt>& Hb = coh_->H_module_basis(q);
        const DenseBuf<Elt> Y = mul(T, Hb);
        for (const auto& rel : rels) {
            if (rel.q_dim != q) continue;
            for (uint32_t col = 0; col < Y.cols; ++col) {
                for (uint32_t a = 0; a < d; ++a) {
                    Elt acc = Y.at(rel.s_lhs_idx_in_Kq * d + a, col);
                    for (uint32_t b = 0; b < d; ++b) {
                        const Elt mv = rel.M.get_entry(a, b);
                        if (mv != 0) acc = ctx.sub(acc, ctx.mul(mv, Y.at(rel.s_rhs_idx_in_Kq * d + b, col)));
                    }
                    if (acc != 0)
                        throw std::runtime_error("HeckeBoundaryOperator (Z/ell^k): runtime assertion (i) FAILED -- Theta^bd z "
                                                 "violates relation '" + rel.tag + "'");
                }
                ++cert.relation_checks;
            }
        }
        if (q + 1 < num_degrees_) {
            const DenseBuf<Elt> dY = mul(flat_coboundary_ring(q), Y);
            for (uint32_t i = 0; i < dY.rows; ++i)
                for (uint32_t col = 0; col < dY.cols; ++col)
                    if (dY.at(i, col) != 0)
                        throw std::runtime_error("HeckeBoundaryOperator (Z/ell^k): runtime assertion (ii) FAILED -- d(Theta^bd z) != 0");
        }
        cert.cocycle_checks += Y.cols;
        // (iii) relation closure: Theta C^q_eq inside C^q_eq, by solve, residual recomputed.
        const DenseBuf<Elt>& C = coh_->module_degree(q).C;
        if (C.cols > 0) {
            const DenseBuf<Elt> W = mul(T, C);
            std::vector<std::vector<Elt>> co;
            try {
                co = engine::elim::solve_multi_rhs_module<Elt>(ctx, engine::cview(C), engine::cview(W));
            } catch (const std::runtime_error& e) {
                throw std::runtime_error(std::string("HeckeBoundaryOperator (Z/ell^k): runtime assertion (iii) FAILED -- ") + e.what());
            }
            DenseBuf<Elt> TE(C.cols, C.cols);
            for (uint32_t col = 0; col < C.cols; ++col)
                for (uint32_t j = 0; j < C.cols; ++j) TE.at(j, col) = co[col][j];
            const DenseBuf<Elt> R = mul(C, TE);
            for (uint32_t a = 0; a < dim; ++a)
                for (uint32_t col = 0; col < C.cols; ++col)
                    if (R.at(a, col) != W.at(a, col))
                        throw std::runtime_error("HeckeBoundaryOperator (Z/ell^k): the closure residual is not identically zero");
            cert.relation_closure_checks += C.cols;
        }
        DenseBuf<Elt> M(Hb.cols, Hb.cols);
        if (Hb.cols > 0) {
            const auto co = coh_->H_class_coordinates(q, engine::cview(Y));
            for (uint32_t col = 0; col < Hb.cols; ++col)
                for (uint32_t i = 0; i < Hb.cols; ++i) M.at(i, col) = co[col][i];
        }
        theta[q] = std::move(T);
        A[q] = std::move(M);
    }
    // (iv) d Theta = Theta d on the C^q_eq basis ([C, prop:bdryop](1)).
    for (uint32_t q = 0; q + 1 < num_degrees_; ++q) {
        const DenseBuf<Elt> D = flat_coboundary_ring(q);
        const DenseBuf<Elt>& C = coh_->module_degree(q).C;
        const DenseBuf<Elt> lhs = mul(D, mul(theta[q], C));
        const DenseBuf<Elt> rhs = mul(theta[q + 1], mul(D, C));
        for (uint32_t i = 0; i < lhs.rows; ++i)
            for (uint32_t j = 0; j < lhs.cols; ++j)
                if (lhs.at(i, j) != rhs.at(i, j))
                    throw std::runtime_error("HeckeBoundaryOperator (Z/ell^k): runtime assertion (iv) FAILED -- d Theta^bd != Theta^bd d");
        cert.chain_map_checks += C.cols;
    }
    cert.routes_bit_identical = routes_equal;
    Theta_module_ = std::move(theta);
    A_module_ = std::move(A);
    cert_ = cert;
}

template<class Elt>
const engine::DenseBuf<Elt>& HeckeBoundaryOperatorT<Elt>::Theta_module(uint32_t q) const {
    if (ring_k_ < 2) throw std::logic_error("HeckeBoundaryOperator::Theta_module: a Z/ell^k accessor, but this operator is over F_ell");
    require_computed();
    check_degree(q);
    return Theta_module_[q];
}

template<class Elt>
const engine::DenseBuf<Elt>& HeckeBoundaryOperatorT<Elt>::H_matrix_module(uint32_t q) const {
    if (ring_k_ < 2) throw std::logic_error("HeckeBoundaryOperator::H_matrix_module: a Z/ell^k accessor, but this operator is over F_ell");
    require_computed();
    check_degree(q);
    return A_module_[q];
}

template class HeckeBoundaryOperatorT<uint32_t>;
template class HeckeBoundaryOperatorT<uint64_t>;

} // namespace hecke
