// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 2, WP-7 (phase E4b) -- HeckeExt/HeckeBoundary. See
// the header for the objects, the contract labels and the placement.

#include "HeckeBoundary.h"

#include "../Hecke/HeckeOperatorFamily.h" // hecke::tau_construction_mutex()
#include "../Hecke/SubspaceOps.h"
#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Hecke/HeckeSettings.h"

#include <functional>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {

// ===========================================================================
// The pinned Bernoulli numerators.
// ===========================================================================

namespace {

// num(B_k/k) for k = 2, 4, ..., 26 -- index (k / 2 - 1). Exact Fraction
// arithmetic (B_k by the recursion sum_{j<m} C(m+1, j) B_j = 0), reduced:
//   B_12/12  = -691/32760,   B_16/16 = -3617/8160,   B_18/18 = 43867/14364,
//   B_20/20  = -174611/6600, B_22/22 = 77683/276,    B_24/24 = -236364091/65520,
//   B_26/26  = 657931/12.
// This is the whole range of [E] Claim 1.8's Kummer list (k <= 26); the
// mod-ell Bernoulli machinery (bernoulli_mod) is phase E6 and is not here.
constexpr int64_t kNumBkOverK[13] = {
    1,          // k = 2
    -1,         // k = 4
    1,          // k = 6
    -1,         // k = 8
    1,          // k = 10
    -691,       // k = 12
    1,          // k = 14
    -3617,      // k = 16
    43867,      // k = 18
    -174611,    // k = 20   = 283 * 617
    77683,      // k = 22   = 131 * 593
    -236364091, // k = 24   = 103 * 2294797
    657931,     // k = 26
};

} // namespace

int64_t pinned_bernoulli_numerator(uint32_t k, bool& known) {
    known = (k >= 2 && k <= 26 && (k % 2u) == 0u);
    if (!known) return 0;
    return kNumBkOverK[k / 2 - 1];
}

bool pinned_kummer_divides(uint32_t k, uint32_t ell, bool& known) {
    const int64_t num = pinned_bernoulli_numerator(k, known);
    if (!known) return false;
    ENGINE_CHECK_MSG(ell >= 2, "pinned_kummer_divides: ell < 2");
    const int64_t a = (num < 0) ? -num : num;
    return (a % static_cast<int64_t>(ell)) == 0;
}

// ===========================================================================
// Local helpers (MatrixT<uint32_t>, ModCtx; no raw mod_* call).
// ===========================================================================

namespace {

using hecke::HStack;
using hecke::SolveColumnsExact;

// rank / kernel with the empty shapes handled here (the substrate's RREF is
// not asked about a 0 x c or r x 0 matrix).
uint32_t rank_of(const Matrix& M) {
    if (M.rows == 0 || M.cols == 0) return 0;
    return hecke::RankOf(M);
}

Matrix kernel_as_matrix(const Matrix& M) {
    if (M.cols == 0) return Matrix(0, 0, M.p);
    if (M.rows == 0) return Matrix(M.cols, M.p); // everything: the identity
    return hecke::KernelAsMatrix(M);
}

Matrix to_matrix(const FlatMatrix& F) { return F.to_matrix(); }

Matrix scalar_matrix(uint32_t d, uint32_t lambda, uint32_t ell) {
    Matrix I(d, ell);
    for (uint32_t i = 0; i < d; ++i) I.data[i][i] = lambda % ell;
    return I;
}

Matrix vstack(const Matrix& A, const Matrix& B) {
    ENGINE_CHECK(A.cols == B.cols && A.p == B.p);
    Matrix out(A.rows + B.rows, A.cols, A.p);
    for (uint32_t i = 0; i < A.rows; ++i) out.data[i] = A.data[i];
    for (uint32_t i = 0; i < B.rows; ++i) out.data[A.rows + i] = B.data[i];
    return out;
}

Matrix image_as_matrix(Matrix M) {
    std::vector<Matrix> im = M.image();
    Matrix out(M.rows, static_cast<uint32_t>(im.size()), M.p);
    for (size_t j = 0; j < im.size(); ++j)
        for (uint32_t i = 0; i < M.rows; ++i) out.data[i][j] = im[j].data[i][0];
    return out;
}

// 1 + p^{n+1} mod ell.
uint32_t eisenstein_eigenvalue(uint32_t ell, uint64_t p, uint32_t n) {
    const engine::ModCtx<uint32_t> ctx(ell);
    const uint32_t pm = static_cast<uint32_t>(p % static_cast<uint64_t>(ell));
    return ctx.add(1u % ell, ctx.pow(pm, n + 1));
}

hecke::FpPoly charpoly_of(const Matrix& A) {
    if (A.rows == 0) return hecke::FpPoly::one(A.p);
    return hecke::CharPolyBerkowitz<uint32_t>(A);
}

// Golden hashing, byte-identical to Golden/GoldenHarness.cpp.
uint64_t hash_matrix(const Matrix& M) {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, M.rows);
    h = engine::FnvU64(h, M.cols);
    for (uint32_t r = 0; r < M.rows; ++r)
        for (uint32_t c = 0; c < M.cols; ++c) h = engine::FnvU64(h, static_cast<uint64_t>(M.data[r][c]));
    return h;
}
uint64_t hash_scalar(uint64_t v) { return engine::FnvU64(engine::kFnvOffset64, v); }
uint64_t hash_poly(const hecke::FpPoly& P) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, P.coeffs.size());
    for (uint32_t c : P.coeffs) h = engine::FnvU64(h, static_cast<uint64_t>(c));
    return h;
}
uint64_t hash_values(const std::vector<uint32_t>& v) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, v.size());
    for (uint32_t c : v) h = engine::FnvU64(h, static_cast<uint64_t>(c));
    return h;
}

std::string json_matrix(const Matrix& M) {
    std::ostringstream oss;
    oss << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        oss << (r ? ",[" : "[");
        for (uint32_t c = 0; c < M.cols; ++c) oss << (c ? "," : "") << M.data[r][c];
        oss << "]";
    }
    oss << "]";
    return oss.str();
}

std::string json_poly(const hecke::FpPoly& P) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < P.coeffs.size(); ++i) oss << (i ? "," : "") << P.coeffs[i];
    oss << "]";
    return oss.str();
}

std::string json_bool(bool b) { return b ? "true" : "false"; }

} // namespace

// ===========================================================================
// The certificate.
// ===========================================================================

uint32_t HeckeBoundaryCertificate::verify() const {
    std::vector<std::string> f;
    return verify(f);
}

uint32_t HeckeBoundaryCertificate::verify(std::vector<std::string>& failures) const {
    uint32_t bad = 0;
    auto item = [&](bool ok, const std::string& what) {
        if (!ok) { ++bad; failures.push_back(what); }
    };
    // Every item runs under a guard: misshaped data (a tampered dimension,
    // say) fails the item instead of throwing out of verify().
    auto guarded = [&](const char* name, const std::function<void()>& body) {
        try { body(); }
        catch (const std::exception& e) { item(false, std::string(name) + ": threw: " + e.what()); }
    };
    const uint32_t k = interior_basis.cols;
    const bool small_ell = (ell == 2 || ell == 3);

    // 1  r0 injective.
    guarded("item 1", [&] {
        item(r0.rows == dim_H0_bd && r0.cols == dim_H0_K && rank_of(r0) == dim_H0_K,
             "item 1: r^0 is not injective (lem:conerelative(4))");
    });

    // 2  H^1_! = ker r1.
    guarded("item 2", [&] {
        const bool shape = (r1.rows == dim_H1_bd && r1.cols == dim_H1_K && interior_basis.rows == dim_H1_K);
        item(shape, "item 2: r^1 / interior basis shapes");
        if (shape) {
            item((r1 * interior_basis).is_zero(), "item 2: r^1 * B != 0");
            item(rank_of(interior_basis) == k, "item 2: the interior basis is not of full column rank");
            item(k + rank_of(r1) == dim_H1_K, "item 2: dim H^1_! + rank r^1 != dim H^1");
        }
    });

    // 3  T_p-stability per op.
    for (const Op& op : ops) {
        const std::string tag = " (p=" + std::to_string(op.p) + ",nu=" + std::to_string(op.nu) + ")";
        guarded("item 3", [&] {
            if (!op.stable) {
                item(small_ell, "item 3: H^1_! not T_p-stable at ell >= 5" + tag);
                return;
            }
            const bool shape = (op.A1.rows == dim_H1_K && op.A1.cols == dim_H1_K && op.A_interior.rows == k &&
                                op.A_interior.cols == k && interior_basis.rows == dim_H1_K);
            item(shape, "item 3: operator shapes" + tag);
            if (shape) item(op.A1 * interior_basis == interior_basis * op.A_interior, "item 3: A B != B A_!" + tag);
        });
    }

    // 4  The joint Eisenstein eigenspace.
    uint32_t rank_r1E = 0;
    bool item4_shape = false;
    guarded("item 4", [&] {
        Matrix stacked(0, dim_H1_K, ell);
        bool have = false, ok_primes = true;
        for (uint64_t p : eisenstein_primes) {
            const Op* found = nullptr;
            for (const Op& op : ops) if (op.p == p && op.nu == 1) { found = &op; break; }
            if (!found || found->A1.rows != dim_H1_K || found->A1.cols != dim_H1_K) { ok_primes = false; break; }
            const uint32_t lambda = eisenstein_eigenvalue(ell, p, n);
            stacked = vstack(stacked, found->A1 - scalar_matrix(dim_H1_K, lambda, ell));
            have = true;
        }
        item(ok_primes, "item 4: a stacked Eisenstein prime has no nu = 1 operator stored");
        if (!ok_primes) return;
        const uint32_t jdim = (dim_H1_K == 0) ? 0 : (have ? dim_H1_K - rank_of(stacked) : dim_H1_K);
        item(jdim == joint_eigenspace_dim, "item 4: joint eigenspace dimension differs from the stored one");
        const uint32_t j = joint_eigenspace_dim;
        const bool shape = (j == 0) ? (eisenstein_space.cols == 0)
                                    : (eisenstein_space.rows == dim_H1_K && eisenstein_space.cols == j);
        item(shape, "item 4: the stored joint eigenspace is misshaped");
        if (!shape) return;
        item4_shape = true;
        if (j > 0) {
            item(rank_of(eisenstein_space) == j, "item 4: the joint eigenspace basis is not of full column rank");
            if (have) item((stacked * eisenstein_space).is_zero(), "item 4: (A_p - (1 + p^{n+1})) E != 0");
            if (r1.cols == dim_H1_K && r1.rows == dim_H1_bd) rank_r1E = rank_of(r1 * eisenstein_space);
            else item(false, "item 4: r^1 misshaped against E");
        }
        item(eisenstein_defined == (j == 1), "item 4: eisenstein_defined != (j == 1)");
        if (j == 1) {
            item(eisenstein_vector.rows == dim_H1_K && eisenstein_vector.cols == 1 && eisenstein_vector == eisenstein_space,
                 "item 4: the Eisenstein vector is not the stored line");
        } else {
            item(eisenstein_vector.rows == 0 && eisenstein_vector.cols == 0, "item 4: a vector is stored although E is not a line");
        }
        item(r1_rank_on_joint_space == rank_r1E, "item 4: r1_rank_on_joint_space differs from rank(r^1 E)");
        item(eisenstein_in_interior == (j > 0 && rank_r1E == 0), "item 4: eisenstein_in_interior != (E != 0 and r^1 E == 0)");
    });

    // 5  Harder.
    guarded("item 5", [&] {
        bool known = false;
        const bool div = pinned_kummer_divides(n + 2, ell, known);
        const bool expect_app = known && (n + 1 < ell) && joint_eigenspace_dim >= 1 && dim_H1_K > 0;
        item(harder_applicable == expect_app, "item 5: harder_applicable disagrees with its hypotheses");
        if (!harder_applicable) {
            item(harder_verdict == HarderVerdict::kNotApplicable, "item 5: a verdict is stored on an inapplicable cell");
            return;
        }
        item(kummer == div, "item 5: the Kummer flag disagrees with the pinned table");
        if (!item4_shape) { item(false, "item 5: cannot recompute the verdict (item 4 misshaped)"); return; }
        const uint32_t j = joint_eigenspace_dim;
        HarderVerdict v;
        if (j == 1 || rank_r1E == 0) v = (kummer == (rank_r1E == 0)) ? HarderVerdict::kConsistent : HarderVerdict::kRefuted;
        else if (rank_r1E == j) v = (!kummer) ? HarderVerdict::kConsistent : HarderVerdict::kRefuted;
        else v = HarderVerdict::kInconclusive;
        item(harder_verdict == v, "item 5: the stored Harder verdict differs from the recomputed one");
        item(v != HarderVerdict::kRefuted, "item 5: Harder's criterion is REFUTED on this cell");
    });

    // 6  The direct sum.
    guarded("item 6", [&] {
        if (eisenstein_defined && !eisenstein_in_interior) {
            const bool shape = (interior_basis.rows == dim_H1_K && eisenstein_vector.rows == dim_H1_K && eisenstein_vector.cols == 1);
            item(shape, "item 6: shapes for the direct sum");
            if (shape) {
                const Matrix BV = HStack({ interior_basis, eisenstein_vector }, dim_H1_K, ell);
                item(rank_of(BV) == k + 1, "item 6: H^1_! + Eis is not direct");
            }
            item(harder_split_direct == (k + 1 == dim_H1_K), "item 6: harder_split_direct disagrees with the dimensions");
        } else {
            item(!harder_split_direct, "item 6: harder_split_direct set without a complementary Eisenstein line");
        }
    });
    return bad;
}

// ===========================================================================
// The object.
// ===========================================================================

HeckeBoundary::HeckeBoundary(const EquivariantCohomologyOnK* coh_K, const SimplicialComplex& K,
                             const hecke::SL2ZTessellationTables& model, HeckeBoundaryOptions options)
    : coh_K_(coh_K), K_(&K), model_(&model), options_(std::move(options)) {
    // (slot-8 WP-3) [E] sec:ring (b): H^1_! and the joint eigenspace are field notions.
    hecke::RequireFieldCoefficientRing("HeckeBoundary", coh_K ? coh_K->coefficient_ring_k() : 1u);
    if (coh_K_ == nullptr) throw std::invalid_argument("HeckeBoundary: null bulk cohomology");
    if (!coh_K_->is_cohomology_computed()) throw std::runtime_error("HeckeBoundary: the bulk cohomology is not computed");
    ell_ = coh_K_->get_prime();
    const uint32_t d_V = coh_K_->get_representation_dim();
    if (d_V == 0) throw std::runtime_error("HeckeBoundary: representation dimension 0");
    n_ = d_V - 1;
    if (K_->get_simplices().size() != 3 || K_->get_num_q_simplices(0) != 7)
        throw std::runtime_error("HeckeBoundary: K is not the 7-vertex model");
    if (options_.operators.empty()) throw std::invalid_argument("HeckeBoundary: no operator requested");
}

HeckeBoundary::~HeckeBoundary() = default;

void HeckeBoundary::require_computed(const char* who) const {
    if (!computed_) throw std::runtime_error(std::string("HeckeBoundary::") + who + ": compute() has not run");
}

void HeckeBoundary::compute() {
    if (computed_) return;
    build_boundary_side();
    for (const auto& pn : options_.operators) build_pipeline(pn.first, pn.second);
    build_interior();
    build_eisenstein_and_harder();
    for (auto& pl : pipelines_) {
        build_split(*pl);
        if (options_.run_diagnostics) run_diagnostics(*pl);
    }
    // The certificate must hold on what was just built (a theorem at
    // 6 in R^x; at ell in {2, 3} item 3 is recorded, so verify() passes
    // unless something structural is wrong).
    std::vector<std::string> failures;
    const uint32_t bad = cert_.verify(failures);
    if (bad != 0) {
        std::string msg = "HeckeBoundary::compute(): the certificate fails on the freshly built object:";
        for (const auto& f : failures) msg += " [" + f + "]";
        throw std::runtime_error(msg);
    }
    computed_ = true;
}

void HeckeBoundary::build_boundary_side() {
    // (slot-3 WP-5) The five construction steps that used to live here are
    // hecke::SL2ZBoundaryPair's constructor, verbatim and in order.
    pair_ = std::make_unique<hecke::SL2ZBoundaryPair>(coh_K_, *K_);
    const auto& maps = pair_->restriction().get_induced_H_eq_maps();
    for (uint32_t q = 0; q < 2; ++q) {
        const uint32_t rows = dim_H_bd(q), cols = dim_H_bulk(q);
        if (q < maps.size() && maps[q].get_rows() == rows && maps[q].get_cols() == cols) {
            r_[q] = to_matrix(maps[q]);
        } else if (q < maps.size() && maps[q].get_rows() == 0 && maps[q].get_cols() == 0 && (rows == 0 || cols == 0)) {
            r_[q] = Matrix(rows, cols, ell_);
        } else {
            throw std::runtime_error("HeckeBoundary: the restriction map r^" + std::to_string(q) + " has an unexpected shape");
        }
        rank_r_[q] = rank_of(r_[q]);
    }
    cert_.ell = ell_;
    cert_.n = n_;
    cert_.dim_H0_K = dim_H_bulk(0);
    cert_.dim_H1_K = dim_H_bulk(1);
    cert_.dim_H0_bd = dim_H_bd(0);
    cert_.dim_H1_bd = dim_H_bd(1);
    cert_.r0 = r_[0];
    cert_.r1 = r_[1];
}

void HeckeBoundary::build_pipeline(uint64_t p, uint32_t nu) {
    if (has_pipeline(p, nu)) throw std::invalid_argument("HeckeBoundary: duplicate (p, nu) in the operator list");
    auto pl = std::make_unique<Pipeline>();
    pl->p = p;
    pl->nu = nu;
    {
        // Coset / tau construction is serialised process-wide (audit 02
        // Sec. 6.5 hazards 1-3; HeckeOperatorFamily.h).
        std::lock_guard<std::mutex> lock(hecke::tau_construction_mutex());
        pl->cosets = std::make_unique<hecke::HeckeCosetTables>(p, nu, ell_, n_);
        pl->tau = std::make_unique<hecke::HeckeTauTable>(*K_, *model_, *pl->cosets, ell_, n_);
        pl->taub = std::make_unique<hecke::HeckeBoundaryTauTable>(*K_, *model_, *pl->cosets);
    }
    pl->op = std::make_unique<hecke::HeckeOperatorOnK>(coh_K_, pl->cosets.get(), pl->tau.get());
    pl->op->compute();
    pl->bd_op = std::make_unique<hecke::HeckeBoundaryOperator>(&pair_->boundary_cohomology(), pl->cosets.get(), pl->taub.get());
    pl->bd_op->compute();
    const auto& A = pl->op->get_H_matrices();
    ENGINE_CHECK_MSG(A.size() == 3, "HeckeBoundary: the bulk operator does not report three degrees");
    for (uint32_t q = 0; q < 3; ++q) pl->A_bulk[q] = A[q];
    for (uint32_t q = 0; q < 2; ++q) pl->A_bd[q] = pl->bd_op->A(q);
    if (nu == 1 && p != ell_) {
        const uint32_t bmax = n_ / ell_;
        for (uint32_t b = 0; b <= bmax; ++b) pl->chi.push_back(hecke::HeckeBoundaryOperator::chi(ell_, p, n_, b));
    }
    pipelines_.push_back(std::move(pl));
}

void HeckeBoundary::build_interior() {
    const uint32_t d1 = dim_H_bulk(1);
    if (d1 == 0) {
        cert_.interior_basis = Matrix(0, 0, ell_);
    } else {
        cert_.interior_basis = kernel_as_matrix(r_[1]); // d1 x k, the RREF kernel basis (D5b)
        ENGINE_CHECK(cert_.interior_basis.rows == d1);
    }
    const uint32_t k = cert_.interior_basis.cols;
    cert_.ops.clear();
    for (auto& pl : pipelines_) {
        HeckeBoundaryCertificate::Op op;
        op.p = pl->p;
        op.nu = pl->nu;
        op.A1 = pl->A_bulk[1];
        op.A_bd1 = pl->A_bd[1];
        if (k == 0) {
            pl->A_interior = Matrix(0, 0, ell_);
            pl->interior_stable = true;
        } else {
            try {
                pl->A_interior = SolveColumnsExact(cert_.interior_basis, pl->A_bulk[1] * cert_.interior_basis,
                                                   "HeckeBoundary(interior)");
                pl->interior_stable = true;
            } catch (const std::exception& e) {
                pl->interior_stable = false;
                pl->A_interior = Matrix(0, 0, ell_);
                if (ell_ >= 5)
                    throw std::runtime_error("HeckeBoundary: H^1_! is not T_p-stable at ell >= 5 (p=" + std::to_string(pl->p) +
                                             ", nu=" + std::to_string(pl->nu) + "): " + e.what());
            }
        }
        op.A_interior = pl->A_interior;
        op.stable = pl->interior_stable;
        cert_.ops.push_back(std::move(op));
    }
}

void HeckeBoundary::build_eisenstein_and_harder() {
    const uint32_t d1 = dim_H_bulk(1);
    cert_.eisenstein_primes.clear();
    cert_.joint_eigenspace_dim = 0;
    cert_.eisenstein_space = Matrix(0, 0, ell_);
    cert_.eisenstein_defined = false;
    cert_.eisenstein_vector = Matrix(0, 0, ell_);
    cert_.r1_rank_on_joint_space = 0;
    cert_.eisenstein_in_interior = false;
    cert_.harder_applicable = false;
    cert_.kummer = false;
    cert_.harder_verdict = HeckeBoundaryCertificate::HarderVerdict::kNotApplicable;
    cert_.harder_split_direct = false;
    bernoulli_num_ = pinned_bernoulli_numerator(n_ + 2, bernoulli_known_);
    if (d1 == 0) return;

    // Route (a) of obl:iota(v), the rule of HeckeIotaSignValidation: stack
    // (A_p - (1 + p^{n+1})) over p in {2, 3, 5} \ {ell}, nu = 1, and stop once
    // two primes leave a line.
    Matrix stacked(0, d1, ell_);
    uint32_t used = 0;
    uint32_t jdim = d1;
    for (uint64_t p : { 2ull, 3ull, 5ull }) {
        if (p == ell_) continue;
        if (!has_pipeline(p, 1)) continue;
        if (used >= 2 && jdim == 1) break;
        const Pipeline& pl = pipeline(p, 1);
        const uint32_t lambda = eisenstein_eigenvalue(ell_, p, n_);
        stacked = vstack(stacked, pl.A_bulk[1] - scalar_matrix(d1, lambda, ell_));
        cert_.eisenstein_primes.push_back(p);
        ++used;
        jdim = d1 - rank_of(stacked);
    }
    if (used == 0) {
        cert_.joint_eigenspace_dim = d1;
        cert_.eisenstein_space = Matrix(d1, ell_); // no prime available: everything
    } else {
        cert_.joint_eigenspace_dim = jdim;
        cert_.eisenstein_space = kernel_as_matrix(stacked);
        ENGINE_CHECK(cert_.eisenstein_space.cols == jdim && (jdim == 0 || cert_.eisenstein_space.rows == d1));
    }
    const uint32_t j = cert_.joint_eigenspace_dim;
    if (j > 0) cert_.r1_rank_on_joint_space = rank_of(r_[1] * cert_.eisenstein_space);
    cert_.eisenstein_in_interior = (j > 0) && (cert_.r1_rank_on_joint_space == 0);
    if (j == 1) {
        cert_.eisenstein_defined = true;
        cert_.eisenstein_vector = cert_.eisenstein_space;
    }
    // Harder ([E] cl:harder): n < ell - 1, k = n + 2 in the pinned range, E != 0.
    if (bernoulli_known_ && n_ + 1 < ell_ && j >= 1) {
        bool known = false;
        cert_.kummer = pinned_kummer_divides(n_ + 2, ell_, known);
        cert_.harder_applicable = known;
        using HV = HeckeBoundaryCertificate::HarderVerdict;
        const uint32_t rk = cert_.r1_rank_on_joint_space;
        if (j == 1 || rk == 0) cert_.harder_verdict = (cert_.kummer == (rk == 0)) ? HV::kConsistent : HV::kRefuted;
        else if (rk == j) cert_.harder_verdict = (!cert_.kummer) ? HV::kConsistent : HV::kRefuted;
        else cert_.harder_verdict = HV::kInconclusive;
    }
    // The split.
    if (cert_.eisenstein_defined && !cert_.eisenstein_in_interior)
        cert_.harder_split_direct = (cert_.interior_basis.cols + 1 == d1);
}

void HeckeBoundary::build_split(Pipeline& pl) {
    pl.charpoly_A1 = charpoly_of(pl.A_bulk[1]);
    pl.charpoly_interior = pl.interior_stable ? charpoly_of(pl.A_interior) : hecke::FpPoly::zero(ell_);
    // A_bd^{(1)} restricted to im r^1 (the quotient H^1 / H^1_! transported by r^1).
    pl.quotient_stable = false;
    pl.A_quotient_bd = Matrix(0, 0, ell_);
    if (rank_r_[1] == 0) {
        pl.quotient_stable = true;
    } else {
        const Matrix Im = image_as_matrix(r_[1]); // dim H^1_bd x rank r^1
        ENGINE_CHECK(Im.cols == rank_r_[1]);
        try {
            pl.A_quotient_bd = SolveColumnsExact(Im, pl.A_bd[1] * Im, "HeckeBoundary(quotient)");
            pl.quotient_stable = true;
        } catch (const std::exception& e) {
            if (ell_ >= 5)
                throw std::runtime_error("HeckeBoundary: im r^1 is not A_bd-stable at ell >= 5: " + std::string(e.what()));
        }
    }
    pl.charpoly_quotient = pl.quotient_stable ? charpoly_of(pl.A_quotient_bd) : hecke::FpPoly::zero(ell_);
    pl.charpoly_product_holds = pl.interior_stable && pl.quotient_stable &&
                                (pl.charpoly_A1 == pl.charpoly_interior * pl.charpoly_quotient);
    // Recorded: does the quotient's characteristic polynomial divide
    // prod_b (X - chi_b(p))?  ([C, cl:chi](2), CONJ; the H^1 multiset.)
    pl.quotient_charpoly_divides_chi = false;
    if (pl.quotient_stable && !pl.chi.empty()) {
        const engine::ModCtx<uint32_t> ctx(ell_);
        hecke::FpPoly prod = hecke::FpPoly::one(ell_);
        for (uint32_t c : pl.chi) prod = prod * hecke::FpPoly(ell_, std::vector<uint32_t>{ ctx.neg(c % ell_), 1u % ell_ });
        if (pl.charpoly_quotient.degree() == 0) pl.quotient_charpoly_divides_chi = true;
        else pl.quotient_charpoly_divides_chi = hecke::FpPoly::mod(prod, pl.charpoly_quotient).is_zero();
    }
}

void HeckeBoundary::run_diagnostics(Pipeline& pl) {
    pl.cochain_commutes = hecke::cochain_commutes(*coh_K_, *pl.op, *pl.bd_op, pair_->restriction());
    pl.walk = hecke::cusp_support_walk(*pl.tau, *pl.taub);
    pl.diagnostics_run = true;
}

// --- accessors ------------------------------------------------------------

const EquivariantCohomologyOnK& HeckeBoundary::boundary_cohomology() const {
    require_computed("boundary_cohomology");
    return pair_->boundary_cohomology();
}

const EquivariantCochainComplexMorphismOnK& HeckeBoundary::restriction() const {
    require_computed("restriction");
    return pair_->restriction();
}

const SimplicialComplex& HeckeBoundary::K_bd() const {
    if (!pair_) throw std::runtime_error("HeckeBoundary::K_bd: compute() has not run");
    return pair_->K_bd();
}

const hecke::SL2ZBoundaryPair& HeckeBoundary::pair() const {
    require_computed("pair");
    return *pair_;
}

const Matrix& HeckeBoundary::r(uint32_t q) const {
    require_computed("r");
    if (q > 1) throw std::out_of_range("HeckeBoundary::r: q > 1");
    return r_[q];
}

uint32_t HeckeBoundary::rank_r(uint32_t q) const {
    require_computed("rank_r");
    if (q > 1) throw std::out_of_range("HeckeBoundary::rank_r: q > 1");
    return rank_r_[q];
}

uint32_t HeckeBoundary::dim_H_bulk(uint32_t q) const {
    const auto& d = coh_K_->get_H_eq_dimensions();
    return q < d.size() ? d[q] : 0u;
}

uint32_t HeckeBoundary::dim_H_bd(uint32_t q) const {
    if (!pair_) throw std::runtime_error("HeckeBoundary::dim_H_bd: compute() has not run");
    const auto& d = pair_->boundary_cohomology().get_H_eq_dimensions();
    return q < d.size() ? d[q] : 0u;
}

const HeckeBoundary::Pipeline& HeckeBoundary::pipeline(size_t i) const {
    require_computed("pipeline");
    if (i >= pipelines_.size()) throw std::out_of_range("HeckeBoundary::pipeline: index");
    return *pipelines_[i];
}

bool HeckeBoundary::has_pipeline(uint64_t p, uint32_t nu) const {
    for (const auto& pl : pipelines_) if (pl->p == p && pl->nu == nu) return true;
    return false;
}

const HeckeBoundary::Pipeline& HeckeBoundary::pipeline(uint64_t p, uint32_t nu) const {
    for (const auto& pl : pipelines_) if (pl->p == p && pl->nu == nu) return *pl;
    throw std::out_of_range("HeckeBoundary::pipeline: no (p=" + std::to_string(p) + ", nu=" + std::to_string(nu) + ")");
}

const Matrix& HeckeBoundary::interior_basis() const {
    require_computed("interior_basis");
    return cert_.interior_basis;
}

uint32_t HeckeBoundary::dim_interior() const {
    require_computed("dim_interior");
    return cert_.interior_basis.cols;
}

const Matrix& HeckeBoundary::interior_operator(uint64_t p, uint32_t nu) const {
    require_computed("interior_operator");
    const Pipeline& pl = pipeline(p, nu);
    if (!pl.interior_stable) throw std::runtime_error("HeckeBoundary::interior_operator: H^1_! is not stable under this operator");
    return pl.A_interior;
}

bool HeckeBoundary::interior_stable(uint64_t p, uint32_t nu) const {
    require_computed("interior_stable");
    return pipeline(p, nu).interior_stable;
}

const char* HeckeBoundary::harder_verdict_name(HeckeBoundaryCertificate::HarderVerdict v) {
    using HV = HeckeBoundaryCertificate::HarderVerdict;
    switch (v) {
        case HV::kConsistent: return "consistent";
        case HV::kRefuted: return "refuted";
        case HV::kInconclusive: return "inconclusive";
        default: return "not_applicable";
    }
}

const HeckeBoundaryCertificate& HeckeBoundary::certificate() const {
    require_computed("certificate");
    return cert_;
}

// --- the golden collectors --------------------------------------------------

std::vector<GoldenObject> HeckeBoundary::golden_objects(uint64_t p, uint32_t nu) const {
    require_computed("golden_objects");
    const Pipeline& pl = pipeline(p, nu);
    std::vector<GoldenObject> out;
    auto scalar = [&](const std::string& name, uint64_t v) { out.push_back(GoldenObject{ name, 'A', v, 0, hash_scalar(v) }); };
    auto matrix = [&](const std::string& name, char tier, const Matrix& M) {
        out.push_back(GoldenObject{ name, tier, M.rows, M.cols, hash_matrix(M) });
    };
    auto poly = [&](const std::string& name, const hecke::FpPoly& P) {
        out.push_back(GoldenObject{ name, 'A', P.coeffs.size(), 1, hash_poly(P) });
    };
    // Tier A -- the plan's list.
    scalar("dim_H0_bd", dim_H_bd(0));
    scalar("dim_H1_bd", dim_H_bd(1));
    scalar("rank_r^0", rank_r_[0]);
    scalar("rank_r^1", rank_r_[1]);
    poly("charpoly_A_bd^0", charpoly_of(pl.A_bd[0]));
    poly("charpoly_A_bd^1", charpoly_of(pl.A_bd[1]));
    out.push_back(GoldenObject{ "chi_multiset", 'A', pl.chi.size(), 1, hash_values(pl.chi) });
    // Tier A -- the E4b additions.
    scalar("dim_H1_interior", cert_.interior_basis.cols);
    scalar("eis_joint_dim", cert_.joint_eigenspace_dim);
    scalar("eis_rank_r1", cert_.r1_rank_on_joint_space);
    // Tier B -- basis-dependent.
    matrix("r^0", 'B', r_[0]);
    matrix("r^1", 'B', r_[1]);
    matrix("A_bd^0", 'B', pl.A_bd[0]);
    matrix("A_bd^1", 'B', pl.A_bd[1]);
    matrix("interior_basis", 'B', cert_.interior_basis);
    return out;
}

uint64_t HeckeBoundary::golden_digest() const {
    require_computed("golden_digest");
    uint64_t h = engine::kFnvOffset64;
    for (const auto& pl : pipelines_) {
        h = engine::FnvU64(h, pl->p);
        h = engine::FnvU64(h, pl->nu);
        for (const GoldenObject& g : golden_objects(pl->p, pl->nu)) {
            h = engine::FnvBytes(h, g.object);
            h = engine::FnvU64(h, static_cast<uint64_t>(g.tier));
            h = engine::FnvU64(h, g.rows);
            h = engine::FnvU64(h, g.cols);
            h = engine::FnvU64(h, g.hash);
        }
    }
    return h;
}

// --- text -------------------------------------------------------------------

std::string HeckeBoundary::to_string_offset(uint32_t offset) const {
    const std::string pad(offset, ' ');
    std::ostringstream oss;
    oss << pad << "HeckeBoundary (E4b): ell=" << ell_ << " n=" << n_ << (computed_ ? "" : " [not computed]") << "\n";
    if (!computed_) return oss.str();
    oss << pad << "  dim H^*(K) = (" << dim_H_bulk(0) << "," << dim_H_bulk(1) << "," << dim_H_bulk(2) << ")  dim H^*(bd) = ("
        << dim_H_bd(0) << "," << dim_H_bd(1) << ")  rank r^0=" << rank_r_[0] << " rank r^1=" << rank_r_[1] << "\n";
    oss << pad << "  H^1_! = ker r^1: dim " << cert_.interior_basis.cols << "\n";
    oss << pad << "  Eisenstein line: " << (cert_.eisenstein_defined ? "defined" : "undefined") << " (joint eigenspace dim "
        << cert_.joint_eigenspace_dim << " over";
    for (uint64_t p : cert_.eisenstein_primes) oss << " " << p;
    oss << "); rank(r^1 E)=" << cert_.r1_rank_on_joint_space << "; in H^1_!: " << (cert_.eisenstein_in_interior ? "yes" : "no") << "\n";
    oss << pad << "  Harder: applicable=" << (cert_.harder_applicable ? "yes" : "no") << " kummer=" << (cert_.kummer ? "yes" : "no")
        << " num(B_k/k)=" << (bernoulli_known_ ? std::to_string(bernoulli_num_) : std::string("unknown"))
        << " verdict=" << harder_verdict_name(cert_.harder_verdict) << " split_direct=" << (cert_.harder_split_direct ? "yes" : "no") << "\n";
    for (const auto& pl : pipelines_) {
        oss << pad << "  (p=" << pl->p << ",nu=" << pl->nu << "): interior_stable=" << (pl->interior_stable ? "yes" : "no")
            << " charpoly_product=" << (pl->charpoly_product_holds ? "yes" : "no") << " quotient|chi=" << (pl->quotient_charpoly_divides_chi ? "yes" : "no");
        if (pl->diagnostics_run)
            oss << " cochain_commutes=" << (pl->cochain_commutes.holds() ? "true" : "false") << " eta_zero=" << (pl->walk.eta_zero() ? "true" : "false");
        oss << "\n";
    }
    return oss.str();
}

std::string HeckeBoundary::to_json() const {
    require_computed("to_json");
    std::ostringstream j;
    j << "{\"schema\":\"hecke_boundary_parabolic/v1\",\"ell\":" << ell_ << ",\"n\":" << n_
      << ",\"conventions\":{\"polynomials\":\"coefficients low-to-high degree\",\"matrices\":\"array of rows, canonical residues in [0,ell)\","
         "\"column_convention\":\"T[z_i] = sum_j A_{ji} [z_j]\"}";
    // Per-complex record fields ([C, sec:recordschema] item 5; slot-2 owner confirmation 1).
    const uint32_t q_max_bulk = static_cast<uint32_t>(coh_K_->get_H_eq_dimensions().size()) - 1;
    j << ",\"bulk\":{\"model\":\"sl2z_7vertex\",\"q_max\":" << q_max_bulk << ",\"truncated\":false,\"q_max_source\":\"derived\",\"degrees\":[";
    for (uint32_t q = 0; q <= q_max_bulk; ++q) j << (q ? "," : "") << q;
    j << "],\"dim_H\":[";
    for (uint32_t q = 0; q <= q_max_bulk; ++q) j << (q ? "," : "") << dim_H_bulk(q);
    j << "]}";
    const hecke::HeckeBoundaryOperator& bd0 = *pipelines_.front()->bd_op;
    j << ",\"boundary\":{\"boundary_model\":\"" << bd0.boundary_model() << "\",\"boundary_is_group_cohomology\":"
      << json_bool(bd0.boundary_is_group_cohomology()) << ",\"q_max\":" << bd0.q_max() << ",\"truncated\":" << json_bool(bd0.truncated())
      << ",\"q_max_source\":\"" << bd0.q_max_source() << "\",\"degrees\":[";
    {
        const auto degs = bd0.reported_degrees();
        for (size_t i = 0; i < degs.size(); ++i) j << (i ? "," : "") << degs[i];
    }
    j << "],\"dim_H\":[" << dim_H_bd(0) << "," << dim_H_bd(1) << "]}";
    j << ",\"restriction\":{\"r0\":" << json_matrix(r_[0]) << ",\"r1\":" << json_matrix(r_[1]) << ",\"rank_r0\":" << rank_r_[0]
      << ",\"rank_r1\":" << rank_r_[1] << "}";
    j << ",\"interior\":{\"dim\":" << cert_.interior_basis.cols << ",\"basis\":" << json_matrix(cert_.interior_basis) << "}";
    j << ",\"eisenstein\":{\"defined\":" << json_bool(cert_.eisenstein_defined) << ",\"joint_eigenspace_dim\":" << cert_.joint_eigenspace_dim
      << ",\"primes\":[";
    for (size_t i = 0; i < cert_.eisenstein_primes.size(); ++i) j << (i ? "," : "") << cert_.eisenstein_primes[i];
    j << "],\"space\":" << json_matrix(cert_.eisenstein_space) << ",\"vector\":" << json_matrix(cert_.eisenstein_vector)
      << ",\"rank_r1_on_space\":" << cert_.r1_rank_on_joint_space << ",\"in_interior\":" << json_bool(cert_.eisenstein_in_interior) << "}";
    j << ",\"harder\":{\"applicable\":" << json_bool(cert_.harder_applicable) << ",\"k\":" << (n_ + 2)
      << ",\"bernoulli_numerator_known\":" << json_bool(bernoulli_known_) << ",\"num_Bk_over_k\":" << (bernoulli_known_ ? bernoulli_num_ : 0)
      << ",\"kummer\":" << json_bool(cert_.kummer) << ",\"verdict\":\"" << harder_verdict_name(cert_.harder_verdict) << "\""
      << ",\"split_direct\":" << json_bool(cert_.harder_split_direct) << "}";
    j << ",\"operators\":[";
    for (size_t i = 0; i < pipelines_.size(); ++i) {
        const Pipeline& pl = *pipelines_[i];
        j << (i ? "," : "") << "{\"p\":" << pl.p << ",\"nu\":" << pl.nu << ",\"A1\":" << json_matrix(pl.A_bulk[1])
          << ",\"A_bd0\":" << json_matrix(pl.A_bd[0]) << ",\"A_bd1\":" << json_matrix(pl.A_bd[1])
          << ",\"interior_stable\":" << json_bool(pl.interior_stable) << ",\"A_interior\":" << json_matrix(pl.A_interior)
          << ",\"quotient_stable\":" << json_bool(pl.quotient_stable) << ",\"A_quotient_bd\":" << json_matrix(pl.A_quotient_bd)
          << ",\"charpoly_A1\":" << json_poly(pl.charpoly_A1) << ",\"charpoly_interior\":" << json_poly(pl.charpoly_interior)
          << ",\"charpoly_quotient\":" << json_poly(pl.charpoly_quotient) << ",\"charpoly_product_holds\":" << json_bool(pl.charpoly_product_holds)
          << ",\"chi\":[";
        for (size_t b = 0; b < pl.chi.size(); ++b) j << (b ? "," : "") << pl.chi[b];
        j << "],\"quotient_charpoly_divides_chi\":" << json_bool(pl.quotient_charpoly_divides_chi);
        j << ",\"diagnostics\":{\"run\":" << json_bool(pl.diagnostics_run);
        if (pl.diagnostics_run) {
            j << ",\"cochain_commutes\":" << json_bool(pl.cochain_commutes.holds()) << ",\"cochain_vectors\":" << pl.cochain_commutes.vectors
              << ",\"cochain_failing\":" << pl.cochain_commutes.failing << ",\"eta_zero\":" << json_bool(pl.walk.eta_zero())
              << ",\"cusp_supported\":" << json_bool(pl.walk.cusp_supported()) << ",\"walk_slots\":" << pl.walk.slots
              << ",\"walk_off_boundary\":" << pl.walk.slots_off_boundary << ",\"walk_differ\":" << pl.walk.slots_differ
              << ",\"tau_solve_regime\":\"" << (pl.tau->used_mod_ell_solve() ? "mod_ell" : "Z") << "\",\"taub_solve_regime\":\""
              << pl.taub->solve_regime() << "\"";
        }
        j << "}}";
    }
    j << "]";
    j << ",\"certificate\":{\"verify\":" << cert_.verify() << "}";
    j << "}";
    return j.str();
}

} // namespace hecke_ext
