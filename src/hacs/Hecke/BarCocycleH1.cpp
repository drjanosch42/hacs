// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S8 -- see BarCocycleH1.h for the contract references and the
// binding presentation choice.

#include "BarCocycleH1.h"

#include "SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h" // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/ModCtx.h"
#include "HeckeSettings.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace hecke {

namespace {

using engine::DenseBuf;
using engine::ModCtx;

// (matrix-engine Phase 3 Stage B) The scalar arithmetic of this file goes
// through one engine::ModCtx per modulus instead of the free mod_* functions:
// the context derives the Barrett constants once, where mod_mul re-derives the
// reduction on every call. Every ModCtx primitive returns, by contract
// (ENGINE_API.md Sec. 1) and by ModContextValidation, exactly the value the
// legacy primitive returns -- the canonical residue in [0, p) is unique, so
// this cannot move a bit.
ModCtx<uint32_t> ctx_of(uint32_t p) { return ModCtx<uint32_t>(p); }

// ---------------------------------------------------------------------------
// Small exact-F_ell polynomial helpers (coefficients LOW -> HIGH degree).
//
// These stay LOCAL on purpose: Route B is the independent cross-check of the
// pipeline's charpoly/minpoly (contract C16), so it must not share FpPoly's
// implementation with the route it is checking.
// ---------------------------------------------------------------------------

using Poly = std::vector<uint32_t>;

Poly poly_add(const Poly& a, const Poly& b, uint32_t p) {
    const ModCtx<uint32_t> ctx = ctx_of(p);
    Poly r(std::max(a.size(), b.size()), 0);
    for (size_t i = 0; i < r.size(); ++i) {
        uint32_t va = (i < a.size()) ? a[i] : 0;
        uint32_t vb = (i < b.size()) ? b[i] : 0;
        r[i] = ctx.add(va, vb);
    }
    return r;
}

Poly poly_mul(const Poly& a, const Poly& b, uint32_t p) {
    if (a.empty() || b.empty()) return Poly{};
    const ModCtx<uint32_t> ctx = ctx_of(p);
    Poly r(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < b.size(); ++j) {
            r[i + j] = ctx.add(r[i + j], ctx.mul(a[i], b[j]));
        }
    }
    return r;
}

Poly poly_neg(const Poly& a, uint32_t p) {
    const ModCtx<uint32_t> ctx = ctx_of(p);
    Poly r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = ctx.neg(a[i]);
    return r;
}

// Determinant of a square matrix of polynomials by first-row cofactor
// expansion (division-free; dimensions here are the H^1 dims, <= 5).
Poly poly_matrix_det(const std::vector<std::vector<Poly>>& P, uint32_t p) {
    const size_t n = P.size();
    if (n == 0) return Poly{ 1 };
    if (n == 1) return P[0][0];
    Poly det{};
    for (size_t j = 0; j < n; ++j) {
        // minor: drop row 0, column j
        std::vector<std::vector<Poly>> minor(n - 1, std::vector<Poly>(n - 1));
        for (size_t r = 1; r < n; ++r) {
            size_t cc = 0;
            for (size_t c = 0; c < n; ++c) {
                if (c == j) continue;
                minor[r - 1][cc++] = P[r][c];
            }
        }
        Poly term = poly_mul(P[0][j], poly_matrix_det(minor, p), p);
        if (j % 2 == 1) term = poly_neg(term, p);
        det = poly_add(det, term, p);
    }
    return det;
}

// Exact division of a (monic-divisor) polynomial: a = q * b + r, deg r <
// deg b; returns true iff r == 0. b must be monic and nonempty.
bool poly_divides_monic(const Poly& b, const Poly& a, uint32_t p) {
    if (b.empty() || b.back() != 1) return false;
    const ModCtx<uint32_t> ctx = ctx_of(p);
    Poly rem = a;
    const size_t db = b.size() - 1;
    while (rem.size() > db) {
        const uint32_t lead = rem.back();
        if (lead != 0) {
            const size_t shift = rem.size() - 1 - db;
            for (size_t i = 0; i <= db; ++i) {
                rem[shift + i] = ctx.sub(rem[shift + i], ctx.mul(lead, b[i]));
            }
        }
        rem.pop_back();
    }
    for (uint32_t v : rem)
        if (v != 0) return false;
    return true;
}

// Evaluates the polynomial (low -> high) at the square matrix A (Horner).
FlatMatrix poly_eval_at_matrix(const Poly& poly, const FlatMatrix& A) {
    const uint32_t n = A.get_rows();
    const uint32_t p = A.get_p();
    FlatMatrix result = FlatMatrix::CreateZero(n, n, p);   // zero
    const FlatMatrix Id(n, p);                             // identity
    for (size_t i = poly.size(); i-- > 0;) {
        result = result * A;
        result = result + (Id * poly[i]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Exact linear solve: sum_k x_k * cols[k] = b over F_ell. Returns true iff
// consistent (free coordinates set to 0).
//
// (matrix-engine Phase 3 Stage B) ABSORBED into engine::elim::solve_multi_rhs
// (00-implementation-plan.md, Phase 3: "absorb the two hand-rolled
// RREF-with-RHS solvers ... ONLY IF the engine's solve reproduces their exact
// pivot rule and their Z-first / mod-ell-fallback policy"). The argument, in
// full, because the condition is conditional:
//
//  1. POLICY. This solver has no Z-first / mod-ell-fallback policy to
//     reproduce. That policy belongs to the OTHER hand-rolled solver, the tau
//     triangle-fill solve (HeckeTauTable.cpp:783-908; determinism hazard
//     02 Sec. 6.5(2)), which the plan explicitly does NOT migrate in any phase
//     and which is untouched. Route B is a pure F_ell computation: every value
//     that reaches this function is already a canonical residue mod ell, and
//     the only division is mod_inv of a nonzero pivot.
//  2. PIVOT RULE. The old body ran Gauss-Jordan on the augmented matrix
//     [cols | b] with `col` scanning 0..Kc-1 (never the right-hand column),
//     pivot row = the topmost row at or below the current pivot row with a
//     nonzero entry, leading 1, elimination above AND below. That is exactly
//     the D5b rule -- leftmost pivot column, topmost non-zero row, leading 1,
//     eliminate above and below -- which engine::elim::rref implements and
//     which solve_multi_rhs runs on [B | Y] (DenseElim.cpp:455-467).
//  3. THE RESTRICTION TO THE FIRST Kc COLUMNS IS NOT A DIFFERENCE. RREF is a
//     unique function of the matrix and is produced column by column from the
//     left, so the first Kc columns of RREF([B | b]) are RREF(B) and the row
//     operations applied are the same in both routines; the two agree entry
//     for entry on the whole augmented matrix. The old routine declared
//     inconsistency when a row beyond the rank had a nonzero right-hand entry;
//     in the unrestricted RREF that is exactly the statement "column Kc is a
//     pivot column", which is the condition solve_multi_rhs reports. Kc = 0 is
//     included: a nonzero b then makes column 0 a pivot.
//  4. THE PARTICULAR SOLUTION. Both take x[c] = (right-hand entry of the pivot
//     row of column c) at pivot columns and 0 at free columns. (At both call
//     sites below the columns are in fact linearly independent -- powers_vec
//     stops growing exactly when a dependency appears, and H_reps_ + B_basis_
//     is rank-checked in the constructor -- so the solution is unique and the
//     free-column convention is not even exercised.)
//  5. REPORTING. solve_multi_rhs THROWS where the old routine returned false.
//     For MinPolyOfMatrix an inconsistent system is the expected outcome (it is
//     how the minimal polynomial's degree is found), so the throw is caught
//     here and turned back into `false`; it is control flow, not an error path,
//     and it runs at most deg(minpoly) <= dim H^1 times.
//
// BarCocycleSolverValidation pins 2-5 empirically: it keeps the pre-Stage-B
// body as an oracle and compares it to this function over fixed-seed systems
// that are consistent, inconsistent, rank-deficient, wide, tall and empty.
// ---------------------------------------------------------------------------

bool solve_in_columns(const std::vector<FlatMatrix>& cols, const FlatMatrix& b,
                      uint32_t ell, std::vector<uint32_t>& x_out) {
    const uint32_t R = b.get_rows();
    const uint32_t Kc = static_cast<uint32_t>(cols.size());
    for (uint32_t j = 0; j < Kc; ++j) {
        if (cols[j].get_rows() != R || cols[j].get_cols() != 1) {
            throw std::invalid_argument("solve_in_columns: column shape mismatch.");
        }
    }
    DenseBuf<uint32_t> B(R, Kc), Y(R, 1);
    for (uint32_t j = 0; j < Kc; ++j)
        for (uint32_t r = 0; r < R; ++r) B.at(r, j) = cols[j].get_entry(r, 0);
    for (uint32_t r = 0; r < R; ++r) Y.at(r, 0) = b.get_entry(r, 0);

    const ModCtx<uint32_t> ctx = ctx_of(ell);
    try {
        const std::vector<std::vector<uint32_t>> coords =
            engine::elim::solve_multi_rhs<uint32_t>(ctx, cview(B), cview(Y));
        x_out = coords.empty() ? std::vector<uint32_t>(Kc, 0) : coords[0];
        x_out.resize(Kc, 0);
        return true;
    } catch (const std::runtime_error&) {
        return false; // "not in image of basis" == the old routine's inconsistency
    }
}

// Rank of a set of column vectors (each R x 1) over F_ell.
// (matrix-engine Phase 3 Stage B) One engine::elim::rref on the vectors laid
// out as rows, instead of a FlatMatrix round trip. Rank is a basis-independent
// invariant of the matrix, so this is value-preserving for any correct
// elimination.
uint32_t rank_of_columns(const std::vector<FlatMatrix>& cols, uint32_t R, uint32_t ell) {
    if (cols.empty() || R == 0) return 0;
    DenseBuf<uint32_t> M(static_cast<uint32_t>(cols.size()), R);
    for (size_t i = 0; i < cols.size(); ++i)
        for (uint32_t r = 0; r < R; ++r) M.at(static_cast<uint32_t>(i), r) = cols[i].get_entry(r, 0);
    const ModCtx<uint32_t> ctx = ctx_of(ell);
    return engine::elim::rref<uint32_t>(ctx, view(M)).rank;
}

// vec(M) as an (r*c) x 1 column (row-major).
FlatMatrix vec_of_matrix(const FlatMatrix& M) {
    std::vector<uint32_t> data;
    data.reserve(static_cast<size_t>(M.get_rows()) * M.get_cols());
    for (uint32_t r = 0; r < M.get_rows(); ++r)
        for (uint32_t c = 0; c < M.get_cols(); ++c) data.push_back(M.get_entry(r, c));
    const uint32_t total = M.get_rows() * M.get_cols();
    return total == 0 ? FlatMatrix::CreateZero(0, 1, M.get_p())
                      : FlatMatrix(total, 1, M.get_p(), data);
}

// The shared word-reduction context (Stage S2). Function-local statics keep
// the plan's hecke_matrix(cosets) signature without threading a model
// argument through the validation-only bar route.
const SL2ZTessellationTables& word_model() {
    static const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    static const SL2ZTessellationTables model(K);
    return model;
}

} // namespace

// ---------------------------------------------------------------------------
// charpoly / minpoly helpers
// ---------------------------------------------------------------------------

std::vector<uint32_t> CharPolyOfMatrix(const FlatMatrix& A) {
    if (A.get_rows() != A.get_cols()) {
        throw std::invalid_argument("CharPolyOfMatrix: matrix not square.");
    }
    const uint32_t n = A.get_rows();
    const uint32_t p = A.get_p();
    if (n == 0) return Poly{ 1 };

    // det(X I - A) on polynomial entries.
    const ModCtx<uint32_t> ctx = ctx_of(p);
    std::vector<std::vector<Poly>> P(n, std::vector<Poly>(n));
    for (uint32_t r = 0; r < n; ++r) {
        for (uint32_t c = 0; c < n; ++c) {
            const uint32_t neg_a = ctx.neg(A.get_entry(r, c) % p);
            if (r == c) {
                P[r][c] = Poly{ neg_a, 1 }; // X - a_rr
            } else {
                P[r][c] = Poly{ neg_a };
            }
        }
    }
    Poly cp = poly_matrix_det(P, p);
    cp.resize(n + 1, 0);

    // Self-checks (division-free route sanity; throw on any failure).
    if (cp[n] != 1) {
        throw std::logic_error("CharPolyOfMatrix: result is not monic.");
    }
    uint32_t trace = 0;
    for (uint32_t i = 0; i < n; ++i)
        trace = ctx.add(trace, A.get_entry(i, i) % p);
    if (cp[n - 1] != ctx.neg(trace)) {
        throw std::logic_error("CharPolyOfMatrix: X^{n-1} coefficient != -trace.");
    }
    if (!poly_eval_at_matrix(cp, A).is_zero()) {
        throw std::logic_error("CharPolyOfMatrix: Cayley-Hamilton self-check failed.");
    }
    return cp;
}

std::vector<uint32_t> MinPolyOfMatrix(const FlatMatrix& A) {
    if (A.get_rows() != A.get_cols()) {
        throw std::invalid_argument("MinPolyOfMatrix: matrix not square.");
    }
    const uint32_t n = A.get_rows();
    const uint32_t p = A.get_p();
    if (n == 0) return Poly{ 1 };

    std::vector<FlatMatrix> powers_vec; // vec(A^0), vec(A^1), ...
    FlatMatrix power(n, p);             // identity
    powers_vec.push_back(vec_of_matrix(power));

    for (uint32_t k = 1; k <= n; ++k) {
        power = power * A; // A^k
        const FlatMatrix target = vec_of_matrix(power);
        std::vector<uint32_t> coeffs;
        if (solve_in_columns(powers_vec, target, p, coeffs)) {
            // A^k = sum_j coeffs[j] A^j  =>  minpoly = X^k - sum coeffs[j] X^j.
            Poly mp(k + 1, 0);
            const ModCtx<uint32_t> ctx = ctx_of(p);
            for (uint32_t j = 0; j < k; ++j) mp[j] = ctx.neg(coeffs[j]);
            mp[k] = 1;
            // Self-checks.
            if (!poly_eval_at_matrix(mp, A).is_zero()) {
                throw std::logic_error("MinPolyOfMatrix: minpoly(A) != 0.");
            }
            if (!poly_divides_monic(mp, CharPolyOfMatrix(A), p)) {
                throw std::logic_error("MinPolyOfMatrix: minpoly does not divide charpoly.");
            }
            return mp;
        }
        powers_vec.push_back(target);
    }
    // Unreachable by Cayley-Hamilton.
    throw std::logic_error("MinPolyOfMatrix: no dependency up to degree n (impossible).");
}

// ---------------------------------------------------------------------------
// BarCocycleH1
// ---------------------------------------------------------------------------

FlatMatrix BarCocycleH1::rho_of_exact(const Mat22& gamma) const {
    const std::array<uint32_t, 4> g_mod = gamma.reduced_mod(ell_);
    const FlatMatrix g_fm(2, 2, ell_,
                          std::vector<uint32_t>{ g_mod[0], g_mod[1], g_mod[2], g_mod[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym_);
}

BarCocycleH1::BarCocycleH1(uint32_t ell, uint32_t n_sym)
    : ell_(ell), n_sym_(n_sym), d_(n_sym + 1) {
    // (slot-8 WP-3) [E] sec:ring (b): the bar-cocycle H^1 route is F_ell-only.
    RequireFieldCoefficientRing("BarCocycleH1");
    if (ell_ < 5) {
        throw std::invalid_argument(
            "BarCocycleH1: ell >= 5 required (amalgam torsion orders 4, 6 must "
            "be invertible; hypothesis (H3)).");
    }

    const Mat22 S = Mat22::S();
    const Mat22 U = Mat22::ST();
    const Mat22 T = Mat22::T();
    rho_S_ = rho_of_exact(S);
    rho_U_ = rho_of_exact(U);
    rho_T_ = rho_of_exact(T);
    rho_S_inv_ = rho_of_exact(S.inv_unimodular());
    rho_T_inv_ = rho_of_exact(T.inv_unimodular());
    rho_neg_I_ = rho_of_exact(Mat22::I().neg());

    // Presentation self-checks (loud; these pin the -I conventions,
    // rho_n(-I) = (-1)^n, and the generator dictionary).
    const FlatMatrix Id(d_, ell_);
    const FlatMatrix S2 = rho_S_ * rho_S_;
    const FlatMatrix U2 = rho_U_ * rho_U_;
    const FlatMatrix U3 = U2 * rho_U_;
    if (rho_S_ * rho_S_inv_ != Id || rho_T_ * rho_T_inv_ != Id ||
        S2 * S2 != Id || U3 * U3 != Id || S2 != rho_neg_I_ || U3 != rho_neg_I_ ||
        rho_T_ != rho_S_inv_ * rho_U_) {
        throw std::logic_error(
            "BarCocycleH1: presentation self-check failed (S^4 = (ST)^6 = 1, "
            "S^2 = (ST)^3 = -I, T = S^{-1} ST).");
    }

    // Norm elements N_S = sum_{j<4} rho(S)^j, N_U = sum_{j<6} rho(U)^j.
    FlatMatrix N_S = Id + rho_S_ + S2 + (S2 * rho_S_);
    FlatMatrix N_U = Id + rho_U_ + U2 + U3 + (U3 * rho_U_) + (U3 * U2);
    const FlatMatrix match_S = Id + rho_S_;             // phi(S^2) side
    const FlatMatrix match_U = Id + rho_U_ + U2;        // phi(U^3) side

    // Constraint matrix (3d x 2d): [N_S | 0; 0 | N_U; match_S | -match_U].
    const ModCtx<uint32_t> ctx = ctx_of(ell_);
    std::vector<uint32_t> cdata(static_cast<size_t>(3 * d_) * (2 * d_), 0);
    const uint32_t cols2 = 2 * d_;
    for (uint32_t r = 0; r < d_; ++r) {
        for (uint32_t c = 0; c < d_; ++c) {
            cdata[static_cast<size_t>(r) * cols2 + c] = N_S.get_entry(r, c);
            cdata[static_cast<size_t>(d_ + r) * cols2 + d_ + c] = N_U.get_entry(r, c);
            cdata[static_cast<size_t>(2 * d_ + r) * cols2 + c] = match_S.get_entry(r, c);
            cdata[static_cast<size_t>(2 * d_ + r) * cols2 + d_ + c] =
                ctx.neg(match_U.get_entry(r, c));
        }
    }
    constraint_ = FlatMatrix(3 * d_, cols2, ell_, cdata);

    // Cocycles Z = ker(constraint), coboundaries B = im(D),
    // D = [rho(S) - 1; rho(U) - 1] : V -> V^2.
    {
        FlatMatrix constraint_copy = constraint_; // kernel() is non-const
        Z_basis_ = constraint_copy.kernel();
    }
    std::vector<uint32_t> ddata(static_cast<size_t>(2 * d_) * d_, 0);
    const FlatMatrix dS = rho_S_ - Id;
    const FlatMatrix dU = rho_U_ - Id;
    for (uint32_t r = 0; r < d_; ++r) {
        for (uint32_t c = 0; c < d_; ++c) {
            ddata[static_cast<size_t>(r) * d_ + c] = dS.get_entry(r, c);
            ddata[static_cast<size_t>(d_ + r) * d_ + c] = dU.get_entry(r, c);
        }
    }
    FlatMatrix D(2 * d_, d_, ell_, ddata);
    B_basis_ = D.image();

    // B is contained in Z (every coboundary is a cocycle) -- assert it.
    for (const FlatMatrix& b : B_basis_) {
        if (!(constraint_ * b).is_zero()) {
            throw std::logic_error(
                "BarCocycleH1: a coboundary generator violates the cocycle "
                "conditions (B not contained in Z).");
        }
    }
    if (B_basis_.size() > Z_basis_.size()) {
        throw std::logic_error("BarCocycleH1: dim B > dim Z.");
    }
    dim_H1_ = static_cast<uint32_t>(Z_basis_.size() - B_basis_.size());

    // H^1 representatives: greedily extend the (independent) B basis to a
    // basis of Z by Z-basis vectors.
    std::vector<FlatMatrix> accumulated = B_basis_;
    uint32_t cur_rank = rank_of_columns(accumulated, 2 * d_, ell_);
    if (cur_rank != B_basis_.size()) {
        throw std::logic_error("BarCocycleH1: B basis is not independent.");
    }
    for (const FlatMatrix& z : Z_basis_) {
        if (H_reps_.size() == dim_H1_) break;
        accumulated.push_back(z);
        const uint32_t new_rank = rank_of_columns(accumulated, 2 * d_, ell_);
        if (new_rank > cur_rank) {
            H_reps_.push_back(z);
            cur_rank = new_rank;
        } else {
            accumulated.pop_back();
        }
    }
    if (H_reps_.size() != dim_H1_) {
        throw std::logic_error("BarCocycleH1: failed to extend B to a basis of Z.");
    }

    // Mayer-Vietoris dimension cross-check (binding, plan stage S8):
    // H^1(Gamma;V) = V^{S^2} / (V^S + V^U); for even n this is the plan's
    // V / (V^S + V^{ST}).
    {
        FlatMatrix kS = rho_S_ - Id;
        FlatMatrix kU = rho_U_ - Id;
        FlatMatrix kC = rho_neg_I_ - Id;
        std::vector<FlatMatrix> VS = kS.kernel();
        std::vector<FlatMatrix> VU = kU.kernel();
        std::vector<FlatMatrix> VC = kC.kernel();
        std::vector<FlatMatrix> sum_gens = VS;
        sum_gens.insert(sum_gens.end(), VU.begin(), VU.end());
        for (const FlatMatrix& v : sum_gens) {
            if (!(kC * v).is_zero()) {
                throw std::logic_error(
                    "BarCocycleH1: V^S + V^U not contained in V^{S^2}.");
            }
        }
        const uint32_t dim_sum = rank_of_columns(sum_gens, d_, ell_);
        const uint32_t dim_VC = static_cast<uint32_t>(VC.size());
        if (dim_VC < dim_sum || dim_H1_ != dim_VC - dim_sum) {
            std::ostringstream oss;
            oss << "BarCocycleH1: Mayer-Vietoris dimension cross-check failed: "
                << "dim H^1 = " << dim_H1_ << " but dim V^{S^2} - dim(V^S + V^U) = "
                << dim_VC << " - " << dim_sum << " (ell=" << ell_ << ", n=" << n_sym_
                << ").";
            throw std::logic_error(oss.str());
        }
    }
}

FlatMatrix BarCocycleH1::eval_phi(const FlatMatrix& x, const FlatMatrix& y,
                                  const Mat22& kappa) const {
    const SL2ZTessellationTables& model = word_model();
    const std::vector<std::pair<char, int64_t>> word = model.word_in_T_S(kappa);
    if (!(SL2ZTessellationTables::evaluate_word(word) == kappa)) {
        throw std::logic_error(
            "BarCocycleH1::eval_phi: word_in_T_S did not reconstruct kappa.");
    }

    const FlatMatrix Id(d_, ell_);
    const FlatMatrix phi_T = rho_S_inv_ * (y - x);                 // T = S^{-1} U
    const FlatMatrix phi_T_inv = (rho_T_inv_ * phi_T) * (ell_ - 1); // -rho(T^{-1}) phi(T)
    const FlatMatrix phi_neg_I = x + (rho_S_ * x);                 // phi(S^2)

    FlatMatrix acc(d_, 1, ell_); // phi(word so far)
    FlatMatrix rho_acc = Id;     // rho(word so far)
    auto step = [&](const FlatMatrix& phi_letter, const FlatMatrix& rho_letter) {
        acc = acc + (rho_acc * phi_letter); // phi(w a) = phi(w) + rho(w) phi(a)
        rho_acc = rho_acc * rho_letter;
    };

    for (const auto& lw : word) {
        if (lw.first == '-') {
            step(phi_neg_I, rho_neg_I_);
        } else if (lw.first == 'S') {
            for (int64_t k = 0; k < lw.second; ++k) step(x, rho_S_);
        } else if (lw.first == 'T') {
            if (lw.second >= 0) {
                for (int64_t k = 0; k < lw.second; ++k) step(phi_T, rho_T_);
            } else {
                for (int64_t k = 0; k < -lw.second; ++k) step(phi_T_inv, rho_T_inv_);
            }
        } else {
            throw std::logic_error("BarCocycleH1::eval_phi: unknown word letter.");
        }
    }

    // The accumulated representation matrix must be rho_n(kappa mod ell) --
    // a loud consistency check of the whole word expansion.
    if (rho_acc != rho_of_exact(kappa)) {
        throw std::logic_error(
            "BarCocycleH1::eval_phi: accumulated rho(word) != rho_n(kappa).");
    }
    return acc;
}

std::vector<uint32_t> BarCocycleH1::project_to_H(const FlatMatrix& z) const {
    std::vector<FlatMatrix> cols = H_reps_;
    cols.insert(cols.end(), B_basis_.begin(), B_basis_.end());
    std::vector<uint32_t> coeffs;
    if (!solve_in_columns(cols, z, ell_, coeffs)) {
        throw std::runtime_error(
            "BarCocycleH1::project_to_H: vector not in Z = span(H_reps, B).");
    }
    return std::vector<uint32_t>(coeffs.begin(), coeffs.begin() + dim_H1_);
}

FlatMatrix BarCocycleH1::hecke_matrix(const HeckeCosetTables& cosets) const {
    // Consistency: the cosets tables must have been built with this (ell, n).
    const CosetRhoT<uint32_t> probe_block = cosets.rho_t_block(1);   // (slot-14 WP-10) streamed tables too
    const FlatMatrix& probe = probe_block.get();
    if (probe.get_p() != ell_ || probe.get_rows() != d_ || probe.get_cols() != d_) {
        throw std::invalid_argument(
            "BarCocycleH1::hecke_matrix: HeckeCosetTables (ell, n_sym) mismatch.");
    }
    if (dim_H1_ == 0) return FlatMatrix::CreateZero(0, 0, ell_);

    const uint32_t m = cosets.num_cosets();
    const Mat22& g = cosets.g();
    const int64_t N = g.det(); // det g = N = p^r > 0
    const Mat22 adj_g{ g.d, checked_neg<int64_t>(g.b), checked_neg<int64_t>(g.c), g.a };

    // rho_n(delta_k) (det N; NEVER inverted, invariant I7).
    std::vector<FlatMatrix> rho_delta(m + 1, FlatMatrix());
    for (uint32_t k = 1; k <= m; ++k) rho_delta[k] = rho_of_exact(cosets.delta(k));

    // Left-coset permutation gamma * j and kappa_j(gamma) =
    // delta_{gamma*j}^{-1} gamma delta_j in Gamma, for gamma in {S, U}.
    // Exact integers only: N * kappa = adj(g) * s_k * gamma * s_j^{-1} * g,
    // divisibility by N certifies delta_k^{-1} gamma delta_j in Gamma.
    const Mat22 gens[2] = { Mat22::S(), Mat22::ST() };
    std::vector<std::vector<uint32_t>> perm(2, std::vector<uint32_t>(m + 1, 0));
    std::vector<std::vector<Mat22>> kappa(2, std::vector<Mat22>(m + 1, Mat22::I()));
    for (uint32_t gi = 0; gi < 2; ++gi) {
        const Mat22& gamma = gens[gi];
        std::vector<bool> used(m + 1, false);
        for (uint32_t j = 1; j <= m; ++j) {
            uint32_t found = 0;
            for (uint32_t k = 1; k <= m; ++k) {
                const Mat22 B = ((adj_g * cosets.s(k)) * (gamma * cosets.s_inv(j))) * g;
                if (B.a % N != 0 || B.b % N != 0 || B.c % N != 0 || B.d % N != 0) continue;
                const Mat22 kp{ B.a / N, B.b / N, B.c / N, B.d / N };
                if (kp.det() != 1) {
                    throw std::logic_error(
                        "BarCocycleH1::hecke_matrix: divisible candidate kappa has "
                        "det != 1.");
                }
                // Exact reconstruction check: delta_k * kappa == gamma * delta_j.
                if (!((cosets.delta(k) * kp) == (gamma * cosets.delta(j)))) {
                    throw std::logic_error(
                        "BarCocycleH1::hecke_matrix: kappa reconstruction identity "
                        "failed.");
                }
                ++found;
                perm[gi][j] = k;
                kappa[gi][j] = kp;
            }
            if (found != 1) {
                std::ostringstream oss;
                oss << "BarCocycleH1::hecke_matrix: gamma*j not unique (found "
                    << found << " candidates for j=" << j << ").";
                throw std::logic_error(oss.str());
            }
            if (used[perm[gi][j]]) {
                throw std::logic_error(
                    "BarCocycleH1::hecke_matrix: j -> gamma*j is not injective.");
            }
            used[perm[gi][j]] = true;
        }
    }

    // Columns: (T_g phi)(gamma) = sum_j rho_n(delta_{gamma*j}) phi(kappa_j(gamma)),
    // evaluated at gamma = S and gamma = U; the resulting pair is asserted to
    // be a cocycle, then projected to the stored H^1 basis.
    // (matrix-engine Phase 3) The columns land in one row-major buffer.
    std::vector<uint32_t> T_buf(static_cast<size_t>(dim_H1_) * dim_H1_, 0);
    for (uint32_t col = 0; col < dim_H1_; ++col) {
        const FlatMatrix& rep = H_reps_[col];
        FlatMatrix x(d_, 1, ell_), y(d_, 1, ell_);
        for (uint32_t r = 0; r < d_; ++r) {
            x.set_entry(r, 0, rep.get_entry(r, 0));
            y.set_entry(r, 0, rep.get_entry(d_ + r, 0));
        }

        FlatMatrix w[2] = { FlatMatrix(d_, 1, ell_), FlatMatrix(d_, 1, ell_) };
        for (uint32_t gi = 0; gi < 2; ++gi) {
            for (uint32_t j = 1; j <= m; ++j) {
                const FlatMatrix phi_val = eval_phi(x, y, kappa[gi][j]);
                w[gi] = w[gi] + (rho_delta[perm[gi][j]] * phi_val);
            }
        }

        FlatMatrix z(2 * d_, 1, ell_);
        for (uint32_t r = 0; r < d_; ++r) {
            z.set_entry(r, 0, w[0].get_entry(r, 0));
            z.set_entry(d_ + r, 0, w[1].get_entry(r, 0));
        }
        if (!(constraint_ * z).is_zero()) {
            throw std::logic_error(
                "BarCocycleH1::hecke_matrix: (T_g phi)(S), (T_g phi)(ST) do not "
                "satisfy the cocycle conditions ([C, prop:barformula] violated).");
        }
        const std::vector<uint32_t> coords = project_to_H(z);
        for (uint32_t row = 0; row < dim_H1_; ++row) {
            T_buf[static_cast<size_t>(row) * dim_H1_ + col] = coords[row];
        }
    }
    return FlatMatrix(dim_H1_, dim_H1_, ell_, T_buf);
}

} // namespace hecke
