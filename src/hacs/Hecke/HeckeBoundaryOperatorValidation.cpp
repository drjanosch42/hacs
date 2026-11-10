// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions WP-4b -- validation of the boundary Hecke operator. See
// HeckeBoundaryOperatorValidation.h for the independence statement and the
// arm list. [C] = docs/latex/gl2_contract.tex, cited by label.

#include "HeckeBoundaryOperatorValidation.h"

#include "HeckeBoundaryOperator.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "HeckeOperatorOnK.h"
#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "FpPoly.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hecke::HeckeBoundaryOperatorT;
using hecke::HeckeBoundaryTauTableT;
using hecke::HeckeCosetTablesT;
using hecke::HeckeOperatorOnKT;
using hecke::HeckeTauTableT;
using hecke::Mat22;
using hecke::SL2ZTessellationTables;

struct CheckLog {
    uint32_t failures = 0;
    uint64_t checks = 0;
    void fail(const std::string& msg) {
        ++failures;
        std::cout << "HeckeBoundaryOperatorValidation: FAILED -- " << msg << std::endl;
    }
    void check(bool ok, const std::string& msg) {
        ++checks;
        if (!ok) fail(msg);
    }
};

// =========================================================================
// Section 0 -- our own arithmetic.
// =========================================================================

template<class Elt>
Elt own_powmod(Elt base, uint64_t e, Elt ell) {
    const engine::ModCtx<Elt> ctx(ell);
    Elt r = 1 % ell;
    Elt b = base % ell;
    while (e > 0) {
        if (e & 1u) r = ctx.mul(r, b);
        b = ctx.mul(b, b);
        e >>= 1;
    }
    return r;
}

// chi_b(p) = p^b + p^{n+1-b} mod ell ([C, prop:bdrychar]).
template<class Elt>
Elt own_chi(Elt ell, uint64_t p, uint32_t n, uint32_t b) {
    const engine::ModCtx<Elt> ctx(ell);
    const Elt pm = static_cast<Elt>(p % static_cast<uint64_t>(ell));
    return ctx.add(own_powmod<Elt>(pm, b, ell), own_powmod<Elt>(pm, n + 1 - b, ell));
}

template<class Elt>
FlatMatrixT<Elt> own_rho(const Mat22& g, uint32_t n, Elt ell) {
    std::array<Elt, 4> m;
    if constexpr (std::is_same_v<Elt, uint32_t>) m = g.reduced_mod(ell); else m = g.reduced_mod64(ell);
    return ComputeSymmetricPowerMatrixFM(FlatMatrixT<Elt>(2, 2, ell, std::vector<Elt>{ m[0], m[1], m[2], m[3] }), n);
}

// Homogeneous polynomials in F_ell[x, y]: degree D, coefficient a_k of x^{D-k} y^k.
template<class Elt>
std::vector<Elt> own_hpoly_mul(const std::vector<Elt>& a, const std::vector<Elt>& b, Elt ell) {
    const engine::ModCtx<Elt> ctx(ell);
    std::vector<Elt> c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            c[i + j] = ctx.add(c[i + j], ctx.mul(a[i], b[j]));
    return c;
}

// The basis x^{n - b ell} Nm^b of V^T, b = 0..floor(n / ell), Nm = y^ell - x^{ell-1} y,
// as d_V x 1 columns in the fiber basis u_k = x^{n-k} y^k -- by polynomial arithmetic.
template<class Elt>
std::vector<FlatMatrixT<Elt>> own_invariant_basis(uint32_t n, Elt ell) {
    const engine::ModCtx<Elt> ctx(ell);
    const uint64_t l = static_cast<uint64_t>(ell);
    const uint64_t bmax = static_cast<uint64_t>(n) / l;
    std::vector<FlatMatrixT<Elt>> out;
    std::vector<Elt> Nm; // only needed when bmax >= 1, i.e. ell <= n
    if (bmax >= 1) {
        Nm.assign(static_cast<size_t>(l) + 1, 0);
        Nm[static_cast<size_t>(l)] = 1;
        Nm[1] = ctx.neg(1 % ell);
    }
    for (uint64_t b = 0; b <= bmax; ++b) {
        std::vector<Elt> poly(static_cast<size_t>(n - b * l) + 1, 0); // x^{n - b ell}
        poly[0] = 1 % ell;
        for (uint64_t i = 0; i < b; ++i) poly = own_hpoly_mul<Elt>(poly, Nm, ell);
        if (poly.size() != static_cast<size_t>(n) + 1) throw std::runtime_error("own_invariant_basis: degree.");
        out.emplace_back(n + 1, 1, ell, poly);
    }
    return out;
}

template<class Elt>
bool flat_equal(const FlatMatrixT<Elt>& X, const FlatMatrixT<Elt>& Y) {
    if (X.get_rows() != Y.get_rows() || X.get_cols() != Y.get_cols()) return false;
    for (uint32_t r = 0; r < X.get_rows(); ++r)
        for (uint32_t c = 0; c < X.get_cols(); ++c)
            if (X.get_entry(r, c) != Y.get_entry(r, c)) return false;
    return true;
}

template<class Elt>
FlatMatrixT<Elt> flat_of(const MatrixT<Elt>& A) { return FlatMatrixT<Elt>(A); }

template<class Elt>
FlatMatrixT<Elt> hcat(const FlatMatrixT<Elt>& A, const FlatMatrixT<Elt>& B) {
    const uint32_t R = A.get_rows(), CA = A.get_cols(), CB = B.get_cols();
    std::vector<Elt> data(static_cast<size_t>(R) * (CA + CB), 0);
    for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = 0; j < CA; ++j) data[static_cast<size_t>(i) * (CA + CB) + j] = A.get_entry(i, j);
        for (uint32_t j = 0; j < CB; ++j) data[static_cast<size_t>(i) * (CA + CB) + CA + j] = B.get_entry(i, j);
    }
    return FlatMatrixT<Elt>(R, CA + CB, A.get_p(), data);
}

template<class Elt>
FlatMatrixT<Elt> flatten_rows(const FlatMatrixT<Elt>& cochain, Elt ell) {
    const uint32_t total = cochain.get_rows() * cochain.get_cols();
    std::vector<Elt> d(total);
    for (uint32_t r = 0; r < cochain.get_rows(); ++r)
        for (uint32_t c = 0; c < cochain.get_cols(); ++c) d[static_cast<size_t>(r) * cochain.get_cols() + c] = cochain.get_entry(r, c);
    return FlatMatrixT<Elt>(total, 1, ell, d);
}

template<class Elt>
FlatMatrixT<Elt> unit_column(uint32_t dim, uint32_t b, Elt ell) {
    std::vector<Elt> d(dim, 0);
    d[b] = 1 % ell;
    return FlatMatrixT<Elt>(dim, 1, ell, d);
}

template<class Elt>
FlatMatrixT<Elt> column_of(const FlatMatrixT<Elt>& M, uint32_t b, Elt ell) {
    std::vector<Elt> d(M.get_rows());
    for (uint32_t r = 0; r < M.get_rows(); ++r) d[r] = M.get_entry(r, b);
    return FlatMatrixT<Elt>(M.get_rows(), 1, ell, d);
}

template<class Elt>
std::string multiset_str(const std::vector<Elt>& v) {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < v.size(); ++i) oss << (i ? "," : "") << v[i];
    oss << "}";
    return oss.str();
}

uint32_t simplex_slot(const SimplicialComplex& K, uint32_t q, std::vector<uint32_t> verts) {
    std::sort(verts.begin(), verts.end());
    const auto& list = K.get_simplices().at(q);
    for (uint32_t s = 0; s < list.size(); ++s)
        if (list[s] == verts) return s;
    throw std::runtime_error("HeckeBoundaryOperatorValidation: simplex not found in K_bd.");
}

struct Ctx {
    uint64_t p;
    uint32_t nu;
    uint64_t ell;
    uint32_t n;
    const char* width;
    std::string str() const {
        std::ostringstream oss;
        oss << "p=" << p << " nu=" << nu << " ell=" << ell << " n=" << n << " " << width;
        return oss.str();
    }
};

// =========================================================================
// Section 1 -- the boundary-only arms (A)-(D) on one cell, either width.
// =========================================================================

struct Tally {
    uint32_t cells = 0;
    uint32_t chi_cells = 0, chi_vectors = 0, eis_cells = 0;
    uint32_t h1_multiset_cells = 0, h1_multiset_match = 0;
    uint32_t beta_cells = 0, beta_differs_h0 = 0, beta_descends_h0 = 0, beta_descends_h1 = 0, beta_scalar_h1 = 0;
    uint32_t beta_breaks_equiv_h1 = 0, beta_breaks_equiv_h0 = 0;
    uint32_t equiv_asserted = 0, equiv_recorded_hold = 0, equiv_recorded_fail = 0;
    uint32_t ell23_cells = 0;
    uint32_t rank_r1_nonzero = 0;
};

// Our own route B, transported through the stored projector with OUR class
// representatives / lifts (K_bd = {0},{1},{2},{3}; {0,1},{1,2},{2,3}).
template<class Elt>
MatrixT<Elt> own_route_B(const EquivariantCohomologyOnKT<Elt>& coh, const SimplicialComplex& Kbd, uint32_t q,
                         const FlatMatrixT<Elt>& Mop, uint32_t n, Elt ell) {
    const engine::ModCtx<Elt> ctx(ell);
    const uint32_t d = n + 1;
    const uint32_t d_q = coh.get_H_eq_dimensions().at(q);
    MatrixT<Elt> A(d_q, d_q, ell);
    if (d_q == 0) return A;
    const auto& H = coh.get_H_eq_bases_fm().at(q);
    const FlatMatrixT<Elt>& P = coh.get_H_eq_projectors_from_C_eq_flat_fm().at(q);
    const FlatMatrixT<Elt> rhoT = own_rho<Elt>(Mat22::T(), n, ell);
    const uint32_t v0 = simplex_slot(Kbd, 0, { 0 }), v1 = simplex_slot(Kbd, 0, { 1 }), v2 = simplex_slot(Kbd, 0, { 2 }),
                   v3 = simplex_slot(Kbd, 0, { 3 });
    const uint32_t e01 = simplex_slot(Kbd, 1, { 0, 1 }), e12 = simplex_slot(Kbd, 1, { 1, 2 }), e23 = simplex_slot(Kbd, 1, { 2, 3 });
    for (uint32_t i = 0; i < d_q; ++i) {
        const FlatMatrixT<Elt> z = flatten_rows<Elt>(H[i], ell);
        std::vector<Elt> m(d, 0);
        if (q == 0) {
            for (uint32_t k = 0; k < d; ++k) m[k] = z.get_entry(v0 * d + k, 0);
        } else {
            for (uint32_t k = 0; k < d; ++k)
                m[k] = ctx.add(ctx.add(z.get_entry(e01 * d + k, 0), z.get_entry(e12 * d + k, 0)), z.get_entry(e23 * d + k, 0));
        }
        const FlatMatrixT<Elt> Mm = Mop * FlatMatrixT<Elt>(d, 1, ell, m);
        const uint32_t Nq = Kbd.get_num_q_simplices(q);
        std::vector<Elt> y(static_cast<size_t>(Nq) * d, 0);
        if (q == 0) {
            const FlatMatrixT<Elt> TMm = rhoT * Mm;
            for (uint32_t k = 0; k < d; ++k) {
                y[v0 * d + k] = Mm.get_entry(k, 0);
                y[v1 * d + k] = Mm.get_entry(k, 0);
                y[v2 * d + k] = Mm.get_entry(k, 0);
                y[v3 * d + k] = TMm.get_entry(k, 0);
            }
        } else {
            for (uint32_t k = 0; k < d; ++k) y[e01 * d + k] = Mm.get_entry(k, 0);
        }
        const FlatMatrixT<Elt> coords = P * FlatMatrixT<Elt>(Nq * d, 1, ell, y);
        for (uint32_t j = 0; j < d_q; ++j) A.data[j][i] = coords.get_entry(j, 0);
    }
    return A;
}

template<class Elt>
struct BoundaryCell {
    std::unique_ptr<HeckeCosetTablesT<Elt>> cosets;
    std::unique_ptr<HeckeBoundaryTauTableT<Elt>> taub;
    SimplicialComplex Kbd;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh;
    std::unique_ptr<HeckeBoundaryOperatorT<Elt>> op;
};

template<class Elt>
std::unique_ptr<BoundaryCell<Elt>> build_boundary_cell(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                                                       const Ctx& c) {
    auto cell = std::make_unique<BoundaryCell<Elt>>();
    cell->cosets = std::make_unique<HeckeCosetTablesT<Elt>>(c.p, c.nu, static_cast<Elt>(c.ell), c.n);
    cell->taub = std::make_unique<HeckeBoundaryTauTableT<Elt>>(K, model, *cell->cosets);
    cell->Kbd = CreateSL2ZBoundaryPathTriangulation();
    cell->lssc = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(
        cell->Kbd, GenerateSL2ZBoundaryRelationsElt<Elt>(cell->Kbd, c.n, static_cast<Elt>(c.ell)), c.n + 1,
        static_cast<Elt>(c.ell));
    cell->coh = std::make_unique<EquivariantCohomologyOnKT<Elt>>(cell->lssc.get());
    cell->coh->compute_cohomology();
    cell->op = std::make_unique<HeckeBoundaryOperatorT<Elt>>(cell->coh.get(), cell->cosets.get(), cell->taub.get());
    cell->op->compute();
    return cell;
}

// Arms (A)-(D). Returns the H^0 control matrix outcome for (E)'s use through `out`.
template<class Elt>
struct BoundaryArmsOut {
    std::vector<Elt> chi;         // {chi_b}
    std::vector<Elt> chi_shift;   // {chi_{b+1}}
    bool oracle_regime = false;   // n even, p != ell, nu == 1
    bool beta_descends_h0 = false;
    MatrixT<Elt> A_beta0;         // dim H^0 square (if oracle_regime)
    bool beta_descends_h1 = false;
    MatrixT<Elt> A_beta1;         // dim H^1 square (if oracle_regime && n < ell)
};

template<class Elt>
BoundaryArmsOut<Elt> run_boundary_arms(const BoundaryCell<Elt>& cell, const Ctx& c, CheckLog& log, Tally& tally) {
    BoundaryArmsOut<Elt> out;
    const Elt ell = static_cast<Elt>(c.ell);
    const uint32_t n = c.n, d = n + 1;
    const std::string where = "[" + c.str() + "]";
    const HeckeBoundaryOperatorT<Elt>& op = *cell.op;
    const EquivariantCohomologyOnKT<Elt>& coh = *cell.coh;
    const engine::ModCtx<Elt> ctx(ell);
    ++tally.cells;

    // ---- (A) model recording and dimensions -------------------------------
    const FlatMatrixT<Elt> rhoT = own_rho<Elt>(Mat22::T(), n, ell);
    const uint32_t dimVT = d - (rhoT - FlatMatrixT<Elt>(d, ell)).rank(); // dim ker(rho(T) - I) = dim coker
    const uint64_t floor_n_ell = static_cast<uint64_t>(n) / c.ell;
    const auto& dims = coh.get_H_eq_dimensions();
    log.check(std::string(op.boundary_model()) == "circle", where + " boundary_model != circle");
    log.check(op.boundary_is_group_cohomology() == (c.ell != 2), where + " boundary_is_group_cohomology != (2 in R^x)");
    log.check(op.q_max() == 1 && std::string(op.q_max_source()) == "derived" && !op.truncated(),
              where + " q_max / truncated record fields");
    log.check(op.reported_degrees() == std::vector<uint32_t>{ 0, 1 } && op.num_degrees() == 2, where + " reported degrees");
    log.check(dims.size() >= 2 && dims[0] == dimVT && dims[1] == dimVT, where + " dim H^q != dim V^T (prop:circlemodel(2))");
    log.check(op.dim_H(0) == dims[0] && op.dim_H(1) == dims[1], where + " dim_H accessor");
    log.check(dimVT == floor_n_ell + 1, where + " dim V^T != floor(n/ell) + 1 (prop:bdrychar(1))");
    for (uint32_t q = 2; q < dims.size(); ++q) log.check(dims[q] == 0, where + " H^q != 0 above the boundary dimension");
    if (c.ell == 2 || c.ell == 3) {
        ++tally.ell23_cells;
        if (c.ell == 2) {
            // lem:ell2unbounded: H^q(bd; V) has dim 2 dim V^T for every q >= 1 -- the circle model does NOT.
            log.check(dims[1] != 2 * dimVT && op.num_degrees() == 2,
                      where + " at ell = 2 the circle model must not report lem:ell2unbounded's dimensions");
        }
        if (c.n <= 4 && c.p == 2 && c.nu == 1) {
            std::cout << "  ell=" << c.ell << " n=" << n << " p=" << c.p << ": boundary_model=circle"
                      << " boundary_is_group_cohomology=" << (op.boundary_is_group_cohomology() ? "true" : "false")
                      << " dim H^0=" << dims[0] << " dim H^1=" << dims[1] << " (dim V^T=" << dimVT
                      << "; lem:ell2unbounded would give " << dimVT << ", " << 2 * dimVT << ", " << 2 * dimVT
                      << ", ... at ell=2)" << std::endl;
        }
    }

    // ---- (B) route equality ------------------------------------------------
    const auto cert = op.certificates();
    log.check(cert.routes_bit_identical && cert.groupform_agrees, where + " certificate flags");
    log.check(cert.shapiro_available == (c.nu == 1) && cert.shapiro_agrees == (c.nu == 1), where + " shapiro flags");
    log.check(cert.relation_checks > 0 || (dims[0] == 0 && dims[1] == 0), where + " relation checks ran");
    for (uint32_t q = 0; q < 2; ++q) {
        const MatrixT<Elt>& A = op.A(q);
        log.check(A.rows == dims[q] && A.cols == dims[q], where + " A shape");
        if (c.nu == 1) log.check(A.data == op.A_shapiro(q).data, where + " A_shapiro != A at q=" + std::to_string(q));
        if (op.groupform_available(q)) log.check(A.data == op.A_groupform(q).data, where + " A_groupform != A at q=" + std::to_string(q));
        // Our own route B.
        const auto& bc = cell.taub->boundary_cosets();
        FlatMatrixT<Elt> Mop = FlatMatrixT<Elt>::CreateZero(d, d, ell);
        bool have = false;
        if (q == 0) {
            for (const auto& x : bc) Mop = Mop + own_rho<Elt>(x.delta, n, ell);
            have = true;
        } else if (c.nu == 1) {
            const int64_t p = static_cast<int64_t>(c.p);
            FlatMatrixT<Elt> S = FlatMatrixT<Elt>::CreateZero(d, d, ell);
            for (int64_t a = 0; a < p; ++a) S = S + own_rho<Elt>(Mat22{ 1, a, 0, 1 }, n, ell);
            Mop = own_rho<Elt>(Mat22{ 1, 0, 0, p }, n, ell) * S + own_rho<Elt>(Mat22{ p, 0, 0, 1 }, n, ell);
            have = true;
        }
        if (have) {
            const MatrixT<Elt> AB = own_route_B<Elt>(coh, cell.Kbd, q, Mop, n, ell);
            log.check(AB.data == A.data, where + " OUR route B != A at q=" + std::to_string(q));
        }
        // apply_direct on the standard basis == the stored columns.
        const FlatMatrixT<Elt>& Th = op.Theta_flat(q);
        const uint32_t dim = op.N_q(q) * d;
        bool cols_ok = Th.get_rows() == dim && Th.get_cols() == dim;
        for (uint32_t b = 0; cols_ok && b < dim; ++b) {
            const FlatMatrixT<Elt> e = unit_column<Elt>(dim, b, ell);
            cols_ok = flat_equal<Elt>(op.apply_direct(q, e), column_of<Elt>(Th, b, ell)) && flat_equal<Elt>(op.apply_Theta(q, e), column_of<Elt>(Th, b, ell));
        }
        log.check(cols_ok, where + " apply_direct / apply_Theta on the standard basis at q=" + std::to_string(q));
    }
    {
        const std::string js = op.to_json();
        log.check(js.find("\"schema\":\"hecke_boundary_operator/v1\"") != std::string::npos &&
                      js.find("\"boundary_model\":\"circle\"") != std::string::npos &&
                      js.find(std::string("\"boundary_is_group_cohomology\":") + (c.ell != 2 ? "true" : "false")) != std::string::npos &&
                      js.find("\"routes_bit_identical\":true") != std::string::npos &&
                      js.find("\"truncated\":false") != std::string::npos,
                  where + " JSON record fields");
    }

    // ---- (C) the characters (n even, p != ell, nu = 1) ---------------------
    out.oracle_regime = (n % 2 == 0) && (c.p != c.ell) && (c.nu == 1);
    if (out.oracle_regime) {
        ++tally.chi_cells;
        const std::vector<FlatMatrixT<Elt>> W = own_invariant_basis<Elt>(n, ell);
        const std::vector<FlatMatrixT<Elt>> W_op = HeckeBoundaryOperatorT<Elt>::invariant_monomial_basis(n, ell);
        log.check(W.size() == floor_n_ell + 1 && W_op.size() == W.size(), where + " invariant basis size");
        FlatMatrixT<Elt> Wmat = FlatMatrixT<Elt>::CreateZero(d, 0, ell);
        for (uint32_t b = 0; b < W.size(); ++b) {
            log.check(flat_equal<Elt>(W[b], W_op[b]), where + " own Nm-basis != the object's at b=" + std::to_string(b));
            log.check(flat_equal<Elt>(rhoT * W[b], W[b]), where + " x^{n-b ell} Nm^b is not T-invariant, b=" + std::to_string(b));
            Wmat = hcat<Elt>(Wmat, W[b]);
        }
        log.check(Wmat.rank() == W.size() && W.size() == dims[0], where + " the monomials are not a basis of V^T = H^0");
        for (uint32_t b = 0; b < W.size(); ++b) {
            const Elt chi_b = own_chi<Elt>(ell, c.p, n, b);
            out.chi.push_back(chi_b);
            out.chi_shift.push_back(own_chi<Elt>(ell, c.p, n, b + 1));
            log.check(chi_b == HeckeBoundaryOperatorT<Elt>::chi(ell, c.p, n, b), where + " chi mismatch");
            // Cochain level: Theta lift(w_b) == chi_b lift(w_b) (H^0 = Z^0).
            const FlatMatrixT<Elt> phi = op.lift_class(0, W[b]);
            log.check(flat_equal<Elt>(op.apply_Theta(0, phi), phi * chi_b), where + " Theta^bd on the 0-cocycle of x^{n-b ell} Nm^b != chi_b, b=" + std::to_string(b));
            // Cohomology level: A [w_b] == chi_b [w_b].
            const FlatMatrixT<Elt> coords = op.class_coordinates(0, W[b]);
            log.check(!coords.is_zero(), where + " [w_b] = 0 in H^0");
            log.check(flat_equal<Elt>(flat_of<Elt>(op.A(0)) * coords, coords * chi_b), where + " A_bd^{(0)} [w_b] != chi_b [w_b], b=" + std::to_string(b));
            ++tally.chi_vectors;
        }
        // charpoly(A^{(0)}) == prod (X - chi_b), asserted; the H^1 multiset recorded.
        hecke::FpPolyT<Elt> prod = hecke::FpPolyT<Elt>::one(ell);
        for (const Elt chi_b : out.chi) prod = prod * hecke::FpPolyT<Elt>(ell, std::vector<Elt>{ ctx.neg(chi_b), 1 % ell });
        log.check(hecke::CharPolyBerkowitz<Elt>(op.A(0)) == prod, where + " charpoly(A_bd^{(0)}) != prod (X - chi_b)");
        ++tally.h1_multiset_cells;
        if (hecke::CharPolyBerkowitz<Elt>(op.A(1)) == prod) ++tally.h1_multiset_match;
        else std::cout << "  RECORDED " << where << ": charpoly(A_bd^{(1)}) != prod_b (X - chi_b) (cl:chi(2) instance)" << std::endl;
        // n < ell: [y^n] carries 1 + p^{n+1} on H^1 (prop:bdrychar(3)).
        if (n < c.ell) {
            ++tally.eis_cells;
            const FlatMatrixT<Elt> yn = unit_column<Elt>(d, n, ell);
            const FlatMatrixT<Elt> coords = op.class_coordinates(1, yn);
            const Elt eis = own_chi<Elt>(ell, c.p, n, 0);
            log.check(!coords.is_zero() && dims[1] == 1, where + " [y^n] = 0 in H^1 or dim H^1 != 1 at n < ell");
            log.check(flat_equal<Elt>(flat_of<Elt>(op.A(1)) * coords, coords * eis), where + " A_bd^{(1)} [y^n] != 1 + p^{n+1}");
        }
        // The plan's two worked examples, as arithmetic.
        if (n == 6 && c.ell == 5 && c.p == 2) log.check(out.chi == std::vector<Elt>{ 4, 1 }, where + " (6,5,2) multiset != {4,1}");
        if (n == 16 && c.ell == 7 && c.p == 3) log.check(out.chi == std::vector<Elt>{ 6, 0, 1 }, where + " (16,7,3) multiset != {6,0,1}");

        // ---- (D) the beta_j negative control ---------------------------------
        ++tally.beta_cells;
        const int64_t p = static_cast<int64_t>(c.p);
        FlatMatrixT<Elt> Mb = own_rho<Elt>(Mat22{ p, 0, 0, 1 }, n, ell);
        for (int64_t b = 0; b < p; ++b) Mb = Mb + own_rho<Elt>(Mat22{ 1, b, 0, p }, n, ell);
        log.check(flat_equal<Elt>(Mb, op.beta_control_fiber_operator(0)), where + " own beta sum != the object's");
        for (uint32_t b = 0; b < W.size(); ++b) {
            log.check(flat_equal<Elt>(Mb * W[b], W[b] * out.chi_shift[b]), where + " beta control on w_b != chi_{b+1}, b=" + std::to_string(b));
        }
        out.A_beta0 = op.matrix_on_H_from_fiber_operator(0, Mb, out.beta_descends_h0);
        if (out.beta_descends_h0) ++tally.beta_descends_h0;
        std::vector<Elt> ms = out.chi, mss = out.chi_shift;
        std::sort(ms.begin(), ms.end());
        std::sort(mss.begin(), mss.end());
        if (ms != mss) {
            ++tally.beta_differs_h0;
            log.check(!(out.A_beta0.data == op.A(0).data), where + " the beta control does NOT differ from A_bd^{(0)} although {chi_b} != {chi_{b+1}}");
        }
        if (n < c.ell) {
            const FlatMatrixT<Elt> Mb1 = op.beta_control_fiber_operator(1);
            out.A_beta1 = op.matrix_on_H_from_fiber_operator(1, Mb1, out.beta_descends_h1);
            if (out.beta_descends_h1) ++tally.beta_descends_h1;
            const Elt chi1 = own_chi<Elt>(ell, c.p, n, 1);
            if (out.A_beta1.rows == 1 && out.A_beta1.data[0][0] == chi1) ++tally.beta_scalar_h1;
        }
    }
    return out;
}

// =========================================================================
// Section 2 -- the bulk side and (E), uint32_t only.
// =========================================================================

struct BulkCell {
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeOperatorOnK> op;
};

std::unique_ptr<BulkCell> build_bulk_cell(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                                          const hecke::HeckeCosetTables& cosets, const Ctx& c) {
    auto b = std::make_unique<BulkCell>();
    const uint32_t ell = static_cast<uint32_t>(c.ell);
    b->lssc = std::make_unique<LocalSystemSimplicialComplex>(K, GenerateSL2ZRelationsElt<uint32_t>(K, c.n, ell), c.n + 1, ell);
    b->coh = std::make_unique<EquivariantCohomologyOnK>(b->lssc.get());
    b->coh->compute_cohomology();
    b->tau = std::make_unique<hecke::HeckeTauTable>(K, model, cosets, ell, c.n);
    b->op = std::make_unique<hecke::HeckeOperatorOnK>(b->coh.get(), &cosets, b->tau.get());
    b->op->compute();
    return b;
}

void run_restriction_arm(const SimplicialComplex& K, const BoundaryCell<uint32_t>& cell, const BulkCell& bulk,
                         const BoundaryArmsOut<uint32_t>& bo, const Ctx& c, CheckLog& log, Tally& tally) {
    const uint32_t ell = static_cast<uint32_t>(c.ell);
    const std::string where = "[" + c.str() + "]";
    SimplicialComplexMorphism scm(&cell.Kbd, &K);
    if (!scm.is_valid()) { log.fail(where + " K_bd is not a subcomplex of K"); return; }
    EquivariantCochainComplexMorphismOnK res(cell.coh.get(), bulk.coh.get(), &scm, FlatMatrix(c.n + 1, ell));
    res.compute_induced_maps_on_cohomology();
    const auto& maps = res.get_induced_H_eq_maps();
    const auto& dims_bd = cell.coh->get_H_eq_dimensions();
    const auto& dims_K = bulk.coh->get_H_eq_dimensions();
    const auto& A_K = bulk.op->get_H_matrices();
    const bool assert_here = (c.ell >= 5);
    for (uint32_t q = 0; q < 2; ++q) {
        if (maps.size() <= q) { log.fail(where + " restriction map missing at q=" + std::to_string(q)); continue; }
        const FlatMatrix& R = maps[q];
        log.check(R.get_rows() == dims_bd[q] && R.get_cols() == dims_K[q], where + " r^q shape at q=" + std::to_string(q));
        const FlatMatrix lhs = R * flat_of<uint32_t>(A_K[q]);
        const FlatMatrix rhs = flat_of<uint32_t>(cell.op->A(q)) * R;
        const bool holds = flat_equal<uint32_t>(lhs, rhs);
        if (assert_here) {
            log.check(holds, where + " obl:taubdry(vii) FAILED: r^q A^{(q)} != A_bd^{(q)} r^q at q=" + std::to_string(q));
            ++tally.equiv_asserted;
        } else {
            if (holds) ++tally.equiv_recorded_hold; else ++tally.equiv_recorded_fail;
            if (!holds) std::cout << "  RECORDED " << where << ": r^" << q << " A != A_bd r^" << q << " (model objects at ell in {2,3})" << std::endl;
        }
        if (q == 1 && R.rank() > 0) ++tally.rank_r1_nonzero;
        // The negative control must break (vii) where it can be seen.
        if (bo.oracle_regime && q == 0 && bo.beta_descends_h0 && dims_K[0] > 0 && R.rank() > 0) {
            const FlatMatrix rhs_b = flat_of<uint32_t>(bo.A_beta0) * R;
            if (!flat_equal<uint32_t>(lhs, rhs_b)) ++tally.beta_breaks_equiv_h0;
        }
        if (bo.oracle_regime && q == 1 && c.n < c.ell && dims_K[1] > 0 && R.rank() > 0 && bo.chi.size() == 1) {
            const uint32_t chi0 = bo.chi[0], chi1 = own_chi<uint32_t>(ell, c.p, c.n, 1);
            if (chi0 != chi1) {
                // A_bd^{(1)} = chi_0 on the 1-dim H^1; the right-coset convention would put chi_1 there.
                const FlatMatrix rhs_b = R * chi1;
                log.check(!flat_equal<uint32_t>(lhs, rhs_b), where + " the right-coset scalar chi_1 does NOT break (vii)");
                ++tally.beta_breaks_equiv_h1;
                if (bo.beta_descends_h1) {
                    const FlatMatrix rhs_b2 = flat_of<uint32_t>(bo.A_beta1) * R;
                    log.check(!flat_equal<uint32_t>(lhs, rhs_b2), where + " the naive right-coset H^1 group form does NOT break (vii)");
                }
            }
        }
    }
    // q = 2 is vacuous: the boundary has no 2-simplices, so H^2(bd) = 0 and r^2 = 0.
    log.check(dims_bd.size() < 3 || dims_bd[2] == 0, where + " H^2(bd) != 0");
    // The worked instance of [C, sec:oraclesnew](A): (n, ell, p) = (2, 5, 2) --
    // H^0 and H^1 of the boundary are one-dimensional and both carry
    // chi_0(2) = 1 + 2^3 = 9 = 4 (mod 5), "the same 4 as A^{(1)}(T_2)" of
    // [C, sec:workedinstance], forced by (vii) with rank r^1 = 1.
    if (c.n == 2 && c.ell == 5 && c.p == 2 && c.nu == 1) {
        const MatrixT<uint32_t>& A0 = cell.op->A(0);
        const MatrixT<uint32_t>& A1 = cell.op->A(1);
        const bool ok = A0.rows == 1 && A0.data[0][0] == 4 && A1.rows == 1 && A1.data[0][0] == 4 && A_K[1].rows == 1 &&
                        A_K[1].data[0][0] == 4 && maps.size() > 1 && maps[1].rank() == 1;
        log.check(ok, where + " the worked instance (2,5,2) of sec:oraclesnew(A) does not read 4 / 4 / 4 / rank 1");
        std::cout << "  worked instance (n,ell,p)=(2,5,2): A_bd^{(0)}=[" << A0.data[0][0] << "] A_bd^{(1)}=[" << A1.data[0][0]
                  << "] bulk A^{(1)}(T_2)=[" << A_K[1].data[0][0] << "] rank r^1=" << (maps.size() > 1 ? maps[1].rank() : 0)
                  << " (chi_0(2) = 1 + 2^3 = 9 = 4 mod 5; right-coset chi_1(2) = 2 + 2^2 = 6 = 1 mod 5 breaks (vii))" << std::endl;
    }
    if (bo.oracle_regime && ((c.n == 6 && c.ell == 5 && c.p == 2) || (c.n == 16 && c.ell == 7 && c.p == 3))) {
        std::cout << "  " << where << ": H^0 multiset {chi_b} = " << multiset_str<uint32_t>(bo.chi) << ", beta control {chi_{b+1}} = "
                  << multiset_str<uint32_t>(bo.chi_shift) << "; dim H^0(Gamma)=" << dims_K[0] << " dim H^1(Gamma)=" << dims_K[1]
                  << " rank r^1=" << (maps.size() > 1 ? maps[1].rank() : 0) << std::endl;
    }
}

} // namespace

uint32_t HeckeBoundaryOperatorValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    CheckLog log;
    Tally tally;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);

        const std::vector<std::pair<uint64_t, uint32_t>> prs = { { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 2, 2 }, { 3, 2 } };
        const std::vector<uint64_t> ells = { 2, 3, 5, 7, 11, 13, 101, 691, 134217757ull, 2147483647ull };
        const std::vector<uint32_t> ns = { 0, 2, 4, 6, 8, 10, 12, 14, 16 };
        for (const auto& pr : prs) {
            for (uint64_t ell : ells) {
                for (uint32_t n : ns) {
                    const Ctx c{ pr.first, pr.second, ell, n, "u32" };
                    try {
                        const auto cell = build_boundary_cell<uint32_t>(K, model, c);
                        const BoundaryArmsOut<uint32_t> bo = run_boundary_arms<uint32_t>(*cell, c, log, tally);
                        const auto bulk = build_bulk_cell(K, model, *cell->cosets, c);
                        run_restriction_arm(K, *cell, *bulk, bo, c, log, tally);
                    } catch (const std::exception& e) {
                        log.fail("[" + c.str() + "] EXCEPTION: " + e.what());
                    }
                }
            }
        }
        // Odd n ([C, rmk:bdryoddn]): the -I self-relations make the circle model
        // the ZERO complex at 2 in R^x and the full circle model at ell = 2;
        // compute() must run through with 0 x 0 matrices.
        for (uint64_t ell : { 2ull, 5ull, 691ull }) {
            for (uint32_t n : { 1u, 3u, 7u }) {
                const Ctx c{ 2, 1, ell, n, "u32 odd-n" };
                try {
                    const auto cell = build_boundary_cell<uint32_t>(K, model, c);
                    const auto& dims = cell->coh->get_H_eq_dimensions();
                    const uint32_t d = n + 1;
                    const uint32_t dimVT = d - (own_rho<uint32_t>(Mat22::T(), n, static_cast<uint32_t>(ell)) -
                                                FlatMatrixT<uint32_t>(d, static_cast<uint32_t>(ell))).rank();
                    const uint32_t expect = (ell == 2) ? dimVT : 0u;
                    log.check(dims.size() >= 2 && dims[0] == expect && dims[1] == expect,
                              "[" + c.str() + "] odd n: dim H^q != " + std::to_string(expect) + " (rmk:bdryoddn)");
                    log.check(cell->op->A(0).rows == expect && cell->op->A(1).rows == expect && cell->op->certificates().routes_bit_identical,
                              "[" + c.str() + "] odd n: A shapes / certificate");
                } catch (const std::exception& e) {
                    log.fail("[" + c.str() + "] EXCEPTION: " + e.what());
                }
            }
        }
        // uint64_t arm: routes and oracles at a 61-bit prime.
        for (const auto& pr : std::vector<std::pair<uint64_t, uint32_t>>{ { 2, 1 }, { 3, 1 } }) {
            for (uint32_t n : { 2u, 10u, 16u }) {
                const Ctx c{ pr.first, pr.second, 2305843009213693951ull, n, "u64" };
                try {
                    const auto cell = build_boundary_cell<uint64_t>(K, model, c);
                    (void)run_boundary_arms<uint64_t>(*cell, c, log, tally);
                } catch (const std::exception& e) {
                    log.fail("[" + c.str() + "] EXCEPTION: " + e.what());
                }
            }
        }
        // The control must have bitten somewhere.
        log.check(tally.beta_differs_h0 > 0, "no cell where the beta control differs on H^0");
        log.check(tally.beta_breaks_equiv_h1 > 0, "no cell where the right-coset scalar breaks (vii) on H^1");
        log.check(tally.equiv_asserted > 0 && tally.rank_r1_nonzero > 0, "restriction equivariance never asserted / r^1 always 0");
    } catch (const std::exception& e) {
        std::cout << "HeckeBoundaryOperatorValidation: EXCEPTION during setup: " << e.what() << std::endl;
        return 1000;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "BOUNDARY_OP cells=" << tally.cells << " checks=" << log.checks << " failures=" << log.failures
              << " chi_cells=" << tally.chi_cells << " chi_vectors=" << tally.chi_vectors << " eis_cells=" << tally.eis_cells
              << " h1_multiset=" << tally.h1_multiset_match << "/" << tally.h1_multiset_cells
              << " beta_cells=" << tally.beta_cells << " beta_differs_h0=" << tally.beta_differs_h0
              << " beta_descends_h0=" << tally.beta_descends_h0 << " beta_descends_h1=" << tally.beta_descends_h1
              << " beta_scalar_h1=" << tally.beta_scalar_h1 << " beta_breaks_h1=" << tally.beta_breaks_equiv_h1
              << " beta_breaks_h0=" << tally.beta_breaks_equiv_h0
              << " equiv_asserted=" << tally.equiv_asserted << " equiv_recorded_ell23=" << tally.equiv_recorded_hold << "/"
              << (tally.equiv_recorded_hold + tally.equiv_recorded_fail) << " rank_r1_nonzero=" << tally.rank_r1_nonzero
              << " ell23_cells=" << tally.ell23_cells << " (" << secs << " s)" << std::endl;
    return log.failures;
}
