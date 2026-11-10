// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// obl:iota(ii), (iii), (v) of docs/latex/gl2_contract.tex chapter 9 (v1.2.0),
// and the definition-consistency oracle "the Eisenstein class lies in
// H^1_{eq,+}" that owner decision OQ3 (2026-09-16) made true BY DEFINITION:
// def:iota(iv) now reads iota^* := -iota^*_geo, with iota^*_geo the geometric
// pullback (iota^*_geo f)(sigma) = rho_n(iota) f(iota_K sigma). See
// HeckeIotaSignValidation.h for what is GATING (G1-G11) and what is RECORDED,
// and docs/implementation/oq3_resolution.md for the decision and its cost.
//
// Independence discipline: iota^* is built here from def:iota alone -- the
// element, the live rho_n, the vertex permutation re-derived from the shipped
// exact coordinates, the signs recomputed, the global sign of def:iota(iv) --
// and applied to the PUBLIC stored objects of EquivariantCohomologyOnK
// (C/Z/B/H_eq bases, projectors), the public A^{(1)}(T_p) of
// HeckeOperatorFamily and the public restriction morphism
// EquivariantCochainComplexMorphismOnK. No existing computational path is
// modified. Both involutions are built, iota^* is asserted to be -iota^*_geo
// entrywise, and the geometric sign is printed next to the normalised one so
// that a third party can reproduce either reading of a record.

#include "HeckeIotaSignValidation.h"

#include "ExactInt.h"
#include "HeckeOperatorFamily.h"
#include "QuadPoint.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "ValidationSupport.h"

namespace hecke {
namespace iota_oracle {

using SimplexTable = std::vector<std::vector<std::vector<uint32_t>>>;
using Relation = LocalSystemSimplicialComplex::SimplexRelation;

// =========================================================================
// Section 0 -- the element (def:iota(i)) and its symmetric power (def:iota(ii)).
// =========================================================================

Mat22 iota_mat() { return Mat22{ -1, 0, 0, 1 }; }

// c_iota(gamma) = iota gamma iota^{-1} = iota gamma iota, as the exact product.
Mat22 c_iota(const Mat22& m) {
    const Mat22 iv = iota_mat();
    return iv * m * iv;
}

FlatMatrix mat22_mod(const Mat22& g, uint32_t ell) {
    const std::array<uint32_t, 4> r = g.reduced_mod(ell);
    return FlatMatrix(2, 2, ell, std::vector<uint32_t>{ r[0], r[1], r[2], r[3] });
}

// The LIVE rho_n of the pipeline (LocalSystemSL2Z derives every relation
// matrix through exactly this call).
FlatMatrix rho_n(const Mat22& g, uint32_t n, uint32_t ell) {
    return ComputeSymmetricPowerMatrixFM(mat22_mod(g, ell), n);
}

// eq:rhoniota, the closed form: diag((-1)^{n-j})_{0<=j<=n}.
FlatMatrix rho_iota_closed_form(uint32_t n, uint32_t ell) {
    const uint32_t d = n + 1;
    std::vector<uint32_t> data(static_cast<size_t>(d) * d, 0);
    for (uint32_t j = 0; j < d; ++j) {
        const bool minus = ((n - j) % 2u) == 1u;
        data[static_cast<size_t>(j) * d + j] = minus ? (ell - 1) : 1u;
    }
    return FlatMatrix(d, d, ell, data);
}

// =========================================================================
// Section 1 -- iota_K on a stored complex (def:iota(iii)): image index and
// orientation sign of every simplex, from a vertex permutation.
// =========================================================================

// IotaOnComplex is declared in HeckeIotaSignValidation.h (slot-4 WP-3 export).

uint32_t find_simplex_index(const SimplexTable& simplices, uint32_t q, std::vector<uint32_t> verts) {
    return hecke::vsupport::find_simplex_index(simplices, q, std::move(verts),
                                               "HeckeIotaSignValidation: image simplex is not a stored simplex "
                             "-- hypothesis (I1) fails for this complex.");
}

int sort_parity(const std::vector<uint32_t>& img) {
    int inversions = 0;
    for (size_t a = 0; a + 1 < img.size(); ++a)
        for (size_t b = a + 1; b < img.size(); ++b)
            if (img[a] > img[b]) ++inversions;
    return (inversions % 2 == 0) ? 1 : -1;
}

IotaOnComplex build_iota(const SimplicialComplex& K, const std::vector<uint32_t>& vperm) {
    IotaOnComplex io;
    io.vperm = vperm;
    const SimplexTable& simplices = K.get_simplices();
    io.image_idx.assign(simplices.size(), {});
    io.image_sign.assign(simplices.size(), {});
    for (uint32_t q = 0; q < simplices.size(); ++q) {
        const auto& list = simplices[q];
        io.image_idx[q].assign(list.size(), 0);
        io.image_sign[q].assign(list.size(), 1);
        for (uint32_t s = 0; s < list.size(); ++s) {
            std::vector<uint32_t> img;
            for (uint32_t v : list[s]) img.push_back(vperm.at(v));
            io.image_sign[q][s] = sort_parity(img);
            io.image_idx[q][s] = find_simplex_index(simplices, q, img);
        }
    }
    return io;
}

// lem:iotamodel's printed sign table, by VERTEX SET.
const std::vector<std::vector<uint32_t>>& minus_sign_simplices_7vertex() {
    static const std::vector<std::vector<uint32_t>> rows = {
        { 0, 1 }, { 2, 3 }, { 1, 2 }, { 0, 1, 4 }, { 2, 3, 5 }, { 1, 2, 6 },
    };
    return rows;
}

// =========================================================================
// Section 2 -- dense helpers in the flat coordinates eq:flatindex.
// =========================================================================

// iota_C^{(q)}: block (global_sign * eps_sigma) * rho at block (sigma, |iota_K sigma|).
//   global_sign = +1: the GEOMETRIC pullback iota^*_geo, (f)(sigma) -> rho f(iota_K sigma);
//   global_sign = -1: def:iota(iv) since v1.2.0, iota^* := -iota^*_geo.
FlatMatrix iota_flat(const IotaOnComplex& io, uint32_t q, const FlatMatrix& rho, uint32_t ell, int global_sign) {
    if (global_sign != 1 && global_sign != -1) throw std::logic_error("iota_flat: global_sign must be +-1");
    const uint32_t d_V = rho.get_rows();
    const uint32_t N = static_cast<uint32_t>(io.image_idx.at(q).size());
    const uint32_t dim = N * d_V;
    std::vector<uint32_t> data(static_cast<size_t>(dim) * dim, 0);
    for (uint32_t s = 0; s < N; ++s) {
        const uint32_t t = io.image_idx[q][s];
        const bool minus = (io.image_sign[q][s] * global_sign) < 0;
        for (uint32_t a = 0; a < d_V; ++a)
            for (uint32_t b = 0; b < d_V; ++b) {
                uint32_t v = rho.get_entry(a, b);
                if (minus && v != 0) v = ell - v;
                data[static_cast<size_t>(s * d_V + a) * dim + (t * d_V + b)] = v;
            }
    }
    return FlatMatrix(dim, dim, ell, data);
}

// D (x) I_{d_V} for the simplex-major flat layout.
FlatMatrix kron_identity(const FlatMatrix& D, uint32_t d_V, uint32_t ell) {
    const uint32_t R = D.get_rows(), C = D.get_cols();
    std::vector<uint32_t> data(static_cast<size_t>(R) * d_V * C * d_V, 0);
    const uint32_t cols = C * d_V;
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) {
            const uint32_t v = D.get_entry(i, j);
            if (v == 0) continue;
            for (uint32_t k = 0; k < d_V; ++k)
                data[static_cast<size_t>(i * d_V + k) * cols + (j * d_V + k)] = v;
        }
    return FlatMatrix(R * d_V, cols, ell, data);
}

// A list of N_q x d_V cochains (row s = value on simplex s) as one dim_q x k
// multi-vector, column t = cochain t flattened to s*d_V + j.
FlatMatrix multivector(const std::vector<FlatMatrix>& basis, uint32_t dim_q, uint32_t ell) {
    const uint32_t k = static_cast<uint32_t>(basis.size());
    std::vector<uint32_t> data(static_cast<size_t>(dim_q) * k, 0);
    for (uint32_t t = 0; t < k; ++t) {
        const FlatMatrix& c = basis[t];
        const uint32_t d_V = c.get_cols();
        if (c.get_rows() * d_V != dim_q)
            throw std::runtime_error("HeckeIotaSignValidation: basis cochain has the wrong shape.");
        for (uint32_t s = 0; s < c.get_rows(); ++s)
            for (uint32_t j = 0; j < d_V; ++j)
                data[static_cast<size_t>(s * d_V + j) * k + t] = c.get_entry(s, j);
    }
    return FlatMatrix(dim_q, k, ell, data);
}

} // namespace iota_oracle
} // namespace hecke

namespace {

using namespace hecke::iota_oracle;
using hecke::Mat22;
using hecke::QuadPoint;
using hecke::SL2ZTessellationTables;

FlatMatrix hcat(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows()) throw std::runtime_error("hcat: row mismatch");
    const uint32_t R = A.get_rows(), CA = A.get_cols(), CB = B.get_cols();
    std::vector<uint32_t> data(static_cast<size_t>(R) * (CA + CB), 0);
    for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = 0; j < CA; ++j) data[static_cast<size_t>(i) * (CA + CB) + j] = A.get_entry(i, j);
        for (uint32_t j = 0; j < CB; ++j) data[static_cast<size_t>(i) * (CA + CB) + CA + j] = B.get_entry(i, j);
    }
    return FlatMatrix(R, CA + CB, A.get_p(), data);
}

FlatMatrix vcat(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_cols() != B.get_cols()) throw std::runtime_error("vcat: column mismatch");
    const uint32_t C = A.get_cols();
    std::vector<uint32_t> data;
    data.reserve(static_cast<size_t>(A.get_rows() + B.get_rows()) * C);
    for (uint32_t i = 0; i < A.get_rows(); ++i) for (uint32_t j = 0; j < C; ++j) data.push_back(A.get_entry(i, j));
    for (uint32_t i = 0; i < B.get_rows(); ++i) for (uint32_t j = 0; j < C; ++j) data.push_back(B.get_entry(i, j));
    return FlatMatrix(A.get_rows() + B.get_rows(), C, A.get_p(), data);
}

// The relation matrix of degree q built from the LSSC's own relation list --
// block row r: +I at s_lhs, -M at s_rhs (I - M when they coincide). All rows,
// vacuous ones included; nothing is short-circuited here.
FlatMatrix relation_matrix(const std::vector<Relation>& rels, uint32_t q, uint32_t N_q, uint32_t d_V, uint32_t ell) {
    std::vector<const Relation*> rows;
    for (const auto& r : rels) if (r.q_dim == q) rows.push_back(&r);
    const uint32_t cols = N_q * d_V;
    const uint32_t nrows = static_cast<uint32_t>(rows.size()) * d_V;
    std::vector<uint32_t> data(static_cast<size_t>(nrows) * cols, 0);
    for (uint32_t r = 0; r < rows.size(); ++r) {
        const Relation& rel = *rows[r];
        for (uint32_t a = 0; a < d_V; ++a) {
            data[static_cast<size_t>(r * d_V + a) * cols + (rel.s_lhs_idx_in_Kq * d_V + a)] = 1u;
        }
        for (uint32_t a = 0; a < d_V; ++a)
            for (uint32_t b = 0; b < d_V; ++b) {
                const uint32_t m = rel.M.get_entry(a, b);
                if (m == 0) continue;
                uint32_t& slot = data[static_cast<size_t>(r * d_V + a) * cols + (rel.s_rhs_idx_in_Kq * d_V + b)];
                slot = ModArithmetic::mod_sub(slot, m, ell);
            }
    }
    return FlatMatrix(nrows, cols, ell, data);
}

// dim ker(M - lambda I) for a square M.
uint32_t eigenspace_dim(const FlatMatrix& M, uint32_t lambda, uint32_t ell) {
    const uint32_t d = M.get_rows();
    if (d == 0) return 0;
    return d - (M - FlatMatrix::CreateScalar(d, lambda, ell)).rank();
}

bool is_scalar_multiple(const FlatMatrix& X, const FlatMatrix& Y, uint32_t lambda) {
    return X == (Y * lambda);
}

// =========================================================================
// Section 3 -- classical constants, for the RECORDED comparison only.
// =========================================================================

// d_k = dim M_k(SL_2(Z)), k even >= 4 (obl:iota(v)).
uint32_t dim_Mk(uint32_t k) {
    if (k % 12 == 2) return k / 12;
    return k / 12 + 1;
}

// rmk:dickson: n in the semigroup <ell+1, ell^2-ell>.
bool is_dickson_exceptional(uint32_t n, uint32_t ell) {
    if (n == 0) return false;
    const uint64_t d1 = static_cast<uint64_t>(ell) + 1u;
    const uint64_t d2 = static_cast<uint64_t>(ell) * (static_cast<uint64_t>(ell) - 1u);
    for (uint64_t b = 0; b * d2 <= n; ++b)
        if ((static_cast<uint64_t>(n) - b * d2) % d1 == 0) return true;
    return false;
}

// =========================================================================
// Section 4 -- the cochain-level checks on ONE local system (G3, G4, G5).
// Shared by the 7-vertex model and the cusp path.
// =========================================================================

struct IotaOnCohomology {
    std::vector<FlatMatrix> iota_C;     // per degree, (N_q d_V)^2: def:iota(iv), iota^* = -iota^*_geo
    std::vector<FlatMatrix> iota_C_geo; // per degree, the geometric pullback iota^*_geo
    std::vector<FlatMatrix> iota_H;     // per degree, dim H^q x dim H^q (0x0 if empty)
    std::vector<uint32_t> dim_plus, dim_minus;
};

struct CheckLog {
    uint32_t failures = 0;
    uint64_t checks = 0;
    void fail(const std::string& msg) { ++failures; std::cout << "HeckeIotaSignValidation: FAILED -- " << msg << std::endl; }
};

// G3 -- obl:iota(ii) on a relation list with exact group elements.
void check_relation_automorphism(const std::vector<SL2ZRelationDatum>& data, const IotaOnComplex& io,
                                 const FlatMatrix& rho_i, uint32_t n, uint32_t ell, const std::string& where,
                                 CheckLog& log) {
    for (const auto& d : data) {
        const uint32_t q = d.rel.q_dim;
        const uint32_t il = io.image_idx[q][d.rel.s_lhs_idx_in_Kq];
        const uint32_t ir = io.image_idx[q][d.rel.s_rhs_idx_in_Kq];
        const int el = io.image_sign[q][d.rel.s_lhs_idx_in_Kq];
        const int er = io.image_sign[q][d.rel.s_rhs_idx_in_Kq];
        ++log.checks;
        if (el != er) log.fail(where + ": relation '" + d.rel.tag + "' has unequal orientation signs on its two sides under iota_K.");
        // The image triple (iota lhs, iota rhs, c_iota(gamma)) must be the
        // INVERSE of a row (lhs', rhs', gamma'): lhs' = iota rhs, rhs' = iota lhs,
        // gamma' = c_iota(gamma)^{-1}.
        const Mat22 want_g = c_iota(d.g_r).inv_unimodular();
        bool found = false;
        for (const auto& e : data) {
            if (e.rel.q_dim != q) continue;
            if (e.rel.s_lhs_idx_in_Kq == ir && e.rel.s_rhs_idx_in_Kq == il && e.g_r == want_g) { found = true; break; }
        }
        ++log.checks;
        if (!found) log.fail(where + ": image of relation '" + d.rel.tag + "' under (iota_K, c_iota) is not the inverse of a listed row.");
        // rho_n(iota) rho_n(gamma) == rho_n(iota gamma iota) rho_n(iota) entrywise over F_ell.
        ++log.checks;
        if (!(rho_i * d.rel.M == rho_n(c_iota(d.g_r), n, ell) * rho_i))
            log.fail(where + ": rho_n(iota) rho_n(gamma) != rho_n(iota gamma iota) rho_n(iota) for '" + d.rel.tag + "'.");
        // And the stored M is the live rho_n of the stored g_r (invariant I3, re-asserted).
        ++log.checks;
        if (!(d.rel.M == rho_n(d.g_r, n, ell))) log.fail(where + ": stored relation matrix is not rho_n(g_r) for '" + d.rel.tag + "'.");
    }
}

// G4 + G5 -- iota_C on the flat cochains of a computed cohomology object.
IotaOnCohomology build_and_check_iota_on_cohomology(const EquivariantCohomologyOnK& coh, const IotaOnComplex& io,
                                                     const FlatMatrix& rho_i, uint32_t ell, const std::string& where,
                                                     CheckLog& log) {
    IotaOnCohomology out;
    const SimplicialComplex& K = *coh.get_lssc()->get_base_K_complex();
    const auto& rels = coh.get_lssc()->get_user_defined_simplex_relations();
    const uint32_t d_V = rho_i.get_rows();
    const uint32_t Q = static_cast<uint32_t>(K.get_simplices().size());
    const auto& dims = coh.get_H_eq_dimensions();
    const auto& C = coh.get_C_eq_bases_fm();
    const auto& Z = coh.get_Z_eq_bases_fm();
    const auto& B = coh.get_B_eq_bases_fm();
    const auto& H = coh.get_H_eq_bases_fm();
    const auto& P = coh.get_H_eq_projectors_from_C_eq_flat_fm();

    for (uint32_t q = 0; q < Q; ++q) {
        out.iota_C_geo.push_back(iota_flat(io, q, rho_i, ell, +1));
        out.iota_C.push_back(iota_flat(io, q, rho_i, ell, -1));
    }

    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t N_q = K.get_num_q_simplices(q);
        const uint32_t dim_q = N_q * d_V;
        const FlatMatrix& I_C = out.iota_C[q];

        // def:iota(iv), v1.2.0: iota^* == -iota^*_geo entrywise (the relation
        // a third party uses to reproduce a record in either reading).
        ++log.checks;
        if (!(I_C == out.iota_C_geo[q] * (ell - 1u))) log.fail(where + ": iota_C != -iota_C_geo at q=" + std::to_string(q));

        // (iii-a) involution as a full matrix.
        ++log.checks;
        if (!(I_C * I_C == FlatMatrix(dim_q, ell))) log.fail(where + ": iota_C^2 != I at q=" + std::to_string(q));

        // (iii-b) commutation with d as full matrices.
        if (q + 1 < Q) {
            const FlatMatrix D = kron_identity(K.coboundary_operator(q, ell), d_V, ell); // N_{q+1} d_V x N_q d_V
            ++log.checks;
            if (!(D * I_C == out.iota_C[q + 1] * D)) log.fail(where + ": d iota_C != iota_C d at q=" + std::to_string(q));
        }

        // (ii-b) equivariance ENTRYWISE on every relation: A_rel iota_C C_eq == 0.
        if (q < C.size() && !C[q].empty()) {
            const FlatMatrix Cq = multivector(C[q], dim_q, ell);
            const FlatMatrix A = relation_matrix(rels, q, N_q, d_V, ell);
            ++log.checks;
            if (!(A * Cq).is_zero()) log.fail(where + ": the stored C_eq does not satisfy the relation list at q=" + std::to_string(q));
            ++log.checks;
            if (!(A * (I_C * Cq)).is_zero()) log.fail(where + ": iota_C C_eq violates a relation at q=" + std::to_string(q));
            log.checks += static_cast<uint64_t>(A.get_rows()) * Cq.get_cols();
        }

        // (iii-c) iota_C Z in span Z, iota_C B in span B.
        if (q < Z.size() && !Z[q].empty()) {
            const FlatMatrix Zq = multivector(Z[q], dim_q, ell);
            ++log.checks;
            if (hcat(Zq, I_C * Zq).rank() != Zq.rank()) log.fail(where + ": iota_C Z_eq leaves span(Z_eq) at q=" + std::to_string(q));
        }
        if (q < B.size() && !B[q].empty()) {
            const FlatMatrix Bq = multivector(B[q], dim_q, ell);
            ++log.checks;
            if (hcat(Bq, I_C * Bq).rank() != Bq.rank()) log.fail(where + ": iota_C B_eq leaves span(B_eq) at q=" + std::to_string(q));
        }

        // iota^* on H^q = P_q iota_C H_eq.
        const uint32_t dH = (q < dims.size()) ? dims[q] : 0;
        if (dH == 0 || q >= H.size() || q >= P.size()) {
            out.iota_H.push_back(FlatMatrix::CreateZero(0, 0, ell));
            out.dim_plus.push_back(0);
            out.dim_minus.push_back(0);
            continue;
        }
        const FlatMatrix Hq = multivector(H[q], dim_q, ell);
        const FlatMatrix& Pq = P[q];
        if (Pq.get_rows() != dH || Pq.get_cols() != dim_q) throw std::runtime_error("HeckeIotaSignValidation: projector shape mismatch.");
        const FlatMatrix IH = Pq * (I_C * Hq);
        ++log.checks;
        if (!(Pq * Hq == FlatMatrix(dH, ell))) log.fail(where + ": P_q H_eq != I at q=" + std::to_string(q));
        // The geometric involution on H^q is -iota^* (recorded relation, asserted).
        ++log.checks;
        if (!(Pq * (out.iota_C_geo[q] * Hq) == IH * (ell - 1u))) log.fail(where + ": iota^*_geo != -iota^* on H^q at q=" + std::to_string(q));
        ++log.checks;
        if (!(IH * IH == FlatMatrix(dH, ell))) log.fail(where + ": (iota^*)^2 != I on H^q at q=" + std::to_string(q));
        // The projector-free cross-check: iota_C H - H iota^* lies in span(B).
        if (q < B.size() && !B[q].empty()) {
            const FlatMatrix Bq = multivector(B[q], dim_q, ell);
            const FlatMatrix diff = (I_C * Hq) - (Hq * IH);
            ++log.checks;
            if (hcat(Bq, diff).rank() != Bq.rank()) log.fail(where + ": iota_C H_eq - H_eq iota^* not in span(B_eq) at q=" + std::to_string(q));
        } else {
            ++log.checks;
            if (!((I_C * Hq) == (Hq * IH))) log.fail(where + ": iota_C H_eq != H_eq iota^* with B_eq = 0 at q=" + std::to_string(q));
        }
        const uint32_t dp = eigenspace_dim(IH, 1u, ell);
        const uint32_t dm = eigenspace_dim(IH, ell - 1u, ell);
        ++log.checks;
        if (dp + dm != dH) log.fail(where + ": dim H_+ + dim H_- != dim H at q=" + std::to_string(q));
        out.iota_H.push_back(IH);
        out.dim_plus.push_back(dp);
        out.dim_minus.push_back(dm);
    }
    return out;
}

// =========================================================================
// Section 5 -- the cusp path (route (b) substrate).
// =========================================================================

struct CuspPath {
    SimplicialComplex K;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::vector<SL2ZRelationDatum> data;
};

std::unique_ptr<CuspPath> build_cusp_path(uint32_t n, uint32_t ell) {
    auto cp = std::make_unique<CuspPath>();
    const SimplexTable s = { { { 0 }, { 1 }, { 2 }, { 3 } }, { { 0, 1 }, { 1, 2 }, { 2, 3 } } };
    cp->K = SimplicialComplex(s, "K_cusp_path_v0v1v2v3");
    // The model's cusp row (v3, v0, T): f(v3) = rho_n(T) f(v0), exactly as
    // TestSL2Z_To_S1Cusp_RestrictionMorphism spells it.
    std::vector<Relation> rels;
    rels.emplace_back(0, find_simplex_index(s, 0, { 3 }), find_simplex_index(s, 0, { 0 }), rho_n(Mat22::T(), n, ell), "cusp_T");
    cp->data.push_back(SL2ZRelationDatum{ rels.back(), Mat22::T() });
    cp->lssc = std::make_unique<LocalSystemSimplicialComplex>(cp->K, rels, n + 1, ell);
    cp->coh = std::make_unique<EquivariantCohomologyOnK>(cp->lssc.get());
    cp->coh->compute_cohomology();
    return cp;
}

// =========================================================================
// Section 6 -- one grid cell.
// =========================================================================

struct CellResult {
    uint32_t n = 0, ell = 0;
    uint32_t dimH0 = 0, dimH1 = 0, dimH2 = 0;
    uint32_t dim_plus = 0, dim_minus = 0;
    bool dickson = false;
    uint32_t pred_plus = 0, pred_minus = 0;
    // route (a)
    std::vector<uint64_t> primes_used;
    uint32_t eig_dim = 0;          // joint eigenspace dimension
    int eps_a = 0;                 // +1 / -1 / 0 undefined
    // route (b)
    uint32_t dimH1_bdry = 0, bdry_plus = 0, bdry_minus = 0;
    uint32_t rank_res = 0;
    int eps_b = 0;
    // iota^* on ker(res) (the cuspidal / interior classes): its +- split.
    uint32_t cusp_dim = 0, cusp_plus = 0, cusp_minus = 0;
    // iota^* on H^0: +1 / -1 / 2 = not scalar / 0 = H^0 empty.
    int h0_sign = 0;
    bool lift_same = false;
    std::string note;
};

void run_cell(const SimplicialComplex& K, const SL2ZTessellationTables& model, const IotaOnComplex& ioK,
              const IotaOnComplex& io_path, uint32_t n, uint32_t ell, CellResult& r, CheckLog& log) {
    r.n = n; r.ell = ell;
    r.dickson = is_dickson_exceptional(n, ell);
    r.pred_plus = dim_Mk(n + 2);
    r.pred_minus = dim_Mk(n + 2) - 1;
    const std::string where = "(n=" + std::to_string(n) + ", ell=" + std::to_string(ell) + ")";

    // --- G2: rho_n(iota), live vs closed form; the other lift.
    const FlatMatrix rho_i = rho_n(iota_mat(), n, ell);
    const FlatMatrix rho_neg_i = rho_n(iota_mat().neg(), n, ell);
    ++log.checks;
    if (!(rho_i == rho_iota_closed_form(n, ell))) log.fail(where + ": live rho_n(iota) != diag((-1)^{n-j}) (eq:rhoniota).");
    ++log.checks;
    {
        const uint32_t sgn = (n % 2 == 0) ? 1u : (ell - 1u);
        if (!(rho_neg_i == rho_i * sgn)) log.fail(where + ": rho_n(-iota) != (-1)^n rho_n(iota).");
    }
    r.lift_same = (rho_neg_i == rho_i);
    if (n % 2 == 0) { ++log.checks; if (!r.lift_same) log.fail(where + ": the two lifts differ at even n."); }

    // --- The base system and its cohomology.
    const std::vector<SL2ZRelationDatum> data = GenerateSL2ZRelationsWithGroupData(K, n, ell);
    std::vector<Relation> rels;
    for (const auto& d : data) rels.push_back(d.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    const auto& dims = coh.get_H_eq_dimensions();
    r.dimH0 = dims.size() > 0 ? dims[0] : 0;
    r.dimH1 = dims.size() > 1 ? dims[1] : 0;
    r.dimH2 = dims.size() > 2 ? dims[2] : 0;

    // --- G3 on the model's 31 rows.
    check_relation_automorphism(data, ioK, rho_i, n, ell, where + " model", log);

    // --- G4/G5 on the model.
    const IotaOnCohomology ioc = build_and_check_iota_on_cohomology(coh, ioK, rho_i, ell, where + " model", log);
    if (r.dimH1 > 0) { r.dim_plus = ioc.dim_plus[1]; r.dim_minus = ioc.dim_minus[1]; }
    if (r.dimH0 > 0) {
        // obl:iota(v), the degree-0 instance (v1.2.0): iota^* on H^0 = V^Gamma
        // is -rho_n(iota) on the invariants; for n = 0 it is -id ("constants
        // are odd", oq3_resolution.md section 2 (R1) item 2). This is what pins
        // the OVERALL sign of iota_C -- the geometric iota_C_geo = -iota_C
        // passes every symmetry check above (it is an involution commuting
        // with d and preserving the relations) but gives +1 here.
        const FlatMatrix& I0 = ioc.iota_H[0];
        if (I0 == FlatMatrix(r.dimH0, ell)) r.h0_sign = +1;
        else if (I0 == FlatMatrix::CreateScalar(r.dimH0, ell - 1u, ell)) r.h0_sign = -1;
        else r.h0_sign = 2;
        if (n == 0) {
            ++log.checks;
            if (r.dimH0 != 1 || r.h0_sign != -1) log.fail(where + ": iota^* on H^0(Gamma; F_ell) is not -1 (obl:iota(v) degree-0 anchor, def:iota(iv) v1.2.0).");
        }
    }
    if (n % 2 == 1) return; // H^* = 0 (rmk:negI); nothing else to ask.
    if (r.dimH1 == 0) { r.note = "dim H^1 = 0"; return; }
    const FlatMatrix& I1 = ioc.iota_H[1];

    // --- G6/G7: route (a), the T_p-eigenline of eigenvalue 1 + p^{n+1}.
    FlatMatrix v_a = FlatMatrix::CreateZero(0, 0, ell); // the Eisenstein eigenvector, if 1-dimensional
    {
        hecke::HeckeOperatorFamily fam(&coh, K, model, ell, n);
        std::vector<uint64_t> candidates = { 2, 3, 5 };
        FlatMatrix stacked = FlatMatrix::CreateZero(0, r.dimH1, ell);
        uint32_t used = 0;
        for (uint64_t p : candidates) {
            if (p == ell) continue;
            if (used >= 2 && r.eig_dim == 1) break;
            const FlatMatrix A(fam.T_p(p)[1]);
            ++log.checks;
            if (!(I1 * A == A * I1)) log.fail(where + ": iota^* does not commute with A^(1)(T_" + std::to_string(p) + ") (prop:iotacommutes(3c)).");
            const uint32_t lambda = ModArithmetic::mod_add(1u, ModArithmetic::mod_pow(static_cast<uint32_t>(p % ell), n + 1, ell), ell);
            stacked = vcat(stacked, A - FlatMatrix::CreateScalar(r.dimH1, lambda, ell));
            r.primes_used.push_back(p);
            ++used;
            r.eig_dim = r.dimH1 - stacked.rank();
        }
        if (r.eig_dim == 1) {
            const std::vector<FlatMatrix> ker = stacked.kernel();
            if (ker.size() != 1) throw std::runtime_error("HeckeIotaSignValidation: kernel size disagrees with rank.");
            v_a = ker[0];
            const FlatMatrix w = I1 * v_a;
            ++log.checks;
            if (w == v_a) r.eps_a = +1;
            else if (w == v_a * (ell - 1u)) r.eps_a = -1;
            else log.fail(where + ": the Eisenstein T_p-eigenline is not an iota^*-eigenline.");
            // obl:iota(v) v1.2.0, GATING: by def:iota(iv) the Eisenstein class
            // lies in H^1_{eq,+}. A '-' here would mean the geometric sign is
            // not uniform across cells, i.e. the definition is ill-posed.
            if (r.eps_a != 0) {
                ++log.checks;
                if (r.eps_a != +1) log.fail(where + ": route (a): the Eisenstein T_p-eigenline lies in H^1_{eq,-} (def:iota(iv) v1.2.0 puts it in H^1_{eq,+}).");
            }
        }
    }

    // --- G8: route (b), restriction to the cusp path.
    {
        std::unique_ptr<CuspPath> cp = build_cusp_path(n, ell);
        check_relation_automorphism(cp->data, io_path, rho_i, n, ell, where + " cusp", log);
        const IotaOnCohomology ioc_b = build_and_check_iota_on_cohomology(*cp->coh, io_path, rho_i, ell, where + " cusp", log);
        const auto& bd = cp->coh->get_H_eq_dimensions();
        r.dimH1_bdry = bd.size() > 1 ? bd[1] : 0;
        if (r.dimH1_bdry > 0) { r.bdry_plus = ioc_b.dim_plus[1]; r.bdry_minus = ioc_b.dim_minus[1]; }

        SimplicialComplexMorphism scm(&cp->K, &K);
        if (!scm.is_valid()) throw std::runtime_error("HeckeIotaSignValidation: cusp path is not a subcomplex of K.");
        EquivariantCochainComplexMorphismOnK res(cp->coh.get(), &coh, &scm, FlatMatrix(n + 1, ell));
        res.compute_induced_maps_on_cohomology();
        const FlatMatrix& R = res.get_induced_H_eq_maps().at(1); // dim H^1(bdry) x dim H^1(Gamma)
        if (R.get_rows() != r.dimH1_bdry || R.get_cols() != r.dimH1) throw std::runtime_error("HeckeIotaSignValidation: res shape mismatch.");
        r.rank_res = R.rank();
        // The +- split of ker(res), the classes invisible at the cusp.
        {
            FlatMatrix Kb = FlatMatrix(r.dimH1, ell); // ker of a 0-row map is everything
            if (R.get_rows() > 0) {
                const std::vector<FlatMatrix> kv = R.kernel();
                Kb = FlatMatrix::CreateZero(r.dimH1, 0, ell);
                for (const auto& v : kv) Kb = hcat(Kb, v);
            }
            r.cusp_dim = Kb.get_cols();
            if (r.cusp_dim > 0) {
                const FlatMatrix Id(r.dimH1, ell);
                r.cusp_plus = r.cusp_dim - ((I1 - Id) * Kb).rank();
                r.cusp_minus = r.cusp_dim - ((I1 + Id) * Kb).rank();
                ++log.checks;
                if (r.cusp_plus + r.cusp_minus != r.cusp_dim) log.fail(where + ": ker(res) is not iota^*-stable or not split.");
            }
        }
        if (r.dimH1_bdry > 0) {
            const FlatMatrix& Ib = ioc_b.iota_H[1];
            ++log.checks;
            if (!(R * I1 == Ib * R)) log.fail(where + ": res iota^* != iota^*_bdry res.");
            if (r.rank_res == 1) {
                ++log.checks;
                if (is_scalar_multiple(R * I1, R, 1u)) r.eps_b = +1;
                else if (is_scalar_multiple(R * I1, R, ell - 1u)) r.eps_b = -1;
                else log.fail(where + ": rank(res) = 1 but res iota^* is not +-res.");
                // obl:iota(v) v1.2.0, GATING (route (b)): the boundary-visible
                // class lies in H^1_{eq,+}.
                if (r.eps_b != 0) {
                    ++log.checks;
                    if (r.eps_b != +1) log.fail(where + ": route (b): the boundary-visible class lies in H^1_{eq,-} (def:iota(iv) v1.2.0 puts it in H^1_{eq,+}).");
                }
            }
            // G9: the Eisenstein T_p-eigenline is boundary-visible, and where
            // both routes are defined they name the same sign.
            if (r.eps_a != 0) {
                ++log.checks;
                if ((R * v_a).is_zero()) log.fail(where + ": the Eisenstein T_p-eigenline restricts to 0 on the cusp.");
            }
            if (r.eps_a != 0 && r.eps_b != 0) {
                ++log.checks;
                if (r.eps_a != r.eps_b) log.fail(where + ": route (a) sign != route (b) sign.");
            }
        }
    }
}

std::string sign_str(int e) { return e > 0 ? "+" : (e < 0 ? "-" : "?"); }

} // namespace

uint32_t HeckeIotaSignValidation() {
    CheckLog log;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);
        const SimplexTable& simplices = K.get_simplices();

        std::cout << "HeckeIotaSignValidation: iota = diag(-1,1), rho_n(iota) = live ComputeSymmetricPowerMatrixFM, "
                     "(iota^*_geo f)(sigma) = rho_n(iota) f(iota_K sigma) in flat coordinates s*d_V+j, and iota^* := -iota^*_geo "
                     "(gl2_contract.tex def:iota(iv), v1.2.0; owner decision OQ3 of 2026-09-16). "
                     "GATING = G1..G11 of the header, including 'Eisenstein in H^1_{eq,+}' and iota^* = -1 on H^0(Gamma; F_ell); "
                     "the geometric sign and the dim split are RECORDED." << std::endl;

        // ---------------- G1: iota_K from the shipped coordinates ----------------
        std::vector<uint32_t> vperm(7, 0);
        for (uint32_t v = 0; v < 7; ++v) {
            const QuadPoint z = model.vertex_position(v);
            const QuadPoint img(hecke::checked_neg<int64_t>(z.u), z.v, z.w, z.D);
            int hit = -1;
            for (uint32_t w = 0; w < 7; ++w) if (model.vertex_position(w) == img) hit = static_cast<int>(w);
            if (hit < 0) { log.fail("z -> -conj(z) sends v" + std::to_string(v) + " off the vertex set."); return log.failures; }
            vperm[v] = static_cast<uint32_t>(hit);
        }
        const std::vector<uint32_t> expected{ 3, 2, 1, 0, 5, 4, 6 };
        ++log.checks;
        if (vperm != expected) log.fail("vertex permutation differs from lem:iotamodel.");
        const IotaOnComplex ioK = build_iota(K, vperm);
        {
            uint32_t minus_count = 0;
            for (uint32_t q = 0; q < simplices.size(); ++q)
                for (uint32_t s = 0; s < simplices[q].size(); ++s) {
                    const auto& tbl = minus_sign_simplices_7vertex();
                    const bool expect_minus = std::find(tbl.begin(), tbl.end(), simplices[q][s]) != tbl.end();
                    ++log.checks;
                    if ((ioK.image_sign[q][s] < 0) != expect_minus) log.fail("sign table mismatch at q=" + std::to_string(q) + " sigma=" + std::to_string(s));
                    if (ioK.image_sign[q][s] < 0) ++minus_count;
                    ++log.checks;
                    if (ioK.image_idx[q][ioK.image_idx[q][s]] != s) log.fail("iota_K not an involution on simplices.");
                }
            std::cout << "HeckeIotaSignValidation: G1 iota_K = {0<->3, 1<->2, 4<->5, 6}, sign -1 on " << minus_count
                      << " of 23 simplices (lem:iotamodel: 6)." << std::endl;
        }
        if (log.failures != 0) return log.failures;
        const IotaOnComplex io_path = build_iota(SimplicialComplex({ { { 0 }, { 1 }, { 2 }, { 3 } }, { { 0, 1 }, { 1, 2 }, { 2, 3 } } }, "path"),
                                                 std::vector<uint32_t>{ 3, 2, 1, 0 });

        // ---------------- The grid ----------------
        const std::vector<uint32_t> ns = { 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30 };
        const std::vector<uint32_t> ells = { 5, 7, 11, 13, 101, 134217757u, 2147483647u };
        std::vector<CellResult> cells;
        for (uint32_t ell : ells)
            for (uint32_t n : ns) {
                CellResult r;
                try {
                    run_cell(K, model, ioK, io_path, n, ell, r, log);
                } catch (const std::exception& e) {
                    log.fail("EXCEPTION at n=" + std::to_string(n) + " ell=" + std::to_string(ell) + ": " + e.what());
                    r.n = n; r.ell = ell; r.note = std::string("exception: ") + e.what();
                }
                cells.push_back(r);
            }
        // Degree-0 anchor (obl:iota(v), degree-0 sentence, v1.2.0): n = 0,
        // H^0 = F_ell, iota^* = -1 (geometric: +1). GATING inside run_cell.
        int h0_anchor = 0;
        for (uint32_t ell : { 5u, 13u, 134217757u }) {
            CellResult r;
            try { run_cell(K, model, ioK, io_path, 0, ell, r, log); }
            catch (const std::exception& e) { log.fail("EXCEPTION at anchor n=0 ell=" + std::to_string(ell) + ": " + e.what()); }
            std::cout << "HeckeIotaSignValidation: degree-0 anchor n=0 ell=" << ell << ": dim H^* = (" << r.dimH0 << "," << r.dimH1
                      << "," << r.dimH2 << "), iota^* on H^0 = " << (r.h0_sign == 1 ? "+1" : (r.h0_sign == -1 ? "-1" : "not scalar"))
                      << " (contract v1.2.0: -1; geometric iota^*_geo: +1). This pins the overall sign of iota_C." << std::endl;
            if (h0_anchor == 0) h0_anchor = r.h0_sign;
            else if (h0_anchor != r.h0_sign) h0_anchor = 2;
        }
        // Odd-n control cells (rmk:iotasign first bullet): the two lifts differ,
        // and H^* = 0 so no sign exists to differ on.
        for (uint32_t n : { 3u, 7u })
            for (uint32_t ell : { 5u, 13u }) {
                CellResult r;
                try { run_cell(K, model, ioK, io_path, n, ell, r, log); }
                catch (const std::exception& e) { log.fail("EXCEPTION at odd control n=" + std::to_string(n) + ": " + e.what()); }
                std::cout << "HeckeIotaSignValidation: odd-n control n=" << n << " ell=" << ell << ": dim H^* = (" << r.dimH0 << ","
                          << r.dimH1 << "," << r.dimH2 << "), rho_n(-iota) == rho_n(iota): " << (r.lift_same ? "yes" : "no (= -rho_n(iota))") << std::endl;
                ++log.checks;
                if (r.dimH0 + r.dimH1 + r.dimH2 != 0) log.fail("odd-n control has H^* != 0.");
            }

        // ---------------- Report ----------------
        std::cout << "HeckeIotaSignValidation: per cell (iota^* = -iota^*_geo, def:iota(iv) v1.2.0; every sign below is the geometric "
                     "one negated) -- dim H^1 (+,-) vs predicted (d_{n+2}, d_{n+2}-1) [ok / MIRROR = (d-1, d) / DIFF; "
                     "D = Dickson locus, prediction n/a]; H0 = iota^* on H^0; route (a) T_p-eigenline sign; route (b) boundary sign with "
                     "the +- split of ker(res); lift = rho_n(-iota) == rho_n(iota)." << std::endl;
        std::cout << "    n    ell         dimH^*  H0   H1(+,-)  pred(+,-) split   | eig-dim primes  eps_a | dimH1(bdry) (+,-) rank(res) eps_b  ker(res)(+,-) | lift" << std::endl;
        uint32_t defined_a = 0, defined_b = 0, both = 0, skipped_a = 0, skipped_b = 0, plus_cells = 0, minus_cells = 0;
        uint32_t split_ok = 0, split_mirror = 0, split_cells = 0, dickson_cells = 0, lift_same_cells = 0;
        for (const auto& r : cells) {
            const bool split_applies = !r.dickson;
            const bool split_match = (r.dim_plus == r.pred_plus && r.dim_minus == r.pred_minus);
            const bool split_mirrored = (r.dim_plus == r.pred_minus && r.dim_minus == r.pred_plus);
            if (r.dickson) ++dickson_cells;
            if (split_applies) { ++split_cells; if (split_match) ++split_ok; else if (split_mirrored) ++split_mirror; }
            if (r.eps_a != 0) ++defined_a; else ++skipped_a;
            if (r.eps_b != 0) ++defined_b; else ++skipped_b;
            if (r.eps_a != 0 && r.eps_b != 0) ++both;
            const int e = (r.eps_a != 0) ? r.eps_a : r.eps_b;
            if (e > 0) ++plus_cells; else if (e < 0) ++minus_cells;
            if (r.lift_same) ++lift_same_cells;
            std::ostringstream primes;
            for (size_t i = 0; i < r.primes_used.size(); ++i) primes << (i ? "," : "") << r.primes_used[i];
            const char* h0 = (r.h0_sign == 1) ? "+1" : (r.h0_sign == -1 ? "-1" : (r.h0_sign == 2 ? "mx" : " ."));
            std::cout << "  " << std::setw(3) << r.n << " " << std::setw(10) << r.ell << "   (" << r.dimH0 << "," << std::setw(2) << r.dimH1 << "," << r.dimH2 << ")"
                      << "  " << h0
                      << "   (" << std::setw(2) << r.dim_plus << "," << std::setw(2) << r.dim_minus << ")"
                      << "   (" << std::setw(2) << r.pred_plus << "," << std::setw(2) << r.pred_minus << ")"
                      << (r.dickson ? "  D      " : (split_match ? "  ok     " : (split_mirrored ? "  MIRROR " : "  DIFF   ")))
                      << " |   " << std::setw(3) << r.eig_dim << "   " << std::setw(6) << primes.str() << "    " << sign_str(r.eps_a)
                      << "   |     " << std::setw(3) << r.dimH1_bdry << "    (" << r.bdry_plus << "," << r.bdry_minus << ")     " << std::setw(3) << r.rank_res
                      << "      " << sign_str(r.eps_b) << "      (" << r.cusp_plus << "," << r.cusp_minus << ")     | " << (r.lift_same ? "same" : "DIFF")
                      << (r.note.empty() ? "" : ("   " + r.note)) << std::endl;
        }
        // G10 -- one sign on the whole grid, and (v1.2.0) that sign is '+':
        // def:iota(iv) defines '+' as the side of the Eisenstein classes, so a
        // '-' anywhere means the geometric sign is not uniform across cells and
        // the definition is ill-posed. The per-cell gates above already count
        // each such cell; this is the grid-level statement.
        ++log.checks;
        if (plus_cells != 0 && minus_cells != 0) log.fail("the Eisenstein sign is NOT the same in every cell (" + std::to_string(plus_cells) + " plus, " + std::to_string(minus_cells) + " minus).");
        ++log.checks;
        if (plus_cells + minus_cells == 0) log.fail("no cell defined a sign.");
        ++log.checks;
        if (minus_cells != 0) log.fail("def:iota(iv) v1.2.0: the Eisenstein class is in H^1_{eq,-} on " + std::to_string(minus_cells) + " cell(s).");

        const std::string verdict = (plus_cells != 0 && minus_cells != 0) ? "mixed"
                                  : (plus_cells != 0 ? "plus" : (minus_cells != 0 ? "minus" : "undetermined"));
        // The geometric reading is the negation of the normalised one, by
        // def:iota(iv) and the entrywise gate iota_C == -iota_C_geo above.
        const std::string geometric = (verdict == "plus") ? "minus" : (verdict == "minus" ? "plus" : verdict);
        const char* h0_txt = (h0_anchor == -1) ? "-1" : (h0_anchor == 1 ? "+1" : "mixed");
        const char* h0_geo = (h0_anchor == -1) ? "+1" : (h0_anchor == 1 ? "-1" : "mixed");
        std::cout << "HeckeIotaSignValidation: obl:iota(v) v1.2.0 (definition-consistency oracle, GATING) -- the T_p-eigenline of eigenvalue "
                     "1+p^{n+1} and the boundary-visible class lie in H^1_{eq," << (verdict == "plus" ? "+" : (verdict == "minus" ? "-" : "?"))
                  << "} on " << (plus_cells + minus_cells) << " of " << cells.size() << " cells (" << defined_a << " by route (a), " << defined_b
                  << " by route (b), " << both << " by both, agreeing); under the geometric involution iota^*_geo = -iota^* they lie in H^1_{eq,"
                  << (geometric == "plus" ? "+" : (geometric == "minus" ? "-" : "?")) << "} (RECORDED). Route (a) undefined on " << skipped_a
                  << " cells (joint eigenspace not 1-dimensional or dim H^1 = 0), route (b) undefined on " << skipped_b
                  << " (rank(res) != 1). RECORDED split (d_{n+2}, d_{n+2}-1) matched on " << split_ok << " of " << split_cells
                  << " off-Dickson cells, MIRRORED as (d_{n+2}-1, d_{n+2}) on " << split_mirror << "; " << dickson_cells
                  << " cells on the Dickson locus (prediction n/a). iota^* on H^0(Gamma; F_ell) at n = 0: " << h0_txt
                  << " (GATING: -1; geometric: " << h0_geo << "). rho_n(-iota) == rho_n(iota) on "
                  << lift_same_cells << "/" << cells.size() << " even-n cells." << std::endl;
        std::cout << "IOTA_SIGN eisenstein=" << verdict << " (geometric=" << geometric << ") cells=" << cells.size() << " signed=" << (plus_cells + minus_cells)
                  << " route_a=" << defined_a << " route_b=" << defined_b << " both=" << both << " skipped_a=" << skipped_a
                  << " skipped_b=" << skipped_b << " split_match=" << split_ok << "/" << split_cells << " split_mirror=" << split_mirror
                  << "/" << split_cells << " dickson=" << dickson_cells << " h0_anchor=" << h0_txt << " (geometric=" << h0_geo << ")"
                  << " lift_same=" << lift_same_cells << "/" << cells.size() << " checks=" << log.checks << " gating_failures=" << log.failures << std::endl;
        return log.failures;
    } catch (const std::exception& e) {
        std::cout << "HeckeIotaSignValidation: EXCEPTION during setup: " << e.what() << std::endl;
        return 1000;
    }
}
