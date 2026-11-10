// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 4, WP-3 (phase E3) -- HeckeInvolutionValidation.
// See the header for the arms.  The oracle is hecke::iota_oracle, exported by
// this package from Hecke/HeckeIotaSignValidation.h; the production object
// never calls it.

#include "HeckeInvolutionValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)

#include "HeckeSignSplit.h"
#include "../Hecke/DicksonCount.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeBoundaryOperator.h"
#include "../Hecke/HeckeConeAction.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeInvolution.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckePairHomotopy.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/HeckeIotaSignValidation.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/SL2ZBoundaryPair.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../Util/ValidationCheck.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hecke::HeckeInvolution;
using hecke::IotaSimplicialMap;
using hecke::SignSplitStatus;
using hecke::SL2ZBoundaryPair;
using hecke::SL2ZTessellationTables;
using Elt = uint32_t;

oc_util::Checker Check("  [FAIL] ", true, 60);

template <class T>
void CheckEq(const T& got, const T& want, const std::string& what) {
    ++Check.checks;
    if (!(got == want)) {
        ++Check.fails;
        if (Check.fails <= 60)
            std::cout << "  [FAIL] " << what << ": got " << got << ", want " << want << std::endl;
    }
}

// --------------------------------------------------------------------------
// Local helpers (the validation's own, never the production object's).
// --------------------------------------------------------------------------

FlatMatrix vcat(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_cols() != B.get_cols()) throw std::runtime_error("vcat: column mismatch");
    const uint32_t C = A.get_cols();
    std::vector<uint32_t> d;
    d.reserve(static_cast<size_t>(A.get_rows() + B.get_rows()) * C);
    for (uint32_t i = 0; i < A.get_rows(); ++i) for (uint32_t j = 0; j < C; ++j) d.push_back(A.get_entry(i, j));
    for (uint32_t i = 0; i < B.get_rows(); ++i) for (uint32_t j = 0; j < C; ++j) d.push_back(B.get_entry(i, j));
    return FlatMatrix(A.get_rows() + B.get_rows(), C, A.get_p(), d);
}

MatrixT<uint32_t> to_ext(const FlatMatrix& M) {
    MatrixT<uint32_t> out(M.get_rows(), M.get_cols(), M.get_p());
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) out.data[i][j] = M.get_entry(i, j);
    return out;
}

FlatMatrix flat_of_dense(const engine::DenseBuf<uint32_t>& D, uint32_t p) {
    const uint32_t R = D.rows, C = D.cols;
    if (R == 0u || C == 0u) return FlatMatrix::CreateZero(R, C, p);
    std::vector<uint32_t> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(i) * C + j] = D.at(i, j);
    return FlatMatrix(R, C, p, d);
}

FlatMatrix flat_of_block(const engine::BlockMat<uint32_t>& B, uint32_t p) {
    return flat_of_dense(B.to_dense(engine::Layout::RowMajor), p);
}

uint32_t dim_Mk(uint32_t k) { return (k % 12u == 2u) ? (k / 12u) : (k / 12u + 1u); }

uint32_t eigenspace_dim(const FlatMatrix& M, uint32_t lambda, uint32_t ell) {
    const uint32_t d = M.get_rows();
    if (d == 0u) return 0u;
    return d - (M - FlatMatrix::CreateScalar(d, lambda, ell)).rank();
}

// --------------------------------------------------------------------------
// One grid cell.
// --------------------------------------------------------------------------

struct Cell {
    uint32_t n = 0, ell = 0;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::unique_ptr<HeckeInvolution> iota;
};

std::unique_ptr<Cell> build_cell(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                                 uint32_t n, uint32_t ell, bool exact_rows = true) {
    auto c = std::make_unique<Cell>();
    c->n = n; c->ell = ell;
    const std::vector<SL2ZRelationDatum> data = GenerateSL2ZRelationsWithGroupData(K, n, ell);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    rels.reserve(data.size());
    for (const auto& d : data) rels.push_back(d.rel);
    c->lssc = std::make_unique<LocalSystemSimplicialComplex>(K, rels, n + 1, ell);
    c->coh = std::make_unique<EquivariantCohomologyOnK>(c->lssc.get());
    c->coh->compute_cohomology();
    c->iota = std::make_unique<HeckeInvolution>(c->coh.get(),
                                                hecke::BuildIotaSimplicialMapOnModel(K, model),
                                                "iota^* (bulk)");
    if (exact_rows) c->iota->set_exact_relation_data(data);
    c->iota->compute();
    return c;
}

// --------------------------------------------------------------------------
// The per-cell verdict, for the IOTA_SIGN_PRODUCTION line.
// --------------------------------------------------------------------------

struct CellVerdict {
    uint32_t n = 0, ell = 0;
    uint32_t dimH0 = 0, dimH1 = 0, dimH2 = 0;
    uint32_t dim_plus = 0, dim_minus = 0;
    bool dickson = false;
    int eps_a = 0, eps_b = 0;     // +1 / -1 / 0 = undefined
    uint32_t eig_dim = 0, rank_res = 0;
    bool split_ok = false, split_mirror = false;
};

} // namespace

uint32_t HeckeInvolutionValidation(void) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HeckeInvolutionValidation (hecke-extensions slot 4, WP-3 / phase E3) ===" << std::endl;
    std::cout << "    the PRODUCTION involution iota^* of [C, sec:involution] "
                 "(Hecke/HeckeInvolution) and the sign split (HeckeExt/HeckeSignSplit)." << std::endl;
    std::cout << "    iota^* := -iota^*_geo ([C, def:iota](iv), v1.2.0; owner decision OQ3 of 2026-09-16): "
                 "'+' is BY DEFINITION the Eisenstein side." << std::endl;
    std::cout << "    The oracle is hecke::iota_oracle (the involution HeckeIotaSignValidation builds from "
                 "[C, def:iota] alone); the comparison is ENTRYWISE and the production object never calls it."
              << std::endl;
    Check.reset();

    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);

        // ------------------------------------------------------------------
        // A0 -- iota_K itself, production vs oracle.
        // ------------------------------------------------------------------
        const IotaSimplicialMap iotaK = hecke::BuildIotaSimplicialMapOnModel(K, model);
        const hecke::iota_oracle::IotaOnComplex oracleK =
            hecke::iota_oracle::build_iota(K, std::vector<uint32_t>{ 3, 2, 1, 0, 5, 4, 6 });
        Check(iotaK.vertex_perm == oracleK.vperm, "A0: production iota_K vertex permutation == oracle's");
        Check(iotaK.image_index == oracleK.image_idx, "A0: production image indices == oracle's");
        Check(iotaK.image_sign == oracleK.image_sign, "A0: production orientation signs == oracle's");
        CheckEq(iotaK.num_minus_signs(), 6u, "A0: [C, lem:iotamodel] has six minus signs");
        std::cout << "    A0: iota_K = {0<->3, 1<->2, 4<->5, 6}, sign -1 on " << iotaK.num_minus_signs()
                  << " of 23 simplices, identical to the oracle's." << std::endl;

        // ------------------------------------------------------------------
        // A1 -- the 77-cell grid.
        // ------------------------------------------------------------------
        const std::vector<uint32_t> ns = { 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30 };
        const std::vector<uint32_t> ells = { 5, 7, 11, 13, 101, 134217757u, 2147483647u };
        std::vector<CellVerdict> verdicts;
        uint32_t plus_cells = 0, minus_cells = 0, defined_a = 0, defined_b = 0, both = 0;
        uint32_t split_ok = 0, split_mirror = 0, split_cells = 0, dickson_cells = 0;

        for (uint32_t ell : ells)
            for (uint32_t n : ns) {
                CellVerdict v;
                v.n = n; v.ell = ell;
                v.dickson = hecke::IsDicksonExceptionalCount(n, ell);
                const std::string where = "(n=" + std::to_string(n) + ", ell=" + std::to_string(ell) + ")";
                std::unique_ptr<Cell> c = build_cell(K, model, n, ell);
                const auto& dims = c->coh->get_H_eq_dimensions();
                v.dimH0 = dims.size() > 0 ? dims[0] : 0u;
                v.dimH1 = dims.size() > 1 ? dims[1] : 0u;
                v.dimH2 = dims.size() > 2 ? dims[2] : 0u;

                // --- the oracle's own construction, from def:iota alone.
                const FlatMatrix rho_o = hecke::iota_oracle::rho_n(hecke::iota_oracle::iota_mat(), n, ell);
                Check(c->iota->rho_iota() == rho_o, where + " A1: production rho_n(iota) == oracle's");
                Check(rho_o == hecke::iota_oracle::rho_iota_closed_form(n, ell),
                      where + " A1: rho_n(iota) == diag((-1)^{n-j}) ([C, eq:rhoniota])");

                const uint32_t Q = c->iota->num_degrees();
                for (uint32_t q = 0; q < Q; ++q) {
                    const FlatMatrix prod = c->iota->cochain_matrix(q, -1);
                    const FlatMatrix prod_geo = c->iota->cochain_matrix(q, +1);
                    const FlatMatrix orc = hecke::iota_oracle::iota_flat(oracleK, q, rho_o, ell, -1);
                    const FlatMatrix orc_geo = hecke::iota_oracle::iota_flat(oracleK, q, rho_o, ell, +1);
                    Check(prod == orc, where + " A1: production iota_C^(" + std::to_string(q)
                                           + ") == oracle, ENTRYWISE");
                    Check(prod_geo == orc_geo, where + " A1: production iota_C_geo^(" + std::to_string(q)
                                                   + ") == oracle, ENTRYWISE");
                    Check(prod == prod_geo * (ell - 1u),
                          where + " A1: iota^* == -iota^*_geo at q=" + std::to_string(q));
                    // iota^2 = I and iota d = d iota, on the MATERIALISED matrices.
                    const uint32_t flat = prod.get_rows();
                    Check(prod * prod == FlatMatrix(flat, ell),
                          where + " A1: iota_C^2 == I at q=" + std::to_string(q));
                    if (q + 1 < Q) {
                        const FlatMatrix D = hecke::iota_oracle::kron_identity(
                            K.coboundary_operator(q, ell), n + 1u, ell);
                        const FlatMatrix hi = c->iota->cochain_matrix(q + 1, -1);
                        Check(D * prod == hi * D,
                              where + " A1: iota_C d == d iota_C at q=" + std::to_string(q));
                    }
                    // iota^* on H^q: the production matrix against P_q (oracle iota_C) H_eq.
                    const uint32_t dH = (q < dims.size()) ? dims[q] : 0u;
                    if (dH != 0u) {
                        const auto& H = c->coh->get_H_eq_bases_fm();
                        const auto& P = c->coh->get_H_eq_projectors_from_C_eq_flat_fm();
                        const FlatMatrix Hq = hecke::iota_oracle::multivector(H[q], flat, ell);
                        const FlatMatrix want = P[q] * (orc * Hq);
                        Check(c->iota->matrix(q) == want,
                              where + " A1: production iota^* on H^" + std::to_string(q)
                                  + " == P_q (oracle iota_C) H_eq, ENTRYWISE");
                        Check(c->iota->matrix(q) * c->iota->matrix(q) == FlatMatrix(dH, ell),
                              where + " A1: (iota^*)^2 == I on H^" + std::to_string(q));
                    }
                }

                // --- the split.
                CheckEq(static_cast<int>(c->iota->sign_split_status()),
                        static_cast<int>(SignSplitStatus::Split), where + " A1: 2 in R^x so the split applies");
                v.dim_plus = c->iota->dim_H_plus(1);
                v.dim_minus = c->iota->dim_H_minus(1);
                Check(v.dim_plus + v.dim_minus == v.dimH1, where + " A1: dim H^1_+ + dim H^1_- == dim H^1");
                {
                    const FlatMatrix& I1 = c->iota->matrix(1);
                    Check(v.dim_plus == eigenspace_dim(I1, 1u, ell),
                          where + " A1: dim H^1_+ == dim ker(iota^* - 1)");
                    Check(v.dim_minus == eigenspace_dim(I1, ell - 1u, ell),
                          where + " A1: dim H^1_- == dim ker(iota^* + 1)");
                }
                if (!v.dickson) {
                    ++split_cells;
                    const uint32_t dk = dim_Mk(n + 2u);
                    v.split_ok = (v.dim_plus == dk && v.dim_minus == dk - 1u);
                    v.split_mirror = (v.dim_plus == dk - 1u && v.dim_minus == dk);
                    if (v.split_ok) ++split_ok;
                    else if (v.split_mirror) ++split_mirror;
                } else {
                    ++dickson_cells;
                }

                // --- [C, obl:iota](v) through HeckeSignSplit, p in {2, 3, 5}.
                std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
                FlatMatrix stacked = FlatMatrix::CreateZero(0, v.dimH1, ell);
                FlatMatrix v_a = FlatMatrix::CreateZero(0, 0, ell);
                {
                    hecke::HeckeOperatorFamily fam(c->coh.get(), K, model, ell, n);
                    uint32_t used = 0;
                    for (uint64_t p : { 2ull, 3ull, 5ull }) {
                        if (p == ell) continue;
                        const std::vector<Matrix> Aq = fam.T_p(p);
                        const FlatMatrix A(Aq[1]);
                        ops.emplace_back(p, to_ext(A));
                        // [C, obl:iota](v) in EVERY degree, not only q = 1
                        // (asserted here: the whole grid has 6 in R^x, so
                        // [C, prop:iotacommutes](3c) applies).
                        for (uint32_t q = 0; q < c->iota->num_degrees() && q < Aq.size(); ++q) {
                            if (c->iota->dim_H(q) == 0u) continue;
                            const FlatMatrix Aqq(Aq[q]);
                            Check(c->iota->matrix(q) * Aqq == Aqq * c->iota->matrix(q),
                                  where + " A1: [C, obl:iota](v) [iota^*, A^(" + std::to_string(q)
                                      + ")(T_" + std::to_string(p) + ")] = 0");
                        }
                        if (v.dimH1 == 0u) continue;
                        if (used < 2u || v.eig_dim != 1u) {
                            const uint32_t lambda = ModArithmetic::mod_add(
                                1u, ModArithmetic::mod_pow(static_cast<uint32_t>(p % ell), n + 1u, ell), ell);
                            stacked = vcat(stacked, A - FlatMatrix::CreateScalar(v.dimH1, lambda, ell));
                            ++used;
                            v.eig_dim = v.dimH1 - stacked.rank();
                        }
                    }
                }
                if (v.dimH1 != 0u) {
                    hecke_ext::HeckeSignSplit split(ell, n, 1, to_ext(c->iota->matrix(1)),
                                                    to_ext(c->iota->H_plus_basis(1)),
                                                    to_ext(c->iota->H_minus_basis(1)), ops);
                    split.compute();
                    CheckEq(split.certificate().verify(), 0u, where + " A1: SignSplitCertificate::verify() == 0");
                    CheckEq(split.dim_plus(), v.dim_plus, where + " A1: HeckeSignSplit dim H^1_+");
                    CheckEq(split.dim_minus(), v.dim_minus, where + " A1: HeckeSignSplit dim H^1_-");
                    for (size_t i = 0; i < split.num_operators(); ++i)
                        Check(split.op(i).commutes && split.op(i).restricted,
                              where + " A1: A^(1)(T_p) restricts to both signs");
                    (void)split.golden_objects();   // defined, never emitted
                }

                // --- route (a): the Eisenstein T_p-eigenline.
                if (v.dimH1 != 0u && v.eig_dim == 1u) {
                    const std::vector<FlatMatrix> ker = stacked.kernel();
                    if (ker.size() == 1u) {
                        v_a = ker[0];
                        const FlatMatrix w = c->iota->matrix(1) * v_a;
                        if (w == v_a) v.eps_a = +1;
                        else if (w == v_a * (ell - 1u)) v.eps_a = -1;
                        else Check(false, where + " A1: the Eisenstein T_p-eigenline is not an iota^*-eigenline");
                        if (v.eps_a != 0) {
                            ++defined_a;
                            Check(v.eps_a == +1,
                                  where + " A1: [C, obl:iota](v) GATING -- the Eisenstein class lies in H^1_+");
                        }
                    }
                }

                // --- route (b): the boundary-visible class, through the PRODUCTION restriction.
                {
                    SL2ZBoundaryPair pair(c->coh.get(), K);
                    HeckeInvolution iota_bd(&pair.boundary_cohomology(),
                                            hecke::BuildIotaSimplicialMapFromVertexPermutation(
                                                pair.K_bd(), hecke::IotaBoundaryPathVertexPermutation()),
                                            "iota^*_bdry");
                    iota_bd.compute();
                    const auto& r = pair.restriction().get_induced_H_eq_maps();
                    Check(hecke::IotaCommutesWithRestriction(*c->iota, iota_bd, r),
                          where + " A1/A4: r^q iota^* == iota^*_bdry r^q ([C, eq:iotacochain] naturality)");
                    if (r.size() > 1u && r[1].get_rows() != 0u && r[1].get_cols() != 0u) {
                        const FlatMatrix& R = r[1];
                        v.rank_res = R.rank();
                        if (v.rank_res == 1u) {
                            const FlatMatrix RI = R * c->iota->matrix(1);
                            if (RI == R) v.eps_b = +1;
                            else if (RI == R * (ell - 1u)) v.eps_b = -1;
                            else Check(false, where + " A1: rank(r^1) = 1 but r^1 iota^* is not +-r^1");
                            if (v.eps_b != 0) {
                                ++defined_b;
                                Check(v.eps_b == +1,
                                      where + " A1: [C, obl:iota](v) GATING -- the boundary-visible class "
                                              "lies in H^1_+");
                            }
                        }
                        if (v.eps_a != 0 && v_a.get_rows() != 0u)
                            Check(!(R * v_a).is_zero(),
                                  where + " A1: the Eisenstein eigenline is boundary-visible");
                    }
                }
                if (v.eps_a != 0 && v.eps_b != 0) {
                    ++both;
                    CheckEq(v.eps_a, v.eps_b, where + " A1: route (a) sign == route (b) sign");
                }
                const int e = (v.eps_a != 0) ? v.eps_a : v.eps_b;
                if (e > 0) ++plus_cells; else if (e < 0) ++minus_cells;
                verdicts.push_back(v);
            }

        // --- the degree-0 anchor, from the PRODUCTION object.
        int h0_anchor = 0, h0_anchor_geo = 0;
        for (uint32_t ell : { 5u, 13u, 134217757u }) {
            std::unique_ptr<Cell> c = build_cell(K, model, 0, ell);
            const auto& dims = c->coh->get_H_eq_dimensions();
            const uint32_t d0 = dims.empty() ? 0u : dims[0];
            CheckEq(d0, 1u, "A1: dim H^0(Gamma; F_ell) == 1 at n = 0");
            if (d0 != 1u) continue;
            const FlatMatrix& I0 = c->iota->matrix(0);
            const int s = (I0 == FlatMatrix(1, ell)) ? +1
                        : (I0 == FlatMatrix::CreateScalar(1, ell - 1u, ell) ? -1 : 2);
            Check(s == -1,
                  "A1: [C, obl:iota](v) degree-0 anchor -- iota^* = -1 on H^0(Gamma; F_ell) at n = 0 "
                  "(constants are ODD; this is what pins the global sign)");
            // The geometric involution, built by the SAME production code with
            // global_sign = +1, must give +1 there -- the tamper control of A5.
            const auto& H = c->coh->get_H_eq_bases_fm();
            const auto& P = c->coh->get_H_eq_projectors_from_C_eq_flat_fm();
            const uint32_t flat = c->coh->get_lssc()->get_base_K_complex()->get_num_q_simplices(0) * 1u;
            const FlatMatrix H0 = hecke::iota_oracle::multivector(H[0], flat, ell);
            const FlatMatrix I0_geo = P[0] * (c->iota->cochain_matrix(0, +1) * H0);
            const int sg = (I0_geo == FlatMatrix(1, ell)) ? +1
                         : (I0_geo == FlatMatrix::CreateScalar(1, ell - 1u, ell) ? -1 : 2);
            Check(sg == +1, "A1: the GEOMETRIC involution gives +1 on H^0 at n = 0 (recorded)");
            if (h0_anchor == 0) { h0_anchor = s; h0_anchor_geo = sg; }
            else if (h0_anchor != s) h0_anchor = 2;
        }

        const std::string verdict = (plus_cells != 0u && minus_cells != 0u) ? "mixed"
                                  : (plus_cells != 0u ? "plus" : (minus_cells != 0u ? "minus" : "undetermined"));
        const std::string geometric = (verdict == "plus") ? "minus" : (verdict == "minus" ? "plus" : verdict);
        // The reference line of HeckeIotaSignValidation, REPRODUCED FROM THE
        // PRODUCTION OBJECT (slot-4 exit criterion item 3).
        const std::string reference = "eisenstein=plus (geometric=minus)";
        const std::string produced = "eisenstein=" + verdict + " (geometric=" + geometric + ")";
        CheckEq(produced, reference,
                "A1: the production object reproduces HeckeIotaSignValidation's reference sign line");
        CheckEq(h0_anchor, -1, "A1: the production degree-0 anchor is -1");
        CheckEq(h0_anchor_geo, +1, "A1: the geometric degree-0 anchor is +1 (recorded)");
        CheckEq(split_mirror, 0u, "A1: the recorded +- split is never the MIRROR split");
        CheckEq(split_ok, split_cells, "A1: dim H^1_+- == (d_{n+2}, d_{n+2}-1) on every off-Dickson cell");

        {
            std::ostringstream dk;
            for (const auto& v : verdicts)
                if (v.dickson)
                    dk << " (" << v.n << "," << v.ell << "):(" << v.dim_plus << "," << v.dim_minus
                       << ")/pred(" << dim_Mk(v.n + 2u) << "," << (dim_Mk(v.n + 2u) - 1u) << ")";
            std::cout << "    A1: RECORDED dim H^1_+- on the " << dickson_cells
                      << " Dickson cells (the prediction of [C, obl:iota](v) is n/a there):"
                      << dk.str() << std::endl;
        }
        std::cout << "IOTA_SIGN_PRODUCTION " << produced << " cells=" << verdicts.size()
                  << " signed=" << (plus_cells + minus_cells) << " route_a=" << defined_a
                  << " route_b=" << defined_b << " both=" << both << " split_match=" << split_ok << "/"
                  << split_cells << " split_mirror=" << split_mirror << "/" << split_cells
                  << " dickson=" << dickson_cells << " h0_anchor=" << (h0_anchor == -1 ? "-1" : "?")
                  << " (geometric=" << (h0_anchor_geo == 1 ? "+1" : "?") << ")" << std::endl;

        // ------------------------------------------------------------------
        // A2 -- (10, 691).
        // ------------------------------------------------------------------
        {
            const uint32_t n = 10, ell = 691;
            std::unique_ptr<Cell> c = build_cell(K, model, n, ell);
            CheckEq(c->iota->dim_H(1), 3u, "A2: dim H^1 = 3 at (10, 691)");
            CheckEq(c->iota->dim_H_plus(1), 2u, "A2: dim H^1_+ = 2 at (10, 691) ([E] Claim 1.8: H^1_+ = T_m)");
            CheckEq(c->iota->dim_H_minus(1), 1u, "A2: dim H^1_- = 1 at (10, 691) ([E]: H^1_- = F_691)");

            std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
            {
                hecke::HeckeOperatorFamily fam(c->coh.get(), K, model, ell, n);
                for (uint64_t p : { 2ull, 3ull, 5ull }) ops.emplace_back(p, to_ext(FlatMatrix(fam.T_p(p)[1])));
            }
            hecke_ext::HeckeSignSplit split(ell, n, 1, to_ext(c->iota->matrix(1)),
                                            to_ext(c->iota->H_plus_basis(1)),
                                            to_ext(c->iota->H_minus_basis(1)), ops);
            split.compute();
            CheckEq(split.certificate().verify(), 0u, "A2: SignSplitCertificate::verify() == 0 at (10, 691)");

            // certificate = cyclic on BOTH signs.
            const auto& cp = split.certificate().plus_block_cyclic;
            const auto& cm = split.certificate().minus_block_cyclic;
            Check(!cp.empty(), "A2: the plus side has at least one block");
            Check(!cm.empty(), "A2: the minus side has at least one block");
            for (size_t i = 0; i < cp.size(); ++i)
                Check(cp[i] != 0u, "A2: certificate = cyclic on H^1_+ at (10, 691)");
            for (size_t i = 0; i < cm.size(); ++i)
                Check(cm[i] != 0u, "A2: certificate = cyclic on H^1_- at (10, 691)");

            // The Eisenstein line, DEFINED, and in the bottom Loewy layer
            // (soc(V) == m_S V is exactly "bottom Loewy layer").
            const hecke_ext::EisensteinLine& eis = split.eisenstein_line();
            Check(eis.applicable, "A2: the Eisenstein line is applicable at (10, 691)");
            Check(eis.defined, "A2: the Eisenstein line is DEFINED at (10, 691) ([E] D6, dim_F soc = 1)");
            CheckEq(eis.dim_socle_over_F, 1u, "A2: dim_F soc(X^+_{4,m}) = 1");
            Check(eis.socle_equals_mV,
                  "A2: soc(X^+_{4,m}) == m_S X^+_{4,m} -- the BOTTOM LOEWY LAYER ([E] sec:tests)");
            CheckEq(eis.status, std::string("defined"), "A2: the Eisenstein line status");

            // The UNSPLIT H^1: socle 2-dimensional, so the line is undefined there.
            {
                hecke::HeckeModuleDecomposition dec(ops, ell);
                dec.compute();
                CheckEq(static_cast<uint32_t>(dec.blocks().size()), 1u,
                        "A2: the unsplit H^1 has one maximal ideal at (10, 691)");
                if (dec.blocks().size() == 1u) {
                    const hecke_ext::LocalAlgebra L = hecke_ext::ComputeLocalAlgebra(dec.blocks()[0], ops, ell);
                    Check(L.ok, "A2: the unsplit block is certified local");
                    if (L.ok && !L.n.empty()) {
                        MatrixT<uint32_t> stack(static_cast<uint32_t>(L.n.size()) * L.D, L.D, ell);
                        for (size_t i = 0; i < L.n.size(); ++i)
                            for (uint32_t r = 0; r < L.D; ++r)
                                for (uint32_t cc = 0; cc < L.D; ++cc)
                                    stack.data[static_cast<uint32_t>(i) * L.D + r][cc] = L.n[i].data[r][cc];
                        const MatrixT<uint32_t> soc = hecke::KernelAsMatrix(stack);
                        CheckEq(soc.cols, 2u,
                                "A2: the UNSPLIT socle is 2-dimensional at (10, 691), so the Eisenstein "
                                "line is \"undefined\" there ([E] D6)");
                        CheckEq(L.f, 1u, "A2: residue degree f = 1 at (10, 691)");
                    }
                }
            }

            // ----------------------------------------------------------------
            // A3 -- the unsplit blocks are the direct sum of the per-sign blocks.
            // ----------------------------------------------------------------
            {
                const auto* du = split.decomposition_unsplit();
                const auto* dp = split.decomposition(hecke_ext::SignSide::Plus);
                const auto* dm = split.decomposition(hecke_ext::SignSide::Minus);
                Check(du != nullptr && dp != nullptr && dm != nullptr, "A3: all three decompositions exist");
                if (du && dp && dm) {
                    uint32_t sum_u = 0, sum_p = 0, sum_m = 0;
                    for (const auto& b : du->blocks()) sum_u += b.dim;
                    for (const auto& b : dp->blocks()) sum_p += b.dim;
                    for (const auto& b : dm->blocks()) sum_m += b.dim;
                    CheckEq(sum_p + sum_m, sum_u,
                            "A3: the per-sign block dimensions sum to the unsplit dimension");
                    // BLOCK BY BLOCK (E3 acceptance (c)).  Two blocks sit at the
                    // SAME maximal ideal exactly when Stage 2's cross-block
                    // evaluation test says so (hecke_ext::SameMaximalIdeal), so
                    // that -- and not a residue-degree coincidence -- is the
                    // matching used here: every per-sign block matches EXACTLY
                    // ONE unsplit block, and the per-sign dimensions at each
                    // unsplit block sum to that block's dimension.
                    std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> fam_p, fam_m;
                    for (size_t i = 0; i < split.num_operators(); ++i) {
                        fam_p.emplace_back(split.op(i).p, split.op(i).A_plus);
                        fam_m.emplace_back(split.op(i).p, split.op(i).A_minus);
                    }
                    std::vector<hecke_ext::LocalAlgebra> LU;
                    for (const auto& bu : du->blocks()) LU.push_back(hecke_ext::ComputeLocalAlgebra(bu, ops, ell));
                    std::vector<uint32_t> covered(LU.size(), 0u);
                    auto match_side = [&](const hecke::HeckeModuleDecomposition* d,
                                          const std::vector<std::pair<uint64_t, MatrixT<uint32_t>>>& fam,
                                          const char* side) {
                        for (const auto& b : d->blocks()) {
                            const hecke_ext::LocalAlgebra Lb = hecke_ext::ComputeLocalAlgebra(b, fam, ell);
                            Check(Lb.ok, std::string("A3: the ") + side + " block is certified local");
                            if (!Lb.ok) continue;
                            uint32_t hits = 0, which = 0;
                            for (uint32_t i = 0; i < LU.size(); ++i) {
                                if (!LU[i].ok) continue;
                                if (hecke_ext::SameMaximalIdeal(LU[i], Lb.t).same_ideal) { ++hits; which = i; }
                            }
                            CheckEq(hits, 1u,
                                    std::string("A3: the ") + side + " block sits at EXACTLY ONE "
                                    "unsplit maximal ideal ([XA M5] cross-block evaluation test)");
                            if (hits == 1u) covered[which] += b.dim;
                        }
                    };
                    match_side(dp, fam_p, "plus");
                    match_side(dm, fam_m, "minus");
                    for (uint32_t i = 0; i < LU.size(); ++i)
                        CheckEq(covered[i], du->blocks()[i].dim,
                                "A3: the unsplit block is the DIRECT SUM of the per-sign blocks at the "
                                "same maximal ideal (E3 acceptance (c), block by block)");
                }
            }

            // ----------------------------------------------------------------
            // A4 -- boundary and cone compatibility, at (10, 691).
            // ----------------------------------------------------------------
            {
                SL2ZBoundaryPair pair(c->coh.get(), K);
                HeckeInvolution iota_bd(&pair.boundary_cohomology(),
                                        hecke::BuildIotaSimplicialMapFromVertexPermutation(
                                            pair.K_bd(), hecke::IotaBoundaryPathVertexPermutation()),
                                        "iota^*_bdry");
                iota_bd.compute();
                const auto& r = pair.restriction().get_induced_H_eq_maps();
                Check(hecke::IotaCommutesWithRestriction(*c->iota, iota_bd, r),
                      "A4: r^q iota^* == iota^*_bdry r^q at (10, 691)");

                std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
                const uint32_t d_V = c->iota->d_V();
                // iota_c := iota^* (+) iota^*_bdry on the cone's flat layout
                // (bulk part first, boundary part second; [C, eq:conediff]).
                auto iota_c_flat = [&](uint32_t q) {
                    const uint32_t nb = cone->bulk_part_dim(q);
                    const uint32_t nd = cone->boundary_part_dim(q);
                    const uint32_t dim = nb + nd;
                    std::vector<uint32_t> d(static_cast<size_t>(dim) * dim, 0u);
                    if (nb != 0u) {
                        const FlatMatrix A = c->iota->cochain_matrix(q, -1);
                        for (uint32_t i = 0; i < nb; ++i)
                            for (uint32_t j = 0; j < nb; ++j)
                                d[static_cast<size_t>(i) * dim + j] = A.get_entry(i, j);
                    }
                    if (nd != 0u) {
                        const FlatMatrix B = iota_bd.cochain_matrix(q - 1u, -1);
                        for (uint32_t i = 0; i < nd; ++i)
                            for (uint32_t j = 0; j < nd; ++j)
                                d[static_cast<size_t>(nb + i) * dim + (nb + j)] = B.get_entry(i, j);
                    }
                    return FlatMatrix(dim, dim, ell, d);
                };
                (void)d_V;
                for (uint32_t q = 0; q + 1 <= cone->q_max(); ++q) {
                    const FlatMatrix Iq = iota_c_flat(q);
                    const FlatMatrix Iq1 = iota_c_flat(q + 1u);
                    const FlatMatrix Dq = flat_of_block(cone->D_block(q), ell);
                    Check(Dq * Iq == Iq1 * Dq,
                          "A4: D_c iota_c == iota_c D_c at q=" + std::to_string(q)
                              + " (it follows from iota d = d iota and r iota = iota_bd r)");
                    Check(Iq * Iq == FlatMatrix(Iq.get_rows(), ell),
                          "A4: iota_c^2 == id at q=" + std::to_string(q));
                }
                // iota_c on H^q_c, and the RECORDED mirror split of [C, rmk:iotapairing].
                std::ostringstream mirror;
                for (uint32_t q = 0; q <= cone->q_max(); ++q) {
                    const uint32_t dHc = cone->dim_Hc(q);
                    if (dHc == 0u) { mirror << " q" << q << ":(0,0)"; continue; }
                    const FlatMatrix Ic = flat_of_block(cone->P_block(q), ell) *
                                          (iota_c_flat(q) * flat_of_block(cone->H_block(q), ell));
                    Check(Ic * Ic == FlatMatrix(dHc, ell),
                          "A4: (iota_c)^2 == id on H^" + std::to_string(q) + "_c");
                    const uint32_t dp = eigenspace_dim(Ic, 1u, ell);
                    const uint32_t dm = eigenspace_dim(Ic, ell - 1u, ell);
                    Check(dp + dm == dHc, "A4: H^q_c splits under iota_c");
                    mirror << " q" << q << ":(" << dp << "," << dm << ")";
                }
                std::cout << "    A4: RECORDED dim H^q_{c,+-} at (10, 691) =" << mirror.str()
                          << "  (against the mirror split [C, rmk:iotapairing] predicts; audit F13)"
                          << std::endl;

                // [iota_c, A_c^{(q)}] -- COMPUTED AND RECORDED, NEVER ASSERTED
                // (slot-4 plan WP-3 scope 4): it needs iota h = h iota on the
                // stored C_eq, which no [C] statement provides.  Reported to
                // WP-8 as a contract finding.
                {
                    hecke::HeckeCosetTables cosets(2ull, 1u, ell, n);
                    hecke::HeckeTauTable tau(K, model, cosets, ell, n);
                    hecke::HeckeBoundaryTauTable taub(K, model, cosets);
                    hecke::HeckePairHomotopy eta(tau, taub);
                    hecke::HeckeOperatorOnK bulk_op(c->coh.get(), &cosets, &tau);
                    bulk_op.compute();
                    hecke::HeckeBoundaryOperator bd_op(&pair.boundary_cohomology(), &cosets, &taub);
                    bd_op.compute();
                    hecke::HeckeConeAction act(cone.get(), &bulk_op, &bd_op, &eta);
                    act.compute();
                    std::ostringstream rec;
                    uint32_t degrees = 0, commuting = 0;
                    for (uint32_t q = 0; q <= cone->q_max(); ++q) {
                        const uint32_t dHc = cone->dim_Hc(q);
                        if (dHc == 0u) continue;
                        const FlatMatrix Ic = flat_of_block(cone->P_block(q), ell) *
                                              (iota_c_flat(q) * flat_of_block(cone->H_block(q), ell));
                        const FlatMatrix Ac = FlatMatrix(act.A_c(q));
                        ++degrees;
                        const bool ok = (Ic * Ac == Ac * Ic);
                        if (ok) ++commuting;
                        rec << " q" << q << ":" << (ok ? "yes" : "NO");
                    }
                    std::cout << "    A4: RECORDED [iota_c, A_c^{(q)}(T_2)] = 0 at (10, 691) on "
                              << commuting << "/" << degrees << " degrees --" << rec.str()
                              << "  (NEVER asserted: it needs iota h = h iota on the stored C_eq, "
                                 "which no [C] statement provides; slot-4 WP-3 scope 4)" << std::endl;
                    ++Check.checks;   // the recording itself ran
                }
            }
        }

        // ------------------------------------------------------------------
        // A4b -- the compatibility arms on the HeckeConeValidation grid.
        // r^q iota^* = iota^*_bdry r^q and D_c iota_c = iota_c D_c are
        // ASSERTED; dim H^q_{c,+-} is RECORDED.  ell = 2 is in the grid on
        // purpose: iota_c exists there ([C, prop:iotasplit](2) removes the
        // SPLIT, not the involution), so the two identities must still hold.
        // ------------------------------------------------------------------
        {
            uint32_t cone_cells = 0, cone_degrees = 0;
            std::ostringstream rec;
            for (uint32_t ell : { 2u, 3u, 5u, 7u, 691u }) {
                for (uint32_t n : { 0u, 2u, 10u }) {
                    const std::string where = "(n=" + std::to_string(n) + ", ell=" + std::to_string(ell) + ")";
                    std::unique_ptr<Cell> c = build_cell(K, model, n, ell);
                    SL2ZBoundaryPair pair(c->coh.get(), K);
                    HeckeInvolution iota_bd(&pair.boundary_cohomology(),
                                            hecke::BuildIotaSimplicialMapFromVertexPermutation(
                                                pair.K_bd(), hecke::IotaBoundaryPathVertexPermutation()),
                                            "iota^*_bdry");
                    iota_bd.compute();
                    Check(hecke::IotaCommutesWithRestriction(*c->iota, iota_bd,
                                                             pair.restriction().get_induced_H_eq_maps()),
                          where + " A4b: r^q iota^* == iota^*_bdry r^q");
                    std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
                    auto iota_c_flat = [&](uint32_t q) {
                        const uint32_t nb = cone->bulk_part_dim(q);
                        const uint32_t nd = cone->boundary_part_dim(q);
                        const uint32_t dim = nb + nd;
                        std::vector<uint32_t> d(static_cast<size_t>(dim) * dim, 0u);
                        if (nb != 0u) {
                            const FlatMatrix A = c->iota->cochain_matrix(q, -1);
                            for (uint32_t i = 0; i < nb; ++i)
                                for (uint32_t j = 0; j < nb; ++j)
                                    d[static_cast<size_t>(i) * dim + j] = A.get_entry(i, j);
                        }
                        if (nd != 0u) {
                            const FlatMatrix B = iota_bd.cochain_matrix(q - 1u, -1);
                            for (uint32_t i = 0; i < nd; ++i)
                                for (uint32_t j = 0; j < nd; ++j)
                                    d[static_cast<size_t>(nb + i) * dim + (nb + j)] = B.get_entry(i, j);
                        }
                        return FlatMatrix(dim, dim, ell, d);
                    };
                    for (uint32_t q = 0; q + 1 <= cone->q_max(); ++q) {
                        const FlatMatrix Iq = iota_c_flat(q);
                        const FlatMatrix Dq = flat_of_block(cone->D_block(q), ell);
                        Check(Dq * Iq == iota_c_flat(q + 1u) * Dq,
                              where + " A4b: D_c iota_c == iota_c D_c at q=" + std::to_string(q));
                        Check(Iq * Iq == FlatMatrix(Iq.get_rows(), ell),
                              where + " A4b: iota_c^2 == id at q=" + std::to_string(q));
                    }
                    rec << " " << where << ":";
                    for (uint32_t q = 0; q <= cone->q_max(); ++q) {
                        const uint32_t dHc = cone->dim_Hc(q);
                        if (dHc == 0u) { rec << "(0,0)"; continue; }
                        const FlatMatrix Ic = flat_of_block(cone->P_block(q), ell) *
                                              (iota_c_flat(q) * flat_of_block(cone->H_block(q), ell));
                        Check(Ic * Ic == FlatMatrix(dHc, ell),
                              where + " A4b: (iota_c)^2 == id on H^" + std::to_string(q) + "_c");
                        const uint32_t dp = eigenspace_dim(Ic, 1u, ell);
                        const uint32_t dm = eigenspace_dim(Ic, ell - 1u, ell);
                        if (ell != 2u)
                            Check(dp + dm == dHc, where + " A4b: H^q_c splits under iota_c at 2 in R^x");
                        rec << "(" << dp << "," << dm << ")";
                        ++cone_degrees;
                    }
                    ++cone_cells;
                }
            }
            std::cout << "    A4b: cone compatibility asserted on " << cone_cells
                      << " cells; RECORDED dim H^q_{c,+-} (ell = 2's pair is ker(iota_c - id) twice, "
                      << "[C, prop:iotasplit](2)):" << rec.str() << std::endl;
            (void)cone_degrees;
        }

        // ------------------------------------------------------------------
        // A5 -- the tamper arm.
        // ------------------------------------------------------------------
        {
            const uint32_t n = 4, ell = 13;
            const std::vector<SL2ZRelationDatum> data = GenerateSL2ZRelationsWithGroupData(K, n, ell);
            std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
            for (const auto& d : data) rels.push_back(d.rel);
            LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
            EquivariantCohomologyOnK coh(&lssc);
            coh.compute_cohomology();

            // (a) a flipped orientation sign on one simplex fails [C, obl:iota](i).
            for (uint32_t q = 0; q < 3u; ++q) {
                IotaSimplicialMap bad = hecke::BuildIotaSimplicialMapOnModel(K, model);
                bad.image_sign[q][0] = -bad.image_sign[q][0];
                bool threw = false;
                std::string msg;
                try {
                    HeckeInvolution inv(&coh, bad, "tampered sign");
                    inv.compute();
                } catch (const std::exception& e) { threw = true; msg = e.what(); }
                Check(threw, "A5: a flipped orientation sign at q=" + std::to_string(q) + " is rejected");
                Check(msg.find("obl:iota](i)") != std::string::npos,
                      "A5: ... and it fails [C, obl:iota](i) EXACTLY, not another clause");
            }
            // (b) a tampered vertex permutation fails (i) too.
            {
                IotaSimplicialMap bad = hecke::BuildIotaSimplicialMapFromVertexPermutation(
                    K, std::vector<uint32_t>{ 3, 2, 1, 0, 5, 4, 6 });
                bad.vertex_perm[6] = 0;   // no longer an involution
                bool threw = false;
                try {
                    HeckeInvolution inv(&coh, bad, "tampered vertex perm");
                    inv.compute();
                } catch (const std::exception&) { threw = true; }
                Check(threw, "A5: a non-involutive vertex permutation is rejected");
            }
            // (c) +iota^*_geo in place of iota^* fails the degree-0 anchor.
            //     (The degree-0 anchor is the ONLY clause that separates them:
            //      iota^*_geo passes (i), (ii), (iii) and the split.)
            {
                std::unique_ptr<Cell> c0 = build_cell(K, model, 0, 13);
                const auto& H = c0->coh->get_H_eq_bases_fm();
                const auto& P = c0->coh->get_H_eq_projectors_from_C_eq_flat_fm();
                const uint32_t flat = 7u;   // 7 vertices, d_V = 1
                const FlatMatrix H0 = hecke::iota_oracle::multivector(H[0], flat, 13u);
                const FlatMatrix geo = P[0] * (c0->iota->cochain_matrix(0, +1) * H0);
                const FlatMatrix norm = c0->iota->matrix(0);
                Check(!(geo == norm), "A5: +iota^*_geo differs from iota^* on H^0 at n = 0");
                Check(norm == FlatMatrix::CreateScalar(1, 12u, 13u),
                      "A5: iota^* = -1 on H^0 (the anchor the tamper would break)");
                Check(geo == FlatMatrix(1, 13u), "A5: +iota^*_geo = +1 there, i.e. it FAILS the anchor");
            }
        }

        // ------------------------------------------------------------------
        // A6 -- the small-ell cells the 77-cell grid does not have.
        // ------------------------------------------------------------------
        {
            // ell = 3: 2 in R^x so the split applies, but 6 is NOT, so
            // [C, prop:iotacommutes](3c) is unavailable and the commutation is
            // RECORDED rather than asserted.
            uint32_t ell3_ops = 0, ell3_commuting = 0;
            for (uint32_t n : { 2u, 4u, 10u }) {
                std::unique_ptr<Cell> c = build_cell(K, model, n, 3u);
                const std::string where = "(n=" + std::to_string(n) + ", ell=3)";
                CheckEq(static_cast<int>(c->iota->sign_split_status()),
                        static_cast<int>(SignSplitStatus::Split), where + " A6: the split applies at ell = 3");
                for (uint32_t q = 0; q < c->iota->num_degrees(); ++q) {
                    const FlatMatrix prod = c->iota->cochain_matrix(q, -1);
                    const FlatMatrix orc = hecke::iota_oracle::iota_flat(
                        oracleK, q, hecke::iota_oracle::rho_n(hecke::iota_oracle::iota_mat(), n, 3u), 3u, -1);
                    Check(prod == orc, where + " A6: production iota_C == oracle at q=" + std::to_string(q));
                }
                std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
                {
                    hecke::HeckeOperatorFamily fam(c->coh.get(), K, model, 3u, n);
                    for (uint64_t p : { 2ull, 5ull }) ops.emplace_back(p, to_ext(FlatMatrix(fam.T_p(p)[1])));
                }
                if (c->iota->dim_H(1) != 0u) {
                    hecke_ext::HeckeSignSplit split(3u, n, 1, to_ext(c->iota->matrix(1)),
                                                    to_ext(c->iota->H_plus_basis(1)),
                                                    to_ext(c->iota->H_minus_basis(1)), ops);
                    split.compute();
                    CheckEq(split.certificate().verify(), 0u, where + " A6: the certificate verifies at ell = 3");
                    Check(!split.certificate().commutation_asserted,
                          where + " A6: the commutation is RECORDED, not asserted, at ell = 3");
                    for (size_t i = 0; i < split.num_operators(); ++i) {
                        ++ell3_ops;
                        if (split.op(i).commutes) ++ell3_commuting;
                    }
                }
            }
            std::cout << "    A6: ell = 3 -- [iota^*, A^(1)] = 0 RECORDED on " << ell3_commuting << "/"
                      << ell3_ops << " operators ([C, prop:iotacommutes](3c) is unavailable at 6 not in R^x)."
                      << std::endl;

            // ell = 2: no split at all ([C, prop:iotasplit](2)).
            for (uint32_t n : { 0u, 2u }) {
                std::unique_ptr<Cell> c = build_cell(K, model, n, 2u);
                const std::string where = "(n=" + std::to_string(n) + ", ell=2)";
                CheckEq(static_cast<int>(c->iota->sign_split_status()),
                        static_cast<int>(SignSplitStatus::NotApplicableCharTwo),
                        where + " A6: sign_split = not_applicable at ell = 2 ([C, prop:iotasplit](2))");
                CheckEq(c->iota->dim_H_plus(1), 0u, where + " A6: no +- basis is offered at ell = 2");
                CheckEq(c->iota->dim_H_minus(1), 0u, where + " A6: ... on either side");
                // [C, rmk:iotaelltwo]: at ell = 2, iota^* = iota^*_geo identically.
                for (uint32_t q = 0; q < c->iota->num_degrees(); ++q)
                    Check(c->iota->cochain_matrix(q, -1) == c->iota->cochain_matrix(q, +1),
                          where + " A6: iota^* == iota^*_geo at ell = 2 ([C, rmk:iotaelltwo])");
                const auto& cert = c->iota->certificate();
                for (uint32_t q = 0; q < c->iota->num_degrees(); ++q)
                    Check(cert.char_two_rank_n[q] + cert.char_two_dim_ker_n[q] == cert.dim_H[q],
                          where + " A6: prop:iotasplit(2)'s a and b are recorded consistently");
                CheckEq(c->iota->certificate().verify(), 0u, where + " A6: the certificate verifies at ell = 2");
                // The HeckeExt side must refuse to split there as well.
                if (c->iota->dim_H(1) != 0u) {
                    hecke_ext::HeckeSignSplit split(2u, n, 1, to_ext(c->iota->matrix(1)),
                                                    MatrixT<uint32_t>(c->iota->dim_H(1), 0, 2u),
                                                    MatrixT<uint32_t>(c->iota->dim_H(1), 0, 2u), {});
                    split.compute();
                    CheckEq(static_cast<int>(split.status()),
                            static_cast<int>(hecke_ext::SignSplitStatus::NotApplicable),
                            where + " A6: HeckeSignSplit reports not_applicable at ell = 2");
                    CheckEq(split.eisenstein_line().status, std::string("not_applicable"),
                            where + " A6: no Eisenstein line at ell = 2");
                    Check(!split.certificate().commutation_asserted,
                          where + " A6: [C, obl:iota](v)'s commutation is NOT asserted at ell = 2 "
                                  "(the +- statement it is about does not exist there)");
                    CheckEq(static_cast<uint32_t>(split.num_operators()), 0u,
                            where + " A6: no operator is restricted at ell = 2");
                }
            }
        }
    } catch (const std::exception& e) {
        ++Check.fails;
        std::cout << "  [FAIL] uncaught exception: " << e.what() << std::endl;
    }

    // PERF1 audit R2: under ENGINE_BACKEND=legacy the (iii-c) carry checks must take the FlatMatrix route.
    // (S24 CL-L2) The route and its counter exist only in an OC_ENABLE_LEGACY=ON build (Hecke/legacy/
    // HeckeInvolutionLegacy.inc); an OFF build prints SKIP-ARM instead of the IOTA_CARRY_ROUTE line.
    if (oc::legacy::RunLegacyArm("HeckeInvolutionValidation", "IOTA_CARRY_ROUTE")) {
#if OC_ENABLE_LEGACY
        const bool legacy = engine::elim::use_legacy();
        const uint64_t lc = hecke::iota_carry_legacy_calls();
        std::cout << "IOTA_CARRY_ROUTE use_legacy=" << (legacy ? 1 : 0) << " legacy_route_calls=" << lc << std::endl;
        if (legacy && lc == 0u) {
            ++Check.fails;
            std::cout << "  [FAIL] legacy backend active but the iota carry checks never took the FlatMatrix route" << std::endl;
        }
#endif
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "IOTA_PRODUCTION checks=" << Check.checks << " failures=" << Check.fails
              << " seconds=" << secs << std::endl;
    std::cout << "    checks " << Check.checks << ", failing " << Check.fails << std::endl;
    if (Check.fails == 0u) std::cout << "=== HeckeInvolutionValidation PASSED ===" << std::endl;
    else std::cout << "=== HeckeInvolutionValidation FAILED ===" << std::endl;
    return Check.fails;
}
