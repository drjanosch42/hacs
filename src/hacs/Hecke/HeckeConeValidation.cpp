// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-5 (phase E5-2) -- HeckeConeValidation.  See the
// header for the gates G1..G11 and the grid.

#include "HeckeConeValidation.h"

#include "SL2ZBoundaryPair.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../HeckeExt/HeckeBoundary.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using hecke::SL2ZBoundaryPair;
using hecke::SL2ZTessellationTables;
using hecke_ext::HeckeBoundary;
using hecke_ext::HeckeBoundaryOptions;

namespace {

struct CheckLog {
    uint32_t failures = 0;
    uint64_t checks = 0;
    void fail(const std::string& msg) {
        ++failures;
        std::cout << "HeckeConeValidation: FAILED -- " << msg << std::endl;
    }
    void check(bool ok, const std::string& msg) {
        ++checks;
        if (!ok) fail(msg);
    }
};

// =========================================================================
// Section 0 -- own arithmetic and constants.
// =========================================================================

// d_k = dim M_k(SL_2(Z)), k even >= 2.
uint32_t own_dim_Mk(uint32_t k) {
    if (k % 12 == 2) return k / 12;
    return k / 12 + 1;
}

// [E] Claim 1.8's Kummer list as (n, ell), k = n + 2.
const std::vector<std::pair<uint32_t, uint32_t>>& kummer_cells() {
    static const std::vector<std::pair<uint32_t, uint32_t>> L = {
        { 10, 691 }, { 14, 3617 }, { 16, 43867 }, { 18, 283 }, { 18, 617 },
        { 20, 131 }, { 20, 593 }, { 22, 103 }, { 22, 2294797 }, { 24, 657931 },
    };
    return L;
}
bool is_kummer_cell(uint32_t n, uint32_t ell) {
    for (const auto& c : kummer_cells()) if (c.first == n && c.second == ell) return true;
    return false;
}

bool flat_equal(const FlatMatrix& X, const FlatMatrix& Y) {
    if (X.get_rows() != Y.get_rows() || X.get_cols() != Y.get_cols()) return false;
    for (uint32_t r = 0; r < X.get_rows(); ++r)
        for (uint32_t c = 0; c < X.get_cols(); ++c)
            if (X.get_entry(r, c) != Y.get_entry(r, c)) return false;
    return true;
}

bool flat_is_zero(const FlatMatrix& X) {
    for (uint32_t r = 0; r < X.get_rows(); ++r)
        for (uint32_t c = 0; c < X.get_cols(); ++c)
            if (X.get_entry(r, c) != 0) return false;
    return true;
}

FlatMatrix hcat(const FlatMatrix& A, const FlatMatrix& B) {
    const uint32_t R = A.get_rows(), CA = A.get_cols(), CB = B.get_cols();
    std::vector<uint32_t> d(static_cast<size_t>(R) * (CA + CB), 0);
    for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = 0; j < CA; ++j) d[static_cast<size_t>(i) * (CA + CB) + j] = A.get_entry(i, j);
        for (uint32_t j = 0; j < CB; ++j) d[static_cast<size_t>(i) * (CA + CB) + CA + j] = B.get_entry(i, j);
    }
    if (R == 0 || CA + CB == 0) return FlatMatrix::CreateZero(R, CA + CB, A.get_p());
    return FlatMatrix(R, CA + CB, A.get_p(), d);
}

uint32_t own_rank(const FlatMatrix& A) {
    if (A.get_rows() == 0 || A.get_cols() == 0) return 0;
    return A.rank();
}

// Every column of Y lies in the column span of B: rank [B | Y] == rank B.
bool own_in_span(const FlatMatrix& B, const FlatMatrix& Y) {
    if (Y.get_cols() == 0) return true;
    if (B.get_cols() == 0) return flat_is_zero(Y);
    return own_rank(hcat(B, Y)) == own_rank(B);
}

// The kernel of A as a column matrix (own route through FlatMatrix::kernel).
FlatMatrix own_kernel_cols(const FlatMatrix& A) {
    const uint32_t n = A.get_cols();
    if (n == 0) return FlatMatrix::CreateZero(0, 0, A.get_p());
    if (A.get_rows() == 0) return FlatMatrix(n, A.get_p());   // the identity
    std::vector<FlatMatrix> ker = A.kernel();
    const uint32_t k = static_cast<uint32_t>(ker.size());
    if (k == 0) return FlatMatrix::CreateZero(n, 0, A.get_p());
    std::vector<uint32_t> d(static_cast<size_t>(n) * k, 0);
    for (uint32_t t = 0; t < k; ++t)
        for (uint32_t i = 0; i < n; ++i) d[static_cast<size_t>(i) * k + t] = ker[t].get_entry(i, 0);
    return FlatMatrix(n, k, A.get_p(), d);
}

FlatMatrix flat_of_matrix(const Matrix& M) { return FlatMatrix(M); }

FlatMatrix flat_of_dense(const engine::DenseBuf<uint32_t>& D, uint32_t p) {
    const uint32_t R = D.rows, C = D.cols;
    if (R == 0 || C == 0) return FlatMatrix::CreateZero(R, C, p);
    std::vector<uint32_t> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(i) * C + j] = D.at(i, j);
    return FlatMatrix(R, C, p, d);
}

FlatMatrix flat_of_block(const engine::BlockMat<uint32_t>& B, uint32_t p) {
    return flat_of_dense(B.to_dense(engine::Layout::RowMajor), p);
}

std::string join(const std::vector<uint32_t>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) oss << (i ? "," : "") << v[i];
    return oss.str();
}

// =========================================================================
// Section 1 -- the own dense cone differential (eq:conediff, assembled from
// the simplicial coboundaries and the vertex-list restriction) and the own
// zig-zag connecting map.
// =========================================================================

// The flat restriction r^q : C^q(K; V) -> C^q(K_bd; V) from the vertex lists.
FlatMatrix own_r_flat(const SimplicialComplex& K, const SimplicialComplex& Kb, uint32_t q, uint32_t dV, uint32_t p) {
    const uint32_t NK = K.get_num_q_simplices(q), Nb = Kb.get_num_q_simplices(q);
    if (NK == 0 || Nb == 0) return FlatMatrix::CreateZero(Nb * dV, NK * dV, p);
    std::vector<uint32_t> d(static_cast<size_t>(Nb) * dV * NK * dV, 0);
    const uint32_t cols = NK * dV;
    for (uint32_t s = 0; s < Nb; ++s) {
        const int64_t t = K.find_simplex_index(q, Kb.get_simplices()[q][s]);
        if (t < 0) throw std::runtime_error("own_r_flat: a boundary simplex is not in K");
        for (uint32_t k = 0; k < dV; ++k) d[static_cast<size_t>(s * dV + k) * cols + (static_cast<uint32_t>(t) * dV + k)] = 1;
    }
    return FlatMatrix(Nb * dV, cols, p, d);
}

// d_q (x) I_dV of a complex: N_{q+1} dV x N_q dV.
FlatMatrix own_coboundary_tensor(const SimplicialComplex& K, uint32_t q, uint32_t dV, uint32_t p, bool negate) {
    const uint32_t Ns = K.get_num_q_simplices(q), Nt = K.get_num_q_simplices(q + 1);
    if (Ns == 0 || Nt == 0) return FlatMatrix::CreateZero(Nt * dV, Ns * dV, p);
    const FlatMatrix D1 = K.coboundary_operator(q, p);
    const engine::ModCtx<uint32_t> ctx(p);
    std::vector<uint32_t> d(static_cast<size_t>(Nt) * dV * Ns * dV, 0);
    const uint32_t cols = Ns * dV;
    for (uint32_t i = 0; i < Nt; ++i)
        for (uint32_t j = 0; j < Ns; ++j) {
            uint32_t s = D1.get_entry(i, j);
            if (s == 0) continue;
            if (negate) s = ctx.neg(s);
            for (uint32_t k = 0; k < dV; ++k) d[static_cast<size_t>(i * dV + k) * cols + (j * dV + k)] = s;
        }
    return FlatMatrix(Nt * dV, cols, p, d);
}

// D_c^q : Cc^q -> Cc^{q+1} in H_c-indexing: [[-d_q(K), 0], [r^q, d_{q-1}(K_bd)]].
FlatMatrix own_cone_differential(const SimplicialComplex& K, const SimplicialComplex& Kb, uint32_t q, uint32_t dV, uint32_t p) {
    const uint32_t topK = K.get_dimension(), topB = Kb.get_dimension();
    const uint32_t nK_q = (q <= topK) ? K.get_num_q_simplices(q) * dV : 0;
    const uint32_t nK_q1 = (q + 1 <= topK) ? K.get_num_q_simplices(q + 1) * dV : 0;
    const uint32_t nb_qm1 = (q >= 1 && q - 1 <= topB) ? Kb.get_num_q_simplices(q - 1) * dV : 0;
    const uint32_t nb_q = (q <= topB) ? Kb.get_num_q_simplices(q) * dV : 0;
    const uint32_t rows = nK_q1 + nb_q, cols = nK_q + nb_qm1;
    if (rows == 0 || cols == 0) return FlatMatrix::CreateZero(rows, cols, p);
    std::vector<uint32_t> d(static_cast<size_t>(rows) * cols, 0);
    auto put = [&](const FlatMatrix& M, uint32_t r0, uint32_t c0) {
        for (uint32_t i = 0; i < M.get_rows(); ++i)
            for (uint32_t j = 0; j < M.get_cols(); ++j) d[static_cast<size_t>(r0 + i) * cols + c0 + j] = M.get_entry(i, j);
    };
    if (nK_q > 0 && nK_q1 > 0) put(own_coboundary_tensor(K, q, dV, p, true), 0, 0);
    if (nb_qm1 > 0 && nb_q > 0) put(own_coboundary_tensor(Kb, q - 1, dV, p, false), nK_q1, nK_q);
    if (nK_q > 0 && nb_q > 0) put(own_r_flat(K, Kb, q, dV, p), nK_q1, 0);
    return FlatMatrix(rows, cols, p, d);
}

// =========================================================================
// Section 2 -- one cell of the SL(2,Z) grid.
// =========================================================================

struct CellRecord {
    uint32_t n = 0, ell = 0;
    std::vector<uint32_t> dim_Hc, dim_H, dim_Hbd;
    uint32_t rank_r0 = 0, rank_r1 = 0, dim_interior = 0;
    std::string note;
};

struct Tally {
    uint32_t cells = 0, kummer = 0, ell23 = 0, h2_nonzero_ell23 = 0, h0c_zero = 0;
};

// Everything asserted on one cone (SL(2,Z) or the test double) that needs
// only the cone and its two complexes.
void check_cone_generic(CheckLog& log, const std::string& where, const EquivariantMappingCone& cone,
                        const SimplicialComplex& K, const SimplicialComplex& Kb, bool sl2z_model) {
    const uint32_t p = cone.ell(), dV = cone.d_V();
    const ConeCertificate& cert = cone.certificate();
    const uint32_t Q = cone.q_max();

    // G1: the certificate.
    {
        std::vector<std::string> f;
        const uint32_t bad = cert.verify(f);
        log.check(bad == 0, where + " ConeCertificate::verify() fails: " + (f.empty() ? std::string("?") : f[0]));
    }

    // G2: the own dense differential and D^2.
    std::vector<FlatMatrix> Dq;
    for (uint32_t q = 0; q <= Q; ++q) {
        Dq.push_back(own_cone_differential(K, Kb, q, dV, p));
        const FlatMatrix stored = flat_of_block(cone.D_block(q), p);
        log.check(flat_equal(Dq.back(), stored), where + " stored D_c^" + std::to_string(q) + " != own eq:conediff assembly");
    }
    for (uint32_t q = 0; q + 1 <= Q; ++q) {
        const FlatMatrix DD = Dq[q + 1] * Dq[q];
        log.check(flat_is_zero(DD), where + " own D^2 != 0 at q = " + std::to_string(q));
    }
    // the own D annihilates the stored cocycle bases and the stored coboundaries are in its image
    for (uint32_t q = 0; q <= Q; ++q) {
        const FlatMatrix H = flat_of_block(cone.H_block(q), p);
        log.check(flat_is_zero(Dq[q] * H), where + " own D kills the stored H_c basis, q = " + std::to_string(q));
        if (q >= 1) {
            const FlatMatrix B = flat_of_block(cone.B_block(q), p);
            const FlatMatrix DC = Dq[q - 1] * flat_of_block(cone.Ceq_block(q - 1), p);
            log.check(own_in_span(DC, B) && own_rank(DC) == B.get_cols(), where + " stored B_c^" + std::to_string(q) + " is not im(own D) on Ceq");
        }
    }

    // G3: exactness re-derived from the stored maps.
    for (const ConeLesSpot& s : cone.les_spots()) {
        const std::string sp = where + " spot " + s.term + "^" + std::to_string(s.q);
        if (!s.asserted) continue;
        const uint32_t q = static_cast<uint32_t>(s.q);
        uint32_t rin = 0, rout = 0;
        bool comp_ok = true;
        if (s.term == "Hc") {
            const FlatMatrix& out = cone.pi(q);
            rout = own_rank(out);
            if (q >= 1) { const FlatMatrix& in = cone.iota(q - 1); rin = own_rank(in); comp_ok = flat_is_zero(out * in); }
        } else if (s.term == "H_bulk") {
            const FlatMatrix& in = cone.pi(q);
            const FlatMatrix& out = cone.r(q);
            rin = own_rank(in); rout = own_rank(out); comp_ok = flat_is_zero(out * in);
            // im pi = ker r explicitly
            const FlatMatrix Kr = own_kernel_cols(out);
            log.check(own_in_span(Kr, in), sp + " im pi not inside ker r");
            log.check(own_in_span(in, Kr), sp + " ker r not inside im pi");
        } else {
            const FlatMatrix& in = cone.r(q);
            const FlatMatrix& out = cone.iota(q);
            rin = own_rank(in); rout = own_rank(out); comp_ok = flat_is_zero(out * in);
        }
        log.check(comp_ok, sp + " composite != 0");
        log.check(rin + rout == s.dim_middle, sp + " not exact: " + std::to_string(rin) + " + " + std::to_string(rout) + " != " + std::to_string(s.dim_middle));
        log.check(s.exact && s.composite_zero, sp + " the object records it as not exact");
    }

    // G4 (part): the own zig-zag == the stored r^q == the morphism's induced map.
    const EquivariantCohomologyOnK& bulk = cone.bulk();
    const EquivariantCohomologyOnK& bd = cone.boundary();
    const std::vector<FlatMatrix>& induced = cone.restriction().get_induced_H_eq_maps();
    for (uint32_t q = 0; q <= Q; ++q) {
        if (!cone.r_built(q)) continue;
        const uint32_t dk = (q < bulk.get_H_eq_dimensions().size()) ? bulk.get_H_eq_dimensions()[q] : 0;
        const uint32_t db = (q < bd.get_H_eq_dimensions().size()) ? bd.get_H_eq_dimensions()[q] : 0;
        const FlatMatrix& stored = cone.r(q);
        log.check(stored.get_rows() == db && stored.get_cols() == dk, where + " r^" + std::to_string(q) + " shape");
        if (dk == 0 || db == 0) continue;
        // zig-zag: lift (z, 0) into Cc^q, apply the own D, read the boundary part, project.
        const FlatMatrix HK = flat_of_dense(bulk.H_eq_block(q).to_flat(engine::Layout::RowMajor), p);   // nK x dk
        const uint32_t nK = cone.bulk_part_dim(q), fd = cone.flat_dim(q);
        std::vector<uint32_t> lift(static_cast<size_t>(fd) * dk, 0);
        for (uint32_t i = 0; i < nK; ++i)
            for (uint32_t j = 0; j < dk; ++j) lift[static_cast<size_t>(i) * dk + j] = HK.get_entry(i, j);
        const FlatMatrix L(fd, dk, p, lift);
        const FlatMatrix DL = Dq[q] * L;    // flat_{q+1} x dk; the bulk part is -d z = 0
        const uint32_t nK1 = cone.bulk_part_dim(q + 1), nb = cone.boundary_part_dim(q + 1);
        bool bulk_part_zero = true;
        for (uint32_t i = 0; i < nK1 && bulk_part_zero; ++i)
            for (uint32_t j = 0; j < dk; ++j) if (DL.get_entry(i, j) != 0) { bulk_part_zero = false; break; }
        log.check(bulk_part_zero, where + " zig-zag: the lift of a bulk cocycle is not a cocycle in the bulk part, q = " + std::to_string(q));
        std::vector<uint32_t> bdpart(static_cast<size_t>(nb) * dk, 0);
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < dk; ++j) bdpart[static_cast<size_t>(i) * dk + j] = DL.get_entry(nK1 + i, j);
        const FlatMatrix Bd(nb, dk, p, bdpart);
        const FlatMatrix Pbd = flat_of_dense(bd.P_block(q).to_dense(engine::Layout::RowMajor), p);      // db x nb
        const FlatMatrix zig = Pbd * Bd;
        log.check(flat_equal(zig, stored), where + " zig-zag connecting map != stored r^" + std::to_string(q));
        log.check(q < induced.size() && flat_equal(induced[q], stored), where + " morphism's induced map != stored r^" + std::to_string(q));
    }

    // G5 (generic part): H^0_c = 0; the first conedims identity at every ell; the second where H^2_eq = 0.
    log.check(cone.dim_Hc(0) == 0, where + " H^0_c != 0");
    if (Q >= 1 && cone.r_built(1) && cone.r_built(0) && cone.bulk_fields().reported(1) && cone.boundary_fields().reported(1) &&
        cone.boundary_fields().reported(0) && cone.bulk_fields().reported(0)) {
        const uint32_t dH0 = bulk.get_H_eq_dimensions()[0], dH0bd = bd.get_H_eq_dimensions()[0];
        const uint32_t dH1 = (1 < bulk.get_H_eq_dimensions().size()) ? bulk.get_H_eq_dimensions()[1] : 0;
        const uint32_t ker_r1 = dH1 - own_rank(cone.r(1));
        log.check(cone.dim_Hc(1) + dH0 == dH0bd + ker_r1, where + " dim H^1_c != dim H^0(bd) - dim H^0 + dim ker r^1");
    }
    if (sl2z_model) {
        // The circle model's shape: boundary top degree 1, bulk top degree 2, cone q_max 2, nothing truncated.
        log.check(cone.bulk_fields().q_max == 2 && !cone.bulk_fields().truncated && cone.bulk_fields().q_max_source == "derived" &&
                      cone.bulk_fields().degrees == std::vector<uint32_t>{ 0, 1, 2 },
                  where + " bulk degree fields");
        log.check(cone.boundary_fields().q_max == 1 && !cone.boundary_fields().truncated && cone.boundary_fields().q_max_source == "derived" &&
                      cone.boundary_fields().degrees == std::vector<uint32_t>{ 0, 1 },
                  where + " boundary degree fields");
        log.check(Q == 2 && !cone.truncated() && cone.cone_fields().q_max_source == "derived" && cone.cone_fields().degrees == std::vector<uint32_t>{ 0, 1, 2 },
                  where + " cone degree fields");
        for (const ConeLesSpot& s : cone.les_spots()) log.check(s.asserted, where + " a spot is not asserted on the model complexes");
        log.check(cone.boundary_model() == "circle" && cone.boundary_is_group_cohomology_known() &&
                      cone.boundary_is_group_cohomology() == (p != 2),
                  where + " boundary labels");
        const std::string js = cone.to_json();
        log.check(js.find("\"boundary_model\":\"circle\"") != std::string::npos &&
                      js.find(std::string("\"boundary_is_group_cohomology\":") + (p != 2 ? "true" : "false")) != std::string::npos &&
                      js.find("\"schema\":\"equivariant_mapping_cone/v1\"") != std::string::npos &&
                      js.find("\"derived_from\":[\"bulk\",\"boundary\"]") != std::string::npos,
                  where + " to_json record fields");
    }
}

CellRecord run_cell(CheckLog& log, Tally& tally, const SimplicialComplex& K, const SL2ZTessellationTables& model, uint32_t n, uint32_t ell,
                    uint64_t* digest_acc) {
    CellRecord rec;
    rec.n = n; rec.ell = ell;
    const std::string where = "(" + std::to_string(n) + ", " + std::to_string(ell) + "):";
    LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, ell), n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    SL2ZBoundaryPair pair(&coh, K);
    std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
    ++tally.cells;
    rec.dim_Hc = cone->dim_Hc_all();
    rec.dim_H = coh.get_H_eq_dimensions();
    rec.dim_Hbd = pair.boundary_cohomology().get_H_eq_dimensions();
    rec.rank_r0 = cone->rank_r(0);
    rec.rank_r1 = cone->rank_r(1);
    if (digest_acc) *digest_acc = engine::FnvU64(*digest_acc, cone->golden_digest());

    check_cone_generic(log, where, *cone, K, pair.K_bd(), true);

    // G4 (rest) and G5 (rest): against HeckeExt/HeckeBoundary.
    HeckeBoundaryOptions o;
    o.operators = { { 2, 1 }, { 2, 2 } };
    o.run_diagnostics = false;
    HeckeBoundary hb(&coh, K, model, o);
    hb.compute();
    for (uint32_t q = 0; q < 2; ++q) {
        const FlatMatrix ext = flat_of_matrix(hb.r(q));
        log.check(flat_equal(ext, cone->r(q)), where + " cone r^" + std::to_string(q) + " != HeckeBoundary::r(" + std::to_string(q) + ")");
        log.check(hb.rank_r(q) == cone->rank_r(q), where + " rank r^" + std::to_string(q) + " differs from HeckeBoundary's");
    }
    rec.dim_interior = hb.dim_interior();
    const uint32_t coker_r0 = rec.dim_Hbd[0] - rec.rank_r0;
    log.check(cone->dim_Hc(1) == coker_r0 + rec.dim_interior, where + " dim H^1_c - dim coker r^0 != dim H^1_! (HeckeBoundary)");
    // H^2_eq = 0 asserted at 6 in R^x, recorded at ell in {2, 3}; then the second conedims identity.
    const uint32_t dH2 = rec.dim_H[2];
    if (ell >= 5) {
        log.check(dH2 == 0, where + " H^2_eq != 0 at 6 in R^x");
        log.check(cone->dim_Hc(2) + rec.rank_r1 == rec.dim_Hbd[1], where + " dim H^2_c != dim H^1(bd) - rank r^1");
    } else {
        ++tally.ell23;
        if (dH2 != 0) ++tally.h2_nonzero_ell23;
        rec.note += " [ell in {2,3}: dim H^2_eq = " + std::to_string(dH2) + ", recorded]";
    }
    if (cone->dim_Hc(0) == 0) ++tally.h0c_zero;

    // G6: the measured (10, 691) and the Kummer cells through the cone.
    if (is_kummer_cell(n, ell)) {
        ++tally.kummer;
        const uint32_t two_dim_S = 2 * (own_dim_Mk(n + 2) - 1);
        log.check(cone->dim_Hc(1) == 1 + two_dim_S, where + " Kummer cell: dim H^1_c = " + std::to_string(cone->dim_Hc(1)) + " != 1 + 2 dim S_k = " + std::to_string(1 + two_dim_S));
        log.check(cone->dim_Hc(2) == 0, where + " Kummer cell: dim H^2_c != 0");
        log.check(rec.rank_r1 == 1 && rec.dim_interior == two_dim_S, where + " Kummer cell: rank r^1 / dim H^1_! are not WP-7's");
        rec.note += " [Kummer]";
        if (n == 10 && ell == 691) log.check(cone->dim_Hc(1) == 3, where + " dim H^1_c != 3 at (10, 691)");
    }
    return rec;
}

// =========================================================================
// Section 3 -- the truncation test double: the cylinder rel. both boundary
// circles (a declared truncation of the bulk, open:ell2 rider (i)).
// =========================================================================

struct CylinderPair {
    SimplicialComplex K, Kb;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_K, lssc_b;
    std::unique_ptr<EquivariantCohomologyOnK> coh_K, coh_b;
    std::unique_ptr<SimplicialComplexMorphism> scm;
    std::unique_ptr<EquivariantCochainComplexMorphismOnK> res;
};

std::unique_ptr<CylinderPair> make_cylinder_pair(uint32_t p) {
    auto cp = std::make_unique<CylinderPair>();
    // The rectangle of LocalSystemPresets (v4 ~ v0, v5 ~ v1, e45 ~ e01 give the cylinder).
    cp->K = SimplicialComplex({ { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 } },
                                { { 0, 1 }, { 0, 2 }, { 1, 2 }, { 1, 3 }, { 2, 3 }, { 2, 4 }, { 3, 4 }, { 3, 5 }, { 4, 5 } },
                                { { 0, 1, 2 }, { 1, 2, 3 }, { 2, 3, 4 }, { 3, 4, 5 } } },
                              "K_Rectangle_cone_double");
    // Both boundary circles: 0-2-4 (4 ~ 0) and 1-3-5 (5 ~ 1).
    cp->Kb = SimplicialComplex({ { { 0 }, { 1 }, { 2 }, { 3 }, { 4 }, { 5 } },
                                 { { 0, 2 }, { 1, 3 }, { 2, 4 }, { 3, 5 } } },
                               "K_Rectangle_boundary_circles");
    const uint32_t dV = 2;
    FlatMatrix M(dV, dV, p, { 1, 1, 0, 1 });   // unipotent: V^M and V_M are lines
    auto idx = [](const SimplicialComplex& C, uint32_t q, std::vector<uint32_t> v) {
        const int64_t i = C.find_simplex_index(q, v);
        if (i < 0) throw std::runtime_error("make_cylinder_pair: simplex not found");
        return static_cast<uint32_t>(i);
    };
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> relK;
    relK.emplace_back(0, idx(cp->K, 0, { 4 }), idx(cp->K, 0, { 0 }), M, "cyl_v4_v0");
    relK.emplace_back(0, idx(cp->K, 0, { 5 }), idx(cp->K, 0, { 1 }), M, "cyl_v5_v1");
    relK.emplace_back(1, idx(cp->K, 1, { 4, 5 }), idx(cp->K, 1, { 0, 1 }), M, "cyl_e45_e01");
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> relB;
    relB.emplace_back(0, idx(cp->Kb, 0, { 4 }), idx(cp->Kb, 0, { 0 }), M, "bd_v4_v0");
    relB.emplace_back(0, idx(cp->Kb, 0, { 5 }), idx(cp->Kb, 0, { 1 }), M, "bd_v5_v1");
    cp->lssc_K = std::make_unique<LocalSystemSimplicialComplex>(cp->K, relK, dV, p);
    cp->lssc_b = std::make_unique<LocalSystemSimplicialComplex>(cp->Kb, relB, dV, p);
    cp->coh_K = std::make_unique<EquivariantCohomologyOnK>(cp->lssc_K.get());
    cp->coh_K->compute_cohomology();
    cp->coh_b = std::make_unique<EquivariantCohomologyOnK>(cp->lssc_b.get());
    cp->coh_b->compute_cohomology();
    cp->scm = std::make_unique<SimplicialComplexMorphism>(&cp->Kb, &cp->K);
    if (!cp->scm->is_valid()) throw std::runtime_error("make_cylinder_pair: the boundary is not a subcomplex");
    cp->res = std::make_unique<EquivariantCochainComplexMorphismOnK>(cp->coh_b.get(), cp->coh_K.get(), cp->scm.get(), FlatMatrix(dV, p));
    cp->res->compute_induced_maps_on_cohomology();
    return cp;
}

void run_truncation_double(CheckLog& log, uint32_t p) {
    const std::string where = "cylinder double (ell = " + std::to_string(p) + "):";
    std::unique_ptr<CylinderPair> cp = make_cylinder_pair(p);
    // The untruncated cone first: a finite pair, every spot asserted.
    {
        EquivariantMappingConeOptions o;
        o.name = "cylinder rel. both circles";
        EquivariantMappingCone cone(cp->coh_K.get(), cp->coh_b.get(), cp->res.get(), o);
        cone.compute();
        check_cone_generic(log, where + " (finite)", cone, cp->K, cp->Kb, false);
        log.check(cone.q_max() == 2 && !cone.truncated(), where + " finite cone fields");
        for (const ConeLesSpot& s : cone.les_spots()) log.check(s.asserted && s.exact, where + " finite: a spot is not asserted / exact");
        // H^2_c = coker r^1 must be non-zero here, so that the truncated arm below records something.
        log.check(cone.dim_Hc(2) == cp->coh_b->get_H_eq_dimensions()[1] - cone.rank_r(1), where + " finite: dim H^2_c");
        log.check(cone.dim_Hc(2) > 0, where + " finite: H^2_c = 0 makes the truncated arm vacuous");
    }
    // A set q_max on the finite bulk is rejected (open:ell2 rider (i)).
    {
        EquivariantMappingConeOptions o;
        o.override_bulk_fields = true;
        o.bulk_fields = ComplexDegreeFields::derived_from(*cp->coh_K);
        o.bulk_fields.q_max = 1;     // set, not derived, on a finite complex
        o.bulk_fields.q_max_source = "set";
        bool rejected = false;
        try { EquivariantMappingCone cone(cp->coh_K.get(), cp->coh_b.get(), cp->res.get(), o); } catch (const std::invalid_argument&) { rejected = true; }
        log.check(rejected, where + " a set q_max on the finite bulk was accepted");
        o.bulk_fields = ComplexDegreeFields::declared_truncation(2);   // q_max == top: no scaffolding degree
        rejected = false;
        try { EquivariantMappingCone cone(cp->coh_K.get(), cp->coh_b.get(), cp->res.get(), o); } catch (const std::invalid_argument&) { rejected = true; }
        log.check(rejected, where + " a declared truncation without a scaffolding degree was accepted");
    }
    // The declared truncation of the bulk at q_max = 1 (degree 2 is scaffolding).
    {
        EquivariantMappingConeOptions o;
        o.name = "cylinder rel. both circles, bulk truncated at 1";
        o.override_bulk_fields = true;
        o.bulk_fields = ComplexDegreeFields::declared_truncation(1);
        EquivariantMappingCone cone(cp->coh_K.get(), cp->coh_b.get(), cp->res.get(), o);
        cone.compute();
        log.check(cone.bulk_fields().truncated && cone.bulk_fields().q_max == 1 && cone.bulk_fields().q_max_source == "set",
                  where + " truncated: bulk fields");
        log.check(!cone.boundary_fields().truncated && cone.boundary_fields().q_max == 1, where + " truncated: boundary fields (derived)");
        log.check(cone.truncated() && cone.q_max() == 1 && cone.cone_fields().q_max_source == "derived" && cone.cone_fields().degrees == std::vector<uint32_t>{ 0, 1 },
                  where + " truncated: cone fields (min(q_max_K, q_max_bd + 1) = 1)");
        std::vector<std::string> f;
        log.check(cone.certificate().verify(f) == 0, where + " truncated: certificate fails: " + (f.empty() ? "?" : f[0]));
        // The stored objects: degrees 0, 1 with bases; Cc^2 as scaffolding (D_1 into it), no D_2.
        bool threw = false;
        try { (void)cone.D_block(2); } catch (const std::out_of_range&) { threw = true; }
        log.check(threw, where + " truncated: D_block(2) exists");
        threw = false;
        try { (void)cone.H_block(2); } catch (const std::out_of_range&) { threw = true; }
        log.check(threw, where + " truncated: H_block(2) exists (a reported group above q_max)");
        log.check(cone.Ceq_block(2).cols() > 0, where + " truncated: the scaffolding cochain group Cc^2 is missing");
        // The spots: every spot asserted except H_bd^1 (its target H^2_c is unreported); there the
        // rank of r^1 and the cokernel dimension are recorded.
        uint32_t asserted = 0, unasserted = 0;
        for (const ConeLesSpot& s : cone.les_spots()) {
            if (s.asserted) { ++asserted; log.check(s.exact && s.composite_zero, where + " truncated: an asserted spot fails"); }
            else {
                ++unasserted;
                log.check(s.term == "H_bd" && s.q == 1, where + " truncated: the unasserted spot is not H_bd^1");
                log.check(s.in_built && !s.out_built, where + " truncated: H_bd^1's incoming map missing or outgoing map built");
                log.check(s.rank_in == cone.rank_r(1) && s.coker_in_dim == s.dim_middle - s.rank_in && s.coker_in_dim > 0,
                          where + " truncated: the recorded rank / cokernel at H_bd^1");
            }
        }
        // Hc^0, H_bulk^0, H_bd^0, Hc^1, H_bulk^1 asserted (5); H_bd^1 not.
        log.check(asserted == 5 && unasserted == 1, where + " truncated: spot census " + std::to_string(asserted) + " / " + std::to_string(unasserted));
        log.check(!cone.pi_built(2) && !cone.r_built(2) && !cone.iota_built(1) && cone.r_built(1) && cone.pi_built(1) && cone.iota_built(0),
                  where + " truncated: the map families' built flags");
        // The incoming composite at H_bulk^1 (r^1 . pi^1 = 0) is asserted -- open:ell2's spelled-out clause.
        log.check(flat_is_zero(cone.r(1) * cone.pi(1)), where + " truncated: r^1 pi^1 != 0");
        const std::string js = cone.to_json();
        log.check(js.find("\"status\":\"unreported_neighbour\"") != std::string::npos && js.find("\"coker_in_dim_recorded\":") != std::string::npos &&
                      js.find("\"cone\":{\"q_max\":1,\"truncated\":true") != std::string::npos,
                  where + " truncated: to_json");
        // OI-2 negative control (slot-5 WP-4): here bulk.reported(2) == false (the bulk is
        // declared-truncated at q_max = 1), so H2_zero == false with NO corruption at all --
        // this is a genuinely constructed cell (not the shipped grid, where H^2_eq is always
        // reported and zero) that distinguishes the pre-WP-4 shared guard from the split one.
        // Under the shared guard both eq:conedims identities were gated on H2_zero, so
        // identity (1) would never even be EVALUATED here; under the split guard it is
        // evaluated unconditionally. Corrupt r^1's ENTRIES (never its shape, so C13's shape
        // gate -- which returns from verify() early on any shape mismatch -- stays clear) to
        // break identity (1)'s own equation, and confirm it is caught while identity (2)
        // stays silent (its guard still requires H2_zero, which is false here).
        {
            ConeCertificate c = cone.certificate();
            log.check(!c.bulk.reported(2), where + " OI-2 control needs bulk.reported(2) == false");
            log.check(c.r.size() > 1 && c.r[1].get_rows() > 0 && c.r[1].get_cols() > 0 && cone.rank_r(1) > 0,
                      where + " OI-2 control needs a non-empty, non-zero-rank r^1 to corrupt");
            if (!c.bulk.reported(2) && c.r.size() > 1 && c.r[1].get_rows() > 0 && c.r[1].get_cols() > 0 && cone.rank_r(1) > 0) {
                for (uint32_t i = 0; i < c.r[1].get_rows(); ++i)
                    for (uint32_t j = 0; j < c.r[1].get_cols(); ++j) c.r[1].set_entry(i, j, 0);   // force rank r^1 = 0
                std::vector<std::string> msgs;
                (void)c.verify(msgs);
                bool id1_fired = false, id2_fired = false;
                for (const std::string& m : msgs) {
                    if (m.find("eq:conedims (1)") != std::string::npos) id1_fired = true;
                    if (m.find("eq:conedims (2)") != std::string::npos) id2_fired = true;
                }
                log.check(id1_fired, where + " OI-2 control: corrupting r^1 with H^2_eq unreported must still fail the "
                                             "unconditional eq:conedims (1)");
                log.check(!id2_fired, where + " OI-2 control: eq:conedims (2) must stay gated (H^2_eq unreported here)");
            }
        }
    }
}

// =========================================================================
// Section 4 -- the tamper arm.
// =========================================================================

void run_tamper(CheckLog& log, const SimplicialComplex& K, uint32_t n, uint32_t ell) {
    const std::string where = "tamper (" + std::to_string(n) + ", " + std::to_string(ell) + "):";
    LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, ell), n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    SL2ZBoundaryPair pair(&coh, K);
    std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
    const ConeCertificate& c0 = cone->certificate();
    log.check(c0.verify() == 0 && c0.failing_items().empty(), where + " the untampered certificate fails");
    log.check(cone->dim_Hc(1) > 0 && c0.r[1].get_rows() > 0 && c0.r[1].get_cols() > 0, where + " the cell has no H^1_c / r^1 to tamper");
    auto has = [](const std::vector<uint32_t>& v, uint32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); };
    auto none_of = [&](const std::vector<uint32_t>& v, std::initializer_list<uint32_t> items) {
        for (uint32_t x : items) if (has(v, x)) return false;
        return true;
    };
    // (a) one entry of a stored cone basis vector of H^1_c.
    {
        ConeCertificate c = c0;
        engine::DenseBuf<uint32_t> Hd = c.H[1].to_flat(engine::Layout::ColMajor);
        const uint32_t v = Hd.at(0, 0);
        Hd.at(0, 0) = (v + 1) % ell;
        c.H[1] = engine::BlockMat<uint32_t>::from_flat(c.H[1].pool(), std::move(Hd), c.H[1].row_dims(), ell, false);
        const std::vector<uint32_t> items = c.failing_items();
        log.check(!items.empty() && has(items, 5), where + " (a) a corrupted H_c basis entry does not fail C5 (items " + join(items) + ")");
        log.check(none_of(items, { 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13 }), where + " (a) a corrupted H_c basis entry fails a non-basis item (items " + join(items) + ")");
    }
    // (b) one entry of the stored r^1.
    {
        ConeCertificate c = c0;
        const uint32_t v = c.r[1].get_entry(0, 0);
        c.r[1].set_entry(0, 0, (v + 1) % ell);
        const std::vector<uint32_t> items = c.failing_items();
        log.check(!items.empty() && has(items, 6), where + " (b) a corrupted r^1 entry does not fail C6 (items " + join(items) + ")");
        log.check(none_of(items, { 1, 2, 3, 4, 5, 10, 12, 13 }), where + " (b) a corrupted r^1 entry fails a basis item (items " + join(items) + ")");
    }
    // (c) the recorded dim H^0_c.
    {
        ConeCertificate c = c0;
        c.dim_Hc[0] = 1;
        const std::vector<uint32_t> items = c.failing_items();
        log.check(has(items, 13) || has(items, 10), where + " (c) a corrupted dim H^0_c passes (items " + join(items) + ")");
    }
}

// =========================================================================
// Section 5 -- the collector digest (in-process and in child processes).
// =========================================================================

const std::vector<std::pair<uint32_t, uint32_t>>& collect_grid() {
    static const std::vector<std::pair<uint32_t, uint32_t>> g = { { 2, 5 }, { 10, 691 }, { 16, 7 }, { 12, 134217757u }, { 6, 2 } };
    return g;
}

uint64_t collect_digest(const SimplicialComplex& K) {
    uint64_t h = engine::kFnvOffset64;
    for (const auto& cell : collect_grid()) {
        const uint32_t n = cell.first, ell = cell.second;
        LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, ell), n + 1, ell);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        SL2ZBoundaryPair pair(&coh, K);
        std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
        h = engine::FnvU64(h, cone->golden_digest());
        // the collectors are defined: at least dim_Hc^q per degree, Hc^q / Pc^q, and the maps
        const std::vector<ConeGoldenObject> g = cone->golden_objects();
        if (g.size() < 3 * (cone->q_max() + 1)) throw std::runtime_error("golden_objects: too few records");
    }
    return h;
}

std::string find_self_exe() {
    char buf[4096];
    const ssize_t k = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (k <= 0) return std::string();
    buf[k] = '\0';
    return std::string(buf);
}

bool child_digest(const std::string& exe, uint32_t threads, uint64_t& digest_out) {
    const std::string cmd = "ENGINE_THREADS=" + std::to_string(threads) + " HECKE_CONE_CHILD=1 \"" + exe + "\" HeckeConeValidation 2>/dev/null";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) return false;
    std::string out;
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) out += line;
    const int rc = ::pclose(f);
    if (rc != 0) return false;
    const std::string key = "CONE_COLLECT digest=0x";
    const size_t pos = out.find(key);
    if (pos == std::string::npos) return false;
    digest_out = std::strtoull(out.c_str() + pos + key.size(), nullptr, 16);
    return true;
}

double status_mb(const char* key) {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1.0;
    char line[256];
    double out = -1.0;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, key, std::strlen(key)) == 0) { out = std::atof(line + std::strlen(key)) / 1024.0; break; }
    std::fclose(f);
    return out;
}

} // namespace

uint32_t HeckeConeValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    CheckLog log;
    Tally tally;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);

        if (const char* child = std::getenv("HECKE_CONE_CHILD"); child && *child) {
            const uint64_t d = collect_digest(K);
            std::cout << "CONE_COLLECT digest=0x" << std::hex << d << std::dec << std::endl;
            return 0;
        }

        std::cout << "HeckeConeValidation: the equivariant mapping cone Cone(r) of the pair (7-vertex model, circle model) "
                     "(EquivariantMappingCone / Hecke/SL2ZBoundaryPair): eq:conediff, Dcone^2 = 0, the long exact sequence "
                     "eq:coneles exact at every asserted spot, the connecting map +r^q by an own zig-zag, H^0_c = 0, eq:conedims, "
                     "the degree fields of open:ell2, ConeCertificate::verify() with a tamper arm; GATING = G1..G11 of the header."
                  << std::endl;

        // ---- the optional large-n record (never asserted) ----------------------
        if (const char* big = std::getenv("HECKE_CONE_LARGE"); big && *big) {
            const uint32_t n = static_cast<uint32_t>(std::strtoul(big, nullptr, 10));
            const auto tb = std::chrono::steady_clock::now();
            LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, 134217757u), n + 1, 134217757u);
            EquivariantCohomologyOnK coh(&lssc);
            coh.compute_cohomology();
            const auto tc = std::chrono::steady_clock::now();
            // slot-5 WP-5 (risk R-1): the cone's INPUT FLOOR, measured rather
            // than inferred -- the process high-water mark after the bulk
            // cohomology the cone is required to read in every degree, before
            // a single cone object exists.  Measurement output only.
            const double hwm_inputs = status_mb("VmHWM:");
            SL2ZBoundaryPair pair(&coh, K);
            std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
            const auto td = std::chrono::steady_clock::now();
            // slot-4 WP-4 (E5 cost): the construction / verify() split.  compute()
            // runs verify() itself; this second, identical call measures its share
            // (the certificate is a pure function of the stored matrices, so the
            // two calls do the same work and neither changes a value).
            const uint32_t bad_large = cone->certificate().verify();
            const auto te = std::chrono::steady_clock::now();
            const double t_verify = std::chrono::duration<double>(te - td).count();
            const double t_cone = std::chrono::duration<double>(td - tc).count();
            log.check(bad_large == 0, "CONE_LARGE: the certificate fails on the large cell");
            std::cout << "  CONE_LARGE n=" << n << " cohomology " << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double>(tc - tb).count() << " s, cone " << t_cone
                      << " s (build " << (t_cone - t_verify) << " s + verify " << t_verify
                      << " s), dim_Hc=[" << join(cone->dim_Hc_all()) << "], inputs_VmHWM=" << std::setprecision(1) << hwm_inputs
                      << " MB, VmHWM=" << std::setprecision(1) << status_mb("VmHWM:") << " MB" << std::endl;
        }

        // ---- Section 2: the grid -------------------------------------------------
        std::vector<std::pair<uint32_t, uint32_t>> grid;
        const uint32_t ells[] = { 2, 3, 5, 7, 691, 134217757u, 2147483647u };
        for (uint32_t ell : ells)
            for (uint32_t n = 0; n <= 16; n += 2) grid.emplace_back(n, ell);
        grid.emplace_back(20, 5);
        grid.emplace_back(10, 13);
        for (const auto& kc : kummer_cells())
            if (!(kc.first <= 16 && (kc.second == 691))) grid.emplace_back(kc.first, kc.second);   // (10, 691) is on the grid already
        std::vector<CellRecord> records;
        uint64_t digest_a = engine::kFnvOffset64;
        for (const auto& cell : grid) records.push_back(run_cell(log, tally, K, model, cell.first, cell.second, &digest_a));
        log.check(tally.kummer == 10, "not all ten Kummer cells ran (" + std::to_string(tally.kummer) + ")");
        log.check(tally.h0c_zero == tally.cells, "H^0_c != 0 on some cell");

        // ---- Section 3: the truncation double --------------------------------------
        run_truncation_double(log, 7);
        run_truncation_double(log, 2);

        // ---- Section 4: the tamper arm -----------------------------------------------
        run_tamper(log, K, 10, 691);
        run_tamper(log, K, 10, 13);

        // ---- Section 5: the collector digest -----------------------------------------
        {
            const uint64_t d_in1 = collect_digest(K);
            const uint64_t d_in2 = collect_digest(K);
            log.check(d_in1 == d_in2, "collector digest differs between two in-process builds");
            const std::string exe = find_self_exe();
            if (exe.empty()) {
                std::cout << "  collector child arm SKIPPED: /proc/self/exe unavailable" << std::endl;
            } else {
                for (uint32_t t : { 1u, 16u }) {
                    uint64_t d = 0;
                    const bool ok = child_digest(exe, t, d);
                    log.check(ok, "collector child arm: the ENGINE_THREADS=" + std::to_string(t) + " child produced no digest");
                    if (ok) log.check(d == d_in1, "collector digest differs under ENGINE_THREADS=" + std::to_string(t));
                }
            }
            std::cout << "  collector digest 0x" << std::hex << d_in1 << std::dec << " (two in-process builds and the ENGINE_THREADS=1 / 16 children); grid digest 0x"
                      << std::hex << digest_a << std::dec << std::endl;
        }

        // ---- the table ------------------------------------------------------------
        std::cout << "  (n, ell): dim H^q_c | dim H^q_eq | dim H^q(bd) | rank r^0, r^1 | dim H^1_!" << std::endl;
        for (const CellRecord& r : records)
            std::cout << "  (" << std::setw(2) << r.n << ", " << std::setw(10) << r.ell << "): [" << join(r.dim_Hc) << "] | [" << join(r.dim_H)
                      << "] | [" << join(r.dim_Hbd) << "] | " << r.rank_r0 << ", " << r.rank_r1 << " | " << r.dim_interior << r.note << std::endl;
    } catch (const std::exception& e) {
        log.fail(std::string("exception: ") + e.what());
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "CONE cells=" << tally.cells << " kummer=" << tally.kummer << " ell23_cells=" << tally.ell23
              << " ell23_H2_nonzero=" << tally.h2_nonzero_ell23 << " checks=" << log.checks << " failures=" << log.failures << " time=" << std::fixed
              << std::setprecision(2) << secs << "s" << std::endl;
    return log.failures;
}
