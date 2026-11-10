// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 2, WP-7 (phase E4b) -- HeckeBoundaryParabolicValidation.
// See the header for the independence statement and the gate list. [C] =
// docs/latex/gl2_contract.tex, [E] = docs/latex/hecke_extensions.tex.

#include "HeckeBoundaryParabolicValidation.h"

#include "HeckeBoundary.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeBoundaryDiagnostics.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using hecke_ext::HeckeBoundary;
using hecke_ext::HeckeBoundaryCertificate;
using hecke_ext::HeckeBoundaryOptions;
using hecke::SL2ZTessellationTables;

struct CheckLog {
    uint32_t failures = 0;
    uint64_t checks = 0;
    void fail(const std::string& msg) {
        ++failures;
        std::cout << "HeckeBoundaryParabolicValidation: FAILED -- " << msg << std::endl;
    }
    void check(bool ok, const std::string& msg) {
        ++checks;
        if (!ok) fail(msg);
    }
};

// =========================================================================
// Section 0 -- our own arithmetic and classical constants.
// =========================================================================

// d_k = dim M_k(SL_2(Z)), k even >= 2 (obl:iota(v), cl:dimH1).
uint32_t own_dim_Mk(uint32_t k) {
    if (k % 12 == 2) return k / 12;
    return k / 12 + 1;
}

// rmk:dickson: n in the semigroup <ell + 1, ell^2 - ell>.
bool own_is_dickson(uint32_t n, uint32_t ell) {
    if (n == 0) return false;
    const uint64_t d1 = static_cast<uint64_t>(ell) + 1u;
    const uint64_t d2 = static_cast<uint64_t>(ell) * (static_cast<uint64_t>(ell) - 1u);
    for (uint64_t b = 0; b * d2 <= n; ++b)
        if ((static_cast<uint64_t>(n) - b * d2) % d1 == 0) return true;
    return false;
}

bool own_is_prime(uint64_t m) {
    if (m < 2) return false;
    for (uint64_t d = 2; d * d <= m; ++d) if (m % d == 0) return false;
    return true;
}

uint32_t own_eis(uint32_t ell, uint64_t p, uint32_t n) {
    const engine::ModCtx<uint32_t> ctx(ell);
    return ctx.add(1u % ell, ctx.pow(static_cast<uint32_t>(p % ell), n + 1));
}

// [E] Claim 1.8's published Kummer list, (k, ell).
const std::vector<std::pair<uint32_t, uint32_t>>& kummer_list() {
    static const std::vector<std::pair<uint32_t, uint32_t>> L = {
        { 12, 691 }, { 16, 3617 }, { 18, 43867 }, { 20, 283 }, { 20, 617 },
        { 22, 131 }, { 22, 593 }, { 24, 103 }, { 24, 2294797 }, { 26, 657931 },
    };
    return L;
}

bool in_kummer_list(uint32_t k, uint32_t ell) {
    for (const auto& kl : kummer_list()) if (kl.first == k && kl.second == ell) return true;
    return false;
}

FlatMatrix flat_of(const Matrix& M) { return FlatMatrix(M); }

bool flat_equal(const FlatMatrix& X, const FlatMatrix& Y) {
    if (X.get_rows() != Y.get_rows() || X.get_cols() != Y.get_cols()) return false;
    for (uint32_t r = 0; r < X.get_rows(); ++r)
        for (uint32_t c = 0; c < X.get_cols(); ++c)
            if (X.get_entry(r, c) != Y.get_entry(r, c)) return false;
    return true;
}

FlatMatrix hcat(const FlatMatrix& A, const FlatMatrix& B) {
    const uint32_t R = A.get_rows(), CA = A.get_cols(), CB = B.get_cols();
    std::vector<uint32_t> d(static_cast<size_t>(R) * (CA + CB), 0);
    for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = 0; j < CA; ++j) d[static_cast<size_t>(i) * (CA + CB) + j] = A.get_entry(i, j);
        for (uint32_t j = 0; j < CB; ++j) d[static_cast<size_t>(i) * (CA + CB) + CA + j] = B.get_entry(i, j);
    }
    return FlatMatrix(R, CA + CB, A.get_p(), d);
}

FlatMatrix vcat(const FlatMatrix& A, const FlatMatrix& B) {
    const uint32_t C = A.get_cols();
    std::vector<uint32_t> d;
    d.reserve(static_cast<size_t>(A.get_rows() + B.get_rows()) * C);
    for (uint32_t i = 0; i < A.get_rows(); ++i) for (uint32_t j = 0; j < C; ++j) d.push_back(A.get_entry(i, j));
    for (uint32_t i = 0; i < B.get_rows(); ++i) for (uint32_t j = 0; j < C; ++j) d.push_back(B.get_entry(i, j));
    return FlatMatrix(A.get_rows() + B.get_rows(), C, A.get_p(), d);
}

uint32_t flat_rank(const FlatMatrix& M) {
    if (M.get_rows() == 0 || M.get_cols() == 0) return 0;
    return M.rank();
}

// =========================================================================
// Section 1 -- the own restriction r^q from the stored bases.
// =========================================================================

uint32_t simplex_index(const SimplicialComplex& K, uint32_t q, std::vector<uint32_t> verts) {
    std::sort(verts.begin(), verts.end());
    const auto& list = K.get_simplices().at(q);
    for (uint32_t s = 0; s < list.size(); ++s) if (list[s] == verts) return s;
    throw std::runtime_error("HeckeBoundaryParabolicValidation: boundary simplex not found in K");
}

// Column i = P_bd^q * (restriction of the i-th H^q_eq(K) basis cochain to K_bd
// by vertex-list matching).
FlatMatrix own_restriction(const HeckeBoundary& hb, uint32_t q) {
    const EquivariantCohomologyOnK& cK = hb.bulk_cohomology();
    const EquivariantCohomologyOnK& cB = hb.boundary_cohomology();
    const SimplicialComplex& K = hb.K();
    const SimplicialComplex& Kbd = hb.K_bd();
    const uint32_t ell = hb.ell(), d = hb.d_V();
    const uint32_t dK = hb.dim_H_bulk(q), dB = hb.dim_H_bd(q);
    FlatMatrix R = FlatMatrix::CreateZero(dB, dK, ell);
    if (dK == 0 || dB == 0) return R;
    const auto& HK = cK.get_H_eq_bases_fm().at(q);
    const FlatMatrix& P = cB.get_H_eq_projectors_from_C_eq_flat_fm().at(q);
    const uint32_t Nb = Kbd.get_num_q_simplices(q);
    std::vector<uint32_t> bd_to_K(Nb);
    for (uint32_t s = 0; s < Nb; ++s) bd_to_K[s] = simplex_index(K, q, Kbd.get_simplices()[q][s]);
    for (uint32_t i = 0; i < dK; ++i) {
        std::vector<uint32_t> flat(static_cast<size_t>(Nb) * d, 0);
        for (uint32_t s = 0; s < Nb; ++s)
            for (uint32_t k = 0; k < d; ++k) flat[static_cast<size_t>(s) * d + k] = HK[i].get_entry(bd_to_K[s], k);
        const FlatMatrix coords = P * FlatMatrix(Nb * d, 1, ell, flat);
        for (uint32_t j = 0; j < dB; ++j) R.set_entry(j, i, coords.get_entry(j, 0));
    }
    return R;
}

// =========================================================================
// Section 2 -- one cell.
// =========================================================================

struct Tally {
    uint32_t cells = 0, cells_small_ell = 0;
    uint32_t stable_ops = 0, unstable_ops_small_ell = 0;
    uint32_t eis_defined = 0, eis_undefined_dickson = 0, eis_undefined_kummer = 0, eis_undefined_other = 0;
    uint32_t harder_applicable = 0, harder_consistent = 0, harder_inconclusive = 0, kummer_cells = 0, kummer_interior_all = 0;
    uint32_t classical_cells = 0, classical_match = 0;
    uint32_t dickson_cells = 0, dickson_ge = 0;
    uint32_t recorded_dim_cells = 0;
    uint32_t split_direct_cells = 0, charpoly_product = 0, charpoly_ops = 0, chi0_quotient = 0;
    uint32_t quotient_divides_chi = 0, quotient_chi_ops = 0;
    uint32_t even_interior_ell_gt_n = 0, ell_gt_n_cells = 0;
    // diagnostics
    uint32_t diag_pairs = 0, diag_commutes = 0, diag_eta_zero = 0, diag_cusp_supported = 0;
    uint32_t diag_implication_ok = 0;
    uint32_t walk_p2_pinned = 0;
};

struct CellRecord {
    uint32_t n, ell;
    uint32_t dimH1, dim_int, rank_r1, dimH1_bd;
    bool eis_def, in_int, harder_app, kummer, consistent, split_direct, dickson;
    uint32_t joint_dim, rank_r1E;
    std::string verdict;
    std::string note;
};

std::unique_ptr<HeckeBoundary> build_cell(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                                          const LocalSystemSimplicialComplex& lssc, const EquivariantCohomologyOnK& coh,
                                          uint32_t ell, uint32_t n, const std::vector<std::pair<uint64_t, uint32_t>>& ops,
                                          bool diagnostics) {
    (void)lssc; (void)ell; (void)n;
    HeckeBoundaryOptions o;
    o.operators = ops;
    o.run_diagnostics = diagnostics;
    auto hb = std::make_unique<HeckeBoundary>(&coh, K, model, o);
    hb->compute();
    return hb;
}

void run_cell(const SimplicialComplex& K, const SL2ZTessellationTables& model, uint32_t n, uint32_t ell,
              const std::vector<std::pair<uint64_t, uint32_t>>& ops, bool diagnostics, CheckLog& log, Tally& tally,
              std::vector<CellRecord>& records, std::vector<std::string>& diag_lines,
              std::unique_ptr<HeckeBoundary>* keep = nullptr) {
    const std::string where = "[n=" + std::to_string(n) + " ell=" + std::to_string(ell) + "]";
    const bool small_ell = (ell == 2 || ell == 3);
    ++tally.cells;
    if (small_ell) ++tally.cells_small_ell;

    LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, ell), n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    std::unique_ptr<HeckeBoundary> hb = build_cell(K, model, lssc, coh, ell, n, ops, diagnostics);
    const HeckeBoundary& H = *hb;
    const uint32_t d1 = H.dim_H_bulk(1), k = H.dim_interior(), rr1 = H.rank_r(1);
    const engine::ModCtx<uint32_t> ctx(ell);

    CellRecord rec{ n, ell, d1, k, rr1, H.dim_H_bd(1), H.eisenstein_defined(), H.eisenstein_in_interior(),
                    H.harder_applicable(), H.kummer(), H.harder_consistent(), H.harder_split_direct(),
                    own_is_dickson(n, ell), H.joint_eigenspace_dim(), H.eisenstein_rank_r1(),
                    HeckeBoundary::harder_verdict_name(H.harder_verdict()), "" };
    bool kummer_known = false;
    const bool kummer_cell = hecke_ext::pinned_kummer_divides(n + 2, ell, kummer_known) && kummer_known && (n + 1 < ell);

    // ---- G1: the certificate ---------------------------------------------
    {
        std::vector<std::string> f;
        const uint32_t bad = H.certificate().verify(f);
        log.check(bad == 0, where + " certificate.verify() = " + std::to_string(bad) + (f.empty() ? "" : " first: " + f.front()));
    }

    // ---- G2: own r^q, own kernel, the dimension bookkeeping ----------------
    for (uint32_t q = 0; q < 2; ++q) {
        const FlatMatrix own = own_restriction(H, q);
        log.check(flat_equal(own, flat_of(H.r(q))), where + " own r^" + std::to_string(q) + " != the object's r^" + std::to_string(q));
        log.check(flat_rank(own) == H.rank_r(q), where + " rank r^" + std::to_string(q));
    }
    log.check(H.rank_r(0) == H.dim_H_bulk(0), where + " r^0 is not injective (lem:conerelative(4))");
    log.check(k + rr1 == d1, where + " dim H^1_! + rank r^1 != dim H^1 (eq:dimbook)");
    {
        const FlatMatrix own_r1 = own_restriction(H, 1);
        const FlatMatrix B = flat_of(H.interior_basis());
        if (d1 > 0) {
            log.check((own_r1.get_rows() == 0) || (own_r1 * B).is_zero(), where + " own r^1 * B != 0");
            FlatMatrix Kown = FlatMatrix(d1, ell);
            if (own_r1.get_rows() > 0) {
                const std::vector<FlatMatrix> kv = own_r1.kernel();
                Kown = FlatMatrix::CreateZero(d1, 0, ell);
                for (const auto& v : kv) Kown = hcat(Kown, v);
            }
            log.check(Kown.get_cols() == k, where + " own ker r^1 has dimension " + std::to_string(Kown.get_cols()) + " != " + std::to_string(k));
            if (k > 0) log.check(flat_rank(hcat(Kown, B)) == k, where + " own ker r^1 != span of the object's interior basis");
        } else {
            log.check(k == 0 && B.get_cols() == 0, where + " dim H^1 = 0 but the interior basis is not empty");
        }
    }

    // ---- G3: T_p-stability, re-multiplied on FlatMatrix --------------------
    for (size_t i = 0; i < H.num_pipelines(); ++i) {
        const HeckeBoundary::Pipeline& pl = H.pipeline(i);
        const std::string tag = where + " (p=" + std::to_string(pl.p) + ",nu=" + std::to_string(pl.nu) + ")";
        if (!pl.interior_stable) {
            if (small_ell) ++tally.unstable_ops_small_ell;
            log.check(small_ell, tag + " H^1_! is not T_p-stable at ell >= 5");
            continue;
        }
        ++tally.stable_ops;
        const FlatMatrix A = flat_of(pl.A_bulk[1]), B = flat_of(H.interior_basis()), Ai = flat_of(pl.A_interior);
        log.check(Ai.get_rows() == k && Ai.get_cols() == k, tag + " A_! shape");
        if (k > 0) log.check(flat_equal(A * B, B * Ai), tag + " A B != B A_! on FlatMatrix");
        log.check(flat_equal(flat_of(H.interior_operator(pl.p, pl.nu)), Ai), tag + " interior_operator accessor");
    }

    // ---- G4: the joint Eisenstein eigenspace, re-derived -------------------
    if (d1 > 0) {
        FlatMatrix stacked = FlatMatrix::CreateZero(0, d1, ell);
        uint32_t used = 0, jdim = d1;
        for (uint64_t p : { 2ull, 3ull, 5ull }) {
            if (p == ell || !H.has_pipeline(p, 1)) continue;
            if (used >= 2 && jdim == 1) break;
            const FlatMatrix A = flat_of(H.pipeline(p, 1).A_bulk[1]);
            stacked = vcat(stacked, A - FlatMatrix::CreateScalar(d1, own_eis(ell, p, n), ell));
            ++used;
            jdim = d1 - flat_rank(stacked);
        }
        const uint32_t j_own = (used == 0) ? d1 : jdim;
        log.check(j_own == H.joint_eigenspace_dim(), where + " own joint eigenspace dim != the object's");
        log.check(H.eisenstein_defined() == (j_own == 1), where + " Eisenstein 'defined' flag");
        const FlatMatrix E = flat_of(H.eisenstein_space());
        const FlatMatrix r1 = flat_of(H.r(1));
        if (used > 0 && j_own > 0) {
            // Own kernel, same span as the stored E.
            const std::vector<FlatMatrix> kv = stacked.kernel();
            FlatMatrix Kown = FlatMatrix::CreateZero(d1, 0, ell);
            for (const auto& v : kv) Kown = hcat(Kown, v);
            log.check(Kown.get_cols() == j_own, where + " own joint kernel size");
            log.check(flat_rank(hcat(Kown, E)) == j_own, where + " own joint eigenspace != the object's space");
            const uint32_t rkE = (r1.get_rows() == 0) ? 0 : flat_rank(r1 * E);
            log.check(rkE == H.eisenstein_rank_r1(), where + " own rank(r^1 E) != the object's");
            log.check(H.eisenstein_in_interior() == (rkE == 0), where + " in_interior flag != (r^1 E == 0)");
        }
        if (H.eisenstein_defined()) {
            ++tally.eis_defined;
            const FlatMatrix v = flat_of(H.eisenstein_vector());
            log.check(flat_equal(v, E), where + " the Eisenstein vector is not the stored line");
            // Route (b): where rank r^1 = 1 and n < ell (H^1(bd) a line carrying
            // chi_0), the line's image is the boundary-visible class iff r^1 v != 0,
            // and then A_bd^{(1)} acts on it by 1 + p^{n+1}.
            if (rr1 == 1 && n < ell && !H.eisenstein_in_interior() && !small_ell) {
                for (size_t i = 0; i < H.num_pipelines(); ++i) {
                    const HeckeBoundary::Pipeline& pl = H.pipeline(i);
                    if (pl.nu != 1 || pl.p == ell) continue;
                    const FlatMatrix w = r1 * v;
                    log.check(flat_equal(flat_of(pl.A_bd[1]) * w, w * own_eis(ell, pl.p, n)),
                              where + " r^1(Eis) is not the 1 + p^{n+1} class on H^1(bd), p=" + std::to_string(pl.p));
                }
            }
        } else {
            if (rec.dickson) ++tally.eis_undefined_dickson;
            else if (kummer_cell) ++tally.eis_undefined_kummer;
            else ++tally.eis_undefined_other;
            // Off the Dickson and Kummer loci (with k in the pinned range) the
            // joint eigenspace must be a line.
            if (!small_ell && !rec.dickson && n > 0 && kummer_known && !kummer_cell)
                log.check(false, where + " the Eisenstein line is undefined off the Dickson / Kummer loci (joint dim " +
                                     std::to_string(H.joint_eigenspace_dim()) + ")");
            if (kummer_cell)
                rec.note += " [Kummer: joint eigenspace dim " + std::to_string(H.joint_eigenspace_dim()) + ", rank(r^1 E) = " +
                            std::to_string(H.eisenstein_rank_r1()) + "]";
        }
    }

    // ---- G5: Harder --------------------------------------------------------
    {
        using HV = HeckeBoundaryCertificate::HarderVerdict;
        bool known = false;
        const bool div = hecke_ext::pinned_kummer_divides(n + 2, ell, known);
        const bool expect_app = known && (n + 1 < ell) && H.joint_eigenspace_dim() >= 1 && d1 > 0;
        log.check(H.harder_applicable() == expect_app, where + " harder_applicable flag");
        if (H.harder_applicable()) {
            ++tally.harder_applicable;
            log.check(H.kummer() == div, where + " kummer flag != pinned divisibility");
            log.check(H.kummer() == in_kummer_list(n + 2, ell), where + " pinned divisibility disagrees with [E] Claim 1.8's list");
            const HV v = H.harder_verdict();
            if (!small_ell) log.check(v != HV::kRefuted, where + " HARDER REFUTED: (ell | num(B_k/k)) = " + (H.kummer() ? "yes" : "no") +
                                                          " but rank(r^1 E) = " + std::to_string(H.eisenstein_rank_r1()) + " on a joint eigenspace of dim " +
                                                          std::to_string(H.joint_eigenspace_dim()));
            if (v == HV::kConsistent) ++tally.harder_consistent;
            if (v == HV::kInconclusive) { ++tally.harder_inconclusive; rec.note += " [Harder inconclusive mod ell]"; }
            if (H.kummer()) {
                ++tally.kummer_cells;
                if (k == d1) ++tally.kummer_interior_all;
                rec.note += (k == d1) ? " [Kummer: H^1_! = H^1]" : " [Kummer: H^1_! != H^1, rank r^1 = " + std::to_string(rr1) + "]";
            }
        }
        if ((n == 10 && ell == 5) || (n == 10 && ell == 7)) {
            rec.note += " [RECORDED (XA m5): outside n < ell-1; joint dim " + std::to_string(H.joint_eigenspace_dim()) +
                        ", rank(r^1 E) = " + std::to_string(H.eisenstein_rank_r1()) + "]";
        }
    }

    // ---- G6: the split ---------------------------------------------------------
    for (size_t i = 0; i < H.num_pipelines(); ++i) {
        const HeckeBoundary::Pipeline& pl = H.pipeline(i);
        const std::string tag = where + " (p=" + std::to_string(pl.p) + ",nu=" + std::to_string(pl.nu) + ")";
        if (!small_ell) {
            log.check(pl.quotient_stable, tag + " im r^1 is not A_bd-stable");
            log.check(pl.charpoly_product_holds, tag + " charpoly(A) != charpoly(A_!) * charpoly(A_bd | im r^1)");
            // Own recomputation of the product identity.
            if (pl.interior_stable && pl.quotient_stable) {
                const hecke::FpPoly lhs = hecke::CharPolyBerkowitz<uint32_t>(pl.A_bulk[1]);
                const hecke::FpPoly rhs = (pl.A_interior.rows ? hecke::CharPolyBerkowitz<uint32_t>(pl.A_interior) : hecke::FpPoly::one(ell)) *
                                          (pl.A_quotient_bd.rows ? hecke::CharPolyBerkowitz<uint32_t>(pl.A_quotient_bd) : hecke::FpPoly::one(ell));
                log.check(lhs == rhs, tag + " own charpoly product identity");
            }
            if (rr1 == 1 && n < ell && pl.nu == 1 && pl.p != ell) {
                log.check(pl.A_quotient_bd.rows == 1 && pl.A_quotient_bd.data[0][0] == own_eis(ell, pl.p, n),
                          tag + " the quotient H^1 / H^1_! does not carry chi_0(p) = 1 + p^{n+1}");
                ++tally.chi0_quotient;
            }
        }
        ++tally.charpoly_ops;
        if (pl.charpoly_product_holds) ++tally.charpoly_product;
        if (!pl.chi.empty() && pl.quotient_stable) {
            ++tally.quotient_chi_ops;
            if (pl.quotient_charpoly_divides_chi) ++tally.quotient_divides_chi;
            else rec.note += " [quotient charpoly does not divide prod(X - chi_b), p=" + std::to_string(pl.p) + "]";
        }
    }
    if (H.harder_split_direct()) {
        ++tally.split_direct_cells;
        log.check(k + 1 == d1 && rr1 == 1, where + " split_direct but dims disagree");
        const FlatMatrix BV = hcat(flat_of(H.interior_basis()), flat_of(H.eisenstein_vector()));
        log.check(flat_rank(BV) == d1, where + " H^1_! (+) Eis does not span H^1");
    }

    // ---- G7: the classical dimensions -------------------------------------
    if (n == 0) {
        log.check(d1 == 0 && k == 0 && rr1 == 0, where + " n = 0: dim H^1 != 0");
    } else if (small_ell) {
        ++tally.recorded_dim_cells;
    } else {
        const uint32_t dk = own_dim_Mk(n + 2);
        const uint32_t two_dim_S = 2 * (dk - 1);
        bool known = false;
        const bool div = hecke_ext::pinned_kummer_divides(n + 2, ell, known);
        if (rec.dickson) {
            ++tally.dickson_cells;
            if (k >= two_dim_S) ++tally.dickson_ge;
            rec.note += " [Dickson: dim H^1_! = " + std::to_string(k) + " vs 2 dim S_k = " + std::to_string(two_dim_S) + "]";
        } else if (n + 1 < ell && known) {
            // ell > n + 1, k in the pinned range: the classical count, at the
            // Kummer cells included (where [E] sec:tests' "H^1_! = H^1" would
            // instead need rank r^1 = 0 -- measured: rank r^1 = 1 on all ten).
            ++tally.classical_cells;
            const bool ok = (k == two_dim_S) && (rr1 == 1);
            if (ok) ++tally.classical_match;
            log.check(ok, where + " classical: dim H^1_! = " + std::to_string(k) + " != 2 dim S_k = " + std::to_string(two_dim_S) +
                              " or rank r^1 = " + std::to_string(rr1) + " != 1" + (div ? " (Kummer cell)" : ""));
        } else {
            ++tally.recorded_dim_cells;
            if (n + 1 >= ell)
                rec.note += " [ell <= n+1: dim H^1_! = " + std::to_string(k) + ", 2 dim S_k = " + std::to_string(two_dim_S) +
                            ", rank r^1 = " + std::to_string(rr1) + "]";
        }
        if (ell > n) {
            ++tally.ell_gt_n_cells;
            if (k % 2 == 0) ++tally.even_interior_ell_gt_n;
        }
    }

    // ---- G8: the diagnostics --------------------------------------------------
    if (diagnostics) {
        for (size_t i = 0; i < H.num_pipelines(); ++i) {
            const HeckeBoundary::Pipeline& pl = H.pipeline(i);
            log.check(pl.diagnostics_run, where + " diagnostics not run");
            if (!pl.diagnostics_run) continue;
            ++tally.diag_pairs;
            const bool cc = pl.cochain_commutes.holds(), ez = pl.walk.eta_zero();
            if (cc) ++tally.diag_commutes;
            if (ez) ++tally.diag_eta_zero;
            if (pl.walk.cusp_supported()) ++tally.diag_cusp_supported;
            // The one theorem: eta_zero => cochain_commutes ([C, rmk:conestrict](2)).
            log.check(!ez || cc, where + " eta_zero = true but cochain_commutes = false (rmk:conestrict(2))");
            if (!ez || cc) ++tally.diag_implication_ok;
            const uint32_t m = pl.walk.num_cosets;
            log.check(pl.walk.slots == m * (4 + 3), where + " walk slot count");
            log.check(pl.cochain_commutes.vectors > 0 || (H.dim_H_bulk(0) == 0 && d1 == 0), where + " cochain identity compared on no vector");
            if (pl.p == 2 && pl.nu == 1) {
                // The recorded shape of the CANONICAL tau's p = 2 table
                // (docs/implementation/tau_table_p2.txt, GeometricNearest since
                // the 2026-09-16 iota-symmetric re-baseline): 10 of the 12 vertex
                // slots (every j at v0 and v3, j in {2, 3} at v1 and v2) and 5 of
                // the 9 edge slots (lines 34, 35, 40, 49, 50: the interior edges
                // {v0,v4} and {v4,v6}) leave the boundary. [C, rmk:conestrict](1)
                // still quotes 8 / 2 with "the j = 1 slots all stay on X_bd",
                // which describes the pre-re-baseline table -- a stale count,
                // recorded for lane C; its conclusion (tau is not a map of
                // pairs) holds a fortiori. A regression gate on the
                // construction, not a branch on the verdict.
                const bool pinned = pl.walk.off_boundary_per_degree.size() == 2 && pl.walk.off_boundary_per_degree[0] == 10 &&
                                    pl.walk.off_boundary_per_degree[1] == 5;
                log.check(pinned, where + " the p = 2 cusp-support walk does not reproduce tau_table_p2.txt's 10 / 5 interior slots");
                if (pinned) ++tally.walk_p2_pinned;
            }
            std::ostringstream oss;
            oss << "  DIAG " << where << " p=" << pl.p << " nu=" << pl.nu << ": cochain_commutes=" << (cc ? "true" : "false") << " ("
                << pl.cochain_commutes.failing << "/" << pl.cochain_commutes.vectors << " vectors fail) eta_zero=" << (ez ? "true" : "false")
                << " cusp_supported=" << (pl.walk.cusp_supported() ? "true" : "false") << " (slots " << pl.walk.slots << ", off-boundary "
                << pl.walk.slots_off_boundary << " = q0:" << pl.walk.off_boundary_per_degree[0] << " + q1:" << pl.walk.off_boundary_per_degree[1]
                << ", differ " << pl.walk.slots_differ << " = q0:" << pl.walk.differ_per_degree[0] << " + q1:" << pl.walk.differ_per_degree[1] << ")";
            diag_lines.push_back(oss.str());
        }
    }

    // ---- G11: the JSON ------------------------------------------------------
    {
        const std::string js = H.to_json();
        log.check(js.find("\"schema\":\"hecke_boundary_parabolic/v1\"") != std::string::npos, where + " schema string");
        log.check(js.find("\"bulk\":{\"model\":\"sl2z_7vertex\",\"q_max\":2,\"truncated\":false,\"q_max_source\":\"derived\",\"degrees\":[0,1,2]") != std::string::npos,
                  where + " bulk per-complex record fields");
        log.check(js.find("\"boundary\":{\"boundary_model\":\"circle\"") != std::string::npos &&
                      js.find("\"q_max\":1,\"truncated\":false,\"q_max_source\":\"derived\",\"degrees\":[0,1]") != std::string::npos,
                  where + " boundary per-complex record fields");
        log.check(js.find(std::string("\"boundary_is_group_cohomology\":") + (ell != 2 ? "true" : "false")) != std::string::npos,
                  where + " boundary_is_group_cohomology");
        log.check(js.find("\"certificate\":{\"verify\":0}") != std::string::npos, where + " certificate in JSON");
    }

    records.push_back(rec);
    if (keep) *keep = std::move(hb);
}

// =========================================================================
// Section 3 -- the tamper arm on one certificate.
// =========================================================================

// verify() must reject a tampered copy WITHOUT throwing (a thrown verify is
// a defect of the certificate, counted as a failure here).
bool rejects(const HeckeBoundaryCertificate& c, CheckLog& log, const std::string& what) {
    try {
        return c.verify() != 0;
    } catch (const std::exception& e) {
        log.fail(what + ": verify() threw instead of rejecting: " + e.what());
        return true;
    }
}

void tamper_arm(const HeckeBoundary& H, CheckLog& log) {
    const HeckeBoundaryCertificate& c0 = H.certificate();
    const std::string where = "[tamper n=" + std::to_string(c0.n) + " ell=" + std::to_string(c0.ell) + "]";
    log.check(c0.verify() == 0, where + " the untampered certificate fails");
    const uint32_t ell = c0.ell;
    // T0: a wrong entry in the joint eigenspace.
    if (c0.joint_eigenspace_dim > 0 && c0.eisenstein_space.rows > 0) {
        HeckeBoundaryCertificate c = c0;
        c.eisenstein_space.data[0][0] = (c.eisenstein_space.data[0][0] + 1) % ell;
        if (c.eisenstein_defined) c.eisenstein_vector = c.eisenstein_space;
        log.check(rejects(c, log, where + " T0"), where + " T0: a tampered joint eigenspace passes");
    }
    // T1: an interior vector outside ker r^1 (the Eisenstein vector, which restricts non-trivially here).
    if (c0.eisenstein_defined && !c0.eisenstein_in_interior && c0.interior_basis.cols > 0) {
        HeckeBoundaryCertificate c = c0;
        for (uint32_t i = 0; i < c.interior_basis.rows; ++i) c.interior_basis.data[i][0] = c.eisenstein_vector.data[i][0];
        log.check(rejects(c, log, where + " T1"), where + " T1: a non-interior vector in the interior basis passes");
    }
    // T2: a wrong A_!.
    if (!c0.ops.empty() && c0.ops[0].stable && c0.ops[0].A_interior.rows > 0) {
        HeckeBoundaryCertificate c = c0;
        c.ops[0].A_interior.data[0][0] = (c.ops[0].A_interior.data[0][0] + 1) % ell;
        log.check(rejects(c, log, where + " T2"), where + " T2: a wrong A_! passes");
    }
    // T3: the Kummer flag flipped.
    if (c0.harder_applicable) {
        HeckeBoundaryCertificate c = c0;
        c.kummer = !c.kummer;
        log.check(rejects(c, log, where + " T3"), where + " T3: a flipped Kummer flag passes");
    }
    // T4: the in_interior flag flipped.
    if (c0.joint_eigenspace_dim > 0) {
        HeckeBoundaryCertificate c = c0;
        c.eisenstein_in_interior = !c.eisenstein_in_interior;
        log.check(rejects(c, log, where + " T4"), where + " T4: a flipped in_interior flag passes");
    }
    // T5: a zero Eisenstein vector.
    if (c0.eisenstein_defined) {
        HeckeBoundaryCertificate c = c0;
        for (uint32_t i = 0; i < c.eisenstein_vector.rows; ++i) c.eisenstein_vector.data[i][0] = 0;
        log.check(rejects(c, log, where + " T5"), where + " T5: a zero Eisenstein vector passes");
    }
    // T6: the split flag set wrongly.
    {
        HeckeBoundaryCertificate c = c0;
        c.harder_split_direct = !c.harder_split_direct;
        log.check(rejects(c, log, where + " T6"), where + " T6: a wrong split flag passes");
    }
    // T7: a wrong dimension.
    {
        HeckeBoundaryCertificate c = c0;
        c.dim_H1_K += 1;
        log.check(rejects(c, log, where + " T7"), where + " T7: a wrong dim H^1 passes");
    }
    // T8: a wrong Harder verdict.
    if (c0.harder_applicable) {
        HeckeBoundaryCertificate c = c0;
        c.harder_verdict = (c.harder_verdict == HeckeBoundaryCertificate::HarderVerdict::kConsistent)
                               ? HeckeBoundaryCertificate::HarderVerdict::kInconclusive
                               : HeckeBoundaryCertificate::HarderVerdict::kConsistent;
        log.check(rejects(c, log, where + " T8"), where + " T8: a wrong Harder verdict passes");
    }
}

// =========================================================================
// Section 4 -- the collector digest (in-process and in a child process).
// =========================================================================

const std::vector<std::pair<uint32_t, uint32_t>>& collect_grid() {
    static const std::vector<std::pair<uint32_t, uint32_t>> g = { { 2, 5 }, { 10, 691 }, { 16, 7 }, { 12, 134217757u } };
    return g;
}

uint64_t collect_digest(const SimplicialComplex& K, const SL2ZTessellationTables& model) {
    uint64_t h = engine::kFnvOffset64;
    for (const auto& cell : collect_grid()) {
        const uint32_t n = cell.first, ell = cell.second;
        LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt<uint32_t>(K, n, ell), n + 1, ell);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        HeckeBoundaryOptions o;
        o.operators = { { 2, 1 }, { 3, 1 }, { 2, 2 } };
        o.run_diagnostics = false;
        HeckeBoundary hb(&coh, K, model, o);
        hb.compute();
        h = engine::FnvU64(h, hb.golden_digest());
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
    const std::string cmd = "ENGINE_THREADS=" + std::to_string(threads) + " HECKE_BOUNDARY_PARABOLIC_CHILD=1 \"" + exe +
                            "\" HeckeBoundaryParabolicValidation 2>/dev/null";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) return false;
    std::string out;
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) out += line;
    const int rc = ::pclose(f);
    if (rc != 0) return false;
    const std::string key = "BOUNDARY_COLLECT digest=0x";
    const size_t pos = out.find(key);
    if (pos == std::string::npos) return false;
    digest_out = std::strtoull(out.c_str() + pos + key.size(), nullptr, 16);
    return true;
}

} // namespace

uint32_t HeckeBoundaryParabolicValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    CheckLog log;
    Tally tally;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);

        // The child mode: the collector digest only.
        if (const char* child = std::getenv("HECKE_BOUNDARY_PARABOLIC_CHILD"); child && *child) {
            const uint64_t d = collect_digest(K, model);
            std::cout << "BOUNDARY_COLLECT digest=0x" << std::hex << d << std::dec << std::endl;
            return 0;
        }

        std::cout << "HeckeBoundaryParabolicValidation: H^1_! = ker r^1, the Eisenstein line, Harder's criterion, the Harder split "
                     "(HeckeExt/HeckeBoundary) and the diagnostics cochain_commutes / eta_zero (Hecke/HeckeBoundaryDiagnostics). "
                     "GATING = G1..G11 of the header; the diagnostics' verdicts and the ell in {2, 3} behaviour are RECORDED." << std::endl;

        // ---- Section 0: the pinned Bernoulli table against [E] Claim 1.8 ------
        {
            bool known = false;
            for (const auto& kl : kummer_list())
                log.check(hecke_ext::pinned_kummer_divides(kl.first, kl.second, known) && known,
                          "pinned table: (k, ell) = (" + std::to_string(kl.first) + ", " + std::to_string(kl.second) + ") is not Kummer");
            log.check(hecke_ext::pinned_bernoulli_numerator(20, known) == -174611 && 174611 == 283 * 617, "pinned: num(B_20/20)");
            log.check(hecke_ext::pinned_bernoulli_numerator(22, known) == 77683 && 77683 == 131 * 593, "pinned: num(B_22/22)");
            log.check(hecke_ext::pinned_bernoulli_numerator(24, known) == -236364091 && 236364091 == 103ll * 2294797ll, "pinned: num(B_24/24)");
            for (uint64_t q : { 691ull, 3617ull, 43867ull, 657931ull, 283ull, 617ull, 131ull, 593ull, 103ull, 2294797ull })
                log.check(own_is_prime(q), "pinned: " + std::to_string(q) + " is not prime");
            // Non-Kummer controls.
            for (const auto& kl : std::vector<std::pair<uint32_t, uint32_t>>{ { 12, 13 }, { 12, 17 }, { 12, 101 }, { 12, 134217757u }, { 16, 691 },
                                                                              { 18, 691 }, { 22, 11 }, { 24, 5 }, { 26, 13 }, { 4, 5 } })
                log.check(!hecke_ext::pinned_kummer_divides(kl.first, kl.second, known) && known,
                          "pinned table: (" + std::to_string(kl.first) + ", " + std::to_string(kl.second) + ") wrongly Kummer");
            log.check(hecke_ext::pinned_bernoulli_numerator(28, known) == 0 && !known, "pinned table: k = 28 must be unknown");
        }

        // ---- Section 1: the grid ---------------------------------------------
        const std::vector<uint32_t> ells = { 2, 3, 5, 7, 691, 134217757u, 2147483647u };
        const std::vector<uint32_t> ns = { 0, 2, 4, 6, 8, 10, 12, 14, 16 };
        const std::vector<std::pair<uint64_t, uint32_t>> ops_full = { { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 2, 2 } };
        const std::vector<std::pair<uint64_t, uint32_t>> ops_kummer = { { 2, 1 }, { 3, 1 } };
        std::vector<CellRecord> records;
        std::vector<std::string> diag_lines;
        std::unique_ptr<HeckeBoundary> keep_10_691, keep_10_13;
        for (uint32_t ell : ells)
            for (uint32_t n : ns) {
                try {
                    run_cell(K, model, n, ell, ops_full, /*diagnostics=*/true, log, tally, records, diag_lines,
                             (n == 10 && ell == 691) ? &keep_10_691 : nullptr);
                } catch (const std::exception& e) {
                    log.fail("[n=" + std::to_string(n) + " ell=" + std::to_string(ell) + "] EXCEPTION: " + e.what());
                }
            }
        // The Dickson loci of obl:iota(v)'s record and a non-Kummer control at ell = 13.
        for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{ { 20, 5 }, { 24, 5 }, { 10, 13 } }) {
            try {
                run_cell(K, model, cell.first, cell.second, ops_kummer, /*diagnostics=*/false, log, tally, records, diag_lines,
                         (cell.first == 10 && cell.second == 13) ? &keep_10_13 : nullptr);
            } catch (const std::exception& e) {
                log.fail("[n=" + std::to_string(cell.first) + " ell=" + std::to_string(cell.second) + "] EXCEPTION: " + e.what());
            }
        }
        // ---- Section 2: the Kummer list ([E] Claim 1.8), all n < ell - 1 -------
        for (const auto& kl : kummer_list()) {
            if (kl.first == 12) continue; // (10, 691) is in the grid
            try {
                run_cell(K, model, kl.first - 2, kl.second, ops_kummer, /*diagnostics=*/false, log, tally, records, diag_lines);
            } catch (const std::exception& e) {
                log.fail("[n=" + std::to_string(kl.first - 2) + " ell=" + std::to_string(kl.second) + "] EXCEPTION: " + e.what());
            }
        }
        log.check(tally.kummer_cells == 10, "not all ten Kummer cells were asserted (" + std::to_string(tally.kummer_cells) + ")");
        log.check(tally.harder_applicable > tally.kummer_cells, "no non-Kummer cell exercised the 'only if' direction");

        // ---- Section 3: the tamper arm ------------------------------------------
        if (keep_10_691) tamper_arm(*keep_10_691, log); else log.fail("no (10, 691) object kept for the tamper arm");
        if (keep_10_13) tamper_arm(*keep_10_13, log); else log.fail("no (10, 13) object kept for the tamper arm");

        // ---- Section 4: the collectors ------------------------------------------
        {
            if (keep_10_691) {
                const auto objs = keep_10_691->golden_objects(2, 1);
                log.check(objs.size() == 15, "golden_objects: 15 records expected, got " + std::to_string(objs.size()));
                const std::vector<std::string> names = { "dim_H0_bd", "dim_H1_bd", "rank_r^0", "rank_r^1", "charpoly_A_bd^0", "charpoly_A_bd^1",
                                                         "chi_multiset", "dim_H1_interior", "eis_joint_dim", "eis_rank_r1", "r^0", "r^1",
                                                         "A_bd^0", "A_bd^1", "interior_basis" };
                for (size_t i = 0; i < names.size() && i < objs.size(); ++i) {
                    log.check(objs[i].object == names[i], "golden_objects: record " + std::to_string(i) + " is " + objs[i].object);
                    log.check(objs[i].tier == (i < 10 ? 'A' : 'B'), "golden_objects: tier of " + objs[i].object);
                }
            }
            const uint64_t d_a = collect_digest(K, model);
            const uint64_t d_b = collect_digest(K, model);
            log.check(d_a == d_b, "collector digest differs between two in-process builds");
            const std::string exe = find_self_exe();
            if (exe.empty()) {
                std::cout << "  collector child arm SKIPPED: /proc/self/exe unavailable" << std::endl;
            } else {
                for (uint32_t t : { 1u, 16u }) {
                    uint64_t d = 0;
                    const bool ok = child_digest(exe, t, d);
                    log.check(ok, "collector child arm: the ENGINE_THREADS=" + std::to_string(t) + " child produced no digest");
                    if (ok) log.check(d == d_a, "collector digest differs under ENGINE_THREADS=" + std::to_string(t));
                }
            }
            std::cout << "  collector digest 0x" << std::hex << d_a << std::dec << " (two in-process builds and the ENGINE_THREADS=1 / 16 children)" << std::endl;
        }

        // ---- Report ---------------------------------------------------------------
        std::cout << "HeckeBoundaryParabolicValidation: per cell -- dim H^1, dim H^1_!, rank r^1, dim H^1(bd); E = the joint 1+p^{n+1} "
                     "eigenspace: line? / dim / rank(r^1 E) / in H^1_!; Harder applicable / Kummer / verdict; split_direct; D = Dickson" << std::endl;
        std::cout << "    n         ell  dimH1  H1_!  rk_r1  H1bd | line jdim rkE in_int | harder kummer verdict        | split | note" << std::endl;
        for (const auto& r : records) {
            std::cout << "  " << std::setw(3) << r.n << " " << std::setw(11) << r.ell << "  " << std::setw(5) << r.dimH1 << " " << std::setw(5) << r.dim_int
                      << "  " << std::setw(5) << r.rank_r1 << " " << std::setw(5) << r.dimH1_bd << " |  " << (r.eis_def ? "y" : "n") << "   " << std::setw(3)
                      << r.joint_dim << "  " << std::setw(2) << r.rank_r1E << "    " << (r.joint_dim ? (r.in_int ? "y" : "n") : "-") << "    |   "
                      << (r.harder_app ? "y" : "n") << "      " << (r.harder_app ? (r.kummer ? "y" : "n") : "-") << "    " << std::left << std::setw(15)
                      << (r.harder_app ? r.verdict : std::string("-")) << std::right << "|   " << (r.split_direct ? "y" : "n") << "   | "
                      << (r.dickson ? "D" : "") << r.note << std::endl;
        }
        for (const auto& l : diag_lines) std::cout << l << std::endl;
        std::cout << "BOUNDARY_DIAG cells=" << tally.diag_pairs << " cochain_commutes=" << tally.diag_commutes << "/" << tally.diag_pairs
                  << " eta_zero=" << tally.diag_eta_zero << "/" << tally.diag_pairs << " cusp_supported=" << tally.diag_cusp_supported << "/"
                  << tally.diag_pairs << " implication_ok=" << tally.diag_implication_ok << "/" << tally.diag_pairs << " walk_p2_pinned="
                  << tally.walk_p2_pinned << std::endl;
    } catch (const std::exception& e) {
        std::cout << "HeckeBoundaryParabolicValidation: EXCEPTION during setup: " << e.what() << std::endl;
        return 1000;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "BOUNDARY_PARABOLIC cells=" << tally.cells << " checks=" << log.checks << " failures=" << log.failures
              << " stable_ops=" << tally.stable_ops << " unstable_ops_ell23=" << tally.unstable_ops_small_ell
              << " eis_defined=" << tally.eis_defined << " eis_undefined_dickson=" << tally.eis_undefined_dickson
              << " eis_undefined_kummer=" << tally.eis_undefined_kummer << " eis_undefined_other=" << tally.eis_undefined_other
              << " harder_applicable=" << tally.harder_applicable << " harder_consistent=" << tally.harder_consistent
              << " harder_inconclusive=" << tally.harder_inconclusive << " kummer_cells=" << tally.kummer_cells
              << " kummer_interior_all=" << tally.kummer_interior_all << " classical=" << tally.classical_match << "/" << tally.classical_cells
              << " dickson_ge=" << tally.dickson_ge << "/" << tally.dickson_cells << " recorded_dims=" << tally.recorded_dim_cells
              << " split_direct=" << tally.split_direct_cells << " charpoly_product=" << tally.charpoly_product << "/" << tally.charpoly_ops
              << " chi0_quotient=" << tally.chi0_quotient << " quotient_divides_chi=" << tally.quotient_divides_chi << "/" << tally.quotient_chi_ops
              << " even_interior_ell_gt_n=" << tally.even_interior_ell_gt_n << "/" << tally.ell_gt_n_cells
              << " ell23_cells=" << tally.cells_small_ell << " (" << std::fixed << std::setprecision(2) << secs << " s)" << std::endl;
    return log.failures;
}
