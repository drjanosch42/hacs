// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeReductionCoverageSlowValidation.h"
#include "HeckeReductionConeValidation.h"   // (slot-9 WP-4) the E8 item 4b coverage

// hecke-extensions slot 8, WP-4 (E8 item 4a): the reduction and Bockstein arms
// -- see the header for the arm list.  The file name ends in `Validation` so the
// include guards and the raw-mod census treat it as the test harness it is.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeReduction.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/ModCtx.h"

namespace {

using engine::DenseBuf;
using engine::ModCtx;

uint64_t* g_checks = nullptr;
uint32_t* g_fail = nullptr;
uint32_t g_printed = 0;

void check(bool ok, const std::string& what) {
    ++*g_checks;
    if (!ok) {
        ++*g_fail;
        if (g_printed++ < 60) std::cout << "  [FAIL] " << what << std::endl;
    }
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "(";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ")";
    return o.str();
}

// ---- one cell: the F_ell side and the Z/ell^K sides -------------------------------
template<class Elt>
struct Side {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh;
    std::vector<std::unique_ptr<hecke::HeckeCosetTablesT<Elt>>> cosets;
    std::vector<std::unique_ptr<hecke::HeckeTauTableT<Elt>>> tau;
    std::vector<std::unique_ptr<hecke::HeckeOperatorOnKT<Elt>>> op;   // one per prime, in order
    uint32_t k = 1;
};

template<class Elt>
void add_ops(Side<Elt>& s, uint32_t n, uint64_t ell, const std::vector<uint64_t>& primes) {
    for (uint64_t p : primes) {
        s.cosets.push_back(std::make_unique<hecke::HeckeCosetTablesT<Elt>>(p, 1, static_cast<Elt>(ell), n));
        s.tau.push_back(std::make_unique<hecke::HeckeTauTableT<Elt>>(s.K, *s.model, *s.cosets.back(),
                                                                       static_cast<Elt>(ell), n));
        s.op.push_back(std::make_unique<hecke::HeckeOperatorOnKT<Elt>>(s.coh.get(), s.cosets.back().get(),
                                                                         s.tau.back().get()));
        s.op.back()->compute();
    }
}

// The F_ell run at the default settings.
template<class Elt>
std::unique_ptr<Side<Elt>> field_side(uint32_t n, uint64_t ell, const std::vector<uint64_t>& primes) {
    auto s = std::make_unique<Side<Elt>>();
    s->model = std::make_unique<hecke::SL2ZTessellationTables>(s->K);
    auto rels = GenerateSL2ZRelationsElt<Elt>(s->K, n, static_cast<Elt>(ell));
    s->lssc = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(s->K, std::move(rels), n + 1, static_cast<Elt>(ell));
    s->coh = std::make_unique<EquivariantCohomologyOnKT<Elt>>(s->lssc.get());
    s->coh->compute_cohomology();
    add_ops(*s, n, ell, primes);
    return s;
}

// The Z/ell^k run: every object is CONSTRUCTED under the ring (the ring is read
// once, at construction), exactly WP-3's recipe (build_ring_cell).
template<class Elt>
std::unique_ptr<Side<Elt>> ring_side(uint32_t n, uint64_t ell, uint32_t k, const std::vector<uint64_t>& primes) {
    hecke::ScopedCoefficientRing ring(k);
    auto s = std::make_unique<Side<Elt>>();
    s->k = k;
    s->model = std::make_unique<hecke::SL2ZTessellationTables>(s->K);
    const ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeReductionValidation (E8 item 4a)", ell, k);
    auto rels = hecke::GenerateSL2ZRelationsRing<Elt>(s->K, n, ctx);
    s->lssc = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(s->K, std::move(rels), n + 1, ctx.p);
    s->coh = std::make_unique<EquivariantCohomologyOnKT<Elt>>(s->lssc.get());
    s->coh->compute_cohomology();
    add_ops(*s, n, ell, primes);
    return s;
}

// A D5b basis list (vector of N_q x d_V FlatMatrix) as the columns of a dim_q x m buffer.
template<class Elt>
DenseBuf<Elt> basis_columns(const std::vector<FlatMatrixT<Elt>>& B, uint32_t dim) {
    DenseBuf<Elt> M(dim, static_cast<uint32_t>(B.size()));
    for (uint32_t t = 0; t < B.size(); ++t) {
        const uint32_t r = B[t].get_rows(), c = B[t].get_cols();
        for (uint32_t s = 0; s < r; ++s)
            for (uint32_t v = 0; v < c; ++v) M.at(s * c + v, t) = B[t].get_entry(s, v);
    }
    return M;
}

// (R1): (b_j mod ell) - sum_i R(i, j) h_i in B^q(F_ell).
template<class Elt>
bool red_is_class_map(const EquivariantCohomologyOnKT<Elt>& ring, const EquivariantCohomologyOnKT<Elt>& field,
                      uint32_t q, const DenseBuf<Elt>& R) {
    const Elt ell = ring.ring_context().ell;
    const ModCtx<Elt> F(ell);
    const DenseBuf<Elt>& Hb = ring.H_module_basis(q);
    const uint32_t dim = Hb.rows;
    const DenseBuf<Elt> Hf = basis_columns(field.get_H_eq_bases_fm()[q], dim);
    const DenseBuf<Elt> Bf = basis_columns(field.get_B_eq_bases_fm()[q], dim);
    if (Hf.cols != R.rows || Hb.cols != R.cols) return false;
    DenseBuf<Elt> D(dim, Hb.cols);
    for (uint32_t s = 0; s < dim; ++s)
        for (uint32_t j = 0; j < Hb.cols; ++j) {
            Elt acc = static_cast<Elt>(Hb.at(s, j) % ell);
            for (uint32_t i = 0; i < Hf.cols; ++i) acc = F.sub(acc, F.mul(Hf.at(s, i), R.at(i, j)));
            D.at(s, j) = acc;
        }
    if (D.cols == 0) return true;
    if (Bf.cols == 0) {
        for (uint32_t s = 0; s < dim; ++s)
            for (uint32_t j = 0; j < D.cols; ++j)
                if (D.at(s, j) != 0) return false;
        return true;
    }
    try {
        (void)engine::elim::solve_multi_rhs(F, engine::cview(Bf), engine::cview(D));
    } catch (const std::runtime_error&) {
        return false;
    }
    return true;
}

// Theta over Z/ell^{k_hi} reduces entrywise mod m to `lo` (a dense Theta or an F_ell FlatMatrix).
template<class Elt>
bool theta_reduces(const DenseBuf<Elt>& hi, Elt m, const DenseBuf<Elt>& lo) {
    if (hi.rows != lo.rows || hi.cols != lo.cols) return false;
    for (uint32_t i = 0; i < hi.rows; ++i)
        for (uint32_t j = 0; j < hi.cols; ++j)
            if (hi.at(i, j) % m != lo.at(i, j)) return false;
    return true;
}
template<class Elt>
bool theta_reduces(const DenseBuf<Elt>& hi, Elt m, const FlatMatrixT<Elt>& lo) {
    if (hi.rows != lo.get_rows() || hi.cols != lo.get_cols()) return false;
    for (uint32_t i = 0; i < hi.rows; ++i)
        for (uint32_t j = 0; j < hi.cols; ++j)
            if (hi.at(i, j) % m != lo.get_entry(i, j)) return false;
    return true;
}

struct BulkSpec {
    uint32_t n;
    uint64_t ell;
    std::vector<uint32_t> levels;      // ascending, starting at 2
    std::vector<uint64_t> primes;
    int predicted_htc;                 // -1: not pre-stated
    bool cochain_checks;               // (R3)
};

struct CellOutcome {
    uint32_t dim_H0 = 0, beta_rank0 = 0, htc = 0;
    bool certified_somewhere = false;
    std::vector<uint32_t> torsion;     // ladic_type^1 torsion of the first certified level
};

template<class Elt>
CellOutcome bulk_cell_body(const BulkSpec& cs) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string tag = "(" + std::to_string(cs.n) + "," + std::to_string(cs.ell) + ")";
    const Elt ell = static_cast<Elt>(cs.ell);
    CellOutcome out;
    const auto f = field_side<Elt>(cs.n, cs.ell, cs.primes);
    const EquivariantCohomologyOnKT<Elt>& cf = *f->coh;
    const std::vector<uint32_t> dims = cf.get_H_eq_dimensions();
    std::vector<std::unique_ptr<Side<Elt>>> rs;
    for (uint32_t k : cs.levels) rs.push_back(ring_side<Elt>(cs.n, cs.ell, k, cs.primes));
    ENGINE_CHECK(!rs.empty() && rs.front()->k == 2);
    const EquivariantCohomologyOnKT<Elt>& c2 = *rs.front()->coh;

    // The Bockstein (from the Z/ell^2 object) and (B1)-(B3).
    std::vector<DenseBuf<Elt>> beta;   // beta^0, beta^1
    std::vector<uint32_t> brank;
    for (uint32_t q = 0; q + 1 < dims.size(); ++q) {
        beta.push_back(hecke::BocksteinH(c2, cf, q));
        brank.push_back(hecke::RankOverField(ell, beta.back()));
        check(beta.back().rows == dims[q + 1] && beta.back().cols == dims[q],
              "(B) " + tag + ": beta^" + std::to_string(q) + " is dim H^{q+1} x dim H^q");
        for (size_t i = 0; i < cs.primes.size(); ++i)
            check(hecke::BocksteinEquivariant(ell, f->op[i]->get_H_matrices()[q + 1], f->op[i]->get_H_matrices()[q],
                                              beta.back()),
                  "(B1) " + tag + " p=" + std::to_string(cs.primes[i]) + ": beta^" + std::to_string(q) +
                      " o A(T_p) = A(T_p) o beta^" + std::to_string(q));
    }
    const std::vector<uint32_t> len2 = c2.H_lengths();
    for (uint32_t q = 0; q < dims.size(); ++q) {
        const uint32_t prev = q ? brank[q - 1] : 0, here = q < brank.size() ? brank[q] : 0;
        check(len2[q] + prev + here == 2 * dims[q],
              "(B2) " + tag + ": length H^" + std::to_string(q) + "(Z/ell^2) = " + std::to_string(len2[q]) +
                  " = 2 dim H^q(F_ell) - rank beta^{q-1} - rank beta^q (" + std::to_string(2 * dims[q]) + " - " +
                  std::to_string(prev) + " - " + std::to_string(here) + ")");
    }
    const hecke::BocksteinBookkeeping bk = hecke::BocksteinCounts(c2, cf);
    out.dim_H0 = bk.dim_H0;
    out.beta_rank0 = bk.bockstein_rank0;
    out.htc = bk.higher_torsion_count;
    check(bk.bockstein_rank0 == brank[0] && bk.dim_H0 == dims[0], "(B3) " + tag + ": BocksteinCounts agrees with beta^0");
    check(bk.bockstein_rank0 + bk.higher_torsion_count == bk.dim_H0,
          "(B3) " + tag + ": bockstein_rank^0 + higher_torsion_count == dim H^0(F_ell)");
    if (cs.predicted_htc >= 0)
        check(bk.higher_torsion_count == static_cast<uint32_t>(cs.predicted_htc),
              "(B3) " + tag + ": higher_torsion_count = " + std::to_string(bk.higher_torsion_count) + ", pre-stated " +
                  std::to_string(cs.predicted_htc));

    std::ostringstream coker;
    std::vector<DenseBuf<Elt>> red_prev;
    for (size_t li = 0; li < rs.size(); ++li) {
        const Side<Elt>& r = *rs[li];
        const EquivariantCohomologyOnKT<Elt>& ck = *r.coh;
        const std::string tk = tag + " Z/ell^" + std::to_string(r.k);
        if (!out.certified_somewhere && ck.ladic_applicable() && ck.ladic_type_certified()) {
            out.certified_somewhere = true;
            out.torsion = ck.ladic_type(1).torsion;
            uint32_t ge2 = 0, eq1 = 0;
            for (uint32_t e : out.torsion) (e >= 2 ? ge2 : eq1)++;
            check(bk.higher_torsion_count == ge2,
                  "(B3) " + tk + ": higher_torsion_count == #{e_i >= 2} = " + std::to_string(ge2) + " (certified)");
            check(bk.bockstein_rank0 == eq1, "(B3) " + tk + ": bockstein_rank^0 == #{e_i = 1} = " + std::to_string(eq1));
            check(out.torsion.size() == dims[0], "(B3) " + tk + ": t = #torsion summands == dim H^0(F_ell)");
        }
        std::vector<DenseBuf<Elt>> red;
        coker << " K=" << r.k << ":";
        for (uint32_t q = 0; q < dims.size(); ++q) {
            // (R0) the basis tier B freezes red_H^q on reads back as itself: the (c5)
            // coordinates of the b_i are the identity (WP-12 M-1's pipeline clause).
            {
                const DenseBuf<Elt>& Hb = ck.H_module_basis(q);
                bool ident = true;
                if (Hb.cols > 0) {
                    const auto co = ck.H_class_coordinates(q, engine::cview(Hb));
                    ident = co.size() == Hb.cols;
                    for (uint32_t j = 0; ident && j < Hb.cols; ++j) {
                        ident = co[j].size() == Hb.cols;
                        for (uint32_t i = 0; ident && i < Hb.cols; ++i) ident = (co[j][i] == (i == j ? 1u : 0u));
                    }
                }
                check(ident, "(R0) " + tk + ": H_class_coordinates(q, H_module_basis(q)) is the identity, q = " +
                                 std::to_string(q));
            }
            red.push_back(hecke::ReduceHToField(ck, cf, q));
            const DenseBuf<Elt>& R = red.back();
            check(R.rows == dims[q] && R.cols == ck.H_invariant_factors(q).size(),
                  "(R1) " + tk + ": red_H^" + std::to_string(q) + " is dim H^q(F_ell) x t_q");
            check(red_is_class_map(ck, cf, q, R), "(R1) " + tk + ": red_H^" + std::to_string(q) + " is the class map");
            const uint32_t rr = hecke::RankOverField(ell, R);
            coker << " q" << q << "=" << (dims[q] - rr);
            if (r.k == 2) {
                const uint32_t bq = q < brank.size() ? brank[q] : 0;
                check(rr + bq == dims[q], "(R4) " + tk + ": rank red_H^" + std::to_string(q) + " + rank beta^" +
                                              std::to_string(q) + " == dim H^q(F_ell) (exactness)");
                if (q < beta.size())
                    check(hecke::ComposesToZero(ell, beta[q], R),
                          "(R4) " + tk + ": beta^" + std::to_string(q) + " o red_H^" + std::to_string(q) + " = 0");
            }
            for (size_t i = 0; i < cs.primes.size(); ++i) {
                const std::string tp = tk + " p=" + std::to_string(cs.primes[i]) + (cs.primes[i] == cs.ell ? " (U_ell)" : "");
                check(hecke::ReductionToFieldEquivariant(ell, f->op[i]->get_H_matrices()[q], r.op[i]->H_matrix_module(q), R),
                      "(R2) " + tp + ": A(T_p) o red_H^" + std::to_string(q) + " = red_H^" + std::to_string(q) + " o A(T_p)");
                if (cs.cochain_checks)
                    check(theta_reduces(r.op[i]->Theta_module(q), ell, f->op[i]->get_Theta_flat_matrices()[q]),
                          "(R3) " + tp + ": Theta^" + std::to_string(q) + " reduces entrywise mod ell to the F_ell Theta_flat");
            }
        }
        if (li > 0) {
            const Side<Elt>& lo = *rs[li - 1];
            const std::string tl = tag + " Z/ell^" + std::to_string(r.k) + " -> Z/ell^" + std::to_string(lo.k);
            for (uint32_t q = 0; q < dims.size(); ++q) {
                const DenseBuf<Elt> Rl = hecke::ReduceHBetweenLevels(ck, *lo.coh, q);
                const std::vector<uint32_t>& el = lo.coh->H_invariant_factors(q);
                for (size_t i = 0; i < cs.primes.size(); ++i) {
                    check(hecke::ReductionBetweenLevelsEquivariant(ck.ring_context(), el, lo.op[i]->H_matrix_module(q),
                                                                  r.op[i]->H_matrix_module(q), Rl),
                          "(R5) " + tl + " p=" + std::to_string(cs.primes[i]) + ": A o red = red o A on H^" + std::to_string(q));
                    if (cs.cochain_checks)
                        check(theta_reduces(r.op[i]->Theta_module(q), lo.coh->ring_context().p, lo.op[i]->Theta_module(q)),
                              "(R5) " + tl + " p=" + std::to_string(cs.primes[i]) + ": Theta^" + std::to_string(q) +
                                  " reduces entrywise");
                }
                // functoriality: red_lo o red_{hi -> lo} == red_hi (over F_ell)
                const DenseBuf<Elt>& Rlo = red_prev[q];
                bool fun = Rlo.cols == Rl.rows && Rlo.rows == red[q].rows && Rl.cols == red[q].cols;
                const ModCtx<Elt> F(ell);
                for (uint32_t i = 0; fun && i < Rlo.rows; ++i)
                    for (uint32_t j = 0; j < Rl.cols; ++j) {
                        Elt acc = 0;
                        for (uint32_t m = 0; m < Rl.rows; ++m)
                            acc = F.add(acc, F.mul(Rlo.at(i, m), static_cast<Elt>(Rl.at(m, j) % ell)));
                        if (acc != red[q].at(i, j) % ell) { fun = false; break; }
                    }
                check(fun, "(R5) " + tl + ": red_H^" + std::to_string(q) + "(lo) o red(hi -> lo) == red_H^" +
                               std::to_string(q) + "(hi)");
            }
        }
        red_prev = std::move(red);
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  (R)(B) " << tag << ": dim H^q(F_ell) = " << vec_str(dims) << ", rank beta^0 = " << brank[0]
              << ", rank beta^1 = " << (brank.size() > 1 ? brank[1] : 0) << ", bockstein_rank^0 = " << bk.bockstein_rank0
              << ", higher_torsion_count = " << bk.higher_torsion_count
              << (out.certified_somewhere ? ", e = " + vec_str(out.torsion) : std::string(", uncertified"))
              << "; red cokernel length" << coker.str() << "; primes " << cs.primes.size() << "  " << secs << " s"
              << std::endl;
    return out;
}

template<class Elt>
void bulk_cell(const BulkSpec& cs) {
    try {
        (void)bulk_cell_body<Elt>(cs);
    } catch (const std::exception& e) {
        check(false, "(R)(B) (" + std::to_string(cs.n) + "," + std::to_string(cs.ell) + "): threw " + e.what());
    }
}

void bulk_cell_auto(const BulkSpec& cs) {
    // Z/ell^K in uint32_t where ell^K fits it, else uint64_t (F-E's u64 route).
    unsigned __int128 m = 1;
    for (uint32_t i = 0; i < cs.levels.back(); ++i) m *= cs.ell;
    if (m <= 0xFFFFFFFFull) bulk_cell<uint32_t>(cs);
    else bulk_cell<uint64_t>(cs);
}

// ---- D6 at (10, 691) ----------------------------------------------------------------
void d6_arm() {
    const uint32_t n = 10;
    const uint64_t ell = 691, M = ell * ell;
    const auto r = ring_side<uint32_t>(n, ell, 2, {2});
    const auto& op = *r->op[0];
    const auto fc = op.free_charpoly_H1();
    // (x - a_E)(x - a_D)^2 over Z, a_E = 1 + 2^11 = 2049, a_D = tau(2) = -24, low -> high, mod 691^2.
    const int64_t aE = 2049, aD = -24;
    // (x - aE)(x^2 - 2 aD x + aD^2) = x^3 - (aE + 2 aD) x^2 + (2 aE aD + aD^2) x - aE aD^2
    const int64_t c[4] = {-aE * aD * aD, 2 * aE * aD + aD * aD, -(aE + 2 * aD), 1};
    std::vector<uint64_t> want(4);
    for (int i = 0; i < 4; ++i) want[i] = static_cast<uint64_t>(((c[i] % static_cast<int64_t>(M)) + M) % M);
    bool eq = fc.precision == 2 && fc.modulus == M && fc.coeffs.size() == 4;
    for (int i = 0; eq && i < 4; ++i) eq = (fc.coeffs[i] == want[i]);
    std::ostringstream got;
    for (size_t i = 0; i < fc.coeffs.size(); ++i) got << (i ? "," : "") << fc.coeffs[i];
    check(eq, "(D6) (10,691) Z/691^2: charpoly_A(T_2)^1_free = (x - 2049)(x + 24)^2 mod 691^2, got [" + got.str() + "]");
    const uint64_t aEm = static_cast<uint64_t>(aE % static_cast<int64_t>(M));
    const uint64_t aDm = static_cast<uint64_t>(((aD % static_cast<int64_t>(M)) + M) % M);
    check(aEm % ell == aDm % ell, "(D6) a_E == a_Delta (mod 691): 2049 + 24 = 3 * 691");
    check(aEm != aDm, "(D6) a_E != a_Delta (mod 691^2): the eigenvalues separate over Z/691^2");
    // The eigenlines over Z/691^2: ker(A - a I) on H^1 (free of rank 3 here).
    const ModCtx<uint32_t>& ctx = r->coh->ring_context();
    const DenseBuf<uint32_t>& A = op.H_matrix_module(1);
    check(A.rows == 3 && A.cols == 3 && r->coh->H_invariant_factors(1) == std::vector<uint32_t>{2, 2, 2},
          "(D6) H^1(Z/691^2) is free of rank 3");
    std::ostringstream lines;
    uint32_t lenE = 0, lenD = 0;
    std::vector<std::vector<uint32_t>> freegens[2];   // the unit-pivot (free) Howell rows of each eigenspace
    for (int which = 0; which < 2; ++which) {
        const uint32_t a = static_cast<uint32_t>(which == 0 ? aEm : aDm);
        DenseBuf<uint32_t> S(A.rows, A.cols);
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) S.at(i, j) = (i == j) ? ctx.sub(A.at(i, j), a) : A.at(i, j);
        const auto ker = engine::elim::kernel_module(ctx, engine::cview(S));
        const uint32_t len = engine::elim::length(ctx, ker);
        (which == 0 ? lenE : lenD) = len;
        lines << (which == 0 ? " ker(A - a_E): length " : "; ker(A - a_Delta): length ") << len << ", Howell rows";
        for (uint32_t i = 0; i < ker.rows(); ++i) {
            if (ker.pivot_v[i] == 0) {
                std::vector<uint32_t> g(ker.H.cols);
                for (uint32_t j = 0; j < ker.H.cols; ++j) g[j] = ker.H.at(i, j);
                freegens[which].push_back(g);
            }
            lines << " [";
            for (uint32_t j = 0; j < ker.H.cols; ++j) lines << (j ? "," : "") << ker.H.at(i, j);
            lines << "]/v" << ker.pivot_v[i];
        }
    }
    check(lenE >= 1 && lenD >= 1, "(D6) both eigenvalues have a non-zero eigenspace over Z/691^2");
    // "Separate": a free Eisenstein eigenline and a free cuspidal eigenplane over Z/691^2.
    check(freegens[0].size() == 1 && freegens[1].size() == 2,
          "(D6) over Z/691^2 the Eisenstein eigenline is free of rank 1 and the cuspidal eigenspace free of rank 2");
    if (freegens[0].size() == 1 && freegens[1].size() == 2) {
        // Their sum's index in H^1: the 691-adic valuation of det[e | c1 | c2] (recorded -- the
        // Ramanujan congruence says the Eisenstein line is congruent to a cusp line mod 691).
        const auto& e = freegens[0][0];
        const auto& c1 = freegens[1][0];
        const auto& c2 = freegens[1][1];
        auto m = [&](uint32_t a, uint32_t b) { return ctx.mul(a, b); };
        const uint32_t det = ctx.sub(ctx.add(ctx.add(m(e[0], ctx.sub(m(c1[1], c2[2]), m(c1[2], c2[1]))),
                                                     m(e[2], ctx.sub(m(c1[0], c2[1]), m(c1[1], c2[0])))), 0),
                                     m(e[1], ctx.sub(m(c1[0], c2[2]), m(c1[2], c2[0]))));
        lines << "; det[eis | cusp] = " << det << ", 691-adic valuation " << ctx.valuation(det)
              << " (the sum of the eigenspaces has index 691^" << ctx.valuation(det) << " in H^1)";
    }
    std::cout << "  (D6) (10,691) over Z/691^2, T_2: charpoly_free = [" << got.str() << "] = (x - 2049)(x + 24)^2; "
              << "a_E = 2049 = 1 + 2^11, a_Delta = -24 = tau(2) (the classical normalisation);" << lines.str()
              << std::endl;
}

} // namespace

void HeckeReductionBulkArms(uint64_t& checks, uint32_t& failures) {
    g_checks = &checks;
    g_fail = &failures;
    const uint32_t f0 = failures;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<uint64_t> P = {2, 3, 5};
    // The pre-stated cells ([E] sec:ring (e); slot-8 plan WP-4 scope 2-3): higher_torsion_count
    // 0 at (10, 691), (6, 5), (16, 5); 1 at the positive controls (30, 5) (Z/25) and (60, 5)
    // (Z/25 (+) Z/5), at both k.
    bulk_cell_auto({10, 691, {2}, P, 0, true});
    bulk_cell_auto({6, 5, {2, 3}, P, 0, true});
    bulk_cell_auto({16, 5, {2, 3}, P, 0, true});
    bulk_cell_auto({30, 5, {2, 3}, P, 1, true});
    bulk_cell_auto({60, 5, {2, 3}, P, 1, true});
    try { d6_arm(); }
    catch (const std::exception& e) { check(false, std::string("(D6) threw: ") + e.what()); }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  E8 item 4a arms: " << (failures - f0) << " failing, " << secs << " s" << std::endl;
}

uint32_t HeckeReductionCoverageSlowValidation() {
    uint64_t checks = 0;
    uint32_t failures = 0;
    g_checks = &checks;
    g_fail = &failures;
    g_printed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeReductionCoverageSlowValidation (slot-8 WP-4, E8 item 4a (d) on the bulk): the reduction and "
                 "Bockstein equivariances on the E2 case list at ell >= 5, Z/ell^2, p in {2, 3, 5}"
              << std::endl;
    // The E2 case list at ell >= 5 ([X] Sec. 1.6's 31 ext-subgrid cells, then the 23 M1 cells of
    // HeckeExtEndToEndValidation not already listed; 52 distinct).  Every one admits a Z/ell^2 run
    // (ell >= 5, ell^2 < 2^64).
    std::vector<std::pair<uint32_t, uint64_t>> cells;
    for (uint32_t n : {2u, 4u, 10u, 16u, 22u, 24u})
        for (uint64_t ell : {5ull, 13ull, 691ull, 10007ull, 134217757ull}) cells.push_back({n, ell});
    cells.push_back({32, 5});
    const std::vector<std::pair<uint32_t, uint64_t>> m1 = {
        {6, 5},    {8, 7},    {12, 11},  {12, 5},    {34, 5},       {58, 7},     {58, 5},
        {82, 7},   {22, 5},   {26, 7},   {32, 131},  {160, 691},    {198, 691},  {10, 691},
        {14, 3617}, {16, 43867}, {18, 283}, {18, 617}, {20, 131}, {20, 593}, {22, 103},
        {22, 2294797}, {24, 657931}};
    for (const auto& c : m1) {
        bool dup = false;
        for (const auto& d : cells) dup = dup || (d == c);
        if (!dup) cells.push_back(c);
    }
    const char* env = std::getenv("HECKE_REDUCTION_COVERAGE_LARGE");
    const bool large = env != nullptr && std::string(env) == "1";
    uint32_t run = 0, left_out = 0;
    for (const auto& c : cells) {
        if (!large && (c == std::make_pair(160u, uint64_t(691)) || c == std::make_pair(198u, uint64_t(691)))) {
            ++left_out;
            continue;
        }
        bulk_cell_auto({c.first, c.second, {2}, {2, 3, 5}, -1, true});
        ++run;
    }
    if (!large)
        std::cout << "  E8_COVERAGE (160, 691), (198, 691): left out for cost; set HECKE_REDUCTION_COVERAGE_LARGE=1 "
                     "(read-only) to include them"
                  << std::endl;
    // (slot-9 WP-4, E8 item 4b) the boundary and the cone at K = 2, p = 2, on the same list.
    HeckeReductionConeCoverageArms(checks, failures, large);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  E8_COVERAGE cells=" << cells.size() << " run=" << run << " left_out=" << left_out
              << " excluded_by_ring=0 checks=" << checks << " failing=" << failures << " " << secs << " s" << std::endl;
    std::cout << "EFFECTIVE HeckeReductionCoverageSlowValidation arm=" << (large ? "large" : "default")
              << " cells=" << run << std::endl;
    std::cout << "HeckeReductionCoverageSlowValidation: " << (failures == 0 ? "PASS" : "FAIL") << " (" << checks
              << " checks, " << failures << " failing)" << std::endl;
    return failures;
}
