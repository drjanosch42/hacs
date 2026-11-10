// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-6 (phase E5-3) -- HeckeConeActionValidation.
// See the header for the gates A1..A13 and the grid.

#include "HeckeConeActionValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)

#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryOperator.h"
#include "HeckeConeAction.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"
#include "HeckeOperatorOnK.h"
#include "HeckePairHomotopy.h"
#include "HeckeTauTable.h"
#include "SL2ZBoundaryPair.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantMappingCone.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using hecke::ChainTerm;
using hecke::ConeActionCertificate;
using hecke::ConeRelationsCertificate;
using hecke::EtaVertexPolicy;
using hecke::HeckeBoundaryOperator;
using hecke::HeckeBoundaryTauTable;
using hecke::HeckeConeAction;
using hecke::HeckeConeRelativeRoute;
using hecke::HeckeConeStrictPairRoute;
using hecke::HeckeCosetTables;
using hecke::HeckeOperatorOnK;
using hecke::HeckePairHomotopy;
using hecke::HeckeTauTable;
using hecke::Mat22;
using hecke::SL2ZBoundaryPair;
using hecke::SL2ZTessellationTables;
using hecke::TauBoundaryVertexPolicy;
using hecke::TauVertexPolicy;

namespace {

using Elt = uint32_t;

struct CheckLog {
    uint32_t failures = 0;
    uint64_t checks = 0;
    void fail(const std::string& msg) {
        ++failures;
        if (failures <= 40) std::cout << "HeckeConeActionValidation: FAILED -- " << msg << std::endl;
    }
    void check(bool ok, const std::string& msg) {
        ++checks;
        if (!ok) fail(msg);
    }
};

// =========================================================================
// Section 0 -- own linear algebra (this TU re-derives, never re-calls).
// =========================================================================

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

FlatMatrix own_mul(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() == 0 || B.get_cols() == 0 || A.get_cols() == 0)
        return FlatMatrix::CreateZero(A.get_rows(), B.get_cols(), A.get_p());
    return A * B;
}

FlatMatrix own_add(const FlatMatrix& A, const FlatMatrix& B) {
    const engine::ModCtx<Elt> ctx(A.get_p());
    if (A.get_rows() == 0 || A.get_cols() == 0) return A;
    std::vector<Elt> d(static_cast<size_t>(A.get_rows()) * A.get_cols());
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            d[static_cast<size_t>(i) * A.get_cols() + j] = ctx.add(A.get_entry(i, j), B.get_entry(i, j));
    return FlatMatrix(A.get_rows(), A.get_cols(), A.get_p(), d);
}

FlatMatrix own_sub(const FlatMatrix& A, const FlatMatrix& B) {
    const engine::ModCtx<Elt> ctx(A.get_p());
    if (A.get_rows() == 0 || A.get_cols() == 0) return A;
    std::vector<Elt> d(static_cast<size_t>(A.get_rows()) * A.get_cols());
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            d[static_cast<size_t>(i) * A.get_cols() + j] = ctx.sub(A.get_entry(i, j), B.get_entry(i, j));
    return FlatMatrix(A.get_rows(), A.get_cols(), A.get_p(), d);
}

FlatMatrix own_transpose(const FlatMatrix& A) {
    const uint32_t R = A.get_rows(), C = A.get_cols();
    if (R == 0 || C == 0) return FlatMatrix::CreateZero(C, R, A.get_p());
    std::vector<Elt> d(static_cast<size_t>(C) * R);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(j) * R + i] = A.get_entry(i, j);
    return FlatMatrix(C, R, A.get_p(), d);
}

FlatMatrix own_kernel_cols(const FlatMatrix& A) {
    const uint32_t n = A.get_cols();
    if (n == 0) return FlatMatrix::CreateZero(0, 0, A.get_p());
    if (A.get_rows() == 0) return FlatMatrix(n, A.get_p());
    std::vector<FlatMatrix> ker = A.kernel();
    const uint32_t k = static_cast<uint32_t>(ker.size());
    if (k == 0) return FlatMatrix::CreateZero(n, 0, A.get_p());
    std::vector<Elt> d(static_cast<size_t>(n) * k, 0);
    for (uint32_t t = 0; t < k; ++t)
        for (uint32_t i = 0; i < n; ++i) d[static_cast<size_t>(i) * k + t] = ker[t].get_entry(i, 0);
    return FlatMatrix(n, k, A.get_p(), d);
}

FlatMatrix flat_of_dense(const engine::DenseBuf<Elt>& D, Elt p) {
    const uint32_t R = D.rows, C = D.cols;
    if (R == 0 || C == 0) return FlatMatrix::CreateZero(R, C, p);
    std::vector<Elt> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(i) * C + j] = D.at(i, j);
    return FlatMatrix(R, C, p, d);
}

FlatMatrix flat_of_block(const engine::BlockMat<Elt>& B, Elt p) {
    return flat_of_dense(B.to_dense(engine::Layout::RowMajor), p);
}

FlatMatrix flat_of_matrix(const MatrixT<Elt>& M) {
    FlatMatrix out(M.rows, M.cols, M.p);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) out.set_entry(i, j, M.data[i][j]);
    return out;
}

MatrixT<Elt> matrix_of_flat(const FlatMatrix& M) {
    MatrixT<Elt> out(M.get_rows(), M.get_cols(), M.get_p());
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) out.data[i][j] = M.get_entry(i, j);
    return out;
}

bool matrices_equal(const MatrixT<Elt>& X, const MatrixT<Elt>& Y) {
    return X.rows == Y.rows && X.cols == Y.cols && X.data == Y.data;
}

// d_q (x) I of a complex, in the flat layout; `negate` gives -d.
FlatMatrix own_coboundary_tensor(const SimplicialComplex& C, uint32_t q, uint32_t dV, Elt p, bool negate) {
    const uint32_t Ns = C.get_num_q_simplices(q);
    const uint32_t Nt = (q + 1 <= C.get_dimension()) ? C.get_num_q_simplices(q + 1) : 0;
    if (Ns == 0 || Nt == 0) return FlatMatrix::CreateZero(Nt * dV, Ns * dV, p);
    const FlatMatrix D1 = C.coboundary_operator(q, p);
    const engine::ModCtx<Elt> ctx(p);
    const uint32_t cols = Ns * dV;
    std::vector<Elt> d(static_cast<size_t>(Nt) * dV * cols, 0);
    for (uint32_t i = 0; i < Nt; ++i)
        for (uint32_t j = 0; j < Ns; ++j) {
            Elt s = D1.get_entry(i, j);
            if (s == 0) continue;
            if (negate) s = ctx.neg(s);
            for (uint32_t k = 0; k < dV; ++k) d[static_cast<size_t>(i * dV + k) * cols + (j * dV + k)] = s;
        }
    return FlatMatrix(Nt * dV, cols, p, d);
}

// The flat restriction r^q : C^q(K; V) -> C^q(K_bd; V) from the vertex lists.
FlatMatrix own_r_flat(const SimplicialComplex& K, const SimplicialComplex& Kb, uint32_t q, uint32_t dV, Elt p) {
    const uint32_t NK = (q <= K.get_dimension()) ? K.get_num_q_simplices(q) : 0;
    const uint32_t Nb = (q <= Kb.get_dimension()) ? Kb.get_num_q_simplices(q) : 0;
    if (NK == 0 || Nb == 0) return FlatMatrix::CreateZero(Nb * dV, NK * dV, p);
    std::vector<Elt> d(static_cast<size_t>(Nb) * dV * NK * dV, 0);
    const uint32_t cols = NK * dV;
    for (uint32_t s = 0; s < Nb; ++s) {
        const int64_t t = K.find_simplex_index(q, Kb.get_simplices()[q][s]);
        if (t < 0) throw std::runtime_error("own_r_flat: a boundary simplex is not in K");
        for (uint32_t k = 0; k < dV; ++k)
            d[static_cast<size_t>(s * dV + k) * cols + (static_cast<uint32_t>(t) * dV + k)] = 1;
    }
    return FlatMatrix(Nb * dV, cols, p, d);
}

// =========================================================================
// Section 1 -- the OWN coset transport of the eta-table ([C, eq:derivedh]).
// Independent of HeckeConeAction's assembly: its own rho_n, its own product
// order, its own accumulation.
// =========================================================================

FlatMatrix own_rho(const Mat22& g, uint32_t n_sym, Elt ell) {
    const std::array<Elt, 4> r = g.reduced_mod(ell);
    const FlatMatrix g_fm(2, 2, ell, std::vector<Elt>{ r[0], r[1], r[2], r[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym);
}

FlatMatrix own_h(const HeckePairHomotopy& eta, uint32_t q, uint32_t rows, uint32_t cols, uint32_t dV,
                 uint32_t n_sym, Elt ell) {
    if (rows == 0 || cols == 0 || q == 0) return FlatMatrix::CreateZero(rows, cols, ell);
    const uint32_t qb = q - 1;
    if (qb >= eta.num_degrees()) return FlatMatrix::CreateZero(rows, cols, ell);
    const engine::ModCtx<Elt> ctx(ell);
    std::vector<Elt> data(static_cast<size_t>(rows) * cols, 0);
    const std::vector<uint32_t>& slots = eta.boundary_simplices(qb);
    for (uint32_t s = 0; s < slots.size(); ++s) {
        for (uint32_t j = 1; j <= eta.num_cosets(); ++j) {
            for (const ChainTerm& ct : eta.eta(j, qb, slots[s])) {
                int64_t cm = static_cast<int64_t>(ct.coeff) % static_cast<int64_t>(ell);
                if (cm < 0) cm += static_cast<int64_t>(ell);
                const Elt cmod = static_cast<Elt>(cm);
                if (cmod == 0) continue;
                const FlatMatrix B = own_rho(eta.cosets().delta(j) * ct.gamma, n_sym, ell);
                const size_t col0 = static_cast<size_t>(ct.sigma_idx) * dV;
                for (uint32_t rr = 0; rr < dV; ++rr)
                    for (uint32_t cc = 0; cc < dV; ++cc) {
                        const Elt v = B.get_entry(rr, cc);
                        if (v == 0) continue;
                        Elt& cell = data[(static_cast<size_t>(s) * dV + rr) * cols + col0 + cc];
                        cell = ctx.add(cell, ctx.mul(cmod, v));
                    }
            }
        }
    }
    return FlatMatrix(rows, cols, ell, data);
}

// =========================================================================
// Section 2 -- the per-cell objects.
// =========================================================================

struct Cell {
    uint32_t n = 0, ell = 0;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::unique_ptr<SL2ZBoundaryPair> pair;
    std::unique_ptr<EquivariantMappingCone> cone;
};

std::unique_ptr<Cell> build_cell(const SimplicialComplex& K, uint32_t n, uint32_t ell) {
    auto c = std::make_unique<Cell>();
    c->n = n; c->ell = ell;
    c->lssc = std::make_unique<LocalSystemSimplicialComplex>(K, GenerateSL2ZRelationsElt<Elt>(K, n, ell), n + 1, ell);
    c->coh = std::make_unique<EquivariantCohomologyOnK>(c->lssc.get());
    c->coh->compute_cohomology();
    c->pair = std::make_unique<SL2ZBoundaryPair>(c->coh.get(), K);
    c->cone = c->pair->make_cone();
    return c;
}

struct Op {
    uint64_t p = 0;
    uint32_t nu = 0;
    std::unique_ptr<HeckeCosetTables> cosets;
    std::unique_ptr<HeckeTauTable> tau;
    std::unique_ptr<HeckeBoundaryTauTable> taub;
    std::unique_ptr<HeckePairHomotopy> eta;
    std::unique_ptr<HeckeOperatorOnK> bulk;
    std::unique_ptr<HeckeBoundaryOperator> bd;
    std::unique_ptr<HeckeConeAction> act;
};

std::unique_ptr<Op> build_op(const SimplicialComplex& K, const SL2ZTessellationTables& model, const Cell& cell,
                             uint64_t p, uint32_t nu,
                             EtaVertexPolicy epol = EtaVertexPolicy::CanonicalTreePath,
                             TauBoundaryVertexPolicy tpol = TauBoundaryVertexPolicy::NearestLineVertex) {
    auto o = std::make_unique<Op>();
    o->p = p; o->nu = nu;
    o->cosets = std::make_unique<HeckeCosetTables>(p, nu, cell.ell, cell.n);
    o->tau = std::make_unique<HeckeTauTable>(K, model, *o->cosets, cell.ell, cell.n);
    o->taub = std::make_unique<HeckeBoundaryTauTable>(K, model, *o->cosets, tpol);
    o->eta = std::make_unique<HeckePairHomotopy>(*o->tau, *o->taub, epol);
    o->bulk = std::make_unique<HeckeOperatorOnK>(cell.coh.get(), o->cosets.get(), o->tau.get());
    o->bulk->compute();
    o->bd = std::make_unique<HeckeBoundaryOperator>(&cell.pair->boundary_cohomology(), o->cosets.get(), o->taub.get());
    o->bd->compute();
    o->act = std::make_unique<HeckeConeAction>(cell.cone.get(), o->bulk.get(), o->bd.get(), o->eta.get());
    o->act->compute();
    return o;
}

struct Tally {
    uint32_t cells = 0, ops = 0, ell23_ops = 0;
    uint32_t restriction_recorded_ok = 0, restriction_recorded_total = 0;
    uint32_t relations_recorded_ok = 0, relations_recorded_total = 0;
    uint32_t eta_zero_true = 0, cochain_commutes_true = 0;
    uint32_t commutator_pairs = 0, recursion_degrees = 0;
    uint32_t tau_ties = 0;
    uint32_t residual_degrees = 0, residual_flat_nonzero = 0;
    uint32_t phi_flat_identity_true = 0;
    // [OD] 2026-09-22 Q2: cells whose EMITTED record says the r-half of
    // obl:coneexact(iii) was asserted, over all cells and over ell in {2,3}.
    uint32_t record_r_asserted = 0, ell23_record_r_asserted = 0;
    // The item-12 tamper re-runs verify() twice, so it is run ONCE PER CELL
    // (on the cell's first operator) rather than once per operator: the
    // clause is a statement about the certificate, not about the operator.
    uint32_t item12_cell = 0, item12_arms = 0;
};

// -------------------------------------------------------------------------
// The per-operator arms A1..A7, A10, A11.
// -------------------------------------------------------------------------
void check_op(CheckLog& log, Tally& tally, const std::string& where, const SimplicialComplex& K,
              const SimplicialComplex& Kb, const Cell& cell, const Op& op) {
    const Elt p = cell.ell;
    const uint32_t dV = cell.n + 1, Q = cell.cone->q_max();
    const HeckeConeAction& act = *op.act;
    const ConeActionCertificate& cert = act.certificate();
    ++tally.ops;
    if (p < 5) ++tally.ell23_ops;

    // A1: the certificate.
    {
        std::vector<std::string> f;
        const uint32_t bad = cert.verify(f);
        log.check(bad == 0, where + " ConeActionCertificate::verify() fails: " + (f.empty() ? std::string("?") : f[0]));
    }

    // A2: obl:eta(v) against the OWN coset transport.
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t rows = cell.cone->boundary_part_dim(q), cols = cell.cone->bulk_part_dim(q);
        const FlatMatrix mine = own_h(*op.eta, q, rows, cols, dV, cell.n, p);
        log.check(flat_equal(mine, act.h_flat(q)),
                  where + " stored h^" + std::to_string(q) + " != own eq:derivedh transport");
        if (q == 0) log.check(flat_is_zero(act.h_flat(q)), where + " h^0 != 0");
    }

    // A3: the OWN dense residual bracket of lem:conesign, and Dcone Phi = Phi Dcone.
    std::vector<FlatMatrix> dK, dB, rq;
    for (uint32_t q = 0; q <= Q + 1; ++q) dK.push_back(own_coboundary_tensor(K, q, dV, p, false));
    for (uint32_t q = 0; q <= Q + 1; ++q) dB.push_back(own_coboundary_tensor(Kb, q, dV, p, false));
    for (uint32_t q = 0; q <= Q + 1; ++q) rq.push_back(own_r_flat(K, Kb, q, dV, p));
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t tgt = (q <= Kb.get_dimension()) ? Kb.get_num_q_simplices(q) * dV : 0;
        const uint32_t nb = cell.cone->bulk_part_dim(q);
        if (tgt == 0 || nb == 0) continue;
        FlatMatrix acc = FlatMatrix::CreateZero(tgt, nb, p);
        if (q > 0 && cell.cone->boundary_part_dim(q) > 0) acc = own_add(acc, own_mul(dB[q - 1], act.h_flat(q)));
        if (q + 1 <= Q && cell.cone->boundary_part_dim(q + 1) == tgt) acc = own_add(acc, own_mul(act.h_flat(q + 1), dK[q]));
        const FlatMatrix rTh = own_mul(rq[q], cert.Theta[q]);
        acc = own_add(acc, rTh);
        FlatMatrix Thr = FlatMatrix::CreateZero(tgt, nb, p);
        if (q + 1 <= Q && cert.Theta_bd[q + 1].get_rows() == tgt) Thr = own_mul(cert.Theta_bd[q + 1], rq[q]);
        acc = own_sub(acc, Thr);
        // [C, lem:derivedhomotopy] / [C, lem:conesign] are statements about the
        // complex Ceq(K, Rel; L) (the coset-transport formula uses phi's
        // Gamma-equivariance), so the bracket is asserted zero on the stored
        // C^q_eq basis and the full flat value is RECORDED.
        const FlatMatrix Ceqb = flat_of_dense(cell.coh->C_eq_block(q).to_flat(engine::Layout::RowMajor), p);
        log.check(flat_is_zero(own_mul(acc, Ceqb)),
                  where + " own residual (dh + hd + r Theta - Theta^bd r) != 0 on C^q_eq at q = " + std::to_string(q));
        log.check(flat_equal(acc, act.residual(q)), where + " the object's residual != the own one at q = " + std::to_string(q));
        log.check(flat_is_zero(act.residual_on_Ceq(q)), where + " residual_on_Ceq() != 0 at q = " + std::to_string(q));
        if (!flat_is_zero(acc)) ++tally.residual_flat_nonzero;
        ++tally.residual_degrees;
        // the (i) identity in its "dh + hd = Theta^bd r - r Theta" form, on C_eq
        FlatMatrix lhs = FlatMatrix::CreateZero(tgt, nb, p);
        if (q > 0 && cell.cone->boundary_part_dim(q) > 0) lhs = own_add(lhs, own_mul(dB[q - 1], act.h_flat(q)));
        if (q + 1 <= Q && cell.cone->boundary_part_dim(q + 1) == tgt) lhs = own_add(lhs, own_mul(act.h_flat(q + 1), dK[q]));
        log.check(flat_equal(own_mul(lhs, Ceqb), own_mul(own_sub(Thr, rTh), Ceqb)),
                  where + " dh + hd != Theta^bd r - r Theta on C^q_eq at q = " + std::to_string(q));
        // the stored flat restriction agrees with the own one
        if (q <= Kb.get_dimension() && q <= K.get_dimension())
            log.check(flat_equal(rq[q], flat_of_block(cell.cone->r_flat_block(q), p)),
                      where + " the cone's flat r^" + std::to_string(q) + " != the own vertex-list assembly");
    }
    for (uint32_t q = 0; q <= Q; ++q) {
        const FlatMatrix Dq = flat_of_block(cell.cone->D_block(q), p);
        const FlatMatrix left = own_mul(Dq, act.Phi_flat(q));
        FlatMatrix right = FlatMatrix::CreateZero(left.get_rows(), left.get_cols(), p);
        if (q + 1 <= Q) right = own_mul(act.Phi_flat(q + 1), Dq);
        const FlatMatrix Cc = flat_of_dense(cell.cone->Ceq_block(q).to_flat(engine::Layout::RowMajor), p);
        log.check(flat_equal(own_mul(left, Cc), own_mul(right, Cc)),
                  where + " Dcone Phi != Phi Dcone on the equivariant cone cochains at q = " + std::to_string(q));
        if (q + 1 <= Q) {
            const FlatMatrix Dq1 = flat_of_block(cell.cone->D_block(q + 1), p);
            log.check(flat_is_zero(own_mul(Dq1, Dq)), where + " Dcone^2 != 0 at q = " + std::to_string(q));
        }
    }

    // A4: the two routes on the whole standard basis.
    log.check(cert.routes_bit_identical, where + " the two bracketings of h are not recorded bit-identical");
    if (cert.phi_flat_identity) ++tally.phi_flat_identity_true;
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t fd = cell.cone->flat_dim(q);
        for (uint32_t b = 0; b < fd; ++b) {
            std::vector<Elt> e(fd, 0);
            e[b] = 1;
            const FlatMatrix eb(fd, 1, p, e);
            const FlatMatrix y1 = act.apply_Phi(q, eb);
            const FlatMatrix y2 = act.apply_Phi_direct(q, eb);
            bool col_ok = true;
            for (uint32_t i = 0; i < fd && col_ok; ++i) col_ok = (y1.get_entry(i, 0) == act.Phi_flat(q).get_entry(i, b));
            log.check(col_ok, where + " apply_Phi != the stored column at q = " + std::to_string(q));
            log.check(flat_equal(y1, y2), where + " apply_Phi_direct != apply_Phi at q = " + std::to_string(q) +
                                              ", basis vector " + std::to_string(b));
        }
    }

    // A5: obl:coneexact(ii).
    for (uint32_t q = 0; q <= Q; ++q) {
        const FlatMatrix Zq = flat_of_dense(cell.cone->Z_block(q).to_flat(engine::Layout::RowMajor), p);
        const FlatMatrix Dq = flat_of_block(cell.cone->D_block(q), p);
        if (Zq.get_cols() > 0)
            log.check(flat_is_zero(own_mul(Dq, own_mul(act.Phi_flat(q), Zq))),
                      where + " Dcone Phi z != 0 on a stored cocycle at q = " + std::to_string(q));
        const FlatMatrix Hq = flat_of_dense(cell.cone->H_block(q).to_flat(engine::Layout::RowMajor), p);
        const FlatMatrix Pq = flat_of_block(cell.cone->P_block(q), p);
        const MatrixT<Elt> mine = matrix_of_flat(own_mul(Pq, own_mul(act.Phi_flat(q), Hq)));
        log.check(matrices_equal(mine, act.A_c(q)), where + " A_c^" + std::to_string(q) + " != own P Phi H");
        log.check(act.A_c(q).rows == cell.cone->dim_Hc(q) && act.A_c(q).cols == cell.cone->dim_Hc(q),
                  where + " A_c^" + std::to_string(q) + " is not dim H^q_c square");
    }

    // A6: obl:coneexact(iii), against the pipeline's own operators.
    const std::vector<MatrixT<Elt>>& A_bulk = op.bulk->get_H_matrices();
    const std::vector<MatrixT<Elt>>& A_bd = op.bd->get_H_matrices();
    for (uint32_t q = 0; q <= Q; ++q) {
        if (cell.cone->pi_built(q) && q < A_bulk.size()) {
            const FlatMatrix Am = flat_of_matrix(A_bulk[q]), Acm = flat_of_matrix(act.A_c(q));
            log.check(flat_equal(own_mul(cell.cone->pi(q), Acm), own_mul(Am, cell.cone->pi(q))),
                      where + " pi^" + std::to_string(q) + " A_c != A pi");
        }
        if (cell.cone->r_built(q) && q < A_bulk.size() && q < A_bd.size()) {
            const FlatMatrix Am = flat_of_matrix(A_bulk[q]), Abm = flat_of_matrix(A_bd[q]);
            const bool ok = flat_equal(own_mul(cell.cone->r(q), Am), own_mul(Abm, cell.cone->r(q)));
            ++tally.restriction_recorded_total;
            if (ok) ++tally.restriction_recorded_ok;
            // [C, obl:coneexact](iii): asserted at EVERY ell since slot-5 WP-4 (OI-3) --
            // clause (i) (C2, above, unconditional in ell) proves r^q A = A_bd r^q on
            // cohomology with no hypothesis on ell, so this own re-derivation is now a real
            // assertion at ell in {2,3} too, not merely a recorded measurement there.
            log.check(ok, where + " r^" + std::to_string(q) + " A != A_bd r");
            // [OD] 2026-09-22 Q2: and the CERTIFICATE says so.  The flag the
            // record emits must report the assertion item 8 performs, which
            // carries no ell guard -- so it is true HERE, at ell in {2, 3},
            // where the retired six_invertible would have said "false".
            log.check(cert.restriction_equivariance_asserted,
                      where + " the certificate denies asserting the r-half at q = " + std::to_string(q));
        }
        if (cell.cone->iota_built(q) && q < A_bd.size() && q + 1 <= Q) {
            const FlatMatrix Abm = flat_of_matrix(A_bd[q]), Ac1 = flat_of_matrix(act.A_c(q + 1));
            log.check(flat_equal(own_mul(cell.cone->iota(q), Abm), own_mul(Ac1, cell.cone->iota(q))),
                      where + " iota^" + std::to_string(q) + " A_bd != A_c iota");
        }
    }

    // A11: obl:eta(vi).
    log.check(!cert.eta_zero || cert.cochain_commutes, where + " eta_zero holds but cochain_commutes does not");
    if (cert.eta_zero) ++tally.eta_zero_true;
    if (cert.cochain_commutes) ++tally.cochain_commutes_true;

    // A10: the tamper arm (one entry of h; one entry of A_c).
    if (act.h_flat(1).get_rows() > 0 && act.h_flat(1).get_cols() > 0) {
        ConeActionCertificate t = cert;
        t.h[1].set_entry(0, 0, (t.h[1].get_entry(0, 0) + 1) % p);
        const std::vector<uint32_t> items = t.failing_items();
        const bool has1 = std::find(items.begin(), items.end(), 1u) != items.end();
        log.check(has1, where + " a corrupted h entry does not fail obl:eta(v) (item 1)");
    }
    if (act.A_c(1).rows > 0) {
        ConeActionCertificate t = cert;
        t.A_c[1].data[0][0] = (t.A_c[1].data[0][0] + 1) % p;
        const std::vector<uint32_t> items = t.failing_items();
        const bool has6 = std::find(items.begin(), items.end(), 6u) != items.end();
        log.check(has6, where + " a corrupted A_c entry does not fail the assembly item (6)");
        const bool has1 = std::find(items.begin(), items.end(), 1u) != items.end();
        log.check(!has1, where + " a corrupted A_c entry wrongly fails obl:eta(v)");
    }

    // A10b ([OD] 2026-09-22 Q2): item 12, record honesty.  POSITIVE CONTROL
    // first -- the untouched certificate fails NOTHING, so the refusal below
    // is the single corruption's and not a pre-existing failure -- then the
    // one flipped flag, which must fail item 12 and item 12 ALONE.  The check
    // stands behind no shape gate, so it fires on this certificate whatever
    // else it carries.
    if (tally.item12_cell != tally.cells) {
        tally.item12_cell = tally.cells;
        ++tally.item12_arms;
        const std::vector<uint32_t> clean = cert.failing_items();
        log.check(clean.empty(), where + " the untouched certificate already fails an item");
        ConeActionCertificate t = cert;
        t.restriction_equivariance_asserted = false;
        const std::vector<uint32_t> items = t.failing_items();
        const bool has12 = std::find(items.begin(), items.end(), 12u) != items.end();
        log.check(has12, where + " a record denying the r-half assertion does not fail item 12");
        log.check(items.size() == 1, where + " the denied r-half assertion fails more than item 12");
    }

    // A12: the record.
    const std::string js = act.to_json();
    log.check(js.find("\"schema\":\"hecke_cone_action/v1\"") != std::string::npos &&
                  js.find("\"boundary_model\":\"circle\"") != std::string::npos &&
                  js.find(std::string("\"boundary_is_group_cohomology\":") + (p != 2 ? "true" : "false")) != std::string::npos &&
                  js.find("\"kind\":\"cochain_map\",\"degree_shift\":-1,\"source\":\"bulk\"") != std::string::npos &&
                  js.find("\"tau_boundary_fnv64\"") != std::string::npos,
              where + " to_json record fields");
    // [OD] 2026-09-22 Q2: the emitted flag reports the assertion the run
    // PERFORMED.  Asserted on EVERY cell -- the whole point is that ell in
    // {2, 3} now reads `true`, where the retired six_invertible read `false`
    // about an assertion the run had already made.
    const bool rec_asserted =
            js.find("\"restriction_equivariance_asserted\":true") != std::string::npos;
    log.check(rec_asserted, where + " the record does not report the r-half as asserted");
    if (rec_asserted) {
        ++tally.record_r_asserted;
        if (p < 5) ++tally.ell23_record_r_asserted;
    }

    // A13: the golden collectors are defined and digest-stable in-process.
    const std::vector<ConeGoldenObject> g1 = act.golden_objects();
    log.check(g1.size() >= 6u * (Q + 1) + 3u, where + " too few golden collector records");
    log.check(act.golden_digest() == act.golden_digest(), where + " the golden digest is not stable in-process");
}

// -------------------------------------------------------------------------
// A7 -- obl:coneexact(vi) on a cell: commutativity and the nu = 2 recursion.
// -------------------------------------------------------------------------
void check_relations(CheckLog& log, Tally& tally, const std::string& where, const SimplicialComplex& K,
                     const SL2ZTessellationTables& model, const Cell& cell,
                     const std::vector<const Op*>& ops_nu1, const Op* op_nu2, uint64_t rec_prime) {
    const Elt p = cell.ell;
    const uint32_t Q = cell.cone->q_max();
    ConeRelationsCertificate rc;
    rc.ell = p; rc.n_sym = cell.n; rc.q_max = Q; rc.six_invertible = (p >= 5);
    for (const Op* o : ops_nu1) {
        rc.primes.push_back(o->p);
        std::vector<MatrixT<Elt>> per_q;
        for (uint32_t q = 0; q <= Q; ++q) per_q.push_back(o->act->A_c(q));
        rc.A_c.push_back(std::move(per_q));
    }
    if (op_nu2 != nullptr) {
        rc.recursion_present = true;
        rc.recursion_prime = rec_prime;
        for (uint32_t q = 0; q <= Q; ++q) rc.A_c_primitive_p2.push_back(op_nu2->act->A_c(q));
        for (const Op* o : ops_nu1)
            if (o->p == rec_prime)
                for (uint32_t q = 0; q <= Q; ++q) rc.A_c_p.push_back(o->act->A_c(q));
    }
    std::vector<std::string> f;
    const uint32_t bad = rc.verify(f);
    tally.commutator_pairs += rc.commutator_pairs;
    tally.recursion_degrees += rc.recursion_degrees;
    tally.relations_recorded_total += rc.commutator_pairs + rc.recursion_degrees;
    tally.relations_recorded_ok += (rc.commutator_pairs - rc.commutator_failures) +
                                   (rc.recursion_degrees - rc.recursion_failures);
    log.check(bad == 0, where + " ConeRelationsCertificate fails: " + (f.empty() ? std::string("?") : f[0]));
    if (p >= 5) {
        log.check(rc.commutator_failures == 0, where + " [A_c(T_p), A_c(T_p')] != 0");
        log.check(rc.recursion_failures == 0, where + " the nu = 2 cone recursion fails");
    }

    // The bulk analogue against HeckeOperatorFamily's INDEPENDENT double-coset
    // route, carried to the cone through pi ([C, obl:coneexact](iii) + (vi)).
    if (op_nu2 != nullptr && p >= 5) {
        hecke::HeckeOperatorFamily fam(cell.coh.get(), K, model, p, cell.n);
        const std::vector<MatrixT<Elt>> full = fam.T_p_pow_direct(rec_prime, 2);
        const std::vector<MatrixT<Elt>> prim = fam.T_p_pow_primitive_direct(rec_prime, 2);
        const engine::ModCtx<Elt> ctx(p);
        Elt pn = 1;
        const Elt pmod = static_cast<Elt>(rec_prime % p);
        for (uint32_t i = 0; i < cell.n; ++i) pn = ctx.mul(pn, pmod);
        for (uint32_t q = 0; q <= Q && q < full.size(); ++q) {
            if (full[q].rows == 0) continue;
            // full = primitive + p^n I ([C, sec:primfull]) -- re-derived here.
            FlatMatrix built = flat_of_matrix(prim[q]);
            for (uint32_t i = 0; i < built.get_rows(); ++i)
                built.set_entry(i, i, ctx.add(built.get_entry(i, i), pn));
            log.check(flat_equal(built, flat_of_matrix(full[q])),
                      where + " T_p_pow_direct != T_p_pow_primitive_direct + p^n I at q = " + std::to_string(q));
            // pi A_c(T'_{p^2}) == A(T'_{p^2}) pi against the independent route.
            if (cell.cone->pi_built(q)) {
                const FlatMatrix Acm = flat_of_matrix(op_nu2->act->A_c(q));
                const FlatMatrix Aprim = flat_of_matrix(prim[q]);
                log.check(flat_equal(own_mul(cell.cone->pi(q), Acm), own_mul(Aprim, cell.cone->pi(q))),
                          where + " pi A_c(T'_{p^2}) != A(T'_{p^2}) pi (independent double-coset route) at q = " + std::to_string(q));
            }
        }
    }
}

// -------------------------------------------------------------------------
// A14 -- obl:coneexact(vii), first sentence: the strict-pair cross-route of
// [C, prop:coneroutes](1) (slot-6 WP-4).  Own log and tally, so the
// pre-existing CONE_ACTION accounting is untouched.
// -------------------------------------------------------------------------
struct StrictTally {
    uint32_t ops = 0, asserted = 0, recorded = 0, built = 0, not_built = 0;
    uint32_t agree = 0, recorded_disagree = 0, degrees = 0;
    uint32_t fills_int64 = 0, fills_widened = 0, cochain_commutes = 0, tau_differs = 0;
    std::vector<std::string> recorded_notes;   // every recorded disagreement / build failure
    double seconds = 0.0;
};

void check_strict_pair(CheckLog& log, StrictTally& t, const std::string& where, const SimplicialComplex& K,
                       const SL2ZTessellationTables& model, const Cell& cell, const Op& op) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t Q = cell.cone->q_max();
    const bool six = (cell.ell >= 5);
    ++t.ops;
    HeckeConeStrictPairRoute route(op.act.get());
    try {
        route.compute();
    } catch (const std::exception& e) {
        // Reached only at 6 in R^x (the route RECORDS at ell in {2, 3}).
        log.fail(where + " strict pair ABORTED: " + e.what());
        t.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    const HeckeConeStrictPairRoute::Record& r = route.record();
    log.check(route.computed(), where + " strict pair: compute() returned without completing");
    log.check(r.asserted == six && r.six_invertible == six,
              where + " strict pair: the assert/record split does not follow 6 in R^x");
    if (six) ++t.asserted; else ++t.recorded;
    if (!r.built) {
        ++t.not_built;
        log.check(!six, where + " strict pair: not built at 6 in R^x without an abort");
        t.recorded_notes.push_back(where + ": strict datum not built (" + r.build_failure + ")");
        t.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    ++t.built;
    const HeckeTauTable& ts = *route.tau_strict();
    log.check(ts.is_strict_pair() && ts.policy() == TauVertexPolicy::BoundaryStrict,
              where + " strict pair: tau^s does not carry BoundaryStrict");
    log.check(ts.certify() == 0, where + " strict pair: tau^s does not re-certify");
    log.check(std::string(ts.solve_regime()) == "Z" && !ts.used_mod_ell_solve() && ts.num_fills_mod_ell() == 0,
              where + " strict pair: tau^s is not a datum over Z");
    log.check(ts.strict_boundary_table() == op.taub.get(),
              where + " strict pair: tau^s was not built against the normative tau_bd");
    const uint32_t bd_simplices = static_cast<uint32_t>(model.boundary_simplices(0).size() +
                                                        model.boundary_simplices(1).size());
    log.check(r.strict_boundary_slots == op.cosets->num_cosets() * bd_simplices,
              where + " strict pair: step (B) did not pre-assign m * |K_bd| slots");
    // tau^s o j = j o tau_bd, re-read slot by slot from the two public tables.
    for (uint32_t q = 0; q < 2; ++q)
        for (uint32_t s : model.boundary_simplices(q))
            for (uint32_t j = 1; j <= op.cosets->num_cosets(); ++j) {
                const auto& a = ts.tau(j, q, s);
                const auto& b = op.taub->tau(j, q, s);
                bool same = a.size() == b.size();
                for (size_t k = 0; same && k < a.size(); ++k)
                    same = a[k].coeff == b[k].coeff && a[k].gamma == b[k].gamma && a[k].sigma_idx == b[k].sigma_idx;
                log.check(same, where + " strict pair: tau^s(s_j sigma) != tau_bd(s_j sigma) on a boundary slot");
            }
    log.check(r.eta_zero && route.eta_strict()->eta_zero() && r.eta_nonzero_slots == 0,
              where + " strict pair: eta on (tau^s, tau_bd) is not identically zero");
    // NON-VACUITY: the shipped tau is not a map of pairs wherever its eta is
    // non-zero, so there the strict table must be a DIFFERENT table.
    if (!op.eta->eta_zero())
        log.check(r.tau_differs, where + " strict pair: tau^s equals the shipped tau although the shipped eta != 0");
    if (r.tau_differs) ++t.tau_differs;
    if (r.cochain_commutes) ++t.cochain_commutes;
    t.fills_int64 += r.fills_int64;
    t.fills_widened += r.fills_widened;
    log.check(r.degrees_compared == Q + 1, where + " strict pair: not every degree was compared");
    const HeckeConeAction& as = *route.action_strict();
    bool all_same = true;
    for (uint32_t q = 0; q <= Q; ++q) {
        log.check(flat_is_zero(as.h_flat(q)), where + " strict pair: h^s is not zero in degree " + std::to_string(q));
        // The comparison re-done here from the two public objects.
        const bool same = matrices_equal(op.act->A_c(q), as.A_c(q));
        ++t.degrees;
        if (!same) all_same = false;
        if (six) {
            log.check(same, where + " strict pair: A_c^" + std::to_string(q) + " differs (6 in R^x)");
        } else if (!same) {
            t.recorded_notes.push_back(where + ": A_c^" + std::to_string(q) + " differs (RECORDED, ell in {2,3})");
        }
    }
    log.check(all_same == r.disagreeing_degrees.empty(),
              where + " strict pair: the route's record disagrees with the re-done comparison");
    if (all_same) ++t.agree;
    else if (!six) ++t.recorded_disagree;
    t.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    (void)K;
}

// -------------------------------------------------------------------------
// A15 -- obl:coneexact(vii), second sentence: route 3 of
// [C, prop:coneroutes](2) (slot-6 WP-5).  Own log and tally, so every
// pre-existing printed line (CONE_ACTION, CONE_STRICT_PAIR) is untouched.
// -------------------------------------------------------------------------
struct Route3Tally {
    uint32_t ops = 0, agree = 0, degrees = 0, theta_prime_differs = 0, e_nonzero = 0;
    uint32_t naive_leaves_ker_r = 0;           // Theta (uncorrected) moves some H(ker r) cocycle off ker r
    uint32_t tier_b_A_c = 0;                   // the relabel: A_c^q objects carrying tier 'B'
    uint32_t legacy_cells = 0;
    std::vector<uint32_t> dim_H_rel_sum;       // summed over operators, per degree
    double seconds = 0.0;
};

void check_route3(CheckLog& log, Route3Tally& t, const std::string& where, const SimplicialComplex& K,
                  const SimplicialComplex& Kb, const SL2ZTessellationTables& model, const Cell& cell, const Op& op) {
    const auto t0 = std::chrono::steady_clock::now();
    const Elt p = cell.ell;
    const uint32_t dV = cell.n + 1, Q = cell.cone->q_max();
    const HeckeConeAction& act = *op.act;
    const ConeActionCertificate& cert = act.certificate();
    ++t.ops;
    HeckeConeRelativeRoute route(op.act.get());
    try {
        route.compute();
    } catch (const std::exception& e) {
        log.fail(where + " route 3 ABORTED: " + e.what());
        t.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return;
    }
    const HeckeConeRelativeRoute::Record& r = route.record();
    log.check(route.computed() && r.agrees() && r.degrees_compared == Q + 1,
              where + " route 3: compute() returned without an agreeing comparison in every degree");
    const uint32_t bd_simplices = static_cast<uint32_t>(model.boundary_simplices(0).size() +
                                                        model.boundary_simplices(1).size());
    log.check(r.vanishing_simplices == bd_simplices, where + " route 3: ker r does not force the boundary simplices to zero");
    log.check(route.relative_cohomology().is_relative(), where + " route 3: the relative object is not relative");
    log.check(r.dim_H_rel.size() == Q + 1 && r.dim_H_rel[0] == 0, where + " route 3: H^0(ker r) != 0 ([C, lem:conerelative](4))");
    if (t.dim_H_rel_sum.size() < r.dim_H_rel.size()) t.dim_H_rel_sum.resize(r.dim_H_rel.size(), 0);
    for (size_t q = 0; q < r.dim_H_rel.size(); ++q) t.dim_H_rel_sum[q] += r.dim_H_rel[q];
    // ext_0, E and Theta' re-derived here from the vertex lists and the public h.
    std::vector<FlatMatrix> ext0;
    for (uint32_t q = 0; q <= Q; ++q) {
        ext0.push_back(own_transpose(own_r_flat(K, Kb, q, dV, p)));
        log.check(flat_equal(ext0[q], route.ext0_flat(q)), where + " route 3: ext_0^" + std::to_string(q) + " differs");
    }
    for (uint32_t q = 1; q <= Q; ++q) {
        const FlatMatrix E = own_mul(ext0[q - 1], act.h_flat(q));
        log.check(flat_equal(E, route.E_flat(q)), where + " route 3: E^" + std::to_string(q) + " != ext_0 h");
    }
    for (uint32_t q = 0; q <= Q; ++q) {
        FlatMatrix T = cert.Theta[q];
        if (T.get_rows() == 0) continue;
        if (q > 0 && cell.cone->bulk_part_dim(q - 1) > 0)
            T = own_add(T, own_mul(own_coboundary_tensor(K, q - 1, dV, p, false), route.E_flat(q)));
        if (q + 1 <= Q && cell.cone->bulk_part_dim(q + 1) > 0)
            T = own_add(T, own_mul(route.E_flat(q + 1), own_coboundary_tensor(K, q, dV, p, false)));
        log.check(flat_equal(T, route.Theta_prime_flat(q)), where + " route 3: Theta'^" + std::to_string(q) + " != Theta + dE + Ed");
    }
    // The comparison re-done here in the intertwining form A_c M == M A'.
    bool naive_leaves = false;
    for (uint32_t q = 0; q <= Q; ++q) {
        const FlatMatrix& M = route.correspondence(q);
        const uint32_t dH = r.dim_H_rel[q];
        ++t.degrees;
        if (dH == 0) { log.check(act.A_c(q).rows == 0, where + " route 3: H^q(ker r) = 0 but A_c^q is not empty"); continue; }
        log.check(flat_equal(own_mul(flat_of_matrix(act.A_c(q)), M), own_mul(M, flat_of_matrix(route.A_rel(q)))),
                  where + " route 3: A_c M != M A' in degree " + std::to_string(q));
        log.check(matrices_equal(route.A_c_transported(q), act.A_c(q)),
                  where + " route 3: M A' M^{-1} != A_c in degree " + std::to_string(q));
        // NON-VACUITY of the correction: does the UNCORRECTED Theta keep H(ker r) inside ker r?
        if (q <= Kb.get_dimension() && Kb.get_num_q_simplices(q) > 0) {
            const FlatMatrix Hr = flat_of_dense(route.relative_cohomology().H_eq_block(q).to_flat(engine::Layout::RowMajor), p);
            if (!flat_is_zero(own_mul(own_r_flat(K, Kb, q, dV, p), own_mul(cert.Theta[q], Hr)))) naive_leaves = true;
        }
    }
    if (naive_leaves) ++t.naive_leaves_ker_r;
    // NON-VACUITY: where the shipped Theta is not a map of pairs, Theta' must differ from it on C_eq.
    if (!cert.cochain_commutes)
        log.check(r.theta_prime_differs && r.E_nonzero && !r.r_commutes_normative,
                  where + " route 3: Theta' == Theta on C_eq although the shipped Theta is not a map of pairs");
    if (r.theta_prime_differs) ++t.theta_prime_differs;
    if (r.E_nonzero) ++t.e_nonzero;
    if (r.agrees()) ++t.agree;
    // The relabel ([WP5-3] finding 5): A_c^q is tier B in the unit itself.
    for (const ConeGoldenObject& g : act.golden_objects())
        if (g.object.compare(0, 4, "A_c^") == 0) {
            log.check(g.tier == 'B', where + " " + g.object + " is not labelled tier B by the unit");
            if (g.tier == 'B') ++t.tier_b_A_c;
        }
    t.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

#if OC_ENABLE_LEGACY   // (S24 CL-L2) the legacy cohomology arm exists only in an ON build
// The relative sub-complex's legacy arm (ScopedLegacyCohomologyPath) gives
// the same C / Z / B / H / P entrywise as the block arm.
void check_route3_legacy(CheckLog& log, Route3Tally& t, const std::string& where, const Op& op) {
    HeckeConeRelativeRoute route(op.act.get());
    route.compute();
    const EquivariantCohomologyOnK& blk = route.relative_cohomology();
    EquivariantCohomologyOnK leg(blk.get_lssc(), blk.vanishing_simplices());
    {
        const EquivariantCohomologyOptions::ScopedLegacyCohomologyPath force(true);
        leg.compute_cohomology();
    }
    log.check(leg.get_H_eq_dimensions() == blk.get_H_eq_dimensions(), where + " route 3 legacy arm: dim H^* differs");
    const auto same_lists = [](const std::vector<std::vector<FlatMatrix>>& a, const std::vector<std::vector<FlatMatrix>>& b) {
        if (a.size() != b.size()) return false;
        for (size_t q = 0; q < a.size(); ++q) {
            if (a[q].size() != b[q].size()) return false;
            for (size_t i = 0; i < a[q].size(); ++i) if (!flat_equal(a[q][i], b[q][i])) return false;
        }
        return true;
    };
    log.check(same_lists(leg.get_C_eq_bases_fm(), blk.get_C_eq_bases_fm()), where + " route 3 legacy arm: C_rel differs");
    log.check(same_lists(leg.get_Z_eq_bases_fm(), blk.get_Z_eq_bases_fm()), where + " route 3 legacy arm: Z_rel differs");
    log.check(same_lists(leg.get_B_eq_bases_fm(), blk.get_B_eq_bases_fm()), where + " route 3 legacy arm: B_rel differs");
    log.check(same_lists(leg.get_H_eq_bases_fm(), blk.get_H_eq_bases_fm()), where + " route 3 legacy arm: H_rel differs");
    const auto& Pl = leg.get_H_eq_projectors_from_C_eq_flat_fm();
    const auto& Pb = blk.get_H_eq_projectors_from_C_eq_flat_fm();
    bool ps = Pl.size() == Pb.size();
    for (size_t q = 0; ps && q < Pl.size(); ++q) ps = flat_equal(Pl[q], Pb[q]);
    log.check(ps, where + " route 3 legacy arm: P_rel differs");
    ++t.legacy_cells;
}
#endif

// -------------------------------------------------------------------------
// A8 -- obl:coneexact(viii)(a): the four policy combinations.
// -------------------------------------------------------------------------
void check_independence(CheckLog& log, const std::string& where, const SimplicialComplex& K,
                        const SL2ZTessellationTables& model, const Cell& cell, uint64_t p_h, uint32_t nu) {
    const uint32_t Q = cell.cone->q_max();
    struct Arm { EtaVertexPolicy e; TauBoundaryVertexPolicy t; const char* name; };
    const Arm arms[4] = {
        { EtaVertexPolicy::CanonicalTreePath, TauBoundaryVertexPolicy::NearestLineVertex, "canonical/nearest" },
        { EtaVertexPolicy::LegacyBfsPath,     TauBoundaryVertexPolicy::NearestLineVertex, "legacybfs/nearest" },
        { EtaVertexPolicy::CanonicalTreePath, TauBoundaryVertexPolicy::NextLineVertex,    "canonical/next" },
        { EtaVertexPolicy::LegacyBfsPath,     TauBoundaryVertexPolicy::NextLineVertex,    "legacybfs/next" },
    };
    std::vector<std::unique_ptr<Op>> built;
    for (const Arm& a : arms) built.push_back(build_op(K, model, cell, p_h, nu, a.e, a.t));
    for (size_t i = 1; i < built.size(); ++i) {
        for (uint32_t q = 0; q <= Q; ++q)
            log.check(matrices_equal(built[0]->act->A_c(q), built[i]->act->A_c(q)),
                      where + " A_c^" + std::to_string(q) + " differs between policy arms 0 and " +
                          std::string(arms[i].name));
    }
    // NON-VACUITY: the two eta policies must move the eta-table, and the two
    // tau_bd policies must move the tau_bd-table (slot-3 audit; a rule that
    // leaves a table byte-identical would make the arm prove nothing).
    log.check(built[0]->eta->table_fnv64() != built[1]->eta->table_fnv64(),
              where + " the two eta vertex policies give the SAME eta table hash (the arm would be vacuous)");
    log.check(built[0]->taub->table_fnv64() != built[2]->taub->table_fnv64(),
              where + " the two tau_bd vertex policies give the SAME tau_bd table hash (the arm would be vacuous)");
    log.check(built[0]->eta->tau_boundary_table_fnv64() != built[2]->eta->tau_boundary_table_fnv64(),
              where + " eta's recorded tau_bd hash does not move with the tau_bd policy");
    // ... and the objects that MAY move do move somewhere.
    bool h_moved = false, thbd_moved = false;
    for (uint32_t q = 0; q <= Q; ++q) {
        if (!flat_equal(built[0]->act->h_flat(q), built[1]->act->h_flat(q))) h_moved = true;
        if (!flat_equal(built[0]->act->h_flat(q), built[2]->act->h_flat(q))) h_moved = true;
        if (!flat_equal(built[0]->act->certificate().Theta_bd[q], built[2]->act->certificate().Theta_bd[q]))
            thbd_moved = true;
    }
    log.check(h_moved, where + " h_flat never moves between the four policy arms (the arm would be vacuous)");
    log.check(thbd_moved, where + " Theta^bd_flat does not move with the tau_bd policy");
    // The cone itself carries no Hecke datum: its bases and Dcone are the same
    // object in all four arms, which the shared cone pointer makes exact.
    log.check(built[0]->act->cone().golden_digest() == built[3]->act->cone().golden_digest(),
              where + " the cone's own golden digest moved (it must not: the cone carries no Hecke datum)");
    std::cout << "  " << where << " INDEPENDENCE eta_hash " << std::hex << built[0]->eta->table_fnv64() << " / "
              << built[1]->eta->table_fnv64() << ", tau_bd_hash " << built[0]->taub->table_fnv64() << " / "
              << built[2]->taub->table_fnv64() << std::dec << "; A_c identical on all four arms" << std::endl;
}

// -------------------------------------------------------------------------
// A9 -- obl:coneexact(viii)(b): the negative control h' = h + xi.
// -------------------------------------------------------------------------
void check_negative_control(CheckLog& log, const std::string& where, const SimplicialComplex& K,
                            const SimplicialComplex& Kb, const Cell& cell, const Op& op) {
    const Elt p = cell.ell;
    const uint32_t dV = cell.n + 1, Q = cell.cone->q_max();
    const HeckeConeAction& act = *op.act;
    const ConeActionCertificate& cert = act.certificate();
    if (Q < 1 || cell.cone->bulk_part_dim(1) == 0 || cell.cone->boundary_part_dim(1) == 0) {
        log.fail(where + " (viii)(b): the cell has no degree-1 h to perturb");
        return;
    }
    // v_0: a stored H^0(bd) cocycle (a flat 0-cochain of K_bd in C^0_eq, killed by d^bd).
    const FlatMatrix Hbd0 = flat_of_dense(cell.pair->boundary_cohomology().H_eq_block(0).to_flat(engine::Layout::RowMajor), p);
    if (Hbd0.get_cols() == 0) { log.fail(where + " (viii)(b): H^0(bd) = 0, no v_0"); return; }
    const uint32_t nb0 = Hbd0.get_rows();
    std::vector<Elt> v0d(nb0);
    for (uint32_t i = 0; i < nb0; ++i) v0d[i] = Hbd0.get_entry(i, 0);
    const FlatMatrix v0(nb0, 1, p, v0d);
    log.check(flat_is_zero(own_mul(own_coboundary_tensor(Kb, 0, dV, p, false), v0)), where + " (viii)(b): v_0 is not a cocycle");

    // u: a functional on C^1(K; V) vanishing on the EQUIVARIANT coboundaries
    // B^1_eq = d(C^0_eq) -- not on the flat ones: K is contractible, so the
    // flat H^1 vanishes and every equivariant 1-cocycle IS a flat coboundary,
    // which would make a flat-annihilating xi identically zero on cocycles.
    // The contract's complex is Ceq(K, Rel; L) throughout ([C, def:pair]), and
    // [C, prop:coneambiguity](2) splits C^1 = B^1 (+) H~^1 (+) W^1 THERE; this
    // is that construction with a rank-one lambda.
    const FlatMatrix d0 = own_coboundary_tensor(K, 0, dV, p, false);       // nK1 x nK0
    const FlatMatrix Ceq0 = flat_of_dense(cell.coh->C_eq_block(0).to_flat(engine::Layout::RowMajor), p);
    const FlatMatrix Beq1 = own_mul(d0, Ceq0);                             // nK1 x dim C^0_eq
    const FlatMatrix leftnull = own_kernel_cols(own_transpose(Beq1));      // nK1 x k
    const FlatMatrix Hbulk1 = flat_of_dense(cell.coh->H_eq_block(1).to_flat(engine::Layout::RowMajor), p); // nK1 x dH1
    const FlatMatrix kerr1 = own_kernel_cols(cell.cone->r(1));             // dH1 x kdim
    if (kerr1.get_cols() == 0) { log.fail(where + " (viii)(b): ker r^1 = 0, the ambiguity is zero here"); return; }
    const FlatMatrix Zker = own_mul(Hbulk1, kerr1);                        // nK1 x kdim
    int64_t chosen = -1;
    for (uint32_t t = 0; t < leftnull.get_cols() && chosen < 0; ++t) {
        for (uint32_t c = 0; c < Zker.get_cols(); ++c) {
            const engine::ModCtx<Elt> ctx(p);
            Elt acc = 0;
            for (uint32_t i = 0; i < Zker.get_rows(); ++i)
                acc = ctx.add(acc, ctx.mul(leftnull.get_entry(i, t), Zker.get_entry(i, c)));
            if (acc != 0) { chosen = static_cast<int64_t>(t); break; }
        }
    }
    if (chosen < 0) { log.fail(where + " (viii)(b): no functional vanishing on B^1_eq separates ker r^1"); return; }
    const uint32_t nK1 = cell.cone->bulk_part_dim(1);
    std::vector<Elt> xid(static_cast<size_t>(nb0) * nK1, 0);
    {
        const engine::ModCtx<Elt> ctx(p);
        for (uint32_t i = 0; i < nb0; ++i)
            for (uint32_t j = 0; j < nK1; ++j)
                xid[static_cast<size_t>(i) * nK1 + j] = ctx.mul(v0.get_entry(i, 0), leftnull.get_entry(j, static_cast<uint32_t>(chosen)));
    }
    const FlatMatrix xi(nb0, nK1, p, xid);
    log.check(!flat_is_zero(xi), where + " (viii)(b): xi is zero");
    // xi is a degree-(-1) cochain map OF THE CONTRACT'S COMPLEX: d^bd xi = 0
    // (its image is spanned by the cocycle v_0) and xi d = 0 on C^0_eq.
    log.check(flat_is_zero(own_mul(own_coboundary_tensor(Kb, 0, dV, p, false), xi)), where + " (viii)(b): d xi != 0");
    log.check(flat_is_zero(own_mul(xi, Beq1)), where + " (viii)(b): xi d != 0 on C^0_eq");
    // ... and it is genuinely non-zero on the equivariant cocycles it must move.
    log.check(!flat_is_zero(own_mul(xi, Zker)), where + " (viii)(b): xi vanishes on ker r^1");

    // The tampered certificate: h' = h + xi, Phi' rebuilt, A_c' re-projected.
    ConeActionCertificate t = cert;
    t.h[1] = own_add(t.h[1], xi);
    {
        const uint32_t nb = t.bulk_part_dim[1], nd = t.boundary_part_dim[1], fd = nb + nd;
        std::vector<Elt> d(static_cast<size_t>(fd) * fd, 0);
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < nb; ++j) d[static_cast<size_t>(i) * fd + j] = t.Theta[1].get_entry(i, j);
        for (uint32_t i = 0; i < nd; ++i) {
            for (uint32_t j = 0; j < nb; ++j) d[static_cast<size_t>(nb + i) * fd + j] = t.h[1].get_entry(i, j);
            for (uint32_t j = 0; j < nd; ++j) d[static_cast<size_t>(nb + i) * fd + nb + j] = t.Theta_bd[1].get_entry(i, j);
        }
        t.Phi[1] = FlatMatrix(fd, fd, p, d);
    }
    t.A_c[1] = matrix_of_flat(own_mul(t.P[1], own_mul(t.Phi[1], t.H[1])));
    const std::vector<uint32_t> items = t.failing_items();
    const bool has1 = std::find(items.begin(), items.end(), 1u) != items.end();
    std::string joined;
    for (uint32_t x : items) joined += (joined.empty() ? "" : ",") + std::to_string(x);
    log.check(has1, where + " (viii)(b): the tampered h' does not fail obl:eta(v) (items " + joined + ")");
    log.check(items.size() == 1, where + " (viii)(b): the tampered h' fails more than obl:eta(v) (items " + joined + ")");
    log.check(!matrices_equal(t.A_c[1], act.A_c(1)),
              where + " (viii)(b): the tampered h' does not change A_c^(1) -- the ambiguity would be invisible");
    std::cout << "  " << where << " NEGATIVE CONTROL xi != 0, failing items {" << joined
              << "}, A_c^(1) changed: " << (matrices_equal(t.A_c[1], act.A_c(1)) ? "no" : "yes") << std::endl;
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

uint32_t HeckeConeActionValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    CheckLog log;
    Tally tally;
    CheckLog slog;        // A14 (slot-6 WP-4): its own log, so CONE_ACTION's accounting is unchanged
    StrictTally stally;
    CheckLog rlog;        // A15 (slot-6 WP-5): route 3, its own log for the same reason
    Route3Tally rtally;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);

        std::cout << "HeckeConeActionValidation: the Hecke action on the mapping cone (Hecke/HeckeConeAction): "
                     "h = eta~ by coset transport ([C, eq:derivedh]), Phi_g = (Theta, Theta^bd + h) ([C, eq:conemap]), "
                     "the residual bracket of [C, lem:conesign] asserted zero, A_c^{(q)} by [C, alg:assembly] on Cone(r), "
                     "ConeActionCertificate ([C, obl:eta](v), [C, obl:coneexact](i)-(iii)) and ConeRelationsCertificate "
                     "([C, obl:coneexact](vi)); GATING = A1..A13 of the header."
                  << std::endl;

        if (const char* big = std::getenv("HECKE_CONE_ACTION_LARGE"); big && *big) {
            const uint32_t n = static_cast<uint32_t>(std::strtoul(big, nullptr, 10));
            const auto tb = std::chrono::steady_clock::now();
            std::unique_ptr<Cell> cell = build_cell(K, n, 134217757u);
            const auto tc = std::chrono::steady_clock::now();
            std::unique_ptr<Op> op = build_op(K, model, *cell, 2, 1);
            const auto td = std::chrono::steady_clock::now();
            // slot-4 WP-4 (E5 cost): the construction / verify() split.  compute()
            // runs verify() itself; this second, identical call measures its share
            // (the certificate is a pure function of the stored matrices, so the
            // two calls do the same work and neither changes a value).
            const uint32_t bad_large = op->act->certificate().verify();
            const auto te = std::chrono::steady_clock::now();
            const double t_verify = std::chrono::duration<double>(te - td).count();
            const double t_act = std::chrono::duration<double>(td - tc).count();
            log.check(bad_large == 0, "CONE_ACTION_LARGE: the certificate fails on the large cell");
            std::cout << "  CONE_ACTION_LARGE n=" << n << " cone " << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double>(tc - tb).count() << " s, action "
                      << t_act << " s (build " << (t_act - t_verify) << " s + verify " << t_verify
                      << " s), VmHWM=" << std::setprecision(1)
                      << status_mb("VmHWM:") << " MB" << std::endl;
        }

        // ---- the grid ------------------------------------------------------
        const uint32_t ells[] = { 2, 3, 5, 7, 691, 134217757u, 2147483647u };
        for (Elt ell : ells) {
            for (uint32_t n = 0; n <= 16; n += 2) {
                const std::string where = "(" + std::to_string(n) + ", " + std::to_string(ell) + ")";
                std::unique_ptr<Cell> cell = build_cell(K, n, ell);
                ++tally.cells;
                // (p, nu) = (2, 1), (3, 1) everywhere; the wider set on a sub-grid.
                std::vector<std::pair<uint64_t, uint32_t>> pnu = { { 2, 1 }, { 3, 1 } };
                const bool wide = (n % 4 == 0 && (ell == 5 || ell == 7 || ell == 691 || ell == 2 || ell == 3));
                if (wide) { pnu.push_back({ 5, 1 }); pnu.push_back({ 2, 2 }); }
                if (ell <= 7) pnu.push_back({ static_cast<uint64_t>(ell), 1 });   // U_ell at the small ells
                std::vector<std::unique_ptr<Op>> ops;
                std::vector<const Op*> nu1;
                const Op* nu2 = nullptr;
                for (const auto& c : pnu) {
                    ops.push_back(build_op(K, model, *cell, c.first, c.second));
                    const Op* o = ops.back().get();
                    check_op(log, tally, where + " p=" + std::to_string(c.first) + " nu=" + std::to_string(c.second),
                             K, cell->pair->K_bd(), *cell, *o);
                    check_strict_pair(slog, stally,
                                      where + " p=" + std::to_string(c.first) + " nu=" + std::to_string(c.second),
                                      K, model, *cell, *o);
                    check_route3(rlog, rtally, where + " p=" + std::to_string(c.first) + " nu=" + std::to_string(c.second),
                                 K, cell->pair->K_bd(), model, *cell, *o);
                    if (c.second == 1) nu1.push_back(o);
                    else if (c.first == 2 && c.second == 2) nu2 = o;
                }
                check_relations(log, tally, where, K, model, *cell, nu1, nu2, 2);
                tally.tau_ties += ops.front()->taub->num_vertex_ties();
            }
        }

        // ---- A14: the plain constructor refuses BoundaryStrict -------------
        {
            const HeckeCosetTables cs(2, 1, 691, 4);
            bool refused = false;
            try {
                HeckeTauTable bad(K, model, cs, 691, 4, TauVertexPolicy::BoundaryStrict);
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            slog.check(refused, "strict pair: the plain HeckeTauTable constructor accepted BoundaryStrict");
            // The WIDENED integer tier ([OD] OP2) is never needed on the grid
            // (every fill solves in int64), so it is exercised here by forcing
            // it (hecke::ScopedExactTier): tau^s built on the widened tier
            // alone must be the SAME table, slot for slot, as the default one.
            struct WideCase { uint64_t p; uint32_t nu; uint32_t ell; uint32_t n; };
            const WideCase wide_cases[] = { { 2, 1, 691, 10 }, { 3, 1, 2, 4 }, { 2, 2, 5, 6 } };
            uint32_t wide_fills = 0, wide_slots = 0;
            for (const WideCase& wc : wide_cases) {
                const HeckeCosetTables cw(wc.p, wc.nu, wc.ell, wc.n);
                const HeckeBoundaryTauTable tbw(K, model, cw);
                const HeckeTauTable t_auto(K, model, cw, wc.ell, wc.n, tbw);
                std::unique_ptr<HeckeTauTable> t_wide;
                {
                    const hecke::ScopedExactTier force(hecke::ExactTier::ForceWide);
                    t_wide = std::make_unique<HeckeTauTable>(K, model, cw, wc.ell, wc.n, tbw);
                }
                const std::string w = "strict pair widened tier (p=" + std::to_string(wc.p) + ", nu=" +
                                      std::to_string(wc.nu) + ", ell=" + std::to_string(wc.ell) + ")";
                // (a fill with zero right-hand side needs no solve and counts as int64)
                slog.check(t_wide->num_fills_widened() > 0, w + ": the forced widened tier solved no fill");
                slog.check(t_auto.num_fills_widened() == 0 &&
                               t_auto.num_fills_int64() == t_wide->num_fills_int64() + t_wide->num_fills_widened(),
                           w + ": the default tier used the widened solver, or the fill counts differ");
                slog.check(t_wide->certify() == 0 && std::string(t_wide->solve_regime()) == "Z",
                           w + ": the widened-tier table does not certify over Z");
                wide_fills += t_wide->num_fills_widened();
                for (uint32_t q = 0; q < 3; ++q)
                    for (uint32_t s = 0; s < K.get_num_q_simplices(q); ++s)
                        for (uint32_t j = 1; j <= cw.num_cosets(); ++j) {
                            const auto& a = t_auto.tau(j, q, s);
                            const auto& b = t_wide->tau(j, q, s);
                            bool same = a.size() == b.size();
                            for (size_t k = 0; same && k < a.size(); ++k)
                                same = a[k].coeff == b[k].coeff && a[k].gamma == b[k].gamma &&
                                       a[k].sigma_idx == b[k].sigma_idx;
                            slog.check(same, w + ": the widened tier gives a different tau^s slot");
                            ++wide_slots;
                        }
            }
            std::cout << "  strict pair widened tier (forced, [OD] OP2): " << wide_fills << " fills on 3 cells, "
                      << wide_slots << " slots identical to the int64-tier tau^s" << std::endl;
        }

        // ---- A8 / A9 on the distinguished cell (10, 691) and one small ell --
        {
            std::unique_ptr<Cell> c691 = build_cell(K, 10, 691);
            check_independence(log, "(10, 691) p=2 nu=1", K, model, *c691, 2, 1);
            std::unique_ptr<Op> op = build_op(K, model, *c691, 2, 1);
            check_negative_control(log, "(10, 691) p=2 nu=1", K, c691->pair->K_bd(), *c691, *op);
            std::unique_ptr<Cell> c2 = build_cell(K, 4, 2);
            check_independence(log, "(4, 2) p=3 nu=1", K, model, *c2, 3, 1);
            std::unique_ptr<Cell> c5 = build_cell(K, 6, 5);
            check_independence(log, "(6, 5) p=2 nu=2", K, model, *c5, 2, 2);
        }

        std::cout << "  asserted: r^q A = A_bd r^q on " << tally.restriction_recorded_ok << " / "
                  << tally.restriction_recorded_total << " (cell, q) pairs (asserted at every ell since slot-5 WP-4/OI-3; "
                  << "proved by clause (i) with no hypothesis on ell); "
                  << "obl:coneexact(vi) on " << tally.relations_recorded_ok << " / " << tally.relations_recorded_total
                  << " instances (" << tally.commutator_pairs << " commutator pairs, " << tally.recursion_degrees
                  << " recursion degrees)" << std::endl;
        std::cout << "  recorded: eta_zero true on " << tally.eta_zero_true << " / " << tally.ops
                  << " operators, cochain_commutes true on " << tally.cochain_commutes_true << " / " << tally.ops
                  << "; the eq:conesign bracket is zero on C_eq in " << tally.residual_degrees << " / "
                  << tally.residual_degrees << " degrees and on the FULL FLAT space in "
                  << (tally.residual_degrees - tally.residual_flat_nonzero) << " of them (recorded: the transport "
                  << "formula uses phi's equivariance, so the flat identity is not claimed); Dcone Phi = Phi Dcone "
                  << "holds FLAT on " << tally.phi_flat_identity_true << " / " << tally.ops << " operators"
                  << "; tau_bd nearest-vertex TIES over the grid: " << tally.tau_ties
                  << " (measured: ties DO occur, so a tie rule would also be non-vacuous; the shipped second policy is "
                     "the non-tie NextLineVertex, which moves the table on EVERY slot -- the arm asserts that it does)"
                  << std::endl;
        std::cout << "  strict pair ([C, prop:coneroutes](1), [C, obl:coneexact](vii)): A_c^s == A_c entrywise on "
                  << stally.agree << " / " << stally.ops << " operators (" << stally.degrees << " degrees); asserted on "
                  << stally.asserted << " (6 in R^x), recorded on " << stally.recorded << " (ell in {2,3}): "
                  << stally.recorded_disagree << " recorded disagreement(s), " << stally.not_built
                  << " not built; eta^s == 0 on " << stally.built << " / " << stally.built
                  << "; tau^s != tau on " << stally.tau_differs << "; solve_regime Z: " << stally.fills_int64
                  << " int64 fills, " << stally.fills_widened << " widened, 0 mod-ell; cochain_commutes on "
                  << stally.cochain_commutes << " / " << stally.built << std::endl;
        for (const std::string& note : stally.recorded_notes) std::cout << "    strict pair RECORDED: " << note << std::endl;

        // ---- A15: the relative sub-complex's legacy arm on three cells -----
        if (oc::legacy::RunLegacyArm("HeckeConeActionValidation", "A15-route3-legacy")) {   // (S24 CL-L2)
#if OC_ENABLE_LEGACY
            const struct { uint32_t n, ell; uint64_t p; uint32_t nu; } lc[] = { { 10, 691, 2, 1 }, { 4, 2, 3, 1 }, { 6, 3, 2, 2 } };
            for (const auto& c : lc) {
                std::unique_ptr<Cell> cl = build_cell(K, c.n, c.ell);
                std::unique_ptr<Op> ol = build_op(K, model, *cl, c.p, c.nu);
                check_route3_legacy(rlog, rtally, "(" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ") p=" +
                                                      std::to_string(c.p) + " nu=" + std::to_string(c.nu), *ol);
            }
#endif
        }
        std::cout << "  route 3 ([C, prop:coneroutes](2), [C, obl:coneexact](vii)): M A' M^{-1} == A_c entrywise on "
                  << rtally.agree << " / " << rtally.ops << " operators (" << rtally.degrees
                  << " degrees), ASSERTED at every ell; dim H^q(ker r) summed over operators = [";
        for (size_t q = 0; q < rtally.dim_H_rel_sum.size(); ++q) std::cout << (q ? "," : "") << rtally.dim_H_rel_sum[q];
        std::cout << "] (== dim H^q_c); Theta' != Theta on C_eq on " << rtally.theta_prime_differs << " / " << rtally.ops
                  << ", E != 0 on " << rtally.e_nonzero << "; the UNCORRECTED Theta moves H(ker r) off ker r on "
                  << rtally.naive_leaves_ker_r << " / " << rtally.ops << "; relative legacy arm identical on "
                  << rtally.legacy_cells << " cells; A_c^q tier B in the unit on " << rtally.tier_b_A_c << " objects"
                  << std::endl;
    } catch (const std::exception& e) {
        log.fail(std::string("exception: ") + e.what());
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "CONE_ACTION cells=" << tally.cells << " operators=" << tally.ops << " ell23_operators=" << tally.ell23_ops
              << " record_r_asserted=" << tally.record_r_asserted << "/" << tally.ops
              << " ell23_record_r_asserted=" << tally.ell23_record_r_asserted << "/" << tally.ell23_ops
              << " item12_tamper_arms=" << tally.item12_arms
              << " checks=" << log.checks << " failures=" << log.failures << " time=" << std::fixed << std::setprecision(2)
              << secs << "s" << std::endl;
    std::cout << "CONE_STRICT_PAIR operators=" << stally.ops << " asserted=" << stally.asserted
              << " recorded=" << stally.recorded << " agree=" << stally.agree
              << " recorded_disagreements=" << stally.recorded_disagree << " not_built=" << stally.not_built
              << " fills_int64=" << stally.fills_int64 << " fills_widened=" << stally.fills_widened
              << " checks=" << slog.checks << " failures=" << slog.failures << " time=" << std::fixed
              << std::setprecision(2) << stally.seconds << "s" << std::endl;
    std::cout << "CONE_ROUTE3 operators=" << rtally.ops << " agree=" << rtally.agree << " degrees=" << rtally.degrees
              << " theta_prime_differs=" << rtally.theta_prime_differs << " naive_leaves_ker_r=" << rtally.naive_leaves_ker_r
              << " legacy_cells=" << rtally.legacy_cells << " checks=" << rlog.checks << " failures=" << rlog.failures
              << " time=" << std::fixed << std::setprecision(2) << rtally.seconds << "s" << std::endl;
    return log.failures + slog.failures + rlog.failures;
}
