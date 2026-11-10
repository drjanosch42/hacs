// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-6 (phase E5-3) -- the cone Hecke action.
// See HeckeConeAction.h for the contract references and the design notes.

#include "HeckeConeAction.h"
#include "../../ModArithmetic/Kernels.h"
#include "CoefficientRing.h"

#include "FpPoly.h"
#include "HeckeModelDescriptor.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../LocalSystemUtils.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"
#include "HeckeSettings.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace hecke {

namespace {

using Elt = uint32_t;
using engine::ModCtx;

Elt coeff_mod32(int64_t c, Elt ell) {
    const int64_t m = static_cast<int64_t>(ell);
    int64_t r = c % m;
    if (r < 0) r += m;
    return static_cast<Elt>(r);
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

// A * B with the empty shapes handled (FlatMatrix::operator* wants agreeing
// inner dimensions; a 0-column / 0-row factor gives the zero matrix).
FlatMatrix mul(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_cols() != B.get_rows())
        throw std::runtime_error("HeckeConeAction: inner dimension mismatch in a product.");
    if (A.get_rows() == 0 || B.get_cols() == 0 || A.get_cols() == 0)
        return FlatMatrix::CreateZero(A.get_rows(), B.get_cols(), A.get_p());
    return A * B;
}

FlatMatrix add(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols())
        throw std::runtime_error("HeckeConeAction: shape mismatch in a sum.");
    if (A.get_rows() == 0 || A.get_cols() == 0) return A;
    const ModCtx<Elt> ctx(A.get_p());
    std::vector<Elt> d(static_cast<size_t>(A.get_rows()) * A.get_cols());
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            d[static_cast<size_t>(i) * A.get_cols() + j] = ctx.add(A.get_entry(i, j), B.get_entry(i, j));
    return FlatMatrix(A.get_rows(), A.get_cols(), A.get_p(), d);
}

FlatMatrix sub(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols())
        throw std::runtime_error("HeckeConeAction: shape mismatch in a difference.");
    if (A.get_rows() == 0 || A.get_cols() == 0) return A;
    const ModCtx<Elt> ctx(A.get_p());
    std::vector<Elt> d(static_cast<size_t>(A.get_rows()) * A.get_cols());
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            d[static_cast<size_t>(i) * A.get_cols() + j] = ctx.sub(A.get_entry(i, j), B.get_entry(i, j));
    return FlatMatrix(A.get_rows(), A.get_cols(), A.get_p(), d);
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

uint64_t hash_flat(const FlatMatrix& M) {
    engine::FnvHasher h;
    h.feed(static_cast<uint64_t>(M.get_rows()));
    h.feed(static_cast<uint64_t>(M.get_cols()));
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) h.feed(static_cast<uint64_t>(M.get_entry(i, j)));
    return h.value();
}

uint64_t hash_matrix(const MatrixT<Elt>& M) {
    engine::FnvHasher h;
    h.feed(static_cast<uint64_t>(M.rows));
    h.feed(static_cast<uint64_t>(M.cols));
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) h.feed(static_cast<uint64_t>(M.data[i][j]));
    return h.value();
}

void emit_flat_json(std::ostringstream& oss, const FlatMatrix& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.get_rows(); ++r) {
        if (r) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.get_cols(); ++c) oss << (c ? "," : "") << M.get_entry(r, c);
        oss << "]";
    }
    oss << "]";
}

void emit_matrix_json(std::ostringstream& oss, const MatrixT<Elt>& M) {
    oss << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        if (r) oss << ",";
        oss << "[";
        for (uint32_t c = 0; c < M.cols; ++c) oss << (c ? "," : "") << M.data[r][c];
        oss << "]";
    }
    oss << "]";
}

} // namespace

// ===========================================================================
// ConeActionCertificate
// ===========================================================================

uint32_t ConeActionCertificate::verify() const {
    std::vector<std::string> f;
    return verify(f);
}

std::vector<uint32_t> ConeActionCertificate::failing_items() const {
    std::vector<std::string> f;
    (void)verify(f);
    std::vector<uint32_t> items;
    for (const std::string& s : f) {
        const size_t c = s.find(':');
        if (c == std::string::npos || s.size() < 2 || s[0] != 'C') continue;
        const uint32_t n = static_cast<uint32_t>(std::strtoul(s.c_str() + 1, nullptr, 10));
        bool seen = false;
        for (uint32_t x : items) seen = seen || (x == n);
        if (!seen) items.push_back(n);
    }
    return items;
}

uint32_t ConeActionCertificate::verify(std::vector<std::string>& failures) const {
    const size_t before = failures.size();
    auto fail = [&](uint32_t item, const std::string& msg) {
        failures.push_back("C" + std::to_string(item) + ": " + msg);
    };
    const uint32_t Q = q_max;
    const Elt p = ell;

    // ---- C10 structural: the shapes and the block form of Phi --------------
    auto sized = [&](const std::vector<FlatMatrix>& v) { return v.size() >= static_cast<size_t>(Q) + 1; };
    if (!sized(Theta) || !sized(Theta_bd) || !sized(h) || !sized(Phi) || !sized(Ceq) || !sized(Z) ||
        !sized(H) || !sized(P) || flat_dim.size() < static_cast<size_t>(Q) + 1 ||
        bulk_part_dim.size() < static_cast<size_t>(Q) + 1 || boundary_part_dim.size() < static_cast<size_t>(Q) + 1 ||
        A_c.size() < static_cast<size_t>(Q) + 1) {
        fail(10, "a stored family is shorter than q_max + 1");
        return static_cast<uint32_t>(failures.size() - before);
    }
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t nb = bulk_part_dim[q], nd = boundary_part_dim[q], fd = flat_dim[q];
        if (nb + nd != fd) { fail(10, "flat_dim != bulk + boundary at q = " + std::to_string(q)); continue; }
        if (Theta[q].get_rows() != nb || Theta[q].get_cols() != nb) fail(10, "Theta shape at q = " + std::to_string(q));
        if (Theta_bd[q].get_rows() != nd || Theta_bd[q].get_cols() != nd) fail(10, "Theta^bd shape at q = " + std::to_string(q));
        if (h[q].get_rows() != nd || h[q].get_cols() != nb) fail(10, "h shape at q = " + std::to_string(q));
        if (Phi[q].get_rows() != fd || Phi[q].get_cols() != fd) { fail(10, "Phi shape at q = " + std::to_string(q)); continue; }
        // Phi == [[Theta, 0], [h, Theta^bd]] entrywise.
        bool ok = true;
        for (uint32_t i = 0; i < fd && ok; ++i) {
            for (uint32_t j = 0; j < fd; ++j) {
                const Elt v = Phi[q].get_entry(i, j);
                Elt w = 0;
                if (i < nb && j < nb) w = Theta[q].get_entry(i, j);
                else if (i < nb) w = 0;
                else if (j < nb) w = h[q].get_entry(i - nb, j);
                else w = Theta_bd[q].get_entry(i - nb, j - nb);
                if (v != w) { ok = false; break; }
            }
        }
        if (!ok) fail(10, "Phi is not [[Theta, 0], [h, Theta^bd]] at q = " + std::to_string(q));
    }

    // ---- C1 obl:eta(v): h == the coset transport of the stored eta ---------
    if (eta_table == nullptr || cosets == nullptr) {
        fail(1, "no eta-table to re-derive eq:derivedh from");
    } else {
        const Mat22& g = cosets->g();
        for (uint32_t q = 0; q <= Q; ++q) {
            const uint32_t nb = bulk_part_dim[q], nd = boundary_part_dim[q];
            if (nd == 0 || nb == 0) {
                if (!flat_is_zero(h[q])) fail(1, "h != 0 in a degree with an empty side, q = " + std::to_string(q));
                continue;
            }
            if (q == 0 || q - 1 >= eta_num_degrees || q - 1 >= boundary_simplices.size()) {
                if (!flat_is_zero(h[q])) fail(1, "h != 0 in a degree eta is not built in, q = " + std::to_string(q));
                continue;
            }
            const uint32_t qb = q - 1;
            const ModCtx<Elt> ctx(p);
            std::vector<Elt> data(static_cast<size_t>(nd) * nb, 0);
            std::map<std::array<int64_t, 5>, FlatMatrix> cache;
            const std::vector<uint32_t>& bslots = boundary_simplices[qb];
            if (static_cast<uint64_t>(bslots.size()) * d_V != nd) {
                fail(1, "the boundary slot list does not match boundary_part_dim at q = " + std::to_string(q));
                continue;
            }
            bool shape_ok = true;
            for (uint32_t s = 0; s < bslots.size(); ++s) {
                for (uint32_t j = 1; j <= m; ++j) {
                    for (const ChainTerm& ct : eta_table->eta(j, qb, bslots[s])) {
                        const Elt cmod = coeff_mod32(ct.coeff, p);
                        if (cmod == 0) continue;
                        const uint64_t col0 = static_cast<uint64_t>(ct.sigma_idx) * d_V;
                        if (col0 + d_V > nb) { shape_ok = false; break; }
                        const std::array<int64_t, 5> key{ static_cast<int64_t>(j), ct.gamma.a, ct.gamma.b,
                                                          ct.gamma.c, ct.gamma.d };
                        auto it = cache.find(key);
                        if (it == cache.end()) {
                            const std::array<Elt, 4> r = (cosets->delta(j) * ct.gamma).reduced_mod(p);
                            const FlatMatrix g_fm(2, 2, p, std::vector<Elt>{ r[0], r[1], r[2], r[3] });
                            it = cache.emplace(key, ComputeSymmetricPowerMatrixFM(g_fm, n_sym)).first;
                        }
                        const FlatMatrix& B = it->second;
                        for (uint32_t rr = 0; rr < d_V; ++rr)
                            for (uint32_t cc = 0; cc < d_V; ++cc) {
                                const Elt v = B.get_entry(rr, cc);
                                if (v == 0) continue;
                                Elt& cell = data[(static_cast<size_t>(s) * d_V + rr) * nb + col0 + cc];
                                cell = ctx.add(cell, ctx.mul(cmod, v));
                            }
                    }
                    if (!shape_ok) break;
                }
                if (!shape_ok) break;
            }
            (void)g;
            if (!shape_ok) { fail(1, "an eta term names a simplex outside C^q(K) at q = " + std::to_string(q)); continue; }
            const FlatMatrix rebuilt(nd, nb, p, data);
            if (!flat_equal(rebuilt, h[q]))
                fail(1, "the stored h^" + std::to_string(q) + " differs from eq:derivedh recomputed from the eta-table");
        }
    }

    // ---- C2 obl:coneexact(i): d h + h d == Theta^bd r - r Theta ------------
    // On the equivariant sub-complex (the contract's complex C^*_eq); the flat
    // identity is recorded in phi_flat_identity and is not an item.
    for (uint32_t q = 0; q <= Q; ++q) {
        // target C^q(K_bd; V): only meaningful when the boundary degree q exists.
        if (q + 1 > Q) continue;                      // no h^{q+1} stored
        const uint32_t tgt = boundary_part_dim[q + 1]; // = N_q(K_bd) d_V
        const uint32_t nb = bulk_part_dim[q];
        if (tgt == 0 || nb == 0) continue;
        if (q >= d_bd.size() || q >= d_bulk.size() || q >= r_flat.size()) { fail(2, "missing d / r at q = " + std::to_string(q)); continue; }
        // (d h)^q = d_bd_{q-1} h^q ; h^0 is empty so this term is 0 at q = 0.
        FlatMatrix term = FlatMatrix::CreateZero(tgt, nb, p);
        if (boundary_part_dim[q] > 0 && d_bd[q > 0 ? q - 1 : 0].get_rows() == tgt && q > 0)
            term = mul(d_bd[q - 1], h[q]);
        // (h d)^q = h^{q+1} d^K_q
        const FlatMatrix hd = mul(h[q + 1], d_bulk[q]);
        FlatMatrix lhs = add(term, hd);
        // r^q Theta^q - Theta^bd_q r^q ; Theta^bd in BOUNDARY degree q is the
        // stored Theta_bd[q + 1] (cone degree q + 1 carries boundary degree q).
        const FlatMatrix rTh = mul(r_flat[q], Theta[q]);
        const FlatMatrix Thr = mul(Theta_bd[q + 1], r_flat[q]);
        const FlatMatrix rhs = sub(Thr, rTh);
        // ON THE EQUIVARIANT SUB-COMPLEX (the contract's complex; see the
        // header note "WHY C^*_eq AND NOT THE FULL FLAT SPACE").
        if (q >= Ceq_bulk.size() || Ceq_bulk[q].get_rows() != nb) { fail(2, "missing C_eq(bulk) at q = " + std::to_string(q)); continue; }
        if (!flat_equal(mul(lhs, Ceq_bulk[q]), mul(rhs, Ceq_bulk[q])))
            fail(2, "d h + h d != Theta^bd r - r Theta on C^q_eq at q = " + std::to_string(q));
    }

    // ---- C3 Dcone^2 = 0 ---------------------------------------------------
    for (uint32_t q = 0; q + 1 <= Q && q + 1 < D.size(); ++q) {
        if (D[q].get_rows() == 0 || D[q].get_cols() == 0) continue;
        if (D[q + 1].get_cols() != D[q].get_rows()) { fail(3, "D shapes do not compose at q = " + std::to_string(q)); continue; }
        if (!flat_is_zero(mul(D[q + 1], D[q]))) fail(3, "D^2 != 0 at q = " + std::to_string(q));
    }

    // ---- C4 Dcone Phi == Phi Dcone on the equivariant sub-complex ----------
    for (uint32_t q = 0; q <= Q; ++q) {
        if (q >= D.size() || q >= Ceq.size()) { fail(4, "missing D / Ceq at q = " + std::to_string(q)); continue; }
        const FlatMatrix& Cq = Ceq[q];
        if (Cq.get_cols() == 0) continue;
        const FlatMatrix PhiC = mul(Phi[q], Cq);
        const FlatMatrix left = mul(D[q], PhiC);
        const FlatMatrix DC = mul(D[q], Cq);
        FlatMatrix right = FlatMatrix::CreateZero(left.get_rows(), left.get_cols(), p);
        if (q + 1 <= Q) right = mul(Phi[q + 1], DC);
        else if (DC.get_rows() != 0 && !flat_is_zero(DC)) { fail(4, "the scaffolding degree carries a non-zero D Ceq"); continue; }
        if (!flat_equal(left, right)) fail(4, "Dcone Phi != Phi Dcone on C_eq at q = " + std::to_string(q));
    }

    // ---- C5 obl:coneexact(ii): Dcone Phi z == 0 on the stored cocycles -----
    for (uint32_t q = 0; q <= Q; ++q) {
        if (Z[q].get_cols() == 0) continue;
        if (Z[q].get_rows() != flat_dim[q]) { fail(5, "Z shape at q = " + std::to_string(q)); continue; }
        const FlatMatrix y = mul(Phi[q], Z[q]);
        if (q >= D.size()) { fail(5, "missing D at q = " + std::to_string(q)); continue; }
        if (!flat_is_zero(mul(D[q], y))) fail(5, "Dcone Phi z != 0 on a stored cocycle at q = " + std::to_string(q));
    }

    // ---- C6 obl:coneexact(ii): A_c == P Phi H -----------------------------
    for (uint32_t q = 0; q <= Q; ++q) {
        if (H[q].get_rows() != flat_dim[q] || P[q].get_cols() != flat_dim[q]) { fail(6, "H / P shape at q = " + std::to_string(q)); continue; }
        const MatrixT<Elt> rebuilt = matrix_of_flat(mul(P[q], mul(Phi[q], H[q])));
        if (!matrices_equal(rebuilt, A_c[q])) fail(6, "A_c^" + std::to_string(q) + " != P Phi H recomputed");
    }

    // ---- C7 / C8 / C9 obl:coneexact(iii): the LES intertwines --------------
    for (uint32_t q = 0; q <= Q; ++q) {
        if (q < pi_built.size() && pi_built[q] && q < A_bulk.size()) {
            const MatrixT<Elt>& A = A_bulk[q];
            const FlatMatrix Af = matrix_of_flat(pi[q]).rows == 0 ? pi[q] : pi[q];  // shape carrier
            FlatMatrix Am(A.rows, A.cols, p);
            for (uint32_t i = 0; i < A.rows; ++i) for (uint32_t j = 0; j < A.cols; ++j) Am.set_entry(i, j, A.data[i][j]);
            FlatMatrix Acm(A_c[q].rows, A_c[q].cols, p);
            for (uint32_t i = 0; i < A_c[q].rows; ++i) for (uint32_t j = 0; j < A_c[q].cols; ++j) Acm.set_entry(i, j, A_c[q].data[i][j]);
            if (Af.get_rows() == A.rows && Af.get_cols() == A_c[q].rows) {
                if (!flat_equal(mul(pi[q], Acm), mul(Am, pi[q])))
                    fail(7, "pi^" + std::to_string(q) + " A_c != A pi^" + std::to_string(q));
            } else {
                fail(7, "pi^" + std::to_string(q) + " shape does not match (A, A_c)");
            }
        }
        if (q < r_built.size() && r_built[q] && q < A_bulk.size() && q < A_bd.size()) {
            const MatrixT<Elt>& A = A_bulk[q];
            const MatrixT<Elt>& Ab = A_bd[q];
            FlatMatrix Am(A.rows, A.cols, p), Abm(Ab.rows, Ab.cols, p);
            for (uint32_t i = 0; i < A.rows; ++i) for (uint32_t j = 0; j < A.cols; ++j) Am.set_entry(i, j, A.data[i][j]);
            for (uint32_t i = 0; i < Ab.rows; ++i) for (uint32_t j = 0; j < Ab.cols; ++j) Abm.set_entry(i, j, Ab.data[i][j]);
            const bool ok = (r[q].get_rows() == Ab.rows && r[q].get_cols() == A.rows) &&
                            flat_equal(mul(r[q], Am), mul(Abm, r[q]));
            // [C, obl:coneexact](iii): the r-half is asserted at EVERY ell (slot-5 WP-4,
            // OI-3). Clause (i) (dh + hd = Theta^bd r - r Theta on the stored C_eq basis,
            // asserted at every ell with no six-in-R^x carve-out) already proves this identity
            // on cohomology with no hypothesis on ell: for an equivariant cocycle z (dz = 0),
            // r Theta z - Theta^bd r z = -(dh + hd)z = -d(hz), a coboundary in the boundary
            // complex, so r^q A^{(q)} = A_bd^{(q)} r^q follows from clause (i) and clause
            // (ii)'s assembly alone. The former six_invertible carve-out mirrored
            // obl:taubdry(vii), which has no h/eta and so has no proof at ell in {2, 3} --
            // that asymmetry is why the cone chapter's clause (iii) may assert what the
            // boundary chapter's obl:taubdry(vii) still only records. Measured 424/424
            // (cell, q) pairs, the 74 operators at ell in {2, 3} included.
            if (!ok)
                fail(8, "r^" + std::to_string(q) + " A != A_bd r^" + std::to_string(q));
        }
        if (q < iota_built.size() && iota_built[q] && q < A_bd.size() && q + 1 <= Q) {
            const MatrixT<Elt>& Ab = A_bd[q];
            FlatMatrix Abm(Ab.rows, Ab.cols, p);
            for (uint32_t i = 0; i < Ab.rows; ++i) for (uint32_t j = 0; j < Ab.cols; ++j) Abm.set_entry(i, j, Ab.data[i][j]);
            FlatMatrix Ac1(A_c[q + 1].rows, A_c[q + 1].cols, p);
            for (uint32_t i = 0; i < A_c[q + 1].rows; ++i) for (uint32_t j = 0; j < A_c[q + 1].cols; ++j) Ac1.set_entry(i, j, A_c[q + 1].data[i][j]);
            if (iota[q].get_rows() == A_c[q + 1].rows && iota[q].get_cols() == Ab.rows) {
                if (!flat_equal(mul(iota[q], Abm), mul(Ac1, iota[q])))
                    fail(9, "iota^" + std::to_string(q) + " A_bd != A_c iota^" + std::to_string(q));
            } else {
                fail(9, "iota^" + std::to_string(q) + " shape does not match (A_bd, A_c)");
            }
        }
    }

    // ---- C11 obl:eta(vi): eta_zero => cochain_commutes ---------------------
    if (eta_zero && !cochain_commutes)
        fail(11, "eta_zero holds but r Theta != Theta^bd r (a contradiction in the stored data)");

    // ---- C12 record honesty ([C] rule A6; [OD] 2026-09-22 Q2) --------------
    // Item 8 above carries NO six_invertible guard, so the r-half of
    // obl:coneexact(iii) is asserted at every ell.  A certificate whose
    // emitted flag says it was not asserted misreports its own run, which is
    // exactly what rule A6 forbids.  Unconditional -- deliberately NOT behind
    // any shape or built-ness gate, so an otherwise empty certificate still
    // fails it.
    if (!restriction_equivariance_asserted)
        fail(12, "restriction_equivariance_asserted is false, but item 8 asserts the r-half at every ell");

    return static_cast<uint32_t>(failures.size() - before);
}

// ===========================================================================
// ConeRelationsCertificate -- obl:coneexact(vi)
// ===========================================================================

uint32_t ConeRelationsCertificate::verify() const {
    std::vector<std::string> f;
    return verify(f);
}

std::vector<uint32_t> ConeRelationsCertificate::failing_items() const {
    std::vector<std::string> f;
    (void)verify(f);
    std::vector<uint32_t> items;
    for (const std::string& s : f) {
        if (s.size() < 2 || s[0] != 'R') continue;
        const uint32_t n = static_cast<uint32_t>(std::strtoul(s.c_str() + 1, nullptr, 10));
        bool seen = false;
        for (uint32_t x : items) seen = seen || (x == n);
        if (!seen) items.push_back(n);
    }
    return items;
}

uint32_t ConeRelationsCertificate::verify(std::vector<std::string>& failures) const {
    const size_t before = failures.size();
    commutator_pairs = 0; commutator_failures = 0;
    recursion_degrees = 0; recursion_failures = 0;
    const Elt p = ell;
    auto as_flat = [&](const MatrixT<Elt>& A) {
        FlatMatrix M(A.rows, A.cols, p);
        for (uint32_t i = 0; i < A.rows; ++i) for (uint32_t j = 0; j < A.cols; ++j) M.set_entry(i, j, A.data[i][j]);
        return M;
    };

    // R1: [A_c(T_p), A_c(T_p')] = 0.
    for (size_t a = 0; a + 1 < A_c.size(); ++a) {
        for (size_t b = a + 1; b < A_c.size(); ++b) {
            for (uint32_t q = 0; q <= q_max && q < A_c[a].size() && q < A_c[b].size(); ++q) {
                const MatrixT<Elt>& X = A_c[a][q];
                const MatrixT<Elt>& Y = A_c[b][q];
                if (X.rows != Y.rows || X.cols != Y.cols) {
                    failures.push_back("R1: A_c shapes differ between two primes at q = " + std::to_string(q));
                    continue;
                }
                if (X.rows == 0) continue;
                ++commutator_pairs;
                const FlatMatrix Xf = as_flat(X), Yf = as_flat(Y);
                if (!flat_is_zero(sub(mul(Xf, Yf), mul(Yf, Xf)))) {
                    ++commutator_failures;
                    if (six_invertible)
                        failures.push_back("R1: [A_c(T_" + std::to_string(primes[a]) + "), A_c(T_" +
                                           std::to_string(primes[b]) + ")] != 0 at q = " + std::to_string(q));
                }
            }
        }
    }

    // R2: A_c(T_{p^2}) = A_c(T_p)^2 - p^{n+1} I, with the FULL
    //     T_{p^2} = T_{Delta'_{p^2}} + p^n I ([C, sec:primfull]).
    if (recursion_present) {
        const ModCtx<Elt> ctx(p);
        Elt pn = 1;
        const Elt pmod = static_cast<Elt>(recursion_prime % p);
        for (uint32_t i = 0; i < n_sym; ++i) pn = ctx.mul(pn, pmod);
        const Elt pn1 = ctx.mul(pn, pmod);
        for (uint32_t q = 0; q <= q_max && q < A_c_p.size() && q < A_c_primitive_p2.size(); ++q) {
            const MatrixT<Elt>& Ap = A_c_p[q];
            const MatrixT<Elt>& Apr = A_c_primitive_p2[q];
            if (Ap.rows != Apr.rows || Ap.cols != Apr.cols) {
                failures.push_back("R2: A_c(T_p) and A_c(T'_{p^2}) shapes differ at q = " + std::to_string(q));
                continue;
            }
            if (Ap.rows == 0) continue;
            ++recursion_degrees;
            const FlatMatrix Apf = as_flat(Ap);
            FlatMatrix full = as_flat(Apr);            // T'_{p^2}
            for (uint32_t i = 0; i < full.get_rows(); ++i)  // + p^n I  ==> the FULL T_{p^2}
                full.set_entry(i, i, ctx.add(full.get_entry(i, i), pn));
            FlatMatrix rhs = mul(Apf, Apf);            // A_c(T_p)^2
            for (uint32_t i = 0; i < rhs.get_rows(); ++i)   // - p^{n+1} I
                rhs.set_entry(i, i, ctx.sub(rhs.get_entry(i, i), pn1));
            if (!flat_equal(full, rhs)) {
                ++recursion_failures;
                if (six_invertible)
                    failures.push_back("R2: A_c(T_{p^2}) != A_c(T_p)^2 - p^{n+1} I at q = " + std::to_string(q));
            }
        }
    }
    return static_cast<uint32_t>(failures.size() - before);
}

// ===========================================================================
// HeckeConeAction
// ===========================================================================

HeckeConeAction::HeckeConeAction(const EquivariantMappingCone* cone, const HeckeOperatorOnKT<uint32_t>* bulk_op,
                                 const HeckeBoundaryOperatorT<uint32_t>* bd_op,
                                 const HeckePairHomotopyT<uint32_t>* eta)
    : cone_(cone), bulk_(bulk_op), bd_(bd_op), eta_(eta) {
    // (slot-9 WP-4; E8 item 4b) Route 1 over the Z/ell^k IN FORCE: the cone and
    // both operators over that ring.  Every other combination keeps slot 8's refusal.
    {
        const uint32_t kc = cone ? cone->coefficient_ring_k() : 1u, kb = bulk_op ? bulk_op->coefficient_ring_k() : 1u;
        const uint32_t kd = bd_op ? bd_op->coefficient_ring_k() : 1u;
        if (kc >= 2 && kc == kb && kc == kd && CoefficientRingK() == kc) {
            if (eta_ == nullptr) throw std::invalid_argument("HeckeConeAction: null input");
            if (!cone_->computed()) throw std::runtime_error("HeckeConeAction: the cone is not computed");
            ring_k_ = kc;
            module_ = std::make_unique<ModuleConeActionT<uint32_t>>(&cone_->module_cone(), bulk_, bd_, eta_);
            ell_ = cone_->ell();
            d_V_ = cone_->d_V();
            n_sym_ = d_V_ - 1;
            q_max_ = cone_->q_max();
            m_ = eta_->num_cosets();
            return;
        }
        // (slot-8 WP-3) [E] sec:ring (b): the cone action is a surface; its Z/ell^k form is item 4b's.
        RequireFieldCoefficientRing("HeckeConeAction", std::max(kc, std::max(kb, kd)));
    }
    if (cone_ == nullptr || bulk_ == nullptr || bd_ == nullptr || eta_ == nullptr)
        throw std::invalid_argument("HeckeConeAction: null input");
    if (!cone_->computed()) throw std::runtime_error("HeckeConeAction: the cone is not computed");
    ell_ = cone_->ell();
    d_V_ = cone_->d_V();
    if (d_V_ == 0) throw std::runtime_error("HeckeConeAction: representation dimension 0");
    n_sym_ = d_V_ - 1;
    q_max_ = cone_->q_max();
    if (bd_->ell() != ell_ || bd_->d_V() != d_V_)
        throw std::runtime_error("HeckeConeAction: the boundary operator's (ell, d_V) differ from the cone's");
    if (&bd_->cohomology() != &cone_->boundary())
        throw std::runtime_error("HeckeConeAction: the boundary operator was built on a different boundary cohomology");
    if (&eta_->cosets() != &bd_->cosets())
        throw std::runtime_error("HeckeConeAction: eta and the boundary operator use different coset tables");
    if (eta_->tau_boundary_table_fnv64() != bd_->taub().table_fnv64())
        throw std::runtime_error("HeckeConeAction: eta was built against a different tau_bd table");
    // The eta-PROVENANCE check ([X] v1.10 open item 9, slot-6 WP-5 finding;
    // hecke-extensions slot 7, WP-11): the above pins eta's tau_bd half to
    // bd_'s; this pins the tau half to bulk_'s, by the SAME pointer-identity
    // test HeckeConeStrictPairRoute::compute() already ran on the normative
    // triple after the fact. Without it, this constructor would silently
    // accept an eta certified against SOME tau (a cochain with "the right
    // boundary", obl:eta(i)-(iv)) that is not bulk_'s -- i.e. not the eta~ of
    // [C, def:coneaction] relative to THIS cone action -- and only the later,
    // far more expensive compute() residual check (or, at ell in {2,3}, no
    // check at all along that particular route) would ever notice.
    if (&eta_->tau() != &bulk_->tau())
        throw std::runtime_error("HeckeConeAction: eta was not built on the bulk operator's tau");
    m_ = eta_->num_cosets();
    // The scaffolding degree must be empty: this unit builds Phi in degrees
    // 0 .. q_max, and the cochain-map identity at q_max needs Cc^{q_max+1} = 0.
    if (cone_->flat_dim(q_max_ + 1) != 0)
        throw std::runtime_error("HeckeConeAction: the cone's scaffolding degree is non-empty (a truncated model is E9's)");
}

HeckeConeAction::~HeckeConeAction() = default;

uint64_t HeckeConeAction::p_hecke() const { return bd_->p_hecke(); }
uint32_t HeckeConeAction::nu() const { return bd_->nu(); }
uint64_t HeckeConeAction::N() const { return bd_->N(); }

void HeckeConeAction::check_degree(uint32_t q) const {
    if (q > q_max_) throw std::out_of_range("HeckeConeAction: degree out of range.");
}

void HeckeConeAction::require_computed(const char* who) const {
    if (!computed_) throw std::runtime_error(std::string("HeckeConeAction::") + who + ": compute() not run.");
}

FlatMatrix HeckeConeAction::rho_of_exact(const Mat22& gamma_exact) const {
    const std::array<Elt, 4> r = gamma_exact.reduced_mod(ell_);
    const FlatMatrix g_fm(2, 2, ell_, std::vector<Elt>{ r[0], r[1], r[2], r[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym_);
}

bool HeckeConeAction::boundary_degree(uint32_t q, uint32_t& qb_out) const {
    if (q == 0) return false;
    qb_out = q - 1;
    return qb_out < eta_->num_degrees() && cone_->boundary_part_dim(q) > 0 && cone_->bulk_part_dim(q) > 0;
}

FlatMatrix HeckeConeAction::coboundary_tensor(const SimplicialComplex& C, uint32_t q, uint32_t dV, uint32_t ell) {
    const uint32_t Ns = C.get_num_q_simplices(q);
    const uint32_t Nt = (q + 1 <= C.get_dimension()) ? C.get_num_q_simplices(q + 1) : 0;
    if (Ns == 0 || Nt == 0) return FlatMatrix::CreateZero(Nt * dV, Ns * dV, ell);
    const FlatMatrix D1 = C.coboundary_operator(q, ell);
    const uint32_t cols = Ns * dV;
    std::vector<Elt> d(static_cast<size_t>(Nt) * dV * cols, 0);
    for (uint32_t i = 0; i < Nt; ++i)
        for (uint32_t j = 0; j < Ns; ++j) {
            const Elt s = D1.get_entry(i, j);
            if (s == 0) continue;
            for (uint32_t k = 0; k < dV; ++k) d[static_cast<size_t>(i * dV + k) * cols + (j * dV + k)] = s;
        }
    return FlatMatrix(Nt * dV, cols, ell, d);
}

FlatMatrix HeckeConeAction::assemble_h(uint32_t q, bool fused) const {
    const uint32_t rows = cone_->boundary_part_dim(q), cols = cone_->bulk_part_dim(q);
    if (rows == 0 || cols == 0) return FlatMatrix::CreateZero(rows, cols, ell_);
    uint32_t qb = 0;
    if (!boundary_degree(q, qb)) return FlatMatrix::CreateZero(rows, cols, ell_);
    const ModCtx<Elt> ctx(ell_);
    const Mat22& g = eta_->g();
    std::vector<Elt> data(static_cast<size_t>(rows) * cols, 0);
    std::map<std::array<int64_t, 5>, FlatMatrix> cache;
    const std::vector<uint32_t>& bslots = eta_->boundary_simplices(qb);
    ENGINE_CHECK_MSG(static_cast<uint64_t>(bslots.size()) * d_V_ == rows,
                     "HeckeConeAction: the boundary slot list does not match the cone's boundary part.");
    for (uint32_t s = 0; s < bslots.size(); ++s) {
        for (uint32_t j = 1; j <= m_; ++j) {
            for (const ChainTerm& ct : eta_->eta(j, qb, bslots[s])) {
                const Elt cmod = coeff_mod32(ct.coeff, ell_);
                if (cmod == 0) continue;
                const uint64_t col0 = static_cast<uint64_t>(ct.sigma_idx) * d_V_;
                ENGINE_CHECK_MSG(col0 + d_V_ <= cols, "HeckeConeAction: an eta term names a simplex outside C^q(K).");
                const std::array<int64_t, 5> key{ static_cast<int64_t>(j), ct.gamma.a, ct.gamma.b, ct.gamma.c, ct.gamma.d };
                auto it = cache.find(key);
                if (it == cache.end()) {
                    FlatMatrix block = fused ? eta_->cosets().rho_s_inv_block(j).get() * rho_of_exact(g * ct.gamma)
                                             : rho_of_exact(eta_->cosets().delta(j) * ct.gamma);
                    it = cache.emplace(key, std::move(block)).first;
                }
                const FlatMatrix& B = it->second;
                for (uint32_t rr = 0; rr < d_V_; ++rr)
                    for (uint32_t cc = 0; cc < d_V_; ++cc) {
                        const Elt v = B.get_entry(rr, cc);
                        if (v == 0) continue;
                        Elt& cell = data[(static_cast<size_t>(s) * d_V_ + rr) * cols + col0 + cc];
                        cell = ctx.add(cell, ctx.mul(cmod, v));
                    }
            }
        }
    }
    return FlatMatrix(rows, cols, ell_, data);
}

HeckeConeAction::RhoMemo::Identity HeckeConeAction::rho_memo_identity() const {
    RhoMemo::Identity id;
    id.owner = static_cast<const void*>(this);
    id.delta_table = static_cast<const void*>(&eta_->cosets());
    id.ell = static_cast<uint64_t>(ell_);
    id.n_sym = static_cast<uint64_t>(n_sym_);
    id.m = static_cast<uint64_t>(m_);
    return id;
}

FlatMatrix HeckeConeAction::apply_h_direct(uint32_t q, const FlatMatrix& c_flat) const {
    RhoMemo memo;
    return apply_h_direct(q, c_flat, memo);
}

// The memo-taking form.  `memo` only ever CACHES rho_of_exact(delta_j gamma),
// a pure function of its key, so the arithmetic below is term for term the
// pre-slot-4 arithmetic whether the table arrives empty or warm.
FlatMatrix HeckeConeAction::apply_h_direct(uint32_t q, const FlatMatrix& c_flat, RhoMemo& cache) const {
    require_field("apply_h_direct");
    const uint32_t rows = cone_->boundary_part_dim(q), cols = cone_->bulk_part_dim(q);
    if (c_flat.get_p() != ell_ || c_flat.get_cols() != 1 || c_flat.get_rows() != cols)
        throw std::invalid_argument("HeckeConeAction::apply_h_direct: wrong operand shape.");
    if (rows == 0 || cols == 0) return FlatMatrix::CreateZero(rows, 1, ell_);
    uint32_t qb = 0;
    if (!boundary_degree(q, qb)) return FlatMatrix::CreateZero(rows, 1, ell_);
    const ModCtx<Elt> ctx(ell_);
    // (slot-5 WP-6) the identity stamp on the caller's table; see Hecke/RhoMemo.h.
    cache.bind(rho_memo_identity());
    RhoMemo::Table& rho_table = cache.table();
    std::vector<Elt> y(rows, 0);
    const std::vector<uint32_t>& bslots = eta_->boundary_simplices(qb);
    for (uint32_t s = 0; s < bslots.size(); ++s) {
        for (uint32_t j = 1; j <= m_; ++j) {
            for (const ChainTerm& ct : eta_->eta(j, qb, bslots[s])) {
                const Elt cmod = coeff_mod32(ct.coeff, ell_);
                if (cmod == 0) continue;
                const uint64_t col0 = static_cast<uint64_t>(ct.sigma_idx) * d_V_;
                const std::array<int64_t, 5> key{ static_cast<int64_t>(j), ct.gamma.a, ct.gamma.b, ct.gamma.c, ct.gamma.d };
                auto it = rho_table.find(key);
                if (it == rho_table.end())
                    it = rho_table.emplace(key, rho_of_exact(eta_->cosets().delta(j) * ct.gamma)).first;
                const FlatMatrix& B = it->second;
                for (uint32_t rr = 0; rr < d_V_; ++rr) {
                    Elt acc = 0;
                    for (uint32_t cc = 0; cc < d_V_; ++cc) {
                        const Elt v = B.get_entry(rr, cc);
                        if (v == 0) continue;
                        acc = ctx.add(acc, ctx.mul(v, c_flat.get_entry(static_cast<uint32_t>(col0) + cc, 0)));
                    }
                    if (acc == 0) continue;
                    Elt& cell = y[static_cast<size_t>(s) * d_V_ + rr];
                    cell = ctx.add(cell, ctx.mul(cmod, acc));
                }
            }
        }
    }
    return FlatMatrix(rows, 1, ell_, y);
}

FlatMatrix HeckeConeAction::apply_Phi(uint32_t q, const FlatMatrix& x) const {
    require_field("apply_Phi");
    require_computed("apply_Phi");
    check_degree(q);
    return mul(Phi_[q], x);
}

FlatMatrix HeckeConeAction::apply_Phi_direct(uint32_t q, const FlatMatrix& x) const {
    // The pre-package behaviour verbatim: three fresh, call-local tables.
    RhoMemos memo;
    return apply_Phi_direct(q, x, memo);
}

FlatMatrix HeckeConeAction::apply_Phi_direct(uint32_t q, const FlatMatrix& x, RhoMemos& memo) const {
    require_field("apply_Phi_direct");
    check_degree(q);
    const uint32_t nb = cone_->bulk_part_dim(q), nd = cone_->boundary_part_dim(q);
    if (x.get_p() != ell_ || x.get_cols() != 1 || x.get_rows() != nb + nd)
        throw std::invalid_argument("HeckeConeAction::apply_Phi_direct: wrong operand shape.");
    // Split the cone cochain.
    FlatMatrix c = FlatMatrix::CreateZero(nb, 1, ell_), cb = FlatMatrix::CreateZero(nd, 1, ell_);
    if (nb > 0) { std::vector<Elt> d(nb); for (uint32_t i = 0; i < nb; ++i) d[i] = x.get_entry(i, 0); c = FlatMatrix(nb, 1, ell_, d); }
    if (nd > 0) { std::vector<Elt> d(nd); for (uint32_t i = 0; i < nd; ++i) d[i] = x.get_entry(nb + i, 0); cb = FlatMatrix(nd, 1, ell_, d); }
    // Bulk: [C, eq:heckechain] through the exact tau-terms (route 2).
    FlatMatrix yb = FlatMatrix::CreateZero(nb, 1, ell_);
    if (nb > 0) yb = bulk_->apply_route2(q, c, memo.bulk);
    // Boundary: [C, eq:bdrychain] matrix-free, plus h evaluated matrix-free.
    FlatMatrix yd = FlatMatrix::CreateZero(nd, 1, ell_);
    if (nd > 0) {
        uint32_t qb = q - 1;
        yd = bd_->apply_direct(qb, cb, memo.bd);
        const FlatMatrix hy = apply_h_direct(q, c, memo.h);
        yd = add(yd, hy);
    }
    std::vector<Elt> out(static_cast<size_t>(nb) + nd, 0);
    for (uint32_t i = 0; i < nb; ++i) out[i] = yb.get_entry(i, 0);
    for (uint32_t i = 0; i < nd; ++i) out[nb + i] = yd.get_entry(i, 0);
    if (out.empty()) return FlatMatrix::CreateZero(0, 1, ell_);
    return FlatMatrix(static_cast<uint32_t>(out.size()), 1, ell_, out);
}

void HeckeConeAction::compute() {
    if (ring_k_ >= 2) {   // (slot-9 WP-4) route 1 over Z/ell^k; the field route below is untouched
        if (computed_) return;
        module_->compute();
        computed_ = true;
        return;
    }
    if (computed_) return;
    const Elt p = ell_;
    const SimplicialComplex& K = *cone_->bulk().get_lssc()->get_base_K_complex();
    const SimplicialComplex& Kb = *cone_->boundary().get_lssc()->get_base_K_complex();

    // ---- h in two bracketings, asserted bit-identical ---------------------
    h_.assign(q_max_ + 1, FlatMatrix::CreateZero(0, 0, p));
    for (uint32_t q = 0; q <= q_max_; ++q) {
        FlatMatrix fused = assemble_h(q, /*fused=*/true);
        const FlatMatrix direct = assemble_h(q, /*fused=*/false);
        if (!flat_equal(fused, direct)) {
            std::ostringstream oss;
            oss << "HeckeConeAction::compute: the fused bracketing rho(s_j^{-1}) rho(g gamma) and the direct "
                   "rho((delta_j gamma) mod ell) of [C, eq:derivedh] differ at q=" << q << ".";
            throw std::runtime_error(oss.str());
        }
        h_[q] = std::move(fused);
    }
    cert_.routes_bit_identical = true;

    // ---- the pieces of Phi -----------------------------------------------
    const std::vector<FlatMatrix>& Theta_bulk = bulk_->get_Theta_flat_matrices();
    std::vector<FlatMatrix> Theta(q_max_ + 1, FlatMatrix::CreateZero(0, 0, p));
    std::vector<FlatMatrix> Theta_bd(q_max_ + 1, FlatMatrix::CreateZero(0, 0, p));
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t nb = cone_->bulk_part_dim(q), nd = cone_->boundary_part_dim(q);
        if (nb > 0) {
            if (q >= Theta_bulk.size() || Theta_bulk[q].get_rows() != nb || Theta_bulk[q].get_cols() != nb)
                throw std::runtime_error("HeckeConeAction: the bulk Theta_flat shape does not match the cone's bulk part.");
            Theta[q] = Theta_bulk[q];
        } else {
            Theta[q] = FlatMatrix::CreateZero(0, 0, p);
        }
        if (nd > 0) {
            const uint32_t qb = q - 1;
            if (qb >= bd_->num_degrees() || bd_->Theta_flat(qb).get_rows() != nd)
                throw std::runtime_error("HeckeConeAction: the boundary Theta_flat shape does not match the cone's boundary part.");
            Theta_bd[q] = bd_->Theta_flat(qb);
        } else {
            Theta_bd[q] = FlatMatrix::CreateZero(0, 0, p);
        }
    }

    // ---- Phi = [[Theta, 0], [h, Theta^bd]] --------------------------------
    Phi_.assign(q_max_ + 1, FlatMatrix::CreateZero(0, 0, p));
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t nb = cone_->bulk_part_dim(q), nd = cone_->boundary_part_dim(q), fd = nb + nd;
        ENGINE_CHECK_MSG(fd == cone_->flat_dim(q), "HeckeConeAction: the cone's flat layout is inconsistent.");
        if (fd == 0) { Phi_[q] = FlatMatrix::CreateZero(0, 0, p); continue; }
        std::vector<Elt> d(static_cast<size_t>(fd) * fd, 0);
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < nb; ++j) d[static_cast<size_t>(i) * fd + j] = Theta[q].get_entry(i, j);
        for (uint32_t i = 0; i < nd; ++i) {
            for (uint32_t j = 0; j < nb; ++j) d[static_cast<size_t>(nb + i) * fd + j] = h_[q].get_entry(i, j);
            for (uint32_t j = 0; j < nd; ++j) d[static_cast<size_t>(nb + i) * fd + nb + j] = Theta_bd[q].get_entry(i, j);
        }
        Phi_[q] = FlatMatrix(fd, fd, p, d);
    }

    // ---- the simplicial coboundaries and the flat restriction -------------
    std::vector<FlatMatrix> d_bulk, d_bd, r_flat;
    for (uint32_t q = 0; q <= q_max_ + 1; ++q) d_bulk.push_back(coboundary_tensor(K, q, d_V_, p));
    for (uint32_t q = 0; q <= q_max_ + 1; ++q) d_bd.push_back(coboundary_tensor(Kb, q, d_V_, p));
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t nb = (q <= K.get_dimension()) ? K.get_num_q_simplices(q) * d_V_ : 0;
        const uint32_t nd = (q <= Kb.get_dimension()) ? Kb.get_num_q_simplices(q) * d_V_ : 0;
        if (nb == 0 || nd == 0) r_flat.push_back(FlatMatrix::CreateZero(nd, nb, p));
        else r_flat.push_back(flat_of_block(cone_->r_flat_block(q), p));
    }

    // ---- the bulk's stored C^q_eq basis (the contract's complex) -----------
    std::vector<FlatMatrix> Ceq_bulk;
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t nb = cone_->bulk_part_dim(q);
        if (nb == 0) { Ceq_bulk.push_back(FlatMatrix::CreateZero(0, 0, p)); continue; }
        Ceq_bulk.push_back(flat_of_dense(cone_->bulk().C_eq_block(q).to_flat(engine::Layout::RowMajor), p));
        ENGINE_CHECK_MSG(Ceq_bulk.back().get_rows() == nb,
                         "HeckeConeAction: the bulk C_eq basis does not match the cone's bulk part.");
    }

    // ---- the residual of [C, eq:conesign], asserted zero on C_eq ----------
    bool resid_flat_all_zero = true;
    resid_.assign(q_max_ + 1, FlatMatrix::CreateZero(0, 0, p));
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const uint32_t tgt = (q <= Kb.get_dimension()) ? Kb.get_num_q_simplices(q) * d_V_ : 0;
        const uint32_t nb = cone_->bulk_part_dim(q);
        if (tgt == 0 || nb == 0) { resid_[q] = FlatMatrix::CreateZero(tgt, nb, p); continue; }
        FlatMatrix acc = FlatMatrix::CreateZero(tgt, nb, p);
        if (q > 0 && cone_->boundary_part_dim(q) > 0) acc = add(acc, mul(d_bd[q - 1], h_[q]));         // d h
        if (q + 1 <= q_max_ && cone_->boundary_part_dim(q + 1) == tgt) acc = add(acc, mul(h_[q + 1], d_bulk[q])); // h d
        acc = add(acc, mul(r_flat[q], Theta[q]));                                                        // + r Theta
        if (q + 1 <= q_max_ && Theta_bd[q + 1].get_rows() == tgt) acc = sub(acc, mul(Theta_bd[q + 1], r_flat[q])); // - Theta^bd r
        resid_[q] = acc;
        if (!flat_is_zero(acc)) resid_flat_all_zero = false;
        // [C, lem:derivedhomotopy] is a statement about Ceq(K, Rel; L), so the
        // bracket is asserted zero ON THE STORED C^q_eq BASIS (header note).
        if (!flat_is_zero(mul(acc, Ceq_bulk[q]))) {
            std::ostringstream oss;
            oss << "HeckeConeAction::compute: the residual (d h + h d + r Theta - Theta^bd r) of [C, eq:conesign] "
                   "is non-zero on C^q_eq at q=" << q << " -- the derived homotopy does not satisfy "
                   "[C, lem:derivedhomotopy].";
            throw std::runtime_error(oss.str());
        }
    }

    // ---- Dcone Phi == Phi Dcone: asserted on C_eq, recorded flat ----------
    bool flat_ok = true;
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const FlatMatrix Dq = flat_of_block(cone_->D_block(q), p);
        const FlatMatrix left = mul(Dq, Phi_[q]);
        FlatMatrix right = FlatMatrix::CreateZero(left.get_rows(), left.get_cols(), p);
        if (q + 1 <= q_max_) right = mul(Phi_[q + 1], Dq);
        else ENGINE_CHECK_MSG(Dq.get_rows() == 0, "HeckeConeAction: a non-empty scaffolding degree.");
        if (!flat_equal(left, right)) flat_ok = false;
        const FlatMatrix Cq = flat_of_dense(cone_->Ceq_block(q).to_flat(engine::Layout::RowMajor), p);
        if (Cq.get_cols() != 0 && !flat_equal(mul(left, Cq), mul(right, Cq))) {
            std::ostringstream oss;
            oss << "HeckeConeAction::compute: Dcone Phi != Phi Dcone on the equivariant cone cochains at q=" << q << ".";
            throw std::runtime_error(oss.str());
        }
    }
    cert_.phi_flat_identity = flat_ok;

    // ---- the matrix-free second route on the standard basis ---------------
    // Every basis vector of every degree is still compared, one by one, and the
    // predicate is unchanged: Phi^q e_b == apply_Phi_direct(q, e_b).  What the
    // slot-4 WP-4 form does not do is issue one fd x fd by fd x 1 product per
    // basis vector: the left-hand sides of a block of kColumnBlock consecutive
    // basis vectors are the corresponding columns of Phi^q I_block, so they come
    // out of ONE product.  Over a field that product is entrywise
    // sum_k Phi[i][k] * I[k][b] = Phi[i][b] -- the same canonical residue the
    // per-vector product returns -- so the comparison is bit-identical to the
    // pre-slot-4 one, vector by vector, and the failure message still names the
    // first offending basis vector in the same order.  The matrix-free route
    // stays strictly per vector (it is matrix-free by definition); it is given a
    // shared rho memo -- since slot-5 WP-6 THREE of them, one per coset table
    // (h's, the bulk's route 2 and the boundary's direct route), each caching a
    // pure function of its own (j, gamma) key and changing no term.
    {
        constexpr uint32_t kColumnBlock = 64;
        for (uint32_t q = 0; q <= q_max_; ++q) {
            const uint32_t fd = cone_->flat_dim(q);
            RhoMemos memo;
            for (uint32_t b0 = 0; b0 < fd; b0 += kColumnBlock) {
                const uint32_t w = std::min(kColumnBlock, fd - b0);
                std::vector<Elt> eblk(static_cast<size_t>(fd) * w, 0);
                for (uint32_t t = 0; t < w; ++t) eblk[static_cast<size_t>(b0 + t) * w + t] = 1;
                const FlatMatrix Y1 = mul(Phi_[q], FlatMatrix(fd, w, p, eblk));
                for (uint32_t t = 0; t < w; ++t) {
                    const uint32_t b = b0 + t;
                    std::vector<Elt> e(fd, 0);
                    e[b] = 1;
                    const FlatMatrix eb(fd, 1, p, e);
                    const FlatMatrix y2 = apply_Phi_direct(q, eb, memo);
                    bool same = (Y1.get_rows() == y2.get_rows() && y2.get_cols() == 1);
                    for (uint32_t i = 0; same && i < fd; ++i) same = (Y1.get_entry(i, t) == y2.get_entry(i, 0));
                    if (!same) {
                        std::ostringstream oss;
                        oss << "HeckeConeAction::compute: the matrix-free route (apply_route2 / apply_direct / eq:derivedh) "
                               "differs from the stored Phi at q=" << q << ", basis vector " << b << ".";
                        throw std::runtime_error(oss.str());
                    }
                }
            }
        }
    }

    // ---- level (b): A_c^{(q)} = P Phi H ([C, alg:assembly] on Cone(r)) -----
    A_.assign(q_max_ + 1, MatrixT<Elt>(0, 0, p));
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const FlatMatrix Hq = flat_of_dense(cone_->H_block(q).to_flat(engine::Layout::RowMajor), p);
        const FlatMatrix Pq = flat_of_block(cone_->P_block(q), p);
        A_[q] = matrix_of_flat(mul(Pq, mul(Phi_[q], Hq)));
    }

    // ---- fill the certificate --------------------------------------------
    cert_.ell = ell_; cert_.d_V = d_V_; cert_.n_sym = n_sym_;
    cert_.p_hecke = p_hecke(); cert_.N = N(); cert_.nu = nu(); cert_.m = m_;
    cert_.q_max = q_max_;
    cert_.eta_num_degrees = eta_->num_degrees();
    cert_.six_invertible = (ell_ >= 5);
    // [OD] 2026-09-22 Q2: item 8's fail() has carried no six_invertible guard
    // since slot-5 WP-4, and [C] v1.7.0 asserts the r-half at every ell, so
    // the run asserts it unconditionally and the record says so.
    cert_.restriction_equivariance_asserted = true;
    cert_.ring_is_field = true;
    cert_.flat_dim.clear(); cert_.bulk_part_dim.clear(); cert_.boundary_part_dim.clear();
    cert_.Theta.clear(); cert_.Theta_bd.clear(); cert_.h.clear(); cert_.Phi.clear();
    cert_.D.clear(); cert_.Ceq.clear(); cert_.Z.clear(); cert_.H.clear(); cert_.P.clear();
    cert_.A_c.clear(); cert_.A_bulk.clear(); cert_.A_bd.clear();
    cert_.pi.clear(); cert_.r.clear(); cert_.iota.clear();
    cert_.pi_built.clear(); cert_.r_built.clear(); cert_.iota_built.clear();
    for (uint32_t q = 0; q <= q_max_; ++q) {
        cert_.flat_dim.push_back(cone_->flat_dim(q));
        cert_.bulk_part_dim.push_back(cone_->bulk_part_dim(q));
        cert_.boundary_part_dim.push_back(cone_->boundary_part_dim(q));
        cert_.Theta.push_back(Theta[q]);
        cert_.Theta_bd.push_back(Theta_bd[q]);
        cert_.h.push_back(h_[q]);
        cert_.Phi.push_back(Phi_[q]);
        cert_.D.push_back(flat_of_block(cone_->D_block(q), p));
        cert_.Ceq.push_back(flat_of_dense(cone_->Ceq_block(q).to_flat(engine::Layout::RowMajor), p));
        cert_.Z.push_back(flat_of_dense(cone_->Z_block(q).to_flat(engine::Layout::RowMajor), p));
        cert_.H.push_back(flat_of_dense(cone_->H_block(q).to_flat(engine::Layout::RowMajor), p));
        cert_.P.push_back(flat_of_block(cone_->P_block(q), p));
        cert_.A_c.push_back(A_[q]);
        cert_.pi.push_back(cone_->pi_built(q) ? cone_->pi(q) : FlatMatrix::CreateZero(0, 0, p));
        cert_.r.push_back(cone_->r_built(q) ? cone_->r(q) : FlatMatrix::CreateZero(0, 0, p));
        cert_.iota.push_back(cone_->iota_built(q) ? cone_->iota(q) : FlatMatrix::CreateZero(0, 0, p));
        cert_.pi_built.push_back(cone_->pi_built(q) ? 1 : 0);
        cert_.r_built.push_back(cone_->r_built(q) ? 1 : 0);
        cert_.iota_built.push_back(cone_->iota_built(q) ? 1 : 0);
    }
    cert_.d_bulk = d_bulk;
    cert_.d_bd = d_bd;
    cert_.r_flat = r_flat;
    cert_.Ceq_bulk = Ceq_bulk;
    cert_.residual_flat_zero = resid_flat_all_zero;
    {
        const std::vector<MatrixT<Elt>>& Ab = bulk_->get_H_matrices();
        for (uint32_t q = 0; q <= q_max_; ++q)
            cert_.A_bulk.push_back(q < Ab.size() ? Ab[q] : MatrixT<Elt>(0, 0, p));
        const std::vector<MatrixT<Elt>>& Ad = bd_->get_H_matrices();
        for (uint32_t q = 0; q <= q_max_; ++q)
            cert_.A_bd.push_back(q < Ad.size() ? Ad[q] : MatrixT<Elt>(0, 0, p));
    }
    cert_.eta_table = eta_;
    cert_.cosets = &eta_->cosets();
    cert_.boundary_simplices.clear();
    for (uint32_t qb = 0; qb < eta_->num_degrees(); ++qb)
        cert_.boundary_simplices.push_back(eta_->boundary_simplices(qb));
    cert_.eta_zero = eta_->eta_zero();
    // cochain_commutes := r Theta == Theta^bd r entrywise, in every degree in
    // which both sides exist ([C, obl:eta](vi); RECORDED, gates nothing).
    {
        bool cc = true;
        for (uint32_t q = 0; q <= q_max_ && cc; ++q) {
            const uint32_t tgt = (q <= Kb.get_dimension()) ? Kb.get_num_q_simplices(q) * d_V_ : 0;
            if (tgt == 0 || cone_->bulk_part_dim(q) == 0) continue;
            if (q + 1 > q_max_ || Theta_bd[q + 1].get_rows() != tgt) continue;
            // on C^q_eq(K) -- the same predicate Hecke/HeckeBoundaryDiagnostics
            // records ([C, obl:eta](vi); RECORDED, gates nothing).
            cc = flat_equal(mul(mul(r_flat[q], Theta[q]), Ceq_bulk[q]),
                            mul(mul(Theta_bd[q + 1], r_flat[q]), Ceq_bulk[q]));
        }
        cert_.cochain_commutes = cc;
    }
    cert_.eta_table_fnv64 = eta_->table_fnv64();
    cert_.tau_table_fnv64 = eta_->tau_table_fnv64();
    cert_.tau_boundary_table_fnv64 = eta_->tau_boundary_table_fnv64();
    cert_.solve_regime_eta = eta_->solve_regime();
    cert_.solve_regime_tau = eta_->solve_regime_tau();
    cert_.solve_regime_tau_boundary = eta_->solve_regime_tau_boundary();
    cert_.eta_policy = hecke::to_string(eta_->policy());
    cert_.tau_boundary_policy = hecke::to_string(bd_->taub().policy());
    // item 8's predicate, recorded whether or not it is asserted.
    {
        bool ok = true;
        for (uint32_t q = 0; q <= q_max_; ++q) {
            if (!cone_->r_built(q)) continue;
            const MatrixT<Elt>& A = cert_.A_bulk[q];
            const MatrixT<Elt>& Ad = cert_.A_bd[q];
            FlatMatrix Am(A.rows, A.cols, p), Adm(Ad.rows, Ad.cols, p);
            for (uint32_t i = 0; i < A.rows; ++i) for (uint32_t j = 0; j < A.cols; ++j) Am.set_entry(i, j, A.data[i][j]);
            for (uint32_t i = 0; i < Ad.rows; ++i) for (uint32_t j = 0; j < Ad.cols; ++j) Adm.set_entry(i, j, Ad.data[i][j]);
            if (cert_.r[q].get_rows() != Ad.rows || cert_.r[q].get_cols() != A.rows) { ok = false; break; }
            ok = ok && flat_equal(mul(cert_.r[q], Am), mul(Adm, cert_.r[q]));
        }
        cert_.restriction_equivariance = ok;
    }

    // ---- the certificate on the freshly built data ------------------------
    {
        std::vector<std::string> f;
        const uint32_t bad = cert_.verify(f);
        if (bad != 0) {
            std::ostringstream oss;
            oss << "HeckeConeAction::compute: ConeActionCertificate failed (" << bad << " item(s)): "
                << (f.empty() ? std::string("?") : f[0]);
            throw std::runtime_error(oss.str());
        }
    }
    computed_ = true;
}

const FlatMatrix& HeckeConeAction::h_flat(uint32_t q) const { require_field("h_flat"); require_computed("h_flat"); check_degree(q); return h_[q]; }
const FlatMatrix& HeckeConeAction::Phi_flat(uint32_t q) const { require_field("Phi_flat"); require_computed("Phi_flat"); check_degree(q); return Phi_[q]; }
const FlatMatrix& HeckeConeAction::residual(uint32_t q) const { require_field("residual"); require_computed("residual"); check_degree(q); return resid_[q]; }
FlatMatrix HeckeConeAction::Ceq_bulk(uint32_t q) const {
    require_field("Ceq_bulk");
    require_computed("Ceq_bulk");
    check_degree(q);
    return cert_.Ceq_bulk[q];
}
FlatMatrix HeckeConeAction::residual_on_Ceq(uint32_t q) const {
    require_field("residual_on_Ceq");
    require_computed("residual_on_Ceq");
    check_degree(q);
    if (resid_[q].get_cols() == 0 || cert_.Ceq_bulk[q].get_cols() == 0)
        return FlatMatrix::CreateZero(resid_[q].get_rows(), cert_.Ceq_bulk[q].get_cols(), ell_);
    return mul(resid_[q], cert_.Ceq_bulk[q]);
}
const MatrixT<uint32_t>& HeckeConeAction::A_c(uint32_t q) const { require_field("A_c"); require_computed("A_c"); check_degree(q); return A_[q]; }
const std::vector<MatrixT<uint32_t>>& HeckeConeAction::get_H_matrices() const { require_field("get_H_matrices"); require_computed("get_H_matrices"); return A_; }
uint32_t HeckeConeAction::dim_Hc(uint32_t q) const { require_field("dim_Hc"); check_degree(q); return cone_->dim_Hc(q); }
const ConeActionCertificate& HeckeConeAction::certificate() const { require_field("certificate"); require_computed("certificate"); return cert_; }

std::vector<ConeGoldenObject> HeckeConeAction::golden_objects() const {
    require_field("golden_objects");
    require_computed("golden_objects");
    std::vector<ConeGoldenObject> out;
    const std::string suffix = "_p" + std::to_string(p_hecke()) + "_r" + std::to_string(nu());
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const std::string Q = "^" + std::to_string(q);
        const MatrixT<Elt>& A = A_[q];
        // tier A: char-poly / rank / trace of A_c^{(q)}.
        const FpPolyT<Elt> cp = CharPolyBerkowitz(A);
        engine::FnvHasher hp;
        hp.feed(cp.coeffs.size());
        for (Elt c : cp.coeffs) hp.feed(static_cast<uint64_t>(c));
        out.push_back(ConeGoldenObject{ "charpoly_A_c" + Q + suffix, 'A', cp.coeffs.size(), 1, hp.value() });
        MatrixT<Elt> copy = A;
        const uint64_t rk = (A.rows == 0 || A.cols == 0) ? 0 : copy.rank();
        out.push_back(ConeGoldenObject{ "rank_A_c" + Q + suffix, 'A', 1, 1, engine::HashScalar64(rk) });
        const ModCtx<Elt> ctx(ell_);
        Elt tr = 0;
        for (uint32_t i = 0; i < A.rows && i < A.cols; ++i) tr = ctx.add(tr, A.data[i][i]);
        out.push_back(ConeGoldenObject{ "trace_A_c" + Q + suffix, 'A', 1, 1, engine::HashScalar64(tr) });
        // tier B: the choice-dependent objects ([C, sec:cone] code map, OP7) --
        // the raw A_c^{(q)} first: it is stored-basis dependent, so [X] Sec. 1.6
        // / OP7 put it in tier B.  (slot-6 WP-5, [WP5-3] finding 5: labelled
        // 'A' here until then; the golden harness published it as 'B' through
        // a tier override, which is now gone, so the published value is
        // unchanged.  This moves only golden_digest(), which nothing prints.)
        out.push_back(ConeGoldenObject{ "A_c" + Q + suffix, 'B', A.rows, A.cols, hash_matrix(A) });
        out.push_back(ConeGoldenObject{ "h_flat" + Q + suffix, 'B', h_[q].get_rows(), h_[q].get_cols(), hash_flat(h_[q]) });
        out.push_back(ConeGoldenObject{ "Theta_bd_flat" + Q + suffix, 'B', cert_.Theta_bd[q].get_rows(),
                                        cert_.Theta_bd[q].get_cols(), hash_flat(cert_.Theta_bd[q]) });
    }
    out.push_back(ConeGoldenObject{ "eta_table_fnv64" + suffix, 'B', 1, 1, engine::HashScalar64(cert_.eta_table_fnv64) });
    out.push_back(ConeGoldenObject{ "tau_table_fnv64" + suffix, 'B', 1, 1, engine::HashScalar64(cert_.tau_table_fnv64) });
    out.push_back(ConeGoldenObject{ "tau_bd_table_fnv64" + suffix, 'B', 1, 1, engine::HashScalar64(cert_.tau_boundary_table_fnv64) });
    return out;
}

uint64_t HeckeConeAction::golden_digest() const {
    uint64_t h = engine::kFnvOffset64;
    for (const ConeGoldenObject& o : golden_objects()) {
        h = engine::FnvBytes(h, o.object);
        h = engine::FnvU64(h, static_cast<uint64_t>(o.tier));
        h = engine::FnvU64(h, o.rows);
        h = engine::FnvU64(h, o.cols);
        h = engine::FnvU64(h, o.hash);
    }
    return h;
}

std::string HeckeConeAction::to_string_offset(uint32_t offset) const {
    require_field("to_string");
    const std::string pre(offset, ' ');
    std::ostringstream oss;
    oss << pre << "HeckeConeAction (Phi_g on Cone(r); [C, def:coneaction])\n";
    oss << pre << "  ell = " << ell_ << ", n = " << n_sym_ << ", p = " << p_hecke() << ", nu = " << nu()
        << ", m = " << m_ << ", q_max(H^*_c) = " << q_max_ << "\n";
    if (!computed_) { oss << pre << "  (not computed)\n"; return oss.str(); }
    oss << pre << "  eta policy = " << cert_.eta_policy << ", tau_bd policy = " << cert_.tau_boundary_policy << "\n";
    oss << pre << "  eta_zero = " << (cert_.eta_zero ? "true" : "false")
        << ", cochain_commutes = " << (cert_.cochain_commutes ? "true" : "false")
        << ", routes_bit_identical = " << (cert_.routes_bit_identical ? "true" : "false") << "\n";
    for (uint32_t q = 0; q <= q_max_; ++q) {
        oss << pre << "  q = " << q << ": flat " << cone_->flat_dim(q) << " = " << cone_->bulk_part_dim(q) << " + "
            << cone_->boundary_part_dim(q) << ", h " << h_[q].get_rows() << "x" << h_[q].get_cols()
            << ", dim H^q_c = " << cone_->dim_Hc(q) << ", A_c " << A_[q].rows << "x" << A_[q].cols << "\n";
    }
    return oss.str();
}

std::string HeckeConeAction::to_json() const {
    require_field("to_json");
    require_computed("to_json");
    std::ostringstream oss;
    oss << "{\"schema\":\"hecke_cone_action/v1\"";
    oss << ",\"ell\":" << ell_ << ",\"n_sym\":" << n_sym_ << ",\"d_V\":" << d_V_
        << ",\"p\":" << p_hecke() << ",\"nu\":" << nu() << ",\"N\":" << N() << ",\"m\":" << m_;
    oss << ",\"boundary_model\":\"" << cone_->boundary_model() << "\"";
    oss << ",\"boundary_is_group_cohomology\":"
        << (cone_->boundary_is_group_cohomology() ? "true" : "false");
    // The per-complex degree fields ([C, sec:recordschema] item 5; WP-5
    // finding 2: a derived complex names its sources, a chain/cochain MAP
    // carries its SOURCE complex's fields).
    oss << ",\"degrees\":{";
    oss << "\"cone\":" << cone_->cone_fields().to_json();
    oss << ",\"bulk\":" << cone_->bulk_fields().to_json();
    oss << ",\"boundary\":" << cone_->boundary_fields().to_json();
    oss << ",\"h\":{\"kind\":\"cochain_map\",\"degree_shift\":-1,\"source\":\"bulk\","
        << "\"q_max\":" << cone_->bulk_fields().q_max
        << ",\"truncated\":" << (cone_->bulk_fields().truncated ? "true" : "false")
        << ",\"q_max_source\":\"derived\",\"degrees\":[";
    for (size_t i = 0; i < cone_->bulk_fields().degrees.size(); ++i)
        oss << (i ? "," : "") << cone_->bulk_fields().degrees[i];
    oss << "]}}";
    oss << ",\"policies\":{\"eta_vertex\":\"" << cert_.eta_policy << "\",\"tau_boundary_vertex\":\""
        << cert_.tau_boundary_policy << "\"}";
    oss << ",\"tables\":{\"eta_fnv64\":\"0x" << std::hex << cert_.eta_table_fnv64 << std::dec
        << "\",\"tau_fnv64\":\"0x" << std::hex << cert_.tau_table_fnv64 << std::dec
        << "\",\"tau_boundary_fnv64\":\"0x" << std::hex << cert_.tau_boundary_table_fnv64 << std::dec << "\"}";
    oss << ",\"solve_regimes\":{\"eta\":\"" << cert_.solve_regime_eta << "\",\"tau\":\"" << cert_.solve_regime_tau
        << "\",\"tau_boundary\":\"" << cert_.solve_regime_tau_boundary << "\"}";
    oss << ",\"eta_zero\":" << (cert_.eta_zero ? "true" : "false")
        << ",\"cochain_commutes\":" << (cert_.cochain_commutes ? "true" : "false");
    oss << ",\"per_degree\":[";
    for (uint32_t q = 0; q <= q_max_; ++q) {
        if (q) oss << ",";
        oss << "{\"q\":" << q << ",\"flat_dim\":" << cone_->flat_dim(q)
            << ",\"bulk_part_dim\":" << cone_->bulk_part_dim(q)
            << ",\"boundary_part_dim\":" << cone_->boundary_part_dim(q)
            << ",\"dim_Hc\":" << cone_->dim_Hc(q) << ",\"A_c\":";
        emit_matrix_json(oss, A_[q]);
        oss << ",\"h_rows\":" << h_[q].get_rows() << ",\"h_cols\":" << h_[q].get_cols() << "}";
    }
    oss << "]";
    oss << ",\"certificate\":{\"routes_bit_identical\":" << (cert_.routes_bit_identical ? "true" : "false")
        << ",\"phi_flat_identity\":" << (cert_.phi_flat_identity ? "true" : "false")
        << ",\"residual_flat_zero\":" << (cert_.residual_flat_zero ? "true" : "false")
        << ",\"restriction_equivariance\":" << (cert_.restriction_equivariance ? "true" : "false")
        // [OD] 2026-09-22 Q2: the record reports the assertion the run
        // PERFORMS (item 8, unconditional in ell), not the retired
        // six_invertible carve-out.  Certificate item 12 refuses a `false`.
        << ",\"restriction_equivariance_asserted\":" << (cert_.restriction_equivariance_asserted ? "true" : "false")
        << ",\"items_failing\":" << cert_.verify() << "}";
    oss << "}";
    (void)emit_flat_json;
    return oss.str();
}

// ===========================================================================
// HeckeConeStrictPairRoute (slot-6 WP-4) -- [C, prop:coneroutes](1),
// [C, obl:coneexact](vii) first sentence.  See the header note.
// ===========================================================================

namespace {
// (slot-10 WP-3; decision 3 (A)) The cross-routes' ring rule: honoured when the
// normative action is over the Z/ell^k IN FORCE (the rule of HeckeConeAction's
// own constructor), and then only in the uint32_t regime; every other
// combination keeps slot 8's refusal (the ring in force, or a composite input).
void CrossRouteRingGuard(const char* who, const HeckeConeAction* normative) {
    const uint32_t kn = normative ? normative->coefficient_ring_k() : 1u;
    if (kn >= 2 && CoefficientRingK() == kn) {
        (void)CoefficientRingContext<uint32_t>(who, normative->ell(), kn);
        return;
    }
    RequireFieldCoefficientRing(who, kn);
}
} // namespace

namespace {
// (S13 WP-12, slot-10 m-14) the one body of both routes' RequireRingRegime.
void RequireRouteRingRegime(const char* who, uint64_t ell, uint32_t k) {
    if (k >= 2) (void)CoefficientRingContext<uint32_t>(who, ell, k);
}
} // namespace

void HeckeConeStrictPairRoute::RequireRingRegime(uint64_t ell, uint32_t k) {
    RequireRouteRingRegime("HeckeConeStrictPairRoute", ell, k);
}

void HeckeConeRelativeRoute::RequireRingRegime(uint64_t ell, uint32_t k) {
    RequireRouteRingRegime("HeckeConeRelativeRoute", ell, k);
}

HeckeConeStrictPairRoute::HeckeConeStrictPairRoute(const HeckeConeAction* normative) : norm_(normative) {
    // (slot-8 WP-3) the cone's cross-route; (slot-10 WP-3, decision 3 (A)) honoured over the
    // Z/ell^k in force when the normative action is over it, in the uint32_t regime.
    CrossRouteRingGuard("HeckeConeStrictPairRoute", normative);
    if (norm_ == nullptr) throw std::invalid_argument("HeckeConeStrictPairRoute: null normative action");
    if (!norm_->computed())
        throw std::invalid_argument("HeckeConeStrictPairRoute: the normative HeckeConeAction is not computed");
}

HeckeConeStrictPairRoute::HeckeConeStrictPairRoute(const HeckeConeAction* normative,
                                                   const HeckeTauTableT<uint32_t>* strict_table)
    : HeckeConeStrictPairRoute(normative) {
    if (strict_table == nullptr) throw std::invalid_argument("HeckeConeStrictPairRoute: null strict table (validation seam)");
    tau_ext_ = strict_table;
}

HeckeConeStrictPairRoute::~HeckeConeStrictPairRoute() = default;

void HeckeConeStrictPairRoute::compute() {
    if (computed_) return;
    if (norm_->is_composite()) { compute_composite(); return; }   // (slot-10 WP-3)
    const HeckeConeAction& norm = *norm_;
    const HeckeOperatorOnKT<uint32_t>& bulk = norm.bulk_operator();
    const HeckeBoundaryOperatorT<uint32_t>& bd = norm.boundary_operator();
    const HeckeTauTableT<uint32_t>& tau = bulk.tau();
    const HeckeBoundaryTauTableT<uint32_t>& taub = bd.taub();

    rec_ = Record{};
    rec_.ell = norm.ell();
    rec_.n_sym = norm.n_sym();
    rec_.nu = norm.nu();
    rec_.p_hecke = norm.p_hecke();
    rec_.q_max = norm.q_max();
    rec_.six_invertible = (rec_.ell >= 5);
    rec_.asserted = rec_.six_invertible;
    rec_.tau_normative_fnv64 = norm.eta().tau_table_fnv64();

    // The normative triple must be ONE datum: the eta it carries was built on
    // the bulk operator's tau, on the boundary operator's tau_bd and coset
    // tables. (Since hecke-extensions slot 7, WP-11: HeckeConeAction's own
    // constructor now enforces BOTH halves -- the eta-provenance check --
    // so `norm` cannot exist otherwise; this repeats the tau half, cheaply,
    // as a local invariant of the objects this route is about to rebuild
    // from, not as the sole guard against a mismatched normative triple.)
    ENGINE_CHECK_MSG(&norm.eta().tau() == &tau,
                     "HeckeConeStrictPairRoute: the normative eta was not built on the bulk operator's tau");
    ENGINE_CHECK_MSG(&bulk.cosets() == &bd.cosets() && &tau.cosets() == &bulk.cosets(),
                     "HeckeConeStrictPairRoute: the normative objects use different coset tables");
    ENGINE_CHECK_MSG(!tau.is_strict_pair(),
                     "HeckeConeStrictPairRoute: the normative tau is itself the strict-pair table");

    const auto stop = [&](const std::string& what) {
        std::ostringstream oss;
        oss << "HeckeConeStrictPairRoute (ell = " << rec_.ell << ", n = " << rec_.n_sym << ", p = " << rec_.p_hecke
            << ", nu = " << rec_.nu << "): " << what
            << " -- at 6 in R^x this contradicts [C, prop:coneroutes](1) / [C, prop:strictpair]; ABORTED "
               "([C, obl:coneexact](vii)).";
        throw std::runtime_error(oss.str());
    };

    try {
        // (1) tau^s: its own certified integer table, its own fill search --
        //     (slot-10 WP-3) or the validation seam's table, re-certified.
        if (tau_ext_ != nullptr) {
            rec_.external_table = true;
            if (!tau_ext_->is_strict_pair() || &tau_ext_->cosets() != &bulk.cosets() ||
                tau_ext_->strict_boundary_table() != &taub)
                throw std::runtime_error("the supplied table is not the strict-pair table of the normative (cosets, tau_bd)");
            const uint32_t code = tau_ext_->certify();
            if (code != 0)
                throw std::runtime_error("the supplied strict table fails its certificate (code " + std::to_string(code) + ")");
        } else {
            tau_s_ = std::make_unique<HeckeTauTableT<uint32_t>>(tau.complex(), tau.model(), bulk.cosets(), tau.ell(),
                                                                tau.n_sym(), taub);
        }
        const HeckeTauTableT<uint32_t>& ts = *tau_strict();
        // (2) eta^s on (tau^s, tau_bd), certified as always ([C, obl:eta](i)-(iv)).
        eta_s_ = std::make_unique<HeckePairHomotopyT<uint32_t>>(ts, taub, norm.eta_policy());
        // (3) Theta^s on the same cohomology and coset tables.
        bulk_s_ = std::make_unique<HeckeOperatorOnKT<uint32_t>>(&bulk.cohomology_of_L(), &bulk.cosets(), &ts);
        bulk_s_->compute();
        // (4) Phi^s on the SAME cone and the SAME boundary operator.
        act_s_ = std::make_unique<HeckeConeAction>(&norm.cone(), bulk_s_.get(), &bd, eta_s_.get());
        act_s_->compute();
        rec_.built = true;
    } catch (const std::exception& e) {
        act_s_.reset();
        bulk_s_.reset();
        eta_s_.reset();
        tau_s_.reset();
        if (rec_.six_invertible) stop(std::string("the strict-pair datum could not be built or certified: ") + e.what());
        rec_.build_failure = e.what();   // RECORDED at ell in {2, 3}
        computed_ = true;
        return;
    }

    rec_.solve_regime = tau_strict()->solve_regime();
    rec_.fills_int64 = tau_strict()->num_fills_int64();
    rec_.fills_widened = tau_strict()->num_fills_widened();
    rec_.strict_boundary_slots = tau_strict()->num_strict_boundary_slots();
    rec_.eta_zero = eta_s_->eta_zero();
    rec_.eta_nonzero_slots = eta_s_->num_nonzero_slots();
    rec_.tau_strict_fnv64 = eta_s_->tau_table_fnv64();
    rec_.tau_differs = (rec_.tau_strict_fnv64 != rec_.tau_normative_fnv64);
    rec_.cochain_commutes = act_s_->certificate().cochain_commutes;
    // Strictness makes every defect (j tau_bd - tau^s j) vanish, so the
    // certified eta-table on (tau^s, tau_bd) is identically zero at EVERY ell
    // (a statement over Z, not about R): a non-zero table is a defect of the
    // construction, never a recorded outcome.
    ENGINE_CHECK_MSG(rec_.solve_regime == "Z", "HeckeConeStrictPairRoute: the strict-pair table is not over Z");
    ENGINE_CHECK_MSG(rec_.eta_zero, "HeckeConeStrictPairRoute: eta on (tau^s, tau_bd) is not identically zero");

    // (5) The comparison, entrywise, in every computed degree.
    for (uint32_t q = 0; q <= rec_.q_max; ++q) {
        const MatrixT<uint32_t>& A = norm.A_c(q);
        const MatrixT<uint32_t>& As = act_s_->A_c(q);
        ++rec_.degrees_compared;
        const bool same = (A.rows == As.rows && A.cols == As.cols && A.p == As.p && A.data == As.data);
        if (!same) rec_.disagreeing_degrees.push_back(q);
    }
    if (!rec_.disagreeing_degrees.empty() && rec_.six_invertible) {
        std::ostringstream oss;
        oss << "A_c^{(q)} of the strict pair differs from the normative cone's in degree(s)";
        for (uint32_t q : rec_.disagreeing_degrees) oss << " " << q;
        stop(oss.str());
    }
    computed_ = true;
}

std::string HeckeConeStrictPairRoute::to_string() const {
    std::ostringstream oss;
    oss << "HeckeConeStrictPairRoute (ell = " << rec_.ell << ", n = " << rec_.n_sym << ", p = " << rec_.p_hecke
        << ", nu = " << rec_.nu << "): " << (rec_.asserted ? "asserted" : "recorded") << ", ";
    if (!rec_.built) {
        oss << "strict datum NOT built (" << rec_.build_failure << ")";
        return oss.str();
    }
    oss << "solve_regime " << rec_.solve_regime << " (" << rec_.fills_int64 << " int64 / " << rec_.fills_widened
        << " widened fills), " << rec_.strict_boundary_slots << " boundary slots, eta_zero "
        << (rec_.eta_zero ? "true" : "false") << ", A_c agrees in "
        << (rec_.degrees_compared - rec_.disagreeing_degrees.size()) << " / " << rec_.degrees_compared << " degrees";
    return oss.str();
}

// ===========================================================================
// HeckeConeRelativeRoute (slot-6 WP-5) -- [C, prop:coneroutes](2), "route 3",
// [C, obl:coneexact](vii) second sentence.  See the header note.  Every
// failure below aborts, at every ell.
// ===========================================================================

HeckeConeRelativeRoute::HeckeConeRelativeRoute(const HeckeConeAction* normative) : norm_(normative) {
    // (slot-8 WP-3) the cone's cross-route; (slot-10 WP-3, decision 3 (A)) honoured over the
    // Z/ell^k in force when the normative action is over it, in the uint32_t regime.
    CrossRouteRingGuard("HeckeConeRelativeRoute", normative);
    if (norm_ == nullptr) throw std::invalid_argument("HeckeConeRelativeRoute: null normative action");
    if (!norm_->computed())
        throw std::invalid_argument("HeckeConeRelativeRoute: the normative HeckeConeAction is not computed");
}

HeckeConeRelativeRoute::~HeckeConeRelativeRoute() = default;

void HeckeConeRelativeRoute::require_computed(const char* who, uint32_t q) const {
    if (!computed_) throw std::runtime_error(std::string("HeckeConeRelativeRoute::") + who + ": compute() not run.");
    if (q > rec_.q_max) throw std::out_of_range(std::string("HeckeConeRelativeRoute::") + who + ": degree out of range.");
}

void HeckeConeRelativeRoute::compute() {
    if (computed_) return;
    if (norm_->is_composite()) { compute_composite(); return; }   // (slot-10 WP-3)
    const HeckeConeAction& norm = *norm_;
    const EquivariantMappingCone& cone = norm.cone();
    const ConeActionCertificate& cert = norm.certificate();
    const EquivariantCohomologyOnK& bulk = cone.bulk();
    const EquivariantCohomologyOnK& bdc = cone.boundary();
    const SimplicialComplex& K = *bulk.get_lssc()->get_base_K_complex();
    const SimplicialComplex& Kb = *bdc.get_lssc()->get_base_K_complex();
    const Elt p = norm.ell();
    const uint32_t dV = norm.d_V(), Q = norm.q_max();

    rec_ = Record{};
    rec_.ell = p;
    rec_.n_sym = norm.n_sym();
    rec_.nu = norm.nu();
    rec_.p_hecke = norm.p_hecke();
    rec_.q_max = Q;

    const auto stop = [&](const std::string& what) {
        std::ostringstream oss;
        oss << "HeckeConeRelativeRoute (ell = " << rec_.ell << ", n = " << rec_.n_sym << ", p = " << rec_.p_hecke
            << ", nu = " << rec_.nu << "): " << what
            << " -- contradicts [C, prop:coneroutes](2) / [C, lem:conerelative]; ABORTED at every ell "
               "([C, obl:coneexact](vii)).";
        throw std::runtime_error(oss.str());
    };
    const auto NK = [&](uint32_t q) -> uint32_t { return q <= K.get_dimension() ? K.get_num_q_simplices(q) : 0; };
    const auto NB = [&](uint32_t q) -> uint32_t { return q <= Kb.get_dimension() ? Kb.get_num_q_simplices(q) : 0; };
    const auto dense = [&](const engine::BlockMat<Elt>& B) { return flat_of_dense(B.to_flat(engine::Layout::RowMajor), p); };
    const auto zero = [&](uint32_t r, uint32_t c) { return FlatMatrix::CreateZero(r, c, p); };

    ENGINE_CHECK_MSG(cert.Theta.size() == Q + 1 && cert.h.size() == Q + 1 && cert.r_flat.size() == Q + 1 &&
                         cert.d_bulk.size() >= Q + 1 && cert.Ceq_bulk.size() == Q + 1 && cert.Theta_bd.size() == Q + 1,
                     "HeckeConeRelativeRoute: the normative certificate is misshaped");

    // ---- (1) ext_0 from the stored flat restriction, and ker r's simplices --
    ext0_.assign(Q + 1, zero(0, 0));
    std::vector<std::vector<uint32_t>> vanishing(std::min<uint32_t>(Q, K.get_dimension()) + 1);
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t nk = NK(q), nb = NB(q);
        const FlatMatrix& R = cert.r_flat[q];   // nb dV x nk dV
        if (R.get_rows() != nb * dV || R.get_cols() != nk * dV) stop("r_flat^" + std::to_string(q) + " is misshaped");
        std::vector<Elt> e(static_cast<size_t>(nk) * dV * nb * dV, 0);
        std::vector<char> hit(nk, 0);
        for (uint32_t s = 0; s < nb; ++s) {
            int64_t t_found = -1;
            for (uint32_t t = 0; t < nk; ++t) {
                bool nz = false, ident = true;
                for (uint32_t a = 0; a < dV; ++a)
                    for (uint32_t b = 0; b < dV; ++b) {
                        const Elt v = R.get_entry(s * dV + a, t * dV + b);
                        if (v != 0) nz = true;
                        if (v != static_cast<Elt>(a == b ? 1 : 0)) ident = false;
                    }
                if (!nz) continue;
                if (!ident || t_found >= 0)
                    stop("r_flat^" + std::to_string(q) + " is not an identity-fibre block selection (ext_0 undefined)");
                t_found = t;
            }
            if (t_found < 0 || hit[static_cast<uint32_t>(t_found)])
                stop("r_flat^" + std::to_string(q) + " does not come from an injective simplicial inclusion");
            hit[static_cast<uint32_t>(t_found)] = 1;
            for (uint32_t a = 0; a < dV; ++a)
                e[(static_cast<size_t>(t_found) * dV + a) * (static_cast<size_t>(nb) * dV) + s * dV + a] = 1;
        }
        ext0_[q] = (nk == 0 || nb == 0) ? zero(nk * dV, nb * dV) : FlatMatrix(nk * dV, nb * dV, p, e);
        if (q < vanishing.size())
            for (uint32_t t = 0; t < nk; ++t)
                if (hit[t]) { vanishing[q].push_back(t); ++rec_.vanishing_simplices; }
        // R3.1: a section of r, and equivariant (C_eq(K_bd) -> C_eq(K), read on the bulk's own A_C rows).
        if (nb > 0) {
            if (!flat_equal(mul(R, ext0_[q]), FlatMatrix::CreateScalar(nb * dV, 1, p)))
                stop("r ext_0 != I in degree " + std::to_string(q));
            if (nk > 0) {
                const FlatMatrix AC = flat_of_block(bulk.A_C_block(q), p);
                const FlatMatrix Cbd = dense(bdc.C_eq_block(q));
                if (AC.get_rows() > 0 && Cbd.get_cols() > 0 && !flat_is_zero(mul(AC, mul(ext0_[q], Cbd))))
                    stop("ext_0 does not map C^" + std::to_string(q) + "_eq(K_bd) into C^" + std::to_string(q) + "_eq(K)");
            }
        }
    }

    // ---- (2) E = ext_0 o h and Theta' = Theta + d E + E d ([C, eq:route3]) --
    E_.assign(Q + 1, zero(0, 0));
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t cols = cone.bulk_part_dim(q);
        if (q == 0) { E_[q] = zero(0, cols); continue; }
        const uint32_t rows = cone.bulk_part_dim(q - 1);
        const FlatMatrix& h = cert.h[q];                          // boundary_part_dim(q) x cols
        if (h.get_rows() != ext0_[q - 1].get_cols() || h.get_cols() != cols) stop("h^" + std::to_string(q) + " is misshaped");
        E_[q] = (rows == 0 || cols == 0) ? zero(rows, cols) : mul(ext0_[q - 1], h);
        if (!flat_is_zero(E_[q])) rec_.E_nonzero = true;
    }
    Thp_.assign(Q + 1, zero(0, 0));
    for (uint32_t q = 0; q <= Q; ++q) {
        FlatMatrix T = cert.Theta[q];
        if (q > 0 && cone.bulk_part_dim(q - 1) > 0 && T.get_rows() > 0)
            T = add(T, mul(cert.d_bulk[q - 1], E_[q]));           // d E
        if (q + 1 <= Q && cone.bulk_part_dim(q + 1) > 0 && T.get_rows() > 0)
            T = add(T, mul(E_[q + 1], cert.d_bulk[q]));           // E d
        Thp_[q] = std::move(T);
    }

    // ---- (3) R3.3 on C^q_eq(K) ------------------------------------------
    {
        bool rc = true;
        const std::vector<MatrixT<Elt>>& Ab = norm.bulk_operator().get_H_matrices();
        for (uint32_t q = 0; q <= Q; ++q) {
            const FlatMatrix& C = cert.Ceq_bulk[q];
            if (cone.bulk_part_dim(q) == 0 || C.get_cols() == 0) continue;
            const FlatMatrix TC = mul(Thp_[q], C);
            if (!flat_equal(TC, mul(cert.Theta[q], C))) rec_.theta_prime_differs = true;
            if (q + 1 <= Q && cone.bulk_part_dim(q + 1) > 0 &&
                !flat_equal(mul(cert.d_bulk[q], TC), mul(Thp_[q + 1], mul(cert.d_bulk[q], C))))
                stop("Theta' is not a cochain map on C_eq in degree " + std::to_string(q));
            if (NB(q) > 0) {
                if (q + 1 > Q || cert.Theta_bd[q + 1].get_rows() != NB(q) * dV)
                    stop("Theta^bd in boundary degree " + std::to_string(q) + " is missing");
                const FlatMatrix& Tbd = cert.Theta_bd[q + 1];     // boundary degree q
                const FlatMatrix rhs = mul(Tbd, mul(cert.r_flat[q], C));
                if (!flat_equal(mul(cert.r_flat[q], TC), rhs))
                    stop("r Theta' != Theta^bd r on the nose on C_eq in degree " + std::to_string(q));
                if (!flat_equal(mul(cert.r_flat[q], mul(cert.Theta[q], C)), rhs)) rc = false;
            }
            const FlatMatrix He = dense(bulk.H_eq_block(q));
            const FlatMatrix Pe = flat_of_block(bulk.P_block(q), p);
            const MatrixT<Elt> Ap = (He.get_cols() == 0) ? MatrixT<Elt>(0, 0, p) : matrix_of_flat(mul(Pe, mul(Thp_[q], He)));
            if (q >= Ab.size() || !matrices_equal(Ap, Ab[q]))
                stop("P_eq Theta' H_eq != A^{(" + std::to_string(q) + ")}: Theta' does not induce the bulk operator");
        }
        rec_.r_commutes_normative = rc;
    }

    // ---- (4) ker r: the relative sub-complex, computed ---------------------
    {
        const EquivariantCohomologyOptions::ScopedRetainIntermediates keep(true);
        rel_ = std::make_unique<EquivariantCohomologyOnK>(bulk.get_lssc(), vanishing);
        rel_->compute_cohomology();
    }

    // ---- (5) per degree: A', M, the comparison ---------------------------
    Arel_.assign(Q + 1, MatrixT<Elt>(0, 0, p));
    Atr_.assign(Q + 1, MatrixT<Elt>(0, 0, p));
    M_.assign(Q + 1, zero(0, 0));
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t nb = cone.bulk_part_dim(q), fd = cone.flat_dim(q);
        uint32_t dim_c = 0, dim_h = 0;
        FlatMatrix Hr = zero(nb, 0);
        if (q <= K.get_dimension()) {
            dim_c = rel_->C_eq_dimension(q);
            const uint32_t expect = bulk.C_eq_dimension(q) - (q <= Kb.get_dimension() ? bdc.C_eq_dimension(q) : 0u);
            if (dim_c != expect) stop("dim C^" + std::to_string(q) + "_rel != dim C_eq(K) - dim C_eq(K_bd) (r is onto)");
            dim_h = rel_->get_H_eq_dimensions()[q];
            if (dim_h > 0) Hr = dense(rel_->H_eq_block(q));
        }
        rec_.dim_C_rel.push_back(dim_c);
        rec_.dim_H_rel.push_back(dim_h);
        // R3.4: Theta' H_rel inside Z^q_rel.
        const FlatMatrix Y = mul(Thp_[q], Hr);                   // nb x dim_h
        if (dim_h > 0) {
            const FlatMatrix ACr = flat_of_block(rel_->A_C_block(q), p);
            if (ACr.get_rows() > 0 && !flat_is_zero(mul(ACr, Y)))
                stop("Theta' does not preserve ker r in degree " + std::to_string(q));
            if (q + 1 <= Q && cone.bulk_part_dim(q + 1) > 0 && !flat_is_zero(mul(cert.d_bulk[q], Y)))
                stop("Theta' H_rel is not a cocycle in degree " + std::to_string(q));
        }
        const FlatMatrix Pr = (dim_h > 0) ? flat_of_block(rel_->P_block(q), p) : zero(0, nb);
        const FlatMatrix Ar = mul(Pr, Y);                         // dim_h x dim_h
        Arel_[q] = matrix_of_flat(Ar);
        // R3.5: the correspondence c |-> (c, 0).
        const uint32_t dHc = cone.dim_Hc(q);
        if (dHc != dim_h)
            stop("dim H^" + std::to_string(q) + "(ker r) = " + std::to_string(dim_h) + " != dim H^" + std::to_string(q) +
                 "_c = " + std::to_string(dHc) + " (lem:conerelative(3))");
        FlatMatrix X = zero(fd, dim_h);
        if (fd > 0 && dim_h > 0) {
            std::vector<Elt> xd(static_cast<size_t>(fd) * dim_h, 0);
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t j = 0; j < dim_h; ++j) xd[static_cast<size_t>(i) * dim_h + j] = Hr.get_entry(i, j);
            X = FlatMatrix(fd, dim_h, p, xd);
            if (!flat_is_zero(mul(flat_of_block(cone.D_block(q), p), X)))
                stop("(z, 0) is not a cone cocycle in degree " + std::to_string(q));
        }
        FlatMatrix M = zero(dHc, dHc);
        MatrixT<Elt> T(dHc, dHc, p);
        if (dHc > 0) {
            M = mul(flat_of_block(cone.P_block(q), p), X);
            if (M.rank() != dHc)
                stop("the correspondence H^" + std::to_string(q) + "(ker r) -> H^" + std::to_string(q) + "_c is not invertible");
            T = matrix_of_flat(mul(M, mul(Ar, M.inverse())));
        }
        M_[q] = M;
        Atr_[q] = T;
        // R3.6: the comparison, entrywise.
        ++rec_.degrees_compared;
        if (!matrices_equal(T, norm.A_c(q))) rec_.disagreeing_degrees.push_back(q);
    }
    if (!rec_.disagreeing_degrees.empty()) {
        std::ostringstream oss;
        oss << "M A'^{(q)} M^{-1} of route 3 differs from the normative A_c^{(q)} in degree(s)";
        for (uint32_t q : rec_.disagreeing_degrees) oss << " " << q;
        stop(oss.str());
    }
    computed_ = true;
}

const EquivariantCohomologyOnK& HeckeConeRelativeRoute::relative_cohomology() const {
    require_computed("relative_cohomology", 0);
    return *rel_;
}
void HeckeConeRelativeRoute::require_field_route(const char* who) const {
    if (rec_.ring_k >= 2)
        throw std::logic_error(std::string("HeckeConeRelativeRoute::") + who + ": a field (F_ell) accessor, but this route is over " +
                               CoefficientRingText(rec_.ring_k) + " -- use the *_module twin");
}
void HeckeConeRelativeRoute::require_module_route(const char* who) const {
    if (rec_.ring_k < 2)
        throw std::logic_error(std::string("HeckeConeRelativeRoute::") + who + ": a Z/ell^k accessor, but this route is over F_ell");
}
const FlatMatrix& HeckeConeRelativeRoute::ext0_flat(uint32_t q) const { require_computed("ext0_flat", q); require_field_route("ext0_flat"); return ext0_[q]; }
const FlatMatrix& HeckeConeRelativeRoute::E_flat(uint32_t q) const { require_computed("E_flat", q); require_field_route("E_flat"); return E_[q]; }
const FlatMatrix& HeckeConeRelativeRoute::Theta_prime_flat(uint32_t q) const { require_computed("Theta_prime_flat", q); require_field_route("Theta_prime_flat"); return Thp_[q]; }
const MatrixT<uint32_t>& HeckeConeRelativeRoute::A_rel(uint32_t q) const { require_computed("A_rel", q); require_field_route("A_rel"); return Arel_[q]; }
const FlatMatrix& HeckeConeRelativeRoute::correspondence(uint32_t q) const { require_computed("correspondence", q); require_field_route("correspondence"); return M_[q]; }
const MatrixT<uint32_t>& HeckeConeRelativeRoute::A_c_transported(uint32_t q) const { require_computed("A_c_transported", q); require_field_route("A_c_transported"); return Atr_[q]; }
const engine::DenseBuf<uint32_t>& HeckeConeRelativeRoute::Theta_prime_module(uint32_t q) const { require_computed("Theta_prime_module", q); require_module_route("Theta_prime_module"); return Thp_m_[q]; }
const engine::DenseBuf<uint32_t>& HeckeConeRelativeRoute::A_rel_module(uint32_t q) const { require_computed("A_rel_module", q); require_module_route("A_rel_module"); return Arel_m_[q]; }
const engine::DenseBuf<uint32_t>& HeckeConeRelativeRoute::correspondence_module(uint32_t q) const { require_computed("correspondence_module", q); require_module_route("correspondence_module"); return M_m_[q]; }
const engine::DenseBuf<uint32_t>& HeckeConeRelativeRoute::inverse_correspondence_module(uint32_t q) const { require_computed("inverse_correspondence_module", q); require_module_route("inverse_correspondence_module"); return N_m_[q]; }
const engine::DenseBuf<uint32_t>& HeckeConeRelativeRoute::A_c_transported_module(uint32_t q) const { require_computed("A_c_transported_module", q); require_module_route("A_c_transported_module"); return Atr_m_[q]; }

std::string HeckeConeRelativeRoute::to_string() const {
    std::ostringstream oss;
    oss << "HeckeConeRelativeRoute (ell = " << rec_.ell << ", n = " << rec_.n_sym << ", p = " << rec_.p_hecke
        << ", nu = " << rec_.nu << "): ";
    if (!computed_) { oss << "not computed"; return oss.str(); }
    oss << rec_.vanishing_simplices << " forced-zero simplices, dim H^*(ker r) = [";
    for (size_t i = 0; i < rec_.dim_H_rel.size(); ++i) oss << (i ? "," : "") << rec_.dim_H_rel[i];
    oss << "], E " << (rec_.E_nonzero ? "!= 0" : "== 0") << ", A_c agrees in "
        << (rec_.degrees_compared - rec_.disagreeing_degrees.size()) << " / " << rec_.degrees_compared << " degrees";
    return oss.str();
}

// ===========================================================================
// (slot-9 WP-4; E8 item 4b) Route 1 over Z/ell^k -- the header note above
// ModuleConeActionT.
// ===========================================================================
void HeckeConeAction::require_field(const char* who) const {
    if (ring_k_ >= 2)
        throw std::logic_error(std::string("HeckeConeAction::") + who + ": a field (F_ell) accessor, but this cone action is over " +
                               CoefficientRingText(ring_k_) + " -- use module_action() ([E] sec:ring (f))");
}

const ModuleConeActionT<uint32_t>& HeckeConeAction::module_action() const {
    if (ring_k_ < 2) throw std::logic_error("HeckeConeAction::module_action: this cone action is over F_ell");
    require_computed("module_action");
    return *module_;
}

namespace {
template<class Elt>
engine::DenseBuf<Elt> mca_mul(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& A, const engine::DenseBuf<Elt>& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows, "ModuleConeAction: product shape mismatch");
    engine::DenseBuf<Elt> C(A.rows, B.cols);
    if (A.rows > 0 && B.cols > 0 && A.cols > 0) engine::kernels::gemm<Elt>(ctx, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}
template<class Elt>
engine::DenseBuf<Elt> mca_coboundary(const engine::ModCtx<Elt>& ctx, const SimplicialComplex& C, uint32_t q, uint32_t d) {
    const FlatMatrixT<Elt> D1 = C.template coboundary_operator_elt<Elt>(q, ctx.p);
    engine::DenseBuf<Elt> D(D1.get_rows() * d, D1.get_cols() * d);
    for (uint32_t i = 0; i < D1.get_rows(); ++i)
        for (uint32_t j = 0; j < D1.get_cols(); ++j) {
            const Elt v = D1.get_entry(i, j);
            if (v != 0) for (uint32_t a = 0; a < d; ++a) D.at(i * d + a, j * d + a) = v;
        }
    return D;
}
// X (t_tgt x t_src) row i compared in [0, ell^{e_i}) of the target.
template<class Elt>
bool mca_equal_c5(const engine::ModCtx<Elt>& ctx, const engine::DenseBuf<Elt>& X, const engine::DenseBuf<Elt>& Y,
                  const std::vector<uint32_t>& e_tgt, uint64_t& checks) {
    if (X.rows != Y.rows || X.cols != Y.cols || X.rows != e_tgt.size()) return false;
    for (uint32_t i = 0; i < X.rows; ++i) {
        const Elt m = (e_tgt[i] >= ctx.k) ? Elt(0) : ctx.ell_pow(e_tgt[i]);
        for (uint32_t j = 0; j < X.cols; ++j) {
            ++checks;
            const Elt a = m ? static_cast<Elt>(X.at(i, j) % m) : X.at(i, j);
            const Elt b = m ? static_cast<Elt>(Y.at(i, j) % m) : Y.at(i, j);
            if (a != b) return false;
        }
    }
    return true;
}
} // namespace

template<class Elt>
ModuleConeActionT<Elt>::ModuleConeActionT(const ModuleConeT<Elt>* cone, const HeckeOperatorOnKT<Elt>* bulk_op,
                                          const HeckeBoundaryOperatorT<Elt>* bd_op, const HeckePairHomotopyT<Elt>* eta)
    : cone_(cone), bulk_(bulk_op), bd_(bd_op), eta_(eta) {
    if (!cone_ || !bulk_ || !bd_ || !eta_) throw std::invalid_argument("ModuleConeAction: null input");
    if (!cone_->computed()) throw std::runtime_error("ModuleConeAction: the cone is not computed");
    if (!bulk_->is_composite() || !bd_->is_composite())
        throw std::logic_error("ModuleConeAction: the bulk and the boundary operators must be over Z/ell^k");
    ctx_ = cone_->ring_context();
    if (&bulk_->cohomology_of_L() != &cone_->bulk())
        throw std::runtime_error("ModuleConeAction: the bulk operator was built on a different bulk cohomology");
    if (&bd_->cohomology() != &cone_->boundary())
        throw std::runtime_error("ModuleConeAction: the boundary operator was built on a different boundary cohomology");
    if (static_cast<uint64_t>(bd_->ell()) != static_cast<uint64_t>(ctx_.ell) || bd_->d_V() != cone_->d_V())
        throw std::runtime_error("ModuleConeAction: the boundary operator's (ell, d_V) differ from the cone's");
    if (&eta_->cosets() != &bd_->cosets())
        throw std::runtime_error("ModuleConeAction: eta and the boundary operator use different coset tables");
    if (eta_->tau_boundary_table_fnv64() != bd_->taub().table_fnv64())
        throw std::runtime_error("ModuleConeAction: eta was built against a different tau_bd table");
    if (&eta_->tau() != &bulk_->tau())   // the eta-provenance check, as over F_ell
        throw std::runtime_error("ModuleConeAction: eta was not built on the bulk operator's tau");
}

template<class Elt>
void ModuleConeActionT<Elt>::require_computed(const char* who) const {
    if (!computed_) throw std::runtime_error(std::string("ModuleConeAction::") + who + ": compute() has not run");
}

template<class Elt>
void ModuleConeActionT<Elt>::compute() {
    if (computed_) return;
    using engine::DenseBuf;
    const engine::ModCtx<Elt>& ctx = ctx_;
    const Elt p = ctx.p;
    const ModuleConeT<Elt>& C = *cone_;
    const uint32_t d = C.d_V(), n_sym = d - 1, Q = C.q_max();
    const uint32_t top_K = C.top_bulk(), top_bd = C.top_boundary();
    const uint32_t m = eta_->num_cosets();
    const SimplicialComplex* K = C.bulk().get_lssc()->get_base_K_complex();
    const SimplicialComplex* Kb = C.boundary().get_lssc()->get_base_K_complex();
    std::map<std::array<Elt, 4>, DenseBuf<Elt>> rho_memo;
    auto rho = [&](const Mat22& g) -> const DenseBuf<Elt>& {
        std::array<Elt, 4> key;
        if constexpr (std::is_same_v<Elt, uint32_t>) { const auto a = g.reduced_mod(p); key = {a[0], a[1], a[2], a[3]}; }
        else { const auto a = g.reduced_mod64(p); key = {a[0], a[1], a[2], a[3]}; }
        auto it = rho_memo.find(key);
        if (it == rho_memo.end()) {
            const FlatMatrixT<Elt> R = RhoOfExactRing<Elt>(ctx, g, n_sym);
            DenseBuf<Elt> b(d, d);
            for (uint32_t i = 0; i < d; ++i)
                for (uint32_t j = 0; j < d; ++j) b.at(i, j) = R.get_entry(i, j);
            it = rho_memo.emplace(key, std::move(b)).first;
        }
        return it->second;
    };
    auto coeff_ring = [&](int64_t c) -> Elt {
        if (c >= 0) return static_cast<Elt>(static_cast<uint64_t>(c) % static_cast<uint64_t>(p));
        const Elt r = static_cast<Elt>(static_cast<uint64_t>(-c) % static_cast<uint64_t>(p));
        return r == 0 ? Elt(0) : static_cast<Elt>(p - r);
    };
    auto fail = [](const std::string& what) { throw std::runtime_error("ModuleConeAction (Z/ell^k): " + what); };

    // h^q in both bracketings (degrees 0 .. Q + 1; zero where the boundary degree q - 1 has no eta).
    rec_ = Record();
    rec_.routes_bit_identical = true;
    rec_.eta_zero = true;
    std::vector<DenseBuf<Elt>> hh(Q + 2);
    const Mat22& g = eta_->g();
    for (uint32_t q = 0; q <= Q + 1; ++q) {
        const uint32_t rows = C.boundary_part_dim(q), cols = C.bulk_part_dim(q);
        DenseBuf<Elt> F(rows, cols), Dr(rows, cols);
        if (rows > 0 && cols > 0 && q >= 1 && q - 1 < eta_->num_degrees()) {
            const uint32_t qb = q - 1;
            const std::vector<uint32_t>& bslots = eta_->boundary_simplices(qb);
            ENGINE_CHECK_MSG(static_cast<uint64_t>(bslots.size()) * d == rows, "ModuleConeAction: boundary slot list vs the cone's boundary part");
            for (uint32_t s = 0; s < bslots.size(); ++s)
                for (uint32_t j = 1; j <= m; ++j)
                    for (const ChainTerm& ct : eta_->eta(j, qb, bslots[s])) {
                        rec_.eta_zero = false;
                        const Elt c = coeff_ring(ct.coeff);
                        if (c == 0) continue;
                        const uint64_t col0 = static_cast<uint64_t>(ct.sigma_idx) * d;
                        ENGINE_CHECK_MSG(col0 + d <= cols, "ModuleConeAction: an eta term names a simplex outside C^q(K)");
                        const DenseBuf<Elt> Bf = mca_mul(ctx, rho(eta_->cosets().s_inv(j)), rho(g * ct.gamma));
                        const DenseBuf<Elt>& Bd = rho(eta_->cosets().delta(j) * ct.gamma);
                        for (uint32_t a = 0; a < d; ++a)
                            for (uint32_t b = 0; b < d; ++b) {
                                Elt& ef = F.at(s * d + a, static_cast<uint32_t>(col0) + b);
                                ef = ctx.add(ef, ctx.mul(c, Bf.at(a, b)));
                                Elt& ed = Dr.at(s * d + a, static_cast<uint32_t>(col0) + b);
                                ed = ctx.add(ed, ctx.mul(c, Bd.at(a, b)));
                            }
                    }
            for (uint32_t i = 0; i < rows && rec_.routes_bit_identical; ++i)
                for (uint32_t j = 0; j < cols; ++j)
                    if (F.at(i, j) != Dr.at(i, j)) { rec_.routes_bit_identical = false; break; }
            if (!rec_.routes_bit_identical) fail("the fused and the direct bracketing of h differ at q = " + std::to_string(q));
        }
        hh[q] = std::move(F);
    }
    // Phi^q = [[Theta^q, 0], [h^q, Theta^bd_{q-1}]].
    std::vector<DenseBuf<Elt>> Ph(Q + 2);
    for (uint32_t q = 0; q <= Q + 1; ++q) {
        const uint32_t nK = C.bulk_part_dim(q), nb = C.boundary_part_dim(q);
        DenseBuf<Elt> P(nK + nb, nK + nb);
        if (nK > 0) {
            const DenseBuf<Elt>& T = bulk_->Theta_module(q);
            ENGINE_CHECK_MSG(T.rows == nK && T.cols == nK, "ModuleConeAction: Theta shape");
            for (uint32_t i = 0; i < nK; ++i)
                for (uint32_t j = 0; j < nK; ++j) P.at(i, j) = T.at(i, j);
        }
        if (nb > 0) {
            const DenseBuf<Elt>& Tb = bd_->Theta_module(q - 1);
            ENGINE_CHECK_MSG(Tb.rows == nb && Tb.cols == nb, "ModuleConeAction: Theta^bd shape");
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t j = 0; j < nb; ++j) P.at(nK + i, nK + j) = Tb.at(i, j);
            for (uint32_t i = 0; i < nb; ++i)
                for (uint32_t j = 0; j < nK; ++j) P.at(nK + i, j) = hh[q].at(i, j);
        }
        Ph[q] = std::move(P);
    }
    // Item 2: the residual (d h + h d + r Theta - Theta^bd r) kills the bulk C^q_eq basis.
    for (uint32_t q = 0; q <= std::min(top_K, top_bd); ++q) {
        const DenseBuf<Elt>& CK = C.bulk().module_degree(q).C;
        if (CK.cols == 0) continue;
        const DenseBuf<Elt>& R = C.r_flat(q);
        DenseBuf<Elt> res = mca_mul(ctx, R, mca_mul(ctx, bulk_->Theta_module(q), CK));                  // r Theta
        const DenseBuf<Elt> tr = mca_mul(ctx, bd_->Theta_module(q), mca_mul(ctx, R, CK));              // Theta^bd r
        if (q >= 1 && hh[q].rows > 0) {                                                                // d_bd h^q
            const DenseBuf<Elt> dh = mca_mul(ctx, mca_coboundary(ctx, *Kb, q - 1, d), mca_mul(ctx, hh[q], CK));
            for (uint32_t i = 0; i < res.rows; ++i) for (uint32_t j = 0; j < res.cols; ++j) res.at(i, j) = ctx.add(res.at(i, j), dh.at(i, j));
        }
        if (q + 1 <= Q + 1 && hh[q + 1].rows > 0 && q + 1 <= top_K) {                                  // h^{q+1} d
            const DenseBuf<Elt> hd = mca_mul(ctx, hh[q + 1], mca_mul(ctx, mca_coboundary(ctx, *K, q, d), CK));
            for (uint32_t i = 0; i < res.rows; ++i) for (uint32_t j = 0; j < res.cols; ++j) res.at(i, j) = ctx.add(res.at(i, j), hd.at(i, j));
        }
        for (uint32_t i = 0; i < res.rows; ++i)
            for (uint32_t j = 0; j < res.cols; ++j) {
                ++rec_.residual_checks;
                if (res.at(i, j) != tr.at(i, j)) fail("item 2 (obl:coneexact(i)): d h + h d + r Theta - Theta^bd r != 0 on C^" + std::to_string(q) + "_eq");
            }
    }
    // Items 4, 5: D_c Phi = Phi D_c on Cc^q_eq; D_c Phi z = 0 on the cone cocycles.
    for (uint32_t q = 0; q <= Q; ++q) {
        const auto& G = C.degree(q);
        if (G.C.cols == 0) continue;
        const DenseBuf<Elt> D = C.D_flat(q);
        const DenseBuf<Elt> lhs = mca_mul(ctx, D, mca_mul(ctx, Ph[q], G.C));
        const DenseBuf<Elt> rhs = mca_mul(ctx, Ph[q + 1], mca_mul(ctx, D, G.C));
        for (uint32_t i = 0; i < lhs.rows; ++i)
            for (uint32_t j = 0; j < lhs.cols; ++j) {
                ++rec_.chain_map_checks;
                if (lhs.at(i, j) != rhs.at(i, j)) fail("item 4: D_c Phi != Phi D_c on Cc^" + std::to_string(q) + "_eq");
            }
        if (G.Zc.cols > 0) {
            const DenseBuf<Elt> z = mca_mul(ctx, D, mca_mul(ctx, Ph[q], mca_mul(ctx, G.C, G.Zc)));
            for (uint32_t i = 0; i < z.rows; ++i)
                for (uint32_t j = 0; j < z.cols; ++j) {
                    ++rec_.cocycle_checks;
                    if (z.at(i, j) != 0) fail("item 5: D_c Phi z != 0 on a cone cocycle at q = " + std::to_string(q));
                }
        }
    }
    // Item 6: A_c^q on the cyclic summands.
    std::vector<DenseBuf<Elt>> A(Q + 1);
    for (uint32_t q = 0; q <= Q; ++q) {
        const DenseBuf<Elt>& Hb = C.H_module_basis(q);
        DenseBuf<Elt> M(Hb.cols, Hb.cols);
        if (Hb.cols > 0) {
            const DenseBuf<Elt> Y = mca_mul(ctx, Ph[q], Hb);
            const auto co = C.H_class_coordinates(q, engine::cview(Y));
            for (uint32_t j = 0; j < Hb.cols; ++j)
                for (uint32_t i = 0; i < Hb.cols; ++i) M.at(i, j) = co[j][i];
        }
        A[q] = std::move(M);
    }
    // Items 7-9: every map of eq:coneles intertwines (T^bd, T^c, T) on the nose.
    for (uint32_t q = 0; q <= Q; ++q) {
        if (C.pi_built(q)) {
            const DenseBuf<Elt>& P = C.pi(q);
            if (!mca_equal_c5(ctx, mca_mul(ctx, P, A[q]), mca_mul(ctx, bulk_->H_matrix_module(q), P),
                              C.bulk().H_invariant_factors(q), rec_.intertwining_checks))
                fail("item 7: pi A_c != A pi at q = " + std::to_string(q));
        }
        if (C.r_built(q) && q <= top_bd) {
            const DenseBuf<Elt>& R = C.r(q);
            if (!mca_equal_c5(ctx, mca_mul(ctx, R, bulk_->H_matrix_module(q)), mca_mul(ctx, bd_->H_matrix_module(q), R),
                              C.boundary().H_invariant_factors(q), rec_.intertwining_checks))
                fail("item 8: r A != A_bd r at q = " + std::to_string(q));
        }
        if (C.iota_built(q)) {
            const DenseBuf<Elt>& I = C.iota(q);
            if (!mca_equal_c5(ctx, mca_mul(ctx, I, bd_->H_matrix_module(q)), mca_mul(ctx, A[q + 1], I),
                              C.H_invariant_factors(q + 1), rec_.intertwining_checks))
                fail("item 9: iota A_bd != A_c iota at q = " + std::to_string(q));
        }
    }
    hh.resize(Q + 1);
    Ph.resize(Q + 1);
    h_ = std::move(hh);
    Phi_ = std::move(Ph);
    A_ = std::move(A);
    computed_ = true;
}

template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeActionT<Elt>::h(uint32_t q) const {
    require_computed("h");
    if (q >= h_.size()) throw std::out_of_range("ModuleConeAction::h: degree out of range");
    return h_[q];
}
template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeActionT<Elt>::Phi(uint32_t q) const {
    require_computed("Phi");
    if (q >= Phi_.size()) throw std::out_of_range("ModuleConeAction::Phi: degree out of range");
    return Phi_[q];
}
template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeActionT<Elt>::A_c(uint32_t q) const {
    require_computed("A_c");
    if (q >= A_.size()) throw std::out_of_range("ModuleConeAction::A_c: degree out of range");
    return A_[q];
}

template class ModuleConeActionT<uint32_t>;
template class ModuleConeActionT<uint64_t>;

// ===========================================================================
// (hecke-extensions slot 10, WP-3; slot-10 decision 3 (A)) [C, obl:coneexact]
// (vii) OVER Z/ell^k: the strict-pair route and route 3 on the dense module
// objects of slot-9 WP-4 (the header notes of the two classes).  Only a
// composite normative action reaches these functions (ell >= 5; the cone
// refuses ell in {2, 3}), so every failure throws.  The F_ell paths above are
// untouched.
// ===========================================================================
namespace {
using CBuf = engine::DenseBuf<uint32_t>;

// Row i of X reduced into [0, ell^{e_i}) ((c5) form; e_i >= k: no reduction).
void cx_reduce_rows(const ModCtx<uint32_t>& ctx, CBuf& X, const std::vector<uint32_t>& e) {
    ENGINE_CHECK_MSG(X.rows == e.size(), "cross-route (Z/ell^k): (c5) row count != #summands of the target");
    for (uint32_t i = 0; i < X.rows; ++i) {
        if (e[i] >= ctx.k) continue;
        const uint32_t m = ctx.ell_pow(e[i]);
        for (uint32_t j = 0; j < X.cols; ++j) X.at(i, j) = X.at(i, j) % m;
    }
}
// The (c5) product A B (a composite of homomorphisms; well defined, see
// HeckeReductionConeValidation's c5_commute note), rows reduced for e_tgt.
CBuf cx_mul_c5(const ModCtx<uint32_t>& ctx, const CBuf& A, const CBuf& B, const std::vector<uint32_t>& e_tgt) {
    CBuf P = mca_mul(ctx, A, B);
    cx_reduce_rows(ctx, P, e_tgt);
    return P;
}
bool cx_equal(const CBuf& a, const CBuf& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}
bool cx_is_identity_c5(const ModCtx<uint32_t>& ctx, const CBuf& X, const std::vector<uint32_t>& e, uint64_t& checks) {
    if (X.rows != X.cols || X.rows != e.size()) return false;
    for (uint32_t i = 0; i < X.rows; ++i) {
        const uint32_t m = (e[i] >= ctx.k) ? 0u : ctx.ell_pow(e[i]);
        for (uint32_t j = 0; j < X.cols; ++j) {
            ++checks;
            const uint32_t v = m ? X.at(i, j) % m : X.at(i, j);
            const uint32_t want = (i == j) ? (m == 1 ? 0u : 1u) : 0u;
            if (v != want) return false;
        }
    }
    return true;
}
CBuf cx_add(const ModCtx<uint32_t>& ctx, CBuf A, const CBuf& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols, "cross-route (Z/ell^k): sum shape mismatch");
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) A.at(i, j) = ctx.add(A.at(i, j), B.at(i, j));
    return A;
}
// Columns of coordinate lists (co[col][i]) as a t x ncol matrix.
CBuf cx_of_coords(const std::vector<std::vector<uint32_t>>& co, uint32_t t) {
    CBuf M(t, static_cast<uint32_t>(co.size()));
    for (uint32_t c = 0; c < co.size(); ++c) {
        ENGINE_CHECK_MSG(co[c].size() == t, "cross-route (Z/ell^k): a coordinate column has the wrong length");
        for (uint32_t i = 0; i < t; ++i) M.at(i, c) = co[c][i];
    }
    return M;
}
} // namespace

void HeckeConeStrictPairRoute::compute_composite() {
    const HeckeConeAction& norm = *norm_;
    const HeckeOperatorOnKT<uint32_t>& bulk = norm.bulk_operator();
    const HeckeBoundaryOperatorT<uint32_t>& bd = norm.boundary_operator();
    const HeckeTauTableT<uint32_t>& tau = bulk.tau();
    const HeckeBoundaryTauTableT<uint32_t>& taub = bd.taub();
    const ModuleConeT<uint32_t>& C = norm.cone().module_cone();
    const ModCtx<uint32_t>& ctx = C.ring_context();

    rec_ = Record{};
    rec_.ell = norm.ell();
    rec_.n_sym = norm.n_sym();
    rec_.nu = norm.nu();
    rec_.p_hecke = norm.p_hecke();
    rec_.q_max = norm.q_max();
    rec_.ring_k = norm.coefficient_ring_k();
    rec_.six_invertible = (rec_.ell >= 5);
    ENGINE_CHECK_MSG(rec_.six_invertible, "HeckeConeStrictPairRoute (Z/ell^k): a composite cone exists only at ell >= 5");
    rec_.asserted = true;
    rec_.tau_normative_fnv64 = norm.eta().tau_table_fnv64();

    ENGINE_CHECK_MSG(&norm.eta().tau() == &tau,
                     "HeckeConeStrictPairRoute: the normative eta was not built on the bulk operator's tau");
    ENGINE_CHECK_MSG(&bulk.cosets() == &bd.cosets() && &tau.cosets() == &bulk.cosets(),
                     "HeckeConeStrictPairRoute: the normative objects use different coset tables");
    ENGINE_CHECK_MSG(!tau.is_strict_pair(), "HeckeConeStrictPairRoute: the normative tau is itself the strict-pair table");

    const auto stop = [&](const std::string& what) {
        std::ostringstream oss;
        oss << "HeckeConeStrictPairRoute (" << CoefficientRingText(rec_.ring_k) << ", ell = " << rec_.ell << ", n = " << rec_.n_sym
            << ", p = " << rec_.p_hecke << ", nu = " << rec_.nu << "): " << what
            << " -- at 6 in R^x this contradicts [C, prop:coneroutes](1) / [C, prop:strictpair]; ABORTED "
               "([C, obl:coneexact](vii) over Z/ell^k).";
        throw std::runtime_error(oss.str());
    };

    try {
        // (1) tau^s: integral, ring-free -- the SAME construction as over F_ell (or the seam's table).
        if (tau_ext_ != nullptr) {
            rec_.external_table = true;
            if (!tau_ext_->is_strict_pair() || &tau_ext_->cosets() != &bulk.cosets() ||
                tau_ext_->strict_boundary_table() != &taub)
                throw std::runtime_error("the supplied table is not the strict-pair table of the normative (cosets, tau_bd)");
            const uint32_t code = tau_ext_->certify();
            if (code != 0)
                throw std::runtime_error("the supplied strict table fails its certificate (code " + std::to_string(code) + ")");
        } else {
            tau_s_ = std::make_unique<HeckeTauTableT<uint32_t>>(tau.complex(), tau.model(), bulk.cosets(), tau.ell(),
                                                                tau.n_sym(), taub);
        }
        const HeckeTauTableT<uint32_t>& ts = *tau_strict();
        // (2) eta^s on (tau^s, tau_bd): integral, certified as always.
        eta_s_ = std::make_unique<HeckePairHomotopyT<uint32_t>>(ts, taub, norm.eta_policy());
        // (3) Theta^s: HeckeOperatorOnK's module route on the SAME composite bulk cohomology.
        bulk_s_ = std::make_unique<HeckeOperatorOnKT<uint32_t>>(&bulk.cohomology_of_L(), &bulk.cosets(), &ts);
        if (!bulk_s_->is_composite()) throw std::runtime_error("Theta^s was not built over Z/ell^k");
        bulk_s_->compute();
        // (4) Phi^s: route 1 on the SAME composite cone and boundary operator, eta^s.
        act_s_ = std::make_unique<HeckeConeAction>(&norm.cone(), bulk_s_.get(), &bd, eta_s_.get());
        if (!act_s_->is_composite()) throw std::runtime_error("Phi^s was not built over Z/ell^k");
        act_s_->compute();
        rec_.built = true;
    } catch (const std::exception& e) {
        act_s_.reset();
        bulk_s_.reset();
        eta_s_.reset();
        tau_s_.reset();
        stop(std::string("the strict-pair datum could not be built or certified: ") + e.what());
    }

    const HeckeTauTableT<uint32_t>& ts = *tau_strict();
    rec_.solve_regime = ts.solve_regime();
    rec_.fills_int64 = ts.num_fills_int64();
    rec_.fills_widened = ts.num_fills_widened();
    rec_.strict_boundary_slots = ts.num_strict_boundary_slots();
    rec_.eta_zero = eta_s_->eta_zero();
    rec_.eta_nonzero_slots = eta_s_->num_nonzero_slots();
    rec_.tau_strict_fnv64 = eta_s_->tau_table_fnv64();
    rec_.tau_differs = (rec_.tau_strict_fnv64 != rec_.tau_normative_fnv64);
    ENGINE_CHECK_MSG(rec_.solve_regime == "Z", "HeckeConeStrictPairRoute: the strict-pair table is not over Z");
    ENGINE_CHECK_MSG(rec_.eta_zero, "HeckeConeStrictPairRoute: eta on (tau^s, tau_bd) is not identically zero");

    // cochain_commutes over the ring, on the bulk C^q_eq basis: r Theta^s C == Theta^bd r C.
    {
        bool cc = true;
        for (uint32_t q = 0; q <= std::min(C.top_bulk(), C.top_boundary()); ++q) {
            const CBuf& CK = C.bulk().module_degree(q).C;
            if (CK.cols == 0) continue;
            const CBuf& R = C.r_flat(q);
            if (!cx_equal(mca_mul(ctx, R, mca_mul(ctx, bulk_s_->Theta_module(q), CK)),
                          mca_mul(ctx, bd.Theta_module(q), mca_mul(ctx, R, CK))))
                cc = false;
        }
        rec_.cochain_commutes = cc;
    }

    // (5) The comparison, entrywise, in the cone's one stored cyclic-summand basis.
    const ModuleConeActionT<uint32_t>& A0 = norm.module_action();
    const ModuleConeActionT<uint32_t>& As = act_s_->module_action();
    for (uint32_t q = 0; q <= rec_.q_max; ++q) {
        ++rec_.degrees_compared;
        if (!cx_equal(A0.A_c(q), As.A_c(q))) rec_.disagreeing_degrees.push_back(q);
    }
    if (!rec_.disagreeing_degrees.empty()) {
        std::ostringstream oss;
        oss << "A_c^{(q)} of the strict pair differs from the normative cone's in degree(s)";
        for (uint32_t q : rec_.disagreeing_degrees) oss << " " << q;
        stop(oss.str());
    }
    computed_ = true;
}

void HeckeConeRelativeRoute::compute_composite() {
    const HeckeConeAction& norm = *norm_;
    const ModuleConeT<uint32_t>& C = norm.cone().module_cone();
    const ModuleConeActionT<uint32_t>& MA = norm.module_action();
    const HeckeOperatorOnKT<uint32_t>& bop = norm.bulk_operator();
    const HeckeBoundaryOperatorT<uint32_t>& bdop = norm.boundary_operator();
    const EquivariantCohomologyOnK& bulk = C.bulk();
    const EquivariantCohomologyOnK& bdc = C.boundary();
    const SimplicialComplex& K = *bulk.get_lssc()->get_base_K_complex();
    const SimplicialComplex& Kb = *bdc.get_lssc()->get_base_K_complex();
    const ModCtx<uint32_t>& ctx = C.ring_context();
    const uint32_t dV = C.d_V(), Q = C.q_max(), topK = C.top_bulk(), topB = C.top_boundary();

    rec_ = Record{};
    rec_.ell = norm.ell();
    rec_.n_sym = norm.n_sym();
    rec_.nu = norm.nu();
    rec_.p_hecke = norm.p_hecke();
    rec_.q_max = Q;
    rec_.ring_k = norm.coefficient_ring_k();

    const auto stop = [&](const std::string& what) {
        std::ostringstream oss;
        oss << "HeckeConeRelativeRoute (" << CoefficientRingText(rec_.ring_k) << ", ell = " << rec_.ell << ", n = " << rec_.n_sym
            << ", p = " << rec_.p_hecke << ", nu = " << rec_.nu << "): " << what
            << " -- contradicts [C, prop:coneroutes](2) / [C, lem:conerelative]; ABORTED "
               "([C, obl:coneexact](vii) over Z/ell^k).";
        throw std::runtime_error(oss.str());
    };
    const auto NK = [&](uint32_t q) -> uint32_t { return q <= topK ? K.get_num_q_simplices(q) : 0; };
    const auto NB = [&](uint32_t q) -> uint32_t { return q <= topB ? Kb.get_num_q_simplices(q) : 0; };
    const auto solves = [&](const CBuf& basis, const CBuf& Y) -> bool {
        if (Y.cols == 0) return true;
        if (basis.cols == 0) {
            for (uint32_t i = 0; i < Y.rows; ++i)
                for (uint32_t j = 0; j < Y.cols; ++j)
                    if (Y.at(i, j) != 0) return false;
            return true;
        }
        try {
            (void)engine::elim::solve_multi_rhs_module<uint32_t>(ctx, engine::cview(basis), engine::cview(Y));
        } catch (const std::runtime_error&) {
            return false;
        }
        return true;
    };
    // d^q (x) I of K (bulk_part_dim(q+1) x bulk_part_dim(q)); only for q + 1 <= topK.
    std::vector<CBuf> dK(Q + 1);
    for (uint32_t q = 0; q + 1 <= topK && q <= Q; ++q) dK[q] = mca_coboundary(ctx, K, q, dV);

    // ---- (1) ext_0 from the stored flat restriction, and ker r's simplices; R3.1 ----
    std::vector<CBuf> ext0(Q + 1);
    std::vector<std::vector<uint32_t>> vanishing(std::min<uint32_t>(Q, topK) + 1);
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t nk = NK(q), nb = NB(q);
        ext0[q] = CBuf(nk * dV, nb * dV);
        if (nb == 0 || nk == 0) continue;
        if (q > std::min(topK, topB)) stop("no flat restriction in degree " + std::to_string(q));
        const CBuf& R = C.r_flat(q);
        if (R.rows != nb * dV || R.cols != nk * dV) stop("r_flat^" + std::to_string(q) + " is misshaped");
        std::vector<char> hit(nk, 0);
        for (uint32_t s = 0; s < nb; ++s) {
            int64_t t_found = -1;
            for (uint32_t t = 0; t < nk; ++t) {
                bool nz = false, ident = true;
                for (uint32_t a = 0; a < dV; ++a)
                    for (uint32_t b = 0; b < dV; ++b) {
                        const uint32_t v = R.at(s * dV + a, t * dV + b);
                        if (v != 0) nz = true;
                        if (v != (a == b ? 1u : 0u)) ident = false;
                    }
                if (!nz) continue;
                if (!ident || t_found >= 0)
                    stop("r_flat^" + std::to_string(q) + " is not an identity-fibre block selection (ext_0 undefined)");
                t_found = t;
            }
            if (t_found < 0 || hit[static_cast<uint32_t>(t_found)])
                stop("r_flat^" + std::to_string(q) + " does not come from an injective simplicial inclusion");
            hit[static_cast<uint32_t>(t_found)] = 1;
            for (uint32_t a = 0; a < dV; ++a) ext0[q].at(static_cast<uint32_t>(t_found) * dV + a, s * dV + a) = 1;
        }
        if (q < vanishing.size())
            for (uint32_t t = 0; t < nk; ++t)
                if (hit[t]) { vanishing[q].push_back(t); ++rec_.vanishing_simplices; }
        // R3.1: a section of r (r ext_0 = I), mapping C^q_eq(K_bd) into C^q_eq(K) (a module solve).
        const CBuf RE = mca_mul(ctx, R, ext0[q]);
        for (uint32_t i = 0; i < RE.rows; ++i)
            for (uint32_t j = 0; j < RE.cols; ++j)
                if (RE.at(i, j) != (i == j ? 1u : 0u)) stop("r ext_0 != I in degree " + std::to_string(q));
        if (!solves(bulk.module_degree(q).C, mca_mul(ctx, ext0[q], bdc.module_degree(q).C)))
            stop("ext_0 does not map C^" + std::to_string(q) + "_eq(K_bd) into C^" + std::to_string(q) + "_eq(K)");
    }

    // ---- (2) E = ext_0 o h and Theta' = Theta + d E + E d ([C, eq:route3]) ----
    std::vector<CBuf> E(Q + 2);
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t cols = C.bulk_part_dim(q);
        if (q == 0) { E[q] = CBuf(0, cols); continue; }
        const uint32_t rows = C.bulk_part_dim(q - 1);
        const CBuf& h = MA.h(q);                               // boundary_part_dim(q) x cols
        if (h.rows != ext0[q - 1].cols || h.cols != cols) stop("h^" + std::to_string(q) + " is misshaped");
        E[q] = (rows == 0 || cols == 0) ? CBuf(rows, cols) : mca_mul(ctx, ext0[q - 1], h);
        for (uint32_t i = 0; i < E[q].rows && !rec_.E_nonzero; ++i)
            for (uint32_t j = 0; j < E[q].cols; ++j)
                if (E[q].at(i, j) != 0) { rec_.E_nonzero = true; break; }
    }
    Thp_m_.assign(Q + 1, CBuf());
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t nK = C.bulk_part_dim(q);
        if (nK == 0) { Thp_m_[q] = CBuf(0, 0); continue; }
        CBuf T = bop.Theta_module(q);
        if (q >= 1 && C.bulk_part_dim(q - 1) > 0) T = cx_add(ctx, std::move(T), mca_mul(ctx, dK[q - 1], E[q]));   // d E
        if (q + 1 <= Q && q + 1 <= topK && C.bulk_part_dim(q + 1) > 0)
            T = cx_add(ctx, std::move(T), mca_mul(ctx, E[q + 1], dK[q]));                                             // E d
        Thp_m_[q] = std::move(T);
    }

    // ---- (3) R3.3 on the bulk C^q_eq basis ----
    {
        bool rc = true;
        for (uint32_t q = 0; q <= std::min(Q, topK); ++q) {
            const CBuf& CK = bulk.module_degree(q).C;
            if (C.bulk_part_dim(q) == 0 || CK.cols == 0) continue;
            const CBuf TC = mca_mul(ctx, Thp_m_[q], CK);
            if (!cx_equal(TC, mca_mul(ctx, bop.Theta_module(q), CK))) rec_.theta_prime_differs = true;
            if (q + 1 <= topK && q + 1 <= Q &&
                !cx_equal(mca_mul(ctx, dK[q], TC), mca_mul(ctx, Thp_m_[q + 1], mca_mul(ctx, dK[q], CK))))
                stop("Theta' is not a cochain map on C_eq in degree " + std::to_string(q));
            if (NB(q) > 0) {
                const CBuf& R = C.r_flat(q);
                const CBuf rhs = mca_mul(ctx, bdop.Theta_module(q), mca_mul(ctx, R, CK));
                if (!cx_equal(mca_mul(ctx, R, TC), rhs))
                    stop("r Theta' != Theta^bd r on the nose on C_eq in degree " + std::to_string(q));
                if (!cx_equal(mca_mul(ctx, R, mca_mul(ctx, bop.Theta_module(q), CK)), rhs)) rc = false;
            }
            const CBuf& Hb = bulk.H_module_basis(q);
            if (Hb.cols > 0) {
                std::vector<std::vector<uint32_t>> co;
                try {
                    co = bulk.H_class_coordinates(q, engine::cview(mca_mul(ctx, Thp_m_[q], Hb)));
                } catch (const std::runtime_error& e) {
                    stop("Theta' H_eq is not a bulk cocycle in degree " + std::to_string(q) + " (" + e.what() + ")");
                }
                if (!cx_equal(cx_of_coords(co, Hb.cols), bop.H_matrix_module(q)))
                    stop("Theta' does not induce the bulk operator A^{(" + std::to_string(q) + ")}");
            }
        }
        rec_.r_commutes_normative = rc;
    }

    // ---- (4) ker r: the relative sub-complex over Z/ell^k (its module route) ----
    rel_ = std::make_unique<EquivariantCohomologyOnK>(bulk.get_lssc(), vanishing);
    if (!rel_->is_composite() || rel_->is_flat_route()) stop("the relative sub-complex was not built by the module route");
    rel_->compute_cohomology();

    // ---- (5) per degree: A', M, N, the comparison ----
    Arel_m_.assign(Q + 1, CBuf());
    M_m_.assign(Q + 1, CBuf());
    N_m_.assign(Q + 1, CBuf());
    Atr_m_.assign(Q + 1, CBuf());
    for (uint32_t q = 0; q <= Q; ++q) {
        const uint32_t nK = C.bulk_part_dim(q), fd = C.flat_dim(q);
        const std::vector<uint32_t>& ec = C.H_invariant_factors(q);
        const uint32_t tc = static_cast<uint32_t>(ec.size());
        std::vector<uint32_t> er;
        CBuf Hr(nK, 0);
        uint32_t rank_c = 0;
        if (q <= topK) {
            rank_c = rel_->C_eq_rank(q);
            const uint32_t expect = bulk.C_eq_rank(q) - (q <= topB ? bdc.C_eq_rank(q) : 0u);
            if (rank_c != expect) stop("rank C^" + std::to_string(q) + "_rel != rank C_eq(K) - rank C_eq(K_bd) (r is onto)");
            er = rel_->H_invariant_factors(q);
            Hr = rel_->H_module_basis(q);
        }
        const uint32_t tr = static_cast<uint32_t>(er.size());
        rec_.dim_C_rel.push_back(rank_c);
        rec_.dim_H_rel.push_back(tr);
        rec_.H_rel_invariant_factors.push_back(er);
        if (er != ec)
            stop("the invariant factors of H^" + std::to_string(q) + "(ker r) and H^" + std::to_string(q) +
                 "_c differ (lem:conerelative(3))");
        // R3.4: Theta' H_rel lies in the cocycles of ker r; A' its (c5) matrix.
        CBuf Ar(tr, tr);
        if (tr > 0) {
            std::vector<std::vector<uint32_t>> co;
            try {
                co = rel_->H_class_coordinates(q, engine::cview(mca_mul(ctx, Thp_m_[q], Hr)));
            } catch (const std::runtime_error& e) {
                stop("Theta' does not preserve the cocycles of ker r in degree " + std::to_string(q) + " (" + e.what() + ")");
            }
            Ar = cx_of_coords(co, tr);
        }
        // R3.5: M (z |-> [(z, 0)]) and N ([(c, c_bd)] |-> [c + d ext_0 c_bd]), mutually inverse.
        CBuf M(tc, tr), N(tr, tc);
        if (tr > 0) {
            CBuf X(fd, tr);
            for (uint32_t i = 0; i < nK; ++i)
                for (uint32_t j = 0; j < tr; ++j) X.at(i, j) = Hr.at(i, j);
            std::vector<std::vector<uint32_t>> co;
            try {
                co = C.H_class_coordinates(q, engine::cview(X));
            } catch (const std::runtime_error& e) {
                stop("(z, 0) is not a cone cocycle in degree " + std::to_string(q) + " (" + e.what() + ")");
            }
            M = cx_of_coords(co, tc);
            const CBuf& Bc = C.H_module_basis(q);                  // fd x tc
            const uint32_t nb = fd - nK;
            CBuf Y(nK, tc);
            for (uint32_t i = 0; i < nK; ++i)
                for (uint32_t j = 0; j < tc; ++j) Y.at(i, j) = Bc.at(i, j);
            if (q >= 1 && nb > 0) {
                CBuf cb(nb, tc);
                for (uint32_t i = 0; i < nb; ++i)
                    for (uint32_t j = 0; j < tc; ++j) cb.at(i, j) = Bc.at(nK + i, j);
                if (ext0[q - 1].cols != nb || q > topK) stop("the cone's boundary part is misshaped in degree " + std::to_string(q));
                Y = cx_add(ctx, std::move(Y), mca_mul(ctx, dK[q - 1], mca_mul(ctx, ext0[q - 1], cb)));
            }
            try {
                co = rel_->H_class_coordinates(q, engine::cview(Y));
            } catch (const std::runtime_error& e) {
                stop("c + d ext_0 c_bd is not a cocycle of ker r in degree " + std::to_string(q) + " (" + e.what() + ")");
            }
            N = cx_of_coords(co, tr);
            if (!cx_is_identity_c5(ctx, cx_mul_c5(ctx, M, N, ec), ec, rec_.c5_checks))
                stop("M N != I in degree " + std::to_string(q) + " (lem:conerelative(3))");
            if (!cx_is_identity_c5(ctx, cx_mul_c5(ctx, N, M, er), er, rec_.c5_checks))
                stop("N M != I in degree " + std::to_string(q) + " (lem:conerelative(3))");
        }
        // R3.6: M A' N == A_c^{(q)}, entrywise in (c5) form.
        const CBuf T = (tc > 0) ? cx_mul_c5(ctx, M, cx_mul_c5(ctx, Ar, N, er), ec) : CBuf(0, 0);
        ++rec_.degrees_compared;
        const CBuf& Ac = MA.A_c(q);
        rec_.c5_checks += static_cast<uint64_t>(T.rows) * T.cols;
        if (!cx_equal(T, Ac)) rec_.disagreeing_degrees.push_back(q);
        Arel_m_[q] = std::move(Ar);
        M_m_[q] = std::move(M);
        N_m_[q] = std::move(N);
        Atr_m_[q] = T;
    }
    if (!rec_.disagreeing_degrees.empty()) {
        std::ostringstream oss;
        oss << "M A'^{(q)} N of route 3 differs from the normative A_c^{(q)} in degree(s)";
        for (uint32_t q : rec_.disagreeing_degrees) oss << " " << q;
        stop(oss.str());
    }
    computed_ = true;
}

} // namespace hecke
