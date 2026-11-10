// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 9, WP-4 (E8 item 4b) -- THE DENSE MODULE ROUTE OF THE
// CONE OVER Z/ell^k.  See EquivariantMappingCone.h (the note above
// ConeModuleCertificate) and [E] v1.10.0-pre.1 sec:ring (f).  The F_ell cone
// (EquivariantMappingCone.cpp, the tape route) is not touched by this file.

#include "EquivariantMappingCone.h"

#include "SimplicialCohomology.h"
#include "../ModArithmetic/DenseElim.h"
#include "../ModArithmetic/DenseView.h"
#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/Kernels.h"
#include "../ModArithmetic/ModCtx.h"
#include "Hecke/HeckeSettings.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using engine::cview;
using engine::view;

template<class Elt>
DenseBuf<Elt> mc_mul(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows, "ModuleCone: product shape mismatch");
    DenseBuf<Elt> C(A.rows, B.cols);
    if (A.rows > 0 && B.cols > 0 && A.cols > 0) engine::kernels::gemm<Elt>(ctx, cview(A), cview(B), view(C));
    return C;
}

template<class Elt>
DenseBuf<Elt> mc_transpose(const DenseBuf<Elt>& A) {
    DenseBuf<Elt> T(A.cols, A.rows);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    return T;
}

template<class Elt>
DenseBuf<Elt> mc_hcat(const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows, "ModuleCone: hcat height mismatch");
    DenseBuf<Elt> C(A.rows, A.cols + B.cols);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) C.at(i, A.cols + j) = B.at(i, j);
    }
    return C;
}

// The composition length of the COLUMN module of A.
template<class Elt>
uint32_t mc_col_length(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    const DenseBuf<Elt> T = mc_transpose(A);
    return engine::elim::length<Elt>(ctx, cview(T));
}

// The relation matrix of (+)_i Z/ell^{e_i}: diag(ell^{e_i}) (ell^k == 0 mod ell^k).
template<class Elt>
DenseBuf<Elt> mc_rel(const ModCtx<Elt>& ctx, const std::vector<uint32_t>& e) {
    const uint32_t t = static_cast<uint32_t>(e.size());
    DenseBuf<Elt> D(t, t);
    for (uint32_t i = 0; i < t; ++i) {
        ENGINE_CHECK_MSG(e[i] >= 1 && e[i] <= ctx.k, "ModuleCone: an exponent outside [1, k]");
        D.at(i, i) = (e[i] >= ctx.k) ? Elt(0) : ctx.ell_pow(e[i]);
    }
    return D;
}

template<class Elt>
Elt mc_mod_pow(const ModCtx<Elt>& ctx, Elt v, uint32_t e) {
    return (e >= ctx.k) ? v : static_cast<Elt>(v % ctx.ell_pow(e));
}

template<class Elt>
uint32_t mc_sum(const std::vector<uint32_t>& e) {
    uint32_t s = 0;
    for (uint32_t x : e) s += x;
    return s;
}

// length(im F) for F : A -> M in (c5) form (t_M x t_A), M = (+) Z/ell^{e_i}.
template<class Elt>
uint32_t mc_im_length(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& F, const std::vector<uint32_t>& eM) {
    if (eM.empty()) return 0;
    ENGINE_CHECK_MSG(F.rows == eM.size(), "ModuleCone: map height != #summands of its target");
    const DenseBuf<Elt> D = mc_rel(ctx, eM);
    return mc_col_length(ctx, mc_hcat(F, D)) - mc_col_length(ctx, D);
}

// Every row i of the column y is 0 in Z/ell^{e_i}.
template<class Elt>
bool mc_is_zero_class(const ModCtx<Elt>& ctx, const std::vector<Elt>& y, const std::vector<uint32_t>& e) {
    if (y.size() != e.size()) return false;
    for (size_t i = 0; i < y.size(); ++i)
        if (mc_mod_pow(ctx, y[i], e[i]) != 0) return false;
    return true;
}

template<class Elt>
std::vector<Elt> mc_apply(const ModCtx<Elt>& ctx, const DenseBuf<Elt>& F, const std::vector<Elt>& x) {
    ENGINE_CHECK_MSG(F.cols == x.size(), "ModuleCone: apply shape mismatch");
    std::vector<Elt> y(F.rows, 0);
    for (uint32_t i = 0; i < F.rows; ++i) {
        Elt acc = 0;
        for (uint32_t j = 0; j < F.cols; ++j)
            if (x[j] != 0 && F.at(i, j) != 0) acc = ctx.add(acc, ctx.mul(F.at(i, j), x[j]));
        y[i] = acc;
    }
    return y;
}

template<class Elt>
std::string mc_str(Elt v) { return std::to_string(static_cast<uint64_t>(v)); }

} // namespace

// ===========================================================================
// RestrictionFlatRing
// ===========================================================================
template<class Elt>
engine::DenseBuf<Elt> RestrictionFlatRing(const SimplicialComplex& K, const SimplicialComplex& K_bd, uint32_t q,
                                          uint32_t d_V, const engine::ModCtx<Elt>& ctx) {
    (void)ctx;
    const uint32_t NK = K.get_num_q_simplices(q), Nb = K_bd.get_num_q_simplices(q);
    DenseBuf<Elt> R(Nb * d_V, NK * d_V);
    const auto& bs = K_bd.get_simplices();
    for (uint32_t s = 0; s < Nb; ++s) {
        const int64_t sK = K.find_simplex_index(q, bs.at(q).at(s));
        if (sK < 0) throw std::invalid_argument("RestrictionFlatRing: a simplex of K_bd is not a simplex of K");
        for (uint32_t a = 0; a < d_V; ++a) R.at(s * d_V + a, static_cast<uint32_t>(sK) * d_V + a) = 1;
    }
    return R;
}

// ===========================================================================
// ConeModuleCertificate::verify -- re-derives every item from the stored data.
// ===========================================================================
template<class Elt>
uint32_t ConeModuleCertificate<Elt>::verify() const {
    std::vector<std::string> f;
    return verify(f);
}

template<class Elt>
uint32_t ConeModuleCertificate<Elt>::verify(std::vector<std::string>& failures) const {
    const size_t before = failures.size();
    const uint32_t nt = static_cast<uint32_t>(term_exps.size());
    if (term_names.size() != nt || maps.size() + 1 != nt || map_names.size() != maps.size() ||
        im_lengths.size() != maps.size()) {
        failures.push_back("C11: the sequence is misshapen (terms / maps / names / image lengths)");
        return static_cast<uint32_t>(failures.size() - before);
    }
    // C1: X_{q+1} X_q = 0.
    for (size_t q = 0; q + 1 < X.size(); ++q) {
        if (X[q + 1].cols != X[q].rows) { failures.push_back("C1: X_" + std::to_string(q + 1) + " X_" + std::to_string(q) + " shape"); continue; }
        const DenseBuf<Elt> P = mc_mul(ctx, X[q + 1], X[q]);
        for (uint32_t i = 0; i < P.rows; ++i)
            for (uint32_t j = 0; j < P.cols; ++j)
                if (P.at(i, j) != 0) { failures.push_back("C1: D_c^2 != 0 at q = " + std::to_string(q)); i = P.rows; break; }
    }
    // C11 (length identity at every term) and C14 (composites zero).
    for (uint32_t i = 0; i < nt; ++i) {
        const std::vector<uint32_t>& eM = term_exps[i];
        if (i + 1 < nt && maps[i].rows != term_exps[i + 1].size()) failures.push_back("C11: map " + map_names[i] + " height");
        if (i + 1 < nt && maps[i].cols != eM.size()) failures.push_back("C11: map " + map_names[i] + " width");
        const uint32_t in_len = (i > 0) ? mc_im_length(ctx, maps[i - 1], eM) : 0u;
        const uint32_t out_len = (i + 1 < nt) ? mc_im_length(ctx, maps[i], term_exps[i + 1]) : 0u;
        if (i > 0 && in_len != im_lengths[i - 1]) failures.push_back("C11: the stored image length of " + map_names[i - 1] + " is not re-derived");
        if (mc_sum<Elt>(eM) != in_len + out_len)
            failures.push_back("C11: length(" + term_names[i] + ") = " + std::to_string(mc_sum<Elt>(eM)) + " != " +
                               std::to_string(in_len) + " + " + std::to_string(out_len) + " (the length identity)");
        if (i > 0 && i + 1 < nt) {
            const DenseBuf<Elt> GF = mc_mul(ctx, maps[i], maps[i - 1]);
            for (uint32_t c = 0; c < GF.cols; ++c) {
                std::vector<Elt> y(GF.rows);
                for (uint32_t r = 0; r < GF.rows; ++r) y[r] = GF.at(r, c);
                if (!mc_is_zero_class(ctx, y, term_exps[i + 1])) {
                    failures.push_back("C14: " + map_names[i] + " o " + map_names[i - 1] + " != 0 at " + term_names[i]);
                    break;
                }
            }
        }
    }
    // C14: every stored witness -- x in ker(out) and x = in w1 + D_M w2.
    for (const Witness& w : witnesses) {
        if (w.term >= nt) { failures.push_back("C14: a witness names no term"); continue; }
        const std::vector<uint32_t>& eM = term_exps[w.term];
        if (w.x.size() != eM.size() || w.w2.size() != eM.size()) { failures.push_back("C14: witness shape at " + term_names[w.term]); continue; }
        if (w.term + 1 < nt) {
            const std::vector<Elt> gx = mc_apply(ctx, maps[w.term], w.x);
            if (!mc_is_zero_class(ctx, gx, term_exps[w.term + 1]))
                failures.push_back("C14: a stored kernel generator at " + term_names[w.term] + " is not in ker " + map_names[w.term]);
        }
        std::vector<Elt> lhs(eM.size(), 0);
        if (w.term > 0) {
            if (w.w1.size() != maps[w.term - 1].cols) { failures.push_back("C14: witness w1 shape at " + term_names[w.term]); continue; }
            lhs = mc_apply(ctx, maps[w.term - 1], w.w1);
        } else if (!w.w1.empty()) {
            failures.push_back("C14: witness w1 at the first term");
            continue;
        }
        for (size_t r = 0; r < eM.size(); ++r) {
            const Elt d = (eM[r] >= ctx.k) ? Elt(0) : ctx.ell_pow(eM[r]);
            lhs[r] = ctx.add(lhs[r], ctx.mul(d, w.w2[r]));
        }
        if (lhs != w.x) failures.push_back("C14: the residual in(w1) + D_M w2 - x is not zero at " + term_names[w.term]);
    }
    // C10, C12, C11 (eq:conedims in lengths) by term name.
    auto len_of = [&](const std::string& name, bool& found) -> uint32_t {
        for (uint32_t i = 0; i < nt; ++i)
            if (term_names[i] == name) { found = true; return mc_sum<Elt>(term_exps[i]); }
        found = false;
        return 0;
    };
    auto im_of = [&](const std::string& name, bool& found) -> uint32_t {
        for (size_t i = 0; i < map_names.size(); ++i)
            if (map_names[i] == name) { found = true; return im_lengths[i]; }
        found = false;
        return 0;
    };
    bool f0 = false, f1 = false, f2 = false, f3 = false, f4 = false, f5 = false, f6 = false;
    const uint32_t Hc0 = len_of("Hc^0", f0);
    if (f0 && Hc0 != 0) failures.push_back("C10: H^0_c != 0 (length " + std::to_string(Hc0) + ")");
    const uint32_t H2 = len_of("H^2", f1);
    if (f1 && six_invertible && H2 != 0) failures.push_back("C12: H^2 != 0 at 6 in R^x (length " + std::to_string(H2) + ")");
    const uint32_t Hc1 = len_of("Hc^1", f2), Hb0 = len_of("Hb^0", f3), H0 = len_of("H^0", f4), H1 = len_of("H^1", f5);
    const uint32_t im_r1 = im_of("r^1", f6);
    if (f2 && f3 && f4 && f5 && f6) {
        if (H1 < im_r1 || Hc1 + H0 != Hb0 + (H1 - im_r1))
            failures.push_back("C11: l(H^1_c) != l(H^0(bd)) - l(H^0) + l(ker r^1) (eq:conedims, first)");
        bool g1 = false, g2 = false;
        const uint32_t Hc2 = len_of("Hc^2", g1), Hb1 = len_of("Hb^1", g2);
        if (g1 && g2 && f1 && H2 == 0 && Hc2 + im_r1 != Hb1)
            failures.push_back("C11: l(H^2_c) != l(H^1(bd)) - l(im r^1) (eq:conedims, second, H^2 = 0)");
    }
    return static_cast<uint32_t>(failures.size() - before);
}

template<class Elt>
std::vector<uint32_t> ConeModuleCertificate<Elt>::failing_items() const {
    std::vector<std::string> f;
    verify(f);
    std::vector<uint32_t> out;
    for (const std::string& s : f) {
        if (s.size() < 2 || s[0] != 'C') continue;
        const uint32_t item = static_cast<uint32_t>(std::stoul(s.substr(1)));
        if (std::find(out.begin(), out.end(), item) == out.end()) out.push_back(item);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ===========================================================================
// ModuleConeT
// ===========================================================================
template<class Elt>
ModuleConeT<Elt>::ModuleConeT(const EquivariantCohomologyOnKT<Elt>* bulk, const EquivariantCohomologyOnKT<Elt>* boundary,
                              std::vector<Buf> r_flat)
    : bulk_(bulk), bd_(boundary), r_flat_(std::move(r_flat)) {
    if (bulk_ == nullptr || bd_ == nullptr) throw std::invalid_argument("ModuleCone: null input");
    if (!bulk_->is_composite() || !bd_->is_composite())
        throw std::logic_error("ModuleCone: the bulk and the boundary must both be over Z/ell^k (the dense module route)");
    ctx_ = bulk_->ring_context();
    if (bd_->ring_context().p != ctx_.p || bd_->ring_context().k != ctx_.k)
        throw std::logic_error("ModuleCone: the bulk (Z/" + mc_str<Elt>(ctx_.p) + ") and the boundary (Z/" +
                               mc_str<Elt>(bd_->ring_context().p) + ") are over different rings");
    // [E] sec:ring (f): the cone is honoured at ell >= 5 only (the default
    // SmallEllPolicy); at ell in {2, 3} the inputs are the Bredon objects.
    if (ctx_.ell == 2 || ctx_.ell == 3 || bulk_->is_flat_route() || bd_->is_flat_route())
        hecke::RefuseCoefficientRing("EquivariantMappingCone (Z/ell^k)", ctx_.k,
                                     " at ell = " + mc_str<Elt>(ctx_.ell) +
                                         " (the cone over Z/ell^k is honoured at ell >= 5 only; [E] sec:ring (f))");
    if (!bulk_->is_cohomology_computed() || !bd_->is_cohomology_computed())
        throw std::runtime_error("ModuleCone: an input complex is not computed");
    d_V_ = bulk_->get_representation_dim();
    if (d_V_ == 0 || bd_->get_representation_dim() != d_V_) throw std::invalid_argument("ModuleCone: fibre dimension mismatch");
    top_K_ = bulk_->num_degrees() - 1;
    top_bd_ = bd_->num_degrees() - 1;
    q_max_ = std::max(top_K_, top_bd_ + 1);
    const SimplicialComplex* K = bulk_->get_lssc()->get_base_K_complex();
    const SimplicialComplex* Kb = bd_->get_lssc()->get_base_K_complex();
    const uint32_t nr = std::min(top_K_, top_bd_) + 1;
    if (r_flat_.size() < nr) throw std::invalid_argument("ModuleCone: r_flat is missing a degree");
    for (uint32_t q = 0; q < nr; ++q)
        if (r_flat_[q].rows != Kb->get_num_q_simplices(q) * d_V_ || r_flat_[q].cols != K->get_num_q_simplices(q) * d_V_)
            throw std::invalid_argument("ModuleCone: r_flat[" + std::to_string(q) + "] is not N_q(K_bd) d_V x N_q(K) d_V");
}

template<class Elt>
void ModuleConeT<Elt>::require_computed(const char* who) const {
    if (!computed_) throw std::runtime_error(std::string("ModuleCone::") + who + ": compute() has not run");
}

template<class Elt>
uint32_t ModuleConeT<Elt>::bulk_part_dim(uint32_t q) const {
    return (q <= top_K_) ? bulk_->get_lssc()->get_base_K_complex()->get_num_q_simplices(q) * d_V_ : 0u;
}

template<class Elt>
uint32_t ModuleConeT<Elt>::boundary_part_dim(uint32_t q) const {
    if (q == 0 || q - 1 > top_bd_) return 0;
    return bd_->get_lssc()->get_base_K_complex()->get_num_q_simplices(q - 1) * d_V_;
}

template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeT<Elt>::r_flat(uint32_t q) const {
    if (q > std::min(top_K_, top_bd_)) throw std::out_of_range("ModuleCone::r_flat: degree out of range");
    return r_flat_[q];
}

// D_c^q (c, c_bd) = (-d c, d_bd c_bd + r c), dense over Z/ell^k.
template<class Elt>
engine::DenseBuf<Elt> ModuleConeT<Elt>::D_flat(uint32_t q) const {
    const SimplicialComplex* K = bulk_->get_lssc()->get_base_K_complex();
    const SimplicialComplex* Kb = bd_->get_lssc()->get_base_K_complex();
    const uint32_t d = d_V_;
    Buf D(flat_dim(q + 1), flat_dim(q));
    const uint32_t NKq = bulk_part_dim(q) / d, NKq1 = bulk_part_dim(q + 1) / d;
    const uint32_t Nbqm1 = boundary_part_dim(q) / d, Nbq = boundary_part_dim(q + 1) / d;
    if (NKq > 0 && NKq1 > 0) {
        const FlatMatrixT<Elt> D1 = K->template coboundary_operator_elt<Elt>(q, ctx_.p);
        ENGINE_CHECK_MSG(D1.get_rows() == NKq1 && D1.get_cols() == NKq, "ModuleCone: bulk coboundary shape");
        for (uint32_t i = 0; i < NKq1; ++i)
            for (uint32_t j = 0; j < NKq; ++j) {
                const Elt s = D1.get_entry(i, j);
                if (s == 0) continue;
                for (uint32_t a = 0; a < d; ++a) D.at(i * d + a, j * d + a) = ctx_.neg(s);
            }
    }
    if (Nbqm1 > 0 && Nbq > 0) {
        const FlatMatrixT<Elt> Db = Kb->template coboundary_operator_elt<Elt>(q - 1, ctx_.p);
        ENGINE_CHECK_MSG(Db.get_rows() == Nbq && Db.get_cols() == Nbqm1, "ModuleCone: boundary coboundary shape");
        for (uint32_t i = 0; i < Nbq; ++i)
            for (uint32_t j = 0; j < Nbqm1; ++j) {
                const Elt s = Db.get_entry(i, j);
                if (s == 0) continue;
                for (uint32_t a = 0; a < d; ++a) D.at((NKq1 + i) * d + a, (NKq + j) * d + a) = s;
            }
    }
    if (NKq > 0 && Nbq > 0 && q <= top_K_ && q <= top_bd_) {
        const Buf& R = r_flat_[q];
        for (uint32_t i = 0; i < R.rows; ++i)
            for (uint32_t j = 0; j < R.cols; ++j) D.at(NKq1 * d + i, j) = R.at(i, j);
    }
    return D;
}

template<class Elt>
void ModuleConeT<Elt>::compute() {
    if (computed_) return;
    const ModCtx<Elt>& ctx = ctx_;
    deg_.assign(q_max_ + 2, Degree());
    // C_q: the concatenated Howell bases, asserted to be the Howell form of the block sum.
    for (uint32_t q = 0; q <= q_max_ + 1; ++q) {
        const Buf none_K(bulk_part_dim(q), 0), none_b(boundary_part_dim(q), 0);
        const Buf& CK = (q <= top_K_) ? bulk_->module_degree(q).C : none_K;
        const Buf& Cb = (q >= 1 && q - 1 <= top_bd_) ? bd_->module_degree(q - 1).C : none_b;
        ENGINE_CHECK_MSG(CK.rows == bulk_part_dim(q) && Cb.rows == boundary_part_dim(q), "ModuleCone: input C^q_eq height");
        Degree& G = deg_[q];
        G.cK = CK.cols;
        G.cb = Cb.cols;
        G.C = Buf(flat_dim(q), CK.cols + Cb.cols);
        for (uint32_t i = 0; i < CK.rows; ++i)
            for (uint32_t j = 0; j < CK.cols; ++j) G.C.at(i, j) = CK.at(i, j);
        for (uint32_t i = 0; i < Cb.rows; ++i)
            for (uint32_t j = 0; j < Cb.cols; ++j) G.C.at(CK.rows + i, CK.cols + j) = Cb.at(i, j);
        if (G.C.cols > 0) {
            const Buf Ct = mc_transpose(G.C);
            const engine::elim::HowellResult<Elt> h = engine::elim::howell<Elt>(ctx, cview(Ct));
            bool same = h.rows() == Ct.rows;
            for (uint32_t r = 0; same && r < Ct.rows; ++r) {
                if (h.pivot_v[r] != 0) same = false;
                for (uint32_t c = 0; same && c < Ct.cols; ++c) same = (h.H.at(r, c) == Ct.at(r, c));
            }
            if (!same)
                throw std::runtime_error("ModuleCone: the concatenated Howell bases of C^q_eq(K) and C^{q-1}_eq(K_bd) are not "
                                         "the Howell form of Cc^" + std::to_string(q) + "_eq (Sec. 13.5 uniqueness; [E] (f))");
        }
    }
    // X_q: D_c in C coordinates (the cone's (c1)).
    for (uint32_t q = 0; q <= q_max_; ++q) {
        const Buf Dq = D_flat(q);
        const Buf& C0 = deg_[q].C;
        const Buf& C1 = deg_[q + 1].C;
        const Buf DC = mc_mul(ctx, Dq, C0);
        Buf& X = deg_[q].X;
        X = Buf(C1.cols, C0.cols);
        if (C0.cols > 0 && C1.cols > 0) {
            std::vector<std::vector<Elt>> co;
            try {
                co = engine::elim::solve_multi_rhs_module<Elt>(ctx, cview(C1), cview(DC));
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("ModuleCone: D_c Cc^" + std::to_string(q) + "_eq is not inside Cc^" + std::to_string(q + 1) +
                                         "_eq (" + e.what() + ")");
            }
            for (uint32_t t = 0; t < C0.cols; ++t)
                for (uint32_t j = 0; j < C1.cols; ++j) X.at(j, t) = co[t][j];
        } else {
            for (uint32_t i = 0; i < DC.rows; ++i)
                for (uint32_t t = 0; t < DC.cols; ++t)
                    if (DC.at(i, t) != 0) throw std::runtime_error("ModuleCone: D_c Cc^q_eq is not inside Cc^{q+1}_eq = 0");
        }
    }
    // Z, H with the cyclic-summand basis ([E] (c2)-(c4)).
    for (uint32_t q = 0; q <= q_max_; ++q) {
        Degree& G = deg_[q];
        const uint32_t cq = G.C.cols;
        const engine::elim::HowellResult<Elt> Zh = engine::elim::kernel_module<Elt>(ctx, cview(G.X));
        G.Zc = Buf(cq, Zh.rows());
        for (uint32_t c = 0; c < Zh.rows(); ++c)
            for (uint32_t i = 0; i < cq; ++i) G.Zc.at(i, c) = Zh.H.at(c, i);
        const Buf none(cq, 0);
        const Buf& Bm = (q > 0) ? deg_[q - 1].X : none;
        G.dec = engine::elim::cyclic_decomposition<Elt>(ctx, cview(G.Zc), cview(Bm));
        G.length_H = mc_sum<Elt>(G.dec.exponents);
        G.H = mc_mul(ctx, G.C, G.dec.basis);
    }
    computed_ = true;   // the accessors below are needed by the maps; reset on failure
    try {
        // The three map families of eq:coneles, (c5) matrices.
        pi_.clear(); r_.clear(); iota_.clear();
        for (uint32_t q = 0; q <= std::min(q_max_, top_K_); ++q) {
            const Buf& Hc = deg_[q].H;
            const uint32_t tK = static_cast<uint32_t>(bulk_->H_invariant_factors(q).size());
            Buf M(tK, Hc.cols);
            if (Hc.cols > 0) {
                Buf Y(bulk_part_dim(q), Hc.cols);
                for (uint32_t i = 0; i < Y.rows; ++i)
                    for (uint32_t j = 0; j < Hc.cols; ++j) Y.at(i, j) = Hc.at(i, j);
                const auto co = bulk_->H_class_coordinates(q, cview(Y));
                for (uint32_t j = 0; j < Hc.cols; ++j)
                    for (uint32_t i = 0; i < tK; ++i) M.at(i, j) = co[j][i];
            }
            pi_.push_back(std::move(M));
        }
        for (uint32_t q = 0; q <= top_K_; ++q) {
            const uint32_t tK = static_cast<uint32_t>(bulk_->H_invariant_factors(q).size());
            if (q > top_bd_) { r_.push_back(Buf(0, tK)); continue; }
            const uint32_t tb = static_cast<uint32_t>(bd_->H_invariant_factors(q).size());
            Buf M(tb, tK);
            if (tK > 0) {
                const Buf Y = mc_mul(ctx, r_flat_[q], bulk_->H_module_basis(q));
                const auto co = bd_->H_class_coordinates(q, cview(Y));
                for (uint32_t j = 0; j < tK; ++j)
                    for (uint32_t i = 0; i < tb; ++i) M.at(i, j) = co[j][i];
            }
            r_.push_back(std::move(M));
        }
        for (uint32_t q = 0; q <= top_bd_ && q + 1 <= q_max_; ++q) {
            const Buf& Hb = bd_->H_module_basis(q);
            const uint32_t tc = deg_[q + 1].dec.rows();
            Buf M(tc, Hb.cols);
            if (Hb.cols > 0) {
                Buf Y(flat_dim(q + 1), Hb.cols);
                const uint32_t off = bulk_part_dim(q + 1);
                ENGINE_CHECK_MSG(off + Hb.rows == Y.rows, "ModuleCone: iota target layout");
                for (uint32_t i = 0; i < Hb.rows; ++i)
                    for (uint32_t j = 0; j < Hb.cols; ++j) Y.at(off + i, j) = Hb.at(i, j);
                const auto co = H_class_coordinates(q + 1, cview(Y));
                for (uint32_t j = 0; j < Hb.cols; ++j)
                    for (uint32_t i = 0; i < tc; ++i) M.at(i, j) = co[j][i];
            }
            iota_.push_back(std::move(M));
        }

        // The sequence Hc^0 -> H^0 -> Hb^0 -> Hc^1 -> ... -> H^{q_max} -> Hb^{q_max}.
        ConeModuleCertificate<Elt> cert;
        cert.ctx = ctx;
        cert.six_invertible = (ctx.ell >= 5);
        for (uint32_t q = 0; q <= q_max_; ++q) cert.X.push_back(deg_[q].X);
        const std::vector<uint32_t> empty;
        for (uint32_t q = 0; q <= q_max_; ++q) {
            cert.term_names.push_back("Hc^" + std::to_string(q));
            cert.term_exps.push_back(deg_[q].dec.exponents);
            cert.term_names.push_back("H^" + std::to_string(q));
            cert.term_exps.push_back(q <= top_K_ ? bulk_->H_invariant_factors(q) : empty);
            cert.term_names.push_back("Hb^" + std::to_string(q));
            cert.term_exps.push_back(q <= top_bd_ ? bd_->H_invariant_factors(q) : empty);
        }
        const uint32_t nt = static_cast<uint32_t>(cert.term_exps.size());
        for (uint32_t i = 0; i + 1 < nt; ++i) {
            const uint32_t q = i / 3, which = i % 3;
            const uint32_t t_src = static_cast<uint32_t>(cert.term_exps[i].size());
            const uint32_t t_tgt = static_cast<uint32_t>(cert.term_exps[i + 1].size());
            Buf M;
            std::string name;
            if (which == 0) { name = "pi^" + std::to_string(q); M = (q < pi_.size()) ? pi_[q] : Buf(t_tgt, t_src); }
            else if (which == 1) { name = "r^" + std::to_string(q); M = (q < r_.size()) ? r_[q] : Buf(t_tgt, t_src); }
            else { name = "iota^" + std::to_string(q); M = (q < iota_.size()) ? iota_[q] : Buf(t_tgt, t_src); }
            ENGINE_CHECK_MSG(M.rows == t_tgt && M.cols == t_src, "ModuleCone: a map of eq:coneles is misshapen");
            // Reduce every entry into [0, ell^{e_i}) of its target row (c5).
            for (uint32_t r = 0; r < M.rows; ++r)
                for (uint32_t c = 0; c < M.cols; ++c) M.at(r, c) = mc_mod_pow(ctx, M.at(r, c), cert.term_exps[i + 1][r]);
            cert.map_names.push_back(name);
            cert.im_lengths.push_back(mc_im_length(ctx, M, cert.term_exps[i + 1]));
            cert.maps.push_back(std::move(M));
        }

        // Item 14: (iv)'s non-field half at every spot.
        for (uint32_t i = 0; i < nt; ++i) {
            const std::vector<uint32_t>& eM = cert.term_exps[i];
            const uint32_t tM = static_cast<uint32_t>(eM.size());
            const Buf DM = mc_rel(ctx, eM);
            // (a) the composite, by solves: every column of out o in lies in rel(target).
            if (i > 0 && i + 1 < nt && cert.term_exps[i + 1].size() > 0) {
                const Buf GF = mc_mul(ctx, cert.maps[i], cert.maps[i - 1]);
                if (GF.cols > 0) {
                    const Buf DN = mc_rel(ctx, cert.term_exps[i + 1]);
                    try {
                        (void)engine::elim::solve_multi_rhs_module<Elt>(ctx, cview(DN), cview(GF));
                    } catch (const std::runtime_error& e) {
                        throw std::runtime_error("ModuleCone: obl:coneexact(iv) (non-field half) FAILED -- " + cert.map_names[i] +
                                                 " o " + cert.map_names[i - 1] + " != 0 at " + cert.term_names[i] + " (" +
                                                 e.what() + ")");
                    }
                }
            }
            if (tM == 0) continue;
            // (b) the Howell generators of ker(out), each exhibited as an image of in.
            std::vector<std::vector<Elt>> gens;
            if (i + 1 < nt && !cert.term_exps[i + 1].empty()) {
                const Buf S = mc_hcat(cert.maps[i], mc_rel(ctx, cert.term_exps[i + 1]));
                const engine::elim::HowellResult<Elt> Kh = engine::elim::kernel_module<Elt>(ctx, cview(S));
                for (uint32_t g = 0; g < Kh.rows(); ++g) {
                    std::vector<Elt> x(tM);
                    for (uint32_t c = 0; c < tM; ++c) x[c] = Kh.H.at(g, c);
                    gens.push_back(std::move(x));
                }
            } else {
                for (uint32_t c = 0; c < tM; ++c) {
                    std::vector<Elt> x(tM, 0);
                    x[c] = 1;
                    gens.push_back(std::move(x));
                }
            }
            const uint32_t tA = (i > 0) ? cert.maps[i - 1].cols : 0u;
            const Buf B = (i > 0) ? mc_hcat(cert.maps[i - 1], DM) : DM;
            for (std::vector<Elt>& x : gens) {
                for (uint32_t r = 0; r < tM; ++r) x[r] = mc_mod_pow(ctx, x[r], eM[r]);
                bool zero = true;
                for (Elt v : x) zero = zero && (v == 0);
                if (zero) continue;
                Buf Y(tM, 1);
                for (uint32_t r = 0; r < tM; ++r) Y.at(r, 0) = x[r];
                std::vector<std::vector<Elt>> co;
                try {
                    co = engine::elim::solve_multi_rhs_module<Elt>(ctx, cview(B), cview(Y));
                } catch (const std::runtime_error& e) {
                    throw std::runtime_error("ModuleCone: obl:coneexact(iv) (non-field half) FAILED -- a generator of ker " +
                                             (i + 1 < nt ? cert.map_names[i] : std::string("(the zero map)")) + " at " +
                                             cert.term_names[i] + " is not an image of " +
                                             (i > 0 ? cert.map_names[i - 1] : std::string("(the zero map)")) + " (" + e.what() + ")");
                }
                typename ConeModuleCertificate<Elt>::Witness w;
                w.term = i;
                w.x = x;
                w.w1.assign(co[0].begin(), co[0].begin() + tA);
                w.w2.assign(co[0].begin() + tA, co[0].end());
                cert.witnesses.push_back(std::move(w));
            }
        }
        std::vector<std::string> failures;
        if (cert.verify(failures) != 0) {
            std::ostringstream oss;
            oss << "ModuleCone: the certificate fails (" << failures.size() << " item(s)): " << failures.front();
            throw std::runtime_error(oss.str());
        }
        cert_ = std::move(cert);
        iv_certified_ = true;
    } catch (...) {
        computed_ = false;
        throw;
    }
}

template<class Elt>
const typename ModuleConeT<Elt>::Degree& ModuleConeT<Elt>::degree(uint32_t q) const {
    require_computed("degree");
    if (q > q_max_) throw std::out_of_range("ModuleCone::degree: degree out of range");
    return deg_[q];
}

template<class Elt>
std::vector<uint32_t> ModuleConeT<Elt>::H_lengths() const {
    require_computed("H_lengths");
    std::vector<uint32_t> out;
    for (uint32_t q = 0; q <= q_max_; ++q) out.push_back(deg_[q].length_H);
    return out;
}

template<class Elt>
const std::vector<uint32_t>& ModuleConeT<Elt>::H_invariant_factors(uint32_t q) const { return degree(q).dec.exponents; }

template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeT<Elt>::H_module_basis(uint32_t q) const { return degree(q).H; }

template<class Elt>
std::vector<std::vector<Elt>> ModuleConeT<Elt>::H_class_coordinates(uint32_t q, engine::DenseView<const Elt> Y) const {
    const Degree& G = degree(q);
    const uint32_t ny = Y.cols();
    ENGINE_CHECK_MSG(Y.rows() == G.C.rows, "ModuleCone::H_class_coordinates: Y must be flat_dim(q) x ny");
    if (ny == 0) return {};
    Buf Yc(G.C.cols, ny);
    if (G.C.cols > 0) {
        const auto co = engine::elim::solve_multi_rhs_module<Elt>(ctx_, cview(G.C), Y);   // throws if not in Cc^q_eq
        for (uint32_t t = 0; t < ny; ++t)
            for (uint32_t j = 0; j < G.C.cols; ++j) Yc.at(j, t) = co[t][j];
    } else {
        for (uint32_t i = 0; i < Y.rows(); ++i)
            for (uint32_t t = 0; t < ny; ++t)
                if (Y.at(i, t) != 0) throw std::runtime_error("ModuleCone::H_class_coordinates: a column is not in Cc^q_eq (= 0)");
    }
    if (G.dec.Gamma.cols == 0) {
        for (uint32_t i = 0; i < Yc.rows; ++i)
            for (uint32_t t = 0; t < ny; ++t)
                if (Yc.at(i, t) != 0) throw std::runtime_error("ModuleCone::H_class_coordinates: a column is not a cocycle (Z^q_c = 0)");
        return std::vector<std::vector<Elt>>(ny);
    }
    return engine::elim::cyclic_coordinates<Elt>(ctx_, G.dec, cview(Yc));   // throws if not a cocycle
}

template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeT<Elt>::pi(uint32_t q) const {
    require_computed("pi");
    if (q >= pi_.size()) throw std::out_of_range("ModuleCone::pi: not built in this degree");
    return pi_[q];
}
template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeT<Elt>::r(uint32_t q) const {
    require_computed("r");
    if (q >= r_.size()) throw std::out_of_range("ModuleCone::r: not built in this degree");
    return r_[q];
}
template<class Elt>
const engine::DenseBuf<Elt>& ModuleConeT<Elt>::iota(uint32_t q) const {
    require_computed("iota");
    if (q >= iota_.size()) throw std::out_of_range("ModuleCone::iota: not built in this degree");
    return iota_[q];
}

template<class Elt>
std::vector<uint32_t> ModuleConeT<Elt>::les_im_lengths() const {
    require_computed("les_im_lengths");
    return cert_.im_lengths;
}

template<class Elt>
const ConeModuleCertificate<Elt>& ModuleConeT<Elt>::certificate() const {
    require_computed("certificate");
    return cert_;
}

template struct ConeModuleCertificate<uint32_t>;
template struct ConeModuleCertificate<uint64_t>;
template class ModuleConeT<uint32_t>;
template class ModuleConeT<uint64_t>;
template engine::DenseBuf<uint32_t> RestrictionFlatRing<uint32_t>(const SimplicialComplex&, const SimplicialComplex&, uint32_t,
                                                                  uint32_t, const engine::ModCtx<uint32_t>&);
template engine::DenseBuf<uint64_t> RestrictionFlatRing<uint64_t>(const SimplicialComplex&, const SimplicialComplex&, uint32_t,
                                                                  uint32_t, const engine::ModCtx<uint64_t>&);
