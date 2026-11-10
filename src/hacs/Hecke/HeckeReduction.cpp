// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/HeckeReduction.cpp -- hecke-extensions slot 8, WP-4 (E8 item 4a).
// See HeckeReduction.h and [E] v1.9.0-pre.1 sec:ring (c5), (d).

#include "HeckeReduction.h"

#include "CoefficientRing.h"
#include "HeckeSettings.h"

#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <stdexcept>
#include <string>

namespace hecke {

namespace {

using engine::DenseBuf;
using engine::ModCtx;

// A dim_q-long flat cochain of a FlatMatrix N_q x d_V (the L1 layout: entry
// (s, v) at s * d_V + v), written into column `col` of `dst`.
template<class Elt>
void put_flat_column(DenseBuf<Elt>& dst, uint32_t col, const FlatMatrixT<Elt>& f) {
    const uint32_t r = f.get_rows(), c = f.get_cols();
    ENGINE_CHECK_MSG(static_cast<uint64_t>(r) * c == dst.rows, "HeckeReduction: a D5b basis vector has the wrong length");
    for (uint32_t s = 0; s < r; ++s)
        for (uint32_t v = 0; v < c; ++v) dst.at(s * c + v, col) = f.get_entry(s, v);
}

// P (h x m, a FlatMatrix over F_ell) applied to column j of Y (m x ny, entries
// already in [0, ell)) -> out column j (h rows).
template<class Elt>
void apply_projector(const ModCtx<Elt>& F, const FlatMatrixT<Elt>& P, const DenseBuf<Elt>& Y, uint32_t j,
                     DenseBuf<Elt>& out, uint32_t out_col) {
    const uint32_t h = P.get_rows(), m = P.get_cols();
    ENGINE_CHECK_MSG(m == Y.rows && h == out.rows, "HeckeReduction: projector shape mismatch");
    for (uint32_t i = 0; i < h; ++i) {
        Elt acc = 0;
        for (uint32_t s = 0; s < m; ++s) {
            const Elt y = Y.at(s, j);
            if (y == 0) continue;
            const Elt pv = P.get_entry(i, s);
            if (pv == 0) continue;
            acc = F.add(acc, F.mul(pv, y));
        }
        out.at(i, out_col) = acc;
    }
}

template<class Elt>
Elt field_prime_of(const EquivariantCohomologyOnKT<Elt>& field, uint32_t q) {
    const auto& P = field.get_H_eq_projectors_from_C_eq_flat_fm();
    ENGINE_CHECK_MSG(q < P.size(), "HeckeReduction: degree out of range on the F_ell object");
    return P[q].get_p();
}

} // namespace

template<class Elt>
engine::DenseBuf<Elt> ReduceHToField(const EquivariantCohomologyOnKT<Elt>& ring,
                                     const EquivariantCohomologyOnKT<Elt>& field, uint32_t q) {
    ENGINE_CHECK_MSG(ring.is_composite() && !field.is_composite(),
                     "ReduceHToField: needs a Z/ell^k object and an F_ell object");
    const ModCtx<Elt>& ctx = ring.ring_context();
    const Elt ell = ctx.ell;
    ENGINE_CHECK_MSG(field_prime_of(field, q) == ell, "ReduceHToField: the F_ell object is at another prime");
    const ModCtx<Elt> F(ell);
    const DenseBuf<Elt>& Hb = ring.H_module_basis(q);   // dim_q x t
    const FlatMatrixT<Elt>& P = field.get_H_eq_projectors_from_C_eq_flat_fm()[q];   // h x dim_q
    if (P.get_rows() == 0) return DenseBuf<Elt>(0, Hb.cols);   // H^q(F_ell) = 0 (P may be stored 0 x 0)
    ENGINE_CHECK_MSG(P.get_cols() == Hb.rows, "ReduceHToField: the two objects' cochain spaces differ");
    DenseBuf<Elt> Y(Hb.rows, Hb.cols);
    for (uint32_t s = 0; s < Hb.rows; ++s)
        for (uint32_t j = 0; j < Hb.cols; ++j) Y.at(s, j) = static_cast<Elt>(Hb.at(s, j) % ell);
    DenseBuf<Elt> R(P.get_rows(), Hb.cols);
    for (uint32_t j = 0; j < Hb.cols; ++j) apply_projector(F, P, Y, j, R, j);
    return R;
}

template<class Elt>
engine::DenseBuf<Elt> ReduceHBetweenLevels(const EquivariantCohomologyOnKT<Elt>& hi,
                                           const EquivariantCohomologyOnKT<Elt>& lo, uint32_t q) {
    ENGINE_CHECK_MSG(hi.is_composite() && lo.is_composite(), "ReduceHBetweenLevels: needs two Z/ell^k objects");
    const ModCtx<Elt>& ch = hi.ring_context();
    const ModCtx<Elt>& cl = lo.ring_context();
    ENGINE_CHECK_MSG(ch.ell == cl.ell && ch.k > cl.k, "ReduceHBetweenLevels: needs Z/ell^{k_hi} -> Z/ell^{k_lo}, k_hi > k_lo");
    const DenseBuf<Elt>& Hb = hi.H_module_basis(q);
    const uint32_t t_hi = Hb.cols;
    const uint32_t t_lo = static_cast<uint32_t>(lo.H_invariant_factors(q).size());
    DenseBuf<Elt> R(t_lo, t_hi);
    if (t_hi == 0 || t_lo == 0) return R;
    DenseBuf<Elt> Y(Hb.rows, t_hi);
    for (uint32_t s = 0; s < Hb.rows; ++s)
        for (uint32_t j = 0; j < t_hi; ++j) Y.at(s, j) = static_cast<Elt>(Hb.at(s, j) % cl.p);
    const std::vector<std::vector<Elt>> co = lo.H_class_coordinates(q, engine::cview(Y));
    ENGINE_CHECK_MSG(co.size() == t_hi, "ReduceHBetweenLevels: coordinate count");
    for (uint32_t j = 0; j < t_hi; ++j) {
        ENGINE_CHECK_MSG(co[j].size() == t_lo, "ReduceHBetweenLevels: coordinate length");
        for (uint32_t i = 0; i < t_lo; ++i) R.at(i, j) = co[j][i];
    }
    return R;
}

template<class Elt>
engine::DenseBuf<Elt> BocksteinH(const EquivariantCohomologyOnKT<Elt>& ring2,
                                 const EquivariantCohomologyOnKT<Elt>& field, uint32_t q) {
    ENGINE_CHECK_MSG(ring2.is_composite() && ring2.coefficient_ring_k() == 2 && !field.is_composite(),
                     "BocksteinH: needs a Z/ell^2 object and an F_ell object");
    const ModCtx<Elt>& c2 = ring2.ring_context();
    const Elt ell = c2.ell;
    ENGINE_CHECK_MSG(q + 1 < ring2.num_degrees(), "BocksteinH: q + 1 is not a degree of the complex");
    ENGINE_CHECK_MSG(field_prime_of(field, q + 1) == ell, "BocksteinH: the F_ell object is at another prime");
    const ModCtx<Elt> F(ell);
    const auto& md = ring2.module_degree(q);
    const auto& mn = ring2.module_degree(q + 1);
    const DenseBuf<Elt>& C = md.C;    // dim_q x c_q
    const DenseBuf<Elt>& X = ring2.module_X(q);    // c_{q+1} x c_q (S23 G2-E: guarded against a lean object)
    const DenseBuf<Elt>& Cn = mn.C;   // dim_{q+1} x c_{q+1}
    ENGINE_CHECK_MSG(X.cols == C.cols && X.rows == Cn.cols, "BocksteinH: d^q is not C^q_eq -> C^{q+1}_eq");
    const std::vector<FlatMatrixT<Elt>>& Hf = field.get_H_eq_bases_fm()[q];
    const FlatMatrixT<Elt>& Pn = field.get_H_eq_projectors_from_C_eq_flat_fm()[q + 1];
    const uint32_t h = static_cast<uint32_t>(Hf.size());
    DenseBuf<Elt> B(Pn.get_rows(), h);
    if (h == 0 || Pn.get_rows() == 0) return B;   // H^q or H^{q+1}(F_ell) = 0 (P may be stored 0 x 0)
    ENGINE_CHECK_MSG(Pn.get_cols() == Cn.rows, "BocksteinH: the two objects' cochain spaces differ");

    // z = (C mod ell) c over F_ell, one column per D5b class.
    DenseBuf<Elt> Cm(C.rows, C.cols);
    for (uint32_t s = 0; s < C.rows; ++s)
        for (uint32_t j = 0; j < C.cols; ++j) Cm.at(s, j) = static_cast<Elt>(C.at(s, j) % ell);
    DenseBuf<Elt> Z(C.rows, h);
    for (uint32_t t = 0; t < h; ++t) put_flat_column(Z, t, Hf[t]);
    // Throws std::runtime_error if a class representative is not in C^q_eq(F_ell)
    // = colspan (C mod ell): the layout or the equivariant space would be wrong.
    const std::vector<std::vector<Elt>> co = engine::elim::solve_multi_rhs(F, engine::cview(Cm), engine::cview(Z));

    DenseBuf<Elt> W(Cn.rows, h);
    for (uint32_t t = 0; t < h; ++t) {
        // y = X c over Z/ell^2 with c lifted to [0, ell): the coordinates of d z~.
        std::vector<Elt> wc(X.rows, 0);
        for (uint32_t i = 0; i < X.rows; ++i) {
            Elt acc = 0;
            for (uint32_t j = 0; j < X.cols; ++j) {
                const Elt cj = co[t][j];
                if (cj == 0) continue;
                acc = c2.add(acc, c2.mul(X.at(i, j), cj));
            }
            if (acc % ell != 0)
                throw std::runtime_error("BocksteinH: d of the lift is not divisible by ell (degree " + std::to_string(q) +
                                         ", class " + std::to_string(t) + ")");
            wc[i] = static_cast<Elt>((acc / ell) % ell);
        }
        // w = (C^{q+1} mod ell) wc, a flat cocycle over F_ell.
        for (uint32_t s = 0; s < Cn.rows; ++s) {
            Elt acc = 0;
            for (uint32_t i = 0; i < Cn.cols; ++i) {
                if (wc[i] == 0) continue;
                acc = F.add(acc, F.mul(static_cast<Elt>(Cn.at(s, i) % ell), wc[i]));
            }
            W.at(s, t) = acc;
        }
    }
    for (uint32_t t = 0; t < h; ++t) apply_projector(F, Pn, W, t, B, t);
    return B;
}

template<class Elt>
uint32_t RankOverField(Elt ell, const engine::DenseBuf<Elt>& M) {
    if (M.rows == 0 || M.cols == 0) return 0;
    const ModCtx<Elt> F(ell);
    DenseBuf<Elt> c(M.rows, M.cols);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) c.at(i, j) = static_cast<Elt>(M.at(i, j) % ell);
    return engine::elim::rref(F, engine::view(c)).rank;
}

template<class Elt>
BocksteinBookkeeping BocksteinCounts(const EquivariantCohomologyOnKT<Elt>& ring2,
                                     const EquivariantCohomologyOnKT<Elt>& field) {
    BocksteinBookkeeping b;
    b.dim_H0 = field.get_H_eq_dimensions()[0];
    const DenseBuf<Elt> B0 = BocksteinH(ring2, field, 0);
    b.bockstein_rank0 = RankOverField(ring2.ring_context().ell, B0);
    ENGINE_CHECK_MSG(b.bockstein_rank0 <= b.dim_H0, "BocksteinCounts: rank exceeds dim H^0");
    b.higher_torsion_count = b.dim_H0 - b.bockstein_rank0;
    return b;
}

// ---- certificates -------------------------------------------------------------
namespace {
// out = L (a x b) * R (b x c) mod ctx.p, with element accessors.
template<class Elt, class LA, class RA>
DenseBuf<Elt> product(const ModCtx<Elt>& ctx, uint32_t a, uint32_t b, uint32_t c, LA l, RA r) {
    DenseBuf<Elt> out(a, c);
    for (uint32_t i = 0; i < a; ++i)
        for (uint32_t j = 0; j < c; ++j) {
            Elt acc = 0;
            for (uint32_t m = 0; m < b; ++m) acc = ctx.add(acc, ctx.mul(l(i, m), r(m, j)));
            out.at(i, j) = acc;
        }
    return out;
}
} // namespace

template<class Elt>
bool ReductionToFieldEquivariant(Elt ell, const MatrixT<Elt>& A_field, const engine::DenseBuf<Elt>& A_ring,
                                 const engine::DenseBuf<Elt>& R) {
    const uint32_t h = R.rows, t = R.cols;
    if (A_field.rows != h || A_field.cols != h || A_ring.rows != t || A_ring.cols != t) return false;
    const ModCtx<Elt> F(ell);
    auto r = [&](uint32_t i, uint32_t j) { return static_cast<Elt>(R.at(i, j) % ell); };
    const DenseBuf<Elt> L = product<Elt>(F, h, h, t, [&](uint32_t i, uint32_t j) { return A_field.data[i][j]; }, r);
    const DenseBuf<Elt> Rt = product<Elt>(F, h, t, t, r, [&](uint32_t i, uint32_t j) {
        return static_cast<Elt>(A_ring.at(i, j) % ell);
    });
    for (uint32_t i = 0; i < h; ++i)
        for (uint32_t j = 0; j < t; ++j)
            if (L.at(i, j) != Rt.at(i, j)) return false;
    return true;
}

template<class Elt>
bool ReductionBetweenLevelsEquivariant(const engine::ModCtx<Elt>& ctx_hi, const std::vector<uint32_t>& exps_lo,
                                       const engine::DenseBuf<Elt>& A_lo, const engine::DenseBuf<Elt>& A_hi,
                                       const engine::DenseBuf<Elt>& R) {
    const uint32_t tl = R.rows, th = R.cols;
    if (exps_lo.size() != tl || A_lo.rows != tl || A_lo.cols != tl || A_hi.rows != th || A_hi.cols != th) return false;
    auto a_lo = [&](uint32_t i, uint32_t j) { return A_lo.at(i, j); };
    auto a_hi = [&](uint32_t i, uint32_t j) { return A_hi.at(i, j); };
    auto r = [&](uint32_t i, uint32_t j) { return R.at(i, j); };
    const DenseBuf<Elt> L = product<Elt>(ctx_hi, tl, tl, th, a_lo, r);
    const DenseBuf<Elt> Rt = product<Elt>(ctx_hi, tl, th, th, r, a_hi);
    for (uint32_t i = 0; i < tl; ++i) {
        if (exps_lo[i] > ctx_hi.k) return false;
        const Elt m = ctx_hi.ell_pow(exps_lo[i]);
        for (uint32_t j = 0; j < th; ++j)
            if (L.at(i, j) % m != Rt.at(i, j) % m) return false;
    }
    return true;
}

template<class Elt>
bool BocksteinEquivariant(Elt ell, const MatrixT<Elt>& A_next, const MatrixT<Elt>& A_this,
                          const engine::DenseBuf<Elt>& B) {
    const uint32_t hn = B.rows, h = B.cols;
    if (A_next.rows != hn || A_next.cols != hn || A_this.rows != h || A_this.cols != h) return false;
    const ModCtx<Elt> F(ell);
    auto b = [&](uint32_t i, uint32_t j) { return static_cast<Elt>(B.at(i, j) % ell); };
    const DenseBuf<Elt> L = product<Elt>(F, hn, hn, h, [&](uint32_t i, uint32_t j) { return A_next.data[i][j]; }, b);
    const DenseBuf<Elt> Rt = product<Elt>(F, hn, h, h, b, [&](uint32_t i, uint32_t j) { return A_this.data[i][j]; });
    for (uint32_t i = 0; i < hn; ++i)
        for (uint32_t j = 0; j < h; ++j)
            if (L.at(i, j) != Rt.at(i, j)) return false;
    return true;
}

template<class Elt>
bool ComposesToZero(Elt ell, const engine::DenseBuf<Elt>& X, const engine::DenseBuf<Elt>& Y) {
    if (X.cols != Y.rows) return false;
    const ModCtx<Elt> F(ell);
    const DenseBuf<Elt> P = product<Elt>(
        F, X.rows, X.cols, Y.cols, [&](uint32_t i, uint32_t j) { return static_cast<Elt>(X.at(i, j) % ell); },
        [&](uint32_t i, uint32_t j) { return static_cast<Elt>(Y.at(i, j) % ell); });
    for (uint32_t i = 0; i < P.rows; ++i)
        for (uint32_t j = 0; j < P.cols; ++j)
            if (P.at(i, j) != 0) return false;
    return true;
}

#define HECKE_REDUCTION_INSTANTIATE(Elt)                                                                            \
    template engine::DenseBuf<Elt> ReduceHToField<Elt>(const EquivariantCohomologyOnKT<Elt>&,                       \
                                                       const EquivariantCohomologyOnKT<Elt>&, uint32_t);            \
    template engine::DenseBuf<Elt> ReduceHBetweenLevels<Elt>(const EquivariantCohomologyOnKT<Elt>&,                 \
                                                             const EquivariantCohomologyOnKT<Elt>&, uint32_t);      \
    template engine::DenseBuf<Elt> BocksteinH<Elt>(const EquivariantCohomologyOnKT<Elt>&,                           \
                                                   const EquivariantCohomologyOnKT<Elt>&, uint32_t);                \
    template uint32_t RankOverField<Elt>(Elt, const engine::DenseBuf<Elt>&);                                        \
    template BocksteinBookkeeping BocksteinCounts<Elt>(const EquivariantCohomologyOnKT<Elt>&,                       \
                                                       const EquivariantCohomologyOnKT<Elt>&);                      \
    template bool ReductionToFieldEquivariant<Elt>(Elt, const MatrixT<Elt>&, const engine::DenseBuf<Elt>&,          \
                                                   const engine::DenseBuf<Elt>&);                                   \
    template bool ReductionBetweenLevelsEquivariant<Elt>(const engine::ModCtx<Elt>&, const std::vector<uint32_t>&, \
                                                         const engine::DenseBuf<Elt>&, const engine::DenseBuf<Elt>&, \
                                                         const engine::DenseBuf<Elt>&);                             \
    template bool BocksteinEquivariant<Elt>(Elt, const MatrixT<Elt>&, const MatrixT<Elt>&,                          \
                                            const engine::DenseBuf<Elt>&);                                          \
    template bool ComposesToZero<Elt>(Elt, const engine::DenseBuf<Elt>&, const engine::DenseBuf<Elt>&);

HECKE_REDUCTION_INSTANTIATE(uint32_t)
HECKE_REDUCTION_INSTANTIATE(uint64_t)
#undef HECKE_REDUCTION_INSTANTIATE

// ===========================================================================
// (slot-9 WP-4; E8 item 4b) The cone -- the header note.
// ===========================================================================
namespace {
template<class Elt>
void refuse_bredon_cone(const char* who, const ModCtx<Elt>& c) {
    if (IsBredonPrime(static_cast<uint64_t>(c.ell)))
        RefuseCoefficientRing(who, c.k, " at ell = " + std::to_string(static_cast<uint64_t>(c.ell)) +
                                            " (the cone over Z/ell^k is honoured at ell >= 5 only; [E] sec:ring (f))");
}
} // namespace

engine::DenseBuf<uint32_t> ReduceConeHcToField(const ModuleConeT<uint32_t>& ring, const EquivariantMappingCone& field,
                                               uint32_t q) {
    const ModCtx<uint32_t>& ctx = ring.ring_context();
    refuse_bredon_cone("ReduceConeHcToField", ctx);
    ENGINE_CHECK_MSG(!field.is_composite() && field.ell() == ctx.ell, "ReduceConeHcToField: needs the F_ell cone at the same prime");
    const uint32_t ell = ctx.ell;
    const ModCtx<uint32_t> F(ell);
    const DenseBuf<uint32_t>& Hb = ring.H_module_basis(q);
    const DenseBuf<uint32_t> P = field.P_block(q).to_dense(engine::Layout::RowMajor);   // dim_Hc x flat_dim
    DenseBuf<uint32_t> R(field.dim_Hc(q), Hb.cols);
    if (R.rows == 0 || Hb.cols == 0) return R;
    ENGINE_CHECK_MSG(P.rows == R.rows && P.cols == Hb.rows, "ReduceConeHcToField: the two cones' cochain spaces differ");
    for (uint32_t i = 0; i < P.rows; ++i)
        for (uint32_t j = 0; j < Hb.cols; ++j) {
            uint32_t acc = 0;
            for (uint32_t s = 0; s < P.cols; ++s) {
                const uint32_t y = Hb.at(s, j) % ell;
                if (y != 0 && P.at(i, s) != 0) acc = F.add(acc, F.mul(P.at(i, s), y));
            }
            R.at(i, j) = acc;
        }
    return R;
}

template<class Elt>
engine::DenseBuf<Elt> ReduceConeHcBetweenLevels(const ModuleConeT<Elt>& hi, const ModuleConeT<Elt>& lo, uint32_t q) {
    const ModCtx<Elt>& ch = hi.ring_context();
    const ModCtx<Elt>& cl = lo.ring_context();
    refuse_bredon_cone("ReduceConeHcBetweenLevels", ch);
    ENGINE_CHECK_MSG(ch.ell == cl.ell && ch.k > cl.k, "ReduceConeHcBetweenLevels: needs Z/ell^{k_hi} -> Z/ell^{k_lo}, k_hi > k_lo");
    const DenseBuf<Elt>& Hb = hi.H_module_basis(q);
    const uint32_t t_hi = Hb.cols, t_lo = static_cast<uint32_t>(lo.H_invariant_factors(q).size());
    DenseBuf<Elt> R(t_lo, t_hi);
    if (t_hi == 0 || t_lo == 0) return R;
    DenseBuf<Elt> Y(Hb.rows, t_hi);
    for (uint32_t s = 0; s < Hb.rows; ++s)
        for (uint32_t j = 0; j < t_hi; ++j) Y.at(s, j) = static_cast<Elt>(Hb.at(s, j) % cl.p);
    const std::vector<std::vector<Elt>> co = lo.H_class_coordinates(q, engine::cview(Y));
    for (uint32_t j = 0; j < t_hi; ++j)
        for (uint32_t i = 0; i < t_lo; ++i) R.at(i, j) = co[j][i];
    return R;
}

engine::DenseBuf<uint32_t> BocksteinConeHc(const ModuleConeT<uint32_t>& ring2, const EquivariantMappingCone& field,
                                           uint32_t q) {
    const ModCtx<uint32_t>& c2 = ring2.ring_context();
    refuse_bredon_cone("BocksteinConeHc", c2);
    ENGINE_CHECK_MSG(c2.k == 2 && !field.is_composite() && field.ell() == c2.ell,
                     "BocksteinConeHc: needs a Z/ell^2 cone and the F_ell cone at the same prime");
    ENGINE_CHECK_MSG(q + 1 <= ring2.q_max(), "BocksteinConeHc: q + 1 is not a reported degree of the cone");
    const uint32_t ell = c2.ell;
    const ModCtx<uint32_t> F(ell);
    const auto& md = ring2.degree(q);
    const auto& mn = ring2.degree(q + 1);
    const DenseBuf<uint32_t>& C = md.C;
    const DenseBuf<uint32_t>& X = md.X;
    const DenseBuf<uint32_t>& Cn = mn.C;
    ENGINE_CHECK_MSG(X.cols == C.cols && X.rows == Cn.cols, "BocksteinConeHc: D_c is not Cc^q_eq -> Cc^{q+1}_eq");
    const DenseBuf<uint32_t> Hf = field.H_block(q).to_flat(engine::Layout::ColMajor);   // flat_dim x dim H^q_c
    const DenseBuf<uint32_t> Pn = field.P_block(q + 1).to_dense(engine::Layout::RowMajor);
    const uint32_t h = field.dim_Hc(q);
    DenseBuf<uint32_t> B(field.dim_Hc(q + 1), h);
    if (h == 0 || B.rows == 0) return B;
    ENGINE_CHECK_MSG(Hf.rows == C.rows && Pn.cols == Cn.rows, "BocksteinConeHc: the two cones' cochain spaces differ");
    DenseBuf<uint32_t> Cm(C.rows, C.cols);
    for (uint32_t s = 0; s < C.rows; ++s)
        for (uint32_t j = 0; j < C.cols; ++j) Cm.at(s, j) = C.at(s, j) % ell;
    DenseBuf<uint32_t> Z(Hf.rows, h);
    for (uint32_t s = 0; s < Hf.rows; ++s)
        for (uint32_t t = 0; t < h; ++t) Z.at(s, t) = Hf.at(s, t);
    const std::vector<std::vector<uint32_t>> co = engine::elim::solve_multi_rhs(F, engine::cview(Cm), engine::cview(Z));
    DenseBuf<uint32_t> W(Cn.rows, h);
    for (uint32_t t = 0; t < h; ++t) {
        std::vector<uint32_t> wc(X.rows, 0);
        for (uint32_t i = 0; i < X.rows; ++i) {
            uint32_t acc = 0;
            for (uint32_t j = 0; j < X.cols; ++j)
                if (co[t][j] != 0) acc = c2.add(acc, c2.mul(X.at(i, j), co[t][j]));
            if (acc % ell != 0)
                throw std::runtime_error("BocksteinConeHc: D_c of the lift is not divisible by ell (degree " + std::to_string(q) + ")");
            wc[i] = (acc / ell) % ell;
        }
        for (uint32_t s = 0; s < Cn.rows; ++s) {
            uint32_t acc = 0;
            for (uint32_t i = 0; i < Cn.cols; ++i)
                if (wc[i] != 0) acc = F.add(acc, F.mul(Cn.at(s, i) % ell, wc[i]));
            W.at(s, t) = acc;
        }
    }
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t t = 0; t < h; ++t) {
            uint32_t acc = 0;
            for (uint32_t s = 0; s < Pn.cols; ++s)
                if (W.at(s, t) != 0 && Pn.at(i, s) != 0) acc = F.add(acc, F.mul(Pn.at(i, s), W.at(s, t)));
            B.at(i, t) = acc;
        }
    return B;
}

template<class Elt>
bool ReductionIntertwines(Elt ell, const engine::DenseBuf<Elt>& X_field, const engine::DenseBuf<Elt>& R_src,
                          const engine::DenseBuf<Elt>& X_ring, const engine::DenseBuf<Elt>& R_tgt) {
    const uint32_t tf = X_field.rows, sf = X_field.cols, s = R_src.cols, t = R_tgt.cols;
    if (R_src.rows != sf || R_tgt.rows != tf || X_ring.rows != t || X_ring.cols != s) return false;
    const ModCtx<Elt> F(ell);
    const DenseBuf<Elt> L = product<Elt>(F, tf, sf, s, [&](uint32_t i, uint32_t j) { return static_cast<Elt>(X_field.at(i, j) % ell); },
                                         [&](uint32_t i, uint32_t j) { return static_cast<Elt>(R_src.at(i, j) % ell); });
    const DenseBuf<Elt> Rt = product<Elt>(F, tf, t, s, [&](uint32_t i, uint32_t j) { return static_cast<Elt>(R_tgt.at(i, j) % ell); },
                                          [&](uint32_t i, uint32_t j) { return static_cast<Elt>(X_ring.at(i, j) % ell); });
    for (uint32_t i = 0; i < tf; ++i)
        for (uint32_t j = 0; j < s; ++j)
            if (L.at(i, j) != Rt.at(i, j)) return false;
    return true;
}
template engine::DenseBuf<uint32_t> ReduceConeHcBetweenLevels<uint32_t>(const ModuleConeT<uint32_t>&, const ModuleConeT<uint32_t>&, uint32_t);
template engine::DenseBuf<uint64_t> ReduceConeHcBetweenLevels<uint64_t>(const ModuleConeT<uint64_t>&, const ModuleConeT<uint64_t>&, uint32_t);
template bool ReductionIntertwines<uint32_t>(uint32_t, const engine::DenseBuf<uint32_t>&, const engine::DenseBuf<uint32_t>&,
                                             const engine::DenseBuf<uint32_t>&, const engine::DenseBuf<uint32_t>&);
template bool ReductionIntertwines<uint64_t>(uint64_t, const engine::DenseBuf<uint64_t>&, const engine::DenseBuf<uint64_t>&,
                                             const engine::DenseBuf<uint64_t>&, const engine::DenseBuf<uint64_t>&);

} // namespace hecke
