// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_ISOTROPY_MODEL_H
#define INTEGRAL_MODEL_ISOTROPY_MODEL_H

// ===========================================================================
// Line 2, L2-1a (hecke-extensions slot 9, WP-10): the isotropy-resolution model
// R = Iso_. of [C, def:isomodel] on the UNREFINED tessellation (Xbulk, Xbdry),
// its boundary subcomplex R_d = Isob_., the comparison map eps = cmpI, and the
// cochain complex C^._I(R) = Hom_{Z[Gamma]}(R, V_R) over ANY engine::ModCtx
// (a field F_ell or a chain ring Z/ell^k).  [C] = docs/latex/gl2_contract.tex
// (v1.8.1), cited by label; the chapter is [C, sec:integral], the model
// [C, sec:isomodel].
//
// WHAT IS BUILT (the construction of [C, prop:isomodel] (1), made effective):
//
//   * Cells.  For s in {0, 1, 2} the orbit representatives Sigma_s of
//     [C, def:isomodel] (Sigma_0 = {v0, v1, v2, v4, v6}; Sigma_1 = the nine
//     edges of K other than v3v5 and v5v6; Sigma_2 = the five triangles), each
//     with its stabiliser generator g_sigma (ST at v4, S at v6, -I elsewhere) of
//     order m_sigma in {6, 4, 2}.  One free Z[Gamma]-generator e_{s,sigma,t}
//     per (s, sigma, t), t >= 0; degree q = s + t.  So rank Iso_q = w_q with
//     w_0 = 5, w_1 = 14, w_q = 19 (q >= 2), and rank Isob_q = 3, 6, 6, ...
//     ([C, prop:isomodel] (2) -- NOT the 7/18/23 of the integral-line brief).
//   * The differential d = d^(0) + d^(1) + d^(2) as a sparse matrix over the
//     integral group ring Z[Gamma], every group element an exact 2x2 integer
//     matrix (hecke::Mat22, overflow-checked), every multiplicity an int64:
//       d^(0) = (-1)^s (id (x) d^F), d^F the periodic resolution of Z over
//               Z[<g_sigma>] (g - 1 from odd t, the norm from even t >= 2);
//       d^(1)  the lift of the simplicial boundary through the transporters of
//               [C, lem:model7coincidences] (2), at t = 0 the boundary itself,
//               at t >= 1 by the Z-linear contracting homotopy of the column;
//       d^(2)  the perturbation G_{2,t} -> G_{0,t+1}, again by the homotopy.
//     Boundary summands are lifted first and d^(1) of a boundary edge lands in
//     boundary vertex summands only, so R_d is a subcomplex ([C, IM2]).
//   * eps = cmpI : R -> C_.(Xbulk), the augmentation on G_{s,0}, 0 on t >= 1.
//   * Every construction step is certified in place (ENGINE_CHECK, always on):
//     each cycle fed to a contracting homotopy is checked to be one, and
//     certify() re-derives d o d = 0 over Z[Gamma], R_d closed under d, and
//     eps d = d eps exactly over Z (on C_.(Xbulk) = (+) Z[Gamma / Gamma_rho]).
//   * The degree range is a CONSTRUCTION PARAMETER q_max (default 5): the
//     complex is built in degrees 0 .. q_max + 1, so H^q of the cochain complex
//     is available for q <= q_max.  There is no settings key (slot-9 plan
//     Sec. 2.3 item 14).
//
// WHAT IT GIVES A CALLER (WP-11, M0, builds on this API):
//
//   coboundary(ctx, n, q)   delta^q : C^q_I -> C^{q+1}_I over ctx, the dense
//                           (w_{q+1} d_V) x (w_q d_V) matrix, d_V = n + 1,
//                           block (b, b') = rho_n(a_{b' b}) reduced mod ctx.p,
//                           flat index = cell * d_V + component (the L1 layout).
//                           C_I(Z) (x) R = C_I(R) by construction (the entries
//                           are reductions of integers), [C, prop:isomodel] (3).
//   boundary_coboundary     the same on C^._{R_d} (boundary cells only).
//   comparison(ctx, n, q)   eps^* : C^q_eq -> C^q_I as a (w_q d_V) x (N_q d_V)
//                           matrix on the FLAT K-cochain layout of the shipped
//                           pipeline (row s of a K-cochain = its value on the
//                           s-th stored q-simplex of K): f |-> (f(sigma) at
//                           (q, sigma, 0), 0 elsewhere).
//   cohomology_lengths      the composition lengths of H^q(C_I (x) R), q <=
//                           q_max: dimensions over a field, lengths over Z/ell^k.
//   rho_n(ctx, g, n)        rho_n(g) mod ctx.p ([C, def:rhon]), division-free,
//                           valid over every ModCtx.
//
// DEPENDENCIES ([X] slot-9 WP-10 scope item 3): Hecke/ public headers and
// ModArithmetic/ only; nothing from HeckeExt/, nothing from the 7-vertex
// pipeline (the model descriptor below is [C, def:model7] written out, and
// HeckeIsotropyModelValidation cross-checks it against the shipped K and
// relation list).  No type wider than __int128; no bignum.
// ===========================================================================

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../Hecke/ExactInt.h"
#include "../../Hecke/HeckeModelDescriptor.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"
#include "../../../ModArithmetic/EnvLever.h"

namespace integral_model {

// A group element as its exact matrix [[a, b], [c, d]] (row-major); the key
// order is lexicographic on (a, b, c, d) -- the canonical coset
// representatives below are minima in this order.
using GKey = std::array<int64_t, 4>;
GKey key_of(const hecke::Mat22& g);
hecke::Mat22 mat_of(const GKey& k);

// An element of the integral group ring Z[Gamma]: finitely many (g, n) with
// n != 0.  Ordered map (no hash container on a result path).
struct GroupRingElt {
    std::map<GKey, int64_t> terms;

    bool is_zero() const { return terms.empty(); }
    static GroupRingElt unit(const hecke::Mat22& g, int64_t n = 1);
    void add(const GKey& g, int64_t n);                  // overflow-checked; drops zeros
    void add(const GroupRingElt& o, int64_t scale = 1);  // this += scale * o
    GroupRingElt operator*(const GroupRingElt& o) const; // product in Z[Gamma]
    bool operator==(const GroupRingElt& o) const { return terms == o.terms; }
    int64_t max_abs_coefficient() const;
    int64_t max_abs_entry() const;                        // over the matrices
    std::string to_string() const;
};

// One free generator e_{s, sigma, t} of Iso_q (q = s + t).
struct Cell {
    uint32_t s = 0;        // simplex dimension, 0..2
    uint32_t rep = 0;      // index into Sigma_s (IsotropyModel::reps(s))
    uint32_t t = 0;        // resolution degree
    bool boundary = false; // sigma in Sigma^d_s (a summand of R_d)
};

// One non-zero coefficient of d_q : Iso_q -> Iso_{q-1}:
//   d(e_col) = sum over entries with this col of  coeff * e_row.
struct DiffEntry {
    uint32_t row = 0;      // cell index in degree q - 1
    uint32_t col = 0;      // cell index in degree q
    GroupRingElt coeff;
};

// A simplex of K ([C, def:model7]) as its stored (increasing) vertex list.
using Simplex = std::vector<uint32_t>;

// A face transport: the stored face f of a representative equals g * rho for
// the representative rho of the orbit of f, orientation-preserving
// ([C, lem:model7coincidences] (2)).
struct FaceTransport {
    Simplex face;          // the face as a simplex of K
    uint32_t rep = 0;      // index into Sigma_{dim face}
    hecke::Mat22 g;        // face = g * reps(dim)[rep]
};

// ---------------------------------------------------------------------------
// (hecke-extensions slot 17, WP-7) THE MODEL DESCRIPTOR AS DATA.  Everything
// the construction of [C, prop:isomodel] (1) (and LocalModel's of
// [C, prop:isomodelell] (1)) reads of the model: the orbit representatives
// Sigma_s, their stabiliser generators and orders, the boundary flags, the
// transports of every other simplex of the fundamental complex K' onto its
// representative (simplex = g * rep, positionally, orientation-preserving),
// the vertex labels on the horocycle, and the hecke::SimplicialModelDescriptor
// header (name, N_q) -- the same Hecke/HeckeModelDescriptor record the shipped
// pipeline states its model by.  The face lists are DERIVED from this data
// (face_i = sigma minus its i-th vertex, sign (-1)^i times the parity of the
// stored orientation of the face), never written by hand.  The shipped
// [C, def:model7] model is ONE instance (SL2Z7VertexModelData()); a
// Gamma-equivariant subdivision is another (SubdivideEdgeOrbit) -- the
// refinement half of [C, rmk:isomodellmp] (3): nothing below reads a vertex
// count, an edge list or a special vertex.
// ---------------------------------------------------------------------------
struct ModelData {
    hecke::SimplicialModelDescriptor header;   // name, dimension 2, N_q of K' (checked against the data)
    std::array<std::vector<Simplex>, 3> reps;
    std::array<std::vector<hecke::Mat22>, 3> gen;
    std::array<std::vector<uint32_t>, 3> order;
    std::array<std::vector<bool>, 3> bdry;
    std::vector<FaceTransport> nonreps;        // every simplex of K' not in reps: face = g * reps[dim][rep]
    std::vector<uint32_t> horocycle;           // vertex labels of K' on Xbdry
    // (hecke-extensions slot 18, WP-7) THE SUBDIVISION RECORD, written by SubdivideEdgeOrbit
    // and empty on an unrefined (root) descriptor; read by RefinedStrictLift.h's
    // BuildSubdivisionMaps, never by the construction of the model (value-free).
    //   collapse : every vertex label created by a subdivision -> the ROOT descriptor's
    //              vertex label it collapses to (a midpoint of the stored edge [x, y]
    //              goes to (the collapse of) x): the simplicial vertex map Y' -> Xbulk;
    //   refines  : every simplex of the ROOT descriptor's K' (stored orientation) -> the
    //              signed simplices of this K' whose sum it is, as chains (the
    //              subdivision chain map Xbulk -> Y').  Root simplices that were never
    //              split are absent (they map to themselves).
    std::map<uint32_t, uint32_t> collapse;
    std::map<Simplex, std::vector<std::pair<int, Simplex>>> refines;
    // N_q of K' = |reps[q]| + |nonreps of dimension q|.
    std::vector<uint32_t> simplex_counts() const;
};

// [C, def:model7] / [C, lem:model7coincidences] (2) / [C, def:isomodel] as data
// (header = hecke::SL2Z7VertexModel()).
ModelData SL2Z7VertexModelData();

// A Gamma-equivariant subdivision ([C, rmk:orbifoldrefine] (1)): every simplex of
// K' in the orbit of the edge representative `edge` gets a midpoint (one new
// orbit of vertices, stabiliser = Gamma_edge), the edge orbit becomes two edge
// orbits, and every triangle of K' with a face in that orbit is split through
// the midpoint and its opposite vertex (one new interior edge orbit per split,
// stabiliser = the triangle's).  Works on ANY ModelData (iterate it for several
// edges); refuses (ENGINE_CHECK) an edge that is not a representative, and a
// descriptor with non-representative triangles (none on the shipped model), and
// (slot-17 WP-17) an edge or split triangle whose stabiliser is not the centre
// {+-I} -- the midpoint inherits the edge's stabiliser, right only when it fixes
// the edge pointwise.
// A boundary edge subdivides the horocycle (its midpoints are horocycle vertices).
ModelData SubdivideEdgeOrbit(const ModelData& d, const Simplex& edge);

struct Certificates {
    bool d_squared_zero = false;        // d_{q-1} d_q = 0 in Z[Gamma], every q
    bool boundary_subcomplex = false;   // d(R_d) subset of R_d  ([C, IM2])
    bool eps_chain_map = false;         // eps d = d eps over Z ([C, IM1])
    bool eps_boundary = false;          // eps(R_d) subset of C_.(Xbdry)
    uint64_t d_squared_terms_checked = 0;
    uint64_t eps_cells_checked = 0;
    int64_t max_abs_coefficient = 0;    // over every d entry
    int64_t max_abs_entry = 0;          // max |matrix entry| of a group element in d
    uint64_t distinct_group_elements = 0;
    std::string first_failure;          // empty when all hold
    bool all() const { return d_squared_zero && boundary_subcomplex && eps_chain_map && eps_boundary; }
};

class IsotropyModel {
public:
    // Builds Iso_q for q = 0 .. q_max + 1.  ENGINE_CHECKs every homotopy input
    // is a cycle; throws hecke::HeckeOverflow if an exact product overflows.
    explicit IsotropyModel(uint32_t q_max = 5);
    // (slot-17 WP-7) The same construction on any model descriptor given as data;
    // the default constructor is IsotropyModel(SL2Z7VertexModelData(), q_max).
    IsotropyModel(const ModelData& data, uint32_t q_max);

    const ModelData& data() const { return data_; }
    // True iff this is the shipped [C, def:model7] descriptor (comparison() and
    // K_simplices() describe only it).
    bool is_shipped_model() const { return shipped_; }

    uint32_t q_max() const { return q_max_; }
    uint32_t q_top() const { return q_max_ + 1; }

    // ---- the model descriptor ([C, def:model7], [C, lem:model7coincidences]) --
    static const std::vector<std::vector<Simplex>>& K_simplices();   // [q] stored order
    const std::vector<Simplex>& reps(uint32_t s) const { return reps_[s]; }
    const hecke::Mat22& stabiliser_generator(uint32_t s, uint32_t rep) const { return gen_[s][rep]; }
    uint32_t stabiliser_order(uint32_t s, uint32_t rep) const { return order_[s][rep]; }
    bool is_boundary_rep(uint32_t s, uint32_t rep) const { return bdry_[s][rep]; }
    // The transports used by d^(1) at t = 0: faces_[s][rep] lists the faces of
    // reps(s)[rep] with their sign (-1)^i.
    const std::vector<std::pair<int, FaceTransport>>& faces(uint32_t s, uint32_t rep) const { return faces_[s][rep]; }
    // Every non-representative simplex of K with its transport to its
    // representative (K simplex = g * rep).
    const std::vector<FaceTransport>& non_representatives() const { return nonreps_; }

    // ---- the complex over Z[Gamma] --------------------------------------------
    uint32_t rank(uint32_t q) const { return static_cast<uint32_t>(cells_.at(q).size()); }      // w_q
    uint32_t boundary_rank(uint32_t q) const;                                                     // w^d_q
    const std::vector<Cell>& cells(uint32_t q) const { return cells_.at(q); }
    uint32_t cell_index(uint32_t s, uint32_t rep, uint32_t t) const;
    // d_q for q = 1 .. q_top (entries sorted by (col, row)).
    const std::vector<DiffEntry>& differential(uint32_t q) const { return diff_.at(q); }
    // The distinct group elements occurring in any coefficient of d.
    std::vector<GKey> group_elements() const;

    // Re-derives the certificates of [C, def:isomodel] (IM1), (IM2).
    Certificates certify() const;

    // ---- the cochain complex over a ModCtx --------------------------------------
    template<class Elt>
    engine::DenseBuf<Elt> coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> boundary_coboundary(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    template<class Elt>
    engine::DenseBuf<Elt> comparison(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q) const;
    // lengths[q] of H^q(C_I (x) R) for q = 0 .. q_max (dimensions over a field);
    // boundary = true: of C_{R_d} (x) R.
    template<class Elt>
    std::vector<uint32_t> cohomology_lengths(const engine::ModCtx<Elt>& ctx, uint32_t n,
                                             bool boundary = false) const;

private:
    uint32_t q_max_;
    ModelData data_;
    bool shipped_ = false;
    std::array<std::vector<Simplex>, 3> reps_;
    std::array<std::vector<hecke::Mat22>, 3> gen_;
    std::array<std::vector<uint32_t>, 3> order_;
    std::array<std::vector<bool>, 3> bdry_;
    std::array<std::vector<std::vector<std::pair<int, FaceTransport>>>, 3> faces_;
    std::vector<FaceTransport> nonreps_;
    std::vector<std::vector<Cell>> cells_;
    std::vector<std::vector<DiffEntry>> diff_;   // diff_[q], q >= 1

    void load_descriptor(const ModelData& d);
    void build_differentials();
    template<class Elt>
    engine::DenseBuf<Elt> coboundary_impl(const engine::ModCtx<Elt>& ctx, uint32_t n, uint32_t q,
                                          bool boundary_only) const;
};

// rho_n(g) mod ctx.p ([C, def:rhon]; column j = coefficients of
// (a x + c y)^{n-j} (b x + d y)^j in the basis x^{n-t} y^t), division-free:
// a Pascal table by additions and the closed double sum.  Valid over every
// ModCtx (field or composite); O(d_V^3).
//
// (slot-23 F3L) With the F3L route on (the default; L23_F3L=0 is the rollback, read once)
// rho_n runs rho_n_recurrence -- the column recurrence (a x + c y) f_{j+1} = (b x + d y) f_j,
// O(d_V^2), solved ascending when a is a unit mod ctx.p and descending when c is -- and falls
// back to rho_n_cubic (the closed double sum above) when neither is a unit.  Both are the
// unique residues of [C, def:rhon]: equal entrywise (OrbifoldF3lValidation).
template<class Elt>
engine::DenseBuf<Elt> rho_n(const engine::ModCtx<Elt>& ctx, const hecke::Mat22& g, uint32_t n);
// The closed double sum, O(d_V^3): the reference (and the route with L23_F3L=0).
template<class Elt>
engine::DenseBuf<Elt> rho_n_cubic(const engine::ModCtx<Elt>& ctx, const hecke::Mat22& g, uint32_t n);
// The recurrence; false (M untouched) when neither a nor c is a unit mod ctx.p.
template<class Elt>
bool rho_n_recurrence(const engine::ModCtx<Elt>& ctx, const hecke::Mat22& g, uint32_t n, engine::DenseBuf<Elt>& M);

// The F3L route switch (slot-23 F3L; process-wide): L23_F3L=0 (read once) turns it off, any
// other value but 1 warns and leaves it on.  It selects rho_n's recurrence here and, in the
// orbifold campaign at ell = 3, the bit-sliced F_3 elimination (ScopedTernaryF3).  It never
// changes a value.
bool F3lRouteEnabled();
std::string F3lRouteSource();    // "default" | "env" | "scoped"
const engine::lever::EnvLever& F3lRouteLever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedF3lRoute : engine::lever::EnvLever::Scoped {   // process-wide override (validation use; not nestable across threads)
    explicit ScopedF3lRoute(bool on) : Scoped(F3lRouteLever(), on) {}
};
struct RhoNTotals { uint64_t recurrence = 0, cubic = 0, fallback = 0; };   // rho_n calls so far (diagnostic)
RhoNTotals rho_n_totals();

}  // namespace integral_model

#endif
