// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_DESCRIPTOR_H
#define ORBIFOLD_MP_DESCRIPTOR_H

// ===========================================================================
// (hecke-extensions slot 22, MP-1a) Orbifold triangulations of [SL_2(Z)\X] as
// finite data, and the decidable Moerdijk--Pronk predicates on them.
// Plan: docs/implementation/slot22/MP_impl_plan.md v0.2 Sec. 2.2; mathematics:
// MP_math_plan.md v0.2 ([MP]) Defs. 1.1-1.5, Sec. 3.1; predictions
// MP_predictions.md P1 (committed before the first run).
//
// THE DESCRIPTOR.  integral_model::ModelData (Integral/Model/IsotropyModel.h)
// already is the Gamma-triangulation descriptor; FromModelData() reads off it
// what the MP rules need: the vertex orbits (representative label, stabiliser
// generator, order), every vertex label of the fundamental complex K' as a
// POINT of X, (orbit, g) = g * iota(orbit), and the top simplices (the
// transversal Sigma_top, all representatives on every ModelData in the tree).
// A point's canonical key is (orbit, least GKey of g * Gamma_orbit), so two
// labels are one point of X iff their keys agree (|Gamma_v| <= 6).
// R_good ([MP] Sec. 3.1; triangle centres, which SubdivideEdgeOrbit cannot
// make) is a hand table copied from mp_rules.py (RGoodDescriptor()).
//
// THE PREDICATES (all exact, on hecke::Mat22; nothing reads a vertex name):
//   * (SC): for top simplices s, s' and every vertex transporter gamma with
//     gamma iota(s') meeting iota(s): equal vertex sets => s = s' and gamma
//     fixes iota(s) pointwise; every lifted top simplex has distinct vertices.
//   * invariant order: no top simplex has two vertices in one orbit; the
//     stored order is then the orbit index (OrbitOrdered refuses otherwise).
//   * adapted, SUBGROUP form ([MP] Def. 1.3 (MP-ii)): every face f (>= 2
//     vertices) of every top simplex has a vertex v with Gamma_w <= Gamma_v
//     (the actual, transported stabilisers) for every vertex w of f -- tested
//     by generator membership, never by orders.  ell = 0: the full isotropy;
//     ell = 2, 3: the Sylow datum T^(ell) of [MP] Def. 1.4.  The order-only
//     form (mp_rules.py's) is kept beside it for the negative control.
//   * good ([MP] Def. 1.3 (MP-good), the finite-group criterion): for each
//     vertex orbit, the transporters T(v) of the closed star St(v) generate a
//     finite group; the closure is enumerated and declared infinite once it
//     exceeds 6 elements (finite subgroups of SL_2(Z) are cyclic of order <= 6).
//     The stronger "T(v) inside one stabiliser Gamma_p, p in St(v)" of
//     mp_rules.py is computed beside it (R-22b).
//   * collar: no top simplex meets both the horocycle and an elliptic vertex.
//   * quotient counts (V, E, F, chi) by orbit keys of lifted simplices.
//
// No modular arithmetic, no golden object, no setting.  Pure host C++.
// ===========================================================================

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../../Hecke/ExactInt.h"
#include "../../Integral/Model/IsotropyModel.h"

namespace orbifold_mp {

using hecke::Mat22;
using integral_model::GKey;

// A finite cyclic subgroup of SL_2(Z) by a generator and its exact order.
struct CyclicGroup {
    Mat22 gen = Mat22::I();
    uint32_t order = 1;
};
std::vector<Mat22> Elements(const CyclicGroup& H);                // I, g, ..., g^{m-1}
bool Contains(const CyclicGroup& H, const Mat22& x);               // x in H
bool IsSubgroup(const CyclicGroup& A, const CyclicGroup& B);       // A <= B (generator membership)
CyclicGroup Conjugate(const Mat22& g, const CyclicGroup& H);       // g H g^{-1}
CyclicGroup Sylow(const CyclicGroup& H, uint32_t ell);             // ell = 0: H itself

// A point of X: g * iota(orbit).
struct Point {
    uint32_t orbit = 0;
    Mat22 g = Mat22::I();
};
using PointKey = std::pair<uint32_t, GKey>;

struct Descriptor {
    std::string name;
    std::vector<uint32_t> orbit_label;   // orbit index -> the K' label of its representative
    std::vector<CyclicGroup> orbit_stab; // orbit index -> Gamma of the representative
    std::map<uint32_t, Point> point;     // K' vertex label -> point of X
    std::vector<std::vector<uint32_t>> top;   // Sigma_top (K' labels, stored order)
    std::set<uint32_t> boundary;         // horocycle labels

    PointKey key(const Point& p) const;
    PointKey key_of_label(uint32_t label) const { return key(point.at(label)); }
    CyclicGroup stabiliser(const Point& p) const;   // g Gamma_orbit g^{-1}
    std::vector<Point> lifted(uint32_t t) const;    // the points of iota(top[t])
    uint32_t elliptic_order(uint32_t label) const { return orbit_stab.at(point.at(label).orbit).order; }
};

// The descriptor of a ModelData (ENGINE_CHECK: every triangle a representative).
Descriptor FromModelData(const integral_model::ModelData& d, const std::string& name);
// R_good of [MP] Sec. 3.1, the table of mp_rules.py make("R_good") (its labels).
Descriptor RGoodDescriptor();

// The vertex transporters A(s0, s1) = {gamma : gamma iota(s1) meets iota(s0)}, sorted by GKey.
std::vector<Mat22> Arrows(const Descriptor& D, uint32_t s0, uint32_t s1);

// Lifted simplex (any dimension) -> its Gamma-orbit key (min over translates).
std::vector<PointKey> SimplexOrbitKey(const Descriptor& D, const std::vector<Point>& simplex);

// ---- the predicates ------------------------------------------------------
// One face of a simplex as the stabilisers of its vertices (the predicate input; plantable).
bool AdaptedFace(const std::vector<CyclicGroup>& vertex_stabs, bool subgroup_form);
bool Adapted(const Descriptor& D, uint32_t ell, bool subgroup_form);

struct GoodResult {
    bool good = true;                     // the finite-group criterion
    bool good_strong = true;              // T(v) inside one Gamma_p, p in St(v)
    std::vector<uint32_t> fails_at;       // representative labels failing the finite-group criterion
    std::vector<uint32_t> fails_at_strong;
};
GoodResult Good(const Descriptor& D);

bool Collar(const Descriptor& D);

struct SCResult {
    bool ok = true;
    uint64_t pairs_checked = 0;
    std::string first_failure;
};
SCResult SimplicialComplexProperty(const Descriptor& D);

bool InvariantOrder(const Descriptor& D);
// The labels of a simplex sorted by orbit index; ENGINE_CHECK refuses two vertices in one orbit.
std::vector<uint32_t> OrbitOrdered(const Descriptor& D, const std::vector<uint32_t>& simplex);

struct QuotientCounts {
    uint32_t V = 0, E = 0, F = 0;
    int64_t chi = 0;
};
QuotientCounts Quotient(const Descriptor& D);

// The ordered-chain bridge: on a ModelData, for every representative simplex and every face of it, the
// sign IsotropyModel's C_. uses equals the standard simplicial sign of the orbit-ordered simplices.
// Returns the number of faces checked; *ok = false on the first disagreement.
uint64_t OrderedChainSignBridge(const integral_model::ModelData& d, bool* ok);

}  // namespace orbifold_mp

#endif
