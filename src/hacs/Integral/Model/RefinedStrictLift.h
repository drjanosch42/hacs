// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_MODEL_REFINED_STRICT_LIFT_H
#define INTEGRAL_MODEL_REFINED_STRICT_LIFT_H

// ===========================================================================
// hecke-extensions slot 18, WP-7 (upper core): THE STRICT LIFT ON A SYNTHETIC
// REFINED DESCRIPTOR -- [C, sec:isomodelell] code map item (v), checked by
// [C, IL5] ([C] = docs/latex/gl2_contract.tex v1.18.0-pre.1, read with referee
// F1's repair M-1: Iso^(ell)(Y') is PROJECTIVE, not free, so the route is
// [C, prop:isomodelell](4),(5) under the hypothesis "Iso^(ell)(Y') is an
// ell-local isotropy model over Y'" -- certified by LocalModel::certify() --
// and the strict identity below is a CERTIFICATE of the built map, never a
// hypothesis).
//
// WHAT IS BUILT.
//   * BuildSubdivisionMaps(X, Y): from Y's subdivision record (ModelData::
//     collapse / refines, written by SubdivideEdgeOrbit), the two Gamma-
//     equivariant chain maps between the orbit chain complexes, on orbit
//     representatives (a chain is sum coeff * g . rep, the RepTerm list):
//       collapse  c : C_.(Y') -> C_.(Xbulk), the simplicial vertex map
//                 (a midpoint goes to the first vertex of its stored edge;
//                 a degenerate image is 0);
//       subdivide sd: C_.(Xbulk) -> C_.(Y'), a simplex to the sum of its pieces.
//     CertifySubdivisionMaps re-derives d c = c d, d sd = sd d and c o sd = id
//     exactly on canonical orbit chains (cosets of the stabilisers).
//   * StrictComparisonOver(src, dst, base): the strict lift of a base chain map
//     (dst's base) <- (src's base) between two ell-local models on DIFFERENT
//     descriptors ([C, lem:strictlifting]: lift f o eps_src along the surjective
//     quasi-isomorphism eps_dst; y_0 the t = 0 section of base(g . rep), the
//     correction by dst.contract, every value averaged by e_x as in
//     StrictComparison).  Phi_Y : Iso(Xbulk) -> Iso^(ell)(Y') over sd and
//     Psi_Y : Iso^(ell)(Y') -> Iso(Xbulk) over c: the comparison of the models
//     over Y' and over Xbulk through which [C, IL5]'s transported T'_p runs.
//   * RefinedStrictLift(A, tau): the strict lift tau' on A = Iso^(ell)(Y') of the
//     base datum f = sd o tau^s o c (tau^s the SHIPPED strict pair datum of
//     [C, prop:strictlift], evaluated through the shipped lift's (S) identity
//     eps tau_Iso = tau^s eps; the shipped coset tables' s_j, delta_j and (T1)
//     helper; no new coset or tau-table).  The induction of
//     [C, lem:strictlifting] degree by degree on the generators s_j e'_c
//     (y_0 = the t = 0 section of f(s_j sigma_c), z = tau'(d x) - d y_0 in the
//     acyclic ker eps', k = A.contract(z)), then -- the projective repair --
//     tau' := tau~ o s, where tau~ is the twisted-equivariant extension of the
//     raw values to the free cover (+)_c R[Gamma] e~_c and s(a e'_c) = a e_c e~_c
//     the Gamma-linear section: tau'(s_j e'_c) = |Q_c|^{-1} sum_{u in Q_c}
//     tau~(s_j u e~_c).  This is what makes tau' well defined on R[Gamma] e_c
//     (the C-1 averaging of [C, prop:isomodelell](1),(4), now for a twisted
//     map; load-bearing at the NON-central Q_c: <S> at ell = 3, <U^2> at
//     ell = 2).  (T2) and strictness survive the average because the lower-
//     degree tau' and f are twisted-equivariant (the report, Sec. 2).
//   * certify(): re-derived from the stored values, exactly in Q[Gamma]:
//     (T1) on the Schreier generators h = s_i x s_{i.x}^{-1}, x in {T, S}, of
//     Gamma_0(N), the conjugate g^{-1} h g formed independently of the helper;
//     (T2) d tau' = tau' d on every generator; (T3) aug eps' tau' = aug eps' in
//     degree 0; (S) eps' tau' = f eps' on every generator of degree <= 2
//     (STRICT over Y'); integer sizes (max |numerator|, max denominator, terms).
//   * theta_hat(ctx, M, q): the coset transport [C, eq:cosettransport] on
//     C^q_{Iso^(ell)(Y')} in LocalModel's layout (cell * dim + component):
//     (Theta phi)(e'_c) = sum_j rho(delta_j) phi(tau'(s_j e'_c)).  (The average
//     enters the transport only as the projector: with the raw values in place
//     of tau'(s_j e'_c) the sum over the cosets gives P Theta_raw = Theta -- a
//     change of coset representatives -- so the average is tested on tau'
//     itself: changed_by_average counts the values tau~(s_j u e~_c) != raw,
//     i.e. the raw table is NOT a map on R[Gamma] e_c.)
//   * perturb(q, j, cell): NEGATIVE-CONTROL HOOK (validation only): one stored
//     raw value gets e'_cell added; certify() must then fail.
//
// MP CONSTRAINT ([C, rmk:isomodellmp](3), the code map's item (v)).  Nothing
// here reads a vertex count, an edge list, the 7-vertex model or a special
// vertex: Y' is any descriptor carrying a subdivision record of the descriptor
// the shipped lift is built on; -I lies in every stabiliser and acts on the
// coefficients by its scalar (a module with -I |-> -1 is admissible input);
// no Bredon object appears.  The refined descriptor stays out of
// Hecke/HeckeModelDescriptor's registry and out of the golden harness.
//
// DEPENDENCIES: IsotropyModel, IsotropyModelEll, StrictLift, PeriodicColumn,
// Hecke/ExactInt.h, ModArithmetic/ only.  No HeckeExt/, no bignum.
// ===========================================================================

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "IsotropyModel.h"
#include "IsotropyModelEll.h"
#include "StrictLift.h"

namespace integral_model {

// coeff * g . reps(s)[rep] -- one term of an orbit chain.
struct RepTerm {
    uint32_t rep = 0;
    hecke::Mat22 g;
    int64_t coeff = 0;
};
using RepChain = std::vector<RepTerm>;
// [s][rep] -> the image of reps(s)[rep].
using RepChainMap = std::array<std::vector<RepChain>, 3>;
// A canonical orbit chain: (rep, canonical coset key) -> multiplicity.
using CanonChain = std::map<std::pair<uint32_t, GKey>, Rat>;

// The canonical form of an orbit chain of dimension s of the model M.
CanonChain Canonical(const IsotropyModel& M, uint32_t s, const RepChain& x);

struct SubdivisionMaps {
    RepChainMap collapse;    // over Y's representatives, chains of X
    RepChainMap subdivide;   // over X's representatives, chains of Y
};
// X: the root descriptor (empty subdivision record); Y: a refinement of it.
SubdivisionMaps BuildSubdivisionMaps(const IsotropyModel& X, const IsotropyModel& Y);

struct SubdivisionCertificate {
    bool collapse_chain_map = false;     // d c = c d
    bool subdivide_chain_map = false;    // d sd = sd d
    bool collapse_subdivide_id = false;  // c o sd = id
    uint64_t checks = 0;
    std::string first_failure;
    bool all() const { return collapse_chain_map && subdivide_chain_map && collapse_subdivide_id; }
};
SubdivisionCertificate CertifySubdivisionMaps(const IsotropyModel& X, const IsotropyModel& Y, const SubdivisionMaps& m);

// The strict lift of a base chain map between two models on (possibly) different
// descriptors; base[s][src rep] is a chain of dst's descriptor.
ComparisonMap StrictComparisonOver(const LocalModel& src, const LocalModel& dst, const RepChainMap& base,
                                   LiftAveraging avg = LiftAveraging::On);
// chain map, strict over the base (eps_dst f = base eps_src), well defined (e_x f(x) = f(x)).
ComparisonCertificate CertifyComparisonOver(const LocalModel& src, const LocalModel& dst, const RepChainMap& base,
                                            const ComparisonMap& f);

struct RefinedLiftCertificate {
    bool t1_twisted_equivariance = false;
    bool t2_chain_map = false;
    bool t3_augmentation = false;
    bool strict = false;                 // eps' tau' = f eps'
    uint64_t t1_checks = 0, t2_checks = 0, t3_checks = 0, strict_checks = 0;
    uint32_t schreier_generators = 0;
    uint64_t changed_by_average = 0;     // values tau'(s_j e'_c) != the raw value
    int64_t max_abs_numerator = 0;
    int64_t max_denominator = 1;
    std::vector<uint64_t> terms;         // per degree, over all stored values
    std::string first_failure;
    bool all() const { return t1_twisted_equivariance && t2_chain_map && t3_augmentation && strict; }
};

class RefinedStrictLift {
public:
    // A: the ell-local model on the refined descriptor Y; X: the model the shipped
    // lift `tau` is built on (its descriptor is Y's root); maps from
    // BuildSubdivisionMaps(X, Y).  All four must outlive this object.
    RefinedStrictLift(const LocalModel& A, const StrictLift& tau, const SubdivisionMaps& maps);

    const LocalModel& model() const { return *A_; }
    uint32_t num_cosets() const { return m_; }
    uint32_t q_top() const { return static_cast<uint32_t>(raw_.size()) - 1; }

    // tau'(s_j e'_cell) (averaged) and the raw induction value.
    LChain value(uint32_t q, uint32_t j, uint32_t cell) const;
    const LChain& raw(uint32_t q, uint32_t j, uint32_t cell) const { return raw_.at(q).at(j - 1).at(cell); }
    // tau' of any chain of degree q with canonical coefficients.
    LChain apply(uint32_t q, const LChain& x) const;
    // The base datum f = sd o tau^s o c on the orbit chain g . sigma_rep of Y (dimension s), canonical.
    CanonChain base(uint32_t s, uint32_t rep, const hecke::Mat22& g) const;

    RefinedLiftCertificate certify() const;

    template<class Elt>
    engine::DenseBuf<Elt> theta_hat(const engine::ModCtx<Elt>& ctx, const CoefficientModule<Elt>& M, uint32_t q) const;

    // NEGATIVE-CONTROL HOOK (validation only).
    void perturb(uint32_t q, uint32_t j, uint32_t cell);

private:
    const LocalModel* A_;
    const StrictLift* tau_;
    const SubdivisionMaps* maps_;
    uint32_t m_ = 0;
    std::vector<std::vector<std::vector<LChain>>> raw_;   // [q][j - 1][cell]

    LChain tilde(uint32_t q, const hecke::Mat22& gamma, uint32_t cell) const;   // tau~(gamma e~_cell)
    LChain base_section(uint32_t s, uint32_t rep, const hecke::Mat22& g) const; // t = 0 section of f(g . sigma)
    void build();
};

}  // namespace integral_model

#endif
