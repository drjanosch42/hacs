// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_FINALITY_ROUTES_H
#define HECKE_FINALITY_ROUTES_H

// ===========================================================================
// hecke-extensions slot 12, FIN -- the finality certificates (d0) "dickson"
// and (d1) "boundary" ([E] v1.13.0 `sec:certgen`, `cert:dickson`,
// `cert:boundary`; the design is docs/implementation/slot12/
// WP-12_finality_proposal.md v2, Sec. 3.1 / 3.4 / 5, owner-approved
// 2026-09-26).
//
//   (d0) Proposition D.  V a certified local T_S-block of X_1 = H^0(Gamma; M_n),
//        ell >= 5, n even, ell in S.  By [C, prop:heckeH0](1) the Dickson
//        monomials L^a D^b span X_1 and T_p (p != ell) acts on L^a D^b by the
//        scalar (1 + p) p^a, which depends on a mod (ell - 1) only.  If the
//        S-eigenvalue functions a |-> (1 + q) q^a (q in S \ {ell}) single out
//        ONE class among the classes present in Sym^n, every T_p|_V is a
//        scalar and T_infty|_V = F_ell[U_ell|_V] = A_S.
//   (d1) Proposition F'.  V a certified local T_S-block of X_4 = H^1(Gamma;
//        M_n), ell >= 5, n even, ell in S.  If r^1|_V is injective and T_p|_V
//        lies in A_S for the least prime p != ell of every class of
//        (Z/ell)^x, then T_infty|_V = A_S -- by [C, prop:bdrytranscription]
//        and [C, lem:bdrytorus](2) (the boundary operators away from ell
//        factor through the torus T(F_ell), so Theta^bd_p depends on p mod ell
//        only) and [C, thm:conewelldef](a) (r^1 intertwines).
//
// Both routes certify EXACTLY the predicate (a)-(c) certify ([E] `cl:gen`(b):
// T_infty|_V = A_S), run only when (a)-(c) did not pass, and on a failure the
// verdict falls through to the existing reasons (proposal Sec. 5 item 2).
//
// The functions here are pipeline-free (matrices in, strings out), so the
// record's verify() can recompute (d0) from the record alone, and
// BoundaryFinalityCertificate::verify() recomputes (d1) from the data it
// stores (the block basis, r^1, the ambient operators).
// ===========================================================================

#include "HeckeLocalAlgebra.h"   // ExtMatrix

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// The closed value lists of `certificate_routes.d0` / `.d1` ([E] sec:schema,
// hecke_ext/v4; unchanged in v5).  `failed:<class>` (d0) names the second class the
// S-eigenvalues could not separate from the first; `failed:<p>` (d1) names
// the first representative whose T_p|_V is not in A_S.
inline constexpr const char* kRouteNotRun           = "not_run";
inline constexpr const char* kRoutePassed           = "passed";
inline constexpr const char* kRouteNotApplicableUEll = "not_applicable:u_ell";
inline constexpr const char* kRouteNotInjective     = "not_injective";
inline constexpr const char* kRouteFailedPrefix     = "failed:";
// (S12-FIN audit G1) (d1) not run because streaming R(ell) would exceed its
// cost cap (BoundaryStreamWithinBudget); resumable by re-running with a larger
// `u_ell` budget -- the cap is tied to it, no new settings key.
inline constexpr const char* kRouteNotRunBudget     = "not_run:budget";

// (S12-FIN audit G1) THE COST CAP of (d1)'s representative stream ([X] Sec.
// 1.3.1 R3: no unbounded computation without a budget).  Each representative
// p of R(ell) outside the anemic pool (p > 31) needs one streamed T_p pipeline
// with p + 1 cosets; its cost is predicted by the frozen u_ell RSS model
// (HeckeUEllBudget.h, PredictUEllRssMb(n, p): the same (cosets, n) law as
// U_ell's).  The route runs iff the SUM over those representatives is at
// most kBoundaryStreamBudgetFactor x the `u_ell` budget in force -- i.e. the
// whole stream costs no more than four admitted U_ell builds.  Fixed and
// documented ([E] cert:boundary); every census cell (ell <= 13, n <= 200)
// lies within it (worst (200, 13): ~ 2.5 GB-equivalent against 8 GB at the
// default 2048 MB budget); (10, 691) (~ 690 streamed primes up to 20 983) and
// every ell = 10007 cell lie far outside.
inline constexpr uint32_t kBoundaryStreamBudgetFactor = 4;
double BoundaryStreamCostMb(uint32_t n, uint32_t ell);
bool   BoundaryStreamWithinBudget(uint32_t n, uint32_t ell, uint32_t u_ell_budget_mb);

// True iff `v` is in (d0)'s / (d1)'s closed value list.
bool IsDicksonRouteValue(const std::string& v);
bool IsBoundaryRouteValue(const std::string& v);

// The least prime q != ell with q = cls (mod ell), 1 <= cls < ell.
uint64_t LeastPrimeInClass(uint32_t ell, uint32_t cls);

// R(ell): the least prime of every class of (Z/ell)^x, in class order 1 ..
// ell - 1 (the key `boundary_reps`).
std::vector<uint64_t> BoundaryRepresentatives(uint32_t ell);

// The classes a mod (ell - 1) of the Dickson monomials L^a D^b of degree n,
// n = a (ell + 1) + b (ell^2 - ell), ascending.
std::vector<uint32_t> DicksonClassesPresent(uint32_t n, uint32_t ell);

// (d0) on one block, given the block's stored operators (t_p|_V for p in S,
// ascending, S may contain ell = U_ell).  Returns a closed-list value, or
// "invalid:<why>" when the data contradict [C, prop:heckeH0](1) (a t_q that is
// not a scalar, or no present class carrying the observed eigenvalues) -- a
// value the driver refuses by ENGINE_CHECK and the record's verify() by code.
// Preconditions (the caller's): ell >= 5, n even, n >= 1, V in X_1.
std::string DicksonRoute(uint32_t n, uint32_t ell, const std::vector<uint64_t>& primes,
                         const std::vector<ExtMatrix>& t);

// (d1) on one block: everything it decided with, so verify() can recompute it.
struct BoundaryFinalityCertificate {
    bool     present = false;
    uint32_t n = 0, ell = 0;
    ExtMatrix V;                          // ambient basis of the block (columns)
    ExtMatrix r1;                         // r^1 : H^1 -> H^1(bd), in the stored bases
    std::vector<ExtMatrix> generators;    // t_p|_V for p in S (A_S's generators)
    ExtMatrix coords;                     // A_S's RREF coordinates (dim_A x D^2)
    std::vector<uint64_t>  reps;          // R(ell), class order
    std::vector<ExtMatrix> reps_ambient;  // A^{(1)}(T_p) on H^1, p in reps (empty if not reached)
    std::vector<ExtMatrix> reps_restricted; // t_p|_V (D x D), index-aligned with reps_ambient
                                            // (slot-13 WP-7: the record's boundary_data.t)
    uint32_t    rank_r1 = 0;
    uint32_t    budget_mb = 0;                // the u_ell budget the cap was taken against
    std::string outcome = kRouteNotRun;

    // 0 iff every recorded decision re-derives: code 1 the outcome is not in
    // the closed list; 2 rank(r^1 V) != rank_r1 or the injectivity reading
    // disagrees; 3 the representatives are not R(ell); 4 a representative's
    // T_p|_V is not a restriction (V not stable) or its A_S membership, by the
    // independent HeckeAlgebraInvariants route, disagrees with the outcome;
    // 5 the cost cap's verdict (not_run:budget iff over the cap) disagrees.
    uint32_t verify() const;
};

// Runs (d1).  `ell_in_S` false -> not_applicable:u_ell (nothing else read);
// then not_injective; then not_run:budget when the stream is over the cap of
// `u_ell_budget_mb` (no representative is computed).
// `ambient_of(p)` returns A^{(1)}(T_p) on the whole H^1 (called for each
// representative in class order until the first failure).
template<class AmbientFn>
BoundaryFinalityCertificate BoundaryRoute(uint32_t n, uint32_t ell, bool ell_in_S,
                                          uint32_t u_ell_budget_mb,
                                          const ExtMatrix& V, const ExtMatrix& r1,
                                          const std::vector<ExtMatrix>& generators,
                                          const ExtMatrix& coords, AmbientFn ambient_of);

// Membership of X in the span whose RREF rows are `coords` (dim_A x D^2).
bool InAlgebraSpan(const ExtMatrix& coords, const ExtMatrix& X);

// (slot-13 WP-7, hecke_ext/v5; [E] v1.14.0-pre.1 cert:boundary) (d1)
// re-derived from RECORD DATA alone: `generators` the ideal's stored t_q
// (q in `primes` = S, ascending), `reps` = R(ell), `r1` = r^1|_V,
// `rank_r1`, `t` = the t_p|_V.  0 iff (H1) and (H2) hold on the data and
// the data are well formed; otherwise a clause index (1 shape / keys, 2 an
// entry outside [0, ell), 3 rank(r1) != rank_r1, 4 rank_r1 != D (H1),
// 5 a t_p for p in S differs from the stored t_p, 6 a t_p does not commute
// with the stored generators, 7 a t_p outside A_S (H2) -- membership by the
// algebra-closure route of HeckeAlgebraInvariants).  verify() code 32.
uint32_t BoundaryDataRederive(uint32_t ell, const std::vector<uint64_t>& primes,
                              const std::vector<ExtMatrix>& generators,
                              const std::vector<uint64_t>& reps, uint32_t rank_r1,
                              const ExtMatrix& r1, const std::vector<ExtMatrix>& t);

} // namespace hecke_ext

// ---------------------------------------------------------------------------
#include "../Hecke/HeckeLinAlg.h"

namespace hecke_ext {

template<class AmbientFn>
BoundaryFinalityCertificate BoundaryRoute(uint32_t n, uint32_t ell, bool ell_in_S,
                                          uint32_t u_ell_budget_mb,
                                          const ExtMatrix& V, const ExtMatrix& r1,
                                          const std::vector<ExtMatrix>& generators,
                                          const ExtMatrix& coords, AmbientFn ambient_of) {
    BoundaryFinalityCertificate c;
    c.present = true;
    c.n = n;
    c.ell = ell;
    if (!ell_in_S) {
        c.outcome = kRouteNotApplicableUEll;
        return c;
    }
    c.V = V;
    c.r1 = r1;
    c.generators = generators;
    c.coords = coords;
    c.reps = BoundaryRepresentatives(ell);
    c.rank_r1 = (r1.rows == 0) ? 0u : hecke::RankOf(r1 * V);
    if (c.rank_r1 != V.cols) {
        c.outcome = kRouteNotInjective;
        return c;
    }
    c.budget_mb = u_ell_budget_mb;
    if (!BoundaryStreamWithinBudget(n, ell, u_ell_budget_mb)) {
        c.outcome = kRouteNotRunBudget;
        return c;
    }
    c.outcome = kRoutePassed;
    for (uint64_t p : c.reps) {
        c.reps_ambient.push_back(ambient_of(p));
        c.reps_restricted.push_back(hecke::SolveColumnsExact(V, c.reps_ambient.back() * V,
                                                             "BoundaryRoute(restriction)"));
        if (!InAlgebraSpan(coords, c.reps_restricted.back())) {
            c.outcome = std::string(kRouteFailedPrefix) + std::to_string(p);
            break;
        }
    }
    return c;
}

} // namespace hecke_ext

#endif // HECKE_FINALITY_ROUTES_H
