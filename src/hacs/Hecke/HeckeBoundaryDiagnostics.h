// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_BOUNDARY_DIAGNOSTICS_H
#define HECKE_BOUNDARY_DIAGNOSTICS_H

// hecke-extensions slot 2, WP-7 (phase E4b) -- the two RECORDED diagnostics
// of docs/latex/gl2_contract.tex ([C], cited by label) sec:cone, remark
// rmk:conestrict, and of [X] hecke_extensions_implementation_plan.md
// Sec. 0.4 C-A3(iii)(5):
//
//   eta_zero         := (eta == 0), i.e. whether the shipped bulk datum tau
//                       is a MAP OF PAIRS against the certified boundary
//                       datum tau_bd: tau(s_j . i(sigma)) == j(tau_bd(s_j .
//                       i(sigma))) as chains of C_*(X) on every stored
//                       boundary slot (j, sigma) ([C, rmk:conestrict]: "h = 0
//                       is admissible iff the eta-table is identically zero,
//                       i.e. iff tau j = j tau_bd EXACTLY"). Evaluated by the
//                       CUSP-SUPPORT WALK of the bulk table: every stored
//                       slot of HeckeTauTable::tau(j, q, sigma) at a boundary
//                       simplex sigma is compared, term by term in canonical
//                       (KuSimplexKey) form over Z, with the boundary table's
//                       chain, and it is also recorded whether the bulk chain
//                       stays on the boundary subcomplex at all.
//   cochain_commutes := (r^q . Theta_g == Theta_g^bd . r^q entrywise on
//                       C^q_eq(K)), the cochain-level restriction identity,
//                       evaluated on the stored C_eq basis of every degree the
//                       boundary complex reports, through the shipped
//                       restriction morphism EquivariantCochainComplexMorphismOnK
//                       and the two operators' apply_Theta().
//
// BOTH ARE RECORDED AND GATE NOTHING ([C, rmk:conestrict](2), rule A6 of [C];
// [V Sec. 2.7] of docs/implementation/hecke_extensions_OQ2_verdict.md): a
// true value licenses nothing beyond the statement that names it. eta_zero
// implies cochain_commutes, not conversely ([C, rmk:conestrict](2)) -- that
// implication is the one CONSISTENCY assertion a consumer may make about the
// pair, and HeckeBoundaryParabolicValidation makes it. Expected on the shipped
// tau: eta_zero = false and cochain_commutes = false (the bulk tau is not a
// map of pairs, docs/implementation/tau_table_p2.txt:13-15,35; [WP1-4a]
// finding 8) -- neither is a defect.
//
// Placement (hecke_extensions_slot2_plan.md Sec. 3 WP-7 scope 2): this unit
// consumes HeckeOperatorOnK (whose header carries the block model) and the
// tau tables, so by rule 2 of check_heckeext_includes.sh it lives in Hecke/,
// beside HeckeTauTable / HeckeBoundaryDatum, and is consumed one-way by
// HeckeExt/HeckeBoundary. The eta-table of WP-8 (Hecke/HeckePairHomotopy)
// can compare its own eta == 0 verdict with cusp_support_walk() without
// crossing the guard. uint32_t only: the restriction morphism is uint32_t.
// No raw mod_* call (the comparisons are exact integer / entrywise).

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryOperator.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "../EquivariantCochainComplexMorphismOnK.h"

namespace hecke {

// The cusp-support walk of the bulk tau over the boundary slots.
struct CuspSupportWalk {
    uint32_t num_cosets = 0;            // m
    uint32_t slots = 0;                 // #(j, q, sigma): j in [1, m], q <= boundary_dim, sigma boundary
    uint32_t slots_off_boundary = 0;    // bulk chain carries a term on a non-boundary simplex of K
    uint32_t slots_differ = 0;          // bulk chain != tau_bd chain as canonical chains over Z
    uint32_t slots_equal = 0;           // == slots - slots_differ
    std::vector<uint32_t> off_boundary_per_degree; // [q]
    std::vector<uint32_t> differ_per_degree;       // [q]
    std::vector<std::string> witnesses; // the first few differing slots, "(j=..,q=..,sigma=..)"

    // The recorded diagnostics.
    bool cusp_supported() const { return slots_off_boundary == 0; }
    bool eta_zero() const { return slots_differ == 0; }
    std::string to_string() const;
};

// Walk every stored boundary slot of `tau` (the bulk datum, HeckeTauTable::
// tau(j, q, sigma)) against `taub` (the certified boundary datum), both built
// on the SAME HeckeCosetTables (j ranges over taub.num_cosets(); the bulk
// table throws std::out_of_range if it disagrees). Chains are compared after
// canonicalisation through taub.model().canonical_key (the same canonical
// map HeckeBoundaryDatum's certification uses), exactly over Z. Records only;
// never throws on a difference.
CuspSupportWalk cusp_support_walk(const HeckeTauTable& tau, const HeckeBoundaryTauTable& taub);

// The cochain identity r^q Theta_g = Theta_g^bd r^q on the stored C_eq basis.
struct CochainCommutes {
    uint64_t p = 0;
    uint32_t nu = 0;
    std::vector<uint32_t> vectors_per_degree; // [q] #C^q_eq(K) basis vectors compared (0 above the boundary dim: vacuous)
    std::vector<uint32_t> failing_per_degree; // [q] #vectors on which the identity fails
    uint64_t vectors = 0;
    uint64_t failing = 0;

    bool holds() const { return failing == 0; }
    std::string to_string() const;
};

// bulk_coh: the computed bulk cohomology (K, the 7-vertex model); bulk_op: the
// computed HeckeOperatorOnK on it; bd_op: the computed boundary operator on
// the circle model; res: the restriction morphism with DOMAIN the boundary
// cohomology and CODOMAIN the bulk cohomology (the direction of
// HeckeBoundaryOperatorValidation / HeckeIotaSignValidation). Consistency of
// the four (ell, d_V, the two cohomology objects) is ENGINE_CHECKed. Records
// only; never throws on a failing identity.
CochainCommutes cochain_commutes(const EquivariantCohomologyOnK& bulk_coh, const HeckeOperatorOnK& bulk_op,
                                 const HeckeBoundaryOperator& bd_op,
                                 const EquivariantCochainComplexMorphismOnK& res);

} // namespace hecke

#endif // HECKE_BOUNDARY_DIAGNOSTICS_H
