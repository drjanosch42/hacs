// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_CENSUS_SPLIT_H
#define HECKE_EXT_CENSUS_SPLIT_H

// ===========================================================================
// hecke-extensions slot 11, WP-7 -- THE CENSUS SPLIT: the congruence census's
// computed columns, per maximal ideal of H^1 ([E] `sec:census` (d1), (d2),
// (d6), (e); slot-11 plan Sec. 3 "WP-7" scope 2, [AUD] B-2).
//
// WHAT THIS COMPUTES.  For one cell (n, ell) it runs the shipped extension
// driver (HeckeExt/HeckeExtDriver) with the options the published campaign
// used (`with_cone`, `with_sign_split`, the default family {2, 3, 5, 7}), so
// its record is the published record's computation, and then, for every
// maximal ideal m whose OWN group is H^1:
//
//   (d1) the BOUNDARY SPLIT.  E_m := the generalised eigenspace of m inside
//        the unsplit H^1(Ga; M_n) (every published record has sign = "all"),
//        re-derived by re-running Stage 2 (hecke::HeckeModuleDecomposition)
//        on the driver's OWN operators (`cell_operators(1)`); deterministic,
//        so its blocks are the record's ideals -- ASSERTED block by block on
//        `witness_poly` and the restricted t_p (entrywise).  Then
//        rank_r1 := rank(r^1 |_{E_m}) with r^1 : H^1 -> H^1(bd) the
//        restriction of the driver's own SL2ZBoundaryPair (the map
//        HeckeBoundary.h:293 uses on the joint Eisenstein eigenspace).
//        Verdict: `eisenstein` iff rank_r1 = dim E_m, `cuspidal` iff 0,
//        `mixed` iff strictly between.  A MOD-ell statement about
//        ker r^1 = H^1_!(F_ell), NOT about reductions of cusp forms
//        ([E] `sec:tests`: for ell <= n + 1 "H^1_! is the cuspidal part"
//        fails in both directions).
//   (d2) the SIGN content.  dim_plus := dim(E_m cap H^1_+), dim_minus :=
//        dim(E_m cap H^1_-) for the production iota^* (HeckeInvolution, the
//        driver's own); E_m is iota^*-stable, so they sum to dim E_m
//        (asserted).
//   (d6) the GLOVER RECONCILIATION (ell in {5, 7, 11, 13}): the ideal's
//        residue class c over the run's anemic primes (HeckeAtlas's
//        AtlasIdealCharacters, the same reading HeckeExtEndToEndValidation's
//        identity-6 arm uses), and, THROUGH HeckeIdentityCertificate's item-6
//        evaluation (IdentityResidueRhs; never a second implementation):
//          glover_multiplicity := the coefficient of c in [H^1(M_n)],
//          glover_main         := its coefficient in sum_i theta^{j_i}[H^1(M_{n_i})],
//          glover_correction   := its coefficient in the H^0 / H^2 terms,
//        the main term being the RHS evaluated on a copy of the certificate
//        whose H^0 / H^2 classes are removed.  Identity 6 is
//        multiplicity - main == correction; a class where it fails is
//        REPORTED (`identity6_mismatch`), never re-fitted.
//
// Per-cell invariants, asserted here and reported in `failures`:
//   sum dim_E = dim H^1; sum rank_r1 = rank r^1 = dim H^1 - dim H^1_!;
//   sum dim_plus = dim H^1_+ (and minus); identity 6's item verdict.
//
// THE LINE ([E] `sec:census` (e)), one per H^1 row, under the read-only
// interlock HECKE_CENSUS_SPLIT_ALL=1 of HeckeCensusSplitValidation:
//   CENSUS_SPLIT n= ell= ideal= witness= dim_E= rank_r1= verdict= dim_plus=
//     dim_minus= glover_mult= glover_main= glover_corr= glover= t_p=<fnv64>
// `t_p` is appended AFTER [E]'s fields (an addition, not a change): the
// FNV-1a-64 of the canonical text of the restricted operators
// ("<p>:[[a,b],[c,d]];..." ascending in p), which the census tool
// recomputes from the published record to apply [E]'s matching rule
// (witness_poly AND t_p, exact, one-to-one) without re-reading matrices.
//
// Refusal: F_ell only ([E] `sec:census` (j)); a non-default coefficient
// ring is refused by the driver it calls (exit 5).
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t>; includes Hecke/ pipeline
// headers one-way (as HeckeExtDriver); no Engine/, Cuda/, Console/,
// BlockFlatMatrix.h, no unordered_ container.  Compiled into cohomology_app
// (cmake/slot11/WP7.cmake).  No record member, no settings key, no golden
// line: the census is a derived artefact (`hecke_census/v1`).
// ===========================================================================

#include "HeckeIdentityCertificate.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {

struct CensusSplitRow {
    uint32_t ideal_id = 0;                 // the record's ideal id
    std::vector<uint32_t> witness;         // witness_poly, low -> high
    std::vector<uint64_t> operator_primes; // ascending
    uint64_t t_p_fnv = 0;                  // CensusTpFingerprint of the restricted t_p
    uint32_t dim_E = 0;
    uint32_t rank_r1 = 0;
    std::string verdict;                   // eisenstein | cuspidal | mixed
    uint32_t dim_plus = 0, dim_minus = 0;
    bool     sign_computed = false;

    // (d6) -- `glover_evaluated` false: every Glover column reads
    // `glover_token` (not_evaluated), the numbers are unused.
    bool        glover_evaluated = false;
    int64_t     glover_mult = 0, glover_main = 0, glover_corr = 0;
    std::string glover_token;              // equal | differs_by:<d> | class_level:<ids> | not_evaluated
    bool        identity6_mismatch = false;
};

struct CensusSplitCell {
    uint32_t n = 0, ell = 0;
    uint32_t dim_H1 = 0;
    uint32_t rank_r1 = 0;                  // rank r^1 on the whole H^1
    uint32_t dim_H1_plus = 0, dim_H1_minus = 0;
    std::vector<CensusSplitRow> rows;      // the H^1-own ideals, record order
    std::string glover_status;             // evaluated | not_evaluated:<why>
    uint32_t identity_code = 0;            // IdentityCertificate::verify(), 0 = pass / n.a.
    std::string identity_detail;           // item 6's detail text
    std::vector<std::string> failures;     // empty iff every invariant held
    double seconds = 0.0;
};

struct CensusSplitOptions {
    // 0 = the u_ell setting in force (the shipped `auto`).  Otherwise the
    // cell runs under `u_ell = budget_mb=<value>` (ext_u691's 8192 at the
    // over-budget ell = 691 cells), source `spec`.
    uint32_t u_ell_budget_mb = 0;
    // Evaluate (d6) at ell in {5, 7, 11, 13}.
    bool glover = true;
};

// The canonical text's FNV-1a-64 of a family of restricted operators (see
// the header note); `primes` ascending, index-aligned with `t`.
uint64_t CensusTpFingerprint(const std::vector<uint64_t>& primes,
                             const std::vector<MatrixT<uint32_t>>& t);

// Runs one cell.  Throws (ENGINE_CHECK) only on a structural failure -- a
// re-derived block that is not the record's ideal, a malformed map; a failed
// INVARIANT is reported in `failures`, never thrown, so a census run lists
// every cell.  `weights` / `weight_primes`: the (d6) weight-cell multisets
// of this ell and the anemic primes they are over (CensusGloverWeights);
// null disables (d6) for the cell.
CensusSplitCell ComputeCensusSplit(uint32_t n, uint32_t ell, const CensusSplitOptions& opts,
                                   const std::vector<std::vector<ResidueCharacterMultiset>>* weights,
                                   const std::vector<uint64_t>* weight_primes);

// The residue multisets of the weight cells m = 0, 2, ..., ell - 1, by the
// same driver options as a census cell; `primes_out` the anemic primes they
// are over.  False (with `why`) where some weight cell's ideal is not local
// or its anemic system does not generate its residue field.
bool CensusGloverWeights(uint32_t ell, std::vector<std::vector<ResidueCharacterMultiset>>& out,
                         std::vector<uint64_t>& primes_out, std::string& why);

// The CENSUS_SPLIT line of one row ([E] `sec:census` (e) + `t_p`).
std::string CensusSplitLine(const CensusSplitCell& cell, const CensusSplitRow& row);

} // namespace hecke_ext

#endif // HECKE_EXT_CENSUS_SPLIT_H
