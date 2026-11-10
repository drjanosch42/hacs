// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_CENSUS_LATTICE_H
#define HECKE_EXT_CENSUS_LATTICE_H

// ===========================================================================
// hecke-extensions slot 13, WP-9 -- C-F9, THE LATTICE TEST ON THE CENSUS
// (docs/implementation/slot12/WP-12_finality_proposal.md v2 Sec. 7, "[PR] Sec. 7";
//  slot-13 plan Sec. WP-9).  A candidate, NOT a claim: nothing here is gated by
//  [C] or [E]; a row on which the candidate fails is a FINDING, reported as such.
//
// C-F9.  For ell >= 5 and a non-Dickson-type maximal ideal m of the Hecke algebra
// on H^1(Gamma; Sym^n Z_ell), Gamma = SL_2(Z):
//     (T_q - a_q)(T_q' - a_q') H^1(Z_ell)_m  is contained in  ell H^1(Z_ell)_m
// for all q, q' in the operator primes S of the cell -- i.e. m_A^2 = 0 on the
// lattice mod ell.
//
// What one cell computes (per H^1-own ideal of the record, record order):
//
//  (L) THE LATTICE MOD ell^2, by the shipped module route (hida::RunModuleCell,
//      K = 2: H^1(Gamma; Sym^n Z/ell^2) = (+)_i Z/ell^{a_i} with T_q for q in
//      S, U_ell first).  Since H^2(Gamma; -) = 0 for ell >= 5 (ell does not
//      divide 12), H^1(Z/ell^2) = H^1(Z_ell) / ell^2, so the m-part of the
//      module is the lattice H = H^1(Z_ell)_m mod ell^2 (with its torsion,
//      truncated at ell^2, on a Dickson-type m).  The m-part is cut out
//      WITHOUT the F_ell block: e_m = prod_q (1 - P_q), P_q the Fitting unit
//      projector (T_q - a~_q)^{N_ord} (hecke::ordinary::OrdinaryProjector),
//      a~_q the lift in [0, ell) of the block's eigenvalue; e_m is checked
//      idempotent and of rank dim_E mod ell.  `free` = e_m M free over
//      Z/ell^2 (non-Dickson-type: H^1(Z_ell)_m torsion-free); `torsion_gens`
//      the number of cyclic summands of order ell.
//      C-F9 is decided by X_q X_q' e_m == 0 mod ell entrywise, for every pair
//      q <= q' in S (a column of X_q X_q' e_m lies in ell M iff every
//      coordinate is divisible by ell; e_m M is a direct summand, so
//      ell (e_m M) = (e_m M) meet ell M).
//  (F) THE SAME TEST OVER F_ell on the record's block V (the driver's own
//      operators, re-matched exactly as HeckeCensusSplit does): the two routes
//      must agree -- by the UCT with H^2 = 0 the F_ell block IS H / ell H.
//  (D) THE DETECTOR: some single X_q e_m is non-zero mod ell exactly when the
//      row is non-reduced (m_A != 0); recorded as `m_nonzero`.
//  (W) THE SERRE-WEIGHT MECHANISM ([PR] Sec. 7).  The Glover factors
//      F(a, b) = Sym^a (x) det^b of Sym^n(F_ell^2) (GloverFactors, with
//      multiplicity) filter Sym^n GL_2(F_ell)-equivariantly; H^2 = 0 makes
//      H^1(Gamma; -) right exact, so H^1(Sym^n)_{m'} carries a filtration by
//      Hecke modules (T_q, q != ell -- U_ell does not preserve a GL_2(F_ell)
//      filtration) whose graded pieces are quotients of H^1(Gamma; F(a, b))_{m'},
//      where T_q acts as q^b T_q(Sym^a) (theta^b-twist, [E] identities item 6).
//      Here m' = m restricted to S \ {ell}.  Per row: `sw_count` = the number
//      of factors (with multiplicity) with H^1(F(a, b))_{m'} != 0 -- the
//      "Serre weights of rho-bar_m occurring in Sym^n", read cohomologically;
//      `sw_dim` = the sum of those dimensions; `sw_scalar` = every such piece is
//      scalar for T_q (q != ell).  The mechanism's consequences, checked:
//        dim H^1(Sym^n)_{m'} <= sw_dim                      (a theorem: right exactness)
//        LL_{A'}(V) <= sw_count, where A' = <T_q : q != ell> (when sw_scalar)
//      and the Loewy lengths of V, V^+, V^- (under A and under A') recorded.
//
// Everything here is F_ell / Z/ell^2 arithmetic on the pipeline's own objects;
// ell >= 5 is a precondition (ENGINE_CHECK), as is ell^2 < 2^32.
// ===========================================================================

#include "HeckeHidaTower.h"
#include "HeckeLocalAlgebra.h"   // ExtMatrix

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {

struct CensusLatticeRow {
    uint32_t ideal_id = 0;
    uint32_t dim_E = 0, dim_plus = 0, dim_minus = 0;
    std::vector<uint64_t> primes;       // S, ascending (the record's operator primes)
    std::vector<uint32_t> a;            // a_q mod ell, aligned with primes
    std::vector<uint32_t> witness_poly; // the record ideal's witness_poly (slot-16 WP-3: the census join)
    bool     eigen_ok = false;          // residue degree 1: every t_q - a_q nilpotent on V

    // (L)
    uint32_t lattice_rank = 0;          // rank_F_ell(e_m) = dim H / ell H
    uint32_t top_rank = 0;              // dim ell H (the order-ell^2 layer)
    bool     free_ = false;             // top_rank == lattice_rank
    uint32_t torsion_gens = 0;          // lattice_rank - top_rank
    bool     cf9_lattice = false;       // every pair product in ell H
    std::string first_fail;             // "q,q'" of the first failing pair, "" if none
    uint32_t pairs = 0;

    // (F), (D)
    bool     cf9_field = false;         // every pair product zero on V over F_ell
    bool     m_nonzero = false;         // some single X_q e_m != 0 mod ell

    // Loewy lengths over F_ell (0 = the zero space)
    uint32_t loewy_V = 0, loewy_Vplus = 0, loewy_Vminus = 0;         // under A (all of S)
    uint32_t loewy_V_off = 0, loewy_Vplus_off = 0, loewy_Vminus_off = 0; // under A' (S \ {ell})

    // (W)
    bool        sw_evaluated = false;
    std::string sw_why;                 // why not evaluated
    uint32_t    sw_count = 0, sw_dim = 0, sw_distinct = 0;   // with multiplicity; dimension; distinct (a, b)
    bool        sw_scalar = false;
    std::string sw_factors;             // "a:b:mult:dim;..." for the factors with dim > 0
    uint32_t    dim_mprime = 0;         // dim H^1(Sym^n)_{m'} over F_ell
    bool        mech_dim_ok = false;    // dim_mprime <= sw_dim
    bool        mech_loewy_ok = false;  // loewy_V_off <= sw_count (when sw_scalar)

    std::string status;                 // held | failed
    std::vector<std::string> failures;  // internal-consistency failures (not C-F9 failures)
};

struct CensusLatticeCell {
    uint32_t n = 0, ell = 0;
    uint32_t dim_H1 = 0;
    uint32_t module_gens = 0;           // generators of H^1(Z/ell^2)
    std::vector<CensusLatticeRow> rows; // H^1-own ideals, record order
    std::vector<std::string> failures;  // cell-level consistency failures
    double seconds = 0.0;
    double driver_s = 0.0, module_s = 0.0;   // the F_ell driver run; the Z/ell^2 module run
};

struct CensusLatticeOptions {
    uint32_t u_ell_budget_mb = 0;       // 0 = the u_ell setting in force
    bool     serre_weights = true;      // arm (W)
    // (slot-16 WP-3) join only: the F_ell driver, its blocks, the eigenvalues, the signs and the
    // Loewy lengths of V -- no Z/ell^2 module run, no (F)/(L)/(W); every row reads status
    // "join_only".  What HECKE_CENSUS_LATTICE_JOIN_ONLY=1 runs: a cheap pre-flight of the census join.
    bool     join_only = false;
};

// H^1(Gamma; Sym^a F_ell) with T_q (q in the list), cached per (a, ell) for (W).
class CensusWeightCache {
public:
    struct Entry {
        bool ok = false;
        std::string why;
        uint32_t dim = 0;
        std::map<uint64_t, MatrixT<uint32_t>> ops;   // q -> T_q on H^1(Sym^a)
    };
    const Entry& get(uint32_t a, uint32_t ell);
    // Pre-seed (a, ell) from an already-run cell (n < ell: Sym^n is its own factor).
    void seed(uint32_t a, uint32_t ell, const Entry& e) { cache_[{a, ell}] = e; }
private:
    std::map<std::pair<uint32_t, uint32_t>, Entry> cache_;
};

// The detector of (L): every product X_q X_q' e (q <= q') is 0 mod ell
// entrywise.  `first_fail` receives "i,j" (indices into X) of the first failing
// pair.  Public so that the validation can plant a Loewy-length-3 module.
bool PairProductsInEllM(const std::vector<hecke::ordinary::RingMat>& X, const hecke::ordinary::RingMat& e,
                        std::string* first_fail, uint32_t* pairs);

CensusLatticeCell ComputeCensusLattice(uint32_t n, uint32_t ell, const CensusLatticeOptions& opts,
                                       CensusWeightCache* weights);

// "CENSUS_LATTICE\t<n>\t<ell>\t<ideal_id>\t..." -- the columns of
// CensusLatticeHeader() after the tag.
std::string CensusLatticeHeader();
std::string CensusLatticeLine(const CensusLatticeCell& cell, const CensusLatticeRow& row);

} // namespace hecke_ext

#endif // HECKE_EXT_CENSUS_LATTICE_H
