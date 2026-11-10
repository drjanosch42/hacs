// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_SATURATION_DEFECT_H
#define HECKE_EXT_SATURATION_DEFECT_H

// ===========================================================================
// (hecke-extensions slot 16, WP-5) THE SATURATION-DEFECT TOR TERM, per block.
// Note and predictions: docs/implementation/slot16/WP-5_tor_note.md (committed
// before this code).  Owner-approved package of 2026-10-01 (the WP-17 carry list).
//
// For a torsion-free block L = Lambda_m (Lambda = H^1(Gamma; Sym^n Z_ell), [C,
// lem:freepart]: torsion lives only on the Dickson locus) and T = T'_{m'}, the
// algebra of the T_p, p in S (default {2, 3, 5, 7, ell}), at its joint F_ell-
// rational system ([C, cor:stablecert]):
//
//   I_m := Tor_1^{Z_ell}(T^sat / T, F_ell) = (T^sat / T)[ell]
//        ~= (T cap ell T^sat) / ell T = ker(T / ell -> End(L / ell)),
//
// an ideal of T / ell with (T / ell) / I_m ~= A_1, the census algebra ([C,
// def:satdata], [C, prop:saturation](1)-(3), [C, prop:satdefect]).  At every
// STABLE K (dim A_K / ell = rank_T = ceil(f / 2), certified by [C,
// cor:stablecert](1)) T / ell ~= A_K / ell A_K, and I_m is the kernel of
// A_K / ell A_K -> End(B_K / ell) -- what this unit computes (route R-ker), in
// the regular representation of A_K / ell A_K (the cyclic decomposition of
// A_K inside End(B_K) by engine::elim, the one elimination).  Route R-len is
// [C, prop:satdefect](1) at K = 1: dim I_m = eps_1 = rank_T - dim A_1.  The
// two are the two sides of the Tor identity and are compared per block.
//
// Per block it reports: dim I; minimal generators as polynomials in
// x_q = T_q - a_q (F_ell coefficients, in a monomial basis of T / ell chosen
// greedily in degree order); the radical layers dim m^j I / m^{j+1} I (the
// first is the number of minimal generators), the Loewy length of I, dim soc(I),
// dim Ann(I), principal?; length(T^sat / T) = delta^sat and its ell^j-
// filtration eps_j = length (T^sat / T)[ell^j] = j rank_T - length(A_j) (a
// plateau is final, [C, prop:satdefect](2)), hence the invariant factors of
// T^sat / T when final.  Optionally the forms-side cross-route over F_ell (the
// joint algebra of HeckeFormsHeckeAlgebraValidation's comparison): dim ker(J ->
// A_1) <= dim I_m, with equality when the S-primes generate T^f_m / ell.
//
// Refused by name, never computed: torsion blocks ("separate treatment": [C]
// asserts no stopping rule for T^coh_m there, cor:stablecert(3)); blocks not
// stable by KMAX; ell^K >= 2^64 (native uint64_t module route only).
//
// Pipeline side only, console-free; writes nothing, emits no record and no
// golden object.  Lines: SAT_DEFECT (one per block).
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "HeckeRingHeckeAlgebra.h"

namespace hecke_ext {
namespace satdefect {

// The F_ell structure of A_K / ell A_K and of the kernel I of its reduction map,
// read on ONE module B = (Z/ell^K)^D (every exponent K >= 2) with commuting ops,
// at the maximal ideal m = (ell, T_p - a_p).  Not a stability statement: the
// caller decides whether A_K / ell A_K is T / ell (dim_A == rank T).
struct DefectReading {
    uint32_t K = 0;
    uint32_t dim_A = 0;                  // dim A_K / ell A_K (the number of cyclic summands of A_K)
    uint32_t length_A = 0;               // length A_K (the sum of the summand exponents)
    std::vector<uint32_t> A_exponents;   // the cyclic summands of A_K, descending
    std::vector<uint32_t> hilbert;       // of A_K / ell A_K at m
    uint32_t dim_image = 0;              // dim of the image in End(B / ell) (= dim A_1 when stable)
    std::vector<uint32_t> quotient_hilbert;  // of (A_K / ell A_K) / I at m
    uint32_t dim_I = 0;
    std::vector<uint32_t> I_layers;      // dim m^j I / m^{j+1} I, j = 0, 1, ...
    uint32_t I_loewy = 0;                // the number of non-zero layers (0 iff I = 0)
    uint32_t I_socle = 0;                // dim {w in I : m w = 0}
    uint32_t I_ann = 0;                  // dim Ann_{A/ell}(I)
    uint32_t min_generators = 0;         // = I_layers[0] (0 iff I = 0)
    bool principal = true;               // min_generators <= 1
    bool I_is_m = false;                 // I == m (the hidden case)
    bool I_in_m = true;                  // I subset of m (always: I is proper)
    std::vector<std::string> generators; // polynomials in (T_p - a_p), one per minimal generator
    std::vector<std::string> monomial_basis;  // the chosen monomials, degree order
};
DefectReading ReadDefect(const ringalg::OpModule& B, const std::vector<uint64_t>& system);

struct SatOptions {
    uint32_t kmax = 6;                   // the largest precision read (KMAX)
    std::vector<uint64_t> primes;        // S; empty = {2, 3, 5, 7, ell}
    bool certify = true;                 // ringalg::CertifyStable on every stable / reduced_rank1 verdict
    bool forms = true;                   // the forms-side cross-route over F_ell (run at k = n + 2 >= 12 only)
};

struct FormsCross {
    bool ran = false;
    uint32_t dim_V = 0;                  // dim M_k(F_ell)_m
    uint32_t T_dim = 0;                  // dim T^f_m / ell (all generator primes)
    uint32_t Tc = 0;                     // dim of the algebra of the S-primes on V_m
    uint32_t joint = 0;                  // dim of the joint algebra on V_m (+) B / ell
    bool M_gen = false;                  // the generator primes give dim M_k (cl:mkfaithful)
    bool S_gen = false;                  // Tc == T_dim: the S-primes generate T^f_m / ell
    uint32_t kernel = 0;                 // joint - dim A_1
};

struct SatBlock {
    std::vector<uint64_t> system;        // aligned with SatCell::primes
    uint32_t D = 0, free_rank = 0, rank_T = 0;
    std::vector<uint32_t> torsion;
    std::string status;                  // ok | refused:torsion_block | refused:not_stable_at_KMAX
    std::string verdict;                 // ringalg's: reduced_rank1 | stable | (refused) torsion_* / not_stable_at_KMAX
    std::string stable_cert = "none";    // certified | reading | none
    uint32_t cert_from_cls = 0, cert_from_K = 0;   // (slot-23 G1, [C, obl:precisionreuse](vii)) the run levels the certificate's modules came from
    uint32_t K_stab = 0;
    std::vector<uint32_t> A1_hilbert;
    uint32_t dim_A1 = 0;
    DefectReading at;                    // the reading at K_stab (ok, stable blocks)
    uint32_t eps1_len = 0;               // R-len: rank_T - dim A_1
    std::vector<uint32_t> eps;           // eps_1, eps_2, ... (one per level read)
    uint32_t sat_length = 0;             // delta^sat when final, else the last eps (a lower bound)
    bool sat_final = false;
    std::vector<uint32_t> sat_structure; // the invariant factors of T^sat / T (exponents, descending), when final
    std::vector<uint32_t> ak_len;        // length(A_K), K = 2, 3, ... (the route's readings)
    FormsCross forms;
};

struct SatCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t kmax = 0;
    std::vector<uint64_t> primes;
    uint32_t e_max = 0, K_cls = 0;
    std::vector<SatBlock> blocks;
    // (slot-23 G1, [C, obl:precisionreuse](vi)) K_run = Lv_read, THE LEVELS READ (the per-K route's list, byte-stable; through
    // v1.23.0-pre.2 every level read was also run); Lv_run = the levels at which the module route was RUN; module_seconds
    // counts the runs only.  ringalg::LevelCache derives the levels below the latest run ([C, lem:precisionreduction]).
    std::vector<uint32_t> K_run;
    std::vector<uint32_t> Lv_run;
    std::string level_mode;              // "reuse:<policy>" | "per_K:<reason>"
    double seconds = 0.0, module_seconds = 0.0;
    // (S24 SATPERF) wall seconds per post-module stage: prepass, systems, localize, cert, forms (a level run triggered
    // inside a stage is counted there too; module_seconds says how much).  Diagnostic only: printed by the validation's
    // SAT_DEFECT_TIMES line, never part of a SAT_DEFECT line or a record.
    std::map<std::string, double> stage_seconds;
};

// n >= 2 even, ell >= 5 prime, ell^kmax < 2^64 (else hecke::NotImplementedInThisBuild).
SatCell ComputeSaturationDefectCell(uint32_t n, uint64_t ell, const SatOptions& opts);

// The invariant factors (exponents, descending) of a finite Z_ell-module G from its
// filtration eps_j = length G[ell^j], j = 1 .. J, with G = G[ell^J] (the caller passes the
// filtration up to the plateau, the repeated value dropped).  Throws on a sequence that
// is not such a filtration.
std::vector<uint32_t> SaturationStructure(const std::vector<uint32_t>& eps);

// The census columns of one block: sat_defect_dim, sat_defect_loewy, sat_length
// ("not_computed:<reason>" on a refused block; sat_length ">=x" when not final).
struct CensusColumns {
    std::string sat_defect_dim, sat_defect_loewy, sat_length;
};
CensusColumns Columns(const SatBlock& b);

// "SAT_DEFECT n=.. ell=.. primes={..} system=(..) D=.. status=.. dim_I=.. ..."
std::string SatDefectLine(const SatCell& c, const SatBlock& b);

} // namespace satdefect
} // namespace hecke_ext

#endif // HECKE_EXT_SATURATION_DEFECT_H
