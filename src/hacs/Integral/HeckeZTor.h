// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_TOR_H
#define INTEGRAL_HECKEZ_TOR_H

// ===========================================================================
// HeckeZTor -- slot-25 package HZ-TOR: Tor terms, separation, splitting, the glue
// (docs/implementation/slot25/00-slot25-plan.md v0.2 Sec. 3, wave 3; the card "HZ-TOR" of
// docs/implementation/slot24/heckez/06-work-packages.md; the mathematics in 03-bockstein-tor.md v0.3, B2, B7, B8, B11-B14;
// the prototype s03/ext2.py).  Contract labels of the HeckeZ section, which enters [C] as 1.25.0-pre.3 on a branch not yet in
// hecke-extensions, are cited in the "(Z0 `label`)" form; existing [C] labels in the bracket form "C, <label>" in square brackets.
//
// THE OBJECT.  For a cell (N, n), a prime ell and the read set P (HZ-LOC's ReadPrimes: P plus T_ell when ell !| N, ell <= 23170;
// the slot-25 amendment of 04 Sec. 1.5), the export (Integral/Descent/DescentModuleExport.h) gives H^1 (x) Z/ell^K with the FULL
// matrices of T_p (torsion first, then free; column convention; torsion rows mod ell^{sigma_i}):
//
//     A_p = [[ C_p, beta_p ], [ 0, B_p ]]      C_p on tors_ell, B_p on L / ell^K, beta_p : L / ell^K -> tors_ell (the glue)
//
// The torsion generators are the export's torsion cocycles, so tors_ell inside H^1 / ell^K is CANONICAL (03 B13, implementation
// note: the 25 candidate lines of ext2.py are not needed).  Everything below is read at K = e := sigma_max (B12(1): the class
// splits iff it splits modulo ell^K for any K >= e), except the J' read (K_J >= e).
//
// WHAT IS READ, per (cell, ell):
//  (B2)   the universal-coefficient types are DERIVED, not stored: dim H^q(Gamma; V / ell) = b_q + t_q + t_{q+1} (UctPredict; the
//         validation compares it with the Lean-kernel tables, K5).
//  (blocks) the joint T'-blocks of H^1 / ell^e, T' = Z[T_p : p in P]: per operator the certified factorisation of charpoly(A_p mod
//         ell) (hecke::FactorPoly), the idempotents lifted to Z/ell^e by Newton (3E^2 - 2E^3), refined over P.  Each block m has
//         r_m = rank L_m and t_m = the number of cyclic summands of tors_m, and its residual system (pi_p per p).
//  (K6)   separation per operator: gcd(chi(T_p | L) mod ell, chi(T_p | tors) mod ell) = 1 in F_ell[x] with a Bezout pair (u, v),
//         u chi_L + v chi_T = 1, checked by multiplication (03 B7, the separation certificate; chi(T_p | tors) mod ell = prod_j P_j,
//         [C, prop:torsionhecke](1)(a), by the Grothendieck-group identity [G / ell] = [G[ell]]).  A block with r_m = 0 or t_m = 0
//         is SEPARATED (B7: disjoint T'-supports are disjoint T-supports): split, J_m = 0, certified for all of T.
//  (B12)  per common block (r_m, t_m > 0): the split test h B'_p - C_p h = beta'_p (p in P) for h in Hom(L_m / ell^e, tors_ell),
//         solved by a Smith form with transforms over Z/ell^e (row j of the unknowns mod ell^{sigma_j}).  Solvable: `split(T')`
//         (split for T only under generation, B12(2)), with the section h (checked by substitution).  Unsolvable: `non_split`,
//         certified for T, with a dual witness y (y M = 0, y . rhs != 0; the Z/ell^e Farkas certificate of K7), checked by
//         substitution; and B14: the block is a maximal ideal common to Spec T^fr and Spec T^tors.
//  (B13)  the J bounds: |J_m| <= |Hom_{T'}(L_m / ell^e, tors_m)| (the kernel of the same system); J_m = `zero` when separated or
//         when that bound is 1, else `bounds` (never `exact`: CB-3, 03 Sec. 0.1).  And per ell the computed glue of the image
//         A'_{K_J} of T' on H^1 / ell^{K_J}: j'_{K_J} = |ker(A'_{K_J} -> End(L / ell^{K_J}) x End(tors))| -- an UPPER bound for
//         |J cap T'| (K_J >= e; j' = 1 certifies J cap T' = 0), a lower bound for J only for K_J >= e + c^sat (B13), so it never
//         makes J exact here.
//  (B8)   level 1, ell in {2, 3}: every T_p, p != ell, is 1 + p on H^1 / ell ([C, prop:ltwothreescalar]) -- checked on the export's
//         matrices -- so the blocks are cut by T_ell alone and a common block is the maximal ideal m_x of T: `coincidence =
//         certified(B8)`.  Elsewhere a common block is a T'-block: `certified(B14)` when non-split, else `not_certified(T'_block)`.
//  (K7)   the cap: a block is admitted to the Lean checker iff r_m <= 6 and dim tors_m / ell <= 6, else `lean = absent(cap)`.
//  theory = H (group cohomology); H_c carries groups only (`hecke = not_run`, 03 Sec. 2.7, F04-7).
// Refusals: `blocked(export)` (the export refused; its line), `open(precision)` (K_max < e: never at the export's widths).
// No settings key, no env variable, no record key, no golden object; the descent, EXPORT, EXPORT-Z, HZ-LOC code is not edited.
//
// INTERFACE FOR EXPORT-Z STEP 2.  SplitFromModuleZ(Z, l) runs the B12 solver on EXPORT-Z's per-ell blocks (tau's ell-block,
// M_block, F_local mod ell^{sigma_max}) and returns the per-ell verdict; ModuleZSplitWord folds it into the one word EXPORT-Z's
// `split` field carries ("split(separated)" | "split(T')" | "non_split" | "no_torsion" | "blocked(export)").  The verdict does not
// depend on the basis of L (it is a property of the extension class), so step 2's global basis does not change it -- but
// FromModuleZ takes M and F from ModuleZ::free_view (one basis-tagged accessor: local in step 1, global in step 2; audit M1, EXZ2-P3).
// Audit N2: ModuleZSplitWord runs the ell's sequentially (step 2 should call SplitFromModuleZ inside its per-prime lanes);
// SplitFromModuleZ reads with Z.ps, does NOT append T_ell, and reads j' at K_J = sigma_max, so its word can be weaker than
// ReadTor's (split(T') where ReadTor says split(separated)); each is a true statement about its own T'.
// Audit N3: the J' read uses dense D^2-vector spans, timed only up to D = 28; HZ-RUN must time one large unit first.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "HeckeZLocal.h"

namespace integral_descent {
namespace modexport {
struct ModuleAtK;
struct ModuleZ;
}  // namespace modexport
}  // namespace integral_descent

namespace heckez {
namespace tor {

using heckez::local::Mat;
using heckez::local::Zlk;

// ---- the input: the extension 0 -> tors_ell -> H^1 / ell^K -> L / ell^K -> 0 with the T_p ---------------------------------------
struct ExtensionInput {
    uint32_t N = 1, n = 0;
    uint64_t ell = 0;
    uint32_t K = 0;                       // the ring Z/ell^K of the matrices (K >= sigma_max)
    std::vector<uint32_t> sigma;          // the torsion exponents (generators 0 .. t-1), each <= K
    uint32_t r = 0;                       // rank L
    std::vector<uint32_t> ps;
    std::vector<Mat> A;                   // per p: (t + r) x (t + r), column convention; row i mod ell^{exp_i}
    uint32_t t() const { return static_cast<uint32_t>(sigma.size()); }
    uint32_t D() const { return t() + r; }
    uint32_t e() const;                   // sigma_max (0: no torsion)
    uint32_t exp(uint32_t i) const { return i < t() ? sigma[i] : K; }
};
// From the export's module at K (ModuleAtK of DescentModuleExport.h).
ExtensionInput FromModuleAtK(uint32_t N, uint32_t n, const integral_descent::modexport::ModuleAtK& M);
// From EXPORT-Z step 1, local part l (tau's ell-block, M_block, F_local at K = sigma_max(ell)).
ExtensionInput FromModuleZ(const integral_descent::modexport::ModuleZ& Z, size_t l);
// The same extension at K' <= K (rows reduced).
ExtensionInput Reduce(const ExtensionInput& X, uint32_t K);

// ---- linear algebra over Z/ell^E --------------------------------------------------------------------------------------------------
// The system M x = b over Z/ell^E (M: rows x cols, row-major).  Smith form with transforms U M V = diag(ell^{s_i}).
struct LinSolve {
    bool solvable = false;
    std::vector<uint64_t> x;              // a solution (solvable)
    std::vector<uint64_t> y;              // a dual witness (unsolvable): y M = 0, y . b != 0
    uint32_t kernel_length = 0;           // log_ell |ker M| (in (Z/ell^E)^cols)
};
LinSolve SolveLinear(const Zlk& R, const std::vector<uint64_t>& M, size_t rows, size_t cols, const std::vector<uint64_t>& b);

// ---- K6: the separation certificate per operator ---------------------------------------------------------------------------------
struct Separation {
    uint32_t p = 0;
    std::vector<uint64_t> chi_L, chi_T;   // mod ell, low -> high (monic)
    std::vector<uint64_t> gcd;            // monic gcd (1: separated at T_p)
    std::vector<uint64_t> u, v;           // u chi_L + v chi_T = 1 (separated) -- checked
    bool separated = false;
    bool bezout_checked = false;
};

// ---- the per-block read ---------------------------------------------------------------------------------------------------------
struct BlockRead {
    std::vector<std::vector<uint64_t>> residue;   // per p: the monic irreducible pi_p mod ell of the block (low -> high)
    uint32_t r_m = 0, t_m = 0;            // rank L_m; dim_F_ell tors_m / ell
    uint32_t len_tors = 0;                // log_ell |tors_m|
    std::string kind;                     // "common" | "free_only" | "torsion_only"
    std::string split;                    // "split(separated)" | "split(T')" | "non_split"
    bool section_checked = false;         // h B' - C h = beta' verified (split(T'))
    bool witness_checked = false;         // y M = 0, y . beta' != 0 verified (non_split)
    uint32_t hom_length = 0;              // log_ell |Hom_T'(L_m / ell^e, tors_m)|
    std::string J;                        // "zero" | "bounds(<=ell^b)"
    std::string coincidence;              // "none(separated)" | "certified(B8)" | "certified(B14)" | "not_certified(T'_block)"
    std::string lean;                     // "in_cap" | "absent(cap)"
    std::vector<uint64_t> h;              // the section (split(T')): t x r_m row-major (row j mod ell^{sigma_j}); block basis
    std::vector<uint64_t> y;              // the witness (non_split)
    std::string text() const;
};

struct TorRead {
    uint32_t N = 1, n = 0;
    uint64_t ell = 0;
    std::vector<uint32_t> P;              // the read set (ReadPrimes)
    std::string t_ell;                    // ReadPrimes' word
    uint32_t K = 0, K_max = 0, e = 0, r = 0;
    std::vector<uint32_t> sigma;
    std::string status;                   // "ok" | "no_torsion" | "blocked(export)" | "open(precision)"
    std::string detail;
    std::string theory = "H";             // H_c: groups only (hecke = not_run)
    std::vector<Separation> sep;          // K6, per p in P
    bool b8_scalar = false;               // level 1, ell in {2, 3}: T_p == 1 + p on H^1 / ell for every p != ell (checked)
    bool b8_applies = false;
    std::vector<BlockRead> blocks;        // sorted by the residue text
    std::string split;                    // per ell: "split(separated)" | "split(T')" | "non_split" | "no_torsion" | ...
    uint32_t K_J = 0;                     // the K of the J' read
    int32_t jprime_length = -1;           // log_ell j'_{K_J}; -1: not read
    uint32_t algebra_length = 0;          // log_ell |A'_{K_J}|
    uint32_t exact_J = 0;                 // must stay 0 (a tripwire: no code path writes "exact", so 0 holds by construction; audit N1)
    double secs = 0;
    std::string text() const;             // canonical (digest input; no timings)
    uint64_t digest() const;
};

struct TorOptions {
    uint32_t K_J_extra = 4;               // K_J = min(K_max, e + K_J_extra)
    bool jprime = true;                   // run the J' read
    uint32_t K_read = 0;                  // 0: the split / Hom solve at K = e; else at min(K_read, K) >= e (B12(1): same verdict)
};

// The read on a given extension (the solver; EXPORT-Z's entry goes through here too).
TorRead ReadExtension(const ExtensionInput& X, const TorOptions& opt = {}, const ExtensionInput* XJ = nullptr);
// The cell driver: the export at (N, n, ell) with ReadPrimes(N, ell, P), then ReadExtension.
TorRead ReadTor(uint32_t N, uint32_t n, uint64_t ell, const std::vector<uint32_t>& P, const TorOptions& opt = {});

// (B2) predicted dim_F_ell H^q(Gamma; V / ell), q = 0 .. 3, from the integral groups b_q, t_q(ell) (q = 0 .. 3, t_4 := t_2:
// 2-periodicity from degree 2, [C, thm:descentcert](6)).
std::vector<uint32_t> UctPredict(const std::vector<uint32_t>& b, const std::vector<uint32_t>& t);

// ---- the EXPORT-Z step-2 interface -----------------------------------------------------------------------------------------------
TorRead SplitFromModuleZ(const integral_descent::modexport::ModuleZ& Z, size_t l, const TorOptions& opt = {});
// The one word of ModuleZ::split over every torsion prime: "non_split" if some ell is; else "split(T')" if some ell is; else
// "split(separated)"; "no_torsion" when T = 0; "blocked(export)" when Z is.
std::string ModuleZSplitWord(const integral_descent::modexport::ModuleZ& Z, const TorOptions& opt = {});

}  // namespace tor
}  // namespace heckez

#endif
