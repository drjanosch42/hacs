// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_CHARPOLY_CRT_H
#define INTEGRAL_CHARPOLY_CRT_H

// ===========================================================================
// IntegralCharpolyCrt -- Line 2, milestone M-0 (hecke-extensions slot 8, WP-7):
// char_Z(T_p | Lambda^1) by CRT over existing F_ell runs of the shipped pipeline.
//
// [IB] = docs/implementation/integral_line_design_brief.md, Sec. 6.2 M-0,
// Sec. 5.2 ("certification against a pinned oracle needs no big integer"),
// Sec. 3.4 (the pinned oracle O5). Lambda^1 := H^1(SL_2(Z); Sym^n Z^2) / torsion,
// of rank 2 d_k - 1, k = n + 2, d_k = dim M_k ([C, cl:dimH1]).
//
// NOTHING NEW (plan Sec. "WP-7", [IB] OD-3 (c), OD-5 (a)):
//   * the F_ell runs are the shipped pipeline, called through its public API
//     exactly as `--bench hecke` calls it (relations -> EquivariantCohomologyOnK
//     -> HeckeCosetTables -> HeckeTauTable -> HeckeOperatorOnK); nothing in the
//     pipeline changes, so the F_ell path is untouched (M-0 item 4);
//   * no big integer, no type wider than __int128, no bignum library, no
//     reconstruction ([C, thm:backend](3), [C, obl:backend](iv) verbatim);
//     the output at a cell without an oracle is the RESIDUE VECTOR plus the
//     certified bound, which determines char_Z uniquely;
//   * the Deligne coefficient bound is carried as an UPPER BOUND ON BIT LENGTH,
//     computed in integer arithmetic (a rational upper bound a/b >= log2 p
//     taken from p^b <= 2^a in unsigned __int128);
//   * the pinned oracle O5 is compared DIGIT-WISE: every decimal root of O5 is
//     reduced mod each CRT prime by Horner's rule on its decimal digits, and the
//     reduced oracle polynomial is expanded over F_ell.
//
// THE CERTIFICATE (M-0 item 1). Let Q = char_Z(T_p | Lambda^1) and P = O5.
// At a good prime ell (ell > n + 1, ell != p; the torsion primes of H^1 are
// 2, 3 and ell <= n - 1, [IB] Prop. 5.1) the pipeline's H^1 over F_ell is
// Lambda^1 (x) F_ell and A^{(1)}(T_p) is T_p's reduction, so
// charpoly(A^{(1)}(T_p)) = Q mod ell. Every coefficient of Q satisfies
// |c| <= (2 + p^{k-1}) (1 + 2 p^{(k-1)/2})^{2 (d_k - 1)} < 2^L  (Deligne;
// eigenvalue 1 + p^{k-1} once, cusp eigenvalues |a_p| <= 2 p^{(k-1)/2} twice);
// every coefficient of P satisfies |c| < 2^{L_P} (computed from P's own roots).
// If P == Q mod ell for primes whose product M exceeds 2^{max(L, L_P) + 1},
// then |P - Q| < M coefficient-wise and P == Q mod M, hence P = Q over Z.
// The product is bounded below by 2^{sum (bitlen(ell) - 1)}, so the prime list
// is closed once that sum reaches max(L, L_P) + 2.
//
// Placement: src/hacs/Integral/ (Line 2's layer, plan
// shared-file rule: WP-7 only). Includes Hecke/ public headers and the
// top-level pipeline headers only; nothing from HeckeExt/.
// ===========================================================================

#include "../Hecke/FpPoly.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace integral {

// ---------------------------------------------------------------------------
// O5 -- the pinned oracle, FROZEN IN SOURCE. Transcribed from [IB] Sec. 3.4
// (the table "char(T_2 | Lambda^1)" and the n = 10 line for T_3), whose every
// row [IBA] Sec. 1 verified (docs/implementation/integral_line_design_brief_audit.md:
// "The oracle table (Sec. 3.4, O5) is right in every row, multiplicities
// included"). `probe_rank_h1` is the free rank of H^1 printed by
// docs/implementation/integral_line_probe.py (group and model rows agree),
// frozen here as a second, independent transcription; it must equal deg P.
// A disagreement anywhere is a FINDING, never a table edit ([IB] Sec. 6.4 K6).
// ---------------------------------------------------------------------------
struct O5Row {
    uint32_t n;                 // Sym^n; weight k = n + 2
    uint32_t p;                 // Hecke prime
    const char* verbatim;       // the [IB] Sec. 3.4 text of the row
    const char* eis_root;       // decimal Eisenstein root (= 1 + p^{k-1})
    const char* cusp_root;      // decimal cusp root a_p, nullptr if S_k = 0
    uint32_t cusp_mult;         // multiplicity of the cusp root (2, or 0)
    uint32_t probe_rank_h1;     // integral_line_probe.py: rank of H^1 over Z
};
const std::vector<O5Row>& O5Table();

// dim M_k for even k >= 0 (M_2 = 0).
uint32_t DimMk(uint32_t k);
// rank Lambda^1 = 2 d_k - 1 for even n >= 2 (O3, [C, cl:dimH1]); 0 for n = 0.
uint32_t RankLambda1(uint32_t n);

// Deligne bound on the coefficients of char_Z(T_p | Lambda^1), as bit lengths.
struct DeligneBitBound {
    uint32_t log2p_num = 0;     // log2 p <= num / den  (p^den <= 2^num, exact in u128)
    uint32_t log2p_den = 0;
    uint64_t bits_eis = 0;      // p^{k-1} <= 2^bits_eis
    uint64_t bits_sqrt = 0;     // p^{(k-1)/2} <= 2^bits_sqrt  (= ceil(bits_eis / 2))
    uint64_t bits = 0;          // every |coefficient| < 2^bits
};
DeligneBitBound CharpolyCoeffBitBound(uint32_t n, uint32_t p);

// Signed decimal string -> residue in [0, ell), by Horner's rule on its digits
// (word arithmetic only; no parse into a machine integer).
uint32_t ReduceDecimalMod(const std::string& dec, uint32_t ell);
// P = O5 row reduced digit-wise mod ell and expanded over F_ell.
hecke::FpPoly OracleCharpolyMod(const O5Row& row, uint32_t ell);
// |every coefficient of P| < 2^{result}: sum over the roots r (with
// multiplicity) of bitlen(1 + |r|), from the decimal digits.
uint64_t OracleCoeffBitBound(const O5Row& row);

// Deterministic primality (exact for 32-bit inputs).
bool IsPrimeU32(uint32_t x);
// The CRT prime sequence: primes descending from 2^30 (every one >> n + 1 and
// != 2, 3 for every n this layer is ever run at). Deterministic; the first
// `count` are returned.
std::vector<uint32_t> CrtPrimes(uint32_t count);

// The certificate prime list for a bound of `bits_needed`: the shortest prefix
// of `primes` with sum(bitlen(ell) - 1) >= bits_needed. Returns its length.
uint32_t CertificatePrefix(const std::vector<uint32_t>& primes, uint64_t bits_needed,
                           uint64_t* bits_covered = nullptr);

// One F_ell run of the shipped pipeline at (n, ell): H^1 once, then
// A^{(1)}(T_p) for each p in `ps`, and its charpoly (Berkowitz).
struct FellRun {
    uint32_t n = 0, ell = 0;
    std::vector<uint32_t> dims;              // dim H^q, q = 0, 1, 2
    std::map<uint32_t, hecke::FpPoly> charpoly;  // p -> charpoly(A^{(1)}(T_p)) mod ell
};
FellRun RunPipelineModEll(uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps);

// Oracle-free checks at one prime (M-0 item 3): deg = rank Lambda^1 (O3);
// (x - 1 - p^{k-1}) divides; the cofactor is a square (every cusp eigenvalue
// appears twice, Eichler-Shimura). Each returns true on success.
bool EisensteinFactorDivides(const hecke::FpPoly& chi, uint32_t n, uint32_t p, uint32_t ell,
                             hecke::FpPoly* cofactor_out = nullptr);
bool IsSquareMonic(const hecke::FpPoly& f);   // needs ell odd

} // namespace integral

#endif // INTEGRAL_CHARPOLY_CRT_H
