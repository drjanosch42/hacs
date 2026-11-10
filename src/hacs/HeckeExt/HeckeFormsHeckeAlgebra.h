// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_FORMS_HECKE_ALGEBRA_H
#define HECKE_EXT_FORMS_HECKE_ALGEBRA_H

// ===========================================================================
// hecke-extensions slot 12, WP-7 -- the FORMS-SIDE Hecke algebra T_m / ell of
// S_k(F_ell)_m at level one, from q-expansions.  Pipeline-free.
//
// WHY.  The slot-11 WP-20 referee (docs/implementation/slot11/WP-20_referee.md
// M-1, Sec. 7 item 1; [EX]'s \RCOMP rows) computed OUT OF TREE that the
// cohomological local algebra A of the census is a PROPER QUOTIENT of the
// forms-side T_m / ell at (n, ell) = (82, 7) (forms: F_7[x,y,z]/(x,y,z)^2,
// embedding dimension 3; A: embedding dimension 2) and at (142, 7) (forms:
// dim 6, Hilbert (1, 3, 2), Loewy length 3; A = T_m / (7, m^2)).  This unit
// computes the forms side in tree so that the datum is pipeline data.
//
// THE ROUTE (exact over F_ell, no search, no randomness).
//  * k = n + 2, k = 12 d + k', k' in {0, 4, 6, 8, 10, 14} (k' = 14 iff
//    k == 2 mod 12), d = dim S_k.  The d forms
//        f_j = Delta^j E_6^{2(d - j)} E_{k'},   j = 1 .. d,
//    (E_0 = 1, E_8 = E_4^2, E_10 = E_4 E_6, E_14 = E_4^2 E_6) have integral
//    q-expansions with leading term q^j (coefficient 1), so they are a
//    Z-basis of S_k(Z) (Miller: a cusp form of weight k is determined by
//    a_1 .. a_d) and their reductions a basis of S_k(F_ell) = S_k(Z) (x) F_ell.
//    Every series is computed mod ell to a caller-given precision N.
//  * T_p on q-expansions: a_n(T_p f) = a_{np} + p^{k-1} a_{n/p}; the image is
//    read in the basis from a_1 .. a_d (unitriangular) and CHECKED against
//    every coefficient up to the precision the input supports
//    (a_n(T_p f) is known for n p < N): an operator whose image is not in
//    S_k to that precision throws.  N > d p is required (ENGINE_CHECK).
//  * The algebra is HeckeExt/HeckeAlgebraInvariants' ComputeAlgebraInvariants
//    (the certificate-free route: rad = ker Frob^e, Berlekamp, Hilbert of
//    gr_m); the block at a maximal ideal is the joint generalised eigenspace
//    of an F_ell-rational system (lambda_p), V_m = ker stack_p (T_p - lambda_p)^d,
//    and T_m / ell is F_ell[T_p |_{V_m}] -- faithful on V_m because T / ell
//    acts faithfully on S_k(F_ell) (q-expansion duality: the pairing
//    (t, f) -> a_1(t f) is perfect over Z at level one) and V_m is the
//    m-primary component.  Since slot-15 WP-19 this is PROVED for M_k(F_ell)
//    as well ([E] cl:mkfaithful, ell >= 5, including (ell-1) | k, where
//    T(M_k) = F_ell x T(S_k)): dim T(X) = dim X and dim T(X)_m = dim X_m for
//    X = S_k, M_k -- checked per block by the validation (arms (M), (T)).
//  * Generators (slot-15 WP-19 (d)): the validation uses the primes <= max(31, k/12)
//    and ell; they generate T(S_k)/ell by the Sturm bound ([E] cl:sturm (a)) and
//    T(M_k)/ell iff the algebra has dimension dim M_k ([E] cl:mkfaithful), checked
//    per cell (M_gen).  wpm1/forms_local.py's set {2,3,5,7,ell} + primes <= k/12
//    is a subset, so no alignment is needed.
//  * F9 (dim V = dim T(S_k)_m + dim T(M_k)_m on an H^1 row) holds exactly where the
//    block has no Dickson torsion, h = dim H^0(Gamma; M_n)_lambda = 0 ([E]
//    cl:formsF9: D = s + m + h, h by [C] prop:heckeH0); the validation gates F9
//    there and prints `F9_SKIP torsion` where h > 0.
//  * Notation: T_m/ell is T^f_m/ell = T^Lambda_m/ell ([C] prop:cohforms (3)); the census
//    algebra A on H^1(Gamma; M_n)_m is a different quotient of T^coh/ell, a quotient
//    of T_m/ell on torsion-free blocks (prop:cohforms (5)).
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t>, single-threaded, no hash
// container; it consumes Hecke/SubspaceOps and HeckeAlgebraInvariants only.
// ===========================================================================

#include "HeckeAlgebraInvariants.h"
#include "../../ModArithmetic/EnvLever.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {
namespace forms {

// A q-series mod ell, coefficients a_0 .. a_{N-1}.
using QSeries = std::vector<uint32_t>;

// The level-one cusp space S_k(F_ell) to precision N.
struct CuspSpace {
    uint32_t k = 0, ell = 0, precision = 0;
    uint32_t d = 0;                     // dim S_k
    uint32_t k_prime = 0;               // the E_{k'} factor
    std::vector<QSeries> basis;         // f_1 .. f_d, leading q^j (f_0 .. f_d on M_k)
    bool with_constant = false;         // (slot-12 WP-12) true: the space is M_k, dim d + 1
};

// ---------------------------------------------------------------------------
// (slot-23 G2-X, docs/implementation/slot23/G2_plan.md Sec. 3, G2X_report.md) The truncated q-series product
// c = a * b mod ell, c of length N (c_k = sum_{i+j=k, i<|a|, j<|b|} a_i b_j mod ell, k < N), inputs reduced mod ell.
// The value is unique, so every route below returns the same bytes:
//   * MulSeriesNaive  -- the O(N^2) product through slot 23 (one % per term): the oracle and the rollback arm;
//   * MulSeriesSchoolbook -- the same sum, the reduction delayed into a uint64_t accumulator (ell < 2^16);
//   * MulSeriesFast   -- exact integer convolution by NTT over two NTT primes P1 = 5*2^25+1, P2 = 7*2^26+1 and CRT
//     (exact while min(|a|,|b|) (ell-1)^2 < P1 P2 and |a|+|b|-1 <= 2^25), the schoolbook below
//     kFastMulSchoolbookMin or N < kFastMulNttMinN; ell outside [2, 2^16): the naive route.  The transform-length
//     limit: |a|+|b|-1 <= 2^25 (the 2-adic order of P1 - 1), i.e. N <= 2^24 + 1 for balanced products (the sat route's
//     N = 2 (k/12+1) pmax + 2 reaches it at (k/12+1) ell ~ 8.4e6).  Past it (G2-X audit M1) the product is not
//     quadratic: min(|a|,|b|) <= kFastMulChunkSchoolbookMax takes the schoolbook, anything else MulSeriesChunked;
//   * MulSeriesChunked -- the product split into blocks of length 2^chunk_lg (chunk_lg <= 24), each block product by
//     MulSeriesFast within both limits, accumulated mod ell (chunk_lg is a parameter only so the validation can test
//     the block assembly at small sizes; the production route uses 24).
//   * MulSeries       -- what the forms route calls: MulSeriesFast unless HECKE_FORMS_FASTMUL=0 (or a scope says
//     naive).  The same switch selects, in EisensteinSeries, a divisor sieve (O(N log N)) over the O(N^2) trial
//     division -- the same uint64_t sums, so the same bytes.
// HECKE_FORMS_FASTMUL is read-only in the env-hook sense: it cannot change a value (gated by
// HeckeFormsFastMulValidation).  Unset / "1": fast; "0": naive; anything else warns and stays fast.  When it is set
// the first product of the process prints one `EFFECTIVE HECKE_FORMS_FASTMUL mul=fast|naive (...)` line.
constexpr size_t kFastMulSchoolbookMin = 64;     // min(|a|, |b|) below it: the schoolbook
constexpr size_t kFastMulNttMinN = 1024;         // N below it: the schoolbook (the NTT does not pay at small N)
constexpr size_t kFastMulChunkSchoolbookMax = 4096;   // past the length limit: min(|a|, |b|) up to it takes the schoolbook
QSeries MulSeries(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N);
QSeries MulSeriesNaive(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N);
QSeries MulSeriesSchoolbook(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N);
QSeries MulSeriesFast(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N);
QSeries MulSeriesChunked(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N, uint32_t chunk_lg);
// (S25 LEVERS-25) The env arm HECKE_FORMS_FASTMUL=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool FormsFastMulEnabled();
struct FormsFastMulCounters {
    uint64_t ntt = 0, schoolbook = 0, naive = 0, fallback = 0;   // products by route (process totals)
    uint64_t chunked = 0;                                        // MulSeriesChunked calls (its blocks count as ntt / schoolbook)
    uint64_t sieve = 0, divisor_loop = 0;                        // EisensteinSeries by route
};
FormsFastMulCounters FormsFastMulCountersNow();
// The full line with source=default|env|scoped and the counters (the validation prints it).
std::string FormsFastMulEffectiveLine();
class ScopedFormsFastMul {   // in-process A/B: true = fast, false = naive; restores on destruction
public:
    explicit ScopedFormsFastMul(bool on);
    ~ScopedFormsFastMul() = default;
    ScopedFormsFastMul(const ScopedFormsFastMul&) = delete;
    ScopedFormsFastMul& operator=(const ScopedFormsFastMul&) = delete;
private:
    engine::lever::EnvLever::Scoped guard_;
};

// ---------------------------------------------------------------------------
// (slot-24 FNTT, docs/implementation/slot24/FNTT_report.md; G2X_report.md Sec. 5 items 1-2) The FNTT route of
// BuildCuspSpace / BuildModularSpace: the same basis series f_j = Delta^j E_6^{2(d-j)} E_{k'} (and f_0 = E_6^{2d} E_{k'}),
// computed in the transform domain of the G2-X NTT (P1, P2, CRT) with
//   * one NTT plan per space (no per-product plan builds above 2^20);
//   * the repeated multiplicands' spectra cached: F(E_6^2) for the chain G_i = E_{k'} E_6^{2i}, F(Delta) for the
//     chain Delta^j, and F(Delta^j) shared between f_j = Delta^j G_{d-j} and Delta^{j+1} (6 transforms per j, not 12);
//   * BuildModularSpace computing S_k and f_0 in one pass (f_0 = G_d, a link of the same chain: no second S_k);
//   * Dpow / Epow not retained: the G_i are written into the basis slots and multiplied by Delta^j in place, so the
//     live series are the basis plus O(1) spectra (forms RSS / 2 at ell >= 10^4);
//   * cache-blocked radix-2 butterflies, AVX2 (runtime dispatch, scalar fallback) -- the same Shoup butterflies;
//   * HeckeOnCusp's elimination with the reduction mod ell delayed into the uint64_t accumulator (ell < 2^16).
// Every value is the product mod (ell, q^N) in the commutative ring F_ell[q]/(q^N), which is unique: no lever can change
// a byte (gated by HeckeFormsFastMulValidation arms (N*)).  The route applies when the fast product is on, N >= 1024,
// and the G2-X exactness bounds hold for two length-N operands (2N - 1 <= 2^25, N (ell-1)^2 < P1 P2); otherwise the
// legacy (G2-X) code runs, verbatim.
// Rollback lever HECKE_FORMS_NTT_CACHE (read-only, cannot change a value): unset / "1": the FNTT route; exactly "0": the
// G2-X route (and its HeckeOnCusp loop) verbatim; anything else warns and stays on.  When it is set, the first space
// built in the process prints one `EFFECTIVE HECKE_FORMS_NTT_CACHE route=fntt|legacy simd=avx2|scalar (...)` line.
bool FormsNttCacheEnabled();
bool FormsNttSimdAvailable();   // the CPU has AVX2 (the FNTT butterflies then take it unless a scope says scalar)
struct FormsNttCounters {
    uint64_t fntt_spaces = 0, legacy_spaces = 0;          // BuildCuspSpace / BuildModularSpace calls by route
    uint64_t forward = 0, inverse = 0;                    // FNTT transforms (each over both primes)
    uint64_t simd_transforms = 0, scalar_transforms = 0;  // ... by butterfly kernel
    uint64_t hoc_delayed = 0, hoc_legacy = 0;             // HeckeOnCusp by elimination loop
};
FormsNttCounters FormsNttCountersNow();
std::string FormsNttEffectiveLine();
class ScopedFormsNttCache {   // in-process A/B: true = the FNTT route, false = the G2-X route; restores on destruction
public:
    explicit ScopedFormsNttCache(bool on);
    ~ScopedFormsNttCache() = default;
    ScopedFormsNttCache(const ScopedFormsNttCache&) = delete;
    ScopedFormsNttCache& operator=(const ScopedFormsNttCache&) = delete;
private:
    engine::lever::EnvLever::Scoped guard_;
};
class ScopedFormsNttSimd {    // validation hook: false forces the scalar butterflies inside the FNTT route
public:
    explicit ScopedFormsNttSimd(bool on);
    ~ScopedFormsNttSimd();
    ScopedFormsNttSimd(const ScopedFormsNttSimd&) = delete;
    ScopedFormsNttSimd& operator=(const ScopedFormsNttSimd&) = delete;
private:
    int saved_;
};
// The FNTT engine's own general product (one plan, both spectra, the blocked / SIMD butterflies, the Barrett CRT) of two
// series of length <= N, for the validation; where the route does not apply it returns MulSeriesFast's value.
QSeries MulSeriesFntt(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N);
bool FnttApplicable(uint32_t ell, uint32_t N);

// ---------------------------------------------------------------------------
// (slot-24 FNTT3, docs/implementation/slot24/FNTT3_predictions.md, FNTT3_report.md) The three-prime route of
// BuildCuspSpace / BuildModularSpace past the two-prime CRT range.  A third NTT prime P3 = 45 * 2^23 + 1 (root 7; P3 < P2,
// so every butterfly / Montgomery / Shoup range bound of the FNTT engine holds) and a three-limb Garner CRT recover the exact
// integer convolution coefficient in [0, P1 P2 P3), P1 P2 P3 ~ 2.98e25; P3 caps the transform at 2^23, so the product is
// chunked: blocks of length B = 2^22 (nb = 1, B = N when 2N - 1 <= 2^23), the block spectra multiplied and summed per output
// block in the transform domain, one inverse per output block.  Exact while nb B (ell-1)^2 < P1 P2 P3 (Fntt3Applicable; at
// ell < 2^27 every N < 1.6e9); no length limit.  The same basis series as every other route (the truncated product in
// F_ell[q]/(q^N) is unique).  SELECTION: a space takes it only when the two-prime FNTT route is not exact
// (!FnttApplicable), ell >= 2^16, the fast product and HECKE_FORMS_NTT_CACHE are on and the lever is on -- exactly the
// spaces FormsSpaceSupported refused before FNTT3; every other space keeps its route.
// Rollback lever HECKE_FORMS_NTT3 (read-only, cannot change a value): unset / "1": on; exactly "0": off, i.e. the FNTT
// refusals (FormsSpaceSupported false, BuildCuspSpace's ENGINE_CHECK, the sat pre-run refusal); anything else warns and
// stays on.  When it is set, the first query prints one `EFFECTIVE HECKE_FORMS_NTT3 three_prime=on|off (...)` line.
bool FormsNtt3Enabled();
bool Fntt3Applicable(uint32_t ell, uint32_t N);   // the exactness predicate above (ell in [5, 2^27), N >= 1024)
uint32_t Fntt3Blocks(uint32_t N);                 // nb at N (under the current probe's block length)
// The engine's three-limb Garner step (validation): r_i in [0, 2 P_i) -> (the x in [0, P1 P2 P3) with x = r_i mod P_i) mod ell.
uint32_t Fntt3GarnerForValidation(uint32_t r1, uint32_t r2, uint32_t r3, uint32_t ell);
// (FNTT3 audit F2) read-only views of an engine built at (ell, N) (Fntt3Applicable): its own constants (the primes, P3's root,
// Garner's compile-time constants, its P1 P2 mod ell and Barrett constant, its layout, each plan's primitive L-th root) and its
// own Garner step on triples r = (r1, r2, r3, r1, ...), so the validation checks what the engine uses, not a recomputation.
struct Fntt3EngineConstants {
    uint32_t p1 = 0, p2 = 0, p3 = 0, g3 = 0;
    uint32_t inv_p1_mod_p2 = 0, p12_mod_p3 = 0, inv_p12_mod_p3 = 0;
    uint32_t p12_mod_ell = 0;
    uint64_t barrett = 0;
    uint32_t B = 0, nb = 0, lg = 0;
    uint32_t w1 = 0, w2 = 0, w3 = 0;   // the plans' w_L (L = 2^lg) mod P1, P2, P3
};
Fntt3EngineConstants Fntt3EngineConstantsForValidation(uint32_t ell, uint32_t N);
std::vector<uint32_t> Fntt3EngineGarnerForValidation(uint32_t ell, uint32_t N, const std::vector<uint32_t>& r);
// The three-prime engine's own product of two series of length <= N (validation); ENGINE_CHECK unless Fntt3Applicable.
QSeries MulSeriesFntt3(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N);
struct FormsNtt3Counters {
    uint64_t spaces = 0;                    // BuildCuspSpace / BuildModularSpace on the three-prime route
    uint64_t forward = 0, inverse = 0;      // series transformed / products inverted (each over 3 primes x nb blocks)
    uint64_t pointwise_blocks = 0;          // block-pair pointwise products (nb (nb+1) / 2 per product)
    uint64_t multiblock_engines = 0;        // engines with nb > 1 (the chunked product)
};
FormsNtt3Counters FormsNtt3CountersNow();
std::string FormsNtt3EffectiveLine();
class ScopedFormsNtt3 {        // in-process A/B: true = on, false = the rollback (the refusals); restores on destruction
public:
    explicit ScopedFormsNtt3(bool on);
    ~ScopedFormsNtt3() = default;
    ScopedFormsNtt3(const ScopedFormsNtt3&) = delete;
    ScopedFormsNtt3& operator=(const ScopedFormsNtt3&) = delete;
private:
    engine::lever::EnvLever::Scoped guard_;
};
class ScopedFormsNtt3Probe {   // validation only: block length 2^block_lg (4..22), and force = the three-prime route for
public:                        // every space where it is exact (also where the two-prime one applies: the identity check)
    ScopedFormsNtt3Probe(uint32_t block_lg, bool force);
    ~ScopedFormsNtt3Probe();
    ScopedFormsNtt3Probe(const ScopedFormsNtt3Probe&) = delete;
    ScopedFormsNtt3Probe& operator=(const ScopedFormsNtt3Probe&) = delete;
private:
    uint32_t saved_lg_;
    bool saved_force_;
};

// Delta, E_4, E_6 mod ell to precision N.
QSeries DeltaSeries(uint32_t ell, uint32_t N);
QSeries EisensteinSeries(uint32_t weight, uint32_t ell, uint32_t N);   // weight 4 or 6

// Throws (ENGINE_CHECK) on an odd k, k < 12, ell < 5 (E_4 / E_6 integral but the
// validation grid never needs it: kept to the grid's range), N < 2.
// (slot-24 ELL-W audit F6) The forms cross-route's coefficient-prime range: BuildCuspSpace needs 5 <= ell < this.  Exported
// so a caller can refuse a cell BEFORE its module run (HeckeSaturationDefect, the HECKE_SAT_DEFECT_CELLS parser).
// (slot-24 FNTT) Raised from 2^16 to 2^27 (every residue < P1, so the NTT inputs need no reduction), but above the word
// range ell >= kFormsWordEll a space is built only by the FNTT route: FormsSpaceSupported(ell, N) is the full condition.
inline constexpr uint32_t kFormsMaxEllExclusive = 1u << 27;
inline constexpr uint32_t kFormsWordEll = 1u << 16;   // below it every route (naive, schoolbook, G2-X NTT) is exact and fast
// (slot-24 FNTT) true iff BuildCuspSpace / BuildModularSpace accept (ell, N): 5 <= ell < 2^16 (any N); or ell < 2^27 and
// either N < kFastMulNttMinN (the naive products, quadratic in a small N) or the FNTT route is active (HECKE_FORMS_FASTMUL and
// HECKE_FORMS_NTT_CACHE not 0) and exact (FnttApplicable: 2N - 1 <= 2^25 and N (ell-1)^2 < P1 P2, i.e. about
// 2 (k/12 + 1) ell^3 < 7.9e16 at the sat route's precision: ell <= 114868 at n = 300, 77702 at n = 1000; ell = 100003 to n = 464).  Otherwise the
// only route left would be the O(N^2) naive one, so the space is refused (ENGINE_CHECK) instead of running for days.
// (slot-24 FNTT3) ... or the three-prime route is on (HECKE_FORMS_NTT3 not 0) and exact (Fntt3Applicable): with it every
// sat-route cell with ell < 2^27 and (k/12 + 1) ell < 8e8 is admitted, e.g. (1000, 100003); HECKE_FORMS_NTT3=0 restores the
// two-prime figures above.
bool FormsSpaceSupported(uint32_t ell, uint32_t N);
// (slot-24 FNTT) the sat route's forms precision (HeckeSaturationDefect BuildForms): N = 2 (k/12 + 1) pmax + 2 with pmax the
// largest of ell and the primes <= max(31, k/12) -- for the pre-run refusal of a cell (FormsSpaceSupported(ell, N)).
uint32_t SatFormsPrecision(uint32_t k, uint32_t ell);
CuspSpace BuildCuspSpace(uint32_t k, uint32_t ell, uint32_t N);

// (slot-12 WP-12) The full space M_k(F_ell) = F_ell f_0 (+) S_k(F_ell), f_0 =
// E_6^{2d} E_{k'} (constant term 1), to precision N: `basis` is f_0 .. f_d,
// `with_constant` is set, and HeckeOnCusp / LocaliseAtSystem act on it (the
// coordinates read from a_0 .. a_d).  The forms-side T / ell of M_k is what acts
// on H^1(Gamma; M_n(Z)) / tors ([E] cl:sturmmodl), Eisenstein systems included.
CuspSpace BuildModularSpace(uint32_t k, uint32_t ell, uint32_t N);

// The matrix of T_p on S_k(F_ell) in the basis (column j = coordinates of
// T_p f_j).  Needs precision > d p.  Throws when T_p f_j is not in the span to
// the available precision.
ExtMatrix HeckeOnCusp(const CuspSpace& S, uint32_t p);

// The d Sturm-range operators T_1 .. T_d as matrices (T_1 = I; T_{p^{r+1}} =
// T_p T_{p^r} - p^{k-1} T_{p^{r-1}}, T_{mn} = T_m T_n for coprime m, n), for
// the duality check dim span{T_1 .. T_d} = d.  Needs precision > d * d.
std::vector<ExtMatrix> SturmRangeOperators(const CuspSpace& S);

// The block at an F_ell-rational system (lambda_p), p in `primes` (index-
// aligned), and T_m / ell on it.
struct FormsBlock {
    std::vector<uint64_t> primes;             // the generators of T_m / ell
    std::vector<uint64_t> system_primes;      // the primes the system fixes
    std::vector<uint32_t> system;             // lambda_p, index-aligned
    uint32_t              dim = 0;            // dim V_m
    ExtMatrix             V;                  // d x dim, columns span V_m
    std::vector<ExtMatrix> t;                 // T_p |_{V_m}, index-aligned with primes
    AlgebraInvariants     A;                  // valid iff dim > 0
};

// `T` holds T_p on S_k for p in `primes`; the system fixes the primes in
// `system_primes` (each of which must be in `primes`).  dim == 0 when no form
// carries the system.
FormsBlock LocaliseAtSystem(const CuspSpace& S, const std::vector<uint64_t>& primes,
                            const std::vector<ExtMatrix>& T,
                            const std::vector<uint64_t>& system_primes,
                            const std::vector<uint32_t>& system);

// The Jordan type (block sizes, descending) of a nilpotent matrix; throws if
// N is not nilpotent.
std::vector<uint32_t> NilpotentJordanType(const ExtMatrix& N);

// The unique lambda in F_ell with t - lambda nilpotent, or -1 when t has no
// single F_ell-rational eigenvalue.
int64_t RationalEigenvalue(const ExtMatrix& t);

} // namespace forms
} // namespace hecke_ext

#endif // HECKE_EXT_FORMS_HECKE_ALGEBRA_H
