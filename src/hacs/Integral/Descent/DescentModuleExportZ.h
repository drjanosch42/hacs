// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_MODULE_EXPORT_Z_H
#define INTEGRAL_DESCENT_DESCENT_MODULE_EXPORT_Z_H

// ===========================================================================
// HeckeZ package EXPORT-Z, STEP 1 (hecke-extensions slot 25; design docs/implementation/slot25/EXPORTZ_design.md, plan
// 00-slot25-plan.md Sec. 3 "EXPORT-Z", owner ruling 2026-11-06: two steps).  [C] = docs/latex/gl2_contract.tex, cited by existing
// labels only: prop:torsionhecke (1), (3) (the torsion Hecke module and its read), prop:strictliftN, prop:heilbronn (the operators,
// through EXPORT), prop:badprimesN (the torsion primes).
//
// THE OBJECT.  A = H^1(Gamma_0(N); Sym^n Z^2) = T (+) L-lift, T = (+)_i Z/d_i the torsion, L = A/T = Z^r.  In generators "torsion
// first, then lifts of a basis of L" every T_p (p in P, p !| N) is
//
//     [[ tau_p, M_p ], [ 0, F_p ]]      (column convention: T_p g_j = sum_i A_ij g_i)
//
// STEP 1 (this file) assembles, from the per-ell exports (DescentModuleExport.h) at K = sigma_max(ell), for every torsion prime ell
// of the record ({2} u {3 if n even} u B(M_N), hecke3::RecordTorsionPrimes):
//   tau_p   on T = (+)_ell T_ell, block-diagonal over ell (T_p preserves each ell-primary part); the ell-block is the export's torsion
//           block, row i reduced mod d_i = ell^{sigma_i}.  Exact integers in [0, d_i): in-word.  Generators of T: the export's torsion
//           cocycles, ell ascending, within ell in the export's (Smith) order.
//   M_p     (L -> T) per ell: the export's mixed block, t_ell x r, row i mod d_i, AGAINST THE EXPORT'S LOCAL FREE GENERATORS at ell
//           (the free cocycles of DescentRecord.module(ell, K)), not against a global basis of L: step 1 has none.
//   F_p     NOT in step 1: `free_block = "pending(step 2)"`.  The export's local free block mod ell^K is carried as `F_local` (labelled
//           with its modulus) because M's ell-block is only meaningful beside it (the Ext class), never as F.
// The splitting is named, not chosen silently: `splitting = "export_retraction/v1"` = (+)_ell pi_ell, pi_ell the export's torsion
// coordinates (ModuleExport::retraction: rows J of L0 mod ell^{sigma_a}, deterministic by EXPORT's pivot rule); each pi_ell is
// printed by its FNV-1a-64 (RetractionFnv).  Every local free generator lies in ker pi_ell (it is a combination of the columns
// s0.. of L0^{-1}), so s = ker pi restricted to the local free generators is the identity and M_p = pi T_p s needs no correction.
// The matrices are a PRESENTATION of the Z[T_P]-module A: unique up to the block-triangular group [[Aut T, Hom(L, T)], [0, GL_r(Z)]];
// the basis-free invariants (d_i, r, the record's char_Z(T_p | L) and torsion canon) are printed beside them by the console.
// `split = "unknown"` (HZ-TOR decides it); `entry_encoding = "word"` (every entry of tau and M is < d_i < 2^31).
//
// INTERFACE FOR STEP 2 (the global F over Z, wave 4).  Given a Z-basis b_1..b_r of L (as cocycles of a global integral model):
//   C_ell in GL_r(Z/ell^K): b mod ell^K expressed in the export's local free generators (their free coordinates; the torsion part
//   is dropped since the local free generators span ker pi_ell mod T);  then  M_p^global|_ell = M_p^local|_ell * C_ell  (row i mod
//   d_i) -- that is the block against the ker pi lifts of B; as built (fix round 1, audit F2) step 2 serves the block against the
//   classes [b_j] themselves, M_span = M_kerpi - pi_B F + tau pi_B, see STEP 2 --, F_p = the global free block over Z, and
//   C_ell F_p = F_local,p C_ell mod ell^K (a check, not a formula).  Hom(L, T) =
//   (+)_ell Hom(L, T_ell), so no further gluing.  Step 2 sets `free_block = "exact"`, fills `F` per op and `M_basis = "global"`,
//   keeps every other field.
//
// STEP 2 (EXZ2-P3, ExportModuleZ2 below; docs/implementation/slot25/EXPORTZ2_design.md Sec. 5, 7.5).  The free block over Z in ONE
// picked basis: B = BuildFreeBasis (DescentFreeBasisZ.h, rule iso_dixon/v3 on the free tree carrier Iso_T, certified by
// det(Phi_cert^T B) = +-1 and the rank witness), F_p = FreeBlockCrt (DescentFreeBlockCrt.h: RNS lanes, proven CRT, exact BigInt).  B is
// carried to every torsion prime's export route (RouteBasis: the pullback along Psi^T_ell : Iso^(ell)_T -> Iso_T, [C, prop:isomodeltree]
// (3); J-coordinates at ell in {2, 3}, the edge cochains M_N at ell >= 5) and the export runs once with
// ExportOptions::basis = B there: C_ell = basis_coords (B in the export's local free generators, mod ell^{K_max}), F_basis (T_p in B
// mod ell^{K_max}, by the route's own operator), pi_B = pi_ell(B) and pi_TB = pi_ell(T_p B) (the retraction on B and on the route's
// T_p B).
// THE FREE GENERATORS ARE THE CLASSES [b_j] OF THE CERTIFIED COCYCLES (fix round 1, audit F2, owner's choice (b)): the Z^r summand of
// H^1 = T (+) Z^r is span{[b_j]}, and the served M is the mixed block of that complement,
//     T_p [b_j] = sum_i F_ij [b_i] + sum_a M_aj g_a      (g_a the torsion generators; row a mod d_a),
//     M_p|_ell = M_kerpi - pi_B F_p + tau_p|_ell pi_B   (row a mod d_a),   M_kerpi = M_p^local|_ell C_ell,
// where M_kerpi is the block against the lifts s_j = [b_j] - sum_a pi_a(b_j) g_a in ker pi (the step-1 splitting export_retraction/v1)
// and pi_B is exported beside it (the change of complement, s_j -> [b_j]: [[1, pi_B], [0, 1]]).  CHECKED in production at every
// torsion prime (incl. 2, 3), with no M_local and no C:
//     pi_ell(T_p B) == M_p|_ell + pi_B F_p   (row a mod d_a)   (the M oracle: the torsion coordinates of both sides of the identity),
//     F_p mod ell^{K_max} == F_basis   (the global oracle: their free coordinates),     C_ell F_p == F_local,p C_ell mod ell^{sigma_max}.
// The route of pi_ell(T_p B): the export's own operator on the transported B -- at ell = 2, 3 the tree transport of T~_Iso (IsoHeckeN on
// the ell-local tree, the same lift as the lanes: the M oracle there is independent of M_local and C, NOT of T~_Iso); at ell >= 5 the
// shipped Heilbronn operator H_{N,p} on M_N (independent of T~_Iso).
// tau and the local blocks are the step-1 values of the same exports (ExportOptions::basis changes no other field).  free_block =
// "exact", M_basis = "global", entry_encoding = "exact" (F: nf::BigInt; tau, M: words < d_i).  A refusal of the free basis or of the CRT
// keeps the step-1 object with free_block = "refused" and the reason in free_refusal (M stays local).
//
// REFUSALS: an export refusal at any torsion prime makes the whole object `blocked(export)` with that export's refusal line (no
// partial tau / M is served; d, tau, local are cleared and free_rank = 0, so nothing of an earlier, accepted export survives -- the
// console prints the torsion fields of a blocked value as the word "blocked", never as [] / 0).  A defect (ENGINE_CHECK) propagates.  No settings key, no env variable, no record key, no golden object.
// ===========================================================================

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "DescentModuleExport.h"
#include "DescentFreeBasisZ.h"
#include "DescentFreeBlockCrt.h"
#include "IsoTreeCarrierZ.h"
#include "../../NumberField/BigInt.h"

namespace integral_descent {
namespace modexport {

inline constexpr const char* kSplittingV1 = "export_retraction/v1";
inline constexpr const char* kFreeBlockPending = "pending(step 2)";

// FNV-1a-64 of the canonical text of pi_ell ("export_retraction/v1|N=..|n=..|ell=..|route=..|K_run=..|c=..|sigma=..|" then the rows,
// entries ',' rows ';').  ENGINE_CHECK on an ok export with its retraction kept.
uint64_t RetractionFnv(const ModuleExport& E);

struct ModuleZ;
struct LocalPartZ;
// (step 2, fix round 1) FNV-1a-64 of pi_B at one torsion prime: "pi_B/v1|N=..|n=..|ell=..|basis_fnv=0x<16 hex>|d=..,..|" then the rows
// (entries ',' rows ';').  And of op o's mixed block M over every torsion prime: "M_span/v1|N=..|n=..|p=..|basis_fnv=0x..|" then per
// ell with t > 0 "ell=<ell>|d=..|" and its rows.  Both on the served (span) convention; ENGINE_CHECK on a non-step-2 object.
uint64_t PiBFnv(const ModuleZ& Z, const LocalPartZ& P);
uint64_t MixedBlockFnv(const ModuleZ& Z, size_t o);
// (step 2, fix round 1, audit F4 (ii)) the M oracle at one torsion prime: piTB == M + piB Fk, row i mod d[t0 + i] (piTB, M, piB: t x r;
// Fk: F mod ell^K, r x r).  ExportModuleZ2 ENGINE_CHECKs it on every op at every torsion prime; the validation's teeth call it.
bool MixedOracleHolds(const std::vector<uint64_t>& d, uint32_t t0, uint32_t t, uint32_t r, const std::vector<uint64_t>& piTB,
                      const std::vector<uint64_t>& M, const std::vector<uint64_t>& piB, const std::vector<uint64_t>& Fk);

// One torsion prime's part.
struct LocalPartZ {
    uint32_t ell = 0;
    std::shared_ptr<const ModuleExport> E;   // the export (default width; cocycles and retraction kept)
    uint32_t K = 0;                          // sigma_max(ell); 0: no ell-torsion (nothing served at ell)
    uint32_t t0 = 0, t = 0;                  // offset and number of its torsion generators in the global list
    ModuleAtK at;                            // E->at(K) (empty when K == 0)
    uint64_t retraction_fnv = 0;
    // (step 2, EXZ2-P3) the global basis B at this ell: C = B in the export's local free generators (r x r, mod ell^{K_C}, K_C = the
    // basis export's K_max); M_global per op (t x r, row i mod d_{t0+i}); F_basis per op (the route's T_p in B mod ell^{K_C})
    uint32_t K_C = 0;
    std::vector<uint64_t> C;
    std::vector<std::vector<uint64_t>> M_global;   // M_span (fix round 1, audit F2): against the classes [b_j] themselves, see STEP 2
    std::vector<std::vector<uint64_t>> F_basis;
    // (fix round 1, audit F2) pi_B = pi_ell(B) (t x r, row i mod d_{t0+i}: the torsion coordinates of the classes [b_j]), its FNV
    // (PiBFnv), and per op pi_TB = pi_ell(T_p B) (the route's own operator; the M oracle's input)
    std::vector<uint64_t> pi_B;
    uint64_t pi_B_fnv = 0;
    std::vector<std::vector<uint64_t>> pi_TB;
};

// (step 2) M and F in ONE basis for the HZ-TOR read (audit HZTOR M1): basis_tag "local" (M_block, F_local) or "global" (M_global, F mod
// ell^K); K = sigma_max(ell)
struct FreeView {
    std::string basis_tag;
    uint32_t ell = 0, K = 0;
    std::vector<uint64_t> M_block;           // t x r, row i mod d_{t0+i}
    std::vector<uint64_t> F_block;           // r x r mod ell^K
};

// (step 2) the free block over Z
struct FreeBlockZ {
    std::string basis_rule, q0_key, saturation, bound_basis;
    uint64_t basis_fnv = 0, phi_fnv = 0;     // PackedFnv(B), PackedFnv(Phi_cert)
    uint32_t bits_B = 0, bits_Phi = 0, K_pairs = 0;
    std::vector<std::vector<nf::BigInt>> F; // per op: r x r row-major, exact
    std::vector<uint64_t> F_fnv;             // MatrixFnv per op
    std::vector<std::string> proof;          // "apriori" | "apost" | "rank0"
    std::vector<uint32_t> primes_used, primes_apriori, bits_F;
    // stage timings (s) and the process RSS growth samples (bytes): carrier, basis (pairs, sums, sat, cert), lanes, bound, lift, assembly
    double t_carrier = 0, t_pairs = 0, t_sums = 0, t_sat = 0, t_cert = 0, t_basis = 0;
    double t_bound = 0, t_inverse = 0, t_lanes = 0, t_lift = 0, t_crt = 0, t_exports = 0, t_assembly = 0;
    uint32_t lanes = 0;
    std::string lane_route = "cpu";          // (fix round 1, audit F5) the lanes' route: "cpu" | "gpu" (ModuleZ2Options::lanes); F is
                                             // bit-identical on both (not printed by the console: it is not a value)
};

struct ModuleZ {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps, torsion_primes;
    std::string splitting = kSplittingV1;
    std::string free_block = kFreeBlockPending;
    std::string M_basis = "local";           // step 1: per ell against the export's local free generators
    std::string entry_encoding = "word";
    std::string split = "unknown";
    std::string free_refusal;                // (step 2) when free_block == "refused"
    FreeBlockZ global;                       // (step 2) filled when free_block == "exact"
    bool ok = false;
    ExportRefusal code = ExportRefusal::None;
    std::string refusal;                     // the first refused export's line, prefixed "ell = <ell>: "
    uint32_t free_rank = 0;                  // r = rank L (b_1)
    std::vector<uint32_t> tor_ell, tor_sigma;   // per global torsion generator
    std::vector<uint64_t> d;                 // d_i = ell^{sigma_i}
    std::vector<LocalPartZ> local;           // per torsion prime, ascending
    std::vector<std::vector<uint64_t>> tau;  // per p: t x t row-major (block-diagonal), row i mod d_i
    std::vector<uint64_t> M_fnv;             // (step 2, fix round 1) per p: MixedBlockFnv (M_span over every ell, with basis_fnv and d)
    double seconds = 0;
    uint32_t torsion_rank() const { return static_cast<uint32_t>(d.size()); }
    uint32_t largest_torsion_prime() const;  // the largest ell with ell-torsion (0: T = 0)
    std::string status() const { return ok ? "ok" : "blocked(export)"; }
    // M_p's ell-block of local part l: t_ell x r row-major (row i mod d_{t0 + i}), against the local free generators
    std::vector<uint64_t> M_block(size_t o, size_t l) const;
    // the local free block mod ell^K of part l (r x r row-major) -- NOT F (step 2)
    std::vector<uint64_t> F_local(size_t o, size_t l) const;
    // (step 2, EXZ2-P3) M and F of op o at part l in ONE basis: the global one when free_block == "exact", else the local one
    // (HZTOR audit M1: HeckeZTor::FromModuleZ reads only this).  free_view_local: always the step-1 pair.
    FreeView free_view(size_t o, size_t l) const;
    FreeView free_view_local(size_t o, size_t l) const;
    // (step 2) F_o mod ell^K in the basis B (free_block == "exact"; ell^K < 2^63): the exact F reduced, no re-run ([C] item 8 export_word)
    // (slot-27 EXW) ell is uint64_t (was uint32_t; the callers pass u32 and are unchanged): the at-root read reaches ell >= 2^32
    std::vector<uint64_t> free_mod(size_t o, uint64_t ell, uint32_t K) const;
    // (step 2) T_p on T (+) Z^r as one matrix [[tau, M_global], [0, F]] ((t + r)^2 row-major; rows < t mod d_i): the assembly
    std::vector<nf::BigInt> full_matrix(size_t o) const;
};

// (slot-27 EXW, docs/implementation/slot27/EXW_design.md Sec. 3.1) F_o mod q of a free block over Z (FreeBlockZ::F[o], r x r row-major),
// canonical residues in [0, q); q >= 2, q < 2^63 (ENGINE_CHECK).  The reduction of ModuleZ::free_mod, for a FreeBlockRun without the
// step-1 assembly.
std::vector<uint64_t> FreeBlockModQ(const FreeBlockZ& G, size_t o, uint64_t q);

struct ModuleZOptions {
    ExportOptions export_opt;                // passed to ExportModule when no provider is given (keep_cocycles forced on)
};
using ExportProvider = std::function<std::shared_ptr<const ModuleExport>(uint32_t ell)>;

// Step 1 of EXPORT-Z for (N, n, P = ps).  provider (optional) serves the export at ell (the console's per-handle cache); it must
// return an export of (N, n, ell, ps) with cocycles kept.  The torsion primes are hecke3::RecordTorsionPrimes(N, n).
ModuleZ ExportModuleZ(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const ModuleZOptions& opt = {},
                      const ExportProvider& provider = {});

// ---- STEP 2 (EXZ2-P3) ----------------------------------------------------------------------------------------------------------
// (fix round 1, audit F5) the lanes' route: FromDevice = the existing ENGINE_DEVICE switch (Gpu iff engine::cuda::device_enabled(),
// i.e. ENGINE_DEVICE=gpu and a device; the default ENGINE_DEVICE=cpu keeps the CPU lanes, the golden path), or forced.  No value
// depends on it (FreeBlockCrt's routes are bit-identical; HeckeZExportZ2SlowValidation arm (D) compares them on real bases).
enum class LaneChoice { FromDevice, Cpu, Gpu };
struct ModuleZ2Options {
    ModuleZOptions step1;
    uint32_t threads = 0;                    // the free block's CPU lanes (0: ENGINE_THREADS, capped at 16)
    LaneChoice lanes = LaneChoice::FromDevice;
    freez::BoundBasis bound = freez::BoundBasis::Auto;   // the Sec. 1.6 bound: Auto (Phi_0 while its inverse is cheap, else Phi_cert)
    uint64_t rss_cap_bytes = 3ull << 30;     // ONE [RssCeiling] (O-3) for the pairs, the sums and the lanes (P1b's default)
    uint32_t rss_lanes = 0;                  // (slot-27 EXW fix, M-EXW2) FreeBlockOptions::rss_lanes (0: the run's lanes, as before)
    bool verbose = false;                    // one EXZ2_P3_STAGE line per stage on stdout (measurement)
    // (slot-26 Z1, UP-Z (a)) p | N in ps is U_p: the free block by the free-model U_p (IsoHeckeN, `prop:strictliftN` (9)), the ell >= 5
    // exports by Heilbronn at ell >= 5, by the tree route at ell in {2, 3} (S26 Z2; ExportOptions::allow_bad_primes).  Default false.
    bool allow_bad_primes = false;
};
// Step 2 for (N, n, P = ps): step 1 (the same exports, run once each with the global basis) + the free block over Z + M in that basis.
ModuleZ ExportModuleZ2(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const ModuleZ2Options& opt = {});
// The two halves of ExportModuleZ2 (the validations reuse one basis for several reads).  RunFreeBlock: BuildFreeBasis + FreeBlockCrt on
// the carrier (refusal non-empty: a resource refusal of either; F is then not served).  AssembleModuleZ2: the exports at every torsion
// prime in B (or plain when refused), step 1 on them, then C_ell, the oracle checks, M_global.  cache (optional): the exports by ell,
// filled when absent and reused when present (they must be ExportInBasis(.., run.FB.B, ..) when F exists).
struct FreeBlockRun {
    freez::FreeBasisResult FB;
    freez::FreeBlockResult CR;
    FreeBlockZ G;
    std::string refusal;
};
using ExportCache = std::map<uint32_t, std::shared_ptr<const ModuleExport>>;
FreeBlockRun RunFreeBlock(const freez::IsoTreeCarrierZ& car, const std::vector<uint32_t>& ps, const ModuleZ2Options& opt = {});
ModuleZ AssembleModuleZ2(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const FreeBlockRun& run, const ModuleZ2Options& opt = {},
                         ExportCache* cache = nullptr);
// B (dim C^1_{Iso_T} x r, exact, the free tree carrier's cocycles) carried to the export route's carrier at ell, reduced mod ell^{K_run}
// (K_run = LocalWidth at 2, 3; RouteMaxK at ell >= 5): (Phi^Z_T)^* to Iso, then Psi_T(ell)^* to the ell-local tree (J-coordinates at
// ell in {2, 3}; the edge cochains = M_N at ell >= 5, n even).  `K_run` receives the width.  ENGINE_CHECK on a carrier mismatch.
// (As built: the pullback along Psi^T_ell : Iso^(ell)_T -> Iso_T of [C, prop:isomodeltree](3) directly -- the same class map as through
// Iso, Phi^Z_T Psi^Z_T being homotopic to the identity -- 3 tree cells instead of Iso's.)
Mat RouteBasis(uint32_t N, uint32_t n, const freez::PackedZ& B, uint32_t ell, uint32_t& K_run);
// The export at ell in the global basis (keep_cocycles, ExportOptions::basis = RouteBasis(...), K_run explicit).
ModuleExport ExportInBasis(uint32_t N, uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps, const freez::PackedZ& B,
                           const ExportOptions& eo = {});

// ---- the step-2 hook: T_p mod q_1 .. q_s in ONE caller-supplied integral basis B (a batch over primes) ----------------------
// B: cochain_dim x r row-major SIGNED integers in the cokernel carrier M_N (the same Z-cochains at every q; |entries| < 2^62).  For
// each q (a prime >= 5, q !| N; route "cokernel"), B is reduced mod q^{K_run} ONCE and the export runs with ExportOptions::basis:
// out[s].F[p] = the matrix of T_p in B mod q_s^{K_max} (K_max = 1 at q >= 46341: the field), or out[s].why.  Primes are independent
// (residue number system): `threads` > 1 runs them on that many std::threads, results in the order of qs, bit-identical to 1 thread.
// q < 5 and q | N are refused by name (out[s].why); B is reduced mod q^{RouteMaxK(q)}, the K_run passed to the export.  A defect
// in a lane (an ENGINE_CHECK, bad_alloc) is caught per lane and the first one in qs order is rethrown after the join, as on 1 thread.
struct BasisBatchEntry {
    uint32_t q = 0, K = 0;                   // the modulus q^K of F
    bool ok = false;
    std::string why;                         // the export's refusal or ModuleExport::basis_why
    std::vector<Mat> F;                      // per p (r x r)
};
// lane_probe (optional, a test seam): called with q at the start of every lane, inside the lane's catch (HeckeZExportZValidation
// arm (S) injects a defect through it).
using LaneProbe = std::function<void(uint32_t q)>;
std::vector<BasisBatchEntry> MatricesInBasisBatch(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const std::vector<uint32_t>& qs,
                                                  const std::vector<int64_t>& B, uint32_t rows, uint32_t r, uint32_t threads = 1,
                                                  const LaneProbe& lane_probe = {});

}  // namespace modexport
}  // namespace integral_descent

#endif
