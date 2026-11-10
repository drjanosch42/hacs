// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_DESCENT_DESCENT_MODULE_EXPORT_H
#define INTEGRAL_DESCENT_DESCENT_MODULE_EXPORT_H

// ===========================================================================
// HeckeZ package EXPORT (hecke-extensions slot 25; docs/implementation/slot25/00-slot25-plan.md v0.2 Sec. 2.1, the card
// "EXPORT" of docs/implementation/slot24/heckez/06-work-packages.md).  [C] = docs/latex/gl2_contract.tex, cited by existing
// labels only: prop:torsionhecke (3), (3'), (4) (the torsion read and the free quotient), prop:strictliftN (2), (3), (5) (the
// cochain-level lifts and their transport to the ell-local tree model), prop:heilbronn (the shipped operator on the cokernel
// presentation, ell >= 5), prop:descentmaps (2)(g) (the precision rule: a Smith form over Z/ell^K is certified when its rank
// equals the rational rank).
//
// WHAT IT IS.  The descent's local reads (DescentHecke.h (C)-(E): the tree model at ell in {2, 3}, the cokernel presentation
// M_N / Rel_N at ell >= 5) made into ONE OBJECT: for (N, n, ell) and the Hecke primes P (p !| N),
//
//     H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K  =  (Z/ell^K)^{b_1}  (+)  (+)_j Z/ell^{min(sigma_j, K)}
//
// as a list of generators (cochains of the route: torsion cocycles first, then free cocycles) with their exponents, and the
// matrix of every T_p, p in P, on those generators (column convention T_p g_j = sum_i A_ij g_i, row i reduced mod
// ell^{exp_i}) -- the torsion block, the free block AND the block "free generator -> torsion coordinates".  It computes
// nothing new: the operators are the descent's (hecke3::IsoHeckeN transported by hecke3::TreeTransport at 2 and 3, the
// shipped hecke::level::HeilbronnOperator at ell >= 5), the elimination is hecke3::SmithLevelsLeft.  The descent record
// (DescentRecord.h, item 7 of sec:recordschema) is not touched: no hashed field, no record key, no golden object.
//
// THE READ (one run at the route's ring width K_run serves every K <= K_max):
//  (1) Smith form of d^0 over Z/ell^{K_run} with left transform: L0 d^0 R0 = Sigma0, rank s0 certified against rank_Q d^0
//      (prop:descentmaps (2)(g)); the torsion generators are the columns J (sigma_j >= 1) of L0^{-1} (prop:torsionhecke (3));
//      the columns s0 .. c-1 of L0^{-1} span the free part F of coker d^0.
//  (2) (the tree only: H^1 = Z^1 / B^1, C^2 != 0) Smith form of (d^1 restricted to F)^T with left transform L1, rank s1
//      certified against rank_Q d^1; the rows s1 .. of L1 are the coordinates (in F) of a basis of the free part of H^1; their
//      number must be b_1 (LevelNClosedForm).  On the cokernel presentation H^1 = coker d^0 and F itself is the free part.
//  (3) T_p applied to the generators (a cochain map); coordinates by L0 (torsion rows mod ell^{sigma_i}; F rows), then by
//      L1^{-T} (the kernel coordinates).  Checks, each a refusal by name: torsion goes to torsion (zero F coordinates mod
//      ell^{K_run - sigma_max}), the relations ell^{sigma_j} g_j are carried into the relations (an endomorphism), the
//      image of a free generator has zero coordinates on the non-kernel rows of L1 (mod ell^{K_max}), every image is a
//      cocycle (d^1 T g = 0 mod ell^{K_max}); before that the generators themselves (the tree): d^1 g = 0 mod ell^{K_run} (free),
//      mod ell^{K_run - sigma_max} (torsion), so a cocycle served at K <= K_max is a cocycle mod ell^K.
//  At ell >= 46341 (ell^2 >= 2^31) K_run = 1: the field F_ell (K_max = 1 iff no ell-torsion, else GroupUncertified).
//  Precision: K_max = K_run - sigma_max - 2 sigma1_max (sigma1 the exponents of (2)); the module is served iff
//  K_max >= max(1, sigma_max) (then the mixed block, read mod ell^{sigma_i}, does not see the free generators' error).
//
// REFUSALS (R-E1 of the card: a missing lift refuses that (N, ell, p) by name; the record word is `blocked(export)`):
//   BadEll                 ell not prime, or ell >= 5 with ell^1 >= 2^31 (no word ring);
//   NotPrime / PDividesN   a Hecke prime not prime / dividing N (U_p, p | N, is deferred: [R-Q4] of DescentHecke.h);
//   Lift                   the strict lift at p refused (IsoHeckeN; D4's code in `why`);
//   RssCeiling             rule 19, decided before anything is built;
//   GroupUncertified       rank_K d^0 != rank_Q d^0 (or, on the cokernel route, the free rank != b_1);
//   KernelUncertified      rank_K (d^1 | F) != rank_Q d^1, or the kernel's rank != b_1;
//   ModulePrecision        K_max < max(1, sigma_max): the module is not certified at this width;
//   NotEndomorphism        a torsion image with a non-zero free coordinate, or a relation not carried into the relations;
//   NotCocycle             d^1 g or d^1 T g not 0 at its precision (see (3)), or a free image off the kernel;
//   IntegerCap / DomainCap a named cap while an operator is assembled.
// A defect (ENGINE_CHECK, hecke::HeckeInternalError) is never a refusal: it propagates.
//
// No settings key, no env variable, no record key, no golden object.  Single-threaded except inside the shipped operators.
// ===========================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "DescentHecke.h"
#include "../../Hecke/HeckeLevelNHeilbronn.h"

namespace integral_descent {
namespace modexport {

using maps::Mat;
using maps::ZMod;

enum class ExportRefusal { None, BadEll, NotPrime, PDividesN, Lift, RssCeiling, GroupUncertified, KernelUncertified,
                           ModulePrecision, NotEndomorphism, NotCocycle, IntegerCap, DomainCap };
const char* ExportRefusalName(ExportRefusal r);

// H^1 (x) Z/ell^K read from an export (ModuleExport::at): every generator with its exponent (torsion min(sigma_j, K), free K),
// the T_p as rank x rank matrices in the column convention, entry (i, j) in [0, ell^{exps[i]}).
struct ModuleAtK {
    uint32_t ell = 0, K = 0;
    uint32_t torsion_count = 0, free_rank = 0;   // generators 0 .. torsion_count-1 are torsion, then free_rank free ones
    std::vector<uint32_t> exps;
    std::vector<uint32_t> ps;
    std::vector<std::vector<uint64_t>> ops;      // per p, row-major
    uint32_t rank() const { return static_cast<uint32_t>(exps.size()); }
    // the invariant factors (exponents), descending: K (free_rank times), then the torsion exponents
    std::vector<uint32_t> invariant_factors() const;
    // the free block of op o (free_rank x free_rank, mod ell^K): T_p on L / ell^K, L = H^1 / tors
    Mat free_block(size_t o) const;
    // det(x - free block) over Z/ell^K, high -> low (division-free Berkowitz); empty free part: {1}
    std::vector<uint32_t> free_charpoly(size_t o) const;
};

struct ExportOptions {
    uint32_t K_run = 0;                          // 0: the route's default width (2^30, 3^19; the largest ell^K < 2^31 at ell >= 5)
    uint64_t rss_cap_bytes = 40ull << 30;
    bool keep_cocycles = true;                   // keep the generators as cochains (`cocycles`)
    // (slot-25 EXPORT-Z, the step-2 entry point) a CALLER-SUPPLIED basis B of L (x) Z/ell^K: cochain_dim x b_1 cocycles of the
    // route's carrier (non-owning; entries reduced mod ell^{K_run} by the read).  When set, the read also returns the matrix of
    // every T_p in B (ModuleExport::F_basis, T_p b_j = sum_i F_ij b_i mod T, mod ell^{K_max}), or says why not (basis_why).
    // Step 2 calls it at many word primes q >= 5 (route "cokernel", K_run = 1 for q >= 46341: the field F_q) with ONE integral B.
    const Mat* basis = nullptr;
    // (slot-26 Z1, UP-Z (a); UPZ_design.md Sec. 1.3 R1) a prime p | N in `ps` is U_p^{(N)} = [Gamma_0(N) diag(1, p) Gamma_0(N)] (the
    // adjoint of the classical U_p on p-old spaces, [C, prop:levelconehecke](3)(d)) on the cokernel route (ell >= 5), the shipped
    // HeilbronnOperator with the admissibility filter of [C, prop:heilbronn] (p | N stated there).  Default false: p | N is refused
    // `PDividesN` byte for byte as before.  (S26 Z2) On the tree route (ell in {2, 3}) it opens U_p^{(N)} of [C, prop:strictliftN](9), the
    // level-1 lift PairLift(p) with the index-p transversal, transported to the tree (hecke3::IsoHeckeN, TreeTransport).
    bool allow_bad_primes = false;
    // (S26 Z2) the IsoHeckeN route for p | N at ell in {2, 3} (validation forces it; kAuto picks by size); value-neutral.
    hecke3::IsoRoute iso_route = hecke3::IsoRoute::kAuto;
};

struct ModuleExport {
    uint32_t N = 1, n = 0, ell = 0;
    std::vector<uint32_t> ps;
    std::string route;                           // "tree" (ell in {2, 3}) | "cokernel" (ell >= 5)
    std::string carrier;                         // what the cocycles' coordinates are ("C^1_T (J-coordinates)" | "M_N (cokernel presentation)")
    uint32_t K_run = 0;                          // the ring width of the read
    uint32_t K_max = 0;                          // the largest K served by at()
    uint32_t sigma_max = 0, sigma1_max = 0;
    uint32_t rank_Q0 = 0, rank_Q1 = 0, b1 = 0;   // rational ranks of d^0, d^1 (cokernel: 0) and b_1
    uint32_t cochain_dim = 0;                    // c = dim C^1 of the route
    uint32_t free_rank = 0;                      // = b1 when ok
    std::vector<uint32_t> sigma;                 // the torsion exponents (Smith order, ascending)
    Mat cocycles;                                // cochain_dim x rank (keep_cocycles), mod ell^{K_run}: torsion columns, then free
    // (slot-25 EXPORT-Z) the retraction pi_ell : C^1 -> T_ell of this export (keep_cocycles): torsion_count x cochain_dim, row a
    // = row J_a of L0 reduced mod ell^{sigma_a} -- the torsion coordinates of a cocycle.  pi_ell vanishes on B^1 and on every free
    // generator, and pi_ell(g_b) = delta_ab on the torsion generators; it is EXPORT-Z's `export_retraction/v1` at this ell.
    Mat retraction;
    // (slot-25 EXPORT-Z) the caller's basis B (ExportOptions::basis): basis_ok iff B's free coordinates form a unit matrix mod ell
    // (B is a basis of L / ell^{K_max}); then basis_coords (b_1 x b_1, B in the export's free generators) and per p F_basis (b_1 x
    // b_1, the matrix of T_p in B, mod ell^{K_max}); else basis_why names the reason (shape, not a cocycle, singular).
    bool basis_given = false, basis_ok = false;
    std::string basis_why;
    Mat basis_coords;
    std::vector<Mat> F_basis;
    // (EXZ2-P3 fix round 1, audit F2) the torsion coordinates of the caller's basis and of its images, by the retraction (keep_cocycles
    // and basis_ok): pi_B = pi_ell(B) (torsion_count x b_1, row a mod ell^{sigma_a}) and per p pi_TB = pi_ell(T_p B) (the route's own
    // operator applied to B, the same read as the torsion rows of T).  With them [b_j] = sum_a pi_B(a, j) g_a + sum_q basis_coords(q, j)
    // x_q in H^1 (x) Z/ell^{K_max}; EXPORT-Z step 2 serves its mixed block against the classes [b_j] (M_span) and checks
    // pi_TB == M_span + pi_B F (row a mod ell^{sigma_a}).  Empty without a basis or without keep_cocycles.
    Mat pi_B;
    std::vector<Mat> pi_TB;
    std::vector<Mat> T;                          // per p: rank x rank, column convention; torsion rows mod ell^{sigma_i}, free rows mod ell^{K_max}
    bool ok = false;
    ExportRefusal code = ExportRefusal::None;
    std::string refusal;                         // "REFUSED [<code>] ..." when refused
    double seconds = 0;
    uint64_t rss_pred = 0, rss_meas = 0;
    uint32_t rank() const { return static_cast<uint32_t>(sigma.size()) + free_rank; }
    // "ok" | "blocked(export)" (the record word of R-E1)
    std::string status() const { return ok ? "ok" : "blocked(export)"; }
    // the module at 1 <= K <= K_max (ENGINE_CHECK on ok and the range)
    ModuleAtK at(uint32_t K) const;
};

// The export of H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K with T_p, p in ps (any order; kept as given).
ModuleExport ExportModule(uint32_t N, uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps, const ExportOptions& opt = {});

// (slot-25 HZ-RAT-PERF (c)) the cokernel route's Hecke application, value-neutral: acc (D x k row-major, entries < ctx.p) += H X
// mod ctx.p, every column at once (per term the d x d block times the d x k slab, Shoup's multiply), parallel over column blocks.
// Exact residues: equal to H.apply column by column.  `threads` >= 1; no value depends on it.
void ApplyHeilbronnColumns(const hecke::level::HeilbronnOperator& H, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint64_t>& acc,
                           uint32_t threads);
// (slot-25 HZ-RAT-PERF (c)) the D x D matrix of H_{N,p} over ctx (D = m_N (n + 1), row-major), accumulated from the partial operators
// over slices of `mer` (= MerelSet(p)), `slice` matrices each, on up to `threads` workers (one accumulator each, summed mod ctx.p):
// equal to HeilbronnOperator(lvl, p, n, ctx).dense() entrywise, without the whole term table.
std::vector<uint32_t> HeilbronnDense(const hecke::level::Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                     const std::vector<hecke::Mat22>& mer, size_t slice, uint32_t threads);
// acc (D x k) += M X mod q for a D x D matrix M (entries < q < 2^31), parallel over row blocks.
void DenseApplyColumns(const std::vector<uint32_t>& M, size_t D, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint64_t>& acc,
                       uint64_t q, uint32_t threads);
// The thread count the export's Hecke application uses on this thread (1 unless a ScopedExportThreads is alive; the record sets it
// from its `threads`).  No value depends on it.
uint32_t ExportThreads();
struct ScopedExportThreads {
    uint32_t prev;
    explicit ScopedExportThreads(uint32_t t);
    ~ScopedExportThreads();
};

// The read of (1)-(3) on given data (the routes call it; IntegralDescent-style validations may call it on toy complexes):
// d0 = d^0 (c x a), d1 = d^1 (c' x c) or nullptr (H^1 = coker d^0), ops the cochain maps on C^1.
void ReadModule(ModuleExport& E, const ZMod& Z, const Mat& d0, uint32_t rank_Q0, const Mat* d1, uint32_t rank_Q1, uint32_t b1,
                const std::vector<hecke3::BatchOp>& ops, bool keep_cocycles, const Mat* basis = nullptr);

// Predicted peak RSS (bytes) of an export (the route's descent read plus the generator block and the kernel transform).
uint64_t PredictExportRss(uint32_t ell, uint32_t N, uint32_t n);
// (slot-25 EXPORT-Z audit F9) the cokernel route's default ring width at ell >= 5: the largest K with ell^K < 2^31 (K_run when
// ExportOptions::K_run = 0).  MatricesInBasisBatch reduces B mod q^{RouteMaxK(q)} and passes that K_run explicitly.
uint32_t RouteMaxK(uint32_t ell);

}  // namespace modexport
}  // namespace integral_descent

#endif
