// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, G2-A / G2-B) HeckeModuleStreamValidation (standard) and
// HeckeModuleStreamSlowValidation (slow): docs/implementation/slot23/G2_plan.md Sec. 2.7.
//
// The levers under test (engine::elim::module_stream_enabled(), env HECKE_MODULE_STREAM, default on):
//   G2-A  Theta_eq (HeckeOperatorOnKT::compute_module, (iii)), X_q (EquivariantCohomologyOnKT::
//         compute_cohomology_module) and the C^q_eq coordinates of H_class_coordinates (A^{(q)}'s first solve)
//         read off the unit pivots of C (C[piv, :] = I), the residual recomputed
//         exactly over every row (engine::elim::solve_unit_pivot_readoff); any other outcome takes the
//         Howell solve (solve_multi_rhs_module), which is also the whole legacy route;
//   G2-B  the d o d certificate X_q X_{q-1} = 0 by one GEMM; cyclic_decomposition's column clear i-outer;
//   G2-C  Theta^{(q)} assembled coset-outer, Omega_i per (degree, coset) and dropped, the fused memos per coset, the
//         T_ell low-rank terms of one (sigma, sigma_k) block accumulated by one grouped GEMM;
//   G2-D  route 2 built in sigma-row chunks (ceil(N_q / 2) block rows when N_q > 7, else all N_q: 7 | 6 + 5 | 5),
//         coset by coset with a memo scoped to (chunk, coset), compared entrywise with Theta's rows, dropped;
//   G2-E  Theta_eq read off the product Theta C (engine::elim::solve_unit_pivot_readoff_product: W never
//         materialised); the lean retention (ModuleRetention::Lean + release_module_intermediates(), RunModuleT only).
// The entrywise twin is the legacy route (beside the independent oracle (R), S24 CL-O), selected IN PROCESS by engine::elim::ScopedModuleStream(false) -- the same
// switch as the rollback arm HECKE_MODULE_STREAM=0 -- so the validation does not depend on the environment.
//
// Arms (standard):
//   (U)  unit level: the read-off against solve_multi_rhs_module on random injective unit-pivot B over
//        Z/5^3, Z/7^2, Z/2^5 (u32) and Z/7^12 (u64), entry for entry; cyclic_decomposition stream vs legacy on
//        random (G, Rl) over the same rings and Z/5^30 (u128), every field (exponents, basis, Gamma, U,
//        position).
//   (N)  negative controls: a right-hand side moved off im B at a NON-pivot row must make the read-off report
//        Residual and the Howell solve throw its legacy message; a B with B[piv, :] != I must report
//        NotIdentity; the operator-level control (ScopedReadoffCorruption: one read-off entry +1) must fire
//        the residual check, fall back to the Howell solve (counter howell) and leave every object unchanged,
//        at (20, 5, 3) u32 and (40, 7, 12) u64 (audit m3 (a)); an inconsistent Theta_eq (hecke::
//        ScopedThetaEqPerturbation: W = T C moved off im C) must fail (iii) with the legacy text, byte for
//        byte, on the stream route, at the same two cells (audit m3 (b)); the d o d predicate
//        (engine::elim::product_is_zero, GEMM and scalar) on planted zero / non-zero / empty products over
//        Z/5^3, Z/2^5 and Z/7^12 (audit m1).
//   (O)  object level, stream vs legacy, entrywise: C, piv, X, Zc, H, dec.{exponents, basis, Gamma, U,
//        position}, dK, length_H of the cohomology, and per Hecke prime p in {2, 3, 5, 7, ell} A^{(q)},
//        Theta^{(q)}, Theta^{(q)}_eq, the certificate bundle (routes_bit_identical, relation / cocycle checks,
//        theta_eq_solves, residual), q = 0, 1, 2; cells (n, ell, K) = (20, 5, 3), (60, 5, 6) (K_cls = 3),
//        (30, 7, 4), (24, 11, 2), (40, 13, 3), (24, 43, 2), (12, 101, 2), (10, 1009, 2) (ell >= 94: the T_ell
//        low-rank term), the u64 word (40, 7, 12), and Hida's lattice L_n at (8, 5, 3) (the non-unit-pivot
//        fallback: counter howell >= 1).  Non-vacuity: the stream run takes the read-off >= 1 time and no
//        legacy solve, the d o d GEMM and the interchanged clear; the legacy run none of them.
//   (S23 G2-C/D/E) (U) the product read-off against the read-off on Y for row blocks 1, 4, 7, 256, with (N) its
//        Residual (Y off im B, X(0, 0) + 1) and NotIdentity controls; (N) a corrupted route-2 block entry
//        (hecke::ScopedRoute2Perturbation) must throw the O12(a) text byte for byte on both routes, at (20, 5, 3),
//        (10, 101, 2) (low-rank route 2) and (40, 7, 12) u64, at three sites: the last entry of q = 0 (one chunk) and,
//        at q = 1 (two chunks, rows 0..5 and 6..10), entry (0, 0) in the first chunk and the last entry in the second
//        (audit m1: the chunk offset under a negative control); the corruption / inconsistent-Theta_eq controls also at
//        (10, 101, 2); (O) adds (16, 211, 2), (12, 101, 3), (10, 101, 5) u64, (8, 1009, 4) u64, (6, 10007, 2) (the
//        kGroupCap early flush: lr=grouped,early_flush >= 1 asserted, audit m2; 0 asserted at ell <= 101) and
//        (8, 1291, 3) u32 (1291^3 > 2^31: Barrett axpy on the u32 word, audit m3), and every (O) arm checks
//        the counters omega=streamed (sum 3 (p + 1)), route2=blocked (23 sigma block rows per operator), lr=grouped
//        (iff T_ell is low-rank), w=nonpivot on the stream run and omega=table, route2=full, lr=scalar on the legacy run; (L) the lean
//        retention at (24, 43, 2), (12, 101, 2), (40, 7, 12) u64: the kept objects equal the legacy route's, X / Zc /
//        Theta / Theta_eq released and their guarded accessors throw their named text, retention=lean counted.
//   (B)  ell in {2, 3} (the Bredon flat-coordinate route, which the levers do not touch): (10, 2, 3) and
//        (10, 3, 2), lengths and invariant factors equal, and no module-stream counter moved.
//   (S)  cell level, byte for byte: the SAT_DEFECT lines (header + one line per block) of
//        ComputeSaturationDefectCell at (60, 5) KMAX 6 (K_cls = 3), (108, 7) KMAX 6, (160, 11) KMAX 6, (50, 13) KMAX 3,
//        (24, 43), (16, 101), (10, 1009); K_run and Lv_run equal.
//   One `MODULE_STREAM_CELL <arm> seconds stream= legacy= peak_rss_mib=` line per object / cell arm.
//   (C)  (S23 G2-G) cyclic_decomposition / invariant_factors with the lean work buffers (N never formed, one Howell
//        form of [Gamma^T | I] for kernel and coordinates, buffers dropped once read, length(colspan Rl) after s = 0) against
//        the legacy buffers, in process, field by field (exponents, position, basis, Gamma, U; invariant_factors (G, Rl)
//        and (G); cyclic_coordinates) on random (G, Rl) of eight shapes (nr = 0, ng = 0, m = 1, wide, s = 0, Rl inside /
//        outside span G, cohomology-shaped; audit m3 adds m = 0, Rl = 0 with nr > 0, Rl = I, G = I, socle-only Rl, socle-only G:
//        fourteen) over Z/5^3, Z/43^2, Z/2^5, Z/1291^3 (u32), Z/7^12, Z/10007^4 (u64), Z/5^30
//        (u128); the counters cyc_buffers=lean|legacy, invf=lean|legacy; (N) the F_ell refusal and a row-count mismatch
//        throw the same text on both routes, and an ell^{k-1} perturbation of G is seen by the comparator.  The (O) arms
//        also assert cyc_buffers / invf routing; (B) asserts invariant_factors' buffers are its only lever step.
//        (audit m2) An INDEPENDENT oracle on tiny modules (Z/2^3, Z/3^2, Z/2^4, Z/5^2 u32, Z/3^2 u64; m <= 3, the fourteen
//        shapes): |M[ell^j]| counted by brute force over (Z/ell^k)^m equals prod ell^{min(e_i, j)} for 0 <= j <= k, with its
//        own control (an extra exponent 1 is rejected on every M != 0), and a fault injected on the lean route
//        (el::ScopedLeanBufferCorruption) is detected by both entry points' checks and by the oracle, the legacy route inert.
//   (T)  (S23 G2-F) the module route's tau table, hecke::TauTableContents::ChainsOnly (RunModuleT on the stream route), against
//        the Full table on the same coset tables, per Hecke prime in {2, 3, 5, 7, ell}: the chain records term for term,
//        B_tab / B / fill counters / regime / to_string() equal, certify() == 0 on both, every field-route accessor of the
//        ChainsOnly table (xi_terms, rho_pool*, the pool corrupter) and the field operator's constructor refuse with the named
//        text, stored_bytes() strictly smaller, and (fix F2) a perturbed vertex record is rejected by certify() with 20 on both; the module operator on it (stream route) equals the stream route on the Full table
//        and the legacy route on the Full table (A^{(q)}, Theta, Theta_eq, certificates), at (20, 5, 3), (12, 101, 2),
//        (6, 10007, 2) u32 and (40, 7, 12), (8, 1009, 4) u64.  The (L) arms build ChainsOnly tables (as RunModuleT does) and the
//        (S) arms assert the counters tau=chains on the stream run and tau=pooled on the legacy run.
//   (V)  (S23 G2-T) the module-eval lever (engine::elim::module_eval_enabled(), env HECKE_MODULE_EVAL, default on), against
//        its rollback in process (el::ScopedModuleEval(false), the same switch as HECKE_MODULE_EVAL=0): (a) unit level,
//        ComputeSymmetricPowerRowsRing (one Pascal row, no FlatMatrix carrier, the column steps in a wavefront with uinv
//        folded into one reduction) against ComputeSymmetricPowerMatrixRing entry for entry on random keys of every route
//        ([U_a], [U_b], [S], [Z], each counted) at n in {0, 1, 2, 7, 8, 9, 30, 64} over Z/5^3, Z/1009^2, Z/10007^2,
//        Z/1291^3, Z/3^20 and Z/65521^2 (u32, p > 2^31: the unfused step; the last two above floor(2^32 / sqrt 3), so
//        a wrongly fused u32 step overflows there, audit F2), Z/7^12, Z/10007^4 (u64, fused) and Z/3^40 (u64, p > 2^62:
//        unfused);
//        factor_low_rank with the divisibility scan, the valuation floor and the rank-1 GEMM update (fast_scan) against
//        the per-entry scan (t, U, V) on singular rho_n blocks and planted sums with forced valuations (t >= 2 asserted);
//        (b) operator level, the stream route with the lever on vs off: every object of (O) (CompareObjects), and the
//        counters eval=fast (3 per operator) on the on-run, eval=legacy on the off-run, at (20, 5, 3), (12, 101, 2),
//        (6, 10007, 2), (8, 1291, 3) u32 and (40, 7, 12), (10, 101, 5) u64 and Hida's L_n (8, 5, 3); (c) cell level, the
//        SAT_DEFECT lines byte for byte at (16, 101) KMAX 6 and (10, 1009) KMAX 6.  The (O) arms assert eval=fast on the
//        stream run and no eval counter on the legacy run (the lever is consulted by the stream route only).
//        (S23 G2-T2) (d) every rho fill path available on this CPU / build (scalar, AVX2, AVX-512F; RhoFillPath, env
//        HECKE_RHO_SIMD), each FORCED by ScopedRhoFillPath: ComputeSymmetricPowerRowsRing == ComputeSymmetricPowerMatrixRing
//        entrywise into a dirty buffer at every n in 0..140 (every tail of the 8-, 32-, 64-lane wavefronts) and
//        n in {191, 192, 193, 255, 256, 257, 300}, on keys of every route, the all-(p - 1) key, w = 0 and t = 0, over Z/5^3,
//        Z/1009^2, Z/10007^2, Z/23159^2, Z/3^19, Z/46337^2 (p just below 2^31), Z/2^31 (the SIMD gate's edge), Z/2^2,
//        Z/1291^3 and Z/65521^2 (u32 above 2^31: scalar whatever is forced), Z/10007^4 and Z/3^40 (u64: scalar); the [U] fill
//        counters prove which path ran; (e) operator level, each SIMD path against scalar (CompareObjects) at (20, 5, 3),
//        (12, 101, 2), (70, 101, 2), (30, 7, 11), (8, 1291, 3) u32 and (10, 101, 5) u64; (f) the SAT_DEFECT
//        lines per path at (16, 101) KMAX 6 and (70, 1009) KMAX 3; (g) route 2's memo across chunks
//        (HECKE_ROUTE2_SHARED_MEMO, el::ScopedRoute2SharedMemo) on vs off: every object at (20, 5, 3), (16, 211, 2),
//        (6, 10007, 2) u32, (40, 7, 12), (10, 101, 5) u64, the route2_degrees counters, and the SAT_DEFECT lines at (16, 101).
//   (P)  (S23 G2-P) the parallel assembly (engine::elim::module_parallel_enabled(), env HECKE_MODULE_PARALLEL, default
//        on; effective when the consumer pool has > 1 worker): the stream route run in process on consumer pools of 1, 2,
//        7 and 16 workers (ScopedConsumerPool); every object of (O) (CompareObjects) and the stream counters (Omega,
//        route-2 blocks, grouped / early-flushed GEMMs, read-offs, eval) equal pool 1's (the serial assembly), the
//        counter par=parallel on every degree at > 1 worker and par=serial at 1, no elimination-hook region at 1 worker and
//        Howell steps split over the hook at > 1 worker for n >= 40; the rollback (ScopedModuleParallel(false))
//        at 7 workers equals pool 1; at 7 workers under each of G2-T2's four lever combinations (rho fill auto / forced
//        scalar x route-2 memo shared / chunked) every object equals pool 1's (G2-T2 audit F3); at (20, 5, 3), (12, 101, 2), (60, 5, 6), (6, 10007, 2) (early flush), (8, 1291, 3) u32,
//        (40, 7, 12), (10, 101, 5) u64 and Hida's L_n (8, 5, 3); cell level, the SAT_DEFECT lines byte for byte at
//        pools 1 / 2 / 7 / 16 (and 7 with both G2-T2 levers off) at (16, 101) KMAX 6, (10, 1009) KMAX 6 and (40, 211) KMAX 4.
//        (G2-P audit F1) the parallel read-off is asserted to have run (module_parallel_readoff_regions) at (60, 5, 6) u32 and
//        (40, 7, 12) u64 on every pool > 1; (P) (N) its negative paths on a 7-worker pool: unit level over Z/5^6 u32 and Z/7^12 u64
//        (m = 1024, 8 row blocks): Ok == serial, Residual for a perturbed row in the LAST block and for X(0, 0) + 1, a residual
//        injected on a pool worker that is not the caller (el::ScopedReadoffWorkerFault(1)) -> Residual, a throw there (mode 2)
//        rethrown on the caller with its text, then Ok again; operator level at (60, 5, 6) u32 and (40, 7, 12) u64: the worker
//        residual in every parallel read-off -> the Howell fallback, every object == pool 1's; the worker throw escapes compute()
//        with its text and the same pool then reproduces pool 1; the corrupted read-off and inconsistent Theta_eq controls.
//   (H)  (S24 G2-H) the lean-peak lever (engine::elim::module_lean_peak_enabled(), env HECKE_MODULE_LEAN_PEAK, default on)
//        against its rollback (ScopedModuleLeanPeak(false)): (a) unit_pivot_residual_product's status == the product
//        read-off's on random unit-pivot inputs of five shapes (audit F1: nb = 150 > 2 * 64, 550 non-pivot rows) and eleven
//        kinds (consistent, off im B at a non-pivot / pivot row, ell^{k-1} torsion, perturb_row at a non-pivot / pivot row,
//        the corruption hook, B[piv, :] != I; audit F1: off im B at the LAST non-pivot row and at the last pivot row,
//        perturb_row at the last non-pivot row -- the last tile / hook range / pivot block), row blocks
//        1 / 3 / 64 / 256, serial and over a 7-worker hook, over Z/5^3, Z/43^2, Z/1291^3 u32, Z/7^12 u64; (b) howell /
//        kernel_module / image_module / cyclic_decomposition / invariant_factors lever on == off (the adopted store, the copy
//        and the pivot-loop length all taken) over Z/5^3, Z/2^5, Z/10007^4 u64, Z/5^30 u128; (c) the lean retention with the
//        early release at pools 1 / 7: every kept object, A^{(q)} and the certificates == the lever-off lean run == the
//        legacy run, the counters residual / route2_inplace per operator degree, early_release once, no product read-off;
//        (d) on the lean route the corrupted certificate, the inconsistent Theta_eq and the corrupted route-2 entry (four
//        sites incl. audit F2's interior block row, shared and chunked memo) give the lever-off outcome and text byte for byte; (e) the SAT_DEFECT lines with the
//        lever on at pools 1 / 4 / 16 and with HECKE_MODULE_MT off == the lever off at one worker.
//   (SP) (S24 SATPERF; HeckeSatPerfValidation, its own name) the sat cell's lever HECKE_SAT_PERF (per-level ModuleDerived
//        memo, the pre-pass handed to the certificate, EndCtx::mul_fast) on at pools 1 / 2 / 4 against ScopedSatPerf(false)
//        at one worker, at (142, 7) KMAX 5, (168, 13) KMAX 2, (60, 5) KMAX 6 (K_cls = 3, a torsion block), (40, 211) KMAX 4
//        (u32 at ell^K ~ 2^31): the SAT_DEFECT lines byte for byte, K_run / Lv_run, and the memo / narrow / split counters;
//        (b) planted modules over Z/1009^6 (r = 30: the u128 rows of mul_fast beside its uint64_t rows) and Z/7^3 (r = 40):
//        RationalSystems, every Localize block (exponents, e, ops) and its ReadAlgebra, on at pools 1 / 4 == off at one worker.
//   (R)  (S24 CL-O; CLEANUP_plan_1030.md CL-O, audits A8 / B9) an oracle INDEPENDENT of the legacy route (and of the stream
//        route and engine::elim): engine::ref (RefLinAlg's naive GEMM, Howell form and Smith rule) on the flat cochain complex
//        rebuilt from the relations and the simplicial coboundary.  It proves C^q_eq = colspan C, X, that the b_i are cocycles
//        whose classes give H^q = (+) Z/ell^{e_i} (orders, generation, length), |H^q[ell^j]|, the exponents and dK by the naive
//        Smith rule, and per T_p: Theta C = C Theta_eq, the cochain-map identity, A^{(q)} as the action on the classes b_j,
//        and Theta mod ell = the F_ell field route's Theta entrywise (block comment above RunOracle).  Run on the stream run
//        of every (O) cell (the legacy run equals it entrywise there) and on the lean objects of every (L) cell; negative
//        controls (one perturbed entry of a copy: C, X, H, an exponent, an extra summand, dK, Theta by ell and by 1,
//        Theta_eq, A) at (20, 5, 3) u32 and (10, 101, 5) u64.  One MODULE_STREAM_ORACLE line per run, ORACLE_NEGATIVE per control.
// Slow: (S23 G2-P) (P) (100, 10007) KMAX 4 at pools 1 and 8.  (S23 G2-T) (V) (c) at (100, 10007) KMAX 4, with the two runs' seconds.  (S23 G2-F) (T) the tau tables of T_10007 alone at n = 300, ChainsOnly then Full, with stored_bytes and peak RSS above
// the RSS before each; (S23 G2-C/D/E) (O) (60, 1009, 4) u64 and (L) (100, 1009, 2); then (300, 1009) and (500, 43) sat cells, stream vs legacy, byte for byte, with per-run time and peak RSS
// (VmHWM after a /proc/self/clear_refs reset; recorded, not gated) against G2_plan.md Sec. 2.6's targets.
// (S24 VAL-IMPL, VALSTREAM_audit_1030.md row 1) Every arm at (6, 10007, 2) -- (O), (T), (V) eval and memo, (P), (H) (c) --
// and the (S) cell (160, 11) KMAX 6 run in the Slow name, unchanged, not in the standard one; the standard (O) / (V) / (P) /
// (H) lists above that name (6, 10007, 2), and (S)'s (160, 11), describe the pair of names together.

#include "HeckeModuleStreamValidation.h"

#include "HeckeSaturationDefect.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeLowRank.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/RefLinAlg.h"   // (S24 CL-O) the (R) oracle
#include "../../ModArithmetic/ModCtxU128.h"
#include "../../Engine/EngineRuntime.h"   // (S23 G2-P) ScopedConsumerPool
#include "../../Engine/ThreadPool.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <functional>
#include <utility>
#include <malloc.h>   // (S23 G2-F) malloc_trim, the (T) RSS arm only (glibc; this tree builds on Linux only)

namespace {

namespace el = engine::elim;
namespace sd = hecke_ext::satdefect;
using engine::DenseBuf;

uint32_t g_fail = 0, g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  FAIL: " << what << "\n";
    }
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string Fx(double v, int prec = 2) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

template<class Elt>
bool Same(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != b.at(i, j)) return false;
    return true;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
};

template<class Elt>
Elt Below(Rng& r, Elt m) {
    if constexpr (std::is_same<Elt, engine::u128>::value) {
        const engine::u128 x = (static_cast<engine::u128>(r.next()) << 64) | r.next();
        return x % m;
    } else {
        return static_cast<Elt>(r.next() % static_cast<uint64_t>(m));
    }
}

el::ModuleStreamTotals Delta(const el::ModuleStreamTotals& a, const el::ModuleStreamTotals& b) {
    el::ModuleStreamTotals d;
    d.readoff = b.readoff - a.readoff; d.howell = b.howell - a.howell; d.legacy = b.legacy - a.legacy;
    d.dd_gemm = b.dd_gemm - a.dd_gemm; d.dd_scalar = b.dd_scalar - a.dd_scalar;
    d.cyc_interchanged = b.cyc_interchanged - a.cyc_interchanged; d.cyc_legacy = b.cyc_legacy - a.cyc_legacy;
    d.omega_streamed = b.omega_streamed - a.omega_streamed; d.omega_table = b.omega_table - a.omega_table;
    d.lr_grouped = b.lr_grouped - a.lr_grouped; d.lr_scalar = b.lr_scalar - a.lr_scalar;
    d.route2_blocked = b.route2_blocked - a.route2_blocked; d.route2_full = b.route2_full - a.route2_full;
    d.w_nonpivot = b.w_nonpivot - a.w_nonpivot; d.lean_released = b.lean_released - a.lean_released;
    d.lr_early_flush = b.lr_early_flush - a.lr_early_flush;
    d.cyc_lean = b.cyc_lean - a.cyc_lean; d.cyc_lean_legacy = b.cyc_lean_legacy - a.cyc_lean_legacy;
    d.invf_lean = b.invf_lean - a.invf_lean; d.invf_legacy = b.invf_legacy - a.invf_legacy;
    d.tau_chains = b.tau_chains - a.tau_chains; d.tau_pooled = b.tau_pooled - a.tau_pooled;
    d.eval_fast = b.eval_fast - a.eval_fast; d.eval_legacy = b.eval_legacy - a.eval_legacy;   // (S23 G2-T)
    return d;
}
std::string Txt(const el::ModuleStreamTotals& d) {
    return "readoff=" + std::to_string(d.readoff) + " howell=" + std::to_string(d.howell) + " legacy=" +
           std::to_string(d.legacy) + " dd_gemm=" + std::to_string(d.dd_gemm) + " dd_scalar=" + std::to_string(d.dd_scalar) +
           " cyc_new=" + std::to_string(d.cyc_interchanged) + " cyc_old=" + std::to_string(d.cyc_legacy) +
           " omega_streamed=" + std::to_string(d.omega_streamed) + " omega_table=" + std::to_string(d.omega_table) +
           " lr_grouped=" + std::to_string(d.lr_grouped) + " lr_scalar=" + std::to_string(d.lr_scalar) +
           " route2_blocked=" + std::to_string(d.route2_blocked) + " route2_full=" + std::to_string(d.route2_full) +
           " w_nonpivot=" + std::to_string(d.w_nonpivot) + " lean=" + std::to_string(d.lean_released) +
           " lr_early_flush=" + std::to_string(d.lr_early_flush) + " cyc_lean=" + std::to_string(d.cyc_lean) +
           " cyc_lean_legacy=" + std::to_string(d.cyc_lean_legacy) + " invf_lean=" + std::to_string(d.invf_lean) +
           " invf_legacy=" + std::to_string(d.invf_legacy) + " tau_chains=" + std::to_string(d.tau_chains) +
           " tau_pooled=" + std::to_string(d.tau_pooled) + " eval_fast=" + std::to_string(d.eval_fast) +
           " eval_legacy=" + std::to_string(d.eval_legacy);
}
uint64_t Sum(const el::ModuleStreamTotals& d) {
    return d.readoff + d.howell + d.legacy + d.dd_gemm + d.dd_scalar + d.cyc_interchanged + d.cyc_legacy + d.omega_streamed +
           d.omega_table + d.lr_grouped + d.lr_scalar + d.route2_blocked + d.route2_full + d.w_nonpivot + d.lean_released + d.lr_early_flush +
           d.cyc_lean + d.cyc_lean_legacy + d.invf_lean + d.invf_legacy + d.tau_chains + d.tau_pooled + d.eval_fast +
           d.eval_legacy;
}

// ---- (U) / (N) unit level -------------------------------------------------------------------------------
template<class Elt>
void UnitReadoff(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    Rng r(seed);
    const uint32_t m = 37, nb = 11, ny = 9;
    // piv: nb strictly increasing rows of [0, m)
    std::vector<uint32_t> piv;
    for (uint32_t i = 0; i < m && piv.size() < nb; ++i)
        if ((r.next() % 3) == 0 || m - i == nb - piv.size()) piv.push_back(i);
    DenseBuf<Elt> B(m, nb);
    std::vector<char> is_piv(m, 0);
    for (uint32_t c = 0; c < nb; ++c) is_piv[piv[c]] = 1;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t c = 0; c < nb; ++c) B.at(i, c) = Below<Elt>(r, ctx.p);
    for (uint32_t rr = 0; rr < nb; ++rr)
        for (uint32_t c = 0; c < nb; ++c) B.at(piv[rr], c) = (rr == c) ? Elt(1) : Elt(0);
    DenseBuf<Elt> Xt(nb, ny), Y(m, ny);
    for (uint32_t i = 0; i < nb; ++i)
        for (uint32_t t = 0; t < ny; ++t) Xt.at(i, t) = Below<Elt>(r, ctx.p);
    engine::kernels::gemm<Elt>(ctx, engine::cview(B), engine::cview(Xt), engine::view(Y));
    DenseBuf<Elt> X;
    const el::ReadoffStatus st = el::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(B), piv, engine::cview(Y), X, 5);
    const std::vector<std::vector<Elt>> co = el::solve_multi_rhs_module<Elt>(ctx, engine::cview(B), engine::cview(Y));
    bool eq = st == el::ReadoffStatus::Ok && X.rows == nb && X.cols == ny;
    for (uint32_t t = 0; eq && t < ny; ++t)
        for (uint32_t j = 0; j < nb; ++j) eq = eq && X.at(j, t) == co[t][j] && X.at(j, t) == Xt.at(j, t);
    Check(eq, "(U) " + tag + ": read-off == solve_multi_rhs_module == the planted X, entry for entry");
    // (S23 G2-E) the read-off on a product Y = A G (never materialised) against the read-off on Y itself:
    // A = [B | 0] (m x (nb + 3)), G = [Xt ; random], so A G = B Xt = Y; row blocks 1, 4, 7, 256.
    {
        const uint32_t l = nb + 3;
        DenseBuf<Elt> A(m, l), G(l, ny);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < l; ++j) A.at(i, j) = j < nb ? B.at(i, j) : Elt(0);
        for (uint32_t i = 0; i < l; ++i)
            for (uint32_t t = 0; t < ny; ++t) G.at(i, t) = i < nb ? Xt.at(i, t) : Below<Elt>(r, ctx.p);
        bool peq = true;
        for (uint32_t rb : {1u, 4u, 7u, 256u}) {
            DenseBuf<Elt> Xp;
            const el::ReadoffStatus sp = el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A),
                                                                                  engine::cview(G), Xp, rb);
            peq = peq && sp == el::ReadoffStatus::Ok && Same(Xp, X);
        }
        Check(peq, "(U) " + tag + ": the read-off on the product A G == the read-off on Y, for row blocks 1, 4, 7, 256");
        uint32_t npr = 0;
        while (npr < m && is_piv[npr]) ++npr;
        DenseBuf<Elt> Xq;
        Check(el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A), engine::cview(G), Xq, 4, npr) ==
                  el::ReadoffStatus::Residual,
              "(N) " + tag + ": the product read-off with Y moved off im B at a non-pivot row -> Residual");
        DenseBuf<Elt> Xc;
        el::ReadoffStatus sc;
        {
            const el::ScopedReadoffCorruption cor(true);
            sc = el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A), engine::cview(G), Xc, 4);
        }
        Check(sc == el::ReadoffStatus::Residual, "(N) " + tag + ": the product read-off with X(0, 0) + 1 -> Residual");
        DenseBuf<Elt> B4 = B;
        B4.at(piv[1], 0) = Elt(1);
        Check(el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(B4), piv, engine::cview(A), engine::cview(G), Xc) ==
                  el::ReadoffStatus::NotIdentity,
              "(N) " + tag + ": the product read-off with B[piv, :] != I -> NotIdentity");
    }
    // (N) a non-pivot row moved off im B
    uint32_t np = 0;
    while (np < m && is_piv[np]) ++np;
    DenseBuf<Elt> Y2 = Y;
    Y2.at(np, 3) = ctx.add(Y2.at(np, 3), Elt(1));
    DenseBuf<Elt> X2;
    const el::ReadoffStatus st2 = el::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(B), piv, engine::cview(Y2), X2, 7);
    Check(st2 == el::ReadoffStatus::Residual, "(N) " + tag + ": a right-hand side off im B at a non-pivot row -> Residual");
    std::string msg;
    try {
        (void)el::solve_multi_rhs_module<Elt>(ctx, engine::cview(B), engine::cview(Y2));
    } catch (const std::runtime_error& e) { msg = e.what(); }
    Check(msg == "Vector is not in image of basis (right-hand side 3).",
          "(N) " + tag + ": the Howell fallback throws the legacy message, got '" + msg + "'");
    // (N) B[piv, :] != I
    DenseBuf<Elt> B3 = B;
    B3.at(piv[2], 4) = Elt(1);
    DenseBuf<Elt> X3;
    Check(el::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(B3), piv, engine::cview(Y), X3) == el::ReadoffStatus::NotIdentity,
          "(N) " + tag + ": B[piv, :] != I -> NotIdentity");
    std::vector<uint32_t> bad = piv;
    std::swap(bad[0], bad[1]);
    Check(el::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(B), bad, engine::cview(Y), X3) == el::ReadoffStatus::NotIdentity,
          "(N) " + tag + ": non-increasing piv -> NotIdentity");
}

// (U) (S23 G2-C) the valuation by multiplication (hecke::lowrank::FastValuation) == ModCtx::valuation on every
// residue class sampled (random, ell^v u for every v <= k, 0), and factor_low_rank stream == legacy (t, U, V) on
// random rho_n blocks of matrices singular mod ell (the U_ell lever's input) and on planted low-rank matrices.
template<class Elt>
void UnitFastValuation(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    Rng r(seed);
    const hecke::lowrank::FastValuation<Elt> fv(ctx, true);
    bool eq = fv.fast;
    uint64_t n = 0;
    for (uint32_t it = 0; it < 4000; ++it) {
        Elt x = Below<Elt>(r, ctx.p);
        if (it % 2 == 1) {
            const uint32_t v = static_cast<uint32_t>(r.next() % (ctx.k + 1));
            Elt u = Below<Elt>(r, ctx.p);
            for (uint32_t s2 = 0; s2 < v; ++s2) u = ctx.mul(u, ctx.ell);
            x = u;
        }
        if (it == 0) x = 0;
        if (it == 2) x = static_cast<Elt>(ctx.p - 1);
        eq = eq && fv(x) == ctx.valuation(x);
        ++n;
    }
    Check(eq, "(U) " + tag + ": FastValuation == ModCtx::valuation on " + std::to_string(n) + " residues (fast path taken)");
    bool feq = true;
    uint32_t nontriv = 0;
    const uint32_t d = 23;
    for (uint32_t cs = 0; cs < 24; ++cs) {
        std::vector<Elt> R(static_cast<size_t>(d) * d);
        if (cs % 2 == 0) {   // rho_n of [[a, b], [ell c, ell e]] (singular mod ell)
            const Elt a = Below<Elt>(r, ctx.p), b = Below<Elt>(r, ctx.p);
            const Elt c = ctx.mul(Below<Elt>(r, ctx.p), ctx.ell), e = ctx.mul(Below<Elt>(r, ctx.p), ctx.ell);
            const FlatMatrixT<Elt> M = ComputeSymmetricPowerMatrixRing<Elt>(ctx, c, e, a, b, d - 1);
            R = hecke::lowrank::dense_rows_of<Elt>(M);
        } else {             // a planted sum of min(k, 3) outer products with forced valuations
            for (uint32_t s2 = 0; s2 < std::min<uint32_t>(ctx.k, 3); ++s2) {
                std::vector<Elt> u(d), v(d);
                for (auto& x : u) { x = Below<Elt>(r, ctx.p); for (uint32_t z = 0; z < s2; ++z) x = ctx.mul(x, ctx.ell); }
                for (auto& x : v) x = Below<Elt>(r, ctx.p);
                for (uint32_t i = 0; i < d; ++i)
                    for (uint32_t j = 0; j < d; ++j) R[static_cast<size_t>(i) * d + j] = ctx.add(R[static_cast<size_t>(i) * d + j], ctx.mul(u[i], v[j]));
            }
        }
        hecke::lowrank::LowRankT<Elt> fa, fb;
        {
            const el::ScopedModuleStream st(true);
            fa = hecke::lowrank::factor_low_rank<Elt>(ctx, std::vector<Elt>(R), d, d, d);
        }
        {
            const el::ScopedModuleStream st(false);
            fb = hecke::lowrank::factor_low_rank<Elt>(ctx, std::vector<Elt>(R), d, d, d);
        }
        feq = feq && fa.t == fb.t && fa.U == fb.U && fa.V == fb.V;
        nontriv += fa.t >= 2 ? 1 : 0;
    }
    Check(feq, "(U) " + tag + ": factor_low_rank stream == legacy (t, U, V) on 24 blocks");
    Check(nontriv > 0, "(U) " + tag + ": some block has t >= 2 (non-vacuity), " + std::to_string(nontriv));
}

template<class Elt>
void UnitCyclic(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed, uint32_t cases) {
    Rng r(seed);
    bool all = true;
    uint32_t nontrivial = 0;
    for (uint32_t cs = 0; cs < cases; ++cs) {
        const uint32_t m = 6 + static_cast<uint32_t>(r.next() % 10), ng = 3 + static_cast<uint32_t>(r.next() % 8),
                       nr = static_cast<uint32_t>(r.next() % 6);
        DenseBuf<Elt> G(m, ng), Rl(m, nr);
        // entries with forced valuations so the Smith steps meet non-unit pivots
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < ng; ++j) {
                Elt v = Below<Elt>(r, ctx.p);
                const uint32_t sh = static_cast<uint32_t>(r.next() % ctx.k);
                for (uint32_t s = 0; s < sh; ++s) v = ctx.mul(v, ctx.ell);
                G.at(i, j) = v;
            }
            for (uint32_t j = 0; j < nr; ++j) Rl.at(i, j) = 0;
        }
        // Rl = G * W (so Rl lies in span G) with a random W
        if (nr > 0) {
            for (uint32_t j = 0; j < nr; ++j)
                for (uint32_t i = 0; i < m; ++i) {
                    Elt acc = 0;
                    for (uint32_t l = 0; l < ng; ++l) {
                        Elt w = Below<Elt>(r, ctx.p);
                        if (r.next() & 1) w = ctx.mul(w, ctx.ell);
                        acc = ctx.add(acc, ctx.mul(G.at(i, l), (l * 7 + j) % 3 == 0 ? w : Elt(0)));
                    }
                    Rl.at(i, j) = acc;
                }
        }
        el::CyclicDecomposition<Elt> a, b;
        {
            const el::ScopedModuleStream s(true);
            a = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
        }
        {
            const el::ScopedModuleStream s(false);
            b = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
        }
        const bool eq = a.exponents == b.exponents && a.position == b.position && Same(a.basis, b.basis) &&
                        Same(a.Gamma, b.Gamma) && Same(a.U, b.U);
        all = all && eq;
        bool nontriv = false;
        for (uint32_t e : a.exponents) nontriv = nontriv || (e >= 1 && e < ctx.k);
        nontrivial += nontriv ? 1 : 0;
    }
    Check(all, "(U) " + tag + ": cyclic_decomposition stream == legacy on " + std::to_string(cases) + " random (G, Rl), every field");
    Check(nontrivial > 0, "(U) " + tag + ": some case has a torsion exponent 1 <= e < k (non-vacuity), " + std::to_string(nontrivial));
}

// ---- (C) (S23 G2-G) cyclic_decomposition / invariant_factors: the lean work buffers vs the legacy buffers ----------
// One random (G, Rl) per case, shapes chosen to reach every branch of the lean code: nr = 0 and nr > 0, Rl inside
// span G (the cohomology's B^q in Z^q) and Rl NOT inside span G (M = (colspan G + colspan Rl) / colspan Rl), ng = 0,
// G = Rl = 0 (s = 0: the early return), m = 1, wide (ng >> m), entries with forced valuations (Howell closures, the
// Smith steps' non-unit pivots) and a unit-heavy "cohomology-shaped" case.  Compared field by field:
// cyclic_decomposition's exponents, position, basis, Gamma, U, and invariant_factors(G, Rl) and invariant_factors(G)
// (also against the decomposition's exponents); then cyclic_coordinates of combinations of the columns of [Rl | G]
// on both decompositions.  Counters: cyc_buffers=lean / invf=lean count every stream call, =legacy every legacy call.
// (N): a field context is refused with the same composite-only text on both routes (u32 / u64); a row-count mismatch
// throws the same ENGINE_CHECK text on both; and the comparison is not vacuous -- adding ell^{k-1} to G's first column
// changes some case's result, and the comparator sees it.
template<class Elt>
DenseBuf<Elt> RandVal(const engine::ModCtx<Elt>& ctx, Rng& r, uint32_t m, uint32_t n, uint32_t mode) {
    DenseBuf<Elt> A(m, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            Elt v = Below<Elt>(r, ctx.p);
            const uint32_t sh = mode == 0 ? static_cast<uint32_t>(r.next() % ctx.k)
                                          : ((r.next() % 5) == 0 ? static_cast<uint32_t>(r.next() % ctx.k) : 0u);
            for (uint32_t z = 0; z < sh; ++z) v = ctx.mul(v, ctx.ell);
            if (mode == 2 && (r.next() % 3) == 0) v = 0;
            A.at(i, j) = v;
        }
    return A;
}

template<class Elt>
DenseBuf<Elt> MulScalar(const engine::ModCtx<Elt>& ctx, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    DenseBuf<Elt> C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) {
            Elt acc = 0;
            for (uint32_t l = 0; l < A.cols; ++l) acc = ctx.add(acc, ctx.mul(A.at(i, l), B.at(l, j)));
            C.at(i, j) = acc;
        }
    return C;
}

template<class Elt>
bool SameDec(const el::CyclicDecomposition<Elt>& a, const el::CyclicDecomposition<Elt>& b) {
    return a.exponents == b.exponents && a.position == b.position && Same(a.basis, b.basis) && Same(a.Gamma, b.Gamma) &&
           Same(a.U, b.U);
}

// The fourteen shapes of arm (C) (audit m3 adds 8..13): 0 Rl = G W (torsion in W), 1 nr = 0, 2 ng = 0, 3 m = 1, 4 wide,
// 5 G = Rl = 0 (s = 0), 6 cohomology-shaped, 7 Rl outside span G, 8 m = 0, 9 Rl = 0 with nr > 0, 10 Rl = I (M = 0 with s > 0),
// 11 G = I (full rank), 12 socle-only Rl (ell^{k-1} x), 13 socle-only G.
struct CycDims {
    uint32_t m_lo, m_span, ng_lo, ng_span, nr_span;     // m in [m_lo, m_lo + m_span), ...
    uint32_t wm_lo, wm_span, wng_lo, wng_span;          // the wide shape
};
constexpr uint32_t kCycShapes = 14;

template<class Elt>
DenseBuf<Elt> SocleOnly(const engine::ModCtx<Elt>& ctx, Rng& r, uint32_t m, uint32_t n) {
    DenseBuf<Elt> A(m, n);
    const Elt sl = ctx.ell_pow(ctx.k - 1);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = ctx.mul(sl, Below<Elt>(r, static_cast<Elt>(ctx.ell)));
    return A;
}

template<class Elt>
void CycCase(const engine::ModCtx<Elt>& ctx, Rng& r, uint32_t shape, const CycDims& dm, uint32_t& m, uint32_t& ng,
             uint32_t& nr, DenseBuf<Elt>& G, DenseBuf<Elt>& Rl) {
    m = dm.m_lo + static_cast<uint32_t>(r.next() % dm.m_span);
    ng = dm.ng_lo + static_cast<uint32_t>(r.next() % dm.ng_span);
    nr = static_cast<uint32_t>(r.next() % dm.nr_span);
    if (shape == 1) nr = 0;
    if (shape == 2) ng = 0;
    if (shape == 3) m = 1;
    if (shape == 4) {                                    // wide
        m = dm.wm_lo + static_cast<uint32_t>(r.next() % dm.wm_span);
        ng = dm.wng_lo + static_cast<uint32_t>(r.next() % dm.wng_span);
    }
    if (shape == 8) m = 0;
    if (shape == 9 && nr == 0) nr = 1;
    if (shape == 10) nr = m;
    if (shape == 11) ng = m;
    if (shape == 5 || shape == 9) {                      // 5: G = Rl = 0, the early return; 9: Rl = 0 only
        G = shape == 5 ? DenseBuf<Elt>(m, ng) : RandVal<Elt>(ctx, r, m, ng, 0);
        Rl = DenseBuf<Elt>(m, nr);
    } else if (shape == 6) {                             // cohomology-shaped: G unit-heavy, Rl = G W (inside span G)
        G = RandVal<Elt>(ctx, r, m, ng, 1);
        Rl = MulScalar<Elt>(ctx, G, RandVal<Elt>(ctx, r, ng, nr, 2));
    } else if (shape == 10 || shape == 11) {             // Rl = I / G = I, the other one random
        DenseBuf<Elt> I(m, m);
        for (uint32_t i = 0; i < m; ++i) I.at(i, i) = 1;
        G = shape == 10 ? RandVal<Elt>(ctx, r, m, ng, 0) : I;
        Rl = shape == 10 ? I : RandVal<Elt>(ctx, r, m, nr, 0);
    } else if (shape == 12 || shape == 13) {             // socle-only Rl / G, the other one random
        G = shape == 12 ? RandVal<Elt>(ctx, r, m, ng, 0) : SocleOnly<Elt>(ctx, r, m, ng);
        Rl = shape == 12 ? SocleOnly<Elt>(ctx, r, m, nr) : RandVal<Elt>(ctx, r, m, nr, 0);
    } else if (shape == 7 || shape == 3 || shape == 8) { // Rl NOT inside span G (in general)
        G = RandVal<Elt>(ctx, r, m, ng, 0);
        Rl = RandVal<Elt>(ctx, r, m, nr, 0);
    } else {                                             // Rl = G W, torsion in W
        G = RandVal<Elt>(ctx, r, m, ng, 0);
        Rl = MulScalar<Elt>(ctx, G, RandVal<Elt>(ctx, r, ng, nr, 0));
    }
}

template<class Elt>
void UnitCyclicBuffers(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed, uint32_t cases) {
    Rng r(seed);
    bool all = true, coords_eq = true;
    uint32_t torsion = 0, rl_outside = 0, early = 0, sensitive = 0, perturbed = 0, zero_mod = 0, m0 = 0;
    const el::ModuleStreamTotals c0 = el::module_stream_totals();
    const CycDims dims{4, 14, 1, 12, 7, 3, 3, 12, 8};
    for (uint32_t cs = 0; cs < cases; ++cs) {
        const uint32_t shape = cs % kCycShapes;
        uint32_t m = 0, ng = 0, nr = 0;
        DenseBuf<Elt> G, Rl;
        CycCase<Elt>(ctx, r, shape, dims, m, ng, nr, G, Rl);
        el::CyclicDecomposition<Elt> a, b;
        std::vector<uint32_t> ia, ib, ja, jb;
        {
            const el::ScopedModuleStream st(true);
            a = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            ia = el::invariant_factors<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            ja = el::invariant_factors<Elt>(ctx, engine::cview(G));
        }
        {
            const el::ScopedModuleStream st(false);
            b = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            ib = el::invariant_factors<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            jb = el::invariant_factors<Elt>(ctx, engine::cview(G));
        }
        const bool eq = SameDec(a, b) && ia == ib && ja == jb && ia == a.exponents;
        if (!eq)
            std::cout << "  (C) " << tag << " case " << cs << " (m=" << m << " ng=" << ng << " nr=" << nr << " shape=" << shape
                      << ") differs\n";
        all = all && eq;
        // (c5) on both decompositions: two combinations of the columns of [Rl | G]
        if (nr + ng > 0) {
            DenseBuf<Elt> Wc(nr + ng, 2);
            for (uint32_t j = 0; j < nr + ng; ++j)
                for (uint32_t t = 0; t < 2; ++t) Wc.at(j, t) = Below<Elt>(r, ctx.p);
            DenseBuf<Elt> N(m, nr + ng);
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t j = 0; j < nr; ++j) N.at(i, j) = Rl.at(i, j);
                for (uint32_t j = 0; j < ng; ++j) N.at(i, nr + j) = G.at(i, j);
            }
            const DenseBuf<Elt> Y = MulScalar<Elt>(ctx, N, Wc);
            coords_eq = coords_eq && el::cyclic_coordinates<Elt>(ctx, a, engine::cview(Y)) ==
                                         el::cyclic_coordinates<Elt>(ctx, b, engine::cview(Y));
        }
        for (uint32_t e : a.exponents)
            if (e >= 1 && e < ctx.k) { ++torsion; break; }
        if (shape == 7 && nr > 0) ++rl_outside;
        if (a.Gamma.cols == 0) ++early;
        if (a.Gamma.cols > 0 && a.exponents.empty()) ++zero_mod;
        if (m == 0) ++m0;
        // non-vacuity of the comparator: G's first column + ell^{k-1}
        if (ng > 0 && shape != 5) {
            DenseBuf<Elt> G2 = G;
            for (uint32_t i = 0; i < m; ++i) G2.at(i, 0) = ctx.add(G2.at(i, 0), ctx.ell_pow(ctx.k - 1));
            el::CyclicDecomposition<Elt> c;
            {
                const el::ScopedModuleStream st(true);
                c = el::cyclic_decomposition<Elt>(ctx, engine::cview(G2), engine::cview(Rl));
            }
            ++perturbed;
            if (!SameDec(a, c)) ++sensitive;
        }
    }
    const el::ModuleStreamTotals d = Delta(c0, el::module_stream_totals());
    Check(all, "(C) " + tag + ": cyclic_decomposition (exponents, position, basis, Gamma, U) and invariant_factors (G, Rl) / (G), "
               "lean buffers == legacy buffers on " + std::to_string(cases) + " cases");
    Check(coords_eq, "(C) " + tag + ": cyclic_coordinates on the lean decomposition == on the legacy one");
    Check(cases >= kCycShapes && torsion > 0 && rl_outside > 0 && early > 0 && zero_mod > 0 && m0 > 0,
          "(C) " + tag + ": non-vacuity: all " + std::to_string(kCycShapes) + " shapes, torsion exponents " +
              std::to_string(torsion) + ", Rl outside span G " + std::to_string(rl_outside) + ", s = 0 " + std::to_string(early) +
              ", M = 0 with s > 0 " + std::to_string(zero_mod) + ", m = 0 " + std::to_string(m0));
    Check(sensitive > 0, "(N) " + tag + ": adding ell^{k-1} to G's first column changes some decomposition (the comparator is "
                         "not vacuous), " + std::to_string(sensitive) + " of " + std::to_string(perturbed));
    Check(d.cyc_lean == cases + perturbed && d.cyc_lean_legacy == cases && d.invf_lean == 2 * cases && d.invf_legacy == 2 * cases,
          "(C) " + tag + ": counters cyc_buffers=lean:" + std::to_string(d.cyc_lean) + ",legacy:" + std::to_string(d.cyc_lean_legacy) +
              " invf=lean:" + std::to_string(d.invf_lean) + ",legacy:" + std::to_string(d.invf_legacy));
    // (N) refusals: the same text on both routes
    auto text_of = [](auto&& f) {
        std::string t;
        try { f(); } catch (const std::exception& e) { t = e.what(); }
        return t;
    };
    const DenseBuf<Elt> Gf(3, 2), Rbad(4, 1);
    std::string ra, rb;
    {
        const el::ScopedModuleStream st(true);
        ra = text_of([&] { (void)el::cyclic_decomposition<Elt>(ctx, engine::cview(Gf), engine::cview(Rbad)); }) + "|" +
             text_of([&] { (void)el::invariant_factors<Elt>(ctx, engine::cview(Gf), engine::cview(Rbad)); });
    }
    {
        const el::ScopedModuleStream st(false);
        rb = text_of([&] { (void)el::cyclic_decomposition<Elt>(ctx, engine::cview(Gf), engine::cview(Rbad)); }) + "|" +
             text_of([&] { (void)el::invariant_factors<Elt>(ctx, engine::cview(Gf), engine::cview(Rbad)); });
    }
    Check(ra.size() > 1 && ra == rb && ra.find("same row count") != std::string::npos,
          "(N) " + tag + ": a row-count mismatch throws the same text on both routes: '" + ra + "'");
    if constexpr (!std::is_same<Elt, engine::u128>::value) {
        const engine::ModCtx<Elt> field(static_cast<Elt>(ctx.ell));
        const DenseBuf<Elt> Rf(3, 1);
        std::string fa, fb;
        {
            const el::ScopedModuleStream st(true);
            fa = text_of([&] { (void)el::cyclic_decomposition<Elt>(field, engine::cview(Gf), engine::cview(Rf)); }) + "|" +
                 text_of([&] { (void)el::invariant_factors<Elt>(field, engine::cview(Gf), engine::cview(Rf)); });
        }
        {
            const el::ScopedModuleStream st(false);
            fb = text_of([&] { (void)el::cyclic_decomposition<Elt>(field, engine::cview(Gf), engine::cview(Rf)); }) + "|" +
                 text_of([&] { (void)el::invariant_factors<Elt>(field, engine::cview(Gf), engine::cview(Rf)); });
        }
        Check(fa.size() > 1 && fa == fb && fa.find("composite-only") != std::string::npos,
              "(N) " + tag + ": the field context F_ell is refused with the same composite-only text on both routes");
    }
}

// (C) (S23 G2-G audit m2) THE INDEPENDENT ORACLE on tiny modules: N = colspan[Rl | G] and R = colspan Rl are enumerated
// in (Z/ell^k)^m, and |M[ell^j]| = |{y in N : ell^j y in R}| / |R| must equal prod_i ell^{min(e_i, j)} for every
// 0 <= j <= k, for the exponents of the lean cyclic_decomposition and of the lean invariant_factors (the default route; the
// legacy route is compared with them field by field in UnitCyclicBuffers).  Its own control: the exponent list with one more
// exponent 1 appended is rejected on EVERY case with M != 0 (|M[ell]| moves).  FAULT INJECTION on the lean route
// (el::ScopedLeanBufferCorruption: relation matrix entry (0, 0) + 1): the corrupted decomposition must be caught -- by
// its own sum-of-exponents check (it throws), by the comparison with the legacy route, or by the oracle -- on some case,
// for each of the two entry points; the oracle itself must catch some corruption that passes the step-4 check (pooled
// over the oracle rings); and the scope must leave the legacy route unchanged.
template<class Elt>
std::set<std::vector<Elt>> BruteSpan(const engine::ModCtx<Elt>& ctx, const DenseBuf<Elt>& A) {
    const uint32_t m = A.rows;
    std::set<std::vector<Elt>> S{std::vector<Elt>(m, Elt(0))};
    std::vector<std::vector<Elt>> todo{std::vector<Elt>(m, Elt(0))};
    while (!todo.empty()) {
        const std::vector<Elt> x = todo.back();
        todo.pop_back();
        for (uint32_t j = 0; j < A.cols; ++j) {
            std::vector<Elt> y = x;
            for (uint32_t i = 0; i < m; ++i) y[i] = ctx.add(y[i], A.at(i, j));
            if (S.insert(y).second) todo.push_back(y);
        }
    }
    return S;
}

// true iff |M[ell^j]| = prod_i ell^{min(e_i, j)} for every 0 <= j <= k
template<class Elt>
bool OracleAgrees(const engine::ModCtx<Elt>& ctx, const std::set<std::vector<Elt>>& SN, const std::set<std::vector<Elt>>& SR,
                  const std::vector<uint32_t>& e) {
    Elt lj = 1;
    for (uint32_t j = 0; j <= ctx.k; ++j) {
        uint64_t cnt = 0;
        for (const auto& y : SN) {
            std::vector<Elt> z = y;
            for (auto& t : z) t = ctx.mul(t, lj);
            if (SR.count(z) != 0) ++cnt;
        }
        uint64_t want = 1;
        for (uint32_t x : e)
            for (uint32_t t = 0; t < std::min(x, j); ++t) want *= static_cast<uint64_t>(ctx.ell);
        if (cnt % SR.size() != 0 || cnt / SR.size() != want) return false;
        if (j < ctx.k) lj = ctx.mul(lj, static_cast<Elt>(ctx.ell));
    }
    return true;
}

// The oracle's own catches of an injected fault (corruptions that pass the step-4 check), pooled over the oracle rings.
uint32_t g_oracle_injection_caught = 0, g_oracle_injection_cases = 0;

template<class Elt>
void UnitCyclicOracle(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed, uint32_t cases) {
    Rng r(seed);
    const CycDims dims{1, 3, 1, 4, 4, 2, 2, 5, 3};     // m <= 3 (m = 0 by shape 8), ng <= 7, nr <= 3
    uint32_t agree = 0, nonzero = 0, ctrl_rejected = 0, inj_cyc = 0, inj_invf = 0, inj_oracle = 0, inj_throw = 0;
    bool legacy_inert = true;
    for (uint32_t cs = 0; cs < cases; ++cs) {
        uint32_t m = 0, ng = 0, nr = 0;
        DenseBuf<Elt> G, Rl;
        CycCase<Elt>(ctx, r, cs % kCycShapes, dims, m, ng, nr, G, Rl);
        el::CyclicDecomposition<Elt> a, b;
        std::vector<uint32_t> ia, ib;
        {
            const el::ScopedModuleStream st(true);
            a = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            ia = el::invariant_factors<Elt>(ctx, engine::cview(G), engine::cview(Rl));
        }
        {
            const el::ScopedModuleStream st(false);
            b = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            ib = el::invariant_factors<Elt>(ctx, engine::cview(G), engine::cview(Rl));
        }
        DenseBuf<Elt> N(m, nr + ng);
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < nr; ++j) N.at(i, j) = Rl.at(i, j);
            for (uint32_t j = 0; j < ng; ++j) N.at(i, nr + j) = G.at(i, j);
        }
        const auto SN = BruteSpan<Elt>(ctx, N), SR = BruteSpan<Elt>(ctx, Rl);
        const bool ok = OracleAgrees<Elt>(ctx, SN, SR, a.exponents) && OracleAgrees<Elt>(ctx, SN, SR, ia);
        if (!ok)
            std::cout << "  (C) oracle " << tag << " case " << cs << " (m=" << m << " ng=" << ng << " nr=" << nr << " shape="
                      << cs % kCycShapes << "): |M[ell^j]| differs from the exponents\n";
        if (ok) ++agree;
        if (SN.size() != SR.size()) {
            ++nonzero;
            std::vector<uint32_t> e1 = ia;
            e1.push_back(1);
            if (!OracleAgrees<Elt>(ctx, SN, SR, e1)) ++ctrl_rejected;
        }
        // fault injection: the lean route corrupted, the legacy route untouched
        const el::ScopedLeanBufferCorruption cor(true);
        {
            bool threw = false;
            el::CyclicDecomposition<Elt> c;
            {
                const el::ScopedModuleStream st(true);
                try { c = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl)); } catch (const std::exception&) { threw = true; }
            }
            if (threw) ++inj_throw;
            else if (!OracleAgrees<Elt>(ctx, SN, SR, c.exponents)) ++inj_oracle;
            if (threw || !SameDec(c, b)) ++inj_cyc;
        }
        {
            bool threw = false;
            std::vector<uint32_t> ic;
            {
                const el::ScopedModuleStream st(true);
                try { ic = el::invariant_factors<Elt>(ctx, engine::cview(G), engine::cview(Rl)); } catch (const std::exception&) { threw = true; }
            }
            if (threw) ++inj_throw;
            else if (!OracleAgrees<Elt>(ctx, SN, SR, ic)) ++inj_oracle;
            if (threw || ic != ib) ++inj_invf;
        }
        {
            const el::ScopedModuleStream st(false);
            const el::CyclicDecomposition<Elt> d = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            const std::vector<uint32_t> id = el::invariant_factors<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            legacy_inert = legacy_inert && SameDec(d, b) && id == ib;
        }
    }
    Check(agree == cases, "(C) oracle " + tag + ": |M[ell^j]| (brute force over (Z/ell^k)^m, m <= 3) == prod ell^{min(e_i, j)}, "
                          "0 <= j <= k, for the lean cyclic_decomposition and invariant_factors, on " + std::to_string(agree) +
                          " of " + std::to_string(cases) + " cases (" + std::to_string(kCycShapes) + " shapes)");
    Check(nonzero > 0 && ctrl_rejected == nonzero,
          "(N) oracle " + tag + ": the exponent list with one more exponent 1 is rejected on every case with M != 0, " +
              std::to_string(ctrl_rejected) + " of " + std::to_string(nonzero));
    g_oracle_injection_caught += inj_oracle;
    g_oracle_injection_cases += cases;
    std::cout << "MODULE_STREAM_ORACLE " << tag << " cases=" << cases << " agree=" << agree << " nonzero=" << nonzero
              << " ctrl_rejected=" << ctrl_rejected << " inj_cyc=" << inj_cyc << " inj_invf=" << inj_invf << " inj_throw=" << inj_throw
              << " inj_oracle=" << inj_oracle << " legacy_inert=" << (legacy_inert ? 1 : 0) << "\n";
    Check(inj_cyc > 0 && inj_invf > 0,
          "(N) " + tag + ": fault injection on the lean route (relation matrix (0, 0) + 1) is detected: cyclic_decomposition " +
              std::to_string(inj_cyc) + ", invariant_factors " + std::to_string(inj_invf) + " of " + std::to_string(cases) +
              " cases (vs legacy; thrown by the step-4 check " + std::to_string(inj_throw) + ", rejected by the oracle " +
              std::to_string(inj_oracle) + ")");
    Check(legacy_inert, "(N) " + tag + ": the lean-route fault injection leaves the legacy route unchanged");
}

// ---- (O) object level ------------------------------------------------------------------------------------
template<class Elt>
struct ObjRun {
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh;
    std::vector<std::unique_ptr<hecke::HeckeCosetTablesT<Elt>>> cosets;
    std::vector<std::unique_ptr<hecke::HeckeTauTableT<Elt>>> tau;
    std::vector<std::unique_ptr<hecke::HeckeOperatorOnKT<Elt>>> ops;
    std::vector<uint64_t> primes;
    el::ModuleStreamTotals delta;
    double seconds = 0.0;
};

template<class Elt>
std::unique_ptr<ObjRun<Elt>> RunObjects(uint32_t n, uint64_t ell, uint32_t K, bool stream, FibreRepresentation rep,
                                        bool corrupt = false, bool lean = false, bool eval = true) {
    auto r = std::make_unique<ObjRun<Elt>>();
    const double t0 = Now();
    const el::ScopedModuleStream s(stream);
    const el::ScopedModuleEval se(eval);   // (S23 G2-T) consulted by the stream route only
    std::unique_ptr<el::ScopedReadoffCorruption> cor;
    if (corrupt) cor.reset(new el::ScopedReadoffCorruption(true));
    const el::ModuleStreamTotals before = el::module_stream_totals();
    const hecke::ScopedCoefficientRing ring(K);
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        r->Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        r->model.reset(new hecke::SL2ZTessellationTables(*r->Kx));
    }
    const engine::ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeModuleStreamValidation", ell, K);
    r->lssc.reset(new LocalSystemSimplicialComplexT<Elt>(
        *r->Kx,
        rep == FibreRepresentation::HidaLatticeH ? hecke::GenerateSL2ZRelationsRingHida<Elt>(*r->Kx, n, ctx)
                                                 : hecke::GenerateSL2ZRelationsRing<Elt>(*r->Kx, n, ctx),
        n + 1, ctx.p));
    r->coh.reset(new EquivariantCohomologyOnKT<Elt>(r->lssc.get()));
    if (lean) r->coh->set_module_early_release(true);   // (S24 G2-H) as RunModuleT does (inert with the lever off)
    r->coh->compute_cohomology();
    if (lean) r->coh->release_module_intermediates();   // (S23 G2-E) as RunModuleT does
    r->primes = {2, 3, 5, 7};
    if (ell > 7) r->primes.push_back(ell);
    for (uint64_t p : r->primes) {
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            r->cosets.emplace_back(new hecke::HeckeCosetTablesT<Elt>(p, 1, static_cast<Elt>(ell), n));
            // (S23 G2-F) the lean run builds its tau tables as RunModuleT does on the stream route: chain records only
            r->tau.emplace_back(new hecke::HeckeTauTableT<Elt>(
                *r->Kx, *r->model, *r->cosets.back(), static_cast<Elt>(ell), n, hecke::TauVertexPolicy::GeometricNearest,
                lean ? hecke::TauTableContents::ChainsOnly : hecke::TauTableContents::Full));
        }
        r->ops.emplace_back(new hecke::HeckeOperatorOnKT<Elt>(r->coh.get(), r->cosets.back().get(), r->tau.back().get(), rep));
        if (lean) r->ops.back()->set_module_retention(hecke::ModuleRetention::Lean);
        r->ops.back()->compute();
    }
    r->delta = Delta(before, el::module_stream_totals());
    r->seconds = Now() - t0;
    return r;
}

// ---- (R) (S24 CL-O) the independent oracle -------------------------------------------------------------------------
// Checks the module route's results against the DEFINITIONS, by engine::ref (RefLinAlg: naive mod_*_w GEMM, the naive
// Howell form and the naive Smith rule) and by schoolbook loops here -- never through the legacy module route, the
// stream route or engine::elim.  The flat cochain complex is rebuilt from the relations and the simplicial coboundary
// (A_Cq: I at s_lhs, -M at s_rhs, I - M when equal; D_q = D_1 (x) I), and every claim is proved by lengths
// (length(colspan M) = sum of (K - v) over the Howell pivots of M or M^T: one Smith form, so the two agree):
//   (R1) A_Cq C = 0;  (R2) length(colspan C) = c_q K = length(ker A_Cq) = dim_q K - length(A_Cq)  [so colspan C = C^q_eq];
//   (R3) D_q C_q = C_{q+1} X_q  (X determined: C_{q+1} is injective);
//   (R4) A_Cq H = 0 and D_q H = 0  (the b_i are equivariant cocycles);
//   (R5) ell^{e_i} b_i in B^q = colspan(D_{q-1} C_{q-1}) for every i;
//   (R6) length(B + colspan H) = length(Z^q) = c_q K - length(D_q C_q), and sum e_i = length(Z) - length(B) = length_H:
//        so (+)_i Z/ell^{e_i} -> H^q, b_i |-> [b_i], is onto between modules of the same length, an isomorphism;
//   (R7) |H^q[ell^j]| = ell^{sum_i min(e_i, j)} for 0 < j < K, counted as length(Z) - length(ell^j Z + B);
//   (R8) ref::invariant_factors(H, B) = (e_i) and (R9) ref::invariant_factors(D_q C_q) = dK_q (the naive Smith rule);
// and per Hecke operator T_p, q = 0, 1, 2:
//   (T1) Theta C = C Theta_eq;  (T2) D_q Theta_q C_q = Theta_{q+1} D_q C_q (a cochain map on C_eq);
//   (T3) 0 <= A_ij < ell^{e_i} and Theta H - H A in B^q column by column (T_p b_j = sum_i A_ij b_i in H^q);
//   (T4) Theta mod ell = the F_ell operator's flat Theta (the field route, HeckeOperatorOnKT<uint32_t>), entrywise.
// Its negative controls perturb one entry of a copy of the inputs (C, X, H, an exponent, dK, Theta by ell and by 1,
// Theta_eq, A) and require a rejection each.  HeckeSaturationDefect's sat cells are not re-derived here (they read
// these objects through RunModuleT): see the catalogue.
namespace rl = engine::ref;

template<class Elt>
struct OracleDegree {
    DenseBuf<Elt> C, X, H;
    std::vector<uint32_t> piv, exps, dK;
    uint32_t length_H = 0;
    bool have_X = false;
};
template<class Elt>
struct OracleOp {
    uint64_t p = 0;
    std::vector<DenseBuf<Elt>> Theta, Theta_eq, A;   // Theta / Theta_eq empty when released (lean)
};
template<class Elt>
struct OracleInput {
    std::vector<OracleDegree<Elt>> deg;
    std::vector<OracleOp<Elt>> ops;
};
using FieldThetas = std::vector<std::vector<DenseBuf<uint32_t>>>;   // [operator][q], the F_ell flat Theta

template<class Elt>
Elt OMul(Elt a, Elt b, Elt p) {
    return static_cast<Elt>(static_cast<unsigned __int128>(a) * b % p);
}

// a + b and a - b mod p without overflow (p up to 2^32 - 1 in the u32 word: 1291^3 > 2^31).
template<class Elt>
Elt OAdd(Elt a, Elt b, Elt p) {
    return static_cast<Elt>((static_cast<unsigned __int128>(a) + b) % p);
}
template<class Elt>
Elt OSub(Elt a, Elt b, Elt p) {
    return static_cast<Elt>((static_cast<unsigned __int128>(a) + (p - b % p)) % p);
}

template<class Elt>
DenseBuf<Elt> OProd(Elt p, const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {
    ENGINE_CHECK(A.cols == B.rows);
    DenseBuf<Elt> R(A.rows, B.cols);
    if (A.rows && B.cols) {
        if (A.cols == 0) {
            for (uint32_t i = 0; i < R.rows; ++i)
                for (uint32_t j = 0; j < R.cols; ++j) R.at(i, j) = 0;
        } else {
            rl::gemm<Elt>(p, engine::cview(A), engine::cview(B), engine::view(R), 0);
        }
    }
    return R;
}

template<class Elt>
bool OZero(const DenseBuf<Elt>& A) {
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j) != 0) return false;
    return true;
}

template<class Elt>
DenseBuf<Elt> OCat(const DenseBuf<Elt>& A, const DenseBuf<Elt>& B) {   // [A | B]
    ENGINE_CHECK(A.rows == B.rows);
    DenseBuf<Elt> R(A.rows, A.cols + B.cols);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) R.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) R.at(i, A.cols + j) = B.at(i, j);
    }
    return R;
}

template<class Elt>
DenseBuf<Elt> OScaleCols(Elt p, const DenseBuf<Elt>& A, const std::vector<Elt>& s) {   // column j times s[j]
    DenseBuf<Elt> R(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) R.at(i, j) = OMul<Elt>(A.at(i, j), s[j], p);
    return R;
}

// length(colspan M) over Z/ell^K by ref::howell on the shorter orientation (a row span and a column span of one
// matrix have the same length: one Smith form).
template<class Elt>
uint32_t OLen(Elt ell, uint32_t K, const DenseBuf<Elt>& M) {
    if (M.rows == 0 || M.cols == 0) return 0;
    rl::RefHowell<Elt> h;
    if (M.rows <= M.cols) {
        h = rl::howell<Elt>(ell, K, engine::cview(M));
    } else {
        DenseBuf<Elt> T(M.cols, M.rows);
        for (uint32_t i = 0; i < M.rows; ++i)
            for (uint32_t j = 0; j < M.cols; ++j) T.at(j, i) = M.at(i, j);
        h = rl::howell<Elt>(ell, K, engine::cview(T));
    }
    uint32_t len = 0;
    for (uint32_t v : h.pivot_v) len += K - v;
    return len;
}

template<class Elt>
DenseBuf<Elt> OColumn(const DenseBuf<Elt>& A, uint32_t j) {
    DenseBuf<Elt> R(A.rows, 1);
    for (uint32_t i = 0; i < A.rows; ++i) R.at(i, 0) = A.at(i, j);
    return R;
}

// The flat A_Cq of the relations of degree q (every relation, the vacuous ones give zero rows).
template<class Elt>
DenseBuf<Elt> OAC(const LocalSystemSimplicialComplexT<Elt>& lssc, uint32_t q, uint32_t N_q, uint32_t d, Elt p) {
    const auto& rels = lssc.get_user_defined_simplex_relations();
    uint32_t nr = 0;
    for (const auto& rel : rels) nr += rel.q_dim == q ? 1u : 0u;
    DenseBuf<Elt> A(nr * d, N_q * d);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) A.at(i, j) = 0;
    uint32_t r = 0;
    for (const auto& rel : rels) {
        if (rel.q_dim != q) continue;
        const uint32_t L = rel.s_lhs_idx_in_Kq, R = rel.s_rhs_idx_in_Kq;
        ENGINE_CHECK(rel.M.get_rows() == d && rel.M.get_cols() == d && L < N_q && R < N_q);
        for (uint32_t a = 0; a < d; ++a) {
            A.at(r * d + a, L * d + a) = OAdd<Elt>(A.at(r * d + a, L * d + a), 1, p);
            for (uint32_t b = 0; b < d; ++b) {
                Elt& x = A.at(r * d + a, R * d + b);
                x = OSub<Elt>(x, static_cast<Elt>(rel.M.get_entry(a, b) % p), p);
            }
        }
        ++r;
    }
    return A;
}

// The flat coboundary D_q = D_1 (x) I_d (N_{q+1} d x N_q d).
template<class Elt>
DenseBuf<Elt> OD(const SimplicialComplex& Kx, uint32_t q, uint32_t d, Elt p) {
    const FlatMatrixT<Elt> D1 = Kx.template coboundary_operator_elt<Elt>(q, p);
    const uint32_t Ns = Kx.get_num_q_simplices(q), Nd = Kx.get_num_q_simplices(q + 1);
    ENGINE_CHECK(D1.get_rows() == Nd && D1.get_cols() == Ns);
    DenseBuf<Elt> D(Nd * d, Ns * d);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) D.at(i, j) = 0;
    for (uint32_t i = 0; i < Nd; ++i)
        for (uint32_t j = 0; j < Ns; ++j) {
            const Elt s = static_cast<Elt>(D1.get_entry(i, j) % p);
            if (s)
                for (uint32_t c = 0; c < d; ++c) D.at(i * d + c, j * d + c) = s;
        }
    return D;
}

// The fixed part of the oracle: the flat complex of one cell (independent of every route).
template<class Elt>
struct OracleComplex {
    Elt ell = 0, p = 0;
    uint32_t K = 0, d = 0;
    std::vector<uint32_t> dim;
    std::vector<DenseBuf<Elt>> AC, D;   // D[q] for q = 0, 1
    std::vector<uint32_t> lenAC;
};
template<class Elt>
OracleComplex<Elt> BuildOracleComplex(const ObjRun<Elt>& r, uint64_t ell, uint32_t K) {
    OracleComplex<Elt> oc;
    oc.ell = static_cast<Elt>(ell);
    oc.K = K;
    Elt p = 1;
    for (uint32_t i = 0; i < K; ++i) p = static_cast<Elt>(p * oc.ell);
    oc.p = p;
    oc.d = r.lssc->get_representation_dim();
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t Nq = r.Kx->get_num_q_simplices(q);
        oc.dim.push_back(Nq * oc.d);
        oc.AC.push_back(OAC<Elt>(*r.lssc, q, Nq, oc.d, p));
        oc.lenAC.push_back(OLen<Elt>(oc.ell, K, oc.AC.back()));
        if (q < 2) oc.D.push_back(OD<Elt>(*r.Kx, q, oc.d, p));
    }
    return oc;
}

template<class Elt>
OracleInput<Elt> OracleInputOf(const ObjRun<Elt>& r, bool lean) {
    OracleInput<Elt> in;
    for (uint32_t q = 0; q < 3; ++q) {
        const auto& M = r.coh->module_degree(q);
        OracleDegree<Elt> g;
        g.C = M.C;
        g.piv = M.piv;
        g.H = M.H;
        g.exps = M.dec.exponents;
        g.dK = M.dK;
        g.length_H = M.length_H;
        g.have_X = !lean;
        if (!lean) g.X = M.X;
        in.deg.push_back(std::move(g));
    }
    for (size_t i = 0; i < r.ops.size(); ++i) {
        OracleOp<Elt> o;
        o.p = r.primes[i];
        for (uint32_t q = 0; q < 3; ++q) {
            o.A.push_back(r.ops[i]->H_matrix_module(q));
            if (!lean) {
                o.Theta.push_back(r.ops[i]->Theta_module(q));
                o.Theta_eq.push_back(r.ops[i]->Theta_eq_module(q));
            }
        }
        in.ops.push_back(std::move(o));
    }
    return in;
}

// The F_ell operators of the same cell (the field route: GenerateSL2ZRelationsWithGroupData, HeckeOperatorOnKT<uint32_t>).
FieldThetas FieldThetasOf(uint32_t n, uint64_t ell, const std::vector<uint64_t>& primes) {
    FieldThetas out;
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const uint32_t e = static_cast<uint32_t>(ell);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*Kx, n, e)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(*Kx, rels, n + 1, e);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    for (uint64_t p : primes) {
        std::unique_ptr<hecke::HeckeCosetTablesT<uint32_t>> cos;
        std::unique_ptr<hecke::HeckeTauTableT<uint32_t>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cos.reset(new hecke::HeckeCosetTablesT<uint32_t>(p, 1, e, n));
            tau.reset(new hecke::HeckeTauTableT<uint32_t>(*Kx, *model, *cos, e, n, hecke::TauVertexPolicy::GeometricNearest,
                                                          hecke::TauTableContents::Full));
        }
        hecke::HeckeOperatorOnKT<uint32_t> op(&coh, cos.get(), tau.get());
        op.compute();
        std::vector<DenseBuf<uint32_t>> th;
        for (const auto& F : op.get_Theta_flat_matrices()) {
            DenseBuf<uint32_t> b(F.get_rows(), F.get_cols());
            for (uint32_t i = 0; i < b.rows; ++i)
                for (uint32_t j = 0; j < b.cols; ++j) b.at(i, j) = F.get_entry(i, j);
            th.push_back(std::move(b));
        }
        out.push_back(std::move(th));
    }
    return out;
}

// One verdict per oracle clause; the caller Checks them (positive) or requires a false one (negative control).
struct OracleVerdicts {
    std::vector<std::pair<std::string, bool>> v;
    void add(const std::string& w, bool ok) { v.emplace_back(w, ok); }
    std::string failed() const {
        std::string s;
        for (const auto& x : v)
            if (!x.second) s += (s.empty() ? "" : "; ") + x.first;
        return s;
    }
};

template<class Elt>
OracleVerdicts RunOracle(const OracleComplex<Elt>& oc, const OracleInput<Elt>& in, const FieldThetas* field,
                         const std::vector<DenseBuf<Elt>>* theta_from = nullptr /* [op * 3 + q], lean inputs */) {
    OracleVerdicts out;
    const Elt ell = oc.ell, p = oc.p;
    const uint32_t K = oc.K;
    std::vector<DenseBuf<Elt>> B(3), DC(3);
    std::vector<uint32_t> LB(3), LZ(3);
    for (uint32_t q = 0; q < 3; ++q) {
        const OracleDegree<Elt>& g = in.deg[q];
        const std::string tq = "q=" + std::to_string(q);
        const uint32_t cq = g.C.cols;
        out.add(tq + " (R1) A_C C = 0", g.C.rows == oc.dim[q] && OZero(OProd<Elt>(p, oc.AC[q], g.C)));
        const uint32_t lc = OLen<Elt>(ell, K, g.C);
        out.add(tq + " (R2) colspan C = ker A_C, free of rank c_q (lengths " + std::to_string(lc) + ")",
                lc == cq * K && lc + oc.lenAC[q] == oc.dim[q] * K);
        if (q < 2) DC[q] = OProd<Elt>(p, oc.D[q], g.C);
        if (q < 2 && g.have_X)
            out.add(tq + " (R3) D C = C' X", g.X.rows == in.deg[q + 1].C.cols && g.X.cols == cq &&
                                                 Same(DC[q], OProd<Elt>(p, in.deg[q + 1].C, g.X)));
        B[q] = q == 0 ? DenseBuf<Elt>(oc.dim[0], 0) : DC[q - 1];
        LB[q] = OLen<Elt>(ell, K, B[q]);
        LZ[q] = cq * K - (q < 2 ? OLen<Elt>(ell, K, DC[q]) : 0u);
        const DenseBuf<Elt>& H = g.H;
        const uint32_t t = H.cols;
        bool shape = H.rows == oc.dim[q] && g.exps.size() == t;
        out.add(tq + " (R4) the b_i are equivariant cocycles",
                shape && OZero(OProd<Elt>(p, oc.AC[q], H)) && (q == 2 || OZero(OProd<Elt>(p, oc.D[q], H))));
        if (!shape) continue;
        bool orders = true;
        uint32_t sum = 0;
        for (uint32_t i = 0; i < t; ++i) {
            const uint32_t e = g.exps[i];
            sum += e;
            if (e < 1 || e > K) { orders = false; continue; }
            Elt pe = 1;
            for (uint32_t s = 0; s < e; ++s) pe = static_cast<Elt>(pe * ell);   // ell^e (= 0 mod p at e = K)
            pe = static_cast<Elt>(pe % p);
            const DenseBuf<Elt> v = OScaleCols<Elt>(p, OColumn(H, i), std::vector<Elt>{pe});
            orders = orders && OLen<Elt>(ell, K, OCat(B[q], v)) == LB[q];
        }
        out.add(tq + " (R5) ell^{e_i} b_i in B", orders);
        const uint32_t lg = OLen<Elt>(ell, K, OCat(B[q], H));
        out.add(tq + " (R6) B + span(b) = Z, sum e_i = length(Z / B) = length_H (" + std::to_string(sum) + ", " +
                    std::to_string(LZ[q]) + " - " + std::to_string(LB[q]) + ")",
                lg == LZ[q] && LZ[q] >= LB[q] && sum == LZ[q] - LB[q] && sum == g.length_H);
        bool tors = true;
        Elt pj = 1;
        for (uint32_t j = 1; j < K; ++j) {
            pj = static_cast<Elt>(pj * ell);
            uint32_t want = 0;
            for (uint32_t e : g.exps) want += std::min(e, j);
            const uint32_t lj = OLen<Elt>(ell, K, OCat(OScaleCols<Elt>(p, H, std::vector<Elt>(t, pj)), B[q]));
            tors = tors && LZ[q] >= lj && LZ[q] - lj == want;
        }
        out.add(tq + " (R7) |H[ell^j]| = ell^{sum min(e_i, j)}, 0 < j < K", tors);
        std::vector<uint32_t> se = g.exps;
        std::sort(se.begin(), se.end(), [](uint32_t a, uint32_t b) { return a > b; });
        out.add(tq + " (R8) ref::invariant_factors(H, B) = (e_i)",
                se == g.exps && rl::invariant_factors<Elt>(ell, K, engine::cview(H), engine::cview(B[q])) == g.exps);
        if (q < 2) {
            const DenseBuf<Elt> none(DC[q].rows, 0);
            out.add(tq + " (R9) ref::invariant_factors(D C) = dK",
                    rl::invariant_factors<Elt>(ell, K, engine::cview(DC[q]), engine::cview(none)) == g.dK);
        } else {
            out.add(tq + " (R9) dK empty at the top degree", g.dK.empty());
        }
    }
    for (size_t o = 0; o < in.ops.size(); ++o) {
        const OracleOp<Elt>& op = in.ops[o];
        const std::string tp = "T_" + std::to_string(op.p);
        auto theta = [&](uint32_t q) -> const DenseBuf<Elt>& {
            return theta_from ? (*theta_from)[o * 3 + q] : op.Theta[q];
        };
        const bool have_theta = theta_from != nullptr || op.Theta.size() == 3;
        std::vector<DenseBuf<Elt>> TC(3);
        for (uint32_t q = 0; q < 3; ++q) {
            const OracleDegree<Elt>& g = in.deg[q];
            const std::string tq = tp + " q=" + std::to_string(q);
            if (have_theta) {
                const DenseBuf<Elt>& Th = theta(q);
                const bool sq = Th.rows == oc.dim[q] && Th.cols == oc.dim[q];
                out.add(tq + " Theta shape", sq);
                if (!sq) continue;
                TC[q] = OProd<Elt>(p, Th, g.C);
                if (op.Theta_eq.size() == 3)
                    out.add(tq + " (T1) Theta C = C Theta_eq", op.Theta_eq[q].rows == g.C.cols &&
                                                                  Same(TC[q], OProd<Elt>(p, g.C, op.Theta_eq[q])));
            }
            const DenseBuf<Elt>& A = op.A[q];
            const uint32_t t = g.H.cols;
            bool ok = A.rows == t && A.cols == t && g.exps.size() == t;
            if (ok)
                for (uint32_t i = 0; i < t; ++i) {
                    Elt pe = 1;
                    for (uint32_t s = 0; s < g.exps[i]; ++s) pe = static_cast<Elt>(pe * ell);
                    for (uint32_t j = 0; j < t; ++j) ok = ok && (g.exps[i] >= K || A.at(i, j) < pe);
                }
            if (ok && have_theta && t > 0) {
                const DenseBuf<Elt> TH = OProd<Elt>(p, theta(q), g.H), HA = OProd<Elt>(p, g.H, A);
                DenseBuf<Elt> Y(TH.rows, t);
                for (uint32_t i = 0; i < TH.rows; ++i)
                    for (uint32_t j = 0; j < t; ++j) Y.at(i, j) = OSub<Elt>(TH.at(i, j), HA.at(i, j), p);
                ok = OLen<Elt>(ell, K, OCat(B[q], Y)) == LB[q];
            }
            out.add(tq + " (T3) T_p b_j = sum_i A_ij b_i in H^q, A_ij < ell^{e_i}", ok);
            if (field != nullptr && have_theta) {
                const DenseBuf<uint32_t>& F = (*field)[o][q];
                const DenseBuf<Elt>& Th = theta(q);
                bool eq = F.rows == Th.rows && F.cols == Th.cols;
                for (uint32_t i = 0; eq && i < F.rows; ++i)
                    for (uint32_t j = 0; eq && j < F.cols; ++j) eq = static_cast<uint64_t>(Th.at(i, j) % ell) == F.at(i, j);
                out.add(tq + " (T4) Theta mod ell = the F_ell operator's flat Theta", eq);
            }
        }
        if (have_theta)
            for (uint32_t q = 0; q < 2; ++q)
                if (TC[q].rows && TC[q + 1].rows)
                    out.add(tp + " q=" + std::to_string(q) + " (T2) D Theta_q C = Theta_{q+1} D C",
                            Same(OProd<Elt>(p, oc.D[q], TC[q]), OProd<Elt>(p, theta(q + 1), DC[q])));
    }
    return out;
}

// The positive arm: every clause Checked, one ORACLE line.
void OracleReport(const std::string& t, const OracleVerdicts& v, double seconds) {
    uint32_t bad = 0;
    for (const auto& x : v.v) {
        Check(x.second, t + " oracle " + x.first);
        bad += x.second ? 0u : 1u;
    }
    std::cout << "MODULE_STREAM_ORACLE " << t << " clauses=" << v.v.size() << " failing=" << bad << " seconds=" << Fx(seconds, 3)
              << "\n";
}

// The negative controls: each perturbation of one entry of a copy must be rejected.
template<class Elt>
void OracleNegatives(const std::string& t, const OracleComplex<Elt>& oc, const OracleInput<Elt>& in, const FieldThetas* field) {
    const Elt ell = oc.ell, p = oc.p;
    auto bump = [p](Elt& x, Elt by) { x = OAdd<Elt>(x, static_cast<Elt>(by % p), p); };
    struct Kind { std::string name; std::function<bool(OracleInput<Elt>&)> f; };
    std::vector<Kind> kinds;
    auto pow_ell = [ell](uint32_t e) {
        Elt r = 1;
        for (uint32_t s = 0; s < e; ++s) r = static_cast<Elt>(r * ell);   // ell^e <= p fits the word
        return r;
    };
    kinds.push_back({"C(0, 0) + 1 at q=1", [&](OracleInput<Elt>& x) {
        auto& C = x.deg[1].C;
        if (!C.rows || !C.cols) return false;
        bump(C.at(0, 0), 1);
        return true;
    }});
    kinds.push_back({"X(0, 0) + ell^{K-1} at q=0", [&](OracleInput<Elt>& x) {
        auto& X = x.deg[0].X;
        if (!x.deg[0].have_X || !X.rows || !X.cols) return false;
        bump(X.at(0, 0), pow_ell(oc.K - 1));
        return true;
    }});
    kinds.push_back({"H(piv_0, 0) + 1 at q=1", [&](OracleInput<Elt>& x) {
        auto& g = x.deg[1];
        if (!g.H.cols) return false;
        bump(g.H.at(g.piv.empty() ? 0 : g.piv[0], 0), 1);
        return true;
    }});
    kinds.push_back({"e_0 - 1 at q=1", [&](OracleInput<Elt>& x) {
        auto& e = x.deg[1].exps;
        if (e.empty() || e[0] < 2) return false;
        --e[0];
        return true;
    }});
    kinds.push_back({"an extra summand b = 0, e = 1 at q=1", [&](OracleInput<Elt>& x) {
        auto& g = x.deg[1];
        DenseBuf<Elt> zero(g.H.rows, 1);
        for (uint32_t i = 0; i < g.H.rows; ++i) zero.at(i, 0) = 0;
        g.H = OCat(g.H, zero);
        g.exps.push_back(1);
        g.length_H += 1;
        return true;
    }});
    kinds.push_back({"dK_0 + 1 at q=0", [&](OracleInput<Elt>& x) {
        auto& v = x.deg[0].dK;
        if (v.empty()) return false;
        ++v[0];
        return true;
    }});
    kinds.push_back({"Theta(piv_0, piv_0) + ell at q=1 (invisible mod ell)", [&](OracleInput<Elt>& x) {
        const auto& g = x.deg[1];
        if (x.ops.empty() || x.ops[0].Theta.size() != 3 || g.piv.empty()) return false;
        bump(x.ops[0].Theta[1].at(g.piv[0], g.piv[0]), ell);
        return true;
    }});
    kinds.push_back({"Theta(0, 0) + 1 at q=0", [&](OracleInput<Elt>& x) {
        if (x.ops.empty() || x.ops[0].Theta.size() != 3) return false;
        bump(x.ops[0].Theta[0].at(0, 0), 1);
        return true;
    }});
    kinds.push_back({"Theta_eq(0, 0) + 1 at q=1", [&](OracleInput<Elt>& x) {
        if (x.ops.empty() || x.ops[0].Theta_eq.size() != 3 || !x.ops[0].Theta_eq[1].rows) return false;
        bump(x.ops[0].Theta_eq[1].at(0, 0), 1);
        return true;
    }});
    kinds.push_back({"A(0, 0) + 1 at q=1, the last operator", [&](OracleInput<Elt>& x) {
        if (x.ops.empty() || !x.ops.back().A[1].rows) return false;
        auto& A = x.ops.back().A[1];
        const Elt pe = pow_ell(x.deg[1].exps[0]);   // stays in [0, ell^{e_0}): only (T3)'s membership can see it
        A.at(0, 0) = static_cast<Elt>((static_cast<unsigned __int128>(A.at(0, 0)) + 1) % pe);
        return true;
    }});
    uint32_t ran = 0;
    for (const auto& k : kinds) {
        OracleInput<Elt> x = in;
        if (!k.f(x)) {
            std::cout << "  ORACLE_NEGATIVE " << t << " '" << k.name << "' not applicable\n";
            continue;
        }
        ++ran;
        const OracleVerdicts v = RunOracle<Elt>(oc, x, field);
        const std::string f = v.failed();
        Check(!f.empty(), t + " oracle negative control '" + k.name + "' rejected");
        std::cout << "  ORACLE_NEGATIVE " << t << " '" << k.name << "' rejected_by={" << f << "}\n";
    }
    Check(ran >= 8, t + " oracle negative controls: at least 8 of " + std::to_string(kinds.size()) + " applicable (" +
                        std::to_string(ran) + ")");
}

// (R) on a full stream run (and, with `neg`, its negative controls); `field` selects (T4).
template<class Elt>
void OracleOnRun(const std::string& t, const ObjRun<Elt>& s, uint32_t n, uint64_t ell, uint32_t K, bool with_field, bool neg) {
    const double t0 = Now();
    const OracleComplex<Elt> oc = BuildOracleComplex<Elt>(s, ell, K);
    const OracleInput<Elt> in = OracleInputOf<Elt>(s, false);
    FieldThetas field;
    if (with_field) field = FieldThetasOf(n, ell, s.primes);
    const OracleVerdicts v = RunOracle<Elt>(oc, in, with_field ? &field : nullptr);
    OracleReport(t, v, Now() - t0);
    if (neg) OracleNegatives<Elt>(t, oc, in, with_field ? &field : nullptr);
}

// (R) on a LEAN stream run (X, Zc, Theta, Theta_eq released): every clause but (R3), (T1) and (T4), on the Theta of a
// full stream run `f` of the same cell (which OracleOnRun has checked by (T1) / (T2) / (T4)).
template<class Elt>
void OracleOnLeanRun(const std::string& t, const ObjRun<Elt>& ln, const ObjRun<Elt>& f, uint64_t ell, uint32_t K) {
    const double t0 = Now();
    const OracleComplex<Elt> oc = BuildOracleComplex<Elt>(ln, ell, K);
    std::vector<DenseBuf<Elt>> th;
    for (const auto& op : f.ops)
        for (uint32_t q = 0; q < 3; ++q) th.push_back(op->Theta_module(q));
    const OracleVerdicts v = RunOracle<Elt>(oc, OracleInputOf<Elt>(ln, true), nullptr, &th);
    OracleReport(t, v, Now() - t0);
}

// (R) alone, no legacy run (the Slow name's share of the standard (O) / (L) cells: their legacy comparison stays in the
// standard name): a full stream run, and with `lean` the lean objects as in LeanArm.
template<class Elt>
void OracleObjectArm(uint32_t n, uint64_t ell, uint32_t K, bool lean = false) {
    const std::string t = std::string("(R) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    const auto f = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
    OracleOnRun<Elt>(t, *f, n, ell, K, true, false);
    if (lean) {
        const auto ln = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, false, /*lean=*/true);
        OracleOnLeanRun<Elt>(t + " lean", *ln, *f, ell, K);
    }
}

template<class Elt>
void CompareObjects(const std::string& t, const ObjRun<Elt>& a, const ObjRun<Elt>& b) {
    const uint32_t nq = a.coh->num_degrees();
    Check(nq == 3 && b.coh->num_degrees() == 3, t + " three degrees");
    for (uint32_t q = 0; q < nq && q < 3; ++q) {
        const auto& A = a.coh->module_degree(q);
        const auto& B = b.coh->module_degree(q);
        const std::string tq = t + " q=" + std::to_string(q);
        Check(Same(A.C, B.C) && A.piv == B.piv, tq + " C, piv");
        Check(Same(A.X, B.X), tq + " X");
        Check(Same(A.Zc, B.Zc), tq + " Zc");
        Check(Same(A.H, B.H), tq + " H");
        Check(A.dec.exponents == B.dec.exponents && A.dec.position == B.dec.position && Same(A.dec.basis, B.dec.basis) &&
                  Same(A.dec.Gamma, B.dec.Gamma) && Same(A.dec.U, B.dec.U),
              tq + " dec (exponents, position, basis, Gamma, U)");
        Check(A.dK == B.dK && A.length_H == B.length_H, tq + " dK, length_H");
    }
    Check(a.ops.size() == b.ops.size(), t + " operator counts");
    for (size_t i = 0; i < a.ops.size() && i < b.ops.size(); ++i) {
        const auto& x = *a.ops[i];
        const auto& y = *b.ops[i];
        const std::string tp = t + " T_" + std::to_string(a.primes[i]);
        for (uint32_t q = 0; q < 3; ++q) {
            const std::string tq = tp + " q=" + std::to_string(q);
            Check(Same(x.H_matrix_module(q), y.H_matrix_module(q)), tq + " A^{(q)}");
            Check(Same(x.Theta_module(q), y.Theta_module(q)), tq + " Theta");
            Check(Same(x.Theta_eq_module(q), y.Theta_eq_module(q)), tq + " Theta_eq");
        }
        const auto ca = x.certificates(), cb = y.certificates();
        Check(ca.routes_bit_identical && cb.routes_bit_identical && ca.relation_checks == cb.relation_checks &&
                  ca.cocycle_checks == cb.cocycle_checks && ca.theta_eq_solves == cb.theta_eq_solves &&
                  ca.theta_eq_residual_max == cb.theta_eq_residual_max && ca.theta_eq_residual_max == 0 &&
                  ca.theta_eq_certified == cb.theta_eq_certified,
              tp + " certificate bundle (O12(a), (i), (ii), (iii), residual)");
        Check(ca.theta_eq_solves > 0, tp + " (iii) ran (non-vacuity)");
    }
}

template<class Elt>
// (S24 CL-O) oracle: 0 = none (the cell's (R) runs elsewhere), 1 = (R) on the stream run, 2 = (R) + its negative controls.
void ObjectArm(uint32_t n, uint64_t ell, uint32_t K, FibreRepresentation rep = FibreRepresentation::SymN, int oracle = 1) {
    const bool hida = rep == FibreRepresentation::HidaLatticeH;
    const std::string t = std::string("(O) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (hida ? " Hida L_n" : "") + (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    const double rss0 = oc_util::PeakRssMiB();
    const auto s = RunObjects<Elt>(n, ell, K, true, rep);
    const auto l = RunObjects<Elt>(n, ell, K, false, rep);
    CompareObjects<Elt>(t, *s, *l);
    // (S24 CL-O) the independent oracle on the stream run (the legacy run equals it entrywise, above); the F_ell
    // comparison (T4) on SymN only (the field route has no Hida lattice)
    if (oracle > 0) OracleOnRun<Elt>(t, *s, n, ell, K, !hida, oracle > 1);
    // non-vacuity / routing
    Check(s->delta.readoff >= 1 && s->delta.legacy == 0 && s->delta.dd_gemm >= 1 && s->delta.dd_scalar == 0 &&
              s->delta.cyc_interchanged >= 1 && s->delta.cyc_legacy == 0 && s->delta.cyc_lean >= 1 &&
              s->delta.cyc_lean_legacy == 0 && s->delta.invf_lean >= 1 && s->delta.invf_legacy == 0,
          t + " stream run: read-off, d o d GEMM, interchanged clear, lean decomposition buffers taken; no legacy step (" +
              Txt(s->delta) + ")");
    Check(l->delta.readoff == 0 && l->delta.howell == 0 && l->delta.legacy >= 1 && l->delta.dd_gemm == 0 && l->delta.dd_scalar >= 1 &&
              l->delta.cyc_interchanged == 0 && l->delta.cyc_legacy >= 1 && l->delta.cyc_lean == 0 &&
              l->delta.cyc_lean_legacy >= 1 && l->delta.invf_lean == 0 && l->delta.invf_legacy >= 1,
          t + " legacy run: only legacy steps (" + Txt(l->delta) + ")");
    if (hida)
        Check(s->delta.howell >= 1, t + " the non-unit-pivot fallback is taken (howell >= 1) (" + Txt(s->delta) + ")");
    else
        Check(s->delta.howell == 0, t + " SymN: every solve is a read-off (howell == 0) (" + Txt(s->delta) + ")");
    // (S23 G2-C / G2-D / G2-E) the stream assembly: Omega per (degree, coset) at every coset of every operator
    // (sum over p of 3 (p + 1)), route 2 per sigma-row block (7 + 11 + 5 = 23 simplices per operator), the product
    // read-off on the SymN Theta_eq solves; the grouped low-rank GEMM iff T_ell (ell | N) is among the operators.
    uint64_t cosets = 0;
    for (uint64_t p : s->primes) cosets += 3 * (p + 1);
    const uint64_t blocks = 23ull * s->primes.size();
    const bool lr = !hida;   // ell is always among the primes {2, 3, 5, 7, ell}: T_ell is low-rank (SymN only)
    Check(s->delta.omega_streamed == cosets && s->delta.omega_table == 0 && s->delta.route2_blocked == blocks &&
              s->delta.route2_full == 0 && s->delta.lr_scalar == 0 && (lr ? s->delta.lr_grouped >= 1 : s->delta.lr_grouped == 0) &&
              (hida ? true : s->delta.w_nonpivot >= 1) && s->delta.lean_released == 0,
          t + " stream run: Omega streamed at every coset (" + std::to_string(cosets) + "), route 2 blocked (" +
              std::to_string(blocks) + "), grouped low-rank GEMM iff T_ell, product read-off (" + Txt(s->delta) + ")");
    Check(l->delta.omega_streamed == 0 && l->delta.omega_table == 3 * s->primes.size() && l->delta.route2_blocked == 0 &&
              l->delta.route2_full == 3 * s->primes.size() && l->delta.lr_grouped == 0 && l->delta.lr_scalar == (lr ? 3u : 0u) &&
              l->delta.w_nonpivot == 0 && l->delta.lean_released == 0,
          t + " legacy run: the Omega table, route 2 materialised, scalar low-rank iff T_ell (" + Txt(l->delta) + ")");
    // (audit m2) the kGroupCap early flush: never at ell <= 101 (the chains are short), always at (6, 10007, 2)
    if (ell <= 101)
        Check(s->delta.lr_early_flush == 0 && l->delta.lr_early_flush == 0, t + " no early flush at ell <= 101 (" + Txt(s->delta) + ")");
    if (ell == 10007)
        Check(s->delta.lr_early_flush >= 1 && l->delta.lr_early_flush == 0,
              t + " the kGroupCap early flush is reached on the stream run (" + Txt(s->delta) + ")");
    // (S23 G2-T) the module-eval lever: on (default) on the stream run, three operator degrees per operator; never consulted
    // by the legacy run
    Check(s->delta.eval_fast == 3 * s->primes.size() && s->delta.eval_legacy == 0 && l->delta.eval_fast == 0 &&
              l->delta.eval_legacy == 0,
          t + " eval=fast on the stream run (3 per operator), no eval counter on the legacy run (" + Txt(s->delta) + "; " +
              Txt(l->delta) + ")");
    std::cout << "MODULE_STREAM_CELL " << t << " seconds stream=" << Fx(s->seconds, 3) << " legacy=" << Fx(l->seconds, 3)
              << " peak_rss_mib=" << Fx(std::max(rss0, oc_util::PeakRssMiB()), 1) << " stream_counts{" << Txt(s->delta) << "}\n";
}

// ---- (V) (S23 G2-T) the module-eval lever ---------------------------------------------------------------------------
// (a) ComputeSymmetricPowerRowsRing == ComputeSymmetricPowerMatrixRing, entry for entry, on keys of every route.
template<class Elt>
void UnitRhoRows(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    Rng r(seed);
    bool eq = true;
    uint64_t cases = 0, route_n[4] = {0, 0, 0, 0};
    std::vector<Elt> rows, work;
    for (uint32_t n : {0u, 1u, 2u, 7u, 8u, 9u, 30u, 64u}) {
        const std::vector<Elt> pascal = RingPascalRow<Elt>(ctx, n);
        for (uint32_t it = 0; it < 48; ++it) {
            Elt a = Below<Elt>(r, ctx.p), b = Below<Elt>(r, ctx.p), c = Below<Elt>(r, ctx.p), e = Below<Elt>(r, ctx.p);
            switch (it % 6) {
                case 1: a = ctx.mul(a, ctx.ell); break;                                   // [U_b] (b a unit, mostly)
                case 2: a = ctx.mul(a, ctx.ell); b = ctx.mul(b, ctx.ell); break;          // [S]
                case 3: a = ctx.mul(ctx.mul(a, ctx.ell), ctx.ell); b = ctx.mul(b, ctx.ell); break;   // [S], deeper
                case 4: a = 0; b = 0; break;                                              // [Z]
                case 5: c = ctx.mul(c, ctx.ell); e = ctx.mul(e, ctx.ell); break;          // singular mod ell
                default: break;
            }
            RhoRingRoute route = RhoRingRoute::UnitA;
            const FlatMatrixT<Elt> M = ComputeSymmetricPowerMatrixRing<Elt>(ctx, a, b, c, e, n, &route);
            ComputeSymmetricPowerRowsRing<Elt>(ctx, a, b, c, e, n, pascal, rows, work);
            eq = eq && rows == hecke::lowrank::dense_rows_of<Elt>(M);
            ++route_n[static_cast<uint32_t>(route)];
            ++cases;
        }
    }
    Check(eq, "(V) " + tag + ": ComputeSymmetricPowerRowsRing == ComputeSymmetricPowerMatrixRing entrywise on " +
                  std::to_string(cases) + " keys");
    Check(route_n[0] > 0 && route_n[1] > 0 && route_n[2] > 0 && route_n[3] > 0,
          "(V) " + tag + ": every route taken (unit_a " + std::to_string(route_n[0]) + ", unit_b " + std::to_string(route_n[1]) +
              ", split " + std::to_string(route_n[2]) + ", first_row_zero " + std::to_string(route_n[3]) + ")");
}

// (V) (d) (S23 G2-T2) the rho fill paths (RhoFillPath, HECKE_RHO_SIMD): every path available on this CPU / build
// (scalar, AVX2, AVX-512F) forced by ScopedRhoFillPath, ComputeSymmetricPowerRowsRing against
// ComputeSymmetricPowerMatrixRing (the G2-F route, no SIMD) entry for entry into a dirty buffer, at every n in 0..140
// (every tail length of the 8-, 32- and 64-lane wavefronts) and n in {191, 192, 193, 255, 256, 257, 300}, on keys of
// every route plus the all-(p - 1) key, a key with w = 0 and one with t = 0; the counters: on a SIMD ring (u32,
// p <= 2^31) every [U] fill took the forced path, elsewhere (u32 p > 2^31, u64) every one took the scalar path.
const std::vector<RhoFillPath>& AvailableRhoPaths() {
    static const std::vector<RhoFillPath> v = [] {
        std::vector<RhoFillPath> r = {RhoFillPath::Scalar};
        if (rho_fill_path_available(RhoFillPath::Avx2)) r.push_back(RhoFillPath::Avx2);
        if (rho_fill_path_available(RhoFillPath::Avx512)) r.push_back(RhoFillPath::Avx512);
        return r;
    }();
    return v;
}
template<class Elt>
void UnitRhoPaths(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    const bool simd_ring = std::is_same<Elt, uint32_t>::value && static_cast<uint64_t>(ctx.p) <= (uint64_t(1) << 31);
    std::vector<uint32_t> ns;
    for (uint32_t n = 0; n <= 140; ++n) ns.push_back(n);
    for (uint32_t n : {191u, 192u, 193u, 255u, 256u, 257u, 300u}) ns.push_back(n);
    const std::vector<RhoFillPath>& paths = AvailableRhoPaths();
    std::vector<char> eq(paths.size(), 1);
    std::vector<RhoFillCounts> moved(paths.size());
    uint64_t cases = 0;
    Rng r(seed);
    std::vector<Elt> rows, work;
    for (uint32_t n : ns) {
        const std::vector<Elt> pascal = RingPascalRow<Elt>(ctx, n);
        const uint32_t nkeys = n <= 140 ? 9u : 4u;
        for (uint32_t it = 0; it < nkeys; ++it) {
            Elt a = Below<Elt>(r, ctx.p), b = Below<Elt>(r, ctx.p), c = Below<Elt>(r, ctx.p), e = Below<Elt>(r, ctx.p);
            switch (it) {
                case 1: a = ctx.mul(a, ctx.ell); break;                                     // [U_b] (mostly)
                case 2: a = ctx.mul(a, ctx.ell); b = ctx.mul(b, ctx.ell); break;            // [S]
                case 3: a = b = c = e = static_cast<Elt>(ctx.p - 1); break;                 // all -1 ([U_a])
                case 4: a = 0; b = 0; break;                                                // [Z]
                case 5: c = 0; break;                                                       // w = 0 (a unit: [U_a])
                case 6: e = 0; break;                                                       // t = 0 (a unit: [U_a])
                case 7: a = ctx.mul(a, ctx.ell); e = static_cast<Elt>(ctx.p - 1); break;    // [U_b], t near -1
                case 8: c = ctx.mul(c, ctx.ell); e = ctx.mul(e, ctx.ell); break;            // singular mod ell
                default: break;
            }
            // the oracle once per key (the G2-F route, no SIMD), then every path forced in turn
            const std::vector<Elt> M = hecke::lowrank::dense_rows_of<Elt>(ComputeSymmetricPowerMatrixRing<Elt>(ctx, a, b, c, e, n));
            for (size_t pi = 0; pi < paths.size(); ++pi) {
                const ScopedRhoFillPath sp(paths[pi]);
                const RhoFillCounts c0 = rho_fill_counts();
                rows.assign(static_cast<size_t>(n + 1) * (n + 1), static_cast<Elt>(ctx.p - 1));   // a dirty buffer
                ComputeSymmetricPowerRowsRing<Elt>(ctx, a, b, c, e, n, pascal, rows, work);
                const RhoFillCounts c1 = rho_fill_counts();
                moved[pi].avx512 += c1.avx512 - c0.avx512;
                moved[pi].avx2 += c1.avx2 - c0.avx2;
                moved[pi].scalar += c1.scalar - c0.scalar;
                eq[pi] = eq[pi] && rows == M;
            }
            ++cases;
        }
    }
    for (size_t pi = 0; pi < paths.size(); ++pi) {
        const RhoFillPath path = paths[pi];
        const uint64_t d512 = moved[pi].avx512, d2 = moved[pi].avx2, ds = moved[pi].scalar;
        const std::string t = "(V) " + tag + " path=" + to_text(path);
        Check(eq[pi] != 0, t + ": ComputeSymmetricPowerRowsRing == ComputeSymmetricPowerMatrixRing entrywise on " +
                               std::to_string(cases) + " keys");
        bool routed;
        if (simd_ring && path == RhoFillPath::Avx512) routed = d512 > 0 && d2 == 0 && ds == 0;
        else if (simd_ring && path == RhoFillPath::Avx2) routed = d2 > 0 && d512 == 0 && ds == 0;
        else routed = ds > 0 && d2 == 0 && d512 == 0;
        Check(routed, t + ": every [U] fill took " + (simd_ring ? to_text(path) : "scalar") + " (fills avx512 " +
                          std::to_string(d512) + ", avx2 " + std::to_string(d2) + ", scalar " + std::to_string(ds) + ")");
    }
}

// (a) factor_low_rank with fast_scan (divisibility scan, valuation floor, rank-1 GEMM update) == the per-entry scan.
template<class Elt>
void UnitFactorScan(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    Rng r(seed);
    bool feq = true;
    uint32_t nontriv = 0, blocks = 0;
    for (uint32_t d : {1u, 5u, 23u, 40u}) {
        for (uint32_t cs = 0; cs < 16; ++cs) {
            std::vector<Elt> R(static_cast<size_t>(d) * d, Elt(0));
            if (cs % 2 == 0) {   // rho_n of a matrix singular mod ell: [[ell c, ell e], [a, b]] or with a split
                const Elt a = Below<Elt>(r, ctx.p), b = Below<Elt>(r, ctx.p);
                Elt c = ctx.mul(Below<Elt>(r, ctx.p), ctx.ell), e = ctx.mul(Below<Elt>(r, ctx.p), ctx.ell);
                if (cs % 4 == 2) c = ctx.mul(c, ctx.ell);
                R = hecke::lowrank::dense_rows_of<Elt>(ComputeSymmetricPowerMatrixRing<Elt>(ctx, c, e, a, b, d - 1));
            } else {             // a planted sum of min(k, 3) outer products, term s2 of valuation >= s2 (+1 when cs % 4 == 3)
                for (uint32_t s2 = 0; s2 < std::min<uint32_t>(ctx.k, 3); ++s2) {
                    std::vector<Elt> u(d), v(d);
                    const uint32_t vv = s2 + (cs % 4 == 3 ? 1u : 0u);
                    for (auto& x : u) { x = Below<Elt>(r, ctx.p); for (uint32_t z = 0; z < vv; ++z) x = ctx.mul(x, ctx.ell); }
                    for (auto& x : v) x = Below<Elt>(r, ctx.p);
                    for (uint32_t i = 0; i < d; ++i)
                        for (uint32_t j = 0; j < d; ++j)
                            R[static_cast<size_t>(i) * d + j] = ctx.add(R[static_cast<size_t>(i) * d + j], ctx.mul(u[i], v[j]));
                }
            }
            const el::ScopedModuleStream st(true);   // FastValuation on (the module route's setting)
            const hecke::lowrank::LowRankT<Elt> fa = hecke::lowrank::factor_low_rank<Elt>(ctx, std::vector<Elt>(R), d, d, d, true);
            const hecke::lowrank::LowRankT<Elt> fb = hecke::lowrank::factor_low_rank<Elt>(ctx, std::vector<Elt>(R), d, d, d, false);
            feq = feq && fa.t == fb.t && fa.U == fb.U && fa.V == fb.V && fa.rows == fb.rows && fa.cols == fb.cols;
            nontriv += fa.t >= 2 ? 1 : 0;
            ++blocks;
        }
    }
    Check(feq, "(V) " + tag + ": factor_low_rank fast_scan == per-entry scan (t, U, V) on " + std::to_string(blocks) + " blocks");
    if (ctx.k >= 2) Check(nontriv > 0, "(V) " + tag + ": some block has t >= 2 (non-vacuity), " + std::to_string(nontriv));
}

// (b) operator level: the stream route with HECKE_MODULE_EVAL on against off, every object and the counters.
template<class Elt>
void EvalObjectArm(uint32_t n, uint64_t ell, uint32_t K, FibreRepresentation rep = FibreRepresentation::SymN) {
    const bool hida = rep == FibreRepresentation::HidaLatticeH;
    const std::string t = std::string("(V) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (hida ? " Hida L_n" : "") + (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " eval on vs off";
    const auto f = RunObjects<Elt>(n, ell, K, true, rep, false, false, /*eval=*/true);
    const auto o = RunObjects<Elt>(n, ell, K, true, rep, false, false, /*eval=*/false);
    CompareObjects<Elt>(t, *f, *o);
    const uint64_t deg = 3 * f->primes.size();
    Check(f->delta.eval_fast == deg && f->delta.eval_legacy == 0 && o->delta.eval_fast == 0 && o->delta.eval_legacy == deg,
          t + ": eval=fast on the on-run, eval=legacy on the off-run (" + std::to_string(deg) + " operator degrees each) (" +
              Txt(f->delta) + "; " + Txt(o->delta) + ")");
    Check(f->delta.omega_streamed == o->delta.omega_streamed && f->delta.lr_grouped == o->delta.lr_grouped &&
              f->delta.route2_blocked == o->delta.route2_blocked && f->delta.lr_early_flush == o->delta.lr_early_flush,
          t + ": the same assembly (omega, grouped GEMMs, route-2 blocks, early flushes) on both runs");
    std::cout << "MODULE_STREAM_CELL " << t << " seconds eval_on=" << Fx(f->seconds, 3) << " eval_off=" << Fx(o->seconds, 3) << "\n";
}

// (V) (e) (S23 G2-T2) operator level: the stream route (eval on) with each available rho fill path against the scalar
// path, every object of (O) (CompareObjects); the counters as in (d).
template<class Elt>
void PathObjectArm(uint32_t n, uint64_t ell, uint32_t K) {
    unsigned __int128 pr = 1;
    for (uint32_t i = 0; i < K; ++i) pr *= ell;
    const bool simd_ring = std::is_same<Elt, uint32_t>::value && pr <= (static_cast<unsigned __int128>(1) << 31);
    std::unique_ptr<ObjRun<Elt>> ref;
    {
        const ScopedRhoFillPath sp(RhoFillPath::Scalar);
        ref = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
    }
    for (RhoFillPath path : AvailableRhoPaths()) {
        if (path == RhoFillPath::Scalar) continue;
        const std::string t = std::string("(V) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) +
                              ")" + (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " rho path " + to_text(path) +
                              " vs scalar";
        const RhoFillCounts c0 = rho_fill_counts();
        std::unique_ptr<ObjRun<Elt>> o;
        {
            const ScopedRhoFillPath sp(path);
            o = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
        }
        const RhoFillCounts c1 = rho_fill_counts();
        CompareObjects<Elt>(t, *o, *ref);
        const uint64_t dsimd = path == RhoFillPath::Avx512 ? c1.avx512 - c0.avx512 : c1.avx2 - c0.avx2;
        const uint64_t dother = path == RhoFillPath::Avx512 ? c1.avx2 - c0.avx2 : c1.avx512 - c0.avx512;
        const uint64_t ds = c1.scalar - c0.scalar;
        Check(simd_ring ? (dsimd > 0 && dother == 0 && ds == 0) : (dsimd == 0 && dother == 0 && ds > 0),
              t + ": the fills took " + (simd_ring ? to_text(path) : "scalar") + " (" + std::to_string(dsimd) + " " +
                  to_text(path) + ", " + std::to_string(dother) + " other SIMD, " + std::to_string(ds) + " scalar)");
        std::cout << "MODULE_STREAM_CELL " << t << " seconds path=" << Fx(o->seconds, 3) << " scalar=" << Fx(ref->seconds, 3) << "\n";
    }
}

// (V) (g) (S23 G2-T2) operator level: the route-2 memo lever (HECKE_ROUTE2_SHARED_MEMO) on against off
// (el::ScopedRoute2SharedMemo), every object of (O) (CompareObjects); the counters: the on-run's q = 1 degrees (N_1 > 7)
// all shared, the off-run's all chunked, the same route-2 block rows and grouped GEMMs on both.
template<class Elt>
void MemoObjectArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t = std::string("(V) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " route-2 memo shared vs chunked";
    const el::Route2MemoCounts a0 = el::route2_memo_counts();
    std::unique_ptr<ObjRun<Elt>> sh, ch;
    {
        const el::ScopedRoute2SharedMemo m(true);
        sh = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
    }
    const el::Route2MemoCounts a1 = el::route2_memo_counts();
    {
        const el::ScopedRoute2SharedMemo m(false);
        ch = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
    }
    const el::Route2MemoCounts a2 = el::route2_memo_counts();
    CompareObjects<Elt>(t, *sh, *ch);
    const uint64_t ops = sh->primes.size();
    Check(a1.shared - a0.shared == ops && a1.chunked == a0.chunked && a2.chunked - a1.chunked == ops && a2.shared == a1.shared,
          t + ": route2_degrees shared " + std::to_string(a1.shared - a0.shared) + " on the on-run, chunked " +
              std::to_string(a2.chunked - a1.chunked) + " on the off-run (" + std::to_string(ops) + " operators each)");
    Check(sh->delta.route2_blocked == ch->delta.route2_blocked && sh->delta.omega_streamed == ch->delta.omega_streamed &&
              sh->delta.eval_fast == ch->delta.eval_fast,
          t + ": the same route-2 block rows, Omega evaluations and eval counters on both runs");
    std::cout << "MODULE_STREAM_CELL " << t << " seconds shared=" << Fx(sh->seconds, 3) << " chunked=" << Fx(ch->seconds, 3) << "\n";
}

// (N) operator level: one read-off entry +1 before the residual -> the residual fires -> Howell -> unchanged.
template<class Elt>
void CorruptionArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t = "(N) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " corrupted read-off";
    const auto c = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, /*corrupt=*/true);
    const auto l = RunObjects<Elt>(n, ell, K, false, FibreRepresentation::SymN);
    CompareObjects<Elt>(t, *c, *l);
    Check(c->delta.readoff == 0 && c->delta.howell >= 1,
          t + ": every corrupted read-off fired the residual check and fell back to the Howell solve (" + Txt(c->delta) + ")");
}

// (N) (audit m3 (b)) an INCONSISTENT Theta_eq: W = T C moved off im C (hecke::ScopedThetaEqPerturbation) must make (iii)
// fail on the stream route (read-off -> Residual -> the Howell solve) with exactly the legacy route's text.
template<class Elt>
void InconsistentThetaEqArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t = "(N) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " inconsistent Theta_eq";
    std::string msg[2];
    el::ModuleStreamTotals d[2];
    for (int pass = 0; pass < 2; ++pass) {
        const hecke::ScopedThetaEqPerturbation pert(true);
        const el::ModuleStreamTotals b0 = el::module_stream_totals();
        try {
            (void)RunObjects<Elt>(n, ell, K, pass == 0, FibreRepresentation::SymN);
        } catch (const std::runtime_error& e) { msg[pass] = e.what(); }
        d[pass] = Delta(b0, el::module_stream_totals());
    }
    const std::string pre = "HeckeOperatorOnK (Z/ell^k): runtime assertion (iii) failed -- ";
    Check(!msg[1].empty() && msg[1].rfind(pre, 0) == 0, t + ": the legacy route throws the (iii) text, got '" + msg[1] + "'");
    Check(msg[0] == msg[1], t + ": the stream route throws the legacy text byte for byte, got '" + msg[0] + "'");
    Check(d[0].howell >= 1 && d[0].legacy == 0 && d[1].legacy >= 1 && d[1].readoff == 0,
          t + ": the stream run fell back to the Howell solve, the legacy run took only legacy solves (" + Txt(d[0]) + " | " +
              Txt(d[1]) + ")");
}

// (N) (audit m1) the d o d predicate on planted inputs: engine::elim::product_is_zero, GEMM and the scalar loop.
template<class Elt>
void UnitDd(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    Rng r(seed);
    const uint32_t m = 23, l = 17, n = 19, l0 = 9;   // Xq = [A | 0] (m x l), Xp = [0 ; B] (l x n): Xq Xp = 0
    DenseBuf<Elt> Xq(m, l), Xp(l, n);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < l; ++j) Xq.at(i, j) = j < l0 ? Below<Elt>(r, ctx.p) : Elt(0);
    for (uint32_t i = 0; i < l; ++i)
        for (uint32_t j = 0; j < n; ++j) Xp.at(i, j) = i >= l0 ? Below<Elt>(r, ctx.p) : Elt(0);
    // a product that vanishes only modulo ell^k: (ell^{k-1}) (ell) = ell^k = 0
    Elt top = 1;
    for (uint32_t s = 0; s + 1 < ctx.k; ++s) top = ctx.mul(top, ctx.ell);
    for (uint32_t i = 0; i < m; ++i) Xq.at(i, l0 - 1) = 0;   // column l0 - 1 of Xq meets row l0 - 1 of Xp only at (5, 7)
    Xq.at(5, l0 - 1) = top;
    Xp.at(l0 - 1, 7) = ctx.ell;
    for (int g = 0; g < 2; ++g)
        Check(el::product_is_zero<Elt>(ctx, engine::cview(Xq), engine::cview(Xp), g == 0),
              "(N) " + tag + (g == 0 ? " GEMM" : " scalar") + ": a planted zero product (one entry ell^k = 0) -> true");
    // one non-zero entry of the product, deep in the matrix
    DenseBuf<Elt> Xq2 = Xq;
    uint32_t lj = l0;
    while (lj < l && Xp.at(lj, n - 1) == 0) ++lj;
    Check(lj < l, "(N) " + tag + ": Xp has a non-zero entry in its last column (setup)");
    if (lj < l) {
        Xq2.at(m - 1, lj) = 1;
        for (int g = 0; g < 2; ++g)
            Check(!el::product_is_zero<Elt>(ctx, engine::cview(Xq2), engine::cview(Xp), g == 0),
                  "(N) " + tag + (g == 0 ? " GEMM" : " scalar") + ": a planted non-zero product -> false (the d o d check fires)");
    }
    // empty inner dimension: the product is zero on both routes
    const DenseBuf<Elt> e1(m, 0), e2(0, n);
    for (int g = 0; g < 2; ++g)
        Check(el::product_is_zero<Elt>(ctx, engine::cview(e1), engine::cview(e2), g == 0),
              "(N) " + tag + (g == 0 ? " GEMM" : " scalar") + ": an empty inner dimension -> true");
}

// (L) (S23 G2-E) the lean retention, as RunModuleT runs it: every kept object (C, piv, H, dec, dK; A^{(q)}, the
// certificate bundle) equal to the legacy route's (Full); the released objects' accessors throw their named text.
template<class Elt>
void LeanArm(uint32_t n, uint64_t ell, uint32_t K, bool oracle = true) {   // (S24 CL-O) oracle: (R) here (else in the Slow name)
    const std::string t = "(L) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " lean";
    const auto s = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, false, /*lean=*/true);
    const auto l = RunObjects<Elt>(n, ell, K, false, FibreRepresentation::SymN);
    for (uint32_t q = 0; q < 3; ++q) {
        const auto& A = s->coh->module_degree(q);
        const auto& B = l->coh->module_degree(q);
        const std::string tq = t + " q=" + std::to_string(q);
        Check(Same(A.C, B.C) && A.piv == B.piv && Same(A.H, B.H) && A.dec.exponents == B.dec.exponents &&
                  A.dec.position == B.dec.position && Same(A.dec.basis, B.dec.basis) && Same(A.dec.Gamma, B.dec.Gamma) &&
                  Same(A.dec.U, B.dec.U) && A.dK == B.dK && A.length_H == B.length_H,
              tq + " the kept cohomology objects equal the legacy route's");
        Check(A.X.rows == 0 && A.X.cols == 0 && A.Zc.rows == 0 && A.Zc.cols == 0 && B.X.cols == B.C.cols,
              tq + " X, Zc released (0 x 0) on the lean object only");
        for (int which = 0; which < 2; ++which) {
            std::string msg;
            try {
                (void)(which == 0 ? s->coh->module_X(q) : s->coh->module_Zc(q));
            } catch (const std::logic_error& e) { msg = e.what(); }
            const std::string want = which == 0 ? "module_X: released" : "module_Zc: released";
            Check(msg.find(want) != std::string::npos, tq + " the released accessor throws its named text, got '" + msg + "'");
        }
        Check(Same(l->coh->module_X(q), B.X) && Same(l->coh->module_Zc(q), B.Zc), tq + " module_X / module_Zc on a full object");
    }
    Check(s->coh->module_intermediates_released() && !l->coh->module_intermediates_released(), t + " released flags");
    for (size_t i = 0; i < s->ops.size(); ++i) {
        const auto& x = *s->ops[i];
        const auto& y = *l->ops[i];
        const std::string tp = t + " T_" + std::to_string(s->primes[i]);
        for (uint32_t q = 0; q < 3; ++q)
            Check(Same(x.H_matrix_module(q), y.H_matrix_module(q)), tp + " q=" + std::to_string(q) + " A^{(q)}");
        const auto ca = x.certificates(), cb = y.certificates();
        Check(ca.routes_bit_identical && cb.routes_bit_identical && ca.relation_checks == cb.relation_checks &&
                  ca.cocycle_checks == cb.cocycle_checks && ca.theta_eq_solves == cb.theta_eq_solves &&
                  ca.theta_eq_residual_max == 0 && cb.theta_eq_residual_max == 0,
              tp + " certificate bundle equal");
        Check(x.module_objects_released() && !y.module_objects_released(), tp + " released on the lean operator only");
        for (int which = 0; which < 2; ++which) {
            std::string msg;
            try {
                (void)(which == 0 ? x.Theta_module(1) : x.Theta_eq_module(1));
            } catch (const std::logic_error& e) { msg = e.what(); }
            const std::string want = which == 0 ? "Theta_module: released under ModuleRetention::Lean"
                                                : "Theta_eq_module: released under ModuleRetention::Lean";
            Check(msg.find(want) != std::string::npos, tp + " the released accessor throws its named text, got '" + msg + "'");
        }
    }
    Check(s->delta.lean_released == 1 + s->ops.size() && l->delta.lean_released == 0,
          t + " counter retention=lean: the cohomology object and every operator (" + Txt(s->delta) + ")");
    bool chains = true, pooled = true;   // (S23 G2-F) the lean run's operators ran on chain-records-only tau tables
    for (size_t i = 0; i < s->tau.size(); ++i) chains = chains && s->tau[i]->chains_only();
    for (size_t i = 0; i < l->tau.size(); ++i) pooled = pooled && !l->tau[i]->chains_only();
    Check(chains && pooled && !s->tau.empty(), t + " the lean run's tau tables are ChainsOnly, the legacy run's Full");
    // (S24 CL-O) the independent oracle on the lean objects, with a full stream run's Theta (itself checked by (R) first)
    if (!oracle) return;
    const auto f = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
    OracleOnRun<Elt>(t + " full", *f, n, ell, K, true, false);
    OracleOnLeanRun<Elt>(t, *s, *f, ell, K);
}

// (N) (S23 G2-D) a corrupted route-2 block entry must fire [C, obl:agreement](i) with the legacy text on both routes.
// (audit m1) Three sites: q = 0's last entry (one chunk), and q = 1 (two chunks) in its first and its second chunk;
// (G2-H audit F2) a fourth, q = 1's interior block row sigma = 3.
template<class Elt>
void Route2PerturbationArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t0 = "(N) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                           (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " corrupted route-2 block";
    const struct { hecke::Route2Site site; const char* name; const char* q; } sites[] = {
        {hecke::Route2Site::Degree0Last, " (q=0, last entry)", "0"},
        {hecke::Route2Site::Degree1FirstChunk, " (q=1, first chunk)", "1"},
        {hecke::Route2Site::Degree1LastChunk, " (q=1, second chunk)", "1"},
        {hecke::Route2Site::Degree1Middle, " (q=1, interior block row sigma = 3)", "1"},   // (G2-H audit F2)
    };
    for (const auto& st : sites) {
        const std::string t = t0 + st.name;
        std::string msg[2];
        for (int pass = 0; pass < 2; ++pass) {
            const hecke::ScopedRoute2Perturbation pert(st.site);
            try {
                (void)RunObjects<Elt>(n, ell, K, pass == 0, FibreRepresentation::SymN);
            } catch (const std::runtime_error& e) { msg[pass] = e.what(); }
        }
        const std::string want = std::string("HeckeOperatorOnK (Z/ell^k): the fused route and route 2 differ at q=") + st.q +
                                 " (O12(a) over Z/ell^k)";
        Check(msg[1] == want, t + ": the legacy route throws the O12(a) text, got '" + msg[1] + "'");
        Check(msg[0] == msg[1], t + ": the stream route throws the same text byte for byte, got '" + msg[0] + "'");
    }
}

// (B) the Bredon flat route at ell in {2, 3}: untouched by the levers (but invariant_factors' work buffers, S23 G2-G).
void BredonArm(uint32_t n, uint32_t ell, uint32_t K) {
    const std::string t = "(B) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")";
    SimplicialComplex Kx = CreateSL2ZFundamentalDomainTriangulation();
    const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(ell, K);
    std::vector<uint32_t> len[2];
    std::vector<std::vector<uint32_t>> inv[2];
    el::ModuleStreamTotals d[2];
    const hecke::ScopedCoefficientRing ring(K);
    for (int pass = 0; pass < 2; ++pass) {
        const el::ScopedModuleStream s(pass == 0);
        const el::ModuleStreamTotals b0 = el::module_stream_totals();
        auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(Kx, n, ctx);
        LocalSystemSimplicialComplex l(Kx, std::move(rels), n + 1, ctx.p);
        EquivariantCohomologyOnK x(&l);
        x.compute_cohomology();
        Check(x.is_flat_route(), t + " runs the flat-coordinate route");
        len[pass] = x.H_lengths();
        for (uint32_t q = 0; q < x.num_degrees(); ++q) inv[pass].push_back(x.H_invariant_factors(q));
        d[pass] = Delta(b0, el::module_stream_totals());
    }
    Check(len[0] == len[1] && inv[0] == inv[1], t + " lengths and invariant factors stream == legacy");
    // (S23 G2-G) the flat route's only lever-selected step is invariant_factors' work buffers (value-neutral):
    // stream = invf lean only, legacy = invf legacy only, the same number of calls; no other module-stream step.
    const uint64_t il0 = d[0].invf_lean, il1 = d[1].invf_legacy;
    el::ModuleStreamTotals e0 = d[0], e1 = d[1];
    e0.invf_lean = 0; e1.invf_legacy = 0;
    Check(Sum(e0) == 0 && Sum(e1) == 0 && il0 >= 1 && il0 == il1,
          t + " no module-stream step on the flat route but invariant_factors' buffers (invf lean " + std::to_string(il0) +
              " == legacy " + std::to_string(il1) + ")");
}

// ---- (T) (S23 G2-F) the module route's tau table: chain records only -------------------------------------------
template<class Elt>
bool SameRecords(const hecke::HeckeTauTableT<Elt>& a, const hecke::HeckeTauTableT<Elt>& b, const SimplicialComplex& Kx) {
    if (a.num_cosets() != b.num_cosets()) return false;
    for (uint32_t q = 0; q < 3; ++q)
        for (uint32_t s = 0; s < Kx.get_num_q_simplices(q); ++s)
            for (uint32_t i = 1; i <= a.num_cosets(); ++i) {
                const std::vector<hecke::ChainTerm>& x = a.tau(i, q, s);
                const std::vector<hecke::ChainTerm>& y = b.tau(i, q, s);
                if (x.size() != y.size()) return false;
                for (size_t k = 0; k < x.size(); ++k)
                    if (x[k].coeff != y[k].coeff || !(x[k].gamma == y[k].gamma) || x[k].sigma_idx != y[k].sigma_idx) return false;
            }
    return true;
}

template<class Elt>
bool SameTableData(const hecke::HeckeTauTableT<Elt>& a, const hecke::HeckeTauTableT<Elt>& b) {
    return a.certified_bound_B() == b.certified_bound_B() && a.certified_bound_B_tab() == b.certified_bound_B_tab() &&
           a.used_mod_ell_solve() == b.used_mod_ell_solve() && a.num_fills_int64() == b.num_fills_int64() &&
           a.num_fills_widened() == b.num_fills_widened() && a.num_fills_mod_ell() == b.num_fills_mod_ell() &&
           a.policy() == b.policy() && a.to_string() == b.to_string();
}

// Every field-route accessor of a ChainsOnly table throws std::logic_error naming itself and ChainsOnly.
template<class Elt>
void ChainsOnlyRefusals(const std::string& t, hecke::HeckeTauTableT<Elt>& c) {
    const std::vector<std::pair<std::string, std::function<void()>>> calls = {
        {"xi_terms", [&] { (void)c.xi_terms(1, 0, 0); }},
        {"rho_pool", [&] { (void)c.rho_pool(); }},
        {"rho_pool_is_low_rank", [&] { (void)c.rho_pool_is_low_rank(); }},
        {"rho_pool_block", [&] { (void)c.rho_pool_block(0); }},
        {"rho_pool_key", [&] { (void)c.rho_pool_key(0); }},
        {"rho_pool_low_rank", [&] { (void)c.rho_pool_low_rank(0); }},
        {"corrupt_rho_pool_for_validation", [&] { c.corrupt_rho_pool_for_validation(); }},
    };
    for (const auto& kv : calls) {
        std::string msg;
        try {
            kv.second();
        } catch (const std::logic_error& e) { msg = e.what(); }
        Check(msg.find("HeckeTauTable::" + kv.first + ": this table was built TauTableContents::ChainsOnly") != std::string::npos,
              t + " ChainsOnly: " + kv.first + " throws its named text, got '" + msg + "'");
    }
}

template<class Elt>
void SameOperator(const std::string& t, const hecke::HeckeOperatorOnKT<Elt>& x, const hecke::HeckeOperatorOnKT<Elt>& y) {
    for (uint32_t q = 0; q < 3; ++q) {
        const std::string tq = t + " q=" + std::to_string(q);
        Check(Same(x.H_matrix_module(q), y.H_matrix_module(q)), tq + " A^{(q)}");
        Check(Same(x.Theta_module(q), y.Theta_module(q)), tq + " Theta");
        Check(Same(x.Theta_eq_module(q), y.Theta_eq_module(q)), tq + " Theta_eq");
    }
    const auto ca = x.certificates(), cb = y.certificates();
    Check(ca.routes_bit_identical && cb.routes_bit_identical && ca.relation_checks == cb.relation_checks &&
              ca.cocycle_checks == cb.cocycle_checks && ca.theta_eq_solves == cb.theta_eq_solves &&
              ca.theta_eq_residual_max == 0 && cb.theta_eq_residual_max == 0 && ca.theta_eq_certified == cb.theta_eq_certified &&
              ca.theta_eq_solves > 0,
          t + " certificate bundle equal (O12(a), (i), (ii), (iii), residual; (iii) ran)");
}

// (T) at (n, ell, K): per Hecke prime p in {2, 3, 5, 7, ell}, on ONE coset table, the Full and the ChainsOnly tau table:
// the chain records term for term (coeff, gamma, sigma_k), B_tab, B, the fill counters, the regime and to_string()
// equal; certify() == 0 on both; every field-route accessor of the ChainsOnly table refuses with its named text (and
// the field operator's constructor with it, u32); stored_bytes() strictly smaller.  Then the module operator on the
// ChainsOnly table on the stream route against (a) the same route on the Full table and (b) the legacy route
// (HECKE_MODULE_STREAM=0's pair: Full table, compute_module_legacy): A^{(q)}, Theta^{(q)}, Theta^{(q)}_eq and the
// certificate bundle entrywise.
template<class Elt>
void TauContentsArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t = "(T) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                          (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    const double t0 = Now();
    const hecke::ScopedCoefficientRing ring(K);
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const engine::ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeModuleStreamValidation", ell, K);
    LocalSystemSimplicialComplexT<Elt> lssc(*Kx, hecke::GenerateSL2ZRelationsRing<Elt>(*Kx, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<Elt> coh(&lssc);
    {
        const el::ScopedModuleStream s(true);
        coh.compute_cohomology();
    }
    std::vector<uint64_t> primes = {2, 3, 5, 7};
    if (ell > 7) primes.push_back(ell);
    uint64_t bytes_full = 0, bytes_chains = 0;
    for (uint64_t p : primes) {
        const std::string tp = t + " T_" + std::to_string(p);
        std::unique_ptr<hecke::HeckeCosetTablesT<Elt>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<Elt>> full, chains;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<Elt>(p, 1, static_cast<Elt>(ell), n));
            full.reset(new hecke::HeckeTauTableT<Elt>(*Kx, *model, *cosets, static_cast<Elt>(ell), n));
            chains.reset(new hecke::HeckeTauTableT<Elt>(*Kx, *model, *cosets, static_cast<Elt>(ell), n,
                                                       hecke::TauVertexPolicy::GeometricNearest,
                                                       hecke::TauTableContents::ChainsOnly));
        }
        Check(chains->chains_only() && !full->chains_only() && full->contents() == hecke::TauTableContents::Full,
              tp + " contents flags");
        Check(SameRecords(*full, *chains, *Kx), tp + " chain records equal term for term (coeff, gamma, sigma_k)");
        Check(SameTableData(*full, *chains), tp + " B_tab, B, fill counters, regime, policy, to_string() equal");
        Check(full->certify() == 0 && chains->certify() == 0, tp + " certify() == 0 on both tables");
        Check(!full->xi_terms(1, 0, 0).empty() && !full->rho_pool().empty(), tp + " the Full table keeps its xi table and pool");
        ChainsOnlyRefusals(tp, *chains);
        const uint64_t bf = full->stored_bytes(), bc = chains->stored_bytes();
        Check(bc < bf, tp + " stored_bytes ChainsOnly " + std::to_string(bc) + " < Full " + std::to_string(bf));
        bytes_full += bf;
        bytes_chains += bc;
        if constexpr (std::is_same<Elt, uint32_t>::value) {
            if (p == primes.back()) {   // the field operator's constructor reads xi_terms: it must refuse a ChainsOnly table
                const hecke::ScopedCoefficientRing field(1);
                std::vector<LocalSystemSimplicialComplex::SimplexRelation> frels;
                for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*Kx, n, static_cast<uint32_t>(ell)))
                    frels.push_back(datum.rel);
                LocalSystemSimplicialComplex fl(*Kx, std::move(frels), n + 1, static_cast<uint32_t>(ell));
                EquivariantCohomologyOnK fc(&fl);
                fc.compute_cohomology();
                std::string msg;
                try {
                    hecke::HeckeOperatorOnKT<uint32_t> fop(&fc, cosets.get(), chains.get());
                } catch (const std::logic_error& e) { msg = e.what(); }
                Check(msg.find("HeckeTauTable::xi_terms: this table was built TauTableContents::ChainsOnly") != std::string::npos,
                      tp + " the field operator refuses a ChainsOnly table, got '" + msg + "'");
                bool ok = true;
                try {
                    hecke::HeckeOperatorOnKT<uint32_t> fop(&fc, cosets.get(), full.get());
                } catch (const std::exception&) { ok = false; }
                Check(ok, tp + " the field operator accepts the Full table (control)");
            }
        }
        hecke::HeckeOperatorOnKT<Elt> opc(&coh, cosets.get(), chains.get()), opf(&coh, cosets.get(), full.get()),
            opl(&coh, cosets.get(), full.get());
        {
            const el::ScopedModuleStream s(true);
            opc.compute();
            opf.compute();
        }
        {
            const el::ScopedModuleStream s(false);
            opl.compute();
        }
        SameOperator(tp + " stream ChainsOnly vs stream Full", opc, opf);
        SameOperator(tp + " stream ChainsOnly vs legacy Full", opc, opl);
        // (fix F2) negative control: the same record perturbation on both tables (the operators are done with them);
        // certify() must still reject a defective ChainsOnly table, with the code it returns on the Full one.
        full->corrupt_record_for_validation();
        chains->corrupt_record_for_validation();
        const uint32_t ef = full->certify(), ec = chains->certify();
        Check(ef == 20 && ec == 20, tp + " a perturbed vertex record: certify() == 20 on both tables (Full " +
                                        std::to_string(ef) + ", ChainsOnly " + std::to_string(ec) + ")");
    }
    std::cout << "MODULE_STREAM_CELL " << t << " tau_stored_bytes full=" << bytes_full << " chains=" << bytes_chains
              << " saved=" << (bytes_full - bytes_chains) << " seconds=" << Fx(Now() - t0, 3) << "\n";
}

// (T) slow: the tau tables of T_ell alone at a larger ell -- the object G2-F removes -- built ChainsOnly first, then Full
// (beside it, so the records can be compared), each with its peak RSS above the RSS before it (VmHWM after a
// /proc/self/clear_refs reset and malloc_trim; recorded, not gated) and its stored_bytes() (gated: ChainsOnly < Full).
void TauRssArm(uint32_t n, uint32_t ell) {
    const std::string t = "(T) tau tables of T_" + std::to_string(ell) + " at n=" + std::to_string(n);
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<hecke::HeckeCosetTablesT<uint32_t>> cosets;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
        cosets.reset(new hecke::HeckeCosetTablesT<uint32_t>(ell, 1, ell, n));
    }
    std::unique_ptr<hecke::HeckeTauTableT<uint32_t>> chains, full;
    double sec[2] = {0, 0}, delta[2] = {-1, -1};
    bool scoped = true;
    for (int pass = 0; pass < 2; ++pass) {
        malloc_trim(0);
        const double base = oc_util::CurrentRssMiB();
        scoped = oc_util::ResetPeakRss() && scoped;
        const double t0 = Now();
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            (pass == 0 ? chains : full)
                .reset(new hecke::HeckeTauTableT<uint32_t>(*Kx, *model, *cosets, ell, n, hecke::TauVertexPolicy::GeometricNearest,
                                                           pass == 0 ? hecke::TauTableContents::ChainsOnly
                                                                     : hecke::TauTableContents::Full));
        }
        sec[pass] = Now() - t0;
        delta[pass] = oc_util::PeakRssMiB() - base;
    }
    Check(SameRecords(*full, *chains, *Kx) && SameTableData(*full, *chains), t + ": chain records and table data equal");
    const uint64_t bc = chains->stored_bytes(), bf = full->stored_bytes();
    Check(bc < bf, t + ": stored_bytes ChainsOnly " + std::to_string(bc) + " < Full " + std::to_string(bf));
    std::cout << "MODULE_STREAM_CELL " << t << " cosets=" << cosets->num_cosets() << " chains: seconds=" << Fx(sec[0], 3)
              << " stored_bytes=" << bc << " peak_rss_delta_mib=" << Fx(delta[0], 1) << " full: seconds=" << Fx(sec[1], 3)
              << " stored_bytes=" << bf << " peak_rss_delta_mib=" << Fx(delta[1], 1)
              << " bytes_per_coset_saved=" << Fx(static_cast<double>(bf - bc) / cosets->num_cosets(), 1)
              << " rss_scope=" << (scoped ? "run" : "process") << "\n";
}

// ---- (S) cell level ----------------------------------------------------------------------------------------
struct CellOut {
    std::string text;
    std::vector<uint32_t> K_run, Lv_run;
    double seconds = 0.0, peak_mib = -1.0;
    bool rss_scoped = false;
    el::ModuleStreamTotals delta;   // (S23 G2-F) the counters this cell moved (tau=chains|pooled)
};

std::string Txt(const std::vector<uint32_t>& v) {
    std::ostringstream s;
    s << "[";
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str() + "]";
}

CellOut RunCell(uint32_t n, uint64_t ell, uint32_t kmax, bool stream, bool eval = true) {
    const el::ScopedModuleStream s(stream);
    const el::ScopedModuleEval se(eval);   // (S23 G2-T)
    CellOut r;
    r.rss_scoped = oc_util::ResetPeakRss();
    sd::SatOptions o;
    o.kmax = kmax;
    const el::ModuleStreamTotals before = el::module_stream_totals();
    const double t0 = Now();
    const sd::SatCell c = sd::ComputeSaturationDefectCell(n, ell, o);
    r.seconds = Now() - t0;
    r.delta = Delta(before, el::module_stream_totals());
    r.peak_mib = oc_util::PeakRssMiB();
    std::ostringstream os;
    os << "SAT_CELL n=" << c.n << " ell=" << c.ell << " kmax=" << c.kmax << " e_max=" << c.e_max << " K_cls=" << c.K_cls
       << " blocks=" << c.blocks.size() << "\n";
    for (const sd::SatBlock& b : c.blocks) os << sd::SatDefectLine(c, b) << "\n";
    r.text = os.str();
    r.K_run = c.K_run;
    r.Lv_run = c.Lv_run;
    return r;
}

std::string FirstDiff(const std::string& a, const std::string& b) {
    std::istringstream x(a), y(b);
    std::string l1, l2;
    for (int i = 1;; ++i) {
        const bool g1 = static_cast<bool>(std::getline(x, l1)), g2 = static_cast<bool>(std::getline(y, l2));
        if (!g1 && !g2) return "(none)";
        if (!g1 || !g2 || l1 != l2)
            return "line " + std::to_string(i) + ":\n    stream: " + (g1 ? l1 : "<end>") + "\n    legacy: " + (g2 ? l2 : "<end>");
    }
}

void CellArm(uint32_t n, uint64_t ell, uint32_t kmax, const std::string& note = "") {
    const std::string t = "(S) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax) + note;
    const CellOut s = RunCell(n, ell, kmax, true);
    const CellOut l = RunCell(n, ell, kmax, false);
    const bool same = s.text == l.text;
    Check(same, t + ": the SAT_DEFECT lines are byte-identical; first difference " + (same ? std::string("(none)") : FirstDiff(s.text, l.text)));
    Check(s.K_run == l.K_run && s.Lv_run == l.Lv_run, t + ": K_run, Lv_run equal (" + Txt(s.K_run) + ", " + Txt(s.Lv_run) + ")");
    size_t lines = 0;
    for (char ch : s.text) lines += ch == '\n';
    Check(lines >= 2, t + ": at least one block line (non-vacuity)");
    // (S23 G2-F) RunModuleT's tau tables: chain records only on the stream route, Full on the legacy route
    Check(s.delta.tau_chains >= 1 && s.delta.tau_pooled == 0 && l.delta.tau_chains == 0 && l.delta.tau_pooled >= 1,
          t + ": tau=chains on the stream run, tau=pooled on the legacy run (stream " + Txt(s.delta) + "; legacy " +
              Txt(l.delta) + ")");
    std::cout << "MODULE_STREAM_CELL " << t << " K_run=" << Txt(s.K_run) << " Lv_run=" << Txt(s.Lv_run) << " lines=" << lines
              << " seconds stream=" << Fx(s.seconds, 3) << " legacy=" << Fx(l.seconds, 3) << " peak_rss_mib stream=" << Fx(s.peak_mib, 1)
              << " legacy=" << Fx(l.peak_mib, 1) << " rss_scope=" << (s.rss_scoped && l.rss_scoped ? "run" : "process") << "\n";
}

// (V) (c) (S23 G2-T) cell level: the stream route with HECKE_MODULE_EVAL on against off, the SAT_DEFECT lines byte for byte.
void EvalCellArm(uint32_t n, uint64_t ell, uint32_t kmax) {
    const std::string t = "(V) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax) + " eval on vs off";
    const CellOut f = RunCell(n, ell, kmax, true, true);
    const CellOut o = RunCell(n, ell, kmax, true, false);
    const bool same = f.text == o.text;
    Check(same, t + ": the SAT_DEFECT lines are byte-identical; first difference " + (same ? std::string("(none)") : FirstDiff(f.text, o.text)));
    Check(f.K_run == o.K_run && f.Lv_run == o.Lv_run, t + ": K_run, Lv_run equal (" + Txt(f.K_run) + ", " + Txt(f.Lv_run) + ")");
    size_t lines = 0;
    for (char ch : f.text) lines += ch == '\n';
    Check(lines >= 2, t + ": at least one block line (non-vacuity)");
    Check(f.delta.eval_fast >= 3 && f.delta.eval_legacy == 0 && o.delta.eval_fast == 0 && o.delta.eval_legacy >= 3 &&
              f.delta.eval_fast == o.delta.eval_legacy,
          t + ": eval=fast on the on-run, eval=legacy on the off-run (" + Txt(f.delta) + "; " + Txt(o.delta) + ")");
    std::cout << "MODULE_STREAM_CELL " << t << " K_run=" << Txt(f.K_run) << " lines=" << lines << " seconds eval_on=" << Fx(f.seconds, 3)
              << " eval_off=" << Fx(o.seconds, 3) << "\n";
}

// ---- (P) (S23 G2-P) the parallel assembly: the same objects at every consumer-pool size ----------------------------
// The pools are created in process (engine::tape::ScopedConsumerPool, as EngineDeterminismValidation does), so one
// process runs the stream route at 1, 2, 7 and 16 workers; the 1-worker run is the serial assembly (the reference).
std::vector<uint32_t> ParallelSizes(bool slow) { return slow ? std::vector<uint32_t>{1, 8} : std::vector<uint32_t>{1, 2, 7, 16}; }

bool SameAssemblyCounts(const el::ModuleStreamTotals& a, const el::ModuleStreamTotals& b) {
    return a.omega_streamed == b.omega_streamed && a.route2_blocked == b.route2_blocked && a.lr_grouped == b.lr_grouped &&
           a.lr_early_flush == b.lr_early_flush && a.readoff == b.readoff && a.howell == b.howell && a.w_nonpivot == b.w_nonpivot &&
           a.eval_fast == b.eval_fast && a.eval_legacy == b.eval_legacy;
}

template<class Elt>
void ParallelObjectArm(uint32_t n, uint64_t ell, uint32_t K, FibreRepresentation rep = FibreRepresentation::SymN,
                       bool readoff_split = false) {
    const bool hida = rep == FibreRepresentation::HidaLatticeH;
    const std::string t0 = std::string("(P) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                           (hida ? " Hida L_n" : "") + (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    std::unique_ptr<ObjRun<Elt>> ref;
    std::ostringstream secs;
    for (uint32_t w : ParallelSizes(false)) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleTileWork extent(0);   // (S24 PAR-L1) the extent-only tiling, so the splits stay exercised here
        const uint64_t p0 = el::module_parallel_degrees(true), s0 = el::module_parallel_degrees(false);
        const uint64_t h0 = el::module_parallel_howell_regions(), q0 = el::module_parallel_readoff_regions();
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        const uint64_t dp = el::module_parallel_degrees(true) - p0, ds = el::module_parallel_degrees(false) - s0;
        const uint64_t dh = el::module_parallel_howell_regions() - h0, dq = el::module_parallel_readoff_regions() - q0;
        if (w == 1) Check(dh == 0 && dq == 0, t0 + " pool 1: no hook region (howell " + std::to_string(dh) + ", readoff " + std::to_string(dq) + ")");
        else secs << " [hook howell=" << dh << " readoff=" << dq << "]";
        if (w > 1 && n >= 40) Check(dh > 0, t0 + " pool " + std::to_string(w) + ": the Howell forms split over the hook (" + std::to_string(dh) + ")");
        if (w > 1 && readoff_split)   // (G2-P audit F1) the parallel read-off path is asserted, not only printed
            Check(dq > 0, t0 + " pool " + std::to_string(w) + ": the product read-off ran in hook blocks (" + std::to_string(dq) + ")");
        const uint64_t degs = 3ull * r->primes.size();
        const std::string t = t0 + " pool " + std::to_string(w);
        Check(w == 1 ? (dp == 0 && ds == degs) : (dp == degs && ds == 0),
              t + ": " + (w == 1 ? "serial" : "parallel") + " assembly on every operator degree (par=parallel:" + std::to_string(dp) +
                  ",serial:" + std::to_string(ds) + ")");
        secs << " w" << w << "=" << Fx(r->seconds, 3);
        if (!ref) { ref = std::move(r); continue; }
        CompareObjects<Elt>(t + " vs pool 1", *r, *ref);
        Check(SameAssemblyCounts(r->delta, ref->delta), t + ": the stream counters equal pool 1's (" + Txt(r->delta) + "; " +
                                                            Txt(ref->delta) + ")");
    }
    {   // the rollback: HECKE_MODULE_PARALLEL=0 (ScopedModuleParallel(false)) at 7 workers is the serial assembly
        engine::tape::ThreadPool pool(7);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleParallel off(false);
        const uint64_t p0 = el::module_parallel_degrees(true);
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        Check(el::module_parallel_degrees(true) == p0, t0 + " pool 7, lever off: no parallel degree");
        CompareObjects<Elt>(t0 + " pool 7 lever off vs pool 1", *r, *ref);
        Check(SameAssemblyCounts(r->delta, ref->delta), t0 + " pool 7 lever off: the stream counters equal pool 1's");
    }
    // (G2-T2 audit F3) the parallel assembly under G2-T2's two levers: the rho fill path (SIMD dispatched vs forced
    // scalar) x the route-2 memo (shared across chunks vs G2-D's chunks), each at 7 workers against pool 1's defaults
    for (int combo = 0; combo < 4; ++combo) {
        const bool simd = (combo & 1) == 0, shared = (combo & 2) == 0;
        engine::tape::ThreadPool pool(7);
        const engine::tape::ScopedConsumerPool sp(&pool);
        std::unique_ptr<ScopedRhoFillPath> fp;
        if (!simd) fp.reset(new ScopedRhoFillPath(RhoFillPath::Scalar));
        const el::ScopedRoute2SharedMemo m(shared);
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        CompareObjects<Elt>(t0 + " pool 7 rho_fill=" + (simd ? "auto" : "scalar") + " route2_memo=" + (shared ? "shared" : "chunked") +
                                " vs pool 1",
                            *r, *ref);
    }
    {   // (S24 PAR-L1) the default tile-work bound (HECKE_MODULE_TILE_WORK unset) at 16 workers: the same objects as pool 1
        engine::tape::ThreadPool pool(16);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleTileWork bound(size_t(1) << 20);
        const uint64_t c0 = el::module_tile_work_capped() + el::module_tile_work_serial();
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        const uint64_t dc = el::module_tile_work_capped() + el::module_tile_work_serial() - c0;
        if (n >= 40) Check(dc > 0, t0 + " pool 16 tile-work bound: the bound acted (" + std::to_string(dc) + " splits capped or serial)");
        CompareObjects<Elt>(t0 + " pool 16 tile-work bound 2^20 vs pool 1", *r, *ref);
        Check(SameAssemblyCounts(r->delta, ref->delta), t0 + " pool 16 tile-work bound: the stream counters equal pool 1's");
    }
    std::cout << "MODULE_STREAM_CELL " << t0 << " seconds" << secs.str() << "\n";
}

void ParallelCellArm(uint32_t n, uint64_t ell, uint32_t kmax, bool slow = false) {
    const std::string t0 = "(P) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax);
    std::string ref;
    std::ostringstream secs;
    size_t lines = 0;
    for (uint32_t w : ParallelSizes(slow)) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const uint64_t p0 = el::module_parallel_degrees(true);
        const CellOut c = RunCell(n, ell, kmax, true);
        const uint64_t dp = el::module_parallel_degrees(true) - p0;
        Check(w == 1 ? dp == 0 : dp > 0, t0 + " pool " + std::to_string(w) + ": parallel degrees " + std::to_string(dp));
        secs << " w" << w << "=" << Fx(c.seconds, 3);
        if (w == 1) {
            ref = c.text;
            for (char ch : ref) lines += ch == '\n';
            continue;
        }
        const bool same = c.text == ref;
        Check(same, t0 + " pool " + std::to_string(w) + " vs pool 1: the SAT_DEFECT lines are byte-identical; first difference " +
                        (same ? std::string("(none)") : FirstDiff(c.text, ref)));
    }
    {   // (G2-T2 audit F3) 7 workers with both G2-T2 levers off (forced scalar fill, chunked route-2 memo)
        engine::tape::ThreadPool pool(7);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const ScopedRhoFillPath fp(RhoFillPath::Scalar);
        const el::ScopedRoute2SharedMemo m(false);
        const CellOut c = RunCell(n, ell, kmax, true);
        const bool same = c.text == ref;
        Check(same, t0 + " pool 7 rho_fill=scalar route2_memo=chunked vs pool 1: the SAT_DEFECT lines are byte-identical; first "
                         "difference " + (same ? std::string("(none)") : FirstDiff(c.text, ref)));
    }
    Check(lines >= 2, t0 + ": at least one block line (non-vacuity)");
    std::cout << "MODULE_STREAM_CELL " << t0 << " lines=" << lines << " seconds" << secs.str() << "\n";
}

// (P) (G2-P audit F1) the parallel product read-off's negative paths, unit level, on a 7-worker pool bound as the
// elimination hook: m = 1024 rows in 8 row blocks of 128 (m ny l >= 2^20, so the parallel pass runs; asserted by
// module_parallel_readoff_regions()).  Ok equals the serial read-off; Residual for Y moved off im B at a non-pivot
// row of the LAST block, for X(0, 0) + 1, and for a residual injected on a pool worker that is not the calling
// thread (el::ScopedReadoffWorkerFault(1)); a std::runtime_error thrown on such a worker (mode 2) is rethrown on
// the caller with its text; afterwards the same pool still reads off Ok (not wedged).
template<class Elt>
void ParallelReadoffFaultUnit(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    const std::string t0 = "(P) (N) parallel read-off " + tag;
    Rng r(seed);
    const uint32_t m = 1024, nb = 48, ny = 48, l = nb + 4, rb = 128;
    std::vector<uint32_t> piv;
    for (uint32_t c = 0; c < nb; ++c) piv.push_back(c * (m / nb) + static_cast<uint32_t>(r.next() % (m / nb)));
    std::vector<char> is_piv(m, 0);
    for (uint32_t c = 0; c < nb; ++c) is_piv[piv[c]] = 1;
    DenseBuf<Elt> B(m, nb), Xt(nb, ny), A(m, l), G(l, ny);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t c = 0; c < nb; ++c) B.at(i, c) = Below<Elt>(r, ctx.p);
    for (uint32_t rr = 0; rr < nb; ++rr)
        for (uint32_t c = 0; c < nb; ++c) B.at(piv[rr], c) = (rr == c) ? Elt(1) : Elt(0);
    for (uint32_t i = 0; i < nb; ++i)
        for (uint32_t t = 0; t < ny; ++t) Xt.at(i, t) = Below<Elt>(r, ctx.p);
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < l; ++j) A.at(i, j) = j < nb ? B.at(i, j) : Elt(0);
    for (uint32_t i = 0; i < l; ++i)
        for (uint32_t t = 0; t < ny; ++t) G.at(i, t) = i < nb ? Xt.at(i, t) : Below<Elt>(r, ctx.p);
    uint32_t last_np = m - 1;
    while (is_piv[last_np]) --last_np;
    Check(last_np >= m - rb, t0 + ": the perturbed non-pivot row lies in the last row block (setup)");
    auto readoff = [&](DenseBuf<Elt>& X, uint32_t perturb) {
        return el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A), engine::cview(G), X, rb, perturb);
    };
    DenseBuf<Elt> Xs;
    const el::ReadoffStatus ss = readoff(Xs, ~0u);   // serial: no hook on this thread
    Check(ss == el::ReadoffStatus::Ok && Same(Xs, Xt), t0 + ": the serial read-off recovers the planted X (setup)");
    engine::tape::ThreadPool pool(7);
    const el::ParallelFor hook = [&pool](uint32_t nt, const std::function<void(uint32_t)>& fn) {
        pool.parallel_for(nt, pool.size(), [&](uint32_t t, uint32_t) { fn(t); });
    };
    const el::ScopedParallelHook sh(&hook);
    const el::ScopedModuleParallel on(true);
    const uint64_t q0 = el::module_parallel_readoff_regions();
    DenseBuf<Elt> Xp;
    Check(readoff(Xp, ~0u) == el::ReadoffStatus::Ok && Same(Xp, Xs), t0 + ": pool 7, Ok and X == the serial read-off's");
    Check(el::module_parallel_readoff_regions() == q0 + 1, t0 + ": pool 7 ran the parallel read-off (non-vacuity)");
    DenseBuf<Elt> Xn;
    Check(readoff(Xn, last_np) == el::ReadoffStatus::Residual,
          t0 + ": pool 7, Y moved off im B at a non-pivot row of the last block -> Residual");
    {
        const el::ScopedReadoffCorruption cor(true);
        Check(readoff(Xn, ~0u) == el::ReadoffStatus::Residual, t0 + ": pool 7, X(0, 0) + 1 -> Residual");
    }
    const uint64_t f0 = el::readoff_worker_faults();
    {
        const el::ScopedReadoffWorkerFault f(1);
        Check(readoff(Xn, ~0u) == el::ReadoffStatus::Residual, t0 + ": pool 7, a residual raised on a non-calling worker -> Residual");
    }
    Check(el::readoff_worker_faults() == f0 + 1, t0 + ": the residual fault fired on a pool worker, not on the caller");
    std::string msg;
    {
        const el::ScopedReadoffWorkerFault f(2);
        try {
            (void)readoff(Xn, ~0u);
        } catch (const std::runtime_error& e) { msg = e.what(); }
    }
    Check(el::readoff_worker_faults() == f0 + 2, t0 + ": the throwing fault fired on a pool worker, not on the caller");
    Check(msg == "solve_unit_pivot_readoff_product: injected worker fault (validation only)",
          t0 + ": an exception thrown on a non-calling worker is rethrown on the caller with its text, got '" + msg + "'");
    DenseBuf<Elt> Xa;
    Check(readoff(Xa, ~0u) == el::ReadoffStatus::Ok && Same(Xa, Xs), t0 + ": after the faults the pool still reads off Ok (not wedged)");
}

// (P) (G2-P audit F1) operator level on a 7-worker consumer pool: a residual raised on a non-calling worker of the
// parallel read-off makes the stream route fall back to the Howell solve (every object equals pool 1's); an
// exception thrown there escapes compute() with its text (caught and reported, no terminate, no wedge); the
// read-off corruption and the inconsistent Theta_eq controls give their serial outcomes on this pool.
template<class Elt>
void ParallelFaultArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t0 = "(P) (N) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                           (std::is_same<Elt, uint64_t>::value ? " u64" : " u32") + " pool 7";
    std::unique_ptr<ObjRun<Elt>> ref;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        ref = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
    }
    engine::tape::ThreadPool pool(7);
    const engine::tape::ScopedConsumerPool sp(&pool);
    {
        const uint64_t f0 = el::readoff_worker_faults(), q0 = el::module_parallel_readoff_regions();
        std::unique_ptr<ObjRun<Elt>> r;
        {
            const el::ScopedReadoffWorkerFault f(1);
            r = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
        }
        const uint64_t df = el::readoff_worker_faults() - f0, dq = el::module_parallel_readoff_regions() - q0;
        Check(dq > 0 && df == dq, t0 + ": a residual fault fired on a pool worker in every parallel read-off (faults " +
                                      std::to_string(df) + ", parallel read-offs " + std::to_string(dq) + ")");
        Check(r->delta.howell >= 1, t0 + ": the worker residual fell back to the Howell solve (" + Txt(r->delta) + ")");
        CompareObjects<Elt>(t0 + " worker residual fault vs pool 1", *r, *ref);
    }
    {
        const uint64_t f0 = el::readoff_worker_faults();
        std::string msg;
        {
            const el::ScopedReadoffWorkerFault f(2);
            try {
                (void)RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);
            } catch (const std::runtime_error& e) { msg = e.what(); }
        }
        Check(el::readoff_worker_faults() == f0 + 1, t0 + ": the throwing fault fired once on a pool worker");
        Check(msg == "solve_unit_pivot_readoff_product: injected worker fault (validation only)",
              t0 + ": the exception thrown on a non-calling worker escapes compute() with its text, got '" + msg + "'");
        auto r = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN);   // the same pool afterwards
        CompareObjects<Elt>(t0 + " after the worker exception vs pool 1", *r, *ref);
    }
    CorruptionArm<Elt>(n, ell, K);
    InconsistentThetaEqArm<Elt>(n, ell, K);
}

// (V) (f) (S23 G2-T2) cell level: the SAT_DEFECT lines with each available rho fill path against the scalar path.
void PathCellArm(uint32_t n, uint64_t ell, uint32_t kmax) {
    CellOut ref;
    {
        const ScopedRhoFillPath sp(RhoFillPath::Scalar);
        ref = RunCell(n, ell, kmax, true, true);
    }
    for (RhoFillPath path : AvailableRhoPaths()) {
        if (path == RhoFillPath::Scalar) continue;
        const std::string t = "(V) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax) +
                              " rho path " + to_text(path) + " vs scalar";
        const RhoFillCounts c0 = rho_fill_counts();
        CellOut o;
        {
            const ScopedRhoFillPath sp(path);
            o = RunCell(n, ell, kmax, true, true);
        }
        const RhoFillCounts c1 = rho_fill_counts();
        const uint64_t dsimd = path == RhoFillPath::Avx512 ? c1.avx512 - c0.avx512 : c1.avx2 - c0.avx2;
        const bool same = o.text == ref.text;
        Check(same, t + ": the SAT_DEFECT lines are byte-identical; first difference " +
                        (same ? std::string("(none)") : FirstDiff(o.text, ref.text)));
        Check(dsimd > 0, t + ": the fills took " + to_text(path) + " (" + std::to_string(dsimd) + ")");
        std::cout << "MODULE_STREAM_CELL " << t << " seconds path=" << Fx(o.seconds, 3) << " scalar=" << Fx(ref.seconds, 3) << "\n";
    }
}

// (V) (g) (S23 G2-T2) cell level: the SAT_DEFECT lines with the route-2 memo shared against chunked.
void MemoCellArm(uint32_t n, uint64_t ell, uint32_t kmax) {
    const std::string t = "(V) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax) +
                          " route-2 memo shared vs chunked";
    CellOut sh, ch;
    {
        const el::ScopedRoute2SharedMemo m(true);
        sh = RunCell(n, ell, kmax, true, true);
    }
    {
        const el::ScopedRoute2SharedMemo m(false);
        ch = RunCell(n, ell, kmax, true, true);
    }
    const bool same = sh.text == ch.text;
    Check(same, t + ": the SAT_DEFECT lines are byte-identical; first difference " +
                    (same ? std::string("(none)") : FirstDiff(sh.text, ch.text)));
    std::cout << "MODULE_STREAM_CELL " << t << " seconds shared=" << Fx(sh.seconds, 3) << " chunked=" << Fx(ch.seconds, 3) << "\n";
}


// ---- (MT) (S24 MT-1) HECKE_MODULE_MT: the module route's splits at every consumer-pool size ------------------------
// The reference of every comparison is the route of BEFORE MT-1: HECKE_MODULE_MT=0 (ScopedModuleMt(false)) on a
// 1-worker pool (the serial G2-P route, with cyclic_decomposition's full pivot scan).  Each arm then runs the lever
// on at pools 1 / 2 / 7 / 16 (ScopedConsumerPool; at pool 1 only the pivot-search exit is in effect, at > 1 every
// split) and the lever off at 7.
std::vector<uint32_t> MtSizes() { return {1, 2, 7, 16}; }
// (MT-1 audit) the unit-level cyclic_decomposition arm also at odd pool sizes 3 / 13 (tile counts not dividing the extent)
std::vector<uint32_t> MtUnitSizes() { return {1, 2, 3, 7, 13, 16}; }

struct MtCounts {
    uint64_t v[6];
    static MtCounts now() {
        MtCounts c;
        for (uint32_t i = 0; i < 6; ++i) c.v[i] = el::module_mt_regions(static_cast<el::ModuleMtEvent>(i));
        return c;
    }
    MtCounts minus(const MtCounts& b) const {
        MtCounts c;
        for (uint32_t i = 0; i < 6; ++i) c.v[i] = v[i] - b.v[i];
        return c;
    }
    uint64_t total() const { uint64_t s = 0; for (uint32_t i = 0; i < 5; ++i) s += v[i]; return s; }   // the split regions
    uint64_t all() const { return total() + v[5]; }                                                     // + pivot_exit
    std::string txt() const {
        return "cyc_search=" + std::to_string(v[0]) + " cyc_rows=" + std::to_string(v[1]) + " gemm=" + std::to_string(v[2]) +
               " check=" + std::to_string(v[3]) + " assembly=" + std::to_string(v[4]) + " pivot_exit=" + std::to_string(v[5]);
    }
};

// A relation-matrix-shaped input for cyclic_decomposition whose pivot columns often hold no unit: G = W0 * diag(ell^{e_j})
// (most column valuations >= 1), Rl = G * Z for a random Z (so Rl lies in colspan G) plus, when `extra`, a few columns
// outside it.  Large enough (m >= 128) for the tile thresholds of the splits.
template<class Elt>
void MtCycInput(const engine::ModCtx<Elt>& ctx, Rng& r, uint32_t m, uint32_t ng, uint32_t nr, uint32_t mode, DenseBuf<Elt>& G,
                DenseBuf<Elt>& Rl) {
    G = RandVal<Elt>(ctx, r, m, ng, mode == 0 ? 1u : 2u);
    for (uint32_t j = 0; j < ng; ++j) {
        const uint32_t e = (mode == 1) ? 1u + static_cast<uint32_t>(r.next() % (ctx.k - 1)) : ((r.next() % 3) == 0 ? 1u : 0u);
        const Elt f = ctx.ell_pow(std::min<uint32_t>(e, ctx.k - 1));
        for (uint32_t i = 0; i < m; ++i) G.at(i, j) = ctx.mul(G.at(i, j), f);
    }
    const DenseBuf<Elt> Z = RandVal<Elt>(ctx, r, ng, nr, 2);
    Rl = MulScalar<Elt>(ctx, G, Z);
    for (uint32_t j = 0; j < nr; j += 7)   // columns outside colspan G (a fresh random column times ell)
        for (uint32_t i = 0; i < m; ++i) Rl.at(i, j) = ctx.mul(Below<Elt>(r, ctx.p), ctx.ell);
}

template<class Elt>
void MtUnitCyclic(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed, uint32_t cases) {
    Rng r(seed);
    MtCounts c0 = MtCounts::now();
    for (uint32_t cs = 0; cs < cases; ++cs) {
        const uint32_t mode = cs % 3;
        const uint32_t m = 128 + static_cast<uint32_t>(r.next() % 96), ng = 96 + static_cast<uint32_t>(r.next() % 160);
        const uint32_t nr = (cs % 4 == 3) ? 0u : 16 + static_cast<uint32_t>(r.next() % 96);
        DenseBuf<Elt> G, Rl;
        MtCycInput<Elt>(ctx, r, m, ng, nr, mode, G, Rl);
        const std::string t = "(MT) (U) cyclic_decomposition " + tag + " case " + std::to_string(cs) + " (m=" + std::to_string(m) +
                              " ng=" + std::to_string(ng) + " nr=" + std::to_string(nr) + " mode=" + std::to_string(mode) + ")";
        el::CyclicDecomposition<Elt> ref;
        {
            engine::tape::ThreadPool pool(1);
            const engine::tape::ScopedConsumerPool sp(&pool);
            const el::ScopedModuleStream s(true);
            const el::ScopedModuleMt off(false);
            ref = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
        }
        for (uint32_t w : MtUnitSizes()) {
            engine::tape::ThreadPool pool(w);
            const engine::tape::ScopedConsumerPool sp(&pool);
            const engine::tape::ScopedHostElimHook hook;   // as compute_cohomology_module installs it
            const el::ScopedModuleStream s(true);
            const el::ScopedModuleMt on(true);
            const el::CyclicDecomposition<Elt> d = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
            Check(SameDec(d, ref), t + " pool " + std::to_string(w) + ": exponents, position, basis, Gamma, U equal the pre-MT route's");
            if (w > 1) {   // every split forced (thresholds off): the split code on every step
                const el::ScopedModuleMtForceSplit force;
                const el::CyclicDecomposition<Elt> f = el::cyclic_decomposition<Elt>(ctx, engine::cview(G), engine::cview(Rl));
                Check(SameDec(f, ref), t + " pool " + std::to_string(w) + " splits forced: equal to the pre-MT route's");
            }
        }
    }
    const MtCounts d = MtCounts::now().minus(c0);
    Check(d.v[0] > 0 && d.v[1] > 0, "(MT) (U) cyclic_decomposition " + tag + ": the tiled pivot search and the row splits ran (forced; " + d.txt() + ")");
    std::cout << "  (MT) (U) cyclic_decomposition " << tag << ": " << cases << " cases, " << d.txt() << "\n";
}

template<class Elt>
void MtUnitSolves(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    Rng r(seed);
    const MtCounts c0 = MtCounts::now();
    for (uint32_t cs = 0; cs < 6; ++cs) {
        const std::string t = "(MT) (U) solves " + tag + " case " + std::to_string(cs);
        const uint32_t m = 400 + static_cast<uint32_t>(r.next() % 200), nb = 60 + static_cast<uint32_t>(r.next() % 60);
        const uint32_t ny = 128 + static_cast<uint32_t>(r.next() % 64);
        // B with B[piv, :] = I (piv = every 3rd row from 1), the rest random; X random; Y = B X (+ a planted defect)
        DenseBuf<Elt> B = RandVal<Elt>(ctx, r, m, nb, 1);
        std::vector<uint32_t> piv;
        for (uint32_t c = 0; c < nb; ++c) piv.push_back(1 + 3 * c);
        for (uint32_t c = 0; c < nb; ++c)
            for (uint32_t j = 0; j < nb; ++j) B.at(piv[c], j) = (c == j) ? Elt(1) : Elt(0);
        const DenseBuf<Elt> X0 = RandVal<Elt>(ctx, r, nb, ny, 1);
        DenseBuf<Elt> Y = MulScalar<Elt>(ctx, B, X0);
        const bool residual = (cs % 3) == 2, bad_rhs = (cs % 2) == 1;
        if (residual) Y.at(2, ny / 2) = ctx.add(Y.at(2, ny / 2), Elt(1));   // row 2 is not a pivot row: Residual
        if (bad_rhs) {                                                               // off the image for two rhs (the smaller throws)
            Y.at(0, ny - 3) = ctx.add(Y.at(0, ny - 3), Elt(1));
            Y.at(0, ny - 1) = ctx.add(Y.at(0, ny - 1), Elt(1));
        }
        // the references: lever off, pool 1
        DenseBuf<Elt> Pref(m, ny), Xref;
        el::ReadoffStatus st_ref;
        std::string msg_ref;
        std::vector<std::vector<Elt>> co_ref;
        {
            engine::tape::ThreadPool pool(1);
            const engine::tape::ScopedConsumerPool sp(&pool);
            const el::ScopedModuleMt off(false);
            engine::kernels::gemm<Elt>(ctx, engine::cview(B), engine::cview(X0), engine::view(Pref));
            st_ref = el::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(B), piv, engine::cview(Y), Xref);
            try { co_ref = el::solve_multi_rhs_module<Elt>(ctx, engine::cview(B), engine::cview(Y)); }
            catch (const std::runtime_error& e) { msg_ref = e.what(); }
        }
        Check(residual == (st_ref == el::ReadoffStatus::Residual) || bad_rhs, t + ": the reference read-off status is the planted one");
        Check(bad_rhs == !msg_ref.empty() || residual, t + ": the reference solve throws iff an rhs is off the image");
        for (uint32_t w : MtSizes()) {
            engine::tape::ThreadPool pool(w);
            const engine::tape::ScopedConsumerPool sp(&pool);
            const engine::tape::ScopedHostElimHook hook;
            const el::ScopedModuleMt on(true);
            const std::string tw = t + " pool " + std::to_string(w);
            DenseBuf<Elt> P(m, ny), X;
            el::gemm_rows_hook<Elt>(el::module_mt_hook(), ctx, engine::cview(B), engine::cview(X0), engine::view(P));
            Check(Same(P, Pref), tw + ": gemm_rows_hook = kernels::gemm entrywise");
            const el::ReadoffStatus st = el::solve_unit_pivot_readoff<Elt>(ctx, engine::cview(B), piv, engine::cview(Y), X);
            Check(st == st_ref && (st != el::ReadoffStatus::Ok || Same(X, Xref)), tw + ": the read-off's status and X");
            std::string msg;
            std::vector<std::vector<Elt>> co;
            try { co = el::solve_multi_rhs_module<Elt>(ctx, engine::cview(B), engine::cview(Y)); }
            catch (const std::runtime_error& e) { msg = e.what(); }
            Check(msg == msg_ref && (!msg.empty() || co == co_ref), tw + ": solve_multi_rhs_module: the same coordinates, or the same first message ('" + msg + "')");
        }
    }
    const MtCounts d = MtCounts::now().minus(c0);
    Check(d.v[2] > 0, "(MT) (U) solves " + tag + ": the row-block / rhs splits ran (" + d.txt() + ")");
}

template<class Elt>
void MtObjectArm(uint32_t n, uint64_t ell, uint32_t K, FibreRepresentation rep = FibreRepresentation::SymN, bool expect_check = false,
                 bool expect_search = false) {
    const bool hida = rep == FibreRepresentation::HidaLatticeH;
    const std::string t0 = std::string("(MT) (O) (") + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                           (hida ? " Hida L_n" : "") + (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    std::unique_ptr<ObjRun<Elt>> ref;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleMt off(false);
        ref = RunObjects<Elt>(n, ell, K, true, rep);
    }
    std::ostringstream secs;
    secs << " ref=" << Fx(ref->seconds, 3);
    for (uint32_t w : MtSizes()) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleMt on(true);
        const el::ScopedModuleTileWork extent(0);   // (S24 PAR-L1) the extent-only tiling, so the splits stay exercised here
        const MtCounts c0 = MtCounts::now();
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        const MtCounts d = MtCounts::now().minus(c0);
        const std::string t = t0 + " pool " + std::to_string(w);
        if (w == 1) Check(d.total() == 0, t + ": no split region at one worker (" + d.txt() + ")");
        else Check(d.v[4] > 0, t + ": the (sigma, sigma_k) assembly tasks ran (" + d.txt() + ")");
        Check(d.v[5] > 0, t + ": the part-(0) pivot-search exit ran (" + d.txt() + ")");   // (MT-1 audit F2)
        if (w > 1 && expect_check) Check(d.v[3] > 0 && d.v[2] > 0, t + ": the per-column checks and the row-block products ran (" + d.txt() + ")");
        if (w > 1 && expect_search) Check(d.v[0] > 0, t + ": cyclic_decomposition's tiled pivot search ran (" + d.txt() + ")");
        secs << " w" << w << "=" << Fx(r->seconds, 3) << "[" << d.txt() << "]";
        CompareObjects<Elt>(t + " vs the pre-MT route", *r, *ref);
        Check(SameAssemblyCounts(r->delta, ref->delta), t + ": the stream counters equal the pre-MT route's (" + Txt(r->delta) + "; " +
                                                            Txt(ref->delta) + ")");
    }
    {
        engine::tape::ThreadPool pool(7);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleMt off(false);
        const MtCounts c0 = MtCounts::now();
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        Check(MtCounts::now().minus(c0).all() == 0, t0 + " pool 7 lever off: no MT region, no pivot-search exit");
        CompareObjects<Elt>(t0 + " pool 7 lever off vs the pre-MT route", *r, *ref);
    }
    {   // (S24 PAR-L1) the default tile-work bound at 16 workers: the same objects as the pre-MT route
        engine::tape::ThreadPool pool(16);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleMt on(true);
        const el::ScopedModuleTileWork bound(size_t(1) << 20);
        auto r = RunObjects<Elt>(n, ell, K, true, rep);
        CompareObjects<Elt>(t0 + " pool 16 tile-work bound 2^20 vs the pre-MT route", *r, *ref);
    }
    std::cout << "MODULE_MT_CELL " << t0 << " seconds" << secs.str() << "\n";
}

void MtCellArm(uint32_t n, uint64_t ell, uint32_t kmax) {
    const std::string t0 = "(MT) (C) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax);
    std::string ref;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleMt off(false);
        ref = RunCell(n, ell, kmax, true).text;
    }
    std::ostringstream secs;
    for (uint32_t w : MtSizes()) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleMt on(true);
        const MtCounts c0 = MtCounts::now();
        const CellOut c = RunCell(n, ell, kmax, true);
        const MtCounts d = MtCounts::now().minus(c0);
        Check(d.v[5] > 0 && (w > 1 || d.total() == 0), t0 + " pool " + std::to_string(w) + ": the pivot-search exit ran, no split at one worker (" + d.txt() + ")");
        secs << " w" << w << "=" << Fx(c.seconds, 3);
        const bool same = c.text == ref;
        Check(same, t0 + " pool " + std::to_string(w) + " vs the pre-MT route: the SAT_DEFECT lines are byte-identical; first difference " +
                        (same ? std::string("(none)") : FirstDiff(c.text, ref)));
    }
    size_t lines = 0;
    for (char ch : ref) lines += ch == '\n';
    Check(lines >= 2, t0 + ": at least one block line (non-vacuity)");
    std::cout << "MODULE_MT_CELL " << t0 << " lines=" << lines << " seconds" << secs.str() << "\n";
}

// ---- (H) (S24 G2-H) the lean-peak lever HECKE_MODULE_LEAN_PEAK against its rollback ---------------------------------
struct LpCounts {
    uint64_t v[6] = {};
    static LpCounts now() {
        LpCounts c;
        for (uint32_t i = 0; i < 6; ++i) c.v[i] = el::lean_peak_events(static_cast<el::LeanPeakEvent>(i));
        return c;
    }
    LpCounts minus(const LpCounts& o) const {
        LpCounts d;
        for (uint32_t i = 0; i < 6; ++i) d.v[i] = v[i] - o.v[i];
        return d;
    }
    uint64_t all() const { uint64_t s = 0; for (uint64_t x : v) s += x; return s; }
    std::string txt() const {
        return "adopt=" + std::to_string(v[0]) + " copy=" + std::to_string(v[1]) + " residual=" + std::to_string(v[2]) +
               " route2_inplace=" + std::to_string(v[3]) + " early_release=" + std::to_string(v[4]) + " length_only=" +
               std::to_string(v[5]);
    }
};

// (H) (a) unit level: unit_pivot_residual_product's status == solve_unit_pivot_readoff_product's on random unit-pivot B and
// products A G of six kinds -- consistent (A = B Z), off im B at a non-pivot row, at a pivot row, a torsion perturbation
// ell^{k-1} E (in or off im B depending on E G), the perturb_row hook at a pivot and a non-pivot row, the corruption hook,
// B[piv, :] != I, l = 0, nb = 0 and nb = m -- for row blocks 1, 3, 64, 256, serially and over a 7-worker hook.
template<class Elt>
void UnitResidual(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    const std::string t0 = "(H) (a) " + tag;
    Rng r(seed);
    engine::tape::ThreadPool pool(7);
    const el::ParallelFor hook = [&pool](uint32_t nt, const std::function<void(uint32_t)>& fn) {
        pool.parallel_for(nt, pool.size(), [&](uint32_t t, uint32_t) { fn(t); });
    };
    uint32_t cases = 0, oks = 0, residuals = 0, agree = 0;
    const Elt torsion = ctx.ell_pow(ctx.k - 1);
    for (uint32_t shape = 0; shape < 5; ++shape) {
        // shape 0: small; 1: tall (several 128-row tiles at the hook threshold); 2: nb = 0; 3: nb = m;
        // 4 (G2-H audit F1): nb = 150 > 2 * 64 (two full pivot blocks of 64 and a partial one at row blocks 64 / 256) and
        // m - nb = 550 non-pivot rows (four full 128-row tiles and a partial fifth)
        const uint32_t m = shape == 1 ? 600 : (shape == 4 ? 700 : 37), l = shape == 1 ? 70 : (shape == 4 ? 160 : 13),
                       ny = shape == 1 ? 40 : (shape == 4 ? 30 : 9);
        const uint32_t nb = shape == 2 ? 0u : (shape == 3 ? m : (shape == 1 ? 60u : (shape == 4 ? 150u : 11u)));
        std::vector<uint32_t> piv;
        for (uint32_t i = 0; i < m && piv.size() < nb; ++i)
            if ((r.next() % 3) == 0 || m - i == nb - piv.size()) piv.push_back(i);
        std::vector<char> is_piv(m, 0);
        for (uint32_t c = 0; c < nb; ++c) is_piv[piv[c]] = 1;
        DenseBuf<Elt> B(m, nb), Z(nb, l), G(l, ny);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t c = 0; c < nb; ++c) B.at(i, c) = Below<Elt>(r, ctx.p);
        for (uint32_t rr = 0; rr < nb; ++rr)
            for (uint32_t c = 0; c < nb; ++c) B.at(piv[rr], c) = (rr == c) ? Elt(1) : Elt(0);
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < l; ++j) Z.at(i, j) = Below<Elt>(r, ctx.p);
        for (uint32_t i = 0; i < l; ++i)
            for (uint32_t t = 0; t < ny; ++t) G.at(i, t) = Below<Elt>(r, ctx.p);
        DenseBuf<Elt> A0(m, l);
        if (nb > 0) engine::kernels::gemm<Elt>(ctx, engine::cview(B), engine::cview(Z), engine::view(A0));
        uint32_t first_np = 0, first_p = nb > 0 ? piv[0] : ~0u;
        while (first_np < m && is_piv[first_np]) ++first_np;
        // (G2-H audit F1) the LAST non-pivot row (in the last row tile / the last hook range) and the last pivot row
        uint32_t last_np = m;
        for (uint32_t i = m; i-- > 0;)
            if (!is_piv[i]) { last_np = i; break; }
        const uint32_t last_p = nb > 0 ? piv[nb - 1] : ~0u;
        for (uint32_t kind = 0; kind < 11; ++kind) {
            if (shape == 4 && kind != 0 && kind != 8 && kind != 9 && kind != 10) continue;   // the new shape: the new kinds
            DenseBuf<Elt> A = A0;
            uint32_t perturb = ~0u;
            bool corrupt = false;
            if (kind == 1 && first_np < m) A.at(first_np, l - 1) = ctx.add(A.at(first_np, l - 1), Elt(1));
            if (kind == 2 && nb > 0) A.at(first_p, 0) = ctx.add(A.at(first_p, 0), Elt(1));
            if (kind == 3) {   // torsion: ell^{k-1} times a random row block
                for (uint32_t i = 0; i < m; i += 5)
                    for (uint32_t j = 0; j < l; ++j) A.at(i, j) = ctx.add(A.at(i, j), ctx.mul(torsion, Below<Elt>(r, ctx.p)));
            }
            if (kind == 4 && first_np < m) perturb = first_np;
            if (kind == 5 && nb > 0) perturb = first_p;
            if (kind == 6) corrupt = true;
            // (G2-H audit F1) faults confined to the last non-pivot row / the last pivot block: a residual that skipped
            // the last tile, the last hook range or the last (partial) pivot block would pass them
            if (kind == 8 && last_np < m) A.at(last_np, l / 2) = ctx.add(A.at(last_np, l / 2), Elt(1));
            if (kind == 9 && last_np < m) perturb = last_np;
            if (kind == 10 && nb > 0) A.at(last_p, l - 1) = ctx.add(A.at(last_p, l - 1), Elt(1));
            DenseBuf<Elt> Bx = B;
            if (kind == 7 && nb >= 2) Bx.at(piv[1], 0) = Elt(1);   // B[piv, :] != I
            for (uint32_t rb : {1u, 3u, 64u, 256u}) {
                for (int h = 0; h < 2; ++h) {
                    std::unique_ptr<el::ScopedParallelHook> sh;
                    if (h == 1) sh.reset(new el::ScopedParallelHook(&hook));
                    std::unique_ptr<el::ScopedReadoffCorruption> cor;
                    if (corrupt) cor.reset(new el::ScopedReadoffCorruption(true));
                    DenseBuf<Elt> X;
                    const el::ReadoffStatus a = el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(Bx), piv, engine::cview(A),
                                                                                         engine::cview(G), X, rb, perturb);
                    const el::ReadoffStatus b = el::unit_pivot_residual_product<Elt>(ctx, engine::cview(Bx), piv, engine::cview(A),
                                                                                    engine::cview(G), rb, perturb);
                    ++cases;
                    if (a == b) ++agree;
                    if (b == el::ReadoffStatus::Ok) ++oks;
                    if (b == el::ReadoffStatus::Residual) ++residuals;
                    if (a != b)
                        Check(false, t0 + " shape " + std::to_string(shape) + " kind " + std::to_string(kind) + " rb " + std::to_string(rb) +
                                         (h ? " hook" : " serial") + ": the residual's status == the product read-off's");
                }
            }
        }
        // l = 0: Y = 0, in im B; with the non-pivot perturbation off it
        if (shape == 0) {
            DenseBuf<Elt> A(m, 0), G0(0, ny), X;
            Check(el::unit_pivot_residual_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A), engine::cview(G0), 7) ==
                          el::ReadoffStatus::Ok &&
                      el::unit_pivot_residual_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A), engine::cview(G0), 7, first_np) ==
                          el::solve_unit_pivot_readoff_product<Elt>(ctx, engine::cview(B), piv, engine::cview(A), engine::cview(G0), X, 7,
                                                                    first_np),
                  t0 + ": l = 0 (Y = 0) Ok, and its non-pivot perturbation as the product read-off");
            std::vector<uint32_t> bad = piv;
            std::swap(bad[0], bad[1]);
            Check(el::unit_pivot_residual_product<Elt>(ctx, engine::cview(B), bad, engine::cview(A0), engine::cview(G)) ==
                      el::ReadoffStatus::NotIdentity,
                  t0 + ": non-increasing piv -> NotIdentity");
        }
    }
    Check(agree == cases && oks > 0 && residuals > 0,
          t0 + ": " + std::to_string(agree) + " / " + std::to_string(cases) + " statuses agree (Ok " + std::to_string(oks) +
              ", Residual " + std::to_string(residuals) + ": both met)");
}

// (H) (b) unit level: howell / kernel_module / image_module / cyclic_decomposition with the lever on == off, field by field;
// the adopted store is taken on unit-pivot inputs (adopt >= 1) and the copy where closure rows appear (ell * random: copy >= 1).
template<class Elt>
void UnitHowellAdopt(const engine::ModCtx<Elt>& ctx, const std::string& tag, uint64_t seed) {
    const std::string t0 = "(H) (b) " + tag;
    Rng r(seed);
    auto same_h = [](const el::HowellResult<Elt>& a, const el::HowellResult<Elt>& b) {
        return Same(a.H, b.H) && a.pivots == b.pivots && a.pivot_v == b.pivot_v;
    };
    uint64_t adopt = 0, copy = 0, lonly = 0;
    bool eq = true;
    for (uint32_t c = 0; c < 24; ++c) {
        const uint32_t m = 3 + static_cast<uint32_t>(r.next() % 30), n = 3 + static_cast<uint32_t>(r.next() % 30);
        DenseBuf<Elt> A(m, n), Rl(m, 1 + static_cast<uint32_t>(r.next() % 6));
        const bool tors = (c % 3) == 2;
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < n; ++j)
                A.at(i, j) = tors ? ctx.mul(static_cast<Elt>(ctx.ell), Below<Elt>(r, ctx.p)) : Below<Elt>(r, ctx.p);
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < Rl.cols; ++j) Rl.at(i, j) = ctx.mul(static_cast<Elt>(ctx.ell), Below<Elt>(r, ctx.p));
        el::HowellResult<Elt> h[2][3];
        el::CyclicDecomposition<Elt> cd[2];
        std::vector<uint32_t> inf[2];
        for (int on = 0; on < 2; ++on) {
            const el::ScopedModuleStream ms(true);
            const el::ScopedModuleLeanPeak lp(on == 1);
            const LpCounts c0 = LpCounts::now();
            h[on][0] = el::howell<Elt>(ctx, engine::cview(A));
            h[on][1] = el::kernel_module<Elt>(ctx, engine::cview(A));
            h[on][2] = el::image_module<Elt>(ctx, engine::cview(A));
            cd[on] = el::cyclic_decomposition<Elt>(ctx, engine::cview(A), engine::cview(Rl));
            inf[on] = el::invariant_factors<Elt>(ctx, engine::cview(A), engine::cview(Rl));
            const LpCounts d = LpCounts::now().minus(c0);
            if (on == 0) eq = eq && d.all() == 0;
            else { adopt += d.v[0]; copy += d.v[1]; lonly += d.v[5]; }
        }
        for (int k = 0; k < 3; ++k) eq = eq && same_h(h[0][k], h[1][k]);
        eq = eq && cd[0].exponents == cd[1].exponents && cd[0].position == cd[1].position && Same(cd[0].basis, cd[1].basis) &&
             Same(cd[0].Gamma, cd[1].Gamma) && Same(cd[0].U, cd[1].U) && inf[0] == inf[1];
    }
    Check(eq, t0 + ": Howell / kernel / image / cyclic_decomposition / invariant_factors equal with the lever on and off; no "
                   "lever event with it off");
    Check(adopt > 0 && copy > 0 && lonly > 0, t0 + ": the adopted store, the copy and the pivot-loop length all taken (adopt=" +
                                                  std::to_string(adopt) + " copy=" + std::to_string(copy) + " length_only=" +
                                                  std::to_string(lonly) + ")");
}

// (H) (c) object level, lean retention (as RunModuleT): the lever on (pools 1 and 7) against off and against the legacy
// route -- every kept cohomology object, A^{(q)}, the certificate bundle -- and the lever's counters: residual and
// route2_inplace once per operator degree, early_release once, none of them with the lever off.
template<class Elt>
void LeanPeakObjectArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t0 = "(H) (c) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                           (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    std::unique_ptr<ObjRun<Elt>> off, leg;
    LpCounts doff;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleLeanPeak lp(false);
        const LpCounts c0 = LpCounts::now();
        off = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, false, /*lean=*/true);
        doff = LpCounts::now().minus(c0);
        leg = RunObjects<Elt>(n, ell, K, false, FibreRepresentation::SymN);
    }
    Check(doff.all() == 0, t0 + " lever off: no lever event (" + doff.txt() + ")");
    auto compare = [&](const std::string& t, const ObjRun<Elt>& a, const ObjRun<Elt>& b, bool b_lean) {
        for (uint32_t q = 0; q < 3; ++q) {
            const auto& A = a.coh->module_degree(q);
            const auto& B = b.coh->module_degree(q);
            Check(Same(A.C, B.C) && A.piv == B.piv && Same(A.H, B.H) && A.dec.exponents == B.dec.exponents &&
                      A.dec.position == B.dec.position && Same(A.dec.basis, B.dec.basis) && Same(A.dec.Gamma, B.dec.Gamma) &&
                      Same(A.dec.U, B.dec.U) && A.dK == B.dK && A.length_H == B.length_H,
                  t + " q=" + std::to_string(q) + ": the kept cohomology objects equal");
            Check(A.X.rows == 0 && A.X.cols == 0 && A.Zc.rows == 0 && A.Zc.cols == 0 && (!b_lean || (B.X.rows == 0 && B.Zc.rows == 0)),
                  t + " q=" + std::to_string(q) + ": X, Zc released (0 x 0)");
        }
        Check(a.coh->module_intermediates_released(), t + ": the cohomology object ends released");
        for (size_t i = 0; i < a.ops.size() && i < b.ops.size(); ++i) {
            const auto& x = *a.ops[i];
            const auto& y = *b.ops[i];
            const std::string tp = t + " T_" + std::to_string(a.primes[i]);
            for (uint32_t q = 0; q < 3; ++q)
                Check(Same(x.H_matrix_module(q), y.H_matrix_module(q)), tp + " q=" + std::to_string(q) + " A^{(q)}");
            const auto ca = x.certificates(), cb = y.certificates();
            Check(ca.routes_bit_identical && cb.routes_bit_identical && ca.relation_checks == cb.relation_checks &&
                      ca.cocycle_checks == cb.cocycle_checks && ca.theta_eq_solves == cb.theta_eq_solves &&
                      ca.theta_eq_residual_max == 0 && cb.theta_eq_residual_max == 0 && ca.theta_eq_certified && cb.theta_eq_certified,
                  tp + " certificate bundle equal (O12(a), (i), (ii), (iii))");
            Check(x.module_objects_released(), tp + " released on the lean operator");
        }
    };
    for (uint32_t w : {1u, 7u}) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleLeanPeak lp(true);
        const LpCounts c0 = LpCounts::now();
        const auto on = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, false, /*lean=*/true);
        const LpCounts d = LpCounts::now().minus(c0);
        const std::string t = t0 + " pool " + std::to_string(w);
        compare(t + " lever on vs off", *on, *off, true);
        compare(t + " lever on vs the legacy route", *on, *leg, false);
        const uint64_t degs = 3 * on->ops.size();
        Check(d.v[2] == degs && d.v[3] == degs && d.v[4] == 1 && d.v[5] >= 1 && d.v[0] >= 1,
              t + ": residual and route2_inplace once per operator degree (" + std::to_string(degs) +
                  "), early_release once, the pivot-loop length and the adopted store taken (" + d.txt() + ")");
        Check(on->delta.w_nonpivot == 0 && on->delta.readoff >= degs && on->delta.lean_released == 1 + on->ops.size(),
              t + ": no product read-off (w=nonpivot 0), every (iii) a read-off certificate, retention=lean counted once per object (" +
                  Txt(on->delta) + ")");
    }
}

// (H) (d) the negative controls on the lean route, lever on against off: the corrupted read-off certificate falls back to
// the Howell solve and leaves every object unchanged; an inconsistent Theta_eq throws the (iii) text, a corrupted route-2
// entry (three sites) the O12(a) text -- byte for byte the lever-off texts.
template<class Elt>
void LeanPeakNegativeArm(uint32_t n, uint64_t ell, uint32_t K) {
    const std::string t0 = "(H) (d) (" + std::to_string(n) + ", " + std::to_string(ell) + ", K=" + std::to_string(K) + ")" +
                           (std::is_same<Elt, uint64_t>::value ? " u64" : " u32");
    {
        std::unique_ptr<ObjRun<Elt>> run[2];
        el::ModuleStreamTotals d[2];
        for (int on = 0; on < 2; ++on) {
            const el::ScopedModuleLeanPeak lp(on == 1);
            run[on] = RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, /*corrupt=*/true, /*lean=*/true);
            d[on] = run[on]->delta;
        }
        for (uint32_t i = 0; i < run[0]->ops.size(); ++i)
            for (uint32_t q = 0; q < 3; ++q)
                Check(Same(run[0]->ops[i]->H_matrix_module(q), run[1]->ops[i]->H_matrix_module(q)),
                      t0 + " corrupted certificate: T_" + std::to_string(run[0]->primes[i]) + " q=" + std::to_string(q) + " A^{(q)} equal");
        Check(d[1].howell >= 3 * run[1]->ops.size() && d[1].howell == d[0].howell,
              t0 + " corrupted certificate: every operator degree fell back to the Howell solve, as with the lever off (" + Txt(d[1]) +
                  " | " + Txt(d[0]) + ")");
    }
    {
        std::string msg[2];
        for (int on = 0; on < 2; ++on) {
            const el::ScopedModuleLeanPeak lp(on == 1);
            const hecke::ScopedThetaEqPerturbation pert(true);
            try {
                (void)RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, false, /*lean=*/true);
            } catch (const std::runtime_error& e) { msg[on] = e.what(); }
        }
        Check(!msg[0].empty() && msg[0].rfind("HeckeOperatorOnK (Z/ell^k): runtime assertion (iii) failed -- ", 0) == 0 && msg[1] == msg[0],
              t0 + " inconsistent Theta_eq: the (iii) text, lever on == off, got '" + msg[1] + "' / '" + msg[0] + "'");
    }
    const struct { hecke::Route2Site site; const char* name; const char* q; } sites[] = {
        {hecke::Route2Site::Degree0Last, " (q=0, last entry)", "0"},
        {hecke::Route2Site::Degree1FirstChunk, " (q=1, first chunk)", "1"},
        {hecke::Route2Site::Degree1LastChunk, " (q=1, last chunk)", "1"},
        {hecke::Route2Site::Degree1Middle, " (q=1, interior block row sigma = 3)", "1"},   // (G2-H audit F2)
    };
    for (const auto& st : sites) {
        for (int memo = 0; memo < 2; ++memo) {   // the shared memo (one chunk) and G2-D's two chunks
            const el::ScopedRoute2SharedMemo sm(memo == 0);
            std::string msg[2];
            for (int on = 0; on < 2; ++on) {
                const el::ScopedModuleLeanPeak lp(on == 1);
                const hecke::ScopedRoute2Perturbation pert(st.site);
                try {
                    (void)RunObjects<Elt>(n, ell, K, true, FibreRepresentation::SymN, false, /*lean=*/true);
                } catch (const std::runtime_error& e) { msg[on] = e.what(); }
            }
            const std::string want = std::string("HeckeOperatorOnK (Z/ell^k): the fused route and route 2 differ at q=") + st.q +
                                     " (O12(a) over Z/ell^k)";
            Check(msg[0] == want && msg[1] == want, t0 + " corrupted route-2 entry" + st.name + (memo ? " chunked" : " shared") +
                                                        ": the O12(a) text, lever on == off, got '" + msg[1] + "'");
        }
    }
}

// (H) (e) cell level: the SAT_DEFECT lines with the lever on at pools 1 / 4 / 16, and with HECKE_MODULE_MT off, against the
// lever off at one worker, byte for byte.
void LeanPeakCellArm(uint32_t n, uint64_t ell, uint32_t kmax) {
    const std::string t0 = "(H) (e) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax);
    std::string ref;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleLeanPeak lp(false);
        ref = RunCell(n, ell, kmax, true).text;
    }
    std::ostringstream secs;
    for (int variant = 0; variant < 4; ++variant) {
        const uint32_t w = variant == 0 ? 1u : (variant == 1 ? 4u : 16u);
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const el::ScopedModuleLeanPeak lp(true);
        const el::ScopedModuleMt mt(variant != 3);
        const LpCounts c0 = LpCounts::now();
        const CellOut c = RunCell(n, ell, kmax, true);
        const LpCounts d = LpCounts::now().minus(c0);
        const std::string t = t0 + " pool " + std::to_string(w) + (variant == 3 ? " HECKE_MODULE_MT off" : "");
        Check(d.v[2] >= 1 && d.v[3] >= 1 && d.v[4] >= 1, t + ": the lever's lean parts ran (" + d.txt() + ")");
        const bool same = c.text == ref;
        Check(same, t + " vs the lever off: the SAT_DEFECT lines are byte-identical; first difference " +
                        (same ? std::string("(none)") : FirstDiff(c.text, ref)));
        secs << " v" << variant << "=" << Fx(c.seconds, 3);
    }
    std::cout << "MODULE_LEAN_PEAK_CELL " << t0 << " seconds" << secs.str() << "\n";
}

// ---- (S24 SATPERF) the sat cell's post-module lever HECKE_SAT_PERF against its rollback ----------------------------
// The reference is the rollback (ScopedSatPerf(false): no memo, the serial u128 End(N) product -- the route of
// 4dd1a7e0) on ONE worker; the lever on runs at pools 1 / 2 / 4.  Every SAT_DEFECT line byte for byte, K_run / Lv_run,
// and the lever's counters: memo hits on, none off; narrow (uint64_t) products on, none off; row-split products at
// pool > 1 when r >= 24 (n >= 142), none at one worker.
struct SpDelta {
    uint64_t hits = 0, misses = 0, narrow = 0, wide = 0, par = 0;
    std::string txt() const {
        return "memo=" + std::to_string(hits) + "/" + std::to_string(misses) + " mul=narrow:" + std::to_string(narrow) +
               ",wide:" + std::to_string(wide) + ",parallel:" + std::to_string(par);
    }
};
SpDelta SpSince(const hecke_ext::ringalg::SatPerfCounters& a) {
    const hecke_ext::ringalg::SatPerfCounters b = hecke_ext::ringalg::SatPerfCounts();
    SpDelta d;
    d.hits = b.memo_hits - a.memo_hits;
    d.misses = b.memo_misses - a.memo_misses;
    d.narrow = b.mul_narrow - a.mul_narrow;
    d.wide = b.mul_wide - a.mul_wide;
    d.par = b.mul_parallel - a.mul_parallel;
    return d;
}
void SatPerfCellArm(uint32_t n, uint64_t ell, uint32_t kmax, bool expect_split, bool expect_wide) {
    namespace ra = hecke_ext::ringalg;
    const std::string t0 = "(SP) (" + std::to_string(n) + ", " + std::to_string(ell) + ") KMAX " + std::to_string(kmax);
    CellOut ref;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const ra::ScopedSatPerf off(false);
        const ra::SatPerfCounters c0 = ra::SatPerfCounts();
        ref = RunCell(n, ell, kmax, true);
        const SpDelta d = SpSince(c0);
        Check(d.hits == 0 && d.misses == 0 && d.narrow == 0 && d.wide == 0 && d.par == 0,
              t0 + " lever off: the legacy route only (" + d.txt() + ")");
    }
    std::ostringstream secs;
    secs << " off_w1=" << Fx(ref.seconds, 3);
    for (uint32_t w : {1u, 2u, 4u}) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const ra::ScopedSatPerf on(true);
        const ra::SatPerfCounters c0 = ra::SatPerfCounts();
        const CellOut c = RunCell(n, ell, kmax, true);
        const SpDelta d = SpSince(c0);
        const std::string t = t0 + " pool " + std::to_string(w);
        Check(d.hits > 0 && d.misses > 0, t + ": the memo served the certificate (" + d.txt() + ")");
        Check(d.narrow > 0, t + ": uint64_t products ran (" + d.txt() + ")");
        if (expect_wide) Check(d.wide > 0, t + ": u128 products ran (ell^K wide) (" + d.txt() + ")");
        if (w == 1) Check(d.par == 0, t + ": no row split at one worker (" + d.txt() + ")");
        else if (expect_split) Check(d.par > 0, t + ": the whole-module products split over the pool (" + d.txt() + ")");
        const bool same = c.text == ref.text;
        Check(same, t + " vs the rollback: the SAT_DEFECT lines are byte-identical; first difference " +
                        (same ? std::string("(none)") : FirstDiff(c.text, ref.text)));
        Check(c.K_run == ref.K_run && c.Lv_run == ref.Lv_run, t + ": K_run, Lv_run equal (" + Txt(c.K_run) + ", " + Txt(c.Lv_run) + ")");
        secs << " on_w" << w << "=" << Fx(c.seconds, 3);
    }
    size_t lines = 0;
    for (char ch : ref.text) lines += ch == '\n';
    Check(lines >= 2, t0 + ": at least one block line (non-vacuity)");
    std::cout << "SAT_PERF_CELL " << t0 << " lines=" << lines << " seconds" << secs.str() << "\n";
}

// (SP) (b) a planted module over Z/ell^K with entries near ell^K (so the products take mul_fast's u128 rows as well as its
// uint64_t ones): T = an upper-triangular r x r matrix with eigenvalues {1, 2, 3} mod ell and random entries above the
// diagonal, the ops {T, T^2 + 3T} (commuting).  RationalSystems, Localize (every block: exponents, ops) and ReadAlgebra of
// every block with the lever on at pools 1 / 4 against the lever off at one worker, entry for entry.
std::string PlantedSatPerfText(uint64_t ell, uint32_t K, uint32_t r, uint64_t seed) {
    namespace ra = hecke_ext::ringalg;
    uint64_t q = 1;
    for (uint32_t i = 0; i < K; ++i) q *= ell;
    uint64_t s = seed;
    auto rnd = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (s >> 11) % q; };
    std::vector<uint64_t> T(static_cast<size_t>(r) * r, 0);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = i; j < r; ++j) T[static_cast<size_t>(i) * r + j] = j == i ? (1 + i % 3 + ell * rnd()) % q : rnd();
    auto mulq = [&](const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) {
        std::vector<uint64_t> C(A.size(), 0);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t k = 0; k < r; ++k)
                for (uint32_t j = 0; j < r; ++j)
                    C[static_cast<size_t>(i) * r + j] = static_cast<uint64_t>(
                        (static_cast<unsigned __int128>(A[static_cast<size_t>(i) * r + k]) * B[static_cast<size_t>(k) * r + j] +
                         C[static_cast<size_t>(i) * r + j]) % q);
        return C;
    };
    std::vector<uint64_t> T2 = mulq(T, T);
    for (size_t k = 0; k < T2.size(); ++k) T2[k] = static_cast<uint64_t>((static_cast<unsigned __int128>(T[k]) * 3 + T2[k]) % q);
    ra::OpModule M;
    M.ell = ell;
    M.exps.assign(r, K);
    M.primes = {2, 3};
    for (const auto* A : {&T, &T2}) {
        ra::RingMat R;
        R.ell = ell;
        R.row_exp = M.exps;
        R.a = *A;
        M.op.push_back(R);
    }
    std::ostringstream os;
    for (const ra::JointSystem& js : ra::RationalSystems(M)) {
        os << "system=" << Txt(std::vector<uint32_t>(js.a.begin(), js.a.end())) << " dim=" << js.dim;
        const ra::Block B = ra::Localize(M, js.a);
        os << " exps=" << Txt(B.local.exps) << " e=";
        uint64_t h = 1469598103934665603ull;
        for (uint64_t x : B.e.a) h = (h ^ x) * 1099511628211ull;
        for (const ra::RingMat& X : B.local.op)
            for (uint64_t x : X.a) h = (h ^ x) * 1099511628211ull;
        os << h;
        const ra::AlgebraReading A = ra::ReadAlgebra(B.local, js.a);
        os << " length=" << A.length << " dim=" << A.dim_mod_ell << " hilbert=" << Txt(A.hilbert) << "\n";
    }
    return os.str();
}
void SatPerfPlantedArm(uint64_t ell, uint32_t K, uint32_t r, uint64_t seed, bool expect_wide) {
    namespace ra = hecke_ext::ringalg;
    const std::string t0 = "(SP) planted Z/" + std::to_string(ell) + "^" + std::to_string(K) + " r=" + std::to_string(r);
    std::string ref;
    {
        engine::tape::ThreadPool pool(1);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const ra::ScopedSatPerf off(false);
        ref = PlantedSatPerfText(ell, K, r, seed);
    }
    for (uint32_t w : {1u, 4u}) {
        engine::tape::ThreadPool pool(w);
        const engine::tape::ScopedConsumerPool sp(&pool);
        const engine::tape::ScopedHostElimHook hook;   // as the sat cell installs it around its post-module stages
        const ra::ScopedSatPerf on(true);
        const ra::SatPerfCounters c0 = ra::SatPerfCounts();
        const std::string got = PlantedSatPerfText(ell, K, r, seed);
        const SpDelta d = SpSince(c0);
        const std::string t = t0 + " pool " + std::to_string(w);
        Check(d.narrow > 0 && (d.wide > 0) == expect_wide, t + ": uint64_t rows ran, u128 rows " + (expect_wide ? "ran" : "did not") + " (" + d.txt() + ")");
        Check(w == 1 ? d.par == 0 : d.par > 0, t + ": the row split " + (w == 1 ? "did not run" : "ran") + " (" + d.txt() + ")");
        Check(got == ref, t + " vs the rollback: every system, block and reading equal; first difference " +
                              (got == ref ? std::string("(none)") : FirstDiff(got, ref)));
    }
    size_t lines = 0;
    for (char ch : ref) lines += ch == '\n';
    Check(lines >= 2, t0 + ": at least two systems (non-vacuity)");
    std::cout << "SAT_PERF_PLANTED " << t0 << " systems=" << lines << "\n";
}

} // namespace

uint32_t HeckeModuleStreamValidation() {
    g_fail = g_checks = 0;
    g_oracle_injection_caught = g_oracle_injection_cases = 0;
    const double t0 = Now();
    std::cout << "=== HeckeModuleStreamValidation (slot-23 G2-A..E: the module-stream levers vs the legacy route) ===\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_STREAM " << el::module_stream_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_EVAL " << el::module_eval_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_PARALLEL " << el::module_parallel_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_RHO_SIMD " << rho_fill_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_ROUTE2_SHARED_MEMO " << el::route2_shared_memo_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_LEAN_PEAK " << el::module_lean_peak_effective_text() << "\n";
    try {
        // (U) / (N) unit level
        UnitReadoff<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 11);
        UnitReadoff<uint32_t>(engine::ModCtx<uint32_t>::prime_power(7u, 2), "Z/7^2 u32", 12);
        UnitReadoff<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 5), "Z/2^5 u32", 13);
        UnitReadoff<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 14);
        UnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 21, 40);
        UnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(7u, 2), "Z/7^2 u32", 22, 40);
        UnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 5), "Z/2^5 u32", 23, 40);
        UnitCyclic<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 24, 40);
        UnitCyclic<engine::u128>(engine::ModCtx<engine::u128>::prime_power(5, 30), "Z/5^30 u128", 25, 12);
        // (C) (S23 G2-G) the lean work buffers of cyclic_decomposition / invariant_factors
        UnitCyclicBuffers<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 51, 96);
        UnitCyclicBuffers<uint32_t>(engine::ModCtx<uint32_t>::prime_power(43u, 2), "Z/43^2 u32", 52, 64);
        UnitCyclicBuffers<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 5), "Z/2^5 u32", 53, 64);
        UnitCyclicBuffers<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1291u, 3), "Z/1291^3 u32", 54, 48);
        UnitCyclicBuffers<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 55, 64);
        UnitCyclicBuffers<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64", 56, 48);
        UnitCyclicBuffers<engine::u128>(engine::ModCtx<engine::u128>::prime_power(5, 30), "Z/5^30 u128", 57, 16);
        UnitCyclicOracle<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 3), "Z/2^3 u32", 61, 280);   // (audit m2)
        UnitCyclicOracle<uint32_t>(engine::ModCtx<uint32_t>::prime_power(3u, 2), "Z/3^2 u32", 62, 280);
        UnitCyclicOracle<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 4), "Z/2^4 u32", 63, 140);
        UnitCyclicOracle<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 2), "Z/5^2 u32", 64, 140);
        UnitCyclicOracle<uint64_t>(engine::ModCtx<uint64_t>::prime_power(3ull, 2), "Z/3^2 u64", 65, 140);
        Check(g_oracle_injection_caught > 0,
              "(N) oracle: catches lean-route fault injections that pass the step-4 check, " +
                  std::to_string(g_oracle_injection_caught) + " over " + std::to_string(g_oracle_injection_cases) +
                  " cases of the five oracle rings");
        UnitFastValuation<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 41);
        UnitFastValuation<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1009u, 2), "Z/1009^2 u32", 42);
        UnitFastValuation<uint32_t>(engine::ModCtx<uint32_t>::prime_power(65521u, 2), "Z/65521^2 u32", 43);
        UnitFastValuation<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 44);
        UnitFastValuation<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64", 45);
        UnitDd<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32 d o d", 31);
        UnitDd<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 5), "Z/2^5 u32 d o d", 32);
        UnitDd<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64 d o d", 33);
        // (N) operator level
        CorruptionArm<uint32_t>(20, 5, 3);
        CorruptionArm<uint64_t>(40, 7, 12);
        InconsistentThetaEqArm<uint32_t>(20, 5, 3);
        InconsistentThetaEqArm<uint64_t>(40, 7, 12);
        CorruptionArm<uint32_t>(10, 101, 2);          // (S23 G2-E) the product read-off beside the T_ell low-rank path
        InconsistentThetaEqArm<uint32_t>(10, 101, 2);
        Route2PerturbationArm<uint32_t>(20, 5, 3);    // (S23 G2-D)
        Route2PerturbationArm<uint32_t>(10, 101, 2);  //   the low-rank route 2
        Route2PerturbationArm<uint64_t>(40, 7, 12);
        // (O) object level
        ObjectArm<uint32_t>(20, 5, 3, FibreRepresentation::SymN, 2);   // (S24 CL-O) + the oracle's negative controls
        ObjectArm<uint32_t>(60, 5, 6, FibreRepresentation::SymN, 0);   // (S24 CL-O) its (R) runs in the Slow name
        ObjectArm<uint32_t>(30, 7, 4, FibreRepresentation::SymN, 0);   // (S24 CL-O) its (R) runs in the Slow name
        ObjectArm<uint32_t>(24, 11, 2);
        ObjectArm<uint32_t>(40, 13, 3, FibreRepresentation::SymN, 0);  // (S24 CL-O) its (R) runs in the Slow name
        ObjectArm<uint32_t>(24, 43, 2);
        ObjectArm<uint32_t>(12, 101, 2);
        ObjectArm<uint32_t>(10, 1009, 2);
        ObjectArm<uint64_t>(40, 7, 12, FibreRepresentation::SymN, 0);  // (S24 CL-O) its (R) runs in the Slow name
        ObjectArm<uint64_t>(8, 5, 3, FibreRepresentation::HidaLatticeH);
        // (S23 G2-C / G2-D / G2-E) ell >= 94 (T_ell low-rank, the grouped GEMM) at K >= 2 in both words; the lean retention
        ObjectArm<uint32_t>(16, 211, 2);
        ObjectArm<uint32_t>(12, 101, 3);
        ObjectArm<uint64_t>(10, 101, 5, FibreRepresentation::SymN, 2);   // 101^5 > 2^32: the u64 word on the low-rank path (+ CL-O controls)
        ObjectArm<uint64_t>(8, 1009, 4);
        ObjectArm<uint32_t>(8, 1291, 3);              // (audit m3) 1291^3 > 2^31, < 2^32: u32 word, Barrett axpy
        LeanArm<uint32_t>(24, 43, 2, false);              // (S24 CL-O) its (R) runs in the Slow name
        LeanArm<uint32_t>(12, 101, 2);
        LeanArm<uint64_t>(40, 7, 12, false);              // (S24 CL-O) its (R) runs in the Slow name
        // (T) (S23 G2-F) the module route's ChainsOnly tau table against the Full one
        TauContentsArm<uint32_t>(20, 5, 3);
        TauContentsArm<uint32_t>(12, 101, 2);         // T_101: the factored (U_ell low-rank) pool on the Full table
        TauContentsArm<uint64_t>(40, 7, 12);
        TauContentsArm<uint64_t>(8, 1009, 4);
        // (V) (S23 G2-T) the module-eval lever against its rollback
        UnitRhoRows<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 71);
        UnitRhoRows<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1009u, 2), "Z/1009^2 u32", 72);
        UnitRhoRows<uint32_t>(engine::ModCtx<uint32_t>::prime_power(10007u, 2), "Z/10007^2 u32", 73);
        UnitRhoRows<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1291u, 3), "Z/1291^3 u32 (p > 2^31, unfused)", 74);
        // (G2-T audit F2) above floor(2^32 / sqrt 3) ~ 2.48e9, where 3 (p - 1)^2 overflows the u64 accumulator: a
        // fused step here (a wrong u32 gate) gives wrong residues, so these two catch the gate; Z/1291^3 cannot.
        UnitRhoRows<uint32_t>(engine::ModCtx<uint32_t>::prime_power(3u, 20), "Z/3^20 u32 (p > 2.48e9, unfused)", 78);
        UnitRhoRows<uint32_t>(engine::ModCtx<uint32_t>::prime_power(65521u, 2), "Z/65521^2 u32 (p > 2.48e9, unfused)", 79);
        UnitRhoRows<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 75);
        UnitRhoRows<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64", 76);
        UnitRhoRows<uint64_t>(engine::ModCtx<uint64_t>::prime_power(3ull, 40), "Z/3^40 u64 (p > 2^62, unfused)", 77);
        UnitFactorScan<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 81);
        UnitFactorScan<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1009u, 2), "Z/1009^2 u32", 82);
        UnitFactorScan<uint32_t>(engine::ModCtx<uint32_t>::prime_power(10007u, 2), "Z/10007^2 u32", 83);
        UnitFactorScan<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1291u, 3), "Z/1291^3 u32", 84);
        UnitFactorScan<uint32_t>(engine::ModCtx<uint32_t>::prime_power(3u, 19), "Z/3^19 u32", 85);
        UnitFactorScan<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 86);
        UnitFactorScan<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64", 87);
        EvalObjectArm<uint32_t>(20, 5, 3);
        EvalObjectArm<uint32_t>(12, 101, 2);
        EvalObjectArm<uint32_t>(8, 1291, 3);
        EvalObjectArm<uint64_t>(40, 7, 12);
        EvalObjectArm<uint64_t>(10, 101, 5);
        EvalObjectArm<uint64_t>(8, 5, 3, FibreRepresentation::HidaLatticeH);
        // (V) (d)-(f) (S23 G2-T2) every available rho fill path against the scalar one
        std::cout << "  RHO_FILL_PATHS available=";
        for (RhoFillPath pth : AvailableRhoPaths()) std::cout << to_text(pth) << (pth == AvailableRhoPaths().back() ? "" : ",");
        std::cout << " auto=" << to_text(rho_fill_path_resolved()) << "\n";
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 171);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1009u, 2), "Z/1009^2 u32", 172);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(10007u, 2), "Z/10007^2 u32", 173);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(23159u, 2), "Z/23159^2 u32", 174);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(3u, 19), "Z/3^19 u32", 175);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(46337u, 2), "Z/46337^2 u32 (p just below 2^31)", 176);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 31), "Z/2^31 u32 (p = 2^31, the SIMD gate's edge)", 177);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 2), "Z/2^2 u32", 178);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1291u, 3), "Z/1291^3 u32 (p > 2^31: scalar, unfused)", 179);
        UnitRhoPaths<uint32_t>(engine::ModCtx<uint32_t>::prime_power(65521u, 2), "Z/65521^2 u32 (p > 2.48e9: scalar, unfused)", 180);
        UnitRhoPaths<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64 (fused)", 181);
        UnitRhoPaths<uint64_t>(engine::ModCtx<uint64_t>::prime_power(3ull, 40), "Z/3^40 u64 (p > 2^62, unfused)", 182);
        PathObjectArm<uint32_t>(20, 5, 3);
        PathObjectArm<uint32_t>(12, 101, 2);
        PathObjectArm<uint32_t>(70, 101, 2);          // n + 1 > 64: full blocks of both SIMD widths and a tail
        PathObjectArm<uint32_t>(30, 7, 11);           // 7^11 = 1977326743 < 2^31: the SIMD path near the gate
        PathObjectArm<uint32_t>(8, 1291, 3);          // p > 2^31: the scalar path whatever is forced
        PathObjectArm<uint64_t>(10, 101, 5);          // u64: the scalar path whatever is forced
        PathCellArm(16, 101, 6);
        PathCellArm(70, 1009, 3);
        // (V) (g) (S23 G2-T2) the route-2 memo lever against its rollback
        MemoObjectArm<uint32_t>(20, 5, 3);            // dense route-2 terms
        MemoObjectArm<uint32_t>(16, 211, 2);          // T_211 low-rank route-2 terms
        MemoObjectArm<uint64_t>(40, 7, 12);
        MemoObjectArm<uint64_t>(10, 101, 5);
        MemoCellArm(16, 101, 6);
        EvalCellArm(16, 101, 6);
        EvalCellArm(10, 1009, 6);
        // (P) (S23 G2-P) the parallel assembly at consumer-pool sizes 1 / 2 / 7 / 16
        ParallelObjectArm<uint32_t>(20, 5, 3);
        ParallelObjectArm<uint32_t>(12, 101, 2);
        ParallelObjectArm<uint32_t>(60, 5, 6, FibreRepresentation::SymN, true);   // u32 Howell forms and read-offs large enough for the hook
        ParallelObjectArm<uint32_t>(8, 1291, 3);
        ParallelObjectArm<uint64_t>(40, 7, 12, FibreRepresentation::SymN, true);
        ParallelObjectArm<uint64_t>(10, 101, 5);
        ParallelObjectArm<uint64_t>(8, 5, 3, FibreRepresentation::HidaLatticeH);
        ParallelCellArm(16, 101, 6);
        ParallelCellArm(10, 1009, 6);
        ParallelCellArm(40, 211, 4);
        // (P) (N) (G2-P audit F1) the parallel read-off's negative paths: unit level, then operator level
        ParallelReadoffFaultUnit<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 6), "Z/5^6 u32", 71);
        ParallelReadoffFaultUnit<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 72);
        ParallelFaultArm<uint32_t>(60, 5, 6);
        ParallelFaultArm<uint64_t>(40, 7, 12);
        // (B) ell in {2, 3}
        BredonArm(10, 2, 3);
        BredonArm(10, 3, 2);
        // (S) cell level
        CellArm(60, 5, 6, " (K_cls = 3)");
        CellArm(108, 7, 6);
        CellArm(50, 13, 3);
        CellArm(24, 43, 6);
        CellArm(16, 101, 6);
        CellArm(10, 1009, 6);
        // (H) (S24 G2-H) the lean-peak lever against its rollback
        UnitResidual<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 261);
        UnitResidual<uint32_t>(engine::ModCtx<uint32_t>::prime_power(43u, 2), "Z/43^2 u32", 262);
        UnitResidual<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1291u, 3), "Z/1291^3 u32", 263);
        UnitResidual<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 264);
        UnitHowellAdopt<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 3), "Z/5^3 u32", 271);
        UnitHowellAdopt<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 5), "Z/2^5 u32", 272);
        UnitHowellAdopt<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64", 273);
        UnitHowellAdopt<engine::u128>(engine::ModCtx<engine::u128>::prime_power(5, 30), "Z/5^30 u128", 274);
        LeanPeakObjectArm<uint32_t>(24, 43, 2);
        LeanPeakObjectArm<uint32_t>(60, 5, 6);
        LeanPeakObjectArm<uint32_t>(12, 101, 2);       // T_101 low-rank: route 2 in place by grouped GEMMs
        LeanPeakObjectArm<uint64_t>(40, 7, 12);
        LeanPeakObjectArm<uint64_t>(10, 101, 5);
        LeanPeakNegativeArm<uint32_t>(20, 5, 3);
        LeanPeakNegativeArm<uint32_t>(10, 101, 2);
        LeanPeakNegativeArm<uint64_t>(40, 7, 12);
        LeanPeakCellArm(60, 5, 6);
        LeanPeakCellArm(16, 101, 6);
        LeanPeakCellArm(120, 1009, 3);
    } catch (const std::exception& e) {
        Check(false, std::string("an arm threw: ") + e.what());
    }
    std::cout << "  EFFECTIVE HECKE_MODULE_STREAM " << el::module_stream_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_EVAL " << el::module_eval_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_PARALLEL " << el::module_parallel_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_LEAN_PEAK " << el::module_lean_peak_effective_text() << "\n";
    const double s = Now() - t0;
    std::cout << "HeckeModuleStreamValidation: " << g_checks << " checks, " << g_fail << " failing, " << Fx(s, 1) << " s, peak_rss_mib="
              << Fx(oc_util::PeakRssMiB(), 1) << "\n";
    return g_fail ? 1u : 0u;
}

uint32_t HeckeModuleStreamSlowValidation() {
    g_fail = g_checks = 0;
    const double t0 = Now();
    std::cout << "=== HeckeModuleStreamSlowValidation (slot-23 G2-A..E at (300, 1009) and (500, 43), sat route; + the (6, 10007, 2) arms and (160, 11)) ===\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_STREAM " << el::module_stream_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_EVAL " << el::module_eval_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_PARALLEL " << el::module_parallel_effective_text() << "\n";
    try {
        // (S23 G2-C / G2-D / G2-E) object level at ell >= 94 with a larger d: the u64 word on the low-rank path, the lean retention
        // (S24 VAL-IMPL, VALSTREAM_audit_1030.md row 1) moved here from HeckeModuleStreamValidation, unchanged: every arm at
        // (6, 10007, 2) (the kGroupCap early flush: (O), (T), (V) eval / memo, (P), (H)) and the (S) cell (160, 11) KMAX 6
        const uint32_t moved0 = g_checks;
        ObjectArm<uint32_t>(6, 10007, 2, FibreRepresentation::SymN, 0);   // (S24 CL-O: its (R) below) chains of > 2048 terms in one block: the group cap's early flush
        TauContentsArm<uint32_t>(6, 10007, 2);        // 10008 cosets of T_10007
        EvalObjectArm<uint32_t>(6, 10007, 2);
        MemoObjectArm<uint32_t>(6, 10007, 2);
        ParallelObjectArm<uint32_t>(6, 10007, 2);     // the kGroupCap early flush inside the parallel row tasks
        LeanPeakObjectArm<uint32_t>(6, 10007, 2);     // the kGroupCap early flush, in place
        CellArm(160, 11, 6);
        std::cout << "  MODULE_STREAM_SLOW moved_from_standard checks=" << (g_checks - moved0) << "\n";
        ObjectArm<uint64_t>(60, 1009, 4);
        LeanArm<uint32_t>(100, 1009, 2);
        // (S24 CL-O) the independent oracle (R) at the standard (O) / (L) cells whose oracle costs about 1 s or more
        const uint32_t oracle0 = g_checks;
        OracleObjectArm<uint32_t>(60, 5, 6);
        OracleObjectArm<uint32_t>(6, 10007, 2);       // the moved (O) cell (kept out of moved_from_standard above)
        OracleObjectArm<uint32_t>(40, 13, 3);
        OracleObjectArm<uint32_t>(30, 7, 4);
        OracleObjectArm<uint32_t>(24, 43, 2, /*lean=*/true);
        OracleObjectArm<uint64_t>(40, 7, 12, /*lean=*/true);
        std::cout << "  MODULE_STREAM_SLOW oracle_for_standard_cells checks=" << (g_checks - oracle0) << "\n";
        TauRssArm(300, 10007);                        // (S23 G2-F) the tau tables of T_10007 alone, ChainsOnly vs Full
        EvalCellArm(100, 10007, 4);                   // (S23 G2-T) the module-eval lever at a T_10007 sat cell
        ParallelCellArm(100, 10007, 4, /*slow=*/true); // (S23 G2-P) pool 1 vs pool 8 at a T_10007 sat cell
        CellArm(300, 1009, 6);
        CellArm(500, 43, 6);
    } catch (const std::exception& e) {
        Check(false, std::string("an arm threw: ") + e.what());
    }
    std::cout << "  EFFECTIVE HECKE_MODULE_STREAM " << el::module_stream_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_EVAL " << el::module_eval_effective_text() << "\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_PARALLEL " << el::module_parallel_effective_text() << "\n";
    const double s = Now() - t0;
    std::cout << "HeckeModuleStreamSlowValidation: " << g_checks << " checks, " << g_fail << " failing, " << Fx(s, 1) << " s\n";
    return g_fail ? 1u : 0u;
}

// (S24 MT-1) HECKE_MODULE_MT: the module route's splits and the pivot-search exit, at consumer-pool sizes 1 / 2 / 7 / 16,
// against the pre-MT route (the lever off on one worker).  Registered by Registry/Slot24/MT1.cpp.
uint32_t HeckeModuleMtValidation() {
    g_fail = g_checks = 0;
    const double t0 = Now();
    std::cout << "=== HeckeModuleMtValidation (slot-24 MT-1: HECKE_MODULE_MT at pools 1 / 2 / 7 / 16 vs the pre-MT route) ===\n";
    std::cout << "  EFFECTIVE HECKE_MODULE_MT " << el::module_mt_effective_text() << "\n";
    try {
        // (U) unit level: cyclic_decomposition's (c2)-(c3) (tiled pivot search, row splits), the read-off, the rhs solves,
        // gemm_rows_hook; u32 and u64 words
        MtUnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(5u, 6), "Z/5^6 u32", 241, 9);
        MtUnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(1009u, 3), "Z/1009^3 u32", 242, 6);
        MtUnitCyclic<uint64_t>(engine::ModCtx<uint64_t>::prime_power(7ull, 12), "Z/7^12 u64", 243, 6);
        // (MT-1 audit) tie-heavy rings: every non-unit entry of Z/4 has valuation 1, Z/3^4 few valuation classes
        MtUnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(2u, 2), "Z/4 u32", 246, 6);
        MtUnitCyclic<uint32_t>(engine::ModCtx<uint32_t>::prime_power(3u, 4), "Z/3^4 u32", 247, 6);
        MtUnitSolves<uint32_t>(engine::ModCtx<uint32_t>::prime_power(101u, 4), "Z/101^4 u32", 244);
        MtUnitSolves<uint64_t>(engine::ModCtx<uint64_t>::prime_power(10007ull, 4), "Z/10007^4 u64", 245);
        // (O) object level: every (O) object and the stream counters
        MtObjectArm<uint32_t>(60, 5, 6, FibreRepresentation::SymN, true);
        MtObjectArm<uint32_t>(12, 101, 2);
        MtObjectArm<uint32_t>(6, 10007, 2);
        MtObjectArm<uint32_t>(8, 1291, 3);
        MtObjectArm<uint64_t>(40, 7, 12, FibreRepresentation::SymN, true);
        MtObjectArm<uint64_t>(8, 5, 3, FibreRepresentation::HidaLatticeH);
        MtObjectArm<uint32_t>(120, 1009, 2, FibreRepresentation::SymN, true, true);
        // (C) cell level: the SAT_DEFECT lines
        MtCellArm(16, 101, 6);
        MtCellArm(40, 211, 4);
        MtCellArm(120, 1009, 3);
    } catch (const std::exception& e) {
        Check(false, std::string("an arm threw: ") + e.what());
    }
    std::cout << "  EFFECTIVE HECKE_MODULE_MT " << el::module_mt_effective_text() << "\n";
    const double s = Now() - t0;
    std::cout << "HeckeModuleMtValidation: " << g_checks << " checks, " << g_fail << " failing, " << Fx(s, 1) << " s, peak_rss_mib="
              << Fx(oc_util::PeakRssMiB(), 1) << "\n";
    return g_fail ? 1u : 0u;
}

// (S24 SATPERF) HECKE_SAT_PERF: the sat cell's per-level memo of the whole-module derivations and the End(N) product's
// uint64_t / row-split route, at pools 1 / 2 / 4, against the rollback on one worker.  Registered by
// Registry/Slot24/SATPERF.cpp (docs/implementation/slot24/SATPERF_report.md).
uint32_t HeckeSatPerfValidation() {
    g_fail = g_checks = 0;
    const double t0 = Now();
    std::cout << "=== HeckeSatPerfValidation (slot-24 SATPERF: HECKE_SAT_PERF on at pools 1 / 2 / 4 vs the rollback) ===\n";
    std::cout << "  " << hecke_ext::ringalg::SatPerfEffectiveLine(false) << "\n";
    try {
        SatPerfCellArm(142, 7, 5, /*expect_split=*/true, /*expect_wide=*/false);   // r = 25; WP-5's hidden datum, K_stab >= 3
        SatPerfCellArm(168, 13, 2, true, false);                                    // r = 29; the hidden-chain onset at ell = 13
        SatPerfCellArm(60, 5, 6, false, false);                                     // K_cls = 3: a torsion block, the certificate at K != K_cls
        SatPerfCellArm(40, 211, 4, false, false);                                   // a u32 module at ell^K ~ 2^31
        SatPerfPlantedArm(1009, 6, 30, 2411, true);                                      // ell^K ~ 2^60: u128 and uint64_t rows, split
        SatPerfPlantedArm(7, 3, 40, 2412, false);
    } catch (const std::exception& e) {
        Check(false, std::string("an arm threw: ") + e.what());
    }
    std::cout << "  " << hecke_ext::ringalg::SatPerfEffectiveLine() << "\n";
    const double s = Now() - t0;
    std::cout << "HeckeSatPerfValidation: " << g_checks << " checks, " << g_fail << " failing, " << Fx(s, 1) << " s\n";
    return g_fail ? 1u : 0u;
}
