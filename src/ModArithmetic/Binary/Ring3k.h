// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_BINARY_RING3K_H
#define ENGINE_BINARY_RING3K_H

// =====================================================================
// Ring3k.h -- Smith / Howell / kernel_module over Z/3^k, k <= 19, on word-sized entries
// (hecke-extensions slot 23, package Z3K; GPU_scout.md target 3, CPU part at ell = 3; ENGINE_API.md Sec. 8.4)
// =====================================================================
//
// The ternary analogue of BIN-2's engine::ring2k ladder (Ring2k.h), for the one word rung the orbifold census needs.
// Two variants per entry point (picked by k below).  THE DENSE BLOCKED VARIANT (smith_levels, howell_blocked):
// entries are uint32_t, CANONICAL in [0, 3^k), k <= 19 (3^19 < 2^31, the Shoup / Barrett bound of ModCtx<uint32_t>).
// Unlike Z/2^k there is no wraparound shortcut: every product reduces, through the engine's own
// modulus-only layer -- ModCtx<uint32_t>::prime_power(3, k) (or ModCtx(3) at k = 1) and kernels::gemm, which is
// exact on a composite modulus (ENGINE_API.md Sec. 13.2) and routes Z/9, Z/27, Z/81 to the VNNI microkernel (Sec. 6.1).
// The pivot-unit inverses go through ModCtx::inv (the unit inverse, canonical residue out; S24 CL-G3 retired Mod3k<1>).
//
// THE SMALL-MODULUS VARIANT (k <= 5; smith_lengths and the small Howell pass): uint16_t entries, the Sec. 13.5 pass
// on the rows themselves in the engine's own order (a row whose pivot-column entry is zero is skipped, which is what
// makes the engine fast on the sparse census coboundaries), each row update one loop modulo a compile-time 3^k that
// GCC vectorises (target_clones skylake-avx512 / haswell / default); Smith by lengths: L_j = length(rowspan(A mod 3^j)),
// L_j - L_(j-1) = #{sigma < j}, L_1 = the F_3 rank by TritMatrix.  Z3K measured the dense variant SLOWER than the
// engine on the census's sparse Z/9 matrices (Z3K_report.md Sec. 2), hence the split.
//
// What is computed (every output is a unique function of the input, ENGINE_API.md Sec. 13.5 / 13.6):
//   * smith(A, k)         the Smith valuations sigma_1 <= ... (each < k; the non-zero diagonal 3^sigma of the Smith
//                         form over Z/3^k), so colspan / rowspan(A) = sum Z/3^(k - sigma).  elim::invariant_factors
//                         is {k - sigma} descending.  Algorithm: valuation LEVELS (Ring2k.h's, with F_3 for F_2).  At
//                         level s the F_3 column rank profile of A mod 3 (TER-1's TritMatrix echelon, rank r_s) is
//                         moved to the front; a blocked right-looking LU whose pivot is the first UNIT (non-zero mod 3)
//                         entry of each of those r_s columns -- one always exists, and no column is ever skipped (the
//                         BIN-2 audit-F4 rule) -- gives r_s invariants sigma = s; the Schur complement on the remaining
//                         rows and columns is 0 mod 3, is divided by 3 exactly and is level s + 1 over Z/3^(k-s-1).
//   * howell(A, k)        the Howell form, equal entrywise to engine::elim::howell: a blocked MINIMAL-VALUATION LU
//                         echelon (rowspan unchanged), the Sec. 13.5 pass with its closure rows on the <= n echelon
//                         rows, then a blocked reduce-above (floor quotients per block, one GEMM per block).
//   * kernel_module(A, k) the Howell form of {x : A x = 0} by Sec. 13.6's rule (the rows of howell([A^T | I_n])
//                         leading at >= m), equal entrywise to engine::elim::kernel_module.
// References (unblocked, validation use): smith_naive (global minimal-valuation pivoting), howell_naive (the
// Sec. 13.5 pass on A itself).
//
// Single-threaded in the LU / pass, except inside smithpar::ScopedEnable (slot-24 PAR-2K, SmithPar.h; the integral
// descent's Smith calls only): the k > 5 blocked LU then splits its trailing GEMM (tiles of C, one kernels::gemm each), its
// panel TRSM (column tiles) and its panel (left-looking: only the rows the pivot search reads are brought up to date
// serially) over threads, and smith_levels its per-level passes; bit-identical at every thread count.  Params (panel, reduce_block) and counters are process-wide relaxed atomics; no value depends on them.
// Gate: OrbifoldMod3kValidation (Z3K): every variant vs the references vs engine::elim on random, planted and real inputs.

#include <cstdint>
#include <vector>

#include "../ModArithmetic/DenseBuf.h"

namespace engine {
namespace ring3k {

// The largest k this rung carries: 3^19 = 1162261467 < 2^31.
constexpr uint32_t kMaxK = 19;

struct SmithResult {
    uint32_t k = 0;
    std::vector<uint32_t> sigma;         // ascending; each < k
    std::vector<uint32_t> level_ranks;   // level_ranks[s] = #{sigma == s} (smith() only)
    uint32_t length() const { uint32_t s = 0; for (uint32_t x : sigma) s += k - x; return s; }
};

struct HowellForm {
    uint32_t k = 0;
    DenseBuf<uint32_t> H;                // h x n row-major, canonical in [0, 3^k)
    std::vector<uint32_t> pivots;        // strictly increasing leading columns
    std::vector<uint32_t> pivot_v;       // row i leads with exactly 3^pivot_v[i], 0 <= v < k
    uint32_t length() const { uint32_t s = 0; for (uint32_t v : pivot_v) s += k - v; return s; }
};

// Tunables (validation / tuning; values never affect a result).
struct Params {
    uint32_t panel = 64;          // LU panel width (columns)
    uint32_t reduce_block = 64;   // reduce-above block (pivots)
};
Params params();
void   set_params(const Params& p);

// Instrumentation (diagnostic only).
struct Counters { uint64_t lu_calls = 0, lu_pivots = 0, closure_rows = 0, smith_levels = 0, gemm_calls = 0; };
Counters counters();
void     reset_counters();

// Every entry point takes A by value (pass an rvalue to avoid a copy); A must be canonical mod 3^k (ENGINE_CHECKed
// entrywise at entry: O(m n), cheap against the elimination), 1 <= k <= kMaxK.  Any layout / ld is accepted; the
// working copy is compact row-major.
//
// The production entry points pick the variant by k (no value depends on the choice; the gate compares all of them):
//   k <= 5 (3^k <= 243, the census's Z/9): the SMALL-MODULUS rows -- uint16_t entries, the Sec. 13.5 pass on the rows
//     themselves in the engine's sparse-friendly order (a row with a zero in the pivot column is skipped), each row
//     update one vectorised loop modulo a compile-time 3^k; smith reads the Smith valuations off the lengths
//     L_j = length(rowspan(A mod 3^j)), j = 1..k (L_1 = the F_3 rank by TritMatrix; L_j - L_(j-1) = #{sigma < j});
//   k > 5 (the census's Z/3^19): smith_levels / howell_blocked (dense, blocked; kernels::gemm).
SmithResult smith(DenseBuf<uint32_t> A, uint32_t k);
HowellForm  howell(DenseBuf<uint32_t> A, uint32_t k);
HowellForm  kernel_module(const DenseBuf<uint32_t>& A, uint32_t k);

// The variants themselves (validation use; smith_lengths needs k <= 5).
SmithResult smith_levels(DenseBuf<uint32_t> A, uint32_t k);
SmithResult smith_lengths(const DenseBuf<uint32_t>& A, uint32_t k);
HowellForm  howell_blocked(DenseBuf<uint32_t> A, uint32_t k);

// References (unblocked; validation use).
SmithResult smith_naive(DenseBuf<uint32_t> A, uint32_t k);
HowellForm  howell_naive(DenseBuf<uint32_t> A, uint32_t k);

} // namespace ring3k
} // namespace engine

#endif
