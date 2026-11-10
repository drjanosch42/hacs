// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// TapeElim.h -- the panel-driven elimination driver on the tape (Phase 4, D4)
// =====================================================================
//
// Block RREF / kernel / image / factor_basis / solve_multi_rhs with the SAME
// results as engine::blk (BlockElim.h), the heavy operations issued as tape
// programs.  Rank is data-dependent, so the elimination is not one static
// program: a host driver walks the block columns and per panel records
//
//   P1: GATHER (candidate rows of the panel and of the pivot columns)
//       -> SCHUR_UPDATE  S -= L_g . R[:, j]        (tiled over threads)
//       -> SCATTER (S -> S_c)  -> PANEL_RREF(S_c) -> PIVOT_SEARCH(S, Q)
//   host: reads Q (pivot columns), prows (pivot rows), the pivot block B;
//         classifies B (I / alpha I / general -> inverse on the host)
//   P2: SCATTER (new rows' panel part) ; GATHER (A[prows, trail], L_p)
//       -> SCHUR_UPDATE  T -= L_p . R[:, trail]    -> TRSM (B^{-1} T, no-op for I)
//       -> GATHER (C = R[:, Q]) -> SCHUR_UPDATE  R[0:rk, c0:] -= C . R_new
//
// Pivot rule: PANEL_RREF's leftmost-pivot / topmost-row rule inside the
// panel and the row rank profile of PIVOT_SEARCH -- both pure functions of
// the panel's contents, so the pivot sequence is schedule-independent.  The
// output is the unique RREF (one row per pivot, ascending pivot columns, no
// zero rows) and every derived basis is the D5b function of it -- identical,
// entrywise, to blk::rref / kernel_basis / image_basis / factor_basis /
// solve_multi_rhs (EngineTapeEliminationValidation).

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../ModArithmetic/BlockElim.h"
#include "../ModArithmetic/BlockMat.h"
#include "Executor.h"

namespace engine {
namespace tape {

struct TapeElimStats {
    uint64_t panels = 0, panels_with_pivots = 0, programs = 0, nodes = 0, tasks = 0;
    uint64_t identity_pivots = 0, scalar_pivots = 0, general_pivots = 0;
    // Wall seconds (diagnostic; ENGINE_TRACE_ELIM=1 makes the bench print the
    // process-wide totals): host bookkeeping + recording, compile (passes),
    // and execution, per program class.
    double s_host = 0.0, s_compile = 0.0, s_run_p1 = 0.0, s_run_p2 = 0.0;
    double s_serial = 0.0;   // of s_run_p1: the host-serial PANEL_RREF / PIVOT_SEARCH nodes (the Amdahl floor)
    // (hecke-extensions slot 9, WP-6) what the per-panel programs P1 / P2 were
    // COMPILED under -- the non-vacuity evidence that a caller's pass options
    // reach the elimination driver (diagnostic, never on a result path):
    // programs compiled with batch = false, programs compiled under a device
    // policy (effective_policy() != Cpu: `hybrid`'s cost model may still keep
    // every node on the CPU), programs whose batching pass formed >= 1 batch,
    // programs with >= 1 GPU-assigned node, and those nodes.
    uint64_t batch_off_programs = 0, device_policy_programs = 0, batched_programs = 0, gpu_programs = 0,
             gpu_nodes = 0;
    uint64_t device_elims = 0;   // (slot-22 G2) calls that ran as one device-resident elimination (cuda::imma_rref)
};
// (slot-22 G2) The device-resident elimination route (ENGINE_API.md Sec. 11.9). Taken iff
//   * ENGINE_GPU_ELIM is not "0" (re-read per call; the rollback arm, retired in OFF builds by
//     S24 CL-R2: honoured only with OC_ENABLE_LEGACY=ON, refused at start-up otherwise),
//   * the context is a uint32_t FIELD with 2 <= p <= 127 (cuda::imma_elim_eligible),
//   * the matrix has >= ENGINE_GPU_ELIM_MIN entries (default 1048576; re-read per call),
//   * the calling thread's elimination options select ENGINE_DEVICE=gpu (elim_pass_options():
//     the ScopedPassOptions / thread-local overrides apply; never under cpu or hybrid),
//   * cuda::available() && cuda::imma_supported()
// -- checked in this order, so the cpu default touches no CUDA. Then tape::rref (and with it
// kernel_basis / image_basis / factor_basis stage 1 / solve_multi_rhs's elimination route) and
// every uint32_t DenseElim RREF (via elim::set_device_rref_hook, installed by this TU) run as one
// cuda::imma_rref. Values: the unique RREF / pivots / row rank profile, so nothing can move. A
// CudaError raised before any output was written falls back to the host route (counted). In an
// ENGINE_VERIFY_DEVICE=ON build (or with ENGINE_GPU_ELIM_VERIFY=1) every device elimination is
// re-run on the host and compared entrywise (rank, pivots, RREF, pivot_rows); a difference throws.
bool gpu_elim_enabled();
uint64_t gpu_elim_min_entries();
bool device_elim_wanted(const ModCtx<uint32_t>& ctx, uint32_t m, uint32_t n);
struct DeviceElimTotals {
    uint64_t tape_rrefs = 0;      // tape::rref calls that ran on the device
    uint64_t dense_rrefs = 0;     // DenseElim in-place RREFs that ran on the device
    uint64_t rank_profiles = 0;   // DenseElim column_rank_profile calls that ran on the device
    uint64_t fallbacks = 0;       // device attempts that fell back to the host after a CudaError
    uint64_t verified = 0;        // device eliminations re-run on the host and compared (ENGINE_VERIFY_DEVICE build, or ENGINE_GPU_ELIM_VERIFY=1)
};
DeviceElimTotals device_elim_totals();
// (audit R1) `EFFECTIVE DEVICE_ELIM calls=<n> device=<n> fallbacks=<n>` (device = tape + dense + profiles, calls = device +
// fallbacks), or "" when no device elimination was attempted. Printed to stderr, once per process, at exit; every fallback
// also prints `[imma_rref] fallback to the host route (...)`. ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR=1 (test-only) simulates the CudaError.
std::string device_elim_summary_line();

// Process-wide accumulation of every tape::rref call's stats (mutex-guarded;
// never on a result path).
TapeElimStats tape_elim_totals();
void reset_tape_elim_totals();

// `rank_hint`: an upper bound on the rank known to the caller (factor_basis:
// exactly a, the identity augmentation; solve_multi_rhs: nb).  R is reserved
// at min(rank_hint, m, n) rows up front so it never reallocates (memory
// profile 05 Sec. 5.2, F2 -> F3); a larger rank still grows it (values are
// unaffected either way).  0 = unknown: geometric growth.
template<class Elt>
blk::BlockRrefResult<Elt> rref(Executor<Elt>& ex, const BlockMat<Elt>& A, TapeElimStats* stats = nullptr,
                               uint32_t rank_hint = 0);

template<class Elt>
BlockMat<Elt> kernel_basis(Executor<Elt>& ex, const BlockMat<Elt>& A, Layout layout = Layout::ColMajor);

template<class Elt>
BlockMat<Elt> image_basis(Executor<Elt>& ex, const BlockMat<Elt>& A, Layout layout = Layout::ColMajor);
// (matrix-engine Phase 4) The D5b image basis -- the pivot columns of A -- of
// a DENSE a x k multi-vector A over the row partition row_dims, eliminated
// over column panels of width `chunk` (the panel RREF is host-serial, so a
// 9d-wide single panel would be one serial node).  Result: a x rank,
// column-major, ONE contiguous window (no structural-zero detection), the
// same values blk::image_basis(A) produces.
template<class Elt>
BlockMat<Elt> image_basis(Executor<Elt>& ex, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                          const std::vector<uint32_t>& row_dims, DenseView<const Elt> A,
                          uint32_t chunk = 256, TapeElimStats* stats = nullptr);

// factor_basis WITHOUT the ambient-wide identity augmentation (memory profile
// 05 Sec. 5.2 F4 / Sec. 7 item 1).  Stage 1 eliminates M = [Rl | G] alone --
// pivots, independent_relations and the basis are those of blk::factor_basis
// because appending columns never changes the pivot columns among the earlier
// ones.  Stage 2 (only with want_projector) rebuilds the projector -- the rows
// of the unique RREF transform U of [M | I] that belong to the G pivots --
// from two facts: (1) after the M columns the transform of the pivot rows is
// M[pivot_rows, pivots]^{-1} scattered into the pivot_rows columns (the rows of
// R are combinations of the pivot rows alone), and (2) the remaining rows of U
// are the unique RREF basis of the left null space of M, which is the RREF of
// Y = I[act, :] - M[act, pivots] . U_top0 (act = the non-pivot rows), and the
// final top rows are U_top0 reduced against Y's pivot columns.  Uniqueness of
// the RREF makes the result entrywise identical to blk::factor_basis
// (EngineTapeEliminationValidation), with the workspace `rank x (nr + ng)`
// plus `(a - rank) x a` twice instead of `a x (nr + ng + a)`.  With
// want_projector = false the projector is left EMPTY (0 x 0) and stage 2 is
// skipped -- the W^{q} = C^q / Z^q calls of compute_cohomology() need only the
// basis.
template<class Elt>
blk::FactorBasisResult<Elt> factor_basis(Executor<Elt>& ex, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                                         const std::vector<uint32_t>& row_dims,
                                         DenseView<const Elt> Rl, DenseView<const Elt> G,
                                         uint32_t chunk = 256, TapeElimStats* stats = nullptr,
                                         bool want_projector = true);

// `certify`: when B has a unit row per column (a D5b kernel basis) the
// coordinates are a row gather and membership is certified by one exact tape
// product COMPAREd against Y (the elimination route's throw on a miss).  A
// caller that certifies the result itself by the same exact residual
// (HeckeOperatorOnK's A-N3) passes false and skips that product; without
// unit rows the elimination route runs either way and certifies by
// construction.
template<class Elt>
DenseBuf<Elt> solve_multi_rhs(Executor<Elt>& ex, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                              const std::vector<uint32_t>& row_dims,
                              DenseView<const Elt> B, DenseView<const Elt> Y,
                              uint32_t chunk = 256, TapeElimStats* stats = nullptr, bool certify = true);

} // namespace tape
} // namespace engine
