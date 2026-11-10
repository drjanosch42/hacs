// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// DenseElim.h -- in-buffer dense elimination over F_p (Phase 2, plan D5/D5b)
// =====================================================================
//
// Blocked, GEMM-bound RREF on a DenseView, plus everything the legacy
// substrate derives from an RREF: pivot columns / rank, kernel basis, image
// basis, the transform U, inverse, multi-RHS solve, complement basis and the
// block-elimination step "B -= L * R".
//
// WHY THE RESULT IS LEGACY-IDENTICAL (the whole point of this file)
// -----------------------------------------------------------------
// Every object produced here is UNIQUELY determined by its input, so a blocked
// algorithm only has to be CORRECT to be bit-identical to the scalar legacy
// code -- reassociation, blocking and the choice of pivot ROWS cannot show
// up in the output:
//
//   * RREF(A) is unique (theorem); its pivot columns are the column rank
//     profile of A (the lexicographically first independent column set),
//     which is what "leftmost pivot column" produces in ANY row order.
//   * U := trailing m columns of RREF([A | I_m]) (D5b) is therefore unique
//     as well -- it is computed literally as that augmented RREF, never as
//     "some U with U*A = RREF(A)".
//   * kernel / image / complement bases and the projector are fixed
//     functions of (RREF, pivots, original columns) per the D5b table.
//   * B - L*R with L = B[:, pivot cols] is plain exact arithmetic.
//
// The pivot rule is nevertheless kept verbatim (leftmost column with a
// non-zero in the Schur complement, topmost such row from the current row
// downward): it is what makes the pivot columns the rank profile, and it
// keeps the row permutation identical to the legacy one, so even the
// intermediate factorisation is the one the scalar code walks through.
//
// ALGORITHM (D5: recursive blocked PLU + TRSM, ported not linked)
// ----------------------------------------------------------------
//   1. Column-panel PLU with column-order pivoting, IN PLACE / packed:
//      top level walks panels of `panel` columns; inside a panel the columns
//      are bisected down to `base` columns, where a left-looking (Crout)
//      scalar factorisation runs with delayed accumulation (one reduction
//      per updated entry instead of one per pivot).  After the left half of
//      a bisection, its k1 pivots update the right half with ONE unit-lower
//      TRSM (pivot rows) and ONE GEMM (Schur complement, GemmOp::Sub); the
//      same two calls with the whole panel's rank update the trailing
//      columns at the top level.  Multipliers live in the pivot columns
//      (packed L); row swaps move whole rows so L travels with its row.
//      The column pivot sequence is exactly the legacy one because the right
//      half / trailing part is always the Schur complement w.r.t. every
//      pivot to its left before its own pivots are searched.
//   2. RREF from the packed factorisation: with U1 = U[:, pivots] (upper
//      triangular, k x k), RREF = U1^{-1} U, i.e. a diagonal scaling (one
//      batch_inv) and a unit-upper TRSM on the k x (n-k) non-pivot block.
//      For a full-column-rank input that block is empty and the RREF is I
//      -- the reason a full-rank elimination is ~3x cheaper than legacy
//      Gauss-Jordan, which reduces above and below at every pivot.
//
// Costs, 1677 x 1419 full rank: ~1e9 modmul in GEMM (30 Gmodmul/s), ~2e7
// in the Crout base, ~1e8 in TRSM -- vs 3.4e9 scalar modmul for legacy.
//
// ELEMENT TYPES: uint32_t and uint64_t (u64 is correct, not fast: the u64
// gemm is a plain loop).  All arithmetic is exact; every accumulator holds
// at most ctx.k_max products (ENGINE_API.md Sec. 5).

#include <cstdint>
#include <functional>
#include <vector>

#include "DenseBuf.h"
#include "DenseView.h"
#include "EngineConfig.h"
#include "EnvLever.h"
#include "ModCtx.h"

namespace engine {
namespace elim {

// Tunables (plan: panel in [128, 512]).  Read once from the environment for
// tuning runs -- ENGINE_ELIM_PANEL, ENGINE_ELIM_BASE, ENGINE_ELIM_TRSM_BASE --
// and otherwise these defaults.  Values never affect results.
struct Params {
    uint32_t panel     = 256;   // top-level column panel width
    uint32_t base      = 1;     // bisection stops here; scalar Crout base below.  (Phase 4 closure:
                                // 32 -> 1 -- every column step is then a vectorised rank-k update
                                // over the pool hook instead of scalar Crout dots; hecke 256's
                                // base-panel time 0.52 s -> 0.025 s summed over its 81 PLU calls)
    uint32_t trsm_base = 32;    // recursive TRSM leaf size
};
const Params& params();
// Validation hook: override the tunables for the rest of the process (results
// must not change -- DenseElimValidation proves it by sweeping them).
void set_params(const Params& p);

struct RrefResult {
    uint32_t rank = 0;
    std::vector<uint32_t> pivots;   // pivot columns, ascending (== column rank profile)
};

// Diagnostic phase counters of the dense PLU driver (wall seconds, summed
// over every call in the process; accumulated only under ENGINE_TRACE_NODES,
// never read by a computation).
struct PluPhaseStats {
    double base_panel = 0.0, trsm = 0.0, trailing_gemm = 0.0, to_rref = 0.0, total = 0.0;
    double base_rows = 0.0, base_rows_par = 0.0;   // of base_panel: the row passes, and those split over the hook
    uint64_t calls = 0;
};
PluPhaseStats plu_phase_totals();
void reset_plu_phase_totals();

// (slot-22 G2) The device elimination hook (ENGINE_API.md Sec. 11.9). ModArithmetic links no
// CUDA, so the MatrixEngine library installs it once at static initialisation (TapeElim.cpp).
// Every uint32_t in-place RREF of this file (rref, rref_with_transform, inverse, solve_multi_rhs,
// complement_basis -- they all go through one internal routine) and column_rank_profile ask it
// first; it returns false (and touches nothing) unless the calling thread's device policy is
// ENGINE_DEVICE=gpu, ENGINE_GPU_ELIM is not 0 (a retired lever since S24 CL-R2: =0 only in an
// OC_ENABLE_LEGACY=ON build), the context is a field with p <= 127, the matrix is
// large enough and an IMMA-capable device exists -- so the cpu default never reaches CUDA. When it
// returns true, A holds RREF(A) (want_rref; zero rows at the bottom) or is untouched (rank profile
// only) and res holds rank and pivots: the same unique objects, so no value can change.
using DeviceRrefHook = bool (*)(const ModCtx<uint32_t>& ctx, DenseView<uint32_t> A, bool want_rref, RrefResult& res);
void           set_device_rref_hook(DeviceRrefHook h);
DeviceRrefHook device_rref_hook();

// (slot-22 BIN-1) The native F_2 route (ENGINE_API.md Sec. 8.1). OPT-IN, default OFF: with
// ENGINE_BINARY_F2=1 (read once per process; anything else = off) or a ScopedBinaryF2(true) in
// force, every in-place RREF of this file (the same internal routine the device hook guards) and
// column_rank_profile on a FIELD context with p == 2 (uint32_t or uint64_t) packs A into a
// binary::BitMatrix, eliminates by M4RI and unpacks. RREF, rank and pivots are unique functions of
// A, so no value can change (BinaryF2Validation). The device hook is asked first; at p = 2 it only
// fires under ENGINE_DEVICE=gpu.
bool binary_f2_enabled();
const lever::EnvLever& binary_f2_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedBinaryF2 : lever::EnvLever::Scoped {          // process-wide override (validation use; not nestable across threads)
    explicit ScopedBinaryF2(bool on) : Scoped(binary_f2_lever(), on) {}
};
struct BinaryF2Totals { uint64_t rrefs = 0, rank_profiles = 0; };   // calls routed so far (diagnostic)
BinaryF2Totals binary_f2_totals();

// (slot-23 TER-1) The native F_3 route (ENGINE_API.md Sec. 8.3), the twin of the F_2 route above.
// OPT-IN, default OFF: with ENGINE_TERNARY_F3=1 (read once per process; anything else = off) or a
// ScopedTernaryF3(true) in force, the same in-place RREF routine and column_rank_profile on a FIELD
// context with p == 3 (uint32_t or uint64_t) pack A into a bit-sliced binary::TritMatrix, eliminate
// (Gauss--Jordan) and unpack. RREF, rank and pivots are unique functions of A, so no value can
// change (TernaryF3Validation). The device hook and the F_2 route are asked first (neither fires at p = 3
// on the cpu default).
bool ternary_f3_enabled();
const lever::EnvLever& ternary_f3_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedTernaryF3 : lever::EnvLever::Scoped {         // process-wide override (validation use; not nestable across threads)
    explicit ScopedTernaryF3(bool on) : Scoped(ternary_f3_lever(), on) {}
};
struct TernaryF3Totals { uint64_t rrefs = 0, rank_profiles = 0; };   // calls routed so far (diagnostic)
TernaryF3Totals ternary_f3_totals();

// In-place RREF of A (any layout; a row-contiguous view is the fast path,
// anything else is relaid out once).  D5b conventions: leading 1, zero rows
// at the bottom.
template<class Elt>
RrefResult rref(const ModCtx<Elt>& ctx, DenseView<Elt> A);

// The column rank profile of A -- the pivot columns of its RREF, ascending --
// from the packed PLU alone (no back-substitution): the pivots are fixed by
// factor(), so they are exactly rref()'s.  A is DESTROYED (left in packed
// PLU form); pass a scratch copy.  A must be row-contiguous.
template<class Elt>
RrefResult column_rank_profile(const ModCtx<Elt>& ctx, DenseView<Elt> A);

// R = RREF(A) (m x n) and U = trailing m columns of RREF([A | I_m]) (m x m),
// eliminated over all n + m columns (D5b: unique).  Returns the pivots of A.
template<class Elt>
RrefResult rref_with_transform(const ModCtx<Elt>& ctx, DenseView<const Elt> A,
                               DenseView<Elt> R, DenseView<Elt> U);

// Kernel basis from an RREF R (k x n, k >= rank rows) and its pivots: one ROW
// per free column in ascending order; 1 at the free column, -R[i][j_free] at
// pivot i, 0 elsewhere.  Shape (n - rank) x n; empty when full column rank.
template<class Elt>
DenseBuf<Elt> kernel_basis(const ModCtx<Elt>& ctx, DenseView<const Elt> R,
                           const std::vector<uint32_t>& pivots);

// Image basis: the ORIGINAL columns of A at the pivot indices, ascending, one
// per ROW of the result (rank x m).  For A = alpha*I this yields alpha*e_j
// (Phase-0 owner decision Q11).
template<class Elt>
DenseBuf<Elt> image_basis(DenseView<const Elt> A, const std::vector<uint32_t>& pivots);

// Ainv = A^{-1} via RREF([A | I]).  Returns false (Ainv untouched) if A is
// singular or not square.
template<class Elt>
bool inverse(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<Elt> Ainv);

// Coordinates of every column of Y in the columns of B: result[t][j] is the
// coefficient of column j of B in column t of Y, read off RREF([B | Y]).
// Throws std::runtime_error (legacy message) when a column of Y is not in the
// span -- i.e. when it becomes a pivot column.
template<class Elt>
std::vector<std::vector<Elt>> solve_multi_rhs(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                              DenseView<const Elt> Y);

// factor_basis_block's rule (D5b): greedy leftmost-independent subset of the
// generator columns G modulo the relation columns Rl, read off the pivots of
// RREF([Rl | G | I]); returns the indices into G.  If `projector` is given it
// receives the identity-block rows of that RREF belonging to the generator
// pivots (dim_H x ambient) -- the projector legacy returns.
template<class Elt>
std::vector<uint32_t> complement_basis(const ModCtx<Elt>& ctx, DenseView<const Elt> Rl,
                                       DenseView<const Elt> G, DenseBuf<Elt>* projector);

// The block-elimination step of BlockFlatMatrix (compute_elimination_factor_
// and_apply): given B (m x n) and an RREF R (kR x n) with `pivots`, form
// L = B[:, pivots] padded to m x l_cols with zero columns, and replace B by
// B - L * R in place.  Returns L.
template<class Elt>
DenseBuf<Elt> eliminate_against(const ModCtx<Elt>& ctx, DenseView<Elt> B, DenseView<const Elt> R,
                                const std::vector<uint32_t>& pivots, uint32_t l_cols);

// C -= L * R (all Canonical; C must not overlap L or R).
template<class Elt>
void sub_product(const ModCtx<Elt>& ctx, DenseView<Elt> C, DenseView<const Elt> L,
                 DenseView<const Elt> R);

// ===========================================================================
// Z/ell^k: the Howell form and its derived entry points (ENGINE_API.md
// Sec. 13.5 / 13.6; hecke-extensions slot 8 WP-1, E8 item 2)
// ===========================================================================
//
// Each entry point below takes a COMPOSITE context (ModCtx::prime_power,
// Sec. 13.1) and ENGINE_CHECKs !ctx.is_field(): the ring is selected by the
// context alone (Sec. 13.8), and the field route never reaches this code.
// Every result is a unique function of its input, with its rule stated at the
// definition; none is the D5b basis of Sec. 8, and none claims to reduce
// entrywise to an F_ell object (Sec. 13.10 F3).  Indexing: columns and rows
// are 0-based throughout (Sec. 13.5's definition is written 1-based; review m3).

// The Howell form of rowspan(A) (Sec. 13.5): the non-zero rows H (h x n,
// row-major), their leading columns (strictly increasing) and valuations
// (row i leads with exactly ell^{pivot_v[i]}, 0 <= v < k).  h <= n and h may
// exceed rows(A): over Z/4, howell([[2, 1]]) = [[2, 1], [0, 2]].
template<class Elt> struct HowellResult {
    DenseBuf<Elt>          H;
    std::vector<uint32_t>  pivots;
    std::vector<uint32_t>  pivot_v;
    uint32_t rows() const { return static_cast<uint32_t>(pivots.size()); }
};

// Valuation pivoting (Sec. 13.5): columns left to right; among the active
// rows the entry of MINIMAL valuation, ties to the TOPMOST; scale to ell^v;
// eliminate the other active rows exactly (divide_exact); Howell closure
// ell^{k-v} * pivot row appended at the BOTTOM of the active list; then
// reduce above, pivot columns left to right.  The output is unique (Sec. 13.5
// "Uniqueness"): a function of rowspan(A) alone.
template<class Elt>
HowellResult<Elt> howell(const ModCtx<Elt>& ctx, DenseView<const Elt> A);

// The Howell form of the RIGHT kernel K = {x in R^n : A x = 0} (Sec. 13.6):
// the rows of howell([A^T | I_n]) whose leading column is >= rows(A),
// restricted to their last n columns.
template<class Elt>
HowellResult<Elt> kernel_module(const ModCtx<Elt>& ctx, DenseView<const Elt> A);

// The Howell form of the COLUMN module {A x} of R^m, i.e. howell(A^T)
// (Sec. 13.6).  Each row is one generator of length rows(A).  The rows are
// neither a subset of A's columns nor a minimal generating set (review m12).
template<class Elt>
HowellResult<Elt> image_module(const ModCtx<Elt>& ctx, DenseView<const Elt> A);

// For each column y_t of Y (m x ny), the coordinates x_t in R^{nb} with
// B x_t = y_t, REDUCED MODULO kernel_module(ctx, B): left to right over the
// kernel's pivot columns j, x_t[j] in [0, ell^{v_j}) (Sec. 13.6; unique).
// result[t][j] is the coefficient of column j of B (the Sec. 8 layout).  If
// some y_t is not in im B, throws std::runtime_error("Vector is not in image
// of basis (right-hand side t).") for the SMALLEST such t (review m8).
template<class Elt>
std::vector<std::vector<Elt>> solve_multi_rhs_module(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                                     DenseView<const Elt> Y);

// (slot-23 G2-A; G2_plan.md Sec. 2.1) THE UNIT-PIVOT READ-OFF -- a caller-side
// helper beside solve_multi_rhs_module, which is untouched.  When B (m x nb) has
// rows `piv` (nb strictly increasing row indices) with B[piv, :] = I_nb -- the
// case of a basis whose columns are Howell rows with unit pivots, reduced above
// -- B is injective and the unique solution of B X = Y is X = Y[piv, :]; with
// ker B = 0 that is exactly solve_multi_rhs_module's reduced representative,
// entry for entry.  The helper checks B[piv, :] = I (O(nb^2)), reads X (nb x ny,
// row-major) off Y, and recomputes the residual Y - B X EXACTLY over ALL m rows
// (kernels::gemm in row blocks of `row_block`, so the m x ny product is never
// materialised).  Returns:
//   Ok          -- X written, the residual is identically zero;
//   NotIdentity -- B[piv, :] != I (or piv malformed): X untouched; the caller
//                  takes the Howell solve;
//   Residual    -- some column of Y is not in im B: X holds the read-off, the
//                  caller takes the Howell solve, which throws its own named
//                  message (so the error text is the legacy one, byte for byte).
// Composite or field context alike (the arithmetic is modulus-only).
enum class ReadoffStatus { Ok, NotIdentity, Residual };
template<class Elt>
ReadoffStatus solve_unit_pivot_readoff(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                       const std::vector<uint32_t>& piv, DenseView<const Elt> Y,
                                       DenseBuf<Elt>& X, uint32_t row_block = 256);

// (slot-23 G2-E; G2_plan.md Sec. 2.5) THE SAME READ-OFF ON A PRODUCT, Y = A G (A m x l, G l x ny), with Y
// never materialised: Y is formed in row blocks of `row_block`; its pivot rows go straight into X (nb x ny),
// only its m - nb non-pivot rows are kept; the residual Y - B X is then recomputed EXACTLY over ALL m rows (a
// pivot row against the copy in X, a non-pivot row against the kept row), in row blocks.  The same B[piv, :] = I
// check first (NotIdentity before any product is formed), the same statuses, the same validation-only
// ScopedReadoffCorruption hook (X(0, 0) + 1 before the residual).  `perturb_row` (validation only; ~0u = none)
// adds 1 to Y(perturb_row, 0) as Y is formed -- HeckeOperatorOnK's ScopedThetaEqPerturbation.  Its caller
// (HeckeOperatorOnKT::compute_module's Theta_eq: B = G = C, A = Theta) forms Y = W = Theta C itself only on a
// fallback.  The result is the same unique function of (B, A G) as solve_unit_pivot_readoff's.
template<class Elt>
ReadoffStatus solve_unit_pivot_readoff_product(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                               const std::vector<uint32_t>& piv, DenseView<const Elt> A,
                                               DenseView<const Elt> G, DenseBuf<Elt>& X, uint32_t row_block = 256,
                                               uint32_t perturb_row = ~0u);

// (slot-23 G2-B; audit m1) THE d o d PREDICATE, factored out so that it can be tested on planted inputs:
// true iff A B = 0 entrywise (A m x l, B l x n).  gemm = true: one exact kernels::gemm (Sec. 5:
// bit-identical sums); gemm = false: the legacy scalar triple loop (the HECKE_MODULE_STREAM=0 route).
// Both compute the same exact product, so they return the same answer on every input.
template<class Elt>
bool product_is_zero(const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B, bool gemm);

// (slot-23 G2) THE MODULE-STREAM LEVER, HECKE_MODULE_STREAM: default ON; "0"
// = the legacy route (the Howell solve for Theta_eq and X_q, the scalar d o d
// loop, cyclic_decomposition's i-inner column clear), bit for bit; any other
// value warns once on stderr and stays on.  Read once per process; a
// ScopedModuleStream overrides it process-wide (validation use, not nestable
// across threads).  Every route it selects computes a unique function of its
// input with exact arithmetic, so it CANNOT change a value
// (HeckeModuleStreamValidation).  The counters are diagnostic (the EFFECTIVE
// line's solve=readoff|howell).
// (S25 LEVERS-25) The env arm HECKE_MODULE_STREAM=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool module_stream_enabled();
const char* module_stream_env_value();   // its value, or nullptr
const lever::EnvLever& module_stream_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedModuleStream : lever::EnvLever::Scoped {
    explicit ScopedModuleStream(bool on) : Scoped(module_stream_lever(), on) {}
};
// (S23 G2-T) THE MODULE-EVAL LEVER, HECKE_MODULE_EVAL: default ON; "0" = the G2-F term evaluation of the
// stream route (ComputeSymmetricPowerMatrixRing through its FlatMatrix carrier and dense_rows_of, the Pascal row
// rebuilt per term, factor_low_rank's per-entry valuation scan), bit for bit; any other value warns once on stderr
// and stays on.  On: rho_n mod ell^k straight into a row-major buffer from one Pascal row per operator
// (ComputeSymmetricPowerRowsRing, the column steps in a wavefront), and the scan's divisibility test by
// ell^{best_v} (one multiplication per entry; the same pivot).  Consulted ONLY by
// HeckeOperatorOnKT::compute_module_stream (so HECKE_MODULE_STREAM=0 leaves it inert).  Read once per process; a
// ScopedModuleEval overrides it process-wide, not per thread (validation use; two threads holding opposite scopes race
// on the override, as for ScopedModuleStream).  Every route computes the same canonical residues, so it CANNOT change a
// value (HeckeModuleStreamValidation (V)).  When HECKE_MODULE_EVAL is set, one line
// "EFFECTIVE HECKE_MODULE_EVAL mode=fast|legacy source=env (HECKE_MODULE_EVAL=<v>) eval=fast:<a>,legacy:<b>" is
// printed at exit, also when the lever was never consulted (then eval=fast:0,legacy:0; audit F3); the counts
// (operator degrees per route) are also on the HECKE_MODULE_STREAM line.
// (S25 LEVERS-25) The env arm HECKE_MODULE_EVAL=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool module_eval_enabled();
const lever::EnvLever& module_eval_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedModuleEval : lever::EnvLever::Scoped {
    explicit ScopedModuleEval(bool on) : Scoped(module_eval_lever(), on) {}
};
std::string module_eval_effective_text();
// (S23 G2-P) THE MODULE-PARALLEL LEVER, HECKE_MODULE_PARALLEL: default ON; "0" = the serial assembly of the stream
// route, verbatim; any other value warns once on stderr and stays on.  On, and only when the consumer pool
// (engine::tape::consumer_pool(), ENGINE_THREADS) has more than one worker, HeckeOperatorOnKT::compute_module_stream
// assembles Theta^{(q)} and route 2's chunks in coset batches: the distinct keys listed serially, evaluated by the
// pool (rho_n, factor_low_rank, the fused products), accumulated by one task per sigma row in the serial order.  So
// ENGINE_THREADS=1 IS the serial route.  Consulted by compute_module_stream (the assembly and its hook),
// EquivariantCohomologyOnKT::compute_cohomology_module (the hook), howell_rows and solve_unit_pivot_readoff_product
// (the hook splits; G2-P audit F2).  Read once per process; a ScopedModuleParallel overrides it process-wide, not
// per thread (validation use).  The thread count and schedule cannot change a value (HeckeModuleStreamValidation (P)).
// The Howell and read-off splits run over the elimination hook (tape::ScopedHostElimHook), so ENGINE_HOST_ELIM_HOOK=0
// keeps THOSE TWO serial while the assembly stays parallel (no hook is installed; hook=howell:0,readoff:0).  When
// HECKE_MODULE_PARALLEL is set, or ENGINE_HOST_ELIM_HOOK=0, one line
// "EFFECTIVE HECKE_MODULE_PARALLEL mode=on|off source=env (HECKE_MODULE_PARALLEL=<v>) threads=<largest pool seen>
// par=parallel:<a>,serial:<b> hook=howell:<c>,readoff:<e> elim_hook=on|off[ howell_readoff=serial]" (operator degrees per
// path; the Howell steps and the product read-offs split over the elimination hook, see howell_rows /
// solve_unit_pivot_readoff_product; elim_hook=off howell_readoff=serial under ENGINE_HOST_ELIM_HOOK=0) is printed at
// exit, also when never consulted.
// (S25 LEVERS-25) The env arm HECKE_MODULE_PARALLEL=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool module_parallel_enabled();
void module_parallel_count(bool parallel, uint64_t degrees, uint32_t threads);   // threads: the pool size seen
const lever::EnvLever& module_parallel_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedModuleParallel : lever::EnvLever::Scoped {
    explicit ScopedModuleParallel(bool on) : Scoped(module_parallel_lever(), on) {}
};
std::string module_parallel_effective_text();
uint64_t module_parallel_degrees(bool parallel);   // the counters (validation)
uint64_t module_parallel_howell_regions();          // howell_rows steps split over the hook
uint64_t module_parallel_readoff_regions();         // product read-offs run in hook blocks
// (S24 MT-1) THE MODULE-MT LEVER, HECKE_MODULE_MT: default ON; "0" = G2-P's parallel route, verbatim; any other
// value warns once on stderr and stays on.  Two parts:
//   (0) AT EVERY THREAD COUNT (owner ruling 2026-10-20, census mode included): cyclic_decomposition's (c2) pivot
//       search on the lean route (HECKE_MODULE_STREAM on) stops at the first unit in the serial column-major
//       order.  The serial pivot is the FIRST entry of minimal valuation under a strict <; a unit has valuation 0,
//       the least possible, so no later entry replaces it: the same pivot (MT1_report.md Sec. 2.1).
//   (a)-(d) the splits, in effect ONLY where HECKE_MODULE_PARALLEL is on AND an elimination hook is installed on the
//       calling thread (compute_cohomology_module / compute_module_stream install one when the consumer pool has
//       more than one worker); ENGINE_THREADS=1, HECKE_MODULE_PARALLEL=0 and ENGINE_HOST_ELIM_HOOK=0 run none:
//   (a) cyclic_decomposition (lean route): the pivot search's columns after t in tiles (per tile the first minimum
//       in the serial order, exit at a unit, combined in tile order with the same strict <); the row operations on
//       the relation matrix, the U rows and the Gamma rows over disjoint row tiles, each entry receiving the serial
//       update sequence;
//   (b) the dense products of the module route (Y = Theta H, the (iii) fallback W = Theta C and C TE in
//       compute_module_stream; the H basis C . dec.basis in compute_cohomology_module) as row-block products
//       (gemm_rows_hook / the same scalar loop per row): every entry is the same exact sum (ENGINE_API.md Sec. 5);
//   (c) the runtime assertions (i)/(ii) of compute_module_stream evaluated per column over the hook; on any failure
//       the serial loop re-runs and throws its own first message;
//   (d) the T_ell assembly's accumulation step (G2-P (D)) as one task per (sigma row, sigma_k block) instead of per
//       sigma row: each block receives the same GEMM / axpy calls in the same order (blocks are disjoint);
//   (e) the solve_unit_pivot_readoff residual blocks and solve_from_augmented's right-hand sides over the hook;
//       compute_cohomology_module's D C (X_q) in column tiles.  (The opt-in forms overlap HECKE_MODULE_MT_FORMS was
//       removed at the MT-1 audit, F1: after FNTT it no longer paid and it needed an Engine/ include in HeckeExt.)
// Read once per process; ScopedModuleMt overrides it process-wide (validation use).  When HECKE_MODULE_MT is set,
// one line "EFFECTIVE HECKE_MODULE_MT mode=on|off source=env (HECKE_MODULE_MT=<v>) regions=cyc_search:<a>,cyc_rows:<b>,
// gemm:<c>,check:<e>,assembly:<f> pivot_exit=<g>" (split regions taken and cyclic_decomposition calls on the part-(0)
// search, process totals; " (HECKE_MODULE_STREAM off: inert)" or " (HECKE_MODULE_PARALLEL off: splits inert, pivot-search
// exit on)" appended where it applies) is printed at exit when HECKE_MODULE_MT is set, also when the lever is never
// consulted.  The thread count and schedule cannot change a value (HeckeModuleMtValidation).
bool module_mt_enabled();          // the lever alone (part (0); the splits also need module_mt_hook())
const lever::EnvLever& module_mt_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedModuleMt : lever::EnvLever::Scoped {
    explicit ScopedModuleMt(bool on) : Scoped(module_mt_lever(), on) {}
};
// VALIDATION ONLY: cyclic_decomposition's splits ignore their size thresholds (any step with >= 2 rows / columns splits
// over the hook) for the scope; process-wide; never set outside a validation.
struct ScopedModuleMtForceSplit {
    int saved_;
    ScopedModuleMtForceSplit();
    ~ScopedModuleMtForceSplit();
    ScopedModuleMtForceSplit(const ScopedModuleMtForceSplit&) = delete;
    ScopedModuleMtForceSplit& operator=(const ScopedModuleMtForceSplit&) = delete;
};
// (S24 PAR-L1) the module routes' work bound per hook tile (DenseElim.cpp, mod_tiles; env HECKE_MODULE_TILE_WORK, default 2^20
// modular MACs per tile; 0 = the extent-only count of S23 G2-P / S24 MT-1).  Tiling is value-neutral.  The scoped override wins
// over the env (process-wide; validation use: the split arms pin 0 so the split code stays exercised on small inputs).
size_t module_tile_work();
uint64_t module_tile_work_capped();   // splits the bound reduced
uint64_t module_tile_work_serial();   // splits the bound made serial
struct ScopedModuleTileWork {
    long long saved_;
    explicit ScopedModuleTileWork(size_t work);
    ~ScopedModuleTileWork();
    ScopedModuleTileWork(const ScopedModuleTileWork&) = delete;
    ScopedModuleTileWork& operator=(const ScopedModuleTileWork&) = delete;
};
enum class ModuleMtEvent : uint32_t { CycSearch = 0, CycRows = 1, Gemm = 2, Check = 3, Assembly = 4, PivotExit = 5 };   // PivotExit: not a split (MT-1 audit F2)
void module_mt_count(ModuleMtEvent e, uint64_t k = 1);
uint64_t module_mt_regions(ModuleMtEvent e);
std::string module_mt_effective_text();
// VALIDATION ONLY (G2-P audit F1): a fault injected into pass 2 of the PARALLEL product read-off, on a pool
// worker that is not the region's calling thread.  mode 1: that block reports a residual (the call returns
// Residual); mode 2: that block throws std::runtime_error("solve_unit_pivot_readoff_product: injected worker fault
// (validation only)"), which the pool rethrows on the caller after the region.  At most one fault per call; while
// in force the calling thread's own blocks wait (bounded, 10 s) until a worker has taken one, so the fault does
// fire off the caller.  readoff_worker_faults() counts the faults fired (process total).  Process-wide; never
// set outside a validation.
struct ScopedReadoffWorkerFault {
    int saved_;
    explicit ScopedReadoffWorkerFault(int mode);
    ~ScopedReadoffWorkerFault();
    ScopedReadoffWorkerFault(const ScopedReadoffWorkerFault&) = delete;
    ScopedReadoffWorkerFault& operator=(const ScopedReadoffWorkerFault&) = delete;
};
uint64_t readoff_worker_faults();
// (S23 G2-T2) THE ROUTE-2 MEMO LEVER, HECKE_ROUTE2_SHARED_MEMO: default ON; "0" = G2-D's sigma-row chunks of route 2
// (ceil(N_q / 2) block rows when N_q > 7) with the rho memo scoped to (chunk, coset), verbatim; on: one chunk of all
// N_q block rows, so the memo key(delta_i gamma) -> rho (or its factors) spans (degree, coset) -- every distinct key
// of a coset is evaluated once instead of once per chunk (2 chunks at q = 1) -- at the price of a route-2 buffer of
// N_q d x N_q d instead of ceil(N_q / 2) d x N_q d (the size of Theta^{(q)}).  Any other value warns once and stays
// on; ScopedRoute2SharedMemo overrides it process-wide, not per thread (validation use).  Consulted ONLY by
// HeckeOperatorOnKT::compute_module_stream (inert under HECKE_MODULE_STREAM=0).  The same terms are summed into the
// same entries (exact sums mod ell^k) and compared with the same rows of Theta, so it CANNOT change a value
// (HeckeModuleStreamValidation (V) (g)).  When HECKE_ROUTE2_SHARED_MEMO is set, one line
// "EFFECTIVE HECKE_ROUTE2_SHARED_MEMO mode=shared|chunked source=env (HECKE_ROUTE2_SHARED_MEMO=<v>)
// route2_degrees=shared:<a>,chunked:<b>" is printed at exit (operator degrees with N_q > 7, per mode), also when
// the lever is never consulted.
// (S25 LEVERS-25) The env arm HECKE_ROUTE2_SHARED_MEMO=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool route2_shared_memo_enabled();
const lever::EnvLever& route2_shared_memo_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedRoute2SharedMemo : lever::EnvLever::Scoped {
    explicit ScopedRoute2SharedMemo(bool on) : Scoped(route2_shared_memo_lever(), on) {}
};
struct Route2MemoCounts { uint64_t shared = 0, chunked = 0; };
void route2_memo_count(bool shared);
Route2MemoCounts route2_memo_counts();
std::string route2_shared_memo_effective_text();
// (S24 G2-H) THE LEAN-PEAK LEVER, HECKE_MODULE_LEAN_PEAK: default ON; "0" = the route of before G2-H, verbatim; any
// other value warns once on stderr and stays on.  It acts only on the stream route (HECKE_MODULE_STREAM on); its parts
// (b)-(d) act only under the lean retention of the sat / Loewy runner (hecke_ext::RunModuleT):
//   (a) howell_rows: when no Howell closure row was appended and every row of the store is a pivot row, the store's
//       rows are permuted in place into pivot order and the store BECOMES H (DenseBuf::adopt) instead of being
//       copied -- the same rows in the same order, so the same H (the reduce-above runs on it unchanged);
//   (e) the lean cyclic_decomposition / invariant_factors take length(colspan Rl) from howell_rows' pivot loop alone
//       (the pivots and valuations are final before the reduce-above, which is all length() reads): no H is formed;
//   (b) HeckeOperatorOnKT::compute_module_stream under ModuleRetention::Lean: (iii) is certified by
//       unit_pivot_residual_product below (Theta_eq is never formed: the lean runner never reads it);
//   (c) there, route 2's terms are SUBTRACTED from Theta^{(q)} in place, after Y = Theta H and (i)-(iii), and the
//       O12(a) certificate is "Theta - route 2 == 0 entrywise" (no route-2 buffer): the same exact sums mod ell^k;
//   (d) EquivariantCohomologyOnKT::compute_cohomology_module with set_module_early_release(true) (RunModuleT only):
//       first Zc_q = ker X_q and dK_q for every q (ascending), then dec_q and H_q in DESCENDING q, X_q / Zc_q freed as
//       soon as their last reader has run (they are released by release_module_intermediates() anyway).  Every step
//       is a pure function of its inputs, so the order cannot change a value.
// Read once per process; ScopedModuleLeanPeak overrides it process-wide (validation use).  When HECKE_MODULE_LEAN_PEAK
// is set, one line "EFFECTIVE HECKE_MODULE_LEAN_PEAK mode=on|off source=env (HECKE_MODULE_LEAN_PEAK=<v>)
// howell=adopt:<a>,copy:<b> residual=<c> route2_inplace=<e> early_release=<f> length_only=<g>" (Howell forms per
// path on the stream route; operator degrees certified by the residual; operator degrees with route 2 in place;
// cohomology objects with the early release; lengths from the pivot loop) is printed at exit, also when never
// consulted (" (HECKE_MODULE_STREAM off: inert)" appended then).  No part can change a value
// (HeckeModuleStreamValidation (H)).
bool module_lean_peak_enabled();
const lever::EnvLever& module_lean_peak_lever();   // (S24 CL-5) the EnvLever behind it (EnvLever.h)
struct ScopedModuleLeanPeak : lever::EnvLever::Scoped {
    explicit ScopedModuleLeanPeak(bool on) : Scoped(module_lean_peak_lever(), on) {}
};
enum class LeanPeakEvent : uint32_t { HowellAdopt = 0, HowellCopy = 1, Residual = 2, Route2InPlace = 3, EarlyRelease = 4, LengthOnly = 5 };
void lean_peak_count(LeanPeakEvent e, uint64_t k = 1);
uint64_t lean_peak_events(LeanPeakEvent e);
std::string module_lean_peak_effective_text();

// (S24 G2-H) THE READ-OFF CERTIFICATE WITHOUT THE READ-OFF.  The same statuses as solve_unit_pivot_readoff_product
// on (B, piv, A, G) -- NotIdentity on the same malformed piv / B[piv, :] != I, else Ok iff the residual Y - B X of the
// read-off X = Y[piv, :] of Y = A G vanishes over every row -- but X and Y are never formed.  With B[piv, :] = I the
// pivot rows of that residual are identically zero, and its non-pivot rows are R = Y[np, :] - B[np, :] Y[piv, :] =
// (A[np, :] - B[np, :] A[piv, :]) G (associativity in Z/ell^k, exactly).  So Rt = A[np, :] - B[np, :] A[piv, :]
// is formed in tiles of <= 128 non-pivot rows (the tile's rows of A, minus one GEMM per block of <= 64 pivots), and
// each tile's Rt G is compared with 0: the work buffers are O(d) rows, never (m - nb) x l.  perturb_row (validation only) adds 1 to Y(perturb_row, 0), as in the product
// read-off: at a non-pivot row R(w, 0) + 1, at a pivot row piv[u] the residual column 0 moves by -B[np, u]; the
// ScopedReadoffCorruption hook (X(0, 0) + 1 there, so the pivot row piv[0] differs from its copy) makes it Residual
// whenever nb > 0 and Y is not empty.  HECKE_MODULE_PARALLEL with an elimination hook: Rt's row tiles and the
// check's blocks over the hook (disjoint rows; Residual iff some block fails).  Flops (m - nb) l (nb + ny), against
// m l ny + m nb ny for the read-off.
template<class Elt>
ReadoffStatus unit_pivot_residual_product(const ModCtx<Elt>& ctx, DenseView<const Elt> B,
                                          const std::vector<uint32_t>& piv, DenseView<const Elt> A,
                                          DenseView<const Elt> G, uint32_t row_block = 256,
                                          uint32_t perturb_row = ~0u);
struct ModuleStreamTotals {
    uint64_t readoff = 0;        // read-off solves taken (Ok)
    uint64_t howell = 0;         // Howell solves on the stream route (NotIdentity / no pivots / Residual fallback)
    uint64_t legacy = 0;         // Howell solves with the lever off
    uint64_t dd_gemm = 0;        // d o d certificates by GEMM
    uint64_t dd_scalar = 0;      // d o d certificates by the scalar loop (lever off)
    uint64_t cyc_interchanged = 0, cyc_legacy = 0;   // cyclic_decomposition column clears, per call
    // (S23 G2-C / G2-D / G2-E) HeckeOperatorOnKT::compute_module / RunModuleT:
    uint64_t omega_streamed = 0;  // cosets whose Omega_i was evaluated per (degree, coset) and dropped (stream route)
    uint64_t omega_table = 0;     // operator degrees assembled with the legacy Omega table + operator-wide memos
    uint64_t lr_grouped = 0;      // grouped low-rank GEMMs ([c U_1 ... c U_m][V_1; ...; V_m] into one d x d block)
    uint64_t lr_scalar = 0;       // operator degrees with the legacy scalar low-rank accumulation (lever off)
    uint64_t route2_blocked = 0;  // route-2 sigma block ROWS built and compared, then dropped -- one per simplex sigma of
                                  // each degree (N_0 + N_1 + N_2 = 23 per operator), NOT one per chunk (a chunk holds up
                                  // to ceil(N_q / 2) of them when N_q > 7: 1 + 2 + 1 = 4 chunks per operator) (audit n6)
    uint64_t route2_full = 0;     // operator degrees with the legacy materialised route 2 (lever off)
    uint64_t w_nonpivot = 0;      // Theta_eq read-offs on the product (W's non-pivot rows only, never W)
    uint64_t lean_released = 0;   // objects released under the lean retention (operators + cohomology objects)
    uint64_t lr_early_flush = 0;  // (audit m2) grouped low-rank GEMMs flushed early, at the kGroupCap stacked-width cap
                                  // (a subset of lr_grouped; > 0 only for chains of >= 4096 / min(k, d) terms per group)
    // (S23 G2-G) cyclic_decomposition / invariant_factors with the lean work buffers (stream route: N never formed,
    // one Howell form of [Gamma^T | I] for both the kernel and the coordinates, every work buffer dropped once read,
    // length(colspan Rl) after the s = 0 return), or with the legacy buffers (lever off).  Per call.
    uint64_t cyc_lean = 0, cyc_lean_legacy = 0;
    uint64_t invf_lean = 0, invf_legacy = 0;
    // (S23 G2-F) hecke_ext::RunModuleT's tau tables: built TauTableContents::ChainsOnly (stream route: chain records
    // only, no xi-term table, no rho pool) or Full (lever off).  Per table.
    uint64_t tau_chains = 0, tau_pooled = 0;
    // (S23 G2-T) HeckeOperatorOnKT::compute_module_stream's term evaluation, per operator degree: HECKE_MODULE_EVAL
    // on (fast) or off (legacy).
    uint64_t eval_fast = 0, eval_legacy = 0;
};
enum class ModuleStreamEvent { OmegaStreamed, OmegaTable, LrGrouped, LrScalar, Route2Blocked, Route2Full, WNonpivot,
                               LeanReleased, LrEarlyFlush, TauChainsOnly, TauPooled, EvalFast, EvalLegacy };
void module_stream_count(ModuleStreamEvent e, uint64_t n = 1);
ModuleStreamTotals module_stream_totals();
// VALIDATION ONLY (HeckeModuleStreamValidation's negative control): while in force, the read-off adds 1 to
// X(0, 0) before the residual is recomputed, so the residual check must fire (Residual) and the caller must
// fall back to the Howell solve.  Process-wide, not nestable across threads.
struct ScopedReadoffCorruption {
    int saved_;
    explicit ScopedReadoffCorruption(bool on);
    ~ScopedReadoffCorruption();
    ScopedReadoffCorruption(const ScopedReadoffCorruption&) = delete;
    ScopedReadoffCorruption& operator=(const ScopedReadoffCorruption&) = delete;
};
// VALIDATION ONLY (S23 G2-G audit m2; HeckeModuleStreamValidation (C)'s fault injection): while in force, the
// LEAN route of cyclic_decomposition / invariant_factors adds 1 to the relation matrix's entry (0, 0) once it is
// filled (s > 0 and at least one relation), so that a lean-only defect is simulated; the legacy route ignores it.
// Process-wide, not nestable across threads.
struct ScopedLeanBufferCorruption {
    int saved_;
    explicit ScopedLeanBufferCorruption(bool on);
    ~ScopedLeanBufferCorruption();
    ScopedLeanBufferCorruption(const ScopedLeanBufferCorruption&) = delete;
    ScopedLeanBufferCorruption& operator=(const ScopedLeanBufferCorruption&) = delete;
};
void module_stream_count_solve(bool stream_on, ReadoffStatus st, bool readoff_tried);
void module_stream_count_dd(bool gemm);
// One line: "mode=stream|legacy source=env|default|scoped (HECKE_MODULE_STREAM=<v>) solve=readoff:<a>,howell:<b>,
// legacy:<c> dd=gemm:<d>,scalar:<e> cyc=interchanged:<f>,legacy:<g> omega=streamed:<h>,table:<i>
// lr=grouped:<j>,scalar:<k> route2=blocked:<l>,full:<m> w=nonpivot:<o> retention=lean:<r>" (S23 G2-C/D/E),
// then " cyc_buffers=lean:<s>,legacy:<t> invf=lean:<u>,legacy:<v>" (G2-G), " tau=chains:<w>,pooled:<x>" (G2-F) and
// " eval=fast:<y>,legacy:<z>" (G2-T).
std::string module_stream_effective_text();

// The composition length of rowspan(A): sum over its Howell pivots of
// (k - v_i), so |rowspan(A)| = ell^length (Sec. 13.6).  Rank-nullity:
// length(kernel_module(A)) + length(image_module(A)) = cols(A) * k.
template<class Elt>
uint32_t length(const ModCtx<Elt>& ctx, DenseView<const Elt> A);
// The same quantity read off a Howell result already in hand.
template<class Elt>
uint32_t length(const ModCtx<Elt>& ctx, const HowellResult<Elt>& h);

// The exponents e_1 >= e_2 >= ... >= 1 of M = (colspan G + colspan Rl) /
// colspan Rl  ~=  (+)_i Z/ell^{e_i}  (columns in R^m, the complement_basis
// convention of Sec. 8), by Sec. 13.6's SMITH RULE -- not from Howell pivots
// (Sec. 13.10 F2).  G is m x ng, Rl is m x nr (nr may be 0).
template<class Elt>
std::vector<uint32_t> invariant_factors(const ModCtx<Elt>& ctx, DenseView<const Elt> G,
                                        DenseView<const Elt> Rl);
// The case Rl = {} : M = colspan A.
template<class Elt>
std::vector<uint32_t> invariant_factors(const ModCtx<Elt>& ctx, DenseView<const Elt> A);

// (hecke-extensions slot 8, WP-3; E8 item 3) THE CYCLIC-SUMMAND BASIS RULE --
// [X] v1.11 E8 finding M3, stated once in [E] v1.9.0-pre.1 sec:ring (c):
//  (c1) presentation: Gamma (m x s) = the Howell generators of
//       N = colspan[Rl | G], columns in Howell-row order; the relation matrix
//       Rel (s x t) = [ kernel_module(Gamma) rows | solve_multi_rhs_module(
//       Gamma, Rl) ], as columns, in that order;
//  (c2) Smith elimination of Rel on the trailing submatrix: pivot = smallest
//       valuation anywhere, then leftmost column, then topmost row, moved to
//       the diagonal by ONE row swap and ONE column swap; the pivot's COLUMN
//       is scaled by inv(unit_part) (a relation operation) so the pivot is
//       exactly ell^sigma; the pivot's ROW is cleared first (column
//       operations), its COLUMN second (row operations), each by the
//       divide_exact quotient; stop when every trailing entry is 0;
//  (c3) every row operation is mirrored on Gamma by the inverse column
//       operation (row_i -= q row_j  <->  gamma_j += q gamma_i; a row swap
//       swaps two columns of Gamma); column operations touch no generator;
//  (c4) diagonal position i with ell^sigma, 1 <= sigma < k, is a summand
//       Z/ell^sigma generated by the transformed gamma_i; a position whose
//       diagonal entry is 0 or missing is Z/ell^k; a unit diagonal entry
//       gives nothing.  Summands are ordered by exponent DESCENDING, ties by
//       diagonal position ascending (a stable sort); sum e_i ==
//       length(N) - length(colspan Rl) is ENGINE_CHECKed;
//  (c5) coordinates: a class x in M has coordinates c_i with x = sum c_i b_i,
//       the i-th reduced into [0, ell^{e_i}) -- unique.
// The exponents equal invariant_factors(ctx, G, Rl) (Sec. 13.6's rule, which
// reads only valuations).  ENGINE_CHECKs a composite context.
template<class Elt> struct CyclicDecomposition {
    std::vector<uint32_t> exponents;   // e_1 >= e_2 >= ... >= 1  (c4)
    DenseBuf<Elt> basis;               // m x t: column i is b_i, generating Z/ell^{e_i}
    // The coordinate machinery of (c5): Gamma (m x s, c1) and the accumulated
    // row operations U (s x s, c2-c3), so that Gamma_transformed = Gamma U^{-1}
    // and a vector y = Gamma x has transformed coordinates U x.
    DenseBuf<Elt> Gamma;
    DenseBuf<Elt> U;
    std::vector<uint32_t> position;    // summand i -> its diagonal position (a row of U)
    uint32_t rows() const { return static_cast<uint32_t>(exponents.size()); }
};
template<class Elt>
CyclicDecomposition<Elt> cyclic_decomposition(const ModCtx<Elt>& ctx, DenseView<const Elt> G,
                                              DenseView<const Elt> Rl);
// (c5) for every column y_t of Y (m x ny; each must lie in colspan N, else
// solve_multi_rhs_module's named runtime_error): result[t][i] = the
// coordinate of b_i, in [0, ell^{e_i}).
template<class Elt>
std::vector<std::vector<Elt>> cyclic_coordinates(const ModCtx<Elt>& ctx, const CyclicDecomposition<Elt>& d,
                                                 DenseView<const Elt> Y);

// ---------------------------------------------------------------------------
// Backend selection.  backend_is_legacy() (ENGINE_BACKEND=legacy) is the
// process-wide rollback; ScopedLegacy is a thread-local override so the
// validations and the micro-benchmark can call the OLD FlatMatrix /
// BlockFlatMatrix code paths in the same process as the new ones.
// (S24 CL-L2) Both exist only in an OC_ENABLE_LEGACY=ON build (definitions:
// legacy/DenseElimLegacy.inc); an OFF build has no legacy route to select, and
// ENGINE_BACKEND=legacy is refused there (EngineConfig.h).
// ---------------------------------------------------------------------------
#if OC_ENABLE_LEGACY
bool legacy_forced();
#endif

// (matrix-engine Phase 4) Inner parallelism for the elimination when it runs
// as a host-serial tape node (PANEL_RREF) or on the host: a THREAD-LOCAL hook
// `parallel_for(ntasks, fn)` that the tape executor installs while it runs
// such a node on the calling thread, backed by its thread pool.  Only the
// trailing update of the blocked PLU (A22 -= L21 U12, split over disjoint row
// tiles) uses it, so every value is the same exact integer whatever the tile
// count; nullptr (the default) is the serial path.
using ParallelFor = std::function<void(uint32_t ntasks, const std::function<void(uint32_t)>& fn)>;
const ParallelFor* parallel_hook();
struct ScopedParallelHook {
    const ParallelFor* saved_;
    explicit ScopedParallelHook(const ParallelFor* hook);
    ~ScopedParallelHook();
    ScopedParallelHook(const ScopedParallelHook&) = delete;
    ScopedParallelHook& operator=(const ScopedParallelHook&) = delete;
};
// (S23 G2-P audit F3) Marks the calling thread as running a hook task for its scope, exactly as run_hook does:
// parallel_hook() returns nullptr inside, so an elimination called from a task of some OTHER pool region (e.g.
// HeckeOperatorOnKT::compute_module_stream's coset-batch tasks, run partly by the calling thread, which carries
// the installed hook) is serial on every thread alike -- no nested region opened from a task.
struct ScopedInHookTask {
    bool saved_;
    ScopedInHookTask();
    ~ScopedInHookTask();
    ScopedInHookTask(const ScopedInHookTask&) = delete;
    ScopedInHookTask& operator=(const ScopedInHookTask&) = delete;
};
// (S24 MT-1) parallel_hook() when HECKE_MODULE_MT and HECKE_MODULE_PARALLEL are both on, else nullptr (see above).
const ParallelFor* module_mt_hook();
// (S24 MT-1) C = A B (op Set) as row-block GEMMs over `par` (module_mt_hook()); serial kernels::gemm when par is
// null or the product is below the split threshold.  Every entry is the same exact sum mod ell^k whatever the row
// split (ENGINE_API.md Sec. 5), so the result is bit-identical to kernels::gemm(ctx, A, B, C).
template<class Elt>
void gemm_rows_hook(const ParallelFor* par, const ModCtx<Elt>& ctx, DenseView<const Elt> A, DenseView<const Elt> B,
                    DenseView<Elt> C);
#if OC_ENABLE_LEGACY
struct ScopedLegacy {
    ScopedLegacy();
    ~ScopedLegacy();
    ScopedLegacy(const ScopedLegacy&) = delete;
    ScopedLegacy& operator=(const ScopedLegacy&) = delete;
};
inline bool use_legacy() { return backend_is_legacy() || legacy_forced(); }
#endif

} // namespace elim
} // namespace engine
