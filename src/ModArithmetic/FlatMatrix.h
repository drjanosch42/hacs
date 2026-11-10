// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#ifndef FLAT_MATRIX_H
#define FLAT_MATRIX_H

#include "ModArithmetic.h"
#include "LegacyConfig.h"    // OC_ENABLE_LEGACY guards the legacy members below (S24 CL-L2)
#include "ObjectCounters.h"  // (matrix-engine Phase 3) opt-in construction census
#include "helper.h" // For offset_string
#include "Matrix.h" // For conversion
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>   // For to_string
#include <iomanip>   // For to_string
#include <algorithm> // For std::min, std::swap, std::all_of, std::equal etc.
#include <numeric>   // For std::iota in kernel of zero matrix
#include <memory>    // For std::unique_ptr (RREF caches)

// ===========================================================================
// Stage S13 -- element-width generalization ([C, thm:backend] "Mod-ell
// backend"). `FlatMatrixT<Elt>` is the row-major flat F_ell matrix over an
// element type Elt (uint32_t for the primary regime ell < 2^32, uint64_t for
// ell < 2^64). Modular arithmetic picks mod_mul vs mod_mul64 by element width
// through the ModArithmetic::mod_*_w overloads, so the uint32_t instantiation
// is BYTE-IDENTICAL to the pre-S13 code. Dimensions/indices/pivot-column
// caches stay uint32_t; only the stored values and the modulus become Elt.
//
//   using FlatMatrix   = FlatMatrixT<uint32_t>;  (default, source-compatible)
//   using FlatMatrix64 = FlatMatrixT<uint64_t>;  (extended regime)
// ===========================================================================

template<class Elt>
class FlatMatrixT {
private:
    uint32_t rows_;
    uint32_t cols_;
    Elt p_; // Prime modulus

    mutable std::vector<Elt> flat_data_; // Data stored in row-major order. Can be empty if special.

    mutable bool is_scalar_matrix_;  // True if M = alpha * I. alpha can be 0.
    mutable Elt scalar_value_;  // Stores alpha if is_scalar_matrix_ is true.

    // Cached RREF - marked mutable as they are lazy-computed by const functions.
    // (matrix-engine Phase 1, defect #20) Owning unique_ptr, not raw new/delete;
    // BlockFlatMatrix was migrated in an earlier pass, FlatMatrix was not.
    // (matrix-engine Phase 2, defect #7) The cache is a FLAT row-major buffer
    // of rows_*cols_ entries: the engine eliminates in place inside it through
    // a DenseView, so there is no vector<vector> round trip any more.
    mutable std::unique_ptr<std::vector<Elt>> row_echelon_form_cache_;
    mutable std::unique_ptr<std::vector<uint32_t>> pivot_columns_cache_;

#ifdef ENGINE_COUNT_OBJECTS
    // (matrix-engine Phase 3) Construction census; see ObjectCounters.h. Present
    // ONLY under -DENGINE_COUNT_OBJECTS, so layout and sizeof are untouched by
    // default. Deliberately NOT named in any mem-init list: it is then default-
    // initialised by every constructor, copy and move included.
    engine::counters::FlatMatrixCtorTick object_counter_tick_;
#endif


    // Private helper methods
    void _clear_cached_rref() const; // Clears RREF cache

    // (matrix-engine Phase 1, defect #24) _ensure_data_populated_if_empty() was
    // deleted here: 39 lines of self-argument, two const_casts of `this`, around
    // the single call `populate_flat_data()` -- which is itself const, because
    // flat_data_ is mutable.  Its call sites now spell out the one condition that
    // mattered, `if (!is_scalar_matrix_) populate_flat_data();`, which keeps the
    // scalar-storage optimisation intact (an unconditional call would materialise
    // a d_V x d_V buffer for every identity block).

    // Analyzes flat_data_ to set scalar flags.
    // Assumes flat_data_ is populated.
    void _determine_and_set_flags_from_data() const;

#if OC_ENABLE_LEGACY
    // (S24 CL-L2) The pre-Phase-2 scalar route, selected by engine::elim::use_legacy()
    // (ENGINE_BACKEND=legacy / ScopedLegacy) -- ON builds only; legacy/FlatMatrixLegacy.inc.
    // Internal Gaussian elimination logic.
    void _gaussian_elimination_core(
        std::vector<std::vector<Elt>>& mat_2d,
        std::vector<uint32_t>& pivots,
        uint32_t num_mat_rows,
        uint32_t num_mat_cols) const;
    void _legacy_row_echelon_form() const;
    std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> _legacy_rref_and_transform() const;
    std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> _legacy_elimination_factor_and_apply(const FlatMatrixT<Elt>& pivot_rref_block) const;
    FlatMatrixT<Elt> _legacy_inverse() const;
#endif

public:
    // Constructors
    FlatMatrixT(); // Default constructor
    FlatMatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime); // Defaults to zero matrix (scalar 0)
    FlatMatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime, const std::vector<Elt>& init_flat_data);
    FlatMatrixT(uint32_t dim, Elt prime); // Creates identity matrix (scalar 1, or scalar 0 if dim=0)

    // Adopts `canonical_data` (row-major, every entry already a canonical
    // residue mod `prime` -- the engine's buffers are) WITHOUT the copy and
    // the re-reduction of the vector constructor; only the scalar flag is
    // determined.  (matrix-engine Phase 4: the Theta_eq hand-over.)
    static FlatMatrixT FromCanonical(uint32_t num_rows, uint32_t num_cols, Elt prime, std::vector<Elt>&& canonical_data);
    static FlatMatrixT CreateZero(uint32_t num_rows, uint32_t num_cols, Elt prime);
    static FlatMatrixT CreateScalar(uint32_t dim, Elt scalar_val, Elt prime);

    FlatMatrixT(const MatrixT<Elt>& other_matrix); // Constructor from standard Matrix

    // Destructor
    ~FlatMatrixT();

    // Copy/Move semantics
    FlatMatrixT(const FlatMatrixT& other);
    FlatMatrixT& operator=(const FlatMatrixT& other);
    FlatMatrixT(FlatMatrixT&& other) noexcept;
    FlatMatrixT& operator=(FlatMatrixT&& other) noexcept;
    void swap(FlatMatrixT& other);

    // Accessors
    uint32_t get_rows() const { return rows_; }
    uint32_t get_cols() const { return cols_; }
    Elt get_p() const { return p_; }

    // Check current special status (based on flags)
    bool is_currently_scalar(Elt& val) const {
        if (is_scalar_matrix_) {
            val = scalar_value_;
            return true;
        }
        return false;
    }

    // Element access
    Elt get_entry(uint32_t r, uint32_t c) const;
    void set_entry(uint32_t r, uint32_t c, Elt value); // Modifies matrix, may change flags
    // Extract consecutive rows as submatrix, useful for block matrix operations
    FlatMatrixT extract_sub_matrix_rows(uint32_t row_start_idx, uint32_t num_rows_to_extract) const;

    // Conversion and Output
    MatrixT<Elt> to_matrix() const;
    std::string to_string(void) const;
    std::string to_string_offset(uint32_t offset) const;
    std::string to_string_transposed(void) const;
    std::string to_string_transposed_offset(const uint32_t offset) const;

    // Operations
    FlatMatrixT transpose() const; // Transpose might change from non-scalar to scalar 0

    bool is_zero() const;

    FlatMatrixT operator+(const FlatMatrixT& other) const;
    FlatMatrixT operator-(const FlatMatrixT& other) const;
    FlatMatrixT operator*(const FlatMatrixT& other) const;
    FlatMatrixT operator*(Elt scalar) const;

    // Non-const due to potentially calling is_zero()
    bool operator==(const FlatMatrixT& other) const;
    bool operator!=(const FlatMatrixT& other) const;

    uint32_t nonzero_entries() const;

    // Linear Algebra
    void compute_row_echelon_form() const; // Computes and caches RREF and pivots
    // The cached RREF, row-major, rows_*cols_ entries (entry (i,j) at i*cols_+j).
    const std::vector<Elt>& get_row_echelon_form_data() const;
    FlatMatrixT get_row_echelon_form_as_flat_matrix() const;
    const std::vector<uint32_t>& get_pivot_columns() const;
    // Returns a pair: (RREF_of_this, Transformation_C_such_that_C_times_this_is_RREF)
    std::pair<FlatMatrixT, FlatMatrixT> get_rref_and_transform_matrix() const;
    // Takes (B0) and a pivot_rref (R). Returns (B0_new, L_factor)
    // such that B0_new = B0 - L_factor * R, and B0_new is "zeroed out" by R's pivots.
    // B0 is m x n. R is k x n. L_factor will be m x k. B0_new is m x n.
    std::pair<FlatMatrixT, FlatMatrixT> compute_elimination_factor_and_apply(const FlatMatrixT& pivot_rref_block) const;

    // (matrix-engine Phase 1, defect #19) const: everything these touch is either
    // a mutable lazy cache (the RREF) or the mutable scalar-flag pair that is a
    // pure function of the data, so none of them changes the matrix's value.
    // Callers no longer have to deep-copy a matrix just to ask its rank -- which
    // in the block-elimination inner loop meant recomputing the pivot block's
    // RREF once per target row.
    uint32_t rank() const;
    bool is_invertible() const;
    // Inverse itself can be const if is_invertible (non-const) is called first to set up state.
    // Or, make it non-const too. Let's keep const for now, assumes state ready or uses const RREF methods.
    FlatMatrixT inverse() const;
    std::vector<FlatMatrixT> kernel() const;
    std::vector<FlatMatrixT> image() const;

    // Ensures flat_data_ is allocated and reflects the current matrix state (special or general).
    void populate_flat_data() const;

    const std::vector<Elt>& get_flat_data_unsafe() const { return flat_data_; }
};

using FlatMatrix = FlatMatrixT<uint32_t>;
using FlatMatrix64 = FlatMatrixT<uint64_t>;

template<class Elt>
FlatMatrixT<Elt> operator*(Elt scalar, const FlatMatrixT<Elt>& mat);

// Free functions ported from Matrix utilities
/**
 * @brief rho_n(A): the matrix of the induced action of the 2x2 matrix A on
 *        Sym^n_sym(F_ell^2), in the ordered monomial basis
 *        u_k = x^(n_sym-k) y^k, k = 0 .. n_sym.
 *
 * matrix-engine Phase 1 (items 7.0/7.5): O(d_V^2) via closed forms for the
 * structured generators plus a two-term column recurrence in the general case.
 * See the branch table at the definition in FlatMatrix.cpp. Correct for EVERY
 * 2x2 matrix over F_ell, singular ones included (Hecke coset representatives
 * have determinant p, hence det == 0 mod ell at the bad prime ell = p).
 */
template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixFM(const FlatMatrixT<Elt>& A, uint32_t n_sym);

/**
 * @brief Pre-Phase-1 reference evaluation of rho_n (direct double sum with
 *        mod_nCr/mod_pow per term). Kept ONLY as the bit-comparison oracle of
 *        SymmetricPowerValidation; it is super-cubic and must not be called
 *        from production code.
 */
template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixFM_reference(const FlatMatrixT<Elt>& A, uint32_t n_sym);

// ===========================================================================
// (hecke-extensions slot 8, WP-3; E8-P5, docs/implementation/
// e8_ladic_investigation.md Sec. 4.2; ENGINE_API.md Sec. 13.10 F4) rho_n OVER
// A COMPOSITE MODULUS Z/ell^k.
//
// rho_n(g) is integral and reduces mod ell^k, but ComputeSymmetricPowerMatrixFM
// cannot compute it there: its binomials are Lucas's theorem over factorial
// tables that invert (ell-1)! and beyond, and its two recurrences divide by a
// (resp. b) under a FIELD test `a != 0`.  These are SEPARATE functions: the
// F_ell route is not touched, and a caller selects this one by
// `!ctx.is_field()` alone (ENGINE_API.md Sec. 13.8).  Each ENGINE_CHECKs a
// composite context.
//
//  * ComputeSymmetricPowerMatrixRing -- the production route, O(d_V^2) on every
//    path the pipeline takes:
//      [U_a] a a unit of Z/ell^k: the [R1] forward recurrence with inv(a) (the
//            division is exact in Z[y], hence mod ell^k once a is a unit);
//      [U_b] else b a unit: the mirrored [R2] recurrence with inv(b);
//      [S]   else ell | a and ell | b: the diag(ell, 1) SPLIT
//            g = diag(ell, 1) * g',  g' = [[a/ell, b/ell], [c, d]]  (integer
//            division of the canonical representatives, exact), so
//            rho_n(g) = rho_n(diag(ell, 1)) rho_n(g') -- row k scaled by
//            ell^(n-k) -- recursing on g' (U_ell's cosets, det ell, land in
//            [U_a] / [U_b] after one split);
//      [Z]   a == b == 0: the first row of g vanishes, entry(n, j) = c^(n-j) d^j.
//    Binomials come from a Pascal row built by ADDITIONS mod ell^k; nothing
//    here calls mod_nCr, mod_pow or the thread-local factorial tables (M2(a)).
//  * ComputeSymmetricPowerMatrixRingOracle -- the DIVISION-FREE oracle,
//    O(d_V^3): entry(k, j) = sum_t C(n-j, k-t) a^(n-j-k+t) c^(k-t) C(j, t)
//    b^(j-t) d^t, the coefficient of y^k in (a + c y)^(n-j) (b + d y)^j, with a
//    Pascal table mod ell^k.  Validation only (HeckeReductionValidation (P)).
//
// Both return a d_V x d_V FlatMatrixT over the modulus ctx.p = ell^k (the
// FlatMatrix is a storage carrier here; nothing field-only is run on it), in
// the basis and column convention of ComputeSymmetricPowerMatrixFM.  Inputs
// are canonical residues in [0, ell^k).
// ===========================================================================
namespace engine { template<class Elt> struct ModCtx; }
enum class RhoRingRoute : uint32_t { UnitA = 0, UnitB = 1, Split = 2, FirstRowZero = 3 };
const char* to_text(RhoRingRoute r);
template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixRing(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d,
                                                 uint32_t n_sym, RhoRingRoute* route = nullptr);
template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixRingOracle(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d,
                                                       uint32_t n_sym);
// (S23 G2-T) The module route's term evaluation (HECKE_MODULE_EVAL, DenseElim.h): the SAME entries as
// ComputeSymmetricPowerMatrixRing, row-major d_V x d_V into `out` (resized), without the FlatMatrix carrier, from
// the caller's Pascal row (`pascal` = RingPascalRow(ctx, n_sym), built once per (ell^k, n)), the column steps of
// [U_a] / [U_b] interleaved in a skewed wavefront.  Value-neutral: every entry is the same expression of the same
// canonical residues.  `work` is scratch (reused across calls).
template<class Elt>
std::vector<Elt> RingPascalRow(const engine::ModCtx<Elt>& ctx, uint32_t n_sym);
template<class Elt>
void ComputeSymmetricPowerRowsRing(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d, uint32_t n_sym,
                                   const std::vector<Elt>& pascal, std::vector<Elt>& out, std::vector<Elt>& work);
// (S23 G2-T2) THE RHO FILL PATH of ComputeSymmetricPowerRowsRing's [U_a] / [U_b] column steps (u32, p <= 2^31 only;
// u64 words and u32 moduli above 2^31 always take the scalar wavefront): Auto (the default) = AVX-512F if the CPU has
// it, else AVX2, else the scalar 8-lane wavefront (G2-T); chosen at runtime (__builtin_cpu_supports, target
// attributes: no -march flag, an x86-64 baseline build runs the AVX paths on a CPU that has them; non-x86 builds have
// the scalar path only).  Env HECKE_RHO_SIMD (read once): "0" / "scalar" = the scalar wavefront, "avx2" / "avx512" =
// that path (if unavailable: a warning, Auto stays), "1" / "auto" = Auto; anything else warns once, Auto stays.
// (S25 LEVERS-25) "0" / "scalar" are the retired rollback readings: REFUSED in OFF builds (OC_ENABLE_LEGACY=OFF,
// exit 1, at the first read and at cohomology_app's start-up); the scalar path stays in every build.
// ScopedRhoFillPath overrides it process-wide, not per thread (validation use; throws if the path is unavailable).
// Every path computes the same canonical residues (Shoup products and canonical sums), so it CANNOT change a value
// (HeckeModuleStreamValidation (V) forces each available path).  When HECKE_RHO_SIMD is set, one line
// "EFFECTIVE HECKE_RHO_SIMD path=<p> source=env (HECKE_RHO_SIMD=<v>) cpu=avx2:<0|1>,avx512:<0|1>
// fills=avx512:<a>,avx2:<b>,scalar:<c>" is printed at exit (fills = [U] column-step runs per path).
enum class RhoFillPath : int { Auto = 0, Scalar = 1, Avx2 = 2, Avx512 = 3 };
const char* to_text(RhoFillPath p);
bool rho_fill_path_available(RhoFillPath p);   // Auto / Scalar always; Avx2 / Avx512 by CPU and build
RhoFillPath rho_fill_path_resolved();          // the path a u32 fill takes now (never Auto)
void rho_fill_env_check();                     // (S25 LEVERS-25 fix 1) reads only the raw HECKE_RHO_SIMD value (the OFF-build refusal); never resolves ENGINE_KERNEL
struct ScopedRhoFillPath {
    int saved_ = -1;
    explicit ScopedRhoFillPath(RhoFillPath p);
    ~ScopedRhoFillPath();
    ScopedRhoFillPath(const ScopedRhoFillPath&) = delete;
    ScopedRhoFillPath& operator=(const ScopedRhoFillPath&) = delete;
};
struct RhoFillCounts { uint64_t avx512 = 0, avx2 = 0, scalar = 0; };
RhoFillCounts rho_fill_counts();
std::string rho_fill_effective_text();

template<class Elt>
std::vector<Elt> change_of_basis_fm(const std::vector<FlatMatrixT<Elt>>& basis, const FlatMatrixT<Elt>& vector);
template<class Elt>
std::vector<std::vector<Elt>> change_of_basis_fm(const std::vector<FlatMatrixT<Elt>>& basis, const std::vector<FlatMatrixT<Elt>>& vectors);

// Multi-right-hand-side change of basis (matrix-engine Phase 1, item 7.1).
// Same contract and same coordinates as calling the single-RHS
// change_of_basis_fm(basis, vectors[t]) once per t -- BIT-IDENTICALLY so (see
// the proof note at the definition in FlatMatrix.cpp) -- but with ONE
// augmented RREF of [basis | vectors] instead of vectors.size() RREFs whose
// first basis.size() columns are identical every time.
// result[t][j] is the coefficient of basis[j] in vectors[t].
// Throws (like the single-RHS overload) if some vectors[t] is not in
// span(basis); the message names the offending right-hand side.
template<class Elt>
std::vector<std::vector<Elt>> change_of_basis_fm_multi(const std::vector<FlatMatrixT<Elt>>& basis,
                                                       const std::vector<FlatMatrixT<Elt>>& vectors);

// (matrix-engine Phase 2, Stage B) acc += A * B for two GENERAL, buffer-backed
// FlatMatrixT operands, written straight into the caller's row-major
// `acc_rows x acc_cols` buffer via engine::kernels::gemm (GemmOp::Add).
//
// Returns false -- having touched nothing -- when the engine route is not
// admissible: ENGINE_BACKEND=legacy, a scalar-stored (alpha*I) operand, a
// degenerate modulus (p < 2) or a zero dimension. The caller must then run its
// own accumulation, which is what BlockFlatMatrixT<Elt>::operator* does.
//
// It exists because that operator accumulates in a plain std::vector<Elt>, so
// it cannot reach the engine through FlatMatrixT::operator* without allocating
// one temporary FlatMatrix per k. Defined in FlatMatrix.cpp, which is the only
// translation unit that includes the engine headers.
template<class Elt>
bool flat_gemm_accumulate_engine(const FlatMatrixT<Elt>& A, const FlatMatrixT<Elt>& B,
                                 std::vector<Elt>& acc, uint32_t acc_rows, uint32_t acc_cols);

#endif // FLAT_MATRIX_H
