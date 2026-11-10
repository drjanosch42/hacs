// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "FlatMatrix.h"
#include "ModArithmetic.h"
#include "EngineCheck.h"   // ENGINE_CHECK / ENGINE_DEBUG_CHECK (matrix-engine Phase 0 policy)
// matrix-engine Phase 2, Stage B (retrofit): the dense arithmetic below no
// longer hand-rolls its inner loops -- it hands non-owning views of
// flat_data_ to engine::kernels, and every elimination path routes through
// engine::elim (DenseElim.h).
#include "LegacyConfig.h"   // (S25 LEVERS-25) the HECKE_RHO_SIMD=0 / scalar refusal
#include "EngineConfig.h"  // engine::backend_is_legacy() -- runtime rollback
#include "ModCtx.h"
#include "DenseBuf.h"
#include "DenseView.h"
#include "Kernels.h"
#include "DenseElim.h"
#include <type_traits>
#include <atomic>     // (S23 G2-T2) the rho fill path override and counters
#include <cstdio>
#include <cstdlib>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(RHO_FILL_PORTABLE_ONLY)
#  include <immintrin.h>   // (S23 G2-T2) the dispatched SIMD column steps (target attributes; no -m flag needed)
#endif
#include <cstring>    // std::memcmp (legacy/ShadowLegacy.inc, ENGINE_SHADOW_LEGACY)
#include <numeric>    // For std::iota
#include <iostream>   // For debugging if needed

// ===========================================================================
// matrix-engine Phase 2, Stage B -- the engine bridge.
//
// WHAT CHANGES AND WHAT CANNOT.  Every retrofitted operation below computes
// the SAME F_p value it computed before: the engine accumulates products of
// canonical residues EXACTLY (uint64_t for Elt = uint32_t, unsigned __int128
// for uint64_t) over k-blocks of at most ctx.k_max terms and reduces once per
// block (ENGINE_API.md Sec. 5), so reassociation, register blocking,
// vectorisation and the choice of CPU path cannot move a single bit. The two
// routes are therefore bit-identical BY CONSTRUCTION, not by luck -- and
// -DENGINE_SHADOW_LEGACY=ON runs both on every call and memcmps them anyway.
//
// TWO ROLLBACKS, both only in an OC_ENABLE_LEGACY=ON build since S24 CL-L2.
// `ENGINE_BACKEND=legacy` (runtime, no rebuild) sends every
// retrofitted path back to the pre-Phase-2 loop; -DENGINE_SHADOW_LEGACY=ON
// (compile time; needs -DOC_ENABLE_LEGACY=ON since S24 CL-L1, the
// comparisons are legacy/ShadowLegacy.inc) keeps both and compares. See
// CMakeLists.txt and EngineConfig.h.
//
// SCALAR (alpha*I) STORAGE IS UNTOUCHED.  FlatMatrix stores alpha*I as a
// single scalar with no buffer at all; every one of those fast paths is
// exactly the branch it was, and the engine is asked only for the
// general x general case, where both operands really do own a row-major
// buffer. Nothing here materialises a scalar block.
// ===========================================================================
namespace {

// A ModCtx costs a handful of divisions (plus, for uint64_t, a five-step
// Newton inverse) to build, and the block layer calls these operators
// thousands of times per elimination on small blocks -- so the context for the
// modulus last seen is cached per thread. ModCtx is a trivially copyable
// aggregate of scalars; one slot suffices because the pipeline works with one
// modulus at a time and a change of modulus simply re-inits it.
template<class Elt>
const engine::ModCtx<Elt>& engine_ctx(Elt p) {
    static thread_local engine::ModCtx<Elt> cached;
    if (cached.p != p) cached.init(p);
    return cached;
}

// Is the engine route admissible here? `ENGINE_BACKEND=legacy` is the runtime
// rollback and engine::elim::ScopedLegacy the thread-local one -- the two are
// read through the single predicate engine::elim::use_legacy(), so ONE guard
// rolls back products and elimination together (the micro-benchmark's
// `--backend legacy` column depends on that: it must mean "the whole
// pre-Phase-2 substrate", not "pre-Phase-2 elimination with engine
// products").  p < 2 is not a modulus ModCtx accepts (a default-constructed
// FlatMatrixT carries p_ == 0, and 0-dimensional matrices never reach a
// kernel anyway).
template<class Elt>
inline bool engine_path(Elt p) {
#if OC_ENABLE_LEGACY
    return p >= Elt(2) && !engine::elim::use_legacy();
#else
    return p >= Elt(2);   // (S24 CL-L2) an OFF build has no legacy backend to select
#endif
}

// Non-owning, row-major DenseBuf over an existing flat_data_ buffer: no copy,
// no allocation, ld == cols. The const_cast is safe for read-only operands --
// they are handed straight to a DenseView<const Elt>, which is the only thing
// a kernel ever gets for an input.
template<class Elt>
inline engine::DenseBuf<Elt> dense_wrap(const Elt* d, uint32_t rows, uint32_t cols) {
    return engine::DenseBuf<Elt>(const_cast<Elt*>(d), static_cast<size_t>(rows) * cols,
                                 rows, cols, engine::Layout::RowMajor, cols);
}

#ifdef ENGINE_SHADOW_LEGACY
#include "legacy/ShadowLegacy.inc"   // (S24 CL-L1) the shadow comparisons
#endif

// ---------------------------------------------------------------------------
// matrix-engine Phase 2 -- elimination retrofit helpers.
//
// Every elimination path of FlatMatrix (compute_row_echelon_form,
// get_rref_and_transform_matrix, compute_elimination_factor_and_apply,
// inverse) now routes through engine::elim on a DenseView over flat storage.
// The pre-Phase-2 scalar core (_gaussian_elimination_core) is kept and is
// selected by engine::elim::use_legacy() -- ENGINE_BACKEND=legacy at runtime
// or a ScopedLegacy guard (validations, micro-bench).  (S24 CL-L2) It lives in
// legacy/FlatMatrixLegacy.inc and exists only in an OC_ENABLE_LEGACY=ON build;
// the OFF build carries the engine route alone.  A build with
// -DENGINE_SHADOW_LEGACY=1 runs BOTH on every call and throws
// std::logic_error on the first differing entry.  See DenseElim.h for why
// the two must agree bit for bit.

} // namespace

#if OC_ENABLE_LEGACY
#include "legacy/FlatMatrixLegacy.inc"   // (S24 CL-L2) the pre-Phase-2 scalar elimination route
#endif

// --- Private Helper Methods ---

template<class Elt>
void FlatMatrixT<Elt>::_clear_cached_rref() const {
    row_echelon_form_cache_.reset();
    pivot_columns_cache_.reset();
}

// (matrix-engine Phase 1, defect #24) _ensure_data_populated_if_empty() lived
// here.  Deleted: 39 lines of commentary arguing with itself, two const_casts of
// `this`, wrapped around one conditional `populate_flat_data()` call.  Its call
// sites now spell that condition out directly.

template<class Elt>
void FlatMatrixT<Elt>::_determine_and_set_flags_from_data() const {
    _clear_cached_rref(); // Data changed (or flags might change based on data), so RREF is invalid

    if (rows_ == 0 || cols_ == 0) {
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
        if (!flat_data_.empty()) flat_data_.clear();
        return;
    }

    if (flat_data_.empty()) {
        // This implies it was a special matrix whose data wasn't populated for this call,
        // or it's a new matrix being initialized from data that turned out to be empty.
        // If it's for a non-0-dim matrix and flat_data_ is empty, this is typically an error
        // for this function, which expects populated data to analyze.
        // However, if we treat "empty data for non-0-dim" as implicitly "scalar 0", then:
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
        return;
        // Or, more strictly:
        // throw std::logic_error("_determine_and_set_flags_from_data called with unpopulated flat_data_ for non-0-dim matrix");
    }
    if (flat_data_.size() != (static_cast<size_t>(rows_) * cols_)) {
        throw std::logic_error("_determine_and_set_flags_from_data: flat_data_ size mismatch.");
    }


    bool all_elements_are_zero = true;
    for (Elt x : flat_data_) {
        if (x != 0) {
            all_elements_are_zero = false;
            break;
        }
    }

    if (all_elements_are_zero) {
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
        // (matrix-engine Phase 4, memory profile 05 Sec. 7 item 4) A scalar
        // state owns no data: release the buffer, exactly as CreateScalar /
        // CreateIdentity leave it.  Every reader goes through
        // populate_flat_data() or is_currently_scalar(), which re-materialise.
        std::vector<Elt>().swap(flat_data_);
        return;
    }

    // Not all zero. Check for M = alpha * I, alpha != 0. Requires square.
    if (rows_ == cols_ && rows_ > 0) { // rows_ > 0 implicit from not all_elements_are_zero if square
        Elt potential_scalar = flat_data_[0]; // data[0*cols + 0]
        // If M(0,0) is 0, but matrix is not all-zero, cannot be alpha*I (alpha!=0)
        if (potential_scalar == 0) {
            is_scalar_matrix_ = false; // General matrix
        }
        else {
            bool is_indeed_scalar = true;
            for (uint32_t i = 0; i < rows_; ++i) {
                for (uint32_t j = 0; j < cols_; ++j) {
                    Elt expected_val = (i == j) ? potential_scalar : 0;
                    if (flat_data_[static_cast<size_t>(i) * cols_ + j] != expected_val) {
                        is_indeed_scalar = false;
                        break;
                    }
                }
                if (!is_indeed_scalar) break;
            }
            if (is_indeed_scalar) {
                is_scalar_matrix_ = true;
                scalar_value_ = potential_scalar;
                std::vector<Elt>().swap(flat_data_);   // alpha*I owns no data (see the all-zero branch)
            }
            else {
                is_scalar_matrix_ = false; // General matrix
            }
        }
    }
    else { // Not square (and not all zero), so it's a general matrix
        is_scalar_matrix_ = false;
    }
}

// --- Constructors ---

template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT()
    : rows_(0), cols_(0), p_(0), // Or some default prime, e.g., 2. Or throw if p needs to be valid.
    is_scalar_matrix_(true), scalar_value_(0),
    row_echelon_form_cache_(), pivot_columns_cache_() {
    // p_ = 2; // Example: default to a small prime if a 0x0 matrix needs a prime.
    // Or, perhaps a 0x0 matrix doesn't strictly need a prime until used in an operation.
    // If p_ must always be a valid prime, then a default constructor might not be suitable
    // unless you have a global default prime or are okay with a temporary invalid state.
}

template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime)
    : rows_(num_rows), cols_(num_cols), p_(prime),
    flat_data_(), // Empty by default
    is_scalar_matrix_(true),
    scalar_value_(0),       // Represents a zero matrix
    row_echelon_form_cache_(), pivot_columns_cache_() {}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::FromCanonical(uint32_t num_rows, uint32_t num_cols, Elt prime, std::vector<Elt>&& canonical_data) {
    const size_t expected_size = static_cast<size_t>(num_rows) * num_cols;
    if (canonical_data.size() != expected_size) throw std::invalid_argument("FromCanonical: data size does not match matrix dimensions.");
    FlatMatrixT out(num_rows, num_cols, prime);
    if (expected_size == 0) return out;
    out.flat_data_ = std::move(canonical_data);
    out.is_scalar_matrix_ = false;
    out.scalar_value_ = 0;
    out._determine_and_set_flags_from_data();
    return out;
}

template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT(uint32_t num_rows, uint32_t num_cols, Elt prime, const std::vector<Elt>& init_flat_data)
    : rows_(num_rows), cols_(num_cols), p_(prime),
    flat_data_(), // Will be set
    is_scalar_matrix_(false), // To be determined by data
    scalar_value_(0),
    row_echelon_form_cache_(), pivot_columns_cache_() {

    size_t expected_size = (static_cast<size_t>(rows_) * cols_);
    if (init_flat_data.size() != expected_size) {
        if (expected_size == 0 && init_flat_data.empty()) { // 0-dim matrix, empty data is fine
            is_scalar_matrix_ = true; // It's scalar 0
            scalar_value_ = 0;
            return;
        }
        throw std::invalid_argument("Initial data size does not match matrix dimensions.");
    }

    if (expected_size > 0) {
        flat_data_ = init_flat_data;
        for (Elt& x : flat_data_) {
            x = ModArithmetic::mod_add_w(x, Elt(0), p_); // Ensure data is mod p
        }
        _determine_and_set_flags_from_data(); // Sets scalar flags based on data
    }
    else { // 0-dim matrix
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
    }
}

template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT(uint32_t dim, Elt prime) // Identity constructor
    : rows_(dim), cols_(dim), p_(prime),
    flat_data_(), // Empty by default for scalar
    is_scalar_matrix_(true),
    scalar_value_((dim == 0) ? 0 : 1), // If dim=0, it's scalar 0. Else scalar 1.
    row_echelon_form_cache_(), pivot_columns_cache_() {}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::CreateZero(uint32_t num_rows, uint32_t num_cols, Elt prime) {
    return FlatMatrixT<Elt>(num_rows, num_cols, prime); // Default constructor is scalar 0
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::CreateScalar(uint32_t dim, Elt scalar_val, Elt prime) {
    FlatMatrixT<Elt> fm(dim, dim, prime); // This creates a scalar 0 matrix of dim x dim
    fm.is_scalar_matrix_ = true; // Re-affirm, though already true
    fm.scalar_value_ = ModArithmetic::mod_add_w(scalar_val, Elt(0), prime);
    // flat_data_ remains empty
    return fm;
}

template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT(const MatrixT<Elt>& other_matrix)
    : rows_(other_matrix.rows), cols_(other_matrix.cols), p_(other_matrix.p),
    flat_data_(),
    is_scalar_matrix_(false), // To be determined
    scalar_value_(0),
    row_echelon_form_cache_(), pivot_columns_cache_() {
    if (rows_ == 0 || cols_ == 0) {
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
    }
    else {
        flat_data_.resize((static_cast<size_t>(rows_) * cols_));
        for (uint32_t i = 0; i < rows_; ++i) {
            for (uint32_t j = 0; j < cols_; ++j) {
                flat_data_[static_cast<size_t>(i) * cols_ + j] = other_matrix.data[i][j];
            }
        }
        _determine_and_set_flags_from_data(); // Sets flags based on copied data
    }
}

// Destructor
template<class Elt>
FlatMatrixT<Elt>::~FlatMatrixT() {
    _clear_cached_rref();
}

// Copy constructor
template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT(const FlatMatrixT<Elt>& other)
    : rows_(other.rows_), cols_(other.cols_), p_(other.p_),
    flat_data_(other.flat_data_), // Vector copy
    is_scalar_matrix_(other.is_scalar_matrix_),
    scalar_value_(other.scalar_value_),
    row_echelon_form_cache_(), // Cache is not copied
    pivot_columns_cache_() {}

// Copy assignment operator
template<class Elt>
FlatMatrixT<Elt>& FlatMatrixT<Elt>::operator=(const FlatMatrixT<Elt>& other) {
    if (this != &other) {
        _clear_cached_rref(); // Clear existing cache of LHS

        rows_ = other.rows_;
        cols_ = other.cols_;
        p_ = other.p_;
        flat_data_ = other.flat_data_;
        is_scalar_matrix_ = other.is_scalar_matrix_;
        scalar_value_ = other.scalar_value_;
        // Caches remain nullptr, will be recomputed on demand for LHS
    }
    return *this;
}

// Move constructor
template<class Elt>
FlatMatrixT<Elt>::FlatMatrixT(FlatMatrixT<Elt>&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), p_(other.p_),
    flat_data_(std::move(other.flat_data_)),
    is_scalar_matrix_(other.is_scalar_matrix_),
    scalar_value_(other.scalar_value_),
    row_echelon_form_cache_(std::move(other.row_echelon_form_cache_)), // Pilfer cache
    pivot_columns_cache_(std::move(other.pivot_columns_cache_)) {
    other.rows_ = 0; other.cols_ = 0;
    other.is_scalar_matrix_ = true; other.scalar_value_ = 0; // Valid moved-from state (scalar 0, 0x0)
}

// Move assignment operator
template<class Elt>
FlatMatrixT<Elt>& FlatMatrixT<Elt>::operator=(FlatMatrixT<Elt>&& other) noexcept {
    if (this != &other) {
        _clear_cached_rref(); // Clear existing cache of LHS

        rows_ = other.rows_;
        cols_ = other.cols_;
        p_ = other.p_;
        flat_data_ = std::move(other.flat_data_);
        is_scalar_matrix_ = other.is_scalar_matrix_;
        scalar_value_ = other.scalar_value_;
        row_echelon_form_cache_ = std::move(other.row_echelon_form_cache_); // Pilfer cache
        pivot_columns_cache_ = std::move(other.pivot_columns_cache_);

        other.rows_ = 0; other.cols_ = 0;
        other.is_scalar_matrix_ = true; other.scalar_value_ = 0;
    }
    return *this;
}

template<class Elt>
void FlatMatrixT<Elt>::swap(FlatMatrixT<Elt>& other) {
    using std::swap;
    swap(rows_, other.rows_);
    swap(cols_, other.cols_);
    swap(p_, other.p_);
    swap(flat_data_, other.flat_data_);
    swap(is_scalar_matrix_, other.is_scalar_matrix_);
    swap(scalar_value_, other.scalar_value_);
    swap(row_echelon_form_cache_, other.row_echelon_form_cache_);
    swap(pivot_columns_cache_, other.pivot_columns_cache_);
}

// --- Element Access and Data Population ---
template<class Elt>
bool FlatMatrixT<Elt>::is_zero() const {
    if (is_scalar_matrix_) {
        return scalar_value_ == 0;
    }

    // General matrix (is_scalar_matrix_ is false). Data must be populated.
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24) ensure general data exists

    if (flat_data_.empty()) { // True for 0-dim matrices or if _ensure... didn't populate
        if (static_cast<size_t>(rows_) * cols_ == 0) return true; // 0-dim is zero
        // This case (general, non-0-dim, empty data) should ideally not be hit if logic is sound.
        // If it is, treat as error or assume zero and fix flags.
        // For robustness, let's assume it means it should be zero.
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
        _clear_cached_rref();
        return true;
    }

    // Check all elements in flat_data_
    // std::all_of is from <algorithm>
    bool all_elements_are_zero = std::all_of(flat_data_.begin(), flat_data_.end(), [](Elt x) { return x == 0; });

    if (all_elements_are_zero) {
        // All elements in flat_data_ are zero. Update flags to scalar zero.
        is_scalar_matrix_ = true;
        scalar_value_ = 0;
        if (!flat_data_.empty() && (rows_ > 0 && cols_ > 0)) {
            // Optional: clear flat_data_ to save memory for scalar zero state,
            // but populate_flat_data() or set_entry() would repopulate if needed.
            // flat_data_.clear();
            // flat_data_.shrink_to_fit();
        }
        _clear_cached_rref(); // State changed (became scalar zero)
        return true;
    }

    return false; // General matrix with at least one non-zero element
}

template<class Elt>
Elt FlatMatrixT<Elt>::get_entry(uint32_t r, uint32_t c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix indices out of range in get_entry.");
    }
    if (is_scalar_matrix_) {
        return (r == c) ? scalar_value_ : 0;
    }
    // General matrix: flat_data_ must be populated
    if (flat_data_.empty() && (static_cast<size_t>(rows_) * cols_ > 0)) {
        throw std::logic_error("get_entry: General matrix has unpopulated flat_data_.");
    }
    return flat_data_[static_cast<size_t>(r) * cols_ + c];
}

template<class Elt>
void FlatMatrixT<Elt>::set_entry(uint32_t r, uint32_t c, Elt value) {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix indices out of range in set_entry.");
    }

    Elt val_mod_p = ModArithmetic::mod_add_w(value, Elt(0), p_);

    // If it's currently scalar, and this set_entry maintains that scalar property or makes it zero
    if (is_scalar_matrix_) {
        if (r == c && val_mod_p == scalar_value_) return; // No change to diagonal scalar value
        if (r != c && val_mod_p == 0) return; // No change to off-diagonal zero
    }

    // If we reach here, the matrix will likely become general, or its data needs explicit update.
    populate_flat_data(); // Ensure flat_data_ exists and reflects current state
                          // If it was scalar, it's now general with populated data.

    flat_data_[static_cast<size_t>(r) * cols_ + c] = val_mod_p;
    is_scalar_matrix_ = false; // Assume it becomes general after an arbitrary set_entry.
                               // This is a safe default. _determine_and_set_flags will correct it.
    _determine_and_set_flags_from_data(); // Re-evaluate if it became scalar (incl. scalar 0)
}

template<class Elt>
void FlatMatrixT<Elt>::populate_flat_data() const {
    // If it's general and already populated, do nothing.
    if (!is_scalar_matrix_ && !flat_data_.empty() && flat_data_.size() == (static_cast<size_t>(rows_) * cols_)) {
        return;
    }

    // Ensure flat_data_ has correct size, initialized to 0.
    // This also handles case where it was scalar and flat_data_ was empty.
    // (slot-14 WP-3, G1) every index product in this file is size_t since
    // WP-3; the one remaining limit is the storage's own size_type.
    ENGINE_CHECK_MSG(static_cast<size_t>(rows_) * cols_ <= flat_data_.max_size(),
                     "FlatMatrix: rows*cols exceeds the storage's size_type");
    flat_data_.assign((static_cast<size_t>(rows_) * cols_), 0);

    if (is_scalar_matrix_ && scalar_value_ != 0) {
        // Scalar_value_ != 0 implies it should be square if _determine_and_set_flags worked.
        if (rows_ != cols_ && (rows_ > 0 || cols_ > 0)) { // (rows_ > 0 || cols_ > 0) means not 0x0
            // This is an inconsistent state: non-zero scalar implies square.
            // For safety, make it general. This shouldn't be hit if logic is sound.
            // is_scalar_matrix_ = false; // Mark as general, it's now just some data.
            // The above assign already filled it with zeros.
            // This state should be caught by _determine_and_set_flags_from_data.
            // For now, if it's marked scalar non-zero, proceed as if square.
            // This might indicate a flaw if a non-square matrix is flagged as scalar non-zero.
            // The _determine_and_set_flags logic should prevent this.
        }
        for (uint32_t i = 0; i < std::min(rows_, cols_); ++i) { // Iterate up to min dim for diagonal
            flat_data_[static_cast<size_t>(i) * cols_ + i] = scalar_value_;
        }
    }
    // If is_scalar_matrix_ is true and scalar_value_ == 0, flat_data_ is already all zeros.
    // If is_scalar_matrix_ was false (general) and flat_data_ was empty, it's now populated with zeros.
    // The caller might need to call _determine_and_set_flags_from_data if this implicit zeroing
    // of a previously general (but empty data) matrix changes its nature.
    // However, a general matrix should not have empty data if non-0-dim.
}

// In FlatMatrix.cpp (definition)
template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::extract_sub_matrix_rows(uint32_t row_start_idx, uint32_t num_rows_to_extract) const {
    if (row_start_idx > rows_) {
        throw std::out_of_range("row_start_idx cannot be greater than the number of rows in extract_sub_matrix_rows.");
    }
    if (row_start_idx + num_rows_to_extract > rows_) {
        throw std::out_of_range("Extraction window [row_start_idx, row_start_idx + num_rows_to_extract) exceeds original row dimensions.");
    }

    if (num_rows_to_extract == 0) {
        return FlatMatrixT<Elt>::CreateZero(0, cols_, p_);
    }

    uint32_t new_rows = num_rows_to_extract;

    if (is_scalar_matrix_) {
        if (scalar_value_ == 0) {
            // Original is scalar 0. Extracted part is also scalar 0.
            return FlatMatrixT<Elt>::CreateZero(new_rows, cols_, p_);
        }
        else { // Original is s*I (scalar_value_ != 0), and rows_ == cols_ by invariant.
               // The extracted sub-matrix will be (new_rows x original_cols_).
               // Original_cols_ is this->cols_ (which is == this->rows_).

            // Check if the extracted part can also be represented as s*I_new
            // This happens if:
            // 1. The extracted matrix is square: new_rows == this->cols_
            // 2. The extraction starts at original row 0: row_start_idx == 0
            //    (This ensures the diagonal of s*I aligns with the top-left of the new matrix)
            if (new_rows == this->cols_ && row_start_idx == 0) {
                // Result is s * Identity_new_rows_x_new_rows
                return FlatMatrixT<Elt>::CreateScalar(new_rows, scalar_value_, p_);
            }
            else {
                // Result is a general matrix (a rectangular slice of s*I, or an off-center square slice).
                FlatMatrixT<Elt> result(new_rows, this->cols_, p_); // Initially scalar 0.
                // We need to populate its data.
                result.populate_flat_data(); // Makes it general, 0-filled.

                for (uint32_t r_n = 0; r_n < new_rows; ++r_n) { // r_n is row in the new (result) matrix
                    uint32_t r_o = row_start_idx + r_n; // Corresponding row in the original this->matrix
                    // The entry this->get_entry(r_o, c_o) is scalar_value_ if r_o == c_o, else 0.
                    // We want to set result.flat_data_[static_cast<size_t>(r_n) * this->cols_ + c_o].
                    // We only need to set non-zero entries.
                    // A non-zero entry occurs if original (r_o, c_o) was on the diagonal (r_o == c_o)
                    // and this c_o is within the column bounds of the original matrix.
                    uint32_t c_o_for_diag = r_o; // For s*I, non-zero is at (r_o, r_o)
                    if (c_o_for_diag < this->cols_) { // If original column r_o is valid
                        result.flat_data_[static_cast<size_t>(r_n) * this->cols_ + c_o_for_diag] = scalar_value_;
                    }
                }
                // No need to call _determine_and_set_flags_from_data here because we know it's
                // general (unless the only case it could be scalar was caught above).
                // It's already flagged as general because populate_flat_data() was called
                // on a matrix that wasn't already s*I of the *new* dimensions.
                // The constructor `FlatMatrix(new_rows, this->cols_, p_)` made it scalar 0.
                // `populate_flat_data()` made its `flat_data_` all zeros.
                // Then we set specific entries. So `is_scalar_matrix_` should be false.
                // To be absolutely safe or if `populate_flat_data` logic changes:
                result.is_scalar_matrix_ = false; // Explicitly mark as general after populating.
                                                  // _determine_and_set_flags_from_data() would be more robust
                                                  // if there's any doubt, but slower.
                                                  // Given the logic, it should be general here.
                return result;
            }
        }
    }

    // General matrix case (original was general)
    // flat_data_ should be populated for the original matrix.
    // (Ensured by previous check or this would be an inconsistent state for a general matrix)
    // General matrix case (original was general)
    if (flat_data_.empty() && (static_cast<size_t>(rows_) * cols_ > 0)) { // Should be populated
        throw std::logic_error("extract_sub_matrix_rows: General matrix has unpopulated flat_data_ (this should not happen).");
    }

    // First, check if the segment to be extracted is all zeros.
    // This avoids allocating and copying new_data if the result is just a zero matrix.
    bool all_extracted_entries_are_zero = true;
    if (num_rows_to_extract > 0) { // Only check if we are actually extracting something
        for (uint32_t r_offset = 0; r_offset < num_rows_to_extract; ++r_offset) {
            uint32_t original_r = row_start_idx + r_offset;
            size_t row_mem_start = static_cast<size_t>(original_r) * this->cols_;
            for (uint32_t c = 0; c < this->cols_; ++c) {
                if (this->flat_data_[row_mem_start + c] != 0) {
                    all_extracted_entries_are_zero = false;
                    goto end_zero_check; // Exit both loops once a non-zero is found
                }
            }
        }
    end_zero_check:;
    }
    else { // num_rows_to_extract is 0, which should have been caught earlier.
          // If somehow reached, it's trivially all zero for an empty extraction.
        all_extracted_entries_are_zero = true;
    }


    if (all_extracted_entries_are_zero) {
        return FlatMatrixT<Elt>::CreateZero(num_rows_to_extract, this->cols_, p_);
    }
    else {
        // The segment is not all zeros, so we need to construct the data.
        std::vector<Elt> new_data;
        new_data.resize(static_cast<size_t>(num_rows_to_extract) * this->cols_); // Pre-allocate

        size_t destination_idx = 0;
        for (uint32_t r_offset = 0; r_offset < num_rows_to_extract; ++r_offset) {
            uint32_t original_r = row_start_idx + r_offset;
            size_t source_row_start_in_flat_data = static_cast<size_t>(original_r) * this->cols_;

            // Efficiently copy the entire row
            std::copy(
                this->flat_data_.begin() + source_row_start_in_flat_data,
                this->flat_data_.begin() + source_row_start_in_flat_data + this->cols_,
                new_data.begin() + destination_idx
            );
            destination_idx += this->cols_;
        }

        // Construct from new_data. The constructor will call _determine_and_set_flags_from_data().
        // If FlatMatrix had a constructor that takes std::vector<uint32_t>&&, we could std::move(new_data).
        // Assuming current constructor takes const&:
        return FlatMatrixT<Elt>(num_rows_to_extract, this->cols_, p_, new_data);
    }
}

// --- Conversion and Output --- (Largely similar, use get_entry)

template<class Elt>
MatrixT<Elt> FlatMatrixT<Elt>::to_matrix() const {
    std::vector<std::vector<Elt>> matrix_data(rows_, std::vector<Elt>(cols_));
    for (uint32_t i = 0; i < rows_; ++i) {
        for (uint32_t j = 0; j < cols_; ++j) {
            matrix_data[i][j] = get_entry(i, j);
        }
    }
    return MatrixT<Elt>(rows_, cols_, p_, matrix_data);
}

template<class Elt>
std::string FlatMatrixT<Elt>::to_string(void) const {
    return to_string_offset(0);
}

template<class Elt>
std::string FlatMatrixT<Elt>::to_string_offset(const uint32_t offset) const {
    std::ostringstream oss;
    if (is_scalar_matrix_) {
        if (scalar_value_ == 0) {
            oss << offset_string(offset) << "Zero Matrix (Scalar 0) (" << rows_ << "x" << cols_ << ")\n";
        }
        else {
            oss << offset_string(offset) << "Scalar Matrix (" << rows_ << "x" << cols_ << "), scalar = " << scalar_value_ << "\n";
        }
    }
    else {
        oss << offset_string(offset) << "General Matrix (" << rows_ << "x" << cols_ << ")\n";
    }

    if (rows_ > 0 && cols_ > 0) {
        for (uint32_t i = 0; i < rows_; ++i) {
            oss << offset_string(offset);
            oss << "[";
            for (uint32_t j = 0; j < cols_; ++j) {
                oss << std::setw(3) << get_entry(i, j);
                if (j < cols_ - 1) oss << " ";
            }
            oss << " ]\n";
        }
    }
    return oss.str();
}

template<class Elt>
std::string FlatMatrixT<Elt>::to_string_transposed(void) const {
    return to_string_transposed_offset(0);
}

template<class Elt>
std::string FlatMatrixT<Elt>::to_string_transposed_offset(const uint32_t offset) const {
    std::ostringstream oss;
    if (is_scalar_matrix_) {
        if (scalar_value_ == 0) { // Zero matrix
            oss << offset_string(offset) << "Zero Matrix (Scalar 0) (transposed " << cols_ << "x" << rows_ << ")\n";
        }
        else { // Non-zero scalar matrix (implies square for this flag)
            oss << offset_string(offset) << "Scalar Matrix (transposed " << cols_ << "x" << rows_ << "), scalar = " << scalar_value_ << "\n";
        }
    }
    else {
        oss << offset_string(offset) << "General Matrix (transposed " << cols_ << "x" << rows_ << ")\n";
    }

    if (cols_ > 0 && rows_ > 0) { // Dimensions for iteration are swapped
        for (uint32_t j = 0; j < cols_; ++j) { // Iterate original columns (new rows)
            oss << offset_string(offset);
            oss << "[";
            for (uint32_t i = 0; i < rows_; ++i) { // Iterate original rows (new columns)
                oss << std::setw(3) << get_entry(i, j); // get_entry(original_row, original_col)
                if (i < rows_ - 1) oss << " ";
            }
            oss << ((j == 0 && (rows_ > 0 || cols_ > 0)) ? " ]^T\n" : " ]\n");
        }
    }
    return oss.str();
}

// --- Operations ---

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::transpose() const {
    if (is_scalar_matrix_) {
        if (scalar_value_ == 0) { // Zero matrix (scalar 0)
            return FlatMatrixT<Elt>::CreateZero(cols_, rows_, p_); // Dimensions swap
        }
        else { // Non-zero scalar matrix (alpha*I, implies square by _d_a_s_f_f_d)
         // if (rows_ != cols_) { /* This should be an error state if scalar_value_ != 0 */ }
            return FlatMatrixT<Elt>::CreateScalar(rows_, scalar_value_, p_); // Dimensions (rows_) don't change
        }
    }

    // General matrix
    FlatMatrixT<Elt> result(cols_, rows_, p_); // Default constructor makes it scalar 0
    result.populate_flat_data(); // Now it's general, data filled with 0s.
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)

    // (matrix-engine Phase 2, Stage B) engine::kernels::transpose is the same
    // permutation of the same bytes, walked in 16x16 tiles instead of with a
    // stride-cols_ scatter, so it touches each cache line once instead of
    // once per element. No arithmetic is involved at all here, hence nothing
    // to reassociate: the two routes move identical values.
    const bool use_engine = engine_path<Elt>(p_) && rows_ > 0 && cols_ > 0;
    if (use_engine) {
        const engine::DenseBuf<Elt> src = dense_wrap<Elt>(flat_data_.data(), rows_, cols_);
        engine::DenseBuf<Elt> dst = dense_wrap<Elt>(result.flat_data_.data(), cols_, rows_);
        engine::kernels::transpose<Elt>(engine::cview(src), engine::view(dst));
    }
#ifdef ENGINE_SHADOW_LEGACY
    std::vector<Elt> shadow_engine = shadow_park<Elt>(use_engine, result.flat_data_, static_cast<size_t>(cols_) * rows_);
    const bool run_legacy = true;
#else
    const bool run_legacy = !use_engine;
#endif
    if (run_legacy) {
        for (uint32_t i = 0; i < rows_; ++i) {
            for (uint32_t j = 0; j < cols_; ++j) {
                result.flat_data_[static_cast<size_t>(j) * result.cols_ + i] = flat_data_[static_cast<size_t>(i) * cols_ + j];
            }
        }
    }
#ifdef ENGINE_SHADOW_LEGACY
    shadow_restore<Elt>(use_engine, "FlatMatrix::transpose", shadow_engine, result.flat_data_);
#endif
    result._determine_and_set_flags_from_data(); // Check if transpose became special
    return result;
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::operator+(const FlatMatrixT<Elt>& other) const {

    if (p_ != other.p_) {
        std::ostringstream oss;
        oss << "FlatMatrix::operator+: Prime moduli mismatch for operation. Left operand p=" << p_
            << ", Right operand p=" << other.p_ << ".";
        throw std::invalid_argument(oss.str());
    }

    if ((rows_ != other.rows_) || (cols_ != other.cols_)) {
        std::ostringstream oss;
        oss << "FlatMatrix::operator+: Dimensions mismatch for operation. Left operand is " << rows_
            << "x" << cols_ << " and right operand is " << other.rows_ << "x" << other.cols_ << ".";
        throw std::invalid_argument(oss.str());
    }

    //if (rows_ != other.rows_ || cols_ != other.cols_ || p_ != other.p_) {
    //    throw std::invalid_argument("Matrix dimensions or prime moduli do not match for addition.");
    //}

    if (is_scalar_matrix_ && other.is_scalar_matrix_) {
        Elt new_scalar_val = ModArithmetic::mod_add_w(scalar_value_, other.scalar_value_, p_);
        // IMPORTANT: The resulting matrix must have the *original dimensions* if it's scalar zero,
        // or if it's scalar non-zero, the original dimensions must have been square.
        if (new_scalar_val == 0) {
            return FlatMatrixT<Elt>::CreateZero(rows_, cols_, p_); // Use original (matching) rows_ and cols_
        }
        else {
            // If sum is non-zero scalar, originals must have been square and compatible
            // or one was zero (potentially non-square) and the other non-zero square, making sum non-square.
            // This case needs care. If rows_ != cols_ here, a non-zero scalar is problematic for a non-square matrix.
            if (rows_ != cols_) {
                // This case implies an attempt to create a non-zero scalar value on a non-square matrix.
                // This should degenerate to a general matrix addition.
                // For simplicity, let the general path handle it if resulting scalar is non-zero AND matrix non-square.
                // However, current logic would likely hit this if one is 1x0 (scalar 0) and other is 1x1 (scalar X!=0)
                // which would have failed the initial dimension check.
                // So, if we are here, rows_ == other.rows_ and cols_ == other.cols_.
                // If new_scalar_val != 0, it must be that rows_ == cols_.
                std::ostringstream oss;
                oss << "FlatMatrix::operator+: Logic error - attempting to create non-zero scalar " << new_scalar_val
                    << " for potentially non-square matrix " << rows_ << "x" << cols_ << " from scalar addition.";
                // This situation (non-zero scalar for non-square) should ideally not be formed via this path
                // if is_scalar_matrix_ for non-zero value implies square.
                // If truly non-square but sum is non-zero scalar (e.g. 0 + X, X!=0 on a non-square matrix if allowed)
                // then result should be general.
                // For now, assume if new_scalar_val !=0, rows_ == cols_ must hold.
                if (rows_ != cols_) throw std::logic_error(oss.str());
                return FlatMatrixT<Elt>::CreateScalar(rows_, new_scalar_val, p_); // rows_ is the dimension
            }
        }
    }

    // At least one is general, or one is scalar non-zero and other is general.
    // (Handled scalar 0 + B = B and A + scalar 0 = A by get_entry if one is scalar 0)
    FlatMatrixT<Elt> result(rows_, cols_, p_);
    // Result is initially scalar 0. populate_flat_data will make it general 0-filled.
    result.populate_flat_data();

    // (matrix-engine Phase 2, Stage B) When BOTH operands really own a buffer,
    // the sum is result = A followed by engine::kernels::axpy(alpha = 1, B):
    // a straight pointer walk instead of rows*cols get_entry() calls, each of
    // which re-tested the bounds, the scalar flag and the emptiness of
    // flat_data_. If either side is scalar-stored (alpha*I with no buffer) the
    // legacy get_entry loop below is kept verbatim -- materialising the scalar
    // just to add it would throw away FlatMatrix's storage optimisation.
    const bool both_general = !is_scalar_matrix_ && !other.is_scalar_matrix_;
    const bool use_engine = both_general && engine_path<Elt>(p_) && rows_ > 0 && cols_ > 0;
    if (use_engine) {
        result.flat_data_ = this->flat_data_;   // canonical entries, copied verbatim
        const engine::DenseBuf<Elt> X = dense_wrap<Elt>(other.flat_data_.data(), rows_, cols_);
        engine::DenseBuf<Elt> Y = dense_wrap<Elt>(result.flat_data_.data(), rows_, cols_);
        engine::kernels::axpy<Elt>(engine_ctx<Elt>(p_), Elt(1), engine::cview(X),
                                   engine::view(Y), /*subtract=*/false);
    }
#ifdef ENGINE_SHADOW_LEGACY
    std::vector<Elt> shadow_engine = shadow_park<Elt>(use_engine, result.flat_data_, static_cast<size_t>(rows_) * cols_);
    const bool run_legacy = true;
#else
    const bool run_legacy = !use_engine;
#endif
    if (run_legacy) {
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                result.flat_data_[static_cast<size_t>(r) * result.cols_ + c] = ModArithmetic::mod_add_w( // Use result.cols_
                    this->get_entry(r, c),
                    other.get_entry(r, c),
                    p_
                );
            }
        }
    }
#ifdef ENGINE_SHADOW_LEGACY
    shadow_restore<Elt>(use_engine, "FlatMatrix::operator+", shadow_engine, result.flat_data_);
#endif
    result._determine_and_set_flags_from_data();
    return result;
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::operator-(const FlatMatrixT<Elt>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_ || p_ != other.p_) {
        throw std::invalid_argument("Matrix dimensions or prime moduli do not match for subtraction.");
    }

    // (Phase 1, defect #18) A - B used to build a full negated temporary and then
    // call operator+, i.e. one extra allocation, one extra flag rescan and one
    // extra pass over the data per subtraction -- and subtraction is what the
    // block elimination does on every (pivot, target) block pair.  The two
    // branches below are the two branches the old composition could reach, with
    // the arithmetic spelled out identically as mod_add_w(a, mod_minus_w(b)).
    //
    // Note on which branch is reachable: negation maps alpha*I to (-alpha)*I and
    // nothing else, so if `other` is flagged general its negation is general too
    // -- the old code could therefore never take operator+'s scalar/scalar path
    // unless `other` was already scalar, which is exactly the first branch here.
    if (is_scalar_matrix_ && other.is_scalar_matrix_) {
        Elt neg = ModArithmetic::mod_minus_w(other.scalar_value_, other.p_);
        Elt new_scalar_val = ModArithmetic::mod_add_w(scalar_value_, neg, p_);
        if (new_scalar_val == 0) {
            return FlatMatrixT<Elt>::CreateZero(rows_, cols_, p_);
        }
        if (rows_ != cols_) {
            std::ostringstream oss;
            oss << "FlatMatrix::operator-: Logic error - attempting to create non-zero scalar " << new_scalar_val
                << " for potentially non-square matrix " << rows_ << "x" << cols_ << " from scalar subtraction.";
            throw std::logic_error(oss.str());
        }
        return FlatMatrixT<Elt>::CreateScalar(rows_, new_scalar_val, p_);
    }

    FlatMatrixT<Elt> result(rows_, cols_, p_);
    result.populate_flat_data();

    // (matrix-engine Phase 2, Stage B) Same retrofit as operator+, with
    // axpy's `subtract` flag. The legacy expression mod_add(a, mod_minus(b))
    // and the kernel's ctx.sub(a, b) are the two spellings of the unique
    // canonical residue of a - b in [0, p); they agree entry for entry.
    const bool both_general = !is_scalar_matrix_ && !other.is_scalar_matrix_;
    const bool use_engine = both_general && engine_path<Elt>(p_) && rows_ > 0 && cols_ > 0;
    if (use_engine) {
        result.flat_data_ = this->flat_data_;
        const engine::DenseBuf<Elt> X = dense_wrap<Elt>(other.flat_data_.data(), rows_, cols_);
        engine::DenseBuf<Elt> Y = dense_wrap<Elt>(result.flat_data_.data(), rows_, cols_);
        engine::kernels::axpy<Elt>(engine_ctx<Elt>(p_), Elt(1), engine::cview(X),
                                   engine::view(Y), /*subtract=*/true);
    }
#ifdef ENGINE_SHADOW_LEGACY
    std::vector<Elt> shadow_engine = shadow_park<Elt>(use_engine, result.flat_data_, static_cast<size_t>(rows_) * cols_);
    const bool run_legacy = true;
#else
    const bool run_legacy = !use_engine;
#endif
    if (run_legacy) {
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                result.flat_data_[static_cast<size_t>(r) * result.cols_ + c] = ModArithmetic::mod_add_w(
                    this->get_entry(r, c),
                    ModArithmetic::mod_minus_w(other.get_entry(r, c), p_),
                    p_);
            }
        }
    }
#ifdef ENGINE_SHADOW_LEGACY
    shadow_restore<Elt>(use_engine, "FlatMatrix::operator-", shadow_engine, result.flat_data_);
#endif
    result._determine_and_set_flags_from_data();
    return result;
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::operator*(const FlatMatrixT<Elt>& other) const {
    if (cols_ != other.rows_ || p_ != other.p_) {
        throw std::invalid_argument("Matrix dimensions or prime moduli do not match for multiplication.");
    }

    // Handle scalar cases, especially for zero dimensions
    if (is_scalar_matrix_ && other.is_scalar_matrix_) {
        Elt res_scalar = ModArithmetic::mod_mul_w(scalar_value_, other.scalar_value_, p_);
        // Result is (this->rows_ x other.cols_).
        // If res_scalar is 0, it's a zero matrix of that size.
        if (res_scalar == 0) {
            return FlatMatrixT<Elt>::CreateZero(rows_, other.cols_, p_);
        }
        else {
            // If res_scalar != 0, then scalar_value_ != 0 and other.scalar_value_ != 0.
            // For this->is_scalar_matrix_ (non-zero scalar) to be true, this must be square (rows_ == cols_).
            // For other.is_scalar_matrix_ (non-zero scalar) to be true, other must be square (other.rows_ == other.cols_).
            // Since cols_ == other.rows_, all dimensions are equal: rows_ == cols_ == other.rows_ == other.cols_.
            // So the result is also square of dimension rows_ (or other.cols_).
            return FlatMatrixT<Elt>::CreateScalar(rows_, res_scalar, p_);
        }
    }

    // If one is scalar (and not both, handled above).
    // Result dimensions are (this->rows_ x other.cols_).
    if (is_scalar_matrix_) { // 'this' is scalar, 'other' is general or scalar (but previous block handles scalar-scalar)
                            // So 'other' must be effectively general here.
        if (scalar_value_ == 0) { // 0 * B = 0 (of size rows_ x other.cols_)
            return FlatMatrixT<Elt>::CreateZero(rows_, other.cols_, p_);
        }
        // s * B where s = scalar_value_
        return other * scalar_value_; // scalar multiplication
    }

    if (other.is_scalar_matrix_) { // 'other' is scalar, 'this' is general
        if (other.scalar_value_ == 0) { // A * 0 = 0 (of size rows_ x other.cols_)
            return FlatMatrixT<Elt>::CreateZero(rows_, other.cols_, p_);
        }
        // A * s where s = other.scalar_value_
        return *this * other.scalar_value_; // scalar multiplication
    }

    // Both are general (not scalar)
    FlatMatrixT<Elt> result(rows_, other.cols_, p_); // Starts scalar 0
    if (rows_ == 0 || other.cols_ == 0) { // If result is 0-dimensional (e.g. 0xM or Nx0)
        return result; // Which is already scalar 0 of the correct 0-dim size.
    }
    // If cols_ (inner dimension) is 0, the sum loop won't run, result remains scalar 0. Correct.
    // Example: (N x 0) * (0 x M) -> result (N x M) scalar 0.

    result.populate_flat_data(); // Becomes general, 0-filled

    // These _ensure_data_populated calls are defensive. If general, data should be there.
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)
    if (!other.is_scalar_matrix_) other.populate_flat_data();

    // This check is important for general matrices that might have had 0 as inner dim.
    if (cols_ == 0) { // Inner dimension is 0
        // result is already scalar 0 of correct size rows_ x other.cols_.
        // If populate_flat_data made it general 0-filled, _determine_and_set_flags will fix it.
        result._determine_and_set_flags_from_data(); // Ensure it's scalar 0 if all entries are 0
        return result;
    }

    // (matrix-engine Phase 1) Same products, same modular sum, reassociated into
    // i-k-j order with a zero-skip on A[i][k].  Modular arithmetic over F_p is
    // exact, so summation order is irrelevant to the result and skipping a term
    // that is identically zero changes nothing -- this is bit-identical to the
    // i-j-k loop it replaces, and it reads B row-major instead of column-strided.
    //
    // Why it matters here and not only in Phase 2's GEMM work: the block
    // elimination multiplies the d_V x d_V transform C returned by
    // `get_rref_and_transform_matrix()` into every block of the pivot row, and
    // for the width-1 block columns that dominate `factor_basis_block` that C is
    // an elementary transform with at most two non-zeros per row.  The skip turns
    // those O(m^2 k) products into O(m k).
    //
    // (matrix-engine Phase 2, Stage B) That loop is now the LEGACY route. The
    // engine route hands the same product to engine::kernels::gemm through
    // non-owning DenseViews of the two flat_data_ buffers -- an 8x16
    // outer-product microkernel with an AVX-512 body picked at runtime,
    // k-blocked at ctx.k_max so each accumulator stays exact
    // (ENGINE_API.md Sec. 5/6; 28 Gmodmul/s at n = 128 against the 0.21 this
    // loop reaches after the primitives were inlined).
    //
    // WHAT IS GIVEN UP, AND WHY IT IS THE RIGHT TRADE: gemm does the full
    // O(m n k) work, so the `if (a == 0) continue` zero-skip above is gone.
    // The skip pays only while A's density is below the ratio of the two
    // rates, i.e. below ~1 %; the elementary d_V x d_V transforms the Phase-1
    // comment is about carry ~2/d_V ~ 1.6 % non-zeros at d_V = 129, so the
    // engine is ahead there too, and it is ahead by two orders of magnitude on
    // every dense product in the pipeline.
    const bool use_engine = engine_path<Elt>(p_);
    if (use_engine) {
        const engine::DenseBuf<Elt> A = dense_wrap<Elt>(this->flat_data_.data(), rows_, cols_);
        const engine::DenseBuf<Elt> B = dense_wrap<Elt>(other.flat_data_.data(),
                                                        other.rows_, other.cols_);
        engine::DenseBuf<Elt> C = dense_wrap<Elt>(result.flat_data_.data(), rows_, other.cols_);
        engine::kernels::gemm<Elt>(engine_ctx<Elt>(p_), engine::cview(A), engine::cview(B),
                                   engine::view(C), engine::kernels::GemmOp::Set);
    }
#ifdef ENGINE_SHADOW_LEGACY
    std::vector<Elt> shadow_engine = shadow_park<Elt>(use_engine, result.flat_data_, static_cast<size_t>(rows_) * other.cols_);
    const bool run_legacy = true;
#else
    const bool run_legacy = !use_engine;
#endif
    if (run_legacy) {
        for (uint32_t i = 0; i < rows_; ++i) {
            const Elt* a_row = this->flat_data_.data() + static_cast<size_t>(i) * this->cols_;
            Elt* c_row = result.flat_data_.data() + static_cast<size_t>(i) * result.cols_;
            for (uint32_t k = 0; k < cols_; ++k) { // cols_ == other.rows_
                const Elt a = a_row[k];
                if (a == 0) continue;
                const Elt* b_row = other.flat_data_.data() + static_cast<size_t>(k) * other.cols_;
                for (uint32_t j = 0; j < other.cols_; ++j) {
                    c_row[j] = ModArithmetic::mod_add_w(c_row[j], ModArithmetic::mod_mul_w(a, b_row[j], p_), p_);
                }
            }
        }
    }
#ifdef ENGINE_SHADOW_LEGACY
    shadow_restore<Elt>(use_engine, "FlatMatrix::operator*", shadow_engine, result.flat_data_);
#endif
    result._determine_and_set_flags_from_data();
    return result;
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::operator*(Elt scalar) const {
    Elt s_mod_p = ModArithmetic::mod_add_w(scalar, Elt(0), p_);

    if (is_scalar_matrix_) {
        Elt new_scalar_val = ModArithmetic::mod_mul_w(scalar_value_, s_mod_p, p_);
        // CreateScalar takes square dimension. If original was scalar 0 non-square, this is tricky.
        // But scalar 0 non-square * s = scalar 0 non-square.
        // If scalar_value_ was 0, new_scalar_val is 0. Result is scalar 0 of original dims.
        if (scalar_value_ == 0) { // Handles s_mod_p == 0 too
            return FlatMatrixT<Elt>::CreateZero(rows_, cols_, p_);
        }
        // If scalar_value_ !=0, implies original was square.
        return FlatMatrixT<Elt>::CreateScalar(rows_, new_scalar_val, p_);
    }

    // General matrix
    if (s_mod_p == 0) { // Multiplying general matrix by 0
        return FlatMatrixT<Elt>::CreateZero(rows_, cols_, p_);
    }

    FlatMatrixT<Elt> result(rows_, cols_, p_); // Starts scalar 0
    result.populate_flat_data(); // Becomes general 0-filled
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)

    // (matrix-engine Phase 2, Stage B) engine::kernels::scal: same product
    // s_mod_p * x entry by entry, but through the regime's Shoup form of the
    // (single, reused) scalar instead of one 64-bit division per element.
    const bool use_engine = engine_path<Elt>(p_) && rows_ > 0 && cols_ > 0;
    if (use_engine) {
        result.flat_data_ = flat_data_;
        engine::DenseBuf<Elt> X = dense_wrap<Elt>(result.flat_data_.data(), rows_, cols_);
        engine::kernels::scal<Elt>(engine_ctx<Elt>(p_), s_mod_p, engine::view(X));
    }
#ifdef ENGINE_SHADOW_LEGACY
    std::vector<Elt> shadow_engine = shadow_park<Elt>(use_engine, result.flat_data_, flat_data_.size());
    const bool run_legacy = true;
#else
    const bool run_legacy = !use_engine;
#endif
    if (run_legacy) {
        for (size_t i = 0; i < flat_data_.size(); ++i) {
            result.flat_data_[i] = ModArithmetic::mod_mul_w(flat_data_[i], s_mod_p, p_);
        }
    }
#ifdef ENGINE_SHADOW_LEGACY
    shadow_restore<Elt>(use_engine, "FlatMatrix::operator*(scalar)", shadow_engine, result.flat_data_);
#endif
    result._determine_and_set_flags_from_data();
    return result;
}

template<class Elt>
FlatMatrixT<Elt> operator*(Elt scalar, const FlatMatrixT<Elt>& mat) {
    return mat * scalar;
}

template<class Elt>
bool FlatMatrixT<Elt>::operator==(const FlatMatrixT<Elt>& other) const {
    if (this == &other) return true;
    if (rows_ != other.rows_ || cols_ != other.cols_ || p_ != other.p_) return false;
    if (rows_ == 0 || cols_ == 0) return true; // Both are 0-dim and dims match

    Elt this_scalar_val, other_scalar_val;
    bool this_is_curr_scalar = this->is_currently_scalar(this_scalar_val);
    bool other_is_curr_scalar = other.is_currently_scalar(other_scalar_val);

    if (this_is_curr_scalar && other_is_curr_scalar) {
        return this_scalar_val == other_scalar_val;
    }

    // If one is scalar and the other is not, they are different by this logic
    // unless we proceed to full content comparison.
    if (this_is_curr_scalar != other_is_curr_scalar) {
        // cosnt SAFE: Use get_entry which handles all cases,
        // we do not populate the flat data vectors of scalar matrices
        for (uint32_t r = 0; r < rows_; ++r) {
            for (uint32_t c = 0; c < cols_; ++c) {
                if (this->get_entry(r, c) != other.get_entry(r, c)) {
                    return false;
                }
            }
        }
        return true;
    }

    // If both are general (is_scalar_matrix_ is false for both)
    // Note: _ensure_data_populated_if_empty is problematic in a const method
    // if it tries to modify 'this' or 'other' via const_cast.
    // The get_entry() path would be safe but slow.

    // THIS PATH (direct flat_data_ comparison) IS ONLY SAFE IF BOTH ARE KNOWN TO BE GENERAL
    // AND _ensure_data_populated_if_empty CANNOT BE CALLED SAFELY HERE in a const method.
    // The code will reach here ONLY IF BOTH are general (is_scalar_matrix_ false for both).
    // In that case, their flat_data_ SHOULD be populated.

    // this->_ensure_data_populated_if_empty(); // Problematic in const method
    // other._ensure_data_populated_if_empty(); // Problematic for const other

    if (this->flat_data_.size() != other.flat_data_.size()) { // This shall not happen, therefore it throws an exception
        throw std::logic_error("operator==: flat_data_ size mismatch for general matrices with same dims.");
    }
    return this->flat_data_ == other.flat_data_;

    // return true;
}

template<class Elt>
bool FlatMatrixT<Elt>::operator!=(const FlatMatrixT<Elt>& other) const {
    return !(*this == other);
}

template<class Elt>
uint32_t FlatMatrixT<Elt>::nonzero_entries() const {
    if (this->is_zero()) { // is_zero() might update flags if it was general becoming zero
        return 0;
    }
    // Now, matrix is known to be non-zero.
    if (is_scalar_matrix_) { // And scalar_value_ != 0
        // For s*I (s!=0), non-zero entries are on the diagonal.
        // This implies rows_ == cols_ if _determine_and_set_flags logic is robust.
        return std::min(rows_, cols_); // Number of diagonal entries
    }

    // General non-zero matrix
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)
    uint32_t count = 0;
    for (Elt x : flat_data_) {
        if (x != 0) count++;
    }
    return count;
}

// --- Linear Algebra ---

template<class Elt>
void FlatMatrixT<Elt>::compute_row_echelon_form() const {
    if (row_echelon_form_cache_ && pivot_columns_cache_) return; // Already computed

    // std::cout << "  Computing FLAT row echelon form for " << rows_ << "x" << cols_ << " matrix ..." << std::endl;

    _clear_cached_rref(); // Clear any partial/old cache
    row_echelon_form_cache_ = std::make_unique<std::vector<Elt>>();
    pivot_columns_cache_ = std::make_unique<std::vector<uint32_t>>();

    if (rows_ == 0 || cols_ == 0) { // For 0-dim matrices
        // Flat cache of rows_*cols_ == 0 entries. Pivots empty.
        return;
    }

    if (is_scalar_matrix_) {
        row_echelon_form_cache_->assign(static_cast<size_t>(rows_) * cols_, Elt(0));
        if (scalar_value_ != 0) { // RREF of alpha*I (alpha!=0) is I
            // This implies rows_ == cols_ due to how non-zero scalar flag is set.
            uint32_t num_pivots = std::min(rows_, cols_); // Should be rows_ or cols_ if square
            for (uint32_t i = 0; i < num_pivots; ++i) {
                (*row_echelon_form_cache_)[static_cast<size_t>(i) * cols_ + i] = 1;
                pivot_columns_cache_->push_back(i);
            }
        } // else scalar_value_ == 0 (Zero Matrix), RREF is all zeros, pivots_cache_ is empty. Correct.
        return;
    }

    // General matrix
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)
    if (flat_data_.empty() && (static_cast<size_t>(rows_) * cols_ > 0)) { // Should not happen
        throw std::logic_error("compute_row_echelon_form: general matrix has empty flat_data_");
    }

    // (matrix-engine Phase 2, defect #7) One copy of the data into the flat
    // cache; the engine then eliminates IN that buffer through a non-owning
    // DenseBuf -- no vector<vector> round trip.
#if OC_ENABLE_LEGACY
    if (engine::elim::use_legacy()) { _legacy_row_echelon_form(); return; }   // legacy/FlatMatrixLegacy.inc
#endif
    row_echelon_form_cache_->assign(flat_data_.begin(), flat_data_.end());
    {
        const engine::ModCtx<Elt> ctx(p_);
        engine::DenseBuf<Elt> buf(row_echelon_form_cache_->data(), row_echelon_form_cache_->size(),
                                  rows_, cols_, engine::Layout::RowMajor);
        engine::elim::RrefResult res = engine::elim::rref<Elt>(ctx, engine::view(buf));
        *pivot_columns_cache_ = std::move(res.pivots);
    }
#ifdef ENGINE_SHADOW_LEGACY
    {
        std::vector<std::vector<Elt>> legacy_2d = flat_to_2d<Elt>(flat_data_, rows_, cols_);
        std::vector<uint32_t> legacy_pivots;
        _gaussian_elimination_core(legacy_2d, legacy_pivots, rows_, cols_);
        shadow_check_ref<Elt>(legacy_2d, legacy_pivots, *row_echelon_form_cache_, *pivot_columns_cache_, rows_, cols_);
    }
#endif
}

template<class Elt>
const std::vector<Elt>& FlatMatrixT<Elt>::get_row_echelon_form_data() const {
    compute_row_echelon_form();
    return *row_echelon_form_cache_;
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::get_row_echelon_form_as_flat_matrix() const {
    compute_row_echelon_form(); // Ensure cache is populated

    if (rows_ == 0 || cols_ == 0) {
        return FlatMatrixT<Elt>(rows_, cols_, p_); // Return 0-dim matrix (scalar 0)
    }

    // The cache is already flat row-major (Phase 2); hand it to the DATA ctor.
    ENGINE_CHECK(row_echelon_form_cache_->size() == static_cast<size_t>(rows_) * cols_);
    return FlatMatrixT<Elt>(rows_, cols_, p_, *row_echelon_form_cache_); // This will call _determine_and_set_flags
}

template<class Elt>
const std::vector<uint32_t>& FlatMatrixT<Elt>::get_pivot_columns() const {
    compute_row_echelon_form();
    return *pivot_columns_cache_;
}

template<class Elt>
std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> FlatMatrixT<Elt>::get_rref_and_transform_matrix() const {
    // Special case: Current matrix (*this) is scalar
    Elt current_scalar_val;
    if (this->is_currently_scalar(current_scalar_val)) {
        if (rows_ == 0) { // Handle 0-row matrix (already scalar 0 usually)
            FlatMatrixT<Elt> rref_A = FlatMatrixT<Elt>::CreateZero(0, cols_, p_);
            FlatMatrixT<Elt> C_transform = FlatMatrixT<Elt>::CreateScalar(0, 1, p_); // 0x0 Identity
            return { rref_A, C_transform };
        }
        if (cols_ == 0 && rows_ > 0) { // 0-col but has rows (scalar 0)
            FlatMatrixT<Elt> rref_A = FlatMatrixT<Elt>::CreateZero(rows_, 0, p_);
            FlatMatrixT<Elt> C_transform = FlatMatrixT<Elt>(rows_, p_); // Identity of size rows_ x rows_
            return { rref_A, C_transform };
        }

        if (current_scalar_val == 0) { // Zero matrix (M = 0*I or 0 for non-square)
            FlatMatrixT<Elt> rref_A = FlatMatrixT<Elt>::CreateZero(rows_, cols_, p_);
            FlatMatrixT<Elt> C_transform = FlatMatrixT<Elt>(rows_, p_); // Identity of size rows_ x rows_
                                                           // (C*0 = 0)
            return { rref_A, C_transform };
        }
        else { // Non-zero scalar matrix (M = s*I, s != 0). Must be square.
            if (rows_ != cols_) {
                // This state (non-zero scalar, non-square) should ideally not occur
                // if _determine_and_set_flags_from_data is robust.
                // If it does, fall through to general Gaussian elimination.
            }
            else {
                FlatMatrixT<Elt> rref_A = FlatMatrixT<Elt>(rows_, p_); // Identity matrix (rows_ x rows_)
                Elt s_inv = ModArithmetic::mod_inv_w(current_scalar_val, p_);
                FlatMatrixT<Elt> C_transform = FlatMatrixT<Elt>::CreateScalar(rows_, s_inv, p_); // s_inv * I
                return { rref_A, C_transform };
            }
        }
    }

    // General case
    if (rows_ == 0) {
        FlatMatrixT<Elt> rref_A = FlatMatrixT<Elt>::CreateZero(0, cols_, p_);
        FlatMatrixT<Elt> C_transform = FlatMatrixT<Elt>::CreateScalar(0, 1, p_);
        return { rref_A, C_transform };
    }
    if (cols_ == 0 && rows_ > 0) {
        FlatMatrixT<Elt> rref_A = FlatMatrixT<Elt>::CreateZero(rows_, 0, p_);
        FlatMatrixT<Elt> C_transform = FlatMatrixT<Elt>(rows_, p_);
        return { rref_A, C_transform };
    }

    uint32_t m = rows_;
    uint32_t n = cols_;
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)

#if OC_ENABLE_LEGACY
    if (engine::elim::use_legacy()) return _legacy_rref_and_transform();   // legacy/FlatMatrixLegacy.inc
#endif

    // (matrix-engine Phase 2) Engine route: the SAME definition -- U is the
    // trailing m columns of RREF([A | I_m]) eliminated over all n + m columns
    // (plan D5b; unique, hence bit-identical to the legacy route).
    std::vector<Elt> rref_A_flat_data(static_cast<size_t>(m) * n);
    std::vector<Elt> C_transform_flat_data(static_cast<size_t>(m) * m);
    {
        const engine::ModCtx<Elt> ctx(p_);
        engine::DenseBuf<Elt> A(m, n);
        for (uint32_t r = 0; r < m; ++r)
            for (uint32_t c_a = 0; c_a < n; ++c_a) A.at(r, c_a) = this->get_entry(r, c_a);
        engine::DenseBuf<Elt> R(rref_A_flat_data.data(), rref_A_flat_data.size(), m, n, engine::Layout::RowMajor);
        engine::DenseBuf<Elt> U(C_transform_flat_data.data(), C_transform_flat_data.size(), m, m, engine::Layout::RowMajor);
        engine::elim::rref_with_transform<Elt>(ctx, engine::cview(A), engine::view(R), engine::view(U));
    }
    std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> out{ FlatMatrixT<Elt>(m, n, p_, rref_A_flat_data),
                                                      FlatMatrixT<Elt>(m, m, p_, C_transform_flat_data) };
#ifdef ENGINE_SHADOW_LEGACY
    shadow_check_pair<Elt>(out, _legacy_rref_and_transform(), "get_rref_and_transform_matrix (RREF)", "get_rref_and_transform_matrix (U)");
#endif
    return out;
}

template<class Elt>
std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> FlatMatrixT<Elt>::compute_elimination_factor_and_apply(const FlatMatrixT<Elt>& pivot_rref_block) const {
    // B is *this matrix. R is pivot_rref_block.
    uint32_t m_B = this->rows_;
    uint32_t n_B = this->cols_;
    uint32_t k_R = pivot_rref_block.rows_; // R.rows
    uint32_t n_R = pivot_rref_block.cols_;
    Elt prime = this->p_;

    if (n_B != n_R) {
        throw std::invalid_argument("Column count mismatch for compute_elimination_factor_and_apply. B.cols != R.cols");
    }
    if (prime != pivot_rref_block.get_p()) {
        throw std::invalid_argument("Prime modulus mismatch.");
    }

    // (Phase 1, defect #6) The two deep copies that used to be made here -- one of
    // *this*, one of the pivot block -- solely to call the then-non-const
    // is_zero()/rank() are gone.  This is the innermost loop of the block
    // elimination, and the R copy in particular threw away the pivot block's RREF
    // cache once per target row: the same pivot block is now RREF'd once.
    // Special case: If B0 (*this) is already zero
    if (this->is_zero()) {
        FlatMatrixT<Elt> L_factor = FlatMatrixT<Elt>::CreateZero(m_B, k_R, prime);
        return { *this, L_factor }; // B_new is B (which is zero), L is zero
    }

    // Special case: If R (pivot_rref_block) is zero
    if (pivot_rref_block.is_zero()) {
        FlatMatrixT<Elt> L_factor = FlatMatrixT<Elt>::CreateZero(m_B, k_R, prime); // Or any matrix, as L*0 = 0
        return { *this, L_factor }; // B_new is B, L can be zero
    }

    Elt R_scalar_val;
    bool R_is_scalar = pivot_rref_block.is_currently_scalar(R_scalar_val);

    // Special case: If R is Identity (and square, which it should be if scalar and RREF)
    if (R_is_scalar && R_scalar_val == 1 && pivot_rref_block.get_rows() == pivot_rref_block.get_cols() && pivot_rref_block.get_rows() > 0) {
        // R is Identity. L_factor = B. B_new = B - B*I = 0.
        // L_factor needs to be m_B x k_R. If R is I (k_R x k_R),
        // B must be m_B x k_R for B*I to make sense in B - B*I = 0.
        // This implies B.cols (n_B) == R.rows (k_R).
        if (n_B == k_R) { // If B's columns match R's rows (which are also R's cols)
            FlatMatrixT<Elt> L_factor = *this; // L = B. (m_B x k_R)
            FlatMatrixT<Elt> B_new = FlatMatrixT<Elt>::CreateZero(m_B, n_B, prime); // B_new = 0
            return { B_new, L_factor };
        }
        // If n_B != k_R, R=I case is more complex for L. Fall through to general.
    }


    // General case (already implemented, with slight modifications for clarity)
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)

    uint32_t rank_R = pivot_rref_block.rank(); // Computes/caches RREF & pivots ON the pivot block
    const std::vector<uint32_t>& R_pivot_cols_indices = pivot_rref_block.get_pivot_columns();

    if (rank_R == 0) { // Should have been caught by the is_zero() check earlier, but defensive.
        FlatMatrixT<Elt> L_factor = FlatMatrixT<Elt>::CreateZero(m_B, k_R, prime);
        return { *this, L_factor };
    }

#if OC_ENABLE_LEGACY
    if (engine::elim::use_legacy()) return _legacy_elimination_factor_and_apply(pivot_rref_block);   // legacy/FlatMatrixLegacy.inc
#endif

    // (matrix-engine Phase 2) Engine route: L = B[:, pivots of R] and
    // B_new = B - L*R in ONE in-place GEMM (GemmOp::Sub) on a copy of B --
    // no L*R temporary, no separate subtraction pass.  Same values by exact
    // arithmetic; same representation: an all-zero L is CreateZero (as the
    // legacy set_entry loop leaves it), otherwise the DATA constructor.
    // *this may still be a non-zero SCALAR block (alpha*I with empty storage):
    // materialise through get_entry, exactly as the legacy loop read it.
    std::vector<Elt> b_new(static_cast<size_t>(m_B) * n_B);
    if (!is_scalar_matrix_ && flat_data_.size() == b_new.size()) {
        b_new = flat_data_;
    } else {
        for (uint32_t i = 0; i < m_B; ++i)
            for (uint32_t j = 0; j < n_B; ++j) b_new[static_cast<size_t>(i) * n_B + j] = this->get_entry(i, j);
    }
    engine::DenseBuf<Elt> L_buf;
    {
        const engine::ModCtx<Elt> ctx(prime);
        engine::DenseBuf<Elt> B(b_new.data(), b_new.size(), m_B, n_B, engine::Layout::RowMajor);
        engine::DenseBuf<Elt> R_copy;
        const engine::DenseBuf<Elt>* R_ptr;
        if (!pivot_rref_block.is_scalar_matrix_ &&
            pivot_rref_block.flat_data_.size() == static_cast<size_t>(k_R) * n_R) {
            // read-only wrap of R's flat storage (gemm only reads it)
            R_copy = engine::DenseBuf<Elt>(const_cast<Elt*>(pivot_rref_block.flat_data_.data()),
                                           pivot_rref_block.flat_data_.size(), k_R, n_R,
                                           engine::Layout::RowMajor);
            R_ptr = &R_copy;
        } else {
            R_copy = engine::DenseBuf<Elt>(k_R, n_R);
            for (uint32_t i = 0; i < k_R; ++i)
                for (uint32_t j = 0; j < n_R; ++j) R_copy.at(i, j) = pivot_rref_block.get_entry(i, j);
            R_ptr = &R_copy;
        }
        L_buf = engine::elim::eliminate_against<Elt>(ctx, engine::view(B), engine::cview(*R_ptr),
                                                     R_pivot_cols_indices, k_R);
    }
    bool L_all_zero = true;
    for (size_t i = 0; i < L_buf.footprint() && L_all_zero; ++i) L_all_zero = (L_buf.data()[i] == 0);
    std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> out{
        FlatMatrixT<Elt>(m_B, n_B, prime, b_new),
        L_all_zero ? FlatMatrixT<Elt>::CreateZero(m_B, k_R, prime)
                   : FlatMatrixT<Elt>(m_B, k_R, prime,
                                      std::vector<Elt>(L_buf.data(), L_buf.data() + L_buf.footprint())) };
#ifdef ENGINE_SHADOW_LEGACY
    shadow_check_pair<Elt>(out, _legacy_elimination_factor_and_apply(pivot_rref_block), "compute_elimination_factor_and_apply (B_new)", "compute_elimination_factor_and_apply (L)");
#endif
    return out;
}

template<class Elt>
uint32_t FlatMatrixT<Elt>::rank() const { // (defect #19) const: is_zero() and the RREF cache are both const
    if (this->is_zero()) { // is_zero() updates flags if general matrix becomes scalar 0
        return 0;
    }
    // Now matrix is known non-zero.
    if (is_scalar_matrix_) { // And scalar_value_ != 0
        // Rank of alpha*I (alpha!=0) is full rank (min(rows,cols), but should be square)
        return std::min(rows_, cols_);
    }
    // General non-zero matrix
    compute_row_echelon_form(); // compute_row_echelon_form is const
    return pivot_columns_cache_->size();
}

template<class Elt>
bool FlatMatrixT<Elt>::is_invertible() const { // (defect #19)
    if (rows_ != cols_ || rows_ == 0) { // Must be non-empty square
        return false;
    }
    return rank() == rows_;
}

template<class Elt>
FlatMatrixT<Elt> FlatMatrixT<Elt>::inverse() const {
    // (Phase 1, defect #6) The full deep copy that used to be made here purely to
    // reach the then-non-const is_invertible() is gone; is_invertible() is const
    // and its side effects (RREF cache, scalar flags) are exactly the ones the
    // branch below wants to see on *this*.
    if (!is_invertible()) {
        throw std::runtime_error("Matrix is not invertible.");
    }

    if (is_scalar_matrix_) { // Implies scalar_value_ != 0 because it's invertible
        Elt inv_scalar = ModArithmetic::mod_inv_w(scalar_value_, p_);
        return FlatMatrixT<Elt>::CreateScalar(rows_, inv_scalar, p_);
    }

    // General invertible matrix (this is const, use its data)
    if (!is_scalar_matrix_) populate_flat_data(); // (defect #24)
    if (!is_scalar_matrix_ && flat_data_.empty() && rows_ > 0) { // Should not happen for invertible general
        throw std::logic_error("Inverse: General invertible matrix has empty flat_data_");
    }

#if OC_ENABLE_LEGACY
    if (engine::elim::use_legacy()) return _legacy_inverse();   // legacy/FlatMatrixLegacy.inc
#endif

    // (matrix-engine Phase 2) Engine route: RREF([A | I]) in the engine, the
    // inverse read off the trailing block -- the same augmented elimination.
    FlatMatrixT<Elt> inv_matrix(rows_, cols_, p_); // Starts scalar 0
    inv_matrix.populate_flat_data(); // Becomes general 0-filled
    {
        const engine::ModCtx<Elt> ctx(p_);
        engine::DenseBuf<Elt> A(rows_, cols_);
        for (uint32_t i = 0; i < rows_; ++i)
            for (uint32_t j = 0; j < cols_; ++j) A.at(i, j) = this->get_entry(i, j);
        engine::DenseBuf<Elt> Ainv(inv_matrix.flat_data_.data(), inv_matrix.flat_data_.size(),
                                   rows_, cols_, engine::Layout::RowMajor);
        const bool ok = engine::elim::inverse<Elt>(ctx, engine::cview(A), engine::view(Ainv));
        ENGINE_CHECK_MSG(ok, "inverse: is_invertible() held but the augmented RREF found a singular matrix");
    }
    inv_matrix._determine_and_set_flags_from_data(); // Check if inverse became special
#ifdef ENGINE_SHADOW_LEGACY
    shadow_check_equal<Elt>(inv_matrix, _legacy_inverse(), "inverse");
#endif
    return inv_matrix;
}

template<class Elt>
std::vector<FlatMatrixT<Elt>> FlatMatrixT<Elt>::kernel() const { // (defect #19)
    std::vector<FlatMatrixT<Elt>> kernel_basis;
    if (cols_ == 0) {
        return kernel_basis; // Kernel of an N x 0 matrix is {0} in a 0-dim space, basis is empty.
    }

    // Make a copy to safely call rank() and compute RREF without altering 'this'
    // if kernel() itself were const and rank() non-const.
    // Since kernel() is non-const, it can modify 'this's cache.
    // If 'this' is already RREF'd by a previous non-const call, this is efficient.
    // If not, compute_row_echelon_form (const) populates mutable cache.

    // Ensure RREF is computed. rank() calls compute_row_echelon_form().
    // To avoid making rank() non-const for this, we can call compute_row_echelon_form() directly.
    // However, kernel often follows rank or invertibility checks.
    // Let's stick to the existing pattern of relying on compute_row_echelon_form for cache.
    this->compute_row_echelon_form(); // Ensures RREF data is in cache

    // Check for trivial kernel if invertible (square and full rank)
    // This check should ideally use cached rank if available.
    // rank() is non-const, but compute_row_echelon_form() is const and called above.
    // We can directly use pivot_columns_cache_ here.
    if (rows_ == cols_ && rows_ != 0 && pivot_columns_cache_ && pivot_columns_cache_->size() == rows_) {
        return kernel_basis; // Invertible matrix has trivial kernel
    }


    const std::vector<Elt>& rref_data = *row_echelon_form_cache_;   // flat, row-major (Phase 2)
    const auto& pivots = *pivot_columns_cache_;
    for (uint32_t p_col : pivots) {
        if (p_col >= cols_) throw std::logic_error("Kernel: pivot column out of bounds.");
    }
    if (rref_data.size() != static_cast<size_t>(rows_) * cols_) {
        throw std::logic_error("Kernel: RREF cache size mismatch.");
    }

    // (matrix-engine Phase 2) D5b convention, built in one buffer by
    // engine::elim::kernel_basis: one vector per free column ascending, 1 at
    // the free column, -RREF[i][j_free] at pivot i.  Same values as the
    // per-vector loop this replaces; each vector still goes through the DATA
    // constructor, so its flags are set exactly as before.
    const engine::ModCtx<Elt> ctx(p_);
    const engine::DenseBuf<Elt> R(const_cast<Elt*>(rref_data.data()), rref_data.size(), rows_, cols_,
                                  engine::Layout::RowMajor);
    const engine::DenseBuf<Elt> K = engine::elim::kernel_basis<Elt>(ctx, engine::cview(R), pivots);
    kernel_basis.reserve(K.rows);
    for (uint32_t t = 0; t < K.rows; ++t) {
        const Elt* kt = K.data() + static_cast<size_t>(t) * K.ld;
        kernel_basis.emplace_back(cols_, 1, p_, std::vector<Elt>(kt, kt + cols_));
    }
    return kernel_basis;
}

template<class Elt>
std::vector<FlatMatrixT<Elt>> FlatMatrixT<Elt>::image() const { // (defect #19)
    std::vector<FlatMatrixT<Elt>> image_basis;
    if (cols_ == 0 || rows_ == 0) return image_basis;

    // is_zero() can modify flags if a general matrix turns out to be zero.
    // Make a copy if image() were const. Since it's non-const, it can modify 'this'.
    if (this->is_zero()) {
        return image_basis;
    }

    // If scalar non-zero (implies invertible, square), image is the full space.
    // This check is safe now because is_zero() above would have handled scalar zero.
    // So if is_scalar_matrix_ is true here, scalar_value_ must be non-zero.
    //
    // Owner decision Q11 (matrix-engine plan Sec. 7): the returned basis is the
    // list of PIVOT COLUMNS OF THE ORIGINAL MATRIX, exactly as in the general
    // branch below and as in MatrixT::image() / BlockFlatMatrixT::flat_image().
    // For A = alpha*I that is alpha*e_j, not e_j -- this fast path used to
    // return the standard basis e_j, so the three classes disagreed on the
    // scalar case (same span, different vectors). They now agree entrywise.
    if (is_scalar_matrix_ /* && scalar_value_ != 0 implied */) {
        if (rows_ != cols_) { /* This would be an inconsistent state if scalar_value_ != 0 */ }
        for (uint32_t j = 0; j < rows_; ++j) { // Column j of alpha*I
            std::vector<Elt> basis_vector_data(rows_, 0);
            basis_vector_data[j] = scalar_value_;
            image_basis.emplace_back(rows_, 1, p_, basis_vector_data);
        }
        return image_basis;
    }

    // General non-zero matrix
    this->compute_row_echelon_form(); // Ensures RREF data is in cache (pivots)
    // _ensure_data_populated_if_empty(); // Not strictly needed if get_entry handles scalar flags or if general matrix always has data

    for (uint32_t pivot_col_idx : *pivot_columns_cache_) {
        // Create data for a column vector (rows_ x 1)
        std::vector<Elt> basis_vector_data(rows_);
        for (uint32_t i_row = 0; i_row < rows_; ++i_row) {
            // Get entry from original matrix 'this'
            basis_vector_data[i_row] = this->get_entry(i_row, pivot_col_idx);
        }
        // Construct FlatMatrix from this data vector
        image_basis.emplace_back(rows_, 1, p_, basis_vector_data);
    }
    return image_basis;
}

// --- Free functions ported from Matrix utilities ---

// ===========================================================================
// rho_n = ComputeSymmetricPowerMatrixFM  --  matrix-engine Phase 1, items
// 7.0 / 7.5 (docs/matrix-engine/00-implementation-plan.md, "Phase 1").
//
// CONVENTION (unchanged, and the reason every variant below is bit-identical
// to the reference implementation): Sym^n(V) has the ordered monomial basis
//     u_k = x^(n-k) y^k,    k = 0 .. n = n_sym,     d_V = n_sym + 1,
// and for A = [[a,b],[c,d]] acting on V = <x,y> the j-th COLUMN of rho_n(A)
// holds the coordinates of
//     A_*(u_j) = (a x + c y)^(n-j) * (b x + d y)^j.
// Dehomogenising with x = 1 turns the column into the univariate polynomial
//     P_j(y) = (a + c y)^(n-j) * (b + d y)^j          (deg <= n),
// whose coefficient of y^k is exactly the entry (k, j).  rho_n(A) is therefore
// a mathematically UNIQUE object; any correct evaluation route reproduces the
// reference matrix entry for entry.  `ComputeSymmetricPowerMatrixFM_reference`
// (below) keeps the original O(d_V^2) *terms* x O(d_V^2) *flag rescans* route
// alive purely as the oracle of `SymmetricPowerValidation`.
//
// BRANCH STRUCTURE (all branches O(d_V^2) or better; none calls mod_pow, and
// the only mod_nCr calls are the O(d_V) of a single binomial row):
//
//   [C1] b == 0 && c == 0            diagonal        entry(j,j)   = a^(n-j) d^j
//        -- covers rho_n(alpha*I) = alpha^n * I, in particular rho_n(-I) and
//           rho_n(I), and every diagonal Hecke coset rep [[1,0],[0,p]].
//   [C2] a == 0 && d == 0            anti-diagonal   entry(n-j,j) = c^(n-j) b^j
//        -- covers rho_n(S), S = [[0,-1],[1,0]]: entry(n-j,j) = (-1)^j.
//   [C3] a == 0 && b == 0            first row of A vanishes: A_*(u_j) is a
//                                    multiple of y^n; row n only.
//   [C4] c == 0 && d == 0            second row vanishes; row 0 only.
//   [C5] c == 0 && a == 1 && d == 1  upper unitriangular [[1,b],[0,1]]
//                                    entry(t,j) = C(j,t) b^(j-t)   (t <= j)
//        -- b = 1 is rho_n(T): the upper unitriangular Pascal matrix.
//   [C6] b == 0 && a == 1 && d == 1  lower unitriangular [[1,0],[c,1]]
//                                    entry(i,j) = C(n-j,i-j) c^(i-j) (i >= j)
//   [R1] a != 0                      forward two-term recurrence in j
//   [R2] a == 0 (hence b != 0 after [C3])   mirrored backward recurrence
//
// The recurrences.  [R1] starts from P_0 = (a + c y)^n (one binomial row,
// O(d_V)) and advances j -> j+1 by
//     P_{j+1} = P_j * (b + d y) / (a + c y),
// the division being EXACT in F_ell[y] because P_j carries the factor
// (a + c y)^(n-j) with n - j >= 1.  With q = P_j*(b+dy) and r = P_{j+1},
// q_k = a r_k + c r_{k-1} gives the forward substitution
//     r_k = (b*P_j[k] + d*P_j[k-1] - c*r_{k-1}) / a,        k = 0 .. n,
// i.e. O(d_V) multiplications per column and O(d_V^2) overall.  It needs
// a != 0 in F_ell.  [R2] is the mirror image: start from P_n = (b + d y)^n and
// walk j -> j-1 by P_{j-1} = P_j * (a + c y) / (b + d y) (exact since
// (b + d y)^j divides P_j for j >= 1), i.e.
//     r_k = (a*P_j[k] + c*P_j[k-1] - d*r_{k-1}) / b,
// which needs b != 0.
//
// COMPLETENESS of the branch set.  If a == 0 and b == 0, [C3] has already
// fired; otherwise a != 0 selects [R1] and a == 0 leaves b != 0 for [R2].  No
// determinant hypothesis is used anywhere: det A = 1 would give "a and b not
// both zero" for free, but the callers pass matrices of determinant p (Hecke
// coset representatives, HeckeCosetTables.cpp / HeckeTauTable.cpp) and hence,
// at the bad prime ell = p, SINGULAR matrices mod ell -- [C1]-[C4] and the two
// recurrences are all valid for arbitrary 2x2 matrices over F_ell, singular or
// not.  The exactness of the two divisions rests only on the factorisation of
// P_j displayed above, never on det A.
// ===========================================================================

namespace {

// [ z^0, z^1, ..., z^(len-1) ] mod prime.  z^0 = 1 for every z, including
// z = 0 -- matching mod_pow_w(0, 0, p) = 1 in the reference implementation.
template<class Elt>
std::vector<Elt> rho_power_row(Elt z, uint32_t len, Elt prime) {
    std::vector<Elt> pz(len);
    if (len == 0) return pz;
    pz[0] = Elt(1) % prime;
    for (uint32_t k = 1; k < len; ++k) pz[k] = ModArithmetic::mod_mul_w(pz[k - 1], z, prime);
    return pz;
}

// (a + c y)^n as a coefficient vector of length n+1: coeff[k] = C(n,k) a^(n-k) c^k.
// O(n) multiplications plus the n+1 tabulated mod_nCr lookups.  0^0 = 1, which
// is exactly what mod_pow_w(0, 0, p) returns in the reference implementation.
template<class Elt>
std::vector<Elt> rho_binomial_row(Elt a, Elt c, uint32_t n, Elt prime) {
    const uint32_t d_V = n + 1;
    const std::vector<Elt> pa = rho_power_row<Elt>(a, d_V, prime);
    const std::vector<Elt> pc = rho_power_row<Elt>(c, d_V, prime);
    std::vector<Elt> out(d_V);
    for (uint32_t k = 0; k < d_V; ++k) {
        const Elt bin = ModArithmetic::mod_nCr_w(Elt(n), Elt(k), prime);
        out[k] = ModArithmetic::mod_mul_w(bin,
                     ModArithmetic::mod_mul_w(pa[n - k], pc[k], prime), prime);
    }
    return out;
}

} // namespace

template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixFM(const FlatMatrixT<Elt>& A, uint32_t n_sym) {
    if (A.get_rows() != 2 || A.get_cols() != 2) {
        throw std::runtime_error("ComputeSymmetricPowerMatrixFM: Input matrix A must be 2x2.");
    }
    const Elt prime = A.get_p();
    ENGINE_CHECK_MSG(prime >= 2, "ComputeSymmetricPowerMatrixFM: modulus must be >= 2.");

    const uint32_t n = n_sym;
    const uint32_t d_V = n_sym + 1;

    const Elt a = A.get_entry(0, 0);
    const Elt b = A.get_entry(0, 1);
    const Elt c = A.get_entry(1, 0);
    const Elt d = A.get_entry(1, 1);

    // Row-major (k, j) -> k*d_V + j, filled once and handed to the data
    // constructor, which reduces mod p and runs _determine_and_set_flags_from_data()
    // exactly once (the reference route ran it once per set_entry).
    std::vector<Elt> out(static_cast<size_t>(d_V) * d_V, Elt(0));
    auto at = [d_V](uint32_t k, uint32_t j) -> size_t {
        return static_cast<size_t>(k) * d_V + j;
    };

    // --- [C1] diagonal: entry(j,j) = a^(n-j) d^j -------------------------
    if (b == 0 && c == 0) {
        const std::vector<Elt> pa = rho_power_row<Elt>(a, d_V, prime);
        const std::vector<Elt> pd = rho_power_row<Elt>(d, d_V, prime);
        for (uint32_t j = 0; j < d_V; ++j) {
            out[at(j, j)] = ModArithmetic::mod_mul_w(pa[n - j], pd[j], prime);
        }
        return FlatMatrixT<Elt>(d_V, d_V, prime, out);
    }

    // --- [C2] anti-diagonal: entry(n-j,j) = c^(n-j) b^j ------------------
    if (a == 0 && d == 0) {
        const std::vector<Elt> pc = rho_power_row<Elt>(c, d_V, prime);
        const std::vector<Elt> pb = rho_power_row<Elt>(b, d_V, prime);
        for (uint32_t j = 0; j < d_V; ++j) {
            out[at(n - j, j)] = ModArithmetic::mod_mul_w(pc[n - j], pb[j], prime);
        }
        return FlatMatrixT<Elt>(d_V, d_V, prime, out);
    }

    // --- [C3] first row of A vanishes: entry(n,j) = c^(n-j) d^j ----------
    if (a == 0 && b == 0) {
        const std::vector<Elt> pc = rho_power_row<Elt>(c, d_V, prime);
        const std::vector<Elt> pd = rho_power_row<Elt>(d, d_V, prime);
        for (uint32_t j = 0; j < d_V; ++j) {
            out[at(n, j)] = ModArithmetic::mod_mul_w(pc[n - j], pd[j], prime);
        }
        return FlatMatrixT<Elt>(d_V, d_V, prime, out);
    }

    // --- [C4] second row of A vanishes: entry(0,j) = a^(n-j) b^j ---------
    if (c == 0 && d == 0) {
        const std::vector<Elt> pa = rho_power_row<Elt>(a, d_V, prime);
        const std::vector<Elt> pb = rho_power_row<Elt>(b, d_V, prime);
        for (uint32_t j = 0; j < d_V; ++j) {
            out[at(0, j)] = ModArithmetic::mod_mul_w(pa[n - j], pb[j], prime);
        }
        return FlatMatrixT<Elt>(d_V, d_V, prime, out);
    }

    // --- [C5] upper unitriangular [[1,b],[0,1]] (b != 0 here) ------------
    //     entry(t,j) = C(j,t) b^(j-t).  b == 1 is rho_n(T) = Pascal.
    if (c == 0 && a == Elt(1) % prime && d == Elt(1) % prime) {
        std::vector<Elt> pb(d_V);
        pb[0] = Elt(1) % prime;
        for (uint32_t k = 1; k < d_V; ++k) pb[k] = ModArithmetic::mod_mul_w(pb[k - 1], b, prime);
        for (uint32_t j = 0; j < d_V; ++j) {
            for (uint32_t t = 0; t <= j; ++t) {
                const Elt bin = ModArithmetic::mod_nCr_w(Elt(j), Elt(t), prime);
                out[at(t, j)] = ModArithmetic::mod_mul_w(bin, pb[j - t], prime);
            }
        }
        return FlatMatrixT<Elt>(d_V, d_V, prime, out);
    }

    // --- [C6] lower unitriangular [[1,0],[c,1]] (c != 0 here) ------------
    //     entry(i,j) = C(n-j, i-j) c^(i-j).
    if (b == 0 && a == Elt(1) % prime && d == Elt(1) % prime) {
        std::vector<Elt> pc(d_V);
        pc[0] = Elt(1) % prime;
        for (uint32_t k = 1; k < d_V; ++k) pc[k] = ModArithmetic::mod_mul_w(pc[k - 1], c, prime);
        for (uint32_t j = 0; j < d_V; ++j) {
            for (uint32_t i = j; i < d_V; ++i) {
                const Elt bin = ModArithmetic::mod_nCr_w(Elt(n - j), Elt(i - j), prime);
                out[at(i, j)] = ModArithmetic::mod_mul_w(bin, pc[i - j], prime);
            }
        }
        return FlatMatrixT<Elt>(d_V, d_V, prime, out);
    }

    std::vector<Elt> cur, nxt(d_V);

    if (a != 0) {
        // --- [R1] forward recurrence, columns 0 -> n ---------------------
        const Elt ainv = ModArithmetic::mod_inv_w(a, prime);
        cur = rho_binomial_row<Elt>(a, c, n, prime);       // P_0 = (a + c y)^n
        for (uint32_t k = 0; k < d_V; ++k) out[at(k, 0)] = cur[k];

        for (uint32_t j = 0; j + 1 < d_V; ++j) {
            Elt r_prev = 0;    // r_{k-1}
            Elt cur_prev = 0;  // P_j[k-1]
            for (uint32_t k = 0; k < d_V; ++k) {
                // q_k = b*P_j[k] + d*P_j[k-1]
                const Elt q = ModArithmetic::mod_add_w(
                    ModArithmetic::mod_mul_w(b, cur[k], prime),
                    ModArithmetic::mod_mul_w(d, cur_prev, prime), prime);
                const Elt r = ModArithmetic::mod_mul_w(
                    ModArithmetic::mod_sub_w(q, ModArithmetic::mod_mul_w(c, r_prev, prime), prime),
                    ainv, prime);
                nxt[k] = r;
                cur_prev = cur[k];
                r_prev = r;
            }
            // Exactness certificate: the y^(n+1) coefficient of the dividend,
            // d*P_j[n], must be cancelled by c*r_n (zero remainder).
            ENGINE_DEBUG_CHECK_MSG(
                ModArithmetic::mod_mul_w(d, cur[n], prime)
                    == ModArithmetic::mod_mul_w(c, r_prev, prime),
                "rho_n forward recurrence: non-zero remainder (division not exact).");
            cur.swap(nxt);
            for (uint32_t k = 0; k < d_V; ++k) out[at(k, j + 1)] = cur[k];
        }
    } else {
        // --- [R2] mirrored backward recurrence, columns n -> 0 -----------
        // a == 0 and (by [C3]) b != 0.
        const Elt binv = ModArithmetic::mod_inv_w(b, prime);
        cur = rho_binomial_row<Elt>(b, d, n, prime);       // P_n = (b + d y)^n
        for (uint32_t k = 0; k < d_V; ++k) out[at(k, n)] = cur[k];

        for (uint32_t j = n; j > 0; --j) {
            Elt r_prev = 0;
            Elt cur_prev = 0;
            for (uint32_t k = 0; k < d_V; ++k) {
                // q_k = a*P_j[k] + c*P_j[k-1]   (a == 0, kept for symmetry)
                const Elt q = ModArithmetic::mod_add_w(
                    ModArithmetic::mod_mul_w(a, cur[k], prime),
                    ModArithmetic::mod_mul_w(c, cur_prev, prime), prime);
                const Elt r = ModArithmetic::mod_mul_w(
                    ModArithmetic::mod_sub_w(q, ModArithmetic::mod_mul_w(d, r_prev, prime), prime),
                    binv, prime);
                nxt[k] = r;
                cur_prev = cur[k];
                r_prev = r;
            }
            ENGINE_DEBUG_CHECK_MSG(
                ModArithmetic::mod_mul_w(c, cur[n], prime)
                    == ModArithmetic::mod_mul_w(d, r_prev, prime),
                "rho_n backward recurrence: non-zero remainder (division not exact).");
            cur.swap(nxt);
            for (uint32_t k = 0; k < d_V; ++k) out[at(k, j - 1)] = cur[k];
        }
    }

    return FlatMatrixT<Elt>(d_V, d_V, prime, out);
}

// ---------------------------------------------------------------------------
// Reference (pre-Phase-1) implementation, kept ONLY as the oracle of
// SymmetricPowerValidation.  Direct O(d_V^2)-term double sum with a mod_nCr and
// two mod_pow per term and a full d_V x d_V flag rescan per set_entry; this is
// the routine measured at 5.94 s for a single n = 160 evaluation at -O0
// (docs/matrix-engine/02-pipeline-consumer-audit.md Sec. 5.5 C).  Do not call
// it from production code.
// ---------------------------------------------------------------------------
template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixFM_reference(const FlatMatrixT<Elt>& A, uint32_t n_sym) {
    if (A.get_rows() != 2 || A.get_cols() != 2) {
        throw std::runtime_error("ComputeSymmetricPowerMatrixFM_reference: Input matrix A must be 2x2.");
    }
    Elt prime = A.get_p();

    uint32_t d_V = n_sym + 1;
    FlatMatrixT<Elt> B(d_V, d_V, prime); // zero matrix

    Elt a = A.get_entry(0, 0);
    Elt b = A.get_entry(0, 1);
    Elt c = A.get_entry(1, 0);
    Elt d = A.get_entry(1, 1);

    // Need to populate B's flat data so we can accumulate with mod_add
    B.populate_flat_data();

    for (uint32_t j_col = 0; j_col < d_V; ++j_col) {
        for (uint32_t s = 0; s <= (n_sym - j_col); ++s) {
            Elt term1_coeff_ncr = ModArithmetic::mod_nCr_w(Elt(n_sym - j_col), Elt(s), prime);
            Elt term1_val_a = ModArithmetic::mod_pow_w(a, Elt(s), prime);
            Elt term1_val_c = ModArithmetic::mod_pow_w(c, Elt((n_sym - j_col) - s), prime);
            Elt term1 = ModArithmetic::mod_mul_w(term1_coeff_ncr, ModArithmetic::mod_mul_w(term1_val_a, term1_val_c, prime), prime);

            for (uint32_t t = 0; t <= j_col; ++t) {
                Elt term2_coeff_ncr = ModArithmetic::mod_nCr_w(Elt(j_col), Elt(t), prime);
                Elt term2_val_b = ModArithmetic::mod_pow_w(b, Elt(t), prime);
                Elt term2_val_d = ModArithmetic::mod_pow_w(d, Elt(j_col - t), prime);
                Elt term2 = ModArithmetic::mod_mul_w(term2_coeff_ncr, ModArithmetic::mod_mul_w(term2_val_b, term2_val_d, prime), prime);

                Elt total_coeff = ModArithmetic::mod_mul_w(term1, term2, prime);

                if (total_coeff == 0) continue;

                uint32_t power_x = s + t;
                uint32_t k_row = n_sym - power_x;

                if (k_row < d_V) {
                    Elt old_val = B.get_entry(k_row, j_col);
                    B.set_entry(k_row, j_col, ModArithmetic::mod_add_w(old_val, total_coeff, prime));
                }
            }
        }
    }
    return B;
}

// ---------------------------------------------------------------------------
// (hecke-extensions slot 8, WP-3; E8-P5) rho_n over Z/ell^k -- see FlatMatrix.h.
// Every product and sum goes through the composite ctx; no mod_nCr, no pow.
// ---------------------------------------------------------------------------
const char* to_text(RhoRingRoute r) {
    switch (r) {
        case RhoRingRoute::UnitA: return "unit_a";
        case RhoRingRoute::UnitB: return "unit_b";
        case RhoRingRoute::Split: return "split";
        case RhoRingRoute::FirstRowZero: return "first_row_zero";
    }
    return "?";
}

namespace {

// Row n of Pascal's triangle mod ell^k, by additions only (O(n^2)).
template<class Elt>
std::vector<Elt> ring_pascal_row(const engine::ModCtx<Elt>& ctx, uint32_t n) {
    std::vector<Elt> row(static_cast<size_t>(n) + 1, Elt(0));
    row[0] = 1;
    for (uint32_t r = 1; r <= n; ++r)
        for (uint32_t j = r; j >= 1; --j) row[j] = ctx.add(row[j], row[j - 1]);
    return row;
}

// [ z^0, ..., z^(len-1) ] mod ell^k by repeated multiplication (z^0 = 1).
template<class Elt>
std::vector<Elt> ring_power_row(const engine::ModCtx<Elt>& ctx, Elt z, uint32_t len) {
    std::vector<Elt> pz(len);
    if (len == 0) return pz;
    pz[0] = 1;
    for (uint32_t k = 1; k < len; ++k) pz[k] = ctx.mul(pz[k - 1], z);
    return pz;
}

// The production route, row-major d_V x d_V into `out` (zero-initialised).
template<class Elt>
void rho_ring_fill(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d, uint32_t n,
                   std::vector<Elt>& out, RhoRingRoute* route, uint32_t depth) {
    const uint32_t d_V = n + 1;
    auto at = [d_V](uint32_t k, uint32_t j) -> size_t { return static_cast<size_t>(k) * d_V + j; };
    const bool a_unit = ctx.valuation(a) == 0, b_unit = ctx.valuation(b) == 0;
    if (a_unit || b_unit) {
        if (route && depth == 0) *route = a_unit ? RhoRingRoute::UnitA : RhoRingRoute::UnitB;
        // [U_a]: P_0 = (a + c y)^n, P_{j+1} = P_j (b + d y) / (a + c y);
        // [U_b]: P_n = (b + d y)^n, P_{j-1} = P_j (a + c y) / (b + d y).
        const Elt u = a_unit ? a : b, w = a_unit ? c : d;       // the divisor u + w y
        const Elt s = a_unit ? b : a, t = a_unit ? d : c;       // the multiplier s + t y
        const Elt uinv = ctx.inv(u);
        const std::vector<Elt> bin = ring_pascal_row<Elt>(ctx, n);
        const std::vector<Elt> pu = ring_power_row<Elt>(ctx, u, d_V), pw = ring_power_row<Elt>(ctx, w, d_V);
        std::vector<Elt> cur(d_V), nxt(d_V);
        for (uint32_t k = 0; k < d_V; ++k) cur[k] = ctx.mul(bin[k], ctx.mul(pu[n - k], pw[k]));
        const uint32_t j0 = a_unit ? 0 : n;
        for (uint32_t k = 0; k < d_V; ++k) out[at(k, j0)] = cur[k];
        for (uint32_t step = 0; step < n; ++step) {
            Elt r_prev = 0, cur_prev = 0;
            for (uint32_t k = 0; k < d_V; ++k) {
                // (s + t y) P = (u + w y) R:  r_k = (s P[k] + t P[k-1] - w r_{k-1}) / u
                const Elt q = ctx.add(ctx.mul(s, cur[k]), ctx.mul(t, cur_prev));
                const Elt r = ctx.mul(ctx.sub(q, ctx.mul(w, r_prev)), uinv);
                nxt[k] = r;
                cur_prev = cur[k];
                r_prev = r;
            }
            ENGINE_DEBUG_CHECK_MSG(ctx.mul(t, cur[n]) == ctx.mul(w, r_prev),
                                   "rho_n over Z/ell^k: non-zero remainder (division not exact).");
            cur.swap(nxt);
            const uint32_t j = a_unit ? step + 1 : n - step - 1;
            for (uint32_t k = 0; k < d_V; ++k) out[at(k, j)] = cur[k];
        }
        return;
    }
    if (a == 0 && b == 0) {
        // [Z] the first row vanishes: column j is (c y)^(n-j) (d y)^j = c^(n-j) d^j y^n.
        if (route && depth == 0) *route = RhoRingRoute::FirstRowZero;
        const std::vector<Elt> pc = ring_power_row<Elt>(ctx, c, d_V), pd = ring_power_row<Elt>(ctx, d, d_V);
        for (uint32_t j = 0; j < d_V; ++j) out[at(n, j)] = ctx.mul(pc[n - j], pd[j]);
        return;
    }
    // [S] ell | a, ell | b (as integers: canonical representatives with
    // valuation >= 1), not both zero: g = diag(ell, 1) g', and row k of
    // rho_n(g') is scaled by ell^(n-k).  Each split lowers min(v(a), v(b)) (a
    // zero entry stays zero), so the recursion ends in [U_a] / [U_b] after at
    // most k - 1 splits.
    if (route && depth == 0) *route = RhoRingRoute::Split;
    ENGINE_CHECK_MSG(depth < ctx.k, "rho_n over Z/ell^k: split recursion deeper than k");
    rho_ring_fill<Elt>(ctx, static_cast<Elt>(a / ctx.ell), static_cast<Elt>(b / ctx.ell), c, d, n, out, route,
                       depth + 1);
    Elt scale = 1;                  // ell^(n-kk), built from kk = n down to 0
    for (uint32_t kk = d_V; kk-- > 0;) {
        if (scale != 1) {
            for (uint32_t j = 0; j < d_V; ++j) out[at(kk, j)] = ctx.mul(out[at(kk, j)], scale);
        }
        scale = ctx.mul(scale, ctx.ell);
    }
}

} // namespace

template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixRing(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d,
                                                 uint32_t n_sym, RhoRingRoute* route) {
    ENGINE_CHECK_MSG(!ctx.is_field(),
                     "ComputeSymmetricPowerMatrixRing: composite-only (Z/ell^k); F_ell uses ComputeSymmetricPowerMatrixFM");
    ENGINE_CHECK_MSG(a < ctx.p && b < ctx.p && c < ctx.p && d < ctx.p,
                     "ComputeSymmetricPowerMatrixRing: entries must be canonical residues mod ell^k");
    const uint32_t d_V = n_sym + 1;
    std::vector<Elt> out(static_cast<size_t>(d_V) * d_V, Elt(0));
    rho_ring_fill<Elt>(ctx, a, b, c, d, n_sym, out, route, 0);
    return FlatMatrixT<Elt>(d_V, d_V, ctx.p, out);
}

// ---------------------------------------------------------------------------
// (S23 G2-T2) The dispatched SIMD column steps of rho_ring_fill_rows ([U_a] / [U_b], u32, p <= 2^31) -- see
// FlatMatrix.h (RhoFillPath, HECKE_RHO_SIMD).  The SAME wavefront as the scalar loop below, G lanes wide (AVX2: 4 ymm =
// 32 lanes; AVX-512F: 4 zmm = 64 lanes): at time kk lane i holds row kk - i of column step done + i, so with X the
// lane vector of time kk - 1 and A = X shifted up by one lane (lane 0 <- the finished column's row kk),
//     X' = s' A + t' A_prev + (-w') X          (A_prev = the A of time kk - 1)
// is exactly the scalar step r_k = s' P[k] + t' P[k-1] + (-w') r_{k-1} on every lane at once.  Each product is a
// Shoup multiplication by a per-key constant (the w' = floor(w 2^32 / p) of ModCtx::shoup_precompute: a lazy residue
// in [0, 2p), 2p <= 2^32), brought to [0, p) by min(r, r - p) (r - p wraps above r when r < p); each sum of two
// canonical residues likewise.  So every lane holds the canonical residue of the same ring element as the scalar
// step: the bytes are equal (HeckeModuleStreamValidation (V) forces every path and compares).  Lanes of rows < 0 hold
// 0 (zero start, zero padding of the finished column), lanes past row n compute values never stored.  Each lane
// vector is stored into a skewed buffer (time-major); the block's columns are then read back row by row (a gather of
// the diagonal when the block is full, the scalar loop on a tail block).
// ---------------------------------------------------------------------------
namespace {

// x86-64 GCC / Clang: the AVX2 / AVX-512F bodies under target attributes, chosen at runtime; elsewhere (aarch64, other
// compilers) the scalar path only.  -DRHO_FILL_PORTABLE_ONLY compiles the non-x86 branch on x86 (a compile check).
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(RHO_FILL_PORTABLE_ONLY)
#  define RHO_HAVE_X86_SIMD 1
#  define RHO_AVX2_FN   __attribute__((target("avx2")))
#  define RHO_AVX512_FN __attribute__((target("avx512f")))
#else
#  define RHO_HAVE_X86_SIMD 0
#endif

// (S24 CL-5) the CPU facts from CpuFeatures.h (false on a build without the bodies)
bool rho_cpu_has_avx2() { return RHO_HAVE_X86_SIMD && engine::cpu::avx2(); }
bool rho_cpu_has_avx512() { return RHO_HAVE_X86_SIMD && engine::cpu::avx512f(); }

// -1 = follow HECKE_RHO_SIMD, else the RhoFillPath value (ScopedRhoFillPath)
std::atomic<int> g_rho_fill_override{-1};
std::atomic<uint64_t> g_rho_fill_count[4] = {};   // indexed by RhoFillPath: [1] scalar, [2] avx2, [3] avx512

RhoFillPath rho_fill_env() {
    static const RhoFillPath p = [] {
        const char* e = std::getenv("HECKE_RHO_SIMD");
        if (e == nullptr) return RhoFillPath::Auto;
        const std::string v(e);
        if (v == "0" || v == "scalar") {
#if !OC_ENABLE_LEGACY
            // (S25 LEVERS-25) the rollback readings are retired in OFF builds; the scalar path itself stays
            // (production off x86 / without AVX2, and ScopedRhoFillPath(Scalar) in HeckeModuleStreamValidation (V)).
            oc::legacy::refuse_compiled_out(std::string("HECKE_RHO_SIMD=") + e);
#endif
            return RhoFillPath::Scalar;
        }
        if (v == "1" || v == "auto") return RhoFillPath::Auto;
        if (v == "avx2" || v == "avx512") {
            const RhoFillPath q = v == "avx2" ? RhoFillPath::Avx2 : RhoFillPath::Avx512;
            if (rho_fill_path_available(q)) return q;
            std::fprintf(stderr, "WARNING HECKE_RHO_SIMD=%s: not available on this CPU / build, the automatic choice stays\n", e);
            return RhoFillPath::Auto;
        }
        std::fprintf(stderr, "WARNING HECKE_RHO_SIMD=%s is none of 0, scalar, 1, auto, avx2, avx512: the automatic choice stays\n", e);
        return RhoFillPath::Auto;
    }();
    return p;
}
void print_rho_fill_effective_at_exit() {
    std::fprintf(stdout, "EFFECTIVE HECKE_RHO_SIMD %s\n", rho_fill_effective_text().c_str());
    std::fflush(stdout);
}
// registered at load (as HECKE_MODULE_EVAL's, G2-T audit F3): a set HECKE_RHO_SIMD always prints its line
[[maybe_unused]] const bool g_rho_fill_noted_at_load = [] {
    if (std::getenv("HECKE_RHO_SIMD") != nullptr) std::atexit(&print_rho_fill_effective_at_exit);
    return true;
}();

#if RHO_HAVE_X86_SIMD
// ---- AVX2: 8 u32 lanes per ymm --------------------------------------------------------------------------------------
// x w mod p for 8 lanes x < 2^32, w < p, ws = floor(w 2^32 / p), p <= 2^31: canonical.
RHO_AVX2_FN inline __m256i rho_shoup8(__m256i x, __m256i w, __m256i ws, __m256i pv) {
    const __m256i qe = _mm256_srli_epi64(_mm256_mul_epu32(x, ws), 32);        // even lanes: hi 32 bits of x ws
    const __m256i qo = _mm256_mul_epu32(_mm256_srli_epi64(x, 32), ws);        // odd lanes: hi 32 bits in place
    const __m256i q = _mm256_blend_epi32(qe, qo, 0xAA);
    const __m256i r = _mm256_sub_epi32(_mm256_mullo_epi32(x, w), _mm256_mullo_epi32(q, pv));   // in [0, 2p)
    return _mm256_min_epu32(r, _mm256_sub_epi32(r, pv));
}
RHO_AVX2_FN inline __m256i rho_addmod8(__m256i a, __m256i b, __m256i pv) {
    const __m256i s = _mm256_add_epi32(a, b);                                  // < 2p <= 2^32
    return _mm256_min_epu32(s, _mm256_sub_epi32(s, pv));
}
template<int NV>
RHO_AVX2_FN void rho_steps_avx2(uint32_t p, const uint32_t sw_[2], const uint32_t tw_[2], const uint32_t ww_[2],
                                uint32_t n, uint32_t* base, uint32_t* sk, uint32_t g, uint32_t G) {
    const __m256i pv = _mm256_set1_epi32(static_cast<int>(p));
    const __m256i sw = _mm256_set1_epi32(static_cast<int>(sw_[0])), sws = _mm256_set1_epi32(static_cast<int>(sw_[1]));
    const __m256i tw = _mm256_set1_epi32(static_cast<int>(tw_[0])), tws = _mm256_set1_epi32(static_cast<int>(tw_[1]));
    const __m256i ww = _mm256_set1_epi32(static_cast<int>(ww_[0])), wws = _mm256_set1_epi32(static_cast<int>(ww_[1]));
    const __m256i rot = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    __m256i X[NV], A[NV], Ap[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) { X[v] = _mm256_setzero_si256(); Ap[v] = _mm256_setzero_si256(); }
    const uint32_t steps = n + g;                                              // lane g - 1 reaches row n at time n + g - 1
    for (uint32_t kk = 0; kk < steps; ++kk) {
        __m256i prev = _mm256_set1_epi32(static_cast<int>(base[kk]));          // base is zero past row n
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {                                         // A = X shifted up one lane
            const __m256i r = _mm256_permutevar8x32_epi32(X[v], rot);
            A[v] = _mm256_blend_epi32(r, prev, 0x01);
            prev = r;
        }
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m256i y = rho_addmod8(rho_addmod8(rho_shoup8(A[v], sw, sws, pv), rho_shoup8(Ap[v], tw, tws, pv), pv),
                                          rho_shoup8(X[v], ww, wws, pv), pv);
            Ap[v] = A[v];
            X[v] = y;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(sk + static_cast<size_t>(kk) * G + 8 * v), y);
        }
    }
}
// the block's columns back into the row-major output: row k of lane i is sk[(k + i) G + i]
template<int NV>
RHO_AVX2_FN void rho_writeout_avx2(uint32_t n, uint32_t* out, const uint32_t* sk, uint32_t G, uint32_t done, bool a_unit) {
    const uint32_t d_V = n + 1;
    __m256i idx[NV];
    for (int v = 0; v < NV; ++v) {
        alignas(32) int32_t t[8];
        for (int w = 0; w < 8; ++w) t[w] = static_cast<int32_t>((8 * v + w) * (G + 1));
        idx[v] = _mm256_load_si256(reinterpret_cast<const __m256i*>(t));
    }
    const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    for (uint32_t k = 0; k < d_V; ++k) {
        uint32_t* row = out + static_cast<size_t>(k) * d_V;
        const int* src = reinterpret_cast<const int*>(sk + static_cast<size_t>(k) * G);
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m256i y = _mm256_i32gather_epi32(src, idx[v], 4);
            if (a_unit) _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + done + 1 + 8 * v), y);   // columns done+1+i
            else _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + (n - done - 1) - 8 * v - 7),    // columns n-done-1-i
                                     _mm256_permutevar8x32_epi32(y, rev));
        }
    }
}

// ---- AVX-512F: 16 u32 lanes per zmm ---------------------------------------------------------------------------------
RHO_AVX512_FN inline __m512i rho_shoup16(__m512i x, __m512i w, __m512i ws, __m512i pv) {
    const __m512i qe = _mm512_srli_epi64(_mm512_mul_epu32(x, ws), 32);
    const __m512i qo = _mm512_mul_epu32(_mm512_srli_epi64(x, 32), ws);
    const __m512i q = _mm512_mask_blend_epi32(0xAAAA, qe, qo);
    const __m512i r = _mm512_sub_epi32(_mm512_mullo_epi32(x, w), _mm512_mullo_epi32(q, pv));
    return _mm512_min_epu32(r, _mm512_sub_epi32(r, pv));
}
RHO_AVX512_FN inline __m512i rho_addmod16(__m512i a, __m512i b, __m512i pv) {
    const __m512i s = _mm512_add_epi32(a, b);
    return _mm512_min_epu32(s, _mm512_sub_epi32(s, pv));
}
template<int NV>
RHO_AVX512_FN void rho_steps_avx512(uint32_t p, const uint32_t sw_[2], const uint32_t tw_[2], const uint32_t ww_[2],
                                    uint32_t n, uint32_t* base, uint32_t* sk, uint32_t g, uint32_t G) {
    const __m512i pv = _mm512_set1_epi32(static_cast<int>(p));
    const __m512i sw = _mm512_set1_epi32(static_cast<int>(sw_[0])), sws = _mm512_set1_epi32(static_cast<int>(sw_[1]));
    const __m512i tw = _mm512_set1_epi32(static_cast<int>(tw_[0])), tws = _mm512_set1_epi32(static_cast<int>(tw_[1]));
    const __m512i ww = _mm512_set1_epi32(static_cast<int>(ww_[0])), wws = _mm512_set1_epi32(static_cast<int>(ww_[1]));
    __m512i X[NV], A[NV], Ap[NV];
#pragma GCC unroll 4
    for (int v = 0; v < NV; ++v) { X[v] = _mm512_setzero_si512(); Ap[v] = _mm512_setzero_si512(); }
    const uint32_t steps = n + g;
    for (uint32_t kk = 0; kk < steps; ++kk) {
        __m512i prev = _mm512_set1_epi32(static_cast<int>(base[kk]));
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {                                         // lane 0 <- prev's lane 15, lane j <- X lane j-1
            A[v] = _mm512_alignr_epi32(X[v], prev, 15);
            prev = X[v];
        }
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m512i y = rho_addmod16(rho_addmod16(rho_shoup16(A[v], sw, sws, pv), rho_shoup16(Ap[v], tw, tws, pv), pv),
                                           rho_shoup16(X[v], ww, wws, pv), pv);
            Ap[v] = A[v];
            X[v] = y;
            _mm512_storeu_si512(sk + static_cast<size_t>(kk) * G + 16 * v, y);
        }
    }
}
template<int NV>
RHO_AVX512_FN void rho_writeout_avx512(uint32_t n, uint32_t* out, const uint32_t* sk, uint32_t G, uint32_t done, bool a_unit) {
    const uint32_t d_V = n + 1;
    __m512i idx[NV];
    for (int v = 0; v < NV; ++v) {
        alignas(64) int32_t t[16];
        for (int w = 0; w < 16; ++w) t[w] = static_cast<int32_t>((16 * v + w) * (G + 1));
        idx[v] = _mm512_load_si512(t);
    }
    const __m512i rev = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (uint32_t k = 0; k < d_V; ++k) {
        uint32_t* row = out + static_cast<size_t>(k) * d_V;
        const uint32_t* src = sk + static_cast<size_t>(k) * G;
#pragma GCC unroll 4
        for (int v = 0; v < NV; ++v) {
            const __m512i y = _mm512_i32gather_epi32(idx[v], src, 4);
            if (a_unit) _mm512_storeu_si512(row + done + 1 + 16 * v, y);
            else _mm512_storeu_si512(row + (n - done - 1) - 16 * v - 15, _mm512_permutexvar_epi32(rev, y));
        }
    }
}
#endif   // RHO_HAVE_X86_SIMD

// The column steps of [U_a] / [U_b] on the SIMD path `path` (Avx2 / Avx512, available): `base` holds the column j0
// (d_V entries); every other column of `out` is written.  u32, p <= 2^31 (the caller checks).
void rho_steps_simd(const engine::ModCtx<uint32_t>& ctx, RhoFillPath path, uint32_t s1, uint32_t t1, uint32_t nw1,
                    uint32_t t, uint32_t w, uint32_t n, bool a_unit, const uint32_t* base_col, uint32_t* out,
                    std::vector<uint32_t>& work) {
#if RHO_HAVE_X86_SIMD
    const uint32_t d_V = n + 1;
    const uint32_t G = path == RhoFillPath::Avx512 ? 64u : 32u;              // 4 vectors of 16 / 8 lanes
    const size_t blen = static_cast<size_t>(d_V) + G;
    work.assign(blen + blen * G, 0u);                                          // base (zero padded) | skew buffer
    uint32_t* base = work.data();
    uint32_t* sk = base + blen;
    std::copy(base_col, base_col + d_V, base);
    const uint32_t sw[2] = {s1, ctx.shoup_precompute(s1)}, tw[2] = {t1, ctx.shoup_precompute(t1)},
                   ww[2] = {nw1, ctx.shoup_precompute(nw1)};
    (void)t; (void)w;
    for (uint32_t done = 0; done < n;) {
        const uint32_t g = std::min<uint32_t>(G, n - done);
        if (path == RhoFillPath::Avx512) rho_steps_avx512<4>(ctx.p, sw, tw, ww, n, base, sk, g, G);
        else rho_steps_avx2<4>(ctx.p, sw, tw, ww, n, base, sk, g, G);
        for (uint32_t i = 0; i < g; ++i) {   // the scalar step's remainder check, lane by lane (Debug / Check builds)
            ENGINE_DEBUG_CHECK_MSG(ctx.mul(t, i == 0 ? base[n] : sk[static_cast<size_t>(n + i - 1) * G + i - 1]) ==
                                       ctx.mul(w, sk[static_cast<size_t>(n + i) * G + i]),
                                   "rho_n over Z/ell^k: non-zero remainder (division not exact).");
        }
        if (g == G) {
            if (path == RhoFillPath::Avx512) rho_writeout_avx512<4>(n, out, sk, G, done, a_unit);
            else rho_writeout_avx2<4>(n, out, sk, G, done, a_unit);
        } else {
            for (uint32_t k = 0; k < d_V; ++k) {
                uint32_t* row = out + static_cast<size_t>(k) * d_V;
                const uint32_t* sp = sk + static_cast<size_t>(k) * G;
                for (uint32_t i = 0; i < g; ++i) row[a_unit ? done + 1 + i : n - done - 1 - i] = sp[static_cast<size_t>(i) * (G + 1)];
            }
        }
        for (uint32_t k = 0; k < d_V; ++k) base[k] = sk[static_cast<size_t>(k + g - 1) * G + (g - 1)];   // lane g - 1
        done += g;
    }
#else
    (void)ctx; (void)path; (void)s1; (void)t1; (void)nw1; (void)t; (void)w; (void)n; (void)a_unit; (void)base_col;
    (void)out; (void)work;
    ENGINE_CHECK_MSG(false, "rho_steps_simd: no SIMD path in this build");
#endif
}

} // namespace

const char* to_text(RhoFillPath p) {
    switch (p) {
        case RhoFillPath::Auto: return "auto";
        case RhoFillPath::Scalar: return "scalar";
        case RhoFillPath::Avx2: return "avx2";
        case RhoFillPath::Avx512: return "avx512";
    }
    return "?";
}
bool rho_fill_path_available(RhoFillPath p) {
    switch (p) {
        case RhoFillPath::Auto: case RhoFillPath::Scalar: return true;
        case RhoFillPath::Avx2: return rho_cpu_has_avx2();
        case RhoFillPath::Avx512: return rho_cpu_has_avx512();
    }
    return false;
}
void rho_fill_env_check() { (void)rho_fill_env(); }

RhoFillPath rho_fill_path_resolved() {
    const int o = g_rho_fill_override.load(std::memory_order_relaxed);
    const RhoFillPath p = o >= 0 ? static_cast<RhoFillPath>(o) : rho_fill_env();
    if (p != RhoFillPath::Auto) return p;
    switch (engine::kernels::isa_auto(rho_cpu_has_avx2(), rho_cpu_has_avx512())) {   // (S24 CL-5) under the ENGINE_KERNEL ceiling
        case engine::cpu::IsaPath::Avx512: return RhoFillPath::Avx512;
        case engine::cpu::IsaPath::Avx2: return RhoFillPath::Avx2;
        default: return RhoFillPath::Scalar;
    }
}
ScopedRhoFillPath::ScopedRhoFillPath(RhoFillPath p) {
    ENGINE_CHECK_MSG(rho_fill_path_available(p), "ScopedRhoFillPath: that path is not available on this CPU / build");
    saved_ = g_rho_fill_override.exchange(static_cast<int>(p));
}
ScopedRhoFillPath::~ScopedRhoFillPath() { g_rho_fill_override.store(saved_); }
RhoFillCounts rho_fill_counts() {
    RhoFillCounts c;
    c.scalar = g_rho_fill_count[1].load();
    c.avx2 = g_rho_fill_count[2].load();
    c.avx512 = g_rho_fill_count[3].load();
    return c;
}
std::string rho_fill_effective_text() {
    const int o = g_rho_fill_override.load(std::memory_order_relaxed);
    const char* v = std::getenv("HECKE_RHO_SIMD");
    const RhoFillCounts c = rho_fill_counts();
    std::string s = std::string("path=") + to_text(rho_fill_path_resolved()) +
                    " source=" + (o >= 0 ? "scoped" : (v != nullptr ? "env" : "default"));
    if (v != nullptr) s += std::string(" (HECKE_RHO_SIMD=") + v + ")";
    s += std::string(" cpu=avx2:") + (rho_cpu_has_avx2() ? "1" : "0") + ",avx512:" + (rho_cpu_has_avx512() ? "1" : "0");
    s += " fills=avx512:" + std::to_string(c.avx512) + ",avx2:" + std::to_string(c.avx2) + ",scalar:" + std::to_string(c.scalar);
    return s;
}

// ---------------------------------------------------------------------------
// (S23 G2-T) rho_n over Z/ell^k straight into a row-major buffer, for the module route's term evaluation
// (HECKE_MODULE_EVAL; FlatMatrix.h).  The SAME recurrences, routes and per-entry operations as rho_ring_fill
// above -- the [U_a] / [U_b] step r_k = (s P[k] + t P[k-1] - w r_{k-1}) / u, the [Z] row, the [S] split and its
// row scaling -- with two changes that cannot move a value: (1) the Pascal row is the caller's (built once per
// (ell^k, n) by RingPascalRow, the same additions as ring_pascal_row); (2) the column steps of [U_a] / [U_b] are
// computed G at a time in a skewed wavefront (lane L at row k - L + 1), so that G independent dependency chains
// run interleaved instead of one latency-bound chain.  Every entry is the same expression of the same canonical
// residues, so every byte equals rho_ring_fill's (HeckeModuleStreamValidation (V) compares the two).
// ---------------------------------------------------------------------------
namespace {

template<class Elt>
void rho_ring_fill_rows(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d, uint32_t n,
                        const std::vector<Elt>& bin, Elt* out, std::vector<Elt>& work, uint32_t depth) {
    const uint32_t d_V = n + 1;
    auto at = [d_V](uint32_t k, uint32_t j) -> size_t { return static_cast<size_t>(k) * d_V + j; };
    const bool a_unit = ctx.valuation(a) == 0, b_unit = ctx.valuation(b) == 0;
    if (a_unit || b_unit) {
        const Elt u = a_unit ? a : b, w = a_unit ? c : d;       // the divisor u + w y
        const Elt s = a_unit ? b : a, t = a_unit ? d : c;       // the multiplier s + t y
        const Elt uinv = ctx.inv(u);
        const std::vector<Elt> pu = ring_power_row<Elt>(ctx, u, d_V), pw = ring_power_row<Elt>(ctx, w, d_V);
        constexpr uint32_t G = 8;                                   // lanes of the wavefront
        const uint32_t j0 = a_unit ? 0 : n;
        // (S23 G2-T2) the dispatched SIMD column steps (u32, p <= 2^31: the fused gate below and Shoup's 2p <= 2^32),
        // the same canonical residues as the scalar loop (see rho_steps_simd)
        if constexpr (std::is_same_v<Elt, uint32_t>) {
            const RhoFillPath path = rho_fill_path_resolved();
            if (path != RhoFillPath::Scalar && ctx.shoup_available()) {   // n = 0: no column step
                std::vector<uint32_t> col(d_V);
                for (uint32_t k = 0; k < d_V; ++k) col[k] = ctx.mul(bin[k], ctx.mul(pu[n - k], pw[k]));
                for (uint32_t k = 0; k < d_V; ++k) out[at(k, j0)] = col[k];
                const uint32_t s1 = ctx.mul(s, uinv), t1 = ctx.mul(t, uinv), nw1 = ctx.neg(ctx.mul(w, uinv));
                rho_steps_simd(ctx, path, s1, t1, nw1, t, w, n, a_unit, col.data(), out, work);
                g_rho_fill_count[static_cast<int>(path)].fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        g_rho_fill_count[static_cast<int>(RhoFillPath::Scalar)].fetch_add(1, std::memory_order_relaxed);
        work.assign(static_cast<size_t>(G + 1) * d_V, Elt(0));      // lane 0 = the last finished column
        Elt* base = work.data();
        for (uint32_t k = 0; k < d_V; ++k) base[k] = ctx.mul(bin[k], ctx.mul(pu[n - k], pw[k]));
        for (uint32_t k = 0; k < d_V; ++k) out[at(k, j0)] = base[k];
        // (G2-T) the step with uinv folded in: r_k = s' P[k] + t' P[k-1] + (-w') r_{k-1}, s' = s uinv, t' = t uinv,
        // w' = w uinv, as ONE exact sum of three products reduced once (Accum: u64 on u32, u128 on u64) when
        // 3 (p - 1)^2 fits the accumulator (p <= 2^31 on u32, p < 2^62 on u64) -- the residue of the same ring
        // element as rho_ring_fill's (s P[k] + t P[k-1] - w r_{k-1}) uinv, so the same canonical value; else the
        // step exactly as rho_ring_fill writes it.  (G2-T audit F5) The u32 gate is conservative: the exact limit is
        // 3 (p - 1)^2 < 2^64, p <= ~2.48e9 (HeckeModuleStreamValidation (V) checks Z/3^20 and Z/65521^2 above it).
        // On u64 ctx.reduce(u128) is a library `%`; a Montgomery / fp_reduce_u64 step is left to a later package.
        using Acc = typename engine::ModCtx<Elt>::Accum;
        const bool fused = std::is_same_v<Elt, uint32_t> ? static_cast<uint64_t>(ctx.p) <= (uint64_t(1) << 31)
                                                         : static_cast<uint64_t>(ctx.p) < (uint64_t(1) << 62);
        const Elt s1 = ctx.mul(s, uinv), t1 = ctx.mul(t, uinv), nw1 = ctx.neg(ctx.mul(w, uinv));
        for (uint32_t done = 0; done < n;) {
            const uint32_t g = std::min<uint32_t>(G, n - done);
            // wavefront: at time kk, lane L (column step done + L - 1) computes its row kk + 1 - L, which reads
            // lane L - 1's rows k and k - 1 (finished at times kk - 1, kk - 2) and its own row k - 1 (time kk - 1)
            for (uint32_t kk = 0; kk + 1 < d_V + g; ++kk) {
                for (uint32_t L = 1; L <= g; ++L) {
                    if (kk + 1 < L) break;                          // lanes L.. have not started at this kk
                    const uint32_t k = kk + 1 - L;
                    if (k > n) continue;                            // lane L has finished
                    const Elt* src = base + static_cast<size_t>(L - 1) * d_V;
                    Elt* dst = base + static_cast<size_t>(L) * d_V;
                    const Elt cur_prev = k > 0 ? src[k - 1] : Elt(0);
                    const Elt r_prev = k > 0 ? dst[k - 1] : Elt(0);
                    if (fused) {
                        dst[k] = ctx.reduce(static_cast<Acc>(s1) * src[k] + static_cast<Acc>(t1) * cur_prev +
                                            static_cast<Acc>(nw1) * r_prev);
                        continue;
                    }
                    // (s + t y) P = (u + w y) R:  r_k = (s P[k] + t P[k-1] - w r_{k-1}) / u   (rho_ring_fill's step)
                    const Elt q = ctx.add(ctx.mul(s, src[k]), ctx.mul(t, cur_prev));
                    dst[k] = ctx.mul(ctx.sub(q, ctx.mul(w, r_prev)), uinv);
                }
            }
            for (uint32_t L = 1; L <= g; ++L) {
                ENGINE_DEBUG_CHECK_MSG(ctx.mul(t, base[static_cast<size_t>(L - 1) * d_V + n]) ==
                                           ctx.mul(w, base[static_cast<size_t>(L) * d_V + n]),
                                       "rho_n over Z/ell^k: non-zero remainder (division not exact).");
            }
            for (uint32_t k = 0; k < d_V; ++k) {
                Elt* row = out + static_cast<size_t>(k) * d_V;
                for (uint32_t L = 1; L <= g; ++L) {
                    const uint32_t step = done + L - 1;
                    const uint32_t j = a_unit ? step + 1 : n - step - 1;
                    row[j] = base[static_cast<size_t>(L) * d_V + k];
                }
            }
            std::copy(base + static_cast<size_t>(g) * d_V, base + static_cast<size_t>(g + 1) * d_V, base);
            done += g;
        }
        return;
    }
    if (a == 0 && b == 0) {
        // [Z] the first row vanishes: column j is (c y)^(n-j) (d y)^j = c^(n-j) d^j y^n; every other row is zero
        // (the only route that does not write every entry -- the caller does not zero `out`).
        std::fill(out, out + static_cast<size_t>(d_V) * d_V, Elt(0));
        const std::vector<Elt> pc = ring_power_row<Elt>(ctx, c, d_V), pd = ring_power_row<Elt>(ctx, d, d_V);
        for (uint32_t j = 0; j < d_V; ++j) out[at(n, j)] = ctx.mul(pc[n - j], pd[j]);
        return;
    }
    // [S] as rho_ring_fill: g = diag(ell, 1) g', row k of rho_n(g') scaled by ell^(n-k).
    ENGINE_CHECK_MSG(depth < ctx.k, "rho_n over Z/ell^k: split recursion deeper than k");
    rho_ring_fill_rows<Elt>(ctx, static_cast<Elt>(a / ctx.ell), static_cast<Elt>(b / ctx.ell), c, d, n, bin, out, work,
                            depth + 1);
    Elt scale = 1;                  // ell^(n-kk), built from kk = n down to 0
    for (uint32_t kk = d_V; kk-- > 0;) {
        if (scale != 1) {
            for (uint32_t j = 0; j < d_V; ++j) out[at(kk, j)] = ctx.mul(out[at(kk, j)], scale);
        }
        scale = ctx.mul(scale, ctx.ell);
    }
}

} // namespace

template<class Elt>
std::vector<Elt> RingPascalRow(const engine::ModCtx<Elt>& ctx, uint32_t n_sym) {
    ENGINE_CHECK_MSG(!ctx.is_field(), "RingPascalRow: composite-only (Z/ell^k)");
    return ring_pascal_row<Elt>(ctx, n_sym);
}

template<class Elt>
void ComputeSymmetricPowerRowsRing(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d, uint32_t n_sym,
                                   const std::vector<Elt>& pascal, std::vector<Elt>& out, std::vector<Elt>& work) {
    ENGINE_CHECK_MSG(!ctx.is_field(),
                     "ComputeSymmetricPowerRowsRing: composite-only (Z/ell^k); F_ell uses ComputeSymmetricPowerMatrixFM");
    ENGINE_CHECK_MSG(a < ctx.p && b < ctx.p && c < ctx.p && d < ctx.p,
                     "ComputeSymmetricPowerRowsRing: entries must be canonical residues mod ell^k");
    ENGINE_CHECK_MSG(pascal.size() == static_cast<size_t>(n_sym) + 1,
                     "ComputeSymmetricPowerRowsRing: the Pascal row must be RingPascalRow(ctx, n_sym)");
    const uint32_t d_V = n_sym + 1;
    out.resize(static_cast<size_t>(d_V) * d_V);   // every route writes every entry ([Z] zeroes its rows itself)
    rho_ring_fill_rows<Elt>(ctx, a, b, c, d, n_sym, pascal, out.data(), work, 0);
}

template<class Elt>
FlatMatrixT<Elt> ComputeSymmetricPowerMatrixRingOracle(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d,
                                                       uint32_t n_sym) {
    ENGINE_CHECK_MSG(!ctx.is_field(), "ComputeSymmetricPowerMatrixRingOracle: composite-only (Z/ell^k)");
    const uint32_t n = n_sym, d_V = n_sym + 1;
    // Pascal's triangle mod ell^k, rows 0 .. n, by additions.
    std::vector<std::vector<Elt>> C(d_V);
    for (uint32_t r = 0; r <= n; ++r) {
        C[r].assign(static_cast<size_t>(r) + 1, Elt(1));
        for (uint32_t j = 1; j < r; ++j) C[r][j] = ctx.add(C[r - 1][j - 1], C[r - 1][j]);
    }
    const std::vector<Elt> pa = ring_power_row<Elt>(ctx, a, d_V), pb = ring_power_row<Elt>(ctx, b, d_V);
    const std::vector<Elt> pc = ring_power_row<Elt>(ctx, c, d_V), pd = ring_power_row<Elt>(ctx, d, d_V);
    std::vector<Elt> out(static_cast<size_t>(d_V) * d_V, Elt(0));
    for (uint32_t j = 0; j <= n; ++j) {
        const uint32_t m = n - j;                 // (a + c y)^m (b + d y)^j
        for (uint32_t k = 0; k <= n; ++k) {
            Elt acc = 0;
            const uint32_t t_lo = (k > m) ? k - m : 0, t_hi = std::min(j, k);
            for (uint32_t t = t_lo; t <= t_hi; ++t) {
                // y^(k-t) from (a + c y)^m, y^t from (b + d y)^j
                const Elt left = ctx.mul(C[m][k - t], ctx.mul(pa[m - (k - t)], pc[k - t]));
                const Elt right = ctx.mul(C[j][t], ctx.mul(pb[j - t], pd[t]));
                acc = ctx.add(acc, ctx.mul(left, right));
            }
            out[static_cast<size_t>(k) * d_V + j] = acc;
        }
    }
    return FlatMatrixT<Elt>(d_V, d_V, ctx.p, out);
}

template<class Elt>
std::vector<Elt> change_of_basis_fm(const std::vector<FlatMatrixT<Elt>>& basis, const FlatMatrixT<Elt>& vector) {
    if (basis.empty() || vector.get_cols() != 1 || vector.get_rows() != basis[0].get_rows()) {
        throw std::runtime_error("Invalid input for change of basis.");
    }
    for (uint32_t j = 0; j < basis.size(); ++j) {
        if (basis[j].get_cols() != 1 || basis[j].get_rows() != basis[0].get_rows()) {
            throw std::runtime_error("Basis vectors have to be column vectors of same size.");
        }
    }

    uint32_t num_rows = basis[0].get_rows();
    uint32_t num_basis = static_cast<uint32_t>(basis.size());
    FlatMatrixT<Elt> augmented_matrix(num_rows, num_basis + 1, basis[0].get_p());
    augmented_matrix.populate_flat_data();
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < num_basis; ++j) {
            augmented_matrix.set_entry(i, j, basis[j].get_entry(i, 0));
        }
        augmented_matrix.set_entry(i, num_basis, vector.get_entry(i, 0));
    }

    std::vector<Elt> coefficients(num_basis, 0);
    const auto& pivots = augmented_matrix.get_pivot_columns();
    const std::vector<Elt>& rref = augmented_matrix.get_row_echelon_form_data();   // flat row-major
    const size_t aug_cols = static_cast<size_t>(num_basis) + 1;
    for (uint32_t i = 0; i < pivots.size(); i++) {
        if (pivots[i] >= num_basis) {
            throw std::runtime_error("Vector is not in image of basis.");
        }
        coefficients[pivots[i]] = rref[i * aug_cols + num_basis];
    }

    return coefficients;
}

template<class Elt>
std::vector<std::vector<Elt>> change_of_basis_fm(const std::vector<FlatMatrixT<Elt>>& basis, const std::vector<FlatMatrixT<Elt>>& vectors) {
    std::vector<std::vector<Elt>> result;
    for (const FlatMatrixT<Elt>& v : vectors) {
        result.push_back(change_of_basis_fm(basis, v));
    }
    return result;
}

// --- Multi-RHS change of basis (matrix-engine Phase 1, item 7.1) ----------
//
// change_of_basis_fm(basis, v) rebuilds the augmented matrix [basis | v] and
// re-runs a full Gauss-Jordan elimination on EVERY call, although its first
// |basis| columns are the same every time.  This overload solves all
// right-hand sides in ONE elimination of [basis | v_1 ... v_m].
//
// BIT-IDENTITY (why the result is the same object, not merely a mathematically
// equal one).  _gaussian_elimination_core scans columns strictly left to
// right, so:
//   (1) the pivot search over the first |basis| columns, the row swaps it
//       performs and the row multiples it subtracts are functions of those
//       columns alone -- appending further columns on the right cannot change
//       them.  Hence the pivot list restricted to the basis block, and the row
//       order, are the same as in the single-RHS run.
//   (2) every one of those row operations is applied to ALL columns from the
//       pivot column rightwards, so right-hand column |basis| + t receives
//       exactly the operations the single-RHS run applies to its column
//       |basis|.
//   (3) a right-hand side that lies in span(basis) is annihilated outside the
//       pivot rows by the time its own column is scanned, so it never becomes
//       a pivot column and never triggers a further row operation.  (If one
//       does become a pivot column, that RHS is not in the span and we throw,
//       just as the single-RHS overload does.  Only in that -- rejected --
//       case could a right-hand column influence a column to its right.)
// The coordinates are in any case unique whenever the basis columns are
// independent; the caller's residual certificate re-verifies them against the
// unfactored data.
template<class Elt>
std::vector<std::vector<Elt>> change_of_basis_fm_multi(const std::vector<FlatMatrixT<Elt>>& basis,
                                                       const std::vector<FlatMatrixT<Elt>>& vectors) {
    if (basis.empty()) {
        throw std::runtime_error("Invalid input for change of basis.");
    }
    const uint32_t num_rows = basis[0].get_rows();
    const uint32_t num_basis = static_cast<uint32_t>(basis.size());
    const Elt prime = basis[0].get_p();
    for (uint32_t j = 0; j < num_basis; ++j) {
        if (basis[j].get_cols() != 1 || basis[j].get_rows() != num_rows) {
            throw std::runtime_error("Basis vectors have to be column vectors of same size.");
        }
    }
    const uint32_t num_rhs = static_cast<uint32_t>(vectors.size());
    for (uint32_t t = 0; t < num_rhs; ++t) {
        if (vectors[t].get_cols() != 1 || vectors[t].get_rows() != num_rows) {
            throw std::runtime_error("Invalid input for change of basis.");
        }
    }

    std::vector<std::vector<Elt>> result(num_rhs, std::vector<Elt>(num_basis, 0));
    if (num_rhs == 0) return result;

    // One augmented matrix [basis | vectors], built in a single pass (no
    // set_entry loop: that would re-run the scalar-flag scan per entry).
    const uint32_t num_cols = num_basis + num_rhs;
    std::vector<Elt> aug(static_cast<size_t>(num_rows) * num_cols, 0);
    for (uint32_t i = 0; i < num_rows; ++i) {
        const size_t row_off = static_cast<size_t>(i) * num_cols;
        for (uint32_t j = 0; j < num_basis; ++j) {
            aug[row_off + j] = basis[j].get_entry(i, 0);
        }
        for (uint32_t t = 0; t < num_rhs; ++t) {
            aug[row_off + num_basis + t] = vectors[t].get_entry(i, 0);
        }
    }
    const FlatMatrixT<Elt> augmented_matrix(num_rows, num_cols, prime, aug);

    const std::vector<uint32_t>& pivots = augmented_matrix.get_pivot_columns();
    const std::vector<Elt>& rref = augmented_matrix.get_row_echelon_form_data();   // flat row-major
    for (uint32_t i = 0; i < pivots.size(); ++i) {
        if (pivots[i] >= num_basis) {
            throw std::runtime_error(
                "Vector is not in image of basis (right-hand side " +
                std::to_string(pivots[i] - num_basis) + ").");
        }
        const size_t row_off = static_cast<size_t>(i) * num_cols;
        for (uint32_t t = 0; t < num_rhs; ++t) {
            result[t][pivots[i]] = rref[row_off + num_basis + t];
        }
    }
    return result;
}

// --- Block-layer GEMM entry point (matrix-engine Phase 2, Stage B) --------
//
// BlockFlatMatrixT<Elt>::operator* accumulates its per-block partial sums in a
// plain std::vector<Elt>, not in a FlatMatrixT, so it cannot reach the engine
// through operator* without materialising one temporary FlatMatrix per k. This
// is the direct route: acc += A*B straight into the caller's buffer with
// GemmOp::Add. It lives HERE, next to the bridge helpers, so that the engine
// headers stay confined to this translation unit.
template<class Elt>
bool flat_gemm_accumulate_engine(const FlatMatrixT<Elt>& A, const FlatMatrixT<Elt>& B,
                                 std::vector<Elt>& acc, uint32_t acc_rows, uint32_t acc_cols) {
    Elt scalar_val = 0;
    // A scalar-stored (alpha*I) operand owns no buffer; the caller keeps its
    // legacy route for those, which is also the cheaper one (a scal, not a gemm).
    if (A.is_currently_scalar(scalar_val) || B.is_currently_scalar(scalar_val)) return false;
    const Elt p = A.get_p();
    if (!engine_path<Elt>(p)) return false;
    if (acc_rows == 0 || acc_cols == 0 || A.get_cols() == 0) return false;
    ENGINE_CHECK_MSG(A.get_rows() == acc_rows && B.get_cols() == acc_cols &&
                     A.get_cols() == B.get_rows() && B.get_p() == p,
                     "flat_gemm_accumulate_engine: shape or modulus mismatch");
    ENGINE_CHECK_MSG(acc.size() == static_cast<size_t>(acc_rows) * acc_cols,
                     "flat_gemm_accumulate_engine: accumulator size mismatch");

    const engine::DenseBuf<Elt> Ab = dense_wrap<Elt>(A.get_flat_data_unsafe().data(),
                                                     A.get_rows(), A.get_cols());
    const engine::DenseBuf<Elt> Bb = dense_wrap<Elt>(B.get_flat_data_unsafe().data(),
                                                     B.get_rows(), B.get_cols());
    engine::DenseBuf<Elt> Cb = dense_wrap<Elt>(acc.data(), acc_rows, acc_cols);
    engine::kernels::gemm<Elt>(engine_ctx<Elt>(p), engine::cview(Ab), engine::cview(Bb),
                               engine::view(Cb), engine::kernels::GemmOp::Add);
    return true;
}

// Defect #32 (matrix-engine Phase 0): factor_basis_fm was deleted here. It had
// zero callers anywhere in the repository; factor_basis_block (Matrix.cpp) is
// the block-layer routine that is actually used.

// ===========================================================================
// Explicit instantiations (Stage S13): uint32_t (primary regime, alias
// FlatMatrix) and uint64_t (extended regime, alias FlatMatrix64).
// ===========================================================================
template class FlatMatrixT<uint32_t>;
template class FlatMatrixT<uint64_t>;

#define INSTANTIATE_FLATMATRIX_FREE(Elt) \
    template FlatMatrixT<Elt> operator*(Elt, const FlatMatrixT<Elt>&); \
    template FlatMatrixT<Elt> ComputeSymmetricPowerMatrixFM(const FlatMatrixT<Elt>&, uint32_t); \
    template FlatMatrixT<Elt> ComputeSymmetricPowerMatrixFM_reference(const FlatMatrixT<Elt>&, uint32_t); \
    template FlatMatrixT<Elt> ComputeSymmetricPowerMatrixRing(const engine::ModCtx<Elt>&, Elt, Elt, Elt, Elt, uint32_t, RhoRingRoute*); \
    template FlatMatrixT<Elt> ComputeSymmetricPowerMatrixRingOracle(const engine::ModCtx<Elt>&, Elt, Elt, Elt, Elt, uint32_t); \
    template std::vector<Elt> RingPascalRow(const engine::ModCtx<Elt>&, uint32_t); \
    template void ComputeSymmetricPowerRowsRing(const engine::ModCtx<Elt>&, Elt, Elt, Elt, Elt, uint32_t, const std::vector<Elt>&, \
                                                std::vector<Elt>&, std::vector<Elt>&); \
    template std::vector<Elt> change_of_basis_fm(const std::vector<FlatMatrixT<Elt>>&, const FlatMatrixT<Elt>&); \
    template std::vector<std::vector<Elt>> change_of_basis_fm(const std::vector<FlatMatrixT<Elt>>&, const std::vector<FlatMatrixT<Elt>>&); \
    template std::vector<std::vector<Elt>> change_of_basis_fm_multi(const std::vector<FlatMatrixT<Elt>>&, const std::vector<FlatMatrixT<Elt>>&); \
    template bool flat_gemm_accumulate_engine(const FlatMatrixT<Elt>&, const FlatMatrixT<Elt>&, std::vector<Elt>&, uint32_t, uint32_t);

INSTANTIATE_FLATMATRIX_FREE(uint32_t)
INSTANTIATE_FLATMATRIX_FREE(uint64_t)
#undef INSTANTIATE_FLATMATRIX_FREE
