// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BlockFlatMatrixValidation.cpp
//
// ===========================================================================
// matrix-engine Phase 3 -- re-pointed and shrunk.
//
// `BlockFlatMatrixT` is now the LEGACY ARM ONLY (see the banner in
// BlockFlatMatrix.h): the live block model is `engine::BlockMat` / `engine::blk`,
// covered structurally by BlockMatValidation and algorithmically by
// BlockEliminationValidation.  The ~20 members that existed only so that THIS
// file could exercise them -- CreateZero/CreateIdentity, swap, transpose,
// operator+/-/scalar/==/!=, get_block_rref_intermediate_form,
// get_block_pivot_coords_in_rref, get_true_row_echelon_form_as_block_matrix,
// block_inverse/block_kernel/block_image and the *_for_test getters -- were
// deleted together with the checks below that used them, because
// BlockEliminationValidation already covers the same properties ENTRYWISE on
// the new model (block RREF, kernel, image, factor_basis, solve, both element
// widths, seven primes, non-uniform and zero-width block grids, and the
// structural I / alpha*I / +-M / I-M pivot paths), and BlockMatValidation covers
// the container itself (dedup/refcount, CSR order, views, transpose, add/sub/
// scale, gemm, builder, L1 layout).
//
// What remains here is exactly the surface the legacy arm still calls, checked
// ENTRYWISE against the independent `FlatMatrix` implementation of the same
// operation.  That entrywise requirement is plan Sec. D5b: the old
// `verify_flat_kernel_properties` / `verify_flat_image_properties` checked only
// size, M*K == 0, rank and MUTUAL SPAN INCLUSION, so a rewrite that changed the
// kernel-basis convention would have passed this suite while silently changing
// every shipped cohomology-level matrix.  Nothing below compares by span.
//
// The oracle is `FlatMatrix`, which is where D5b's conventions are written down
// (leftmost pivot column; one kernel vector per free column in ascending column
// index with a 1 at the free column and -RREF[i][j_free] at the i-th pivot
// column; image = the ORIGINAL columns at the pivot indices, ascending).
// ===========================================================================

#include "../pch.h" // If using precompiled headers
#include "BlockFlatMatrixValidation.h"
#include "BlockFlatMatrix.h"
#include "../FlatMatrix.h"      // the entrywise oracle
#include "../ModArithmetic.h"   // For modular operations
#include "../helper.h"          // For offset_string

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_verbosity = 0;
uint32_t g_total = 0;
uint32_t g_passed = 0;

bool check(const std::string& name, bool condition, const std::string& detail = "") {
    ++g_total;
    if (condition) {
        ++g_passed;
        if (g_verbosity > 1) std::cout << "  [PASS] " << name << std::endl;
    } else {
        std::cout << "  [FAIL] " << name << (detail.empty() ? "" : " - ") << detail << std::endl;
    }
    return condition;
}

template<class Fn>
bool check_throws(const std::string& name, Fn body, const char* expected) {
    ++g_total;
    std::string actual = "none";
    try { body(); }
    catch (const std::out_of_range&)    { actual = "std::out_of_range"; }
    catch (const std::invalid_argument&){ actual = "std::invalid_argument"; }
    catch (const std::runtime_error&)   { actual = "std::runtime_error"; }
    catch (const std::logic_error&)     { actual = "std::logic_error"; }
    catch (...)                         { actual = "unknown"; }
    const bool ok = (actual == expected);
    if (ok) {
        ++g_passed;
        if (g_verbosity > 1) std::cout << "  [PASS] " << name << " (threw " << expected << ")" << std::endl;
    } else {
        std::cout << "  [FAIL] " << name << " (threw " << actual << ", expected " << expected << ")" << std::endl;
    }
    return ok;
}

void section(const std::string& name, uint32_t p) {
    if (g_verbosity > 0)
        std::cout << "\n--- Testing BlockFlatMatrix: " << name << " (p=" << p << ") ---" << std::endl;
}

// --- Deterministic LCG (unchanged: the historical shapes/seeds are preserved
//     so that the large randomized inputs are literally the same matrices this
//     suite has always run on). ------------------------------------------
struct SimpleLCG {
    uint64_t current_val;
    static constexpr uint64_t A = 1664525;
    static constexpr uint64_t C = 1013904223;

    explicit SimpleLCG(uint64_t seed) : current_val(seed) {}

    uint32_t next_uint32() {
        current_val = (A * current_val + C);
        return static_cast<uint32_t>(current_val >> 32);
    }
    uint32_t next_mod(uint32_t max_val) { return max_val == 0 ? 0 : next_uint32() % max_val; }
    uint32_t next_range(uint32_t min_val, uint32_t max_val) {
        if (min_val > max_val) std::swap(min_val, max_val);
        if (min_val == max_val) return min_val;
        return min_val + next_mod(max_val - min_val + 1);
    }
};

std::vector<uint32_t> generate_block_dims(SimpleLCG& prng, uint32_t target_total_dim,
                                          uint32_t min_block_dim, uint32_t max_block_dim,
                                          uint32_t max_num_blocks) {
    std::vector<uint32_t> block_dims;
    uint32_t sum = 0, guard = 0;
    if (target_total_dim == 0) return {};
    uint32_t lo = min_block_dim == 0 ? 1 : min_block_dim;
    uint32_t hi = std::max(lo, max_block_dim);
    while (sum < target_total_dim && block_dims.size() < max_num_blocks &&
           guard < 2 * max_num_blocks + target_total_dim) {
        ++guard;
        uint32_t b = prng.next_range(lo, hi);
        if (sum + b > target_total_dim) b = target_total_dim - sum;
        if (b > 0) { block_dims.push_back(b); sum += b; }
        else if (sum == target_total_dim) break;
    }
    if (sum < target_total_dim) {
        if (block_dims.empty()) block_dims.push_back(target_total_dim - sum);
        else block_dims.back() += target_total_dim - sum;
    }
    return block_dims;
}

// --- Entrywise comparison helpers.  NEVER by span (plan Sec. D5b). --------

bool same_entries(const FlatMatrix& a, const FlatMatrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (uint32_t r = 0; r < a.get_rows(); ++r)
        for (uint32_t c = 0; c < a.get_cols(); ++c)
            if (a.get_entry(r, c) != b.get_entry(r, c)) return false;
    return true;
}

// A list of BlockFlatMatrix column vectors equals a list of FlatMatrix column
// vectors: same count, same order, same entries.  This is the D5b check the old
// verify_flat_kernel_properties / verify_flat_image_properties never made.
bool same_column_lists(const std::vector<BlockFlatMatrix>& bfm_cols,
                       const std::vector<FlatMatrix>& fm_cols) {
    if (bfm_cols.size() != fm_cols.size()) return false;
    for (size_t t = 0; t < fm_cols.size(); ++t) {
        const FlatMatrix flat = bfm_cols[t].to_flat_matrix();
        if (flat.get_cols() != 1 || fm_cols[t].get_cols() != 1) return false;
        if (flat.get_rows() != fm_cols[t].get_rows()) return false;
        for (uint32_t i = 0; i < flat.get_rows(); ++i)
            if (flat.get_entry(i, 0) != fm_cols[t].get_entry(i, 0)) return false;
    }
    return true;
}

// --- Building a (BlockFlatMatrix, FlatMatrix) twin pair -------------------

// Fill `B` block by block from the row-major dense content of `F`.
void fill_from_dense(BlockFlatMatrix& B, const FlatMatrix& F, uint32_t p) {
    uint32_t r0 = 0;
    for (uint32_t br = 0; br < B.get_num_block_rows(); ++br) {
        const uint32_t R = B.get_block_row_dim(br);
        uint32_t c0 = 0;
        for (uint32_t bc = 0; bc < B.get_num_block_cols(); ++bc) {
            const uint32_t C = B.get_block_col_dim(bc);
            if (R > 0 && C > 0) {
                std::vector<uint32_t> buf(static_cast<size_t>(R) * C);
                bool nonzero = false;
                for (uint32_t i = 0; i < R; ++i)
                    for (uint32_t j = 0; j < C; ++j) {
                        const uint32_t v = F.get_entry(r0 + i, c0 + j);
                        buf[static_cast<size_t>(i) * C + j] = v;
                        if (v != 0) nonzero = true;
                    }
                if (nonzero) B.set_block_entry(br, bc, FlatMatrix(R, C, p, buf));
            }
            c0 += C;
        }
        r0 += R;
    }
}

// ===========================================================================
// one_case -- every surviving BlockFlatMatrix capability, entrywise against
// the FlatMatrix implementation of the same operation.
// ===========================================================================
void one_case(const std::string& tag, const BlockFlatMatrix& B_in, const FlatMatrix& F, uint32_t p) {
    // (1) content: to_flat_matrix and the per-entry accessor
    {
        const BlockFlatMatrix& B = B_in;
        check(tag + " to_flat_matrix entrywise == FlatMatrix twin", same_entries(B.to_flat_matrix(), F));
        bool ok = B.get_total_rows() == F.get_rows() && B.get_total_cols() == F.get_cols();
        for (uint32_t r = 0; ok && r < F.get_rows(); ++r)
            for (uint32_t c = 0; c < F.get_cols(); ++c)
                if (B.get_entry(r, c) != F.get_entry(r, c)) { ok = false; break; }
        check(tag + " get_entry entrywise == FlatMatrix twin", ok);
        check(tag + " is_zero == FlatMatrix::is_zero", B.is_zero() == F.is_zero());
    }

    // (2) rank and the true scalar RREF, entrywise, plus the pivot list.
    //     D5b: rows emitted one per definitive pivot in ascending pivot-column
    //     order, zero rows at the bottom -- the ordering factor_basis_block
    //     indexes arithmetically.
    {
        const BlockFlatMatrix B = B_in;                  // fresh RREF cache
        const uint32_t rank_b = B.flat_rank();
        const uint32_t rank_f = F.rank();
        check(tag + " flat_rank == FlatMatrix::rank", rank_b == rank_f,
              std::to_string(rank_b) + " vs " + std::to_string(rank_f));

        const BlockFlatMatrix B2 = B_in;
        auto [rref_b, pivots_b] = B2.get_true_row_echelon_form_as_flat_matrix();
        const FlatMatrix rref_f = F.get_row_echelon_form_as_flat_matrix();
        check(tag + " true RREF entrywise == FlatMatrix RREF", same_entries(rref_b, rref_f));

        const std::vector<uint32_t>& piv_f = F.get_pivot_columns();
        bool ok = pivots_b.size() == piv_f.size();
        for (size_t i = 0; ok && i < piv_f.size(); ++i)
            ok = (pivots_b[i].first == i) && (pivots_b[i].second == piv_f[i]);
        check(tag + " true RREF pivots (row i, ascending pivot column) == FlatMatrix", ok);
    }

    // (3) kernel basis -- ENTRYWISE, one vector per free column in ascending
    //     column index (D5b's root convention).
    {
        const BlockFlatMatrix B = B_in;
        check(tag + " flat_kernel entrywise == FlatMatrix::kernel",
              same_column_lists(B.flat_kernel(), F.kernel()));
    }

    // (4) image basis -- ENTRYWISE, the ORIGINAL columns at the pivot indices.
    //     Skipped for a non-zero scalar (alpha*I) twin: FlatMatrix::image()'s
    //     scalar fast path returns e_j where MatrixT::image() and
    //     BlockFlatMatrixT::flat_image() return alpha*e_j (audit defect #34).
    //     That divergence is pinned deliberately by FlatMatrixValidation's Q11
    //     section, which compares all three classes on exactly that input; it
    //     must not be re-asserted here with the opposite sign.
    {
        uint32_t sv = 0;
        const bool scalar_twin = F.is_currently_scalar(sv) && sv != 0;
        if (!scalar_twin) {
            const BlockFlatMatrix B = B_in;
            check(tag + " flat_image entrywise == FlatMatrix::image",
                  same_column_lists(B.flat_image(), F.image()));
        }
    }

    // (5) invertibility and the flat inverse (square inputs only).
    if (F.get_rows() == F.get_cols()) {
        const BlockFlatMatrix B = B_in;
        const bool inv_b = B.is_invertible();
        check(tag + " is_invertible == FlatMatrix::is_invertible", inv_b == F.is_invertible());
        if (inv_b) {
            const BlockFlatMatrix B2 = B_in;
            check(tag + " flat_inverse entrywise == FlatMatrix::inverse",
                  same_entries(B2.flat_inverse(), F.inverse()));
        } else {
            const BlockFlatMatrix B2 = B_in;
            check_throws(tag + " flat_inverse on a singular matrix throws",
                         [&]() { (void)B2.flat_inverse(); }, "std::runtime_error");
        }
    }
    (void)p;
}

// Build a random rank-`K` m x n matrix as U*V, block it with `rd` x `cd`, and
// run one_case on it.  Historical shapes/seeds preserved.
void random_low_rank_case(const std::string& tag, uint64_t seed, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t min_b, uint32_t max_b, uint32_t p) {
    SimpleLCG prng(seed);
    const uint32_t max_r_blocks = (M > 0 && min_b > 0) ? (M / min_b + 5) : (M > 0 ? 1 : 0);
    const uint32_t max_c_blocks = (N > 0 && min_b > 0) ? (N / min_b + 5) : (N > 0 ? 1 : 0);
    const std::vector<uint32_t> rd = generate_block_dims(prng, M, min_b, max_b, max_r_blocks);
    const std::vector<uint32_t> cd = generate_block_dims(prng, N, min_b, max_b, max_c_blocks);

    uint32_t m = 0; for (uint32_t d : rd) m += d;
    uint32_t n = 0; for (uint32_t d : cd) n += d;
    if (m == 0 || n == 0 || K == 0 || K > std::min(m, n)) {
        if (g_verbosity > 0) std::cout << "  skipping " << tag << " (degenerate dimensions)" << std::endl;
        return;
    }

    FlatMatrix U(m, K, p), V(K, n, p);
    for (uint32_t r = 0; r < m; ++r) for (uint32_t c = 0; c < K; ++c) U.set_entry(r, c, prng.next_mod(p));
    for (uint32_t r = 0; r < K; ++r) for (uint32_t c = 0; c < n; ++c) V.set_entry(r, c, prng.next_mod(p));
    const FlatMatrix F = U * V;

    BlockFlatMatrix B(rd, cd, p);
    fill_from_dense(B, F, p);
    one_case(tag, B, F, p);
}

} // namespace

uint32_t BlockFlatMatrixValidation(int verbosity) {
    g_verbosity = verbosity;
    g_total = 0;
    g_passed = 0;

    const uint32_t p7 = 7;

    if (verbosity >= 0) std::cout << "Starting BlockFlatMatrixValidation (legacy arm; entrywise vs FlatMatrix)..." << std::endl;

    // =====================================================================
    // 1. Container: constructors, dimensions, per-entry access, dedup.
    // =====================================================================
    try {
        section("Constructors, dimensions, entry access, dedup", p7);

        BlockFlatMatrix empty(p7);
        check("Empty ctor: 0 block rows", empty.get_num_block_rows() == 0);
        check("Empty ctor: 0 block cols", empty.get_num_block_cols() == 0);
        check("Empty ctor: 0 total rows/cols", empty.get_total_rows() == 0 && empty.get_total_cols() == 0);
        check("Empty ctor: p carried", empty.get_p() == p7);
        check("Empty ctor: to_flat_matrix is 0x0",
              same_entries(empty.to_flat_matrix(), FlatMatrix::CreateZero(0, 0, p7)));

        BlockFlatMatrix zero_dims({ 0 }, { 0 }, p7);
        check("Zero-dim ctor: 1x1 block grid", zero_dims.get_num_block_rows() == 1 && zero_dims.get_num_block_cols() == 1);
        check("Zero-dim ctor: block dims are 0",
              zero_dims.get_block_row_dim(0) == 0 && zero_dims.get_block_col_dim(0) == 0);
        check("Zero-dim ctor: to_flat_matrix is 0x0",
              same_entries(zero_dims.to_flat_matrix(), FlatMatrix::CreateZero(0, 0, p7)));

        // Zero blocks, non-uniform dims.
        const std::vector<uint32_t> rd_z = { 1, 2 }, cd_z = { 3, 1 };
        BlockFlatMatrix zeros(rd_z, cd_z, p7);
        check("Zero-block ctor: total dims", zeros.get_total_rows() == 3 && zeros.get_total_cols() == 4);
        check("Zero-block ctor: is_zero", zeros.is_zero());
        check("Zero-block ctor: to_flat_matrix is the 3x4 zero matrix",
              same_entries(zeros.to_flat_matrix(), FlatMatrix::CreateZero(3, 4, p7)));

        // Block-data constructor with non-uniform block dimensions.
        const FlatMatrix fm11(1, 1, p7, { 1 }), fm12(1, 2, p7, { 2, 3 });
        const FlatMatrix fm21(2, 1, p7, { 4, 5 }), fm22(2, 2, p7, { 6, 0, 1, 2 });
        BlockFlatMatrix data({ 1, 2 }, { 1, 2 }, { { fm11, fm12 }, { fm21, fm22 } }, p7);
        const FlatMatrix data_twin(3, 3, p7, { 1, 2, 3, 4, 6, 0, 5, 1, 2 });
        check("Block-data ctor: get_entry(0,2)", data.get_entry(0, 2) == 3);
        check("Block-data ctor: get_entry(2,1)", data.get_entry(2, 1) == 1);
        one_case("[data ctor 3x3 p=7]", data, data_twin, p7);

        check_throws("get_block_row_dim out of range throws",
                     [&]() { (void)data.get_block_row_dim(99); }, "std::out_of_range");
        check_throws("get_block_col_dim out of range throws",
                     [&]() { (void)data.get_block_col_dim(99); }, "std::out_of_range");
        check_throws("get_entry out of range throws",
                     [&]() { (void)data.get_entry(99, 0); }, "std::out_of_range");

        // Deduplication: identical block content must be stored ONCE, so two
        // layout slots holding equal blocks must return the SAME reference.
        // (This is what the deleted get_unique_blocks_for_test() used to spell
        // out; BlockMatValidation checks the same property on BlockPool with
        // explicit refcounts.)
        {
            BlockFlatMatrix dedup({ 2, 2 }, { 2, 2 }, p7);
            const FlatMatrix blk(2, 2, p7, { 1, 2, 3, 4 });
            dedup.set_block_entry(0, 0, blk);
            dedup.set_block_entry(1, 1, FlatMatrix(2, 2, p7, { 1, 2, 3, 4 }));  // equal content
            dedup.set_block_entry(0, 1, FlatMatrix(2, 2, p7, { 4, 3, 2, 1 }));  // different
            check("set_block_entry: equal content deduplicated to one block",
                  &dedup.get_block_entry(0, 0) == &dedup.get_block_entry(1, 1));
            check("set_block_entry: different content is a different block",
                  &dedup.get_block_entry(0, 0) != &dedup.get_block_entry(0, 1));
            check("get_block_entry: content round trip",
                  same_entries(dedup.get_block_entry(0, 1), FlatMatrix(2, 2, p7, { 4, 3, 2, 1 })));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in Constructors: " << e.what() << std::endl;
        check("Constructors section", false, "unhandled exception");
    }

    // =====================================================================
    // 2. operator* -- the ONE arithmetic operator the legacy arm of
    //    compute_cohomology still calls (D1_block * w_block).
    // =====================================================================
    try {
        section("Block product", p7);

        // A * B against the FlatMatrix product of the twins, entrywise.
        const FlatMatrix a00(2, 2, p7, { 1, 2, 3, 4 }), a01(2, 1, p7, { 5, 6 });
        const FlatMatrix a10(1, 2, p7, { 0, 1 }), a11(1, 1, p7, { 2 });
        BlockFlatMatrix A({ 2, 1 }, { 2, 1 }, { { a00, a01 }, { a10, a11 } }, p7);
        const FlatMatrix A_twin(3, 3, p7, { 1, 2, 5, 3, 4, 6, 0, 1, 2 });

        const FlatMatrix b00(2, 3, p7, { 1, 0, 2, 3, 1, 0 }), b10(1, 3, p7, { 4, 5, 6 });
        BlockFlatMatrix Bm({ 2, 1 }, { 3 }, { { b00 }, { b10 } }, p7);
        const FlatMatrix B_twin(3, 3, p7, { 1, 0, 2, 3, 1, 0, 4, 5, 6 });

        check("A_twin matches A", same_entries(A.to_flat_matrix(), A_twin));
        check("B_twin matches Bm", same_entries(Bm.to_flat_matrix(), B_twin));
        check("operator*: (A*B).to_flat_matrix entrywise == A_twin * B_twin",
              same_entries((A * Bm).to_flat_matrix(), A_twin * B_twin));

        // Product with an identity-shaped operand and with a zero operand.
        BlockFlatMatrix I3({ 2, 1 }, { 2, 1 }, p7);
        I3.set_block_entry(0, 0, FlatMatrix(2, p7));
        I3.set_block_entry(1, 1, FlatMatrix(1, p7));
        check("operator*: A * I == A entrywise", same_entries((A * I3).to_flat_matrix(), A_twin));
        BlockFlatMatrix Z3({ 2, 1 }, { 2, 1 }, p7);
        check("operator*: A * 0 == 0 entrywise",
              same_entries((A * Z3).to_flat_matrix(), FlatMatrix::CreateZero(3, 3, p7)));

        // Zero-width blocks in the contracted dimension.
        BlockFlatMatrix M1({ 1, 0 }, { 0, 1 },
                           { { FlatMatrix::CreateZero(1, 0, p7), FlatMatrix(1, 1, p7, { 5 }) },
                             { FlatMatrix::CreateZero(0, 0, p7), FlatMatrix::CreateZero(0, 1, p7) } }, p7);
        BlockFlatMatrix M2({ 0, 1 }, { 1, 0 },
                           { { FlatMatrix::CreateZero(0, 1, p7), FlatMatrix::CreateZero(0, 0, p7) },
                             { FlatMatrix(1, 1, p7, { 2 }), FlatMatrix::CreateZero(1, 0, p7) } }, p7);
        check("operator*: 0-dim blocks (M1*M2) entrywise",
              same_entries((M1 * M2).to_flat_matrix(),
                           FlatMatrix(1, 1, p7, { ModArithmetic::mod_mul(5, 2, p7) })));

        // Bm has ONE block column (width 3); A has TWO block rows (2 and 1).
        check_throws("operator*: inner block-grid mismatch throws",
                     [&]() { (void)(Bm * A); }, "std::invalid_argument");
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in Block product: " << e.what() << std::endl;
        check("Block product section", false, "unhandled exception");
    }

    // =====================================================================
    // 3. Elimination-derived results on small hand-built inputs, entrywise.
    // =====================================================================
    try {
        section("RREF / rank / kernel / image / inverse (small)", p7);

        // Zero 3x3, blocked {2,1}x{2,1}.
        {
            BlockFlatMatrix Z({ 2, 1 }, { 2, 1 }, p7);
            one_case("[zero 3x3 p=7]", Z, FlatMatrix::CreateZero(3, 3, p7), p7);
        }
        // Identity 3x3.
        {
            BlockFlatMatrix I({ 2, 1 }, { 2, 1 }, p7);
            I.set_block_entry(0, 0, FlatMatrix(2, p7));
            I.set_block_entry(1, 1, FlatMatrix(1, p7));
            const FlatMatrix I_twin(3, 3, p7, { 1, 0, 0, 0, 1, 0, 0, 0, 1 });
            one_case("[identity 3x3 p=7]", I, I_twin, p7);
        }
        // Block-diagonal, full rank, non-trivial diagonal.
        {
            BlockFlatMatrix FR({ 2, 1 }, { 2, 1 }, p7);
            FR.set_block_entry(0, 0, FlatMatrix(2, 2, p7, { 2, 0, 0, 1 }));
            FR.set_block_entry(1, 1, FlatMatrix(1, 1, p7, { 3 }));
            const FlatMatrix FR_twin(3, 3, p7, { 2, 0, 0, 0, 1, 0, 0, 0, 3 });
            one_case("[full rank 3x3 p=7]", FR, FR_twin, p7);
        }
        // Rank deficient square (two identical rows).
        {
            const FlatMatrix one(1, 1, p7, { 1 });
            const FlatMatrix minus_one(1, 1, p7, { ModArithmetic::mod_minus(1, p7) });
            BlockFlatMatrix RD({ 1, 1 }, { 1, 1 }, { { one, minus_one }, { one, minus_one } }, p7);
            const FlatMatrix RD_twin(2, 2, p7, { 1, ModArithmetic::mod_minus(1, p7),
                                                 1, ModArithmetic::mod_minus(1, p7) });
            one_case("[rank-deficient 2x2 p=7]", RD, RD_twin, p7);
        }
        // Wide and tall rectangles with non-uniform block dims.
        {
            SimpleLCG prng(4242);
            const std::vector<uint32_t> rd = { 3, 1, 2 }, cd = { 2, 4, 1 };
            FlatMatrix F(6, 7, p7);
            for (uint32_t r = 0; r < 6; ++r) for (uint32_t c = 0; c < 7; ++c) F.set_entry(r, c, prng.next_mod(p7));
            BlockFlatMatrix B(rd, cd, p7);
            fill_from_dense(B, F, p7);
            one_case("[6x7 non-uniform p=7]", B, F, p7);

            const std::vector<uint32_t> rd2 = { 2, 4, 1 }, cd2 = { 3, 1, 2 };
            FlatMatrix G(7, 6, p7);
            for (uint32_t r = 0; r < 7; ++r) for (uint32_t c = 0; c < 6; ++c) G.set_entry(r, c, prng.next_mod(p7));
            BlockFlatMatrix C(rd2, cd2, p7);
            fill_from_dense(C, G, p7);
            one_case("[7x6 non-uniform p=7]", C, G, p7);
        }
        // Zero-width block columns/rows inside a non-degenerate matrix.
        {
            SimpleLCG prng(4243);
            const std::vector<uint32_t> rd = { 3, 0, 2 }, cd = { 2, 0, 3 };
            FlatMatrix F(5, 5, p7);
            for (uint32_t r = 0; r < 5; ++r) for (uint32_t c = 0; c < 5; ++c) F.set_entry(r, c, prng.next_mod(p7));
            BlockFlatMatrix B(rd, cd, p7);
            fill_from_dense(B, F, p7);
            one_case("[5x5 zero-width blocks p=7]", B, F, p7);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in small elimination: " << e.what() << std::endl;
        check("Small elimination section", false, "unhandled exception");
    }

    // =====================================================================
    // 4. Several primes on a mid-sized random input.  (Audit gap S-6: this
    //    suite used to run at p = 7 only.)
    // =====================================================================
    try {
        for (uint32_t p : { 2u, 3u, 7u, 134217757u }) {
            section("Random 40x33, several primes", p);
            random_low_rank_case("[rand 40x33 rank 20 p=" + std::to_string(p) + "]",
                                 5150 + p, 40, 33, 20, 3, 9, p);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in multi-prime section: " << e.what() << std::endl;
        check("Multi-prime section", false, "unhandled exception");
    }

    // =====================================================================
    // 5. The historical large randomized shapes -- same seeds, same matrices,
    //    now checked entrywise instead of by span.
    // =====================================================================
    try {
        section("Large randomized shapes", p7);

        // (a) ~50-100 square-ish, near-full rank (historical seed 101).
        {
            SimpleLCG prng(101);
            const uint32_t dim = prng.next_range(50, 100);
            const uint32_t min_b = prng.next_range(5, 15);
            const uint32_t max_b = prng.next_range(min_b, 20);
            const uint32_t reduction = prng.next_mod(dim / 4 + 1);
            const uint32_t rank = (dim > reduction) ? (dim - reduction) : 1;
            random_low_rank_case("[large dense ~" + std::to_string(dim) + " p=7]",
                                 101, dim, dim, rank, min_b, max_b, p7);
        }
        // (b) 800 x 400 rank 200 (historical seed 1001).
        random_low_rank_case("[M800_N400_K200 p=7]", 1001, 800, 400, 200, 20, 80, p7);
        // (c) 400 x 800 rank 100 (historical seed 1002).
        random_low_rank_case("[M400_N800_K100 p=7]", 1002, 400, 800, 100, 15, 60, p7);

        // (d) Block diagonal 300x300 of 3x3 diagonal blocks, ~half of them zero
        //     (historical seed 2001).  This is the structural-shortcut shape:
        //     most block rows are structurally empty.
        {
            SimpleLCG prng(2001);
            const uint32_t bs = 3, nb = 100;
            const std::vector<uint32_t> dims(nb, bs);
            BlockFlatMatrix B(dims, dims, p7);
            FlatMatrix F(nb * bs, nb * bs, p7);
            for (uint32_t i = 0; i < nb; ++i) {
                if (prng.next_mod(2) != 0) continue;
                std::vector<uint32_t> buf(static_cast<size_t>(bs) * bs, 0);
                for (uint32_t j = 0; j < bs; ++j) {
                    const uint32_t v = prng.next_mod(p7 - 1) + 1;
                    buf[static_cast<size_t>(j) * bs + j] = v;
                    F.set_entry(i * bs + j, i * bs + j, v);
                }
                B.set_block_entry(i, i, FlatMatrix(bs, bs, p7, buf));
            }
            one_case("[block diagonal 300x300 p=7]", B, F, p7);
        }

        // (e) 1000x1000 of 10x10 blocks, 70 % structurally zero (historical
        //     seed 3001): the sparse-grid pivot search.
        {
            SimpleLCG prng(3001);
            const uint32_t bs = 10, nb = 100;
            const std::vector<uint32_t> dims(nb, bs);
            BlockFlatMatrix B(dims, dims, p7);
            FlatMatrix F(nb * bs, nb * bs, p7);
            for (uint32_t br = 0; br < nb; ++br) {
                for (uint32_t bc = 0; bc < nb; ++bc) {
                    const uint32_t type = prng.next_mod(100);
                    if (type < 70) continue;                       // 70 % structurally zero
                    std::vector<uint32_t> buf(static_cast<size_t>(bs) * bs, 0);
                    if (type < 90) {                               // 20 % diagonal
                        for (uint32_t j = 0; j < bs; ++j)
                            buf[static_cast<size_t>(j) * bs + j] = prng.next_mod(p7 - 1) + 1;
                    } else {                                       // 10 % dense
                        for (uint32_t i = 0; i < bs; ++i)
                            for (uint32_t j = 0; j < bs; ++j)
                                buf[static_cast<size_t>(i) * bs + j] = prng.next_mod(p7);
                    }
                    B.set_block_entry(br, bc, FlatMatrix(bs, bs, p7, buf));
                    for (uint32_t i = 0; i < bs; ++i)
                        for (uint32_t j = 0; j < bs; ++j)
                            F.set_entry(br * bs + i, bc * bs + j, buf[static_cast<size_t>(i) * bs + j]);
                }
            }
            one_case("[sparse block 1000x1000 p=7]", B, F, p7);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in large shapes: " << e.what() << std::endl;
        check("Large shapes section", false, "unhandled exception");
    }

    if (verbosity >= 0) {
        std::cout << "\nBlockFlatMatrixValidation Summary:" << std::endl;
        std::cout << "  Tests run:    " << g_total << std::endl;
        std::cout << "  Tests passed: " << g_passed << std::endl;
        std::cout << "  Tests failed: " << (g_total - g_passed) << std::endl;
        if (g_total == 0) std::cout << "  WARNING: No tests were run!" << std::endl;
    }
    return (g_total > 0 && g_total == g_passed) ? 0 : 1;
}
