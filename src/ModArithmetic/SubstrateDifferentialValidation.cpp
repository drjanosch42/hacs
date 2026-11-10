// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Phase 0 -- substrate differential harness. See the header for scope.
//
// ---------------------------------------------------------------------------
// STATUS ON THE UNMODIFIED Phase-0-entry TREE -- branch `matrix-engine`,
// parent commit e629344, built `cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .`
// (GCC 11.4, -O2 -g -DNDEBUG). Measured 2026-09-05:
//
//     774 cases run, 711 passed, 63 FAILED, wall clock 0.007 s.
//
// EVERY one of those 63 failures is EXPECTED on this tree, and each is
// evidence for a Phase-0 correctness fix landing in parallel. Re-run and
// update this block once they land; the expected end state is 0 failures.
//
//   (1) 51 failures at ell = 4294967291 (> 2^31) -- the SHARED arithmetic
//       defects, not a substrate disagreement:
//         * defect #33, ModArithmetic.cpp mod_add: `uint32_t result = a + b;`
//           wraps mod 2^32 (case `p=4294967291 arith/mod_add(p-1,p-1)`);
//         * defect #2, ModArithmetic.cpp mod_inv: int32_t accumulators, and
//           `int32_t r = p` is already negative for p > 2^31 (case
//           `p=4294967291 arith/mod_inv(x)*x==1`).
//       The `arith/...` pre-check cases run BEFORE any matrix work at each
//       prime precisely so the blame lands on the primitives. The remaining
//       49 are the downstream rank / rref / kernel / image / inverse
//       disagreements they cause (including one guarded section that THROWS,
//       "BlockFlatMatrix (when flattened) is not invertible").
//
//   (2) 12 failures -- 2 per prime at each of ell in {3, 7, 23, 134217757,
//       2147483647, 2^61-1}, namely
//           `6x6-scalar-alphaI image/basis`
//           `6x6-scalar-alphaI CreateScalar/image-basis`
//       FlatMatrixT::image() short-circuits its scalar path to the standard
//       basis e_j (FlatMatrix.cpp), whereas MatrixT::image() returns the
//       actual pivot columns alpha*e_j. The SPANS agree -- the companion
//       `image/span` cases PASS -- so this is a basis-convention divergence,
//       and it is the second fix landing in parallel (align FlatMatrix's
//       scalar path to alpha*e_j). Note it does NOT fire at ell = 2, where
//       the only nonzero scalar is alpha = 1 and the two conventions
//       coincide; that is why ell = 2 is in the ladder.
//
//   (3) ell = 2147483647 (= 2^31 - 1) passes everything except (2): a + b <=
//       2*(p-1) < 2^32 never wraps, and p fits in int32_t. It is in the
//       ladder as the boundary point that must KEEP working after the fix
//       (cf. the plan's boundary test at p = 2^31 + 11).
//
//   (4) Nothing else fails. All cases at ell in {2, 3, 7, 23, 134217757,
//       2147483647} and all 64-bit cases at 2^61-1 (whose mod_add64 /
//       mod_inv64 already carry the correct wide accumulators) agree across
//       all three substrates, including the non-uniform block partitions
//       {3,7,2} x {5,1,6}, the zero-width-block partition {3,0,2} x {5,0,6},
//       the cache-coherence sequences and the algebraic identities.
// ---------------------------------------------------------------------------
// ===========================================================================

#include "SubstrateDifferentialValidation.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ModArithmetic.h"
#include "Matrix.h"
#include "FlatMatrix.h"
#include "LegacyConfig.h"   // (S24 CL-L2) oc::legacy::RunLegacyArm (SKIP-ARM)

namespace {

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64). No <random>: libstdc++'s distributions are
// not guaranteed reproducible across versions, and this harness's whole value
// is that the SAME matrices are fed to all three substrates every run and on
// every machine.
// ---------------------------------------------------------------------------
class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed) {}
    uint64_t next() {
        s_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = s_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Uniform-enough residue in [0, p). Bias is irrelevant here (we only need
    // reproducible, "generic" matrices), but the modulo is taken on a full
    // 64-bit draw so small primes still see every residue.
    template <class Elt>
    Elt residue(Elt p) { return static_cast<Elt>(next() % static_cast<uint64_t>(p)); }
private:
    uint64_t s_;
};

uint32_t g_failures = 0;
uint32_t g_cases = 0;

// Runs one section, converting an escaping exception into a single FAIL
// rather than aborting the whole harness. This matters because a broken
// arithmetic regime (see the STATUS block) makes downstream code THROW
// ("... is not invertible"), and one such throw must not hide every later
// prime's results.
template <class F>
void Guarded(const std::string& section, F&& f);

void Check(bool ok, const std::string& case_name) {
    ++g_cases;
    if (ok) {
        std::cout << "  [PASS] " << case_name << std::endl;
    } else {
        std::cout << "  [FAIL] " << case_name << std::endl;
        ++g_failures;
    }
}

template <class F>
void Guarded(const std::string& section, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        Check(false, section + " [EXCEPTION: " + e.what() + "]");
    } catch (...) {
        Check(false, section + " [EXCEPTION: non-std]");
    }
}

// ---------------------------------------------------------------------------
// Dense helpers. Everything is compared through the same neutral
// representation (row-major vector<vector<Elt>>), so no substrate's own
// equality operator can mask a difference.
// ---------------------------------------------------------------------------
template <class Elt>
using Dense = std::vector<std::vector<Elt>>;

template <class Elt>
Dense<Elt> RandomDense(Rng& rng, uint32_t rows, uint32_t cols, Elt p) {
    Dense<Elt> d(rows, std::vector<Elt>(cols, Elt(0)));
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) d[i][j] = rng.template residue<Elt>(p);
    return d;
}

template <class Elt>
Dense<Elt> FromMatrix(const MatrixT<Elt>& m) { return m.data; }

template <class Elt>
Dense<Elt> FromFlat(const FlatMatrixT<Elt>& f) {
    Dense<Elt> d(f.get_rows(), std::vector<Elt>(f.get_cols(), Elt(0)));
    for (uint32_t i = 0; i < f.get_rows(); ++i)
        for (uint32_t j = 0; j < f.get_cols(); ++j) d[i][j] = f.get_entry(i, j);
    return d;
}

// Entrywise equality, tolerating a difference in ROW COUNT only when the
// surplus rows are identically zero (the three substrates disagree on whether
// an RREF keeps its trailing zero rows; that is a shape convention, not an
// arithmetic difference, so it is reported separately as `*/rref-shape`).
template <class Elt>
bool EqualUpToZeroRows(const Dense<Elt>& a, const Dense<Elt>& b, bool& same_shape) {
    const size_t ra = a.size(), rb = b.size();
    const size_t ca = ra ? a[0].size() : 0, cb = rb ? b[0].size() : 0;
    same_shape = (ra == rb && ca == cb);
    if (ca != cb && ra != 0 && rb != 0) return false;
    const size_t common = ra < rb ? ra : rb;
    for (size_t i = 0; i < common; ++i)
        for (size_t j = 0; j < ca; ++j)
            if (a[i][j] != b[i][j]) return false;
    for (size_t i = common; i < ra; ++i)
        for (size_t j = 0; j < ca; ++j) if (a[i][j] != Elt(0)) return false;
    for (size_t i = common; i < rb; ++i)
        for (size_t j = 0; j < cb; ++j) if (b[i][j] != Elt(0)) return false;
    return true;
}

template <class Elt>
bool ExactlyEqual(const Dense<Elt>& a, const Dense<Elt>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j) if (a[i][j] != b[i][j]) return false;
    }
    return true;
}

// The canonical form of a SPAN: stack the given vectors as rows and take the
// RREF (MatrixT's Gauss-Jordan). Two bases span the same subspace iff their
// canonical forms, stripped of zero rows, coincide. This is what lets the
// harness distinguish "different basis, same subspace" (acceptable, worth
// reporting) from "different subspace" (a real defect).
template <class Elt>
Dense<Elt> SpanCanonical(const std::vector<std::vector<Elt>>& row_vectors, uint32_t dim, Elt p) {
    if (row_vectors.empty() || dim == 0) return Dense<Elt>();
    MatrixT<Elt> m(static_cast<uint32_t>(row_vectors.size()), dim, p, row_vectors);
    Dense<Elt> r = m.get_row_echelon_form_as_matrix().data;
    // Strip trailing zero rows.
    while (!r.empty()) {
        bool zero = true;
        for (Elt v : r.back()) if (v != Elt(0)) { zero = false; break; }
        if (!zero) break;
        r.pop_back();
    }
    return r;
}

// Column-vector basis (each element is an L x 1 matrix) -> rows of length L.
template <class Elt>
std::vector<std::vector<Elt>> ColsToRows_M(const std::vector<MatrixT<Elt>>& basis) {
    std::vector<std::vector<Elt>> out;
    for (const auto& v : basis) {
        std::vector<Elt> row(v.rows);
        for (uint32_t i = 0; i < v.rows; ++i) row[i] = v.data[i][0];
        out.push_back(std::move(row));
    }
    return out;
}

template <class Elt>
std::vector<std::vector<Elt>> ColsToRows_F(const std::vector<FlatMatrixT<Elt>>& basis) {
    std::vector<std::vector<Elt>> out;
    for (const auto& v : basis) {
        std::vector<Elt> row(v.get_rows());
        for (uint32_t i = 0; i < v.get_rows(); ++i) row[i] = v.get_entry(i, 0);
        out.push_back(std::move(row));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Substrate constructors from one dense source of truth.
// ---------------------------------------------------------------------------
template <class Elt>
MatrixT<Elt> MakeMatrix(const Dense<Elt>& d, uint32_t rows, uint32_t cols, Elt p) {
    if (rows == 0 || cols == 0) return MatrixT<Elt>(rows, cols, p);
    return MatrixT<Elt>(rows, cols, p, d);
}

template <class Elt>
FlatMatrixT<Elt> MakeFlat(const Dense<Elt>& d, uint32_t rows, uint32_t cols, Elt p) {
    std::vector<Elt> flat;
    flat.reserve(static_cast<size_t>(rows) * cols);
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) flat.push_back(d[i][j]);
    if (flat.empty()) return FlatMatrixT<Elt>(rows, cols, p);
    return FlatMatrixT<Elt>(rows, cols, p, flat);
}

std::string Tag(const std::string& prime_label, const std::string& what) {
    return "p=" + prime_label + " " + what;
}

// (S24 CL-L2) The third substrate's results on one shape: filled by legacy_block_shape (legacy/, ON builds); in an
// OFF build `have` stays false and every three-way check compares Matrix and FlatMatrix only.
template <class Elt>
struct BlockShape {
    bool have = false;
    Dense<Elt> construct, rref;
    uint32_t rank = 0;
    std::vector<std::vector<Elt>> ker, img;
};

} // namespace   (S24 CL-L2: the legacy .inc includes its headers at file scope)
#if OC_ENABLE_LEGACY
#include "legacy/SubstrateDifferentialLegacyArms.inc"   // (S24 CL-L2) the BlockFlatMatrix substrate
#endif
namespace {

// ---------------------------------------------------------------------------
// (0) Arithmetic pre-check. Runs BEFORE the matrix comparisons so that a
// failure at p > 2^31 is attributed to the shared primitives (defects #2/#33)
// rather than being misread as a substrate disagreement.
// ---------------------------------------------------------------------------
void ArithmeticPrecheck32(uint32_t p, const std::string& label) {
    // mod_add on the worst case a = b = p-1: the true answer is p-2.
    const uint32_t a = p - 1;
    const uint32_t got_add = ModArithmetic::mod_add(a, a, p);
    const uint32_t want_add = static_cast<uint32_t>(
        ((static_cast<uint64_t>(a) + a) % p));
    Check(got_add == want_add, Tag(label, "arith/mod_add(p-1,p-1)"));

    // mod_inv on a handful of residues: a * a^{-1} == 1.
    bool inv_ok = true;
    Rng rng(0xA11CE ^ p);
    for (int k = 0; k < 32 && inv_ok; ++k) {
        uint32_t x = rng.residue<uint32_t>(p);
        if (x == 0) x = 1;
        try {
            const uint32_t xi = ModArithmetic::mod_inv(x, p);
            if (ModArithmetic::mod_mul(x, xi, p) != 1u) inv_ok = false;
        } catch (const std::exception&) { inv_ok = false; }
    }
    Check(inv_ok, Tag(label, "arith/mod_inv(x)*x==1"));
}

// ---------------------------------------------------------------------------
// (1) The core differential battery on one shape.
// ---------------------------------------------------------------------------
template <class Elt>
void CompareOneShape(const Dense<Elt>& d, uint32_t rows, uint32_t cols, Elt p,
                     const std::vector<uint32_t>& row_dims,
                     const std::vector<uint32_t>& col_dims,
                     const std::string& label, const std::string& shape_name) {
    const std::string base = shape_name + " ";

    MatrixT<Elt> M = MakeMatrix(d, rows, cols, p);
    FlatMatrixT<Elt> F = MakeFlat(d, rows, cols, p);
#if OC_ENABLE_LEGACY
    const BlockShape<Elt> B = legacy_block_shape<Elt>(d, row_dims, col_dims, p);   // legacy/SubstrateDifferentialLegacyArms.inc
#else
    const BlockShape<Elt> B;   // (S24 CL-L2) the third substrate is compiled out: two-way checks
    (void)row_dims; (void)col_dims;
#endif

    // --- entrywise round-trip: all three hold the same matrix to begin with.
    Check(ExactlyEqual(FromMatrix(M), FromFlat(F)) && (!B.have || ExactlyEqual(FromMatrix(M), B.construct)),
          Tag(label, base + "construction/entrywise"));

    // --- rank
    const uint32_t rk_m = M.rank();
    const uint32_t rk_f = F.rank();
    const uint32_t rk_b = B.rank;
    Check(rk_m == rk_f && (!B.have || rk_m == rk_b), Tag(label, base + "rank"));

    // --- RREF, entrywise
    const Dense<Elt> rref_m = FromMatrix(M.get_row_echelon_form_as_matrix());
    const Dense<Elt> rref_f = FromFlat(F.get_row_echelon_form_as_flat_matrix());
    bool shape_mf = false, shape_mb = false;
    const bool eq_mf = EqualUpToZeroRows(rref_m, rref_f, shape_mf);
    Check(eq_mf, Tag(label, base + "rref/entrywise Matrix-vs-FlatMatrix"));
    if (B.have) {
        const bool eq_mb = EqualUpToZeroRows(rref_m, B.rref, shape_mb);
        Check(eq_mb, Tag(label, base + "rref/entrywise Matrix-vs-BlockFlatMatrix"));
    }

    // --- kernel: dimension, spans, and (separately) the bases themselves
    auto ker_m = ColsToRows_M(M.kernel());
    auto ker_f = ColsToRows_F(F.kernel());
    const auto& ker_b = B.ker;
    Check(ker_m.size() == ker_f.size() && (!B.have || ker_m.size() == ker_b.size()),
          Tag(label, base + "kernel/dimension"));
    const Dense<Elt> ksp_m = SpanCanonical(ker_m, cols, p);
    const Dense<Elt> ksp_f = SpanCanonical(ker_f, cols, p);
    Check(ExactlyEqual(ksp_m, ksp_f) && (!B.have || ExactlyEqual(ksp_m, SpanCanonical(ker_b, cols, p))),
          Tag(label, base + "kernel/span"));
    Check(ker_m == ker_f && (!B.have || ker_m == ker_b), Tag(label, base + "kernel/basis"));

    // --- image: dimension, spans, and (separately) the bases themselves
    auto img_m = ColsToRows_M(M.image());
    auto img_f = ColsToRows_F(F.image());
    const auto& img_b = B.img;
    Check(img_m.size() == img_f.size() && (!B.have || img_m.size() == img_b.size()),
          Tag(label, base + "image/dimension"));
    const Dense<Elt> isp_m = SpanCanonical(img_m, rows, p);
    const Dense<Elt> isp_f = SpanCanonical(img_f, rows, p);
    Check(ExactlyEqual(isp_m, isp_f) && (!B.have || ExactlyEqual(isp_m, SpanCanonical(img_b, rows, p))),
          Tag(label, base + "image/span"));
    Check(img_m == img_f && (!B.have || img_m == img_b), Tag(label, base + "image/basis"));

    // --- rank-nullity: dim ker + rank = cols
    Check(static_cast<uint32_t>(ker_m.size()) + rk_m == cols,
          Tag(label, base + "identity/dim-ker+rank==cols"));
}

// ---------------------------------------------------------------------------
// (2) Cache-coherence sequences (defect #9: MatrixT's RREF cache is never
// invalidated while `data` is public; FlatMatrix / BlockFlatMatrix invalidate
// on set_entry / set_block_entry). Each sequence is compared against a FRESH
// object built from the mutated data -- the only oracle that cannot itself be
// poisoned by a stale cache.
// ---------------------------------------------------------------------------
template <class Elt>
void CacheCoherence(Rng& rng, Elt p, const std::string& label) {
    const uint32_t n = 6;
    Dense<Elt> d = RandomDense(rng, n, n, p);
    // Force a rank drop so the mutation below is observable.
    d[3] = d[1];

    // (a) FlatMatrix: rank; set_entry; rank
    {
        FlatMatrixT<Elt> F = MakeFlat(d, n, n, p);
        (void)F.rank();                              // populates the RREF cache
        F.set_entry(3, 0, ModArithmetic::mod_add_w(F.get_entry(3, 0), Elt(1), p));
        const uint32_t after = F.rank();

        Dense<Elt> d2 = d;
        d2[3][0] = ModArithmetic::mod_add_w(d2[3][0], Elt(1), p);
        FlatMatrixT<Elt> Fresh = MakeFlat(d2, n, n, p);
        Check(after == Fresh.rank(), Tag(label, "cache/FlatMatrix rank;set_entry;rank"));
    }

    // (b) FlatMatrix: rank; is_zero; rank  (is_zero() may re-flag a general
    //     matrix as the scalar-zero special case, which must not corrupt the
    //     cached rank).
    {
        FlatMatrixT<Elt> F = MakeFlat(d, n, n, p);
        const uint32_t r1 = F.rank();
        const bool z = F.is_zero();
        const uint32_t r2 = F.rank();
        Check(r1 == r2 && !z, Tag(label, "cache/FlatMatrix rank;is_zero;rank"));
    }

    // (c) Same two sequences on an actually-zero matrix, where is_zero() DOES
    //     flip the internal representation.
    {
        Dense<Elt> z(n, std::vector<Elt>(n, Elt(0)));
        FlatMatrixT<Elt> F = MakeFlat(z, n, n, p);
        const uint32_t r1 = F.rank();
        const bool iz = F.is_zero();
        const uint32_t r2 = F.rank();
        Check(r1 == 0 && r2 == 0 && iz, Tag(label, "cache/FlatMatrix zero rank;is_zero;rank"));
    }

#if OC_ENABLE_LEGACY
    legacy_block_cache_coherence<Elt>(d, p, label);   // (d), (e): legacy/SubstrateDifferentialLegacyArms.inc
#endif

}

// ---------------------------------------------------------------------------
// (3) Algebraic identities, checked in all three substrates.
// ---------------------------------------------------------------------------
template <class Elt>
void AlgebraicIdentities(Rng& rng, Elt p, const std::string& label) {
    const uint32_t m = 5, k = 4, n = 6;
    const Dense<Elt> da = RandomDense(rng, m, k, p);
    const Dense<Elt> db = RandomDense(rng, k, n, p);

    // (AB)^T == B^T A^T
    {
        MatrixT<Elt> A = MakeMatrix(da, m, k, p), Bm = MakeMatrix(db, k, n, p);
        MatrixT<Elt> lhs = (A * Bm).transpose();
        MatrixT<Elt> At = A.transpose(), Bt = Bm.transpose();
        MatrixT<Elt> rhs = Bt * At;
        Check(ExactlyEqual(FromMatrix(lhs), FromMatrix(rhs)),
              Tag(label, "identity/(AB)^T==B^T A^T [Matrix]"));

        FlatMatrixT<Elt> Af = MakeFlat(da, m, k, p), Bf = MakeFlat(db, k, n, p);
        FlatMatrixT<Elt> lhsf = (Af * Bf).transpose();
        FlatMatrixT<Elt> rhsf = Bf.transpose() * Af.transpose();
        Check(ExactlyEqual(FromFlat(lhsf), FromFlat(rhsf)),
              Tag(label, "identity/(AB)^T==B^T A^T [FlatMatrix]"));

        // Cross-substrate: the products themselves must agree entrywise.
        Check(ExactlyEqual(FromMatrix(A * Bm), FromFlat(Af * Bf)),
              Tag(label, "product/Matrix-vs-FlatMatrix"));

#if OC_ENABLE_LEGACY
        legacy_block_product<Elt>(da, db, FromMatrix(A * Bm), p, label);   // legacy/SubstrateDifferentialLegacyArms.inc
#endif
    }

    // rank A == rank A^T
    {
        MatrixT<Elt> A = MakeMatrix(da, m, k, p);
        MatrixT<Elt> At = A.transpose();
        FlatMatrixT<Elt> Af = MakeFlat(da, m, k, p);
        FlatMatrixT<Elt> Aft = Af.transpose();
        Check(A.rank() == At.rank() && Af.rank() == Aft.rank() && A.rank() == Af.rank(),
              Tag(label, "identity/rank A==rank A^T"));
    }

    // A * A^{-1} == I for a random invertible A (retry until invertible; over
    // F_2 a 5x5 random matrix is singular ~30% of the time, so this loops).
    {
        bool done = false;
        for (int attempt = 0; attempt < 64 && !done; ++attempt) {
            const uint32_t d = 5;
            const Dense<Elt> dm = RandomDense(rng, d, d, p);
            MatrixT<Elt> A = MakeMatrix(dm, d, d, p);
            if (!A.is_invertible()) continue;
            done = true;
            MatrixT<Elt> Ainv = A.inverse();
            MatrixT<Elt> I = MakeMatrix(dm, d, d, p); // re-made below as identity
            MatrixT<Elt> Id(d, p);                    // identity ctor
            Check(ExactlyEqual(FromMatrix(A * Ainv), FromMatrix(Id)),
                  Tag(label, "identity/A*A^{-1}==I [Matrix]"));

            FlatMatrixT<Elt> Af = MakeFlat(dm, d, d, p);
            FlatMatrixT<Elt> Afinv = Af.inverse();
            FlatMatrixT<Elt> Idf(d, p);
            Check(ExactlyEqual(FromFlat(Af * Afinv), FromFlat(Idf)),
                  Tag(label, "identity/A*A^{-1}==I [FlatMatrix]"));
            // The inverses themselves must agree entrywise across substrates.
            Check(ExactlyEqual(FromMatrix(Ainv), FromFlat(Afinv)),
                  Tag(label, "inverse/Matrix-vs-FlatMatrix"));

#if OC_ENABLE_LEGACY
            legacy_block_inverse<Elt>(dm, FromMatrix(Ainv), p, label);   // legacy/SubstrateDifferentialLegacyArms.inc
#endif
            (void)I;
        }
        if (!done) Check(false, Tag(label, "identity/A*A^{-1}==I (no invertible sample found)"));
    }
}

// ---------------------------------------------------------------------------
// (4) The full battery at one prime, for one element width.
// ---------------------------------------------------------------------------
template <class Elt>
void RunBattery(Elt p, const std::string& label, uint64_t seed) {
    std::cout << "--- substrate differential battery at ell = " << static_cast<uint64_t>(p)
              << " (" << (sizeof(Elt) * 8) << "-bit substrate) ---" << std::endl;
    Rng rng(seed);

    // (a) square, generic
    {
        const uint32_t n = 12;
        const Dense<Elt> d = RandomDense(rng, n, n, p);
        Guarded(Tag(label, "12x12-generic"), [&] {
            CompareOneShape(d, n, n, p, {3, 7, 2}, {5, 1, 6}, label, "12x12-generic"); });
    }
    // (b) wide and tall (non-square RREF / kernel / image paths)
    {
        const Dense<Elt> d = RandomDense(rng, 7, 12, p);
        Guarded(Tag(label, "7x12-wide"), [&] {
            CompareOneShape(d, 7, 12, p, {3, 4}, {5, 1, 6}, label, "7x12-wide"); });
    }
    {
        const Dense<Elt> d = RandomDense(rng, 12, 7, p);
        Guarded(Tag(label, "12x7-tall"), [&] {
            CompareOneShape(d, 12, 7, p, {3, 7, 2}, {4, 3}, label, "12x7-tall"); });
    }
    // (c) rank-deficient: two duplicated rows and a zero row
    {
        const uint32_t n = 12;
        Dense<Elt> d = RandomDense(rng, n, n, p);
        d[5] = d[2];
        d[9] = d[2];
        for (auto& v : d[7]) v = Elt(0);
        Guarded(Tag(label, "12x12-rank-deficient"), [&] {
            CompareOneShape(d, n, n, p, {3, 7, 2}, {5, 1, 6}, label, "12x12-rank-deficient"); });
    }
    // (d) ZERO-WIDTH / ZERO-HEIGHT blocks in the partition
    {
        const uint32_t rows = 5, cols = 11;
        const Dense<Elt> d = RandomDense(rng, rows, cols, p);
        Guarded(Tag(label, "5x11-zero-width-blocks"), [&] {
            CompareOneShape(d, rows, cols, p, {3, 0, 2}, {5, 0, 6}, label, "5x11-zero-width-blocks"); });
    }
    // (e) the all-zero matrix (FlatMatrix's scalar-zero special case)
    {
        const uint32_t n = 6;
        const Dense<Elt> d(n, std::vector<Elt>(n, Elt(0)));
        Guarded(Tag(label, "6x6-zero"), [&] {
            CompareOneShape(d, n, n, p, {2, 4}, {3, 3}, label, "6x6-zero"); });
    }
    // (f) a SCALAR matrix alpha*I with alpha != 1 -- FlatMatrix's dedicated
    //     scalar representation. This is where FlatMatrixT::image()'s scalar
    //     short-circuit (returning e_j instead of alpha*e_j) shows up: the
    //     `image/span` case passes while `image/basis` fails, on the
    //     unmodified tree. See the STATUS block at the top of this file.
    {
        const uint32_t n = 6;
        Elt alpha = (p > Elt(2)) ? Elt(2) : Elt(1);
        Dense<Elt> d(n, std::vector<Elt>(n, Elt(0)));
        for (uint32_t i = 0; i < n; ++i) d[i][i] = alpha;
        Guarded(Tag(label, "6x6-scalar-alphaI"), [&] {
            CompareOneShape(d, n, n, p, {2, 4}, {2, 4}, label, "6x6-scalar-alphaI"); });
        // The FlatMatrix built from a dense buffer above is a GENERAL matrix
        // that happens to be scalar; also exercise the genuinely special
        // representation produced by CreateScalar.
        FlatMatrixT<Elt> S = FlatMatrixT<Elt>::CreateScalar(n, alpha, p);
        MatrixT<Elt> Sm = MakeMatrix(d, n, n, p);
        Check(ExactlyEqual(FromFlat(S), FromMatrix(Sm)),
              Tag(label, "6x6-scalar-alphaI CreateScalar/entrywise"));
        Check(S.rank() == Sm.rank(), Tag(label, "6x6-scalar-alphaI CreateScalar/rank"));
        auto img_s = ColsToRows_F(S.image());
        auto img_sm = ColsToRows_M(Sm.image());
        Check(ExactlyEqual(SpanCanonical(img_s, n, p), SpanCanonical(img_sm, n, p)),
              Tag(label, "6x6-scalar-alphaI CreateScalar/image-span"));
        Check(img_s == img_sm, Tag(label, "6x6-scalar-alphaI CreateScalar/image-basis"));
    }

    Guarded(Tag(label, "cache-coherence"), [&] { CacheCoherence(rng, p, label); });
    Guarded(Tag(label, "algebraic-identities"), [&] { AlgebraicIdentities(rng, p, label); });
}

} // namespace

uint32_t SubstrateDifferentialValidation() {
    g_failures = 0;
    g_cases = 0;
    std::cout << "================= SubstrateDifferentialValidation =================\n"
                 "Matrix vs FlatMatrix vs BlockFlatMatrix, fixed-seed inputs.\n"
                 "See the STATUS block at the head of "
                 "ModArithmetic/SubstrateDifferentialValidation.cpp for the cases that\n"
                 "are EXPECTED to fail on the unmodified Phase-0-entry tree.\n"
              << std::endl;
    (void)oc::legacy::RunLegacyArm("SubstrateDifferentialValidation", "BlockFlatMatrix");   // (S24 CL-L2) the third substrate

    try {
        // uint32_t regime. The last two primes straddle 2^31, which is where
        // defects #2 (mod_inv) and #33 (mod_add) bite.
        struct P32 { uint32_t p; const char* label; };
        const P32 primes32[] = {
            {2u, "2"},
            {3u, "3"},
            {7u, "7"},
            {23u, "23"},
            {134217757u, "134217757"},      // ~2^27, the benchmark prime
            {2147483647u, "2147483647"},    // 2^31 - 1 (boundary; must pass)
            {4294967291u, "4294967291"},    // > 2^31 (defects #2 / #33)
        };
        uint64_t seed = 0x5EED0001ull;
        for (const auto& e : primes32) {
            Guarded(Tag(e.label, "arith-precheck"), [&] { ArithmeticPrecheck32(e.p, e.label); });
            Guarded(Tag(e.label, "battery"), [&] { RunBattery<uint32_t>(e.p, e.label, seed); });
            seed += 0x1000ull;
        }

        // 64-bit regime (Stage S13's MatrixT<uint64_t> / FlatMatrixT<uint64_t>
        // / BlockFlatMatrixT<uint64_t>), at the Mersenne prime 2^61 - 1.
        Guarded(Tag("2^61-1", "battery"),
                [&] { RunBattery<uint64_t>(2305843009213693951ull, "2^61-1", seed); });
    } catch (const std::exception& e) {
        std::cout << "SubstrateDifferentialValidation: TOP-LEVEL EXCEPTION: " << e.what()
                  << std::endl;
        ++g_failures;
    }

    std::cout << "\nSubstrateDifferentialValidation: " << (g_cases - g_failures) << "/" << g_cases
              << " cases passed; " << g_failures << " failure(s)." << std::endl;
    return g_failures;
}
