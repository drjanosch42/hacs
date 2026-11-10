// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "DenseElimValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DenseBuf.h"
#include "DenseElim.h"
#include "DenseView.h"
#include "FlatMatrix.h"
#include "ModCtx.h"
#include "RefLinAlg.h"
#include "LegacyConfig.h"   // (S24 CL-L2) oc::legacy::RunLegacyArm (SKIP-ARM)

// See DenseElimValidation.h.

namespace {

using engine::DenseBuf;
using engine::DenseView;
using engine::ModCtx;
using engine::cview;
using engine::view;

struct Splitmix {
    uint64_t s;
    explicit Splitmix(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t m) { return m <= 1 ? 0 : next() % m; }
};

uint32_t g_fail = 0;
uint64_t g_checks = 0;

void fail(const std::string& what) {
    ++g_fail;
    if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
}
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) fail(what);
}
template<class T> std::string str(T v) { return std::to_string(v); }

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------
template<class Elt>
std::vector<Elt> random_entries(Splitmix& rng, Elt p, size_t n) {
    std::vector<Elt> v(n);
    for (auto& x : v) x = static_cast<Elt>(rng.below(p));
    return v;
}

// rows x cols of rank <= r: L (rows x r) * R (r x cols), exact product.
template<class Elt>
std::vector<Elt> ranked_matrix(Splitmix& rng, const ModCtx<Elt>& ctx, uint32_t rows, uint32_t cols, uint32_t r) {
    std::vector<Elt> out(static_cast<size_t>(rows) * cols, Elt(0));
    if (r == 0) return out;
    const std::vector<Elt> L = random_entries<Elt>(rng, ctx.p, static_cast<size_t>(rows) * r);
    const std::vector<Elt> R = random_entries<Elt>(rng, ctx.p, static_cast<size_t>(r) * cols);
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) {
            Elt s = 0;
            for (uint32_t t = 0; t < r; ++t) s = ctx.add(s, ctx.mul(L[static_cast<size_t>(i) * r + t], R[static_cast<size_t>(t) * cols + j]));
            out[static_cast<size_t>(i) * cols + j] = s;
        }
    return out;
}

template<class Elt>
DenseBuf<Elt> to_buf(const std::vector<Elt>& flat, uint32_t rows, uint32_t cols) {
    DenseBuf<Elt> b(rows, cols);
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) b.at(i, j) = flat[static_cast<size_t>(i) * cols + j];
    return b;
}

// ---------------------------------------------------------------------------
// Entrywise comparators
// ---------------------------------------------------------------------------
template<class Elt>
bool same_buf(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) { return a.same_entries(b); }

template<class Elt>
bool same_buf_fm(const DenseBuf<Elt>& a, const FlatMatrixT<Elt>& f) {
    if (a.rows != f.get_rows() || a.cols != f.get_cols()) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(i, j) != f.get_entry(i, j)) return false;
    return true;
}

// rows of `a` vs a list of column vectors
template<class Elt>
bool same_rows_vs_columns(const DenseBuf<Elt>& a, const std::vector<FlatMatrixT<Elt>>& cols) {
    if (a.rows != cols.size()) return false;
    for (uint32_t t = 0; t < a.rows; ++t) {
        if (cols[t].get_cols() != 1 || cols[t].get_rows() != a.cols) return false;
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.at(t, j) != cols[t].get_entry(j, 0)) return false;
    }
    return true;
}

template<class Elt>
bool same_rows_vs_vectors(const DenseBuf<Elt>& a, const std::vector<std::vector<Elt>>& v) {
    if (a.rows != v.size()) return false;
    for (uint32_t t = 0; t < a.rows; ++t) {
        if (v[t].size() != a.cols) return false;
        for (uint32_t j = 0; j < a.cols; ++j) if (a.at(t, j) != v[t][j]) return false;
    }
    return true;
}

template<class Elt>
std::vector<FlatMatrixT<Elt>> columns_of(const DenseBuf<Elt>& a, Elt p) {
    std::vector<FlatMatrixT<Elt>> out;
    for (uint32_t j = 0; j < a.cols; ++j) {
        std::vector<Elt> d(a.rows);
        for (uint32_t i = 0; i < a.rows; ++i) d[i] = a.at(i, j);
        out.emplace_back(a.rows, 1, p, d);
    }
    return out;
}

} // namespace   (S24 CL-L2: the legacy .inc includes its headers at file scope)
#if OC_ENABLE_LEGACY
#include "legacy/DenseElimLegacyArms.inc"   // (S24 CL-L2) the ScopedLegacy / factor_basis_block halves
#endif
namespace {

// ---------------------------------------------------------------------------
// One case
// ---------------------------------------------------------------------------
struct CaseOpts {
    bool legacy = true;       // compare against the OLD FlatMatrix paths
    bool oracle = true;       // compare against RefLinAlg
    bool transform = true;    // rref_with_transform / inverse / solve / complement / eliminate
};

template<class Elt>
void run_case(Splitmix& rng, const ModCtx<Elt>& ctx, uint32_t rows, uint32_t cols, uint32_t rank_target,
              const CaseOpts& o, const std::string& tag) {
    const Elt p = ctx.p;
    const uint32_t full = std::min(rows, cols);
    const std::vector<Elt> data = (rank_target >= full)
        ? random_entries<Elt>(rng, p, static_cast<size_t>(rows) * cols)
        : ranked_matrix<Elt>(rng, ctx, rows, cols, rank_target);
    const DenseBuf<Elt> A = to_buf<Elt>(data, rows, cols);

    // --- engine RREF ---------------------------------------------------------
    DenseBuf<Elt> R = A;
    const engine::elim::RrefResult res = engine::elim::rref<Elt>(ctx, view(R));
    check(res.rank == res.pivots.size() && res.rank <= full, tag + " rank/pivot count");
    for (uint32_t i = 1; i < res.pivots.size(); ++i) check(res.pivots[i - 1] < res.pivots[i], tag + " pivots ascending");
    // structural: leading ones, zero rows at the bottom
    {
        bool ok = true;
        for (uint32_t i = 0; i < res.rank && ok; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                const Elt v = R.at(i, j);
                if (j < res.pivots[i] && v != 0) { ok = false; break; }
                if (j == res.pivots[i] && v != 1) { ok = false; break; }
            }
            for (uint32_t i2 = 0; i2 < res.rank; ++i2) if (i2 != i && R.at(i2, res.pivots[i]) != 0) ok = false;
        }
        for (uint32_t i = res.rank; i < rows && ok; ++i)
            for (uint32_t j = 0; j < cols; ++j) if (R.at(i, j) != 0) { ok = false; break; }
        check(ok, tag + " RREF structure");
    }
    // column-major input must give the same answer
    if (rows * cols <= 70000) {
        DenseBuf<Elt> Ac(rows, cols, engine::Layout::ColMajor);
        for (uint32_t i = 0; i < rows; ++i) for (uint32_t j = 0; j < cols; ++j) Ac.at(i, j) = A.at(i, j);
        const engine::elim::RrefResult rc = engine::elim::rref<Elt>(ctx, view(Ac));
        check(rc.pivots == res.pivots && Ac.same_entries(R), tag + " col-major layout identical");
    }
    const DenseBuf<Elt> K = engine::elim::kernel_basis<Elt>(ctx, cview(R), res.pivots);
    const DenseBuf<Elt> I = engine::elim::image_basis<Elt>(cview(A), res.pivots);
    check(K.rows == cols - res.rank && K.cols == cols, tag + " kernel shape");
    check(I.rows == res.rank && I.cols == rows, tag + " image shape");

    // --- oracle --------------------------------------------------------------
    if (o.oracle) {
        DenseBuf<Elt> Rr = A;
        const std::vector<uint32_t> piv_ref = engine::ref::rref<Elt>(p, view(Rr));
        check(piv_ref == res.pivots, tag + " pivots vs RefLinAlg");
        check(same_buf(R, Rr), tag + " RREF vs RefLinAlg");
        const auto ker_ref = engine::ref::kernel_basis<Elt>(p, cview(A));
        check(same_rows_vs_vectors(K, ker_ref), tag + " kernel vs RefLinAlg");
    }

    // --- legacy FlatMatrix paths (ScopedLegacy) and the retrofitted wrappers --
    if (o.legacy) {
#if OC_ENABLE_LEGACY
        legacy_flat_paths_arm<Elt>(rows, cols, p, data, res, R, K, I, tag);   // legacy/DenseElimLegacyArms.inc
#endif
        // retrofitted wrappers, engine mode
        {
            const FlatMatrixT<Elt> M(rows, cols, p, data);
            check(M.get_pivot_columns() == res.pivots, tag + " retrofit pivots");
            check(same_buf_fm(R, M.get_row_echelon_form_as_flat_matrix()), tag + " retrofit RREF");
            check(same_rows_vs_columns(K, M.kernel()), tag + " retrofit kernel");
            check(same_rows_vs_columns(I, M.image()), tag + " retrofit image");
        }
    }

    if (!o.transform) return;

    // --- transform U = trailing block of RREF([A | I]) -------------------------
    {
        DenseBuf<Elt> Rt(rows, cols), U(rows, rows);
        const engine::elim::RrefResult rt = engine::elim::rref_with_transform<Elt>(ctx, cview(A), view(Rt), view(U));
        check(rt.pivots == res.pivots && same_buf(Rt, R), tag + " rref_with_transform RREF");
        // U * A == RREF(A) exactly
        DenseBuf<Elt> UA(rows, cols);
        engine::ref::gemm<Elt>(p, cview(U), cview(A), view(UA), 0);
        check(same_buf(UA, R), tag + " U*A == RREF(A)");
        // the augmented definition, via the oracle
        DenseBuf<Elt> aug(rows, cols + rows);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) aug.at(i, j) = A.at(i, j);
            aug.at(i, cols + i) = 1;
        }
        if (o.oracle) {
            engine::ref::rref<Elt>(p, view(aug));
            bool ok = true;
            for (uint32_t i = 0; i < rows && ok; ++i)
                for (uint32_t j = 0; j < rows; ++j) if (aug.at(i, cols + j) != U.at(i, j)) { ok = false; break; }
            check(ok, tag + " U == trailing block of RefLinAlg RREF([A|I])");
        }
        if (o.legacy) {
            std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> eng;
#if OC_ENABLE_LEGACY
            legacy_transform_arm<Elt>(rows, cols, p, data, Rt, U, tag);   // legacy/DenseElimLegacyArms.inc
#endif
            {
                const FlatMatrixT<Elt> M(rows, cols, p, data);
                eng = M.get_rref_and_transform_matrix();
            }
            check(same_buf_fm(Rt, eng.first) && same_buf_fm(U, eng.second), tag + " transform vs retrofit get_rref_and_transform_matrix");
        }
    }

    // --- inverse ---------------------------------------------------------------
    if (rows == cols && rows > 0) {
        DenseBuf<Elt> Ainv(rows, rows);
        const bool ok = engine::elim::inverse<Elt>(ctx, cview(A), view(Ainv));
        check(ok == (res.rank == rows), tag + " inverse existence");
        if (ok) {
            DenseBuf<Elt> P(rows, rows);
            engine::ref::gemm<Elt>(p, cview(A), cview(Ainv), view(P), 0);
            bool is_id = true;
            for (uint32_t i = 0; i < rows && is_id; ++i)
                for (uint32_t j = 0; j < rows; ++j) if (P.at(i, j) != (i == j ? 1 : 0)) { is_id = false; break; }
            check(is_id, tag + " A * inv(A) == I");
        }
        if (o.legacy) {
            bool eng_ok = false;
            FlatMatrixT<Elt> eng;
            try {
                const FlatMatrixT<Elt> M(rows, cols, p, data);
                eng = M.inverse(); eng_ok = true;
            } catch (const std::runtime_error&) {}
#if OC_ENABLE_LEGACY
            legacy_inverse_arm<Elt>(rows, cols, p, data, ok, eng_ok, Ainv, eng, tag);   // legacy/DenseElimLegacyArms.inc
#else
            // (S24 CL-L2) OFF: the existence check and the entrywise check of the retrofit half alone.
            check(eng_ok == ok, tag + " inverse existence vs retrofit");
            if (ok && eng_ok) check(same_buf_fm(Ainv, eng), tag + " inverse vs retrofit FlatMatrix::inverse");
#endif
        }
    }

    // --- solve_multi_rhs: basis = pivot columns of A, Y = random combinations --
    if (res.rank > 0 && rows <= 300) {
        const uint32_t nb = res.rank, ny = std::min<uint32_t>(5, rows);
        DenseBuf<Elt> Bm(rows, nb), Y(rows, ny), C(nb, ny);
        for (uint32_t j = 0; j < nb; ++j) for (uint32_t i = 0; i < rows; ++i) Bm.at(i, j) = A.at(i, res.pivots[j]);
        for (uint32_t j = 0; j < nb; ++j) for (uint32_t t = 0; t < ny; ++t) C.at(j, t) = static_cast<Elt>(rng.below(p));
        engine::ref::gemm<Elt>(p, cview(Bm), cview(C), view(Y), 0);
        const auto coords = engine::elim::solve_multi_rhs<Elt>(ctx, cview(Bm), cview(Y));
        bool ok = coords.size() == ny;
        for (uint32_t t = 0; t < ny && ok; ++t)
            for (uint32_t j = 0; j < nb; ++j) if (coords[t][j] != C.at(j, t)) { ok = false; break; }
        check(ok, tag + " solve_multi_rhs recovers the coefficients");
        if (o.legacy) {
            const auto basis = columns_of<Elt>(Bm, p);
            const auto vecs = columns_of<Elt>(Y, p);
            std::vector<std::vector<Elt>> eng;
#if OC_ENABLE_LEGACY
            legacy_solve_arm<Elt>(basis, vecs, coords, tag);   // legacy/DenseElimLegacyArms.inc
#endif
            eng = change_of_basis_fm_multi<Elt>(basis, vecs);
            check(eng == coords, tag + " solve_multi_rhs vs retrofit change_of_basis_fm_multi");
        }
        // out-of-span rejection: append a vector outside span when rank < rows
        if (res.rank < rows) {
            DenseBuf<Elt> Y2(rows, 1);
            // e_i for a row i that is not in the row span... use a random vector; with
            // overwhelming probability it is outside a proper subspace for p > 3
            bool rejected = false;
            for (uint32_t i = 0; i < rows; ++i) Y2.at(i, 0) = static_cast<Elt>(rng.below(p));
            try { (void)engine::elim::solve_multi_rhs<Elt>(ctx, cview(Bm), cview(Y2)); }
            catch (const std::runtime_error&) { rejected = true; }
            // verify against the oracle whether Y2 is in the span
            DenseBuf<Elt> aug(rows, nb + 1);
            for (uint32_t i = 0; i < rows; ++i) { for (uint32_t j = 0; j < nb; ++j) aug.at(i, j) = Bm.at(i, j); aug.at(i, nb) = Y2.at(i, 0); }
            const auto pv = engine::ref::rref<Elt>(p, view(aug));
            const bool outside = !pv.empty() && pv.back() == nb;
            check(rejected == outside, tag + " solve_multi_rhs out-of-span rejection");
        }
    }

    // --- complement_basis vs legacy factor_basis_block -------------------------
    if (rows <= 70 && rows > 0) {
        const uint32_t a = rows;
        const uint32_t nr = std::min<uint32_t>(a / 3 + 1, a), ng = std::min<uint32_t>(a / 2 + 2, a + 2);
        DenseBuf<Elt> Rl(a, nr), G(a, ng);
        for (uint32_t i = 0; i < a; ++i) {
            for (uint32_t j = 0; j < nr; ++j) Rl.at(i, j) = static_cast<Elt>(rng.below(p));
            for (uint32_t j = 0; j < ng; ++j) G.at(i, j) = (j % 3 == 2 && j > 0) ? G.at(i, j - 1) : static_cast<Elt>(rng.below(p));
        }
        DenseBuf<Elt> proj;
        const std::vector<uint32_t> gens = engine::elim::complement_basis<Elt>(ctx, cview(Rl), cview(G), &proj);
#if OC_ENABLE_LEGACY
        if (o.legacy) legacy_complement_arm<Elt>(a, p, Rl, G, gens, proj, tag);   // legacy/DenseElimLegacyArms.inc
#endif
    }

    // --- eliminate_against vs compute_elimination_factor_and_apply -------------
    if (rows > 0 && cols > 0 && rows <= 300) {
        // R: RREF of a random kR x cols matrix; B: a random rows x cols matrix
        const uint32_t kR = std::max<uint32_t>(1, std::min<uint32_t>(cols, rows / 2 + 1));
        const std::vector<Elt> rdata = random_entries<Elt>(rng, p, static_cast<size_t>(kR) * cols);
        DenseBuf<Elt> Rb = to_buf<Elt>(rdata, kR, cols);
        const engine::elim::RrefResult rr = engine::elim::rref<Elt>(ctx, view(Rb));
        const std::vector<Elt> bdata = random_entries<Elt>(rng, p, static_cast<size_t>(rows) * cols);
        DenseBuf<Elt> Bb = to_buf<Elt>(bdata, rows, cols);
        const DenseBuf<Elt> L = engine::elim::eliminate_against<Elt>(ctx, view(Bb), cview(Rb), rr.pivots, kR);
        // B_new has zeros at R's pivot columns
        bool zeros = true;
        for (uint32_t i = 0; i < rows && zeros; ++i) for (uint32_t c : rr.pivots) if (Bb.at(i, c) != 0) { zeros = false; break; }
        check(zeros, tag + " eliminate_against clears the pivot columns");
        if (o.legacy && rr.rank > 0) {
            std::vector<Elt> rref_flat(static_cast<size_t>(kR) * cols);
            for (uint32_t i = 0; i < kR; ++i) for (uint32_t j = 0; j < cols; ++j) rref_flat[static_cast<size_t>(i) * cols + j] = Rb.at(i, j);
            std::pair<FlatMatrixT<Elt>, FlatMatrixT<Elt>> eng;
#if OC_ENABLE_LEGACY
            legacy_eliminate_arm<Elt>(rows, cols, kR, p, bdata, rref_flat, Bb, L, tag);   // legacy/DenseElimLegacyArms.inc
#endif
            {
                const FlatMatrixT<Elt> B(rows, cols, p, bdata), Rf(kR, cols, p, rref_flat);
                eng = B.compute_elimination_factor_and_apply(Rf);
            }
            check(same_buf_fm(Bb, eng.first) && same_buf_fm(L, eng.second), tag + " eliminate_against vs retrofit compute_elimination_factor_and_apply");
        }
    }
}

template<class Elt>
void run_width(const std::vector<Elt>& primes, const char* wname) {
    const uint32_t sizes[] = {1, 2, 7, 16, 17, 64, 65, 129, 256, 517};
    for (Elt p : primes) {
        const ModCtx<Elt> ctx(p);
        for (uint32_t n : sizes) {
            std::vector<std::pair<uint32_t, uint32_t>> shapes{ {n, n} };
            if (n <= 129) { shapes.push_back({n, n + 5}); shapes.push_back({n + 5, n}); }
            for (const auto& sh : shapes) {
                const uint32_t rows = sh.first, cols = sh.second, full = std::min(rows, cols);
                std::vector<uint32_t> ranks{ 0, 1, std::min<uint32_t>(3, full), full / 2, full, full > 0 ? full - 1 : 0 };
                CaseOpts o;
                if (n >= 517) {
                    // reduced set: the big case runs only at the two production-shaped primes
                    // per width and at ranks {half, full}; no legacy (its scalar RREF of the
                    // 517 x 1034 augmented matrix alone costs more than the rest of the suite)
                    const bool big_prime = (sizeof(Elt) == 4) ? (p == Elt(134217757u) || p == Elt(4294967291u))
                                                              : (p == Elt(2305843009213693951ull));
                    if (!big_prime) continue;
                    ranks = { full / 2, full };
                    o.transform = false;
                } else if (n >= 256) {
                    ranks = { 0, full / 2, full };
                    o.transform = (p == Elt(134217757u));
                }
                std::sort(ranks.begin(), ranks.end());
                ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
                for (uint32_t r : ranks) {
                    Splitmix rng(0x5EEDu ^ (static_cast<uint64_t>(rows) << 40) ^ (static_cast<uint64_t>(cols) << 20) ^
                                 (static_cast<uint64_t>(r) << 8) ^ static_cast<uint64_t>(p));
                    const std::string tag = std::string(wname) + " p=" + str(static_cast<uint64_t>(p)) + " " + str(rows) +
                                            "x" + str(cols) + " r<=" + str(r);
                    run_case<Elt>(rng, ctx, rows, cols, r, o, tag);
                }
            }
        }
    }
}

// Blocking sweep: the SAME input must give identical bits for every panel /
// base / trsm_base choice, including the fully unblocked one.
template<class Elt>
void run_param_sweep(Elt p) {
    const ModCtx<Elt> ctx(p);
    const engine::elim::Params saved = engine::elim::params();
    const engine::elim::Params settings[] = {
        {1, 1, 1}, {2, 1, 2}, {3, 2, 1}, {8, 4, 4}, {16, 16, 5}, {33, 7, 16}, {64, 32, 32}, {1000, 1000, 1000}, saved };
    const uint32_t shapes[][2] = { {1, 1}, {5, 9}, {9, 5}, {40, 37}, {67, 130}, {130, 67}, {150, 150} };
    for (const auto& sh : shapes) {
        const uint32_t rows = sh[0], cols = sh[1], full = std::min(rows, cols);
        for (uint32_t r : { full / 3, full }) {
            Splitmix rng(0xA11CEu ^ (static_cast<uint64_t>(rows) << 24) ^ (static_cast<uint64_t>(cols) << 8) ^ r ^ static_cast<uint64_t>(p));
            const std::vector<Elt> data = (r >= full) ? random_entries<Elt>(rng, p, static_cast<size_t>(rows) * cols)
                                                      : ranked_matrix<Elt>(rng, ctx, rows, cols, r);
            const DenseBuf<Elt> A = to_buf<Elt>(data, rows, cols);
            DenseBuf<Elt> R0 = A;
            engine::elim::set_params(saved);
            const engine::elim::RrefResult res0 = engine::elim::rref<Elt>(ctx, view(R0));
            for (const auto& s : settings) {
                engine::elim::set_params(s);
                DenseBuf<Elt> R = A;
                const engine::elim::RrefResult res = engine::elim::rref<Elt>(ctx, view(R));
                check(res.pivots == res0.pivots && R.same_entries(R0),
                      "param sweep p=" + str(static_cast<uint64_t>(p)) + " " + str(rows) + "x" + str(cols) +
                      " panel=" + str(s.panel) + " base=" + str(s.base) + " trsm=" + str(s.trsm_base));
                DenseBuf<Elt> Rt(rows, cols), U(rows, rows);
                engine::elim::rref_with_transform<Elt>(ctx, cview(A), view(Rt), view(U));
                DenseBuf<Elt> UA(rows, cols);
                engine::ref::gemm<Elt>(p, cview(U), cview(A), view(UA), 0);
                check(same_buf(UA, R0) && same_buf(Rt, R0), "param sweep transform p=" + str(static_cast<uint64_t>(p)) +
                      " " + str(rows) + "x" + str(cols) + " panel=" + str(s.panel));
            }
            engine::elim::set_params(saved);
        }
    }
    engine::elim::set_params(saved);
}

} // namespace

// (S24 VAL-IMPL, VALSTREAM_audit_1030.md row 13) The grid is split between two names; their union is the pre-split
// grid, case for case (every case seeds its own Splitmix from its shape, rank and prime, so the split moves no input):
//   DenseElimValidation (standard): the u32 width at all five primes, the u64 width at 2^61-1, the parameter sweeps at
//     134217757 (u32) and 2^61-1 (u64);
//   DenseElimSlowValidation (slow): the u64 width at the five primes below 2^32 (2, 3, 134217757, 2^31-1, 2^32-5) and
//     the parameter sweeps at 2^32-5 and 2 (u32).
namespace {
uint32_t RunDenseElim(const char* name, bool slow) {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    g_fail = 0; g_checks = 0;
    std::cout << "=== " << name << " (plan D5/D5b: blocked elimination vs RefLinAlg vs legacy, entrywise"
              << (slow ? "; the u64 width below 2^32 and two parameter sweeps" : "") << ") ===" << std::endl;
    (void)oc::legacy::RunLegacyArm(name, "legacy");   // (S24 CL-L2) OFF: the ScopedLegacy / factor_basis_block halves are compiled out
    std::cout << "  params: panel=" << engine::elim::params().panel << " base=" << engine::elim::params().base
              << " trsm_base=" << engine::elim::params().trsm_base << std::endl;
    try {
        if (!slow) {
            run_width<uint32_t>({ 2u, 3u, 134217757u, 2147483647u, 4294967291u }, "u32");
            run_width<uint64_t>({ 2305843009213693951ull }, "u64");
            run_param_sweep<uint32_t>(134217757u);
            run_param_sweep<uint64_t>(2305843009213693951ull);
        } else {
            run_width<uint64_t>({ 2ull, 3ull, 134217757ull, 2147483647ull, 4294967291ull }, "u64");
            run_param_sweep<uint32_t>(4294967291u);
            run_param_sweep<uint32_t>(2u);
        }
    } catch (const std::exception& e) {
        fail(std::string("exception: ") + e.what());
    }
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  " << g_checks << " checks, " << g_fail << " failures, " << secs << " s" << std::endl;
    if (g_fail == 0) std::cout << name << ": PASS" << std::endl;
    else std::cout << name << ": FAIL (" << g_fail << ")" << std::endl;
    return g_fail;
}
} // namespace

uint32_t DenseElimValidation() { return RunDenseElim("DenseElimValidation", false); }

uint32_t DenseElimSlowValidation() { return RunDenseElim("DenseElimSlowValidation", true); }
