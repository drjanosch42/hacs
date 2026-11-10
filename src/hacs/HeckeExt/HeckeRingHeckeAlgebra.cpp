// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 14, WP-6) The Z/ell^K Hecke-algebra route -- see
// HeckeRingHeckeAlgebra.h for what is computed and why.
#include "HeckeRingHeckeAlgebra.h"

#include "HeckeHidaTower.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorFamily.h"   // tau_construction_mutex
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/LadicCell.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/LadicInduction.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>

namespace hecke_ext {
namespace ringalg {

using hecke::ordinary::u128;
using Poly64 = hecke::FpPolyT<uint64_t>;
using Elem = std::vector<uint64_t>;   // an endomorphism of (+)_s Z/ell^{b_s}: row-major reps, row s mod ell^{b_s}

namespace {

// ell^k, or 0 when it does not fit uint64_t.
using engine::scalar::powfit;
inline uint64_t PowFit(uint64_t ell, uint32_t k) { return powfit(ell, k); }

// (slot-23 G1 audit F10) the u32 / u64 word boundary of the native module route: ONE definition, read by RunModule
// (which word a run uses) and by LevelCache's WordOf (the "same word" rule of the run-level policy).
bool FitsU32Word(uint64_t q) { return q < (uint64_t(1) << 32); }

// (S24 SATPERF) the End(N) product's split: rows per pool task, and the smallest r split at all (the blocks' End rings,
// r <= ~16, never are; the whole module's, r = 2 d_k - 1 >= 25 from n = 142 on, are).
constexpr uint32_t kSatMulParMin = 24;
constexpr uint32_t kSatMulRowsPerTask = 8;
std::atomic<uint64_t> g_satperf_mul[3];        // narrow, wide, parallel (process totals, the EFFECTIVE line)
std::atomic<uint64_t> g_satperf_memo[2];       // hits, misses

// ---- the arithmetic of End(N), N = (+)_s Z/ell^{b_s} --------------------------
struct EndCtx {
    uint64_t ell = 0;
    uint32_t r = 0;
    uint32_t Kr = 0;                    // the embedding ring Z/ell^{Kr}, Kr = max(2, max b_s)
    uint64_t q = 0;                     // ell^{Kr}
    std::vector<uint32_t> b;
    std::vector<uint64_t> rowmod;       // ell^{b_s}
    std::vector<uint64_t> scale;        // ell^{Kr - b_s}

    explicit EndCtx(uint64_t ell_, const std::vector<uint32_t>& exps) : ell(ell_), b(exps) {
        r = static_cast<uint32_t>(exps.size());
        uint32_t kmax = 1;
        for (uint32_t x : exps) {
            ENGINE_CHECK_MSG(x >= 1, "HeckeRingHeckeAlgebra: module exponents must be >= 1");
            kmax = std::max(kmax, x);
        }
        Kr = std::max<uint32_t>(2, kmax);
        q = PowFit(ell, Kr);
        ENGINE_CHECK_MSG(q != 0, "HeckeRingHeckeAlgebra: ell^K >= 2^64 on the native End context (a digit module reads through DEndCtx)");
        for (uint32_t x : exps) {
            rowmod.push_back(PowFit(ell, x));
            scale.push_back(PowFit(ell, Kr - x));
        }
    }
    Elem identity() const {
        Elem e(static_cast<size_t>(r) * r, 0);
        for (uint32_t i = 0; i < r; ++i) e[static_cast<size_t>(i) * r + i] = 1 % rowmod[i];
        return e;
    }
    Elem canonical(Elem x) const {
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) x[static_cast<size_t>(i) * r + j] %= rowmod[i];
        return x;
    }
    Elem mul(const Elem& A, const Elem& B) const { return SatPerfEnabled() ? mul_fast(A, B) : mul_legacy(A, B); }
    // (S24 SATPERF) the same product: each output row is the exact sum_k A_ik B_kj reduced mod ell^{b_i} -- in uint64_t
    // when r * max A * max B < 2^64 (no partial sum can wrap), else in u128 exactly as mul_legacy -- and the rows are
    // split over engine::elim::parallel_hook() when one is installed and r >= kSatMulParMin (disjoint rows, the same sums).
    Elem mul_fast(const Elem& A, const Elem& B) const {
        Elem C(static_cast<size_t>(r) * r, 0);
        if (r == 0) return C;
        uint64_t ma = 0, mb = 0;
        for (uint64_t x : A) ma = std::max(ma, x);
        for (uint64_t x : B) mb = std::max(mb, x);
        const bool narrow = ma == 0 || mb == 0 || static_cast<u128>(ma) * mb <= static_cast<u128>(~uint64_t(0) / r);
        const bool big = (q >> 62) != 0;
        auto rows = [&](uint32_t i0, uint32_t i1) {
            if (narrow) {
                std::vector<uint64_t> acc(r);
                for (uint32_t i = i0; i < i1; ++i) {
                    std::fill(acc.begin(), acc.end(), 0);
                    const uint64_t* Ai = &A[static_cast<size_t>(i) * r];
                    for (uint32_t k = 0; k < r; ++k) {
                        const uint64_t a = Ai[k];
                        if (!a) continue;
                        const uint64_t* Bk = &B[static_cast<size_t>(k) * r];
                        for (uint32_t j = 0; j < r; ++j) acc[j] += a * Bk[j];
                    }
                    const uint64_t m = rowmod[i];
                    uint64_t* Ci = &C[static_cast<size_t>(i) * r];
                    for (uint32_t j = 0; j < r; ++j) Ci[j] = acc[j] % m;
                }
                return;
            }
            std::vector<u128> acc(r);
            for (uint32_t i = i0; i < i1; ++i) {
                const uint64_t m = rowmod[i];
                std::fill(acc.begin(), acc.end(), 0);
                for (uint32_t k = 0; k < r; ++k) {
                    const uint64_t a = A[static_cast<size_t>(i) * r + k];
                    if (!a) continue;
                    const uint64_t* Bk = &B[static_cast<size_t>(k) * r];
                    for (uint32_t j = 0; j < r; ++j) {
                        acc[j] += static_cast<u128>(a) * Bk[j];
                        if (big || (acc[j] >> 126)) acc[j] %= m;
                    }
                }
                for (uint32_t j = 0; j < r; ++j) C[static_cast<size_t>(i) * r + j] = static_cast<uint64_t>(acc[j] % m);
            }
        };
        const engine::elim::ParallelFor* par = engine::elim::parallel_hook();
        g_satperf_mul[narrow ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        if (par != nullptr && r >= kSatMulParMin) {
            const uint32_t nt = std::min<uint32_t>(64, (r + kSatMulRowsPerTask - 1) / kSatMulRowsPerTask);
            g_satperf_mul[2].fetch_add(1, std::memory_order_relaxed);
            (*par)(nt, [&](uint32_t t) {
                rows(static_cast<uint32_t>(static_cast<uint64_t>(r) * t / nt), static_cast<uint32_t>(static_cast<uint64_t>(r) * (t + 1) / nt));
            });
        } else {
            rows(0, r);
        }
        return C;
    }
    Elem mul_legacy(const Elem& A, const Elem& B) const {
        Elem C(static_cast<size_t>(r) * r, 0);
        std::vector<u128> acc(r);
        const bool big = (q >> 62) != 0;   // products may reach 2^128: reduce every step
        for (uint32_t i = 0; i < r; ++i) {
            const uint64_t m = rowmod[i];
            std::fill(acc.begin(), acc.end(), 0);
            for (uint32_t k = 0; k < r; ++k) {
                const uint64_t a = A[static_cast<size_t>(i) * r + k];
                if (!a) continue;
                const uint64_t* Bk = &B[static_cast<size_t>(k) * r];
                for (uint32_t j = 0; j < r; ++j) {
                    acc[j] += static_cast<u128>(a) * Bk[j];
                    if (big || (acc[j] >> 126)) acc[j] %= m;
                }
            }
            for (uint32_t j = 0; j < r; ++j) C[static_cast<size_t>(i) * r + j] = static_cast<uint64_t>(acc[j] % m);
        }
        return C;
    }
    // A + c B (c an integer in [0, q))
    Elem axpy(const Elem& A, const Elem& B, uint64_t c) const {
        Elem C(A.size());
        for (uint32_t i = 0; i < r; ++i) {
            const uint64_t m = rowmod[i];
            for (uint32_t j = 0; j < r; ++j) {
                const size_t k = static_cast<size_t>(i) * r + j;
                C[k] = static_cast<uint64_t>((static_cast<u128>(A[k] % m) + static_cast<u128>(c % m) * (B[k] % m)) % m);
            }
        }
        return C;
    }
    Elem scal(const Elem& A, uint64_t c) const {
        Elem C(A.size());
        for (uint32_t i = 0; i < r; ++i) {
            const uint64_t m = rowmod[i];
            for (uint32_t j = 0; j < r; ++j) {
                const size_t k = static_cast<size_t>(i) * r + j;
                C[k] = static_cast<uint64_t>(static_cast<u128>(A[k] % m) * (c % m) % m);
            }
        }
        return C;
    }
    // x - c * 1
    Elem minus_scalar(const Elem& A, uint64_t c) const {
        Elem C = A;
        for (uint32_t i = 0; i < r; ++i) {
            const uint64_t m = rowmod[i];
            const size_t k = static_cast<size_t>(i) * r + i;
            C[k] = static_cast<uint64_t>((static_cast<u128>(C[k] % m) + m - c % m) % m);
        }
        return C;
    }
    bool is_zero(const Elem& A) const {
        for (uint64_t x : A) if (x) return false;
        return true;
    }
    // The embedding End(N) -> (Z/ell^{Kr})^{r^2}, entry (i, j) -> rep * ell^{Kr - b_i}:
    // injective, and a Z-linear isomorphism onto its image (the subgroup lengths
    // of End(N) are read in the embedding).
    void embed_row(const Elem& A, uint64_t* out) const {
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                const size_t k = static_cast<size_t>(i) * r + j;
                out[k] = static_cast<uint64_t>(static_cast<u128>(A[k] % rowmod[i]) * scale[i] % q);
            }
    }
    Elem unembed(const uint64_t* v) const {
        Elem A(static_cast<size_t>(r) * r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                const size_t k = static_cast<size_t>(i) * r + j;
                ENGINE_CHECK_MSG(v[k] % scale[i] == 0, "HeckeRingHeckeAlgebra: a Howell row left the image of End(N)");
                A[k] = (v[k] / scale[i]) % rowmod[i];
            }
        return A;
    }
    // ---- (slot-15 WP-5) the context interface shared with DEndCtx ------------
    Elem op(const OpModule& N, size_t i) const {
        ENGINE_CHECK_MSG(!N.is_digit() && N.op[i].row_exp == b, "HeckeRingHeckeAlgebra: operator on a different module");
        return canonical(N.op[i].a);
    }
    RingMat to_ringmat(const Elem& x) const {
        RingMat M;
        M.ell = ell;
        M.row_exp = b;
        M.a = canonical(x);
        return M;
    }
    uint32_t rank_mod_ell(const Elem& x) const { return hecke::ordinary::RankModEll(to_ringmat(x)); }
    // 3 e^2 - 2 e^3 (the Newton step of the idempotent lift)
    Elem newton(const Elem& e2, const Elem& e3) const { return axpy(scal(e2, 3), e3, q - 2); }
    // The subgroup of End(N) generated by the (non-zero) `nz`: a generating set of
    // it (the Howell rows, un-embedded) and its length.  engine::elim::howell is
    // the one elimination (ENGINE_API.md Sec. 13.5).
    std::pair<std::vector<Elem>, uint32_t> span(const std::vector<const Elem*>& nz) const {
        const uint32_t cols = r * r;
        engine::DenseBuf<uint64_t> G(static_cast<uint32_t>(nz.size()), cols);
        for (uint32_t i = 0; i < nz.size(); ++i) embed_row(*nz[i], &G.at(i, 0));
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, Kr);
        const engine::elim::HowellResult<uint64_t> H = engine::elim::howell(ctx, engine::cview(G));
        std::vector<Elem> basis;
        basis.reserve(H.rows());
        std::vector<uint64_t> row(cols);
        for (uint32_t i = 0; i < H.rows(); ++i) {
            for (uint32_t j = 0; j < cols; ++j) row[j] = H.H.at(i, j);
            basis.push_back(unembed(row.data()));
        }
        return {basis, engine::elim::length(ctx, H)};
    }
    // The module exponents of e N (e an endomorphism of N) by the SMITH rule
    // (engine::elim::invariant_factors), descending -- the certificate's second
    // entry point beside the cyclic decomposition of localize().
    std::vector<uint32_t> image_invariants(const Elem& e) const {
        const uint32_t K = *std::max_element(b.begin(), b.end());
        ENGINE_CHECK(K >= 2);
        const uint64_t qK = PowFit(ell, K);
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, K);
        engine::DenseBuf<uint64_t> G(r, r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j)
                G.at(i, j) = static_cast<uint64_t>(static_cast<u128>(e[static_cast<size_t>(i) * r + j]) * PowFit(ell, K - b[i]) % qK);
        std::vector<uint32_t> f = engine::elim::invariant_factors(ctx, engine::cview(G));
        std::sort(f.rbegin(), f.rend());
        return f;
    }
    // e_m N as a submodule of (Z/ell^K)^r (g_i -> ell^{K - a_i}); the cyclic-summand
    // rule; the T_p restricted to it (every restricted column re-verified).
    void localize(const OpModule& N, const Elem& e, Block& B) const;
};

// ---- (slot-15 WP-5) the digit arithmetic of End(N), for ell^K >= 2^64 --------
// The same interface as EndCtx on engine::ladic's digit ring (W = uint64_t words,
// Z/ell^{Kr} for any Kr): an element is r^2 entries of L digit words each,
// entry (i, j) canonical mod ell^{b_i}.  Every elimination is engine::ladic's
// (howell, length, invariant_factors, cyclic_decomposition, cyclic_coordinates):
// the Sec. 13.5 - 13.6 objects in digit arithmetic, no new elimination.  The
// word width is fixed at 64 (ENGINE_LADIC_WIDTH selects the MODULE run's width;
// it moves no value, [E] sec:ring).
struct DEndCtx {
    using W = uint64_t;
    uint64_t ell = 0;
    uint32_t r = 0;
    uint32_t Kr = 0;
    std::vector<uint32_t> b;
    engine::ladic::DigitLayout<W> lay;
    uint32_t L = 0;
    std::unique_ptr<engine::ladic::Ring<W>> R;

    static uint32_t KrOf(const std::vector<uint32_t>& exps) {
        uint32_t kmax = 1;
        for (uint32_t x : exps) {
            ENGINE_CHECK_MSG(x >= 1, "HeckeRingHeckeAlgebra: module exponents must be >= 1");
            kmax = std::max(kmax, x);
        }
        return std::max<uint32_t>(2, kmax);
    }
    DEndCtx(uint64_t ell_, const std::vector<uint32_t>& exps)
        : ell(ell_), r(static_cast<uint32_t>(exps.size())), Kr(KrOf(exps)), b(exps),
          lay(engine::ladic::DigitLayout<W>::make(ell_, KrOf(exps))), L(lay.L),
          R(new engine::ladic::Ring<W>(lay)) {}

    size_t off(uint32_t i, uint32_t j) const { return (static_cast<size_t>(i) * r + j) * L; }
    Elem zero() const { return Elem(static_cast<size_t>(r) * r * L, 0); }
    Elem identity() const {
        Elem e = zero();
        for (uint32_t i = 0; i < r; ++i) R->one(&e[off(i, i)]);
        return e;
    }
    Elem canonical(Elem x) const {
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) R->low(&x[off(i, j)], &x[off(i, j)], b[i]);
        return x;
    }
    Elem mul(const Elem& A, const Elem& B) const {
        Elem C = zero();
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t k = 0; k < r; ++k) {
                const W* a = &A[off(i, k)];
                if (R->is_zero(a)) continue;
                for (uint32_t j = 0; j < r; ++j) {
                    const W* bb = &B[off(k, j)];
                    if (!R->is_zero(bb)) R->addmul(&C[off(i, j)], a, bb);
                }
            }
        return canonical(C);
    }
    Elem axpy(const Elem& A, const Elem& B, uint64_t c) const {
        std::vector<W> cw(L);
        R->from_u64(cw.data(), c);
        Elem C = A;
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) R->addmul(&C[off(i, j)], cw.data(), &B[off(i, j)]);
        return canonical(C);
    }
    Elem scal(const Elem& A, uint64_t c) const {
        std::vector<W> cw(L);
        R->from_u64(cw.data(), c);
        Elem C = zero();
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) R->mul(&C[off(i, j)], cw.data(), &A[off(i, j)]);
        return canonical(C);
    }
    Elem minus_scalar(const Elem& A, uint64_t c) const {
        std::vector<W> cw(L);
        R->from_u64(cw.data(), c);
        Elem C = A;
        for (uint32_t i = 0; i < r; ++i) R->sub(&C[off(i, i)], &C[off(i, i)], cw.data());
        return canonical(C);
    }
    bool is_zero(const Elem& A) const {
        for (W x : A) if (x) return false;
        return true;
    }
    Elem newton(const Elem& e2, const Elem& e3) const {
        const Elem a = scal(e2, 3), c = scal(e3, 2);
        Elem z = zero();
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) R->sub(&z[off(i, j)], &a[off(i, j)], &c[off(i, j)]);
        return canonical(z);
    }
    // the first nd base-ell digits (low -> high) of a ring element
    std::vector<uint32_t> ell_digits(const W* x, uint32_t nd) const {
        std::vector<uint32_t> a(Kr);
        R->to_ell_digits(a.data(), x);
        a.resize(nd, 0);
        return a;
    }
    void from_digits(W* z, const uint32_t* a, uint32_t nd) const {
        std::vector<uint32_t> full(Kr, 0);
        for (uint32_t d = 0; d < std::min(nd, Kr); ++d) full[d] = a[d];
        R->from_ell_digits(z, full.data());
    }
    Elem op(const OpModule& N, size_t pi) const {
        ENGINE_CHECK_MSG(N.is_digit() && N.exps == b, "HeckeRingHeckeAlgebra: digit operator on a different module");
        const std::vector<uint32_t>& d = N.dop[pi];
        ENGINE_CHECK_MSG(d.size() == static_cast<size_t>(r) * r * N.digit_K, "HeckeRingHeckeAlgebra: digit operator of the wrong size");
        Elem x = zero();
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j)
                from_digits(&x[off(i, j)], &d[(static_cast<size_t>(i) * r + j) * N.digit_K], N.digit_K);
        return canonical(x);
    }
    RingMat to_ringmat(const Elem&) const { return RingMat{}; }   // a digit idempotent is not exported (Block::e)
    uint32_t rank_mod_ell(const Elem& x) const {
        RingMat M;
        M.ell = ell;
        M.row_exp = b;
        M.a.resize(static_cast<size_t>(r) * r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) M.a[static_cast<size_t>(i) * r + j] = static_cast<uint64_t>(x[off(i, j)] % ell);
        return hecke::ordinary::RankModEll(M);
    }
    std::pair<std::vector<Elem>, uint32_t> span(const std::vector<const Elem*>& nz) const {
        engine::ladic::LadicMat<W> G(lay, static_cast<uint32_t>(nz.size()), r * r);
        std::vector<W> t(L);
        for (uint32_t k = 0; k < nz.size(); ++k)
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) {
                    R->shl(t.data(), &(*nz[k])[off(i, j)], Kr - b[i]);
                    G.set(k, i * r + j, t.data());
                }
        const engine::ladic::Howell<W> H = engine::ladic::howell(G);
        std::vector<Elem> basis;
        basis.reserve(H.rows());
        for (uint32_t k = 0; k < H.rows(); ++k) {
            Elem A = zero();
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) {
                    H.H.get(k, i * r + j, t.data());
                    ENGINE_CHECK_MSG(R->is_zero(t.data()) || R->valuation(t.data()) >= Kr - b[i],
                                     "HeckeRingHeckeAlgebra: a Howell row left the image of End(N) (digit route)");
                    R->shr(&A[off(i, j)], t.data(), Kr - b[i]);
                }
            basis.push_back(canonical(A));
        }
        return {basis, engine::ladic::length(H)};
    }
    std::vector<uint32_t> image_invariants(const Elem& e) const {
        const uint32_t K = *std::max_element(b.begin(), b.end());
        ENGINE_CHECK(K >= 2 && K == Kr);
        engine::ladic::LadicMat<W> G(lay, r, r);
        std::vector<W> t(L);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                R->shl(t.data(), &e[off(i, j)], K - b[i]);
                G.set(i, j, t.data());
            }
        std::vector<uint32_t> f = engine::ladic::invariant_factors(G);
        std::sort(f.rbegin(), f.rend());
        return f;
    }
    void localize(const OpModule& N, const Elem& e, Block& B) const;
};

// ---- the context-generic algebra (EndCtx native, DEndCtx digit) --------------
template<class C>
std::pair<std::vector<Elem>, uint32_t> Span(const C& c, const std::vector<Elem>& gens) {
    std::vector<const Elem*> nz;
    for (const Elem& g : gens)
        if (!c.is_zero(g)) nz.push_back(&g);
    if (nz.empty() || c.r == 0) return {{}, 0};
    return c.span(nz);
}

template<class C>
uint32_t SpanLength(const C& c, const std::vector<Elem>& gens) { return Span(c, gens).second; }

template<class C>
std::vector<Elem> Scaled(const C& c, const std::vector<Elem>& xs, uint64_t f) {
    std::vector<Elem> out;
    out.reserve(xs.size());
    for (const Elem& x : xs) out.push_back(c.scal(x, f));
    return out;
}

std::vector<Elem> Concat(std::vector<Elem> a, const std::vector<Elem>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// ---- F_ell polynomials: the idempotent polynomial of a root -------------------
Poly64 CharpolyModEll(uint64_t ell, const RingMat& A) {
    RingMat Abar;
    Abar.ell = ell;
    Abar.row_exp.assign(A.n(), 1);
    Abar.a.resize(A.a.size());
    for (size_t k = 0; k < A.a.size(); ++k) Abar.a[k] = A.a[k] % ell;
    const hida::Poly chi = hida::CharpolyFree(Abar);   // ascending, monic, mod ell
    return Poly64(ell, std::vector<uint64_t>(chi.begin(), chi.end()));
}

std::vector<uint64_t> RootsModEll(const Poly64& f, uint64_t ell) {
    std::vector<uint64_t> roots;
    for (uint64_t a = 0; a < ell; ++a)
        if (f.eval(a) == 0) roots.push_back(a);
    return roots;
}

// E with E = 1 mod (x - a)^mult, E = 0 mod chi / (x - a)^mult (mult >= 1).
Poly64 IdempotentPoly(const Poly64& chi, uint64_t a, uint64_t ell) {
    const Poly64 lin(ell, {(ell - a % ell) % ell, 1});
    Poly64 g = chi, h = Poly64::one(ell);
    uint32_t mult = 0;
    for (;;) {
        Poly64 qq(ell), rr(ell);
        Poly64::divrem(g, lin, qq, rr);
        if (!rr.is_zero()) break;
        g = qq;
        h = h * lin;
        ++mult;
    }
    ENGINE_CHECK_MSG(mult >= 1, "HeckeRingHeckeAlgebra: the eigenvalue is not a root of the characteristic polynomial");
    Poly64 u(ell), v(ell);
    const Poly64 d = Poly64::extgcd(h, g, u, v);   // u h + v g = 1
    ENGINE_CHECK(d == Poly64::one(ell));
    return Poly64::mod(v * g, chi);
}

// E(T) in End(N) by Paterson-Stockmeyer (coefficients lifted to [0, ell)).
template<class C>
Elem EvalPoly(const C& c, const Poly64& E, const Elem& T) {
    const Elem Z = c.scal(T, 0);
    const int64_t d = E.degree();
    if (d < 0) return Z;
    const uint32_t s = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(d + 1)))));
    std::vector<Elem> pw(s + 1);
    pw[0] = c.identity();
    for (uint32_t i = 1; i <= s; ++i) pw[i] = c.mul(pw[i - 1], T);
    const uint32_t chunks = static_cast<uint32_t>((d + 1 + s - 1) / s);
    Elem R = Z;
    for (uint32_t jj = chunks; jj-- > 0;) {
        Elem Cj = Z;
        for (uint32_t i = 0; i < s; ++i) {
            const uint64_t k = static_cast<uint64_t>(jj) * s + i;
            if (static_cast<int64_t>(k) > d) break;
            const uint64_t coef = E.coeff(static_cast<uint32_t>(k));
            if (coef) Cj = c.axpy(Cj, pw[i], coef);
        }
        R = (jj + 1 == chunks) ? Cj : c.axpy(c.mul(R, pw[s]), Cj, 1);
    }
    return R;
}

// The joint idempotent of `system` from the idempotent polynomials, Newton-lifted.
// e is a polynomial in the T_p with integer coefficients, so e lies in the image of
// T' = Z_ell[T_p : p in S] ([C, cor:stablecert], B1: e is the idempotent of m').
template<class C>
Elem JointIdempotent(const C& c, const OpModule& N, const std::vector<Poly64>& E) {
    Elem e = c.identity();
    for (size_t i = 0; i < N.primes.size(); ++i) e = c.mul(e, EvalPoly(c, E[i], c.op(N, i)));
    for (uint32_t it = 0;; ++it) {
        const Elem e2 = c.mul(e, e);
        if (e2 == e) break;
        ENGINE_CHECK_MSG(it < 64, "HeckeRingHeckeAlgebra: the Newton lift of e_m did not converge");
        const Elem e3 = c.mul(e2, e);
        // 3 e^2 - 2 e^3
        e = c.newton(e2, e3);
    }
    for (size_t i = 0; i < N.primes.size(); ++i) {
        const Elem T = c.op(N, i);
        ENGINE_CHECK_MSG(c.mul(e, T) == c.mul(T, e), "HeckeRingHeckeAlgebra: e_m does not commute with T_p");
    }
    return e;
}

std::vector<Poly64> IdempotentPolys(const OpModule& N, const std::vector<Poly64>& chi, const std::vector<uint64_t>& a) {
    ENGINE_CHECK(a.size() == N.primes.size() && chi.size() == N.primes.size());
    std::vector<Poly64> E;
    for (size_t i = 0; i < a.size(); ++i) E.push_back(IdempotentPoly(chi[i], a[i] % N.ell, N.ell));
    return E;
}

std::vector<Poly64> Charpolys(const OpModule& N) {
    ENGINE_CHECK_MSG(!N.is_digit(), "HeckeRingHeckeAlgebra: Charpolys reads a module mod ell (ReduceModEll first)");
    std::vector<Poly64> chi;
    for (const RingMat& T : N.op) chi.push_back(CharpolyModEll(N.ell, T));
    return chi;
}

void EndCtx::localize(const OpModule& N, const Elem& e, Block& B) const {
    const EndCtx& c = *this;
    uint32_t K = 1;
    for (uint32_t x : N.exps) K = std::max(K, x);
    const uint32_t r = c.r;
    const uint64_t q = PowFit(N.ell, K);
    std::vector<uint64_t> sc(r);
    for (uint32_t i = 0; i < r; ++i) sc[i] = PowFit(N.ell, K - N.exps[i]);
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(N.ell, K);
    engine::DenseBuf<uint64_t> G(r, r), Rl(r, 0);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j)
            G.at(i, j) = static_cast<uint64_t>(static_cast<u128>(e[static_cast<size_t>(i) * r + j]) * sc[i] % q);
    const engine::elim::CyclicDecomposition<uint64_t> dec =
        engine::elim::cyclic_decomposition(ctx, engine::cview(G), engine::cview(Rl));
    const uint32_t t = dec.rows();
    B.local.exps = dec.exponents;
    ENGINE_CHECK_MSG(t == c.rank_mod_ell(e), "HeckeRingHeckeAlgebra: e_m N has rank_F e_m generators");
    for (size_t pi = 0; pi < N.primes.size(); ++pi) {
        const Elem T = c.op(N, pi);
        engine::DenseBuf<uint64_t> Y(r, t);
        for (uint32_t s = 0; s < t; ++s) {
            std::vector<uint64_t> x(r), y(r, 0);
            for (uint32_t i = 0; i < r; ++i) {
                const uint64_t v = dec.basis.at(i, s);
                ENGINE_CHECK_MSG(v % sc[i] == 0, "HeckeRingHeckeAlgebra: a cyclic generator left the image of N");
                x[i] = (v / sc[i]) % c.rowmod[i];
            }
            for (uint32_t i = 0; i < r; ++i) {
                u128 acc = 0;
                for (uint32_t k = 0; k < r; ++k) {
                    acc += static_cast<u128>(T[static_cast<size_t>(i) * r + k]) * x[k];
                    if ((q >> 62) || (acc >> 126)) acc %= c.rowmod[i];
                }
                y[i] = static_cast<uint64_t>(acc % c.rowmod[i]);
                Y.at(i, s) = static_cast<uint64_t>(static_cast<u128>(y[i]) * sc[i] % q);
            }
        }
        const std::vector<std::vector<uint64_t>> co = engine::elim::cyclic_coordinates(ctx, dec, engine::cview(Y));
        RingMat X;
        X.ell = N.ell;
        X.row_exp = dec.exponents;
        X.a.assign(static_cast<size_t>(t) * t, 0);
        for (uint32_t s = 0; s < t; ++s) {
            // re-verify: sum_s' co[s][s'] b_{s'} == T b_s in the embedding
            for (uint32_t i = 0; i < r; ++i) {
                u128 acc = 0;
                for (uint32_t s2 = 0; s2 < t; ++s2) acc = (acc + static_cast<u128>(co[s][s2]) * dec.basis.at(i, s2)) % q;
                ENGINE_CHECK_MSG(static_cast<uint64_t>(acc) == Y.at(i, s), "HeckeRingHeckeAlgebra: T_p restricted to e_m N does not re-verify");
            }
            for (uint32_t s2 = 0; s2 < t; ++s2) {
                const uint64_t m2 = PowFit(N.ell, dec.exponents[s2]);
                X.a[static_cast<size_t>(s2) * t + s] = co[s][s2] % m2;
                // a homomorphism of (+) Z/ell^{b}: b_{s2} > b_s forces ell^{b_{s2} - b_s} | entry
                if (dec.exponents[s2] > dec.exponents[s])
                    ENGINE_CHECK_MSG(X.a[static_cast<size_t>(s2) * t + s] % PowFit(N.ell, dec.exponents[s2] - dec.exponents[s]) == 0,
                                     "HeckeRingHeckeAlgebra: the restricted operator is not a homomorphism");
            }
        }
        B.local.op.push_back(X);
    }
}

void DEndCtx::localize(const OpModule& N, const Elem& e, Block& B) const {
    const DEndCtx& c = *this;
    uint32_t K = 1;
    for (uint32_t x : N.exps) K = std::max(K, x);
    ENGINE_CHECK(K == Kr);
    using LM = engine::ladic::LadicMat<W>;
    LM G(lay, r, r), Rl(lay, r, 0);
    std::vector<W> t0(L), t1(L), bv(L);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            R->shl(t0.data(), &e[off(i, j)], K - b[i]);
            G.set(i, j, t0.data());
        }
    const engine::ladic::Cyclic<W> dec = engine::ladic::cyclic_decomposition(G, Rl);
    const uint32_t t = dec.rows();
    B.local.exps = dec.exponents;
    ENGINE_CHECK_MSG(t == c.rank_mod_ell(e), "HeckeRingHeckeAlgebra: e_m N has rank_F e_m generators (digit route)");
    uint32_t dK = 1;
    for (uint32_t x : dec.exponents) dK = std::max(dK, x);
    B.local.digit_K = dK;
    // the generators b_s in N's coordinates (entry i mod ell^{b_i})
    std::vector<Elem> xs(t, Elem(static_cast<size_t>(r) * L, 0));
    for (uint32_t s = 0; s < t; ++s)
        for (uint32_t i = 0; i < r; ++i) {
            dec.basis.get(i, s, t0.data());
            ENGINE_CHECK_MSG(R->is_zero(t0.data()) || R->valuation(t0.data()) >= K - b[i],
                             "HeckeRingHeckeAlgebra: a cyclic generator left the image of N (digit route)");
            W* xi = &xs[s][static_cast<size_t>(i) * L];
            R->shr(xi, t0.data(), K - b[i]);
            R->low(xi, xi, b[i]);
        }
    for (size_t pi = 0; pi < N.primes.size(); ++pi) {
        const Elem T = c.op(N, pi);
        LM Y(lay, r, t);
        for (uint32_t s = 0; s < t; ++s)
            for (uint32_t i = 0; i < r; ++i) {
                R->zero(t0.data());
                for (uint32_t k = 0; k < r; ++k) {
                    const W* a = &T[off(i, k)];
                    const W* xk = &xs[s][static_cast<size_t>(k) * L];
                    if (!R->is_zero(a) && !R->is_zero(xk)) R->addmul(t0.data(), a, xk);
                }
                R->low(t0.data(), t0.data(), b[i]);
                R->shl(t1.data(), t0.data(), K - b[i]);
                Y.set(i, s, t1.data());
            }
        const LM co = engine::ladic::cyclic_coordinates(dec, Y);   // t x t, row s = the coordinates of column s
        std::vector<uint32_t> X(static_cast<size_t>(t) * t * dK, 0);
        for (uint32_t s = 0; s < t; ++s) {
            // re-verify: sum_s' co[s][s'] b_{s'} == T b_s in the embedding
            for (uint32_t i = 0; i < r; ++i) {
                R->zero(t0.data());
                for (uint32_t s2 = 0; s2 < t; ++s2) {
                    co.get(s, s2, t1.data());
                    dec.basis.get(i, s2, bv.data());
                    R->addmul(t0.data(), t1.data(), bv.data());
                }
                Y.get(i, s, t1.data());
                ENGINE_CHECK_MSG(R->eq(t0.data(), t1.data()),
                                 "HeckeRingHeckeAlgebra: T_p restricted to e_m N does not re-verify (digit route)");
            }
            for (uint32_t s2 = 0; s2 < t; ++s2) {
                co.get(s, s2, t1.data());
                R->low(t1.data(), t1.data(), dec.exponents[s2]);
                if (dec.exponents[s2] > dec.exponents[s])
                    ENGINE_CHECK_MSG(R->is_zero(t1.data()) || R->valuation(t1.data()) >= dec.exponents[s2] - dec.exponents[s],
                                     "HeckeRingHeckeAlgebra: the restricted operator is not a homomorphism (digit route)");
                const std::vector<uint32_t> dg = c.ell_digits(t1.data(), dK);
                std::copy(dg.begin(), dg.end(), X.begin() + (static_cast<size_t>(s2) * t + s) * dK);
            }
        }
        B.local.dop.push_back(std::move(X));
    }
}

// (S24 SATPERF) e_in: the joint idempotent JointIdempotent(c, N, E) already computed (ModuleDerived), else computed here.
template<class C>
Block LocalizeWithT(const OpModule& N, const std::vector<Poly64>& E, const Elem* e_in = nullptr) {
    Block B;
    B.local.ell = N.ell;
    B.local.primes = N.primes;
    const C c(N.ell, N.exps);
    uint32_t K = 1;
    for (uint32_t x : N.exps) K = std::max(K, x);
    ENGINE_CHECK_MSG(K >= 2, "HeckeRingHeckeAlgebra: Localize needs a module over Z/ell^K, K >= 2");
    const Elem e = e_in ? *e_in : JointIdempotent(c, N, E);
    B.e = c.to_ringmat(e);
    if (c.rank_mod_ell(e) == 0) {
        for (size_t i = 0; i < N.primes.size(); ++i) B.local.op.push_back(RingMat{N.ell, {}, {}});
        return B;
    }
    c.localize(N, e, B);
    return B;
}

Block LocalizeWith(const OpModule& N, const std::vector<uint64_t>& system, const std::vector<Poly64>& E) {
    (void)system;
    return N.is_digit() ? LocalizeWithT<DEndCtx>(N, E) : LocalizeWithT<EndCtx>(N, E);
}

std::string JoinU(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    return os.str();
}
std::string JoinU64(const std::vector<uint64_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    return os.str();
}


// ---- the module run ----------------------------------------------------------
// H^1(Gamma; Sym^n Z/ell^K) and T_p on it by the shipped module route -- the
// recipe of hida::RunModuleCell (EquivariantCohomologyOnKT under
// ScopedCoefficientRing(K), HeckeOperatorOnKT's H_matrix_module(1)) on the
// element type Elt: uint32_t while ell^K < 2^32 (the console's narrow band),
// uint64_t above.  Both are the same objects; HeckeRingLoewyValidation (W)
// compares this runner with hida::RunModuleCell (uint64_t) block by block.
template<class Elt>
OpModule RunModuleT(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& primes) {
    ENGINE_CHECK_MSG(ell >= 5 && n >= 1 && K >= 2, "HeckeRingHeckeAlgebra: the module route needs ell >= 5, n >= 1, K >= 2");
    const hecke::ScopedCoefficientRing ring(K);
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const engine::ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeRingHeckeAlgebra", ell, K);
    LocalSystemSimplicialComplexT<Elt> lssc(*Kx, hecke::GenerateSL2ZRelationsRing<Elt>(*Kx, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<Elt> coh(&lssc);
    // (S23 G2-E) the lean retention on the stream route: this runner reads dec, C, piv, H only (and the operators'
    // A^{(1)}), so X / Zc and each operator's Theta / Theta_eq are released as soon as they are certified.
    const bool lean = engine::elim::module_stream_enabled();
    // (S24 G2-H) HECKE_MODULE_LEAN_PEAK (d): X / Zc freed inside the H^q loop already (no effect with the lever off)
    if (lean) coh.set_module_early_release(true);
    coh.compute_cohomology();
    if (lean) coh.release_module_intermediates();
    OpModule M;
    M.ell = ell;
    M.exps = coh.module_degree(1).dec.exponents;
    M.primes = primes;
    for (uint64_t p : primes) {
        std::unique_ptr<hecke::HeckeCosetTablesT<Elt>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<Elt>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<Elt>(p, 1, static_cast<Elt>(ell), n));
            // (S23 G2-F) on the stream route the module operator's tau table holds its chain records only: no
            // xi-term table and no rho pool (the module route reads tau() and evaluates rho_n mod ell^K itself).
            // The records are the Full table's, byte for byte; HECKE_MODULE_STREAM=0 builds the Full table.
            tau.reset(new hecke::HeckeTauTableT<Elt>(*Kx, *model, *cosets, static_cast<Elt>(ell), n,
                                                     hecke::TauVertexPolicy::GeometricNearest,
                                                     lean ? hecke::TauTableContents::ChainsOnly
                                                          : hecke::TauTableContents::Full));
        }
        engine::elim::module_stream_count(lean ? engine::elim::ModuleStreamEvent::TauChainsOnly
                                               : engine::elim::ModuleStreamEvent::TauPooled);
        hecke::HeckeOperatorOnKT<Elt> opk(&coh, cosets.get(), tau.get());
        if (lean) opk.set_module_retention(hecke::ModuleRetention::Lean);
        opk.compute();
        const engine::DenseBuf<Elt>& A = opk.H_matrix_module(1);
        RingMat T;
        T.ell = ell;
        T.row_exp = M.exps;
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) T.a.push_back(static_cast<uint64_t>(A.at(i, j)));
        M.op.push_back(hecke::ordinary::Canonical(T));
    }
    return M;
}

// (slot-15 WP-5) THE DIGIT MODULE RUN: H^1(Gamma; Sym^n Z/ell^K) and the T_p on
// its cyclic summands by the (C2) digit route (hecke::ladic_cell::RunCell) at
// exactly the working precision K -- its level driver, its canonicalisation, its
// certificates (the K-mode at K, the UCT check against the pre-pass type), its
// per-cell rule and caps.  `native_range` (ModuleRoute::Digit only, the oracle of
// HeckeRingLoewyValidation (C3)) lets it run below 2^64 as well.
OpModule RunModuleDigit(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& primes, bool native_range) {
    namespace lc = hecke::ladic_cell;
    lc::Request rq;
    rq.n_sym = n;
    rq.ell = ell;
    rq.module_K = K;
    rq.k = 2;                                   // the UCT reading's precision only
    rq.primes = primes;
    rq.hida = false;
    rq.width = lc::EngineLadicWidth();
    rq.precision_source = lc::PrecisionSource::Prepass;   // the same pre-pass e_max as ComputeRingLoewyCell
    rq.keep_hecke = true;
    rq.allow_native_range = native_range;
    const lc::CellRecord rec = lc::RunCell(rq);
    ENGINE_CHECK_MSG(rec.precision.working == K, "HeckeRingHeckeAlgebra: the digit run did not run at the requested K");
    // H^q(Z/ell^K) is exact at every K; the check that binds here is the UCT one: the digit run's
    // H^q(Z/ell^K) is the universal-coefficient reading of the pre-pass Z_(ell)-type at K (the K-mode
    // certificate at K is a statement about reading that type, and fails -- rightly -- at K <= e_max).
    ENGINE_CHECK_MSG(rec.precision.prepass && rec.precision.uct_match && (K <= rec.precision.e_max || rec.precision.certified),
                     "HeckeRingHeckeAlgebra: the digit run's certificates (the UCT type; the K-mode above e_max) did not hold");
    ENGINE_CHECK_MSG(rec.objects.dec_exponents.size() >= 2 && rec.objects.A.size() == primes.size(),
                     "HeckeRingHeckeAlgebra: the digit run returned no H^1 objects");
    OpModule M;
    M.ell = ell;
    M.exps = rec.objects.dec_exponents[1];
    M.primes = primes;
    M.digit_K = K;
    const size_t r = M.exps.size();
    for (size_t pi = 0; pi < primes.size(); ++pi) {
        ENGINE_CHECK(rec.objects.A[pi].size() >= 2);
        const std::vector<uint32_t>& ser = rec.objects.A[pi][1];   // rows, cols, then K base-ell digits per entry
        ENGINE_CHECK_MSG(ser.size() == 2 + r * r * K && ser[0] == r && ser[1] == r,
                         "HeckeRingHeckeAlgebra: the digit run's A^(1) has the wrong shape");
        std::vector<uint32_t> d(ser.begin() + 2, ser.end());
        for (size_t i = 0; i < r; ++i)            // entry (i, j) canonical mod ell^{b_i}
            for (size_t j = 0; j < r; ++j)
                for (uint32_t k = M.exps[i]; k < K; ++k) d[(i * r + j) * K + k] = 0;
        M.dop.push_back(std::move(d));
    }
    return M;
}

// ---- the algebra reading, context-generic ------------------------------------
template<class C>
AlgebraReading ReadAlgebraT(const OpModule& N, const std::vector<uint64_t>& system, uint32_t shift) {
    AlgebraReading R;
    const C c(N.ell, N.exps);
    std::vector<Elem> T;
    for (size_t i = 0; i < N.primes.size(); ++i) T.push_back(c.op(N, i));
    const Elem I = c.identity();
    // A = Z_ell[T_p] inside End(N): close the span of {1} under the T_p.
    auto sp = Span(c, {I});
    std::vector<Elem> basis = sp.first;
    uint32_t L = sp.second;
    for (uint32_t it = 0;; ++it) {
        ENGINE_CHECK_MSG(it <= c.r * c.r * c.Kr + 2, "HeckeRingHeckeAlgebra: the algebra closure did not terminate");
        std::vector<Elem> cand = basis;
        for (const Elem& Tp : T)
            for (const Elem& b : basis) cand.push_back(c.mul(Tp, b));
        auto nx = Span(c, cand);
        if (nx.second == L) break;
        basis = nx.first;
        L = nx.second;
    }
    const uint64_t f = PowFit(N.ell, shift);
    ENGINE_CHECK(f != 0);
    auto phi_len = [&](const std::vector<Elem>& xs) { return SpanLength(c, shift ? Scaled(c, xs, f) : xs); };
    const std::vector<Elem> ellA = Scaled(c, basis, N.ell);
    const uint32_t lenA = phi_len(basis);
    const uint32_t lenEll = phi_len(ellA);
    R.length = lenA;
    R.dim_mod_ell = lenA - lenEll;
    // m = (ell, T_p - a_p)
    std::vector<Elem> mg = {c.scal(I, N.ell)};
    for (size_t i = 0; i < T.size(); ++i) mg.push_back(c.minus_scalar(T[i], system[i] % N.ell));
    std::vector<Elem> mi = basis;
    uint32_t prev = lenA;
    for (uint32_t it = 0; prev != lenEll; ++it) {
        ENGINE_CHECK_MSG(it < 4096, "HeckeRingHeckeAlgebra: the m-adic filtration did not reach ell A");
        std::vector<Elem> prod;
        for (const Elem& g : mg)
            for (const Elem& b : mi) prod.push_back(c.mul(g, b));
        mi = Span(c, prod).first;
        const uint32_t cur = phi_len(Concat(mi, ellA));
        ENGINE_CHECK(cur <= prev);
        ENGINE_CHECK_MSG(cur < prev, "HeckeRingHeckeAlgebra: the algebra is not local at m (read a block, not a module)");
        R.hilbert.push_back(prev - cur);
        prev = cur;
    }
    uint32_t sum = 0;
    for (uint32_t h : R.hilbert) sum += h;
    ENGINE_CHECK_MSG(sum == R.dim_mod_ell, "HeckeRingHeckeAlgebra: the Hilbert function does not sum to dim A / ell");
    return R;
}

// ---- (slot-15 WP-5) the certificate's re-derivations --------------------------
// (4): A = the Z-span of the monomials prod_p T_p^{i_p}, 0 <= i_p < D, built prime by
// prime (A_{j+1} = span{ a T_p^i : a in A_j, i < D }).  Cayley-Hamilton on the free
// cover (Z/ell^K)^D of the D-generated block (the representing matrices lift the T_p)
// puts T_p^D in the span of the lower powers, so this IS Z[T_p : p in S] in End(N):
// a construction independent of ReadAlgebra's closure-to-a-fixed-point.
struct MonomialOut {
    uint32_t length = 0, dim = 0, generators = 0;
};
template<class C>
MonomialOut MonomialReadingT(const OpModule& B, uint32_t shift) {
    MonomialOut out;
    const C c(B.ell, B.exps);
    const uint32_t D = B.rank();
    const Elem I = c.identity();
    std::vector<Elem> basis = Span(c, {I}).first;
    out.generators = 1;
    for (size_t pi = 0; pi < B.primes.size(); ++pi) {
        const Elem T = c.op(B, pi);
        std::vector<Elem> pw = {I};
        for (uint32_t i = 1; i < D; ++i) pw.push_back(c.mul(pw.back(), T));
        std::vector<Elem> cand;
        cand.reserve(basis.size() * pw.size());
        for (const Elem& a : basis)
            for (const Elem& t : pw) cand.push_back(c.mul(a, t));
        out.generators = static_cast<uint32_t>(std::min<uint64_t>(uint64_t(out.generators) * D, 0xffffffffu));
        basis = Span(c, cand).first;
    }
    const uint64_t f = PowFit(B.ell, shift);
    ENGINE_CHECK(f != 0);
    auto phi_len = [&](const std::vector<Elem>& xs) { return SpanLength(c, shift ? Scaled(c, xs, f) : xs); };
    out.length = phi_len(basis);
    out.dim = out.length - phi_len(Scaled(c, basis, B.ell));
    return out;
}
MonomialOut MonomialReading(const OpModule& B, uint32_t shift) {
    return B.is_digit() ? MonomialReadingT<DEndCtx>(B, shift) : MonomialReadingT<EndCtx>(B, shift);
}

// (1): the T_p commute on the block (exactly); every T_p - a_p is nilpotent on the
// block mod ell (so the block's T'-system is a, and with rank_F e = the joint
// generalised eigenspace's dimension, e is the idempotent of m').
template<class C>
bool OpsCommuteT(const OpModule& B) {
    const C c(B.ell, B.exps);
    std::vector<Elem> T;
    for (size_t i = 0; i < B.primes.size(); ++i) T.push_back(c.op(B, i));
    for (size_t i = 0; i < T.size(); ++i)
        for (size_t j = i + 1; j < T.size(); ++j)
            if (!(c.mul(T[i], T[j]) == c.mul(T[j], T[i]))) return false;
    return true;
}
bool OpsCommute(const OpModule& B) { return B.is_digit() ? OpsCommuteT<DEndCtx>(B) : OpsCommuteT<EndCtx>(B); }

// (2): the exponents of e_m N by the Smith rule (not the cyclic decomposition).
template<class C>
std::vector<uint32_t> BlockInvariantsT(const OpModule& N, const std::vector<Poly64>& E, const Elem* e_in = nullptr) {
    const C c(N.ell, N.exps);
    if (e_in) return c.image_invariants(*e_in);   // (S24 SATPERF) the memo's JointIdempotent(c, N, E)
    const Elem e = JointIdempotent(c, N, E);
    return c.image_invariants(e);
}
template<class C>
Elem JointIdempotentOf(const OpModule& N, const std::vector<Poly64>& E) {
    const C c(N.ell, N.exps);
    return JointIdempotent(c, N, E);
}

} // namespace

OpModule RunModule(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& primes, ModuleRoute route) {
    const uint64_t q = PowFit(ell, K);
    auto labelled = [](OpModule M, const char* w) {
        M.word = w;
        return M;
    };
    if (route == ModuleRoute::Digit) return labelled(RunModuleDigit(n, ell, K, primes, /*native_range=*/q != 0), "digit");
    if (q == 0) {
        if (route == ModuleRoute::Hida)
            throw hecke::NotImplementedInThisBuild("HeckeRingHeckeAlgebra: the Hida module runner is uint64_t; ell^K >= 2^64 at (n, ell, K) = (" +
                                                   std::to_string(n) + ", " + std::to_string(ell) + ", " + std::to_string(K) +
                                                   ") runs on ModuleRoute::Native (the digit route)");
        return labelled(RunModuleDigit(n, ell, K, primes, /*native_range=*/false), "digit");
    }
    if (route == ModuleRoute::Hida) {
        std::vector<uint64_t> run = {ell};
        for (uint64_t p : primes)
            if (p != ell) run.push_back(p);
        const hida::ModuleCell mc = hida::RunModuleCell(n, ell, K, run);
        OpModule M;
        M.ell = ell;
        M.exps = mc.exps;
        M.primes = primes;
        for (uint64_t p : primes) M.op.push_back(mc.at(p));
        M.word = "hida";
        return M;
    }
    if (FitsU32Word(q)) return labelled(RunModuleT<uint32_t>(n, ell, K, primes), "u32");
    return labelled(RunModuleT<uint64_t>(n, ell, K, primes), "u64");
}

std::string HilbertText(const std::vector<uint32_t>& h) { return "[" + JoinU(h) + "]"; }

// ---------------------------------------------------------------------------
AlgebraReading ReadAlgebra(const OpModule& N, const std::vector<uint64_t>& system, uint32_t shift) {
    if (N.rank() == 0) return AlgebraReading{};
    ENGINE_CHECK(system.size() == N.primes.size() &&
                 (N.is_digit() ? N.dop.size() : N.op.size()) == N.primes.size());
    return N.is_digit() ? ReadAlgebraT<DEndCtx>(N, system, shift) : ReadAlgebraT<EndCtx>(N, system, shift);
}

OpModule ReduceModEll(const OpModule& N) {
    OpModule M;
    M.ell = N.ell;
    M.primes = N.primes;
    M.exps.assign(N.rank(), 1);
    if (N.is_digit()) {
        const size_t r = N.rank();
        for (const std::vector<uint32_t>& d : N.dop) {
            RingMat B;
            B.ell = N.ell;
            B.row_exp = M.exps;
            B.a.resize(r * r);
            for (size_t k = 0; k < r * r; ++k) B.a[k] = d[k * N.digit_K];   // the lowest base-ell digit
            M.op.push_back(B);
        }
        return M;
    }
    for (const RingMat& A : N.op) {
        RingMat B;
        B.ell = N.ell;
        B.row_exp = M.exps;
        B.a.resize(A.a.size());
        for (size_t k = 0; k < A.a.size(); ++k) B.a[k] = A.a[k] % N.ell;
        M.op.push_back(B);
    }
    return M;
}

namespace {
// RationalSystems with the characteristic polynomials of N / ell already computed (== Charpolys(ReduceModEll(N)))
std::vector<JointSystem> RationalSystemsWith(const OpModule& N, const std::vector<Poly64>& chi) {
    const OpModule Nb = ReduceModEll(N);
    const EndCtx c(N.ell, Nb.exps);
    struct Live { std::vector<uint64_t> a; Elem e; };
    std::vector<Live> live = {{{}, c.identity()}};
    for (size_t i = 0; i < Nb.primes.size(); ++i) {
        const Elem T = c.op(Nb, i);
        std::vector<Live> next;
        const std::vector<uint64_t> roots = RootsModEll(chi[i], N.ell);
        std::vector<Elem> ep;
        for (uint64_t a : roots) ep.push_back(EvalPoly(c, IdempotentPoly(chi[i], a, N.ell), T));
        for (const Live& L : live)
            for (size_t k = 0; k < roots.size(); ++k) {
                Elem e = c.mul(L.e, ep[k]);
                if (c.is_zero(e)) continue;
                std::vector<uint64_t> a = L.a;
                a.push_back(roots[k]);
                next.push_back({a, e});
            }
        live.swap(next);
    }
    std::vector<JointSystem> out;
    uint32_t total = 0;
    for (const Live& L : live) {
        const uint32_t d = c.rank_mod_ell(L.e);
        ENGINE_CHECK(d > 0);
        out.push_back({L.a, d});
        total += d;
    }
    ENGINE_CHECK_MSG(total <= N.rank(), "HeckeRingHeckeAlgebra: the joint eigenspaces overlap");
    std::sort(out.begin(), out.end(), [](const JointSystem& x, const JointSystem& y) { return x.a < y.a; });
    return out;
}
} // namespace

std::vector<JointSystem> RationalSystems(const OpModule& N) {
    return RationalSystemsWith(N, Charpolys(ReduceModEll(N)));
}

Block Localize(const OpModule& N, const std::vector<uint64_t>& system) {
    const std::vector<Poly64> chi = Charpolys(ReduceModEll(N));
    return LocalizeWith(N, system, IdempotentPolys(N, chi, system));
}

// ---------------------------------------------------------------------------
// (S24 SATPERF) the lever and the per-module memo -- see the header.
namespace {
// HECKE_SAT_PERF (default on; exactly "0" = the route of 4dd1a7e0; anything else warns and stays on)
constexpr engine::lever::Policy kSatPerfPolicy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING HeckeRingHeckeAlgebra: HECKE_SAT_PERF='%s' is neither 0 nor 1; the SATPERF route stays ON "
                                    "(only exactly 0 selects the rollback)\n"};
engine::lever::EnvLever g_satperf("HECKE_SAT_PERF", kSatPerfPolicy);
} // namespace

bool SatPerfEnabled() { return g_satperf.enabled(); }
ScopedSatPerf::ScopedSatPerf(bool on) : guard_(g_satperf, on) {}
SatPerfCounters SatPerfCounts() {
    SatPerfCounters c;
    c.memo_hits = g_satperf_memo[0].load();
    c.memo_misses = g_satperf_memo[1].load();
    c.mul_narrow = g_satperf_mul[0].load();
    c.mul_wide = g_satperf_mul[1].load();
    c.mul_parallel = g_satperf_mul[2].load();
    return c;
}
std::string SatPerfEffectiveLine(bool counters) {
    const SatPerfCounters c = SatPerfCounts();
    std::ostringstream os;
    os << g_satperf.effective_line("route", "memo+par", "legacy");
    if (g_satperf.is_set()) os << " (HECKE_SAT_PERF=" << g_satperf.raw() << ")";
    if (!counters) return os.str();   // (audit F9) the cell-start line: the configuration only, no (still zero) counters
    os << " memo_hits=" << c.memo_hits << " memo_misses=" << c.memo_misses << " mul_narrow=" << c.mul_narrow
       << " mul_wide=" << c.mul_wide << " mul_parallel=" << c.mul_parallel;
    return os.str();
}

struct ModuleDerived::Impl {
    const OpModule& N;
    bool have_chi = false, have_sys = false;
    std::vector<Poly64> chi;
    std::vector<JointSystem> sys;
    struct Entry {
        std::vector<Poly64> E;
        bool have_e = false, have_block = false, have_inv = false;
        Elem e;
        Block block;
        std::vector<uint32_t> inv;
    };
    std::map<std::vector<uint64_t>, Entry> per;
    explicit Impl(const OpModule& M) : N(M) {}
    static void count(bool hit) { g_satperf_memo[hit ? 0 : 1].fetch_add(1, std::memory_order_relaxed); }
    const std::vector<Poly64>& charpolys() {
        count(have_chi);
        if (!have_chi) { chi = Charpolys(ReduceModEll(N)); have_chi = true; }
        return chi;
    }
    Entry& entry(const std::vector<uint64_t>& a) {
        auto it = per.find(a);
        if (it != per.end()) return it->second;
        Entry& en = per[a];
        en.E = IdempotentPolys(N, charpolys(), a);
        return en;
    }
    const Elem& idempotent(Entry& en) {
        count(en.have_e);
        if (!en.have_e) {
            en.e = N.is_digit() ? JointIdempotentOf<DEndCtx>(N, en.E) : JointIdempotentOf<EndCtx>(N, en.E);
            en.have_e = true;
        }
        return en.e;
    }
};

ModuleDerived::ModuleDerived(const OpModule& N) : impl_(new Impl(N)) {}
ModuleDerived::~ModuleDerived() = default;
const OpModule& ModuleDerived::module() const { return impl_->N; }
const std::vector<JointSystem>& ModuleDerived::systems() {
    Impl& m = *impl_;
    Impl::count(m.have_sys);
    if (!m.have_sys) { m.sys = RationalSystemsWith(m.N, m.charpolys()); m.have_sys = true; }
    return m.sys;
}
const Block& ModuleDerived::block(const std::vector<uint64_t>& system) {
    Impl& m = *impl_;
    Impl::Entry& en = m.entry(system);
    Impl::count(en.have_block);
    if (!en.have_block) {
        // == Localize(N, system) == LocalizeWith(N, system, IdempotentPolys(N, chi, system)), e = JointIdempotent(c, N, E)
        const Elem& e = m.idempotent(en);
        en.block = m.N.is_digit() ? LocalizeWithT<DEndCtx>(m.N, en.E, &e) : LocalizeWithT<EndCtx>(m.N, en.E, &e);
        en.have_block = true;
    }
    return en.block;
}
const std::vector<uint32_t>& ModuleDerived::invariants(const std::vector<uint64_t>& system) {
    Impl& m = *impl_;
    Impl::Entry& en = m.entry(system);
    Impl::count(en.have_inv);
    if (!en.have_inv) {
        const Elem& e = m.idempotent(en);
        en.inv = m.N.is_digit() ? BlockInvariantsT<DEndCtx>(m.N, en.E, &e) : BlockInvariantsT<EndCtx>(m.N, en.E, &e);
        en.have_inv = true;
    }
    return en.inv;
}

// ---------------------------------------------------------------------------
// (slot-23 G1) ONE MODULE RUN PER CELL -- see the header; [C, lem:precisionreduction], [C, obl:precisionreuse].
OpModule ReduceToLevel(const OpModule& N, uint32_t K, bool native_out) {
    ENGINE_CHECK_MSG(K >= 1, "HeckeRingHeckeAlgebra: ReduceToLevel needs K >= 1");
    // [C, lem:precisionreduction](4): nothing determines a level above the levels run -- never derive upwards.  On a
    // native module a K at or above every exponent returns the module itself (min(a_i, K) = a_i, nothing is derived);
    // on a digit module a K above its precision digit_K is refused.
    ENGINE_CHECK_MSG(!N.is_digit() || K <= N.digit_K,
                     "HeckeRingHeckeAlgebra: ReduceToLevel above the digit module's precision ([C, lem:precisionreduction](4))");
    OpModule M;
    M.ell = N.ell;
    M.primes = N.primes;
    M.exps.resize(N.rank());
    for (uint32_t i = 0; i < N.rank(); ++i) M.exps[i] = std::min(N.exps[i], K);   // [C, lem:precisionreduction](2)
    const size_t r = N.rank();
    if (!N.is_digit()) {
        for (const RingMat& A : N.op) M.op.push_back(hecke::ordinary::Reduce(A, K));   // row i mod ell^{min(a_i, K)}
        return M;
    }
    if (native_out && PowFit(N.ell, K) != 0) {   // the native form below 2^64: value = sum of the kept digits times ell^d
        for (const std::vector<uint32_t>& d : N.dop) {
            RingMat B;
            B.ell = N.ell;
            B.row_exp = M.exps;
            B.a.assign(r * r, 0);
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < r; ++j) {
                    u128 v = 0, w = 1;
                    for (uint32_t k = 0; k < M.exps[i]; ++k) {
                        v += static_cast<u128>(d[(i * r + j) * N.digit_K + k]) * w;
                        w *= N.ell;
                    }
                    B.a[i * r + j] = static_cast<uint64_t>(v);
                }
            M.op.push_back(B);
        }
        return M;
    }
    M.digit_K = K;   // the lowest min(a_i, K) base-ell digits of row i, reshaped to K digits per entry
    for (const std::vector<uint32_t>& d : N.dop) {
        std::vector<uint32_t> e(r * r * K, 0);
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < r; ++j)
                for (uint32_t k = 0; k < M.exps[i]; ++k) e[(i * r + j) * K + k] = d[(i * r + j) * N.digit_K + k];
        M.dop.push_back(std::move(e));
    }
    return M;
}

void CheckLevelModule(const OpModule& M, uint32_t D, uint32_t K) {
    const std::string who = "HeckeRingHeckeAlgebra: [C, obl:precisionreuse](iii) ";
    ENGINE_CHECK_MSG(M.rank() == D, (who + "summands: the module read has " + std::to_string(M.rank()) + " summands, D = " +
                                     std::to_string(D) + " ([C, lem:precisionreduction](2): no summand vanishes)").c_str());
    for (uint32_t a : M.exps)
        ENGINE_CHECK_MSG(a >= 1 && a <= K, (who + "exponent: an exponent " + std::to_string(a) + " outside [1, K = " + std::to_string(K) + "]").c_str());
    ENGINE_CHECK_MSG((M.is_digit() ? M.dop.size() : M.op.size()) == M.primes.size(), (who + "summands: one operator per prime").c_str());
    if (!M.is_digit())
        for (const RingMat& A : M.op) ENGINE_CHECK_MSG(A.row_exp == M.exps, (who + "exponent: an operator's row exponents").c_str());
    ENGINE_CHECK_MSG(OpsCommute(M), (who + "commute: the T_p do not commute on the module read at K = " + std::to_string(K)).c_str());
}

void CheckRunModule(const OpModule& run) {
    // the case K = 1 of [C, lem:precisionreduction](2) is ReduceModEll (the route has always used it): entry for entry.
    // Run-only (neither side depends on a derived module): once per run.
    const OpModule one = ReduceToLevel(run, 1, /*native_out=*/true), ref = ReduceModEll(run);
    bool same = one.exps == ref.exps && one.op.size() == ref.op.size();
    for (size_t i = 0; same && i < one.op.size(); ++i) same = one.op[i] == ref.op[i];
    ENGINE_CHECK_MSG(same, "HeckeRingHeckeAlgebra: [C, obl:precisionreuse](iii) ReduceModEll: the K = 1 reduction of the run is not ReduceModEll");
}

void CheckDerivedModule(const OpModule& run, const OpModule& derived, uint32_t K) {
    const std::string who = "HeckeRingHeckeAlgebra: [C, obl:precisionreuse](iii) ";
    // (slot-23 G1 audit F3) the checks that tie the derived module to its run, before the structural ones (whose summand
    // check fires on a rank mismatch): (a) every entry canonical -- row i in [0, ell^{a_i}) on a native module, every
    // base-ell digit in [0, ell) and the digits at and above a_i zero on a digit module; (b) M|1 = run|1 entry for entry.
    if (derived.rank() == run.rank() && (derived.is_digit() ? derived.dop.size() : derived.op.size()) == derived.primes.size()) {
        const size_t r = derived.rank();
        if (!derived.is_digit()) {
            for (const RingMat& A : derived.op)
                for (size_t i = 0; i < r; ++i) {
                    const uint64_t q = PowFit(derived.ell, derived.exps[i]);
                    for (size_t j = 0; j < r; ++j)
                        ENGINE_CHECK_MSG(q == 0 || A.a[i * r + j] < q,
                                         (who + "range: an entry of row " + std::to_string(i) + " is not canonical mod ell^" +
                                          std::to_string(derived.exps[i]) + " at K = " + std::to_string(K)).c_str());
                }
        } else {
            for (const std::vector<uint32_t>& d : derived.dop)
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < r; ++j)
                        for (uint32_t k = 0; k < derived.digit_K; ++k) {
                            const uint32_t x = d[(i * r + j) * derived.digit_K + k];
                            ENGINE_CHECK_MSG(x < derived.ell && (k < derived.exps[i] || x == 0),
                                             (who + "range: a digit of row " + std::to_string(i) + " outside its exponent at K = " +
                                              std::to_string(K)).c_str());
                        }
        }
        const OpModule a = ReduceModEll(derived), b = ReduceModEll(run);
        bool same = a.exps == b.exps && a.op.size() == b.op.size();
        for (size_t i = 0; same && i < a.op.size(); ++i) same = a.op[i] == b.op[i];
        ENGINE_CHECK_MSG(same, (who + "ReduceModEll: the module derived at K = " + std::to_string(K) +
                                " does not reduce mod ell to the run's reduction").c_str());
    }
    CheckLevelModule(derived, run.rank(), K);
}

namespace {
// HECKE_PRECISION_REUSE (default on; exactly "0" = the per-K route; anything else warns and stays on)
constexpr engine::lever::Policy kReusePolicy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING HeckeRingHeckeAlgebra: HECKE_PRECISION_REUSE='%s' is neither 0 nor 1; the derived route stays ON "
                                    "(only exactly 0 selects the per-K route)\n"};
engine::lever::EnvArmRetiredLever g_reuse("HECKE_PRECISION_REUSE", kReusePolicy);
std::atomic<int> g_policy_override{-1};   // -1 = Auto
std::atomic<bool> g_inject_failure{false};
// the word a run at level k uses: 0 = u32 (ell^k < 2^32), 1 = u64, 2 = the digit route (ell^k >= 2^64)
int WordOf(uint64_t ell, uint32_t k) {
    const uint64_t q = PowFit(ell, k);
    return q == 0 ? 2 : (FitsU32Word(q) ? 0 : 1);
}
}  // namespace

bool PrecisionReuseEnabled() { return g_reuse.enabled(); }
RunLevelPolicy PrecisionRunPolicy() {
    const int o = g_policy_override.load(std::memory_order_relaxed);
    return o < 0 ? RunLevelPolicy::Auto : static_cast<RunLevelPolicy>(o);
}
const char* RunLevelPolicyName(RunLevelPolicy p) {
    switch (p) {
        case RunLevelPolicy::Auto: return "auto";
        case RunLevelPolicy::AtRequest: return "at_request";
        case RunLevelPolicy::Lazy: return "lazy";
        case RunLevelPolicy::SameWord: return "same_word";
        case RunLevelPolicy::Max: return "max";
    }
    return "?";
}
std::string PrecisionReuseEffectiveLine() {
    g_reuse.mark_said();   // a caller printing the full line: the cell's short line would repeat it
    return g_reuse.effective_line("levels", "derived", "per_K") + " policy=" + RunLevelPolicyName(PrecisionRunPolicy()) +
           " (sat / Loewy cells: one module run, the levels below it derived, [C, obl:precisionreuse]; "
           "per_K = one module run per level read, the oracle and rollback arm)";
}
ScopedPrecisionReuse::ScopedPrecisionReuse(bool on) : guard_(g_reuse, on) {}
ScopedRunLevelPolicy::ScopedRunLevelPolicy(RunLevelPolicy p) : saved_(g_policy_override.exchange(static_cast<int>(p))) {}
ScopedRunLevelPolicy::~ScopedRunLevelPolicy() { g_policy_override.store(saved_); }
ScopedInjectRunFailure::ScopedInjectRunFailure() : saved_(g_inject_failure.exchange(true)) {}
ScopedInjectRunFailure::~ScopedInjectRunFailure() { g_inject_failure.store(saved_); }

LevelCache::LevelCache(uint32_t n, uint64_t ell, const std::vector<uint64_t>& primes, ModuleRoute route, uint32_t K_cls,
                       uint32_t K_need)
    : n_(n), ell_(ell), primes_(primes), route_(route), K_cls_(K_cls), K_need_(K_need), policy_(PrecisionRunPolicy()),
      reuse_(PrecisionReuseEnabled()) {
    ENGINE_CHECK_MSG(K_cls >= 2 && K_need >= K_cls, "HeckeRingHeckeAlgebra: LevelCache needs 2 <= K_cls <= K_need");
    {   // the EFFECTIVE convention (the P4R / P4I pattern): when the variable is set (any value), the first cell of the process
        // says what it selects -- the variable's reading, not a scoped override -- so a silently ignored setting can be seen
        if (g_reuse.is_set()) g_reuse.say_once(std::cout, g_reuse.short_line("levels", "derived", "per_K"));
    }
    // [C, obl:precisionreuse](i): a level is derived only in the situation of [C, lem:precisionreduction](1) -- ell >= 5,
    // level 1, degree 1 (the cells' own preconditions), the native or the digit module route, a tau-table over Z at every
    // run level (enforced: the module route refuses a mod-ell fill, RefuseCoefficientRing).  Anything else: per K, by name.
    if (!reuse_) mode_ = "per_K:rollback(HECKE_PRECISION_REUSE=0)";
    else if (ell < 5) { reuse_ = false; mode_ = "per_K:ell<5"; }
    else if (route == ModuleRoute::Hida) { reuse_ = false; mode_ = "per_K:route_hida"; }
    else mode_ = std::string("reuse:") + RunLevelPolicyName(policy_);
}

uint32_t LevelCache::plan(uint32_t K) const {
    // [C, obl:precisionreuse](ii): a run at some K'' with K <= K'' <= K_need, and K_cls <= K* always (the free-rank reading
    // needs it: the first run is at least K_cls).  Which K'' is free and moves no printed value; it can change the word and
    // the peak, so the default never leaves the word of the request -- the word the per-K route runs that level in.
    const bool first = runs_.empty();
    const uint32_t lo = first ? std::max(K, K_cls_) : K;
    if (policy_ == RunLevelPolicy::Max) return std::max(lo, K_need_);
    if (policy_ == RunLevelPolicy::AtRequest) return lo;
    const int w = WordOf(ell_, lo);
    if (w == 2) return lo;   // the digit route: its cost grows with the digits, and its caps -- no look-ahead
    const bool look = policy_ == RunLevelPolicy::SameWord || !first || (policy_ == RunLevelPolicy::Auto && ell_ <= kEagerEllMax);
    if (!look) return lo;
    uint32_t hi = lo;
    while (hi < K_need_ && WordOf(ell_, hi + 1) == w) ++hi;
    return hi;
}

OpModule LevelCache::run_at(uint32_t K) {
    const auto tm = std::chrono::steady_clock::now();
    OpModule M = RunModule(n_, ell_, K, primes_, route_);
    secs_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tm).count();   // (vi) the runs only
    runs_.push_back(K);
    return M;
}

const OpModule& LevelCache::get(uint32_t K) {
    auto it = read_.find(K);
    if (it != read_.end()) return it->second;   // a reading already taken is kept ((i), (ii))
    ENGINE_CHECK_MSG(K >= 2, "HeckeRingHeckeAlgebra: the module route reads levels K >= 2 (K = 1 is ReduceModEll)");
    if (!reuse_) {   // the per-K route: one run at exactly K
        src_[K] = K;
        return read_.emplace(K, run_at(K)).first->second;
    }
    if (runs_.empty() || K > kstar_) {   // (ii) a level above K*: a (re)run at K'' >= K, which becomes K*
        const uint32_t K2 = plan(K);
        ENGINE_CHECK(K2 >= K && K2 <= std::max(K_need_, K));
        if (K2 == K) {   // the per-K route's own run at K: a failure here is the per-K route's failure, unchanged
            run_ = run_at(K2);
        } else {
            try {
                if (g_inject_failure.load()) throw std::runtime_error("injected run failure (ScopedInjectRunFailure, validation only)");
                run_ = run_at(K2);
            } catch (const std::exception& e) {
                // (i) a refusal, an exception or a cap of a run above the request: the per-K route for the levels still
                // unread, the readings already taken kept
                reuse_ = false;
                mode_ = "per_K:failure_at_K=" + std::to_string(K2) + "(" + e.what() + ")";
                // (slot-23 G1 audit F5) never silent: the fallback costs the speed-up and may hide a defect of the run
                std::fprintf(stderr, "WARNING HeckeRingHeckeAlgebra: precision reuse fell back to per_K at (n, ell) = (%u, %llu): "
                                     "the run at K = %u failed: %s\n", n_, static_cast<unsigned long long>(ell_), K2, e.what());
                src_[K] = K;
                return read_.emplace(K, run_at(K)).first->second;
            }
        }
        kstar_ = K2;
        CheckRunModule(run_);   // (iii) the K = 1 case, once per run
    }
    // (ii) K <= K*: read M_{K*}|K ([C, lem:precisionreduction](2)); K = K* gives the run module itself
    OpModule M = ReduceToLevel(run_, K, /*native_out=*/route_ == ModuleRoute::Native);
    CheckDerivedModule(run_, M, K);   // (iii)
    src_[K] = kstar_;
    return read_.emplace(K, std::move(M)).first->second;
}

uint32_t LevelCache::source(uint32_t K) const {
    auto it = src_.find(K);
    return it == src_.end() ? 0u : it->second;
}
std::vector<uint32_t> LevelCache::levels_read() const {
    std::vector<uint32_t> v;
    for (const auto& kv : read_) v.push_back(kv.first);
    return v;
}
std::vector<uint32_t> LevelCache::levels_run() const {
    std::vector<uint32_t> v = runs_;
    std::sort(v.begin(), v.end());
    return v;
}

// ---------------------------------------------------------------------------
const AlgebraReading* LoewyBlock::last_T() const { return history.empty() ? nullptr : &history.back().full; }

const AlgebraReading* LoewyBlock::certified_T() const {
    if (history.empty()) return nullptr;
    const LoewyStep& s = history.back();
    return s.has_free ? &s.free_part : &s.full;
}

const char* LoewyBlock::cert() const {
    if (torsion.empty()) return "full";
    return free_rank > 0 ? "free" : "torsion";
}

bool LoewyBlock::hidden() const {
    const AlgebraReading* t = certified_T();
    return A1.dim_mod_ell == 1 && t && t->dim_mod_ell > 1;
}

bool LoewyBlock::stable_family() const {
    return verdict == "stable" || verdict == "torsion_free_stable" || verdict == "torsion_stable" || verdict == "reduced_rank1";
}

void CheckWithinRankT(const AlgebraReading& reading, uint32_t rank_T) {
    ENGINE_CHECK_MSG(reading.dim_mod_ell <= rank_T,
                     "HeckeRingHeckeAlgebra: dim A_K/ell exceeds rank T (the rank formula or the free-rank reading is wrong)");
}

// ---------------------------------------------------------------------------
// (slot-15 WP-5) [C, cor:stablecert] (B1 form) on one block: see the header.  n_setting == nullptr:
// the algebraic core only (CheckStableReading, no whole-module check (2')).
namespace {
// (S24 SATPERF) dcls / dK: the per-cell memos of N_cls / N_K (ModuleDerived), or nullptr (re-derive here, the route of
// 4dd1a7e0).  Every object read from a memo is the one the nullptr branch computes, by the same function.
StableCertificate CertifyBody(const uint32_t* n_setting, const OpModule& N_cls, const OpModule& N_K, const LoewyBlock& claim,
                              uint32_t K_cls, const hecke::ordinary::TorsionPrepass* prepass = nullptr,
                              ModuleDerived* dcls = nullptr, ModuleDerived* dK = nullptr) {
    const std::string who = "HeckeRingHeckeAlgebra: CertifyStable ([C, cor:stablecert]) refuses the verdict '" + claim.verdict +
                             "' at the system (" + JoinU64(claim.system) + "): ";
    auto need = [&](bool ok, const std::string& what) { ENGINE_CHECK_MSG(ok, (who + what).c_str()); };
    StableCertificate cert;
    cert.verdict = claim.verdict;
    if (!n_setting) cert.label = "cor:stablecert (core only: no Setting check (2'))";   // audit M5
    need(claim.stable_family(), "not a stable-family verdict");
    need(N_cls.ell >= 5 && N_cls.ell == N_K.ell && N_cls.primes == N_K.primes && !N_cls.primes.empty(),
         "the Setting (ell >= 5, one prime set S) does not hold");
    need(claim.system.size() == N_cls.primes.size(), "the system is not aligned with S");
    for (uint64_t p : N_cls.primes) need(p >= 2 && hecke::IsPrimeU64(p), "S must consist of primes");
    auto maxexp = [](const OpModule& N) {
        uint32_t k = 0;
        for (uint32_t x : N.exps) k = std::max(k, x);
        return k;
    };
    need(K_cls >= 2 && maxexp(N_cls) == K_cls, "N_cls is not the module at K_cls");
    if (n_setting) {   // (2') K_cls separates the torsion from the free part of the WHOLE module: the summands of order
        // ell^{K_cls} are rank H^1(Gamma; V) (x) Z_ell = 2 d_k - 1 many (the dimension formula), the others are
        // exactly the torsion multiset of the pre-pass ([C, prop:torsionprepass], from rho_n(S), rho_n(T) alone,
        // no cochain complex) -- so K_cls > e_max, and no torsion summand hides at order ell^{K_cls}.
        const uint32_t n = *n_setting;
        need(n >= 2 && n % 2 == 0, "the Setting needs n > 0 even");
        // (slot-18 WP-5 (e)) the caller's pre-pass when ComputeRingLoewyCell hands it on (it ran once per cell)
        const hecke::ordinary::TorsionPrepass pp_own =
            prepass ? hecke::ordinary::TorsionPrepass{} : hecke::ordinary::TorsionExponentPrepass(n, N_cls.ell);
        const hecke::ordinary::TorsionPrepass& pp = prepass ? *prepass : pp_own;
        need(pp.n_sym == n && pp.ell == N_cls.ell, "the torsion pre-pass handed on is not at (n, ell)");
        uint32_t fr = 0;
        std::vector<uint32_t> tor;
        for (uint32_t x : N_cls.exps) {
            if (x == K_cls) ++fr;
            else tor.push_back(x);
        }
        std::sort(tor.rbegin(), tor.rend());
        std::vector<uint32_t> ptor = pp.torsion;
        std::sort(ptor.rbegin(), ptor.rend());
        need(K_cls > pp.e_max, "K_cls <= e_max");
        need(fr == hecke::ordinary::FreeRankH1(n), "the module at K_cls has " + std::to_string(fr) +
                                                       " summands of order ell^K_cls, the free rank of H^1 is " +
                                                       std::to_string(hecke::ordinary::FreeRankH1(n)));
        need(tor == ptor, "the module's torsion at K_cls is not the pre-pass torsion");
    }
    ENGINE_CHECK_MSG((!dcls || &dcls->module() == &N_cls) && (!dK || &dK->module() == &N_K),
                     "HeckeRingHeckeAlgebra: CertifyBody's memos are not of N_cls / N_K");
    std::vector<Poly64> E;
    Block Bc_own;
    if (!dcls) {
        const std::vector<Poly64> chiN = Charpolys(ReduceModEll(N_cls));
        E = IdempotentPolys(N_cls, chiN, claim.system);
        Bc_own = LocalizeWith(N_cls, claim.system, E);
    }

    // (1)-(2) at K_cls: the block, the Smith exponents, f and the torsion
    const Block& Bc = dcls ? dcls->block(claim.system) : Bc_own;
    need(Bc.local.rank() == claim.D && claim.D > 0, "the block at K_cls has " + std::to_string(Bc.local.rank()) +
                                                        " generators, the claim D = " + std::to_string(claim.D));
    {   // a CONSISTENCY check (audit M1): rank_F(e mod ell) equals RationalSystems' joint generalised
        // eigenspace dimension (the same projector construction reduced mod ell).  e = e_{m'} itself follows from
        // e in Z[T_p], e^2 = e, e != 0 and the nilpotency of T_p - a_p on e(N / ell), checked below.
        bool found = false;
        const std::vector<JointSystem> sys_own = dcls ? std::vector<JointSystem>{} : RationalSystems(N_cls);
        for (const JointSystem& js : dcls ? dcls->systems() : sys_own)
            if (js.a == claim.system) found = (js.dim == claim.D);
        need(found, "the joint generalised eigenspace of the system is not the block");
    }
    const std::vector<uint32_t> inv = dcls ? dcls->invariants(claim.system)
                                           : (N_cls.is_digit() ? BlockInvariantsT<DEndCtx>(N_cls, E) : BlockInvariantsT<EndCtx>(N_cls, E));
    std::vector<uint32_t> exps_c = Bc.local.exps;
    std::sort(exps_c.rbegin(), exps_c.rend());
    need(inv == exps_c, "the Smith exponents of e M at K_cls differ from the cyclic decomposition's");
    uint32_t f = 0;
    std::vector<uint32_t> torsion;
    for (uint32_t x : inv) {
        if (x == K_cls) ++f;
        else torsion.push_back(x);
    }
    std::sort(torsion.rbegin(), torsion.rend());
    need(f == claim.free_rank, "f = rank(e L') re-derived as " + std::to_string(f) + ", the claim " + std::to_string(claim.free_rank));
    need(torsion == claim.torsion, "the torsion exponents re-derived differ from the claim");
    const uint32_t rank_T = (f + 1) / 2;   // [C, lem:heckerankone], B1: every idempotent of T^L
    need(rank_T == claim.rank_T, "rank_T = ceil(f/2) = " + std::to_string(rank_T) + ", the claim " + std::to_string(claim.rank_T));
    const uint32_t t = torsion.empty() ? 0u : torsion.front();

    // the level, the block read at it, and the claimed reading
    uint32_t level = 0, shift = 0;
    const AlgebraReading* reading = nullptr;
    OpModule readmod;
    if (claim.verdict == "reduced_rank1") {
        need(torsion.empty() && claim.history.empty(), "reduced_rank1 needs a torsion-free block read at K = 1");
        need(OpsCommute(Bc.local), "the T_p do not commute on the block");
        readmod = ReduceModEll(Bc.local);
        level = 1;
        reading = &claim.A1;
    } else {
        need(!claim.history.empty(), "no recorded step");
        const LoewyStep& st = claim.history.back();
        const uint32_t K = claim.K_stab;
        need(st.K == K && maxexp(N_K) == K, "N_K is not the module at the claimed K_stab = " + std::to_string(K));
        std::vector<Poly64> EK;
        Block Bk_own;
        if (!dK) {
            EK = IdempotentPolys(N_K, Charpolys(ReduceModEll(N_K)), claim.system);
            Bk_own = LocalizeWith(N_K, claim.system, EK);
        }
        const Block& Bk = dK ? dK->block(claim.system) : Bk_own;
        need(Bk.local.rank() == claim.D, "the block at K has " + std::to_string(Bk.local.rank()) + " generators");
        need(OpsCommute(Bk.local), "the T_p do not commute on the block");
        std::vector<uint32_t> invK = dK ? dK->invariants(claim.system)
                                        : (N_K.is_digit() ? BlockInvariantsT<DEndCtx>(N_K, EK) : BlockInvariantsT<EndCtx>(N_K, EK));
        std::vector<uint32_t> want = torsion;
        if (K >= K_cls) {
            for (uint32_t i = 0; i < f; ++i) want.push_back(K);
        } else {
            need(torsion.empty(), "a torsion block read below K_cls");
            want.assign(f, K);
        }
        std::sort(want.rbegin(), want.rend());
        need(invK == want, "the Smith exponents of the block at K are not the free rank f at K plus the torsion");
        readmod = Bk.local;
        if (claim.verdict == "stable") {
            need(torsion.empty(), "'stable' on a block with torsion");
            level = K;
            reading = &st.full;
        } else if (claim.verdict == "torsion_free_stable") {
            need(!torsion.empty() && f > 0 && K >= K_cls && K > t, "'torsion_free_stable' needs torsion, f > 0, K >= K_cls");
            level = K - t;
            shift = t;
            need(st.has_free && st.free_level == level, "the free part was not read at level K - t");
            reading = &st.free_part;
        } else {   // torsion_stable: [C, lem:freepart](3)
            need(!torsion.empty() && f == 0 && K >= K_cls && K >= t, "'torsion_stable' needs a pure-torsion block at K >= t");
            level = K;
            reading = &st.full;
        }
    }
    {   // (1) T_p - a_p nilpotent on the block mod ell
        const OpModule Rb = ReduceModEll(claim.verdict == "reduced_rank1" ? Bc.local : readmod);
        const EndCtx c(Rb.ell, Rb.exps);
        for (size_t i = 0; i < Rb.primes.size(); ++i) {
            const Elem X = c.minus_scalar(c.op(Rb, i), claim.system[i] % Rb.ell);
            Elem P = c.identity();
            for (uint32_t k = 0; k < Rb.rank(); ++k) P = c.mul(P, X);
            need(c.is_zero(P), "T_" + std::to_string(Rb.primes[i]) + " - a_p is not nilpotent on the block mod ell");
        }
    }
    // (4) A'_K by the monomial construction
    const MonomialOut mono = MonomialReading(readmod, shift);
    need(mono.length == reading->length, "length(A'_K) re-derived as " + std::to_string(mono.length) + ", the route read " +
                                             std::to_string(reading->length));
    need(mono.dim == reading->dim_mod_ell, "dim A'_K/ell re-derived as " + std::to_string(mono.dim) + ", the route read " +
                                               std::to_string(reading->dim_mod_ell));
    uint32_t hsum = 0;
    for (uint32_t h : reading->hilbert) hsum += h;
    need(hsum == mono.dim && (reading->hilbert.empty() || reading->hilbert.front() == 1),
         "the recorded Hilbert function does not sum to dim A'_K/ell (or does not start at 1)");
    // (5) the criterion
    int64_t eps = 0;
    if (f > 0) {
        need(mono.dim == rank_T, "the criterion dim A'_K/ell = ceil(f/2) fails: " + std::to_string(mono.dim) + " != " +
                                     std::to_string(rank_T));
        // (6) eps_K = level * rank_T - length >= 0 ([C, prop:satdefect](1): eps^sat_K)
        eps = static_cast<int64_t>(level) * rank_T - static_cast<int64_t>(mono.length);
        need(eps >= 0, "eps_K < 0");
        need(eps == claim.eps, "eps_K re-derived as " + std::to_string(eps) + ", the claim " + std::to_string(claim.eps));
        // [C, prop:satdefect](2) along the recorded levels: 0 <= eps^sat non-decreasing, dim <= rank_T
        int64_t prev = -1;
        for (const LoewyStep& s : claim.history) {
            const bool tors = !torsion.empty();
            if (tors && !s.has_free) continue;
            const AlgebraReading& rd = tors ? s.free_part : s.full;
            const uint32_t lv = tors ? s.free_level : s.K;
            need(rd.dim_mod_ell <= rank_T, "a recorded step exceeds rank_T");
            const int64_t e = static_cast<int64_t>(lv) * rank_T - static_cast<int64_t>(rd.length);
            need(e >= 0 && e >= prev, "the recorded eps^sat is negative or decreasing ([C, prop:satdefect](2))");
            prev = e;
        }
        need(claim.verdict == "reduced_rank1" || prev == eps, "the last recorded step is not the certified one");
    } else {
        need(claim.eps == 0, "a pure-torsion block carries eps != 0");
    }
    cert.level = level;
    cert.f = f;
    cert.rank_T = rank_T;
    cert.length = mono.length;
    cert.dim = mono.dim;
    cert.torsion = torsion;
    cert.eps = eps;
    cert.monomial_generators = mono.generators;
    return cert;
}
} // namespace

StableCertificate CertifyStable(uint32_t n, const OpModule& N_cls, const OpModule& N_K, const LoewyBlock& claim, uint32_t K_cls) {
    return CertifyBody(&n, N_cls, N_K, claim, K_cls);
}

StableCertificate CertifyStable(uint32_t n, ModuleDerived& cls, ModuleDerived& K, const LoewyBlock& claim, uint32_t K_cls,
                                const hecke::ordinary::TorsionPrepass* prepass) {
    return CertifyBody(&n, cls.module(), K.module(), claim, K_cls, prepass, &cls, &K);
}

StableCertificate CheckStableReading(const OpModule& N_cls, const OpModule& N_K, const LoewyBlock& claim, uint32_t K_cls) {
    return CertifyBody(nullptr, N_cls, N_K, claim, K_cls);
}

LoewyCell ComputeRingLoewyCell(uint32_t n, uint64_t ell, const LoewyOptions& opts) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(ell >= 5 && n >= 1, "HeckeRingHeckeAlgebra: ell >= 5, n >= 1 (the module route)");
    ENGINE_CHECK_MSG(n % 2 == 0, "HeckeRingHeckeAlgebra: n must be even (odd n gives H^1 = 0 on this model, the -I gap)");
    ENGINE_CHECK_MSG(opts.kmax >= 2, "HeckeRingHeckeAlgebra: KMAX >= 2");
    LoewyCell C;
    C.n = n;
    C.ell = ell;
    C.kmax = opts.kmax;
    std::vector<uint64_t> primes = opts.primes.empty() ? std::vector<uint64_t>{2, 3, 5, 7, ell} : opts.primes;
    std::sort(primes.begin(), primes.end());
    primes.erase(std::unique(primes.begin(), primes.end()), primes.end());
    C.primes = primes;
    // [C, prop:torsionprepass]; (slot-18 WP-5 (e)) once per cell: the caller's run if handed in, else this one, and the
    // same object goes to every certificate below.
    const hecke::ordinary::TorsionPrepass pp_own =
        opts.prepass ? hecke::ordinary::TorsionPrepass{} : hecke::ordinary::TorsionExponentPrepass(n, ell);
    const hecke::ordinary::TorsionPrepass& pp = opts.prepass ? *opts.prepass : pp_own;
    ENGINE_CHECK_MSG(pp.n_sym == n && pp.ell == ell, "HeckeRingHeckeAlgebra: LoewyOptions::prepass is not at (n, ell)");
    C.e_max = pp.e_max;
    C.K_cls = std::max<uint32_t>(2, C.e_max + 1);
    const uint32_t kneed = std::max(C.kmax, C.K_cls);
    if (PowFit(ell, kneed) == 0 && opts.route == ModuleRoute::Hida)
        throw hecke::NotImplementedInThisBuild("HeckeRingHeckeAlgebra: ell^K >= 2^64 at (n, ell, K) = (" + std::to_string(n) +
                                               ", " + std::to_string(ell) + ", " + std::to_string(kneed) +
                                               ") on the Hida module runner; the digit route is ModuleRoute::Native");
    // (slot-23 G1) the level cache: one module run per cell, the levels below it derived ([C, lem:precisionreduction],
    // [C, obl:precisionreuse]); HECKE_PRECISION_REUSE=0 is the per-K route (one run per level read), the oracle.
    LevelCache levels(n, ell, primes, opts.route, C.K_cls, kneed);
    auto get = [&](uint32_t K) -> const OpModule& { return levels.get(K); };
    // (S24 SATPERF) the certificates' per-level memos (the route's own readings below keep N2's E, as before)
    std::map<uint32_t, std::unique_ptr<ModuleDerived>> memo;
    auto derived = [&](uint32_t K) -> ModuleDerived& {
        const OpModule& M = get(K);
        std::unique_ptr<ModuleDerived>& p = memo[K];
        if (!p) p.reset(new ModuleDerived(M));
        return *p;
    };
    const OpModule& N2 = get(2);
    C.exps_K2 = N2.exps;   // presentation-ordered: on a derived level the run's order ([C, obl:precisionreuse](iv): printed nowhere, and never from a derived level)
    const std::vector<Poly64> chi = Charpolys(ReduceModEll(N2));   // basis-free: M_K / ell = H^1(F_ell) at every K
    for (const JointSystem& js : RationalSystems(N2)) {
        LoewyBlock b;
        b.system = js.a;
        const std::vector<Poly64> E = IdempotentPolys(N2, chi, js.a);
        const Block Bc = LocalizeWith(get(C.K_cls), js.a, E);
        b.D = Bc.local.rank();
        ENGINE_CHECK_MSG(b.D == js.dim, "HeckeRingHeckeAlgebra: the block at K_cls has dim_F H^1(F_ell)_m generators");
        uint32_t t = 0;
        for (uint32_t x : Bc.local.exps) {
            if (x == C.K_cls) ++b.free_rank;
            else { b.torsion.push_back(x); t = std::max(t, x); }
        }
        std::sort(b.torsion.rbegin(), b.torsion.rend());
        const bool tors = !b.torsion.empty();
        b.rank_T = (b.free_rank + 1) / 2;
        b.A1 = ReadAlgebra(ReduceModEll(Bc.local), js.a);
        bool done = false;
        if (!tors && b.A1.dim_mod_ell == 1 && b.rank_T == 1) {
            b.verdict = "reduced_rank1";
            done = true;
        }
        for (uint32_t K = 2; K <= kneed && !done; ++K) {
            const Block Bk = (K == C.K_cls) ? Bc : LocalizeWith(get(K), js.a, E);
            ENGINE_CHECK(Bk.local.rank() == b.D);
            if (!tors)
                for (uint32_t x : Bk.local.exps) ENGINE_CHECK_MSG(x == K, "HeckeRingHeckeAlgebra: a free block has a torsion summand");
            LoewyStep st;
            st.K = K;
            st.full = ReadAlgebra(Bk.local, js.a);
            if (!tors) CheckWithinRankT(st.full, b.rank_T);   // audit S-2; the full reading of a torsion block is uncertified
            if (tors && K >= C.K_cls && b.free_rank > 0) {
                st.has_free = true;
                st.free_level = K - t;
                st.free_part = ReadAlgebra(Bk.local, js.a, t);
                CheckWithinRankT(st.free_part, b.rank_T);     // audit S-2, on the certified free part
            }
            b.history.push_back(st);
            b.K_stab = K;
            if (!tors) {
                b.eps = static_cast<int64_t>(K) * st.full.dim_mod_ell - st.full.length;
                if (st.full.dim_mod_ell == b.rank_T) { b.verdict = "stable"; done = true; }
            } else if (K >= C.K_cls) {
                if (b.free_rank == 0) { b.verdict = "torsion_stable"; b.eps = 0; done = true; }
                else {
                    b.eps = static_cast<int64_t>(st.free_level) * st.free_part.dim_mod_ell - st.free_part.length;
                    if (st.free_part.dim_mod_ell == b.rank_T) { b.verdict = "torsion_free_stable"; done = true; }
                }
            }
        }
        if (!done) b.verdict = tors ? "torsion_not_stable_at_KMAX" : "not_stable_at_KMAX";
        // (slot-15 WP-5) the certificate of the stopping verdict ([C, cor:stablecert], B1 form)
        if (b.stable_family()) {
            if (opts.certify) {
                // [C, obl:precisionreuse](vii): unchanged on the modules read; on a derived module `certified` rests on
                // [C, lem:precisionreduction](1) as well, and the run level each came from is recorded beside it
                const uint32_t KK = b.verdict == "reduced_rank1" ? C.K_cls : b.K_stab;
                if (SatPerfEnabled()) {   // (S24 SATPERF) the certificate's whole-module derivations from the per-level memos
                    ModuleDerived& dc = derived(C.K_cls);
                    ModuleDerived& dk = derived(KK);
                    (void)CertifyBody(&n, dc.module(), dk.module(), b, C.K_cls, &pp, &dc, &dk);
                } else {
                    (void)CertifyBody(&n, get(C.K_cls), get(KK), b, C.K_cls, &pp);
                }
                b.cert_from_cls = levels.source(C.K_cls);
                b.cert_from_K = levels.source(KK);
                b.stable_cert = "certified";
            } else {
                b.stable_cert = "reading";
            }
        }
        C.blocks.push_back(b);
    }
    C.K_run = levels.levels_read();   // (vi) Lv_read: the per-K route's list, byte-stable
    C.Lv_run = levels.levels_run();
    C.level_mode = levels.mode();
    C.module_seconds = levels.module_seconds();
    C.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return C;
}

std::string LoewyRingLine(const LoewyCell& c, const LoewyBlock& b) {
    std::ostringstream os;
    os << "LOEWY_RING n=" << c.n << " ell=" << c.ell << " primes={" << JoinU64(c.primes) << "} system=("
       << JoinU64(b.system) << ") D=" << b.D << " A1_hilbert=" << HilbertText(b.A1.hilbert);
    const AlgebraReading* T = b.last_T();
    os << " AK_hilbert=" << (T ? HilbertText(T->hilbert) : std::string("-")) << " K_stab=" << b.K_stab << " eps=" << b.eps
       << " free_rank=" << b.free_rank << " torsion=[" << JoinU(b.torsion) << "] rank_T=" << b.rank_T;
    if (!b.history.empty() && b.history.back().has_free)
        os << " free_level=" << b.history.back().free_level << " free_hilbert=" << HilbertText(b.history.back().free_part.hilbert);
    os << " len_AK=" << (T ? T->length : 0) << " e_max=" << c.e_max << " hidden=" << (b.hidden() ? 1 : 0)
       << " cert=" << b.cert() << " cf9=" << (!b.nonreduced_A1() ? "trivial" : (b.cf9_holds() ? "held" : "FAIL"))
       << " verdict=" << b.verdict << " stable_cert=" << b.stable_cert;   // (slot-15 WP-5) one new field, appended
    return os.str();
}

} // namespace ringalg
} // namespace hecke_ext
