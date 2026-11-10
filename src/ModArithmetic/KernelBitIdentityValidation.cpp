// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "KernelBitIdentityValidation.h"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DenseBuf.h"
#include "DenseView.h"
#include "FlatMatrix.h"
#include "Kernels.h"
#include "ModArithmetic.h"
#include "ModCtx.h"
#include "RefLinAlg.h"
#include "RnsKernels.h"
#include "../Engine/EngineTestUtil.h"

// See KernelBitIdentityValidation.h.

namespace {

using engine::DenseBuf;
using engine::DenseView;
using engine::Invariant;
using engine::Layout;
using engine::ModCtx;
using engine::Structure;
using engine::kernels::GemmOp;
using engine::kernels::KernelPath;
using engine::kernels::Side;

struct Splitmix : engine::tape::test::SplitMix64 {
    explicit Splitmix(uint64_t seed) : SplitMix64(seed) {}
    uint64_t below(uint64_t m) { return m <= 1 ? 0 : next() % m; }
};

uint32_t g_fail = 0;
uint32_t g_checks = 0;

void fail(const std::string& what) {
    ++g_fail;
    if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
}
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) fail(what);
}

using engine::tape::test::str;
const char* lname(Layout l) { return l == Layout::RowMajor ? "R" : "C"; }
using engine::tape::test::opname;
using engine::tape::test::opint;

const Layout kLayouts[2] = {Layout::RowMajor, Layout::ColMajor};
const uint32_t kSizes[] = {1, 2, 7, 16, 17, 64, 65, 129, 256, 517};

// A random buffer whose padding (ld > minor dim) is filled with garbage.
template<class Elt>
DenseBuf<Elt> random_buf(Splitmix& rng, Elt p, uint32_t rows, uint32_t cols, Layout layout,
                         uint32_t pad = 0, Structure st = Structure::Generic) {
    const uint32_t minor = layout == Layout::RowMajor ? cols : rows;
    DenseBuf<Elt> b(rows, cols, layout, minor + pad, Invariant::Canonical, st);
    Elt* d = b.data();
    for (size_t i = 0; i < b.footprint(); ++i) d[i] = static_cast<Elt>(rng.next());   // garbage everywhere
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) b.at(i, j) = static_cast<Elt>(rng.below(p));
    return b;
}

template<class Elt>
DenseBuf<Elt> clone_layout(const DenseBuf<Elt>& src) {   // same entries, same layout, no padding
    DenseBuf<Elt> b(src.rows, src.cols, src.layout);
    for (uint32_t i = 0; i < src.rows; ++i)
        for (uint32_t j = 0; j < src.cols; ++j) b.at(i, j) = src.at(i, j);
    b.invariant = src.invariant; b.structure = src.structure;
    return b;
}

template<class Elt>
FlatMatrixT<Elt> to_flat(DenseView<const Elt> v, Elt p) {
    std::vector<Elt> d(static_cast<size_t>(v.rows()) * v.cols());
    for (uint32_t i = 0; i < v.rows(); ++i)
        for (uint32_t j = 0; j < v.cols(); ++j) d[static_cast<size_t>(i) * v.cols() + j] = v.at(i, j);
    return FlatMatrixT<Elt>(v.rows(), v.cols(), p, d);
}

template<class Elt>
bool equal_flat(DenseView<const Elt> v, const FlatMatrixT<Elt>& f) {
    if (v.rows() != f.get_rows() || v.cols() != f.get_cols()) return false;
    for (uint32_t i = 0; i < v.rows(); ++i)
        for (uint32_t j = 0; j < v.cols(); ++j)
            if (v.at(i, j) != f.get_entry(i, j)) return false;
    return true;
}

template<class Elt>
std::string tagp(Elt p) { return std::string(sizeof(Elt) == 4 ? "u32" : "u64") + " p=" + str(p); }

// ---------------------------------------------------------------------------
// gemm: kernel (each available path) vs oracle, and vs FlatMatrix
// ---------------------------------------------------------------------------
template<class Elt>
void gemm_case(const ModCtx<Elt>& ctx, Splitmix& rng, uint32_t m, uint32_t k, uint32_t n,
               Layout la, Layout lb, Layout lc, GemmOp op, bool transA, bool transB,
               bool vs_flat, uint32_t pad) {
    const Elt p = ctx.p;
    // operands, optionally stored transposed and viewed back through the flag
    DenseBuf<Elt> Ab = transA ? random_buf<Elt>(rng, p, k, m, la, pad) : random_buf<Elt>(rng, p, m, k, la, pad);
    DenseBuf<Elt> Bb = transB ? random_buf<Elt>(rng, p, n, k, lb, pad) : random_buf<Elt>(rng, p, k, n, lb, pad);
    DenseBuf<Elt> C0 = random_buf<Elt>(rng, p, m, n, lc, pad);
    DenseView<const Elt> A = transA ? DenseView<const Elt>(Ab).transposed_view() : DenseView<const Elt>(Ab);
    DenseView<const Elt> B = transB ? DenseView<const Elt>(Bb).transposed_view() : DenseView<const Elt>(Bb);

    const std::string tag = "gemm " + tagp(p) + " m=" + str(m) + " k=" + str(k) + " n=" + str(n) +
                            " A" + lname(la) + (transA ? "t" : "") + " B" + lname(lb) + (transB ? "t" : "") +
                            " C" + lname(lc) + " " + opname(op) + " pad=" + str(pad);

    DenseBuf<Elt> Cref = clone_layout(C0);
    engine::ref::gemm<Elt>(p, A, B, DenseView<Elt>(Cref), opint(op));

    std::vector<KernelPath> paths = {KernelPath::Portable};
    if (engine::kernels::avx2_kernel_available()) paths.push_back(KernelPath::Avx2);   // (slot-24 AVX2K)
    if (engine::kernels::avx512_kernel_available()) paths.push_back(KernelPath::Avx512);
    for (KernelPath path : paths) {
        engine::kernels::set_kernel_path(path);
        DenseBuf<Elt> C = C0;   // copy keeps padding garbage too
        engine::kernels::gemm<Elt>(ctx, A, B, DenseView<Elt>(C), op);
        check(engine::same_entries(DenseView<const Elt>(C), DenseView<const Elt>(Cref)),
              tag + " [" + engine::kernels::kernel_path_name(path) + "] != oracle");
        // padding untouched
        bool pad_ok = true;
        if (pad) {
            for (uint32_t i = 0; i < C.major_dim(); ++i)
                for (uint32_t j = C.minor_dim(); j < C.ld; ++j)
                    if (C.data()[static_cast<size_t>(i) * C.ld + j] != C0.data()[static_cast<size_t>(i) * C0.ld + j]) pad_ok = false;
        }
        check(pad_ok, tag + " padding written");
    }
    engine::kernels::set_kernel_path(KernelPath::Auto);

    if (vs_flat && op == GemmOp::Set) {
        const FlatMatrixT<Elt> F = to_flat<Elt>(A, p) * to_flat<Elt>(B, p);
        check(equal_flat<Elt>(DenseView<const Elt>(Cref), F), tag + " oracle != FlatMatrix::operator*");
    }
}

template<class Elt>
void gemm_battery(const ModCtx<Elt>& ctx, Splitmix& rng, uint32_t max_size, uint32_t max_size_ops) {
    for (uint32_t s : kSizes) {
        if (s > max_size) continue;
        for (Layout la : kLayouts) for (Layout lb : kLayouts) {
            gemm_case<Elt>(ctx, rng, s, s, s, la, lb, Layout::RowMajor, GemmOp::Set, false, false, true, 0);
            if (s <= max_size_ops) {
                gemm_case<Elt>(ctx, rng, s, s, s, la, lb, Layout::ColMajor, GemmOp::Set, false, false, false, 3);
                gemm_case<Elt>(ctx, rng, s, s, s, la, lb, Layout::RowMajor, GemmOp::Add, false, false, false, 0);
                gemm_case<Elt>(ctx, rng, s, s, s, la, lb, Layout::ColMajor, GemmOp::Sub, false, false, false, 1);
            }
        }
        if (s <= max_size_ops) {
            // transposed views (a row-major buffer viewed transposed IS column-major, and vice versa)
            gemm_case<Elt>(ctx, rng, s, s, s, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, GemmOp::Set, true, false, true, 0);
            gemm_case<Elt>(ctx, rng, s, s, s, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, GemmOp::Set, false, true, true, 0);
            gemm_case<Elt>(ctx, rng, s, s, s, Layout::ColMajor, Layout::ColMajor, Layout::ColMajor, GemmOp::Add, true, true, false, 2);
        }
    }
    // rectangular, including the k > k-block shapes
    struct Shape { uint32_t m, k, n; };
    const Shape shapes[] = {{7, 517, 16}, {65, 1, 129}, {1, 64, 1}, {17, 16, 65}, {129, 7, 2},
                            {16, 1100, 16}, {3, 300, 33}, {2, 2, 517}, {517, 2, 2}};
    for (const Shape& sh : shapes) {
        if (std::max(sh.m, std::max(sh.k, sh.n)) > std::max(max_size, 1100u)) continue;
        if (sizeof(Elt) == 8 && sh.k > 300 && max_size < 256) continue;
        for (Layout la : kLayouts) for (Layout lb : kLayouts) {
            gemm_case<Elt>(ctx, rng, sh.m, sh.k, sh.n, la, lb, Layout::RowMajor, GemmOp::Set, false, false, true, 0);
            gemm_case<Elt>(ctx, rng, sh.m, sh.k, sh.n, la, lb, Layout::ColMajor, GemmOp::Sub, false, false, false, 5);
        }
    }
}

// ---------------------------------------------------------------------------
// gemv / gemm_nt
// ---------------------------------------------------------------------------
template<class Elt>
void gemv_battery(const ModCtx<Elt>& ctx, Splitmix& rng, uint32_t max_size) {
    const Elt p = ctx.p;
    for (uint32_t m : kSizes) for (uint32_t k : {1u, 7u, 65u, 129u, 517u, 1100u}) {
        if (m > max_size || (k > 129 && m > 65)) continue;
        for (Layout la : kLayouts) for (GemmOp op : {GemmOp::Set, GemmOp::Add, GemmOp::Sub}) {
            DenseBuf<Elt> A = random_buf<Elt>(rng, p, m, k, la, 2);
            DenseBuf<Elt> xcol = random_buf<Elt>(rng, p, k, 1, Layout::ColMajor);
            DenseBuf<Elt> xrow = random_buf<Elt>(rng, p, 1, k, Layout::RowMajor, 4);   // same vector as a row
            for (uint32_t t = 0; t < k; ++t) xrow.at(0, t) = xcol.at(t, 0);
            DenseBuf<Elt> y0 = random_buf<Elt>(rng, p, m, 1, Layout::ColMajor);
            DenseBuf<Elt> yref = clone_layout(y0);
            engine::ref::gemv<Elt>(p, DenseView<const Elt>(A), DenseView<const Elt>(xcol), DenseView<Elt>(yref), opint(op));
            const std::string tag = "gemv " + tagp(p) + " m=" + str(m) + " k=" + str(k) + " A" + lname(la) + " " + opname(op);
            DenseBuf<Elt> y1 = y0;
            engine::kernels::gemv<Elt>(ctx, DenseView<const Elt>(A), DenseView<const Elt>(xcol), DenseView<Elt>(y1), op);
            check(engine::same_entries(DenseView<const Elt>(y1), DenseView<const Elt>(yref)), tag + " (x column)");
            DenseBuf<Elt> y2 = y0;
            engine::kernels::gemv<Elt>(ctx, DenseView<const Elt>(A), DenseView<const Elt>(xrow).transposed_view(), DenseView<Elt>(y2), op);
            check(engine::same_entries(DenseView<const Elt>(y2), DenseView<const Elt>(yref)), tag + " (x row^T)");
            // gemm on the same data must agree too
            DenseBuf<Elt> y3 = y0;
            engine::kernels::gemm<Elt>(ctx, DenseView<const Elt>(A), DenseView<const Elt>(xcol), DenseView<Elt>(y3), op);
            check(engine::same_entries(DenseView<const Elt>(y3), DenseView<const Elt>(yref)), tag + " (via gemm n=1)");
        }
    }
    // gemm_nt: C = A * B^T with B given n x k
    for (uint32_t s : {1u, 7u, 17u, 65u, 129u}) {
        if (s > max_size) continue;
        for (Layout la : kLayouts) for (Layout lb : kLayouts) {
            DenseBuf<Elt> A = random_buf<Elt>(rng, p, s, s + 3, la);
            DenseBuf<Elt> B = random_buf<Elt>(rng, p, s + 1, s + 3, lb);   // n x k
            DenseBuf<Elt> C(s, s + 1), Cref(s, s + 1);
            engine::ref::gemm<Elt>(p, DenseView<const Elt>(A), DenseView<const Elt>(B).transposed_view(), DenseView<Elt>(Cref), 0);
            engine::kernels::gemm_nt<Elt>(ctx, DenseView<const Elt>(A), DenseView<const Elt>(B), DenseView<Elt>(C), GemmOp::Set);
            check(engine::same_entries(DenseView<const Elt>(C), DenseView<const Elt>(Cref)),
                  "gemm_nt " + tagp(p) + " s=" + str(s) + " A" + lname(la) + " B" + lname(lb));
        }
    }
}

// ---------------------------------------------------------------------------
// trmm / structured applies
// ---------------------------------------------------------------------------
template<class Elt>
void structured_battery(const ModCtx<Elt>& ctx, Splitmix& rng, uint32_t max_size) {
    const Elt p = ctx.p;
    for (uint32_t m : kSizes) {
        if (m > max_size) continue;
        for (uint32_t n : {1u, 17u, 64u}) for (Layout lt : kLayouts) for (bool upper : {true, false}) {
            // T carries garbage in the unread triangle AND on the diagonal
            DenseBuf<Elt> T = random_buf<Elt>(rng, p, m, m, lt, 1, upper ? Structure::UpperUnitri : Structure::LowerUnitri);
            DenseBuf<Elt> B = random_buf<Elt>(rng, p, m, n, lt == Layout::RowMajor ? Layout::ColMajor : Layout::RowMajor);
            DenseBuf<Elt> C(m, n, Layout::RowMajor), Cref(m, n, Layout::ColMajor);
            engine::ref::trmm<Elt>(p, DenseView<const Elt>(T), DenseView<const Elt>(B), DenseView<Elt>(Cref), upper);
            engine::kernels::trmm<Elt>(ctx, DenseView<const Elt>(T), DenseView<const Elt>(B), DenseView<Elt>(C));
            check(engine::same_entries(DenseView<const Elt>(C), DenseView<const Elt>(Cref)),
                  std::string("trmm ") + tagp(p) + " m=" + str(m) + " n=" + str(n) + " T" + lname(lt) + (upper ? " upper" : " lower"));
        }
    }
    // Generic-tagged T must be rejected
    {
        DenseBuf<Elt> T = random_buf<Elt>(rng, p, 4, 4, Layout::RowMajor), B(4, 2), C(4, 2);
        bool caught = false;
        try { engine::kernels::trmm<Elt>(ctx, DenseView<const Elt>(T), DenseView<const Elt>(B), DenseView<Elt>(C)); }
        catch (const std::logic_error&) { caught = true; }
        check(caught, "trmm must ENGINE_CHECK a Generic-tagged T");
    }
    // signed anti-diagonal and diagonal
    for (uint32_t d : {1u, 2u, 7u, 16u, 17u, 64u}) for (Layout lm : kLayouts) {
        if (d > max_size) continue;
        const uint32_t n = 13;
        DenseBuf<Elt> M = random_buf<Elt>(rng, p, d, d, lm, 0, Structure::SignedAntiDiag);
        DenseBuf<Elt> Mclean(d, d);
        for (uint32_t i = 0; i < d; ++i) {
            const Elt s = (rng.next() & 1) ? Elt(1) : Elt(p - 1);
            M.at(i, d - 1 - i) = s; Mclean.at(i, d - 1 - i) = s;
        }
        DenseBuf<Elt> D = random_buf<Elt>(rng, p, d, d, lm, 0, Structure::Diagonal);
        DenseBuf<Elt> Dclean(d, d);
        for (uint32_t i = 0; i < d; ++i) Dclean.at(i, i) = D.at(i, i);
        for (Side side : {Side::Left, Side::Right}) {
            const uint32_t br = side == Side::Left ? d : n, bc = side == Side::Left ? n : d;
            DenseBuf<Elt> B = random_buf<Elt>(rng, p, br, bc, lm == Layout::RowMajor ? Layout::ColMajor : Layout::RowMajor, 2);
            DenseBuf<Elt> C(br, bc, lm), Cref(br, bc);
            const std::string tag = tagp(p) + " d=" + str(d) + " M" + lname(lm) + (side == Side::Left ? " left" : " right");
            if (side == Side::Left) engine::ref::gemm<Elt>(p, DenseView<const Elt>(Mclean), DenseView<const Elt>(B), DenseView<Elt>(Cref), 0);
            else                    engine::ref::gemm<Elt>(p, DenseView<const Elt>(B), DenseView<const Elt>(Mclean), DenseView<Elt>(Cref), 0);
            engine::kernels::apply_signed_antidiag<Elt>(ctx, DenseView<const Elt>(M), DenseView<const Elt>(B), DenseView<Elt>(C), side);
            check(engine::same_entries(DenseView<const Elt>(C), DenseView<const Elt>(Cref)), "apply_signed_antidiag " + tag);
            if (side == Side::Left) engine::ref::gemm<Elt>(p, DenseView<const Elt>(Dclean), DenseView<const Elt>(B), DenseView<Elt>(Cref), 0);
            else                    engine::ref::gemm<Elt>(p, DenseView<const Elt>(B), DenseView<const Elt>(Dclean), DenseView<Elt>(Cref), 0);
            engine::kernels::apply_diagonal<Elt>(ctx, DenseView<const Elt>(D), DenseView<const Elt>(B), DenseView<Elt>(C), side);
            check(engine::same_entries(DenseView<const Elt>(C), DenseView<const Elt>(Cref)), "apply_diagonal " + tag);
        }
    }
}

// ---------------------------------------------------------------------------
// transpose / copy / axpy / scal / canonicalise / swap_rows / guards
// ---------------------------------------------------------------------------
template<class Elt>
void elementwise_battery(const ModCtx<Elt>& ctx, Splitmix& rng, uint32_t max_size) {
    const Elt p = ctx.p;
    for (uint32_t m : kSizes) for (uint32_t n : {1u, 7u, 16u, 65u}) {
        if (m > max_size) continue;
        for (Layout ls : kLayouts) for (Layout ld : kLayouts) {
            DenseBuf<Elt> S = random_buf<Elt>(rng, p, m, n, ls, 3);
            DenseBuf<Elt> T(n, m, ld, (ld == Layout::RowMajor ? m : n) + 5), Tref(n, m);
            engine::ref::transpose<Elt>(DenseView<const Elt>(S), DenseView<Elt>(Tref));
            engine::kernels::transpose<Elt>(DenseView<const Elt>(S), DenseView<Elt>(T));
            const std::string tag = tagp(p) + " m=" + str(m) + " n=" + str(n) + " " + lname(ls) + "->" + lname(ld);
            check(engine::same_entries(DenseView<const Elt>(T), DenseView<const Elt>(Tref)), "transpose " + tag);
            DenseBuf<Elt> Cp(m, n, ld);
            engine::kernels::copy<Elt>(DenseView<const Elt>(S), DenseView<Elt>(Cp));
            check(engine::same_entries(DenseView<const Elt>(Cp), DenseView<const Elt>(S)), "copy " + tag);
            // transpose of a transposed view is a relayout
            DenseBuf<Elt> R(m, n, ld);
            engine::kernels::transpose<Elt>(DenseView<const Elt>(S).transposed_view(), DenseView<Elt>(R));
            check(engine::same_entries(DenseView<const Elt>(R), DenseView<const Elt>(S)), "transpose(view^T) " + tag);
        }
        // axpy / scal against the legacy primitives
        for (Layout l : kLayouts) for (bool sub : {false, true}) {
            DenseBuf<Elt> X = random_buf<Elt>(rng, p, m, n, l, 1);
            DenseBuf<Elt> Y = random_buf<Elt>(rng, p, m, n, l == Layout::RowMajor ? Layout::ColMajor : Layout::RowMajor);
            const Elt alpha = static_cast<Elt>(rng.below(p));
            DenseBuf<Elt> Yref = clone_layout(Y);
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) {
                const Elt t = ModArithmetic::mod_mul_w(alpha, X.at(i, j), p);
                Yref.at(i, j) = sub ? ModArithmetic::mod_sub_w(Y.at(i, j), t, p) : ModArithmetic::mod_add_w(Y.at(i, j), t, p);
            }
            engine::kernels::axpy<Elt>(ctx, alpha, DenseView<const Elt>(X), DenseView<Elt>(Y), sub);
            check(engine::same_entries(DenseView<const Elt>(Y), DenseView<const Elt>(Yref)),
                  std::string("axpy ") + tagp(p) + " m=" + str(m) + " n=" + str(n) + (sub ? " sub" : " add"));
            DenseBuf<Elt> Xs = clone_layout(X), Xref = clone_layout(X);
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) Xref.at(i, j) = ModArithmetic::mod_mul_w(alpha, X.at(i, j), p);
            engine::kernels::scal<Elt>(ctx, alpha, DenseView<Elt>(Xs));
            check(engine::same_entries(DenseView<const Elt>(Xs), DenseView<const Elt>(Xref)), "scal " + tagp(p) + " m=" + str(m));
        }
    }
    // canonicalise Lazy2p (representable only when 2p fits Elt)
    if (static_cast<double>(p) * 2.0 < static_cast<double>(std::numeric_limits<Elt>::max())) {
        DenseBuf<Elt> L(33, 17, Layout::ColMajor, 40, Invariant::Lazy2p);
        DenseBuf<Elt> want(33, 17);
        for (uint32_t i = 0; i < 33; ++i) for (uint32_t j = 0; j < 17; ++j) {
            const Elt v = static_cast<Elt>(rng.below(2ull * p));
            L.at(i, j) = v; want.at(i, j) = v % p;
        }
        // arithmetic on a Lazy2p operand must be refused
        {
            DenseBuf<Elt> B(17, 3), C(33, 3);
            bool caught = false;
            try { engine::kernels::gemm<Elt>(ctx, DenseView<const Elt>(L), DenseView<const Elt>(B), DenseView<Elt>(C)); }
            catch (const std::logic_error&) { caught = true; }
            check(caught, "gemm must ENGINE_CHECK a Lazy2p operand " + tagp(p));
        }
        DenseView<Elt> Lv(L);
        engine::kernels::canonicalise<Elt>(ctx, Lv.sub(0, 10, 0, 17));
        check(L.invariant == Invariant::Lazy2p, "canonicalise of a sub-view must not retag the buffer " + tagp(p));
        engine::kernels::canonicalise<Elt>(ctx, Lv);
        check(L.invariant == Invariant::Canonical && engine::same_entries(DenseView<const Elt>(L), DenseView<const Elt>(want)),
              "canonicalise Lazy2p " + tagp(p));
    }
    // swap_rows
    {
        DenseBuf<Elt> S = random_buf<Elt>(rng, p, 9, 5, Layout::ColMajor, 2);
        DenseBuf<Elt> S0 = clone_layout(S);
        engine::kernels::swap_rows<Elt>(DenseView<Elt>(S), 2, 7);
        bool ok = true;
        for (uint32_t i = 0; i < 9; ++i) for (uint32_t j = 0; j < 5; ++j) {
            const uint32_t src = i == 2 ? 7 : i == 7 ? 2 : i;
            if (S.at(i, j) != S0.at(src, j)) ok = false;
        }
        check(ok, "swap_rows " + tagp(p));
    }
    // ENGINE_CHECK guards: shape mismatch, aliasing
    {
        DenseBuf<Elt> A = random_buf<Elt>(rng, p, 4, 5, Layout::RowMajor), B = random_buf<Elt>(rng, p, 6, 3, Layout::RowMajor), C(4, 3);
        bool caught = false;
        try { engine::kernels::gemm<Elt>(ctx, DenseView<const Elt>(A), DenseView<const Elt>(B), DenseView<Elt>(C)); }
        catch (const std::logic_error&) { caught = true; }
        check(caught, "gemm must ENGINE_CHECK a shape mismatch " + tagp(p));
        DenseBuf<Elt> Sq = random_buf<Elt>(rng, p, 8, 8, Layout::RowMajor);
        DenseView<Elt> whole(Sq);
        caught = false;
        try { engine::kernels::gemm<Elt>(ctx, DenseView<const Elt>(whole.sub(0, 4, 0, 8)), DenseView<const Elt>(whole.sub(0, 8, 0, 4)), whole.sub(0, 4, 0, 4)); }
        catch (const std::logic_error&) { caught = true; }
        check(caught, "gemm must ENGINE_CHECK an aliased C " + tagp(p));
        // disjoint windows of ONE buffer are allowed (Schur-update shape)
        DenseBuf<Elt> Sref = clone_layout(Sq);
        engine::ref::gemm<Elt>(p, DenseView<const Elt>(Sref).sub(4, 4, 0, 4), DenseView<const Elt>(Sref).sub(0, 4, 4, 4), DenseView<Elt>(Sref).sub(4, 4, 4, 4), 2);
        engine::kernels::gemm<Elt>(ctx, DenseView<const Elt>(whole.sub(4, 4, 0, 4)), DenseView<const Elt>(whole.sub(0, 4, 4, 4)), whole.sub(4, 4, 4, 4), GemmOp::Sub);
        check(engine::same_entries(DenseView<const Elt>(Sq), DenseView<const Elt>(Sref)), "gemm Schur update on disjoint windows " + tagp(p));
    }
}

// Accum64 -> Canonical (u32 modulus only)
void accum64_battery(const ModCtx<uint32_t>& ctx, Splitmix& rng) {
    DenseBuf<uint64_t> acc(37, 21, Layout::RowMajor, 25, Invariant::Accum64);
    DenseBuf<uint32_t> dst(37, 21, Layout::ColMajor), want(37, 21);
    for (uint32_t i = 0; i < 37; ++i) for (uint32_t j = 0; j < 21; ++j) {
        const uint64_t v = rng.next();
        acc.at(i, j) = v; want.at(i, j) = static_cast<uint32_t>(v % ctx.p);
    }
    engine::kernels::canonicalise(ctx, DenseView<const uint64_t>(acc), DenseView<uint32_t>(dst));
    check(engine::same_entries(DenseView<const uint32_t>(dst), DenseView<const uint32_t>(want)), "canonicalise Accum64 p=" + str(ctx.p));
}

// ---------------------------------------------------------------------------
// the oracle's RREF / kernel basis vs FlatMatrix (D5b)
// ---------------------------------------------------------------------------
template<class Elt>
void oracle_rref_battery(Elt p, Splitmix& rng) {
    for (uint32_t m : {1u, 2u, 7u, 16u, 17u, 40u}) for (uint32_t n : {1u, 3u, 16u, 17u, 33u}) for (uint32_t r : {0u, 1u, 3u, 100u}) {
        // rank-deficient by construction: A = L (m x r) * R (r x n) when r < min(m, n)
        DenseBuf<Elt> A(m, n);
        const uint32_t rr = std::min(r, std::min(m, n));
        if (rr == std::min(m, n)) {
            A = random_buf<Elt>(rng, p, m, n, Layout::RowMajor);
        } else {
            DenseBuf<Elt> L = random_buf<Elt>(rng, p, m, rr, Layout::RowMajor), R = random_buf<Elt>(rng, p, rr, n, Layout::RowMajor);
            engine::ref::gemm<Elt>(p, DenseView<const Elt>(L), DenseView<const Elt>(R), DenseView<Elt>(A), 0);
        }
        const FlatMatrixT<Elt> F = to_flat<Elt>(DenseView<const Elt>(A), p);
        DenseBuf<Elt> M = clone_layout(A);
        const std::vector<uint32_t> piv = engine::ref::rref<Elt>(p, DenseView<Elt>(M));
        const std::string tag = "oracle rref " + tagp(p) + " m=" + str(m) + " n=" + str(n) + " r=" + str(rr);
        check(equal_flat<Elt>(DenseView<const Elt>(M), F.get_row_echelon_form_as_flat_matrix()), tag + " != FlatMatrix RREF");
        check(piv == F.get_pivot_columns(), tag + " pivots");
        const auto ker = engine::ref::kernel_basis<Elt>(p, DenseView<const Elt>(A));
        const std::vector<FlatMatrixT<Elt>> fk = F.kernel();
        bool ok = ker.size() == fk.size();
        for (size_t v = 0; ok && v < ker.size(); ++v) {
            const FlatMatrixT<Elt>& kv = fk[v];
            for (uint32_t t = 0; t < n; ++t) {
                const Elt want = (kv.get_rows() == n) ? kv.get_entry(t, 0) : kv.get_entry(0, t);
                if (ker[v][t] != want) ok = false;
            }
        }
        check(ok, tag + " kernel basis != FlatMatrix::kernel()");
    }
}

// ---------------------------------------------------------------------------
// optional timing table
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// (slot-25 EXZ2-P2) the RNS Garner step: portable / AVX2 / AVX-512 bodies against the scalar oracle (plain % arithmetic,
// Horner digits, no Shoup), every tail 1 .. 40 + long planes, worst-case digits (q_i - 1) and residues, and a round trip
// x = d_0 + d_1 q_0 for random x < q_0 q_1 (u64) with the sign rule (P - 1)/2.
// ---------------------------------------------------------------------------
void rns_battery(Splitmix& rng) {
    namespace rns = engine::rns;
    const std::vector<std::vector<uint32_t>> sets = {
        {1073741789u, 1073741783u, 1073741741u, 1073741723u, 1073741719u, 1073741717u, 1073741689u, 1073741671u, 1073741663u},
        {2147483647u, 1073741789u, 3u, 5u, 65537u},      // q at 2^31 - 1 (the Shoup edge), tiny moduli, mixed order
        {1000003u, 2147483629u, 7u}};
    const rns::IsaPath paths[3] = {rns::IsaPath::Portable, rns::IsaPath::Avx2, rns::IsaPath::Avx512};
    uint32_t sizes[45];
    for (uint32_t i = 0; i <= 40; ++i) sizes[i] = i;
    sizes[41] = 63; sizes[42] = 257; sizes[43] = 1031; sizes[44] = 4099;
    for (const auto& primes : sets) {
        const rns::GarnerPlan plan(primes);
        const uint32_t L = static_cast<uint32_t>(primes.size());
        for (uint32_t E : sizes) {
            if (E == 0) continue;   // (EXZ2-P2 fix round 1, audit F11) no checks on empty vectors: they compared nothing
            for (int worst = 0; worst < 2; ++worst) {
                std::vector<std::vector<uint32_t>> dg(L, std::vector<uint32_t>(E)), res(L, std::vector<uint32_t>(E));
                for (uint32_t k = 0; k < L; ++k)
                    for (uint32_t e = 0; e < E; ++e) {
                        dg[k][e] = worst ? primes[k] - 1 : static_cast<uint32_t>(rng.below(primes[k]));
                        res[k][e] = worst ? primes[k] - 1 - (e % 2) : static_cast<uint32_t>(rng.below(primes[k]));
                    }
                std::vector<const uint32_t*> dp(L);
                for (uint32_t k = 0; k < L; ++k) dp[k] = dg[k].data();
                for (uint32_t k = 0; k < L; ++k) {
                    std::vector<uint32_t> ref(E);
                    rns::garner_step_oracle(plan, k, dp.data(), res[k].data(), ref.data(), E);
                    for (rns::IsaPath ip : paths) {
                        if (ip == rns::IsaPath::Avx2 && !rns::garner_avx2_available()) continue;
                        if (ip == rns::IsaPath::Avx512 && !rns::garner_avx512_available()) continue;
                        std::vector<uint32_t> out(E, 0xDEADBEEFu);
                        rns::garner_step(plan, k, dp.data(), res[k].data(), out.data(), E, ip);
                        check(out == ref, std::string("garner_step ") + engine::cpu::isa_path_name(ip) + " L=" + str(L) + " k=" + str(k) +
                                              " E=" + str(E) + (worst ? " worst" : " random") + " q=" + str(primes[k]));
                    }
                }
            }
        }
        // round trip on the first two primes (P = q_0 q_1 < 2^62): random x, its residues, digits, value and sign
        if (L >= 2) {
            const rns::GarnerPlan two({primes[0], primes[1]});
            const uint64_t P = uint64_t(primes[0]) * primes[1];
            const uint32_t E = 333;
            std::vector<uint64_t> x(E);
            std::vector<std::vector<uint32_t>> res(2, std::vector<uint32_t>(E));
            for (uint32_t e = 0; e < E; ++e) {
                x[e] = e == 0 ? 0 : e == 1 ? P - 1 : e == 2 ? (P - 1) / 2 : e == 3 ? (P + 1) / 2 : rng.below(P);
                res[0][e] = static_cast<uint32_t>(x[e] % primes[0]);
                res[1][e] = static_cast<uint32_t>(x[e] % primes[1]);
            }
            for (rns::IsaPath ip : paths) {
                if (ip == rns::IsaPath::Avx2 && !rns::garner_avx2_available()) continue;
                if (ip == rns::IsaPath::Avx512 && !rns::garner_avx512_available()) continue;
                std::vector<std::vector<uint32_t>> dg;
                rns::garner_digits(two, {res[0].data(), res[1].data()}, E, dg, ip);
                bool ok = true, sign_ok = true;
                for (uint32_t e = 0; e < E; ++e) {
                    ok = ok && uint64_t(dg[0][e]) + uint64_t(dg[1][e]) * primes[0] == x[e];
                    sign_ok = sign_ok && rns::garner_negative(two, dg, e) == (x[e] > (P - 1) / 2);
                }
                check(ok, std::string("garner_digits round trip ") + engine::cpu::isa_path_name(ip) + " q0=" + str(primes[0]));
                check(sign_ok, std::string("garner_negative ") + engine::cpu::isa_path_name(ip) + " q0=" + str(primes[0]));
            }
        }
    }
    std::cout << "  RNS garner_step (slot-25 EXZ2-P2): portable" << (rns::garner_avx2_available() ? " + avx2" : "")
              << (rns::garner_avx512_available() ? " + avx512" : "") << " vs the scalar oracle; resolved path "
              << engine::cpu::isa_path_name(rns::garner_resolved_path(rns::IsaPath::Auto)) << std::endl;
}

void timing_table() {
    const char* env = std::getenv("ENGINE_KERNEL_TIMING");
    if (env == nullptr || env[0] == '\0' || env[0] == '0') {
        std::cout << "  (set ENGINE_KERNEL_TIMING=1 for the GEMM throughput table)" << std::endl;
        return;
    }
    const ModCtx<uint32_t> ctx(134217757u);
    Splitmix rng(0x7157ULL);
    std::vector<KernelPath> paths = {KernelPath::Portable};
    if (engine::kernels::avx2_kernel_available()) paths.push_back(KernelPath::Avx2);   // (slot-24 AVX2K)
    if (engine::kernels::avx512_kernel_available()) paths.push_back(KernelPath::Avx512);
    std::cout << "  GEMM single-thread throughput, p = 134217757, C = A*B (row-major x row-major)" << std::endl;
    for (KernelPath path : paths) {
        engine::kernels::set_kernel_path(path);
        std::cout << "   path " << engine::kernels::kernel_path_name(path) << ":";
        for (uint32_t n : {64u, 128u, 256u, 512u}) {
            DenseBuf<uint32_t> A = random_buf<uint32_t>(rng, ctx.p, n, n, Layout::RowMajor);
            DenseBuf<uint32_t> B = random_buf<uint32_t>(rng, ctx.p, n, n, Layout::RowMajor);
            DenseBuf<uint32_t> C(n, n);
            engine::kernels::gemm<uint32_t>(ctx, DenseView<const uint32_t>(A), DenseView<const uint32_t>(B), DenseView<uint32_t>(C));
            uint64_t reps = 0, sink = 0;
            const auto t0 = std::chrono::steady_clock::now();
            double secs = 0;
            do {
                engine::kernels::gemm<uint32_t>(ctx, DenseView<const uint32_t>(A), DenseView<const uint32_t>(B), DenseView<uint32_t>(C));
                sink += C.at(reps % n, (reps * 7) % n);
                ++reps;
                secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            } while (secs < 0.5);
            const double gmm = static_cast<double>(n) * n * n * static_cast<double>(reps) / secs / 1e9;
            std::cout << "  n=" << n << ": " << gmm << " G";
            if (sink == 0xFFFFFFFFFFFFFFFFull) std::cout << "!";   // consume
        }
        std::cout << std::endl;
    }
    engine::kernels::set_kernel_path(KernelPath::Auto);
}

} // namespace

uint32_t KernelBitIdentityValidation() {
    g_fail = 0; g_checks = 0;
    std::cout << "KernelBitIdentityValidation: kernels vs RefLinAlg oracle vs FlatMatrix" << std::endl;
    std::cout << "  AVX-512 microkernel " << (engine::kernels::avx512_kernel_available() ? "available: both paths are run and compared"
                                                                                          : "NOT available: portable path only") << std::endl;
    std::cout << "  AVX2 microkernel (slot-24 AVX2K) " << (engine::kernels::avx2_kernel_available() ? "available: run and compared too"
                                                                                                    : "NOT available") << std::endl;
    Splitmix rng(0xB17B17ULL);

    // uint32_t
    {
        const ModCtx<uint32_t> c1(134217757u);     // project prime, k_max 1023 (k-blocks of 256)
        gemm_battery<uint32_t>(c1, rng, 517, 129);
        gemv_battery<uint32_t>(c1, rng, 517);
        structured_battery<uint32_t>(c1, rng, 129);
        elementwise_battery<uint32_t>(c1, rng, 129);
        accum64_battery(c1, rng);
        oracle_rref_battery<uint32_t>(134217757u, rng);

        const ModCtx<uint32_t> c2(3u);
        gemm_battery<uint32_t>(c2, rng, 65, 65);
        structured_battery<uint32_t>(c2, rng, 17);
        oracle_rref_battery<uint32_t>(3u, rng);

        const ModCtx<uint32_t> c3(2147483647u);   // k_max 4: k-blocking at the bound
        gemm_battery<uint32_t>(c3, rng, 129, 65);
        gemv_battery<uint32_t>(c3, rng, 65);
        elementwise_battery<uint32_t>(c3, rng, 17);

        const ModCtx<uint32_t> c4(4294967291u);   // k_max 1: every product reduced; no Shoup
        gemm_battery<uint32_t>(c4, rng, 129, 65);
        structured_battery<uint32_t>(c4, rng, 17);
        elementwise_battery<uint32_t>(c4, rng, 17);
        accum64_battery(c4, rng);
    }
    // uint64_t
    {
        const ModCtx<uint64_t> d1(134217757ull);
        gemm_battery<uint64_t>(d1, rng, 256, 65);
        gemv_battery<uint64_t>(d1, rng, 129);
        structured_battery<uint64_t>(d1, rng, 65);
        elementwise_battery<uint64_t>(d1, rng, 65);
        oracle_rref_battery<uint64_t>(134217757ull, rng);

        const ModCtx<uint64_t> d2(9223372036854775783ull);   // 2^63-25, k_max 4
        gemm_battery<uint64_t>(d2, rng, 65, 65);
        gemv_battery<uint64_t>(d2, rng, 65);
        structured_battery<uint64_t>(d2, rng, 17);
        elementwise_battery<uint64_t>(d2, rng, 17);

        const ModCtx<uint64_t> d3(4294967311ull);            // 2^32+15
        gemm_battery<uint64_t>(d3, rng, 65, 17);
    }

    rns_battery(rng);

    timing_table();

    std::cout << "KernelBitIdentityValidation: " << (g_fail == 0 ? "PASS" : "FAIL") << " (" << g_checks
              << " checks, " << g_fail << " failing)" << std::endl;
    return g_fail;
}
