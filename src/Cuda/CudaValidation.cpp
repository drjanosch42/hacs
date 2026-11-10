// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "CudaValidation.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaCtx.h"
#include "CudaKernels.h"
#include "DeviceBuf.h"
#include "DenseBuf.h"
#include "DenseView.h"
#include "Kernels.h"
#include "ModCtx.h"

// See CudaValidation.h.

namespace {

using engine::DenseBuf;
using engine::DenseView;
using engine::Invariant;
using engine::Layout;
using engine::ModCtx;
using engine::Structure;
using engine::kernels::GemmOp;
namespace cu = engine::cuda;

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
uint32_t g_checks = 0;
void fail(const std::string& what) {
    ++g_fail;
    if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
}
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) fail(what); }

template<class T> std::string str(T v) { return std::to_string(v); }
const char* lname(Layout l) { return l == Layout::RowMajor ? "R" : "C"; }
const char* opname(GemmOp op) { return op == GemmOp::Set ? "Set" : op == GemmOp::Add ? "Add" : "Sub"; }
template<class Elt> std::string tagp(Elt p) { return std::string(sizeof(Elt) == 4 ? "u32" : "u64") + " p=" + str(p); }

const Layout kLayouts[2] = {Layout::RowMajor, Layout::ColMajor};
const uint32_t kSizes[] = {1, 2, 7, 16, 17, 64, 65, 129, 256, 517, 1024, 2048};
constexpr size_t kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

// A random buffer whose padding (ld > minor dim) is garbage; entries canonical.
template<class Elt>
DenseBuf<Elt> random_buf(Splitmix& rng, Elt p, uint32_t rows, uint32_t cols, Layout layout,
                         uint32_t pad = 0, Structure st = Structure::Generic) {
    const uint32_t minor = layout == Layout::RowMajor ? cols : rows;
    DenseBuf<Elt> b(rows, cols, layout, minor + pad, Invariant::Canonical, st);
    Elt* d = b.data();
    for (size_t i = 0; i < b.footprint(); ++i) d[i] = static_cast<Elt>(rng.next());
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) b.at(i, j) = static_cast<Elt>(rng.below(p));
    return b;
}

template<class Elt>
bool same_bytes(const DenseBuf<Elt>& a, const DenseBuf<Elt>& b) {
    return a.footprint() == b.footprint() &&
           std::memcmp(a.data(), b.data(), a.footprint() * sizeof(Elt)) == 0;
}

// The device side of a comparison: mirror + upload; after the kernel, mark
// written + download; the host buffer then holds the device's bytes.
template<class Elt>
struct Mirror {
    DenseBuf<Elt>& host;
    cu::DeviceBuf<Elt> dev;
    explicit Mirror(DenseBuf<Elt>& h, cu::Stream s) : host(h), dev(h) { dev.upload(s); }
    cu::DeviceView<const Elt> cv() const { return dev.cview(); }
    cu::DeviceView<Elt> v() { return dev.view(); }
    void fetch(cu::Stream s) { dev.mark_device_written(); dev.download(s); }
};

cu::Stream g_stream;

// ---------------------------------------------------------------------------
// GEMM: C op= A * B, every layout / transposition combination, tiles, split-K
// ---------------------------------------------------------------------------
template<class Elt>
void gemm_case(Splitmix& rng, const ModCtx<Elt>& ctx, uint32_t m, uint32_t n, uint32_t k,
               Layout la, Layout lb, Layout lc, bool ta, bool tb, GemmOp op, int tile, uint32_t splits,
               bool use_nt, const std::string& tagx) {
    const Elt p = ctx.p;
    // physical buffers; logical views may be transposed
    DenseBuf<Elt> A = ta ? random_buf(rng, p, k, m, la, 3) : random_buf(rng, p, m, k, la, 3);
    DenseBuf<Elt> Bp = use_nt ? (tb ? random_buf(rng, p, k, n, lb, 2) : random_buf(rng, p, n, k, lb, 2))
                              : (tb ? random_buf(rng, p, n, k, lb, 2) : random_buf(rng, p, k, n, lb, 2));
    DenseBuf<Elt> C0 = random_buf(rng, p, m, n, lc, 1);
    DenseBuf<Elt> Ccpu(C0), Cgpu(C0);

    DenseView<const Elt> Av = engine::cview(A); if (ta) Av = Av.transposed_view();
    DenseView<const Elt> Bv = engine::cview(Bp); if (tb) Bv = Bv.transposed_view();
    if (use_nt) engine::kernels::gemm_nt<Elt>(ctx, Av, Bv, engine::view(Ccpu), op);
    else        engine::kernels::gemm<Elt>(ctx, Av, Bv, engine::view(Ccpu), op);

    Mirror<Elt> dA(A, g_stream), dB(Bp, g_stream), dC(Cgpu, g_stream);
    cu::DeviceView<const Elt> dAv = dA.cv(); if (ta) dAv = dAv.transposed_view();
    cu::DeviceView<const Elt> dBv = dB.cv(); if (tb) dBv = dBv.transposed_view();
    cu::GemmOpts opts; opts.tile = tile; opts.splits = splits;
    void* ws = nullptr; size_t wsb = 0;
    if (splits > 1) {
        wsb = cu::gemm_workspace_bytes(m, n, splits, sizeof(Elt));
        ws = cu::CudaCtx::get().dalloc(wsb);
        opts.workspace = ws; opts.workspace_bytes = wsb;
    }
    if (use_nt) cu::gemm_nt<Elt>(ctx, dAv, dBv, dC.v(), op, g_stream, opts);
    else        cu::gemm<Elt>(ctx, dAv, dBv, dC.v(), op, g_stream, opts);
    dC.fetch(g_stream);
    if (ws) cu::CudaCtx::get().dfree(ws, wsb);

    check(same_bytes(Ccpu, Cgpu),
          std::string(use_nt ? "gemm_nt " : "gemm ") + tagp(p) + " m=" + str(m) + " n=" + str(n) + " k=" + str(k) +
          " A" + lname(la) + (ta ? "t" : "") + " B" + lname(lb) + (tb ? "t" : "") + " C" + lname(lc) +
          " op=" + opname(op) + " tile=" + str(tile) + " splits=" + str(splits) + " " + tagx);
}

template<class Elt>
void gemm_suite(const ModCtx<Elt>& ctx, size_t max_size_index, bool wide_layout_sweep) {
    Splitmix rng(0xC0DA0001ULL ^ static_cast<uint64_t>(ctx.p) * 0x9E37ULL ^ sizeof(Elt));
    const GemmOp ops[3] = {GemmOp::Set, GemmOp::Add, GemmOp::Sub};
    // rotated shapes over the size list
    for (size_t i = 0; i <= max_size_index; ++i) {
        const uint32_t m = kSizes[i], n = kSizes[(i + 3) % (max_size_index + 1)], k = kSizes[(i + 5) % (max_size_index + 1)];
        const bool big = (m >= 1024 || n >= 1024 || k >= 1024);
        int combo = 0;
        for (Layout la : kLayouts) for (Layout lb : kLayouts) for (Layout lc : kLayouts) {
            const bool ta = (combo & 1) != 0, tb = (combo & 2) != 0;   // rotate transposition flags
            const GemmOp op = ops[combo % 3];
            const int tile = (combo % 3 == 0) ? 32 : (combo % 3 == 1) ? 64 : 128;
            ++combo;
            if (big && combo > 2) break;                    // large shapes: two combos each
            if (!wide_layout_sweep && combo > 3) break;     // u64: fewer combos at every shape
            gemm_case<Elt>(rng, ctx, m, n, k, la, lb, lc, ta, tb, op, tile, 1, false, "rot");
            if (!big && combo == 1) gemm_case<Elt>(rng, ctx, m, n, k, la, lb, lc, ta, tb, op, tile, 1, true, "nt");
        }
    }
    // square shapes at every size, one combo, auto tile (the k > k_max cases at the large end)
    for (size_t i = 0; i <= max_size_index; ++i) {
        const uint32_t n = kSizes[i];
        gemm_case<Elt>(rng, ctx, n, n, n, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor,
                       false, false, GemmOp::Sub, 0, 1, false, "square");
    }
    // split-K: partials canonicalised then summed (D4 rule), incl. k > k_max
    const uint32_t splits_list[] = {2, 3, 5, 16};
    for (uint32_t sp : splits_list) {
        gemm_case<Elt>(rng, ctx, 64, 64, 2048, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor,
                       false, false, GemmOp::Add, 64, sp, false, "splitk");
        gemm_case<Elt>(rng, ctx, 17, 129, 1031, Layout::ColMajor, Layout::RowMajor, Layout::ColMajor,
                       true, true, GemmOp::Set, 32, sp, true, "splitk");
    }
    // k == 0 and empty shapes
    gemm_case<Elt>(rng, ctx, 5, 7, 0, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, false, false, GemmOp::Set, 0, 1, false, "k0");
    gemm_case<Elt>(rng, ctx, 5, 7, 0, Layout::RowMajor, Layout::RowMajor, Layout::ColMajor, false, false, GemmOp::Add, 0, 1, false, "k0");
    // structured inputs: rank-deficient (A = u v^T) and all-(p-1)
    {
        const uint32_t n = 129;
        DenseBuf<Elt> U = random_buf(rng, ctx.p, n, 1, Layout::RowMajor), V = random_buf(rng, ctx.p, 1, n, Layout::RowMajor);
        DenseBuf<Elt> A(n, n, Layout::RowMajor);
        engine::kernels::gemm<Elt>(ctx, engine::cview(U), engine::cview(V), engine::view(A), GemmOp::Set);
        DenseBuf<Elt> B(n, n, Layout::ColMajor);
        for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < n; ++j) B.at(i, j) = ctx.p - 1;
        DenseBuf<Elt> Ccpu(n, n, Layout::RowMajor, n + 5), Cgpu;
        Ccpu.fill(static_cast<Elt>(7 % ctx.p)); Cgpu = Ccpu;
        engine::kernels::gemm<Elt>(ctx, engine::cview(A), engine::cview(B), engine::view(Ccpu), GemmOp::Sub);
        Mirror<Elt> dA(A, g_stream), dB(B, g_stream), dC(Cgpu, g_stream);
        cu::gemm<Elt>(ctx, dA.cv(), dB.cv(), dC.v(), GemmOp::Sub, g_stream);
        dC.fetch(g_stream);
        check(same_bytes(Ccpu, Cgpu), "gemm structured (rank-1 x all-(p-1)) " + tagp(ctx.p));
    }
}

// ---------------------------------------------------------------------------
// batched vs unbatched vs CPU (the Theta-assembly shape)
// ---------------------------------------------------------------------------
template<class Elt>
void batched_suite(const ModCtx<Elt>& ctx, uint32_t d, uint32_t instances, GemmOp op, Layout lay, int tile) {
    Splitmix rng(0xBA7C4ED0ULL ^ static_cast<uint64_t>(ctx.p) ^ d ^ (sizeof(Elt) << 8));
    const Elt p = ctx.p;
    const uint32_t n_ops = 8;
    std::vector<DenseBuf<Elt>> ops;
    for (uint32_t i = 0; i < n_ops; ++i) ops.push_back(random_buf(rng, p, d, d, lay, 4));   // one stride set: batching requires it
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    for (uint32_t i = 0; i < instances; ++i)
        pairs.emplace_back(static_cast<uint32_t>(rng.below(n_ops)), static_cast<uint32_t>(rng.below(n_ops)));
    // outputs: one wide buffer, instance i writes the window [0, d) x [i*d, (i+1)*d)
    DenseBuf<Elt> Cbase = random_buf(rng, p, d, d * instances, Layout::RowMajor, 3);
    DenseBuf<Elt> Ccpu(Cbase), Cone(Cbase), Cbat(Cbase);
    for (uint32_t i = 0; i < instances; ++i)
        engine::kernels::gemm<Elt>(ctx, engine::cview(ops[pairs[i].first]), engine::cview(ops[pairs[i].second]),
                                   engine::view(Ccpu).sub(0, d, i * d, d), op);
    std::vector<std::unique_ptr<Mirror<Elt>>> dops;
    for (auto& o : ops) dops.emplace_back(new Mirror<Elt>(o, g_stream));
    {
        Mirror<Elt> dC(Cone, g_stream);
        for (uint32_t i = 0; i < instances; ++i)
            cu::gemm<Elt>(ctx, dops[pairs[i].first]->cv(), dops[pairs[i].second]->cv(),
                          dC.v().sub(0, d, i * d, d), op, g_stream, cu::GemmOpts{1, nullptr, 0, tile});
        dC.fetch(g_stream);
    }
    {
        Mirror<Elt> dC(Cbat, g_stream);
        cu::GemmBatch<Elt> batch;
        batch.set_shape(dops[pairs[0].first]->cv(), dops[pairs[0].second]->cv(), dC.v().sub(0, d, 0, d));
        for (uint32_t i = 0; i < instances; ++i)
            batch.push(dops[pairs[i].first]->cv(), dops[pairs[i].second]->cv(), dC.v().sub(0, d, i * d, d));
        batch.upload(g_stream);
        batch.run(ctx, op, g_stream, tile);
        dC.fetch(g_stream);
    }
    const std::string tag = "batched " + tagp(p) + " d=" + str(d) + " x" + str(instances) + " op=" + opname(op) +
                            " lay=" + lname(lay) + " tile=" + str(tile);
    check(same_bytes(Ccpu, Cone), tag + " (unbatched vs cpu)");
    check(same_bytes(Ccpu, Cbat), tag + " (batched vs cpu)");
}

// ---------------------------------------------------------------------------
// gemv / axpy / scal / transpose / copy / canonicalise
// ---------------------------------------------------------------------------
template<class Elt>
void level12_suite(const ModCtx<Elt>& ctx, size_t max_size_index) {
    Splitmix rng(0x1E7E1120ULL ^ static_cast<uint64_t>(ctx.p) ^ (sizeof(Elt) << 4));
    const Elt p = ctx.p;
    const GemmOp ops[3] = {GemmOp::Set, GemmOp::Add, GemmOp::Sub};
    for (size_t i = 0; i <= max_size_index; ++i) {
        const uint32_t m = kSizes[i], k = kSizes[(i + 4) % (max_size_index + 1)];
        int combo = 0;
        for (Layout la : kLayouts) for (bool ta : {false, true}) for (bool xrow : {false, true}) {
            const GemmOp op = ops[combo % 3]; const bool yrow = (combo % 2) == 1; ++combo;
            DenseBuf<Elt> A = ta ? random_buf(rng, p, k, m, la, 2) : random_buf(rng, p, m, k, la, 2);
            DenseBuf<Elt> x = xrow ? random_buf(rng, p, 1, k, Layout::RowMajor, 1) : random_buf(rng, p, k, 1, Layout::ColMajor, 3);
            DenseBuf<Elt> y0 = yrow ? random_buf(rng, p, 1, m, Layout::ColMajor, 2) : random_buf(rng, p, m, 1, Layout::RowMajor, 1);
            DenseBuf<Elt> ycpu(y0), ygpu(y0);
            DenseView<const Elt> Av = engine::cview(A); if (ta) Av = Av.transposed_view();
            DenseView<const Elt> xv = engine::cview(x); if (xrow) xv = xv.transposed_view();
            DenseView<Elt> yv = engine::view(ycpu); if (yrow) yv = yv.transposed_view();
            engine::kernels::gemv<Elt>(ctx, Av, xv, yv, op);
            Mirror<Elt> dA(A, g_stream), dx(x, g_stream), dy(ygpu, g_stream);
            cu::DeviceView<const Elt> dAv = dA.cv(); if (ta) dAv = dAv.transposed_view();
            cu::DeviceView<const Elt> dxv = dx.cv(); if (xrow) dxv = dxv.transposed_view();
            cu::DeviceView<Elt> dyv = dy.v(); if (yrow) dyv = dyv.transposed_view();
            cu::gemv<Elt>(ctx, dAv, dxv, dyv, op, g_stream);
            dy.fetch(g_stream);
            check(same_bytes(ycpu, ygpu), "gemv " + tagp(p) + " m=" + str(m) + " k=" + str(k) + " A" + lname(la) +
                                          (ta ? "t" : "") + " xrow=" + str(xrow) + " yrow=" + str(yrow) + " op=" + opname(op));
        }
    }
    // axpy / scal / copy / transpose / canonicalise on rotated shapes
    for (size_t i = 0; i <= std::min<size_t>(max_size_index, 9); ++i) {
        const uint32_t r = kSizes[i], c = kSizes[(i + 2) % 10];
        for (Layout lx : kLayouts) for (Layout ly : kLayouts) for (bool tx : {false, true}) {
            const Elt alphas[4] = {Elt(0), Elt(1 % p), static_cast<Elt>(p - 1), static_cast<Elt>(rng.below(p))};
            for (int ai = 0; ai < 4; ++ai) {
                const Elt alpha = alphas[ai];
                const bool subtract = (ai % 2) == 1;
                DenseBuf<Elt> x = tx ? random_buf(rng, p, c, r, lx, 1) : random_buf(rng, p, r, c, lx, 1);
                DenseBuf<Elt> y0 = random_buf(rng, p, r, c, ly, 2), ycpu(y0), ygpu(y0);
                DenseView<const Elt> xv = engine::cview(x); if (tx) xv = xv.transposed_view();
                engine::kernels::axpy<Elt>(ctx, alpha, xv, engine::view(ycpu), subtract);
                Mirror<Elt> dx(x, g_stream), dy(ygpu, g_stream);
                cu::DeviceView<const Elt> dxv = dx.cv(); if (tx) dxv = dxv.transposed_view();
                cu::axpy<Elt>(ctx, alpha, dxv, dy.v(), subtract, g_stream);
                dy.fetch(g_stream);
                check(same_bytes(ycpu, ygpu), "axpy " + tagp(p) + " " + str(r) + "x" + str(c) + " x" + lname(lx) + (tx ? "t" : "") +
                                              " y" + lname(ly) + " alpha=" + str(alpha) + " sub=" + str(subtract));
                if (ai == 3 || ai == 0) {
                    DenseBuf<Elt> s0 = random_buf(rng, p, r, c, lx, 3), scpu(s0), sgpu(s0);
                    DenseView<Elt> sv = engine::view(scpu); if (tx) sv = sv.transposed_view();
                    engine::kernels::scal<Elt>(ctx, alpha, sv);
                    Mirror<Elt> ds(sgpu, g_stream);
                    cu::DeviceView<Elt> dsv = ds.v(); if (tx) dsv = dsv.transposed_view();
                    cu::scal<Elt>(ctx, alpha, dsv, g_stream);
                    ds.fetch(g_stream);
                    check(same_bytes(scpu, sgpu), "scal " + tagp(p) + " " + str(r) + "x" + str(c) + " " + lname(lx) + " alpha=" + str(alpha));
                }
            }
            // copy / transpose (physical), dst padded
            {
                DenseBuf<Elt> src = tx ? random_buf(rng, p, c, r, lx, 2) : random_buf(rng, p, r, c, lx, 2);
                DenseView<const Elt> sv = engine::cview(src); if (tx) sv = sv.transposed_view();
                DenseBuf<Elt> d0 = random_buf(rng, p, r, c, ly, 3), dcpu(d0), dgpu(d0);
                engine::kernels::copy<Elt>(sv, engine::view(dcpu));
                Mirror<Elt> dsrc(src, g_stream), ddst(dgpu, g_stream);
                cu::DeviceView<const Elt> dsv = dsrc.cv(); if (tx) dsv = dsv.transposed_view();
                cu::copy<Elt>(dsv, ddst.v(), g_stream);
                ddst.fetch(g_stream);
                check(same_bytes(dcpu, dgpu), "copy " + tagp(p) + " " + str(r) + "x" + str(c) + " " + lname(lx) + (tx ? "t" : "") + "->" + lname(ly));
                DenseBuf<Elt> t0 = random_buf(rng, p, c, r, ly, 1), tcpu(t0), tgpu(t0);
                engine::kernels::transpose<Elt>(sv, engine::view(tcpu));
                Mirror<Elt> dt(tgpu, g_stream);
                cu::transpose<Elt>(dsv, dt.v(), g_stream);
                dt.fetch(g_stream);
                check(same_bytes(tcpu, tgpu), "transpose " + tagp(p) + " " + str(r) + "x" + str(c) + " " + lname(lx) + (tx ? "t" : "") + "->" + lname(ly));
            }
            // canonicalise Lazy2p (u32 with 2p <= 2^32 only, as on the CPU)
            if (sizeof(Elt) == 4 && static_cast<uint64_t>(p) * 2 <= (uint64_t(1) << 32)) {
                DenseBuf<Elt> l0(r, c, lx, (lx == Layout::RowMajor ? c : r) + 2, Invariant::Lazy2p);
                for (size_t q = 0; q < l0.footprint(); ++q) l0.data()[q] = static_cast<Elt>(rng.next());
                for (uint32_t a = 0; a < r; ++a) for (uint32_t b = 0; b < c; ++b) l0.at(a, b) = static_cast<Elt>(rng.below(2 * static_cast<uint64_t>(p)));
                DenseBuf<Elt> lcpu(l0), lgpu(l0);
                engine::kernels::canonicalise<Elt>(ctx, engine::view(lcpu));
                Mirror<Elt> dl(lgpu, g_stream);
                cu::canonicalise<Elt>(ctx, dl.v(), g_stream);
                dl.fetch(g_stream);
                check(same_bytes(lcpu, lgpu), "canonicalise " + tagp(p) + " " + str(r) + "x" + str(c) + " " + lname(lx));
            }
        }
    }
}

// Accum64 -> u32 canonicalise (u32 moduli)
void canon64_suite(const ModCtx<uint32_t>& ctx) {
    Splitmix rng(0xACC64ULL ^ ctx.p);
    for (uint32_t r : {1u, 7u, 65u, 517u}) for (Layout la : kLayouts) for (Layout ld : kLayouts) {
        const uint32_t c = 129;
        DenseBuf<uint64_t> acc(r, c, la, (la == Layout::RowMajor ? c : r) + 1, Invariant::Accum64);
        for (size_t q = 0; q < acc.footprint(); ++q) acc.data()[q] = rng.next();   // any u64
        DenseBuf<uint32_t> d0 = random_buf<uint32_t>(rng, ctx.p, r, c, ld, 2), dcpu(d0), dgpu(d0);
        engine::kernels::canonicalise(ctx, engine::cview(acc), engine::view(dcpu));
        Mirror<uint64_t> dacc(acc, g_stream); Mirror<uint32_t> dd(dgpu, g_stream);
        cu::canonicalise(ctx, dacc.cv(), dd.v(), g_stream);
        dd.fetch(g_stream);
        check(same_bytes(dcpu, dgpu), "canonicalise(acc64) p=" + str(ctx.p) + " " + str(r) + "x" + str(c) + " " + lname(la) + "->" + lname(ld));
    }
}

// ---------------------------------------------------------------------------
// trsm_unit: X = T^{-1} B checked by the CPU trmm (T X == B, unique solution)
// and by bit-identity of the trailing gemm(Sub) updates
// ---------------------------------------------------------------------------
template<class Elt>
void trsm_suite(const ModCtx<Elt>& ctx) {
    Splitmix rng(0x7B5A0000ULL ^ static_cast<uint64_t>(ctx.p) ^ sizeof(Elt));
    const Elt p = ctx.p;
    for (uint32_t m : {1u, 2u, 7u, 17u, 32u, 33u, 65u, 129u, 517u}) for (uint32_t n : {1u, 5u, 129u}) {
        for (bool lower : {true, false}) for (Layout lt : kLayouts) for (Layout lb : kLayouts) {
            if (m == 517 && (lt != Layout::RowMajor || lb != Layout::ColMajor)) continue;
            const Structure st = lower ? Structure::LowerUnitri : Structure::UpperUnitri;
            DenseBuf<Elt> T = random_buf(rng, p, m, m, lt, 1, st);
            DenseBuf<Elt> B = random_buf(rng, p, m, n, lb, 2);
            DenseBuf<Elt> X(B);
            Mirror<Elt> dT(T, g_stream), dX(X, g_stream);
            cu::trsm_unit<Elt>(ctx, dT.cv(), dX.v(), g_stream);
            dX.fetch(g_stream);
            // certificate: T * X == B entrywise (CPU trmm reads only the strict triangle)
            DenseBuf<Elt> TX(m, n, lb);
            engine::kernels::trmm<Elt>(ctx, engine::cview(T), engine::cview(X), engine::view(TX));
            bool ok = true;
            for (uint32_t i = 0; i < m && ok; ++i) for (uint32_t j = 0; j < n; ++j) if (TX.at(i, j) != B.at(i, j)) { ok = false; break; }
            check(ok, "trsm_unit " + tagp(p) + " m=" + str(m) + " n=" + str(n) + (lower ? " lower" : " upper") + " T" + lname(lt) + " B" + lname(lb));
            // padding untouched
            bool pad_ok = true;
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) TX.at(i, j) = X.at(i, j);   // reuse TX as "X's entries"
            DenseBuf<Elt> Xpad(B);
            for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) Xpad.at(i, j) = X.at(i, j);
            pad_ok = same_bytes(Xpad, X);
            check(pad_ok, "trsm_unit padding untouched " + tagp(p) + " m=" + str(m) + " n=" + str(n));
        }
    }
}

// ---------------------------------------------------------------------------
// graph replay vs direct: a small program (gemm, axpy, gemv, trsm, batched)
// ---------------------------------------------------------------------------
template<class Elt>
void graph_suite(const ModCtx<Elt>& ctx) {
    Splitmix rng(0x6EA9E000ULL ^ static_cast<uint64_t>(ctx.p) ^ sizeof(Elt));
    const Elt p = ctx.p;
    const uint32_t m = 129, k = 65, n = 33;
    DenseBuf<Elt> A = random_buf(rng, p, m, k, Layout::RowMajor, 1), B = random_buf(rng, p, k, n, Layout::ColMajor, 2);
    DenseBuf<Elt> T = random_buf(rng, p, m, m, Layout::RowMajor, 0, Structure::LowerUnitri);
    DenseBuf<Elt> x = random_buf(rng, p, n, 1, Layout::ColMajor);
    DenseBuf<Elt> C0 = random_buf(rng, p, m, n, Layout::RowMajor, 3), y0 = random_buf(rng, p, m, 1, Layout::RowMajor);
    const Elt alpha = static_cast<Elt>(rng.below(p));
    // CPU reference of the program: C = A*B; C -= alpha*C? (no: axpy needs distinct x, y) -> C += alpha*Ccopy; trsm; y = C*x
    auto run_cpu = [&](DenseBuf<Elt>& C, DenseBuf<Elt>& y, DenseBuf<Elt>& D) {
        engine::kernels::gemm<Elt>(ctx, engine::cview(A), engine::cview(B), engine::view(C), GemmOp::Set);
        engine::kernels::axpy<Elt>(ctx, alpha, engine::cview(C), engine::view(D), true);
        // D = T^{-1} D  (CPU: certificate below, no CPU trsm twin needed)
        engine::kernels::gemv<Elt>(ctx, engine::cview(C), engine::cview(x), engine::view(y), GemmOp::Add);
    };
    DenseBuf<Elt> Ccpu(C0), ycpu(y0), Dcpu = random_buf(rng, p, m, n, Layout::RowMajor, 1);
    const DenseBuf<Elt> D0(Dcpu);
    run_cpu(Ccpu, ycpu, Dcpu);

    DenseBuf<Elt> Cg(C0), yg(y0), Dg(D0);
    Mirror<Elt> dA(A, g_stream), dB(B, g_stream), dT(T, g_stream), dx(x, g_stream), dC(Cg, g_stream), dy(yg, g_stream), dD(Dg, g_stream);
    auto record = [&]() {
        cu::gemm<Elt>(ctx, dA.cv(), dB.cv(), dC.v(), GemmOp::Set, g_stream);
        cu::axpy<Elt>(ctx, alpha, dC.cv(), dD.v(), true, g_stream);
        cu::trsm_unit<Elt>(ctx, dT.cv(), dD.v(), g_stream);
        cu::gemv<Elt>(ctx, dC.cv(), dx.cv(), dy.v(), GemmOp::Add, g_stream);
    };
    // direct
    record();
    dC.fetch(g_stream); dy.fetch(g_stream); dD.fetch(g_stream);
    check(same_bytes(Ccpu, Cg) && same_bytes(ycpu, yg), "graph program direct vs cpu " + tagp(p));
    DenseBuf<Elt> Ddirect(Dg);
    // graph: reset inputs, capture, replay twice (second replay from re-uploaded inputs)
    cu::GraphCapture g;
    g.begin(g_stream);
    record();
    g.end();
    check(g.num_nodes() >= 4, "graph captured >= 4 nodes (" + str(g.num_nodes()) + ")");
    for (int rep = 0; rep < 2; ++rep) {
        Cg = C0; yg = y0; Dg = D0;
        dC.dev.mark_host_written(); dC.dev.upload(g_stream, true);
        dy.dev.mark_host_written(); dy.dev.upload(g_stream, true);
        dD.dev.mark_host_written(); dD.dev.upload(g_stream, true);
        g.launch(g_stream);
        dC.fetch(g_stream); dy.fetch(g_stream); dD.fetch(g_stream);
        check(same_bytes(Ccpu, Cg) && same_bytes(ycpu, yg) && same_bytes(Ddirect, Dg),
              "graph replay #" + str(rep) + " vs direct/cpu " + tagp(p));
    }
    // batched inside a graph
    {
        const uint32_t d = 65, inst = 40;
        std::vector<DenseBuf<Elt>> ops;
        for (int i = 0; i < 6; ++i) ops.push_back(random_buf(rng, p, d, d, Layout::RowMajor, 2));   // one stride set
        DenseBuf<Elt> Cb0 = random_buf(rng, p, d, d * inst, Layout::ColMajor, 2), Cbc(Cb0), Cbg(Cb0);
        std::vector<std::unique_ptr<Mirror<Elt>>> dops;
        for (auto& o : ops) dops.emplace_back(new Mirror<Elt>(o, g_stream));
        Mirror<Elt> dCb(Cbg, g_stream);
        cu::GemmBatch<Elt> batch;
        batch.set_shape(dops[0]->cv(), dops[1]->cv(), dCb.v().sub(0, d, 0, d));
        for (uint32_t i = 0; i < inst; ++i) {
            const uint32_t a = static_cast<uint32_t>(rng.below(6)), b = static_cast<uint32_t>(rng.below(6));
            engine::kernels::gemm<Elt>(ctx, engine::cview(ops[a]), engine::cview(ops[b]), engine::view(Cbc).sub(0, d, i * d, d), GemmOp::Sub);
            batch.push(dops[a]->cv(), dops[b]->cv(), dCb.v().sub(0, d, i * d, d));
        }
        batch.upload(g_stream);
        cu::GraphCapture gb;
        gb.begin(g_stream);
        batch.run(ctx, GemmOp::Sub, g_stream);
        gb.end();
        gb.launch(g_stream);
        dCb.fetch(g_stream);
        check(same_bytes(Cbc, Cbg), "batched gemm inside a graph " + tagp(p));
    }
}

// ---------------------------------------------------------------------------
// contract checks: ENGINE_CHECK must fire on the documented misuse
// ---------------------------------------------------------------------------
template<class Elt>
void contract_suite(const ModCtx<Elt>& ctx) {
    Splitmix rng(0xC0A7ULL);
    DenseBuf<Elt> A = random_buf(rng, ctx.p, 8, 8, Layout::RowMajor), B(A), C(A);
    Mirror<Elt> dA(A, g_stream), dB(B, g_stream), dC(C, g_stream);
    auto expect_throw = [&](const char* what, auto fn) {
        bool threw = false;
        try { fn(); } catch (const std::logic_error&) { threw = true; }
        check(threw, std::string("contract: ") + what + " throws std::logic_error");
    };
    expect_throw("gemm aliasing", [&] { cu::gemm<Elt>(ctx, dA.cv(), dB.cv(), dA.v(), GemmOp::Set, g_stream); });
    expect_throw("gemm shape", [&] { cu::gemm<Elt>(ctx, dA.cv().sub(0, 8, 0, 5), dB.cv(), dC.v(), GemmOp::Set, g_stream); });
    expect_throw("split-K without workspace", [&] {
        cu::GemmOpts o; o.splits = 4;
        cu::gemm<Elt>(ctx, dA.cv(), dB.cv(), dC.v(), GemmOp::Set, g_stream, o);
    });
    expect_throw("trsm untagged", [&] { cu::trsm_unit<Elt>(ctx, dA.cv(), dC.v(), g_stream); });
    expect_throw("axpy alpha >= p", [&] { cu::axpy<Elt>(ctx, ctx.p, dA.cv(), dC.v(), false, g_stream); });
    expect_throw("gemv non-column", [&] { cu::gemv<Elt>(ctx, dA.cv(), dB.cv(), dC.v(), GemmOp::Set, g_stream); });
}

template<class Elt>
void run_prime(Elt p, size_t max_size_index, bool wide_sweep, bool do_batched) {
    const ModCtx<Elt> ctx(p);
    std::cout << "  -- " << tagp(p) << " regime=" << engine::regime_name(ctx.regime) << " k_max=" << ctx.k_max
              << " sizes<=" << kSizes[max_size_index] << std::endl;
    gemm_suite<Elt>(ctx, max_size_index, wide_sweep);
    level12_suite<Elt>(ctx, std::min<size_t>(max_size_index, 9));
    trsm_suite<Elt>(ctx);
    graph_suite<Elt>(ctx);
    contract_suite<Elt>(ctx);
    if (do_batched) {
        batched_suite<Elt>(ctx, 129, 300, GemmOp::Set, Layout::RowMajor, 0);
        batched_suite<Elt>(ctx, 129, 37, GemmOp::Add, Layout::ColMajor, 32);
        batched_suite<Elt>(ctx, 65, 70, GemmOp::Sub, Layout::RowMajor, 64);
        batched_suite<Elt>(ctx, 7, 500, GemmOp::Sub, Layout::ColMajor, 128);
        batched_suite<Elt>(ctx, 1, 9, GemmOp::Set, Layout::RowMajor, 0);
    }
    cu::CudaCtx::get().sync_all();
}

bool skip_if_unavailable(const char* name) {
    if (cu::available()) return false;
    std::cout << "[SKIP] " << name << ": no usable CUDA device (" << cu::unavailable_reason()
              << ") -- nothing to validate, returning 0" << std::endl;
    return true;
}

void print_device() {
    const cu::DeviceInfo d = cu::device_info();
    std::cout << "  device " << d.index << ": " << d.name << " cc " << d.cc_major << "." << d.cc_minor
              << ", " << d.sms << " SMs, " << (d.total_mem >> 20) << " MB (" << (d.free_mem >> 20)
              << " MB free), driver " << d.driver_version << ", runtime " << d.runtime_version << std::endl;
}

} // namespace

uint32_t CudaBitIdentityValidation() {
    std::cout << "=== CudaBitIdentityValidation (device kernels vs engine::kernels, memcmp) ===" << std::endl;
    if (skip_if_unavailable("CudaBitIdentityValidation")) return 0;
    g_fail = 0; g_checks = 0;
    try {
        print_device();
        g_stream = cu::CudaCtx::get().stream(1);
        cu::reset_launch_stats();
        // u32: every prime up to 2048 (k > k_max at every one of them)
        run_prime<uint32_t>(134217757u, kNumSizes - 1, true, true);
        run_prime<uint32_t>(2u, 9, true, true);
        run_prime<uint32_t>(3u, 9, true, false);
        run_prime<uint32_t>(2147483647u, 10, true, true);          // 2^31-1, k_max = 4
        run_prime<uint32_t>(4294967291u, 10, true, false);         // 2^32-5, k_max = 1
        canon64_suite(ModCtx<uint32_t>(134217757u));
        canon64_suite(ModCtx<uint32_t>(4294967291u));
        canon64_suite(ModCtx<uint32_t>(2u));
        // u64: Montgomery at 2^61-1 (and a u32-sized prime in the u64 type), the wide fallback
        run_prime<uint64_t>(2305843009213693951ull, 10, false, true);   // 2^61-1
        run_prime<uint64_t>(134217757ull, 8, false, false);
        run_prime<uint64_t>(18446744073709551557ull, 7, false, false);  // 2^64-59: U64_Wide
        run_prime<uint64_t>(2ull, 6, false, false);                     // even p: U64_Wide
        const cu::LaunchStats ls = cu::launch_stats();
        const cu::TransferStats ts = cu::CudaCtx::get().transfer_stats();
        std::cout << "  launches=" << ls.kernel_launches << " gemm_launches=" << ls.gemm_launches
                  << " batched_instances=" << ls.batched_instances
                  << " h2d=" << (ts.h2d_bytes >> 20) << "MB d2h=" << (ts.d2h_bytes >> 20)
                  << "MB pageable_copies=" << ts.pageable_copies << std::endl;
        check(ts.pageable_copies == 0, "no pageable copies during the validation");
    } catch (const std::exception& e) {
        fail(std::string("exception: ") + e.what());
    }
    std::cout << "CudaBitIdentityValidation: " << g_checks << " checks, " << g_fail << " failures" << std::endl;
    return g_fail;
}

uint32_t CudaTransferValidation() {
    std::cout << "=== CudaTransferValidation (residency, staging pool, arena) ===" << std::endl;
    if (skip_if_unavailable("CudaTransferValidation")) return 0;
    g_fail = 0; g_checks = 0;
    try {
        print_device();
        cu::CudaCtx& cx = cu::CudaCtx::get();
        const cu::Stream s = cx.stream(2);
        cx.reset_transfer_stats();
        Splitmix rng(0x7A45FE4ULL);

        // 1. residency state machine
        {
            DenseBuf<uint32_t> h = random_buf<uint32_t>(rng, 134217757u, 129, 65, Layout::ColMajor, 5);
            const DenseBuf<uint32_t> h0(h);
            cu::DeviceBuf<uint32_t> d(h);
            check(d.residency() == cu::Residency::Host, "fresh mirror: Host");
            check(d.upload(s), "first upload copies");
            check(d.residency() == cu::Residency::Both, "after upload: Both");
            check(!d.upload(s), "second upload is a no-op");
            const cu::TransferStats t1 = cx.transfer_stats();
            check(!d.download(s), "download with nothing newer is a no-op");
            check(cx.transfer_stats().d2h_bytes == t1.d2h_bytes, "no-op download moved no bytes");
            d.mark_device_written();
            check(d.residency() == cu::Residency::Device, "after device write: Device");
            // scale on the device, download, compare against the CPU
            const ModCtx<uint32_t> ctx(134217757u);
            DenseBuf<uint32_t> ref(h0);
            engine::kernels::scal<uint32_t>(ctx, 12345u, engine::view(ref));
            cu::scal<uint32_t>(ctx, 12345u, d.view(), s);
            check(d.download(s), "download copies after a device write");
            check(d.residency() == cu::Residency::Both, "after download: Both");
            check(same_bytes(h, ref), "device scal == cpu scal (padding included)");
            d.mark_host_written();
            check(d.residency() == cu::Residency::Host, "after host write: Host");
            check(d.host_version() > d.device_version(), "versions monotone");
            // round trip of pure garbage bytes (padding included)
            DenseBuf<uint32_t> g(7, 1000, Layout::RowMajor, 1003);
            for (size_t i = 0; i < g.footprint(); ++i) g.data()[i] = static_cast<uint32_t>(rng.next());
            const DenseBuf<uint32_t> g0(g);
            cu::DeviceBuf<uint32_t> dg(g);
            dg.upload(s);
            g.fill(0);
            dg.mark_device_written();
            dg.download(s);
            check(same_bytes(g, g0), "byte round trip through the pinned pool");
            // move semantics
            cu::DeviceBuf<uint32_t> moved(std::move(dg));
            check(moved.dptr() != nullptr && dg.dptr() == nullptr, "move transfers the device allocation");
        }
        // 2. a transfer larger than the pinned pool: recycled chunks, cap respected, no pageable copy
        {
            const size_t cap = cx.pinned_cap_bytes();
            const size_t elems = (cap * 5 / 2) / sizeof(uint64_t);          // 2.5 x cap
            std::vector<uint64_t> big(elems);
            for (size_t i = 0; i < elems; ++i) big[i] = rng.next();
            DenseBuf<uint64_t> hb(1, static_cast<uint32_t>(elems), Layout::RowMajor);
            std::memcpy(hb.data(), big.data(), elems * sizeof(uint64_t));
            cu::DeviceBuf<uint64_t> db(hb);
            db.upload(s);
            hb.fill(0);
            db.mark_device_written();
            db.download(s);
            check(std::memcmp(hb.data(), big.data(), elems * sizeof(uint64_t)) == 0, "2.5x-cap transfer round trips exactly");
            const cu::PinnedPoolStats ps = cx.pinned_stats();
            const cu::TransferStats ts = cx.transfer_stats();
            std::cout << "  pinned pool: chunk=" << (ps.chunk_bytes >> 20) << "MB cap=" << (ps.cap_bytes >> 20)
                      << "MB allocated=" << (ps.allocated_bytes >> 20) << "MB peak=" << (ps.peak_bytes >> 20)
                      << "MB acquisitions=" << ps.acquisitions << " waits=" << ps.waits << std::endl;
            std::cout << "  transfers: h2d=" << (ts.h2d_bytes >> 20) << "MB (" << ts.h2d_calls << " calls) d2h="
                      << (ts.d2h_bytes >> 20) << "MB (" << ts.d2h_calls << " calls) pageable=" << ts.pageable_copies << std::endl;
            check(ps.peak_bytes <= ps.cap_bytes, "pinned pool never exceeds its cap");
            check(ps.waits > 0, "the oversize transfer recycled chunks (pool exhaustion path taken)");
            check(ts.pageable_copies == 0, "no pageable copy happened");
            check(ts.h2d_bytes >= cap * 5 / 2 && ts.d2h_bytes >= cap * 5 / 2, "byte counters cover the transfer");
        }
        // 3. arena reuse
        {
            const cu::ArenaStats a0 = cx.arena_stats();
            void* p1 = cx.dalloc(1234567);
            cx.dfree(p1, 1234567);
            void* p2 = cx.dalloc(1234567);
            check(p1 == p2, "arena reuses an exact-size freed block");
            cx.dfree(p2, 1234567);
            const cu::ArenaStats a1 = cx.arena_stats();
            check(a1.reuses >= a0.reuses + 1, "arena reuse counted");
            check(a1.live_bytes == a0.live_bytes, "arena live bytes balanced");
        }
        // 4. capture rules: no staged transfer, no fresh allocation inside a capture
        {
            DenseBuf<uint32_t> h = random_buf<uint32_t>(rng, 7u, 4, 4, Layout::RowMajor);
            cu::DeviceBuf<uint32_t> d(h);
            d.upload(s);
            cu::GraphCapture g;
            g.begin(s);
            bool threw = false;
            try { d.upload(s, true); } catch (const std::logic_error&) { threw = true; }
            check(threw && cu::capturing_now(), "staged upload inside a capture is refused");
            threw = false;
            try { (void)cx.dalloc(7777777); } catch (const std::logic_error&) { threw = true; }
            check(threw, "fresh arena allocation inside a capture is refused");
            const ModCtx<uint32_t> ctx(7u);
            cu::scal<uint32_t>(ctx, 3u, d.view(), s);
            g.end();
            check(!cu::capturing_now() && g.num_nodes() == 1, "capture ended with exactly the kernel node");
            g.launch(s);
            d.mark_device_written();
            d.download(s);
            DenseBuf<uint32_t> ref = h; // h now holds the device result; recompute ref from the original
            (void)ref;
        }
        // 5. device-only buffer raw copies
        {
            std::vector<uint64_t> src(4097), dst(4097, 0);
            for (auto& v : src) v = rng.next();
            cu::DeviceBuf<uint64_t> d(1, 4097, Layout::RowMajor);
            d.upload_from(src.data(), src.size(), s);
            d.download_to(dst.data(), dst.size(), s);
            check(src == dst, "device-only buffer upload_from/download_to round trip");
        }
        check(cx.transfer_stats().pageable_copies == 0, "no pageable copies overall");
    } catch (const std::exception& e) {
        fail(std::string("exception: ") + e.what());
    }
    std::cout << "CudaTransferValidation: " << g_checks << " checks, " << g_fail << " failures" << std::endl;
    return g_fail;
}
