// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "CudaBench.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CudaCtx.h"
#include "CudaKernels.h"
#include "DeviceBuf.h"
#include "Kernels.h"
#include "ModCtx.h"
#include "PerformanceSmokeValidation.h"

namespace engine {
namespace cuda {

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}
using kernels::GemmOp;

// GPU time of `fn` on stream s (events), in seconds; fn must only enqueue on s.
template<class F>
double gpu_time(CudaCtx& cx, Stream s, F fn) {
    Event a = cx.create_timing_event(), b = cx.create_timing_event();
    cx.record(a, s);
    fn();
    cx.record(b, s);
    cx.sync(b);
    const double t = cx.elapsed_ms(a, b) * 1e-3;
    cx.destroy(a); cx.destroy(b);
    return t;
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2 == 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// ---- the micro hooks -----------------------------------------------------------
micro::MicroResult GemmGpu(uint32_t n, uint32_t k, uint32_t ell, uint32_t repeat, const std::string& params) {
    micro::MicroResult res;
    res.op = "gemm"; res.params = params;
    res.work_modmul = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(k);
    CudaCtx& cx = CudaCtx::get();
    const Stream s = cx.stream(0);
    const ModCtx<uint32_t> ctx(ell);
    DenseBuf<uint32_t> A = micro::EngineMatrix(n, k, ell, 0x01);
    DenseBuf<uint32_t> B = micro::EngineMatrix(k, n, ell, 0x02);
    DenseBuf<uint32_t> C(n, n, Layout::RowMajor);
    DeviceBuf<uint32_t> dA(A), dB(B), dC(C);
    dA.upload(s); dB.upload(s);
    GemmOpts opts; opts.splits = suggest_splits(n, n, k);
    std::unique_ptr<char, void(*)(char*)> ws(nullptr, [](char*) {});
    void* wsp = nullptr; size_t wsb = 0;
    if (opts.splits > 1) { wsb = gemm_workspace_bytes(n, n, opts.splits, 4); wsp = cx.dalloc(wsb); opts.workspace = wsp; opts.workspace_bytes = wsb; }
    // (slot-22 G1) the IMMA route when the modulus qualifies and ENGINE_GPU_IMMA is not 0
    const bool imma = imma_eligible(ctx) && imma_policy() && imma_supported();
    void* iws = nullptr; size_t iwsb = 0;
    if (imma) { iwsb = imma_workspace_bytes(n, n, k); iws = cx.dalloc(iwsb); opts.imma_workspace = iws; opts.imma_workspace_bytes = iwsb; }
    std::vector<double> ts;
    for (uint32_t i = 0; i < repeat; ++i)
        ts.push_back(gpu_time(cx, s, [&] { gemm<uint32_t>(ctx, dA.cview(), dB.cview(), dC.view(), GemmOp::Set, s, opts); }));
    dC.mark_device_written(); dC.download(s);
    if (wsp) cx.dfree(wsp, wsb);
    if (iws) cx.dfree(iws, iwsb);
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = micro::ChecksumBuf(C);
    std::ostringstream note;
    if (imma) note << "device IMMA kernel (int8 tensor cores, operand conversion included; GPU time, operands resident)";
    else      note << "device kernel (tile auto, splits=" << opts.splits << "; GPU time, operands resident)";
    res.note = note.str();
    return res;
}

micro::MicroResult GemvGpu(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) {
    micro::MicroResult res;
    res.op = "gemv"; res.params = params;
    res.work_modmul = static_cast<double>(n) * static_cast<double>(n);
    CudaCtx& cx = CudaCtx::get();
    const Stream s = cx.stream(0);
    const ModCtx<uint32_t> ctx(ell);
    DenseBuf<uint32_t> A = micro::EngineMatrix(n, n, ell, 0x03);
    DenseBuf<uint32_t> x = micro::EngineMatrix(n, 1, ell, 0x04);
    DenseBuf<uint32_t> y(n, 1, Layout::RowMajor);
    DeviceBuf<uint32_t> dA(A), dx(x), dy(y);
    dA.upload(s); dx.upload(s);
    std::vector<double> ts;
    for (uint32_t i = 0; i < repeat; ++i)
        ts.push_back(gpu_time(cx, s, [&] { gemv<uint32_t>(ctx, dA.cview(), dx.cview(), dy.view(), GemmOp::Set, s); }));
    dy.mark_device_written(); dy.download(s);
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = micro::ChecksumBuf(y);
    res.note = "device kernel (GPU time, operands resident)";
    return res;
}

micro::MicroResult TransposeGpu(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) {
    micro::MicroResult res;
    res.op = "transpose"; res.params = params;
    CudaCtx& cx = CudaCtx::get();
    const Stream s = cx.stream(0);
    DenseBuf<uint32_t> A = micro::EngineMatrix(n, n, ell, 0x05);
    DenseBuf<uint32_t> T(n, n, Layout::RowMajor);
    DeviceBuf<uint32_t> dA(A), dT(T);
    dA.upload(s);
    std::vector<double> ts;
    for (uint32_t i = 0; i < repeat; ++i)
        ts.push_back(gpu_time(cx, s, [&] { transpose<uint32_t>(dA.cview(), dT.view(), s); }));
    dT.mark_device_written(); dT.download(s);
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = micro::ChecksumBuf(T);
    std::ostringstream note;
    const double elems = static_cast<double>(n) * n;
    note << std::fixed << std::setprecision(2) << (res.min_s > 0 ? elems / res.min_s / 1e9 : 0.0) << " Gelem/s (GPU time)";
    res.note = note.str();
    return res;
}

// ---- the suite ------------------------------------------------------------------
struct Row { std::string what; double t; double rate; double roof; std::string note; };
void print_row(const Row& r) {
    std::cout << "  " << std::left << std::setw(46) << r.what << std::right
              << std::fixed << std::setprecision(6) << std::setw(11) << r.t << " s";
    if (r.rate > 0) std::cout << std::setprecision(1) << std::setw(10) << r.rate << " G/s";
    else std::cout << std::setw(14) << "";
    if (r.roof > 0) std::cout << std::setprecision(1) << std::setw(7) << (100.0 * r.rate / r.roof) << " % of " << std::setw(5) << static_cast<int>(r.roof);
    if (!r.note.empty()) std::cout << "  " << r.note;
    std::cout << std::endl;
}

template<class Elt>
DenseBuf<Elt> rnd(uint32_t rows, uint32_t cols, Elt p, uint64_t seed, Layout lay = Layout::RowMajor) {
    DenseBuf<Elt> b(rows, cols, lay);
    uint64_t st = seed;
    for (uint32_t i = 0; i < rows; ++i) for (uint32_t j = 0; j < cols; ++j) {
        uint64_t z = (st += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
        b.at(i, j) = static_cast<Elt>(z % p);
    }
    return b;
}

template<class Elt>
void dense_rows(CudaCtx& cx, Stream s, Elt p, const std::vector<uint32_t>& sizes, double roof, uint32_t repeat,
                const char* label) {
    const ModCtx<Elt> ctx(p);
    for (uint32_t n : sizes) {
        DenseBuf<Elt> A = rnd<Elt>(n, n, p, 11), B = rnd<Elt>(n, n, p, 12);
        DeviceBuf<Elt> dA(A), dB(B), dC(n, n, Layout::RowMajor);
        dA.upload(s); dB.upload(s);
        const double work = static_cast<double>(n) * n * n;
        for (int tile : {64, 128}) {
            if (sizeof(Elt) == 8 && tile == 128) continue;
            GemmOpts o; o.tile = tile; o.splits = suggest_splits(n, n, n);
            void* ws = nullptr; size_t wsb = 0;
            if (o.splits > 1) { wsb = gemm_workspace_bytes(n, n, o.splits, sizeof(Elt)); ws = cx.dalloc(wsb); o.workspace = ws; o.workspace_bytes = wsb; }
            std::vector<double> ts;
            for (uint32_t r = 0; r < repeat; ++r)
                ts.push_back(gpu_time(cx, s, [&] { gemm<Elt>(ctx, dA.cview(), dB.cview(), dC.view(), GemmOp::Set, s, o); }));
            const double t = *std::min_element(ts.begin(), ts.end());
            std::ostringstream w; w << label << " gemm " << n << "^3 tile=" << tile << " splits=" << o.splits;
            print_row(Row{w.str(), t, work / t / 1e9, roof, ""});
            if (ws) cx.dfree(ws, wsb);
        }
    }
}

int suite(uint32_t repeat) {
    CudaCtx& cx = CudaCtx::get();
    const DeviceInfo d = device_info();
    std::cout << "=== bench micro cuda (Phase 5 throughput table, repeat=" << repeat << ") ===\n"
              << "  device " << d.name << " cc " << d.cc_major << "." << d.cc_minor << ", " << d.sms
              << " SMs, driver " << d.driver_version << ", runtime " << d.runtime_version << "\n"
              << "  rates in G modmul/s (GPU time from events, operands resident); roofline = doc 04 Sec. 1.5\n";
    const Stream s = cx.stream(0);
    const uint32_t ell = 134217757u;

    std::cout << "-- dense u32 GEMM, ell = " << ell << " (k_max = 1023: n >= 1024 k-blocks)\n";
    dense_rows<uint32_t>(cx, s, ell, {512, 1024, 2048, 4096}, 4931.0, repeat, "u32");
    std::cout << "-- dense u64 GEMM, ell = 2^61-1 (Montgomery)\n";
    dense_rows<uint64_t>(cx, s, 2305843009213693951ull, {512, 1024, 2048}, 629.0, repeat, "u64");
    std::cout << "-- dense u32 GEMM at the u32 boundary primes (eager reduction regimes)\n";
    dense_rows<uint32_t>(cx, s, 2147483647u, {2048}, 2460.0, repeat, "u32 p=2^31-1");
    dense_rows<uint32_t>(cx, s, 4294967291u, {2048}, 2460.0, repeat, "u32 p=2^32-5");

    // ---- batched small-block program ----
    {
        const uint32_t dd = 129, inst = 2000, nops = 8;
        const ModCtx<uint32_t> ctx(ell);
        std::vector<DenseBuf<uint32_t>> ops;
        for (uint32_t i = 0; i < nops; ++i) ops.push_back(rnd<uint32_t>(dd, dd, ell, 100 + i));
        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        uint64_t st = 77;
        for (uint32_t i = 0; i < inst; ++i) { st = st * 6364136223846793005ull + 1442695040888963407ull; pairs.emplace_back((st >> 33) % nops, (st >> 40) % nops); }
        const double work = static_cast<double>(inst) * dd * dd * dd;
        std::cout << "-- batched small-block GEMM: " << inst << " x (" << dd << "x" << dd << "x" << dd << "), " << nops
                  << " distinct operands, ell = " << ell << "  (" << std::setprecision(2) << work / 1e9 << " G modmul)\n";
        // CPU single thread
        DenseBuf<uint32_t> Ccpu(dd, dd * inst, Layout::RowMajor);
        double tcpu = 1e9;
        for (uint32_t r = 0; r < repeat; ++r) {
            const auto t0 = Clock::now();
            for (uint32_t i = 0; i < inst; ++i)
                kernels::gemm<uint32_t>(ctx, cview(ops[pairs[i].first]), cview(ops[pairs[i].second]), view(Ccpu).sub(0, dd, i * dd, dd), GemmOp::Set);
            tcpu = std::min(tcpu, secs(t0, Clock::now()));
        }
        print_row(Row{"cpu single thread (kernels::gemm loop)", tcpu, work / tcpu / 1e9, 0, ""});
        std::vector<std::unique_ptr<DeviceBuf<uint32_t>>> dops;
        for (auto& o : ops) { dops.emplace_back(new DeviceBuf<uint32_t>(o)); dops.back()->upload(s); }
        DenseBuf<uint32_t> Cg(dd, dd * inst, Layout::RowMajor);
        DeviceBuf<uint32_t> dC(Cg);
        auto unbatched = [&](int tile) {
            for (uint32_t i = 0; i < inst; ++i)
                gemm<uint32_t>(ctx, dops[pairs[i].first]->cview(), dops[pairs[i].second]->cview(), dC.view().sub(0, dd, i * dd, dd), GemmOp::Set, s, GemmOpts{1, nullptr, 0, tile});
        };
        for (int tile : {32, 64}) {
            double t = 1e9;
            for (uint32_t r = 0; r < repeat; ++r) t = std::min(t, gpu_time(cx, s, [&] { unbatched(tile); }));
            std::ostringstream w; w << "gpu " << inst << " stream launches, tile=" << tile;
            print_row(Row{w.str(), t, work / t / 1e9, 4931.0, ""});
        }
        dC.mark_device_written(); dC.download(s);
        const bool same_unb = std::memcmp(Cg.data(), Ccpu.data(), Cg.footprint() * 4) == 0;
        for (int tile : {32, 64}) {
            GemmBatch<uint32_t> batch;
            batch.set_shape(dops[0]->cview(), dops[1]->cview(), dC.view().sub(0, dd, 0, dd));
            for (uint32_t i = 0; i < inst; ++i) batch.push(dops[pairs[i].first]->cview(), dops[pairs[i].second]->cview(), dC.view().sub(0, dd, i * dd, dd));
            batch.upload(s);
            double t = 1e9;
            for (uint32_t r = 0; r < repeat; ++r) t = std::min(t, gpu_time(cx, s, [&] { batch.run(ctx, GemmOp::Set, s, tile); }));
            std::ostringstream w; w << "gpu batched (1 launch), tile=" << tile;
            print_row(Row{w.str(), t, work / t / 1e9, 4931.0, tcpu / t > 0 ? "" : ""});
            GraphCapture g; g.begin(s); batch.run(ctx, GemmOp::Set, s, tile); g.end();
            double tg = 1e9;
            for (uint32_t r = 0; r < repeat; ++r) tg = std::min(tg, gpu_time(cx, s, [&] { g.launch(s); }));
            std::ostringstream w2; w2 << "gpu batched, graph replay, tile=" << tile;
            std::ostringstream n2; n2 << std::setprecision(1) << std::fixed << (tcpu / tg) << "x cpu";
            print_row(Row{w2.str(), tg, work / tg / 1e9, 4931.0, n2.str()});
        }
        {
            GraphCapture g; g.begin(s); unbatched(32); g.end();
            double tg = 1e9;
            for (uint32_t r = 0; r < repeat; ++r) tg = std::min(tg, gpu_time(cx, s, [&] { g.launch(s); }));
            std::ostringstream w2; w2 << "gpu " << inst << " launches as one graph (" << g.num_nodes() << " nodes)";
            print_row(Row{w2.str(), tg, work / tg / 1e9, 4931.0, ""});
        }
        dC.mark_device_written(); dC.download(s);
        const bool same_bat = std::memcmp(Cg.data(), Ccpu.data(), Cg.footprint() * 4) == 0;
        std::cout << "  memcmp vs cpu: unbatched " << (same_unb ? "identical" : "DIFFERENT") << ", batched/graph "
                  << (same_bat ? "identical" : "DIFFERENT") << "\n";
        if (!same_unb || !same_bat) return 1;
    }

    // ---- transfers ----
    {
        const size_t bytes = 64u << 20;
        std::vector<char> host(bytes, 1);
        void* dev = cx.dalloc(bytes);
        void* pin = cx.pinned_alloc(bytes);
        std::memset(pin, 2, bytes);
        std::cout << "-- transfers, " << (bytes >> 20) << " MB\n";
        auto row = [&](const char* what, auto fn) {
            double t = 1e9;
            for (uint32_t r = 0; r < repeat; ++r) { const auto t0 = Clock::now(); fn(); t = std::min(t, secs(t0, Clock::now())); }
            std::ostringstream n; n << std::fixed << std::setprecision(1) << (bytes / t / 1e9) << " GB/s";
            print_row(Row{what, t, 0, 0, n.str()});
        };
        row("h2d staged through the pinned pool", [&] { cx.upload(host.data(), dev, bytes, s); cx.sync(s); });
        row("d2h staged through the pinned pool", [&] { cx.download(host.data(), dev, bytes, s); });
        row("h2d direct from a pinned buffer", [&] { cx.upload_pageable(pin, dev, bytes); });
        row("d2h direct into a pinned buffer", [&] { cx.download_pageable(pin, dev, bytes); });
        row("h2d direct from pageable memory", [&] { cx.upload_pageable(host.data(), dev, bytes); });
        row("d2h direct into pageable memory", [&] { cx.download_pageable(host.data(), dev, bytes); });
        cx.pinned_free(pin);
        cx.dfree(dev, bytes);
    }

    // ---- launch cost ----
    {
        const ModCtx<uint32_t> ctx(ell);
        DeviceBuf<uint32_t> v(1, 1, Layout::RowMajor);
        cx.memset_d(v.dptr(), 0, 4, s);
        const uint32_t N = 1000;
        std::cout << "-- launch cost, " << N << " one-element kernels\n";
        double t_sync = 1e9, t_async = 1e9, t_graph = 1e9;
        for (uint32_t r = 0; r < repeat; ++r) {
            auto t0 = Clock::now();
            for (uint32_t i = 0; i < N; ++i) { scal<uint32_t>(ctx, 3u, v.view(), s); cx.sync(s); }
            t_sync = std::min(t_sync, secs(t0, Clock::now()));
            t0 = Clock::now();
            for (uint32_t i = 0; i < N; ++i) scal<uint32_t>(ctx, 3u, v.view(), s);
            cx.sync(s);
            t_async = std::min(t_async, secs(t0, Clock::now()));
        }
        GraphCapture g; g.begin(s);
        for (uint32_t i = 0; i < N; ++i) scal<uint32_t>(ctx, 3u, v.view(), s);
        g.end();
        for (uint32_t r = 0; r < repeat; ++r) {
            const auto t0 = Clock::now();
            g.launch(s); cx.sync(s);
            t_graph = std::min(t_graph, secs(t0, Clock::now()));
        }
        std::ostringstream a, b, c;
        a << std::fixed << std::setprecision(2) << t_sync / N * 1e6 << " us per kernel (launch + sync)";
        b << std::fixed << std::setprecision(2) << t_async / N * 1e6 << " us per kernel (async stream launch)";
        c << std::fixed << std::setprecision(2) << t_graph / N * 1e6 << " us per node (graph of " << g.num_nodes() << " nodes, replayed)";
        print_row(Row{"stream launch + sync", t_sync, 0, 0, a.str()});
        print_row(Row{"stream launch, async", t_async, 0, 0, b.str()});
        print_row(Row{"graph replay", t_graph, 0, 0, c.str()});
    }
    const TransferStats ts = cx.transfer_stats();
    const PinnedPoolStats ps = cx.pinned_stats();
    std::cout << "  totals: h2d " << (ts.h2d_bytes >> 20) << " MB, d2h " << (ts.d2h_bytes >> 20) << " MB, pinned pool peak "
              << (ps.peak_bytes >> 20) << " MB (cap " << (ps.cap_bytes >> 20) << " MB)\n";
    std::cout << "BENCH mode=micro op=cuda device=\"" << d.name << "\"\n";
    return 0;
}

} // namespace

void InstallMicroHooks() {
    if (!available()) return;
    micro::GpuHooks h;
    h.gemm = &GemmGpu; h.gemv = &GemvGpu; h.transpose = &TransposeGpu; h.suite = &suite;
    micro::SetGpuHooks(h);
}

} // namespace cuda
} // namespace engine
