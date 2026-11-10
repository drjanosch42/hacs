// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// CudaImmaValidation.cpp -- see CudaImmaValidation.h (slot-22 G1).
#include "CudaImmaValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
#include "../Engine/EngineTestUtil.h"
#include "../Engine/Executor.h"
#include "../Engine/Passes.h"
#include "../Engine/Recorder.h"
#include "../Engine/ThreadPool.h"
#include "LegacyConfig.h"          // (S24 CL-R2) RunLegacyArm
#include "RetiredDeviceLevers.h"   // (S24 CL-R2) the EFFECTIVE line

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
uint64_t g_imma_cases = 0;   // cases that took the IMMA route (asserted per case)
void fail(const std::string& what) {
    ++g_fail;
    if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
}
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) fail(what); }

template<class T> std::string str(T v) { return std::to_string(v); }
const char* lname(Layout l) { return l == Layout::RowMajor ? "R" : "C"; }
const char* opname(GemmOp op) { return op == GemmOp::Set ? "Set" : op == GemmOp::Add ? "Add" : "Sub"; }

cu::Stream g_stream;

uint64_t imma_launches() { return cu::launch_stats().imma_launches; }

// Operand patterns.
enum class Pat { Random, Zero, ExtremePos, ExtremeNeg, ZeroLines, DupCols };
const char* patname(Pat q) {
    switch (q) {
        case Pat::Random: return "random";
        case Pat::Zero: return "zero";
        case Pat::ExtremePos: return "extreme+";
        case Pat::ExtremeNeg: return "extreme-";
        case Pat::ZeroLines: return "zero-lines";
        case Pat::DupCols: return "dup-cols";
    }
    return "?";
}

// A buffer (logical rows x cols of the PHYSICAL buffer) whose padding is garbage.
// ExtremePos: every entry floor(p/2) (balanced +hb); ExtremeNeg: p - floor(p/2) (balanced -hb
// for odd p; for even p it is +hb again, both signs are then the same value).
DenseBuf<uint32_t> make_buf(Splitmix& rng, uint32_t p, uint32_t rows, uint32_t cols, Layout layout, uint32_t pad, Pat pat) {
    const uint32_t minor = layout == Layout::RowMajor ? cols : rows;
    DenseBuf<uint32_t> b(rows, cols, layout, minor + pad, Invariant::Canonical);
    uint32_t* d = b.data();
    for (size_t i = 0; i < b.footprint(); ++i) d[i] = static_cast<uint32_t>(rng.next());
    const uint32_t hb = p / 2;
    const uint32_t lines[] = {0, 31, 32, 63, 64, 127, 128, 255, 256};
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) {
            uint32_t v = 0;
            switch (pat) {
                case Pat::Random: case Pat::ZeroLines: case Pat::DupCols: v = static_cast<uint32_t>(rng.below(p)); break;
                case Pat::Zero: v = 0; break;
                case Pat::ExtremePos: v = hb; break;
                case Pat::ExtremeNeg: v = p - hb; break;
            }
            b.at(i, j) = v;
        }
    if (pat == Pat::ZeroLines) {
        for (uint32_t z : lines) {
            if (z < rows) for (uint32_t j = 0; j < cols; ++j) b.at(z, j) = 0;
            if (z < cols) for (uint32_t i = 0; i < rows; ++i) b.at(i, z) = 0;
        }
    }
    if (pat == Pat::DupCols && cols >= 2) {
        for (uint32_t j = 1; j < cols; j += 3) {
            const uint32_t src = static_cast<uint32_t>(rng.below(j));
            for (uint32_t i = 0; i < rows; ++i) b.at(i, j) = b.at(i, src);
        }
    }
    return b;
}

bool same_bytes(const DenseBuf<uint32_t>& a, const DenseBuf<uint32_t>& b) {
    return a.footprint() == b.footprint() && std::memcmp(a.data(), b.data(), a.footprint() * sizeof(uint32_t)) == 0;
}

template<class Elt>
struct Mirror {
    DenseBuf<Elt>& host;
    cu::DeviceBuf<Elt> dev;
    explicit Mirror(DenseBuf<Elt>& h, cu::Stream s) : host(h), dev(h) { dev.upload(s); }
    cu::DeviceView<const Elt> cv() const { return dev.cview(); }
    cu::DeviceView<Elt> v() { return dev.view(); }
    void fetch(cu::Stream s) { dev.mark_device_written(); dev.download(s); }
};

struct DevWs {
    void* p = nullptr; size_t bytes = 0;
    explicit DevWs(size_t b) : bytes(b) { if (b) p = cu::CudaCtx::get().dalloc(b); }
    ~DevWs() { if (p) cu::CudaCtx::get().dfree(p, bytes); }
    DevWs(const DevWs&) = delete;
    DevWs& operator=(const DevWs&) = delete;
};

enum class Entry { Gemm, GemmNt, Schur };

struct Case {
    uint32_t m, n, k;
    Layout la = Layout::RowMajor, lb = Layout::RowMajor, lc = Layout::RowMajor;
    bool ta = false, tb = false;
    GemmOp op = GemmOp::Set;
    Entry entry = Entry::Gemm;
    Pat pa = Pat::Random, pb = Pat::Random;
    int imma = 1;                 // GemmOpts::imma
    int ws = 1;                   // 1: full workspace, 0: none, -1: one byte short (K chunks), -2: below one 64-deep chunk
    uint32_t ws_k = 0;            // > 0: a workspace for a K chunk of ws_k (a multiple of 64): ceil(k / ws_k) chunks
    uint32_t flush = 0;           // GemmOpts::imma_flush_tiles
    bool expect_imma = true;
    int expect_launches = -1;     // >= 0: the exact IMMA launch count (else 1 / 0 from expect_imma)
    std::string tag;
};

// One case: CPU twin vs device, memcmp of the whole C footprint, and the IMMA launch count.
void run_case(const ModCtx<uint32_t>& ctx, Splitmix& rng, const Case& c) {
    const uint32_t p = ctx.p;
    const uint32_t m = c.m, n = c.n, k = c.k;
    const bool schur = c.entry == Entry::Schur;
    const GemmOp op = schur ? GemmOp::Sub : c.op;
    const bool nt = c.entry == Entry::GemmNt;
    DenseBuf<uint32_t> A = c.ta ? make_buf(rng, p, k, m, c.la, 3, c.pa) : make_buf(rng, p, m, k, c.la, 3, c.pa);
    // B physical: gemm: k x n (tb: n x k transposed); gemm_nt: n x k (tb: k x n transposed)
    const bool b_kn = nt ? c.tb : !c.tb;
    DenseBuf<uint32_t> Bp = b_kn ? make_buf(rng, p, k, n, c.lb, 2, c.pb) : make_buf(rng, p, n, k, c.lb, 2, c.pb);
    DenseBuf<uint32_t> C0 = make_buf(rng, p, m, n, c.lc, 1, Pat::Random);
    DenseBuf<uint32_t> Ccpu(C0), Cgpu(C0);

    DenseView<const uint32_t> Av = engine::cview(A); if (c.ta) Av = Av.transposed_view();
    DenseView<const uint32_t> Bv = engine::cview(Bp); if (c.tb) Bv = Bv.transposed_view();
    if (nt) engine::kernels::gemm_nt<uint32_t>(ctx, Av, Bv, engine::view(Ccpu), op);
    else    engine::kernels::gemm<uint32_t>(ctx, Av, Bv, engine::view(Ccpu), op);

    Mirror<uint32_t> dA(A, g_stream), dB(Bp, g_stream), dC(Cgpu, g_stream);
    cu::DeviceView<const uint32_t> dAv = dA.cv(); if (c.ta) dAv = dAv.transposed_view();
    cu::DeviceView<const uint32_t> dBv = dB.cv(); if (c.tb) dBv = dBv.transposed_view();
    const size_t need = c.ws_k ? cu::imma_workspace_bytes(m, n, c.ws_k) : cu::imma_workspace_bytes(m, n, k);
    const size_t below = cu::imma_workspace_bytes(m, n, std::min<uint32_t>(k, 64)) - 1;
    DevWs ws(c.ws == 0 ? 0 : c.ws == -1 ? need - 1 : c.ws == -2 ? below : need);
    cu::GemmOpts o;
    o.imma = c.imma;
    o.imma_workspace = ws.p; o.imma_workspace_bytes = ws.bytes;
    o.imma_flush_tiles = c.flush;
    const uint64_t before = imma_launches();
    if (schur)   cu::schur_update<uint32_t>(ctx, dAv, dBv, dC.v(), g_stream, o);
    else if (nt) cu::gemm_nt<uint32_t>(ctx, dAv, dBv, dC.v(), op, g_stream, o);
    else         cu::gemm<uint32_t>(ctx, dAv, dBv, dC.v(), op, g_stream, o);
    dC.fetch(g_stream);
    const uint64_t got = imma_launches() - before;
    const std::string what = std::string(schur ? "schur_update" : nt ? "gemm_nt" : "gemm") + " p=" + str(p) + " m=" + str(m) +
                             " n=" + str(n) + " k=" + str(k) + " A" + lname(c.la) + (c.ta ? "t" : "") + " B" + lname(c.lb) +
                             (c.tb ? "t" : "") + " C" + lname(c.lc) + " op=" + opname(op) + " A:" + patname(c.pa) + " B:" +
                             patname(c.pb) + " imma=" + str(c.imma) + " ws=" + str(c.ws) + " ws_k=" + str(c.ws_k) + " flush=" + str(c.flush) +
                             (c.tag.empty() ? "" : " [" + c.tag + "]");
    check(same_bytes(Ccpu, Cgpu), what + ": device bytes == engine::kernels bytes");
    const uint64_t want = c.expect_launches >= 0 ? static_cast<uint64_t>(c.expect_launches) : (c.expect_imma ? 1u : 0u);
    check(got == want, what + ": IMMA launches " + str(got) + " (expected " + str(want) + ")");
    if (got) ++g_imma_cases;
}

// (A) + (B): random and structured operands for one modulus.
void arm_ab(const ModCtx<uint32_t>& ctx, bool big) {
    Splitmix rng(0x1A3A0000ULL ^ ctx.p);
    struct S { uint32_t m, n, k; };
    std::vector<S> sizes = {{1, 1, 1}, {2, 3, 5}, {7, 16, 17}, {16, 8, 32}, {33, 17, 65}, {64, 64, 64}, {65, 129, 127},
                            {128, 128, 128}, {129, 257, 200}, {300, 200, 1000}, {517, 259, 1025}};
    if (big) sizes.push_back({1024, 1031, 1024});
    uint32_t rot = 0;
    for (const S& s : sizes) {
        for (int v = 0; v < 3; ++v, ++rot) {
            Case c;
            c.m = s.m; c.n = s.n; c.k = s.k;
            c.la = (rot & 1) ? Layout::ColMajor : Layout::RowMajor;
            c.lb = (rot & 2) ? Layout::ColMajor : Layout::RowMajor;
            c.lc = (rot & 4) ? Layout::ColMajor : Layout::RowMajor;
            c.ta = (rot % 5) == 1; c.tb = (rot % 7) == 2;
            c.op = (v == 0) ? GemmOp::Set : (v == 1) ? GemmOp::Add : GemmOp::Sub;
            c.entry = (rot % 6 == 3) ? Entry::GemmNt : (rot % 6 == 5) ? Entry::Schur : Entry::Gemm;
            run_case(ctx, rng, c);
        }
    }
    // (B) structured
    const Pat pats[][2] = {{Pat::Zero, Pat::Random}, {Pat::Random, Pat::Zero}, {Pat::ExtremePos, Pat::ExtremePos},
                           {Pat::ExtremeNeg, Pat::ExtremePos}, {Pat::ExtremeNeg, Pat::ExtremeNeg},
                           {Pat::ZeroLines, Pat::ZeroLines}, {Pat::DupCols, Pat::DupCols}, {Pat::ZeroLines, Pat::DupCols}};
    for (const auto& pp : pats)
        for (const S& s : std::vector<S>{{129, 300, 257}, {260, 130, 513}}) {
            Case c;
            c.m = s.m; c.n = s.n; c.k = s.k;
            c.pa = pp[0]; c.pb = pp[1];
            c.op = (rot++ % 3 == 0) ? GemmOp::Sub : GemmOp::Set;
            c.lb = (rot & 1) ? Layout::ColMajor : Layout::RowMajor;
            run_case(ctx, rng, c);
        }
}

// (C) K-chunking.
void arm_c() {
    // forced flush intervals around their boundaries (extreme residues: the accumulator is as large as it gets)
    for (uint32_t p : {2u, 13u, 127u}) {
        const ModCtx<uint32_t> ctx(p);
        Splitmix rng(0xC0C0ULL ^ p);
        for (uint32_t f : {1u, 2u, 3u}) {
            for (uint32_t k : {64 * f - 1, 64 * f, 64 * f + 1, 3 * 64 * f + 16, 5 * 64 * f - 16}) {
                for (Pat q : {Pat::ExtremePos, Pat::Random}) {
                    Case c;
                    c.m = 40; c.n = 72; c.k = k; c.flush = f;
                    c.pa = q; c.pb = (q == Pat::ExtremePos) ? Pat::ExtremeNeg : Pat::Random;
                    c.op = GemmOp::Sub;
                    c.tag = "forced flush";
                    run_case(ctx, rng, c);
                }
            }
        }
    }
    // K chunks (a workspace below the whole K): first chunk op, then Add / Sub; with and without forced flushes
    for (uint32_t p : {2u, 13u, 127u}) {
        const ModCtx<uint32_t> ctx(p);
        Splitmix rng(0xC4C4ULL ^ p);
        for (uint32_t wk : {64u, 128u, 320u}) {
            for (uint32_t k : {wk - 1, wk + 1, 3 * wk, 1000u}) {
                for (GemmOp op : {GemmOp::Set, GemmOp::Add, GemmOp::Sub}) {
                    Case c;
                    c.m = 131; c.n = 66; c.k = k; c.ws_k = wk; c.op = op;
                    c.pa = (op == GemmOp::Add) ? Pat::ExtremePos : Pat::Random;
                    c.pb = (op == GemmOp::Add) ? Pat::ExtremeNeg : Pat::Random;
                    c.lb = (op == GemmOp::Sub) ? Layout::ColMajor : Layout::RowMajor;
                    c.flush = (op == GemmOp::Sub) ? 1 : 0;
                    c.expect_launches = static_cast<int>(k <= wk ? 1 : (k + wk - 1) / wk);
                    c.tag = "K chunks of " + str(wk);
                    run_case(ctx, rng, c);
                }
            }
        }
    }
    // the true int32 bound: K = 64 T(p) +- (no flush up to 64 T(p) after padding, a flush above)
    for (uint32_t p : {127u, 125u}) {
        const ModCtx<uint32_t> ctx = (p == 125u) ? ModCtx<uint32_t>::prime_power(5u, 3) : ModCtx<uint32_t>(p);
        Splitmix rng(0xB0B0ULL ^ p);
        const uint32_t T = cu::imma_max_flush_tiles(p);
        const uint32_t K0 = 64 * T;
        std::cout << "  (C) p=" << p << ": T(p) = " << T << " k-tiles, K0 = 64 T = " << K0 << std::endl;
        for (uint32_t k : {K0 - 16, K0, K0 + 1, K0 + 16, K0 + 64, K0 + 65}) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                Case c;
                c.m = 24; c.n = 20; c.k = k;
                c.pa = Pat::ExtremePos; c.pb = sgn ? Pat::ExtremeNeg : Pat::ExtremePos;
                c.tag = "int32 bound";
                run_case(ctx, rng, c);
            }
        }
    }
    // a flush interval above T(p) is refused (ENGINE_CHECK), never silently used
    {
        const ModCtx<uint32_t> ctx(127u);
        bool threw = false;
        try {
            Splitmix rng(0xBADULL);
            Case c;
            c.m = 8; c.n = 8; c.k = 64;
            c.flush = cu::imma_max_flush_tiles(127u) + 1;
            run_case(ctx, rng, c);
        } catch (const std::logic_error&) {
            threw = true;
        }
        check(threw, "imma_flush_tiles > T(127) is refused (std::logic_error)");
    }
}

// Sets / unsets one environment variable for a scope (arms (D) and (F)).
struct EnvGuard {
    std::string name, old; bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* o = std::getenv(n); had = o != nullptr; if (had) old = o;
        if (v) setenv(n, v, 1); else unsetenv(n);
    }
    ~EnvGuard() { if (had) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

// (D, audit R7) the direct-layer policy: one GemmBatch at p = 13 whose prepare_imma runs under
// ENGINE_GPU_IMMA = prep_env and whose run (imma = -1) under run_env; expects IMMA launches iff
// both are "on" (nullptr), and the CPU bytes in every case.
void arm_d_batch_policy(Splitmix& rng, const char* prep_env, const char* run_env) {
    const uint32_t p = 13u, d = 65u, k = 130u, ninst = 6u;
    const ModCtx<uint32_t> ctx(p);
    DenseBuf<uint32_t> A = make_buf(rng, p, d, k, Layout::RowMajor, 0, Pat::Random);
    DenseBuf<uint32_t> B = make_buf(rng, p, k, d, Layout::ColMajor, 0, Pat::Random);
    DenseBuf<uint32_t> C0 = make_buf(rng, p, d * ninst, d, Layout::RowMajor, 0, Pat::Random);
    DenseBuf<uint32_t> Ccpu(C0), Cdev(C0);
    for (uint32_t i = 0; i < ninst; ++i)
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(Ccpu).sub(i * d, d, 0, d), GemmOp::Add);
    cu::DeviceBuf<uint32_t> dA(A), dB(B);
    dA.upload(g_stream); dB.upload(g_stream);
    Mirror<uint32_t> dC(Cdev, g_stream);
    cu::GemmBatch<uint32_t> batch;
    for (uint32_t i = 0; i < ninst; ++i) {
        cu::DeviceView<uint32_t> c = dC.v().sub(i * d, d, 0, d);
        if (i == 0) batch.set_shape(dA.cview(), dB.cview(), c);
        batch.push(dA.cview(), dB.cview(), c);
    }
    batch.upload(g_stream);
    const std::string what = std::string("policy GemmBatch: prepare under ENGINE_GPU_IMMA=") + (prep_env ? prep_env : "unset") +
                             ", run under " + (run_env ? run_env : "unset");
    bool prepared = false;
    {   EnvGuard g("ENGINE_GPU_IMMA", prep_env); prepared = batch.prepare_imma(ctx); }
    check(prepared == (prep_env == nullptr), what + ": prepare_imma " + (prepared ? "true" : "false"));
    uint64_t got = 0;
    {
        EnvGuard g("ENGINE_GPU_IMMA", run_env);
        const uint64_t before = imma_launches();
        batch.run(ctx, GemmOp::Add, g_stream, 0, -1);
        dC.fetch(g_stream);
        got = imma_launches() - before;
    }
    const bool expect = prep_env == nullptr && run_env == nullptr;
    check(same_bytes(Ccpu, Cdev), what + ": device bytes == engine::kernels bytes");
    check((got > 0) == expect, what + ": IMMA launches " + str(got) + " (expected " + (expect ? ">0" : "0") + ")");
    if (got) ++g_imma_cases;
}

// (D) routing.
void arm_d() {
    Splitmix rng(0xD0D0ULL);
    // non-qualifying u32 moduli: the u32 kernel even with imma = 1 and a workspace
    std::vector<ModCtx<uint32_t>> nq;
    nq.push_back(ModCtx<uint32_t>::prime_power(2u, 7));   // 128
    nq.push_back(ModCtx<uint32_t>(131u));
    nq.push_back(ModCtx<uint32_t>(65521u));
    nq.push_back(ModCtx<uint32_t>(134217757u));
    for (const ModCtx<uint32_t>& ctx : nq) {
        check(!cu::imma_eligible(ctx), "p=" + str(ctx.p) + " is not IMMA-eligible");
        for (GemmOp op : {GemmOp::Set, GemmOp::Sub}) {
            Case c;
            c.m = 130; c.n = 70; c.k = 300; c.op = op; c.expect_imma = false; c.tag = "non-qualifying modulus";
            run_case(ctx, rng, c);
        }
    }
    const ModCtx<uint32_t> c13(13u);
    check(cu::imma_eligible(c13), "p=13 is IMMA-eligible");
    {   Case c; c.m = 130; c.n = 70; c.k = 300; c.imma = 0; c.expect_imma = false; c.tag = "imma=0"; run_case(c13, rng, c); }
    {   Case c; c.m = 130; c.n = 70; c.k = 300; c.ws = 0; c.expect_imma = false; c.tag = "no workspace"; run_case(c13, rng, c); }
    {   Case c; c.m = 130; c.n = 70; c.k = 300; c.ws = -1; c.expect_launches = 2; c.tag = "short workspace: K chunks 256 + 44"; run_case(c13, rng, c); }
    {   Case c; c.m = 130; c.n = 70; c.k = 300; c.ws = -2; c.expect_imma = false; c.tag = "workspace below one 64-deep chunk"; run_case(c13, rng, c); }
    {   Case c; c.m = 130; c.n = 70; c.k = 0; c.op = GemmOp::Set; c.expect_imma = false; c.tag = "k=0"; run_case(c13, rng, c); }
    {   Case c; c.m = 130; c.n = 70; c.k = 0; c.op = GemmOp::Add; c.expect_imma = false; c.tag = "k=0"; run_case(c13, rng, c); }
    // (audit R7) imma = -1 follows ENGINE_GPU_IMMA at the kernel layer itself, tested under BOTH
    // states whatever the process environment is ("0" -> no launch, unset -> launch)
    // (S24 CL-R2) ENGINE_GPU_IMMA is a RETIRED lever: the "0" states are legacy arms (an OFF build refuses "0")
    for (const char* env : {"0", static_cast<const char*>(nullptr)}) {
        if (env != nullptr && !oc::legacy::RunLegacyArm("CudaImmaBitIdentityValidation", "D-policy-off")) continue;
        EnvGuard g("ENGINE_GPU_IMMA", env);
        Case c; c.m = 130; c.n = 70; c.k = 300; c.imma = -1; c.expect_imma = (env == nullptr);
        c.tag = std::string("policy ENGINE_GPU_IMMA=") + (env ? env : "unset");
        std::cout << "  (D) imma = -1 under ENGINE_GPU_IMMA=" << (env ? env : "unset") << ": expect "
                  << (c.expect_imma ? "IMMA" : "u32") << std::endl;
        run_case(c13, rng, c);
    }
    // GemmBatch: prepare_imma refuses under "0"; a prepared batch run (imma = -1) under "0" falls back
    if (oc::legacy::RunLegacyArm("CudaImmaBitIdentityValidation", "D-batch-policy-off")) {   // (S24 CL-R2)
        arm_d_batch_policy(rng, "0", nullptr);
        arm_d_batch_policy(rng, nullptr, "0");
    }
    arm_d_batch_policy(rng, nullptr, nullptr);
    // a uint64_t context is never IMMA-eligible (G1 is u32 only)
    {
        const ModCtx<uint64_t> c64(13ull);
        check(!cu::imma_eligible(c64), "uint64_t p=13 is not IMMA-eligible");
        DenseBuf<uint64_t> A(90, 200), B(200, 50), C(90, 50), R(90, 50);
        for (uint32_t i = 0; i < 90; ++i) for (uint32_t j = 0; j < 200; ++j) A.at(i, j) = rng.below(13);
        for (uint32_t i = 0; i < 200; ++i) for (uint32_t j = 0; j < 50; ++j) B.at(i, j) = rng.below(13);
        engine::kernels::gemm<uint64_t>(c64, engine::cview(A), engine::cview(B), engine::view(R), GemmOp::Set);
        Mirror<uint64_t> dA(A, g_stream), dB(B, g_stream), dC(C, g_stream);
        DevWs ws(cu::imma_workspace_bytes(90, 50, 200));
        cu::GemmOpts o; o.imma = 1; o.imma_workspace = ws.p; o.imma_workspace_bytes = ws.bytes;
        const uint64_t before = imma_launches();
        cu::gemm<uint64_t>(c64, dA.cv(), dB.cv(), dC.v(), GemmOp::Set, g_stream, o);
        dC.fetch(g_stream);
        check(imma_launches() == before, "uint64_t p=13: no IMMA launch");
        check(std::memcmp(C.data(), R.data(), C.footprint() * sizeof(uint64_t)) == 0, "uint64_t p=13: device bytes == engine::kernels bytes");
    }
}

// (E) GemmBatch.
void arm_e(uint32_t p, uint32_t d, uint32_t k, uint32_t ninst, GemmOp op, Layout lc) {
    const ModCtx<uint32_t> ctx(p);
    Splitmix rng(0xEEE0ULL ^ (p * 131u + d));
    std::vector<DenseBuf<uint32_t>> As, Bs;
    for (int i = 0; i < 5; ++i) As.push_back(make_buf(rng, p, d, k, Layout::RowMajor, 0, i == 0 ? Pat::ExtremePos : Pat::Random));
    for (int i = 0; i < 4; ++i) Bs.push_back(make_buf(rng, p, k, d, Layout::ColMajor, 0, i == 0 ? Pat::ExtremeNeg : Pat::Random));
    DenseBuf<uint32_t> C0 = make_buf(rng, p, d * ninst, d, lc, 0, Pat::Random);
    DenseBuf<uint32_t> Ccpu(C0), Cimma(C0), Cold(C0);
    std::vector<std::pair<int, int>> inst;
    for (uint32_t i = 0; i < ninst; ++i) inst.emplace_back(static_cast<int>(rng.below(5)), static_cast<int>(rng.below(4)));
    for (uint32_t i = 0; i < ninst; ++i)
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(As[inst[i].first]), engine::cview(Bs[inst[i].second]),
                                        engine::view(Ccpu).sub(i * d, d, 0, d), op);
    std::vector<std::unique_ptr<cu::DeviceBuf<uint32_t>>> dA, dB;
    for (auto& a : As) { dA.push_back(std::make_unique<cu::DeviceBuf<uint32_t>>(a)); dA.back()->upload(g_stream); }
    for (auto& b : Bs) { dB.push_back(std::make_unique<cu::DeviceBuf<uint32_t>>(b)); dB.back()->upload(g_stream); }
    const std::string what = "GemmBatch p=" + str(p) + " d=" + str(d) + " k=" + str(k) + " x" + str(ninst) + " op=" + opname(op) + " C" + lname(lc);
    for (int arm = 0; arm < 2; ++arm) {
        DenseBuf<uint32_t>& Ch = arm == 0 ? Cimma : Cold;
        Mirror<uint32_t> dC(Ch, g_stream);
        cu::GemmBatch<uint32_t> batch;
        for (uint32_t i = 0; i < ninst; ++i) {
            cu::DeviceView<const uint32_t> a = dA[inst[i].first]->cview(), b = dB[inst[i].second]->cview();
            cu::DeviceView<uint32_t> c = dC.v().sub(i * d, d, 0, d);
            if (i == 0) batch.set_shape(a, b, c);
            batch.push(a, b, c);
        }
        batch.upload(g_stream);
        const bool prepared = (arm == 0) ? batch.prepare_imma(ctx) : false;
        if (arm == 0) check(prepared == cu::imma_policy(), what + ": prepare_imma follows the policy");
        const uint64_t before = imma_launches();
        batch.run(ctx, op, g_stream, 0, arm == 0 ? 1 : 0);
        dC.fetch(g_stream);
        const uint64_t got = imma_launches() - before;
        check(same_bytes(Ccpu, Ch), what + (arm == 0 ? " (IMMA)" : " (u32 kernel)") + ": device bytes == engine::kernels bytes");
        check((got > 0) == (arm == 0 && prepared), what + ": IMMA launches " + str(got));
        if (got) ++g_imma_cases;
    }
}

// (F) the tape.

uint64_t g_last_blocks = 0, g_last_drops = 0;

std::vector<uint64_t> tape_family(uint32_t p, engine::tape::DevicePolicy pol, bool verify, uint64_t& gpu_nodes, uint64_t& imma_nodes) {
    using namespace engine::tape;
    using namespace engine::tape::test;
    const ModCtx<uint32_t> ctx(p);
    ThreadPool pool(4);
    Executor<uint32_t> ex(&pool);
    ex.set_verify_device(verify);
    PassOptions opts; opts.batch = true; opts.policy = pol;
    std::vector<uint64_t> h;
    Rng rng(0xF00DULL ^ p);
    // 1. a plain GEMM (and its replay: the second run captures the graph); kept for step 5
    const DenseBuf<uint32_t> A1 = random_buf<uint32_t>(rng, 300, 777, p), B1 = random_buf<uint32_t>(rng, 777, 260, p, Layout::ColMajor);
    DenseBuf<uint32_t> C1(300, 260), C1b(300, 260), C1c(300, 260);
    Recorder<uint32_t> rec1(ctx);
    Program<uint32_t> prog1;
    {
        const BufId a = rec1.input(cview(A1)), b = rec1.input(cview(B1)), c = rec1.output(view(C1));
        rec1.gemm(rec1.whole(a), rec1.whole(b), rec1.whole(c), GemmOp::Set);
        prog1 = rec1.finish();
        ex.run(prog1, rec1.bindings(), opts);
        h.push_back(hash_view(cview(C1)));
        Bindings<uint32_t> bb = rec1.bindings(); bb.outputs[0] = view(C1b);
        ex.run(static_cast<const Program<uint32_t>&>(prog1), bb);
        h.push_back(hash_view(cview(C1b)));
        h.push_back(hash_view(cview(C1b)) == hash_view(cview(C1)) ? 1 : 0);
    }
    // 2. a scaled accumulate (alpha != 1, Sub) into a temp, then out
    {
        const DenseBuf<uint32_t> A = random_buf<uint32_t>(rng, 129, 513, p), B = random_buf<uint32_t>(rng, 513, 200, p);
        const DenseBuf<uint32_t> C0 = random_buf<uint32_t>(rng, 129, 200, p);
        DenseBuf<uint32_t> C(C0);
        Recorder<uint32_t> rec(ctx);
        const BufId a = rec.input(cview(A)), b = rec.input(cview(B)), c = rec.output(view(C));
        rec.gemm(rec.whole(a), rec.whole(b), rec.whole(c), GemmOp::Sub, static_cast<uint32_t>(p - 1 > 1 ? p - 2 : 1));
        Program<uint32_t> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        h.push_back(hash_view(cview(C)));
    }
    // 3. Theta-like gemm_dense (batched small blocks, shared pool operands)
    {
        auto bpool = std::make_shared<engine::BlockPool<uint32_t>>();
        const engine::BlockMat<uint32_t> A = make_input<uint32_t>(rng, bpool, p, Shape::CqLike, std::vector<uint32_t>(9, 129),
                                                                  std::vector<uint32_t>(9, 129), 30, 8);
        const DenseBuf<uint32_t> X = random_buf<uint32_t>(rng, A.cols(), 33, p, Layout::ColMajor);
        DenseBuf<uint32_t> Y(A.rows(), 33, Layout::ColMajor);
        Recorder<uint32_t> rec(ctx);
        const BufId bX = rec.input(cview(X)), bY = rec.output(view(Y));
        rec.record_gemm_dense(A, rec.whole(bX), rec.whole(bY), GemmOp::Set);
        Program<uint32_t> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        h.push_back(hash_view(cview(Y)));
    }
    // 4. a GEMM whose IMMA workspace (4 MiB) outgrows the slot's block: the block grows, program 1 loses its graph
    {
        const DenseBuf<uint32_t> A = random_buf<uint32_t>(rng, 1100, 1500, p), B = random_buf<uint32_t>(rng, 1500, 1200, p);
        DenseBuf<uint32_t> C(1100, 1200);
        Recorder<uint32_t> rec(ctx);
        const BufId a = rec.input(cview(A)), b = rec.input(cview(B)), c = rec.output(view(C));
        rec.gemm(rec.whole(a), rec.whole(b), rec.whole(c), GemmOp::Set);
        Program<uint32_t> prog = rec.finish();
        ex.run(prog, rec.bindings(), opts);
        h.push_back(hash_view(cview(C)));
    }
    // 5. program 1 again (cached, re-pointed to the new block, graph re-captured)
    {
        Bindings<uint32_t> bb = rec1.bindings(); bb.outputs[0] = view(C1c);
        ex.run(static_cast<const Program<uint32_t>&>(prog1), bb);
        ex.run(static_cast<const Program<uint32_t>&>(prog1), bb);
        h.push_back(hash_view(cview(C1c)) == hash_view(cview(C1)) ? 1 : 0);
    }
    gpu_nodes = ex.device_stats().gpu_nodes;
    imma_nodes = ex.device_stats().imma_nodes;
    g_last_blocks = ex.device_stats().imma_blocks;
    g_last_drops = ex.device_stats().imma_graph_drops;
    return h;
}

void arm_f() {
    using engine::tape::DevicePolicy;
    for (uint32_t p : {13u, 2u}) {
        uint64_t g0 = 0, i0 = 0;
        std::vector<uint64_t> ref;
        {
            EnvGuard d("ENGINE_DEVICE", "cpu");
            ref = tape_family(p, DevicePolicy::Cpu, false, g0, i0);
        }
        struct Arm { const char* imma_env; bool verify; const char* name; bool expect; };
        const Arm arms[] = {{nullptr, true, "gpu verify", true}, {nullptr, false, "gpu (no verify)", true},
                            {"0", true, "gpu verify ENGINE_GPU_IMMA=0", false}, {"0", false, "gpu ENGINE_GPU_IMMA=0", false}};
        for (const Arm& a : arms) {
            // (S24 CL-R2) the ENGINE_GPU_IMMA=0 arms are legacy arms (an OFF build refuses "0")
            if (a.imma_env != nullptr && !oc::legacy::RunLegacyArm("CudaImmaBitIdentityValidation", "F-imma-off")) continue;
            EnvGuard d("ENGINE_DEVICE", "gpu");
            EnvGuard im("ENGINE_GPU_IMMA", a.imma_env);
            const bool expect = a.expect && cu::imma_policy();
            uint64_t gn = 0, in = 0;
            const uint64_t before = imma_launches();
            const std::vector<uint64_t> got = tape_family(p, DevicePolicy::Gpu, a.verify, gn, in);
            const uint64_t launched = imma_launches() - before;
            const std::string what = std::string("tape p=") + str(p) + " " + a.name;
            check(got == ref, what + ": hashes == the cpu policy's");
            check(gn > 0, what + ": GPU nodes executed (" + str(gn) + ")");
            check((launched > 0) == expect && (in > 0) == expect,
                  what + ": IMMA launches " + str(launched) + ", IMMA-prepared nodes " + str(in));
            // the slot-0 block grew (1 MiB -> 4 MiB) and dropped program 1's graph, which was then re-captured on the new block
            check(expect ? (g_last_blocks >= 2 && g_last_drops >= 1) : (g_last_blocks == 0 && g_last_drops == 0),
                  what + ": IMMA blocks " + str(g_last_blocks) + ", graph drops " + str(g_last_drops));
            std::cout << "  (F) " << what << ": gpu nodes " << gn << ", IMMA-prepared nodes " << in << ", IMMA launches " << launched
                      << ", IMMA blocks " << g_last_blocks << ", graph drops " << g_last_drops << std::endl;
            if (launched) ++g_imma_cases;
        }
    }
}

bool skip_if_unavailable(const char* name) {
    if (cu::available()) return false;
    std::cout << "[SKIP] " << name << ": no usable CUDA device (" << cu::unavailable_reason()
              << ") -- nothing to validate, returning 0" << std::endl;
    return true;
}

} // namespace

uint32_t CudaImmaBitIdentityValidation() {
    std::cout << "=== CudaImmaBitIdentityValidation (int8 tensor-core GEMM vs engine::kernels, memcmp) ===" << std::endl;
    std::cout << engine::lever::retired_device_lever_line(engine::lever::gpu_imma_lever()) << std::endl;   // (S24 CL-R2)
    if (skip_if_unavailable("CudaImmaBitIdentityValidation")) return 0;
    if (!cu::imma_supported()) {
        std::cout << "[SKIP] CudaImmaBitIdentityValidation: the device / build has no sm_80+ IMMA image -- every GEMM takes "
                     "the u32 kernel, nothing to validate, returning 0" << std::endl;
        return 0;
    }
    g_fail = 0; g_checks = 0; g_imma_cases = 0;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        const cu::DeviceInfo di = cu::device_info();
        std::cout << "  device " << di.index << ": " << di.name << " cc " << di.cc_major << "." << di.cc_minor << "; "
                  << cu::kernel_layer_version() << "; policy ENGINE_GPU_IMMA " << (cu::imma_policy() ? "on" : "off") << std::endl;
        g_stream = cu::CudaCtx::get().stream(1);
        // (A) + (B)
        for (uint32_t p : {2u, 3u, 5u, 13u, 127u}) arm_ab(ModCtx<uint32_t>(p), p == 13u || p == 127u);
        arm_ab(ModCtx<uint32_t>::prime_power(2u, 2), false);   // 4
        arm_ab(ModCtx<uint32_t>::prime_power(3u, 2), false);   // 9
        arm_ab(ModCtx<uint32_t>::prime_power(5u, 3), false);   // 125
        std::cout << "  (A)+(B) done: " << g_checks << " checks" << std::endl;
        arm_c();
        std::cout << "  (C) done: " << g_checks << " checks" << std::endl;
        arm_d();
        std::cout << "  (D) done: " << g_checks << " checks" << std::endl;
        for (uint32_t p : {2u, 13u, 127u}) {
            arm_e(p, 65, 65, 50, GemmOp::Set, Layout::RowMajor);
            arm_e(p, 129, 129, 20, GemmOp::Sub, Layout::ColMajor);
        }
        arm_e(13u, 7, 300, 70, GemmOp::Add, Layout::RowMajor);
        std::cout << "  (E) done: " << g_checks << " checks" << std::endl;
        arm_f();
        cu::CudaCtx::get().sync_all();
    } catch (const std::exception& e) {
        std::cout << "  [EXCEPTION] " << e.what() << std::endl;   // always printed, past the 40-line cap
        fail(std::string("exception: ") + e.what());
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "CUDA_IMMA checks=" << g_checks << " failing=" << g_fail << " imma_cases=" << g_imma_cases
              << " policy=" << (cu::imma_policy() ? "on" : "off") << " wall_s=" << secs << std::endl;
    std::cout << "CudaImmaBitIdentityValidation: " << g_checks << " checks, " << g_fail << " failures" << std::endl;
    return g_fail;
}
