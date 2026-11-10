// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "KernelAvx2BitIdentityValidation.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DenseBuf.h"
#include "DenseView.h"
#include "Kernels.h"
#include "ModCtx.h"
#include "RefLinAlg.h"
#include "../Engine/EngineTestUtil.h"

// See KernelAvx2BitIdentityValidation.h.

namespace {

using engine::DenseBuf;
using engine::DenseView;
using engine::Invariant;
using engine::Layout;
using engine::ModCtx;
using engine::Structure;
using engine::kernels::GemmOp;
using engine::kernels::KernelPath;
using engine::kernels::VnniPath;
using engine::kernels::Avx2PairPath;

struct Rng : engine::tape::test::SplitMix64 {
    explicit Rng(uint64_t seed) : SplitMix64(seed) {}
    uint32_t below(uint32_t m) { return m <= 1 ? 0 : static_cast<uint32_t>(next() % m); }
};

uint32_t g_fail = 0, g_checks = 0;
void check(bool ok, const std::string& what) { engine::tape::test::bit_check(g_checks, g_fail, ok, what); }
using engine::tape::test::str;
using engine::tape::test::opname;
using engine::tape::test::opint;

struct PathGuard {
    KernelPath kp = engine::kernels::kernel_path();
    VnniPath vp = engine::kernels::vnni_path();
    Avx2PairPath ap = engine::kernels::avx2_pair_path();
    ~PathGuard() {
        engine::kernels::set_kernel_path(kp);
        engine::kernels::set_vnni_path(vp);
        engine::kernels::set_avx2_pair_path(ap);
    }
};

// fill 0: uniform residues; 1: every entry p - 1; 2: a mix of 0, 1, p - 1 and uniform.
DenseBuf<uint32_t> make_buf(Rng& rng, uint32_t p, uint32_t rows, uint32_t cols, Layout layout, uint32_t pad, int fill) {
    const uint32_t minor = layout == Layout::RowMajor ? cols : rows;
    DenseBuf<uint32_t> b(rows, cols, layout, minor + pad, Invariant::Canonical, Structure::Generic);
    uint32_t* d = b.data();
    for (size_t i = 0; i < b.footprint(); ++i) d[i] = static_cast<uint32_t>(rng.next());   // padding garbage
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) {
            uint32_t v;
            if (fill == 1) v = p - 1;
            else if (fill == 2) { const uint32_t t = rng.below(4); v = t == 0 ? 0 : t == 1 ? 1 : t == 2 ? p - 1 : rng.below(p); }
            else v = rng.below(p);
            b.at(i, j) = v;
        }
    return b;
}

// The arms: (kernel path, AVX2 pair route, name); VNNI off throughout, so the u32 microkernels and the AVX2 pair
// route themselves are compared. The pair route only engages for p <= AVX2_PAIR_MAX_MODULUS.
struct Arm { KernelPath kp; Avx2PairPath ap; const char* name; };
std::vector<Arm> arms() {
    std::vector<Arm> a = {{KernelPath::Portable, Avx2PairPath::Off, "portable"}};
    if (engine::kernels::avx2_kernel_available()) {
        a.push_back({KernelPath::Avx2, Avx2PairPath::Off, "avx2-u64"});
        a.push_back({KernelPath::Avx2, Avx2PairPath::Force, "avx2-pair"});
        a.push_back({KernelPath::Avx2, Avx2PairPath::Auto, "avx2-auto"});
    }
    if (engine::kernels::avx512_kernel_available()) a.push_back({KernelPath::Avx512, Avx2PairPath::Off, "avx512"});
    return a;
}

// One product on every arm against the oracle (and hence against each other), whole buffer compared.
void gemm_case(const ModCtx<uint32_t>& ctx, Rng& rng, uint32_t m, uint32_t k, uint32_t n,
               Layout lb, Layout lc, GemmOp op, int fill, bool window, Layout la = Layout::RowMajor) {
    const uint32_t p = ctx.p;
    const uint32_t o = window ? 3 : 0, q = window ? 2 : 0, pad = window ? 5 : 0;
    DenseBuf<uint32_t> Ab = make_buf(rng, p, m + o, k + q, la, pad, fill);
    DenseBuf<uint32_t> Bb = make_buf(rng, p, k + o, n + q, lb, pad, fill);
    DenseBuf<uint32_t> C0 = make_buf(rng, p, m + o, n + q, lc, pad, fill == 1 ? 1 : 0);
    DenseView<const uint32_t> A = DenseView<const uint32_t>(Ab).sub(o, m, q, k);
    DenseView<const uint32_t> B = DenseView<const uint32_t>(Bb).sub(o, k, q, n);
    const std::string tag = "gemm p=" + str(p) + " m=" + str(m) + " k=" + str(k) + " n=" + str(n) +
                            (la == Layout::RowMajor ? "" : " AC") + " B" + (lb == Layout::RowMajor ? "R" : "C") + " C" + (lc == Layout::RowMajor ? "R" : "C") +
                            " " + opname(op) + (fill == 1 ? " worst" : fill == 2 ? " mixed" : "") + (window ? " window" : "");
    DenseBuf<uint32_t> Cref = C0;
    engine::ref::gemm<uint32_t>(p, A, B, DenseView<uint32_t>(Cref).sub(o, m, q, n), opint(op));
    PathGuard guard;
    engine::kernels::set_vnni_path(VnniPath::Off);
    for (const Arm& arm : arms()) {
        engine::kernels::set_kernel_path(arm.kp);
        engine::kernels::set_avx2_pair_path(arm.ap);
        DenseBuf<uint32_t> C = C0;
        engine::kernels::gemm<uint32_t>(ctx, A, B, DenseView<uint32_t>(C).sub(o, m, q, n), op);
        bool same = true;
        for (size_t i = 0; i < C.footprint() && same; ++i) same = C.data()[i] == Cref.data()[i];
        check(same, tag + " [" + arm.name + "] != oracle (or wrote outside the window)");
    }
}

const uint32_t kModuli[] = {2u, 3u, 131u, 211u, 271u, 349u, 691u, 997u, 2048u, 2053u, 4093u, 4096u, 4099u,
                            8191u, 8209u, 16381u, 16411u, 65521u,
                            134217757u, 1518500213u, 1518500249u, 2147483647u, 2147483659u, 4294967291u};

// (P) + (L): every modulus, every column remainder 1..33 and row count 1..17.
void arm_lanes(Rng& rng) {
    for (uint32_t p : kModuli) {
        const ModCtx<uint32_t> ctx(p);
        for (uint32_t n = 1; n <= 33; ++n)
            gemm_case(ctx, rng, 9, 37, n, Layout::RowMajor, Layout::RowMajor, n % 3 == 0 ? GemmOp::Sub : GemmOp::Add, n % 5 == 0 ? 2 : 0, n % 4 == 1);
        for (uint32_t m = 1; m <= 17; ++m)
            gemm_case(ctx, rng, m, 21, 19, Layout::RowMajor, Layout::RowMajor, m % 2 ? GemmOp::Set : GemmOp::Sub, 0, m % 3 == 0);
    }
}

// (K) + (W): k across the k-block and the k_max bound; the worst case (every entry p - 1).
void arm_kblock(Rng& rng) {
    for (uint32_t p : kModuli) {
        const ModCtx<uint32_t> ctx(p);
        for (uint32_t k : {1u, 2u, 255u, 256u, 257u, 513u, 1029u}) {
            gemm_case(ctx, rng, 8, k, 16, Layout::RowMajor, Layout::RowMajor, GemmOp::Add, 1, false);
            gemm_case(ctx, rng, 11, k, 27, Layout::RowMajor, Layout::RowMajor, GemmOp::Sub, 1, true);
            gemm_case(ctx, rng, 5, k, 13, Layout::RowMajor, Layout::RowMajor, GemmOp::Set, 0, false);
        }
    }
}

// (Y) layouts: B column-major (packed), C column-major (strided epilogue), windows, every op.
void arm_layouts(Rng& rng) {
    for (uint32_t p : {271u, 16381u, 134217757u, 2147483647u, 4294967291u}) {
        const ModCtx<uint32_t> ctx(p);
        for (Layout lb : {Layout::RowMajor, Layout::ColMajor})
            for (Layout lc : {Layout::RowMajor, Layout::ColMajor})
                for (GemmOp op : {GemmOp::Set, GemmOp::Add, GemmOp::Sub}) {
                    gemm_case(ctx, rng, 13, 300, 23, lb, lc, op, 0, true);
                    gemm_case(ctx, rng, 64, 70, 40, lb, lc, op, 2, false);
                }
    }
    // a larger product spanning several n panels (NC = 512) and k-blocks
    const ModCtx<uint32_t> ctx(349u);
    gemm_case(ctx, rng, 40, 600, 530, Layout::RowMajor, Layout::RowMajor, GemmOp::Add, 0, false);
}

// (F) (integration 3, AVX2K audit F3) the gaps the audit's stress driver covered: A column-major (the strided A reads of the
// pair packer and the u64 microkernel); the pair route's column-major-B packer in the worst case near p = 4096 with odd
// k-blocks (kc = 128: k = kc - 1, kc + 1, 2 kc + 1); k across the u64 microkernel's k_max (4 .. 8) near 2^30.5 .. 2^32.
void arm_audit_f3(Rng& rng) {
    for (uint32_t p : {271u, 4093u, 4096u, 16381u, 2147483647u}) {
        const ModCtx<uint32_t> ctx(p);
        for (Layout lb : {Layout::RowMajor, Layout::ColMajor}) {
            gemm_case(ctx, rng, 13, 129, 23, lb, Layout::RowMajor, GemmOp::Add, 1, true, Layout::ColMajor);
            gemm_case(ctx, rng, 17, 70, 40, lb, Layout::ColMajor, GemmOp::Sub, 2, false, Layout::ColMajor);
        }
    }
    for (uint32_t p : {4093u, 4096u})
        for (uint32_t k : {127u, 129u, 257u}) {
            const ModCtx<uint32_t> ctx(p);
            gemm_case(ctx, rng, 9, k, 21, Layout::ColMajor, Layout::RowMajor, GemmOp::Add, 1, false);
            gemm_case(ctx, rng, 16, k, 33, Layout::ColMajor, Layout::ColMajor, GemmOp::Sub, 1, true);
        }
    for (uint32_t p : {1518500249u, 2147483647u, 4294967291u}) {
        const ModCtx<uint32_t> ctx(p);
        for (uint32_t k = 3; k <= 9; ++k) {
            gemm_case(ctx, rng, 8, k, 16, Layout::RowMajor, Layout::RowMajor, GemmOp::Add, 1, false);
            gemm_case(ctx, rng, 5, k, 19, Layout::ColMajor, Layout::RowMajor, GemmOp::Sub, 1, true, Layout::ColMajor);
        }
    }
}

// (A) the Auto selection and the VNNI exclusion of the forced AVX2 path.
void arm_auto() {
    PathGuard guard;
    engine::kernels::set_kernel_path(KernelPath::Auto);
    const KernelPath want = engine::kernels::avx512_kernel_available() ? KernelPath::Avx512
                          : engine::kernels::avx2_kernel_available() ? KernelPath::Avx2 : KernelPath::Portable;
    const char* env = std::getenv("ENGINE_KERNEL");
    if (env == nullptr || env[0] == '\0') {
        check(engine::kernels::active_kernel_path() == want,
              std::string("Auto resolved to ") + engine::kernels::kernel_path_name(engine::kernels::active_kernel_path()) +
              ", expected " + engine::kernels::kernel_path_name(want));
    } else {
        std::cout << "  (A) ENGINE_KERNEL=" << env << " set: Auto follows it, resolved "
                  << engine::kernels::kernel_path_name(engine::kernels::active_kernel_path()) << std::endl;
    }
    if (engine::kernels::avx2_kernel_available()) {
        engine::kernels::set_kernel_path(KernelPath::Avx2);
        engine::kernels::set_vnni_path(VnniPath::Auto);
        check(!engine::kernels::vnni_route_taken(271, 512, 512, 512) && !engine::kernels::vnni_route_taken(101, 512, 512, 512),
              "a forced AVX2 path took a VNNI route under Auto");
        // (S) the pair route is taken under Avx2 + Auto for p <= 4096 and a product above the size threshold, not above 4096
        engine::kernels::set_avx2_pair_path(Avx2PairPath::Auto);
        check(engine::kernels::avx2_pair_route_taken(349, 64, 64, 64) && engine::kernels::avx2_pair_route_taken(4096, 64, 64, 64) &&
              !engine::kernels::avx2_pair_route_taken(4099, 64, 64, 64) && !engine::kernels::avx2_pair_route_taken(349, 2, 64, 64),
              "AVX2 pair route selection (p <= 4096, m >= 4)");
        Rng rng(0x5EEDULL);
        const ModCtx<uint32_t> ctx(349u);
        DenseBuf<uint32_t> A = make_buf(rng, 349, 64, 64, Layout::RowMajor, 0, 0);
        DenseBuf<uint32_t> B = make_buf(rng, 349, 64, 64, Layout::RowMajor, 0, 0);
        DenseBuf<uint32_t> C(64, 64);
        const uint64_t r0 = engine::kernels::avx2_pair_routed_calls();
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(C), GemmOp::Set);
        check(engine::kernels::avx2_pair_routed_calls() == r0 + 1, "the AVX2 pair route was not taken (counter did not move)");
        // (R) an entry >= p behind a Canonical tag is refused on the pair route (A side and B side)
        for (int side = 0; side < 2; ++side) {
            DenseBuf<uint32_t> G = make_buf(rng, 349, 64, 64, Layout::RowMajor, 0, 0);
            DenseBuf<uint32_t> H = make_buf(rng, 349, 64, 64, Layout::RowMajor, 0, 0);
            (side == 0 ? G : H).at(9, 3) = 349;
            bool threw = false;
            try {
                engine::kernels::gemm<uint32_t>(ctx, engine::cview(G), engine::cview(H), engine::view(C), GemmOp::Set);
            } catch (const std::logic_error&) { threw = true; }
            check(threw, std::string("(R) a Canonical-tagged entry p at p = 349 (") + (side == 0 ? "A" : "B") +
                         ") was not refused on the AVX2 pair route");
        }
    }
}

} // namespace

uint32_t KernelAvx2BitIdentityValidation() {
    std::cout << "--- KernelAvx2BitIdentityValidation (slot-24 AVX2K: the AVX2 u32 GEMM microkernel vs portable vs AVX-512"
                 " vs RefLinAlg) ---" << std::endl;
    g_fail = 0; g_checks = 0;
    const bool a2 = engine::kernels::avx2_kernel_available(), a5 = engine::kernels::avx512_kernel_available();
    std::cout << "  avx2_kernel_available=" << (a2 ? 1 : 0) << " avx512_kernel_available=" << (a5 ? 1 : 0) << std::endl;
    if (!a2) std::cout << "  [SKIP arm] no AVX2 on this CPU / build: the portable arm alone is compared to the oracle" << std::endl;
    if (!a5) std::cout << "  [SKIP arm] no AVX-512 on this CPU / build: the AVX-512 arm is not run" << std::endl;
    Rng rng(0xA2A2B17ULL);
    try {
        arm_lanes(rng);
        arm_kblock(rng);
        arm_layouts(rng);
        arm_audit_f3(rng);
        arm_auto();
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    const bool ok = g_fail == 0;
    std::cout << "AVX2K_GEMM avx2=" << (a2 ? 1 : 0) << " avx512=" << (a5 ? 1 : 0) << " checks=" << g_checks
              << " failing=" << g_fail << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
