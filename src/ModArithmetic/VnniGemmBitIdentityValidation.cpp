// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "VnniGemmBitIdentityValidation.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DenseBuf.h"
#include "DenseElim.h"
#include "DenseView.h"
#include "FlatMatrix.h"
#include "Kernels.h"
#include "ModCtx.h"
#include "RefLinAlg.h"
#include "../Engine/EngineTestUtil.h"

// See VnniGemmBitIdentityValidation.h.

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

struct Rng : engine::tape::test::SplitMix64 {
    explicit Rng(uint64_t seed) : SplitMix64(seed) {}
    uint32_t below(uint32_t m) { return m <= 1 ? 0 : static_cast<uint32_t>(next() % m); }
};

uint32_t g_fail = 0, g_checks = 0;
void check(bool ok, const std::string& what) { engine::tape::test::bit_check(g_checks, g_fail, ok, what); }
using engine::tape::test::str;
using engine::tape::test::opname;
using engine::tape::test::opint;

// Restores the process-wide kernel / VNNI settings on scope exit (also on a throw).
struct PathGuard {
    KernelPath kp = engine::kernels::kernel_path();
    VnniPath vp = engine::kernels::vnni_path();
    ~PathGuard() { engine::kernels::set_kernel_path(kp); engine::kernels::set_vnni_path(vp); }
};

// fill = 0: uniform residues; fill = 1: every entry p - 1 (the overflow worst case); fill >= 2: every entry fill - 2
// (slot-24 G4: chosen limb patterns for the two-limb lane bound).
DenseBuf<uint32_t> make_buf(Rng& rng, uint32_t p, uint32_t rows, uint32_t cols, Layout layout,
                            uint32_t pad, int fill) {
    const uint32_t minor = layout == Layout::RowMajor ? cols : rows;
    DenseBuf<uint32_t> b(rows, cols, layout, minor + pad, Invariant::Canonical, Structure::Generic);
    uint32_t* d = b.data();
    for (size_t i = 0; i < b.footprint(); ++i) d[i] = static_cast<uint32_t>(rng.next());   // padding garbage
    for (uint32_t i = 0; i < rows; ++i)
        for (uint32_t j = 0; j < cols; ++j) b.at(i, j) = fill == 1 ? p - 1 : fill >= 2 ? static_cast<uint32_t>(fill - 2) : rng.below(p);
    return b;
}

// The arms one product is computed on: (kernel path, vnni path, label).
struct Arm { KernelPath kp; VnniPath vp; const char* name; };
std::vector<Arm> arms() {
    std::vector<Arm> a = {{KernelPath::Portable, VnniPath::Off, "portable"}};
    // (slot-24 AVX2K) the AVX2 arm: under Auto it never routes to VNNI (the Zen 3 fallback arm)
    if (engine::kernels::avx2_kernel_available()) a.push_back({KernelPath::Avx2, VnniPath::Auto, "u32-avx2"});
    if (engine::kernels::avx512_kernel_available()) {
        a.push_back({KernelPath::Avx512, VnniPath::Off, "u32-avx512"});
        a.push_back({KernelPath::Auto, VnniPath::Auto, "auto"});
    }
    if (engine::kernels::vnni_kernel_available()) a.push_back({KernelPath::Auto, VnniPath::Force, "vnni-force"});
    return a;
}

ModCtx<uint32_t> make_ctx(uint32_t ell, uint32_t k) {
    return k == 1 ? ModCtx<uint32_t>(ell) : ModCtx<uint32_t>::prime_power(ell, k);
}

// One gemm case on every arm against the RefLinAlg oracle (C's padding must stay untouched).
void gemm_case(const ModCtx<uint32_t>& ctx, Rng& rng, uint32_t m, uint32_t k, uint32_t n,
               Layout la, Layout lb, Layout lc, bool tA, bool tB, GemmOp op, uint32_t pad, int fill,
               bool window, int fillB = -1) {
    if (fillB < 0) fillB = fill;
    const uint32_t p = ctx.p;
    // window: operate on an interior sub-view of larger buffers (offsets 3 / 5).
    const uint32_t o = window ? 3 : 0, q = window ? 5 : 0;
    DenseBuf<uint32_t> Ab = tA ? make_buf(rng, p, k + o, m + q, la, pad, fill) : make_buf(rng, p, m + o, k + q, la, pad, fill);
    DenseBuf<uint32_t> Bb = tB ? make_buf(rng, p, n + o, k + q, lb, pad, fillB) : make_buf(rng, p, k + o, n + q, lb, pad, fillB);
    DenseBuf<uint32_t> C0 = make_buf(rng, p, m + o, n + q, lc, pad, fill);
    DenseView<const uint32_t> A = tA ? DenseView<const uint32_t>(Ab).sub(o, k, q, m).transposed_view()
                                     : DenseView<const uint32_t>(Ab).sub(o, m, q, k);
    DenseView<const uint32_t> B = tB ? DenseView<const uint32_t>(Bb).sub(o, n, q, k).transposed_view()
                                     : DenseView<const uint32_t>(Bb).sub(o, k, q, n);
    const std::string tag = "gemm p=" + str(p) + " m=" + str(m) + " k=" + str(k) + " n=" + str(n) +
                            " A" + (la == Layout::RowMajor ? "R" : "C") + (tA ? "t" : "") +
                            " B" + (lb == Layout::RowMajor ? "R" : "C") + (tB ? "t" : "") +
                            " C" + (lc == Layout::RowMajor ? "R" : "C") + " " + opname(op) +
                            " pad=" + str(pad) + (fill == 1 ? " worst" : fill >= 2 ? " a=" + str(fill - 2) + " b=" + str(fillB - 2) : "") +
                            (window ? " window" : "");

    DenseBuf<uint32_t> Cref = C0;
    engine::ref::gemm<uint32_t>(p, A, B, DenseView<uint32_t>(Cref).sub(o, m, q, n), opint(op));

    PathGuard guard;
    for (const Arm& arm : arms()) {
        engine::kernels::set_kernel_path(arm.kp);
        engine::kernels::set_vnni_path(arm.vp);
        DenseBuf<uint32_t> C = C0;   // keeps the padding garbage and the frame outside the window
        engine::kernels::gemm<uint32_t>(ctx, A, B, DenseView<uint32_t>(C).sub(o, m, q, n), op);
        bool same = true;   // whole buffer, padding and frame included
        for (size_t i = 0; i < C.footprint() && same; ++i) same = C.data()[i] == Cref.data()[i];
        check(same, tag + " [" + arm.name + "] != oracle (or wrote outside the window)");
    }
}

void arm_gemm(Rng& rng) {
    const struct { uint32_t ell, k; } mods[] = {{2, 1}, {3, 1}, {5, 1}, {7, 1}, {13, 1}, {37, 1}, {101, 1}, {103, 1},
                                               {127, 1}, {2, 2}, {3, 2}, {11, 2}, {5, 3}, {2, 7}};
    const Layout L[2] = {Layout::RowMajor, Layout::ColMajor};
    const GemmOp ops[3] = {GemmOp::Set, GemmOp::Add, GemmOp::Sub};
    uint32_t cases = 0;
    for (const auto& md : mods) {
        const ModCtx<uint32_t> ctx = make_ctx(md.ell, md.k);
        // the tile / block edges: 8 x 32 tiles, MC = 128 rows, k-quads of 4, VNNI_KC = 1024
        const uint32_t shapes[][3] = {{8, 16, 32}, {9, 17, 33}, {31, 63, 65}, {64, 64, 64}, {129, 130, 131},
                                      {17, 1029, 40}, {40, 2051, 37}, {1, 300, 300}, {300, 3, 300}, {7, 64, 31}};
        for (const auto& s : shapes) {
            for (int v = 0; v < 4; ++v) {   // a rotating layout / op / view selection per shape
                const uint32_t sel = static_cast<uint32_t>(rng.next());
                gemm_case(ctx, rng, s[0], s[1], s[2], L[sel & 1], L[(sel >> 1) & 1], L[(sel >> 2) & 1],
                          (sel >> 3) & 1, (sel >> 4) & 1, ops[(sel >> 5) % 3], (sel >> 7) & 1 ? 3 : 0, 0,
                          (sel >> 8) & 1);
                ++cases;
            }
        }
    }
    // every layout / transposition / op combination once at p = 127 on a ragged shape
    const ModCtx<uint32_t> c127(127);
    for (int la = 0; la < 2; ++la) for (int lb = 0; lb < 2; ++lb) for (int lc = 0; lc < 2; ++lc)
        for (int ta = 0; ta < 2; ++ta) for (int tb = 0; tb < 2; ++tb) for (const GemmOp op : ops) {
            gemm_case(c127, rng, 41, 77, 70, L[la], L[lb], L[lc], ta, tb, op, 1, 0, (la ^ tb) & 1);
            ++cases;
        }
    std::cout << "  (G) gemm cases=" << cases << std::endl;
}

// (G2) (slot-24 G4) the two-limb route, 128 < p <= 2^14: the plan's primes {131, 547, 691, 997, 16381}, the boundary
// primes just above 127 (131, 137) and just below 2^14 (16381), census primes 503 / 1009, 8191, and the composite
// moduli 3^5, 2^8, 5^6, 127^2 and 2^14 itself (entries up to 16383 = two full 7-bit limbs). Shapes across the 8 x 32
// tile, the 128-row block, the 512-term k-block (513, 1029, 1539) and the 1024-column B block (1025, 2100).
void arm_gemm2(Rng& rng) {
    const struct { uint32_t ell, k; } mods[] = {{131, 1}, {137, 1}, {503, 1}, {547, 1}, {691, 1}, {997, 1}, {1009, 1},
                                               {8191, 1}, {16381, 1}, {3, 5}, {2, 8}, {5, 6}, {127, 2}, {2, 14}};
    const Layout L[2] = {Layout::RowMajor, Layout::ColMajor};
    const GemmOp ops[3] = {GemmOp::Set, GemmOp::Add, GemmOp::Sub};
    uint32_t cases = 0;
    const uint64_t r0 = engine::kernels::vnni2_routed_calls();
    for (const auto& md : mods) {
        const ModCtx<uint32_t> ctx = make_ctx(md.ell, md.k);
        check(engine::kernels::vnni_limbs_for(ctx.p) == 2, "(G2) p=" + str(ctx.p) + " is not a two-limb modulus");
        const uint32_t shapes[][3] = {{8, 16, 32}, {9, 17, 33}, {31, 63, 65}, {64, 64, 64}, {129, 130, 131},
                                      {17, 513, 40}, {23, 1029, 37}, {40, 1539, 33}, {1, 300, 300}, {300, 3, 300},
                                      {7, 64, 31}, {12, 40, 1025}};
        for (const auto& sh : shapes) {
            for (int v = 0; v < 3; ++v) {
                const uint32_t sel = static_cast<uint32_t>(rng.next());
                gemm_case(ctx, rng, sh[0], sh[1], sh[2], L[sel & 1], L[(sel >> 1) & 1], L[(sel >> 2) & 1],
                          (sel >> 3) & 1, (sel >> 4) & 1, ops[(sel >> 5) % 3], (sel >> 7) & 1 ? 3 : 0, 0,
                          (sel >> 8) & 1);
                ++cases;
            }
        }
    }
    // every layout / transposition / op combination once at p = 691 and at p = 16381 on a ragged shape
    for (const uint32_t ell : {691u, 16381u}) {
        const ModCtx<uint32_t> c(ell);
        for (int la = 0; la < 2; ++la) for (int lb = 0; lb < 2; ++lb) for (int lc = 0; lc < 2; ++lc)
            for (int ta = 0; ta < 2; ++ta) for (int tb = 0; tb < 2; ++tb) for (const GemmOp op : ops) {
                gemm_case(c, rng, 41, 77, 70, L[la], L[lb], L[lc], ta, tb, op, 1, 0, (la ^ tb) & 1);
                ++cases;
            }
    }
    // n > VNNI2_NC = 1024 (the second / third B block), with m > 128 and k > 512 at once
    const struct { uint32_t ell, m, k, n; Layout la, lb, lc; bool tB; GemmOp op; bool window; } wide[] = {
        {691, 9, 20, 1025, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, false, GemmOp::Set, false},
        {16381, 17, 5, 2100, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor, false, GemmOp::Sub, false},
        {997, 130, 530, 2100, Layout::ColMajor, Layout::RowMajor, Layout::RowMajor, true, GemmOp::Add, true},
        {131, 130, 33, 3075, Layout::RowMajor, Layout::ColMajor, Layout::ColMajor, true, GemmOp::Sub, true},
    };
    for (const auto& c : wide) {
        gemm_case(ModCtx<uint32_t>(c.ell), rng, c.m, c.k, c.n, c.la, c.lb, c.lc, false, c.tB, c.op, 1, 0, c.window);
        ++cases;
    }
    const uint64_t routed = engine::kernels::vnni2_routed_calls() - r0;
    if (engine::kernels::vnni_kernel_available())   // the vnni-force arm routes every case through the two-limb route
        check(routed >= cases, "(G2) two-limb cases not all routed (" + str(routed) + " < " + str(cases) + ")");
    std::cout << "  (G2) two-limb gemm cases=" << cases << " routed2=" << routed << std::endl;
}

// (B) (slot-24 G4 audit F6) a sampled sweep of the packer's b~ = 128 b mod p: every 64th modulus in (128, 2^14]
// plus 129 .. 160 and 16353 .. 16384 (composites included, they are valid moduli), each with B = the 1 x p row of
// every residue b < p and A = an 8 x 1 column of limb patterns {128 (isolates b~), p-1, 1, 127, 129, (p-1)/2, 255,
// p-128} (mod p), forced through the two-limb route; every cell against (a b) mod p in u64.
void arm_btilde(Rng&) {
    if (!engine::kernels::vnni_kernel_available()) {
        std::cout << "  (B) SKIP the b~ sweep: no avx512_vnni on this CPU" << std::endl;
        return;
    }
    PathGuard guard;
    engine::kernels::set_kernel_path(KernelPath::Auto);
    engine::kernels::set_vnni_path(VnniPath::Force);
    std::vector<uint32_t> mods;
    for (uint32_t p = 129; p <= 160; ++p) mods.push_back(p);
    for (uint32_t p = 192; p < 16353; p += 64) mods.push_back(p);
    for (uint32_t p = 16353; p <= 16384; ++p) mods.push_back(p);
    const uint64_t r0 = engine::kernels::vnni2_routed_calls();
    uint32_t bad_mods = 0;
    for (const uint32_t p : mods) {
        const ModCtx<uint32_t> ctx(p);   // init() accepts any modulus >= 2; the check below is independent of ctx
        const uint32_t av[8] = {128u % p, p - 1, 1, 127, 129u % p, (p - 1) / 2, 255u % p, p - 128};
        DenseBuf<uint32_t> A(8, 1, Layout::RowMajor), B(1, p, Layout::RowMajor), C(8, p, Layout::RowMajor);
        for (uint32_t i = 0; i < 8; ++i) A.at(i, 0) = av[i];
        for (uint32_t b = 0; b < p; ++b) B.at(0, b) = b;
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(C), GemmOp::Set);
        bool same = true;
        for (uint32_t i = 0; i < 8 && same; ++i)
            for (uint32_t b = 0; b < p && same; ++b)
                same = C.at(i, b) == static_cast<uint32_t>((static_cast<uint64_t>(av[i]) * b) % p);
        if (!same) ++bad_mods;
        check(same, "(B) b~ sweep p=" + str(p) + ": a two-limb product differs from (a b) mod p");
    }
    const uint64_t routed = engine::kernels::vnni2_routed_calls() - r0;
    check(routed == mods.size(), "(B) b~ sweep: " + str(routed) + " of " + str(mods.size()) + " moduli routed two-limb");
    std::cout << "  (B) b~ sweep moduli=" << mods.size() << " routed2=" << routed << " bad=" << bad_mods << std::endl;
}

// (N) (audit R1) n > VNNI_NC = 2048: the second and later iterations of the outer j0 loop (the packed-B base
// B + k0*brs + j0*bcs, the C column offset j0 + jp*32), alone and with m > VNNI_MC = 128 at once. Every arm runs,
// vnni-force included (k = 5 is below the Auto threshold, so only Force routes that one).
void arm_wide(Rng& rng) {
    const struct { uint32_t ell, m, k, n; Layout la, lb, lc; bool tB; GemmOp op; bool window; } cs[] = {
        {13, 9, 20, 2051, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, false, GemmOp::Set, false},
        {127, 17, 5, 4100, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor, false, GemmOp::Sub, false},
        {101, 130, 33, 2100, Layout::ColMajor, Layout::RowMajor, Layout::RowMajor, true, GemmOp::Add, true},
        {7, 130, 33, 2100, Layout::RowMajor, Layout::ColMajor, Layout::ColMajor, true, GemmOp::Sub, true},
    };
    uint32_t cases = 0;
    const uint64_t r0 = engine::kernels::vnni_routed_calls();
    for (const auto& c : cs) {
        gemm_case(ModCtx<uint32_t>(c.ell), rng, c.m, c.k, c.n, c.la, c.lb, c.lc, false, c.tB, c.op, 1, 0, c.window);
        ++cases;
    }
    const uint64_t routed = engine::kernels::vnni_routed_calls() - r0;
    if (engine::kernels::vnni_kernel_available())   // the vnni-force arm routes every case
        check(routed >= cases, "(N) the n > 2048 shapes were not all routed (" + str(routed) + " < " + str(cases) + ")");
    std::cout << "  (N) n > 2048 cases=" << cases << " routed=" << routed << std::endl;
}

void arm_worst(Rng& rng) {
    // every entry p - 1: the largest exact i32 lane sum, across 1, 2 and 3 k-blocks (+3 ragged)
    const uint32_t ells[] = {127, 2, 101};
    const uint32_t ks[] = {1024, 1027, 2048, 3075};
    uint32_t cases = 0;
    for (uint32_t ell : ells)
        for (uint32_t k : ks)
            for (const GemmOp op : {GemmOp::Set, GemmOp::Sub}) {
                gemm_case(ModCtx<uint32_t>(ell), rng, 17, k, 45, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor,
                          false, false, op, 0, 1, false);
                ++cases;
            }
    // (slot-24 G4) the two-limb lane bound: every entry p - 1 across 1, 2 and 3 k-blocks of 512 (+3 ragged).
    // At p = 2^14 the entry 16383 has both limbs = 127 and b~ = 128 * 16383 mod 2^14 = 16256: a lane sum of
    // 512 * 127 * (16383 + 16256) = 2122318336 < 2^31 per full block.
    const struct { uint32_t ell, k; } m2[] = {{2, 14}, {16381, 1}, {131, 1}, {997, 1}};
    const uint32_t ks2[] = {512, 515, 1024, 1539};
    for (const auto& md : m2)
        for (uint32_t k : ks2)
            for (const GemmOp op : {GemmOp::Set, GemmOp::Sub}) {
                gemm_case(make_ctx(md.ell, md.k), rng, 17, k, 45, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor,
                          false, false, op, 0, 1, false);
                ++cases;
            }
    // The largest lane at p = 16381: A = 16255 (limbs 127, 126), B = 16380 (b~ = 16253), the maximiser of
    // 127 b + 126 b~ over b < 16381: 512 * 4128138 = 2113606656 per full block.
    for (uint32_t k : ks2) {
        gemm_case(ModCtx<uint32_t>(16381), rng, 17, k, 45, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor,
                  false, false, GemmOp::Sub, 0, 2 + 16255, false, 2 + 16380);
        ++cases;
    }
    std::cout << "  (W) worst-case cases=" << cases << std::endl;
}

// rref + FlatMatrix product / rank / RREF: Off vs Auto vs Force must agree exactly.
void arm_elim(Rng& rng) {
    const uint32_t ells[] = {5, 13, 127, 131, 691, 16381};   // (slot-24 G4) + the two-limb route
    const uint32_t shapes[][2] = {{300, 410}, {517, 263}, {96, 1100}};
    std::vector<VnniPath> vps = {VnniPath::Off, VnniPath::Auto};
    if (engine::kernels::vnni_kernel_available()) vps.push_back(VnniPath::Force);
    PathGuard guard;
    uint32_t cases = 0;
    for (uint32_t ell : ells) {
        const ModCtx<uint32_t> ctx(ell);
        for (const auto& s : shapes) {
            // rank-deficient: (rows x r) * (r x cols), r below both dims
            const uint32_t r = std::min(s[0], s[1]) * 3 / 4;
            std::vector<uint32_t> X(static_cast<size_t>(s[0]) * r), Y(static_cast<size_t>(r) * s[1]);
            for (auto& x : X) x = rng.below(ell);
            for (auto& y : Y) y = rng.below(ell);
            const FlatMatrix FX(s[0], r, ell, X), FY(r, s[1], ell, Y);

            std::vector<uint32_t> ref_rref, ref_piv, ref_prod, ref_frref;
            uint32_t ref_rank = 0, ref_frank = 0;
            for (size_t v = 0; v < vps.size(); ++v) {
                engine::kernels::set_vnni_path(vps[v]);
                const FlatMatrix P = FX * FY;   // FlatMatrix::operator* -> engine gemm
                P.populate_flat_data();
                const std::vector<uint32_t> prod = P.get_flat_data_unsafe();
                const uint32_t frank = P.rank();
                const std::vector<uint32_t> frref = P.get_row_echelon_form_data();

                DenseBuf<uint32_t> M(s[0], s[1], Layout::RowMajor);
                for (uint32_t i = 0; i < s[0]; ++i)
                    for (uint32_t j = 0; j < s[1]; ++j) M.at(i, j) = prod[static_cast<size_t>(i) * s[1] + j];
                const engine::elim::RrefResult rr = engine::elim::rref(ctx, engine::view(M));
                std::vector<uint32_t> rref(M.data(), M.data() + static_cast<size_t>(s[0]) * s[1]);
                if (v == 0) {
                    ref_prod = prod; ref_frank = frank; ref_frref = frref; ref_rank = rr.rank; ref_piv = rr.pivots;
                    ref_rref = rref;
                    // the Off arm against the oracle product and the oracle RREF
                    DenseBuf<uint32_t> O(s[0], s[1], Layout::RowMajor);
                    DenseBuf<uint32_t> XA(s[0], r, Layout::RowMajor), YB(r, s[1], Layout::RowMajor);
                    std::copy(X.begin(), X.end(), XA.data());
                    std::copy(Y.begin(), Y.end(), YB.data());
                    engine::ref::gemm<uint32_t>(ell, engine::cview(XA), engine::cview(YB), engine::view(O), 0);
                    check(std::equal(prod.begin(), prod.end(), O.data()), "elim p=" + str(ell) + " Off product != oracle");
                    const std::vector<uint32_t> opiv = engine::ref::rref<uint32_t>(ell, engine::view(O));
                    check(opiv == rr.pivots && std::equal(rref.begin(), rref.end(), O.data()),
                          "elim p=" + str(ell) + " Off rref != oracle");
                } else {
                    const std::string tag = "elim p=" + str(ell) + " " + str(s[0]) + "x" + str(s[1]) + " vnni=" + str(int(vps[v]));
                    check(prod == ref_prod, tag + " FlatMatrix product differs from Off");
                    check(frank == ref_frank && frref == ref_frref, tag + " FlatMatrix rank / RREF differs from Off");
                    check(rr.rank == ref_rank && rr.pivots == ref_piv && rref == ref_rref, tag + " elim::rref differs from Off");
                }
            }
            ++cases;
        }
    }
    std::cout << "  (E) elimination cases=" << cases << " x arms=" << vps.size() << std::endl;
}

void arm_refusals_and_route(Rng& rng) {
    PathGuard guard;
    // (AVX2K audit F1) the VNNI routes hang off the AVX-512 path: pin it (when the CPU has it) so that the route expectations below
    // hold under every ENGINE_KERNEL setting (Auto follows the env since AVX2K; ENGINE_KERNEL=avx2|portable would take no VNNI route)
    if (engine::kernels::avx512_kernel_available()) engine::kernels::set_kernel_path(KernelPath::Avx512);
    std::cout << "  (R)/(S) kernel path pinned to " << engine::kernels::kernel_path_name(engine::kernels::active_kernel_path())
              << " (ENGINE_KERNEL=" << (std::getenv("ENGINE_KERNEL") ? std::getenv("ENGINE_KERNEL") : "unset") << ")" << std::endl;
    // Force without VNNI must throw; with VNNI it must not.
    bool threw = false;
    try { engine::kernels::set_vnni_path(VnniPath::Force); } catch (const std::logic_error&) { threw = true; }
    check(threw == !engine::kernels::vnni_kernel_available(), "(R) set_vnni_path(Force) refusal does not match availability");
    engine::kernels::set_vnni_path(VnniPath::Auto);

    // p > 2^14 never routes; Off never routes; the route predicate is a pure function of (p, shape, setting).
    // (slot-24 G4) 128 < p <= 2^14 routes (two-limb) exactly when VNNI is available.
    const bool av = engine::kernels::vnni_kernel_available();
    // (slot-25 PERF-VF (a)) the two-limb work-per-thread floor: m n k >= vnni2_work_floor() * vnni_route_threads()
    // (PERF-VF audit N5) every two-limb Auto expectation is guarded by passes(...), so an extreme ENGINE_VNNI2_FLOOR
    // (a calibration knob) moves the expectation with the route instead of failing the check.
    const uint32_t T = engine::kernels::vnni_route_threads();
    auto passes = [](uint32_t m, uint32_t n, uint32_t k, uint32_t t) { return engine::kernels::vnni2_work_passes(m, n, k, t); };
    check(engine::kernels::vnni_route_taken(131, 512, 512, 512) == (av && passes(512, 512, 512, T)),
          "(R) p = 131 two-limb route does not match availability");
    check(engine::kernels::vnni_route_taken(16381, 512, 512, 512) == (av && passes(512, 512, 512, T)),
          "(R) p = 16381 two-limb route does not match availability");
    check(engine::kernels::vnni_route_taken(16384, 512, 512, 512) == (av && passes(512, 512, 512, T)),
          "(R) p = 2^14 two-limb route does not match availability");
    check(!engine::kernels::vnni_route_taken(16411, 512, 512, 512), "(R) p = 16411 > 2^14 routed");
    check(!engine::kernels::vnni_route_taken(134217757, 512, 512, 512), "(R) p = 134217757 routed");
    check(engine::kernels::vnni_limbs_for(127) == 1 && engine::kernels::vnni_limbs_for(128) == 1 &&
          engine::kernels::vnni_limbs_for(129) == 2 && engine::kernels::vnni_limbs_for(16384) == 2 &&
          engine::kernels::vnni_limbs_for(16385) == 0, "(R) vnni_limbs_for boundaries");
    // (G4 audit F1) Auto's large-p shape threshold: from p = 2^13 the two-limb route needs m, n >= 128
    check(std::getenv("ENGINE_VNNI2_FLOOR") != nullptr ||
              engine::kernels::vnni2_work_floor() == engine::kernels::VNNI2_WORK_FLOOR_DEFAULT,
          "(R) the two-limb work floor is not the default with ENGINE_VNNI2_FLOOR unset");
    // (S26 LEVERS-26) the default floor is 2^20; the 2^18 block still runs under ENGINE_VNNI2_FLOOR=262144.
    if (engine::kernels::vnni2_work_floor() == (uint64_t(1) << 20)) {
        check(passes(128, 128, 64, 1) && !passes(127, 128, 64, 1), "(R) floor 2^20 at 1 thread: the 128 x 128 x 64 boundary");
        check(passes(256, 256, 256, 16) && !passes(255, 256, 256, 16), "(R) floor 2^20 at 16 threads: the 2^24 boundary");
        check(passes(256, 256, 512, 32) && !passes(256, 255, 512, 32), "(R) floor 2^20 at 32 threads: the 2^25 boundary");
    }
    if (engine::kernels::vnni2_work_floor() == (uint64_t(1) << 18)) {
        check(passes(64, 64, 64, 1) && !passes(63, 64, 64, 1), "(R) floor 2^18 at 1 thread: the 64^3 boundary");
        check(passes(256, 256, 64, 16) && !passes(255, 256, 64, 16), "(R) floor 2^18 at 16 threads: the 2^22 boundary");
        check(passes(256, 256, 128, 32) && !passes(256, 255, 128, 32), "(R) floor 2^18 at 32 threads: the 2^23 boundary");
    }
    // (PERF-VF audit S1) floor * threads saturates: a floor > 0 never lets a 1 x 1 x 1 product pass at 32 threads
    // (a wrapping product, e.g. floor 2^62 x 32 = 0 mod 2^64, would).
    check(engine::kernels::vnni2_work_floor() == 0 || !passes(1, 1, 1, 32),
          "(R) floor * threads wrapped: a 1 x 1 x 1 product passes the work floor at 32 threads");
    check(T >= 1 && T <= 32, "(R) vnni_route_threads() outside the pool's 1..32");
    check(engine::kernels::vnni_route_taken(691, 64, 64, 64) == (av && passes(64, 64, 64, T)),
          "(R) Auto p = 691, 64^3 does not follow the work floor");
    check(engine::kernels::vnni_route_taken(691, 256, 256, 128) == (av && passes(256, 256, 128, T)),
          "(R) Auto p = 691, 256 x 128 x 256 (2^23) not routed");
    check(engine::kernels::vnni_route_taken(13, 64, 64, 8 * 2) == av, "(R) the one-limb route is not under the two-limb floor");
    check(engine::kernels::vnni_route_taken(8191, 64, 64, 64) == (av && passes(64, 64, 64, T)),
          "(R) Auto p = 8191, 64^3 does not match availability and the work floor");
    check(!engine::kernels::vnni_route_taken(8209, 64, 64, 64), "(R) Auto p = 8209, 64^3 routed (large-p threshold)");
    // (PERF-VF audit S2) the G4-F1 contrast at any pool size: 64 x 1024 x 1024 = 2^26 >= 2^20 x 32 (S26 LEVERS-26) clears the default
    // floor at every T <= 32, so only the p >= 2^13 => m, n >= 128 threshold separates p = 8191 from p = 8209.
    if (engine::kernels::vnni2_work_floor() == engine::kernels::VNNI2_WORK_FLOOR_DEFAULT)
        check(passes(64, 1024, 1024, T), "(R) 64 x 1024 x 1024 does not clear the default work floor");
    check(engine::kernels::vnni_route_taken(8191, 64, 1024, 1024) == (av && passes(64, 1024, 1024, T)),
          "(R) Auto p = 8191, 64 x 1024 x 1024 does not match availability (below the large-p threshold)");
    check(!engine::kernels::vnni_route_taken(8209, 64, 1024, 1024), "(R) Auto p = 8209, 64 x 1024 x 1024 routed (large-p threshold, m = 64)");
    check(!engine::kernels::vnni_route_taken(16381, 127, 512, 512), "(R) Auto p = 16381, m = 127 routed");
    check(!engine::kernels::vnni_route_taken(16381, 512, 127, 512), "(R) Auto p = 16381, n = 127 routed");
    check(engine::kernels::vnni_route_taken(12289, 128, 128, 16) == (av && passes(128, 128, 16, T)),
          "(R) Auto p = 12289, 128 x 16 x 128 does not match availability and the work floor");
    check(engine::kernels::vnni_route_taken(12289, 256, 256, 128) == (av && passes(256, 256, 128, T)),
          "(R) Auto p = 12289, 256 x 128 x 256 not routed");
    engine::kernels::set_vnni_path(VnniPath::Off);
    check(!engine::kernels::vnni_route_taken(13, 512, 512, 512), "(R) Off routed");
    check(!engine::kernels::vnni_route_taken(691, 512, 512, 512), "(R) Off routed p = 691");
    engine::kernels::set_vnni_path(VnniPath::Auto);
    check(engine::kernels::vnni_route_taken(13, 512, 512, 512) == engine::kernels::vnni_kernel_available(),
          "(R) Auto route at p = 13, 512^3 does not match availability");
    check(!engine::kernels::vnni_route_taken(13, 1, 512, 512) || !engine::kernels::vnni_kernel_available(),
          "(R) Auto routed a 1-row product");

    if (!engine::kernels::vnni_kernel_available()) {
        std::cout << "  (R)/(S) SKIP the routed arms: no avx512_vnni on this CPU (the u32 fallback is what Auto runs)" << std::endl;
        return;
    }
    // (S) the counter moves on a routed call and not on an Off call
    const ModCtx<uint32_t> ctx(13);
    DenseBuf<uint32_t> A = make_buf(rng, 13, 64, 64, Layout::RowMajor, 0, 0), B = make_buf(rng, 13, 64, 64, Layout::RowMajor, 0, 0);
    DenseBuf<uint32_t> C(64, 64, Layout::RowMajor);
    const uint64_t c0 = engine::kernels::vnni_routed_calls();
    engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(C), GemmOp::Set);
    const uint64_t c1 = engine::kernels::vnni_routed_calls();
    engine::kernels::set_vnni_path(VnniPath::Off);
    engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(C), GemmOp::Set);
    const uint64_t c2 = engine::kernels::vnni_routed_calls();
    check(c1 == c0 + 1 && c2 == c1, "(S) routed-call counter: Auto +" + str(c1 - c0) + ", Off +" + str(c2 - c1));
    {   // (slot-24 G4) the two-limb counter moves on a p = 691 routed call only
        // (slot-25 PERF-VF) 256 x 128 x 256 = 2^23 = floor 2^18 x the pool's 32-thread cap: routed at any thread count
        engine::kernels::set_vnni_path(VnniPath::Auto);
        const ModCtx<uint32_t> c691(691);
        DenseBuf<uint32_t> A2 = make_buf(rng, 691, 256, 128, Layout::RowMajor, 0, 0), B2 = make_buf(rng, 691, 128, 256, Layout::RowMajor, 0, 0);
        DenseBuf<uint32_t> C2(256, 256, Layout::RowMajor);
        const uint64_t d0 = engine::kernels::vnni2_routed_calls(), e0 = engine::kernels::vnni_routed_calls();
        engine::kernels::gemm<uint32_t>(c691, engine::cview(A2), engine::cview(B2), engine::view(C2), GemmOp::Set);
        const uint64_t d1 = engine::kernels::vnni2_routed_calls(), e1 = engine::kernels::vnni_routed_calls();
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B), engine::view(C), GemmOp::Set);
        const uint64_t d2 = engine::kernels::vnni2_routed_calls();
        const uint64_t want = passes(256, 256, 128, T) ? 1 : 0;   // (PERF-VF audit N5) 1 at every floor <= 2^18
        check(d1 == d0 + want && e1 == e0 + want && d2 == d1, "(S) two-limb counter: p=691 +" + str(d1 - d0) + ", p=13 +" + str(d2 - d1));
    }

    // (R) an operand entry >= 128 behind a Canonical tag on a routed call is an ENGINE_CHECK refusal
    engine::kernels::set_vnni_path(VnniPath::Force);
    DenseBuf<uint32_t> Bad = make_buf(rng, 127, 64, 64, Layout::RowMajor, 0, 0);
    Bad.at(5, 7) = 200;
    threw = false;
    try {
        engine::kernels::gemm<uint32_t>(ModCtx<uint32_t>(127), engine::cview(A), engine::cview(Bad), engine::view(C), GemmOp::Set);
    } catch (const std::logic_error&) { threw = true; }
    check(threw, "(R) a Canonical-tagged entry 200 at p = 127 was not refused on the VNNI route");
    // (slot-24 G4) an entry >= 2^14 behind a Canonical tag on the two-limb route is refused (A side and B side)
    for (int side = 0; side < 2; ++side) {
        DenseBuf<uint32_t> G = make_buf(rng, 16381, 64, 64, Layout::RowMajor, 0, 0);
        DenseBuf<uint32_t> H = make_buf(rng, 16381, 64, 64, Layout::RowMajor, 0, 0);
        (side == 0 ? G : H).at(9, 3) = 20000;
        threw = false;
        try {
            engine::kernels::gemm<uint32_t>(ModCtx<uint32_t>(16381), engine::cview(G), engine::cview(H), engine::view(C), GemmOp::Set);
        } catch (const std::logic_error&) { threw = true; }
        check(threw, std::string("(R) a Canonical-tagged entry 20000 at p = 16381 (") + (side == 0 ? "A" : "B") +
                     ") was not refused on the two-limb route");
    }
}

} // namespace

uint32_t VnniGemmBitIdentityValidation() {
    std::cout << "--- VnniGemmBitIdentityValidation (slot-23 PERF-P5 + slot-24 G4: the VNNI routes of gemm<uint32_t>, one limb"
                 " p <= 128, two limbs 128 < p <= 2^14) ---" << std::endl;
    g_fail = 0; g_checks = 0;
    const bool vnni = engine::kernels::vnni_kernel_available();
    std::cout << "  vnni_kernel_available=" << (vnni ? 1 : 0) << " avx512_kernel_available="
              << (engine::kernels::avx512_kernel_available() ? 1 : 0) << std::endl;
    const uint64_t routed0 = engine::kernels::vnni_routed_calls(), routed20 = engine::kernels::vnni2_routed_calls();
    Rng rng(0x5032335035ULL);   // fixed seed
    try {
        arm_gemm(rng);
        arm_gemm2(rng);
        arm_btilde(rng);
        arm_wide(rng);
        arm_worst(rng);
        arm_elim(rng);
        arm_refusals_and_route(rng);
    } catch (const std::exception& e) {
        check(false, std::string("unexpected exception: ") + e.what());
    }
    const uint64_t routed = engine::kernels::vnni_routed_calls() - routed0;
    const uint64_t routed2 = engine::kernels::vnni2_routed_calls() - routed20;
    if (vnni) check(routed > routed2 && routed2 > 0, "the one-limb or the two-limb VNNI route was never taken");
    std::cout << "VNNI_GEMM vnni=" << (vnni ? "available" : "unavailable") << " checks=" << g_checks
              << " failing=" << g_fail << " routed=" << routed << " routed2=" << routed2
              << " verdict=" << (g_fail == 0 ? "OK" : "FAIL") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
