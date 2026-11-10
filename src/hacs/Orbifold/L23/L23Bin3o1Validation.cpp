// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "L23Bin3o1Validation.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../../../ModArithmetic/Binary/Ring2k.h"
#include "AmalgamReader.h"

// See L23Bin3o1Validation.h.

namespace {

namespace r2 = engine::ring2k;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}
template<class T> std::string str(T v) { return std::to_string(v); }
uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

std::vector<r2::KernelPath> paths() {
    std::vector<r2::KernelPath> p;
    for (r2::KernelPath k : {r2::KernelPath::Portable, r2::KernelPath::Avx2, r2::KernelPath::Avx512})
        if (r2::kernel_path_available(k)) p.push_back(k);
    return p;
}
struct ScopedPath {
    explicit ScopedPath(r2::KernelPath p) { r2::set_kernel_path(p); }
    ~ScopedPath() { r2::set_kernel_path(r2::KernelPath::Auto); }
};

// (K) kernel_shift_rows against the kernel_axpy sweep, one element type.
template<class T>
uint64_t shift_kernel_case(uint32_t n, size_t w, size_t ld, T f, uint64_t seed) {
    std::vector<T> X(static_cast<size_t>(n + 1) * ld), Y;
    uint64_t s = seed;
    for (auto& x : X) x = static_cast<T>(splitmix(s));
    Y = X;
    r2::kernel_shift_rows(X.data(), ld, n, w, f);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = n; j-- > i;) r2::kernel_axpy(Y.data() + static_cast<size_t>(j) * ld, Y.data() + static_cast<size_t>(j + 1) * ld, f, w);
    uint64_t bad = 0;
    for (size_t i = 0; i < X.size(); ++i) bad += X[i] != Y[i];   // the padding columns [w, ld) must be untouched too
    return bad;
}
template<class T>
void shift_kernel_type(const char* name, uint64_t& cases) {
    const uint32_t lanes = 64 / sizeof(T);   // AVX-512 lanes; the portable / AVX2 ones divide it
    const std::vector<uint32_t> ns = {0, 1, 2, 15, 16, 17, 31, 32, 33, 47, 100};
    const std::vector<size_t> ws = {1, lanes / 4 - 1, lanes / 4, lanes / 2 + 1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3, 100};
    uint64_t seed = 0x5eed0000 + sizeof(T);
    for (r2::KernelPath p : paths()) {
        ScopedPath sp(p);
        for (uint32_t n : ns)
            for (size_t w : ws) {
                if (w == 0) continue;
                for (int fi = 0; fi < 4; ++fi) {
                    const T f = fi == 0 ? T(~T(0)) : fi == 1 ? T(1) : fi == 2 ? T(3) : static_cast<T>(splitmix(seed) | 1u);
                    const size_t ld = (fi & 1) ? w : w + 5;   // contiguous and strided
                    const uint64_t bad = shift_kernel_case<T>(n, w, ld, f, splitmix(seed));
                    ++cases;
                    check(bad == 0, std::string("(K) kernel_shift_rows<") + name + "> path=" + r2::kernel_path_name(p) + " n=" + str(n) +
                                        " w=" + str(w) + " ld=" + str(ld) + " f#" + str(fi) + ": " + str(bad) + " entries differ");
                }
            }
    }
}

bool same_div(const l23::Divisors& a, const l23::Divisors& b) { return a == b; }
bool same_divmap(const std::map<uint32_t, l23::Divisors>& a, const std::map<uint32_t, l23::Divisors>& b) { return a == b; }
std::string diff_lattice(const l23::LatticeCell& a, const l23::LatticeCell& b) {
    std::string d;
    auto f = [&](bool eq, const char* name) { if (!eq) d += std::string(d.empty() ? "" : ",") + name; };
    f(a.ell == b.ell && a.n == b.n && a.d == b.d, "cell");
    f(a.K == b.K && a.m == b.m && a.e_A == b.e_A && a.e_1 == b.e_1, "K/m/e");
    f(a.rung == b.rung && a.reruns == b.reruns, "rung");
    f(a.rank_q_A == b.rank_q_A && a.z == b.z && a.rank_lambda == b.rank_lambda && a.h0 == b.h0, "ranks");
    f(a.torsion == b.torsion, "torsion");
    f(same_divmap(a.ed_tp, b.ed_tp) && same_divmap(a.ed_eis, b.ed_eis), "ed_tp/eis");
    f(same_div(a.ed_tell, b.ed_tell) && same_div(a.ed_tell2, b.ed_tell2), "ed_tell");
    f(a.commute == b.commute, "commute");
    f(a.pm_done == b.pm_done && a.iota_ok == b.iota_ok && a.dim_plus == b.dim_plus && a.dim_minus == b.dim_minus, "pm");
    f(same_divmap(a.ed_tp_plus, b.ed_tp_plus) && same_divmap(a.ed_tp_minus, b.ed_tp_minus) && same_divmap(a.ed_eis_plus, b.ed_eis_plus) &&
          same_divmap(a.ed_eis_minus, b.ed_eis_minus) && same_div(a.ed_tell_plus, b.ed_tell_plus) && same_div(a.ed_tell_minus, b.ed_tell_minus),
      "ed_pm");
    f(a.ok == b.ok && a.failures == b.failures, "ok/failures");
    f(a.log == b.log, "log");
    f(a.ladic2_line() == b.ladic2_line(), "ladic2_line");
    return d;
}
bool same_jordan(const l23::Jordan01& a, const l23::Jordan01& b) {
    return a.dim == b.dim && a.rank == b.rank && a.eig0 == b.eig0 && a.eig1 == b.eig1 && a.other == b.other && a.valid == b.valid;
}
bool same_layers(const l23::LayerCell& a, const l23::LayerCell& b) {
    return a.ell == b.ell && a.n == b.n && a.a_max == b.a_max && a.E == b.E && a.h0 == b.h0 && a.h1 == b.h1 && a.dim_br == b.dim_br &&
           a.dim_sub == b.dim_sub && a.dim_t1 == b.dim_t1 && a.dim_t1_br == b.dim_t1_br && a.dim_cusp == b.dim_cusp &&
           a.dim_cusp_br == b.dim_cusp_br && a.dim_cusp_sub == b.dim_cusp_sub && a.dim_cusp_t1 == b.dim_cusp_t1 &&
           a.br_stable == b.br_stable && a.cusp_stable == b.cusp_stable && a.t1_cocycles == b.t1_cocycles &&
           same_jordan(a.j_whole, b.j_whole) && same_jordan(a.j_br, b.j_br) && same_jordan(a.j_quot, b.j_quot) &&
           same_jordan(a.j_cusp, b.j_cusp) && a.l_done == b.l_done && a.dim_L == b.dim_L && a.dim_L_br == b.dim_L_br &&
           a.L_in_T1 == b.L_in_T1 && a.L_in_sub == b.L_in_sub && a.L_stable == b.L_stable;
}
bool same_mod(const l23::ModCell& a, const l23::ModCell& b) {
    return a.ell == b.ell && a.n == b.n && a.k == b.k && a.h0 == b.h0 && a.length == b.length && a.image_length == b.image_length;
}
std::string mod_str(const l23::ModCell& c) {
    std::string s = "h0=" + str(c.h0) + " len=" + str(c.length) + " img=";
    for (uint32_t x : c.image_length) s += str(x) + ",";
    return s;
}

} // namespace

uint32_t L23Bin3o1Validation() {
    g_fail = 0;
    g_checks = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "L23Bin3o1Validation: the BIN3O1 route of the degree-1 amalgam reader against the old route" << std::endl;
    const l23::Bin3o1Counters c0 = l23::bin3o1_counters();

    // (K) the wavefront Taylor sweep kernel.
    uint64_t kcases = 0;
    shift_kernel_type<uint8_t>("u8", kcases);
    shift_kernel_type<uint16_t>("u16", kcases);
    shift_kernel_type<uint32_t>("u32", kcases);
    shift_kernel_type<uint64_t>("u64", kcases);
    std::cout << "  (K) kernel_shift_rows = the axpy sweep: " << kcases << " cases" << std::endl;

    // (R) the relation matrix and B^1; (S) the structural operators.
    uint64_t rcases = 0, scases = 0;
    const std::vector<uint32_t> rel_n = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 30, 31, 63, 64, 65, 100, 127, 128, 129};
    const std::vector<uint32_t> ks2 = {1, 2, 5, 8, 9, 16, 17, 31, 32, 33, 64, 65, 100, 128, 192, 300};   // R1 R2 R3 R4 R5
    for (uint32_t k : ks2)
        for (uint32_t n : rel_n) {
            if (k > 64 && n > 65) continue;   // the wide rungs: smaller n
            const uint64_t bad = l23::relation_selfcheck(2, n, k);
            ++rcases;
            check(bad == 0, "(R) relation / B^1 ell=2 k=" + str(k) + " n=" + str(n) + ": " + str(bad));
        }
    for (uint32_t k : {1u, 2u, 39u, 40u, 80u})
        for (uint32_t n : {1u, 2u, 6u, 7u, 16u, 17u, 40u, 41u}) {
            const uint64_t bad = l23::relation_selfcheck(3, n, k);
            ++rcases;
            check(bad == 0, "(R) relation / B^1 ell=3 k=" + str(k) + " n=" + str(n) + ": " + str(bad));
        }
    for (r2::KernelPath p : paths()) {
        ScopedPath sp(p);
        uint64_t seed = 0xb1730001;
        for (uint32_t k : {1u, 2u, 8u, 9u, 16u, 17u, 32u, 33u, 64u, 65u, 192u})
            for (uint32_t n : {0u, 1u, 2u, 15u, 16u, 17u, 33u, 70u})
                for (uint32_t r : {1u, 7u, 8u, 9u, 17u, 33u, 64u, 65u}) {
                    if (k > 64 && r > 17) continue;
                    const uint64_t bad = l23::rows_selfcheck(k, n, r, splitmix(seed));
                    ++scases;
                    check(bad == 0, std::string("(S) rows path=") + r2::kernel_path_name(p) + " k=" + str(k) + " n=" + str(n) + " r=" + str(r) +
                                        ": " + str(bad));
                }
    }
    {
        uint64_t seed = 0x7a2;
        for (uint32_t n : {0u, 1u, 2u, 3u, 7u, 8u, 63u, 64u, 65u, 200u, 511u, 512u})
            for (uint32_t r : {1u, 7u, 8u, 9u, 64u, 100u}) {
                const uint64_t bad = l23::taylor2_selfcheck(n, r, splitmix(seed));
                ++scases;
                check(bad == 0, "(S) taylor2 n=" + str(n) + " r=" + str(r) + ": " + str(bad));
            }
    }
    for (uint32_t k : {1u, 2u, 16u, 64u, 192u}) {   // the old dense-Sym^n self-check on the route
        l23::ScopedBin3o1 on(true);
        for (uint32_t n : {5u, 16u, 31u}) {
            const uint32_t bad = l23::rho_selfcheck(2, n, k);
            ++scases;
            check(bad == 0, "(S) rho_selfcheck (route on) ell=2 k=" + str(k) + " n=" + str(n) + ": " + str(bad));
        }
    }
    std::cout << "  (R) relation / B^1: " << rcases << " cases; (S) operators: " << scases << " cases" << std::endl;

    // (H) h1_mod.
    uint64_t hcases = 0;
    const std::vector<std::pair<uint32_t, int64_t>> ops = {{2, 0}, {2, 1}, {3, 4}, {3, 0}, {5, 6}, {7, 8}, {7, 0}};
    auto h1_pair = [&](uint32_t ell, uint32_t n, uint32_t k, uint32_t mutate) {
        l23::ModCell a, b;
        { l23::ScopedBin3o1 off(false); a = l23::h1_mod(ell, n, k, ops, mutate); }
        { l23::ScopedBin3o1 on(true); b = l23::h1_mod(ell, n, k, ops, mutate); }
        ++hcases;
        check(same_mod(a, b), "(H) h1_mod ell=" + str(ell) + " n=" + str(n) + " k=" + str(k) + " mutate=" + str(mutate) + ": off " + mod_str(a) +
                                  " on " + mod_str(b));
        return b;
    };
    for (uint32_t n = 1; n <= 130; ++n) h1_pair(2, n, 1, 0);
    for (uint32_t n : {200u, 255u, 256u, 257u, 400u, 511u, 512u, 513u}) h1_pair(2, n, 1, 0);
    for (uint32_t n : {10u, 24u, 37u, 64u}) h1_pair(2, n, 1, 1);
    for (uint32_t k : {2u, 3u, 8u, 16u, 40u})
        for (uint32_t n : {4u, 9u, 22u, 33u, 64u, 65u}) h1_pair(2, n, k, 0);
    for (uint32_t k : {1u, 2u})
        for (uint32_t n : {6u, 11u, 30u, 64u}) h1_pair(3, n, k, 0);
    // In-run negative control: mutate = 1 (drop r_inf) changes some image length on the route.
    {
        bool moved = false;
        for (uint32_t n : {10u, 22u, 34u, 46u}) {
            l23::ScopedBin3o1 on(true);
            const l23::ModCell a = l23::h1_mod(2, n, 1, ops, 0), b = l23::h1_mod(2, n, 1, ops, 1);
            moved = moved || a.image_length != b.image_length;
        }
        check(moved, "(C) negative control: mutate = 1 moves no h1_mod image on the F_2 route");
    }
    std::cout << "  (H) h1_mod on = off: " << hcases << " cells" << std::endl;

    // (L) read_lattice, h1_layers, oddn_row.
    uint64_t lcases = 0;
    auto lat_pair = [&](uint32_t ell, uint32_t n, uint32_t kmin, uint32_t kcap, uint32_t mt, uint32_t mutate) {
        l23::ReaderOptions o;
        o.ell = ell;
        o.k_min = kmin;
        o.k_cap = kcap;
        o.m_target = mt;
        o.mutate = mutate;
        o.pm = ell == 3;
        l23::LatticeCell a, b;
        { l23::ScopedBin3o1 off(false); a = l23::read_lattice(n, o); }
        { l23::ScopedBin3o1 on(true); b = l23::read_lattice(n, o); }
        ++lcases;
        const std::string d = diff_lattice(a, b);
        check(d.empty(), "(L) read_lattice ell=" + str(ell) + " n=" + str(n) + " K=" + str(kmin) + ".." + str(kcap) + " m_target=" + str(mt) +
                             " mutate=" + str(mutate) + ": differs in " + d);
        return b;
    };
    for (uint32_t n : {2u, 3u, 4u, 8u, 10u, 11u, 12u, 16u, 22u, 32u, 45u, 64u, 98u, 100u, 101u, 128u, 150u, 200u, 201u, 256u})
        lat_pair(2, n, 64, 64, 40, 0);
    for (uint32_t n : {12u, 24u, 60u, 100u, 130u}) lat_pair(2, n, 0, 1024, std::max<uint32_t>((n + 1) / 3 + 10, 40), 0);   // full: R4 / R5
    for (uint32_t n : {12u, 13u, 40u}) {
        lat_pair(2, n, 64, 64, 40, 1);
        lat_pair(2, n, 64, 64, 40, 2);
    }
    for (uint32_t n : {4u, 10u, 11u, 30u, 60u}) lat_pair(3, n, 78, 78, 40, 0);
    lat_pair(3, 12, 78, 78, 40, 1);
    for (uint32_t ell : {2u, 3u})
        for (uint32_t n : {6u, 9u, 20u, 33u}) {
            l23::LayerCell a, b;
            const uint32_t E = ell == 2 ? 4 : 2;
            { l23::ScopedBin3o1 off(false); a = l23::h1_layers(ell, n, 6, E); }
            { l23::ScopedBin3o1 on(true); b = l23::h1_layers(ell, n, 6, E); }
            ++lcases;
            check(same_layers(a, b), "(L) h1_layers ell=" + str(ell) + " n=" + str(n));
        }
    for (uint32_t n : {5u, 9u, 21u, 33u, 63u}) {
        l23::OddnRow a, b;
        { l23::ScopedBin3o1 off(false); a = l23::oddn_row(n, 6); }
        { l23::ScopedBin3o1 on(true); b = l23::oddn_row(n, 6); }
        ++lcases;
        check(a.v == b.v && a.line() == b.line(), "(L) oddn_row n=" + str(n) + ": off " + a.line() + " on " + b.line());
    }
    std::cout << "  (L) read_lattice / h1_layers / oddn_row on = off: " << lcases << " cells" << std::endl;

    // (C) counters, the EFFECTIVE line, the comparator's negative control.
    const l23::Bin3o1Counters c1 = l23::bin3o1_counters();
    const uint64_t split = c1.split_accepted - c0.split_accepted, fallback = c1.split_fallback - c0.split_fallback;
    const uint64_t odd = c1.odd_reduced - c0.odd_reduced, rankq = c1.rankq_blocks - c0.rankq_blocks;
    const uint64_t f2 = c1.f2_cells - c0.f2_cells, reused = c1.hecke_reused - c0.hecke_reused;
    check(split > 0, "(C) the even-n split was never accepted (vacuous)");
    check(fallback == 0, "(C) the even-n split fell back " + str(fallback) + " times (expected never)");
    check(odd > 0, "(C) the odd-n reduction never ran (vacuous)");
    check(rankq > 0, "(C) rank_q_relation never read the blocks (vacuous)");
    check(f2 > 0, "(C) h1_mod never took the F_2 route (vacuous)");
    check(reused > 0, "(C) h1_mod never reused T_p Z (vacuous)");
    // (BIN3O1 audit F1) the other two fallbacks, and the production line that shows them all.
    check(c1.rankq_fallback == c0.rankq_fallback, "(C) rank_q_relation fell back to the full A " + str(c1.rankq_fallback - c0.rankq_fallback) + " times");
    check(c1.parity_fallback == c0.parity_fallback, "(C) read_lattice found no parity structure " + str(c1.parity_fallback - c0.parity_fallback) + " times");
    {
        const std::string e = l23::bin3o1_route_line();
        check(e.rfind("BIN3O1_ROUTE route=on split=", 0) == 0 && e.find(" fallback=") != std::string::npos, "(C) route line: " + e);
    }
    {
        l23::ScopedBin3o1 off(false);
        const std::string e = l23::bin3o1_effective_line();
        check(!l23::bin3o1_enabled() && e.rfind("EFFECTIVE l23_bin3o1=off source=scoped", 0) == 0, "(C) scoped off: " + e);
    }
    {
        l23::ScopedBin3o1 on(true);
        const std::string e = l23::bin3o1_effective_line();
        check(l23::bin3o1_enabled() && e.rfind("EFFECTIVE l23_bin3o1=on source=scoped", 0) == 0, "(C) scoped on: " + e);
    }
    {
        // The comparator sees one flipped entry: the shift kernel on a perturbed copy differs in exactly one entry before the sweep.
        std::vector<uint64_t> X(17 * 9), Y;
        uint64_t s = 99;
        for (auto& x : X) x = splitmix(s);
        Y = X;
        Y[5 * 9 + 3] ^= 1;
        uint64_t diff = 0;
        for (size_t i = 0; i < X.size(); ++i) diff += X[i] != Y[i];
        r2::kernel_shift_rows(X.data(), 9, 16, 9, ~uint64_t(0));
        r2::kernel_shift_rows(Y.data(), 9, 16, 9, ~uint64_t(0));
        uint64_t after = 0;
        for (size_t i = 0; i < X.size(); ++i) after += X[i] != Y[i];
        check(diff == 1 && after >= 1, "(C) negative control: a flipped entry is not seen (before " + str(diff) + ", after " + str(after) + ")");
    }
    std::cout << "  (C) counters: split_accepted=" << split << " split_fallback=" << fallback << " odd_reduced=" << odd << " rankq_blocks=" << rankq
              << " f2_cells=" << f2 << " hecke_reused=" << reused << std::endl;
    std::cout << "  " << l23::bin3o1_effective_line() << std::endl;

    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "BIN3O1 shift_kernel=%llu relation=%llu operators=%llu h1_mod=%llu lattice=%llu split=%llu/%llu odd=%llu rankq=%llu f2=%llu "
                  "reused=%llu checks=%llu failing=%u s=%.4g verdict=%s",
                  (unsigned long long)kcases, (unsigned long long)rcases, (unsigned long long)scases, (unsigned long long)hcases,
                  (unsigned long long)lcases, (unsigned long long)split, (unsigned long long)(split + fallback), (unsigned long long)odd,
                  (unsigned long long)rankq, (unsigned long long)f2, (unsigned long long)reused, (unsigned long long)g_checks, g_fail, sec,
                  g_fail ? "FAIL" : "OK");
    std::cout << buf << std::endl;
    return g_fail;
}
