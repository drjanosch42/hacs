// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// NumberFieldCyclicCubicValidation (standard) -- the cyclic cubic field K of prime conductor ell, its class number
// h_K = [E_K : C] by saturation of the cyclotomic units, and membership of ell in Lambda-bar of arXiv:2610.17865
// (ell = 1 mod 3 and the 2-rank of Cl(K) is >= 1, i.e. 2 | h <=> 4 | h) -- hecke-extensions slot 23, package C3
// (docs/implementation/slot23/arxiv_2610_17865_assessment.md sec. 5; report docs/implementation/slot23/C3_report.md).
//
//   F  field data for every prime ell = 1 mod 3, 7 <= ell < 2000: 4 ell = a^2 + 27 b^2 with a = 1 mod 3, the period
//      polynomial x^3 + x^2 - (ell-1)/3 x - (ell(a+3) - 1)/27 computed exactly from the structure constants, its
//      discriminant ell^2 b^2, d_K = det Tr(eta_i eta_j) = ell^2; Gauss's polynomials at ell = 7, 13, 19, 31, 37.
//   H  h_K for the 80 primes ell = 1 mod 3 below 1000: the cyclotomic units exact (xi, u = sigma(xi)/xi, v^2 = u),
//      the saturation complete up to Cusick's bound, the analytic class number formula (digamma route) equal to
//      R(C) to 1e-10 (the independent check, of the identity R(C) = hR), h_an = hR / R within 1e-6 of h (a
//      consistency check only: it cannot detect an incomplete saturation), 3 not dividing h, p = 2 mod 3 to even
//      powers, v_2(h) even; h against the table pinned below (the first run of this code, 2026-10-15, reproduced
//      by the C3 audit's independent mpmath computation, C3_audit.md finding 2; further anchors:
//      the paper's Lambda = {163, 277, 349, 547, 607, ...} lies in Lambda-bar, and no ell < 163 is in Lambda-bar
//      since the least discriminant of a totally real A_4 quartic field is 26569 = 163^2).
//   D  determinism: the whole table again at +64 bits and the ell < 400 part again at the same precision, equal
//      results (h, the found roots, their certificates) and equal SHA-256 digests.
//   N  controls: starting the saturation from <v^k, sigma(v)^k> (k = 2, 3) must find index k^2 h; the brief's
//      variant x^3 - x^2 - ... is not the period polynomial; non-units have |N| != 1.
// No env hook, no golden object, writes nothing.
#include "CyclicCubicValidation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "BigInt.h"
#include "CyclicCubic.h"
#include "Sha256.h"

using namespace nf;

namespace {

uint32_t g_fail = 0, g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string Fx(double x, int prec) {
    std::ostringstream os;
    os << std::setprecision(prec) << x;
    return os.str();
}

bool IsPrime(uint32_t n) {
    if (n < 2) return false;
    for (uint32_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

std::vector<uint32_t> Conductors(uint32_t lo, uint32_t hi) {
    std::vector<uint32_t> v;
    for (uint32_t l = lo; l < hi; ++l)
        if (l % 3 == 1 && IsPrime(l)) v.push_back(l);
    return v;
}

std::string ResultKey(const CubicClassResult& R) {
    std::ostringstream os;
    os << R.ell << ":" << R.h << ":";
    for (uint32_t q : R.roots) os << q << ",";
    os << ":" << R.root_certificate;
    return os.str();
}

std::string Digest(const std::vector<CubicClassResult>& rs) {
    Sha256 s;
    for (const auto& R : rs) {
        const std::string k = ResultKey(R) + "\n";
        s.update(k.data(), k.size());
    }
    return s.hex();
}

// ---- F: field data ---------------------------------------------------------------------------------------------
void F() {
    const auto ells = Conductors(7, 2000);
    uint32_t bad = 0;
    for (uint32_t l : ells) {
        const CyclicCubicField K = cyclic_cubic_field(l);
        const BigInt L(l);
        bool ok = 4 * static_cast<int64_t>(l) == K.a * K.a + 27 * K.b * K.b && ((K.a % 3) + 3) % 3 == 1 && K.b > 0;
        ok = ok && K.poly[2] == 1 && K.poly[1] == -static_cast<int64_t>((l - 1) / 3) && 27 * K.e3 == static_cast<int64_t>(l) * (K.a + 3) - 1;
        ok = ok && K.poly_disc == L * L * BigInt(K.b) * BigInt(K.b) && K.basis_disc == L * L;
        if (!ok) {
            ++bad;
            std::cout << "  field ell=" << l << " a=" << K.a << " b=" << K.b << " poly=[" << K.poly[2] << "," << K.poly[1] << "," << K.poly[0]
                      << "] disc=" << K.poly_disc << " dK=" << K.basis_disc << std::endl;
        }
        Check(ok, "F ell=" + std::to_string(l) + ": a, b, period polynomial, disc = ell^2 b^2, d_K = ell^2");
    }
    std::cout << "CUBIC_FIELDS range=[7,2000) fields=" << ells.size() << " bad=" << bad << std::endl;
    // Gauss's period polynomials (classical): x^3 + x^2 + c1 x + c0.
    const std::map<uint32_t, std::pair<int64_t, int64_t>> gauss = {{7, {-2, -1}}, {13, {-4, 1}}, {19, {-6, -7}}, {31, {-10, -8}}, {37, {-12, 11}}};
    for (const auto& kv : gauss) {
        const CyclicCubicField K = cyclic_cubic_field(kv.first);
        std::cout << "CUBIC_FIELD ell=" << kv.first << " a=" << K.a << " b=" << K.b << " poly=x^3+x^2" << (K.poly[1] < 0 ? "" : "+") << K.poly[1]
                  << "x" << (K.poly[0] < 0 ? "" : "+") << K.poly[0] << " disc=" << K.poly_disc << " dK=" << K.basis_disc << std::endl;
        Check(K.poly[1] == kv.second.first && K.poly[0] == kv.second.second, "F Gauss period polynomial at ell=" + std::to_string(kv.first));
    }
    // Field arithmetic: the norm is multiplicative and sigma has order 3.
    const CyclicCubicField K = cyclic_cubic_field(163);
    const CubicElt x = {BigInt(3), BigInt(-7), BigInt(11)}, y = {BigInt(-2), BigInt(5), BigInt(1)};
    Check(cubic_norm(K, cubic_mul(K, x, y)) == cubic_norm(K, x) * cubic_norm(K, y), "F N(xy) = N(x) N(y) at ell=163");
    Check(cubic_sigma(cubic_sigma(cubic_sigma(x))) == x, "F sigma^3 = 1");
    Check(cubic_mul(K, x, cubic_one()) == x, "F 1 = -(eta_0 + eta_1 + eta_2) is the identity");
}

// ---- H: class numbers ------------------------------------------------------------------------------------------
std::vector<CubicClassResult> g_table;

void H() {
    // Pinned: h_K != 1 for the primes ell = 1 mod 3 below 1000 (all others have h = 1).
    const std::map<uint32_t, uint64_t> pinned = {{163, 4}, {277, 4}, {313, 7}, {349, 4}, {397, 4}, {547, 4},
                                                 {607, 4}, {709, 4}, {853, 4}, {877, 7}, {937, 4}};
    const std::set<uint32_t> paper_Lambda = {163, 277, 349, 547, 607};
    const auto ells = Conductors(7, 1000);
    Check(ells.size() == 80, "H 80 primes ell = 1 mod 3 below 1000");
    std::vector<uint32_t> lambda_bar;
    for (uint32_t l : ells) {
        const CubicClassResult R = cyclic_cubic_class_number(l);
        g_table.push_back(R);
        std::cout << "CUBIC_CLASS ell=" << l << " a=" << R.a << " b=" << R.b << " h=" << R.h << " v2=" << R.v2 << " rank2="
                  << (R.rank2 >= 0 ? std::to_string(R.rank2) : std::string(">=2")) << " lambda_bar=" << (R.in_lambda_bar ? 1 : 0)
                  << " R=" << Fx(R.R, 10) << " hR_an=" << Fx(R.hR_analytic, 10) << " h_an=" << Fx(R.h_analytic, 12) << " rel=" << Fx(R.rel_err, 3)
                  << " R_low=" << Fx(R.R_low, 5) << " sat_bound=" << R.sat_bound << " tests=" << R.root_tests
                  << " closest_reject_log2=" << Fx(R.min_reject_log2, 4) << " prec=" << R.prec
                  << (R.root_certificate.empty() ? "" : " cert=" + R.root_certificate) << std::endl;
        const std::string tag = "H ell=" + std::to_string(l) + ": ";
        Check(R.field_ok && R.units_ok, tag + "field data and cyclotomic units exact");
        Check(R.rel_err < 1e-10, tag + "analytic hR (digamma) = R(C)");
        Check(std::fabs(R.h_analytic - static_cast<double>(R.h)) < 1e-6, tag + "h_an = hR / R equals the index (consistency)");
        uint64_t prod = 1;
        for (uint32_t q : R.roots) prod *= q;
        Check(prod == R.h, tag + "h = product of the found roots");
        Check(R.h % 3 != 0, tag + "3 does not divide h (genus theory)");
        uint64_t h = R.h;
        bool even_ok = true;
        for (uint64_t p = 2; p <= h; ++p) {
            unsigned e = 0;
            while (h % p == 0) {
                h /= p;
                ++e;
            }
            if (e && p % 3 == 2 && e % 2) even_ok = false;
        }
        Check(even_ok && R.v2 % 2 == 0, tag + "primes p = 2 mod 3 divide h to even powers");
        const auto it = pinned.find(l);
        Check(R.h == (it == pinned.end() ? 1u : it->second), tag + "h equals the pinned table");
        if (R.in_lambda_bar) lambda_bar.push_back(l);
        if (l < 163) Check(!R.in_lambda_bar, tag + "not in Lambda-bar (no totally real A_4 quartic of discriminant < 163^2)");
        if (paper_Lambda.count(l)) Check(R.in_lambda_bar && R.rank2 == 2, tag + "the paper's Lambda lies in Lambda-bar, Cl[2] = (Z/2)^2");
    }
    std::ostringstream os;
    for (std::size_t i = 0; i < lambda_bar.size(); ++i) os << (i ? "," : "") << lambda_bar[i];
    bool all_in = true;
    for (uint32_t l : paper_Lambda) all_in = all_in && std::find(lambda_bar.begin(), lambda_bar.end(), l) != lambda_bar.end();
    std::string extra;
    for (uint32_t l : lambda_bar)
        if (l < 607 && !paper_Lambda.count(l)) extra += (extra.empty() ? "" : ",") + std::to_string(l);
    std::cout << "CUBIC_LAMBDA_BAR range=[7,1000) members=" << os.str() << " count=" << lambda_bar.size()
              << " paper_Lambda=163,277,349,547,607 all_in=" << (all_in ? 1 : 0) << " in_lambda_bar_below_607_not_listed="
              << (extra.empty() ? "none" : extra) << std::endl;
    Check(lambda_bar == std::vector<uint32_t>({163, 277, 349, 397, 547, 607, 709, 853, 937}), "H Lambda-bar below 1000");
}

// ---- D: determinism -------------------------------------------------------------------------------------------
void D() {
    std::vector<CubicClassResult> hi, again;
    for (const auto& R : g_table) {
        hi.push_back(cyclic_cubic_class_number(R.ell, 64));
        if (R.ell < 400) again.push_back(cyclic_cubic_class_number(R.ell));
    }
    std::vector<CubicClassResult> low(g_table.begin(), g_table.begin() + static_cast<std::ptrdiff_t>(again.size()));
    const std::string d0 = Digest(g_table), d1 = Digest(hi), d2 = Digest(low), d3 = Digest(again);
    std::cout << "CUBIC_DETERMINISM digest=" << d0.substr(0, 16) << " digest_plus64=" << d1.substr(0, 16) << " fields=" << g_table.size()
              << " rerun_lt400=" << (d2 == d3 ? "equal" : "DIFFERENT") << std::endl;
    Check(d0 == d1, "D table equal at +64 bits of precision");
    Check(d2 == d3, "D rerun at the same precision equal");
}

// ---- N: controls -----------------------------------------------------------------------------------------------
void N() {
    for (uint32_t l : {7u, 163u, 313u}) {
        const uint64_t h = cyclic_cubic_class_number(l).h;
        for (unsigned k : {2u, 3u}) {
            const CubicClassResult R = cyclic_cubic_class_number(l, 0, k);
            std::cout << "CUBIC_CONTROL ell=" << l << " start_power=" << k << " index=" << R.h << " expected=" << k * k * h << std::endl;
            Check(R.h == k * k * h, "N ell=" + std::to_string(l) + " start_power=" + std::to_string(k) + ": index k^2 h");
        }
    }
    // The brief's variant x^3 - x^2 - (ell-1)/3 x - (ell(a+3)-1)/27 is not the period polynomial (e1 = -1 forces +x^2).
    const CyclicCubicField K = cyclic_cubic_field(7);
    Check(K.poly[2] == 1, "N the period polynomial has +x^2 (e1 = eta_0 + eta_1 + eta_2 = -1), not -x^2");
    // A non-unit: N(eta_0) = e3 at ell = 7 is 1 (eta_0 is a unit there), at 13 it is -1; 2 eta_0 has norm 8 e3.
    const CubicElt e0 = {BigInt(2), BigInt(0), BigInt(0)};
    Check(cubic_norm(K, e0).abs() == BigInt(8), "N N(2 eta_0) = 8 at ell = 7");
}

}  // namespace

uint32_t NumberFieldCyclicCubicValidation() {
    g_fail = g_checks = 0;
    g_table.clear();
    const double t0 = Now();
    std::cout << "NumberFieldCyclicCubicValidation (slot 23 C3; cyclic cubic fields of prime conductor, h = [E : C], Lambda-bar of arXiv:2610.17865)"
              << std::endl;
    try {
        F();
        H();
        D();
        N();
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    std::cout << "NF_CUBIC_SUMMARY checks=" << g_checks << " failing=" << g_fail << " secs=" << Fx(Now() - t0, 3) << std::endl;
    return g_fail;
}
