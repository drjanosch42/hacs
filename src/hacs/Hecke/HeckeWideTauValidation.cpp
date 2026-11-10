// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 24, package ELL-W) The wide tier of the tau table; see HeckeWideTauValidation.h.

#include "HeckeWideTauValidation.h"

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "../Util/Timing.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using oc_util::Seconds;
using engine::scalar::powmod;

using hecke::ExactTier;
using hecke::HeckeCosetTables;
using hecke::HeckeOverflow;
using hecke::HeckeTauTable;
using hecke::Mat22;
using hecke::Mat22W;
using hecke::ScopedExactTier;
using hecke::SL2ZTessellationTables;

uint32_t g_checks = 0;
uint32_t g_fails = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::cout << "  *** FAIL [HeckeWideTau] " << what << std::endl;
    }
}

// ---- [1] the helpers --------------------------------------------------------------------------------------------

bool SameMat(const Mat22& x, const Mat22& y) { return x == y; }

// The reference value of A * B in __int128, or false if an entry leaves int64.
bool RefProduct(const Mat22& A, const Mat22& B, Mat22& out) {
    const __int128 e[4] = {
        static_cast<__int128>(A.a) * B.a + static_cast<__int128>(A.b) * B.c,
        static_cast<__int128>(A.a) * B.b + static_cast<__int128>(A.b) * B.d,
        static_cast<__int128>(A.c) * B.a + static_cast<__int128>(A.d) * B.c,
        static_cast<__int128>(A.c) * B.b + static_cast<__int128>(A.d) * B.d };
    for (const __int128 x : e)
        if (x > static_cast<__int128>(INT64_MAX) || x < static_cast<__int128>(INT64_MIN)) return false;
    out = Mat22{ static_cast<int64_t>(e[0]), static_cast<int64_t>(e[1]), static_cast<int64_t>(e[2]), static_cast<int64_t>(e[3]) };
    return true;
}

void ArmHelpers(bool lever_on) {
    const uint32_t c0 = g_checks, f0 = g_fails;
    std::mt19937_64 rng(0x5eed24e11u);
    // (a) int64-fitting operands: the narrow value, under Auto and under ForceWide; no escalation under Auto.
    hecke::reset_wide_product_escalations();
    uint32_t agree = 0;
    for (int it = 0; it < 20000; ++it) {
        const int64_t lim = (it % 2) ? (int64_t(1) << 30) : 1000;
        std::uniform_int_distribution<int64_t> d(-lim, lim);
        const Mat22 A{ d(rng), d(rng), d(rng), d(rng) };
        const Mat22 B{ d(rng), d(rng), d(rng), d(rng) };
        const Mat22 narrow = A * B;   // |entry| <= 2^61: int64
        const Mat22 autov = hecke::mul_tiered(A, B);
        Mat22 widev{};
        int64_t det_w = 0;
        {
            const ScopedExactTier g(ExactTier::ForceWide);
            widev = hecke::mul_tiered(A, B);
            det_w = hecke::det_tiered(A);
        }
        if (SameMat(narrow, autov) && SameMat(narrow, widev) && hecke::det_tiered(A) == A.det() && det_w == A.det()) ++agree;
    }
    Check(agree == 20000, "[1](a) mul_tiered / det_tiered differ from the int64 value on int64-fitting operands");
    Check(hecke::wide_product_escalations() == 0, "[1](a) Auto escalated on int64-fitting operands");
    // (b) fault injection: int64 intermediates overflow, the result fits.  operator* throws; mul_tiered is exact.
    {
        const int64_t t = int64_t(1) << 32;
        const Mat22 A{ t, t, 0, 1 };
        const Mat22 B{ t, 0, -t, 1 };   // A B = [[0, 2^32], [-2^32, 1]]: (0,0) = 2^64 - 2^64
        bool threw = false;
        try { (void)(A * B); } catch (const HeckeOverflow&) { threw = true; }
        Check(threw, "[1](b) the int64 product did not overflow (vacuous fault injection)");
        if (!lever_on) {
            // HECKE_TAU_WIDE=0: the helpers ARE the narrow tier -- every overflow propagates, nothing escalates.
            const int64_t b = (int64_t(1) << 62) - 1;
            const Mat22 U{ t + 1, t, t, t - 1 };
            const Mat22 G{ b, b - 1, b - 1, b - 2 };
            uint32_t thrown = 0;
            try { (void)hecke::mul_tiered(A, B); } catch (const HeckeOverflow&) { ++thrown; }
            try { (void)hecke::det_tiered(U); } catch (const HeckeOverflow&) { ++thrown; }
            try { (void)hecke::det_tiered(G); } catch (const HeckeOverflow&) { ++thrown; }
            try { (void)hecke::inv_unimodular_tiered(G); } catch (const HeckeOverflow&) { ++thrown; }
            Check(thrown == 4, "[1] HECKE_TAU_WIDE=0: a tiered helper escalated (" + std::to_string(thrown) + "/4 threw)");
            Check(hecke::wide_product_escalations() == 0, "[1] HECKE_TAU_WIDE=0: an escalation was counted");
            std::cout << "  [1] helpers (HECKE_TAU_WIDE=0, the rollback): " << (g_checks - c0) << " checks, " << (g_fails - f0)
                      << " failing (the int64 value on fitting operands; every overflow propagates)" << std::endl;
            return;
        }
        hecke::reset_wide_product_escalations();
        const Mat22 got = hecke::mul_tiered(A, B);
        Mat22 ref{};
        Check(RefProduct(A, B, ref) && SameMat(got, ref) && got == (Mat22{ 0, t, -t, 1 }), "[1](b) mul_tiered is not the exact product");
        Check(hecke::wide_product_escalations() == 1, "[1](b) the escalation was not counted");
        // A unimodular matrix whose determinant's int64 intermediates overflow: det = (2^32+1)(2^32-1) - 2^64 = -1.
        const Mat22 U{ t + 1, t, t, t - 1 };
        threw = false;
        try { (void)U.det(); } catch (const HeckeOverflow&) { threw = true; }
        Check(threw, "[1](b) det: the int64 determinant did not overflow (vacuous)");
        Check(hecke::det_tiered(U) == -1, "[1](b) det_tiered is not -1");
        const Mat22 Ui = hecke::inv_unimodular_tiered(U);
        Mat22 id{};
        Check(RefProduct(U, Ui, id) && id == Mat22::I(), "[1](b) inv_unimodular_tiered is not the inverse");
        // (c) ForceNarrow rethrows (the negative control).
        threw = false;
        {
            const ScopedExactTier g(ExactTier::ForceNarrow);
            try { (void)hecke::mul_tiered(A, B); } catch (const HeckeOverflow&) { threw = true; }
        }
        Check(threw == true, "[1](c) ForceNarrow did not rethrow the int64 overflow");
        // (d) a result outside int64 throws (abort, never wrap).
        const Mat22 C{ t, t, 0, 1 };
        const Mat22 D{ t, 0, t, 1 };   // (0,0) = 2^65
        threw = false;
        try { (void)hecke::mul_tiered(C, D); } catch (const HeckeOverflow&) { threw = true; }
        Check(threw, "[1](d) a product outside int64 did not throw");
        // (e) the extreme operand: B = 2^62 - 1 (the wide clause's bound), all four entries; intermediates 2^124.
        const int64_t b = (int64_t(1) << 62) - 1;
        const Mat22 G{ b, b - 1, b - 1, b - 2 };   // det = b(b-2) - (b-1)^2 = -1, intermediates ~ 2^124
        Check(hecke::det_tiered(G) == -1, "[1](e) the extreme determinant is not -1");
        const Mat22 Gi = hecke::inv_unimodular_tiered(G);   // = [[-(b-2), b-1], [b-1, -b]]
        Check(Gi == (Mat22{ -(b - 2), b - 1, b - 1, -b }), "[1](e) the extreme inverse is wrong");
        Check(hecke::mul_tiered(G, Gi) == Mat22::I() && hecke::mul_tiered(Gi, G) == Mat22::I(),
              "[1](e) G G^{-1} != I with intermediates ~ 2^124");
    }
    std::cout << "  [1] helpers: " << (g_checks - c0) << " checks, " << (g_fails - f0) << " failing (20000 random pairs, "
              << "fault injection, ForceNarrow, overflowing result, extreme operands)" << std::endl;
}

// ---- tau dumps ----------------------------------------------------------------------------------------------------

struct TauDump {
    std::vector<int64_t> flat;
    uint64_t B = 0, B_tab = 0;
    bool wide = false;
    uint32_t certify = 99;
};

TauDump Dump(const HeckeTauTable& tab, const SimplicialComplex& K) {
    TauDump d;
    d.B = tab.certified_bound_B();
    d.B_tab = tab.certified_bound_B_tab();
    d.wide = tab.wide_tier();
    d.certify = tab.certify();
    for (uint32_t i = 1; i <= tab.num_cosets(); ++i)
        for (uint32_t q = 0; q <= 2; ++q)
            for (uint32_t s = 0; s < K.get_simplices()[q].size(); ++s) {
                const std::vector<hecke::ChainTerm>& terms = tab.tau(i, q, s);
                d.flat.push_back(static_cast<int64_t>(terms.size()));
                for (const hecke::ChainTerm& t : terms) {
                    d.flat.push_back(t.coeff);
                    d.flat.push_back(t.gamma.a);
                    d.flat.push_back(t.gamma.b);
                    d.flat.push_back(t.gamma.c);
                    d.flat.push_back(t.gamma.d);
                    d.flat.push_back(static_cast<int64_t>(t.sigma_idx));
                }
            }
    return d;
}

TauDump BuildUnder(const SimplicialComplex& K, const SL2ZTessellationTables& model, uint64_t p, uint32_t ell, uint32_t n,
                   ExactTier tier) {
    const ScopedExactTier g(tier);
    const HeckeCosetTables cosets(p, 1, ell, n);
    const HeckeTauTable tab(K, model, cosets, ell, n);
    return Dump(tab, K);
}

// ---- [2] forced-wide = narrow on ell <= 23167 -----------------------------------------------------------------------

void ArmForcedWide(const SimplicialComplex& K, const SL2ZTessellationTables& model, bool slow) {
    struct Case { uint64_t p; uint32_t ell, n; };
    // standard: ell <= 5003; slow: the largest narrow prime 23167 (two builds, ~30 s)
    const std::vector<Case> cases = slow ? std::vector<Case>{ {23167, 23167, 0} }
                                         : std::vector<Case>{ {5, 5, 0}, {691, 691, 0}, {5003, 5003, 0}, {2, 691, 10}, {3, 691, 10} };
    for (const Case& c : cases) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string tag = "[2] p=" + std::to_string(c.p) + " ell=" + std::to_string(c.ell) + " n=" + std::to_string(c.n);
        try {
            hecke::reset_wide_product_escalations();
            const TauDump a = BuildUnder(K, model, c.p, c.ell, c.n, ExactTier::Auto);
            const uint64_t esc = hecke::wide_product_escalations();
            const TauDump w = BuildUnder(K, model, c.p, c.ell, c.n, ExactTier::ForceWide);
            Check(a.certify == 0 && w.certify == 0, tag + ": certify() != 0");
            Check(!a.wide && !w.wide, tag + ": a table at N <= 23170 is not in the narrow tier");
            Check(esc == 0, tag + ": Auto performed a wide-product escalation (" + std::to_string(esc) + ")");
            Check(a.B == w.B && a.B_tab == w.B_tab, tag + ": B / B_tab differ under ForceWide");
            Check(a.flat == w.flat && !a.flat.empty(), tag + ": the tau chains differ under ForceWide (or are empty)");
            std::cout << "  " << tag << ": " << a.flat.size() << " tau scalars identical under ForceWide, B_tab=" << a.B_tab
                      << " B=" << a.B << ", narrow, escalations(Auto)=" << esc << " (" << Seconds(t0) << " s)" << std::endl;
        } catch (const std::exception& e) {
            Check(false, tag + " threw: " + e.what());
        }
    }
}

// ---- [3] the boundary 23173 -----------------------------------------------------------------------------------------

// (ELL-W audit F4; slow name) ForceWide = Auto ENTRYWISE on a WIDE table: every tiered site (the (T1) translate, the edge
// relation, clause (a)'s determinants, twisted_conjugate, word_in_T_S) evaluated in __int128 inside the real pipeline.
void ArmForcedWideAtWide(const SimplicialComplex& K, const SL2ZTessellationTables& model, bool lever_on) {
    if (!lever_on) return;   // under the rollback the table is refused (arm [3] of the standard name asserts it)
    const auto t0 = std::chrono::steady_clock::now();
    try {
        const TauDump a = BuildUnder(K, model, 23173, 691, 0, ExactTier::Auto);
        const TauDump w = BuildUnder(K, model, 23173, 691, 0, ExactTier::ForceWide);
        Check(a.certify == 0 && a.wide && w.certify == 0 && w.wide, "[3w] ell=23173 does not certify wide under Auto / ForceWide");
        Check(w.B == a.B && w.B_tab == a.B_tab, "[3w] ForceWide: B / B_tab differ at ell=23173");
        Check(w.flat == a.flat && !a.flat.empty(), "[3w] ForceWide: the tau chains differ at ell=23173 (or are empty)");
        std::cout << "  [3w] ell=23173: ForceWide = Auto entrywise (" << w.flat.size() << " tau scalars, the __int128 forms on a wide "
                  << "table) (" << Seconds(t0) << " s)" << std::endl;
    } catch (const std::exception& e) {
        Check(false, std::string("[3w] ForceWide at ell=23173 threw: ") + e.what());
    }
}

void ArmBoundary(const SimplicialComplex& K, const SL2ZTessellationTables& model, bool lever_on) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string msg;
    TauDump a;
    bool built = false;
    try {
        hecke::reset_wide_product_escalations();
        a = BuildUnder(K, model, 23173, 691, 0, ExactTier::Auto);
        built = true;
    } catch (const std::exception& e) {
        msg = e.what();
    }
    if (lever_on) {
        Check(built && a.certify == 0 && a.wide, "[3] ell=23173 does not certify in the wide tier (" + msg + ")");
        Check(a.B == 4ull * 23173ull * 23173ull, "[3] ell=23173: B is not 4 N^2");
        Check(a.B_tab <= a.B, "[3] ell=23173: B_tab > B");
    } else {
        Check(!built && msg.find("certification FAILED (code 51)") != std::string::npos,
              "[3] HECKE_TAU_WIDE=0: ell=23173 was not refused with code 51 (" + msg + ")");
    }
    std::string nmsg;
    bool nbuilt = false;
    try {
        (void)BuildUnder(K, model, 23173, 691, 0, ExactTier::ForceNarrow);
        nbuilt = true;
    } catch (const std::exception& e) {
        nmsg = e.what();
    }
    Check(!nbuilt && (nmsg.find("overflow") != std::string::npos || nmsg.find("code 51") != std::string::npos),
          "[3] ForceNarrow: ell=23173 produced a table or a different failure (" + nmsg + ")");
    std::cout << "  [3] ell=23173: " << (built ? "certified, wide_tier=" + std::to_string(a.wide) + " B_tab=" + std::to_string(a.B_tab) +
                                                     " B=" + std::to_string(a.B) + " escalations=" +
                                                     std::to_string(hecke::wide_product_escalations())
                                           : "refused: " + msg)
              << "; ForceNarrow refused (" << nmsg.substr(0, 60) << ") (" << Seconds(t0) << " s)" << std::endl;
}

// ---- [4] oracles at n = 10 ------------------------------------------------------------------------------------------

uint64_t MulMod(uint64_t a, uint64_t b, uint64_t m) { return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % m); }

// tau(p) mod ell from Delta = q prod (1 - q^n)^24 = q (prod (1 - q^n)^3)^8 and Jacobi's
// prod (1 - q^n)^3 = sum_{k >= 0} (-1)^k (2k + 1) q^{k(k+1)/2}: the coefficient of q^{p-1} of the 8th power.
uint64_t RamanujanTauMod(uint64_t p, uint64_t ell) {
    const uint64_t L = p;   // coefficients 0 .. p - 1
    std::vector<std::pair<uint64_t, uint64_t>> eta3;   // (exponent, coefficient mod ell)
    for (uint64_t k = 0; k * (k + 1) / 2 < L; ++k) {
        const uint64_t c = (2 * k + 1) % ell;
        eta3.emplace_back(k * (k + 1) / 2, (k % 2) ? (ell - c) % ell : c);
    }
    std::vector<uint64_t> f(L, 0);
    for (const auto& t : eta3) f[t.first] = t.second;
    for (int pow = 2; pow <= 8; ++pow) {
        std::vector<uint64_t> g(L, 0);
        for (uint64_t i = 0; i < L; ++i) {
            if (f[i] == 0) continue;
            for (const auto& t : eta3) {
                if (i + t.first >= L) break;
                g[i + t.first] = (g[i + t.first] + MulMod(f[i], t.second, ell)) % ell;
            }
        }
        f.swap(g);
    }
    return f[p - 1];
}

// The characteristic polynomial X^3 + c2 X^2 + c1 X + c0 of a 3 x 3 matrix mod ell, as {c0, c1, c2}.
std::array<uint64_t, 3> CharPoly3(const Matrix& A, uint64_t ell) {
    auto a = [&](int i, int j) { return static_cast<uint64_t>(A.data[i][j]) % ell; };
    auto sub = [&](uint64_t x, uint64_t y) { return (x + ell - y % ell) % ell; };
    const uint64_t tr = (a(0, 0) + a(1, 1) + a(2, 2)) % ell;
    uint64_t m2 = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j) m2 = (m2 + sub(MulMod(a(i, i), a(j, j), ell), MulMod(a(i, j), a(j, i), ell))) % ell;
    const uint64_t det = sub((MulMod(a(0, 0), sub(MulMod(a(1, 1), a(2, 2), ell), MulMod(a(1, 2), a(2, 1), ell)), ell) +
                              MulMod(a(0, 2), sub(MulMod(a(1, 0), a(2, 1), ell), MulMod(a(1, 1), a(2, 0), ell)), ell)) % ell,
                             MulMod(a(0, 1), sub(MulMod(a(1, 0), a(2, 2), ell), MulMod(a(1, 2), a(2, 0), ell)), ell));
    return { (ell - det) % ell, m2, (ell - tr) % ell };
}

// (X - t)^2 (X - e) = X^3 - (2t + e) X^2 + (t^2 + 2te) X - t^2 e.
std::array<uint64_t, 3> Expected(uint64_t t, uint64_t e, uint64_t ell) {
    const uint64_t t2 = MulMod(t, t, ell);
    return { (ell - MulMod(t2, e, ell)) % ell, (t2 + MulMod(2 * t % ell, e, ell)) % ell, (ell - (2 * t + e) % ell) % ell };
}

std::string PolyText(const std::array<uint64_t, 3>& c) {
    return "X^3 + " + std::to_string(c[2]) + " X^2 + " + std::to_string(c[1]) + " X + " + std::to_string(c[0]);
}

void ArmOracle(const SimplicialComplex& K, const SL2ZTessellationTables& model, const std::string& tag, uint64_t p, uint32_t ell,
               uint64_t eis, bool expect_wide, bool lever_on) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t n = 10;
    try {
        LocalSystemSimplicialComplex lssc(K, GenerateSL2ZRelationsElt(K, n, ell), n + 1, ell);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        const std::vector<uint32_t> dims = coh.get_H_eq_dimensions();
        Check(dims.size() == 3 && dims[0] == 0 && dims[1] == 3 && dims[2] == 0, tag + ": dim H^* is not (0, 3, 0)");
        const HeckeCosetTables cosets(p, 1, ell, n);
        std::string msg;
        try {
            hecke::reset_wide_product_escalations();
            const HeckeTauTable tau(K, model, cosets, ell, n);
            const uint64_t esc = hecke::wide_product_escalations();
            Check(tau.wide_tier() == expect_wide, tag + ": wide_tier() is not " + std::to_string(expect_wide));
            hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
            op.compute();
            const Matrix& A1 = op.get_H_matrices()[1];
            Check(A1.rows == 3 && A1.cols == 3, tag + ": A^(1) is not 3 x 3");
            const uint64_t t = RamanujanTauMod(p, ell);
            const auto got = CharPoly3(A1, ell);
            const auto want = Expected(t, eis % ell, ell);
            Check(got == want, tag + ": charpoly " + PolyText(got) + " != (X - tau(p))^2 (X - eis) = " + PolyText(want));
            std::cout << "  " << tag << ": tau(" << p << ") = " << t << " mod " << ell << ", eis = " << eis % ell
                      << ", charpoly " << PolyText(got) << (got == want ? " OK" : " MISMATCH") << ", wide_tier=" << tau.wide_tier()
                      << " B_tab=" << tau.certified_bound_B_tab() << " B=" << tau.certified_bound_B() << " escalations=" << esc
                      << " (" << Seconds(t0) << " s)" << std::endl;
            if (expect_wide && !lever_on) Check(false, tag + ": HECKE_TAU_WIDE=0 but the wide table was accepted");
        } catch (const std::exception& e) {
            msg = e.what();
            const bool refused51 = msg.find("certification FAILED (code 51)") != std::string::npos;
            Check(expect_wide && !lever_on && refused51, tag + " threw: " + msg);
            if (expect_wide && !lever_on)
                std::cout << "  " << tag << ": HECKE_TAU_WIDE=0 -- refused with code 51, as before ELL-W (" << Seconds(t0)
                          << " s)" << std::endl;
        }
    } catch (const std::exception& e) {
        Check(false, tag + " (cohomology / cosets) threw: " + std::string(e.what()));
    }
}

// ---- [5] the clause arithmetic --------------------------------------------------------------------------------------

void ArmClause(const SimplicialComplex& K, const SL2ZTessellationTables& model, bool lever_on) {
    // (ELL-W audit F3) the clause itself, hecke::tau_bound_clause -- the function certify() calls -- at its boundaries.
    using U = unsigned __int128;
    auto fourNN = [](uint64_t N) { return static_cast<U>(4) * N * N; };
    const U i64max = static_cast<U>(INT64_MAX);
    Check(hecke::tau_bound_clause(23170, fourNN(23170), true) == 0 && hecke::tau_bound_clause(23170, fourNN(23170), false) == 0,
          "[5] N = 23170: not narrow");
    Check(hecke::tau_bound_clause(23171, fourNN(23171), true) == 0 && hecke::tau_bound_clause(23171, fourNN(23171), false) == 51,
          "[5] N = 23171: not wide (0) / not 51 without the wide tier");
    Check(hecke::tau_bound_clause(1048575, fourNN(1048575), true) == 0, "[5] N = 1048575 (B = 4N^2) is not certified wide");
    Check(hecke::tau_bound_clause(1048576, fourNN(1048576), true) == 52, "[5] N = 1048576 (B = 4N^2) is not code 52");
    Check(hecke::tau_bound_clause(1048576, fourNN(1048576), false) == 51, "[5] N = 1048576 without the wide tier is not 51");
    const uint64_t N = 30011;   // a B_tab-dominated B: the largest B with 2 N B <= 2^63 - 1, and one past it
    const U Bmax = i64max / (2 * static_cast<U>(N));
    Check(hecke::tau_bound_clause(N, Bmax, true) == 0 && hecke::tau_bound_clause(N, Bmax + 1, true) == 52,
          "[5] the B_tab-dominated edge 2 N B = 2^63 - 1 is off by one");
    Check(hecke::tau_bound_clause(N, static_cast<U>(UINT64_MAX), true) == 52 && hecke::tau_bound_clause(N, i64max + 1, true) == 52 &&
              hecke::tau_bound_clause(N, static_cast<U>(1) << 127, true) == 52,
          "[5] the saturation edges (B = UINT64_MAX, 2^63, 2^127) are not code 52");
    Check(hecke::tau_bound_clause(2, 2147483647ull, false) == 0 && hecke::tau_bound_clause(2, 2147483648ull, false) == 51,
          "[5] the narrow edge B = 2^31 - 1 / 2^31 is off by one");
    // The pre-flight: N = 2^20 (p = 2, r = 20; 8 N^3 = 2^63 > 2^63 - 1) is refused with code 52 BEFORE the build.
    if (lever_on) {
        const auto t0 = std::chrono::steady_clock::now();
        std::string msg;
        try {
            const HeckeCosetTables cosets(2, 20, 691, 0);
            const auto t1 = std::chrono::steady_clock::now();
            try {
                const HeckeTauTable tab(K, model, cosets, 691, 0);
                (void)tab;
            } catch (const std::exception& e) {
                msg = e.what();
            }
            const double tau_s = Seconds(t1);
            Check(msg.find("certification FAILED (code 52)") != std::string::npos && msg.find("before the build") != std::string::npos,
                  "[5] N = 2^20: not refused with code 52 before the build (" + msg + ")");
            Check(tau_s < 5.0, "[5] N = 2^20: the code-52 refusal took " + std::to_string(tau_s) + " s (not a pre-flight)");
            std::cout << "  [5] N = 2^20 (p = 2, r = 20): refused with code 52 before the build in " << tau_s << " s (cosets + refusal "
                      << Seconds(t0) << " s)" << std::endl;
        } catch (const std::exception& e) {
            Check(false, std::string("[5] N = 2^20 coset tables threw: ") + e.what());
        }
    }
    auto fits = [](uint64_t N) {
        const unsigned __int128 B = static_cast<unsigned __int128>(4) * N * N;
        return static_cast<unsigned __int128>(2) * N * B <= static_cast<unsigned __int128>(INT64_MAX);
    };
    Check(fits((1u << 20) - 1) && !fits(1u << 20), "[5] 8 N^3 <= 2^63 - 1 is not exactly N <= 2^20 - 1");
    Check(4ull * 23170 * 23170 <= 2147483647ull && 4ull * 23171 * 23171 > 2147483647ull, "[5] the narrow limit is not N <= 23170");
    std::cout << "  [5] tau_bound_clause: 0 / 51 / 52 at N = 23170, 23171, 1048575, 1048576, the B_tab edge, the saturation edges; "
              << "clause (e): narrow iff N <= 23170 (B = 4N^2); wide (2 N B <= 2^63 - 1, B = 4 N^2) iff N <= 1048575; "
                 "largest prime 1048573" << std::endl;
}

// ---- (ENV) HECKE_WIDE_TAU_CELLS=<ell>[,<ell>...] (read-only; slow name only; REPLACES the slow arms) ------------------
// The tau table of U_ell (p = ell, n = 0) at each listed prime: certify(), the tier, B_tab, B, the escalations, seconds.
// A measurement hook for the report's B_tab / reach table; never writes anything.
void ArmEnvCells(const SimplicialComplex& K, const SL2ZTessellationTables& model, const std::string& env) {
    std::vector<uint64_t> ells;
    std::string item;
    for (size_t i = 0; i <= env.size(); ++i) {
        if (i == env.size() || env[i] == ',') {
            if (!item.empty()) {
                bool digits = item.size() <= 10;
                for (char ch : item) digits = digits && ch >= '0' && ch <= '9';
                const uint64_t v = digits ? std::stoull(item) : 0;
                Check(digits && v >= 5 && v < (1ull << 31), "(ENV) HECKE_WIDE_TAU_CELLS: '" + item + "' is not an integer in [5, 2^31)");
                if (digits && v >= 5 && v < (1ull << 31)) ells.push_back(v);
            }
            item.clear();
        } else {
            item.push_back(env[i]);
        }
    }
    Check(!ells.empty(), "(ENV) HECKE_WIDE_TAU_CELLS is set but names no cell: 0 cells is not a pass");
    uint32_t ok = 0;
    for (const uint64_t ell : ells) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            hecke::reset_wide_product_escalations();
            hecke::reset_exact_tier_escalations();
            const HeckeCosetTables cosets(ell, 1, static_cast<uint32_t>(ell), 0);
            const HeckeTauTable tab(K, model, cosets, static_cast<uint32_t>(ell), 0);
            const uint32_t code = tab.certify();
            Check(code == 0, "(ENV) ell=" + std::to_string(ell) + ": certify() = " + std::to_string(code));
            ok += code == 0 ? 1 : 0;
            std::cout << "WIDE_TAU_CELL ell=" << ell << " certify=" << code << " wide_tier=" << tab.wide_tier()
                      << " B_tab=" << tab.certified_bound_B_tab() << " B=" << tab.certified_bound_B()
                      << " B_tab_over_N=" << static_cast<double>(tab.certified_bound_B_tab()) / static_cast<double>(ell)
                      << " wide_escalations=" << hecke::wide_product_escalations()
                      << " step1_escalations=" << hecke::exact_tier_escalations() << " seconds=" << Seconds(t0) << std::endl;
        } catch (const std::exception& e) {
            Check(false, "(ENV) ell=" + std::to_string(ell) + " threw: " + e.what());
            std::cout << "WIDE_TAU_CELL ell=" << ell << " refused=\"" << e.what() << "\" seconds=" << Seconds(t0) << std::endl;
        }
    }
    std::cout << "EFFECTIVE HeckeWideTauSlowValidation arm=CELLS cells=" << ells.size() << " ok=" << ok << std::endl;
}

uint32_t Run(bool slow) {
    g_checks = 0;
    g_fails = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool lever_on = hecke::tau_wide_tier_enabled();
    std::cout << hecke::tau_wide_tier_effective_line() << std::endl;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);
        if (!slow) {
            ArmHelpers(lever_on);
            ArmForcedWide(K, model, false);
            ArmBoundary(K, model, lever_on);
            ArmOracle(K, model, "[4](b) U_1009 mod 1009 (narrow; Eichler-Shimura, ell > n + 1)", 1009, 1009, 1, false, lever_on);
            ArmOracle(K, model, "[4](c) U_30011 mod 30011", 30011, 30011, 1, true, lever_on);
            ArmClause(K, model, lever_on);
        } else if (const char* env = std::getenv("HECKE_WIDE_TAU_CELLS")) {
            ArmEnvCells(K, model, env);
        } else {
            ArmForcedWide(K, model, true);
            ArmForcedWideAtWide(K, model, lever_on);
            ArmOracle(K, model, "[4](a) T_30011 mod 13", 30011, 13, 1 + powmod<uint64_t>(30011, 11, 13), true, lever_on);
        }
    } catch (const std::exception& e) {
        Check(false, std::string("top-level exception: ") + e.what());
    }
    std::cout << (slow ? "HECKE_WIDE_TAU_SLOW" : "HECKE_WIDE_TAU") << " checks=" << g_checks << " failing=" << g_fails
              << " wide_tier=" << (lever_on ? "on" : "off") << " wall_s=" << Seconds(t0)
              << " verdict=" << (g_fails == 0 ? "OK" : "FAIL") << std::endl;
    return g_fails;
}

} // namespace

uint32_t HeckeWideTauValidation() { return Run(false); }
uint32_t HeckeWideTauSlowValidation() { return Run(true); }
