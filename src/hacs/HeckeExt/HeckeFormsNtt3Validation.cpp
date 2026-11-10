// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 24, FNTT3) HeckeFormsNtt3Validation (standard).
//
// The route under test: the three-prime chunked engine of HeckeFormsHeckeAlgebra (P1 = 5*2^25+1, P2 = 7*2^26+1,
// P3 = 45*2^23+1, three-limb Garner CRT, blocks of 2^22 with the spectral sums per output block), reached by
// MulSeriesFntt3 and by BuildCuspSpace / BuildModularSpace where the two-prime route is not exact (ell >= 2^16).  The
// validation sets ScopedFormsNtt3 itself, so it does not depend on HECKE_FORMS_NTT3.  docs/implementation/slot24/
// FNTT3_predictions.md / FNTT3_report.md.
//
// Grid:
//   (G)  Garner: random x in [0, P1 P2 P3) (u128) and the edges (0, P1 P2 - 1, P1 P2, P1 P2 P3 - 1, ...), the residues
//        r_i = x mod P_i and their lazy representatives r_i + P_i, against x mod ell for ell in {5, 65537, 1000003, 2^27 - 39}.
//   (G2) (FNTT3 audit F2) the ENGINE's own constants (engines at four (ell, N), up to the production layout): the primes, P3's
//        root, Garner's constants, the engine's P1 P2 mod ell and Barrett constant, its layout, each plan's w_L of order exactly L;
//        and the engine's own Garner step on 1006 x per engine (plain and lazy residues) against u128 x mod ell.
//   (A)  products against the naive oracle: ell in {5, 691, 65521, 65537, 100003, 1000003, 134217689}, N in {1024, 1500, 3001},
//        block length 2^22 (nb = 1) and the probe's 2^9 (nb 2 .. 6) and 2^4 (nb 64 .. 188), unequal / short / truncated
//        operands, the square and the all-(ell-1) square, SIMD and scalar butterflies.
//   (I)  identity with the two-prime engine wherever that is exact: N = 600001 at ell in {40009, 65521}, block 2^22 and 2^17.
//   (B)  past the two-prime CRT bound (N (ell-1)^2 >= P1 P2; FnttApplicable false, Fntt3Applicable true):
//        the all-(ell-1) square == (n + 1) mod ell (ell = 1000003, N = 200000; ell = 134217689, N = 2^20 + 3), and random
//        products at 300 sampled coefficients against an exact u128 dot product (ell = 1000003, N = 200000; blocks 2^22, 2^14),
//        SIMD and scalar; (FNTT3 audit F1) the production block 2^22 with nb = 2 (ell = 1000003, N = 2^22 + 5003) and nb = 4
//        (ell = 134217689, N = 3 2^22 + 17; the third limb live): the closed form and sampled exact dot products; E_4^2 == E_8 and E_4 E_6 == E_10 (sigma_7 / sigma_9 by a sieve) at ell = 1000003, N = 200000.
//   (S)  spaces: the forced three-prime route (ScopedFormsNtt3Probe force) == the two-prime route byte for byte on M_k and
//        S_k at the sat route's precision for (12, 1009), (14, 691), (60, 691), (102, 10007), (300, 43) (k' = 0 .. 14, d = 0
//        included), blocks 2^22 and 2^12; past the bound, the default route at (k, ell, N) in {(26, 1000003, 200000),
//        (36, 1000003, 200000), (30, 300007, 10^6)} takes the three-prime engine (counters), equals the chunked layout 2^15 byte for byte, its
//        first 2501 coefficients equal the HeckeSturmBound E_4 / E_6 / Delta construction under the naive product, and
//        T_2, T_3, T_5, T_7 are in the space (HeckeOnCusp's residual check over every n p < N) on both HeckeOnCusp loops.
//   (R)  the refusal lifted / the rollback: FormsSpaceSupported at (1000, 100003), (466, 100003), (10, 1000003), (300, 1000003)
//        true by default and false under ScopedFormsNtt3(false); the HECKE_FORMS_NTT_CACHE=0 / HECKE_FORMS_FASTMUL=0 scopes still
//        refuse; under the rollback BuildModularSpace throws naming FormsSpaceSupported and builds no three-prime space;
//        ComputeSaturationDefectCell(10, 1000003): the forms refusal no longer speaks by default (the KMAX one does), and
//        speaks again under the rollback; spaces admitted before FNTT3 keep their route (two-prime / legacy counters).
//   (C)  non-vacuity: three-prime spaces, multi-block engines, forward / inverse / pointwise counters moved.

#include "HeckeFormsNtt3Validation.h"

#include "HeckeFormsHeckeAlgebra.h"
#include "HeckeSaturationDefect.h"
#include "HeckeSturmBound.h"
#include "../Hecke/HeckeSettings.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fm = hecke_ext::forms;
namespace sd = hecke_ext::satdefect;
using fm::QSeries;
using u128 = unsigned __int128;

constexpr uint32_t kP1 = 167772161u, kP2 = 469762049u, kP3 = 377487361u;

oc_util::Checker Check("  FAIL: ", false);

QSeries Random(std::mt19937_64& rng, size_t len, uint32_t ell) {
    QSeries s(len);
    for (auto& x : s) x = static_cast<uint32_t>(rng() % ell);
    return s;
}

bool SameMatrix(const hecke_ext::ExtMatrix& a, const hecke_ext::ExtMatrix& b) {
    return a.rows == b.rows && a.cols == b.cols && a.p == b.p && a.data == b.data;
}

bool SameSpace(const fm::CuspSpace& a, const fm::CuspSpace& b) {
    return a.k == b.k && a.ell == b.ell && a.precision == b.precision && a.d == b.d && a.k_prime == b.k_prime &&
           a.with_constant == b.with_constant && a.basis == b.basis;
}

// c_n = sum_{i <= n, i < |a|, n - i < |b|} a_i b_{n-i} mod ell, exactly (u128 accumulator)
uint32_t DotCoeff(const QSeries& a, const QSeries& b, uint32_t ell, size_t n) {
    u128 acc = 0;
    const size_t lo = n + 1 > b.size() ? n + 1 - b.size() : 0, hi = std::min(n + 1, a.size());
    for (size_t i = lo; i < hi; ++i) acc += static_cast<uint64_t>(a[i]) * b[n - i];
    return static_cast<uint32_t>(acc % ell);
}

std::vector<uint64_t> PrimesUpTo(uint32_t b) {
    std::vector<uint64_t> out;
    for (uint32_t p = 2; p <= b; ++p) {
        bool pr = true;
        for (uint32_t q = 2; q * q <= p; ++q)
            if (p % q == 0) pr = false;
        if (pr) out.push_back(p);
    }
    return out;
}

uint32_t SatN(uint32_t k, uint32_t ell) {   // HeckeSaturationDefect.cpp BuildForms
    std::vector<uint64_t> P = PrimesUpTo(std::max(31u, k / 12));
    if (std::find(P.begin(), P.end(), ell) == P.end()) P.push_back(ell);
    return 2 * (k / 12 + 1) * static_cast<uint32_t>(P.back()) + 2;
}

// ---------------------------------------------------------------------------
void ArmG(std::mt19937_64& rng) {
    std::cout << "-- (G) three-limb Garner against u128 x mod ell\n";
    const u128 P12 = static_cast<u128>(kP1) * kP2, P123 = P12 * kP3;
    std::vector<u128> xs = {0, 1, P12 - 1, P12, P12 + 1, static_cast<u128>(kP1) * kP3, P123 - 1, P123 - P12, P123 / 2};
    for (int i = 0; i < 4000; ++i) {
        const u128 r = (static_cast<u128>(rng()) << 64) | rng();
        xs.push_back(r % P123);
    }
    for (uint32_t ell : {5u, 65537u, 1000003u, 134217689u}) {
        bool ok = true;
        for (const u128 x : xs) {
            const uint32_t r1 = static_cast<uint32_t>(x % kP1), r2 = static_cast<uint32_t>(x % kP2), r3 = static_cast<uint32_t>(x % kP3);
            const uint32_t want = static_cast<uint32_t>(x % ell);
            ok = ok && fm::Fntt3GarnerForValidation(r1, r2, r3, ell) == want &&
                 fm::Fntt3GarnerForValidation(r1 + kP1, r2 + kP2, r3 + kP3, ell) == want &&   // the lazy [0, 2P) range
                 fm::Fntt3GarnerForValidation(r1, r2 + kP2, r3, ell) == want;
        }
        Check(ok, "(G) Garner3 == x mod ell for " + std::to_string(xs.size()) + " x in [0, P1 P2 P3) at ell=" + std::to_string(ell));
    }
}

uint64_t PowModU(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    b %= m;
    for (; e; e >>= 1, b = static_cast<uint64_t>(static_cast<u128>(b) * b % m))
        if (e & 1) r = static_cast<uint64_t>(static_cast<u128>(r) * b % m);
    return r;
}

// (G2, FNTT3 audit F2) the ENGINE's own constants (read from engines built at (ell, N)) against values derived here, and the
// engine's own Garner step (its P1 P2 mod ell and Barrett constant) against u128 x mod ell
void ArmG2(std::mt19937_64& rng) {
    std::cout << "-- (G2) the engine's own constants and Garner step\n";
    const u128 P12 = static_cast<u128>(kP1) * kP2, P123 = P12 * kP3;
    struct At {
        uint32_t ell, N;
    };
    for (const At& a : {At{5, 1024}, At{65537, 3001}, At{1000003, 200000}, At{134217689, (1u << 22) + 5}}) {
        const fm::Fntt3EngineConstants c = fm::Fntt3EngineConstantsForValidation(a.ell, a.N);
        const std::string w = "ell=" + std::to_string(a.ell) + " N=" + std::to_string(a.N);
        Check(c.p1 == kP1 && c.p2 == kP2 && c.p3 == kP3 && c.g3 == 7, "(G2) the primes and P3's root at " + w);
        Check(static_cast<uint64_t>(kP1) * c.inv_p1_mod_p2 % kP2 == 1 && c.p12_mod_p3 == static_cast<uint32_t>(P12 % kP3) &&
                  static_cast<uint64_t>(c.p12_mod_p3) * c.inv_p12_mod_p3 % kP3 == 1,
              "(G2) Garner's constants P1^{-1} mod P2, P1 P2 mod P3, (P1 P2)^{-1} mod P3 at " + w);
        Check(c.p12_mod_ell == static_cast<uint32_t>(P12 % a.ell) && c.barrett == ~uint64_t{0} / a.ell,
              "(G2) the engine's P1 P2 mod ell and Barrett constant at " + w);
        const uint64_t L = uint64_t{1} << c.lg;
        bool roots = true;   // w_L has order exactly L mod each prime: w^L = 1, w^{L/2} = -1
        for (const auto& pw : {std::pair<uint64_t, uint32_t>{kP1, c.w1}, {kP2, c.w2}, {kP3, c.w3}})
            roots = roots && PowModU(pw.second, L, pw.first) == 1 && PowModU(pw.second, L / 2, pw.first) == pw.first - 1;
        Check(roots, "(G2) each plan's w_L is a primitive L-th root (L = 2^" + std::to_string(c.lg) + ") at " + w);
        const bool lay = (2 * static_cast<uint64_t>(a.N) - 1 <= (uint64_t{1} << 23))
                             ? (c.nb == 1 && c.B == a.N)
                             : (c.B == (1u << 22) && c.lg == 23 && c.nb == (a.N + c.B - 1) / c.B);
        Check(lay, "(G2) the engine's layout (B, nb, lg) at " + w);
        std::vector<u128> xs = {0, 1, P12 - 1, P12, P123 - 1, P123 - P12};
        for (int i = 0; i < 1000; ++i) xs.push_back(((static_cast<u128>(rng()) << 64) | rng()) % P123);
        std::vector<uint32_t> r;
        for (size_t i = 0; i < xs.size(); ++i) {
            const uint32_t lazy = (i % 2 == 1) ? 1u : 0u;   // every other triple in the lazy [P, 2P) representatives
            r.push_back(static_cast<uint32_t>(xs[i] % kP1) + lazy * kP1);
            r.push_back(static_cast<uint32_t>(xs[i] % kP2) + lazy * kP2);
            r.push_back(static_cast<uint32_t>(xs[i] % kP3) + lazy * kP3);
        }
        const std::vector<uint32_t> got = fm::Fntt3EngineGarnerForValidation(a.ell, a.N, r);
        bool ok = got.size() == xs.size();
        for (size_t i = 0; ok && i < xs.size(); ++i) ok = got[i] == static_cast<uint32_t>(xs[i] % a.ell);
        Check(ok, "(G2) the engine's Garner step == x mod ell on " + std::to_string(xs.size()) + " x at " + w);
    }
}

void ArmA(std::mt19937_64& rng) {
    std::cout << "-- (A) three-prime products == naive (blocks 2^22, 2^9, 2^4; SIMD and scalar)\n";
    for (uint32_t ell : {5u, 691u, 65521u, 65537u, 100003u, 1000003u, 134217689u}) {
        for (uint32_t N : {1024u, 1500u, 3001u}) {
            const QSeries a = Random(rng, N, ell), b = Random(rng, N - 5, ell), c = Random(rng, 700, ell), top(N, ell - 1);
            const QSeries ab = fm::MulSeriesNaive(a, b, ell, N), ac = fm::MulSeriesNaive(a, c, ell, N), cc = fm::MulSeriesNaive(c, c, ell, N),
                          aa = fm::MulSeriesNaive(a, a, ell, N), tt = fm::MulSeriesNaive(top, top, ell, N);
            const QSeries ab_trunc = fm::MulSeriesNaive(a, b, ell, N / 2 + 1);
            for (uint32_t lg : {22u, 9u, 4u}) {
                const fm::ScopedFormsNtt3Probe probe(lg, false);
                const std::string w = "ell=" + std::to_string(ell) + " N=" + std::to_string(N) + " block=2^" + std::to_string(lg) +
                                      " nb=" + std::to_string(fm::Fntt3Blocks(N));
                Check(fm::Fntt3Applicable(ell, N), "(A) Fntt3Applicable at " + w);
                if (lg == 22) Check(fm::Fntt3Blocks(N) == 1, "(A) one block at " + w);
                else Check(fm::Fntt3Blocks(N) == (N + (1u << lg) - 1) >> lg, "(A) nb = ceil(N / B) at " + w);
                for (bool simd : {true, false}) {
                    const fm::ScopedFormsNttSimd s(simd);
                    const std::string ws = w + (simd ? " simd" : " scalar");
                    Check(fm::MulSeriesFntt3(a, b, ell, N) == ab, "(A) a*b != naive at " + ws);
                    Check(fm::MulSeriesFntt3(a, c, ell, N) == ac && fm::MulSeriesFntt3(c, a, ell, N) == ac, "(A) a*c, c*a (|c| = 700) != naive at " + ws);
                    Check(fm::MulSeriesFntt3(c, c, ell, N) == cc, "(A) c*c (short square) != naive at " + ws);
                    Check(fm::MulSeriesFntt3(a, a, ell, N) == aa, "(A) the square a*a (one object) != naive at " + ws);
                    Check(fm::MulSeriesFntt3(top, top, ell, N) == tt, "(A) the all-(ell-1) square != naive at " + ws);
                    if (N / 2 + 1 >= 1024)
                        Check(fm::MulSeriesFntt3(a, b, ell, N / 2 + 1) == ab_trunc, "(A) a*b truncated to N/2+1 != naive at " + ws);
                }
            }
        }
    }
}

void ArmI(std::mt19937_64& rng) {
    std::cout << "-- (I) three-prime == two-prime engine where both are exact\n";
    for (uint32_t ell : {40009u, 65521u}) {
        const uint32_t N = 600001;
        const QSeries a = Random(rng, N, ell), b = Random(rng, N - 77, ell), top(N, ell - 1);
        Check(fm::FnttApplicable(ell, N), "(I) the two-prime route applies at ell=" + std::to_string(ell));
        const QSeries ab2 = fm::MulSeriesFntt(a, b, ell, N), tt2 = fm::MulSeriesFntt(top, top, ell, N);
        for (uint32_t lg : {22u, 17u}) {
            const fm::ScopedFormsNtt3Probe probe(lg, false);
            const std::string w = "ell=" + std::to_string(ell) + " N=600001 block=2^" + std::to_string(lg) + " nb=" + std::to_string(fm::Fntt3Blocks(N));
            Check(fm::MulSeriesFntt3(a, b, ell, N) == ab2, "(I) a*b three-prime != two-prime at " + w);
            Check(fm::MulSeriesFntt3(top, top, ell, N) == tt2, "(I) all-(ell-1) square three-prime != two-prime at " + w);
        }
    }
}

// sigma_e(n) mod ell for n < N (a sieve)
QSeries SigmaMod(uint32_t e, uint32_t ell, uint32_t N) {
    QSeries s(N, 0);
    for (uint32_t d = 1; d < N; ++d) {
        uint64_t pw = 1;
        for (uint32_t i = 0; i < e; ++i) pw = pw * d % ell;
        for (uint32_t m = d; m < N; m += d) {
            const uint64_t v = s[m] + pw;
            s[m] = static_cast<uint32_t>(v >= ell ? v - ell : v);
        }
    }
    return s;
}

void ArmB(std::mt19937_64& rng) {
    std::cout << "-- (B) past the two-prime CRT bound: closed forms, exact dot products, E_4^2 = E_8, E_4 E_6 = E_10\n";
    struct Edge {
        uint32_t ell, N, lg;
    };
    for (const Edge& e : {Edge{1000003u, 200000u, 22u}, Edge{1000003u, 200000u, 14u}, Edge{134217689u, (1u << 20) + 3, 22u}}) {
        const fm::ScopedFormsNtt3Probe probe(e.lg, false);
        const std::string w = "ell=" + std::to_string(e.ell) + " N=" + std::to_string(e.N) + " block=2^" + std::to_string(e.lg) +
                              " nb=" + std::to_string(fm::Fntt3Blocks(e.N));
        Check(!fm::FnttApplicable(e.ell, e.N) && fm::Fntt3Applicable(e.ell, e.N),
              "(B) past the two-prime bound, inside the three-prime one at " + w);
        const QSeries top(e.N, e.ell - 1);
        for (bool simd : {true, false}) {
            const fm::ScopedFormsNttSimd s(simd);
            const QSeries sq = fm::MulSeriesFntt3(top, top, e.ell, e.N);
            bool ok = sq.size() == e.N;
            for (uint32_t n = 0; ok && n < e.N; ++n) ok = sq[n] == (n + 1) % e.ell;
            Check(ok, std::string("(B) all-(ell-1) square == (n + 1) mod ell (") + (simd ? "simd" : "scalar") + ") at " + w);
        }
    }
    {
        const uint32_t ell = 1000003, N = 200000;
        const QSeries a = Random(rng, N, ell), b = Random(rng, N - 3, ell);
        std::vector<size_t> ns = {0, 1, 4095, 4096, 16383, 16384, 16385, 32767, 32768, 99999, 100000, N - 2, N - 1};
        for (int i = 0; i < 287; ++i) ns.push_back(rng() % N);
        std::vector<uint32_t> want;
        for (size_t n : ns) want.push_back(DotCoeff(a, b, ell, n));
        for (uint32_t lg : {22u, 14u}) {
            const fm::ScopedFormsNtt3Probe probe(lg, false);
            for (bool simd : {true, false}) {
                const fm::ScopedFormsNttSimd s(simd);
                const QSeries c = fm::MulSeriesFntt3(a, b, ell, N);
                bool ok = c.size() == N;
                for (size_t i = 0; ok && i < ns.size(); ++i) ok = c[ns[i]] == want[i];
                Check(ok, "(B) random a*b at " + std::to_string(ns.size()) + " coefficients == the exact u128 dot product (ell=1000003 N=200000 block=2^" +
                              std::to_string(lg) + (simd ? " simd)" : " scalar)"));
            }
        }
        // (FNTT3 audit F1) the production block 2^22 with nb = 2 and nb = 4 (the layout of (466, 100003) and (1000, *)): the
        // all-(ell-1) square's closed form, and a random product at sampled coefficients against exact u128 dot products; at
        // ell = 134217689 the coefficients reach ~2.3e23 > P1 P2, so the third limb is live
        struct Prod {
            uint32_t ell, N, nb, samples;
        };
        for (const Prod& e : {Prod{1000003u, (1u << 22) + 5003, 2, 60}, Prod{134217689u, 3 * (1u << 22) + 17, 4, 24}}) {
            const std::string w = "ell=" + std::to_string(e.ell) + " N=" + std::to_string(e.N) + " block=2^22";
            Check(fm::Fntt3Blocks(e.N) == e.nb && !fm::FnttApplicable(e.ell, e.N) && fm::Fntt3Applicable(e.ell, e.N),
                  "(B) production layout nb=" + std::to_string(e.nb) + ", past the two-prime bound, at " + w);
            {
                const QSeries top(e.N, e.ell - 1);
                const QSeries sq = fm::MulSeriesFntt3(top, top, e.ell, e.N);
                bool ok = sq.size() == e.N;
                for (uint32_t n = 0; ok && n < e.N; ++n) ok = sq[n] == (n + 1) % e.ell;
                Check(ok, "(B) all-(ell-1) square == (n + 1) mod ell at " + w + " nb=" + std::to_string(e.nb));
            }
            const QSeries x = Random(rng, e.N, e.ell), y = Random(rng, e.N - 11, e.ell);
            const QSeries c = fm::MulSeriesFntt3(x, y, e.ell, e.N);
            std::vector<size_t> ns = {0, (1u << 22) - 1, 1u << 22, (1u << 23) - 1, 1u << 23, size_t{e.N} - 1};
            for (uint32_t i = 0; i < e.samples; ++i) ns.push_back(rng() % e.N);
            bool ok = c.size() == e.N;
            for (size_t i = 0; ok && i < ns.size(); ++i)
                if (ns[i] < e.N) ok = c[ns[i]] == DotCoeff(x, y, e.ell, ns[i]);
            Check(ok, "(B) random x*y at " + std::to_string(ns.size()) + " coefficients (block edges included) == exact u128 dot products at " + w +
                          " nb=" + std::to_string(e.nb));
        }
        // the Eisenstein identities: E_8 = 1 + 480 sum sigma_7(n) q^n, E_10 = 1 - 264 sum sigma_9(n) q^n
        const QSeries E4 = fm::EisensteinSeries(4, ell, N), E6 = fm::EisensteinSeries(6, ell, N);
        const QSeries s7 = SigmaMod(7, ell, N), s9 = SigmaMod(9, ell, N);
        QSeries E8(N), E10(N);
        E8[0] = E10[0] = 1;
        for (uint32_t n = 1; n < N; ++n) {
            E8[n] = static_cast<uint32_t>(480ull * s7[n] % ell);
            E10[n] = static_cast<uint32_t>((ell - 264ull * s9[n] % ell) % ell);
        }
        for (uint32_t lg : {22u, 15u}) {
            const fm::ScopedFormsNtt3Probe probe(lg, false);
            Check(fm::MulSeriesFntt3(E4, E4, ell, N) == E8, "(B) E_4^2 == E_8 at ell=1000003 N=200000 block=2^" + std::to_string(lg));
            Check(fm::MulSeriesFntt3(E4, E6, ell, N) == E10, "(B) E_4 E_6 == E_10 at ell=1000003 N=200000 block=2^" + std::to_string(lg));
        }
    }
}

void ArmS() {
    std::cout << "-- (S) spaces: forced three-prime == two-prime; past the bound == HeckeSturmBound, T_p in the space\n";
    struct Cell {
        uint32_t k, ell;
    };
    for (const Cell& c : {Cell{12, 1009}, Cell{14, 691}, Cell{60, 691}, Cell{102, 10007}, Cell{300, 43}}) {
        const uint32_t N = std::max<uint32_t>(SatN(c.k, c.ell), 1024);
        const std::string tag = "(S) (k, ell) = (" + std::to_string(c.k) + ", " + std::to_string(c.ell) + ") N=" + std::to_string(N);
        Check(fm::FnttApplicable(c.ell, N), tag + ": the two-prime route applies");
        const fm::FormsNttCounters n0 = fm::FormsNttCountersNow();
        const fm::CuspSpace M2 = fm::BuildModularSpace(c.k, c.ell, N), S2 = fm::BuildCuspSpace(c.k, c.ell, N);
        Check(fm::FormsNttCountersNow().fntt_spaces == n0.fntt_spaces + 2, tag + ": the default spaces took the two-prime route");
        for (uint32_t lg : {22u, 12u}) {
            const fm::ScopedFormsNtt3Probe probe(lg, true);
            const fm::FormsNtt3Counters t0 = fm::FormsNtt3CountersNow();
            const fm::CuspSpace M3 = fm::BuildModularSpace(c.k, c.ell, N), S3 = fm::BuildCuspSpace(c.k, c.ell, N);
            const std::string w = tag + " block=2^" + std::to_string(lg) + " nb=" + std::to_string(fm::Fntt3Blocks(N));
            Check(fm::FormsNtt3CountersNow().spaces == t0.spaces + 2, w + ": the forced spaces took the three-prime route");
            Check(SameSpace(M3, M2) && SameSpace(S3, S2), w + ": M_k and S_k three-prime == two-prime, byte for byte");
        }
    }
    struct Big {
        uint32_t k, ell, N;
    };
    for (const Big& c : {Big{26, 1000003, 200000}, Big{36, 1000003, 200000}, Big{30, 300007, 1000000}}) {
        const uint32_t N = c.N;
        const std::string tag = "(S) (k, ell) = (" + std::to_string(c.k) + ", " + std::to_string(c.ell) + ") N=" + std::to_string(N);
        Check(!fm::FnttApplicable(c.ell, N) && fm::FormsSpaceSupported(c.ell, N), tag + ": past the two-prime bound, supported");
        const fm::FormsNtt3Counters t0 = fm::FormsNtt3CountersNow();
        const fm::CuspSpace M = fm::BuildModularSpace(c.k, c.ell, N);
        Check(fm::FormsNtt3CountersNow().spaces == t0.spaces + 1, tag + ": the default space took the three-prime route");
        {
            const fm::ScopedFormsNtt3Probe probe(15, false);
            Check(SameSpace(fm::BuildModularSpace(c.k, c.ell, N), M),
                  tag + ": the chunked layout (block 2^15, nb=" + std::to_string(fm::Fntt3Blocks(N)) + ") == the one-block layout");
        }
        // the first 2501 coefficients against the HeckeSturmBound construction under the naive product
        const uint32_t n0 = 2501;
        const QSeries E4 = hecke_ext::EisensteinSeries(4, n0, c.ell), E6 = hecke_ext::EisensteinSeries(6, n0, c.ell),
                      D = hecke_ext::DeltaSeries(n0, c.ell);
        auto mul = [&](const QSeries& x, const QSeries& y) { return fm::MulSeriesNaive(x, y, c.ell, n0); };
        QSeries base(n0, 0);
        base[0] = 1;
        const uint32_t kp = M.k_prime, d = M.d;
        if (kp == 4) base = E4;
        else if (kp == 6) base = E6;
        else if (kp == 8) base = mul(E4, E4);
        else if (kp == 10) base = mul(E4, E6);
        else if (kp == 14) base = mul(mul(E4, E4), E6);
        const QSeries E6sq = mul(E6, E6);
        bool ok = M.basis.size() == d + 1;
        QSeries Dj(n0, 0);
        Dj[0] = 1;
        for (uint32_t j = 0; ok && j <= d; ++j) {
            QSeries G = base;
            for (uint32_t i = 0; i < d - j; ++i) G = mul(G, E6sq);
            const QSeries f = mul(Dj, G);
            ok = M.basis[j].size() == N && std::equal(f.begin(), f.end(), M.basis[j].begin());
            Dj = mul(Dj, D);
        }
        Check(ok, tag + ": the first 2501 coefficients of the M_k basis == the HeckeSturmBound construction");
        // T_p is in the space to the full precision (HeckeOnCusp's residual check over every n p < N throws otherwise)
        std::vector<hecke_ext::ExtMatrix> Tf, Tl;
        std::string err;
        try {
            for (bool on : {true, false}) {
                const fm::ScopedFormsNttCache s(on);
                for (uint32_t p : {2u, 3u, 5u, 7u}) (on ? Tf : Tl).push_back(fm::HeckeOnCusp(M, p));
            }
        } catch (const std::exception& e) {
            err = e.what();
        }
        bool same = err.empty() && Tf.size() == 4 && Tl.size() == 4;
        for (size_t i = 0; same && i < Tf.size(); ++i) same = SameMatrix(Tf[i], Tl[i]);
        Check(same, tag + ": T_2, T_3, T_5, T_7 are in the space (both HeckeOnCusp loops, equal) " + err);
    }
}

void ArmR() {
    std::cout << "-- (R) the FNTT refusal lifted by default; the rollback HECKE_FORMS_NTT3=0 restores it\n";
    struct Cell {
        uint32_t n, ell;
    };
    const Cell cells[] = {{1000, 100003}, {466, 100003}, {10, 1000003}, {300, 1000003}};
    for (const Cell& c : cells) {
        const uint32_t N = fm::SatFormsPrecision(c.n + 2, c.ell);
        const std::string w = "(n, ell) = (" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ") N=" + std::to_string(N);
        Check(!fm::FnttApplicable(c.ell, N) && fm::Fntt3Applicable(c.ell, N), "(R) past the two-prime bound, three-prime exact at " + w);
        Check(fm::FormsSpaceSupported(c.ell, N), "(R) supported by default at " + w);
        {
            const fm::ScopedFormsNtt3 off(false);
            Check(!fm::FormsSpaceSupported(c.ell, N), "(R) refused under the rollback at " + w);
        }
        {
            const fm::ScopedFormsNttCache off(false);
            Check(!fm::FormsSpaceSupported(c.ell, N), "(R) refused under HECKE_FORMS_NTT_CACHE=0 at " + w);
        }
        {
            const fm::ScopedFormsFastMul off(false);
            Check(!fm::FormsSpaceSupported(c.ell, N), "(R) refused under HECKE_FORMS_FASTMUL=0 at " + w);
        }
    }
    {   // the target cell's layout
        const uint32_t N = fm::SatFormsPrecision(1002, 100003);
        Check(N == 16800506 && fm::Fntt3Blocks(N) == 5, "(R) (1000, 100003): N = 16800506, nb = 5 blocks of 2^22");
    }
    {   // under the rollback a space past the bound throws (naming the predicate) and no three-prime space is built
        const fm::ScopedFormsNtt3 off(false);
        const fm::FormsNtt3Counters t0 = fm::FormsNtt3CountersNow();
        std::string msg;
        try {
            (void)fm::BuildModularSpace(26, 1000003, 200000);
        } catch (const std::logic_error& e) {
            msg = e.what();
        }
        Check(msg.find("FormsSpaceSupported") != std::string::npos && fm::FormsNtt3CountersNow().spaces == t0.spaces,
              "(R) rollback: BuildModularSpace(26, 1000003, 200000) refused by name, no three-prime space: '" + msg + "'");
    }
    // the sat route's pre-run refusal: by default the forms check passes, so the KMAX refusal (ell^6 >= 2^64) speaks; under the
    // rollback the forms refusal speaks again -- both at once (before the module run)
    for (bool on : {true, false}) {
        const fm::ScopedFormsNtt3 s(on);
        const auto t0 = std::chrono::steady_clock::now();
        std::string msg;
        try {
            sd::SatOptions o;
            (void)sd::ComputeSaturationDefectCell(10, 1000003, o);
        } catch (const hecke::NotImplementedInThisBuild& e) {
            msg = e.what();
        } catch (const std::exception& e) {
            msg = std::string("WRONG KIND: ") + e.what();
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const bool forms_refusal = msg.find("FormsSpaceSupported") != std::string::npos;
        Check(!msg.empty() && msg.find("WRONG KIND") == std::string::npos && forms_refusal == !on && sec < 10.0,
              std::string("(R) ComputeSaturationDefectCell(10, 1000003) ") + (on ? "default: not the forms refusal" : "rollback: the forms refusal") +
                  ", at once: '" + msg + "'");
    }
    // spaces admitted before FNTT3 keep their route: two-prime at (24, 1009) sat precision; legacy at ell >= 2^16, N < 1024
    {
        const fm::FormsNttCounters n0 = fm::FormsNttCountersNow();
        const fm::FormsNtt3Counters t0 = fm::FormsNtt3CountersNow();
        (void)fm::BuildModularSpace(26, 1009, fm::SatFormsPrecision(26, 1009));
        (void)fm::BuildModularSpace(26, 1000003, 1000);
        const fm::FormsNttCounters n1 = fm::FormsNttCountersNow();
        Check(n1.fntt_spaces == n0.fntt_spaces + 1 && n1.legacy_spaces == n0.legacy_spaces + 1 && fm::FormsNtt3CountersNow().spaces == t0.spaces,
              "(R) admitted spaces keep their route (two-prime at (26, 1009); legacy at (26, 1000003), N = 1000)");
    }
}

}  // namespace

uint32_t HeckeFormsNtt3Validation() {
    Check.reset();
    std::cout << "=== HeckeFormsNtt3Validation (slot-24 FNTT3: the three-prime chunked forms NTT == independent oracles) ===\n";
    std::cout << fm::FormsNtt3EffectiveLine() << "\n";
    const fm::ScopedFormsNtt3 on(true);   // the default arm, independent of HECKE_FORMS_NTT3
    const fm::ScopedFormsNttCache cache(true);
    const fm::ScopedFormsFastMul fast(true);
    const fm::FormsNtt3Counters c0 = fm::FormsNtt3CountersNow();
    std::mt19937_64 rng(0x46e7733ull);
    const auto t0 = std::chrono::steady_clock::now();
    ArmG(rng);
    ArmG2(rng);
    ArmA(rng);
    ArmI(rng);
    ArmB(rng);
    ArmS();
    ArmR();
    const fm::FormsNtt3Counters c1 = fm::FormsNtt3CountersNow();
    std::cout << "-- (C) non-vacuity: spaces=" << (c1.spaces - c0.spaces) << " forward=" << (c1.forward - c0.forward)
              << " inverse=" << (c1.inverse - c0.inverse) << " pointwise_blocks=" << (c1.pointwise_blocks - c0.pointwise_blocks)
              << " multiblock_engines=" << (c1.multiblock_engines - c0.multiblock_engines) << "\n";
    Check(c1.spaces > c0.spaces && c1.multiblock_engines > c0.multiblock_engines && c1.forward > c0.forward && c1.inverse > c0.inverse &&
              c1.pointwise_blocks > c1.inverse - c0.inverse + c0.pointwise_blocks,
          "(C) three-prime spaces, multi-block engines and block-pair products were exercised");
    std::cout << fm::FormsNtt3EffectiveLine() << "\n";
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeFormsNtt3Validation: checks=" << Check.checks << " failing=" << Check.fails << " seconds=" << sec << "\n";
    return Check.fails;
}
