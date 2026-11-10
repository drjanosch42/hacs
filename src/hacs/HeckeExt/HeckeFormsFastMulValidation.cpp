// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, G2-X) HeckeFormsFastMulValidation (standard).
//
// The route under test: hecke_ext::forms::MulSeriesFast (NTT over P1 = 5*2^25+1, P2 = 7*2^26+1, CRT, then mod ell;
// the delayed-reduction schoolbook MulSeriesSchoolbook below kFastMulSchoolbookMin) and the E_4 / E_6 divisor sieve
// of EisensteinSeries.  The oracle: MulSeriesNaive and the trial-division loop, i.e. the routes through slot 23,
// selected by ScopedFormsFastMul(false) -- the same switch as the rollback arm HECKE_FORMS_FASTMUL=0.  Both scopes
// are set here, so the validation does not depend on the environment.  Every comparison is byte for byte.
//
// Grid:
//   (P1) products on random inputs: ell in {2, 3, 5, 7, 691, 1009, 10007, 65521 (the largest prime below the word
//        limit 2^16)}, lengths around the schoolbook threshold (63, 64, 65) and around powers of two of the transform
//        (|a| + |b| - 1 = 2^j - 1, 2^j, 2^j + 1), unequal lengths, N below / at / above |a| + |b| - 1, N < |a|;
//        fast = schoolbook = naive.  Each product also through MulSeries under both scopes.
//   (P2) extreme inputs: every entry ell - 1 (the CRT bound's worst case) at ell = 65521, length 20000; sparse inputs
//        (one non-zero entry, the pentagonal series); a zero series; empty series; the square (one object passed
//        twice) and equal-content inputs (the square branch); N = 0, 1.
//   (P3) the fallback: ell = 65537 (>= 2^16) runs the naive route (fallback counter moves), values equal.
//   (P5) (G2-X audit M1 / M2) the transform-length guard and the chunked product: |a| = 2^25 - 63, |b| = 65 at
//        ell = 65521, N = 2^25 + 1 (|a| + |b| - 1 = 2^25 + 1 > 2^25): the fallback counter moves, the NTT counter does
//        not, and the value equals the schoolbook (~2 s, ~1 GiB); MulSeriesChunked at small block sizes (chunk_lg in
//        {1, 3, 6, 9}) against the naive oracle on random, unequal, truncated and empty inputs (the block assembly).
//   (P4) an independent identity at a length the oracle cannot reach in this tier: E_4 * E_4 = E_8 = 1 + 480 sum
//        sigma_7(n) q^n mod ell at N = 300001 (transform length 2^20) for ell = 65521 and 1009; and
//        E_4^3 - E_6^2 = 1728 Delta at N = 100001, ell = 10007.
//   (E)  EisensteinSeries(4 | 6): the sieve against the trial-division loop at N in {1, 2, 3, 97, 5000} and every ell of (P1) >= 5.
//   (F)  the real cross-route objects at the sat route's precision N = 2 (k/12 + 1) pmax + 2 (BuildForms): (n, ell) in
//        {(10, 5), (40, 5), (108, 7), (300, 43), (60, 691), (24, 1009)}: BuildModularSpace and BuildCuspSpace basis
//        series and HeckeOnCusp(M, p) for every generator prime, fast against naive, byte for byte.
//   (S)  whole SAT_DEFECT cells (the forms cross-route inside): (40, 5) and (24, 1009), every printed line equal
//        (timings excepted, which are not printed by SatDefectLine), with the forms cross-route having run.
//   (C)  non-vacuity: the fast scope took the NTT >= 1 times, the schoolbook >= 1, the fallback >= 1, the chunked
//        product >= 1, the sieve >= 1;
//        the naive scope took the naive product and the divisor loop >= 1, and no fast route.
//   (slot-24 FNTT, after the arms above so their random inputs are unchanged)
//   (N1) the FNTT engine's product (MulSeriesFntt) == schoolbook, ell up to 65521, SIMD and scalar butterflies; the CRT worst
//        case; == MulSeriesFast at N = 600001; the FnttApplicable bounds.
//   (N2) ten sat-precision cells (k' in {0, 4, 6, 8, 10, 14}, d = 0 included, ell up to 65521): FNTT (SIMD, scalar) == G2-X
//        (ScopedFormsNttCache(false)) on the M_k / S_k basis and T_p (delayed == legacy HeckeOnCusp loop); the M_k basis ==
//        the HeckeSturmBound E_4 / E_6 / Delta construction.  (N3) both HeckeOnCusp loops throw the same message.
//   (N5) ell in {65537, 100003, 131071, 1000003}: FNTT == naive; the basis == HeckeSturmBound; T_p on both loops;
//        SatFormsPrecision == BuildForms; FormsSpaceSupported and its refusals.
//   (N6) (slot-24 FNTT audit) the all-(ell-1) square at the largest N with N (ell-1)^2 < P1 P2 (ell = 1000003, 300007)
//        == the closed form (n + 1) mod ell, SIMD and scalar; ell >= 2^16 at N = 1000 (65537, 1000003, 2^27 - 39) on the
//        legacy route == HeckeSturmBound, T_p on both loops; ComputeSaturationDefectCell refuses before the module run.

#include "HeckeFormsFastMulValidation.h"

#include "HeckeFormsHeckeAlgebra.h"
#include "HeckeSaturationDefect.h"
#include "HeckeSturmBound.h"   // (slot-24 FNTT) the independent E_4 / E_6 / Delta of arm (N2)
#include "../Hecke/HeckeSettings.h"   // (slot-24 FNTT audit) hecke::NotImplementedInThisBuild, arm (N6)
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fm = hecke_ext::forms;
namespace sd = hecke_ext::satdefect;
using fm::QSeries;

oc_util::Checker Check("  FAIL: ", false);

QSeries Random(std::mt19937_64& rng, size_t len, uint32_t ell) {
    QSeries s(len);
    for (auto& x : s) x = static_cast<uint32_t>(rng() % ell);
    return s;
}

std::string Where(uint32_t ell, size_t la, size_t lb, uint32_t N) {
    return "ell=" + std::to_string(ell) + " |a|=" + std::to_string(la) + " |b|=" + std::to_string(lb) + " N=" + std::to_string(N);
}

// fast, schoolbook (when ell < 2^16), MulSeries under both scopes, against the naive oracle
void Compare(const QSeries& a, const QSeries& b, uint32_t ell, uint32_t N, const std::string& tag, bool same_object = false) {
    const QSeries& bb = same_object ? a : b;
    const QSeries ref = fm::MulSeriesNaive(a, bb, ell, N);
    const QSeries fast = fm::MulSeriesFast(a, bb, ell, N);
    Check(fast == ref, tag + ": MulSeriesFast != naive at " + Where(ell, a.size(), bb.size(), N));
    if (ell < (1u << 16)) Check(fm::MulSeriesSchoolbook(a, bb, ell, N) == ref, tag + ": schoolbook != naive at " + Where(ell, a.size(), bb.size(), N));
    {
        const fm::ScopedFormsFastMul s(true);
        Check(fm::MulSeries(a, bb, ell, N) == ref, tag + ": MulSeries (fast scope) != naive at " + Where(ell, a.size(), bb.size(), N));
    }
    {
        const fm::ScopedFormsFastMul s(false);
        Check(fm::MulSeries(a, bb, ell, N) == ref, tag + ": MulSeries (naive scope) != naive at " + Where(ell, a.size(), bb.size(), N));
    }
}

void ArmP1(std::mt19937_64& rng) {
    std::cout << "-- (P1) random products around the thresholds\n";
    const uint32_t ells[] = {2, 3, 5, 7, 691, 1009, 10007, 65521};
    const size_t lens[] = {1, 2, 63, 64, 65, 127, 128, 129, 300, 511, 512, 513, 1100};
    uint32_t n = 0;
    for (uint32_t ell : ells)
        for (size_t la : lens)
            for (size_t lb : {size_t{1}, size_t{64}, la, la + 1, size_t{513}}) {
                const QSeries a = Random(rng, la, ell), b = Random(rng, lb, ell);
                const uint32_t full = static_cast<uint32_t>(la + lb - 1);
                for (uint32_t N : {1u, static_cast<uint32_t>(std::max<size_t>(la, 2) - 1), full - (full > 1 ? 1u : 0u), full, full + 1,
                                   static_cast<uint32_t>(la)}) {
                    if (N == 0) continue;
                    Compare(a, b, ell, N, "(P1)");
                    ++n;
                }
            }
    // lengths whose |a| + |b| - 1 sits on a transform power of two
    for (uint32_t ell : {7u, 65521u})
        for (uint32_t j : {7u, 10u, 13u}) {
            const size_t p = size_t{1} << j;
            for (size_t la : {p / 2, p / 2 + 1}) {
                const size_t lb = p + 1 - la;   // la + lb - 1 = p
                for (size_t dl : {size_t{0}, size_t{1}}) {
                    const QSeries a = Random(rng, la, ell), b = Random(rng, lb + dl, ell);
                    Compare(a, b, ell, static_cast<uint32_t>(la + lb + dl), "(P1 pow2)");
                    Compare(a, b, ell, static_cast<uint32_t>(p / 2), "(P1 pow2 trunc)");
                    n += 2;
                }
            }
        }
    std::cout << "   products=" << n << "\n";
}

void ArmP2(std::mt19937_64& rng) {
    std::cout << "-- (P2) extreme, sparse, empty and square inputs\n";
    {
        const uint32_t ell = 65521;
        const QSeries a(20000, ell - 1), b(20000, ell - 1);
        Compare(a, b, ell, 20000, "(P2 max)");
        Compare(a, b, ell, 39999, "(P2 max full)");
        Compare(a, a, ell, 20000, "(P2 max square)", true);
    }
    {
        QSeries a(5000, 0), b = Random(rng, 5000, 1009);
        a[4321] = 1008;
        Compare(a, b, 1009, 5000, "(P2 monomial)");
        Compare(b, a, 1009, 5000, "(P2 monomial swapped)");
        Compare(QSeries(5000, 0), b, 1009, 5000, "(P2 zero)");
        Compare(QSeries(), b, 1009, 5000, "(P2 empty a)");
        Compare(b, QSeries(), 1009, 5000, "(P2 empty b)");
        Compare(b, b, 1009, 5000, "(P2 square)", true);
        const QSeries c = b;
        Compare(b, c, 1009, 5000, "(P2 equal content)");
        Compare(b, c, 1009, 1, "(P2 N=1)");
        Check(fm::MulSeriesFast(b, c, 1009, 0).empty() && fm::MulSeriesNaive(b, c, 1009, 0).empty(), "(P2) N = 0 gives the empty series");
    }
    {   // the pentagonal series (DeltaSeries' factor), sparse
        const uint32_t ell = 691, N = 9000;
        QSeries P(N, 0);
        for (int64_t m = -static_cast<int64_t>(N); m <= static_cast<int64_t>(N); ++m) {
            const int64_t e = m * (3 * m - 1) / 2;
            if (e < 0 || e >= static_cast<int64_t>(N)) continue;
            P[e] = (m % 2 == 0) ? 1u : ell - 1u;
        }
        Compare(P, P, ell, N, "(P2 pentagonal square)", true);
        for (uint32_t ell2 : {5u, 7u, 691u, 1009u, 65521u}) {
            const fm::ScopedFormsFastMul s(true);
            const QSeries Df = fm::DeltaSeries(ell2, 6000);
            const fm::ScopedFormsFastMul s2(false);
            Check(Df == fm::DeltaSeries(ell2, 6000), "(P2) DeltaSeries fast == naive at ell=" + std::to_string(ell2));
        }
    }
}

void ArmP3(std::mt19937_64& rng) {
    std::cout << "-- (P3) the fallback above the word limit\n";
    const fm::FormsFastMulCounters c0 = fm::FormsFastMulCountersNow();
    const uint32_t ell = 65537;
    const QSeries a = Random(rng, 700, ell), b = Random(rng, 900, ell);
    Check(fm::MulSeriesFast(a, b, ell, 1200) == fm::MulSeriesNaive(a, b, ell, 1200), "(P3) ell = 65537: fast (fallback) == naive");
    Check(fm::FormsFastMulCountersNow().fallback > c0.fallback, "(P3) ell = 65537 took the fallback");
}

void ArmP5(std::mt19937_64& rng) {
    std::cout << "-- (P5) the transform-length guard and the chunked product\n";
    {
        const uint32_t ell = 65521;
        const size_t la = (size_t{1} << 25) - 63, lb = 65;
        const uint32_t N = (1u << 25) + 1;   // = la + lb - 1: one past the transform length 2^25
        const QSeries a = Random(rng, la, ell), b = Random(rng, lb, ell);
        const fm::FormsFastMulCounters c0 = fm::FormsFastMulCountersNow();
        const QSeries fast = fm::MulSeriesFast(a, b, ell, N);
        const fm::FormsFastMulCounters c1 = fm::FormsFastMulCountersNow();
        Check(c1.fallback == c0.fallback + 1 && c1.ntt == c0.ntt, "(P5) |a| + |b| - 1 = 2^25 + 1 took the length fallback, not the NTT");
        Check(fast == fm::MulSeriesSchoolbook(a, b, ell, N), "(P5) length fallback == schoolbook at " + Where(ell, la, lb, N));
    }
    uint32_t n = 0;
    for (uint32_t ell : {2u, 7u, 1009u, 65521u})
        for (uint32_t cl : {1u, 3u, 6u, 9u})
            for (const auto& L : {std::pair<size_t, size_t>{1, 1}, {5, 300}, {300, 5}, {512, 512}, {1500, 2100}, {4097, 1023}}) {
                const QSeries a = Random(rng, L.first, ell), b = Random(rng, L.second, ell);
                const uint32_t full = static_cast<uint32_t>(L.first + L.second - 1);
                for (uint32_t N : {1u, static_cast<uint32_t>(L.first), full, full + 3}) {
                    Check(fm::MulSeriesChunked(a, b, ell, N, cl) == fm::MulSeriesNaive(a, b, ell, N),
                          "(P5) chunked (chunk_lg=" + std::to_string(cl) + ") != naive at " + Where(ell, L.first, L.second, N));
                    ++n;
                }
            }
    {
        const QSeries b = Random(rng, 700, 691);
        Check(fm::MulSeriesChunked(QSeries(), b, 691, 900, 4) == QSeries(900, 0), "(P5) chunked with an empty operand is zero");
        Check(fm::MulSeriesChunked(b, b, 691, 0, 4).empty(), "(P5) chunked at N = 0 is the empty series");
    }
    std::cout << "   chunked products=" << n << "\n";
}

QSeries SigmaSeries(uint32_t e, uint32_t c, uint32_t ell, uint32_t N) {   // 1 + c sum sigma_e(n) q^n mod ell, by a sieve
    std::vector<uint64_t> s(N, 0);
    for (uint32_t t = 1; t < N; ++t) {
        uint64_t p = 1, b = t % ell;
        for (uint32_t i = 0; i < e; ++i) p = p * b % ell;
        for (uint64_t m = t; m < N; m += t) s[m] = (s[m] + p) % ell;
    }
    QSeries out(N, 0);
    out[0] = 1 % ell;
    for (uint32_t n = 1; n < N; ++n) out[n] = static_cast<uint32_t>(s[n] * (c % ell) % ell);
    return out;
}

void ArmP4() {
    std::cout << "-- (P4) independent identities at long length (fast route only)\n";
    const fm::ScopedFormsFastMul s(true);
    for (uint32_t ell : {65521u, 1009u}) {
        const uint32_t N = 300001;
        const QSeries E4 = fm::EisensteinSeries(4, ell, N);
        Check(fm::MulSeries(E4, E4, ell, N) == SigmaSeries(7, 480, ell, N), "(P4) E_4^2 == E_8 at N = 300001, ell = " + std::to_string(ell));
    }
    {
        const uint32_t ell = 10007, N = 100001;
        const QSeries E4 = fm::EisensteinSeries(4, ell, N), E6 = fm::EisensteinSeries(6, ell, N), D = fm::DeltaSeries(ell, N);
        const QSeries E43 = fm::MulSeries(fm::MulSeries(E4, E4, ell, N), E4, ell, N), E62 = fm::MulSeries(E6, E6, ell, N);
        bool ok = true;
        for (uint32_t i = 0; i < N && ok; ++i)
            ok = (E43[i] + ell - E62[i]) % ell == static_cast<uint32_t>(1728ull * D[i] % ell);
        Check(ok, "(P4) E_4^3 - E_6^2 == 1728 Delta at N = 100001, ell = 10007");
    }
}

void ArmE() {
    std::cout << "-- (E) the divisor sieve against the trial-division loop\n";
    for (uint32_t ell : {5u, 7u, 691u, 1009u, 10007u, 65521u})
        for (uint32_t N : {1u, 2u, 3u, 97u, 5000u})
            for (uint32_t w : {4u, 6u}) {
                QSeries f, nv;
                {
                    const fm::ScopedFormsFastMul s(true);
                    f = fm::EisensteinSeries(w, ell, N);
                }
                {
                    const fm::ScopedFormsFastMul s(false);
                    nv = fm::EisensteinSeries(w, ell, N);
                }
                Check(f == nv, "(E) E_" + std::to_string(w) + " sieve == loop at ell=" + std::to_string(ell) + " N=" + std::to_string(N));
            }
}

bool SameMatrix(const hecke_ext::ExtMatrix& a, const hecke_ext::ExtMatrix& b) {
    return a.rows == b.rows && a.cols == b.cols && a.p == b.p && a.data == b.data;
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

void ArmF() {
    std::cout << "-- (F) the cross-route spaces and operators at the sat route's precision\n";
    const std::pair<uint32_t, uint32_t> cells[] = {{10, 5}, {40, 5}, {108, 7}, {300, 43}, {60, 691}, {24, 1009}};
    for (const auto& c : cells) {
        const uint32_t n = c.first, ell = c.second, k = n + 2;
        std::vector<uint64_t> P = PrimesUpTo(std::max(31u, k / 12));   // as HeckeSaturationDefect.cpp BuildForms
        if (std::find(P.begin(), P.end(), ell) == P.end()) P.push_back(ell);
        const uint32_t N = 2 * (k / 12 + 1) * static_cast<uint32_t>(P.back()) + 2;
        fm::CuspSpace Mf, Mn, Sf, Sn;
        std::vector<hecke_ext::ExtMatrix> Tf, Tn;
        double tf = 0, tn = 0;
        {
            const fm::ScopedFormsFastMul s(true);
            const auto t0 = std::chrono::steady_clock::now();
            Mf = fm::BuildModularSpace(k, ell, N);
            for (uint64_t p : P) Tf.push_back(fm::HeckeOnCusp(Mf, static_cast<uint32_t>(p)));
            tf = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            Sf = fm::BuildCuspSpace(k, ell, N);
        }
        {
            const fm::ScopedFormsFastMul s(false);
            const auto t0 = std::chrono::steady_clock::now();
            Mn = fm::BuildModularSpace(k, ell, N);
            for (uint64_t p : P) Tn.push_back(fm::HeckeOnCusp(Mn, static_cast<uint32_t>(p)));
            tn = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            Sn = fm::BuildCuspSpace(k, ell, N);
        }
        const std::string tag = "(F) (n, ell) = (" + std::to_string(n) + ", " + std::to_string(ell) + ") N=" + std::to_string(N);
        Check(Mf.basis == Mn.basis && Mf.d == Mn.d && Mf.k_prime == Mn.k_prime && Mf.with_constant == Mn.with_constant,
              tag + ": M_k basis series fast == naive");
        Check(Sf.basis == Sn.basis && Sf.d == Sn.d, tag + ": S_k basis series fast == naive");
        bool same = Tf.size() == Tn.size();
        for (size_t i = 0; same && i < Tf.size(); ++i) same = SameMatrix(Tf[i], Tn[i]);
        Check(same, tag + ": T_p on M_k fast == naive for every generator prime");
        std::cout << "   " << tag << " dim M_k=" << Mf.basis.size() << " primes=" << P.size() << " forms_s fast=" << tf << " naive=" << tn << "\n";
    }
}

// ---------------------------------------------------------------------------
// (slot-24 FNTT) arms (N1) - (N4): the FNTT route (cached spectra, one pass for M_k, no Dpow / Epow tables, blocked /
// AVX2 butterflies, Barrett CRT, HeckeOnCusp's delayed reduction) against the schoolbook, the G2-X route (the rollback
// arm HECKE_FORMS_NTT_CACHE=0, ScopedFormsNttCache(false)) and the independent HeckeSturmBound E_4 / E_6 / Delta.

// (N1) the engine's product against the schoolbook (and MulSeriesFast), SIMD and scalar butterflies
void ArmN1(std::mt19937_64& rng) {
    std::cout << "-- (N1) the FNTT engine's product == schoolbook, SIMD and scalar butterflies (ell up to 65521)\n";
    const uint32_t ells[] = {5, 7, 691, 1009, 10007, 20011, 40009, 65521};
    uint32_t n = 0;
    for (uint32_t ell : ells)
        for (uint32_t N : {1024u, 1500u, 4097u, 9000u}) {
            Check(fm::FnttApplicable(ell, N), "(N1) the FNTT route applies at ell=" + std::to_string(ell) + " N=" + std::to_string(N));
            for (const auto& L : {std::pair<size_t, size_t>{N, N}, {N, N / 3}, {65, N}, {N + 7, N - 1}}) {
                const QSeries a = Random(rng, L.first, ell), b = Random(rng, L.second, ell);
                const QSeries ref = fm::MulSeriesSchoolbook(a, b, ell, N);
                for (bool simd : {true, false}) {
                    const fm::ScopedFormsNttSimd s(simd);
                    Check(fm::MulSeriesFntt(a, b, ell, N) == ref, std::string("(N1) FNTT (") + (simd ? "simd" : "scalar") + ") != schoolbook at " +
                                                                      Where(ell, a.size(), b.size(), N));
                }
                ++n;
            }
            const QSeries a = Random(rng, N, ell);
            Check(fm::MulSeriesFntt(a, a, ell, N) == fm::MulSeriesSchoolbook(a, a, ell, N), "(N1) FNTT square != schoolbook at " + Where(ell, N, N, N));
        }
    {   // the CRT worst case: every entry ell - 1 at ell = 65521, N = 20000; and a longer product against MulSeriesFast
        const uint32_t ell = 65521, N = 20000;
        const QSeries a(N, ell - 1);
        for (bool simd : {true, false}) {
            const fm::ScopedFormsNttSimd s(simd);
            Check(fm::MulSeriesFntt(a, a, ell, N) == fm::MulSeriesSchoolbook(a, a, ell, N), "(N1) all ell-1 at ell = 65521, N = 20000");
        }
        for (uint32_t e2 : {40009u, 65521u}) {
            const uint32_t N2 = 600001;   // transform length 2^21 (> the G2-X plan cache, > the butterfly block)
            const QSeries x = Random(rng, N2, e2), y = Random(rng, N2, e2);
            Check(fm::MulSeriesFntt(x, y, e2, N2) == fm::MulSeriesFast(x, y, e2, N2), "(N1) FNTT == MulSeriesFast at " + Where(e2, N2, N2, N2));
        }
    }
    // the applicability bounds: N < 1024, and the CRT bound N (ell - 1)^2 < P1 P2 (65521: N <= 18358 * 10^3 is fine; the
    // length bound 2N - 1 <= 2^25 binds first)
    Check(!fm::FnttApplicable(65521, 1023) && !fm::FnttApplicable(fm::kFormsMaxEllExclusive, 4096) && !fm::FnttApplicable(3, 4096) &&
              fm::FnttApplicable(65537, 4096) && fm::FnttApplicable(65521, 1u << 24) && !fm::FnttApplicable(65521, (1u << 24) + 1) &&
              fm::FnttApplicable(1000003, 78000) && !fm::FnttApplicable(1000003, 79000),
          "(N1) FnttApplicable: N >= 1024, 5 <= ell < 2^27, 2N - 1 <= 2^25, N (ell-1)^2 < P1 P2");
    std::cout << "   products=" << n << "\n";
}

// (N2) + (N3) the spaces and T_p: FNTT (SIMD, scalar) == G2-X; the basis prefix == the HeckeSturmBound construction
void ArmN2() {
    std::cout << "-- (N2) BuildCuspSpace / BuildModularSpace / HeckeOnCusp: FNTT == G2-X route == HeckeSturmBound series\n";
    // k' = 0, 4, 6, 8, 10, 14 and d = 0 (k = 14); ell up to 65521, all at the sat route's precision N = 2 (k/12+1) pmax + 2
    const std::pair<uint32_t, uint32_t> cells[] = {{10, 65521}, {12, 1009}, {14, 691}, {18, 10007}, {20, 20011}, {40, 40009},
                                                   {300, 43},   {60, 691},  {24, 1009}, {102, 10007}};
    for (const auto& c : cells) {
        const uint32_t n = c.first, ell = c.second, k = n + 2;
        std::vector<uint64_t> P = PrimesUpTo(std::max(31u, k / 12));
        if (std::find(P.begin(), P.end(), ell) == P.end()) P.push_back(ell);
        const uint32_t N = 2 * (k / 12 + 1) * static_cast<uint32_t>(P.back()) + 2;
        const std::string tag = "(N2) (n, ell) = (" + std::to_string(n) + ", " + std::to_string(ell) + ") N=" + std::to_string(N);
        Check(fm::FnttApplicable(ell, N), tag + ": the FNTT route applies");
        fm::CuspSpace Mlg, Slg;
        std::vector<hecke_ext::ExtMatrix> Tlg;
        {
            const fm::ScopedFormsFastMul f(true);
            const fm::ScopedFormsNttCache s(false);
            const fm::FormsNttCounters r0 = fm::FormsNttCountersNow();
            Mlg = fm::BuildModularSpace(k, ell, N);
            Slg = fm::BuildCuspSpace(k, ell, N);
            for (uint64_t p : P) Tlg.push_back(fm::HeckeOnCusp(Mlg, static_cast<uint32_t>(p)));
            // (slot-24 FNTT audit F4) the rollback scope really takes the G2-X code: both spaces and every HeckeOnCusp
            // (a rollback that silently stayed on the FNTT route would pass every equality above, being value-neutral)
            const fm::FormsNttCounters r1 = fm::FormsNttCountersNow();
            Check(r1.legacy_spaces == r0.legacy_spaces + 2 && r1.fntt_spaces == r0.fntt_spaces && r1.forward == r0.forward &&
                      r1.hoc_legacy == r0.hoc_legacy + P.size() && r1.hoc_delayed == r0.hoc_delayed,
                  tag + ": the rollback scope (ScopedFormsNttCache(false)) took the G2-X spaces and HeckeOnCusp loop");
        }
        for (bool simd : {true, false}) {
            const fm::ScopedFormsFastMul f(true);
            const fm::ScopedFormsNttCache s(true);
            const fm::ScopedFormsNttSimd v(simd);
            const fm::FormsNttCounters c0 = fm::FormsNttCountersNow();
            const fm::CuspSpace M = fm::BuildModularSpace(k, ell, N), S = fm::BuildCuspSpace(k, ell, N);
            std::vector<hecke_ext::ExtMatrix> T;
            for (uint64_t p : P) T.push_back(fm::HeckeOnCusp(M, static_cast<uint32_t>(p)));
            const fm::FormsNttCounters c1 = fm::FormsNttCountersNow();
            const std::string t2 = tag + (simd ? " simd" : " scalar");
            Check(c1.fntt_spaces == c0.fntt_spaces + 2 && c1.legacy_spaces == c0.legacy_spaces, t2 + ": both spaces took the FNTT route");
            Check(M.basis == Mlg.basis && M.d == Mlg.d && M.k_prime == Mlg.k_prime && M.with_constant && M.precision == N,
                  t2 + ": M_k basis FNTT == G2-X");
            Check(S.basis == Slg.basis && S.d == Slg.d && !S.with_constant, t2 + ": S_k basis FNTT == G2-X");
            bool same = T.size() == Tlg.size();
            for (size_t i = 0; same && i < T.size(); ++i) same = SameMatrix(T[i], Tlg[i]);
            Check(same, t2 + ": T_p on M_k (delayed reduction) == G2-X loop for every generator prime");
            Check(c1.hoc_delayed == c0.hoc_delayed + P.size(), t2 + ": HeckeOnCusp took the delayed-reduction loop");
            if (simd && fm::FormsNttSimdAvailable()) Check(c1.simd_transforms > c0.simd_transforms, t2 + ": the AVX2 butterflies ran");
            if (!simd) Check(c1.scalar_transforms > c0.scalar_transforms && c1.simd_transforms == c0.simd_transforms, t2 + ": the scalar butterflies ran");
        }
        {   // independent: the HeckeSturmBound E_4 / E_6 / Delta (its own sieve, pentagonal power) and the schoolbook, to
            // precision Ns <= N (truncation commutes with the product): f_j = Delta^j E_6^{2(d-j)} E_{k'}, f_0 = E_6^{2d} E_{k'}
            const uint32_t Ns = std::min<uint32_t>(N, 2500);
            const QSeries E4 = hecke_ext::EisensteinSeries(4, Ns, ell), E6 = hecke_ext::EisensteinSeries(6, Ns, ell),
                          D = hecke_ext::DeltaSeries(Ns, ell);
            auto mul = [&](const QSeries& a, const QSeries& b) { return fm::MulSeriesSchoolbook(a, b, ell, Ns); };
            QSeries base(Ns, 0);
            base[0] = 1;
            const uint32_t kp = Mlg.k_prime, d = Mlg.d;
            if (kp == 4) base = E4;
            else if (kp == 6) base = E6;
            else if (kp == 8) base = mul(E4, E4);
            else if (kp == 10) base = mul(E4, E6);
            else if (kp == 14) base = mul(mul(E4, E4), E6);
            const QSeries E6sq = mul(E6, E6);
            std::vector<QSeries> G(d + 1), Dp(d + 1);
            G[0] = base;
            Dp[0] = QSeries(Ns, 0);
            Dp[0][0] = 1;
            for (uint32_t i = 1; i <= d; ++i) G[i] = mul(G[i - 1], E6sq), Dp[i] = mul(Dp[i - 1], D);
            bool ok = Mlg.basis.size() == d + 1;
            for (uint32_t j = 0; ok && j <= d; ++j) {
                const QSeries f = mul(Dp[j], G[d - j]);
                ok = std::equal(f.begin(), f.end(), Mlg.basis[j].begin());
            }
            Check(ok, tag + ": the M_k basis to precision " + std::to_string(Ns) + " == the HeckeSturmBound series construction");
        }
    }
    {   // (N3) the residual check throws the same message on both loops (a basis series perturbed past the coordinates)
        const uint32_t k = 26, ell = 1009, N = 6056;
        fm::CuspSpace M;
        {
            const fm::ScopedFormsNttCache s(true);
            M = fm::BuildModularSpace(k, ell, N);
        }
        M.basis[1][N - 2] = (M.basis[1][N - 2] + 1) % ell;
        std::string e_new, e_old;
        for (bool on : {true, false}) {
            const fm::ScopedFormsNttCache s(on);
            try {
                (void)fm::HeckeOnCusp(M, 2);
            } catch (const std::runtime_error& e) {
                (on ? e_new : e_old) = e.what();
            }
        }
        Check(!e_new.empty() && e_new == e_old, "(N3) a perturbed series: both HeckeOnCusp loops throw the same '" + e_old + "'");
    }
}

// (N5) above the word range (slot-24 FNTT lifts the forms ell < 2^16 limit within the FNTT route's exactness bounds): the
// engine's product against the naive oracle, the spaces against the G2-X route (naive products at these N) and the
// HeckeSturmBound construction, T_p on both HeckeOnCusp loops, FormsSpaceSupported / SatFormsPrecision, the refusals.
void ArmN5(std::mt19937_64& rng) {
    std::cout << "-- (N5) ell >= 2^16: FNTT == naive == HeckeSturmBound; the support predicate and its refusals\n";
    const uint32_t ells[] = {65537, 100003, 131071, 1000003};
    for (uint32_t ell : ells) {
        for (uint32_t N : {1024u, 3001u}) {
            const std::string w = "ell=" + std::to_string(ell) + " N=" + std::to_string(N);
            Check(fm::FnttApplicable(ell, N) && fm::FormsSpaceSupported(ell, N), "(N5) FNTT applies / supported at " + w);
            const QSeries a = Random(rng, N, ell), b = Random(rng, N - 5, ell), top(N, ell - 1);
            for (bool simd : {true, false}) {
                const fm::ScopedFormsNttSimd s(simd);
                Check(fm::MulSeriesFntt(a, b, ell, N) == fm::MulSeriesNaive(a, b, ell, N), "(N5) FNTT product != naive at " + w);
                Check(fm::MulSeriesFntt(top, top, ell, N) == fm::MulSeriesNaive(top, top, ell, N), "(N5) all ell-1 square != naive at " + w);
            }
        }
        for (uint32_t k : {12u, 16u, 18u, 20u, 22u, 26u, 50u}) {   // k' = 0, 4, 6, 8, 10, 14 (twice); d = 1 .. 4
            const uint32_t N = 2501;
            const std::string tag = "(N5) (k, ell) = (" + std::to_string(k) + ", " + std::to_string(ell) + ") N=" + std::to_string(N);
            // (the G2-X route refuses ell >= 2^16 at N >= 1024 -- FormsSpaceSupported -- so the basis oracle here is the
            // independent construction below; T_p is compared on both HeckeOnCusp loops on the same space)
            const fm::FormsNttCounters c0 = fm::FormsNttCountersNow();
            const fm::CuspSpace Mf = fm::BuildModularSpace(k, ell, N);
            Check(fm::FormsNttCountersNow().fntt_spaces == c0.fntt_spaces + 1, tag + ": the space took the FNTT route");
            std::vector<hecke_ext::ExtMatrix> Tf, Tl;
            for (bool on : {true, false}) {
                const fm::ScopedFormsNttCache s(on);
                for (uint32_t p : {2u, 3u, 5u, 7u}) (on ? Tf : Tl).push_back(fm::HeckeOnCusp(Mf, p));
            }
            bool same = Tf.size() == Tl.size();
            for (size_t i = 0; same && i < Tf.size(); ++i) same = SameMatrix(Tf[i], Tl[i]);
            Check(same, tag + ": T_2,3,5,7 delayed-reduction loop == legacy HeckeOnCusp loop");
            const QSeries E4 = hecke_ext::EisensteinSeries(4, N, ell), E6 = hecke_ext::EisensteinSeries(6, N, ell),
                          D = hecke_ext::DeltaSeries(N, ell);
            auto mul = [&](const QSeries& x, const QSeries& y) { return fm::MulSeriesNaive(x, y, ell, N); };
            QSeries base(N, 0);
            base[0] = 1;
            const uint32_t kp = Mf.k_prime, d = Mf.d;
            if (kp == 4) base = E4;
            else if (kp == 6) base = E6;
            else if (kp == 8) base = mul(E4, E4);
            else if (kp == 10) base = mul(E4, E6);
            else if (kp == 14) base = mul(mul(E4, E4), E6);
            const QSeries E6sq = mul(E6, E6);
            bool ok = Mf.basis.size() == d + 1;
            QSeries Dj(N, 0);
            Dj[0] = 1;
            for (uint32_t j = 0; ok && j <= d; ++j) {
                QSeries G = base;
                for (uint32_t i = 0; i < d - j; ++i) G = mul(G, E6sq);
                ok = mul(Dj, G) == Mf.basis[j];
                Dj = mul(Dj, D);
            }
            Check(ok, tag + ": the M_k basis == the HeckeSturmBound series construction (naive products)");
        }
    }
    // the predicate and the sat route's precision
    auto sat_n = [](uint32_t k, uint32_t ell) {   // HeckeSaturationDefect.cpp BuildForms, verbatim
        std::vector<uint64_t> P = PrimesUpTo(std::max(31u, k / 12));
        if (std::find(P.begin(), P.end(), ell) == P.end()) P.push_back(ell);
        return 2 * (k / 12 + 1) * static_cast<uint32_t>(P.back()) + 2;
    };
    bool prec = true;
    for (uint32_t k : {12u, 42u, 302u, 1002u, 2002u})
        for (uint32_t ell : {5u, 7u, 43u, 691u, 10007u, 65521u, 65537u, 100003u}) prec = prec && fm::SatFormsPrecision(k, ell) == sat_n(k, ell);
    Check(prec, "(N5) SatFormsPrecision == the BuildForms precision on the (k, ell) grid");
    // (slot-24 FNTT3) the two-prime refusals below are now the rollback's (HECKE_FORMS_NTT3=0); HeckeFormsNtt3Validation
    // checks that they are lifted by default
    const fm::ScopedFormsNtt3 ntt3_off(false);
    Check(fm::FormsSpaceSupported(100003, fm::SatFormsPrecision(302, 100003)) && fm::FormsSpaceSupported(30011, fm::SatFormsPrecision(1002, 30011)) &&
              !fm::FormsSpaceSupported(100003, fm::SatFormsPrecision(1002, 100003)) &&
              !fm::FormsSpaceSupported(1000003, fm::SatFormsPrecision(12, 1000003)) && fm::FormsSpaceSupported(65521, 1u << 26) &&
              !fm::FormsSpaceSupported(3, 2000) && !fm::FormsSpaceSupported(fm::kFormsMaxEllExclusive, 2000) &&
              fm::FormsSpaceSupported(1000003, 1000),
          "(N5) FormsSpaceSupported: (300, 100003) and (1000, 30011) yes; (1000, 100003), (10, 1000003) no; ell < 2^16 any N; N < 1024 yes");
    {   // with the FNTT route off, ell >= 2^16 at N >= 1024 is refused (the naive route would be quadratic), and a space throws
        const fm::ScopedFormsNttCache s(false);
        Check(!fm::FormsSpaceSupported(100003, 3001) && fm::FormsSpaceSupported(100003, 1000) && fm::FormsSpaceSupported(65521, 3001),
              "(N5) HECKE_FORMS_NTT_CACHE=0: ell >= 2^16 needs N < 1024");
        bool threw = false;
        try {
            (void)fm::BuildModularSpace(26, 100003, 3001);
        } catch (const std::logic_error&) {
            threw = true;
        }
        Check(threw, "(N5) BuildModularSpace refuses (100003, N = 3001) with the FNTT route off");
    }
    {
        bool threw = false;
        try {
            (void)fm::BuildCuspSpace(12, 1000003, fm::SatFormsPrecision(12, 1000003));
        } catch (const std::logic_error&) {
            threw = true;
        }
        Check(threw, "(N5) BuildCuspSpace refuses (12, 1000003) at the sat precision (N (ell-1)^2 >= P1 P2)");
    }
}

// (N6, slot-24 FNTT audit F1-F3) the CRT bound at its edge, the legacy route above 2^16 below N = 1024, the pre-run refusal
void ArmN6() {
    std::cout << "-- (N6) audit: N (ell-1)^2 < P1 P2 at its edge; ell >= 2^16 at N < 1024 (legacy route); the sat route's pre-run refusal\n";
    // (F1) at the largest N the bound admits, the all-(ell - 1) square has c_n = (n + 1)(ell - 1)^2 exactly: its top coefficient
    // N (ell - 1)^2 is within (ell - 1)^2 of P1 P2, and c_n mod ell = (n + 1) mod ell is a closed form (no O(N^2) oracle needed)
    const unsigned __int128 P = static_cast<unsigned __int128>(167772161u) * 469762049u;
    for (uint32_t ell : {1000003u, 300007u}) {
        const uint64_t e1 = ell - 1;
        const uint32_t Nmax = static_cast<uint32_t>((P - 1) / (static_cast<unsigned __int128>(e1) * e1));
        const std::string w = "ell=" + std::to_string(ell) + " Nmax=" + std::to_string(Nmax);
        Check(fm::FnttApplicable(ell, Nmax) && !fm::FnttApplicable(ell, Nmax + 1), "(N6) FnttApplicable's edge is N (ell-1)^2 < P1 P2 at " + w);
        const QSeries top(Nmax, ell - 1), top2(Nmax, ell - 1);
        for (bool simd : {true, false}) {
            const fm::ScopedFormsNttSimd s(simd);
            const QSeries sq = fm::MulSeriesFntt(top, top, ell, Nmax), pr = fm::MulSeriesFntt(top, top2, ell, Nmax);
            bool ok = sq.size() == Nmax && pr == sq;
            for (uint32_t n = 0; ok && n < Nmax; ++n) ok = sq[n] == (n + 1) % ell;
            Check(ok, std::string("(N6) all ell-1 square at the CRT edge == (n + 1) mod ell (") + (simd ? "simd" : "scalar") + ") at " + w);
        }
    }
    // (F2) ell >= 2^16 at N < 1024: the legacy route (naive products) -- the space against the HeckeSturmBound construction,
    // T_p on both HeckeOnCusp loops; ell up to the largest prime below kFormsMaxEllExclusive
    for (uint32_t ell : {65537u, 1000003u, 134217689u}) {
        const uint32_t N = 1000;
        for (uint32_t k : {26u, 34u}) {
            const std::string tag = "(N6) (k, ell) = (" + std::to_string(k) + ", " + std::to_string(ell) + ") N=1000";
            Check(fm::FormsSpaceSupported(ell, N) && !fm::FnttApplicable(ell, N), tag + ": supported, not on the FNTT route");
            const fm::FormsNttCounters c0 = fm::FormsNttCountersNow();
            const fm::CuspSpace M = fm::BuildModularSpace(k, ell, N);
            Check(fm::FormsNttCountersNow().legacy_spaces == c0.legacy_spaces + 1, tag + ": the space took the legacy route");
            const QSeries E4 = hecke_ext::EisensteinSeries(4, N, ell), E6 = hecke_ext::EisensteinSeries(6, N, ell),
                          D = hecke_ext::DeltaSeries(N, ell);
            auto mul = [&](const QSeries& x, const QSeries& y) { return fm::MulSeriesNaive(x, y, ell, N); };
            const uint32_t kp = M.k_prime, d = M.d;
            QSeries base(N, 0);
            base[0] = 1;
            if (kp == 4) base = E4;
            else if (kp == 6) base = E6;
            else if (kp == 8) base = mul(E4, E4);
            else if (kp == 10) base = mul(E4, E6);
            else if (kp == 14) base = mul(mul(E4, E4), E6);
            const QSeries E6sq = mul(E6, E6);
            bool ok = M.basis.size() == d + 1;
            QSeries Dj(N, 0);
            Dj[0] = 1;
            for (uint32_t j = 0; ok && j <= d; ++j) {
                QSeries G = base;
                for (uint32_t i = 0; i < d - j; ++i) G = mul(G, E6sq);
                ok = mul(Dj, G) == M.basis[j];
                Dj = mul(Dj, D);
            }
            Check(ok, tag + ": the M_k basis == the HeckeSturmBound series construction");
            std::vector<hecke_ext::ExtMatrix> Tf, Tl;
            for (bool on : {true, false}) {
                const fm::ScopedFormsNttCache s(on);
                for (uint32_t p : {2u, 3u, 5u}) (on ? Tf : Tl).push_back(fm::HeckeOnCusp(M, p));
            }
            bool same = Tf.size() == Tl.size();
            for (size_t i = 0; same && i < Tf.size(); ++i) same = SameMatrix(Tf[i], Tl[i]);
            Check(same, tag + ": T_2,3,5 delayed-reduction loop == legacy loop");
        }
    }
    // (F3) the sat route refuses an unsupported cell before its module run (ComputeSaturationDefectCell), by name and at once;
    // and the predicate's edge at ell = 100003 is n = 464 / 466 (N (ell-1)^2 vs P1 P2)
    const fm::ScopedFormsNtt3 ntt3_off(false);   // (slot-24 FNTT3) the two-prime refusals: the rollback HECKE_FORMS_NTT3=0
    Check(fm::FormsSpaceSupported(100003, fm::SatFormsPrecision(466, 100003)) &&
              !fm::FormsSpaceSupported(100003, fm::SatFormsPrecision(468, 100003)),
          "(N6) FormsSpaceSupported at ell = 100003: n = 464 yes, n = 466 no");
    for (const auto& c : {std::pair<uint32_t, uint64_t>{10, 1000003}, {20, 300007}}) {
        const auto t0 = std::chrono::steady_clock::now();
        std::string msg;
        try {
            sd::SatOptions o;   // (the default KMAX = 6: were the forms pre-check dropped, the ell^KMAX >= 2^64 refusal
                                // would speak instead -- the same kind, without "FormsSpaceSupported": the check fails at once)
            (void)sd::ComputeSaturationDefectCell(c.first, c.second, o);
        } catch (const hecke::NotImplementedInThisBuild& e) {
            msg = e.what();
        } catch (const std::exception& e) {
            msg = std::string("WRONG KIND: ") + e.what();
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Check(msg.find("FormsSpaceSupported") != std::string::npos && msg.find("WRONG KIND") == std::string::npos && sec < 10.0,
              "(N6) ComputeSaturationDefectCell(" + std::to_string(c.first) + ", " + std::to_string(c.second) +
                  ") is refused before the module run (NotImplementedInThisBuild naming FormsSpaceSupported): '" + msg + "'");
    }
}

std::string SatText(uint32_t n, uint64_t ell, bool fast, bool& forms_ran) {
    const fm::ScopedFormsFastMul s(fast);
    sd::SatOptions o;
    const sd::SatCell c = sd::ComputeSaturationDefectCell(n, ell, o);
    std::ostringstream os;
    os << "SAT_CELL n=" << c.n << " ell=" << c.ell << " blocks=" << c.blocks.size() << "\n";
    forms_ran = false;
    for (const sd::SatBlock& b : c.blocks) {
        os << sd::SatDefectLine(c, b) << "\n";
        os << "  forms ran=" << b.forms.ran << " dim_V=" << b.forms.dim_V << " T_dim=" << b.forms.T_dim << " Tc=" << b.forms.Tc
           << " joint=" << b.forms.joint << " M_gen=" << b.forms.M_gen << " S_gen=" << b.forms.S_gen << " kernel=" << b.forms.kernel << "\n";
        forms_ran = forms_ran || b.forms.ran;
    }
    return os.str();
}

void ArmS() {
    std::cout << "-- (S) whole SAT_DEFECT cells, fast against naive\n";
    for (const auto& c : {std::pair<uint32_t, uint64_t>{40, 5}, std::pair<uint32_t, uint64_t>{24, 1009}}) {
        bool rf = false, rn = false;
        const std::string f = SatText(c.first, c.second, true, rf), nv = SatText(c.first, c.second, false, rn);
        const std::string tag = "(S) (" + std::to_string(c.first) + ", " + std::to_string(c.second) + ")";
        Check(f == nv, tag + ": every SAT_DEFECT line fast == naive");
        Check(rf && rn, tag + ": the forms cross-route ran");
        std::cout << f;
    }
}

long StatusKiB(const char* key) {   // VmHWM / VmRSS of this process, KiB (-1 if unreadable)
    std::ifstream f("/proc/self/status");
    std::string line;
    const std::string k = std::string(key) + ":";
    while (std::getline(f, line))
        if (line.compare(0, k.size(), k) == 0) return std::stol(line.substr(k.size()));
    return -1;
}

// (ENV) HECKE_FORMS_FASTMUL_CELLS="n:ell[:naive|:legacy],..." (read-only, replaces the arms; ":legacy", slot-24 FNTT: the
// default route, then the G2-X route with the FNTT route off, compared byte for byte): the sat route's forms stage alone
// (BuildForms of HeckeSaturationDefect.cpp: BuildModularSpace at N = 2 (k/12 + 1) pmax + 2, T_p for every generator
// prime, the M_gen algebra) on the fast route, and with ":naive" also on the naive route with the two compared byte for
// byte; prints time and the process VmHWM after each route (run one cell per process for a clean forms-stage peak).
uint32_t CellsArm(const std::string& env) {
    std::cout << "-- (ENV) HECKE_FORMS_FASTMUL_CELLS=\"" << env << "\"\n";
    uint32_t cells = 0;
    for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(env)) {
        const std::string& item = t.text;
        const std::vector<std::string>& f = t.raw;
        if (f.size() < 2 || f.size() > 3 || (f.size() == 3 && f[2] != "naive" && f[2] != "legacy")) {
            Check(false, "(ENV) cell '" + item + "' is not n:ell[:naive|:legacy]");
            continue;
        }
        const uint32_t n = static_cast<uint32_t>(std::stoul(f[0])), ell = static_cast<uint32_t>(std::stoul(f[1])), k = n + 2;
        std::vector<uint64_t> P = PrimesUpTo(std::max(31u, k / 12));
        if (std::find(P.begin(), P.end(), ell) == P.end()) P.push_back(ell);
        const uint32_t N = 2 * (k / 12 + 1) * static_cast<uint32_t>(P.back()) + 2;
        // route: 0 = naive, 1 = fast (the default: FNTT unless HECKE_FORMS_NTT_CACHE=0), 2 = fast with the FNTT route off (G2-X)
        auto run = [&](int route, fm::CuspSpace& M, std::vector<hecke_ext::ExtMatrix>& T) {
            const bool fast = route != 0;
            const fm::ScopedFormsFastMul s(fast);
            std::unique_ptr<fm::ScopedFormsNttCache> legacy;
            if (route == 2) legacy = std::make_unique<fm::ScopedFormsNttCache>(false);
            const auto t0 = std::chrono::steady_clock::now();
            M = fm::BuildModularSpace(k, ell, N);
            for (uint64_t p : P) T.push_back(fm::HeckeOnCusp(M, static_cast<uint32_t>(p)));
            const bool M_gen = hecke_ext::ComputeAlgebraInvariants(T, ell).dim_A == M.basis.size();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "FORMS_STAGE n=" << n << " ell=" << ell << " k=" << k << " N=" << N << " dim_M=" << M.basis.size()
                      << " primes=" << P.size() << " route="
                      << (!fast ? "naive"
                          : (fm::FormsNttCacheEnabled() && fm::FnttApplicable(ell, N)) ? "fntt"
                          : (fm::FormsNttCacheEnabled() && ell >= fm::kFormsWordEll && fm::FormsNtt3Enabled() && fm::Fntt3Applicable(ell, N))
                              ? "fntt3"   // (slot-24 FNTT3)
                              : "g2x")
                      << " M_gen=" << M_gen << " seconds=" << sec
                      << " VmHWM_MiB=" << StatusKiB("VmHWM") / 1024 << " VmRSS_MiB=" << StatusKiB("VmRSS") / 1024 << std::endl;
        };
        fm::CuspSpace Mf, Mn;
        std::vector<hecke_ext::ExtMatrix> Tf, Tn;
        run(1, Mf, Tf);
        if (f.size() == 3) {
            run(f[2] == "naive" ? 0 : 2, Mn, Tn);
            bool same = Mf.basis == Mn.basis && Tf.size() == Tn.size();
            for (size_t i = 0; same && i < Tf.size(); ++i) same = SameMatrix(Tf[i], Tn[i]);
            Check(same, "(ENV) (" + f[0] + ", " + f[1] + "): forms stage fast == " + f[2] + " (basis series and every T_p)");
        }
        ++cells;
    }
    std::cout << oc_util::cells::EffectiveLine("HeckeFormsFastMulValidation", "HECKE_FORMS_FASTMUL_CELLS", cells) << "\n";
    Check(cells > 0, "(ENV) HECKE_FORMS_FASTMUL_CELLS names no valid cell: 0 cells is not a pass");
    return cells;
}

}  // namespace

uint32_t HeckeFormsFastMulValidation() {
    Check.reset();
    std::cout << "=== HeckeFormsFastMulValidation (slot-23 G2-X: fast q-series product and divisor sieve == the naive routes) ===\n";
    std::cout << fm::FormsFastMulEffectiveLine() << "\n";
    std::cout << fm::FormsNttEffectiveLine() << "\n";
    if (const char* env = std::getenv("HECKE_FORMS_FASTMUL_CELLS")) {
        CellsArm(env);
        std::cout << "HeckeFormsFastMulValidation: checks=" << Check.checks << " failing=" << Check.fails << "\n";
        return Check.fails;
    }
    const fm::FormsFastMulCounters c0 = fm::FormsFastMulCountersNow();
    std::mt19937_64 rng(0x6732d58ull);
    ArmP1(rng);
    ArmP2(rng);
    ArmP3(rng);
    ArmP5(rng);
    ArmP4();
    ArmE();
    ArmF();
    ArmS();
    ArmN1(rng);   // (slot-24 FNTT) after the G2-X arms, so their random inputs are unchanged
    ArmN2();
    ArmN5(rng);
    ArmN6();      // (slot-24 FNTT audit) the CRT edge, ell >= 2^16 below N = 1024, the pre-run refusal
    std::cout << fm::FormsNttEffectiveLine() << "\n";
    const fm::FormsFastMulCounters c1 = fm::FormsFastMulCountersNow();
    std::cout << "-- (C) non-vacuity: ntt=" << (c1.ntt - c0.ntt) << " schoolbook=" << (c1.schoolbook - c0.schoolbook)
              << " naive=" << (c1.naive - c0.naive) << " fallback=" << (c1.fallback - c0.fallback) << " chunked=" << (c1.chunked - c0.chunked) << " sieve=" << (c1.sieve - c0.sieve)
              << " divisor_loop=" << (c1.divisor_loop - c0.divisor_loop) << "\n";
    Check(c1.ntt > c0.ntt && c1.schoolbook > c0.schoolbook && c1.fallback > c0.fallback && c1.chunked > c0.chunked && c1.sieve > c0.sieve,
          "(C) the fast scope took the NTT, the schoolbook, the fallback, the chunked product and the sieve");
    Check(c1.naive > c0.naive && c1.divisor_loop > c0.divisor_loop, "(C) the naive scope took the naive product and the divisor loop");
    {   // the naive scope never reaches a fast route
        const fm::ScopedFormsFastMul s(false);
        const fm::FormsFastMulCounters a = fm::FormsFastMulCountersNow();
        std::mt19937_64 r2(7);
        const QSeries x = Random(r2, 4000, 1009), y = Random(r2, 4000, 1009);
        (void)fm::MulSeries(x, y, 1009, 4000);
        (void)fm::EisensteinSeries(4, 1009, 500);
        const fm::FormsFastMulCounters b = fm::FormsFastMulCountersNow();
        Check(b.ntt == a.ntt && b.schoolbook == a.schoolbook && b.sieve == a.sieve && b.naive == a.naive + 1 &&
                  b.divisor_loop == a.divisor_loop + 1,
              "(C) the naive scope takes the naive routes only");
    }
    std::cout << fm::FormsFastMulEffectiveLine() << "\n";
    std::cout << "HeckeFormsFastMulValidation: checks=" << Check.checks << " failing=" << Check.fails << "\n";
    return Check.fails;
}
