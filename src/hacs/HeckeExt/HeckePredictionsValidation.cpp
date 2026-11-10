// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 4, WP-7 (phase E6) -- see HeckePredictionsValidation.h
// for the arm list, HeckePredictions.h for the Bernoulli unit and its three
// Kummer branches, and GloverFactors.h for the composition-factor recursion and
// the convention its det-twists are stated in.
//
// This TU obeys check_heckeext_includes.sh rule 2 (no Console/, no Cuda/, no
// Engine/, no BlockFlatMatrix.h) and rule 4 (no unordered_).  It is the only
// place in the package that touches a pipeline object, and it touches exactly
// one: ComputeSymmetricPowerMatrixFM on 2 x 2 inputs.

#include "HeckePredictionsValidation.h"

#include "GloverFactors.h"
#include "HeckeBoundary.h" // pinned_bernoulli_numerator -- the table WP-7 must reproduce
#include "HeckePredictions.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using hecke_ext::BernoulliOutOfRange;
using hecke_ext::GloverDecomposition;
using hecke_ext::GloverFactors;
using hecke_ext::KummerVerdict;
using hecke_ext::bernoulli_max_index;
using hecke_ext::bernoulli_mod;
using hecke_ext::bernoulli_mod_table;
using hecke_ext::kummer_divides;
using hecke_ext::kummer_verdict;
using hecke_ext::pinned_bernoulli_numerator;

oc_util::Checker Check("  *** FAIL [predictions] ");

// ===========================================================================
// Small exact-integer helpers (validation-local; nothing here is shipped).
// ===========================================================================

using i128 = __int128;

i128 i128_abs(i128 x) { return x < 0 ? -x : x; }

std::string i128_to_dec(i128 x) {
    if (x == 0) return "0";
    const bool neg = x < 0;
    i128 v = neg ? -x : x;
    std::string s;
    while (v > 0) {
        s.push_back(static_cast<char>('0' + static_cast<int>(v % 10)));
        v /= 10;
    }
    if (neg) s.push_back('-');
    std::reverse(s.begin(), s.end());
    return s;
}

i128 i128_from_dec(const char* s) {
    i128 v = 0;
    bool neg = false;
    size_t i = 0;
    if (s[0] == '-') { neg = true; i = 1; }
    for (; s[i] != '\0'; ++i) v = v * 10 + static_cast<i128>(s[i] - '0');
    return neg ? -v : v;
}

i128 i128_gcd(i128 a, i128 b) {
    a = i128_abs(a);
    b = i128_abs(b);
    while (b != 0) { const i128 t = a % b; a = b; b = t; }
    return a;
}

bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % m);
}

uint64_t powmod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1u) r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

uint64_t invmod(uint64_t a, uint64_t p) { return powmod(a % p, p - 2, p); } // p prime

// von Staudt-Clausen: den(B_k) = product of the primes p with (p - 1) | k,
// for even k >= 2.  Computed HERE, never read from the unit.
uint64_t von_staudt_clausen_denominator(uint32_t k) {
    uint64_t d = 1;
    for (uint64_t p = 2; p <= static_cast<uint64_t>(k) + 1; ++p) {
        if (!is_prime_u64(p)) continue;
        if (k % (p - 1) == 0) d *= p;
    }
    return d;
}

// ===========================================================================
// The classical Bernoulli numerators / denominators, k = 2 .. 42 even.
// Arm A compares the k <= 26 half against HeckeExt/HeckeBoundary's OWN pin
// (which is num(B_k/k), not num(B_k)); arm B is the extension past it.
// ===========================================================================
struct BernRef {
    uint32_t k;
    const char* num_bk;      // num(B_k)
    uint64_t den_bk;         // den(B_k)
    const char* num_bk_over_k; // num(B_k / k)
};

const BernRef kBernRef[] = {
    {  2, "1", 6, "1" },
    {  4, "-1", 30, "-1" },
    {  6, "1", 42, "1" },
    {  8, "-1", 30, "-1" },
    { 10, "5", 66, "1" },
    { 12, "-691", 2730, "-691" },
    { 14, "7", 6, "1" },
    { 16, "-3617", 510, "-3617" },
    { 18, "43867", 798, "43867" },
    { 20, "-174611", 330, "-174611" },
    { 22, "854513", 138, "77683" },
    { 24, "-236364091", 2730, "-236364091" },
    { 26, "8553103", 6, "657931" },
    // --- beyond HeckeBoundary's pin -------------------------------------
    { 28, "-23749461029", 870, "-3392780147" },
    { 30, "8615841276005", 14322, "1723168255201" },
    { 32, "-7709321041217", 510, "-7709321041217" },
    { 34, "2577687858367", 6, "151628697551" },
    { 36, "-26315271553053477373", 1919190, "-26315271553053477373" },
    { 38, "2929993913841559", 6, "154210205991661" },
    { 40, "-261082718496449122051", 13530, "-261082718496449122051" },
    { 42, "1520097643918070802691", 1806, "1520097643918070802691" },
};
const size_t kBernRefCount = sizeof(kBernRef) / sizeof(kBernRef[0]);

// ===========================================================================
// Arms A + B: reconstruct num(B_k) AS AN INTEGER from bernoulli_mod alone.
// ===========================================================================

// Four primes just under 2^30; their product ~1.3e36 bounds 2 * |num(B_42)|
// (~3e21) with a wide margin and still fits __int128.
std::vector<uint64_t> CrtPrimes() {
    std::vector<uint64_t> ps;
    for (uint64_t c = (1ull << 30) - 1; ps.size() < 4; c -= 2)
        if (is_prime_u64(c)) ps.push_back(c);
    return ps;
}

// Incremental Garner CRT over the given primes.  Returns the symmetric
// representative in (-M/2, M/2].
i128 CrtSymmetric(const std::vector<uint64_t>& p, const std::vector<uint64_t>& r) {
    assert(p.size() == r.size());
    i128 x = static_cast<i128>(r[0]);
    i128 M = static_cast<i128>(p[0]);
    for (size_t i = 1; i < p.size(); ++i) {
        const uint64_t pi = p[i];
        const uint64_t x_mod = static_cast<uint64_t>(((x % static_cast<i128>(pi)) + pi) % pi);
        const uint64_t m_mod = static_cast<uint64_t>(M % static_cast<i128>(pi));
        const uint64_t t = mulmod((r[i] + pi - x_mod) % pi, invmod(m_mod, pi), pi);
        x += M * static_cast<i128>(t);
        M *= static_cast<i128>(pi);
    }
    if (x > M / 2) x -= M;
    return x;
}

void ArmPinAndBeyond() {
    std::cout << "  [A/B] num(B_k) and num(B_k/k) reconstructed from bernoulli_mod by CRT\n";
    const std::vector<uint64_t> ps = CrtPrimes();
    Check(ps.size() == 4, "CRT prime set");

    uint32_t pinned_compared = 0;
    uint32_t beyond_compared = 0;

    for (size_t idx = 0; idx < kBernRefCount; ++idx) {
        const BernRef& ref = kBernRef[idx];
        const uint32_t k = ref.k;

        // von Staudt-Clausen, re-derived here.
        const uint64_t den = von_staudt_clausen_denominator(k);
        Check(den == ref.den_bk,
              "von Staudt-Clausen denominator at k = " + std::to_string(k));

        // num(B_k) mod ell_i = (B_k mod ell_i) * den(B_k), for four primes.
        std::vector<uint64_t> res;
        for (uint64_t ell : ps) {
            const uint32_t b = bernoulli_mod(k, static_cast<uint32_t>(ell));
            res.push_back(mulmod(b, den % ell, ell));
        }
        const i128 num = CrtSymmetric(ps, res);
        const i128 want_num = i128_from_dec(ref.num_bk);
        Check(num == want_num,
              "num(B_" + std::to_string(k) + ") = " + i128_to_dec(num) + ", expected "
                  + ref.num_bk);

        // num(B_k/k) = num(B_k) / gcd(num(B_k), k).
        const i128 g = i128_gcd(num, static_cast<i128>(k));
        const i128 num_over_k = num / g;
        const i128 want_over_k = i128_from_dec(ref.num_bk_over_k);
        Check(num_over_k == want_over_k,
              "num(B_" + std::to_string(k) + "/" + std::to_string(k) + ") = "
                  + i128_to_dec(num_over_k) + ", expected " + ref.num_bk_over_k);

        // --- THE PIN.  Entrywise against HeckeExt/HeckeBoundary's own table. ---
        bool known = false;
        const int64_t pinned = pinned_bernoulli_numerator(k, known);
        if (k <= 26) {
            Check(known, "HeckeBoundary pin known at k = " + std::to_string(k));
            Check(static_cast<i128>(pinned) == num_over_k,
                  "PIN MISMATCH at k = " + std::to_string(k) + ": bernoulli_mod gives "
                      + i128_to_dec(num_over_k) + ", HeckeBoundary pins "
                      + std::to_string(pinned));
            std::cout << "        k = " << (k < 10 ? " " : "") << k
                      << "  num(B_k/k) = " << i128_to_dec(num_over_k)
                      << "   pinned = " << pinned << "   "
                      << (static_cast<i128>(pinned) == num_over_k ? "AGREE" : "DIFFER") << "\n";
            ++pinned_compared;
        } else {
            Check(!known, "HeckeBoundary pin correctly absent at k = " + std::to_string(k));
            std::cout << "        k = " << k << "  num(B_k/k) = " << i128_to_dec(num_over_k)
                      << "   (beyond the shipped pin)\n";
            ++beyond_compared;
        }
    }
    Check(pinned_compared == 13, "all 13 pinned entries compared");
    Check(beyond_compared == 8, "8 entries reconstructed beyond the pin (k = 28 .. 42)");
}

// ===========================================================================
// Arm C: the Worpitzky / Stirling closed form, an independent second formula.
//   B_w = sum_{m=0}^{w} 1/(m+1) sum_{j=0}^{m} (-1)^j C(m, j) j^w      (0^0 = 1)
// Valid mod ell for w < ell - 1, where every m + 1 <= w + 1 <= ell - 1 is a unit
// and every C(m, j) with m < ell needs no Lucas reduction.
// ===========================================================================
uint32_t WorpitzkyBernoulli(uint32_t w, uint64_t ell) {
    // j^w for j = 0 .. w  (0^0 = 1, matching the recursion's B_0 = 1).
    std::vector<uint64_t> jw(w + 1);
    for (uint32_t j = 0; j <= w; ++j) jw[j] = (w == 0) ? 1u % ell : powmod(j % ell, w, ell);

    // inverses of 1 .. w + 1
    std::vector<uint64_t> inv(w + 2);
    for (uint32_t i = 1; i <= w + 1; ++i) inv[i] = invmod(i, ell);

    uint64_t total = 0;
    for (uint32_t m = 0; m <= w; ++m) {
        uint64_t inner = 0;
        uint64_t c = 1 % ell; // C(m, 0)
        for (uint32_t j = 0; j <= m; ++j) {
            const uint64_t term = mulmod(c, jw[j], ell);
            inner = (j % 2u == 0u) ? (inner + term) % ell : (inner + ell - term) % ell;
            if (j < m) c = mulmod(mulmod(c, (m - j) % ell, ell), inv[j + 1], ell);
        }
        total = (total + mulmod(inner, inv[m + 1], ell)) % ell;
    }
    return static_cast<uint32_t>(total);
}

void ArmIndependentFormula() {
    std::cout << "  [C] bernoulli_mod against the Worpitzky/Stirling closed form\n";
    uint32_t full_primes = 0, full_values = 0;

    for (uint64_t ell = 2; ell <= 131; ++ell) {
        if (!is_prime_u64(ell)) continue;
        ++full_primes;
        const std::vector<uint32_t> tab =
            bernoulli_mod_table(static_cast<uint32_t>(ell) - 2u, static_cast<uint32_t>(ell));
        Check(tab.size() == ell - 1, "table length at ell = " + std::to_string(ell));
        for (uint32_t w = 0; w + 1 < ell; ++w) {
            const uint32_t got = tab[w];
            const uint32_t want = WorpitzkyBernoulli(w, ell);
            Check(got == want, "Worpitzky mismatch at (w, ell) = (" + std::to_string(w) + ", "
                                   + std::to_string(ell) + "): " + std::to_string(got) + " vs "
                                   + std::to_string(want));
            ++full_values;
        }
    }
    std::cout << "        exhaustive: " << full_primes << " primes ell <= 131, " << full_values
              << " values of w, every w < ell - 1\n";

    // Large ell, sampled -- this is where the reach past the pinned table shows.
    struct Sample { uint32_t ell; uint32_t stride; };
    const Sample samples[] = { { 691, 23 }, { 1009, 37 }, { 3617, 1205 } };
    uint32_t sampled = 0;
    uint32_t w_max_seen = 0;
    for (const Sample& s : samples) {
        const std::vector<uint32_t> tab = bernoulli_mod_table(s.ell - 2u, s.ell);
        for (uint32_t w = 0; w + 1 < s.ell; w += s.stride) {
            Check(tab[w] == WorpitzkyBernoulli(w, s.ell),
                  "Worpitzky mismatch at (w, ell) = (" + std::to_string(w) + ", "
                      + std::to_string(s.ell) + ")");
            ++sampled;
            w_max_seen = std::max(w_max_seen, w);
        }
        // Always include the top of the range.
        const uint32_t w_top = s.ell - 2u;
        Check(tab[w_top] == WorpitzkyBernoulli(w_top, s.ell),
              "Worpitzky mismatch at the top of the range, ell = " + std::to_string(s.ell));
        ++sampled;
        w_max_seen = std::max(w_max_seen, w_top);
    }
    std::cout << "        sampled: ell in {691, 1009, 3617}, " << sampled
              << " values, largest w validated = " << w_max_seen << "\n";
}

// ===========================================================================
// Arm D: the Kummer congruence, the warrant for kummer_verdict's third branch.
// ===========================================================================
void ArmKummerCongruence() {
    std::cout << "  [D] the Kummer congruence B_k/k == B_{k'}/k' mod ell\n";
    uint32_t pairs = 0;
    for (uint64_t ell = 5; ell <= 101; ++ell) {
        if (!is_prime_u64(ell)) continue;
        const std::vector<uint32_t> tab =
            bernoulli_mod_table(static_cast<uint32_t>(ell) - 2u, static_cast<uint32_t>(ell));
        const uint32_t period = static_cast<uint32_t>(ell) - 1u;
        for (uint32_t kp = 2; kp < period; kp += 2) {
            const uint64_t lhs = mulmod(tab[kp], invmod(kp, ell), ell);
            // Every k <= 6 * period with k == kp (mod period) must give the same
            // value of B_k/k -- read through kummer_verdict, which is the branch
            // under test.
            for (uint32_t r = 1; r <= 6; ++r) {
                const uint32_t k = kp + r * period;
                const KummerVerdict v = kummer_verdict(k, static_cast<uint32_t>(ell));
                Check(v.applicable && v.reason == "kummer_congruence",
                      "kummer_verdict branch at (k, ell) = (" + std::to_string(k) + ", "
                          + std::to_string(ell) + ") is " + v.reason);
                Check(v.b_over_k_mod_ell == lhs,
                      "Kummer congruence at (k, k', ell) = (" + std::to_string(k) + ", "
                          + std::to_string(kp) + ", " + std::to_string(ell) + ")");
                Check(v.divides == (lhs == 0), "verdict follows B_{k'}/k' at k = "
                                                   + std::to_string(k));
                ++pairs;
            }
        }
    }
    std::cout << "        " << pairs << " (k, k', ell) congruence pairs\n";
}

// ===========================================================================
// Arm E: the Kummer / Harder predicate against the pinned numerators.
// ===========================================================================
void ArmKummerPredicate() {
    std::cout << "  [E] kummer_divides against integer divisibility of the pinned numerator\n";

    std::vector<uint64_t> primes;
    for (uint64_t p = 2; p <= 100; ++p)
        if (is_prime_u64(p)) primes.push_back(p);
    // The Kummer primes of [E] Claim 1.8 and their neighbours.
    const uint64_t extra[] = { 103, 131, 283, 593, 617, 691, 2294797, 3617, 43867, 657931 };
    for (uint64_t p : extra) primes.push_back(p);

    uint32_t cells = 0, hits = 0;
    for (size_t idx = 0; idx < kBernRefCount; ++idx) {
        const BernRef& ref = kBernRef[idx];
        const i128 num_over_k = i128_from_dec(ref.num_bk_over_k);
        for (uint64_t ell : primes) {
            const bool truth = (i128_abs(num_over_k) % static_cast<i128>(ell)) == 0;
            const KummerVerdict v = kummer_verdict(ref.k, static_cast<uint32_t>(ell));
            Check(v.applicable, "kummer_verdict applicable at k = " + std::to_string(ref.k));
            Check(v.divides == truth,
                  "kummer_divides(" + std::to_string(ref.k) + ", " + std::to_string(ell)
                      + ") = " + (v.divides ? "true" : "false") + " but ell | num(B_k/k) is "
                      + (truth ? "true" : "false") + " [" + v.reason + "]");
            Check(kummer_divides(ref.k, static_cast<uint32_t>(ell)) == truth,
                  "kummer_divides plain form at k = " + std::to_string(ref.k));
            ++cells;
            if (truth) ++hits;
        }
    }
    std::cout << "        " << cells << " (k, ell) cells, " << hits << " with ell | num(B_k/k)\n";

    // The four named divisors of [E] Claim 1.8 / the slot-4 plan.
    Check(kummer_divides(12, 691), "691 | num(B_12/12)");
    Check(kummer_divides(16, 3617), "3617 | num(B_16/16)");
    Check(kummer_divides(24, 103), "103 | num(B_24/24)");
    Check(kummer_divides(24, 2294797), "2294797 | num(B_24/24)");
    Check(kummer_divides(26, 657931), "657931 | num(B_26/26)");
    Check(!kummer_divides(26, 13), "13 does NOT divide num(B_26/26) -- the 13 is in B_26, not in B_26/26");
    std::cout << "        named divisors: 691|B_12, 3617|B_16, 103*2294797|B_24, "
                 "657931|num(B_26/26)\n";

    // Regularity of every prime below 37, and the first irregular pair.
    uint32_t regular_primes = 0;
    for (uint64_t ell = 3; ell <= 31; ++ell) {
        if (!is_prime_u64(ell)) continue;
        bool regular = true;
        for (uint32_t k = 2; k + 1 < ell; k += 2)
            if (kummer_divides(k, static_cast<uint32_t>(ell))) regular = false;
        Check(regular, "ell = " + std::to_string(ell) + " must be a REGULAR prime");
        ++regular_primes;
    }
    uint32_t irregular_index_count = 0;
    uint32_t irregular_k = 0;
    for (uint32_t k = 2; k + 1 < 37; k += 2)
        if (kummer_divides(k, 37)) { ++irregular_index_count; irregular_k = k; }
    Check(irregular_index_count == 1, "37 has exactly one irregular index");
    Check(irregular_k == 32, "the irregular index of 37 is 32, got " + std::to_string(irregular_k));
    std::cout << "        " << regular_primes
              << " primes 3 <= ell <= 31 certified REGULAR; 37 irregular at k = " << irregular_k
              << " (the classical first irregular prime)\n";
}

// ===========================================================================
// Arm F: the refusal.
// ===========================================================================
void ArmRefusal() {
    std::cout << "  [F] the named refusal outside 0 <= w < ell - 1\n";
    const uint32_t ells[] = { 2, 3, 5, 7, 11, 13, 691 };
    uint32_t refusals = 0, accepts = 0;
    for (uint32_t ell : ells) {
        Check(bernoulli_max_index(ell) == ell - 2u,
              "bernoulli_max_index at ell = " + std::to_string(ell));
        for (uint32_t w = 0; w < ell + 3u; ++w) {
            bool threw = false;
            uint32_t value = 0;
            try {
                value = bernoulli_mod(w, ell);
            } catch (const BernoulliOutOfRange&) {
                threw = true;
            }
            (void)value;
            const bool should_throw = (w >= ell - 1u);
            Check(threw == should_throw,
                  "bernoulli_mod(" + std::to_string(w) + ", " + std::to_string(ell) + ") "
                      + (threw ? "threw" : "returned") + ", expected "
                      + (should_throw ? "a refusal" : "a value"));
            if (threw) ++refusals; else ++accepts;
        }
    }
    bool threw0 = false;
    try { (void)kummer_divides(0, 691); } catch (const BernoulliOutOfRange&) { threw0 = true; }
    Check(threw0, "kummer_divides(0, ell) must refuse");
    Check(!kummer_verdict(0, 691).applicable, "kummer_verdict(0, ell) is not applicable");
    Check(kummer_verdict(1, 691).applicable && !kummer_verdict(1, 691).divides,
          "kummer_verdict(1, ell): num(B_1/1) = -1, no prime divides it");
    Check(kummer_verdict(9, 691).applicable && kummer_verdict(9, 691).divides
              && kummer_verdict(9, 691).reason == "odd_k_vanishes",
          "kummer_verdict at odd k >= 3: B_k = 0");
    std::cout << "        " << refusals << " refusals, " << accepts
              << " accepted, plus the k = 0 / k = 1 / odd-k cases\n";
}

// ===========================================================================
// The Glover arms.
// ===========================================================================

// --- F_{ell^2} as F_ell[t]/(t^2 + Bt + C), elements indexed a + b * ell -----
struct GF2 {
    uint32_t ell = 0;
    uint32_t m = 0; // ell^2 - 1
    uint32_t B = 0, C = 0;
    std::vector<uint32_t> mul_tab; // ell^2 x ell^2
    std::vector<uint32_t> pow_gen; // gamma^e, e = 0 .. m - 1
    std::vector<uint32_t> dlog;    // index -> e, or m for 0

    uint32_t mul(uint32_t x, uint32_t y) const { return mul_tab[static_cast<size_t>(x) * (ell * ell) + y]; }
    uint32_t add(uint32_t x, uint32_t y) const {
        const uint32_t a = (x % ell + y % ell) % ell;
        const uint32_t b = (x / ell + y / ell) % ell;
        return a + b * ell;
    }
    uint32_t emb(uint32_t a) const { return a % ell; } // F_ell -> F_{ell^2}
};

GF2 MakeGF2(uint32_t ell) {
    GF2 F;
    F.ell = ell;
    F.m = ell * ell - 1u;
    bool found = false;
    for (uint32_t b = 0; b < ell && !found; ++b) {
        for (uint32_t c = 0; c < ell && !found; ++c) {
            bool irr = true;
            for (uint32_t x = 0; x < ell; ++x)
                if ((x * x + b * x + c) % ell == 0) { irr = false; break; }
            if (irr) { F.B = b; F.C = c; found = true; }
        }
    }
    assert(found);

    const uint32_t q = ell * ell;
    F.mul_tab.assign(static_cast<size_t>(q) * q, 0u);
    for (uint32_t x = 0; x < q; ++x) {
        const uint32_t xa = x % ell, xb = x / ell;
        for (uint32_t y = 0; y < q; ++y) {
            const uint32_t ya = y % ell, yb = y / ell;
            // (xa + xb t)(ya + yb t) = xa ya + (xa yb + xb ya) t + xb yb t^2,
            // t^2 = -B t - C.
            const uint32_t hi = (xb * yb) % ell;
            const uint32_t a = ((xa * ya) % ell + ell - (hi * F.C) % ell) % ell;
            const uint32_t b = ((xa * yb + xb * ya) % ell + ell - (hi * F.B) % ell) % ell;
            F.mul_tab[static_cast<size_t>(x) * q + y] = a + b * ell;
        }
    }

    // a generator of F_{ell^2}^*
    uint32_t gen = 0;
    for (uint32_t g = 1; g < q; ++g) {
        uint32_t x = 1u % ell, order = 0;
        for (uint32_t k = 1; k <= F.m; ++k) {
            x = F.mul(x, g);
            if (x == 1u % ell) { order = k; break; }
        }
        if (order == F.m) { gen = g; break; }
    }
    assert(gen != 0);

    F.pow_gen.assign(F.m, 0u);
    F.dlog.assign(q, F.m);
    uint32_t x = 1u % ell;
    for (uint32_t e = 0; e < F.m; ++e) {
        F.pow_gen[e] = x;
        F.dlog[x] = e;
        x = F.mul(x, gen);
    }
    return F;
}

// --- the characteristic polynomial of a d x d matrix over F_ell -------------
// Hessenberg reduction + Cohen Alg. 2.2.9, 1-indexed internally.  Returns the
// monic coefficient vector, index = degree.
std::vector<uint32_t> CharPolyFp(std::vector<uint32_t> A, uint32_t d, uint32_t ell) {
    // H is 1-indexed: H[i * (d + 1) + j], 1 <= i, j <= d.
    std::vector<uint32_t> H(static_cast<size_t>(d + 1) * (d + 1), 0u);
    for (uint32_t i = 1; i <= d; ++i)
        for (uint32_t j = 1; j <= d; ++j)
            H[static_cast<size_t>(i) * (d + 1) + j] = A[static_cast<size_t>(i - 1) * d + (j - 1)];
    A.clear();

    auto at = [&](uint32_t i, uint32_t j) -> uint32_t& { return H[static_cast<size_t>(i) * (d + 1) + j]; };
    auto mmul = [&](uint32_t a, uint32_t b) { return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % ell); };
    auto msub = [&](uint32_t a, uint32_t b) { return (a + ell - b) % ell; };
    auto madd = [&](uint32_t a, uint32_t b) { return (a + b) % ell; };

    for (uint32_t m = 2; m + 1 <= d; ++m) {
        uint32_t piv = 0;
        for (uint32_t i = m; i <= d; ++i)
            if (at(i, m - 1) != 0u) { piv = i; break; }
        if (piv == 0u) continue;
        if (piv != m) {
            for (uint32_t j = 1; j <= d; ++j) std::swap(at(piv, j), at(m, j));
            for (uint32_t i = 1; i <= d; ++i) std::swap(at(i, piv), at(i, m));
        }
        const uint32_t t_inv = static_cast<uint32_t>(invmod(at(m, m - 1), ell));
        for (uint32_t i = m + 1; i <= d; ++i) {
            const uint32_t u = mmul(at(i, m - 1), t_inv);
            if (u == 0u) continue;
            for (uint32_t j = m - 1; j <= d; ++j) at(i, j) = msub(at(i, j), mmul(u, at(m, j)));
            for (uint32_t j = 1; j <= d; ++j) at(j, m) = madd(at(j, m), mmul(u, at(j, i)));
        }
    }

    // P[m] as a coefficient vector of length m + 1.
    std::vector<std::vector<uint32_t>> P(d + 1);
    P[0] = { 1u % ell };
    for (uint32_t m = 1; m <= d; ++m) {
        std::vector<uint32_t> cur(m + 1, 0u);
        // (X - H[m][m]) * P[m-1]
        const uint32_t h = at(m, m);
        for (uint32_t i = 0; i < P[m - 1].size(); ++i) {
            cur[i + 1] = madd(cur[i + 1], P[m - 1][i]);
            cur[i] = msub(cur[i], mmul(h, P[m - 1][i]));
        }
        uint32_t t = 1u % ell;
        for (uint32_t i = 1; i + 1 <= m; ++i) {
            t = mmul(t, at(m - i + 1, m - i));
            const uint32_t coef = mmul(t, at(m - i, m));
            if (coef != 0u)
                for (uint32_t j = 0; j < P[m - i - 1].size(); ++j)
                    cur[j] = msub(cur[j], mmul(coef, P[m - i - 1][j]));
        }
        P[m] = cur;
    }
    return P[d];
}

// det(c * I - A) over F_ell, by plain Gaussian elimination -- the independent
// cross-check of CharPolyFp.
uint32_t DetShifted(const std::vector<uint32_t>& A, uint32_t d, uint32_t c, uint32_t ell) {
    std::vector<uint32_t> M(A);
    for (uint32_t i = 0; i < d; ++i) {
        uint32_t& e = M[static_cast<size_t>(i) * d + i];
        e = (c + ell - e) % ell;
    }
    // Every off-diagonal entry must be negated too: c I - A.
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j)
            if (i != j) {
                uint32_t& e = M[static_cast<size_t>(i) * d + j];
                e = (ell - e) % ell;
            }
    uint32_t det = 1u % ell;
    for (uint32_t col = 0; col < d; ++col) {
        uint32_t piv = d;
        for (uint32_t r = col; r < d; ++r)
            if (M[static_cast<size_t>(r) * d + col] != 0u) { piv = r; break; }
        if (piv == d) return 0u;
        if (piv != col) {
            for (uint32_t j = 0; j < d; ++j)
                std::swap(M[static_cast<size_t>(piv) * d + j], M[static_cast<size_t>(col) * d + j]);
            det = (ell - det) % ell;
        }
        const uint32_t p = M[static_cast<size_t>(col) * d + col];
        det = static_cast<uint32_t>((static_cast<uint64_t>(det) * p) % ell);
        const uint32_t pinv = static_cast<uint32_t>(invmod(p, ell));
        for (uint32_t r = col + 1; r < d; ++r) {
            const uint32_t f = static_cast<uint32_t>(
                (static_cast<uint64_t>(M[static_cast<size_t>(r) * d + col]) * pinv) % ell);
            if (f == 0u) continue;
            for (uint32_t j = col; j < d; ++j) {
                uint32_t& e = M[static_cast<size_t>(r) * d + j];
                e = (e + ell
                     - static_cast<uint32_t>((static_cast<uint64_t>(f)
                                              * M[static_cast<size_t>(col) * d + j]) % ell))
                    % ell;
            }
        }
    }
    return det;
}

// Root multiset of a polynomial over F_ell, found in F_{ell^2}.
// Returns exponent multiplicities: out[e] = multiplicity of gamma^e.
bool RootExponents(const std::vector<uint32_t>& poly, const GF2& F, std::vector<uint32_t>& out) {
    const uint32_t q = F.ell * F.ell;
    std::vector<uint32_t> p;
    p.reserve(poly.size());
    for (uint32_t c : poly) p.push_back(F.emb(c));
    out.assign(F.m, 0u);
    uint32_t found = 0;
    const uint32_t deg0 = static_cast<uint32_t>(poly.size()) - 1u;
    for (uint32_t mu = 1; mu < q; ++mu) {
        for (;;) {
            // Horner
            uint32_t v = 0;
            for (size_t i = p.size(); i-- > 0;) v = F.add(F.mul(v, mu), p[i]);
            if (v != 0u) break;
            // synthetic division by (x - mu)
            const size_t n = p.size();
            std::vector<uint32_t> qpoly(n - 1, 0u);
            uint32_t carry = 0;
            for (size_t i = n - 1; i-- > 0;) {
                carry = F.add(p[i + 1], F.mul(carry, mu));
                qpoly[i] = carry;
            }
            p.swap(qpoly);
            out[F.dlog[mu]] += 1u;
            ++found;
            if (p.size() <= 1) break;
        }
        if (p.size() <= 1) break;
    }
    return found == deg0;
}

struct SemisimpleClass {
    uint32_t a = 0, b = 0;   // eigenvalue exponents: alpha = gamma^a, beta = gamma^b
    uint32_t g[4] = { 0, 0, 0, 0 }; // the 2x2 matrix over F_ell, row-major
    bool in_sl2 = false;
};

std::vector<SemisimpleClass> SemisimpleClasses(const GF2& F) {
    const uint32_t ell = F.ell;
    std::vector<SemisimpleClass> out;
    // Split (including the scalars): eigenvalue exponents that are multiples of
    // ell + 1, i.e. exactly F_ell^*.
    std::vector<uint32_t> fl;
    for (uint32_t e = 0; e < F.m; ++e)
        if (e % (ell + 1u) == 0u) fl.push_back(e);
    for (size_t i = 0; i < fl.size(); ++i) {
        for (size_t j = i; j < fl.size(); ++j) {
            SemisimpleClass c;
            c.a = fl[i];
            c.b = fl[j];
            const uint32_t u = F.pow_gen[c.a];
            const uint32_t v = F.pow_gen[c.b];
            assert(u < ell && v < ell);
            c.g[0] = u; c.g[1] = 0; c.g[2] = 0; c.g[3] = v;
            c.in_sl2 = (((c.a + c.b) % F.m) == 0u);
            out.push_back(c);
        }
    }
    // Non-split: alpha = gamma^e with (ell + 1) not dividing e, beta = alpha^ell.
    for (uint32_t e = 0; e < F.m; ++e) {
        if (e % (ell + 1u) == 0u) continue;
        const uint32_t e2 = static_cast<uint32_t>((static_cast<uint64_t>(e) * ell) % F.m);
        if (e > e2) continue; // one representative per Frobenius pair
        SemisimpleClass c;
        c.a = e;
        c.b = e2;
        const uint32_t alpha = F.pow_gen[e];
        const uint32_t beta = F.pow_gen[e2];
        const uint32_t tr = F.add(alpha, beta);
        const uint32_t nm = F.mul(alpha, beta);
        assert(tr < ell && nm < ell);
        // companion matrix of x^2 - tr x + nm
        c.g[0] = 0; c.g[1] = (ell - nm) % ell; c.g[2] = 1u % ell; c.g[3] = tr;
        c.in_sl2 = (((c.a + c.b) % F.m) == 0u);
        out.push_back(c);
    }
    return out;
}

struct GloverArmStats {
    uint32_t cells = 0;
    uint32_t sl2_cells = 0;
    uint32_t eigen_checks = 0;
};

void GloverBrauerArm(uint32_t ell, uint32_t n_max, bool full_det_check, GloverArmStats& st) {
    const GF2 F = MakeGF2(ell);
    const std::vector<SemisimpleClass> classes = SemisimpleClasses(F);

    for (uint32_t n = 0; n <= n_max; ++n) {
        const GloverDecomposition dec = GloverFactors(n, ell);
        const uint32_t d = n + 1u;

        for (size_t ci = 0; ci < classes.size(); ++ci) {
            const SemisimpleClass& c = classes[ci];

            FlatMatrix g(2, 2, ell);
            g.set_entry(0, 0, c.g[0]);
            g.set_entry(0, 1, c.g[1]);
            g.set_entry(1, 0, c.g[2]);
            g.set_entry(1, 1, c.g[3]);
            const FlatMatrix rho = ComputeSymmetricPowerMatrixFM(g, n);
            Check(rho.get_rows() == d && rho.get_cols() == d, "rho_n shape");

            std::vector<uint32_t> A(static_cast<size_t>(d) * d, 0u);
            for (uint32_t i = 0; i < d; ++i)
                for (uint32_t j = 0; j < d; ++j)
                    A[static_cast<size_t>(i) * d + j] = rho.get_entry(i, j);

            const std::vector<uint32_t> cp = CharPolyFp(A, d, ell);
            Check(cp.size() == static_cast<size_t>(d) + 1u && cp[d] == 1u % ell,
                  "char poly is monic of degree n + 1");

            if (full_det_check || n <= 2u) {
                // Independent cross-check of the Hessenberg routine.
                for (uint32_t x = 0; x < ell; ++x) {
                    uint32_t v = 0;
                    for (size_t i = cp.size(); i-- > 0;)
                        v = static_cast<uint32_t>((static_cast<uint64_t>(v) * x + cp[i]) % ell);
                    Check(v == DetShifted(A, d, x, ell),
                          "charpoly(x) != det(xI - rho_n(g)) at ell = " + std::to_string(ell)
                              + ", n = " + std::to_string(n) + ", x = " + std::to_string(x));
                }
            }

            std::vector<uint32_t> actual;
            const bool split = RootExponents(cp, F, actual);
            Check(split, "char poly of rho_n(g) must split over F_{ell^2} at ell = "
                             + std::to_string(ell) + ", n = " + std::to_string(n));
            if (!split) continue;

            // The predicted Teichmueller-exponent multiset, from the factor list.
            std::vector<uint32_t> predicted(F.m, 0u);
            for (const hecke_ext::GloverFactor& f : dec.factors) {
                for (uint32_t t = 0; t <= f.sym; ++t) {
                    const uint64_t e = (static_cast<uint64_t>(c.a) * (f.sym - t)
                                        + static_cast<uint64_t>(c.b) * t
                                        + static_cast<uint64_t>(c.a + c.b) * f.det_twist)
                                       % F.m;
                    predicted[static_cast<size_t>(e)] += f.multiplicity;
                }
            }
            Check(predicted == actual,
                  "BRAUER MISMATCH at ell = " + std::to_string(ell) + ", n = " + std::to_string(n)
                      + ", class (" + std::to_string(c.a) + ", " + std::to_string(c.b) + ")");
            ++st.eigen_checks;
            ++st.cells;
            if (c.in_sl2) ++st.sl2_cells;
        }
    }
}

void ArmGloverStructure() {
    std::cout << "  [G] GloverFactors: dimension, weight, central character, irreducibility\n";
    const uint32_t ells[] = { 2, 3, 5, 7, 11, 13, 691 };
    uint32_t cases = 0;
    for (uint32_t ell : ells) {
        const uint32_t period = (ell > 2u) ? ell - 1u : 1u;
        for (uint32_t n = 0; n <= 4u * ell + 5u; ++n) {
            const GloverDecomposition dec = GloverFactors(n, ell);
            Check(dec.dimension() == n + 1u,
                  "dim at (n, ell) = (" + std::to_string(n) + ", " + std::to_string(ell) + ")");
            Check(dec.n == n && dec.ell == ell, "echoed parameters");
            uint32_t total = 0;
            for (size_t i = 0; i < dec.factors.size(); ++i) {
                const hecke_ext::GloverFactor& f = dec.factors[i];
                Check(f.sym <= ell - 1u, "sym <= ell - 1");
                Check(f.det_twist < period, "det twist in Z/(ell - 1)");
                Check(f.multiplicity >= 1u, "multiplicity >= 1");
                Check(((f.sym + 2u * f.det_twist) % period) == (n % period),
                      "central character sym + 2 det == n mod ell - 1");
                if (i > 0)
                    Check(dec.factors[i - 1].sym < f.sym
                              || (dec.factors[i - 1].sym == f.sym
                                  && dec.factors[i - 1].det_twist < f.det_twist),
                          "canonical order");
                total += f.multiplicity;
            }
            Check(total == dec.length(), "length");
            const std::vector<uint32_t> sl2 = dec.sl2_multiplicities();
            uint32_t sl2_dim = 0, sl2_len = 0;
            for (uint32_t r = 0; r < ell; ++r) { sl2_dim += sl2[r] * (r + 1u); sl2_len += sl2[r]; }
            Check(sl2_dim == n + 1u, "SL_2 view keeps the dimension");
            Check(sl2_len == dec.length(), "SL_2 view keeps the length");
            if (n <= ell - 1u) {
                Check(dec.factors.size() == 1u && dec.factors[0].sym == n
                          && dec.factors[0].det_twist == 0u && dec.factors[0].multiplicity == 1u,
                      "Sym^n is irreducible for n <= ell - 1, at n = " + std::to_string(n));
            }
            ++cases;
        }
    }
    std::cout << "        " << cases << " (n, ell) decompositions\n";
    std::cout << "        example  " << GloverFactors(15, 5).to_string() << "\n";
    std::cout << "        example  " << GloverFactors(21, 7).to_string() << "\n";
    std::cout << "        json     " << GloverFactors(8, 5).to_json() << "\n";
}

void ArmGloverBrauer() {
    std::cout << "  [H] GloverFactors against the Brauer character of the real rho_n\n";
    GloverArmStats st;
    struct Cell { uint32_t ell; uint32_t n_max; bool full; };
    const Cell cells[] = {
        { 2, 6, true }, { 3, 9, true }, { 5, 15, true }, { 7, 21, true }, { 11, 22, false },
    };
    for (const Cell& c : cells) {
        const GloverArmStats before = st;
        GloverBrauerArm(c.ell, c.n_max, c.full, st);
        std::cout << "        ell = " << (c.ell < 10 ? " " : "") << c.ell << "  n <= " << c.n_max
                  << "  classes x n = " << (st.cells - before.cells)
                  << "  (SL_2 classes: " << (st.sl2_cells - before.sl2_cells) << ")"
                  << (c.full ? "  [det(xI - rho) cross-check at every x]" : "") << "\n";
    }
    std::cout << "        total " << st.cells << " (class, n) cells, " << st.sl2_cells
              << " of them in SL_2(F_ell); every Brauer character matched exponent for exponent\n";
}

} // namespace

uint32_t GloverBrauerCrossCheck(uint32_t ell, uint32_t n_max, uint32_t& cells,
                                uint32_t& sl2_cells, uint32_t& checks) {
    const uint32_t checks0 = Check.checks, fails0 = Check.fails;
    GloverArmStats st;
    GloverBrauerArm(ell, n_max, true, st);
    cells = st.cells;
    sl2_cells = st.sl2_cells;
    checks = Check.checks - checks0;
    return Check.fails - fails0;
}

uint32_t HeckePredictionsValidation() {
    Check.reset();
    std::cout << "================= HeckePredictionsValidation (slot 4 WP-7, E6) "
                 "=================\n";
    try {
        ArmPinAndBeyond();
        ArmIndependentFormula();
        ArmKummerCongruence();
        ArmKummerPredicate();
        ArmRefusal();
        ArmGloverStructure();
        ArmGloverBrauer();
    } catch (const std::exception& e) {
        std::cout << "HeckePredictionsValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        ++Check.fails;
    }
    std::cout << "PREDICTIONS checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    return Check.fails;
}
