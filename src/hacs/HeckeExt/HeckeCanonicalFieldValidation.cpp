// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 5, WP-7 (phase E6) -- see HeckeCanonicalFieldValidation.h
// for the arm list and HeckeCanonicalField.h for the unit.
//
// This TU obeys check_heckeext_includes.sh rule 2 (no Console/, no Cuda/, no
// Engine/, no BlockFlatMatrix.h) and rule 4 (no unordered_).  Arms A-H touch NO
// pipeline object: FpPoly arithmetic, the unit and hecke::MathSettings.  Arm I
// (slot-7 WP-4) runs BuildAtlas under `conway`, which raises at the first cell
// (n = 0, ell = 5) -- the one place the driver runs here.

#include "HeckeCanonicalFieldValidation.h"

#include "HeckeAtlas.h"
#include "HeckeCanonicalField.h"
#include "../Hecke/HeckeSettings.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using hecke::FpPoly;
using hecke_ext::CanonicalField;
using hecke_ext::CanonicalFieldPolynomial;
using hecke_ext::CanonicalRoot;
using hecke_ext::CanonicalSplit;
using hecke_ext::CertifyCanonicalSplit;
using hecke_ext::FieldElt;
using hecke_ext::FieldPolynomialFromSettings;
using hecke_ext::FindCanonicalRoot;
using hecke_ext::FqPoly;
using hecke_ext::SplitInCanonicalField;

oc_util::Checker Check("  *** FAIL [canonfield] ");

std::string PolyText(const FpPoly& p) {
    std::string s = "[";
    for (size_t i = 0; i < p.coeffs.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(p.coeffs[i]);
    }
    return s + "]";
}

// ===========================================================================
// Validation-local oracles.  Nothing here is shipped, and nothing here shares
// a line with the unit.
// ===========================================================================

// The k-th monic polynomial of degree r over F_ell in CanonicalLess order.
// hecke::CanonicalLess compares the LOW -> HIGH coefficient vector
// lexicographically, so c_0 is the MOST significant digit of k and c_{r-1} the
// least.  Spelled out here independently of the unit's odometer.
FpPoly MonicAt(uint32_t ell, uint32_t r, uint64_t k) {
    std::vector<uint32_t> c(static_cast<size_t>(r) + 1u, 0u);
    c[r] = 1u;
    uint64_t t = k;
    for (uint32_t i = r; i-- > 0;) {
        c[i] = static_cast<uint32_t>(t % ell);
        t /= ell;
    }
    return FpPoly(ell, c);
}

// The k-th polynomial of degree EXACTLY d with an arbitrary leading
// coefficient is not needed: a monic divisor suffices, because F_ell[X] is a
// UFD and any factor can be scaled monic.
//
// INDEPENDENT irreducibility oracle: f (monic, degree r >= 1) is reducible iff
// some monic g with 1 <= deg g <= r/2 divides it.  Trial division only -- no
// gcd, no Frobenius, nothing hecke::CertifyIrreducible does.
bool BruteIrreducible(const FpPoly& f) {
    const uint32_t ell = f.ell;
    const int64_t r = f.degree();
    if (r < 1) return false;
    for (int64_t d = 1; 2 * d <= r; ++d) {
        uint64_t count = 1;
        for (int64_t i = 0; i < d; ++i) count *= ell;
        for (uint64_t k = 0; k < count; ++k) {
            const FpPoly g = MonicAt(ell, static_cast<uint32_t>(d), k);
            const FpPoly rem = FpPoly::mod(f, g);
            if (rem.is_zero()) return false;
        }
    }
    return true;
}

// (1/r) sum_{d | r} mu(d) ell^{r/d} -- the number of monic irreducibles of
// degree r over F_ell.
int64_t MoebiusIrreducibleCount(uint32_t ell, uint32_t r) {
    auto mu = [](uint32_t n) -> int {
        int sign = 1;
        for (uint32_t p = 2; p * p <= n; ++p) {
            if (n % p != 0) continue;
            n /= p;
            if (n % p == 0) return 0;
            sign = -sign;
        }
        if (n > 1) sign = -sign;
        return sign;
    };
    int64_t total = 0;
    for (uint32_t d = 1; d <= r; ++d) {
        if (r % d != 0) continue;
        const int m = mu(d);
        if (m == 0) continue;
        int64_t pw = 1;
        for (uint32_t i = 0; i < r / d; ++i) pw *= static_cast<int64_t>(ell);
        total += static_cast<int64_t>(m) * pw;
    }
    return total / static_cast<int64_t>(r);
}

// The independent model of F: an element is an FpPoly reduced modulo C, and
// multiplication is FpPoly::mod(a * b, C).  This is the repository's own
// polynomial arithmetic, which knows nothing of CanonicalField's reduction
// table.
FpPoly OracleMul(const FpPoly& a, const FpPoly& b, const FpPoly& C) {
    return FpPoly::mod(a * b, C);
}

// A cheap deterministic pseudo-random stream (seeded, never a clock).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    uint32_t below(uint32_t n) { return static_cast<uint32_t>(next() % n); }
};

FieldElt RandomElt(const CanonicalField& F, Rng& rng) {
    FieldElt a(F.degree(), 0u);
    for (uint32_t i = 0; i < F.degree(); ++i) a[i] = rng.below(F.ell());
    return a;
}

FqPoly RandomFqPoly(const CanonicalField& F, Rng& rng, uint32_t max_deg) {
    FqPoly p;
    const uint32_t d = rng.below(max_deg + 1u);
    for (uint32_t i = 0; i <= d; ++i) p.coeffs.push_back(RandomElt(F, rng));
    return hecke_ext::FqTrim(F, std::move(p));
}

// ===========================================================================
// Arm A -- C_{ell,r} against brute-force minimality
// ===========================================================================
void ArmCanonicalPolynomial() {
    std::cout << "-- arm A: C_{ell,r} = the CanonicalLess-least monic irreducible ([E] def:canon), "
                 "against exhaustive trial division\n";

    struct Cell { uint32_t ell, r; };
    const std::vector<Cell> cells = {
        {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8},
        {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5},
        {5, 1}, {5, 2}, {5, 3}, {5, 4},
        {7, 1}, {7, 2}, {7, 3},
        {11, 1}, {11, 2}, {11, 3},
        {13, 1}, {13, 2},
        {17, 2}, {19, 2}, {23, 2}, {31, 2}, {101, 2}, {691, 2},
    };

    for (const Cell& c : cells) {
        uint64_t space = 1;
        for (uint32_t i = 0; i < c.r; ++i) space *= c.ell;
        const FpPoly C = CanonicalFieldPolynomial(c.ell, c.r);
        // The local oracle's own enumeration order, checked before it is used
        // as one: MonicAt must be strictly CanonicalLess-ascending in k.
        for (uint64_t k = 0; k + 1 < space && k < 32; ++k) {
            Check(hecke::CanonicalLess(MonicAt(c.ell, c.r, k), MonicAt(c.ell, c.r, k + 1)),
                  "A: the oracle's MonicAt enumerates in CanonicalLess order");
        }
        Check(C.ell == c.ell && C.degree() == static_cast<int64_t>(c.r) && C.is_monic(),
              "A: C_{" + std::to_string(c.ell) + "," + std::to_string(c.r) + "} is monic of degree r");
        Check(hecke::CertifyIrreducible(C) == 0u, "A: C is a certified irreducible");
        Check(BruteIrreducible(C),
              "A: C is irreducible by the INDEPENDENT trial-division oracle too");

        // Exhaustive minimality, where the space is small enough to walk.  The
        // bound also keeps the two-oracle agreement cheap: BruteIrreducible is
        // O(ell^{r/2}) divisions per candidate.
        if (space <= 4096) {
            uint64_t first = space;         // sentinel
            int64_t  found = 0;
            for (uint64_t k = 0; k < space; ++k) {
                const FpPoly cand = MonicAt(c.ell, c.r, k);
                const bool brute = BruteIrreducible(cand);
                const bool cert = (hecke::CertifyIrreducible(cand) == 0u);
                if (brute != cert) {
                    Check(false, "A: the two irreducibility oracles disagree on " + PolyText(cand));
                }
                if (brute) {
                    ++found;
                    if (first == space) first = k;
                    // Every irreducible other than C is CanonicalLess-ABOVE C.
                    if (!(cand == C)) {
                        Check(hecke::CanonicalLess(C, cand),
                              "A: C is CanonicalLess-below the irreducible " + PolyText(cand));
                    }
                }
            }
            Check(first != space && MonicAt(c.ell, c.r, first) == C,
                  "A: C is the FIRST irreducible in the enumeration at (" +
                      std::to_string(c.ell) + ", " + std::to_string(c.r) + ")");
            Check(found == MoebiusIrreducibleCount(c.ell, c.r),
                  "A: the irreducible count matches the Moebius formula at (" +
                      std::to_string(c.ell) + ", " + std::to_string(c.r) + ") -- got " +
                      std::to_string(found) + ", want " +
                      std::to_string(MoebiusIrreducibleCount(c.ell, c.r)));
        }
    }

    // The classical pins, spelled out so a change in the order is visible.
    struct Pin { uint32_t ell, r; std::vector<uint32_t> coeffs; };
    // NOTE the order these follow from: c_0 is compared FIRST, so
    // C_{2,3} = Y^3 + Y^2 + 1 and NOT the more familiar Y^3 + Y + 1 (the two
    // agree in c_0 = 1 and the former has the smaller c_1).
    const std::vector<Pin> pins = {
        {2, 1, {0, 1}},                   // Y
        {2, 2, {1, 1, 1}},                // Y^2 + Y + 1
        {2, 3, {1, 0, 1, 1}},             // Y^3 + Y^2 + 1
        {2, 4, {1, 0, 0, 1, 1}},          // Y^4 + Y^3 + 1
        {3, 1, {0, 1}},                   // Y
        {3, 2, {1, 0, 1}},                // Y^2 + 1
        {5, 2, {1, 1, 1}},                // Y^2 + Y + 1
        {7, 2, {1, 0, 1}},                // Y^2 + 1
    };
    for (const Pin& p : pins) {
        const FpPoly C = CanonicalFieldPolynomial(p.ell, p.r);
        Check(C == FpPoly(p.ell, p.coeffs),
              "A: the pinned C_{" + std::to_string(p.ell) + "," + std::to_string(p.r) +
                  "} = " + PolyText(FpPoly(p.ell, p.coeffs)) + " (got " + PolyText(C) + ")");
    }

    // Determinism: a second call is the same object, and is_canonical() agrees.
    for (const Cell& c : cells) {
        const FpPoly a = CanonicalFieldPolynomial(c.ell, c.r);
        const FpPoly b = CanonicalFieldPolynomial(c.ell, c.r);
        Check(a == b, "A: C_{ell,r} is a deterministic function of (ell, r) ([E] cl:canon)");
        const CanonicalField F(c.ell, c.r);
        Check(F.is_canonical() && F.defining_poly() == a,
              "A: CanonicalField(ell, r) adopts C_{ell,r}");
    }
}

// ===========================================================================
// Arm B -- the field arithmetic against the FpPoly model
// ===========================================================================
void ArmFieldArithmetic() {
    std::cout << "-- arm B: F = F_ell[Y]/(C) against the independent FpPoly-mod-C model\n";

    struct Cell { uint32_t ell, r; };
    const std::vector<Cell> cells = {{2, 1}, {2, 3}, {2, 6}, {3, 1}, {3, 4},
                                     {5, 2}, {5, 3}, {7, 2}, {13, 3}, {691, 2}};
    Rng rng(0x5EEDC0DEull);

    for (const Cell& c : cells) {
        const CanonicalField F(c.ell, c.r);
        const FpPoly& C = F.defining_poly();
        Check(F.ell() == c.ell && F.degree() == c.r, "B: the field reports (ell, r)");
        Check(F.is_zero(F.zero()) && !F.is_zero(F.one()), "B: zero and one");
        Check(F.to_poly(F.one()) == FpPoly::one(c.ell), "B: one round-trips through to_poly");

        for (uint32_t trial = 0; trial < 64; ++trial) {
            const FieldElt a = RandomElt(F, rng);
            const FieldElt b = RandomElt(F, rng);
            const FieldElt d = RandomElt(F, rng);

            // mul against the oracle
            const FpPoly want = OracleMul(F.to_poly(a), F.to_poly(b), C);
            Check(F.to_poly(F.mul(a, b)) == want, "B: mul == FpPoly::mod(a*b, C)");

            // add / sub against FpPoly
            Check(F.to_poly(F.add(a, b)) == F.to_poly(a) + F.to_poly(b), "B: add == FpPoly +");
            Check(F.to_poly(F.sub(a, b)) == F.to_poly(a) - F.to_poly(b), "B: sub == FpPoly -");
            Check(F.is_zero(F.add(a, F.neg(a))), "B: a + (-a) = 0");

            // ring laws
            Check(F.mul(F.mul(a, b), d) == F.mul(a, F.mul(b, d)), "B: mul is associative");
            Check(F.mul(a, b) == F.mul(b, a), "B: mul is commutative");
            Check(F.mul(a, F.add(b, d)) == F.add(F.mul(a, b), F.mul(a, d)),
                  "B: mul distributes over add");
            Check(F.mul(a, F.one()) == a, "B: one is the identity");

            // inverse
            if (!F.is_zero(a)) {
                Check(F.mul(a, F.inv(a)) == F.one(), "B: a * a^{-1} = 1");
            }

            // pow and frobenius
            const uint64_t e = rng.next() % 97ull;
            FieldElt naive = F.one();
            for (uint64_t i = 0; i < e; ++i) naive = F.mul(naive, a);
            Check(F.pow(a, e) == naive, "B: pow == repeated multiplication");
            Check(F.frobenius(a) == F.pow(a, c.ell), "B: frobenius(a) = a^ell");

            // Fermat: a^{ell^r} = a, and frob fixes exactly F_ell.
            FieldElt fr = a;
            for (uint32_t i = 0; i < c.r; ++i) fr = F.frobenius(fr);
            Check(fr == a, "B: a^{ell^r} = a (Fermat in F)");
            const bool in_prime_field = (F.to_poly(a).degree() <= 0);
            Check((F.frobenius(a) == a) == in_prime_field,
                  "B: frob(a) = a iff a lies in F_ell");
        }

        // eval() as a ring homomorphism F_ell[X] -> F.
        for (uint32_t trial = 0; trial < 16; ++trial) {
            std::vector<uint32_t> pc, qc;
            for (uint32_t i = 0; i < 4; ++i) pc.push_back(rng.below(c.ell));
            for (uint32_t i = 0; i < 3; ++i) qc.push_back(rng.below(c.ell));
            const FpPoly P(c.ell, pc), Q(c.ell, qc);
            const FieldElt x = RandomElt(F, rng);
            Check(F.eval(P * Q, x) == F.mul(F.eval(P, x), F.eval(Q, x)),
                  "B: eval(P*Q) = eval(P)*eval(Q)");
            Check(F.eval(P + Q, x) == F.add(F.eval(P, x), F.eval(Q, x)),
                  "B: eval(P+Q) = eval(P)+eval(Q)");
        }

        // element_at is a bijection onto F in CanonicalLess order.
        uint64_t size = 1;
        bool small = true;
        for (uint32_t i = 0; i < c.r; ++i) {
            if (size > 4096) { small = false; break; }
            size *= c.ell;
        }
        if (small && size <= 4096) {
            std::vector<FieldElt> seen;
            seen.reserve(static_cast<size_t>(size));
            bool ordered = true;
            for (uint64_t k = 0; k < size; ++k) {
                const FieldElt e2 = F.element_at(k);
                if (k > 0 && !CanonicalField::less(seen.back(), e2)) ordered = false;
                seen.push_back(e2);
            }
            Check(ordered, "B: element_at enumerates F strictly increasing in CanonicalLess order");
            std::sort(seen.begin(), seen.end());
            Check(std::unique(seen.begin(), seen.end()) == seen.end(),
                  "B: element_at is injective on [0, |F|)");
            Check(seen.size() == static_cast<size_t>(size), "B: ... and therefore onto F");
            // less() IS hecke::CanonicalLess on the reduced representatives.
            for (uint64_t k = 0; k + 1 < size && k < 64; ++k) {
                const FieldElt x = F.element_at(k), y = F.element_at(k + 1);
                Check(CanonicalField::less(x, y) ==
                          hecke::CanonicalLess(F.to_poly(x), F.to_poly(y)),
                      "B: CanonicalField::less == hecke::CanonicalLess on to_poly");
            }
        }
    }
}

// ===========================================================================
// Arm C -- polynomials over F
// ===========================================================================
void ArmFqPoly() {
    std::cout << "-- arm C: FqPoly, the only new arithmetic of Stage 6\n";

    struct Cell { uint32_t ell, r; };
    const std::vector<Cell> cells = {{2, 1}, {2, 4}, {3, 2}, {5, 3}, {13, 2}};
    Rng rng(0xA5A5F00Dull);

    for (const Cell& c : cells) {
        const CanonicalField F(c.ell, c.r);
        for (uint32_t trial = 0; trial < 40; ++trial) {
            const FqPoly a = RandomFqPoly(F, rng, 6);
            const FqPoly b = RandomFqPoly(F, rng, 4);
            const FqPoly d = RandomFqPoly(F, rng, 3);

            Check(hecke_ext::FqAdd(F, a, b).coeffs == hecke_ext::FqAdd(F, b, a).coeffs,
                  "C: FqAdd is commutative");
            Check(hecke_ext::FqMul(F, a, b).coeffs == hecke_ext::FqMul(F, b, a).coeffs,
                  "C: FqMul is commutative");
            Check(hecke_ext::FqMul(F, a, hecke_ext::FqAdd(F, b, d)).coeffs ==
                      hecke_ext::FqAdd(F, hecke_ext::FqMul(F, a, b),
                                       hecke_ext::FqMul(F, a, d)).coeffs,
                  "C: FqMul distributes over FqAdd");
            Check(hecke_ext::FqSub(F, hecke_ext::FqAdd(F, a, b), b).coeffs == a.coeffs,
                  "C: (a + b) - b = a");

            if (!b.is_zero()) {
                FqPoly q, rem;
                hecke_ext::FqDivRem(F, a, b, q, rem);
                Check(rem.degree() < b.degree(), "C: deg rem < deg b");
                const FqPoly back = hecke_ext::FqAdd(F, hecke_ext::FqMul(F, q, b), rem);
                Check(back.coeffs == a.coeffs, "C: a = q*b + rem, re-multiplied out");
            }

            // gcd, against the definition, on a constructed common factor.
            if (!d.is_zero() && !a.is_zero() && !b.is_zero()) {
                const FqPoly ad = hecke_ext::FqMul(F, a, d);
                const FqPoly bd = hecke_ext::FqMul(F, b, d);
                const FqPoly g = hecke_ext::FqGcd(F, ad, bd);
                Check(!g.is_zero() && hecke_ext::FqMod(F, ad, g).is_zero() &&
                          hecke_ext::FqMod(F, bd, g).is_zero(),
                      "C: gcd divides both");
                Check(hecke_ext::FqMod(F, g, hecke_ext::FqMonic(F, d)).is_zero(),
                      "C: the constructed common factor divides the gcd");
                Check(F.is_zero(F.sub(g.coeffs.back(), F.one())) || g.is_zero(),
                      "C: the gcd is monic");
            }

            // FqPowMod against naive repeated multiplication.
            if (b.degree() >= 1) {
                const uint64_t e = rng.next() % 17ull;
                FqPoly naive = hecke_ext::FqMod(F, hecke_ext::FqConstant(F, F.one()), b);
                for (uint64_t i = 0; i < e; ++i)
                    naive = hecke_ext::FqMod(F, hecke_ext::FqMul(F, naive, a), b);
                Check(hecke_ext::FqPowMod(F, a, e, b).coeffs == naive.coeffs,
                      "C: FqPowMod == repeated multiply-and-reduce");
            }

            // FqEval as a ring homomorphism.
            const FieldElt x = RandomElt(F, rng);
            Check(hecke_ext::FqEval(F, hecke_ext::FqMul(F, a, b), x) ==
                      F.mul(hecke_ext::FqEval(F, a, x), hecke_ext::FqEval(F, b, x)),
                  "C: FqEval(a*b) = FqEval(a)*FqEval(b)");
            Check(hecke_ext::FqEval(F, hecke_ext::FqAdd(F, a, b), x) ==
                      F.add(hecke_ext::FqEval(F, a, x), hecke_ext::FqEval(F, b, x)),
                  "C: FqEval(a+b) = FqEval(a)+FqEval(b)");
            // An F_ell-polynomial evaluated through FqFromFp agrees with
            // CanonicalField::eval, which is the report layer's own route.
            std::vector<uint32_t> pc;
            for (uint32_t i = 0; i < 5; ++i) pc.push_back(rng.below(c.ell));
            const FpPoly P(c.ell, pc);
            Check(hecke_ext::FqEval(F, hecke_ext::FqFromFp(F, P), x) == F.eval(P, x),
                  "C: FqFromFp + FqEval == CanonicalField::eval");
        }
    }
}

// ===========================================================================
// Arm D -- root finding against exhaustion over F
// ===========================================================================
void ArmRootFinding() {
    std::cout << "-- arm D: [E] alg:split step 1 against EXHAUSTIVE root search over F\n";

    struct Cell { uint32_t ell, r; };
    const std::vector<Cell> cells = {{2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6},
                                     {3, 1}, {3, 2}, {3, 3}, {3, 4},
                                     {5, 1}, {5, 2}, {5, 3},
                                     {7, 2}, {11, 2}, {13, 2}};

    uint32_t roots_found = 0;
    uint32_t max_candidates = 0;
    for (const Cell& c : cells) {
        uint64_t size = 1;
        for (uint32_t i = 0; i < c.r; ++i) size *= c.ell;
        if (size > 4096) continue;

        const CanonicalField F(c.ell, c.r);
        // EVERY monic irreducible pi_w of degree r -- the report layer's real
        // input, where pi_w is the block's witness and C is the project's.
        for (uint64_t k = 0; k < size; ++k) {
            const FpPoly pi = MonicAt(c.ell, c.r, k);
            if (hecke::CertifyIrreducible(pi) != 0u) continue;

            // The truth: every element of F that pi kills.
            std::vector<FieldElt> truth;
            for (uint64_t j = 0; j < size; ++j) {
                const FieldElt x = F.element_at(j);
                if (F.is_zero(F.eval(pi, x))) truth.push_back(x);
            }
            Check(truth.size() == c.r,
                  "D: an irreducible of degree r has exactly r roots in F (got " +
                      std::to_string(truth.size()) + ")");

            const CanonicalRoot root = FindCanonicalRoot(F, pi);
            Check(root.ok, "D: FindCanonicalRoot succeeds on " + PolyText(pi) + " at ell = " +
                               std::to_string(c.ell) + " (" + root.failure + ")");
            if (!root.ok) continue;
            ++roots_found;
            if (root.candidates_used > max_candidates) max_candidates = root.candidates_used;

            Check(std::find(truth.begin(), truth.end(), root.rho) != truth.end(),
                  "D: the root found is a true root");
            std::vector<FieldElt> got = root.conjugates;
            std::vector<FieldElt> want = truth;
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            Check(got == want, "D: the r conjugates are EXACTLY the root set");

            FieldElt least = truth[0];
            for (const FieldElt& x : truth) {
                if (CanonicalField::less(x, least)) least = x;
            }
            Check(root.rho_canonical == least,
                  "D: rho_canonical is the CanonicalLess-least root");
            Check(root.conjugates.front() == root.rho_canonical,
                  "D: conjugates[0] == rho_canonical");
            for (uint32_t i = 0; i + 1 < c.r; ++i) {
                Check(F.frobenius(root.conjugates[i]) == root.conjugates[i + 1],
                      "D: conjugates[i+1] = conjugates[i]^ell");
            }
            Check(F.frobenius(root.conjugates.back()) == root.conjugates.front(),
                  "D: the Frobenius orbit closes");
        }
    }
    std::cout << "   CANONFIELD_ROOTS irreducibles_split=" << roots_found
              << " max_candidates_used=" << max_candidates << "\n";
    Check(roots_found >= 100, "D: the arm actually split a substantial number of polynomials");

    // Refusals, reported and never thrown.
    {
        const CanonicalField F(5, 2);
        const FpPoly reducible(5u, std::vector<uint32_t>{0u, 0u, 1u});   // Y^2
        const CanonicalRoot bad = FindCanonicalRoot(F, reducible);
        Check(!bad.ok && !bad.failure.empty(), "D: a reducible pi_w is REPORTED, not thrown");
        const FpPoly wrong_degree(5u, std::vector<uint32_t>{2u, 1u});    // degree 1 in F_25
        const CanonicalRoot bad2 = FindCanonicalRoot(F, wrong_degree);
        Check(!bad2.ok, "D: deg pi_w != [F : F_ell] is REPORTED");
        const FpPoly non_monic(5u, std::vector<uint32_t>{1u, 1u, 2u});
        Check(!FindCanonicalRoot(F, non_monic).ok, "D: a non-monic pi_w is REPORTED");
    }
}

// ===========================================================================
// Arm E -- the split, its certificate and a nine-way tamper
// ===========================================================================
void ArmSplitAndCertificate() {
    std::cout << "-- arm E: [E] alg:split steps 1-3, the certificate and the tamper arm\n";

    struct Cell { uint32_t ell, r; };
    const std::vector<Cell> cells = {{2, 1}, {2, 3}, {3, 2}, {5, 2}, {5, 3}, {7, 2}, {691, 2}};
    Rng rng(0xBEEFCAFEull);

    for (const Cell& c : cells) {
        const CanonicalField F(c.ell, c.r);
        // A witness polynomial that is NOT C, where one exists: that is the
        // case the report layer meets.
        FpPoly pi = F.defining_poly();
        uint64_t space = 1;
        for (uint32_t i = 0; i < c.r; ++i) space *= c.ell;
        for (uint64_t k = 0; k < space && k < 4096; ++k) {
            const FpPoly cand = MonicAt(c.ell, c.r, k);
            if (cand == F.defining_poly()) continue;
            if (hecke::CertifyIrreducible(cand) == 0u) { pi = cand; break; }
        }

        const std::vector<uint64_t> primes = {2, 3, 5, 7};
        std::vector<FpPoly> character;
        for (size_t j = 0; j < primes.size(); ++j) {
            std::vector<uint32_t> cf;
            for (uint32_t i = 0; i < c.r; ++i) cf.push_back(rng.below(c.ell));
            character.push_back(FpPoly(c.ell, cf));
        }

        const CanonicalSplit s = SplitInCanonicalField(c.ell, pi, primes, character);
        Check(s.ok, "E: the split succeeds at (" + std::to_string(c.ell) + ", " +
                        std::to_string(c.r) + "): " + s.failure);
        if (!s.ok) continue;

        Check(s.field_polynomial == CanonicalFieldPolynomial(c.ell, c.r),
              "E: the split names C_{ell,r}");
        Check(s.r == c.r && s.witness_poly == pi, "E: the split carries (r, pi_w)");
        for (size_t j = 0; j < primes.size(); ++j) {
            Check(s.eigenvalues[j] == F.eval(character[j], s.rho_canonical),
                  "E: a_q = P_q(rho_canonical), re-evaluated independently");
        }
        Check(s.orbit.size() == c.r && s.orbit[0] == s.eigenvalues,
              "E: orbit[0] == eigenvalues ([E] alg:split step 2)");
        for (uint32_t i = 0; i < c.r; ++i) {
            for (size_t j = 0; j < primes.size(); ++j) {
                Check(s.orbit[i][j] == F.eval(character[j], s.conjugates[i]),
                      "E: orbit[i][j] = P_{q_j}(rho^{ell^i})");
            }
        }
        Check(CertifyCanonicalSplit(s, character) == 0u, "E: the certificate returns 0");

        // Determinism, and independence of the caller's prime ORDER (the
        // eigenvalues must permute with the primes, nothing else).
        const CanonicalSplit again = SplitInCanonicalField(c.ell, pi, primes, character);
        Check(again.to_json() == s.to_json(), "E: two runs produce byte-identical JSON");
        {
            std::vector<uint64_t> rev(primes.rbegin(), primes.rend());
            std::vector<FpPoly> rev_ch(character.rbegin(), character.rend());
            const CanonicalSplit t = SplitInCanonicalField(c.ell, pi, rev, rev_ch);
            Check(t.ok && t.rho_canonical == s.rho_canonical,
                  "E: the canonical root does not depend on the prime order");
            for (size_t j = 0; j < primes.size(); ++j) {
                Check(t.eigenvalues[j] == s.eigenvalues[primes.size() - 1 - j],
                      "E: the eigenvalues permute with the primes");
            }
        }

        // The JSON fragment carries its schema and the data it claims.
        const std::string js = s.to_json();
        Check(js.find("\"schema\":\"hecke_canonical_split/v1\"") != std::string::npos,
              "E: to_json names its schema");
        Check(js.find("\"field_polynomial\":") != std::string::npos &&
                  js.find("\"frobenius_orbit\":") != std::string::npos,
              "E: to_json carries C_{ell,r} and the Frobenius orbit");
    }

    // ----------------------------------------------------------------------
    // THE TAMPER ARM.  Each corruption must fail EXACTLY its own clause.
    // ----------------------------------------------------------------------
    {
        const uint32_t ell = 5, r = 3;
        const CanonicalField F(ell, r);
        FpPoly pi = F.defining_poly();
        for (uint64_t k = 0; k < 125; ++k) {
            const FpPoly cand = MonicAt(ell, r, k);
            if (!(cand == F.defining_poly()) && hecke::CertifyIrreducible(cand) == 0u) {
                pi = cand;
                break;
            }
        }
        const std::vector<uint64_t> primes = {2, 3};
        const std::vector<FpPoly> character = {FpPoly(ell, std::vector<uint32_t>{1u, 2u, 0u}),
                                               FpPoly(ell, std::vector<uint32_t>{0u, 0u, 3u})};
        const CanonicalSplit good = SplitInCanonicalField(ell, pi, primes, character);
        Check(good.ok && CertifyCanonicalSplit(good, character) == 0u,
              "E-tamper: the positive control certifies");

        auto expect = [&](CanonicalSplit t, uint32_t clause, const std::string& what) {
            const uint32_t rc = CertifyCanonicalSplit(t, character);
            Check(rc == clause, "E-tamper: " + what + " fails clause " +
                                    std::to_string(clause) + " (got " + std::to_string(rc) + ")");
        };

        { CanonicalSplit t = good; t.ok = false;                       expect(t, 1, "ok cleared"); }
        { CanonicalSplit t = good; t.r = r + 1;                        expect(t, 1, "a wrong r"); }
        { CanonicalSplit t = good;
          t.witness_poly = FpPoly(ell, std::vector<uint32_t>{0u, 0u, 0u, 1u});   // X^3
          t.r = 3;                                                     expect(t, 2, "a reducible pi_w"); }
        { CanonicalSplit t = good;
          std::vector<uint32_t> cf = t.field_polynomial.coeffs; cf[0] = (cf[0] + 1u) % ell;
          t.field_polynomial = FpPoly(ell, cf);                        expect(t, 3, "a non-canonical C"); }
        { CanonicalSplit t = good;
          // Move the root to an element that is NOT a root -- a shifted root
          // could land on another conjugate, which clause 5 would catch first,
          // and this arm is about clause 4.
          for (uint32_t add = 1; add < ell; ++add) {
              FieldElt cand = good.rho_canonical;
              cand[0] = (cand[0] + add) % ell;
              if (!F.is_zero(F.eval(pi, cand))) { t.rho_canonical = cand; break; }
          }
          Check(t.rho_canonical != good.rho_canonical,
                "E-tamper: a non-root was found to move the anchor to");
          expect(t, 4, "a moved root"); }
        { CanonicalSplit t = good; t.conjugates.pop_back();            expect(t, 5, "a dropped conjugate"); }
        { CanonicalSplit t = good; std::swap(t.conjugates[1], t.conjugates[2]);
                                                                       expect(t, 5, "a permuted orbit"); }
        { CanonicalSplit t = good; t.conjugates[1] = t.conjugates[0];  expect(t, 5, "a repeated conjugate"); }
        { CanonicalSplit t = good;
          // Re-anchor on a NON-least conjugate: clauses 4-6 still hold, 7 must
          // catch it.  The eigenvalues and the orbit are recomputed for it, so
          // 8 and 9 stay green and 7 is the only clause that can fire.
          size_t worst = 0;
          for (size_t i = 1; i < t.conjugates.size(); ++i) {
              if (CanonicalField::less(t.conjugates[worst], t.conjugates[i])) worst = i;
          }
          if (worst != 0) {
              std::vector<FieldElt> rot;
              for (size_t i = 0; i < t.conjugates.size(); ++i)
                  rot.push_back(t.conjugates[(worst + i) % t.conjugates.size()]);
              t.conjugates = rot;
              t.rho_canonical = rot[0];
              t.eigenvalues.clear();
              for (const FpPoly& P : character) t.eigenvalues.push_back(F.eval(P, t.rho_canonical));
              t.orbit.clear();
              for (const FieldElt& x : t.conjugates) {
                  std::vector<FieldElt> row;
                  for (const FpPoly& P : character) row.push_back(F.eval(P, x));
                  t.orbit.push_back(row);
              }
              expect(t, 7, "a non-least anchor");
          }
        }
        { CanonicalSplit t = good; t.eigenvalues[0][0] =
              (t.eigenvalues[0][0] + 1u) % ell;                        expect(t, 8, "a moved eigenvalue"); }
        { CanonicalSplit t = good; t.orbit[1][0][0] =
              (t.orbit[1][0][0] + 1u) % ell;                           expect(t, 9, "a moved orbit entry"); }
    }
}

// ===========================================================================
// Arm F -- the (2, 30) case
// ===========================================================================
void ArmTwoThirty() {
    std::cout << "-- arm F: the (2, 30) case -- the field a HeckeLocalityValidation arm-7 witness "
                 "polynomial would be reported in\n";

    const FpPoly C = CanonicalFieldPolynomial(2, 30);
    Check(C.degree() == 30 && C.is_monic(), "F: C_{2,30} is monic of degree 30");
    Check(hecke::CertifyIrreducible(C) == 0u, "F: C_{2,30} is a certified irreducible");
    std::cout << "   CANONFIELD_230 C_2_30 = " << PolyText(C) << "\n";

    // MINIMALITY AT (2, 30), independently of the unit's odometer.  The
    // candidate space is 2^30, so the sweep is over the two halves separately:
    //
    //   (i)  c_0 = 0 -- the half the unit PRUNES.  The prune's warrant is the
    //        theorem "f(0) = 0 => Y | f", which is checked here as a property
    //        of FpPoly::mod on sampled degree-30 polynomials (the statement
    //        holds for all of them, so a sample validates the RULE; walking
    //        2^29 instances would validate nothing more).
    //   (ii) c_0 = 1, below C -- walked EXHAUSTIVELY, and each candidate is
    //        shown reducible by EXHIBITING a proper factor: trial division by
    //        every monic of degree <= 6 first, and where that finds nothing,
    //        hecke::FactorPoly's factor list with the product multiplied out
    //        here and the factor count asserted >= 2.  A reducibility verdict
    //        is therefore a witness, never CertifyIrreducible's word.
    {
        for (uint64_t s = 0; s < 64; ++s) {
            const FpPoly f = MonicAt(2, 30, s * 8191u);
            const FpPoly Y(2u, std::vector<uint32_t>{0u, 1u});
            Check(FpPoly::mod(f, Y).is_zero() == (f.coeff(0) == 0u),
                  "F(i): f(0) = 0 iff Y | f -- the warrant for the c_0 = 0 prune");
        }

        // The index of C in the FULL enumeration, recomputed from C's
        // coefficients with c_0 the MOST significant digit.
        uint64_t k_of_C = 0;
        for (uint32_t i = 0; i < 30; ++i) k_of_C = (k_of_C << 1) | (C.coeff(i) != 0u ? 1u : 0u);
        const uint64_t half = 1ull << 29;      // the first index with c_0 = 1
        Check(k_of_C >= half, "F(ii): C_{2,30} has c_0 = 1 (it must: c_0 = 0 would give Y | C)");

        uint32_t below = 0, by_small_divisor = 0, by_factorization = 0;
        for (uint64_t k = half; k < k_of_C; ++k) {
            ++below;
            const FpPoly cand = MonicAt(2, 30, k);
            bool red = false;
            for (uint32_t d = 1; d <= 6 && !red; ++d) {
                const uint64_t count = 1ull << d;
                for (uint64_t j = 0; j < count && !red; ++j) {
                    if (FpPoly::mod(cand, MonicAt(2, d, j)).is_zero()) red = true;
                }
            }
            if (red) {
                ++by_small_divisor;
                continue;
            }
            const hecke::FpPolyFactorization fac = hecke::FactorPoly(cand);
            uint32_t with_multiplicity = 0;
            FpPoly product = FpPoly::one(2u);
            for (const hecke::FpPolyFactor& f : fac.factors) {
                with_multiplicity += f.multiplicity;
                product = product * FpPoly::pow(f.pi, f.multiplicity);
            }
            Check(product == cand, "F(ii): the exhibited factorization multiplies back to the "
                                   "candidate");
            Check(with_multiplicity >= 2,
                  "F(ii): the candidate below C_{2,30} really is a product of >= 2 factors");
            ++by_factorization;
        }
        Check(below == by_small_divisor + by_factorization,
              "F(ii): every candidate strictly below C_{2,30} was shown reducible by a witness (" +
                  std::to_string(by_small_divisor) + " by a divisor of degree <= 6, " +
                  std::to_string(by_factorization) + " by an exhibited factorization, of " +
                  std::to_string(below) + ")");
        std::cout << "   CANONFIELD_230 candidates_below_C=" << below
                  << " small_divisor=" << by_small_divisor
                  << " factorised=" << by_factorization << "\n";
    }

    const CanonicalField F(2, 30);
    Check(F.is_canonical() && F.degree() == 30, "F: F = F_2[Y]/(C_{2,30})");

    // Field arithmetic at r = 30 against the FpPoly model.
    {
        Rng rng(0x2A30u);
        for (uint32_t t = 0; t < 8; ++t) {
            const FieldElt a = RandomElt(F, rng), b = RandomElt(F, rng);
            Check(F.to_poly(F.mul(a, b)) == FpPoly::mod(F.to_poly(a) * F.to_poly(b), C),
                  "F: mul at r = 30 == the FpPoly model");
            if (!F.is_zero(a)) Check(F.mul(a, F.inv(a)) == F.one(), "F: inv at r = 30");
        }
    }

    // THE STRESS CASE: split a degree-30 irreducible over F_2 -- the shape of
    // arm 7's witness polynomial -- inside F.  This is the char-2 trace-map
    // branch at the largest residue degree the repository produces.
    uint32_t split_done = 0;
    uint32_t seen = 0;
    const uint64_t first_c0_one = 1ull << 29;   // c_0 is the MOST significant digit
    for (uint64_t k = first_c0_one; k < first_c0_one + (1ull << 12) && split_done < 2; ++k) {
        const FpPoly pi = MonicAt(2, 30, k);
        if (hecke::CertifyIrreducible(pi) != 0u) continue;
        ++seen;
        // The first is C itself (k = the index of C); split the NEXT ones, so
        // the embedding is a genuine one and not the identity.
        if (pi == C) continue;
        const std::vector<uint64_t> primes = {2, 3};
        const std::vector<FpPoly> character = {
            FpPoly(2u, std::vector<uint32_t>(30, 1u)),
            FpPoly(2u, std::vector<uint32_t>{1u, 0u, 1u, 1u})};
        const CanonicalSplit s = SplitInCanonicalField(F, pi, primes, character);
        Check(s.ok, "F: a degree-30 irreducible over F_2 splits in F_2[Y]/(C_{2,30}): " + s.failure);
        if (!s.ok) continue;
        Check(F.is_zero(F.eval(pi, s.rho_canonical)), "F: pi_w(rho) = 0 at r = 30");
        Check(s.conjugates.size() == 30, "F: 30 conjugates");
        Check(CertifyCanonicalSplit(s, character) == 0u, "F: the certificate returns 0 at r = 30");
        std::cout << "   CANONFIELD_230 pi_w = " << PolyText(pi)
                  << " candidates_used=" << s.candidates_used << "\n";
        ++split_done;
    }
    Check(split_done == 2, "F: two degree-30 witnesses were split (found " +
                               std::to_string(split_done) + " of 2; irreducibles seen " +
                               std::to_string(seen) + ")");
}

// ===========================================================================
// Arm G -- the two settings keys
// ===========================================================================
void ArmSettings() {
    std::cout << "-- arm G: the `field_polynomial` and `split_field_report` mathematical keys\n";

    using hecke::FieldPolynomialMode;
    using hecke::MathSettings;
    using hecke::SplitFieldReport;

    const MathSettings& d = MathSettings::defaults();
    Check(d.is_default(), "G: the default MathSettings is default");
    Check(d.fingerprint() == 0ull, "G: the all-defaults fingerprint is still 0");
    Check(d.field_polynomial == FieldPolynomialMode::Least, "G: field_polynomial defaults to least");
    Check(d.split_field_report == SplitFieldReport::Canonical,
          "G: split_field_report defaults to canonical ([OD] OQ10)");

    const std::vector<std::string>& keys = MathSettings::keys();
    const std::vector<std::string> want = {"locality",         "u_ell",
                                           "sturm",            "sturm_unavailable",
                                           "commutant_budget", "final_mode",
                                           "field_polynomial", "split_field_report",
                                           // slot-7 WP-2, appended after these two
                                           "presentation",     "predictions",
                                           // slot-8 WP-3, appended at the tail
                                           "coefficient_ring"};
    Check(keys == want, "G: the registered key order (the two new keys APPENDED at the tail)");
    Check(MathSettings::known_key("field_polynomial") &&
              MathSettings::known_key("split_field_report"),
          "G: both keys are registered");

    // Parsing, canonical text, round trip, fingerprint.
    {
        MathSettings s;
        Check(s.set("field_polynomial", "conway", hecke::SettingSource::Spec),
              "G: set(field_polynomial = conway) is ACCEPTED (it is a legal value)");
        Check(s.field_polynomial == FieldPolynomialMode::Conway, "G: the parsed mode");
        Check(s.canonical_text() == "field_polynomial = conway\n", "G: the canonical text");
        Check(s.fingerprint() != 0ull, "G: a non-default field_polynomial fingerprints non-zero");
        Check(s.text("field_polynomial") == "conway", "G: text() is set()'s inverse");
        const std::string j = s.to_json();
        Check(j.find("\"field_polynomial\":{\"value\":\"conway\",\"source\":\"spec\"}") !=
                  std::string::npos,
              "G: to_json carries field_polynomial with its source");
        Check(j.find("\"split_field_report\":{\"value\":\"canonical\",\"source\":\"default\"}") !=
                  std::string::npos,
              "G: to_json carries split_field_report at its default");
        Check(s.set("field_polynomial", "least", hecke::SettingSource::Default),
              "G: back to least");
        Check(s.fingerprint() == 0ull, "G: the fingerprint is a function of the VALUES");
    }
    {
        MathSettings s;
        Check(s.set("split_field_report", "own_field", hecke::SettingSource::Console),
              "G: set(split_field_report = own_field)");
        Check(s.canonical_text() == "split_field_report = own_field\n", "G: the canonical text");
        Check(s.text("split_field_report") == "own_field", "G: text() is set()'s inverse");
        Check(s.fingerprint() != 0ull, "G: a non-default split_field_report fingerprints non-zero");
    }
    // Both non-default: the text is in REGISTRY ORDER.
    {
        MathSettings s;
        s.set("split_field_report", "own_field", hecke::SettingSource::Spec);
        s.set("field_polynomial", "conway", hecke::SettingSource::Spec);
        Check(s.canonical_text() ==
                  "field_polynomial = conway\nsplit_field_report = own_field\n",
              "G: the canonical text is in registry order, not in set() order");
    }
    // Refusals.
    for (const char* bad : {"", "Least", "conway ", "canonical", "own field", "1"}) {
        MathSettings s;
        bool threw = false;
        try { s.set("field_polynomial", bad, hecke::SettingSource::Spec); }
        catch (const std::invalid_argument&) { threw = true; }
        Check(threw, std::string("G: field_polynomial refuses \"") + bad + "\"");
    }
    for (const char* bad : {"", "own-field", "ownfield", "least", "Canonical"}) {
        MathSettings s;
        bool threw = false;
        try { s.set("split_field_report", bad, hecke::SettingSource::Spec); }
        catch (const std::invalid_argument&) { threw = true; }
        Check(threw, std::string("G: split_field_report refuses \"") + bad + "\"");
    }

    // The consumer.  `least` gives C_{ell,r}; `conway` raises the NAMED error.
    {
        const hecke::ScopedMathSettings restore(hecke::math_settings());
        Check(hecke::math_settings().field_polynomial == FieldPolynomialMode::Least,
              "G: the thread's settings start at the default");
        Check(FieldPolynomialFromSettings(5, 3) == CanonicalFieldPolynomial(5, 3),
              "G: `least` returns C_{ell,r}");
        {
            const hecke::ScopedSplitField guard(FieldPolynomialMode::Conway,
                                                SplitFieldReport::Canonical);
            Check(hecke::math_settings().fingerprint() != 0ull,
                  "G: the scoped non-default fingerprint is non-zero");
            bool named = false;
            std::string msg;
            try {
                (void)FieldPolynomialFromSettings(5, 3);
            } catch (const hecke::NotImplementedInThisBuild& e) {
                named = true;
                msg = e.what();
            }
            Check(named, "G: `conway` raises hecke::NotImplementedInThisBuild, never a parse error");
            Check(msg.compare(0, std::string(hecke::kNotImplementedInThisBuildTag).size(),
                              hecke::kNotImplementedInThisBuildTag) == 0,
                  "G: ... with the literal not_implemented_in_this_build: tag (console exit 5)");
            Check(msg.find("conway") != std::string::npos,
                  "G: ... naming the value that could not be honoured");
        }
        Check(hecke::math_settings().fingerprint() == 0ull,
              "G: the scope restored the defaults");
        Check(FieldPolynomialFromSettings(2, 4) == CanonicalFieldPolynomial(2, 4),
              "G: ... and `least` works again afterwards");
    }
}

// ===========================================================================
// Arm H -- the deterministic operation count of alg:split step 1 (slot-7 WP-4)
// ===========================================================================
//
// A constructed witness pi_w of degree r over F_ell: the first certified
// irreducible X^r + sum c_i X^i whose coefficients are walked by a fixed LCG
// seeded with (ell, r).  Deterministic, and NOT C_{ell,r} (C's root is Y, which
// would make the embedding the identity).
FpPoly ConstructedWitness(uint32_t ell, uint32_t r) {
    uint64_t state = 0x9E3779B97F4A7C15ull ^ (static_cast<uint64_t>(ell) << 20) ^ r;
    const FpPoly C = CanonicalFieldPolynomial(ell, r);
    for (uint32_t attempt = 0; attempt < 100000u; ++attempt) {
        std::vector<uint32_t> c(static_cast<size_t>(r) + 1u, 0u);
        c[r] = 1u;
        for (uint32_t i = 0; i < r; ++i) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            c[i] = static_cast<uint32_t>((state >> 33) % ell);
        }
        const FpPoly pi(ell, c);
        if (pi != C && hecke::CertifyIrreducible(pi) == 0u) return pi;
    }
    Check(false, "H: no constructed witness found");
    return FpPoly(ell);
}

// THE PRE-WP-4 ROUTE, kept here verbatim as an oracle: [E] alg:split step 1
// through the generic FqPoly layer (FqPowMod / FqMul / FqMod), exactly as
// HeckeCanonicalField.cpp computed it before slot-7 WP-4 replaced the
// arithmetic with the flat quotient ring and the precomputed Frobenius.  The
// fixed candidate sequence and the gcd outcomes must be identical, so the
// two routes must agree on rho, rho_canonical, the conjugates and
// candidates_used.  (The orbit uses F.frobenius, i.e. pow(a, ell), where the
// shipped route uses the Frobenius matrix.)
FqPoly OldSplitPoly(const CanonicalField& F, const FqPoly& f, const FieldElt& s) {
    using namespace hecke_ext;
    const uint32_t ell = F.ell();
    const uint32_t r = F.degree();
    FqPoly h;
    if (ell == 2u) {
        h.coeffs.push_back(F.zero());
        h.coeffs.push_back(s);
        h = FqTrim(F, std::move(h));
        if (h.is_zero()) return FqPoly();
        FqPoly t = FqMod(F, h, f);
        FqPoly acc = t;
        for (uint32_t i = 1; i < r; ++i) {
            t = FqMod(F, FqMul(F, t, t), f);
            acc = FqAdd(F, acc, t);
        }
        return acc;
    }
    h.coeffs.push_back(s);
    h.coeffs.push_back(F.one());
    h = FqTrim(F, std::move(h));
    FqPoly u = FqMod(F, h, f);
    for (uint32_t i = 1; i < r; ++i) {
        u = FqPowMod(F, u, static_cast<uint64_t>(ell), f);
        u = FqMod(F, FqMul(F, u, h), f);
    }
    u = FqPowMod(F, u, static_cast<uint64_t>((ell - 1u) / 2u), f);
    return FqSub(F, u, FqConstant(F, F.one()));
}

CanonicalRoot OldFindCanonicalRoot(const CanonicalField& F, const FpPoly& pi_w) {
    using namespace hecke_ext;
    CanonicalRoot out;
    const uint32_t r = F.degree();
    uint64_t field_size = 1;
    for (uint32_t i = 0; i < r; ++i) {
        if (field_size > 4096u + F.ell()) break;
        field_size *= F.ell();
    }
    const uint64_t begin = F.ell();
    const uint64_t cap = field_size < begin + 4096u ? field_size : begin + 4096u;
    FqPoly f = FqMonic(F, FqFromFp(F, pi_w));
    while (f.degree() > 1) {
        bool split = false;
        for (uint64_t k = begin; k < cap; ++k) {
            ++out.candidates_used;
            const FqPoly g = FqGcd(F, f, OldSplitPoly(F, f, F.element_at(k)));
            if (g.degree() >= 1 && g.degree() < f.degree()) {
                f = FqMonic(F, g);
                split = true;
                break;
            }
        }
        if (!split) {
            out.failure = "old route: no split within the cap";
            return out;
        }
    }
    out.rho = F.neg(f.coeffs[0]);
    std::vector<FieldElt> conj;
    FieldElt cur = out.rho;
    for (uint32_t i = 0; i < r; ++i) {
        conj.push_back(cur);
        cur = F.frobenius(cur);
    }
    size_t least = 0;
    for (size_t i = 1; i < conj.size(); ++i) {
        if (CanonicalField::less(conj[i], conj[least])) least = i;
    }
    out.rho_canonical = conj[least];
    cur = out.rho_canonical;
    for (uint32_t i = 0; i < r; ++i) {
        out.conjugates.push_back(cur);
        cur = F.frobenius(cur);
    }
    out.ok = true;
    return out;
}

// Least-squares slope of log(y) against log(x).
double FitExponent(const std::vector<double>& x, const std::vector<double>& y) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double n = static_cast<double>(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const double lx = std::log(x[i]), ly = std::log(y[i]);
        sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

void ArmOperationCount() {
    std::cout << "-- arm H: the deterministic operation count of FindCanonicalRoot "
                 "(slot-7 WP-4), constructed pi_w at r = 2..20\n";
    std::cout << "   CANONFIELD_OPS ell r candidates fl_mul f_mul f_inv poly_mul poly_divrem "
                 "frob_apply\n";
    for (uint32_t ell : {5u, 691u, 134217757u}) {
        std::vector<double> xs, ys;
        for (uint32_t r = 2; r <= 20; ++r) {
            const FpPoly pi = ConstructedWitness(ell, r);
            const CanonicalField F(ell, r);
            hecke_ext::CanonicalFieldOpCounts oc;
            CanonicalRoot root;
            {
                const hecke_ext::ScopedCanonicalFieldOpCounter scope(oc);
                root = FindCanonicalRoot(F, pi);
            }
            Check(root.ok, "H: FindCanonicalRoot succeeds at ell = " + std::to_string(ell) +
                               ", r = " + std::to_string(r) + " (" + root.failure + ")");
            Check(oc.candidates == root.candidates_used,
                  "H: the counter's candidates == candidates_used");
            std::cout << "   CANONFIELD_OPS " << ell << " " << r << " " << root.candidates_used
                      << " " << oc.fl_mul << " " << oc.f_mul << " " << oc.f_inv << " "
                      << oc.poly_mul << " " << oc.poly_divrem << " " << oc.frob_apply << "\n";
            // Two runs count the same: the count is a function of the input.
            hecke_ext::CanonicalFieldOpCounts again;
            {
                const hecke_ext::ScopedCanonicalFieldOpCounter scope(again);
                (void)FindCanonicalRoot(F, pi);
            }
            Check(again.fl_mul == oc.fl_mul && again.f_mul == oc.f_mul &&
                      again.poly_mul == oc.poly_mul && again.candidates == oc.candidates,
                  "H: the count is deterministic");
            if (r >= 10) {
                xs.push_back(static_cast<double>(r));
                ys.push_back(static_cast<double>(oc.fl_mul));
            }
            // The pre-WP-4 route as the oracle (cheap only at small r).
            if (r <= 10) {
                hecke_ext::CanonicalFieldOpCounts old_oc;
                CanonicalRoot old;
                {
                    const hecke_ext::ScopedCanonicalFieldOpCounter scope(old_oc);
                    old = OldFindCanonicalRoot(F, pi);
                }
                Check(old.ok && root.ok && old.candidates_used == root.candidates_used &&
                          old.rho == root.rho && old.rho_canonical == root.rho_canonical &&
                          old.conjugates == root.conjugates,
                      "H: the shipped route == the pre-WP-4 route (candidates_used, rho, "
                      "rho_canonical, conjugates) at ell = " + std::to_string(ell) +
                          ", r = " + std::to_string(r));
                std::cout << "   CANONFIELD_OPS_OLD " << ell << " " << r << " "
                          << old.candidates_used << " " << old_oc.fl_mul << " " << old_oc.f_mul
                          << " " << old_oc.f_inv << " " << old_oc.poly_mul << " "
                          << old_oc.poly_divrem << "\n";
            }
        }
        std::cout << "   CANONFIELD_OPS_FIT ell=" << ell
                  << " exponent_of_r(fl_mul, r=10..20)=" << FitExponent(xs, ys) << "\n";
    }
    // The two routes on every monic irreducible of small (ell, r), ell = 2 and
    // 3 included (the trace-map branch and the (ell - 1)/2 = 1 exponent).
    {
        uint32_t compared = 0;
        struct Cell { uint32_t ell, r; };
        for (const Cell c : {Cell{2, 2}, Cell{2, 5}, Cell{2, 8}, Cell{3, 3}, Cell{3, 5},
                             Cell{7, 3}, Cell{13, 4}}) {
            const CanonicalField F(c.ell, c.r);
            uint64_t size = 1;
            for (uint32_t i = 0; i < c.r; ++i) size *= c.ell;
            for (uint64_t k = 0; k < size && k < 3000u; ++k) {
                const FpPoly pi = MonicAt(c.ell, c.r, k);
                if (hecke::CertifyIrreducible(pi) != 0u) continue;
                const CanonicalRoot a = FindCanonicalRoot(F, pi);
                const CanonicalRoot b = OldFindCanonicalRoot(F, pi);
                Check(a.ok && b.ok && a.candidates_used == b.candidates_used && a.rho == b.rho &&
                          a.rho_canonical == b.rho_canonical && a.conjugates == b.conjugates,
                      "H: shipped == pre-WP-4 route on " + PolyText(pi) + " over F_" +
                          std::to_string(c.ell));
                ++compared;
            }
        }
        std::cout << "   CANONFIELD_OPS_ORACLE irreducibles_compared=" << compared << "\n";
        Check(compared >= 200, "H: the route comparison covered a substantial set");
    }
    // No counter installed outside a scope.
    Check(hecke_ext::CanonicalFieldOpCounter() == nullptr, "H: the scope uninstalled the counter");
}

// ===========================================================================
// Arm I -- the `field_polynomial` routing of every production entry point
// (slot-7 WP-4; slot-7 plan Sec. 2.3 item 3)
// ===========================================================================
template <class Fn>
bool RaisesNotImplemented(Fn&& fn) {
    try {
        fn();
    } catch (const hecke::NotImplementedInThisBuild& e) {
        return std::string(e.what()).find("conway") != std::string::npos;
    }
    return false;
}

void ArmConwayRouting() {
    std::cout << "-- arm I: `field_polynomial = conway` reaches EVERY production entry point "
                 "as NotImplementedInThisBuild\n";
    using hecke::FieldPolynomialMode;
    using hecke::SplitFieldReport;

    // Under `least`: the field is C_{ell,r}, and a split made now is the
    // input the conway arms below re-certify.
    const FpPoly pi = ConstructedWitness(691u, 5u);
    const std::vector<uint64_t> primes = {2, 3};
    const std::vector<FpPoly> character = {FpPoly(691u, std::vector<uint32_t>{3u, 1u}),
                                           FpPoly(691u, std::vector<uint32_t>{0u, 0u, 7u})};
    const CanonicalSplit least_split = SplitInCanonicalField(691u, pi, primes, character);
    Check(least_split.ok && CertifyCanonicalSplit(least_split, character) == 0u,
          "I: under `least` the split certifies");
    Check(CanonicalField(691u, 5u).defining_poly() == CanonicalFieldPolynomial(691u, 5u),
          "I: under `least` CanonicalField(ell, r) is C_{ell,r} (byte-identical routing)");

    uint32_t arms = 0;
    {
        const hecke::ScopedSplitField guard(FieldPolynomialMode::Conway, SplitFieldReport::Canonical);
        // (1) the constructor every production path builds its field with
        Check(RaisesNotImplemented([] { (void)CanonicalField(691u, 5u); }),
              "I(1): CanonicalField(ell, r) raises NotImplementedInThisBuild under conway");
        ++arms;
        // (2) `--bench ext`'s entry point (BenchDriver RunExt)
        Check(RaisesNotImplemented([&] { (void)SplitInCanonicalField(691u, pi, primes, character); }),
              "I(2): SplitInCanonicalField(ell, ...) -- the --bench ext path -- raises under conway");
        ++arms;
        // (3) the certificate's clause 3
        Check(RaisesNotImplemented([&] { (void)CertifyCanonicalSplit(least_split, character); }),
              "I(3): CertifyCanonicalSplit clause 3 raises under conway (never certifies "
              "against `least`)");
        ++arms;
        // (4) the atlas (BuildAtlas builds one CanonicalField per (ell, r))
        Check(RaisesNotImplemented([] { (void)hecke_ext::BuildAtlas(5u); }),
              "I(4): BuildAtlas raises under conway");
        ++arms;
        // (5) the settings consumer itself
        Check(RaisesNotImplemented([] { (void)FieldPolynomialFromSettings(691u, 5u); }),
              "I(5): FieldPolynomialFromSettings raises under conway");
        ++arms;
        // The console's exit-5 arm is ConsoleSettingsValidation (g)'s
        // `canonical_field_polynomial` under `set field_polynomial = "conway"`.
    }
    Check(hecke::math_settings().fingerprint() == 0ull, "I: the scope restored the defaults");
    const CanonicalSplit again = SplitInCanonicalField(691u, pi, primes, character);
    Check(again.to_json() == least_split.to_json(),
          "I: back under `least`, the split is byte-identical to the one before");
    std::cout << "   CANONFIELD_CONWAY arms=" << arms << "\n";
}

} // namespace

uint32_t HeckeCanonicalFieldValidation() {
    Check.reset();
    std::cout << "================= HeckeCanonicalFieldValidation (slot 5 WP-7, E6) "
                 "=================\n";
    try {
        ArmCanonicalPolynomial();
        ArmFieldArithmetic();
        ArmFqPoly();
        ArmRootFinding();
        ArmSplitAndCertificate();
        ArmTwoThirty();
        ArmSettings();
        ArmOperationCount();
        ArmConwayRouting();
    } catch (const std::exception& e) {
        std::cout << "HeckeCanonicalFieldValidation: TOP-LEVEL EXCEPTION: " << e.what()
                  << std::endl;
        ++Check.fails;
    }
    std::cout << "CANONICAL_FIELD checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    return Check.fails;
}
