// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// PolyAlg.h -- Q(alpha) arithmetic, Newton polygons and Hensel lifting
// (slot-19 WP-NF; [E] sec:lvnf "Q(alpha) arithmetic and ell-adic factorisation
// (Hensel, Newton polygon)").
//
// Polynomials are coefficient vectors, constant term first.
#include <cstdint>
#include <vector>

#include "BigInt.h"
#include "BigRational.h"

namespace nf {

// ---- Newton polygon of f in Z[x] at a prime p (lower convex hull of (i, v_p(a_i))).
struct NewtonSegment {
    int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // endpoints; slope (y1 - y0)/(x1 - x0)
    BigRational slope() const { return BigRational(BigInt(y1 - y0), BigInt(x1 - x0)); }
    int64_t length() const { return x1 - x0; }  // number of roots with valuation -slope
};
std::vector<NewtonSegment> newton_polygon(const std::vector<BigInt>& f, uint32_t p);

// ---- polynomials over Z/N (N < 2^32), coefficients reduced, trimmed (no trailing zeros).
using PolyMod = std::vector<uint32_t>;
PolyMod poly_trim(PolyMod a);
PolyMod poly_add(const PolyMod& a, const PolyMod& b, uint32_t N);
PolyMod poly_sub(const PolyMod& a, const PolyMod& b, uint32_t N);
PolyMod poly_mul(const PolyMod& a, const PolyMod& b, uint32_t N);
// a = q b + r, deg r < deg b; b's leading coefficient must be a unit mod N.
void poly_divrem(const PolyMod& a, const PolyMod& b, uint32_t N, PolyMod& q, PolyMod& r);
// s, t with s a + t b = 1 over F_p (a, b coprime mod the prime p); NF_CHECK otherwise.
void poly_xgcd_coprime(const PolyMod& a, const PolyMod& b, uint32_t p, PolyMod& s, PolyMod& t);
PolyMod poly_from_bigint(const std::vector<BigInt>& f, uint32_t N);

// Hensel: f in Z[x] monic, f == g h (mod p) with g monic and gcd(g, h) = 1 mod p.
// Returns (G, H) with f == G H (mod p^M), G == g, H == h (mod p), G monic, deg G = deg g.
struct HenselPair {
    PolyMod G, H;
    uint32_t modulus = 0;  // p^M
};
HenselPair hensel_lift(const std::vector<BigInt>& f, const PolyMod& g, const PolyMod& h, uint32_t p, unsigned M);

// ---- Q(alpha) = Q[x]/(f), f in Q[x] irreducible of degree d (not checked), elements of degree < d.
class NumberFieldQ {
public:
    using Elt = std::vector<BigRational>;
    explicit NumberFieldQ(std::vector<BigRational> minpoly);  // leading coefficient != 0
    unsigned degree() const { return d_; }
    Elt from_int(int64_t v) const;
    Elt alpha() const;  // the class of x (d >= 2)
    Elt add(const Elt& a, const Elt& b) const;
    Elt sub(const Elt& a, const Elt& b) const;
    Elt mul(const Elt& a, const Elt& b) const;
    Elt pow(const Elt& a, uint64_t e) const;
    Elt inv(const Elt& a) const;  // NF_CHECK(a != 0); extended Euclid over Q
    bool is_zero(const Elt& a) const;
    bool equal(const Elt& a, const Elt& b) const;

private:
    Elt reduce(std::vector<BigRational> a) const;
    std::vector<BigRational> f_;  // monic after construction
    unsigned d_ = 0;
};

}  // namespace nf
