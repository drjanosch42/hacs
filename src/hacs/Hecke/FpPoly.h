// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_FP_POLY_H
#define HECKE_FP_POLY_H

// Stage S10 -- dense univariate polynomial arithmetic over F_ell.
//
// The exact F_ell[X] toolkit backing the level-(c) block-diagonalization of
// the plan ("Definition: the three matrix output levels", level (c), in
// docs/implementation/hecke_action_implementation_plan.md Sec. 3): add / sub /
// mul / divrem / gcd / extgcd / eval, modular exponentiation (general
// g^e mod f and the Frobenius powers X^{ell^k} mod f), the division-free
// Berkowitz characteristic polynomial (binding choice of the plan: valid for
// ALL ell, including ell <= d, unlike Faddeev-LeVerrier which divides by the
// integers 1..d), the Krylov/RREF minimal polynomial, and certified
// factorization over the NON-algebraically-closed field F_ell.
//
// COEFFICIENT ORDER (binding, amendment A-N4): coefficients are stored
// LOW -> HIGH degree, i.e. coeffs[i] is the coefficient of X^i. This is the
// same order in which every to_json() writer under Hecke/ emits polynomial
// coefficient lists (constant term first). The zero polynomial is the empty
// coefficient vector; all stored polynomials are trimmed (no trailing zero
// coefficients) with entries reduced to canonical residues in [0, ell).
//
// MODULUS RANGE: ell must be prime. Coefficient arithmetic is backed by the
// ModArithmetic::mod_*_w width dispatchers, so the DEFAULT FpPoly = FpPolyT
// <uint32_t> is byte-identical to the pre-S13 code and exact for ell < 2^32,
// while the Stage-S13 FpPoly64 = FpPolyT<uint64_t> handles ell < 2^64 via the
// __int128 backend ([C, thm:backend]).
//
// CERTIFICATION INVARIANT (binding for this stage): randomization
// (Cantor-Zassenhaus equal-degree splitting, driven by a DETERMINISTIC-seed
// PRNG) may be used to FIND factors, but never to certify them. Every
// factorization returned by FactorPoly has already passed the deterministic,
// unconditional certificate CertifyFactorization (product identity + the
// gcd/Frobenius irreducibility proof for every claimed-irreducible factor);
// FactorPoly throws if the certificate fails.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

template<class Elt> class MatrixT; // ModArithmetic dense matrix (fwd; see ModArithmetic/Matrix.h)

namespace hecke {

template<class Elt>
class FpPolyT {
public:
    Elt ell;                    // prime modulus (0 only for the default-constructed sentinel)
    std::vector<Elt> coeffs;    // low -> high degree (A-N4); trimmed; zero poly <=> empty

    // Default ctor: the zero polynomial over the sentinel modulus ell = 0.
    // Provided so FpPolyT can live in std::map (PrimaryBlock uses
    // std::map<uint64_t, FpPoly>); arithmetic on a sentinel-modulus
    // polynomial throws.
    FpPolyT();
    explicit FpPolyT(Elt ell_);                     // zero polynomial over F_ell
    FpPolyT(Elt ell_, std::vector<Elt> c);          // reduces entries mod ell and trims

    static FpPolyT zero(Elt ell_);
    static FpPolyT one(Elt ell_);
    static FpPolyT X(Elt ell_);
    static FpPolyT constant(Elt ell_, Elt c);

    void trim();                       // drop trailing zero coefficients
    bool is_zero() const;
    int64_t degree() const;            // -1 for the zero polynomial
    Elt leading_coeff() const;         // 0 for the zero polynomial
    Elt coeff(uint32_t i) const;       // 0 beyond the stored degree
    bool is_monic() const;

    bool operator==(const FpPolyT& other) const;
    bool operator!=(const FpPolyT& other) const;

    FpPolyT operator+(const FpPolyT& other) const;
    FpPolyT operator-(const FpPolyT& other) const;
    FpPolyT operator*(const FpPolyT& other) const;
    FpPolyT scalar_mul(Elt c) const;
    FpPolyT negate() const;
    FpPolyT monic() const;             // throws on the zero polynomial
    FpPolyT derivative() const;
    Elt eval(Elt x) const;             // Horner

    // a = q*b + r with deg r < deg b. General (monic or non-monic) divisor;
    // throws on b == 0.
    static void divrem(const FpPolyT& a, const FpPolyT& b, FpPolyT& q, FpPolyT& r);
    static FpPolyT mod(const FpPolyT& a, const FpPolyT& b);        // remainder
    static FpPolyT divide_exact(const FpPolyT& a, const FpPolyT& b); // throws if remainder != 0

    // Monic gcd (Euclid); gcd(0, 0) = 0.
    static FpPolyT gcd(FpPolyT a, FpPolyT b);
    // g = u*a + v*b with g the monic gcd (extended Euclid).
    static FpPolyT extgcd(const FpPolyT& a, const FpPolyT& b, FpPolyT& u, FpPolyT& v);

    // g^e mod f by square-and-multiply on the exponent bits.
    static FpPolyT powmod(const FpPolyT& g, uint64_t e, const FpPolyT& f);

    // X^{ell^k} mod f via k successive powmod(., ell, f) steps.
    static FpPolyT frobenius_powmod(uint32_t k, const FpPolyT& f);

    // Plain (non-modular) power, small exponents.
    static FpPolyT pow(const FpPolyT& g, uint32_t e);

    std::string to_string() const;     // human-readable, high -> low degree
};

using FpPoly = FpPolyT<uint32_t>;
using FpPoly64 = FpPolyT<uint64_t>;

// Canonical strict order used everywhere a deterministic factor/block order
// is needed: by degree, then lexicographically on the (low -> high)
// coefficient vector.
template<class Elt>
bool CanonicalLess(const FpPolyT<Elt>& a, const FpPolyT<Elt>& b);

// ---------------------------------------------------------------------------
// Matrix polynomials (over ModArithmetic's MatrixT<Elt>; A.p is the modulus).
// ---------------------------------------------------------------------------

// Characteristic polynomial det(X*I - A), monic of degree d = A.rows.
// BINDING ALGORITHM (plan, stage S10): the division-free O(d^4) Berkowitz
// algorithm -- correct for every prime ell, including ell <= d.
template<class Elt>
FpPolyT<Elt> CharPolyBerkowitz(const MatrixT<Elt>& A);

// Minimal polynomial via Krylov sequences + the Matrix RREF/kernel machinery.
template<class Elt>
FpPolyT<Elt> MinPolyKrylov(const MatrixT<Elt>& A);

// P(A) by Horner's rule (matrix products over F_ell).
template<class Elt>
MatrixT<Elt> EvalPolyAtMatrix(const FpPolyT<Elt>& P, const MatrixT<Elt>& A);

// ---------------------------------------------------------------------------
// Factorization over F_ell, with deterministic unconditional certification.
// ---------------------------------------------------------------------------

template<class Elt>
struct FpPolyFactorT {
    FpPolyT<Elt> pi;         // monic irreducible
    uint32_t multiplicity;   // >= 1
};

template<class Elt>
struct FpPolyFactorizationT {
    Elt ell = 0;
    Elt leading_coeff = 1;                       // f = leading_coeff * prod pi^mult
    std::vector<FpPolyFactorT<Elt>> factors;     // canonically ordered (CanonicalLess), pairwise distinct
};

using FpPolyFactor = FpPolyFactorT<uint32_t>;
using FpPolyFactorization = FpPolyFactorizationT<uint32_t>;
using FpPolyFactor64 = FpPolyFactorT<uint64_t>;
using FpPolyFactorization64 = FpPolyFactorizationT<uint64_t>;

// Squarefree decomposition (deterministic; char-p Yun variant).
template<class Elt>
std::vector<std::pair<FpPolyT<Elt>, uint32_t>> SquarefreeDecomposition(const FpPolyT<Elt>& f);

// Deterministic, unconditional irreducibility certificate: 0 iff pi (monic,
// deg >= 1) satisfies gcd(pi, X^{ell^i} - X) = 1 for 1 <= i <= floor(deg pi / 2)
// AND X^{ell^{deg pi}} == X (mod pi). Nonzero return = not certified.
template<class Elt>
uint32_t CertifyIrreducible(const FpPolyT<Elt>& pi);

// Deterministic, unconditional factorization certificate: 0 iff the product
// identity holds and every claimed-irreducible factor passes CertifyIrreducible.
template<class Elt>
uint32_t CertifyFactorization(const FpPolyT<Elt>& f, const FpPolyFactorizationT<Elt>& fac);

// Full factorization of a nonzero f (squarefree -> distinct-degree ->
// equal-degree Cantor-Zassenhaus with a DETERMINISTIC-seed PRNG); the result
// is ALWAYS passed through CertifyFactorization before being returned.
template<class Elt>
FpPolyFactorizationT<Elt> FactorPoly(const FpPolyT<Elt>& f,
                                     uint64_t seed = 0x510C0FFEE5EED123ull);

} // namespace hecke

#endif // HECKE_FP_POLY_H
