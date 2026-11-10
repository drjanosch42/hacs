// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Polred.h -- polynomial reduction on top of the in-house LLL (hecke-extensions
// slot 21, T1; for T2's reduced field polynomial at 37, owner_q2_q4.md Q2 (b):
// "LLL + a polredabs analogue at degree 37"). STL only, exact.
//
// Source: Cohen GTM 138, Algorithm 4.4.11 (POLRED: LLL-reduce an integral basis
// w_1..w_n of K for a quadratic form approximating T2, take the characteristic
// polynomials of the reduced vectors) and Algorithm 4.4.12 (the
// "pseudo-canonical" choice: minimise v(Q) = (|disc Q|, size Q, |a_{n-1}|, ...,
// |a_0|) lexicographically, then the sign normalisation of step 3).
//
// What is NOT here: the maximal order (Cohen step 1, round 2) and the
// numerical T2 Gram matrix (step 2 for a field that is not totally real). The
// caller passes a Z-basis w_i of an order (as numerator rows in the power basis
// of alpha, with denominators) and an integral positive definite Gram matrix of
// the form to reduce -- exact (the trace form of a totally real field,
// trace_form_gram(); T2 = Tr(x xbar) of a CM field) or a scaled, rounded T2.
// "size" in v(Q) is then that Gram matrix's value on the candidate.
#include <cstdint>
#include <vector>

#include "BigInt.h"
#include "Lll.h"

namespace nf {

using IntPoly = std::vector<BigInt>;  // coefficients, constant term first

// p_0 .. p_N, p_k = sum of the k-th powers of the roots of the monic f in Z[x]
// (Newton's identities; p_0 = deg f).
std::vector<BigInt> power_sums(const IntPoly& f, unsigned N);
// The trace form Tr(alpha^{i+j}), 0 <= i, j < n, of Q[x]/(f), f monic in Z[x];
// positive definite iff f is totally real (then it is T2 on the power basis).
IntMat trace_form_gram(const IntPoly& f);
// disc(f) = det Tr(alpha^{i+j}) for monic f in Z[x] (0 iff f has a repeated root).
BigInt poly_discriminant(const IntPoly& f);
// a b mod f in Z[x], f monic, a and b of degree < deg f.
IntPoly poly_mulmod_monic(const IntPoly& a, const IntPoly& b, const IntPoly& f);
// The characteristic polynomial (monic, constant term first, degree n) of
// h(alpha)/den in Q(alpha), f monic in Z[x] of degree n, deg h < n, den > 0;
// NF_CHECK that it lies in Z[y] (h(alpha)/den an algebraic integer).
IntPoly charpoly_element(const IntPoly& f, const IntPoly& h, const BigInt& den = BigInt(1));
// Cohen 4.4.12 step 3: with d the largest degree d != n (mod 2) with a_d != 0,
// replace P by (-1)^n P(-X) if a_d > 0 (so that a_d < 0); P monic of degree n.
IntPoly polred_sign_normalise(IntPoly P);

struct PolredCandidate {
    std::vector<BigInt> coords;  // the reduced vector in the w basis (= row of the LLL's H)
    IntPoly num;                 // the element num(alpha) / den in the power basis
    BigInt den;
    BigInt size;                 // the Gram matrix's value on it (the reduced Gram diagonal)
    IntPoly charpoly;
    BigInt disc;                 // disc(charpoly)
    bool generates = false;      // disc != 0: charpoly squarefree, so the element generates K
    IntPoly normalised;          // polred_sign_normalise(charpoly) if generates
};
struct PolredResult {
    LllResult lll;
    std::vector<PolredCandidate> cands;  // one per reduced basis vector, in LLL order
    int best = -1;                       // index of the 4.4.12 choice, -1 if no candidate generates
    IntPoly best_poly() const { return best >= 0 ? cands[static_cast<std::size_t>(best)].normalised : IntPoly(); }
};

// POLRED on the order with Z-basis w_i = omega_num[i](alpha) / omega_den[i]
// (rows of length n = deg f) for the integral positive definite Gram matrix
// gram (n x n) of the form on that basis.
PolredResult polred(const IntPoly& f, const IntMat& omega_num, const std::vector<BigInt>& omega_den, const IntMat& gram,
                    const BigRational& delta = lll_default_delta());
// The same on Z[alpha] (w_i = alpha^i).
PolredResult polred_power_basis(const IntPoly& f, const IntMat& gram, const BigRational& delta = lll_default_delta());

}  // namespace nf
