// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// MpReal.h -- multiprecision binary floating-point reals and complex numbers on
// nf::BigInt (hecke-extensions slot 21, T2: the numerical conjugates of the
// degree-ell Ribet field polynomial, owner_q2_q4.md Q2 (b) "in-house
// multiprecision complex arithmetic (exp / log / roots of unity on BigInt)";
// owner ruling of 2026-10-03: no MPFR, no PARI). STL only.
//
// A value is man * 2^exp with a BigInt mantissa. Every operation that takes a
// precision `prec` returns its result truncated (toward zero) to at most prec
// mantissa bits, so its relative error is below 2^(1 - prec); additions drop an
// operand that lies more than prec + 2 bits below the other. There is no
// rigorous error bookkeeping (no interval arithmetic): the callers carry guard
// bits and certify their final results independently (T2: the same polynomial
// at two precisions and exactly by CRT, RibetField.h).
//
// The transcendental functions (exp, log, e^{i t}, pi, log 2) are the textbook
// ones: Taylor series after argument reduction by 2^t and t squarings (exp,
// e^{i t}), Newton's iteration y <- y + x e^{-y} - 1 with doubling precision
// (log), Machin's formula (pi) and 2 atanh(1/3) (log 2). They are accurate to
// the requested precision up to a few ulps (the guard bits are internal).
#include <cstdint>
#include <string>

#include "BigInt.h"

namespace nf {

struct MpReal {
    BigInt man;
    int64_t exp = 0;  // value = man * 2^exp
};

struct MpComplex {
    MpReal re, im;
};

// Exact magnitude shifts of a BigInt (the sign is kept; shr truncates toward zero).
BigInt bigint_shl(const BigInt& x, std::size_t bits);
BigInt bigint_shr(const BigInt& x, std::size_t bits);

MpReal mp_from_int(const BigInt& v);
MpReal mp_from_i64(int64_t v);
MpReal mp_from_double(double v);  // exact (a double is a dyadic rational)
MpReal mp_norm(MpReal x, unsigned prec);
inline bool mp_is_zero(const MpReal& x) { return x.man.is_zero(); }
inline int mp_sign(const MpReal& x) { return x.man.sign(); }
MpReal mp_neg(MpReal x);
MpReal mp_abs(MpReal x);
MpReal mp_ldexp(MpReal x, int64_t k);  // x * 2^k, exact

MpReal mp_add(const MpReal& a, const MpReal& b, unsigned prec);
MpReal mp_sub(const MpReal& a, const MpReal& b, unsigned prec);
MpReal mp_mul(const MpReal& a, const MpReal& b, unsigned prec);
MpReal mp_mul_i64(const MpReal& a, int64_t k, unsigned prec);
MpReal mp_div_u32(const MpReal& a, uint32_t d, unsigned prec);
MpReal mp_div(const MpReal& a, const MpReal& b, unsigned prec);  // NF_CHECK(b != 0)
int mp_cmp(const MpReal& a, const MpReal& b);                   // exact

// log2 |x| and x as doubles (approximate; mp_to_double may return +-inf).
double mp_log2_abs(const MpReal& x);  // -inf for x = 0
double mp_to_double(const MpReal& x);

// The nearest integer to x (ties away from zero) and log2 of the distance |x - round(x)|
// (-inf when x is an integer at its precision).
BigInt mp_round(const MpReal& x, double* log2_dist = nullptr);

// Constants and elementary functions at `prec` bits.
MpReal mp_pi(unsigned prec);
MpReal mp_ln2(unsigned prec);
// A context caches pi and log 2 at one precision (prec + 64 guard bits).
class MpContext {
public:
    explicit MpContext(unsigned prec);
    unsigned prec() const { return prec_; }
    const MpReal& pi() const { return pi_; }
    const MpReal& ln2() const { return ln2_; }
    MpReal exp(const MpReal& x) const;         // |x| < 2^40
    MpReal log(const MpReal& x) const;         // x > 0
    MpComplex expi(const MpReal& t) const;     // e^{i t} = cos t + i sin t, |t| < 2^40
    MpReal exp_at(const MpReal& x, unsigned prec) const;  // prec <= this->prec()

private:
    unsigned prec_;
    MpReal pi_, ln2_;
};

MpComplex mp_cadd(const MpComplex& a, const MpComplex& b, unsigned prec);
MpComplex mp_csub(const MpComplex& a, const MpComplex& b, unsigned prec);
MpComplex mp_cmul(const MpComplex& a, const MpComplex& b, unsigned prec);
MpComplex mp_cmul_real(const MpComplex& a, const MpReal& r, unsigned prec);
MpReal mp_cnorm(const MpComplex& a, unsigned prec);  // |a|^2

// Decimal rendering with `digits` significant digits (for reports and self-tests).
std::string mp_to_string(const MpReal& x, unsigned digits);

}  // namespace nf
