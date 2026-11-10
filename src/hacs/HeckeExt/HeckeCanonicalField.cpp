// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 5, WP-7 (phase E6) -- HeckeExt/HeckeCanonicalField.
// See the header for the mathematics, the determinism statement and the
// certify-after-find rule.  [E] `def:canon`, `cl:canon`, `alg:split`.

#include "HeckeCanonicalField.h"

#include "../Hecke/HeckeSettings.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <sstream>

namespace hecke_ext {

namespace {

// The field's modular-arithmetic context.  Built ONCE per CanonicalField (it
// is a member, `ctx_`) rather than per operation: `init(prime)` costs a
// division and a regime decision, which is a real constant factor when a
// degree-r field multiply is r^2 of them.  Routing everything through ModCtx
// also keeps this file free of raw ModArithmetic::mod_* calls, so the census
// of docs/matrix-engine/tools/raw_mod_calls.sh needs no new allow-list entry.
engine::ModCtx<uint32_t> ctx_of(uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::HeckeCanonicalField: ell < 2");
    return engine::ModCtx<uint32_t>(ell);
}

// The JSON rendering of one F_ell-polynomial's coefficient list, low -> high
// (A-N4), the convention every to_json() under Hecke/ uses.
std::string fp_poly_json(const hecke::FpPoly& p) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < p.coeffs.size(); ++i) {
        if (i) oss << ",";
        oss << p.coeffs[i];
    }
    oss << "]";
    return oss.str();
}

// (slot-7 WP-4) The calling thread's operation counter; see the header.
thread_local CanonicalFieldOpCounts* t_op_counts = nullptr;

} // namespace

// ===========================================================================
// The operation counter
// ===========================================================================

ScopedCanonicalFieldOpCounter::ScopedCanonicalFieldOpCounter(CanonicalFieldOpCounts& counts)
    : previous_(t_op_counts) {
    t_op_counts = &counts;
}

ScopedCanonicalFieldOpCounter::~ScopedCanonicalFieldOpCounter() { t_op_counts = previous_; }

CanonicalFieldOpCounts* CanonicalFieldOpCounter() { return t_op_counts; }

// ===========================================================================
// C_{ell,r}
// ===========================================================================

hecke::FpPoly CanonicalFieldPolynomial(uint32_t ell, uint32_t r) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::CanonicalFieldPolynomial: ell < 2");
    ENGINE_CHECK_MSG(r >= 1, "hecke_ext::CanonicalFieldPolynomial: r = 0");

    // THE ORDER.  hecke::CanonicalLess is "degree, then `a.coeffs < b.coeffs`"
    // -- std::vector's lexicographic comparison on the LOW -> HIGH coefficient
    // vector, so the CONSTANT coefficient c_0 is the MOST significant digit and
    // c_{r-1} the least.  The candidates are therefore walked as an odometer
    // over (c_0, ..., c_{r-1}) whose rightmost digit turns fastest.  (It is the
    // opposite of the base-ell counting order, which would compare c_{r-1}
    // first; getting this backwards makes the answer a different, and larger,
    // irreducible.)
    //
    // THE ONE PRUNE.  For r >= 2, c_0 = 0 means f(0) = 0, i.e. Y | f, so the
    // whole c_0 = 0 subtree is reducible and is skipped as a subtree rather
    // than walked -- without it the sweep at (2, 30) would have to reject
    // 2^29 candidates one at a time before reaching c_0 = 1.  It is sound (a
    // theorem, not a heuristic) and it is exactly a prune: the enumeration
    // order of what remains is untouched, so the result is still the
    // CanonicalLess-least irreducible.  At r = 1 there is no prune -- Y itself
    // is the answer.
    //
    // The enumeration cannot run out -- there are (1/r) sum_{d|r} mu(d)
    // ell^{r/d} >= 1 monic irreducibles of degree r for every r >= 1, and by
    // the density ell / (r (ell - 1)) among the candidates that survive the
    // prune the first one is reached after O(r) certificates.  The loop is
    // bounded anyway so that a defect surfaces as a named failure, never a
    // hang.
    constexpr uint64_t kMaxCandidates = 1ull << 22;

    std::vector<uint32_t> c(static_cast<size_t>(r) + 1, 0u);
    c[r] = 1u;
    if (r >= 2) c[0] = 1u;   // the prune
    for (uint64_t tries = 0; tries < kMaxCandidates; ++tries) {
        const hecke::FpPoly cand(ell, c);
        if (cand.degree() == static_cast<int64_t>(r) && hecke::CertifyIrreducible(cand) == 0u) {
            return cand;
        }
        // Odometer step, rightmost digit first.
        uint32_t i = r;
        bool carried_out = true;
        while (i-- > 0) {
            if (++c[i] < ell) { carried_out = false; break; }
            c[i] = 0u;
        }
        if (carried_out) break;                    // the whole space is exhausted
        if (r >= 2 && c[0] == 0u) c[0] = 1u;       // (unreachable: carries only raise c_0)
    }
    ENGINE_CHECK_MSG(false,
                     "hecke_ext::CanonicalFieldPolynomial: no irreducible of the requested degree "
                     "was reached within the candidate bound, which is impossible");
    return hecke::FpPoly(ell);
}

hecke::FpPoly FieldPolynomialFromSettings(uint32_t ell, uint32_t r) {
    const hecke::MathSettings& s = hecke::math_settings();
    if (s.field_polynomial == hecke::FieldPolynomialMode::Conway) {
        // A LEGAL value this build cannot honour ([OD] OQ9).  The named error,
        // never a parse error and never a silent fall back to `least`: the
        // console reports it as exit code 5.
        throw hecke::NotImplementedInThisBuild(
            "field_polynomial = conway: Conway polynomial tables are not shipped in this build "
            "(the project-canonical `least` polynomial of [E] def:canon is what every record "
            "names); use field_polynomial = least");
    }
    return CanonicalFieldPolynomial(ell, r);
}

// ===========================================================================
// CanonicalField
// ===========================================================================

CanonicalField::CanonicalField(uint32_t ell, uint32_t r)
    // (slot-7 WP-4) Through the `field_polynomial` SETTING, never
    // CanonicalFieldPolynomial directly: under `least` (the default) this is
    // C_{ell,r}, byte for byte as before; under `conway` every production
    // path that builds its field here (`--bench ext`, the atlas, the record
    // consumer to come) raises hecke::NotImplementedInThisBuild instead of
    // silently computing with `least` ([OD] OQ9; slot-7 plan Sec. 2.3 item 3).
    : ell_(ell), r_(r), C_(FieldPolynomialFromSettings(ell, r)) {
    init();
}

CanonicalField::CanonicalField(uint32_t ell, const hecke::FpPoly& C) : ell_(ell), C_(C) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::CanonicalField: ell < 2");
    ENGINE_CHECK_MSG(C.ell == ell, "hecke_ext::CanonicalField: C is over a different modulus");
    ENGINE_CHECK_MSG(C.degree() >= 1, "hecke_ext::CanonicalField: deg C < 1");
    ENGINE_CHECK_MSG(C.is_monic(), "hecke_ext::CanonicalField: C is not monic");
    ENGINE_CHECK_MSG(hecke::CertifyIrreducible(C) == 0u,
                     "hecke_ext::CanonicalField: C is not a certified irreducible");
    r_ = static_cast<uint32_t>(C.degree());
    init();
}

void CanonicalField::init() {
    ctx_ = ctx_of(ell_);
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    // red_[i] = Y^{r+i} mod C, for 0 <= i < r.  Built by one shift-and-reduce
    // step at a time: Y^{r} mod C = -(c_0 + ... + c_{r-1} Y^{r-1}) because C is
    // monic, and Y^{r+i+1} = Y * Y^{r+i}, reduced with the same rule.
    red_.assign(r_, FieldElt(r_, 0u));
    FieldElt cur(r_, 0u);
    for (uint32_t i = 0; i < r_; ++i) cur[i] = ctx.neg(C_.coeff(i));
    red_[0] = cur;
    for (uint32_t i = 1; i < r_; ++i) {
        const uint32_t top = cur[r_ - 1];
        for (uint32_t j = r_ - 1; j > 0; --j) cur[j] = cur[j - 1];
        cur[0] = 0u;
        if (top != 0u) {
            for (uint32_t j = 0; j < r_; ++j) cur[j] = ctx.add(cur[j], ctx.mul(top, red_[0][j]));
        }
        red_[i] = cur;
    }
}

bool CanonicalField::is_canonical() const {
    return C_ == CanonicalFieldPolynomial(ell_, r_);
}

FieldElt CanonicalField::one() const {
    FieldElt a(r_, 0u);
    a[0] = 1u % ell_;
    return a;
}

FieldElt CanonicalField::generator() const {
    FieldElt a(r_, 0u);
    if (r_ == 1) {
        // F = F_ell[Y]/(C) with deg C = 1: Y is the root of C, i.e. -C(0).
        const engine::ModCtx<uint32_t>& ctx = ctx_;
        a[0] = ctx.neg(C_.coeff(0));
    } else {
        a[1] = 1u % ell_;
    }
    return a;
}

FieldElt CanonicalField::from_scalar(uint32_t a) const {
    FieldElt e(r_, 0u);
    e[0] = a % ell_;
    return e;
}

FieldElt CanonicalField::from_poly(const hecke::FpPoly& p) const {
    ENGINE_CHECK_MSG(p.is_zero() || p.ell == ell_,
                     "hecke_ext::CanonicalField::from_poly: wrong modulus");
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    FieldElt out(r_, 0u);
    for (size_t i = 0; i < p.coeffs.size(); ++i) {
        const uint32_t c = p.coeffs[i] % ell_;
        if (c == 0u) continue;
        if (i < r_) {
            out[i] = ctx.add(out[i], c);
        } else {
            // Reduce Y^i by repeated use of red_: fold the high part down one
            // power at a time.  Called only on small inputs (a P_p has degree
            // < r), so the simple loop is the right one.
            FieldElt acc(r_, 0u);
            acc[0] = 1u % ell_;
            for (size_t k = 0; k < i; ++k) {
                // acc <- acc * Y
                const uint32_t top = acc[r_ - 1];
                for (uint32_t j = r_ - 1; j > 0; --j) acc[j] = acc[j - 1];
                acc[0] = 0u;
                if (top != 0u) {
                    for (uint32_t j = 0; j < r_; ++j)
                        acc[j] = ctx.add(acc[j], ctx.mul(top, red_[0][j]));
                }
            }
            for (uint32_t j = 0; j < r_; ++j) out[j] = ctx.add(out[j], ctx.mul(c, acc[j]));
        }
    }
    return out;
}

hecke::FpPoly CanonicalField::to_poly(const FieldElt& a) const {
    ENGINE_CHECK_MSG(a.size() == r_, "hecke_ext::CanonicalField::to_poly: wrong width");
    return hecke::FpPoly(ell_, a);   // the constructor trims
}

bool CanonicalField::is_zero(const FieldElt& a) const {
    ENGINE_CHECK_MSG(a.size() == r_, "hecke_ext::CanonicalField::is_zero: wrong width");
    for (uint32_t x : a) {
        if (x != 0u) return false;
    }
    return true;
}

FieldElt CanonicalField::add(const FieldElt& a, const FieldElt& b) const {
    ENGINE_CHECK_MSG(a.size() == r_ && b.size() == r_,
                     "hecke_ext::CanonicalField::add: wrong width");
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    FieldElt out(r_, 0u);
    for (uint32_t i = 0; i < r_; ++i) out[i] = ctx.add(a[i], b[i]);
    return out;
}

FieldElt CanonicalField::sub(const FieldElt& a, const FieldElt& b) const {
    ENGINE_CHECK_MSG(a.size() == r_ && b.size() == r_,
                     "hecke_ext::CanonicalField::sub: wrong width");
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    FieldElt out(r_, 0u);
    for (uint32_t i = 0; i < r_; ++i) out[i] = ctx.sub(a[i], b[i]);
    return out;
}

FieldElt CanonicalField::neg(const FieldElt& a) const {
    ENGINE_CHECK_MSG(a.size() == r_, "hecke_ext::CanonicalField::neg: wrong width");
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    FieldElt out(r_, 0u);
    for (uint32_t i = 0; i < r_; ++i) out[i] = ctx.neg(a[i]);
    return out;
}

FieldElt CanonicalField::mul(const FieldElt& a, const FieldElt& b) const {
    ENGINE_CHECK_MSG(a.size() == r_ && b.size() == r_,
                     "hecke_ext::CanonicalField::mul: wrong width");
    const engine::ModCtx<uint32_t>& ctx = ctx_;
    // Schoolbook product into 2r - 1 residues, then one reduction pass with
    // red_.  r is small here (r = f, and f <= dim_block <~ 200), and this runs
    // once per report, so the simple route is the right one.
    uint64_t n_mul = 0;   // (slot-7 WP-4) published once, below
    std::vector<uint32_t> prod(2u * static_cast<size_t>(r_), 0u);
    for (uint32_t i = 0; i < r_; ++i) {
        if (a[i] == 0u) continue;
        for (uint32_t j = 0; j < r_; ++j) {
            if (b[j] == 0u) continue;
            prod[i + j] = ctx.add(prod[i + j], ctx.mul(a[i], b[j]));
            ++n_mul;
        }
    }
    // Fold the coefficients of Y^{r+i} down, HIGHEST FIRST, so that a fold
    // never feeds an index that has still to be folded (red_ has degree < r,
    // so a fold only ever writes below r anyway -- the order is for clarity).
    for (size_t i = 2u * static_cast<size_t>(r_); i-- > r_;) {
        const uint32_t c = prod[i];
        if (c == 0u) continue;
        prod[i] = 0u;
        const FieldElt& row = red_[i - r_];
        for (uint32_t j = 0; j < r_; ++j) prod[j] = ctx.add(prod[j], ctx.mul(c, row[j]));
        n_mul += r_;
    }
    if (CanonicalFieldOpCounts* oc = t_op_counts) {
        ++oc->f_mul;
        oc->fl_mul += n_mul;
    }
    return FieldElt(prod.begin(), prod.begin() + r_);
}

FieldElt CanonicalField::inv(const FieldElt& a) const {
    ENGINE_CHECK_MSG(!is_zero(a), "hecke_ext::CanonicalField::inv: the zero element");
    if (CanonicalFieldOpCounts* oc = t_op_counts) ++oc->f_inv;
    // Extended Euclid in F_ell[Y] against C: u*a + v*C = 1.
    hecke::FpPoly u(ell_), v(ell_);
    const hecke::FpPoly g = hecke::FpPoly::extgcd(to_poly(a), C_, u, v);
    ENGINE_CHECK_MSG(g.degree() == 0,
                     "hecke_ext::CanonicalField::inv: gcd(a, C) is not a unit, so C is reducible");
    return from_poly(u);
}

FieldElt CanonicalField::pow(const FieldElt& a, uint64_t e) const {
    FieldElt result = one();
    FieldElt base = a;
    while (e != 0) {
        if (e & 1ull) result = mul(result, base);
        e >>= 1;
        if (e != 0) base = mul(base, base);
    }
    return result;
}

FieldElt CanonicalField::frobenius(const FieldElt& a) const { return pow(a, ell_); }

FieldElt CanonicalField::eval(const hecke::FpPoly& P, const FieldElt& x) const {
    ENGINE_CHECK_MSG(P.is_zero() || P.ell == ell_,
                     "hecke_ext::CanonicalField::eval: wrong modulus");
    ENGINE_CHECK_MSG(x.size() == r_, "hecke_ext::CanonicalField::eval: wrong width");
    FieldElt acc = zero();
    for (size_t i = P.coeffs.size(); i-- > 0;) {
        acc = mul(acc, x);
        acc = add(acc, from_scalar(P.coeffs[i]));
    }
    return acc;
}

FieldElt CanonicalField::element_at(uint64_t index) const {
    // CanonicalLess order on F: the zero element first (degree -1), then the
    // elements of degree 0, 1, ..., r - 1, and inside each degree d
    // lexicographically on (c_0, ..., c_d) with c_0 the MOST significant digit
    // and c_d != 0.  For ell = 2 this is 0, 1, Y, Y + 1, Y^2, ... -- "s = 0, 1,
    // Y, ..." of [E] Alg. 6.4.
    FieldElt out(r_, 0u);
    if (index == 0) return out;
    uint64_t idx = index - 1;
    for (uint32_t d = 0; d < r_; ++d) {
        uint64_t block = ell_ - 1u;          // c_d ranges over the ell - 1 non-zero values
        bool overflow = false;
        for (uint32_t i = 0; i < d; ++i) {
            if (block > (~0ull) / ell_) { overflow = true; break; }
            block *= ell_;
        }
        if (overflow || idx < block) {
            out[d] = static_cast<uint32_t>(idx % (ell_ - 1u)) + 1u;
            uint64_t q = idx / (ell_ - 1u);
            for (uint32_t i = d; i-- > 0;) {
                out[i] = static_cast<uint32_t>(q % ell_);
                q /= ell_;
            }
            return out;
        }
        idx -= block;
    }
    ENGINE_CHECK_MSG(false, "hecke_ext::CanonicalField::element_at: index >= |F|");
    return out;
}

bool CanonicalField::less(const FieldElt& a, const FieldElt& b) {
    // Degree of the reduced representative first (the zero element, degree
    // -1, is least), then lexicographic low -> high.  This is exactly
    // hecke::CanonicalLess on to_poly(a), to_poly(b).
    size_t da = a.size(), db = b.size();
    while (da > 0 && a[da - 1] == 0u) --da;
    while (db > 0 && b[db - 1] == 0u) --db;
    if (da != db) return da < db;
    for (size_t i = 0; i < da; ++i) {
        if (a[i] != b[i]) return a[i] < b[i];
    }
    return false;
}

std::string CanonicalField::to_text(const FieldElt& a) const {
    ENGINE_CHECK_MSG(a.size() == r_, "hecke_ext::CanonicalField::to_text: wrong width");
    std::ostringstream oss;
    oss << "[";
    for (uint32_t i = 0; i < r_; ++i) {
        if (i) oss << ",";
        oss << a[i];
    }
    oss << "]";
    return oss.str();
}

// ===========================================================================
// FqPoly -- univariate polynomials over F
// ===========================================================================

FqPoly FqTrim(const CanonicalField& F, FqPoly a) {
    while (!a.coeffs.empty() && F.is_zero(a.coeffs.back())) a.coeffs.pop_back();
    return a;
}

FqPoly FqFromFp(const CanonicalField& F, const hecke::FpPoly& p) {
    FqPoly out;
    out.coeffs.reserve(p.coeffs.size());
    for (uint32_t c : p.coeffs) out.coeffs.push_back(F.from_scalar(c));
    return FqTrim(F, std::move(out));
}

FqPoly FqConstant(const CanonicalField& F, const FieldElt& c) {
    FqPoly out;
    out.coeffs.push_back(c);
    return FqTrim(F, std::move(out));
}

FqPoly FqAdd(const CanonicalField& F, const FqPoly& a, const FqPoly& b) {
    FqPoly out;
    const size_t n = a.coeffs.size() > b.coeffs.size() ? a.coeffs.size() : b.coeffs.size();
    out.coeffs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const FieldElt x = i < a.coeffs.size() ? a.coeffs[i] : F.zero();
        const FieldElt y = i < b.coeffs.size() ? b.coeffs[i] : F.zero();
        out.coeffs.push_back(F.add(x, y));
    }
    return FqTrim(F, std::move(out));
}

FqPoly FqSub(const CanonicalField& F, const FqPoly& a, const FqPoly& b) {
    FqPoly out;
    const size_t n = a.coeffs.size() > b.coeffs.size() ? a.coeffs.size() : b.coeffs.size();
    out.coeffs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const FieldElt x = i < a.coeffs.size() ? a.coeffs[i] : F.zero();
        const FieldElt y = i < b.coeffs.size() ? b.coeffs[i] : F.zero();
        out.coeffs.push_back(F.sub(x, y));
    }
    return FqTrim(F, std::move(out));
}

FqPoly FqMul(const CanonicalField& F, const FqPoly& a, const FqPoly& b) {
    if (a.is_zero() || b.is_zero()) return FqPoly();
    if (CanonicalFieldOpCounts* oc = t_op_counts) ++oc->poly_mul;
    FqPoly out;
    out.coeffs.assign(a.coeffs.size() + b.coeffs.size() - 1u, F.zero());
    for (size_t i = 0; i < a.coeffs.size(); ++i) {
        if (F.is_zero(a.coeffs[i])) continue;
        for (size_t j = 0; j < b.coeffs.size(); ++j) {
            if (F.is_zero(b.coeffs[j])) continue;
            out.coeffs[i + j] = F.add(out.coeffs[i + j], F.mul(a.coeffs[i], b.coeffs[j]));
        }
    }
    return FqTrim(F, std::move(out));
}

void FqDivRem(const CanonicalField& F, const FqPoly& a, const FqPoly& b, FqPoly& q, FqPoly& rem) {
    ENGINE_CHECK_MSG(!b.is_zero(), "hecke_ext::FqDivRem: division by the zero polynomial");
    if (CanonicalFieldOpCounts* oc = t_op_counts) ++oc->poly_divrem;
    q = FqPoly();
    rem = a;
    if (a.is_zero() || a.degree() < b.degree()) return;
    const FieldElt binv = F.inv(b.coeffs.back());
    const size_t qn = static_cast<size_t>(a.degree() - b.degree()) + 1u;
    q.coeffs.assign(qn, F.zero());
    for (size_t k = qn; k-- > 0;) {
        if (rem.coeffs.size() <= k + b.coeffs.size() - 1u) continue;
        const FieldElt c = F.mul(rem.coeffs[k + b.coeffs.size() - 1u], binv);
        if (F.is_zero(c)) continue;
        q.coeffs[k] = c;
        for (size_t j = 0; j < b.coeffs.size(); ++j) {
            rem.coeffs[k + j] = F.sub(rem.coeffs[k + j], F.mul(c, b.coeffs[j]));
        }
        rem = FqTrim(F, std::move(rem));
    }
    q = FqTrim(F, std::move(q));
    rem = FqTrim(F, std::move(rem));
}

FqPoly FqMod(const CanonicalField& F, const FqPoly& a, const FqPoly& b) {
    FqPoly q, rem;
    FqDivRem(F, a, b, q, rem);
    return rem;
}

FqPoly FqMonic(const CanonicalField& F, const FqPoly& a) {
    ENGINE_CHECK_MSG(!a.is_zero(), "hecke_ext::FqMonic: the zero polynomial");
    const FieldElt c = F.inv(a.coeffs.back());
    FqPoly out;
    out.coeffs.reserve(a.coeffs.size());
    for (const FieldElt& x : a.coeffs) out.coeffs.push_back(F.mul(x, c));
    return FqTrim(F, std::move(out));
}

FqPoly FqGcd(const CanonicalField& F, FqPoly a, FqPoly b) {
    while (!b.is_zero()) {
        FqPoly r = FqMod(F, a, b);
        a = b;
        b = r;
    }
    if (a.is_zero()) return a;
    return FqMonic(F, a);
}

FqPoly FqPowMod(const CanonicalField& F, const FqPoly& g, uint64_t e, const FqPoly& f) {
    ENGINE_CHECK_MSG(!f.is_zero(), "hecke_ext::FqPowMod: the zero modulus");
    FqPoly result = FqMod(F, FqConstant(F, F.one()), f);
    FqPoly base = FqMod(F, g, f);
    while (e != 0) {
        if (e & 1ull) result = FqMod(F, FqMul(F, result, base), f);
        e >>= 1;
        if (e != 0) base = FqMod(F, FqMul(F, base, base), f);
    }
    return result;
}

FieldElt FqEval(const CanonicalField& F, const FqPoly& a, const FieldElt& x) {
    FieldElt acc = F.zero();
    for (size_t i = a.coeffs.size(); i-- > 0;) {
        acc = F.mul(acc, x);
        acc = F.add(acc, a.coeffs[i]);
    }
    return acc;
}

// ===========================================================================
// [E] `alg:split` step 1 -- root finding with the FIXED candidate sequence
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// (slot-7 WP-4) THE QUOTIENT RING A = F[X]/(f), FLAT.
//
// What the root finder spends its time on is arithmetic in A = F[X]/(f) for
// the current monic factor f (degree d >= 2) of pi_w: the splitting polynomial
// below is a power of X + s in A.  Before WP-4 that arithmetic went through the
// generic FqPoly layer -- FqMul (d^2 field multiplies, each a freshly
// allocated vector with a Barrett reduction per coefficient product) followed
// by FqDivRem (another ~d^2 field multiplies, an F.inv per division, an
// FqTrim per quotient coefficient) -- and every power by ell was a
// square-and-multiply of ~1.5 log2(ell) such products.  The operation count of
// that route is O(r^5 log ell) F_ell-multiplications per candidate (arm H of
// HeckeCanonicalFieldValidation prints it).
//
// This class computes THE SAME ELEMENTS of A by two levers of the slot-7 plan
// (Sec. 3 WP-4 scope 3), neither of which touches the candidate sequence:
//
//   (1) THE FROBENIUS AS A PRECOMPUTED MAP.  u |-> u^ell is F_ell-linear on A
//       (A has characteristic ell), and on u = sum_i u_i X^i it is
//       sum_i Frob_F(u_i) * (X^ell)^i.  With Frob_F as an r x r matrix over
//       F_ell (one per field) and P_i = X^{i ell} mod f (d elements of A, one
//       set per f, shared by every candidate of that f), one application costs
//       about half a product in A instead of ~1.5 log2(ell) products.  This
//       replaces the r - 1 powerings by ell of the old route.
//   (2) NO CHURN: an element of A is one flat array of d*r residues; a product
//       accumulates the bivariate (X, Y) product in uint64 with ONE reduction
//       per k_max products (ModCtx::k_max, the engine's delayed-accumulation
//       bound), folds Y^{>= r} with the field's reduction rows and X^{>= d}
//       with precomputed rows X^{d+t} mod f -- no allocation in the hot loop,
//       no F.inv (f is monic), no trimming.
//
// Every result is an exact residue in [0, ell), so every element this class
// returns is bit-identical to what the FqPoly route computed; the gcd taken
// from it, the factor kept, the candidate that splits and `candidates_used`
// are therefore unchanged (HeckeCanonicalFieldValidation arm H compares both
// routes candidate by candidate).
class QuotientRing {
public:
    using Elt = std::vector<uint32_t>;   // d * r residues, index i * r + j <-> X^i Y^j

    // frob_rows[j * r + k] = coefficient k of (Y^j)^ell in F.
    QuotientRing(const CanonicalField& F, const FqPoly& f, const std::vector<uint32_t>& frob_rows)
        : F_(F), ctx_(F.mod_ctx()), red_(F.reduction_rows()), frob_rows_(frob_rows),
          r_(F.degree()), d_(static_cast<uint32_t>(f.degree())) {
        ENGINE_CHECK_MSG(d_ >= 2, "hecke_ext::QuotientRing: deg f < 2");
        ENGINE_CHECK_MSG(f.coeffs.back() == F.one(), "hecke_ext::QuotientRing: f is not monic");
        k_max_ = ctx_.k_max == 0 ? 1 : ctx_.k_max;
        acc_.assign(2u * static_cast<size_t>(r_), 0u);
        q_.assign((2u * static_cast<size_t>(d_) - 1u) * r_, 0u);
        // R_0 = X^d mod f = -(f_0 + ... + f_{d-1} X^{d-1}); R_{t+1} = X * R_t.
        R_.assign(static_cast<size_t>(d_ > 1 ? d_ - 1 : 0), Elt());
        Elt cur(size(), 0u);
        for (uint32_t m = 0; m < d_; ++m)
            for (uint32_t j = 0; j < r_; ++j) cur[m * r_ + j] = ctx_.neg(f.coeffs[m][j]);
        if (!R_.empty()) R_[0] = cur;
        for (size_t t = 1; t < R_.size(); ++t) {
            Elt next(size(), 0u);
            times_x(R_[t - 1], next, /*use_R0=*/cur);
            R_[t] = next;
        }
        R0_ = cur;
        // P_i = X^{i ell} mod f.  P_1 by left-to-right square-and-multiply
        // where "multiply" is by X (a shift).
        P_.assign(d_, Elt());
        P_[0] = one();
        Elt p1 = one();
        const uint64_t e = F.ell();
        int top = 63;
        while (top >= 0 && ((e >> top) & 1ull) == 0) --top;
        for (int b = top; b >= 0; --b) {
            Elt sq(size(), 0u);
            mul(p1, p1, sq);
            if ((e >> b) & 1ull) {
                times_x(sq, p1, R0_);
            } else {
                p1.swap(sq);
            }
        }
        if (d_ > 1) P_[1] = p1;
        for (uint32_t i = 2; i < d_; ++i) {
            P_[i].assign(size(), 0u);
            mul(P_[i - 1], p1, P_[i]);
        }
    }

    size_t size() const { return static_cast<size_t>(d_) * r_; }
    Elt zero() const { return Elt(size(), 0u); }
    Elt one() const {
        Elt o(size(), 0u);
        o[0] = 1u % F_.ell();
        return o;
    }

    // out = a * b in A.  out may not alias a or b.
    void mul(const Elt& a, const Elt& b, Elt& out) {
        uint64_t n_mul = 0;
        for (uint32_t t = 0; t + 1 < 2u * d_; ++t) {
            begin_acc();
            const uint32_t lo = t + 1 > d_ ? t + 1 - d_ : 0u;
            const uint32_t hi = t < d_ - 1 ? t : d_ - 1;
            for (uint32_t i = lo; i <= hi; ++i) n_mul += conv(&a[i * r_], &b[(t - i) * r_]);
            n_mul += fold(&q_[t * r_]);
        }
        reduce_x(out, n_mul);
        publish(n_mul, /*poly_mul=*/1, /*frob=*/0);
    }

    // out = a^ell in A (the precomputed Frobenius).  out may not alias a.
    void frob(const Elt& a, Elt& out) {
        uint64_t n_mul = 0;
        // w_i = Frob_F(a_i), into q_[i * r .. ] (reused as scratch; d <= 2d-1 rows).
        for (uint32_t i = 0; i < d_; ++i) {
            begin_acc();
            const uint32_t* ai = &a[i * r_];
            for (uint32_t j = 0; j < r_; ++j) {
                if (ai[j] == 0u) continue;
                bump(1);
                const uint32_t* row = &frob_rows_[static_cast<size_t>(j) * r_];
                for (uint32_t k = 0; k < r_; ++k) acc_[k] += static_cast<uint64_t>(ai[j]) * row[k];
                n_mul += r_;
            }
            for (uint32_t k = 0; k < r_; ++k) q_[i * r_ + k] = ctx_.reduce(acc_[k]);
        }
        // out_m = sum_i w_i * P_i[m].
        for (uint32_t m = 0; m < d_; ++m) {
            begin_acc();
            for (uint32_t i = 0; i < d_; ++i) n_mul += conv(&q_[i * r_], &P_[i][m * r_]);
            n_mul += fold(&out[m * r_]);
        }
        publish(n_mul, 0, 1);
    }

    // out = a * (X + s) in A.  out may not alias a.
    void mul_linear(const Elt& a, const FieldElt& s, Elt& out) {
        uint64_t n_mul = 0;
        const uint32_t* top = &a[(d_ - 1) * r_];
        for (uint32_t m = 0; m < d_; ++m) {
            begin_acc();
            if (m >= 1) add_plain(&a[(m - 1) * r_]);
            n_mul += conv(top, &R0_[m * r_]);
            n_mul += conv(s.data(), &a[m * r_]);
            n_mul += fold(&out[m * r_]);
        }
        publish(n_mul, 0, 0);
    }

    // out = a^e in A, e >= 1.
    void pow(const Elt& a, uint64_t e, Elt& out) {
        ENGINE_CHECK_MSG(e >= 1, "hecke_ext::QuotientRing::pow: e = 0");
        int top = 63;
        while (((e >> top) & 1ull) == 0) --top;
        Elt cur = a, tmp(size(), 0u);
        for (int b = top - 1; b >= 0; --b) {
            mul(cur, cur, tmp);
            if ((e >> b) & 1ull) {
                mul(tmp, a, cur);
            } else {
                cur.swap(tmp);
            }
        }
        out.swap(cur);
    }

    FqPoly to_fq(const Elt& a) const {
        FqPoly p;
        p.coeffs.reserve(d_);
        for (uint32_t m = 0; m < d_; ++m)
            p.coeffs.emplace_back(a.begin() + m * r_, a.begin() + (m + 1) * r_);
        return FqTrim(F_, std::move(p));
    }

private:
    // out = X * a (the shift, with X^d replaced by R0).
    void times_x(const Elt& a, Elt& out, const Elt& R0) {
        uint64_t n_mul = 0;
        const uint32_t* top = &a[(d_ - 1) * r_];
        for (uint32_t m = 0; m < d_; ++m) {
            begin_acc();
            if (m >= 1) add_plain(&a[(m - 1) * r_]);
            n_mul += conv(top, &R0[m * r_]);
            n_mul += fold(&out[m * r_]);
        }
        publish(n_mul, 0, 0);
    }

    // After the product's X-coefficients sit in q_ (2d - 1 field elements):
    // out_m = q_m + sum_{t >= d} q_t * (X^t mod f)_m.
    void reduce_x(Elt& out, uint64_t& n_mul) {
        for (uint32_t m = 0; m < d_; ++m) {
            begin_acc();
            add_plain(&q_[m * r_]);
            for (uint32_t t = d_; t + 1 < 2u * d_; ++t) n_mul += conv(&q_[t * r_], &R_[t - d_][m * r_]);
            n_mul += fold(&out[m * r_]);
        }
    }

    void begin_acc() {
        std::fill(acc_.begin(), acc_.end(), 0u);
        terms_ = 0;
    }
    // Make room for `k` more products per accumulator entry.
    void bump(uint64_t k) {
        if (terms_ + k > k_max_) {
            for (uint64_t& v : acc_) v = ctx_.reduce(v);
            terms_ = 1;
        }
        terms_ += k;
    }
    void add_plain(const uint32_t* a) {
        bump(1);
        for (uint32_t j = 0; j < r_; ++j) acc_[j] += a[j];
    }
    // acc += a(Y) * b(Y), both of degree < r.  Returns the products done.
    uint64_t conv(const uint32_t* a, const uint32_t* b) {
        bool az = true, bz = true;
        for (uint32_t j = 0; j < r_ && az; ++j) az = a[j] == 0u;
        if (az) return 0;
        for (uint32_t j = 0; j < r_ && bz; ++j) bz = b[j] == 0u;
        if (bz) return 0;
        bump(r_);
        uint64_t n = 0;
        for (uint32_t j = 0; j < r_; ++j) {
            const uint64_t aj = a[j];
            if (aj == 0u) continue;
            uint64_t* dst = &acc_[j];
            for (uint32_t l = 0; l < r_; ++l) dst[l] += aj * b[l];
            n += r_;
        }
        return n;
    }
    // Reduce acc (degree <= 2r - 2 in Y) modulo C and ell into out[0 .. r).
    uint64_t fold(uint32_t* out) {
        uint64_t n = 0;
        for (uint64_t& v : acc_) v = ctx_.reduce(v);
        terms_ = 1;
        for (size_t i = 2u * static_cast<size_t>(r_) - 1u; i-- > r_;) {
            const uint64_t c = acc_[i];
            if (c == 0u) continue;
            if (terms_ + 1 > k_max_) {
                for (uint32_t j = 0; j < r_; ++j) acc_[j] = ctx_.reduce(acc_[j]);
                terms_ = 1;
            }
            ++terms_;
            const FieldElt& row = red_[i - r_];
            for (uint32_t j = 0; j < r_; ++j) acc_[j] += c * row[j];
            n += r_;
        }
        for (uint32_t j = 0; j < r_; ++j) out[j] = ctx_.reduce(acc_[j]);
        return n;
    }
    void publish(uint64_t n_mul, uint64_t poly_mul, uint64_t frob) {
        if (CanonicalFieldOpCounts* oc = t_op_counts) {
            oc->fl_mul += n_mul;
            oc->poly_mul += poly_mul;
            oc->frob_apply += frob;
        }
    }

    const CanonicalField& F_;
    const engine::ModCtx<uint32_t>& ctx_;
    const std::vector<FieldElt>& red_;
    const std::vector<uint32_t>& frob_rows_;
    uint32_t r_, d_;
    uint64_t k_max_ = 1;
    uint64_t terms_ = 0;
    std::vector<uint64_t> acc_;   // 2r entries (the last is unused)
    std::vector<uint32_t> q_;     // (2d - 1) field elements of scratch
    std::vector<Elt> R_;          // R_[t] = X^{d+t} mod f, 0 <= t < d - 1
    Elt R0_;                      // X^d mod f
    std::vector<Elt> P_;          // P_[i] = X^{i ell} mod f, 0 <= i < d
};

// The Frobenius of F as a matrix: rows[j * r + k] = coefficient k of (Y^j)^ell.
std::vector<uint32_t> FrobeniusRows(const CanonicalField& F) {
    const uint32_t r = F.degree();
    std::vector<uint32_t> rows(static_cast<size_t>(r) * r, 0u);
    for (uint32_t j = 0; j < r; ++j) {
        FieldElt e(r, 0u);
        e[j] = 1u % F.ell();
        const FieldElt fe = F.frobenius(e);
        for (uint32_t k = 0; k < r; ++k) rows[static_cast<size_t>(j) * r + k] = fe[k];
    }
    return rows;
}

// a^ell in F through the matrix (== F.frobenius(a), exactly).
FieldElt ApplyFrobenius(const CanonicalField& F, const std::vector<uint32_t>& rows,
                        const FieldElt& a) {
    const uint32_t r = F.degree();
    const engine::ModCtx<uint32_t>& ctx = F.mod_ctx();
    const uint64_t k_max = ctx.k_max == 0 ? 1 : ctx.k_max;
    std::vector<uint64_t> acc(r, 0u);
    uint64_t terms = 0, n_mul = 0;
    for (uint32_t j = 0; j < r; ++j) {
        if (a[j] == 0u) continue;
        if (terms + 1 > k_max) {
            for (uint64_t& v : acc) v = ctx.reduce(v);
            terms = 1;
        }
        ++terms;
        for (uint32_t k = 0; k < r; ++k)
            acc[k] += static_cast<uint64_t>(a[j]) * rows[static_cast<size_t>(j) * r + k];
        n_mul += r;
    }
    FieldElt out(r, 0u);
    for (uint32_t k = 0; k < r; ++k) out[k] = ctx.reduce(acc[k]);
    if (CanonicalFieldOpCounts* oc = t_op_counts) {
        oc->fl_mul += n_mul;
        ++oc->frob_apply;
    }
    return out;
}

// The splitting polynomial of [E] Alg. 6.4 step 1, for the candidate s.
//
//   ell odd:  (X + s)^{(q-1)/2} - 1 mod f,  q = ell^r = |F|.
//             The huge exponent is never formed: with
//             (q-1)/2 = (1 + ell + ... + ell^{r-1}) * (ell-1)/2 the first
//             factor is accumulated by u <- u^ell * (X + s), r - 1 times, and
//             only the small exponent (ell-1)/2 is a square-and-multiply.
//             u^ell is the precomputed Frobenius of A (lever (1) above).  All
//             steps are literal powers in F[X]/(f), so the identity
//             h^{(q-1)/2} = (h^{1+ell+...})^{(ell-1)/2} is exact.
//
//   ell = 2:  the trace map Tr_{F/F_2}(s*X) = sum_{i<r} (s X)^{2^i} mod f,
//             the standard char-2 substitute -- (q-1)/2 is not an integer
//             there, and squaring is a bijection, so the quadratic-residue
//             test of the odd case has no analogue.  Squaring is the
//             Frobenius of A at ell = 2.
//
// In both cases gcd(f, split_poly) collects exactly the roots on one side of a
// two-valued invariant, so it is a proper factor unless every root of f falls
// on the same side; the caller then moves to the next candidate.  (Before
// slot-7 WP-4 this was computed through FqPowMod / FqMul / FqMod; the value is
// the same element of F[X]/(f).)
FqPoly split_poly(const CanonicalField& F, QuotientRing& A, const FieldElt& s) {
    const uint32_t ell = F.ell();
    const uint32_t r = F.degree();
    const uint32_t one = 1u % ell;
    QuotientRing::Elt u = A.zero(), v = A.zero();
    if (ell == 2u) {
        if (F.is_zero(s)) return FqPoly();
        for (uint32_t j = 0; j < r; ++j) u[r + j] = s[j];   // s * X  (deg f >= 2)
        QuotientRing::Elt acc = u;
        for (uint32_t i = 1; i < r; ++i) {
            A.frob(u, v);
            u.swap(v);
            for (size_t k = 0; k < acc.size(); ++k) acc[k] ^= u[k];   // + in F_2
        }
        return A.to_fq(acc);
    }
    for (uint32_t j = 0; j < r; ++j) u[j] = s[j];   // X + s  (deg f >= 2)
    u[r] = one;
    for (uint32_t i = 1; i < r; ++i) {
        A.frob(u, v);
        A.mul_linear(v, s, u);
    }
    A.pow(u, static_cast<uint64_t>((ell - 1u) / 2u), v);
    v[0] = F.mod_ctx().sub(v[0], one);
    return A.to_fq(v);
}

// The cap on the fixed candidate sequence, per splitting step.  The sequence
// is deterministic, so this is not a retry budget in the R3 sense: it is the
// point at which the unit REPORTS a failure instead of scanning all of F.
// Every root found on any input this repository produces is reached within a
// handful of candidates (recorded as `candidates_used`).
constexpr uint64_t kMaxCandidatesPerStep = 4096;

// WHERE THE FIXED SEQUENCE STARTS, and why it is not at s = 0.
//
// [E] Alg. 6.4 writes the sequence as "s = 0, 1, Y, ...", and the elements of
// F in CanonicalLess order are exactly 0, 1, ..., ell - 1, Y, ... -- so the
// first `ell` of them are the CONSTANTS, the copy of F_ell inside F.  Not one
// of them can ever split pi_w, by the following one-line theorem:
//
//   the r roots of pi_w are a single Frobenius orbit, r_i = rho^{ell^i};
//   for s in F_ell the map x |-> x + s commutes with Frobenius, so the
//   r_i + s are again a single Frobenius orbit; and both two-valued
//   invariants the splitting polynomial uses -- the quadratic character
//   chi(x) in {+1, -1} for odd ell, and Tr_{F/F_2}(x) in F_2 for ell = 2 --
//   are Frobenius-INVARIANT (chi(x^ell) = chi(x)^ell = chi(x) because
//   chi(x) = +-1 and ell is odd; Tr(x^2) = Tr(x)).  So the invariant takes
//   the same value at every root and the gcd is 1 or f: never a proper
//   factor.
//
// Skipping them is therefore not a heuristic and not a change of answer: it
// removes candidates that are PROVED useless, leaving the order of what
// remains untouched.  It is worth ell rejected candidates per splitting step,
// which at ell = 691 was measured as 0.47 s on one (160, 691) cell and 5.4 s
// for a single r = 10 block -- against 3 candidates and microseconds after.
//
// Below the top of the recursion f is a proper factor of pi_w whose roots need
// not be a full orbit, so a constant COULD split there; the sequence skips
// them anyway, because it must be one fixed sequence and the non-constant
// candidates split just as well.
uint64_t first_non_constant_candidate(const CanonicalField& F) {
    return static_cast<uint64_t>(F.ell());   // 0, then the ell - 1 non-zero constants
}

} // namespace

CanonicalRoot FindCanonicalRoot(const CanonicalField& F, const hecke::FpPoly& pi_w) {
    CanonicalRoot out;
    const uint32_t r = F.degree();

    if (pi_w.is_zero() || pi_w.ell != F.ell()) {
        out.failure = "pi_w is zero or over a different modulus";
        return out;
    }
    if (!pi_w.is_monic()) {
        out.failure = "pi_w is not monic";
        return out;
    }
    if (pi_w.degree() != static_cast<int64_t>(r)) {
        out.failure = "deg pi_w != [F : F_ell]";
        return out;
    }
    if (hecke::CertifyIrreducible(pi_w) != 0u) {
        out.failure = "pi_w is not a certified irreducible over F_ell";
        return out;
    }

    // The search.  `f` is always a monic factor of pi_w that splits completely
    // in F; it shrinks strictly until it is linear.
    // The candidate sequence is the elements of F in CanonicalLess order, so
    // there is no point scanning past |F| = ell^r (it would repeat, because
    // element_at drops digits beyond r - 1).
    uint64_t field_size = 1;
    for (uint32_t i = 0; i < r; ++i) {
        if (field_size > kMaxCandidatesPerStep + first_non_constant_candidate(F)) break;
        field_size *= F.ell();
    }
    const uint64_t begin = first_non_constant_candidate(F);
    const uint64_t cap = field_size < begin + kMaxCandidatesPerStep
                             ? field_size
                             : begin + kMaxCandidatesPerStep;

    // (slot-7 WP-4) The Frobenius of F as a matrix, once per call; the
    // quotient ring of each f, once per f (shared by all of f's candidates).
    const std::vector<uint32_t> frob_rows = FrobeniusRows(F);

    FqPoly f = FqMonic(F, FqFromFp(F, pi_w));
    while (f.degree() > 1) {
        bool split = false;
        QuotientRing A(F, f, frob_rows);
        for (uint64_t k = begin; k < cap; ++k) {
            ++out.candidates_used;
            if (CanonicalFieldOpCounts* oc = t_op_counts) ++oc->candidates;
            const FqPoly g = FqGcd(F, f, split_poly(F, A, F.element_at(k)));
            if (g.degree() >= 1 && g.degree() < f.degree()) {
                f = FqMonic(F, g);
                split = true;
                break;
            }
        }
        if (!split) {
            out.failure = "the fixed candidate sequence did not split pi_w within the cap";
            return out;
        }
    }
    // f = X + c with c in F, so the root is -c.
    out.rho = F.neg(f.coeffs[0]);

    // The r conjugates rho^{ell^i}, and the CanonicalLess-least of them.
    std::vector<FieldElt> conj;
    conj.reserve(r);
    FieldElt cur = out.rho;
    for (uint32_t i = 0; i < r; ++i) {
        conj.push_back(cur);
        cur = ApplyFrobenius(F, frob_rows, cur);
    }
    if (cur != out.rho) {
        out.failure = "the Frobenius orbit of the root does not close after r steps";
        return out;
    }
    size_t least = 0;
    for (size_t i = 1; i < conj.size(); ++i) {
        if (CanonicalField::less(conj[i], conj[least])) least = i;
    }
    out.rho_canonical = conj[least];
    out.conjugates.clear();
    out.conjugates.reserve(r);
    cur = out.rho_canonical;
    for (uint32_t i = 0; i < r; ++i) {
        out.conjugates.push_back(cur);
        cur = ApplyFrobenius(F, frob_rows, cur);
    }

    // Certify-after-find: pi_w(rho_canonical) = 0 in F, and the r conjugates
    // are pairwise distinct roots.
    if (!F.is_zero(F.eval(pi_w, out.rho_canonical))) {
        out.failure = "pi_w(rho) != 0 in F";
        return out;
    }
    for (uint32_t i = 0; i < r; ++i) {
        if (!F.is_zero(F.eval(pi_w, out.conjugates[i]))) {
            out.failure = "a conjugate is not a root of pi_w";
            return out;
        }
        for (uint32_t j = i + 1; j < r; ++j) {
            if (out.conjugates[i] == out.conjugates[j]) {
                out.failure = "two conjugates coincide";
                return out;
            }
        }
    }
    out.ok = true;
    return out;
}

// ===========================================================================
// The split form of a block's residue data
// ===========================================================================

CanonicalSplit SplitInCanonicalField(const CanonicalField& F, const hecke::FpPoly& pi_w,
                                     const std::vector<uint64_t>& primes,
                                     const std::vector<hecke::FpPoly>& character) {
    ENGINE_CHECK_MSG(primes.size() == character.size(),
                     "hecke_ext::SplitInCanonicalField: primes and character disagree in length");

    CanonicalSplit s;
    s.ell = F.ell();
    s.r = F.degree();
    s.field_polynomial = F.defining_poly();
    s.witness_poly = pi_w;
    s.primes = primes;

    const CanonicalRoot root = FindCanonicalRoot(F, pi_w);
    s.candidates_used = root.candidates_used;
    if (!root.ok) {
        s.failure = root.failure;
        return s;
    }
    s.rho_canonical = root.rho_canonical;
    s.conjugates = root.conjugates;

    for (size_t j = 0; j < character.size(); ++j) {
        const hecke::FpPoly& P = character[j];
        ENGINE_CHECK_MSG(P.is_zero() || P.ell == F.ell(),
                         "hecke_ext::SplitInCanonicalField: a P_q is over a different modulus");
        if (P.degree() >= static_cast<int64_t>(s.r)) {
            s.failure = "a character polynomial has degree >= r (it is not reduced mod pi_w)";
            return s;
        }
    }
    s.eigenvalues.reserve(character.size());
    for (const hecke::FpPoly& P : character) s.eigenvalues.push_back(F.eval(P, s.rho_canonical));
    s.orbit.reserve(s.r);
    for (uint32_t i = 0; i < s.r; ++i) {
        std::vector<FieldElt> row;
        row.reserve(character.size());
        for (const hecke::FpPoly& P : character) row.push_back(F.eval(P, s.conjugates[i]));
        s.orbit.push_back(std::move(row));
    }
    s.ok = true;
    return s;
}

CanonicalSplit SplitInCanonicalField(uint32_t ell, const hecke::FpPoly& pi_w,
                                     const std::vector<uint64_t>& primes,
                                     const std::vector<hecke::FpPoly>& character) {
    ENGINE_CHECK_MSG(ell >= 2, "hecke_ext::SplitInCanonicalField: ell < 2");
    if (pi_w.is_zero() || pi_w.degree() < 1) {
        CanonicalSplit s;
        s.ell = ell;
        s.witness_poly = pi_w;
        s.primes = primes;
        s.failure = "pi_w is zero or constant";
        return s;
    }
    const CanonicalField F(ell, static_cast<uint32_t>(pi_w.degree()));
    return SplitInCanonicalField(F, pi_w, primes, character);
}

std::string CanonicalSplit::to_json() const {
    std::ostringstream oss;
    oss << "{\"schema\":\"hecke_canonical_split/v1\""
        << ",\"ok\":" << (ok ? "true" : "false")
        << ",\"ell\":" << ell << ",\"r\":" << r
        << ",\"field_polynomial\":" << fp_poly_json(field_polynomial)
        << ",\"witness_poly\":" << fp_poly_json(witness_poly)
        << ",\"candidates_used\":" << candidates_used;
    if (!ok) {
        oss << ",\"failure\":\"" << failure << "\"}";
        return oss.str();
    }
    const CanonicalField F(ell, field_polynomial);
    oss << ",\"rho_canonical\":" << F.to_text(rho_canonical) << ",\"conjugates\":[";
    for (size_t i = 0; i < conjugates.size(); ++i) {
        if (i) oss << ",";
        oss << F.to_text(conjugates[i]);
    }
    oss << "],\"primes\":[";
    for (size_t i = 0; i < primes.size(); ++i) {
        if (i) oss << ",";
        oss << primes[i];
    }
    oss << "],\"eigenvalues\":[";
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        if (i) oss << ",";
        oss << F.to_text(eigenvalues[i]);
    }
    oss << "],\"frobenius_orbit\":[";
    for (size_t i = 0; i < orbit.size(); ++i) {
        if (i) oss << ",";
        oss << "[";
        for (size_t j = 0; j < orbit[i].size(); ++j) {
            if (j) oss << ",";
            oss << F.to_text(orbit[i][j]);
        }
        oss << "]";
    }
    oss << "]}";
    return oss.str();
}

uint32_t CertifyCanonicalSplit(const CanonicalSplit& s,
                               const std::vector<hecke::FpPoly>& character) {
    if (!s.ok) return 1u;
    if (s.ell < 2u || s.r < 1u) return 1u;
    if (s.witness_poly.is_zero() || s.witness_poly.ell != s.ell) return 1u;
    if (s.witness_poly.degree() != static_cast<int64_t>(s.r)) return 1u;

    if (!s.witness_poly.is_monic()) return 2u;
    if (hecke::CertifyIrreducible(s.witness_poly) != 0u) return 2u;

    // Clause 3: C_{ell,r} is RE-ENUMERATED, never read back from the record --
    // through the same `field_polynomial` setting the constructor consults
    // (slot-7 WP-4), so under `conway` the certificate raises
    // hecke::NotImplementedInThisBuild rather than certifying against `least`.
    if (s.field_polynomial != FieldPolynomialFromSettings(s.ell, s.r)) return 3u;

    const CanonicalField F(s.ell, s.field_polynomial);

    if (s.rho_canonical.size() != s.r) return 4u;
    if (!F.is_zero(F.eval(s.witness_poly, s.rho_canonical))) return 4u;

    if (s.conjugates.size() != s.r) return 5u;
    if (s.conjugates[0] != s.rho_canonical) return 5u;
    for (uint32_t i = 0; i + 1 < s.r; ++i) {
        if (F.frobenius(s.conjugates[i]) != s.conjugates[i + 1]) return 5u;
    }
    if (F.frobenius(s.conjugates[s.r - 1u]) != s.conjugates[0]) return 5u;

    for (uint32_t i = 0; i < s.r; ++i) {
        if (!F.is_zero(F.eval(s.witness_poly, s.conjugates[i]))) return 6u;
        for (uint32_t j = i + 1; j < s.r; ++j) {
            if (s.conjugates[i] == s.conjugates[j]) return 6u;
        }
    }

    for (uint32_t i = 0; i < s.r; ++i) {
        if (CanonicalField::less(s.conjugates[i], s.rho_canonical)) return 7u;
    }

    if (character.size() != s.primes.size()) return 8u;
    if (s.eigenvalues.size() != character.size()) return 8u;
    for (size_t j = 0; j < character.size(); ++j) {
        if (s.eigenvalues[j] != F.eval(character[j], s.rho_canonical)) return 8u;
    }

    if (s.orbit.size() != s.r) return 9u;
    for (uint32_t i = 0; i < s.r; ++i) {
        if (s.orbit[i].size() != character.size()) return 9u;
        for (size_t j = 0; j < character.size(); ++j) {
            if (s.orbit[i][j] != F.eval(character[j], s.conjugates[i])) return 9u;
        }
    }
    if (!s.orbit.empty() && s.orbit[0] != s.eigenvalues) return 9u;
    return 0u;
}

// ===========================================================================
// The own-field report form -- tier (i), zero new arithmetic
// ===========================================================================

std::string OwnFieldReportJson(uint32_t ell, uint32_t w_hat_exponent,
                               const hecke::FpPoly& pi_w,
                               const std::vector<uint64_t>& primes,
                               const std::vector<hecke::FpPoly>& character) {
    ENGINE_CHECK_MSG(primes.size() == character.size(),
                     "hecke_ext::OwnFieldReportJson: primes and character disagree in length");
    std::ostringstream oss;
    oss << "{\"schema\":\"hecke_own_field/v1\",\"ell\":" << ell
        << ",\"w_hat_exponent\":" << w_hat_exponent
        << ",\"witness_poly\":" << fp_poly_json(pi_w) << ",\"P\":{";
    for (size_t i = 0; i < primes.size(); ++i) {
        if (i) oss << ",";
        oss << "\"" << primes[i] << "\":" << fp_poly_json(character[i]);
    }
    oss << "}}";
    return oss.str();
}

} // namespace hecke_ext
