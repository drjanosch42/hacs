// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S10 -- dense univariate polynomial arithmetic over F_ell.
// See FpPoly.h for the binding conventions (A-N4 coefficient order,
// division-free Berkowitz charpoly, deterministic certification invariant).
//
// Stage S13: templated on the element type Elt via the ModArithmetic mod_*_w
// width dispatchers. FpPoly = FpPolyT<uint32_t> is byte-identical to the
// pre-S13 code; FpPoly64 = FpPolyT<uint64_t> supports ell < 2^64.

#include "FpPoly.h"

#include "../../ModArithmetic/ModArithmetic.h"
#include "../../ModArithmetic/Matrix.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace hecke {

namespace {

using ModArithmetic::mod_add_w;
using ModArithmetic::mod_inv_w;
using ModArithmetic::mod_minus_w;
using ModArithmetic::mod_mul_w;
using ModArithmetic::mod_sub_w;

template<class Elt>
void require_same_modulus(const FpPolyT<Elt>& a, const FpPolyT<Elt>& b, const char* op) {
    if (a.ell == 0 || b.ell == 0 || a.ell != b.ell) {
        throw std::invalid_argument(std::string("FpPoly::") + op +
                                    ": mismatched or sentinel moduli");
    }
}

// splitmix64 -- the deterministic-seed PRNG used by the Cantor-Zassenhaus
// SEARCH (never by any certificate).
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    template<class Elt>
    Elt next_mod(Elt m) { return static_cast<Elt>(next() % m); }
};

} // namespace

// ---------------------------------------------------------------------------
// FpPoly basics
// ---------------------------------------------------------------------------

template<class Elt> FpPolyT<Elt>::FpPolyT() : ell(0), coeffs() {}

template<class Elt> FpPolyT<Elt>::FpPolyT(Elt ell_) : ell(ell_), coeffs() {}

template<class Elt>
FpPolyT<Elt>::FpPolyT(Elt ell_, std::vector<Elt> c) : ell(ell_), coeffs(std::move(c)) {
    if (ell < 2) throw std::invalid_argument("FpPoly: modulus must be a prime >= 2");
    for (Elt& x : coeffs) x %= ell;
    trim();
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::zero(Elt ell_) { return FpPolyT(ell_, {}); }
template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::one(Elt ell_) { return FpPolyT(ell_, {1}); }
template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::X(Elt ell_) { return FpPolyT(ell_, {0, 1}); }
template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::constant(Elt ell_, Elt c) { return FpPolyT(ell_, {c}); }

template<class Elt> void FpPolyT<Elt>::trim() {
    while (!coeffs.empty() && coeffs.back() == 0) coeffs.pop_back();
}

template<class Elt> bool FpPolyT<Elt>::is_zero() const { return coeffs.empty(); }

template<class Elt> int64_t FpPolyT<Elt>::degree() const {
    return static_cast<int64_t>(coeffs.size()) - 1;
}

template<class Elt> Elt FpPolyT<Elt>::leading_coeff() const { return coeffs.empty() ? 0 : coeffs.back(); }

template<class Elt> Elt FpPolyT<Elt>::coeff(uint32_t i) const {
    return (i < coeffs.size()) ? coeffs[i] : 0;
}

template<class Elt> bool FpPolyT<Elt>::is_monic() const { return leading_coeff() == 1; }

template<class Elt> bool FpPolyT<Elt>::operator==(const FpPolyT& other) const {
    return ell == other.ell && coeffs == other.coeffs;
}
template<class Elt> bool FpPolyT<Elt>::operator!=(const FpPolyT& other) const { return !(*this == other); }

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::operator+(const FpPolyT& other) const {
    require_same_modulus(*this, other, "add");
    FpPolyT r(ell);
    r.coeffs.resize(std::max(coeffs.size(), other.coeffs.size()), 0);
    for (size_t i = 0; i < r.coeffs.size(); ++i) {
        r.coeffs[i] = mod_add_w(coeff(static_cast<uint32_t>(i)),
                                other.coeff(static_cast<uint32_t>(i)), ell);
    }
    r.trim();
    return r;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::operator-(const FpPolyT& other) const {
    require_same_modulus(*this, other, "sub");
    FpPolyT r(ell);
    r.coeffs.resize(std::max(coeffs.size(), other.coeffs.size()), 0);
    for (size_t i = 0; i < r.coeffs.size(); ++i) {
        r.coeffs[i] = mod_sub_w(coeff(static_cast<uint32_t>(i)),
                                other.coeff(static_cast<uint32_t>(i)), ell);
    }
    r.trim();
    return r;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::operator*(const FpPolyT& other) const {
    require_same_modulus(*this, other, "mul");
    if (is_zero() || other.is_zero()) return FpPolyT::zero(ell);
    FpPolyT r(ell);
    r.coeffs.assign(coeffs.size() + other.coeffs.size() - 1, 0);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        if (coeffs[i] == 0) continue;
        for (size_t j = 0; j < other.coeffs.size(); ++j) {
            r.coeffs[i + j] = mod_add_w(r.coeffs[i + j],
                                        mod_mul_w(coeffs[i], other.coeffs[j], ell), ell);
        }
    }
    r.trim();
    return r;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::scalar_mul(Elt c) const {
    if (ell == 0) throw std::invalid_argument("FpPoly::scalar_mul on sentinel modulus");
    FpPolyT r(ell);
    c %= ell;
    if (c == 0) return r;
    r.coeffs.resize(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) r.coeffs[i] = mod_mul_w(coeffs[i], c, ell);
    r.trim();
    return r;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::negate() const {
    if (ell == 0) throw std::invalid_argument("FpPoly::negate on sentinel modulus");
    FpPolyT r(ell);
    r.coeffs.resize(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) r.coeffs[i] = mod_minus_w(coeffs[i], ell);
    return r;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::monic() const {
    if (is_zero()) throw std::invalid_argument("FpPoly::monic of the zero polynomial");
    if (is_monic()) return *this;
    return scalar_mul(mod_inv_w(leading_coeff(), ell));
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::derivative() const {
    if (ell == 0) throw std::invalid_argument("FpPoly::derivative on sentinel modulus");
    FpPolyT r(ell);
    if (coeffs.size() <= 1) return r;
    r.coeffs.resize(coeffs.size() - 1);
    for (size_t i = 1; i < coeffs.size(); ++i) {
        // i mod ell as a field element (i can exceed ell in the inseparable regime).
        r.coeffs[i - 1] = mod_mul_w(coeffs[i], static_cast<Elt>(i % ell), ell);
    }
    r.trim();
    return r;
}

template<class Elt> Elt FpPolyT<Elt>::eval(Elt x) const {
    if (ell == 0) throw std::invalid_argument("FpPoly::eval on sentinel modulus");
    x %= ell;
    Elt acc = 0;
    for (size_t i = coeffs.size(); i-- > 0;) {
        acc = mod_add_w(mod_mul_w(acc, x, ell), coeffs[i], ell);
    }
    return acc;
}

template<class Elt>
void FpPolyT<Elt>::divrem(const FpPolyT& a, const FpPolyT& b, FpPolyT& q, FpPolyT& r) {
    require_same_modulus(a, b, "divrem");
    if (b.is_zero()) throw std::invalid_argument("FpPoly::divrem: division by zero");
    const Elt ell = a.ell;
    q = FpPolyT::zero(ell);
    r = a;
    const int64_t db = b.degree();
    if (r.degree() < db) return;
    const Elt lc_inv = mod_inv_w(b.leading_coeff(), ell);
    q.coeffs.assign(static_cast<size_t>(r.degree() - db) + 1, 0);
    while (!r.is_zero() && r.degree() >= db) {
        const int64_t k = r.degree() - db;
        const Elt t = mod_mul_w(r.leading_coeff(), lc_inv, ell);
        q.coeffs[static_cast<size_t>(k)] = t;
        // r -= t * X^k * b
        for (int64_t i = 0; i <= db; ++i) {
            const size_t idx = static_cast<size_t>(k + i);
            r.coeffs[idx] = mod_sub_w(r.coeffs[idx],
                                      mod_mul_w(t, b.coeffs[static_cast<size_t>(i)], ell), ell);
        }
        r.trim();
    }
    q.trim();
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::mod(const FpPolyT& a, const FpPolyT& b) {
    FpPolyT q, r;
    divrem(a, b, q, r);
    return r;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::divide_exact(const FpPolyT& a, const FpPolyT& b) {
    FpPolyT q, r;
    divrem(a, b, q, r);
    if (!r.is_zero()) throw std::runtime_error("FpPoly::divide_exact: nonzero remainder");
    return q;
}

template<class Elt> FpPolyT<Elt> FpPolyT<Elt>::gcd(FpPolyT a, FpPolyT b) {
    require_same_modulus(a, b, "gcd");
    while (!b.is_zero()) {
        FpPolyT r = mod(a, b);
        a = b;
        b = r;
    }
    if (a.is_zero()) return a;
    return a.monic();
}

template<class Elt>
FpPolyT<Elt> FpPolyT<Elt>::extgcd(const FpPolyT& a, const FpPolyT& b, FpPolyT& u, FpPolyT& v) {
    require_same_modulus(a, b, "extgcd");
    const Elt ell = a.ell;
    FpPolyT r0 = a, r1 = b;
    FpPolyT s0 = FpPolyT::one(ell), s1 = FpPolyT::zero(ell);
    FpPolyT t0 = FpPolyT::zero(ell), t1 = FpPolyT::one(ell);
    while (!r1.is_zero()) {
        FpPolyT q, r;
        divrem(r0, r1, q, r);
        FpPolyT s2 = s0 - q * s1;
        FpPolyT t2 = t0 - q * t1;
        r0 = r1; r1 = r;
        s0 = s1; s1 = s2;
        t0 = t1; t1 = t2;
    }
    if (r0.is_zero()) { u = s0; v = t0; return r0; }
    const Elt c = mod_inv_w(r0.leading_coeff(), ell);
    u = s0.scalar_mul(c);
    v = t0.scalar_mul(c);
    return r0.scalar_mul(c);
}

template<class Elt>
FpPolyT<Elt> FpPolyT<Elt>::powmod(const FpPolyT& g, uint64_t e, const FpPolyT& f) {
    require_same_modulus(g, f, "powmod");
    if (f.degree() < 1) throw std::invalid_argument("FpPoly::powmod: modulus of degree < 1");
    FpPolyT base = mod(g, f);
    FpPolyT res = mod(FpPolyT::one(g.ell), f); // = 1 since deg f >= 1
    while (e > 0) {
        if (e & 1ull) res = mod(res * base, f);
        e >>= 1;
        if (e > 0) base = mod(base * base, f);
    }
    return res;
}

template<class Elt>
FpPolyT<Elt> FpPolyT<Elt>::frobenius_powmod(uint32_t k, const FpPolyT& f) {
    if (f.degree() < 1) {
        throw std::invalid_argument("FpPoly::frobenius_powmod: modulus of degree < 1");
    }
    FpPolyT h = mod(FpPolyT::X(f.ell), f);
    for (uint32_t i = 0; i < k; ++i) {
        h = powmod(h, static_cast<uint64_t>(f.ell), f); // X^{ell^{i+1}} = (X^{ell^i})^ell
    }
    return h;
}

template<class Elt>
FpPolyT<Elt> FpPolyT<Elt>::pow(const FpPolyT& g, uint32_t e) {
    if (g.ell == 0) throw std::invalid_argument("FpPoly::pow on sentinel modulus");
    FpPolyT res = FpPolyT::one(g.ell);
    FpPolyT base = g;
    while (e > 0) {
        if (e & 1u) res = res * base;
        e >>= 1;
        if (e > 0) base = base * base;
    }
    return res;
}

template<class Elt>
std::string FpPolyT<Elt>::to_string() const {
    std::ostringstream oss;
    if (is_zero()) {
        oss << "0";
    } else {
        bool first = true;
        for (size_t i = coeffs.size(); i-- > 0;) {
            const Elt c = coeffs[i];
            if (c == 0) continue;
            if (!first) oss << " + ";
            first = false;
            if (i == 0) {
                oss << c;
            } else {
                if (c != 1) oss << c << "*";
                oss << "X";
                if (i > 1) oss << "^" << i;
            }
        }
    }
    oss << " (mod " << ell << ")";
    return oss.str();
}

template<class Elt>
bool CanonicalLess(const FpPolyT<Elt>& a, const FpPolyT<Elt>& b) {
    if (a.degree() != b.degree()) return a.degree() < b.degree();
    return a.coeffs < b.coeffs; // lexicographic on the low->high coefficient vector
}

// ---------------------------------------------------------------------------
// Matrix polynomials
// ---------------------------------------------------------------------------

template<class Elt>
FpPolyT<Elt> CharPolyBerkowitz(const MatrixT<Elt>& A) {
    if (A.rows != A.cols) throw std::invalid_argument("CharPolyBerkowitz: non-square matrix");
    const Elt ell = A.p;
    const uint32_t d = A.rows;
    if (d == 0) return FpPolyT<Elt>::one(ell);

    // Berkowitz's division-free algorithm. C holds the coefficients of the
    // characteristic polynomial of the leading principal r x r submatrix,
    // HIGH -> LOW degree (C[0] = 1 = coefficient of X^r).
    std::vector<Elt> C = {1, mod_minus_w(A.data[0][0], ell)};
    for (uint32_t r = 2; r <= d; ++r) {
        const uint32_t n = r - 1; // size of the previous leading submatrix M
        std::vector<Elt> t(r + 1, 0);
        t[0] = 1;
        t[1] = mod_minus_w(A.data[n][n], ell);
        std::vector<Elt> v(n);
        for (uint32_t i = 0; i < n; ++i) v[i] = A.data[i][n]; // v = S
        for (uint32_t j = 2; j <= r; ++j) {
            Elt dot = 0;
            for (uint32_t i = 0; i < n; ++i) {
                dot = mod_add_w(dot, mod_mul_w(A.data[n][i], v[i], ell), ell);
            }
            t[j] = mod_minus_w(dot, ell);
            if (j < r) { // v <- M v
                std::vector<Elt> nv(n, 0);
                for (uint32_t i = 0; i < n; ++i) {
                    Elt acc = 0;
                    for (uint32_t k = 0; k < n; ++k) {
                        acc = mod_add_w(acc, mod_mul_w(A.data[i][k], v[k], ell), ell);
                    }
                    nv[i] = acc;
                }
                v.swap(nv);
            }
        }
        // newC = T * C, with T the (r+1) x r lower-triangular Toeplitz matrix.
        std::vector<Elt> newC(r + 1, 0);
        for (uint32_t q = 0; q < r; ++q) {
            if (C[q] == 0) continue;
            for (uint32_t i = q; i <= r && i - q <= r; ++i) {
                newC[i] = mod_add_w(newC[i], mod_mul_w(t[i - q], C[q], ell), ell);
            }
        }
        C.swap(newC);
    }

    // Convert HIGH -> LOW into the binding LOW -> HIGH order (A-N4).
    std::vector<Elt> low_to_high(d + 1);
    for (uint32_t i = 0; i <= d; ++i) low_to_high[i] = C[d - i];
    return FpPolyT<Elt>(ell, std::move(low_to_high));
}

template<class Elt>
MatrixT<Elt> EvalPolyAtMatrix(const FpPolyT<Elt>& P, const MatrixT<Elt>& A) {
    if (A.rows != A.cols) throw std::invalid_argument("EvalPolyAtMatrix: non-square matrix");
    if (P.ell != A.p) throw std::invalid_argument("EvalPolyAtMatrix: modulus mismatch");
    const uint32_t d = A.rows;
    const Elt ell = A.p;
    MatrixT<Elt> R(d, d, ell); // zero
    for (size_t i = P.coeffs.size(); i-- > 0;) {
        R = R * A;
        const Elt c = P.coeffs[i];
        if (c != 0) {
            for (uint32_t k = 0; k < d; ++k) R.data[k][k] = mod_add_w(R.data[k][k], c, ell);
        }
    }
    return R;
}

template<class Elt>
FpPolyT<Elt> MinPolyKrylov(const MatrixT<Elt>& A_in) {
    if (A_in.rows != A_in.cols) throw std::invalid_argument("MinPolyKrylov: non-square matrix");
    const Elt ell = A_in.p;
    const uint32_t d = A_in.rows;
    if (d == 0) return FpPolyT<Elt>::one(ell);
    const MatrixT<Elt>& A = A_in;

    FpPolyT<Elt> m = FpPolyT<Elt>::one(ell);
    for (uint32_t start = 0; start < d; ++start) {
        if (m.degree() == static_cast<int64_t>(d)) break; // cannot grow further
        // Krylov matrix with columns v, Av, ..., A^d v for v = e_start.
        MatrixT<Elt> Kr(d, d + 1, ell);
        MatrixT<Elt> w(d, 1, ell);
        w.data[start][0] = 1;
        for (uint32_t j = 0; j <= d; ++j) {
            for (uint32_t i = 0; i < d; ++i) Kr.data[i][j] = w.data[i][0];
            if (j < d) w = A * w;
        }
        // Defect #31 (matrix-engine Phase 1): rank() and kernel() are both
        // driven by the SAME cached RREF of Kr, so read the pivot columns once
        // and take the rank from them -- one elimination, and one cache
        // staleness scan instead of two.  kernel() below then hits that cache.
        // (kernel() is deliberately kept rather than open-coding the kernel
        // vector from the RREF: the normalisation check right after it is the
        // canary for Matrix::kernel()'s basis convention -- plan Q1.)
        const uint32_t j = static_cast<uint32_t>(Kr.get_pivot_columns().size());
        if (j > d) throw std::runtime_error("MinPolyKrylov: impossible Krylov rank");
        std::vector<MatrixT<Elt>> ker = Kr.kernel();
        if (ker.empty()) throw std::runtime_error("MinPolyKrylov: Krylov matrix has full column rank");
        const MatrixT<Elt>& k0 = ker[0];
        if (k0.data[j][0] != 1) throw std::runtime_error("MinPolyKrylov: unexpected kernel normalization");
        std::vector<Elt> local_coeffs(j + 1);
        for (uint32_t i = 0; i <= j; ++i) local_coeffs[i] = k0.data[i][0];
        FpPolyT<Elt> local(ell, std::move(local_coeffs)); // monic, degree j
        // m <- lcm(m, local)
        FpPolyT<Elt> g = FpPolyT<Elt>::gcd(m, local);
        m = FpPolyT<Elt>::divide_exact(m * local, g);
    }
    m = m.monic();
    // Unconditional self-check: m(A) = 0.
    if (!EvalPolyAtMatrix(m, A).is_zero()) {
        throw std::runtime_error("MinPolyKrylov: computed polynomial does not annihilate A");
    }
    return m;
}

// ---------------------------------------------------------------------------
// Factorization
// ---------------------------------------------------------------------------

namespace {

// ell-th root of f in F_ell[X], valid when f' = 0 (so f in F_ell[X^ell]).
template<class Elt>
FpPolyT<Elt> pth_root(const FpPolyT<Elt>& f) {
    const Elt ell = f.ell;
    if (f.is_zero()) return f;
    FpPolyT<Elt> h(ell);
    h.coeffs.assign(static_cast<size_t>(f.degree() / ell) + 1, 0);
    for (size_t i = 0; i < f.coeffs.size(); ++i) {
        if (i % ell == 0) {
            h.coeffs[i / ell] = f.coeffs[i];
        } else if (f.coeffs[i] != 0) {
            throw std::runtime_error("pth_root: polynomial is not an ell-th power");
        }
    }
    h.trim();
    return h;
}

// Char-p squarefree decomposition (deterministic). f monic, deg >= 1.
template<class Elt>
void sfd_rec(const FpPolyT<Elt>& f, uint64_t scale,
             std::vector<std::pair<FpPolyT<Elt>, uint32_t>>& out) {
    if (f.degree() < 1) return;
    if (scale > 0xFFFFFFFFull) throw std::runtime_error("SquarefreeDecomposition: multiplicity overflow");
    const FpPolyT<Elt> fp = f.derivative();
    if (fp.is_zero()) {
        // Inseparable corner: f = h(X^ell) = pth_root(f)^ell.
        sfd_rec(pth_root(f), scale * static_cast<uint64_t>(f.ell), out);
        return;
    }
    FpPolyT<Elt> c = FpPolyT<Elt>::gcd(f, fp);
    FpPolyT<Elt> w = FpPolyT<Elt>::divide_exact(f, c); // product of pi with ell-coprime multiplicity
    uint32_t i = 1;
    while (w.degree() > 0) {
        FpPolyT<Elt> y = FpPolyT<Elt>::gcd(w, c);
        FpPolyT<Elt> z = FpPolyT<Elt>::divide_exact(w, y); // multiplicity-exactly-i part
        if (z.degree() > 0) {
            out.emplace_back(z, static_cast<uint32_t>(static_cast<uint64_t>(i) * scale));
        }
        c = FpPolyT<Elt>::divide_exact(c, y);
        w = y;
        ++i;
        if (i > f.degree() + 1) throw std::runtime_error("SquarefreeDecomposition: non-termination");
    }
    if (c.degree() > 0) {
        // Residual part: all multiplicities divisible by ell -> an ell-th power.
        sfd_rec(pth_root(c), scale * static_cast<uint64_t>(f.ell), out);
    }
}

// Distinct-degree factorization of a monic squarefree g.
template<class Elt>
std::vector<std::pair<FpPolyT<Elt>, uint32_t>> distinct_degree(FpPolyT<Elt> g) {
    std::vector<std::pair<FpPolyT<Elt>, uint32_t>> out;
    const Elt ell = g.ell;
    if (g.degree() < 1) return out;
    FpPolyT<Elt> h = FpPolyT<Elt>::mod(FpPolyT<Elt>::X(ell), g); // X^{ell^0} mod g
    uint32_t e = 0;
    while (g.degree() >= 2 * static_cast<int64_t>(e + 1)) {
        ++e;
        h = FpPolyT<Elt>::powmod(h, static_cast<uint64_t>(ell), g); // X^{ell^e} mod g
        FpPolyT<Elt> d = FpPolyT<Elt>::gcd(h - FpPolyT<Elt>::mod(FpPolyT<Elt>::X(ell), g), g);
        if (d.degree() > 0) {
            out.emplace_back(d, e);
            g = FpPolyT<Elt>::divide_exact(g, d);
            if (g.degree() < 1) return out;
            h = FpPolyT<Elt>::mod(h, g);
        }
    }
    if (g.degree() > 0) out.emplace_back(g, static_cast<uint32_t>(g.degree()));
    return out;
}

template<class Elt>
FpPolyT<Elt> random_nonconstant_below(Elt ell, int64_t deg_bound, SplitMix64& rng) {
    for (;;) {
        std::vector<Elt> c(static_cast<size_t>(deg_bound), 0);
        for (auto& x : c) x = rng.next_mod(ell);
        FpPolyT<Elt> a(ell, std::move(c));
        if (a.degree() >= 1) return a;
    }
}

// Equal-degree factorization (Cantor-Zassenhaus) of monic squarefree h.
template<class Elt>
void equal_degree(const FpPolyT<Elt>& h, uint32_t e, SplitMix64& rng,
                  std::vector<FpPolyT<Elt>>& out) {
    const Elt ell = h.ell;
    if (h.degree() == static_cast<int64_t>(e)) {
        out.push_back(h.monic());
        return;
    }
    const uint32_t kMaxTries = 4096; // deterministic sequence; failure prob ~ 2^-kMaxTries
    for (uint32_t attempt = 0; attempt < kMaxTries; ++attempt) {
        FpPolyT<Elt> a = random_nonconstant_below(ell, h.degree(), rng);
        // Cheap lucky split.
        FpPolyT<Elt> g0 = FpPolyT<Elt>::gcd(a, h);
        if (g0.degree() > 0 && g0.degree() < h.degree()) {
            equal_degree(g0, e, rng, out);
            equal_degree(FpPolyT<Elt>::divide_exact(h, g0), e, rng, out);
            return;
        }
        FpPolyT<Elt> g(ell);
        if (ell == 2) {
            // Trace-map variant (terminates for ell = 2).
            FpPolyT<Elt> c = FpPolyT<Elt>::mod(a, h);
            FpPolyT<Elt> t = c;
            for (uint32_t i = 1; i < e; ++i) {
                t = FpPolyT<Elt>::powmod(t, 2ull, h);
                c = c + t;
            }
            g = FpPolyT<Elt>::gcd(c, h);
        } else {
            // b = a^{(ell^e - 1)/2} mod h via the iterated-Frobenius "norm".
            FpPolyT<Elt> u = FpPolyT<Elt>::mod(a, h);
            FpPolyT<Elt> nrm = u;
            for (uint32_t i = 1; i < e; ++i) {
                u = FpPolyT<Elt>::powmod(u, static_cast<uint64_t>(ell), h);
                nrm = FpPolyT<Elt>::mod(nrm * u, h);
            }
            FpPolyT<Elt> b = FpPolyT<Elt>::powmod(nrm, static_cast<uint64_t>((ell - 1) / 2), h);
            g = FpPolyT<Elt>::gcd(b - FpPolyT<Elt>::one(ell), h);
        }
        if (g.degree() > 0 && g.degree() < h.degree()) {
            equal_degree(g, e, rng, out);
            equal_degree(FpPolyT<Elt>::divide_exact(h, g), e, rng, out);
            return;
        }
    }
    throw std::runtime_error("equal_degree: Cantor-Zassenhaus split budget exhausted");
}

} // namespace

template<class Elt>
std::vector<std::pair<FpPolyT<Elt>, uint32_t>> SquarefreeDecomposition(const FpPolyT<Elt>& f) {
    if (f.is_zero()) throw std::invalid_argument("SquarefreeDecomposition of the zero polynomial");
    std::vector<std::pair<FpPolyT<Elt>, uint32_t>> out;
    if (f.degree() < 1) return out;
    sfd_rec(f.monic(), 1, out);
    return out;
}

template<class Elt>
uint32_t CertifyIrreducible(const FpPolyT<Elt>& pi_in) {
    if (pi_in.is_zero() || pi_in.degree() < 1) return 1;
    const FpPolyT<Elt> pi = pi_in.monic();
    const Elt ell = pi.ell;
    const uint32_t n = static_cast<uint32_t>(pi.degree());
    const FpPolyT<Elt> x_mod_pi = FpPolyT<Elt>::mod(FpPolyT<Elt>::X(ell), pi);
    FpPolyT<Elt> h = x_mod_pi; // X^{ell^0} mod pi
    for (uint32_t i = 1; i <= n; ++i) {
        h = FpPolyT<Elt>::powmod(h, static_cast<uint64_t>(ell), pi); // X^{ell^i} mod pi
        if (i <= n / 2) {
            if (FpPolyT<Elt>::gcd(h - x_mod_pi, pi).degree() != 0) return 2;
        }
    }
    // X^{ell^n} == X (mod pi): every root lies in F_{ell^n}.
    if (h != x_mod_pi) return 3;
    return 0;
}

template<class Elt>
uint32_t CertifyFactorization(const FpPolyT<Elt>& f, const FpPolyFactorizationT<Elt>& fac) {
    if (f.is_zero()) return 1;
    if (fac.ell != f.ell) return 2;
    if (fac.leading_coeff % f.ell != f.leading_coeff()) return 3;
    FpPolyT<Elt> product = FpPolyT<Elt>::constant(f.ell, fac.leading_coeff);
    for (size_t j = 0; j < fac.factors.size(); ++j) {
        const FpPolyFactorT<Elt>& fj = fac.factors[j];
        if (fj.multiplicity == 0) return 4;
        if (fj.pi.ell != f.ell) return 5;
        if (!fj.pi.is_monic() || fj.pi.degree() < 1) return 6;
        if (j > 0 && !CanonicalLess(fac.factors[j - 1].pi, fj.pi)) return 7; // strict order => pairwise distinct
        if (CertifyIrreducible(fj.pi) != 0) return 8;
        product = product * FpPolyT<Elt>::pow(fj.pi, fj.multiplicity);
    }
    if (product != f) return 9;
    return 0;
}

template<class Elt>
FpPolyFactorizationT<Elt> FactorPoly(const FpPolyT<Elt>& f, uint64_t seed) {
    if (f.is_zero()) throw std::invalid_argument("FactorPoly of the zero polynomial");
    FpPolyFactorizationT<Elt> fac;
    fac.ell = f.ell;
    fac.leading_coeff = f.leading_coeff();
    if (f.degree() >= 1) {
        SplitMix64 rng(seed);
        for (const auto& sf : SquarefreeDecomposition(f)) {
            for (const auto& dd : distinct_degree(sf.first)) {
                std::vector<FpPolyT<Elt>> irreducibles;
                equal_degree(dd.first, dd.second, rng, irreducibles);
                for (auto& pi : irreducibles) {
                    fac.factors.push_back(FpPolyFactorT<Elt>{std::move(pi), sf.second});
                }
            }
        }
    }
    std::sort(fac.factors.begin(), fac.factors.end(),
              [](const FpPolyFactorT<Elt>& a, const FpPolyFactorT<Elt>& b) {
                  return CanonicalLess(a.pi, b.pi);
              });
    // CERTIFICATION (deterministic, unconditional; ALWAYS run).
    const uint32_t cert = CertifyFactorization(f, fac);
    if (cert != 0) {
        throw std::runtime_error("FactorPoly: certification failed (code " +
                                 std::to_string(cert) + ") for " + f.to_string());
    }
    return fac;
}

// ===========================================================================
// Explicit instantiations (Stage S13): uint32_t (default, alias FpPoly) and
// uint64_t (extended regime, alias FpPoly64).
// ===========================================================================
#define INSTANTIATE_FPPOLY(Elt) \
    template class FpPolyT<Elt>; \
    template bool CanonicalLess<Elt>(const FpPolyT<Elt>&, const FpPolyT<Elt>&); \
    template FpPolyT<Elt> CharPolyBerkowitz<Elt>(const MatrixT<Elt>&); \
    template MatrixT<Elt> EvalPolyAtMatrix<Elt>(const FpPolyT<Elt>&, const MatrixT<Elt>&); \
    template FpPolyT<Elt> MinPolyKrylov<Elt>(const MatrixT<Elt>&); \
    template std::vector<std::pair<FpPolyT<Elt>, uint32_t>> SquarefreeDecomposition<Elt>(const FpPolyT<Elt>&); \
    template uint32_t CertifyIrreducible<Elt>(const FpPolyT<Elt>&); \
    template uint32_t CertifyFactorization<Elt>(const FpPolyT<Elt>&, const FpPolyFactorizationT<Elt>&); \
    template FpPolyFactorizationT<Elt> FactorPoly<Elt>(const FpPolyT<Elt>&, uint64_t);

INSTANTIATE_FPPOLY(uint32_t)
INSTANTIATE_FPPOLY(uint64_t)
#undef INSTANTIATE_FPPOLY

} // namespace hecke
