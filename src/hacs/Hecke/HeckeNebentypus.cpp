// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-10) Nebentypus on the fixed model by Shapiro -- see
// HeckeNebentypus.h ([C, sec:nebentypus]; [GT, conv:nebentypus],
// [GT, bridge:nebentypus-dictionary]).
#include "HeckeNebentypus.h"

#include <algorithm>
#include <array>
#include <map>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeLevelNHeilbronn.h"   // MerelSet
#include "HeckeTauTable.h"     // ChainTerm
#include "../LocalSystemSL2Z.h" // GenerateSL2ZRelationsWithGroupData
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace hecke {
namespace neb {

namespace {

using engine::scalar::gcd;
using engine::scalar::modi;
using engine::scalar::mulmod;
using engine::scalar::powmod;

using u128 = unsigned __int128;

uint64_t Lcm(uint64_t a, uint64_t b) { return a / gcd<uint64_t>(a, b) * b; }
std::vector<uint64_t> PrimeDivisors(uint64_t n) {
    std::vector<uint64_t> out;
    for (uint64_t p = 2; p * p <= n; ++p)
        if (n % p == 0) {
            out.push_back(p);
            while (n % p == 0) n /= p;
        }
    if (n > 1) out.push_back(n);
    return out;
}
// Inverse of a unit a mod m (extended Euclid); throws on a non-unit.
uint64_t InvModN(uint64_t a, uint64_t m) {
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "InvModN: not a unit");
    return x;
}

// rho_n(h) mod ell as d x d F_ell codes (memoised on the reduced key).
class RhoFl {
public:
    RhoFl(uint32_t ell, uint32_t n) : ell_(ell), n_(n) {}
    uint32_t index_of(const Mat22& g, std::vector<std::vector<uint32_t>>& store) {
        const std::array<uint32_t, 4> key = g.reduced_mod(ell_);
        auto it = memo_.find(key);
        if (it != memo_.end()) return it->second;
        const FlatMatrixT<uint32_t> A(2, 2, ell_, std::vector<uint32_t>{key[0], key[1], key[2], key[3]});
        const FlatMatrixT<uint32_t> R = ComputeSymmetricPowerMatrixFM(A, n_);
        const uint32_t d = n_ + 1;
        std::vector<uint32_t> b(static_cast<size_t>(d) * d);
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t c = 0; c < d; ++c) b[static_cast<size_t>(r) * d + c] = R.get_entry(r, c);
        store.push_back(std::move(b));
        const uint32_t idx = static_cast<uint32_t>(store.size() - 1);
        memo_.emplace(key, idx);
        return idx;
    }
private:
    uint32_t ell_, n_;
    std::map<std::array<uint32_t, 4>, uint32_t> memo_;
};

uint32_t CoeffMod(int64_t c, uint32_t p) {
    const int64_t r = c % static_cast<int64_t>(p);
    return static_cast<uint32_t>(r < 0 ? r + static_cast<int64_t>(p) : r);
}

} // namespace

// ============================================================================
// F_q
// ============================================================================
Fq::Fq(uint32_t ell, const FpPoly& C) : ell_(ell), C_(C) {
    ENGINE_CHECK_MSG(ell >= 2 && ell < (1u << 31) && hecke::IsPrimeU64(ell), "Fq: ell must be a prime < 2^31");
    ENGINE_CHECK_MSG(C.ell == ell && C.is_monic() && C.degree() >= 1, "Fq: C must be monic of degree >= 1 over F_ell");
    ENGINE_CHECK_MSG(hecke::CertifyIrreducible(C) == 0, "Fq: C is not certified irreducible (hecke::CertifyIrreducible)");
    d_ = static_cast<uint32_t>(C.degree());
    q_ = 1;
    for (uint32_t i = 0; i < d_; ++i) {
        q_ *= ell_;
        if (d_ >= 2 && q_ > (1ull << 22))
            throw NebentypusRefusal("field_too_large", "F_q with d >= 2 needs q = ell^d <= 2^22 (log tables)");
    }
    qm1_primes_ = PrimeDivisors(q_ - 1);
    if (d_ >= 2) {
        // a generator of F_q^x by search in code order, then the exp / log tables.
        auto is_gen = [&](E g) {
            if (g == 0) return false;
            for (uint64_t r : qm1_primes_) {
                E x = 1, b = g;
                uint64_t e = (q_ - 1) / r;
                while (e) { if (e & 1) x = poly_mul(x, b); b = poly_mul(b, b); e >>= 1; }
                if (x == 1) return false;
            }
            return true;
        };
        E g = 2;
        while (!is_gen(g)) ++g;
        exp_.assign(q_ - 1, 0);
        log_.assign(q_, 0);
        E x = 1;
        for (uint64_t i = 0; i + 1 < q_; ++i) {
            exp_[i] = x;
            log_[x] = static_cast<uint32_t>(i);
            x = poly_mul(x, g);
        }
        ENGINE_CHECK_MSG(x == 1, "Fq: the generator does not have order q - 1");
    }
}

Fq Fq::Prime(uint32_t ell) { return Fq(ell, FpPoly(ell, std::vector<uint32_t>{0, 1})); }

std::string Fq::name() const {
    return d_ == 1 ? "F_" + std::to_string(ell_) : "F_{" + std::to_string(ell_) + "^" + std::to_string(d_) + "}";
}

std::vector<uint32_t> Fq::digits(E a) const {
    std::vector<uint32_t> c(d_, 0);
    for (uint32_t i = 0; i < d_; ++i) { c[i] = static_cast<uint32_t>(a % ell_); a /= ell_; }
    return c;
}

Fq::E Fq::from_digits(const std::vector<uint32_t>& c) const {
    uint64_t x = 0;
    for (uint32_t i = d_; i-- > 0;) x = x * ell_ + (i < c.size() ? c[i] % ell_ : 0);
    return static_cast<E>(x);
}

Fq::E Fq::from_int(int64_t a) const { return static_cast<E>(modi(a, ell_)); }

Fq::E Fq::add(E a, E b) const {
    if (d_ == 1) { const uint64_t s = static_cast<uint64_t>(a) + b; return static_cast<E>(s >= ell_ ? s - ell_ : s); }
    uint64_t out = 0, pw = 1;
    for (uint32_t i = 0; i < d_; ++i) {
        const uint64_t s = a % ell_ + b % ell_;
        out += (s >= ell_ ? s - ell_ : s) * pw;
        a /= ell_; b /= ell_; pw *= ell_;
    }
    return static_cast<E>(out);
}

Fq::E Fq::neg(E a) const {
    if (d_ == 1) return a == 0 ? 0 : ell_ - a;
    uint64_t out = 0, pw = 1;
    for (uint32_t i = 0; i < d_; ++i) {
        const uint64_t c = a % ell_;
        out += (c == 0 ? 0 : ell_ - c) * pw;
        a /= ell_; pw *= ell_;
    }
    return static_cast<E>(out);
}

Fq::E Fq::sub(E a, E b) const { return add(a, neg(b)); }

Fq::E Fq::mul(E a, E b) const {
    if (d_ == 1) return static_cast<E>(static_cast<uint64_t>(a) * b % ell_);
    if (a == 0 || b == 0) return 0;
    uint64_t s = static_cast<uint64_t>(log_[a]) + log_[b];
    if (s >= q_ - 1) s -= q_ - 1;
    return exp_[s];
}

Fq::E Fq::inv(E a) const {
    ENGINE_CHECK_MSG(a != 0, "Fq::inv: zero");
    if (d_ == 1) return static_cast<E>(powmod<uint64_t>(a, ell_ - 2, ell_));
    const uint64_t l = log_[a];
    return exp_[l == 0 ? 0 : (q_ - 1) - l];
}

Fq::E Fq::pow(E a, uint64_t e) const {
    E r = 1, b = a;
    while (e) { if (e & 1) r = mul(r, b); b = mul(b, b); e >>= 1; }
    return r;
}

Fq::E Fq::poly_mul(E a, E b) const {
    const std::vector<uint32_t> x = digits(a), y = digits(b);
    std::vector<uint64_t> z(2 * d_ - 1, 0);
    for (uint32_t i = 0; i < d_; ++i)
        for (uint32_t j = 0; j < d_; ++j) z[i + j] = (z[i + j] + static_cast<uint64_t>(x[i]) * y[j]) % ell_;
    // reduce by the monic C: Y^d = -sum_{i<d} c_i Y^i.
    for (uint32_t k = 2 * d_ - 1; k-- > d_;) {
        const uint64_t t = z[k];
        if (t == 0) continue;
        z[k] = 0;
        for (uint32_t i = 0; i < d_; ++i)
            z[k - d_ + i] = (z[k - d_ + i] + (ell_ - t) * C_.coeff(i)) % ell_;
    }
    std::vector<uint32_t> c(d_);
    for (uint32_t i = 0; i < d_; ++i) c[i] = static_cast<uint32_t>(z[i]);
    return from_digits(c);
}

bool Fq::less(E a, E b) const {
    const std::vector<uint32_t> x = digits(a), y = digits(b);
    auto deg = [&](const std::vector<uint32_t>& v) {
        int64_t dg = -1;
        for (uint32_t i = 0; i < d_; ++i) if (v[i]) dg = i;
        return dg;
    };
    const int64_t dx = deg(x), dy = deg(y);
    if (dx != dy) return dx < dy;
    return x < y;   // lexicographic on c_0, c_1, ...
}

uint64_t Fq::order_of(E a) const {
    ENGINE_CHECK_MSG(a != 0, "Fq::order_of: zero");
    uint64_t o = q_ - 1;
    for (uint64_t r : qm1_primes_)
        while (o % r == 0 && pow(a, o / r) == 1) o /= r;
    return o;
}

Fq::E Fq::least_root_of_unity(uint64_t o) const {
    ENGINE_CHECK_MSG(o >= 1 && (q_ - 1) % o == 0, "Fq::least_root_of_unity: o must divide q - 1");
    if (o == 1) return 1;
    // a generator of F_q^x, then zeta = g^{(q-1)/o} of exact order o.
    E g = 0;
    if (d_ >= 2) g = exp_[1];
    else {
        for (E c = 2; c < ell_ && g == 0; ++c) if (order_of(c) == q_ - 1) g = c;
        if (ell_ == 2) g = 1;
    }
    const E zeta = pow(g, (q_ - 1) / o);
    ENGINE_CHECK_MSG(order_of(zeta) == o, "Fq::least_root_of_unity: zeta has the wrong order");
    E best = 0;
    bool have = false;
    E z = 1;
    for (uint64_t k = 1; k <= o; ++k) {
        z = mul(z, zeta);
        if (gcd<uint64_t>(k, o) != 1) continue;
        if (!have || less(z, best)) { best = z; have = true; }
    }
    return best;
}

std::string Fq::text(E a) const {
    if (d_ == 1) return std::to_string(a);
    std::string s = "[";
    const std::vector<uint32_t> c = digits(a);
    for (uint32_t i = 0; i < d_; ++i) s += (i ? "," : "") + std::to_string(c[i]);
    return s + "]";
}

// ============================================================================
// Dense linear algebra over F_q
// ============================================================================
FqMat FqMul(const Fq& F, const FqMat& A, const FqMat& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows, "FqMul: shapes");
    FqMat C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t k = 0; k < A.cols; ++k) {
            const uint32_t a = A.at(i, k);
            if (a == 0) continue;
            for (uint32_t j = 0; j < B.cols; ++j) {
                const uint32_t b = B.at(k, j);
                if (b) C.at(i, j) = F.add(C.at(i, j), F.mul(a, b));
            }
        }
    return C;
}

FqMat FqAdd(const Fq& F, const FqMat& A, const FqMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.cols == B.cols, "FqAdd: shapes");
    FqMat C(A.rows, A.cols);
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] = F.add(A.a[i], B.a[i]);
    return C;
}

FqMat FqScale(const Fq& F, const FqMat& A, uint32_t s) {
    FqMat C = A;
    for (uint32_t& x : C.a) x = F.mul(x, s);
    return C;
}

FqMat FqIdentity(uint32_t n) {
    FqMat I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.at(i, i) = 1;
    return I;
}

bool FqEqual(const FqMat& A, const FqMat& B) { return A.rows == B.rows && A.cols == B.cols && A.a == B.a; }

std::vector<uint32_t> FqRref(const Fq& F, FqMat& A) {
    std::vector<uint32_t> piv;
    uint32_t r = 0;
    for (uint32_t c = 0; c < A.cols && r < A.rows; ++c) {
        uint32_t p = r;
        while (p < A.rows && A.at(p, c) == 0) ++p;
        if (p == A.rows) continue;
        if (p != r)
            for (uint32_t k = 0; k < A.cols; ++k) std::swap(A.at(p, k), A.at(r, k));
        const uint32_t iv = F.inv(A.at(r, c));
        for (uint32_t k = c; k < A.cols; ++k) A.at(r, k) = F.mul(A.at(r, k), iv);
        for (uint32_t i = 0; i < A.rows; ++i) {
            if (i == r || A.at(i, c) == 0) continue;
            const uint32_t f = F.neg(A.at(i, c));
            for (uint32_t k = c; k < A.cols; ++k)
                if (A.at(r, k)) A.at(i, k) = F.add(A.at(i, k), F.mul(f, A.at(r, k)));
        }
        piv.push_back(c);
        ++r;
    }
    return piv;
}

uint32_t FqRank(const Fq& F, FqMat A) { return static_cast<uint32_t>(FqRref(F, A).size()); }

FqMat FqColumnBasis(const Fq& F, const FqMat& A) {
    FqMat R = A;
    const std::vector<uint32_t> piv = FqRref(F, R);
    FqMat B(A.rows, static_cast<uint32_t>(piv.size()));
    for (uint32_t j = 0; j < piv.size(); ++j)
        for (uint32_t i = 0; i < A.rows; ++i) B.at(i, j) = A.at(i, piv[j]);
    return B;
}

FqMat FqSolve(const Fq& F, const FqMat& A, const FqMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows, "FqSolve: shapes");
    FqMat M(A.rows, A.cols + B.cols);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) M.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) M.at(i, A.cols + j) = B.at(i, j);
    }
    const std::vector<uint32_t> piv = FqRref(F, M);
    ENGINE_CHECK_MSG(piv.size() >= A.cols, "FqSolve: A is not of full column rank or B not in its span");
    for (uint32_t j = 0; j < A.cols; ++j) ENGINE_CHECK_MSG(piv[j] == j, "FqSolve: A is not of full column rank");
    for (size_t j = A.cols; j < piv.size(); ++j)
        ENGINE_CHECK_MSG(piv[j] < A.cols, "FqSolve: B is not in the column span of A");
    FqMat X(A.cols, B.cols);
    for (uint32_t i = 0; i < A.cols; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) X.at(i, j) = M.at(i, A.cols + j);
    return X;
}

std::vector<uint32_t> FqCharpoly(const Fq& F, const FqMat& A0) {
    ENGINE_CHECK_MSG(A0.rows == A0.cols, "FqCharpoly: a square matrix");
    const uint32_t n = A0.rows;
    FqMat H = A0;
    // Upper Hessenberg form by similarity (Cohen, Alg. 2.2.9).
    for (uint32_t j = 0; j + 2 < n; ++j) {
        uint32_t i = j + 1;
        while (i < n && H.at(i, j) == 0) ++i;
        if (i == n) continue;
        if (i != j + 1) {
            for (uint32_t k = 0; k < n; ++k) std::swap(H.at(i, k), H.at(j + 1, k));
            for (uint32_t k = 0; k < n; ++k) std::swap(H.at(k, i), H.at(k, j + 1));
        }
        const uint32_t piv_inv = F.inv(H.at(j + 1, j));
        for (uint32_t k = j + 2; k < n; ++k) {
            if (H.at(k, j) == 0) continue;
            const uint32_t u = F.mul(H.at(k, j), piv_inv);
            for (uint32_t c = 0; c < n; ++c) H.at(k, c) = F.sub(H.at(k, c), F.mul(u, H.at(j + 1, c)));
            for (uint32_t r = 0; r < n; ++r) H.at(r, j + 1) = F.add(H.at(r, j + 1), F.mul(u, H.at(r, k)));
        }
    }
    // p_0 = 1; p_m = (X - h_mm) p_{m-1} - sum_{i=1}^{m-1} h_{m-i,m} prod_{j=m-i+1}^{m} h_{j,j-1} p_{m-i-1}.
    std::vector<std::vector<uint32_t>> P(n + 1);
    P[0] = {1};
    auto h = [&](uint32_t r, uint32_t c) { return H.at(r - 1, c - 1); };   // 1-based
    for (uint32_t m = 1; m <= n; ++m) {
        std::vector<uint32_t> cur(m + 1, 0);
        for (uint32_t k = 0; k < P[m - 1].size(); ++k) {
            cur[k + 1] = F.add(cur[k + 1], P[m - 1][k]);
            cur[k] = F.sub(cur[k], F.mul(h(m, m), P[m - 1][k]));
        }
        uint32_t prod = 1;
        for (uint32_t i = 1; i <= m - 1; ++i) {
            prod = F.mul(prod, h(m - i + 1, m - i));
            const uint32_t coef = F.mul(h(m - i, m), prod);
            if (coef == 0) continue;
            for (uint32_t k = 0; k < P[m - i - 1].size(); ++k) cur[k] = F.sub(cur[k], F.mul(coef, P[m - i - 1][k]));
        }
        P[m] = std::move(cur);
    }
    return P[n];
}

std::string FqPolyText(const Fq& F, const std::vector<uint32_t>& f) {
    std::string s = "[";
    for (size_t i = 0; i < f.size(); ++i) s += (i ? "," : "") + F.text(f[i]);
    return s + "]";
}

// ============================================================================
// W_K(F_q) and the Teichmuller lift
// ============================================================================
GaloisRing::GaloisRing(const Fq& F, uint32_t K) : F_(&F), K_(K), d_(F.degree()) {
    ENGINE_CHECK_MSG(K >= 1, "GaloisRing: K >= 1");
    u128 m = 1;
    for (uint32_t i = 0; i < K; ++i) {
        m *= F.ell();
        ENGINE_CHECK_MSG(m < (static_cast<u128>(1) << 62), "GaloisRing: ell^K < 2^62");
    }
    mod_ = static_cast<uint64_t>(m);
    C_.assign(d_ + 1, 0);
    for (uint32_t i = 0; i <= d_; ++i) C_[i] = F.defining_poly().coeff(i);   // the coefficientwise lift in [0, ell)
}

GaloisRing::W GaloisRing::one() const { W w(d_, 0); w[0] = 1 % mod_; return w; }
GaloisRing::W GaloisRing::scalar(uint64_t s) const { W w(d_, 0); w[0] = s % mod_; return w; }

GaloisRing::W GaloisRing::lift(Fq::E x) const {
    const std::vector<uint32_t> c = F_->digits(x);
    W w(d_);
    for (uint32_t i = 0; i < d_; ++i) w[i] = c[i];
    return w;
}

GaloisRing::W GaloisRing::add(const W& a, const W& b) const {
    W w(d_);
    for (uint32_t i = 0; i < d_; ++i) w[i] = (a[i] + b[i]) % mod_;
    return w;
}

GaloisRing::W GaloisRing::mul(const W& a, const W& b) const {
    std::vector<uint64_t> z(2 * d_ - 1, 0);
    for (uint32_t i = 0; i < d_; ++i)
        for (uint32_t j = 0; j < d_; ++j) z[i + j] = (z[i + j] + mulmod(a[i], b[j], mod_)) % mod_;
    for (uint32_t k = 2 * d_ - 1; k-- > d_;) {
        const uint64_t t = z[k];
        if (t == 0) continue;
        z[k] = 0;
        for (uint32_t i = 0; i < d_; ++i) z[k - d_ + i] = (z[k - d_ + i] + mod_ - mulmod(t, C_[i], mod_)) % mod_;
    }
    W w(d_);
    for (uint32_t i = 0; i < d_; ++i) w[i] = z[i];
    return w;
}

GaloisRing::W GaloisRing::pow(const W& a, uint64_t e) const {
    W r = one(), b = a;
    while (e) { if (e & 1) r = mul(r, b); b = mul(b, b); e >>= 1; }
    return r;
}

Fq::E GaloisRing::reduce(const W& a) const {
    std::vector<uint32_t> c(d_);
    for (uint32_t i = 0; i < d_; ++i) c[i] = static_cast<uint32_t>(a[i] % F_->ell());
    return F_->from_digits(c);
}

GaloisRing::W GaloisRing::teichmuller(Fq::E x) const {
    ENGINE_CHECK_MSG(x != 0, "GaloisRing::teichmuller: x != 0");
    W y = lift(x);
    for (uint32_t i = 1; i < K_; ++i) y = pow(y, F_->q());   // lift(x)^{q^{K-1}}
    return y;
}

// ============================================================================
// Dirichlet characters
// ============================================================================
std::vector<uint64_t> UnitsMod(uint64_t N) {
    std::vector<uint64_t> u;
    if (N <= 1) return {0};
    for (uint64_t a = 1; a < N; ++a) if (gcd<uint64_t>(a, N) == 1) u.push_back(a);
    return u;
}

bool DirichletCharacter::is_unit(int64_t a) const { return N <= 1 || gcd<uint64_t>(modi(a, N), N) == 1; }

int64_t DirichletCharacter::exponent_at(int64_t a) const {
    ENGINE_CHECK_MSG(is_unit(a), "DirichletCharacter: a non-unit argument");
    if (N <= 1) return 0;
    return expo[modi(a, N)];
}

uint64_t DirichletCharacter::order() const {
    uint64_t g = L;
    for (int64_t e : expo) if (e >= 0) g = gcd<uint64_t>(g, static_cast<uint64_t>(e));
    return L / g;
}

int DirichletCharacter::parity() const {
    if (N <= 2) return 1;
    const int64_t e = exponent_at(static_cast<int64_t>(N) - 1);
    ENGINE_CHECK_MSG((2 * e) % static_cast<int64_t>(L) == 0, "DirichletCharacter: chi(-1)^2 != 1");
    return e == 0 ? 1 : -1;
}

DirichletCharacter DirichletCharacter::inverse() const {
    DirichletCharacter c = *this;
    c.conrey = 0;
    for (int64_t& e : c.expo) if (e > 0) e = static_cast<int64_t>(L) - e;
    if (N > 1 && conrey) c.conrey = InvModN(conrey, N);   // Conrey: chi_N(m)^{-1} = chi_N(m^{-1})
    return c;
}

DirichletCharacter DirichletCharacter::mul(const DirichletCharacter& o) const {
    ENGINE_CHECK_MSG(N == o.N, "DirichletCharacter::mul: moduli differ");
    DirichletCharacter c;
    c.N = N;
    c.L = Lcm(L, o.L);
    c.conrey = (N > 1 && conrey && o.conrey) ? (conrey * o.conrey) % N : (N <= 1 ? 1 : 0);
    c.expo.assign(expo.size(), -1);
    for (size_t i = 0; i < expo.size(); ++i)
        if (expo[i] >= 0)
            c.expo[i] = static_cast<int64_t>((static_cast<uint64_t>(expo[i]) * (c.L / L) +
                                              static_cast<uint64_t>(o.expo[i]) * (c.L / o.L)) % c.L);
    return c;
}

std::string DirichletCharacter::label() const {
    return std::to_string(N) + "." + (conrey ? std::to_string(conrey) : std::string("?"));
}

DirichletCharacter DirichletCharacter::Trivial(uint64_t N) {
    DirichletCharacter c;
    c.N = std::max<uint64_t>(N, 1);
    c.conrey = 1;
    c.L = 1;
    c.expo.assign(c.N, -1);
    for (uint64_t u : UnitsMod(c.N)) c.expo[u] = 0;
    return c;
}

DirichletCharacter DirichletCharacter::Conrey(uint64_t N, uint64_t m) {
    ENGINE_CHECK_MSG(N >= 1 && N < (1ull << 31), "DirichletCharacter::Conrey: N in [1, 2^31)");
    if (N == 1) return Trivial(1);
    ENGINE_CHECK_MSG(gcd<uint64_t>(m % N, N) == 1, "DirichletCharacter::Conrey: gcd(m, N) != 1");
    // Component data: (Q = p^e, denominators) and the per-component exponent function.
    struct Comp { uint64_t p = 0, e = 0, Q = 1, g = 0, phi = 1; std::vector<uint32_t> dlog; };
    std::vector<Comp> comps;
    uint64_t n = N;
    for (uint64_t p : PrimeDivisors(N)) {
        Comp c;
        c.p = p;
        while (n % p == 0) { n /= p; ++c.e; c.Q *= p; }
        c.phi = c.Q / p * (p - 1);
        if (p != 2) {
            // the least g primitive modulo p^max(e, 2) (hence modulo every p^e).  This IS the LMFDB / Conrey
            // convention ("the least positive integer g_p which is a primitive root for all p^e", LMFDB knowl
            // character.dirichlet.conrey), also at e = 1: at p = 40487 the least primitive root mod p (5) is not one
            // mod p^2, and the generator is 10, here as in LMFDB (S16 WP-10/11 repairs, audit finding 10; pinned in (F)).
            const uint64_t Q2 = c.e >= 2 ? c.Q : p * p;
            const uint64_t phi2 = Q2 / p * (p - 1);
            const std::vector<uint64_t> pr = PrimeDivisors(phi2);
            for (uint64_t g = 2;; ++g) {
                if (g % p == 0) continue;
                bool ok = true;
                for (uint64_t r : pr) if (powmod<uint64_t>(g, phi2 / r, Q2) == 1) { ok = false; break; }
                if (ok) { c.g = g; break; }
            }
            // the discrete-log table modulo Q, built once (O(phi(Q)); S16 WP-10/11 repairs, audit finding 7: it was a
            // linear scan per unit, O(phi(N) phi(Q)) in all).  Values unchanged: dlog[g^k mod Q] = k, k < phi.
            c.dlog.assign(c.Q, UINT32_MAX);
            uint64_t x = 1;
            for (uint64_t k = 0; k < c.phi; ++k) {
                ENGINE_CHECK_MSG(c.dlog[x] == UINT32_MAX, "DirichletCharacter::Conrey: g is not a generator modulo p^e");
                c.dlog[x] = static_cast<uint32_t>(k);
                x = mulmod(x, c.g, c.Q);
            }
        }
        if (p == 2 && c.e >= 3) {
            // the 2-adic log table of 5 modulo 2^e, built once (audit finding 7, as for odd p).
            c.dlog.assign(c.Q, UINT32_MAX);
            uint64_t x = 1;
            for (uint64_t k = 0; k < c.Q / 4; ++k) { c.dlog[x] = static_cast<uint32_t>(k); x = mulmod(x, 5, c.Q); }
        }
        comps.push_back(c);
    }
    uint64_t L = 1;
    for (const Comp& c : comps) {
        if (c.p != 2) L = Lcm(L, c.phi);
        else if (c.e >= 2) L = Lcm(L, c.e >= 3 ? Lcm(2, c.Q / 4) : 2);
    }
    // discrete logs, per component.
    auto dlog_odd = [](const Comp& c, uint64_t a) -> uint64_t {
        const uint32_t k = c.dlog[a % c.Q];
        ENGINE_CHECK_MSG(k != UINT32_MAX, "DirichletCharacter::Conrey: no discrete log");
        return k;
    };
    // 2^e: a = eps * 5^k mod 2^e, eps in {+1, -1} (eps the residue mod 4).
    auto dlog_two = [](const Comp& c, uint64_t a, int& eps) -> uint64_t {
        a %= c.Q;
        eps = (a % 4 == 1) ? 1 : -1;
        if (c.e <= 2) return 0;
        const uint64_t t = eps == 1 ? a : (c.Q - a) % c.Q;
        const uint32_t k = c.dlog[t];
        ENGINE_CHECK_MSG(k != UINT32_MAX, "DirichletCharacter::Conrey: no 2-adic log");
        return k;
    };
    DirichletCharacter chi;
    chi.N = N;
    chi.conrey = m % N;
    chi.L = L;
    chi.expo.assign(N, -1);
    for (uint64_t a : UnitsMod(N)) {
        uint64_t k = 0;   // numerator over L
        for (const Comp& c : comps) {
            if (c.p != 2) {
                const uint64_t num = mulmod(dlog_odd(c, m % c.Q), dlog_odd(c, a % c.Q), c.phi);   // / phi
                k = (k + num * (L / c.phi)) % L;
            } else if (c.e >= 2) {
                int em = 1, ea = 1;
                const uint64_t am = dlog_two(c, m, em), aa = dlog_two(c, a, ea);
                if (em == -1 && ea == -1) k = (k + L / 2) % L;   // (1 - eps_m)(1 - eps_a) / 8 = 1/2
                if (c.e >= 3) {
                    const uint64_t den = c.Q / 4;
                    k = (k + mulmod(am, aa, den) * (L / den)) % L;
                }
            }
        }
        chi.expo[a] = static_cast<int64_t>(k);
    }
    return chi;
}

std::vector<DirichletCharacter> AllCharacters(uint64_t N) {
    std::vector<DirichletCharacter> out;
    if (N <= 1) { out.push_back(DirichletCharacter::Trivial(1)); return out; }
    for (uint64_t m : UnitsMod(N)) out.push_back(DirichletCharacter::Conrey(N, m));
    return out;
}

Fq::E CharacterValues::at(int64_t a) const {
    if (N <= 1) return 1;
    const uint64_t r = modi(a, N);
    ENGINE_CHECK_MSG(gcd<uint64_t>(r, N) == 1, "CharacterValues::at: a non-unit argument");
    return val[r];
}

CharacterValues ReduceCharacter(const DirichletCharacter& chi, const Fq& F) {
    const uint64_t o = chi.order();
    if (o % F.ell() == 0)
        throw NebentypusRefusal("character_wild", "ell = " + std::to_string(F.ell()) + " divides ord chi = " + std::to_string(o) +
                                                      " (a ramified coefficient ring; rmk:nebfield (2))");
    if ((F.q() - 1) % o != 0)
        throw NebentypusRefusal("character_field", "ord chi = " + std::to_string(o) + " does not divide q - 1 = " +
                                                       std::to_string(F.q() - 1) + " (" + F.name() + ")");
    CharacterValues v;
    v.N = chi.N;
    v.order = o;
    v.s = F.least_root_of_unity(o);
    v.val.assign(std::max<uint64_t>(chi.N, 1), 0);
    if (chi.N <= 1) { v.val[0] = 1; return v; }
    const uint64_t step = chi.L / o;
    for (uint64_t a : UnitsMod(chi.N)) {
        const uint64_t k = static_cast<uint64_t>(chi.expo[a]);
        ENGINE_CHECK_MSG(k % step == 0, "ReduceCharacter: an exponent is not a multiple of L / ord");
        v.val[a] = F.pow(v.s, k / step);
    }
    return v;
}

std::vector<GaloisRing::W> LiftCharacter(const CharacterValues& psi, const GaloisRing& W) {
    std::vector<GaloisRing::W> out(psi.val.size(), GaloisRing::W(W.field().degree(), 0));
    for (size_t a = 0; a < psi.val.size(); ++a)
        if (psi.val[a] != 0) out[a] = W.teichmuller(psi.val[a]);
    return out;
}

// ============================================================================
// prop:neborbit (4) in characteristic 0
// ============================================================================
EulerPrediction EulerCharacteristicFormula(uint64_t N, uint32_t n, const DirichletCharacter& chi) {
    ENGINE_CHECK_MSG(chi.N == std::max<uint64_t>(N, 1), "EulerCharacteristicFormula: chi has another modulus");
    EulerPrediction E;
    E.m = level::IndexGamma0(N);
    E.parity_ok = (chi.parity() == 1) == (n % 2 == 0);
    if (!E.parity_ok) return E;
    const int64_t L = static_cast<int64_t>(chi.L);
    const int64_t d = n + 1;
    int64_t sumS = 0, sumU = 0;
    const uint64_t NN = std::max<uint64_t>(N, 1);
    for (uint64_t x = 0; x < NN; ++x) {
        if ((x * x + 1) % NN == 0) {   // lambda: (c : d) S = lambda (c : d)
            ++E.nu2;
            const int64_t k = chi.exponent_at(-static_cast<int64_t>(x));   // chi(-lambda) = e(k / L)
            for (int64_t m = 0; m <= static_cast<int64_t>(n); ++m)
                if (((4 * k + L * (static_cast<int64_t>(n) - 2 * m)) % (4 * L) + 4 * L) % (4 * L) == 0) ++sumS;
        }
        if ((x * x + 1 + NN - x) % NN == 0) {   // mu: mu^2 - mu + 1 = 0 (x < N)
            ++E.nu3;
            const int64_t k = (L - chi.exponent_at(static_cast<int64_t>(x)) % L) % L;      // chi(mu^{-1})
            for (int64_t m = 0; m <= static_cast<int64_t>(n); ++m)
                if (((6 * k + L * (static_cast<int64_t>(n) - 2 * m)) % (6 * L) + 6 * L) % (6 * L) == 0) ++sumU;
        }
    }
    const int64_t mm = static_cast<int64_t>(E.m);
    ENGINE_CHECK_MSG((mm - E.nu2) % 2 == 0 && (mm - E.nu3) % 3 == 0, "EulerCharacteristicFormula: orbit counts");
    E.dim_S = (mm - E.nu2) / 2 * d + sumS;
    E.dim_U = (mm - E.nu3) / 3 * d + sumU;
    E.h1_minus_h0 = mm * d - E.dim_S - E.dim_U;
    return E;
}

// ============================================================================
// The character model as a cokernel over F_q
// ============================================================================
BlockScalar CharacterScalar(const CharacterValues& chi) {
    const CharacterValues c = chi;
    return [c](const Mat22& h) -> uint32_t {
        // chi(a_h) is multiplicative only on Delta_0(N): assert c_h = 0 mod N (S16 WP-10/11 repairs, audit finding 9).
        ENGINE_CHECK_MSG(c.N <= 1 || modi(h.c, c.N) == 0, "CharacterScalar: h not in Delta_0(N) (c_h != 0 mod N)");
        return c.at(h.a);
    };
}

BlockScalar TableScalar(uint64_t N, std::vector<uint32_t> table) {
    const uint64_t NN = std::max<uint64_t>(N, 1);
    ENGINE_CHECK_MSG(table.size() == NN, "TableScalar: the table has N entries");
    return [NN, table](const Mat22& h) -> uint32_t {
        ENGINE_CHECK_MSG(NN == 1 || modi(h.c, NN) == 0, "TableScalar: h not in Delta_0(N) (c_h != 0 mod N)");   // audit finding 9
        const uint64_t a = modi(h.a, NN);
        ENGINE_CHECK_MSG(NN == 1 || gcd<uint64_t>(a, NN) == 1, "TableScalar: a_h is not a unit mod N (h not in Delta_0(N))");
        return table[a];
    };
}

TwistedMonomial TwistedMonomialOf(const level::CosetLevel& A, const Mat22& gamma, uint32_t n_sym, const Fq& F,
                                  const BlockScalar& s) {
    ENGINE_CHECK_MSG(gamma.det() == 1, "TwistedMonomialOf: gamma in SL_2(Z)");
    TwistedMonomial M;
    M.m = A.m();
    M.d = n_sym + 1;
    M.perm.assign(M.m + 1, 0);
    M.scal.assign(M.m + 1, 0);
    M.blk.assign(M.m + 1, 0);
    RhoFl rho(F.ell(), n_sym);
    for (uint32_t j = 1; j <= M.m; ++j) {
        const Mat22 x = A.rep(j) * gamma;
        const uint32_t jj = A.class_of(x);
        const Mat22 h = x * A.rep(jj).inv_unimodular();
        ENGINE_CHECK_MSG(A.contains(h), "TwistedMonomialOf: h_j(gamma) not in the subgroup (lem:inducedchar)");
        M.perm[j] = jj;
        M.scal[j] = s(h);
        M.blk[j] = rho.index_of(h, M.blocks);
    }
    return M;
}

std::vector<uint32_t> TwistedMonomial::apply(const Fq& F, const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == static_cast<size_t>(m) * d, "TwistedMonomial::apply: operand length");
    std::vector<uint32_t> y(x.size(), 0);
    for (uint32_t j = 1; j <= m; ++j) {
        const std::vector<uint32_t>& B = blocks[blk[j]];
        const uint32_t* xs = &x[static_cast<size_t>(perm[j] - 1) * d];
        for (uint32_t r = 0; r < d; ++r) {
            uint32_t acc = 0;
            for (uint32_t c = 0; c < d; ++c)
                if (B[static_cast<size_t>(r) * d + c] && xs[c]) acc = F.add(acc, F.mul(B[static_cast<size_t>(r) * d + c], xs[c]));
            y[static_cast<size_t>(j - 1) * d + r] = F.mul(scal[j], acc);
        }
    }
    return y;
}

namespace {

// Insert v into a fully reduced row basis (rows keyed by their pivot); returns true if independent.
struct RowBasis {
    const Fq* F = nullptr;
    uint32_t D = 0;
    std::vector<std::vector<uint32_t>> rows;
    std::vector<uint32_t> piv;
    std::vector<int32_t> row_of;   // column -> row or -1
    void init(const Fq& f, uint32_t dim) { F = &f; D = dim; row_of.assign(dim, -1); }
    void reduce(std::vector<uint32_t>& v) const {
        for (size_t r = 0; r < rows.size(); ++r) {
            const uint32_t c = v[piv[r]];
            if (c == 0) continue;
            const uint32_t f = F->neg(c);
            const std::vector<uint32_t>& row = rows[r];
            for (uint32_t k = 0; k < D; ++k)
                if (row[k]) v[k] = F->add(v[k], F->mul(f, row[k]));
        }
    }
    bool insert(std::vector<uint32_t> v) {
        reduce(v);
        uint32_t p = 0;
        while (p < D && v[p] == 0) ++p;
        if (p == D) return false;
        const uint32_t iv = F->inv(v[p]);
        for (uint32_t& x : v) x = F->mul(x, iv);
        // keep the basis fully reduced: clear column p in the other rows.
        for (auto& row : rows) {
            const uint32_t c = row[p];
            if (c == 0) continue;
            const uint32_t f = F->neg(c);
            for (uint32_t k = 0; k < D; ++k)
                if (v[k]) row[k] = F->add(row[k], F->mul(f, v[k]));
        }
        row_of[p] = static_cast<int32_t>(rows.size());
        rows.push_back(std::move(v));
        piv.push_back(p);
        return true;
    }
};

// The spanning vectors of M^<gamma> orbit by orbit (prop:neborbit (1)): for an orbit
// representative j0 and every r, (1/o) sum_{i<o} M^i e_{j0,r} (supported on the orbit).
// Certifies first that gamma^o acts as the identity (the orbit's block product).
std::vector<std::vector<uint32_t>> OrbitFixedVectors(const TwistedMonomial& M, uint32_t order, const Fq& F, uint32_t& orbits) {
    const uint32_t m = M.m, d = M.d, D = m * d;
    std::vector<uint32_t> inv_perm(m + 1, 0);
    for (uint32_t j = 1; j <= m; ++j) inv_perm[M.perm[j]] = j;
    auto blockmul = [&](uint32_t j, const std::vector<uint32_t>& v) {   // s_j B_j v
        std::vector<uint32_t> w(d, 0);
        const std::vector<uint32_t>& B = M.blocks[M.blk[j]];
        for (uint32_t r = 0; r < d; ++r) {
            uint32_t acc = 0;
            for (uint32_t c = 0; c < d; ++c)
                if (B[static_cast<size_t>(r) * d + c] && v[c]) acc = F.add(acc, F.mul(B[static_cast<size_t>(r) * d + c], v[c]));
            w[r] = F.mul(M.scal[j], acc);
        }
        return w;
    };
    // gamma^o = id: perm^o = id and B_j B_{perm j} ... B_{perm^{o-1} j} = Id (as the composite of apply).
    for (uint32_t j = 1; j <= m; ++j) {
        uint32_t k = j;
        for (uint32_t i = 0; i < order; ++i) k = M.perm[k];
        ENGINE_CHECK_MSG(k == j, "OrbitFixedVectors: gamma^o does not fix the coset");
        for (uint32_t r = 0; r < d; ++r) {
            // (M^o e_{j, r}) lives on the block perm^{-o} j = j.
            std::vector<uint32_t> v(d, 0);
            v[r] = 1;
            uint32_t b = j;
            for (uint32_t i = 0; i < order; ++i) { b = inv_perm[b]; v = blockmul(b, v); }
            for (uint32_t c = 0; c < d; ++c)
                ENGINE_CHECK_MSG(v[c] == (c == r ? 1u : 0u),
                                 "OrbitFixedVectors: gamma^o acts non-trivially (-I acts by chi(-1)(-1)^n != 1)");
        }
    }
    const uint32_t inv_o = F.inv(F.from_int(order));
    std::vector<uint8_t> seen(m + 1, 0);
    std::vector<std::vector<uint32_t>> out;
    orbits = 0;
    for (uint32_t j0 = 1; j0 <= m; ++j0) {
        if (seen[j0]) continue;
        ++orbits;
        uint32_t k = j0;
        for (uint32_t i = 0; i < order; ++i) { seen[k] = 1; k = M.perm[k]; }
        for (uint32_t r = 0; r < d; ++r) {
            std::vector<uint32_t> f(D, 0);
            std::vector<uint32_t> v(d, 0);
            v[r] = 1;
            uint32_t b = j0;
            for (uint32_t i = 0; i < order; ++i) {
                for (uint32_t c = 0; c < d; ++c) {
                    uint32_t& o = f[static_cast<size_t>(b - 1) * d + c];
                    o = F.add(o, F.mul(inv_o, v[c]));
                }
                b = inv_perm[b];
                v = blockmul(b, v);
            }
            out.push_back(std::move(f));
        }
    }
    return out;
}

} // namespace

NebCokernel::NebCokernel(const level::Gamma0LevelN& lvl, uint32_t n, const Fq& F, const CharacterValues& chi) : F_(&F) {
    if (F.ell() < 5)
        throw NebentypusRefusal("nebentypus_small_ell", "ell in {2, 3}: no cokernel and no comparison with group cohomology");
    ENGINE_CHECK_MSG(chi.N == std::max<uint64_t>(lvl.N(), 1), "NebCokernel: chi has another modulus");
    // D in 64 bits before the cap (S16 WP-10/11 repairs, audit finding 6: the uint32 product could wrap below the cap).
    const uint64_t D64 = static_cast<uint64_t>(lvl.m()) * (static_cast<uint64_t>(n) + 1);
    if (D64 > kNebDenseCap)
        throw NebentypusRefusal("neb_dense_cap", "D = m_N (n + 1) = " + std::to_string(D64) + " > " +
                                                     std::to_string(kNebDenseCap) + " (dense elimination only)");
    D_ = static_cast<uint32_t>(D64);
    const Fq::E chim1 = lvl.N() <= 2 ? 1u : chi.at(static_cast<int64_t>(lvl.N()) - 1);
    const Fq::E eps = (n % 2 == 0) ? chim1 : F.neg(chim1);
    if (eps != 1) {
        ENGINE_CHECK_MSG(eps == F.neg(1), "NebCokernel: chi(-1)^2 != 1");
        parity_zero_ = true;   // prop:neborbit (2): H^* = 0
        return;
    }
    const BlockScalar s = CharacterScalar(chi);
    const TwistedMonomial MS = TwistedMonomialOf(lvl, Mat22::S(), n, F, s);
    const TwistedMonomial MU = TwistedMonomialOf(lvl, Mat22::ST(), n, F, s);
    const std::vector<std::vector<uint32_t>> vS = OrbitFixedVectors(MS, 2, F, orbits_S_);
    const std::vector<std::vector<uint32_t>> vU = OrbitFixedVectors(MU, 3, F, orbits_U_);
    {
        RowBasis b;
        b.init(F, D_);
        for (const auto& v : vS) dim_S_ += b.insert(v) ? 1 : 0;
    }
    {
        RowBasis b;
        b.init(F, D_);
        for (const auto& v : vU) dim_U_ += b.insert(v) ? 1 : 0;
    }
    RowBasis b;
    b.init(F, D_);
    for (const auto& v : vS) b.insert(v);
    for (const auto& v : vU) b.insert(v);
    rref_ = std::move(b.rows);
    pivots_ = std::move(b.piv);
    std::vector<uint8_t> is_piv(D_, 0);
    for (uint32_t p : pivots_) is_piv[p] = 1;
    for (uint32_t c = 0; c < D_; ++c) if (!is_piv[c]) free_.push_back(c);
}

std::vector<uint32_t> NebCokernel::coords(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(!parity_zero_ && x.size() == D_, "NebCokernel::coords: operand");
    std::vector<uint32_t> w = x;
    for (size_t r = 0; r < rref_.size(); ++r) {
        const uint32_t c = w[pivots_[r]];
        if (c == 0) continue;
        const uint32_t f = F_->neg(c);
        for (uint32_t k = 0; k < D_; ++k)
            if (rref_[r][k]) w[k] = F_->add(w[k], F_->mul(f, rref_[r][k]));
    }
    std::vector<uint32_t> out(free_.size());
    for (size_t i = 0; i < free_.size(); ++i) out[i] = w[free_[i]];
    return out;
}

bool NebCokernel::in_rel(const std::vector<uint32_t>& x) const {
    const std::vector<uint32_t> c = coords(x);
    return std::all_of(c.begin(), c.end(), [](uint32_t v) { return v == 0; });
}

uint32_t NebCokernel::RelEscapes(const HeilbronnChi& H) const {
    ENGINE_CHECK_MSG(H.D() == D_, "NebCokernel::RelEscapes: fibre ranks differ");
    uint32_t bad = 0;
    for (const auto& r : rref_) bad += in_rel(H.apply(r)) ? 0 : 1;
    return bad;
}

FqMat NebCokernel::OnH1(const HeilbronnChi& H) const {
    ENGINE_CHECK_MSG(!parity_zero_, "NebCokernel::OnH1: H^1 = 0 (parity)");
    ENGINE_CHECK_MSG(RelEscapes(H) == 0, "NebCokernel::OnH1: H_{N,p,chi}(Rel) is not in Rel (rmk:nebheilbronn)");
    const uint32_t h = h1();
    FqMat M(h, h);
    std::vector<uint32_t> e(D_, 0);
    for (uint32_t j = 0; j < h; ++j) {
        e[free_[j]] = 1;
        const std::vector<uint32_t> c = coords(H.apply(e));
        e[free_[j]] = 0;
        for (uint32_t i = 0; i < h; ++i) M.at(i, j) = c[i];
    }
    return M;
}

HeilbronnChi::HeilbronnChi(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t n, const Fq& F, const CharacterValues& chi)
    : HeilbronnChi(lvl, p, n, F, chi, false) {}

HeilbronnChi HeilbronnChi::Central(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t n, const Fq& F, const CharacterValues& chi) {
    return HeilbronnChi(lvl, p, n, F, chi, true);
}

HeilbronnChi HeilbronnChi::Power(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t nu, uint32_t n, const Fq& F,
                                 const CharacterValues& chi) {
    return HeilbronnChi(lvl, p, n, F, chi, false, nu);
}

HeilbronnChi::HeilbronnChi(const level::Gamma0LevelN& lvl, uint64_t p, uint32_t n, const Fq& F, const CharacterValues& chi,
                           bool central, uint32_t nu)
    : F_(&F), N_(lvl.N()), p_(p), nu_(nu), central_(central), m_(lvl.m()), d_(n + 1) {
    ENGINE_CHECK_MSG(p >= 2 && p < (1ull << 20) && hecke::IsPrimeU64(p), "HeilbronnChi: p must be a prime < 2^20");
    ENGINE_CHECK_MSG(N_ < (1ull << 20), "HeilbronnChi: N < 2^20");
    ENGINE_CHECK_MSG(chi.N == std::max<uint64_t>(N_, 1), "HeilbronnChi: chi has another modulus");
    ENGINE_CHECK_MSG(!central || N_ % p != 0, "HeilbronnChi::Central: T(p, p) needs p !| N");
    ENGINE_CHECK_MSG(nu >= 1 && (!central || nu == 1), "HeilbronnChi: nu >= 1 (nu = 1 for T(p, p))");
    uint64_t pnu = 1;
    for (uint32_t i = 0; i < nu; ++i) {
        pnu *= p;
        ENGINE_CHECK_MSG(pnu < (1ull << 20), "HeilbronnChi::Power: p^nu < 2^20 (MerelSet's domain)");
    }
    const int64_t det_want = central ? static_cast<int64_t>(p * p) : static_cast<int64_t>(pnu);
    const std::vector<Mat22> mer = central ? std::vector<Mat22>{Mat22{static_cast<int64_t>(p), 0, 0, static_cast<int64_t>(p)}}
                                           : level::MerelSet(pnu);
    RhoFl rho(F.ell(), n);
    const bool p_divides_N = N_ % p_ == 0;
    for (const Mat22& eps : mer) {
        const Mat22 eps_adj = eps.adj();
        for (uint32_t j = 1; j <= m_; ++j) {
            if (!lvl.act_defined(j, eps)) {
                ENGINE_CHECK_MSG(p_divides_N, "HeilbronnChi: an inadmissible pair at p !| N (prop:heilbronn)");
                ++dropped_;
                continue;
            }
            const uint32_t i = lvl.act(j, eps);
            Term t;
            t.dst = i;
            t.src = j;
            // (slot-17 WP-6) attempt-then-escalate; level::ScopedForceWideHeilbronn (validation only) forces the wide route.
            auto wide = [&]() {
                // (slot-17 WP-6) the __int128 route of HeilbronnOperator (slot-16 WP-13 F-2), attempt-then-escalate: the
                // int64 triple product overflows at composite N with a case-(iii) representative (first met at
                // N = 30030).  The same products and checks in Mat22W; chi reads a_{adj h} mod N and rho_n the residues
                // mod ell -- exactly the keys of the int64 route, so the term is the one it would have produced.
                // (slot-18 WP-5 (d)) the product of the shared wide route: Gamma0LevelN::cocycle_wide(j, eps) =
                // w_j eps w_{j.eps}^{-1} in __int128 -- the twin of cocycle() used by the int64 attempt below.
                const Mat22W hw = lvl.cocycle_wide(j, eps);
                const Mat22W adj_w{hw.d, -hw.b, -hw.c, hw.a};
                ENGINE_CHECK_MSG(adj_w == Mat22W::from(lvl.rep(i)) * Mat22W::from(eps_adj) * Mat22W::from(lvl.rep(j).inv_unimodular()),
                     "HeilbronnChi: adj h_j(eps) != w_{j.eps} adj(eps) w_j^{-1} (int128)");
                const __int128 NN = static_cast<__int128>(std::max<uint64_t>(N_, 1));
                ENGINE_CHECK_MSG(adj_w.det() == static_cast<__int128>(det_want) && adj_w.c % NN == 0,
                     "HeilbronnChi: adj h_j(eps) not in Delta_0(N) of the determinant (int128)");
                __int128 aN = adj_w.a % NN;
                if (aN < 0) aN += NN;
                t.scal = chi.at(static_cast<int64_t>(aN));
                const std::array<uint32_t, 4> r = adj_w.reduced_mod(F.ell());
                t.block = rho.index_of(Mat22{r[0], r[1], r[2], r[3]}, blocks_);
                ++wide_terms_;
            };
            if (level::ScopedForceWideHeilbronn::active()) {
                wide();
            } else try {
                const Mat22 h = lvl.cocycle(j, eps);   // w_j eps w_i^{-1}
                const Mat22 adj_h = h.adj();
                ENGINE_CHECK_MSG(adj_h == lvl.rep(i) * eps_adj * lvl.rep(j).inv_unimodular(),
                                 "HeilbronnChi: adj h_j(eps) != w_{j.eps} adj(eps) w_j^{-1}");
                ENGINE_CHECK_MSG(adj_h.det() == det_want && modi(adj_h.c, std::max<uint64_t>(N_, 1)) == 0,
                                 "HeilbronnChi: adj h_j(eps) not in Delta_0(N) of det p (p^2 for T(p, p))");
                // rmk:nebheilbronn: chi(d_{h_j(eps)}) = chi(a_{adj h}); d_h is a unit modulo N.
                t.scal = chi.at(adj_h.a);
                t.block = rho.index_of(adj_h, blocks_);
            } catch (const HeckeOverflow&) {
                wide();
            }
            terms_.push_back(t);
        }
    }
    if (!p_divides_N) ENGINE_CHECK_MSG(dropped_ == 0, "HeilbronnChi: p !| N but a pair was dropped");
}

std::vector<uint32_t> HeilbronnChi::apply(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == static_cast<size_t>(D()), "HeilbronnChi::apply: operand length");
    const Fq& F = *F_;
    std::vector<uint32_t> y(x.size(), 0);
    for (const Term& t : terms_) {
        const std::vector<uint32_t>& B = blocks_[t.block];
        const uint32_t* xs = &x[static_cast<size_t>(t.src - 1) * d_];
        uint32_t* ys = &y[static_cast<size_t>(t.dst - 1) * d_];
        for (uint32_t r = 0; r < d_; ++r) {
            uint32_t acc = 0;
            for (uint32_t c = 0; c < d_; ++c)
                if (B[static_cast<size_t>(r) * d_ + c] && xs[c]) acc = F.add(acc, F.mul(B[static_cast<size_t>(r) * d_ + c], xs[c]));
            if (acc) ys[r] = F.add(ys[r], F.mul(t.scal, acc));
        }
    }
    return y;
}

// ============================================================================
// The character model on the fixed model
// ============================================================================
std::vector<LocalSystemSimplicialComplex::SimplexRelation> InducedRelationsChi(
    const SimplicialComplex& K, const level::CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
    const BlockScalar& s) {
    using Rel = LocalSystemSimplicialComplex::SimplexRelation;
    const uint32_t m = A.m(), d = n_sym + 1, D = m * d;
    const uint32_t base_prime = ctx.is_field() ? ctx.p : static_cast<uint32_t>(ctx.ell);
    const std::vector<SL2ZRelationDatum> base = GenerateSL2ZRelationsWithGroupData(K, n_sym, base_prime);
    std::map<std::array<int64_t, 4>, FlatMatrixT<uint32_t>> by_gen;
    std::vector<Rel> out;
    for (const SL2ZRelationDatum& dd : base) {
        const std::array<int64_t, 4> key{dd.g_r.a, dd.g_r.b, dd.g_r.c, dd.g_r.d};
        auto it = by_gen.find(key);
        if (it == by_gen.end()) {
            const level::MonomialAction M = level::MonomialOf(A, dd.g_r, n_sym, ctx);
            std::vector<uint32_t> buf(static_cast<size_t>(D) * D, 0);
            for (uint32_t j = 1; j <= m; ++j) {
                const Mat22 x = A.rep(j) * dd.g_r;
                const Mat22 h = x * A.rep(M.perm[j]).inv_unimodular();
                const uint64_t sc = s(h) % ctx.p;
                const level::Block& B = M.blocks[M.blk[j]];
                for (uint32_t r = 0; r < d; ++r)
                    for (uint32_t c = 0; c < d; ++c)
                        buf[static_cast<size_t>((j - 1) * d + r) * D + (M.perm[j] - 1) * d + c] =
                            static_cast<uint32_t>(sc * B.dense[static_cast<size_t>(r) * d + c] % ctx.p);
            }
            it = by_gen.emplace(key, FlatMatrixT<uint32_t>(D, D, ctx.p, buf)).first;
        }
        out.push_back(Rel(dd.rel.q_dim, dd.rel.s_lhs_idx_in_Kq, dd.rel.s_rhs_idx_in_Kq, it->second, dd.rel.tag));
    }
    return out;
}

level::BlockCochainMap BuildLevelCorrespondenceChi(const engine::ModCtx<uint32_t>& ctx, const level::CorrespondenceSpec& spec,
                                                   const SimplicialComplex& K, uint32_t n_sym, uint32_t q,
                                                   const BlockScalar& s) {
    // eq:levelcorr exactly as level::BuildLevelCorrespondence (HeckeLevelEll.cpp), with
    // eq:nebcorr's scalar s(E) on every block term.
    ENGINE_CHECK_MSG(spec.A && spec.B, "BuildLevelCorrespondenceChi: A and B must be set");
    ENGINE_CHECK_MSG((spec.g_cosets == nullptr) == (spec.tau == nullptr),
                     "BuildLevelCorrespondenceChi: the level-1 datum is (cosets, tau) or the identity");
    const bool identity = spec.tau == nullptr;
    if (!identity) ENGINE_CHECK_MSG(spec.tau->n_sym() == n_sym, "BuildLevelCorrespondenceChi: tau's n_sym differs");
    ENGINE_CHECK_MSG(q < 3, "BuildLevelCorrespondenceChi: q in {0, 1, 2}");
    level::BlockCochainMap F;
    F.ctx = ctx;
    F.q = q;
    F.N_q = K.get_num_q_simplices(q);
    F.d = n_sym + 1;
    F.m_src = spec.A->m();
    F.m_dst = spec.B->m();
    level::RhoCache rho(ctx, n_sym);
    const Mat22 g = spec.g();
    const int64_t detg = g.det();
    const std::vector<ChainTerm> id_chain_proto{ChainTerm{1, Mat22::I(), 0}};
    std::vector<Mat22> w_inv(F.m_src + 1);
    for (uint32_t a = 1; a <= F.m_src; ++a) w_inv[a] = spec.A->rep(a).inv_unimodular();
    for (uint32_t i = 1; i <= F.m_dst; ++i) {
        const Mat22 v_i = spec.B->rep(i);
        for (const Mat22& r_t : spec.r) {
            const Mat22 x = r_t * v_i;
            uint32_t j = 1;
            Mat22 h = x, delta1 = Mat22::I();
            if (!identity) {
                j = spec.g_cosets->class_index_of(x);
                h = x * spec.g_cosets->s_inv(j);
                ENGINE_CHECK_MSG(h.det() == 1 && (h.c % static_cast<int64_t>(spec.g_cosets->N())) == 0,
                                 "BuildLevelCorrespondenceChi: r_t v_i s_j^{-1} is not in Gamma''_g");
                delta1 = spec.g_cosets->delta(j);
            }
            Mat22 k;
            ENGINE_CHECK_MSG(level::ConjugateByInverse(g, h, k) && k.det() == 1,
                             "BuildLevelCorrespondenceChi: g^{-1} h g is not in Gamma");
            const uint32_t a = spec.A->class_of(k);
            ENGINE_CHECK_MSG(spec.A->contains(k * w_inv[a]), "BuildLevelCorrespondenceChi: k w_a^{-1} not in A");
            const Mat22 vd = v_i * delta1;
            for (uint32_t sigma = 0; sigma < F.N_q; ++sigma) {
                std::vector<ChainTerm> id_chain;
                if (identity) { id_chain = id_chain_proto; id_chain[0].sigma_idx = sigma; }
                const std::vector<ChainTerm>& chain = identity ? id_chain : spec.tau->tau(j, q, sigma);
                const Mat22 wa = spec.A->rep(a);
                for (const ChainTerm& ct : chain) {
                    const uint32_t c0 = CoeffMod(ct.coeff, ctx.p);
                    if (c0 == 0) continue;
                    const uint32_t a2 = spec.A->class_of(wa * ct.gamma);
                    const Mat22 E = vd * ct.gamma * w_inv[a2];
                    ENGINE_CHECK_MSG(E.det() == detg, "BuildLevelCorrespondenceChi: a block matrix has det != det g");
                    const uint32_t c = static_cast<uint32_t>(static_cast<uint64_t>(c0) * (s(E) % ctx.p) % ctx.p);
                    if (c == 0) continue;
                    F.terms.push_back(level::BlockTerm{sigma, i, ct.sigma_idx, a2, c, rho.block_of(E, F.blocks)});
                }
            }
        }
    }
    return F;
}

// ============================================================================
// The Gamma_1(N) model
// ============================================================================
Gamma1LevelN::Gamma1LevelN(uint64_t N) : N_(N), lvl0_(N) {
    ENGINE_CHECK_MSG(N >= 3 && N < (1ull << 20), "Gamma1LevelN: N in [3, 2^20) (N <= 2: Gamma_1 = Gamma_0)");
    units_ = UnitsMod(N);
    // m = m_0(N) phi(N) must fit m()'s uint32 (S16 WP-10/11 repairs, audit finding 6); refused before the O(m) table check.
    ENGINE_CHECK_MSG(static_cast<uint64_t>(lvl0_.m()) * units_.size() < (1ull << 32),
                     "Gamma1LevelN: m_0(N) phi(N) >= 2^32 (the index does not fit uint32)");
    rank_.assign(N, UINT32_MAX);
    for (size_t i = 0; i < units_.size(); ++i) rank_[units_[i]] = static_cast<uint32_t>(i);
    for (uint32_t J = 1; J <= m(); ++J) {
        ENGINE_CHECK_MSG(rep(J).det() == 1 && class_of(rep(J)) == J, "Gamma1LevelN: the table is not self-consistent");
    }
    ENGINE_CHECK_MSG(rep(1) == Mat22::I(), "Gamma1LevelN: w_(1,1) != I");
}

Mat22 Gamma1LevelN::sigma(uint64_t u) const {
    u %= N_;
    ENGINE_CHECK_MSG(gcd<uint64_t>(u, N_) == 1, "Gamma1LevelN::sigma: u not a unit");
    if (u == 1) return Mat22::I();
    const int64_t up = static_cast<int64_t>(u);
    const int64_t alpha = static_cast<int64_t>(InvModN(u, N_));
    const int64_t beta = (alpha * up - 1) / static_cast<int64_t>(N_);
    const Mat22 s{alpha, beta, static_cast<int64_t>(N_), up};
    ENGINE_CHECK_MSG(s.det() == 1, "Gamma1LevelN::sigma: det != 1");
    return s;
}

uint32_t Gamma1LevelN::index(uint32_t j, uint64_t u) const {
    const uint32_t r = rank_[u % N_];
    ENGINE_CHECK_MSG(r != UINT32_MAX && j >= 1 && j <= lvl0_.m(), "Gamma1LevelN::index: out of range");
    return (j - 1) * static_cast<uint32_t>(units_.size()) + r + 1;
}

Mat22 Gamma1LevelN::rep(uint32_t J) const {
    ENGINE_CHECK_MSG(J >= 1 && J <= m(), "Gamma1LevelN::rep: out of range");
    const uint32_t phi = static_cast<uint32_t>(units_.size());
    const uint32_t j = (J - 1) / phi + 1;
    const uint64_t u = units_[(J - 1) % phi];
    return sigma(u) * lvl0_.rep(j);
}

uint32_t Gamma1LevelN::class_of(const Mat22& gamma) const {
    const uint32_t j = lvl0_.class_of(gamma);
    const Mat22 h = gamma * lvl0_.rep(j).inv_unimodular();   // in Gamma_0(N)
    ENGINE_CHECK_MSG(lvl0_.contains(h), "Gamma1LevelN::class_of: gamma w_j^{-1} not in Gamma_0(N)");
    return index(j, modi(h.d, N_));
}

bool Gamma1LevelN::contains(const Mat22& g) const {
    return g.det() == 1 && modi(g.c, N_) == 0 && modi(g.d, N_) == 1;
}

bool Gamma1LevelN::contains_wide(const Mat22W& g) const {
    const __int128 NN = static_cast<__int128>(N_);
    __int128 dN = g.d % NN;
    if (dN < 0) dN += NN;
    return g.det() == 1 && g.c % NN == 0 && dN == (NN == 1 ? 0 : 1);
}

std::vector<Mat22> Gamma1SubTransversal(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 1 && hecke::IsPrimeU64(p), "Gamma1SubTransversal: p prime");
    std::vector<Mat22> r;
    for (int64_t t = 0; t < static_cast<int64_t>(p); ++t) r.push_back(Mat22{1, 0, static_cast<int64_t>(N) * t, 1});
    if (N % p != 0) {
        // d0 = p k, k = p^{-1} mod N (so d0 = 1 mod N, p | d0), then x d0 + y N = 1.
        const uint64_t k = N == 1 ? 1 : InvModN(p % N, N);
        const int64_t d0 = static_cast<int64_t>(p * k);
        const int64_t NN = static_cast<int64_t>(N);
        // extended Euclid on (d0, N)
        int64_t old_r = d0, rr = NN, old_s = 1, ss = 0, old_t = 0, tt = 1;
        while (rr) {
            const int64_t q = old_r / rr;
            int64_t tmp = old_r - q * rr; old_r = rr; rr = tmp;
            tmp = old_s - q * ss; old_s = ss; ss = tmp;
            tmp = old_t - q * tt; old_t = tt; tt = tmp;
        }
        ENGINE_CHECK_MSG(old_r == 1, "Gamma1SubTransversal: gcd(d0, N) != 1");
        // old_s d0 + old_t N = 1 -> [[old_s, -old_t], [N, d0]] has det old_s d0 + old_t N = 1.
        const Mat22 x{old_s, -old_t, NN, d0};
        ENGINE_CHECK_MSG(x.det() == 1, "Gamma1SubTransversal: det != 1");
        r.push_back(x);
    }
    return r;
}

level::BlockCochainMap DiamondOnModel(const Gamma1LevelN& G, uint64_t dd, const SimplicialComplex& K, uint32_t n_sym,
                                      const engine::ModCtx<uint32_t>& ctx, uint32_t q) {
    const uint64_t N = G.N();
    ENGINE_CHECK_MSG(gcd<uint64_t>(dd % N, N) == 1, "DiamondOnModel: d must be a unit mod N");
    level::BlockCochainMap F;
    F.ctx = ctx;
    F.q = q;
    F.N_q = K.get_num_q_simplices(q);
    F.d = n_sym + 1;
    F.m_src = F.m_dst = G.m();
    level::RhoCache rho(ctx, n_sym);
    const uint64_t dinv = InvModN(dd % N, N);
    const uint32_t phi = static_cast<uint32_t>(G.units().size());
    // (R f)_{(j,u)} = rho(sigma_u sigma_{u'}^{-1}) f_{(j,u')}, u' = d^{-1} u.
    std::vector<uint32_t> src(G.m() + 1), blk(G.m() + 1);
    std::vector<Mat22> R(G.m() + 1);
    for (uint32_t J = 1; J <= G.m(); ++J) {
        const uint32_t j = (J - 1) / phi + 1;
        const uint64_t u = G.units()[(J - 1) % phi];
        const uint64_t u2 = mulmod(dinv, u, N);
        src[J] = G.index(j, u2);
        R[J] = G.sigma(u) * G.sigma(u2).inv_unimodular();
        ENGINE_CHECK_MSG(G.level0().contains(R[J]) && modi(R[J].d, N) == dd % N,
                         "DiamondOnModel: sigma_u sigma_{u'}^{-1} is not in Gamma_0(N) with d = d");
        blk[J] = rho.block_of(R[J], F.blocks);
    }
    // R is a Gamma-module map: it commutes with the monomial action of S and T (checked exactly on
    // the matrices: h_J(gamma) R_{J.gamma} = R_J h_{rho J}(gamma) with the same target coset).
    for (const Mat22& gam : {Mat22::S(), Mat22::T()}) {
        for (uint32_t J = 1; J <= G.m(); ++J) {
            const Mat22 x = G.rep(J) * gam;
            const uint32_t Jg = G.class_of(x);
            const Mat22 hJ = x * G.rep(Jg).inv_unimodular();
            const Mat22 lhs = hJ * R[Jg];                         // (M R)_J, source src[Jg]
            const uint32_t Js = src[J];
            const Mat22 y = G.rep(Js) * gam;
            const uint32_t Jsg = G.class_of(y);
            const Mat22 rhs = R[J] * (y * G.rep(Jsg).inv_unimodular());   // (R M)_J, source Jsg
            ENGINE_CHECK_MSG(src[Jg] == Jsg && lhs == rhs, "DiamondOnModel: <d> does not commute with the Gamma-action");
        }
    }
    for (uint32_t sigma = 0; sigma < F.N_q; ++sigma)
        for (uint32_t J = 1; J <= G.m(); ++J)
            F.terms.push_back(level::BlockTerm{sigma, J, sigma, src[J], 1, blk[J]});
    return F;
}

} // namespace neb
} // namespace hecke
