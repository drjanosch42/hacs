// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P3) the MTT measure over a coefficient ring O = (Z/ell^R)[x]/(F) -- see CoefficientRingMeasure.h.
// Spec docs/implementation/slot21/P3_spec.md; the Python oracle of record docs/implementation/slot21/p3/p3lib.py (an independent
// implementation: nothing below was ported from it beyond the conventions of the spec).
#include "CoefficientRingMeasure.h"

#include "Bernoulli.h"
#include "MBlockReader.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "../../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {
namespace lvalue {
namespace oring {

namespace {
using u128 = unsigned __int128;
using Elt = ORing::Elt;

using engine::scalar::pow_wrap;

// ---- polynomials over F_p (low -> high), small degree --------------------------------------------------------------------
Vec Trim(Vec a) {
    while (!a.empty() && a.back() == 0) a.pop_back();
    return a;
}
Vec ModP(const Vec& a, uint64_t p) {
    Vec o(a.size());
    for (size_t i = 0; i < a.size(); ++i) o[i] = a[i] % p;
    return Trim(o);
}
uint64_t InvModP(uint64_t a, uint64_t p) {
    const birch::Ring F = birch::MakeRing(p, 1);
    return F.inv(a % p);
}
// a = q b + r over F_p (b non-zero, trimmed)
void DivModP(Vec a, const Vec& b, uint64_t p, Vec& q, Vec& r) {
    a = ModP(a, p);
    const Vec bb = ModP(b, p);
    ENGINE_CHECK_MSG(!bb.empty(), "oring: division by the zero polynomial");
    const uint64_t ib = InvModP(bb.back(), p);
    q.assign(a.size() >= bb.size() ? a.size() - bb.size() + 1 : 1, 0);
    while (a.size() >= bb.size() && !a.empty()) {
        const uint64_t c = static_cast<uint64_t>(static_cast<u128>(a.back()) * ib % p);
        const size_t s = a.size() - bb.size();
        q[s] = c;
        for (size_t i = 0; i < bb.size(); ++i) a[s + i] = (a[s + i] + p - static_cast<uint64_t>(static_cast<u128>(c) * bb[i] % p)) % p;
        a = Trim(a);
    }
    r = a;
}
Vec MulP(const Vec& a, const Vec& b, uint64_t p) {
    if (a.empty() || b.empty()) return {};
    Vec o(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (size_t j = 0; j < b.size(); ++j) o[i + j] = (o[i + j] + static_cast<uint64_t>(static_cast<u128>(a[i]) * b[j] % p)) % p;
    return Trim(o);
}
Vec SubP(const Vec& a, const Vec& b, uint64_t p) {
    Vec o(std::max(a.size(), b.size()), 0);
    for (size_t i = 0; i < o.size(); ++i) o[i] = ((i < a.size() ? a[i] : 0) + p - (i < b.size() ? b[i] % p : 0)) % p;
    return Trim(o);
}
Vec MonicGcdP(Vec a, Vec b, uint64_t p) {
    a = ModP(a, p);
    b = ModP(b, p);
    while (!b.empty()) {
        Vec q, r;
        DivModP(a, b, p, q, r);
        a = b;
        b = r;
    }
    if (!a.empty()) {
        const uint64_t iv = InvModP(a.back(), p);
        for (auto& c : a) c = static_cast<uint64_t>(static_cast<u128>(c) * iv % p);
    }
    return a;
}
Vec PowModP(const Vec& base, uint64_t e, const Vec& m, uint64_t p) {
    Vec r{1 % p}, b, q;
    DivModP(base, m, p, q, b);
    while (e) {
        if (e & 1) {
            Vec t = MulP(r, b, p), rr;
            DivModP(t, m, p, q, rr);
            r = rr;
        }
        Vec t = MulP(b, b, p), bb;
        DivModP(t, m, p, q, bb);
        b = bb;
        e >>= 1;
    }
    return r;
}
// X^{p^s} mod m
Vec FrobPow(uint32_t s, const Vec& m, uint64_t p) {
    Vec x{0, 1 % p};
    for (uint32_t i = 0; i < s; ++i) x = PowModP(x, p, m, p);
    return x;
}
// s, t with s a + t b = 1 over F_p
void XgcdP(const Vec& a, const Vec& b, uint64_t p, Vec& s, Vec& t) {
    Vec r0 = ModP(a, p), r1 = ModP(b, p), s0{1}, s1{}, t0{}, t1{1};
    while (!r1.empty()) {
        Vec q, r;
        DivModP(r0, r1, p, q, r);
        r0 = r1;
        r1 = r;
        Vec ns = SubP(s0, MulP(q, s1, p), p), nt = SubP(t0, MulP(q, t1, p), p);
        s0 = s1;
        s1 = ns;
        t0 = t1;
        t1 = nt;
    }
    ENGINE_CHECK_MSG(r0.size() == 1, "oring::XgcdP: not coprime");
    const uint64_t iv = InvModP(r0[0], p);
    s = s0;
    t = t0;
    for (auto& c : s) c = static_cast<uint64_t>(static_cast<u128>(c) * iv % p);
    for (auto& c : t) c = static_cast<uint64_t>(static_cast<u128>(c) * iv % p);
}

// ---- polynomials over Z/ell^R ----------------------------------------------------------------------------------------------
Vec MulR(const birch::Ring& R, const Vec& a, const Vec& b) {
    if (a.empty() || b.empty()) return {};
    Vec o(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (size_t j = 0; j < b.size(); ++j) o[i + j] = R.add(o[i + j], R.mul(a[i], b[j]));
    return o;
}

// a solution of M y = b over F_p (M square, invertible): Gauss-Jordan; false if singular
bool SolveModP(birch::Mat M, Vec b, uint64_t p, Vec& y) {
    const uint32_t n = M.r;
    for (auto& v : M.a) v %= p;
    for (auto& v : b) v %= p;
    for (uint32_t c = 0; c < n; ++c) {
        uint32_t piv = c;
        while (piv < n && M.at(piv, c) == 0) ++piv;
        if (piv == n) return false;
        if (piv != c) {
            for (uint32_t j = 0; j < n; ++j) std::swap(M.at(piv, j), M.at(c, j));
            std::swap(b[piv], b[c]);
        }
        const uint64_t iv = InvModP(M.at(c, c), p);
        for (uint32_t j = 0; j < n; ++j) M.at(c, j) = static_cast<uint64_t>(static_cast<u128>(M.at(c, j)) * iv % p);
        b[c] = static_cast<uint64_t>(static_cast<u128>(b[c]) * iv % p);
        for (uint32_t i = 0; i < n; ++i) {
            if (i == c || M.at(i, c) == 0) continue;
            const uint64_t t = M.at(i, c);
            for (uint32_t j = 0; j < n; ++j) M.at(i, j) = (M.at(i, j) + p - static_cast<uint64_t>(static_cast<u128>(t) * M.at(c, j) % p)) % p;
            b[i] = (b[i] + p - static_cast<uint64_t>(static_cast<u128>(t) * b[c] % p)) % p;
        }
    }
    y = b;
    return true;
}
} // namespace

// =====================================================================================================================
// ORing
// =====================================================================================================================
ORing::ORing(uint64_t ell, uint32_t R, const Vec& F) : Z_(birch::MakeRing(ell, R)) {
    ENGINE_CHECK_MSG(F.size() >= 2, "ORing: F of degree >= 1");
    F_.resize(F.size());
    for (size_t i = 0; i < F.size(); ++i) F_[i] = F[i] % Z_.q;
    ENGINE_CHECK_MSG(F_.back() == 1 % Z_.q, "ORing: F must be monic");
    d_ = static_cast<uint32_t>(F.size() - 1);
}
Elt ORing::scalar(uint64_t a) const {
    Elt o(d_, 0);
    o[0] = a % Z_.q;
    return o;
}
Elt ORing::x() const {
    if (d_ == 1) return scalar(Z_.neg(F_[0]));
    Elt o(d_, 0);
    o[1] = 1 % Z_.q;
    return o;
}
Elt ORing::add(const Elt& a, const Elt& b) const {
    Elt o(d_);
    for (uint32_t i = 0; i < d_; ++i) o[i] = Z_.add(a[i], b[i]);
    return o;
}
Elt ORing::sub(const Elt& a, const Elt& b) const {
    Elt o(d_);
    for (uint32_t i = 0; i < d_; ++i) o[i] = Z_.sub(a[i], b[i]);
    return o;
}
Elt ORing::neg(const Elt& a) const {
    Elt o(d_);
    for (uint32_t i = 0; i < d_; ++i) o[i] = Z_.neg(a[i]);
    return o;
}
Elt ORing::smul(uint64_t s, const Elt& a) const {
    Elt o(d_);
    s %= Z_.q;
    for (uint32_t i = 0; i < d_; ++i) o[i] = Z_.mul(s, a[i]);
    return o;
}
Elt ORing::mul(const Elt& a, const Elt& b) const {
    std::vector<uint64_t> t(2 * d_ - 1, 0);
    for (uint32_t i = 0; i < d_; ++i)
        if (a[i])
            for (uint32_t j = 0; j < d_; ++j) t[i + j] = Z_.add(t[i + j], Z_.mul(a[i], b[j]));
    for (uint32_t s = 2 * d_ - 1; s-- > d_;) {
        const uint64_t c = t[s];
        if (!c) continue;
        for (uint32_t i = 0; i <= d_; ++i) t[s - d_ + i] = Z_.sub(t[s - d_ + i], Z_.mul(c, F_[i]));
    }
    t.resize(d_);
    return t;
}
Elt ORing::pow(const Elt& a, uint64_t e) const {
    Elt r = one(), b = a;
    while (e) {
        if (e & 1) r = mul(r, b);
        b = mul(b, b);
        e >>= 1;
    }
    return r;
}
Elt ORing::eval(const Vec& P, const Elt& z) const {
    Elt acc = zero();
    for (size_t i = P.size(); i-- > 0;) acc = add(mul(acc, z), scalar(P[i]));
    return acc;
}
bool ORing::is_zero(const Elt& a) const {
    return std::all_of(a.begin(), a.end(), [](uint64_t v) { return v == 0; });
}
birch::Mat ORing::mulmat(const Elt& a) const {
    birch::Mat M(d_, d_);
    Elt col = a;
    const Elt xx = x();
    for (uint32_t j = 0; j < d_; ++j) {
        for (uint32_t i = 0; i < d_; ++i) M.at(i, j) = col[i];
        col = mul(col, xx);
    }
    return M;
}
bool ORing::unit(const Elt& a) const {
    const birch::Mat M = mulmat(a);
    Vec e1(d_, 0), y;
    e1[0] = 1;
    return SolveModP(M, e1, Z_.ell, y);
}
Elt ORing::inv(const Elt& a) const {
    Vec e1(d_, 0), y0;
    e1[0] = 1;
    ENGINE_CHECK_MSG(SolveModP(mulmat(a), e1, Z_.ell, y0), "ORing::inv: not a unit");
    Elt y(y0.begin(), y0.end());
    const Elt two = scalar(2);
    for (uint32_t it = 0; it <= Z_.m + 1; ++it) y = mul(y, sub(two, mul(a, y)));
    ENGINE_CHECK_MSG(mul(a, y) == one(), "ORing::inv: Newton did not converge");
    return y;
}
uint64_t ORing::trace(const Elt& a) const {
    const birch::Mat M = mulmat(a);
    uint64_t t = 0;
    for (uint32_t i = 0; i < d_; ++i) t = Z_.add(t, M.at(i, i));
    return t;
}
uint64_t ORing::norm(const Elt& a) const {
    const Vec cp = birch::CharPoly(Z_, mulmat(a));
    return (d_ % 2 == 0) ? cp[0] : Z_.neg(cp[0]);
}
Elt ORing::reduce_to(const ORing& T, const Elt& a) const {
    ENGINE_CHECK_MSG(T.d() == d_ && T.ell() == ell(), "ORing::reduce_to: rings differ");
    Elt o(d_);
    for (uint32_t i = 0; i < d_; ++i) o[i] = a[i] % T.q();
    return o;
}
std::string ORing::str(const Elt& a) const {
    std::ostringstream s;
    s << "[";
    for (uint32_t i = 0; i < d_; ++i) s << (i ? "," : "") << a[i];
    s << "]";
    return s.str();
}
std::string ORing::name() const {
    std::ostringstream s;
    s << "Z_" << Z_.ell << "[x]/(";
    for (uint32_t i = d_ + 1; i-- > 0;) {
        const uint64_t c = F_[i];
        if (i < d_ && c == 0) continue;
        if (i < d_) s << " + ";
        if (i == 0 || c != 1) s << c;
        if (i >= 1) s << "x";
        if (i >= 2) s << "^" << i;
    }
    s << ") mod " << Z_.ell << "^" << Z_.m;
    return s.str();
}

// =====================================================================================================================
// polynomials
// =====================================================================================================================
bool IrreducibleModEll(const Vec& g0, uint64_t p) {
    const Vec g = ModP(g0, p);
    if (g.size() < 2) return false;
    const uint32_t f = static_cast<uint32_t>(g.size() - 1);
    if (f == 1) return true;
    Vec gm = g;
    const uint64_t il = InvModP(gm.back(), p);
    for (auto& c : gm) c = static_cast<uint64_t>(static_cast<u128>(c) * il % p);
    // X^{p^f} == X mod g, and gcd(X^{p^{f/s}} - X, g) = 1 for every prime s | f
    if (SubP(FrobPow(f, gm, p), Vec{0, 1}, p).size() != 0) return false;
    uint32_t m = f;
    for (uint32_t s = 2; s <= m; ++s) {
        if (m % s) continue;
        while (m % s == 0) m /= s;
        const Vec h = SubP(FrobPow(f / s, gm, p), Vec{0, 1}, p);
        if (MonicGcdP(h, gm, p).size() != 1) return false;
    }
    return true;
}

uint32_t MultiplicityModEll(const Vec& f0, const Vec& g, uint64_t p) {
    Vec f = ModP(f0, p);
    uint32_t m = 0;
    while (!f.empty()) {
        Vec q, r;
        DivModP(f, g, p, q, r);
        if (!r.empty()) break;
        f = Trim(q);
        ++m;
    }
    return m;
}

namespace {
Vec HenselFactor(const birch::Ring& R, const Vec& chi, const Vec& gbar, uint32_t e) {
    const uint64_t p = R.ell;
    if (!IrreducibleModEll(gbar, p)) throw std::invalid_argument("P3: gbar is not irreducible mod ell");
    const uint32_t mult = MultiplicityModEll(chi, gbar, p);
    if (mult < e) throw std::invalid_argument("P3: gbar^e does not divide charpoly(T_2 | W^-) mod ell");
    if (mult > e) throw std::invalid_argument("P3: gbar^{e+1} divides charpoly(T_2 | W^-) mod ell (wrong multiplicity e)");
    Vec Fb{1};
    for (uint32_t i = 0; i < e; ++i) Fb = MulP(Fb, ModP(gbar, p), p);
    Vec Gb, r;
    DivModP(chi, Fb, p, Gb, r);
    Gb = Trim(Gb);
    ENGINE_CHECK_MSG(r.empty(), "HenselFactor: division");
    Vec s, t;
    XgcdP(Fb, Gb, p, s, t);
    Vec F(Fb.size(), 0), G(Gb.size(), 0);
    for (size_t i = 0; i < Fb.size(); ++i) F[i] = Fb[i];
    for (size_t i = 0; i < Gb.size(); ++i) G[i] = Gb[i];
    for (uint32_t i = 1; i < R.m; ++i) {
        const uint64_t Ni = pow_wrap(p, i);
        const Vec FG = MulR(R, F, G);
        Vec d(chi.size(), 0);
        for (size_t j = 0; j < chi.size(); ++j) {
            const uint64_t err = R.sub(chi[j] % R.q, j < FG.size() ? FG[j] : 0);
            ENGINE_CHECK_MSG(err % Ni == 0, "HenselFactor: lifting invariant");
            d[j] = (err / Ni) % p;
        }
        d = Trim(d);
        if (d.empty()) continue;
        Vec q, dF, dG;
        DivModP(MulP(t, d, p), Fb, p, q, dF);
        DivModP(MulP(s, d, p), Gb, p, q, dG);
        for (size_t j = 0; j < dF.size(); ++j) F[j] = R.add(F[j], R.mul(Ni % R.q, dF[j]));
        for (size_t j = 0; j < dG.size(); ++j) G[j] = R.add(G[j], R.mul(Ni % R.q, dG[j]));
    }
    const Vec FG = MulR(R, F, G);
    for (size_t j = 0; j < chi.size(); ++j) ENGINE_CHECK_MSG(FG[j] == chi[j] % R.q, "HenselFactor: F G != chi");
    return F;
}
} // namespace

// =====================================================================================================================
// the eigen-symbol
// =====================================================================================================================
std::vector<Elt> KernelVectorLocal(const ORing& O, const birch::Mat& A, const Elt& theta) {
    const uint32_t r = A.r;
    std::vector<std::vector<Elt>> B(r, std::vector<Elt>(r));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) B[i][j] = O.scalar(A.at(i, j));
    for (uint32_t i = 0; i < r; ++i) B[i][i] = O.sub(B[i][i], theta);
    std::vector<uint32_t> pivcols;
    uint32_t row = 0;
    for (uint32_t c = 0; c < r && row < r; ++c) {
        uint32_t piv = row;
        while (piv < r && !O.unit(B[piv][c])) ++piv;
        if (piv == r) continue;
        std::swap(B[row], B[piv]);
        const Elt iv = O.inv(B[row][c]);
        for (auto& v : B[row]) v = O.mul(iv, v);
        for (uint32_t i = 0; i < r; ++i) {
            if (i == row || O.is_zero(B[i][c])) continue;
            const Elt t = B[i][c];
            for (uint32_t j = 0; j < r; ++j) B[i][j] = O.sub(B[i][j], O.mul(t, B[row][j]));
        }
        pivcols.push_back(c);
        ++row;
    }
    if (pivcols.size() + 1 != r) {
        throw std::invalid_argument("P3: residual eigenspace of rank >= 2 (residual rank of T_2 - theta is " + std::to_string(pivcols.size()) +
                                    " of " + std::to_string(r) + ")");
    }
    uint32_t fc = 0;
    while (std::find(pivcols.begin(), pivcols.end(), fc) != pivcols.end()) ++fc;
    std::vector<Elt> y(r, O.zero());
    y[fc] = O.one();
    for (size_t i = 0; i < pivcols.size(); ++i) y[pivcols[i]] = O.neg(B[i][fc]);
    // the eliminated rows below the pivots must vanish (theta an exact root of the charpoly: rank exactly r - 1)
    for (uint32_t i = row; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) ENGINE_CHECK_MSG(O.is_zero(B[i][j]), "KernelVectorLocal: theta is not an eigenvalue to the precision");
    return y;
}

namespace {
std::vector<Vec> WSignBasisCopy(const birch::SymRho& rho, int sign) {
    const birch::Ring& R = rho.ring();
    const uint32_t n = rho.n(), parity = sign < 0 ? 1u : 0u;
    std::vector<uint32_t> cols;
    for (uint32_t t = 0; t <= n; ++t)
        if (t % 2 == parity) cols.push_back(t);
    birch::Mat A(2 * (n + 1), static_cast<uint32_t>(cols.size()));
    const birch::Mat2 U{0, -1, 1, 1};
    for (uint32_t j = 0; j < cols.size(); ++j) {
        Vec e(n + 1, 0);
        e[cols[j]] = 1 % R.q;
        const Vec s = rho.S(e);
        const Vec u1 = rho.apply(U, e), u2 = rho.apply(U, u1);
        for (uint32_t t = 0; t <= n; ++t) {
            A.at(t, j) = R.add(e[t], s[t]);
            A.at(n + 1 + t, j) = R.add(R.add(e[t], u1[t]), u2[t]);
        }
    }
    std::vector<Vec> gens;
    std::vector<uint32_t> exps;
    birch::KernelGenerators(R, A, gens, exps);
    std::vector<Vec> basis;
    for (size_t g = 0; g < gens.size(); ++g) {
        if (exps[g] != R.m) throw std::invalid_argument("P3: W(Z/ell^R) has a finite pivot (W is not free)");
        Vec v(n + 1, 0);
        for (uint32_t j = 0; j < cols.size(); ++j) v[cols[j]] = gens[g][j];
        basis.push_back(v);
    }
    return basis;
}

birch::Mat HeckeOnBasisCopy(const birch::SymRho& rho, const birch::PivotBasis& B, uint64_t p) {
    const birch::Ring& R = rho.ring();
    const uint32_t r = static_cast<uint32_t>(B.basis.size()), len = rho.n() + 1;
    birch::Mat T(r, r);
    for (uint32_t j = 0; j < r; ++j) {
        const Vec img = birch::HeckeTp(rho, B.basis[j], p);
        const Vec c = B.coords(img);
        Vec back(len, 0);
        for (uint32_t i = 0; i < r; ++i)
            if (c[i])
                for (uint32_t t = 0; t < len; ++t) back[t] = R.add(back[t], R.mul(c[i], B.basis[i][t]));
        ENGINE_CHECK_MSG(back == img, "P3: T_p image not in the span of W");
        for (uint32_t i = 0; i < r; ++i) T.at(i, j) = c[i];
    }
    return T;
}
} // namespace

std::vector<Elt> NormaliseFirstUnit(const ORing& O, std::vector<Elt> P, uint32_t* t0) {
    uint32_t t = 0;
    while (t < P.size() && !O.unit(P[t])) ++t;
    ENGINE_CHECK_MSG(t < P.size(), "NormaliseFirstUnit: no coordinate outside the maximal ideal");
    const Elt iv = O.inv(P[t]);
    for (auto& v : P) v = O.mul(iv, v);
    if (t0) *t0 = t;
    return P;
}

std::vector<Elt> HeckeTpO(const birch::SymRho& rho, const ORing& O, const std::vector<Elt>& P, uint64_t p) {
    const uint32_t n1 = static_cast<uint32_t>(P.size());
    std::vector<Elt> out(n1, O.zero());
    for (uint32_t j = 0; j < O.d(); ++j) {
        Vec Pj(n1);
        for (uint32_t t = 0; t < n1; ++t) Pj[t] = P[t][j];
        const Vec img = birch::HeckeTp(rho, Pj, p);
        for (uint32_t t = 0; t < n1; ++t) out[t][j] = img[t];
    }
    return out;
}

Elt UnitRootO(const ORing& O, const Elt& a, uint32_t k) {
    if (!O.unit(a)) throw std::invalid_argument("P3: not ordinary (a_ell lies in the maximal ideal of O)");
    const Elt lk = O.scalar(O.base().pow(O.ell(), k - 1));
    Elt x = a;
    for (uint32_t it = 0; it < O.R() + 3; ++it) x = O.sub(a, O.mul(lk, O.inv(x)));
    ENGINE_CHECK_MSG(O.is_zero(O.add(O.sub(O.mul(x, x), O.mul(a, x)), lk)), "UnitRootO: not a root");
    ENGINE_CHECK_MSG(O.unit(x), "UnitRootO: the root is not a unit");
    return x;
}

Elt GaloisGenerator(const ORing& O, uint32_t e) {
    if (e == 2 && O.d() == 2) return O.sub(O.scalar(O.base().neg(O.F()[1])), O.x());
    ENGINE_CHECK_MSG(e == 1, "GaloisGenerator: unramified, or ramified quadratic");
    // Newton for F at x^ell (F separable mod ell)
    Vec dF;
    for (size_t i = 1; i < O.F().size(); ++i) dF.push_back(O.base().mul(O.F()[i], i % O.q()));
    Elt z = O.pow(O.x(), O.ell());
    for (uint32_t it = 0; it < 2 * O.R() + 4; ++it) z = O.sub(z, O.mul(O.eval(O.F(), z), O.inv(O.eval(dF, z))));
    ENGINE_CHECK_MSG(O.is_zero(O.eval(O.F(), z)), "GaloisGenerator: Newton");
    return z;
}

std::vector<Elt> ApplySigma(const ORing& O, const Elt& sx, const std::vector<Elt>& v) {
    std::vector<Elt> o;
    o.reserve(v.size());
    for (const Elt& a : v) o.push_back(O.eval(Vec(a.begin(), a.end()), sx));
    return o;
}

OSymbolCell RunOSymbolCellIn(uint32_t n, const ORing& O, const Elt& theta, int sign) {
    const auto t0c = std::chrono::steady_clock::now();
    OSymbolCell C;
    C.n = n;
    C.ell = O.ell();
    C.R = O.R();
    C.sign = sign;
    C.O = O;
    C.theta = theta;
    const birch::Ring& R = O.base();
    const birch::SymRho rho(n, R);
    try {
        const birch::PivotBasis B = birch::SaturatedBasis(R, WSignBasisCopy(rho, sign));
        ENGINE_CHECK_MSG(B.free_summand && !B.basis.empty(), "P3: W^{sign} not a free summand");
        C.dimW = static_cast<uint32_t>(B.basis.size());
        const birch::Mat T2 = HeckeOnBasisCopy(rho, B, 2);
        const std::vector<Elt> y = KernelVectorLocal(O, T2, theta);
        std::vector<Elt> P(n + 1, O.zero());
        for (size_t i = 0; i < y.size(); ++i)
            for (uint32_t t = 0; t <= n; ++t)
                if (B.basis[i][t]) P[t] = O.add(P[t], O.smul(B.basis[i][t], y[i]));
        C.P = NormaliseFirstUnit(O, P, &C.t0);
        C.eigen_all = true;
        for (uint64_t p : std::vector<uint64_t>{2, 3, 5, 7, C.ell}) {
            const std::vector<Elt> img = HeckeTpO(rho, O, C.P, p);
            const Elt a = img[C.t0];
            for (uint32_t t = 0; t <= n; ++t)
                if (img[t] != O.mul(a, C.P[t])) C.eigen_all = false;
            C.a_p[p] = a;
        }
        C.ordinary = O.unit(C.a_p[C.ell]);
        if (C.ordinary) C.alpha = UnitRootO(O, C.a_p[C.ell], n + 2);
    } catch (const std::invalid_argument& ex) {
        C.refusal = ex.what();
    }
    C.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count();
    return C;
}

OSymbolCell RunOSymbolCell(uint32_t n, uint64_t ell, uint32_t Rp, const Vec& gbar, uint32_t e, int sign, const Elt* theta_override) {
    const birch::Ring R = birch::MakeRing(ell, Rp);
    const birch::SymRho rho(n, R);
    OSymbolCell C;
    C.n = n;
    C.ell = ell;
    C.R = Rp;
    C.sign = sign;
    C.e = e;
    C.gbar = gbar;
    try {
        const birch::PivotBasis Bm = birch::SaturatedBasis(R, WSignBasisCopy(rho, -1));
        ENGINE_CHECK_MSG(Bm.free_summand && !Bm.basis.empty(), "P3: W^- not a free summand");
        C.chi_T2 = birch::CharPoly(R, HeckeOnBasisCopy(rho, Bm, 2));
        C.chi_T3 = birch::CharPoly(R, HeckeOnBasisCopy(rho, Bm, 3));
        C.F = HenselFactor(R, C.chi_T2, gbar, e);
    } catch (const std::invalid_argument& ex) {
        C.refusal = ex.what();
        return C;
    }
    const ORing O(ell, Rp, C.F);
    const Elt theta = theta_override ? *theta_override : O.x();
    OSymbolCell D = RunOSymbolCellIn(n, O, theta, sign);
    D.e = e;
    D.gbar = gbar;
    D.F = C.F;
    D.chi_T2 = C.chi_T2;
    D.chi_T3 = C.chi_T3;
    return D;
}

// =====================================================================================================================
// the measure
// =====================================================================================================================
OMeasure BuildOMeasure(const birch::SymRho& rho, const ORing& O, const std::vector<Elt>& P, const Elt& a_ell, const Elt& alpha, uint32_t M,
                       bool ell_n_factor) {
    const uint32_t n = rho.n(), d = O.d();
    ENGINE_CHECK_MSG(P.size() == n + 1, "BuildOMeasure: P has length n + 1");
    ENGINE_CHECK_MSG(rho.ring().q == O.q(), "BuildOMeasure: rho over the base ring of O");
    OMeasure out;
    out.O = O;
    out.n = n;
    out.M = M;
    out.ell_n_factor = ell_n_factor;
    out.a_ell = a_ell;
    out.alpha = alpha;
    // the c-tables of the components (WP-MEAS's BuildMeasure; its mu at alpha = 1 is not used)
    std::vector<ModularSymbolMeasure> comp;
    for (uint32_t j = 0; j < d; ++j) {
        Vec Pj(n + 1);
        for (uint32_t t = 0; t <= n; ++t) Pj[t] = P[t][j];
        comp.push_back(birch::BuildMeasure(rho, Pj, 1, 1, M, true));
    }
    out.c.assign(M + 2, {});
    for (uint32_t m = 0; m < M + 2; ++m) {
        const size_t sz = comp[0].c[m].size();
        out.c[m].assign(sz, O.zero());
        for (size_t a = 0; a < sz; ++a)
            for (uint32_t j = 0; j < d; ++j) out.c[m][a][j] = comp[j].c[m][a];
    }
    const Elt ia = O.inv(alpha);
    const Elt ln = O.scalar(ell_n_factor ? O.base().pow(O.ell(), n) : 1);
    out.mu.assign(M + 1, {});
    Elt am = O.one();
    for (uint32_t m = 0; m <= M; ++m) {
        const Elt am1 = O.mul(am, ia), w = O.mul(am1, ln);
        const auto& cur = out.c[m + 1];
        const auto& prev = out.c[m];
        auto& mu = out.mu[m];
        mu.assign(cur.size(), O.zero());
        for (size_t a = 0; a < cur.size(); ++a) mu[a] = O.sub(O.mul(am, cur[a]), O.mul(w, prev[a % prev.size()]));
        am = am1;
    }
    return out;
}

uint64_t OMeasure::fnv_units(uint32_t level) const {
    ENGINE_CHECK_MSG(level <= M, "OMeasure::fnv_units: level <= M");
    uint64_t h = engine::kFnvOffset64;
    for (uint64_t a = 1; a < mu[level].size(); ++a)
        if (a % O.ell())
            for (uint64_t v : mu[level][a]) h = engine::FnvU64(h, v);
    return h;
}
uint64_t OMeasure::fnv_trace_norm(uint32_t level) const {
    ENGINE_CHECK_MSG(level <= M, "OMeasure::fnv_trace_norm: level <= M");
    uint64_t h = engine::kFnvOffset64;
    for (uint64_t a = 1; a < mu[level].size(); ++a)
        if (a % O.ell()) {
            h = engine::FnvU64(h, O.trace(mu[level][a]));
            h = engine::FnvU64(h, O.norm(mu[level][a]));
        }
    return h;
}
Elt OMeasure::unit_total(uint32_t level) const {
    Elt s = O.zero();
    for (uint64_t a = 1; a < mu[level].size(); ++a)
        if (a % O.ell()) s = O.add(s, mu[level][a]);
    return s;
}

uint32_t ThreeTermFailuresO(const OMeasure& me, uint32_t m) {
    ENGINE_CHECK_MSG(m + 1 <= me.M, "ThreeTermFailuresO: m + 1 <= M");
    const ORing& O = me.O;
    const uint64_t Mm = pow_wrap(O.ell(), m);
    const Elt ln = O.scalar(O.base().pow(O.ell(), me.n));
    const auto &next = me.c[m + 2], &cur = me.c[m + 1], &prev = me.c[m];
    uint32_t bad = 0;
    for (uint64_t a = 0; a < Mm; ++a) {
        Elt s = O.zero();
        for (uint64_t b = 0; b < O.ell(); ++b) s = O.add(s, next[a + b * Mm]);
        if (s != O.sub(O.mul(me.a_ell, cur[a]), O.mul(ln, prev[a % prev.size()]))) ++bad;
    }
    return bad;
}

uint32_t DistributionFailuresO(const OMeasure& me, uint32_t m) {
    ENGINE_CHECK_MSG(m + 1 <= me.M, "DistributionFailuresO: m + 1 <= M");
    const ORing& O = me.O;
    const uint64_t Mm = pow_wrap(O.ell(), m);
    uint32_t bad = 0;
    for (uint64_t a = 0; a < Mm; ++a) {
        Elt s = O.zero();
        for (uint64_t b = 0; b < O.ell(); ++b) s = O.add(s, me.mu[m + 1][a + b * Mm]);
        if (s != me.mu[m][a]) ++bad;
    }
    return bad;
}

std::pair<uint32_t, uint32_t> MomentFailuresO(const birch::SymRho& rho, const OMeasure& me, const std::vector<Elt>& P,
                                              const std::vector<uint32_t>& chars) {
    const ORing& O = me.O;
    const birch::Ring& Z = O.base();
    const uint64_t ell = O.ell(), q = O.q();
    const uint32_t n = me.n, M = me.M, d = O.d();
    const uint64_t prec = pow_wrap(ell, std::min(M, O.R()));
    const uint64_t MM = pow_wrap(ell, M);
    const Elt ia = O.inv(me.alpha);
    std::vector<Vec> comps(d, Vec(n + 1));
    for (uint32_t j = 0; j < d; ++j)
        for (uint32_t t = 0; t <= n; ++t) comps[j][t] = P[t][j];
    uint32_t bad = 0, tot = 0;
    for (uint32_t i : chars) {
        const uint32_t ebar = static_cast<uint32_t>((ell - 1 - i % (ell - 1)) % (ell - 1));
        std::vector<uint64_t> wt(ell, 0);
        for (uint64_t r = 1; r < ell; ++r) wt[r] = TeichmullerPowMod(r, ebar, ell, O.R()) % q;
        // the Riemann sums, all j at once
        std::vector<Elt> mom(n + 1, O.zero());
        for (uint64_t a = 0; a < MM; ++a) {
            const uint64_t r = a % ell;
            if (r == 0 || O.is_zero(me.mu[M][a])) continue;
            const Elt y = O.smul(wt[r], me.mu[M][a]);
            uint64_t pa = 1 % q;
            for (uint32_t j = 0; j <= n; ++j) {
                mom[j] = O.add(mom[j], O.smul(pa, y));
                pa = Z.mul(pa, a % q);
            }
        }
        std::vector<Vec> ev(d);
        for (uint32_t c = 0; c < d; ++c) {
            if (i == 0) {
                ev[c].assign(n + 1, 0);
                for (uint32_t j = 0; j <= n; ++j) ev[c][j] = comps[c][n - j];
            } else {
                ev[c] = birch::EvAllJ(rho, comps[c], birch::TeichmullerCharacter(i, Z));
            }
        }
        for (uint32_t j = 0; j <= n; ++j) {
            Elt E;
            if (i == 0)
                E = O.mul(O.sub(O.one(), O.smul(Z.pow(ell, j), ia)), O.sub(O.one(), O.smul(Z.pow(ell, n - j), ia)));
            else
                E = ia;
            Elt evj(d);
            for (uint32_t c = 0; c < d; ++c) evj[c] = ev[c][j];
            Elt lhs = O.smul(rho.binom(n, j), mom[j]);
            if (j % 2 == 1) lhs = O.neg(lhs);
            const Elt diff = O.sub(lhs, O.mul(E, evj));
            ++tot;
            if (std::any_of(diff.begin(), diff.end(), [&](uint64_t v) { return v % prec != 0; })) ++bad;
        }
    }
    return {bad, tot};
}

// =====================================================================================================================
// route B: the m-block reader
// =====================================================================================================================
RouteBReading RouteBRead(uint32_t n, uint64_t ell, uint32_t R, uint64_t lambda0, int sign) {
    RouteBReading out;
    const uint32_t RB = hidapair::Zq::MaxPrecision(ell, R + 4);
    const birch::Ring Rg = birch::MakeRing(ell, RB);
    const hidapair::Zq Zq(ell, RB);
    const birch::SymRho rho(n, Rg);
    const birch::PivotBasis B = birch::SaturatedBasis(Rg, WSignBasisCopy(rho, sign));
    const birch::Mat T2 = HeckeOnBasisCopy(rho, B, 2);
    hidapair::Mat A(T2.r, hidapair::Vec(T2.c, 0));
    for (uint32_t i = 0; i < T2.r; ++i)
        for (uint32_t j = 0; j < T2.c; ++j) A[i][j] = T2.at(i, j);
    const mblock::Block blk = mblock::BlockOf(Zq, A, lambda0);
    out.block_rank = static_cast<uint32_t>(blk.basis.size());
    if (blk.basis.empty()) {
        out.refusal = "route B: empty block";
        return out;
    }
    const mblock::RootReading rr = mblock::RootsOfNilpotentCharpoly(Zq, blk.charpoly);
    if (!rr.complete) {
        out.refusal = rr.refusal;
        return out;
    }
    out.roots = static_cast<uint32_t>(rr.roots.size());
    std::vector<bool> used(rr.orders.size(), false);
    for (const mblock::Root& rt : rr.roots) {
        const mblock::LocalOrder& LO = rr.orders[rt.order];
        used[rt.order] = true;
        out.e = std::max(out.e, LO.e());
        const mblock::EigenLineO E = mblock::EigenLineAt(LO, blk.t, blk.charpoly, rt.rho);
        // O_pi at precision ell^R: F = pi^e - ell u
        Vec F(LO.e() + 1, 0);
        const birch::Ring Rr = birch::MakeRing(ell, R);
        F[0] = Rr.neg(Rr.mul(ell % Rr.q, LO.u() % Rr.q));
        F[LO.e()] = 1;
        RouteBRoot X;
        X.Opi = ORing(ell, R, F);
        X.eigen_check = E.eigen_check;
        X.rho = Elt(LO.e());
        for (uint32_t i = 0; i < LO.e(); ++i) X.rho[i] = rt.rho[i] % Rr.q;
        // ambient: block coordinates -> W coordinates -> symbol coordinates
        std::vector<Elt> P(n + 1, X.Opi.zero());
        for (size_t b = 0; b < blk.basis.size(); ++b) {
            Elt gb(LO.e());
            for (uint32_t i = 0; i < LO.e(); ++i) gb[i] = E.g[b][i] % Rr.q;
            for (size_t w = 0; w < blk.basis[b].size(); ++w) {
                const uint64_t cw = blk.basis[b][w] % Rr.q;
                if (!cw) continue;
                for (uint32_t t = 0; t <= n; ++t) {
                    const uint64_t s = Rr.mul(cw, B.basis[w][t] % Rr.q);
                    if (s) P[t] = X.Opi.add(P[t], X.Opi.smul(s, gb));
                }
            }
        }
        X.P = NormaliseFirstUnit(X.Opi, P, &X.t0);
        out.found.push_back(X);
    }
    out.orders = static_cast<uint32_t>(std::count(used.begin(), used.end(), true));
    out.complete = true;
    return out;
}

Elt PhiRho(const ORing& O, const RouteBRoot& B, uint64_t lambda0, const Elt& a) {
    const ORing& T = B.Opi;
    const Elt z = T.add(T.scalar(lambda0), B.rho);
    ENGINE_CHECK_MSG(T.is_zero(T.eval(O.F(), z)), "PhiRho: F(lambda0 + rho) != 0 in O_pi");
    return T.eval(Vec(a.begin(), a.end()), z);
}

} // namespace oring
} // namespace lvalue
} // namespace hecke_ext
