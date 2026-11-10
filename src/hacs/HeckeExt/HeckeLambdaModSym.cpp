// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P8) Lambda-adic modular symbols, truncated: the
// ordinary g-part of the modular symbols per weight and sign, the eigensymbol,
// the operator-valued masses, the normaliser of record, the interpolation and
// the gluing depth j_k.  See HeckeLambdaModSym.h.
#include "HeckeLambdaModSym.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../../ModArithmetic/Fnv.h"
#include "LValue/HidaPairing.h"
#include "LValue/MBlockReader.h"
#include "LValue/ModularSymbolMeasure.h"

namespace hecke_ext {
namespace lambda {
namespace msym {

namespace {

using u128 = unsigned __int128;

double Secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

bool IsZero(const SVec& v) {
    for (uint64_t x : v)
        if (x) return false;
    return true;
}

SVec LinComb(const bm::Ring& R, const std::vector<SVec>& basis, const SVec& c, size_t len) {
    SVec out(len, 0);
    for (size_t i = 0; i < basis.size(); ++i)
        if (c[i] % R.q)
            for (size_t t = 0; t < len; ++t)
                if (basis[i][t]) out[t] = R.add(out[t], R.mul(c[i] % R.q, basis[i][t] % R.q));
    return out;
}

SMat FromColumns(const std::vector<SVec>& cols, uint32_t rows) {
    SMat A(rows, static_cast<uint32_t>(cols.size()));
    for (uint32_t j = 0; j < A.c; ++j)
        for (uint32_t i = 0; i < rows; ++i) A.at(i, j) = cols[j][i];
    return A;
}

SVec Column(const SMat& A, uint32_t j) {
    SVec v(A.r);
    for (uint32_t i = 0; i < A.r; ++i) v[i] = A.at(i, j);
    return v;
}

uint64_t PowModU(uint64_t b, uint64_t e, uint64_t q) { return engine::scalar::powmod<uint64_t>(b, e, q); }

using engine::scalar::pow_wrap;

uint32_t MaxPrec(uint64_t ell) {
    uint32_t Q = 0;
    for (u128 x = ell; x < (static_cast<u128>(1) << 62); x *= ell) ++Q;
    return Q;
}

// p_0 = u_0 - u_n
SVec P0Vector(const bm::Ring& R, uint32_t n) {
    SVec v(n + 1, 0);
    v[0] = 1 % R.q;
    v[n] = R.neg(1 % R.q);
    return v;
}

// a column-convention matrix A restricted to a basis: column j = coords(A applied ...) is done by the callers.
SMat Sub(const bm::Ring& R, const SMat& A, const SMat& B) {
    SMat C = A;
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] = R.sub(A.a[i] % R.q, B.a[i] % R.q);
    return C;
}
SMat AddScalar(const bm::Ring& R, const SMat& A, uint64_t s) {
    SMat C = A;
    for (uint32_t i = 0; i < C.r; ++i) C.at(i, i) = R.add(C.at(i, i) % R.q, s % R.q);
    return C;
}

uint64_t NextPow2AtLeast(uint64_t x) {
    uint64_t e = 1;
    while (e < x) e <<= 1;
    return e;
}

}  // namespace

// ===========================================================================
// dense helpers
// ===========================================================================
SMat Identity(const bm::Ring& R, uint32_t n) {
    SMat I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.at(i, i) = 1 % R.q;
    return I;
}

SMat MatPolyOf(const bm::Ring& R, const SVec& poly, const SMat& A) {
    SMat acc(A.r, A.c);
    for (size_t i = poly.size(); i-- > 0;) {
        acc = bm::MatMul(R, acc, A);
        for (uint32_t d = 0; d < A.r; ++d) acc.at(d, d) = R.add(acc.at(d, d), poly[i] % R.q);
    }
    return acc;
}

SMat MatPow(const bm::Ring& R, const SMat& A, uint64_t e) {
    SMat out = Identity(R, A.r), b = A;
    while (e) {
        if (e & 1) out = bm::MatMul(R, out, b);
        e >>= 1;
        if (e) b = bm::MatMul(R, b, b);
    }
    return out;
}

bool InverseUnit(const bm::Ring& R, const SMat& A, SMat& out) {
    ENGINE_CHECK(A.r == A.c);
    const uint32_t n = A.r;
    SMat M = A, I = Identity(R, n);
    for (uint32_t c = 0; c < n; ++c) {
        uint32_t p = n;
        for (uint32_t i = c; i < n; ++i)
            if (R.unit(M.at(i, c))) {
                p = i;
                break;
            }
        if (p == n) return false;
        if (p != c)
            for (uint32_t j = 0; j < n; ++j) {
                std::swap(M.at(p, j), M.at(c, j));
                std::swap(I.at(p, j), I.at(c, j));
            }
        const uint64_t iv = R.inv(M.at(c, c));
        for (uint32_t j = 0; j < n; ++j) {
            M.at(c, j) = R.mul(M.at(c, j), iv);
            I.at(c, j) = R.mul(I.at(c, j), iv);
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (i == c) continue;
            const uint64_t f = M.at(i, c);
            if (!f) continue;
            for (uint32_t j = 0; j < n; ++j) {
                M.at(i, j) = R.sub(M.at(i, j), R.mul(f, M.at(c, j)));
                I.at(i, j) = R.sub(I.at(i, j), R.mul(f, I.at(c, j)));
            }
        }
    }
    out = I;
    return true;
}

SMat Reduce(const SMat& A, uint64_t q) {
    SMat B = A;
    for (auto& x : B.a) x %= q;
    return B;
}
SVec Reduce(const SVec& v, uint64_t q) {
    SVec w = v;
    for (auto& x : w) x %= q;
    return w;
}
bool MatEq(const SMat& A, const SMat& B) { return A.r == B.r && A.c == B.c && A.a == B.a; }

uint32_t RankModEll(const bm::Ring& R, const SMat& A) {
    const uint64_t l = R.ell;
    std::vector<std::vector<uint64_t>> M(A.r, std::vector<uint64_t>(A.c));
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) M[i][j] = A.at(i, j) % l;
    uint32_t rk = 0;
    for (uint32_t c = 0; c < A.c && rk < A.r; ++c) {
        uint32_t p = A.r;
        for (uint32_t i = rk; i < A.r; ++i)
            if (M[i][c]) {
                p = i;
                break;
            }
        if (p == A.r) continue;
        std::swap(M[p], M[rk]);
        const uint64_t iv = PowModU(M[rk][c], l - 2, l);
        for (auto& x : M[rk]) x = x * iv % l;
        for (uint32_t i = 0; i < A.r; ++i) {
            if (i == rk || !M[i][c]) continue;
            const uint64_t f = M[i][c];
            for (uint32_t j = 0; j < A.c; ++j) M[i][j] = (M[i][j] + l * l - f * M[rk][j] % l) % l;
        }
        ++rk;
    }
    return rk;
}

SMat AddScalarPublic(const bm::Ring& R, const SMat& A, uint64_t s) { return AddScalar(R, A, s); }

bool NilpotentModEll(const bm::Ring& R, const SMat& A) {
    const bm::Ring F = bm::MakeRing(R.ell, 1);
    const SMat B = MatPow(F, Reduce(A, R.ell), NextPow2AtLeast(std::max<uint32_t>(A.r, 1)));
    for (uint64_t x : B.a)
        if (x) return false;
    return true;
}

bool FreeBasis::coords(const bm::Ring& R, const SVec& w, SVec& c) const {
    const uint32_t r = rank();
    SVec sel(r);
    for (uint32_t i = 0; i < r; ++i) sel[i] = w[rows[i]] % R.q;
    c = r ? bm::MatVec(R, minv, sel) : SVec{};
    const SVec back = LinComb(R, b, c, w.size());
    for (size_t t = 0; t < w.size(); ++t)
        if (back[t] != w[t] % R.q) return false;
    return true;
}

FreeBasis MakeFreeBasis(const bm::Ring& R, const std::vector<SVec>& vecs) {
    FreeBasis B;
    B.b = vecs;
    const uint32_t r = static_cast<uint32_t>(vecs.size());
    if (r == 0) return B;
    const size_t len = vecs[0].size();
    const uint64_t l = R.ell;
    // greedy pivot positions mod ell (rows of a unit minor)
    std::vector<std::vector<uint64_t>> M(r, std::vector<uint64_t>(len));
    for (uint32_t j = 0; j < r; ++j)
        for (size_t t = 0; t < len; ++t) M[j][t] = vecs[j][t] % l;
    for (uint32_t j = 0; j < r; ++j) {
        size_t piv = len;
        for (size_t t = 0; t < len; ++t)
            if (M[j][t]) {
                piv = t;
                break;
            }
        ENGINE_CHECK_MSG(piv < len, "msym::MakeFreeBasis: the vectors are not a basis of a free direct summand");
        B.rows.push_back(static_cast<uint32_t>(piv));
        const uint64_t iv = PowModU(M[j][piv], l - 2, l);
        for (auto& x : M[j]) x = x * iv % l;
        for (uint32_t h = j + 1; h < r; ++h) {
            const uint64_t f = M[h][piv];
            if (!f) continue;
            for (size_t t = 0; t < len; ++t) M[h][t] = (M[h][t] + l * l - f * M[j][t] % l) % l;
        }
    }
    SMat Mi(r, r);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) Mi.at(i, j) = vecs[j][B.rows[i]] % R.q;
    ENGINE_CHECK_MSG(InverseUnit(R, Mi, B.minv), "msym::MakeFreeBasis: the pivot minor is not a unit");
    return B;
}

// ===========================================================================
// Hecke on symbols
// ===========================================================================
SymbolHecke::SymbolHecke(uint32_t n, const bm::Ring& R) : n_(n), R_(R), binom_(static_cast<size_t>(n + 1) * (n + 1), 0) {
    for (uint32_t t = 0; t <= n; ++t) {
        binom_[static_cast<size_t>(t) * (n + 1)] = 1 % R.q;
        for (uint32_t s = 1; s <= t; ++s)
            binom_[static_cast<size_t>(t) * (n + 1) + s] =
                R.add(binom_[static_cast<size_t>(t - 1) * (n + 1) + s - 1], s <= t - 1 ? binom_[static_cast<size_t>(t - 1) * (n + 1) + s] : 0);
    }
}

uint64_t SymbolHecke::mul(uint64_t a, uint64_t b) const {
    if (R_.q < (uint64_t{1} << 32)) return a * b % R_.q;
    return static_cast<uint64_t>(static_cast<u128>(a) * b % R_.q);
}

SVec SymbolHecke::sflip(const SVec& v) const {
    SVec out(n_ + 1, 0);
    for (uint32_t t = 0; t <= n_; ++t) out[n_ - t] = (t % 2 == 0) ? v[t] : R_.neg(v[t]);
    return out;
}

SVec SymbolHecke::diag(uint64_t a, uint64_t d, const SVec& v) const {
    SVec pa(n_ + 1, 1 % R_.q), pd(n_ + 1, 1 % R_.q), out(n_ + 1, 0);
    for (uint32_t i = 1; i <= n_; ++i) {
        pa[i] = mul(pa[i - 1], a);
        pd[i] = mul(pd[i - 1], d);
    }
    for (uint32_t t = 0; t <= n_; ++t) out[t] = mul(mul(pa[n_ - t], pd[t]), v[t]);
    return out;
}

// out_s = sum_{t >= s} C(t, s) b^{t-s} v_t  (the Taylor shift, Horner-style: one product per step)
SVec SymbolHecke::upper(uint64_t b, const SVec& v) const {
    if (b % R_.q == 0) return v;
    // v as the polynomial sum_t v_t X^t; out = coefficients of v(X + b)
    SVec a = v;
    for (uint32_t i = 0; i < n_; ++i)
        for (uint32_t j = n_; j-- > i;) a[j] = R_.add(a[j], mul(b, a[j + 1]));
    return a;
}

// out_s = sum_{t <= s} C(n - t, s - t) g^{s-t} v_t
SVec SymbolHecke::lower(uint64_t g, const SVec& v) const {
    if (g % R_.q == 0) return v;
    // reversed coordinates w_{n-t} = v_t turn it into the Taylor shift of upper
    SVec a(n_ + 1);
    for (uint32_t t = 0; t <= n_; ++t) a[n_ - t] = v[t];
    for (uint32_t i = 0; i < n_; ++i)
        for (uint32_t j = n_; j-- > i;) a[j] = R_.add(a[j], mul(g, a[j + 1]));
    SVec out(n_ + 1);
    for (uint32_t t = 0; t <= n_; ++t) out[t] = a[n_ - t];
    return out;
}

SVec SymbolHecke::apply(const bm::Mat2& A, const SVec& v) const {
    ENGINE_CHECK_MSG(v.size() == n_ + 1, "msym::SymbolHecke::apply: vector length != n + 1");
    const uint64_t a = R_.from_int(A.a), b = R_.from_int(A.b), c = R_.from_int(A.c);
    const __int128 det = static_cast<__int128>(A.a) * A.d - static_cast<__int128>(A.b) * A.c;
    const uint64_t dt = R_.from_int(static_cast<int64_t>(det));
    if (R_.unit(a)) {
        const uint64_t ia = R_.inv(a);
        SVec w = upper(mul(b, ia), v);
        w = diag(a, mul(dt, ia), w);
        return lower(mul(c, ia), w);
    }
    if (R_.unit(c)) return sflip(apply(bm::Mat2{-A.c, -A.d, A.a, A.b}, v));   // n even: rho(S^{-1}) = rho(S)
    if (R_.unit(b) || R_.unit(R_.from_int(A.d))) return apply(bm::Mat2{A.b, -A.a, A.d, -A.c}, sflip(v));
    throw std::invalid_argument("msym::SymbolHecke::apply: no entry of A is an ell-unit");
}

SVec SymbolHecke::Tp(const SVec& P, uint64_t p) const {
    ENGINE_CHECK_MSG(n_ % 2 == 0, "msym::SymbolHecke: n even");
    SVec out = diag(1, R_.from_int(static_cast<int64_t>(p)), P);
    for (uint64_t b = 0; b < p; ++b) {
        SVec acc(n_ + 1, 0);
        for (const bm::Mat2& g : bm::ManinPath(static_cast<int64_t>(b), static_cast<int64_t>(p))) {
            const SVec x = apply(g, P);
            for (uint32_t t = 0; t <= n_; ++t) acc[t] = R_.add(acc[t], x[t]);
        }
        for (auto& x : acc) x = R_.neg(x);   // Phi({b/p, oo})
        const SVec y = apply(bm::Mat2{static_cast<int64_t>(p), -static_cast<int64_t>(b), 0, 1}, acc);
        for (uint32_t t = 0; t <= n_; ++t) out[t] = R_.add(out[t], y[t]);
    }
    return out;
}

// ===========================================================================
// the sign space
// ===========================================================================
SVec SignSpace::ApplyT(const SVec& v, uint64_t p) const {
    if (wrong_adjoint) {
        const bm::SymRho rho(n, ring);
        return bm::HeckeTpWrongAdjoint(rho, v, p);
    }
    return hecke->Tp(v, p);
}

const SMat& SignSpace::TonW(uint64_t p) {
    auto it = T.find(p);
    if (it != T.end()) return it->second;
    const uint32_t w = W.rank();
    SMat A(w, w);
    std::unique_ptr<bm::SymRho> rho;
    if (wrong_adjoint) rho.reset(new bm::SymRho(n, ring));
    for (uint32_t j = 0; j < w; ++j) {
        const SVec img = wrong_adjoint ? bm::HeckeTpWrongAdjoint(*rho, W.b[j], p) : hecke->Tp(W.b[j], p);
        SVec c;
        ENGINE_CHECK_MSG(W.coords(ring, img, c), "msym::SignSpace: T_p(W) not in W");
        for (uint32_t i = 0; i < w; ++i) A.at(i, j) = c[i];
    }
    return T.emplace(p, A).first->second;
}

SignSpace BuildSignSpace(uint64_t ell, uint32_t n, int c_sign, uint32_t R, uint32_t Emax, bool wrong_adjoint) {
    ENGINE_CHECK_MSG(n % 2 == 0 && n >= 2 && ell >= 5 && (c_sign == 1 || c_sign == -1), "msym::BuildSignSpace: shape");
    SignSpace S;
    S.ell = ell;
    S.n = n;
    S.c_sign = c_sign;
    S.parity = c_sign < 0 ? 0 : 1;   // [C]-minus = geometric-even (p_0 = u_0 - u_n), [C]-plus = geometric-odd
    S.R = R;
    S.wrong_adjoint = wrong_adjoint;
    S.ring = bm::MakeRing(ell, R);
    const uint32_t Emx = std::min<uint32_t>(Emax, MaxPrec(ell) > R ? MaxPrec(ell) - R : 0);
    S.E = Emx;
    const bm::Ring Rh = bm::MakeRing(ell, R + Emx);
    const SymbolHecke Hh(n, Rh);
    std::vector<uint32_t> cols;
    for (uint32_t t = 0; t <= n; ++t)
        if (t % 2 == S.parity) cols.push_back(t);
    bm::Mat A(2 * (n + 1), static_cast<uint32_t>(cols.size()));
    const bm::Mat2 U{0, -1, 1, 1}, Sm{0, -1, 1, 0};
    for (uint32_t j = 0; j < cols.size(); ++j) {
        SVec e(n + 1, 0);
        e[cols[j]] = 1 % Rh.q;
        const SVec s = Hh.apply(Sm, e);
        const SVec u1 = Hh.apply(U, e), u2 = Hh.apply(U, u1);
        for (uint32_t t = 0; t <= n; ++t) {
            A.at(t, j) = Rh.add(e[t], s[t]);
            A.at(n + 1 + t, j) = Rh.add(Rh.add(e[t], u1[t]), u2[t]);
        }
    }
    std::vector<SVec> gens;
    std::vector<uint32_t> exps;
    bm::KernelGenerators(Rh, A, gens, exps);
    std::vector<SVec> red;
    for (size_t g = 0; g < gens.size(); ++g) {
        SVec v(n + 1, 0);
        for (uint32_t j = 0; j < cols.size(); ++j) v[cols[j]] = gens[g][j] % S.ring.q;
        if (!IsZero(v)) red.push_back(v);
    }
    const bm::PivotBasis PB = bm::SaturatedBasis(S.ring, red);
    ENGINE_CHECK_MSG(PB.free_summand,
                     "msym::BuildSignSpace REFUSED: the image of W(Z/ell^{R+E}) in W(Z/ell^R) is not free (torsion of exponent > E)");
    S.W = MakeFreeBasis(S.ring, PB.basis);
    S.hecke = std::make_shared<SymbolHecke>(n, S.ring);
    return S;
}

// ===========================================================================
// the ordinary g-part
// ===========================================================================
OrdBlock OrdinaryPart(SignSpace& S, uint32_t k, uint64_t p0, const std::vector<uint64_t>& g, bool want_p0) {
    const bm::Ring& R = S.ring;
    OrdBlock B;
    B.k = k;
    B.n = S.n;
    B.R = S.R;
    B.c_sign = S.c_sign;
    B.w_rank = S.W.rank();
    const uint32_t w = B.w_rank;
    const size_t len = S.n + 1;
    if (w == 0) {
        B.refusal = "W_zero";
        return B;
    }
    const SMat& T0 = S.TonW(p0);
    // the generalised g(T_{p0})-part: ker g(T)^N (Fitting), N >= R w
    SVec gl(g.begin(), g.end());
    const SMat G = MatPow(R, MatPolyOf(R, gl, T0), NextPow2AtLeast(static_cast<uint64_t>(S.R) * w));
    std::vector<SVec> kg;
    std::vector<uint32_t> ke;
    bm::KernelGenerators(R, G, kg, ke);
    B.g_free = true;
    std::vector<SVec> blockv;
    for (size_t i = 0; i < kg.size(); ++i) {
        if (ke[i] != S.R) B.g_free = false;
        blockv.push_back(LinComb(R, S.W.b, kg[i], len));
    }
    B.g_rank = static_cast<uint32_t>(blockv.size());
    if (!B.g_free) {
        B.refusal = "g_part_not_free";
        return B;
    }
    if (B.g_rank == 0) {
        B.refusal = "g_part_zero";
        return B;
    }
    const FreeBasis Bk = MakeFreeBasis(R, blockv);
    // T_ell on the g-part only (the cost lever)
    SMat A(B.g_rank, B.g_rank);
    for (uint32_t j = 0; j < B.g_rank; ++j) {
        SVec c;
        ENGINE_CHECK_MSG(Bk.coords(R, S.ApplyT(Bk.b[j], S.ell), c), "msym::OrdinaryPart: T_ell does not preserve the g-part");
        for (uint32_t i = 0; i < B.g_rank; ++i) A.at(i, j) = c[i];
    }
    // the ordinary part: im T_ell^N (Fitting), N >= R g_rank
    const SMat AN = MatPow(R, A, NextPow2AtLeast(static_cast<uint64_t>(S.R) * B.g_rank));
    std::vector<SVec> imv;
    for (uint32_t j = 0; j < B.g_rank; ++j) imv.push_back(LinComb(R, Bk.b, Column(AN, j), len));
    const bm::PivotBasis PO = bm::SaturatedBasis(R, imv);
    B.ord_free = PO.free_summand;
    B.ord_rank = static_cast<uint32_t>(PO.basis.size());
    if (!B.ord_free) {
        B.refusal = "ordinary_part_not_free";
        return B;
    }
    if (B.ord_rank == 0) {
        B.refusal = "ordinary_part_zero";
        return B;
    }
    std::vector<SVec> ob = PO.basis;
    {
        const FreeBasis O0 = MakeFreeBasis(R, ob);
        SVec c;
        B.p0_in_block = O0.coords(R, P0Vector(R, S.n), c);
        if (want_p0 && B.p0_in_block) {
            uint32_t iu = O0.rank();
            for (uint32_t i = 0; i < O0.rank(); ++i)
                if (R.unit(c[i])) {
                    iu = i;
                    break;
                }
            ENGINE_CHECK_MSG(iu < O0.rank(), "msym::OrdinaryPart: p_0 is not primitive in the ordinary part");
            std::vector<SVec> nb{P0Vector(R, S.n)};
            for (uint32_t i = 0; i < O0.rank(); ++i)
                if (i != iu) nb.push_back(ob[i]);
            ob = nb;
            B.has_p0 = true;
        }
    }
    B.O = MakeFreeBasis(R, ob);
    // T_ell and T_{p0} on O from the matrices already read (no further Hecke call)
    const uint32_t o = B.ord_rank;
    B.Tell = SMat(o, o);
    SMat Tp0(o, o);
    for (uint32_t j = 0; j < o; ++j) {
        SVec cb, cw, ci, co;
        ENGINE_CHECK(Bk.coords(R, B.O.b[j], cb));
        const SVec img = LinComb(R, Bk.b, bm::MatVec(R, A, cb), len);
        ENGINE_CHECK_MSG(B.O.coords(R, img, ci), "msym::OrdinaryPart: T_ell does not preserve the ordinary part");
        ENGINE_CHECK(S.W.coords(R, B.O.b[j], cw));
        const SVec img2 = LinComb(R, S.W.b, bm::MatVec(R, T0, cw), len);
        ENGINE_CHECK_MSG(B.O.coords(R, img2, co), "msym::OrdinaryPart: T_{p0} does not preserve the ordinary part");
        for (uint32_t i = 0; i < o; ++i) {
            B.Tell.at(i, j) = ci[i];
            Tp0.at(i, j) = co[i];
        }
    }
    B.T[S.ell] = B.Tell;
    B.T[p0] = Tp0;
    // U: the unit root of Y^2 - T_ell Y + ell^{k-1} (Hensel, U <- T_ell - ell^{k-1} U^{-1})
    const uint64_t lk1 = R.pow(R.ell, k - 1);
    SMat U = B.Tell;
    for (uint32_t it = 0; it < 2 * S.R + 4; ++it) {
        SMat Ui;
        ENGINE_CHECK_MSG(InverseUnit(R, U, Ui), "msym::OrdinaryPart: T_ell is not a unit on the ordinary part");
        SMat Un = B.Tell;
        for (size_t i = 0; i < Un.a.size(); ++i) Un.a[i] = R.sub(Un.a[i], R.mul(lk1, Ui.a[i]));
        ++B.hensel_steps;
        if (MatEq(Un, U)) break;
        U = Un;
    }
    B.U = U;
    {
        const SMat U2 = bm::MatMul(R, U, U), TU = bm::MatMul(R, B.Tell, U);
        SMat Z = Sub(R, U2, TU);
        Z = AddScalar(R, Z, lk1);
        B.U_root_ok = true;
        for (uint64_t x : Z.a) B.U_root_ok = B.U_root_ok && x == 0;
    }
    B.U_eq_Tell = MatEq(B.U, B.Tell);
    return B;
}

uint32_t OrdinaryRankAll(SignSpace& S) {
    const bm::Ring& R = S.ring;
    const uint32_t w = S.W.rank();
    if (w == 0) return 0;
    const SMat& A = S.TonW(S.ell);
    const SMat AN = MatPow(R, A, NextPow2AtLeast(static_cast<uint64_t>(S.R) * w));
    std::vector<SVec> cols;
    for (uint32_t j = 0; j < w; ++j) cols.push_back(Column(AN, j));
    const bm::PivotBasis PB = bm::SaturatedBasis(R, cols);
    ENGINE_CHECK_MSG(PB.free_summand, "msym::OrdinaryRankAll: the ordinary part is not free");
    return static_cast<uint32_t>(PB.basis.size());
}

namespace {
SMat OnBlock(const SignSpace& S, const OrdBlock& B, uint64_t p) {
    const uint32_t o = B.O.rank();
    SMat A(o, o);
    for (uint32_t j = 0; j < o; ++j) {
        SVec c;
        ENGINE_CHECK_MSG(B.O.coords(S.ring, S.ApplyT(B.O.b[j], p), c), "msym::OnBlock: T_p does not preserve the block");
        for (uint32_t i = 0; i < o; ++i) A.at(i, j) = c[i];
    }
    return A;
}
} // namespace

// ===========================================================================
// masses
// ===========================================================================
OperatorMeasure::OperatorMeasure(const bm::Ring& R, const MassTables& T, const SMat& Uop, bool ell_n_factor) : R_(R), T_(&T) {
    SMat Ui;
    ENGINE_CHECK_MSG(InverseUnit(R, Uop, Ui), "msym::OperatorMeasure: U is not a unit");
    Uinv_.push_back(Identity(R, Uop.r));
    for (uint32_t m = 1; m <= T.M + 1; ++m) Uinv_.push_back(bm::MatMul(R, Uinv_.back(), Ui));
    lk2_ = ell_n_factor ? R.pow(R.ell, T.k - 2) : 1 % R.q;
}

uint64_t OperatorMeasure::ball(const SVec& x, uint64_t a, uint32_t m) const {
    ENGINE_CHECK(m <= T_->M);
    const SVec y = bm::MatVec(R_, Uinv_[m], x), z = bm::MatVec(R_, Uinv_[m + 1], x);
    uint64_t s = 0;
    for (size_t j = 0; j < T_->c.size(); ++j) {
        const auto& cur = T_->c[j][m + 1];
        const auto& prev = T_->c[j][m];
        if (y[j]) s = R_.add(s, R_.mul(y[j], cur[a % cur.size()] % R_.q));
        if (z[j] && lk2_) s = R_.sub(s, R_.mul(lk2_, R_.mul(z[j], prev[a % prev.size()] % R_.q)));
    }
    return s;
}

uint32_t OperatorMeasure::distribution_failures(const SVec& x) const {
    uint32_t bad = 0;
    for (uint32_t m = 0; m < T_->M; ++m) {
        const uint64_t Mm = pow_wrap(R_.ell, m);
        for (uint64_t a = 0; a < Mm; ++a) {
            uint64_t s = 0;
            for (uint64_t b = 0; b < R_.ell; ++b) s = R_.add(s, ball(x, a + b * Mm, m + 1));
            if (s != ball(x, a, m)) ++bad;
        }
    }
    return bad;
}

namespace {
std::vector<Ball> UnitBalls(uint64_t ell, uint32_t M) {
    std::vector<Ball> out;
    for (uint32_t m = 1; m <= M; ++m) {
        const uint64_t Mm = pow_wrap(ell, m);
        for (uint64_t b = 1; b < Mm; ++b)
            if (b % ell) out.push_back({b, m});
    }
    return out;
}
} // namespace

// ===========================================================================
// h_k
// ===========================================================================
SVec HRing::mul(const SVec& x, const SVec& y) const {
    const uint32_t d = r();
    SVec prod(2 * d, 0);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) prod[i + j] = R.add(prod[i + j], R.mul(x[i] % R.q, y[j] % R.q));
    for (uint32_t i = 2 * d; i-- > d;) {
        const uint64_t c = prod[i];
        if (!c) continue;
        prod[i] = 0;
        for (uint32_t t = 0; t < d; ++t) prod[i - d + t] = R.sub(prod[i - d + t], R.mul(c, F[t] % R.q));
    }
    prod.resize(d);
    return prod;
}

bool HRing::inv(const SVec& x, SVec& out) const {
    const uint32_t d = r();
    SMat Mx(d, d);   // multiplication by x on the basis Y^j
    for (uint32_t j = 0; j < d; ++j) {
        SVec e(d, 0);
        e[j] = 1 % R.q;
        const SVec c = mul(x, e);
        for (uint32_t i = 0; i < d; ++i) Mx.at(i, j) = c[i];
    }
    SMat Mi;
    if (!InverseUnit(R, Mx, Mi)) return false;
    SVec one(d, 0);
    one[0] = 1 % R.q;
    out = bm::MatVec(R, Mi, one);
    return true;
}

bool HRing::unit(const SVec& x) const {
    for (uint64_t c : x)
        if (c % R.ell) return true;
    return false;
}

SVec HRing::trace_gram() const {
    const uint32_t d = r();
    // power sums s_t = Tr(Y^t), t < 2d, by Newton's identities for F monic (e_i from F)
    SVec s(2 * d, 0);
    // Tr(Y^t) as the trace of the companion matrix power
    SMat C(d, d);
    for (uint32_t i = 1; i < d; ++i) C.at(i, i - 1) = 1 % R.q;
    for (uint32_t i = 0; i < d; ++i) C.at(i, d - 1) = R.neg(F[i] % R.q);
    SMat P = Identity(R, d);
    for (uint32_t t = 0; t < 2 * d; ++t) {
        uint64_t tr = 0;
        for (uint32_t i = 0; i < d; ++i) tr = R.add(tr, P.at(i, i));
        s[t] = tr;
        P = bm::MatMul(R, P, C);
    }
    SVec G(static_cast<size_t>(d) * d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) G[static_cast<size_t>(i) * d + j] = s[i + j];
    return G;
}

bool HValuedMass(const HRing& h, const SVec& mus, SVec& out) {
    const uint32_t d = h.r();
    const SVec G = h.trace_gram();
    SMat Gm(d, d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) Gm.at(i, j) = G[static_cast<size_t>(i) * d + j];
    SMat Gi;
    if (!InverseUnit(h.R, Gm, Gi)) return false;
    out = bm::MatVec(h.R, Gi, Reduce(mus, h.R.q));
    return true;
}

// ===========================================================================
// one reading
// ===========================================================================
Reading ReadComponent(const ReadingRequest& req) {
    const auto t0 = std::chrono::steady_clock::now();
    Reading rd;
    rd.req = req;
    const uint64_t ell = req.ell;
    const uint32_t k = req.k, n = k - 2;
    rd.n = n;
    const bool p0side = req.eis && req.c_sign < 0;
    const uint32_t D = p0side ? req.D : 0;
    // precision: the p_0 sign of m_Eis carries D extra digits for the division by ell^d
    uint32_t Rt = req.Kp + D;
    ENGINE_CHECK_MSG(Rt <= MaxPrec(ell), "msym::ReadComponent: ell^{K' + D} >= 2^62");
    rd.R = Rt;
    std::shared_ptr<SignSpace> S = req.space;
    if (S)
        ENGINE_CHECK_MSG(S->n == n && S->c_sign == req.c_sign && S->R == Rt && S->wrong_adjoint == req.wrong_adjoint,
                         "msym::ReadComponent: the cached sign space does not match the request");
    else
        S = std::make_shared<SignSpace>(BuildSignSpace(ell, n, req.c_sign, Rt, req.Emax, req.wrong_adjoint));
    rd.E = S->E;
    rd.space = S;
    auto B = std::make_shared<OrdBlock>(OrdinaryPart(*S, k, req.p0, req.g, p0side));
    rd.block = B;
    rd.w_rank = B->w_rank;
    rd.g_rank = B->g_rank;
    rd.ord_rank = B->ord_rank;
    rd.ord_free = B->ord_free;
    rd.has_p0 = B->has_p0;
    rd.p0_in_block = B->p0_in_block;
    rd.U_eq_Tell = B->U_eq_Tell;
    rd.U_root_ok = B->U_root_ok;
    rd.lk2_vanishes = n >= Rt;   // ell^{k-2} == 0 mod ell^R
    if (!B->refusal.empty()) {
        rd.refusal = B->refusal;
        rd.seconds = Secs(t0);
        return rd;
    }
    const bm::Ring& R = S->ring;
    const uint32_t off = B->has_p0 ? 1 : 0;
    const uint32_t o = B->ord_rank;
    rd.r = o - off;
    if (rd.r == 0) {
        rd.refusal = "S_zero";
        rd.seconds = Secs(t0);
        return rd;
    }
    const uint32_t r = rd.r;
    const SMat& Tp0O = B->T.at(req.p0);
    // T_{p0} on S (the quotient by Z p_0 when has_p0: p_0 is an eigenvector, first column (e, 0, ..))
    SMat TS(r, r);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) TS.at(i, j) = Tp0O.at(i + off, j + off);
    if (B->has_p0) {
        const uint64_t e = R.add(1 % R.q, R.pow(req.p0, k - 1));
        bool eig = Tp0O.at(0, 0) == e;
        for (uint32_t i = 1; i < o; ++i) eig = eig && Tp0O.at(i, 0) == 0;
        ENGINE_CHECK_MSG(eig, "msym::ReadComponent: p_0 is not a T_{p0}-eigenvector with eigenvalue 1 + p0^{k-1}");
    }
    // (BF) at a non-Eisenstein component: m != m_b for every b mod ell - 1, read on T_2, T_3, T_5 mod ell
    const bm::Ring Rk = bm::MakeRing(ell, req.Kp);
    if (!req.eis) {
        rd.bf_checked = true;
        rd.bf_holds = true;
        std::map<uint64_t, SMat> small;
        for (uint64_t p : {2ull, 3ull, 5ull}) {
            if (p == ell) continue;
            const SMat A = p == req.p0 ? Tp0O : OnBlock(*S, *B, p);
            SMat AS(r, r);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) AS.at(i, j) = A.at(i + off, j + off);
            small[p] = AS;
        }
        for (uint64_t b = 0; b + 1 < ell && rd.bf_holds; ++b) {
            bool all = true;
            for (const auto& kv : small) {
                const uint64_t p = kv.first;
                const uint64_t e1 = b % (ell - 1), e2 = ((n + 1) % (ell - 1) + (ell - 1) - b % (ell - 1)) % (ell - 1);
                const uint64_t chi = (PowModU(p, e1, ell) + PowModU(p, e2, ell)) % ell;
                all = all && NilpotentModEll(R, AddScalar(R, kv.second, R.neg(chi % R.q)));
            }
            if (all) {
                rd.bf_holds = false;
                rd.bf_fail_b = static_cast<int>(b);
            }
        }
        if (!rd.bf_holds) {
            rd.refusal = "boundary_component";
            rd.seconds = Secs(t0);
            return rd;
        }
    }
    // the eigensymbol phi, in O coordinates (mod ell^R), then reduced mod ell^{K'}
    SVec phiO(o, 0);
    if (B->has_p0) {
        ENGINE_CHECK_MSG(r == 1, "msym::ReadComponent: m_Eis with r != 1");
        const uint64_t e = Tp0O.at(0, 0), y = Tp0O.at(0, 1), a = Tp0O.at(1, 1);
        const uint64_t am = R.sub(a, e);
        const uint32_t vy = R.val(y), va = R.val(am);
        const uint32_t c = std::min(vy, va);
        ENGINE_CHECK_MSG(c < Rt, "msym::ReadComponent: the cusp line is not read at this precision");
        const uint64_t pc = R.ell_pow(c);
        // r = (T - e) e_1 / ell^c (valid mod ell^{R - c}), primitive
        SVec rr{(y / pc) % R.q, (am / pc) % R.q};
        rd.content = c;
        const uint32_t d = std::min<uint32_t>(R.val(rr[1]), Rt - c);
        rd.depth_d = d;
        if (Rt < c + d + req.Kp) {
            rd.refusal = "depth_beyond_precision";
            rd.seconds = Secs(t0);
            return rd;
        }
        const uint64_t u = rr[0];
        ENGINE_CHECK_MSG(d == 0 || R.unit(u), "msym::ReadComponent: r is not congruent to a unit multiple of p_0");
        // w = (r - u p_0) / ell^d: valid mod ell^{R - c - d} >= ell^{K'}
        SVec wv{0, (rr[1] / R.ell_pow(d)) % R.q};
        rd.r_sym = Reduce(rr, Rk.q);
        rd.w_sym = Reduce(wv, Rk.q);
        phiO = wv;
    } else {
        bool found = false;
        for (uint32_t j = 0; j < r && !found; ++j) {
            SVec v(r, 0);
            v[j] = 1 % R.q;
            std::vector<SVec> cols{v};
            for (uint32_t i = 1; i < r; ++i) cols.push_back(bm::MatVec(R, TS, cols.back()));
            SMat K = FromColumns(cols, r), Ki;
            if (InverseUnit(R, K, Ki)) {
                found = true;
                for (uint32_t i = 0; i < r; ++i) phiO[i + off] = v[i];
            }
        }
        if (!found) {
            rd.refusal = "not_cyclic";
            rd.seconds = Secs(t0);
            return rd;
        }
    }
    // ---- everything below mod ell^{K'}
    const uint32_t Kp = req.Kp;
    std::vector<SVec> Ob;
    for (const SVec& b : B->O.b) Ob.push_back(Reduce(b, Rk.q));
    const FreeBasis OK = MakeFreeBasis(Rk, Ob);
    const SymbolHecke HK(n, Rk);
    const SMat TSk = Reduce(TS, Rk.q);
    rd.phi = Reduce(phiO, Rk.q);
    SVec phiS(r);
    for (uint32_t i = 0; i < r; ++i) phiS[i] = rd.phi[i + off];
    std::vector<SVec> yS{phiS};
    for (uint32_t i = 1; i < r; ++i) yS.push_back(bm::MatVec(Rk, TSk, yS.back()));
    const SMat Y = FromColumns(yS, r);
    SMat Yi;
    ENGINE_CHECK_MSG(InverseUnit(Rk, Y, Yi), "msym::ReadComponent: the cyclic basis is not a basis mod ell^{K'}");
    rd.TS_p0 = bm::MatMul(Rk, Yi, bm::MatMul(Rk, TSk, Y));   // the companion form
    rd.F = bm::CharPoly(Rk, TSk);
    // T_p on S for every p <= B (p != ell), applied to the r vectors y_i only, and U
    auto toO = [&](const SVec& s) {
        SVec x(o, 0);
        for (uint32_t i = 0; i < r; ++i) x[i + off] = s[i];
        return x;
    };
    std::vector<SVec> yAmb;
    for (const SVec& s : yS) yAmb.push_back(LinComb(Rk, OK.b, toO(s), n + 1));
    rd.P_consistent = true;
    auto readPoly = [&](const SMat& My) {
        const SVec c = Column(My, 0);
        const SMat poly = MatPolyOf(Rk, c, rd.TS_p0);
        if (!MatEq(poly, My)) rd.P_consistent = false;
        return c;
    };
    std::vector<uint64_t> ops = req.primes;
    if (std::find(ops.begin(), ops.end(), req.p0) == ops.end()) ops.push_back(req.p0);
    for (uint64_t p : ops) {
        SMat My(r, r);
        for (uint32_t i = 0; i < r; ++i) {
            SVec c;
            ENGINE_CHECK_MSG(OK.coords(Rk, HK.Tp(yAmb[i], p), c), "msym::ReadComponent: T_p does not preserve the ordinary block");
            SVec s(r);
            for (uint32_t t = 0; t < r; ++t) s[t] = c[t + off];
            const SVec sy = bm::MatVec(Rk, Yi, s);
            for (uint32_t t = 0; t < r; ++t) My.at(t, i) = sy[t];
        }
        rd.P[p] = readPoly(My);
    }
    {
        const SMat Uk = Reduce(B->U, Rk.q);
        SMat US(r, r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) US.at(i, j) = Uk.at(i + off, j + off);
        rd.P[ell] = readPoly(bm::MatMul(Rk, Yi, bm::MatMul(Rk, US, Y)));
    }
    // ---- masses
    {
        // the c-tables of the O basis mod ell^{K'} (linear in the symbol)
        auto tables = std::make_shared<MassTables>();
        tables->M = req.M;
        tables->k = k;
        {
            const bm::SymRho rho(n, Rk);
            for (const SVec& b : OK.b) tables->c.push_back(bm::BuildMeasure(rho, b, 1, 1, req.M, true).c);
        }
        rd.tables = tables;
        const OperatorMeasure mu(Rk, *tables, Reduce(B->U, Rk.q));
        rd.balls = UnitBalls(ell, req.M);
        const HRing h{Rk, rd.F};
        std::vector<SVec> yO;
        for (const SVec& s : yS) yO.push_back(toO(s));
        for (const SVec& x : yO) rd.dist_failures += mu.distribution_failures(x);
        rd.gram_unit = true;
        for (const Ball& bl : rd.balls) {
            SVec mus(r);
            for (uint32_t i = 0; i < r; ++i) mus[i] = mu.ball(yO[i], bl.b, bl.M);
            SVec mh;
            if (!HValuedMass(h, mus, mh)) {
                rd.gram_unit = false;
                mh.assign(r, 0);
            }
            rd.muh.push_back(mh);
        }
        // G7: mu_{p_0} with U = 1 on p_0 (its T_ell-eigenvalue 1 + ell^{k-1}) vanishes on every unit ball
        const bm::SymRho rho(n, Rk);
        const lvalue::ModularSymbolMeasure m0 =
            bm::BuildMeasure(rho, P0Vector(Rk, n), Rk.add(1 % Rk.q, Rk.pow(ell, k - 1)), 1, req.M, true);
        for (const Ball& bl : rd.balls)
            if (m0.ball(bl.b, bl.M) != 0) ++rd.p0_unit_mass_nonzero;
    }
    (void)Kp;
    rd.seconds = Secs(t0);
    return rd;
}

std::vector<uint64_t> UnitMasses(const Reading& rd, const SVec& x_O, const SMat* Uoverride) {
    const bm::Ring Rk = bm::MakeRing(rd.req.ell, rd.req.Kp);
    const SMat U = Uoverride ? *Uoverride : Reduce(rd.block->U, Rk.q);
    const OperatorMeasure mu(Rk, *rd.tables, U);
    std::vector<uint64_t> out;
    for (const Ball& bl : rd.balls) out.push_back(mu.ball(Reduce(x_O, Rk.q), bl.b, bl.M));
    return out;
}

// ===========================================================================
// normalisation, interpolation
// ===========================================================================
int FirstUnitBall(const Reading& c) {
    const HRing h{bm::MakeRing(c.req.ell, c.req.Kp), c.F};
    for (size_t i = 0; i < c.muh.size(); ++i)
        if (h.unit(c.muh[i])) return static_cast<int>(i);
    return -1;
}

bool Normalised(const Reading& rd, int b0, std::vector<SVec>& nu) {
    ENGINE_CHECK(b0 >= 0 && static_cast<size_t>(b0) < rd.muh.size());
    const HRing h{bm::MakeRing(rd.req.ell, rd.req.Kp), rd.F};
    SVec iv;
    if (!h.inv(rd.muh[static_cast<size_t>(b0)], iv)) return false;
    nu.clear();
    for (const SVec& m : rd.muh) nu.push_back(h.mul(m, iv));
    return true;
}

std::vector<uint64_t> NewtonWeights(uint64_t ell, const std::vector<uint32_t>& weights, uint32_t Kp, uint32_t kprime) {
    const size_t N = weights.size();
    ENGINE_CHECK_MSG(N >= 1 && N <= ell, "msym::NewtonWeights: shape");
    for (size_t j = 1; j < N; ++j)
        ENGINE_CHECK_MSG(weights[j] == weights[0] + j * (ell - 1), "msym::NewtonWeights: weights not consecutive in one class");
    // B_i(x') and the denominators prod_{b != a} (x_a - x_b) are needed mod ell^{Kp + i}, i < N (WP-HL's ZMod, ell^W < 2^84)
    const ZMod Q(ell, Kp + static_cast<uint32_t>(N) - 1);
    const bm::Ring RK = bm::MakeRing(ell, Kp);
    std::vector<u128> xs;
    for (uint32_t k : weights) xs.push_back(Q.sub(Q.pow((ell + 1) % Q.M, k), 1 % Q.M));
    const u128 xp = Q.sub(Q.pow((ell + 1) % Q.M, kprime), 1 % Q.M);
    std::vector<uint64_t> bi(N);
    {
        u128 Bi = 1 % Q.M;
        for (size_t i = 0; i < N; ++i) {
            if (i) Bi = Q.mul(Bi, Q.sub(xp, xs[i - 1]));
            ENGINE_CHECK_MSG(Q.val(Bi, Q.W) >= i, "msym::NewtonWeights: B_i(x') not divisible by ell^i");
            bi[i] = static_cast<uint64_t>((Bi / Q.pw[i]) % RK.q);
        }
    }
    std::vector<uint64_t> lam(N, 0);
    for (size_t i = 0; i < N; ++i)
        for (size_t a = 0; a <= i; ++a) {
            u128 pr = 1 % Q.M;
            for (size_t b = 0; b <= i; ++b)
                if (b != a) pr = Q.mul(pr, Q.sub(xs[a], xs[b]));
            ENGINE_CHECK_MSG(Q.val(pr, Q.W) == i, "msym::NewtonWeights: a denominator of valuation != i (weights not consecutive)");
            const uint64_t unit = static_cast<uint64_t>((pr / Q.pw[i]) % RK.q);
            lam[a] = RK.add(lam[a], RK.mul(RK.inv(unit), bi[i]));
        }
    return lam;
}

TruncationMap MakeTruncationMap(uint64_t ell, const std::vector<uint32_t>& weights, uint32_t Kp, uint32_t K, uint32_t m,
                                uint32_t centre_k) {
    const size_t N = weights.size();
    ENGINE_CHECK_MSG(N >= 1 && K >= 1 && m >= 1, "msym::MakeTruncationMap: shape");
    for (size_t j = 1; j < N; ++j)
        ENGINE_CHECK_MSG(weights[j] == weights[0] + j * (ell - 1), "msym::MakeTruncationMap REFUSED: weights not consecutive in one class");
    ENGINE_CHECK_MSG(N <= ell, "msym::MakeTruncationMap REFUSED: N > ell");
    ENGINE_CHECK_MSG(N >= K + m - 1, "msym::MakeTruncationMap REFUSED: N < K + m - 1 weights");
    ENGINE_CHECK_MSG(Kp >= K + m - 1, "msym::MakeTruncationMap REFUSED: values mod ell^Kp, Kp < K + m - 1");
    TruncationMap T;
    T.ell = ell;
    T.K = K;
    T.m = m;
    T.Kp = Kp;
    T.centre_k = centre_k;
    T.weights = weights;
    const uint32_t Wt = K + m - 1;                                  // the target modulus before the division by ell^{m-1}
    const ZMod Q(ell, Wt + static_cast<uint32_t>(N) + 1), RT(ell, Wt);
    std::vector<u128> xs;
    for (uint32_t k : weights) xs.push_back(Q.sub(Q.pow((ell + 1) % Q.M, k), 1 % Q.M));
    const u128 xc = Q.sub(Q.pow((ell + 1) % Q.M, centre_k), 1 % Q.M);
    // w_{i,t} = ell^{m-1} e_{i-t}(x_c - x_j : j < i) / ell^i, integral (v e_{i-t} >= i - t, t < m)
    std::vector<std::vector<u128>> w(N, std::vector<u128>(m, 0));
    for (size_t i = 0; i < N; ++i) {
        std::vector<u128> e(i + 1, 0);   // elementary symmetric polynomials of d_j = x_c - x_j, j < i
        e[0] = 1 % Q.M;
        for (size_t j = 0; j < i; ++j) {
            const u128 d = Q.sub(xc, xs[j]);
            for (size_t s = j + 1; s >= 1; --s) e[s] = Q.add(e[s], Q.mul(e[s - 1], d));
        }
        for (uint32_t t = 0; t < m && t <= i; ++t) {
            u128 x = e[i - t];
            // multiply by ell^{m-1} and divide by ell^i exactly
            const uint32_t v = Q.val(x, Q.W);
            ENGINE_CHECK_MSG(v + (m - 1) >= i || x == 0, "msym::MakeTruncationMap: e_{i-t} of too small a valuation");
            if (m - 1 >= i)
                x = Q.mul(x, Q.pw[m - 1 - i]);
            else
                x = x / Q.pw[i - (m - 1)];
            w[i][t] = x % RT.M;
        }
    }
    // Lam_{t,a} = sum_{i >= a} u_{a,i}^{-1} w_{i,t}
    T.lam.assign(m, std::vector<u128>(N, 0));
    for (size_t i = 0; i < N; ++i)
        for (size_t a = 0; a <= i; ++a) {
            u128 pr = 1 % Q.M;
            for (size_t b = 0; b <= i; ++b)
                if (b != a) pr = Q.mul(pr, Q.sub(xs[a], xs[b]));
            ENGINE_CHECK_MSG(Q.val(pr, Q.W) == i, "msym::MakeTruncationMap: a denominator of valuation != i");
            const u128 ui = RT.inv((pr / Q.pw[i]) % RT.M);
            for (uint32_t t = 0; t < m; ++t) T.lam[t][a] = RT.add(T.lam[t][a], RT.mul(ui, w[i][t]));
        }
    return T;
}

bool TruncationMap::Apply(const std::vector<uint64_t>& values, Vec& out) const {
    ENGINE_CHECK_MSG(values.size() == weights.size(), "msym::TruncationMap::Apply: shape");
    const ZMod RT(ell, K + m - 1), RK(ell, K);
    const u128 qv = RT.pw[Kp > K + m - 1 ? K + m - 1 : Kp];
    out.assign(m, 0);
    for (uint32_t t = 0; t < m; ++t) {
        u128 s = 0;
        for (size_t a = 0; a < values.size(); ++a) s = RT.add(s, RT.mul(static_cast<u128>(values[a]) % qv % RT.M, lam[t][a]));
        if (RT.val(s, RT.W) < m - 1) return false;
        out[t] = (s / RT.pw[m - 1]) % RK.M;
    }
    return true;
}

// ===========================================================================
// (U17)(a): the gluing depth
// ===========================================================================
namespace {
namespace hp = hecke_ext::lvalue::hidapair;

hp::M22 ToM22(const bm::Mat2& g) { return hp::M22{g.a, g.b, g.c, g.d}; }

// T_p on a D-fibre (or V-fibre) modular symbol by Manin (WP-BIL's hecke_symbol, through FibreOps)
SVec HeckeSymbolFibre(const hp::FibreOps& F, hp::Fibre f, const SVec& P, uint64_t p) {
    const hp::Zq& Z = F.R();
    SVec out = F.on(f, hp::M22{1, 0, 0, static_cast<int64_t>(p)}, P);
    for (uint64_t b = 0; b < p; ++b) {
        SVec acc(P.size(), 0);
        for (const bm::Mat2& g : bm::ManinPath(static_cast<int64_t>(b), static_cast<int64_t>(p))) acc = F.add(acc, F.on(f, ToM22(g), P));
        const SVec phi = F.scale(Z.neg(1 % Z.q), acc);
        out = F.add(out, F.on(f, hp::M22{static_cast<int64_t>(p), -static_cast<int64_t>(b), 0, 1}, phi));
    }
    return out;
}

// The rank-2 block reading: the ordinary m_Eis block O = {first vector, second} with T_2 = [[e, y], [0, a]] in a basis whose first
// vector spans the Eisenstein summand -- j = v(a - e) - min(v(y), v(a - e)) is the index of A_1 + A_2 in O.
void ReadRankTwo(const bm::Ring& R, const SMat& T2, uint64_t e, GluingReading& g) {
    const uint64_t y = T2.at(0, 1), a = T2.at(1, 1);
    const uint64_t am = R.sub(a, e);
    g.v_a_minus_e = R.val(am);
    g.pivot = std::min(R.val(y), g.v_a_minus_e);
    g.j = g.v_a_minus_e - g.pivot;
}
}  // namespace

GluingReading GluingDual(uint32_t n, uint64_t ell, uint32_t M) {
    const auto t0 = std::chrono::steady_clock::now();
    GluingReading g;
    g.n = n;
    g.M = M;
    g.route = "dual-D";
    const uint32_t E = std::min<uint32_t>(2, MaxPrec(ell) - M);
    const hp::Zq Zh(ell, M + E), Z(ell, M);
    const hp::FibreOps Fh(Zh, n), F(Z, n);
    const bm::Ring R = bm::MakeRing(ell, M), Rh = bm::MakeRing(ell, M + E);
    const hp::Fibre D = hp::Fibre::D;
    // W_D^{geo-even} (dual to H^1(Gamma; V)^{[C]+} under the iota-anti-isometric pairing): the image of W_D(R+E) -> W_D(R)
    std::vector<uint32_t> cols;
    for (uint32_t t = 0; t <= n; t += 2) cols.push_back(t);
    bm::Mat A(2 * (n + 1), static_cast<uint32_t>(cols.size()));
    for (uint32_t j = 0; j < cols.size(); ++j) {
        SVec e(n + 1, 0);
        e[cols[j]] = 1;
        const SVec s = Fh.on(D, hp::M22{0, -1, 1, 0}, e);
        const SVec u1 = Fh.on(D, hp::M22{0, -1, 1, 1}, e), u2 = Fh.on(D, hp::M22{0, -1, 1, 1}, u1);
        for (uint32_t t = 0; t <= n; ++t) {
            A.at(t, j) = Rh.add(e[t], s[t]);
            A.at(n + 1 + t, j) = Rh.add(Rh.add(e[t], u1[t]), u2[t]);
        }
    }
    std::vector<SVec> gens, red;
    std::vector<uint32_t> exps;
    bm::KernelGenerators(Rh, A, gens, exps);
    for (const SVec& gv : gens) {
        SVec v(n + 1, 0);
        for (uint32_t j = 0; j < cols.size(); ++j) v[cols[j]] = gv[j] % R.q;
        if (!IsZero(v)) red.push_back(v);
    }
    const bm::PivotBasis PB = bm::SaturatedBasis(R, red);
    if (!PB.free_summand) {
        g.refusal = "W_D_not_free";
        g.seconds = Secs(t0);
        return g;
    }
    const FreeBasis W = MakeFreeBasis(R, PB.basis);
    const uint32_t w = W.rank();
    SMat T2(w, w);
    for (uint32_t j = 0; j < w; ++j) {
        SVec c;
        ENGINE_CHECK_MSG(W.coords(R, HeckeSymbolFibre(F, D, W.b[j], 2), c), "msym::GluingDual: T_2 W_D not in W_D");
        for (uint32_t i = 0; i < w; ++i) T2.at(i, j) = c[i];
    }
    const uint64_t e2 = R.add(1, R.pow(2, n + 1));
    const SMat G = MatPow(R, AddScalar(R, T2, R.neg(e2)), NextPow2AtLeast(static_cast<uint64_t>(M) * w));
    std::vector<SVec> kg, blockv;
    std::vector<uint32_t> ke;
    bm::KernelGenerators(R, G, kg, ke);
    for (size_t i = 0; i < kg.size(); ++i) {
        if (ke[i] != M) {
            g.refusal = "block_not_free";
            g.seconds = Secs(t0);
            return g;
        }
        blockv.push_back(LinComb(R, W.b, kg[i], n + 1));
    }
    g.block_rank = static_cast<uint32_t>(blockv.size());
    if (blockv.empty()) {
        g.refusal = "block_zero";
        g.seconds = Secs(t0);
        return g;
    }
    const FreeBasis Bk = MakeFreeBasis(R, blockv);
    SMat Al(g.block_rank, g.block_rank);
    for (uint32_t j = 0; j < g.block_rank; ++j) {
        SVec c;
        ENGINE_CHECK_MSG(Bk.coords(R, HeckeSymbolFibre(F, D, Bk.b[j], ell), c), "msym::GluingDual: T_ell does not preserve the block");
        for (uint32_t i = 0; i < g.block_rank; ++i) Al.at(i, j) = c[i];
    }
    const SMat AN = MatPow(R, Al, NextPow2AtLeast(static_cast<uint64_t>(M) * g.block_rank));
    std::vector<SVec> imv;
    for (uint32_t j = 0; j < g.block_rank; ++j) imv.push_back(LinComb(R, Bk.b, Column(AN, j), n + 1));
    const bm::PivotBasis PO = bm::SaturatedBasis(R, imv);
    g.ord_rank = static_cast<uint32_t>(PO.basis.size());
    if (!PO.free_summand || g.ord_rank != 2) {
        g.refusal = PO.free_summand ? "ordinary_block_rank_" + std::to_string(g.ord_rank) : "ordinary_block_not_free";
        g.seconds = Secs(t0);
        return g;
    }
    // p_0^D = u_0 - u_n (the boundary symbol of D, Eisenstein eigenvalues) first
    const FreeBasis O0 = MakeFreeBasis(R, PO.basis);
    SVec c;
    if (!O0.coords(R, P0Vector(R, n), c)) {
        g.refusal = "p0_not_in_block";
        g.seconds = Secs(t0);
        return g;
    }
    const uint32_t iu = R.unit(c[0]) ? 0 : 1;
    ENGINE_CHECK_MSG(R.unit(c[iu]), "msym::GluingDual: p_0 not primitive");
    const FreeBasis O = MakeFreeBasis(R, {P0Vector(R, n), PO.basis[1 - iu]});
    SMat T2O(2, 2);
    for (uint32_t j = 0; j < 2; ++j) {
        SVec cc;
        ENGINE_CHECK(O.coords(R, HeckeSymbolFibre(F, D, O.b[j], 2), cc));
        for (uint32_t i = 0; i < 2; ++i) T2O.at(i, j) = cc[i];
    }
    ENGINE_CHECK_MSG(T2O.at(0, 0) == e2 && T2O.at(1, 0) == 0, "msym::GluingDual: p_0^D is not Eisenstein for T_2");
    ReadRankTwo(R, T2O, e2, g);
    g.seconds = Secs(t0);
    return g;
}

namespace {
// H^1(Gamma; V) = V / (V^S + V^U) (WP-BIL's amalgam presentation on the V fibre), its free part over Z/ell^M, with T_p and
// [C]'s iota^* = - iota^*_geo through the group-cocycle formulas of hidapair::HidaPairingCell (hecke_y, iota_y).
struct AmalgamH1 {
    hp::HidaPairingCell cell;
    bm::Ring R;
    uint32_t n = 0, M = 0, np = 0, h = 0;
    bm::SmithForm SF;
    std::vector<SVec> lifts;
    AmalgamH1(uint32_t n_, uint64_t ell, uint32_t M_) : cell(n_, ell, M_, {}), R(bm::MakeRing(ell, M_)), n(n_), M(M_) {
        const hp::FibreOps& F = cell.F();
        std::vector<SVec> ngens;
        for (const hp::M22& gm : {hp::M22{0, -1, 1, 0}, hp::M22{0, -1, 1, 1}}) {
            bm::Mat A(n + 1, n + 1);
            for (uint32_t j = 0; j <= n; ++j) {
                SVec e(n + 1, 0);
                e[j] = 1;
                const SVec x = F.on(hp::Fibre::V, gm, e);
                for (uint32_t i = 0; i <= n; ++i) A.at(i, j) = R.sub(x[i], e[i]);
            }
            std::vector<SVec> gs;
            std::vector<uint32_t> ex;
            bm::KernelGenerators(R, A, gs, ex);
            for (const SVec& v : gs) ngens.push_back(v);
        }
        SF = bm::Smith(R, FromColumns(ngens, n + 1), true);
        np = static_cast<uint32_t>(SF.v.size());
        h = n + 1 - np;
        SMat Pinv;
        ENGINE_CHECK(InverseUnit(R, SF.P, Pinv));
        for (uint32_t i = np; i <= n; ++i) lifts.push_back(Column(Pinv, i));
    }
    SVec hcoords(const SVec& y) const {
        const SVec z = bm::MatVec(R, SF.P, y);
        return SVec(z.begin() + np, z.end());
    }
    SVec lift(const SVec& x) const { return LinComb(R, lifts, x, n + 1); }
    SVec Tp(const SVec& x, uint64_t p) const { return hcoords(cell.hecke_y(hp::Fibre::V, lift(x), static_cast<uint32_t>(p))); }
    SVec iotaC(const SVec& x) const {
        SVec y = hcoords(cell.iota_y(hp::Fibre::V, lift(x)));
        for (auto& c : y) c = R.neg(c);
        return y;
    }
    // a saturated basis of the [C]-sign part, in H coordinates
    FreeBasis SignPart(int sign) const {
        const uint64_t half = R.inv(2);
        std::vector<SVec> cols;
        for (uint32_t j = 0; j < h; ++j) {
            SVec e(h, 0);
            e[j] = 1;
            const SVec ie = iotaC(e);
            SVec v(h);
            for (uint32_t i = 0; i < h; ++i) v[i] = R.mul(half, sign > 0 ? R.add(e[i], ie[i]) : R.sub(e[i], ie[i]));
            cols.push_back(v);
        }
        const bm::PivotBasis PB = bm::SaturatedBasis(R, cols);
        ENGINE_CHECK_MSG(PB.free_summand, "msym::AmalgamH1: a sign part of H^1 is not free");
        return MakeFreeBasis(R, PB.basis);
    }
    SMat On(const FreeBasis& B, uint64_t p) const {
        const uint32_t b = B.rank();
        SMat A(b, b);
        for (uint32_t j = 0; j < b; ++j) {
            SVec c;
            ENGINE_CHECK_MSG(B.coords(R, Tp(B.b[j], p), c), "msym::AmalgamH1: T_p does not preserve the module");
            for (uint32_t i = 0; i < b; ++i) A.at(i, j) = c[i];
        }
        return A;
    }
    // im A^N inside B (A = T_ell on B): the ordinary part, in H coordinates
    FreeBasis Ordinary(const FreeBasis& B, const SMat& Aell) const {
        const uint32_t b = B.rank();
        if (b == 0) return B;
        const SMat AN = MatPow(R, Aell, NextPow2AtLeast(static_cast<uint64_t>(M) * b));
        std::vector<SVec> imv;
        for (uint32_t j = 0; j < b; ++j) imv.push_back(LinComb(R, B.b, Column(AN, j), h));
        const bm::PivotBasis PO = bm::SaturatedBasis(R, imv);
        ENGINE_CHECK_MSG(PO.free_summand, "msym::AmalgamH1: the ordinary part is not free");
        return MakeFreeBasis(R, PO.basis);
    }
};
}  // namespace

std::pair<uint32_t, uint32_t> AmalgamOrdinaryRanks(uint32_t n, uint64_t ell, uint32_t M) {
    const AmalgamH1 H(n, ell, M);
    uint32_t out[2] = {0, 0};
    for (int s = 0; s < 2; ++s) {
        const FreeBasis B = H.SignPart(s == 0 ? 1 : -1);
        out[s] = B.rank() ? H.Ordinary(B, H.On(B, ell)).rank() : 0;
    }
    return {out[0], out[1]};
}

GluingReading GluingAmalgam(uint32_t n, uint64_t ell, uint32_t M) {
    const auto t0 = std::chrono::steady_clock::now();
    GluingReading g;
    g.n = n;
    g.M = M;
    g.route = "amalgam-V";
    const AmalgamH1 H(n, ell, M);
    const bm::Ring& R = H.R;
    const uint64_t e2 = R.add(1, R.pow(2, n + 1)), e3 = R.add(1, R.pow(3, n + 1)), e5 = R.add(1, R.pow(5, n + 1));
    const hp::Zq Z(ell, M);
    auto toRows = [&](const SMat& X) {
        hp::Mat out(X.r, hp::Vec(X.c));
        for (uint32_t i = 0; i < X.r; ++i)
            for (uint32_t j = 0; j < X.c; ++j) out[i][j] = X.at(i, j);
        return out;
    };
    // the ordinary m_Eis block of each [C]-sign: WP-LV3's joint cut (T_2, T_3, T_5) at the Eisenstein eigenvalues, then T_ell
    uint32_t orank[2] = {0, 0};
    FreeBasis Oplus;
    for (int s = 0; s < 2; ++s) {
        const FreeBasis Hs = H.SignPart(s == 0 ? 1 : -1);
        if (Hs.rank() == 0) continue;
        const hp::Mat r2 = toRows(H.On(Hs, 2)), r3 = toRows(H.On(Hs, 3)), r5 = toRows(H.On(Hs, 5));
        const lvalue::mblock::Block jb = lvalue::mblock::JointBlockOf(Z, r2, e2, {{&r3, e3}, {&r5, e5}});
        if (s == 0) g.block_rank = static_cast<uint32_t>(jb.basis.size());
        if (jb.basis.empty()) continue;
        std::vector<SVec> bv;
        for (const auto& c : jb.basis) bv.push_back(LinComb(R, Hs.b, SVec(c.begin(), c.end()), H.h));
        const FreeBasis Bk = MakeFreeBasis(R, bv);
        const FreeBasis O = H.Ordinary(Bk, H.On(Bk, ell));
        orank[s] = O.rank();
        if (s == 0) Oplus = O;
    }
    g.ord_rank = orank[0];
    g.eis_in_plus = orank[0] == 2 && orank[1] == 1;
    if (!g.eis_in_plus) {
        g.refusal = "ordinary_eis_ranks_plus_" + std::to_string(orank[0]) + "_minus_" + std::to_string(orank[1]);
        g.seconds = Secs(t0);
        return g;
    }
    // M = the ordinary m_Eis block of H^+ (rank 2): A_2 = the Eisenstein line, A_1 = the cusp line, both saturated in M
    const SMat T2 = H.On(Oplus, 2);
    const uint64_t a = R.sub(R.add(T2.at(0, 0), T2.at(1, 1)), e2);
    g.v_a_minus_e = R.val(R.sub(a, e2));
    auto satLine = [&](uint64_t lam) {
        const SMat Q = AddScalar(R, T2, R.neg(lam));
        SVec best;
        uint32_t bc = M + 1;
        for (uint32_t j = 0; j < 2; ++j) {
            const SVec col = Column(Q, j);
            const uint32_t cv = std::min(R.val(col[0]), R.val(col[1]));
            if (cv < bc) {
                bc = cv;
                best = col;
            }
        }
        ENGINE_CHECK_MSG(bc < M, "msym::GluingAmalgam: an eigenline is not read at this precision");
        const uint64_t pc = R.ell_pow(bc);
        return std::make_pair(SVec{best[0] / pc, best[1] / pc}, bc);
    };
    const auto A1 = satLine(e2), A2 = satLine(a);   // (T - e) M lies on the cusp line, (T - a) M on the Eisenstein line
    const uint64_t det = R.sub(R.mul(A1.first[0], A2.first[1]), R.mul(A1.first[1], A2.first[0]));
    g.j = std::min(R.val(det), M - std::max(A1.second, A2.second));
    const bm::SmithForm sf = bm::Smith(R, AddScalar(R, T2, R.neg(e2)), false);
    g.pivot = sf.v.empty() ? M : sf.v[0];
    // iota_! of the [C]-plus cusp symbol (geometric-odd W, T_2-eigenvalue a) lies on the cusp line of M
    {
        const std::vector<hp::Vec> Wo = H.cell.sign_basis(hp::Fibre::V, 1);
        const FreeBasis WO = MakeFreeBasis(R, std::vector<SVec>(Wo.begin(), Wo.end()));
        const uint32_t wo = WO.rank();
        SMat T2w(wo, wo);
        for (uint32_t j = 0; j < wo; ++j) {
            SVec c;
            ENGINE_CHECK(WO.coords(R, H.cell.hecke_symbol(hp::Fibre::V, WO.b[j], 2), c));
            for (uint32_t i = 0; i < wo; ++i) T2w.at(i, j) = c[i];
        }
        std::vector<SVec> kk;
        std::vector<uint32_t> kexp;
        bm::KernelGenerators(R, AddScalar(R, T2w, R.neg(a)), kk, kexp);
        bool ok = false;
        for (size_t i = 0; i < kk.size() && !ok; ++i) {
            if (kexp[i] != M) continue;
            const SVec P = LinComb(R, WO.b, kk[i], n + 1);
            const SVec y = H.hcoords(H.cell.iota_shriek(hp::Fibre::V, P));
            SVec c;
            if (!Oplus.coords(R, y, c)) continue;
            const SVec t = bm::MatVec(R, AddScalar(R, T2, R.neg(a)), c);
            ok = IsZero(t) && !IsZero(c);
        }
        g.shriek_in_cusp_line = ok;
    }
    g.seconds = Secs(t0);
    return g;
}

// ===========================================================================
// the record
// ===========================================================================
std::string RecordLine(const RecordInput& in) {
    std::ostringstream os, lv2;
    os << "HECKE_LAMBDA_MSYM ell=" << in.ell << " a=" << in.a << " K=" << in.K << " m=" << in.m << " centre_k=" << in.centre_k
       << " N=" << in.N << " component=" << in.component << " sign=" << (in.sign > 0 ? "+" : "-") << " rank=" << in.rank
       << " p0=" << in.p0 << " B0=";
    if (in.b0 >= 0)
        os << in.B0_b << "+ell^" << in.B0_M;
    else
        os << "none";
    os << " unit_check=" << (in.unit_ok ? "ok" : "fail") << " weights=[";
    for (size_t i = 0; i < in.weights.size(); ++i) os << (i ? "," : "") << in.weights[i];
    os << "] nu=[";
    bool first = true;
    uint32_t n2 = 0;
    for (const auto& bn : in.nu) {
        std::ostringstream one;
        one << bn.first.b << "+ell^" << bn.first.M << ":[";
        for (size_t i = 0; i < bn.second.size(); ++i) {
            one << (i ? "," : "") << "[";
            for (size_t t = 0; t < bn.second[i].size(); ++t) one << (t ? "," : "") << U128Text(bn.second[i][t]);
            one << "]";
        }
        one << "]";
        if (bn.first.M == 1) {
            os << (first ? "" : ";") << one.str();
            first = false;
        } else {
            lv2 << one.str() << ";";
            ++n2;
        }
    }
    os << "] nu_level2=" << n2 << " further=[";
    for (size_t i = 0; i < in.further.size(); ++i) os << (i ? " " : "") << in.further[i];
    os << "] j=[";
    for (size_t i = 0; i < in.j.size(); ++i) os << (i ? "," : "") << (in.j[i] < 0 ? std::string("-") : std::to_string(in.j[i]));
    os << "]";
    const std::string body = os.str();
    const uint64_t h = engine::FnvBytes(engine::kFnvOffset64, body + "|" + lv2.str());
    std::ostringstream hs;
    hs << std::hex;
    hs.width(16);
    hs.fill('0');
    hs << h;
    return body + " digest=0x" + hs.str();
}

}  // namespace msym
}  // namespace lambda
}  // namespace hecke_ext
