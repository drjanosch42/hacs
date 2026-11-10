// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-BIL) Hida's GL2 x GL2 pairing on the 7-vertex model -- see HidaPairing.h.
// A function-by-function mirror of docs/implementation/slot19/r199/{r199lib,pairlib,small,cell258}.py and r198/r198lib.py.
#include "HidaPairing.h"

#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace hecke_ext {
namespace lvalue {
namespace hidapair {

namespace {
const M22 kS{0, -1, 1, 0};
const M22 kSinv{0, 1, -1, 0};
const M22 kT{1, 1, 0, 1};
const M22 kTi{1, -1, 0, 1};
const M22 kU{0, -1, 1, 1};
const M22 kIota{-1, 0, 0, 1};

int64_t FloorDiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
} // namespace

// =====================================================================================================================
// the fibre
// =====================================================================================================================
FibreOps::FibreOps(const Zq& R, uint32_t n) : R_(R), n_(n) {
    binom_.assign(n + 1, Vec());
    for (uint32_t t = 0; t <= n; ++t) {
        binom_[t].assign(t + 1, 1 % R.q);
        for (uint32_t s = 1; s < t; ++s) binom_[t][s] = R.add(binom_[t - 1][s - 1], binom_[t - 1][s]);
    }
}

Vec FibreOps::add(const Vec& u, const Vec& v) const {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R_.add(u[i], v[i]);
    return o;
}
Vec FibreOps::sub(const Vec& u, const Vec& v) const {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R_.sub(u[i], v[i]);
    return o;
}
Vec FibreOps::scale(uint64_t s, const Vec& u) const {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R_.mul(s, u[i]);
    return o;
}
uint64_t FibreOps::eval(const Vec& v, const Vec& lam) const {
    ENGINE_CHECK_MSG(v.size() == lam.size(), "hidapair::FibreOps::eval: sizes");
    unsigned __int128 acc = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        acc += static_cast<unsigned __int128>(v[i]) * lam[i];
        if ((i & 7) == 7) acc %= R_.q;
    }
    return static_cast<uint64_t>(acc % R_.q);
}

namespace {
Vec Powers(const Zq& R, uint64_t b, uint32_t n) {
    Vec p(n + 1, 1 % R.q);
    for (uint32_t i = 1; i <= n; ++i) p[i] = R.mul(p[i - 1], b);
    return p;
}
} // namespace

// [[1,b],[0,1]]: c'_s = sum_{t >= s} C(t,s) b^{t-s} c_t
Vec FibreOps::upper(uint64_t b, const Vec& c) const {
    if (b == 0) return c;
    const Vec pb = Powers(R_, b, n_);
    Vec out(n_ + 1, 0);
    for (uint32_t t = 0; t <= n_; ++t) {
        if (!c[t]) continue;
        for (uint32_t s = 0; s <= t; ++s) out[s] = R_.add(out[s], R_.mul(R_.mul(binom_[t][s], pb[t - s]), c[t]));
    }
    return out;
}
// [[1,0],[g,1]]: c'_s = sum_{t <= s} C(n-t, s-t) g^{s-t} c_t
Vec FibreOps::lower(uint64_t g, const Vec& c) const {
    if (g == 0) return c;
    const Vec pg = Powers(R_, g, n_);
    Vec out(n_ + 1, 0);
    for (uint32_t t = 0; t <= n_; ++t) {
        if (!c[t]) continue;
        const Vec& B = binom_[n_ - t];
        for (uint32_t s = t; s <= n_; ++s) out[s] = R_.add(out[s], R_.mul(R_.mul(B[s - t], pg[s - t]), c[t]));
    }
    return out;
}
Vec FibreOps::diag(uint64_t a, uint64_t d, const Vec& c) const {
    const Vec pa = Powers(R_, a, n_), pd = Powers(R_, d, n_);
    Vec out(n_ + 1);
    for (uint32_t t = 0; t <= n_; ++t) out[t] = R_.mul(R_.mul(pa[n_ - t], pd[t]), c[t]);
    return out;
}
// (Up^T lam)_t = sum_{s <= t} C(t,s) b^{t-s} lam_s
Vec FibreOps::upperT(uint64_t b, const Vec& lam) const {
    if (b == 0) return lam;
    const Vec pb = Powers(R_, b, n_);
    Vec out(n_ + 1, 0);
    for (uint32_t t = 0; t <= n_; ++t) {
        uint64_t s_ = 0;
        for (uint32_t s = 0; s <= t; ++s)
            if (lam[s]) s_ = R_.add(s_, R_.mul(R_.mul(binom_[t][s], pb[t - s]), lam[s]));
        out[t] = s_;
    }
    return out;
}
// (Lo^T lam)_t = sum_{s >= t} C(n-t, s-t) g^{s-t} lam_s
Vec FibreOps::lowerT(uint64_t g, const Vec& lam) const {
    if (g == 0) return lam;
    const Vec pg = Powers(R_, g, n_);
    Vec out(n_ + 1, 0);
    for (uint32_t t = 0; t <= n_; ++t) {
        const Vec& B = binom_[n_ - t];
        uint64_t s_ = 0;
        for (uint32_t s = t; s <= n_; ++s)
            if (lam[s]) s_ = R_.add(s_, R_.mul(R_.mul(B[s - t], pg[s - t]), lam[s]));
        out[t] = s_;
    }
    return out;
}
// S u_t = (-1)^t u_{n-t}
Vec FibreOps::sflip(const Vec& c) const {
    Vec out(n_ + 1);
    for (uint32_t t = 0; t <= n_; ++t) out[n_ - t] = (t % 2 == 0) ? c[t] : R_.neg(c[t]);
    return out;
}
// (rho(S)^T lam)_t = (-1)^t lam_{n-t}
Vec FibreOps::sflipT(const Vec& c) const {
    Vec out(n_ + 1);
    for (uint32_t t = 0; t <= n_; ++t) out[t] = (t % 2 == 0) ? c[n_ - t] : R_.neg(c[n_ - t]);
    return out;
}

Vec FibreOps::act(const M22& A, const Vec& c) const {
    ENGINE_CHECK_MSG(c.size() == n_ + 1u, "hidapair::FibreOps::act: vector size");
    const uint64_t a = R_.from_int(A.a), b = R_.from_int(A.b), cc = R_.from_int(A.c), d = R_.from_int(A.d);
    const uint64_t det = R_.sub(R_.mul(a, d), R_.mul(b, cc));
    if (R_.unit(a)) {
        const uint64_t ia = R_.inv(a);
        Vec v = upper(R_.mul(b, ia), c);
        v = diag(a, R_.mul(det, ia), v);
        return lower(R_.mul(cc, ia), v);
    }
    const bool odd = (n_ % 2) != 0;
    if (R_.unit(cc)) {
        Vec v = sflip(act(M22{-A.c, -A.d, A.a, A.b}, c));
        return odd ? scale(R_.neg(1 % R_.q), v) : v;
    }
    ENGINE_CHECK_MSG(R_.unit(b) || R_.unit(d), "hidapair::FibreOps::act: no unit entry");
    Vec v = sflip(c);
    if (odd) v = scale(R_.neg(1 % R_.q), v);
    return act(M22{A.b, -A.a, A.d, -A.c}, v);
}

Vec FibreOps::actT(const M22& A, const Vec& lam) const {
    ENGINE_CHECK_MSG(lam.size() == n_ + 1u, "hidapair::FibreOps::actT: vector size");
    const uint64_t a = R_.from_int(A.a), b = R_.from_int(A.b), cc = R_.from_int(A.c), d = R_.from_int(A.d);
    const uint64_t det = R_.sub(R_.mul(a, d), R_.mul(b, cc));
    const bool odd = (n_ % 2) != 0;
    if (R_.unit(a)) {
        const uint64_t ia = R_.inv(a);
        Vec v = lowerT(R_.mul(cc, ia), lam);
        v = diag(a, R_.mul(det, ia), v);
        return upperT(R_.mul(b, ia), v);
    }
    if (R_.unit(cc)) {
        Vec v = actT(M22{-A.c, -A.d, A.a, A.b}, sflipT(lam));
        return odd ? scale(R_.neg(1 % R_.q), v) : v;
    }
    ENGINE_CHECK_MSG(R_.unit(b) || R_.unit(d), "hidapair::FibreOps::actT: no unit entry");
    Vec v = sflipT(actT(M22{A.b, -A.a, A.d, -A.c}, lam));
    return odd ? scale(R_.neg(1 % R_.q), v) : v;
}

Mat FibreOps::matrix(Fibre f, const M22& A) const {
    Mat out(n_ + 1, Vec(n_ + 1, 0));
    for (uint32_t j = 0; j <= n_; ++j) {
        Vec e(n_ + 1, 0);
        e[j] = 1 % R_.q;
        const Vec col = on(f, A, e);
        for (uint32_t i = 0; i <= n_; ++i) out[i][j] = col[i];
    }
    return out;
}

// =====================================================================================================================
// linear algebra over Z/ell^M
// =====================================================================================================================
SmithResult SmithKernel(const Zq& R, const Mat& rows, uint32_t ncols) {
    Mat A = rows;
    const uint32_t nr = static_cast<uint32_t>(A.size());
    for (auto& r : A) {
        ENGINE_CHECK_MSG(r.size() == ncols, "hidapair::SmithKernel: row length");
        for (auto& x : r) x %= R.q;
    }
    Mat Q(ncols, Vec(ncols, 0));
    for (uint32_t i = 0; i < ncols; ++i) Q[i][i] = 1 % R.q;
    std::vector<uint32_t> rset(nr), cset(ncols);
    for (uint32_t i = 0; i < nr; ++i) rset[i] = i;
    for (uint32_t j = 0; j < ncols; ++j) cset[j] = j;
    SmithResult out;
    std::vector<uint64_t> ellpow(R.m + 1, 1);
    for (uint32_t i = 1; i <= R.m; ++i) ellpow[i] = ellpow[i - 1] * R.ell;
    while (!rset.empty() && !cset.empty()) {
        bool found = false;
        uint32_t bv = 0, pi = 0, pj = 0;
        for (uint32_t i : rset) {
            for (uint32_t j : cset) {
                const uint64_t x = A[i][j];
                if (!x) continue;
                const uint32_t v = R.val(x);
                if (!found || v < bv) {
                    found = true; bv = v; pi = i; pj = j;
                    if (v == 0) break;
                }
            }
            if (found && bv == 0) break;
        }
        if (!found) break;
        const uint64_t u = A[pi][pj] / ellpow[bv];
        const uint64_t iu = R.inv(u);
        for (uint32_t j : cset) {
            if (j == pj) continue;
            const uint64_t y = A[pi][j];
            if (!y) continue;
            const uint64_t f = R.mul(y / ellpow[bv], iu);
            for (uint32_t i = 0; i < nr; ++i)
                if (A[i][pj]) A[i][j] = R.sub(A[i][j], R.mul(f, A[i][pj]));
            for (uint32_t i = 0; i < ncols; ++i)
                if (Q[i][pj]) Q[i][j] = R.sub(Q[i][j], R.mul(f, Q[i][pj]));
        }
        for (uint32_t i : rset) {
            if (i == pi) continue;
            const uint64_t y = A[i][pj];
            if (!y) continue;
            const uint64_t f = R.mul(y / ellpow[bv], iu);
            for (uint32_t j : cset)
                if (A[pi][j]) A[i][j] = R.sub(A[i][j], R.mul(f, A[pi][j]));
        }
        out.pivot_valuations.push_back(bv);
        rset.erase(std::find(rset.begin(), rset.end(), pi));
        cset.erase(std::find(cset.begin(), cset.end(), pj));
    }
    for (uint32_t j : cset) {
        Vec col(ncols);
        for (uint32_t i = 0; i < ncols; ++i) col[i] = Q[i][j];
        out.kernel.push_back(col);
    }
    return out;
}

std::vector<std::pair<Vec, uint32_t>> CoordsInBasis(const Zq& R, const std::vector<Vec>& basis, const std::vector<Vec>& vecs) {
    const uint32_t k = static_cast<uint32_t>(basis.size());
    ENGINE_CHECK_MSG(k > 0, "hidapair::CoordsInBasis: empty basis");
    const uint32_t n1 = static_cast<uint32_t>(basis[0].size());
    Mat B = basis;
    Mat T(k, Vec(k, 0));
    for (uint32_t i = 0; i < k; ++i) T[i][i] = 1 % R.q;
    std::vector<uint32_t> pivcols;
    uint32_t r = 0;
    for (uint32_t c = 0; c < n1 && r < k; ++c) {
        uint32_t pr = k;
        for (uint32_t i = r; i < k; ++i)
            if (B[i][c] % R.ell) { pr = i; break; }
        if (pr == k) continue;
        std::swap(B[r], B[pr]);
        std::swap(T[r], T[pr]);
        const uint64_t iv = R.inv(B[r][c]);
        for (auto& x : B[r]) x = R.mul(x, iv);
        for (auto& x : T[r]) x = R.mul(x, iv);
        for (uint32_t i = 0; i < k; ++i) {
            if (i == r || !B[i][c]) continue;
            const uint64_t f = B[i][c];
            for (uint32_t t = 0; t < n1; ++t) B[i][t] = R.sub(B[i][t], R.mul(f, B[r][t]));
            for (uint32_t t = 0; t < k; ++t) T[i][t] = R.sub(T[i][t], R.mul(f, T[r][t]));
        }
        pivcols.push_back(c);
        ++r;
    }
    ENGINE_CHECK_MSG(r == k, "hidapair::CoordsInBasis: basis not saturated mod ell");
    std::vector<std::pair<Vec, uint32_t>> out;
    for (const Vec& v : vecs) {
        Vec y(k);
        for (uint32_t i = 0; i < k; ++i) y[i] = v[pivcols[i]] % R.q;
        Vec w(n1, 0);
        for (uint32_t i = 0; i < k; ++i)
            if (y[i])
                for (uint32_t t = 0; t < n1; ++t)
                    if (B[i][t]) w[t] = R.add(w[t], R.mul(y[i], B[i][t]));
        Vec z(k, 0);
        for (uint32_t i = 0; i < k; ++i)
            if (y[i])
                for (uint32_t j = 0; j < k; ++j) z[j] = R.add(z[j], R.mul(y[i], T[i][j]));
        uint32_t res = R.m;
        for (uint32_t t = 0; t < n1; ++t) res = std::min(res, R.val(R.sub(w[t], v[t] % R.q)));
        out.emplace_back(z, res);
    }
    return out;
}

Mat Transpose(const Mat& A) {
    if (A.empty()) return {};
    Mat T(A[0].size(), Vec(A.size()));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
    return T;
}

Mat Block(const Mat& A, uint32_t lo, uint32_t hi) {
    Mat B;
    for (uint32_t i = lo; i < hi; ++i) B.emplace_back(A[i].begin() + lo, A[i].begin() + hi);
    return B;
}

namespace {
Mat MatInvUnitPivots(const Zq& R, const Mat& A) {
    const size_t m = A.size();
    Mat Aug(m, Vec(2 * m, 0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) Aug[i][j] = A[i][j] % R.q;
        Aug[i][m + i] = 1 % R.q;
    }
    for (size_t c = 0; c < m; ++c) {
        size_t pr = m;
        for (size_t i = c; i < m; ++i)
            if (Aug[i][c] % R.ell) { pr = i; break; }
        ENGINE_CHECK_MSG(pr < m, "hidapair::MatInvUnitPivots: not invertible mod ell");
        std::swap(Aug[c], Aug[pr]);
        const uint64_t iv = R.inv(Aug[c][c]);
        for (auto& x : Aug[c]) x = R.mul(x, iv);
        for (size_t i = 0; i < m; ++i) {
            if (i == c || !Aug[i][c]) continue;
            const uint64_t f = Aug[i][c];
            for (size_t j = 0; j < 2 * m; ++j) Aug[i][j] = R.sub(Aug[i][j], R.mul(f, Aug[c][j]));
        }
    }
    Mat X(m, Vec(m));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < m; ++j) X[i][j] = Aug[i][m + j];
    return X;
}

using zk::MatPow;
using zk::MatVec;

} // namespace

Mat ReduceMat(const Zq& R, const Mat& A) {
    Mat B = A;
    for (auto& r : B)
        for (auto& x : r) x %= R.q;
    return B;
}

// =====================================================================================================================
// O
// =====================================================================================================================
QuadO QuadO::Trivial(const Zq& R) {
    QuadO O;
    O.R_ = R;
    O.trivial_ = true;
    return O;
}
QuadO QuadO::Eisenstein(const Zq& R, uint64_t c0, uint64_t c1) {
    QuadO O;
    O.R_ = R;
    O.trivial_ = false;
    O.c0_ = c0 % R.q;
    O.c1_ = c1 % R.q;
    ENGINE_CHECK_MSG(R.val(O.c0_) == 1 && O.c1_ % R.ell == 0, "hidapair::QuadO::Eisenstein: s^2 = c1 s + c0 is not Eisenstein");
    return O;
}
OElt QuadO::mul(const OElt& x, const OElt& y) const {
    if (trivial_) return {R_.mul(x.a, y.a), 0};
    const uint64_t bd = R_.mul(x.b, y.b);
    return {R_.add(R_.mul(x.a, y.a), R_.mul(bd, c0_)), R_.add(R_.add(R_.mul(x.a, y.b), R_.mul(x.b, y.a)), R_.mul(bd, c1_))};
}
uint32_t QuadO::vpi(const OElt& x, uint32_t cap) const {
    uint64_t qc = 1;
    for (uint32_t i = 0; i < cap; ++i) qc *= R_.ell;
    auto vc = [&](uint64_t a) {
        a %= qc;
        if (a == 0) return cap;
        uint32_t v = 0;
        while (a % R_.ell == 0) { a /= R_.ell; ++v; }
        return v;
    };
    if (trivial_) return vc(x.a);
    return std::min(2 * vc(x.a), 2 * vc(x.b) + 1);
}
OElt QuadO::div_pi(const OElt& x) const {
    if (trivial_) {
        ENGINE_CHECK_MSG(x.a % R_.ell == 0, "hidapair::QuadO::div_pi: not divisible");
        return {x.a / R_.ell, 0};
    }
    const OElt y = mul(x, OElt{R_.neg(c1_), 1 % R_.q});   // s (s - c1) = c0
    ENGINE_CHECK_MSG(y.a % R_.ell == 0 && y.b % R_.ell == 0, "hidapair::QuadO::div_pi: not divisible by pi");
    const uint64_t ui = R_.inv(c0_ / R_.ell);
    return {R_.mul(y.a / R_.ell, ui), R_.mul(y.b / R_.ell, ui)};
}

namespace {
std::vector<std::vector<OElt>> OMatMul(const QuadO& O, const std::vector<std::vector<OElt>>& A, const std::vector<std::vector<OElt>>& B) {
    const size_t n = A.size(), m = B.size(), p = B[0].size();
    std::vector<std::vector<OElt>> C(n, std::vector<OElt>(p));
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < m; ++k)
            for (size_t j = 0; j < p; ++j) C[i][j] = O.add(C[i][j], O.mul(A[i][k], B[k][j]));
    return C;
}

EigenLine FinishEigen(const QuadO& O, const Mat& A, const OElt& lambda, const std::vector<std::vector<OElt>>& cols, uint32_t cap) {
    EigenLine E;
    size_t best = 0;
    for (size_t j = 0; j < cols.size(); ++j) {
        uint32_t c = O.e() * cap;
        for (const OElt& z : cols[j]) c = std::min(c, O.vpi(z, cap));
        E.column_contents.push_back(c);
        if (c < E.column_contents[best]) best = j;
    }
    E.content = E.column_contents[best];
    E.g = cols[best];
    for (uint32_t i = 0; i < E.content; ++i)
        for (OElt& z : E.g) z = O.div_pi(z);
    // check A g = lambda g
    uint32_t chk = O.e() * cap;
    for (size_t a = 0; a < A.size(); ++a) {
        OElt s{0, 0};
        for (size_t b = 0; b < A.size(); ++b) s = O.add(s, O.mul(O.scalar(A[a][b]), E.g[b]));
        chk = std::min(chk, O.vpi(O.sub(s, O.mul(lambda, E.g[a])), cap));
    }
    E.eigen_check = chk;
    return E;
}
} // namespace

EigenLine EigenLineOf(const QuadO& O, const Mat& A, const OElt& lambda, const std::vector<OElt>& others, uint32_t cap) {
    const size_t m = A.size();
    std::vector<std::vector<OElt>> Q(m, std::vector<OElt>(m));
    for (size_t i = 0; i < m; ++i) Q[i][i] = O.scalar(1);
    for (const OElt& mu : others) {
        std::vector<std::vector<OElt>> B(m, std::vector<OElt>(m));
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < m; ++j) B[i][j] = O.sub(O.scalar(A[i][j]), i == j ? mu : OElt{0, 0});
        Q = OMatMul(O, Q, B);
    }
    std::vector<std::vector<OElt>> cols(m, std::vector<OElt>(m));
    for (size_t j = 0; j < m; ++j)
        for (size_t i = 0; i < m; ++i) cols[j][i] = Q[i][j];
    return FinishEigen(O, A, lambda, cols, cap);
}

EigenLine EigenLineByCofactor(const QuadO& O, const Mat& A, const Vec& P, const OElt& rho, uint32_t cap) {
    const size_t r = A.size();
    const size_t deg = P.size() - 1;
    ENGINE_CHECK_MSG(deg == r, "hidapair::EigenLineByCofactor: deg P != size of A");
    std::vector<OElt> qs;
    OElt rem = O.scalar(P[deg]);
    qs.push_back(rem);
    for (size_t d = deg - 1; d >= 1; --d) {
        rem = O.add(O.scalar(P[d]), O.mul(rem, rho));
        qs.push_back(rem);
    }
    std::reverse(qs.begin(), qs.end());   // qs[s] = coefficient of t^s in Q_rho, s = 0..deg-1
    std::vector<Mat> tpow{Mat(r, Vec(r, 0))};
    for (size_t i = 0; i < r; ++i) tpow[0][i][i] = 1 % O.R().q;
    for (size_t s = 1; s < deg; ++s) tpow.push_back(MatMul(O.R(), tpow.back(), A));
    std::vector<std::vector<OElt>> cols(r, std::vector<OElt>(r));
    for (size_t j = 0; j < r; ++j)
        for (size_t a = 0; a < r; ++a) {
            OElt z{0, 0};
            for (size_t s = 0; s < deg; ++s) z = O.add(z, O.mul(qs[s], O.scalar(tpow[s][a][j])));
            cols[j][a] = z;
        }
    return FinishEigen(O, A, rho, cols, cap);
}

OElt PairO(const QuadO& O, const std::vector<OElt>& g, const Mat& Gram, const std::vector<OElt>& h) {
    OElt v{0, 0};
    for (size_t a = 0; a < g.size(); ++a)
        for (size_t b = 0; b < h.size(); ++b)
            if (Gram[a][b]) v = O.add(v, O.mul(O.mul(g[a], O.scalar(Gram[a][b])), h[b]));
    return v;
}

uint64_t SqrtModPrimePower(const Zq& R, uint64_t a) {
    const uint64_t p = R.ell;
    const Zq F(p, 1);
    const uint64_t a1 = a % p;
    ENGINE_CHECK_MSG(a1 != 0 && F.pow(a1, (p - 1) / 2) == 1, "hidapair::SqrtModPrimePower: not a non-zero square mod ell");
    // Tonelli-Shanks mod p
    uint64_t Q = p - 1, S = 0;
    while (Q % 2 == 0) { Q /= 2; ++S; }
    uint64_t z = 2;
    while (F.pow(z, (p - 1) / 2) != p - 1) ++z;
    uint64_t Mm = S, c = F.pow(z, Q), t = F.pow(a1, Q), r = F.pow(a1, (Q + 1) / 2);
    while (t != 1) {
        uint64_t i = 0, tt = t;
        while (tt != 1) { tt = F.mul(tt, tt); ++i; }
        uint64_t b = c;
        for (uint64_t j = 0; j + 1 < Mm - i; ++j) b = F.mul(b, b);
        Mm = i; c = F.mul(b, b); t = F.mul(t, c); r = F.mul(r, b);
    }
    // Hensel: r <- r - (r^2 - a) / (2 r)
    for (uint32_t k = 1; k < R.m; ++k) {
        const Zq Rk(p, k + 1);
        const uint64_t rr = r % Rk.q;
        const uint64_t f = Rk.sub(Rk.mul(rr, rr), a % Rk.q);
        r = Rk.sub(rr, Rk.mul(f, Rk.inv(Rk.mul(2, rr))));
    }
    ENGINE_CHECK_MSG(R.mul(r, r) == a % R.q, "hidapair::SqrtModPrimePower: Hensel lift failed");
    return r;
}

// =====================================================================================================================
// group cocycles of PSL_2(Z) and Manin
// =====================================================================================================================
namespace {
std::vector<char> Word(M22 g) {
    std::vector<char> w;
    while (g.c != 0) {
        const int64_t q = FloorDiv(g.a, g.c);
        for (int64_t i = 0; i < (q > 0 ? q : -q); ++i) w.push_back(q > 0 ? 'T' : 't');
        g = M22{g.a - q * g.c, g.b - q * g.d, g.c, g.d};
        g = M22{g.c, g.d, -g.a, -g.b};
        w.push_back('S');
    }
    const int64_t s = g.a;
    const int64_t b = g.b * s;
    for (int64_t i = 0; i < (b > 0 ? b : -b); ++i) w.push_back(b > 0 ? 'T' : 't');
    return w;
}
const M22& Letter(char x) { return x == 'S' ? kS : (x == 'T' ? kT : kTi); }

struct Cocycle {
    const FibreOps* F;
    Fibre f;
    Vec cS, cT, cTi;
    Cocycle(const FibreOps& F_, Fibre f_, const Vec& xS, const Vec& xU) : F(&F_), f(f_) {
        const Zq& R = F_.R();
        const uint64_t m1 = R.neg(1 % R.q);
        cS = xS;
        const Vec cSinv = F_.scale(m1, F_.on(f, kSinv, xS));
        cT = F_.add(cSinv, F_.on(f, kSinv, xU));
        cTi = F_.scale(m1, F_.on(f, kTi, cT));
    }
    const Vec& c(char x) const { return x == 'S' ? cS : (x == 'T' ? cT : cTi); }
    Vec operator()(const M22& g) const {
        const std::vector<char> w = Word(g);
        if (w.empty()) return Vec(cS.size(), 0);
        Vec acc = c(w.back());
        for (size_t k = w.size() - 1; k-- > 0;) acc = F->add(c(w[k]), F->on(f, Letter(w[k]), acc));
        return acc;
    }
};

std::vector<M22> HeckeAlphas(uint32_t p) {
    std::vector<M22> al;
    for (uint32_t b = 0; b < p; ++b) al.push_back(M22{1, static_cast<int64_t>(b), 0, static_cast<int64_t>(p)});
    al.push_back(M22{static_cast<int64_t>(p), 0, 0, 1});
    return al;
}

std::vector<std::pair<int64_t, int64_t>> Convergents(int64_t a, int64_t m) {
    std::vector<std::pair<int64_t, int64_t>> out;
    int64_t p_prev = 1, q_prev = 0, p_prev2 = 0, q_prev2 = 1, x = a, y = m;
    while (y != 0) {
        const int64_t q = FloorDiv(x, y);
        const int64_t pk = q * p_prev + p_prev2, qk = q * q_prev + q_prev2;
        out.emplace_back(pk, qk);
        p_prev2 = p_prev; q_prev2 = q_prev; p_prev = pk; q_prev = qk;
        const int64_t nx = y, ny = x - q * y;
        x = nx; y = ny;
    }
    return out;
}

std::vector<M22> ManinPath(int64_t a, int64_t m) {
    std::vector<M22> mats;
    int64_t pp = 1, qq = 0;
    for (const auto& c : Convergents(a, m)) {
        const int64_t p = c.first, q = c.second;
        const int64_t det = p * qq - pp * q;
        ENGINE_CHECK_MSG(det == 1 || det == -1, "hidapair::ManinPath: convergent determinant");
        mats.push_back(det == 1 ? M22{p, pp, q, qq} : M22{-p, pp, -q, qq});
        pp = p; qq = q;
    }
    return mats;
}
} // namespace

// =====================================================================================================================
// the cell
// =====================================================================================================================
std::vector<Vec> HidaPairingCell::sign_basis(Fibre f, uint32_t parity, std::vector<uint32_t>* torsion) const {
    const uint32_t n = n_;
    std::vector<Vec> bvecs;
    for (uint32_t t = 0; t <= n / 2; ++t) {
        if (t % 2 != parity) continue;
        Vec v(n + 1, 0);
        if (t < n / 2) {
            v[t] = 1;
            v[n - t] = (t % 2 == 0) ? R_.neg(1) : 1;
        } else if ((n / 2) % 2 == 1) {
            v[t] = 1;
        } else {
            continue;
        }
        const Vec s = F_.add(v, F_.on(f, kS, v));
        ENGINE_CHECK_MSG(std::all_of(s.begin(), s.end(), [](uint64_t x) { return x == 0; }),
                         "hidapair::sign_basis: a seed vector is not killed by 1 + S");
        bvecs.push_back(v);
    }
    std::vector<Vec> cols;
    for (const Vec& v : bvecs) {
        const Vec u1 = F_.on(f, kU, v), u2 = F_.on(f, kU, u1);
        cols.push_back(F_.add(F_.add(v, u1), u2));
    }
    Mat rows(n + 1, Vec(bvecs.size(), 0));
    for (uint32_t i = 0; i <= n; ++i)
        for (size_t j = 0; j < bvecs.size(); ++j) rows[i][j] = cols[j][i];
    const SmithResult sr = SmithKernel(R_, rows, static_cast<uint32_t>(bvecs.size()));
    std::vector<Vec> out;
    for (const Vec& kv : sr.kernel) {
        Vec w(n + 1, 0);
        for (size_t j = 0; j < bvecs.size(); ++j)
            if (kv[j])
                for (uint32_t t = 0; t <= n; ++t)
                    if (bvecs[j][t]) w[t] = R_.add(w[t], R_.mul(kv[j], bvecs[j][t]));
        out.push_back(w);
    }
    if (torsion) {
        torsion->clear();
        for (uint32_t v : sr.pivot_valuations)
            if (v > 0) torsion->push_back(v);
    }
    return out;
}

Vec HidaPairingCell::normalise(Fibre f, const Vec& xS, const Vec& xU) const {
    // a cocycle of PSL_2(Z): (1 + S) c(S) = 0 and (1 + U + U^2) c(U) = 0
    const Vec s1 = F_.add(xS, F_.on(f, kS, xS));
    const Vec u1 = F_.on(f, kU, xU), u2 = F_.on(f, kU, u1);
    const Vec nu = F_.add(F_.add(xU, u1), u2);
    const auto zero = [](const Vec& v) { return std::all_of(v.begin(), v.end(), [](uint64_t x) { return x == 0; }); };
    if (!zero(s1) || !zero(nu))
        throw std::logic_error("hidapair::normalise: (c(S), c(U)) is not a 1-cocycle of PSL_2(Z) ((1+S)c(S) != 0 or N_U c(U) != 0)");
    const uint64_t i2 = R_.inv(2), i3 = R_.inv(3);
    const Vec v = F_.scale(R_.neg(i2), xS);
    const Vec cU = F_.sub(xU, F_.sub(F_.on(f, kU, v), v));
    return F_.scale(R_.neg(i3), F_.add(F_.scale(2, cU), F_.on(f, kU, cU)));
}

Vec HidaPairingCell::hecke_y(Fibre f, const Vec& y, uint32_t p) const {
    const Vec xS(n_ + 1, 0), xU = F_.sub(F_.on(f, kU, y), y);
    const Cocycle c(F_, f, xS, xU);
    const std::vector<M22> al = HeckeAlphas(p);
    Vec out[2];
    const M22 gams[2] = {kS, kU};
    for (int gi = 0; gi < 2; ++gi) {
        Vec acc(n_ + 1, 0);
        for (const M22& ai : al) {
            bool hit = false;
            for (const M22& aj : al) {
                M22 g = Mul(Mul(ai, gams[gi]), Adj(aj));
                const int64_t P = p;
                if (g.a % P == 0 && g.b % P == 0 && g.c % P == 0 && g.d % P == 0) {
                    g = M22{g.a / P, g.b / P, g.c / P, g.d / P};
                    acc = F_.add(acc, F_.on(f, Adj(ai), c(g)));
                    hit = true;
                    break;
                }
            }
            ENGINE_CHECK_MSG(hit, "hidapair::hecke_y: no alpha_j with alpha_i g alpha_j^{-1} in Gamma");
        }
        out[gi] = acc;
    }
    return normalise(f, out[0], out[1]);
}

Vec HidaPairingCell::iota_y(Fibre f, const Vec& y) const {
    const Vec xS(n_ + 1, 0), xU = F_.sub(F_.on(f, kU, y), y);
    const Cocycle c(F_, f, xS, xU);
    const Vec gS = c(kSinv);
    const Vec gU = c(Mul(Mul(kSinv, M22{1, 1, -1, 0}), kS));
    return normalise(f, F_.on(f, kIota, gS), F_.on(f, kIota, gU));
}

Vec HidaPairingCell::hecke_symbol(Fibre f, const Vec& P, uint32_t p) const {
    Vec out = F_.on(f, M22{1, 0, 0, static_cast<int64_t>(p)}, P);
    for (uint32_t b = 0; b < p; ++b) {
        Vec acc(n_ + 1, 0);
        for (const M22& g : ManinPath(b, p)) acc = F_.add(acc, F_.on(f, g, P));
        const Vec phi = F_.scale(R_.neg(1 % R_.q), acc);   // Phi({b/p, oo})
        out = F_.add(out, F_.on(f, M22{static_cast<int64_t>(p), -static_cast<int64_t>(b), 0, 1}, phi));
    }
    return out;
}

Vec HidaPairingCell::theta(const Vec& P) const {
    ENGINE_CHECK_MSG(n_ < R_.ell, "hidapair::theta: needs n < ell");
    Vec out(n_ + 1, 0);
    for (uint32_t s = 0; s <= n_; ++s) {
        if (!P[s]) continue;
        const uint64_t v = (s % 2 == 0) ? P[s] : R_.neg(P[s]);
        out[n_ - s] = R_.mul(v, R_.inv(F_.binom(n_, s)));
    }
    return out;
}

Vec HidaPairingCell::iota_shriek(Fibre f, const Vec& P) const {
    const Vec uP = F_.on(f, kU, P);
    return F_.scale(R_.neg(R_.inv(6)), F_.add(P, F_.scale(2, uP)));
}

uint64_t HidaPairingCell::r198_form(const Vec& P, const Vec& Q) const {
    ENGINE_CHECK_MSG(n_ < R_.ell, "hidapair::r198_form: needs n < ell");
    const Vec X = F_.sub(F_.act(kT, P), F_.act(kTi, P));
    uint64_t s = 0;
    for (uint32_t t = 0; t <= n_; ++t) {
        const uint64_t x = (t % 2 == 0) ? X[t] : R_.neg(X[t]);
        s = R_.add(s, R_.mul(R_.mul(x, R_.inv(F_.binom(n_, t))), Q[n_ - t]));
    }
    return s;
}

Vec HidaPairingCell::dual_coords(const Vec& y) const {
    Vec out;
    for (const Vec& w : Wb_) out.push_back(F_.eval(w, y));
    return out;
}

HidaPairingCell::HidaPairingCell(uint32_t n, uint64_t ell, uint32_t M, const std::vector<uint32_t>& primes)
    : n_(n), R_(ell, M), F_(R_, n) {
    const auto t0 = std::chrono::steady_clock::now();
    if (n % 2 != 0 || n < 2) throw std::logic_error("HidaPairingCell: n must be even and >= 2 (odd n vanishes on this model)");
    if (ell < 5) throw std::logic_error("HidaPairingCell: ell >= 5 (H^1(Gamma, M) = M/(M^S + M^U) needs 6 invertible)");
    const std::vector<Vec> We = sign_basis(Fibre::V, 0, &tors_e_), Wo = sign_basis(Fibre::V, 1, &tors_o_);
    Wb_ = We;
    Wb_.insert(Wb_.end(), Wo.begin(), Wo.end());
    ne_ = static_cast<uint32_t>(We.size());
    no_ = static_cast<uint32_t>(Wo.size());
    const uint32_t m = ne_ + no_;
    ENGINE_CHECK_MSG(m > 0, "HidaPairingCell: W = 0");
    for (uint32_t p : primes) {
        std::vector<Vec> imgs;
        for (const Vec& w : Wb_) imgs.push_back(hecke_symbol(Fibre::V, w, p));
        const auto co = CoordsInBasis(R_, Wb_, imgs);
        Mat A(m, Vec(m, 0));
        for (uint32_t j = 0; j < m; ++j) {
            ENGINE_CHECK_MSG(co[j].second + 1 >= M, "HidaPairingCell: T_p W not in W");
            for (uint32_t i = 0; i < m; ++i) A[i][j] = co[j].first[i];
        }
        Ac_[p] = A;
    }
    // dual representatives: greedy unit pivots, then the inverse of the unit minor
    {
        Mat Bc = Wb_;
        std::vector<uint32_t> piv;
        uint32_t r = 0;
        const uint32_t d = n + 1;
        for (uint32_t c = 0; c < d && r < m; ++c) {
            uint32_t pr = m;
            for (uint32_t i = r; i < m; ++i)
                if (Bc[i][c] % ell) { pr = i; break; }
            if (pr == m) continue;
            std::swap(Bc[r], Bc[pr]);
            const uint64_t iv = R_.inv(Bc[r][c]);
            for (auto& x : Bc[r]) x = R_.mul(x, iv);
            for (uint32_t i = 0; i < m; ++i) {
                if (i == r || !Bc[i][c]) continue;
                const uint64_t f = Bc[i][c];
                for (uint32_t t = 0; t < d; ++t) Bc[i][t] = R_.sub(Bc[i][t], R_.mul(f, Bc[r][t]));
            }
            piv.push_back(c);
            ++r;
        }
        ENGINE_CHECK_MSG(r == m, "HidaPairingCell: W not saturated (no unit minor)");
        Mat sub(m, Vec(m));
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t k = 0; k < m; ++k) sub[i][k] = Wb_[i][piv[k]];
        const Mat X = MatInvUnitPivots(R_, sub);
        for (uint32_t j = 0; j < m; ++j) {
            Vec y(d, 0);
            for (uint32_t k = 0; k < m; ++k) y[piv[k]] = X[k][j];
            ys_.push_back(y);
        }
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < m; ++j)
                ENGINE_CHECK_MSG(F_.eval(Wb_[i], ys_[j]) == (i == j ? 1 % R_.q : 0), "HidaPairingCell: <w_i, y_j> != delta_ij");
    }
    for (uint32_t p : primes) {
        Mat A(m, Vec(m, 0));
        for (uint32_t j = 0; j < m; ++j) {
            const Vec cj = dual_coords(hecke_y(Fibre::D, ys_[j], p));
            for (uint32_t i = 0; i < m; ++i) A[i][j] = cj[i];
        }
        AD_[p] = A;
    }
    for (uint32_t j = 0; j < m; ++j) iotaD_.push_back(dual_coords(iota_y(Fibre::D, ys_[j])));
    seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const Mat& HidaPairingCell::Ac(uint32_t p) const {
    const auto it = Ac_.find(p);
    ENGINE_CHECK_MSG(it != Ac_.end(), "HidaPairingCell::Ac: prime not built");
    return it->second;
}
const Mat& HidaPairingCell::AD(uint32_t p) const {
    const auto it = AD_.find(p);
    ENGINE_CHECK_MSG(it != AD_.end(), "HidaPairingCell::AD: prime not built");
    return it->second;
}

bool HidaPairingCell::adjoint_ok(uint32_t p) const {
    const Mat& A = Ac(p);
    const Mat& D = AD(p);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A.size(); ++j)
            if (D[i][j] != A[j][i]) return false;
    return true;
}

bool HidaPairingCell::iota_ok() const {
    const uint32_t m = ne_ + no_;
    for (uint32_t j = 0; j < m; ++j) {
        const uint64_t want = (j < ne_) ? R_.neg(1 % R_.q) : 1 % R_.q;   // -(sign of w_j)
        for (uint32_t i = 0; i < m; ++i)
            if (iotaD_[j][i] != (i == j ? want : 0)) return false;
    }
    return true;
}

// =====================================================================================================================
// readings
// =====================================================================================================================
namespace {
OElt PairVec(const QuadO& O, const std::vector<OElt>& g, const std::vector<OElt>& h) {
    OElt v{0, 0};
    for (size_t i = 0; i < g.size(); ++i) v = O.add(v, O.mul(g[i], h[i]));
    return v;
}

// v_pi(T g - mu g) for some mu, i.e. T g proportional to g, read at a unit coordinate of g (cross differences).
uint32_t ProportionalCheck(const QuadO& O, const Mat& T, const std::vector<OElt>& g, uint32_t cap) {
    std::vector<OElt> Tg(g.size());
    for (size_t a = 0; a < g.size(); ++a) {
        OElt s{0, 0};
        for (size_t b = 0; b < g.size(); ++b) s = O.add(s, O.mul(O.scalar(T[a][b]), g[b]));
        Tg[a] = s;
    }
    size_t j0 = g.size();
    for (size_t j = 0; j < g.size(); ++j)
        if (O.vpi(g[j], cap) == 0) { j0 = j; break; }
    if (j0 == g.size()) return 0;
    uint32_t v = O.e() * cap;
    for (size_t i = 0; i < g.size(); ++i) v = std::min(v, O.vpi(O.sub(O.mul(Tg[i], g[j0]), O.mul(Tg[j0], g[i])), cap));
    return v;
}
} // namespace

SmallCellReading ReadSmallCell(uint32_t n, uint64_t ell, uint32_t M) {
    const auto t0 = std::chrono::steady_clock::now();
    const HidaPairingCell C(n, ell, M, {2, 3, 5});
    const Zq& R = C.R();
    SmallCellReading out;
    out.n = n; out.ell = ell; out.M = M;
    out.ne = C.ne(); out.no = C.no();
    out.w_torsion_free = C.torsion_even().empty() && C.torsion_odd().empty();
    out.adj2 = C.adjoint_ok(2); out.adj3 = C.adjoint_ok(3); out.adj5 = C.adjoint_ok(5);
    out.iota = C.iota_ok();
    out.t2t5_commute = MatMul(R, C.Ac(2), C.Ac(5)) == MatMul(R, C.Ac(5), C.Ac(2)) &&
                       MatMul(R, C.AD(2), C.AD(5)) == MatMul(R, C.AD(5), C.AD(2));
    const uint32_t k = n + 2, ne = C.ne(), no = C.no(), m = ne + no;
    const uint64_t e2 = R.add(1, R.pow(2, k - 1));
    const Mat& A = C.Ac(2);
    const Mat& AD = C.AD(2);
    const Mat Ae = Block(A, 0, ne), Ao = Block(A, ne, m), ADe = Block(AD, 0, ne), ADo = Block(AD, ne, m);
    const Mat A5e = Block(C.Ac(5), 0, ne), A5o = Block(C.Ac(5), ne, m);
    // p_0 = u_0 - u_n in W^{even}, and p_0^perp in the dual of W^{even}
    Vec p0(n + 1, 0);
    p0[0] = 1;
    p0[n] = R.neg(1);
    const std::vector<Vec> We(C.W().begin(), C.W().begin() + ne);
    const auto cp0 = CoordsInBasis(R, We, {p0});
    ENGINE_CHECK_MSG(cp0[0].second == M, "ReadSmallCell: p_0 not in W^{even}");
    const std::vector<Vec> kerp0 = SmithKernel(R, Mat{cp0[0].first}, ne).kernel;
    // Hida's self-dual form
    Mat G;
    out.hida_applicable = n + 1 < ell;
    if (out.hida_applicable) {
        Mat Bm(m, Vec(m));
        G.assign(m, Vec(m));
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < m; ++j) {
                G[i][j] = C.pair(C.W()[i], C.theta(C.iota_shriek(Fibre::V, C.W()[j])));
                Bm[i][j] = C.r198_form(C.W()[i], C.W()[j]);
            }
        const uint64_t c = R.neg(R.inv(6));   // -1/6
        bool prop = true;
        for (uint32_t i = 0; i < m; ++i)
            for (uint32_t j = 0; j < m; ++j) prop = prop && G[i][j] == R.mul(c, Bm[i][j]);
        out.G_is_minus_sixth_B = prop;
        const SmithResult sr = SmithKernel(R, G, m);
        out.G_pivots = sr.pivot_valuations;
        std::sort(out.G_pivots.begin(), out.G_pivots.end());
        out.G_radical = m - static_cast<uint32_t>(sr.pivot_valuations.size());
    }
    struct Sys {
        std::string name;
        OElt lam;
        std::vector<OElt> oth_e, oth_o;
    };
    std::vector<Sys> systems;
    QuadO O = QuadO::Trivial(R);
    if (no == 1) {
        out.kind = "Z";
        out.charpoly_odd = Charpoly(R, Ao);
        systems.push_back({"f", O.scalar(Ao[0][0]), {O.scalar(e2)}, {}});
    } else if (no == 2) {
        out.charpoly_odd = Charpoly(R, Ao);
        const uint64_t a0 = out.charpoly_odd[0], a1 = out.charpoly_odd[1];
        const uint64_t disc = R.sub(R.mul(a1, a1), R.mul(4, a0));
        const uint32_t vd = R.val(disc);
        const uint64_t i2 = R.inv(2);
        if (vd == 0) {
            out.kind = "split";
            const uint64_t r = SqrtModPrimePower(R, disc);   // ENGINE_CHECKs a square (inert is not supported)
            const uint64_t lp = R.mul(R.add(R.neg(a1), r), i2), lm = R.mul(R.sub(R.neg(a1), r), i2);
            systems.push_back({"f[+r]", O.scalar(lp), {O.scalar(lm), O.scalar(e2)}, {O.scalar(lm)}});
            systems.push_back({"f[-r]", O.scalar(lm), {O.scalar(lp), O.scalar(e2)}, {O.scalar(lp)}});
        } else if (vd == 1) {
            out.kind = "ramified";
            O = QuadO::Eisenstein(R, disc, 0);
            const OElt lp{R.mul(R.neg(a1), i2), i2}, lm{R.mul(R.neg(a1), i2), R.neg(i2)};
            systems.push_back({"f_+", lp, {lm, O.scalar(e2)}, {lm}});
            systems.push_back({"f_-", lm, {lp, O.scalar(e2)}, {lp}});
        } else {
            throw std::runtime_error("ReadSmallCell: v(disc T_2 | S_k) >= 2 is not supported");
        }
    } else {
        throw std::runtime_error("ReadSmallCell: dim S_k <= 2 only");
    }
    const auto prod_diff = [&](const OElt& lam, const std::vector<OElt>& oth) {
        OElt p = O.scalar(1);
        for (const OElt& mu : oth) p = O.mul(p, O.sub(lam, mu));
        return O.vpi(p, M);
    };
    for (const Sys& s : systems) {
        EigenReading E;
        E.name = s.name;
        E.e = O.e();
        const EigenLine ge = EigenLineOf(O, Ae, s.lam, s.oth_e, M);
        const EigenLine he = EigenLineOf(O, ADe, s.lam, s.oth_e, M);
        EigenLine go, ho;
        if (no == 1) {
            go.g = {O.scalar(1)}; ho.g = {O.scalar(1)};
            go.eigen_check = ho.eigen_check = O.e() * M;
        } else {
            go = EigenLineOf(O, Ao, s.lam, s.oth_o, M);
            ho = EigenLineOf(O, ADo, s.lam, s.oth_o, M);
        }
        E.eigen_check = std::min({ge.eigen_check, he.eigen_check, go.eigen_check, ho.eigen_check});
        E.mixed_even = O.vpi(PairVec(O, ge.g, he.g), M);
        E.mixed_odd = O.vpi(PairVec(O, go.g, ho.g), M);
        const auto diff = [](uint32_t a, uint32_t b) { return a >= b ? a - b : 1000u; };   // 1000: content above the product (a defect)
        E.cong_even = diff(prod_diff(s.lam, s.oth_e), ge.content);
        E.cong_odd = diff(no == 1 ? 0u : prod_diff(s.lam, s.oth_o), go.content);
        // T_5-eigen (Q6)
        const uint32_t thr = std::min(ge.eigen_check, no == 1 ? ge.eigen_check : go.eigen_check);
        E.t5_eigen = ProportionalCheck(O, A5e, ge.g, M) >= thr && (no == 1 || ProportionalCheck(O, A5o, go.g, M) >= thr);
        if (out.hida_applicable) {
            uint32_t a_ = O.e() * M;
            for (const Vec& kv : kerp0) {
                OElt z{0, 0};
                for (uint32_t i = 0; i < ne; ++i) z = O.add(z, O.mul(O.scalar(kv[i]), ge.g[i]));
                a_ = std::min(a_, O.vpi(z, M));
            }
            std::vector<OElt> gE(m), gO(m);
            for (uint32_t i = 0; i < ne; ++i) gE[i] = ge.g[i];
            for (uint32_t i = 0; i < no; ++i) gO[ne + i] = go.g[i];
            const uint32_t vg = O.vpi(PairO(O, gE, G, gO), M);
            E.hida_defined = true;
            E.hida = vg >= a_ ? vg - a_ : 1000u;
        }
        out.systems.push_back(E);
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// ---- (258, 37) ----------------------------------------------------------------------------------------------------------
const std::vector<int64_t>& R191Quartic() {
    static const std::vector<int64_t> P{-1009939619134LL, -1705313202814LL, 857664280197LL, -1535621984856LL, 1};
    return P;
}
const std::vector<int64_t>& R191Factor(uint32_t i) {
    static const std::vector<int64_t> P1{1542925779010LL, 150216529622LL, 1};
    static const std::vector<int64_t> P2{-955242198270LL, -1685838514478LL, 1};
    ENGINE_CHECK_MSG(i == 1 || i == 2, "R191Factor: i in {1, 2}");
    return i == 1 ? P1 : P2;
}

namespace {
struct BlockData {
    std::vector<Vec> Bc, BD;
    Mat tbc, tbD, Gram;
    std::vector<std::pair<std::string, std::pair<EigenLine, uint32_t>>> ec;   // name -> (eigenline, v_pi P'(rho))
    std::vector<std::pair<std::string, EigenLine>> eD;
    std::vector<QuadO> Os;
    uint32_t lo = 0, hi = 0;
};

std::pair<std::vector<Vec>, Mat> BlockBasis(const Zq& R, const Mat& A, uint64_t lam) {
    const size_t d = A.size();
    Mat t = A;
    for (size_t i = 0; i < d; ++i) t[i][i] = R.sub(t[i][i], lam % R.q);
    const Mat tK = MatPow(R, t, 4ull * R.m);
    return {SmithKernel(R, tK, static_cast<uint32_t>(d)).kernel, t};
}

} // namespace

Mat Restrict(const Zq& R, const Mat& t, const std::vector<Vec>& B) {
    std::vector<Vec> imgs;
    for (const Vec& b : B) imgs.push_back(MatVec(R, t, b));
    const auto co = CoordsInBasis(R, B, imgs);
    const size_t r = B.size();
    Mat out(r, Vec(r));
    for (size_t j = 0; j < r; ++j) {
        ENGINE_CHECK_MSG(co[j].second + 2 >= R.m, "hidapair::Restrict: block not t-stable to precision");
        for (size_t i = 0; i < r; ++i) out[i][j] = co[j].first[i];
    }
    return out;
}

namespace {

bool SameMod(const Vec& a, const std::vector<int64_t>& b, const Zq& R8) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] % R8.q != R8.from_int(b[i])) return false;
    return true;
}

// The four roots rho of P = P_1 P_2 (two per Eisenstein factor): eigenlines of tb over O_{K_i}, with v_pi P'(rho).
std::vector<std::pair<std::string, std::pair<EigenLine, uint32_t>>> EigenFour(const Zq& R8, const Mat& tb, const Vec& cp,
                                                                              std::vector<QuadO>* Os) {
    std::vector<std::pair<std::string, std::pair<EigenLine, uint32_t>>> out;
    Vec Pc(cp.size());
    for (size_t i = 0; i < cp.size(); ++i) Pc[i] = cp[i] % R8.q;
    const Mat tb8 = ReduceMat(R8, tb);
    if (Os) Os->clear();
    for (uint32_t i = 1; i <= 2; ++i) {
        const std::vector<int64_t>& PI = R191Factor(i);
        // rho^2 + c1 rho + c0 = 0  <=>  rho^2 = (-c1) rho + (-c0)
        const QuadO O = QuadO::Eisenstein(R8, R8.from_int(-PI[0]), R8.from_int(-PI[1]));
        if (Os) Os->push_back(O);
        for (int conj = 0; conj < 2; ++conj) {
            const OElt rho = conj == 0 ? OElt{0, 1} : OElt{R8.from_int(-PI[1]), R8.q - 1};
            const EigenLine E = EigenLineByCofactor(O, tb8, Pc, rho, R8.m);
            OElt dP{0, 0}, pw = O.scalar(1);
            for (size_t deg = 1; deg < Pc.size(); ++deg) {
                dP = O.add(dP, O.mul(O.scalar(R8.mul(deg % R8.q, Pc[deg])), pw));
                pw = O.mul(pw, rho);
            }
            out.push_back({"f_(" + std::to_string(i) + "," + (conj ? "rho')" : "rho)"), {E, O.vpi(dP, R8.m)}});
        }
    }
    return out;
}
} // namespace

Cell258Reading ReadCell258(uint32_t M) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t n = 258;
    const uint64_t ell = 37;
    const HidaPairingCell C(n, ell, M, {2, 3, 5});
    const Zq& R = C.R();
    const Zq R8(ell, std::min<uint32_t>(8, M));
    Cell258Reading out;
    out.M = M;
    out.ne = C.ne(); out.no = C.no();
    out.adj2 = C.adjoint_ok(2); out.adj3 = C.adjoint_ok(3); out.adj5 = C.adjoint_ok(5);
    out.iota = C.iota_ok();
    const uint32_t ne = C.ne(), m = C.ne() + C.no();
    // torsion of H^1(D) = D / ((1+S) D + (1+U+U^2) D) over Z/37^4
    {
        const Zq R4(ell, 4);
        const FibreOps F4(R4, n);
        std::vector<Vec> cols;
        for (uint32_t j = 0; j <= n; ++j) {
            Vec e(n + 1, 0);
            e[j] = 1;
            cols.push_back(F4.add(e, F4.dact(kS, e)));
            const Vec u1 = F4.dact(kU, e), u2 = F4.dact(kU, u1);
            cols.push_back(F4.add(F4.add(e, u1), u2));
        }
        Mat rows(n + 1, Vec(cols.size()));
        for (uint32_t i = 0; i <= n; ++i)
            for (size_t c = 0; c < cols.size(); ++c) rows[i][c] = cols[c][i];
        const SmithResult sr = SmithKernel(R4, rows, static_cast<uint32_t>(cols.size()));
        out.HD_free_rank = (n + 1) - static_cast<uint32_t>(sr.pivot_valuations.size());
        for (uint32_t v : sr.pivot_valuations)
            if (v > 0) out.HD_torsion.push_back(v);
        std::sort(out.HD_torsion.begin(), out.HD_torsion.end());
    }
    BlockData bd[2];
    for (int s = 0; s < 2; ++s) {
        BlockData& b = bd[s];
        BlockSign& S = out.sign[s];
        S.name = s == 0 ? "geo-even" : "geo-odd";
        b.lo = s == 0 ? 0 : ne;
        b.hi = s == 0 ? ne : m;
        const Mat Ab = Block(C.Ac(2), b.lo, b.hi), ADb = Block(C.AD(2), b.lo, b.hi);
        Mat tc, tD;
        std::tie(b.Bc, tc) = BlockBasis(R, Ab, 29);
        std::tie(b.BD, tD) = BlockBasis(R, ADb, 29);
        b.tbc = Restrict(R, tc, b.Bc);
        b.tbD = Restrict(R, tD, b.BD);
        const Vec cpc = Charpoly(R, b.tbc), cpD = Charpoly(R, b.tbD);
        S.rank_c = static_cast<uint32_t>(b.Bc.size());
        S.rank_D = static_cast<uint32_t>(b.BD.size());
        S.charpoly_c_is_P = SameMod(cpc, R191Quartic(), R8);
        S.charpoly_D_is_P = SameMod(cpD, R191Quartic(), R8);
        const Mat t2c = MatMul(R, b.tbc, b.tbc), t2D = MatMul(R, b.tbD, b.tbD);
        S.tr1 = RankModEll(R, b.tbc); S.tr2 = RankModEll(R, t2c); S.tr3 = RankModEll(R, MatMul(R, t2c, b.tbc));
        S.dr1 = RankModEll(R, b.tbD); S.dr2 = RankModEll(R, t2D); S.dr3 = RankModEll(R, MatMul(R, t2D, b.tbD));
        b.Gram.assign(b.Bc.size(), Vec(b.BD.size()));
        for (size_t i = 0; i < b.Bc.size(); ++i)
            for (size_t j = 0; j < b.BD.size(); ++j) b.Gram[i][j] = C.F().eval(b.Bc[i], b.BD[j]);
        const SmithResult gs = SmithKernel(R, b.Gram, static_cast<uint32_t>(b.BD.size()));
        S.gram_pivots = gs.pivot_valuations;
        S.gram_unimodular = gs.pivot_valuations.size() == b.Gram.size() &&
                            std::all_of(gs.pivot_valuations.begin(), gs.pivot_valuations.end(), [](uint32_t v) { return v == 0; });
        b.ec = EigenFour(R8, b.tbc, cpc, &b.Os);
        const auto eD = EigenFour(R8, b.tbD, cpD, nullptr);
        const Mat G8 = ReduceMat(R8, b.Gram);
        for (size_t f = 0; f < b.ec.size(); ++f) {
            const QuadO& O = b.Os[f / 2];
            BlockEigen E;
            E.name = b.ec[f].first;
            E.vpi_dP = b.ec[f].second.second;
            E.content_c = b.ec[f].second.first.content;
            E.eigen_check = std::min(b.ec[f].second.first.eigen_check, eD[f].second.first.eigen_check);
            E.mixed = O.vpi(PairO(O, b.ec[f].second.first.g, G8, eD[f].second.first.g), R8.m);
            S.forms.push_back(E);
        }
    }
    // P9: H^1_!(D) = im(H^1_c(D) -> H^1(D)) at m; W_D^{dpar} lands in the dual block of the W sign `ws`
    for (int dpar = 0; dpar < 2; ++dpar) {
        const int ws = dpar == 0 ? 1 : 0;
        BlockData& b = bd[ws];
        BlockSign& S = out.sign[ws];
        const std::vector<Vec> WD = C.sign_basis(Fibre::D, static_cast<uint32_t>(dpar));
        std::vector<Vec> imgs;
        for (const Vec& w : WD) imgs.push_back(C.hecke_symbol(Fibre::D, w, 2));
        const auto co = CoordsInBasis(R, WD, imgs);
        const size_t d = WD.size();
        Mat A(d, Vec(d));
        for (size_t j = 0; j < d; ++j) {
            ENGINE_CHECK_MSG(co[j].second + 1 >= M, "ReadCell258: T_2 W_D not in W_D");
            for (size_t i = 0; i < d; ++i) A[i][j] = co[j].first[i];
        }
        const size_t bw = b.hi - b.lo;
        std::vector<Vec> phis;
        for (const Vec& w : WD) {
            const Vec dc = C.dual_coords(C.iota_shriek(Fibre::D, w));
            phis.emplace_back(dc.begin() + b.lo, dc.begin() + b.hi);
        }
        const Mat ADb = Block(C.AD(2), b.lo, b.hi);
        bool inter = true;
        for (size_t j = 0; j < d && inter; ++j) {
            const Vec dc = C.dual_coords(C.iota_shriek(Fibre::D, imgs[j]));
            const Vec lhs(dc.begin() + b.lo, dc.begin() + b.hi);
            inter = lhs == MatVec(R, ADb, phis[j]);
        }
        S.p9_rank_WD = static_cast<uint32_t>(d);
        S.p9_intertwines = inter;
        std::vector<Vec> BmD;
        Mat tt;
        std::tie(BmD, tt) = BlockBasis(R, A, 29);
        std::vector<Vec> imgm;
        for (const Vec& bv : BmD) {
            Vec v(bw, 0);
            for (size_t j = 0; j < d; ++j)
                if (bv[j])
                    for (size_t k = 0; k < bw; ++k) v[k] = R.add(v[k], R.mul(bv[j], phis[j][k]));
            imgm.push_back(v);
        }
        const auto co2 = CoordsInBasis(R, b.BD, imgm);
        Mat Cm;
        for (const auto& c : co2) {
            ENGINE_CHECK_MSG(c.second + 2 >= M, "ReadCell258: image of W_D,m not in H^1(D)_m");
            Cm.push_back(c.first);
        }
        const SmithResult ix = SmithKernel(R, Cm, static_cast<uint32_t>(b.BD.size()));
        S.p9_index_pivots = ix.pivot_valuations;
        std::sort(S.p9_index_pivots.begin(), S.p9_index_pivots.end());
        const Mat tb = Restrict(R, tt, BmD);
        const Vec cp = Charpoly(R, tb);
        S.p9_charpoly_is_P = SameMod(cp, R191Quartic(), R8);
        const auto eI = EigenFour(R8, tb, cp, nullptr);
        const Mat G8 = ReduceMat(R8, b.Gram);
        const Mat C8 = ReduceMat(R8, Cm);
        for (size_t f = 0; f < b.ec.size(); ++f) {
            const QuadO& O = b.Os[f / 2];
            const std::vector<OElt>& hI = eI[f].second.first.g;
            std::vector<OElt> h(b.BD.size());
            for (size_t bb = 0; bb < b.BD.size(); ++bb) {
                OElt z{0, 0};
                for (size_t a = 0; a < hI.size(); ++a) z = O.add(z, O.mul(hI[a], O.scalar(C8[a][bb])));
                h[bb] = z;
            }
            S.forms[f].p9 = O.vpi(PairO(O, b.ec[f].second.first.g, G8, h), R8.m);
        }
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

} // namespace hidapair
} // namespace lvalue
} // namespace hecke_ext
