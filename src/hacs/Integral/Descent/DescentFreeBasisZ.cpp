// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EXPORT-Z step 2, package EXZ2-P1b: lattice sums, saturation in three modes (+ the multi-word fallback) and the det = +-1
// certificate (DescentFreeBasisZ.h; docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.2-1.4).

#include "DescentFreeBasisZ.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include "DescentHecke.h"
#include "../../NumberField/Lll.h"
#include "../IntegralCharpolyCrt.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {
namespace freez {

using clk = std::chrono::steady_clock;
using u128 = unsigned __int128;
using i128 = __int128;

namespace {
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

// floor(a / b), b > 0
BigInt FloorDiv(const BigInt& a, const BigInt& b) {
    BigInt q, r;
    BigInt::divmod(a, b, q, r);
    if (r.sign() < 0) q -= BigInt(1);
    return q;
}
// nearest integer to a / b, b > 0 (ties up): floor((2a + b) / 2b)
BigInt RoundDiv(const BigInt& a, const BigInt& b) { return FloorDiv(a * BigInt(2) + b, b * BigInt(2)); }

uint64_t Gcd64(uint64_t a, uint64_t b) {
    while (b) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
// inverse of a mod m (gcd 1 required)
uint64_t InvMod64(uint64_t a, uint64_t m) {
    i128 t0 = 0, t1 = 1, r0 = m, r1 = a % m;
    while (r1) {
        const i128 q = r0 / r1;
        i128 x = r0 - q * r1;
        r0 = r1;
        r1 = x;
        x = t0 - q * t1;
        t0 = t1;
        t1 = x;
    }
    ENGINE_CHECK_MSG(r0 == 1, "InvMod64: not a unit");
    if (t0 < 0) t0 += m;
    return static_cast<uint64_t>(t0);
}
BigInt InvModBig(const BigInt& a, const BigInt& m) {
    BigInt t0(0), t1(1), r0 = m, r1 = a.floor_mod(m);
    while (!r1.is_zero()) {
        BigInt q = r0 / r1;
        BigInt x = r0 - q * r1;
        r0 = std::move(r1);
        r1 = std::move(x);
        x = t0 - q * t1;
        t0 = std::move(t1);
        t1 = std::move(x);
    }
    ENGINE_CHECK_MSG(r0 == BigInt(1), "InvModBig: not a unit");
    return t0.floor_mod(m);
}
uint64_t RedU64(const BigInt& x, uint64_t M) {
    const auto& mg = x.magnitude();
    u128 r = 0;
    for (size_t i = mg.size(); i-- > 0;) r = ((r << 32) | mg[i]) % M;
    uint64_t v = static_cast<uint64_t>(r);
    if (x.is_neg() && v) v = M - v;
    return v;
}
BigInt FromU64(uint64_t v) { return BigInt::from_u64(v); }

struct Split {
    BigInt g;
};

// ---- the rings ------------------------------------------------------------------------------------------------------------------
// RingU64: Z/M, M < 2^62.  ell != 0: local mode, M = ell^k (valuation passes); ell == 0: field / composite mode (a non-unit throws Split).
struct RingU64 {
    using E = uint64_t;
    uint64_t M = 1, ell = 0;
    uint32_t k = 1;
    static RingU64 Local(uint64_t ell, uint32_t k) {
        RingU64 R;
        R.ell = ell;
        R.k = k;
        R.M = 1;
        for (uint32_t i = 0; i < k; ++i) R.M *= ell;
        return R;
    }
    static RingU64 Field(uint64_t M) {
        RingU64 R;
        R.M = M;
        return R;
    }
    uint32_t passes() const { return ell ? k : 1; }
    E zero() const { return 0; }
    bool nz(E a) const { return a != 0; }
    E red(const BigInt& x) const { return RedU64(x, M); }
    BigInt lift(E a) const { return FromU64(a); }
    E add(E a, E b) const { return static_cast<E>((u128(a) + b) % M); }
    E sub(E a, E b) const { return a >= b ? a - b : static_cast<E>(u128(a) + M - b); }
    E mul(E a, E b) const { return static_cast<E>(u128(a) * b % M); }
    int val(E a) const {
        if (ell) {
            int v = 0;
            while (a % ell == 0) {
                a /= ell;
                ++v;
            }
            return v;
        }
        const uint64_t g = Gcd64(a, M);
        if (g != 1) throw Split{FromU64(g)};
        return 0;
    }
    E divpow(E a, int v) const {
        for (int i = 0; i < v; ++i) {
            ENGINE_CHECK_MSG(a % ell == 0, "RingU64::divpow: not divisible");
            a /= ell;
        }
        return a;
    }
    E inv_unit_part(E a, int v) const { return InvMod64(divpow(a, v) % M, M); }
};
// RingBig: Z/M, M multi-word, field / composite mode only (the (v) fallback).
struct RingBig {
    using E = BigInt;
    BigInt M;
    uint32_t passes() const { return 1; }
    E zero() const { return BigInt(0); }
    bool nz(const E& a) const { return !a.is_zero(); }
    E red(const BigInt& x) const { return x.floor_mod(M); }
    BigInt lift(const E& a) const { return a; }
    E add(const E& a, const E& b) const {
        BigInt s = a + b;
        if (s >= M) s -= M;
        return s;
    }
    E sub(const E& a, const E& b) const {
        BigInt s = a - b;
        if (s.is_neg()) s += M;
        return s;
    }
    E mul(const E& a, const E& b) const { return (a * b).floor_mod(M); }
    int val(const E& a) const {
        const BigInt g = BigInt::gcd(a, M);
        if (g != BigInt(1)) throw Split{g};
        return 0;
    }
    E divpow(const E& a, int v) const {
        ENGINE_CHECK_MSG(v == 0, "RingBig: valuation passes are not defined");
        return a;
    }
    E inv_unit_part(const E& a, int) const { return InvModBig(a, M); }
};

// ---- the sparse local Smith pass ----------------------------------------------------------------------------------------------
template <class R>
class LocalSmith {
public:
    using E = typename R::E;
    struct Ent {
        uint32_t c;
        E v;
    };
    LocalSmith(const R& ring, const SparseZ& A) : R_(ring), nr_(A.rows), nc_(A.cols) {
        rows_.resize(nr_);
        colrows_.resize(nc_);
        rowpiv_.assign(nr_, -1);
        colpiv_.assign(nc_, 0);
        for (uint32_t i = 0; i < nr_; ++i)
            for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
                E v = R_.red(A.val[k]);
                if (!R_.nz(v)) continue;
                rows_[i].push_back({A.col[k], std::move(v)});
                colrows_[A.col[k]].push_back(i);
            }
    }
    void run() {
        std::vector<uint32_t> stamp(nr_, UINT32_MAX);
        uint32_t epoch = 0;
        for (uint32_t v = 0; v < R_.passes(); ++v)
            for (uint32_t c = 0; c < nc_; ++c) {
                if (colpiv_[c]) continue;
                ++epoch;
                int64_t best = -1;
                size_t best_nnz = 0;
                std::vector<uint32_t> keep;
                for (uint32_t i : colrows_[c]) {
                    if (stamp[i] == epoch || rowpiv_[i] >= 0) continue;
                    stamp[i] = epoch;
                    const E* a = find(i, c);
                    if (!a) continue;
                    keep.push_back(i);
                    const int vv = R_.val(*a);
                    if (vv != static_cast<int>(v)) continue;
                    if (best < 0 || rows_[i].size() < best_nnz || (rows_[i].size() == best_nnz && i < best)) {
                        best = i;
                        best_nnz = rows_[i].size();
                    }
                }
                colrows_[c].swap(keep);
                if (best < 0) continue;
                pivot(static_cast<uint32_t>(best), c, static_cast<int>(v));
            }
        for (const auto& r : rows_) nnz_end_ += r.size();
    }
    uint32_t pivots() const { return static_cast<uint32_t>(piv_.size()); }
    uint64_t ops() const { return ops_.size(); }
    uint64_t nnz() const { return nnz_end_ + fnnz_; }
    bool row_is_pivot(uint32_t i) const { return rowpiv_[i] >= 0; }
    uint32_t rows() const { return nr_; }
    uint32_t cols() const { return nc_; }
    void apply_U(std::vector<E>& w) const {
        for (const Op& o : ops_)
            if (R_.nz(w[o.p])) w[o.i] = R_.sub(w[o.i], R_.mul(o.g, w[o.p]));
    }
    // wp = U w; y with A y = w (mod M), or false
    bool solve(const std::vector<E>& wp, std::vector<E>& y) const {
        for (uint32_t i = 0; i < nr_; ++i)
            if (rowpiv_[i] < 0 && R_.nz(wp[i])) return false;
        y.assign(nc_, R_.zero());
        for (const Piv& P : piv_) {
            const E& a = wp[P.row];
            if (!R_.nz(a)) continue;
            if (P.v > 0 && R_.val(a) < P.v) return false;
            y[P.col] = R_.mul(R_.divpow(a, P.v), P.uinv);
        }
        for (size_t t = piv_.size(); t-- > 0;) {
            const Piv& P = piv_[t];
            E acc = y[P.col];
            for (const Ent& f : P.f)
                if (R_.nz(y[f.c])) acc = R_.sub(acc, R_.mul(f.v, y[f.c]));
            y[P.col] = acc;
        }
        return true;
    }

private:
    struct Piv {
        uint32_t row, col;
        int v;
        E uinv;
        std::vector<Ent> f;
    };
    struct Op {
        uint32_t i, p;
        E g;
    };
    const R& R_;
    uint32_t nr_, nc_;
    std::vector<std::vector<Ent>> rows_;
    std::vector<std::vector<uint32_t>> colrows_;
    std::vector<int32_t> rowpiv_;
    std::vector<uint8_t> colpiv_;
    std::vector<Piv> piv_;
    std::vector<Op> ops_;
    uint64_t nnz_end_ = 0, fnnz_ = 0;

    const E* find(uint32_t i, uint32_t c) const {
        const auto& r = rows_[i];
        auto it = std::lower_bound(r.begin(), r.end(), c, [](const Ent& e, uint32_t x) { return e.c < x; });
        if (it == r.end() || it->c != c) return nullptr;
        return &it->v;
    }
    void pivot(uint32_t p, uint32_t c, int v) {
        Piv P;
        P.row = p;
        P.col = c;
        P.v = v;
        const E* pv = find(p, c);
        ENGINE_CHECK_MSG(pv != nullptr, "LocalSmith: the pivot entry vanished");
        P.uinv = R_.inv_unit_part(*pv, v);
        const std::vector<Ent> prow = rows_[p];
        for (uint32_t i : colrows_[c]) {
            if (i == p || rowpiv_[i] >= 0) continue;
            const E* a = find(i, c);
            if (!a) continue;
            const E g = R_.mul(R_.divpow(*a, v), P.uinv);
            // row_i -= g row_p (merge)
            std::vector<Ent> out;
            const auto& ri = rows_[i];
            out.reserve(ri.size() + prow.size());
            size_t x = 0, y = 0;
            while (x < ri.size() || y < prow.size()) {
                if (y == prow.size() || (x < ri.size() && ri[x].c < prow[y].c)) {
                    out.push_back(ri[x++]);
                } else if (x == ri.size() || prow[y].c < ri[x].c) {
                    E val = R_.sub(R_.zero(), R_.mul(g, prow[y].v));
                    if (R_.nz(val)) {
                        colrows_[prow[y].c].push_back(i);
                        out.push_back({prow[y].c, std::move(val)});
                    }
                    ++y;
                } else {
                    E val = R_.sub(ri[x].v, R_.mul(g, prow[y].v));
                    if (R_.nz(val)) out.push_back({ri[x].c, std::move(val)});
                    ++x;
                    ++y;
                }
            }
            rows_[i].swap(out);
            ENGINE_CHECK_MSG(find(i, c) == nullptr, "LocalSmith: the elimination left an entry in the pivot column");
            ops_.push_back({i, p, g});
        }
        for (const Ent& e : prow) {
            if (e.c == c) continue;
            P.f.push_back({e.c, R_.mul(R_.divpow(e.v, v), P.uinv)});
        }
        fnnz_ += P.f.size();
        rowpiv_[p] = static_cast<int32_t>(piv_.size());
        colpiv_[c] = 1;
        std::vector<uint32_t>().swap(colrows_[c]);
        std::vector<Ent>().swap(rows_[p]);
        piv_.push_back(std::move(P));
    }
};

// ---- the kernel of the residual map, in insertion order (rref free columns) -------------------------------------------------
template <class R>
std::vector<std::vector<typename R::E>> KernelInOrder(const R& U, const std::vector<std::vector<typename R::E>>& res, uint32_t len) {
    using E = typename R::E;
    const uint32_t r = static_cast<uint32_t>(res.size());
    struct KP {
        uint32_t lead;
        std::vector<E> v, comb;
    };
    std::vector<KP> piv;   // sorted by lead
    std::vector<std::vector<E>> ker;
    for (uint32_t j = 0; j < r; ++j) {
        std::vector<E> v = res[j];
        std::vector<E> comb(r, U.zero());
        comb[j] = U.red(BigInt(1));
        for (const KP& P : piv) {
            if (!U.nz(v[P.lead])) continue;
            const E a = v[P.lead];
            for (uint32_t t = P.lead; t < len; ++t)
                if (U.nz(P.v[t])) v[t] = U.sub(v[t], U.mul(a, P.v[t]));
            for (uint32_t t = 0; t < r; ++t)
                if (U.nz(P.comb[t])) comb[t] = U.sub(comb[t], U.mul(a, P.comb[t]));
        }
        uint32_t lead = len;
        for (uint32_t t = 0; t < len; ++t)
            if (U.nz(v[t])) {
                lead = t;
                break;
            }
        if (lead == len) {
            ker.push_back(std::move(comb));
            continue;
        }
        (void)U.val(v[lead]);   // composite mode: a non-unit lead splits the modulus
        const E inv = U.inv_unit_part(v[lead], 0);
        for (uint32_t t = lead; t < len; ++t)
            if (U.nz(v[t])) v[t] = U.mul(v[t], inv);
        for (uint32_t t = 0; t < r; ++t)
            if (U.nz(comb[t])) comb[t] = U.mul(comb[t], inv);
        KP P{lead, std::move(v), std::move(comb)};
        auto it = std::lower_bound(piv.begin(), piv.end(), lead, [](const KP& a, uint32_t x) { return a.lead < x; });
        piv.insert(it, std::move(P));
    }
    return ker;
}

PackedZ ReplaceColumns(const PackedZ& X, const std::map<uint32_t, std::vector<std::pair<uint32_t, BigInt>>>& repl) {
    PackedZ Y(X.rows);
    for (uint32_t j = 0; j < X.cols; ++j) {
        auto it = repl.find(j);
        if (it != repl.end()) Y.append_col(it->second);
        else Y.append_col(X.col_sparse(j));
    }
    return Y;
}

// One side at one modulus.  ring: Z/M (M = unit^{sigma+1}); uring: Z/unit; sigma_pow = unit^sigma.  Returns the rounds; updates X,
// delta (divided by unit per replaced column), the step's stats.  Throws Split (composite mode).
template <class R>
void SaturateSide(const R& ring, const R& uring, const BigInt& unit, const BigInt& sigma_pow, const BigInt& Mbig, const SparseZ& A,
                  const LocalSmith<R>& S, PackedZ& X, BigInt& delta, SatStep& st) {
    using E = typename R::E;
    const uint32_t nr = A.rows, r = X.cols;
    std::vector<uint32_t> nonpiv;
    for (uint32_t i = 0; i < nr; ++i)
        if (!S.row_is_pivot(i)) nonpiv.push_back(i);
    const uint32_t len = static_cast<uint32_t>(nonpiv.size());
    auto residual = [&](uint32_t j) {
        std::vector<E> w(nr, ring.zero());
        for (uint64_t k = X.colptr[j]; k < X.colptr[j + 1]; ++k) w[X.row[k]] = ring.red(X.value(k));
        S.apply_U(w);
        std::vector<E> v(len, uring.zero());
        for (uint32_t t = 0; t < len; ++t) v[t] = uring.red(ring.lift(w[nonpiv[t]]));
        return v;
    };
    std::vector<std::vector<E>> res(r);
    for (uint32_t j = 0; j < r; ++j) res[j] = residual(j);
    while (true) {
        const std::vector<std::vector<E>> ker = KernelInOrder(uring, res, len);
        if (ker.empty()) break;
        ++st.rounds;
        std::map<uint32_t, std::vector<std::pair<uint32_t, BigInt>>> repl;
        for (const auto& cvec : ker) {
            // the free column f: the last non-zero coordinate of c is c_f = 1 (insertion order)
            uint32_t f = r;
            for (uint32_t t = r; t-- > 0;)
                if (uring.nz(cvec[t])) {
                    f = t;
                    break;
                }
            ENGINE_CHECK_MSG(f < r && uring.lift(cvec[f]) == BigInt(1), "Saturate: the kernel vector is not normalised (c_f = 1)");
            // w = unit^sigma X c, exactly
            std::map<uint32_t, BigInt> wz;
            for (uint32_t t = 0; t < r; ++t) {
                if (!uring.nz(cvec[t])) continue;
                const BigInt ct = uring.lift(cvec[t]);
                for (uint64_t k = X.colptr[t]; k < X.colptr[t + 1]; ++k) wz[X.row[k]] += X.value(k) * ct;
            }
            std::vector<E> w(nr, ring.zero());
            for (auto& kv : wz) {
                kv.second = kv.second * sigma_pow;
                w[kv.first] = ring.red(kv.second);
            }
            S.apply_U(w);
            std::vector<E> y;
            ENGINE_CHECK_MSG(S.solve(w, y), "Saturate: A y = unit^sigma X c has no solution mod M (the kernel condition is wrong)");
            // z = w - A y, exactly; x_f = z / M
            for (uint32_t i = 0; i < nr; ++i) {
                BigInt s(0);
                for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k)
                    if (ring.nz(y[A.col[k]])) s += A.val[k] * ring.lift(y[A.col[k]]);
                if (!s.is_zero()) wz[i] -= s;
            }
            std::vector<std::pair<uint32_t, BigInt>> col;
            for (const auto& kv : wz) {
                if (kv.second.is_zero()) continue;
                BigInt q, rem;
                BigInt::divmod(kv.second, Mbig, q, rem);
                ENGINE_CHECK_MSG(rem.is_zero(), "Saturate: (unit^sigma X c - A y) is not divisible by M");
                col.push_back({kv.first, std::move(q)});
            }
            ENGINE_CHECK_MSG(!col.empty(), "Saturate: a zero replacement column");
            repl[f] = std::move(col);
            BigInt q, rem;
            BigInt::divmod(delta, unit, q, rem);
            ENGINE_CHECK_MSG(rem.is_zero(), "Saturate: the tracked index is not divisible by the modulus (a defect)");
            delta = q;
            ++st.cols;
        }
        X = ReplaceColumns(X, repl);
        for (const auto& kv : repl) res[kv.first] = residual(kv.first);
    }
}

}  // namespace

// =================================================================================================================================
// helpers
// =================================================================================================================================
std::vector<uint32_t> SmallPrimes(uint32_t below) {
    std::vector<uint8_t> comp(below, 0);
    std::vector<uint32_t> out;
    for (uint32_t i = 2; i < below; ++i) {
        if (comp[i]) continue;
        out.push_back(i);
        for (uint64_t j = uint64_t(i) * i; j < below; j += i) comp[j] = 1;
    }
    return out;
}
namespace {
const std::vector<uint32_t>& Primes20() {
    static const std::vector<uint32_t> P = SmallPrimes(1u << 20);
    return P;
}
}  // namespace

std::string FactorString(const BigInt& x0) {
    BigInt x = x0.abs();
    if (x.is_zero()) return "0";
    std::ostringstream o;
    bool first = true;
    for (uint32_t p : Primes20()) {
        if (x == BigInt(1)) break;
        uint32_t e = 0;
        while (x.mod_u32(p) == 0) {
            x = x.div_exact_u32(p);
            ++e;
        }
        if (e) {
            o << (first ? "" : "*") << p;
            if (e > 1) o << "^" << e;
            first = false;
        }
    }
    if (x != BigInt(1)) o << (first ? "" : "*") << "c" << x.bit_length() << "b";
    else if (first) o << "1";
    return o.str();
}

uint32_t TransformBits(const std::vector<std::vector<BigInt>>& U) {
    uint32_t b = 0;
    for (const auto& u : U)
        for (const auto& x : u) b = std::max(b, static_cast<uint32_t>(x.bit_length()));
    return b;
}

uint32_t DetModPrime(std::vector<uint32_t> A, uint32_t r, uint32_t p) {
    uint64_t det = 1;
    for (uint32_t k = 0; k < r; ++k) {
        uint32_t piv = k;
        while (piv < r && A[size_t(piv) * r + k] == 0) ++piv;
        if (piv == r) return 0;
        if (piv != k) {
            for (uint32_t j = 0; j < r; ++j) std::swap(A[size_t(k) * r + j], A[size_t(piv) * r + j]);
            det = (p - det) % p;
        }
        const uint64_t a = A[size_t(k) * r + k];
        det = det * a % p;
        const uint64_t inv = InvMod64(a, p);
        for (uint32_t i = k + 1; i < r; ++i) {
            const uint64_t f = A[size_t(i) * r + k] * inv % p;
            if (!f) continue;
            uint32_t* ri = &A[size_t(i) * r];
            const uint32_t* rk = &A[size_t(k) * r];
            for (uint32_t j = k; j < r; ++j) ri[j] = static_cast<uint32_t>((ri[j] + (p - f) * rk[j]) % p);
        }
    }
    return static_cast<uint32_t>(det);
}

// =================================================================================================================================
// Sec. 1.2 the lattice sums
// =================================================================================================================================
namespace {
// xgcd: g = s a + t b, g >= 0
void XGcd(const BigInt& a, const BigInt& b, BigInt& g, BigInt& s, BigInt& t) {
    BigInt r0 = a, r1 = b, s0(1), s1(0), t0(0), t1(1);
    while (!r1.is_zero()) {
        BigInt q = r0 / r1;
        BigInt x = r0 - q * r1;
        r0 = std::move(r1);
        r1 = std::move(x);
        x = s0 - q * s1;
        s0 = std::move(s1);
        s1 = std::move(x);
        x = t0 - q * t1;
        t0 = std::move(t1);
        t1 = std::move(x);
    }
    if (r0.is_neg()) {
        r0 = -r0;
        s0 = -s0;
        t0 = -t0;
    }
    g = r0;
    s = s0;
    t = t0;
}
void Axpy(std::vector<BigInt>& y, const BigInt& a, const std::vector<BigInt>& x) {   // y += a x
    for (size_t i = 0; i < y.size(); ++i)
        if (!x[i].is_zero()) y[i] += a * x[i];
}

struct HnfT {
    std::vector<std::vector<BigInt>> h, u;   // r basis vectors (length r) and their coefficients (length M)
    std::vector<std::vector<BigInt>> rel;    // the relation lattice basis (length M); stored only when keep_rel
    uint32_t nrel = 0;
    uint32_t peak_bits = 0;                  // the largest coefficient met during the insertion (inline mode: before each reduction)
};
// The structural relations of a diagonal first block (xs[0 .. r-1] = +-g_i e_i): for t >= r, d_t = the order of x_t modulo the
// lattice of G_0 (lcm of g_i / gcd(g_i, x_t[i])) and c_t[i] = d_t x_t[i] / g_i, so rho_t = d_t e_t - sum_i sign_i c_t[i] e_i is a
// relation (X rho_t = 0).
struct StructRel {
    uint32_t r = 0;
    std::vector<int> sign;                   // sign of xs[i][i]
    std::vector<BigInt> d;                   // d_t (1 for t < r)
    std::vector<uint32_t> dbits;
    std::vector<std::vector<BigInt>> c;      // c_t[i] (empty for t < r)
};
StructRel MakeStructRel(const std::vector<std::vector<BigInt>>& xs, uint32_t r) {
    const size_t M = xs.size();
    StructRel S;
    S.r = r;
    std::vector<BigInt> g(r);
    S.sign.resize(r);
    for (uint32_t i = 0; i < r; ++i) {
        for (uint32_t k = 0; k < r; ++k) ENGINE_CHECK_MSG(k == i || xs[i][k].is_zero(), "StructuralReduce: the first block is not diagonal");
        g[i] = xs[i][i].abs();
        ENGINE_CHECK_MSG(!g[i].is_zero(), "StructuralReduce: a zero diagonal entry");
        S.sign[i] = xs[i][i].is_neg() ? -1 : 1;
    }
    S.d.assign(M, BigInt(1));
    S.dbits.assign(M, 1);
    S.c.resize(M);
    for (size_t t = r; t < M; ++t) {
        BigInt dt(1);
        for (uint32_t i = 0; i < r; ++i) {
            const BigInt h = g[i] / BigInt::gcd(g[i], xs[t][i]);
            dt = dt / BigInt::gcd(dt, h) * h;
        }
        S.d[t] = dt;
        S.dbits[t] = static_cast<uint32_t>(dt.bit_length());
        S.c[t].resize(r);
        for (uint32_t i = 0; i < r; ++i) {
            BigInt q, rem;
            BigInt::divmod(dt * xs[t][i], g[i], q, rem);
            ENGINE_CHECK_MSG(rem.is_zero(), "StructuralReduce: d_t x_t is not in the G_0 lattice");
            S.c[t][i] = q;
        }
    }
    return S;
}
// u <- u - q rho_t: u_t -= q d_t, u_i += sign_i q c_t[i]
inline void ApplyRho(std::vector<BigInt>& u, const StructRel& S, size_t t, const BigInt& q) {
    u[t] -= q * S.d[t];
    for (uint32_t i = 0; i < S.r; ++i) {
        const BigInt& ci = S.c[t][i];
        if (ci.is_zero()) continue;
        if (S.sign[i] < 0) u[i] -= q * ci;
        else u[i] += q * ci;
    }
}
// (fix round 1, F1) the inline reduction: every coordinate t >= r with |u_t| >= 2^bits(d_t) into (-d_t/2, d_t/2]; peak = the largest
// coefficient seen (before reducing)
void InlineReduce(std::vector<BigInt>& u, const StructRel& S, uint32_t& peak) {
    for (const auto& x : u) peak = std::max(peak, static_cast<uint32_t>(x.bit_length()));
    for (size_t t = S.r; t < u.size(); ++t) {
        if (u[t].is_zero() || static_cast<uint32_t>(u[t].bit_length()) <= S.dbits[t]) continue;
        const BigInt q = RoundDiv(u[t], S.d[t]);
        if (!q.is_zero()) ApplyRho(u, S, t, q);
    }
}
// column HNF by incremental insertion of xs (M vectors of Z^r); slot i: pivot at row i, zeros below, h_ii > 0.  sr != nullptr: the
// structural relations reduce every coefficient vector after every step (fix round 1, F1; the coefficients of the steps depend on
// the class vectors only, so h is unchanged and u changes by relations of the Z-span of the rho_t).
HnfT HnfInsert(const std::vector<std::vector<BigInt>>& xs, uint32_t r, const StructRel* sr = nullptr, bool keep_rel = true) {
    const uint32_t M = static_cast<uint32_t>(xs.size());
    HnfT T;
    T.h.assign(r, {});
    T.u.assign(r, {});
    auto red = [&](std::vector<BigInt>& u) {
        if (sr) InlineReduce(u, *sr, T.peak_bits);
    };
    // keep the entries above a pivot small during the insertion (symmetric remainders modulo the existing h_kk, k < i): without it the
    // Bezout steps grow the class entries (and the transform) multiplicatively
    auto reduce_above = [&](std::vector<BigInt>& v, std::vector<BigInt>& cv, uint32_t i) {
        for (uint32_t k = i; k-- > 0;) {
            if (T.h[k].empty() || v[k].is_zero()) continue;
            const BigInt q = RoundDiv(v[k], T.h[k][k]);
            if (q.is_zero()) continue;
            Axpy(v, -q, T.h[k]);
            Axpy(cv, -q, T.u[k]);
        }
    };
    for (uint32_t t = 0; t < M; ++t) {
        std::vector<BigInt> x = xs[t], cx(M, BigInt(0));
        cx[t] = BigInt(1);
        bool placed = false;
        for (uint32_t i = r; i-- > 0;) {
            if (x[i].is_zero()) continue;
            if (T.h[i].empty()) {
                if (x[i].is_neg()) {
                    for (auto& v : x) v = -v;
                    for (auto& v : cx) v = -v;
                }
                T.h[i] = std::move(x);
                T.u[i] = std::move(cx);
                reduce_above(T.h[i], T.u[i], i);
                red(T.u[i]);
                placed = true;
                break;
            }
            const BigInt a = T.h[i][i], b = x[i];
            BigInt q, rem;
            BigInt::divmod(b, a, q, rem);
            if (rem.is_zero()) {   // x -= (b / a) h_i
                const BigInt mq = -q;
                Axpy(x, mq, T.h[i]);
                Axpy(cx, mq, T.u[i]);
                red(cx);
                continue;
            }
            BigInt g, s, tt;
            XGcd(a, b, g, s, tt);
            const BigInt ag = a / g, bg = b / g;
            // [h_i, x] <- [s h_i + t x, (a/g) x - (b/g) h_i]
            std::vector<BigInt> nh(r, BigInt(0)), nu(M, BigInt(0));
            Axpy(nh, s, T.h[i]);
            Axpy(nh, tt, x);
            Axpy(nu, s, T.u[i]);
            Axpy(nu, tt, cx);
            std::vector<BigInt> nx(r, BigInt(0)), ncx(M, BigInt(0));
            Axpy(nx, ag, x);
            Axpy(nx, -bg, T.h[i]);
            Axpy(ncx, ag, cx);
            Axpy(ncx, -bg, T.u[i]);
            if (nh[i].is_neg()) {
                for (auto& v : nh) v = -v;
                for (auto& v : nu) v = -v;
            }
            T.h[i] = std::move(nh);
            T.u[i] = std::move(nu);
            x = std::move(nx);
            cx = std::move(ncx);
            ENGINE_CHECK_MSG(x[i].is_zero(), "HnfInsert: the Bezout step left a pivot entry");
            reduce_above(T.h[i], T.u[i], i);
            reduce_above(x, cx, i);
            red(T.u[i]);
            red(cx);
        }
        if (!placed) {
            bool zero = true;
            for (const auto& v : x) zero = zero && v.is_zero();
            ENGINE_CHECK_MSG(zero, "HnfInsert: an inserted vector did not reduce to 0");
            ++T.nrel;
            if (keep_rel) T.rel.push_back(std::move(cx));
        }
    }
    for (uint32_t i = 0; i < r; ++i) ENGINE_CHECK_MSG(!T.h[i].empty(), "HnfInsert: the lattice is not of full rank r");
    // reduce the entries above the diagonal into [0, h_ii)
    for (uint32_t j = 1; j < r; ++j)
        for (uint32_t i = j; i-- > 0;) {
            if (T.h[j][i].is_zero()) continue;
            const BigInt q = FloorDiv(T.h[j][i], T.h[i][i]);
            if (q.is_zero()) continue;
            Axpy(T.h[j], -q, T.h[i]);
            Axpy(T.u[j], -q, T.u[i]);
        }
    return T;
}

// plain size reduction of the transform columns modulo the relation lattice, weighted norm sum x_t^2 4^{w_t}
// LLL (exact, nf::lll_basis, delta 3/4) of one block of relations in the weighted norm (rows rel_t 2^{w_t}), then Babai's nearest plane
// (exact rational Gram-Schmidt of the LLL result) for every transform column
void LllBabaiBlock(std::vector<std::vector<BigInt>>& U, std::vector<std::vector<BigInt>> rel, const std::vector<uint32_t>& wbits) {
    const size_t R = rel.size(), M = wbits.size();
    if (R == 0) return;
    std::vector<BigInt> W1(M), W2(M);
    for (size_t t = 0; t < M; ++t) {
        W1[t] = BigInt::pow(BigInt(2), wbits[t]);
        W2[t] = W1[t] * W1[t];
    }
    nf::IntMat rows(R, std::vector<BigInt>(M));
    for (size_t a = 0; a < R; ++a)
        for (size_t t = 0; t < M; ++t) rows[a][t] = rel[a][t] * W1[t];
    const nf::LllBasisResult L = nf::lll_basis(rows);
    rel = nf::int_mat_mul(L.lll.H, rel);
    for (auto& u : U) {
        std::vector<nf::BigRational> ys(R), nu(R);
        for (size_t a = 0; a < R; ++a) {
            BigInt d(0);
            for (size_t t = 0; t < M; ++t)
                if (!u[t].is_zero() && !rel[a][t].is_zero()) d += u[t] * rel[a][t] * W2[t];
            nf::BigRational y(d);
            for (size_t b = 0; b < a; ++b)
                if (L.lll.mu[a][b] != nf::BigRational(0)) y -= L.lll.mu[a][b] * ys[b];
            ys[a] = y;
            nu[a] = y / L.lll.B[a];
        }
        for (size_t a = R; a-- > 0;) {
            const BigInt c = RoundDiv(nu[a].num(), nu[a].den());
            if (c.is_zero()) continue;
            Axpy(u, -c, rel[a]);
            for (size_t b = 0; b < a; ++b)
                if (L.lll.mu[a][b] != nf::BigRational(0)) nu[b] -= nf::BigRational(c) * L.lll.mu[a][b];
        }
    }
}
// Size reduction of the transform columns modulo the relation lattice: LLL + Babai on the whole relation lattice when it has at most
// `lll_max` vectors; above that the caller applies the structural reduction (pinned: the in-house exact LLL is too slow there --
// measured: a 48-vector block of (23, 20) did not finish in 5 min; a blockwise variant was tried and dropped; open item O-7 for P3: a
// fast integer LLL).  Returns 1 when LLL-reduced, 0 otherwise.
uint32_t SizeReduce(std::vector<std::vector<BigInt>>& U, const std::vector<std::vector<BigInt>>& rel, const std::vector<uint32_t>& wbits,
                    uint32_t lll_max) {
    const size_t R = rel.size();
    if (R == 0 || R > lll_max) return 0;
    LllBabaiBlock(U, rel, wbits);
    return 1;
}
// Above lll_max: reduction by the STRUCTURAL relations of the diagonal first block (StructRel).  u <- u - round(u_t / d_t) rho_t puts
// every coordinate t >= r into (-d_t/2, d_t/2] -- a function of u_t mod d_t only, and rho_s vanishes at every t >= r, t != s -- and
// the first-block coordinates follow from the class (X u = h, G_0 diagonal): the result is canonical modulo the Z-span of the rho_t
// (which is why the inline reduction of HnfInsert leaves it unchanged).  Integer arithmetic only; O(r M r).
void StructuralReduce(std::vector<std::vector<BigInt>>& U, const StructRel& S) {
    for (auto& u : U)
        for (size_t t = S.r; t < u.size(); ++t) {
            if (u[t].is_zero()) continue;
            const BigInt q = RoundDiv(u[t], S.d[t]);
            if (q.is_zero()) continue;
            ApplyRho(u, S, t, q);
        }
}

// columns of [A_0 | A_1 | ...] combined by the coefficient vectors: out col i = sum_t U[i][t] col_t
PackedZ Combine(const std::vector<const PackedZ*>& parts, const std::vector<std::vector<BigInt>>& U, uint32_t rows) {
    std::vector<std::pair<const PackedZ*, uint32_t>> cols;
    for (const PackedZ* P : parts)
        for (uint32_t j = 0; j < P->cols; ++j) cols.push_back({P, j});
    PackedZ out(rows);
    for (const auto& u : U) {
        ENGINE_CHECK_MSG(u.size() == cols.size(), "Combine: shapes");
        std::map<uint32_t, BigInt> acc;
        for (size_t t = 0; t < cols.size(); ++t) {
            if (u[t].is_zero()) continue;
            const PackedZ& P = *cols[t].first;
            const uint32_t j = cols[t].second;
            for (uint64_t k = P.colptr[j]; k < P.colptr[j + 1]; ++k) acc[P.row[k]] += P.value(k) * u[t];
        }
        std::vector<std::pair<uint32_t, BigInt>> e;
        for (auto& kv : acc)
            if (!kv.second.is_zero()) e.push_back({kv.first, std::move(kv.second)});
        out.append_col(e);
    }
    return out;
}
std::vector<uint32_t> ColBits(const std::vector<const PackedZ*>& parts) {
    std::vector<uint32_t> b;
    for (const PackedZ* P : parts)
        for (uint32_t j = 0; j < P->cols; ++j) {
            uint32_t m = 0;
            for (uint64_t k = P->colptr[j]; k < P->colptr[j + 1]; ++k) m = std::max(m, P->bits(k));
            b.push_back(m);
        }
    return b;
}
}  // namespace

namespace {
// the class coordinates x_(s, j) = Phi_0^T b_(s, j) (side B) and the value coordinates y_(s, i) = phi_s,i^T B_0 (side Phi), with g_0
void SumsCoords(const std::vector<Pair>& pairs, uint32_t r, std::vector<std::vector<BigInt>>& xs, std::vector<std::vector<BigInt>>& ys,
                std::vector<BigInt>& g0) {
    xs.clear();
    ys.clear();
    g0.assign(r, BigInt(0));
    for (const Pair& P : pairs) {
        const std::vector<BigInt> G = PairingMatrix(pairs[0].Phi, P.B);   // r x r: (i, j) = phi_0,i . b_s,j
        for (uint32_t j = 0; j < r; ++j) {
            std::vector<BigInt> x(r);
            for (uint32_t i = 0; i < r; ++i) x[i] = G[size_t(i) * r + j];
            xs.push_back(std::move(x));
        }
    }
    // Phi side, in the coordinates of B_0 (B_0 is a Q-basis of L, so the evaluation is injective on L^dual).  Phi_0^T B_0 = G_0 is
    // DIAGONAL (supports: Phi_0 on R0 + {P_i}, B_0 on Pc + {P_j}, R0 and Pc disjoint), so the first block fills the slots directly, as
    // on side B.  (P1b first used the values on B_Sigma: a dense first block, a transform of thousands of bits.)
    for (const Pair& P : pairs) {
        const std::vector<BigInt> G = PairingMatrix(P.Phi, pairs[0].B);   // (i, j) = phi_s,i . b_0,j
        if (&P == &pairs[0])
            for (uint32_t a = 0; a < r; ++a)
                for (uint32_t b = 0; b < r; ++b) {
                    if (a == b) g0[a] = G[size_t(a) * r + a];
                    else ENGINE_CHECK_MSG(G[size_t(a) * r + b].is_zero(), "LatticeSums: Phi_0^T B_0 is not diagonal");
                }
        for (uint32_t a = 0; a < r; ++a)
            ys.push_back(std::vector<BigInt>(G.begin() + static_cast<std::ptrdiff_t>(size_t(a) * r),
                                             G.begin() + static_cast<std::ptrdiff_t>(size_t(a + 1) * r)));
    }
}
// the relation count (K - 1) r decides the path before the HNF: LLL when 0 < (K - 1) r <= lll_max, else the structural reduction
bool LllPath(uint32_t K, uint32_t r, const FreeBasisOptions& opt) {
    const uint64_t R = uint64_t(K - 1) * r;
    return opt.size_reduce && R > 0 && R <= opt.lll_max_relations;
}
uint32_t MaxBits(const std::vector<std::vector<BigInt>>& v) { return TransformBits(v); }
// PredictSumsRss on computed coordinates (see the header)
uint64_t PredictSumsFrom(const IsoTreeCarrierZ& c, const std::vector<Pair>& pairs, const std::vector<std::vector<BigInt>>& xs,
                         const std::vector<std::vector<BigInt>>& ys, const std::vector<BigInt>& g0, const FreeBasisOptions& opt,
                         uint32_t* in3 = nullptr) {
    auto h = [](double b) -> double {
        const double w = std::ceil(b / 32.0);
        return std::max(32.0, 16.0 * std::ceil((4.0 * w + 8.0) / 16.0));
    };
    const double r = c.b1(), K = static_cast<double>(pairs.size()), M = K * r, nv = c.dim(1);
    if (r == 0) return 0;
    // h_ii | g_i (the lattice contains the G_0 lattice), the entries above a pivot are reduced modulo h_kk: class entries <= max g
    double bg = 1;
    for (const BigInt& g : g0) bg = std::max<double>(bg, g.bit_length());
    const double bx = std::max(MaxBits(xs), MaxBits(ys));
    const bool lll = LllPath(static_cast<uint32_t>(K), static_cast<uint32_t>(r), opt);
    double bd = 1;
    if (!lll && M > r) {
        const StructRel SB = MakeStructRel(xs, c.b1()), SP = MakeStructRel(ys, c.b1());
        for (uint32_t b : SB.dbits) bd = std::max<double>(bd, b);
        for (uint32_t b : SP.dbits) bd = std::max<double>(bd, b);
    }
    if (in3) {
        in3[0] = static_cast<uint32_t>(bx);
        in3[1] = static_cast<uint32_t>(bg);
        in3[2] = static_cast<uint32_t>(bd);
    }
    const double b_class = std::max(bx, bg) + 64.0;
    // structural path (inline): u_t < 2 d_t after every step, a Bezout step multiplies by <= bits(class); the first block by X u = h.
    // LLL path (r <= 24, no inline reduction): the unreduced HNF transform, P1b's measured growth <= r (bits x + bits g) (e.g. (8, 10):
    // 1769 <= 22 (75 + 20))
    const double b_trans = lll ? 64.0 + r * (bx + bg) : bd + bx + bg + 64.0;
    double bytes = 0;
    // both sides: the coordinates (M x r), the HNF basis (r x r), the transform (r x M), the stored relations (LLL path), the
    // structural data (M x r c_t + d_t)
    bytes += 2.0 * (M * r + r * r) * (32.0 + h(b_class));
    bytes += 2.0 * (r * M) * (32.0 + h(b_trans));
    if (lll) {
        const double R = (K - 1.0) * r;
        bytes += 2.0 * R * M * (32.0 + h(b_trans));
        // nf::lll_basis: the weighted rows, H, mu and B (exact rationals of ~ 2 R (b_trans + max column bits) bits)
        uint32_t wb = 0;
        for (const Pair& P : pairs) wb = std::max({wb, P.B.max_bits(), P.Phi.max_bits()});
        bytes += 3.0 * R * (R + M) * (64.0 + h(2.0 * (b_trans + wb) + 64.0));
    } else {
        bytes += 2.0 * (M * r + M) * (32.0 + h(bd + bx));
    }
    // the combined columns (packed): nnz <= r min(dim C^1, sum of the parts' nnz), entries of bits(parts) + b_trans + log2 M bits
    double nnzB = 0, nnzP = 0, bB = 0, bP = 0;
    for (const Pair& P : pairs) {
        nnzB += static_cast<double>(P.B.nnz());
        nnzP += static_cast<double>(P.Phi.nnz());
        bB = std::max<double>(bB, P.B.max_bits());
        bP = std::max<double>(bP, P.Phi.max_bits());
    }
    const double lgM = std::ceil(std::log2(M + 1.0));
    auto packed = [&](double nnz, double b) { return nnz * (16.0 + 4.0 * std::ceil((b + b_trans + lgM) / 32.0)); };
    bytes += packed(r * std::min(nv, nnzB), bB) + packed(r * std::min(nv, nnzP), bP);
    // Combine's per-column map and the pairing matrices (transient, one at a time)
    bytes += nv * (64.0 + h(std::max(bB, bP) + b_trans + lgM)) + 2.0 * r * r * (32.0 + h(b_class));
    // + 1 MiB: the allocator's granularity (glibc top pad / mmap threshold, the sampler's page resolution; small cells)
    return static_cast<uint64_t>(std::ceil(1.25 * bytes + 1048576.0));
}
uint64_t PairsResidentBytes(const std::vector<Pair>& pairs) {
    uint64_t b = 0;
    for (const Pair& P : pairs) b += P.B.bytes() + P.Phi.bytes();
    return b;
}
}  // namespace

uint64_t PredictSumsRss(const IsoTreeCarrierZ& c, const std::vector<Pair>& pairs, const FreeBasisOptions& opt) {
    std::vector<std::vector<BigInt>> xs, ys;
    std::vector<BigInt> g0;
    if (c.b1() == 0 || pairs.empty()) return 0;
    SumsCoords(pairs, c.b1(), xs, ys, g0);
    return PredictSumsFrom(c, pairs, xs, ys, g0, opt);
}

SumsResult LatticeSums(const IsoTreeCarrierZ& c, const std::vector<Pair>& pairs, const FreeBasisOptions& opt,
                       const testing::FreeBasisHooks* hooks) {
    const auto t0 = clk::now();
    SumsResult S;
    const uint32_t r = c.b1(), nv = c.dim(1);
    const uint32_t K = static_cast<uint32_t>(pairs.size());
    ENGINE_CHECK_MSG(K >= 1, "LatticeSums: no pairs");
    S.K = K;
    if (r == 0) {
        S.B = PackedZ(nv);
        S.Phi = PackedZ(nv);
        S.delta = BigInt(1);
        S.seconds = since(t0);
        return S;
    }
    std::vector<const PackedZ*> Bs, Ps;
    for (const Pair& P : pairs) {
        Bs.push_back(&P.B);
        Ps.push_back(&P.Phi);
    }
    std::vector<std::vector<BigInt>> xs, ys;
    std::vector<BigInt> g0;
    SumsCoords(pairs, r, xs, ys, g0);
    auto phase = [&](const char* what, clk::time_point t) {
        if (opt.verbose)
            std::cout << "EXZ2_P1B_PHASE cell=" << c.N() << ":" << c.n() << " K=" << K << " " << what << std::fixed << std::setprecision(3)
                      << " t=" << since(t) << std::endl;
    };
    phase("coords", t0);
    // (fix round 1, F1 (b)) the sums' ceiling, before the HNF: the pairs' resident masters + the predicted transient bytes
    uint32_t in3[3] = {0, 0, 0};
    S.rss_pred = PredictSumsFrom(c, pairs, xs, ys, g0, opt, in3);
    S.pred_bits_x = in3[0];
    S.pred_bits_g = in3[1];
    S.pred_bits_d = in3[2];
    {
        // the same 16 MiB process constant as PredictPairsRss, so both are compared with one ceiling on one scale
        const uint64_t need = (16ull << 20) + S.rss_pred + PairsResidentBytes(pairs);
        const uint64_t cap = opt.sums_rss_cap_bytes ? opt.sums_rss_cap_bytes : opt.pairs.rss_cap_bytes;
        if (need > cap) {
            S.refused = true;
            std::ostringstream o;
            o << "REFUSED [RssCeiling] the lattice sums at (N, n) = (" << c.N() << ", " << c.n() << "), K = " << K << ": predicted " << need
              << " bytes (16 MiB + sums " << S.rss_pred << " + resident pairs) > the ceiling " << cap << " bytes";
            S.refusal = o.str();
            S.seconds = since(t0);
            return S;
        }
    }
    const bool lll = LllPath(K, r, opt);
    const bool inl = opt.size_reduce && !lll && K > 1 && !(hooks && hooks->sums_posthoc_only);
    auto tp = clk::now();
    std::vector<BigInt> hB(r), hP(r);   // the HNF diagonals
    {
        const StructRel SR = (opt.size_reduce && !lll) ? MakeStructRel(xs, r) : StructRel{};
        HnfT TB = HnfInsert(xs, r, inl ? &SR : nullptr, lll);
        phase("hnf_B", tp);
        tp = clk::now();
        S.inline_B = inl;
        S.relations_B = TB.nrel;
        S.bits_U0 = TransformBits(TB.u);
        S.bits_U_peak = std::max(TB.peak_bits, S.bits_U0);
        if (opt.size_reduce) {
            S.lll_B = lll && SizeReduce(TB.u, TB.rel, ColBits(Bs), opt.lll_max_relations) != 0;
            if (!S.lll_B) StructuralReduce(TB.u, SR);
        }
        S.bits_U = TransformBits(TB.u);
        phase("reduce_B", tp);
        tp = clk::now();
        S.B = Combine(Bs, TB.u, nv);
        phase("combine_B", tp);
        tp = clk::now();
        for (uint32_t a = 0; a < r; ++a) hB[a] = TB.h[a][a];
    }
    {
        const StructRel SR = (opt.size_reduce && !lll) ? MakeStructRel(ys, r) : StructRel{};
        HnfT TP = HnfInsert(ys, r, inl ? &SR : nullptr, lll);
        phase("hnf_Phi", tp);
        tp = clk::now();
        S.inline_Phi = inl;
        S.relations_Phi = TP.nrel;
        S.bits_V0 = TransformBits(TP.u);
        S.bits_V_peak = std::max(TP.peak_bits, S.bits_V0);
        if (opt.size_reduce) {
            S.lll_Phi = lll && SizeReduce(TP.u, TP.rel, ColBits(Ps), opt.lll_max_relations) != 0;
            if (!S.lll_Phi) StructuralReduce(TP.u, SR);
        }
        S.bits_V = TransformBits(TP.u);
        phase("reduce_Phi", tp);
        tp = clk::now();
        S.Phi = Combine(Ps, TP.u, nv);
        phase("combine_Phi", tp);
        for (uint32_t a = 0; a < r; ++a) hP[a] = TP.h[a][a];
    }
    // Delta_Sigma = |det(Phi_Sigma^T B_Sigma)|: B_Sigma = B_0 G_0^{-1} H_B and Phi_Sigma^T B_0 = H_Phi^T on L (classes), so
    // det = prod h_B,ii * prod h_Phi,ii / prod g_i (exact division CHECKED), and confirmed by det(Phi_Sigma^T B_Sigma) mod two primes
    BigInt num(1), den(1);
    for (uint32_t a = 0; a < r; ++a) {
        num = num * hB[a] * hP[a];
        den = den * g0[a].abs();
    }
    BigInt d, rem;
    BigInt::divmod(num, den, d, rem);
    ENGINE_CHECK_MSG(rem.is_zero() && d.sign() > 0, "LatticeSums: Delta_Sigma is not an integer");
    {
        const std::vector<BigInt> G = PairingMatrix(S.Phi, S.B);
        for (uint32_t q : integral::CrtPrimes(2)) {
            std::vector<uint32_t> A(G.size());
            for (size_t e = 0; e < G.size(); ++e) A[e] = G[e].mod_u32(q);
            const uint32_t dq = DetModPrime(std::move(A), r, q), dd = d.mod_u32(q);
            ENGINE_CHECK_MSG(dq == dd || dq == (q - dd) % q, "LatticeSums: det(Phi_Sigma^T B_Sigma) != +-Delta_Sigma mod a prime");
        }
    }
    S.delta = d;
    S.seconds = since(t0);
    return S;
}

// =================================================================================================================================
// Sec. 1.3 saturation
// =================================================================================================================================
namespace {
struct SideCtx {
    char side;
    const SparseZ* A;
    PackedZ* X;
    uint32_t exp_rank;
};

// local (prime-power) mode at ell: adaptive k, then the side
void LocalPiece(uint64_t ell, SideCtx& sc, BigInt& delta, const FreeBasisOptions& opt, std::vector<SatStep>& log) {
    const auto t0 = clk::now();
    SatStep st;
    st.side = sc.side;
    st.piece = std::to_string(ell);
    st.mode = "local";
    uint32_t k = 1, kmax = 1;
    {
        u128 p = ell;
        while (p * ell < (u128(1) << opt.local_max_k)) {
            p *= ell;
            ++kmax;
        }
    }
    while (true) {
        const RingU64 ring = RingU64::Local(ell, k);
        LocalSmith<RingU64> S(ring, *sc.A);
        S.run();
        if (S.pivots() < sc.exp_rank && k < kmax) {
            k = std::min(kmax, 2 * k);
            continue;
        }
        st.k = k;
        st.smith_pivots = S.pivots();
        st.smith_ops = S.ops();
        st.smith_nnz = S.nnz();
        const RingU64 uring = RingU64::Field(ell);
        const BigInt unit = FromU64(ell), Mbig = FromU64(ring.M), spow = FromU64(ring.M / ell);
        SaturateSide(ring, uring, unit, spow, Mbig, *sc.A, S, *sc.X, delta, st);
        break;
    }
    st.seconds = since(t0);
    log.push_back(st);
}

template <class R>
void FieldSide(const R& ring, const BigInt& m, SideCtx& sc, BigInt& delta, std::vector<SatStep>& log, const char* mode) {
    const auto t0 = clk::now();
    SatStep st;
    st.side = sc.side;
    st.piece = "c" + std::to_string(m.bit_length()) + "b";
    st.mode = mode;
    LocalSmith<R> S(ring, *sc.A);
    S.run();
    st.smith_pivots = S.pivots();
    st.smith_ops = S.ops();
    st.smith_nnz = S.nnz();
    SaturateSide(ring, ring, m, BigInt(1), m, *sc.A, S, *sc.X, delta, st);
    st.seconds = since(t0);
    log.push_back(st);
}

// composite mode on c with the gcd splitting (both sides per piece)
void CompositePiece(const BigInt& c, SideCtx& sb, SideCtx& sp, BigInt& delta, std::vector<SatStep>& log, uint32_t& splits,
                    const BigInt& lim, uint32_t& mw_passes) {
    std::vector<BigInt> stack{c};
    while (!stack.empty()) {
        const BigInt m = stack.back();
        stack.pop_back();
        if (m <= BigInt(1)) continue;
        if (BigInt::gcd(delta, m) == BigInt(1)) continue;
        try {
            if (m < lim) {
                const RingU64 ring = RingU64::Field(RedU64(m, ~0ull >> 1));   // m < 2^62: its own residue
                FieldSide(ring, m, sb, delta, log, "composite64");
                // (fix round 1, F8) the side-Phi pass only while the piece still divides the index
                if (BigInt::gcd(delta, m) != BigInt(1)) FieldSide(ring, m, sp, delta, log, "composite64");
            } else {
                RingBig ring;
                ring.M = m;
                ++mw_passes;
                FieldSide(ring, m, sb, delta, log, "multiword");
                if (BigInt::gcd(delta, m) != BigInt(1)) FieldSide(ring, m, sp, delta, log, "multiword");
            }
        } catch (const Split& e) {
            ++splits;
            const BigInt g = e.g;
            ENGINE_CHECK_MSG(g > BigInt(1) && g < m, "Saturate: a trivial split");
            stack.push_back(m / g);
            stack.push_back(g);
        }
    }
}
}  // namespace

SaturationResult Saturate(const IsoTreeCarrierZ& c, PackedZ& B, PackedZ& Phi, BigInt delta, const FreeBasisOptions& opt,
                          bool escalate_allowed, const testing::FreeBasisHooks* hooks) {
    const auto t0 = clk::now();
    SaturationResult res;
    delta = delta.abs();
    ENGINE_CHECK_MSG(!delta.is_zero(), "Saturate: det(Phi^T B) = 0");
    const SparseZ d1t = c.d(1).transpose();   // (d^1)^T: rows = C^1
    SideCtx sb{'B', &c.d(0), &B, c.dim(0) - c.b0()};
    SideCtx sp{'P', &d1t, &Phi, c.D() + c.b0() - c.b1()};
    std::vector<uint32_t> S = hecke3::RecordTorsionPrimes(c.N(), c.n());
    S.push_back(2);
    S.push_back(3);
    std::sort(S.begin(), S.end());
    S.erase(std::unique(S.begin(), S.end()), S.end());
    const uint32_t lim_bits = hooks ? hooks->cofactor_limit_bits : 62u;
    const uint32_t trial_bound = hooks ? hooks->trial_bound : (1u << 20);
    ENGINE_CHECK_MSG(lim_bits >= 2 && lim_bits <= 62 && trial_bound <= (1u << 20), "Saturate: hooks out of range");
    const BigInt lim = BigInt::pow(BigInt(2), lim_bits);
    while (delta != BigInt(1)) {
        if (res.iterations >= opt.max_iter) {
            res.refused = true;
            std::ostringstream o;
            o << "REFUSED [SaturationCofactor] (N, n) = (" << c.N() << ", " << c.n() << "): |Delta| = " << FactorString(delta)
              << " after " << res.iterations << " iterations";
            res.refusal = o.str();
            break;
        }
        ++res.iterations;
        const BigInt before = delta;
        // the pieces
        std::vector<uint64_t> primes;
        BigInt rest = delta;
        for (uint32_t l : S)
            if (rest.mod_u32(l) == 0) {
                primes.push_back(l);
                while (rest.mod_u32(l) == 0) rest = rest.div_exact_u32(l);
            }
        for (uint32_t l : Primes20()) {
            if (rest == BigInt(1) || l >= trial_bound) break;
            if (rest.mod_u32(l) != 0) continue;
            if (std::find(primes.begin(), primes.end(), uint64_t(l)) == primes.end()) primes.push_back(l);
            while (rest.mod_u32(l) == 0) rest = rest.div_exact_u32(l);
        }
        std::sort(primes.begin(), primes.end());
        if (rest >= lim) {
            if (escalate_allowed) {
                res.wants_escalation = true;
                break;
            }
            if (!opt.multiword || rest.bit_length() > opt.multiword_max_bits) {
                res.refused = true;
                std::ostringstream o;
                o << "REFUSED [SaturationCofactor] (N, n) = (" << c.N() << ", " << c.n() << "): a cofactor of " << rest.bit_length()
                  << " bits at K = K_max (" << (opt.multiword ? "above multiword_max_bits" : "multiword mode disabled") << ")";
                res.refusal = o.str();
                break;
            }
        }
        for (uint64_t l : primes) {
            if (delta.mod_u32(static_cast<uint32_t>(l)) != 0) continue;
            LocalPiece(l, sb, delta, opt, res.log);
            if (delta.mod_u32(static_cast<uint32_t>(l)) != 0) continue;
            LocalPiece(l, sp, delta, opt, res.log);
        }
        if (rest > BigInt(1)) CompositePiece(rest, sb, sp, delta, res.log, res.splits, lim, res.multiword_passes);
        // (fix round 1, F4) an iteration that replaced no column is deterministic: the next one would be identical -- refuse at once
        if (delta == before) {
            res.refused = true;
            res.stalled = true;
            std::ostringstream o;
            o << "REFUSED [SaturationStall] (N, n) = (" << c.N() << ", " << c.n() << "): iteration " << res.iterations
              << " replaced no column; |Delta| = " << FactorString(delta) << ", pieces";
            for (uint64_t l : primes) o << " " << l;
            if (rest > BigInt(1)) o << " c" << rest.bit_length() << "b";
            res.refusal = o.str();
            break;
        }
    }
    res.done = delta == BigInt(1);
    res.delta = delta;
    res.seconds = since(t0);
    return res;
}

// =================================================================================================================================
// Sec. 1.4 the certificate
// =================================================================================================================================
Certificate CertifyBasis(const IsoTreeCarrierZ& c, const PackedZ& B, const PackedZ& Phi, uint32_t skip_q) {
    const auto t0 = clk::now();
    Certificate C;
    const uint32_t r = B.cols, nv = c.dim(1);
    ENGINE_CHECK_MSG(Phi.cols == r && B.rows == nv && Phi.rows == nv, "CertifyBasis: shapes");
    // (fix round 1, F6) the premise rank L <= r at a witness prime: rank_q d^1 + rank_q d^0 >= dim C^1 - r (rank_q <= rank_Q)
    {
        const SparseZ d1t = c.d(1).transpose();   // rows = the columns of d^1
        const SparseZ& d0 = c.d(0);
        auto rank_of = [](const SparseZ& A, uint32_t q, uint32_t len) {
            const std::vector<uint32_t> res = Residues(A, q);
            EchelonModQ E(q, len);
            for (uint32_t i = 0; i < A.rows; ++i) {
                SpVec v;
                for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
                    if (!res[k]) continue;
                    v.idx.push_back(A.col[k]);
                    v.val.push_back(res[k]);
                }
                E.insert(v);
            }
            return E.rank();
        };
        for (uint32_t q : integral::CrtPrimes(64)) {
            if (q == skip_q || (6ull * c.N()) % q == 0) continue;
            if (C.rank_tries >= 8) break;
            ++C.rank_tries;
            C.rank_q = q;
            C.rank_d1 = rank_of(d1t, q, c.dim(2));
            C.rank_d0 = rank_of(d0, q, c.dim(0));
            if (uint64_t(C.rank_d1) + C.rank_d0 + r >= nv) {
                C.rank_ok = true;
                break;
            }
        }
    }
    C.cocycles = ExactCocycles(c, B);
    C.kills_d0 = ExactKillsD0(c, Phi);
    const std::vector<BigInt> G = PairingMatrix(Phi, B);
    uint64_t hb = 0;
    for (uint32_t i = 0; i < r; ++i) {
        BigInt s(0);
        for (uint32_t j = 0; j < r; ++j) s += G[size_t(i) * r + j] * G[size_t(i) * r + j];
        hb += (s.bit_length() + 1) / 2;
    }
    C.hadamard_bits = static_cast<uint32_t>(hb);
    // primes with prod > 2 * Hadamard: sum (bitlen - 1) >= hb + 2
    uint32_t need = static_cast<uint32_t>((hb + 2 + 28) / 29);
    if (need == 0) need = 1;
    const std::vector<uint32_t> P = integral::CrtPrimes(need);
    int sign = 0;
    bool pm1 = true;
    for (uint32_t p : P) {
        std::vector<uint32_t> A(size_t(r) * r);
        for (size_t e = 0; e < A.size(); ++e) A[e] = G[e].mod_u32(p);
        const uint32_t d = DetModPrime(std::move(A), r, p);
        ++C.primes;
        const int s = d == 1 ? 1 : (d == p - 1 ? -1 : 0);
        if (s == 0 || (sign != 0 && s != sign)) {
            pm1 = false;
            break;
        }
        sign = s;
    }
    if (r == 0) sign = 1;
    C.sign = pm1 ? sign : 0;
    C.ok = C.cocycles && C.kills_d0 && pm1 && C.rank_ok;
    C.fnv_B = PackedFnv(B);
    C.fnv_Phi = PackedFnv(Phi);
    C.car_N = c.N(); C.car_n = c.n(); C.car_dim0 = c.dim(0); C.car_dim1 = c.dim(1); C.car_dim2 = c.dim(2);
    C.seconds = since(t0);
    return C;
}

// =================================================================================================================================
// the driver
// =================================================================================================================================
std::string Q0Key(const PairsResult& P) {
    std::ostringstream o;
    o << "q0=" << P.q0 << " q0_confirm=" << P.q0_confirm << " q0_skipped=";
    if (P.q0_skipped.empty()) o << "-";
    for (size_t i = 0; i < P.q0_skipped.size(); ++i) o << (i ? "/" : "") << P.q0_skipped[i];
    return o.str();
}

FreeBasisResult BuildFreeBasis(const IsoTreeCarrierZ& c, const FreeBasisOptions& opt, const testing::FreeBasisHooks* hooks) {
    const auto t0 = clk::now();
    FreeBasisResult R;
    auto tp = clk::now();
    auto vphase = [&](const std::string& what, double t) {
        if (opt.verbose)
            std::cout << "EXZ2_P1B_PHASE cell=" << c.N() << ":" << c.n() << " " << what << std::fixed << std::setprecision(3) << " t=" << t << std::endl;
    };
    R.pairs = BuildPairs(c, opt.pairs, hooks ? &hooks->pairs : nullptr);
    R.t_pairs = since(tp);
    vphase("pairs", R.t_pairs);
    if (R.pairs.refused) {
        R.refused = true;
        R.refusal = R.pairs.refusal;
        R.seconds = since(t0);
        return R;
    }
    bool first = true;
    while (true) {
        const auto ts = clk::now();
        R.sums = LatticeSums(c, R.pairs.pairs, opt, hooks);
        R.t_sums += since(ts);
        if (first) {
            R.sums0 = R.sums;
            first = false;
        }
        if (R.sums.refused) {   // (fix round 1, F1 (b)) the sums' RssCeiling
            R.refused = true;
            R.refusal = R.sums.refusal;
            R.K_used = static_cast<uint32_t>(R.pairs.pairs.size());
            R.seconds = since(t0);
            return R;
        }
        R.B = R.sums.B;
        R.Phi_cert = R.sums.Phi;
        const bool can_escalate = R.pairs.pairs.size() < opt.K_max;
        const auto tt = clk::now();
        SaturationResult s = Saturate(c, R.B, R.Phi_cert, R.sums.delta, opt, can_escalate, hooks);
        R.t_sat += since(tt);
        vphase("saturate K=" + std::to_string(R.pairs.pairs.size()), since(tt));
        s.iterations += R.sat.iterations;
        s.splits += R.sat.splits;
        s.multiword_passes += R.sat.multiword_passes;
        s.log.insert(s.log.begin(), R.sat.log.begin(), R.sat.log.end());
        R.sat = s;
        if (s.wants_escalation) {
            tp = clk::now();
            ExtendPairs(c, R.pairs, opt.pairs);
            R.t_pairs += since(tp);
            if (R.pairs.refused) {   // (fix round 1, F5) a refusal during the escalation is a refusal
                R.refused = true;
                R.refusal = R.pairs.refusal;
                R.K_used = static_cast<uint32_t>(R.pairs.pairs.size());
                R.seconds = since(t0);
                return R;
            }
            continue;
        }
        break;
    }
    R.K_used = static_cast<uint32_t>(R.pairs.pairs.size());
    if (R.sat.refused) {
        R.refused = true;
        R.refusal = R.sat.refusal;
        R.seconds = since(t0);
        return R;
    }
    const auto tc = clk::now();
    R.cert = CertifyBasis(c, R.B, R.Phi_cert, R.pairs.q0);
    R.t_cert = since(tc);
    vphase("certificate", R.t_cert);
    ENGINE_CHECK_MSG(R.cert.rank_ok, "BuildFreeBasis: no rank witness prime (rank_q d^1 + rank_q d^0 < dim C^1 - r at 8 primes)");
    ENGINE_CHECK_MSG(R.cert.ok, "BuildFreeBasis: the tracked index is 1 but det(Phi^T B) != +-1 (a defect)");
    R.saturation = "det_pm1";
    R.fnv_B = PackedFnv(R.B);
    R.fnv_Phi = PackedFnv(R.Phi_cert);
    R.bits_B = R.B.max_bits();
    R.bits_Phi = R.Phi_cert.max_bits();
    R.seconds = since(t0);
    return R;
}

}  // namespace freez
}  // namespace integral_descent
