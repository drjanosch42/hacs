// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 20, WP-LV3) the generic m-block reader -- see MBlockReader.h.
#include "MBlockReader.h"

#include "NumberField/BigInt.h"
#include "NumberField/PolyAlg.h"

#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace hecke_ext {
namespace lvalue {
namespace mblock {

// =====================================================================================================================
// O
// =====================================================================================================================
LocalOrder::LocalOrder(const Zq& R, uint32_t e, uint64_t u) : R_(R), e_(e), u_(u % R.q) {
    ENGINE_CHECK_MSG(e >= 1, "mblock::LocalOrder: e >= 1");
    ENGINE_CHECK_MSG(e == 1 || R.unit(u_), "mblock::LocalOrder: u must be an ell-unit");
    ellu_ = R_.mul(R.ell % R.q, u_);
}
LocalOrder::Elt LocalOrder::scalar(uint64_t a) const {
    Elt x(e_, 0);
    x[0] = a % R_.q;
    return x;
}
LocalOrder::Elt LocalOrder::pi_pow(uint32_t s) const {
    // pi^s = (ell u)^{s / e} pi^{s mod e}
    Elt x(e_, 0);
    x[s % e_] = R_.pow(ellu_, s / e_);
    if (e_ == 1) x[0] = R_.pow(R_.ell % R_.q, s);
    return x;
}
LocalOrder::Elt LocalOrder::add(const Elt& x, const Elt& y) const {
    Elt z(e_);
    for (uint32_t i = 0; i < e_; ++i) z[i] = R_.add(x[i], y[i]);
    return z;
}
LocalOrder::Elt LocalOrder::sub(const Elt& x, const Elt& y) const {
    Elt z(e_);
    for (uint32_t i = 0; i < e_; ++i) z[i] = R_.sub(x[i], y[i]);
    return z;
}
LocalOrder::Elt LocalOrder::mul(const Elt& x, const Elt& y) const {
    if (e_ == 1) return Elt{R_.mul(x[0], y[0])};
    std::vector<uint64_t> c(2 * e_ - 1, 0);
    for (uint32_t i = 0; i < e_; ++i) {
        if (!x[i]) continue;
        for (uint32_t j = 0; j < e_; ++j)
            if (y[j]) c[i + j] = R_.add(c[i + j], R_.mul(x[i], y[j]));
    }
    Elt z(e_);
    for (uint32_t i = 0; i < e_; ++i) z[i] = c[i];
    for (uint32_t i = e_; i < 2 * e_ - 1; ++i) z[i - e_] = R_.add(z[i - e_], R_.mul(c[i], ellu_));
    return z;
}
uint32_t LocalOrder::vpi(const Elt& x) const {
    uint32_t v = cap();
    for (uint32_t i = 0; i < e_; ++i) {
        const uint64_t a = x[i] % R_.q;
        if (!a) continue;
        v = std::min(v, e_ * R_.val(a) + i);
    }
    return v;
}
uint64_t LocalOrder::residue(const Elt& x) const { return x[0] % R_.ell; }
LocalOrder::Elt LocalOrder::div_pi_pow(const Elt& x, uint32_t s) const {
    Elt y = x;
    const uint64_t iu = e_ == 1 ? 1 : R_.inv(u_);
    for (uint32_t k = 0; k < s; ++k) {
        // y / pi: sum_{i>=1} y_i pi^{i-1} + (y_0 / (ell u)) pi^{e-1}
        ENGINE_CHECK_MSG(y[0] % R_.ell == 0, "mblock::LocalOrder::div_pi_pow: not divisible by pi");
        const uint64_t a0 = R_.mul(y[0] / R_.ell, iu);
        Elt z(e_, 0);
        for (uint32_t i = 1; i < e_; ++i) z[i - 1] = y[i];
        z[e_ - 1] = R_.add(z[e_ - 1], a0);
        y = z;
    }
    return y;
}
std::string LocalOrder::to_string() const {
    if (e_ == 1) return "Z_" + std::to_string(R_.ell);
    return "Z_" + std::to_string(R_.ell) + "[pi]/(pi^" + std::to_string(e_) + " - " + std::to_string(R_.ell) + "*" +
           std::to_string(u_ % R_.ell) + "+O(" + std::to_string(R_.ell) + "^2))";
}

LocalOrder::Elt EvalPoly(const LocalOrder& O, const Vec& P, const LocalOrder::Elt& x) {
    LocalOrder::Elt acc = O.zero();
    for (size_t i = P.size(); i-- > 0;) acc = O.add(O.mul(acc, x), O.scalar(P[i]));
    return acc;
}

namespace {
LocalOrder::Elt PairOn(const LocalOrder& O, const std::vector<LocalOrder::Elt>& g, const Mat& Gram, const std::vector<LocalOrder::Elt>& h) {
    LocalOrder::Elt v = O.zero();
    for (size_t a = 0; a < g.size(); ++a)
        for (size_t b = 0; b < h.size(); ++b)
            if (Gram[a][b]) v = O.add(v, O.mul(O.mul(g[a], O.scalar(Gram[a][b])), h[b]));
    return v;
}
} // namespace

// =====================================================================================================================
// roots
// =====================================================================================================================
namespace {
using Elt = LocalOrder::Elt;

// The Taylor coefficients Q_i of P(x0 + X) (P over R, x0 in O).
std::vector<Elt> Taylor(const LocalOrder& O, const Vec& P, const Elt& x0) {
    std::vector<Elt> c;
    for (uint64_t a : P) c.push_back(O.scalar(a));
    const size_t d = c.size() - 1;
    // repeated synthetic division by (X - x0)
    for (size_t k = 0; k < d; ++k)
        for (size_t i = d; i-- > k;) c[i] = O.add(c[i], O.mul(x0, c[i + 1]));
    return c;
}

// Roots in F_ell of sum c_j w^j (c_0, c_d != 0), with multiplicity; brute force (ell is a small prime here).
std::vector<std::pair<uint64_t, uint32_t>> RootsModEll(uint64_t ell, std::vector<uint64_t> c) {
    std::vector<std::pair<uint64_t, uint32_t>> out;
    const Zq F(ell, 1);
    for (auto& x : c) x %= ell;
    for (uint64_t w = 1; w < ell && c.size() > 1; ++w) {
        uint32_t mult = 0;
        for (;;) {
            if (c.size() <= 1) break;
            uint64_t acc = 0;
            for (size_t i = c.size(); i-- > 0;) acc = F.add(F.mul(acc, w), c[i]);
            if (acc != 0) break;
            // divide by (X - w)
            std::vector<uint64_t> q(c.size() - 1, 0);
            uint64_t rem = c.back();
            for (size_t i = c.size() - 1; i-- > 0;) {
                q[i] = rem;
                rem = F.add(c[i], F.mul(rem, w));
            }
            c = q;
            ++mult;
        }
        if (mult) out.push_back({w, mult});
    }
    return out;
}

uint64_t PrimitiveRoot(uint64_t ell) {
    const Zq F(ell, 1);
    std::vector<uint64_t> fac;
    uint64_t m = ell - 1;
    for (uint64_t p = 2; p * p <= m; ++p)
        if (m % p == 0) {
            fac.push_back(p);
            while (m % p == 0) m /= p;
        }
    if (m > 1) fac.push_back(m);
    for (uint64_t g = 2; g < ell; ++g) {
        bool ok = true;
        for (uint64_t p : fac) ok = ok && F.pow(g, (ell - 1) / p) != 1;
        if (ok) return g;
    }
    return 1;
}

struct Finder {
    const Zq& R;
    const Vec& P;
    RootReading& out;
    bool fail(const std::string& why) {
        if (out.refusal.empty()) out.refusal = why;
        return false;
    }
    void emit(uint32_t oi, const Elt& rho, const std::string& name) {
        const LocalOrder& O = out.orders[oi];
        Vec dP;
        for (size_t i = 1; i < P.size(); ++i) dP.push_back(R.mul(i % R.q, P[i]));
        Root r;
        r.order = oi;
        r.rho = rho;
        r.vpi_dP = O.vpi(EvalPoly(O, dP, rho));
        r.name = name;
        out.roots.push_back(r);
    }
    // The mu roots x with v_pi(x) > s0 of P(rho0 + x), inside O.
    bool refine(uint32_t oi, Elt rho0, uint32_t s0, uint32_t mu, const std::string& name, uint32_t depth) {
        const LocalOrder& O = out.orders[oi];
        if (depth > 4 * O.cap() + 8) return fail("mblock: the digit refinement did not terminate");
        const std::vector<Elt> Q = Taylor(O, P, rho0);
        uint32_t z = 0;
        while (z < mu && O.vpi(Q[z]) >= O.cap()) ++z;
        if (z >= 2) return fail("mblock: a multiple root of the block charpoly to the working precision (" + name + ")");
        if (z == 1) emit(oi, rho0, name);
        if (z == mu) return true;
        // lower hull of (i, v_pi Q_i), i = z..mu
        struct Pt { int64_t x, y; };
        std::vector<Pt> hull;
        for (uint32_t i = z; i <= mu; ++i) {
            const Pt q{static_cast<int64_t>(i), static_cast<int64_t>(O.vpi(Q[i]))};
            if (q.y >= static_cast<int64_t>(O.cap()) && i != mu) continue;
            while (hull.size() >= 2) {
                const Pt& o = hull[hull.size() - 2];
                const Pt& a = hull.back();
                if ((a.x - o.x) * (q.y - o.y) - (a.y - o.y) * (q.x - o.x) <= 0) hull.pop_back();
                else break;
            }
            hull.push_back(q);
        }
        if (hull.front().x != static_cast<int64_t>(z) || hull.back().x != static_cast<int64_t>(mu))
            return fail("mblock: the cluster's Newton polygon is degenerate (" + name + ")");
        for (size_t sgi = 1; sgi < hull.size(); ++sgi) {
            const Pt a = hull[sgi - 1], b = hull[sgi];
            const int64_t dx = b.x - a.x, dy = a.y - b.y;
            if (dy <= 0 || dy % dx != 0)
                return fail("mblock: a refinement slope is not integral in v_pi (the root needs a larger ramified extension than " +
                            O.to_string() + ")");
            const uint32_t s = static_cast<uint32_t>(dy / dx);
            if (s <= s0) return fail("mblock: cluster bookkeeping (slope <= the previous digit)");
            std::vector<uint64_t> c;
            for (int64_t i = a.x; i <= b.x; ++i) {
                const uint32_t vi = O.vpi(Q[static_cast<size_t>(i)]);
                const int64_t line = a.y - (i - a.x) * static_cast<int64_t>(s);
                c.push_back(static_cast<int64_t>(vi) == line ? O.residue(O.div_pi_pow(Q[static_cast<size_t>(i)], static_cast<uint32_t>(line))) : 0);
            }
            const auto w = RootsModEll(R.ell, c);
            uint32_t tot = 0;
            for (const auto& x : w) tot += x.second;
            if (tot != static_cast<uint32_t>(dx))
                return fail("mblock: a residual polynomial has an irreducible factor of degree > 1 over F_ell (an unramified residue extension)");
            for (const auto& x : w) {
                const Elt next = O.add(rho0, O.mul(O.scalar(x.first), O.pi_pow(s)));
                if (!refine(oi, next, s, x.second, name, depth + 1)) return false;
            }
        }
        return true;
    }
};
} // namespace

RootReading RootsOfNilpotentCharpoly(const Zq& R, const Vec& P0) {
    RootReading out;
    Vec P = P0;
    for (auto& a : P) a %= R.q;
    const size_t r = P.size() - 1;
    ENGINE_CHECK_MSG(P.size() >= 2 && P.back() == 1 % R.q, "mblock::RootsOfNilpotentCharpoly: P must be monic of degree >= 1");
    for (size_t i = 0; i < r; ++i)
        if (P[i] % R.ell) {
            out.refusal = "mblock: the block charpoly is not == x^r mod ell (the block is not an m-block)";
            return out;
        }
    Finder F{R, P, out};
    // the trivial order Z_ell is order 0
    out.orders.push_back(LocalOrder(R, 1, 1));
    uint32_t z = 0;
    while (z < r && P[z] == 0) ++z;
    if (z >= 2) {
        out.refusal = "mblock: x^2 divides the block charpoly to the working precision (a multiple root at lambda_0)";
        return out;
    }
    if (z == 1) F.emit(0, out.orders[0].scalar(0), "rho[0]");
    // the top-level Newton polygon of P / x^z over Z_ell (the NumberField core)
    std::vector<nf::BigInt> f;
    for (size_t i = z; i <= r; ++i) f.push_back(nf::BigInt::from_u64(P[i]));
    const std::vector<nf::NewtonSegment> segs = nf::newton_polygon(f, static_cast<uint32_t>(R.ell));
    out.newton_segments = static_cast<uint32_t>(segs.size());
    int64_t covered = 0;
    for (const nf::NewtonSegment& sg : segs) {
        const int64_t dx = sg.x1 - sg.x0, dy = sg.y0 - sg.y1;
        covered += dx;
        if (dy <= 0) {
            out.refusal = "mblock: a non-positive slope (a root of valuation 0: not an m-block)";
            return out;
        }
        const int64_t g = std::gcd(dx, dy);
        const uint32_t h = static_cast<uint32_t>(dy / g), e = static_cast<uint32_t>(dx / g);
        std::vector<uint64_t> c;
        for (int64_t i = sg.x0; i <= sg.x1; i += e) {
            const int64_t line = sg.y0 - ((i - sg.x0) / e) * h;
            const uint64_t a = P[static_cast<size_t>(i) + z];
            uint64_t cj = 0;
            if (a && static_cast<int64_t>(R.val(a)) == line) {
                uint64_t x = a;
                for (int64_t k = 0; k < line; ++k) x /= R.ell;
                cj = x % R.ell;
            }
            c.push_back(cj);
        }
        const auto ys = RootsModEll(R.ell, c);
        uint32_t tot = 0;
        for (const auto& y : ys) tot += y.second;
        if (tot != c.size() - 1) {
            out.refusal = "mblock: a residual polynomial of the top Newton polygon has an irreducible factor of degree > 1 over F_ell "
                          "(an unramified residue extension f > 1)";
            return out;
        }
        if (e > 1 && (R.ell - 1) % e != 0) {
            out.refusal = "mblock: a segment of denominator " + std::to_string(e) + " with e not dividing ell - 1";
            return out;
        }
        const std::string sname = std::to_string(h) + "/" + std::to_string(e);
        for (const auto& yy : ys) {
            const uint64_t y = yy.first;
            if (e == 1) {
                const Elt rho0 = out.orders[0].scalar(R.mul(y, R.pow(R.ell % R.q, h)));
                if (!F.refine(0, rho0, h, yy.second, "rho[" + sname + ",y=" + std::to_string(y) + "]", 0)) return out;
                continue;
            }
            uint32_t hp = 1;
            while ((static_cast<uint64_t>(h) * hp) % e != 1) ++hp;
            const uint64_t kk = (static_cast<uint64_t>(h) * hp - 1) / e;
            const Zq F1(R.ell, 1);
            const uint64_t u = R.pow(y, hp);
            const uint64_t cc = F1.pow(F1.inv(y), kk);
            out.orders.push_back(LocalOrder(R, e, u));
            const uint32_t oi = static_cast<uint32_t>(out.orders.size() - 1);
            const LocalOrder& O = out.orders[oi];
            const uint64_t gz = F1.pow(PrimitiveRoot(R.ell), (R.ell - 1) / e);
            uint64_t zeta = 1;
            for (uint32_t t = 0; t < e; ++t, zeta = F1.mul(zeta, gz)) {
                const Elt rho0 = O.mul(O.scalar(F1.mul(zeta, cc)), O.pi_pow(h));
                if (!F.refine(oi, rho0, h, yy.second,
                              "rho[" + sname + ",y=" + std::to_string(y) + ",zeta^" + std::to_string(t) + "]", 0))
                    return out;
            }
        }
    }
    if (covered + z != static_cast<int64_t>(r)) {
        out.refusal = "mblock: the Newton polygon does not cover the degree";
        return out;
    }
    out.complete = out.roots.size() == r && out.refusal.empty();
    if (!out.complete && out.refusal.empty()) out.refusal = "mblock: " + std::to_string(out.roots.size()) + " roots for degree " + std::to_string(r);
    return out;
}

// =====================================================================================================================
// eigen-lines, blocks
// =====================================================================================================================
EigenLineO EigenLineAt(const LocalOrder& O, const Mat& t, const Vec& P, const Elt& rho) {
    const Zq& R = O.R();
    const size_t r = t.size();
    ENGINE_CHECK_MSG(P.size() == r + 1, "mblock::EigenLineAt: deg P != size of t");
    // Q_rho = P / (x - rho): q_{r-1} = 1, q_{i-1} = P_i + rho q_i
    std::vector<Elt> qs(r);
    Elt rem = O.scalar(P[r]);
    qs[r - 1] = rem;
    for (size_t i = r - 1; i >= 1; --i) {
        rem = O.add(O.scalar(P[i]), O.mul(rem, rho));
        qs[i - 1] = rem;
    }
    std::vector<Mat> tpow{Mat(r, Vec(r, 0))};
    for (size_t i = 0; i < r; ++i) tpow[0][i][i] = 1 % R.q;
    for (size_t s = 1; s < r; ++s) tpow.push_back(hidapair::MatMul(R, tpow.back(), t));
    EigenLineO E;
    size_t best = 0;
    std::vector<uint32_t> contents;
    std::vector<std::vector<Elt>> cols(r, std::vector<Elt>(r, O.zero()));
    for (size_t j = 0; j < r; ++j) {
        uint32_t c = O.cap();
        for (size_t a = 0; a < r; ++a) {
            Elt zz = O.zero();
            for (size_t s = 0; s < r; ++s)
                if (tpow[s][a][j]) zz = O.add(zz, O.mul(qs[s], O.scalar(tpow[s][a][j])));
            cols[j][a] = zz;
            c = std::min(c, O.vpi(zz));
        }
        contents.push_back(c);
        if (c < contents[best]) best = j;
    }
    E.content = contents[best];
    E.g = cols[best];
    for (Elt& x : E.g) x = O.div_pi_pow(x, E.content);
    uint32_t chk = O.cap();
    for (size_t a = 0; a < r; ++a) {
        Elt s = O.zero();
        for (size_t b = 0; b < r; ++b)
            if (t[a][b]) s = O.add(s, O.mul(O.scalar(t[a][b]), E.g[b]));
        chk = std::min(chk, O.vpi(O.sub(s, O.mul(rho, E.g[a]))));
    }
    E.eigen_check = chk;
    return E;
}

using zk::MatPow;   // the shared kit (../ZellK.h)
using zk::MatVec;

using hidapair::ReduceMat;
using hidapair::Restrict;

Block BlockOf(const Zq& R, const Mat& A, uint64_t lambda0) {
    const size_t d = A.size();
    Mat t = A;
    for (size_t i = 0; i < d; ++i) t[i][i] = R.sub(t[i][i] % R.q, lambda0 % R.q);
    Block b;
    if (d == 0) return b;
    const Mat tK = MatPow(R, t, 4ull * R.m);
    b.basis = hidapair::SmithKernel(R, tK, static_cast<uint32_t>(d)).kernel;
    if (b.basis.empty()) return b;
    b.t = Restrict(R, t, b.basis);
    b.charpoly = hidapair::Charpoly(R, b.t);
    return b;
}

Block JointBlockOf(const Zq& R, const Mat& A, uint64_t lambda0, const std::vector<std::pair<const Mat*, uint64_t>>& others) {
    Block b = BlockOf(R, A, lambda0);
    for (const auto& o : others) {
        if (b.basis.empty()) return b;
        // the generalised (lambda_p)-eigenspace of A_p restricted to the current block, in block coordinates
        const Mat X = Restrict(R, *o.first, b.basis);
        const Block sub = BlockOf(R, X, o.second);
        std::vector<Vec> nb;
        for (const Vec& c : sub.basis) {
            Vec v(A.size(), 0);
            for (size_t j = 0; j < c.size(); ++j)
                if (c[j])
                    for (size_t i = 0; i < v.size(); ++i) v[i] = R.add(v[i], R.mul(c[j], b.basis[j][i]));
            nb.push_back(v);
        }
        b.basis = nb;
        if (b.basis.empty()) return b;
        Mat t = A;
        for (size_t i = 0; i < t.size(); ++i) t[i][i] = R.sub(t[i][i] % R.q, lambda0 % R.q);
        b.t = Restrict(R, t, b.basis);
        b.charpoly = hidapair::Charpoly(R, b.t);
    }
    return b;
}

uint64_t EisensteinT2(const Zq& R, uint32_t n) { return R.add(1 % R.q, R.pow(2, n + 1)); }

std::vector<std::pair<uint32_t, uint64_t>> EisensteinJoint(const Zq& R, uint32_t n, const std::vector<uint32_t>& primes) {
    std::vector<std::pair<uint32_t, uint64_t>> out;
    for (uint32_t p : primes)
        if (p % R.ell != 0) out.push_back({p, R.add(1 % R.q, R.pow(p, n + 1))});
    return out;
}

uint64_t Lambda0FromSubquotient(const std::string& s, const Zq& R, uint32_t n) {
    size_t p = s.find("T_2");
    while (p != std::string::npos) {
        size_t i = p + 3;
        while (i < s.size() && s[i] == ' ') ++i;
        if (i < s.size() && s[i] == '-') {
            ++i;
            while (i < s.size() && s[i] == ' ') ++i;
            size_t j = i;
            while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
            if (j > i) return std::stoull(s.substr(i, j - i)) % R.q;
        }
        p = s.find("T_2", p + 3);
    }
    return EisensteinT2(R, n);
}

// =====================================================================================================================
// the cell reading
// =====================================================================================================================
namespace {
using hidapair::Fibre;
const hidapair::M22 kS{0, -1, 1, 0};
const hidapair::M22 kU{0, -1, 1, 1};

Vec ReduceV(const Zq& R, const Vec& v) {
    Vec o = v;
    for (auto& x : o) x %= R.q;
    return o;
}

uint32_t LengthOfQuotient(const Zq& R, const std::vector<Mat>& ops, size_t r) {
    // length of L / sum_i im(ops_i), L = (Z/ell^M)^r
    Mat rows(r, Vec());
    for (const Mat& X : ops)
        for (size_t i = 0; i < r; ++i)
            for (size_t j = 0; j < r; ++j) rows[i].push_back(X[i][j]);
    if (r == 0) return 0;
    const uint32_t nc = static_cast<uint32_t>(rows[0].size());
    const hidapair::SmithResult s = hidapair::SmithKernel(R, rows, nc);
    uint32_t len = static_cast<uint32_t>((r - s.pivot_valuations.size()) * R.m);
    for (uint32_t v : s.pivot_valuations) len += v;
    return len;
}
} // namespace

MBlockReading ReadMBlockCell(uint32_t n, uint64_t ell, uint32_t M, uint64_t lambda0, const MBlockOptions& opt) {
    // A two-entry memo of the (immutable, deterministic) cell: a route asked for its index statement on two lattices and for
    // its normalisation record builds the cell once.  Values never depend on it (the cell is a pure function of its key).
    static std::mutex mu;
    static std::vector<std::pair<std::string, std::shared_ptr<const hidapair::HidaPairingCell>>> memo;
    std::string key = std::to_string(n) + ":" + std::to_string(ell) + ":" + std::to_string(M);
    for (uint32_t p : opt.primes) key += ":" + std::to_string(p);
    std::shared_ptr<const hidapair::HidaPairingCell> C;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (const auto& e : memo)
            if (e.first == key) C = e.second;
    }
    if (!C) {
        C = std::make_shared<const hidapair::HidaPairingCell>(n, ell, M, opt.primes);
        std::lock_guard<std::mutex> lk(mu);
        memo.insert(memo.begin(), {key, C});
        if (memo.size() > 2) memo.pop_back();
    }
    return ReadMBlockCell(*C, lambda0, opt);
}

MBlockReading ReadMBlockCell(const hidapair::HidaPairingCell& C, uint64_t lambda0, const MBlockOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    const Zq& R = C.R();
    const uint32_t n = C.n(), M = R.m, k = n + 2;
    MBlockReading out;
    out.n = n;
    out.ell = R.ell;
    out.M = M;
    out.lambda0 = lambda0 % R.q;
    out.ne = C.ne();
    out.no = C.no();
    out.adj2 = C.has_prime(2) && C.adjoint_ok(2);
    out.adj3 = C.has_prime(3) && C.adjoint_ok(3);
    out.adj5 = C.has_prime(5) && C.adjoint_ok(5);
    out.iota = C.iota_ok();
    const uint32_t ne = C.ne(), mtot = C.ne() + C.no();
    if (opt.hd_torsion) {
        const Zq R4(R.ell, std::min<uint32_t>(4, M));
        const hidapair::FibreOps F4(R4, n);
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
        const hidapair::SmithResult sr = hidapair::SmithKernel(R4, rows, static_cast<uint32_t>(cols.size()));
        out.HD_free_rank = (n + 1) - static_cast<uint32_t>(sr.pivot_valuations.size());
        for (uint32_t v : sr.pivot_valuations)
            if (v > 0) out.HD_torsion.push_back(v);
        std::sort(out.HD_torsion.begin(), out.HD_torsion.end());
    }
    // H^1_c(D) = W_D (both parities): the dual basis of H^1(V)/tors (R20-3's change of lattice)
    std::vector<Vec> WDall;
    if (opt.lattice_change) {
        WDall = C.sign_basis(Fibre::D, 0);
        const std::vector<Vec> WDo = C.sign_basis(Fibre::D, 1);
        WDall.insert(WDall.end(), WDo.begin(), WDo.end());
    }
    for (uint64_t pw = R.ell; pw <= n; pw *= R.ell) out.v_nfact += static_cast<uint32_t>(n / pw);
    if (opt.lattice_change) {
        // the whole-lattice index of Theta: rows = the W_D dual coordinates, columns = Theta(y_j) for the basis y_j of H^1(D)/tors
        Mat MT(WDall.size(), Vec(mtot, 0));
        for (uint32_t j = 0; j < mtot; ++j) {
            const Vec& Yv = C.Y()[j];
            Vec TV(n + 1, 0);
            for (uint32_t tt = 0; tt <= n; ++tt) {
                uint64_t c = R.mul(C.F().binom(n, tt), Yv[tt] % R.q);
                if (tt % 2) c = R.neg(c);
                TV[n - tt] = c;
            }
            for (size_t i = 0; i < WDall.size(); ++i) MT[i][j] = C.F().eval(TV, WDall[i]);
        }
        out.theta_index_pivots = hidapair::SmithKernel(R, MT, mtot).pivot_valuations;
        std::sort(out.theta_index_pivots.begin(), out.theta_index_pivots.end());
        int32_t s = 0;
        for (uint32_t v : out.theta_index_pivots) s += static_cast<int32_t>(v);
        out.phi_index = static_cast<int32_t>(mtot * out.v_nfact) - s;
    }
    const uint64_t e2 = EisensteinT2(R, n);
    const Zq Re(R.ell, M >= 5 ? M - 2 : M);
    struct SignData {
        Block bc, bd;
        Mat Gram, GramE, tc, td;
        Vec Pc, Pd;
        RootReading roots;
        std::vector<EigenLineO> gc;
        uint32_t lo = 0, hi = 0;
    } sd[2];
    for (int s = 0; s < 2; ++s) {
        SignData& b = sd[s];
        MBlockSign& S = out.sign[s];
        S.name = s == 0 ? "geo-even" : "geo-odd";
        b.lo = s == 0 ? 0 : ne;
        b.hi = s == 0 ? ne : mtot;
        // m = (ell, T_2 - lambda_0) or, when opt.joint names more primes, the joint m = (ell, T_p - lambda_p : p)
        std::vector<Mat> jc, jd;
        for (const auto& pj : opt.joint) {
            jc.push_back(hidapair::Block(C.Ac(pj.first), b.lo, b.hi));
            jd.push_back(hidapair::Block(C.AD(pj.first), b.lo, b.hi));
        }
        std::vector<std::pair<const Mat*, uint64_t>> oc, od;
        for (size_t i = 0; i < opt.joint.size(); ++i) {
            oc.push_back({&jc[i], opt.joint[i].second});
            od.push_back({&jd[i], opt.joint[i].second});
        }
        b.bc = JointBlockOf(R, hidapair::Block(C.Ac(2), b.lo, b.hi), lambda0, oc);
        b.bd = JointBlockOf(R, hidapair::Block(C.AD(2), b.lo, b.hi), lambda0, od);
        S.rank_c = static_cast<uint32_t>(b.bc.basis.size());
        S.rank_D = static_cast<uint32_t>(b.bd.basis.size());
        if (S.rank_c == 0) {
            S.complete = true;
            continue;
        }
        S.charpoly_c = b.bc.charpoly;
        S.charpoly_D = b.bd.charpoly;
        const Mat t2c = hidapair::MatMul(R, b.bc.t, b.bc.t), t2D = hidapair::MatMul(R, b.bd.t, b.bd.t);
        S.tr1 = hidapair::RankModEll(R, b.bc.t); S.tr2 = hidapair::RankModEll(R, t2c); S.tr3 = hidapair::RankModEll(R, hidapair::MatMul(R, t2c, b.bc.t));
        S.dr1 = hidapair::RankModEll(R, b.bd.t); S.dr2 = hidapair::RankModEll(R, t2D); S.dr3 = hidapair::RankModEll(R, hidapair::MatMul(R, t2D, b.bd.t));
        b.Gram.assign(S.rank_c, Vec(S.rank_D));
        for (uint32_t i = 0; i < S.rank_c; ++i)
            for (uint32_t j = 0; j < S.rank_D; ++j) b.Gram[i][j] = C.F().eval(b.bc.basis[i], b.bd.basis[j]);
        const hidapair::SmithResult gs = hidapair::SmithKernel(R, b.Gram, S.rank_D);
        S.gram_pivots = gs.pivot_valuations;
        S.gram_unimodular = gs.pivot_valuations.size() == S.rank_c && S.rank_c == S.rank_D &&
                            std::all_of(gs.pivot_valuations.begin(), gs.pivot_valuations.end(), [](uint32_t v) { return v == 0; });
        // the congruence kind on W_m: L / sum_p (A_p - e_p) L
        {
            std::vector<Mat> ops;
            for (uint32_t p : opt.primes) {
                if (p % R.ell == 0 || !C.has_prime(p)) continue;
                Mat X = Restrict(R, hidapair::Block(C.Ac(p), b.lo, b.hi), b.bc.basis);
                const uint64_t ep = R.add(1 % R.q, R.pow(p, k - 1));
                for (size_t i = 0; i < X.size(); ++i) X[i][i] = R.sub(X[i][i], ep);
                ops.push_back(X);
            }
            S.cong_length = LengthOfQuotient(R, ops, S.rank_c);
        }
        // the eigen part at Re = Z/ell^{M-2} (the block restriction may lose up to two ell-digits; WP-BIL: 37^8 of 37^10)
        b.Pc = ReduceV(Re, b.bc.charpoly);
        b.Pd = ReduceV(Re, b.bd.charpoly);
        b.tc = ReduceMat(Re, b.bc.t);
        b.td = ReduceMat(Re, b.bd.t);
        b.GramE = ReduceMat(Re, b.Gram);
        b.roots = RootsOfNilpotentCharpoly(Re, b.Pc);
        S.newton_segments = b.roots.newton_segments;
        if (!b.roots.complete) {
            S.refusal = b.roots.refusal;
            continue;
        }
        if (b.Pd != b.Pc || S.rank_c != S.rank_D) {
            S.refusal = "mblock: the W- and H^1(D)-blocks have different charpolys (adjunction fails?)";
            continue;
        }
        for (const Root& rt : b.roots.roots) {
            const LocalOrder& O = b.roots.orders[rt.order];
            const EigenLineO gc = EigenLineAt(O, b.tc, b.Pc, rt.rho);
            const EigenLineO gd = EigenLineAt(O, b.td, b.Pd, rt.rho);
            b.gc.push_back(gc);
            MBlockForm f;
            f.name = rt.name;
            f.order = O.to_string();
            f.e = O.e();
            f.order_index = rt.order;
            f.O = O;
            f.rho = rt.rho;
            f.rho_residue_mod_ell = O.residue(O.div_pi_pow(rt.rho, std::min(O.vpi(rt.rho), O.cap() - 1)));
            f.vpi_dP = rt.vpi_dP;
            f.content_c = gc.content;
            f.content_D = gd.content;
            f.cong = rt.vpi_dP >= gc.content ? rt.vpi_dP - gc.content : 1000u;
            f.eigen_check = std::min(gc.eigen_check, gd.eigen_check);
            f.mixed = O.vpi(PairOn(O, gc.g, b.GramE, gd.g));
            // the boundary line: rho == e_2 - lambda_0 to the precision
            f.eisenstein = O.vpi(O.sub(rt.rho, O.scalar(R.sub(e2, lambda0 % R.q)))) >= O.cap() - O.e();
            if (opt.lattice_change) {
                // theta^{-1} = Theta : D -> V, u_t^v -> (-1)^t C(n, t) u_{n-t} (integral); h_rho as an element of D (O-coefficients),
                // Theta(h_rho) in H^1(V)/tors read on the dual basis of W_D = H^1_c(D); c' = its content; v_pi(c_rho) = -c'.
                std::vector<Elt> y(n + 1, O.zero());
                for (uint32_t bb = 0; bb < S.rank_D; ++bb) {
                    if (O.vpi(gd.g[bb]) >= O.cap()) continue;
                    for (uint32_t i = 0; i < b.hi - b.lo; ++i) {
                        const uint64_t coef = b.bd.basis[bb][i] % R.q;
                        if (!coef) continue;
                        const Elt w = O.mul(gd.g[bb], O.scalar(coef));
                        const Vec& Yv = C.Y()[b.lo + i];
                        for (uint32_t tt = 0; tt <= n; ++tt)
                            if (Yv[tt]) y[tt] = O.add(y[tt], O.mul(w, O.scalar(Yv[tt])));
                    }
                }
                std::vector<Elt> TV(n + 1, O.zero());
                for (uint32_t tt = 0; tt <= n; ++tt) {
                    uint64_t c = C.F().binom(n, tt);
                    if (tt % 2) c = R.neg(c);
                    TV[n - tt] = O.mul(y[tt], O.scalar(c));
                }
                uint32_t content = O.cap();
                for (const Vec& wd : WDall) {
                    Elt z = O.zero();
                    for (uint32_t tt = 0; tt <= n; ++tt)
                        if (wd[tt]) z = O.add(z, O.mul(TV[tt], O.scalar(wd[tt])));
                    content = std::min(content, O.vpi(z));
                }
                f.d_lattice = content;
                f.lattice_change = static_cast<int32_t>(O.e() * out.v_nfact) - static_cast<int32_t>(content);
                f.have_lattice = content < O.cap() && f.lattice_change >= 0;
            }
            S.forms.push_back(f);
        }
        S.complete = true;
    }
    // P9: H^1_!(D) = im(H^1_c(D) -> H^1(D)) at m (WP-BIL's ReadCell258 P9, generic)
    if (opt.p9) {
        for (int dpar = 0; dpar < 2; ++dpar) {
            const int ws = dpar == 0 ? 1 : 0;
            SignData& b = sd[ws];
            MBlockSign& S = out.sign[ws];
            if (!S.complete || S.rank_c == 0 || S.forms.empty()) continue;
            const std::vector<Vec> WD = C.sign_basis(Fibre::D, static_cast<uint32_t>(dpar));
            std::vector<Vec> imgs;
            for (const Vec& w : WD) imgs.push_back(C.hecke_symbol(Fibre::D, w, 2));
            const auto co = hidapair::CoordsInBasis(R, WD, imgs);
            const size_t d = WD.size();
            Mat A(d, Vec(d));
            for (size_t j = 0; j < d; ++j) {
                ENGINE_CHECK_MSG(co[j].second + 1 >= M, "mblock::ReadMBlockCell: T_2 W_D not in W_D");
                for (size_t i = 0; i < d; ++i) A[i][j] = co[j].first[i];
            }
            const size_t bw = b.hi - b.lo;
            std::vector<Vec> phis;
            for (const Vec& w : WD) {
                const Vec dc = C.dual_coords(C.iota_shriek(Fibre::D, w));
                phis.emplace_back(dc.begin() + b.lo, dc.begin() + b.hi);
            }
            const Mat ADb = hidapair::Block(C.AD(2), b.lo, b.hi);
            bool inter = true;
            for (size_t j = 0; j < d && inter; ++j) {
                const Vec dc = C.dual_coords(C.iota_shriek(Fibre::D, imgs[j]));
                const Vec lhs(dc.begin() + b.lo, dc.begin() + b.hi);
                inter = lhs == MatVec(R, ADb, phis[j]);
            }
            S.p9_rank_WD = static_cast<uint32_t>(d);
            S.p9_intertwines = inter;
            const Block bm = BlockOf(R, A, lambda0);
            std::vector<Vec> imgm;
            for (const Vec& bv : bm.basis) {
                Vec v(bw, 0);
                for (size_t j = 0; j < d; ++j)
                    if (bv[j])
                        for (size_t kq = 0; kq < bw; ++kq) v[kq] = R.add(v[kq], R.mul(bv[j], phis[j][kq]));
                imgm.push_back(v);
            }
            const auto co2 = hidapair::CoordsInBasis(R, b.bd.basis, imgm);
            Mat Cm;
            for (const auto& c : co2) {
                ENGINE_CHECK_MSG(c.second + 2 >= M, "mblock::ReadMBlockCell: image of W_D,m not in H^1(D)_m");
                Cm.push_back(c.first);
            }
            const hidapair::SmithResult ix = hidapair::SmithKernel(R, Cm, static_cast<uint32_t>(b.bd.basis.size()));
            S.p9_index_pivots = ix.pivot_valuations;
            std::sort(S.p9_index_pivots.begin(), S.p9_index_pivots.end());
            S.p9_charpoly = bm.charpoly;
            const Vec Pm = ReduceV(Re, bm.charpoly);
            if (Pm != b.Pc) continue;   // the roots of b apply only when the charpolys agree (mod ell^{M-2})
            S.have_p9 = true;
            const Mat tm = ReduceMat(Re, bm.t), Ce = ReduceMat(Re, Cm);
            for (size_t fi = 0; fi < S.forms.size(); ++fi) {
                MBlockForm& f = S.forms[fi];
                const LocalOrder& O = b.roots.orders[f.order_index];
                const EigenLineO eI = EigenLineAt(O, tm, Pm, f.rho);
                std::vector<Elt> h(b.bd.basis.size(), O.zero());
                for (size_t bb = 0; bb < b.bd.basis.size(); ++bb)
                    for (size_t a = 0; a < eI.g.size(); ++a)
                        if (Ce[a][bb]) h[bb] = O.add(h[bb], O.mul(eI.g[a], O.scalar(Ce[a][bb])));
                f.p9 = O.vpi(PairOn(O, b.gc[fi].g, b.GramE, h));
                f.have_p9 = true;
            }
        }
    }
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

} // namespace mblock
} // namespace lvalue
} // namespace hecke_ext
