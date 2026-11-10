// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 20, WP-MEAS) the Mazur-Tate-Teitelbaum measure of an ordinary eigensymbol -- see ModularSymbolMeasure.h.
// A port of R20-1's r201/measlib.py (the oracle of record); conventions N1-N4 of r201/PREDICTIONS.md.
#include "ModularSymbolMeasure.h"

#include "Bernoulli.h"
#include "Gamma0NShapiro.h"   // (slot-22 P6) the level-N measure

#include "NumberField/CwLayer.h"

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../Util/ParallelFor.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace hecke_ext {
namespace lvalue {

namespace {
using u128 = unsigned __int128;

// x mod q for x < 2^64, q < 2^63 (Barrett with m = floor((2^64 - 1) / q); at most two corrections).
struct Barrett {
    uint64_t q = 1, m = 0;
    explicit Barrett(uint64_t qq) : q(qq), m(~uint64_t(0) / qq) {}
    uint64_t red(uint64_t x) const {
        const uint64_t qt = static_cast<uint64_t>((static_cast<u128>(x) * m) >> 64);
        uint64_t r = x - qt * q;
        while (r >= q) r -= q;
        return r;
    }
};

using engine::scalar::pow_wrap;
uint32_t ValU(uint64_t x, uint64_t ell, uint32_t cap) {
    if (x == 0) return cap;
    uint32_t v = 0;
    while (x % ell == 0 && v < cap) {
        x /= ell;
        ++v;
    }
    return v;
}
// the largest Q with ell^Q < 2^62
uint32_t MaxPrecision(uint64_t ell) {
    uint32_t Q = 0;
    for (u128 x = ell; x < (u128(1) << 62); x *= ell) ++Q;
    return Q;
}

} // namespace

namespace birch {

unsigned MeasureThreads() { return engine::threads::configured(engine::threads::kCap16); }

uint64_t MulModFast(uint64_t a, uint64_t b, uint64_t q) {
    if (q < (uint64_t(1) << 32)) return (a % q) * (b % q) % q;
    return static_cast<uint64_t>((static_cast<u128>(a) * b) % q);
}

uint64_t UnitRoot(const Ring& R, uint64_t a_ell, uint32_t k) {
    ENGINE_CHECK_MSG(R.unit(a_ell), "UnitRoot: a_ell must be an ell-unit (ordinary)");
    const uint64_t lk = R.pow(R.ell, k - 1);
    uint64_t x = a_ell % R.q;
    for (uint32_t it = 0; it < R.m + 2; ++it) x = R.sub(a_ell % R.q, R.mul(lk, R.inv(x)));
    ENGINE_CHECK_MSG(R.add(R.sub(R.mul(x, x), R.mul(a_ell % R.q, x)), lk) == 0, "UnitRoot: the iteration did not reach a root");
    ENGINE_CHECK_MSG(R.unit(x), "UnitRoot: the root is not a unit");
    return x;
}

ModularSymbolMeasure BuildMeasure(const SymRho& rho, const Vec& P, uint64_t a_ell, uint64_t alpha, uint32_t M, bool ell_n_factor,
                                  unsigned threads) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    ENGINE_CHECK_MSG(M >= 1, "BuildMeasure: level M >= 1");
    ENGINE_CHECK_MSG(P.size() == n + 1, "BuildMeasure: P has length n + 1");
    ENGINE_CHECK_MSG(R.unit(alpha), "BuildMeasure: alpha must be a unit");
    if (threads == 0) threads = MeasureThreads();
    ModularSymbolMeasure out;
    out.ell = R.ell;
    out.R = R.m;
    out.M = M;
    out.n = n;
    out.q = R.q;
    out.a_ell = a_ell % R.q;
    out.alpha = alpha % R.q;
    out.ell_n_factor = ell_n_factor;
    bool ev = false, od = false;
    for (uint32_t t = 0; t <= n; ++t)
        if (P[t] % R.q) (t % 2 == 0 ? ev : od) = true;
    out.geometric_sign = (ev && !od) ? 1 : (od && !ev) ? -1 : 0;

    Vec Pr(n + 1);
    for (uint32_t t = 0; t <= n; ++t) Pr[t] = P[t] % R.q;
    const uint64_t q = R.q;
    out.c.assign(M + 2, {});
    out.c[0] = {Pr[n]};
    out.c[1] = {Pr[n]};
    const bool fast = q < (uint64_t(1) << 31);
    const Barrett br(q);
    // P(c, d) = sum_t P_t c^{n-t} d^t by the homogeneous Horner of measlib.Hor
    auto horner = [&](uint64_t c, uint64_t d) -> uint64_t {
        uint64_t acc = 0, cp = 1 % q;
        if (fast) {
            for (uint32_t t = n + 1; t-- > 0;) {
                acc = br.red(acc * d + Pr[t] * cp);
                cp = br.red(cp * c);
            }
        } else {
            for (uint32_t t = n + 1; t-- > 0;) {
                acc = R.add(R.mul(acc, d), R.mul(Pr[t], cp));
                cp = R.mul(cp, c);
            }
        }
        return acc;
    };
    for (uint32_t m = 1; m <= M; ++m) {
        const uint64_t Nm = pow_wrap(R.ell, m);
        ENGINE_CHECK_MSG(Nm < (uint64_t(1) << 62), "BuildMeasure: ell^M too large");
        std::vector<uint64_t>& cm = out.c[m + 1];
        cm.assign(Nm, 0);
        oc_util::ParallelChunks(Nm, threads, oc_util::kMeasureChunks, [&](uint64_t s, uint64_t e) {
            for (uint64_t a = s; a < e; ++a) {
                uint64_t sum = 0;
                for (const Mat2& g : ManinPath(static_cast<int64_t>(a), static_cast<int64_t>(Nm)))
                    sum = R.add(sum, horner(R.from_int(g.c), R.from_int(g.d)));
                cm[a] = R.neg(sum);
            }
        });
    }
    const uint64_t ia = R.inv(alpha % q);
    const uint64_t ln = ell_n_factor ? R.pow(R.ell, n) : 1 % q;
    out.mu.assign(M + 1, {});
    uint64_t am = 1 % q;   // alpha^{-m}
    for (uint32_t m = 0; m <= M; ++m) {
        const uint64_t am1 = R.mul(am, ia);
        const std::vector<uint64_t>& cur = out.c[m + 1];
        const std::vector<uint64_t>& prev = out.c[m];
        const uint64_t Mp = prev.size();
        std::vector<uint64_t>& mu = out.mu[m];
        mu.assign(cur.size(), 0);
        for (uint64_t a = 0; a < cur.size(); ++a) mu[a] = R.sub(R.mul(am, cur[a]), R.mul(R.mul(am1, ln), prev[a % Mp]));
        am = am1;
    }
    return out;
}

uint32_t ThreeTermFailures(const ModularSymbolMeasure& me, uint32_t m) {
    ENGINE_CHECK_MSG(m + 1 <= me.M, "ThreeTermFailures: m + 1 <= M");
    const Ring R = MakeRing(me.ell, me.R);
    const uint64_t Mm = pow_wrap(me.ell, m), ln = R.pow(me.ell, me.n);
    const std::vector<uint64_t>&next = me.c[m + 2], &cur = me.c[m + 1], &prev = me.c[m];
    uint32_t bad = 0;
    for (uint64_t a = 0; a < Mm; ++a) {
        uint64_t s = 0;
        for (uint64_t b = 0; b < me.ell; ++b) s = R.add(s, next[a + b * Mm]);
        const uint64_t rhs = R.sub(R.mul(me.a_ell, cur[a]), R.mul(ln, prev[a % prev.size()]));
        if (s != rhs) ++bad;
    }
    return bad;
}

uint32_t DistributionFailures(const ModularSymbolMeasure& me, uint32_t m) {
    ENGINE_CHECK_MSG(m + 1 <= me.M, "DistributionFailures: m + 1 <= M");
    const Ring R = MakeRing(me.ell, me.R);
    const uint64_t Mm = pow_wrap(me.ell, m);
    uint32_t bad = 0;
    for (uint64_t a = 0; a < Mm; ++a) {
        uint64_t s = 0;
        for (uint64_t b = 0; b < me.ell; ++b) s = R.add(s, me.mu[m + 1][a + b * Mm]);
        if (s != me.mu[m][a]) ++bad;
    }
    return bad;
}

DirichletCharacter TeichmullerCharacter(uint32_t i, const Ring& R) {
    DirichletCharacter c;
    const uint64_t ell = R.ell;
    c.M = ell;
    c.chi.assign(ell, 0);
    c.chibar.assign(ell, 0);
    const uint32_t e = static_cast<uint32_t>(i % (ell - 1)), eb = static_cast<uint32_t>((ell - 1 - e) % (ell - 1));
    for (uint64_t a = 1; a < ell; ++a) {
        c.chi[a] = TeichmullerPowMod(a, e, ell, R.m) % R.q;
        c.chibar[a] = TeichmullerPowMod(a, eb, ell, R.m) % R.q;
    }
    c.parity = (e % 2 == 0) ? 1 : -1;
    c.real = (2 * e) % (ell - 1) == 0;
    c.name = "omega^" + std::to_string(e) + " mod " + std::to_string(ell);
    return c;
}

// ---- the symbol-side cell ------------------------------------------------------------------------------
namespace {
Vec LinComb(const Ring& R, const std::vector<Vec>& basis, const Vec& y, uint32_t len) {
    Vec out(len, 0);
    for (size_t i = 0; i < basis.size(); ++i)
        if (y[i])
            for (uint32_t t = 0; t < len; ++t) out[t] = R.add(out[t], R.mul(y[i], basis[i][t]));
    return out;
}
uint32_t VVal(const Ring& R, const Vec& v) {
    uint32_t m = R.m;
    for (uint64_t x : v) m = std::min(m, R.val(x));
    return m;
}
Vec Primitive(const Ring& R, const Vec& v) {
    const uint32_t e = VVal(R, v);
    if (e == 0 || e >= R.m) return v;
    const uint64_t pe = R.ell_pow(e);
    Vec o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = (v[i] / pe) % R.q;
    return o;
}
std::vector<Vec> WSignBasis(const SymRho& rho, uint32_t parity) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    std::vector<uint32_t> cols;
    for (uint32_t t = 0; t <= n; ++t)
        if (t % 2 == parity) cols.push_back(t);
    Mat A(2 * (n + 1), static_cast<uint32_t>(cols.size()));
    const Mat2 U{0, -1, 1, 1};
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
    KernelGenerators(R, A, gens, exps);
    std::vector<Vec> basis;
    for (size_t g = 0; g < gens.size(); ++g) {
        ENGINE_CHECK_MSG(exps[g] == R.m, "RunSymbolCell: W has a finite pivot (W(Z/ell^R) is not free)");
        Vec v(n + 1, 0);
        for (uint32_t j = 0; j < cols.size(); ++j) v[cols[j]] = gens[g][j];
        basis.push_back(v);
    }
    return basis;
}
Mat HeckeOnBasis(const SymRho& rho, const PivotBasis& B, uint64_t p) {
    const Ring& R = rho.ring();
    const uint32_t r = static_cast<uint32_t>(B.basis.size()), len = rho.n() + 1;
    Mat T(r, r);
    for (uint32_t j = 0; j < r; ++j) {
        const Vec img = HeckeTp(rho, B.basis[j], p);
        const Vec c = B.coords(img);
        ENGINE_CHECK_MSG(LinComb(R, B.basis, c, len) == img, "RunSymbolCell: T_p image not in the span of W");
        for (uint32_t i = 0; i < r; ++i) T.at(i, j) = c[i];
    }
    return T;
}
uint64_t PolyEvalMod(const Vec& f, uint64_t x, uint64_t mod) {
    u128 s = 0;
    for (size_t i = f.size(); i-- > 0;) s = (s * x + f[i] % mod) % mod;
    return static_cast<uint64_t>(s);
}
Vec Deriv(const Ring& R, const Vec& f) {
    Vec d;
    for (size_t i = 1; i < f.size(); ++i) d.push_back(R.mul(f[i], i % R.q));
    return d;
}
} // namespace

SymbolCell RunSymbolCell(uint32_t n, uint64_t ell, uint32_t Rprec, const std::map<uint64_t, uint64_t>& sep, bool control) {
    const auto t0 = std::chrono::steady_clock::now();
    SymbolCell C;
    C.n = n;
    C.ell = ell;
    C.R = Rprec;
    C.ring = MakeRing(ell, Rprec);
    const Ring& R = C.ring;
    const SymRho rho(n, R);
    const uint32_t k = n + 2;
    C.Wminus = WSignBasis(rho, 1);
    C.Wplus = WSignBasis(rho, 0);
    const PivotBasis Bm = SaturatedBasis(R, C.Wminus), Bp = SaturatedBasis(R, C.Wplus);
    ENGINE_CHECK_MSG(Bm.free_summand && Bp.free_summand && !Bm.basis.empty(), "RunSymbolCell: W^{+-} not free summands");
    const uint32_t dm = static_cast<uint32_t>(Bm.basis.size());
    auto sepMat = [&](const PivotBasis& B) {
        const uint32_t r = static_cast<uint32_t>(B.basis.size());
        Mat t(r, r);
        for (const auto& pc : sep) {
            const Mat T = HeckeOnBasis(rho, B, pc.first);
            for (uint32_t i = 0; i < r * r; ++i) t.a[i] = R.add(t.a[i], R.mul(pc.second % R.q, T.a[i]));
        }
        return t;
    };
    const Mat tm = sepMat(Bm);
    const Vec cp = CharPoly(R, tm);
    Vec cpl(cp.size());
    for (size_t i = 0; i < cp.size(); ++i) cpl[i] = cp[i] % ell;
    for (uint64_t x = 0; x < ell; ++x)
        if (PolyEvalMod(cpl, x, ell) == 0) C.charpoly_roots.push_back(x);
    uint64_t target = 0;
    for (const auto& pc : sep) target = (target + (pc.second % ell) * ((1 + PowModU64(pc.first % ell, k - 1, ell)) % ell)) % ell;
    // multiplicity of the target root mod ell (synthetic division)
    uint32_t mult = 0;
    {
        Vec g = cpl;
        while (g.size() > 1 && PolyEvalMod(g, target, ell) == 0) {
            ++mult;
            Vec qv(g.size() - 1, 0);
            uint64_t acc = 0;
            for (size_t i = g.size() - 1; i >= 1; --i) {
                acc = (acc * target + g[i]) % ell;
                qv[i - 1] = acc;
            }
            g.swap(qv);
        }
    }
    C.eisenstein_congruent = mult >= 1;
    if (mult == 0 && control) {
        if (C.charpoly_roots.empty()) {
            C.refusal = "control: charpoly(sep | W^-) has no root mod ell";
            return C;
        }
        target = C.charpoly_roots[0];
        const Vec dl = Deriv(MakeRing(ell, 1), cpl);
        mult = (!dl.empty() && PolyEvalMod(dl, target, ell) != 0) ? 1 : 2;
    }
    C.target = target;
    C.target_multiplicity = mult;
    if (mult != 1) {
        C.simple = false;
        C.refusal = "REFUSED: target root " + std::to_string(target) + " has multiplicity " + std::to_string(mult) + " mod ell";
        C.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return C;
    }
    C.simple = true;
    // Hensel lift
    const Vec dcp = Deriv(R, cp);
    uint64_t x = target % R.q;
    for (uint32_t it = 0; it < 2 * Rprec + 2; ++it) {
        const uint64_t dv = PolyEvalMod(dcp, x, R.q);
        ENGINE_CHECK_MSG(R.unit(dv), "RunSymbolCell: not a simple root mod ell");
        x = R.sub(x, R.mul(PolyEvalMod(cp, x, R.q), R.inv(dv)));
    }
    ENGINE_CHECK_MSG(PolyEvalMod(cp, x, R.q) == 0, "RunSymbolCell: Hensel lift failed");
    C.lam = x;
    auto eigvec = [&](const PivotBasis& B, const Mat& t, std::vector<uint32_t>* piv) -> std::vector<Vec> {
        Mat A = t;
        for (uint32_t i = 0; i < A.r; ++i) A.at(i, i) = R.sub(A.at(i, i), C.lam);
        if (piv) *piv = Smith(R, A, false).v;
        std::vector<Vec> gens, full;
        std::vector<uint32_t> exps;
        KernelGenerators(R, A, gens, exps);
        for (size_t g = 0; g < gens.size(); ++g)
            if (exps[g] == R.m) full.push_back(LinComb(R, B.basis, gens[g], n + 1));
        return full;
    };
    const std::vector<Vec> vm = eigvec(Bm, tm, nullptr);
    ENGINE_CHECK_MSG(vm.size() == 1, "RunSymbolCell: the W^- eigen-kernel is not of rank 1");
    Vec rm = Primitive(R, vm[0]);
    uint32_t tu = 0;
    while (tu <= n && !R.unit(rm[tu])) ++tu;
    ENGINE_CHECK_MSG(tu <= n, "RunSymbolCell: r^- has no unit coordinate");
    {
        const uint64_t iv = R.inv(rm[tu]);
        for (auto& v : rm) v = R.mul(v, iv);
    }
    C.r_minus = rm;
    std::vector<uint64_t> primes = {2, 3, 5, 7, ell};
    for (const auto& pc : sep) primes.push_back(pc.first);
    std::sort(primes.begin(), primes.end());
    primes.erase(std::unique(primes.begin(), primes.end()), primes.end());
    for (uint64_t p : primes) {
        const Vec img = HeckeTp(rho, rm, p);
        const uint64_t a = R.mul(img[tu], R.inv(rm[tu]));
        for (uint32_t t = 0; t <= n; ++t)
            ENGINE_CHECK_MSG(img[t] == R.mul(a, rm[t]), "RunSymbolCell: r^- is not a T_p eigenvector");
        C.a_p[p] = a;
    }
    (void)dm;
    // r^+
    const std::vector<Vec> vp = eigvec(Bp, sepMat(Bp), &C.wplus_pivots);
    if (vp.size() == 1) {
        Vec rp = Primitive(R, vp[0]);
        if (R.unit(rp[0])) {
            const uint64_t iv = R.inv(rp[0]);
            for (auto& v : rp) v = R.mul(v, iv);
            C.r_plus = rp;
            C.have_r_plus = true;
        }
    }
    C.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return C;
}

// ---- Bernoulli numbers -----------------------------------------------------------------------------------
BernoulliOverKTable::BernoulliOverKTable(uint64_t ell, uint32_t R, uint32_t kmax) : ell_(ell), R_(R) {
    ENGINE_CHECK_MSG(kmax >= 2, "BernoulliOverKTable: kmax >= 2");
    const uint32_t Qw = std::min<uint32_t>(MaxPrecision(ell), R + 6);
    ENGINE_CHECK_MSG(Qw >= R + 3, "BernoulliOverKTable: working precision too small");
    const uint64_t qw = pow_wrap(ell, Qw), qR = pow_wrap(ell, R);
    const std::vector<nf::BigInt> T = TangentNumbers(kmax / 2 + 1);
    tab_.assign(kmax + 1, LadicReading{});
    for (uint32_t r = 2; r <= kmax; r += 2) {
        uint64_t un = 0;
        const uint32_t vn = ValuationAndUnitPart(T[r / 2 - 1], ell, qw, un);   // T_{r-1}
        const uint64_t p2 = PowModU64(2, r, qw);
        const uint64_t d1 = (p2 + qw - 1) % qw;                                 // 2^r - 1 mod ell^Qw
        ENGINE_CHECK_MSG(d1 != 0, "BernoulliOverKTable: v(2^r - 1) beyond the working precision");
        const uint32_t vd = ValU(d1, ell, Qw);
        ENGINE_CHECK_MSG(Qw - vd >= R, "BernoulliOverKTable: unit part of 2^r - 1 below ell^R");
        const uint64_t ud = (d1 / pow_wrap(ell, vd)) % qR;
        uint64_t u = static_cast<uint64_t>(static_cast<u128>(un % qR) * InvModU64(ud, qR) % qR * InvModU64(p2 % qR, qR) % qR);
        if (((r / 2 - 1) % 2) == 1) u = (qR - u) % qR;   // sign (-1)^{r/2 - 1}
        LadicReading L;
        L.v = static_cast<int32_t>(vn) - static_cast<int32_t>(vd);
        L.unit = u;
        tab_[r] = L;
    }
}
LadicReading BernoulliOverKTable::operator()(uint32_t r) const {
    ENGINE_CHECK_MSG(r >= 2 && r % 2 == 0 && r < tab_.size(), "BernoulliOverKTable: r even in [2, kmax]");
    return tab_[r];
}
uint64_t BernoulliOverKTable::residue(uint32_t r) const {
    const LadicReading L = (*this)(r);
    ENGINE_CHECK_MSG(L.v >= 0, "BernoulliOverKTable::residue: B_r/r is not ell-integral");
    if (static_cast<uint32_t>(L.v) >= R_) return 0;
    const uint64_t qR = pow_wrap(ell_, R_);
    return static_cast<uint64_t>(static_cast<u128>(L.unit) * pow_wrap(ell_, static_cast<uint32_t>(L.v)) % qR);
}

TwistedBernoulliTable::TwistedBernoulliTable(uint64_t ell, uint32_t R, uint32_t rmax) : ell_(ell), R_(R), rmax_(rmax) {
    Q_ = MaxPrecision(ell);
    ENGINE_CHECK_MSG(Q_ >= R + 4, "TwistedBernoulliTable: working precision too small");
    NN_ = pow_wrap(ell, Q_);
    Ls_.assign(Q_ + 2, 0);
    Ls_[0] = 1;
    Ls_[1] = static_cast<uint64_t>(static_cast<u128>(NN_ - ell % NN_) * InvModU64(2, NN_) % NN_);   // ell B_1 = -ell/2
    for (uint32_t s = 2; s <= Q_ + 1; s += 2) {
        // ell^s B_s = ell^s s (B_s / s), B_s/s = sign num / den exactly
        const BernoulliOverK b = ExactBernoulliOverK(s);
        uint64_t un = 0, ud = 0, us = 0;
        const uint32_t vn = ValuationAndUnitPart(b.num, ell, NN_, un);
        const uint32_t vd = ValuationAndUnitPart(b.den, ell, NN_, ud);
        const uint32_t vs = ValuationAndUnitPart(nf::BigInt(s), ell, NN_, us);
        const int32_t v = static_cast<int32_t>(s + vn + vs) - static_cast<int32_t>(vd);
        ENGINE_CHECK_MSG(v >= static_cast<int32_t>(s) - 1, "TwistedBernoulliTable: v(B_s) < -1");
        if (v >= static_cast<int32_t>(Q_)) continue;
        u128 x = static_cast<u128>(un) * us % NN_ * InvModU64(ud, NN_) % NN_;
        if (b.sign < 0) x = (NN_ - x) % NN_;
        Ls_[s] = static_cast<uint64_t>(x * pow_wrap(ell, static_cast<uint32_t>(v)) % NN_);
    }
}
LadicReading TwistedBernoulliTable::operator()(uint32_t r, uint32_t i) const {
    const uint32_t ii = static_cast<uint32_t>(i % (ell_ - 1));
    ENGINE_CHECK_MSG(ii != 0, "TwistedBernoulliTable: chi = omega^i must be non-trivial");
    ENGINE_CHECK_MSG(r >= 1 && r <= rmax_, "TwistedBernoulliTable: 1 <= r <= rmax");
    const auto key = std::make_pair(r, ii);
    const auto hit = cache_.find(key);
    if (hit != cache_.end()) return hit->second;
    auto& S = S_[ii];
    if (S.empty()) {
        std::vector<uint64_t> chi(ell_, 0);
        for (uint64_t a = 1; a < ell_; ++a) chi[a] = TeichmullerPowMod(a, ii, ell_, Q_) % NN_;
        S.assign(rmax_ + 1, 0);
        std::vector<uint64_t> pw(ell_, 1);
        for (uint32_t qq = 0; qq <= rmax_; ++qq) {
            u128 s = 0;
            for (uint64_t a = 1; a < ell_; ++a) {
                s = (s + static_cast<u128>(chi[a]) * pw[a]) % NN_;
                pw[a] = static_cast<uint64_t>(static_cast<u128>(pw[a]) * a % NN_);
            }
            S[qq] = static_cast<uint64_t>(s);
        }
    }
    u128 tot = 0, C = 1;   // C(r, s), exact (s <= Q + 1 <= 13, r <= a few hundred: < 2^100)
    for (uint32_t s = 0; s <= std::min<uint32_t>(r, Q_ + 1); ++s) {
        if (s > 0) C = C * (r - s + 1) / s;
        if (Ls_[s]) tot = (tot + static_cast<u128>(static_cast<uint64_t>(C % NN_)) * Ls_[s] % NN_ * S[r - s]) % NN_;
    }
    LadicReading L;
    if (tot == 0) {
        L.zero = true;
    } else {
        const uint64_t t = static_cast<uint64_t>(tot);
        const uint32_t v = ValU(t, ell_, Q_);
        ENGINE_CHECK_MSG(Q_ - v >= R_, "TwistedBernoulliTable: unit part below ell^R");
        const uint32_t vr = ValU(r, ell_, 64);
        const uint64_t qR = pow_wrap(ell_, R_);
        const uint64_t ur = (r / pow_wrap(ell_, vr)) % qR;
        L.v = static_cast<int32_t>(v) - 1 - static_cast<int32_t>(vr);
        L.unit = static_cast<uint64_t>(static_cast<u128>((t / pow_wrap(ell_, v)) % qR) * InvModU64(ur, qR) % qR);
    }
    cache_[key] = L;
    return L;
}

Vec EisensteinSymbol(uint32_t n, const Ring& R, int32_t* min_valuation) {
    const BernoulliOverKTable B(R.ell, R.m, n + 2);
    Vec v(n + 1, 0);
    int32_t mn = 1 << 20;
    nf::BigInt C(1);   // C(n, j)
    for (uint32_t j = 1; j < n; ++j) {
        C = (C * nf::BigInt(n - j + 1)).div_exact_u32(j);
        if (j % 2 == 0) continue;
        uint64_t uc = 0;
        const uint32_t vc = ValuationAndUnitPart(C, R.ell, R.q, uc);
        const LadicReading b1 = B(j + 1), b2 = B(n - j + 1);
        const int32_t vt = static_cast<int32_t>(vc) + b1.v + b2.v;
        mn = std::min(mn, vt);
        ENGINE_CHECK_MSG(vt >= 0, "EisensteinSymbol: P^Eis is not ell-integral");
        if (static_cast<uint32_t>(vt) >= R.m) continue;
        v[n - j] = R.mul(R.mul(R.mul(uc, b1.unit), b2.unit), R.ell_pow(static_cast<uint32_t>(vt)));
    }
    if (min_valuation) *min_valuation = mn;
    return v;
}

uint32_t WResidualValuation(const SymRho& rho, const Vec& P) {
    const Ring& R = rho.ring();
    const Vec s = rho.S(P);
    const Mat2 U{0, -1, 1, 1};
    const Vec u1 = rho.apply(U, P), u2 = rho.apply(U, u1);
    uint32_t mn = R.m;
    for (size_t t = 0; t < P.size(); ++t) {
        mn = std::min(mn, R.val(R.add(P[t], s[t])));
        mn = std::min(mn, R.val(R.add(R.add(P[t], u1[t]), u2[t])));
    }
    return mn;
}

// ---- Kubota-Leopoldt group rings -------------------------------------------------------------------------
std::vector<uint64_t> GrMul(const std::vector<uint64_t>& x, const std::vector<uint64_t>& y, uint64_t q) {
    const size_t G = x.size();
    ENGINE_CHECK_MSG(y.size() == G, "GrMul: sizes differ");
    std::vector<uint64_t> out(G, 0);
    const bool fast = q < (uint64_t(1) << 31);
    const Barrett br(q);
    for (size_t i = 0; i < G; ++i) {
        const uint64_t a = x[i] % q;
        if (!a) continue;
        for (size_t j = 0; j < G; ++j) {
            const uint64_t b = y[j];
            if (!b) continue;
            size_t t = i + j;
            if (t >= G) t -= G;
            if (fast) {
                out[t] = br.red(out[t] + br.red(a * (b % q)));
            } else {
                out[t] = static_cast<uint64_t>((static_cast<u128>(a) * (b % q) + out[t]) % q);
            }
        }
    }
    return out;
}
std::vector<uint64_t> GrTw1(const std::vector<uint64_t>& g, uint64_t ell, uint64_t q) {
    std::vector<uint64_t> o(g.size());
    uint64_t ui = 1 % q;
    for (size_t i = 0; i < g.size(); ++i) {
        o[i] = MulModFast(g[i], ui, q);
        ui = MulModFast(ui, 1 + ell, q);
    }
    return o;
}
std::vector<uint64_t> GrTwInv(const std::vector<uint64_t>& g, uint64_t ell, uint64_t q, uint32_t k) {
    const size_t G = g.size();
    std::vector<uint64_t> o(G, 0);
    const uint64_t uk = PowModU64((1 + ell) % q, k - 1, q);
    uint64_t ui = 1 % q;
    for (size_t i = 0; i < G; ++i) {
        o[(G - i) % G] = MulModFast(g[i], ui, q);
        ui = MulModFast(ui, uk, q);
    }
    return o;
}
uint64_t GrEval(const std::vector<uint64_t>& g, uint64_t s, uint64_t ell, uint64_t q) {
    const uint64_t us = PowModU64((1 + ell) % q, s, q);
    uint64_t acc = 0, ui = 1 % q;
    for (size_t i = 0; i < g.size(); ++i) {
        acc = (acc + MulModFast(g[i], ui, q)) % q;
        ui = MulModFast(ui, us, q);
    }
    return acc;
}

std::vector<uint64_t> KlGroupRingBernoulli(uint32_t b, uint64_t ell, uint32_t M, uint32_t R) {
    ENGINE_CHECK_MSG(b % (ell - 1) != 0, "KlGroupRingBernoulli: b != 0 mod ell - 1");
    ENGINE_CHECK_MSG(M >= 1, "KlGroupRingBernoulli: M >= 1");
    const uint64_t N = pow_wrap(ell, R), MM = pow_wrap(ell, M), G = pow_wrap(ell, M - 1), u = 1 + ell;
    uint64_t c = PrimitiveRootModPrime(ell);
    if (PowModU64(c, ell - 1, ell * ell) == 1) c += ell;
    const uint64_t ic = InvModU64(c % MM, MM), inv2 = InvModU64(2, N);
    std::vector<uint64_t> deltas(ell), wts(ell);
    for (uint64_t a = 1; a < ell; ++a) {
        deltas[a] = TeichmullerPowMod(a, 1, ell, M) % MM;
        wts[a] = TeichmullerPowMod(a, static_cast<uint32_t>(b % (ell - 1)), ell, R) % N;
    }
    std::vector<uint64_t> num(G, 0);
    uint64_t ui = 1 % MM;
    for (uint64_t i = 0; i < G; ++i) {
        u128 s = 0;
        for (uint64_t a = 1; a < ell; ++a) {
            const uint64_t x = static_cast<uint64_t>(static_cast<u128>(deltas[a]) * ui % MM);
            const uint64_t a2 = static_cast<uint64_t>(static_cast<u128>(ic) * x % MM);
            const int64_t diff = static_cast<int64_t>(x) - static_cast<int64_t>(c * a2);
            ENGINE_CHECK(diff % static_cast<int64_t>(MM) == 0);
            const int64_t fl = diff / static_cast<int64_t>(MM);
            const uint64_t flm = static_cast<uint64_t>(((fl % static_cast<int64_t>(N)) + static_cast<int64_t>(N)) % static_cast<int64_t>(N));
            const uint64_t e1c = (flm + MulModFast((c - 1) % N, inv2, N)) % N;
            s = (s + static_cast<u128>(MulModFast(wts[a], InvModU64(x % N, N), N)) * e1c) % N;
        }
        num[i] = static_cast<uint64_t>(s);
        ui = static_cast<uint64_t>(static_cast<u128>(ui) * u % MM);
    }
    // <c> = u^{i_c}
    const uint64_t wc = TeichmullerPowMod(c, 1, ell, M) % MM;
    const uint64_t cc = static_cast<uint64_t>(static_cast<u128>(c % MM) * InvModU64(wc, MM) % MM);
    uint64_t i_c = G;
    ui = 1 % MM;
    for (uint64_t i = 0; i < G; ++i) {
        if (ui == cc) {
            i_c = i;
            break;
        }
        ui = static_cast<uint64_t>(static_cast<u128>(ui) * u % MM);
    }
    ENGINE_CHECK_MSG(i_c < G, "KlGroupRingBernoulli: <c> not found in 1 + ell Z_ell");
    const uint64_t z = TeichmullerPowMod(c, static_cast<uint32_t>(b % (ell - 1)), ell, R) % N;
    std::vector<uint64_t> inv(G, 0);
    const uint64_t f = InvModU64((1 + N - PowModU64(z, G, N)) % N, N);
    uint64_t zt = 1 % N;
    for (uint64_t t = 0; t < G; ++t) {
        const uint64_t idx = (t * i_c) % G;
        inv[idx] = (inv[idx] + MulModFast(f, zt, N)) % N;
        zt = MulModFast(zt, z, N);
    }
    std::vector<uint64_t> out = GrMul(num, inv, N);
    for (auto& x : out) x = (N - x) % N;   // sign: r201/kl_check.py (calibration); re-gated by (M4a)
    return out;
}

std::map<uint32_t, std::vector<uint64_t>> KlGroupRingLayer(const std::vector<uint32_t>& classes, uint64_t ell, uint32_t M,
                                                           unsigned threads) {
    ENGINE_CHECK_MSG(M >= 1 && ell >= 5 && ell < (uint64_t(1) << 31), "KlGroupRingLayer: ell >= 5, M >= 1");
    if (threads == 0) threads = MeasureThreads();
    const unsigned r = M + 1, W = r + 3 * M;
    nf::CwLog lg;
    {
        nf::CwRing Rg(static_cast<uint32_t>(ell), M, W, threads);
        nf::CwLogOptions opt;
        opt.r = r;
        lg = nf::cw_lambda(Rg, opt);
    }
    const nf::LadicCtx cx(static_cast<uint32_t>(ell), W);
    const std::vector<nf::CwMeasure> ms = nf::cw_measure(cx, M, lg.Lg.data(), lg.s + M, classes, M, threads);
    const uint64_t q = pow_wrap(ell, M);
    std::map<uint32_t, std::vector<uint64_t>> out;
    for (const nf::CwMeasure& m : ms) {
        ENGINE_CHECK_MSG(m.integral, "KlGroupRingLayer: the layer measure is not integral");
        std::vector<uint64_t> g(m.nu.size());
        for (size_t i = 0; i < g.size(); ++i) g[i] = (q - MulModFast(m.nu[i] % q, (ell - 1) % q, q)) % q;
        out[m.w] = g;
    }
    return out;
}

} // namespace birch

// ---- ModularSymbolMeasure ---------------------------------------------------------------------------------
uint64_t ModularSymbolMeasure::ball(uint64_t a, uint32_t m) const {
    ENGINE_CHECK_MSG(m <= M, "ModularSymbolMeasure::ball: m <= M");
    return mu[m][a % mu[m].size()];
}

std::vector<uint64_t> ModularSymbolMeasure::branch(uint32_t a) const {
    const uint64_t MM = pow_wrap(ell, M), G = pow_wrap(ell, M - 1), u = 1 + ell;
    std::vector<uint64_t> deltas(ell), wts(ell);
    for (uint64_t d = 1; d < ell; ++d) {
        deltas[d] = TeichmullerPowMod(d, 1, ell, M) % MM;
        wts[d] = TeichmullerPowMod(d, static_cast<uint32_t>(a % (ell - 1)), ell, R) % q;
    }
    std::vector<uint64_t> g(G, 0);
    uint64_t ui = 1 % MM;
    for (uint64_t i = 0; i < G; ++i) {
        u128 s = 0;
        for (uint64_t d = 1; d < ell; ++d) s = (s + static_cast<u128>(wts[d]) * mu[M][static_cast<uint64_t>(static_cast<u128>(deltas[d]) * ui % MM)]) % q;
        g[i] = static_cast<uint64_t>(s);
        ui = static_cast<uint64_t>(static_cast<u128>(ui) * u % MM);
    }
    return g;
}

uint64_t ModularSymbolMeasure::branch_value(uint32_t a, uint32_t j) const {
    return birch::GrEval(branch(a), j, ell, q);
}

std::vector<std::vector<uint64_t>> ModularSymbolMeasure::moments(const std::vector<uint32_t>& chars) const {
    const uint64_t MM = pow_wrap(ell, M);
    const size_t nc = chars.size();
    std::vector<std::vector<uint64_t>> wt(nc, std::vector<uint64_t>(ell, 0));
    for (size_t ii = 0; ii < nc; ++ii) {
        const uint32_t e = static_cast<uint32_t>((ell - 1 - chars[ii] % (ell - 1)) % (ell - 1));
        for (uint64_t r = 1; r < ell; ++r) wt[ii][r] = TeichmullerPowMod(r, e, ell, R) % q;
    }
    const unsigned T = birch::MeasureThreads();
    const unsigned chunks = static_cast<unsigned>(std::max<uint64_t>(1, std::min<uint64_t>(T, MM / 512 + 1)));
    std::vector<std::vector<std::vector<uint64_t>>> part(chunks, std::vector<std::vector<uint64_t>>(nc, std::vector<uint64_t>(n + 1, 0)));
    const bool fast = q < (uint64_t(1) << 31);
    const Barrett br(q);
    auto mm = [&](uint64_t x, uint64_t y) -> uint64_t { return fast ? br.red(x * y) : static_cast<uint64_t>(static_cast<u128>(x) * y % q); };
    auto ad = [&](uint64_t x, uint64_t y) -> uint64_t { const uint64_t s = x + y; return s >= q ? s - q : s; };
    std::vector<std::thread> ts;
    const uint64_t per = (MM + chunks - 1) / chunks;
    for (unsigned c = 0; c < chunks; ++c) {
        ts.emplace_back([&, c]() {
            std::vector<uint64_t> y(nc);
            auto& acc = part[c];
            const uint64_t s = c * per, e = std::min<uint64_t>(MM, s + per);
            for (uint64_t a = s; a < e; ++a) {
                const uint64_t r = a % ell;
                if (r == 0 || mu[M][a] == 0) continue;
                for (size_t ii = 0; ii < nc; ++ii) y[ii] = mm(wt[ii][r], mu[M][a]);
                const uint64_t aq = a % q;
                uint64_t pa = 1 % q;
                for (uint32_t j = 0; j <= n; ++j) {
                    for (size_t ii = 0; ii < nc; ++ii) acc[ii][j] = ad(acc[ii][j], mm(y[ii], pa));
                    pa = mm(pa, aq);
                }
            }
        });
    }
    for (auto& t : ts) t.join();
    std::vector<std::vector<uint64_t>> out(nc, std::vector<uint64_t>(n + 1, 0));
    for (unsigned c = 0; c < chunks; ++c)
        for (size_t ii = 0; ii < nc; ++ii)
            for (uint32_t j = 0; j <= n; ++j) out[ii][j] = ad(out[ii][j], part[c][ii][j]);
    return out;
}

uint64_t ModularSymbolMeasure::moment(uint32_t j, uint32_t i) const {
    ENGINE_CHECK_MSG(j <= n, "ModularSymbolMeasure::moment: j <= n");
    return moments({i})[0][j];
}

IwasawaInvariants ModularSymbolMeasure::invariants(uint32_t a) const {
    IwasawaInvariants I;
    I.branch = a;
    const std::vector<uint64_t> g = branch(a);
    I.vanishes = std::all_of(g.begin(), g.end(), [](uint64_t x) { return x == 0; });
    const uint64_t G = g.size();
    const uint64_t tmax = std::min<uint64_t>(G, ell);
    // coef_t = sum_i g_i C(i, t), t < tmax (C(i, t) mod q by Pascal, column by column)
    std::vector<uint64_t> col(G, 1 % q), coef(tmax, 0);
    for (uint64_t t = 0; t < tmax; ++t) {
        if (t > 0) {
            // col_t(i) = C(i, t) = C(i-1, t) + C(i-1, t-1)
            std::vector<uint64_t> nc(G, 0);
            for (uint64_t i = 1; i < G; ++i) nc[i] = (nc[i - 1] + col[i - 1]) % q;
            col.swap(nc);
        }
        u128 s = 0;
        for (uint64_t i = 0; i < G; ++i)
            if (g[i] && col[i]) s = (s + static_cast<u128>(g[i]) * col[i]) % q;
        coef[t] = static_cast<uint64_t>(s);
    }
    uint32_t best = R;
    uint64_t lam = 0;
    for (uint64_t t = 0; t < tmax; ++t) {
        const uint32_t v = ValU(coef[t], ell, R);
        if (v < best) {
            best = v;
            lam = t;
        }
    }
    I.mu = best;
    I.lambda = static_cast<uint32_t>(lam);
    I.readable = !I.vanishes && best < std::min<uint32_t>(R, M - 1) && lam < G;
    return I;
}

uint64_t ModularSymbolMeasure::fnv_units(uint32_t level) const {
    ENGINE_CHECK_MSG(level <= M, "ModularSymbolMeasure::fnv_units: level <= M");
    uint64_t h = engine::kFnvOffset64;
    for (uint64_t a = 1; a < mu[level].size(); ++a)
        if (a % ell) h = engine::FnvU64(h, mu[level][a]);
    return h;
}

uint64_t ModularSymbolMeasure::unit_total(uint32_t level) const {
    ENGINE_CHECK_MSG(level <= M, "ModularSymbolMeasure::unit_total: level <= M");
    uint64_t s = 0;
    for (uint64_t a = 1; a < mu[level].size(); ++a)
        if (a % ell) s = (s + mu[level][a]) % q;
    return s;
}

NormalisationRecord ModularSymbolMeasure::normalisation() const {
    NormalisationRecord N;
    N.l_function = "L_ell(f, alpha, .) (the ell-stabilised MTT measure, [E] def:lvmtt)";
    N.s0 = "int_{Z_ell^x} chibar(x) x^j dmu_f, 0 <= j <= n=" + std::to_string(n);
    N.twist = "chi of ell-power conductor (chi = 1, omega^i, ...)";
    N.sign_at_infinity = geometric_sign > 0 ? '-' : '+';   // [C]'s convention: geometric-even = [C]-minus
    N.euler_factors_removed.push_back({"E_{1,j}", "(1 - ell^j/alpha)(1 - beta/ell^{j+1})", 0, true});
    N.euler_factors_removed.push_back({"E_{chi,j}", "alpha^{-m} (chi of conductor ell^m)", 0, true});
    N.period = "Omega_f^{+-} of H^1_c (geometric sign chi(-1)(-1)^j)";
    N.constants.push_back({"MTT", "(-1)^j j! M_chi^{j+1} / (chi(-1)(-2 pi i)^{j+1} tau(chi))", 0, false});
    N.constants.push_back({"stabilisation", std::string("ell^{k-2} on the second term") + (ell_n_factor ? "" : " OMITTED (negative control)"),
                           0, true});
    N.ambiguity = "O^x (the eigensymbol's scaling); index statements: congruences of measures";
    return N;
}

// ---- the accessor ([E] def:lvmeasure; O5) ---------------------------------------------------------------------
ModularSymbolMeasure LValueRoute::measure(const CohomologyClass&, const MeasureRequest&) const {
    if (kind() != RouteKind::Linear) refuse_kind("measure (kind linear, measure-valued; [E] def:lvmeasure)");
    throw hecke::NotImplementedInThisBuild("LValueRoute " + name() + ": measure() is not implemented by this linear route");
}

ModularSymbolMeasure BirchModularSymbolRoute::measure(const CohomologyClass& x, const MeasureRequest& req) const {
    if (req.level < 1) throw std::invalid_argument("BirchModularSymbolRoute::measure: level M >= 1");
    if (domain().model == ModelId::Gamma0NShapiro) {
        // (slot-22 P6) the level-N measure: Phi at the identity coset, a_ell from T_ell on the whole symbol (ell !| N).
        const shapiro::Gamma0NCell& nc = shapiro_cell();
        if (nc.N % nc.ell == 0)
            throw hecke::NotImplementedInThisBuild("BirchModularSymbolRoute::measure: ell | N (U_ell, the multiplicative case) is not in this build");
        const birch::Vec P = symbol(x);
        const birch::SymRho rho(nc.n, nc.R);
        const shapiro::P1N p1(nc.N);
        uint64_t a = 0;
        if (!shapiro::EigenvalueN(rho, p1, P, nc.ell, a))
            throw std::invalid_argument("BirchModularSymbolRoute::measure: not a T_ell eigenclass (level N; a primitive symbol is required)");
        if (!nc.R.unit(a)) throw std::invalid_argument("BirchModularSymbolRoute::measure: not ordinary (a_ell == 0 mod ell)");
        const uint64_t alpha = req.alpha_override >= 0 ? static_cast<uint64_t>(req.alpha_override) % nc.R.q : birch::UnitRoot(nc.R, a, nc.n + 2);
        return shapiro::BuildMeasureN(rho, p1, P, a, alpha, req.level, req.ell_n_factor);
    }
    const birch::BirchConeCell& c = cell();
    const birch::Vec P = symbol(x);
    const birch::Ring& R = c.R;
    const birch::SymRho rho(c.n, R);
    const birch::Vec T = birch::HeckeTp(rho, P, R.ell);
    // a_ell at the coordinate of least valuation (a non-primitive class determines a_ell mod ell^{R - v})
    uint32_t t0 = 0, v0 = R.m;
    for (uint32_t t = 0; t <= c.n; ++t)
        if (R.val(P[t]) < v0) {
            v0 = R.val(P[t]);
            t0 = t;
        }
    if (v0 >= R.m) throw std::invalid_argument("BirchModularSymbolRoute::measure: not a T_ell eigenclass (the zero class has no eigenvalue)");
    if (R.val(T[t0]) < v0) throw std::invalid_argument("BirchModularSymbolRoute::measure: not a T_ell eigenclass");
    const uint64_t pv = R.ell_pow(v0);
    const birch::Ring Rr = birch::MakeRing(R.ell, R.m - v0);
    const uint64_t a = Rr.mul((T[t0] / pv) % Rr.q, Rr.inv((P[t0] / pv) % Rr.q));
    for (uint32_t t = 0; t <= c.n; ++t)
        if (T[t] != R.mul(a, P[t])) throw std::invalid_argument("BirchModularSymbolRoute::measure: not a T_ell eigenclass (T_ell P != a_ell P)");
    if (!R.unit(a)) throw std::invalid_argument("BirchModularSymbolRoute::measure: not ordinary (a_ell == 0 mod ell)");
    const uint64_t alpha = req.alpha_override >= 0 ? static_cast<uint64_t>(req.alpha_override) % R.q : birch::UnitRoot(R, a, c.n + 2);
    return birch::BuildMeasure(rho, P, a, alpha, req.level, req.ell_n_factor);
}

} // namespace lvalue
} // namespace hecke_ext
