// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LIN) the (BS-*) arms of HeckeLValueValidation -- see HeckeLValueBirchArms.h.
//
// Arms (R19-8 Sec. 6 gates; predictions Q1-Q9 of docs/implementation/slot19/wplin/predictions.md):
//   (BS-G5)  rho_n by L D U == the O(n^3) construction from [C] def:rhon; rho multiplicative; Manin matrices in SL_2(Z).
//   (BS-G1)  Psi : H^1_c(R) -> W(R) is an isomorphism (lengths, invariant factors), Psi kills the cone coboundaries,
//            ell^{e_i} Psi(b_i) = 0; negative: the wrong boundary sign g := +c_d leaves W.
//   (BS-G2)  Manin's relations on every class; Gamma-equivariance and additivity on 20 seeded (gamma, r, s, t); T_p P in W.
//   (BS-G3)  Psi o T^c_p = T_p^sym o Psi (the C++ cone action ModuleConeActionT); negative: rho(alpha) for rho(adj alpha);
//            Psi o iota^*_[C] = -rho(iota) o Psi read through pi : H^1_c -> H^1; negative: the geometric sign.
//   (BS-Q5)  the sign rule (ev_{chi,j} vanishes on the geometric sign -chi(-1)(-1)^j), also for non-real chi; chi != chibar.
//   (BS-G4)  R19-8's frozen numbers: r^{+-} as the a_2-eigenline of the cone action at a generic prime, the 9-twist tables;
//            leg (c) at the 8 Eisenstein cells (the route's index statement (L1), r^- == s Bern_k, the twist corollary (L2)).
//   (BS-Q8)  (258, 37): the m-block's structure per geometric sign on H^1_c, pi's rank on it (isign (K)), the unit-j sets.
//   (BS-API) the route through the interface: Domain construction, refusals, the normalisation record.
#include "HeckeLValueBirchArms.h"

#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "LValueRoute.h"

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

namespace {
#include "BirchFrozenR198.inc"

using namespace birch;

struct Lcg {
    uint64_t x = 0x9e3779b97f4a7c15ull;
    uint64_t next() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return x >> 17;
    }
    int64_t range(int64_t lo, int64_t hi) { return lo + static_cast<int64_t>(next() % static_cast<uint64_t>(hi - lo + 1)); }
};

std::string CellName(uint32_t n, uint64_t ell, uint32_t m) {
    return "(" + std::to_string(n) + "," + std::to_string(ell) + ") R=" + (m == 1 ? "F_ell" : "Z/ell^" + std::to_string(m));
}
std::string Join(const std::vector<uint32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return "[" + s + "]";
}
bool AllZero(const Vec& v) {
    for (uint64_t x : v)
        if (x) return false;
    return true;
}
Vec Scale(const Ring& R, uint64_t s, const Vec& v) {
    Vec o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = R.mul(s, v[i]);
    return o;
}
Vec Add(const Ring& R, const Vec& u, const Vec& v) {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R.add(u[i], v[i]);
    return o;
}
Vec Sub(const Ring& R, const Vec& u, const Vec& v) {
    Vec o(u.size());
    for (size_t i = 0; i < u.size(); ++i) o[i] = R.sub(u[i], v[i]);
    return o;
}
Vec Reduce(const Vec& v, uint64_t q) {
    Vec o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = v[i] % q;
    return o;
}
uint32_t MinVal(const Ring& R, const Vec& v) {
    uint32_t mn = R.m;
    for (uint64_t x : v) mn = std::min(mn, R.val(x));
    return mn;
}
// decimal (optionally signed) -> residue mod q
uint64_t DecMod(const std::string& s, const Ring& R) {
    bool neg = false;
    size_t i = 0;
    if (!s.empty() && s[0] == '-') {
        neg = true;
        i = 1;
    }
    uint64_t r = 0;
    for (; i < s.size(); ++i) r = R.add(R.mul(r, 10 % R.q), static_cast<uint64_t>(s[i] - '0') % R.q);
    return neg ? R.neg(r) : r;
}
Vec ParseVec(const char* text, const Ring& R) {
    std::istringstream is(text);
    std::string tok;
    Vec v;
    while (is >> tok) v.push_back(DecMod(tok, R));
    return v;
}
std::vector<std::string> ParseTokens(const char* text) {
    std::istringstream is(text);
    std::string tok;
    std::vector<std::string> v;
    while (is >> tok) v.push_back(tok);
    return v;
}
// v == lambda * w for a unit lambda (read at a unit coordinate of w); both mod q
bool ProportionalByUnit(const Ring& R, const Vec& v, const Vec& w) {
    for (size_t t = 0; t < w.size(); ++t)
        if (R.unit(w[t])) {
            if (!R.unit(v[t])) return false;
            const uint64_t lam = R.mul(v[t], R.inv(w[t]));
            return Scale(R, lam, w) == v;
        }
    return false;
}
// the relation matrix of W: rows (1 + S) and (1 + U + U^2), columns u_t
Mat WRelationMatrix(const SymRho& rho) {
    const Ring& R = rho.ring();
    const uint32_t n = rho.n();
    Mat A(2 * (n + 1), n + 1);
    const Mat2 U{0, -1, 1, 1};
    for (uint32_t t = 0; t <= n; ++t) {
        Vec e(n + 1, 0);
        e[t] = 1 % R.q;
        const Vec s = Add(R, e, rho.S(e));
        const Vec u1 = rho.apply(U, e), u2 = rho.apply(U, u1);
        const Vec uu = Add(R, Add(R, e, u1), u2);
        for (uint32_t i = 0; i <= n; ++i) {
            A.at(i, t) = s[i];
            A.at(n + 1 + i, t) = uu[i];
        }
    }
    return A;
}
Mat HeckeOnBasis(const SymRho& rho, const PivotBasis& B, uint64_t p) {
    const uint32_t r = static_cast<uint32_t>(B.basis.size());
    Mat T(r, r);
    for (uint32_t j = 0; j < r; ++j) {
        const Vec c = B.coords(HeckeTp(rho, B.basis[j], p));
        for (uint32_t i = 0; i < r; ++i) T.at(i, j) = c[i];
    }
    return T;
}
std::pair<int64_t, int64_t> ApplyCusp(const Mat2& g, std::pair<int64_t, int64_t> x) {
    int64_t P = g.a * x.first + g.b * x.second, Q = g.c * x.first + g.d * x.second;
    if (Q < 0 || (Q == 0 && P < 0)) {
        P = -P;
        Q = -Q;
    }
    if (Q == 0) return {1, 0};
    int64_t a = P < 0 ? -P : P, b = Q;
    while (b) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return {P / a, Q / a};
}
std::pair<int64_t, int64_t> RandomCusp(Lcg& rng) {
    int64_t p = rng.range(-20, 20), q = rng.range(1, 15);
    int64_t a = p < 0 ? -p : p, b = q;
    while (b) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    if (a == 0) return {0, 1};
    return {p / a, q / a};
}
Mat2 RandomSL2(Lcg& rng) {
    for (;;) {
        const int64_t a = rng.range(-7, 7), b = rng.range(-7, 7), c = rng.range(1, 9);
        if (a != 0 && (1 + b * c) % a == 0) return Mat2{a, b, c, (1 + b * c) / a};
    }
}
const std::vector<int64_t> kTwistsD = {-3, -4, 5, -7, 8, -8, 12, 13};

// ---- (BS-G5) ----------------------------------------------------------------------------------------
void ArmRhoSelfTest(const BirchCheckFn& check) {
    struct C { uint32_t n; uint64_t ell; uint32_t m; };
    Lcg rng;
    uint32_t trials = 0, bad = 0, mult_bad = 0, manin_bad = 0;
    for (const C& c : std::vector<C>{{10, 691, 3}, {24, 37, 2}, {12, 5, 4}, {40, 1000003, 2}}) {
        const Ring R = MakeRing(c.ell, c.m);
        const SymRho rho(c.n, R);
        for (int it = 0; it < 25; ++it) {
            Mat2 A{rng.range(-9, 9), rng.range(-9, 9), rng.range(-9, 9), rng.range(-9, 9)};
            Mat2 B{rng.range(-9, 9), rng.range(-9, 9), rng.range(-9, 9), rng.range(-9, 9)};
            auto has_unit = [&](const Mat2& X) { return R.unit(R.from_int(X.a)) || R.unit(R.from_int(X.b)) || R.unit(R.from_int(X.c)) || R.unit(R.from_int(X.d)); };
            if (!has_unit(A) || !has_unit(B)) continue;
            const Mat2 AB{A.a * B.a + A.b * B.c, A.a * B.b + A.b * B.d, A.c * B.a + A.d * B.c, A.c * B.b + A.d * B.d};
            if (!has_unit(AB)) continue;
            Vec v(c.n + 1);
            for (auto& x : v) x = rng.next() % R.q;
            const std::vector<uint64_t> D = RhoDirectMatrix(A, c.n, R);
            Vec dv(c.n + 1, 0);
            for (uint32_t i = 0; i <= c.n; ++i)
                for (uint32_t j = 0; j <= c.n; ++j) dv[i] = R.add(dv[i], R.mul(D[static_cast<size_t>(i) * (c.n + 1) + j], v[j]));
            ++trials;
            if (rho.apply(A, v) != dv) ++bad;
            if (rho.apply(AB, v) != rho.apply(A, rho.apply(B, v))) ++mult_bad;
        }
        for (int64_t M : {1, 3, 7, 12, 13, 97})
            for (int64_t a = -M; a <= M; ++a)
                for (const Mat2& g : ManinPath(a, M))
                    if (g.a * g.d - g.b * g.c != 1) ++manin_bad;
    }
    std::cout << "  BIRCH_G5 rho_LDU_vs_direct trials=" << trials << " bad=" << bad << " multiplicative_bad=" << mult_bad
              << " manin_det_bad=" << manin_bad << std::endl;
    check(trials >= 60 && bad == 0, "(BS-G5) rho_n(A) by L D U == the O(n^3) construction from [C] def:rhon (" + std::to_string(trials) + " trials)");
    check(mult_bad == 0, "(BS-G5) rho_n(AB) = rho_n(A) rho_n(B) on vectors (monoid homomorphism)");
    check(manin_bad == 0, "(BS-G5) every Manin matrix lies in SL_2(Z)");
}

// ---- (BS-G1) + (BS-G2) on one cell ---------------------------------------------------------------------
struct G1Result {
    bool iso = false;
    bool wrong_sign_leaves_W = false;
};
G1Result GateG1G2(const BirchCheckFn& check, const BirchConeCell& cell, uint32_t expect_dim_field, bool hecke_in_W) {
    G1Result res;
    const Ring& R = cell.R;
    const SymRho rho(cell.n, R);
    const std::string tc = "(BS-G1) " + CellName(cell.n, cell.ell, cell.m);
    check(cell.coboundaries_checked > 0 && cell.coboundaries_killed, tc + ": Psi kills the cone coboundaries (" + cell.coboundary_route + ", " +
                                                                         std::to_string(cell.coboundaries_checked) + " checked)");
    bool inW = true, torsion_ok = true, wrong_out = false;
    for (size_t i = 0; i < cell.P.size(); ++i) {
        inW = inW && InW(rho, cell.P[i]);
        if (cell.exps[i] < R.m) torsion_ok = torsion_ok && AllZero(Scale(R, R.ell_pow(cell.exps[i]), cell.P[i]));
        if (!InW(rho, cell.P_wrong_sign[i])) wrong_out = true;
    }
    check(inW, tc + ": Psi(b_i) lies in W (Manin's two- and three-term relations, G2)");
    check(torsion_ok, tc + ": ell^{e_i} Psi(b_i) = 0 on every cyclic summand");
    uint32_t LH = 0;
    for (uint32_t e : cell.exps) LH += e;
    const uint32_t Lim = SpanLength(R, cell.P, cell.n + 1);
    std::vector<Vec> wg;
    std::vector<uint32_t> we;
    KernelGenerators(R, WRelationMatrix(rho), wg, we);
    uint32_t LW = 0;
    for (uint32_t e : we) LW += e;
    std::vector<uint32_t> invH = cell.exps, invW = we;
    std::sort(invH.rbegin(), invH.rend());
    std::sort(invW.rbegin(), invW.rend());
    const std::vector<uint32_t> invIm = SpanInvariantFactors(R, cell.P, cell.n + 1);
    res.iso = inW && LH == Lim && Lim == LW && invH == invW && invIm == invH;
    std::cout << "  BIRCH_G1 n=" << cell.n << " ell=" << cell.ell << " m=" << cell.m << " route=\"" << cell.route << "\" Hc_exps=" << Join(invH)
              << " W_exps=" << Join(invW) << " len_Hc=" << LH << " len_imPsi=" << Lim << " len_W=" << LW
              << " coboundaries=" << cell.coboundaries_checked << " wrong_sign_leaves_W=" << (wrong_out ? "yes" : "no")
              << " seconds=" << cell.seconds << std::endl;
    check(res.iso, tc + ": Psi : H^1_c(R) -> W(R) is an isomorphism (lengths " + std::to_string(LH) + " = " + std::to_string(Lim) + " = " +
                       std::to_string(LW) + ", invariant factors " + Join(invH) + " = " + Join(invW) + ")");
    if (cell.m == 1) check(LH == expect_dim_field, tc + ": dim H^1_c = " + std::to_string(expect_dim_field) + " (R19-8 model_check.out)");
    res.wrong_sign_leaves_W = wrong_out;
    // (BS-G2) Gamma-equivariance and additivity on 20 seeded trials of random classes
    Lcg rng;
    rng.x ^= (static_cast<uint64_t>(cell.n) << 32) ^ cell.ell ^ cell.m;
    uint32_t bad = 0;
    for (int it = 0; it < 20; ++it) {
        Vec P(cell.n + 1, 0);
        for (size_t i = 0; i < cell.P.size(); ++i) P = Add(R, P, Scale(R, rng.next() % R.q, cell.P[i]));
        const Mat2 g = RandomSL2(rng);
        const auto r = RandomCusp(rng), t = RandomCusp(rng);
        const std::pair<int64_t, int64_t> s{1, 0};
        const auto gr = ApplyCusp(g, r), gs = ApplyCusp(g, s);
        const Vec lhs = PhiPair(rho, P, gr.first, gr.second, gs.first, gs.second);
        const Vec rhs = rho.apply(g, PhiPair(rho, P, r.first, r.second, s.first, s.second));
        if (lhs != rhs) ++bad;
        if (Add(R, PhiPair(rho, P, r.first, r.second, t.first, t.second), PhiPair(rho, P, t.first, t.second, s.first, s.second)) !=
            PhiPair(rho, P, r.first, r.second, s.first, s.second))
            ++bad;
    }
    check(bad == 0, "(BS-G2) " + CellName(cell.n, cell.ell, cell.m) + ": Phi(gamma{r,s}) = rho(gamma) Phi({r,s}) and additivity, 40 checks (" +
                        std::to_string(bad) + " bad)");
    if (hecke_in_W) {
        bool hw = true;
        for (const Vec& P : cell.P)
            for (uint64_t p : {2ull, 3ull, 5ull}) hw = hw && InW(rho, HeckeTp(rho, P, p));
        check(hw, "(BS-G2) " + CellName(cell.n, cell.ell, cell.m) + ": T_p Psi(b_i) lies in W, p = 2, 3, 5");
    }
    return res;
}

// ---- (BS-G3) -----------------------------------------------------------------------------------------
void GateG3(const BirchCheckFn& check, const BirchConeCell& cell) {
    const Ring& R = cell.R;
    const SymRho rho(cell.n, R);
    const std::string tc = "(BS-G3) " + CellName(cell.n, cell.ell, cell.m);
    const uint32_t t = static_cast<uint32_t>(cell.P.size());
    for (size_t h = 0; h < cell.hecke_primes.size(); ++h) {
        const uint64_t p = cell.hecke_primes[h];
        bool ok = true, wrong_differs = false;
        for (uint32_t i = 0; i < t; ++i) {
            Vec lhs(cell.n + 1, 0);
            for (uint32_t j = 0; j < t; ++j) lhs = Add(R, lhs, Scale(R, cell.A_c[h].at(j, i), cell.P[j]));
            const Vec rhs = HeckeTp(rho, cell.P[i], p);
            ok = ok && (lhs == rhs);
            if (HeckeTpWrongAdjoint(rho, cell.P[i], p) != rhs) wrong_differs = true;
        }
        check(ok, tc + ": Psi o T^c_" + std::to_string(p) + " = T_" + std::to_string(p) +
                      "^sym o Psi (the C++ cone action ModuleConeActionT vs the symbol formula; [C]'s T_p is the forms-side T_p)");
        check(wrong_differs, tc + ": negative: rho(alpha) in place of rho(adj alpha) gives a different T_" + std::to_string(p));
    }
    if (cell.have_iota && cell.have_pi) {
        const SmithForm F = Smith(R, ColumnsToMat(cell.P, cell.n + 1), true);
        const uint32_t tK = cell.iota_C.r;
        bool ok = true, geo_fails = false, solved = true, nonzero = false;
        auto red = [&](const Vec& x) {
            Vec o(x.size());
            for (size_t i = 0; i < x.size(); ++i) o[i] = cell.bulk_exps[i] >= R.m ? x[i] : x[i] % R.ell_pow(cell.bulk_exps[i]);
            return o;
        };
        for (uint32_t i = 0; i < t; ++i) {
            const Vec w = Sub(R, SignPart(R, cell.P[i], +1), SignPart(R, cell.P[i], -1));   // rho(iota) P (n even)
            Vec y;
            if (!Solve(R, F, w, y)) {
                solved = false;
                continue;
            }
            Vec ei(t, 0);
            ei[i] = 1 % R.q;
            const Vec pi_y = red(MatVec(R, cell.pi, y));
            const Vec pi_e = MatVec(R, cell.pi, ei);
            const Vec iota_pi = MatVec(R, cell.iota_C, pi_e);
            Vec minus(tK);
            for (uint32_t r = 0; r < tK; ++r) minus[r] = R.neg(iota_pi[r]);
            if (!AllZero(red(pi_e))) nonzero = true;
            ok = ok && (pi_y == red(minus));
            if (pi_y != red(iota_pi)) geo_fails = true;
        }
        check(solved, tc + ": rho(iota) Psi(b_i) lies in im Psi (Psi^{-1} solved)");
        check(nonzero && ok, tc + ": pi(Psi^{-1}(rho(iota) Psi x)) = -iota^*_[C] pi(x) on every class: Psi o iota^*_[C] = -rho(iota) o Psi, "
                                  "a geometric-even symbol is [C]-minus");
        check(geo_fails, tc + ": negative: +iota^*_[C] (the geometric reading) fails");
    }
}

// ---- (BS-Q5) -----------------------------------------------------------------------------------------
void GateSignRule(const BirchCheckFn& check, const std::shared_ptr<const BirchConeCell>& cellp) {
    const BirchConeCell& cell = *cellp;
    const Ring& R = cell.R;
    const SymRho rho(cell.n, R);
    std::vector<DirichletCharacter> chis = {TrivialCharacter(R)};
    for (int64_t D : kTwistsD) chis.push_back(KroneckerCharacter(D, R));
    const bool has4 = (R.ell - 1) % 4 == 0, has3 = (R.ell - 1) % 3 == 0;
    if (has4) {
        chis.push_back(PrimeConductorCharacter(5, 4, 1, R));
        chis.push_back(PrimeConductorCharacter(5, 4, 3, R));
    }
    if (has3) chis.push_back(PrimeConductorCharacter(7, 3, 1, R));
    uint32_t wrong_nonzero = 0, right_nonzero = 0, nonreal_checked = 0;
    for (const DirichletCharacter& chi : chis) {
        for (const Vec& P : cell.P)
            for (int s : {+1, -1}) {
                const Vec ev = EvAllJ(rho, SignPart(R, P, s), chi);
                for (uint32_t j = 0; j <= cell.n; ++j) {
                    const int geo = chi.parity * ((j % 2 == 0) ? 1 : -1);
                    if (geo != s && ev[j] != 0) ++wrong_nonzero;
                    if (geo == s && ev[j] != 0) ++right_nonzero;
                }
            }
    }
    check(wrong_nonzero == 0 && right_nonzero > 0, "(BS-Q5) " + CellName(cell.n, cell.ell, cell.m) + ": ev_{chi,j} vanishes identically on the "
                                                                                                    "geometric sign -chi(-1)(-1)^j (" +
                                                        std::to_string(chis.size()) + " characters; " + std::to_string(right_nonzero) +
                                                        " right-sign values non-zero)");
    if (has4) {
        const DirichletCharacter c1 = PrimeConductorCharacter(5, 4, 1, R), c3 = PrimeConductorCharacter(5, 4, 3, R);
        bool differs = false;
        for (const Vec& P : cell.P) {
            ++nonreal_checked;
            if (EvAllJ(rho, P, c1) != EvAllJ(rho, P, c3)) differs = true;
        }
        check(c1.chi == c3.chibar && differs, "(BS-Q5) negative: for the order-4 character mod 5, ev_chi != ev_{chibar} (the conjugation is seen)");
    }
    // the interface: evaluate_functional / evaluate_all_j agree with EvAllJ on the unit classes
    const DirichletCharacter chi = KroneckerCharacter(-3, R);
    const uint32_t j = 3;
    const BirchModularSymbolRoute route(cellp, chi, j);
    std::vector<CohomologyClass> cls;
    for (size_t i = 0; i < cell.P.size(); ++i) {
        CohomologyClass x;
        x.exponents = cell.exps;
        x.coordinates.assign(cell.P.size(), 0);
        x.coordinates[i] = 1;
        cls.push_back(x);
    }
    const std::vector<uint64_t> f = route.evaluate_functional(cls);
    bool agree = f.size() == cell.P.size();
    for (size_t i = 0; agree && i < cell.P.size(); ++i) agree = f[i] == EvAllJ(rho, cell.P[i], chi)[j];
    check(route.kind() == RouteKind::Linear && agree, "(BS-API) evaluate_functional (chi_{-3}, j = 3) = ev_{chi,j}(Psi(b_i)) on the basis classes");
    std::cout << "  NORMALISATION (B) " << route.normalisation().to_string() << std::endl;
}

// ---- (BS-G4) frozen numbers at a generic prime ---------------------------------------------------------
void GateFrozenGeneric(const BirchCheckFn& check) {
    const uint64_t ell = 1000003;
    const uint32_t m = 2;
    for (const BirchFrozenWeight& W : kBirchFrozenWeights) {
        const uint32_t n = W.k - 2;
        BirchCellOptions opt;
        opt.hecke_primes = {2};
        opt.coboundary_check = false;
        const BirchConeCell cell = RunBirchConeCell(n, ell, m, opt);
        const Ring& R = cell.R;
        const SymRho rho(n, R);
        const std::string tc = "(BS-G4) k=" + std::to_string(W.k) + " at ell=1000003, R=Z/ell^2";
        const Vec rE = ParseVec(W.r_even, R), rO = ParseVec(W.r_odd, R);
        const uint64_t a2 = R.from_int(W.a2);
        // the a_2-eigenline of the cone action
        Mat E = cell.A_c[0];
        for (uint32_t i = 0; i < E.r; ++i) E.at(i, i) = R.sub(E.at(i, i), a2);
        std::vector<Vec> kg;
        std::vector<uint32_t> ke;
        KernelGenerators(R, E, kg, ke);
        std::vector<Vec> evens, odds;
        for (const Vec& y : kg) {
            Vec w(n + 1, 0);
            for (size_t i = 0; i < y.size(); ++i) w = Add(R, w, Scale(R, y[i], cell.P[i]));
            evens.push_back(SignPart(R, w, +1));
            odds.push_back(SignPart(R, w, -1));
        }
        const PivotBasis Be = SaturatedBasis(R, evens), Bo = SaturatedBasis(R, odds);
        const bool free2 = kg.size() == 2 && ke[0] == m && ke[1] == m;
        check(free2 && Be.basis.size() == 1 && Bo.basis.size() == 1 && Be.free_summand && Bo.free_summand,
              tc + ": ker(T^c_2 - a_2) on H^1_c is free of rank 2, one line per geometric sign (a_2 = " + std::to_string(W.a2) + ")");
        if (Be.basis.size() == 1 && Bo.basis.size() == 1) {
            check(ProportionalByUnit(R, Be.basis[0], rE), tc + ": the geometric-even eigenline is R19-8's r^{geo-even} (every coordinate)");
            check(ProportionalByUnit(R, Bo.basis[0], rO), tc + ": the geometric-odd eigenline is R19-8's r^{geo-odd} (every coordinate)");
        }
        check(InW(rho, rE) && InW(rho, rO) && HeckeTp(rho, rE, 2) == Scale(R, a2, rE) && HeckeTp(rho, rO, 2) == Scale(R, a2, rO),
              tc + ": the frozen r^{+-} lie in W and T_2 r = a_2 r (symbols)");
        uint32_t rows = 0, bad = 0;
        for (const BirchFrozenTwist& T : kBirchFrozenTwists) {
            if (T.k != W.k) continue;
            const DirichletCharacter chi = T.D == 1 ? TrivialCharacter(R) : KroneckerCharacter(T.D, R);
            rows += 2;
            if (EvAllJ(rho, rE, chi) != ParseVec(T.ev_even, R)) ++bad;
            if (EvAllJ(rho, rO, chi) != ParseVec(T.ev_odd, R)) ++bad;
        }
        check(rows == 18 && bad == 0, tc + ": ev_{chi,j}(r^{+-}) = R19-8's 9-twist tables mod ell^2 (" + std::to_string(rows) + " rows, " +
                                          std::to_string(bad) + " differ)");
        // negative: the D = -3 table is not the D = -4 one
        if (W.k == 12) {
            const BirchFrozenTwist* t3 = nullptr;
            for (const BirchFrozenTwist& T : kBirchFrozenTwists)
                if (T.k == 12 && T.D == -3) t3 = &T;
            check(t3 && EvAllJ(rho, rO, KroneckerCharacter(-4, R)) != ParseVec(t3->ev_odd, R),
                  "(BS-G4) negative: ev under chi_{-4} differs from the frozen chi_{-3} table (the gate discriminates twists)");
        }
        std::cout << "  BIRCH_FROZEN k=" << W.k << " a2=" << W.a2 << " rows=" << rows << " differ=" << bad << " seconds=" << cell.seconds << std::endl;
    }
    // k = 12 directly mod 691^3
    {
        const Ring R = MakeRing(691, 3);
        const SymRho rho(10, R);
        const Vec rE = ParseVec(kBirchFrozenWeights[0].r_even, R), rO = ParseVec(kBirchFrozenWeights[0].r_odd, R);
        uint32_t bad = 0;
        for (const BirchFrozenTwist& T : kBirchFrozenTwists) {
            if (T.k != 12) continue;
            const DirichletCharacter chi = T.D == 1 ? TrivialCharacter(R) : KroneckerCharacter(T.D, R);
            if (EvAllJ(rho, rE, chi) != ParseVec(T.ev_even, R) || EvAllJ(rho, rO, chi) != ParseVec(T.ev_odd, R)) ++bad;
        }
        check(bad == 0, "(BS-G4) k=12: the 9-twist tables of Delta's r^{+-} reproduce mod 691^3");
    }
}

// Bern_k = sum_{i even, 2 <= i <= k-2} B_i B_{k-i} / (i! (k-i)!) X^{i-1}, in u_t coordinates (t = i - 1, sign (-1)^t), mod ell^m,
// divided by its ell-content (ell > k: every factorial is a unit).
Vec BernPeriod(uint32_t k, const Ring& R, uint32_t& content) {
    const uint32_t n = k - 2;
    auto bern = [&](uint32_t i) -> std::pair<uint32_t, uint64_t> {   // B_i = ell^v * u
        const BernoulliOverK b = ExactBernoulliOverK(i);
        uint64_t un = 0, ud = 0;
        const uint32_t vn = ValuationAndUnitPart(b.num, R.ell, R.q, un);
        const uint32_t vd = ValuationAndUnitPart(b.den, R.ell, R.q, ud);
        ENGINE_CHECK_MSG(vn >= vd, "BernPeriod: B_i is not ell-integral");
        uint64_t u = R.mul(R.mul(un, R.inv(ud)), static_cast<uint64_t>(i) % R.q);
        if (b.sign < 0) u = R.neg(u);
        return {vn - vd, u};
    };
    std::vector<uint32_t> vs(n + 1, R.m);
    std::vector<uint64_t> us(n + 1, 0);
    uint64_t fact[400];
    fact[0] = 1 % R.q;
    for (uint32_t i = 1; i <= k; ++i) fact[i] = R.mul(fact[i - 1], i % R.q);
    for (uint32_t i = 2; i + 2 <= k; i += 2) {
        const auto bi = bern(i), bk = bern(k - i);
        const uint32_t t = i - 1;
        vs[t] = bi.first + bk.first;
        us[t] = R.mul(R.mul(bi.second, bk.second), R.inv(R.mul(fact[i], fact[k - i])));
        if (t % 2 == 1) us[t] = R.neg(us[t]);
    }
    content = R.m;
    for (uint32_t t = 0; t <= n; ++t)
        if (vs[t] < R.m) content = std::min(content, vs[t]);
    Vec out(n + 1, 0);
    for (uint32_t t = 0; t <= n; ++t)
        if (vs[t] < R.m) out[t] = R.mul(R.ell_pow(vs[t] - content), us[t]);
    return out;
}

// ---- (BS-G4) leg (c) at the Eisenstein cells; (L1), (L2) ---------------------------------------------------
void GateLegC(const BirchCheckFn& check) {
    struct E { uint32_t k; uint64_t ell; };
    const uint32_t m = 3;
    for (const E& c : std::vector<E>{{12, 691}, {16, 3617}, {18, 43867}, {20, 283}, {20, 617}, {22, 131}, {22, 593}, {26, 657931}}) {
        const uint32_t n = c.k - 2;
        BirchCellOptions opt;
        opt.coboundary_check = false;
        const auto cellp = std::make_shared<const BirchConeCell>(RunBirchConeCell(n, c.ell, m, opt));
        const BirchConeCell& cell = *cellp;
        const Ring& R = cell.R;
        const SymRho rho(n, R);
        const std::string tc = "(BS-G4) leg (c) (k,ell)=(" + std::to_string(c.k) + "," + std::to_string(c.ell) + ") R=Z/ell^3";
        const BirchModularSymbolRoute route(cellp, TrivialCharacter(R), 0);
        const IndexStatement I = route.index_statement();
        std::cout << "  LVALUE " << route.name() << " k=" << c.k << " ell=" << c.ell << " m=" << m << " " << I.to_string() << std::endl;
        check(I.holds && I.found == 1 && I.l_valuation == 1, tc + ": (L1) r^+ == unit * p_0 mod ell to depth exactly v(B_k/k) = 1; " + I.to_string());
        // negative: the wrong weight
        if ((c.k + 2) % (c.ell - 1) != 0) {
            const BernoulliReading Bw = BernoulliModEllPowerExact(c.k + 2, c.ell, m);
            if (Bw.v != I.found) check(true, tc + ": negative: v(B_{k+2}/(k+2)) = " + std::to_string(Bw.v) + " != depth (the wrong weight fails L1)");
            else check(false, tc + ": negative control vacuous: v(B_{k+2}/(k+2)) = depth");
        }
        const EisensteinSymbolReading S = ReadEisensteinSymbols(cell);
        check(S.applicable && S.block_rank == 2 && S.have_r_minus && S.v_y == 0 && S.v_delta == 1,
              tc + ": the geometric-even m_Eis block is <p_0, r^+> (rank 2, T_2 w = y p_0 + a w with y a unit, v(a - e_2) = 1); W^-_{m_Eis} rank 1");
        // compare with the frozen period vectors
        const BirchFrozenWeight* W = nullptr;
        for (const BirchFrozenWeight& x : kBirchFrozenWeights)
            if (x.k == c.k) W = &x;
        if (W && S.applicable && S.have_r_minus) {
            const Ring R2 = MakeRing(c.ell, m - S.v_delta);
            check(ProportionalByUnit(R2, Reduce(S.r_plus, R2.q), ParseVec(W->r_even, R2)),
                  tc + ": r^+ (C++, known mod ell^2) is R19-8's r^{geo-even} up to a unit");
            check(ProportionalByUnit(R, S.r_minus, ParseVec(W->r_odd, R)), tc + ": r^- (C++) is R19-8's r^{geo-odd} up to a unit, mod ell^3");
            // r^- == s Bern_k mod ell, not mod ell^2
            uint32_t content = 0;
            const Vec Bn = BernPeriod(c.k, R, content);
            uint32_t vdev = 0;
            bool found_unit = false;
            for (uint32_t t = 0; t <= n && !found_unit; ++t)
                if (R.unit(Bn[t])) {
                    found_unit = true;
                    const uint64_t s = R.mul(S.r_minus[t], R.inv(Bn[t]));
                    vdev = MinVal(R, Sub(R, S.r_minus, Scale(R, s, Bn)));
                }
            const Ring R1 = MakeRing(c.ell, 1), Rq2 = MakeRing(c.ell, 2);
            const SymRho rho1(n, R1), rho2(n, Rq2);
            const Vec B1 = Reduce(Bn, R1.q), B2 = Reduce(Bn, Rq2.q);
            bool eig = true;
            for (uint64_t p : {2ull, 3ull, 5ull, 7ull})
                eig = eig && HeckeTp(rho1, B1, p) == Scale(R1, R1.add(1, R1.pow(p, c.k - 1)), B1);
            std::cout << "  BIRCH_LEGC k=" << c.k << " ell=" << c.ell << " depth(r^+,p_0)=" << S.depth << " v(r^- - s Bern)=" << vdev
                      << " Bern_content=" << content << " Bern_in_W(F_ell)=" << (InW(rho1, B1) ? "yes" : "no")
                      << " Bern_in_W(Z/ell^2)=" << (InW(rho2, B2) ? "yes" : "no") << " seconds=" << cell.seconds << std::endl;
            check(found_unit && content == 0 && vdev == 1, tc + ": r^- == s * Bern_k (mod ell), not mod ell^2 (depth 1)");
            check(InW(rho1, B1) && !InW(rho2, B2), tc + ": Bern_k in W(F_ell), not in W(Z/ell^2)");
            check(eig, tc + ": T_p Bern_k == (1 + p^{k-1}) Bern_k (mod ell), p = 2, 3, 5, 7");
            // (L2) the twist corollary; the r^- unit count against R19-8's non-zero count
            uint32_t plus_bad = 0, minus_units = 0, frozen_nonzero = 0;
            const SymRho rhoR2(n, R2);
            const Vec rp = Reduce(S.r_plus, R2.q);
            for (int64_t D : kTwistsD) {
                const DirichletCharacter chi2 = KroneckerCharacter(D, R2), chi = KroneckerCharacter(D, R);
                for (uint64_t v : EvAllJ(rhoR2, rp, chi2))
                    if (v % c.ell != 0) ++plus_bad;
                const Vec evm = EvAllJ(rho, S.r_minus, chi);
                for (uint32_t j = 0; j <= n; ++j)
                    if (R.unit(evm[j])) ++minus_units;
                for (const BirchFrozenTwist& T : kBirchFrozenTwists)
                    if (T.k == c.k && T.D == D)
                        for (const std::string& tok : ParseTokens(T.ev_odd))
                            if (tok != "0") ++frozen_nonzero;
            }
            check(plus_bad == 0, tc + ": (L2) every twisted value ev_{chi,j}(r^+) == 0 (mod ell) (8 quadratic twists)");
            check(minus_units == frozen_nonzero && frozen_nonzero > 0,
                  tc + ": (L2) on r^-, the ell-unit twisted values = R19-8's non-zero ones (" + std::to_string(minus_units) + " = " +
                      std::to_string(frozen_nonzero) + ")");
        }
    }
    // controls v(B_k/k) = 0: (10, 37), (22, 37)
    for (uint32_t n : {10u, 22u}) {
        BirchCellOptions opt;
        opt.coboundary_check = false;
        const auto cellp = std::make_shared<const BirchConeCell>(RunBirchConeCell(n, 37, m, opt));
        const BirchModularSymbolRoute route(cellp, TrivialCharacter(cellp->R), 0);
        const IndexStatement I = route.index_statement();
        std::cout << "  LVALUE " << route.name() << " k=" << n + 2 << " ell=37 m=" << m << " " << I.to_string() << std::endl;
        check(I.holds && I.found == 0 && I.l_valuation == 0, "(BS-G4) control (" + std::to_string(n) + ",37): (L1) depth 0 = v(B_k/k); " + I.to_string());
    }
}

// ---- (BS-Q8) (258, 37) ----------------------------------------------------------------------------------
void GateCell258(const BirchCheckFn& check, const BirchConeCell& cell) {
    const Ring& R = cell.R;   // F_37
    const uint32_t n = cell.n;
    const SymRho rho(n, R);
    const std::string tc = "(BS-Q8) (258,37) " + std::string(cell.m == 1 ? "F_37" : "Z/37^" + std::to_string(cell.m));
    std::vector<Vec> ev, od;
    for (const Vec& P : cell.P) {
        ev.push_back(SignPart(R, P, +1));
        od.push_back(SignPart(R, P, -1));
    }
    const PivotBasis Be = SaturatedBasis(R, ev), Bo = SaturatedBasis(R, od);
    check(Be.free_summand && Bo.free_summand && Be.basis.size() == 22 && Bo.basis.size() == 21,
          tc + ": W^{geo-even} rank 22 (21 + p_0), W^{geo-odd} rank 21, both free direct summands");
    const Mat Te = HeckeOnBasis(rho, Be, 2), To = HeckeOnBasis(rho, Bo, 2);
    {
        const Vec ce = CharPoly(R, Te), co = CharPoly(R, To);
        const uint64_t e = R.add(1, R.pow(2, 259));
        Vec prod(co.size() + 1, 0);   // (X - e) * co
        for (size_t i = 0; i < co.size(); ++i) {
            prod[i + 1] = R.add(prod[i + 1], co[i]);
            prod[i] = R.sub(prod[i], R.mul(e, co[i]));
        }
        check(prod == ce, tc + ": charpoly(T_2 | W^{even}) = (X - 1 - 2^259) charpoly(T_2 | W^{odd})");
    }
    const SmithForm F = Smith(R, ColumnsToMat(cell.P, n + 1), true);
    struct SignRes { std::vector<Vec> block; std::vector<uint32_t> ranks; uint32_t pi_rank = 0; Mat t; PivotBasis B; };
    auto analyse = [&](const PivotBasis& B, const Mat& T) {
        SignRes s;
        const uint32_t r = static_cast<uint32_t>(B.basis.size());
        Mat t = T;
        for (uint32_t i = 0; i < r; ++i) t.at(i, i) = R.sub(t.at(i, i), 29);
        Mat tN(r, r);
        for (uint32_t i = 0; i < r; ++i) tN.at(i, i) = 1;
        for (uint32_t i = 0; i < r * R.m + 1; ++i) tN = MatMul(R, tN, t);
        std::vector<Vec> kg;
        std::vector<uint32_t> ke;
        KernelGenerators(R, tN, kg, ke);
        for (const Vec& y : kg) {
            Vec w(n + 1, 0);
            for (uint32_t i = 0; i < r; ++i) w = Add(R, w, Scale(R, y[i], B.basis[i]));
            s.block.push_back(w);
        }
        s.B = SaturatedBasis(R, s.block);
        const uint32_t b = static_cast<uint32_t>(s.B.basis.size());
        s.t = Mat(b, b);
        for (uint32_t j = 0; j < b; ++j) {
            const Vec tv = Sub(R, HeckeTp(rho, s.B.basis[j], 2), Scale(R, 29, s.B.basis[j]));
            const Vec c = s.B.coords(tv);
            for (uint32_t i = 0; i < b; ++i) s.t.at(i, j) = c[i] % R.ell;
        }
        // ranks of t, t^2, t^3 mod ell
        const Ring R1 = MakeRing(R.ell, 1);
        Mat tm(b, b);
        for (uint32_t i = 0; i < b * b; ++i) tm.a[i] = s.t.a[i] % R.ell;
        Mat pw = tm;
        for (int e = 1; e <= 3; ++e) {
            s.ranks.push_back(static_cast<uint32_t>(Smith(R1, pw, false).v.size()));
            pw = MatMul(R1, pw, tm);
        }
        // pi on the block, mod ell
        if (cell.have_pi) {
            std::vector<Vec> imgs;
            for (const Vec& w : s.B.basis) {
                Vec y;
                if (!Solve(R, F, w, y)) return s;
                Vec py = MatVec(R, cell.pi, y);
                for (auto& x : py) x %= R.ell;
                imgs.push_back(py);
            }
            s.pi_rank = static_cast<uint32_t>(Smith(R1, ColumnsToMat(imgs, cell.pi.r), false).v.size());
        }
        return s;
    };
    const SignRes E = analyse(Be, Te), O = analyse(Bo, To);
    std::cout << "  BIRCH_258 sign=geo-even([C]-minus) block_rank=" << E.B.basis.size() << " ranks(t,t^2,t^3)=" << Join(E.ranks)
              << " pi_rank=" << E.pi_rank << " | sign=geo-odd([C]-plus) block_rank=" << O.B.basis.size() << " ranks=" << Join(O.ranks)
              << " pi_rank=" << O.pi_rank << std::endl;
    check(E.B.basis.size() == 4 && O.B.basis.size() == 4, tc + ": the block m = (37, T_2 - 29) has rank 4 in each geometric sign");
    check(E.ranks == std::vector<uint32_t>({3, 2, 1}), tc + ": geometric-even = [C]-minus: ranks of t, t^2, t^3 mod 37 = (3,2,1): FREE over T_m");
    check(O.ranks == std::vector<uint32_t>({2, 1, 0}), tc + ": geometric-odd = [C]-plus: ranks (2,1,0): A/x^3 + A/x = m (isign: on H^1_c the free "
                                                         "sign is [C]-minus)");
    if (cell.have_pi)
        check(E.pi_rank == 3 && O.pi_rank == 3, tc + ": pi : H^1_{c,m} -> H^1_m mod 37 has rank 3 of 4 in each sign (isign (K))");
    if (cell.m != 1) return;
    // unit-j sets on the reduction lines (F_37)
    {
        std::vector<Vec> kg;
        std::vector<uint32_t> ke;
        KernelGenerators(R, E.t, kg, ke);
        check(kg.size() == 1, tc + ": ker t on the free block is a line");
        if (kg.size() == 1) {
            Vec v(n + 1, 0);
            for (size_t i = 0; i < kg[0].size(); ++i) v = Add(R, v, Scale(R, kg[0][i], E.B.basis[i]));
            const Vec ev1 = EvAllJ(rho, v, TrivialCharacter(R));
            std::vector<uint32_t> units;
            for (uint32_t j = 0; j <= n; ++j)
                if (R.unit(ev1[j])) units.push_back(j);
            const std::vector<uint32_t> want = {6, 36, 42, 72, 78, 108, 114, 144, 150, 180, 186, 216, 222, 252};
            check(units == want, tc + ": free sign, chi = 1: ev_j is a unit exactly at j in " + Join(want) + " (R19-8; got " + Join(units) + ")");
            uint32_t tw_bad = 0;
            for (int64_t D : kTwistsD)
                for (uint64_t x : EvAllJ(rho, v, KroneckerCharacter(D, R)))
                    if (x) ++tw_bad;
            check(tw_bad == 0, tc + ": free sign: every twisted value (8 D) == 0 (mod 37)");
        }
    }
    {
        // the line ker t cap im t^2 on the m-sign block (prediction Q8)
        const uint32_t b = O.t.r;
        const Mat t2 = MatMul(R, O.t, O.t);
        Vec u;
        for (uint32_t j = 0; j < b && u.empty(); ++j) {
            Vec col(b);
            for (uint32_t i = 0; i < b; ++i) col[i] = t2.at(i, j);
            if (!AllZero(col)) u = col;
        }
        const bool in_ker = !u.empty() && AllZero(MatVec(R, O.t, u));
        std::vector<uint32_t> nonunits;
        uint32_t right = 0;
        if (!u.empty()) {
            Vec v(n + 1, 0);
            for (uint32_t i = 0; i < b; ++i) v = Add(R, v, Scale(R, u[i], O.B.basis[i]));
            const Vec ev1 = EvAllJ(rho, v, TrivialCharacter(R));
            for (uint32_t j = 1; j <= n; j += 2) {
                ++right;
                if (!R.unit(ev1[j])) nonunits.push_back(j);
            }
        }
        bool pattern = nonunits.size() == 24;
        for (uint32_t j : nonunits) pattern = pattern && (j % 36 == 1 || j % 36 == 3 || j % 36 == 5);
        std::cout << "  BIRCH_258_MSIGN line=ker_t_cap_im_t2 in_ker=" << (in_ker ? "yes" : "no") << " right_sign_j=" << right
                  << " nonunits=" << nonunits.size() << " pattern_1_3_5_mod_36=" << (pattern ? "yes" : "no") << std::endl;
        check(in_ker && right == 129 && pattern,
              tc + ": m-sign, the line ker t cap im t^2: the untwisted non-units are exactly j == 1, 3, 5 (mod 36), 24 of 129 (R19-8's "
                   "eigen-reduction pattern)");
    }
}

// ---- (BS-API) ---------------------------------------------------------------------------------------------
template<class F> bool ThrowsNI(F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
template<class F> bool ThrowsLogicNotNI(F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return false;
    } catch (const std::logic_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
void GateApi(const BirchCheckFn& check) {
    Domain d;
    d.model = ModelId::SL2ZSevenVertex;
    d.cohomology = "H^1_c";
    d.n_sym = 10;
    d.ell = 691;
    d.m = 3;
    d.subquotient = "H^1_c";
    const BirchModularSymbolRoute route(d, 1, 0);
    const IndexStatement I = route.index_statement();
    check(route.kind() == RouteKind::Linear && I.holds && I.found == 1, "(BS-API) the declared constructor (Domain, M = 1, j = 0) at (10,691) "
                                                                          "Z/691^3 builds its cone and carries (L1); " + I.to_string());
    std::vector<CohomologyClass> cls;
    for (size_t i = 0; i < route.cell().P.size(); ++i) {
        CohomologyClass x;
        x.coordinates.assign(route.cell().P.size(), 0);
        x.coordinates[i] = 1;
        cls.push_back(x);
    }
    const auto all = route.evaluate_all_j(cls);
    const auto f = route.evaluate_functional(cls);
    bool ok = all.size() == cls.size() && f.size() == cls.size();
    for (size_t i = 0; ok && i < cls.size(); ++i) ok = all[i].size() == 11 && f[i] == all[i][0];
    check(ok, "(BS-API) evaluate_functional = the j-th entry of evaluate_all_j");
    check(ThrowsLogicNotNI([&] { (void)route.congruence_module(); }) && ThrowsLogicNotNI([&] { (void)route.evaluate_pairing({}, {}); }),
          "(BS-API) the linear route refuses the congruence and bilinear evaluations (kind)");
    Domain mp = d;
    mp.model = ModelId::MoerdijkPronk;
    Domain h1 = d;
    h1.cohomology = "H^1";
    check(ThrowsNI([&] { BirchModularSymbolRoute x(mp, 1, 0); }), "(BS-API) the Moerdijk-Pronk model is refused by name (NotImplementedInThisBuild)");
    check(ThrowsLogicNotNI([&] { BirchModularSymbolRoute x(h1, 1, 0); }), "(BS-API) the domain H^1 is refused (the functional lives on H^1_c)");
    check(ThrowsLogicNotNI([&] { BirchModularSymbolRoute x(d, 691, 0); }), "(BS-API) ell | M is refused");
    check(ThrowsLogicNotNI([&] { BirchModularSymbolRoute x(d, 8, 0); }), "(BS-API) conductor 8 (two quadratic characters) needs the explicit character");
    const NormalisationRecord N = BirchModularSymbolRoute(d, 3, 4).normalisation();
    check(N.sign_at_infinity == '+' && N.twist.find("geometric sign chi(-1)(-1)^j=-") != std::string::npos,
          "(BS-API) the record prints both signs: chi_{-3}, j = 4: geometric -, [C] +");
}
} // namespace

void RunBirchArms(const BirchCheckFn& check, bool heavy) {
    const auto t0 = std::chrono::steady_clock::now();
    ArmRhoSelfTest(check);
    // (BS-G1)/(BS-G2): F_ell and Z/ell^3
    struct C { uint32_t n; uint64_t ell; uint32_t dim; };
    uint32_t wrong_cells = 0, cells = 0;
    for (const C& c : std::vector<C>{{10, 691, 3}, {10, 5, 3}, {10, 7, 3}, {22, 144169, 5}, {36, 37, 5}, {40, 37, 7}, {72, 37, 12}}) {
        for (uint32_t m : {1u, 3u}) {
            try {
                const BirchConeCell cell = RunBirchConeCell(c.n, c.ell, m, BirchCellOptions());
                const G1Result r = GateG1G2(check, cell, c.dim, true);
                if (m >= 2) {   // over F_ell the stored D5b basis need not see the boundary sign (recorded only)
                    ++cells;
                    if (r.wrong_sign_leaves_W) ++wrong_cells;
                }
                if (c.n == 72 && c.ell == 37 && m == 3) {
                    bool torsion = false;
                    for (uint32_t e : cell.exps) torsion = torsion || e < 3;
                    check(torsion, "(BS-G1) (72,37) Z/37^3: H^1_c has a torsion summand (the mod-37 class), and Psi matches it with W's");
                }
            } catch (const std::exception& e) {
                check(false, "(BS-G1) " + CellName(c.n, c.ell, m) + " threw: " + e.what());
            }
        }
    }
    check(wrong_cells == cells && cells > 0, "(BS-G1) negative: the wrong boundary sign g := +c_d leaves W at every Z/ell^3 cell (" +
                                                 std::to_string(wrong_cells) + " of " + std::to_string(cells) + ")");
    // (BS-G3) + (BS-Q5)
    std::vector<C> g3 = {{10, 691, 3}, {22, 144169, 5}, {40, 37, 7}};
    if (heavy) g3.push_back({72, 37, 12});   // ~ 12 s: the torsion cell, HECKE_LVALUE_HEAVY=1 only
    for (const C& c : g3) {
        try {
            BirchCellOptions opt;
            opt.hecke_primes = {2, 3};
            opt.iota_through_pi = true;
            const auto cellp = std::make_shared<const BirchConeCell>(RunBirchConeCell(c.n, c.ell, 2, opt));
            std::cout << "  BIRCH_G3 n=" << c.n << " ell=" << c.ell << " m=2 Hc_exps=" << Join(cellp->exps) << " H1_exps=" << Join(cellp->bulk_exps)
                      << " seconds=" << cellp->seconds << std::endl;
            GateG3(check, *cellp);
            if (c.n == 40) GateSignRule(check, cellp);
        } catch (const std::exception& e) {
            check(false, "(BS-G3) " + CellName(c.n, c.ell, 2) + " threw: " + e.what());
        }
    }
    try {
        GateFrozenGeneric(check);
    } catch (const std::exception& e) {
        check(false, std::string("(BS-G4) frozen arm threw: ") + e.what());
    }
    try {
        GateLegC(check);
    } catch (const std::exception& e) {
        check(false, std::string("(BS-G4) leg (c) arm threw: ") + e.what());
    }
    try {
        BirchCellOptions opt;
        opt.pi_matrix = true;
        const BirchConeCell cell = RunBirchConeCell(258, 37, 1, opt);
        (void)GateG1G2(check, cell, 43, false);
        GateCell258(check, cell);
    } catch (const std::exception& e) {
        check(false, std::string("(BS-Q8) (258,37) threw: ") + e.what());
    }
    if (heavy) {
        try {
            BirchCellOptions opt;
            opt.pi_matrix = true;
            opt.iota_through_pi = true;
            const BirchConeCell cell = RunBirchConeCell(258, 37, 2, opt);
            (void)GateG1G2(check, cell, 43, false);
            GateG3(check, cell);
            GateCell258(check, cell);
        } catch (const std::exception& e) {
            check(false, std::string("(BS-Q8) (258,37) Z/37^2 threw: ") + e.what());
        }
    }
    try {
        GateApi(check);
    } catch (const std::exception& e) {
        check(false, std::string("(BS-API) threw: ") + e.what());
    }
    std::cout << "  BIRCH_ARMS seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << (heavy ? " (with HEAVY)" : "") << std::endl;
}

} // namespace lvalue
} // namespace hecke_ext
