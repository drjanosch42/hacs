// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 21, P5 (b), (c)) HeckeLValueIwasawaInvariantsValidation -- see HeckeLValueIwasawaInvariantsValidation.h.
// Spec docs/implementation/slot21/P5_spec.md (v0.1, predictions committed before this code ran).  Every expected value below is the
// spec's derivation, LIT, or an earlier frozen record (MeasureFrozenR201.inc, MeasureFrozenS2.inc), never this code's output.
#include "HeckeLValueIwasawaInvariantsValidation.h"

#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "ModularSymbolMeasure.h"

#include "../../../ModArithmetic/EngineCheck.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {
namespace {

using namespace birch;
using u128 = unsigned __int128;

#include "MeasureFrozenR201.inc"
#include "MeasureFrozenS2.inc"
[[maybe_unused]] const auto kFrozenPolePointsUnused = &FrozenPolePoints;   // (M4d) data, not used by P5

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double PeakRssGb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
}

struct Log {
    uint32_t checks = 0, failing = 0;
    std::map<std::string, uint32_t> per_arm;
    void check(const std::string& arm, bool ok, const std::string& what) {
        ++checks;
        ++per_arm[arm];
        if (!ok) {
            ++failing;
            std::cout << "  FAIL (" << arm << ") " << what << std::endl;
        }
    }
};

uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}
uint64_t MulMod(uint64_t a, uint64_t b, uint64_t q) { return static_cast<uint64_t>(static_cast<u128>(a) * b % q); }
// v_ell(x mod ell^cap), cap for x == 0 mod ell^cap
uint32_t ValCap(uint64_t x, uint64_t ell, uint32_t cap) {
    x %= PowU(ell, cap);
    if (x == 0) return cap;
    uint32_t v = 0;
    while (x % ell == 0) {
        x /= ell;
        ++v;
    }
    return v;
}
uint32_t Mod(int64_t x, uint64_t m) { return static_cast<uint32_t>(((x % static_cast<int64_t>(m)) + static_cast<int64_t>(m)) % static_cast<int64_t>(m)); }
std::string Tag(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + "," + std::to_string(ell) + ")"; }

// ---- the values oracle (spec 1.5): Newton divided differences with tracked precision --------------------------------------
struct PV {
    uint64_t x = 0;   // mod ell^P
    uint32_t P = 0;   // 0 = nothing known
};
std::vector<PV> DividedDiffs(const std::vector<uint64_t>& T, const std::vector<uint64_t>& V, uint64_t ell, uint32_t R, uint32_t& nonint) {
    const size_t N = T.size();
    const uint64_t q = PowU(ell, R);
    std::vector<PV> cur(N), out;
    for (size_t s = 0; s < N; ++s) cur[s] = PV{V[s] % q, R};
    if (N == 0) return out;
    out.push_back(cur[0]);
    for (size_t t = 1; t < N; ++t) {
        std::vector<PV> nx(N - t);
        for (size_t s = 0; s + t < N; ++s) {
            const uint32_t P = std::min(cur[s].P, cur[s + 1].P);
            if (P == 0) continue;
            const uint64_t qP = PowU(ell, P);
            const uint64_t num = (cur[s + 1].x % qP + qP - cur[s].x % qP) % qP;
            const uint64_t den = (T[s + t] + q - T[s]) % q;
            const uint32_t v = ValCap(den, ell, R);
            if (v >= P) continue;
            if (num != 0 && ValCap(num, ell, P) < v) {
                ++nonint;   // a divided difference of an integral power series at integral nodes is integral
                continue;
            }
            const uint32_t P2 = P - v;
            const uint64_t qP2 = PowU(ell, P2), ev = PowU(ell, v);
            nx[s] = PV{MulMod((num / ev) % qP2, InvModU64((den / ev) % qP2, qP2), qP2), P2};
        }
        out.push_back(nx[0]);
        cur.swap(nx);
    }
    return out;
}
// valuation of one entry: exact (>= 0), -1 = zero to its precision P, -2 = unknown
int EntryVal(const PV& e, uint64_t ell) {
    if (e.P == 0) return -2;
    if (e.x % PowU(ell, e.P) == 0) return -1;
    return static_cast<int>(ValCap(e.x, ell, e.P));
}
struct Reading {
    bool certain = false;
    uint32_t mu = 0, lambda = 0;
};
// spec 1.5: certain iff the least exact valuation m is 0, or m <= lb (a known lower bound for mu)
Reading ReadOracle(const std::vector<PV>& dd, uint64_t ell, uint32_t lb) {
    Reading r;
    int m = -1;
    for (const PV& e : dd) {
        const int v = EntryVal(e, ell);
        if (v >= 0 && (m < 0 || v < m)) m = v;
    }
    if (m < 0) return r;
    if (m > 0 && static_cast<int>(lb) < m) return r;
    for (size_t t = 0; t < dd.size(); ++t) {
        const int v = EntryVal(dd[t], ell);
        if (v == m) {
            r.certain = true;
            r.mu = static_cast<uint32_t>(m);
            r.lambda = static_cast<uint32_t>(t);
            return r;
        }
        const bool above = (v > m) || (v == -1 && static_cast<int>(dd[t].P) > m);
        if (!above) return r;
    }
    return r;
}
// the delta_t consistent with a claimed (mu0, lambda0) (spec 1.5); `informative` counts t = lambda0 seen exactly
bool Consistent(const std::vector<PV>& dd, uint32_t mu0, uint32_t lam0, uint64_t ell, uint32_t& informative) {
    for (size_t t = 0; t < dd.size(); ++t) {
        const int v = EntryVal(dd[t], ell);
        if (v == -2) continue;
        if (t < lam0) {
            if (v >= 0 && static_cast<uint32_t>(v) < mu0 + 1) return false;
        } else if (t == lam0) {
            if (v >= 0) {
                if (static_cast<uint32_t>(v) != mu0) return false;
                ++informative;
            } else if (dd[t].P > mu0) {
                return false;
            }
        } else {
            if (v >= 0 && static_cast<uint32_t>(v) < mu0) return false;
        }
    }
    return true;
}

// ---- group-ring invariants (as ModularSymbolMeasure::invariants) ------------------------------------------------------------
struct GrInv {
    bool readable = false;
    uint32_t mu = 0, lambda = 0;
};
GrInv GroupRingInvariants(const std::vector<uint64_t>& g, uint64_t ell, uint32_t R, uint32_t M) {
    const uint64_t q = PowU(ell, R), G = g.size(), tmax = std::min<uint64_t>(G, ell);
    std::vector<uint64_t> col(G, 1 % q);
    GrInv I;
    uint32_t best = R;
    uint64_t lam = 0;
    bool any = false;
    for (uint64_t t = 0; t < tmax; ++t) {
        if (t > 0) {
            std::vector<uint64_t> nc(G, 0);
            for (uint64_t i = 1; i < G; ++i) nc[i] = (nc[i - 1] + col[i - 1]) % q;
            col.swap(nc);
        }
        u128 s = 0;
        for (uint64_t i = 0; i < G; ++i)
            if (g[i] && col[i]) s = (s + static_cast<u128>(g[i]) * col[i]) % q;
        const uint32_t v = ValCap(static_cast<uint64_t>(s), ell, R);
        if (v < R) any = true;
        if (v < best) {
            best = v;
            lam = t;
        }
    }
    I.mu = best;
    I.lambda = static_cast<uint32_t>(lam);
    I.readable = any && best < std::min<uint32_t>(R, M - 1);
    return I;
}

// ---- (I0) Kubota-Leopoldt tables ---------------------------------------------------------------------------------------------
const std::map<uint64_t, std::set<uint32_t>>& IrregularPairs() {
    // STD-recalled (spec 1.6); re-derived at run time by route (ii)
    static const std::map<uint64_t, std::set<uint32_t>> m = {{37, {32}}, {59, {44}}, {67, {58}}, {103, {24}}, {131, {22}}, {691, {12, 200}}};
    return m;
}
struct KlTable {
    uint64_t ell = 0;
    std::map<uint32_t, uint32_t> mu, lambda;   // route (ii), b even
    bool ok = false;
};
KlTable RunKlTable(uint64_t ell, Log& log) {
    const double t0 = Now();
    KlTable K;
    K.ell = ell;
    const uint32_t R = ell > 300 ? 3 : 4, L = R - 1;
    const uint64_t q = PowU(ell, R), u = 1 + ell;
    const uint32_t kmax = static_cast<uint32_t>((ell - 3) + L * (ell - 1));
    const BernoulliOverKTable BT(ell, R, kmax);
    const std::set<uint32_t> irr = IrregularPairs().count(ell) ? IrregularPairs().at(ell) : std::set<uint32_t>{};
    uint32_t classes = 0, uncertain = 0, nonint = 0, route_agree = 0, route_read = 0, mismatch_pred = 0, mu_bad = 0;
    std::ostringstream lam1;
    for (uint32_t b = 2; b <= ell - 3; b += 2) {
        ++classes;
        std::vector<uint64_t> T, V;
        for (uint32_t s = 0; s <= L; ++s) {
            const uint32_t r = b + s * static_cast<uint32_t>(ell - 1);
            T.push_back((PowModU64(u % q, r, q) + q - 1) % q);
            const uint64_t euler = (1 + q - PowModU64(ell % q, r - 1, q)) % q;
            V.push_back((q - MulMod(euler, BT.residue(r), q)) % q);
        }
        const std::vector<PV> dd = DividedDiffs(T, V, ell, R, nonint);
        const Reading rd = ReadOracle(dd, ell, 0);
        if (!rd.certain) {
            ++uncertain;
            continue;
        }
        K.mu[b] = rd.mu;
        K.lambda[b] = rd.lambda;
        if (rd.mu != 0) ++mu_bad;
        if (rd.lambda != (irr.count(b) ? 1u : 0u) || rd.mu != 0) ++mismatch_pred;
        if (rd.lambda > 0) lam1 << b << ":" << rd.lambda << " ";
        // route (i): the group ring at level 2
        const GrInv gi = GroupRingInvariants(KlGroupRingBernoulli(b, ell, 2, R), ell, R, 2);
        route_read += gi.readable;
        route_agree += gi.readable && gi.mu == rd.mu && gi.lambda == rd.lambda;
    }
    const std::string tg = "ell=" + std::to_string(ell);
    log.check("I0", uncertain == 0 && nonint == 0, tg + ": Bernoulli route uncertain at " + std::to_string(uncertain) + ", non-integral " + std::to_string(nonint));
    log.check("I0", route_read == classes && route_agree == classes,
              tg + ": group ring == Bernoulli at " + std::to_string(route_agree) + " / " + std::to_string(classes) + " (readable " + std::to_string(route_read) + ")");
    log.check("I0", mu_bad == 0, tg + ": mu(G_b) != 0 at " + std::to_string(mu_bad) + " classes");
    log.check("I0", mismatch_pred == 0, tg + ": (mu, lambda)(G_b) != (0, [b irregular]) at " + std::to_string(mismatch_pred) + " classes");
    std::cout << "IWA_KL ell=" << ell << " R=" << R << " classes=" << classes << " irregular=" << irr.size() << " lambda>0: "
              << (lam1.str().empty() ? "none " : lam1.str()) << "route_agree=" << route_agree << "/" << classes << " uncertain=" << uncertain
              << " secs=" << (Now() - t0) << std::endl;
    K.ok = uncertain == 0;
    return K;
}

// ---- one cell ---------------------------------------------------------------------------------------------------------------
struct CellSpec {
    uint32_t n = 0;
    uint64_t ell = 0;
    uint32_t d = 0, R = 4, M = 2;
    bool control = true;
};
struct Entry {
    bool det = false;
    uint32_t mu = 0, lambda = 0;
    std::string src;
};

void RunCell(const CellSpec& sp, const std::map<uint64_t, KlTable>& kl, Log& log) {
    const double t0 = Now();
    const uint32_t n = sp.n, k = n + 2;
    const uint64_t ell = sp.ell, lm1 = ell - 1;
    const std::string tag = Tag(n, ell);
    const std::vector<std::pair<std::string, std::map<uint64_t, uint64_t>>> seps = {
        {"T2+3T3+7T5", {{2, 1}, {3, 3}, {5, 7}}}, {"T3", {{3, 1}}}, {"T2", {{2, 1}}}};
    SymbolCell C;
    std::string sepname;
    for (const auto& s : seps) {
        C = RunSymbolCell(n, ell, sp.R, s.second, sp.control);
        sepname = s.first;
        if (C.simple && C.have_r_plus) break;
    }
    log.check("I3", C.simple && C.have_r_plus, tag + ": the symbol cell did not run: " + C.refusal);
    if (!C.simple || !C.have_r_plus) return;
    const Ring& Rg = C.ring;
    const uint64_t q = Rg.q;
    const SymRho rho(n, Rg);
    const uint64_t a_ell = C.a_p.at(ell);
    const uint64_t alpha = UnitRoot(Rg, a_ell, k), ia = Rg.inv(alpha);
    uint32_t dcell = 0;
    if (!sp.control) {
        dcell = Rg.val(Rg.sub(C.a_p.at(2), Rg.add(1 % q, Rg.pow(2, k - 1))));
        log.check("I4", C.eisenstein_congruent && dcell == sp.d, tag + ": Eisenstein-congruent with d = " + std::to_string(dcell) + " (predicted " + std::to_string(sp.d) + ")");
    }
    const double t1 = Now();
    const ModularSymbolMeasure mm = BuildMeasure(rho, C.r_minus, a_ell, alpha, sp.M, true);
    const ModularSymbolMeasure mp = BuildMeasure(rho, C.r_plus, a_ell, alpha, sp.M, true);
    const double t2 = Now();

    // ---- the twisted functionals ev_i = EvAllJ(omega^i) (chibar = omega^{-i}) ----
    auto AllEv = [&](const Vec& P) {
        std::vector<Vec> ev(lm1);
        for (uint32_t i = 0; i < lm1; ++i) ev[i] = EvAllJ(rho, P, TeichmullerCharacter(i, Rg));
        return ev;
    };
    // a generic element of W (mixed parity, not an eigenvector)
    Vec Pgen(n + 1, 0);
    for (size_t i = 0; i < C.Wminus.size(); ++i)
        for (uint32_t t = 0; t <= n; ++t) Pgen[t] = Rg.add(Pgen[t], Rg.mul((i + 1) % q, C.Wminus[i][t]));
    for (size_t i = 0; i < C.Wplus.size(); ++i)
        for (uint32_t t = 0; t <= n; ++t) Pgen[t] = Rg.add(Pgen[t], Rg.mul((i + 2) % q, C.Wplus[i][t]));
    const bool pgen_in_w = InW(rho, Pgen);
    std::map<int, std::vector<Vec>> evs;   // -1: r^-, +1: r^+, 0: generic
    evs[-1] = AllEv(C.r_minus);
    evs[+1] = AllEv(C.r_plus);
    evs[0] = AllEv(Pgen);
    const double t3 = Now();

    // ---- (I1) the tame functional equation (spec 1.1) ----
    {
        uint32_t fe_bad = 0, fe_tot = 0, neg_flip = 0, neg_flip_tot = 0, par_bad = 0, c1_bad = 0;
        uint32_t nc_mid_minus = 0, nc_mid_minus_tot = 0, nc_mid_plus = 0, nc_mid_plus_tot = 0, nc_else = 0, nc_else_tot = 0;
        uint32_t nc_low = 0;
        double nc_expect = 0;
        std::ostringstream nc_list;
        for (int which : {-1, +1, 0}) {
            const std::vector<Vec>& ev = evs.at(which);
            const Vec& P = which < 0 ? C.r_minus : (which > 0 ? C.r_plus : Pgen);
            // conductor 1
            const Vec e1 = EvAllJ(rho, P, TrivialCharacter(Rg));
            for (uint32_t j = 0; j <= n; ++j) {
                if (e1[j] != P[n - j]) ++c1_bad;
                const uint64_t rhs = (j % 2 == 0) ? Rg.neg(e1[n - j]) : e1[n - j];
                ++fe_tot;
                if (e1[j] != rhs) ++fe_bad;
            }
            // conductor ell
            for (uint32_t i = 0; i < lm1; ++i) {
                const uint32_t ib = static_cast<uint32_t>((lm1 - i) % lm1);
                const bool real = (2 * i) % lm1 == 0;
                for (uint32_t j = 0; j <= n; ++j) {
                    const uint64_t other = ev[ib][n - j];
                    const uint64_t rhs = ((i + j) % 2 == 0) ? Rg.neg(other) : other;   // -chi(-1)(-1)^j lambda_{chibar,n-j}
                    ++fe_tot;
                    if (ev[i][j] != rhs) ++fe_bad;
                    if (ev[i][j] != 0 || rhs != 0) {
                        ++neg_flip_tot;
                        if (ev[i][j] == Rg.neg(rhs)) ++neg_flip;
                    }
                    if (which != 0) {
                        const int eps = which;   // r^-: -1, r^+: +1
                        const int par = ((i + j) % 2 == 0) ? 1 : -1;
                        if (par != eps && ev[i][j] != 0) ++par_bad;
                        if (!real) {
                            const uint64_t wrong = ((i + j) % 2 == 0) ? Rg.neg(ev[i][n - j]) : ev[i][n - j];   // chi for chibar
                            // a comparison of two values of equal valuation v has power ell^{R-v} only (spec v0.2, Sec. 8 deviation 2):
                            // pairs below 10^5 are counted as low_power and left out of the negative control (never out of the FE gate);
                            // nc_expect = sum of ell^{-(R-v)} over the counted equal-valuation pairs, the expected number of accidental matches
                            const uint32_t v1 = Rg.val(ev[i][j]), v2 = Rg.val(wrong);
                            const double power = static_cast<double>(PowU(ell, Rg.m - std::min(v1, v2)));
                            const bool low = v1 == v2 && power < 1e5;
                            if (!low && v1 == v2 && (ev[i][j] != 0 || wrong != 0) && !(which < 0 && 2 * j == n)) nc_expect += 1.0 / power;
                            if (low && (ev[i][j] != 0 || wrong != 0) && !(which < 0 && 2 * j == n)) {
                                ++nc_low;
                            } else if (ev[i][j] != 0 || wrong != 0) {
                                const bool match = ev[i][j] == wrong;
                                if (2 * j == n) {
                                    if (which < 0) {
                                        ++nc_mid_minus_tot;
                                        nc_mid_minus += match;
                                    } else {
                                        ++nc_mid_plus_tot;
                                        nc_mid_plus += match;
                                    }
                                } else {
                                    ++nc_else_tot;
                                    nc_else += match;
                                    if (match)
                                        nc_list << (which < 0 ? "-" : "+") << "i" << i << "j" << j << "v" << Rg.val(ev[i][j]) << "," << Rg.val(wrong) << " ";
                                }
                            }
                        }
                    }
                }
            }
        }
        log.check("I1", pgen_in_w, tag + ": the generic vector is not in W");
        log.check("I1", c1_bad == 0, tag + ": EvAllJ(trivial) != P_{n-j} at " + std::to_string(c1_bad));
        log.check("I1", fe_bad == 0, tag + ": tame FE fails at " + std::to_string(fe_bad) + " / " + std::to_string(fe_tot));
        log.check("I1", par_bad == 0, tag + ": parity vanishing fails at " + std::to_string(par_bad));
        log.check("I1", neg_flip == 0 && neg_flip_tot > 0, tag + ": NEG flipped sign matches at " + std::to_string(neg_flip) + " / " + std::to_string(neg_flip_tot));
        log.check("I1", nc_mid_minus == nc_mid_minus_tot && nc_mid_plus == 0 && nc_else == 0 && nc_else_tot > 0,
                  tag + ": NEG chi for chibar: j=n/2 r^- " + std::to_string(nc_mid_minus) + "/" + std::to_string(nc_mid_minus_tot) + ", j=n/2 r^+ " +
                      std::to_string(nc_mid_plus) + "/" + std::to_string(nc_mid_plus_tot) + ", elsewhere " + std::to_string(nc_else) + "/" +
                      std::to_string(nc_else_tot));
        std::cout << "IWA_FE cell=" << tag << " R=" << sp.R << " vectors=3 fe=" << fe_bad << "/" << fe_tot << " parity=" << par_bad << " cond1=" << c1_bad
                  << " neg_flip=" << neg_flip << "/" << neg_flip_tot << " neg_conj: mid- " << nc_mid_minus << "/" << nc_mid_minus_tot << " mid+ " << nc_mid_plus
                  << "/" << nc_mid_plus_tot << " else " << nc_else << "/" << nc_else_tot << " low_power=" << nc_low << " expected_accidental=" << nc_expect << " " << nc_list.str() << std::endl;
    }

    // ---- per sign: (I2) the branch FE; (I3) the table ----
    const uint64_t prec = PowU(ell, std::min(sp.M, sp.R));
    const uint64_t u = 1 + ell;
    std::map<int, std::vector<Entry>> table;
    for (int sign : {-1, +1}) {
        const ModularSymbolMeasure& me = sign < 0 ? mm : mp;
        const Vec& P = sign < 0 ? C.r_minus : C.r_plus;
        const std::vector<Vec>& ev = evs.at(sign);
        const uint32_t parity = sign < 0 ? 1u : 0u;
        const std::string st = tag + (sign < 0 ? " r^-" : " r^+");
        std::vector<std::vector<uint64_t>> br(lm1);
        for (uint32_t b = 0; b < lm1; ++b) br[b] = me.branch(b);
        // (I2)
        uint32_t gr_bad = 0, gr_tot = 0, neg_match = 0, neg_tot = 0;
        for (uint32_t b = parity; b < lm1; b += 2) {
            const uint32_t pb = Mod(static_cast<int64_t>(n) - b, lm1);
            const std::vector<uint64_t> tw = GrTwInv(br[pb], ell, q, n + 1);
            bool ok = true, neg = true, nonzero = false;
            for (size_t i = 0; i < tw.size(); ++i) {
                const uint64_t rhs = (b % 2 == 0) ? Rg.neg(tw[i]) : tw[i];   // -(-1)^b
                if ((br[b][i] + q - rhs) % prec != 0) ok = false;
                if ((br[b][i] + q - Rg.neg(rhs)) % prec != 0) neg = false;
                if (br[b][i] % prec != 0) nonzero = true;
            }
            ++gr_tot;
            gr_bad += !ok;
            if (nonzero) {
                ++neg_tot;
                neg_match += neg;
            }
        }
        // every branch == 0 mod ell^{min(M,R)} (mu >= min(M,R), the plus side of a d = 2 cell at M = 2): the comparison is vacuous, recorded
        const bool vacuous = neg_tot == 0;
        if (!vacuous) {
            log.check("I2", gr_bad == 0, st + ": branch FE fails at " + std::to_string(gr_bad) + " / " + std::to_string(gr_tot));
            log.check("I2", neg_match == 0, st + ": NEG flipped branch sign matches at " + std::to_string(neg_match) + " / " + std::to_string(neg_tot));
        }
        std::cout << "IWA_GRFE cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " M=" << sp.M << " branches=" << gr_tot << " fail=" << gr_bad
                  << " neg=" << neg_match << "/" << neg_tot << (vacuous ? " vacuous" : "") << std::endl;

        // (I3) the table: invariants() and the values oracle
        std::vector<uint32_t> nodes;
        for (uint32_t j = 0; j <= n && nodes.size() < sp.R; ++j)
            if (Rg.val(rho.binom(n, j)) == 0) nodes.push_back(j);
        std::vector<uint64_t> T;
        for (uint32_t j : nodes) T.push_back((Rg.pow(u % q, j) + q - 1) % q);
        const uint32_t lb = (!sp.control && sign > 0) ? sp.d : 0;
        auto& tb = table[sign];
        tb.assign(lm1, Entry{});
        uint32_t nonint = 0, incons = 0, informative = 0, readable = 0, oracle_only = 0, undetermined = 0, both_certain_bad = 0;
        for (uint32_t b = parity; b < lm1; b += 2) {
            std::vector<uint64_t> V;
            for (uint32_t j : nodes) {
                const uint32_t i = Mod(static_cast<int64_t>(j) - b, lm1);
                const uint64_t E = (i == 0) ? Rg.mul(Rg.sub(1 % q, Rg.mul(Rg.pow(ell, j), ia)), Rg.sub(1 % q, Rg.mul(Rg.pow(ell, n - j), ia))) : ia;
                const uint64_t evij = (i == 0) ? P[n - j] : ev[i][j];
                uint64_t val = Rg.mul(Rg.mul(E, evij), Rg.inv(rho.binom(n, j)));
                if (j % 2 == 1) val = Rg.neg(val);
                V.push_back(val);
            }
            const std::vector<PV> dd = DividedDiffs(T, V, ell, sp.R, nonint);
            const Reading rd = ReadOracle(dd, ell, lb);
            const IwasawaInvariants I = me.invariants(b);
            Entry& e = tb[b];
            if (I.readable) {
                ++readable;
                if (!Consistent(dd, I.mu, I.lambda, ell, informative)) ++incons;
                if (rd.certain && (rd.mu != I.mu || rd.lambda != I.lambda)) ++both_certain_bad;
                e = Entry{true, I.mu, I.lambda, rd.certain ? "iv" : "i"};
            } else if (rd.certain) {
                ++oracle_only;
                e = Entry{true, rd.mu, rd.lambda, "v"};
            } else {
                ++undetermined;
            }
        }
        const uint32_t nbr = static_cast<uint32_t>(lm1 / 2);
        log.check("I3", nonint == 0, st + ": non-integral divided differences " + std::to_string(nonint));
        log.check("I3", incons == 0 && both_certain_bad == 0,
                  st + ": values oracle inconsistent with invariants() at " + std::to_string(incons) + " / " + std::to_string(readable) + " (certain and different " +
                      std::to_string(both_certain_bad) + ")");
        if (sp.control) log.check("I3", undetermined == 0, st + ": undetermined branches at a control " + std::to_string(undetermined));
        // (FE-inv), (FE-parity), the near-trivial zero, mu = 0 at the controls
        uint32_t inv_bad = 0, inv_tot = 0, sd_bad = 0, sd_tot = 0, mu_bad = 0;
        std::ostringstream sdl;
        for (uint32_t b = parity; b < lm1; b += 2) {
            if (!tb[b].det) continue;
            const uint32_t pb = Mod(static_cast<int64_t>(n) - b, lm1);
            if (tb[pb].det) {
                ++inv_tot;
                if (tb[pb].mu != tb[b].mu || tb[pb].lambda != tb[b].lambda) ++inv_bad;
            }
            if (pb == b) {
                ++sd_tot;
                sdl << b << ":" << tb[b].lambda << " ";
                if (tb[b].lambda % 2 != (b % 2 == 0 ? 1u : 0u)) ++sd_bad;
            }
            if (sp.control && tb[b].mu != 0) ++mu_bad;
        }
        log.check("I3", inv_bad == 0, st + ": (mu,lambda)_b != (mu,lambda)_{n-b} at " + std::to_string(inv_bad) + " / " + std::to_string(inv_tot));
        log.check("I3", sd_bad == 0, st + ": self-dual parity fails at " + std::to_string(sd_bad) + " / " + std::to_string(sd_tot) + " (" + sdl.str() + ")");
        if (sp.control) log.check("I3", mu_bad == 0, st + ": mu != 0 at " + std::to_string(mu_bad) + " branches of a control");
        std::string ntz = "-";
        if (sign > 0 && alpha % ell == 1) {
            uint32_t bad = 0, tot = 0;
            for (uint32_t b : {0u, static_cast<uint32_t>(n % lm1)}) {
                if (!tb[b].det) continue;
                ++tot;
                if (tb[b].mu == 0 && tb[b].lambda == 0) ++bad;
            }
            log.check("I3", bad == 0 && tot > 0, st + ": near-trivial zero (alpha == 1 mod ell): branch 0 / n has (0,0) at " + std::to_string(bad) + " / " + std::to_string(tot));
            ntz = std::to_string(tot - bad) + "/" + std::to_string(tot);
        }
        std::ostringstream lst;
        uint32_t lampos = 0;
        for (uint32_t b = parity; b < lm1; b += 2) {
            const Entry& e = tb[b];
            if (!e.det) {
                lst << b << ":? ";
                continue;
            }
            if (e.lambda > 0) ++lampos;
            if (e.mu != lb || e.lambda != 0) lst << b << ":(" << e.mu << "," << e.lambda << ")" << e.src << " ";
        }
        std::cout << "IWA_TABLE cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " d=" << dcell << " R=" << sp.R << " M=" << sp.M << " sep=" << sepname
                  << " alpha_mod_ell=" << (alpha % ell) << " branches=" << nbr << " readable=" << readable << " oracle_only=" << oracle_only
                  << " undetermined=" << undetermined << " oracle_informative=" << informative << " lambda>0=" << lampos << " selfdual=" << sdl.str()
                  << "near_trivial_zero=" << ntz << " default=(" << lb << ",0) other: " << (lst.str().empty() ? "none" : lst.str()) << std::endl;
    }

    // ---- (I4) level-1 Greenberg-Vatsal at the Eisenstein cells (spec 1.7) ----
    if (!sp.control) {
        const auto kit = kl.find(ell);
        log.check("I4", kit != kl.end() && kit->second.ok, tag + ": no Kubota-Leopoldt table at ell = " + std::to_string(ell));
        if (kit != kl.end()) {
            const KlTable& K = kit->second;
            uint32_t bad = 0, tot = 0, undet = 0, plus_bad = 0, plus_tot = 0;
            std::ostringstream pole, lam1;
            for (uint32_t a = 1; a < lm1; a += 2) {
                const uint32_t b1 = Mod(static_cast<int64_t>(a) + 1, lm1), b2 = Mod(static_cast<int64_t>(k) - 1 - a, lm1);
                const Entry& e = table[-1][a];
                if (b1 == 0 || b2 == 0) {
                    pole << a << ":" << (e.det ? "(" + std::to_string(e.mu) + "," + std::to_string(e.lambda) + ")" : std::string("?")) << " ";
                    continue;
                }
                ++tot;
                if (!e.det) {
                    ++undet;
                    continue;
                }
                const uint32_t want = K.lambda.at(b1) + K.lambda.at(b2);
                if (want > 0) lam1 << a << ":" << want << " ";
                if (e.mu != 0 || e.lambda != want) {
                    ++bad;
                    std::cout << "  IWA_GV mismatch cell=" << tag << " a=" << a << " (" << e.mu << "," << e.lambda << ") predicted (0," << want << ")" << std::endl;
                }
            }
            for (uint32_t b = 0; b < lm1; b += 2) {
                const Entry& e = table[+1][b];
                if (!e.det) continue;
                ++plus_tot;
                if (e.mu != sp.d) ++plus_bad;
            }
            log.check("I4", bad == 0 && undet == 0, tag + ": GV lambda^-_a = lambda(G_{a+1}) + lambda(G_{k-1-a}) fails at " + std::to_string(bad) + " / " +
                                                         std::to_string(tot) + " (undetermined " + std::to_string(undet) + ")");
            log.check("I4", plus_bad == 0 && plus_tot > 0, tag + ": mu^+ != d at " + std::to_string(plus_bad) + " / " + std::to_string(plus_tot));
            std::cout << "IWA_GV cell=" << tag << " d=" << dcell << " regular_odd=" << tot << " fail=" << bad << " predicted_lambda>0: "
                      << (lam1.str().empty() ? "none " : lam1.str()) << "pole_branches(record): " << pole.str() << "mu+=d at " << (plus_tot - plus_bad) << "/"
                      << plus_tot << std::endl;
        }
    }

    // ---- (I5) the frozen records ----
    for (int sign : {-1, +1}) {
        const FrozenInvariants* fr = nullptr;
        for (const auto& f : FrozenInvariantTables())
            if (f.n == n && f.ell == ell && f.sign == sign) fr = &f;
        for (const auto& f : FrozenInvariantTablesS2())
            if (f.n == n && f.ell == ell && f.sign == sign) fr = &f;
        if (!fr) continue;
        uint32_t bad = 0, cmp = 0, skipped = 0;
        std::ostringstream bl;
        for (uint32_t b = (sign < 0 ? 1u : 0u); b < lm1; b += 2) {
            uint32_t emu = fr->def_mu, elam = fr->def_lambda;
            for (const auto& x : fr->except)
                if (x.branch == b) {
                    emu = x.mu;
                    elam = x.lambda;
                }
            const Entry& e = table[sign][b];
            if (!e.det) {
                ++skipped;
                continue;
            }
            ++cmp;
            if (e.mu != emu || e.lambda != elam) {
                ++bad;
                bl << b << ":(" << e.mu << "," << e.lambda << ")!=(" << emu << "," << elam << ") ";
            }
        }
        log.check("I5", bad == 0 && cmp > 0, tag + (sign < 0 ? " r^-" : " r^+") + ": frozen table mismatches " + std::to_string(bad) + " / " + std::to_string(cmp) + " " + bl.str());
        std::cout << "IWA_FROZEN cell=" << tag << " sign=" << (sign < 0 ? "-" : "+") << " compared=" << cmp << " mismatch=" << bad << " skipped=" << skipped << std::endl;
    }
    std::cout << "IWA_CELL cell=" << tag << " symbol_secs=" << (t1 - t0) << " measure_secs=" << (t2 - t1) << " ev_secs=" << (t3 - t2)
              << " secs=" << (Now() - t0) << std::endl;
}

} // namespace
} // namespace lvalue
} // namespace hecke_ext

uint32_t HeckeLValueIwasawaInvariantsValidation() {
    using namespace hecke_ext::lvalue;
    const double T0 = Now();
    Log log;
    const char* hv = std::getenv("HECKE_LVALUE_IWASAWA_HEAVY");
    const bool heavy = hv && std::string(hv) == "1";
    std::vector<uint64_t> kl_ells = {11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 59, 67, 103, 131};
    std::vector<CellSpec> cells;
    for (uint64_t ell : std::vector<uint64_t>{11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 59, 67, 103, 131}) cells.push_back({10, ell, 0, 5, 2, true});
    cells.push_back({16, 37, 0, 5, 2, true});
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{{30, 37}, {66, 37}, {42, 59}, {56, 67}, {22, 103}, {20, 131}})
        cells.push_back({c.first, c.second, 1, c.first == 66 ? 5u : 4u, 3, false});   // (66,37) at R = 5: spec v0.2, Sec. 8 deviation 1
    if (heavy) {
        kl_ells.push_back(691);
        cells.push_back({282, 37, 2, 4, 2, false});
        cells.push_back({226, 103, 2, 4, 2, false});
        cells.push_back({10, 691, 1, 3, 2, false});
    }
    std::cout << "EFFECTIVE HeckeLValueIwasawaInvariantsValidation arm=" << (heavy ? "HEAVY" : "DEFAULT") << " cells=" << cells.size() << std::endl;
    try {
        std::map<uint64_t, KlTable> kl;
        for (uint64_t ell : kl_ells) kl[ell] = RunKlTable(ell, log);
        for (const CellSpec& sp : cells) RunCell(sp, kl, log);
    } catch (const std::exception& e) {
        log.check("EXC", false, std::string("exception: ") + e.what());
    }
    std::ostringstream arms;
    for (const auto& kv : log.per_arm) arms << kv.first << "=" << kv.second << " ";
    std::cout << "P5_ARMS checks=" << log.checks << " failing=" << log.failing << " per_arm: " << arms.str() << "secs=" << (Now() - T0)
              << " peak_rss_gb=" << PeakRssGb() << std::endl;
    return log.failing == 0 ? 0u : 1u;
}
