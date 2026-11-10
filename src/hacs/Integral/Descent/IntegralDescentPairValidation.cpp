// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D4: IntegralDescentPairValidation / IntegralDescentPairSlowValidation.  See the header for the arms.
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.27), cited by label (sec:descentpair, prop:descentpair, prop:descentpairhecke, the
// code map's (a)-(h)).  The oracle d4_pair_oracle.tsv shares the pair code of cd7_pair_check.py and nothing with cohomology_app;
// its HECKE rows are parsed from the referee's cd7referee_check.out.
#include "IntegralDescentPairValidation.h"

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentMaps.h"
#include "DescentOracle.h"
#include "DescentPair.h"
#include "DescentTreePair.h"

#include "../Model/IntegralHecke.h"
#include "../Model/IsotropyModelEll.h"
#include "../Model/StrictLift.h"
#include "../../Hecke/ExactInt.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace P = integral_descent::pair;
using integral_descent::pair::ConeRank;
using integral_descent::AbelianGroupType;
using integral_descent::maps::Mat;
using integral_descent::maps::Pres;
using integral_descent::maps::ZMod;
using U32Ctx = engine::ModCtx<uint32_t>;
using engine::DenseBuf;
using clk = std::chrono::steady_clock;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 80) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}
std::string mib(uint64_t b) { return std::to_string(b >> 20); }
std::string cellname(uint32_t N, uint32_t n) { return "(N,n)=(" + std::to_string(N) + "," + std::to_string(n) + ")"; }

// ---- the oracle -------------------------------------------------------------------------------------------------------
using Cell = std::array<uint32_t, 2>;
struct Oracle {
    bool found = false;
    std::string error;
    std::map<Cell, std::map<std::pair<std::string, uint32_t>, AbelianGroupType>> rows;
    std::map<std::array<uint32_t, 3>, std::vector<int64_t>> hecke;   // (N, n, p) -> char T_p on H^1_c, high -> low
};
std::string oracle_path() {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/docs/implementation/slot24/d4_pair_oracle.tsv";
#else
    return "docs/implementation/slot24/d4_pair_oracle.tsv";
#endif
}
Oracle LoadOracle() {
    Oracle T;
    std::ifstream f(oracle_path());
    if (!f) {
        T.error = "cannot open " + oracle_path();
        return T;
    }
    T.found = true;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("N\t", 0) == 0) continue;
        std::istringstream is(line);
        if (line.rfind("HECKE\t", 0) == 0) {
            std::string tag, coeffs;
            uint32_t N = 0, n = 0, p = 0;
            if (!(is >> tag >> N >> n >> p >> coeffs)) {
                T.error = "malformed HECKE row: " + line;
                return T;
            }
            std::vector<int64_t> c;
            for (char& ch : coeffs)
                if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
            std::istringstream cs(coeffs);
            int64_t v = 0;
            while (cs >> v) c.push_back(v);
            T.hecke[{N, n, p}] = c;
            continue;
        }
        Cell c{};
        std::string obj, g;
        uint32_t q = 0;
        if (!(is >> c[0] >> c[1] >> obj >> q >> g)) {
            T.error = "malformed row: " + line;
            return T;
        }
        T.rows[c][{obj, q}] = integral_descent::ParseOracleGroup(g);
    }
    return T;
}

const AbelianGroupType* lookup(const P::PairResult& R, const std::string& obj, uint32_t q) {
    auto mt = [&](const std::vector<integral_descent::maps::MapTypes>& v, const std::string& part) -> const AbelianGroupType* {
        if (q >= v.size()) return nullptr;
        if (part == "ker") return &v[q].ker;
        if (part == "im") return &v[q].im;
        return &v[q].coker;
    };
    if (obj == "H") return q < R.H.size() ? &R.H[q] : nullptr;
    if (obj == "Hd") return q < R.Hd.size() ? &R.Hd[q] : nullptr;
    if (obj == "Hc") return q < R.Hc.size() ? &R.Hc[q] : nullptr;
    const size_t u = obj.find('_');
    if (u == std::string::npos) return nullptr;
    const std::string m = obj.substr(0, u), part = obj.substr(u + 1);
    if (m == "res") return mt(R.res, part);
    if (m == "iota") return mt(R.j, part);
    if (m == "proj") return mt(R.proj, part);
    return nullptr;
}

std::string groups_line(const P::PairResult& R) {
    std::ostringstream o;
    o << "  " << cellname(R.N, R.n) << " D=" << R.D << " c=" << R.bt.c << " S={";
    for (size_t i = 0; i < R.primes.size(); ++i) o << (i ? "," : "") << R.primes[i];
    o << "} t=" << fx(R.t_total, 3) << "s  H_c:";
    for (size_t q = 1; q < R.Hc.size(); ++q) o << " " << R.Hc[q].to_string() << (q + 1 < R.Hc.size() ? " |" : "");
    return o.str();
}

uint32_t g_exact = 0, g_cells = 0, g_rows = 0, g_inner = 0, g_rank = 0, g_pieces = 0;

bool same_result(const P::PairResult& A, const P::PairResult& B) {
    if (A.refused || B.refused) return false;
    if (A.H != B.H || A.Hd != B.Hd || A.Hc != B.Hc) return false;
    for (size_t q = 0; q < A.res.size(); ++q)
        if (A.res[q].ker != B.res[q].ker || A.res[q].im != B.res[q].im || A.res[q].coker != B.res[q].coker) return false;
    for (size_t q = 0; q < A.j.size(); ++q)
        if (A.j[q].ker != B.j[q].ker || A.j[q].im != B.j[q].im || A.j[q].coker != B.j[q].coker) return false;
    for (size_t q = 0; q < A.proj.size(); ++q)
        if (A.proj[q].ker != B.proj[q].ker || A.proj[q].im != B.proj[q].im || A.proj[q].coker != B.proj[q].coker) return false;
    return true;
}

// (T), (E), (B), (5) on one oracle cell
void compare_cell(const Oracle& T, const Cell& c, bool print) {
    const P::PairResult R = P::AssembleDescentPair(c[0], c[1]);
    if (!check(!R.refused, "(T) " + cellname(c[0], c[1]) + " not refused: " + R.refusal)) return;
    ++g_cells;
    g_exact += R.exact_spots;
    g_inner += R.checks;
    for (const auto& pc : R.pieces) {
        g_rank += pc.rank_checks;
        ++g_pieces;
        check(pc.route == "E8pair" || pc.rank_checks > 0, "(5) the cone ranks were also read mod ell' (code map (c)) at " +
                                                              cellname(c[0], c[1]) + " ell=" + std::to_string(pc.ell));
    }
    const auto it = T.rows.find(c);
    if (!check(it != T.rows.end(), "(T) " + cellname(c[0], c[1]) + " in the oracle")) return;
    for (const auto& [k, g] : it->second) {
        const AbelianGroupType* x = lookup(R, k.first, k.second);
        ++g_rows;
        check(x && *x == g, "(T) " + cellname(c[0], c[1]) + " " + k.first + "^" + std::to_string(k.second) + ": code " +
                                (x ? x->to_string() : std::string("(missing)")) + " == oracle " + g.to_string());
    }
    if (print) std::cout << groups_line(R) << std::endl;
}

// ---- small matrix helpers -----------------------------------------------------------------------------------------------
Mat Tr(const Mat& A) {
    Mat B(A.c, A.r);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) B.at(j, i) = A.at(i, j);
    return B;
}
bool Eq(const ZMod& Z, const Mat& A, const Mat& B) {
    if (A.r != B.r || A.c != B.c) return false;
    for (size_t i = 0; i < A.a.size(); ++i)
        if (Z.red(A.a[i]) != Z.red(B.a[i])) return false;
    return true;
}
Mat Diag2(const Mat& A, const Mat& B) {
    Mat C(A.r + B.r, A.c + B.c);
    for (uint32_t i = 0; i < A.r; ++i) std::copy(A.row(i), A.row(i) + A.c, C.row(i));
    for (uint32_t i = 0; i < B.r; ++i) std::copy(B.row(i), B.row(i) + B.c, C.row(A.r + i) + A.c);
    return C;
}
Mat SubMat(const ZMod& Z, const Mat& A, const Mat& B) {
    Mat C = A;
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] = Z.sub(Z.red(A.a[i]), Z.red(B.a[i]));
    return C;
}
// the free block of an induced matrix (rows / columns t .. t + free - 1)
Mat FreeBlock(const Mat& F, const Pres& X, const Pres& Y) {
    const uint32_t tx = static_cast<uint32_t>(X.tors.size()), ty = static_cast<uint32_t>(Y.tors.size());
    Mat B(Y.free, X.free);
    for (uint32_t i = 0; i < Y.free; ++i)
        for (uint32_t j = 0; j < X.free; ++j) B.at(i, j) = F.at(ty + i, tx + j);
    return B;
}
// char poly (high -> low, monic) of a square matrix mod Z.m
std::vector<uint32_t> Charpoly(const ZMod& Z, const Mat& A) {
    const U32Ctx ctx = Z.k >= 2 ? U32Ctx::prime_power(Z.ell, Z.k) : U32Ctx(Z.ell);
    DenseBuf<uint32_t> D(A.r, A.c);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) D.at(i, j) = Z.red(A.at(i, j));
    std::vector<uint32_t> lo = integral_model::CharpolyDivisionFree(ctx, D);
    std::reverse(lo.begin(), lo.end());
    return lo;
}
std::vector<uint32_t> ReducePoly(const ZMod& Z, const std::vector<int64_t>& c) {
    std::vector<uint32_t> o;
    for (int64_t v : c) {
        int64_t r = v % static_cast<int64_t>(Z.m);
        if (r < 0) r += static_cast<int64_t>(Z.m);
        o.push_back(static_cast<uint32_t>(r));
    }
    return o;
}
// (x - a)^e, high -> low, mod Z.m
std::vector<uint32_t> PowerLinear(const ZMod& Z, uint64_t a, uint32_t e) {
    std::vector<uint32_t> p{1};
    const uint32_t am = static_cast<uint32_t>(a % Z.m);
    for (uint32_t i = 0; i < e; ++i) {
        std::vector<uint32_t> np(p.size() + 1, 0);
        for (size_t k = 0; k < p.size(); ++k) {
            np[k] = Z.add(np[k], p[k]);
            np[k + 1] = Z.sub(np[k + 1], Z.mul(p[k], am));
        }
        p = np;
    }
    return p;
}
uint64_t PowMod(uint64_t b, uint32_t e, uint64_t m) {
    uint64_t r = 1 % m;
    b %= m;
    for (uint32_t i = 0; i < e; ++i) r = r * b % m;
    return r;
}
std::string PolyStr(const std::vector<uint32_t>& p) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < p.size(); ++i) o << (i ? "," : "") << p[i];
    o << "]";
    return o.str();
}
Mat ConeMap(const Mat& bulk_next, const Mat& bdry) { return Diag2(bulk_next, bdry); }

// ===========================================================================
// (H) Hecke on the pair
// ===========================================================================
// The cusp form spanning S_{n+2}(Gamma_0(N)) on the Hecke cells, as an eta product (audit m1; the auditor's d4audit_hecke.py method):
// (1, 10) Delta = eta(z)^24, (2, 6) (eta(z) eta(2z))^8, (3, 4) (eta(z) eta(3z))^6, (11, 0) (eta(z) eta(11z))^2, all q * prod_m
// (1 - q^{d m})^e; (1, 12) and (13, 0) carry no cusp form.  a_p = the coefficient of q^p, EXACT integers (int64, checked).
bool EtaCell(uint32_t N, uint32_t n, std::vector<std::pair<uint32_t, uint32_t>>& factors) {
    if (N == 1 && n == 10) factors = {{1, 24}};
    else if (N == 2 && n == 6) factors = {{1, 8}, {2, 8}};
    else if (N == 3 && n == 4) factors = {{1, 6}, {3, 6}};
    else if (N == 11 && n == 0) factors = {{1, 2}, {11, 2}};
    else return false;
    return true;
}
int64_t EtaAp(uint32_t N, uint32_t n, uint32_t p) {
    std::vector<std::pair<uint32_t, uint32_t>> f;
    ENGINE_CHECK_MSG(EtaCell(N, n, f), "EtaAp: not an eta-product cell");
    const uint32_t prec = p;   // the coefficient of q^{p-1} of prod_m (1 - q^{d m})^e
    std::vector<int64_t> c(prec, 0);
    c[0] = 1;
    for (const auto& [d, e] : f)
        for (uint32_t m = 1; d * m < prec; ++m)
            for (uint32_t r = 0; r < e; ++r)
                for (uint32_t k = prec - 1; k >= d * m; --k) {
                    ENGINE_CHECK_MSG(c[k - d * m] < (int64_t(1) << 61) && c[k - d * m] > -(int64_t(1) << 61) && c[k] < (int64_t(1) << 61) &&
                                         c[k] > -(int64_t(1) << 61),
                                     "EtaAp: int64");
                    c[k] -= c[k - d * m];
                }
    return c[p - 1];
}
// The expected char T_p on H^1_c (= on H^1, Eichler--Shimura), high -> low, as a RESIDUE mod Z.m of the integer polynomial
// (x - a_p)^2 (x - (1 + p^{n+1}))^e, e = c(N) for n > 0, c(N) - 1 for n = 0 (the Eisenstein part); a_p exact (EtaAp), the product
// formed mod Z.m.  `oracle_ok` = false when the oracle (the referee's row) has (N, n, p) and disagrees with it.
std::vector<uint32_t> ExpectedHc1(const Oracle& T, uint32_t N, uint32_t n, uint32_t p, const ZMod& Z, bool& oracle_ok) {
    auto mul = [&](const std::vector<uint32_t>& A, int64_t r) {   // A * (x - r) mod Z.m
        int64_t rr = r % static_cast<int64_t>(Z.m);
        if (rr < 0) rr += static_cast<int64_t>(Z.m);
        std::vector<uint32_t> B(A.size() + 1, 0);
        for (size_t k = 0; k < A.size(); ++k) {
            B[k] = Z.add(B[k], A[k]);
            B[k + 1] = Z.sub(B[k + 1], Z.mul(A[k], static_cast<uint32_t>(rr)));
        }
        return B;
    };
    const P::PairBetti bt = P::PairBettiOf(N, n);
    const uint32_t e = n > 0 ? bt.c : bt.c - 1;
    const int64_t eis = static_cast<int64_t>((1 + PowMod(p, n + 1, Z.m)) % Z.m);
    std::vector<uint32_t> want{1};
    std::vector<std::pair<uint32_t, uint32_t>> f;
    if (EtaCell(N, n, f)) {
        const int64_t ap = EtaAp(N, n, p);
        want = mul(mul(want, ap), ap);
    }
    for (uint32_t i = 0; i < e; ++i) want = mul(want, eis);
    oracle_ok = true;
    const auto it = T.hecke.find({N, n, p});
    if (it != T.hecke.end()) oracle_ok = ReducePoly(Z, it->second) == want;
    return want;
}

// F_Y G - G F_X lies in the relations of Y: the square X -> Y commutes on cohomology (G induced by g : X -> Y).
bool Commutes(const Pres& X, const Pres& Y, const Mat& g, const Mat& tX, const Mat& tY) {
    ZMod za, zb, zc;
    const Mat G = P::InducedMatrix(X, Y, g, za);
    const Mat FX = P::InducedMatrix(X, X, tX, zb);
    const Mat FY = P::InducedMatrix(Y, Y, tY, zc);
    ZMod z0 = za;
    for (const ZMod* zz : {&zb, &zc})
        if (zz->k < z0.k) z0 = *zz;
    std::string why;
    return integral_descent::maps::IdentityOnCohomology(Y, SubMat(z0, P::MatMulZ(z0, FY, G), P::MatMulZ(z0, G, FX)), why);
}

struct HeckeCounts {
    uint32_t ops = 0, beyond5 = 0, polys = 0, polys_eta = 0, oracle_rows = 0, squares = 0, commute = 0, torsion_cells = 0;
    uint64_t strict_entries = 0;
};

// T_p on the free pair at (N, n) over ctx, cochains to degree Tq = 5 (q <= 4 on cohomology, the pair's q_max): the strict identity,
// cochain maps, theta_hat at N = 1, the equivariance squares of the sequence on cohomology in every degree q <= 3 (iota^q, pi^{q+1},
// res^{q+1}; torsion included), T_p T_p' = T_p' T_p on H^1_c and H^2_c; with `polys` the char polys of H^1_c, H^1 (vs the eta
// product), H^0(d), H^1(d) (vs 1 + p^{n+1}).
void HeckeCell(const Oracle& T, uint32_t N, uint32_t n, const U32Ctx& ctx, const std::vector<uint32_t>& primes, bool polys,
               bool want_torsion, HeckeCounts& K) {
    const uint32_t Tq = 5, qm = 4;
    const P::PairBetti bt = P::PairBettiOf(N, n);
    const P::PairComplex X = P::BuildFreePairN(N, n, ctx, Tq);
    P::PairLocal L;
    P::PieceReport rep;
    rep.min_prec = 1000;
    const std::string at = cellname(N, n) + " mod " + std::to_string(ctx.ell) + "^" + std::to_string(ctx.k);
    if (!check(P::RunPairLes(X, bt, qm, L, rep), "(H) the free pair's sequence at " + at + ": " + rep.why)) return;
    if (want_torsion) {
        if (check(!L.Hc[2].tors.empty(), "(H) tors H^2_c (x) Z_(" + std::to_string(ctx.ell) + ") != 0 at " + at + " (the arm is not vacuous)"))
            ++K.torsion_cells;
    }
    std::map<uint32_t, std::array<std::pair<ZMod, Mat>, 2>> Fc;   // p -> T_p on H^1_c, H^2_c (with their precision)
    for (uint32_t p : primes) {
        if (N % p == 0) continue;
        const P::ConeHecke H = P::BuildConeHecke(N, n, p, ctx, Tq);
        const std::string atp = at + " p=" + std::to_string(p);
        if (!check(!H.refused, "(H) T_p built " + atp + ": " + H.refusal)) continue;
        ++K.ops;
        if (p > 5) ++K.beyond5;
        check(H.strict_identity && H.bdry_closed && H.bdry_dropped == 0 && H.bdry_terms > 0,
              "(H) rres T~_Iso = T~_Isob rres entrywise (T~_Isob assembled separately), tau_Iso(Isob) in Isob " + atp);
        K.strict_entries += H.entries_checked;
        std::vector<Mat> Tb, Tbd;
        for (uint32_t q = 0; q <= Tq; ++q) {
            Tb.push_back(P::FromDenseBuf(H.bulk[q]));
            Tbd.push_back(P::FromDenseBuf(H.bdry[q]));
        }
        bool cm = true;
        for (uint32_t q = 0; q < Tq; ++q) {
            cm = cm && Eq(X.Z, P::MatMulZ(X.Z, X.d[q], Tb[q]), P::MatMulZ(X.Z, Tb[q + 1], X.d[q]));
            cm = cm && Eq(X.Z, P::MatMulZ(X.Z, X.db[q], Tbd[q]), P::MatMulZ(X.Z, Tbd[q + 1], X.db[q]));
        }
        for (uint32_t q = 0; q <= Tq; ++q) cm = cm && Eq(X.Z, P::MatMulZ(X.Z, X.r[q], Tb[q]), P::MatMulZ(X.Z, Tbd[q], X.r[q]));
        for (int j = -1; j + 2 <= static_cast<int>(Tq); ++j) {
            const Mat Dj = P::ConeDifferential(X, j);
            const uint32_t a = static_cast<uint32_t>(j + 1);
            const Mat Phi0 = j >= 0 ? ConeMap(Tb[a], Tbd[static_cast<uint32_t>(j)]) : Tb[a];
            const Mat Phi1 = ConeMap(Tb[a + 1], Tbd[a]);
            cm = cm && Eq(X.Z, P::MatMulZ(X.Z, Dj, Phi0), P::MatMulZ(X.Z, Phi1, Dj));
        }
        check(cm, "(H) T~_Iso, T~_Isob, Phi_p = (T~[1], T~_d) cochain maps and r T~_Iso = T~_Isob r (the pair's r), degrees <= 5 " + atp);
        if (N == 1) {   // prop:strictliftN(3)(d): at N = 1, T~ is theta_hat
            bool same = true;
            for (uint32_t q = 0; q <= Tq; ++q) same = same && Eq(X.Z, Tb[q], P::FromDenseBuf(P::PairLift(p).theta_hat(ctx, n, q)));
            check(same, "(H) at N = 1 the induced-fibre operator is the coset transport theta_hat (prop:strictliftN(3)(d)) " + atp);
        }
        // the sequence is T_p-equivariant ON COHOMOLOGY, every degree: iota^q : H^q(d) -> H^{q+1}_c, pi^{q+1} : H^{q+1}_c -> H^{q+1},
        // res^{q+1} : H^{q+1} -> H^{q+1}(d), q = 0 .. 3 (torsion included); res^0 as well
        check(Commutes(L.H[0], L.Hd[0], X.r[0], Tb[0], Tbd[0]), "(H) res^0 T_p = T_p res^0 " + atp);
        ++K.squares;
        for (uint32_t q = 0; q + 1 <= qm; ++q) {
            const uint32_t cq = ConeRank(X, static_cast<int>(q));   // Cone^q = C^{q+1} (+) C_d^q = H^{q+1}_c's cochains
            Mat jq(cq, X.cb[q]);
            for (uint32_t i = 0; i < X.cb[q]; ++i) jq.at(X.c[q + 1] + i, i) = 1;
            Mat piq(X.c[q + 1], cq);
            for (uint32_t i = 0; i < X.c[q + 1]; ++i) piq.at(i, i) = 1;
            const Mat Phi = ConeMap(Tb[q + 1], Tbd[q]);
            const std::string d = std::to_string(q), d1 = std::to_string(q + 1);
            check(Commutes(L.Hd[q], L.Hc[q + 1], jq, Tbd[q], Phi), "(H) iota^" + d + " T_p = T_p iota^" + d + " on H^" + d + "(d) -> H^" + d1 + "_c " + atp);
            check(Commutes(L.Hc[q + 1], L.H[q + 1], piq, Phi, Tb[q + 1]), "(H) pi^" + d1 + " T_p = T_p pi^" + d1 + " on H^" + d1 + "_c -> H^" + d1 + " " + atp);
            check(Commutes(L.H[q + 1], L.Hd[q + 1], X.r[q + 1], Tb[q + 1], Tbd[q + 1]),
                  "(H) res^" + d1 + " T_p = T_p res^" + d1 + " on H^" + d1 + " -> H^" + d1 + "(d) " + atp);
            K.squares += 3;
        }
        ZMod Z1, Z2, Z3, Z4, Z5;
        const Mat F1c = P::InducedMatrix(L.Hc[1], L.Hc[1], ConeMap(Tb[1], Tbd[0]), Z1);
        const Mat F2c = P::InducedMatrix(L.Hc[2], L.Hc[2], ConeMap(Tb[2], Tbd[1]), Z5);
        Fc[p] = {std::make_pair(Z1, F1c), std::make_pair(Z5, F2c)};
        if (!polys) continue;
        const Mat F1 = P::InducedMatrix(L.H[1], L.H[1], Tb[1], Z2);
        const Mat Fd0 = P::InducedMatrix(L.Hd[0], L.Hd[0], Tbd[0], Z3);
        const Mat Fd1 = P::InducedMatrix(L.Hd[1], L.Hd[1], Tbd[1], Z4);
        const std::vector<uint32_t> got1 = Charpoly(Z1, FreeBlock(F1c, L.Hc[1], L.Hc[1]));
        const std::vector<uint32_t> got2 = Charpoly(Z2, FreeBlock(F1, L.H[1], L.H[1]));
        bool or1 = true, or2 = true;
        const std::vector<uint32_t> want1 = ExpectedHc1(T, N, n, p, Z1, or1), want2 = ExpectedHc1(T, N, n, p, Z2, or2);
        check(got1 == want1, "(H) char T_p on H^1_c = the eta-product value " + atp + " " + PolyStr(got1) + " vs " + PolyStr(want1));
        check(got2 == want2, "(H) char T_p on H^1 (free part) = the eta-product value " + atp + " " + PolyStr(got2) + " vs " + PolyStr(want2));
        K.polys += 2;
        std::vector<std::pair<uint32_t, uint32_t>> f;
        if (EtaCell(N, n, f)) K.polys_eta += 2;
        if (T.hecke.count({N, n, p})) {
            check(or1 && or2, "(H) the oracle's (the referee's) char T_p = the eta-product value " + atp);
            ++K.oracle_rows;
        }
        if (ctx.ell == 10007 && (N == 11 || n == 10) && p >= 11)
            std::cout << "  (H) " << cellname(N, n) << " p=" << p << " char T_p on H^1_c = H^1 = " << PolyStr(got1) << " mod 10007^2 (eta: a_p = "
                      << EtaAp(N, n, p) << ")" << std::endl;
        const uint64_t eis = (1 + PowMod(p, n + 1, Z3.m)) % Z3.m;
        check(Charpoly(Z3, FreeBlock(Fd0, L.Hd[0], L.Hd[0])) == PowerLinear(Z3, eis, bt.bd0),
              "(H) T_p on H^0(d) has char (x - (1 + p^{n+1}))^{c(N)} " + atp);
        check(Charpoly(Z4, FreeBlock(Fd1, L.Hd[1], L.Hd[1])) == PowerLinear(Z4, (1 + PowMod(p, n + 1, Z4.m)) % Z4.m, bt.bd1),
              "(H) T_p on H^1(d) has char (x - (1 + p^{n+1}))^{c(N)} " + atp);
        K.polys += 2;
    }
    // T_p T_p' = T_p' T_p on H^1_c and on H^2_c (torsion included: the commutator vanishes on cohomology)
    for (const auto& [p1, F1] : Fc)
        for (const auto& [p2, F2] : Fc)
            if (p1 < p2)
                for (uint32_t k = 0; k < 2; ++k) {
                    const Pres& Y = L.Hc[k + 1];
                    const ZMod Z = F1[k].first.k <= F2[k].first.k ? F1[k].first : F2[k].first;
                    const Mat& A = F1[k].second;
                    const Mat& B = F2[k].second;
                    std::string why;
                    check(integral_descent::maps::IdentityOnCohomology(Y, SubMat(Z, P::MatMulZ(Z, A, B), P::MatMulZ(Z, B, A)), why),
                          "(H) T_" + std::to_string(p1) + " T_" + std::to_string(p2) + " = T_" + std::to_string(p2) + " T_" + std::to_string(p1) +
                              " on H^" + std::to_string(k + 1) + "_c " + at + ": " + why);
                    ++K.commute;
                }
}

void ArmHecke(const Oracle& T) {
    struct HC {
        uint32_t N, n;
    };
    const std::vector<HC> cells = {{1, 10}, {1, 12}, {11, 0}, {2, 6}, {13, 0}, {3, 4}};
    const std::vector<uint32_t> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23};
    HeckeCounts K;
    for (const HC& hc : cells)
        for (const U32Ctx& ctx : {U32Ctx::prime_power(10007, 2), U32Ctx::prime_power(2, 30), U32Ctx::prime_power(3, 19)})
            HeckeCell(T, hc.N, hc.n, ctx, primes, true, false, K);
    std::cout << "  (H) " << K.ops << " operators T_p (p <= 23, p !| N; " << K.beyond5 << " with p > 5, lifts built on demand) on the free "
              << "pair at (1,10), (1,12), (11,0), (2,6), (13,0), (3,4), cochains to degree 5, mod 10007^2, 2^30, 3^19; " << K.polys
              << " characteristic polynomials compared AS RESIDUES mod those moduli (CRT modulus ~1.2e26: an integer identity only where "
              << "every coefficient is below half of it, e.g. not at (2,6) p = 23) -- " << K.polys_eta << " on H^1_c, H^1 against the "
              << "EXACT INTEGER a_p of the eta products (Delta, (eta eta_2)^8, (eta eta_3)^6, (eta eta_11)^2; Eisenstein 1 + p^{n+1}; " << K.oracle_rows
              << " oracle rows agree), the rest on H^0(d), H^1(d) vs 1 + p^{n+1}; " << K.squares << " equivariance squares on cohomology "
              << "(degrees <= 4, torsion included), " << K.commute << " commutators T_p T_p' on H^1_c, H^2_c; " << K.strict_entries
              << " strict-identity entries" << std::endl;
    // T_p on ell >= 5 torsion (audit m1(b)): tors H^2_c (x) Z_(ell) != 0; no reference value -- the equivariance squares (iota, pi,
    // res in every degree, torsion included) and T_p T_p' = T_p' T_p on H^2_c over Z/ell^K
    HeckeCounts KT;
    struct TC {
        uint32_t N, n, ell;
    };
    for (const TC& tc : std::vector<TC>{{1, 8, 5}, {1, 12, 5}, {1, 12, 7}, {5, 4, 5}})
        HeckeCell(T, tc.N, tc.n, U32Ctx::prime_power(tc.ell, integral_model::DefaultKBudget(tc.ell)), {2, 3, 5, 7, 11, 13}, false, true, KT);
    std::cout << "  (H) ell >= 5 torsion: " << KT.ops << " operators T_p (p <= 13) at (1,8) mod 5^K, (1,12) mod 5^K, 7^K and (5,4) mod 5^K ("
              << KT.torsion_cells << " cells with tors H^2_c (x) Z_(ell) != 0), " << KT.squares << " equivariance squares on cohomology, "
              << KT.commute << " commutators on H^1_c, H^2_c -- no reference value for the torsion action (none in [C])" << std::endl;
    for (uint32_t p : primes) {
        const P::PairLiftCost& c = P::PairLiftCostOf(p);
        check(c.ok && c.rss_growth <= c.rss_pred, "(S) the pair lift at p=" + std::to_string(p) + " within its RSS prediction (rule 19)");
        std::cout << "DESCENT4_LIFT p=" << p << " ok=" << c.ok << " build_s=" << fx(c.build_seconds) << " certify_s=" << fx(c.certify_seconds)
                  << " rss_pred_mib=" << mib(c.rss_pred) << " rss_growth_mib=" << mib(c.rss_growth) << " terms=" << c.terms
                  << " max_coeff=" << c.max_abs_coefficient << " max_entry=" << c.max_abs_entry << std::endl;
    }
}

// ===========================================================================
// (3) the (PC) arm reproduced on D4's objects (level 1, Iso^(3) vs Iso, mod 3^19)
// ===========================================================================
void ArmPC() {
    const U32Ctx ctx = U32Ctx::prime_power(3, 19);
    const ZMod Z(3, 19);
    const uint32_t Tq = 5, qm = 4;
    const integral_model::LocalModel& freeM = P::FreeIsoModel();
    const integral_model::LocalModel& iso3 = integral_descent::Iso3();
    const integral_model::ComparisonMap Phi = integral_model::StrictComparison(freeM, iso3);
    const integral_model::ComparisonMap Psi = integral_model::StrictComparison(iso3, freeM);
    uint32_t cells = 0, ident = 0;
    for (uint32_t n : {0u, 2u, 4u, 6u, 8u, 10u, 12u}) {
        const std::string at = " n=" + std::to_string(n);
        const auto Sym = integral_model::SymModule(ctx, n);
        const P::PairComplex X3 = P::BuildModelPair(iso3, ctx, Sym, Tq, true);
        const P::PairComplex Xf = P::BuildFreePair(ctx, Sym, Tq);
        const P::PairBetti bt = P::PairBettiOf(1, n);
        // Phi^*, Psi^* in reduced coordinates on bulk; on the boundary through the restriction (the pair condition checked)
        std::vector<Mat> PhiR, PsiR, PhiB, PsiB;
        bool pairOk = true;
        for (uint32_t q = 0; q <= Tq; ++q) {
            const Mat PhiS = P::FromDenseBuf(integral_model::Pullback(ctx, Sym, q, freeM, iso3, Phi));   // C_3 full -> C_I
            const Mat PsiS = P::FromDenseBuf(integral_model::Pullback(ctx, Sym, q, iso3, freeM, Psi));   // C_I -> C_3 full
            PhiR.push_back(P::MatMulZ(Z, PhiS, X3.emb[q]));
            PsiR.push_back(P::MatMulZ(Z, X3.coords[q], PsiS));
            PhiB.push_back(P::MatMulZ(Z, P::MatMulZ(Z, Xf.r[q], PhiR[q]), Tr(X3.r[q])));
            PsiB.push_back(P::MatMulZ(Z, P::MatMulZ(Z, X3.r[q], PsiR[q]), Tr(Xf.r[q])));
            pairOk = pairOk && Eq(Z, P::MatMulZ(Z, Xf.r[q], PhiR[q]), P::MatMulZ(Z, PhiB[q], X3.r[q])) &&
                     Eq(Z, P::MatMulZ(Z, X3.r[q], PsiR[q]), P::MatMulZ(Z, PsiB[q], Xf.r[q]));
        }
        check(pairOk, "(3) Phi^*, Psi^* map the pair to the pair (res f^* = f^*_d res; boundary first, prop:isomodelell(4))" + at);
        // groups of the two pairs equal (with the maps)
        P::PairLocal L3, Lf;
        P::PieceReport r3, rf;
        r3.min_prec = rf.min_prec = 1000;
        if (!check(P::RunPairLes(X3, bt, qm, L3, r3), "(3) Iso^(3) pair" + at + ": " + r3.why)) continue;
        if (!check(P::RunPairLes(Xf, bt, qm, Lf, rf), "(3) Iso pair" + at + ": " + rf.why)) continue;
        bool same = true;
        for (uint32_t q = 0; q <= qm; ++q)
            same = same && L3.Hc[q].type() == Lf.Hc[q].type() && L3.Hd[q].type() == Lf.Hd[q].type() && L3.H[q].type() == Lf.H[q].type();
        check(same, "(3) H^q_c, H^q(d), H^q of Iso^(3)'s pair = Iso's, q <= 4" + at);
        // Phi^* induces isomorphisms on H_c and H(d)
        for (uint32_t q = 1; q <= qm; ++q) {
            const Mat PhiC = q >= 1 ? Diag2(PhiR[q], PhiB[q - 1]) : PhiR[q];
            const auto m = integral_descent::maps::InducedMap(L3.Hc[q], Lf.Hc[q], PhiC, L3.Hc[q].free);
            check(m.ok && m.ker.is_zero() && m.coker.is_zero(), "(3) Phi^* is an isomorphism on H^" + std::to_string(q) + "_c" + at);
        }
        for (uint32_t p : {2u, 3u, 5u}) {
            const P::ConeHecke H = P::BuildConeHecke(1, n, p, ctx, Tq);
            if (!check(!H.refused, "(3) T_p built" + at)) continue;
            std::vector<Mat> Tb, Td, T3, T3d;
            for (uint32_t q = 0; q <= Tq; ++q) {
                Tb.push_back(P::FromDenseBuf(H.bulk[q]));
                Td.push_back(P::FromDenseBuf(H.bdry[q]));
            }
            for (uint32_t q = 0; q <= Tq; ++q) {
                T3.push_back(P::MatMulZ(Z, PsiR[q], P::MatMulZ(Z, Tb[q], PhiR[q])));
                T3d.push_back(P::MatMulZ(Z, PsiB[q], P::MatMulZ(Z, Td[q], PhiB[q])));
            }
            // T'_cone a cochain map on Cone(Iso^(3))
            bool cm = true;
            for (int j = -1; j + 2 <= static_cast<int>(Tq); ++j) {
                const uint32_t a = static_cast<uint32_t>(j + 1);
                const Mat Dj = P::ConeDifferential(X3, j);
                const Mat A0 = j >= 0 ? Diag2(T3[a], T3d[static_cast<uint32_t>(j)]) : T3[a];
                const Mat A1 = Diag2(T3[a + 1], T3d[a]);
                cm = cm && Eq(Z, P::MatMulZ(Z, Dj, A0), P::MatMulZ(Z, A1, Dj));
            }
            check(cm, "(3) T'_cone = Psi^*_c T_cone Phi^*_c is a cochain map on Cone(Iso^(3))" + at + " p=" + std::to_string(p));
            // on cohomology: Phi^* T' = T Phi^* on H^q_c and H^q(d), torsion included
            for (uint32_t q = 1; q <= qm; ++q) {
                const Mat PhiC = Diag2(PhiR[q], PhiB[q - 1]);
                const Mat Tc = Diag2(Tb[q], Td[q - 1]);
                const Mat T3c = Diag2(T3[q], T3d[q - 1]);
                ZMod za, zb, zc;
                const Mat Fphi = P::InducedMatrix(L3.Hc[q], Lf.Hc[q], PhiC, za);
                const Mat FT = P::InducedMatrix(Lf.Hc[q], Lf.Hc[q], Tc, zb);
                const Mat FT3 = P::InducedMatrix(L3.Hc[q], L3.Hc[q], T3c, zc);
                ZMod z0 = za;
                for (const ZMod* zz : {&zb, &zc})
                    if (zz->k < z0.k) z0 = *zz;
                std::string why;
                const bool ok = integral_descent::maps::IdentityOnCohomology(
                    Lf.Hc[q], SubMat(z0, P::MatMulZ(z0, FT, Fphi), P::MatMulZ(z0, Fphi, FT3)), why);
                check(ok, "(3) T'_p = T_p on H^" + std::to_string(q) + "_c through Phi^*" + at + " p=" + std::to_string(p) + ": " + why);
                ++ident;
            }
            for (uint32_t q = 0; q <= qm; ++q) {
                ZMod za, zb, zc;
                const Mat Fphi = P::InducedMatrix(L3.Hd[q], Lf.Hd[q], PhiB[q], za);
                const Mat FT = P::InducedMatrix(Lf.Hd[q], Lf.Hd[q], Td[q], zb);
                const Mat FT3 = P::InducedMatrix(L3.Hd[q], L3.Hd[q], T3d[q], zc);
                ZMod z0 = za;
                for (const ZMod* zz : {&zb, &zc})
                    if (zz->k < z0.k) z0 = *zz;
                std::string why;
                check(integral_descent::maps::IdentityOnCohomology(Lf.Hd[q], SubMat(z0, P::MatMulZ(z0, FT, Fphi), P::MatMulZ(z0, Fphi, FT3)), why),
                      "(3) T'_p = T_p on H^" + std::to_string(q) + "(d) through Phi^*" + at + " p=" + std::to_string(p) + ": " + why);
                ++ident;
            }
        }
        if (n == 0 || n == 10) {
            std::ostringstream o;
            o << "  (3) n=" << n << " H_c (3-local, q=1..4):";
            for (uint32_t q = 1; q <= qm; ++q) o << " " << L3.Hc[q].type().to_string();
            std::cout << o.str() << std::endl;
        }
        ++cells;
    }
    std::cout << "  (3) the (PC) arm reproduced: " << cells << " cells n in {0,2,...,12} mod 3^19, " << ident
              << " operator identities T'_p = T_p on H^q_c / H^q(d) (p in {2,3,5}), Iso^(3)'s pair groups = Iso's" << std::endl;
}

// ===========================================================================
// (Q) the E8 pair at level 1
// ===========================================================================
void ArmQ() {
    uint32_t cells = 0;
    for (uint32_t n : {2u, 4u, 6u, 8u, 10u, 12u}) {
        P::PairOptions o;
        const P::PairResult R = P::AssembleDescentPair(1, n, o);   // level 1: ell >= 5 on the E8 pair
        if (!check(!R.refused, "(Q) level 1 n=" + std::to_string(n) + ": " + R.refusal)) continue;
        std::vector<uint32_t> ells;
        for (uint32_t ell : R.primes)
            if (ell >= 5) ells.push_back(ell);
        if (std::find(ells.begin(), ells.end(), 5u) == ells.end()) ells.push_back(5);
        for (uint32_t ell : ells) {
            const uint32_t K = integral_model::DefaultKBudget(ell);
            P::E8Identity id;
            (void)P::BuildE8Pair(n, ell, K, 5, &id);
            const std::string at = " n=" + std::to_string(n) + " ell=" + std::to_string(ell) + " K=" + std::to_string(K);
            check(id.all(), "(Q) eps^* identifies the shipped E8 pair with the Bredon pair (chain maps, restriction, onto)" + at + ":" + id.why);
            // the shipped ModuleCone's invariant factors of H^q_c = (H^q_c (x) Z/ell^K) + Tor(H^{q+1}_c, Z/ell^K)
            for (uint32_t q = 0; q <= 2; ++q) {
                std::vector<uint32_t> want;
                const AbelianGroupType Gq = R.Hc[q].primary_part(ell);
                for (uint32_t i = 0; i < R.Hc[q].free_rank; ++i) want.push_back(K);
                for (const auto& [p, ex] : Gq.torsion)
                    for (uint32_t a : ex) want.push_back(a);
                if (q + 1 < R.Hc.size()) {
                    const AbelianGroupType G1 = R.Hc[q + 1].primary_part(ell);
                    for (const auto& [p, ex] : G1.torsion)
                        for (uint32_t a : ex) want.push_back(a);
                }
                std::vector<uint32_t> got = id.shipped_Hc_invariant_factors[q];
                std::sort(want.begin(), want.end());
                std::sort(got.begin(), got.end());
                check(got == want, "(Q) the shipped cone's H^" + std::to_string(q) + "_c over Z/ell^K = the glued groups by universal coefficients" + at);
            }
        }
        // audit m3: inside the assembly, every E8 piece certified the identity of complexes and ran the ell' rank source
        for (const auto& pc : R.pieces)
            if (pc.ell >= 5)
                check(pc.rank_prime != 0 && pc.rank_checks > 0 && pc.route.find("E8") != std::string::npos,
                      "(Q) the E8 piece at ell=" + std::to_string(pc.ell) + " n=" + std::to_string(n) + " (route " + pc.route +
                          "): identity of complexes and the ell' ranks inside the assembly");
        // (I) the Bredon model at level 1 = the E8 route on every group and map
        o.level1_e8 = false;
        const P::PairResult B = P::AssembleDescentPair(1, n, o);
        check(same_result(R, B), "(I) level 1 n=" + std::to_string(n) + ": the Bredon model LocalModel(desc, ell) = the shipped E8 pair, every group and map");
        ++cells;
    }
    std::cout << "  (Q) level 1, n in {2,...,12}: the Z[1/6]-piece is the shipped E8 item-4b pair (eps^* an identity of complexes onto the "
                 "Bredon cochains; the shipped cone's invariant factors = the glued H_c by universal coefficients); " << cells
              << " cells, the Bredon route identical" << std::endl;
}

// ===========================================================================
// (N) negative controls
// ===========================================================================
void ArmNegative() {
    uint32_t refused = 0;
    auto expect_refused = [&](const P::PairResult& R, const std::string& what) {
        if (check(R.refused, "(N) REFUSED: " + what)) ++refused;
    };
    {
        P::PairOptions o;
        o.test_break_res = true;
        expect_refused(P::AssembleDescentPair(1, 10, o), "a perturbed restriction entry (1,10)");
    }
    {
        P::PairOptions o;
        o.test_rank_offset = 1;
        expect_refused(P::AssembleDescentPair(11, 0, o), "a false rational rank of res^1 (11,0)");
    }
    {
        P::PairOptions o;
        o.test_K_two = 3;
        expect_refused(P::AssembleDescentPair(1, 10, o), "K = 3 at ell = 2 (precision exhausted) (1,10)");
    }
    {
        P::PairOptions o;
        o.rank_prime = 5;
        o.test_skip_prime_rule = true;
        const P::PairResult R = P::AssembleDescentPair(1, 8, o);
        expect_refused(R, "ell' = 5 at (1,8) as the rank source of D^0 (5 | tors H_0 = tors H^2_c, prop:descentpair(6)(c))");
        if (R.refused) std::cout << "  (N) " << R.refusal.substr(0, 200) << std::endl;
    }
    {
        P::PairOptions o;
        o.rank_prime = 5;
        expect_refused(P::AssembleDescentPair(1, 8, o), "ell' = 5 <= n + 1 at (1,8) by rule");
        o.rank_prime = 3;
        expect_refused(P::AssembleDescentPair(1, 2, o), "ell' = 3 by rule (never ell' in {2, 3})");
    }
    {
        P::PairOptions o;
        o.rss_cap_bytes = 1ull << 20;
        expect_refused(P::AssembleDescentPair(2, 2, o), "the RSS ceiling (1 MiB) before anything is built");
    }
    {
        P::PairOptions o;
        o.q_max = 3;
        bool threw = false;
        try {
            (void)P::AssembleDescentPair(1, 0, o);
        } catch (const std::logic_error&) {
            threw = true;
        }
        if (check(threw, "(N) REFUSED: q_max = 3 (code map (g): q_max >= 4 for the pair)")) ++refused;
    }
    {
        // the closed form without the widths (planted fault (g) of cd7_pair_check.py): at (4, 2) the widths are 1, 1, 4
        const std::vector<uint32_t> widths = P::CuspWidths(4);
        const std::vector<uint32_t> ones(widths.size(), 1);
        bool ok1 = false, ok2 = false;
        const AbelianGroupType a = P::BoundaryClosedForm(widths, 2, 2, 1, 30, ok1), b = P::BoundaryClosedForm(ones, 2, 2, 1, 30, ok2);
        if (check(ok1 && ok2 && a != b, "(N) the closed form without the widths differs at (4,2), q=1: " + a.to_string() + " vs " + b.to_string()))
            ++refused;
    }
    const U32Ctx F = U32Ctx::prime_power(10007, 2);
    for (const auto& c : std::vector<std::array<uint32_t, 3>>{{11, 0, 11}, {2, 2, 2}, {1, 10, 4}}) {
        const P::ConeHecke H = P::BuildConeHecke(c[0], c[1], c[2], F, 2);
        const P::PairRefusal want = c[2] == 4 ? P::PairRefusal::NotPrime : P::PairRefusal::PDividesN;
        if (check(H.refused && H.code == want, "(N) REFUSED: T_" + std::to_string(c[2]) + " on the pair at " + cellname(c[0], c[1]) + " (" + H.refusal + ")"))
            ++refused;
    }
    for (uint32_t p : {2u, 7u, 13u}) {   // planted fault: a perturbed pair lift fails its certificate and is never used
        const P::ConeHecke H = P::BuildConeHecke(1, 10, p, F, 2, P::ConeDatum::TestPerturbedLift);
        if (check(H.refused && H.code == P::PairRefusal::LiftCertificate && H.refusal.find("fails its certificate") != std::string::npos,
                  "(N) REFUSED: a perturbed pair lift at p=" + std::to_string(p) + " (" + H.refusal + ")"))
            ++refused;
    }
    {
        const P::ConeHecke H = P::BuildConeHecke(11, 0, 2, F, 2, P::ConeDatum::BulkOnly);
        if (check(H.refused && H.code == P::PairRefusal::BulkOnly, "(N) REFUSED: the bulk-only lift on the cone")) ++refused;
    }
    // C-D7p referee m4: homotopy faults tau + dK + Kd keep (T1)-(T3); (S) / (B) refuse them (the referee's Arm F)
    for (const P::ConeDatum dt : {P::ConeDatum::TestHomotopyIntoIsob, P::ConeDatum::TestHomotopyIntoBulk})
        for (uint32_t p : {2u, 7u}) {
            const P::ConeHecke H = P::BuildConeHecke(1, 10, p, F, 2, dt);
            const std::string what = dt == P::ConeDatum::TestHomotopyIntoIsob ? "into Isob" : "into bulk t = 0";
            // the exact pattern (D4 follow-up, D5 audit round): K into Isob keeps the boundary ((B)-pres holds) but moves eps on the
            // pair and on the boundary ((S), (B)-eps fail); K into the bulk t = 0 cells also leaves the boundary ((B)-pres fails)
            const std::string want = dt == P::ConeDatum::TestHomotopyIntoIsob ? "[T1=1 T2=1 T3=1 S=0 Bpres=1 Beps=0]"
                                                                               : "[T1=1 T2=1 T3=1 S=0 Bpres=0 Beps=0]";
            if (check(H.refused && H.code == P::PairRefusal::LiftCertificate && H.refusal.find(want) != std::string::npos,
                      "(N) REFUSED by exactly " + want + ": a homotopy fault K " + what + " at p=" + std::to_string(p) + " (" + H.refusal + ")"))
                ++refused;
            if (p == 2) std::cout << "  (N) homotopy fault " << what << ": " << H.refusal.substr(H.refusal.rfind('[')) << std::endl;
        }
    {   // C-D7p referee m2: q_top > 6 and q_max > 5 refuse (never throw)
        const P::ConeHecke H = P::BuildConeHecke(1, 10, 2, F, 7);
        if (check(H.refused && H.code == P::PairRefusal::DegreeRange, "(N) REFUSED: q_top = 7 (" + H.refusal + ")")) ++refused;
        P::PairOptions o;
        o.q_max = 6;
        const P::PairResult R = P::AssembleDescentPair(1, 0, o);
        if (check(R.refused && R.refusal.find("[DegreeRange]") != std::string::npos, "(N) REFUSED: q_max = 6 (" + R.refusal + ")")) ++refused;
    }
    {   // C-D7p referee M1: the exception classification -- the named caps refuse, every other exception is a genuine error
        const hecke::HeckeOverflow e1("checked int64 product overflow");
        const std::overflow_error e2("HeckeCosetTables: N = p^r exceeds the uint32_t class-map domain");
        const std::runtime_error e3("HeckeTauTable: certification FAILED (code 3) -- table rejected");
        const std::logic_error e4("x.cpp:1: ENGINE_CHECK failed");
        const std::runtime_error e5("HeckeCosetTables: internal error, s_i has det != 1");
        const std::out_of_range e6("HeckeCosetTables: coset index out of range");
        const hecke::HeckeInternalError e7("CuspKey::reduce_to_infinity: column is not primitive.");   // a HeckeOverflow, not a cap
        bool ok = P::ClassifyPairException(e7) == P::PairRefusal::None && P::ClassifyPairException(e1) == P::PairRefusal::IntegerCap && P::ClassifyPairException(e2) == P::PairRefusal::DomainCap &&
                  P::ClassifyPairException(e3) == P::PairRefusal::InputTableCertificate && P::ClassifyPairException(e4) == P::PairRefusal::None &&
                  P::ClassifyPairException(e5) == P::PairRefusal::None && P::ClassifyPairException(e6) == P::PairRefusal::None;
        if (check(ok, "(N) the refusal classification: HeckeOverflow -> IntegerCap, overflow_error -> DomainCap, a failed input-table "
                      "certification -> InputTableCertificate; logic / internal / range errors and hecke::HeckeInternalError propagate"))
            ++refused;
    }
    // audit m2: a term off Isob planted in tau_Iso(s_1 e_b), the closure test ignored -- the strict identity between the two
    // separately assembled operators refuses it alone (disabling that check fails this control: the audit's F7)
    for (const auto& c : std::vector<std::array<uint32_t, 3>>{{1, 10, 2}, {1, 10, 7}, {11, 0, 13}}) {
        const P::ConeHecke H = P::BuildConeHecke(c[0], c[1], c[2], F, 2, P::ConeDatum::TestOffBoundaryTerm);
        if (check(H.refused && H.code == P::PairRefusal::StrictIdentity && H.refusal.find("NOT detected") == std::string::npos &&
                      H.bdry_dropped > 0,
                  "(N) REFUSED by the strict identity: an off-boundary term in the pair lift at " + cellname(c[0], c[1]) + " p=" +
                      std::to_string(c[2]) + " (" + H.refusal + ")"))
            ++refused;
    }
    {   // audit m4: the ell' rank source switched off is a measurement, never an unrefused result
        P::PairOptions o;
        o.prime_rank_check = false;
        const P::PairResult R = P::AssembleDescentPair(1, 10, o);
        if (check(R.refused && R.refusal.find("prime_rank_check off") != std::string::npos, "(N) REFUSED: prime_rank_check = false at (1,10)"))
            ++refused;
    }
    {   // audit m5: rule 19 for the pair lift -- refused before building above the ceiling
        const uint32_t p = 100003;
        const P::PairLiftCost& c = P::PairLiftCostOf(p);
        if (check(!c.ok && c.code == P::PairRefusal::RssCeiling && c.why.find("rule 19") != std::string::npos && c.terms == 0 && P::PredictPairLiftRss(p) > P::kPairLiftRssCap,
                  "(N) REFUSED: the pair lift at p=100003 (predicted " + mib(c.rss_pred) + " MiB > " + mib(P::kPairLiftRssCap) + " MiB): " + c.why))
            ++refused;
        const P::ConeHecke H = P::BuildConeHecke(1, 10, p, F, 2);
        if (check(H.refused && H.code == P::PairRefusal::RssCeiling, "(N) REFUSED: T_100003 on the pair (" + H.refusal + ")")) ++refused;
    }
    std::cout << "  (N) " << refused << " negative controls refused (perturbed rres, false rank, K = 3 at 2, ell' = 5 | tors H_0, ell' by rule "
                 "x2, RSS ceiling, q_max = 3, closed form without widths, T_11 at level 11 and T_2 at level 2 (p | N), T_4, a perturbed pair lift at "
                 "p = 2, 7, 13, the bulk-only lift, an off-boundary lift term x3 (the strict identity alone), prime_rank_check off, the "
                 "pair lift and T_p at p = 100003 above the lift RSS ceiling, homotopy faults into Isob / bulk x2 by (S)/(B), q_top = 7, q_max = 6, "
                 "the exception classification; every refusal checked for its PairRefusal code)" << std::endl;
}

}  // namespace

uint32_t IntegralDescentPairValidation() {
    g_checks = 0;
    g_fail = 0;
    g_exact = g_cells = g_rows = g_inner = g_rank = g_pieces = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentPairValidation (DESCENT D4: the boundary pair and the cone over Z; [C] sec:descentpair) ===" << std::endl;
    // the D5c audit's N4: which route the 2- and 3-adic pair pieces take in this run (tree = D5c's tree cusp pair, dense = D4's)
    std::cout << "EFFECTIVE IntegralDescentPairValidation route=" << (integral_descent::DescentTreePairEnabled() ? "tree" : "dense")
              << " source=" << integral_descent::DescentTreePairSource() << std::endl;
    const Oracle T = LoadOracle();
    if (check(T.found && T.error.empty(), "(T) the oracle table loads: " + (T.error.empty() ? oracle_path() : T.error))) {
        std::vector<Cell> cells;
        for (const auto& [c, rows] : T.rows) cells.push_back(c);
        for (const Cell& c : cells) compare_cell(T, c, c[0] == 1 || c[1] == 0);
        std::cout << "  (T) " << g_cells << " oracle cells, " << g_rows << " groups / map types compared; (E) " << g_exact
                  << " spots exact as subgroups over " << g_pieces << " local pieces; (B)+(5) " << g_inner
                  << " piece checks (closed form from the widths, H^1_c torsion-free, 48 H^q_c = 0, (5)(e), the boundary free at 2 / "
                  << "the circle model at 3, rres a cochain map), of them " << g_rank << " cone ranks mod ell' = the Betti recursion" << std::endl;
    }
    // (P) periodicity at q_max = 5
    for (const Cell& c : std::vector<Cell>{{1, 0}, {1, 4}, {2, 2}, {3, 0}, {1, 3}}) {
        P::PairOptions o;
        o.q_max = 5;
        const P::PairResult R = P::AssembleDescentPair(c[0], c[1], o);
        if (!check(!R.refused, "(P) q_max = 5 at " + cellname(c[0], c[1]) + ": " + R.refusal)) continue;
        check(R.Hc[5] == R.Hc[3] && R.Hd[4] == R.Hd[2] && R.Hd[5] == R.Hd[3] && R.H[4] == R.H[2] && R.H[5] == R.H[3],
              "(P) periodicity H^5_c = H^3_c, H^{4,5}(d) = H^{2,3}(d), H^{4,5} = H^{2,3} at " + cellname(c[0], c[1]));
    }
    // (I) model independence: the free Iso (model (a)) = Iso^(ell) (model (b))
    for (const Cell& c : std::vector<Cell>{{1, 0}, {1, 2}, {1, 10}, {2, 2}, {11, 0}, {3, 1}}) {
        P::PairOptions o;
        o.route = P::PairRoute::IsoFree;
        const P::PairResult A = P::AssembleDescentPair(c[0], c[1], o);
        const P::PairResult B = P::AssembleDescentPair(c[0], c[1]);
        check(same_result(A, B), "(I) the free Iso pair = the Iso^(ell) pair, every group and map, at " + cellname(c[0], c[1]));
    }
    // (S) the strict datum
    for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u}) {
        const auto& cert = P::PairLiftCertificate(p);
        check(cert.all() && cert.boundary_preserved && cert.boundary_eps,
              "(S) p=" + std::to_string(p) + ": tau_Iso(Isob) in Isob and eps tau^d_Iso = tau_bd eps (" +
                  std::to_string(cert.boundary_checks) + " boundary checks): " + cert.first_failure);
        std::cout << "  (S) p=" << p << ": pair lift certified (T1) " << cert.t1_checks << ", (T2) " << cert.t2_checks << ", (S) "
                  << cert.strict_checks << ", (B) " << cert.boundary_checks << " checks" << std::endl;
    }
    ArmQ();
    ArmPC();
    ArmHecke(T);
    ArmNegative();
    std::cout << "IntegralDescentPairValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0), 3) << " s" << std::endl;
    return g_fail;
}

uint32_t IntegralDescentPairSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentPairSlowValidation (DESCENT D4: the cost of the pair, DESCENT4_COST) ===" << std::endl;
    // D5c: the cost arms measure the design of record, the DENSE pair (Iso^(2), Isob^(2)); the tree cusp pair has its own cost line
    // (IntegralDescentTreePairSlowValidation, DESCENT5c_COST)
    integral_descent::ScopedDescentTreePair dense_route(false);
    // the ladder (DESCENT4_predictions.md): the 2-adic piece on dense Iso^(2) (only_primes = {2}: a measurement, the result is
    // refused by construction) and the boundary alone on Isob.  HECKE_DESCENT_BDRY_CELLS=<N>:<n>[,...] (read-only) replaces it.
    struct LC {
        uint32_t N, n;
    };
    std::vector<LC> ladder = {{4, 2}, {11, 2}, {23, 2}, {47, 2}, {71, 2}, {97, 2}};
    std::string arm = "default";
    try {
        if (const char* e = std::getenv("HECKE_DESCENT_BDRY_CELLS")) {
            ladder.clear();
            std::stringstream ss(e);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                const size_t a = tok.find(':');
                ENGINE_CHECK_MSG(a != std::string::npos, "HECKE_DESCENT_BDRY_CELLS: tokens N:n");
                ladder.push_back({static_cast<uint32_t>(std::stoul(tok.substr(0, a))), static_cast<uint32_t>(std::stoul(tok.substr(a + 1)))});
            }
            arm = "env";
        }
        const uint64_t cap = 40ull << 30;
        std::cout << "EFFECTIVE IntegralDescentPairSlowValidation arm=" << arm << " cells=" << ladder.size() << " rss_cap_gib=" << (cap >> 30)
                  << std::endl;
        for (const LC& c : ladder) {
            const uint32_t D = integral_descent::PsiIndex(c.N) * (c.n + 1);
            std::cout << "  DESCENT4_PREDICT " << cellname(c.N, c.n) << " D=" << D << " rss_pred_mib=" << mib(P::PredictPairRss(2, D, 5))
                      << " cap_gib=" << (cap >> 30) << std::endl;
        }
        for (const LC& c : ladder) {
            P::PairOptions o;
            o.only_primes = {2};
            o.prime_rank_check = false;
            o.rss_cap_bytes = cap;
            const P::PairResult R = P::AssembleDescentPair(c.N, c.n, o);
            if (R.pieces.empty()) {
                check(R.refused, "(C) " + cellname(c.N, c.n) + " refused before building: " + R.refusal);
                std::cout << "DESCENT4_COST N=" << c.N << " n=" << c.n << " D=" << R.D << " route=Iso2 refused pred_rss_mib=" << mib(R.rss_pred)
                          << std::endl;
                continue;
            }
            const P::PieceReport& r = R.pieces[0];
            check(r.ok, "(C) the 2-adic pair piece at " + cellname(c.N, c.n) + ": " + r.why);
            const uint32_t cone = r.c.size() > 2 ? r.c[2] + r.cb[1] : 0;
            std::cout << "DESCENT4_COST N=" << c.N << " n=" << c.n << " D=" << R.D << " route=" << r.route << " K=" << r.K << " T=" << r.T
                      << " c=" << (r.c.size() > 2 ? r.c[2] : 0) << " cb=" << (r.cb.size() > 2 ? r.cb[2] : 0) << " cone=" << cone
                      << " build_s=" << fx(r.build_seconds) << " les_s=" << fx(r.les_seconds) << " total_s=" << fx(r.seconds)
                      << " rss_base_mib=" << mib(r.rss_base) << " rss_peak_mib=" << mib(r.rss_meas) << " growth_mib="
                      << mib(r.rss_meas > r.rss_base ? r.rss_meas - r.rss_base : 0) << " pred_mib=" << mib(r.rss_pred)
                      << " matrices_mib=" << mib(r.matrix_bytes) << std::endl;
            // the boundary alone (Isob^(2): free at 2, ranks 3, 6, 6, ... D -- the full layout IS the reduced one, Q_sigma = 1 on the
            // boundary): LocalModel::boundary_coboundary and its presentations, nothing of the bulk
            {
                const U32Ctx ctx = U32Ctx::prime_power(2, 30);
                const ZMod Z(2, 30);
                const auto tb = clk::now();
                const uint64_t base = integral_descent::CurrentRssBytes();
                integral_descent::RssSampler rs;
                const auto M = integral_descent::InducedSymModule(c.N, c.n, ctx);
                std::vector<Mat> db;
                for (uint32_t q = 0; q <= 4; ++q) db.push_back(P::FromDenseBuf(integral_descent::Iso2().boundary_coboundary(ctx, M, q)));
                const double tbuild = since(tb);
                const P::PairBetti bt = P::PairBettiOf(c.N, c.n);
                std::vector<uint32_t> rdb;
                int64_t prev = 0;
                bool ok = true;
                std::string types;
                const auto ts = clk::now();
                for (uint32_t q = 0; q <= 4; ++q) {
                    const uint32_t cbq = db[q].c;
                    const int64_t x = int64_t(cbq) - int64_t(bt.bd(q)) - prev;
                    rdb.push_back(static_cast<uint32_t>(x));
                    const Pres Hd = integral_descent::maps::BuildPres(Z, q ? db[q - 1] : Mat(cbq, 0), q ? rdb[q - 1] : 0u, db[q], cbq, bt.bd(q));
                    ok = ok && Hd.ok;
                    if (Hd.ok) {
                        bool cok = false;
                        const AbelianGroupType cf = P::BoundaryClosedForm(P::CuspWidths(c.N), c.n, 2, q, 30, cok);
                        AbelianGroupType got = Hd.type();
                        ok = ok && cok && got == cf;
                    }
                    prev = x;
                }
                const double tles = since(ts);
                const uint64_t peak = rs.stop();
                check(ok, "(C) the boundary alone at " + cellname(c.N, c.n) + " (= the closed form)");
                std::cout << "DESCENT4_COST N=" << c.N << " n=" << c.n << " D=" << R.D << " route=Isob2 cb=" << db[2].c << " build_s=" << fx(tbuild)
                          << " les_s=" << fx(tles) << " growth_mib=" << mib(peak > base ? peak - base : 0) << std::endl;
            }
        }
        if (arm == "default") {
            // the pair lift built on demand for larger p (owner scope addition): its build / certify cost, and T_p on the pair at (1, 10)
            // and (11, 0) mod 10007^2 -- char T_p on H^1_c = on H^1, T_p = 1 + p^{n+1} on H^0(d), the strict identity
            for (uint32_t p : {17u, 31u, 61u, 127u}) {
                const auto tl = clk::now();
                const P::PairLiftCost& lc = P::PairLiftCostOf(p);
                check(lc.ok, "(L) the pair lift at p=" + std::to_string(p) + " built and certified: " + lc.why);
                check(lc.rss_growth <= lc.rss_pred, "(L) the pair lift at p=" + std::to_string(p) + " within its RSS prediction (rule 19)");
                std::cout << "DESCENT4_LIFT p=" << p << " ok=" << lc.ok << " build_s=" << fx(lc.build_seconds) << " certify_s="
                          << fx(lc.certify_seconds) << " rss_pred_mib=" << mib(lc.rss_pred) << " rss_growth_mib=" << mib(lc.rss_growth) << " terms=" << lc.terms
                          << " max_coeff=" << lc.max_abs_coefficient << " max_entry=" << lc.max_abs_entry << " wall_s=" << fx(since(tl)) << std::endl;
                if (!lc.ok) continue;
                for (const LC& c : std::vector<LC>{{1, 10}, {11, 0}}) {
                    const U32Ctx ctx = U32Ctx::prime_power(10007, 2);
                    const P::PairComplex X = P::BuildFreePairN(c.N, c.n, ctx, 3);
                    P::PairLocal L;
                    P::PieceReport rep;
                    rep.min_prec = 1000;
                    if (!check(P::RunPairLes(X, P::PairBettiOf(c.N, c.n), 2, L, rep), "(L) the pair at " + cellname(c.N, c.n))) continue;
                    const auto th = clk::now();
                    const P::ConeHecke H = P::BuildConeHecke(c.N, c.n, p, ctx, 3);
                    if (!check(!H.refused && H.strict_identity, "(L) T_p on the pair at p=" + std::to_string(p) + " " + H.refusal)) continue;
                    ZMod z1, z2, z3;
                    const Mat F1c = P::InducedMatrix(L.Hc[1], L.Hc[1], ConeMap(P::FromDenseBuf(H.bulk[1]), P::FromDenseBuf(H.bdry[0])), z1);
                    const Mat F1 = P::InducedMatrix(L.H[1], L.H[1], P::FromDenseBuf(H.bulk[1]), z2);
                    const Mat Fd0 = P::InducedMatrix(L.Hd[0], L.Hd[0], P::FromDenseBuf(H.bdry[0]), z3);
                    const auto a = Charpoly(z1, FreeBlock(F1c, L.Hc[1], L.Hc[1]));
                    const auto b = Charpoly(z2, FreeBlock(F1, L.H[1], L.H[1]));
                    check(a == b, "(L) char T_p on H^1_c = on H^1 at p=" + std::to_string(p) + " " + cellname(c.N, c.n));
                    check(Charpoly(z3, FreeBlock(Fd0, L.Hd[0], L.Hd[0])) ==
                              PowerLinear(z3, (1 + PowMod(p, c.n + 1, z3.m)) % z3.m, P::PairBettiOf(c.N, c.n).bd0),
                          "(L) T_p = 1 + p^{n+1} on H^0(d) at p=" + std::to_string(p) + " " + cellname(c.N, c.n));
                    std::cout << "  (L) p=" << p << " " << cellname(c.N, c.n) << " char T_p on H^1_c = H^1 = " << PolyStr(a)
                              << " mod 10007^2; operator " << fx(since(th)) << " s" << std::endl;
                }
            }
            // above the ceiling: refused before anything is built
            const P::PairResult R = P::AssembleDescentPair(997, 2);
            check(R.refused && R.pieces.empty(), "(C) (997, 2) refused before building: " + R.refusal);
            std::cout << "DESCENT4_COST N=997 n=2 D=" << R.D << " route=Iso2 refused pred_rss_mib=" << mib(R.rss_pred) << std::endl;
            // the full pair at level N (every prime of S, the ell' rank source on): per-piece costs
            for (const LC& c : std::vector<LC>{{23, 2}, {47, 2}}) {
                const P::PairResult F = P::AssembleDescentPair(c.N, c.n);
                if (!check(!F.refused, "(C) the full pair at " + cellname(c.N, c.n) + ": " + F.refusal)) continue;
                std::cout << groups_line(F) << std::endl;
                for (const auto& r : F.pieces)
                    std::cout << "DESCENT4_COST N=" << c.N << " n=" << c.n << " D=" << F.D << " ell=" << r.ell << " route=" << r.route
                              << " build_s=" << fx(r.build_seconds) << " rank_prime_s=" << fx(r.rank_seconds) << " les_s=" << fx(r.les_seconds)
                              << " total_s=" << fx(r.seconds) << " growth_mib=" << mib(r.rss_meas > r.rss_base ? r.rss_meas - r.rss_base : 0)
                              << std::endl;
            }
            // the level-1 pair to larger n (every prime of S)
            for (uint32_t n : {16u, 20u, 24u, 30u}) {
                const P::PairResult L1 = P::AssembleDescentPair(1, n);
                if (!check(!L1.refused, "(C) level 1 n=" + std::to_string(n) + ": " + L1.refusal)) continue;
                std::cout << groups_line(L1) << std::endl;
                for (const auto& r : L1.pieces)
                    std::cout << "DESCENT4_COST N=1 n=" << n << " ell=" << r.ell << " route=" << r.route << " total_s=" << fx(r.seconds)
                              << " growth_mib=" << mib(r.rss_meas > r.rss_base ? r.rss_meas - r.rss_base : 0) << std::endl;
            }
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentPairSlowValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0), 3) << " s" << std::endl;
    return g_fail;
}
