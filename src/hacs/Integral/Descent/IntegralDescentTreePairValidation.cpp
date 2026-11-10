// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT D5c: IntegralDescentTreePairValidation / IntegralDescentTreePairSlowValidation.  See the header for the arms.
// [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.34, sec:treepair and its code map (a)-(g)), cited by label.  The oracle
// d4_pair_oracle.tsv shares nothing with cohomology_app (cd7_pair_check.py's model); its HECKE rows are the C-D7 referee's.
#include "IntegralDescentTreePairValidation.h"
#include "IntegralDescentPairValidation.h"

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "DescentMaps.h"
#include "DescentOracle.h"
#include "DescentPair.h"
#include "DescentTree.h"
#include "DescentTreePair.h"

#include "../Model/IntegralHecke.h"
#include "../Model/TreePair.h"
#include "../../Hecke/ExactInt.h"
#include "../../../ModArithmetic/EngineCheck.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace P = integral_descent::pair;
namespace TP = integral_descent::treepair;
namespace H3 = integral_descent::hecke3;
using integral_descent::AbelianGroupType;
using integral_descent::maps::Mat;
using integral_descent::maps::Pres;
using integral_descent::maps::ZMod;
using integral_model::TreePairFault;
using U32Ctx = engine::ModCtx<uint32_t>;
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
std::string at3(uint32_t ell, uint32_t N, uint32_t n) { return "ell=" + std::to_string(ell) + " " + cellname(N, n); }

// ---- small matrix helpers -----------------------------------------------------------------------------------------------
Mat Identity(uint32_t n) {
    Mat I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.at(i, i) = 1;
    return I;
}
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
bool IsZero(const ZMod& Z, const Mat& A) {
    for (uint32_t x : A.a)
        if (Z.red(x)) return false;
    return true;
}
Mat Red(const ZMod& Z, Mat A) {
    for (auto& x : A.a) x = Z.red(x);
    return A;
}
Mat SubM(const ZMod& Z, const Mat& A, const Mat& B) {
    Mat C = A;
    for (size_t i = 0; i < C.a.size(); ++i) C.a[i] = Z.sub(Z.red(A.a[i]), Z.red(B.a[i]));
    return C;
}
Mat Rows(const Mat& A, uint32_t r0, uint32_t n) {
    Mat B(n, A.c);
    for (uint32_t i = 0; i < n; ++i) std::copy(A.row(r0 + i), A.row(r0 + i) + A.c, B.row(i));
    return B;
}
Mat Cols(const Mat& A, uint32_t c0, uint32_t n) {
    Mat B(A.r, n);
    for (uint32_t i = 0; i < A.r; ++i) std::copy(A.row(i) + c0, A.row(i) + c0 + n, B.row(i));
    return B;
}
Mat Diag2(const Mat& A, const Mat& B) {
    Mat C(A.r + B.r, A.c + B.c);
    for (uint32_t i = 0; i < A.r; ++i) std::copy(A.row(i), A.row(i) + A.c, C.row(i));
    for (uint32_t i = 0; i < B.r; ++i) std::copy(B.row(i), B.row(i) + B.c, C.row(A.r + i) + A.c);
    return C;
}
ZMod MinZ(const ZMod& a, const ZMod& b) { return a.k <= b.k ? a : b; }
// F_a F_b - F_c (all in Y's coordinates) lies in Y's relations
bool SquareCommutes(const Pres& Y, const ZMod& Z, const Mat& Fa, const Mat& Fb, const Mat& Fc, const Mat& Fd, std::string& why) {
    const Mat L = P::MatMulZ(Z, Red(Z, Fa), Red(Z, Fb));
    const Mat R = P::MatMulZ(Z, Red(Z, Fc), Red(Z, Fd));
    return integral_descent::maps::IdentityOnCohomology(Y, SubM(Z, L, R), why);
}
bool IsIdentityOnH(const Pres& Y, const ZMod& Z, const Mat& F, std::string& why) {
    if (F.r != F.c) {
        why = "not square";
        return false;
    }
    return integral_descent::maps::IdentityOnCohomology(Y, SubM(Z, Red(Z, F), Identity(F.r)), why);
}
std::vector<uint32_t> Charpoly(const ZMod& Z, const Mat& A) {
    const U32Ctx ctx = Z.k >= 2 ? U32Ctx::prime_power(Z.ell, Z.k) : U32Ctx(Z.ell);
    engine::DenseBuf<uint32_t> D(A.r, A.c);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) D.at(i, j) = Z.red(A.at(i, j));
    std::vector<uint32_t> lo = integral_model::CharpolyDivisionFree(ctx, D);
    std::reverse(lo.begin(), lo.end());
    return lo;
}
Mat FreeBlock(const Mat& F, const Pres& X, const Pres& Y) {
    const uint32_t tx = static_cast<uint32_t>(X.tors.size()), ty = static_cast<uint32_t>(Y.tors.size());
    Mat B(Y.free, X.free);
    for (uint32_t i = 0; i < Y.free; ++i)
        for (uint32_t j = 0; j < X.free; ++j) B.at(i, j) = F.at(ty + i, tx + j);
    return B;
}
uint64_t PowMod(uint64_t b, uint32_t e, uint64_t m) {
    uint64_t r = 1 % m;
    b %= m;
    for (uint32_t i = 0; i < e; ++i) r = r * b % m;
    return r;
}
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

// ---- the oracle (the reader of IntegralDescentPairValidation, repeated: that one is file-local) ----------------------------------
using Cell = std::array<uint32_t, 2>;
struct Oracle {
    bool found = false;
    std::string error;
    std::map<Cell, std::map<std::pair<std::string, uint32_t>, AbelianGroupType>> rows;
    std::map<std::array<uint32_t, 3>, std::vector<int64_t>> hecke;
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

// ===========================================================================
// (M) the chain-level certificates
// ===========================================================================
void ArmM() {
    for (uint32_t ell : {2u, 3u}) {
        const integral_model::TreePairComparison& C = TP::SharedTreePairComparison(ell);
        const integral_model::TreePairCertificates c = C.certify();
        check(c.all(), "(M) prop:treepair(1)(b), (3)(a)-(c) certified at ell = " + std::to_string(ell) + ": " + c.first_failure);
        std::cout << "  (M) ell=" << ell << ": " << c.checked << " chain-level identities exact in Q[Gamma] to degree " << C.q_top()
                  << " (tables, B, a^(ell) over f_a, Phi^Z_d / Psi^Z_d over r_d / i_d, Psi^Z_d in Isob, Phi^d Psi^d = id, the squares Y_Phi, "
                  << "Y_Psi); built in " << fx(C.build_seconds) << " s" << std::endl;
    }
    // ell = 5: the Bredon tree pair (prop:treepair(2)(c), (6): a check, not a production route)
    const integral_model::TreePairComparison C5(H3::SharedTreeComparison(), 5, 3);
    const integral_model::TreePairCertificates c5 = C5.certify();
    check(c5.all(), "(M) the Bredon tree pair certified at ell = 5: " + c5.first_failure);
    std::cout << "  (M) ell=5 (Bredon, a check): " << c5.checked << " identities" << std::endl;
}

// ===========================================================================
// (X) the closed form against the chain level
// ===========================================================================
void ArmX() {
    struct XC {
        uint32_t ell, N, n;
    };
    uint32_t cells = 0;
    uint64_t entries = 0;
    for (const XC& x : std::vector<XC>{{2, 1, 4}, {2, 1, 5}, {2, 2, 2}, {2, 3, 1}, {2, 5, 0}, {3, 1, 4}, {3, 2, 2}, {3, 5, 2}, {3, 9, 0}}) {
        const TP::TreePairCochains C(x.ell, x.N, x.n, 5);
        const P::PairComplex& X = C.tree();
        const ZMod& Z = C.Z();
        bool ok = true;
        for (uint32_t q = 0; q < X.T; ++q) {
            // a^* (closed form) = the pullback of the chain-level a^(ell)
            ok = ok && Eq(Z, X.r[q], C.axis_star_chain(q, Identity(X.c[q])));
            // d_B (closed form) = the pullback of the cusp model's differential
            if (X.cb[q] && X.cb[q + 1]) ok = ok && Eq(Z, X.db[q], C.cusp_d_chain(q, Identity(X.cb[q])));
            // the tree's d in J-coordinates = TreeTransport's
            if (q < C.transport().q_top() + 1) ok = ok && Eq(Z, X.d[q], C.transport().differential(q));
            entries += uint64_t(X.r[q].a.size()) + X.db[q].a.size() + X.d[q].a.size();
        }
        check(ok, "(X) the closed form of prop:treepair(2) = the chain-level a^(ell), the cusp differential, the tree's d at " + at3(x.ell, x.N, x.n));
        ++cells;
    }
    std::cout << "  (X) " << cells << " cells: a^*, d_B, d_T of the closed form = the chain-level maps' pullbacks, " << entries << " entries" << std::endl;
}

// ===========================================================================
// (T) tree = dense = oracle
// ===========================================================================
void ArmT(const Oracle& T) {
    uint32_t cells = 0, rows = 0, same = 0, tree_pieces = 0;
    double tt = 0, td = 0;
    for (const auto& [c, rws] : T.rows) {
        const auto a = clk::now();
        const P::PairResult Rt = P::AssembleDescentPair(c[0], c[1]);
        tt += since(a);
        if (!check(!Rt.refused, "(T) tree pair at " + cellname(c[0], c[1]) + ": " + Rt.refusal)) continue;
        check(Rt.route_two == "tree", "(T) the 2- and 3-adic pieces ran on the tree at " + cellname(c[0], c[1]));
        for (const auto& pc : Rt.pieces)
            if (pc.ell <= 3) {
                ++tree_pieces;
                check(pc.route == "tree" && pc.rank_checks > 0, "(T) the ell=" + std::to_string(pc.ell) + " piece is the tree cone with its ell' ranks at " +
                                                                    cellname(c[0], c[1]));
            }
        P::PairResult Rd;
        {
            integral_descent::ScopedDescentTreePair dense(false);
            const auto b = clk::now();
            Rd = P::AssembleDescentPair(c[0], c[1]);
            td += since(b);
        }
        check(Rd.route_two == "Iso2", "(T) the scoped rollback selects the dense pair at " + cellname(c[0], c[1]));
        if (check(same_result(Rt, Rd), "(T) tree pair = dense pair, every group and map type, at " + cellname(c[0], c[1]))) ++same;
        for (const auto& [k, g] : rws) {
            const AbelianGroupType* x = lookup(Rt, k.first, k.second);
            ++rows;
            check(x && *x == g, "(T) " + cellname(c[0], c[1]) + " " + k.first + "^" + std::to_string(k.second) + ": tree " +
                                    (x ? x->to_string() : std::string("(missing)")) + " == oracle " + g.to_string());
        }
        ++cells;
    }
    // the D5c audit's m3: extra_primes = {3} with n odd (M^{-I} = 0: the 3-adic cusp cochains vanish) -- the tree route answers as the dense
    uint32_t x3 = 0;
    for (uint32_t n : {1u, 5u}) {
        P::PairOptions o;
        o.extra_primes = {3};
        const P::PairResult Rt = P::AssembleDescentPair(1, n, o);
        P::PairResult Rd;
        {
            integral_descent::ScopedDescentTreePair dense(false);
            Rd = P::AssembleDescentPair(1, n, o);
        }
        bool piece3 = false;
        for (const auto& pc : Rt.pieces) piece3 = piece3 || (pc.ell == 3 && pc.route == "tree");
        if (check(!Rt.refused && !Rd.refused && piece3 && same_result(Rt, Rd),
                  "(T) extra_primes = {3}, n odd, " + cellname(1, n) + ": the tree route (3-adic piece on the tree) = the dense route: " + Rt.refusal +
                      Rd.refusal))
            ++x3;
    }
    std::cout << "  (T) extra_primes = {3} with n odd at (1,1), (1,5): tree = dense on " << x3 << " cells (M^{-I} = 0, the audit's m3)" << std::endl;
    std::cout << "  (T) " << cells << " oracle cells: tree pair = oracle on " << rows << " groups / map types, tree = dense on " << same
              << " cells (every group and map), " << tree_pieces << " tree pieces (2- and 3-adic) each certified by its ell' ranks; tree "
              << fx(tt, 3) << " s, dense " << fx(td, 3) << " s" << std::endl;
}

// ===========================================================================
// (J) the cone maps
// ===========================================================================
struct JCell {
    uint32_t ell, N, n;
};
uint32_t g_jcells = 0, g_jinv = 0;
uint64_t g_jentries = 0;
void JOne(const JCell& x, bool print) {
    const std::string at = at3(x.ell, x.N, x.n);
    const TP::TreePairCochains C(x.ell, x.N, x.n, 5);
    const P::PairComplex& Xt = C.tree();
    const P::PairComplex& Xi = C.iso();
    const ZMod& Z = C.Z();
    const P::PairBetti bt = P::PairBettiOf(x.N, x.n);
    P::PairLocal Lt, Li;
    P::PieceReport rt, ri;
    rt.min_prec = ri.min_prec = 1000;
    if (!check(P::RunPairLes(Xt, bt, 4, Lt, rt), "(J) the tree pair's sequence at " + at + ": " + rt.why)) return;
    if (!check(P::RunPairLes(Xi, bt, 4, Li, ri), "(J) the Iso pair's sequence at " + at + ": " + ri.why)) return;
    // cochain maps and strict blocks, entrywise, cone degrees -1 .. 3
    bool cm = true, strict = true;
    for (int j = -1; j <= 3; ++j) {
        const uint32_t a = static_cast<uint32_t>(j + 1);
        const Mat It = Identity(P::ConeRank(Xt, j)), Ii = Identity(P::ConeRank(Xi, j));
        const Mat Fj = C.j_phi(j, It), Gj = C.j_psi(j, Ii);
        cm = cm && Eq(Z, P::MatMulZ(Z, P::ConeDifferential(Xi, j), Fj), C.j_phi(j + 1, P::ConeDifferential(Xt, j)));
        cm = cm && Eq(Z, P::MatMulZ(Z, P::ConeDifferential(Xt, j), Gj), C.j_psi(j + 1, P::ConeDifferential(Xi, j)));
        g_jentries += Fj.a.size() + Gj.a.size();
        if (j >= 0) {
            const uint32_t q = static_cast<uint32_t>(j);
            // j_Phi coneinc = coneinc Phi_d^*: the columns of the c-part map to (0, Phi_d^* c); coneproj j_Phi = Phi_T^* coneproj
            const Mat Fc = Cols(Fj, Xt.c[a], Xt.cb[q]);
            strict = strict && IsZero(Z, Rows(Fc, 0, Xi.c[a])) && Eq(Z, Rows(Fc, Xi.c[a], Xi.cb[q]), C.phi_d_star(q, Identity(Xt.cb[q])));
            strict = strict && Eq(Z, Rows(Cols(Fj, 0, Xt.c[a]), 0, Xi.c[a]), C.phi_T_star(a, Identity(Xt.c[a])));
            const Mat Gc = Cols(Gj, Xi.c[a], Xi.cb[q]);
            strict = strict && IsZero(Z, Rows(Gc, 0, Xt.c[a])) && Eq(Z, Rows(Gc, Xt.c[a], Xt.cb[q]), C.psi_d_star(q, Identity(Xi.cb[q])));
        }
    }
    check(cm, "(J) j_Phi, j_Psi are cochain maps (entrywise, cone degrees -1..3) at " + at);
    check(strict, "(J) the strict blocks j coneinc = coneinc Phi_d^*, coneproj j = Phi_T^* coneproj (and for j_Psi) at " + at);
    // mutually inverse on H^q_c (q <= 4) and the groups equal; Phi_d^*, Psi_d^* on H^q(d); Phi_T^*, Psi_T^* on H^q
    for (uint32_t q = 1; q <= 4; ++q) {
        check(Lt.Hc[q].type() == Li.Hc[q].type(), "(J) H^" + std::to_string(q) + "_c of the tree pair = of (Iso, Isob) at " + at);
        if (!Lt.Hc[q].g()) continue;
        ZMod za, zb;
        const Mat Fphi = TP::InducedFromImages(Lt.Hc[q], Li.Hc[q], C.j_phi(static_cast<int>(q) - 1, Tr(Lt.Hc[q].gensT)), za);
        const Mat Fpsi = TP::InducedFromImages(Li.Hc[q], Lt.Hc[q], C.j_psi(static_cast<int>(q) - 1, Tr(Li.Hc[q].gensT)), zb);
        const ZMod z0 = MinZ(za, zb);
        std::string w1, w2;
        check(IsIdentityOnH(Lt.Hc[q], z0, P::MatMulZ(z0, Red(z0, Fpsi), Red(z0, Fphi)), w1),
              "(J) H(j_Psi) H(j_Phi) = id on H^" + std::to_string(q) + "_c of the tree at " + at + ": " + w1);
        check(IsIdentityOnH(Li.Hc[q], z0, P::MatMulZ(z0, Red(z0, Fphi), Red(z0, Fpsi)), w2),
              "(J) H(j_Phi) H(j_Psi) = id on H^" + std::to_string(q) + "_c of (Iso, Isob) at " + at + ": " + w2);
        g_jinv += 2;
    }
    for (uint32_t q = 0; q <= 4; ++q) {
        check(Lt.Hd[q].type() == Li.Hd[q].type() && Lt.H[q].type() == Li.H[q].type(), "(J) H^q(d), H^q equal at q=" + std::to_string(q) + " " + at);
        if (Lt.Hd[q].g()) {
            ZMod za, zb;
            const Mat Fphi = TP::InducedFromImages(Lt.Hd[q], Li.Hd[q], C.phi_d_star(q, Tr(Lt.Hd[q].gensT)), za);
            const Mat Fpsi = TP::InducedFromImages(Li.Hd[q], Lt.Hd[q], C.psi_d_star(q, Tr(Li.Hd[q].gensT)), zb);
            const ZMod z0 = MinZ(za, zb);
            std::string w;
            check(IsIdentityOnH(Lt.Hd[q], z0, P::MatMulZ(z0, Red(z0, Fpsi), Red(z0, Fphi)), w),
                  "(J) H(Psi_d^*) H(Phi_d^*) = id on H^" + std::to_string(q) + "(d) at " + at + ": " + w);
            ++g_jinv;
        }
        if (Lt.H[q].g()) {
            ZMod za, zb;
            const Mat Fphi = TP::InducedFromImages(Lt.H[q], Li.H[q], C.phi_T_star(q, Tr(Lt.H[q].gensT)), za);
            const Mat Fpsi = TP::InducedFromImages(Li.H[q], Lt.H[q], C.psi_T_star(q, Tr(Li.H[q].gensT)), zb);
            const ZMod z0 = MinZ(za, zb);
            std::string w;
            check(IsIdentityOnH(Lt.H[q], z0, P::MatMulZ(z0, Red(z0, Fpsi), Red(z0, Fphi)), w),
                  "(J) H(Psi_T^*) H(Phi_T^*) = id on H^" + std::to_string(q) + " at " + at + ": " + w);
            ++g_jinv;
        }
    }
    if (print) {
        std::ostringstream o;
        o << "  (J) " << at << " H_c (" << x.ell << "-local, q=1..4):";
        for (uint32_t q = 1; q <= 4; ++q) o << " " << Lt.Hc[q].type().to_string();
        std::cout << o.str() << std::endl;
    }
    ++g_jcells;
}
void ArmJ() {
    g_jcells = g_jinv = 0;
    g_jentries = 0;
    for (const JCell& x : std::vector<JCell>{{2, 1, 0}, {2, 1, 4}, {2, 2, 2}, {2, 3, 1}, {2, 5, 0}, {2, 1, 3}, {3, 1, 0}, {3, 1, 4}, {3, 2, 2}, {3, 9, 0}})
        JOne(x, x.N == 1 && x.n == 0);
    std::cout << "  (J) " << g_jcells << " cells: j_Phi, j_Psi cochain maps with the strict blocks (" << g_jentries << " entries), " << g_jinv
              << " identities H(j_Psi) H(j_Phi) = id / H(j_Phi) H(j_Psi) = id on H^q_c and the boundary / bulk comparisons on H^q(d), H^q "
              << "(at (1,0): 2-torsion in H^4_c, the five-lemma caveat of the referee)" << std::endl;
}

// ===========================================================================
// (H) Hecke by transport against D4
// ===========================================================================
struct HCount {
    uint32_t ops = 0, ids = 0, polys = 0, oracle_rows = 0;
    uint64_t eta_entries = 0, strict = 0, cocyc = 0, blocks = 0;
    uint32_t eta_deg = 0, eta_skip = 0;
    double eta_s = 0;
};
void HeckeOne(const Oracle& T, uint32_t ell, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, HCount& K, uint32_t qm = 4) {
    const std::string at = at3(ell, N, n) + (qm == 4 ? std::string() : " q_max=" + std::to_string(qm));
    std::vector<uint32_t> qs;   // p !| N (p | N is U_p, refused PDividesN: arm (R))
    for (uint32_t p : ps)
        if (N % p) qs.push_back(p);
    TP::TreePairHeckeOptions ho;
    ho.q_max = qm;
    const TP::TreePairHeckeRead R = TP::ReadTreePairHecke(ell, N, n, qs, ho);
    if (!check(!R.refused, "(H) the tree pair read at " + at + ": " + R.refusal)) return;
    const TP::TreePairCochains C(ell, N, n, qm + 1);
    const P::PairComplex& Xi = C.iso();
    P::PairLocal Li;
    P::PieceReport ri;
    ri.min_prec = 1000;
    if (!check(P::RunPairLes(Xi, P::PairBettiOf(N, n), qm, Li, ri), "(H) the Iso pair at " + at + ": " + ri.why)) return;
    const P::PairBetti bt = P::PairBettiOf(N, n);
    for (const TP::TreePairHeckeOp& op : R.ops) {
        const std::string atp = at + " p=" + std::to_string(op.p);
        if (!check(!op.refused, "(H) T_p transported " + atp + ": " + op.refusal)) continue;
        ++K.ops;
        K.eta_entries += op.cert.eta_entries;
        K.strict += op.cert.strict_entries;
        K.cocyc += op.cert.cocycle_columns;
        K.eta_s += op.cert.eta_seconds;
        K.blocks += op.cert.block_entries;
        K.eta_deg += op.cert.eta_degrees;
        K.eta_skip += op.cert.eta_skipped;
        check(op.cert.eta_entrywise && op.cert.eta_entries > 0 && op.cert.block_entries > 0,
              "(H) the eta identity and the applied operator's blocks checked entrywise (D <= 1300) " + atp);
        // the D5c audit's m2: the eta identity on every cone degree 0 .. q_max - 1 except, at q_max = 5, the top one (4: H^5_c), COUNTED
        if (qm == 4)
            check(op.cert.eta_degrees == 4 && op.cert.eta_skipped == 0, "(H) the eta identity on cone degrees 0..3, none skipped " + atp);
        else
            check(op.cert.eta_degrees == 4 && op.cert.eta_skipped == 1 && op.cert.eta_skipped_top == 4,
                  "(H) q_max = 5: the eta identity on cone degrees 0..3, the top degree 4 counted as skipped " + atp);
        const P::ConeHecke H = P::BuildConeHecke(N, n, op.p, C.ctx(), qm + 1);
        if (!check(!H.refused, "(H) D4's operator " + atp + ": " + H.refusal)) continue;
        for (uint32_t q = 1; q <= qm; ++q) {
            const Pres& Ht = R.L.Hc[q];
            const Pres& Hi = Li.Hc[q];
            if (!Ht.g()) continue;
            ZMod za, zb;
            const Mat Fphi = TP::InducedFromImages(Ht, Hi, C.j_phi(static_cast<int>(q) - 1, Tr(Ht.gensT)), za);
            const Mat Ti = P::InducedMatrix(Hi, Hi, Diag2(P::FromDenseBuf(H.bulk[q]), P::FromDenseBuf(H.bdry[q - 1])), zb);
            const ZMod z0 = MinZ(MinZ(za, zb), op.Z_Hc[q]);
            std::string w;
            check(SquareCommutes(Hi, z0, Ti, Fphi, Fphi, op.on_Hc[q], w),
                  "(H) T_p on H^" + std::to_string(q) + "_c through the tree = D4's through H(j_Phi) " + atp + ": " + w);
            ++K.ids;
        }
        for (uint32_t q = 0; q <= qm; ++q) {
            if (R.L.H[q].g()) {
                ZMod za, zb;
                const Mat Fphi = TP::InducedFromImages(R.L.H[q], Li.H[q], C.phi_T_star(q, Tr(R.L.H[q].gensT)), za);
                const Mat Ti = P::InducedMatrix(Li.H[q], Li.H[q], P::FromDenseBuf(H.bulk[q]), zb);
                const ZMod z0 = MinZ(MinZ(za, zb), op.Z_H[q]);
                std::string w;
                check(SquareCommutes(Li.H[q], z0, Ti, Fphi, Fphi, op.on_H[q], w), "(H) T_p on H^" + std::to_string(q) + " = D4's through H(Phi_T^*) " + atp + ": " + w);
                ++K.ids;
            }
            if (R.L.Hd[q].g()) {
                ZMod za, zb;
                const Mat Fphi = TP::InducedFromImages(R.L.Hd[q], Li.Hd[q], C.phi_d_star(q, Tr(R.L.Hd[q].gensT)), za);
                const Mat Ti = P::InducedMatrix(Li.Hd[q], Li.Hd[q], P::FromDenseBuf(H.bdry[q]), zb);
                const ZMod z0 = MinZ(MinZ(za, zb), op.Z_Hd[q]);
                std::string w;
                check(SquareCommutes(Li.Hd[q], z0, Ti, Fphi, Fphi, op.on_Hd[q], w), "(H) T_p on H^" + std::to_string(q) + "(d) = D4's through H(Phi_d^*) " + atp + ": " + w);
                ++K.ids;
            }
        }
        // the Eisenstein value on H^0(d) and (oracle rows) char T_p on H^1_c
        if (R.L.Hd[0].free) {
            const ZMod z = op.Z_Hd[0];
            check(Charpoly(z, FreeBlock(op.on_Hd[0], R.L.Hd[0], R.L.Hd[0])) == PowerLinear(z, (1 + PowMod(op.p, n + 1, z.m)) % z.m, bt.bd0),
                  "(H) T_p on H^0(d) has char (x - (1 + p^{n+1}))^{c(N)} " + atp);
            ++K.polys;
        }
        const auto it = T.hecke.find({N, n, op.p});
        if (it != T.hecke.end() && R.L.Hc[1].free) {
            const ZMod z = op.Z_Hc[1];
            std::vector<uint32_t> want;
            for (int64_t v : it->second) {
                int64_t r = v % static_cast<int64_t>(z.m);
                if (r < 0) r += static_cast<int64_t>(z.m);
                want.push_back(static_cast<uint32_t>(r));
            }
            check(Charpoly(z, FreeBlock(op.on_Hc[1], R.L.Hc[1], R.L.Hc[1])) == want, "(H) char T_p on H^1_c = the oracle's HECKE row " + atp);
            ++K.oracle_rows;
        }
    }
}
void ArmH(const Oracle& T) {
    HCount K;
    // the D > 1300 mode (generators + kFreivaldsVectors random vectors), forced at small cells: clean on the true operator, the same read
    {
        uint32_t same = 0;
        uint64_t rnd = 0;
        struct FC {
            uint32_t ell, N, n, p;
        };
        for (const FC& f : std::vector<FC>{{2, 1, 10, 3}, {2, 11, 0, 3}, {3, 1, 10, 2}, {3, 11, 0, 2}}) {
            TP::TreePairHeckeOptions o;
            o.force_generators = true;
            const TP::TreePairHeckeRead Rf = TP::ReadTreePairHecke(f.ell, f.N, f.n, {f.p}, o);
            const TP::TreePairHeckeRead Rd = TP::ReadTreePairHecke(f.ell, f.N, f.n, {f.p});
            const std::string at = at3(f.ell, f.N, f.n) + " p=" + std::to_string(f.p);
            if (!check(!Rf.refused && !Rf.ops[0].refused && !Rd.refused && !Rd.ops[0].refused, "(H) the forced generators mode reads " + at + ": " +
                                                                                                    Rf.refusal + Rd.refusal))
                continue;
            bool eq = Rf.ops[0].cert.random_columns > 0 && !Rf.ops[0].cert.eta_entrywise;
            for (uint32_t q = 1; q <= 4; ++q) eq = eq && Eq(Rd.ops[0].Z_Hc[q], Rf.ops[0].on_Hc[q], Rd.ops[0].on_Hc[q]);
            if (check(eq, "(H) generators + random vectors (forced): certified, the same T_p on H^q_c " + at)) ++same;
            rnd += Rf.ops[0].cert.random_columns;
        }
        std::cout << "  (H) the D > 1300 certificate mode forced at 4 cells: generators + " << TP::kFreivaldsVectors << " random vectors per cone degree ("
                  << rnd << " random columns), certified and the same read on " << same << " cells" << std::endl;
    }
    struct HC {
        uint32_t ell, N, n;
    };
    for (const HC& c : std::vector<HC>{{2, 1, 10}, {2, 11, 0}, {2, 2, 6}, {2, 13, 0}, {2, 3, 4}, {3, 1, 10}, {3, 11, 0}, {3, 3, 4}, {3, 2, 2}})
        HeckeOne(T, c.ell, c.N, c.n, {2, 3, 5, 7}, K);
    std::cout << "  (H) " << K.ops << " operators T_p (p <= 7, p !| N) transported to the tree pair (ReadTreePairHecke) at (1,10), (11,0), "
              << "(2,6), (13,0), (3,4) at 2 and (1,10), (11,0), (3,4), (2,2) at 3: " << K.ids << " identities with D4's operator on (Iso, Isob) "
              << "through H(j_Phi), H(Phi_T^*), H(Phi_d^*) (H^q_c, H^q, H^q(d), q <= 4, torsion included), " << K.polys << " Eisenstein values on "
              << "H^0(d), " << K.oracle_rows << " oracle HECKE rows on H^1_c; certificates: " << K.eta_entries << " eta-identity entries (entrywise, "
              << fx(K.eta_s, 3) << " s, " << K.eta_deg << " cone degrees, " << K.eta_skip << " skipped), " << K.blocks
              << " entries of the applied operator's blocks (= T'_T, eta_T, T'_d), " << K.strict << " strict-block entries, " << K.cocyc
              << " cocycle columns" << std::endl;
}
// (Q5) q_max = 5 (the D5c audit's m2): H^q_c, H^q, H^q(d) for q <= 5 against D4's BuildConeHecke(..., 6) at one cell per ell, and the
// skipped top degree of the eta identity counted on the read
void ArmQ5(const Oracle& T) {
    HCount K;
    struct HC {
        uint32_t ell, N, n;
        std::vector<uint32_t> ps;
    };
    for (const HC& c : std::vector<HC>{{2, 1, 10, {3}}, {2, 11, 0, {3}}, {3, 1, 10, {2}}})
        HeckeOne(T, c.ell, c.N, c.n, c.ps, K, 5);
    std::cout << "  (Q5) q_max = 5: " << K.ops << " operators at (1,10), (11,0) at 2 and (1,10) at 3: " << K.ids << " identities with D4's "
              << "operator (q <= 5, H^5_c included); eta identity on " << K.eta_deg << " cone degrees, SKIPPED on " << K.eta_skip
              << " (the top cone degree 4 at q_max = 5: the comparisons stop at degree " << H3::kTreeComparisonTop << "); "
              << K.blocks << " block entries" << std::endl;
}

// ===========================================================================
// (O) a homotopy-type pair operator theta^* = 1 + K^* d + d K^*
// ===========================================================================
// K : Iso_q -> Iso_{q+1}, K(y) = T x(y), x(y) the first cell of degree q + 1 in Isob for y in Isob (into_bulk: also boundary y into the
// bulk -- a K that does not preserve Isob), else the first bulk cell.  theta^*_q = 1 + K^*_q d_q + d_{q-1} K^*_{q-1} on C^q_I, q <= 4.
TP::DenseIsoOp ThetaOp(const TP::TreePairCochains& C, bool into_bulk) {
    const P::PairComplex& X = C.iso();
    const integral_model::LocalModel& L = P::FreeIsoModel();
    const ZMod& Z = C.Z();
    const uint32_t D = C.D();
    const auto M = integral_descent::InducedSymModule(C.N(), C.n(), C.ctx());
    const engine::DenseBuf<uint32_t> RT = M.rho(hecke::Mat22::T());
    std::vector<Mat> Ks;   // K^*_q : C^{q+1} -> C^q, q = 0 .. 4
    for (uint32_t q = 0; q + 1 <= 5; ++q) {
        Mat K(X.c[q], X.c[q + 1]);
        int64_t xb = -1, xn = -1;
        for (uint32_t c = 0; c < L.rank(q + 1); ++c) {
            if (L.cells(q + 1)[c].boundary && xb < 0) xb = c;
            if (!L.cells(q + 1)[c].boundary && xn < 0) xn = c;
        }
        ENGINE_CHECK(xb >= 0 && xn >= 0);
        for (uint32_t y = 0; y < L.rank(q); ++y) {
            const bool b = L.cells(q)[y].boundary;
            const uint32_t x = static_cast<uint32_t>(b && !into_bulk ? xb : xn);
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) K.at(y * D + i, x * D + j) = Z.red(RT.at(i, j));
        }
        Ks.push_back(std::move(K));
    }
    std::vector<Mat> Th;
    for (uint32_t q = 0; q <= 4; ++q) {
        Mat A = Identity(X.c[q]);
        const Mat a = P::MatMulZ(Z, Ks[q], X.d[q]);
        for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = Z.add(A.a[i], a.a[i]);
        if (q >= 1) {
            const Mat b = P::MatMulZ(Z, X.d[q - 1], Ks[q - 1]);
            for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = Z.add(A.a[i], b.a[i]);
        }
        Th.push_back(std::move(A));
    }
    return TP::DenseIsoOp(Z, std::move(Th), D);
}
// the diagonal (T'_T[1], T'_d) is a cochain map on Cone(a^*)^j -> ^{j+1}, entrywise (validation scale)
bool DiagonalIsCochainMap(const TP::TreePairCochains& C, const TP::IsoCochainOp& op, int j, bool& eta_nonzero) {
    const P::PairComplex& X = C.tree();
    const ZMod& Z = C.Z();
    const uint32_t a = static_cast<uint32_t>(j + 1), q = static_cast<uint32_t>(j);
    bool cl = true;
    const Mat A = Identity(X.c[a]);
    const Mat eta = C.eta(op, j, A, cl);
    eta_nonzero = eta_nonzero || !IsZero(Z, eta);
    // (T'_T[1], T'_d) D = D (T'_T[1], T'_d)  <=>  T'_d a^* = a^* T'_T on C^{a}  (the bulk and boundary parts commute with d separately)
    const Mat lhs = C.bdry_hecke(op, a, P::MatMulZ(Z, X.r[a], A), cl);
    const Mat rhs = P::MatMulZ(Z, X.r[a], C.bulk_hecke(op, a, A));
    (void)q;
    return Eq(Z, lhs, rhs);
}
void ArmO(uint32_t& eta_nz_cells, uint32_t& diag_fail_cells) {
    struct OC {
        uint32_t ell, N, n;
    };
    uint32_t cells = 0, idents = 0;
    eta_nz_cells = diag_fail_cells = 0;
    for (const OC& c : std::vector<OC>{{2, 1, 0}, {2, 1, 4}, {2, 2, 2}, {2, 3, 1}, {3, 1, 0}, {3, 1, 4}, {3, 2, 2}}) {
        const std::string at = at3(c.ell, c.N, c.n);
        const TP::TreePairCochains C(c.ell, c.N, c.n, 5);
        P::PairLocal L;
        P::PieceReport rp;
        rp.min_prec = 1000;
        if (!check(P::RunPairLes(C.tree(), P::PairBettiOf(c.N, c.n), 4, L, rp), "(O) the tree pair at " + at)) continue;
        const TP::DenseIsoOp th = ThetaOp(C, false);
        TP::TreePairHeckeOp op;
        TP::TransportOnPresentations(C, L, 3, th, true, op);
        if (!check(!op.refused, "(O) theta^* transported at " + at + ": " + op.refusal)) continue;
        for (uint32_t q = 1; q <= 3; ++q) {
            if (!L.Hc[q].g()) continue;
            std::string w;
            check(IsIdentityOnH(L.Hc[q], op.Z_Hc[q], op.on_Hc[q], w), "(O) T^c(theta^*) = id on H^" + std::to_string(q) + "_c (theta ~ id as pairs) at " + at + ": " + w);
            ++idents;
        }
        bool nz = false, diag_ok = true;
        for (int j = 0; j <= 2; ++j) diag_ok = DiagonalIsCochainMap(C, th, j, nz) && diag_ok;
        eta_nz_cells += nz ? 1 : 0;
        diag_fail_cells += diag_ok ? 0 : 1;
        std::cout << "  (O) " << at << ": eta_T " << (nz ? "!= 0" : "= 0") << ", the diagonal (T'_T[1], T'_d) " << (diag_ok ? "IS" : "is NOT")
                  << " a cochain map (cone degrees 0..2)" << std::endl;
        ++cells;
        // a K leaving Isob: not a pair datum -- refused IsobClosure
        if (c.ell == 2 && c.N == 1 && c.n == 4) {
            const TP::DenseIsoOp bad = ThetaOp(C, true);
            TP::TreePairHeckeOp ob;
            TP::TransportOnPresentations(C, L, 3, bad, false, ob);
            check(ob.refused && ob.code == P::PairRefusal::IsobClosure, "(O)/(R) a K with K(Isob) not in Isob is refused [IsobClosure]: " + ob.refusal);
        }
    }
    std::cout << "  (O) theta^* = 1 + K^* d + d K^* (K(Isob) in Isob, K(y) = T x(y), not a fibre map) on " << cells << " cells: T^c a cochain map "
              << "with its certificates, = id on H^q_c (" << idents << " identities); eta_T != 0 on " << eta_nz_cells << " cells, the diagonal not a "
              << "cochain map on " << diag_fail_cells << " (prop:treepair(4)(a): eta is forced)" << std::endl;
}

// ===========================================================================
// (N) negative controls and refusal codes
// ===========================================================================
// outcome of a run under a planted fault: "caught" (a hecke::HeckeInternalError or a refusal with the fault), "clean" (the same values)
std::string g_caught_by;   // the certificate that caught the last fault (its message, shortened)
std::string Outcome(const std::function<void()>& f) {
    g_caught_by.clear();
    try {
        f();
    } catch (const hecke::HeckeInternalError& e) {
        const std::string w = e.what();
        g_caught_by = w.find("off-diagonal block") != std::string::npos   ? "the applied operator's eta block"
                      : w.find("second component") != std::string::npos   ? "the applied operator's T'_d block"
                      : w.find("first component") != std::string::npos    ? "the applied operator's bulk / strict block"
                      : w.find("non-cocycle") != std::string::npos        ? "D T^c g = 0"
                      : w.find("eta identity") != std::string::npos       ? "the eta identity"
                                                                          : w.substr(0, 90);
        return "caught";
    } catch (const std::exception& e) {
        return std::string("other:") + e.what();
    }
    return "clean";
}
void ArmN() {
    using integral_model::ScopedTreePairFault;
    uint32_t caught = 0, clean = 0;
    auto expect = [&](const std::string& what, const std::string& got, const std::string& want) {
        check(got == want, "(N) " + what + ": " + got + " (expected " + want + ")");
        (got == "caught" ? caught : clean) += 1;
        std::cout << "  (N) " << what << ": " << got << (g_caught_by.empty() ? std::string() : " [by " + g_caught_by + "]") << std::endl;
    };
    // 1. S for S^{-1} in a: caught at ell = 2 for n ODD (the production assembly throws); the same map at ell = 3 (the referee's m3) AND,
    //    on cochains, at ell = 2 for n even: S^{-1} = c S and rho(c) = (-1)^n on M_N, so for n even rho(S^{-1}) = rho(S) (beyond m3)
    {
        ScopedTreePairFault f(TreePairFault::AxisSForSinv);
        expect("S for S^{-1} in a, ell=2, AssembleDescentPair(1,5) (n odd)", Outcome([] { P::AssembleDescentPair(1, 5); }), "caught");
        expect("S for S^{-1} in a, ell=2, AssembleDescentPair(3,1) (n odd)", Outcome([] { P::AssembleDescentPair(3, 1); }), "caught");
        expect("S for S^{-1} in a, ell=2, AssembleDescentPair(1,4) (n even: rho(S^{-1}) = rho(S), the same cochains)",
               Outcome([] { P::AssembleDescentPair(1, 4); }), "clean");
        const U32Ctx ctx = U32Ctx::prime_power(3, 19);
        const auto M = integral_model::TreeInducedSymModule(2, 2, ctx);
        const P::PairComplex A = TP::BuildTreePairComplex(3, ctx, M, 5);
        P::PairComplex B;
        {
            ScopedTreePairFault none(TreePairFault::None);
            B = TP::BuildTreePairComplex(3, ctx, M, 5);
        }
        bool same = true;
        for (uint32_t q = 0; q <= 5; ++q) same = same && Eq(A.Z, A.r[q], B.r[q]);
        expect("S for S^{-1} in a, ell=3, (2,2): the same map (vacuous, m3)", same ? "clean" : "caught", "clean");
        expect("S for S^{-1} in the chain-level a, ell=2 (the comparison's certificate)", Outcome([] {
                   bool ok = false;
                   try {
                       ok = integral_model::TreePairComparison(H3::SharedTreeComparison(), 2, 3).certify().all();
                   } catch (const std::logic_error&) {
                       ok = false;   // a solver's precondition: (L') sees a non-cycle
                   }
                   if (!ok) throw hecke::HeckeInternalError("[TreePairCertificate] the chain-level a is not certified");
               }),
               "caught");
    }
    // 2. the S^{-1} e_U v^U term dropped: caught at ell = 2 and at ell = 3 (the ell = 3 control)
    {
        ScopedTreePairFault f(TreePairFault::AxisDropBeta);
        expect("S^{-1} e_U v^U dropped, ell=2, AssembleDescentPair(1,4)", Outcome([] { P::AssembleDescentPair(1, 4); }), "caught");
        P::PairOptions o3;
        o3.only_primes = {3};
        expect("S^{-1} e_U v^U dropped, ell=3, AssembleDescentPair(2,2) (piece 3)", Outcome([&] { P::AssembleDescentPair(2, 2, o3); }), "caught");
    }
    // 3. 1 - T^2 for 1 - T in B
    {
        ScopedTreePairFault f(TreePairFault::CuspOneMinusT2);
        expect("1 - T^2 in B, ell=2, AssembleDescentPair(1,4)", Outcome([] { P::AssembleDescentPair(1, 4); }), "caught");
        P::PairOptions o3;
        o3.only_primes = {3};
        expect("1 - T^2 in B, ell=3, AssembleDescentPair(2,2) (piece 3)", Outcome([&] { P::AssembleDescentPair(2, 2, o3); }), "caught");
    }
    // 4. + Y_Phi^* in j_Phi, planted into the OPERATOR only (eta() never carries a fault: the D5c audit's m1): caught by the production
    //    route's own certificates -- the applied operator's eta block against eta_T, or D T^c g = 0 -- at 2 and at 3
    {
        ScopedTreePairFault f(TreePairFault::JPhiPlusY);
        struct YC {
            uint32_t ell, N, n, p;
        };
        for (const YC& y : std::vector<YC>{{2, 1, 4, 3}, {3, 1, 4, 2}, {3, 1, 10, 2}, {3, 11, 0, 2}})
            expect("+ Y_Phi^* in j_Phi, " + at3(y.ell, y.N, y.n) + ", ReadTreePairHecke(T_" + std::to_string(y.p) + ")",
                   Outcome([&] { TP::ReadTreePairHecke(y.ell, y.N, y.n, {y.p}); }), "caught");
    }
    // 5. eta dropped (planted into the operator only): theta^* (a homotopy-type operator, eta != 0) and real T_p, at 2 and at 3
    {
        ScopedTreePairFault f(TreePairFault::EtaDropped);
        struct EC {
            uint32_t ell, N, n;
            const char* want;
        };
        for (const EC& e : std::vector<EC>{{2, 1, 4, "caught"}, {2, 2, 2, "caught"}, {3, 1, 4, "caught"}, {3, 1, 0, "caught"}}) {
            const std::string got = Outcome([&] {
                const TP::TreePairCochains C(e.ell, e.N, e.n, 5);
                P::PairLocal L;
                P::PieceReport rp;
                rp.min_prec = 1000;
                if (!P::RunPairLes(C.tree(), P::PairBettiOf(e.N, e.n), 4, L, rp)) throw std::runtime_error("sequence");
                TP::TreePairHeckeOp op;
                TP::TransportOnPresentations(C, L, 3, ThetaOp(C, false), true, op);
            });
            expect(std::string("eta dropped, theta^*, ") + at3(e.ell, e.N, e.n), got, e.want);
        }
        expect("eta dropped, T_3, ell=2 (N,n)=(1,10)", Outcome([] { TP::ReadTreePairHecke(2, 1, 10, {3}); }), "caught");
        expect("eta dropped, T_2, ell=3 (N,n)=(1,10)", Outcome([] { TP::ReadTreePairHecke(3, 1, 10, {2}); }), "caught");
    }
    // 6. the C-D8b referee's Arm (G): a d_B-cocycle-valued error SHARED by the applied operator and eta_T (planted in Y_Psi^* at chain
    //    degree q: alpha |-> (r . alpha) z, z a generator of H^q(d) that is a d_B-cocycle mod ell^K).  The block comparison (shared),
    //    D T^c g = 0 and the identity on the generators (cocycle values, cocycle alphas) cannot see it; in the "generators" mode of
    //    D > 1300 (forced here at small cells) it passes WITHOUT the random vectors and changes the read of H^{q+1}_c when [z] is not in
    //    the image of H^q(a^*); the random vectors and the entrywise identity (D <= 1300) catch it through eta^q d on C^q.  The first
    //    (cell, q, z) whose read changes is pinned by search (deterministic: fixed cells, generators and seeds).
    {
        struct GC {
            uint32_t ell, N, n, p;
        };
        bool found = false;
        uint32_t tried = 0;
        for (const GC& c : std::vector<GC>{{2, 1, 10, 3}, {2, 11, 0, 3}, {2, 1, 4, 3}, {2, 2, 6, 3}, {2, 3, 4, 5}, {3, 1, 10, 2}, {3, 11, 0, 2}}) {
            if (found) break;
            const TP::TreePairHeckeRead R0 = TP::ReadTreePairHecke(c.ell, c.N, c.n, {c.p});
            if (R0.refused || R0.ops.empty() || R0.ops[0].refused) continue;
            const TP::TreePairCochains C0(c.ell, c.N, c.n, 5);
            for (uint32_t q = 1; q <= 3 && !found; ++q) {
                const Pres& Hd = R0.L.Hd[q];
                for (uint32_t g = 0; g < Hd.g() && !found; ++g) {
                    const std::vector<uint32_t> z(Hd.gensT.a.begin() + size_t(g) * Hd.gensT.c, Hd.gensT.a.begin() + size_t(g + 1) * Hd.gensT.c);
                    Mat zc(static_cast<uint32_t>(z.size()), 1);
                    zc.a = z;
                    if (!IsZero(C0.Z(), P::MatMulZ(C0.Z(), C0.tree().db[q], zc))) continue;   // a d_B-cocycle mod ell^K
                    ++tried;
                    integral_descent::treepair::ScopedPlantedCocycleEta plant(q, z, 0xC0C7C1Eull + q);
                    TP::TreePairHeckeOptions o;
                    o.force_generators = true;
                    o.random_vectors = 0;
                    TP::TreePairHeckeRead Rg;
                    const std::string got0 = Outcome([&] { Rg = TP::ReadTreePairHecke(c.ell, c.N, c.n, {c.p}, o); });
                    bool changed = false;
                    if (got0 == "clean" && !Rg.refused && !Rg.ops.empty() && !Rg.ops[0].refused)
                        for (uint32_t qq = 1; qq <= 4; ++qq) changed = changed || !Eq(Rg.ops[0].Z_Hc[qq], Rg.ops[0].on_Hc[qq], R0.ops[0].on_Hc[qq]);
                    if (got0 != "clean" || !changed) continue;
                    found = true;
                    const std::string at = at3(c.ell, c.N, c.n) + " T_" + std::to_string(c.p) + ", Y_Psi^* at q=" + std::to_string(q) +
                                           " + (r . alpha) z, z = generator " + std::to_string(g) + " of H^" + std::to_string(q) + "(d)";
                    expect("cocycle-valued shared error (Arm G), generators only (no random vectors; the read of H^" + std::to_string(q + 1) +
                               "_c CHANGES), " + at, got0, "clean");
                    o.random_vectors = TP::kFreivaldsVectors;
                    expect("cocycle-valued shared error (Arm G), generators + " + std::to_string(TP::kFreivaldsVectors) + " random vectors, " + at,
                           Outcome([&] { TP::ReadTreePairHecke(c.ell, c.N, c.n, {c.p}, o); }), "caught");
                    expect("cocycle-valued shared error (Arm G), the default (entrywise, D <= 1300), " + at,
                           Outcome([&] { TP::ReadTreePairHecke(c.ell, c.N, c.n, {c.p}); }), "caught");
                }
            }
        }
        std::cout << "  (N) Arm (G): " << tried << " (cell, q, z) planted before the first read-changing one" << std::endl;
        check(found, "(N) Arm (G): a cocycle-valued shared error that passes the generators' checks and changes the read was found");
    }
    std::cout << "TREEPAIR_FAULTS caught=" << caught << " clean=" << clean << " (S/S^{-1}: ell=2 caught for n odd, the same cochains for n even and at ell=3; beta dropped: 2, 3 "
              << "caught; 1 - T^2: 2, 3 caught; +Y_Phi: T_p caught at (1,4) at 2 and (1,4), (1,10), (11,0) at 3; eta dropped: theta^* caught at (1,4), (2,2) "
              << "at 2 and (1,4), (1,0) at 3, T_p at (1,10) at 2 and 3 -- the operator faults planted into the operator ONLY, caught by its blocks against "
              << "the separately assembled T'_T, eta_T, T'_d; Arm (G): a cocycle-valued error shared with eta_T, clean on the generators alone, "
              << "caught by the random vectors and entrywise)" << std::endl;
    // refusal codes (prop:treepair(6))
    uint32_t codes = 0;
    auto code_is = [&](const TP::TreePairHeckeRead& R, P::PairRefusal c, const std::string& what) {
        const bool ok = !R.ops.empty() && R.ops[0].refused && R.ops[0].code == c;
        check(ok, "(R) " + what + " refused [" + P::PairRefusalName(c) + "]: " + (R.ops.empty() ? R.refusal : R.ops[0].refusal));
        codes += ok ? 1 : 0;
    };
    code_is(TP::ReadTreePairHecke(2, 1, 4, {4}), P::PairRefusal::NotPrime, "T_4");
    code_is(TP::ReadTreePairHecke(2, 2, 2, {2}), P::PairRefusal::PDividesN, "T_2 at level 2");
    {
        TP::TreePairHeckeOptions o;
        o.q_max = 3;
        code_is(TP::ReadTreePairHecke(2, 1, 4, {3}, o), P::PairRefusal::DegreeRange, "q_max = 3");
        o.q_max = 6;
        code_is(TP::ReadTreePairHecke(2, 1, 4, {3}, o), P::PairRefusal::DegreeRange, "q_max = 6");
        o.q_max = 4;
        o.rss_cap_bytes = 1ull << 20;
        code_is(TP::ReadTreePairHecke(2, 1, 4, {3}, o), P::PairRefusal::RssCeiling, "a 1 MiB ceiling");
    }
    code_is(TP::ReadTreePairHecke(2, 1, 4, {100003}), P::PairRefusal::RssCeiling, "T_100003 (the pair lift's rule 19, through Lift)");
    {
        P::PairOptions o;
        o.rss_cap_bytes = 1ull << 20;
        const P::PairResult R = P::AssembleDescentPair(1, 4, o);
        const bool ok = R.refused && R.pieces.empty() && R.refusal.find("tree cusp pair") != std::string::npos;
        check(ok, "(R) the groups refused before building at a 1 MiB ceiling: " + R.refusal);
        codes += ok ? 1 : 0;
    }
    std::cout << "  (R) " << codes << " refusals by code: NotPrime, PDividesN, DegreeRange x2, RssCeiling (transport), RssCeiling via Lift "
              << "(p = 100003), RssCeiling (groups); IsobClosure in (O); IntegerCap / DomainCap / InputTableCertificate / LiftCertificate are the pair "
              << "lift's (D4's tests)" << std::endl;
}

}  // namespace

uint32_t IntegralDescentTreePairValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentTreePairValidation (DESCENT D5c: the tree cusp pair; [C] sec:treepair) ===" << std::endl;
    std::cout << "EFFECTIVE HeckeDescentTreePair route=" << (integral_descent::DescentTreePairEnabled() ? "tree" : "dense")
              << " source=" << integral_descent::DescentTreePairSource() << std::endl;
    const Oracle T = LoadOracle();
    try {
        // the arms test the TREE route whatever the env says (HECKE_DESCENT_TREE_PAIR=0 / HECKE_DESCENT_TREE=0 select the dense pair as the
        // default; the rollback itself is (RB) below and the dense arm of (T))
        integral_descent::ScopedDescentTreePair tree_route(true);
        ArmM();
        ArmX();
        if (check(T.found && T.error.empty(), "(T) the oracle loads: " + (T.error.empty() ? oracle_path() : T.error))) ArmT(T);
        ArmJ();
        ArmH(T);
        ArmQ5(T);
        uint32_t nz = 0, df = 0;
        ArmO(nz, df);
        check(nz >= 1 && df >= 1, "(O) eta_T != 0 and the diagonal not a cochain map somewhere (prop:treepair(4)(a) not vacuous)");
        ArmN();
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    // (RB) the rollback keeps value-level coverage in --all: D4's standard name on the dense route, in-process
    {
        integral_descent::ScopedDescentTreePair dense(false);
        std::cout << "  (RB) --- IntegralDescentPairValidation on the dense pair (route=" << (integral_descent::DescentTreePairEnabled() ? "tree" : "dense")
                  << ") ---" << std::endl;
        const auto tr = clk::now();
        const uint64_t keep_checks = g_checks;
        const uint32_t keep_fail = g_fail;
        const uint32_t f = IntegralDescentPairValidation();
        g_checks = keep_checks;
        g_fail = keep_fail;
        check(f == 0, "(RB) IntegralDescentPairValidation passes on the dense rollback (" + std::to_string(f) + " failing)");
        std::cout << "  (RB) the dense rollback: IntegralDescentPairValidation " << (f == 0 ? "PASS" : "FAIL") << " in " << fx(since(tr), 3) << " s" << std::endl;
    }
    std::cout << "IntegralDescentTreePairValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0), 3) << " s" << std::endl;
    return g_fail;
}

// ===========================================================================
// the slow companion: the cost ladder
// ===========================================================================
uint32_t IntegralDescentTreePairSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentTreePairSlowValidation (DESCENT D5c: the cost of the tree cusp pair, DESCENT5c_COST) ===" << std::endl;
    struct LC {
        uint32_t N, n;
    };
    std::vector<LC> ladder = {{47, 2}, {97, 2}, {191, 2}, {283, 2}, {383, 2}, {499, 2}};
    std::string arm = "default";
    integral_descent::ScopedDescentTreePair tree_route(true);   // the cost line measures the tree pair (the dense ratio cells are scoped back)
    try {
        if (const char* e = std::getenv("HECKE_DESCENT_TREEPAIR_CELLS")) {
            ladder.clear();
            std::stringstream ss(e);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                const size_t a = tok.find(':');
                ENGINE_CHECK_MSG(a != std::string::npos, "HECKE_DESCENT_TREEPAIR_CELLS: tokens N:n");
                ladder.push_back({static_cast<uint32_t>(std::stoul(tok.substr(0, a))), static_cast<uint32_t>(std::stoul(tok.substr(a + 1)))});
            }
            arm = "env";
        }
        const uint64_t cap = 40ull << 30;
        std::cout << "EFFECTIVE IntegralDescentTreePairSlowValidation arm=" << arm << " cells=" << ladder.size() << " rss_cap_gib=" << (cap >> 30)
                  << " route=" << (integral_descent::DescentTreePairEnabled() ? "tree" : "dense") << std::endl;
        for (const LC& c : ladder) {
            const uint32_t D = integral_descent::PsiIndex(c.N) * (c.n + 1);
            std::cout << "  DESCENT5c_PREDICT " << cellname(c.N, c.n) << " D=" << D << " rss_pred_mib=" << mib(TP::PredictTreePairRss(2, D, 5))
                      << " dense_rss_pred_mib=" << mib(P::PredictPairRss(2, D, 5)) << " cap_gib=" << (cap >> 30) << std::endl;
        }
        for (const LC& c : ladder) {
            P::PairOptions o;
            o.only_primes = {2};
            o.prime_rank_check = false;
            o.rss_cap_bytes = cap;
#if defined(__GLIBC__)
            malloc_trim(0);
#endif
            const P::PairResult R = P::AssembleDescentPair(c.N, c.n, o);
            if (R.pieces.empty()) {
                check(R.refused, "(C) " + cellname(c.N, c.n) + " refused before building: " + R.refusal);
                std::cout << "DESCENT5c_COST N=" << c.N << " n=" << c.n << " D=" << R.D << " route=tree refused pred_rss_mib=" << mib(R.rss_pred) << std::endl;
                continue;
            }
            const P::PieceReport& r = R.pieces[0];
            check(r.ok, "(C) the 2-adic tree pair piece at " + cellname(c.N, c.n) + ": " + r.why);
            const uint64_t growth = r.rss_meas > r.rss_base ? r.rss_meas - r.rss_base : 0;
            check(growth <= r.rss_pred, "(C) the 2-adic tree pair piece within its RSS prediction at " + cellname(c.N, c.n) + " (rule 19)");
            const uint32_t cone = r.c.size() > 2 ? r.c[2] + r.cb[1] : 0;
            std::cout << "DESCENT5c_COST N=" << c.N << " n=" << c.n << " D=" << R.D << " route=" << r.route << " K=" << r.K << " T=" << r.T
                      << " c=" << (r.c.size() > 2 ? r.c[2] : 0) << " cb=" << (r.cb.size() > 2 ? r.cb[2] : 0) << " cone=" << cone
                      << " build_s=" << fx(r.build_seconds) << " les_s=" << fx(r.les_seconds) << " total_s=" << fx(r.seconds)
                      << " rss_base_mib=" << mib(r.rss_base) << " rss_peak_mib=" << mib(r.rss_meas) << " growth_mib=" << mib(growth)
                      << " pred_mib=" << mib(r.rss_pred) << " matrices_mib=" << mib(r.matrix_bytes) << std::endl;
        }
        if (arm == "default") {
            // the dense pair at two cells (the ratio), every group equal on the full pair at (23, 2), (47, 2)
            for (const LC& c : std::vector<LC>{{23, 2}, {47, 2}}) {
                const auto ta = clk::now();
                const P::PairResult A = P::AssembleDescentPair(c.N, c.n);
                const double t_tree = since(ta);
                P::PairResult B;
                double t_dense = 0;
                {
                    integral_descent::ScopedDescentTreePair dense(false);
                    const auto tb = clk::now();
                    B = P::AssembleDescentPair(c.N, c.n);
                    t_dense = since(tb);
                }
                check(same_result(A, B), "(C) the full pair: tree = dense at " + cellname(c.N, c.n));
                for (const auto& r : A.pieces)
                    if (r.ell <= 3)
                        std::cout << "DESCENT5c_COST N=" << c.N << " n=" << c.n << " D=" << A.D << " ell=" << r.ell << " route=" << r.route
                                  << " full_pair total_s=" << fx(r.seconds) << " rank_prime_s=" << fx(r.rank_seconds) << std::endl;
                std::cout << "DESCENT5c_COST N=" << c.N << " n=" << c.n << " full_pair tree_s=" << fx(t_tree) << " dense_s=" << fx(t_dense) << std::endl;
            }
            // the transport and the eta identity's cost (the referee's N5): T_2, T_3 at 2 and T_2 at 3 on growing D, entrywise up to D = 1300
            for (const LC& c : std::vector<LC>{{23, 2}, {47, 2}, {97, 2}, {191, 2}, {383, 2}}) {
                for (uint32_t ell : {2u, 3u}) {
                    const auto th = clk::now();
                    const TP::TreePairHeckeRead R = TP::ReadTreePairHecke(ell, c.N, c.n, ell == 2 ? std::vector<uint32_t>{3} : std::vector<uint32_t>{2});
                    if (!check(!R.refused && !R.ops.empty() && !R.ops[0].refused, "(E) the transport at " + at3(ell, c.N, c.n) + ": " + R.refusal +
                                                                                       (R.ops.empty() ? std::string() : R.ops[0].refusal)))
                        continue;
                    const auto& op = R.ops[0];
                    std::cout << "DESCENT5c_ETA ell=" << ell << " N=" << c.N << " n=" << c.n << " D=" << R.D << " p=" << op.p << " entrywise="
                              << op.cert.eta_entrywise << " eta_entries=" << op.cert.eta_entries << " eta_s=" << fx(op.cert.eta_seconds)
                              << " block_entries=" << op.cert.block_entries << " op_s=" << fx(op.seconds) << " read_s=" << fx(since(th))
                              << " rss_pred_mib=" << mib(R.rss_pred) << " of_which_transport_mib=" << mib(R.rss_pred_transport) << std::endl;
                }
            }
            // the D > 1300 certificate mode (generators + kFreivaldsVectors random vectors; the C-D8b referee's Arm (G)) forced, its cost
            // against D: each application is linear in D per column, the columns fixed (DESCENT5c_report.md Sec. 9.7 extrapolates)
            for (const LC& c : std::vector<LC>{{97, 2}, {191, 2}, {383, 2}}) {
                TP::TreePairHeckeOptions o;
                o.force_generators = true;
                const TP::TreePairHeckeRead R = TP::ReadTreePairHecke(2, c.N, c.n, {3}, o);
                if (!check(!R.refused && !R.ops.empty() && !R.ops[0].refused && R.ops[0].cert.random_columns > 0,
                           "(E) the forced generators + random vectors mode at " + at3(2, c.N, c.n) + ": " + R.refusal))
                    continue;
                const auto& op = R.ops[0];
                std::cout << "DESCENT5c_FREIVALDS ell=2 N=" << c.N << " n=" << c.n << " D=" << R.D << " p=3 random_columns=" << op.cert.random_columns
                          << " block_entries=" << op.cert.block_entries << " eta_entries=" << op.cert.eta_entries << " op_s=" << fx(op.seconds)
                          << " rss_pred_mib=" << mib(R.rss_pred) << " of_which_transport_mib=" << mib(R.rss_pred_transport) << std::endl;
            }
            // T_p by transport against D4 at (23, 2) (D = 72)
            const Oracle T = LoadOracle();
            HCount K;
            HeckeOne(T, 2, 23, 2, {3, 5}, K);
            HeckeOne(T, 3, 23, 2, {2, 5}, K);
            std::cout << "  (H) (23,2): " << K.ops << " operators, " << K.ids << " identities with D4's operator through the comparisons" << std::endl;
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::cout << "IntegralDescentTreePairSlowValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0), 3) << " s" << std::endl;
    return g_fail;
}
