// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3) OrbifoldMPLevelConeValidation -- see the header.
// Gate values: docs/implementation/slot22/MP3_predictions.md P6 (committed before the first run).
#include "OrbifoldMPLevelConeValidation.h"

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "MP3Oracle.h"
#include "MP3bLevelArms.h"
#include "Models.h"
#include "ModuleCochains.h"
#include "Nerve.h"
#include "Refinement.h"
#include "SubgroupMP.h"

#include "../../Hecke/HeckeGammaH.h"
#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelPair.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/PeriodicityClass.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using hecke::Mat22;
using hecke::level::CosetLevel;
using orbifold_mp::CIRepSide;
using orbifold_mp::InternedMPSet;
using orbifold_mp::ModuleRep;
using orbifold_mp::RefinementLabels;
using orbifold_mp::RepSide;
using orbifold_mp::SparseRows;
using orbifold_mp::StringMap;
using orbifold_mp::SubgroupModel;
using orbifold_mp::Vec;

std::string Join(const std::vector<int64_t>& v) { return orbifold_mp::JoinV(v); }
std::string S(uint64_t x) { return std::to_string(x); }

struct Grp {
    std::string name;
    std::unique_ptr<CosetLevel> L;
};

struct Pair {
    std::string A, B;
};

std::vector<Vec> Sub(const std::vector<Vec>& a, const std::vector<Vec>& b, uint32_t m) {
    std::vector<Vec> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i].resize(a[i].size());
        for (size_t j = 0; j < a[i].size(); ++j) out[i][j] = (a[i][j] + m - b[i][j]) % m;
    }
    return out;
}

// Every row exactly one entry 1 and every column hit exactly once (a bijection of coordinates).
bool IsCoordinateBijection(const SparseRows& A) {
    if (A.rows() != A.ncols) return false;
    std::vector<uint8_t> hit(A.ncols, 0);
    for (uint64_t r = 0; r < A.rows(); ++r) {
        if (A.start[r + 1] - A.start[r] != 1 || A.val[A.start[r]] != 1) return false;
        if (hit[A.col[A.start[r]]]++) return false;
    }
    return true;
}

bool DenseEqual(const Vec& a, const std::vector<uint32_t>& b) { return a.size() == b.size() && a == b; }

}  // namespace

uint32_t OrbifoldMPLevelConeValidation() {
    using namespace hecke::level;
    using namespace hecke::level::les;
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    std::cout << "OrbifoldMPLevelConeValidation: MP-3, MP3_predictions.md P6 (model=mp)\n";
    {
        uint32_t rows = 0;
        const bool ok = orbifold_mp::MP3OracleFileAgrees(&rows);
        // (MP-3b, F-MP3 F2) the sha256 of the file actually read, beside the frozen one; both gated.
        const std::string sha = orbifold_mp::MP3OracleFileSha256();
        const bool sha_ok = sha == orbifold_mp::MP3OracleSha256();
        std::cout << "  ORACLE_FILE mp3_oracle.out sha256=" << sha << " frozen=" << orbifold_mp::MP3OracleSha256() << " sha_match=" << sha_ok
                  << " rows=" << rows << " agree=" << ok << "\n";
        T.check(ok, "P6.0 the tracked mp3_oracle.out equals MP3Oracle.inc row for row");
        T.check(sha_ok && rows == 246, "P6.0 (MP-3b) sha256(tracked mp3_oracle.out) = the frozen value of MP3Oracle.inc, 246 rows");
    }

    std::vector<Grp> G;
    G.push_back({"Gamma", std::make_unique<FullLevel>()});
    G.push_back({"Gamma^2", std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(2))});
    G.push_back({"Gamma^3", std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(3))});
    for (uint64_t N : {2ull, 3ull, 4ull, 9ull}) G.push_back({"G0(" + S(N) + ")", std::make_unique<Gamma0LevelN>(N)});
    G.push_back({"G1(4)", std::make_unique<GammaHLevelN>(4, std::vector<uint64_t>{})});
    auto lvl = [&](const std::string& name) -> const CosetLevel& {
        for (const auto& g : G)
            if (g.name == name) return *g.L;
        ENGINE_CHECK_MSG(false, "OrbifoldMPLevelConeValidation: unknown group");
        return *G[0].L;
    };
    const std::vector<Pair> pairs = {{"G0(2)", "G0(4)"}, {"G0(3)", "G0(9)"}, {"G0(4)", "G1(4)"}, {"Gamma", "Gamma^2"}, {"Gamma", "Gamma^3"}};

    const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
    const integral_model::IsotropyModel CI(5);
    const integral_model::PeriodicityClass P(CI);
    const InternedMPSet SG(MM.m7, 3);
    const orbifold_mp::PhiLift Phi(SG, MM.m7, P, 2);
    T.check(Phi.certify().all(), "P6.1 Phi certified to degree 2 on M7 (as P4.5a)");

    // ---- P6.1 the B-descriptors and the MP sets S_B --------------------------------------------------------------------
    std::map<std::string, SubgroupModel> Bm;
    std::map<std::string, std::unique_ptr<InternedMPSet>> SB;
    for (const auto& g : G) {
        if (g.name == "Gamma") continue;
        const auto ts = std::chrono::steady_clock::now();
        Bm.emplace(g.name, orbifold_mp::SubgroupDescriptor(MM.m7, *g.L, g.name));
        const SubgroupModel& M = Bm.at(g.name);
        const auto& O = orbifold_mp::MP3Desc(g.name);
        T.check(M.vertex_orbits == O.vertex_orbits && M.D.top.size() == O.top_orbits && M.minus_I_in == O.minus_I,
                "P6.1a " + g.name + ": vertex B-orbits " + S(M.vertex_orbits) + ", top B-orbits " + S(M.D.top.size()) + ", -I in B = oracle");
        const auto sc = orbifold_mp::SimplicialComplexProperty(M.D);
        T.check(sc.ok && orbifold_mp::InvariantOrder(M.D), "P6.1a " + g.name + ": (SC) and the invariant order on the B-descriptor");
        SB.emplace(g.name, std::make_unique<InternedMPSet>(M.D, 3));
        const InternedMPSet& Sb = *SB.at(g.name);
        bool inB = true;
        for (const Mat22& x : Sb.elements()) inB = inB && g.L->contains(x);
        T.check(inB, "P6.1b " + g.name + ": every arrow of S_B lies in B (" + S(Sb.elements().size()) + " elements)");
        std::vector<int64_t> nd;
        bool cnt = true;
        for (uint32_t k = 0; k <= 3; ++k) {
            nd.push_back(static_cast<int64_t>(Sb.count_nondegenerate(k)));
            if (O.minus_I) cnt = cnt && Sb.count_nondegenerate(k) == O.nd[k];
        }
        if (O.minus_I) T.check(cnt, "P6.1c " + g.name + ": |S_k^nd(B)| = [Gamma:B] |S_k^nd(M7)|, k <= 3");
        uint64_t c = 0;
        T.check(orbifold_mp::FormalDSquared(Sb, 0, &c) && orbifold_mp::FormalDSquared(Sb, 1, &c), "P6.1d " + g.name + ": d o d = 0 formally, q <= 1");
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        std::cout << "  SUBGROUP_SET model=mp tri=M7 B=" << g.name << " index=" << g.L->m() << " -I_in=" << M.minus_I_in << " vertex_orbits=" << M.vertex_orbits
                  << " tops=" << M.D.top.size() << " |S_0..3^nd|=" << Join(nd) << " elements=" << Sb.elements().size() << " build_s=" << std::fixed
                  << std::setprecision(2) << s << std::defaultfloat << "\n";
    }

    // ---- P6.2 the induced module (= LES-1's M_{A,1}) -----------------------------------------------------------------
    {
        uint32_t ok = 0, total = 0;
        for (const auto& g : G)
            for (uint32_t ell : {2u, 3u})
                for (uint32_t n = 0; n <= 2; ++n) {
                    const ModuleRep N = orbifold_mp::CoindRep(*g.L, n, ell, 1);
                    const uint32_t D = N.dim(), m = N.m();
                    const Vec& RS = N.rho(Mat22::S());
                    const Vec& RU = N.rho(Mat22::ST());
                    const Vec& RN = N.rho(Mat22::I().neg());
                    const Vec RS2 = orbifold_mp::MatMulSmall(RS, RS, D, D, D, m);
                    const Vec RU3 = orbifold_mp::MatMulSmall(orbifold_mp::MatMulSmall(RU, RU, D, D, D, m), RU, D, D, D, m);
                    const Vec RST = orbifold_mp::MatMulSmall(RS, N.rho(Mat22::T()), D, D, D, m);
                    bool hom = RS2 == RN && RU3 == RN && orbifold_mp::MatMulSmall(RN, RN, D, D, D, m) == orbifold_mp::IdentitySmall(D, 1, m) && RST == RU;
                    const ModCtx<uint32_t> ctx(ell);
                    bool les1 = true;
                    for (const Mat22& gm : {Mat22::S(), Mat22::T(), Mat22::I().neg()})
                        les1 = les1 && DenseEqual(N.rho(gm), TwistedMonomialMatrix(*g.L, CharacterDatum::Trivial(ell), gm, n, ctx).dense());
                    T.check(hom && les1, "P6.2 Coind_" + g.name + " Sym^" + S(n) + " F_" + S(ell) + ": a representation (S^2 = U^3 = -I, ST = U) and = LES-1 M_{A,1}");
                    ok += (hom && les1) ? 1 : 0;
                    ++total;
                }
        std::cout << "  COIND model=mp 8 groups x F_2, F_3 x n<=2: representation and = LES-1 TwistedMonomialMatrix " << ok << "/" << total << "\n";
    }

    // ---- P6.3 H^q(B; Sym^n) on C_I with Coind_B (Shapiro, the shipped resolution) ---------------------------------------
    uint32_t ci_cells = 0, ci_agree = 0;
    for (const auto& g : G) {
        const uint32_t idx = g.L->m();
        std::ostringstream line;
        line << "  CI_COIND model=C_I B=" << g.name << " index=" << idx << " (ell,n):H^0..q";
        for (uint32_t ell : {2u, 3u, 5u}) {
            if (idx > 4 && ell == 5) continue;
            for (uint32_t n = 0; n <= (idx > 4 ? 2u : 4u); ++n) {
                const uint32_t qt = idx > 4 ? 3 : 4;
                const CIRepSide ci(CI, orbifold_mp::CoindRep(*g.L, n, ell, 1), qt, false);
                const auto& O = orbifold_mp::MP3Group(g.name, ell, n);
                std::vector<int64_t> hs;
                bool ok = true;
                for (uint32_t q = 0; q <= qt; ++q) {
                    hs.push_back(ci.lenH(q));
                    ok = ok && ci.lenH(q) == O.H[q];
                }
                ++ci_cells;
                ci_agree += ok ? 1 : 0;
                T.check(ok, "P6.3 C_I(Coind_" + g.name + " Sym^" + S(n) + " F_" + S(ell) + ") = oracle " + Join(hs));
                line << " (" << ell << "," << n << "):" << Join(hs);
            }
        }
        std::cout << line.str() << "\n";
    }

    // ---- P6.4 .. P6.9 on the MP model, per (ell, n) --------------------------------------------------------------------
    struct Cell {
        uint32_t ell, n;
    };
    const std::vector<Cell> cells = {{2, 0}, {2, 1}, {3, 0}, {3, 2}, {3, 1}};
    uint32_t mp_cells = 0, mp_agree = 0, rk_cells = 0, rk_agree = 0, cone_cells = 0, cone_agree = 0, fc_cells = 0, fc_agree = 0;
    for (const Cell& c : cells) {
        const uint32_t ell = c.ell, n = c.n, d = n + 1, m = ell;
        const bool only_g14 = (ell == 3 && n == 1);   // (3, 1): the only cell where a group without -I is not acyclic
        const ModuleRep Sym = orbifold_mp::SymRep(n, ell, 1);
        std::map<std::string, std::unique_ptr<ModuleRep>> Co;
        std::map<std::string, std::unique_ptr<CIRepSide>> ci;
        std::map<std::string, std::unique_ptr<RepSide>> shap, nerve;
        std::map<std::string, std::vector<SparseRows>> Sh;      // C(S_Gamma; Coind_B) -> C(S_B; Sym), q <= 2
        std::map<std::string, std::vector<std::vector<Vec>>> zsh;   // Phi^*(Z(C_I; Coind_B)) in C^q(S_Gamma; Coind_B)
        std::map<std::string, RefinementLabels> LB;
        std::map<std::string, StringMap> UB;
        for (const auto& g : G) {
            if (only_g14 && g.name != "G1(4)" && g.name != "G0(4)") continue;
            const std::string& b = g.name;
            const bool isG = b == "Gamma";
            Co.emplace(b, std::make_unique<ModuleRep>(isG ? Sym : orbifold_mp::CoindRep(*g.L, n, ell, 1)));
            const ModuleRep& N = *Co.at(b);
            ci.emplace(b, std::make_unique<CIRepSide>(CI, N, 2, false));
            shap.emplace(b, std::make_unique<RepSide>(SG, N, 2, isG || b == "G1(4)"));
            std::vector<std::vector<Vec>> z;
            for (uint32_t q = 0; q <= 2; ++q)
                z.push_back(orbifold_mp::MapAll(orbifold_mp::PhiPullbackRep(Phi, SG, q, CI.rank(q), N), ci.at(b)->src[q].cocycles, m));
            zsh.emplace(b, std::move(z));
            const std::string at = " " + b + " F_" + S(ell) + " n=" + S(n);
            if (isG || b == "G1(4)") {
                const auto& O = orbifold_mp::MP3Group(b, ell, n);
                bool ok = true;
                std::vector<int64_t> hs;
                for (uint32_t q = 0; q <= 2; ++q) {
                    hs.push_back(shap.at(b)->lenH(q));
                    const auto img = shap.at(b)->image(q, zsh.at(b)[q]);
                    ok = ok && hs[q] == O.H[q] && img.cocycles && static_cast<int64_t>(img.image()) == hs[q];
                }
                T.check(ok, "P6.4 H^q(S_Gamma; Coind) = oracle, Phi^* iso from C_I" + at + " " + Join(hs));
            }
            if (isG) continue;
            const SubgroupModel& M = Bm.at(b);
            const InternedMPSet& Sb = *SB.at(b);
            nerve.emplace(b, std::make_unique<RepSide>(Sb, Sym, 2, true));
            LB.emplace(b, orbifold_mp::LabelsToGamma(M));
            UB.emplace(b, orbifold_mp::MapStrings(Sb, SG, LB.at(b), 2));
            T.check(UB.at(b).outside == 0, "P6.5 " + b + ": every string of S_<=2(B) maps into S(M7) under F'_B -> F'_Gamma");
            const CosetLevel& Lb = *g.L;
            std::vector<SparseRows> sh;
            for (uint32_t q = 0; q <= 2; ++q)
                sh.push_back(orbifold_mp::LabelledPullbackRep(Sb, SG, LB.at(b), UB.at(b), q, d, N.dim(), m,
                                                              [&Lb, n, ell](const Mat22& h) { return orbifold_mp::EvaluationMatrix(Lb, h, n, ell, 1); }));
            // P6.5 the nerve of B = oracle; the Shapiro comparison a cochain map and an iso on H^q (via Phi^* from C_I)
            const auto& O = orbifold_mp::MP3Group(b, ell, n);
            std::vector<int64_t> hs;
            bool ok = true;
            for (uint32_t q = 0; q <= 2; ++q) {
                hs.push_back(nerve.at(b)->lenH(q));
                ok = ok && hs[q] == O.H[q];
            }
            ++mp_cells;
            mp_agree += ok ? 1 : 0;
            T.check(ok, "P6.5 H^q(S_B; Sym^n) = oracle" + at + " " + Join(hs));
            bool cm = true;
            for (uint32_t q = 0; q <= 1; ++q) cm = cm && orbifold_mp::CommutesExactly(sh[q + 1], shap.at(b)->d[q], nerve.at(b)->d[q], sh[q], m);
            T.check(cm, "P6.5 the Shapiro comparison C(S_Gamma; Coind_B) -> C(S_B) is a cochain map, q <= 1" + at);
            if (M.minus_I_in) {
                bool bij = true;
                for (uint32_t q = 0; q <= 2; ++q) bij = bij && IsCoordinateBijection(sh[q]);
                T.check(bij, "P6.5 -I in B: the Shapiro comparison is a bijection of coordinates (an isomorphism of complexes), q <= 2" + at);
            }
            bool iso = true;
            for (uint32_t q = 0; q <= 2; ++q) {
                const auto img = nerve.at(b)->image(q, orbifold_mp::MapAll(sh[q], zsh.at(b)[q], m));
                iso = iso && img.cocycles && static_cast<int64_t>(img.image()) == hs[q] && ci.at(b)->lenH(q) == hs[q];
            }
            T.check(iso, "P6.5 the Shapiro comparison o Phi^* : H^q(C_I; Coind_B) -> H^q(S_B) is an isomorphism, q <= 2" + at);
            // P6.9b the forget-coset map to Gamma = Shapiro o res, on the nose; a cochain map; rank = oracle (via the pair rows)
            const orbifold_mp::ResCor rG = orbifold_mp::ResCorMaps(*G[0].L, Lb, n, ell, 1);
            bool same = true;
            for (uint32_t q = 0; q <= 2; ++q) {
                const SparseRows F = orbifold_mp::LabelledPullbackRep(Sb, SG, LB.at(b), UB.at(b), q, d, d, m, [&Sym](const Mat22& h) { return Sym.rho(h); });
                same = same && orbifold_mp::SpEqual(F, orbifold_mp::SpMul(sh[q], orbifold_mp::CoefficientMap(SG, q, rG.res, N.dim(), d, m), m));
            }
            T.check(same, "P6.9b the forget-coset map S_B -> S_Gamma (labels r, rho(r)) = Shapiro o res^Gamma_B on the nose, q <= 2" + at);
            Sh.emplace(b, std::move(sh));
        }

        // ---- the pairs ------------------------------------------------------------------------------------------------
        for (const Pair& pr : pairs) {
            if (!Co.count(pr.A) || !Co.count(pr.B)) continue;
            const CosetLevel& A = lvl(pr.A);
            const CosetLevel& B = lvl(pr.B);
            const auto& O = orbifold_mp::MP3Pair(pr.A, pr.B, ell, n);
            const std::string at = " " + pr.B + "<" + pr.A + " F_" + S(ell) + " n=" + S(n);
            const ModuleRep& NA = *Co.at(pr.A);
            const ModuleRep& NB = *Co.at(pr.B);
            const uint32_t dA = NA.dim(), dB = NB.dim();
            // P6.6 res / cor
            const orbifold_mp::ResCor rc = orbifold_mp::ResCorMaps(A, B, n, ell, 1);
            bool eqv = true;
            for (const Mat22& gm : {Mat22::S(), Mat22::T(), Mat22::I().neg()}) {
                eqv = eqv && orbifold_mp::MatMulSmall(rc.res, NA.rho(gm), dB, dA, dA, m) == orbifold_mp::MatMulSmall(NB.rho(gm), rc.res, dB, dB, dA, m);
                eqv = eqv && orbifold_mp::MatMulSmall(rc.cor, NB.rho(gm), dA, dB, dB, m) == orbifold_mp::MatMulSmall(NA.rho(gm), rc.cor, dA, dA, dB, m);
            }
            const ModCtx<uint32_t> ctx(ell);
            const RestrictionFibreMap rf(A, CharacterDatum::Trivial(ell), B, CharacterDatum::Trivial(ell), n, ctx);
            const bool les1 = DenseEqual(rc.res, rf.alpha().dense()) && DenseEqual(rc.cor, rf.trace().dense()) && rf.index() == rc.index;
            const bool corres = orbifold_mp::MatMulSmall(rc.cor, rc.res, dA, dB, dA, m) == orbifold_mp::IdentitySmall(dA, rc.index, m);
            T.check(eqv && les1 && corres && rc.index == O.index,
                    "P6.6 res, cor Gamma-maps, = LES-1 alpha / omega, cor o res = [A:B] = " + S(rc.index) + at);
            std::vector<SparseRows> Res, Cor;
            for (uint32_t q = 0; q <= 2; ++q) {
                Res.push_back(orbifold_mp::CoefficientMap(SG, q, rc.res, dB, dA, m));
                Cor.push_back(orbifold_mp::CoefficientMap(SG, q, rc.cor, dA, dB, m));
            }
            const RepSide& XA = *shap.at(pr.A);
            const RepSide& XB = *shap.at(pr.B);
            bool cmap = true, cr = true;
            for (uint32_t q = 0; q <= 1; ++q) {
                cmap = cmap && orbifold_mp::CommutesExactly(Res[q + 1], XA.d[q], XB.d[q], Res[q], m);
                cmap = cmap && orbifold_mp::CommutesExactly(Cor[q + 1], XB.d[q], XA.d[q], Cor[q], m);
            }
            for (uint32_t q = 0; q <= 2; ++q)
                cr = cr && orbifold_mp::SpEqual(orbifold_mp::SpMul(Cor[q], Res[q], m),
                                                orbifold_mp::CoefficientMap(SG, q, orbifold_mp::IdentitySmall(dA, rc.index, m), dA, dA, m));
            T.check(cmap && cr, "P6.6 res, cor cochain maps on C(S_Gamma; Coind) (q <= 1), cor o res = [A:B] on the cochains (q <= 2)" + at);
            // P6.7 rank of res on H^q (MP, via Phi^* from C_I) and on C_I
            std::vector<int64_t> rk, rkci;
            for (uint32_t q = 0; q <= 2; ++q) {
                const auto img = XB.image(q, orbifold_mp::MapAll(Res[q], zsh.at(pr.A)[q], m));
                rk.push_back(img.cocycles ? static_cast<int64_t>(img.image()) : -1);
                const auto cimg = ci.at(pr.B)->image(q, orbifold_mp::MapAllDense(orbifold_mp::CICoefficientMap(CI, q, rc.res, dB, dA, m),
                                                                                 ci.at(pr.A)->src[q].cocycles, m));
                rkci.push_back(static_cast<int64_t>(cimg.image()));
            }
            bool rok = true;
            for (uint32_t q = 0; q <= 2; ++q) rok = rok && rk[q] == O.rank[q] && rkci[q] == O.rank[q];
            ++rk_cells;
            rk_agree += rok ? 1 : 0;
            T.check(rok, "P6.7 rank res^A_B on H^q (MP = C_I = oracle) " + Join(rk) + at);
            // P6.8 the cone of res: H^q(A, B) = oracle; the LES identity
            std::vector<SparseRows> cone;
            std::vector<uint64_t> cdim;
            for (uint32_t q = 0; q <= 2; ++q) {
                cone.push_back(orbifold_mp::ConeDifferential(XA.d[q], Res[q], q ? &XB.d[q - 1] : nullptr, q ? static_cast<uint32_t>(XB.dimC[q - 1]) : 0, m));
                cdim.push_back(XA.dimC[q] + (q ? XB.dimC[q - 1] : 0));
            }
            const std::vector<int64_t> hc = orbifold_mp::LengthsOf(cone, cdim, ell, 1);
            bool cok = true;
            for (uint32_t q = 0; q <= 2; ++q) {
                const int64_t les = (q ? O.HB[q - 1] - rk[q - 1] : 0) + (O.HA[q] - rk[q]);
                cok = cok && hc[q] == O.rel[q] && hc[q] == les;
            }
            ++cone_cells;
            cone_agree += cok ? 1 : 0;
            T.check(cok, "P6.8 H^q(Cone res) = H^q(A, B) = oracle, = the LES, q <= 2 " + Join(hc) + at);
            // P6.9 the forget-coset morphism S_B -> S_A (labels in A, not central)
            const InternedMPSet& Sbm = *SB.at(pr.B);
            const RepSide& NBn = *nerve.at(pr.B);
            std::vector<SparseRows> F;
            std::vector<std::vector<Vec>> srcA(3);   // the A-classes in C^q(S_A; Sym) (or C^q(M7; Sym) for A = Gamma)
            const InternedMPSet* SA = &SG;
            const RepSide* NA_nerve = nullptr;
            RefinementLabels LBA;
            StringMap UBA;
            uint32_t exact = 0;
            if (pr.A == "Gamma") {
                NA_nerve = shap.at("Gamma").get();
                LBA = LB.at(pr.B);
                UBA = UB.at(pr.B);
                exact = static_cast<uint32_t>(LBA.h.size());
                for (uint32_t q = 0; q <= 2; ++q) srcA[q] = zsh.at("Gamma")[q];
            } else {
                SA = SB.at(pr.A).get();
                NA_nerve = nerve.at(pr.A).get();
                LBA = orbifold_mp::ForgetCosetLabels(MM.m7, Bm.at(pr.B), B, Bm.at(pr.A), A, &exact);
                UBA = orbifold_mp::MapStrings(Sbm, *SA, LBA, 2);
                for (uint32_t q = 0; q <= 2; ++q) srcA[q] = orbifold_mp::MapAll(Sh.at(pr.A)[q], zsh.at(pr.A)[q], m);
            }
            uint32_t central = 0;
            for (const Mat22& h : LBA.h) central += (h == Mat22::I() || h == Mat22::I().neg()) ? 1 : 0;
            T.check(UBA.outside == 0, "P6.9 the forget-coset labels: every string of S_<=2(B) maps into S_A" + at);
            for (uint32_t q = 0; q <= 2; ++q)
                F.push_back(orbifold_mp::LabelledPullbackRep(Sbm, *SA, LBA, UBA, q, d, d, m, [&Sym](const Mat22& h) { return Sym.rho(h); }));
            bool fcm = true;
            for (uint32_t q = 0; q <= 1; ++q) fcm = fcm && orbifold_mp::CommutesExactly(F[q + 1], NA_nerve->d[q], NBn.d[q], F[q], m);
            std::vector<int64_t> frk;
            bool sq = true;
            for (uint32_t q = 0; q <= 2; ++q) {
                const auto mapped = orbifold_mp::MapAll(F[q], srcA[q], m);
                const auto img = NBn.image(q, mapped);
                frk.push_back(img.cocycles ? static_cast<int64_t>(img.image()) : -1);
                // the square: forget o Shapiro_A = Shapiro_B o res on H^q
                const auto other = orbifold_mp::MapAll(Sh.at(pr.B)[q], orbifold_mp::MapAll(Res[q], zsh.at(pr.A)[q], m), m);
                sq = sq && NBn.image(q, Sub(mapped, other, m)).image() == 0;
            }
            bool frok = fcm && sq;
            for (uint32_t q = 0; q <= 2; ++q) frok = frok && frk[q] == O.rank[q];
            ++fc_cells;
            fc_agree += frok ? 1 : 0;
            std::cout << "  FORGET_COSET model=mp " << pr.B << "->" << pr.A << " F_" << ell << " n=" << n << " labels=" << LBA.h.size()
                      << " (u=I: " << exact << ", central: " << central << ") cochain_map=" << fcm << " rank H^0..2=" << Join(frk)
                      << " square_on_H=" << sq << "\n";
            T.check(frok, "P6.9 the forget-coset map S_B -> S_A: a cochain map (q <= 1), rank on H^q = res's (oracle), = Shapiro o res on H^q" + at);
            std::cout << "  PAIR model=mp " << pr.B << "<" << pr.A << " F_" << ell << " n=" << n << " [A:B]=" << rc.index << " H(A)=" << Join({O.HA[0], O.HA[1], O.HA[2]})
                      << " H(B)=" << Join({NBn.lenH(0), NBn.lenH(1), NBn.lenH(2)}) << " rank_res=" << Join(rk) << " H(A,B)=" << Join(hc) << "\n";
        }
    }

    // ---- P6.8b the cone in degree 3 on the nerve: Gamma^2 < Gamma, Gamma^3 < Gamma, n = 0, F_2 and F_3 (S_4(M7)) -----------
    {
        const InternedMPSet SG4(MM.m7, 4);
        for (const char* bn : {"Gamma^2", "Gamma^3"})
            for (uint32_t ell : {2u, 3u}) {
                const CosetLevel& B = lvl(bn);
                const ModuleRep NA = orbifold_mp::SymRep(0, ell, 1);
                const ModuleRep NB = orbifold_mp::CoindRep(B, 0, ell, 1);
                const orbifold_mp::ResCor rc = orbifold_mp::ResCorMaps(*G[0].L, B, 0, ell, 1);
                const RepSide XA(SG4, NA, 3, false), XB(SG4, NB, 2, false);
                std::vector<SparseRows> cone;
                std::vector<uint64_t> cdim;
                for (uint32_t q = 0; q <= 3; ++q) {
                    cone.push_back(orbifold_mp::ConeDifferential(XA.d[q], orbifold_mp::CoefficientMap(SG4, q, rc.res, NB.dim(), 1, ell),
                                                                 q ? &XB.d[q - 1] : nullptr, q ? static_cast<uint32_t>(XB.dimC[q - 1]) : 0, ell));
                    cdim.push_back(XA.dimC[q] + (q ? XB.dimC[q - 1] : 0));
                }
                const auto ts = std::chrono::steady_clock::now();
                const std::vector<int64_t> hc = orbifold_mp::LengthsOf(cone, cdim, ell, 1);
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
                const auto& O = orbifold_mp::MP3Pair("Gamma", bn, ell, 0);
                bool ok = true;
                for (uint32_t q = 0; q <= 3; ++q) ok = ok && hc[q] == O.rel[q];
                std::cout << "  PAIR_H3 model=mp " << bn << "<Gamma F_" << ell << " n=0 H^0..3(A,B)=" << Join(hc) << " d^3_cone rows=" << cone[3].rows()
                          << " nnz=" << cone[3].nnz() << " wall_s=" << std::fixed << std::setprecision(2) << s << std::defaultfloat << "\n";
                T.check(ok, std::string("P6.8b H^q(Cone res), q <= 3, on the nerve = oracle, ") + bn + "<Gamma F_" + S(ell) + " n=0");
            }
    }

    // ---- P6.8c the pairs on C_I (q <= 3): rank of res and the cone ------------------------------------------------------
    uint32_t cip_cells = 0, cip_agree = 0;
    for (const Pair& pr : pairs) {
        const CosetLevel& A = lvl(pr.A);
        const CosetLevel& B = lvl(pr.B);
        std::ostringstream line;
        line << "  CI_PAIR model=C_I " << pr.B << "<" << pr.A << " (ell,n):rank|H(A,B) q<=3";
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n = 0; n <= (B.m() > 6 ? 2u : 4u); ++n) {
                const ModuleRep NA = orbifold_mp::CoindRep(A, n, ell, 1), NB = orbifold_mp::CoindRep(B, n, ell, 1);
                const CIRepSide ciA(CI, NA, 3, false), ciB(CI, NB, 3, false);
                const orbifold_mp::ResCor rc = orbifold_mp::ResCorMaps(A, B, n, ell, 1);
                const auto& O = orbifold_mp::MP3Pair(pr.A, pr.B, ell, n);
                std::vector<int64_t> rk;
                std::vector<DenseBuf<uint32_t>> dcone;
                std::vector<uint32_t> ddim;
                for (uint32_t q = 0; q <= 3; ++q) {
                    const DenseBuf<uint32_t> R = orbifold_mp::CICoefficientMap(CI, q, rc.res, NB.dim(), NA.dim(), ell);
                    rk.push_back(static_cast<int64_t>(ciB.image(q, orbifold_mp::MapAllDense(R, ciA.src[q].cocycles, ell)).image()));
                    dcone.push_back(orbifold_mp::DenseCone(ciA.d[q], R, q ? &ciB.d[q - 1] : nullptr, q ? ciB.dimC[q - 1] : 0, ell));
                    ddim.push_back(ciA.dimC[q] + (q ? ciB.dimC[q - 1] : 0));
                }
                const std::vector<int64_t> hc = orbifold_mp::DenseLengthsOf(ModCtx<uint32_t>(ell), dcone, ddim);
                bool ok = true;
                for (uint32_t q = 0; q <= 3; ++q) ok = ok && rk[q] == O.rank[q] && hc[q] == O.rel[q];
                ++cip_cells;
                cip_agree += ok ? 1 : 0;
                T.check(ok, "P6.8c C_I: rank res and H^q(Cone res), q <= 3, = oracle " + pr.B + "<" + pr.A + " F_" + S(ell) + " n=" + S(n));
                line << " (" << ell << "," << n << "):" << Join(rk) << "|" << Join(hc);
            }
        std::cout << line.str() << "\n";
    }

    // ---- P6.10 negative controls -----------------------------------------------------------------------------------------
    {
        const uint32_t ell = 3, n = 2, d = 3, m = 3;
        const ModuleRep Sym = orbifold_mp::SymRep(n, ell, 1);
        const CosetLevel& B = lvl("G0(2)");
        const InternedMPSet& Sb = *SB.at("G0(2)");
        const RefinementLabels L = orbifold_mp::LabelsToGamma(Bm.at("G0(2)"));
        const StringMap U = orbifold_mp::MapStrings(Sb, SG, L, 2);
        const RepSide X(SG, Sym, 1, false), Y(Sb, Sym, 1, false);
        // (a) the label factor dropped
        bool tw = true, untw = true;
        for (uint32_t q = 0; q <= 0; ++q) {
            auto F = [&](uint32_t qq, bool drop) {
                return orbifold_mp::LabelledPullbackRep(Sb, SG, L, U, qq, d, d, m, [&Sym](const Mat22& h) { return Sym.rho(h); }, drop);
            };
            tw = tw && orbifold_mp::CommutesExactly(F(q + 1, false), X.d[q], Y.d[q], F(q, false), m);
            untw = untw && orbifold_mp::CommutesExactly(F(q + 1, true), X.d[q], Y.d[q], F(q, true), m);
        }
        uint32_t central = 0;
        for (const Mat22& h : L.h) central += (h == Mat22::I() || h == Mat22::I().neg()) ? 1 : 0;
        std::cout << "  NC P6.10a model=mp forget-coset G0(2)->Gamma F_3 n=2 (labels " << L.h.size() << ", central " << central
                  << "): with rho(r) cochain map=" << tw << "; label factor dropped cochain map=" << untw << " -> " << (untw ? "NC SILENT" : "NC fires") << "\n";
        T.check(tw && !untw, "P6.10a NC: the forget-coset map without its (non-central) label factor is not a cochain map");
        // (b) a wrong coset representative in res (G0(4) < G0(2), class 2 uses v_2 T)
        const CosetLevel& A2 = lvl("G0(2)");
        const CosetLevel& B4 = lvl("G0(4)");
        const orbifold_mp::ResCor bad = orbifold_mp::ResCorMaps(A2, B4, n, ell, 1, 2);
        const ModuleRep NA = orbifold_mp::CoindRep(A2, n, ell, 1), NB = orbifold_mp::CoindRep(B4, n, ell, 1);
        bool eqv = true;
        for (const Mat22& gm : {Mat22::S(), Mat22::T()})
            eqv = eqv && orbifold_mp::MatMulSmall(bad.res, NA.rho(gm), NB.dim(), NA.dim(), NA.dim(), m) ==
                             orbifold_mp::MatMulSmall(NB.rho(gm), bad.res, NB.dim(), NB.dim(), NA.dim(), m);
        const RepSide XA(SG, NA, 1, false), XB(SG, NB, 1, false);
        const bool cm = orbifold_mp::CommutesExactly(orbifold_mp::CoefficientMap(SG, 1, bad.res, NB.dim(), NA.dim(), m), XA.d[0], XB.d[0],
                                                     orbifold_mp::CoefficientMap(SG, 0, bad.res, NB.dim(), NA.dim(), m), m);
        std::cout << "  NC P6.10b model=mp res G0(4)<G0(2) with v_2 T for v_2, F_3 n=2: Gamma-map=" << eqv << " cochain map=" << cm << " -> "
                  << (!eqv && !cm ? "NC fires" : "NC SILENT") << "\n";
        T.check(!eqv && !cm, "P6.10b NC: a wrong coset representative breaks the Gamma-equivariance of res and the cochain map");
        // (c) cor without its rho-blocks
        const orbifold_mp::ResCor nr = orbifold_mp::ResCorMaps(A2, B4, n, ell, 1, -1, true);
        const orbifold_mp::ResCor ok = orbifold_mp::ResCorMaps(A2, B4, n, ell, 1);
        const bool corres = orbifold_mp::MatMulSmall(nr.cor, ok.res, NA.dim(), NB.dim(), NA.dim(), m) == orbifold_mp::IdentitySmall(NA.dim(), ok.index, m);
        std::cout << "  NC P6.10c model=mp cor with identity blocks, G0(4)<G0(2) F_3 n=2: cor o res = [A:B]=" << corres << " -> "
                  << (corres ? "NC SILENT" : "NC fires") << "\n";
        T.check(!corres, "P6.10c NC: cor without rho(h_i)^{-1} does not satisfy cor o res = [A:B]");
        // (d) a label off by T
        RefinementLabels Lbad = L;
        Lbad.h[1] = Lbad.h[1] * Mat22::T();
        const bool same = orbifold_mp::SameSimplex(MM.m7, Bm.at("G0(2)"), 1, Bm.at("G0(2)"), 1, Lbad.h[1] * Bm.at("G0(2)").r[1].inv_unimodular());
        const StringMap Ubad = orbifold_mp::MapStrings(Sb, SG, Lbad, 2);
        std::cout << "  NC P6.10d model=mp forget-coset label r.T on top 1: same simplex=" << same << " strings outside S(M7)=" << Ubad.outside << " -> "
                  << (!same && Ubad.outside > 0 ? "NC fires" : "NC SILENT") << "\n";
        T.check(!same && Ubad.outside > 0, "P6.10d NC: a forget-coset label off by T is not a label (leaves S(M7))");
        (void)B;
    }

    // ---- P7 (MP-3b, F-MP3 F5 / F3): Z/l^k level pairs, R_Z pairs, -I not in A (MP3bLevelArms.cpp) -----------------------
    uint32_t b_cells = 0, b_agree = 0;
    orbifold_mp::RunMP3bLevelArms(T, MM, CI, P, SG, Phi, &b_cells, &b_agree);

    std::cout << "ORACLE mp3_oracle.out cells=" << (ci_cells + mp_cells + rk_cells + cone_cells + fc_cells + cip_cells)
              << " agree=" << (ci_agree + mp_agree + rk_agree + cone_agree + fc_agree + cip_agree) << "\n";
    std::cout << "  LEVEL summary: C_I(Coind) " << ci_agree << "/" << ci_cells << "; nerve S_B " << mp_agree << "/" << mp_cells << "; rank res "
              << rk_agree << "/" << rk_cells << "; cone " << cone_agree << "/" << cone_cells << "; forget-coset " << fc_agree << "/" << fc_cells
              << "; C_I pairs " << cip_agree << "/" << cip_cells << "; MP-3b cells " << b_agree << "/" << b_cells << "\n";
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPLevelConeValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
