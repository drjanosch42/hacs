// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3b) The MP-3b arm of OrbifoldMPLevelConeValidation -- see MP3bLevelArms.h.
// Gate values: docs/implementation/slot22/MP3b_predictions.md P7 (committed before the first run).
#include "MP3bLevelArms.h"

#include "CochainMaps.h"
#include "MP3Oracle.h"
#include "MP3bOracle.h"
#include "ModuleCochains.h"
#include "Refinement.h"
#include "SubgroupMP.h"

#include "../../Hecke/HeckeGammaH.h"
#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelPair.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace orbifold_mp {

namespace {

using engine::DenseBuf;
using hecke::Mat22;
using hecke::level::CosetLevel;

std::string Str(uint64_t x) { return std::to_string(x); }

std::string RingName(uint32_t ell, uint32_t e) { return e == 1 ? "F_" + Str(ell) : "Z/" + Str(PowU(ell, e)); }

bool IsCoordinateBijection(const SparseRows& A) {
    if (A.rows() != A.ncols) return false;
    std::vector<uint8_t> hit(A.ncols, 0);
    for (uint64_t r = 0; r < A.rows(); ++r) {
        if (A.start[r + 1] - A.start[r] != 1 || A.val[A.start[r]] != 1) return false;
        if (hit[A.col[A.start[r]]]++) return false;
    }
    return true;
}

std::vector<Vec> Sub(const std::vector<Vec>& a, const std::vector<Vec>& b, uint32_t m) {
    std::vector<Vec> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i].resize(a[i].size());
        for (size_t j = 0; j < a[i].size(); ++j) out[i][j] = (a[i][j] + m - b[i][j]) % m;
    }
    return out;
}

struct Tri {
    std::string name;
    const Descriptor* D;
    const InternedMPSet* SG;
    const PhiLift* Phi;
};

struct Cell {
    std::string tri, A, B;
    uint32_t ell, e, n, qt;   // qt: the top degree gated on the MP model (C_I: always 3)
};

// The B-side of one subgroup on one triangulation: its descriptor, its nerve S_B to degree qt + 1.
struct SubSide {
    SubgroupModel M;
    std::unique_ptr<InternedMPSet> S;
};

}  // namespace

void RunMP3bLevelArms(MPTally& T, const MPModels& MM, const integral_model::IsotropyModel& CI, const integral_model::PeriodicityClass& P,
                      const InternedMPSet& SG, const PhiLift& Phi, uint32_t* cells_out, uint32_t* agree_out) {
    using namespace hecke::level;
    using namespace hecke::level::les;
    std::cout << "  MP-3b arm (MP3b_predictions.md P7): Z/l^k level pairs, R_Z pairs, -I not in A\n";
    // ---- P7.0 the frozen oracle file ---------------------------------------------------------------------------------
    {
        uint32_t rows = 0;
        std::string sha;
        const bool ok = MP3bOracleFileAgrees(&rows, &sha);
        const bool sha_ok = sha == MP3bOracleSha256();
        std::cout << "  ORACLE_FILE mp3b_oracle.out sha256=" << sha << " frozen=" << MP3bOracleSha256() << " sha_match=" << sha_ok << " rows=" << rows
                  << " agree=" << ok << "\n";
        T.check(ok && sha_ok && rows == 28, "P7.0 the tracked mp3b_oracle.out = MP3bOracle.inc (28 data lines, sha256 = the frozen value)");
    }

    // ---- the levels ----------------------------------------------------------------------------------------------------
    std::map<std::string, std::unique_ptr<CosetLevel>> L;
    L.emplace("Gamma", std::make_unique<FullLevel>());
    L.emplace("Gamma^2", std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(2)));
    for (uint64_t N : {2ull, 3ull, 4ull, 5ull, 9ull, 25ull}) L.emplace("G0(" + Str(N) + ")", std::make_unique<Gamma0LevelN>(N));
    for (uint64_t N : {4ull, 8ull}) L.emplace("G1(" + Str(N) + ")", std::make_unique<GammaHLevelN>(N, std::vector<uint64_t>{}));
    auto lvl = [&](const std::string& g) -> const CosetLevel& {
        ENGINE_CHECK_MSG(L.count(g), "RunMP3bLevelArms: unknown group");
        return *L.at(g);
    };

    // ---- the triangulations: M7 (the caller's S_Gamma, Phi) and R_Z ----------------------------------------------------
    const InternedMPSet SR(MM.rzd, 3);
    const PhiLift PhiR(SR, MM.rzd, P, 2);
    T.check(PhiR.certify().all(), "P7.1 Phi certified to degree 2 on R_Z (as P4.5a)");
    const std::vector<Tri> tris = {{"M7", &MM.m7, &SG, &Phi}, {"R_Z", &MM.rzd, &SR, &PhiR}};
    auto tri_of = [&](const std::string& t) -> const Tri& {
        for (const Tri& x : tris)
            if (x.name == t) return x;
        ENGINE_CHECK_MSG(false, "RunMP3bLevelArms: unknown triangulation");
        return tris[0];
    };

    // ---- P7.1 the subgroup descriptors and nerves (cached per triangulation and group) -------------------------------
    std::map<std::string, SubSide> subs;
    auto sub = [&](const Tri& tr, const std::string& g, uint32_t kmax) -> const SubSide& {
        const std::string key = tr.name + "/" + g + "/" + Str(kmax);
        auto it = subs.find(key);
        if (it != subs.end()) return it->second;
        const CosetLevel& Lg = lvl(g);
        SubSide s{SubgroupDescriptor(*tr.D, Lg, g), nullptr};
        // the oracle row: MP-3b's (E) rows, else MP-3's (B3) rows (M7 only)
        uint32_t vo = 0, tops = 0;
        bool mi = false;
        const uint64_t* nd = nullptr;
        if (MP3bHasDesc(tr.name, g)) {
            const MP3bDescRow& O = MP3bDesc(tr.name, g);
            vo = O.vertex_orbits; tops = O.top_orbits; mi = O.minus_I; nd = O.nd;
        } else {
            ENGINE_CHECK_MSG(tr.name == "M7", "RunMP3bLevelArms: no descriptor oracle row");
            const MP3DescRow& O = MP3Desc(g);
            vo = O.vertex_orbits; tops = O.top_orbits; mi = O.minus_I; nd = O.nd;
        }
        T.check(s.M.vertex_orbits == vo && s.M.D.top.size() == tops && s.M.minus_I_in == mi,
                "P7.1a " + tr.name + " " + g + ": vertex B-orbits " + Str(s.M.vertex_orbits) + ", top B-orbits " + Str(s.M.D.top.size()) + ", -I in B = oracle");
        const auto sc = SimplicialComplexProperty(s.M.D);
        T.check(sc.ok && InvariantOrder(s.M.D), "P7.1a " + tr.name + " " + g + ": (SC) and the invariant order on the B-descriptor");
        s.S = std::make_unique<InternedMPSet>(s.M.D, kmax);
        bool inB = true;
        for (const Mat22& x : s.S->elements()) inB = inB && Lg.contains(x);
        T.check(inB, "P7.1b " + tr.name + " " + g + ": every arrow of S_B lies in B (" + Str(s.S->elements().size()) + " elements)");
        std::vector<int64_t> c;
        bool cnt = true;
        for (uint32_t k = 0; k <= kmax; ++k) {
            c.push_back(static_cast<int64_t>(s.S->count_nondegenerate(k)));
            if (mi) cnt = cnt && s.S->count_nondegenerate(k) == nd[k];
        }
        if (mi) T.check(cnt, "P7.1c " + tr.name + " " + g + ": |S_k^nd(B)| = [Gamma:B] |S_k^nd(" + tr.name + ")|, k <= " + Str(kmax));
        uint64_t ch = 0;
        T.check(FormalDSquared(*s.S, 0, &ch) && FormalDSquared(*s.S, 1, &ch), "P7.1d " + tr.name + " " + g + ": d o d = 0 formally, q <= 1");
        std::cout << "  SUBGROUP_SET model=mp tri=" << tr.name << " B=" << g << " index=" << Lg.m() << " -I_in=" << s.M.minus_I_in
                  << " vertex_orbits=" << s.M.vertex_orbits << " tops=" << s.M.D.top.size() << " |S_0.." << kmax << "^nd|=" << JoinV(c)
                  << " elements=" << s.S->elements().size() << "\n";
        return subs.emplace(key, std::move(s)).first->second;
    };

    const std::vector<Cell> cells = {
        {"M7", "G0(2)", "G0(4)", 2, 2, 0, 2},   {"M7", "G0(2)", "G0(4)", 2, 2, 1, 2},  {"M7", "G0(3)", "G0(9)", 3, 2, 0, 2},
        {"M7", "G0(5)", "G0(25)", 5, 2, 0, 2},  {"R_Z", "Gamma", "Gamma^2", 2, 1, 0, 2}, {"R_Z", "G0(2)", "G0(4)", 3, 1, 2, 2},
        {"R_Z", "G0(2)", "G0(4)", 2, 2, 1, 2},  {"M7", "G1(4)", "G1(8)", 2, 1, 0, 1},  {"M7", "G1(4)", "G1(8)", 3, 1, 1, 1},
    };
    uint32_t ncells = 0, nagree = 0;
    for (const Cell& c : cells) {
        const auto t0 = std::chrono::steady_clock::now();
        const Tri& tr = tri_of(c.tri);
        const InternedMPSet& S = *tr.SG;
        const uint32_t ell = c.ell, e = c.e, n = c.n, d = n + 1, m = PowU(ell, e), qt = c.qt;
        const std::string at = " " + c.tri + " " + c.B + "<" + c.A + " " + RingName(ell, e) + " n=" + Str(n);
        const MP3bPairRow& O = MP3bPair(c.A, c.B, ell, e, n);
        const CosetLevel& LA = lvl(c.A);
        const CosetLevel& LB = lvl(c.B);
        const bool AisG = c.A == "Gamma";
        const ModuleRep Sym = SymRep(n, ell, e);
        const ModuleRep NA = AisG ? Sym : CoindRep(LA, n, ell, e);
        const ModuleRep NB = CoindRep(LB, n, ell, e);
        const uint32_t dA = NA.dim(), dB = NB.dim();
        bool all = true;
        auto gate = [&](bool ok, const std::string& what) { all = T.check(ok, what) && all; };

        // P7.2 C_I with the induced modules, q <= 3
        const CIRepSide ciA(CI, NA, 3, false), ciB(CI, NB, 3, false);
        {
            bool ok = true;
            for (uint32_t q = 0; q <= 3; ++q) ok = ok && ciA.lenH(q) == O.HA[q] && ciB.lenH(q) == O.HB[q];
            gate(ok, "P7.2 C_I(Coind_A), C_I(Coind_B) lengths H^0..3 = oracle" + at);
        }
        // P7.3 C(S_Gamma; Coind), Phi^* an isomorphism from C_I
        const RepSide XA(S, NA, qt, true), XB(S, NB, qt, true);
        std::vector<std::vector<Vec>> zA, zB;
        {
            bool ok = true;
            for (uint32_t q = 0; q <= qt; ++q) {
                zA.push_back(MapAll(PhiPullbackRep(*tr.Phi, S, q, CI.rank(q), NA), ciA.src[q].cocycles, m));
                zB.push_back(MapAll(PhiPullbackRep(*tr.Phi, S, q, CI.rank(q), NB), ciB.src[q].cocycles, m));
                const auto ia = XA.image(q, zA[q]), ib = XB.image(q, zB[q]);
                ok = ok && XA.lenH(q) == O.HA[q] && XB.lenH(q) == O.HB[q] && ia.cocycles && ib.cocycles &&
                     static_cast<int64_t>(ia.image()) == O.HA[q] && static_cast<int64_t>(ib.image()) == O.HB[q];
            }
            gate(ok, "P7.3 H^q(S_Gamma; Coind_A), H^q(S_Gamma; Coind_B) = oracle, Phi^* from C_I an isomorphism, q <= " + Str(qt) + at);
        }
        // P7.4 the nerves S_A (A != Gamma), S_B and their Shapiro comparisons C(S_Gamma; Coind_X) -> C(S_X)
        struct Nerv {
            const SubSide* s = nullptr;
            std::unique_ptr<RepSide> R;
            RefinementLabels Lg;
            StringMap U;
            std::vector<SparseRows> sh;
        };
        auto nerve = [&](const std::string& g, const ModuleRep& NX, const RepSide& Xsrc, const std::vector<std::vector<Vec>>& zX, const int64_t* H) {
            Nerv v;
            v.s = &sub(tr, g, 3);   // S_<=3 always (P7.1d needs S_3; first run: S_<=2 for the Gamma_1 pair threw)
            const CosetLevel& Lg = lvl(g);
            v.R = std::make_unique<RepSide>(*v.s->S, Sym, qt, true);
            v.Lg = LabelsToGamma(v.s->M);
            v.U = MapStrings(*v.s->S, S, v.Lg, qt);
            for (uint32_t q = 0; q <= qt; ++q)
                v.sh.push_back(LabelledPullbackRep(*v.s->S, S, v.Lg, v.U, q, d, NX.dim(), m,
                                                   [&Lg, n, ell, e](const Mat22& h) { return EvaluationMatrix(Lg, h, n, ell, e); }));
            bool hs = v.U.outside == 0, cm = true, bij = true, iso = true;
            for (uint32_t q = 0; q <= qt; ++q) {
                hs = hs && v.R->lenH(q) == H[q];
                if (q < qt) cm = cm && CommutesExactly(v.sh[q + 1], Xsrc.d[q], v.R->d[q], v.sh[q], m);
                if (v.s->M.minus_I_in) bij = bij && IsCoordinateBijection(v.sh[q]);
                const auto img = v.R->image(q, MapAll(v.sh[q], zX[q], m));
                iso = iso && img.cocycles && static_cast<int64_t>(img.image()) == H[q];
            }
            gate(hs, "P7.4 H^q(S_" + g + "; Sym^n) = oracle, S_" + g + " -> S_Gamma inside S(" + c.tri + "), q <= " + Str(qt) + at);
            gate(cm, "P7.4 the Shapiro comparison C(S_Gamma; Coind_" + g + ") -> C(S_" + g + ") is a cochain map, q < " + Str(qt) + at);
            if (v.s->M.minus_I_in) gate(bij, "P7.4 -I in " + g + ": the Shapiro comparison is a bijection of coordinates" + at);
            gate(iso, "P7.4 Shapiro comparison o Phi^* : H^q(C_I; Coind_" + g + ") -> H^q(S_" + g + ") an isomorphism, q <= " + Str(qt) + at);
            return v;
        };
        Nerv nB = nerve(c.B, NB, XB, zB, O.HB);
        Nerv nA;
        if (!AisG) nA = nerve(c.A, NA, XA, zA, O.HA);
        // P7.5 res / cor
        const ResCor rc = ResCorMaps(LA, LB, n, ell, e);
        std::vector<SparseRows> Res, Cor;
        {
            bool eqv = true;
            for (const Mat22& gm : {Mat22::S(), Mat22::T(), Mat22::I().neg()}) {
                eqv = eqv && MatMulSmall(rc.res, NA.rho(gm), dB, dA, dA, m) == MatMulSmall(NB.rho(gm), rc.res, dB, dB, dA, m);
                eqv = eqv && MatMulSmall(rc.cor, NB.rho(gm), dA, dB, dB, m) == MatMulSmall(NA.rho(gm), rc.cor, dA, dA, dB, m);
            }
            const bool corres = MatMulSmall(rc.cor, rc.res, dA, dB, dA, m) == IdentitySmall(dA, rc.index, m);
            gate(eqv && corres && rc.index == O.index, "P7.5 res, cor Gamma-maps over " + RingName(ell, e) + ", cor o res = [A:B] = " + Str(rc.index) + at);
            for (uint32_t q = 0; q <= qt; ++q) {
                Res.push_back(CoefficientMap(S, q, rc.res, dB, dA, m));
                Cor.push_back(CoefficientMap(S, q, rc.cor, dA, dB, m));
            }
            bool cmap = true, cr = true;
            for (uint32_t q = 0; q < qt; ++q) {
                cmap = cmap && CommutesExactly(Res[q + 1], XA.d[q], XB.d[q], Res[q], m);
                cmap = cmap && CommutesExactly(Cor[q + 1], XB.d[q], XA.d[q], Cor[q], m);
            }
            for (uint32_t q = 0; q <= qt; ++q) cr = cr && SpEqual(SpMul(Cor[q], Res[q], m), CoefficientMap(S, q, IdentitySmall(dA, rc.index, m), dA, dA, m));
            gate(cmap && cr, "P7.5 res, cor cochain maps on C(S_Gamma; Coind), cor o res = [A:B] on the cochains" + at);
        }
        // P7.6 the length of the image of res on H^q: MP (q <= qt, via Phi^*) and C_I (q <= 3)
        std::vector<int64_t> rk, rkci;
        {
            bool ok = true;
            for (uint32_t q = 0; q <= qt; ++q) {
                const auto img = XB.image(q, MapAll(Res[q], zA[q], m));
                rk.push_back(img.cocycles ? static_cast<int64_t>(img.image()) : -1);
                ok = ok && rk[q] == O.rank[q];
            }
            for (uint32_t q = 0; q <= 3; ++q) {
                const auto cimg = ciB.image(q, MapAllDense(CICoefficientMap(CI, q, rc.res, dB, dA, m), ciA.src[q].cocycles, m));
                rkci.push_back(static_cast<int64_t>(cimg.image()));
                ok = ok && rkci[q] == O.rank[q];
            }
            gate(ok, "P7.6 length of the image of res^A_B on H^q (MP q <= " + Str(qt) + ", C_I q <= 3) = oracle " + JoinV(rk) + "|" + JoinV(rkci) + at);
        }
        // P7.7 the cone of res: MP (q <= qt) and C_I (q <= 3) = oracle; the LES identity
        std::vector<int64_t> hc, hci;
        {
            std::vector<SparseRows> cone;
            std::vector<uint64_t> cdim;
            for (uint32_t q = 0; q <= qt; ++q) {
                cone.push_back(ConeDifferential(XA.d[q], Res[q], q ? &XB.d[q - 1] : nullptr, q ? static_cast<uint32_t>(XB.dimC[q - 1]) : 0, m));
                cdim.push_back(XA.dimC[q] + (q ? XB.dimC[q - 1] : 0));
            }
            hc = LengthsOf(cone, cdim, ell, e);
            std::vector<DenseBuf<uint32_t>> dcone;
            std::vector<uint32_t> ddim;
            for (uint32_t q = 0; q <= 3; ++q) {
                const DenseBuf<uint32_t> R = CICoefficientMap(CI, q, rc.res, dB, dA, m);
                dcone.push_back(DenseCone(ciA.d[q], R, q ? &ciB.d[q - 1] : nullptr, q ? ciB.dimC[q - 1] : 0, m));
                ddim.push_back(ciA.dimC[q] + (q ? ciB.dimC[q - 1] : 0));
            }
            hci = DenseLengthsOf(RingCtx(ell, e), dcone, ddim);
            bool ok = true;
            for (uint32_t q = 0; q <= qt; ++q) {
                const int64_t les = (q ? O.HB[q - 1] - rk[q - 1] : 0) + (O.HA[q] - rk[q]);
                ok = ok && hc[q] == O.rel[q] && hc[q] == les;
            }
            for (uint32_t q = 0; q <= 3; ++q) ok = ok && hci[q] == O.rel[q];
            gate(ok, "P7.7 H^q(Cone res) = H^q(A, B) = oracle = the LES (MP q <= " + Str(qt) + "), C_I cone q <= 3 " + JoinV(hc) + "|" + JoinV(hci) + at);
        }
        // P7.8 the forget-coset map S_B -> S_A (labels in A; for A = Gamma the representatives r)
        uint32_t exact = 0, labels = 0, central = 0;
        {
            const InternedMPSet& Sb = *nB.s->S;
            const InternedMPSet* SA = &S;
            const RepSide* NAn = &XA;
            RefinementLabels LBA;
            StringMap UBA;
            std::vector<std::vector<Vec>> srcA(qt + 1);
            if (AisG) {
                LBA = nB.Lg;
                UBA = nB.U;
                exact = static_cast<uint32_t>(LBA.h.size());
                for (uint32_t q = 0; q <= qt; ++q) srcA[q] = zA[q];
            } else {
                SA = nA.s->S.get();
                NAn = nA.R.get();
                LBA = ForgetCosetLabels(*tr.D, nB.s->M, LB, nA.s->M, LA, &exact);
                UBA = MapStrings(Sb, *SA, LBA, qt);
                for (uint32_t q = 0; q <= qt; ++q) srcA[q] = MapAll(nA.sh[q], zA[q], m);
            }
            labels = static_cast<uint32_t>(LBA.h.size());
            for (const Mat22& h : LBA.h) central += (h == Mat22::I() || h == Mat22::I().neg()) ? 1 : 0;
            std::vector<SparseRows> F;
            for (uint32_t q = 0; q <= qt; ++q)
                F.push_back(LabelledPullbackRep(Sb, *SA, LBA, UBA, q, d, d, m, [&Sym](const Mat22& h) { return Sym.rho(h); }));
            bool fcm = UBA.outside == 0;
            for (uint32_t q = 0; q < qt; ++q) fcm = fcm && CommutesExactly(F[q + 1], NAn->d[q], nB.R->d[q], F[q], m);
            std::vector<int64_t> frk;
            bool sq = true;
            for (uint32_t q = 0; q <= qt; ++q) {
                const auto mapped = MapAll(F[q], srcA[q], m);
                const auto img = nB.R->image(q, mapped);
                frk.push_back(img.cocycles ? static_cast<int64_t>(img.image()) : -1);
                const auto other = MapAll(nB.sh[q], MapAll(Res[q], zA[q], m), m);
                sq = sq && nB.R->image(q, Sub(mapped, other, m)).image() == 0;
            }
            bool frok = fcm && sq;
            for (uint32_t q = 0; q <= qt; ++q) frok = frok && frk[q] == O.rank[q];
            std::cout << "  FORGET_COSET model=mp tri=" << c.tri << " " << c.B << "->" << c.A << " " << RingName(ell, e) << " n=" << n << " labels=" << labels
                      << " (u=I: " << exact << ", u=-I: " << (labels - exact) << ", central: " << central << ") cochain_map=" << fcm
                      << " rank H^0.." << qt << "=" << JoinV(frk) << " square_on_H=" << sq << "\n";
            gate(frok, "P7.8 the forget-coset map S_B -> S_A: a cochain map, rank on H^q = res's (oracle), = Shapiro o res on H^q" + at);
            // P7.9 (-I not in A, F-MP3 F3): the u = -I branch of ForgetCosetLabels is reached, and is load-bearing
            const MP3bForcedRow& Fr = MP3bForced();
            if (c.tri == Fr.tri && c.A == Fr.A && c.B == Fr.B) {
                gate(!LA.contains(Mat22::I().neg()) && labels - exact >= Fr.at_least,
                     "P7.9 -I not in A: forget-coset labels with u = -I: " + Str(labels - exact) + " >= " + Str(Fr.at_least) + at);
                if (ell == 3 && n == 1) {
                    // NC: every u = -I label h replaced by -h = r^B (r^A)^{-1} (the u = I candidate, not in A)
                    RefinementLabels Lbad = LBA;
                    uint32_t flipped = 0;
                    bool notinA = true;
                    for (size_t t = 0; t < Lbad.h.size(); ++t) {
                        const Mat22 h0 = nB.s->M.r[t] * nA.s->M.r[Lbad.coarse[t]].inv_unimodular();
                        if (Lbad.h[t] == h0) continue;
                        Lbad.h[t] = Lbad.h[t].neg();
                        notinA = notinA && !LA.contains(Lbad.h[t]);
                        ++flipped;
                    }
                    const StringMap Ubad = MapStrings(Sb, *SA, Lbad, qt);
                    std::cout << "  NC P7.9b model=mp " << c.B << "->" << c.A << " u=-I labels replaced by their u=I candidates (" << flipped
                              << ", none in A=" << notinA << "): strings outside S_A=" << Ubad.outside << " -> " << (Ubad.outside > 0 && notinA ? "NC fires" : "NC SILENT")
                              << "\n";
                    gate(flipped == labels - exact && notinA && Ubad.outside > 0,
                         "P7.9b NC: the u = -I labels replaced by r^B (r^A)^{-1} (not in A) leave S_A -- the u = -I branch is load-bearing");
                }
            }
        }
        ++ncells;
        nagree += all ? 1 : 0;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "  PAIR_B model=mp tri=" << c.tri << " " << c.B << "<" << c.A << " " << RingName(ell, e) << " n=" << n << " [A:B]=" << rc.index
                  << " H(A)=" << JoinV(std::vector<int64_t>(O.HA, O.HA + 4)) << " H(B)=" << JoinV(std::vector<int64_t>(O.HB, O.HB + 4))
                  << " rank_res=" << JoinV(rk) << "|CI" << JoinV(rkci) << " H(A,B)=" << JoinV(hc) << "|CI" << JoinV(hci) << " agree=" << all
                  << " wall_s=" << std::fixed << std::setprecision(2) << s << std::defaultfloat << "\n";
    }
    std::cout << "ORACLE mp3b_oracle.out cells=" << ncells << " agree=" << nagree << "\n";
    *cells_out = ncells;
    *agree_out = nagree;
}

}  // namespace orbifold_mp
