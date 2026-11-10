// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, LES-1) HeckeLevelNormalityValidation -- see the header, [C] v1.22.0-pre.3 sec:nebles and
// docs/implementation/slot22/LES1_predictions.md (committed before the first run).
#include "HeckeLevelNormalityValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeGammaH.h"
#include "HeckeGammaHTower.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNSparse.h"
#include "HeckeLevelPair.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using oc_util::Seconds;
using engine::scalar::gcd;
using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::level;
using namespace hecke::level::les;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 60) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

uint64_t ModNN(int64_t x, uint64_t m) { const int64_t r = x % static_cast<int64_t>(m); return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(m) : r); }
std::string YN(bool b) { return b ? "yes" : "no"; }

const Mat22 kMinusI{-1, 0, 0, -1};
const Mat22 kU = Mat22::ST();

// ---- levels ---------------------------------------------------------------------------------------------------
// (slot 23, G-EVT23) the level handles moved to HeckeLevelPair.h (les::LevelHandle, Gamma0Handle, GammaHHandle, ...),
// shared with the golden harness's LESLEVEL point; these are forwarders.
using Level = LevelHandle;
Level Make(std::unique_ptr<CosetLevel> l) { return MakeLevel(std::move(l)); }
Level G0(uint64_t N) { return Gamma0Handle(N); }
// Gamma_H(N), H = { u in G_N : pred(u) } (checked to be a subgroup: the closure of the set is the set).
Level GH(uint64_t N, const std::function<bool(uint64_t)>& pred) { return GammaHHandle(N, pred); }
Level G1(uint64_t N) { return Gamma1Handle(N); }
// Gamma_1(N') cap Gamma_0(L), N' | L: H = { u in G_L : u = 1 mod N' }.
Level G1cap0(uint64_t Np, uint64_t L) { return Gamma1Cap0Handle(Np, L); }
// mu-type: H = { u in G_L : u mod p^r in mu_{p-1} } (Gamma^t_r cap Gamma_0(L), p^r || L).
Level Gmu(uint64_t pr, uint64_t p, uint64_t L) {
    return GH(L, [pr, p](uint64_t u) { return powmod<uint64_t>(u % pr, p - 1, pr) == 1 % pr; });
}

// ---- characters over F_ell ------------------------------------------------------------------------------------
CharacterDatum Chi(uint64_t N, uint32_t p, const std::function<uint32_t(uint64_t)>& f, const std::string& label) {
    CharacterDatum c;
    c.N = N;
    c.table.assign(N, 0);
    for (uint64_t u = 0; u < N; ++u)
        if (N == 1 || gcd<uint64_t>(u, N) == 1) c.table[u] = f(u) % p;
    c.label = label;
    return c;
}
CharacterDatum Trivial(uint64_t N, uint32_t p) { return Chi(N, p, [](uint64_t) { return 1u; }, "1"); }
CharacterDatum Legendre5(uint64_t N, uint32_t p) {   // (./5) on (Z/N)^x, 5 | N
    return Chi(N, p, [p](uint64_t u) { const uint64_t r = u % 5; return (r == 1 || r == 4) ? 1u : p - 1; }, "(./5)");
}
CharacterDatum Chi4mod5(uint64_t N) {                // the order-4 character x -> x mod 5 into F_5, 5 | N
    return Chi(N, 5, [](uint64_t u) { return static_cast<uint32_t>(u % 5); }, "chi_4");
}
CharacterDatum TeichChi(uint64_t p, int64_t e) {     // omega^e on (Z/p^2)^x, values in F_p (P10's TameCharacterTable)
    const tower::ZK R(static_cast<uint32_t>(p), 1);
    CharacterDatum c;
    c.N = p * p;
    c.table = tower::TameCharacterTable(p * p, R, e);
    c.label = "omega^" + std::to_string(((e % static_cast<int64_t>(p - 1)) + static_cast<int64_t>(p - 1)) % static_cast<int64_t>(p - 1));
    return c;
}

// ---- the quotient maps on a coset table ---------------------------------------------------------------------
// QuotientMapIso: moved to HeckeLevelPair.h (slot 23, G-EVT23).

// ============================================================================
// (N1) prop:gamma0normal
// ============================================================================
void ArmGamma0Normal(Checker& g) {
    // (slot 23, G-EVT23) the loop is les::Gamma0NormalityTable (HeckeLevelPair.h), shared with the golden LESLEVEL point
    uint32_t pairs = 0, normal = 0, mism = 0, iso_bad = 0;
    std::map<uint64_t, std::vector<uint64_t>> normal_list;
    for (const Gamma0NormalRow& r : Gamma0NormalityTable()) {
        const uint64_t N = r.N, M = r.M;
        ++pairs;
        const bool predicted = gcd<uint64_t>(N, 24) % M == 0;
        g.check(r.contained, "(N1) Gamma_0(" + std::to_string(M * N) + ") <= Gamma_0(" + std::to_string(N) + ")");
        if (r.normal != predicted) {
            ++mism;
            g.check(false, "(N1) Gamma_0(" + std::to_string(M * N) + ") normal in Gamma_0(" + std::to_string(N) + ") = " + YN(r.normal) +
                               ", predicted " + YN(predicted));
        }
        if (r.normal) {
            ++normal;
            normal_list[N].push_back(M);
            if (!r.iso) ++iso_bad;
            g.check(r.iso, "(N1) Gamma_0(" + std::to_string(N) + ") / Gamma_0(" + std::to_string(M * N) + ") = Z/" + std::to_string(M) +
                               " via a c / N mod M");
        }
    }
    g.check(pairs == 1380 && normal == 89 && mism == 0, "(N1) 1380 pairs, 89 normal, 0 mismatches");
    std::cout << "  GAMMA0_NORMAL pairs=" << pairs << " normal=" << normal << " mismatches=" << mism << " quotient_map_bad=" << iso_bad
              << " (predicted 1380 / 89 / 0 / 0, [C] prop:gamma0normal)\n";
    std::ostringstream os;
    for (const auto& [N, Ms] : normal_list) {
        os << " " << N << ":";
        for (size_t k = 0; k < Ms.size(); ++k) os << (k ? "," : "") << Ms[k];
    }
    std::cout << "  normal pairs (N: M list):" << os.str() << "\n";
}

// ============================================================================
// (N2) the Gamma_1 statements of prop:hidanormalsteps
// ============================================================================
void ArmGamma1Steps(Checker& g) {
    // (slot 23, G-EVT23) the loop is les::Gamma1StepsTable (HeckeLevelPair.h), shared with the golden LESLEVEL point
    uint32_t pairs = 0, ok_a = 0, ok_b = 0, ok_c = 0;
    for (const Gamma1StepRow& r : Gamma1StepsTable()) {
        ++pairs;
        const std::string tag = "(N2) N'=" + std::to_string(r.Np) + " M=" + std::to_string(r.M);
        ok_a += r.a;
        g.check(r.a, tag + ": Gamma_1(N') cap Gamma_0(MN') normal in Gamma_1(N'), quotient Z/M via c/N' mod M");
        ok_b += r.b;
        g.check(r.b, tag + ": Gamma_1(MN') normal in Gamma_1(N') cap Gamma_0(MN'), quotient {d = 1 mod N'} = Z/M");
        ok_c += r.c;
        g.check(r.c, tag + ": Gamma_1(MN') not normal in Gamma_1(N'), index M^2");
    }
    g.check(pairs == 60, "(N2) 60 pairs");
    std::cout << "  GAMMA1_STEPS pairs=" << pairs << " G1(N')^G0(MN')|>G1(N') " << ok_a << " G1(MN')|>G1(N')^G0(MN') " << ok_b
              << " G1(MN')-not-normal-in-G1(N') " << ok_c << " (predicted 60 / 60 / 60 / 60)\n";
}

// ============================================================================
// (N3) the p-tower table of prop:hidanormalsteps
// ============================================================================
void ArmTower(Checker& g) {
    for (uint64_t p : {5ull, 7ull}) {
        const std::string P = "p=" + std::to_string(p);
        // Row 1: Gamma_0(pN) > Gamma_0(N) cap Gamma_1(p), normal, index p - 1, cyclic.
        for (uint64_t N : {1ull, 2ull, 3ull}) {
            const Level A = G0(p * N), B = G1cap0(p, p * N);
            const QuotientGroup Q = QuotientOf(*B, B.perm, *A, A.perm);
            const bool ok = Q.contained && Q.normal && Q.index == p - 1 && Q.cyclic();
            g.check(ok, "(N3) " + P + " N=" + std::to_string(N) + " row 1: Gamma_0(N) cap Gamma_1(p) normal in Gamma_0(pN), (Z/p)^x cyclic of order p-1");
            std::cout << "  TOWER " << P << " N=" << N << " row1 normal=" << YN(Q.normal) << " index=" << Q.index << " cyclic=" << YN(Q.normal && Q.cyclic()) << "\n";
        }
        // Rows 2-4: r = 1 with N in {1, 2, 3}; r = 2 with N = 1.
        std::vector<std::pair<uint64_t, uint64_t>> rN = {{1, 1}, {1, 2}, {1, 3}, {2, 1}};
        for (const auto& [r, N] : rN) {
            uint64_t pr = 1;
            for (uint64_t k = 0; k < r; ++k) pr *= p;
            const Level A = G1cap0(pr, pr * N);          // Gamma_1(p^r) cap Gamma_0(N)
            const Level C = G1cap0(pr, pr * p * N);      // ... cap Gamma_0(p^{r+1} N)
            const Level B = G1cap0(pr * p, pr * p * N);  // Gamma_1(p^{r+1}) cap Gamma_0(N)
            const QuotientGroup Q2 = QuotientOf(*C, C.perm, *A, A.perm);
            const QuotientGroup Q3 = QuotientOf(*B, B.perm, *C, C.perm);
            const QuotientGroup Q4 = QuotientOf(*B, B.perm, *A, A.perm);
            const uint64_t prN = pr * N, L = pr * p * N;
            const bool r2 = Q2.contained && Q2.normal && Q2.index == p &&
                            QuotientMapIso(Q2, *C, p, [prN, p](const Mat22& v) { return ModNN(v.c / static_cast<int64_t>(prN), p); });
            const uint64_t prp = pr * p;
            const bool r3 = Q3.contained && Q3.normal && Q3.index == p &&
                            QuotientMapIso(Q3, *B, p, [pr, prp, p](const Mat22& v) { return ((ModNN(v.d, prp) + prp - 1) % prp) / pr % p; });
            const bool r4 = Q4.contained && !Q4.normal && Q4.index == p * p;
            const std::string t = "(N3) " + P + " r=" + std::to_string(r) + " N=" + std::to_string(N);
            g.check(r2, t + " row 2: Gamma_1(p^r) cap Gamma_0(p^{r+1}) normal in Gamma_1(p^r), Z/p via c/(p^r N) mod p");
            g.check(r3, t + " row 3: Gamma_1(p^{r+1}) normal in Gamma_1(p^r) cap Gamma_0(p^{r+1}), Z/p via d mod p^{r+1}");
            g.check(r4, t + " row 4: Gamma_1(p^{r+1}) not normal in Gamma_1(p^r), index p^2");
            (void)L;
            {   // audit R5: table sizes m_A = phi(p^r) psi(p^r) psi(N), m_C = p m_A, m_B = p^2 m_A
                const uint64_t psiN = N == 1 ? 1 : N == 2 ? 3 : 4;
                const uint64_t mA = (pr / p) * (p - 1) * (pr / p) * (p + 1) * psiN;
                g.check(A.perm.m == mA && C.perm.m == p * mA && B.perm.m == p * p * mA, t + ": table sizes m_A, m_C = p m_A, m_B = p^2 m_A");
            }
            std::cout << "  TOWER " << P << " r=" << r << " N=" << N << " row2 normal=" << YN(Q2.normal) << " index=" << Q2.index
                      << " row3 normal=" << YN(Q3.normal) << " index=" << Q3.index << " row4 normal=" << YN(Q4.normal) << " index=" << Q4.index
                      << " (m_A=" << A.perm.m << ", m_C=" << C.perm.m << ", m_B=" << B.perm.m << ")\n";
        }
        // Rows 5-8, N = 1.
        const uint64_t rmax = p == 5 ? 3 : 2;
        uint64_t pr = p;
        for (uint64_t r = 1; r <= rmax; ++r, pr *= p) {
            const uint64_t pr1 = pr * p;
            const Level Ar = G0(pr), Ar1 = G0(pr1);
            const Level Gt = Gmu(pr, p, pr);              // Gamma^t_r
            const Level Gt1 = Gmu(pr1, p, pr1);           // Gamma^t_{r+1}
            const Level Hr = Gmu(pr, p, pr1);             // Gamma_{H_{r+1,r}}(p^{r+1}) = Gamma^t_r cap Gamma_0(p^{r+1})
            const QuotientGroup Q5 = QuotientOf(*Gt, Gt.perm, *Ar, Ar.perm);
            uint64_t prm1 = 1;
            for (uint64_t k = 1; k < r; ++k) prm1 *= p;
            const bool r5 = Q5.contained && Q5.normal && Q5.index == prm1 && Q5.cyclic();
            // Row 6: Hr = Gt cap Gamma_0(p^{r+1}) (Hr in both, same index as the intersection); not normal in Gt, index p.
            const QuotientGroup Q6a = QuotientOf(*Hr, Hr.perm, *Gt, Gt.perm);
            const QuotientGroup Q6b = QuotientOf(*Hr, Hr.perm, *Ar1, Ar1.perm);
            const uint64_t inter = IntersectionIndex(Gt.perm, Ar1.perm);
            const bool r6eq = Q6a.contained && Q6b.contained && inter == Hr.perm.m;
            const bool r6 = Q6a.contained && !Q6a.normal && Q6a.index == p;
            const QuotientGroup Q7 = QuotientOf(*Gt1, Gt1.perm, *Hr, Hr.perm);
            const bool r7 = Q7.contained && Q7.normal && Q7.index == p && Q7.cyclic();
            const QuotientGroup Q8 = QuotientOf(*Ar1, Ar1.perm, *Ar, Ar.perm);
            const bool r8 = Q8.contained && !Q8.normal && Q8.index == p;
            const std::string t = "(N3) " + P + " r=" + std::to_string(r);
            {   // audit R5: m(Gamma^t_r) = (p+1) p^{2r-2}, m(Gamma_{H_{r+1,r}}) = (p+1) p^{2r-1}
                uint64_t p2r2 = 1;
                for (uint64_t k = 0; k + 2 < 2 * r + 0; ++k) p2r2 *= p;
                g.check(Gt.perm.m == (p + 1) * p2r2 && Hr.perm.m == (p + 1) * p2r2 * p, t + ": table sizes m(Gt_r) = (p+1)p^{2r-2}, m(H) = (p+1)p^{2r-1}");
            }
            g.check(r5, t + " row 5: Gamma^t_r normal in Gamma_0(p^r), cyclic of index p^{r-1}");
            g.check(r6eq, t + " row 6: Gamma^t_r cap Gamma_0(p^{r+1}) = Gamma_{H_{r+1,r}}(p^{r+1})");
            g.check(r6, t + " row 6: Gamma_{H_{r+1,r}}(p^{r+1}) not normal in Gamma^t_r, index p");
            g.check(r7, t + " row 7: Gamma^t_{r+1} normal in Gamma_{H_{r+1,r}}(p^{r+1}), Z/p");
            g.check(r8, t + " row 8: Gamma_0(p^{r+1}) not normal in Gamma_0(p^r), index p");
            std::cout << "  TOWER " << P << " r=" << r << " row5 normal=" << YN(Q5.normal) << " index=" << Q5.index << " row6 equal="
                      << YN(r6eq) << " normal=" << YN(Q6a.normal) << " index=" << Q6a.index << " row7 normal=" << YN(Q7.normal)
                      << " index=" << Q7.index << " row8 normal=" << YN(Q8.normal) << " index=" << Q8.index << " (m(Gt_r)=" << Gt.perm.m
                      << ", m(H)=" << Hr.perm.m << ", m(Gt_{r+1})=" << Gt1.perm.m << ")\n";
        }
    }
}

// ============================================================================
// (N4) level one, rmk:normalN1
// ============================================================================
void ArmLevelOne(Checker& g) {
    const Level Gam = G0(1);
    const Level G2 = Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(2)));
    const Level G3 = Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(3)));
    struct Expect { const Level* L; uint32_t index; bool mI, S, U, U2, T; uint32_t K; };
    const Expect ex[2] = {{&G2, 2, true, false, true, true, false, 3}, {&G3, 3, true, true, false, false, false, 8}};
    for (const Expect& e : ex) {
        const CosetLevel& L = **e.L;
        const auto* CL = dynamic_cast<const CongruenceLevel*>(&L);
        const bool mI = L.contains(kMinusI), S = L.contains(Mat22::S()), U = L.contains(kU), U2 = L.contains(kU * kU), T = L.contains(Mat22::T());
        const bool content = mI == e.mI && S == e.S && U == e.U && U2 == e.U2 && T == e.T;
        const QuotientGroup Q = QuotientOf(L, e.L->perm, *Gam, Gam.perm);
        const bool nq = Q.contained && Q.normal && Q.index == e.index && Q.cyclic();
        g.check(L.m() == e.index && CL != nullptr && CL->K_order() == e.K, "(N4) " + L.name() + ": index and |K|");
        g.check(content, "(N4) " + L.name() + ": members -I, S, U, U^2, T as rmk:normalN1");
        g.check(nq, "(N4) " + L.name() + " normal in Gamma with cyclic quotient Z/" + std::to_string(e.index));
        std::cout << "  LEVEL1 " << L.name() << " index=" << L.m() << " |K|=" << (CL ? CL->K_order() : 0) << " members -I:" << YN(mI)
                  << " S:" << YN(S) << " U:" << YN(U) << " U^2:" << YN(U2) << " T:" << YN(T) << " normal=" << YN(Q.normal)
                  << " cyclic=" << YN(Q.normal && Q.cyclic()) << "\n";
    }
    const uint64_t i23 = IntersectionIndex(G2.perm, G3.perm);
    g.check(i23 == 6, "(N4) [Gamma : Gamma^2 cap Gamma^3] = 6 (Gamma^ab = Z/12 surjects onto Z/6)");
    std::cout << "  LEVEL1 [Gamma : Gamma^2 cap Gamma^3] = " << i23 << " (predicted 6)\n";
    for (uint64_t M : {2ull, 3ull, 4ull, 5ull, 7ull}) {
        const Level GM = Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Principal(M)));
        uint64_t order = M * M * M;
        uint64_t num = 1, den = 1;
        for (uint64_t q = 2; q <= M; ++q)
            if (M % q == 0 && hecke::IsPrimeU64(q)) { num *= q * q - 1; den *= q * q; }
        order = order / den * num;
        const QuotientGroup Q = QuotientOf(*GM, GM.perm, *Gam, Gam.perm);
        const TorsionContent t = TorsionOf(GM.perm);
        const bool ok = GM.perm.m == order && Q.contained && Q.normal && Q.index == order && !Q.abelian();
        const bool tf = M >= 3 ? t.torsion_free() : (t.minus_I && !t.order3 && !t.order4);
        g.check(ok, "(N4) Gamma(" + std::to_string(M) + ") normal in Gamma, |G| = |SL_2(Z/M)| = " + std::to_string(order) + ", non-abelian");
        g.check(tf, "(N4) Gamma(" + std::to_string(M) + ") torsion: " + (M >= 3 ? std::string("torsion-free") : std::string("only -I")));
        std::cout << "  LEVEL1 Gamma(" << M << ") index=" << GM.perm.m << " normal=" << YN(Q.normal) << " |G|=" << Q.index
                  << " abelian=" << YN(Q.normal && Q.abelian()) << " torsion_free=" << YN(t.torsion_free()) << "\n";
    }
    // Gamma(M) <= Gamma_1(M) (M = 4, 5).
    for (uint64_t M : {4ull, 5ull}) {
        const Level GM = Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Principal(M)));
        const Level A = G1(M);
        const QuotientGroup Q = QuotientOf(*GM, GM.perm, *A, A.perm);
        g.check(Q.contained, "(N4) Gamma(" + std::to_string(M) + ") <= Gamma_1(" + std::to_string(M) + ")");
    }
}

// ============================================================================
// (N5) prop:modeltorsion
// ============================================================================
void ArmModelTags(Checker& g) {
    // (slot 23, G-EVT23) the eight levels and their tags are les::ModelTagTable (HeckeLevelPair.h), shared with the golden
    // LESLEVEL point; the expectations stay here, matched by name.
    struct Row { std::string name; bool mI, o3; const char* t2; const char* t3; };
    std::vector<Row> rows;
    rows.push_back({"Gamma", true, true, "bredon", "bredon"});
    rows.push_back({"Gamma_0(2)", true, false, "bredon", "group"});
    rows.push_back({"Gamma_0(4)", true, false, "bredon", "group"});
    rows.push_back({"Gamma_0(3)", true, true, "bredon", "bredon"});
    rows.push_back({"Gamma_0(9)", true, false, "bredon", "group"});
    rows.push_back({"Gamma_1(4)", false, false, "group", "group"});
    rows.push_back({"Gamma^2", true, true, "bredon", "bredon"});
    rows.push_back({"Gamma^3", true, false, "bredon", "group"});
    const std::vector<ModelTagRow> table = ModelTagTable();
    ENGINE_CHECK_MSG(table.size() == rows.size(), "(N5) ModelTagTable has the arm's eight rows");
    for (size_t k = 0; k < rows.size(); ++k) {
        const Row& r = rows[k];
        ENGINE_CHECK_MSG(table[k].name == r.name, "(N5) ModelTagTable row order");
        const TorsionContent& t = table[k].t;
        const std::string t2 = ModelTagName(table[k].t2), t3 = ModelTagName(table[k].t3), t5 = ModelTagName(table[k].t5);
        const bool ok = t.minus_I == r.mI && t.order3 == r.o3 && t2 == r.t2 && t3 == r.t3 && t5 == "group";
        g.check(ok, "(N5) model tags of " + r.name);
        std::cout << "  MODEL_TAG " << r.name << " -I in: " << YN(t.minus_I) << " order-3 element: " << YN(t.order3)
                  << " -> ell=2: model=" << t2 << " ell=3: model=" << t3 << " ell>=5: model=" << t5 << "\n";
    }
    // Gamma_0(N) has an element of order 3 iff x^2 + x + 1 = 0 mod N is solvable (N <= 60).
    uint32_t agree = 0;
    for (uint64_t N = 1; N <= 60; ++N) {
        bool solv = false;
        for (uint64_t x = 0; x < N || (N == 1 && x == 0); ++x) {
            if ((x * x + x + 1) % N == 0) { solv = true; break; }
            if (N == 1) break;
        }
        const Level L = G0(N);
        const bool o3 = TorsionOf(L.perm).order3;
        agree += o3 == solv;
        g.check(o3 == solv, "(N5) Gamma_0(" + std::to_string(N) + ") order-3 element iff x^2+x+1 = 0 mod N solvable");
    }
    std::cout << "  MODEL_TAG order-3 criterion on Gamma_0(N), N <= 60: " << agree << "/60 agree\n";
    // Torsion-free groups (every ell: model = group).
    uint32_t tf = 0, tf_n = 0;
    auto torsion_free = [&](const std::string& name, const Level& L) {
        ++tf_n;
        const TorsionContent t = TorsionOf(L.perm);
        tf += t.torsion_free();
        g.check(t.torsion_free() && ModelTagOf(t, 2) == ModelTag::kGroup && ModelTagOf(t, 3) == ModelTag::kGroup, "(N5) " + name + " torsion-free");
    };
    for (uint64_t N = 4; N <= 24; ++N) torsion_free("Gamma_1(" + std::to_string(N) + ")", G1(N));
    torsion_free("Gamma(3)", Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Principal(3))));
    torsion_free("Gamma(4)", Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Principal(4))));
    torsion_free("Gamma_1(4) cap Gamma_0(8)", G1cap0(4, 8));
    torsion_free("Gamma_1(5) cap Gamma_0(25)", G1cap0(5, 25));
    std::cout << "  MODEL_TAG torsion-free: " << tf << "/" << tf_n << "\n";
}

// ============================================================================
// (L) RestrictionFibreMap = LevelChangeMap; TwistedMonomialMatrix = MonomialOf / MonomialChiOf
// ============================================================================
void ArmLevelChangeOracle(Checker& g) {
    const std::vector<std::pair<uint64_t, uint64_t>> pairs = {{1, 5}, {1, 7}, {1, 11}, {1, 13}, {5, 25}, {2, 4}, {3, 9},
                                                              {7, 49}, {11, 121}, {2, 6}, {1, 6}, {3, 12}, {4, 12}};
    uint64_t cells = 0, diff_res = 0, diff_tr = 0;
    for (const auto& [M, N] : pairs) {
        const Gamma0LevelN big(M), small(N);
        for (uint32_t ell : {5u, 7u})
            for (uint32_t n : {0u, 4u}) {
                const engine::ModCtx<uint32_t> ctx(ell);
                const RestrictionFibreMap rf(big, CharacterDatum::Trivial(ell), small, CharacterDatum::Trivial(ell), n, ctx);
                const LevelChangeMap lr(big, small, LevelChangeKind::kRestriction, n, ctx), lt(big, small, LevelChangeKind::kTrace, n, ctx);
                const std::vector<uint32_t> a = rf.alpha().dense(), b = lr.dense(), c = rf.trace().dense(), d = lt.dense();
                uint64_t dr = a.size() == b.size() ? 0 : UINT64_MAX, dt = c.size() == d.size() ? 0 : UINT64_MAX;
                if (dr == 0) for (size_t k = 0; k < a.size(); ++k) dr += a[k] != b[k];
                if (dt == 0) for (size_t k = 0; k < c.size(); ++k) dt += c[k] != d[k];
                bool fib = true;
                for (uint32_t i = 1; i <= small.m(); ++i) fib = fib && rf.fibre()[i] == lr.fibre()[i];
                ++cells;
                diff_res += dr;
                diff_tr += dt;
                const std::string t = "(L) Gamma_0(" + std::to_string(N) + ") <= Gamma_0(" + std::to_string(M) + ") ell=" + std::to_string(ell) +
                                      " n=" + std::to_string(n);
                g.check(dr == 0 && fib, t + ": RestrictionFibreMap alpha = LevelChangeMap kRestriction entrywise");
                g.check(dt == 0, t + ": RestrictionFibreMap trace = LevelChangeMap kTrace entrywise");
            }
    }
    std::cout << "  LEVELCHANGE_ORACLE cells=" << cells << " differing_entries res=" << diff_res << " trace=" << diff_tr << " (predicted 0 / 0)\n";

    // TwistedMonomialMatrix at chi = 1 against the shipped MonomialOf, gamma in {S, T, -I}, n in {0, 4}.
    std::vector<std::pair<std::string, Level>> lv;
    lv.emplace_back("Gamma_0(25)", G0(25));
    lv.emplace_back("Gamma_1(11)", G1(11));
    lv.emplace_back("Gamma^t_2(5)", Gmu(25, 5, 25));
    lv.emplace_back("Gamma(5)", Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Principal(5))));
    lv.emplace_back("Gamma^3", Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(3))));
    uint64_t mono_diff = 0, mono_cells = 0;
    for (const auto& [name, L] : lv)
        for (uint32_t n : {0u, 4u})
            for (const Mat22& gm : {Mat22::S(), Mat22::T(), kMinusI}) {
                const engine::ModCtx<uint32_t> ctx(7);
                const std::vector<uint32_t> a = TwistedMonomialMatrix(*L, CharacterDatum::Trivial(7), gm, n, ctx).dense();
                const std::vector<uint32_t> b = MonomialOf(*L, gm, n, ctx).dense();
                uint64_t dd = a.size() == b.size() ? 0 : UINT64_MAX;
                if (dd == 0) for (size_t k = 0; k < a.size(); ++k) dd += a[k] != b[k];
                mono_diff += dd;
                ++mono_cells;
                g.check(dd == 0, "(L) TwistedMonomialMatrix(chi = 1) = MonomialOf on " + name + " n=" + std::to_string(n));
            }
    // ... and against P10's MonomialChiOf at n = 0, chi = omega^e on Gamma_0(p^2) and Gamma^t_2.
    uint64_t chi_diff = 0, chi_cells = 0;
    for (uint64_t p : {5ull, 7ull}) {
        const Level A = GH(p * p, [](uint64_t) { return true; });
        const Level B = Gmu(p * p, p, p * p);
        for (int64_t e = 0; e < static_cast<int64_t>(p - 1); e += 2) {
            const CharacterDatum chi = TeichChi(p, e);
            const engine::ModCtx<uint32_t> ctx(static_cast<uint32_t>(p));
            for (const Level* L : {&A, &B})
                for (const Mat22& gm : {Mat22::S(), Mat22::T(), kU}) {
                    const auto* GHL = dynamic_cast<const GammaHLevelN*>(L->lvl.get());
                    const FibreMatrix F = TwistedMonomialMatrix(**L, chi, gm, 0, ctx);
                    const tower::MonomialChi mc = tower::MonomialChiOf(*GHL, gm, chi.table);
                    uint64_t dd = 0;
                    for (uint32_t J = 1; J <= L->perm.m; ++J) {
                        const auto& row = F.row(J);
                        if (row.size() != 1 || row[0].first != mc.tgt[J] || row[0].second[0] != mc.scal[J]) ++dd;
                    }
                    chi_diff += dd;
                    ++chi_cells;
                    g.check(dd == 0, "(L) TwistedMonomialMatrix = MonomialChiOf on " + L->lvl->name() + " chi=" + chi.label);
                }
        }
    }
    std::cout << "  MONOMIAL_ORACLE MonomialOf cells=" << mono_cells << " differing=" << mono_diff << "; MonomialChiOf cells=" << chi_cells
              << " differing_rows=" << chi_diff << " (predicted 0 / 0)\n";
}

// ============================================================================
// (F) the fibre identities of obl:levelconeexact (i)
// ============================================================================
struct FibreCell {
    std::string label;
    const Level* A;
    CharacterDatum chiA;
    const Level* B;
    CharacterDatum chiB;
    uint32_t ell, n;
    bool expect_normal;
    std::vector<Mat22> extra_sigma;     // with the expected order in G (0: no expectation)
    std::vector<uint32_t> extra_order;
};

struct FibreTally {
    uint32_t cells = 0, normal_cells = 0, nm_gated = 0, nm_skipped = 0, failing_cells = 0, b_runs = 0;
    uint64_t identities = 0;
    std::vector<std::string> skipped;
    std::vector<std::pair<double, std::string>> walls;
};

constexpr uint64_t kNmCost = 60000000ull;   // alpha omega = Nm_G gated where m_B [A:B] (n+1)^3 <= 6 * 10^7

void RunFibreCell(Checker& g, const FibreCell& c, FibreTally& tally) {
    const auto tc = std::chrono::steady_clock::now();
    const engine::ModCtx<uint32_t> ctx(c.ell);
    const CosetLevel& A = **c.A;
    const CosetLevel& B = **c.B;
    const uint32_t d = c.n + 1, p = c.ell;
    RhoBlocks rb(ctx, c.n);
    const RestrictionFibreMap rf(A, c.chiA, B, c.chiB, c.n, ctx, {}, &rb);
    const FibreMatrix& al = rf.alpha();
    const FibreMatrix& om = rf.trace();
    const uint32_t fails0 = g.fails;
    const std::string t = "(F) " + c.label;
    std::vector<FibreMatrix> MB;
    uint64_t bad_eq = 0, bad_tr = 0;
    for (const Mat22& gm : {Mat22::S(), Mat22::T(), kMinusI}) {
        const FibreMatrix MA = TwistedMonomialMatrix(A, c.chiA, gm, c.n, ctx, &rb);
        MB.push_back(TwistedMonomialMatrix(B, c.chiB, gm, c.n, ctx, &rb));
        bad_eq += (al * MA).differing_entries(MB.back() * al);
        bad_tr += (MA * om).differing_entries(om * MB.back());
    }
    g.check(bad_eq == 0, t + ": alpha M_A(gamma) = M_B(gamma) alpha, gamma = S, T, -I");
    g.check(bad_tr == 0, t + ": M_A(gamma) omega = omega M_B(gamma)");
    g.check((om * al).differing_entries(FibreMatrix::Identity(A.m(), d, p, rf.index() % p)) == 0, t + ": omega alpha = [A:B]");
    // The retraction of lem:levelresmorph (1): the omega block at (a, i_a), i_a the first i over a.
    FibreMatrix P(A.m(), B.m(), d, p);
    std::vector<uint32_t> first(A.m() + 1, 0);
    for (uint32_t i = 1; i <= B.m(); ++i)
        if (!first[rf.fibre()[i]]) first[rf.fibre()[i]] = i;
    for (uint32_t a = 1; a <= A.m(); ++a)
        for (const auto& e : om.row(a))
            if (e.first == first[a]) P.add(a, e.first, e.second);
    g.check((P * al).differing_entries(FibreMatrix::Identity(A.m(), d, p)) == 0 && al.monomial(), t + ": alpha split injective (P alpha = 1), monomial");
    tally.identities += 4;
    const QuotientGroup Q = QuotientOf(B, c.B->perm, A, c.A->perm);
    g.check(Q.contained && Q.normal == c.expect_normal, t + ": normal = " + YN(c.expect_normal));
    if (Q.normal) {
        ++tally.normal_cells;
        std::vector<Mat22> sig;
        if (Q.index <= 12) {
            for (uint32_t pos = 1; pos < Q.index; ++pos) sig.push_back(B.rep(Q.classes[pos]));
        } else {
            for (const Mat22& s : {Mat22::S(), Mat22::T()})
                if (A.contains(s)) sig.push_back(s);
            sig.push_back(B.rep(Q.classes[1]));
        }
        for (const Mat22& s : c.extra_sigma) sig.push_back(s);
        uint64_t bad_R = 0, bad_Ra = 0, bad_ord = 0;
        std::vector<FibreMatrix> Rs;
        for (size_t k = 0; k < sig.size(); ++k) {
            const FibreMatrix R = ConjugationFibreMap(A, c.chiA, B, c.B->perm, sig[k], c.n, ctx, false, &rb).matrix();
            for (const FibreMatrix& M : MB) bad_R += (R * M).differing_entries(M * R);
            bad_Ra += (R * al).differing_entries(al);
            const uint32_t ord = Q.order_of(Q.pos_of(B.class_of(sig[k])));
            FibreMatrix Pw = R;
            for (uint32_t e = 1; e < ord; ++e) Pw = Pw * R;
            bad_ord += Pw.differing_entries(FibreMatrix::Identity(B.m(), d, p));
            const size_t ex = k + c.extra_sigma.size() - sig.size();
            if (k + c.extra_sigma.size() >= sig.size() && c.extra_order[ex] != 0)
                g.check(ord == c.extra_order[ex], t + ": the explicit sigma has order " + std::to_string(c.extra_order[ex]) + " in G");
            if (Rs.size() < 2) Rs.push_back(R);
        }
        g.check(bad_R == 0, t + ": R_sigma M_B(gamma) = M_B(gamma) R_sigma (" + std::to_string(sig.size()) + " sigma)");
        g.check(bad_Ra == 0, t + ": R_sigma alpha = alpha (alpha(Ind_A) is G-invariant)");
        g.check(bad_ord == 0, t + ": R_sigma^{ord sigma} = 1");
        tally.identities += 3;
        if (sig.size() >= 2) {
            const FibreMatrix Rst = ConjugationFibreMap(A, c.chiA, B, c.B->perm, sig[0] * sig[1], c.n, ctx, false, &rb).matrix();
            g.check((Rs[0] * Rs[1]).differing_entries(Rst) == 0, t + ": R_sigma R_tau = R_{sigma tau}");
            ++tally.identities;
        }
        // R_{sigma b} = R_sigma, b = S w_{1.S}^{-1} in B (a Schreier generator).
        // Audit R4: b := the first Schreier generator of B (j = 1..m_B, s in {S, T}) different from I, so that the check
        // runs at every normal cell (S w_{1.S}^{-1} is I on every table but Gamma^3).
        Mat22 b = Mat22::I();
        for (uint32_t j = 1; j <= B.m() && b == Mat22::I(); ++j)
            for (int si = 0; si < 2 && b == Mat22::I(); ++si) {
                const Mat22 sg = si == 0 ? Mat22::S() : Mat22::T();
                const uint32_t jj = si == 0 ? c.B->perm.S[j] : c.B->perm.T[j];
                b = B.rep(j) * sg * B.rep(jj).inv_unimodular();
            }
        if (B.contains(b) && !(b == Mat22::I())) {
            ++tally.b_runs;
            const FibreMatrix Rsb = ConjugationFibreMap(A, c.chiA, B, c.B->perm, sig[0] * b, c.n, ctx, false, &rb).matrix();
            g.check(Rsb.differing_entries(Rs[0]) == 0, t + ": R_{sigma b} = R_sigma (b in B)");
            ++tally.identities;
        }
        const uint64_t cost = static_cast<uint64_t>(B.m()) * Q.index * d * d * d;
        if (cost <= kNmCost) {
            FibreMatrix Nm(B.m(), B.m(), d, p);
            for (uint32_t pos = 0; pos < Q.index; ++pos) {
                const FibreMatrix R = ConjugationFibreMap(A, c.chiA, B, c.B->perm, B.rep(Q.classes[pos]), c.n, ctx, false, &rb).matrix();
                for (uint32_t i = 1; i <= B.m(); ++i)
                    for (const auto& e : R.row(i)) Nm.add(i, e.first, e.second);
            }
            Nm.canonical();
            g.check((al * om).differing_entries(Nm) == 0, t + ": alpha omega = Nm_G");
            ++tally.nm_gated;
            ++tally.identities;
        } else {
            ++tally.nm_skipped;
            tally.skipped.push_back(c.label);
        }
    }
    ++tally.cells;
    if (g.fails != fails0) ++tally.failing_cells;
    tally.walls.emplace_back(Seconds(tc), c.label);
}

void ArmFibreCells(Checker& g) {
    FibreTally T;
    std::map<std::string, Level> lv;
    auto L = [&](const std::string& key, const std::function<Level()>& mk) -> const Level* {
        auto it = lv.find(key);
        if (it == lv.end()) it = lv.emplace(key, mk()).first;
        return &it->second;
    };
    auto g0 = [&](uint64_t N) { return L("G0(" + std::to_string(N) + ")", [N] { return G0(N); }); };
    const Mat22 I = Mat22::I();
    (void)I;
    auto cell = [&](const std::string& label, const Level* A, CharacterDatum chiA, const Level* B, CharacterDatum chiB, uint32_t ell, uint32_t n,
                    bool normal, std::vector<Mat22> xs = {}, std::vector<uint32_t> xo = {}) {
        FibreCell c{label, A, std::move(chiA), B, std::move(chiB), ell, n, normal, std::move(xs), std::move(xo)};
        RunFibreCell(g, c, T);
    };
    auto S = [](uint64_t x) { return std::to_string(x); };
    // (i) Gamma_0(p) <= Gamma.
    for (uint64_t p : {5ull, 7ull, 11ull})
        for (uint32_t ell : {5u, 7u, 13u, 101u})
            for (uint32_t n : {0u, 2u, 4u, 10u})
                cell("(i) G0(" + S(p) + ")<G ell=" + S(ell) + " n=" + S(n), g0(1), Trivial(1, ell), g0(p), Trivial(p, ell), ell, n, false);
    // (i') non-split level-one cells.
    for (const auto& [p, ell] : std::vector<std::pair<uint64_t, uint32_t>>{{19, 5}, {29, 5}, {13, 7}})
        for (uint32_t n : {0u, 2u, 4u})
            cell("(i') G0(" + S(p) + ")<G ell=" + S(ell) + " n=" + S(n), g0(1), Trivial(1, ell), g0(p), Trivial(p, ell), ell, n, false);
    // (ii) Gamma_0(25) <= Gamma_0(5), chi' = chi o red.
    for (uint32_t ell : {5u, 7u})
        for (uint32_t n : {0u, 2u, 4u}) {
            cell("(ii) G0(25)<G0(5) chi=1 ell=" + S(ell) + " n=" + S(n), g0(5), Trivial(5, ell), g0(25), Trivial(25, ell), ell, n, false);
            cell("(ii) G0(25)<G0(5) chi=(./5) ell=" + S(ell) + " n=" + S(n), g0(5), Legendre5(5, ell), g0(25), Legendre5(25, ell), ell, n, false);
        }
    for (uint32_t n : {1u, 3u}) cell("(ii) G0(25)<G0(5) chi=chi_4 ell=5 n=" + S(n), g0(5), Chi4mod5(5), g0(25), Chi4mod5(25), 5, n, false);
    // (LES9) level-raising cells.
    cell("(LES9) G0(25)<G0(5) ell=11 n=0", g0(5), Trivial(5, 11), g0(25), Trivial(25, 11), 11, 0, false);
    cell("(LES9) G0(49)<G0(7) ell=5 n=0", g0(7), Trivial(7, 5), g0(49), Trivial(49, 5), 5, 0, false);
    cell("(LES9) G0(121)<G0(11) ell=7 n=0", g0(11), Trivial(11, 7), g0(121), Trivial(121, 7), 7, 0, false);
    // (iii) Gamma_1(11) normal in Gamma_0(11), sigma with d = 2 of order 10; (iii') the first Hida step at ell = p.
    const Level* g1_11 = L("G1(11)", [] { return G1(11); });
    for (uint32_t ell : {5u, 7u})
        for (uint32_t n : {0u, 2u, 4u})
            cell("(iii) G1(11)<|G0(11) ell=" + S(ell) + " n=" + S(n), g0(11), Trivial(11, ell), g1_11, Trivial(11, ell), ell, n, true,
                 {Mat22{6, 1, 11, 2}}, {10});
    for (uint64_t p : {5ull, 7ull}) {
        const Level* g1p = L("G1(" + S(p) + ")", [p] { return G1(p); });
        for (uint32_t n : {0u, 2u, 4u})
            cell("(iii') G1(" + S(p) + ")<|G0(" + S(p) + ") ell=" + S(p) + " n=" + S(n), g0(p), Trivial(p, static_cast<uint32_t>(p)), g1p,
                 Trivial(p, static_cast<uint32_t>(p)), static_cast<uint32_t>(p), n, true);
    }
    // (iv) ell in {2, 3}.
    const Level* g1_4 = L("G1(4)", [] { return G1(4); });
    const Level* gam2 = L("Gamma^2", [] { return Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(2))); });
    const Level* gam3 = L("Gamma^3", [] { return Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(3))); });
    for (uint32_t ell : {2u, 3u})
        for (uint32_t n = 0; n <= 4; ++n) {
            cell("(iv) G0(4)<|G0(2) ell=" + S(ell) + " n=" + S(n), g0(2), Trivial(1, ell), g0(4), Trivial(1, ell), ell, n, true);
            cell("(iv) G0(9)<|G0(3) ell=" + S(ell) + " n=" + S(n), g0(3), Trivial(1, ell), g0(9), Trivial(1, ell), ell, n, true);
            cell("(iv) G1(4)<|G0(4) ell=" + S(ell) + " n=" + S(n), g0(4), Trivial(1, ell), g1_4, Trivial(1, ell), ell, n, true);
            cell("(iv) Gamma^2<|G ell=" + S(ell) + " n=" + S(n), g0(1), Trivial(1, ell), gam2, Trivial(1, ell), ell, n, true);
            cell("(iv) Gamma^3<|G ell=" + S(ell) + " n=" + S(n), g0(1), Trivial(1, ell), gam3, Trivial(1, ell), ell, n, true);
        }
    // (LES7) the Gamma_1-type Z/p steps at ell = p, N in {1, 2, 3}, steps 1 and 2.
    for (uint64_t p : {5ull, 7ull})
        for (uint64_t N : {1ull, 2ull, 3ull}) {
            const Level* s1A = L("G1cap0(" + S(p) + "," + S(p * N) + ")", [p, N] { return G1cap0(p, p * N); });
            const Level* s1B = L("G1cap0(" + S(p) + "," + S(p * p * N) + ")", [p, N] { return G1cap0(p, p * p * N); });
            const Level* s2B = L("G1cap0(" + S(p * p) + "," + S(p * p * N) + ")", [p, N] { return G1cap0(p * p, p * p * N); });
            const uint32_t ell = static_cast<uint32_t>(p);
            for (uint32_t n : {0u, 2u, ell}) {
                cell("(LES7) p=" + S(p) + " N=" + S(N) + " step1 n=" + S(n), s1A, Trivial(1, ell), s1B, Trivial(1, ell), ell, n, true);
                cell("(LES7) p=" + S(p) + " N=" + S(N) + " step2 n=" + S(n), s1B, Trivial(1, ell), s2B, Trivial(1, ell), ell, n, true);
            }
        }
    // (LES8) the tower cells Gamma^t_2 normal in Gamma_0(p^2), n = 0, chi = omega^{a-2}, a even.
    for (uint64_t p : {5ull, 7ull}) {
        const Level* A = L("GHall(" + S(p * p) + ")", [p] { return GH(p * p, [](uint64_t) { return true; }); });
        const Level* B = L("Gt2(" + S(p) + ")", [p] { return Gmu(p * p, p, p * p); });
        for (int64_t a = 0; a < static_cast<int64_t>(p - 1); a += 2) {
            const CharacterDatum chi = TeichChi(p, a - 2);
            cell("(LES8) Gt2(" + S(p) + ")<|G0(" + S(p * p) + ") a=" + S(static_cast<uint64_t>(a)) + " chi=" + chi.label, A, chi, B, chi,
                 static_cast<uint32_t>(p), 0, true);
        }
    }
    // (LES8) Gamma(ell) normal in Gamma, G = SL_2(F_ell); and the split cross cells.
    for (uint64_t M : {5ull, 7ull}) {
        const Level* B = L("Gamma(" + S(M) + ")", [M] { return Make(std::make_unique<CongruenceLevel>(CongruenceLevel::Principal(M))); });
        for (uint32_t n = 0; n <= 8; ++n)
            cell("(LES8) Gamma(" + S(M) + ")<|G ell=" + S(M) + " n=" + S(n), g0(1), Trivial(1, static_cast<uint32_t>(M)), B,
                 Trivial(1, static_cast<uint32_t>(M)), static_cast<uint32_t>(M), n, true);
        const uint32_t ell = M == 5 ? 7 : 5;
        for (uint32_t n : {0u, 2u, 4u})
            cell("(LES8x) Gamma(" + S(M) + ")<|G ell=" + S(ell) + " n=" + S(n), g0(1), Trivial(1, ell), B, Trivial(1, ell), ell, n, true);
    }
    g.check(T.failing_cells == 0, "(F) every fibre cell clean");
    std::cout << "  FIBRE_IDENTITIES cells=" << T.cells << " normal_cells=" << T.normal_cells << " b_nontrivial=" << T.b_runs << " identities=" << T.identities
              << " Nm_gated=" << T.nm_gated << " Nm_skipped=" << T.nm_skipped << " failing_cells=" << T.failing_cells << "\n";
    std::sort(T.walls.begin(), T.walls.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    for (size_t k = 0; k < T.walls.size() && k < 5; ++k)
        std::cout << "    slowest cell " << k + 1 << ": " << T.walls[k].second << " wall=" << T.walls[k].first << " s\n";
    for (const std::string& s : T.skipped) std::cout << "    Nm skipped by the cost rule (m_B [A:B] (n+1)^3 > 6e7): " << s << "\n";
}

// ============================================================================
// (NEG) the fibre-level negative controls of obl:levelconeexact (ix)
// ============================================================================
uint64_t EquivarianceDefect(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B, const CharacterDatum& chiB,
                            const FibreMatrix& al, uint32_t n, const engine::ModCtx<uint32_t>& ctx) {
    uint64_t bad = 0;
    for (const Mat22& gm : {Mat22::S(), Mat22::T(), kMinusI})
        bad += (al * TwistedMonomialMatrix(A, chiA, gm, n, ctx)).differing_entries(TwistedMonomialMatrix(B, chiB, gm, n, ctx) * al);
    return bad;
}

struct Gamma1Holder {
    std::unique_ptr<GammaHLevelN> L = std::make_unique<GammaHLevelN>(5, std::vector<uint64_t>{});   // Gamma_1(5)
};

void ArmNegatives(Checker& g) {
    const Gamma0LevelN A5(5), B25(25), B3(3);
    // res_wrong_scalar: chi_4 (order 4) at ell = 5, n = 1.
    {
        const engine::ModCtx<uint32_t> ctx(5);
        RestrictionOptions o;
        o.neg_inverse_scalar = true;
        // FINDING at the first run (LES1_report.md): on the shipped Gamma_0 tables every h_i = v_i w_{a(i)}^{-1} of
        // Gamma_0(25) < Gamma_0(5) has a_{h_i} = 1 mod 5 (value set {1}; +-1 would already suffice, chi_4^2 = 1 on +-1), so the control is vacuous there.
        // Recorded (the gate is the iff: vacuous exactly when the value set is inside {1, 4}); the control is gated at Gamma_1(5) < Gamma_0(5),
        // where h_i = sigma_u carries a_{h_i} = u^{-1} over all of (Z/5)^x.
        const RestrictionFibreMap rf(A5, Chi4mod5(5), B25, Chi4mod5(25), 1, ctx, o);
        const uint64_t bad = EquivarianceDefect(A5, Chi4mod5(5), B25, Chi4mod5(25), rf.alpha(), 1, ctx);
        std::string vals;
        bool pm1 = true;
        {
            std::vector<uint8_t> seen(5, 0);
            for (uint32_t i = 1; i <= B25.m(); ++i) {
                const Mat22 h = B25.rep(i) * A5.rep(rf.fibre()[i]).inv_unimodular();
                seen[ModNN(h.a, 5)] = 1;
            }
            for (uint32_t u = 1; u < 5; ++u)
                if (seen[u]) { vals += (vals.empty() ? "" : ",") + std::to_string(u); pm1 = pm1 && (u == 1 || u == 4); }
        }
        // Gate (audit R2): the control is vacuous iff chi(a_{h_i})^2 = 1 for every i, i.e. (chi_4 of order 4 at N_A = 5)
        // iff every a_{h_i} = +-1 mod 5; a table that made it work (some a_{h_i} not +-1) must show a defect instead.
        g.check(pm1 ? bad == 0 : bad > 0, "(NEG) res_wrong_scalar at Gamma_0(25) < Gamma_0(5): vacuous iff every a_{h_i} = +-1 mod 5 (recorded, gated as an iff)");
        std::cout << "  NEG res_wrong_scalar at Gamma_0(25)<Gamma_0(5) vacuous (a_{h_i} mod 5 in {" << vals << "}, defect " << bad << ")\n";
        const Gamma1Holder g15;
        const RestrictionFibreMap rg(A5, Chi4mod5(5), *g15.L, Trivial(5, 5), 1, ctx, o);
        const uint64_t bad1 = EquivarianceDefect(A5, Chi4mod5(5), *g15.L, Trivial(5, 5), rg.alpha(), 1, ctx);
        const RestrictionFibreMap rg0(A5, Chi4mod5(5), *g15.L, Trivial(5, 5), 1, ctx);
        const uint64_t bad0 = EquivarianceDefect(A5, Chi4mod5(5), *g15.L, Trivial(5, 5), rg0.alpha(), 1, ctx);
        g.check(bad1 > 0 && bad0 == 0, "(NEG) res_wrong_scalar caught at Gamma_1(5) < Gamma_0(5), chi_4, ell = 5, n = 1 (the true alpha is clean)");
        std::cout << "  NEG res_wrong_scalar " << (bad1 > 0 && bad0 == 0 ? "caught" : "MISSED") << " at Gamma_1(5)<Gamma_0(5) (defect entries "
                  << bad1 << ", true alpha " << bad0 << ")\n";
        // With a real character the inverse is the character: no defect (the exception the text names).
        const engine::ModCtx<uint32_t> c7(7);
        const RestrictionFibreMap rr(A5, Legendre5(5, 7), B25, Legendre5(25, 7), 2, c7, o);
        const uint64_t bad_r = EquivarianceDefect(A5, Legendre5(5, 7), B25, Legendre5(25, 7), rr.alpha(), 2, c7);
        g.check(bad_r == 0, "(NEG) res_wrong_scalar with the real character (./5): no defect, as the text says");
    }
    // res_no_block: rho_n(h_i) -> I at Gamma_0(25) <= Gamma_0(5), ell = 7, n = 2.
    {
        const engine::ModCtx<uint32_t> ctx(7);
        RestrictionOptions o;
        o.neg_identity_block = true;
        const RestrictionFibreMap rf(A5, Trivial(5, 7), B25, Trivial(25, 7), 2, ctx, o);
        const uint64_t bad = EquivarianceDefect(A5, Trivial(5, 7), B25, Trivial(25, 7), rf.alpha(), 2, ctx);
        g.check(bad > 0, "(NEG) res_no_block caught");
        std::cout << "  NEG res_no_block " << (bad > 0 ? "caught" : "MISSED") << " (defect entries " << bad << ")\n";
    }
    auto refused = [&](const std::string& name, const std::string& code, const std::function<void()>& f) {
        std::string got = "none";
        try {
            f();
        } catch (const LevelPairRefusal& r) {
            got = r.code();
        }
        g.check(got == code, "(NEG) " + name + " refused by name " + code + " (got " + got + ")");
        std::cout << "  NEG " << name << " " << (got == code ? "caught" : "MISSED") << " (refusal " << got << ")\n";
    };
    const engine::ModCtx<uint32_t> c7(7);
    refused("incompatible_chi", "les_incompatible_character",
            [&] { (void)RestrictionFibreMap(A5, Legendre5(5, 7), B25, Trivial(25, 7), 0, c7); });
    refused("not_contained", "les_not_contained", [&] { (void)RestrictionFibreMap(A5, Trivial(5, 7), B3, Trivial(3, 7), 0, c7); });
    refused("character_level", "les_character_level", [&] { (void)RestrictionFibreMap(A5, Trivial(25, 7), B25, Trivial(25, 7), 0, c7); });
    // sigma_not_normalising: a class of Gamma_0(25) in Gamma_0(5) whose representative does not normalise Gamma_0(25).
    {
        const CosetPermutation pA = PermutationOf(A5), pB = PermutationOf(B25);
        const QuotientGroup Q = QuotientOf(B25, pB, A5, pA);
        g.check(Q.contained && !Q.normal && Q.witness != 0, "(NEG) Gamma_0(25) not normal in Gamma_0(5): a witness class");
        const Mat22 sigma = B25.rep(Q.witness == 0 ? 1 : Q.witness);
        refused("sigma_not_normalising", "les_not_normal",
                [&] { (void)ConjugationFibreMap(A5, Trivial(5, 7), B25, pB, sigma, 2, c7); });
        const FibreMatrix R = ConjugationFibreMap(A5, Trivial(5, 7), B25, pB, sigma, 2, c7, true).matrix();
        uint64_t bad = 0;
        for (const Mat22& gm : {Mat22::S(), Mat22::T()}) {
            const FibreMatrix M = TwistedMonomialMatrix(B25, Trivial(25, 7), gm, 2, c7);
            bad += (R * M).differing_entries(M * R);
        }
        g.check(bad > 0, "(NEG) the unchecked R_sigma formula for a non-normalising sigma is not Gamma-equivariant");
        std::cout << "  NEG sigma_not_normalising_formula " << (bad > 0 ? "caught" : "MISSED") << " (R M_B != M_B R in " << bad << " entries)\n";
    }
}

} // namespace

uint32_t HeckeLevelNormalityValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::cout << "HeckeLevelNormalityValidation ([C] v1.22.0-pre.3 sec:nebles, (LES1); LES1_predictions.md)\n";
    auto arm = [&](const char* name, const std::function<void()>& f) {
        const auto t = std::chrono::steady_clock::now();
        f();
        std::cout << "  ARM " << name << " wall=" << Seconds(t) << " s\n";
    };
    arm("N1", [&] { ArmGamma0Normal(g); });
    arm("N2", [&] { ArmGamma1Steps(g); });
    arm("N3", [&] { ArmTower(g); });
    arm("N4", [&] { ArmLevelOne(g); });
    arm("N5", [&] { ArmModelTags(g); });
    arm("L", [&] { ArmLevelChangeOracle(g); });
    arm("F", [&] { ArmFibreCells(g); });
    arm("NEG", [&] { ArmNegatives(g); });
    std::cout << "  peak_rss=" << oc_util::PeakRssMiB() << " MiB wall=" << Seconds(t0) << " s\n";
    std::cout << "LEVEL_NORMALITY checks=" << g.checks << " failing=" << g.fails << " verdict=" << (g.fails ? "FAIL" : "OK") << "\n";
    return g.fails;
}
