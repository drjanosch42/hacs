// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, WAKE-1) HeckeTameFamilyValidation -- see the header and
// docs/implementation/slot22/WAKE1_spec.md (v0.2: the predictions and the frozen oracle values this validation gates).
#include "HeckeTameFamilyValidation.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeCosetTables.h"
#include "HeckeGammaH.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"
#include "HeckeLevelNSparse.h"
#include "HeckeNebentypus.h"        // neb::Gamma1LevelN (read-only use: the H = {1} table oracle)
#include "HeckeOperatorFamily.h"    // hecke::tau_construction_mutex()
#include "HeckeTameFamily.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using oc_util::Seconds;
using engine::scalar::gcd;

using hecke::Mat22;
using namespace hecke::level;
using namespace hecke::tame;

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


// One ORACLE line per field (R-22b): the C++ value next to the frozen Python value; the agreement is a check.
template <class T>
void Oracle(Checker& g, const std::string& cell, const std::string& field, const T& cpp, const T& py) {
    const bool ok = cpp == py;
    std::cout << "  ORACLE " << cell << " " << field << " cpp=" << cpp << " py=" << py << " agree=" << (ok ? "yes" : "no") << "\n";
    g.check(ok, "ORACLE " + cell + " " + field);
}
std::string YN(bool b) { return b ? "yes" : "no"; }

// H = G_N: PrimeToPart for a prime not dividing phi(N).
GammaHLevelN FullH(uint64_t N) {
    uint64_t phi = 0;
    for (uint64_t u = 1; u < N; ++u) phi += gcd<uint64_t>(u, N) == 1 ? 1 : 0;
    uint64_t q = 2;
    while (phi % q == 0 || !hecke::IsPrimeU64(q)) ++q;
    return GammaHLevelN::PrimeToPart(N, q);
}

// ---- the fixed model (as HeckeLevelNHeilbronnValidation's helpers, which are TU-local) --------------------------
struct Model {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
};
Model& TheModel() {
    static Model m;
    static std::once_flag once;
    std::call_once(once, [] {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        m.K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        m.tess.reset(new hecke::SL2ZTessellationTables(*m.K));
    });
    return m;
}
struct HeckeDatum {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
};
HeckeDatum MakeDatum(uint64_t p, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    HeckeDatum d;
    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
    d.cosets.reset(new hecke::HeckeCosetTables(p, 1, coef, n));
    d.tau.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *d.cosets, coef, n));
    return d;
}
FpMat FromHMat(const HMat& A) {
    FpMat M(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) M.at(i, j) = static_cast<uint32_t>(A.at(i, j));
    return M;
}

// ============================================================================
// (T) the table
// ============================================================================
void ArmTable(Checker& g) {
    const auto t0 = std::chrono::steady_clock::now();
    // the action formula (j,u).gamma = (j.gamma, [u d(h_j(gamma))]) at the WAKE levels
    for (uint64_t N : {253ull, 737ull}) {
        const GammaHLevelN G = GammaHLevelN::PrimeToPart(N, 5);
        g.check(G.index_H() == 5 && G.contains_minus_one() && G.m() == 5 * G.level0().m(),
                "(T) Gamma_H(" + std::to_string(N) + "): [G_N : H] = 5, -1 in H, m = 5 m_0");
        bool ok = true;
        for (const Mat22& gam : {Mat22::S(), Mat22::T(), Mat22::ST()})
            for (uint32_t J = 1; J <= G.m(); ++J) {
                const uint32_t j = G.j_of(J);
                const uint64_t u = G.coset_reps()[G.r_of(J)];
                const uint32_t jg = G.level0().class_of(G.level0().rep(j) * gam);
                const Mat22 h = G.level0().rep(j) * gam * G.level0().rep(jg).inv_unimodular();
                const uint64_t dh = static_cast<uint64_t>(((h.d % static_cast<int64_t>(N)) + static_cast<int64_t>(N)) % static_cast<int64_t>(N));
                const uint32_t want = G.index(jg, G.coset_rank(u * dh % N));
                ok = ok && G.class_of(G.rep(J) * gam) == want && G.contains(G.rep(J) * gam * G.rep(want).inv_unimodular());
            }
        g.check(ok, "(T) Gamma_H(" + std::to_string(N) + "): (j,u).gamma = (j.gamma, [u d(h_j(gamma))]) for gamma in {S, T, ST} (prop:nebgammaH (2))");
        std::cout << "  TABLE N=" << N << " H=" << G.H_text() << " m=" << G.m() << " U=[" << G.coset_reps()[0];
        for (size_t i = 1; i < G.coset_reps().size(); ++i) std::cout << "," << G.coset_reps()[i];
        std::cout << "]\n";
    }
    // H = G_N is Gamma0LevelN's table, H = {1} Gamma1LevelN's (I2)
    for (uint64_t N : {11ull, 253ull}) {
        const GammaHLevelN G = FullH(N);
        const Gamma0LevelN L(N);
        bool eq = G.m() == L.m() && G.index_H() == 1;
        for (uint32_t j = 1; eq && j <= L.m(); ++j) eq = G.rep(j) == L.rep(j);
        g.check(eq, "(T) H = G_N at N = " + std::to_string(N) + ": the table is Gamma0LevelN's entry for entry");
    }
    {
        const GammaHLevelN G(11, {1});
        const hecke::neb::Gamma1LevelN L(11);
        bool eq = G.m() == L.m() && G.m() == 120;
        for (uint32_t J = 1; eq && J <= L.m(); ++J) eq = G.rep(J) == L.rep(J) && G.class_of(L.rep(J)) == J;
        g.check(eq, "(T) H = {1} at N = 11: the table is Gamma1LevelN's entry for entry (m = 120)");
        // N2: -1 not in H refused by name in the cusp routine
        bool refused = false;
        try {
            CuspsOf(G);
        } catch (const std::logic_error& e) {
            refused = std::string(e.what()).find("gammaH_minus_one") != std::string::npos;
        }
        g.check(refused && !G.contains_minus_one(), "(T) N2: H = {1} (no -1) refused by name gammaH_minus_one in CuspsOf");
    }
    // PrimeToPart at the WAKE and the P10 levels: |H| = phi(N) / |p-part|
    struct PT { uint64_t N, p, H, idx; };
    for (const PT& c : {PT{11, 5, 2, 5}, PT{253, 5, 44, 5}, PT{737, 5, 132, 5}, PT{25, 5, 4, 5}, PT{125, 5, 4, 25}, PT{49, 7, 6, 7},
                        PT{121, 11, 10, 11}, PT{33, 5, 4, 5}}) {
        const GammaHLevelN G = GammaHLevelN::PrimeToPart(c.N, c.p);
        g.check(G.H_elements().size() == c.H && G.index_H() == c.idx && G.contains_minus_one(),
                "(T) PrimeToPart(" + std::to_string(c.N) + ", " + std::to_string(c.p) + "): |H| = " + std::to_string(c.H) +
                    ", index " + std::to_string(c.idx) + ", -1 in H; H = " + G.H_text());
    }
    // the sub-transversal: Gamma1SubTransversal's matrices are in Gamma_H(N), distinct (Gamma_H cap Gamma_0(N p))-cosets
    {
        const GammaHLevelN G = GammaHLevelN::PrimeToPart(33, 5);
        for (uint64_t p : {2ull, 3ull, 11ull}) {
            const std::vector<Mat22> r = GammaHSubTransversal(G, p);
            bool distinct = r.size() == (33 % p ? p + 1 : p);
            for (size_t a = 0; a < r.size() && distinct; ++a)
                for (size_t b = a + 1; b < r.size() && distinct; ++b) {
                    const Mat22 x = r[a] * r[b].inv_unimodular();
                    distinct = !(x.c % static_cast<int64_t>(33 * p) == 0);
                }
            g.check(distinct, "(T) GammaHSubTransversal(Gamma_H(33), " + std::to_string(p) + "): " + std::to_string(r.size()) +
                                  " matrices in Gamma_H(33), pairwise distinct modulo Gamma_0(33 p)");
        }
    }
    std::cout << "  (T) " << Seconds(t0) << " s\n";
}

// ============================================================================
// (S) the shipped Heilbronn route at H = G_N
// ============================================================================
void ArmShipped(Checker& g) {
    const auto t0 = std::chrono::steady_clock::now();
    {
        const uint64_t N = 253;
        const engine::ModCtx<uint32_t> ctx(5);
        const GammaHLevelN G = FullH(N);
        const Gamma0LevelN L(N);
        const SparseCokernel QG = SparseCokernel::Build(G, 0, ctx, true);
        const SparseCokernel QL = SparseCokernel::Build(L, 0, ctx, true);
        g.check(PresentationDigest(QG) == PresentationDigest(QL) && QG.h1() == QL.h1(),
                "(S) N = 253, H = G_N: PresentationDigest equals the shipped Gamma0LevelN presentation's");
        const QuotientCoords QC(QG);
        for (uint64_t q : {2ull, 11ull, 23ull}) {
            const HMat ref = OnCokernelSparse(QL, HeilbronnOperator(L, q, 0, ctx));
            const FpMat mine = OnQuotient(QC, GammaHHeilbronn(G, q, 0, ctx));
            g.check(Equal(mine, FromHMat(ref)), "(S) N = 253, H = G_N: [" + std::string(N % q ? "T_" : "U_") + std::to_string(q) +
                                                    "] equals the shipped HeilbronnOperator's entrywise");
        }
    }
    {
        // n = 2 blocks at N = 11 over F_7: GammaHHeilbronn (H = G_N) = HeilbronnOperator, dense
        const engine::ModCtx<uint32_t> ctx(7);
        const GammaHLevelN G = FullH(11);
        const Gamma0LevelN L(11);
        for (uint64_t q : {2ull, 3ull, 11ull}) {
            const HeilbronnOperator H(L, q, 2, ctx);
            const GammaHHeilbronn X(G, q, 2, ctx);
            const std::vector<uint32_t> ref = H.dense();
            const uint32_t D = X.D();
            std::vector<uint64_t> mine(static_cast<size_t>(D) * D, 0);
            std::vector<std::pair<uint32_t, uint32_t>> img;
            for (uint32_t c = 0; c < D; ++c) {
                img.clear();
                X.image(c, img);
                for (const auto& e : img) mine[static_cast<size_t>(e.first) * D + c] += e.second;
            }
            bool eq = ref.size() == mine.size();
            for (size_t i = 0; eq && i < ref.size(); ++i) eq = ref[i] == mine[i] % 7;
            g.check(eq, "(S) N = 11, n = 2, F_7, H = G_N: the blocks rho_2(adj h_J(eps)) of " + X.label() + " equal HeilbronnOperator's");
        }
        // RelEscapes = 0 at Gamma_H(33), n = 2, F_7
        const GammaHLevelN G33 = GammaHLevelN::PrimeToPart(33, 5);
        const SparseCokernel Q = SparseCokernel::Build(G33, 2, ctx, true);
        uint32_t esc = 0;
        esc += RelEscapesSym(Q, GammaHHeilbronn(G33, 2, 2, ctx));
        esc += RelEscapesSym(Q, GammaHHeilbronn(G33, 3, 2, ctx));
        esc += RelEscapesSym(Q, GammaHHeilbronn(G33, 5, 2, ctx));
        esc += RelEscapesSym(Q, GammaHDiamond(G33, 2, 2, ctx));
        g.check(esc == 0, "(S) Gamma_H(33), n = 2, F_7: T_2, U_3, T_5, <2> map Rel into Rel (escapes " + std::to_string(esc) +
                              ", h = " + std::to_string(Q.h1()) + ")");
    }
    std::cout << "  (S) " << Seconds(t0) << " s\n";
}

// ============================================================================
// (M) the model route at Gamma_H(33)
// ============================================================================
void ArmModel(Checker& g) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t p = 5;
    const engine::ModCtx<uint32_t> ctx(p);
    const GammaHLevelN G = GammaHLevelN::PrimeToPart(33, 5);
    Model& M = TheModel();
    const uint32_t D = G.m();
    LocalSystemSimplicialComplex Ls(*M.K, InducedRelationsLevel(*M.K, G, 0, ctx), D, p);
    EquivariantCohomologyOnK Hm(&Ls);
    Hm.compute_cohomology();
    const HFrame f(&Hm, 1);
    const SparseCokernel Q = SparseCokernel::Build(G, 0, ctx, true);
    const QuotientCoords QC(Q);
    g.check(f.t() == Q.h1(), "(M) Gamma_H(33), F_5: dim H^1 of the model = dim Q (" + std::to_string(f.t()) + " vs " +
                                 std::to_string(Q.h1()) + ")");
    std::vector<FpMat> mod, coc;
    std::vector<std::string> names;
    for (uint64_t q : {2ull, 3ull, 11ull}) {
        const HeckeDatum dat = MakeDatum(q, p, 0);
        CorrespondenceSpec spec{&G, &G, dat.cosets.get(), dat.tau.get(), GammaHSubTransversal(G, q),
                                std::string(33 % q ? "T_" : "U_") + std::to_string(q)};
        VerifyTransversal(spec, 33 % q ? static_cast<uint32_t>(q + 1) : static_cast<uint32_t>(q));
        mod.push_back(FromHMat(InducedOnH(f, f, BuildLevelCorrespondence(ctx, spec, *M.K, 0, 1), spec.label)));
        coc.push_back(OnQuotient(QC, GammaHHeilbronn(G, q, 0, ctx)));
        names.push_back(spec.label);
    }
    mod.push_back(FromHMat(InducedOnH(f, f, GammaHDiamondOnModel(G, 2, *M.K, 0, ctx, 1), "<2>")));
    coc.push_back(OnQuotient(QC, GammaHDiamond(G, 2, 0, ctx)));
    names.push_back("<2>");
    // a combination T_2 + 2 U_3 + 3 U_11 + 4 <2>
    FpMat cm(mod[0].rows, mod[0].cols), cc(coc[0].rows, coc[0].cols);
    for (size_t i = 0; i < mod.size(); ++i) {
        cm = Add(cm, Scale(mod[i], i + 1, p), p);
        cc = Add(cc, Scale(coc[i], i + 1, p), p);
    }
    mod.push_back(cm);
    coc.push_back(cc);
    names.push_back("T_2 + 2U_3 + 3U_11 + 4<2>");
    for (size_t i = 0; i < mod.size(); ++i)
        g.check(mod[i].rows == coc[i].rows && Charpoly(mod[i], p) == Charpoly(coc[i], p),
                "(M) Gamma_H(33), F_5: charpoly of " + names[i] + " on the model (eq:levelcorr / fibre map) = on Q (Heilbronn / symbols)");
    std::cout << "  (M) Gamma_H(33): m=" << G.m() << " h=" << Q.h1() << " " << Seconds(t0) << " s\n";
}

// ============================================================================
// (W) the m-parts of Gamma_0(N) and +-Gamma_1(11)
// ============================================================================
void ArmMParts(Checker& g) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<uint64_t> loc{2, 3, 7, 13, 17, 19};
    struct WC { uint64_t N; bool full; uint32_t cusp, fullm; const char* tag; };
    for (const WC& c : {WC{11, true, 2, 3, "W-P1"}, WC{253, true, 4, 6, "W-P2/3"}, WC{737, true, 4, 6, "W-P2/3"},
                        WC{473, true, 6, 8, "W-P2"}, WC{2167, true, 6, 8, "W-P2"}}) {
        const GammaHLevelN G = FullH(c.N);
        const MPartReading r = RunMParts(G, 5, loc);
        const std::string cell = "Gamma_0(" + std::to_string(c.N) + ")";
        std::cout << "  MPART " << cell << " m=" << r.m << " h=" << r.h << " c=" << r.c << " cusp_m=" << r.cusp_m << " full_m=" << r.full_m
                  << " rk_T0=" << r.cusp_m / 2 << " dim<T_3>=" << r.dim_T3 << "\n";
        g.check(r.genus_ok && r.rel_boundary_ok && r.boundary_rank + 1 == r.c && r.rel_escapes == 0,
                "(W) " + cell + ": h = 2g + c - 1, delta(Rel) = 0, rank delta = c - 1, no Rel escape, every restriction stable");
        Oracle(g, cell, std::string(c.tag) + " cusp_m", r.cusp_m, c.cusp);
        Oracle(g, cell, std::string(c.tag) + " full_m", r.full_m, c.fullm);
        if (c.N == 253 || c.N == 737) {
            // N1 (wrong H = G_N): <T_3> acts through T°_{Gamma_0(N)}/5 (rank 2): dim <= 2, oracle 2; <3> = 1
            g.check(r.dim_T3 <= 2 && r.dim_D3_minus_1 == 0, "(W) N1 " + cell + ": dim <T_3> <= 2 != 10 and <3> = 1 (H = G_N)");
            Oracle(g, cell, "N1 dim<T_3>", r.dim_T3, 2u);
        }
    }
    {
        const GammaHLevelN G = GammaHLevelN::PrimeToPart(11, 5);   // = +-Gamma_1(11)
        const MPartReading r = RunMParts(G, 5, loc);
        std::cout << "  MPART +-Gamma_1(11) m=" << r.m << " h=" << r.h << " c=" << r.c << " g=" << r.twelve_g / 12 << " cusp_m=" << r.cusp_m
                  << " full_m=" << r.full_m << "\n";
        g.check(r.genus_ok && r.rel_boundary_ok && r.boundary_rank + 1 == r.c && r.rel_escapes == 0 && r.h == 11 && r.c == 10 &&
                    r.twelve_g == 12,
                "(W) +-Gamma_1(11): h = 11 = 2g + c - 1 with g = 1, c = 10; delta(Rel) = 0; rank delta = 9");
        Oracle(g, "+-Gamma_1(11)", "W-P5 cusp_m", r.cusp_m, 2u);
        Oracle(g, "+-Gamma_1(11)", "W-P5 full_m", r.full_m, 11u);
    }
    std::cout << "  (W) " << Seconds(t0) << " s\n";
}

// ============================================================================
// (H) the HPWWE cells
// ============================================================================
struct HCell {
    uint64_t l1;
    uint32_t m, h, c;            // the oracle's nsym, dim, cusps
    const Presentation* pres;
};

void PrintReading(const std::string& cell, const TameCellReading& r) {
    std::cout << "  TAME_CELL " << cell << " H=" << r.H << " m=" << r.m << " h=" << r.h << " c=" << r.c << " g=" << r.twelve_g / 12
              << " nu2=" << r.nu2 << " nu3=" << r.nu3 << " cusp_m=" << r.cusp_m << " full_m=" << r.full_m << " dimA=" << r.dimA
              << " full_alg=" << r.full_alg_dim << " inside=" << YN(r.members_inside) << " m_A^e=0:e=" << r.radical_nilpotency;
    if (r.v2_done()) std::cout << " dimA/J=" << r.red.dim_quotient << " case=" << r.red.case_name() << " surj=" << YN(r.red.surjective);
    std::cout << " escapes=" << r.rel_escapes << " " << r.seconds << " s peak_rss=" << r.peak_rss_mib << " MiB\n";
}

void ArmHPWWE(Checker& g, const HCell& hc, bool controls) {
    TameCellSpec spec = HpwweSpec(hc.l1);   // (slot 23, G-EVT23) the one spec, shared with the golden TAMEFAM points
    ENGINE_CHECK_MSG(spec.pres == hc.pres, "ArmHPWWE: the cell's presentation is not HpwwePresentation(l1)");
    spec.controls = controls;
    const TameCellReading r = RunTameCell(spec);
    const std::string cell = "Gamma_H(11*" + std::to_string(hc.l1) + ")";
    PrintReading(cell, r);
    for (const std::string& n : r.notes) std::cout << "  NOTE " << cell << ": " << n << "\n";
    // structure (I5, I4)
    g.check(r.genus_ok && r.rel_boundary_ok && r.boundary_rank + 1 == r.c,
            "(H) " + cell + ": h = 2g + c - 1 (Riemann--Hurwitz with nu_2, nu_3, c counted), delta(Rel) = 0, rank delta = c - 1 (I5)");
    g.check(r.rel_escapes == 0, "(H) " + cell + ": every operator maps Rel into Rel and every restriction is stable (I4)");
    Oracle(g, cell, "nsym", r.m, hc.m);
    Oracle(g, cell, "dim", r.h, hc.h);
    Oracle(g, cell, "cusps", r.c, hc.c);
    Oracle(g, cell, "W-P4 cusp_m", r.cusp_m, 20u);
    Oracle(g, cell, "W-P4 full_m", r.full_m, 30u);
    Oracle(g, cell, "W-P4b full_alg_dim", r.full_alg_dim, 20u);
    Oracle(g, cell, "W-P6a dimA", r.dimA, 10u);
    Oracle(g, cell, "W-P6a all_ops_inside", YN(r.members_inside), YN(true));
    // identities (I6, I8), ordinary (I7), N4, N5
    for (const auto& kv : r.identities) {
        const bool negative = kv.first.rfind("N4", 0) == 0;
        g.check(negative ? !kv.second : kv.second, "(H) " + cell + ": " + kv.first + (negative ? " -- fails, as predicted" : ""));
    }
    g.check(r.t5_unipotent, "(H) " + cell + ": T_5 - 1 nilpotent on the cusp m-part (ordinary; I7)");
    g.check(r.nonordinary_dim == 0, "(H) " + cell + ": N5 the generalised T_5 = 0 eigenspace in the cusp m-part is 0");
    if (r.v2_done()) {
        g.check(r.diamonds_power, "(H) " + cell + ": <q> = <3>^{e_q} on V for every Frobenius q < 200 (I8)");
        Oracle(g, cell, "W-P8 dim A/J^red", r.red.dim_quotient, 2u);
        Oracle(g, cell, "W-P8 case", r.red.case_name(), std::string("ii"));
        Oracle(g, cell, "W-P8 A->C surjective", YN(r.red.surjective), YN(true));
        Oracle(g, cell, "W-P8 relations", r.red.relations, 43u);
    }
    if (hc.pres) {
        const uint32_t mu = hc.pres->mu;
        Oracle(g, cell, "W-P6b z^5=0", YN(r.z5), YN(true));
        Oracle(g, cell, "W-P6b y^2=" + std::to_string(mu) + "z^2", YN(r.rel), YN(true));
        Oracle(g, cell, "W-P6b rank{z^i,yz^i}", r.pres_rank, 10u);
        Oracle(g, cell, "W-P7 dim A/(y)", r.dim_y, 2u);
        Oracle(g, cell, "W-P7 dim A/(y+z)", r.dim_ypz, hc.l1 == 23 ? 2u : 5u);
        Oracle(g, cell, "W-P7 dim A/(y-z)", r.dim_ymz, hc.l1 == 23 ? 2u : 5u);
        Oracle(g, cell, "N3 z'^5=0 (z'=<2>-1)", YN(r.shifted_z5), YN(true));
        Oracle(g, cell, "N3 y^2=mu z'^2", YN(r.shifted_rel), YN(false));
        Oracle(g, cell, "W-P8 J^red=(y)", YN(r.J_is_y), YN(true));
        // the decoys: J^red is not (y +- z) at 67 (there they have codimension 5); at 23 (y +- z) = (y) as ideals is possible
        if (hc.l1 == 67) g.check(!r.J_is_ypz && !r.J_is_ymz, "(H) " + cell + ": J^red != (y + z), (y - z) (W-P7: the case-(iii) decoys)");
        Oracle(g, cell, "W-P8w wild: same ideal", YN(r.wild_same), YN(true));
        Oracle(g, cell, "W-P8w wild: surjective", YN(r.red_wild.surjective), YN(true));
        g.check(r.radical_nilpotency == 6, "(H) " + cell + ": m_A^6 = 0, m_A^5 != 0 (the wild truncation u^6 = 0 suffices; " +
                                               std::to_string(r.radical_nilpotency) + ")");
    }
    if (controls) {
        Oracle(g, cell, "N6 swapped dim A/J", r.variants.at("swapped").dim_quotient, 2u);
        g.check(r.variant_same.at("swapped"), "(H) " + cell + ": N6 the swapped ansatz gives the same ideal (psi -> psi^{-1}<.>)");
        Oracle(g, cell, "N7 trivial dim A/J", r.variants.at("trivial").dim_quotient, 1u);
        Oracle(g, cell, "N7 nodiamond dim A/J", r.variants.at("nodiamond").dim_quotient, 1u);
    }
}

// ============================================================================
// (CELLS) WAKE-3: the survey arm, HECKE_TAME_FAMILY_CELLS="p:l0:l1,..." (read-only; WAKE3_spec.md)
// ============================================================================
uint64_t PowModU(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    b %= m;
    while (e) {
        if (e & 1) r = r * b % m;
        b = b * b % m;
        e >>= 1;
    }
    return r;
}

struct SurveyCell { uint32_t p; uint64_t l0, l1; };

// Parses the list; a malformed item is refused by position (one failing check each, never silently dropped).
std::vector<SurveyCell> ParseSurveyCells(Checker& g, const std::string& text) {
    std::vector<SurveyCell> out;
    std::vector<std::string> items;
    {
        std::string cur;
        for (char ch : text) {
            if (ch == ',') { items.push_back(cur); cur.clear(); }
            else cur += ch;
        }
        items.push_back(cur);
    }
    for (size_t k = 0; k < items.size(); ++k) {
        const std::string& it = items[k];
        std::vector<uint64_t> f;
        std::string why;
        {
            std::string cur;
            bool ok = !it.empty();
            for (char ch : it + ":") {
                if (ch == ':') {
                    if (cur.empty() || cur.size() > 9) { ok = false; break; }
                    f.push_back(std::stoull(cur));
                    cur.clear();
                } else if (ch >= '0' && ch <= '9') {
                    cur += ch;
                } else {
                    ok = false;
                    break;
                }
            }
            if (!ok || f.size() != 3) why = "not of the form p:l0:l1 (three decimal integers)";
        }
        if (why.empty()) {
            const uint64_t p = f[0], l0 = f[1], l1 = f[2];
            if (p < 5 || !hecke::IsPrimeU64(p)) why = "p must be a prime >= 5";
            else if (!hecke::IsPrimeU64(l0) || !hecke::IsPrimeU64(l1)) why = "l0 and l1 must be primes";
            else if (l0 == l1 || l0 == p || l1 == p) why = "p, l0, l1 must be pairwise distinct";
            else if (l0 % p != 1) why = "l0 = 1 mod p is required (Delta from l0)";
            else if (l0 * l1 >= (1ull << 20)) why = "l0 l1 < 2^20 is required";
        }
        if (!why.empty()) {
            std::cout << "  REFUSED HeckeTameFamilyValidation cell #" << (k + 1) << " '" << it << "': " << why << "\n";
            g.check(false, "(CELLS) cell #" + std::to_string(k + 1) + " '" + it + "' refused: " + why);
            continue;
        }
        out.push_back(SurveyCell{static_cast<uint32_t>(f[0]), f[1], f[2]});
    }
    return out;
}

void ArmSurveyCell(Checker& g, const SurveyCell& sc) {
    const uint32_t p = sc.p;
    const uint64_t l0 = sc.l0, l1 = sc.l1, N = l0 * l1;
    const std::string cell = "(" + std::to_string(p) + "," + std::to_string(l0) + "," + std::to_string(l1) + ")";
    // the localisation list: every prime q <= 23, q !| pN (U_l at 1 is added by the drivers)
    std::vector<uint64_t> loc;
    for (uint64_t q = 2; q <= 23; ++q)
        if (hecke::IsPrimeU64(q) && N % q && q != p) loc.push_back(q);
    // the preprint's congruence hypotheses (section 1.5)
    const bool h_l1 = l1 % p != 0 && l1 % p != 1 && l1 % p != p - 1;
    const bool h_pth = PowModU(l1 % l0, (l0 - 1) / p, l0) == 1;
    const MPartReading f0 = RunMParts(FullH(l0), p, loc);
    const MPartReading f1 = RunMParts(FullH(N), p, loc);
    const uint32_t rk_l0 = f0.cusp_m / 2, rk0 = f1.cusp_m / 2;
    const bool pass = h_l1 && h_pth && rk_l0 == 1 && rk0 == 2 && f0.cusp_m % 2 == 0 && f1.cusp_m % 2 == 0;
    std::cout << "  FILTER " << cell << " N=" << N << " l1!=0,+-1(p)=" << YN(h_l1) << " l1_pth_power_mod_l0=" << YN(h_pth)
              << " F0:rk_T0(l0)=" << rk_l0 << " (cusp_m=" << f0.cusp_m << ") F1:rk_T0(N)=" << rk0 << " (cusp_m=" << f1.cusp_m
              << " full_m=" << f1.full_m << ") pass=" << YN(pass) << "\n";
    g.check(f1.genus_ok && f1.rel_boundary_ok && f1.boundary_rank + 1 == f1.c && f1.rel_escapes == 0 && f0.rel_escapes == 0,
            "(CELLS) " + cell + ": the Gamma_0 filters' structure checks (h = 2g + c - 1, delta(Rel) = 0, rank delta = c - 1, no escape)");
    if (l0 == 11 && p == 5 && l1 == 263) Oracle(g, "Gamma_0(11*263)", "W-P9 rk T°", rk0, 2u);
    if (!pass) {
        std::cout << "  SURVEY " << cell << " skipped: outside the preprint's hypotheses / the minimal-rank filters (no claim)\n";
        return;
    }
    TameCellSpec spec;
    spec.p = p;
    spec.l0 = l0;
    spec.l1 = l1;
    spec.lean = true;
    spec.loc_primes = loc;
    spec.wild = false;
    spec.identities = false;
    spec.full_algebra_primes.clear();
    for (uint64_t q : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull})
        if (N % q && q != p) spec.full_algebra_primes.push_back(q);
    {
        // g = the smallest odd prime !| pN outside H (3 at l0 = 11: the preprint's)
        const GammaHLevelN G = GammaHLevelN::PrimeToPart(N, p);
        uint64_t q = 3;
        while (!(hecke::IsPrimeU64(q) && N % q && q != p && !G.in_H(q))) ++q;
        spec.g = q;
    }
    const TameCellReading r = RunTameCell(spec);
    std::cout << "  SURVEY p=" << p << " l0=" << l0 << " l1=" << l1 << " N=" << N << " rk0=" << rk0 << " g=" << spec.g << " m=" << r.m
              << " h=" << r.h << " c=" << r.c << " genus=" << r.twelve_g / 12 << " qc_nnz=" << r.qc_nnz << " cusp_m=" << r.cusp_m
              << " full_m=" << r.full_m << " dimA=" << r.dimA << " dimA_TgD=" << r.dimA_TgD << " full_alg=" << r.full_alg_dim
              << " m_A^e=0:e=" << r.radical_nilpotency << " relations=" << r.red.relations << " dimA/J=" << r.red.dim_quotient
              << " case=" << r.red.case_name() << " surj=" << YN(r.red.surjective) << " s=" << r.seconds << " peak_rss=" << r.peak_rss_mib
              << " MiB\n";
    for (const std::string& n : r.notes) std::cout << "  NOTE " << cell << ": " << n << "\n";
    const std::string pre = "(CELLS) " + cell + ": ";
    g.check(r.m == p * (l0 + 1) * (l1 + 1), pre + "V3-P1 m = p (l0+1)(l1+1)");
    g.check(r.genus_ok && r.rel_boundary_ok && r.boundary_rank + 1 == r.c && r.rel_escapes == 0,
            pre + "V3-P2 h = 2g + c - 1, delta(Rel) = 0, rank delta = c - 1, no Rel escape (n <= 13), stable restrictions");
    for (const auto& kv : r.identities) g.check(kv.second, pre + kv.first);
    g.check(r.cusp_m == 4 * p, pre + "V3-P3 cusp m-part = 4p (" + std::to_string(r.cusp_m) + ")");
    g.check(r.full_m == 6 * p, pre + "V3-P3 full m-part = 6p (" + std::to_string(r.full_m) + ")");
    g.check(r.dimA == 2 * p, pre + "V3-P4 dim A = 2p (" + std::to_string(r.dimA) + ")");
    g.check(r.full_alg_dim == 4 * p, pre + "V3-P6 full algebra = 4p (" + std::to_string(r.full_alg_dim) + ")");
    g.check(r.t5_unipotent && r.nonordinary_dim == 0, pre + "V3-P7 T_p - 1 nilpotent on V, no T_p = 0 part");
    g.check(r.diamonds_power, pre + "V3-P8 <q> = <g>^{e_q} on V for every Frobenius q < 200");
    g.check(r.v2_done() && r.red.dim_quotient == 2 && r.red.surjective,
            pre + "V3-P9 dim A/J^red = 2 (case (ii)), A -> C surjective (read " + std::to_string(r.red.dim_quotient) + ", case " +
                r.red.case_name() + ")");
    if (r.v2_done() && r.red.dim_quotient > 2)
        std::cout << "  FINDING " << cell << ": case (iii) (dim A/J^red = " << r.red.dim_quotient
                  << ") -- R-22a: report, do not fix; re-check with q < 400 (WAKE3_spec.md section 4)\n";
    if (l0 == 11 && p == 5 && l1 == 263) {
        const std::string oc = "Gamma_H(11*263)";
        Oracle(g, oc, "nsym", r.m, 15840u);
        Oracle(g, oc, "dim", r.h, 2641u);
        Oracle(g, oc, "cusps", r.c, 20u);
        Oracle(g, oc, "W-P9 cusp_m", r.cusp_m, 20u);
        Oracle(g, oc, "W-P9 full_m", r.full_m, 30u);
        Oracle(g, oc, "W-P9 dimA", r.dimA, 10u);
        Oracle(g, oc, "W-P9 all_ops_inside", YN(r.members_inside), YN(true));
        Oracle(g, oc, "W-P9 dim A/J^red", r.red.dim_quotient, 2u);
        Oracle(g, oc, "W-P9 A->C surjective", YN(r.red.surjective), YN(true));
    }
}

} // namespace

uint32_t HeckeTameFamilyValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::cout << "HeckeTameFamilyValidation (slot-22 WAKE-1; spec docs/implementation/slot22/WAKE1_spec.md v0.2;"
                 " HPWWE arXiv:2610.03401v1; oracle docs/implementation/slot22/wake/)\n";
    ArmTable(g);
    ArmShipped(g);
    ArmModel(g);
    ArmMParts(g);
    // the preprint's presentations (Examples 1.2, 1.3): HpwwePresentation (HeckeTameFamily.h; slot 23, G-EVT23)
    std::cout << "  EFFECTIVE HeckeTameFamilyValidation arm=HPWWE cells=2\n";
    ArmHPWWE(g, HCell{23, 1440, 241, 20, HpwwePresentation(23)}, true);
    ArmHPWWE(g, HCell{67, 4080, 681, 20, HpwwePresentation(67)}, true);
    {
        // l_1 = 43: rk T°_{Gamma_0(473)} = 3 (outside the minimal-rank hypothesis): INFO only, no claim, no gate
        TameCellSpec spec;
        spec.l1 = 43;
        spec.v2 = true;
        spec.wild = false;
        const TameCellReading r = RunTameCell(spec);
        std::cout << "  INFO (no claim, not gated) ";
        PrintReading("Gamma_H(11*43)", r);
    }
    const char* heavy = std::getenv("HECKE_TAMEFAMILY_HEAVY");
    if (heavy != nullptr) {
        const std::string hv(heavy);
        ENGINE_CHECK_MSG(hv == "1", "HECKE_TAMEFAMILY_HEAVY must be 1 when set");
        std::cout << "  EFFECTIVE HeckeTameFamilyValidation arm=HEAVY cells=1 (HECKE_TAMEFAMILY_HEAVY=1: l_1 = 263; INFO l_1 = 197)\n";
        const MPartReading r0 = RunMParts(FullH(11 * 263), 5, {2, 3, 7, 13, 17, 19});
        Oracle(g, "Gamma_0(11*263)", "W-P9 rk T°", r0.cusp_m / 2, 2u);
        TameCellSpec spec;
        spec.l1 = 263;
        spec.wild = false;
        const TameCellReading r = RunTameCell(spec);
        const std::string cell = "Gamma_H(11*263)";
        PrintReading(cell, r);
        g.check(r.genus_ok && r.rel_boundary_ok && r.boundary_rank + 1 == r.c && r.rel_escapes == 0,
                "(HEAVY) " + cell + ": h = 2g + c - 1, delta(Rel) = 0, rank delta = c - 1, no escape, stable restrictions");
        Oracle(g, cell, "nsym", r.m, 15840u);
        Oracle(g, cell, "dim", r.h, 2641u);
        Oracle(g, cell, "cusps", r.c, 20u);
        Oracle(g, cell, "W-P9 cusp_m", r.cusp_m, 20u);
        Oracle(g, cell, "W-P9 full_m", r.full_m, 30u);
        Oracle(g, cell, "W-P9 dimA", r.dimA, 10u);
        Oracle(g, cell, "W-P9 all_ops_inside", YN(r.members_inside), YN(true));
        Oracle(g, cell, "W-P9 dim A/J^red", r.red.dim_quotient, 2u);
        Oracle(g, cell, "W-P9 A->C surjective", YN(r.red.surjective), YN(true));
        TameCellSpec s197;
        s197.l1 = 197;
        s197.wild = false;
        const TameCellReading r197 = RunTameCell(s197);
        std::cout << "  INFO (no claim, not gated) ";
        PrintReading("Gamma_H(11*197)", r197);
    }
    const char* cells = std::getenv("HECKE_TAME_FAMILY_CELLS");
    if (cells != nullptr) {
        // (WAKE-3) read-only survey arm: WAKE3_spec.md; never writes, never changes a default-arm value
        const std::vector<SurveyCell> list = ParseSurveyCells(g, cells);
        std::cout << "  EFFECTIVE HeckeTameFamilyValidation arm=CELLS cells=" << list.size() << "\n";
        for (const SurveyCell& sc : list) ArmSurveyCell(g, sc);
    }
    std::cout << "  peak_rss=" << oc_util::PeakRssMiB() << " MiB wall=" << Seconds(t0) << " s\n";
    std::cout << "TAME_FAMILY checks=" << g.checks << " failing=" << g.fails << " verdict=" << (g.fails ? "FAIL" : "OK") << "\n";
    return g.fails;
}
