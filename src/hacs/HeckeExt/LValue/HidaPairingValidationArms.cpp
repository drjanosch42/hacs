// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-BIL) The `bilinear` arms of HeckeLValueValidation: Hida's GL2 x GL2 pairing
// (HidaPairing.{h,cpp}, HidaPairingRoute.cpp) against R19-9's committed values (docs/implementation/slot19/r199/out/*.out) and
// the predictions docs/implementation/slot19/wpbil/predictions.md (Q1-Q9), gates G1-G5 of R19-9 Sec. 6:
//   (G1) the C++ PoincareDualityPairing's cone-level Gram G^1 == -<Psi(c, c_d), y(v4v6)> on the stored bases, and the route's
//        evaluate_pairing reproduces it; the route's rho / rho^v == the pipeline's builders on S, T, U (R19-9's 9 cells;
//        (258,37), 13 s / 0.86 GB, behind HECKE_LVALUE_BILINEAR_HEAVY=1);
//   (G2) A_D(T_p) = A_c(T_p)^T for p = 2, 3, 5 and iota anti-isometric at the 27 small cells and at (258, 37) mod 37^10;
//   (G3) G_! = -B_{R19-8}/6 exactly, radical Z p_0, unimodular on W/p_0 (every small cell with n + 1 < ell);
//   (G4) the frozen valuation table of R19-9 Sec. 3 (27 cells) and Sec. 4 + P9 ((258, 37));
//   (G5) the cross-kind check: every pairing valuation == the congruence module of its lattice read from A_c(T_2) alone
//        (R19-9 Sec. 3 lemma), the Eisenstein cells' mixed value == v(B_k/k) of Bernoulli.h (the congruence kind's
//        Bernoulli input), and the route's IndexStatement;
//   (NB) negative controls: a perturbed cocycle refused, a representative moved by D^S + D^U changes nothing while u_0^v
//        does, the wrong lattice gives the other module, a planted non-adjoint caught, the route's refusals.
#include "Bernoulli.h"
#include "HidaPairing.h"
#include "LValueRoute.h"

#include "../../Hecke/HeckeContragredient.h"
#include "../../Hecke/HeckePairing.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hecke_ext::lvalue;
using namespace hecke_ext::lvalue::hidapair;

namespace {
using CheckFn = std::function<void(bool, const std::string&)>;

template <class F> bool HThrowsNotImplemented(F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}
template <class F> bool HThrowsLogic(F&& f) {
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
std::string KC(uint32_t k, uint64_t ell) { return "(k=" + std::to_string(k) + ",ell=" + std::to_string(ell) + ")"; }

// ---- the frozen table of R19-9 Sec. 3 (r199/out/small.out): (mixed_even, mixed_odd, hida (-1: n + 1 >= ell)) per system ----
struct Triple { int me, mo, h; };
struct SmallCell {
    uint32_t k;
    uint64_t ell;
    const char* kind;            // "Z" | "split" | "ramified"
    bool eisenstein;             // ell | B_k/k, dim S_k = 1
    std::vector<Triple> systems; // split: sorted by mixed_even
};
std::vector<SmallCell> SmallTable() {
    const Triple eis{1, 0, 0}, zero{0, 0, 0}, zero_noh{0, 0, -1}, ram{1, 1, 1};
    std::vector<SmallCell> T;
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{
             {12, 691}, {16, 3617}, {18, 43867}, {20, 283}, {20, 617}, {22, 131}, {22, 593}, {26, 657931}})
        T.push_back({c.first, c.second, "Z", true, {eis}});
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{{12, 101}, {16, 101}, {18, 101}, {20, 101}, {22, 103}, {26, 101}})
        T.push_back({c.first, c.second, "Z", false, {zero}});
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{{24, 144169}, {28, 131}, {28, 139}})
        T.push_back({c.first, c.second, "ramified", false, {ram, ram}});
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{{24, 103}, {24, 2294797}, {28, 9349}, {28, 362903}})
        T.push_back({c.first, c.second, "split", false, {zero, Triple{1, 0, 0}}});
    for (const auto& c : std::vector<std::pair<uint32_t, uint64_t>>{{12, 5}, {12, 7}, {12, 11}, {16, 7}, {16, 11}})
        T.push_back({c.first, c.second, "Z", false, {zero_noh}});
    T.push_back({12, 23, "Z", false, {zero}});
    return T;
}
// T_2 | S_k over Z (low -> high), R19-9 small.py CUSP (frozen): the odd block of W is S_k.
const std::vector<std::pair<uint32_t, std::vector<int64_t>>>& CuspCharpolys() {
    static const std::vector<std::pair<uint32_t, std::vector<int64_t>>> C{{24, {-20468736, -1080, 1}}, {28, {-195250176, 8280, 1}}};
    return C;
}

void ArmSmall(const CheckFn& Check) {
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t cells = 0;
    for (const SmallCell& c : SmallTable()) {
        const uint32_t n = c.k - 2;
        const uint32_t M = Zq::MaxPrecision(c.ell, 6);
        const std::string tc = KC(c.k, c.ell);
        SmallCellReading r;
        try {
            r = ReadSmallCell(n, c.ell, M);
        } catch (const std::exception& e) {
            Check(false, "(G4) " + tc + " threw: " + e.what());
            continue;
        }
        ++cells;
        const uint32_t dimS = (c.k == 24 || c.k == 28) ? 2u : 1u;
        Check(r.ne == dimS + 1 && r.no == dimS && r.w_torsion_free,
              "(G2) " + tc + ": W saturated, free, ranks (geo-even, geo-odd) = (" + std::to_string(dimS + 1) + "," + std::to_string(dimS) + ")");
        Check(r.adj2 && r.adj3, "(G2) " + tc + ": A_D(T_p) == A_c(T_p)^T for p = 2, 3 (R19-9 P3)");
        Check(r.adj5, "(G2) " + tc + ": A_D(T_5) == A_c(T_5)^T (WP-BIL Q6, predicted, not run in R19-9)");
        Check(r.t2t5_commute, "(G2) " + tc + ": T_2 T_5 = T_5 T_2 on W and on H^1(D)");
        Check(r.iota, "(G2) " + tc + ": <iota P, iota y> = -<P, y>");
        Check(r.kind == c.kind, "(G4) " + tc + ": eigen-system kind " + r.kind + " == " + c.kind);
        for (const auto& cp : CuspCharpolys()) {
            if (cp.first != c.k) continue;
            const Zq R(c.ell, M);
            bool same = r.charpoly_odd.size() == cp.second.size();
            for (size_t i = 0; same && i < cp.second.size(); ++i) same = r.charpoly_odd[i] == R.from_int(cp.second[i]);
            Check(same, "(G4) " + tc + ": charpoly of T_2 on W^{geo-odd} == the integral charpoly of T_2 | S_k");
        }
        const bool hida = n + 1 < c.ell;
        Check(r.hida_applicable == hida, "(G3) " + tc + ": Hida's form applies iff n + 1 < ell");
        if (hida) {
            Check(r.G_is_minus_sixth_B, "(G3) " + tc + ": G_!(P, Q) = <P, theta iota_!(Q)> == -1/6 B_{R19-8}(P, Q) (R19-9 P4)");
            Check(r.G_radical == 1 && std::all_of(r.G_pivots.begin(), r.G_pivots.end(), [](uint32_t v) { return v == 0; }),
                  "(G3) " + tc + ": G_! has radical of rank 1 (Z p_0) and is unimodular on W/p_0 (Hida Thm 2.5)");
        }
        // values
        std::vector<Triple> got;
        for (const EigenReading& E : r.systems) got.push_back({int(E.mixed_even), int(E.mixed_odd), E.hida_defined ? int(E.hida) : -1});
        if (std::string(c.kind) == "split")
            std::sort(got.begin(), got.end(), [](const Triple& a, const Triple& b) { return a.me < b.me; });
        bool same = got.size() == c.systems.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].me == c.systems[i].me && got[i].mo == c.systems[i].mo && got[i].h == c.systems[i].h;
        std::string line = "  HIDA_PAIR_CELL k=" + std::to_string(c.k) + " ell=" + std::to_string(c.ell) + " M=" + std::to_string(M) +
                           " kind=" + r.kind + " W=(" + std::to_string(r.ne) + "," + std::to_string(r.no) + ")";
        for (size_t i = 0; i < r.systems.size(); ++i) {
            const EigenReading& E = r.systems[i];
            line += " " + E.name + ":mixed=" + std::to_string(E.mixed_even) + "/" + std::to_string(E.mixed_odd) +
                    " hida=" + (E.hida_defined ? std::to_string(E.hida) : std::string("-")) + " cong=" + std::to_string(E.cong_even) + "/" +
                    std::to_string(E.cong_odd) + " e=" + std::to_string(E.e);
        }
        line += " adj235=" + std::string(r.adj2 && r.adj3 && r.adj5 ? "yes" : "NO") + " iota=" + (r.iota ? "anti" : "NO");
        std::cout << line << std::endl;
        Check(same, "(G4) " + tc + ": the frozen valuations of R19-9 Sec. 3 (mixed even / mixed odd / Hida H^1_!)");
        for (const EigenReading& E : r.systems) {
            Check(E.eigen_check >= E.e, "(G4) " + tc + " " + E.name + ": eigenvectors exact to v_pi >= e (A g = lambda g)");
            Check(E.t5_eigen, "(G2) " + tc + " " + E.name + ": the eigen-lines are T_5-eigen (Q6)");
            // (G5) the lemma: pairing value == congruence module of the lattice, from A_c alone
            Check(E.mixed_even == E.cong_even && E.mixed_odd == E.cong_odd,
                  "(G5) " + tc + " " + E.name + ": mixed values == the congruence modules of W^{geo-even}, W^{geo-odd} (R19-9 Sec. 3 lemma)");
            if (E.hida_defined)
                Check(E.hida == E.cong_odd, "(G5) " + tc + " " + E.name + ": Hida's H^1_! value == the congruence module of H^1_!");
        }
        if (c.eisenstein) {
            const BernoulliReading B = BernoulliModEllPowerExact(c.k, c.ell, 2);
            Check(B.v_exact && r.systems.size() == 1 && r.systems[0].mixed_even == B.v,
                  "(G5) " + tc + ": mixed geo-even value == v_ell(B_k/k) = " + std::to_string(B.v) + " (Bernoulli.h, the congruence kind's input)");
            // (NB-c) the wrong lattice gives the other module: Hida's H^1_! (0) != the mixed lattice's module (1)
            Check(r.systems.size() == 1 && r.systems[0].hida_defined && r.systems[0].hida != r.systems[0].cong_even,
                  "(NB) " + tc + ": the wrong lattice gives the other module (H^1_! value 0 != the H^1_c module v(B_k/k))");
        }
        // the route's index statement (interface level)
        {
            Domain d;
            d.n_sym = n; d.ell = c.ell; d.m = M; d.level = 1;
            d.cohomology = "H^1_c(M_n) x H^1(D_n)";
            d.subquotient = "geo-even";
            const HidaPairingRoute mixed(d);
            const IndexStatement I = mixed.index_statement();
            uint32_t want = 0;
            for (const Triple& t : c.systems) want += static_cast<uint32_t>(t.me);
            want /= (std::string(c.kind) == "ramified" ? 2u : 1u);
            Check(I.holds && I.found == want, "(G5) " + tc + ": route (H2) mixed geo-even: " + I.to_string());
            if (hida) {
                d.cohomology = "H^1_! x H^1_!";
                const HidaPairingRoute h(d);
                const IndexStatement J = h.index_statement();
                uint32_t wh = 0;
                for (const Triple& t : c.systems) wh += static_cast<uint32_t>(t.h);
                wh /= (std::string(c.kind) == "ramified" ? 2u : 1u);
                Check(J.holds && J.found == wh, "(G5) " + tc + ": route (H1) Hida H^1_!: " + J.to_string());
            }
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  HIDA_PAIR_SMALL cells=" << cells << " seconds=" << secs << std::endl;
    Check(cells == 27, "(G4) all 27 cells of R19-9 Sec. 3 ran");
}

void ArmCell258(const CheckFn& Check) {
    Cell258Reading r;
    try {
        r = ReadCell258(10);
    } catch (const std::exception& e) {
        Check(false, std::string("(G4) (258,37) threw: ") + e.what());
        return;
    }
    std::cout << "  HIDA_PAIR_258 M=10 W=(" << r.ne << "," << r.no << ") adj2=" << r.adj2 << " adj3=" << r.adj3 << " adj5=" << r.adj5
              << " iota=" << r.iota << " H1D_free_rank=" << r.HD_free_rank << " H1D_torsion=" << r.HD_torsion.size()
              << " seconds=" << r.seconds << std::endl;
    Check(r.ne == 22 && r.no == 21, "(G2) (258,37): W saturated of ranks (22, 21)");
    Check(r.adj2 && r.adj3, "(G2) (258,37): A_D(T_p) == A_c(T_p)^T mod 37^10 for p = 2, 3");
    Check(r.adj5, "(G2) (258,37): A_D(T_5) == A_c(T_5)^T mod 37^10 (Q6)");
    Check(r.iota, "(G2) (258,37): iota anti-isometric");
    Check(r.HD_free_rank == 43 && r.HD_torsion.empty(), "(G4) (258,37): H^1(D) = D/(D^S + D^U) torsion-free of rank 43 (Z/37^4)");
    uint32_t sum_free = 0;
    for (int s = 0; s < 2; ++s) {
        const BlockSign& S = r.sign[s];
        const bool free_sign = s == 0;
        std::cout << "  HIDA_PAIR_258_BLOCK " << S.name << " rank=" << S.rank_c << "/" << S.rank_D << " charpoly==P:" << S.charpoly_c_is_P
                  << "/" << S.charpoly_D_is_P << " t-ranks=(" << S.tr1 << "," << S.tr2 << "," << S.tr3 << ")/(" << S.dr1 << "," << S.dr2 << ","
                  << S.dr3 << ") gram_unimodular=" << S.gram_unimodular << " P9: rank W_D=" << S.p9_rank_WD << " intertwines="
                  << S.p9_intertwines << " index_pivots=";
        for (uint32_t v : S.p9_index_pivots) std::cout << v;
        for (const BlockEigen& f : S.forms)
            std::cout << " " << f.name << ":mixed=" << f.mixed << ",P9=" << f.p9 << ",vpiP'=" << f.vpi_dP << ",cont=" << f.content_c;
        std::cout << std::endl;
        Check(S.rank_c == 4 && S.rank_D == 4 && S.charpoly_c_is_P && S.charpoly_D_is_P,
              "(G4) (258,37) " + S.name + ": m-blocks of rank 4 on W and H^1(D), charpoly == R19-1's P mod 37^8");
        const bool tfree = S.tr1 == 3 && S.tr2 == 2 && S.tr3 == 1, tm = S.tr1 == 2 && S.tr2 == 1 && S.tr3 == 0;
        const bool dfree = S.dr1 == 3 && S.dr2 == 2 && S.dr3 == 1, dm = S.dr1 == 2 && S.dr2 == 1 && S.dr3 == 0;
        Check(free_sign ? (tfree && dfree) : (tm && dm),
              "(G4) (258,37) " + S.name + ": block type " + (free_sign ? "free (T_m), t-ranks (3,2,1)" : "m, t-ranks (2,1,0)") + " on both sides");
        Check(S.gram_unimodular, "(G4) (258,37) " + S.name + ": the Gram W_m x H^1(D)_m is unimodular");
        Check(S.p9_intertwines && S.p9_charpoly_is_P && S.p9_index_pivots == std::vector<uint32_t>({0, 0, 0, 1}),
              "(G4) (258,37) P9 into the dual of W^{" + S.name + "}: iota_! T_2 = T_2 iota_!, [H^1(D)_m : H^1_!(D)_m] = 37 (Smith (0,0,0,1))");
        Check(S.forms.size() == 4, "(G4) (258,37) " + S.name + ": four eigenforms");
        for (const BlockEigen& f : S.forms) {
            const std::string tf = "(258,37) " + S.name + " " + f.name;
            Check(f.eigen_check >= 14, "(G4) " + tf + ": eigenvectors exact to v_pi >= 14");
            Check(f.vpi_dP == 3, "(G4) " + tf + ": v_pi P'(rho) = 3 (eta_f, [E] rmk:lvshapes O_{K_i}/pi^3)");
            Check(f.mixed == (free_sign ? 3u : 2u), "(G4) " + tf + ": mixed v_pi<g_f, h_f> = " + (free_sign ? "3 (free sign)" : "2 (m-sign)"));
            Check(f.p9 == 3, "(G4) " + tf + ": P9 v_pi<g_f^c, h_f^!> = 3 on H^1_!(D)_m (both signs)");
            Check(f.mixed == f.vpi_dP - std::min(f.vpi_dP, f.content_c),
                  "(G5) " + tf + ": mixed value == congruence module of W_m (v_pi P'(rho) - content, from A_c alone)");
            if (free_sign) sum_free += f.mixed;
        }
        if (!free_sign) {
            // (NB-c) the wrong lattice gives the other module: H^1(D)_m (2) != H^1_!(D)_m (3) on the m-sign
            bool differ = true;
            for (const BlockEigen& f : S.forms) differ = differ && f.mixed != f.p9;
            Check(differ, "(NB) (258,37) m-sign: the lattice H^1(D)_m gives 2, H^1_!(D)_m gives 3 (the other module)");
        }
    }
    Check(sum_free == 12, "(G4) (258,37): sum over the 4 forms of v_37 = 12/2 = 6 = v_37(disc P) on the free sign (per form 3/2)");
    std::cout << "  HIDA_PAIR_258 per-eigenform v_37 = 3/2 (free sign), sum = " << sum_free / 2 << " = v(disc P)" << std::endl;
    // the route at (258, 37): the m-sign on the two lattices
    Domain d;
    d.n_sym = 258; d.ell = 37; d.m = 10; d.level = 1;
    d.cohomology = "H^1_c(M_n) x H^1(D_n)";
    d.subquotient = "m=(37,T_2-29) geo-odd";
    const IndexStatement I = HidaPairingRoute(d).index_statement();
    d.cohomology = "H^1_c(M_n) x H^1_!(D_n)";
    const IndexStatement J = HidaPairingRoute(d).index_statement();
    std::cout << "  HIDA_PAIR_258_ROUTE mixed " << I.to_string() << std::endl << "  HIDA_PAIR_258_ROUTE P9 " << J.to_string() << std::endl;
    Check(I.holds && I.found == 4 && J.holds && J.found == 6,
          "(G5) (258,37) route (H3) m-sign: mixed norm v_37 = 4 (2 per form), through H^1_!(D) = 6 (eta_f, 3 per form)");
}

// ---- (G1) the model gate on the C++ PoincareDualityPairing --------------------------------------------------------------
void ArmModel(const CheckFn& Check, const std::vector<std::pair<uint32_t, uint32_t>>& cells, const char* arm) {
    for (const auto& c : cells) {
        const uint32_t n = c.first, p = c.second;
        const std::string tc = "(n=" + std::to_string(n) + ",p=" + std::to_string(p) + ")";
        const auto t0 = std::chrono::steady_clock::now();
        try {
            const hecke::ContragredientCell cell(n, p);
            const hecke::PoincareDualityPairing pd(cell);
            const HidaPairingCell C(n, p, 1, {2});
            const FibreOps& F = C.F();
            const Zq& R = C.R();
            const uint32_t d = n + 1;
            // the convention gate: the route's rho / rho^v == the pipeline's builders
            bool conv = true;
            for (const M22& g : {M22{0, -1, 1, 0}, M22{1, 1, 0, 1}, M22{0, -1, 1, 1}}) {
                const FlatMatrix A(2, 2, p, std::vector<uint32_t>{static_cast<uint32_t>(R.from_int(g.a)), static_cast<uint32_t>(R.from_int(g.b)),
                                                                  static_cast<uint32_t>(R.from_int(g.c)), static_cast<uint32_t>(R.from_int(g.d))});
                const FlatMatrix rho = ComputeSymmetricPowerMatrixFM(A, n), drho = ComputeContragredientPowerMatrixFM(A, n);
                const Mat mine = F.matrix(Fibre::V, g), dmine = F.matrix(Fibre::D, g);
                for (uint32_t i = 0; i < d; ++i)
                    for (uint32_t j = 0; j < d; ++j)
                        conv = conv && rho.get_entry(i, j) == mine[i][j] && drho.get_entry(i, j) == dmine[i][j];
            }
            Check(conv, "(G1) " + tc + ": rho_n, rho_n^v of the route == ComputeSymmetricPowerMatrixFM / ComputeContragredientPowerMatrixFM on S, T, U");
            const SimplicialComplex& K = cell.K();
            const int64_t e04 = K.find_simplex_index(1, {0, 4}), e46 = K.find_simplex_index(1, {4, 6}), v0 = K.find_simplex_index(0, {0});
            ENGINE_CHECK_MSG(e04 >= 0 && e46 >= 0 && v0 >= 0, "ArmModel: the 7-vertex model lacks v0, {0,4} or {4,6}");
            const auto& bdK = pd.chain().bd_simplex_in_K[0];
            int64_t b0 = -1;
            for (size_t i = 0; i < bdK.size(); ++i)
                if (static_cast<int64_t>(bdK[i]) == v0) b0 = static_cast<int64_t>(i);
            ENGINE_CHECK_MSG(b0 >= 0, "ArmModel: v0 is not a vertex of K_bd");
            const FlatMatrix& Hc = pd.certificate().Hc[1];
            const FlatMatrix& HD = pd.certificate().HD[1];
            const FlatMatrix& G = pd.gram(1);
            const uint32_t bulk = pd.cone().bulk_part_dim(1);
            const uint32_t hc = Hc.get_cols(), hD = HD.get_cols();
            std::vector<Vec> Ps, ys;
            for (uint32_t i = 0; i < hc; ++i) {
                Vec x(d);
                for (uint32_t t = 0; t < d; ++t)
                    x[t] = R.sub(R.add(Hc.get_entry(static_cast<uint32_t>(e04) * d + t, i), Hc.get_entry(static_cast<uint32_t>(e46) * d + t, i)),
                                 Hc.get_entry(bulk + static_cast<uint32_t>(b0) * d + t, i));
                Ps.push_back(F.sub(F.act(M22{0, -1, 1, 0}, x), x));   // Psi = (rho(S) - 1)(c(v0v4) + c(v4v6) - c_d(v0))
            }
            for (uint32_t j = 0; j < hD; ++j) {
                Vec y(d);
                for (uint32_t t = 0; t < d; ++t) y[t] = HD.get_entry(static_cast<uint32_t>(e46) * d + t, j);
                ys.push_back(y);
            }
            // Psi lands in W(F_p); the route's evaluation on (W-coordinates of Psi, dual coordinates of y)
            const auto co = CoordsInBasis(R, C.W(), Ps);
            Domain dm;
            dm.n_sym = n; dm.ell = p; dm.m = 1; dm.level = 1;
            dm.cohomology = "H^1_c(M_n) x H^1(D_n)";
            const HidaPairingRoute route(dm);
            bool inW = true, formula = true, routeok = true;
            Mat Gf(hc, Vec(hD));
            for (uint32_t i = 0; i < hc; ++i) {
                inW = inW && co[i].second >= 1;
                for (uint32_t j = 0; j < hD; ++j) {
                    Gf[i][j] = F.eval(Ps[i], ys[j]);
                    formula = formula && G.get_entry(i, j) == R.neg(Gf[i][j]);
                    CohomologyClass x, y;
                    x.coordinates = co[i].first;
                    y.coordinates = C.dual_coords(ys[j]);
                    routeok = routeok && route.evaluate_pairing(x, y) == Gf[i][j];
                }
            }
            const uint32_t rk = RankModEll(R, Gf);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "  HIDA_PAIR_MODEL arm=" << arm << " n=" << n << " p=" << p << " dim_H1c=" << hc << " dim_H1D=" << hD
                      << " rank_W=" << C.W().size() << " cup==-<Psi,y(v4v6)>:" << (formula ? "yes" : "NO") << " route:" << (routeok ? "yes" : "NO")
                      << " perfect=" << (pd.perfect(1) ? "yes" : "no") << " seconds=" << secs << std::endl;
            Check(inW && hc == C.W().size(), "(G1) " + tc + ": Psi(H^1_c) lands in W(F_p), dim H^1_c == rank W");
            Check(formula, "(G1) " + tc + ": PoincareDualityPairing gram(1) == -<Psi(c, c_d), y(v4v6)> (R19-9 P2, now on the C++ pairing)");
            Check(routeok, "(G1) " + tc + ": HidaPairingRoute::evaluate_pairing reproduces <Psi, y> = -gram(1)");
            Check(rk == hc && hc == hD && pd.perfect(1), "(G1) " + tc + ": the Gram is perfect over F_p (rank = dim H^1_c = dim H^1(D))");
        } catch (const std::exception& e) {
            Check(false, "(G1) " + tc + " threw: " + e.what());
        }
    }
}

void ArmNegative(const CheckFn& Check) {
    const uint32_t n = 10;
    const uint64_t ell = 691;
    const uint32_t M = 3;
    const HidaPairingCell C(n, ell, M, {2});
    const FibreOps& F = C.F();
    const Zq& R = C.R();
    const M22 S{0, -1, 1, 0}, U{0, -1, 1, 1};
    const Vec& y = C.Y()[0];
    const Vec xU = F.sub(F.dact(U, y), y), zero(n + 1, 0);
    Vec e0(n + 1, 0);
    e0[0] = 1;
    // (a) a perturbed cocycle is refused
    Check(!HThrowsLogic([&] { (void)C.normalise(Fibre::D, zero, xU); }), "(NB) the cocycle (0, (U-1)y) is accepted");
    Check(HThrowsLogic([&] { (void)C.normalise(Fibre::D, zero, F.add(xU, e0)); }),
          "(NB) a perturbed cocycle (c(U) + u_0^v, N_U c(U) != 0) is refused by the normalisation");
    // (b) well defined on D / (D^S + D^U), and not on D
    Vec z(n + 1);
    for (uint32_t t = 0; t <= n; ++t) z[t] = (7 * t + 3) % R.q;
    const Vec moved = F.add(F.add(y, F.add(z, F.dact(S, z))), F.add(F.add(z, F.dact(U, z)), F.dact(U, F.dact(U, z))));
    Check(C.dual_coords(moved) == C.dual_coords(y), "(NB) y + (1+S)z + N_U z has the same values on W (well defined on H^1(D))");
    Check(C.dual_coords(F.add(y, e0)) != C.dual_coords(y), "(NB) y + u_0^v changes a value (u_0^v not in D^S + D^U)");
    // (d) a planted non-adjoint
    Mat AD = C.AD(2);
    AD[0][1] = R.add(AD[0][1], R.q / R.ell);
    bool adj = true;
    const Mat& A = C.Ac(2);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A.size(); ++j) adj = adj && AD[i][j] == A[j][i];
    Check(C.adjoint_ok(2) && !adj, "(NB) a planted non-adjoint A_D + ell^{M-1} E_01 fails the adjunction (the real one passes)");
    // (e) the route's refusals
    Domain d;
    d.n_sym = n; d.ell = ell; d.m = M; d.level = 1;
    d.cohomology = "H^1_c(M_n) x H^1(D_n)";
    const HidaPairingRoute route(d);
    CohomologyClass bad;
    bad.coordinates = {1};
    Check(HThrowsLogic([&] { (void)route.evaluate_pairing(bad, bad); }), "(NB) a class of the wrong rank is refused (logic_error)");
    Check(HThrowsLogic([&] { (void)route.evaluate_functional({}); }) && HThrowsLogic([&] { (void)route.congruence_module(); }),
          "(NB) the bilinear route refuses the linear and congruence evaluations (kind)");
    Domain g0 = d;
    g0.model = ModelId::Gamma0NShapiro;
    Domain hb = d;
    hb.n_sym = 40; hb.ell = 37; hb.cohomology = "H^1_! x H^1_!";
    Domain lv = d;
    lv.cohomology = "H^1(Gamma_0(N);V(chi))";
    Check(HThrowsNotImplemented([&] { (void)HidaPairingRoute(g0).index_statement(); }) &&
              HThrowsNotImplemented([&] { (void)HidaPairingRoute(hb).evaluate_pairing(bad, bad); }) &&
              HThrowsNotImplemented([&] { (void)HidaPairingRoute(lv).index_statement(); }),
          "(NB) the route refuses the Gamma_0(N) model, Hida's H^1_! at n + 1 >= ell, and an unknown lattice (NotImplementedInThisBuild)");
    // the route's values agree with the core on a basis pair, and the normalisation record names no removed Euler factor
    CohomologyClass x1, y1;
    x1.coordinates.assign(C.W().size(), 0);
    y1.coordinates.assign(C.W().size(), 0);
    x1.coordinates[0] = 1;
    y1.coordinates[0] = 1;
    Check(route.evaluate_pairing(x1, y1) == 1, "(G1) <w_0, y_0> = 1 through the route (dual basis)");
    const NormalisationRecord N = route.normalisation();
    Check(N.euler_factors_removed.empty() && N.constants.size() >= 2 && N.constants[1].valuation_known && N.constants[1].ell_valuation == 0,
          "(G5) NormalisationRecord: no Euler factor removed (Hida Thm 5.1, N = 1); v_ell((k-1)!) = 0 at ell >= k");
    std::cout << "  NORMALISATION (H) " << N.to_string() << std::endl << "  ADJUNCTION (H) " << route.adjunction() << std::endl;
}
} // namespace

// Called by HeckeLValueValidation (one Check per assertion).
void RunHidaPairingArms(const std::function<void(bool, const std::string&)>& Check) {
    const auto t0 = std::chrono::steady_clock::now();
    ArmSmall(Check);
    ArmCell258(Check);
    ArmModel(Check, {{10, 691}, {10, 5}, {10, 7}, {12, 5}, {22, 144169}, {24, 7}, {36, 37}, {40, 37}, {72, 37}}, "default");
    ArmNegative(Check);
    // read-only env arm (documented in docs/validation-catalogue.md's env table): the heavier model cells of G1
    const char* env = std::getenv("HECKE_LVALUE_BILINEAR_HEAVY");
    if (env && std::string(env) == "1") {
        std::cout << "  EFFECTIVE HeckeLValueValidation arm=BILINEAR_HEAVY cells=1" << std::endl;
        ArmModel(Check, {{258, 37}}, "heavy");
    }
    std::cout << "  HIDA_PAIR seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << std::endl;
}
