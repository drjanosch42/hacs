// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 20, WP-LV3) HeckeLValueValidation's slot-20 arms -- the L-value code at depth 2:
//   (LV3-R)   the generic m-block reader (MBlockReader.h) on planted polynomials: split, Eisenstein e = 2 and e = 3 clusters, the
//             depth-2 split pair, and its refusals (unramified residue extension, e not dividing ell - 1, a multiple root);
//   (LV3-258) the reader against WP-BIL's quartic-specific ReadCell258 at (258, 37) mod 37^10: every field equal, the four roots
//             identified with R19-1's factors P_1, P_2 (recovered, not read), the per-root values (mixed, P9, v_pi P', content)
//             equal per factor; the route on both lattices; R20-3's change of lattice computed and printed;
//   (LV3-D2)  (H2) and (L1) at depth 2 at (282, 37), (226, 103) and the CROSS-KIND GATE there and at the depth-1 controls
//             (30, 37), (10, 691): congruence (L / sum_p (T_p - e_p) L on the cusp line of W_m) = bilinear (the mixed pairing on
//             the p_0 sign, R19-9's (H2)) = linear (R19-8's (L1) on <p_0, r^+>, read by BirchModularSymbolRoute on the group-
//             cocycle W) = v(B_k/k) (Bernoulli.h); the other sign 0; the lemma value == the pairing value;
//   (LV3-I)   iota^* over Z/ell^k (Hecke/HeckeInvolution.h BuildRingInvolution): iota^2 = id and H = H^+ (+) H^- (ENGINE_CHECKed in
//             the build), the free part's sign ranks d_k / d_k - 1 at a regular and an Eisenstein cell, the uint32_t and uint64_t
//             instantiations agree, F_ell (k = 1) == the field HeckeInvolution's dims, a torsion cell (38, 37) splits its torsion,
//             ell = 3 refused by name.
// Heavy (HECKE_LVALUE_HEAVY=1, in HeckeLValueValidation.cpp): the depth-2 cross-kind against [C]'s own H^1 congruence route
// (ArmDeepRungs) and the C++ cone's (L1) at (282, 37) over Z/37^3.  (HECKE_LVALUE_HEAVY=twist62: the (37; omega^6, 62) cell.)
#include "Bernoulli.h"
#include "BirchModularSymbol.h"
#include "HidaPairing.h"
#include "LValueRoute.h"
#include "MBlockReader.h"

#include "../HeckeExtBindingSupport.h"   // BuildProductionInvolution (the field object, for the k = 1 comparison)

#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/HeckeInvolution.h"
#include "../../Hecke/HeckeOperatorFamily.h"   // tau_construction_mutex
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/SL2ZTessellationTables.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace hecke_ext::lvalue;
using CheckFn = std::function<void(bool, const std::string&)>;

namespace {
std::string Cell(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + "," + std::to_string(ell) + ")"; }
template <class F> bool ThrowsNIB(F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// a monic product of the given factors (low -> high), over R
hidapair::Vec PolyMul(const hidapair::Zq& R, const hidapair::Vec& a, const hidapair::Vec& b) {
    hidapair::Vec c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = R.add(c[i + j], R.mul(a[i], b[j]));
    return c;
}

void ArmReaderPlanted(const CheckFn& Check) {
    const hidapair::Zq R(37, 6);
    const uint64_t q = R.q, l = 37;
    auto lin = [&](uint64_t r) { return hidapair::Vec{R.neg(r % q), 1}; };
    // split, depth 2: (x - 37)(x - 37 - 5*37^2)(x - 2*37)
    {
        const hidapair::Vec P = PolyMul(R, PolyMul(R, lin(l), lin(l + 5 * l * l)), lin(2 * l));
        const mblock::RootReading r = mblock::RootsOfNilpotentCharpoly(R, P);
        bool ok = r.complete && r.roots.size() == 3;
        std::vector<uint32_t> dps;
        for (const auto& x : r.roots) {
            ok = ok && r.orders[x.order].e() == 1;
            dps.push_back(x.vpi_dP);
        }
        std::sort(dps.begin(), dps.end());
        Check(ok && dps == std::vector<uint32_t>({2, 3, 3}),
              "(LV3-R) split cluster (x-37)(x-37-5*37^2)(x-74): three roots in Z_37, v P'(rho) = (2, 3, 3) (the depth-2 pair: 1 + 2)");
    }
    // Eisenstein e = 2: (x^2 - 37*5)(x^2 - 37*2 + 37 x): two quadratic orders
    {
        const hidapair::Vec P1{R.neg(37 * 5), 0, 1}, P2{R.neg(37 * 2), 37, 1};
        const mblock::RootReading r = mblock::RootsOfNilpotentCharpoly(R, PolyMul(R, P1, P2));
        bool ok = r.complete && r.roots.size() == 4;
        uint32_t on1 = 0, on2 = 0;
        for (const auto& x : r.roots) {
            const mblock::LocalOrder& O = r.orders[x.order];
            ok = ok && O.e() == 2;
            if (O.vpi(mblock::EvalPoly(O, P1, x.rho)) >= O.cap() - 4) ++on1;
            if (O.vpi(mblock::EvalPoly(O, P2, x.rho)) >= O.cap() - 4) ++on2;
        }
        Check(ok && on1 == 2 && on2 == 2, "(LV3-R) two Eisenstein quadratics: four roots in O = Z_37[pi]/(pi^2 - 37u), two on each factor (recovered)");
    }
    // e = 3 (3 | 36): x^3 - 37*2
    {
        const mblock::RootReading r = mblock::RootsOfNilpotentCharpoly(R, hidapair::Vec{R.neg(74), 0, 0, 1});
        Check(r.complete && r.roots.size() == 3 && r.orders.back().e() == 3, "(LV3-R) x^3 - 74: three roots, e = 3, the cube roots of unity in F_37");
    }
    // refusals
    {
        // x^2 - 37^2 * 2: slope 1, residual y^2 - 2 irreducible over F_37 (2 is a non-residue mod 37)
        const mblock::RootReading r1 = mblock::RootsOfNilpotentCharpoly(R, hidapair::Vec{R.neg(2 * l * l), 0, 1});
        Check(!r1.complete && r1.refusal.find("unramified") != std::string::npos,
              "(LV3-R) negative: x^2 - 2*37^2 needs an unramified extension (refused by name): " + r1.refusal);
        const mblock::RootReading r2 = mblock::RootsOfNilpotentCharpoly(R, hidapair::Vec{R.neg(37), 0, 0, 0, 0, 1});
        Check(!r2.complete && r2.refusal.find("not dividing") != std::string::npos,
              "(LV3-R) negative: x^5 - 37, 5 does not divide 36 (refused by name): " + r2.refusal);
        const mblock::RootReading r3 = mblock::RootsOfNilpotentCharpoly(R, PolyMul(R, lin(l), lin(l)));
        Check(!r3.complete && r3.refusal.find("multiple root") != std::string::npos,
              "(LV3-R) negative: (x - 37)^2 is a multiple root (refused by name): " + r3.refusal);
        const mblock::RootReading r4 = mblock::RootsOfNilpotentCharpoly(R, lin(1));
        Check(!r4.complete && r4.refusal.find("not == x^r") != std::string::npos, "(LV3-R) negative: x - 1 is not an m-block charpoly");
    }
}

// ---- (258, 37): the generic reader == WP-BIL's quartic reader ------------------------------------------------------------
void ArmCell258(const CheckFn& Check) {
    const auto t0 = std::chrono::steady_clock::now();
    hidapair::Cell258Reading old;
    mblock::MBlockReading gen;
    try {
        old = hidapair::ReadCell258(10);
        const hidapair::HidaPairingCell C(258, 37, 10, {2, 3, 5});
        mblock::MBlockOptions opt;
        opt.p9 = true;
        opt.lattice_change = true;
        gen = mblock::ReadMBlockCell(C, 29, opt);
    } catch (const std::exception& e) {
        Check(false, std::string("(LV3-258) threw: ") + e.what());
        return;
    }
    const hidapair::Zq R8(37, 8);
    {
        uint32_t idx = 0;
        for (uint32_t v : gen.theta_index_pivots) idx += v;
        std::cout << "  LV3_258 theta_index: " << gen.theta_index_pivots.size() << " pivots, [H^1(V)/tors : Theta(H^1(D)/tors)] = 37^" << idx
                  << "; v_37(258!) = " << gen.v_nfact << ", [H^1(D)/tors : Phi_n,*(H^1(V)/tors)] = 37^" << gen.phi_index << std::endl;
        Check(gen.theta_index_pivots.size() == gen.ne + gen.no, "(LV3-258) Theta = theta^{-1} : H^1(D)/tors -> H^1(V)/tors is injective (full rank 43)");
    }
    Check(gen.ne == old.ne && gen.no == old.no && gen.adj2 == old.adj2 && gen.adj3 == old.adj3 && gen.adj5 == old.adj5 &&
              gen.iota == old.iota && gen.HD_free_rank == old.HD_free_rank && gen.HD_torsion == old.HD_torsion,
          "(LV3-258) the cell fields (W ranks, adjunctions, iota, H^1(D) free rank / torsion) == ReadCell258");
    for (int s = 0; s < 2; ++s) {
        const hidapair::BlockSign& O = old.sign[s];
        const mblock::MBlockSign& G = gen.sign[s];
        const std::string ts = "(LV3-258) " + G.name;
        Check(G.complete, ts + ": the generic reader completes (" + G.refusal + ")");
        bool cp = G.charpoly_c.size() == hidapair::R191Quartic().size();
        for (size_t i = 0; cp && i < G.charpoly_c.size(); ++i) cp = G.charpoly_c[i] % R8.q == R8.from_int(hidapair::R191Quartic()[i]);
        Check(G.rank_c == O.rank_c && G.rank_D == O.rank_D && cp,
              ts + ": block ranks == ReadCell258's, charpoly == R19-1's P mod 37^8");
        Check(G.tr1 == O.tr1 && G.tr2 == O.tr2 && G.tr3 == O.tr3 && G.dr1 == O.dr1 && G.dr2 == O.dr2 && G.dr3 == O.dr3,
              ts + ": t-ranks == ReadCell258's");
        Check(G.gram_pivots == O.gram_pivots && G.gram_unimodular == O.gram_unimodular, ts + ": Gram pivots == ReadCell258's");
        Check(G.p9_rank_WD == O.p9_rank_WD && G.p9_intertwines == O.p9_intertwines && G.p9_index_pivots == O.p9_index_pivots && G.have_p9,
              ts + ": P9 (rank W_D, intertwining, [H^1(D)_m : H^1_!(D)_m] pivots) == ReadCell258's");
        Check(G.newton_segments == 1 && G.forms.size() == 4, ts + ": one Newton segment (slope 1/2), four roots");
        // identify each generic root with R19-1's factor (P_i(rho) == 0), then compare per factor
        std::map<int, std::vector<std::vector<uint32_t>>> gv, ov;
        bool ident = true;
        for (const mblock::MBlockForm& f : G.forms) {
            const mblock::LocalOrder& Ord = f.O;
            int which = 0;
            for (uint32_t i = 1; i <= 2; ++i) {
                hidapair::Vec Pi;
                for (int64_t c : hidapair::R191Factor(i)) Pi.push_back(Ord.R().from_int(c));
                // R19-1's factors are known mod 37^8 only: P_i(rho) == 0 to v_pi >= 2 * 8 - 2
                if (Ord.vpi(mblock::EvalPoly(Ord, Pi, f.rho)) >= std::min<uint32_t>(Ord.cap(), 16) - 2) which = static_cast<int>(i);
            }
            ident = ident && which != 0;
            gv[which].push_back({f.mixed, f.p9, f.vpi_dP, f.content_c});
        }
        for (const hidapair::BlockEigen& f : O.forms) {
            const int which = f.name.find("f_(1") == 0 ? 1 : 2;
            ov[which].push_back({f.mixed, f.p9, f.vpi_dP, f.content_c});
        }
        for (auto& kv : gv) std::sort(kv.second.begin(), kv.second.end());
        for (auto& kv : ov) std::sort(kv.second.begin(), kv.second.end());
        Check(ident && gv.size() == 2 && gv[1].size() == 2 && gv[2].size() == 2,
              ts + ": each generic root is a root of R19-1's P_1 or P_2, two each (the factorisation recovered, not read)");
        Check(gv == ov, ts + ": per factor, the multisets (mixed, P9, v_pi P'(rho), content) == ReadCell258's (bit for bit)");
        std::cout << "  LV3_258 " << G.name << " orders=";
        for (const auto& f : G.forms) std::cout << "[" << f.order << "]";
        std::cout << " values(mixed,P9,vP',cont)=";
        for (const auto& f : G.forms) std::cout << " (" << f.mixed << "," << f.p9 << "," << f.vpi_dP << "," << f.content_c << ")";
        std::cout << " c_f(def:lvlattice, v_pi)=";
        for (const auto& f : G.forms) std::cout << " " << (f.have_lattice ? std::to_string(f.lattice_change) : std::string("?"));
        std::cout << std::endl;
        bool lc = true;
        for (const auto& f : G.forms) lc = lc && f.have_lattice && f.d_lattice == 0 && f.lattice_change == static_cast<int32_t>(f.e * gen.v_nfact);
        Check(lc && gen.v_nfact == 6, ts + ": [E] def:lvlattice, R20-3's prediction: V' = V here, d_f = 0, c_f = 6 (v_37 units; e v(258!) = 12 in v_pi) on every eigenform");
    }
    // the route reads the generic reader: same values as WP-BIL's frozen expectations (mixed 3/2, P9 3/3, norms over 4 forms)
    {
        Domain d;
        d.n_sym = 258; d.ell = 37; d.m = 10; d.level = 1;
        d.cohomology = "H^1_c(M_n) x H^1(D_n)";
        d.subquotient = "m=(37,T_2-29) geo-even";
        const IndexStatement Ie = HidaPairingRoute(d).index_statement();
        d.subquotient = "m=(37,T_2-29) geo-odd";
        const IndexStatement Io = HidaPairingRoute(d).index_statement();
        d.cohomology = "H^1_c(M_n) x H^1_!(D_n)";
        const IndexStatement Jo = HidaPairingRoute(d).index_statement();
        Check(Ie.holds && Ie.found == 6 && Io.holds && Io.found == 4 && Jo.holds && Jo.found == 6,
              "(LV3-258) the route (generic reader): mixed norm 6 (free sign, 4 x 3/2) and 4 (m-sign, 4 x 2/2), P9 6; " + Io.to_string());
        d.cohomology = "H^1_c(M_n) x H^1(D_n)";
        d.subquotient = "m=(37,T_2-29) geo-even";
        const NormalisationRecord N = HidaPairingRoute(d).normalisation();
        bool computed = false;
        for (const auto& c : N.constants)
            if (c.name.find("change of lattice") != std::string::npos) computed = c.value.find("not computed") == std::string::npos &&
                                                                                c.value.find("open") == std::string::npos;
        Check(computed, "(LV3-258) the NormalisationRecord carries R20-3's constant computed (no longer 'open')");
        std::cout << "  NORMALISATION (H) (258,37) " << N.to_string() << std::endl;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  LV3_258 seconds=" << secs << std::endl;
}

// ---- depth 2: (H2), (L1), the cross-kind gate ----------------------------------------------------------------------------
struct DepthCell { uint32_t n; uint64_t ell; uint32_t v; };

void ArmDepthTwo(const CheckFn& Check) {
    const std::vector<DepthCell> cells{{282, 37, 2}, {226, 103, 2}, {30, 37, 1}, {10, 691, 1}};
    for (const DepthCell& c : cells) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::string tc = "(LV3-D2) " + Cell(c.n, c.ell);
        const uint32_t k = c.n + 2, M = 6;
        try {
            const hidapair::HidaPairingCell C(c.n, c.ell, M, {2, 3, 5});
            const hidapair::Zq& R = C.R();
            mblock::MBlockOptions opt;
            opt.hd_torsion = false;
            opt.lattice_change = true;
            opt.joint = mblock::EisensteinJoint(R, c.n);   // m_Eis = (ell, T_p - 1 - p^{k-1} : p = 2, 3, 5)
            const mblock::MBlockReading B = mblock::ReadMBlockCell(C, mblock::EisensteinT2(R, c.n), opt);
            const mblock::MBlockSign& E = B.sign[0];   // geo-even: contains p_0
            const mblock::MBlockSign& O = B.sign[1];
            Check(E.complete && O.complete, tc + ": the reader completes on both signs (" + E.refusal + O.refusal + ")");
            Check(C.torsion_even().empty() && C.torsion_odd().empty(), tc + ": W = H^1_c(V) is torsion-free");
            Check(E.rank_c == 2 && O.rank_c == 1, tc + ": m_Eis blocks of rank 2 (<p_0, f>, geo-even) and 1 (f, geo-odd)");
            const mblock::MBlockForm* fE = nullptr;
            const mblock::MBlockForm* fB = nullptr;
            for (const auto& f : E.forms) (f.eisenstein ? fB : fE) = &f;
            Check(fE != nullptr && fB != nullptr && E.forms.size() == 2, tc + ": geo-even: one boundary line (e_2) and one cusp form f");
            if (!fE || !fB || O.forms.size() != 1) continue;
            const mblock::MBlockForm& fO = O.forms[0];
            // (H2): v = depth on the p_0 sign, 0 on the other; each == the lemma's congruence module of its lattice
            Check(fE->mixed == c.v, tc + ": (H2) v<g_f, h_f> = " + std::to_string(c.v) + " on the p_0 sign (got " + std::to_string(fE->mixed) + ")");
            Check(fO.mixed == 0, tc + ": (H2) v = 0 on the other sign (got " + std::to_string(fO.mixed) + ")");
            Check(fE->mixed == fE->cong && fO.mixed == fO.cong && fB->mixed == fB->cong,
                  tc + ": every pairing value == the congruence module of W_m at it (v_pi P'(rho) - content, from A_c alone)");
            // the congruence kind on the cusp line: L / sum_p (T_p - e_p) L
            Check(O.cong_length == c.v, tc + ": congruence: length T(W_m^{odd}) / I_e = " + std::to_string(c.v) + " (got " +
                                            std::to_string(O.cong_length) + ")");
            // the linear kind: (L1) on <p_0, r^+> read by BirchModularSymbolRoute on the group-cocycle W over Z/ell^3
            uint32_t depth = 0;
            bool l1 = false;
            {
                birch::BirchConeCell bc;
                bc.n = c.n;
                bc.ell = c.ell;
                bc.m = 3;
                bc.R = birch::MakeRing(c.ell, 3);
                for (const hidapair::Vec& w : C.W()) {
                    birch::Vec p(w.size());
                    for (size_t i = 0; i < w.size(); ++i) p[i] = w[i] % bc.R.q;
                    bc.P.push_back(p);
                    bc.exps.push_back(3);
                }
                bc.route = "group-cocycle W = ker(1+S) cap ker(1+U+U^2) (hidapair::HidaPairingCell), not the C++ cone";
                const auto cell = std::make_shared<const birch::BirchConeCell>(bc);
                const BirchModularSymbolRoute route(cell, birch::TrivialCharacter(bc.R), 0);
                const IndexStatement I = route.index_statement();
                depth = I.found;
                l1 = I.holds;
                std::cout << "  " << "LV3_L1 " << Cell(c.n, c.ell) << " " << I.to_string() << std::endl;
            }
            Check(l1 && depth == c.v, tc + ": (L1) r^+ == unit * p_0 to depth " + std::to_string(c.v) + " (linear, R19-8; got " + std::to_string(depth) + ")");
            const BernoulliReading Bk = BernoulliModEllPowerExact(k, c.ell, 3);
            Check(Bk.v_exact && Bk.v == c.v, tc + ": v(B_k/k) = " + std::to_string(c.v));
            Check(O.cong_length == fE->mixed && fE->mixed == depth && depth == Bk.v,
                  tc + ": CROSS-KIND: congruence = bilinear = linear = v(B_k/k) = " + std::to_string(c.v));
            // R20-3's constant: n < ell - 1 -> theta is an isomorphism, v = 0; else computed
            // R20-3's predictions: d_f = 0 on both signs, for f and for E_k; c = v(n!) (7 at (282,37), 2 at (226,103), 0 at the controls)
            bool lc = fE->have_lattice && fO.have_lattice && fB->have_lattice && fE->d_lattice == 0 && fO.d_lattice == 0 && fB->d_lattice == 0;
            uint32_t vf = 0;
            for (uint64_t pw = c.ell; pw <= c.n; pw *= c.ell) vf += static_cast<uint32_t>(c.n / pw);
            lc = lc && fE->lattice_change == static_cast<int32_t>(vf) && fO.lattice_change == static_cast<int32_t>(vf) &&
                 fB->lattice_change == static_cast<int32_t>(vf);
            if (c.n < c.ell) lc = lc && fE->lattice_change == 0 && fO.lattice_change == 0;
            Check(lc, tc + ": def:lvlattice's c_f (theta = Phi_n) = v(n!) = " + std::to_string(vf) + ", d_f = 0 for f and E_k on both signs (R20-3)");
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "  LV3_DEPTH n=" << c.n << " ell=" << c.ell << " v(B_k/k)=" << Bk.v << " congruence=" << O.cong_length
                      << " bilinear(p0-sign)=" << fE->mixed << " bilinear(other)=" << fO.mixed << " linear(L1)=" << depth
                      << " c_f(def:lvlattice; even,odd)=" << fE->lattice_change << "," << fO.lattice_change << " seconds=" << secs << std::endl;
        } catch (const std::exception& e) {
            Check(false, tc + " threw: " + e.what());
        }
    }
}

// ---- iota over Z/ell^k ---------------------------------------------------------------------------------------------------
template <class Elt>
struct RingCoh {
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh;
};
template <class Elt>
void BuildRingCoh(RingCoh<Elt>& rc, const SimplicialComplex& K, uint32_t n, uint64_t ell, uint32_t k) {
    rc.ring.reset(new hecke::ScopedCoefficientRing(k));
    const engine::ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeLValue LV3-I", ell, k);
    rc.lssc.reset(new LocalSystemSimplicialComplexT<Elt>(K, hecke::GenerateSL2ZRelationsRing<Elt>(K, n, ctx), n + 1, ctx.p));
    rc.coh.reset(new EquivariantCohomologyOnKT<Elt>(rc.lssc.get()));
    rc.coh->compute_cohomology();
}

void ArmIotaRing(const CheckFn& Check) {
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const hecke::IotaSimplicialMap im = hecke::BuildIotaSimplicialMapOnModel(*Kx, *model);
    struct IC { uint32_t n; uint64_t ell; uint32_t k; bool torsion; };
    for (const IC& c : std::vector<IC>{{22, 37, 3, false}, {30, 37, 3, false}, {38, 37, 3, true}}) {
        const std::string tc = "(LV3-I) " + Cell(c.n, c.ell) + " Z/" + std::to_string(c.ell) + "^" + std::to_string(c.k);
        try {
            RingCoh<uint32_t> r32;
            RingCoh<uint64_t> r64;
            hecke::RingInvolution J32, J64;
            {
                BuildRingCoh(r32, *Kx, c.n, c.ell, c.k);
                J32 = hecke::BuildRingInvolution(*r32.coh, im, c.n, static_cast<uint32_t>(c.ell), "LV3-I");
                r32 = RingCoh<uint32_t>();
            }
            {
                BuildRingCoh(r64, *Kx, c.n, c.ell, c.k);
                J64 = hecke::BuildRingInvolution(*r64.coh, im, c.n, static_cast<uint32_t>(c.ell), "LV3-I");
                r64 = RingCoh<uint64_t>();
            }
            Check(J32.to_json() == J64.to_json(), tc + ": the uint32_t and uint64_t instantiations agree (" + J32.to_json() + ")");
            const hecke::RingInvolutionDegree& D = J32.degrees.at(1);
            const uint32_t kk = c.n + 2;
            const uint32_t d_k = (kk % 12 == 2) ? kk / 12 : kk / 12 + 1;
            const auto count_k = [&](const std::vector<uint32_t>& f) {
                return static_cast<uint32_t>(std::count(f.begin(), f.end(), c.k));
            };
            Check(count_k(D.plus_factors) == d_k && count_k(D.minus_factors) == d_k - 1,
                  tc + ": the free part splits d_k / d_k - 1 = " + std::to_string(d_k) + " / " + std::to_string(d_k - 1) +
                      " ([C] def:iota(iv): + is the Eisenstein side)");
            Check(D.length_plus + D.length_minus == D.length, tc + ": length H^1_+ + length H^1_- = length H^1");
            if (c.torsion) {
                const bool tors = D.exps.size() > 2 * d_k - 1;
                Check(tors, tc + ": H^1 has ell-torsion here, and it is split between the signs (" +
                                std::to_string(D.plus_factors.size() + D.minus_factors.size()) + " summands)");
            }
            // k = 1: the F_ell involution's dims (the field HeckeInvolution, the production object)
            if (!c.torsion) {
                const hecke::ScopedCoefficientRing field(1);
                std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
                for (const auto& d : GenerateSL2ZRelationsWithGroupData(*Kx, c.n, static_cast<uint32_t>(c.ell))) rels.push_back(d.rel);
                LocalSystemSimplicialComplex L(*Kx, std::move(rels), c.n + 1, static_cast<uint32_t>(c.ell));
                EquivariantCohomologyOnK coh(&L);
                coh.compute_cohomology();
                const auto inv = hecke_ext::BuildProductionInvolution(coh, *Kx, *model, c.n, static_cast<uint32_t>(c.ell), "LV3-I field");
                bool same = true;
                for (uint32_t q = 0; q < inv->num_degrees() && q < J32.degrees.size(); ++q)
                    same = same && inv->dim_H_plus(q) == J32.degrees[q].plus_factors.size() &&
                           inv->dim_H_minus(q) == J32.degrees[q].minus_factors.size();
                Check(same, tc + ": torsion-free: #summands of H^q_+- over Z/ell^k == dim H^q_+- of the F_ell HeckeInvolution");
            }
            std::cout << "  LV3_IOTA_RING n=" << c.n << " ell=" << c.ell << " k=" << c.k << " " << J32.to_json() << std::endl;
        } catch (const std::exception& e) {
            Check(false, tc + " threw: " + e.what());
        }
    }
    // ell = 3: refused by name
    {
        bool refused = false;
        try {
            RingCoh<uint32_t> r;
            BuildRingCoh(r, *Kx, 4, 3, 2);
            (void)hecke::BuildRingInvolution(*r.coh, im, 4, 3, "LV3-I ell=3");
        } catch (const hecke::NotImplementedInThisBuild&) {
            refused = true;
        } catch (...) {
            refused = false;
        }
        Check(refused, "(LV3-I) negative: ell = 3 is refused by name (NotImplementedInThisBuild)");
    }
}
} // namespace

// (LV3-I heavy) iota^* on H^1(Sym^282, Z/37^3) directly (the reading console script 049 leaves out for ConsoleDet's budget).
void RunLV3HeavyArms(const CheckFn& Check) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const hecke::IotaSimplicialMap im = hecke::BuildIotaSimplicialMapOnModel(*Kx, *model);
    try {
        RingCoh<uint32_t> r;
        BuildRingCoh(r, *Kx, 282, 37, 3);
        const hecke::RingInvolution J = hecke::BuildRingInvolution(*r.coh, im, 282, 37, "LV3-I heavy");
        const hecke::RingInvolutionDegree& D = J.degrees.at(1);
        const bool ok = D.plus_factors == std::vector<uint32_t>(24, 3) && D.minus_factors == std::vector<uint32_t>(23, 3);
        std::cout << "  LV3_IOTA_RING_282 " << J.to_json() << " seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << std::endl;
        Check(ok, "(LV3-I heavy) (282,37) Z/37^3: H^1_+ = (Z/37^3)^24, H^1_- = (Z/37^3)^23 (d_284 = 24), read directly");
    } catch (const std::exception& e) {
        Check(false, std::string("(LV3-I heavy) (282,37) threw: ") + e.what());
    }
}

// (LV3-L) R20-3's change of lattice on the whole lattice at its prototype cells and the n >= ell cells of R19-9 Sec. 1.2:
// the Smith valuations of Theta = (Phi_n/n!)^{-1} : H^1(D)/tors -> H^1(V)/tors and the def:lvlattice whole-lattice length of Phi_n,* (printed; gated: injective, >= 0, both 0 for n < ell).
void ArmLatticeIndex(const CheckFn& Check) {
    struct LC { uint32_t n; uint64_t ell; };
    for (const LC& c : std::vector<LC>{{10, 5}, {10, 7}, {12, 5}, {24, 7}, {40, 37}, {72, 37}, {30, 37}}) {
        try {
            const hidapair::HidaPairingCell C(c.n, c.ell, 4, {2});
            mblock::MBlockOptions opt;
            opt.hd_torsion = false;
            opt.lattice_change = true;
            opt.primes = {2};
            const mblock::MBlockReading B = mblock::ReadMBlockCell(C, mblock::EisensteinT2(C.R(), c.n), opt);
            uint32_t idx = 0;
            for (uint32_t v : B.theta_index_pivots) idx += v;
            std::cout << "  LV3_THETA_INDEX n=" << c.n << " ell=" << c.ell << " rank=" << B.ne + B.no << " pivots=" << B.theta_index_pivots.size()
                      << " v_ell(n!)=" << B.v_nfact << " v_ell[H^1(D)/tors : Phi_n,*(H^1(V)/tors)]=" << B.phi_index
                      << " v_ell[H^1(V)/tors : Theta(H^1(D)/tors)]=" << idx << std::endl;
            Check(B.theta_index_pivots.size() == B.ne + B.no && B.phi_index >= 0 && (c.n >= c.ell || (idx == 0 && B.phi_index == 0)),
                  "(LV3-L) " + Cell(c.n, c.ell) + ": Theta is injective on free parts" + std::string(c.n < c.ell ? ", Phi_n an isomorphism (n < ell)" : ""));
        } catch (const std::exception& e) {
            Check(false, "(LV3-L) " + Cell(c.n, c.ell) + " threw: " + e.what());
        }
    }
}

// (LV3-L negative control, R20-3) at (40, 37) a generic eigenline has d_f >= 1 (the m_Eis zero is not vacuous): every residue
// lambda_0 of T_2 mod 37, every complete block, the largest d_f.
void ArmLatticeNegative(const CheckFn& Check) {
    try {
        const hidapair::HidaPairingCell C(40, 37, 6, {2});
        uint32_t dmax = 0, lines = 0;
        std::ostringstream os;
        for (uint64_t a = 0; a < 37; ++a) {
            mblock::MBlockOptions opt;
            opt.hd_torsion = false;
            opt.lattice_change = true;
            opt.primes = {2};
            const mblock::MBlockReading B = mblock::ReadMBlockCell(C, a, opt);
            for (const auto& S : B.sign)
                for (const auto& f : S.forms)
                    if (f.have_lattice) {
                        ++lines;
                        dmax = std::max(dmax, f.d_lattice / f.e);
                        if (f.d_lattice) os << " (a=" << a << "," << S.name << ",d=" << f.d_lattice << ",e=" << f.e << ")";
                    }
        }
        std::cout << "  LV3_LATTICE_NEG (40,37) eigenlines=" << lines << " max d_f=" << dmax << os.str() << std::endl;
        Check(dmax >= 1, "(LV3-L) negative control: at (40,37) some generic eigenline has d_f >= 1 (R20-3), so d_f = 0 at m_Eis is not vacuous");
    } catch (const std::exception& e) {
        Check(false, std::string("(LV3-L) (40,37) negative control threw: ") + e.what());
    }
}

void RunLV3Arms(const CheckFn& Check) {
    const auto t0 = std::chrono::steady_clock::now();
    ArmLatticeIndex(Check);
    ArmLatticeNegative(Check);
    ArmReaderPlanted(Check);
    ArmCell258(Check);
    ArmDepthTwo(Check);
    ArmIotaRing(Check);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  LV3_ARMS seconds=" << secs << std::endl;
}
