// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) OrbifoldMPRefinementValidation -- see the header.
// Gate values: docs/implementation/slot22/MP_predictions.md "MP-2 predictions" P4.6 (before the first run).
#include "OrbifoldMPRefinementValidation.h"

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "Models.h"
#include "Nerve.h"
#include "Refinement.h"

#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/PeriodicityClass.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using hecke::Mat22;
using orbifold_mp::InternedMPSet;
using orbifold_mp::MPSide;
using orbifold_mp::RefinementLabels;
using orbifold_mp::SourceH;
using orbifold_mp::SparseRows;
using orbifold_mp::StringMap;
using orbifold_mp::Vec;

std::string MatStr(const Mat22& g) {
    std::ostringstream os;
    os << "[[" << g.a << "," << g.b << "],[" << g.c << "," << g.d << "]]";
    return os.str();
}

std::vector<Vec> Diff(const std::vector<Vec>& a, const std::vector<Vec>& b, uint32_t m) {
    std::vector<Vec> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i].resize(a[i].size());
        for (size_t j = 0; j < a[i].size(); ++j) out[i][j] = (a[i][j] + m - b[i][j]) % m;
    }
    return out;
}

}  // namespace

uint32_t OrbifoldMPRefinementValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    std::cout << "OrbifoldMPRefinementValidation: MP-2, MP_predictions.md P4.6 (model=mp)\n";

    const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
    const InternedMPSet Sc(MM.m7, 3), Sf(MM.rzd, 3);
    const integral_model::IsotropyModel CI(3);
    const integral_model::PeriodicityClass P(CI);
    const orbifold_mp::PhiLift PhiC(Sc, MM.m7, P, 2), PhiF(Sf, MM.rzd, P, 2);
    T.check(PhiC.certify().all() && PhiF.certify().all(), "Phi_M7, Phi_R_Z certified to degree 2 (as P4.5a)");

    // ---- P4.6a: the labels -------------------------------------------------------------------------------------
    const RefinementLabels L = orbifold_mp::LabelsFromRecord(MM.rz, MM.rzd, MM.base, MM.m7);
    uint32_t contained = 0;
    for (uint32_t t = 0; t < L.h.size(); ++t) {
        const bool a = orbifold_mp::ContainedIn(MM.rz, MM.rzd, MM.m7, t, L.coarse[t], L.h[t]);
        const bool b = orbifold_mp::ContainedIn(MM.rz, MM.rzd, MM.m7, t, L.coarse[t], L.h[t] * Mat22::I().neg());
        contained += (a && b) ? 1 : 0;
        std::ostringstream fs, cs;
        for (uint32_t v : MM.rzd.top[t]) fs << v << " ";
        for (uint32_t v : MM.m7.top[L.coarse[t]]) cs << v << " ";
        std::cout << "  REFINE_LABEL model=mp R_Z[" << t << "]=(" << fs.str() << ") in M7[" << L.coarse[t] << "]=(" << cs.str()
                  << ") h=" << MatStr(L.h[t]) << " contained(h)=" << a << " contained(h.-I)=" << b << "\n";
    }
    T.check(L.h.size() == 7 && contained == 7, "P4.6a every R_Z top simplex lies in one M7 top simplex, label h and h.(-I) (7/7)");

    // ---- P4.6b: the string map ---------------------------------------------------------------------------------
    const StringMap U = orbifold_mp::MapStrings(Sf, Sc, L, 3);
    std::cout << "  REFINE_STRINGS model=mp R_Z -> M7 levels 0..3: outside=" << U.outside << " degenerate_images(nd)="
              << U.degenerate_images << "\n";
    T.check(U.outside == 0, "P4.6b every string of S_<=3(R_Z) maps into S(M7)");

    // ---- P4.6c: u^* a cochain map --------------------------------------------------------------------------------
    {
        uint32_t ok = 0, total = 0;
        for (const auto& [ell, k] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {3, 1}, {2, 2}}) {
            const uint32_t m = orbifold_mp::PowU(ell, k);
            for (uint32_t n = 0; n <= 2; ++n)
                for (uint32_t q = 0; q <= 2; ++q) {
                    const SparseRows A = orbifold_mp::SpMul(orbifold_mp::Coboundary(Sf, q, n, m),
                                                            orbifold_mp::RefinementPullback(Sf, Sc, L, U, q, n, ell, k), m);
                    const SparseRows B = orbifold_mp::SpMul(orbifold_mp::RefinementPullback(Sf, Sc, L, U, q + 1, n, ell, k),
                                                            orbifold_mp::Coboundary(Sc, q, n, m), m);
                    const bool eq = orbifold_mp::SpEqual(A, B);
                    T.check(eq, "P4.6c u^*_{q+1} d^q_M7 = d^q_R_Z u^*_q over Z/" + std::to_string(m) + " n=" + std::to_string(n) +
                                    " q=" + std::to_string(q));
                    ok += eq ? 1 : 0;
                    ++total;
                }
        }
        std::cout << "  REFINE_COCHAIN_MAP model=mp (F_2, F_3, Z/4) x n<=2 x q<=2: " << ok << "/" << total << " exact\n";
    }
    // The twist rho(h_{s~_0}) is load-bearing once the labels differ by -I between simplices and n is odd: with h.(-I) on
    // R_Z[0] only, u^* is a cochain map (F_3, n = 1, q <= 2), and the untwisted formula is not (the NC).
    {
        RefinementLabels Lone = L;
        Lone.h[0] = Lone.h[0] * Mat22::I().neg();
        const StringMap Uo = orbifold_mp::MapStrings(Sf, Sc, Lone, 3);
        bool twisted_ok = true, untwisted_ok = true;
        for (uint32_t q = 0; q <= 2; ++q)
            for (bool tw : {true, false}) {
                const SparseRows A = orbifold_mp::SpMul(orbifold_mp::Coboundary(Sf, q, 1, 3),
                                                        orbifold_mp::RefinementPullback(Sf, Sc, Lone, Uo, q, 1, 3, 1, tw), 3);
                const SparseRows B = orbifold_mp::SpMul(orbifold_mp::RefinementPullback(Sf, Sc, Lone, Uo, q + 1, 1, 3, 1, tw),
                                                        orbifold_mp::Coboundary(Sc, q, 1, 3), 3);
                bool& acc = tw ? twisted_ok : untwisted_ok;
                acc = acc && orbifold_mp::SpEqual(A, B);
            }
        std::cout << "  REFINE_TWIST model=mp label h.(-I) on R_Z[0] only, F_3, n=1, q<=2: twisted u^* cochain map=" << twisted_ok
                  << "; untwisted (NC) cochain map=" << untwisted_ok << " -> " << (untwisted_ok ? "NC SILENT" : "NC fires") << "\n";
        T.check(twisted_ok, "P4.6c u^* (with rho(h_{s~_0})) a cochain map for mixed labels, F_3, n = 1");
        T.check(!untwisted_ok, "NC: dropping rho(h_{s~_0}) breaks the cochain map for mixed labels, F_3, n = 1");
    }

    // ---- P4.6d / P4.6e: on cohomology ---------------------------------------------------------------------------
    RefinementLabels Lneg = L, Lone = L;
    for (auto& h : Lneg.h) h = h * Mat22::I().neg();
    Lone.h[0] = Lone.h[0] * Mat22::I().neg();
    const StringMap Uneg = orbifold_mp::MapStrings(Sf, Sc, Lneg, 3), Uone = orbifold_mp::MapStrings(Sf, Sc, Lone, 3);
    T.check(Uneg.outside == 0 && Uone.outside == 0, "P4.6e the relabelled maps land in S(M7)");
    uint32_t iso_cells = 0, cells = 0, indep = 0;
    for (uint32_t ell : {2u, 3u, 5u}) {
        const ModCtx<uint32_t> ctx(ell);
        std::ostringstream line;
        line << "  REFINE_ISO model=mp R_Z->M7 ring=F_" << ell << " n:H^0..2 (u^* Phi^*_M7 iso, = Phi^*_R_Z on H, label-independent)";
        for (uint32_t n : {0u, 2u, 4u}) {
            const MPSide side(Sf, n, ell, 1, 2);
            const auto& O = orbifold_mp::FrozenRow(ell, n);
            bool all = true;
            std::vector<int64_t> hs;
            for (uint32_t q = 0; q <= 2; ++q) {
                const DenseBuf<uint32_t> dq = CI.coboundary(ctx, n, q);
                DenseBuf<uint32_t> dp;
                if (q) dp = CI.coboundary(ctx, n, q - 1);
                const SourceH src = orbifold_mp::Source(ctx, dq, q ? &dp : nullptr, nullptr);
                const auto onM7 = orbifold_mp::MapAll(PhiC.Pullback(q, n, ell, 1), src.cocycles, ell);
                const auto viaU = orbifold_mp::MapAll(orbifold_mp::RefinementPullback(Sf, Sc, L, U, q, n, ell, 1), onM7, ell);
                const auto direct = orbifold_mp::MapAll(PhiF.Pullback(q, n, ell, 1), src.cocycles, ell);
                const auto img = side.image(q, viaU);
                const std::string at = " F_" + std::to_string(ell) + " n=" + std::to_string(n) + " q=" + std::to_string(q);
                bool ok = T.check(img.cocycles && img.image() == src.lenH() && side.lenH(q) == static_cast<int64_t>(src.lenH()) &&
                                      src.lenH() == O.dims[q],
                                  "P4.6d u^* Phi^*_M7 iso H^q(C_I) -> H^q(MP(R_Z))" + at);
                ok = T.check(side.image(q, Diff(viaU, direct, ell)).image() == 0, "P4.6d u^* Phi^*_M7 = Phi^*_R_Z on H^q" + at) && ok;
                bool li = true;
                for (const auto* pr : {&Lneg, &Lone}) {
                    const StringMap& Ux = (pr == &Lneg) ? Uneg : Uone;
                    const auto other = orbifold_mp::MapAll(orbifold_mp::RefinementPullback(Sf, Sc, *pr, Ux, q, n, ell, 1), onM7, ell);
                    li = li && side.image(q, Diff(viaU, other, ell)).image() == 0;
                }
                ok = T.check(li, "P4.6e label independence (h.(-I) everywhere / on one simplex) on H^q" + at) && ok;
                indep += li ? 1 : 0;
                all = all && ok;
                hs.push_back(side.lenH(q));
            }
            ++cells;
            iso_cells += all ? 1 : 0;
            line << " " << n << ":" << orbifold_mp::JoinV(hs);
        }
        std::cout << line.str() << "\n";
    }
    std::cout << "  REFINE summary: iso cells " << iso_cells << "/" << cells << ", label-independent (q-cells) " << indep << "/" << 3 * cells
              << "\n";

    // ---- P4.6f: NC, a label off by T ----------------------------------------------------------------------------
    {
        RefinementLabels Lbad = L;
        Lbad.h[0] = Lbad.h[0] * Mat22::T();
        const bool cont = orbifold_mp::ContainedIn(MM.rz, MM.rzd, MM.m7, 0, Lbad.coarse[0], Lbad.h[0]);
        const StringMap Ubad = orbifold_mp::MapStrings(Sf, Sc, Lbad, 3);
        const bool fires = !cont && Ubad.outside > 0;
        std::cout << "  NC P4.6f model=mp label h.T on R_Z[0]: contained=" << cont << " strings outside S(M7)=" << Ubad.outside << " -> "
                  << (fires ? "NC fires" : "NC SILENT") << "\n";
        T.check(fires, "P4.6f NC: a label off by T fails containment and leaves S(M7)");
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPRefinementValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
