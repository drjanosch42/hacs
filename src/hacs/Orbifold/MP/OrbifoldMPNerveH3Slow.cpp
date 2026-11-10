// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) OrbifoldMPNerveH3Slow -- see the header.
// Gate values: docs/implementation/slot22/MP_predictions.md P4.4 (+ the MP-2 detail, before the first run);
// primary oracle the frozen docs/implementation/slot22/mp/audit/mv_check.out (MvCheckOracle.inc).
#include "OrbifoldMPNerveH3Slow.h"

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "MPSet.h"
#include "Models.h"
#include "Nerve.h"

#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/PeriodicityClass.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

uint32_t OrbifoldMPNerveH3Slow() {
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    std::cout << "OrbifoldMPNerveH3Slow: MP-2, MP_predictions.md P4.4 (model=mp)\n";
    std::cout << "EFFECTIVE OrbifoldMPNerveH3Slow arm=default cells=8 (ell in {2,3}, n in {0,2}, M7 and R_Z; nerve H^3 on S_4)\n";

    const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
    const std::vector<const orbifold_mp::Descriptor*> tris = {&MM.m7, &MM.rzd};
    const std::vector<uint64_t> all4 = {1394560, 1434432};
    const integral_model::IsotropyModel CI(3);
    const integral_model::PeriodicityClass P(CI);
    uint32_t oracle_cells = 0, oracle_agree = 0;
    for (size_t t = 0; t < tris.size(); ++t) {
        const orbifold_mp::Descriptor& D = *tris[t];
        const orbifold_mp::InternedMPSet S(D, 4);
        // F-MP1 F3: the independent enumerator, string for string, k = 4.
        {
            const auto ts = std::chrono::steady_clock::now();
            std::set<std::vector<int64_t>> a, b;
            {
                const orbifold_mp::MPSet ref(D, 4);
                for (const auto& s : ref.strings(4)) a.insert(s.encode());
                T.check(ref.count_all(4) == S.count_all(4), "F-MP1 F3 |S_4| (MPSet) = |S_4| (interned) on " + D.name);
            }
            for (uint64_t i = 0; i < S.count_all(4); ++i) b.insert(S.to_string(4, i).encode());
            const bool eq = a == b;
            std::cout << "  F3 model=mp tri=" << D.name << " MPSet(D,4) |S_4|=" << a.size() << " interned |S_4|=" << S.count_all(4)
                      << " equal_sets=" << eq << " wall_s=" << std::fixed << std::setprecision(2)
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count() << std::defaultfloat << "\n";
            T.check(eq && S.count_all(4) == all4[t], "F-MP1 F3 MPSet(D, 4) = the interned S_4 string for string, |S_4| = " +
                                                         std::to_string(all4[t]) + " on " + D.name);
        }
        const orbifold_mp::PhiLift Phi(S, D, P, 3);
        const orbifold_mp::PhiCertificate c = Phi.certify();
        T.check(c.all(), "Phi certified to degree 3 on " + D.name + ": " + c.first_failure);
        for (uint32_t ell : {2u, 3u})
            for (uint32_t n : {0u, 2u}) {
                const engine::ModCtx<uint32_t> ctx(ell);
                const auto tc = std::chrono::steady_clock::now();
                const orbifold_mp::MPSide side(S, n, ell, 1, 3);
                const orbifold_mp::LengthStats& st = side.stats.at(3);
                const auto& O = orbifold_mp::FrozenRow(ell, n);
                const engine::DenseBuf<uint32_t> d3 = CI.coboundary(ctx, n, 3), d2 = CI.coboundary(ctx, n, 2);
                const orbifold_mp::SourceH src = orbifold_mp::Source(ctx, d3, &d2, nullptr);
                const auto img = side.image(3, orbifold_mp::MapAll(Phi.Pullback(3, n, ell, 1), src.cocycles, ell));
                const int64_t h3 = side.lenH(3);
                const std::string at = D.name + " F_" + std::to_string(ell) + " n=" + std::to_string(n);
                const bool okH = T.check(h3 == static_cast<int64_t>(O.dims[3]) && side.lenH(2) == static_cast<int64_t>(O.dims[2]),
                                         "P4.4 nerve H^3 (and H^2) = the frozen row, " + at);
                T.check(img.cocycles && img.image() == src.lenH() && static_cast<int64_t>(src.lenH()) == h3,
                        "P4.4 Phi^* : H^3(C_I) -> H^3(MP) an isomorphism, " + at);
                ++oracle_cells;
                oracle_agree += okH ? 1 : 0;
                std::cout << "  NERVE_H3 model=mp tri=" << D.name << " ring=F_" << ell << " n=" << n << " H^0..3="
                          << orbifold_mp::JoinV({side.lenH(0), side.lenH(1), side.lenH(2), h3}) << " oracle=[" << O.dims[0] << ","
                          << O.dims[1] << "," << O.dims[2] << "," << O.dims[3] << "] Phi^* image H^3=" << img.image() << " d3: rows=" << st.rows
                          << " cols=" << st.cols << " nnz=" << st.input_nnz << " pivots=" << st.pivots << " fill=" << st.pivot_nnz
                          << " wall_s(d3)=" << std::fixed << std::setprecision(2) << st.wall_s << " cell_s="
                          << std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count() << std::defaultfloat << "\n";
            }
    }
    std::cout << "ORACLE mv_check.out cells=" << oracle_cells << " agree=" << oracle_agree << "\n";
    T.check(oracle_cells == 8 && oracle_agree == 8, "ORACLE mv_check.out: the 8 nerve-H^3 cells agree");

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPNerveH3Slow checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
