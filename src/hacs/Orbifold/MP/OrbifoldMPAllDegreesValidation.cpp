// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) OrbifoldMPAllDegreesValidation -- see the header.
// Gate values: docs/implementation/slot22/MP_predictions.md P4.1, P4.2 (+ the MP-2 detail, before the first run);
// primary oracle the frozen docs/implementation/slot22/mp/audit/mv_check.out (MvCheckOracle.inc).
#include "OrbifoldMPAllDegreesValidation.h"

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "Models.h"
#include "Nerve.h"

#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/IsotropyModelEll.h"
#include "../../Integral/Model/PeriodicityClass.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using orbifold_mp::Descriptor;
using orbifold_mp::InternedMPSet;
using orbifold_mp::MPSide;
using orbifold_mp::SourceH;
using orbifold_mp::SparseRows;
using orbifold_mp::Vec;

// E^* (scaled by `scale`) : H^q(C_I) -> H^{q+2}(C_I) an isomorphism?  Returns (image, len H^q, len H^{q+2}).
struct PerResult {
    uint64_t image = 0, src = 0, dst = 0;
    bool cocycles = true;
    bool iso() const { return cocycles && image == src && src == dst; }
};
PerResult Periodicity(const integral_model::IsotropyModel& M, const integral_model::PeriodicityClass& P, const ModCtx<uint32_t>& ctx,
                      uint32_t ell, uint32_t n, uint32_t q, uint32_t scale) {
    const uint32_t d = n + 1;
    const DenseBuf<uint32_t> dq = M.coboundary(ctx, n, q), dp = M.coboundary(ctx, n, q - 1);
    const SourceH src = orbifold_mp::Source(ctx, dq, &dp, nullptr);
    DenseBuf<uint32_t> E = P.matrix(ctx, n, q);
    for (uint32_t i = 0; i < E.rows; ++i)
        for (uint32_t j = 0; j < E.cols; ++j) E.at(i, j) = static_cast<uint32_t>((static_cast<uint64_t>(E.at(i, j)) * scale) % ell);
    const auto mapped = orbifold_mp::MapAllDense(E, src.cocycles, ell);
    const SparseRows tp = orbifold_mp::FromDense(M.coboundary(ctx, n, q + 1), ell);
    const SparseRows tq = orbifold_mp::FromDense(M.coboundary(ctx, n, q + 2), ell);
    const auto img = orbifold_mp::ImageOfCocycles(mapped, M.rank(q + 2) * d, &tp, &tq, ell, 1);
    const DenseBuf<uint32_t> dq2 = M.coboundary(ctx, n, q + 2), dp2 = M.coboundary(ctx, n, q + 1);
    const SourceH dst = orbifold_mp::Source(ctx, dq2, &dp2, nullptr);
    PerResult R;
    R.image = img.image();
    R.cocycles = img.cocycles;
    R.src = src.lenH();
    R.dst = dst.lenH();
    return R;
}

std::string Join8(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    return os.str();
}

}  // namespace

uint32_t OrbifoldMPAllDegreesValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    std::cout << "OrbifoldMPAllDegreesValidation: MP-2, MP_predictions.md P4.1 / P4.2 (model=mp; C_I, hybrid)\n";

    const integral_model::IsotropyModel CI(7);   // q_max = 7 (R8): H^0..H^7, E_q for q <= 6
    const integral_model::PeriodicityClass P(CI);
    const integral_model::PeriodicityCertificate pc = P.certify();
    T.check(pc.all(), "PeriodicityClass on q_max = 7 certified (e, E_q chain map over Z[Gamma]): " + pc.first_failure);
    std::cout << "  PERIODICITY_CLASS q_max=7 E_q built for q <= " << P.top() << " certified=" << pc.all() << "\n";

    // ---- P4.1a / P4.1b ---------------------------------------------------------------------------------------
    uint32_t oracle_cells = 0, oracle_agree = 0;
    std::vector<std::vector<bool>> per_ok(8, std::vector<bool>(13, false));   // [ell][n]: E^* iso q = 2..5
    for (uint32_t ell : {2u, 3u, 5u, 7u}) {
        const ModCtx<uint32_t> ctx(ell);
        const integral_model::LocalModel L(CI, ell);
        uint32_t per_cells = 0;
        for (uint32_t n = 0; n <= 12; ++n) {
            const auto& O = orbifold_mp::FrozenRow(ell, n);
            const std::vector<uint32_t> hi = CI.cohomology_lengths(ctx, n);
            const std::vector<uint32_t> hl = L.cohomology_lengths(ctx, n);
            for (uint32_t q = 0; q <= 7; ++q) {
                const std::string at = " F_" + std::to_string(ell) + " n=" + std::to_string(n) + " q=" + std::to_string(q);
                const bool a = T.check(hi.at(q) == O.dims[q], "P4.1a dim H^q(C_I) = oracle" + at);
                const bool b = T.check(hl.at(q) == O.dims[q], "P4.1a dim H^q(hybrid) = oracle" + at);
                oracle_cells += 2;
                oracle_agree += (a ? 1 : 0) + (b ? 1 : 0);
            }
            bool all = true;
            std::ostringstream ps;
            for (uint32_t q = 2; q <= 5; ++q) {
                const PerResult R = Periodicity(CI, P, ctx, ell, n, q, 1);
                all = T.check(R.iso(), "P4.1b E^* : H^" + std::to_string(q) + " -> H^" + std::to_string(q + 2) + " an isomorphism on C_I, F_" +
                                           std::to_string(ell) + " n=" + std::to_string(n)) && all;
                ps << (q > 2 ? "," : "") << R.image;
            }
            per_ok[ell][n] = all;
            per_cells += all ? 1 : 0;
            std::cout << "  ALLDEG model=C_I,hybrid ring=F_" << ell << " n=" << n << " H^0..7 C_I=" << Join8(hi) << " hybrid=" << Join8(hl)
                      << " E^* rank H^2..5->H^4..7=" << ps.str() << (all ? " iso" : " NOT-ISO") << "\n";
        }
        std::cout << "  PERIODIC_C_I ring=F_" << ell << " n=0..12 E^*: H^q -> H^{q+2} iso for q=2..5 certified=" << per_cells << "/13\n";
    }
    std::cout << "ORACLE mv_check.out cells=" << oracle_cells << " agree=" << oracle_agree << "\n";
    T.check(oracle_cells == 832 && oracle_agree == oracle_cells, "ORACLE mv_check.out: C_I and hybrid H^0..7, 52 rows (832 values)");

    // ---- P4.1c: NC, ell.E^* (the zero map) -------------------------------------------------------------------
    for (uint32_t ell : {2u, 3u}) {
        const ModCtx<uint32_t> ctx(ell);
        uint32_t fired = 0, armed = 0;
        for (uint32_t n = 0; n <= 12; ++n) {
            if (orbifold_mp::FrozenRow(ell, n).dims[2] == 0) continue;
            ++armed;
            const PerResult R = Periodicity(CI, P, ctx, ell, n, 2, ell);
            fired += R.iso() ? 0 : 1;
        }
        std::cout << "  NC P4.1c " << ell << ".E^* at F_" << ell << " (cells with H^2 != 0): not iso in " << fired << "/" << armed << "\n";
        T.check(armed > 0 && fired == armed, "P4.1c NC: " + std::to_string(ell) + ".E^* is not an isomorphism H^2 -> H^4 at F_" +
                                                 std::to_string(ell) + " (every cell with H^2 != 0)");
    }

    // ---- P4.2: the certificate line ---------------------------------------------------------------------------
    {
        const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
        const std::vector<const Descriptor*> tris = {&MM.m7, &MM.rzd};
        for (const Descriptor* D : tris) {
            const InternedMPSet S(*D, 3);
            const orbifold_mp::PhiLift Phi(S, *D, P, 2);
            const orbifold_mp::PhiCertificate c = Phi.certify();
            T.check(c.all(), "P4.2 Phi certified to degree 2 on " + D->name + ": " + c.first_failure);
            for (uint32_t ell : {2u, 3u, 5u, 7u}) {
                const ModCtx<uint32_t> ctx(ell);
                uint32_t certified = 0, cells = 0;
                for (uint32_t n = 0; n <= 3; ++n) {
                    const MPSide side(S, n, ell, 1, 2);
                    bool ok = c.all() && per_ok[ell][n];
                    for (uint32_t q = 0; q <= 2; ++q) {
                        const DenseBuf<uint32_t> dq = CI.coboundary(ctx, n, q);
                        DenseBuf<uint32_t> dp;
                        if (q) dp = CI.coboundary(ctx, n, q - 1);
                        const SourceH src = orbifold_mp::Source(ctx, dq, q ? &dp : nullptr, nullptr);
                        const auto img = side.image(q, orbifold_mp::MapAll(Phi.Pullback(q, n, ell, 1), src.cocycles, ell));
                        ok = ok && img.cocycles && img.image() == src.lenH() && side.lenH(q) == static_cast<int64_t>(src.lenH());
                    }
                    T.check(ok, "P4.2 (1) E^* iso on C_I and (2) Phi^* iso on H^{<=2}, " + D->name + " F_" + std::to_string(ell) + " n=" +
                                    std::to_string(n));
                    ++cells;
                    certified += ok ? 1 : 0;
                }
                const bool v = certified == cells;
                std::cout << "MP_ALL_DEGREES model=" << D->name << " ring=F_" << ell << " nerve_to=2 periodic_on=C_I from=2 via=thm:mpcomparison certified="
                          << certified << "/" << cells << " scope=bulk verdict=" << (v ? "OK" : "FAIL") << "\n";
                T.check(v && cells == 4, "P4.2 MP_ALL_DEGREES " + D->name + " F_" + std::to_string(ell) + " certified=4/4");
            }
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPAllDegreesValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
