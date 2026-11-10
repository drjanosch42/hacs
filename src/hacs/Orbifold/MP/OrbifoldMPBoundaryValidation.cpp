// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3) OrbifoldMPBoundaryValidation -- see the header.
// Gate values: docs/implementation/slot22/MP3_predictions.md P5 (committed before the first run).
#include "OrbifoldMPBoundaryValidation.h"

#include "Boundary.h"
#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "MP3Oracle.h"
#include "MPSet.h"
#include "Models.h"
#include "ModuleCochains.h"
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
using orbifold_mp::CIRepSide;
using orbifold_mp::Descriptor;
using orbifold_mp::InternedMPSet;
using orbifold_mp::ModuleRep;
using orbifold_mp::RefinementLabels;
using orbifold_mp::RepSide;
using orbifold_mp::SparseRows;
using orbifold_mp::StringMap;
using orbifold_mp::Vec;

std::string MatStr(const Mat22& g) {
    std::ostringstream os;
    os << "[[" << g.a << "," << g.b << "],[" << g.c << "," << g.d << "]]";
    return os.str();
}

std::string RingName(uint32_t ell, uint32_t e) {
    return e == 1 ? "F_" + std::to_string(ell) : "Z/" + std::to_string(orbifold_mp::PowU(ell, e));
}

std::string Join(const std::vector<int64_t>& v) { return orbifold_mp::JoinV(v); }

// j^*_q with the label factor rho(h) (or dropped).
SparseRows JStar(const InternedMPSet& Sb, const InternedMPSet& SX, const RefinementLabels& L, const StringMap& U, uint32_t q,
                 const ModuleRep& N, bool drop = false) {
    return orbifold_mp::LabelledPullbackRep(Sb, SX, L, U, q, N.dim(), N.dim(), N.m(), [&N](const Mat22& h) { return N.rho(h); }, drop);
}

// The derived j^* ranks of MP3_predictions.md P5.9 (q = 0..3; -1 = OUTPUT, gated only against C_I).
std::vector<int64_t> DerivedRanks(uint32_t ell, uint32_t e, uint32_t n) {
    if (e != 1) return {-1, -1, -1, -1};
    if (ell >= 5 && n % 2 == 0 && n + 2 <= ell) return n == 0 ? std::vector<int64_t>{1, 0, 0, 0} : std::vector<int64_t>{0, 1, 0, 0};
    if (ell >= 3 && n % 2 == 1) return {0, 0, 0, 0};
    if (ell == 2 && n == 0) return {1, 1, 1, -1};
    if (ell == 3 && n == 0) return {1, 1, 0, 0};
    return {-1, -1, -1, -1};
}
// The derived relative lengths H^q(X, dX) (P5.10; -1 = OUTPUT).
std::vector<int64_t> DerivedRel(uint32_t ell, uint32_t e, uint32_t n) {
    if (e != 1) return {-1, -1, -1, -1};
    if (ell >= 5 && n % 2 == 0 && n + 2 <= ell) return n == 0 ? std::vector<int64_t>{0, 0, 1, 0} : std::vector<int64_t>{0, 1, 0, 0};
    if (ell >= 3 && n % 2 == 1) return {0, 0, 0, 0};
    if (ell == 2 && n == 0) return {0, 0, 1, -1};
    if (ell == 3 && n == 0) return {0, 0, 1, 1};
    return {-1, -1, -1, -1};
}

}  // namespace

uint32_t OrbifoldMPBoundaryValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    std::cout << "OrbifoldMPBoundaryValidation: MP-3, MP3_predictions.md P5 (model=mp)\n";
    {
        uint32_t rows = 0;
        const bool ok = orbifold_mp::MP3OracleFileAgrees(&rows);
        // (MP-3b, F-MP3 F2) the sha256 of the file actually read, beside the frozen one; both gated.
        const std::string sha = orbifold_mp::MP3OracleFileSha256();
        const bool sha_ok = sha == orbifold_mp::MP3OracleSha256();
        std::cout << "  ORACLE_FILE mp3_oracle.out sha256=" << sha << " frozen=" << orbifold_mp::MP3OracleSha256() << " sha_match=" << sha_ok
                  << " rows=" << rows << " agree=" << ok << "\n";
        T.check(ok, "P5.0 the tracked mp3_oracle.out equals MP3Oracle.inc row for row");
        T.check(sha_ok && rows == 246, "P5.0 (MP-3b) sha256(tracked mp3_oracle.out) = the frozen value of MP3Oracle.inc, 246 rows");
    }

    const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
    const integral_model::IsotropyModel CI(5);
    const integral_model::PeriodicityClass P(CI);
    uint32_t oracle_cells = 0, oracle_agree = 0, ci_cells = 0, ci_agree = 0, jcells = 0, jagree = 0, cone_cells = 0, cone_agree = 0;

    struct Tri {
        std::string name;
        const integral_model::ModelData* d;
        const Descriptor* D;
    };
    const std::vector<Tri> tris = {{"M7", &MM.base, &MM.m7}, {"R_Z", &MM.rz, &MM.rzd}};
    const std::vector<std::pair<uint32_t, uint32_t>> rings = {{2, 1}, {3, 1}, {5, 1}, {7, 1}, {2, 2}, {2, 3}, {3, 2}};

    for (const Tri& tri : tris) {
        const bool m7 = tri.name == "M7";
        // ---- P5.1 the descriptor ----------------------------------------------------------------------------------
        const Descriptor Db = orbifold_mp::BoundaryDescriptor(*tri.d, *tri.D);
        const auto sc = orbifold_mp::SimplicialComplexProperty(Db);
        T.check(Db.top.size() == 3, "P5.1 " + tri.name + ": three boundary edge representatives");
        T.check(sc.ok && orbifold_mp::InvariantOrder(Db), "P5.1 " + tri.name + ": (SC) and the invariant order on dX");
        // ---- P5.2 / P5.3 the MP set ---------------------------------------------------------------------------------
        const InternedMPSet Sb(Db, 4);
        {
            std::ostringstream line;
            line << "  BDRY_SET model=mp tri=" << tri.name << " |S_k(dX)| (all, nd) k=0..4:";
            bool ok = true;
            for (uint32_t k = 0; k <= 4; ++k) {
                line << " (" << Sb.count_all(k) << "," << Sb.count_nondegenerate(k) << ")";
                ok = ok && Sb.count_all(k) == orbifold_mp::MP3BdryCount(k, false) && Sb.count_nondegenerate(k) == orbifold_mp::MP3BdryCount(k, true);
            }
            std::cout << line.str() << " elements=" << Sb.elements().size() << "\n";
            T.check(ok, "P5.2 " + tri.name + ": |S_k(dX)| = the line model's, k = 0..4");
            const orbifold_mp::MPSet ref(Db, 4);
            bool same = true;
            for (uint32_t k = 0; k <= 4; ++k) same = same && ref.count_all(k) == Sb.count_all(k);
            T.check(same, "P5.3 " + tri.name + ": MPSet (MP-1a) and the interned set agree on |S_k(dX)|, k <= 4");
            bool dd = true;
            uint64_t checked = 0, c = 0;
            for (uint32_t q = 0; q <= 2; ++q) {
                dd = dd && orbifold_mp::FormalDSquared(Sb, q, &c);
                checked += c;
            }
            T.check(dd, "P5.3 " + tri.name + ": d o d = 0 formally on S(dX), q <= 2 (" + std::to_string(checked) + " strings)");
        }
        // ---- P5.4 / P5.5 H^q(S(dX)) = H^q(Gamma_d) --------------------------------------------------------------------
        for (const auto& [ell, e] : rings) {
            const uint32_t nmax = e == 1 ? 12 : 4;
            std::ostringstream line;
            line << "  BDRY_H model=mp tri=" << tri.name << " ring=" << RingName(ell, e) << " n:H^0..3";
            bool all = true;
            for (uint32_t n = 0; n <= nmax; ++n) {
                const ModuleRep N = orbifold_mp::SymRep(n, ell, e);
                const RepSide side(Sb, N, 3);
                const auto& O = orbifold_mp::MP3Bdry(ell, e, n);
                std::vector<int64_t> hs;
                bool ok = true;
                for (uint32_t q = 0; q <= 3; ++q) {
                    hs.push_back(side.lenH(q));
                    ok = ok && side.lenH(q) == O.L[q];
                }
                ++oracle_cells;
                oracle_agree += ok ? 1 : 0;
                T.check(ok, "P5.4 " + tri.name + " H^q(S(dX); Sym^" + std::to_string(n) + " " + RingName(ell, e) + ") = oracle " +
                                Join(hs));
                if (m7) {
                    const ModCtx<uint32_t> ctx = orbifold_mp::RingCtx(ell, e);
                    const std::vector<uint32_t> cil = CI.cohomology_lengths(ctx, n, true);
                    bool ok2 = true;
                    for (uint32_t q = 0; q <= 3; ++q) ok2 = ok2 && static_cast<int64_t>(cil[q]) == O.L[q];
                    ++ci_cells;
                    ci_agree += ok2 ? 1 : 0;
                    T.check(ok2, "P5.5 C_{R_d} (shipped boundary model) H^q = oracle, Sym^" + std::to_string(n) + " " + RingName(ell, e));
                    if (n <= 2 && (e == 1 ? ell <= 3 : ell == 2 && e == 2)) {
                        bool same = true;
                        for (uint32_t q = 0; q <= 2; ++q)
                            same = same && orbifold_mp::SpEqual(side.d[q], orbifold_mp::Coboundary(Sb, q, n, N.m()));
                        T.check(same, "P5.4 CoboundaryRep(Sym^n) = Nerve.h's Coboundary on S(dX), " + RingName(ell, e) + " n=" + std::to_string(n));
                    }
                }
                all = all && ok;
                line << " " << n << ":" << Join(hs);
            }
            std::cout << line.str() << (all ? "  (= oracle)" : "  (MISMATCH)") << "\n";
        }

        // ---- P5.6 / P5.7 the labels of j and the string map -------------------------------------------------------------
        const InternedMPSet SX(*tri.D, 3);
        RefinementLabels L = orbifold_mp::BoundaryLabels(Db, *tri.D);
        {
            uint32_t good = 0;
            for (uint32_t t = 0; t < Db.top.size(); ++t) {
                const auto all = orbifold_mp::ContainingLabels(Db, t, *tri.D);
                bool one = all.size() == 2 && all[0].first == all[1].first && all[1].second == all[0].second.neg();
                good += one ? 1 : 0;
                std::ostringstream es, ts;
                for (uint32_t v : Db.top[t]) es << v << " ";
                for (uint32_t v : tri.D->top[L.coarse[t]]) ts << v << " ";
                std::cout << "  BDRY_LABEL model=mp tri=" << tri.name << " edge[" << t << "]=(" << es.str() << ") in " << tri.name << "[" << L.coarse[t]
                          << "]=(" << ts.str() << ") h=" << MatStr(L.h[t]) << " labels=" << all.size() << "\n";
            }
            T.check(good == 3, "P5.6 " + tri.name + ": every boundary edge lies in exactly one top simplex, labels {h, -h} (3/3)");
        }
        const StringMap U = orbifold_mp::MapStrings(Sb, SX, L, 3);
        std::cout << "  BDRY_STRINGS model=mp tri=" << tri.name << " dX -> X levels 0..3: outside=" << U.outside
                  << " degenerate_images(nd)=" << U.degenerate_images << "\n";
        T.check(U.outside == 0, "P5.7 " + tri.name + ": every string of S_<=3(dX) maps into S(X)");

        // ---- P5.8 j^* a cochain map ---------------------------------------------------------------------------------
        {
            uint32_t ok = 0, total = 0;
            for (const auto& [ell, e] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {3, 1}, {2, 2}})
                for (uint32_t n = 0; n <= 2; ++n) {
                    const ModuleRep N = orbifold_mp::SymRep(n, ell, e);
                    const RepSide X(SX, N, 2, false), B(Sb, N, 2, false);
                    for (uint32_t q = 0; q <= 1; ++q) {
                        const bool eq = orbifold_mp::CommutesExactly(JStar(Sb, SX, L, U, q + 1, N), X.d[q], B.d[q], JStar(Sb, SX, L, U, q, N), N.m());
                        T.check(eq, "P5.8 " + tri.name + " j^* d_X = d_dX j^*, " + RingName(ell, e) + " n=" + std::to_string(n) + " q=" + std::to_string(q));
                        ok += eq ? 1 : 0;
                        ++total;
                    }
                }
            std::cout << "  JSTAR_COCHAIN_MAP model=mp tri=" << tri.name << " (F_2, F_3, Z/4) x n<=2 x q<=1: " << ok << "/" << total << " exact\n";
        }

        // ---- P5.9 rank of j^* on H^q = the restriction H^q(Gamma) -> H^q(Gamma_d) (C_I) ------------------------------------
        const orbifold_mp::PhiLift Phi(SX, *tri.D, P, 3);
        T.check(Phi.certify().all(), "P5.9 " + tri.name + ": Phi certified to degree 3 (as P4.5a)");
        std::vector<std::pair<uint32_t, uint32_t>> jrings = m7 ? std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {3, 1}, {5, 1}, {7, 1}, {2, 2}}
                                                               : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {3, 1}};
        for (const auto& [ell, e] : jrings) {
            const uint32_t nmax = !m7 ? 2 : (e == 1 ? 6 : 3);
            std::ostringstream line;
            line << "  JSTAR_RANK model=mp tri=" << tri.name << " ring=" << RingName(ell, e) << " n:rank j^* on H^0..3 (MP = C_I)";
            for (uint32_t n = 0; n <= nmax; ++n) {
                const ModuleRep N = orbifold_mp::SymRep(n, ell, e);
                const CIRepSide ciX(CI, N, 3, false), ciB(CI, N, 3, true);
                const RepSide B(Sb, N, 3);
                const std::vector<int64_t> der = DerivedRanks(ell, e, n);
                std::vector<int64_t> rk;
                bool ok = true;
                for (uint32_t q = 0; q <= 3; ++q) {
                    const auto onX = orbifold_mp::MapAll(orbifold_mp::PhiPullbackRep(Phi, SX, q, CI.rank(q), N), ciX.src[q].cocycles, N.m());
                    const auto onB = orbifold_mp::MapAll(JStar(Sb, SX, L, U, q, N), onX, N.m());
                    const auto img = B.image(q, onB);
                    const auto ciimg = ciB.image(q, orbifold_mp::MapAllDense(orbifold_mp::CIRestrictToBoundary(CI, q, N.dim()), ciX.src[q].cocycles, N.m()));
                    const int64_t r = static_cast<int64_t>(img.image());
                    rk.push_back(r);
                    ok = ok && img.cocycles && ciimg.cocycles && r == static_cast<int64_t>(ciimg.image()) && (der[q] < 0 || der[q] == r);
                }
                ++jcells;
                jagree += ok ? 1 : 0;
                T.check(ok, "P5.9 " + tri.name + " rank j^* on H^q = C_I restriction rank (and derived) " + RingName(ell, e) + " n=" +
                                std::to_string(n) + " " + Join(rk));
                line << " " << n << ":" << Join(rk);
            }
            std::cout << line.str() << "\n";
        }

        // ---- P5.10 the cone of j^* (the pair): exact, = C_I's cone -------------------------------------------------------
        if (m7) {
            for (const auto& [ell, e] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {3, 1}, {5, 1}, {2, 2}}) {
                const uint32_t nmax = e == 1 ? 4 : 2;
                std::ostringstream line;
                line << "  BDRY_CONE model=mp tri=M7 ring=" << RingName(ell, e) << " n:H^0..2(X,dX) (MP = LES = C_I cone)";
                for (uint32_t n = 0; n <= nmax; ++n) {
                    const ModuleRep N = orbifold_mp::SymRep(n, ell, e);
                    const ModCtx<uint32_t> ctx = N.ctx();
                    const RepSide X(SX, N, 2, false), B(Sb, N, 3);
                    const CIRepSide ciX(CI, N, 3, false), ciB(CI, N, 3, true);
                    std::vector<SparseRows> cone;
                    std::vector<uint64_t> cdim;
                    for (uint32_t q = 0; q <= 2; ++q) {
                        cone.push_back(orbifold_mp::ConeDifferential(X.d[q], JStar(Sb, SX, L, U, q, N), q ? &B.d[q - 1] : nullptr,
                                                                     q ? static_cast<uint32_t>(B.dimC[q - 1]) : 0, N.m()));
                        cdim.push_back(SX.count_nondegenerate(q) * N.dim() + (q ? B.dimC[q - 1] : 0));
                    }
                    const std::vector<int64_t> hc = orbifold_mp::LengthsOf(cone, cdim, ell, e);
                    std::vector<DenseBuf<uint32_t>> dcone;
                    std::vector<uint32_t> ddim;
                    for (uint32_t q = 0; q <= 3; ++q) {
                        const DenseBuf<uint32_t> R = orbifold_mp::CIRestrictToBoundary(CI, q, N.dim());
                        dcone.push_back(orbifold_mp::DenseCone(ciX.d[q], R, q ? &ciB.d[q - 1] : nullptr, q ? ciB.dimC[q - 1] : 0, N.m()));
                        ddim.push_back(ciX.dimC[q] + (q ? ciB.dimC[q - 1] : 0));
                    }
                    const std::vector<int64_t> hci = orbifold_mp::DenseLengthsOf(ctx, dcone, ddim);
                    // the LES from the ranks (C_I's X side, the MP boundary side, the MP j^* ranks)
                    std::vector<int64_t> rk;
                    for (uint32_t q = 0; q <= 2; ++q) {
                        const auto onX = orbifold_mp::MapAll(orbifold_mp::PhiPullbackRep(Phi, SX, q, CI.rank(q), N), ciX.src[q].cocycles, N.m());
                        rk.push_back(static_cast<int64_t>(B.image(q, orbifold_mp::MapAll(JStar(Sb, SX, L, U, q, N), onX, N.m())).image()));
                    }
                    const std::vector<int64_t> der = DerivedRel(ell, e, n);
                    bool ok = true;
                    for (uint32_t q = 0; q <= 2; ++q) {
                        const int64_t les = (q ? B.lenH(q - 1) - rk[q - 1] : 0) + (ciX.lenH(q) - rk[q]);
                        ok = ok && hc[q] == les && hc[q] == hci[q] && (der[q] < 0 || der[q] == hc[q]);
                    }
                    ++cone_cells;
                    cone_agree += ok ? 1 : 0;
                    T.check(ok, "P5.10 the cone of j^* is exact (MP = LES = C_I cone, = derived) " + RingName(ell, e) + " n=" + std::to_string(n) + " " +
                                    Join(hc) + " C_I q<=3 " + Join(hci));
                    // lem:ell2unbounded at ell = 2: H^q(dX), H^q(X), H^q(X, dX) all non-zero for q = 2, 3
                    if (ell == 2 && e == 1) {
                        bool nz = B.lenH(2) > 0 && B.lenH(3) > 0 && ciX.lenH(2) > 0 && ciX.lenH(3) > 0 && hc[2] > 0 && hci[3] > 0;
                        T.check(nz, "P5.11 lem:ell2unbounded F_2 n=" + std::to_string(n) + ": H^q(dX), H^q(X), H^q(X,dX) non-zero at q = 2, 3");
                    }
                    line << " " << n << ":" << Join(hc) << "|CI" << Join(hci);
                }
                std::cout << line.str() << "\n";
            }
            // P5.10b the cone in degree 3 on the nerve, n = 0, F_2 and F_3 (S_4(M7))
            {
                const InternedMPSet SX4(*tri.D, 4);
                const StringMap U4 = orbifold_mp::MapStrings(Sb, SX4, L, 3);
                for (uint32_t ell : {2u, 3u}) {
                    const ModuleRep N = orbifold_mp::SymRep(0, ell, 1);
                    const RepSide X(SX4, N, 3, false), B(Sb, N, 3, false);
                    std::vector<SparseRows> cone;
                    std::vector<uint64_t> cdim;
                    for (uint32_t q = 0; q <= 3; ++q) {
                        cone.push_back(orbifold_mp::ConeDifferential(X.d[q], JStar(Sb, SX4, L, U4, q, N), q ? &B.d[q - 1] : nullptr,
                                                                     q ? static_cast<uint32_t>(B.dimC[q - 1]) : 0, N.m()));
                        cdim.push_back(SX4.count_nondegenerate(q) + (q ? B.dimC[q - 1] : 0));
                    }
                    const auto ts = std::chrono::steady_clock::now();
                    const std::vector<int64_t> hc = orbifold_mp::LengthsOf(cone, cdim, ell, 1);
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
                    const std::vector<int64_t> der = DerivedRel(ell, 1, 0);
                    const CIRepSide ciX(CI, N, 4, false), ciB(CI, N, 4, true);
                    std::vector<DenseBuf<uint32_t>> dcone;
                    std::vector<uint32_t> ddim;
                    for (uint32_t q = 0; q <= 3; ++q) {
                        dcone.push_back(orbifold_mp::DenseCone(ciX.d[q], orbifold_mp::CIRestrictToBoundary(CI, q, 1), q ? &ciB.d[q - 1] : nullptr,
                                                               q ? ciB.dimC[q - 1] : 0, N.m()));
                        ddim.push_back(ciX.dimC[q] + (q ? ciB.dimC[q - 1] : 0));
                    }
                    const std::vector<int64_t> hci = orbifold_mp::DenseLengthsOf(N.ctx(), dcone, ddim);
                    bool ok = true;
                    for (uint32_t q = 0; q <= 3; ++q) ok = ok && hc[q] == hci[q] && (der[q] < 0 || der[q] == hc[q]);
                    std::cout << "  BDRY_CONE_H3 model=mp tri=M7 ring=F_" << ell << " n=0 H^0..3(X,dX)=" << Join(hc) << " C_I=" << Join(hci)
                              << " d^3_cone rows=" << cone[3].rows() << " nnz=" << cone[3].nnz() << " wall_s=" << std::fixed << std::setprecision(2) << s
                              << std::defaultfloat << "\n";
                    T.check(ok, "P5.10b the cone of j^* in degree 3 on the nerve = C_I's cone (= derived), F_" + std::to_string(ell) + " n=0");
                    if (ell == 2) T.check(hc[3] > 0, "P5.11 lem:ell2unbounded: H^3(X, dX; F_2) != 0 on the nerve");
                }
            }
        }

        // ---- P5.12 negative controls --------------------------------------------------------------------------------
        if (m7) {
            // (a) the wrong boundary stabiliser: {I} at the horocycle vertices
            const Descriptor Dpl = orbifold_mp::BoundaryDescriptor(*tri.d, *tri.D, true);
            const InternedMPSet Spl(Dpl, 4);
            const RepSide side(Spl, orbifold_mp::SymRep(0, 2, 1), 3);
            std::vector<int64_t> hs;
            for (uint32_t q = 0; q <= 3; ++q) hs.push_back(side.lenH(q));
            const auto& O = orbifold_mp::MP3Bdry(2, 1, 0);
            const bool differs = hs != std::vector<int64_t>{O.L[0], O.L[1], O.L[2], O.L[3]};
            std::cout << "  NC P5.12a model=mp planted trivial horocycle stabiliser: |S_1|=" << Spl.count_all(1) << " H^0..3(F_2, n=0)=" << Join(hs)
                      << " (oracle " << Join({O.L[0], O.L[1], O.L[2], O.L[3]}) << ") -> " << (differs ? "NC fires" : "NC SILENT") << "\n";
            T.check(differs && hs == std::vector<int64_t>{1, 1, 0, 0} && Spl.count_all(1) == 9,
                    "P5.12a NC: the wrong boundary stabiliser gives H^.(<T>; F_2) = [1,1,0,0] and |S_1| = 9, not the oracle");
            // (b) a label off by T
            RefinementLabels Lbad = L;
            Lbad.h[0] = Lbad.h[0] * Mat22::T();
            const bool cont = orbifold_mp::ContainedInSimplex(Db, 0, *tri.D, Lbad.coarse[0], Lbad.h[0]);
            const StringMap Ubad = orbifold_mp::MapStrings(Sb, SX, Lbad, 3);
            std::cout << "  NC P5.12b model=mp label h.T on edge 0: contained=" << cont << " strings outside S(X)=" << Ubad.outside << " -> "
                      << (!cont && Ubad.outside > 0 ? "NC fires" : "NC SILENT") << "\n";
            T.check(!cont && Ubad.outside > 0, "P5.12b NC: a label off by T fails containment and leaves S(X)");
            // (c) mixed labels (h.(-I) on edge 0 only), F_3, n = 1: with the label factor a cochain map, without it not
            RefinementLabels Lmix = L;
            Lmix.h[0] = Lmix.h[0].neg();
            const StringMap Umix = orbifold_mp::MapStrings(Sb, SX, Lmix, 3);
            const ModuleRep N = orbifold_mp::SymRep(1, 3, 1);
            const RepSide X(SX, N, 2, false), B(Sb, N, 2, false);
            bool tw = true, untw = true;
            for (uint32_t q = 0; q <= 1; ++q) {
                tw = tw && orbifold_mp::CommutesExactly(JStar(Sb, SX, Lmix, Umix, q + 1, N), X.d[q], B.d[q], JStar(Sb, SX, Lmix, Umix, q, N), 3);
                untw = untw && orbifold_mp::CommutesExactly(JStar(Sb, SX, Lmix, Umix, q + 1, N, true), X.d[q], B.d[q],
                                                            JStar(Sb, SX, Lmix, Umix, q, N, true), 3);
            }
            std::cout << "  NC P5.12c model=mp mixed labels (h.(-I) on edge 0), F_3, n=1, q<=1: with rho(h) cochain map=" << tw
                      << "; label factor dropped cochain map=" << untw << " -> " << (untw ? "NC SILENT" : "NC fires") << "\n";
            T.check(tw, "P5.12c j^* with the label factor is a cochain map for mixed labels");
            T.check(!untw, "P5.12c NC: dropping the label factor breaks the cochain map for mixed labels at odd n");
        }
    }
    std::cout << "ORACLE mp3_oracle.out cells=" << oracle_cells << " agree=" << oracle_agree << "\n";
    std::cout << "  BDRY summary: C_{R_d} = oracle " << ci_agree << "/" << ci_cells << "; j^* rank = C_I restriction " << jagree << "/" << jcells
              << "; cone exact = C_I cone " << cone_agree << "/" << cone_cells << "\n";

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPBoundaryValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
