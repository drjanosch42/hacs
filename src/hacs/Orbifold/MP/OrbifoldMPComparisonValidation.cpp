// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) OrbifoldMPComparisonValidation -- see the header.
// Gate values: docs/implementation/slot22/MP_predictions.md P4.3 and "MP-2 predictions" P4.5 (before the first run);
// primary oracle the frozen docs/implementation/slot22/mp/audit/mv_check.out (MvCheckOracle.inc).
#include "OrbifoldMPComparisonValidation.h"

#include "CochainMaps.h"
#include "Comparison.h"
#include "MP2Common.h"
#include "Models.h"
#include "Nerve.h"

#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../SimplicialCohomology.h"
#include "../../Integral/Model/IsotropyModel.h"
#include "../../Integral/Model/IsotropyModelEll.h"
#include "../../Integral/Model/PeriodicityClass.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/FlatMatrix.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using orbifold_mp::Descriptor;
using orbifold_mp::InternedMPSet;
using orbifold_mp::MPSide;
using orbifold_mp::PhiLift;
using orbifold_mp::SourceH;
using orbifold_mp::SparseRows;
using orbifold_mp::Vec;

// The shipped Bredon complex C_eq on the 7-vertex K (the pipeline's own bases), as HeckeIsotropyModelValidation reads it.
struct PipelineRun {
    std::vector<uint32_t> dims;              // q = 0, 1, 2
    std::vector<std::vector<Vec>> H;         // flat H_eq basis (cocycles) per q
};

Vec Flatten(const FlatMatrix& f) {
    Vec v(static_cast<size_t>(f.get_rows()) * f.get_cols());
    for (uint32_t s = 0; s < f.get_rows(); ++s)
        for (uint32_t c = 0; c < f.get_cols(); ++c) v[static_cast<size_t>(s) * f.get_cols() + c] = f.get_entry(s, c);
    return v;
}

PipelineRun RunPipeline(const SimplicialComplex& K, uint32_t n, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, rels, n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    PipelineRun out;
    out.dims = coh.get_H_eq_dimensions();
    out.dims.resize(3, 0);
    const auto& Hb = coh.get_H_eq_bases_fm();
    out.H.resize(3);
    for (uint32_t q = 0; q < 3; ++q)
        if (q < Hb.size())
            for (const auto& f : Hb[q]) out.H[q].push_back(Flatten(f));
    return out;
}

std::string Tri(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    return os.str();
}

}  // namespace

uint32_t OrbifoldMPComparisonValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    orbifold_mp::MPTally T;
    std::cout << "OrbifoldMPComparisonValidation: MP-2, MP_predictions.md P4.3 + P4.5 (model=mp)\n";

    const orbifold_mp::MPModels MM = orbifold_mp::BuildM7AndRZ();
    const std::vector<const Descriptor*> tris = {&MM.m7, &MM.rzd};
    const integral_model::IsotropyModel CI(3);   // [C, def:isomodel] on M7, built to degree 4
    const integral_model::PeriodicityClass P(CI);  // its contracting homotopy fill()

    // ---- P4.5a: Phi, certified ---------------------------------------------------------------------------------
    std::vector<std::unique_ptr<InternedMPSet>> S;
    std::vector<std::unique_ptr<PhiLift>> Phi;
    const std::vector<uint64_t> gens = {5 + 127 + 2541 + 53143, 7 + 165 + 2927 + 56349};
    for (size_t t = 0; t < tris.size(); ++t) {
        S.push_back(std::make_unique<InternedMPSet>(*tris[t], 3));
        Phi.push_back(std::make_unique<PhiLift>(*S[t], *tris[t], P, 3));
        const orbifold_mp::PhiCertificate c = Phi[t]->certify();
        std::cout << "  PHI model=mp tri=" << tris[t]->name << " target=Iso(M7) degrees=0..3 generators=" << c.generators_checked
                  << " terms=" << Phi[t]->total_terms() << " distinct_elements=" << Phi[t]->distinct_elements()
                  << " max_abs_coeff=" << Phi[t]->max_abs_coefficient() << " build_s=" << std::fixed << std::setprecision(2)
                  << Phi[t]->build_seconds() << std::defaultfloat << " certify: aug=" << c.aug << " chain_map=" << c.chain_map << "\n";
        T.check(c.all(), "P4.5a Phi certified (aug, d Phi = Phi d over Z[Gamma]) on " + tris[t]->name + ": " + c.first_failure);
        T.check(c.generators_checked == gens[t], "P4.5a Phi certified on every non-degenerate generator of S_<=3(" + tris[t]->name + ")");
    }

    // ---- P4.5b: the cochain-map identity on the matrices --------------------------------------------------------
    {
        uint32_t ok = 0, total = 0;
        for (size_t t = 0; t < tris.size(); ++t)
            for (const auto& [ell, k] : std::vector<std::pair<uint32_t, uint32_t>>{{2, 1}, {3, 1}, {2, 2}}) {
                const uint32_t m = orbifold_mp::PowU(ell, k);
                const ModCtx<uint32_t> ctx = orbifold_mp::RingCtx(ell, k);
                for (uint32_t n = 0; n <= 2; ++n)
                    for (uint32_t q = 0; q <= 2; ++q) {
                        const SparseRows L = orbifold_mp::SpMul(orbifold_mp::Coboundary(*S[t], q, n, m), Phi[t]->Pullback(q, n, ell, k), m);
                        const SparseRows R = orbifold_mp::SpMul(Phi[t]->Pullback(q + 1, n, ell, k),
                                                                orbifold_mp::FromDense(CI.coboundary(ctx, n, q), m), m);
                        const bool eq = orbifold_mp::SpEqual(L, R);
                        T.check(eq, "P4.5b Phi^*_{q+1} delta_I^q = d^q_MP Phi^*_q, " + tris[t]->name + " Z/" + std::to_string(m) +
                                        " n=" + std::to_string(n) + " q=" + std::to_string(q));
                        ok += eq ? 1 : 0;
                        ++total;
                    }
            }
        std::cout << "  PHI_COCHAIN_MAP model=mp (M7, R_Z) x (F_2, F_3, Z/4) x n<=2 x q<=2: " << ok << "/" << total << " exact\n";
    }

    uint32_t oracle_cells = 0, oracle_agree = 0;
    // ---- P4.5c: Phi^* an isomorphism on H^q over F_ell -----------------------------------------------------------
    for (size_t t = 0; t < tris.size(); ++t)
        for (uint32_t ell : {2u, 3u, 5u, 7u}) {
            const ModCtx<uint32_t> ctx(ell);
            std::ostringstream line;
            line << "  PHI_ISO model=mp tri=" << tris[t]->name << " ring=F_" << ell << " n:H^q (C_I=MP, image=iso)";
            uint32_t iso_cells = 0, cells = 0;
            for (uint32_t n = 0; n <= 12; ++n) {
                const uint32_t qtop = (ell != 7 && n <= 3) ? 2 : 1;
                const MPSide side(*S[t], n, ell, 1, qtop);
                const auto& O = orbifold_mp::FrozenRow(ell, n);
                std::vector<uint32_t> hs;
                bool all = true;
                for (uint32_t q = 0; q <= qtop; ++q) {
                    const DenseBuf<uint32_t> dq = CI.coboundary(ctx, n, q);
                    DenseBuf<uint32_t> dp;
                    if (q) dp = CI.coboundary(ctx, n, q - 1);
                    const SourceH src = orbifold_mp::Source(ctx, dq, q ? &dp : nullptr, nullptr);
                    const auto img = side.image(q, orbifold_mp::MapAll(Phi[t]->Pullback(q, n, ell, 1), src.cocycles, ell));
                    const int64_t lmp = side.lenH(q);
                    const bool ok = img.cocycles && img.image() == src.lenH() && static_cast<int64_t>(src.lenH()) == lmp &&
                                    lmp == static_cast<int64_t>(O.dims[q]);
                    T.check(ok, "P4.5c Phi^* iso H^" + std::to_string(q) + "(C_I) -> H^" + std::to_string(q) + "(MP) " + tris[t]->name +
                                    " F_" + std::to_string(ell) + " n=" + std::to_string(n) + " (C_I " + std::to_string(src.lenH()) +
                                    ", MP " + std::to_string(lmp) + ", image " + std::to_string(img.image()) + ", oracle " +
                                    std::to_string(O.dims[q]) + ")");
                    ++oracle_cells;
                    oracle_agree += ok ? 1 : 0;
                    all = all && ok;
                    hs.push_back(static_cast<uint32_t>(lmp));
                }
                ++cells;
                iso_cells += all ? 1 : 0;
                line << " " << n << ":" << Tri(hs);
            }
            line << "  iso=" << iso_cells << "/" << cells;
            std::cout << line.str() << "\n";
        }

    // ---- P4.5d: the same over Z/ell^k --------------------------------------------------------------------------
    for (size_t t = 0; t < tris.size(); ++t) {
        std::ostringstream line;
        line << "  PHI_ISO_ZLK model=mp tri=" << tris[t]->name << " (n,ell,k):lengths H^0..2";
        uint32_t iso_cells = 0, cells = 0;
        for (uint32_t ell : {2u, 3u})
            for (uint32_t k : {2u, 3u})
                for (uint32_t n : {0u, 2u}) {
                    const uint32_t m = orbifold_mp::PowU(ell, k);
                    const ModCtx<uint32_t> ctx = orbifold_mp::RingCtx(ell, k);
                    std::vector<uint32_t> expect;
                    if (n == 0) expect = (ell == 2) ? std::vector<uint32_t>{k, 2, std::min(k, 2u)} : std::vector<uint32_t>{k, 1, 1};
                    else expect = (ell == 2) ? std::vector<uint32_t>{1, k + 1, 2} : std::vector<uint32_t>{0, k, 0};
                    const MPSide side(*S[t], n, ell, k, 2);
                    std::vector<uint32_t> hs;
                    bool all = true;
                    for (uint32_t q = 0; q <= 2; ++q) {
                        const DenseBuf<uint32_t> dq = CI.coboundary(ctx, n, q);
                        DenseBuf<uint32_t> dp;
                        if (q) dp = CI.coboundary(ctx, n, q - 1);
                        const SourceH src = orbifold_mp::Source(ctx, dq, q ? &dp : nullptr, nullptr);
                        const auto img = side.image(q, orbifold_mp::MapAll(Phi[t]->Pullback(q, n, ell, k), src.cocycles, m));
                        const int64_t lmp = side.lenH(q);
                        const bool ok = img.cocycles && img.image() == src.lenH() && static_cast<int64_t>(src.lenH()) == lmp &&
                                        lmp == static_cast<int64_t>(expect[q]);
                        T.check(ok, "P4.5d Phi^* iso on H^" + std::to_string(q) + " over Z/" + std::to_string(ell) + "^" + std::to_string(k) +
                                        " " + tris[t]->name + " n=" + std::to_string(n) + " (C_I " + std::to_string(src.lenH()) + ", MP " +
                                        std::to_string(lmp) + ", image " + std::to_string(img.image()) + ", predicted " +
                                        std::to_string(expect[q]) + ")");
                        all = all && ok;
                        hs.push_back(static_cast<uint32_t>(lmp));
                    }
                    ++cells;
                    iso_cells += all ? 1 : 0;
                    line << " (" << n << "," << ell << "," << k << "):" << Tri(hs);
                }
        line << "  iso=" << iso_cells << "/" << cells;
        std::cout << line.str() << "\n";
    }

    // ---- P4.3 / P4.5f-h: eps^* from the shipped Bredon complex --------------------------------------------------
    {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        struct ECell {
            uint32_t ell, n;
            std::vector<uint32_t> bredon;   // empty: = the frozen row (ell >= 5)
            const char* kind;
        };
        std::vector<ECell> cells;
        for (uint32_t ell : {5u, 7u})
            for (uint32_t n = 0; n <= 12; ++n) cells.push_back({ell, n, {}, "iso"});
        cells.push_back({2, 0, {1, 0, 0}, "defect"});
        cells.push_back({2, 2, {1, 1, 0}, "defect"});
        cells.push_back({3, 0, {1, 0, 0}, "defect"});
        cells.push_back({3, 2, {0, 1, 0}, "defect"});
        for (uint32_t n : {1u, 3u, 5u}) cells.push_back({3, n, {0, 0, 0}, "witness"});
        uint32_t iso_ok = 0, iso_cells = 0;
        for (const ECell& c : cells) {
            const ModCtx<uint32_t> ctx(c.ell);
            const uint32_t qtop = (c.ell >= 5 && c.n > 3) ? 1 : 2;
            const PipelineRun run = RunPipeline(K, c.n, c.ell);
            const MPSide side(*S[0], c.n, c.ell, 1, qtop);
            const auto& O = orbifold_mp::FrozenRow(c.ell, c.n);
            std::vector<uint32_t> br, mp, rk;
            bool all = true;
            for (uint32_t q = 0; q <= qtop; ++q) {
                const DenseBuf<uint32_t> E = CI.comparison(ctx, c.n, q);
                const auto mapped = orbifold_mp::MapAll(Phi[0]->Pullback(q, c.n, c.ell, 1),
                                                        orbifold_mp::MapAllDense(E, run.H[q], c.ell), c.ell);
                const auto img = side.image(q, mapped);
                const int64_t lmp = side.lenH(q);
                const uint32_t want_b = c.bredon.empty() ? O.dims[q] : c.bredon[q];
                const std::string at = " ell=" + std::to_string(c.ell) + " n=" + std::to_string(c.n) + " q=" + std::to_string(q);
                bool ok = T.check(run.dims[q] == want_b, "P4.3/P4.5 shipped Bredon dim H^q_eq = " + std::to_string(want_b) + at +
                                                             " (got " + std::to_string(run.dims[q]) + ")");
                ok = T.check(img.cocycles && img.image() == run.H[q].size(),
                             "P4.5f/g eps^* injective on H^q (rank = Bredon dim)" + at + " (rank " + std::to_string(img.image()) + ")") && ok;
                ok = T.check(lmp == static_cast<int64_t>(O.dims[q]), "P4.5 MP dim H^q = the frozen row" + at) && ok;
                all = all && ok;
                br.push_back(run.dims[q]);
                mp.push_back(static_cast<uint32_t>(lmp));
                rk.push_back(static_cast<uint32_t>(img.image()));
            }
            const bool iso = br == mp && rk == mp;
            if (std::string(c.kind) != "defect") {
                T.check(iso, std::string("P4.5") + (c.kind[0] == 'i' ? "f" : "h") + " eps^* a quasi-isomorphism in q <= " +
                                 std::to_string(qtop) + " at ell=" + std::to_string(c.ell) + " n=" + std::to_string(c.n));
                ++iso_cells;
                iso_ok += (iso && all) ? 1 : 0;
            }
            if (c.ell <= 3) {
                std::vector<int64_t> defect;
                for (size_t q = 0; q < mp.size(); ++q) defect.push_back(static_cast<int64_t>(mp[q]) - rk[q]);
                std::cout << "  EPS_STAR model=mp tri=M7 ring=F_" << c.ell << " n=" << c.n << " kind=" << c.kind << " Bredon(shipped) q=0..2: "
                          << Tri(br) << "  MP: " << Tri(mp) << "  rank eps^*: " << Tri(rk) << "  defect: " << orbifold_mp::JoinV(defect)
                          << (iso ? "  (quasi-iso)" : "") << "\n";
                if (std::string(c.kind) == "defect") {
                    std::vector<int64_t> want;
                    if (c.ell == 2 && c.n == 0) want = {0, 1, 1};
                    if (c.ell == 2 && c.n == 2) want = {0, 1, 2};
                    if (c.ell == 3 && c.n == 0) want = {0, 1, 1};
                    if (c.ell == 3 && c.n == 2) want = {0, 0, 0};
                    T.check(defect == want, "P4.5g the Bredon defect of eps^* at ell=" + std::to_string(c.ell) + " n=" + std::to_string(c.n) +
                                                " = " + orbifold_mp::JoinV(want));
                }
            }
        }
        std::cout << "  EPS_STAR model=mp tri=M7 ring=F_5,F_7 n=0..12 (+ the ell = 3 odd-n witness): quasi-iso cells " << iso_ok << "/"
                  << iso_cells << " (if 6 is a unit; at ell = 3, odd n: an iso although 3 is not a unit -- 'if', not 'iff')\n";
    }

    // ---- P4.5i: the hybrid (LocalModel) = C_I = nerve ------------------------------------------------------------
    for (uint32_t ell : {2u, 3u}) {
        const ModCtx<uint32_t> ctx(ell);
        const integral_model::LocalModel L0(CI, 0), Ll(CI, ell);
        const integral_model::ComparisonMap f = integral_model::StrictComparison(L0, Ll);
        const integral_model::ComparisonCertificate fc = integral_model::CertifyComparison(L0, Ll, f);
        T.check(fc.all(), "P4.5i the strict comparison Iso -> Iso^(" + std::to_string(ell) + ") certified: " + fc.first_failure);
        std::ostringstream line;
        line << "  HYBRID model=LocalModel(ell=" << ell << ") ring=F_" << ell << " n:H^0..2 (hybrid=C_I=oracle; iso to C_I [and to MP, n<=3])";
        uint32_t ok_cells = 0;
        for (uint32_t n = 0; n <= 12; ++n) {
            const std::vector<uint32_t> hl = Ll.cohomology_lengths(ctx, n);
            const std::vector<uint32_t> hi = CI.cohomology_lengths(ctx, n);
            const auto& O = orbifold_mp::FrozenRow(ell, n);
            bool all = true;
            std::vector<uint32_t> hs;
            for (uint32_t q = 0; q <= 2; ++q) {
                const std::string at = " ell=" + std::to_string(ell) + " n=" + std::to_string(n) + " q=" + std::to_string(q);
                bool ok = T.check(hl[q] == hi[q] && hi[q] == O.dims[q], "P4.5i dim H^q(hybrid) = dim H^q(C_I) = oracle" + at);
                ++oracle_cells;
                oracle_agree += ok ? 1 : 0;
                const DenseBuf<uint32_t> dh = Ll.coboundary(ctx, n, q);
                DenseBuf<uint32_t> dph;
                if (q) dph = Ll.coboundary(ctx, n, q - 1);
                const DenseBuf<uint32_t> Ph = Ll.projector(ctx, n, q);
                const SourceH src = orbifold_mp::Source(ctx, dh, q ? &dph : nullptr, &Ph);
                ok = T.check(src.lenH() == hl[q], "P4.5i the hybrid's cocycles / coboundaries give its H^q" + at) && ok;
                const DenseBuf<uint32_t> pb = integral_model::Pullback(ctx, n, q, L0, Ll, f);   // C_hybrid -> C_I
                const auto mappedI = orbifold_mp::MapAllDense(pb, src.cocycles, ell);
                const SparseRows dqI = orbifold_mp::FromDense(CI.coboundary(ctx, n, q), ell);
                SparseRows dpI;
                if (q) dpI = orbifold_mp::FromDense(CI.coboundary(ctx, n, q - 1), ell);
                const auto img = orbifold_mp::ImageOfCocycles(mappedI, CI.rank(q) * (n + 1), q ? &dpI : nullptr, &dqI, ell, 1);
                ok = T.check(img.cocycles && img.image() == hl[q], "P4.5i pullback H^q(hybrid) -> H^q(C_I) an isomorphism" + at) && ok;
                if (n <= 3)
                    for (size_t t = 0; t < tris.size(); ++t) {
                        const MPSide side(*S[t], n, ell, 1, q);
                        const auto img2 = side.image(q, orbifold_mp::MapAll(Phi[t]->Pullback(q, n, ell, 1), mappedI, ell));
                        ok = T.check(img2.cocycles && img2.image() == hl[q] && side.lenH(q) == static_cast<int64_t>(hl[q]),
                                     "P4.5i H^q(hybrid) -> H^q(C_I) -> H^q(MP) an isomorphism on " + tris[t]->name + at) && ok;
                    }
                all = all && ok;
                hs.push_back(hl[q]);
            }
            ok_cells += all ? 1 : 0;
            line << " " << n << ":" << Tri(hs);
        }
        line << "  ok=" << ok_cells << "/13";
        std::cout << line.str() << "\n";
    }

    std::cout << "ORACLE mv_check.out cells=" << oracle_cells << " agree=" << oracle_agree << "\n";
    T.check(oracle_cells == oracle_agree, "ORACLE mv_check.out: every compared cell agrees");

    // ---- P4.5e: the negative control (R5(b)) -------------------------------------------------------------------
    {
        Phi[0]->perturb_for_negative_control(1, 0);
        const orbifold_mp::PhiCertificate c = Phi[0]->certify();
        std::cout << "  NC P4.5e model=mp tri=M7 Phi_1(e_s) + 1.e_0 (one string): certify chain_map=" << c.chain_map << " -> "
                  << (c.chain_map ? "NC SILENT" : "NC fires") << " (" << c.first_failure << ")\n";
        T.check(!c.chain_map, "P4.5e NC: a perturbed Phi_1 fails certify()");
    }
    // (the image certificate bites) the zero map in place of Phi^*: image 0 < length H^2 = 1 at F_2, n = 0, M7.
    {
        const ModCtx<uint32_t> ctx(2);
        const MPSide side(*S[0], 0, 2, 1, 2);
        const DenseBuf<uint32_t> dq = CI.coboundary(ctx, 0, 2), dp = CI.coboundary(ctx, 0, 1);
        const SourceH src = orbifold_mp::Source(ctx, dq, &dp, nullptr);
        const std::vector<Vec> zero(src.cocycles.size(), Vec(side.dimC[2], 0));
        const auto img = side.image(2, zero);
        std::cout << "  NC zero-map model=mp tri=M7 F_2 n=0 q=2: image " << img.image() << " vs length H^2(C_I) " << src.lenH() << " -> "
                  << (img.image() != src.lenH() ? "NC fires" : "NC SILENT") << "\n";
        T.check(img.image() != src.lenH(), "NC: the zero map is not an isomorphism H^2(C_I) -> H^2(MP) at F_2, n = 0 (the image certificate bites)");
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPComparisonValidation checks=" << T.checks << " failing=" << T.failing << " wall_s=" << std::fixed
              << std::setprecision(2) << secs << std::defaultfloat << "\n";
    return T.failing == 0 ? 0 : 1;
}
