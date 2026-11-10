// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeThetaRssDietValidation.cpp -- see the header.
#include "HeckeThetaRssDietValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ModArithmetic/Fnv.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../EquivariantMappingCone.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ModCtx.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"
#include "SL2ZBoundaryPair.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"

namespace {

struct Checker {
    uint64_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) { ++fails; std::cout << "  FAIL: " << what << std::endl; }
    }
};

bool same_flat(const FlatMatrix& a, const FlatMatrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}
bool same_matrix(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}
bool same_block(const hecke::HeckeOperatorOnK::BlockMat& a, const hecke::HeckeOperatorOnK::BlockMat& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (uint32_t i = 0; i < a.rows(); ++i)
        for (uint32_t j = 0; j < a.cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}

// Every point owns its own complex (SimplicialComplex caches Z1 for the first prime it
// is asked for; EngineTapeConsumerValidation.cpp's note).
struct Point {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    uint32_t n = 0, ell = 0;
};

std::unique_ptr<Point> make_point(uint32_t n, uint32_t ell) {
    auto pt = std::make_unique<Point>();
    pt->K = std::make_unique<SimplicialComplex>(CreateSL2ZFundamentalDomainTriangulation());
    pt->model = std::make_unique<hecke::SL2ZTessellationTables>(*pt->K);
    pt->n = n; pt->ell = ell;
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*pt->K, n, ell)) rels.push_back(datum.rel);
    pt->lssc = std::make_unique<LocalSystemSimplicialComplex>(*pt->K, rels, n + 1, ell);
    pt->coh = std::make_unique<EquivariantCohomologyOnK>(pt->lssc.get());
    pt->coh->compute_cohomology();
    return pt;
}

struct Run {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeOperatorOnK> op;
};

std::unique_ptr<Run> run_op(const Point& pt, uint64_t p) {
    auto r = std::make_unique<Run>();
    r->cosets = std::make_unique<hecke::HeckeCosetTables>(p, 1, pt.ell, pt.n);
    r->tau = std::make_unique<hecke::HeckeTauTable>(*pt.K, *pt.model, *r->cosets, pt.ell, pt.n);
    r->op = std::make_unique<hecke::HeckeOperatorOnK>(pt.coh.get(), r->cosets.get(), r->tau.get());
    r->op->compute();
    return r;
}

// ---------------------------------------------------------------------------
// (slot-24 G6) The field-route RSS diet (HeckeOperatorOptions (6), HECKE_FIELD_DIET).
//   (P) program 1 in parts: a small part budget forces several parts at small n;
//       Theta_block / Theta_flat / Theta_eq / A^{(q)} and the certificate
//       counters equal the single-program rollback (ScopedFieldDiet(false))
//       entrywise, certified and memory-lean, for dense T_p (p != ell);
//   (F) HeckeOperatorFamily::release_level_a(): T_p returns the SAME object
//       with the same entries, T_p_pow / T_n built on it are unchanged, a
//       level-(a) accessor (theta_flat_fm) throws std::logic_error;
//   (S) the cone's span predicate by the unit-row route == the dense RREF and
//       == the elimination route (ScopedUnitRowSpan(false)), on in-span,
//       out-of-span and no-unit-row operands; one cone built under each
//       setting has the same golden digest and a certificate that verifies.
bool same_matrices(const std::vector<Matrix>& a, const std::vector<Matrix>& b) {
    if (a.size() != b.size()) return false;
    for (size_t q = 0; q < a.size(); ++q) if (!same_matrix(a[q], b[q])) return false;
    return true;
}

bool g6_arm() {
    namespace Opt = hecke::HeckeOperatorOptions;
    Checker g;
    uint64_t parts_seen = 0, parted_seen = 0, prewalk_skips = 0, prewalk_audits = 0, released = 0, unit_routes = 0, fallbacks = 0, cone_unit_routes = 0;
    // (P)
    struct Cell { uint32_t n, ell; };
    for (const Cell c : {Cell{50, 13}, Cell{64, 101}}) {
        const std::unique_ptr<Point> pt = make_point(c.n, c.ell);
        const uint64_t temp = static_cast<uint64_t>(c.n + 1) * (c.n + 1) * sizeof(uint32_t);
        for (const uint64_t p : {uint64_t{2}, uint64_t{31}}) {
            for (const bool lean_mode : {false, true}) {
                const std::string t = "(G6-P) (n=" + std::to_string(c.n) + ", ell=" + std::to_string(c.ell) + ", p=" +
                                      std::to_string(p) + (lean_mode ? ", lean)" : ", certified)");
                const Opt::ScopedThetaEqCertification cert(!lean_mode);
                const Opt::ScopedMemoryLean lean_on(lean_mode);
                std::unique_ptr<Run> whole, parts;
                {
                    const Opt::ScopedFieldDiet off(false);
                    whole = run_op(*pt, p);
                }
                const Opt::FieldDietStats s0 = Opt::field_diet_stats();
                // (PERF-VF audit S4) the lean-mode runs re-check every bound-decided skip with the pre-walk itself
                // (ScopedPrewalkAudit: an ENGINE_CHECK fires if a skipped pre-walk would have parted); the
                // certified-mode runs take the plain skip path.
                const Opt::ScopedPrewalkAudit audit(lean_mode);
                {
                    const Opt::ScopedFieldDiet on(true);
                    const Opt::ScopedThetaPartBudget budget(2 * temp);   // two shared products per part
                    parts = run_op(*pt, p);
                }
                const Opt::FieldDietStats s1 = Opt::field_diet_stats();
                g.check(s1.parted_degrees > s0.parted_degrees && s1.parts >= s0.parts + 2, t + " program 1 ran in >= 2 parts");
                parted_seen += s1.parted_degrees - s0.parted_degrees;
                parts_seen += s1.parts - s0.parts;
                // (slot-25 PERF-VF (b1)) under the forced small budget only degrees with <= 5 terms (bound <= 2 temps)
                // may skip the pre-walk, and they count as whole; the default budget skips it by the
                // floor(#terms / 2) bound, with the same "whole" outcome and the same A^{(q)}.
                g.check(s1.prewalk_skipped - s0.prewalk_skipped <= s1.whole_degrees - s0.whole_degrees,
                        t + " a pre-walk skipped under the small part budget was not counted whole");
                std::unique_ptr<Run> dflt;
                {
                    const Opt::ScopedFieldDiet on(true);
                    dflt = run_op(*pt, p);
                }
                const Opt::FieldDietStats s2 = Opt::field_diet_stats();
                g.check(s2.prewalk_skipped > s1.prewalk_skipped && s2.parted_degrees == s1.parted_degrees &&
                            s2.whole_degrees - s1.whole_degrees >= s2.prewalk_skipped - s1.prewalk_skipped,
                        t + " the default budget skipped the pre-walk (whole, not parted)");
                prewalk_skips += s2.prewalk_skipped - s1.prewalk_skipped;
                g.check(s2.prewalk_audited - s0.prewalk_audited == (lean_mode ? s2.prewalk_skipped - s0.prewalk_skipped : 0),
                        t + " every pre-walk skip was re-checked under the audit (and none without it)");
                prewalk_audits += s2.prewalk_audited - s0.prewalk_audited;
                g.check(same_matrices(whole->op->get_H_matrices(), dflt->op->get_H_matrices()),
                        t + " A^{(q)} pre-walk skipped == whole");
                const hecke::HeckeOperatorOnK& a = *whole->op;
                const hecke::HeckeOperatorOnK& b = *parts->op;
                g.check(same_matrices(a.get_H_matrices(), b.get_H_matrices()), t + " A^{(q)} parts == whole");
                for (uint32_t q = 0; q < 3; ++q) {
                    g.check(same_block(a.Theta_block(q), b.Theta_block(q)), t + " q=" + std::to_string(q) + " Theta_block parts == whole");
                    if (!lean_mode) {
                        g.check(same_flat(a.get_Theta_flat_matrices()[q], b.get_Theta_flat_matrices()[q]),
                                t + " q=" + std::to_string(q) + " Theta_flat parts == whole");
                        g.check(same_flat(a.get_Theta_eq_matrices()[q], b.get_Theta_eq_matrices()[q]),
                                t + " q=" + std::to_string(q) + " Theta_eq parts == whole");
                    }
                }
                const auto ca = a.certificates(), cb = b.certificates();
                g.check(ca.routes_bit_identical && cb.routes_bit_identical && ca.relation_checks == cb.relation_checks &&
                            ca.cocycle_checks == cb.cocycle_checks && ca.theta_eq_solves == cb.theta_eq_solves &&
                            ca.theta_eq_certified == cb.theta_eq_certified,
                        t + " certificate counters parts == whole");
            }
        }
    }
    // (F)
    {
        const std::unique_ptr<Point> pt = make_point(40, 13);
        hecke::HeckeOperatorFamily fam(pt->coh.get(), *pt->K, *pt->model, 13, 40);
        hecke::HeckeOperatorFamily ref(pt->coh.get(), *pt->K, *pt->model, 13, 40);
        const std::vector<Matrix>& A2 = fam.T_p(2);
        const std::vector<Matrix>* addr = &A2;
        (void)fam.T_p(3);
        const std::vector<Matrix> copy2 = A2;
        const uint32_t r = fam.release_level_a();
        released += r;
        g.check(r == 2, "(G6-F) release_level_a released the two cached pipelines");
        g.check(fam.release_level_a() == 0, "(G6-F) a second release releases nothing");
        g.check(&fam.T_p(2) == addr, "(G6-F) T_p(2) is the same object after the release");
        g.check(same_matrices(fam.T_p(2), copy2) && same_matrices(fam.T_p(2), ref.T_p(2)), "(G6-F) T_p(2) entries unchanged");
        g.check(same_matrices(fam.T_p_pow(3, 2), ref.T_p_pow(3, 2)), "(G6-F) T_{9} by the recursion unchanged");
        g.check(same_matrices(fam.T_n(6), ref.T_n(6)), "(G6-F) T_6 unchanged");
        bool threw = false;
        try { (void)fam.theta_flat_fm(2, 1); } catch (const std::logic_error&) { threw = true; }
        g.check(threw, "(G6-F) theta_flat_fm of a released prime throws std::logic_error");
        g.check(same_matrices(fam.T_p(5), ref.T_p(5)), "(G6-F) a prime built after the release is computed as before");
    }
    // (S)
    {
        const uint32_t ell = 13, a = 1200, nb = 700, ny = 500;
        uint64_t s = 0x9e3779b97f4a7c15ULL;
        auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<uint32_t>(s % ell); };
        // B with unit rows at a scattered row set (a kernel-basis shape), random elsewhere.
        engine::DenseBuf<uint32_t> B(a, nb, engine::Layout::ColMajor);
        std::vector<uint32_t> unit(nb);
        for (uint32_t c = 0; c < nb; ++c) unit[c] = (c * 17u) % a;   // distinct: gcd(17, 1200) = 1
        std::vector<char> is_unit(a, 0);
        for (uint32_t c = 0; c < nb; ++c) is_unit[unit[c]] = 1;
        for (uint32_t c = 0; c < nb; ++c)
            for (uint32_t r = 0; r < a; ++r) B.at(r, c) = is_unit[r] ? (r == unit[c] ? 1u : 0u) : rnd();
        engine::DenseBuf<uint32_t> X(nb, ny, engine::Layout::ColMajor);
        for (uint32_t i = 0; i < nb; ++i) for (uint32_t j = 0; j < ny; ++j) X.at(i, j) = rnd();
        const engine::ModCtx<uint32_t> ctx(ell);
        engine::DenseBuf<uint32_t> Y(a, ny, engine::Layout::ColMajor);
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(B), engine::cview(X), engine::view(Y), engine::kernels::GemmOp::Set);
        engine::DenseBuf<uint32_t> Ybad = Y;
        {   // perturb one entry on a NON-unit row (on a unit row too: both must be caught)
            uint32_t r = 0;
            while (is_unit[r]) ++r;
            Ybad.at(r, ny - 1) = ctx.add(Ybad.at(r, ny - 1), 1);
        }
        engine::DenseBuf<uint32_t> Ybad2 = Y;
        Ybad2.at(unit[3], 0) = ctx.add(Ybad2.at(unit[3], 0), 1);   // changes x, so some other row disagrees (rank 700 < 1200)
        engine::DenseBuf<uint32_t> Bmix(a, nb, engine::Layout::ColMajor);   // B with column 0 doubled: no unit row there
        for (uint32_t c = 0; c < nb; ++c) for (uint32_t r = 0; r < a; ++r) Bmix.at(r, c) = B.at(r, c);
        for (uint32_t r = 0; r < a; ++r) Bmix.at(r, 0) = ctx.add(B.at(r, 0), B.at(r, 0));
        struct Case { const char* name; const engine::DenseBuf<uint32_t>* B; const engine::DenseBuf<uint32_t>* Y; bool want; bool unit_route; };
        for (const Case cs : {Case{"in span", &B, &Y, true, true}, Case{"off a non-unit row", &B, &Ybad, false, true},
                              Case{"off a unit row", &B, &Ybad2, false, true}, Case{"no unit row (fallback)", &Bmix, &Ybad, false, false},
                              Case{"no unit row, in span", &Bmix, &Y, true, false}}) {
            const std::string t = std::string("(G6-S) ") + cs.name;
            ConeOptions::SpanProbe on, off;
            { const ConeOptions::ScopedUnitRowSpan u(true); on = ConeOptions::span_test_for_validation(ell, *cs.B, *cs.Y); }
            { const ConeOptions::ScopedUnitRowSpan u(false); off = ConeOptions::span_test_for_validation(ell, *cs.B, *cs.Y); }
            g.check(on.verdict == cs.want && on.dense == cs.want, t + " unit-row switch on: verdict == dense == expected");
            g.check(off.verdict == cs.want && !off.unit_row_route && !off.fell_back, t + " switch off: the elimination route, same verdict");
            g.check(on.unit_row_route == cs.unit_route && on.fell_back == !cs.unit_route, t + " the route taken");
            if (on.unit_row_route) ++unit_routes;
            if (on.fell_back) ++fallbacks;
        }
        // One cone per setting: same digest, a certificate that verifies.
        // (G6 audit F3) At n = 96 the cone's span tests are above the tape
        // threshold, so the unit-row route really decides them in c_on (at
        // n = 60 (and 64) it never ran: the cone arm compared the elimination route
        // with itself); every stored object is compared, so the shared-Z degree
        // (D_q without rows) is checked against the product copy too.
        const std::unique_ptr<Point> pt = make_point(96, 13);
        {   // (M) the boundary morphism's induced maps: lean basis read == the FlatMatrix list read
            std::unique_ptr<hecke::SL2ZBoundaryPair> p_on, p_off;
            { const Opt::ScopedFieldDiet on(true); p_on = std::make_unique<hecke::SL2ZBoundaryPair>(pt->coh.get(), *pt->K); }
            const std::unique_ptr<Point> pt2 = make_point(96, 13);   // a fresh bulk: its FlatMatrix cache unbuilt
            { const Opt::ScopedFieldDiet off(false); p_off = std::make_unique<hecke::SL2ZBoundaryPair>(pt2->coh.get(), *pt2->K); }
            const auto& m_on = p_on->restriction().get_induced_H_eq_maps();
            const auto& m_off = p_off->restriction().get_induced_H_eq_maps();
            bool same = m_on.size() == m_off.size() && m_on.size() >= 2;
            for (size_t q = 0; same && q < m_on.size(); ++q) same = same_flat(m_on[q], m_off[q]);
            g.check(same, "(G6-M) boundary restriction H^q maps: lean basis read == FlatMatrix list read");
            g.check(m_on.size() >= 2 && m_on[1].get_rows() > 0 && m_on[1].get_cols() > 0, "(G6-M) the H^1 map is non-empty");
        }
        hecke::SL2ZBoundaryPair pair(pt->coh.get(), *pt->K);
        std::unique_ptr<EquivariantMappingCone> c_on, c_off;
        const ConeOptions::UnitRowStats u0 = ConeOptions::unit_row_stats();
        { const ConeOptions::ScopedUnitRowSpan u(true); c_on = pair.make_cone(); }
        const ConeOptions::UnitRowStats u1 = ConeOptions::unit_row_stats();
        { const ConeOptions::ScopedUnitRowSpan u(false); c_off = pair.make_cone(); }
        const ConeOptions::UnitRowStats u2 = ConeOptions::unit_row_stats();
        g.check(u1.calls >= u0.calls + 3, "(G6-S) the unit-row route decided the cone's span tests (diet on)");
        g.check(u2.calls == u1.calls && u2.fallbacks == u1.fallbacks, "(G6-S) the unit-row route did not run (diet off)");
        cone_unit_routes += u1.calls - u0.calls;
        g.check(c_on->golden_digest() == c_off->golden_digest(), "(G6-S) cone golden digest: unit-row route == elimination route");
        {
            bool same = c_on->q_max() == c_off->q_max(), shared_degree = false;
            for (uint32_t q = 0; same && q <= c_on->q_max(); ++q) {
                same = same_block(c_on->Ceq_block(q), c_off->Ceq_block(q)) && same_block(c_on->D_block(q), c_off->D_block(q)) &&
                       same_block(c_on->Z_block(q), c_off->Z_block(q)) && same_block(c_on->B_block(q), c_off->B_block(q)) &&
                       same_block(c_on->H_block(q), c_off->H_block(q)) && same_block(c_on->P_block(q), c_off->P_block(q));
                if (c_on->D_block(q).rows() == 0 && c_on->Z_block(q).cols() > 0) shared_degree = true;
            }
            g.check(same, "(G6-S) every stored cone object (Ceq, D, Z, B, H, P) equal under both settings");
            g.check(shared_degree, "(G6-S) a degree with an empty differential (Z shared with Ceq) is among them");
        }
        { const ConeOptions::ScopedUnitRowSpan u(true); g.check(c_off->certificate().verify() == 0, "(G6-S) certificate verifies (unit-row route)"); }
        { const ConeOptions::ScopedUnitRowSpan u(false); g.check(c_on->certificate().verify() == 0, "(G6-S) certificate verifies (elimination route)"); }
        {   // (R) release_cochain_objects(): the dims, maps, ranks and spots survive; the objects' accessors throw
            std::vector<uint32_t> dims0 = c_on->dim_Hc_all();
            std::vector<FlatMatrix> pi0, r0, io0;
            std::vector<uint32_t> ranks0;
            for (uint32_t q = 0; q <= c_on->q_max() + 1; ++q) {
                if (c_on->pi_built(q)) { pi0.push_back(c_on->pi(q)); ranks0.push_back(c_on->rank_pi(q)); }
                if (c_on->r_built(q)) { r0.push_back(c_on->r(q)); ranks0.push_back(c_on->rank_r(q)); }
                if (c_on->iota_built(q)) { io0.push_back(c_on->iota(q)); ranks0.push_back(c_on->rank_iota(q)); }
            }
            const size_t spots0 = c_on->les_spots().size();
            c_on->release_cochain_objects();
            std::vector<FlatMatrix> pi1, r1, io1;
            std::vector<uint32_t> ranks1;
            for (uint32_t q = 0; q <= c_on->q_max() + 1; ++q) {
                if (c_on->pi_built(q)) { pi1.push_back(c_on->pi(q)); ranks1.push_back(c_on->rank_pi(q)); }
                if (c_on->r_built(q)) { r1.push_back(c_on->r(q)); ranks1.push_back(c_on->rank_r(q)); }
                if (c_on->iota_built(q)) { io1.push_back(c_on->iota(q)); ranks1.push_back(c_on->rank_iota(q)); }
            }
            bool maps_same = pi0.size() == pi1.size() && r0.size() == r1.size() && io0.size() == io1.size() && ranks0 == ranks1;
            for (size_t i = 0; maps_same && i < pi0.size(); ++i) maps_same = same_flat(pi0[i], pi1[i]);
            for (size_t i = 0; maps_same && i < r0.size(); ++i) maps_same = same_flat(r0[i], r1[i]);
            for (size_t i = 0; maps_same && i < io0.size(); ++i) maps_same = same_flat(io0[i], io1[i]);
            g.check(c_on->cochains_released() && c_on->dim_Hc_all() == dims0 && c_on->les_spots().size() == spots0 && maps_same,
                    "(G6-R) a released cone serves the same dims, maps, ranks and spots");
            bool threw = false;
            try { (void)c_on->certificate(); } catch (const std::logic_error&) { threw = true; }
            bool threw2 = false;
            try { (void)c_on->Z_block(1); } catch (const std::logic_error&) { threw2 = true; }
            g.check(threw && threw2, "(G6-R) certificate() / Z_block() of a released cone throw std::logic_error");
        }
    }
    const bool ok = g.fails == 0;
    std::cout << "G6_FIELD_DIET checks=" << g.checks << " failing=" << g.fails << " parted_degrees=" << parted_seen
              << " parts=" << parts_seen << " prewalk_skips=" << prewalk_skips << " prewalk_audits=" << prewalk_audits << " released_pipelines=" << released << " unit_row_routes=" << unit_routes
              << " fallbacks=" << fallbacks << " cone_unit_routes=" << cone_unit_routes << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    return ok;
}

} // namespace

uint32_t HeckeThetaRssDietValidation() {
    namespace Opt = hecke::HeckeOperatorOptions;
    Checker g;
    // (S24 CL-L2) the (T) tape == legacy comparisons need the legacy arm (OC_ENABLE_LEGACY=ON); an OFF build prints
    // SKIP-ARM and runs the (L) / (C) / G6 checks alone.
    const bool have_legacy = oc::legacy::RunLegacyArm("HeckeThetaRssDietValidation", "T-legacy");
    uint64_t digest = engine::kFnvOffset64;
    uint32_t arms = 0, chunked_arms = 0, chunked3_arms = 0;
    const uint32_t kChunk = 512;   // HeckeOperatorOnK.cpp's kResidChunk
    struct Cell { uint32_t n, ell; };
    for (const Cell c : {Cell{50, 13}, Cell{42, 7}, Cell{64, 101}, Cell{114, 13}}) {
        const std::unique_ptr<Point> pt = make_point(c.n, c.ell);
        const uint32_t dimC1 = pt->coh->C_eq_block(1).cols();
        for (const uint64_t p : {uint64_t{2}, uint64_t{3}, uint64_t{31}, uint64_t{c.ell}}) {
            const std::string t = "(n=" + std::to_string(c.n) + ", ell=" + std::to_string(c.ell) +
                                  ", p=" + std::to_string(p) + ")";
            ++arms;
            if (dimC1 > kChunk && dimC1 % kChunk != 0) ++chunked_arms;
            if (dimC1 > 2 * kChunk && dimC1 % kChunk != 0) ++chunked3_arms;
            std::unique_ptr<Run> tape, legacy, lean;
            {
                const Opt::ScopedThetaEqCertification cert(true);
                tape = run_op(*pt, p);
#if OC_ENABLE_LEGACY
                const Opt::ScopedLegacyHeckeOperatorPath leg(true);
                legacy = run_op(*pt, p);
#endif
            }
            {
                const Opt::ScopedThetaEqCertification no_cert(false);
                const Opt::ScopedMemoryLean lean_on(true);
                lean = run_op(*pt, p);
            }
            const hecke::HeckeOperatorOnK& a = *tape->op;
            const hecke::HeckeOperatorOnK& b = have_legacy ? *legacy->op : a;   // OFF: b is never compared
            const hecke::HeckeOperatorOnK& l = *lean->op;
            const auto& Ta = a.get_Theta_flat_matrices();
            const auto& Tb = b.get_Theta_flat_matrices();
            const auto& Ea = a.get_Theta_eq_matrices();
            const auto& Eb = b.get_Theta_eq_matrices();
            const auto& Aa = a.get_H_matrices();
            const auto& Ab = b.get_H_matrices();
            const auto& Al = l.get_H_matrices();
            g.check(Ta.size() == 3 && Tb.size() == 3 && Ea.size() == 3 && Eb.size() == 3 && Aa.size() == 3 &&
                    Ab.size() == 3 && Al.size() == 3, t + " degree counts");
            if (Ta.size() != 3 || Tb.size() != 3 || Ea.size() != 3 || Eb.size() != 3 || Aa.size() != 3 ||
                Ab.size() != 3 || Al.size() != 3)
                continue;
            for (uint32_t q = 0; q < 3; ++q) {
                const std::string tq = t + " q=" + std::to_string(q);
                if (have_legacy) g.check(same_flat(Ta[q], Tb[q]), tq + " (T) Theta_flat tape == legacy");
                if (have_legacy) g.check(same_flat(Ea[q], Eb[q]), tq + " (T) Theta_eq tape == legacy");
                g.check(Ea[q].get_rows() == pt->coh->C_eq_block(q).cols(), tq + " (T) Theta_eq has dim C_eq rows");
                if (have_legacy) g.check(same_matrix(Aa[q], Ab[q]), tq + " (T) A^{(q)} tape == legacy");
                g.check(same_matrix(Aa[q], Al[q]), tq + " (L) A^{(q)} memory-lean == certified");
                g.check(same_block(a.Theta_block(q), l.Theta_block(q)), tq + " (L) Theta_block memory-lean == certified");
                for (uint32_t i = 0; i < Aa[q].rows; ++i)
                    for (uint32_t j = 0; j < Aa[q].cols; ++j) digest = engine::FnvU64(digest, Aa[q].data[i][j]);
                for (uint32_t i = 0; i < Ea[q].get_rows(); ++i)
                    for (uint32_t j = 0; j < Ea[q].get_cols(); ++j) digest = engine::FnvU64(digest, Ea[q].get_entry(i, j));
            }
            const auto ca = a.certificates(), cb = b.certificates(), cl = l.certificates();
            g.check(ca.routes_bit_identical && cb.routes_bit_identical && cl.routes_bit_identical, t + " O12(a) bit-identity");
            if (have_legacy)
                g.check(ca.relation_checks == cb.relation_checks && ca.cocycle_checks == cb.cocycle_checks,
                        t + " (T) assertion (i) / (ii) counters tape == legacy");
            if (have_legacy)
                g.check(ca.theta_eq_solves == cb.theta_eq_solves && ca.theta_eq_residual_max == 0 && cb.theta_eq_residual_max == 0,
                        t + " (T) assertion (iii) / A-N3 counters tape == legacy");
            g.check(ca.theta_eq_certified && !cl.theta_eq_certified && cl.theta_eq_solves == 0,
                    t + " (L) certification levels");
        }
    }
    g.check(chunked_arms > 0, "(C) some arm runs the chunked residual with a ragged last chunk");
    g.check(chunked3_arms > 0, "(C) some arm runs >= 3 residual chunks with a ragged last chunk");
    const bool ok = g.fails == 0;
    char hex[32];
    std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(digest));
    std::cout << "P4A_RSS_DIET arms=" << arms << " checks=" << g.checks << " failing=" << g.fails
              << " chunked_arms=" << chunked_arms << " chunked3_arms=" << chunked3_arms << " digest=0x" << hex << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    const bool ok6 = g6_arm();
    return (ok && ok6) ? 0u : 1u;
}
