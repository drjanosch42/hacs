// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeReductionPipelineValidation.h"

// hecke-extensions slot 8, WP-3 (E8 item 3): HeckeReductionValidation's
// pipeline, rho_n, key and refusal arms -- see the header.  Not a registered
// validation of its own: HeckeReductionValidation() calls
// HeckeReductionPipelineArms() after WP-1's elimination arms.  (The file name
// ends in `Validation` so that the include guards and the raw-mod census treat
// it as the test harness it is.)

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

#include "BarCocycleH1.h"
#include "CoefficientRing.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryOperator.h"
#include "HeckeConeAction.h"
#include "HeckeContragredient.h"
#include "HeckeCosetTables.h"
#include "HeckeCrossChecks.h"
#include "HeckeInvolution.h"
#include "HeckeModuleDecomposition.h"
#include "HeckeOperatorFamily.h"
#include "HeckeOperatorOnK.h"
#include "HeckePairing.h"
#include "HeckeReduction.h"
#include "HeckeReport.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZBoundaryPair.h"
#include "SL2ZTessellationTables.h"
#include "../Bench/BenchDriver.h"
#include "../Campaign/CampaignSpec.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../EquivariantMappingCone.h"
#include "../HeckeExt/HeckeAtlasCli.h"
#include "../HeckeExt/HeckeBoundary.h"
#include "../HeckeExt/HeckeExtDriver.h"
#include "../LocalSystemSL2Z.h"
#include "../../Engine/EngineRuntime.h"
#include "../../Engine/TapeElim.h"
#include "../../ModArithmetic/BlockElim.h"
#include "../../ModArithmetic/BlockKernels.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/ModCtx.h"

namespace {

using engine::DenseBuf;
using engine::ModCtx;

uint64_t* g_checks = nullptr;
uint32_t* g_fail = nullptr;

void check(bool ok, const std::string& what) {
    ++*g_checks;
    if (ok) return;
    ++*g_fail;
    if (*g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
}

template<class F>
bool throws_niitb(F&& f, const std::string& must_contain, std::string* msg_out = nullptr) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild& e) {
        const std::string m = e.what();
        if (msg_out) *msg_out = m;
        return m.find(hecke::kNotImplementedInThisBuildTag) == 0 && m.find(must_contain) != std::string::npos;
    } catch (...) { return false; }
    return false;
}
template<class F>
bool throws_logic(F&& f) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild&) { return false; }
    catch (const std::logic_error&) { return true; } catch (...) { return false; }
    return false;
}
template<class F>
bool throws_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream o; o << "(";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ")";
    return o.str();
}

// ---- (K) the key -------------------------------------------------------------
void key_arm() {
    using hecke::MathSettings;
    const std::vector<std::string>& keys = MathSettings::keys();
    check(keys.size() == 11 && keys.back() == "coefficient_ring", "(K) keys() = 11, coefficient_ring at the tail");
    check(MathSettings::defaults().fingerprint() == 0 && MathSettings::defaults().is_default(),
          "(K) the all-defaults fingerprint is still 0");
    check(MathSettings::defaults().text("coefficient_ring") == "F_ell", "(K) the default's canonical text is F_ell");
    check(hecke::math_settings().coefficient_ring.k == 1, "(K) F_ell is in force outside any scope");
    for (const std::string& ok : {std::string("F_ell"), std::string("Z/ell^2"), std::string("Z/ell^3"),
                                  std::string("Z/ell^10"), std::string("Z/ell^63")}) {
        MathSettings s;
        bool fine = false;
        try { fine = s.set("coefficient_ring", ok, hecke::SettingSource::Console); } catch (...) { fine = false; }
        check(fine && s.text("coefficient_ring") == ok, "(K) '" + ok + "' parses and round-trips");
        if (ok != "F_ell")
            check(s.fingerprint() != 0 && !s.is_default() &&
                      s.canonical_text() == "coefficient_ring = " + ok + "\n",
                  "(K) '" + ok + "' moves the fingerprint");
    }
    for (const std::string& bad : {std::string("Z/ell^1"), std::string("Z/ell^0"), std::string("Z/ell^4097")   /* slot-11 WP-18c: the parse cap 63 -> 4096 */,
                                   std::string("Z/ell^02"), std::string("Z/ell^"), std::string("Z/ell^-2"),
                                   std::string("F_5"), std::string("z/ell^2"), std::string(" Z/ell^2"),
                                   std::string("Z/ell^2 "), std::string("Z/0"), std::string("Z/1"),
                                   std::string("Z/025"), std::string("Z/"), std::string(""), std::string("F")}) {
        MathSettings s;
        check(throws_invalid([&] { s.set("coefficient_ring", bad, hecke::SettingSource::Console); }),
              "(K) '" + bad + "' is std::invalid_argument");
        check(s.is_default(), "(K) and leaves the settings untouched");
    }
    // (slot-24 D6) `Z` is UN-RESERVED: stored as k = 0, canonical text "Z", a non-zero fingerprint; every reader of
    // k refuses it by name (the closure under Z is IntegralDescentRecordValidation's arm (F)).
    {
        MathSettings s;
        bool fine = false;
        try { fine = s.set("coefficient_ring", "Z", hecke::SettingSource::Spec); } catch (...) { fine = false; }
        check(fine && s.coefficient_ring.is_integral() && s.text("coefficient_ring") == "Z" && s.fingerprint() != 0 &&
                  !s.is_default(),
              "(K) 'Z' parses (slot-24 D6: un-reserved), k = 0, round-trips, non-zero fingerprint");
        hecke::ScopedCoefficientRing z(0);
        check(throws_niitb([&] { (void)hecke::CoefficientRingK(); }, "coefficient_ring=Z"),
              "(K) under Z the reader of k refuses (NotImplementedInThisBuild naming the value)");
    }
    for (const std::string& res : {std::string("Z/25"), std::string("Z/7"),
                                   std::string("Z/691")}) {
        MathSettings s;
        check(throws_niitb([&] { s.set("coefficient_ring", res, hecke::SettingSource::Spec); }, "coefficient_ring=" + res),
              "(K) the reserved '" + res + "' is refused INSIDE set() by NotImplementedInThisBuild");
        check(s.is_default(), "(K) and is never stored");
    }
    {
        hecke::ScopedCoefficientRing ring(3);
        check(hecke::CoefficientRingK() == 3 && hecke::math_settings().coefficient_ring_source == hecke::SettingSource::Spec,
              "(K) ScopedCoefficientRing(3) installs Z/ell^3, source spec");
    }
    check(hecke::CoefficientRingK() == 1, "(K) and restores F_ell");
    check(hecke::ExactKthRoot(25, 2) == 5 && hecke::ExactKthRoot(125, 3) == 5 && hecke::ExactKthRoot(477481, 2) == 691 &&
              hecke::ExactKthRoot(18014406294111049ull, 2) == 134217757ull && hecke::ExactKthRoot(1ull << 40, 40) == 2 &&
              hecke::ExactKthRoot(625, 4) == 5,
          "(K) ExactKthRoot recovers the prime of ell^k");
    check(hecke::ExactKthRoot(625, 2) == 0 && hecke::ExactKthRoot(7, 2) == 0 && hecke::ExactKthRoot(24, 2) == 0 &&
              hecke::ExactKthRoot(36, 2) == 0 && hecke::ExactKthRoot(0, 2) == 0,
          "(K) ExactKthRoot rejects a non-prime root (625 = 25^2 is not Z/ell^2) and non-powers");
    std::cout << "  (K) coefficient_ring: 11 keys, grammar / reserved / invalid arms" << std::endl;
}

// ---- (B) the BlockMat / BlockBuilder ModCtx overload ---------------------------
void block_arm() {
    using BM = engine::BlockMat<uint32_t>;
    const ModCtx<uint32_t> comp = ModCtx<uint32_t>::prime_power(5u, 2);
    auto pool = std::make_shared<engine::BlockPool<uint32_t>>();
    const std::vector<uint32_t> dims = {3, 3};
    engine::BlockBuilder<uint32_t> b(pool, dims, dims, comp);
    b.set_scalar(0, 0, 5);                 // ell * I: a non-unit pivot
    b.set_scalar(1, 1, 1);
    b.set_scalar(0, 1, 2);
    const BM A = b.finalize();
    check(!A.ctx().is_field() && A.ctx().ell == 5 && A.ctx().k == 2 && A.p() == 25,
          "(B) BlockBuilder(ModCtx) -> finalize keeps the composite context");
    check(!BM::zeros(pool, dims, dims, comp).ctx().is_field(), "(B) BlockMat::zeros(ModCtx) is composite");
    check(!A.transposed().ctx().is_field(), "(B) transposed() keeps the ring");
    check(!engine::blk::gemm<uint32_t>(A, A).ctx().is_field(), "(B) blk::gemm's result carries A's ring");
    check(!engine::blk::add<uint32_t>(A, A).ctx().is_field(), "(B) blk::add's result carries A's ring");
    check(!engine::blk::scale<uint32_t>(A, 3u).ctx().is_field(), "(B) blk::scale's result carries A's ring");
    DenseBuf<uint32_t> buf(6, 2);
    check(!BM::from_flat(pool, std::move(buf), dims, comp).ctx().is_field(), "(B) from_flat(ModCtx) is composite");
    // The field overload is unchanged: a bare p still declares F_p.
    engine::BlockBuilder<uint32_t> bf(pool, dims, dims, 25u);
    const BM F = bf.finalize();
    check(F.ctx().is_field() && F.p() == 25, "(B) the Elt-p overload still declares a field (Sec. 13.1)");
    // Sec. 13.3's block guards, testable at last.
    check(throws_logic([&] { (void)engine::blk::rref<uint32_t>(A); }), "(B) blk::rref refuses a composite BlockMat");
    check(throws_logic([&] {
              engine::tape::Executor<uint32_t> ex(&engine::tape::consumer_pool());
              (void)engine::tape::rref<uint32_t>(ex, A);
          }),
          "(B) tape::rref refuses a composite BlockMat");
    check(throws_logic([&] { (void)engine::blk::gemm<uint32_t>(A, F); }), "(B) blk::gemm refuses mixing rings");
    std::cout << "  (B) Sec. 13.7: sizeof(BlockMat<u32>) = " << sizeof(BM) << " (unchanged: the member was always a "
              << "full ModCtx), sizeof(BlockBuilder<u32>) = " << sizeof(engine::BlockBuilder<uint32_t>) << std::endl;
}

// ---- polynomials over a ring context (low -> high) ------------------------------
template<class Elt>
std::vector<Elt> charpoly_mod(const ModCtx<Elt>& ctx, const std::vector<std::vector<Elt>>& A) {
    // Berkowitz, independent of the operator's copy.
    const uint32_t n = static_cast<uint32_t>(A.size());
    std::vector<Elt> pc{Elt(1)};
    for (uint32_t r = 1; r <= n; ++r) {
        const uint32_t m = r - 1;
        std::vector<Elt> t(r + 1, 0);
        t[0] = 1;
        t[1] = ctx.neg(A[m][m]);
        std::vector<Elt> v(m);
        for (uint32_t i = 0; i < m; ++i) v[i] = A[i][m];
        for (uint32_t k = 2; k <= r; ++k) {
            Elt acc = 0;
            for (uint32_t i = 0; i < m; ++i) acc = ctx.add(acc, ctx.mul(A[m][i], v[i]));
            t[k] = ctx.neg(acc);
            std::vector<Elt> nv(m, 0);
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < m; ++j) nv[i] = ctx.add(nv[i], ctx.mul(A[i][j], v[j]));
            v.swap(nv);
        }
        std::vector<Elt> np(r + 1, 0);
        for (uint32_t i = 0; i <= r; ++i)
            for (uint32_t j = 0; j < r && j <= i; ++j) np[i] = ctx.add(np[i], ctx.mul(t[i - j], pc[j]));
        pc.swap(np);
    }
    std::reverse(pc.begin(), pc.end());
    return pc;
}
// Does monic d divide f over the field F_ell?  (low -> high)
template<class Elt>
bool divides_mod(const ModCtx<Elt>& F, std::vector<Elt> f, const std::vector<Elt>& d) {
    if (d.empty() || d.back() != 1) return false;
    const size_t dd = d.size() - 1;
    while (f.size() > dd && !f.empty()) {
        const Elt c = f.back();
        const size_t shift = f.size() - 1 - dd;
        for (size_t i = 0; i <= dd; ++i) f[shift + i] = F.sub(f[shift + i], F.mul(c, d[i]));
        f.pop_back();
    }
    for (Elt x : f) if (x != 0) return false;
    return true;
}

// ---- one cell over Z/ell^k -------------------------------------------------------
template<class Elt>
struct RingCell {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    ModCtx<Elt> ctx;
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh;
};
// Built inside the caller's ScopedCoefficientRing(k).
template<class Elt>
std::unique_ptr<RingCell<Elt>> build_ring_cell(uint32_t n, uint64_t ell, uint32_t k) {
    auto c = std::make_unique<RingCell<Elt>>();
    c->model = std::make_unique<hecke::SL2ZTessellationTables>(c->K);
    c->ctx = hecke::CoefficientRingContext<Elt>("HeckeReductionValidation", ell, k);
    auto rels = hecke::GenerateSL2ZRelationsRing<Elt>(c->K, n, c->ctx);
    c->lssc = std::make_unique<LocalSystemSimplicialComplexT<Elt>>(c->K, std::move(rels), n + 1, c->ctx.p);
    c->coh = std::make_unique<EquivariantCohomologyOnKT<Elt>>(c->lssc.get());
    c->coh->compute_cohomology();
    return c;
}

// (P) the rho_n repair on one exact matrix: fast == oracle entrywise, and the
// reduction mod ell equals the F_ell route.
template<class Elt>
void rho_check(const ModCtx<Elt>& ctx, const hecke::Mat22& g, uint32_t n, std::map<std::string, uint64_t>& routes,
               std::set<std::array<Elt, 4>>& seen) {
    std::array<Elt, 4> r;
    if constexpr (std::is_same_v<Elt, uint32_t>) { const auto a = g.reduced_mod(ctx.p); r = {a[0], a[1], a[2], a[3]}; }
    else { const auto a = g.reduced_mod64(ctx.p); r = {a[0], a[1], a[2], a[3]}; }
    if (!seen.insert(r).second) return;
    RhoRingRoute route = RhoRingRoute::UnitA;
    const FlatMatrixT<Elt> fast = ComputeSymmetricPowerMatrixRing<Elt>(ctx, r[0], r[1], r[2], r[3], n, &route);
    const FlatMatrixT<Elt> orc = ComputeSymmetricPowerMatrixRingOracle<Elt>(ctx, r[0], r[1], r[2], r[3], n);
    ++routes[to_text(route)];
    bool same = fast.get_rows() == n + 1 && orc.get_rows() == n + 1;
    for (uint32_t i = 0; same && i <= n; ++i)
        for (uint32_t j = 0; j <= n; ++j)
            if (fast.get_entry(i, j) != orc.get_entry(i, j)) { same = false; break; }
    check(same, "(P) rho_n over Z/ell^k: fast route (" + std::string(to_text(route)) + ") == division-free oracle");
    const Elt ell = ctx.ell;
    const FlatMatrixT<Elt> g_fm(2, 2, ell, std::vector<Elt>{Elt(r[0] % ell), Elt(r[1] % ell), Elt(r[2] % ell), Elt(r[3] % ell)});
    const FlatMatrixT<Elt> fld = ComputeSymmetricPowerMatrixFM(g_fm, n);
    bool red = true;
    for (uint32_t i = 0; red && i <= n; ++i)
        for (uint32_t j = 0; j <= n; ++j)
            if (fast.get_entry(i, j) % ell != fld.get_entry(i, j)) { red = false; break; }
    check(red, "(P) rho_n over Z/ell^k reduces mod ell to ComputeSymmetricPowerMatrixFM");
}

struct CellSpec {
    uint32_t n; uint64_t ell; uint32_t k;
    std::vector<uint32_t> H0, H1;          // predicted invariant factors ([E] sec:ring (e))
    int certified;                          // 1 / 0
    int r0_minus;                           // predicted r_K(d^0) = rank C^0 - r0_minus (-1: not pre-stated)
    std::vector<uint32_t> torsion;          // ladic_type^1 torsion where certified
    uint32_t free_rank;
    uint32_t precision;                     // K - e_max where certified
    std::vector<uint64_t> primes;
};

template<class Elt>
void cell_arm_body(const CellSpec& cs);
template<class Elt>
void cell_arm(const CellSpec& cs) {
    try { cell_arm_body<Elt>(cs); }
    catch (const std::exception& e) {
        check(false, "(C) (" + std::to_string(cs.n) + "," + std::to_string(cs.ell) + ") Z/ell^" + std::to_string(cs.k) +
                         ": threw " + e.what());
    }
}
template<class Elt>
void cell_arm_body(const CellSpec& cs) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::string tag = "(" + std::to_string(cs.n) + "," + std::to_string(cs.ell) + ") Z/ell^" + std::to_string(cs.k);
    // The F_ell run at the same cell (default settings): dims and A^{(1)}(T_p).
    SimplicialComplex Kf = CreateSL2ZFundamentalDomainTriangulation();
    hecke::SL2ZTessellationTables mf(Kf);
    auto rf = GenerateSL2ZRelationsElt<Elt>(Kf, cs.n, static_cast<Elt>(cs.ell));
    LocalSystemSimplicialComplexT<Elt> lf(Kf, std::move(rf), cs.n + 1, static_cast<Elt>(cs.ell));
    EquivariantCohomologyOnKT<Elt> cf(&lf);
    cf.compute_cohomology();
    const std::vector<uint32_t> dims_f = cf.get_H_eq_dimensions();

    hecke::ScopedCoefficientRing ring(cs.k);
    std::unique_ptr<RingCell<Elt>> c;
    try { c = build_ring_cell<Elt>(cs.n, cs.ell, cs.k); }
    catch (const std::exception& e) { check(false, "(C) " + tag + ": pipeline threw: " + e.what()); return; }
    const EquivariantCohomologyOnKT<Elt>& coh = *c->coh;
    check(coh.is_composite() && coh.coefficient_ring_k() == cs.k && coh.ring_context().p == c->ctx.p,
          "(C) " + tag + ": the object is over Z/ell^k");
    const std::vector<uint32_t>& I0 = coh.H_invariant_factors(0);
    const std::vector<uint32_t>& I1 = coh.H_invariant_factors(1);
    const std::vector<uint32_t>& I2 = coh.H_invariant_factors(2);
    check(I0 == cs.H0, "(C) " + tag + ": invariant_factors_H^0 = " + vec_str(I0) + ", predicted " + vec_str(cs.H0));
    check(I1 == cs.H1, "(C) " + tag + ": invariant_factors_H^1 = " + vec_str(I1) + ", predicted " + vec_str(cs.H1));
    check(I2.empty(), "(C) " + tag + ": H^2 = 0");
    for (uint32_t q = 0; q < 3; ++q)
        check(coh.H_invariant_factors(q).size() == dims_f[q],
              "(C) " + tag + ": #cyclic summands of H^" + std::to_string(q) + " == dim H^q(F_ell) (UCT)");
    const std::vector<uint32_t> len = coh.H_lengths();
    uint32_t l1 = 0; for (uint32_t e : I1) l1 += e;
    check(len.size() == 3 && len[1] == l1, "(C) " + tag + ": lengths_H^1 = sum e_i");
    const uint32_t c0 = coh.C_eq_rank(0), c2 = coh.C_eq_rank(2), r0 = coh.r_K(0), r1 = coh.r_K(1);
    check(coh.ladic_applicable(), "(C) " + tag + ": on the applicability range");
    const bool cert = coh.ladic_type_certified();
    check(cert == (cs.certified == 1), "(C) " + tag + ": ladic_type_certified = " + (cert ? "true" : "false"));
    check(r1 == c2, "(C) " + tag + ": r_K(d^1) = rank C^2_eq (" + std::to_string(r1) + " vs " + std::to_string(c2) + ")");
    if (cs.r0_minus >= 0)
        check(r0 + static_cast<uint32_t>(cs.r0_minus) == c0,
              "(C) " + tag + ": r_K(d^0) = " + std::to_string(r0) + " = rank C^0_eq (" + std::to_string(c0) + ") - " +
                  std::to_string(cs.r0_minus));
    const auto st = coh.ladic_level_state();
    check(st.kind == "ladic_level" && st.budget_K == cs.k && st.reached_r0 == r0 && st.reached_r1 == r1 &&
              st.complete == cert,
          "(C) " + tag + ": the ladic_level state (budget {K}, reached {r_K(d^0), r_K(d^1)})");
    if (cert) {
        const auto t1 = coh.ladic_type(1), tz0 = coh.ladic_type(0), tz2 = coh.ladic_type(2);
        check(t1.free_rank == cs.free_rank && t1.torsion == cs.torsion,
              "(C) " + tag + ": ladic_type^1 = (" + std::to_string(t1.free_rank) + "; " + vec_str(t1.torsion) + ")");
        check(tz0.free_rank == 0 && tz0.torsion.empty() && tz2.free_rank == 0 && tz2.torsion.empty(),
              "(C) " + tag + ": ladic_type^0 = ladic_type^2 = (0;)");
    } else {
        check(throws_logic([&] { (void)coh.ladic_type(1); }), "(C) " + tag + ": ladic_type throws at an uncertified level");
    }
    // (slot-8 WP-3r, review M-1) the (c5) coordinates of the cyclic-summand
    // basis itself, read back through the flat-cochain entry point, are I.
    for (uint32_t q = 0; q < 3; ++q) {
        bool ident = false;
        try {
            const DenseBuf<Elt>& B = coh.H_module_basis(q);
            const auto X = coh.H_class_coordinates(q, engine::cview(B));
            const size_t tq = coh.H_invariant_factors(q).size();
            ident = B.cols == tq && X.size() == tq;
            for (size_t j = 0; ident && j < tq; ++j) {
                ident = X[j].size() == tq;
                for (size_t i = 0; ident && i < tq; ++i) ident = X[j][i] == (i == j ? Elt(1) : Elt(0));
            }
        } catch (const std::exception&) { ident = false; }
        check(ident, "(C) " + tag + ": H_class_coordinates(" + std::to_string(q) + ", H_module_basis(" + std::to_string(q) +
                         ")) == I");
    }
    // Field accessors refuse on a composite object; module accessors on a field one.
    check(throws_logic([&] { (void)coh.get_H_eq_dimensions(); }) && throws_logic([&] { (void)coh.C_eq_block(0); }) &&
              throws_logic([&] { (void)coh.get_H_eq_bases_fm(); }) && throws_logic([&] { (void)coh.C_eq_dimension(1); }),
          "(C) " + tag + ": every field accessor (dim, D5b bases) throws logic_error over Z/ell^k");
    check(throws_logic([&] { (void)cf.H_lengths(); }), "(C) the module accessors throw on an F_ell object");

    // (T) + (P) per prime.
    std::map<std::string, uint64_t> routes;
    std::set<std::array<Elt, 4>> seen;
    for (const auto& seed : SL2ZRelationSeeds_7VertexModel()) rho_check<Elt>(c->ctx, seed.g_r, cs.n, routes, seen);
    rho_check<Elt>(c->ctx, hecke::Mat22::I().neg(), cs.n, routes, seen);
    std::ostringstream cps;
    uint32_t d6_asserted = 0;
    for (uint64_t p : cs.primes) {
        const std::string tp = tag + " p=" + std::to_string(p) + (p == cs.ell ? " (U_ell)" : "");
        hecke::HeckeCosetTablesT<Elt> cosets(p, 1, static_cast<Elt>(cs.ell), cs.n);
        hecke::HeckeTauTableT<Elt> tau(c->K, *c->model, cosets, static_cast<Elt>(cs.ell), cs.n);
        for (uint32_t i = 1; i <= cosets.num_cosets(); ++i) rho_check<Elt>(c->ctx, cosets.s_inv(i), cs.n, routes, seen);
        for (uint32_t q = 0; q < 3; ++q)
            for (uint32_t s = 0; s < c->K.get_num_q_simplices(q); ++s)
                for (uint32_t i = 1; i <= cosets.num_cosets(); ++i)
                    for (const hecke::ChainTerm& ct : tau.tau(i, q, s)) {
                        rho_check<Elt>(c->ctx, cosets.g() * ct.gamma, cs.n, routes, seen);
                        rho_check<Elt>(c->ctx, cosets.delta(i) * ct.gamma, cs.n, routes, seen);
                    }
        hecke::HeckeOperatorOnKT<Elt> op(&coh, &cosets, &tau);
        try { op.compute(); }
        catch (const std::exception& e) { check(false, "(T) " + tp + ": compute threw: " + e.what()); continue; }
        const auto cb = op.certificates();
        check(cb.routes_bit_identical && cb.relation_checks > 0 && cb.cocycle_checks > 0 && cb.theta_eq_solves > 0 &&
                  cb.theta_eq_residual_max == 0,
              "(T) " + tp + ": O12(a) over Z/ell^k, runtime assertions (i)/(ii)/(iii), residual 0");
        const DenseBuf<Elt>& A1 = op.H_matrix_module(1);
        bool reduced = A1.rows == I1.size() && A1.cols == I1.size();
        for (uint32_t i = 0; reduced && i < A1.rows; ++i)
            for (uint32_t j = 0; j < A1.cols; ++j)
                if (I1[i] < cs.k && A1.at(i, j) >= c->ctx.ell_pow(I1[i])) { reduced = false; break; }
        check(reduced, "(T) " + tp + ": A^{(1)} is t x t with row i reduced into [0, ell^{e_i}) (c5)");
        check(throws_logic([&] { (void)op.get_H_matrices(); }) && throws_logic([&] { (void)op.Theta_block(1); }) &&
                  throws_logic([&] { (void)op.to_json(); }),
              "(T) " + tp + ": the field accessors throw over Z/ell^k");
        if (cert) {
            const auto fc = op.free_charpoly_H1();
            check(fc.precision == cs.precision && fc.rank == cs.free_rank && fc.coeffs.size() == cs.free_rank + 1 &&
                      fc.coeffs.back() == 1,
                  "(T) " + tp + ": charpoly_A(T_p)^1_free over Z/ell^" + std::to_string(fc.precision) + ", degree " +
                      std::to_string(fc.rank));
            // (slot-8 WP-3r, review m-7) the ell-adic VALUE at weight 12 (n = 10):
            // H^1 is E_12 + Delta twice, so charpoly = (x - (1 + p^11)) (x - tau(p))^2
            // exactly over Z/ell^precision -- the D6 theorem (a_E = 1 + p^11, a_Delta = tau(p)).
            if (cs.n == 10 && cs.torsion.empty() && (p == 2 || p == 3 || p == 5)) {
                const __int128 tau = (p == 2) ? -24 : (p == 3) ? 252 : 4830;
                __int128 aE = 1;
                for (int i = 0; i < 11; ++i) aE *= static_cast<__int128>(p);
                aE += 1;
                // (x - a)(x - b)^2 = x^3 - (a + 2b) x^2 + (2ab + b^2) x - a b^2
                const __int128 want_s[4] = {-aE * tau * tau, 2 * aE * tau + tau * tau, -(aE + 2 * tau), 1};
                const __int128 M = static_cast<__int128>(fc.modulus);
                bool d6 = fc.coeffs.size() == 4 && fc.rank == 3;
                for (size_t i = 0; d6 && i < 4; ++i) {
                    __int128 w = want_s[i] % M;
                    if (w < 0) w += M;
                    d6 = static_cast<__int128>(fc.coeffs[i]) == w;
                }
                check(d6, "(T) " + tp + ": D6 value -- the free charpoly is (x - (1 + p^11)) (x - tau(p))^2 mod ell^" +
                              std::to_string(fc.precision) + " exactly");
                ++d6_asserted;
            }
            // mod ell it divides the F_ell charpoly of A^{(1)}(T_p) (equal where H^1 is free).
            hecke::HeckeCosetTablesT<Elt> cf_cos(p, 1, static_cast<Elt>(cs.ell), cs.n);
            {
                hecke::ScopedCoefficientRing field(1);
                hecke::HeckeTauTableT<Elt> tf(Kf, mf, cf_cos, static_cast<Elt>(cs.ell), cs.n);
                hecke::HeckeOperatorOnKT<Elt> opf(&cf, &cf_cos, &tf);
                opf.compute();
                const auto& Af = opf.get_H_matrices()[1];
                std::vector<std::vector<Elt>> M(Af.rows, std::vector<Elt>(Af.cols));
                for (uint32_t i = 0; i < Af.rows; ++i)
                    for (uint32_t j = 0; j < Af.cols; ++j) M[i][j] = Af.data[i][j];
                const ModCtx<Elt> F(static_cast<Elt>(cs.ell));
                const std::vector<Elt> chf = charpoly_mod<Elt>(F, M);
                std::vector<Elt> fr(fc.coeffs.size());
                for (size_t i = 0; i < fr.size(); ++i) fr[i] = static_cast<Elt>(fc.coeffs[i] % F.p);
                check(divides_mod<Elt>(F, chf, fr), "(T) " + tp + ": the free charpoly mod ell divides the F_ell charpoly");
                if (cs.torsion.empty()) check(chf == fr, "(T) " + tp + ": and EQUALS it (H^1 free)");
            }
            cps << " p=" << p << ":[";
            for (size_t i = 0; i < fc.coeffs.size(); ++i) cps << (i ? "," : "") << fc.coeffs[i];
            cps << "]/" << fc.modulus;
        } else {
            check(throws_logic([&] { (void)op.free_charpoly_H1(); }),
                  "(T) " + tp + ": the free charpoly's accessor throws at an uncertified level");
        }
    }
    if (cs.n == 10 && cert && cs.torsion.empty())
        check(d6_asserted == cs.primes.size(), "(T) " + tag + ": the D6 value asserted at every prime (review m-7)");
    std::ostringstream rs;
    for (const auto& kv : routes) rs << " " << kv.first << "=" << kv.second;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  (C)(T)(P) " << tag << ": H^0 " << vec_str(I0) << " H^1 " << vec_str(I1) << " rank C^0/C^1/C^2 = "
              << c0 << "/" << coh.C_eq_rank(1) << "/" << c2 << " r_K(d^0)=" << r0 << " r_K(d^1)=" << r1
              << " certified=" << (cert ? 1 : 0) << " rho g's=" << seen.size() << " [" << rs.str() << " ]"
              << (cert ? " charpoly_free" + cps.str() : std::string())
              << (d6_asserted ? " D6 asserted at " + std::to_string(d6_asserted) + " prime(s)" : std::string()) << "  " << secs
              << " s" << std::endl;
}

// ---- the read-only REPORT arm (HECKE_REDUCTION_CELL=<n>,<ell>,<k>) ------------------
// Not a gate and not an emitter: prints one cell's Z/ell^k data (the invariant
// factors, the K-mode, and the free charpoly of T_2 where certified) -- the
// slot-8 plan's "reported" cell (150, 5) at k = 4 is run this way.
void report_cell(uint32_t n, uint64_t ell, uint32_t k) {
    const auto t0 = std::chrono::steady_clock::now();
    hecke::ScopedCoefficientRing ring(k);
    auto c = build_ring_cell<uint32_t>(n, ell, k);
    const auto& coh = *c->coh;
    std::cout << "  REPORT (" << n << "," << ell << ") Z/ell^" << k << ": H^0 " << vec_str(coh.H_invariant_factors(0))
              << " H^1 " << vec_str(coh.H_invariant_factors(1)) << " H^2 " << vec_str(coh.H_invariant_factors(2))
              << " rank C^0/C^2 = " << coh.C_eq_rank(0) << "/" << coh.C_eq_rank(2) << " r_K(d^0)=" << coh.r_K(0)
              << " r_K(d^1)=" << coh.r_K(1);
    if (coh.ladic_applicable()) {
        const bool cert = coh.ladic_type_certified();
        std::cout << " certified=" << (cert ? 1 : 0);
        if (cert) {
            const auto t = coh.ladic_type(1);
            std::cout << " ladic_type^1=(" << t.free_rank << ";" << vec_str(t.torsion) << ")";
            hecke::HeckeCosetTables cosets(2, 1, static_cast<uint32_t>(ell), n);
            hecke::HeckeTauTable tau(c->K, *c->model, cosets, static_cast<uint32_t>(ell), n);
            hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
            op.compute();
            const auto fc = op.free_charpoly_H1();
            std::cout << " charpoly_free_p2 over Z/ell^" << fc.precision << " [";
            for (size_t i = 0; i < fc.coeffs.size(); ++i) std::cout << (i ? "," : "") << fc.coeffs[i];
            std::cout << "]";
        }
    }
    std::cout << "  " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s"
              << std::endl;
}

// ---- (X) the refusal closure ----------------------------------------------------
int run_console(const std::string& code) {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return -1;
    buf[n] = 0;
    std::string cmd = std::string(buf) + " -c '" + code + "' >/dev/null 2>&1";
    const int st = std::system(cmd.c_str());
    if (st == -1 || !WIFEXITED(st)) return -1;
    return WEXITSTATUS(st);
}

void refusal_arm() {
    // Field objects built at the default, consumed under the ring.
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    hecke::SL2ZTessellationTables model(K);
    auto rels = GenerateSL2ZRelationsElt<uint32_t>(K, 10, 691u);
    LocalSystemSimplicialComplex lssc(K, rels, 11, 691u);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    hecke::HeckeCosetTables cosets(2, 1, 691u, 10);
    hecke::HeckeTauTable tau(K, model, cosets, 691u, 10);
    auto cell = std::make_unique<hecke::ContragredientCell>(10, 691u);

    // A composite object, for the "input over Z/ell^k" arms.
    std::unique_ptr<RingCell<uint32_t>> comp;
    { hecke::ScopedCoefficientRing ring(2); comp = build_ring_cell<uint32_t>(10, 691, 2); }

    const std::string R = "coefficient_ring=Z/ell^2 is not honoured in this build";
    uint32_t members = 0;
    auto member = [&](const std::string& name, bool ok) { ++members; check(ok, "(X) " + name + " refuses by name"); };
    {
        hecke::ScopedCoefficientRing ring(2);
        // the honouring objects' own refusals
        member("EquivariantCohomologyOnK (an F_ell local system)",
               throws_niitb([&] { EquivariantCohomologyOnK x(&lssc); }, "EquivariantCohomologyOnK: " + R));
        // CHANGED ROW (slot-10 WP-3, decision 3 (A); a refusal through slot 9): the
        // relative sub-complex constructor HONOURS Z/ell^k at ell >= 5 (the cone's route 3
        // over the ring).  Its remaining refusals (ell = 3; an F_ell local system) are
        // HeckeConeCrossRouteRingValidation's arm (X).
        {
            bool ok = false;
            try {
                EquivariantCohomologyOnK x(comp->lssc.get(), std::vector<std::vector<uint32_t>>{{0}});
                ok = x.is_composite() && x.is_relative();
            } catch (...) { ok = false; }
            ++members;
            check(ok, "(X) EquivariantCohomologyOnK (relative sub-complex) HONOURS Z/ell^2 at (10, 691) (changed row, slot-10 WP-3)");
        }
        member("HeckeOperatorOnK (F_ell cohomology, Z/ell^2 in force)",
               throws_niitb([&] { hecke::HeckeOperatorOnK x(&coh, &cosets, &tau); }, "HeckeOperatorOnK: " + R));
        member("HeckeOperatorOnK (contragredient)",
               throws_niitb([&] { hecke::HeckeOperatorOnK x(comp->coh.get(), &cosets, &tau, FibreRepresentation::ContragredientN); },
                            "HeckeOperatorOnK (contragredient): " + R));
        // (10, 3) under Z/ell^2.  CHANGED ROW (slot-9 WP-3, decision 4 (A); was
        // a refusal through slot 8): EquivariantCohomologyOnK HONOURS ell in
        // {2, 3} -- the Bredon object by the flat-coordinate route -- with the
        // O7 values of HeckeBredonTreeRingValidation (H^0 = Z/3, H^1 = (Z/9)^3).
        {
            const ModCtx<uint32_t> c9 = ModCtx<uint32_t>::prime_power(3u, 2);
            auto r9 = hecke::GenerateSL2ZRelationsRing<uint32_t>(K, 10, c9);
            LocalSystemSimplicialComplex l9(K, std::move(r9), 11, 9u);
            bool ok = false;
            try {
                EquivariantCohomologyOnK x(&l9);
                x.compute_cohomology();
                ok = x.is_flat_route() && x.H_lengths() == std::vector<uint32_t>{1, 6, 0} &&
                     x.H_invariant_factors(1) == std::vector<uint32_t>{2, 2, 2};
            } catch (...) { ok = false; }
            ++members;
            check(ok, "(X) EquivariantCohomologyOnK at (10, 3) HONOURS Z/ell^2 (changed row, slot-9 WP-3)");
        }
        // CHANGED ROW (slot-9 WP-3, M-4): through CoefficientRingContext, ell = 2,
        // under EquivariantCohomologyOnK's policy (SmallEllPolicy::Bredon) -- honours.
        {
            bool ok = false;
            try {
                const ModCtx<uint32_t> c = hecke::CoefficientRingContext<uint32_t>("probe", 2, 2, hecke::SmallEllPolicy::Bredon);
                ok = !c.is_field() && c.ell == 2u && c.k == 2 && c.p == 4u;
            } catch (...) { ok = false; }
            ++members;
            check(ok, "(X) CoefficientRingContext at ell = 2 (Bredon policy) HONOURS (changed row, slot-9 WP-3)");
        }
        // New row (slot-9 WP-3): every other caller's (default) policy still refuses at 2 and 3.
        member("CoefficientRingContext at ell = 2 (default policy)",
               throws_niitb([&] { (void)hecke::CoefficientRingContext<uint32_t>("probe", 2, 2); }, "at ell = 2"));
        member("CoefficientRingContext at ell = 3 (default policy)",
               throws_niitb([&] { (void)hecke::CoefficientRingContext<uint64_t>("probe", 3, 4); }, "at ell = 3"));
        member("CoefficientRingContext u32 at 134217757^2",
               throws_niitb([&] { (void)hecke::CoefficientRingContext<uint32_t>("probe", 134217757ull, 2); }, "uint32_t"));
        member("CoefficientRingContext at ell^k >= 2^64",
               throws_niitb([&] { (void)hecke::CoefficientRingContext<uint64_t>("probe", 134217757ull, 3); }, "2^64"));
        // the refusing members (constructor-entry guards; the guard runs before any input check)
        member("HeckeModuleDecomposition",
               throws_niitb([&] { hecke::HeckeModuleDecomposition x({}, 691u); }, "HeckeModuleDecomposition: " + R));
        member("HeckeOperatorFamily",
               throws_niitb([&] { hecke::HeckeOperatorFamily x(&coh, K, model, 691u, 10); }, "HeckeOperatorFamily: " + R));
        member("ContragredientCell", throws_niitb([&] { hecke::ContragredientCell x(10, 691u); }, "ContragredientCell: " + R));
        member("PoincareDualityPairing",
               throws_niitb([&] { hecke::PoincareDualityPairing x(*cell); }, "PoincareDualityPairing: " + R));
        member("HeckeCrossChecks",
               throws_niitb([&] { hecke::HeckeCrossChecks x(&coh, &cosets, &tau); }, "HeckeCrossChecks: " + R));
        member("BarCocycleH1", throws_niitb([&] { hecke::BarCocycleH1 x(691u, 10); }, "BarCocycleH1: " + R));
        member("HeckeBoundaryOperator",
               throws_niitb([&] { hecke::HeckeBoundaryOperator x(nullptr, nullptr, nullptr); }, "HeckeBoundaryOperator: " + R));
        member("SL2ZBoundaryPair", throws_niitb([&] { hecke::SL2ZBoundaryPair x(&coh, K); }, "SL2ZBoundaryPair: " + R));
        member("HeckeInvolution",
               throws_niitb([&] { hecke::HeckeInvolution x(nullptr, hecke::IotaSimplicialMap(), "probe"); }, "HeckeInvolution: " + R));
        member("EquivariantCochainComplexMorphismOnK",
               throws_niitb([&] { EquivariantCochainComplexMorphismOnK x(nullptr, nullptr, nullptr, FlatMatrix(1, 1, 691u)); },
                            "EquivariantCochainComplexMorphismOnK: " + R));
        member("HeckeExt/HeckeBoundary",
               throws_niitb([&] { hecke_ext::HeckeBoundary x(nullptr, K, model, hecke_ext::HeckeBoundaryOptions()); },
                            "HeckeBoundary: " + R));
        member("HeckeConeStrictPairRoute",
               throws_niitb([&] { hecke::HeckeConeStrictPairRoute x(nullptr); }, "HeckeConeStrictPairRoute: " + R));
        member("HeckeConeRelativeRoute",
               throws_niitb([&] { hecke::HeckeConeRelativeRoute x(nullptr); }, "HeckeConeRelativeRoute: " + R));
        // the seven surfaces
        member("surface HeckeExtDriver", throws_niitb([&] { hecke_ext::HeckeExtDriver x(10, 691u); }, "HeckeExtDriver: " + R));
        member("surface EquivariantMappingCone",
               throws_niitb([&] { EquivariantMappingCone x(nullptr, nullptr, nullptr, EquivariantMappingConeOptions()); },
                            "EquivariantMappingCone: " + R));
        member("surface HeckeConeAction",
               throws_niitb([&] { hecke::HeckeConeAction x(nullptr, nullptr, nullptr, nullptr); }, "HeckeConeAction: " + R));
        member("surface HeckeReport",
               throws_niitb([&] { (void)hecke::BuildHeckeReportJson(hecke::HeckeReportInput()); }, "HeckeReport (hecke_report/v2): " + R));
        {
            bench::ModeArgs a;
            a.mode = "cohom"; a.n_sym = 10; a.ell = 691;
            member("surface --bench", throws_niitb([&] { (void)bench::RunMode(a); }, "--bench cohom: " + R));
        }
        {
            // (slot-9 WP-8, C5c) the job loop honours the ring now; the surface
            // that keeps refusing at spec validation is the EXT PASS ([E] (b4)).
            std::istringstream in("coefficient_ring = Z/ell^2\next = true\nn_sym = 10\nell = 691\np = 2\nr = 1\n");
            member("surface campaign (spec validation)",
                   throws_niitb([&] { (void)campaign::ParseCampaignSpec(in, "wp3-arm"); }, "campaign spec wp3-arm: " + R));
        }
        {
            char a0[] = "--verify", a1[] = "/nonexistent/wp3";
            char* av[] = {a0, a1};
            member("surface --atlas", hecke_ext::RunAtlasCli(2, av, {}) == 5);
        }
    }
    // the campaign's ext pass refuses the ring even when the process is at the
    // default (slot-9 WP-8: the job loop honours it, CampaignRingValidation)
    {
        std::istringstream in("coefficient_ring = Z/ell^3\next = true\nn_sym = 10\nell = 691\np = 2\nr = 1\n");
        check(throws_niitb([&] { (void)campaign::ParseCampaignSpec(in, "wp3-arm"); }, "Z/ell^3 is not honoured"),
              "(X) a campaign spec naming Z/ell^3 with ext = true is refused at validation");
        // (slot-24 D6) Z is un-reserved: the campaign honours it with descent cells, and refuses `ext = true` under it.
        std::istringstream z("coefficient_ring = Z\next = true\nn_sym = 10\np = 2\n");
        check(throws_niitb([&] { (void)campaign::ParseCampaignSpec(z, "wp3-arm"); }, "coefficient_ring=Z "),
              "(X) a campaign spec naming Z with ext = true is refused at validation");
    }
    // input objects over Z/ell^k, with F_ell in force
    member("HeckeOperatorOnK (Z/ell^2 cohomology, F_ell in force)",
           throws_niitb([&] { hecke::HeckeOperatorOnK x(comp->coh.get(), &cosets, &tau); }, "HeckeOperatorOnK: " + R));
    member("HeckeCrossChecks (Z/ell^2 input)",
           throws_niitb([&] { hecke::HeckeCrossChecks x(comp->coh.get(), &cosets, &tau); },
                        "HeckeCrossChecks: " + R + " (an input object"));
    member("HeckeOperatorFamily (Z/ell^2 input)",
           throws_niitb([&] { hecke::HeckeOperatorFamily x(comp->coh.get(), K, model, 691u, 10); }, "(an input object"));
    member("SL2ZBoundaryPair (Z/ell^2 input)",
           throws_niitb([&] { hecke::SL2ZBoundaryPair x(comp->coh.get(), K); }, "(an input object"));
    member("HeckeInvolution (Z/ell^2 input)",
           throws_niitb([&] { hecke::HeckeInvolution x(comp->coh.get(), hecke::IotaSimplicialMap(), "probe"); }, "(an input object"));
    member("HeckeExt/HeckeBoundary (Z/ell^2 input)",
           throws_niitb([&] { hecke_ext::HeckeBoundary x(comp->coh.get(), K, model, hecke_ext::HeckeBoundaryOptions()); },
                        "(an input object"));
    // (slot-8 WP-3r, review m-6) a local system over Z/ell^2 (from
    // GenerateSL2ZRelationsRing) consumed with F_ell in force: refused by a
    // named logic_error, never the field route on ModCtx(ell^2).
    {
        std::string msg;
        bool ok = false;
        try { EquivariantCohomologyOnK x(comp->lssc.get()); }
        catch (const hecke::NotImplementedInThisBuild&) { ok = false; }
        catch (const std::logic_error& e) { msg = e.what(); ok = true; }
        catch (...) { ok = false; }
        member("EquivariantCohomologyOnK (a Z/ell^2 local system, F_ell in force; review m-6)",
               ok && msg.find("EquivariantCohomologyOnK: the local system's modulus 477481 = 691^2 is a prime power") !=
                         std::string::npos);
    }
    // (slot-9 WP-3; decision 4 (A)) ell in {2, 3}: EquivariantCohomologyOnK
    // honours Z/ell^k (the Bredon object, lengths and invariant factors; the
    // oracle arm is HeckeBredonTreeRingValidation), and EVERYTHING ELSE refuses
    // there by name: its stored-basis accessors and the K-mode ([C]
    // prop:certifiede(5)), HeckeOperatorOnK, HeckeReduction.
    {
        auto small = [&](uint32_t n, uint32_t ell, uint32_t k) {
            auto c = std::make_unique<RingCell<uint32_t>>();
            c->model = std::make_unique<hecke::SL2ZTessellationTables>(c->K);
            c->ctx = ModCtx<uint32_t>::prime_power(ell, k);
            auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(c->K, n, c->ctx);
            c->lssc = std::make_unique<LocalSystemSimplicialComplex>(c->K, std::move(rels), n + 1, c->ctx.p);
            hecke::ScopedCoefficientRing ring(k);
            c->coh = std::make_unique<EquivariantCohomologyOnK>(c->lssc.get());
            c->coh->compute_cohomology();
            return c;
        };
        const auto c2 = small(4, 2, 2), c2h = small(4, 2, 3), c3 = small(10, 3, 2);
        auto rels_f2 = GenerateSL2ZRelationsElt<uint32_t>(K, 4, 2u);
        LocalSystemSimplicialComplex lssc_f2(K, std::move(rels_f2), 5, 2u);
        EquivariantCohomologyOnK f2(&lssc_f2);
        f2.compute_cohomology();
        for (const auto* c : {c2.get(), c3.get()}) {
            const EquivariantCohomologyOnK& h = *c->coh;
            const std::string e = std::to_string(c->ctx.ell);
            const std::string Rk = "coefficient_ring=Z/ell^2 is not honoured in this build at ell = " + e;
            const std::string A = "EquivariantCohomologyOnK::";
            const DenseBuf<uint32_t> Y(h.flat_module_degree(1).C.rows, 0);
            member("ell = " + e + ": " + A + "module_degree", throws_niitb([&] { (void)h.module_degree(1); }, A + "module_degree: " + Rk));
            member("ell = " + e + ": " + A + "C_eq_rank", throws_niitb([&] { (void)h.C_eq_rank(1); }, A + "C_eq_rank: " + Rk));
            member("ell = " + e + ": " + A + "H_module_basis", throws_niitb([&] { (void)h.H_module_basis(1); }, A + "H_module_basis: " + Rk));
            member("ell = " + e + ": " + A + "H_class_coordinates",
                   throws_niitb([&] { (void)h.H_class_coordinates(1, engine::cview(Y)); }, A + "H_class_coordinates: " + Rk));
            member("ell = " + e + ": " + A + "r_K (K-mode)", throws_niitb([&] { (void)h.r_K(0); }, A + "r_K: " + Rk));
            member("ell = " + e + ": " + A + "ladic_applicable (K-mode)",
                   throws_niitb([&] { (void)h.ladic_applicable(); }, A + "ladic_applicable: " + Rk));
            member("ell = " + e + ": " + A + "ladic_type_certified (K-mode)",
                   throws_niitb([&] { (void)h.ladic_type_certified(); }, A + "ladic_type_certified: " + Rk));
            member("ell = " + e + ": " + A + "ladic_type (K-mode)", throws_niitb([&] { (void)h.ladic_type(1); }, A + "ladic_type: " + Rk));
            member("ell = " + e + ": " + A + "ladic_e_max (K-mode)", throws_niitb([&] { (void)h.ladic_e_max(); }, A + "ladic_e_max: " + Rk));
            member("ell = " + e + ": " + A + "ladic_level_state (K-mode)",
                   throws_niitb([&] { (void)h.ladic_level_state(); }, A + "ladic_level_state: " + Rk));
            {
                hecke::ScopedCoefficientRing ring(2);
                member("ell = " + e + ": HeckeOperatorOnK",
                       throws_niitb([&] { hecke::HeckeOperatorOnK x(c->coh.get(), &cosets, &tau); }, "HeckeOperatorOnK: " + Rk));
            }
            // the honouring accessors do honour
            check(h.is_flat_route() && h.num_degrees() == 3 && h.H_lengths().size() == 3,
                  "(X) ell = " + e + ": num_degrees / H_lengths / H_invariant_factors honour");
        }
        const std::string R2 = "is not honoured in this build at ell = 2";
        member("ell = 2: HeckeReduction ReduceHToField",
               throws_niitb([&] { (void)hecke::ReduceHToField<uint32_t>(*c2->coh, f2, 1); }, R2));
        member("ell = 2: HeckeReduction ReduceHBetweenLevels",
               throws_niitb([&] { (void)hecke::ReduceHBetweenLevels<uint32_t>(*c2h->coh, *c2->coh, 1); }, R2));
        member("ell = 2: HeckeReduction BocksteinH", throws_niitb([&] { (void)hecke::BocksteinH<uint32_t>(*c2->coh, f2, 0); }, R2));
        member("ell = 2: HeckeReduction BocksteinCounts",
               throws_niitb([&] { (void)hecke::BocksteinCounts<uint32_t>(*c2->coh, f2); }, R2));
    }
    // the console's pipeline verbs at ell in {2, 3}: still exit 5 (C5a lifts nothing there)
    {
        const int e2 = run_console("set coefficient_ring = \"Z/ell^2\"; H = cohomology(symmetric_power(sl2z_model(), 4, 2)); print(1);");
        const int e3 = run_console("set coefficient_ring = \"Z/ell^3\"; H = cohomology(symmetric_power(sl2z_model(), 10, 3)); print(1);");
        if (e2 != -1) {
            member("surface console pipeline verbs at ell = 2 (exit 5)", e2 == 5);
            member("surface console pipeline verbs at ell = 3 (exit 5)", e3 == 5);
        }
    }
    // the console's pipeline verbs: exit 5 (C7), named.  (slot-9 WP-12, C5a)
    // Since [E] sec:ring (h)'s lift `cohomology` HONOURS Z/ell^2 at ell >= 5,
    // and `.dims` is a FIELD member (the object's logic_error, exit 3); the
    // surface that still refuses is a verb off (h)'s list, `family`.
    const int ec = run_console("set coefficient_ring = \"Z/ell^2\"; H = cohomology(symmetric_power(sl2z_model(), 10, 691)); F = family(H); print(1);");
    const int eh = run_console("H = cohomology(symmetric_power(sl2z_model(), 10, 691)); set coefficient_ring = \"Z/ell^2\"; T = hecke(H, 2); print(1);");
    // (slot-24 D6) `Z` is un-reserved; the reserved general `Z/<m>` keeps the set()-refusal pin.
    const int ez = run_console("set coefficient_ring = \"Z/25\";");
    const int ei = run_console("set coefficient_ring = \"Z/ell^1\";");
    const int e0 = run_console("set coefficient_ring = \"Z/ell^2\"; set coefficient_ring = \"F_ell\"; print(1);");
    if (ec == -1) {
        std::cout << "  (X) console arms SKIPPED: /proc/self/exe unavailable" << std::endl;
    } else {
        member("surface console pipeline verbs (exit 5)", ec == 5);
        check(eh == 5, "(X) console: `hecke` after `set coefficient_ring` exits 5 at dispatch");
        // The reserved value is refused by name inside MathSettings::set();
        // since slot-8 WP-10 (C4) the binding's `set` path maps the
        // NotImplementedInThisBuild to exit 5 ([E] sec:ring (a)). WP-3 had to
        // accept 3 or 5 here before that fix was merged.
        check(ez == 5, "(X) console: `set coefficient_ring = \"Z/25\"` is refused, exit 5 (got " + std::to_string(ez) + ")");
        check(ei != 0 && ei != 5, "(X) console: `Z/ell^1` is an invalid value, not a build-capability refusal (exit " +
                                      std::to_string(ei) + ")");
        check(e0 == 0, "(X) console: setting and resetting the key computes nothing and exits 0");
    }
    std::cout << "  (X) refusal closure: " << members << " refusing members / surfaces / honouring-object refusals "
              << "checked; console exits family (ring cohomology honoured) " << ec << ", hecke " << eh << ", set Z " << ez << ", set Z/ell^1 "
              << ei << ", round trip " << e0 << std::endl;
}

} // namespace

void HeckeReductionPipelineArms(uint64_t& checks, uint32_t& failures) {
    g_checks = &checks;
    g_fail = &failures;
    const uint32_t f0 = failures;
    key_arm();
    block_arm();
    // (C)(T)(P) the pre-stated cells ([E] sec:ring (e); slot-8 plan WP-3 scope 7).
    const std::vector<uint64_t> P691 = {2, 3, 5}, P5 = {2, 3, 5};
    cell_arm<uint32_t>({10, 691, 2, {}, {2, 2, 2}, 1, 0, {}, 3, 2, P691});
    cell_arm<uint32_t>({16, 5, 2, {}, {2, 2, 2}, 1, 0, {}, 3, 2, P5});
    cell_arm<uint32_t>({6, 5, 2, {1}, {2, 1}, 1, 0, {1}, 1, 1, P5});
    cell_arm<uint32_t>({30, 5, 2, {2}, {2, 2, 2, 2, 2, 2}, 0, 1, {}, 0, 0, P5});
    cell_arm<uint32_t>({30, 5, 3, {2}, {3, 3, 3, 3, 3, 2}, 1, 0, {2}, 5, 1, P5});
    cell_arm<uint32_t>({60, 5, 2, {2, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1}, 0, 1, {}, 0, 0, P5});
    cell_arm<uint32_t>({60, 5, 3, {2, 1}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1}, 1, 0, {2, 1}, 9, 1, P5});
    // (U) the u64 arm: correctness only (F-E).
    cell_arm<uint64_t>({10, 134217757ull, 2, {}, {2, 2, 2}, 1, 0, {}, 3, 2, {2}});
    // (U) the U64_Wide half (slot-8 WP-3r, review M-2): ell^2 = 18446744030759878681 >= 2^63,
    // where the old signed cast of the modulus in HeckeOperatorOnK's coeff_ring was negative.
    cell_arm<uint64_t>({10, 4294967291ull, 2, {}, {2, 2, 2}, 1, 0, {}, 3, 2, {2}});
    refusal_arm();
    if (const char* cellspec = std::getenv("HECKE_REDUCTION_CELL")) {
        unsigned n = 0, k = 0;
        unsigned long long ell = 0;
        if (std::sscanf(cellspec, "%u,%llu,%u", &n, &ell, &k) == 3 && k >= 2) {
            try { report_cell(n, ell, k); }
            catch (const std::exception& e) { std::cout << "  REPORT threw: " << e.what() << std::endl; }
        }
    }
    std::cout << "  E8 item 3 arms: " << (failures - f0) << " failing" << std::endl;
}
