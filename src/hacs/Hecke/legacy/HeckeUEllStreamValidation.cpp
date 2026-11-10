// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/legacy/HeckeUEllStreamValidation.cpp (S24 CL-R1: moved here with git mv, unchanged but for the include
// paths).  The levers it compares are RETIRED (owner ruling 2 of 2026-10-30): the whole name is registered through
// OC_LEGACY_VALIDATION (Registry/Slot14/WP10.cpp), so an OC_ENABLE_LEGACY=OFF build prints
// `[SKIP] HeckeUEllStreamValidation: OC_ENABLE_LEGACY=OFF`; this file is compiled only when ON (cmake/legacy.cmake).
//
// (hecke-extensions slot 14, WP-10) HeckeUEllStreamValidation -- the coset
// streaming lever (HeckeOperatorOptions::ScopedCosetStreaming) and the serial
// phase on the pool (ScopedParallelSetup), HeckeOperatorOnK.h.
//
// Neither lever can change a value: a streamed coset table evaluates rho_of of
// the same exact matrix the default table stores, on demand; the parallel
// setup fills pre-assigned slots with the same calls the serial loops make.
// OWNER DECISION 2026-09-28 (Q8): both, and the slot-13 U_ell lever, are the
// DEFAULTS; the REFERENCE of every arm below is therefore the OPT-OUT path
// (ScopedUEllLowRank(false), ScopedCosetStreaming(false),
// ScopedParallelSetup(false): the pre-WP-10 route, the rollback arm), and
// every arm compares the default (no scope at all) and each explicit lever
// combination against it.  The fan-out threshold is lowered to 0 here
// (ScopedParallelSetupMinWork) so that the small cells run the parallel code.
// This validation holds the levers to that, entrywise, alone and composed:
//   (A) the coset tables: every rho_n block of every table (rho_n(s_i^{-1}),
//       rho_n(t_i), 4 x rho_n(h_i(gamma)), 4 x the Gamma^0(N) cocycles) of a
//       streamed and of a parallel-built table equal to the default table's
//       block, with the same scalar flag; the DoD batteries self_check() and
//       self_check_upper() pass on a streamed table; a streamed table's
//       reference accessors refuse by ENGINE_CHECK; uint32_t and uint64_t.
//   (B) the tau table built under ScopedParallelSetup: the same xi terms
//       (coefficient, simplex, pool slot) and the same rho pool slot by slot,
//       dense and factored (U, V, t under the U_ell lever); certification
//       (f) passes on the parallel path (it runs inside the constructor);
//       and (slot-14 WP-16, audit S-4) its FAILING direction: with one pool
//       slot corrupted, certify() returns 60 serially and in parallel.
//   (C) F_ell, U_ell at ell in {5, 7, 11, 13, 691}, small n (plus T_{ell^2}
//       at 5, T_2 at 691 and the legacy arm): A^{(q)}, Theta_flat^{(q)},
//       Theta_eq^{(q)} and the certificate counters of every lever
//       combination equal to the default route's.
//   (D) Z/ell^2 at ell in {5, 7, 11, 13, 691} and Z/5^3 (the module route):
//       H_matrix_module(q), Theta_module(q) equal, alone and with the lever.
//   (E) ENGINE_THREADS = 1 vs 16 in-process (ScopedConsumerPool): the
//       streamed + parallel (+ lever) run equal to the default run on both.
//   (F) the other consumers of the coset tables on a streamed table: the
//       alpha / beta / omega cross-checks (HeckeCrossChecks, which builds the
//       induced local systems from rho_cocycle and reads rho_t, rho_s_inv)
//       pass at (10, 691) for T_2.
//   (G) printed, not gated: U_691 at n = 128, the peak RSS above the
//       cohomology under the lever with and without streaming.
//   (H) (slot-14 G0) the rank-one closed form (ScopedRankOneClosedForm,
//       default on): lowrank::rank_one_rho_factors against the dense route
//       (ComputeSymmetricPowerMatrixFM + factor_low_rank), (t, U, V) entrywise
//       and U V == rho_n entrywise, over EVERY 2x2 key mod ell at ell in
//       {2, 3, 5, 7} (2 and 3 since the G0 repairs, audit S2) and structured + pseudo-random keys at ell in {11, 13, 691,
//       2411}, n from 0 to 2420 (n < ell, ell - 1, ell, ell + 1, 2 ell, ell^2,
//       ...), uint32_t and uint64_t, with binomial_row_mod against rho_n's
//       own binomials; the negative direction: every NON-singular key refused
//       by both routes (the rank claim), a perturbed factor caught, a
//       composite ring refused; the tau pool (t, U, V) slot by slot with and
//       without the closed form, (f) passing; and U_ell / T_{ell^2} operators
//       (A, Theta_flat, Theta_eq, counters) equal with the closed form, its
//       opt-out and all levers off, at ell in {2, 3, 5, 7, 13, 691, 2411}.
//   (I) (slot-14 G0, audit M-2) the one host fan-out budget
//       (for_each_index): every index runs exactly once, the LOWEST failing
//       index's exception is rethrown, nested fan-outs inside a structural
//       fan-out share the helper set (observed concurrency <= the callers +
//       pool - 1 helpers), at 1, 4 and 16 workers; (G0 repairs, audit S1)
//       the budget per consumer pool: two pools of 4 in two threads, each
//       pool's concurrency <= 4 (gated), the overall peak reported.
#include "../HeckeUEllStreamValidation.h"

#if !OC_ENABLE_LEGACY
#error "HeckeUEllStreamValidation.cpp is an OC_ENABLE_LEGACY=ON source (cmake/legacy.cmake)"
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "../CoefficientRing.h"
#include "../HeckeCosetTables.h"
#include "../HeckeCrossChecks.h"
#include "../HeckeLowRank.h"   // (slot-14 G0) the rank-one closed form
#include "../HeckeOperatorFamily.h"   // tau_construction_mutex()
#include "../HeckeOperatorOnK.h"
#include "../HeckeSettings.h"
#include "../HeckeTauTable.h"
#include "../SL2ZTessellationTables.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../../ModArithmetic/FlatMatrix.h"   // ComputeSymmetricPowerMatrixFM
#include "../../Util/Rss.h"
#include "../../../ModArithmetic/DenseElim.h"   // engine::elim::ScopedLegacy
#include "../../../Engine/EngineRuntime.h"      // ScopedConsumerPool
#include "../../../Engine/ThreadPool.h"

namespace {

using hecke::HeckeOperatorOptions::ScopedCosetStreaming;
using hecke::HeckeOperatorOptions::ScopedParallelSetup;
using hecke::HeckeOperatorOptions::ScopedUEllLowRank;

struct Checker {
    uint64_t checks = 0, failures = 0;
    void operator()(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

// The process defaults, read once at entry (the env opt-outs HECKE_COSET_STREAM=0,
// HECKE_U_ELL_LOW_RANK=0 turn them off; the DEFAULT arm then expects that).
bool g_default_stream = true;
bool g_default_lever = true;   // the U_ell lever's mode is not Off

struct Levers {
    bool stream = false, par = false, lever = false;
    bool dflt = false;   // no scope at all: the process defaults (all three on, Auto)
    std::string name() const {
        if (dflt) return "DEFAULT";
        std::string s;
        if (stream) s += "stream";
        if (par) s += s.empty() ? "par" : "+par";
        if (lever) s += s.empty() ? "lever" : "+lever";
        return s.empty() ? "opt-out" : s;
    }
};

// Every lever combination (the explicit U_ell lever only where ell | N), led
// by the process DEFAULT; the all-off opt-out is the reference itself.
std::vector<Levers> Combos(bool with_lever) {
    std::vector<Levers> out;
    Levers d; d.dflt = true;
    out.push_back(d);
    for (int lv = 0; lv <= (with_lever ? 1 : 0); ++lv)
        for (int st = 0; st <= 1; ++st)
            for (int pa = 0; pa <= 1; ++pa) {
                if (lv == 0 && st == 0 && pa == 0) continue;   // the reference itself
                out.push_back(Levers{ st == 1, pa == 1, lv == 1 });
            }
    return out;
}

template<class Elt>
bool SameBlock(const FlatMatrixT<Elt>& a, const FlatMatrixT<Elt>& b) {
    Elt sa = 0, sb = 0;
    const bool ka = a.is_currently_scalar(sa), kb = b.is_currently_scalar(sb);
    return ka == kb && (!ka || sa == sb) && a.get_p() == b.get_p() && a == b;
}

// The three scopes of a Levers value (none at all for the DEFAULT arm).
struct LeverScope {
    std::unique_ptr<ScopedUEllLowRank> g1;
    std::unique_ptr<ScopedCosetStreaming> g2;
    std::unique_ptr<ScopedParallelSetup> g3;
    explicit LeverScope(const Levers& L) {
        if (L.dflt) return;
        g1.reset(new ScopedUEllLowRank(L.lever));
        g2.reset(new ScopedCosetStreaming(L.stream));
        g3.reset(new ScopedParallelSetup(L.par));
    }
};

struct Model {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    Model() {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*K));
    }
};

struct FieldCell {
    Model M;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    FieldCell(uint32_t n, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*M.K, n, ell)) rels.push_back(datum.rel);
        lssc.reset(new LocalSystemSimplicialComplex(*M.K, std::move(rels), n + 1, ell));
        coh.reset(new EquivariantCohomologyOnK(lssc.get()));
        coh->compute_cohomology();
    }
};

// One operator T_{p^r} on a computed F_ell cell under the given levers; all
// three are in scope for the tables' construction AND compute().
struct FieldOp {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeOperatorOnK> op;
    FieldOp(const FieldCell& c, uint64_t p, uint32_t r, uint32_t n, uint32_t ell, const Levers& L, bool legacy = false) {
        const LeverScope scope(L);
        std::unique_ptr<engine::elim::ScopedLegacy> legacy_elim;
        if (legacy) legacy_elim.reset(new engine::elim::ScopedLegacy());
        hecke::HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath legacy_path(
            legacy || hecke::HeckeOperatorOptions::legacy_hecke_operator_path_forced());
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTables(p, r, ell, n));
            tau.reset(new hecke::HeckeTauTable(*c.M.K, *c.M.model, *cosets, ell, n));
        }
        op.reset(new hecke::HeckeOperatorOnK(c.coh.get(), cosets.get(), tau.get()));
        op->compute();
    }
};

void CompareOps(Checker& g, const std::string& tag, const hecke::HeckeOperatorOnK& a, const hecke::HeckeOperatorOnK& b) {
    const auto& A0 = a.get_H_matrices();
    const auto& A1 = b.get_H_matrices();
    const auto& T0 = a.get_Theta_flat_matrices();
    const auto& T1 = b.get_Theta_flat_matrices();
    const auto& E0 = a.get_Theta_eq_matrices();
    const auto& E1 = b.get_Theta_eq_matrices();
    g(A0.size() == A1.size() && T0.size() == T1.size() && E0.size() == E1.size(), tag + " degree count");
    for (size_t q = 0; q < A0.size() && q < A1.size() && q < T1.size() && q < E1.size(); ++q) {
        g(A0[q].data == A1[q].data, tag + " A^(" + std::to_string(q) + ") entrywise");
        g(T0[q] == T1[q], tag + " Theta_flat^(" + std::to_string(q) + ") entrywise");
        g(E0[q] == E1[q], tag + " Theta_eq^(" + std::to_string(q) + ") entrywise");
    }
    const auto c0 = a.certificates(), c1 = b.certificates();
    g(c0.routes_bit_identical && c1.routes_bit_identical, tag + " O12(a) on both runs");
    g(c0.relation_checks == c1.relation_checks && c0.cocycle_checks == c1.cocycle_checks &&
          c0.theta_eq_solves == c1.theta_eq_solves && c1.theta_eq_residual_max == 0,
      tag + " certificate counters equal");
}

// ---- (A) the coset tables --------------------------------------------------
template<class Elt>
void CompareTables(Checker& g, const std::string& tag, const hecke::HeckeCosetTablesT<Elt>& ref,
                   const hecke::HeckeCosetTablesT<Elt>& t) {
    bool ok = ref.num_cosets() == t.num_cosets() && !ref.streamed();
    for (uint32_t i = 1; ok && i <= ref.num_cosets(); ++i) {
        ok = ok && SameBlock(ref.rho_s_inv(i), t.rho_s_inv_block(i).get());
        ok = ok && SameBlock(ref.rho_t(i), t.rho_t_block(i).get());
        for (uint32_t gen = 0; ok && gen < 4; ++gen) {
            ok = ok && SameBlock(ref.rho_cocycle(gen, i), t.rho_cocycle_block(gen, i).get());
            ok = ok && SameBlock(ref.rho_cocycle_upper(gen, i), t.rho_cocycle_upper_block(gen, i).get());
            ok = ok && ref.cocycle(gen, i) == t.cocycle(gen, i) && ref.perm(gen, i) == t.perm(gen, i);
        }
    }
    g(ok, tag + ": all ten rho_n tables equal to the default table's, block and scalar flag");
}

void ArmTables(Checker& g) {
    struct C { uint64_t p; uint32_t r; uint32_t ell; uint32_t n; };
    const std::vector<C> cells = { {5, 1, 5, 10}, {7, 1, 7, 12}, {13, 1, 13, 8}, {691, 1, 691, 10},
                                   {2, 1, 691, 10}, {3, 1, 13, 12}, {5, 2, 5, 8}, {2, 3, 7, 6} };
    for (const C& c : cells) {
        const std::string tag = "(A) p^r=" + std::to_string(c.p) + "^" + std::to_string(c.r) + " ell=" +
                                std::to_string(c.ell) + " n=" + std::to_string(c.n);
        try {
            std::unique_ptr<hecke::HeckeCosetTables> ref_p;
            {
                ScopedCosetStreaming g2(false);
                ScopedParallelSetup g3(false);
                ref_p.reset(new hecke::HeckeCosetTables(c.p, c.r, c.ell, c.n));
            }
            const hecke::HeckeCosetTables& ref = *ref_p;
            for (const Levers& L : Combos(false)) {
                const LeverScope scope(L);
                const hecke::HeckeCosetTables t(c.p, c.r, c.ell, c.n);
                const bool want = L.dflt ? g_default_stream : L.stream;
                g(t.streamed() == want, tag + " " + L.name() + ": streamed() reports the lever");
                CompareTables<uint32_t>(g, tag + " " + L.name(), ref, t);
                if (want) {
                    g(t.self_check() == 0, tag + " " + L.name() + ": self_check() on the streamed table");
                    g(t.self_check_upper() == 0, tag + " " + L.name() + ": self_check_upper() on the streamed table");
                }
            }
            std::cout << "  " << tag << ": " << ref.num_cosets() << " cosets -- equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
    {   // uint64_t tables (the module route's), at a composite-sized ell.
        const std::string tag = "(A) uint64 p=691 ell=691 n=12";
        try {
            std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> ref_p;
            {
                ScopedCosetStreaming off(false);
                ref_p.reset(new hecke::HeckeCosetTablesT<uint64_t>(691, 1, 691, 12));
            }
            const hecke::HeckeCosetTablesT<uint64_t>& ref = *ref_p;
            ScopedCosetStreaming g2(true);
            ScopedParallelSetup g3(true);
            const hecke::HeckeCosetTablesT<uint64_t> t(691, 1, 691, 12);
            CompareTables<uint64_t>(g, tag + " stream+par", ref, t);
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
    {   // the refusal: a streamed table's reference accessors.
        ScopedCosetStreaming g2(true);
        const hecke::HeckeCosetTables t(5, 1, 5, 10);
        int refused = 0;
        try { (void)t.rho_s_inv(1); } catch (const std::logic_error& e) {
            refused += std::string(e.what()).find("rho_s_inv_block") != std::string::npos;
        }
        try { (void)t.rho_t(1); } catch (const std::logic_error&) { ++refused; }
        try { (void)t.rho_cocycle(0, 1); } catch (const std::logic_error&) { ++refused; }
        try { (void)t.rho_cocycle_upper(0, 1); } catch (const std::logic_error&) { ++refused; }
        g(refused == 4, "(A) a streamed table's four reference accessors refuse (ENGINE_CHECK, naming the block accessor)");
        std::cout << "  (A) streamed table: " << refused << " / 4 reference accessors refused\n";
    }
}

// ---- (B) the tau table under ScopedParallelSetup ----------------------------
void ArmTau(Checker& g) {
    struct C { uint64_t p; uint32_t ell; uint32_t n; bool lever; };
    const std::vector<C> cells = { {5, 5, 10, false}, {5, 5, 10, true}, {691, 691, 22, false}, {691, 691, 22, true},
                                   {2, 691, 12, false}, {13, 13, 24, true}, {3, 7, 12, false} };
    for (const C& c : cells) {
        const std::string tag = "(B) p=" + std::to_string(c.p) + " ell=" + std::to_string(c.ell) + " n=" +
                                std::to_string(c.n) + (c.lever ? " lever" : "");
        try {
            const Model M;
            ScopedUEllLowRank g1(c.lever);
            std::unique_ptr<hecke::HeckeCosetTables> cosets_p;
            {
                ScopedCosetStreaming off(false);
                cosets_p.reset(new hecke::HeckeCosetTables(c.p, 1, c.ell, c.n));
            }
            const hecke::HeckeCosetTables& cosets = *cosets_p;
            std::unique_ptr<hecke::HeckeTauTable> ref, par;
            {
                ScopedParallelSetup off(false);
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                ref.reset(new hecke::HeckeTauTable(*M.K, *M.model, cosets, c.ell, c.n));
            }
            {
                ScopedParallelSetup g3(true);
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                par.reset(new hecke::HeckeTauTable(*M.K, *M.model, cosets, c.ell, c.n));
            }
            bool terms = true;
            for (uint32_t q = 0; q < 3; ++q)
                for (uint32_t s = 0; s < M.K->get_num_q_simplices(q); ++s)
                    for (uint32_t i = 1; i <= cosets.num_cosets(); ++i) {
                        const auto& a = ref->xi_terms(i, q, s);
                        const auto& b = par->xi_terms(i, q, s);
                        terms = terms && a.size() == b.size();
                        for (size_t k = 0; terms && k < a.size(); ++k)
                            terms = a[k].coeff == b[k].coeff && a[k].sigma_idx == b[k].sigma_idx &&
                                    a[k].rho_g_gamma.index() == b[k].rho_g_gamma.index();
                    }
            g(terms, tag + ": the same xi terms (coefficient, simplex, pool slot)");
            g(ref->rho_pool_is_low_rank() == c.lever && par->rho_pool_is_low_rank() == c.lever,
              tag + ": the pool is factored iff the lever is on");
            const size_t n_slots = ref->rho_pool().size();
            bool pool = n_slots == par->rho_pool().size();
            for (size_t s = 0; pool && s < n_slots; ++s) {
                if (c.lever) {
                    const auto& a = ref->rho_pool_low_rank(static_cast<uint32_t>(s));
                    const auto& b = par->rho_pool_low_rank(static_cast<uint32_t>(s));
                    pool = a.rows == b.rows && a.cols == b.cols && a.t == b.t && a.U == b.U && a.V == b.V;
                } else {
                    pool = SameBlock(ref->rho_pool()[s], par->rho_pool()[s]);
                }
            }
            g(pool, tag + ": the rho pool equal slot by slot (" + std::to_string(n_slots) + " slots)");
            if (!c.lever) {
                // The program-cache pattern hash of the fused program (the Omega
                // kinds read through the block accessor on a streamed table):
                // the same key with and without either lever.
                ScopedCosetStreaming g2(true);
                const hecke::HeckeCosetTables streamed(c.p, 1, c.ell, c.n);
                bool same = true;
                for (uint32_t q = 0; q < 3; ++q) {
                    const uint32_t N_q = M.K->get_num_q_simplices(q);
                    const uint64_t h0 = hecke::HeckeOperatorOnK::tau_pattern_hash(*ref, cosets, q, N_q,
                                                                                  cosets.num_cosets(), c.ell);
                    same = same && h0 == hecke::HeckeOperatorOnK::tau_pattern_hash(*par, cosets, q, N_q,
                                                                                   cosets.num_cosets(), c.ell);
                    same = same && h0 == hecke::HeckeOperatorOnK::tau_pattern_hash(*ref, streamed, q, N_q,
                                                                                   streamed.num_cosets(), c.ell);
                }
                g(same, tag + ": the fused program's pattern hash equal (parallel tau; streamed cosets), q = 0, 1, 2");
            }
            {
                // (slot-14 WP-16, audit S-4) certification (f) in the FAILING direction, on both
                // paths: the intact table certifies 0 serially and in parallel; after one pool slot
                // is corrupted (a validation-only mutator) both paths must return code 60.
                uint32_t ok_ser = 0, ok_par = 0, bad_ser = 0, bad_par = 0;
                { ScopedParallelSetup s0(false); ok_ser = par->certify(); }
                { ScopedParallelSetup s1(true); ok_par = par->certify(); }
                ref->corrupt_rho_pool_for_validation();
                { ScopedParallelSetup s0(false); bad_ser = ref->certify(); }
                { ScopedParallelSetup s1(true); bad_par = ref->certify(); }
                g(ok_ser == 0 && ok_par == 0, tag + ": certify() == 0 on the intact table, serial and parallel (got " +
                  std::to_string(ok_ser) + ", " + std::to_string(ok_par) + ")");
                g(bad_ser == 60 && bad_par == 60, tag + ": a corrupted pool slot fails (f) with code 60, serial and parallel (got " +
                  std::to_string(bad_ser) + ", " + std::to_string(bad_par) + ")");
                std::cout << "  " << tag << ": (f) negative arm: corrupted " << (c.lever ? "factored" : "dense")
                          << " pool -> certify() serial " << bad_ser << ", parallel " << bad_par << "\n";
            }
            std::cout << "  " << tag << ": " << n_slots << " pool slots -- equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

// ---- (C) F_ell operators ---------------------------------------------------------
void ArmField(Checker& g) {
    struct C { uint32_t n; uint32_t ell; uint64_t p; uint32_t r; };
    const std::vector<C> cells = {
        {10, 5, 5, 1}, {24, 5, 5, 1}, {12, 7, 7, 1}, {22, 7, 7, 1}, {12, 11, 11, 1}, {24, 13, 13, 1},
        {10, 691, 691, 1}, {36, 691, 691, 1}, {10, 5, 5, 2}, {10, 691, 2, 1}, {12, 13, 3, 1},
    };
    for (const C& c : cells) {
        const std::string tag = "(C) n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) + " T_{" +
                                std::to_string(c.p) + "^" + std::to_string(c.r) + "}";
        try {
            const FieldCell cell(c.n, c.ell);
            const FieldOp def(cell, c.p, c.r, c.n, c.ell, Levers{});
            const bool ell_divides = c.p == c.ell;
            uint32_t runs = 0;
            for (const Levers& L : Combos(ell_divides)) {
                const FieldOp alt(cell, c.p, c.r, c.n, c.ell, L);
                g(alt.cosets->streamed() == (L.dflt ? g_default_stream : L.stream),
                  tag + " " + L.name() + ": the table streams iff the lever is on");
                if (L.dflt)
                    g(alt.tau->rho_pool_is_low_rank() == (g_default_lever && c.p == c.ell),
                      tag + " DEFAULT: the U_ell lever is in effect exactly where ell | N");
                CompareOps(g, tag + " " + L.name(), *def.op, *alt.op);
                ++runs;
            }
            std::cout << "  " << tag << ": " << runs << " lever combinations -- equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
    // The legacy arm (what ENGINE_BACKEND=legacy selects) over a streamed table.
    struct L2 { uint32_t n; uint32_t ell; uint64_t p; };
    for (const L2& c : std::vector<L2>{ {10, 691, 691}, {24, 13, 13}, {10, 691, 2} }) {
        const std::string tag = "(C) legacy arm n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) +
                                " T_" + std::to_string(c.p);
        try {
            const FieldCell cell(c.n, c.ell);
            const FieldOp def(cell, c.p, 1, c.n, c.ell, Levers{});
            const FieldOp st(cell, c.p, 1, c.n, c.ell, Levers{ true, true, false }, true);
            CompareOps(g, tag + " stream+par", *def.op, *st.op);
            if (c.p == c.ell) {
                const FieldOp lv(cell, c.p, 1, c.n, c.ell, Levers{ true, true, true }, true);
                CompareOps(g, tag + " stream+par+lever", *def.op, *lv.op);
            }
            std::cout << "  " << tag << ": equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

// ---- (D) the module route ------------------------------------------------------
void ArmModule(Checker& g) {
    struct C { uint32_t n; uint64_t ell; uint32_t k; };
    const std::vector<C> cells = { {10, 5, 2}, {12, 7, 2}, {12, 11, 2}, {12, 13, 2}, {10, 691, 2}, {24, 5, 3} };
    for (const C& c : cells) {
        const std::string tag = "(D) n=" + std::to_string(c.n) + " Z/" + std::to_string(c.ell) + "^" + std::to_string(c.k);
        try {
            const hecke::ScopedCoefficientRing ring(c.k);
            const engine::ModCtx<uint64_t> ctx =
                hecke::CoefficientRingContext<uint64_t>("HeckeUEllStreamValidation", c.ell, c.k);
            const Model M;
            LocalSystemSimplicialComplexT<uint64_t> lssc(*M.K, hecke::GenerateSL2ZRelationsRing<uint64_t>(*M.K, c.n, ctx),
                                                         c.n + 1, ctx.p);
            EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
            coh.compute_cohomology();
            auto run = [&](const Levers& L, std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>>& cosets,
                           std::unique_ptr<hecke::HeckeTauTableT<uint64_t>>& tau)
                -> std::unique_ptr<hecke::HeckeOperatorOnKT<uint64_t>> {
                const LeverScope scope(L);
                {
                    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                    cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(c.ell, 1, c.ell, c.n));
                    tau.reset(new hecke::HeckeTauTableT<uint64_t>(*M.K, *M.model, *cosets, c.ell, c.n));
                }
                std::unique_ptr<hecke::HeckeOperatorOnKT<uint64_t>> op(
                    new hecke::HeckeOperatorOnKT<uint64_t>(&coh, cosets.get(), tau.get()));
                op->compute();
                return op;
            };
            std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> c0;
            std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> t0;
            const auto def = run(Levers{}, c0, t0);
            uint32_t runs = 0;
            for (const Levers& L : Combos(true)) {
                if (!L.dflt && !L.stream && !L.par) continue;   // the lever alone is HeckeUEllLowRankValidation's arm (C)
                std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> c1;
                std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> t1;
                const auto alt = run(L, c1, t1);
                for (uint32_t q = 0; q < 3; ++q) {
                    const auto& A0 = def->H_matrix_module(q);
                    const auto& A1 = alt->H_matrix_module(q);
                    bool eq = A0.rows == A1.rows && A0.cols == A1.cols;
                    for (uint32_t i = 0; eq && i < A0.rows; ++i)
                        for (uint32_t j = 0; j < A0.cols; ++j)
                            if (A0.at(i, j) != A1.at(i, j)) { eq = false; break; }
                    g(eq, tag + " " + L.name() + " A^(" + std::to_string(q) + ") entrywise");
                    const auto& T0 = def->Theta_module(q);
                    const auto& T1 = alt->Theta_module(q);
                    bool teq = T0.rows == T1.rows && T0.cols == T1.cols;
                    for (uint32_t i = 0; teq && i < T0.rows; ++i)
                        for (uint32_t j = 0; j < T0.cols; ++j)
                            if (T0.at(i, j) != T1.at(i, j)) { teq = false; break; }
                    g(teq, tag + " " + L.name() + " Theta^(" + std::to_string(q) + ") entrywise");
                }
                ++runs;
            }
            std::cout << "  " << tag << ": " << runs << " lever combinations -- equal\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

// ---- (E) 1 vs 16 consumer-pool workers -----------------------------------------
void ArmThreads(Checker& g) {
    const uint32_t n = 36, ell = 691;
    const std::string tag = "(E) n=36 ell=691 U_691";
    try {
        const FieldCell cell(n, ell);
        const FieldOp def(cell, ell, 1, n, ell, Levers{});
        for (const uint32_t workers : { 1u, 16u }) {
            engine::tape::ThreadPool pool(workers);
            engine::tape::ScopedConsumerPool sp(&pool);
            Levers d; d.dflt = true;
            for (const Levers& L : { d, Levers{ true, true, false }, Levers{ true, true, true } }) {
                const FieldOp alt(cell, ell, 1, n, ell, L);
                CompareOps(g, tag + " threads=" + std::to_string(workers) + " " + L.name(), *def.op, *alt.op);
            }
        }
        std::cout << "  " << tag << ": threads 1 and 16, stream+par with and without the lever -- equal\n";
    } catch (const std::exception& e) {
        g(false, tag + " threw: " + e.what());
    }
}

// ---- (F) the other consumers on a streamed table --------------------------------
void ArmCrossChecks(Checker& g) {
    const uint32_t n = 10, ell = 691;
    const std::string tag = "(F) cross-checks n=10 ell=691 T_2";
    try {
        const FieldCell cell(n, ell);
        const FieldOp def(cell, 2, 1, n, ell, Levers{});
        Levers dl; dl.dflt = true;
        const FieldOp st(cell, 2, 1, n, ell, dl);
        uint32_t rc0 = 1, rc1 = 1;
        {
            hecke::HeckeCrossChecks xc(cell.coh.get(), def.cosets.get(), def.tau.get());
            rc0 = xc.run_all(*def.op);
        }
        {
            hecke::HeckeCrossChecks xc(cell.coh.get(), st.cosets.get(), st.tau.get());
            rc1 = xc.run_all(*st.op);
        }
        g(rc0 == 0, tag + ": the default table passes");
        g(st.cosets->streamed() == g_default_stream, tag + ": the DEFAULT coset table streams iff the default is on");
        g(rc1 == 0, tag + ": the streamed table passes (induced local systems, rho_t, rho_s_inv by block)");
        std::cout << "  " << tag << ": rc default " << rc0 << ", streamed " << rc1 << "\n";
    } catch (const std::exception& e) {
        g(false, tag + " threw: " + e.what());
    }
}

// ---- (H) the rank-one closed form (slot-14 G0) ------------------------------------
// Returns 0 = equal, 1 = differs, 2 = only the dense route refused the key,
// 3 = only the closed form refused it; `both_refused` is set when both refused.
template<class Elt>
int CompareClosedForm(const engine::ModCtx<Elt>& ctx, Elt a, Elt b, Elt c, Elt d, uint32_t n,
                      const std::vector<Elt>& binom, bool& both_refused) {
    both_refused = false;
    const Elt ell = ctx.p;
    const uint32_t dV = n + 1;
    const FlatMatrixT<Elt> R = ComputeSymmetricPowerMatrixFM(FlatMatrixT<Elt>(2, 2, ell, std::vector<Elt>{ a, b, c, d }), n);
    bool dense_ok = true, closed_ok = true;
    hecke::lowrank::LowRankT<Elt> ref, got;
    try {
        ref = hecke::lowrank::factor_low_rank<Elt>(ctx, hecke::lowrank::dense_rows_of<Elt>(R), dV, dV, 1u);
    } catch (const std::logic_error&) { dense_ok = false; }
    try {
        got = hecke::lowrank::rank_one_rho_factors<Elt>(ctx, a, b, c, d, n, binom);
    } catch (const std::logic_error&) { closed_ok = false; }
    if (!dense_ok && !closed_ok) { both_refused = true; return 0; }
    if (!dense_ok) return 2;
    if (!closed_ok) return 3;
    if (ref.rows != got.rows || ref.cols != got.cols || ref.t != got.t || ref.U != got.U || ref.V != got.V) return 1;
    for (uint32_t r = 0; r < dV; ++r)          // U V == rho_n entrywise
        for (uint32_t s = 0; s < dV; ++s) {
            const Elt uv = got.t == 0 ? Elt(0) : ctx.mul(got.U[r], got.V[s]);
            if (uv != R.get_entry(r, s)) return 1;
        }
    return 0;
}

template<class Elt>
void ClosedFormGrid(Checker& g, const std::string& width, uint32_t max_n) {
    struct C { uint32_t ell; std::vector<uint32_t> ns; bool exhaustive; uint32_t random_keys; };
    const std::vector<C> cells = {
        // (G0 repairs, audit S2) ell in {2, 3}: the lever is Auto there too (U_2, U_3 run the field
        // pipeline, e.g. sweep_full's cells), so the closed form is live -- every 2x2 key (16 / 81)
        {2, {0, 1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64}, true, 0},
        {3, {0, 1, 2, 3, 8, 9, 10, 26, 27, 28, 80, 81, 82}, true, 0},
        {5, {0, 1, 2, 3, 4, 5, 6, 9, 10, 24, 25, 26, 124, 125, 126}, true, 0},
        {7, {0, 1, 6, 7, 8, 13, 14, 48, 49, 50, 100}, true, 0},
        {11, {0, 2, 10, 11, 12, 22, 120, 121, 122}, false, 300},
        {13, {1, 12, 13, 14, 26, 168, 169, 170, 200}, false, 300},
        {691, {0, 1, 10, 128, 384, 690, 691, 692, 700, 1382, 2080}, false, 40},
        {2411, {0, 2, 10, 128, 384, 1000, 2410, 2411, 2412, 2420}, false, 30},
    };
    uint64_t keys = 0, refused_nonsingular = 0;
    for (const C& c : cells) {
        const engine::ModCtx<Elt> ctx(static_cast<Elt>(c.ell));
        const Elt L = static_cast<Elt>(c.ell);
        for (const uint32_t n : c.ns) {
            if (n > max_n) continue;
            // large n: fewer pseudo-random keys (each dense rho_n is O(d^2); the structured keys all run)
            const uint32_t n_random = n > 500 ? std::min<uint32_t>(c.random_keys, 6) : c.random_keys;
            const std::string at = "(H) " + width + " ell=" + std::to_string(c.ell) + " n=" + std::to_string(n);
            const std::vector<Elt> binom = hecke::lowrank::binomial_row_mod<Elt>(ctx, n);
            {   // the binomial row itself: column 0 of rho_n([[1,1],[1,1]]) is (1 + y)^n
                const FlatMatrixT<Elt> P = ComputeSymmetricPowerMatrixFM(
                    FlatMatrixT<Elt>(2, 2, L, std::vector<Elt>{ 1, 1, 1, 1 }), n);
                bool ok = binom.size() == static_cast<size_t>(n) + 1;
                for (uint32_t k = 0; ok && k <= n; ++k) ok = P.get_entry(k, 0) == binom[k];
                g(ok, at + ": binomial_row_mod == column 0 of rho_n([[1,1],[1,1]]) (Lucas)");
            }
            std::vector<std::array<Elt, 4>> ks;
            if (c.exhaustive) {
                for (Elt a = 0; a < L; ++a)
                    for (Elt b = 0; b < L; ++b)
                        for (Elt cc = 0; cc < L; ++cc)
                            for (Elt d = 0; d < L; ++d) ks.push_back({ a, b, cc, d });
            } else {
                // structured: zero, one row / one column, lambda = 0, lambda = -1, [[1,0],[0,1]] (non-singular)
                ks = { {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 0, 0},
                       {0, 0, 1, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, static_cast<Elt>(L - 1), 0, 0},
                       {0, 0, 1, static_cast<Elt>(L - 1)}, {1, 1, 1, 1}, {2, 4, 3, 6}, {1, 0, 0, 1}, {1, 1, 0, 1} };
                uint64_t x = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(c.ell) << 20) ^ n;
                auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return static_cast<Elt>(x % c.ell); };
                for (uint32_t r = 0; r < n_random; ++r) {
                    // a random singular key ((b, d) = lambda (a, c), or a zero first column), and a
                    // random key -- almost surely non-singular -- for the negative direction
                    const Elt a = rnd(), cc = rnd(), lam = rnd();
                    if (r % 5 == 0) ks.push_back({ 0, rnd(), 0, rnd() });
                    else ks.push_back({ a, ctx.mul(lam, a), cc, ctx.mul(lam, cc) });
                    ks.push_back({ rnd(), rnd(), rnd(), rnd() });
                }
            }
            // the keys in parallel (the fan-out under test in (I) as well), verdicts per key
            std::vector<int> rcs(ks.size(), 0);
            std::vector<char> boths(ks.size(), 0);
            hecke::HeckeOperatorOptions::for_each_index(ks.size(), true, [&](size_t j) {
                bool both = false;
                rcs[j] = CompareClosedForm<Elt>(ctx, ks[j][0], ks[j][1], ks[j][2], ks[j][3], n, binom, both);
                boths[j] = both ? 1 : 0;
            }, UINT64_MAX / 2);
            uint64_t differ = 0, refusal_mismatch = 0, here = 0;
            for (size_t j = 0; j < ks.size(); ++j) {
                const auto& k = ks[j];
                const bool both = boths[j] != 0;
                const int rc = rcs[j];
                const bool singular = ctx.mul(k[0], k[3]) == ctx.mul(k[1], k[2]);
                if (both) {
                    ++refused_nonsingular;
                    if (singular) ++refusal_mismatch;   // a singular key must never be refused
                    continue;
                }
                if (!singular && n >= 1) { ++refusal_mismatch; continue; }   // rank n + 1 >= 2: must be refused
                if (!singular) {
                    // n = 0: rho_0 = [1] has rank 1 for every key; the dense route accepts it, the
                    // closed form (singular keys only) refuses it -- rc == 3 is the expected verdict
                    if (rc != 3) ++refusal_mismatch;
                    continue;
                }
                if (rc != 0) ++differ;
                ++here;
            }
            keys += here;
            g(differ == 0 && refusal_mismatch == 0,
              at + ": " + std::to_string(ks.size()) + " keys (" + std::to_string(here) +
                  " singular), (t, U, V) and U V == rho_n entrywise (differ " + std::to_string(differ) +
                  ", refusal mismatches " + std::to_string(refusal_mismatch) + ")");
        }
    }
    std::cout << "  (H) " << width << ": " << keys << " singular keys equal to the dense route; " << refused_nonsingular
              << " non-singular keys refused by both routes (the rank claim)\n";
}

void ClosedFormTau(Checker& g, uint64_t p, uint32_t ell, uint32_t n) {
    const std::string tag = "(H) tau pool p=" + std::to_string(p) + " ell=" + std::to_string(ell) + " n=" + std::to_string(n);
    try {
        const Model M;
        ScopedUEllLowRank g1(true);
        std::unique_ptr<hecke::HeckeCosetTables> cosets_p;
        {
            ScopedCosetStreaming off(false);
            cosets_p.reset(new hecke::HeckeCosetTables(p, 1, ell, n));
        }
        const hecke::HeckeCosetTables& cosets = *cosets_p;
        std::unique_ptr<hecke::HeckeTauTable> closed, dense;
        {
            hecke::HeckeOperatorOptions::ScopedRankOneClosedForm cf(true);
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            closed.reset(new hecke::HeckeTauTable(*M.K, *M.model, cosets, ell, n));
        }
        {
            hecke::HeckeOperatorOptions::ScopedRankOneClosedForm cf(false);
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            dense.reset(new hecke::HeckeTauTable(*M.K, *M.model, cosets, ell, n));
        }
        const size_t slots = dense->rho_pool().size();
        bool same = slots == closed->rho_pool().size() && closed->rho_pool_is_low_rank() && dense->rho_pool_is_low_rank();
        uint32_t t1 = 0;
        for (size_t s = 0; same && s < slots; ++s) {
            const auto& a = dense->rho_pool_low_rank(static_cast<uint32_t>(s));
            const auto& b = closed->rho_pool_low_rank(static_cast<uint32_t>(s));
            same = a.rows == b.rows && a.cols == b.cols && a.t == b.t && a.U == b.U && a.V == b.V;
            t1 += b.t;
        }
        g(same, tag + ": every factored pool slot (t, U, V) equal with and without the closed form (" +
                    std::to_string(slots) + " slots)");
        g(closed->certify() == 0, tag + ": certify() (f) -- the dense witness -- passes on the closed-form pool");
        std::cout << "  " << tag << ": " << slots << " slots (" << t1 << " rank-1) -- equal\n";
    } catch (const std::exception& e) {
        g(false, tag + " threw: " + e.what());
    }
}

void ArmClosedForm(Checker& g) {
    ClosedFormGrid<uint32_t>(g, "uint32", 1u << 30);
    ClosedFormGrid<uint64_t>(g, "uint64", 700);   // the width is exercised; the largest n run at uint32
    {   // negative: a perturbed closed-form factor is caught entrywise against rho_n
        const engine::ModCtx<uint32_t> ctx(691u);
        const std::vector<uint32_t> binom = hecke::lowrank::binomial_row_mod<uint32_t>(ctx, 40);
        auto f = hecke::lowrank::rank_one_rho_factors<uint32_t>(ctx, 3, 6, 5, 10, 40, binom);
        const FlatMatrixT<uint32_t> R = ComputeSymmetricPowerMatrixFM(
            FlatMatrixT<uint32_t>(2, 2, 691u, std::vector<uint32_t>{ 3, 6, 5, 10 }), 40);
        f.U[7] = ctx.add(f.U[7], 1u);
        bool caught = false;
        for (uint32_t s = 0; s <= 40 && !caught; ++s) caught = ctx.mul(f.U[7], f.V[s]) != R.get_entry(7, s);
        g(caught, "(H) negative: U[7] += 1 on a closed-form factor is caught entrywise against rho_n");
        bool refused = false;
        try { (void)hecke::lowrank::rank_one_rho_factors<uint32_t>(ctx, 1, 2, 3, 4, 40, binom); }
        catch (const std::logic_error& e) { refused = std::string(e.what()).find("rank claim") != std::string::npos; }
        g(refused, "(H) negative: the non-singular key [[1,2],[3,4]] mod 691 is refused, naming the rank claim");
        bool ring_refused = false;
        try {
            const engine::ModCtx<uint32_t> ring = engine::ModCtx<uint32_t>::prime_power(5u, 2u);
            (void)hecke::lowrank::rank_one_rho_factors<uint32_t>(ring, 5, 0, 0, 0, 4, std::vector<uint32_t>(5, 1u));
        } catch (const std::logic_error&) { ring_refused = true; }
        g(ring_refused, "(H) negative: the closed form refuses a composite ring (F_ell only)");
        std::cout << "  (H) negative arm: perturbation caught " << caught << ", non-singular refused " << refused
                  << ", ring refused " << ring_refused << "\n";
    }
    ClosedFormTau(g, 2, 2, 8);     // (G0 repairs, audit S2)
    ClosedFormTau(g, 3, 3, 8);     // (G0 repairs, audit S2)
    ClosedFormTau(g, 5, 5, 24);
    ClosedFormTau(g, 13, 13, 40);
    ClosedFormTau(g, 691, 691, 36);
    ClosedFormTau(g, 2411, 2411, 16);
    // The operators under the lever: the closed form (default) against its opt-out, and against all-off.
    struct C { uint32_t n; uint32_t ell; uint64_t p; uint32_t r; };
    for (const C& c : std::vector<C>{ {8, 2, 2, 1}, {8, 3, 3, 1}, {10, 5, 5, 1}, {24, 5, 5, 2}, {22, 7, 7, 1}, {24, 13, 13, 1},
                                      {36, 691, 691, 1}, {16, 2411, 2411, 1} }) {
        const std::string tag = "(H) op n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) + " T_{" +
                                std::to_string(c.p) + "^" + std::to_string(c.r) + "}";
        try {
            const FieldCell cell(c.n, c.ell);
            std::unique_ptr<FieldOp> off, on;
            {
                hecke::HeckeOperatorOptions::ScopedRankOneClosedForm cf(false);
                off.reset(new FieldOp(cell, c.p, c.r, c.n, c.ell, Levers{ true, true, true }));
            }
            {
                hecke::HeckeOperatorOptions::ScopedRankOneClosedForm cf(true);
                on.reset(new FieldOp(cell, c.p, c.r, c.n, c.ell, Levers{ true, true, true }));
            }
            const FieldOp ref(cell, c.p, c.r, c.n, c.ell, Levers{});   // every lever off: the dense reference
            CompareOps(g, tag + " closed form vs opt-out", *off->op, *on->op);
            CompareOps(g, tag + " closed form vs all-off", *ref.op, *on->op);
            std::cout << "  " << tag << ": closed form == opt-out == all-off\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
    // (G0 repairs, audit m2) A DENSE tau pool (the table built with the lever off) read by an
    // operator computed with the lever in effect: the host assembly factors each slot itself --
    // by the closed form from the slot's reduced key (HeckeTauTable::rho_pool_key), or by the
    // dense factoriser under the opt-out.  Both, and the all-off operator, equal.
    for (const C& c : std::vector<C>{ {8, 3, 3, 1}, {10, 5, 5, 1}, {24, 5, 5, 2}, {36, 691, 691, 1} }) {
        const std::string tag = "(H) dense tau pool, lever on: n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) +
                                " T_{" + std::to_string(c.p) + "^" + std::to_string(c.r) + "}";
        try {
            const FieldCell cell(c.n, c.ell);
            std::unique_ptr<hecke::HeckeCosetTables> cosets;
            std::unique_ptr<hecke::HeckeTauTable> tau;
            {
                ScopedUEllLowRank lever_off(false);
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                cosets.reset(new hecke::HeckeCosetTables(c.p, c.r, c.ell, c.n));
                tau.reset(new hecke::HeckeTauTable(*cell.M.K, *cell.M.model, *cosets, c.ell, c.n));
            }
            g(!tau->rho_pool_is_low_rank(), tag + ": the tau pool is dense");
            bool keys_ok = true;   // rho_pool()[s] IS rho_n(rho_pool_key(s))
            for (uint32_t s = 0; keys_ok && s < tau->rho_pool().size(); ++s) {
                const auto& k = tau->rho_pool_key(s);
                keys_ok = tau->rho_pool()[s] == ComputeSymmetricPowerMatrixFM(
                              FlatMatrixT<uint32_t>(2, 2, c.ell, std::vector<uint32_t>{ k[0], k[1], k[2], k[3] }), c.n);
            }
            g(keys_ok, tag + ": every dense pool slot is rho_n of its rho_pool_key (" +
                           std::to_string(tau->rho_pool().size()) + " slots)");
            std::unique_ptr<hecke::HeckeOperatorOnK> on, off;
            {
                ScopedUEllLowRank lever(true);
                hecke::HeckeOperatorOptions::ScopedRankOneClosedForm cf(true);
                on.reset(new hecke::HeckeOperatorOnK(cell.coh.get(), cosets.get(), tau.get()));
                on->compute();
            }
            {
                ScopedUEllLowRank lever(true);
                hecke::HeckeOperatorOptions::ScopedRankOneClosedForm cf(false);
                off.reset(new hecke::HeckeOperatorOnK(cell.coh.get(), cosets.get(), tau.get()));
                off->compute();
            }
            const FieldOp ref(cell, c.p, c.r, c.n, c.ell, Levers{});
            CompareOps(g, tag + " closed form vs opt-out", *off, *on);
            CompareOps(g, tag + " closed form vs all-off", *ref.op, *on);
            std::cout << "  " << tag << ": closed form == dense factoriser == all-off\n";
        } catch (const std::exception& e) {
            g(false, tag + " threw: " + e.what());
        }
    }
}

// ---- (I) the one host fan-out budget (slot-14 G0, audit M-2) ----------------------
void ArmFanout(Checker& g) {
    namespace HO = hecke::HeckeOperatorOptions;
    for (const uint32_t workers : { 1u, 4u, 16u }) {
        const std::string tag = "(I) workers=" + std::to_string(workers);
        engine::tape::ThreadPool pool(workers);
        engine::tape::ScopedConsumerPool use(&pool);
        {   // every index exactly once
            std::vector<std::atomic<uint32_t>> hits(10007);
            for (auto& h : hits) h.store(0);
            HO::for_each_index(hits.size(), true, [&](size_t j) { hits[j].fetch_add(1); }, UINT64_MAX / 2);
            bool once = true;
            for (auto& h : hits) once = once && h.load() == 1;
            g(once, tag + ": every one of 10007 indices ran exactly once");
        }
        {   // the lowest failing index is rethrown (the serial loop's first failure)
            std::string what;
            try {
                HO::for_each_index(500, true, [&](size_t j) {
                    if (j == 377 || j == 41 || j == 499) throw std::runtime_error("fail at " + std::to_string(j));
                }, UINT64_MAX / 2);
            } catch (const std::runtime_error& e) { what = e.what(); }
            g(what == "fail at 41", tag + ": the lowest failing index's exception surfaces (got '" + what + "')");
        }
        {   // nested fan-outs inside a structural fan-out share the one helper set
            std::atomic<int> active{0}, peak{0};
            std::atomic<uint64_t> sum{0};
            std::vector<std::function<void()>> outer;
            for (int t = 0; t < 3; ++t)
                outer.emplace_back([&]() {
                    HO::for_each_index(4, true, [&](size_t) {
                        HO::for_each_index(2000, true, [&](size_t j) {
                            const int a = active.fetch_add(1) + 1;
                            int p = peak.load();
                            while (a > p && !peak.compare_exchange_weak(p, a)) {}
                            volatile uint64_t x = 0;
                            for (int k = 0; k < 200; ++k) x = x + static_cast<uint64_t>(k) * j;
                            sum.fetch_add(j);
                            active.fetch_sub(1);
                        }, UINT64_MAX / 2);
                    }, UINT64_MAX / 2);
                });
            engine::tape::run_concurrently(outer);
            const uint64_t want = 3ull * 4ull * (1999ull * 2000ull / 2);
            // the bound: the 3 structural callers plus at most (workers - 1) shared helpers
            const int bound = static_cast<int>(workers <= 1 ? 1 : 3 + (workers - 1));
            g(sum.load() == want, tag + ": nested fan-outs ran every item once (sum " + std::to_string(sum.load()) + ")");
            g(peak.load() <= bound, tag + ": nested concurrency " + std::to_string(peak.load()) + " <= " +
                                        std::to_string(bound) + " (3 callers + pool - 1 helpers; was up to 3 x 4 x pool)");
            std::cout << "  " << tag << ": nested peak concurrency " << peak.load() << " (bound " << bound << ")\n";
        }
    }
    {   // (G0 repairs, audit S1) THE BUDGET IS PER CONSUMER POOL: two pools of 4 workers in two
        // threads (the campaign job loop's per-job pools).  Gated: every item once, and each pool's
        // concurrency <= its 4 workers (the caller + 3 helpers).  Reported, not gated (timing): the
        // overall peak, >= 2 x 3 on a quiet box -- under G0's process-wide count it was <= 1 + 1 + 3.
        engine::tape::ThreadPool poolA(4), poolB(4);
        std::atomic<int> active{0}, peak{0};
        std::array<std::atomic<int>, 2> active_p{}, peak_p{};
        for (int k = 0; k < 2; ++k) { active_p[k].store(0); peak_p[k].store(0); }
        std::array<std::atomic<uint64_t>, 2> sum{};
        sum[0].store(0); sum[1].store(0);
        auto bump = [](std::atomic<int>& a, std::atomic<int>& p) {
            const int v = a.fetch_add(1) + 1;
            int q = p.load();
            while (v > q && !p.compare_exchange_weak(q, v)) {}
        };
        auto job = [&](int k, engine::tape::ThreadPool* pool) {
            engine::tape::ScopedConsumerPool use(pool);
            HO::for_each_index(3000, true, [&](size_t j) {
                bump(active, peak);
                bump(active_p[k], peak_p[k]);
                volatile uint64_t x = 0;
                for (int r = 0; r < 20000; ++r) x = x + static_cast<uint64_t>(r) * j;
                sum[k].fetch_add(j);
                active_p[k].fetch_sub(1);
                active.fetch_sub(1);
            }, UINT64_MAX / 2);
        };
        std::thread ta(job, 0, &poolA), tb(job, 1, &poolB);
        ta.join();
        tb.join();
        const uint64_t want = 2999ull * 3000ull / 2;
        g(sum[0].load() == want && sum[1].load() == want, "(I) two pools of 4: every item of both fan-outs ran once");
        g(peak_p[0].load() <= 4 && peak_p[1].load() <= 4,
          "(I) two pools of 4: per-pool concurrency " + std::to_string(peak_p[0].load()) + " / " +
              std::to_string(peak_p[1].load()) + " <= 4 (the caller + 3 helpers per pool)");
        std::cout << "  (I) two pools of 4 in two threads: per-pool peak " << peak_p[0].load() << " / " << peak_p[1].load()
                  << ", overall peak " << peak.load() << " (reported, not gated; >= 6 expected on a quiet box, "
                  << "<= 5 under a process-wide count)\n";
    }
}

// ---- (G) RSS, printed ----------------------------------------------------------------
void ArmRss() {
    const uint32_t n = 128, ell = 691;
    const FieldCell cell(n, ell);
    // The leaner configuration FIRST (a freed-but-retained heap would hide it).
    const Levers arms[2] = { Levers{ true, true, true }, Levers{ false, false, true } };
    double peak[2] = { 0.0, 0.0 };
    for (int a = 0; a < 2; ++a) {
        const bool reset = oc_util::ResetPeakRss();
        const double base = oc_util::CurrentRssMiB();
        {
            const FieldOp op(cell, ell, 1, n, ell, arms[a]);
            (void)op;
        }
        peak[a] = reset ? oc_util::PeakRssMiB() - base : -1.0;
    }
    std::cout << std::fixed << std::setprecision(1)
              << "  (G) U_691 at n = 128 (cosets + tau + operator), peak RSS above the cohomology: lever " << peak[1]
              << " MiB, lever+stream+par " << peak[0] << " MiB (printed, not gated; -1 = VmHWM reset unavailable)\n";
}

} // namespace

uint32_t HeckeUEllStreamValidation() {
    std::cout << "=== HeckeUEllStreamValidation (slot-14 WP-10: coset streaming + the serial phase on the pool; G0: the rank-one closed form + the fan-out budget) ===\n";
    Checker g;
    g_default_stream = hecke::HeckeOperatorOptions::coset_streaming();
    g_default_lever = hecke::HeckeOperatorOptions::u_ell_low_rank_mode() != hecke::HeckeOperatorOptions::UEllLowRankMode::Off;
    std::cout << "EFFECTIVE HeckeUEllStreamValidation default_coset_stream=" << (g_default_stream ? 1 : 0)
              << " default_parallel_setup=" << (hecke::HeckeOperatorOptions::parallel_setup() ? 1 : 0)
              << " default_u_ell_low_rank=" << (g_default_lever ? "auto" : "off")
              << " default_rank1_closed_form=" << (hecke::HeckeOperatorOptions::rank_one_closed_form() ? 1 : 0) << "\n";
    // Every fan-out of >= 2 items runs in parallel here (the production
    // threshold would keep these small cells serial).
    hecke::HeckeOperatorOptions::ScopedParallelSetupMinWork all_parallel(0);
    ArmTables(g);
    ArmTau(g);
    ArmField(g);
    ArmModule(g);
    ArmThreads(g);
    ArmCrossChecks(g);
    {
        const auto t0 = std::chrono::steady_clock::now();
        ArmClosedForm(g);
        const auto t1 = std::chrono::steady_clock::now();
        ArmFanout(g);
        const auto t2 = std::chrono::steady_clock::now();
        std::cout << std::fixed << std::setprecision(2) << "  (H) " << std::chrono::duration<double>(t1 - t0).count()
                  << " s, (I) " << std::chrono::duration<double>(t2 - t1).count() << " s\n";
    }
    ArmRss();
    std::cout << "HeckeUEllStreamValidation: " << g.checks << " checks, " << g.failures << " failures\n";
    return g.failures == 0 ? 0u : 1u;
}
