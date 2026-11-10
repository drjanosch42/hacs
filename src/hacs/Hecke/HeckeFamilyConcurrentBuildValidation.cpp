// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeFamilyConcurrentBuildValidation.cpp -- hecke-extensions slot 9, WP-7 (lever L2).
//
// Lever L2 narrowed HeckeOperatorFamily::build_pipelines()'s hold of the
// process-wide tau_construction_mutex() to the priming of K's simplex index:
// the coset, tau and operator CONSTRUCTIONS of two different families may now
// overlap.  This validation runs exactly that, deterministically enough to be
// a test and concurrently enough for ThreadSanitizer to see every shared write:
//
//   (A) two independent cells, (n, ell) = (10, 691) and (12, 1009), each with
//       its own K / model / cohomology, build U_ell and T_2 on two threads
//       released by one barrier;
//   (B) two families on ONE shared K / model / cohomology, (8, 691), build
//       U_691 and T_3 at once (the shared-K case the priming lock is for);
//
// and requires, per family, (1) every A^{(q)} entrywise equal to the same
// family built serially on the main thread first, and (2) the thread's
// PredicateStats deltas (thread_local since slot-8 WP-6) equal to the serial
// run's -- the counters HeckeFpFilterValidation asserts, i.e. the audit's
// hazard 3 is really gone.  Run under CheckThread
// (`setarch x86_64 -R ./cohomology_app HeckeFamilyConcurrentBuildValidation`)
// for the race claim.  ~2 s on RelWithDebInfo.  Writes nothing.
#include "HeckeFamilyConcurrentBuildValidation.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "HeckeOperatorFamily.h"
#include "Predicates.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"

namespace {

struct Stack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;
    Stack(uint32_t n_sym, uint32_t ell)
        : K(CreateSL2ZFundamentalDomainTriangulation()), model(K),
          lssc(K, Relations(K, n_sym, ell), n_sym + 1, ell), coh(&lssc) {
        coh.compute_cohomology();
    }
    static std::vector<LocalSystemSimplicialComplex::SimplexRelation> Relations(const SimplicialComplex& K,
                                                                                uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& d : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(d.rel);
        return rels;
    }
};

// One family's work: U_ell and T_p, the A^{(q)} matrices and this thread's
// PredicateStats deltas over the construction.
struct FamilyRun {
    std::vector<Matrix> U, T;
    uint64_t fp64_hits = 0, fp64_exact = 0, fp32_hits = 0, fp32_exact = 0;
    std::string error;
};

FamilyRun RunFamily(const Stack& s, uint32_t ell, uint32_t n_sym, uint64_t p) {
    FamilyRun r;
    try {
        hecke::reset_predicate_stats();
        hecke::HeckeOperatorFamily fam(&s.coh, s.K, s.model, ell, n_sym);
        r.U = fam.T_p(ell);
        r.T = fam.T_p(p);
        r.fp64_hits = hecke::predicate_stats_fp64().fp_hits;
        r.fp64_exact = hecke::predicate_stats_fp64().exact_fallbacks;
        r.fp32_hits = hecke::predicate_stats_fp32().fp_hits;
        r.fp32_exact = hecke::predicate_stats_fp32().exact_fallbacks;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    return r;
}

struct Job { const Stack* stack; uint32_t ell, n_sym; uint64_t p; std::string tag; };

uint32_t Compare(const FamilyRun& ser, const FamilyRun& par, const std::string& tag) {
    uint32_t fails = 0;
    auto check = [&](bool ok, const std::string& what) {
        if (!ok) { std::cout << "  FAIL: " << tag << ": " << what << std::endl; ++fails; }
    };
    check(ser.error.empty() && par.error.empty(), "no exception (serial: '" + ser.error + "', concurrent: '" + par.error + "')");
    if (!ser.error.empty() || !par.error.empty()) return fails;
    check(ser.U.size() == par.U.size() && ser.T.size() == par.T.size(), "degree counts");
    for (size_t q = 0; q < ser.U.size() && q < par.U.size(); ++q) check(ser.U[q] == par.U[q], "A^(" + std::to_string(q) + ")(U_ell) entrywise");
    for (size_t q = 0; q < ser.T.size() && q < par.T.size(); ++q) check(ser.T[q] == par.T[q], "A^(" + std::to_string(q) + ")(T_p) entrywise");
    check(ser.fp64_hits == par.fp64_hits && ser.fp64_exact == par.fp64_exact &&
              ser.fp32_hits == par.fp32_hits && ser.fp32_exact == par.fp32_exact,
          "this thread's PredicateStats deltas == the serial run's");
    std::cout << "  " << tag << ": dims";
    for (const Matrix& m : ser.U) std::cout << " " << m.rows;
    std::cout << ", fp64 hits " << ser.fp64_hits << " / exact " << ser.fp64_exact
              << (fails == 0 ? " -- concurrent == serial" : "") << std::endl;
    return fails;
}

uint32_t RunPair(const Job& a, const Job& b) {
    // serial references, on the main thread
    const FamilyRun sa = RunFamily(*a.stack, a.ell, a.n_sym, a.p);
    const FamilyRun sb = RunFamily(*b.stack, b.ell, b.n_sym, b.p);
    // concurrent: two threads released together
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    FamilyRun pa, pb;
    auto body = [&](const Job& j, FamilyRun& out) {
        ready.fetch_add(1);
        while (!go.load()) std::this_thread::yield();
        out = RunFamily(*j.stack, j.ell, j.n_sym, j.p);
    };
    std::thread ta(body, std::cref(a), std::ref(pa));
    std::thread tb(body, std::cref(b), std::ref(pb));
    while (ready.load() < 2) std::this_thread::yield();
    go.store(true);
    ta.join();
    tb.join();
    return Compare(sa, pa, a.tag) + Compare(sb, pb, b.tag);
}

} // namespace

uint32_t HeckeFamilyConcurrentBuildValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "=== HeckeFamilyConcurrentBuildValidation (slot-9 WP-7, L2: two family builds at once == serial) ==="
              << std::endl;
    uint32_t fails = 0;
    try {
        // (A) two independent cells.
        const Stack s1(10, 691), s2(12, 1009);
        fails += RunPair(Job{&s1, 691, 10, 2, "(A) cell (10, 691): U_691, T_2"},
                         Job{&s2, 1009, 12, 2, "(A) cell (12, 1009): U_1009, T_2"});
        // (B) two families on ONE K / model / cohomology.
        const Stack s3(8, 691);
        fails += RunPair(Job{&s3, 691, 8, 3, "(B) shared (8, 691), family 1: U_691, T_3"},
                         Job{&s3, 691, 8, 3, "(B) shared (8, 691), family 2: U_691, T_3"});
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++fails;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeFamilyConcurrentBuildValidation: " << fails << " failing, " << s << " s" << std::endl;
    return fails == 0 ? 0 : 1;
}
