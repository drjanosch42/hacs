// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 23, P4I -- HeckeIotaStreamValidation (see the header).

#include "HeckeIotaStreamValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)

#include "HeckeInvolution.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../../Engine/EngineRuntime.h"
#include "../../Engine/ThreadPool.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using hecke::HeckeInvolution;

uint32_t g_checks = 0;
uint32_t g_failures = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
}

// Everything the involution exposes, per degree.
struct Snapshot {
    bool ok = false;
    std::string error;
    std::vector<FlatMatrix> mats;   // per degree: iota_H, H+, H-, C+, C-, Z+, Z-, B+, B-
    std::vector<uint32_t> dims;
    uint32_t verify = 0;
    HeckeInvolution::StreamStats stats;
};

bool same(const FlatMatrix& a, const FlatMatrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols() || a.get_p() != b.get_p()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}

// First difference between two snapshots ("" when equal).
std::string diff(const Snapshot& a, const Snapshot& b) {
    if (a.ok != b.ok) return "one route threw";
    if (a.dims != b.dims) return "dims";
    if (a.verify != b.verify) return "certificate verify";
    if (a.mats.size() != b.mats.size()) return "object count";
    for (size_t i = 0; i < a.mats.size(); ++i)
        if (!same(a.mats[i], b.mats[i])) return "object " + std::to_string(i) + " (degree " + std::to_string(i / 9) +
                                                 ", slot " + std::to_string(i % 9) + ")";
    return "";
}

struct Cell {
    uint32_t n, ell;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::vector<SL2ZRelationDatum> data;
};

Snapshot run(const Cell& c, const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model) {
    Snapshot s;
    try {
        HeckeInvolution iota(c.coh.get(), hecke::BuildIotaSimplicialMapOnModel(K, model), "iota^* (P4I)");
        iota.set_exact_relation_data(c.data);
        iota.compute();
        s.stats = iota.stream_stats();
        s.verify = iota.certificate().verify();
        for (uint32_t q = 0; q <= iota.q_max(); ++q) {
            s.dims.push_back(iota.dim_H(q));
            s.mats.push_back(iota.matrix(q));
            s.mats.push_back(iota.H_plus_basis(q));
            s.mats.push_back(iota.H_minus_basis(q));
            s.mats.push_back(iota.C_eq_plus_basis(q));
            s.mats.push_back(iota.C_eq_minus_basis(q));
            s.mats.push_back(iota.Z_eq_plus_basis(q));
            s.mats.push_back(iota.Z_eq_minus_basis(q));
            s.mats.push_back(iota.B_eq_plus_basis(q));
            s.mats.push_back(iota.B_eq_minus_basis(q));
        }
        s.ok = true;
    } catch (const std::exception& e) {
        s.ok = false;
        s.error = e.what();
    }
    return s;
}

// (F) (audit F1) A thread-context installer that throws -- validation only.  Registered once, a no-op
// unless the CAPTURING thread armed it (thread-local, restored by ScopedInstallerFault): 1 = the first
// installer call of the armed scope throws, 2 = every call throws.  run_concurrently then skips that
// task (its error recorded); before the fix the admission's later tasks waited for it forever.
const char* const kFaultMsg = "P4I injected thread-context installer fault";
thread_local int g_fault_mode = 0;
std::atomic<uint32_t> g_fault_calls{0};
std::atomic<uint32_t> g_fault_thrown{0};
const bool g_register_fault = [] {
    engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
        const int mode = g_fault_mode;
        return [mode]() -> engine::tape::ThreadContextGuard {
            if (mode == 0) return nullptr;
            const uint32_t call = g_fault_calls.fetch_add(1);
            if (mode == 2 || call == 0u) {
                g_fault_thrown.fetch_add(1);
                throw std::runtime_error(kFaultMsg);
            }
            return nullptr;
        };
    });
    return true;
}();
struct ScopedInstallerFault {
    int saved;
    explicit ScopedInstallerFault(int mode) : saved(g_fault_mode) {
        g_fault_mode = mode;
        g_fault_calls = 0;
        g_fault_thrown = 0;
    }
    ~ScopedInstallerFault() { g_fault_mode = saved; }
};

// Ends the process with a FAIL line if the guarded scope does not return within `secs` (a hang).
class Watchdog {
    std::mutex m_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread t_;
public:
    Watchdog(const std::string& what, int secs)
        : t_([this, what, secs]() {
              std::unique_lock<std::mutex> lk(m_);
              if (!cv_.wait_for(lk, std::chrono::seconds(secs), [this] { return done_; })) {
                  std::cout << "  [FAIL] " << what << ": compute() did not return within " << secs << " s (hang)" << std::endl;
                  std::cout << "P4I_IOTA_STREAM verdict=FAIL (hang)" << std::endl;
                  std::_Exit(1);
              }
          }) {}
    ~Watchdog() {
        { std::lock_guard<std::mutex> g(m_); done_ = true; }
        cv_.notify_all();
        t_.join();
    }
};

} // namespace

uint32_t HeckeIotaStreamValidation() {
    g_checks = 0;
    g_failures = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "\n=== HeckeIotaStreamValidation (slot-23 P4I: the iota phase under a byte budget, streamed split) ===" << std::endl;

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    const std::vector<std::pair<uint32_t, uint32_t>> arms = {
        { 12, 13 }, { 24, 7 }, { 30, 13 }, { 40, 5 }, { 18, 3 }, { 20, 101 }, { 13, 7 }, { 12, 2 }, { 100, 13 },
    };
    uint64_t streamed_families = 0, admissions = 0;
    uint32_t mutations = 0, n_arms = 0, faults = 0;
    bool negative_done = false;

    for (const auto& arm : arms) {
        Cell c;
        c.n = arm.first;
        c.ell = arm.second;
        c.data = GenerateSL2ZRelationsWithGroupData(K, c.n, c.ell);
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& d : c.data) rels.push_back(d.rel);
        c.lssc.reset(new LocalSystemSimplicialComplex(K, rels, c.n + 1, c.ell));
        c.coh.reset(new EquivariantCohomologyOnK(c.lssc.get()));
        c.coh->compute_cohomology();
        ++n_arms;
        const std::string tag = "(n, ell) = (" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ")";
        const bool split = (c.ell != 2u);
        const bool families = split && (c.n % 2u == 0u);   // odd n: C_eq = 0 on this model, nothing to split

        Snapshot ref;
        {
            HeckeInvolution::ScopedIotaStreaming off(false);
            ref = run(c, K, model);
        }
        Check(ref.ok, tag + ": the materialised route computes (" + ref.error + ")");
        Check(!ref.stats.streamed && ref.stats.tasks_iii == 0u && ref.stats.tasks_split == 0u,
              tag + ": the materialised route reports no streaming");
        if (!ref.ok) continue;

        // (X) / (V) over the admission budgets.
        const uint64_t kMax = std::numeric_limits<uint64_t>::max();
        const uint64_t budgets[] = { 0u, 1u, 40000u, kMax };
        for (uint64_t b : budgets) {
            HeckeInvolution::ScopedIotaStreaming on(true);
            HeckeInvolution::ScopedIotaTaskBudget tb(b);
            const Snapshot s = run(c, K, model);
            const std::string bt = tag + " budget " + (b == kMax ? std::string("max") : std::to_string(b));
            Check(s.ok, bt + ": the streamed route computes (" + s.error + ")");
            if (!s.ok) continue;
            const std::string d = diff(ref, s);
            Check(d.empty(), bt + ": streamed == materialised, entrywise (first difference: " + d + ")");
            Check(s.verify == 0u, bt + ": certificate verifies");
            Check(s.stats.streamed, bt + ": streamed");
            Check(!families || s.stats.streamed_families > 0u, bt + ": streamed families > 0");
            Check(families || s.stats.streamed_families == 0u, bt + ": no flat split (ell = 2 or odd n)");
            Check(s.stats.mutations_applied == 0u, bt + ": no mutation applied");
            streamed_families += s.stats.streamed_families;
            admissions += s.stats.tasks_iii + s.stats.tasks_split;
            const auto& st = s.stats;
            const uint64_t want_iii = (b == 0u) ? 2u * std::max<uint64_t>(st.max_task_iii, 1u) : b;
            const uint64_t want_split = (b == 0u) ? 2u * std::max<uint64_t>(st.max_task_split, 1u) : b;
            Check(st.budget_iii == want_iii && (!split || st.budget_split == want_split), bt + ": the budget in force");
            // The bound: the admitted estimate never exceeds max(budget, the largest task).
            Check(st.peak_admitted_iii <= std::max(st.budget_iii, st.max_task_iii), bt + ": obl(iii) admitted <= budget");
            Check(st.peak_admitted_split <= std::max(st.budget_split, st.max_task_split), bt + ": split admitted <= budget");
            Check(st.max_concurrent_iii >= 1u && (!split || st.max_concurrent_split >= 1u), bt + ": tasks admitted");
            if (b == 1u)
                Check(st.max_concurrent_iii == 1u && st.max_concurrent_split <= 1u, bt + ": one task at a time");
            if (b == 40000u && families)
                Check(st.max_task_split > 40000u && st.peak_admitted_split == st.max_task_split,
                      bt + ": a task above the budget ran alone (split, peak " + std::to_string(st.peak_admitted_split) + ")");
        }

        // (S) the serial path: a one-worker consumer pool (run_concurrently runs the tasks in order on the
        // calling thread; every admission finds nothing running).
        {
            engine::tape::ThreadPool pool(1);
            engine::tape::ScopedConsumerPool sp(&pool);
            HeckeInvolution::ScopedIotaStreaming on(true);
            const Snapshot s = run(c, K, model);
            Check(s.ok && diff(ref, s).empty(), tag + ": serial pool, streamed == materialised (" +
                                                    (s.ok ? diff(ref, s) : s.error) + ")");
            Check(!s.ok || s.stats.max_concurrent_iii == 1u, tag + ": serial pool, one task at a time");
        }

        // (L) the legacy elimination on a one-worker consumer pool (run_concurrently is serial there,
        // so the thread-local legacy flag reaches every task).  (S24 CL-L2) ON builds only.
        if ((c.n == 24u || c.n == 12u) && oc::legacy::RunLegacyArm("HeckeIotaStreamValidation", "L")) {
#if OC_ENABLE_LEGACY
            engine::tape::ThreadPool pool(1);
            engine::tape::ScopedConsumerPool sp(&pool);
            engine::elim::ScopedLegacy legacy;
            HeckeInvolution::ScopedIotaStreaming on(true);
            const Snapshot s = run(c, K, model);
            Check(s.ok && diff(ref, s).empty(), tag + ": legacy elimination, streamed == materialised (" +
                                                    (s.ok ? diff(ref, s) : s.error) + ")");
#endif
        }

        // (M) mutations on the split arms.
        if (split && (c.n == 24u || c.n == 30u || c.n == 20u)) {
            for (int kind = 1; kind <= 3; ++kind) {
                const std::vector<uint64_t> mb = { 0u, 40000u };
                for (uint64_t b : mb) {
                    HeckeInvolution::ScopedIotaStreaming on(true);
                    HeckeInvolution::ScopedIotaTaskBudget tb(b);
                    HeckeInvolution::ScopedIotaStreamMutation mut(kind);
                    const Snapshot s = run(c, K, model);
                    const std::string mt = tag + " mutation " + std::to_string(kind) + " budget " +
                                           (b == std::numeric_limits<uint64_t>::max() ? std::string("max") : std::to_string(b));
                    const bool caught = !s.ok || !diff(ref, s).empty();
                    Check(caught, mt + ": caught");
                    Check(!s.ok || s.stats.mutations_applied > 0u, mt + ": applied");
                    ++mutations;
                }
                // The materialised route ignores the mutation hook.
                HeckeInvolution::ScopedIotaStreaming off(false);
                HeckeInvolution::ScopedIotaStreamMutation mut(kind);
                const Snapshot s = run(c, K, model);
                Check(s.ok && diff(ref, s).empty(), tag + " mutation " + std::to_string(kind) + ": ignored when materialised");
            }
        }

        // (F) a throwing thread-context installer: no hang, the injected error surfaces (audit F1).
        if (c.n == 24u) {
            if (!engine::tape::host_fanout_enabled()) {
                std::cout << "  (F) skipped: ENGINE_HOST_THREADS=0 (run_concurrently is serial, installers never run)" << std::endl;
            } else {
                engine::tape::ThreadPool pool(4);
                engine::tape::ScopedConsumerPool sp(&pool);
                for (int streamed = 1; streamed >= 0; --streamed) {
                    for (int mode = 1; mode <= 2; ++mode) {
                        const std::string ft = tag + " installer fault " + (mode == 1 ? "once" : "always") +
                                               (streamed ? ", streamed" : ", materialised");
                        HeckeInvolution::ScopedIotaStreaming st(streamed != 0);
                        Snapshot s;
                        {
                            Watchdog wd(ft, 300);
                            ScopedInstallerFault fault(mode);
                            s = run(c, K, model);
                        }
                        Check(g_fault_thrown.load() > 0u, ft + ": the fault was injected (" +
                                                              std::to_string(g_fault_calls.load()) + " installer calls)");
                        Check(!s.ok && s.error == kFaultMsg, ft + ": the injected error surfaces (got: " +
                                                                 (s.ok ? std::string("no error") : s.error) + ")");
                        ++faults;
                    }
                }
                HeckeInvolution::ScopedIotaStreaming on(true);
                const Snapshot s = run(c, K, model);
                Check(s.ok && diff(ref, s).empty(), tag + ": after the faults, streamed == materialised (" +
                                                        (s.ok ? diff(ref, s) : s.error) + ")");
            }
        }

        // (N) the comparator flags a flipped entry.
        if (!negative_done && split) {
            Snapshot s = ref;
            for (FlatMatrix& m : s.mats) {
                if (m.get_rows() > 0u && m.get_cols() > 0u) {
                    m.set_entry(0, 0, (m.get_entry(0, 0) + 1u) % c.ell);
                    break;
                }
            }
            Check(!diff(ref, s).empty(), tag + ": negative control flagged");
            negative_done = true;
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "P4I_IOTA_STREAM arms=" << n_arms << " checks=" << g_checks << " failing=" << g_failures
              << " streamed_families=" << streamed_families << " admissions=" << admissions << " mutations=" << mutations
              << " faults=" << faults << " verdict=" << (g_failures == 0u ? "OK" : "FAIL") << " " << std::fixed << std::setprecision(1) << secs
              << " s" << std::endl;
    return g_failures == 0u ? 0u : 1u;
}
