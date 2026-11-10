// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Phase 0 -- in-repo benchmark CLI. See BenchDriver.h for the placement /
// design rationale and docs/matrix-engine/03-profiling-baseline.md Sec. 1.3
// for the scratch driver this productises.
// ===========================================================================

#include "BenchDriver.h"
#include "../Util/Rss.h"
#include "../../Engine/TapeElim.h"
#include "../../Engine/EngineTrace.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/LegacyConfig.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../EquivariantCohomologyOnK.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeCrossChecks.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../HeckeExt/HeckeExtDriver.h"
// (hecke-extensions slot 5, WP-7 / E6) The two report forms of [E]
// sec:field, so that --bench ext can TIME them.  No record consumes
// them in this slot; see RunExt.
#include "../HeckeExt/HeckeCanonicalField.h"
#include "../HeckeExt/HeckeLocalAlgebra.h"
// (hecke-extensions slot 7, WP-6) --bench ext --campaign-cell / --u-ell:
// ExtDriverOptions::with_cone / with_sign_split are already reachable through
// HeckeExtDriver.h; this is for hecke::ParseUEll / hecke::ScopedUEll alone.
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/PerformanceSmokeValidation.h"
#include "../../ModArithmetic/ObjectCounters.h"

#include <cstddef>
#include <new>

// ===========================================================================
// (matrix-engine Phase 3) Counting replacement global operator new / delete.
//
// WHY HERE. Phase 3's exit criterion is stated against an allocation baseline
// ("malloc count <= 5 % of today's 24 M", 00-implementation-plan.md Phase 3,
// from 03-profiling-baseline.md Sec. 6), and that 24 M was measured with an
// external profiler on a scratch driver. `--bench <mode> --count-allocs` makes
// the number reproducible from the shipped binary with no tooling: the driver
// snapshots the counters around the measured region and prints the delta.
//
// A replacement global operator new is program-wide, so ALL of the process's
// C++ allocations are counted, not just the bench TU's -- which is exactly what
// is wanted. The counters are plain std::size_t, no atomics: the pipeline is
// single-threaded by project convention (04 Sec. 7.5's thread pool is Phase 4),
// and the cost of two increments is far below one malloc, so leaving them
// always-on keeps `--count-allocs` available in the default build.
//
// Every replaceable form is provided, aligned and nothrow variants included:
// replacing only some of them would pair one library's allocation with the
// other library's deallocation. Bytes are accumulated at allocation time only
// (not every delete carries a size), so `alloc_bytes` is REQUESTED bytes, not
// a high-water mark -- peak RSS on the same line is the memory figure.
// ===========================================================================
namespace {
// Relaxed atomics (Phase 4): the pipeline allocates from several host threads;
// the counters are diagnostics, never on a result path.
std::atomic<std::size_t> g_alloc_count{0};
std::atomic<std::size_t> g_alloc_bytes{0};

void* CountedAlloc(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    // malloc(0) may return nullptr, which operator new must never do.
    return std::malloc(n ? n : 1);
}

void* CountedAllocAligned(std::size_t n, std::size_t align) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    if (align < sizeof(void*)) align = sizeof(void*);
    void* q = nullptr;
    if (posix_memalign(&q, align, n ? n : 1) != 0) return nullptr;
    return q;
}
} // namespace

void* operator new(std::size_t n) {
    void* q = CountedAlloc(n);
    if (!q) throw std::bad_alloc();
    return q;
}
void* operator new[](std::size_t n) {
    void* q = CountedAlloc(n);
    if (!q) throw std::bad_alloc();
    return q;
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return CountedAlloc(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return CountedAlloc(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    void* q = CountedAllocAligned(n, static_cast<std::size_t>(a));
    if (!q) throw std::bad_alloc();
    return q;
}
void* operator new[](std::size_t n, std::align_val_t a) {
    void* q = CountedAllocAligned(n, static_cast<std::size_t>(a));
    if (!q) throw std::bad_alloc();
    return q;
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    return CountedAllocAligned(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    return CountedAllocAligned(n, static_cast<std::size_t>(a));
}

void operator delete(void* q) noexcept { std::free(q); }
void operator delete[](void* q) noexcept { std::free(q); }
void operator delete(void* q, std::size_t) noexcept { std::free(q); }
void operator delete[](void* q, std::size_t) noexcept { std::free(q); }
void operator delete(void* q, const std::nothrow_t&) noexcept { std::free(q); }
void operator delete[](void* q, const std::nothrow_t&) noexcept { std::free(q); }
void operator delete(void* q, std::align_val_t) noexcept { std::free(q); }
void operator delete[](void* q, std::align_val_t) noexcept { std::free(q); }
void operator delete(void* q, std::size_t, std::align_val_t) noexcept { std::free(q); }
void operator delete[](void* q, std::size_t, std::align_val_t) noexcept { std::free(q); }
void operator delete(void* q, std::align_val_t, const std::nothrow_t&) noexcept { std::free(q); }
void operator delete[](void* q, std::align_val_t, const std::nothrow_t&) noexcept { std::free(q); }

namespace bench {
namespace {

// Snapshot of the process-wide allocation counters, so the driver can report
// the delta over one measured repeat.
struct AllocSnapshot {
    std::size_t count = 0;
    std::size_t bytes = 0;
    uint64_t flat_ctors = 0;
    uint64_t block_ctors = 0;
};

AllocSnapshot TakeAllocSnapshot() {
    AllocSnapshot s;
    s.count = g_alloc_count;
    s.bytes = g_alloc_bytes;
    s.flat_ctors = engine::counters::FlatMatrixCtors();
    s.block_ctors = engine::counters::BlockFlatMatrixCtors();
    return s;
}

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

double secs(TimePoint a, TimePoint b) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

// Peak / current resident set size in MiB, from /proc/self/status (VmHWM is
// the kernel's "high water mark" for the whole process lifetime -- so it is the
// peak across ALL repeats, not just the last one).  -1.0 where /proc is
// unavailable (non-Linux), and the caller prints "n/a".  (slot-7 WP-8, gap
// G4b) The one definition is Util/Rss.h, shared with the campaign driver and
// the console.
using oc_util::PeakRssMiB;
using oc_util::CurrentRssMiB;

} // namespace

std::string FormatBenchLine(const std::string& mode, const std::string& params, double total_s,
                            double rss_mb, const std::string& extra) {
    std::ostringstream os;
    os << "BENCH mode=" << mode;
    if (!params.empty()) os << " " << params;
    os << " total_s=" << std::fixed << std::setprecision(4) << total_s << " rss_mb=";
    if (rss_mb < 0.0) os << "n/a";
    else os << std::fixed << std::setprecision(1) << rss_mb;
    if (!extra.empty()) os << " " << extra;
    return os.str();
}

std::size_t ProcessAllocCount() { return g_alloc_count.load(std::memory_order_relaxed); }
std::size_t ProcessAllocBytes() { return g_alloc_bytes.load(std::memory_order_relaxed); }

namespace {

// ---------------------------------------------------------------------------
// Phase-split timing record. One instance per repeat; the driver prints the
// per-repeat breakdown and, for --repeat k > 1, min/median of `total`.
// ---------------------------------------------------------------------------
struct PhaseTimes {
    // (name, seconds) in execution order. Kept as a vector rather than named
    // fields so each mode contributes only the phases it actually runs.
    std::vector<std::pair<std::string, double>> phases;
    double total = 0.0;

    void add(const char* name, double s) {
        phases.emplace_back(name, s);
        // ENGINE_TRACE_RSS=1: the resident set after every phase (memory
        // profile 05 Sec. 8), so a peak can be attributed to a phase.
        if (std::getenv("ENGINE_TRACE_RSS") != nullptr)
            std::cout << "    [rss] after " << name << ": VmRSS=" << std::fixed << std::setprecision(1) << CurrentRssMiB()
                      << " MB VmHWM=" << PeakRssMiB() << " MB\n";
    }
};

void PrintPhases(const PhaseTimes& t, uint32_t repeat_index, uint32_t repeats) {
    if (repeats > 1) std::cout << "  --- repeat " << (repeat_index + 1) << "/" << repeats << " ---\n";
    for (const auto& ph : t.phases) {
        std::cout << "    " << std::left << std::setw(34) << ph.first << std::right
                  << std::fixed << std::setprecision(4) << std::setw(12) << ph.second << " s\n";
    }
    std::cout << "    " << std::left << std::setw(34) << "TOTAL" << std::right
              << std::fixed << std::setprecision(4) << std::setw(12) << t.total << " s\n";
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2 == 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// The BENCH line's mode-specific tail ("dims=4,20,0 xcheck_rc=skipped ..."),
// split into its key=value pairs for BenchRecord::extra.  A token without '='
// keeps an empty value.
std::vector<std::pair<std::string, std::string>> SplitKeyValues(const std::string& text) {
    std::vector<std::pair<std::string, std::string>> kv;
    std::istringstream is(text);
    std::string tok;
    while (is >> tok) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) kv.emplace_back(tok, std::string());
        else kv.emplace_back(tok.substr(0, eq), tok.substr(eq + 1));
    }
    return kv;
}

// The single machine-readable summary line, so a sweep script can grep one
// pattern regardless of mode. Extra key=value pairs are appended by the mode.
//
// (slot-7 WP-8, G4) The line is formatted ONCE, into a string, which is both
// printed and returned in `rec` -- so the console's `bench_line(record)` prints
// exactly the bytes `--bench` prints, by construction rather than by a second
// formatter.  The formatting is the pre-extraction one, manipulator for
// manipulator; afterwards std::cout is left in the format state the old
// piecewise print left it in (std::fixed, precision 1 -- or 4 when rss is
// n/a), so no later print in the same process can change its rendering.
std::string PrintSummary(const std::string& mode, const std::string& params,
                         double total_s, const std::string& extra,
                         BenchRecord* rec = nullptr) {
    const double rss = PeakRssMiB();
    const std::string line = FormatBenchLine(mode, params, total_s, rss, extra);
    std::cout << line << std::endl;
    std::cout << std::fixed << std::setprecision(rss < 0.0 ? 4 : 1);
    if (rec != nullptr) {
        rec->bench_line = line;
        rec->total_s = total_s;
        rec->rss_mb = rss;
        rec->extra = SplitKeyValues(extra);
    }
    return line;
}

std::string DimsString(const std::vector<uint32_t>& dims) {
    std::ostringstream os;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ",";
        os << dims[i];
    }
    return os.str();
}

// ---------------------------------------------------------------------------
// Shared pipeline pieces. Identical to the recipe in every V-stage validation
// (see e.g. Hecke/HeckeEndToEndValidation.cpp's BuildRelations /
// CohomologyFixture / HeckeFixture) -- deliberately duplicated here rather
// than exported from a validation TU, so the bench driver depends on the
// public API only.
// ---------------------------------------------------------------------------
std::vector<LocalSystemSimplicialComplex::SimplexRelation> BuildRelations(
        const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    return rels;
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

// `cohom <n> <ell>`: EquivariantCohomologyOnK on the SL(2,Z) model, all q.
PhaseTimes RunCohom(uint32_t n_sym, uint32_t ell, std::vector<uint32_t>& dims_out,
                    std::vector<uint32_t>* c_eq_dims_out = nullptr) {
    PhaseTimes t;
    const TimePoint t0 = Clock::now();

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    auto rels = BuildRelations(K, n_sym, ell);
    LocalSystemSimplicialComplex lssc(K, std::move(rels), n_sym + 1, ell);   // adopted, not copied (05 Sec. 7 item 4)
    const TimePoint t1 = Clock::now();
    t.add("setup (K + rho_n relations)", secs(t0, t1));

    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    const TimePoint t2 = Clock::now();
    // compute_cohomology() computes ALL degrees q in one call, so a per-q
    // split is not observable through the public API; it is reported as one
    // phase and the per-q dimensions are printed alongside.
    t.add("cohomology (all q)", secs(t1, t2));

    dims_out = coh.get_H_eq_dimensions();
    // dim C^q_eq = dim ker(A_Cq); reported because rank(A_Cq) = cols - dim C^q_eq
    // is the rank of the real matrix that `micro kernel` mimics, and doc 03
    // Sec. 2 tabulates that matrix's SHAPE but not its rank.
    if (c_eq_dims_out) {
        c_eq_dims_out->clear();
        for (uint32_t q = 0; q < dims_out.size(); ++q)
            c_eq_dims_out->push_back(coh.C_eq_dimension(q));   // no _fm cache, no MatrixT copy; valid in lean mode
    }
    t.total = secs(t0, t2);
    return t;
}

// `hecke <n> <ell> <p>`: T_p end to end (cosets -> tau -> Theta_g -> A^{(q)}),
// plus the classical three-map alpha/beta/omega cross-check route.
PhaseTimes RunHecke(uint32_t n_sym, uint32_t ell, uint64_t p_hecke, bool cross_checks,
                    std::vector<uint32_t>& dims_out, uint32_t& xcheck_rc) {
    PhaseTimes t;
    const TimePoint t0 = Clock::now();

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    auto rels = BuildRelations(K, n_sym, ell);
    LocalSystemSimplicialComplex lssc(K, std::move(rels), n_sym + 1, ell);   // adopted, not copied (05 Sec. 7 item 4)
    const TimePoint t1 = Clock::now();
    t.add("setup (K + model + relations)", secs(t0, t1));

    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    const TimePoint t2 = Clock::now();
    t.add("cohomology (all q)", secs(t1, t2));
    dims_out = coh.get_H_eq_dimensions();

    hecke::HeckeCosetTables cosets(p_hecke, /*r=*/1, ell, n_sym);
    const TimePoint t3 = Clock::now();
    t.add("coset tables", secs(t2, t3));

    hecke::HeckeTauTable tau(K, model, cosets, ell, n_sym); // ctor certifies (obl:tau)
    const TimePoint t4 = Clock::now();
    t.add("tau table (+ exact certification)", secs(t3, t4));

    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute();
    const TimePoint t5 = Clock::now();
    // compute() runs level (a) (Theta_flat / Theta_eq), level (b) (the
    // cohomology-level A^{(q)}), the runtime asserts (i)/(ii)/(iii) and the
    // O12(a) two-route bit-identity in one call; the public API exposes no
    // finer split, so it is one phase.
    t.add("operator assembly (Theta + A^(q))", secs(t4, t5));

    // Cohomology-level extraction is O(1) accessor work on top of compute();
    // timed separately so the summary line still carries the phase name the
    // plan asks for, and so a future split of compute() has a slot to land in.
    const TimePoint t5b = Clock::now();
    const auto& A = op.get_H_matrices();
    std::ostringstream shapes;
    for (size_t q = 0; q < A.size(); ++q) {
        if (q) shapes << " ";
        shapes << "A^(" << q << ")=" << A[q].rows << "x" << A[q].cols;
    }
    const TimePoint t6 = Clock::now();
    t.add("cohomology-level extraction", secs(t5b, t6));

    xcheck_rc = 0;
    TimePoint t7 = t6;
    if (cross_checks) {
        hecke::HeckeCrossChecks xc(&coh, &cosets, &tau);
        xcheck_rc = xc.run_all(op);
        t7 = Clock::now();
        t.add("cross-checks (alpha/beta/omega)", secs(t6, t7));
    }

    std::cout << "    [" << shapes.str() << "]\n";
    t.total = secs(t0, t7);
    return t;
}

// `family <n> <ell> <p> <r>`: T_{p^r} by the algebra recursion AND by the
// independent direct double-coset route.
PhaseTimes RunFamily(uint32_t n_sym, uint32_t ell, uint64_t p_hecke, uint32_t r,
                     std::vector<uint32_t>& dims_out, bool& agree_out) {
    PhaseTimes t;
    const TimePoint t0 = Clock::now();

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    auto rels = BuildRelations(K, n_sym, ell);
    LocalSystemSimplicialComplex lssc(K, std::move(rels), n_sym + 1, ell);   // adopted, not copied (05 Sec. 7 item 4)
    const TimePoint t1 = Clock::now();
    t.add("setup (K + model + relations)", secs(t0, t1));

    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    const TimePoint t2 = Clock::now();
    t.add("cohomology (all q)", secs(t1, t2));
    dims_out = coh.get_H_eq_dimensions();

    hecke::HeckeOperatorFamily family(&coh, K, model, ell, n_sym);
    const TimePoint t3 = Clock::now();
    t.add("family construction", secs(t2, t3));

    // (matrix-engine Phase 4, axis (d)) Every pipeline the two routes need --
    // (p, 1) for T_p and the recursion, (p, r - 2j) for the direct route --
    // built concurrently by the family (tau construction serial, compute()
    // per pipeline on the pool).  T_p and the direct route below are then
    // cache hits; the phase carries the whole pipeline cost.
    family.prepare(p_hecke, r);
    const TimePoint t3b = Clock::now();
    t.add("pipelines (p,1)+(p,r-2j) concurrent", secs(t3, t3b));

    // T_p (a cache hit after prepare(); the recursion phase below measures
    // pure linear algebra).
    (void)family.T_p(p_hecke);
    const TimePoint t4 = Clock::now();
    t.add("T_p (full pipeline, cached)", secs(t3b, t4));

    const std::vector<MatrixT<uint32_t>> rec = family.T_p_pow(p_hecke, r);
    const TimePoint t5 = Clock::now();
    t.add("T_{p^r} recursion", secs(t4, t5));

    const std::vector<MatrixT<uint32_t>> dir = family.T_p_pow_direct(p_hecke, r);
    const TimePoint t6 = Clock::now();
    t.add("T_{p^r} direct (double coset)", secs(t5, t6));

    agree_out = (rec.size() == dir.size());
    if (agree_out) {
        for (size_t q = 0; q < rec.size() && agree_out; ++q) {
            if (rec[q].rows != dir[q].rows || rec[q].cols != dir[q].cols) { agree_out = false; break; }
            if (!(rec[q] == dir[q])) agree_out = false;
        }
    }
    const TimePoint t7 = Clock::now();
    t.add("recursion-vs-direct comparison", secs(t6, t7));

    t.total = secs(t0, t7);
    return t;
}

// `decomp <n> <ell> <p>`: module decomposition of H^q over the F_ell-algebra
// generated by A^{(q)}(T_p), for the largest non-empty degree.
PhaseTimes RunDecomp(uint32_t n_sym, uint32_t ell, uint64_t p_hecke,
                     std::vector<uint32_t>& dims_out, uint32_t& blocks_out,
                     uint32_t& cert_out, uint32_t& q_used_out) {
    PhaseTimes t;
    const TimePoint t0 = Clock::now();

    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    auto rels = BuildRelations(K, n_sym, ell);
    LocalSystemSimplicialComplex lssc(K, std::move(rels), n_sym + 1, ell);   // adopted, not copied (05 Sec. 7 item 4)
    const TimePoint t1 = Clock::now();
    t.add("setup (K + model + relations)", secs(t0, t1));

    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    const TimePoint t2 = Clock::now();
    t.add("cohomology (all q)", secs(t1, t2));
    dims_out = coh.get_H_eq_dimensions();

    hecke::HeckeCosetTables cosets(p_hecke, /*r=*/1, ell, n_sym);
    hecke::HeckeTauTable tau(K, model, cosets, ell, n_sym);
    hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
    op.compute();
    const TimePoint t3 = Clock::now();
    t.add("operator assembly (Theta + A^(q))", secs(t2, t3));

    // Decompose the largest non-empty degree (q = 1 on this model for every
    // n_sym > 0; q = 0 for n_sym = 0). Chosen by dimension, not hard-coded.
    const auto& A = op.get_H_matrices();
    uint32_t q_best = 0;
    for (uint32_t q = 0; q < A.size(); ++q) {
        if (A[q].rows > A[q_best].rows) q_best = q;
    }
    q_used_out = q_best;

    blocks_out = 0;
    cert_out = 0;
    if (A[q_best].rows == 0) {
        std::cout << "    [all degrees have dim 0 -- nothing to decompose]\n";
        t.add("module decomposition + certify", 0.0);
        t.total = secs(t0, t3);
        return t;
    }

    std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops = {{p_hecke, A[q_best]}};
    hecke::HeckeModuleDecomposition decomp(std::move(ops), ell);
    decomp.compute();
    cert_out = decomp.certify();
    blocks_out = static_cast<uint32_t>(decomp.blocks().size());
    const TimePoint t4 = Clock::now();
    t.add("module decomposition + certify", secs(t3, t4));

    t.total = secs(t0, t4);
    return t;
}

// ---------------------------------------------------------------------------
// `ext <n> <ell> [family]` -- (hecke-extensions slot 4, WP-2 / E2b) the whole
// extension layer on one cell, through HeckeExt/HeckeExtDriver's public API.
//
// WHAT IT MEASURES, and why the split is exactly this one.  [X] Sec. 6.2 E1
// acceptance (d) says the ext layer must add at most 20 % to the wall; the
// slot-4 plan Sec. 2.3 item 2 fixes the baseline as "the wall of the same run
// WITHOUT the ext layer", i.e. the pipeline for the family plus the Stage-2
// decomposition a campaign job computes anyway.  So the gate is
//
//     ext_s <= 0.20 * (pipeline_s + decomp_s)
//
// with each phase's MIN over the repeats and the ratio formed from those mins.
// `ext_over_hecke1_ratio` is the plan's OWN (literal) baseline, a one-prime
// `hecke` run of the same (n, ell): it measures the FAMILY's size, not the
// ext layer's cost, and is printed as an informational column only.
//
// (hecke-extensions slot 5, WP-7 / E6; slot 6, WP-7 / [OD] E6-1) TWO MORE
// COLUMNS, `canonical_field_s=` and `own_field_s=`, are the measurement E6's
// own performance gate is stated over ([X] Sec. 6.2 E6 (d), Sec. 2.2 Stage 6,
// [E] sec:field): the RESTATED gate is
//
//     canonical_field_s <= 0.10 * total_s
//
// with `total_s` the wall of `HeckeExtDriver::run()` -- i.e. the SAME figure
// the generic BENCH line prints as `total_s=` (`best`, this function's `t.total`)
// -- NOT `ext_s` (Stages 3-5 alone).  The pre-slot-6 `<= 2 * own_field_s` half
// is DELETED: its denominator was tier (i)'s rendering, which [E] sec:field
// calls "zero new arithmetic", so it compared certified root finding with a
// `sprintf` and was not meetable by any admissible implementation
// (`docs/implementation/e6_gate_brief.md` Sec. 2.1, 6.1).  `own_field_s` is
// still measured and printed beside it as a reported (not bounded) column.
// THIS PACKAGE MEASURES AND REPORTS THE RESTATED GATE AND DOES NOT ACT ON IT
// -- E6 acceptance is slot 7's.  Both `canonical_field_s` and `own_field_s`
// are the cost of PRODUCING the respective report form for every
// certified-local ideal of the record: tier (ii) is the root finding of
// alg:split plus the rendering, tier (i) is the rendering alone.  Nothing
// here enters the record, and nothing here is hashed.
// ---------------------------------------------------------------------------
PhaseTimes RunExt(uint32_t n_sym, uint32_t ell, const std::vector<uint64_t>& family,
                  hecke_ext::ExtDriverTimings& et_out,
                  std::vector<uint32_t>& dims_out, uint32_t& ideals_out,
                  uint32_t& streamed_out, double& streamed_rss_out,
                  double& hecke1_s_out,
                  double& canonical_field_s_out, double& own_field_s_out,
                  uint32_t& canonical_field_ideals_out, uint32_t& canonical_field_ok_out,
                  uint32_t& canonical_field_r_max_out,
                  // (slot-7 WP-6) both false = byte-identical with every
                  // pre-WP-6 call site (all four positional-default callers
                  // below keep the two-argument form via the default here).
                  bool with_cone = false, bool with_sign_split = false) {
    PhaseTimes t;

    hecke_ext::ExtDriverOptions opts;
    opts.good_primes = family;
    opts.with_cone = with_cone;
    opts.with_sign_split = with_sign_split;
    hecke_ext::HeckeExtDriver d(n_sym, ell, opts);
    d.run();
    const hecke_ext::ExtDriverTimings& et = d.timings();
    et_out = et;
    dims_out = d.H_dimensions();
    ideals_out = static_cast<uint32_t>(d.record().ideals.size());
    streamed_out = d.streamed().built;
    streamed_rss_out = d.streamed().peak_rss_mb;

    t.add("pipeline (cohomology + S_0 ops)", et.pipeline_s());
    t.add("  of which setup (K + rho_n + H)", et.setup_s);
    t.add("  of which Stage 1 (T_p on S_0)", et.stage1_s);
    t.add("decomposition (Stage 2)", et.decomp_s());
    t.add("ext layer (Stages 3-5)", et.ext_s());
    t.add("  of which Stage 3 (A_S)", et.stage3_s);
    t.add("  of which Stage 4 (certificates)", et.stage4_s);
    t.add("  of which Sturm bound", et.sturm_bound_s);
    t.add("  of which streamed T_q", et.streamed_s);
    t.add("  of which Stage 5 (module)", et.stage5_s);
    // (slot-9 WP-7, L5) The campaign cell's three further phases, which the
    // driver already times: printed ONLY under `--campaign-cell` (with_cone /
    // with_sign_split), so every other `--bench ext` output is byte-unchanged
    // (bench_identity.sh).  proof_range_s is inside Stage 4; cone_s and
    // iota_s are outside Stages 1-5 but inside total_s.
    if (with_cone || with_sign_split) {
        t.add("  of which proof_range (Stage 4)", et.proof_range_s);
        t.add("cone (pair + cone)", et.cone_s);
        t.add("sign split (iota + per-sign)", et.iota_s);
    }

    // (slot-5 WP-7 / E6) The two report forms, timed.  See the header comment.
    // `sink` keeps the optimiser from eliding the rendering.
    //
    // (slot-7 WP-3, stage 2 item 6; [S7] Sec. 2.4 T3) The canonical split is
    // now the record's default report form and runs INSIDE run(): its time is
    // the driver's own `canonical_field_s` (the same root finding + rendering
    // of every certified-local ideal of the unsplit record that this function
    // used to recompute here after run()), so nothing is counted twice and
    // `canonical_field_s` measures what it measured before.  `total_s` now
    // contains it; the E6 (d) denominator is `total_s - canonical_field_s`.
    {
        volatile size_t sink = 0;
        canonical_field_ideals_out = 0;
        canonical_field_ok_out = 0;
        canonical_field_r_max_out = 0;
        const uint32_t n_ideals = static_cast<uint32_t>(d.record().ideals.size());
        for (uint32_t i = 0; i < n_ideals; ++i) {
            const hecke_ext::IdealRecord& id = d.record().ideals[i];
            if (!id.split_present) continue;
            ++canonical_field_ideals_out;
            if (id.split_ok) ++canonical_field_ok_out;
            if (id.split_r > canonical_field_r_max_out) canonical_field_r_max_out = id.split_r;
        }
        canonical_field_s_out = et.canonical_field_s;
        const TimePoint c1 = Clock::now();
        for (uint32_t i = 0; i < n_ideals; ++i) {
            const hecke_ext::LocalAlgebra& la = d.local_algebra(i);
            if (!la.ok || la.witness_poly.is_zero()) continue;
            sink += hecke_ext::OwnFieldReportJson(la.ell, la.N, la.witness_poly, la.primes,
                                                  la.character).size();
        }
        const TimePoint c2 = Clock::now();
        own_field_s_out = secs(c1, c2);
        (void)sink;
    }
    t.add("  [E6 report forms] canonical field", canonical_field_s_out);
    t.add("  [E6 report forms] own field", own_field_s_out);

    t.total = et.total_s;

    // The informational one-prime baseline.
    hecke1_s_out = 0.0;
    if (!family.empty()) {
        std::vector<uint32_t> hdims;
        uint32_t rc = 0;
        const PhaseTimes ht = RunHecke(n_sym, ell, family[0], false, hdims, rc);
        hecke1_s_out = ht.total;
    }
    return t;
}

// ---------------------------------------------------------------------------
// `micro <op> <args>` -- Phase-2 arithmetic micro-benchmarks.
//
// The measurement code itself lives in
// src/ModArithmetic/PerformanceSmokeValidation.{h,cpp} (see the
// header for why: a validation registered in the ModArithmetic static library
// cannot call back into this translation unit, which belongs to
// cohomology_app). This section is CLI only -- parsing, printing and the
// exit-criterion suite table.
// ---------------------------------------------------------------------------

void PrintMicroResult(const micro::MicroResult& r, micro::Backend backend, uint32_t repeats,
                      BenchRecord* rec = nullptr) {
    std::cout << "    " << std::left << std::setw(12) << r.op << std::right
              << std::fixed << std::setprecision(6) << std::setw(12) << r.min_s << " s (min)";
    if (repeats > 1) std::cout << "  " << std::setw(12) << r.median_s << " s (median)";
    std::cout << "\n";
    if (r.work_modmul > 0.0) {
        std::cout << "      rate        = " << std::fixed << std::setprecision(3)
                  << r.gmodmul_per_s() << " Gmodmul/s  (" << std::setprecision(4)
                  << r.work_modmul / 1e9 << " Gmodmul of work)\n";
    }
    if (r.op == "rref" || r.op == "kernel") {
        std::cout << "      rank        = " << r.rank;
        if (r.op == "kernel") std::cout << "   nullity = " << r.nullity;
        std::cout << "\n";
    }
    if (!r.note.empty()) std::cout << "      note        = " << r.note << "\n";
    std::cout << "      checksum    = 0x" << std::hex << r.checksum << std::dec << "\n";

    std::ostringstream extra;
    extra << "backend=" << micro::BackendName(backend)
          << " median_s=" << std::fixed << std::setprecision(6) << r.median_s;
    if (r.work_modmul > 0.0)
        extra << " gmodmul_s=" << std::fixed << std::setprecision(4) << r.gmodmul_per_s();
    if (r.op == "rref" || r.op == "kernel")
        extra << " rank=" << r.rank << " nullity=" << r.nullity;
    extra << " checksum=0x" << std::hex << r.checksum << std::dec;
    PrintSummary("micro", "op=" + r.op + " " + r.params, r.min_s, extra.str(), rec);
    if (rec != nullptr) {
        rec->params = "op=" + r.op + " " + r.params;
        rec->totals = {r.min_s};
        rec->median_s = r.median_s;
        rec->checksum = r.checksum;
    }
}

// A row of the exit-criterion table: measured value, the plan's floor, and
// whether the floor is a lower (rate) or upper (time) bound.
struct SuiteRow {
    std::string label;
    std::string measured;      // formatted
    std::string floor;         // formatted, or "--"
    bool has_floor = false;
    bool meets = false;
    std::string checksum;
};

std::string Fmt(double v, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

std::string Hex(uint64_t v) {
    std::ostringstream os;
    os << "0x" << std::hex << v;
    return os.str();
}

// `micro suite` -- exactly the Phase-2 exit-criterion set of
// docs/matrix-engine/00-implementation-plan.md ("Phase 2", "Exit criterion"):
// GEMM n=128 >= 15 Gmodmul/s, RREF 1677x1419 <= 0.6 s, cohom 128 <= 4 s,
// plus three context rows that carry no plan floor: gemm 512 (the
// PerformanceSmokeValidation size) and the rref/kernel pair at the REAL rank
// of the pipeline's 1677 x 1419 A_C1 (258, see micro::kSuiteRealRank).
int RunMicroSuite(micro::Backend backend, uint32_t repeats, BenchRecord& rec) {
    std::cout << "=== bench micro suite (backend=" << micro::BackendName(backend)
              << ", repeat=" << repeats << ") ===\n";
    std::vector<SuiteRow> rows;

    const micro::MicroResult g128 = micro::RunGemm(128, 128, micro::kDefaultEll, backend, repeats);
    if (!g128.available) { std::cout << "  " << g128.note << std::endl; return 2; }
    PrintMicroResult(g128, backend, repeats);
    rows.push_back({"gemm 128 (Gmodmul/s)", Fmt(g128.gmodmul_per_s(), 3), ">= 15.000", true,
                    g128.gmodmul_per_s() >= 15.0, Hex(g128.checksum)});

    const micro::MicroResult g512 = micro::RunGemm(512, 512, micro::kDefaultEll, backend, repeats);
    if (!g512.available) { std::cout << "  " << g512.note << std::endl; return 2; }
    PrintMicroResult(g512, backend, repeats);
    rows.push_back({"gemm 512 (Gmodmul/s)", Fmt(g512.gmodmul_per_s(), 3), ">=  4.000", true,
                    g512.gmodmul_per_s() >= 4.0, Hex(g512.checksum)});

    const micro::MicroResult rr = micro::RunRref(1677, 1419, micro::kSuiteRrefRank,
                                                 micro::kDefaultEll, backend, repeats);
    if (!rr.available) { std::cout << "  " << rr.note << std::endl; return 2; }
    PrintMicroResult(rr, backend, repeats);
    rows.push_back({"rref 1677x1419 (s)", Fmt(rr.min_s, 4), "<=  0.6000", true,
                    rr.min_s <= 0.6, Hex(rr.checksum)});

    // The two REAL-RANK rows: the pipeline's own 1677 x 1419 A_C1 at
    // n_sym = 128 has rank 258 (= 2*d_V), not 1419 -- see kSuiteRealRank. The
    // full-rank rows above are what the plan's 0.6 s floor was derived from;
    // these two are what the engine will actually be asked to eliminate, and
    // the kernel row is the only one with a non-trivial nullity to return.
    const micro::MicroResult rr258 = micro::RunRref(1677, 1419, micro::kSuiteRealRank,
                                                    micro::kDefaultEll, backend, repeats);
    if (!rr258.available) { std::cout << "  " << rr258.note << std::endl; return 2; }
    PrintMicroResult(rr258, backend, repeats);
    rows.push_back({"rref 1677x1419 r258 (s)", Fmt(rr258.min_s, 4), "--", false, true,
                    Hex(rr258.checksum)});

    const micro::MicroResult kk = micro::RunKernel(1677, 1419, micro::kSuiteRealRank,
                                                   micro::kDefaultEll, backend, repeats);
    if (!kk.available) { std::cout << "  " << kk.note << std::endl; return 2; }
    PrintMicroResult(kk, backend, repeats);
    rows.push_back({"kernel 1677x1419 r258 (s)", Fmt(kk.min_s, 4), "--", false, true,
                    Hex(kk.checksum)});

    // The pipeline-level exit criterion. It exercises the whole consumer stack,
    // so it is backend-agnostic today: the engine reaches it only once the
    // FlatMatrix retrofit of Phase 2 lands.
    double cohom_best = 0.0;
    {
        std::cout << "  --- cohom 128 134217757 ---\n";
        std::vector<double> totals;
        for (uint32_t i = 0; i < repeats; ++i) {
            std::vector<uint32_t> dims;
            const PhaseTimes t = RunCohom(128, 134217757u, dims);
            totals.push_back(t.total);
        }
        cohom_best = *std::min_element(totals.begin(), totals.end());
        std::cout << "    " << std::left << std::setw(12) << "cohom" << std::right
                  << std::fixed << std::setprecision(6) << std::setw(12) << cohom_best
                  << " s (min)   " << std::setw(12) << Median(totals) << " s (median)\n";
    }
    rows.push_back({"cohom 128 (s)", Fmt(cohom_best, 4), "<=  4.0000", true,
                    cohom_best <= 4.0, "--"});

    std::cout << "\n  Phase-2 exit criteria (plan 00-implementation-plan.md, \"Phase 2\"):\n";
    std::cout << "  " << std::left << std::setw(26) << "case" << std::setw(14) << "measured"
              << std::setw(14) << "floor" << std::setw(8) << "status" << "checksum\n";
    std::cout << "  " << std::string(74, '-') << "\n";
    uint32_t missed = 0;
    for (const auto& r : rows) {
        std::cout << "  " << std::left << std::setw(26) << r.label << std::setw(14) << r.measured
                  << std::setw(14) << r.floor << std::setw(8)
                  << (!r.has_floor ? "n/a" : (r.meets ? "MET" : "MISSED")) << r.checksum << "\n";
        if (r.has_floor && !r.meets) ++missed;
    }
    std::cout << std::right;
    std::cout << "\n  " << missed << " of the plan's exit criteria are not met by backend '"
              << micro::BackendName(backend) << "'.\n";
    PrintSummary("micro", "op=suite", cohom_best,
                 "backend=" + std::string(micro::BackendName(backend)) +
                 " criteria_missed=" + std::to_string(missed), &rec);
    rec.params = "op=suite";
    // `suite` is a REPORT, not a gate: it exits 0 whenever every op ran. The
    // gate is PerformanceSmokeValidation (and, for the plan's own floors, the
    // Stage-B raise of those constants).
    return 0;
}

bool ParseU32(const char* s, uint32_t& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || v > 0xFFFFFFFFull) return false;
    out = static_cast<uint32_t>(v);
    return true;
}

bool ParseU64(const char* s, uint64_t& out) {
    if (!s || !*s) return false;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<uint64_t>(v);
    return true;
}

// (S24 CL-L2, CLEANUP plan Sec. 2.4) An unparsable --backend is a usage error (exit 1).  `legacy` in an
// OC_ENABLE_LEGACY=OFF build is named as compiled out, never silently run on the engine.
std::string BackendUsageError(const std::string& requested) {
    if (requested == "legacy" && !oc::legacy::kLegacyBuild)
        return "--bench: " + oc::legacy::compiled_out_text("--backend legacy");
    return "--bench: --backend needs 'legacy' or 'engine'";
}

void PrintMicroUsage() {
    std::cout <<
        "Usage: hacs --bench micro <op> <args...> [--repeat k] [--backend legacy|engine|gpu]\n"
        "\n"
        "Ops (every <ell> defaults to 134217757, the benchmark prime of doc 03):\n"
        "  gemm      <n> [<k>] [<ell>]                 C = A*B, A: n x k, B: k x n; work = n*n*k\n"
        "  gemv      <n> [<ell>]                       y = A*x, A: n x n; work = n*n\n"
        "  rref      <rows> <cols> [<rank>] [<ell>]    RREF of a matrix of prescribed rank\n"
        "  kernel    <rows> <cols> [<rank>] [<ell>]    kernel() of the same matrix family\n"
        "  transpose <n> [<ell>]                       transpose of an n x n matrix\n"
        "  suite                                       the Phase-2 exit-criterion set + floors\n"
        "  cuda                                        the Phase-5 GPU throughput table (needs a device)\n"
        "\n"
        "<rank> omitted, 0, or >= min(rows,cols) means FULL rank (a plain random matrix,\n"
        "which is what doc 03 Sec. 7.3's reference elimination uses); a smaller <rank>\n"
        "builds L(rows x rank) * R(rank x cols) so rank-deficient inputs are covered.\n"
        "\n"
        "All inputs are fixed-seed (splitmix64 over the shape), so two runs -- and the two\n"
        "backends -- see byte-identical inputs; the reported FNV-1a-64 checksum of the\n"
        "result therefore doubles as a bit-identity check between backends.\n"
        "\n"
        "--backend engine selects the Phase-2 ModCtx/DenseBuf/Kernels layer (the default).\n"
        "--backend gpu selects the Phase-5 engine::cuda device kernels (gemm / gemv /\n"
        "transpose; rref / kernel stay CPU-driven and report \"not available\", exit 2).\n"
        "Without a usable CUDA device every gpu op reports \"not available\", exit 2.\n";
}

// Runs one `micro` op -- the body of the pre-extraction RunMicroCli after its
// argument parse, unchanged, printing exactly what it printed.  (slot-7 WP-8,
// G4) Reached from RunMode(), so the console's `micro(...)` verb and the CLI
// run the same code.
int RunMicroOp(const std::string& op, const std::vector<uint32_t>& a, micro::Backend backend,
               uint32_t repeats, BenchRecord& rec) {
    if (op == "suite") return RunMicroSuite(backend, repeats, rec);
    if (op == "cuda") {
        // Phase 5: the GPU throughput table lives behind the same hook the
        // gpu backend uses (installed by main.cpp's main before dispatch).
        const micro::GpuHooks& h = micro::GetGpuHooks();
        if (!h.suite) {
            rec.error = "--bench micro cuda: not available -- no usable CUDA device";
            std::cout << rec.error << std::endl;
            return 2;
        }
        return h.suite(repeats);
    }

    auto arg = [&](size_t i, uint32_t dflt) -> uint32_t { return i < a.size() ? a[i] : dflt; };
    auto need_args = [&](size_t k) -> bool {
        if (a.size() < k) {
            std::ostringstream msg;
            msg << "--bench micro " << op << ": expected at least " << k
                << " argument(s), got " << a.size();
            rec.error = msg.str();
            std::cout << rec.error << std::endl;
            PrintMicroUsage();
            return false;
        }
        return true;
    };

    std::cout << "=== bench micro " << op << " (backend=" << micro::BackendName(backend)
              << ", repeat=" << repeats << ") ===\n";

    micro::MicroResult r;
    if (op == "gemm") {
        if (!need_args(1)) return 1;
        const uint32_t n = a[0];
        r = micro::RunGemm(n, arg(1, n), arg(2, micro::kDefaultEll), backend, repeats);
    } else if (op == "gemv") {
        if (!need_args(1)) return 1;
        r = micro::RunGemv(a[0], arg(1, micro::kDefaultEll), backend, repeats);
    } else if (op == "rref") {
        if (!need_args(2)) return 1;
        r = micro::RunRref(a[0], a[1], arg(2, 0), arg(3, micro::kDefaultEll), backend, repeats);
    } else if (op == "kernel") {
        if (!need_args(2)) return 1;
        r = micro::RunKernel(a[0], a[1], arg(2, 0), arg(3, micro::kDefaultEll), backend, repeats);
    } else if (op == "transpose") {
        if (!need_args(1)) return 1;
        r = micro::RunTranspose(a[0], arg(1, micro::kDefaultEll), backend, repeats);
    } else {
        rec.error = "--bench micro: unknown op '" + op + "'";
        std::cout << rec.error << std::endl;
        PrintMicroUsage();
        return 1;
    }

    if (!r.available) {
        rec.error = "    " + r.op + " " + r.params + ": not available -- " + r.note;
        std::cout << rec.error << std::endl;
        return 2;
    }
    PrintMicroResult(r, backend, repeats, &rec);
    return 0;
}

// Parses one `--bench micro ...` invocation and hands it to RunMode(). `pos`
// are the positional arguments after the `micro` token, i.e. pos[0] is the op
// name.  The parse -- and every message it prints -- is the pre-extraction
// one; `suite` and `cuda` are handed on before the integer parse, as before.
int RunMicroCli(const std::vector<const char*>& pos, micro::Backend backend, uint32_t repeats) {
    if (pos.empty()) { PrintMicroUsage(); return 1; }
    const std::string op = pos[0];
    if (op == "--help" || op == "-h" || op == "help") { PrintMicroUsage(); return 0; }

    ModeArgs args;
    args.mode = "micro";
    args.op = op;
    args.backend = micro::BackendName(backend);
    args.repeats = repeats;
    if (op != "suite" && op != "cuda") {
        // Positional arguments after the op name.
        for (size_t i = 1; i < pos.size(); ++i) {
            uint32_t v = 0;
            if (!ParseU32(pos[i], v)) {
                std::cout << "--bench micro " << op << ": '" << pos[i]
                          << "' is not a non-negative 32-bit integer" << std::endl;
                return 1;
            }
            args.op_args.push_back(v);
        }
    }
    return RunMode(args).exit_code;
}

// The pipeline modes' argument checks, shared by the CLI parse and RunMode()
// (which the console reaches without the CLI).  Returns the message RunBench
// prints, or an empty string.  The CLI runs its own checks first, in its own
// order, so from the CLI this never fires.
std::string PipelineArgsError(const ModeArgs& a) {
    if (a.mode != "cohom" && a.mode != "hecke" && a.mode != "family" && a.mode != "decomp" &&
        a.mode != "ext") {
        return "--bench: unknown mode '" + a.mode + "'";
    }
    if (a.ell < 2) return "--bench: <n_sym> and <ell> must be non-negative integers, ell >= 2";
    if ((a.mode == "hecke" || a.mode == "family" || a.mode == "decomp") && a.p < 2)
        return "--bench: <p> must be a prime >= 2";
    if (a.mode == "family" && a.r == 0) return "--bench: <r> must be a positive integer";
    if (a.mode == "ext") {
        for (uint64_t q : a.family) {
            if (q < 2) return "--bench ext: the family must be a comma-separated list of primes >= 2";
        }
    }
    if (a.repeats == 0) return "--bench: --repeat needs a positive integer";
    return std::string();
}

} // namespace

void PrintBenchUsage(const char* argv0) {
    std::cout <<
        "Usage: " << argv0 << " --bench <mode> <args...> [--repeat k] [--cross-checks]\n"
        "\n"
        "Modes:\n"
        "  cohom  <n_sym> <ell>            EquivariantCohomologyOnK on the SL(2,Z) model, all q\n"
        "  hecke  <n_sym> <ell> <p>        T_p end to end (cosets -> tau -> Theta_g -> A^(q))\n"
        "  family <n_sym> <ell> <p> <r>    T_{p^r} by recursion AND by the direct double-coset route\n"
        "  decomp <n_sym> <ell> <p>        + primary module decomposition of the largest H^q\n"
        "  ext    <n_sym> <ell> [p,q,...]  + the extension layer, Stages 1-5 (default family 2,3,5)\n"
        "  micro  <op> <args...>           Phase-2 arithmetic micro-benchmarks; `micro --help`\n"
        "                                  lists the ops (gemm/gemv/rref/kernel/transpose/suite)\n"
        "\n"
        "Options:\n"
        "  --repeat k          run the whole pipeline k times; print min/median of the totals\n"
        "  --backend b         (mode `micro`) legacy | engine; default engine\n"
        "                      (`legacy` = the pre-Phase-2 substrate; ENGINE_BACKEND\n"
        "                      sets the default)\n"
        "  --cross-checks      (mode `hecke`) additionally run the alpha/beta/omega route\n"
        "                      (off by default: it roughly doubles the measured time)\n"
        "  --lean              memory-lean configuration: drop the cohomology objects no\n"
        "                      production consumer reads and skip the Theta_eq certification\n"
        "                      (A^(q) unchanged; reported as lean=1 on the BENCH line)\n"
        "  --count-allocs      report heap allocations (count and requested bytes) made\n"
        "                      during each measured repeat, via a counting replacement\n"
        "                      global operator new; adds allocs=/alloc_bytes= to the\n"
        "                      BENCH line. With -DENGINE_COUNT_OBJECTS=ON it also reports\n"
        "                      FlatMatrix / BlockFlatMatrix constructions.\n"
        "\n"
        "Every run ends with one machine-readable line:\n"
        "  BENCH mode=... n=... ell=... [p=...] [r=...] total_s=... rss_mb=... <mode-specific keys>\n"
        "(rss_mb is peak RSS over the whole process, from /proc/self/status VmHWM.)\n"
        "\n"
        "NOTE: on this model H^*(K, R; Sym^n) vanishes for ODD n (the -I / odd-n base-model\n"
        "gap); use an even n_sym for a meaningful measurement.\n";
}

int RunBench(int argc, char** argv) {
    if (argc < 1) { PrintBenchUsage("hacs"); return 1; }

    const std::string mode = argv[0];
    if (mode == "--help" || mode == "-h") { PrintBenchUsage("hacs"); return 0; }

    // Strip options first, leaving positional arguments in `pos`.
    std::vector<const char*> pos;
    uint32_t repeats = 1;
    // OFF by default: the Phase-0 exit criterion is
    // `--bench hecke 48 134217757 2` reporting ~5 s (doc 03 Sec. 4.1's 5.29 s
    // at -O2), which is the T_p pipeline ALONE. The alpha/beta/omega route is
    // an independent recomputation roughly as expensive again, so it is
    // opt-in via --cross-checks.
    bool cross_checks = false;
    // (matrix-engine Phase 3) --count-allocs: report the allocation delta over
    // each measured repeat. The counters run unconditionally (two increments per
    // allocation); the flag only decides whether the numbers are printed.
    bool count_allocs = false;
    // (matrix-engine Phase 4, memory profile 05 Sec. 7 items 2-3) --lean: the
    // memory-lean pipeline configuration -- EquivariantCohomologyOnK drops the
    // objects no production consumer reads and HeckeOperatorOnK skips the
    // Theta_eq certification (Theta8/9).  Values of A^{(q)} are unchanged.
    bool lean = false;
    // (S24 CL-R1) The slot-13/14 lever flags (--[no-]u-ell-low-rank,
    // --[no-]coset-stream, --[no-]parallel-setup, all outside the frozen
    // `--bench --help` text) are gone with the retired levers: each lever is
    // its default route (in an OC_ENABLE_LEGACY=ON build the env opt-outs
    // HECKE_U_ELL_LOW_RANK=0 / HECKE_COSET_STREAM=0 / HECKE_PARALLEL_SETUP=0
    // still reach the old routes).
    // (Phase 2, merge) default `engine`, overridable by --backend and by
    // ENGINE_BACKEND: the engine arm is the shipped substrate.
    micro::Backend backend = micro::BackendFromEnv();
    // `--bench <mode> --help` must reach the MODE's usage block, so --help is
    // stripped here rather than rejected by the unknown-option branch below.
    bool want_help = false;
    // (hecke-extensions slot 7, WP-6 / E2 (e')) `ext` only; deliberately
    // OUTSIDE `PrintBenchUsage` / `cli_frozen/bench_help.stdout` (the plan's
    // fallback rule applies if that file ever needs to move; it does not
    // here -- these two flags are undocumented in the frozen text by design,
    // documented instead in docs/performance-reference.md). See
    // ModeArgs::with_cone / with_sign_split / u_ell for what they do.
    bool campaign_cell = false;
    std::string u_ell_opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--repeat") {
            if (i + 1 >= argc || !ParseU32(argv[i + 1], repeats) || repeats == 0) {
                std::cout << "--bench: --repeat needs a positive integer" << std::endl;
                return 1;
            }
            ++i;
        } else if (a == "--help" || a == "-h") {
            want_help = true;
        } else if (a == "--backend") {
            if (i + 1 >= argc || !micro::ParseBackend(std::string(argv[i + 1]), backend)) {
                std::cout << BackendUsageError(i + 1 < argc ? argv[i + 1] : "") << std::endl;
                return 1;
            }
            ++i;
        } else if (a == "--count-allocs") {
            count_allocs = true;
        } else if (a == "--lean") {
            lean = true;
        } else if (a == "--cross-checks") {
            cross_checks = true;
        } else if (a == "--no-cross-checks") {
            cross_checks = false;
        } else if (a == "--campaign-cell") {
            campaign_cell = true;
        } else if (a == "--u-ell") {
            if (i + 1 >= argc) {
                std::cout << "--bench: --u-ell needs a value (auto | off | budget_mb=<M>)" << std::endl;
                return 1;
            }
            u_ell_opt = argv[++i];
        } else if (!a.empty() && a[0] == '-') {
            std::cout << "--bench: unknown option '" << a << "'" << std::endl;
            return 1;
        } else {
            pos.push_back(argv[i]);
        }
    }

    // `micro` takes an op name rather than <n_sym> <ell>, so it is routed
    // before the pipeline-mode argument parsing below.
    if (mode == "micro") {
        if (want_help) { PrintMicroUsage(); return 0; }
        // (slot-7 WP-8, G4) The exception guard is RunMode()'s now, around the
        // same code, printing the same line.
        return RunMicroCli(pos, backend, repeats);
    }

    if (want_help) { PrintBenchUsage("hacs"); return 0; }

    uint32_t n_sym = 0, ell = 0, r = 1;
    uint64_t p_hecke = 0;

    auto need = [&](size_t k) -> bool {
        if (pos.size() < k) {
            std::cout << "--bench " << mode << ": expected " << k << " positional argument(s), got "
                      << pos.size() << std::endl;
            PrintBenchUsage("hacs");
            return false;
        }
        return true;
    };

    if (mode != "cohom" && mode != "hecke" && mode != "family" && mode != "decomp" &&
        mode != "ext" && mode != "micro") {
        std::cout << "--bench: unknown mode '" << mode << "'" << std::endl;
        PrintBenchUsage("hacs");
        return 1;
    }

    // `ext` takes an optional COMMA-SEPARATED family in place of a single p.
    std::vector<uint64_t> ext_family;
    const size_t required = (mode == "cohom" || mode == "ext") ? 2
                                                               : (mode == "family" ? 4 : 3);
    if (!need(required)) return 1;
    if (!ParseU32(pos[0], n_sym) || !ParseU32(pos[1], ell) || ell < 2) {
        std::cout << "--bench: <n_sym> and <ell> must be non-negative integers, ell >= 2" << std::endl;
        return 1;
    }
    if (mode == "ext") {
        // (slot-7 WP-6) Under `--campaign-cell`, the CLI's own default family
        // fallback (no explicit list given) is {2,3,5,7}, the engine's own
        // default -- not the plain `ext` mode's {2,3,5} -- matching
        // CampaignExt.cpp's `c.family = j.ext_family` at an empty spec
        // `family`.  An explicit list (pos[2]) always wins either way, so a
        // script can also pass the union with the cell's own prime directly.
        std::string list = (pos.size() >= 3) ? std::string(pos[2])
                                              : (campaign_cell ? std::string("2,3,5,7")
                                                                : std::string("2,3,5"));
        size_t at = 0;
        while (at <= list.size()) {
            const size_t comma = list.find(',', at);
            const std::string tok = list.substr(at, comma == std::string::npos
                                                           ? std::string::npos
                                                           : comma - at);
            uint64_t q = 0;
            if (tok.empty() || !ParseU64(tok.c_str(), q) || q < 2) {
                std::cout << "--bench ext: the family must be a comma-separated list of primes >= 2"
                          << std::endl;
                return 1;
            }
            ext_family.push_back(q);
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    } else if (mode != "cohom" && (!ParseU64(pos[2], p_hecke) || p_hecke < 2)) {
        std::cout << "--bench: <p> must be a prime >= 2" << std::endl;
        return 1;
    }
    if (mode == "family" && (!ParseU32(pos[3], r) || r == 0)) {
        std::cout << "--bench: <r> must be a positive integer" << std::endl;
        return 1;
    }
    // (slot-7 WP-6) Validated here, at the CLI parse, so a bad --u-ell value
    // is the usual exit-1 usage error, not an exit-2 pipeline exception.
    if (!u_ell_opt.empty()) {
        try {
            hecke::ParseUEll(u_ell_opt);
        } catch (const std::invalid_argument& e) {
            std::cout << "--bench: --u-ell: " << e.what() << std::endl;
            return 1;
        }
    }

    // (slot-7 WP-8, G4) Everything after the parse is RunMode()'s.
    ModeArgs args;
    args.mode = mode;
    args.n_sym = n_sym;
    args.ell = ell;
    args.p = p_hecke;
    args.r = r;
    args.family = ext_family;
    args.repeats = repeats;
    args.cross_checks = cross_checks;
    args.lean = lean;
    args.count_allocs = count_allocs;
    args.with_cone = campaign_cell;
    args.with_sign_split = campaign_cell;
    args.u_ell = u_ell_opt;
    return RunMode(args).exit_code;
}

namespace {

// The pipeline modes: everything RunBench() did after its argument parse,
// unchanged line for line, plus the record.  (slot-7 WP-8, G4)
BenchRecord RunPipelineMode(const ModeArgs& args) {
    BenchRecord rec;
    rec.mode = args.mode;
    const std::string& mode = args.mode;
    const uint32_t n_sym = args.n_sym;
    const uint32_t ell = args.ell;
    const uint64_t p_hecke = args.p;
    const uint32_t r = args.r;
    const uint32_t repeats = args.repeats;
    const bool cross_checks = args.cross_checks;
    const bool count_allocs = args.count_allocs;
    const bool lean = args.lean;
    // (slot-7 WP-6) The two campaign-cell driver options, off by default so
    // every pre-WP-6 call is byte-unchanged.
    const bool with_cone = args.with_cone;
    const bool with_sign_split = args.with_sign_split;

    // The CLI's default family when `ext` is given none (a console call may
    // omit it; the CLI always passes its parsed, non-empty list).  Under
    // `--campaign-cell` (with_cone / with_sign_split both true, RunBench's
    // parse) the fallback is the campaign's own default family {2,3,5,7}
    // (CampaignExt.cpp: spec `family` empty -> the engine default {2,3,5,7}),
    // not the plain `ext` mode's {2,3,5}.
    const std::vector<uint64_t> ext_family =
        (mode == "ext" && args.family.empty())
            ? (with_cone || with_sign_split ? std::vector<uint64_t>{2, 3, 5, 7}
                                             : std::vector<uint64_t>{2, 3, 5})
            : args.family;

    std::ostringstream params;
    params << "n=" << n_sym << " ell=" << ell;
    if (mode == "ext") {
        params << " family=";
        for (size_t i = 0; i < ext_family.size(); ++i) {
            if (i) params << ",";
            params << ext_family[i];
        }
        // (slot-7 WP-6) Printed ONLY when set -- a default `--bench ext` line
        // (with_cone=false, with_sign_split=false, u_ell empty) is therefore
        // byte-unchanged from every pre-WP-6 run.
        if (with_cone) params << " with_cone=1";
        if (with_sign_split) params << " with_sign_split=1";
        if (!args.u_ell.empty()) params << " u_ell=" << args.u_ell;
    } else if (mode != "cohom") {
        params << " p=" << p_hecke;
    }
    if (mode == "family") params << " r=" << r;

    std::cout << "=== bench " << mode << " (" << params.str() << ", repeat=" << repeats << (lean ? ", lean" : "") << ") ===\n";
    EquivariantCohomologyOptions::ScopedRetainIntermediates retain_guard(!lean);
    hecke::HeckeOperatorOptions::ScopedThetaEqCertification theta_eq_guard(!lean);

    std::vector<double> totals;
    std::string extra;
    std::string alloc_extra;
    // `ext`: the three phases' minima over the repeats, and the per-repeat
    // ratios, which is exactly how the gate is stated.
    std::vector<double> ext_pipe, ext_dec, ext_ext, ext_bound, ext_stream, ext_h1;
    // (slot-5 WP-7 / E6) The two report-form columns of the Stage-6 gate.
    std::vector<double> ext_canon, ext_own;
    // (slot-9 WP-7, L5) `--campaign-cell` only: the cone, the sign split and
    // the cell-local proof_range decision, minima over the repeats.
    std::vector<double> ext_cone, ext_iota, ext_proof_range;
    uint32_t ext_canon_ideals = 0, ext_canon_ok = 0, ext_canon_r_max = 0;
    double ext_stream_rss = 0.0;
    uint32_t ext_streamed = 0;
    // (slot-7 WP-6) `--u-ell VALUE`: scoped for the whole measured region (all
    // repeats), reverted on return via RAII, exactly as CampaignSpec.cpp's own
    // `u_ell = ...` line is applied for a spec run -- same SettingSource::Spec
    // stamp. Empty (the default): the process-wide setting is left alone,
    // which is "auto" unless something else already changed it, matching
    // ext_first.spec's own `u_ell = auto` ([S7] Sec 6 confirmation 9).
    // hecke::ParseUEll's std::invalid_argument on a bad value is caught by the
    // try block below, exactly like any other pipeline exception (exit_code 2).
    std::optional<hecke::ScopedUEll> u_ell_guard;
    try {
        if (mode == "ext" && !args.u_ell.empty()) {
            u_ell_guard.emplace(hecke::ParseUEll(args.u_ell), hecke::SettingSource::Spec);
        }
        for (uint32_t i = 0; i < repeats; ++i) {
            std::vector<uint32_t> dims;
            PhaseTimes t;
            std::ostringstream ex;
            const AllocSnapshot a0 = TakeAllocSnapshot();
            if (mode == "cohom") {
                std::vector<uint32_t> c_eq;
                t = RunCohom(n_sym, ell, dims, &c_eq);
                ex << "dims=" << DimsString(dims) << " c_eq_dims=" << DimsString(c_eq);
            } else if (mode == "hecke") {
                uint32_t rc = 0;
                t = RunHecke(n_sym, ell, p_hecke, cross_checks, dims, rc);
                ex << "dims=" << DimsString(dims) << " xcheck_rc=" << (cross_checks ? std::to_string(rc) : std::string("skipped"));
            } else if (mode == "family") {
                bool agree = false;
                t = RunFamily(n_sym, ell, p_hecke, r, dims, agree);
                ex << "dims=" << DimsString(dims) << " rec_eq_direct=" << (agree ? 1 : 0);
            } else if (mode == "ext") {
                hecke_ext::ExtDriverTimings et;
                uint32_t ideals = 0, streamed = 0;
                double streamed_rss = 0.0, hecke1 = 0.0;
                double canon_s = 0.0, own_s = 0.0;
                uint32_t canon_ideals = 0, canon_ok = 0, canon_r_max = 0;
                t = RunExt(n_sym, ell, ext_family, et, dims, ideals, streamed, streamed_rss,
                           hecke1, canon_s, own_s, canon_ideals, canon_ok, canon_r_max,
                           with_cone, with_sign_split);
                ext_canon_r_max = canon_r_max;
                ext_canon.push_back(canon_s);
                ext_own.push_back(own_s);
                ext_canon_ideals = canon_ideals;
                ext_canon_ok = canon_ok;
                ext_pipe.push_back(et.pipeline_s());
                ext_dec.push_back(et.decomp_s());
                ext_ext.push_back(et.ext_s());
                ext_bound.push_back(et.sturm_bound_s);
                ext_stream.push_back(et.streamed_s);
                ext_cone.push_back(et.cone_s);
                ext_iota.push_back(et.iota_s);
                ext_proof_range.push_back(et.proof_range_s);
                ext_h1.push_back(hecke1);
                ext_stream_rss = streamed_rss;
                ext_streamed = streamed;
                const double denom = et.pipeline_s() + et.decomp_s();
                std::cout << "    [ext] this repeat: ext_s/(pipeline_s+decomp_s) = "
                          << std::fixed << std::setprecision(4)
                          << (denom > 0.0 ? et.ext_s() / denom : 0.0)
                          << ", ext_s/hecke1_s = "
                          << (hecke1 > 0.0 ? et.ext_s() / hecke1 : 0.0) << "\n";
                ex << "dims=" << DimsString(dims) << " ideals=" << ideals
                   << " streamed=" << streamed;
            } else {
                uint32_t blocks = 0, cert = 0, q_used = 0;
                t = RunDecomp(n_sym, ell, p_hecke, dims, blocks, cert, q_used);
                ex << "dims=" << DimsString(dims) << " q=" << q_used
                   << " blocks=" << blocks << " certify=" << cert;
            }
            const AllocSnapshot a1 = TakeAllocSnapshot();
            PrintPhases(t, i, repeats);
            if (engine::tape::trace_nodes_enabled()) {
                // Diagnostic: per-op / per-class engine profile of this repeat
                // (process-wide totals since the last reset).
                const engine::tape::TraceTotals tt = engine::tape::trace_totals();
                const double wall = t.total;
                std::cout << "    [engine trace] repeat wall " << std::fixed << std::setprecision(3) << wall
                          << " s: executor " << tt.execute_wall << " s in " << tt.runs << " runs, compile " << tt.compile_wall
                          << " s in " << tt.compiles << " compiles, host outside executor+compile "
                          << (wall - tt.execute_wall - tt.compile_wall) << " s (" << std::setprecision(1)
                          << 100.0 * (wall - tt.execute_wall - tt.compile_wall) / wall << " %)\n";
                std::cout << "    [engine trace] pool: " << tt.pool_regions << " regions, region wall " << std::setprecision(3)
                          << tt.pool_wall << " s (" << std::setprecision(1) << 100.0 * tt.pool_wall / wall << " % of wall), busy "
                          << std::setprecision(3) << tt.pool_busy << " s, mean width used "
                          << std::setprecision(2) << (tt.pool_wall > 0 ? tt.pool_busy / tt.pool_wall : 0.0)
                          << ", mean threads/region " << (tt.pool_regions ? double(tt.pool_thread_sum) / tt.pool_regions : 0.0)
                          << ", idle inside regions " << std::setprecision(1)
                          << (tt.pool_thread_sum && tt.pool_wall > 0 ? 100.0 * (1.0 - tt.pool_busy / (tt.pool_wall * (double(tt.pool_thread_sum) / tt.pool_regions))) : 0.0)
                          << " %, wait for region mutex " << std::setprecision(3) << tt.pool_wait << " s\n";
                const char* cls[3] = {"memory", "compute", "serial"};
                for (uint32_t c = 0; c < 3; ++c) {
                    if (!tt.class_regions[c]) continue;
                    std::cout << "    [engine trace]   class " << std::left << std::setw(8) << cls[c] << std::right
                              << " regions " << std::setw(6) << tt.class_regions[c] << " wall " << std::setprecision(3) << std::setw(7) << tt.class_wall[c]
                              << " s busy " << std::setw(7) << tt.class_busy[c] << " s width " << std::setprecision(2)
                              << (tt.class_wall[c] > 0 ? tt.class_busy[c] / tt.class_wall[c] : 0.0) << "\n";
                }
                for (uint32_t op = 0; op < engine::tape::kTraceOps; ++op) {
                    if (!tt.op_nodes[op] && !tt.op_tasks[op]) continue;
                    std::cout << "    [engine trace]   op " << std::left << std::setw(13) << engine::tape::op_name(static_cast<engine::tape::Op>(op)) << std::right
                              << " nodes " << std::setw(7) << tt.op_nodes[op] << " tasks " << std::setw(7) << tt.op_tasks[op]
                              << " busy " << std::setprecision(3) << std::setw(7) << tt.op_busy[op] << " s\n";
                }
                const engine::elim::PluPhaseStats ps = engine::elim::plu_phase_totals();
                std::cout << "    [engine trace]   dense PLU: " << ps.calls << " calls, total " << std::setprecision(3) << ps.total
                          << " s = base panels " << ps.base_panel << " + TRSM " << ps.trsm << " + trailing GEMM " << ps.trailing_gemm
                          << " + to_rref " << ps.to_rref << " (+ rest); base row passes " << ps.base_rows
                          << " of which hooked " << ps.base_rows_par << "\n";
                engine::elim::reset_plu_phase_totals();
                engine::tape::reset_trace_totals();
            }
            if (std::getenv("ENGINE_TRACE_ELIM") != nullptr) {
                // Diagnostic: where the tape elimination driver's wall time went
                // in this repeat (process-wide totals since the last reset).
                const engine::tape::TapeElimStats te = engine::tape::tape_elim_totals();
                std::cout << "    [tape elim] panels=" << te.panels << " programs=" << te.programs << " nodes=" << te.nodes
                          << " host=" << std::fixed << std::setprecision(3) << te.s_host << " s compile=" << te.s_compile
                          << " s run_p1=" << te.s_run_p1 << " s (serial nodes " << te.s_serial << " s) run_p2=" << te.s_run_p2 << " s\n";
                engine::tape::reset_tape_elim_totals();
            }
            if (count_allocs) {
                const std::size_t n_alloc = a1.count - a0.count;
                const std::size_t n_bytes = a1.bytes - a0.bytes;
                std::cout << "    " << std::left << std::setw(34) << "allocations" << std::right
                          << std::setw(12) << n_alloc << "   ("
                          << std::fixed << std::setprecision(1)
                          << (static_cast<double>(n_bytes) / (1024.0 * 1024.0)) << " MiB requested)\n";
                if (engine::counters::kObjectCountersEnabled) {
                    std::cout << "    " << std::left << std::setw(34) << "FlatMatrix constructions"
                              << std::right << std::setw(12) << (a1.flat_ctors - a0.flat_ctors) << "\n";
                    std::cout << "    " << std::left << std::setw(34) << "BlockFlatMatrix constructions"
                              << std::right << std::setw(12) << (a1.block_ctors - a0.block_ctors) << "\n";
                }
                // The BENCH line reports the LAST repeat: the first one also
                // pays the one-off lazily built tables (factorials, tessellation
                // tables), so the last repeat is the steady-state figure.
                std::ostringstream ac;
                ac << " allocs=" << n_alloc << " alloc_bytes=" << n_bytes;
                if (engine::counters::kObjectCountersEnabled) {
                    ac << " flat_ctors=" << (a1.flat_ctors - a0.flat_ctors)
                       << " block_ctors=" << (a1.block_ctors - a0.block_ctors);
                }
                alloc_extra = ac.str();
            }
            totals.push_back(t.total);
            extra = ex.str();
            rec.dims = dims;
            rec.phases = t.phases;
        }
    } catch (const std::exception& e) {
        std::cout << "--bench " << mode << ": EXCEPTION: " << e.what() << std::endl;
        rec.exit_code = 2;
        rec.error = e.what();
        return rec;
    }

    const double best = *std::min_element(totals.begin(), totals.end());
    if (repeats > 1) {
        std::cout << "  min total    = " << std::fixed << std::setprecision(4) << best << " s\n";
        std::cout << "  median total = " << std::fixed << std::setprecision(4) << Median(totals) << " s\n";
    }
    // The summary line reports the MINIMUM total over the repeats (the
    // standard convention for wall-clock microbenchmarks: the minimum is the
    // least noise-contaminated estimate); the median is printed above it.
    if (mode == "ext" && !ext_pipe.empty()) {
        const double p_min = *std::min_element(ext_pipe.begin(), ext_pipe.end());
        const double d_min = *std::min_element(ext_dec.begin(), ext_dec.end());
        const double e_min = *std::min_element(ext_ext.begin(), ext_ext.end());
        const double b_min = *std::min_element(ext_bound.begin(), ext_bound.end());
        const double s_min = *std::min_element(ext_stream.begin(), ext_stream.end());
        const double h_min = *std::min_element(ext_h1.begin(), ext_h1.end());
        const double ratio = (p_min + d_min) > 0.0 ? e_min / (p_min + d_min) : 0.0;
        std::cout << "  pipeline_s   = " << std::fixed << std::setprecision(4) << p_min << " s (min)\n";
        std::cout << "  decomp_s     = " << d_min << " s (min)\n";
        std::cout << "  ext_s        = " << e_min << " s (min)\n";
        std::cout << "  ext_ratio    = " << ratio << "   (gate: <= 0.20)\n";
        // (hecke-extensions slot 5, WP-7 / slot 6, WP-7; [OD] E6-1) The
        // Stage-6 report-form measurement.  E6-1 RESTATED the gate as
        // `canonical_field_s <= 0.10 * total_s`, `total_s` being the wall of
        // `HeckeExtDriver::run()` (`best`, the same figure the generic BENCH
        // line prints as `total_s=`) -- NOT `ext_s` (Stages 3-5 alone), and
        // the `<= 2 * own_field_s` half is deleted ([X] Sec. 6.2 E6 (d)).
        const double cf_min = *std::min_element(ext_canon.begin(), ext_canon.end());
        const double of_min = *std::min_element(ext_own.begin(), ext_own.end());
        std::cout << "  canonical_field_s = " << std::setprecision(6) << cf_min << " s (min, "
                  << ext_canon_ok << "/" << ext_canon_ideals << " ideals split, max r = "
                  << ext_canon_r_max << ")\n";
        std::cout << "  own_field_s       = " << of_min << " s (min)\n";
        const double total_ratio = (best > 0.0) ? cf_min / best : 0.0;
        // (slot-7 WP-3, item 6; [S7] Sec. 2.4 T3) The split runs inside run(),
        // so total_s contains it; the E6 (d) denominator is total_s minus it.
        const double net = best - cf_min;
        const double net_ratio = (net > 0.0) ? cf_min / net : 0.0;
        std::cout << "  total_s           = " << std::setprecision(6) << best << " s (min)\n";
        std::cout << "  total_s - canonical_field_s = " << net << " s\n";
        std::cout << "  [E6 gate] canonical_field_s/(total_s - canonical_field_s) = "
                  << std::setprecision(4) << net_ratio << " (gate: <= 0.10)\n";
        std::cout << "  [reported] canonical_field_s/total_s = " << std::setprecision(4)
                  << total_ratio << "\n";
        std::cout << "  [E6 report forms, reported only] canonical_field_s/ext_s = "
                  << std::setprecision(4) << (e_min > 0.0 ? cf_min / e_min : 0.0)
                  << ",  canonical_field_s/own_field_s = "
                  << (of_min > 0.0 ? cf_min / of_min : 0.0) << "\n";
        std::ostringstream ee;
        ee << " pipeline_s=" << std::fixed << std::setprecision(4) << p_min
           << " decomp_s=" << d_min
           // (slot-6 WP-9) 6 decimals: at 4, ext_s at (160, 691) had ~3 %
           // resolution, too coarse for the WP-10 head-to-head.  Print only.
           << std::setprecision(6) << " ext_s=" << e_min << std::setprecision(4)
           << " sturm_bound_s=" << b_min << " streamed_s=" << s_min
           << " streamed=" << ext_streamed
           << " streamed_rss_mb=" << std::setprecision(1) << ext_stream_rss
           << " ext_ratio=" << std::setprecision(4) << ratio
           << " ext_over_hecke1_ratio=" << (h_min > 0.0 ? e_min / h_min : 0.0)
           << " hecke1_s=" << h_min
           // (slot-5 WP-7 / E6) reported, never gated here.
           << " canonical_field_s=" << std::setprecision(6) << cf_min
           << " own_field_s=" << of_min
           << " canonical_field_ideals=" << ext_canon_ideals
           << " canonical_field_ok=" << ext_canon_ok
           << " canonical_field_r_max=" << ext_canon_r_max
           // (hecke-extensions slot 6, WP-7 / [OD] E6-1) the gate ratio itself.
           << " canonical_field_total_ratio=" << std::setprecision(4) << total_ratio
           // (slot-7 WP-3, item 6) the T3 denominator and the gate ratio over it.
           << " total_minus_canonical_field_s=" << std::setprecision(6) << net
           << " canonical_field_net_ratio=" << std::setprecision(4) << net_ratio;
        // (slot-9 WP-7, L5) Appended ONLY under `--campaign-cell`, so the
        // default `--bench ext` BENCH line is byte-unchanged.  Print only.
        if (with_cone || with_sign_split) {
            ee << std::setprecision(6)
               << " cone_s=" << *std::min_element(ext_cone.begin(), ext_cone.end())
               << " iota_s=" << *std::min_element(ext_iota.begin(), ext_iota.end())
               << " proof_range_s=" << *std::min_element(ext_proof_range.begin(), ext_proof_range.end());
        }
        extra += ee.str();
    }
    PrintSummary(mode, params.str(), best, extra + alloc_extra + (lean ? " lean=1" : ""),
                 &rec);
    rec.params = params.str();
    rec.totals = totals;
    rec.median_s = Median(totals);
    rec.exit_code = 0;
    return rec;
}

} // namespace

BenchRecord RunMode(const ModeArgs& args) {
    // (slot-8 WP-3) [E] sec:ring (b4): no --bench mode honours a non-default
    // coefficient_ring in slot 8 (no switch was added); refused before the
    // mode's own exception guard, so the caller sees the named error.
    hecke::RequireFieldCoefficientRing("--bench " + args.mode);
    if (args.mode == "micro") {
        BenchRecord rec;
        rec.mode = "micro";
        micro::Backend backend = micro::BackendFromEnv();
        if (!args.backend.empty() && !micro::ParseBackend(args.backend, backend)) {
            rec.exit_code = 1;
            rec.error = BackendUsageError(args.backend);
            std::cout << rec.error << std::endl;
            return rec;
        }
        if (args.repeats == 0) {
            rec.exit_code = 1;
            rec.error = "--bench: --repeat needs a positive integer";
            std::cout << rec.error << std::endl;
            return rec;
        }
        try {
            rec.exit_code = RunMicroOp(args.op, args.op_args, backend, args.repeats, rec);
        } catch (const std::exception& e) {
            std::cout << "--bench micro: EXCEPTION: " << e.what() << std::endl;
            rec.exit_code = 2;
            rec.error = e.what();
        }
        return rec;
    }
    const std::string err = PipelineArgsError(args);
    if (!err.empty()) {
        BenchRecord rec;
        rec.mode = args.mode;
        rec.exit_code = 1;
        rec.error = err;
        std::cout << err << std::endl;
        return rec;
    }
    return RunPipelineMode(args);
}

} // namespace bench
