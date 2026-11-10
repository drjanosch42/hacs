// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// matrix-engine Phase 6 -- the campaign driver. See CampaignDriver.h for the
// design contract (hashing, certify semantics, memory-lean mode, the Phase 4
// batching hook) and docs/matrix-engine/campaign.md for the user-facing docs.

#include "CampaignDriver.h"
#include "CampaignOrbifold.h"
#include "CampaignDescent.h"
#include "CampaignHeckeZ.h"

#include "CampaignExt.h"
#include "CampaignRing.h"          // slot-9 WP-8: the job loop under Z/ell^k
#include "ProgramCacheRelease.h"   // slot-8 WP-5: the between-jobs release

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeCrossChecks.h"
#include "../Hecke/HeckeModelDescriptor.h"
#include "../Util/Rss.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeReport.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../../ModArithmetic/BlockKernels.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../../Engine/EngineRuntime.h"
#include "../../Engine/ThreadPool.h"
#include "../Util/TextIo.h"

namespace campaign {

namespace {

// ===========================================================================
// Phase 6 Part A -- the 2-level split (`jobs` x `threads`).
//
// A shape group's jobs share (n_sym, p, r, degrees, model) and differ ONLY in
// the modulus, so they are independent pipelines of identical shape: the group
// is exactly the unit to run concurrently.  Two things make that safe and one
// makes it fast:
//
//  (1) THREAD CONTEXT.  The consumers' scoped options (memory-lean, the legacy
//      arms, elim::ScopedLegacy) are thread_local, so a job thread would
//      otherwise see the DEFAULTS rather than the campaign's settings.  We
//      install the launching thread's context on every job thread through the
//      engine's own propagator registry (engine::tape::capture_thread_context),
//      i.e. the same mechanism run_concurrently() uses -- not a private copy.
//
//  (2) THE TAU/COSET LOCK.  HeckeTauTable's build is first-match-wins over a
//      fixed order and drives the NON-ATOMIC global PredicateStats counters
//      (audit 02 Sec. 6.5, hazards 1-3).  Every coset+tau construction in this
//      driver therefore runs under hecke::tau_construction_mutex(), the same
//      process-wide lock HeckeOperatorFamily::build_pipelines() has always
//      held.  Everything downstream of it (compute_cohomology, compute(), the
//      report) is a pure function of per-job objects and runs concurrently.
//
//  (3) PER-JOB ENGINE POOLS.  ThreadPool::parallel_for() serialises concurrent
//      callers on one region mutex, so N job threads sharing ONE pool would
//      serialise all of their engine work.  Each job thread therefore installs
//      its OWN ThreadPool(spec.threads) as the consumer pool.  That requires
//      engine::tape::set_consumer_pool_override() to be thread_local; it is
//      PROBED at run time (PerJobPoolsSupported()) rather than assumed, and
//      when the probe fails the driver falls back to sharing the process pool
//      and says so in index.json.
//
// None of this can move a computed value: every job is an independent pure
// function of its (n_sym, ell, p, r), the pool only decides which worker runs
// which disjoint task, and results are stored into a pre-sized slot vector and
// appended in GROUP ORDER, so the campaign's output order is the plan order
// whatever the scheduling was.
// ===========================================================================

// Propagates the calling thread's consumer-pool selection onto every thread
// engine::tape::run_concurrently() starts, so a job's INNER fan-out (the
// degrees q, the (p, r) pipelines) stays on that job's own pool instead of
// falling back to the process-wide one.  A no-op when the override is global.
const bool g_register_pool_context = [] {
    engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
        engine::tape::ThreadPool* pool = engine::tape::consumer_pool_override();
        return [pool]() -> engine::tape::ThreadContextGuard {
            if (pool == nullptr) return nullptr;
            return std::make_shared<engine::tape::ScopedConsumerPool>(pool);
        };
    });
    return true;
}();

// Is the consumer-pool override thread-local?  Probed, not assumed: a second
// thread installs an override and the launching thread checks whether it can
// see it.  Runs once, on the main thread, before any job starts.
bool ProbePerJobPools() {
    static const bool ok = [] {
        engine::tape::ThreadPool probe(1);
        std::promise<void> installed, checked;
        std::future<void> installed_f = installed.get_future();
        std::future<void> checked_f = checked.get_future();
        std::thread t([&] {
            engine::tape::ThreadPool* saved = engine::tape::consumer_pool_override();
            engine::tape::set_consumer_pool_override(&probe);
            installed.set_value();
            checked_f.wait();
            engine::tape::set_consumer_pool_override(saved);
        });
        installed_f.wait();
        const bool leaked = (engine::tape::consumer_pool_override() == &probe);
        checked.set_value();
        t.join();
        return !leaked;   // thread-local iff the launching thread did NOT see it
    }();
    return ok;
}

// RAII: the coset/tau construction lock (see (2) above).
using TauLock = std::lock_guard<std::mutex>;

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

double secs(TimePoint a, TimePoint b) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

// ===========================================================================
// FNV-1a-64 -- unified (hecke-extensions WP-2): shared with
// Golden/GoldenHarness.cpp and PerformanceSmokeValidation.cpp through
// src/ModArithmetic/Fnv.h, so a campaign point that also lies on
// the golden grid reproduces the frozen reference hash by construction
// (CampaignSmokeValidation asserts exactly that).
// ===========================================================================
constexpr uint64_t kFnvOffset = engine::kFnvOffset64;
using engine::FnvU64;

uint64_t HashScalar(uint64_t v) { return FnvU64(kFnvOffset, v); }

template<class Elt>
uint64_t HashMatrixInto(uint64_t h, const MatrixT<Elt>& M) {
    h = FnvU64(h, M.rows);
    h = FnvU64(h, M.cols);
    for (uint32_t r = 0; r < M.rows; ++r)
        for (uint32_t c = 0; c < M.cols; ++c)
            h = FnvU64(h, static_cast<uint64_t>(M.data[r][c]));
    return h;
}

template<class Elt>
uint64_t HashFlatInto(uint64_t h, const FlatMatrixT<Elt>& M) {
    h = FnvU64(h, M.get_rows());
    h = FnvU64(h, M.get_cols());
    for (uint32_t r = 0; r < M.get_rows(); ++r)
        for (uint32_t c = 0; c < M.get_cols(); ++c)
            h = FnvU64(h, static_cast<uint64_t>(M.get_entry(r, c)));
    return h;
}

template<class Elt>
uint64_t HashPolyInto(uint64_t h, const hecke::FpPolyT<Elt>& P) {
    h = FnvU64(h, P.coeffs.size());
    for (Elt c : P.coeffs) h = FnvU64(h, static_cast<uint64_t>(c));
    return h;
}

// The same bytes as HashFlatInto over Theta^{(q)}_flat, but read a chunk of
// rows at a time off the BLOCK model -- so the (N_q*d_V)^2 dense matrix is
// never resident. Identical value to the resident path by construction: both
// feed rows, cols, then the entries in row-major order.
template<class Elt>
uint64_t HashThetaBlockStreamed(const engine::BlockMat<Elt>& M, uint32_t chunk_rows) {
    uint64_t h = kFnvOffset;
    const uint32_t rows = M.rows(), cols = M.cols();
    h = FnvU64(h, rows);
    h = FnvU64(h, cols);
    if (rows == 0 || cols == 0) return h;
    if (chunk_rows == 0) chunk_rows = 1;
    chunk_rows = std::min(chunk_rows, rows);
    engine::DenseBuf<Elt> buf(chunk_rows, cols, engine::Layout::RowMajor);
    for (uint32_t r0 = 0; r0 < rows; r0 += chunk_rows) {
        const uint32_t nr = std::min(chunk_rows, rows - r0);
        engine::DenseView<Elt> win(buf, 0, nr, 0, cols);
        engine::blk::materialise_rows<Elt>(M, r0, nr, win);
        for (uint32_t i = 0; i < nr; ++i)
            for (uint32_t c = 0; c < cols; ++c)
                h = FnvU64(h, static_cast<uint64_t>(win.at(i, c)));
    }
    return h;
}

// ===========================================================================
// Level-(a) streaming source over the block model (audit item 7.19).
// The writer pulls rows 0, 1, 2, ... in order; this materialises them
// `chunk` at a time straight off Theta_block(q).
// ===========================================================================
class ThetaBlockRowSource : public hecke::LevelARowSource {
public:
    ThetaBlockRowSource(const engine::BlockMat<uint32_t>& M, uint32_t chunk_rows)
        : M_(&M), rows_(M.rows()), cols_(M.cols()),
          chunk_(std::max<uint32_t>(1u, std::min(chunk_rows ? chunk_rows : 1u,
                                                 std::max<uint32_t>(1u, M.rows())))),
          buf_(chunk_, std::max<uint32_t>(1u, M.cols()), engine::Layout::RowMajor) {}

    uint32_t rows() const override { return rows_; }
    uint32_t cols() const override { return cols_; }

    void row(uint32_t r, std::vector<uint32_t>& out) const override {
        if (r >= rows_) throw std::out_of_range("ThetaBlockRowSource: row out of range");
        if (!have_ || r < r0_ || r >= r0_ + nr_) fill(r - (r % chunk_));
        out.assign(cols_, 0u);
        const engine::DenseView<const uint32_t> win(buf_, 0, nr_, 0, cols_);
        for (uint32_t c = 0; c < cols_; ++c) out[c] = win.at(r - r0_, c);
    }

private:
    void fill(uint32_t r0) const {
        nr_ = std::min(chunk_, rows_ - r0);
        engine::DenseView<uint32_t> win(buf_, 0, nr_, 0, cols_);
        engine::blk::materialise_rows<uint32_t>(*M_, r0, nr_, win);
        r0_ = r0;
        have_ = true;
    }

    const engine::BlockMat<uint32_t>* M_;
    uint32_t rows_ = 0, cols_ = 0, chunk_ = 1;
    mutable engine::DenseBuf<uint32_t> buf_;
    mutable uint32_t r0_ = 0, nr_ = 0;
    mutable bool have_ = false;
};

// ===========================================================================
// Small helpers
// ===========================================================================

template<class Elt> Elt ModAddElt(Elt a, Elt b, Elt p);
template<> uint32_t ModAddElt<uint32_t>(uint32_t a, uint32_t b, uint32_t p) {
    return ModArithmetic::mod_add(a, b, p);
}
template<> uint64_t ModAddElt<uint64_t>(uint64_t a, uint64_t b, uint64_t p) {
    return ModArithmetic::mod_add64(a, b, p);
}

template<class Elt>
uint64_t TraceOf(const MatrixT<Elt>& A) {
    Elt t = 0;
    for (uint32_t i = 0; i < A.rows && i < A.cols; ++i) t = ModAddElt<Elt>(t, A.data[i][i], A.p);
    return static_cast<uint64_t>(t);
}

template<class Elt>
uint64_t RankOf(const MatrixT<Elt>& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    MatrixT<Elt> c = A;
    return c.rank();
}

template<class Elt>
hecke::FpPolyT<Elt> CharPolyOf(const MatrixT<Elt>& A) {
    if (A.rows == 0) return hecke::FpPolyT<Elt>(A.p, std::vector<Elt>{Elt(1)});
    return hecke::CharPolyBerkowitz(A);
}

std::string Q(uint32_t q) { return "^" + std::to_string(q); }

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
std::string Fmt(double v, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

// Peak resident set size in MiB (VmHWM), and the reset that scopes it to one
// job: `ResetPeakRss()` writes /proc/self/clear_refs so the next PeakRssMiB()
// is THIS job's own peak rather than the process high-water mark -- best
// effort; on a kernel that does not support it the driver reports scope
// "process" instead of pretending the number is per-job.  (slot-7 WP-8, gap
// G4b) The one definition is Util/Rss.h, shared with the bench driver and the
// console; the bodies moved verbatim.
using oc_util::PeakRssMiB;
using oc_util::ResetPeakRss;

// JSON string escaping (hand-rolled writer; repo convention, no third-party
// library anywhere in the output path).
std::string JsonEsc(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned>(c) & 0xffu);
                    out += b;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

// ===========================================================================
// The object collector -- golden vocabulary (point / object / tier / rows /
// cols / hash), so campaign output and GoldenReference.inc speak one language.
// ===========================================================================
class Collector {
public:
    void scalar(const std::string& pt, const std::string& obj, char tier, uint64_t v) {
        objs_.push_back(CanonicalObject{pt, obj, tier, v, 0, HashScalar(v)});
    }
    template<class Elt>
    void matrix(const std::string& pt, const std::string& obj, char tier, const MatrixT<Elt>& M) {
        objs_.push_back(CanonicalObject{pt, obj, tier, M.rows, M.cols, HashMatrixInto(kFnvOffset, M)});
    }
    template<class Elt>
    void poly(const std::string& pt, const std::string& obj, char tier, const hecke::FpPolyT<Elt>& P) {
        objs_.push_back(CanonicalObject{pt, obj, tier, P.coeffs.size(), 1,
                                        HashPolyInto(kFnvOffset, P)});
    }
    void raw(const std::string& pt, const std::string& obj, char tier, uint64_t rows, uint64_t cols,
             uint64_t h) {
        objs_.push_back(CanonicalObject{pt, obj, tier, rows, cols, h});
    }
    std::vector<CanonicalObject>& objects() { return objs_; }

private:
    std::vector<CanonicalObject> objs_;
};

// charpoly / rank / trace (tier A) + the raw matrix (tier B) of a
// cohomology-level operator -- Golden/GoldenHarness.cpp's
// CollectOperatorMatrices, name for name.
template<class Elt>
void CollectOperatorMatrices(Collector& C, const std::string& pt, const std::string& name,
                             const std::vector<MatrixT<Elt>>& A,
                             std::vector<std::string>* charpoly_text) {
    for (uint32_t q = 0; q < A.size(); ++q) {
        const hecke::FpPolyT<Elt> cp = CharPolyOf(A[q]);
        C.poly(pt, "charpoly_" + name + Q(q), 'A', cp);
        C.scalar(pt, "rank_" + name + Q(q), 'A', RankOf(A[q]));
        C.scalar(pt, "trace_" + name + Q(q), 'A', TraceOf(A[q]));
        C.matrix(pt, name + Q(q), 'B', A[q]);
        if (charpoly_text) {
            std::ostringstream os;
            for (size_t i = 0; i < cp.coeffs.size(); ++i) {
                if (i) os << ";";
                os << static_cast<uint64_t>(cp.coeffs[i]);
            }
            if (charpoly_text->size() <= q) charpoly_text->resize(q + 1);
            (*charpoly_text)[q] = os.str();
        }
    }
}

// Basis-independent invariants of a decomposition -- GoldenHarness's
// CollectDecomposition, tier-A half (the tier-B change of basis and the
// RNG-driven witness data are deliberately not campaign output).
void CollectDecompositionInvariants(Collector& C, const std::string& pt, const std::string& name,
                                    const hecke::HeckeModuleDecomposition& D, uint32_t certify_rc) {
    uint64_t h = FnvU64(kFnvOffset, D.blocks().size());
    for (const hecke::PrimaryBlock& b : D.blocks()) {
        h = FnvU64(h, b.dim);
        h = FnvU64(h, b.residue_degree);
        h = FnvU64(h, b.char_poly.size());
        for (const auto& kv : b.char_poly) { h = FnvU64(h, kv.first); h = HashPolyInto(h, kv.second); }
        for (const auto& kv : b.min_poly) { h = FnvU64(h, kv.first); h = HashPolyInto(h, kv.second); }
        for (const auto& kv : b.jordan_partition) {
            h = FnvU64(h, kv.first);
            h = FnvU64(h, kv.second.size());
            for (uint32_t part : kv.second) h = FnvU64(h, part);
        }
    }
    C.raw(pt, name + "_invariants", 'A', D.blocks().size(), 0, h);
    C.scalar(pt, name + "_certify", 'A', certify_rc);
}

void EmitObjectsJson(std::ostream& os, const std::vector<CanonicalObject>& objs) {
    os << "\"objects\":[";
    for (size_t i = 0; i < objs.size(); ++i) {
        const CanonicalObject& o = objs[i];
        if (i) os << ",";
        os << "{\"point\":\"" << JsonEsc(o.point) << "\",\"object\":\"" << JsonEsc(o.object)
           << "\",\"tier\":\"" << o.tier << "\",\"rows\":" << o.rows << ",\"cols\":" << o.cols
           << ",\"hash\":\"" << HexU64(o.hash) << "\"}";
    }
    os << "]";
}

// ===========================================================================
// The per-job pipeline, element-width generic.
//
// Its shape follows Golden/GoldenHarness.cpp's RunCell32 / RunCell64 so that
// point strings, object names and therefore hashes coincide with the frozen
// reference wherever the grids overlap.
//
// STREAMING. Everything small (dims, hashes, certificate counters, timings) is
// collected in memory and emitted first; the one big thing -- the HeckeReport
// with its level (a) -- is written straight into `os` by
// hecke::StreamHeckeReportJson, so the report never exists as a string.
// ===========================================================================
template<class Elt>
void RunJobPipeline(const CampaignSpec& spec, const Job& job, JobResult& res, std::ostream& os,
                    std::vector<std::string>& charpoly_text,
                    const std::string& output_dir, const ExtCellSummary* ext_cell) {
    constexpr bool kIs64 = std::is_same<Elt, uint64_t>::value;
    const TimePoint t0 = Clock::now();

    const hecke::SimplicialModelDescriptor* desc = hecke::FindModelDescriptor(job.model);
    if (desc == nullptr) throw std::runtime_error("campaign: unknown model '" + job.model + "'");

    // Serialised (Phase 6): SL2ZTessellationTables is built from the certified
    // FP-filtered sign predicates, whose global PredicateStats counters are not
    // atomic (audit 02 Sec. 6.5, hazards 1-3).  Under `jobs = 1` the lock is
    // uncontended and this is the pre-Phase-6 code path.
    std::unique_ptr<const SimplicialComplex> K_owned;
    std::unique_ptr<const hecke::SL2ZTessellationTables> model_owned;
    {
        TauLock tau_lk(hecke::tau_construction_mutex());
        K_owned.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        hecke::RequireModel(*desc, K_owned.get(), "campaign");   // the descriptor, not 7/11/5
        model_owned.reset(new hecke::SL2ZTessellationTables(*K_owned));
    }
    const SimplicialComplex& K = *K_owned;
    const hecke::SL2ZTessellationTables& model = *model_owned;

    const Elt ell = static_cast<Elt>(job.ell);
    const uint32_t n = job.n_sym;
    const uint32_t d_V = n + 1;

    auto rels = GenerateSL2ZRelationsElt<Elt>(K, n, ell);
    LocalSystemSimplicialComplexT<Elt> lssc(K, std::move(rels), d_V, ell);   // adopted (05 Sec. 7 item 4)
    const TimePoint t1 = Clock::now();
    res.t_setup = secs(t0, t1);

    EquivariantCohomologyOnKT<Elt> coh(&lssc);
    coh.compute_cohomology();
    const TimePoint t2 = Clock::now();
    res.t_cohomology = secs(t1, t2);
    res.dim_H = coh.get_H_eq_dimensions();

    Collector C;
    const std::string pt = kIs64
        ? ("n=" + std::to_string(n) + " ell=" + std::to_string(job.ell) + " [u64]")
        : job.golden_point_ne();
    const std::string pt_r1 = pt + " p=" + std::to_string(job.p) + " r=1";
    const std::string pt_rr = pt + " p=" + std::to_string(job.p) + " r=" + std::to_string(job.r);

    C.scalar(pt, "num_degrees", 'A', res.dim_H.size());
    for (uint32_t q = 0; q < res.dim_H.size(); ++q)
        C.scalar(pt, "dim_H" + Q(q), 'A', res.dim_H[q]);

    // --- The r = 1 pipeline. ALWAYS run, whatever the job's r is: it is what
    // --- produces Theta^{(q)} and the in-process certificates C3-C5, which
    // --- are never sampled.
    // Serialised (Phase 6), for the same reason as the tessellation tables
    // above and under the SAME process-wide lock HeckeOperatorFamily has
    // always used: tau's build is first-match-wins over a fixed order and
    // drives the non-atomic PredicateStats.  t_cosets therefore also carries
    // this job's wait for the lock when `jobs > 1`.
    std::unique_ptr<hecke::HeckeCosetTablesT<Elt>> cosets_owned;
    std::unique_ptr<hecke::HeckeTauTableT<Elt>> tau_owned;
    TimePoint t3, t4;
    {
        TauLock tau_lk(hecke::tau_construction_mutex());
        cosets_owned.reset(new hecke::HeckeCosetTablesT<Elt>(job.p, /*r=*/1, ell, n));
        t3 = Clock::now();
        // C3: HeckeTauTable's constructor CERTIFIES d.tau = tau.d exactly, or throws.
        tau_owned.reset(new hecke::HeckeTauTableT<Elt>(K, model, *cosets_owned, ell, n));
        t4 = Clock::now();
    }
    hecke::HeckeCosetTablesT<Elt>& cosets = *cosets_owned;
    hecke::HeckeTauTableT<Elt>& tau = *tau_owned;
    res.t_cosets = secs(t2, t3);
    res.tau_certified = true;
    res.t_tau = secs(t3, t4);

    hecke::HeckeOperatorOnKT<Elt> op(&coh, &cosets, &tau);
    op.compute();   // C4 (O12(a)) + assertions (i)/(ii)/(iii) + C5 (A-N3 residual)
    const TimePoint t5 = Clock::now();
    res.t_theta = secs(t4, t5);
    {
        const auto cert = op.certificates();
        res.routes_bit_identical = cert.routes_bit_identical;
        res.relation_checks = cert.relation_checks;
        res.cocycle_checks = cert.cocycle_checks;
        res.theta_eq_solves = cert.theta_eq_solves;
        res.theta_eq_residual_max = static_cast<uint64_t>(cert.theta_eq_residual_max);
    }

    // --- Level (a): Theta^{(q)}_flat hash, and its rank under the size gate.
    // Lean mode reads the hash block-row-wise off Theta_block(q); the resident
    // mode materialises get_Theta_flat_matrices(). Identical hashes either way
    // (asserted by CampaignSmokeValidation against the golden reference).
    uint32_t max_dim = 0;
    for (uint32_t q : job.degrees)
        max_dim = std::max(max_dim, K.get_num_q_simplices(q) * d_V);
    const bool rank_gate_ok = max_dim <= spec.theta_rank_max_dim;

    for (uint32_t q : job.degrees) {
        const engine::BlockMat<Elt>& Tb = op.Theta_block(q);
        const uint64_t hash = spec.lean
            ? HashThetaBlockStreamed<Elt>(Tb, d_V)
            : HashFlatInto<Elt>(kFnvOffset, op.get_Theta_flat_matrices()[q]);
        C.raw(pt_r1, "Theta_flat" + Q(q), 'A', Tb.rows(), Tb.cols(), hash);
        if (rank_gate_ok) {
            FlatMatrixT<Elt> rk = op.get_Theta_flat_matrices()[q];
            C.scalar(pt_r1, "rank_Theta_flat" + Q(q), 'A',
                     (rk.get_rows() == 0 || rk.get_cols() == 0) ? 0 : rk.rank());
        }
    }

    // --- Level (b): A^{(q)}(T_p), r = 1.
    const std::vector<MatrixT<Elt>> A1 = op.get_H_matrices();
    CollectOperatorMatrices<Elt>(C, pt_r1, "A(T_p)", A1, job.r == 1 ? &charpoly_text : nullptr);

    res.external_selected = job.external_certify;
    const TimePoint t6 = Clock::now();

    // --- r >= 2: the STANDARD (full) T_{p^r} by the algebra recursion. The
    // --- independent direct double-coset route is the sampled external check.
    // --- 32-bit only: HeckeOperatorFamily / HeckeCrossChecks are uint32_t.
    std::vector<MatrixT<Elt>> A_r;
    if constexpr (!kIs64) {
        if (job.r >= 2) {
            hecke::HeckeOperatorFamily fam(&coh, K, model, static_cast<uint32_t>(job.ell), n);
            A_r = fam.T_p_pow(job.p, job.r);
            CollectOperatorMatrices<Elt>(C, pt_rr,
                                         "A(T_p^" + std::to_string(job.r) + ")_rec", A_r,
                                         &charpoly_text);
            if (job.external_certify) {
                const TimePoint x0 = Clock::now();
                const std::vector<MatrixT<Elt>> dir = fam.T_p_pow_direct(job.p, job.r);
                bool agree = (dir.size() == A_r.size());
                for (size_t q = 0; q < A_r.size() && agree; ++q) {
                    if (dir[q].rows != A_r[q].rows || dir[q].cols != A_r[q].cols) agree = false;
                    else if (!(dir[q] == A_r[q])) agree = false;
                }
                res.external_ran = true;
                if (!agree) res.external_rc |= 0x2u;
                res.external_note += std::string(res.external_note.empty() ? "" : "; ") +
                                     "T_p_pow_direct " + (agree ? "agrees" : "DISAGREES");
                res.t_external += secs(x0, Clock::now());
            } else {
                res.external_note += std::string(res.external_note.empty() ? "" : "; ") +
                                     "T_p_pow_direct skipped (sampled)";
            }
        }
    } else {
        if (job.r >= 2)
            throw std::runtime_error("campaign: r >= 2 needs ell < 2^32 "
                                     "(HeckeOperatorFamily is the 32-bit instantiation)");
    }
    res.t_family = secs(t6, Clock::now()) - res.t_external;

    // --- External cross-check: the classical alpha/beta/omega factorisation.
    if constexpr (!kIs64) {
        if (job.external_certify) {
            const TimePoint x0 = Clock::now();
            // Elt == uint32_t in this branch (`if constexpr` discards it for
            // uint64_t), and EquivariantCohomologyOnK / HeckeCosetTables /
            // HeckeTauTable / HeckeOperatorOnK ARE the uint32_t instantiations.
            hecke::HeckeCrossChecks xc(&coh, &cosets, &tau);
            const uint32_t rc = xc.run_all(op);
            res.external_ran = true;
            if (rc != 0) res.external_rc |= 0x1u;
            res.external_note += std::string(res.external_note.empty() ? "" : "; ") +
                                 "HeckeCrossChecks rc=" + std::to_string(rc);
            res.t_external += secs(x0, Clock::now());
        } else {
            res.external_note += std::string(res.external_note.empty() ? "" : "; ") +
                                 "HeckeCrossChecks skipped (sampled)";
        }
    } else {
        res.external_note = "n/a (ell >= 2^32: the cross-check layer is 32-bit)";
    }

    // --- Level (c) + the report, per reported degree with d_q > 0.
    // 32-bit only (HeckeModuleDecomposition / HeckeReport are uint32_t).
    // STREAMED FIRST, so that the one big thing in the file is written as it
    // is produced and the (small) object list can still be completed by the
    // decomposition invariants collected here.
    const TimePoint t7 = Clock::now();
    os << "\"reports\":[";
    bool first_report = true;
    if constexpr (!kIs64) {
        for (uint32_t q : job.degrees) {
            if (q >= A1.size() || A1[q].rows == 0) continue;
            std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops = {{job.p, A1[q]}};
            hecke::HeckeModuleDecomposition D(ops, static_cast<uint32_t>(job.ell));
            D.compute();
            const uint32_t rc = D.certify();          // C6, unconditional
            if (rc == 0) ++res.decomp_certified; else ++res.decomp_certify_failures;
            CollectDecompositionInvariants(C, pt_r1, "decomp_A(T_p)" + Q(q), D, rc);

            hecke::HeckeReportInput in;
            in.ell = static_cast<uint32_t>(job.ell);
            in.n_sym = n;
            in.degree = q;
            // (E0b, slot 2 WP-2) The four oc1a fields of the EMBEDDED report
            // are per complex and derived from the complex's own degree count
            // -- A1.size(), the bulk model's degree list -- and NOT from
            // `job.degrees`, which is the REPORTED degree selection that
            // Job::id() carries ([S2] Sec. 2.3 item 1).
            in.num_degrees = static_cast<uint32_t>(A1.size());
            in.level_b = {{job.p, A1[q]}};
            in.decomp = &D;

            // Level (a): streamed off the block model in lean mode, resident
            // otherwise; omitted above the size gate. Same bytes either way.
            const engine::BlockMat<uint32_t>& Tb = op.Theta_block(q);
            const uint64_t entries = static_cast<uint64_t>(Tb.rows()) * Tb.cols();
            const bool want_a = (spec.level_a == LevelAMode::Always) ||
                                (spec.level_a == LevelAMode::Auto &&
                                 entries <= spec.level_a_max_entries);
            ThetaBlockRowSource src(Tb, d_V);
            if (want_a && spec.lean) {
                in.level_a_stream[job.p] = &src;
                res.level_a_state = "streamed";
            } else if (want_a) {
                in.level_a_theta_flat[job.p] = op.get_Theta_flat_matrices()[q];
                res.level_a_state = "resident";
            } else if (spec.level_a == LevelAMode::Never) {
                // (E0b, slot 2 WP-2 / [I G1]) It is the MODE, not the size
                // gate: `level_a = never` omitted level (a) whatever the size,
                // and the pre-E0b text blamed a gate that was never consulted.
                res.level_a_state = "omitted (level_a = never; " + std::to_string(entries) +
                                    " entries)";
            } else {
                res.level_a_state = "omitted (size gate: " + std::to_string(entries) +
                                    " > " + std::to_string(spec.level_a_max_entries) + " entries)";
            }

            if (!first_report) os << ",";
            first_report = false;
            os << "{\"degree\":" << q << ",\"level_a\":\"" << JsonEsc(res.level_a_state)
               << "\",\"report\":";
            hecke::StreamHeckeReportJson(os, in);      // the streamed part
            os << "}";
        }
    }
    os << "],";
    res.t_report = secs(t7, Clock::now());

    // (hecke-extensions slot 5, WP-2 / E2 cont.) THE EXT RECORD, on the cell's
    // FIRST job in plan order ([XA m14]; [X] Sec. 1.5 "the per-job JSON gains
    // `extensions`").  The engine itself ran on the campaign's MAIN thread,
    // before the job loop -- see CampaignExt.h for why it may not run here --
    // so this is a verbatim copy of the standalone record the pass wrote, plus
    // the path it lives at (from which its state sidecars' relative paths are
    // taken).  A non-owner job emits nothing at all, so the pre-WP-2 file is
    // byte-identical whenever `ext = false`.
    if (ext_cell != nullptr) StreamJobExtensions(os, output_dir, *ext_cell);

    // Now the small stuff: dimensions and the complete canonical-object list
    // (the decomposition invariants above included).
    os << "\"dim_H\":[";
    for (size_t q = 0; q < res.dim_H.size(); ++q) { if (q) os << ","; os << res.dim_H[q]; }
    os << "],";

    // (hecke-extensions phase E0b, slot 2 WP-2) THE oc1a FIELDS AT JOB LEVEL
    // ([C] `sec:recordschema` item 5 as revised in v1.4.0; slot-2 owner
    // confirmation 1). They describe THE COMPLEX this job ran on, are derived
    // from its own degree list, and are emitted HERE -- beside `dim_H`, which
    // is that list -- rather than inside `point` in the header, because the
    // header is written before the cohomology runs and a failed job must
    // still produce a well-formed file. `point.degrees` keeps its own,
    // different meaning: the REPORTED degree selection that Job::id() carries
    // ([S2] Sec. 2.3 item 1). Every embedded `reports[].report` carries the
    // same four fields itself, per complex, at `hecke_report/v2`.
    // Un-hashed: the campaign collector has no text object.
    os << "\"q_max\":" << (res.dim_H.empty() ? 0u
                                             : static_cast<uint32_t>(res.dim_H.size() - 1))
       << ",\"truncated\":false,\"q_max_source\":\"derived\",";
    EmitObjectsJson(os, C.objects());
    os << ",";                       // the shared tail follows (see RunOneJob)
    res.objects = C.objects();

    res.t_total = secs(t0, Clock::now());
}

// ===========================================================================
// One job: file handling, the header/trailer, error capture.
// ===========================================================================
// `log`: nullptr means "print progress straight to std::cout as it happens"
// (the sequential path, byte for byte the pre-Phase-6 behaviour); otherwise the
// progress line is buffered there and the caller prints it in group order.
// `allow_rss_reset`: false under concurrent jobs -- /proc/self/clear_refs
// resets the PROCESS high-water mark, so with several jobs in flight a per-job
// reset would attribute another job's peak to this one.  The number reported
// is then the process VmHWM observed at this job's end, and `peak_rss_scope`
// says so; run a group with `jobs = 1` when a per-job figure is what is wanted.
JobResult RunOneJob(const CampaignSpec& spec, const Job& job, const std::string& path,
                    bool verbose, bool allow_rss_reset, std::ostream* log,
                    const std::string& output_dir = std::string(),
                    const ExtCellSummary* ext_cell = nullptr) {
    // (slot-9 WP-8, C5c; decision 6 (A)) Under a non-default coefficient_ring
    // the job writes hecke_ring_report/v1 instead of campaign_job/v1
    // (Campaign/CampaignRing.h, [E] sec:ring (g)).  Spec validation has
    // refused `ext = true` under the ring, so `ext_cell` is null here.  A
    // fingerprint-0 job never takes this branch: its file is byte-unchanged.
    if (spec.math.coefficient_ring.k >= 2)
        return RunRingJob(spec, job, path, verbose, allow_rss_reset, log);
    JobResult res;
    res.job = job;
    res.output_path = path;
    res.lean = spec.lean;

    // (hecke-extensions slot 4, WP-1 / E2a) The MATHEMATICAL settings this job
    // runs under ([X] Sec. 1.5, [OD] R1).  The guard is thread_local and is
    // propagated onto every thread engine::tape::run_concurrently() starts, so
    // concurrent jobs cannot see each other's settings and an inner fan-out
    // never silently falls back to the defaults.  At the defaults this
    // installs the defaults, so nothing moves.
    const hecke::ScopedMathSettings math_guard(spec.math);

    const bool per_job_rss = allow_rss_reset && ResetPeakRss();
    res.peak_rss_scope = per_job_rss ? "job"
                       : (allow_rss_reset ? "process" : "process-concurrent");

    // The header is identical on the success and the failure path, so a failed
    // job's file is still well-formed JSON (it is rewritten from scratch).
    auto write_header = [&](std::ostream& o) {
        o << "{\"schema\":\"campaign_job/v1\",";
        o << "\"campaign\":\"" << JsonEsc(spec.name) << "\",";
        o << "\"index\":" << job.index << ",";
        o << "\"point\":{\"n_sym\":" << job.n_sym << ",\"ell\":" << job.ell
          << ",\"p\":" << job.p << ",\"r\":" << job.r
          << ",\"model\":\"" << JsonEsc(job.model) << "\",\"degrees\":[";
        for (size_t i = 0; i < job.degrees.size(); ++i) { if (i) o << ","; o << job.degrees[i]; }
        o << "]},";
        const hecke::SimplicialModelDescriptor* d = hecke::FindModelDescriptor(job.model);
        o << "\"model_descriptor\":{\"name\":\"" << JsonEsc(job.model)
          << "\",\"dimension\":" << (d ? d->dimension : 0) << ",\"simplex_counts\":[";
        if (d) for (size_t i = 0; i < d->simplex_counts.size(); ++i) {
            if (i) o << ",";
            o << d->simplex_counts[i];
        }
        o << "],\"relation_source\":\"" << JsonEsc(d ? d->relation_source : std::string())
          << "\"},";
        // ONE source of truth for the string (it used to be spelled out here
        // and silently dropped the `q=` component that the group signature in
        // index.json carries -- fixed 2026-09-16, campaign.md Sec. 4).
        o << "\"shape_signature\":\"" << JsonEsc(job.shape().to_string()) << "\",";
        // (slot-4 WP-1 / E2a) [X] Sec. 1.5 rule 2: a setting is RECORDED,
        // never inferred.  The block carries every registered mathematical key
        // with its value and source, and the fingerprint the job id uses.
        o << "\"settings\":" << spec.math.to_json() << ",";
        o << "\"lean\":" << (spec.lean ? "true" : "false") << ",";
    };

    // The certify / timings / status tail, also shared by both paths.
    auto write_tail = [&](std::ostream& o) {
        o << "\"certify\":{\"mode\":\"" << JsonEsc(spec.certify.mode_name())
          << "\",\"spec\":\"" << JsonEsc(spec.certify.text)
          << "\",\"every_k\":" << spec.certify.every_k
          << ",\"in_process\":{\"note\":\"C3-C6 are never sampled\","
          << "\"tau_certified\":" << (res.tau_certified ? "true" : "false")
          << ",\"routes_bit_identical\":" << (res.routes_bit_identical ? "true" : "false")
          << ",\"relation_checks\":" << res.relation_checks
          << ",\"cocycle_checks\":" << res.cocycle_checks
          << ",\"theta_eq_solves\":" << res.theta_eq_solves
          << ",\"theta_eq_residual_max\":" << res.theta_eq_residual_max
          << ",\"decomposition_certified\":" << res.decomp_certified
          << ",\"decomposition_certify_failures\":" << res.decomp_certify_failures
          << "},\"external\":{\"selected\":" << (res.external_selected ? "true" : "false")
          << ",\"ran\":" << (res.external_ran ? "true" : "false")
          << ",\"rc\":" << res.external_rc
          << ",\"note\":\"" << JsonEsc(res.external_note) << "\"}},";
        o << "\"level_a\":\"" << JsonEsc(res.level_a_state) << "\",";
        o << "\"timings_s\":{\"setup\":" << Fmt(res.t_setup, 6)
          << ",\"cohomology\":" << Fmt(res.t_cohomology, 6)
          << ",\"cosets\":" << Fmt(res.t_cosets, 6)
          << ",\"tau\":" << Fmt(res.t_tau, 6)
          << ",\"theta\":" << Fmt(res.t_theta, 6)
          << ",\"family\":" << Fmt(res.t_family, 6)
          << ",\"external\":" << Fmt(res.t_external, 6)
          << ",\"report\":" << Fmt(res.t_report, 6)
          << ",\"total\":" << Fmt(res.t_total, 6) << "},";
        o << "\"peak_rss_mb\":" << Fmt(res.peak_rss_mb, 1)
          << ",\"peak_rss_scope\":\"" << res.peak_rss_scope << "\",";
        o << "\"status\":\"" << (res.ok ? "ok" : "failed") << "\"";
        if (!res.ok) o << ",\"error\":\"" << JsonEsc(res.error) << "\"";
        // The completion TRAILER: `resume` looks for exactly this, and only a
        // SUCCESSFUL job carries it -- a failed or half-written one is re-run.
        o << ",\"campaign_job_complete\":" << (res.ok ? "true" : "false") << "}\n";
    };

    std::vector<std::string> charpoly_text;
    {
        std::ofstream os(path, std::ios::out | std::ios::trunc);
        if (!os) throw std::runtime_error("campaign: cannot open '" + path + "' for writing");
        write_header(os);
        try {
            if (job.ell <= 0xffffffffull)
                RunJobPipeline<uint32_t>(spec, job, res, os, charpoly_text, output_dir, ext_cell);
            else
                RunJobPipeline<uint64_t>(spec, job, res, os, charpoly_text, output_dir, ext_cell);
            res.ok = true;
        } catch (const std::exception& e) {
            res.error = e.what();
            res.ok = false;
        }
        res.ran = true;
        res.charpoly = charpoly_text;
        if (ext_cell != nullptr) { res.ext = *ext_cell; res.ext_owner = true; }
        res.peak_rss_mb = PeakRssMiB();
        if (res.ok) {
            write_tail(os);
            os.flush();
            if (!os) throw std::runtime_error("campaign: write failed on '" + path + "'");
        }
    }
    if (!res.ok) {
        // Rewrite the file from scratch: the streamed body was cut off
        // mid-object, so only a fresh header + the tail is well-formed JSON.
        std::ofstream os(path, std::ios::out | std::ios::trunc);
        if (!os) throw std::runtime_error("campaign: cannot open '" + path + "' for writing");
        write_header(os);
        os << "\"reports\":[],";
        if (ext_cell != nullptr) StreamJobExtensions(os, output_dir, *ext_cell);
        os << "\"dim_H\":[],\"objects\":[],";
        write_tail(os);
        os.flush();
        if (!os) throw std::runtime_error("campaign: write failed on '" + path + "'");
    }

    if (verbose) {
        std::ostringstream line;
        line << "  [" << (res.ok ? "ok  " : "FAIL") << "] " << job.to_string() << "  dims=(";
        for (size_t i = 0; i < res.dim_H.size(); ++i) {
            if (i) line << ",";
            line << res.dim_H[i];
        }
        line << ")  " << Fmt(res.t_total, 3) << " s  " << Fmt(res.peak_rss_mb, 1) << " MB"
             << (res.external_ran ? "  [external]" : "") << "\n";
        if (!res.ok) line << "        error: " << res.error << "\n";
        if (log != nullptr) (*log) << line.str();
        else                std::cout << line.str() << std::flush;
    }
    return res;
}

// ===========================================================================
// THE BATCHING SEAM (Phase 6 Part A).
//
// A shape group's jobs share (n_sym, p, r, degrees, model) and differ only in
// the modulus, so they are independent pipelines of identical shape -- audit
// Sec. 4 axis (e).  `spec.jobs` of them run at a time, each on its own engine
// pool of `spec.threads` workers; `spec.jobs == 1` is the pre-Phase-6 loop,
// reached by a separate branch so the sequential path is unchanged down to the
// order its progress lines are printed.
//
// DETERMINISM.  Results are written into a pre-sized slot vector indexed by
// the job's position in the group and appended to `out` in GROUP ORDER, so the
// campaign's job order, index.json, summary.tsv and every hash are independent
// of how the work was scheduled -- CampaignSmokeValidation pins exactly that
// (batched == sequential == 20 individual campaigns, hash for hash).
// ===========================================================================
void RunJobList(const CampaignSpec& spec, const std::vector<Job>& jobs_in,
                const std::string& jobs_dir, bool verbose, CampaignResult& out,
                const std::string& output_dir = std::string(),
                const ExtPassResult* ext = nullptr) {
    // (slot-5 WP-2) The cell record that job `j` owns, or null.  A non-owner
    // and a campaign without `ext = true` both get null, which is what makes
    // the whole ext surface invisible at the default.
    auto ext_of = [&](const Job& j) -> const ExtCellSummary* {
        if (ext == nullptr || !ext->enabled || !j.ext_owner) return nullptr;
        return ext->find(j.n_sym, j.ell);
    };
    // `resume` skips a complete job file -- EXCEPT when the ext pass CONTINUED
    // this cell's record, in which case the owner's file must be rewritten so
    // that its `"extensions"` carries the resumed record ([X] R3.3 item 4).
    auto may_skip = [&](const Job& j) { return !j.ext_force_rerun; };
    const std::vector<Job>& group_jobs = jobs_in;
    const size_t n = group_jobs.size();

    auto make_skip = [&](const Job& job, const std::string& path) {
        JobResult r;
        r.job = job;
        r.ran = false;
        r.ok = true;
        r.output_path = path;
        return r;
    };

    const uint32_t want = std::max<uint32_t>(1, spec.jobs);

    // ---- the sequential path: unchanged from Phase 6's first cut ----------
    if (want == 1) {
        for (const Job& job : group_jobs) {
            const std::string path = jobs_dir + "/" + job.id() + ".json";
            if (spec.resume && may_skip(job) && JobOutputComplete(path)) {
                out.jobs.push_back(make_skip(job, path));
                ++out.num_skipped;
                if (verbose) std::cout << "  [skip] " << job.to_string() << " (resume)" << std::endl;
                continue;
            }
            JobResult r = RunOneJob(spec, job, path, verbose, /*allow_rss_reset=*/true, nullptr,
                                    output_dir, ext_of(job));
            ReleaseEngineProgramCaches(kJobLoopReleaseThreshold);   // slot-8 WP-5, as below
            if (!r.ok) ++out.num_failed;
            ++out.num_ran;
            out.jobs.push_back(std::move(r));
        }
        return;
    }

    // ---- the concurrent path ---------------------------------------------
    struct Slot {
        std::string path;
        bool skip = false;
        JobResult res;
        std::string log;
    };
    std::vector<Slot> slots(n);
    std::vector<size_t> todo;
    todo.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const Job& job = group_jobs[i];
        slots[i].path = jobs_dir + "/" + job.id() + ".json";
        if (spec.resume && may_skip(job) && JobOutputComplete(slots[i].path)) {
            slots[i].skip = true;
            slots[i].res = make_skip(job, slots[i].path);
            slots[i].log = "  [skip] " + job.to_string() + " (resume)\n";
        } else {
            todo.push_back(i);
        }
    }

    if (!todo.empty()) {
        const size_t conc = std::min<size_t>(want, todo.size());
        std::atomic<size_t> next{0};
        // Captured HERE, on the launching thread, so every job thread inherits
        // the campaign's memory-lean setting and any scoped consumer options.
        const std::vector<engine::tape::ThreadContextInstaller> ctx =
            engine::tape::capture_thread_context();
        const bool per_job_pools = (spec.threads > 0) && ProbePerJobPools();

        auto worker = [&]() {
            std::vector<engine::tape::ThreadContextGuard> guards;
            guards.reserve(ctx.size());
            for (const engine::tape::ThreadContextInstaller& install : ctx)
                guards.push_back(install());
            // This job thread's private engine pool: parallel_for() serialises
            // concurrent callers, so sharing one pool would serialise the
            // group.  Falls back to the process pool when the override is not
            // thread-local (probed above) -- correct either way, just slower.
            std::unique_ptr<engine::tape::ThreadPool> pool;
            std::unique_ptr<engine::tape::ScopedConsumerPool> pool_guard;
            if (per_job_pools) {
                pool.reset(new engine::tape::ThreadPool(spec.threads));
                pool_guard.reset(new engine::tape::ScopedConsumerPool(pool.get()));
            }
            for (;;) {
                const size_t k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= todo.size()) break;
                const size_t i = todo[k];
                std::ostringstream log;
                try {
                    slots[i].res = RunOneJob(spec, group_jobs[i], slots[i].path, verbose,
                                             /*allow_rss_reset=*/false, &log,
                                             output_dir, ext_of(group_jobs[i]));
                } catch (const std::exception& e) {
                    // RunOneJob only throws for an I/O error on the job file;
                    // a job that THREW is already recorded as failed inside it.
                    slots[i].res.job = group_jobs[i];
                    slots[i].res.output_path = slots[i].path;
                    slots[i].res.ran = true;
                    slots[i].res.ok = false;
                    slots[i].res.error = e.what();
                    log << "  [FAIL] " << group_jobs[i].to_string() << "\n        error: "
                        << e.what() << "\n";
                }
                slots[i].log = log.str();
                // (slot-8 WP-5, [KI] Sec. 4c) the between-jobs release: the
                // job loop holds the ext pass's growth (~28 programs per job;
                // WP-5 report Sec. 3), released at a pinned size threshold
                // (ProgramCacheRelease.h says why not after every job).  A
                // concurrent job whose program is dropped here recompiles it
                // on its next miss -- memory-safe and value-free.
                ReleaseEngineProgramCaches(kJobLoopReleaseThreshold);
            }
            pool_guard.reset();
            pool.reset();
            guards.clear();
        };

        std::vector<std::thread> threads;
        threads.reserve(conc - 1);
        for (size_t w = 1; w < conc; ++w) threads.emplace_back(worker);
        worker();                                   // the calling thread joins in
        for (std::thread& t : threads) t.join();
    }

    for (size_t i = 0; i < n; ++i) {
        if (verbose && !slots[i].log.empty()) std::cout << slots[i].log << std::flush;
        if (slots[i].skip) { ++out.num_skipped; }
        else               { if (!slots[i].res.ok) ++out.num_failed; ++out.num_ran; }
        out.jobs.push_back(std::move(slots[i].res));
    }
}

// The per-group entry point named by campaign.md Sec. 2: one shape group
// through the same runner.  Kept because a group IS the natural unit -- its
// jobs are shape-uniform and differ only in the modulus -- and because it is
// what a caller wanting group-at-a-time execution asks for.  RunCampaignSpec()
// itself submits the WHOLE plan at once (see there).
void RunJobGroup(const CampaignSpec& spec, const JobGroup& group, const std::string& jobs_dir,
                 bool verbose, CampaignResult& out, const std::string& output_dir = std::string(),
                 const ExtPassResult* ext = nullptr) {
    RunJobList(spec, group.jobs, jobs_dir, verbose, out, output_dir, ext);
}

void WriteIndexJson(const CampaignSpec& spec, const CampaignResult& res, const std::string& path,
                    const ExtPassResult* ext) {
    std::ofstream os(path, std::ios::out | std::ios::trunc);
    if (!os) throw std::runtime_error("campaign: cannot write '" + path + "'");
    os << "{\"schema\":\"campaign_index/v1\",";
    os << "\"name\":\"" << JsonEsc(spec.name) << "\",";
    os << "\"model\":\"" << JsonEsc(spec.model) << "\",";
    os << "\"spec_origin\":\"" << JsonEsc(spec.origin) << "\",";
    // (slot-9 WP-8) [E] sec:ring (g): ONE top-level key, in a ring campaign
    // only, so a fingerprint-0 index.json is byte-unchanged.
    if (spec.math.coefficient_ring.k >= 2)
        os << "\"coefficient_ring\":\"" << JsonEsc(hecke::CoefficientRingText(spec.math.coefficient_ring.k))
           << "\",";
    os << "\"settings\":{\"certify\":\"" << JsonEsc(spec.certify.text)
       << "\",\"certify_mode\":\"" << JsonEsc(spec.certify.mode_name())
       << "\",\"every_k\":" << spec.certify.every_k
       << ",\"lean\":" << (spec.lean ? "true" : "false")
       << ",\"resume\":" << (spec.resume ? "true" : "false")
       << ",\"jobs\":" << res.jobs_concurrency
       << ",\"threads\":" << spec.threads
       << ",\"per_job_pools\":" << (res.per_job_pools ? "true" : "false")
       << ",\"threads_note\":\"Phase 6 Part A: `jobs` concurrent jobs per shape group, each "
          "on its own engine pool of `threads` workers (threads = 0: the process-wide ThreadPool "
          "sized by ENGINE_THREADS is shared). per_job_pools = false means the consumer-pool "
          "override is not thread-local, so the pool is shared and the parallel_for regions of "
          "concurrent jobs serialise.\""
       << ",\"level_a_max_entries\":" << spec.level_a_max_entries
       << ",\"theta_rank_max_dim\":" << spec.theta_rank_max_dim << "},";
    // (slot-5 WP-2) Emitted ONLY when the ext pass ran, so a default
    // index.json is byte-unchanged.
    if (ext != nullptr && ext->enabled)
        os << "\"extensions\":" << ExtIndexJson(spec, *ext) << ",";
    os << "\"groups\":[";
    for (size_t g = 0; g < res.groups.size(); ++g) {
        if (g) os << ",";
        os << "{\"signature\":\"" << JsonEsc(res.groups[g].sig.to_string()) << "\",\"jobs\":[";
        for (size_t j = 0; j < res.groups[g].jobs.size(); ++j) {
            if (j) os << ",";
            os << "\"" << JsonEsc(res.groups[g].jobs[j].id()) << "\"";
        }
        os << "]}";
    }
    os << "],\"jobs\":[";
    for (size_t i = 0; i < res.jobs.size(); ++i) {
        const JobResult& r = res.jobs[i];
        if (i) os << ",";
        os << "{\"index\":" << r.job.index << ",\"id\":\"" << JsonEsc(r.job.id())
           << "\",\"n_sym\":" << r.job.n_sym << ",\"ell\":" << r.job.ell << ",\"p\":" << r.job.p
           << ",\"r\":" << r.job.r
           << ",\"status\":\"" << (!r.ran ? "skipped" : (r.ok ? "ok" : "failed"))
           << "\",\"external_certify\":" << (r.job.external_certify ? "true" : "false")
           << ",\"file\":\"jobs/" << JsonEsc(r.job.id()) << ".json\"}";
    }
    os << "],\"totals\":{\"jobs\":" << res.jobs.size() << ",\"ran\":" << res.num_ran
       << ",\"skipped\":" << res.num_skipped << ",\"failed\":" << res.num_failed
       << ",\"wall_s\":" << Fmt(res.wall_s, 3)
       << ",\"jobs_per_minute\":" << Fmt(res.jobs_per_minute, 2)
       << ",\"peak_rss_mb\":" << Fmt(res.peak_rss_mb, 1)
       << ",\"peak_rss_scope\":\"campaign process VmHWM\"}}\n";
    if (!os) throw std::runtime_error("campaign: write failed on '" + path + "'");
}

// summary.tsv -- one row per job. This is the plan's "published table of new
// char-polys": charpoly_A<q> holds the coefficients of the characteristic
// polynomial of A^{(q)}(T_p) (or of the recursion's T_{p^r} for r >= 2),
// LOW -> HIGH degree, semicolon separated (amendment A-N4's order).
void WriteSummaryTsv(const CampaignSpec& spec, const CampaignResult& res, const std::string& path,
                     bool ext_enabled) {
    // (slot-9 WP-8) [E] sec:ring (g): a ring campaign keeps the 21 columns,
    // writes `-` in the ten field-valued ones and APPENDS fifteen; a
    // fingerprint-0 campaign writes neither change.
    const bool ring = spec.math.coefficient_ring.k >= 2;
    std::ofstream os(path, std::ios::out | std::ios::trunc);
    if (!os) throw std::runtime_error("campaign: cannot write '" + path + "'");
    os << "index\tn_sym\tell\tp\tr\tmodel\tstatus\tdim_H0\tdim_H1\tdim_H2"
          "\tcharpoly_A0\tcharpoly_A1\tcharpoly_A2\ttrace_A1\trank_A1"
          "\ttheta_flat_hash_q1\tcertify\texternal\ttime_s\trss_mb\tfile";
    // (slot-5 WP-2 / E2 cont.) [X] Sec. 1.5's eight ext columns, APPENDED and
    // only when `ext = true` -- the `q=` / `_q` precedent, so the pinned
    // default form (21 columns, time_s at index 18 and rss_mb at 19) is
    // untouched and every published table stands.
    if (ext_enabled) os << ExtSummaryHeader();
    if (ring) os << RingSummaryHeader();
    os << "\n";
    for (const JobResult& r : res.jobs) {
        // Pull the scalar/poly values back out of the object list by name.
        auto find_obj = [&](const std::string& name) -> const CanonicalObject* {
            for (const CanonicalObject& o : r.objects)
                if (o.object == name) return &o;
            return nullptr;
        };
        const std::string suffix = (r.job.r == 1)
            ? "A(T_p)" : ("A(T_p^" + std::to_string(r.job.r) + ")_rec");
        // The coefficients themselves (low -> high), not a hash: summary.tsv IS
        // the published table.
        auto poly_text = [&](uint32_t q) -> std::string {
            if (q < r.charpoly.size() && !r.charpoly[q].empty()) return r.charpoly[q];
            return "-";
        };
        const CanonicalObject* tr = find_obj("trace_" + suffix + "^1");
        const CanonicalObject* rk = find_obj("rank_" + suffix + "^1");
        const CanonicalObject* th = find_obj("Theta_flat^1");
        if (ring) {
            os << r.job.index << "\t" << r.job.n_sym << "\t" << r.job.ell << "\t" << r.job.p << "\t"
               << r.job.r << "\t" << r.job.model << "\t"
               << (!r.ran ? std::string("skipped") : r.ring_status)
               << "\t-\t-\t-\t-\t-\t-\t-\t-\t-\t"          // dim_H0..2, charpoly_A0..2, trace, rank, theta
               << (r.job.external_certify ? "full" : "sampled-out") << "\t-\t"   // certify, external
               << Fmt(r.t_total, 4) << "\t" << Fmt(r.peak_rss_mb, 1) << "\t"
               << "jobs/" << r.job.id() << ".json" << RingSummaryColumns(spec, r) << "\n";
            continue;
        }
        os << r.job.index << "\t" << r.job.n_sym << "\t" << r.job.ell << "\t" << r.job.p << "\t"
           << r.job.r << "\t" << r.job.model << "\t"
           << (!r.ran ? "skipped" : (r.ok ? "ok" : "failed")) << "\t"
           << (r.dim_H.size() > 0 ? std::to_string(r.dim_H[0]) : "-") << "\t"
           << (r.dim_H.size() > 1 ? std::to_string(r.dim_H[1]) : "-") << "\t"
           << (r.dim_H.size() > 2 ? std::to_string(r.dim_H[2]) : "-") << "\t"
           << poly_text(0) << "\t" << poly_text(1) << "\t" << poly_text(2) << "\t"
           << (tr ? std::to_string(tr->rows) : "-") << "\t"
           << (rk ? std::to_string(rk->rows) : "-") << "\t"
           << (th ? HexU64(th->hash) : "-") << "\t"
           << (r.job.external_certify ? "full" : "sampled-out") << "\t"
           << (r.external_ran ? "ran" : "skipped") << "\t"
           << Fmt(r.t_total, 4) << "\t" << Fmt(r.peak_rss_mb, 1) << "\t"
           << "jobs/" << r.job.id() << ".json";
        if (ext_enabled) os << ExtSummaryRow(r.ext_owner ? &r.ext : nullptr);
        os << "\n";
    }
    if (!os) throw std::runtime_error("campaign: write failed on '" + path + "'");
}

} // namespace

// (slot-8 WP-6) The probe, public: the parallel ext pass gives each worker its
// own engine pool under the same condition the job loop does.
bool PerJobPoolsSupported() { return ProbePerJobPools(); }

// ===========================================================================
// Public entry points
// ===========================================================================

bool JobOutputComplete(const std::string& path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size <= 0) return false;
    const std::streamoff want = std::min<std::streamoff>(size, 256);
    in.seekg(size - want, std::ios::beg);
    std::string tail(static_cast<size_t>(want), '\0');
    in.read(&tail[0], want);
    return tail.find("\"campaign_job_complete\":true") != std::string::npos;
}

CampaignSpec SingleJobSpec(const CampaignSpec& spec, const Job& job,
                           const std::string& output_dir) {
    CampaignSpec s = spec;
    s.output_dir = output_dir;
    s.resume = false;
    s.sections.clear();
    SpecSection sec;
    sec.n_sym = {job.n_sym};
    sec.ell = {job.ell};
    sec.p = {job.p};
    sec.r = {job.r};
    sec.q = job.degrees;
    s.sections.push_back(sec);
    // The job's own external-certify decision must be preserved: a single-job
    // spec restarts the index at 0, which under `sampled` would otherwise
    // always select. Pin the mode instead of re-deriving it.
    if (!job.external_certify) {
        s.certify.mode = CertifyMode::Sampled;
        s.certify.every_k = 0;          // selects() is false for every index
        s.certify.text = spec.certify.text + " [pinned: not selected]";
    } else {
        s.certify.mode = CertifyMode::Full;
        s.certify.every_k = 1;
        s.certify.text = spec.certify.text + " [pinned: selected]";
    }
    return s;
}

CampaignResult RunCampaignSpec(const CampaignSpec& spec, bool verbose) {
    namespace fs = std::filesystem;
    // (slot-22 L23-1) never run the Hecke job grid of an orbifold spec silently.
    if (spec.orbifold.enabled)
        throw std::runtime_error("campaign: an `orbifold` spec runs through RunOrbifoldCampaign (CampaignOrbifold.h)");
    // (slot-24 D6) nor the job grid of a coefficient_ring = Z spec (its cells are descent records).
    if (spec.math.coefficient_ring.is_integral())
        throw std::runtime_error("campaign: a coefficient_ring = Z spec runs through RunDescentCampaignCli (CampaignDescent.h)");
    CampaignResult res;
    res.output_dir = spec.output_dir;
    res.groups = plan_campaign(spec);
    res.jobs_concurrency = std::max<uint32_t>(1, spec.jobs);
    res.threads_per_job = spec.threads;
    res.per_job_pools = (spec.threads > 0) && (res.jobs_concurrency > 1) && ProbePerJobPools();

    std::error_code ec;
    fs::create_directories(spec.output_dir, ec);
    const std::string jobs_dir = spec.output_dir + "/jobs";
    fs::create_directories(jobs_dir, ec);
    if (!fs::is_directory(spec.output_dir) || !fs::is_directory(jobs_dir))
        throw std::runtime_error("campaign: cannot create output directory '" + spec.output_dir + "'");

    const size_t total_jobs = flatten(res.groups).size();
    if (verbose) {
        std::cout << "=== campaign '" << spec.name << "' ===\n";
        std::cout << "  model      = " << spec.model << "\n";
        std::cout << "  output     = " << spec.output_dir << "\n";
        std::cout << "  certify    = " << spec.certify.text
                  << " (in-process certificates C3-C6 always run)\n";
        std::cout << "  lean       = " << (spec.lean ? "true" : "false") << "\n";
        std::cout << "  resume     = " << (spec.resume ? "true" : "false") << "\n";
        std::cout << "  jobs       = " << res.jobs_concurrency << " concurrent per shape group\n";
        std::cout << "  threads    = " << spec.threads
                  << (spec.threads == 0 ? " (0: the process-wide pool, ENGINE_THREADS)" : " per job")
                  << ((res.jobs_concurrency > 1 && spec.threads > 0 && !res.per_job_pools)
                          ? "  [per-job pools UNAVAILABLE: the consumer-pool override is not "
                            "thread-local; the process pool is shared]" : "")
                  << "\n";
        std::cout << "  " << total_jobs << " jobs in " << res.groups.size()
                  << " shape groups\n";
        if (spec.math.coefficient_ring.k >= 2)
            std::cout << "  ring       = " << hecke::CoefficientRingText(spec.math.coefficient_ring.k)
                      << " (job files hecke_ring_report/v1)\n";
        if (spec.ext.ext)
            std::cout << "  ext        = true (family = "
                      << (spec.ext.family_text.empty() ? std::string("default")
                                                       : spec.ext.family_text)
                      << ", ext_groups = " << spec.ext.groups_text
                      << ", sign = " << spec.ext.sign_text << ")\n";
    }

    // Memory-lean mode is a process-wide switch on the O12(a) comparison; the
    // guard restores the previous value when the campaign finishes.
    hecke::HeckeOperatorOptions::ScopedMemoryLean lean_guard(spec.lean);

    const TimePoint t0 = Clock::now();

    // ---- (slot-5 WP-2 / E2 cont.) THE EXT PASS -------------------------
    // One HeckeExtDriver per (n_sym, ell) CELL ([XA m14]), before the job
    // loop and never on a job thread -- CampaignExt.h gives the reasons.
    // Running it here makes the ext half of the output independent of
    // `jobs x threads` BY CONSTRUCTION, which is the seam's determinism
    // contract.  (slot-8 WP-6) `ext_jobs` cells run at once, on the pass's
    // own workers (CampaignExt.h, "THE PARALLEL PASS"); `ext_jobs = 1` is the
    // sequential pass on this thread.
    //
    // The mathematical settings must be in force while the pass runs: it
    // reads the `u_ell` budget from them (never from the spec text) and the
    // driver records them in every cell record.
    ExtPassResult ext_pass;
    if (spec.ext.ext) {
        const hecke::ScopedMathSettings math_guard(spec.math);
        const std::vector<Job> planned = flatten(res.groups);
        ext_pass = RunExtPass(spec, planned, spec.output_dir, verbose);
        // A resumed cell's owner job must be rewritten even under `resume`.
        for (JobGroup& g : res.groups)
            for (Job& j : g.jobs) {
                for (const std::string& id : ext_pass.force_rerun_job_ids)
                    if (id == j.id()) j.ext_force_rerun = true;
            }
        if (verbose) {
            std::cout << "  ext: " << ext_pass.cells.size() << " cells, " << ext_pass.num_ran
                      << " ran, " << ext_pass.num_reused << " reused, " << ext_pass.num_resumed
                      << " resumed, " << ext_pass.num_failed << " unavailable, "
                      << Fmt(ext_pass.wall_s, 2) << " s" << std::endl;
            // (slot-8 WP-6) the pass's timing line, which T-P reads.
            std::cout << ExtPassLine(ext_pass) << std::endl;
        }
    }

    if (res.jobs_concurrency == 1) {
        // Sequential: group at a time, with the group banner, exactly as before.
        for (const JobGroup& g : res.groups) {
            if (verbose) std::cout << "--- group " << g.sig.to_string() << " (" << g.jobs.size()
                                   << " jobs) ---" << std::endl;
            RunJobGroup(spec, g, jobs_dir, verbose, res, spec.output_dir, &ext_pass);
        }
    } else {
        // Concurrent: submit the WHOLE plan as one ordered work queue rather
        // than a group at a time.  Running group by group imposes a join
        // barrier at every group boundary, and a group is only as wide as its
        // number of moduli -- 9 in the shipped sweeps -- so a single expensive
        // job (the sampled external cross-check, ~5x its neighbours) sat on
        // every group's critical path and held the other 8 threads idle.
        // MEASURED on the n_sym = 42..200 sweep: group-at-a-time gave a mean
        // concurrency of 2.4 out of 9.
        //
        // Nothing is lost by crossing group boundaries: the jobs of a group
        // share only their SHAPE, and the one thing that could exploit that --
        // the engine's ProgramCache -- is process-global and keyed by the
        // prime anyway (ENGINE_API Sec. 10.9), so it is shared across groups
        // just as well.  flatten() preserves the plan's group order, and
        // results are still emitted in that order, so index.json, summary.tsv
        // and every hash are unchanged.  Peak RSS is the price: jobs of
        // different n_sym are now in flight together.
        RunJobList(spec, flatten(res.groups), jobs_dir, verbose, res, spec.output_dir,
                   &ext_pass);
    }
    res.wall_s = secs(t0, Clock::now());
    res.peak_rss_mb = PeakRssMiB();
    res.jobs_per_minute = res.wall_s > 0.0
        ? (static_cast<double>(res.num_ran) / res.wall_s) * 60.0 : 0.0;
    res.ext_enabled  = ext_pass.enabled;
    res.ext_cells    = static_cast<uint64_t>(ext_pass.cells.size());
    res.ext_ran      = ext_pass.num_ran;
    res.ext_reused   = ext_pass.num_reused;
    res.ext_resumed  = ext_pass.num_resumed;
    res.ext_failed   = ext_pass.num_failed;
    res.ext_wall_s   = ext_pass.wall_s;

    WriteIndexJson(spec, res, spec.output_dir + "/index.json", &ext_pass);
    WriteSummaryTsv(spec, res, spec.output_dir + "/summary.tsv", spec.ext.ext);

    if (verbose) {
        std::cout << "=== campaign done: " << res.num_ran << " ran, " << res.num_skipped
                  << " skipped, " << res.num_failed << " failed, "
                  << Fmt(res.wall_s, 2) << " s ===" << std::endl;
        std::cout << "CAMPAIGN name=" << spec.name << " jobs=" << res.jobs.size()
                  << " ran=" << res.num_ran << " skipped=" << res.num_skipped
                  << " failed=" << res.num_failed << " wall_s=" << Fmt(res.wall_s, 3)
                  << " jobs_per_min=" << Fmt(res.jobs_per_minute, 1)
                  << " concurrency=" << res.jobs_concurrency << "x" << spec.threads
                  << " per_job_pools=" << (res.per_job_pools ? 1 : 0)
                  << " peak_rss_mb=" << Fmt(res.peak_rss_mb, 1)
                  << " out=" << spec.output_dir << std::endl;
    }
    return res;
}

void PrintCampaignUsage(const char* argv0) {
    std::cout <<
        "Usage: " << argv0 << " --campaign <spec-file> [options]\n"
        "\n"
        "Runs a parameter sweep of (n_sym, ell, p, r, q) Hecke jobs through the same\n"
        "public API as `--bench hecke`, writing one streamed JSON per job plus\n"
        "index.json and summary.tsv. Spec grammar and worked examples:\n"
        "  docs/matrix-engine/campaign.md\n"
        "\n"
        "Options:\n"
        "  --output <dir>   override the spec's `output` directory\n"
        "  --jobs <n>       override `jobs`: concurrent jobs per shape group\n"
        "  --threads <n>    override `threads`: engine workers per concurrent job\n"
        "  --dry-run        print the plan (groups and jobs) and exit\n"
        "  --quiet          suppress per-job progress lines\n"
        "  --help           this text\n"
        "\n"
        "Spec example:\n"
        "  name    = t2_sweep\n"
        "  output  = out/t2_sweep\n"
        "  certify = sampled(every 10)\n"
        "  lean    = true\n"
        "  n_sym   = 0..200 step 2\n"
        "  ell     = 5,13,691,10007,134217757\n"
        "  p       = 2\n"
        "  r       = 1\n";
}

int RunCampaign(int argc, char** argv) {
    if (argc < 1 || std::strcmp(argv[0], "--help") == 0 || std::strcmp(argv[0], "-h") == 0) {
        PrintCampaignUsage("hacs");
        return argc < 1 ? 1 : 0;
    }
    std::string spec_path = argv[0];
    std::string output_override;
    int jobs_override = -1, threads_override = -1;
    bool dry_run = false, verbose = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dry-run") dry_run = true;
        else if (a == "--quiet") verbose = false;
        else if (a == "--output" && i + 1 < argc) output_override = argv[++i];
        else if (a == "--jobs" && i + 1 < argc) jobs_override = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads_override = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") { PrintCampaignUsage("hacs"); return 0; }
        else {
            std::cerr << "campaign: unknown option '" << a << "'" << std::endl;
            PrintCampaignUsage("hacs");
            return 1;
        }
    }

    try {
        CampaignSpec spec = ParseCampaignSpecFile(spec_path);
        if (!output_override.empty()) spec.output_dir = output_override;
        if (jobs_override > 0) spec.jobs = static_cast<uint32_t>(jobs_override);
        if (threads_override >= 0) spec.threads = static_cast<uint32_t>(threads_override);
        // (slot-22 L23-1) an orbifold spec writes orbifold_cell/v2 records (v1 before S24 L23-H1) instead of jobs
        // (Campaign/CampaignOrbifold.h); a spec without `orbifold` never reaches it.
        if (spec.orbifold.enabled) return RunOrbifoldCampaignCli(spec, dry_run, verbose);
        // (slot-24 D6) a coefficient_ring = Z spec writes integral_descent_record/v1 cells (Campaign/CampaignDescent.h).
        // (slot-25 HZ-REC) structure = hecke writes integral_hecke_structure/v1 cells (Campaign/CampaignHeckeZ.h).
        if (spec.math.coefficient_ring.is_integral() && spec.structure_hecke) return RunHeckeZCampaignCli(spec, dry_run, verbose);
        if (spec.math.coefficient_ring.is_integral()) return RunDescentCampaignCli(spec, dry_run, verbose);
        if (dry_run) {
            const std::vector<JobGroup> groups = plan_campaign(spec);
            size_t n = 0;
            std::cout << "campaign '" << spec.name << "': " << groups.size() << " shape groups\n";
            for (const JobGroup& g : groups) {
                std::cout << "  group " << g.sig.to_string() << "  (" << g.jobs.size() << " jobs)\n";
                for (const Job& j : g.jobs) {
                    std::cout << "    #" << j.index << "  " << j.to_string()
                              << (j.external_certify ? "  [external cross-checks]" : "")
                              << "\n";
                    ++n;
                }
            }
            std::cout << "  " << n << " jobs total\n";
            return 0;
        }
        const CampaignResult res = RunCampaignSpec(spec, verbose);
        return res.num_failed == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "campaign: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace campaign
