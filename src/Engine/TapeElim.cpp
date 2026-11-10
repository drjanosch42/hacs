// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// TapeElim.cpp -- see TapeElim.h.
#include "TapeElim.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <type_traits>
#include <chrono>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include "../ModArithmetic/BlockKernels.h"
#include "../ModArithmetic/DenseElim.h"
#include "../Cuda/CudaKernels.h"
#include "../ModArithmetic/Kernels.h"
#include "Passes.h"
#include "EngineRuntime.h"   // (slot-8 WP-6b) elim_pass_options(), note_program_run()
#include "EngineCostModel.h"
#include "../ModArithmetic/RetiredDeviceLevers.h"   // (S24 CL-R2) ENGINE_ELIM_SESSION, ENGINE_GPU_ELIM

namespace engine {
namespace tape {

namespace {

using kernels::GemmOp;

// Row-growing row-major buffer (exact copies on growth).
template<class Elt>
class GrowRows {
public:
    explicit GrowRows(uint32_t n) : n_(n) {}
    void reserve(uint32_t rows) {
        if (rows <= cap_) return;
        DenseBuf<Elt> nb(rows, n_);
        if (rows_ > 0 && n_ > 0)
            kernels::copy<Elt>(DenseView<const Elt>(buf_, 0, rows_, 0, n_), DenseView<Elt>(nb, 0, rows_, 0, n_));
        buf_ = std::move(nb);
        cap_ = rows;
    }
    uint32_t rows() const { return rows_; }
    bool fits(uint32_t k) const { return rows_ + k <= cap_; }   // append(k) keeps the storage
    DenseBuf<Elt>& buf() { return buf_; }
    DenseView<Elt> view() { return DenseView<Elt>(buf_, 0, rows_, 0, n_); }
    DenseView<const Elt> cview() const { return DenseView<const Elt>(buf_, 0, rows_, 0, n_); }
    uint32_t append(uint32_t k) {
        if (rows_ + k > cap_) {
            uint32_t nc = std::max<uint32_t>(cap_ * 2, rows_ + k);
            nc = std::max<uint32_t>(nc, 32);
            DenseBuf<Elt> nb(nc, n_);
            if (rows_ > 0 && n_ > 0)
                kernels::copy<Elt>(DenseView<const Elt>(buf_, 0, rows_, 0, n_), DenseView<Elt>(nb, 0, rows_, 0, n_));
            buf_ = std::move(nb);
            cap_ = nc;
        }
        const uint32_t first = rows_;
        rows_ += k;
        return first;
    }
    DenseBuf<Elt> take() {
        if (rows_ == cap_ && cap_ > 0) return std::move(buf_);   // exactly sized (reserved at the rank bound): no copy; rank 0 keeps the (0 x n) shape
        DenseBuf<Elt> out(rows_, n_);
        if (rows_ > 0 && n_ > 0) kernels::copy<Elt>(cview(), engine::view(out));
        return out;
    }
private:
    uint32_t n_, rows_ = 0, cap_ = 0;
    DenseBuf<Elt> buf_;
};

struct PivBlock { uint32_t bc, first; std::vector<uint32_t> local; };
struct RowRun { uint32_t br, lo, hi, dst; };

std::vector<std::vector<uint32_t>> col_rows_index(uint32_t nbr, uint32_t nbc, const std::vector<uint32_t>& row_ptr,
                                                  const std::vector<uint32_t>& col_idx) {
    std::vector<std::vector<uint32_t>> cr(nbc);
    for (uint32_t br = 0; br < nbr; ++br)
        for (uint32_t k = row_ptr[br]; k < row_ptr[br + 1]; ++k) cr[col_idx[k]].push_back(br);
    return cr;
}

template<class Elt>
void account(const Program<Elt>& prog, const PassOptions& opts, const RunResult& rr, TapeElimStats* st) {
    if (st == nullptr) return;
    ++st->programs; st->nodes += prog.live_nodes(); st->tasks += rr.tasks;
    // (slot-9 WP-6) the options the program was compiled under.
    if (!opts.batch) ++st->batch_off_programs;
    if (opts.effective_policy() != DevicePolicy::Cpu) ++st->device_policy_programs;
    if (prog.stats.batches != 0) ++st->batched_programs;
    if (prog.stats.gpu_nodes != 0) { ++st->gpu_programs; st->gpu_nodes += prog.stats.gpu_nodes; }
}

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// (slot-18 WP-19a, GPU G1) The elimination session: residency across the
// per-panel programs of one tape::rref (ENGINE_API.md Sec. 12.8).  Opened only
// under a device policy (effective_policy() != Cpu) with a usable device, so
// the cpu default never creates a CUDA context and never reaches it.  Adopted:
// A's pool blocks (read-only: uploaded once), R (kept: downloaded at release),
// and the per-panel device scratch Lg / T / Lp / C (dropped).  The driver keeps
// the session contract: R is released (downloaded) before every host access
// to it -- its growth copy and take().  ENGINE_ELIM_SESSION=0 is the rollback
// arm (each program binds its own mirrors, as before).  None of it can change
// a value: the bytes on either side are the exact-integer twins (Sec. 11).
// (S24 CL-R2) A retired lever (RetiredDeviceLevers.h): re-read per call as before in an
// OC_ENABLE_LEGACY=ON build; OFF: always on, `0...` refused.
bool elim_session_enabled() {
    return lever::elim_session_lever().enabled();
}
template<class Elt>
class ElimSession {
public:
    ElimSession(Executor<Elt>& ex, const PassOptions& opts) {
        if (opts.effective_policy() == DevicePolicy::Cpu || !elim_session_enabled()) return;
        if (ex.elim_session_begin()) ex_ = &ex;
    }
    ~ElimSession() {
        if (ex_ == nullptr) return;
        try { ex_->elim_session_end(false); } catch (...) {}   // unwinding: the host result is discarded anyway
    }
    ElimSession(const ElimSession&) = delete;
    ElimSession& operator=(const ElimSession&) = delete;
    bool on() const { return ex_ != nullptr; }
    void adopt(const DenseBuf<Elt>& h, bool keep) { if (ex_) ex_->elim_session_adopt(h, keep); }
    void release(const Elt* base) { if (ex_) ex_->elim_session_release(base); }
    void end() {
        if (ex_ == nullptr) return;
        Executor<Elt>* e = ex_;
        ex_ = nullptr;
        e->elim_session_end(true);
    }
private:
    Executor<Elt>* ex_ = nullptr;
};
// Drops the per-panel scratch adoptions when the panel's scope ends (also on
// the `continue` of a pivot-free panel).
template<class Elt>
struct PanelScratch {
    ElimSession<Elt>* s;
    std::vector<const Elt*> bases;
    explicit PanelScratch(ElimSession<Elt>* s_) : s(s_) {}
    void adopt(const DenseBuf<Elt>& h) { if (s->on()) { s->adopt(h, false); bases.push_back(h.data()); } }
    ~PanelScratch() { for (const Elt* b : bases) { try { s->release(b); } catch (...) {} } }
    PanelScratch(const PanelScratch&) = delete;
    PanelScratch& operator=(const PanelScratch&) = delete;
};

std::mutex& totals_mutex() { static std::mutex m; return m; }
TapeElimStats& totals() { static TapeElimStats t; return t; }
void add_totals(const TapeElimStats& s) {
    std::lock_guard<std::mutex> lk(totals_mutex());
    TapeElimStats& t = totals();
    t.panels += s.panels; t.panels_with_pivots += s.panels_with_pivots; t.programs += s.programs;
    t.nodes += s.nodes; t.tasks += s.tasks; t.identity_pivots += s.identity_pivots;
    t.scalar_pivots += s.scalar_pivots; t.general_pivots += s.general_pivots;
    t.s_host += s.s_host; t.s_compile += s.s_compile; t.s_run_p1 += s.s_run_p1; t.s_run_p2 += s.s_run_p2; t.s_serial += s.s_serial;
    t.batch_off_programs += s.batch_off_programs; t.device_policy_programs += s.device_policy_programs; t.batched_programs += s.batched_programs;
    t.gpu_programs += s.gpu_programs; t.gpu_nodes += s.gpu_nodes;
    t.device_elims += s.device_elims;
}

// ---- (slot-22 G2) the device-resident elimination route (TapeElim.h; ENGINE_API.md Sec. 11.9) ----
struct AtomicDeviceElimTotals { std::atomic<uint64_t> tape_rrefs{0}, dense_rrefs{0}, rank_profiles{0}, fallbacks{0}, verified{0}; };
AtomicDeviceElimTotals& dev_totals() { static AtomicDeviceElimTotals t; return t; }
// (audit R1) The fallback is never silent: one stderr line per fallback (reason + shape), and one
// `EFFECTIVE DEVICE_ELIM calls=<n> device=<n> fallbacks=<n>` line per process at exit when any
// device elimination was attempted. Pure diagnostics: nothing here reaches a record, a hash or a value.
// Test-only hook (read-only in effect: the host route runs and returns the same values):
// ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR=1 raises a CudaError where an OOM / refused cooperative launch would.
void print_device_elim_summary_at_exit() {
    const std::string line = device_elim_summary_line();
    if (!line.empty()) { std::cerr << line << std::endl; }
}
void note_device_elim_attempt() {
    static const bool registered = [] { std::atexit(&print_device_elim_summary_at_exit); return true; }();
    (void)registered;
}
bool simulate_cuda_error() {
    const char* e = std::getenv("ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR");
    return e != nullptr && e[0] == '1';
}
void report_fallback(const char* route, uint32_t m, uint32_t n, const char* what) {
    dev_totals().fallbacks.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "[imma_rref] fallback to the host route (" << route << ", " << m << "x" << n << "): " << what << std::endl;
}
// The ENGINE_VERIFY_DEVICE twin for the device elimination: in a -DENGINE_VERIFY_DEVICE=ON build (or with
// ENGINE_GPU_ELIM_VERIFY=1) every device elimination is re-run on the host (under the thread-local Cpu
// override) and compared entrywise -- rank, pivots, the RREF rows, pivot_rows; a difference throws.
bool device_elim_verify() {
    if (ENGINE_VERIFY_DEVICE != 0) return true;
    const char* e = std::getenv("ENGINE_GPU_ELIM_VERIFY");
    return e != nullptr && e[0] == '1';
}
[[noreturn]] void verify_fail(const std::string& what) {
    throw std::logic_error("ENGINE_VERIFY_DEVICE (device elimination): " + what);
}

// The DenseElim hook: A is row-contiguous and Canonical (rref_contiguous / column_rank_profile).
bool dense_device_rref_hook(const ModCtx<uint32_t>& ctx, DenseView<uint32_t> A, bool want_rref, elim::RrefResult& res) {
    const uint32_t m = A.rows(), n = A.cols();
    if (!device_elim_wanted(ctx, m, n)) return false;
    const bool verify = device_elim_verify();
    DenseBuf<uint32_t> host_copy;
    if (verify) {
        host_copy = DenseBuf<uint32_t>(m, n);
        kernels::copy<uint32_t>(DenseView<const uint32_t>(A), engine::view(host_copy));
    }
    bool wrote = false;
    note_device_elim_attempt();
    try {
        if (simulate_cuda_error()) throw cuda::CudaError("simulated CudaError (ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR=1)");
        cuda::ImmaRowSink sink;
        if (want_rref) sink = [&](uint32_t rank) { wrote = true; return A.sub(0, rank, 0, n); };
        const cuda::ImmaRrefResult r = cuda::imma_rref(
            ctx, m, n, [&](uint32_t r0, uint32_t nr) { return DenseView<const uint32_t>(A.sub(r0, nr, 0, n)); }, sink);
        if (want_rref) {
            for (uint32_t i = r.rank; i < m; ++i) {
                uint32_t* row = A.data() + static_cast<size_t>(i) * A.row_stride();
                std::fill(row, row + n, 0u);
            }
        }
        res.rank = r.rank;
        res.pivots = r.pivots;
        if (verify) {
            ScopedThreadDevice cpu(ThreadDevice::Cpu);
            const elim::RrefResult h = want_rref ? elim::rref<uint32_t>(ctx, engine::view(host_copy))
                                                 : elim::column_rank_profile<uint32_t>(ctx, engine::view(host_copy));
            if (h.rank != res.rank || h.pivots != res.pivots) verify_fail("rank / pivots differ (" + std::to_string(m) + "x" + std::to_string(n) + ")");
            if (want_rref && !same_entries(DenseView<const uint32_t>(A), engine::cview(host_copy)))
                verify_fail("the RREF differs (" + std::to_string(m) + "x" + std::to_string(n) + ")");
            dev_totals().verified.fetch_add(1, std::memory_order_relaxed);
        }
    } catch (const cuda::CudaError& e) {
        if (wrote) throw;   // A partly overwritten: no silent recovery
        report_fallback("DenseElim", m, n, e.what());
        return false;
    }
    (want_rref ? dev_totals().dense_rrefs : dev_totals().rank_profiles).fetch_add(1, std::memory_order_relaxed);
    return true;
}
[[maybe_unused]] const bool g_dense_hook_installed = [] { elim::set_device_rref_hook(&dense_device_rref_hook); return true; }();

// tape::rref on the device: A's rows materialised in slabs, one cuda::imma_rref; pivot_rows in the
// driver's order -- per block column (panel) j, the rows that supplied its pivots, ascending (the
// row rank profile of that panel's Schur complement, PIVOT_SEARCH). With the smallest-original-row
// rule the set of rows chosen up to any column is the row rank profile of that column prefix, the
// same set the panel driver selects (both are unique); only the order inside a panel is the
// driver's convention, applied here. Returns false (nothing written) after a CudaError.
template<class Elt>
bool tape_rref_device(const BlockMat<Elt>& A, blk::BlockRrefResult<Elt>& out) {
    if constexpr (!std::is_same<Elt, uint32_t>::value) {
        (void)A; (void)out;
        return false;
    } else {
        const ModCtx<uint32_t>& ctx = A.ctx();
        const uint32_t m = A.rows(), n = A.cols();
        const uint32_t slab = std::max<uint32_t>(1, std::min<uint32_t>(m, static_cast<uint32_t>((size_t(32) << 20) / (static_cast<size_t>(n) * 4))));
        DenseBuf<uint32_t> buf(slab, n);
        cuda::ImmaRrefResult r;
        note_device_elim_attempt();
        try {
            if (simulate_cuda_error()) throw cuda::CudaError("simulated CudaError (ENGINE_GPU_ELIM_SIMULATE_CUDA_ERROR=1)");
            r = cuda::imma_rref(
                ctx, m, n,
                [&](uint32_t r0, uint32_t nr) {
                    blk::materialise_rows<uint32_t>(A, r0, nr, engine::view(buf).sub(0, nr, 0, n));
                    return DenseView<const uint32_t>(engine::cview(buf).sub(0, nr, 0, n));
                },
                [&](uint32_t rank) {
                    out.R = DenseBuf<uint32_t>(rank, n);
                    return engine::view(out.R);
                },
                slab);
            if (r.rank == 0) out.R = DenseBuf<uint32_t>(0, n);
        } catch (const cuda::CudaError& e) {
            report_fallback("tape::rref", m, n, e.what());
            out = blk::BlockRrefResult<Elt>();
            return false;
        }
        out.rank = r.rank;
        out.pivots = r.pivots;
        out.pivot_rows.clear();
        out.pivot_rows.reserve(r.rank);
        uint32_t t = 0;
        for (uint32_t j = 0; j < A.nbc() && t < r.rank; ++j) {
            const uint32_t c1 = A.col_start(j) + A.col_dim(j);
            const uint32_t t0 = t;
            while (t < r.rank && r.pivots[t] < c1) ++t;
            std::vector<uint32_t> rows(r.pivot_rows.begin() + t0, r.pivot_rows.begin() + t);
            std::sort(rows.begin(), rows.end());
            out.pivot_rows.insert(out.pivot_rows.end(), rows.begin(), rows.end());
        }
        ENGINE_CHECK_MSG(t == r.rank, "tape::rref (device): a pivot outside every block column");
        dev_totals().tape_rrefs.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
}

} // namespace

bool gpu_elim_enabled() {
    // re-read per call: the validations switch it per arm. (S24 CL-R2) A retired lever
    // (RetiredDeviceLevers.h): ON build as before; OFF build always on, `0...` refused.
    return lever::gpu_elim_lever().enabled();
}
uint64_t gpu_elim_min_entries() {
    const char* e = std::getenv("ENGINE_GPU_ELIM_MIN");
    if (e == nullptr || e[0] == '\0') return uint64_t(1) << 20;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(e, &end, 10);
    return (end == e) ? (uint64_t(1) << 20) : static_cast<uint64_t>(v);
}
bool device_elim_wanted(const ModCtx<uint32_t>& ctx, uint32_t m, uint32_t n) {
    if (m == 0 || n == 0 || !gpu_elim_enabled()) return false;
    if (!cuda::imma_elim_eligible(ctx)) return false;
    if (static_cast<uint64_t>(m) * n < gpu_elim_min_entries()) return false;
    if (elim_pass_options().effective_policy() != DevicePolicy::Gpu) return false;
    return cuda::available() && cuda::imma_supported();
}
DeviceElimTotals device_elim_totals() {
    DeviceElimTotals o;
    o.tape_rrefs = dev_totals().tape_rrefs.load(); o.dense_rrefs = dev_totals().dense_rrefs.load();
    o.rank_profiles = dev_totals().rank_profiles.load(); o.fallbacks = dev_totals().fallbacks.load();
    o.verified = dev_totals().verified.load();
    return o;
}
std::string device_elim_summary_line() {
    const DeviceElimTotals t = device_elim_totals();
    const uint64_t device = t.tape_rrefs + t.dense_rrefs + t.rank_profiles;
    const uint64_t calls = device + t.fallbacks;
    if (calls == 0) return std::string();
    return "EFFECTIVE DEVICE_ELIM calls=" + std::to_string(calls) + " device=" + std::to_string(device) +
           " fallbacks=" + std::to_string(t.fallbacks);
}

TapeElimStats tape_elim_totals() { std::lock_guard<std::mutex> lk(totals_mutex()); return totals(); }
void reset_tape_elim_totals() { std::lock_guard<std::mutex> lk(totals_mutex()); totals() = TapeElimStats(); }

// See BlockElim.cpp for the invariant proof; this driver executes the same
// left-looking scheme with the heavy steps as tape programs.
template<class Elt>
blk::BlockRrefResult<Elt> rref(Executor<Elt>& ex, const BlockMat<Elt>& A, TapeElimStats* stats_in, uint32_t rank_hint) {
    using Ref = BlockRef<Elt>;
    TapeElimStats local;
    TapeElimStats* stats = &local;
    // (slot-23 P7 audit NIT) ENGINE_TRACE_VRAM attribution of the elimination's blocks; diagnostic only
    struct TraceTag { const char* prev = cuda::trace_tag(); TraceTag() { cuda::set_trace_tag("tape.rref"); } ~TraceTag() { cuda::set_trace_tag(prev); } } trace_tag_scope;
    struct Publish { TapeElimStats* in; TapeElimStats* loc; ~Publish() { add_totals(*loc); if (in) { in->panels += loc->panels; in->panels_with_pivots += loc->panels_with_pivots; in->programs += loc->programs; in->nodes += loc->nodes; in->tasks += loc->tasks; in->identity_pivots += loc->identity_pivots; in->scalar_pivots += loc->scalar_pivots; in->general_pivots += loc->general_pivots; in->s_host += loc->s_host; in->s_compile += loc->s_compile; in->s_run_p1 += loc->s_run_p1; in->s_run_p2 += loc->s_run_p2; in->s_serial += loc->s_serial; in->batch_off_programs += loc->batch_off_programs; in->device_policy_programs += loc->device_policy_programs; in->batched_programs += loc->batched_programs; in->gpu_programs += loc->gpu_programs; in->gpu_nodes += loc->gpu_nodes; in->device_elims += loc->device_elims; } } } publish{stats_in, &local};
    Clock::time_point t_host = Clock::now();
    const ModCtx<Elt>& ctx = A.ctx();
    // ENGINE_API.md Sec. 13.3: field-only (it reaches elim::rref and inverts a
    // scalar pivot on the host); its own entry guard.
    ENGINE_CHECK_MSG(ctx.is_field(), "tape::rref: field-only; use the Z/ell^k entry points of ENGINE_API.md Sec. 13");
    const BlockPool<Elt>& pool = *A.pool();
    // (slot-9 WP-6) the consumers' options: the ScopedPassOptions override or
    // from_env(), with the thread-local device override applied
    // (EngineRuntime.h): exactly from_env() when neither is installed.
    const PassOptions opts = elim_pass_options();
    const uint32_t m = A.rows(), n = A.cols(), nbr = A.nbr(), nbc = A.nbc();
    blk::BlockRrefResult<Elt> out;
    GrowRows<Elt> R(n);
    if (m == 0 || n == 0) { out.R = DenseBuf<Elt>(0, n); return out; }
    if constexpr (std::is_same<Elt, uint32_t>::value) {   // (slot-22 G2) the device-resident route
        if (device_elim_wanted(ctx, m, n)) {
            const Clock::time_point t_d0 = Clock::now();
            if (tape_rref_device<Elt>(A, out)) {
                ++stats->device_elims;
                if (device_elim_verify()) {   // the host driver on the same input, entrywise
                    blk::BlockRrefResult<Elt> h;
                    {
                        ScopedThreadDevice cpu(ThreadDevice::Cpu);
                        h = rref<Elt>(ex, A, nullptr, rank_hint);
                    }
                    if (h.rank != out.rank || h.pivots != out.pivots || h.pivot_rows != out.pivot_rows ||
                        !same_entries(engine::cview(h.R), engine::cview(out.R)) || h.R.rows != out.R.rows)
                        verify_fail("tape::rref " + std::to_string(m) + "x" + std::to_string(n) + ": rank / pivots / pivot_rows / R differ");
                    dev_totals().verified.fetch_add(1, std::memory_order_relaxed);
                }
                stats->s_host += secs(t_host, t_d0);
                stats->s_run_p1 += secs(t_d0, Clock::now());
                if (std::getenv("ENGINE_TRACE_RSS") != nullptr)
                    std::cout << "        [tape rref " << m << "x" << n << " rank " << out.rank << "] device (imma_rref) "
                              << std::fixed << std::setprecision(3) << secs(t_d0, Clock::now()) << " s\n";
                return out;
            }
        }
    }
    if (rank_hint > 0) R.reserve(std::min<uint32_t>({rank_hint, m, n}));
    // (WP-19a) the elimination session: A's pool blocks (read-only) and R.
    ElimSession<Elt> sess(ex, opts);
    if (sess.on()) {
        const uint32_t nblocks = A.row_ptr().empty() ? 0u : A.row_ptr().back();
        for (uint32_t k = 0; k < nblocks; ++k) {
            const Ref& r = A.ref_at(k);
            if (r.has_data()) sess.adopt(pool.buf(r.pool_id), false);
        }
        sess.adopt(R.buf(), true);
    }

    const auto col_rows = col_rows_index(nbr, nbc, A.row_ptr(), A.col_idx());
    std::vector<char> active(m, 1);
    std::vector<uint32_t> active_count(nbr);
    for (uint32_t br = 0; br < nbr; ++br) active_count[br] = A.row_dim(br);
    std::vector<char> touches_pivot(nbr, 0);
    std::vector<PivBlock> pivblocks;
    std::vector<uint32_t>& pivots = out.pivots;
    std::vector<uint32_t> cand_brs, crow;
    std::vector<RowRun> runs;
    std::vector<char> mark(nbr, 0);

    for (uint32_t j = 0; j < nbc; ++j) {
        const uint32_t d = A.col_dim(j);
        if (d == 0) continue;
        if (stats) ++stats->panels;
        const uint32_t rk = static_cast<uint32_t>(pivots.size());
        if (rk == std::min(m, n)) break;

        // 1. candidates (host)
        cand_brs.clear();
        for (uint32_t br : col_rows[j]) if (active_count[br] > 0 && !mark[br]) { mark[br] = 1; cand_brs.push_back(br); }
        if (rk > 0)
            for (uint32_t br = 0; br < nbr; ++br)
                if (touches_pivot[br] && active_count[br] > 0 && !mark[br]) { mark[br] = 1; cand_brs.push_back(br); }
        for (uint32_t br : cand_brs) mark[br] = 0;
        std::sort(cand_brs.begin(), cand_brs.end());
        crow.clear(); runs.clear();
        for (uint32_t br : cand_brs) {
            const uint32_t rs = A.row_start(br), dr = A.row_dim(br);
            uint32_t i = 0;
            while (i < dr) {
                while (i < dr && !active[rs + i]) ++i;
                if (i == dr) break;
                const uint32_t lo = i;
                while (i < dr && active[rs + i]) ++i;
                runs.push_back(RowRun{br, lo, i, static_cast<uint32_t>(crow.size())});
                for (uint32_t t = lo; t < i; ++t) crow.push_back(rs + t);
            }
        }
        const uint32_t nc = static_cast<uint32_t>(crow.size());
        if (nc == 0) continue;
        const uint32_t c0 = A.col_start(j);
        const uint32_t kcap = std::min(nc, d);

        // 2. + 3. program P1: gather S, L_g; S -= L_g R[:, j]; S_c = S; PANEL_RREF; PIVOT_SEARCH
        DenseBuf<Elt> S(nc, d), Sc(nc, d), Lg(nc, std::max<uint32_t>(rk, 1)), Bblk(kcap, kcap);
        PanelScratch<Elt> scratch(&sess);   // (WP-19a) declared after the buffers: released before they are freed
        if (rk > 0) scratch.adopt(Lg);
        uint32_t q_slot, p_slot;
        {
            Recorder<Elt> rec(ctx);
            const BufId bS = rec.output(engine::view(S));
            const BufId bSc = rec.output(engine::view(Sc));
            const BufId bB = rec.output(engine::view(Bblk));
            rec.zero(rec.whole(bS));
            for (const RowRun& ru : runs) {
                const Ref* r = A.find(ru.br, j);
                if (r == nullptr) continue;
                rec.gather_block_rows(pool, *r, A.row_dim(ru.br), d, ru.lo, ru.hi,
                                      rec.view(bS, ru.dst, ru.hi - ru.lo, 0, d));
            }
            if (rk > 0) {
                const BufId bLg = rec.output(engine::view(Lg));
                const BufId bR = rec.input(R.cview());
                rec.zero(rec.whole(bLg));
                for (const PivBlock& pb : pivblocks) {
                    const uint32_t dcp = A.col_dim(pb.bc);
                    const uint32_t cnt = static_cast<uint32_t>(pb.local.size());
                    const uint32_t il = rec.index_list(pb.local);
                    for (const RowRun& ru : runs) {
                        const Ref* r = A.find(ru.br, pb.bc);
                        if (r == nullptr) continue;
                        rec.gather_block_rows(pool, *r, A.row_dim(ru.br), dcp, ru.lo, ru.hi,
                                              rec.view(bLg, ru.dst, ru.hi - ru.lo, pb.first, cnt), false, il);
                    }
                }
                rec.schur_update(rec.view(bLg, 0, nc, 0, rk), rec.view(bR, 0, rk, c0, d), rec.whole(bS));
            }
            rec.scatter(rec.whole(bS), rec.whole(bSc));
            q_slot = rec.panel_rref(rec.whole(bSc));
            p_slot = rec.pivot_search(rec.whole(bS), q_slot, rec.whole(bB));
            Program<Elt> prog = rec.finish();
            const Clock::time_point t_c0 = Clock::now();
            stats->s_host += secs(t_host, t_c0);
            compile(prog, opts);
            const Clock::time_point t_c1 = Clock::now();
            stats->s_compile += secs(t_c0, t_c1);
            note_program_run(prog.stats.gpu_nodes);   // (WP-6b)
            const RunResult rr = ex.run(static_cast<const Program<Elt>&>(prog), rec.bindings());
            t_host = Clock::now();
            stats->s_run_p1 += secs(t_c1, t_host);
            stats->s_serial += rr.serial_seconds;
            account(prog, opts, rr, stats);
            const ResultSlot& qs = rr.slots[q_slot];
            const ResultSlot& ps = rr.slots[p_slot];
            const uint32_t kj = qs.rank;
            if (kj == 0) continue;
            if (stats) ++stats->panels_with_pivots;
            const std::vector<uint32_t> Q = qs.pivots;
            const std::vector<uint32_t> prows = ps.pivots;

            // host: classify the pivot block
            PivotKind kind = PivotKind::General;
            Elt scalar_inv = 1;
            {
                bool ident = true, scal = true;
                const Elt b00 = Bblk.at(0, 0);
                for (uint32_t t = 0; t < kj && (ident || scal); ++t)
                    for (uint32_t u = 0; u < kj; ++u) {
                        const Elt v = Bblk.at(t, u);
                        if (t == u) { if (v != 1) ident = false; if (v != b00) scal = false; }
                        else if (v != 0) { ident = false; scal = false; break; }
                    }
                if (ident) { kind = PivotKind::Identity; if (stats) ++stats->identity_pivots; }
                else if (scal && b00 != 0) { kind = PivotKind::Scalar; scalar_inv = ctx.inv(b00); if (stats) ++stats->scalar_pivots; }
                else if (stats) ++stats->general_pivots;
            }
            DenseBuf<Elt> Binv(kj, kj);
            if (kind == PivotKind::General) {
                const bool ok = elim::inverse<Elt>(ctx, DenseView<const Elt>(Bblk, 0, kj, 0, kj), engine::view(Binv));
                ENGINE_CHECK_MSG(ok, "tape::rref: pivot block not invertible");
            }

            // 4. + 5. program P2: new rows, trailing part, reduce above
            // (WP-19a) R's growth copy reads the host copy: release (download) first.
            if (sess.on() && !R.fits(kj)) sess.release(R.buf().data());
            const uint32_t first = R.append(kj);
            sess.adopt(R.buf(), true);
            const uint32_t trail0 = c0 + d, nt = n - trail0;
            DenseBuf<Elt> T(kj, std::max<uint32_t>(nt, 1)), Lp(kj, std::max<uint32_t>(rk, 1)), C(std::max<uint32_t>(rk, 1), kj);
            PanelScratch<Elt> scratch2(&sess);
            if (nt > 0) scratch2.adopt(T);
            if (rk > 0) { scratch2.adopt(Lp); scratch2.adopt(C); }
            {
                Recorder<Elt> r2(ctx);
                const BufId bR = r2.output(R.view());
                const BufId bSc2 = r2.input(engine::cview(Sc));
                r2.scatter(r2.view(bSc2, 0, kj, 0, d), r2.view(bR, first, kj, c0, d));
                if (nt > 0) {
                    const BufId bT = r2.output(engine::view(T));
                    r2.zero(r2.view(bT, 0, kj, 0, nt));
                    uint32_t t = 0;
                    std::vector<uint32_t> lrows;
                    while (t < kj) {
                        const uint32_t grow = crow[prows[t]];
                        const uint32_t br = A.block_row_of(grow);
                        const uint32_t rs = A.row_start(br), dr = A.row_dim(br);
                        uint32_t u = t;
                        while (u < kj && A.block_row_of(crow[prows[u]]) == br) ++u;
                        // The run's pivot rows, block-local: ONE row-list gather per block.
                        lrows.clear();
                        for (uint32_t v = t; v < u; ++v) lrows.push_back(crow[prows[v]] - rs);
                        const uint32_t rl = r2.index_list(lrows);
                        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
                            const uint32_t bc = A.col_at(k);
                            if (bc <= j) continue;
                            const uint32_t dc = A.col_dim(bc);
                            if (dc == 0) continue;
                            const Ref& r = A.ref_at(k);
                            r2.gather_block_row_list(pool, r, dr, dc, rl, r2.view(bT, t, u - t, A.col_start(bc) - trail0, dc));
                        }
                        t = u;
                    }
                    ViewRef Tv = r2.view(bT, 0, kj, 0, nt);
                    if (rk > 0) {
                        const BufId bLg = r2.input(engine::cview(Lg));
                        const BufId bLp = r2.output(engine::view(Lp));
                        // Lp^T(i, t) = Lg^T(i, prows[t])  -- one column-list GATHER on the transposed views
                        const uint32_t il = r2.index_list(prows);
                        r2.gather(Elt(0), Elt(1), r2.view(bLg, 0, nc, 0, rk, true), 0, il, r2.view(bLp, 0, kj, 0, rk, true));
                        r2.schur_update(r2.view(bLp, 0, kj, 0, rk), r2.view(bR, 0, rk, trail0, nt), Tv);
                    }
                    const BufId bBinv = (kind == PivotKind::General) ? r2.input(engine::cview(Binv)) : kNoBuf;
                    r2.trsm(kind, kind == PivotKind::General ? r2.whole(bBinv) : ViewRef{}, Tv,
                            r2.view(bR, first, kj, trail0, nt), scalar_inv);
                }
                if (rk > 0) {
                    const BufId bC = r2.output(engine::view(C));
                    const uint32_t il = r2.index_list(Q);
                    r2.gather(Elt(0), Elt(1), r2.view(bR, 0, rk, c0, d), 0, il, r2.view(bC, 0, rk, 0, kj));
                    r2.schur_update(r2.view(bC, 0, rk, 0, kj), r2.view(bR, first, kj, c0, n - c0), r2.view(bR, 0, rk, c0, n - c0));
                }
                Program<Elt> p2 = r2.finish();
                const Clock::time_point t_c0 = Clock::now();
                stats->s_host += secs(t_host, t_c0);
                compile(p2, opts);
                const Clock::time_point t_c1 = Clock::now();
                stats->s_compile += secs(t_c0, t_c1);
                note_program_run(p2.stats.gpu_nodes);   // (WP-6b)
                const RunResult rr2 = ex.run(static_cast<const Program<Elt>&>(p2), r2.bindings());
                t_host = Clock::now();
                stats->s_run_p2 += secs(t_c1, t_host);
                account(p2, opts, rr2, stats);
            }

            // 6. bookkeeping
            PivBlock pb; pb.bc = j; pb.first = rk; pb.local = Q;
            pivblocks.push_back(std::move(pb));
            for (uint32_t t = 0; t < kj; ++t) pivots.push_back(c0 + Q[t]);
            for (uint32_t br : col_rows[j]) touches_pivot[br] = 1;
            for (uint32_t t = 0; t < kj; ++t) {
                const uint32_t grow = crow[prows[t]];
                out.pivot_rows.push_back(grow);
                active[grow] = 0;
                --active_count[A.block_row_of(grow)];
            }
        }
    }
    out.rank = static_cast<uint32_t>(pivots.size());
    sess.release(R.buf().data());   // (WP-19a) R's host copy current before take()
    sess.end();
    out.R = R.take();
    stats->s_host += secs(t_host, Clock::now());
    if (std::getenv("ENGINE_TRACE_RSS") != nullptr)
        std::cout << "        [tape rref " << m << "x" << n << " rank " << out.rank << "] panels " << local.panels
                  << " host " << std::fixed << std::setprecision(3) << local.s_host << " compile " << local.s_compile
                  << " p1 " << local.s_run_p1 << " (serial " << local.s_serial << ") p2 " << local.s_run_p2 << " s\n";
    return out;
}

template<class Elt>
BlockMat<Elt> kernel_basis(Executor<Elt>& ex, const BlockMat<Elt>& A, Layout layout) {
    const blk::BlockRrefResult<Elt> res = rref<Elt>(ex, A);
    return blk::kernel_basis<Elt>(A, res, layout);
}

template<class Elt>
BlockMat<Elt> image_basis(Executor<Elt>& ex, const BlockMat<Elt>& A, Layout layout) {
    const blk::BlockRrefResult<Elt> res = rref<Elt>(ex, A);
    return blk::image_basis<Elt>(A, res, layout);
}

namespace {

template<class Elt>
uint32_t row_start_of(const BlockBuilder<Elt>& b, uint32_t br) {
    uint32_t s = 0;
    for (uint32_t i = 0; i < br; ++i) s += b.row_dim(i);
    return s;
}
template<class Elt>
uint32_t add_dense_chunks(BlockBuilder<Elt>& b, typename BlockPool<Elt>::Id id, DenseView<const Elt> D,
                          uint32_t bc0, uint32_t chunk) {
    const uint32_t cols = D.cols();
    uint32_t bc = bc0;
    for (uint32_t c = 0; c < cols; c += chunk, ++bc) {
        const uint32_t w = std::min<uint32_t>(chunk, cols - c);
        for (uint32_t br = 0; br < b.nbr(); ++br) {
            const uint32_t dr = b.row_dim(br);
            if (dr == 0) continue;
            const uint32_t rs = row_start_of(b, br);
            if (BlockMat<Elt>::is_zero_view(D.sub(rs, dr, c, w))) continue;
            b.set(br, bc, BlockRef<Elt>::dense(id, Elt(1), rs, c));
        }
    }
    return bc - bc0;
}
// A borrowed pool entry held for the duration of an elimination: acquire on
// construction, release on destruction.  The augmented BlockMat's windows
// come and go underneath; without this hold its destructor would free the
// entry (refcount 0) and RECYCLE the id, and a later sweep of the stale id
// could free another thread's freshly interned block (the pool is shared by
// the concurrently running degrees of compute_cohomology(): a 1-in-3 abort in
// CohomologyBlockModelValidation before this hold existed).
template<class Elt>
struct BorrowHold {
    std::shared_ptr<BlockPool<Elt>> pool;
    typename BlockPool<Elt>::Id id = BlockPool<Elt>::kNone;
    BorrowHold(std::shared_ptr<BlockPool<Elt>> p, DenseView<const Elt> v) : pool(std::move(p)), id(pool->borrow(v)) { pool->acquire(id); }
    ~BorrowHold() { if (id != BlockPool<Elt>::kNone) pool->release(id); }
    BorrowHold(const BorrowHold&) = delete;
    BorrowHold& operator=(const BorrowHold&) = delete;
};
std::vector<uint32_t> chunk_dims(uint32_t cols, uint32_t chunk) {
    std::vector<uint32_t> d;
    for (uint32_t c = 0; c < cols; c += chunk) d.push_back(std::min<uint32_t>(chunk, cols - c));
    return d;
}

} // namespace

template<class Elt>
blk::FactorBasisResult<Elt> factor_basis(Executor<Elt>& ex, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                                         const std::vector<uint32_t>& row_dims,
                                         DenseView<const Elt> Rl, DenseView<const Elt> G, uint32_t chunk_,
                                         TapeElimStats* stats, bool want_projector) {
    blk::FactorBasisResult<Elt> out;
    uint32_t a = 0;
    for (uint32_t d : row_dims) a += d;
    const uint32_t nr = Rl.cols(), ng = G.cols();
    ENGINE_CHECK_MSG((nr == 0 || Rl.rows() == a) && (ng == 0 || G.rows() == a), "tape::factor_basis: operand height != sum(row_dims)");
    struct StageTrace {
        bool on = std::getenv("ENGINE_TRACE_RSS") != nullptr;
        Clock::time_point t = Clock::now();
        void mark(const char* what, uint32_t a, uint32_t b) {
            if (!on) return;
            const Clock::time_point now = Clock::now();
            std::cout << "        [factor_basis " << a << "x" << b << "] " << what << ": " << std::fixed << std::setprecision(3) << secs(t, now) << " s\n";
            t = now;
        }
    } strace;
    const uint32_t chunk = std::max<uint32_t>(1, chunk_);
    if (ng == 0) {
        out.basis = BlockMat<Elt>::zeros(pool, row_dims, std::vector<uint32_t>{0}, p);
        out.projector = DenseBuf<Elt>(0, a);
        return out;
    }
    const ModCtx<Elt> ctx(p);
    // M[r, c] for a global column c of M = [Rl | G].
    auto M_at = [&](uint32_t r, uint32_t c) -> Elt { return c < nr ? Rl.at(r, c) : G.at(r, c - nr); };

    // ---- Stage 1: RREF of M = [Rl | G] ----------------------------------------
    std::vector<uint32_t> col_dims = chunk_dims(nr, chunk);
    const uint32_t nrc = static_cast<uint32_t>(col_dims.size());
    for (uint32_t d : chunk_dims(ng, chunk)) col_dims.push_back(d);
    blk::BlockRrefResult<Elt> res;
    {
        BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
        std::unique_ptr<BorrowHold<Elt>> holdR;
        if (nr > 0) { holdR = std::make_unique<BorrowHold<Elt>>(pool, Rl); add_dense_chunks<Elt>(b, holdR->id, Rl, 0, chunk); }
        const BorrowHold<Elt> holdG(pool, G);
        add_dense_chunks<Elt>(b, holdG.id, G, nrc, chunk);
        const BlockMat<Elt> aug = b.finalize();
        res = rref<Elt>(ex, aug, stats, /*rank_hint=*/std::min<uint32_t>(a, nr + ng));
    }
    strace.mark("stage 1 rref [Rl | G]", a, nr + ng);
    // (slot-24 G6) Nothing below reads the RREF itself -- only its rank, pivots
    // and row rank profile (stage 2 re-reads M = [Rl | G] through M_at) -- so
    // the rank x (nr + ng) buffer is released here instead of at return
    // (0.5 GB at the C1a cone's q = 2 call, n_sym = 1000).  Memory only.
    res.R = DenseBuf<Elt>();
    for (uint32_t c : res.pivots) {
        if (c < nr) ++out.independent_relations;
        else out.pivots.push_back(c - nr);
    }
    const uint32_t dim_h = static_cast<uint32_t>(out.pivots.size());
    const uint32_t nrl = out.independent_relations;
    DenseBuf<Elt> basis(a, dim_h, Layout::ColMajor);
    for (uint32_t t = 0; t < dim_h; ++t)
        if (a > 0) kernels::copy<Elt>(G.sub(0, a, out.pivots[t], 1), engine::view(basis).sub(0, a, t, 1));
    out.basis = BlockMat<Elt>::from_flat(pool, std::move(basis), row_dims, p, true);
    if (!want_projector) { out.projector = DenseBuf<Elt>(0, 0); return out; }
    out.projector = DenseBuf<Elt>(dim_h, a);
    if (dim_h == 0 || a == 0) return out;

    // ---- Stage 2: the projector rows of the RREF transform U of [M | I] -------
    const uint32_t rho = res.rank;
    const std::vector<uint32_t>& prows = res.pivot_rows;   // pivot order == ascending pivot columns
    const std::vector<uint32_t>& P = res.pivots;
    ENGINE_CHECK_MSG(prows.size() == rho && P.size() == rho, "tape::factor_basis: rank profile lost");
    const uint32_t na = a - rho;
    // The host-side dense steps of this stage (the inverse's trailing updates)
    // run with the elimination's parallel hook bound to the executor's pool.
    const elim::ParallelFor hook = [&](uint32_t nt, const std::function<void(uint32_t)>& fn) {
        ex.pool().parallel_for(nt, kThreadsComputeBound, [&](uint32_t t, uint32_t) { fn(t); });
    };
    const elim::ScopedParallelHook scoped_hook(&hook);
    // (1) U_top0[:, prows] = M[prows, P]^{-1}  (zero elsewhere)
    DenseBuf<Elt> Winv(rho, rho);
    {
        DenseBuf<Elt> Wm(rho, rho);
        for (uint32_t i = 0; i < rho; ++i)
            for (uint32_t j = 0; j < rho; ++j) Wm.at(i, j) = M_at(prows[i], P[j]);
        const bool ok = elim::inverse<Elt>(ctx, cview(Wm), engine::view(Winv));
        ENGINE_CHECK_MSG(ok, "tape::factor_basis: pivot block M[pivot_rows, pivots] not invertible");
    }
    strace.mark("stage 2 inverse of the rank x rank pivot block", rho, rho);
    // The G-pivot rows of U_top0 (rows nrl .. nrl + dim_h - 1 of U), scattered.
    DenseBuf<Elt> U0(dim_h, a);
    for (uint32_t t = 0; t < dim_h; ++t)
        for (uint32_t j = 0; j < rho; ++j) U0.at(t, prows[j]) = Winv.at(nrl + t, j);
    if (na == 0) { out.projector = std::move(U0); return out; }
    // (2) Y = I[act, :] - M[act, P] . U_top0:  Y[:, act] = I, Y[:, prows] = -M[act, P] . Winv
    std::vector<uint32_t> act;
    act.reserve(na);
    {
        std::vector<char> is_piv(a, 0);
        for (uint32_t r : prows) is_piv[r] = 1;
        for (uint32_t r = 0; r < a; ++r) if (!is_piv[r]) act.push_back(r);
    }
    DenseBuf<Elt> Y(na, a);
    // (slot-24 G6) The rows of Mact . Winv are independent exact products, so
    // the product is formed in row slabs of kYSlab rows (each slab tiled over
    // the hook exactly as the whole product was): live are one slab of Mact
    // and Prod instead of two na x rho matrices (0.45 GB at the C1a cone's
    // q = 1 call, n_sym = 1000).  The same entries of Y.  Memory only.
    constexpr uint32_t kYSlab = 1024;
    for (uint32_t s0 = 0; s0 < na; s0 += kYSlab) {
        const uint32_t ns = std::min<uint32_t>(kYSlab, na - s0);
        DenseBuf<Elt> Mact(ns, rho), Prod(ns, rho);
        for (uint32_t i = 0; i < ns; ++i)
            for (uint32_t j = 0; j < rho; ++j) Mact.at(i, j) = M_at(act[s0 + i], P[j]);
        {
            // Row tiles of Prod are independent (exact per-entry products).
            const uint32_t tiles = std::max<uint32_t>(1, std::min<uint32_t>(16, ns / 64));
            if (tiles <= 1 || static_cast<size_t>(ns) * rho * rho < (size_t(1) << 22))
                kernels::gemm<Elt>(ctx, cview(Mact), cview(Winv), engine::view(Prod), GemmOp::Set);
            else
                hook(tiles, [&](uint32_t t) {
                    const uint32_t i0 = static_cast<uint32_t>((static_cast<uint64_t>(ns) * t) / tiles);
                    const uint32_t i1 = static_cast<uint32_t>((static_cast<uint64_t>(ns) * (t + 1)) / tiles);
                    if (i1 > i0) kernels::gemm<Elt>(ctx, cview(Mact).sub(i0, i1 - i0, 0, rho), cview(Winv), engine::view(Prod).sub(i0, i1 - i0, 0, rho), GemmOp::Set);
                });
        }
        for (uint32_t i = 0; i < ns; ++i) {
            for (uint32_t j = 0; j < rho; ++j) Y.at(s0 + i, prows[j]) = ctx.sub(Elt(0), Prod.at(i, j));
            Y.at(s0 + i, act[s0 + i]) = Elt(1);
        }
    }
    strace.mark("stage 2 Y assembly (Mact . Winv)", na, rho);
    Winv = DenseBuf<Elt>();   // dead: U0 and Y carry everything stage 2 still needs
    // U_bot = RREF(Y), the unique RREF basis of the left null space of M --
    // IN PLACE (Y is (a - rank) x a, the largest object of this stage; the
    // block driver would hold a second copy as its R).  The dense RREF's
    // trailing updates run over the hook installed above.
    const elim::RrefResult res2 = elim::rref<Elt>(ctx, engine::view(Y));
    ENGINE_CHECK_MSG(res2.rank == na, "tape::factor_basis: left null space lost rank");
    strace.mark("stage 2 in-place dense RREF of Y", na, a);
    // U_top = U_top0 - U_top0[:, pivcols(U_bot)] . U_bot  (zeros at U_bot's pivot columns)
    {
        DenseBuf<Elt> C(dim_h, na);
        for (uint32_t t = 0; t < dim_h; ++t)
            for (uint32_t j = 0; j < na; ++j) C.at(t, j) = U0.at(t, res2.pivots[j]);
        kernels::gemm<Elt>(ctx, cview(C), cview(Y).sub(0, na, 0, a), engine::view(U0), GemmOp::Sub);
    }
    strace.mark("stage 2 U_top reduction GEMM", dim_h, a);
    out.projector = std::move(U0);
    return out;
}

template<class Elt>
BlockMat<Elt> image_basis(Executor<Elt>& ex, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                          const std::vector<uint32_t>& row_dims, DenseView<const Elt> A, uint32_t chunk_,
                          TapeElimStats* stats) {
    uint32_t a = 0;
    for (uint32_t d : row_dims) a += d;
    const uint32_t k = A.cols();
    ENGINE_CHECK_MSG(k == 0 || A.rows() == a, "tape::image_basis: operand height != sum(row_dims)");
    const uint32_t chunk = std::max<uint32_t>(1, chunk_);
    blk::BlockRrefResult<Elt> res;
    if (k > 0 && a > 0) {
        BlockBuilder<Elt> b(pool, row_dims, chunk_dims(k, chunk), p);
        const BorrowHold<Elt> hold(pool, A);
        add_dense_chunks<Elt>(b, hold.id, A, 0, chunk);
        const BlockMat<Elt> aug = b.finalize();
        res = rref<Elt>(ex, aug, stats, /*rank_hint=*/std::min<uint32_t>(a, k));
    }
    DenseBuf<Elt> Im(a, res.rank, Layout::ColMajor);
    for (uint32_t t = 0; t < res.rank; ++t)
        kernels::copy<Elt>(A.sub(0, a, res.pivots[t], 1), engine::view(Im).sub(0, a, t, 1));
    return BlockMat<Elt>::from_flat(pool, std::move(Im), row_dims, p, /*detect=*/false);
}

template<class Elt>
DenseBuf<Elt> solve_multi_rhs(Executor<Elt>& ex, std::shared_ptr<BlockPool<Elt>> pool, Elt p,
                              const std::vector<uint32_t>& row_dims,
                              DenseView<const Elt> B, DenseView<const Elt> Y, uint32_t chunk_,
                              TapeElimStats* stats, bool certify) {
    uint32_t a = 0;
    for (uint32_t d : row_dims) a += d;
    const uint32_t nb = B.cols(), ny = Y.cols();
    ENGINE_CHECK_MSG((nb == 0 || B.rows() == a) && (ny == 0 || Y.rows() == a), "tape::solve_multi_rhs: operand height != sum(row_dims)");
    DenseBuf<Elt> result(ny, nb);
    if (ny == 0 || a == 0) return result;
    // ---- Unit-row shortcut -------------------------------------------------
    // When every column c of B has a UNIT ROW u_c (B[u_c, :] = e_c) -- which a
    // D5b kernel basis always has: the identity sits at the free variables --
    // the coordinates of any y in span(B) are forced: y = B x  =>  y[u_c] =
    // x[c].  So X = Y[u, :] is the unique solution whenever one exists, and
    // membership is decided by the EXACT residual Y - B X == 0, computed as
    // one tape GEMM (row-major, no relayout: (B X)^T = X^T B^T) and COMPAREd
    // against Y.  Same values and the same throw as the elimination route
    // (the RREF of [B | Y] is [I | X] when B has full column rank), at the
    // cost of one product instead of an a x (nb + ny) elimination
    // (EngineTapeEliminationValidation covers both routes entrywise).
    if (nb > 0) {
        const Clock::time_point ts0 = Clock::now();
        std::vector<uint32_t> nnz(a, 0), col_of(a, 0);
        std::vector<Elt> val(a, Elt(0));
        {
            const auto bs = B.strides();
            if (bs.cs == 1) {          // row-major B: walk rows
                for (uint32_t r = 0; r < a; ++r) {
                    const Elt* br = bs.base + static_cast<size_t>(r) * bs.rs;
                    for (uint32_t c = 0; c < nb; ++c) if (br[c] != 0) { ++nnz[r]; col_of[r] = c; val[r] = br[c]; }
                }
            } else {                   // column-major B: walk columns
                for (uint32_t c = 0; c < nb; ++c) {
                    const Elt* bc = bs.base + static_cast<size_t>(c) * bs.cs;
                    for (uint32_t r = 0; r < a; ++r) {
                        const Elt v = bc[static_cast<size_t>(r) * bs.rs];
                        if (v != 0) { ++nnz[r]; col_of[r] = c; val[r] = v; }
                    }
                }
            }
        }
        std::vector<uint32_t> unit_row(nb, a);
        for (uint32_t r = 0; r < a; ++r)
            if (nnz[r] == 1 && val[r] == Elt(1) && unit_row[col_of[r]] == a) unit_row[col_of[r]] = r;
        bool all = true;
        for (uint32_t c = 0; c < nb && all; ++c) all = (unit_row[c] < a);
        if (all) {
            {
                // X[t, c] = Y[u_c, t]: for each column t of Y (contiguous when
                // column-major) gather the unit rows into row t of X.
                const auto ys = Y.strides();
                for (uint32_t t = 0; t < ny; ++t) {
                    const Elt* yc = ys.base + static_cast<size_t>(t) * ys.cs;
                    Elt* xr = result.data() + static_cast<size_t>(t) * result.ld;
                    for (uint32_t c = 0; c < nb; ++c) xr[c] = yc[static_cast<size_t>(unit_row[c]) * ys.rs];
                }
            }
            const Clock::time_point ts1 = Clock::now();
            if (!certify) return result;
            const ModCtx<Elt> sctx(p);
            Recorder<Elt> rec(sctx);
            const BufId bB = rec.input(B);
            const BufId bY = rec.input(Y);
            const BufId bX = rec.input(engine::cview(result));
            const BufId bP = rec.temp(ny, a, Layout::RowMajor);
            // P = X^T . B^T  (ny x a), i.e. (B X)^T, then P == Y^T ?
            rec.gemm(rec.whole(bX), rec.whole(bB).transposed_view(), rec.whole(bP), GemmOp::Set);
            const uint32_t slot = rec.compare(rec.whole(bP), rec.whole(bY).transposed_view());
            Program<Elt> prog = rec.finish();
            compile(prog, elim_pass_options());   // (WP-6b) see rref()
            note_program_run(prog.stats.gpu_nodes);
            const RunResult rr = ex.run(static_cast<const Program<Elt>&>(prog), rec.bindings());
            if (stats) { ++stats->programs; stats->nodes += prog.live_nodes(); stats->tasks += rr.tasks; }
            if (std::getenv("ENGINE_TRACE_RSS") != nullptr)
                std::cout << "        [tape solve " << a << "x" << nb << " | " << ny << " unit rows] scan+gather " << std::fixed << std::setprecision(3)
                          << secs(ts0, ts1) << " s, certificate program " << secs(ts1, Clock::now()) << " s\n";
            ENGINE_CHECK(rr.slots[slot].written);
            if (!rr.slots[slot].equal) {
                // Name the first offending right-hand side, as the elimination route does.
                DenseBuf<Elt> P(ny, a);
                kernels::gemm<Elt>(sctx, engine::cview(result), B.transposed_view(), engine::view(P), GemmOp::Set);
                uint32_t bad = ny;
                for (uint32_t t = 0; t < ny && bad == ny; ++t)
                    for (uint32_t r = 0; r < a; ++r)
                        if (P.at(t, r) != Y.at(r, t)) { bad = t; break; }
                throw std::runtime_error("Vector is not in image of basis (right-hand side " + std::to_string(bad) + ").");
            }
            return result;
        }
    }
    const uint32_t chunk = std::max<uint32_t>(1, chunk_);
    std::vector<uint32_t> col_dims = chunk_dims(nb, chunk);
    const uint32_t nbc_b = static_cast<uint32_t>(col_dims.size());
    for (uint32_t d : chunk_dims(ny, chunk)) col_dims.push_back(d);
    BlockBuilder<Elt> b(pool, row_dims, col_dims, p);
    std::unique_ptr<BorrowHold<Elt>> holdB;
    if (nb > 0) { holdB = std::make_unique<BorrowHold<Elt>>(pool, B); add_dense_chunks<Elt>(b, holdB->id, B, 0, chunk); }
    const BorrowHold<Elt> holdY(pool, Y);
    add_dense_chunks<Elt>(b, holdY.id, Y, nbc_b, chunk);
    blk::BlockRrefResult<Elt> res;
    {
        const BlockMat<Elt> aug = b.finalize();
        res = rref<Elt>(ex, aug, stats, /*rank_hint=*/nb);  // rank <= nb when every RHS is in the span
    }
    for (uint32_t i = 0; i < res.rank; ++i) {
        const uint32_t c = res.pivots[i];
        if (c >= nb)
            throw std::runtime_error("Vector is not in image of basis (right-hand side " + std::to_string(c - nb) + ").");
        for (uint32_t t = 0; t < ny; ++t) result.at(t, c) = res.R.at(i, nb + t);
    }
    return result;
}

#define TAPE_ELIM_INSTANTIATE(Elt)                                                                        \
    template blk::BlockRrefResult<Elt> rref<Elt>(Executor<Elt>&, const BlockMat<Elt>&, TapeElimStats*, uint32_t); \
    template BlockMat<Elt> kernel_basis<Elt>(Executor<Elt>&, const BlockMat<Elt>&, Layout);               \
    template BlockMat<Elt> image_basis<Elt>(Executor<Elt>&, const BlockMat<Elt>&, Layout);                \
    template BlockMat<Elt> image_basis<Elt>(Executor<Elt>&, std::shared_ptr<BlockPool<Elt>>, Elt,          \
                                            const std::vector<uint32_t>&, DenseView<const Elt>, uint32_t, TapeElimStats*); \
    template blk::FactorBasisResult<Elt> factor_basis<Elt>(Executor<Elt>&, std::shared_ptr<BlockPool<Elt>>, Elt, \
                                                           const std::vector<uint32_t>&, DenseView<const Elt>, \
                                                           DenseView<const Elt>, uint32_t, TapeElimStats*, bool); \
    template DenseBuf<Elt> solve_multi_rhs<Elt>(Executor<Elt>&, std::shared_ptr<BlockPool<Elt>>, Elt,     \
                                                const std::vector<uint32_t>&, DenseView<const Elt>,       \
                                                DenseView<const Elt>, uint32_t, TapeElimStats*, bool);
TAPE_ELIM_INSTANTIATE(uint32_t)
TAPE_ELIM_INSTANTIATE(uint64_t)
#undef TAPE_ELIM_INSTANTIATE

} // namespace tape
} // namespace engine
