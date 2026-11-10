// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// The production involution iota^* of [C, sec:involution] -- phase E3,
// hecke-extensions slot 4 WP-3.  See HeckeInvolution.h for the sign, the
// independence discipline and what compute() discharges.

#include "HeckeInvolution.h"

#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/EnvLever.h"   // (S25 LEVERS-25) EnvArmRetiredLever
#include "../../ModArithmetic/ModCtx.h"
#include "../../Engine/EngineRuntime.h"   // run_concurrently (slot-10 WP-7)
#include "QuadPoint.h"
#include "HeckeSettings.h"
#include "../Util/Rss.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace hecke {

#if OC_ENABLE_LEGACY
#include "legacy/HeckeInvolutionLegacy.inc"   // (S24 CL-L2) the FlatMatrix pivot routes under ENGINE_BACKEND=legacy
#endif

namespace {

// (slot-23 P4I) ENGINE_TRACE_RSS=1: one line per step of compute() and per
// task of check_obl_iii / build_split, with the wall time since the previous
// mark and the process's VmRSS / VmHWM (MB) -- the shape of the cone's
// ConeTrace.  Measurement output only: gated on the environment, never on a
// result path, and it reads nothing the computation writes.  Task lines come
// from concurrent host tasks; each is written in one piece under a mutex.
bool iota_trace_on() {
    static const bool on = std::getenv("ENGINE_TRACE_RSS") != nullptr;
    return on;
}
void iota_trace_line(const std::string& what, double secs) {
    if (!iota_trace_on()) return;
    static std::mutex mu;
    std::ostringstream os;
    os << "      [iota] " << what << ": " << std::fixed << std::setprecision(3) << secs << " s, VmRSS="
       << std::setprecision(1) << oc_util::ProcStatusMiB("VmRSS:") << " MB VmHWM=" << oc_util::ProcStatusMiB("VmHWM:")
       << " MB\n";
    const std::lock_guard<std::mutex> lock(mu);
    std::cout << os.str() << std::flush;
}
struct IotaTrace {
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    void mark(const std::string& what) {
        const auto now = std::chrono::steady_clock::now();
        iota_trace_line(what, std::chrono::duration<double>(now - t).count());
        t = now;
    }
};

// (slot-23 P4I) The streamed iota route's switches (HeckeInvolution.h).  The
// environment default is read per thread at first use; compute() reads the
// calling thread's values once.
// (S25 LEVERS-25) the env arm is retired: in an OFF build HECKE_IOTA_STREAM=0 is REFUSED (EnvLever.h,
// EnvArmRetiredLever); the materialised route stays in every build (ScopedIotaStreaming(false), the
// comparison of HeckeIotaStreamValidation).  Policy{}: exactly "0" = off, anything else (or unset) = on,
// silently -- the old reading.
const engine::lever::EnvArmRetiredLever g_iota_stream_lever("HECKE_IOTA_STREAM", engine::lever::Policy{});
bool iota_env_stream() { return g_iota_stream_lever.env_enabled(); }
thread_local bool g_iota_streaming = iota_env_stream();
thread_local uint64_t g_iota_task_budget = 0;     // validation only
thread_local int g_iota_stream_mutation = 0;      // validation only

// Memory-bounded admission (P4I).  Every task gets its thread from ONE
// run_concurrently (so the thread context -- consumer pool, host-elimination
// hook, device policy -- is inherited exactly as before), but a task starts
// its work only when it is ADMITTED: first come, first served (each task
// thread draws the next ticket when it reaches the admission, so a task whose
// thread never gets there -- run_concurrently skips the task when a thread
// context installer throws -- strands no later one; audit F1), and only while
// the summed byte estimate of the admitted, still running tasks plus its own
// stays <= `budget` (a task is always admitted when nothing runs, so one
// larger than the budget runs alone).  The exception surfaced is
// run_concurrently's: the first failing task's in task order (all other tasks
// run, as before).  On a one-worker consumer pool run_concurrently is serial
// in task order and every task is admitted at once (nothing else runs).
// `mutate_skip` is the validation-only mutation 3: the last task with a
// non-trivial estimate (> 1 byte: a non-empty family) is not run.
struct Admission {
    uint32_t max_concurrent = 0;
    uint64_t peak_admitted = 0, max_task = 0;
    uint32_t skipped = 0;
};
// The default budget: twice the largest task's estimate, so the largest task
// runs beside smaller ones but never beside a second one of its size.
uint64_t auto_budget(const std::vector<uint64_t>& est) {
    uint64_t m = 1u;
    for (uint64_t e : est) m = std::max(m, e);
    return 2u * m;
}
Admission run_admitted(const std::vector<std::function<void()>>& tasks, const std::vector<uint64_t>& est,
                       uint64_t budget, bool mutate_skip) {
    ENGINE_CHECK_MSG(tasks.size() == est.size(), "HeckeInvolution: one byte estimate per task.");
    Admission res;
    for (uint64_t e : est) res.max_task = std::max(res.max_task, e);
    size_t skip = tasks.size();
    if (mutate_skip)
        for (size_t i = tasks.size(); i-- > 0;)
            if (est[i] > 1u) { skip = i; break; }
    std::mutex mu;
    std::condition_variable cv;
    size_t ticket = 0, serving = 0;   // drawn on arrival / admitted so far
    uint64_t used = 0;
    uint32_t running = 0;
    std::vector<std::function<void()>> gated;
    gated.reserve(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) {
        gated.emplace_back([&, i]() {
            {
                std::unique_lock<std::mutex> lk(mu);
                const size_t mine = ticket++;
                cv.wait(lk, [&] { return serving == mine && (running == 0u || used + est[i] <= budget); });
                ++serving;
                if (i != skip) {
                    used += est[i];
                    ++running;
                    res.max_concurrent = std::max(res.max_concurrent, running);
                    res.peak_admitted = std::max(res.peak_admitted, used);
                }
            }
            cv.notify_all();
            if (i == skip) return;
            struct Release {
                std::mutex& mu; std::condition_variable& cv; uint64_t& used; uint32_t& running; uint64_t e;
                ~Release() {
                    { std::lock_guard<std::mutex> g(mu); used -= e; --running; }
                    cv.notify_all();
                }
            } release{ mu, cv, used, running, est[i] };
            tasks[i]();
        });
    }
    engine::tape::run_concurrently(gated);
    ENGINE_CHECK_MSG(serving == tasks.size() && running == 0u && used == 0u,
                     "HeckeInvolution: the admission did not run every task.");
    res.skipped = (skip < tasks.size()) ? 1u : 0u;
    return res;
}


// ---------------------------------------------------------------------------
// Small local helpers.  Deliberately NOT the oracle's: this translation unit
// re-derives everything (HeckeInvolution.h, "independence from the oracle").
// ---------------------------------------------------------------------------

// Parity of the permutation sorting `img` back into increasing order.
int sort_parity_of(const std::vector<uint32_t>& img) {
    uint32_t inversions = 0;
    for (size_t a = 0; a + 1 < img.size(); ++a)
        for (size_t b = a + 1; b < img.size(); ++b)
            if (img[a] > img[b]) ++inversions;
    return (inversions % 2u == 0u) ? 1 : -1;
}

uint32_t locate_simplex(const std::vector<std::vector<std::vector<uint32_t>>>& simplices,
                        uint32_t q, std::vector<uint32_t> verts, const char* who) {
    std::sort(verts.begin(), verts.end());
    const auto& list = simplices.at(q);
    for (uint32_t k = 0; k < list.size(); ++k)
        if (list[k] == verts) return k;
    ENGINE_CHECK_MSG(false, (std::string(who) + ": the image of a stored simplex is not a stored simplex "
                             "-- hypothesis (I1) of [C, sec:involution] fails for this complex.").c_str());
    return 0u;
}

// The exact integer incidence [t : s] of [C, sec:conventions]'s stored
// orientation (increasing vertex index): (-1)^i when s = t minus its i-th
// vertex, 0 otherwise.  Used for the ell-independent form of iota_K d = d iota_K.
int incidence_sign(const std::vector<uint32_t>& t, const std::vector<uint32_t>& s) {
    if (t.size() != s.size() + 1u) return 0;
    size_t i = 0, j = 0;
    int dropped = -1;
    while (i < t.size()) {
        if (j < s.size() && t[i] == s[j]) { ++i; ++j; continue; }
        if (dropped >= 0) return 0;      // more than one mismatch
        dropped = static_cast<int>(i);
        ++i;
    }
    if (dropped < 0) return 0;
    return (dropped % 2 == 0) ? 1 : -1;
}

// A list of N_q x d_V cochains as one flat_dim x k multi-vector; column t is
// cochain t in the flat layout [C, eq:flatindex] s*d_V + j.
// The same entries as a raw row-major buffer with row stride `stride` >= k,
// written at column offset `col0` (slot-22 PERF1, P3: lets the iota checks
// build [X | iota_C X] in one buffer, without a FlatMatrix copy of X).
void multivector_into(const std::vector<FlatMatrix>& basis, uint32_t flat_dim, const char* who,
                      uint32_t* data, size_t stride, uint32_t col0) {
    const uint32_t k = static_cast<uint32_t>(basis.size());
    for (uint32_t t = 0; t < k; ++t) {
        const FlatMatrix& c = basis[t];
        const uint32_t dv = c.get_cols();
        ENGINE_CHECK_MSG(c.get_rows() * dv == flat_dim,
                         (std::string(who) + ": a stored basis cochain has the wrong shape.").c_str());
        for (uint32_t s = 0; s < c.get_rows(); ++s)
            for (uint32_t j = 0; j < dv; ++j)
                data[static_cast<size_t>(s * dv + j) * stride + col0 + t] = c.get_entry(s, j);
    }
}

FlatMatrix multivector_of(const std::vector<FlatMatrix>& basis, uint32_t flat_dim, uint32_t ell,
                          const char* who) {
    const uint32_t k = static_cast<uint32_t>(basis.size());
    std::vector<uint32_t> data(static_cast<size_t>(flat_dim) * k, 0u);
    multivector_into(basis, flat_dim, who, data.data(), k, 0u);
    return FlatMatrix(flat_dim, k, ell, data);
}

FlatMatrix hcat_of(const FlatMatrix& A, const FlatMatrix& B) {
    ENGINE_CHECK(A.get_rows() == B.get_rows());
    const uint32_t R = A.get_rows(), CA = A.get_cols(), CB = B.get_cols();
    std::vector<uint32_t> data(static_cast<size_t>(R) * (CA + CB), 0u);
    for (uint32_t i = 0; i < R; ++i) {
        for (uint32_t j = 0; j < CA; ++j) data[static_cast<size_t>(i) * (CA + CB) + j] = A.get_entry(i, j);
        for (uint32_t j = 0; j < CB; ++j) data[static_cast<size_t>(i) * (CA + CB) + CA + j] = B.get_entry(i, j);
    }
    return FlatMatrix(R, CA + CB, A.get_p(), data);
}

// Columns of a vector<FlatMatrix> (each rows x 1) as one matrix.
FlatMatrix columns_of(const std::vector<FlatMatrix>& cols, uint32_t rows, uint32_t ell) {
    const uint32_t k = static_cast<uint32_t>(cols.size());
    std::vector<uint32_t> data(static_cast<size_t>(rows) * k, 0u);
    for (uint32_t t = 0; t < k; ++t) {
        ENGINE_CHECK(cols[t].get_rows() == rows && cols[t].get_cols() == 1u);
        for (uint32_t i = 0; i < rows; ++i) data[static_cast<size_t>(i) * k + t] = cols[t].get_entry(i, 0);
    }
    return FlatMatrix(rows, k, ell, data);
}

// rho_n(iota) * M in O(d_V^2): rho is diag((-1)^{n-j}), so this negates the
// rows whose index has (n - a) odd.  Used to keep the relation check of
// [C, obl:iota](ii) at one d_V^3 product per CANDIDATE row rather than three.
FlatMatrix rho_times(const FlatMatrix& rho, const FlatMatrix& M, uint32_t ell) {
    const uint32_t R = M.get_rows(), C = M.get_cols();
    ENGINE_CHECK(rho.get_rows() == R && rho.get_cols() == R);
    // rho_n(iota) = diag((-1)^{n-j}) has entries 1 and ell-1 ONLY
    // ([C, eq:rhoniota], re-asserted in build_rho), so this is a conditional
    // negation, not a modular multiply -- exact, and no raw mod_* call in a
    // pipeline TU (docs/matrix-engine/tools/raw_mod_calls.allow).
    std::vector<uint32_t> data(static_cast<size_t>(R) * C, 0u);
    for (uint32_t a = 0; a < R; ++a) {
        const uint32_t d = rho.get_entry(a, a);
        ENGINE_CHECK_MSG(d == 1u % ell || d == ell - 1u,
                         "rho_times: rho_n(iota) is not diagonal with entries +-1.");
        const bool negate = (d != 1u % ell);
        for (uint32_t b = 0; b < C; ++b) {
            const uint32_t v = M.get_entry(a, b);
            data[static_cast<size_t>(a) * C + b] = (v == 0u) ? 0u : (negate ? ell - v : v);
        }
    }
    return FlatMatrix(R, C, ell, data);
}

// span(A) contains span(B)?  Both have the same number of rows.
//
// (slot-22 PERF1, P3) ONE elimination instead of two.  This used to be
// hcat(A, B).rank() == A.rank(), which eliminated A twice (once inside
// [A | B], once alone).  The pivot columns of RREF[A | B] are its leftmost
// independent columns -- a property of the matrix, not of the elimination --
// so those below cols(A) are exactly the pivots of A: rank A is their number,
// and rank [A | B] == rank A iff no pivot lies at or beyond cols(A).  Same
// predicate, same verdict.
bool span_contains(const FlatMatrix& A, const FlatMatrix& B) {
    if (B.get_cols() == 0u) return true;
    if (A.get_cols() == 0u) return B.is_zero();
    const FlatMatrix aug = hcat_of(A, B);
    const std::vector<uint32_t>& piv = aug.get_pivot_columns();
    const uint32_t cA = A.get_cols();
    const int64_t rA = std::count_if(piv.begin(), piv.end(), [cA](uint32_t c) { return c < cA; });
    return static_cast<int64_t>(piv.size()) == rA;
}

} // namespace

bool HeckeInvolution::iota_streaming() { return g_iota_streaming; }
HeckeInvolution::ScopedIotaStreaming::ScopedIotaStreaming(bool on) : saved_(g_iota_streaming) { g_iota_streaming = on; }
HeckeInvolution::ScopedIotaStreaming::~ScopedIotaStreaming() { g_iota_streaming = saved_; }
HeckeInvolution::ScopedIotaTaskBudget::ScopedIotaTaskBudget(uint64_t bytes) : saved_(g_iota_task_budget) {
    g_iota_task_budget = bytes;
}
HeckeInvolution::ScopedIotaTaskBudget::~ScopedIotaTaskBudget() { g_iota_task_budget = saved_; }
HeckeInvolution::ScopedIotaStreamMutation::ScopedIotaStreamMutation(int kind) : saved_(g_iota_stream_mutation) {
    g_iota_stream_mutation = kind;
}
HeckeInvolution::ScopedIotaStreamMutation::~ScopedIotaStreamMutation() { g_iota_stream_mutation = saved_; }

// ---------------------------------------------------------------------------
// IotaSimplicialMap
// ---------------------------------------------------------------------------

uint32_t IotaSimplicialMap::num_minus_signs() const {
    uint32_t c = 0;
    for (const auto& per_q : image_sign)
        for (int s : per_q) if (s < 0) ++c;
    return c;
}

const std::vector<std::vector<uint32_t>>& IotaModelMinusSignSimplices() {
    // [C, lem:iotamodel]: sign -1 on exactly six of the 23 simplices.
    static const std::vector<std::vector<uint32_t>> rows = {
        { 0, 1 }, { 2, 3 }, { 1, 2 }, { 0, 1, 4 }, { 2, 3, 5 }, { 1, 2, 6 },
    };
    return rows;
}

std::vector<uint32_t> IotaBoundaryPathVertexPermutation() {
    // The horocycle path v0-v1-v2-v3 is reversed by z -> -conj(z)
    // (Re v_0 = -1/2, Re v_1 = -1/6, Re v_2 = +1/6, Re v_3 = +1/2).
    return std::vector<uint32_t>{ 3, 2, 1, 0 };
}

IotaSimplicialMap BuildIotaSimplicialMapFromVertexPermutation(const SimplicialComplex& K,
                                                              const std::vector<uint32_t>& vertex_perm) {
    IotaSimplicialMap io;
    io.vertex_perm = vertex_perm;
    const auto& simplices = K.get_simplices();
    io.image_index.assign(simplices.size(), {});
    io.image_sign.assign(simplices.size(), {});
    for (uint32_t q = 0; q < simplices.size(); ++q) {
        const auto& list = simplices[q];
        io.image_index[q].assign(list.size(), 0u);
        io.image_sign[q].assign(list.size(), 1);
        for (uint32_t s = 0; s < list.size(); ++s) {
            std::vector<uint32_t> img;
            img.reserve(list[s].size());
            for (uint32_t v : list[s]) {
                ENGINE_CHECK_MSG(v < vertex_perm.size(),
                                 "BuildIotaSimplicialMapFromVertexPermutation: vertex out of range.");
                img.push_back(vertex_perm[v]);
            }
            io.image_sign[q][s] = sort_parity_of(img);
            io.image_index[q][s] = locate_simplex(simplices, q, img,
                                                  "BuildIotaSimplicialMapFromVertexPermutation");
        }
    }
    return io;
}

IotaSimplicialMap BuildIotaSimplicialMapOnModel(const SimplicialComplex& K,
                                                const SL2ZTessellationTables& model) {
    // [C, def:iota](iii) on the shipped model.  z = (u + v sqrt(D) i)/w, so
    // -conj(z) = (-u + v sqrt(D) i)/w: an EXACT QuadPoint, compared exactly.
    // Geometry only PROPOSES the permutation -- check_obl_i() certifies it
    // against [C, lem:iotamodel] (invariant I8).
    const uint32_t nv = K.get_num_q_simplices(0);
    std::vector<uint32_t> vperm(nv, 0u);
    for (uint32_t v = 0; v < nv; ++v) {
        const QuadPoint z = model.vertex_position(v);
        const QuadPoint image(checked_neg<int64_t>(z.u), z.v, z.w, z.D);
        int hit = -1;
        for (uint32_t w = 0; w < nv; ++w)
            if (model.vertex_position(w) == image) { hit = static_cast<int>(w); break; }
        ENGINE_CHECK_MSG(hit >= 0,
                         "BuildIotaSimplicialMapOnModel: z -> -conj(z) sends a vertex off the vertex set "
                         "-- hypothesis (I1) of [C, sec:involution] fails.");
        vperm[v] = static_cast<uint32_t>(hit);
    }
    return BuildIotaSimplicialMapFromVertexPermutation(K, vperm);
}

const char* to_string(SignSplitStatus s) {
    switch (s) {
        case SignSplitStatus::Split: return "split";
        case SignSplitStatus::NotApplicableCharTwo: return "not_applicable";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// HeckeInvolution
// ---------------------------------------------------------------------------

HeckeInvolution::HeckeInvolution(const EquivariantCohomologyOnK* coh,
                                 IotaSimplicialMap iota_K,
                                 std::string label)
    : coh_(coh), iota_K_(std::move(iota_K)), label_(std::move(label)) {
    // (slot-8 WP-3) [E] sec:ring (b): the +- split is stored on the F_ell bases.
    RequireFieldCoefficientRing("HeckeInvolution", coh ? coh->coefficient_ring_k() : 1u);
    ENGINE_CHECK_MSG(coh_ != nullptr, "HeckeInvolution: null cohomology object.");
    ENGINE_CHECK_MSG(coh_->is_cohomology_computed(),
                     "HeckeInvolution: the cohomology object must be computed first.");
    ENGINE_CHECK_MSG(coh_->get_lssc() != nullptr, "HeckeInvolution: the cohomology has no local system.");
    ell_ = coh_->get_prime();
    d_V_ = coh_->get_representation_dim();
    ENGINE_CHECK_MSG(d_V_ >= 1u, "HeckeInvolution: representation dimension must be positive.");
    n_sym_ = d_V_ - 1u;
    const SimplicialComplex& K = *coh_->get_lssc()->get_base_K_complex();
    num_degrees_ = static_cast<uint32_t>(K.get_simplices().size());
    ENGINE_CHECK_MSG(iota_K_.num_degrees() == num_degrees_,
                     "HeckeInvolution: the iota_K table does not match the complex's degree count.");
}

HeckeInvolution::~HeckeInvolution() = default;

void HeckeInvolution::set_exact_relation_data(std::vector<SL2ZRelationDatum> data) {
    ENGINE_CHECK_MSG(!computed_, "HeckeInvolution: set_exact_relation_data after compute().");
    exact_data_ = std::move(data);
    has_exact_data_ = true;
}

void HeckeInvolution::require_computed(const char* who) const {
    ENGINE_CHECK_MSG(computed_, (std::string("HeckeInvolution::") + who + ": compute() has not run.").c_str());
}

void HeckeInvolution::build_rho() {
    // [C, def:iota](i)-(ii): iota = diag(-1, 1), rho_n(iota) through the LIVE
    // ComputeSymmetricPowerMatrixFM -- the same call LocalSystemSL2Z uses for
    // every relation matrix.  Checked against the closed form eq:rhoniota
    // diag((-1)^{n-j}).
    const Mat22 iota{ -1, 0, 0, 1 };
    const std::array<uint32_t, 4> r = iota.reduced_mod(ell_);
    const FlatMatrix iota_mod(2, 2, ell_, std::vector<uint32_t>{ r[0], r[1], r[2], r[3] });
    rho_iota_ = ComputeSymmetricPowerMatrixFM(iota_mod, n_sym_);
    ENGINE_CHECK_MSG(rho_iota_.get_rows() == d_V_ && rho_iota_.get_cols() == d_V_,
                     "HeckeInvolution: rho_n(iota) has the wrong shape.");
    {
        std::vector<uint32_t> closed(static_cast<size_t>(d_V_) * d_V_, 0u);
        for (uint32_t j = 0; j < d_V_; ++j) {
            const bool minus = ((n_sym_ - j) % 2u) == 1u;
            closed[static_cast<size_t>(j) * d_V_ + j] = minus ? (ell_ - 1u) : 1u;
        }
        ENGINE_CHECK_MSG(rho_iota_ == FlatMatrix(d_V_, d_V_, ell_, closed),
                         "HeckeInvolution: the live rho_n(iota) is not diag((-1)^{n-j}) ([C, eq:rhoniota]).");
    }
    ENGINE_CHECK_MSG(rho_iota_ * rho_iota_ == FlatMatrix(d_V_, ell_),
                     "HeckeInvolution: rho_n(iota)^2 != I ([C, eq:rhoniota]).");
}

void HeckeInvolution::build_morphism() {
    // [C, sec:codemap]'s production form: ONE PerSimplexBlock per simplex,
    // source simplex |iota_K sigma|, block  -eps_sigma rho_n(iota)  -- the sign
    // of lem:iotamodel NEGATED per def:iota(iv) and folded into the matrix.
    // THE global sign of the whole object is the `-` on the next line and
    // nowhere else ([C, rmk:iotasign]: never flipped at a call site).
    const FlatMatrix rho_plus = rho_iota_;
    const FlatMatrix rho_minus = rho_iota_ * (ell_ - 1u);

    FiberBlockTable table(num_degrees_);
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = static_cast<uint32_t>(iota_K_.image_index[q].size());
        table[q].assign(N_q, {});
        for (uint32_t s = 0; s < N_q; ++s) {
            const uint32_t t = iota_K_.image_index[q][s];
            const int eps = iota_K_.image_sign[q][s];
            // block = -eps * rho_n(iota):  eps = +1 -> -rho, eps = -1 -> +rho.
            table[q][s].push_back(PerSimplexBlock{ t, (eps > 0) ? rho_minus : rho_plus });
        }
    }
    morphism_ = std::make_unique<EquivariantCochainComplexMorphismOnK>(
        coh_, coh_, std::move(table), label_.empty() ? std::string("iota^*") : label_);
    morphism_->compute_induced_maps_on_cohomology();
    iota_H_ = morphism_->get_induced_H_eq_maps();
    iota_H_.resize(num_degrees_, FlatMatrix::CreateZero(0, 0, ell_));
}

// (slot-22 PERF1, P3) iota_C as a signed coordinate permutation.
//
// The morphism's pull-back is h1(s) = sum_b b.F h2(b.s2) over the fibre-block
// row of s.  build_morphism put exactly ONE block in every row, at source
// simplex t = iota_K(s), with F = -eps_s rho_n(iota), and rho_n(iota) is
// diagonal with entries +-1 ([C, eq:rhoniota], asserted by build_rho).  So
// (iota_C x)[s d_V + a] = F[a][a] x[t d_V + a] with F[a][a] in {1, ell - 1}:
// a copy or a negation, never a modular product.  This is not assumed -- every
// row of the morphism's STORED table is checked here, block by block and entry
// by entry (one block, source t, F diagonal with +-1 on the diagonal), so the
// map applied below is the morphism's map, and the value of iota_C x is the one
// apply_to_flat_cochain's block GEMV gives (an exact sum whose only nonzero
// term is F[a][a] x[t d_V + a]).
void HeckeInvolution::build_signed_permutation() {
    const SimplicialComplex& K = *coh_->get_lssc()->get_base_K_complex();
    const FiberBlockTable& table = morphism_->get_fiber_blocks();
    ENGINE_CHECK_MSG(table.size() >= num_degrees_, "HeckeInvolution: the morphism's block table misses a degree.");
    const uint32_t minus_one = ell_ - 1u, one = 1u % ell_;
    iota_src_.assign(num_degrees_, {});
    iota_neg_.assign(num_degrees_, {});
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = K.get_num_q_simplices(q);
        ENGINE_CHECK_MSG(table[q].size() == N_q, "HeckeInvolution: the morphism's block table has the wrong row count.");
        std::vector<uint32_t>& src = iota_src_[q];
        std::vector<uint8_t>& neg = iota_neg_[q];
        src.assign(static_cast<size_t>(N_q) * d_V_, 0u);
        neg.assign(static_cast<size_t>(N_q) * d_V_, 0u);
        for (uint32_t s = 0; s < N_q; ++s) {
            ENGINE_CHECK_MSG(table[q][s].size() == 1u,
                             "HeckeInvolution: a block row of iota_C does not hold exactly one block.");
            const PerSimplexBlock& b = table[q][s][0];
            const uint32_t t = b.s2_idx_in_K2;
            ENGINE_CHECK_MSG(t == iota_K_.image_index[q][s] && t < N_q,
                             "HeckeInvolution: a block of iota_C does not sit at iota_K(sigma).");
            ENGINE_CHECK_MSG(b.F.get_rows() == d_V_ && b.F.get_cols() == d_V_ && b.F.get_p() == ell_,
                             "HeckeInvolution: a block of iota_C has the wrong shape.");
            for (uint32_t a = 0; a < d_V_; ++a) {
                for (uint32_t c = 0; c < d_V_; ++c) {
                    const uint32_t v = b.F.get_entry(a, c);
                    if (a != c) {
                        ENGINE_CHECK_MSG(v == 0u, "HeckeInvolution: a block of iota_C is not diagonal.");
                        continue;
                    }
                    ENGINE_CHECK_MSG(v == one || v == minus_one,
                                     "HeckeInvolution: a diagonal entry of an iota_C block is not +-1.");
                    const size_t r = static_cast<size_t>(s) * d_V_ + a;
                    src[r] = t * d_V_ + a;
                    neg[r] = (v != one) ? 1u : 0u;
                }
            }
        }
    }
}

// iota_C on a raw row-major multi-vector: out row r (k entries, stride
// out_stride) = +-(in row iota_src_[q][r]) (stride in_stride).  `in` and `out`
// may share a buffer when their column ranges are disjoint.
void HeckeInvolution::iota_apply_rows(uint32_t q, const uint32_t* in, size_t in_stride, uint32_t* out,
                                      size_t out_stride, uint32_t k) const {
    ENGINE_CHECK_MSG(q < iota_src_.size(), "HeckeInvolution::iota_apply: degree out of range.");
    const std::vector<uint32_t>& src = iota_src_[q];
    const std::vector<uint8_t>& neg = iota_neg_[q];
    const size_t flat = src.size();
    for (size_t r = 0; r < flat; ++r) {
        const uint32_t* x = in + static_cast<size_t>(src[r]) * in_stride;
        uint32_t* y = out + r * out_stride;
        if (neg[r]) {
            for (uint32_t t = 0; t < k; ++t) y[t] = (x[t] == 0u) ? 0u : ell_ - x[t];
        } else {
            for (uint32_t t = 0; t < k; ++t) y[t] = x[t];
        }
    }
}

// iota_C X for a flat x k multi-vector X, as one FlatMatrix: the
// column-by-column result of apply_to_flat_cochain, entry for entry.
FlatMatrix HeckeInvolution::iota_apply(uint32_t q, const FlatMatrix& X) const {
    ENGINE_CHECK_MSG(q < iota_src_.size(), "HeckeInvolution::iota_apply: degree out of range.");
    const uint32_t flat = X.get_rows(), k = X.get_cols();
    ENGINE_CHECK_MSG(flat == iota_src_[q].size(), "HeckeInvolution::iota_apply: the operand height is not the flat dimension.");
    if (flat == 0u || k == 0u) return FlatMatrix::CreateZero(flat, k, ell_);
    X.populate_flat_data();   // row-major, entry (i, t) at i*k + t, also for a scalar-flagged X
    const std::vector<uint32_t>& in = X.get_flat_data_unsafe();
    ENGINE_CHECK_MSG(in.size() == static_cast<size_t>(flat) * k, "HeckeInvolution::iota_apply: operand data size.");
    std::vector<uint32_t> out(static_cast<size_t>(flat) * k);
    iota_apply_rows(q, in.data(), k, out.data(), k, k);
    return FlatMatrix(flat, k, ell_, out);
}

void HeckeInvolution::check_obl_i() {
    // [C, obl:iota](i).  Aborting.
    const SimplicialComplex& K = *coh_->get_lssc()->get_base_K_complex();
    const auto& simplices = K.get_simplices();

    // The vertex permutation is an involution of the vertex set.
    for (uint32_t v = 0; v < iota_K_.vertex_perm.size(); ++v) {
        ENGINE_CHECK_MSG(iota_K_.vertex_perm[v] < iota_K_.vertex_perm.size(),
                         "[C, obl:iota](i): the vertex permutation leaves the vertex set.");
        ENGINE_CHECK_MSG(iota_K_.vertex_perm[iota_K_.vertex_perm[v]] == v,
                         "[C, obl:iota](i): the vertex permutation is not an involution.");
    }
    // It is simplicial and involutive on simplices, with eps_{iota sigma} = eps_sigma.
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t N_q = static_cast<uint32_t>(simplices[q].size());
        ENGINE_CHECK_MSG(iota_K_.image_index[q].size() == N_q && iota_K_.image_sign[q].size() == N_q,
                         "[C, obl:iota](i): the iota_K table has the wrong degree shape.");
        for (uint32_t s = 0; s < N_q; ++s) {
            const uint32_t t = iota_K_.image_index[q][s];
            ENGINE_CHECK_MSG(t < N_q, "[C, obl:iota](i): iota_K leaves the stored simplex list.");
            ENGINE_CHECK_MSG(iota_K_.image_index[q][t] == s,
                             "[C, obl:iota](i): iota_K is not an involution on simplices.");
            ENGINE_CHECK_MSG(iota_K_.image_sign[q][t] == iota_K_.image_sign[q][s],
                             "[C, obl:iota](i): eps_{iota sigma} != eps_sigma ([C, def:iota](iv)).");
            // The sign is RECOMPUTED here, never read from the table.
            std::vector<uint32_t> img;
            img.reserve(simplices[q][s].size());
            for (uint32_t v : simplices[q][s]) img.push_back(iota_K_.vertex_perm[v]);
            ENGINE_CHECK_MSG(sort_parity_of(img) == iota_K_.image_sign[q][s],
                             "[C, obl:iota](i): the recomputed orientation sign disagrees with the table.");
            ENGINE_CHECK_MSG(locate_simplex(simplices, q, img, "[C, obl:iota](i)") == t,
                             "[C, obl:iota](i): the recomputed image index disagrees with the table.");
        }
    }
    // iota_K commutes with the boundary, over Z: D[iota t][iota s] = eps_t eps_s D[t][s].
    // This is exactly what makes iota_C d = d iota_C hold, and it is
    // ell-independent and exact.
    for (uint32_t q = 0; q + 1 < num_degrees_; ++q) {
        const auto& lo = simplices[q];
        const auto& hi = simplices[q + 1];
        for (uint32_t t = 0; t < hi.size(); ++t)
            for (uint32_t s = 0; s < lo.size(); ++s) {
                const int lhs = incidence_sign(hi[iota_K_.image_index[q + 1][t]],
                                               lo[iota_K_.image_index[q][s]]);
                const int rhs = iota_K_.image_sign[q + 1][t] * iota_K_.image_sign[q][s]
                              * incidence_sign(hi[t], lo[s]);
                ENGINE_CHECK_MSG(lhs == rhs,
                                 "[C, obl:iota](i): iota_K does not commute with the boundary over Z.");
            }
    }

    // The seven-vertex model's printed table ([C, lem:iotamodel]).
    cert_.model_is_seven_vertex = (num_degrees_ == 3u && simplices[0].size() == 7u
                                   && simplices[1].size() == 11u && simplices[2].size() == 5u);
    if (cert_.model_is_seven_vertex) {
        const std::vector<uint32_t> expected{ 3, 2, 1, 0, 5, 4, 6 };
        ENGINE_CHECK_MSG(iota_K_.vertex_perm == expected,
                         "[C, obl:iota](i): the vertex permutation is not lem:iotamodel's "
                         "{0<->3, 1<->2, 4<->5, 6}.");
        const auto& tbl = IotaModelMinusSignSimplices();
        for (uint32_t q = 0; q < num_degrees_; ++q)
            for (uint32_t s = 0; s < simplices[q].size(); ++s) {
                const bool expect_minus = std::find(tbl.begin(), tbl.end(), simplices[q][s]) != tbl.end();
                ENGINE_CHECK_MSG((iota_K_.image_sign[q][s] < 0) == expect_minus,
                                 "[C, obl:iota](i): the orientation sign disagrees with lem:iotamodel's table.");
            }
        ENGINE_CHECK_MSG(iota_K_.num_minus_signs() == 6u,
                         "[C, obl:iota](i): lem:iotamodel's table has six minus signs.");
    }
    cert_.minus_sign_count = iota_K_.num_minus_signs();
    cert_.iota_K = iota_K_;
}

void HeckeInvolution::check_obl_ii() {
    // [C, obl:iota](ii).  Aborting.
    const auto& rels = coh_->get_lssc()->get_user_defined_simplex_relations();
    cert_.relations.clear();
    cert_.relations.reserve(rels.size());
    cert_.exact_relation_data = has_exact_data_;
    if (has_exact_data_)
        ENGINE_CHECK_MSG(exact_data_.size() == rels.size(),
                         "[C, obl:iota](ii): the exact relation data does not match the local system's list.");

    for (size_t r = 0; r < rels.size(); ++r) {
        const auto& rel = rels[r];
        InvolutionCertificate::RelationRow row;
        row.q_dim = rel.q_dim;
        row.s_lhs = rel.s_lhs_idx_in_Kq;
        row.s_rhs = rel.s_rhs_idx_in_Kq;
        row.M = rel.M;
        if (has_exact_data_) {
            ENGINE_CHECK_MSG(exact_data_[r].rel.q_dim == rel.q_dim
                                 && exact_data_[r].rel.s_lhs_idx_in_Kq == rel.s_lhs_idx_in_Kq
                                 && exact_data_[r].rel.s_rhs_idx_in_Kq == rel.s_rhs_idx_in_Kq,
                             "[C, obl:iota](ii): the exact relation data is not index-aligned.");
            row.has_exact = true;
            row.g_r = exact_data_[r].g_r;
        }

        const uint32_t q = rel.q_dim;
        const uint32_t il = iota_K_.image_index[q][rel.s_lhs_idx_in_Kq];
        const uint32_t ir = iota_K_.image_index[q][rel.s_rhs_idx_in_Kq];
        const int el = iota_K_.image_sign[q][rel.s_lhs_idx_in_Kq];
        const int er = iota_K_.image_sign[q][rel.s_rhs_idx_in_Kq];
        // [C, prop:iotacommutes](1): the two sides of a row carry the SAME
        // orientation sign, so the common sign cancels and the image triple is
        // a triple of STORED simplices.
        ENGINE_CHECK_MSG(el == er,
                         "[C, obl:iota](ii): a relation row's two sides carry different orientation signs "
                         "under iota_K ([C, prop:iotacommutes](1)).");

        // The image is the INVERSE of a listed row: find (lhs', rhs') = (iota rhs, iota lhs)
        // whose fibre matrix M' satisfies  rho_n(iota) = M rho_n(iota) M'
        // (equivalently rho_n(iota) M'^{-1} = M rho_n(iota), i.e. the
        // intertwining rho(iota) rho(c_iota(gamma)) = rho(gamma) rho(iota) with
        // rho(c_iota(gamma)) = M'^{-1}).  This is the fibre half of clause (ii)
        // and it is what makes iota_C carry C^q_eq into itself.
        // Candidates are filtered by simplex indices FIRST (there are at most a
        // couple), so the d_V^3 fibre product below runs once per candidate and
        // not once per (row, row) pair.
        bool matched = false;
        for (size_t s = 0; s < rels.size() && !matched; ++s) {
            const auto& other = rels[s];
            if (other.q_dim != q) continue;
            if (other.s_lhs_idx_in_Kq != ir || other.s_rhs_idx_in_Kq != il) continue;
            if (iota_K_.image_sign[q][other.s_lhs_idx_in_Kq] != iota_K_.image_sign[q][other.s_rhs_idx_in_Kq])
                continue;
            if (row.has_exact && has_exact_data_) {
                // The exact Mat22 half: gamma' must be c_iota(gamma)^{-1}.
                const Mat22 iv{ -1, 0, 0, 1 };
                const Mat22 want = (iv * row.g_r * iv).inv_unimodular();
                if (!(exact_data_[s].g_r == want)) continue;
            }
            // rho_n(iota) = M rho_n(iota) M'  <=>  rho_n(iota) M'^{-1} = M rho_n(iota),
            // i.e. the intertwining rho(iota) rho(c_iota(gamma)) = rho(gamma) rho(iota).
            if (!(rel.M * rho_times(rho_iota_, other.M, ell_) == rho_iota_)) continue;
            matched = true;
            row.matched = true;
            row.match_index = static_cast<uint32_t>(s);
        }
        ENGINE_CHECK_MSG(matched,
                         "[C, obl:iota](ii): the image of a relation row under (iota_K, c_iota) is not the "
                         "inverse of a listed row with the intertwining fibre identity.");
        cert_.relations.push_back(std::move(row));
    }
}

void HeckeInvolution::check_obl_iii() {
    // [C, obl:iota](iii).  Aborting.
    const SimplicialComplex& K = *coh_->get_lssc()->get_base_K_complex();
    const auto& dims = coh_->get_H_eq_dimensions();
    const auto& C = coh_->get_C_eq_bases_fm();
    const auto& Z = coh_->get_Z_eq_bases_fm();
    const auto& B = coh_->get_B_eq_bases_fm();
    const auto& H = coh_->get_H_eq_bases_fm();
    const auto& P = coh_->get_H_eq_projectors_from_C_eq_flat_fm();

    cert_.dim_H.assign(num_degrees_, 0u);
    cert_.flat_dim.assign(num_degrees_, 0u);
    cert_.iota_H.assign(num_degrees_, FlatMatrix::CreateZero(0, 0, ell_));

    // (slot-10 WP-7) The checks below are independent exact decisions on
    // read-only inputs, so they run as host tasks (run_concurrently), in the
    // serial order: per degree q, [(iii-a/b) + C^q_eq], [Z^q_eq], [B^q_eq],
    // [the H-level check].  Each task builds and eliminates only its own
    // matrices; nothing it computes is stored except cert_.iota_H, which is
    // copied after the fan-out, and its own pre-sized family_rank_ slot.  run_concurrently rethrows the first failing
    // task's exception IN TASK ORDER (and runs the tasks serially, in order,
    // on a one-worker pool), so the check that aborts is the one the serial
    // loop aborted on.  What the tasks share is made read-only first:
    //  * the shapes are filled in here;
    //  * rho_n(iota)^2 = I is q-independent and decided once (build_rho has
    //    already asserted it; the per-degree check keeps its message);
    //  * iota_C is the signed permutation built (and checked against the
    //    morphism's table) by build_signed_permutation() before this runs;
    //    the tasks only read it;
    //  * the morphism's cached pull-back operator of every degree is still
    //    built here, although these checks no longer use it, so that later
    //    (possibly concurrent) callers of apply_to_flat_cochain find it built,
    //    as before.
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        cert_.flat_dim[q] = K.get_num_q_simplices(q) * d_V_;
        cert_.dim_H[q] = (q < dims.size()) ? dims[q] : 0u;
    }
    const bool rho_squared_is_identity = (rho_iota_ * rho_iota_ == FlatMatrix(d_V_, ell_));
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t flat = cert_.flat_dim[q];
        (void)morphism_->apply_to_flat_cochain(q, FlatMatrix(flat, 1, ell_));
    }
    // (slot-22 PERF1, P3) iota_C is applied as the signed permutation that
    // build_signed_permutation() checked against the morphism's table, to the
    // whole multi-vector at once.
    family_rank_.assign(num_degrees_, std::array<int64_t, 3>{ { -1, -1, -1 } });

    std::vector<std::function<void()>> tasks;
    std::vector<uint64_t> est;   // (P4I) each task's dominant buffer, bytes (admission only)
    auto fam_k = [](const std::vector<std::vector<FlatMatrix>>& fam, uint32_t q) -> uint64_t {
        return (q < fam.size()) ? static_cast<uint64_t>(fam[q].size()) : 0u;
    };
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        const uint32_t flat = cert_.flat_dim[q];

        // (iii-c) iota_C carries the stored C^q_eq, Z^q_eq, B^q_eq into their
        // own spans.  The one elimination of [X | iota_C X] also yields rank X,
        // stored for build_split (family index 0 / 1 / 2 = C / Z / B).
        //
        // (slot-22 PERF1, P3) [X | iota_C X] is assembled ONCE, straight into
        // one row-major buffer (X from the stored cochains, iota_C X by the
        // checked signed permutation), and its column rank profile is taken IN
        // PLACE (engine::elim::column_rank_profile: the pivot columns of the
        // RREF, i.e. exactly rref()'s, from the packed PLU without the
        // back-substitution) -- without the FlatMatrix copies of X, of iota_C X,
        // of the concatenation and of its RREF cache, and without the second
        // elimination of X alone.  The predicate is span_contains's: no pivot
        // at or beyond cols(X); rank X is the number of pivots below cols(X).
        auto carry_check = [this, q, flat](const std::vector<std::vector<FlatMatrix>>& fam, const char* name,
                                           uint32_t fam_idx) {
            if (q >= fam.size() || fam[q].empty()) return;
            const uint32_t k = static_cast<uint32_t>(fam[q].size());
            const size_t w = 2u * static_cast<size_t>(k);
            ENGINE_CHECK_MSG(w <= 0xFFFFFFFFull, "[C, obl:iota](iii): multi-vector too wide.");
            std::vector<uint32_t> aug(static_cast<size_t>(flat) * w, 0u);
            multivector_into(fam[q], flat, "[C, obl:iota](iii)", aug.data(), w, 0u);
            ENGINE_CHECK_MSG(flat == iota_src_[q].size(), "[C, obl:iota](iii): flat dimension mismatch.");
            iota_apply_rows(q, aug.data(), w, aug.data() + k, w, k);   // columns [0,k) -> [k,2k)
            std::vector<uint32_t> piv;
#if OC_ENABLE_LEGACY
            if (flat > 0u && engine::elim::use_legacy()) {
                piv = iota_carry_pivots_legacy(flat, static_cast<uint32_t>(w), ell_, aug);   // legacy/HeckeInvolutionLegacy.inc
            } else
#endif
            if (flat > 0u) {
                const engine::ModCtx<uint32_t> ctx(ell_);
                engine::DenseBuf<uint32_t> buf(aug.data(), aug.size(), flat, static_cast<uint32_t>(w),
                                               engine::Layout::RowMajor);
                piv = engine::elim::column_rank_profile<uint32_t>(ctx, engine::view(buf)).pivots;
            }
            const int64_t rank_X = std::count_if(piv.begin(), piv.end(), [k](uint32_t c) { return c < k; });
            if (iota_trace_on()) {
                std::ostringstream w;
                w << "obl(iii) q=" << q << " carry " << name << " [X | iota X] " << flat << "x" << 2u * k << " ("
                  << std::fixed << std::setprecision(1) << (static_cast<double>(aug.size()) * 4.0 / 1048576.0)
                  << " MB) done";
                iota_trace_line(w.str(), 0.0);
            }
            ENGINE_CHECK_MSG(static_cast<int64_t>(piv.size()) == rank_X,
                             (std::string("[C, obl:iota](iii): iota_C leaves the span of the stored ")
                              + name + ".").c_str());
            family_rank_[q][fam_idx] = rank_X;
        };

        tasks.emplace_back([this, q, &K, &C, rho_squared_is_identity, carry_check]() {
            // (iii-a) iota_C^2 = id, and (iii-b) iota_C d = d iota_C, AS MATRICES
            // over cring in every degree.  Both are decided BLOCKWISE and exactly,
            // in O(N_q^2) integer work, and the decision is a PROOF of the matrix
            // identity rather than a sample of it:
            //
            //   (iii-a)  iota_C is the signed block permutation with block
            //            -eps_sigma rho at (sigma, |iota_K sigma|).  Its square has
            //            the single block (-eps_{iota sigma} rho)(-eps_sigma rho) =
            //            eps_sigma eps_{iota sigma} rho^2 at (sigma, iota^2 sigma) =
            //            (sigma, sigma).  So iota_C^2 = id EXACTLY when
            //            eps_{iota sigma} = eps_sigma for every sigma and rho^2 = I.
            //
            //   (iii-b)  with d = D (x) I and rho a constant matrix, the (t, s')
            //            block of d iota_C is -eps_{iota s'} D[t][iota s'] rho and
            //            that of iota_C d is -eps_t D[iota t][s'] rho; putting
            //            s' = iota s, the identity is exactly
            //            D[iota t][iota s] = eps_t eps_s D[t][s], over Z (hence over
            //            every cring).
            //
            // Both are re-asserted here, on this clause's own terms, and the DENSE
            // matrices are materialised and compared entrywise by
            // HeckeInvolutionValidation on its whole grid -- also against the
            // independent oracle hecke::iota_oracle.
            for (uint32_t s = 0; s < K.get_num_q_simplices(q); ++s) {
                const uint32_t t = iota_K_.image_index[q][s];
                ENGINE_CHECK_MSG(iota_K_.image_index[q][t] == s
                                     && iota_K_.image_sign[q][t] == iota_K_.image_sign[q][s],
                                 "[C, obl:iota](iii): iota_C^2 != id (the block permutation is not an "
                                 "involution, or eps_{iota sigma} != eps_sigma).");
            }
            ENGINE_CHECK_MSG(rho_squared_is_identity,
                             "[C, obl:iota](iii): iota_C^2 != id (rho_n(iota)^2 != I).");
            if (q + 1 < num_degrees_) {
                const auto& simplices = K.get_simplices();
                for (uint32_t t = 0; t < simplices[q + 1].size(); ++t)
                    for (uint32_t s = 0; s < simplices[q].size(); ++s) {
                        const int lhs = incidence_sign(simplices[q + 1][iota_K_.image_index[q + 1][t]],
                                                       simplices[q][iota_K_.image_index[q][s]]);
                        const int rhs = iota_K_.image_sign[q + 1][t] * iota_K_.image_sign[q][s]
                                      * incidence_sign(simplices[q + 1][t], simplices[q][s]);
                        ENGINE_CHECK_MSG(lhs == rhs, "[C, obl:iota](iii): iota_C d != d iota_C.");
                    }
            }
            carry_check(C, "C^q_eq", 0u);
        });
        tasks.emplace_back([&Z, carry_check]() { carry_check(Z, "Z^q_eq", 1u); });
        tasks.emplace_back([&B, carry_check]() { carry_check(B, "B^q_eq", 2u); });
        est.push_back(8u * static_cast<uint64_t>(flat) * fam_k(C, q) + 1u);   // [X | iota_C X], u32
        est.push_back(8u * static_cast<uint64_t>(flat) * fam_k(Z, q) + 1u);
        est.push_back(8u * static_cast<uint64_t>(flat) * fam_k(B, q) + 1u);

        // The cochain map and the H-level matrix agree: iota_C H_eq - H_eq iota^*
        // lands in span(B^q_eq), i.e. the stored iota^* really is the map
        // induced on H^q by iota_C.
        const uint32_t dH = cert_.dim_H[q];
        if (dH != 0u && q < H.size() && q < P.size()) {
            tasks.emplace_back([this, q, flat, dH, &B, &H, &P]() {
                const FlatMatrix Bq = (q < B.size() && !B[q].empty())
                                          ? multivector_of(B[q], flat, ell_, "[C, obl:iota](iii)")
                                          : FlatMatrix::CreateZero(flat, 0, ell_);
                const FlatMatrix Hq = multivector_of(H[q], flat, ell_, "[C, obl:iota](iii)");
                const FlatMatrix& Pq = P[q];
                ENGINE_CHECK_MSG(Pq.get_rows() == dH && Pq.get_cols() == flat,
                                 "[C, obl:iota](iii): the stored projector has the wrong shape.");
                ENGINE_CHECK_MSG(iota_H_[q].get_rows() == dH && iota_H_[q].get_cols() == dH,
                                 "[C, obl:iota](iii): the induced map on H^q has the wrong shape.");
                const FlatMatrix IH_flat = iota_apply(q, Hq);
                ENGINE_CHECK_MSG(Pq * IH_flat == iota_H_[q],
                                 "[C, obl:iota](iii): P_q iota_C H_eq != the induced map on H^q.");
                const FlatMatrix diff = IH_flat - (Hq * iota_H_[q]);
                ENGINE_CHECK_MSG(span_contains(Bq, diff),
                                 "[C, obl:iota](iii): iota_C H_eq - H_eq iota^* is not a coboundary.");
                // [C, prop:iotacommutes](2): (iota^*)^2 = id on H^q.
                ENGINE_CHECK_MSG(iota_H_[q] * iota_H_[q] == FlatMatrix(dH, ell_),
                                 "[C, obl:iota](iii): (iota^*)^2 != id on H^q ([C, prop:iotacommutes](2)).");
                if (iota_trace_on())
                    iota_trace_line("obl(iii) q=" + std::to_string(q) + " H-level check (flat " + std::to_string(flat)
                                    + ", B cols " + std::to_string(Bq.get_cols()) + ", dim H " + std::to_string(dH)
                                    + ") done", 0.0);
            });
            // B, H, iota_C H, the difference, H iota^*, [B | diff] and its RREF cache.
            est.push_back(4u * static_cast<uint64_t>(flat) * (3u * fam_k(B, q) + 6u * dH) + 1u);
        }
    }
    if (streaming_) {
        stream_stats_.tasks_iii = static_cast<uint32_t>(tasks.size());
        const uint64_t budget = (task_budget_ != 0u) ? task_budget_ : auto_budget(est);
        stream_stats_.budget_iii = budget;
        const Admission a = run_admitted(tasks, est, budget, false);
        stream_stats_.max_concurrent_iii = a.max_concurrent;
        stream_stats_.peak_admitted_iii = a.peak_admitted;
        stream_stats_.max_task_iii = a.max_task;
    } else {
        engine::tape::run_concurrently(tasks);
    }
    for (uint32_t q = 0; q < num_degrees_; ++q) cert_.iota_H[q] = iota_H_[q];
}

void HeckeInvolution::build_split() {
    // [C, prop:iotasplit].
    const auto& C = coh_->get_C_eq_bases_fm();
    const auto& Z = coh_->get_Z_eq_bases_fm();
    const auto& B = coh_->get_B_eq_bases_fm();

    C_plus_.assign(num_degrees_, FlatMatrix::CreateZero(0, 0, ell_));
    C_minus_ = C_plus_; Z_plus_ = C_plus_; Z_minus_ = C_plus_;
    B_plus_ = C_plus_; B_minus_ = C_plus_; H_plus_ = C_plus_; H_minus_ = C_plus_;
    cert_.char_two_rank_n.assign(num_degrees_, 0u);
    cert_.char_two_dim_ker_n.assign(num_degrees_, 0u);

    if (ell_ == 2u) {
        // [C, prop:iotasplit](2): the decomposition is FALSE; what holds is the
        // filtration 0 <= im n <= ker n <= H^q for n := iota^* - id (n^2 = 0 in
        // characteristic 2).  No +- basis and no halving of any matrix may be
        // claimed here.  Recorded: b = rank n free F_2[x]/(x^2)-summands and
        // a = dim ker n - b trivial ones.
        split_status_ = SignSplitStatus::NotApplicableCharTwo;
        cert_.split_applicable = false;
        for (uint32_t q = 0; q < num_degrees_; ++q) {
            const uint32_t dH = cert_.dim_H[q];
            if (dH == 0u) continue;
            const FlatMatrix n = iota_H_[q] - FlatMatrix(dH, ell_);
            ENGINE_CHECK_MSG((n * n).is_zero(),
                             "[C, prop:iotasplit](2): (iota^* - id)^2 != 0 in characteristic 2.");
            const uint32_t rk = n.rank();
            cert_.char_two_rank_n[q] = rk;
            cert_.char_two_dim_ker_n[q] = dH - rk;
        }
        cert_.C_plus = C_plus_; cert_.C_minus = C_minus_;
        cert_.Z_plus = Z_plus_; cert_.Z_minus = Z_minus_;
        cert_.B_plus = B_plus_; cert_.B_minus = B_minus_;
        cert_.H_plus = H_plus_; cert_.H_minus = H_minus_;
        return;
    }

    split_status_ = SignSplitStatus::Split;
    cert_.split_applicable = true;
    // 1/2 through the engine's arithmetic context (Phase 2), never a raw
    // ModArithmetic::mod_inv (docs/matrix-engine/tools/raw_mod_calls.allow).
    const engine::ModCtx<uint32_t> ctx(ell_);
    const uint32_t inv2 = ctx.inv(2u % ell_);

    // The flat families: pr_+- applied to the stored basis, then the D5b image
    // basis of the resulting multi-vector.
    // (slot-22 PERF1, P3) pr_+- X = (X +- iota_C X) / 2 for the whole stored
    // multi-vector at once: iota_C X by the checked signed permutation, then
    // entrywise (x + ix) * inv2 and (x - ix) * inv2 -- the same residues the
    // per-column FlatMatrix sum / difference / scalar product gave, assembled
    // into the same row-major FlatMatrix that columns_of built from them.  rank X
    // is check_obl_iii's (read off its elimination of [X | iota_C X]) when that
    // was recorded, else computed here as before.
    auto flat_split = [&](const std::vector<std::vector<FlatMatrix>>& fam, uint32_t q, uint32_t fam_idx,
                          FlatMatrix& plus, FlatMatrix& minus) {
        const uint32_t flat = cert_.flat_dim[q];
        plus = FlatMatrix::CreateZero(flat, 0, ell_);
        minus = FlatMatrix::CreateZero(flat, 0, ell_);
        if (q >= fam.size() || fam[q].empty()) return;
        const uint32_t k = static_cast<uint32_t>(fam[q].size());
        const size_t n = static_cast<size_t>(flat) * k;
        ENGINE_CHECK_MSG(flat == iota_src_[q].size(), "[C, prop:iotasplit]: flat dimension mismatch.");
        // Transients are released at their last reader, so at most three
        // flat x k buffers are alive at once (the per-column version held six).
        std::vector<uint32_t> pd(n), md(n);
        uint32_t rank_X = 0;
        {
            std::vector<uint32_t> x(n, 0u);
            multivector_into(fam[q], flat, "[C, prop:iotasplit]", x.data(), k, 0u);
            const int64_t cached = (q < family_rank_.size()) ? family_rank_[q][fam_idx] : -1;
            rank_X = (cached >= 0) ? static_cast<uint32_t>(cached) : FlatMatrix(flat, k, ell_, x).rank();
            iota_apply_rows(q, x.data(), k, pd.data(), k, k);   // pd holds iota_C X for the moment
            for (size_t e = 0; e < n; ++e) {
                const uint32_t ix = pd[e];
                pd[e] = ctx.mul(ctx.add(x[e], ix), inv2);
                md[e] = ctx.mul(ctx.sub(x[e], ix), inv2);
            }
        }
        {
            const FlatMatrix Pmat(flat, k, ell_, pd);
            std::vector<uint32_t>().swap(pd);
            plus = columns_of(Pmat.image(), flat, ell_);
        }
        {
            const FlatMatrix Mmat(flat, k, ell_, md);
            std::vector<uint32_t>().swap(md);
            minus = columns_of(Mmat.image(), flat, ell_);
        }
        ENGINE_CHECK_MSG(plus.get_cols() + minus.get_cols() == rank_X,
                         "[C, prop:iotasplit](1): the +- images do not exhaust the stored space.");
        if (iota_trace_on())
            iota_trace_line("split q=" + std::to_string(q) + " family " + std::to_string(fam_idx) + " " + std::to_string(flat)
                            + "x" + std::to_string(k) + " -> +" + std::to_string(plus.get_cols()) + " / -"
                            + std::to_string(minus.get_cols()) + " done", 0.0);
    };

    // (slot-23 P4I) The streamed flat split: the same plus / minus as
    // flat_split, entry for entry, holding one flat x k buffer at a time.
    //   * pr_+- X = (X +- iota_C X) / 2 is formed straight from the stored
    //     cochains, one cochain (= one column) at a time, with flat_split's
    //     residue formula: x = X[r][t], ix = +-X[src[r]][t] (iota_apply_rows'
    //     negation), value = (x +- ix) * inv2 via the same ModCtx calls;
    //   * Pmat.image() is the list of Pmat's PIVOT COLUMNS (the leftmost
    //     independent ones, FlatMatrix::image, owner decision Q11), and the
    //     pivot columns of a matrix are a property of the matrix: the column
    //     rank profile (engine::elim::column_rank_profile, "exactly rref()'s
    //     pivots") of the buffer, taken in place, gives the same list; under
    //     ENGINE_BACKEND=legacy the FlatMatrix route's own pivots are taken;
    //   * the buffer is released and the flat x r output is filled from those
    //     pivot columns, recomputed with the same formula -- columns_of's
    //     row-major layout, adopted without a copy (FromCanonical: the entries
    //     are canonical residues, the flags are determined from the data as the
    //     vector constructor does).
    // rank X is check_obl_iii's, as in flat_split.
    const uint32_t mutation = stream_mutation_;
    uint32_t mutation_q = num_degrees_;   // mutations 1 / 2 act on the C family of the lowest degree that has one
    for (uint32_t q = 0; q < num_degrees_; ++q)
        if (q < C.size() && !C[q].empty()) { mutation_q = q; break; }
    std::atomic<uint32_t> streamed_families{0}, mutations_applied{0};
    auto flat_split_streamed = [&](const std::vector<std::vector<FlatMatrix>>& fam, uint32_t q, uint32_t fam_idx,
                                   FlatMatrix& plus, FlatMatrix& minus) {
        const uint32_t flat = cert_.flat_dim[q];
        plus = FlatMatrix::CreateZero(flat, 0, ell_);
        minus = FlatMatrix::CreateZero(flat, 0, ell_);
        if (q >= fam.size() || fam[q].empty()) return;
        const uint32_t k = static_cast<uint32_t>(fam[q].size());
        const size_t n = static_cast<size_t>(flat) * k;
        ENGINE_CHECK_MSG(flat == iota_src_[q].size(), "[C, prop:iotasplit]: flat dimension mismatch.");
        const std::vector<uint32_t>& src = iota_src_[q];
        const std::vector<uint8_t>& neg = iota_neg_[q];
        streamed_families.fetch_add(1u, std::memory_order_relaxed);

        // Stored cochain t as one flat column (multivector_into's entries).
        std::vector<uint32_t> col;
        auto column = [&](uint32_t t) {
            const FlatMatrix& c = fam[q][t];
            const uint32_t dv = c.get_cols();
            ENGINE_CHECK_MSG(static_cast<size_t>(c.get_rows()) * dv == flat,
                             "[C, prop:iotasplit]: a stored basis cochain has the wrong shape.");
            col.resize(flat);
            for (uint32_t s = 0; s < c.get_rows(); ++s)
                for (uint32_t j = 0; j < dv; ++j) col[static_cast<size_t>(s) * dv + j] = c.get_entry(s, j);
        };
        auto value = [&](size_t r, bool plus_side) -> uint32_t {
            const uint32_t x = col[r];
            const uint32_t xs = col[src[r]];
            const uint32_t ix = neg[r] ? ((xs == 0u) ? 0u : ell_ - xs) : xs;
            return plus_side ? ctx.mul(ctx.add(x, ix), inv2) : ctx.mul(ctx.sub(x, ix), inv2);
        };

        uint32_t rank_X = 0;
        {
            const int64_t cached = (q < family_rank_.size()) ? family_rank_[q][fam_idx] : -1;
            if (cached >= 0) {
                rank_X = static_cast<uint32_t>(cached);
            } else {
                std::vector<uint32_t> x(n, 0u);
                multivector_into(fam[q], flat, "[C, prop:iotasplit]", x.data(), k, 0u);
                rank_X = FlatMatrix(flat, k, ell_, x).rank();
            }
        }

        auto pivots_of = [&](bool plus_side) {
            std::vector<uint32_t> buf(n);
            for (uint32_t t = 0; t < k; ++t) {
                column(t);
                for (size_t r = 0; r < flat; ++r) buf[r * k + t] = value(r, plus_side);
            }
            std::vector<uint32_t> piv;
#if OC_ENABLE_LEGACY
            if (engine::elim::use_legacy()) {
                piv = iota_split_pivots_legacy(flat, k, ell_, buf);   // legacy/HeckeInvolutionLegacy.inc
            } else
#endif
            {
                engine::DenseBuf<uint32_t> dbuf(buf.data(), buf.size(), flat, k, engine::Layout::RowMajor);
                piv = engine::elim::column_rank_profile<uint32_t>(ctx, engine::view(dbuf)).pivots;
            }
            return piv;
        };
        auto output = [&](const std::vector<uint32_t>& piv, bool plus_side) -> FlatMatrix {
            const uint32_t r = static_cast<uint32_t>(piv.size());
            if (r == 0u) return FlatMatrix(flat, 0, ell_, std::vector<uint32_t>());
            std::vector<uint32_t> data(static_cast<size_t>(flat) * r);
            for (uint32_t j = 0; j < r; ++j) {
                column(piv[j]);
                for (size_t i = 0; i < flat; ++i) data[i * r + j] = value(i, plus_side);
            }
            return FlatMatrix::FromCanonical(flat, r, ell_, std::move(data));
        };

        const bool target = (fam_idx == 0u && q == mutation_q);
        for (int side = 0; side < 2; ++side) {
            const bool plus_side = (side == 0);
            std::vector<uint32_t> piv = pivots_of(plus_side);
            if (target && mutation == 2u && plus_side && !piv.empty() && piv.size() < k) {
                // Validation-only: the last pivot replaced by the largest non-pivot column.
                std::vector<char> is_piv(k, 0);
                for (uint32_t c : piv) is_piv[c] = 1;
                uint32_t c = k;
                while (c > 0u && is_piv[c - 1u]) --c;
                if (c > 0u) {
                    piv.back() = c - 1u;
                    std::sort(piv.begin(), piv.end());
                    mutations_applied.fetch_add(1u, std::memory_order_relaxed);
                }
            }
            FlatMatrix out = output(piv, plus_side);
            if (target && mutation == 1u && plus_side && out.get_cols() > 0u) {
                out.set_entry(0, 0, ctx.add(out.get_entry(0, 0), 1u % ell_));   // validation-only
                mutations_applied.fetch_add(1u, std::memory_order_relaxed);
            }
            (plus_side ? plus : minus) = std::move(out);
        }
        ENGINE_CHECK_MSG(plus.get_cols() + minus.get_cols() == rank_X,
                         "[C, prop:iotasplit](1): the +- images do not exhaust the stored space.");
        if (iota_trace_on())
            iota_trace_line("split q=" + std::to_string(q) + " family " + std::to_string(fam_idx) + " " + std::to_string(flat)
                            + "x" + std::to_string(k) + " -> +" + std::to_string(plus.get_cols()) + " / -"
                            + std::to_string(minus.get_cols()) + " done (streamed)", 0.0);
    };
    auto split_family = [&](const std::vector<std::vector<FlatMatrix>>& fam, uint32_t q, uint32_t fam_idx,
                            FlatMatrix& plus, FlatMatrix& minus) {
        if (streaming_) flat_split_streamed(fam, q, fam_idx, plus, minus);
        else flat_split(fam, q, fam_idx, plus, minus);
    };

    // (slot-10 WP-7) Every (degree, family) split below is an exact function of
    // read-only inputs (the stored bases, the morphism -- whose pull-back
    // operators check_obl_iii already built -- and iota_H_) writing only its
    // own slots, so they run as host tasks (run_concurrently) in the serial
    // order: per degree q, [C], [Z], [B], [H].  The first failing task's
    // exception IN TASK ORDER is what surfaces (serially, in order, on a
    // one-worker pool), i.e. the one the serial loop surfaced.
    std::vector<std::function<void()>> tasks;
    std::vector<uint64_t> est;   // (P4I) the streamed buffer + the output, bytes (admission only)
    auto split_est = [&](const std::vector<std::vector<FlatMatrix>>& fam, uint32_t q) -> uint64_t {
        const uint64_t k = (q < fam.size()) ? fam[q].size() : 0u;
        return 8u * static_cast<uint64_t>(cert_.flat_dim[q]) * k + 1u;
    };
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        tasks.emplace_back([&, q]() { split_family(C, q, 0u, C_plus_[q], C_minus_[q]); });
        tasks.emplace_back([&, q]() { split_family(Z, q, 1u, Z_plus_[q], Z_minus_[q]); });
        tasks.emplace_back([&, q]() { split_family(B, q, 2u, B_plus_[q], B_minus_[q]); });
        est.push_back(split_est(C, q));
        est.push_back(split_est(Z, q));
        est.push_back(split_est(B, q));
        est.push_back(32u * static_cast<uint64_t>(cert_.dim_H[q]) * cert_.dim_H[q] + 1u);
        tasks.emplace_back([&, q]() {
            const uint32_t dH = cert_.dim_H[q];
            if (dH == 0u) {
                H_plus_[q] = FlatMatrix::CreateZero(0, 0, ell_);
                H_minus_[q] = FlatMatrix::CreateZero(0, 0, ell_);
                return;
            }
            const FlatMatrix I(dH, ell_);
            const FlatMatrix pr_plus = (I + iota_H_[q]) * inv2;
            const FlatMatrix pr_minus = (I - iota_H_[q]) * inv2;
            // Complementary idempotents commuting with everything iota^* commutes with.
            ENGINE_CHECK_MSG(pr_plus * pr_plus == pr_plus && pr_minus * pr_minus == pr_minus
                                 && (pr_plus * pr_minus).is_zero() && pr_plus + pr_minus == I,
                             "[C, prop:iotasplit](1): pr_+- are not complementary idempotents.");
            H_plus_[q] = columns_of(pr_plus.image(), dH, ell_);
            H_minus_[q] = columns_of(pr_minus.image(), dH, ell_);
            ENGINE_CHECK_MSG(H_plus_[q].get_cols() + H_minus_[q].get_cols() == dH,
                             "[C, prop:iotasplit](1): dim H^q_+ + dim H^q_- != dim H^q.");
            // im pr_+- == ker(iota^* -+ id), entrywise on the span.
            for (int sgn = 0; sgn < 2; ++sgn) {
                const FlatMatrix& Ubasis = (sgn == 0) ? H_plus_[q] : H_minus_[q];
                if (Ubasis.get_cols() == 0u) continue;
                const FlatMatrix want = (sgn == 0) ? Ubasis : (Ubasis * (ell_ - 1u));
                ENGINE_CHECK_MSG(iota_H_[q] * Ubasis == want,
                                 "[C, prop:iotasplit](1): im pr_+- is not the corresponding eigenspace.");
            }
        });
    }
    if (streaming_) {
        stream_stats_.tasks_split = static_cast<uint32_t>(tasks.size());
        const uint64_t budget = (task_budget_ != 0u) ? task_budget_ : auto_budget(est);
        stream_stats_.budget_split = budget;
        const Admission a = run_admitted(tasks, est, budget, mutation == 3u);
        stream_stats_.max_concurrent_split = a.max_concurrent;
        stream_stats_.peak_admitted_split = a.peak_admitted;
        stream_stats_.max_task_split = a.max_task;
        stream_stats_.streamed_families = streamed_families.load();
        stream_stats_.mutations_applied = mutations_applied.load() + a.skipped;
        // ONE copy of the flat +- bases: moved into the certificate, which the
        // accessors read (the members are left empty).
        auto give = [](std::vector<FlatMatrix>& from, std::vector<FlatMatrix>& to) {
            to = std::move(from);
            std::vector<FlatMatrix>().swap(from);
        };
        give(C_plus_, cert_.C_plus); give(C_minus_, cert_.C_minus);
        give(Z_plus_, cert_.Z_plus); give(Z_minus_, cert_.Z_minus);
        give(B_plus_, cert_.B_plus); give(B_minus_, cert_.B_minus);
        cert_.H_plus = H_plus_; cert_.H_minus = H_minus_;
        return;
    }
    engine::tape::run_concurrently(tasks);
    cert_.C_plus = C_plus_; cert_.C_minus = C_minus_;
    cert_.Z_plus = Z_plus_; cert_.Z_minus = Z_minus_;
    cert_.B_plus = B_plus_; cert_.B_minus = B_minus_;
    cert_.H_plus = H_plus_; cert_.H_minus = H_minus_;
}

void HeckeInvolution::compute() {
    if (computed_) return;
    // (slot-23 PERF-P2) The host eliminations of (i)-(iii), build_split's image() calls and the
    // certificate split their trailing updates over the consumer pool; run_concurrently()'s task
    // threads inherit the hook (EngineRuntime.h).  Value-free (HostElimThreadDeterminismValidation).
    const engine::tape::ScopedHostElimHook host_elim_hook;
    cert_ = InvolutionCertificate();
    cert_.ell = ell_;
    cert_.n_sym = n_sym_;
    cert_.d_V = d_V_;
    cert_.num_degrees = num_degrees_;

    // (slot-23 P4I) the switches, read once on the calling thread.
    streaming_ = g_iota_streaming;
    task_budget_ = g_iota_task_budget;
    stream_mutation_ = streaming_ ? g_iota_stream_mutation : 0;
    stream_stats_ = StreamStats();
    stream_stats_.streamed = streaming_;
    {
        static std::once_flag effective_once;
        std::call_once(effective_once, []() {
            if (const char* v = std::getenv("HECKE_IOTA_STREAM"))
                std::cout << "EFFECTIVE HECKE_IOTA_STREAM iota=" << (iota_env_stream() ? "streamed" : "materialised")
                          << " (HECKE_IOTA_STREAM=" << v << ")" << std::endl;
        });
    }
    IotaTrace trace;
    trace.mark(std::string("compute enter (") + (streaming_ ? "streamed" : "materialised") + ")");
    build_rho();
    cert_.rho_iota = rho_iota_;
    check_obl_i();
    trace.mark("rho + obl(i)");
    build_morphism();
    trace.mark("morphism + induced maps on H");
    build_signed_permutation();
    check_obl_ii();
    trace.mark("signed permutation + obl(ii)");
    check_obl_iii();
    trace.mark("obl(iii)");
    build_split();
    trace.mark("split");
    if (iota_trace_on() && streaming_) {
        std::ostringstream w;
        w << std::fixed << std::setprecision(1) << "admission: obl(iii) " << stream_stats_.tasks_iii
          << " tasks, budget " << stream_stats_.budget_iii / 1048576.0 << " MB, peak admitted "
          << stream_stats_.peak_admitted_iii / 1048576.0 << " MB, max concurrent " << stream_stats_.max_concurrent_iii
          << "; split " << stream_stats_.tasks_split << " tasks, budget " << stream_stats_.budget_split / 1048576.0
          << " MB, peak admitted " << stream_stats_.peak_admitted_split / 1048576.0 << " MB, max concurrent "
          << stream_stats_.max_concurrent_split << "; streamed families " << stream_stats_.streamed_families;
        iota_trace_line(w.str(), 0.0);
    }

    std::vector<std::string> failures;
    const uint32_t code = cert_.verify(failures);
    trace.mark("certificate verify");
    if (code != 0u) {
        std::ostringstream os;
        os << "HeckeInvolution: the certificate of [C, obl:iota](i)-(iii) failed with code " << code
           << " (" << InvolutionCertificate::verify_message(code) << ")";
        for (const auto& f : failures) os << "\n  " << f;
        ENGINE_CHECK_MSG(false, os.str().c_str());
    }
    computed_ = true;
}

const EquivariantCochainComplexMorphismOnK& HeckeInvolution::morphism() const {
    ENGINE_CHECK_MSG(morphism_ != nullptr, "HeckeInvolution::morphism: compute() has not run.");
    return *morphism_;
}

FlatMatrix HeckeInvolution::cochain_matrix(uint32_t q, int global_sign) const {
    ENGINE_CHECK_MSG(morphism_ != nullptr, "HeckeInvolution::cochain_matrix: compute() has not run.");
    ENGINE_CHECK_MSG(q < num_degrees_, "HeckeInvolution::cochain_matrix: degree out of range.");
    ENGINE_CHECK_MSG(global_sign == 1 || global_sign == -1,
                     "HeckeInvolution::cochain_matrix: global_sign must be +-1.");
    const SimplicialComplex& K = *coh_->get_lssc()->get_base_K_complex();
    const uint32_t N_q = K.get_num_q_simplices(q);
    const uint32_t dim = N_q * d_V_;
    std::vector<uint32_t> data(static_cast<size_t>(dim) * dim, 0u);
    for (uint32_t s = 0; s < N_q; ++s) {
        const uint32_t t = iota_K_.image_index[q][s];
        // def:iota(iv): block  global_sign * eps_sigma * rho_n(iota).
        const bool minus = (iota_K_.image_sign[q][s] * global_sign) < 0;
        for (uint32_t a = 0; a < d_V_; ++a)
            for (uint32_t b = 0; b < d_V_; ++b) {
                uint32_t v = rho_iota_.get_entry(a, b);
                if (minus && v != 0u) v = ell_ - v;
                data[static_cast<size_t>(s * d_V_ + a) * dim + (t * d_V_ + b)] = v;
            }
    }
    return FlatMatrix(dim, dim, ell_, data);
}

FlatMatrix HeckeInvolution::apply_to_flat_cochain(uint32_t q, const FlatMatrix& c_flat) const {
    ENGINE_CHECK_MSG(morphism_ != nullptr, "HeckeInvolution::apply_to_flat_cochain: compute() has not run.");
    return morphism_->apply_to_flat_cochain(q, c_flat);
}

const FlatMatrix& HeckeInvolution::matrix(uint32_t q) const {
    require_computed("matrix");
    ENGINE_CHECK_MSG(q < num_degrees_, "HeckeInvolution::matrix: degree out of range.");
    return iota_H_[q];
}

uint32_t HeckeInvolution::dim_H(uint32_t q) const {
    require_computed("dim_H");
    ENGINE_CHECK_MSG(q < num_degrees_, "HeckeInvolution::dim_H: degree out of range.");
    return cert_.dim_H[q];
}

SignSplitStatus HeckeInvolution::sign_split_status() const {
    require_computed("sign_split_status");
    return split_status_;
}

#define OC_IOTA_BASIS_ACCESSOR(NAME, MEMBER)                                                   \
    const FlatMatrix& HeckeInvolution::NAME(uint32_t q) const {                                \
        require_computed(#NAME);                                                               \
        ENGINE_CHECK_MSG(q < num_degrees_, "HeckeInvolution::" #NAME ": degree out of range."); \
        return MEMBER[q];                                                                      \
    }
OC_IOTA_BASIS_ACCESSOR(H_plus_basis, H_plus_)
OC_IOTA_BASIS_ACCESSOR(H_minus_basis, H_minus_)
// (slot-23 P4I) the flat +- bases live once, in the certificate (a copy of
// the members on the rollback route, the moved members on the streamed one).
OC_IOTA_BASIS_ACCESSOR(C_eq_plus_basis, cert_.C_plus)
OC_IOTA_BASIS_ACCESSOR(C_eq_minus_basis, cert_.C_minus)
OC_IOTA_BASIS_ACCESSOR(Z_eq_plus_basis, cert_.Z_plus)
OC_IOTA_BASIS_ACCESSOR(Z_eq_minus_basis, cert_.Z_minus)
OC_IOTA_BASIS_ACCESSOR(B_eq_plus_basis, cert_.B_plus)
OC_IOTA_BASIS_ACCESSOR(B_eq_minus_basis, cert_.B_minus)
#undef OC_IOTA_BASIS_ACCESSOR

uint32_t HeckeInvolution::dim_H_plus(uint32_t q) const { return H_plus_basis(q).get_cols(); }
uint32_t HeckeInvolution::dim_H_minus(uint32_t q) const { return H_minus_basis(q).get_cols(); }

const InvolutionCertificate& HeckeInvolution::certificate() const {
    require_computed("certificate");
    return cert_;
}

std::string HeckeInvolution::to_string() const { return to_string_offset(0); }

std::string HeckeInvolution::to_string_offset(uint32_t offset) const {
    const std::string pad(offset, ' ');
    std::ostringstream os;
    os << pad << "HeckeInvolution \"" << label_ << "\" [C, sec:involution], iota^* = -iota^*_geo "
       << "(def:iota(iv), v1.2.0)\n";
    os << pad << "  ell = " << ell_ << ", n_sym = " << n_sym_ << ", d_V = " << d_V_
       << ", degrees = " << num_degrees_ << "\n";
    os << pad << "  iota_K vertex perm =";
    for (uint32_t v : iota_K_.vertex_perm) os << " " << v;
    os << ", minus signs = " << iota_K_.num_minus_signs() << "\n";
    if (!computed_) { os << pad << "  (not computed)\n"; return os.str(); }
    os << pad << "  sign_split = " << hecke::to_string(split_status_) << "\n";
    for (uint32_t q = 0; q < num_degrees_; ++q) {
        os << pad << "  q = " << q << ": dim H^q = " << cert_.dim_H[q];
        if (split_status_ == SignSplitStatus::Split)
            os << ", (dim H^q_+, dim H^q_-) = (" << H_plus_[q].get_cols() << ", " << H_minus_[q].get_cols() << ")";
        else
            os << ", rank(iota^* - id) = " << cert_.char_two_rank_n[q]
               << ", dim ker = " << cert_.char_two_dim_ker_n[q];
        os << "\n";
    }
    return os.str();
}

std::string HeckeInvolution::to_json() const {
    std::ostringstream os;
    os << "{\"schema\": \"hecke_involution/v1\", \"label\": \"" << label_ << "\"";
    os << ", \"ell\": " << ell_ << ", \"n_sym\": " << n_sym_ << ", \"d_V\": " << d_V_;
    os << ", \"iota_sign\": \"eisenstein_in_H1_plus\"";
    os << ", \"iota_normalisation\": \"iota^* = -iota^*_geo\"";
    os << ", \"vertex_perm\": [";
    for (size_t i = 0; i < iota_K_.vertex_perm.size(); ++i) os << (i ? ", " : "") << iota_K_.vertex_perm[i];
    os << "], \"orientation_minus_signs\": " << iota_K_.num_minus_signs();
    os << ", \"sign_split\": \"" << hecke::to_string(split_status_) << "\"";
    if (computed_) {
        os << ", \"q_max\": " << q_max() << ", \"truncated\": false, \"q_max_source\": \"derived\"";
        os << ", \"degrees\": [";
        for (uint32_t q = 0; q < num_degrees_; ++q) os << (q ? ", " : "") << q;
        os << "], \"dim_H\": [";
        for (uint32_t q = 0; q < num_degrees_; ++q) os << (q ? ", " : "") << cert_.dim_H[q];
        os << "]";
        if (split_status_ == SignSplitStatus::Split) {
            os << ", \"dim_H_plus\": [";
            for (uint32_t q = 0; q < num_degrees_; ++q) os << (q ? ", " : "") << H_plus_[q].get_cols();
            os << "], \"dim_H_minus\": [";
            for (uint32_t q = 0; q < num_degrees_; ++q) os << (q ? ", " : "") << H_minus_[q].get_cols();
            os << "]";
        } else {
            os << ", \"char_two_rank_n\": [";
            for (uint32_t q = 0; q < num_degrees_; ++q) os << (q ? ", " : "") << cert_.char_two_rank_n[q];
            os << "], \"char_two_dim_ker_n\": [";
            for (uint32_t q = 0; q < num_degrees_; ++q) os << (q ? ", " : "") << cert_.char_two_dim_ker_n[q];
            os << "]";
        }
    }
    os << "}";
    return os.str();
}

// ---------------------------------------------------------------------------
// InvolutionCertificate::verify -- re-derives, never reads.
// ---------------------------------------------------------------------------

const char* InvolutionCertificate::verify_message(uint32_t code) {
    switch (code) {
        case 0: return "ok";
        case 1: return "shape: ell / n_sym / d_V / degree count out of range or inconsistent";
        case 2: return "[C, eq:rhoniota]: rho_n(iota) is not diag((-1)^{n-j}) or rho^2 != I";
        case 3: return "[C, obl:iota](i): iota_K is not an involutive simplicial map with eps_{iota s} = eps_s";
        case 4: return "[C, lem:iotamodel]: the seven-vertex sign table does not have six minus signs";
        case 5: return "[C, obl:iota](ii): a relation row has no matching inverse row";
        case 6: return "[C, obl:iota](ii): the intertwining fibre identity fails on a relation row";
        case 7: return "[C, obl:iota](iii): (iota^*)^2 != id on some H^q";
        case 8: return "[C, prop:iotasplit](1): the +- bases do not exhaust H^q";
        case 9: return "[C, prop:iotasplit](1): a +- basis is not the corresponding eigenspace";
        case 10: return "[C, prop:iotasplit](2): (iota^* - id)^2 != 0 at ell = 2, or the recorded ranks are wrong";
        case 11: return "the split status disagrees with 2 in R^x";
        default: return "unknown code";
    }
}

std::vector<uint32_t> InvolutionCertificate::failing_items() const {
    std::vector<std::string> ignored;
    std::vector<uint32_t> out;
    const uint32_t first = verify(ignored);
    if (first != 0u) out.push_back(first);
    return out;
}

uint32_t InvolutionCertificate::verify() const {
    std::vector<std::string> ignored;
    return verify(ignored);
}

uint32_t InvolutionCertificate::verify(std::vector<std::string>& failures) const {
    failures.clear();
    // (slot-23 PERF-P2) the rank checks on the consumer pool (value-free; a no-op when
    // HeckeInvolution::compute() already installed it on this thread).
    const engine::tape::ScopedHostElimHook host_elim_hook;
    auto fail = [&](uint32_t code, const std::string& what) {
        failures.push_back(std::to_string(code) + ": " + what);
        return code;
    };

    // 1 -- shape.
    if (ell < 2u || d_V == 0u || d_V != n_sym + 1u || num_degrees == 0u
        || iota_K.num_degrees() != num_degrees || dim_H.size() != num_degrees
        || flat_dim.size() != num_degrees || iota_H.size() != num_degrees)
        return fail(1, "inconsistent shape");

    // 2 -- rho_n(iota), RE-DERIVED from eq:rhoniota (not read).
    if (rho_iota.get_rows() != d_V || rho_iota.get_cols() != d_V || rho_iota.get_p() != ell)
        return fail(2, "rho_n(iota) shape");
    for (uint32_t a = 0; a < d_V; ++a)
        for (uint32_t b = 0; b < d_V; ++b) {
            const uint32_t want = (a != b) ? 0u : ((((n_sym - a) % 2u) == 1u) ? (ell - 1u) % ell : 1u % ell);
            if (rho_iota.get_entry(a, b) != want) return fail(2, "rho_n(iota) entry");
        }
    if (!(rho_iota * rho_iota == FlatMatrix(d_V, ell))) return fail(2, "rho_n(iota)^2 != I");

    // 3 -- obl:iota(i), re-derived from the vertex permutation alone.
    for (uint32_t v = 0; v < iota_K.vertex_perm.size(); ++v)
        if (iota_K.vertex_perm[v] >= iota_K.vertex_perm.size()
            || iota_K.vertex_perm[iota_K.vertex_perm[v]] != v)
            return fail(3, "vertex permutation is not an involution");
    for (uint32_t q = 0; q < num_degrees; ++q) {
        const size_t N_q = iota_K.image_index[q].size();
        if (iota_K.image_sign[q].size() != N_q) return fail(3, "sign table shape");
        for (size_t s = 0; s < N_q; ++s) {
            const uint32_t t = iota_K.image_index[q][s];
            if (t >= N_q || iota_K.image_index[q][t] != s) return fail(3, "iota_K is not an involution");
            if (iota_K.image_sign[q][t] != iota_K.image_sign[q][s]) return fail(3, "eps_{iota s} != eps_s");
            if (iota_K.image_sign[q][s] != 1 && iota_K.image_sign[q][s] != -1)
                return fail(3, "orientation sign is not +-1");
        }
    }

    // 4 -- lem:iotamodel's count, on the seven-vertex model only.
    if (model_is_seven_vertex) {
        uint32_t minus = 0;
        for (const auto& per_q : iota_K.image_sign)
            for (int s : per_q) if (s < 0) ++minus;
        if (minus != 6u || minus != minus_sign_count) return fail(4, "six minus signs");
        const std::vector<uint32_t> expected{ 3, 2, 1, 0, 5, 4, 6 };
        if (iota_K.vertex_perm != expected) return fail(4, "lem:iotamodel's vertex permutation");
    }

    // 5/6 -- obl:iota(ii): the stored match is re-checked, never trusted.
    for (const auto& row : relations) {
        if (!row.matched || row.match_index >= relations.size()) return fail(5, "unmatched relation row");
        const auto& other = relations[row.match_index];
        const uint32_t q = row.q_dim;
        if (q >= num_degrees) return fail(5, "relation degree out of range");
        if (other.q_dim != q) return fail(5, "matched row has another degree");
        if (other.s_lhs != iota_K.image_index[q][row.s_rhs]
            || other.s_rhs != iota_K.image_index[q][row.s_lhs])
            return fail(5, "matched row is not the image's inverse");
        if (iota_K.image_sign[q][row.s_lhs] != iota_K.image_sign[q][row.s_rhs])
            return fail(5, "a relation row's two sides carry different orientation signs");
        if (!(row.M * (rho_iota * other.M) == rho_iota))
            return fail(6, "rho_n(iota) != M rho_n(iota) M'");
        if (exact_relation_data && row.has_exact && other.has_exact) {
            const Mat22 iv{ -1, 0, 0, 1 };
            if (!(other.g_r == (iv * row.g_r * iv).inv_unimodular()))
                return fail(6, "the matched exact group element is not c_iota(gamma)^{-1}");
        }
    }

    // 7 -- obl:iota(iii) on H^q.
    for (uint32_t q = 0; q < num_degrees; ++q) {
        const uint32_t dH = dim_H[q];
        if (iota_H[q].get_rows() != dH || iota_H[q].get_cols() != dH) return fail(7, "iota^* shape on H^q");
        if (dH != 0u && !(iota_H[q] * iota_H[q] == FlatMatrix(dH, ell))) return fail(7, "(iota^*)^2 != id");
    }

    // 8/9/10/11 -- the split.
    if (split_applicable != (ell != 2u)) return fail(11, "split status disagrees with 2 in R^x");
    if (split_applicable) {
        if (H_plus.size() != num_degrees || H_minus.size() != num_degrees) return fail(8, "H^q_+- shape");
        for (uint32_t q = 0; q < num_degrees; ++q) {
            const uint32_t dH = dim_H[q];
            if (H_plus[q].get_cols() + H_minus[q].get_cols() != dH) return fail(8, "dim H_+ + dim H_- != dim H");
            if (dH == 0u) continue;
            if (H_plus[q].get_cols() != 0u && !(iota_H[q] * H_plus[q] == H_plus[q]))
                return fail(9, "H^q_+ is not the (+1)-eigenspace");
            if (H_minus[q].get_cols() != 0u && !(iota_H[q] * H_minus[q] == H_minus[q] * (ell - 1u)))
                return fail(9, "H^q_- is not the (-1)-eigenspace");
            if (hcat_of(H_plus[q], H_minus[q]).rank() != dH) return fail(8, "the +- bases are not independent");
        }
    } else {
        for (uint32_t q = 0; q < num_degrees; ++q) {
            const uint32_t dH = dim_H[q];
            if (dH == 0u) continue;
            const FlatMatrix n = iota_H[q] - FlatMatrix(dH, ell);
            if (!(n * n).is_zero()) return fail(10, "(iota^* - id)^2 != 0 at ell = 2");
            const uint32_t rk = n.rank();
            if (char_two_rank_n.size() != num_degrees || char_two_dim_ker_n.size() != num_degrees)
                return fail(10, "the recorded characteristic-2 data is missing");
            if (char_two_rank_n[q] != rk || char_two_dim_ker_n[q] != dH - rk)
                return fail(10, "the recorded rank / kernel dimension is wrong");
        }
    }
    return 0u;
}

// ---------------------------------------------------------------------------
// Naturality against the restriction.
// ---------------------------------------------------------------------------

bool IotaCommutesWithRestriction(const HeckeInvolution& bulk,
                                 const HeckeInvolution& boundary,
                                 const std::vector<FlatMatrix>& r_per_degree) {
    // [C, eq:iotacochain]'s naturality  r^q iota^* = iota^*_bdry r^q  on H^q.
    // r^q has rows = dim H^q(boundary), cols = dim H^q(bulk).
    const uint32_t Q = static_cast<uint32_t>(r_per_degree.size());
    for (uint32_t q = 0; q < Q; ++q) {
        const FlatMatrix& r = r_per_degree[q];
        if (r.get_rows() == 0u || r.get_cols() == 0u) continue;
        if (q >= bulk.num_degrees() || q >= boundary.num_degrees()) continue;
        const FlatMatrix& ib = bulk.matrix(q);
        const FlatMatrix& id = boundary.matrix(q);
        ENGINE_CHECK_MSG(r.get_cols() == ib.get_rows() && r.get_rows() == id.get_rows(),
                         "IotaCommutesWithRestriction: shape mismatch between r^q and the two involutions.");
        if (!(r * ib == id * r)) return false;
    }
    return true;
}


// ===========================================================================
// (hecke-extensions slot 20, WP-LV3) iota^* over Z/ell^k -- see the header.
// ===========================================================================
namespace {
struct RingZq {
    uint64_t ell = 0, q = 1;
    uint32_t k = 0;
    uint64_t mul(uint64_t a, uint64_t b) const { return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % q); }
    uint64_t add(uint64_t a, uint64_t b) const { return static_cast<uint64_t>((static_cast<unsigned __int128>(a) + b) % q); }
    uint64_t sub(uint64_t a, uint64_t b) const { return a >= b ? a - b : a + (q - b); }
    uint32_t val(uint64_t a) const {
        a %= q;
        if (a == 0) return k;
        uint32_t v = 0;
        while (a % ell == 0) { a /= ell; ++v; }
        return v;
    }
    uint64_t inv(uint64_t a) const {
        // a an ell-unit: extended Euclid in signed 128-bit
        __int128 t = 0, nt = 1, r = static_cast<__int128>(q), nr = static_cast<__int128>(a % q);
        while (nr != 0) {
            const __int128 qq = r / nr;
            const __int128 t2 = t - qq * nt; t = nt; nt = t2;
            const __int128 r2 = r - qq * nr; r = nr; nr = r2;
        }
        ENGINE_CHECK_MSG(r == 1, "BuildRingInvolution: not a unit");
        if (t < 0) t += static_cast<__int128>(q);
        return static_cast<uint64_t>(t);
    }
};

// Smith pivot valuations of a dense matrix over Z/ell^k (full valuation pivoting).
std::vector<uint32_t> PivotValuations(const RingZq& R, std::vector<std::vector<uint64_t>> A) {
    std::vector<uint32_t> out;
    const size_t nr = A.size();
    if (nr == 0) return out;
    const size_t nc = A[0].size();
    std::vector<char> rdone(nr, 0), cdone(nc, 0);
    for (;;) {
        bool found = false;
        uint32_t bv = 0;
        size_t pi = 0, pj = 0;
        for (size_t i = 0; i < nr; ++i) {
            if (rdone[i]) continue;
            for (size_t j = 0; j < nc; ++j) {
                if (cdone[j] || A[i][j] == 0) continue;
                const uint32_t v = R.val(A[i][j]);
                if (!found || v < bv) { found = true; bv = v; pi = i; pj = j; }
            }
        }
        if (!found) break;
        uint64_t ev = 1;
        for (uint32_t t = 0; t < bv; ++t) ev *= R.ell;
        const uint64_t iu = R.inv(A[pi][pj] / ev);
        for (size_t i = 0; i < nr; ++i) {
            if (i == pi || rdone[i] || A[i][pj] == 0) continue;
            const uint64_t f = R.mul(A[i][pj] / ev, iu);
            for (size_t j = 0; j < nc; ++j)
                if (!cdone[j] && A[pi][j]) A[i][j] = R.sub(A[i][j], R.mul(f, A[pi][j]));
        }
        rdone[pi] = 1;
        cdone[pj] = 1;
        out.push_back(bv);
    }
    return out;
}
} // namespace

template <class Elt>
RingInvolution BuildRingInvolution(const EquivariantCohomologyOnKT<Elt>& coh, const IotaSimplicialMap& im, uint32_t n_sym,
                                   uint32_t ell, const std::string& label) {
    const uint32_t k = coh.coefficient_ring_k();
    if (ell < 5)
        throw NotImplementedInThisBuild("BuildRingInvolution: iota^* over Z/ell^k needs ell >= 5 (the module route's stored "
                                        "basis; 2 a unit) -- got ell = " + std::to_string(ell));
    ENGINE_CHECK_MSG(k >= 2, "BuildRingInvolution: the cohomology must be over Z/ell^k, k >= 2");
    RingZq R;
    R.ell = ell;
    R.k = k;
    for (uint32_t i = 0; i < k; ++i) R.q *= ell;
    const uint64_t inv2 = R.inv(2);
    RingInvolution out;
    out.ell = ell;
    out.n_sym = n_sym;
    out.k = k;
    out.label = label;
    const uint32_t dV = n_sym + 1;
    const uint32_t ndeg = std::min<uint32_t>(coh.num_degrees(), im.num_degrees());
    for (uint32_t q = 0; q < ndeg; ++q) {
        RingInvolutionDegree D;
        const auto& md = coh.module_degree(q);
        D.exps = md.dec.exponents;
        const uint32_t t = static_cast<uint32_t>(D.exps.size());
        for (uint32_t e : D.exps) D.length += e;
        if (t == 0) {
            out.degrees.push_back(D);
            continue;
        }
        const auto& H = md.H;
        const uint32_t Nq = static_cast<uint32_t>(im.image_index[q].size());
        ENGINE_CHECK_MSG(H.rows == static_cast<size_t>(Nq) * dV && H.cols == t,
                         "BuildRingInvolution: the module basis is not in flat q-cochain coordinates");
        engine::DenseBuf<Elt> Y(H.rows, t);
        const uint64_t qe = R.q;
        for (uint32_t s = 0; s < Nq; ++s) {
            const uint32_t img = im.image_index[q][s];
            const int eps = im.image_sign[q][s];
            for (uint32_t i = 0; i < dV; ++i) {
                const bool rho_minus = ((n_sym - i) % 2u) == 1u;
                const bool neg = (eps > 0) != rho_minus;   // -eps * (-1)^{n-i}
                for (uint32_t j = 0; j < t; ++j) {
                    const uint64_t x = static_cast<uint64_t>(H.at(static_cast<size_t>(img) * dV + i, j)) % qe;
                    Y.at(static_cast<size_t>(s) * dV + i, j) = static_cast<Elt>(neg ? (qe - x) % qe : x);
                }
            }
        }
        const std::vector<std::vector<Elt>> co = coh.H_class_coordinates(q, engine::cview(Y));
        D.iota.assign(t, std::vector<uint64_t>(t, 0));
        for (uint32_t j = 0; j < t; ++j)
            for (uint32_t i = 0; i < t; ++i) D.iota[i][j] = static_cast<uint64_t>(co[j][i]);
        // iota^2 = id on H^q, read mod ell^{e_i}
        for (uint32_t j = 0; j < t; ++j)
            for (uint32_t i = 0; i < t; ++i) {
                uint64_t s2 = 0;
                for (uint32_t l = 0; l < t; ++l) s2 = R.add(s2, R.mul(D.iota[i][l], D.iota[l][j]));
                uint64_t mi = 1;
                for (uint32_t z = 0; z < D.exps[i]; ++z) mi *= ell;
                ENGINE_CHECK_MSG(s2 % mi == (i == j ? 1u % mi : 0u),
                                 "BuildRingInvolution: iota^* is not an involution on H^q");
            }
        // H^q_+- = (1 +- iota)/2 H^q, embedded in (Z/ell^k)^t by x_i -> ell^{k - e_i} x_i
        for (int sg = 0; sg < 2; ++sg) {
            std::vector<std::vector<uint64_t>> A(t, std::vector<uint64_t>(t, 0));
            for (uint32_t i = 0; i < t; ++i) {
                uint64_t sh = 1;
                for (uint32_t z = D.exps[i]; z < k; ++z) sh *= ell;
                for (uint32_t j = 0; j < t; ++j) {
                    const uint64_t id = i == j ? 1u : 0u;
                    const uint64_t v = sg == 0 ? R.add(id, D.iota[i][j]) : R.sub(id, D.iota[i][j]);
                    A[i][j] = R.mul(R.mul(v, inv2), sh);
                }
            }
            std::vector<uint32_t> f;
            for (uint32_t v : PivotValuations(R, A))
                if (v < k) f.push_back(k - v);
            std::sort(f.rbegin(), f.rend());
            uint32_t len = 0;
            for (uint32_t e : f) len += e;
            if (sg == 0) {
                D.plus_factors = f;
                D.length_plus = len;
            } else {
                D.minus_factors = f;
                D.length_minus = len;
            }
        }
        ENGINE_CHECK_MSG(D.length_plus + D.length_minus == D.length,
                         "BuildRingInvolution: H^q != H^+ (+) H^- (lengths do not add up)");
        for (uint32_t i = 0; i < t; ++i)
            if (D.exps[i] == k) D.trace_on_free += D.iota[i][i] == 1 ? 1 : (D.iota[i][i] == R.q - 1 ? -1 : 0);
        out.degrees.push_back(D);
    }
    return out;
}

template RingInvolution BuildRingInvolution<uint32_t>(const EquivariantCohomologyOnKT<uint32_t>&, const IotaSimplicialMap&, uint32_t,
                                                      uint32_t, const std::string&);
template RingInvolution BuildRingInvolution<uint64_t>(const EquivariantCohomologyOnKT<uint64_t>&, const IotaSimplicialMap&, uint32_t,
                                                      uint32_t, const std::string&);

std::string RingInvolution::to_json() const {
    std::ostringstream os;
    auto list = [&](const std::vector<uint32_t>& v) {
        os << "[";
        for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
        os << "]";
    };
    os << "{\"schema\":\"hecke_involution_ring/v1\",\"ell\":" << ell << ",\"n_sym\":" << n_sym << ",\"k\":" << k
       << ",\"sign\":\"iota^* = -iota^*_geo ([C] def:iota(iv))\",\"degrees\":[";
    for (size_t q = 0; q < degrees.size(); ++q) {
        const RingInvolutionDegree& D = degrees[q];
        os << (q ? "," : "") << "{\"q\":" << q << ",\"invariant_factors\":";
        list(D.exps);
        os << ",\"invariant_factors_plus\":";
        list(D.plus_factors);
        os << ",\"invariant_factors_minus\":";
        list(D.minus_factors);
        os << ",\"length\":" << D.length << ",\"length_plus\":" << D.length_plus << ",\"length_minus\":" << D.length_minus << "}";
    }
    os << "]}";
    return os.str();
}


} // namespace hecke
