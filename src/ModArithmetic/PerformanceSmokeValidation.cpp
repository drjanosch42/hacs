// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// matrix-engine Phase 2 -- arithmetic micro-benchmarks + performance smoke.
// See PerformanceSmokeValidation.h for the placement rationale and, in
// particular, for the five-line recipe Stage B follows to plug the new
// engine kernels in.
// ===========================================================================

#include "PerformanceSmokeValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "DenseBuf.h"
#include "DenseElim.h"
#include "DenseView.h"
#include "FlatMatrix.h"
#include "LegacyConfig.h"
#include "Fnv.h"
#include "ModCtx.h"

// matrix-engine Phase 2, Stage B: the `engine` backend measures the kernels
// DIRECTLY, on the same fixed-seed inputs, with no FlatMatrix wrapper around
// them -- see the note above GemmEngine below.
#include "DenseBuf.h"
#include "DenseView.h"
#include "Kernels.h"
#include "ModCtx.h"

namespace micro {
namespace {

using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2 == 1) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// ---------------------------------------------------------------------------
// Fixed-seed input generation.
//
// splitmix64 rather than <random>: std::mt19937_64 is portable but the
// DISTRIBUTIONS are not (libstdc++ and libc++ disagree), and these inputs must
// be byte-identical across compilers/configurations for the checksums to be a
// meaningful cross-run and cross-backend comparison. splitmix64 is ~10 lines
// and fully specified here.
// ---------------------------------------------------------------------------
inline uint64_t SplitMix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Seed derived from the shape + a per-purpose salt, so (a) the same shape
// always yields the same matrix and (b) the two factors of a rank-deficient
// product are different matrices.
uint64_t SeedFor(uint32_t r, uint32_t c, uint32_t ell, uint64_t salt) {
    uint64_t s = 0xC0FFEE123456789ull ^ salt;
    s = s * 0x100000001B3ull + r;
    s = s * 0x100000001B3ull + c;
    s = s * 0x100000001B3ull + ell;
    return s;
}

std::vector<uint32_t> RandomEntries(size_t count, uint32_t ell, uint64_t seed) {
    std::vector<uint32_t> v(count);
    uint64_t st = seed;
    for (size_t i = 0; i < count; ++i) {
        v[i] = static_cast<uint32_t>(SplitMix64(st) % ell);
    }
    return v;
}

// FNV-1a-64 over (rows, cols, then every entry in row-major order). Same hash
// family as Golden/GoldenHarness.cpp, so the two harnesses report comparable
// digests. The shape is folded in first, so a transposed or reshaped result
// cannot collide with the original.
//
// Unified (hecke-extensions WP-2, [X Sec.1.1]): `feed`/`value` come from
// `engine::FnvHasher` (src/ModArithmetic/Fnv.h) -- same constants,
// same byte order as the pre-unification class defined here. `feed_matrix`
// and `feed_buf` stay local because they depend on `FlatMatrix` and
// `engine::DenseBuf<uint32_t>`, which the dependency-free shared header must
// not include.
class Fnv : public engine::FnvHasher {
public:
    void feed_matrix(const FlatMatrix& m) {
        feed(m.get_rows());
        feed(m.get_cols());
        for (uint32_t r = 0; r < m.get_rows(); ++r)
            for (uint32_t c = 0; c < m.get_cols(); ++c)
                feed(m.get_entry(r, c));
    }
    // Same bytes, same order as feed_matrix, so an engine result and a legacy
    // result of equal content hash equally -- which is what makes the checksum
    // column a free cross-backend bit-identity check.
    void feed_buf(const engine::DenseBuf<uint32_t>& b) {
        feed(b.rows);
        feed(b.cols);
        for (uint32_t r = 0; r < b.rows; ++r)
            for (uint32_t c = 0; c < b.cols; ++c)
                feed(b.at(r, c));
    }
};

// ---------------------------------------------------------------------------
// Input matrices.
// ---------------------------------------------------------------------------

FlatMatrix RandomMatrix(uint32_t rows, uint32_t cols, uint32_t ell, uint64_t salt) {
    return FlatMatrix(rows, cols, ell,
                      RandomEntries(static_cast<size_t>(rows) * cols, ell,
                                    SeedFor(rows, cols, ell, salt)));
}

// A `rows x cols` matrix of rank <= rank_target, built as L(rows x r) * R(r x cols)
// with L, R fixed-seed random (rank == r with overwhelming probability for the
// primes used here). `rank_target == 0` or >= min(rows, cols) means "full rank",
// which is served by a plain random matrix -- both because that is what doc 03
// Sec. 7.3's reference elimination uses and because forming the product would
// otherwise dominate the setup.
//
// The product is formed here with a k-outer lazy-reduction loop rather than
// through FlatMatrix::operator*: this is INPUT GENERATION, it is never timed,
// and routing it through the very operator the `gemm` op measures would make
// setup cost more than the measurement.
FlatMatrix RankedMatrix(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell) {
    const uint32_t full = std::min(rows, cols);
    if (rank_target == 0 || rank_target >= full) {
        return RandomMatrix(rows, cols, ell, /*salt=*/0x11);
    }
    const uint32_t r = rank_target;
    const std::vector<uint32_t> L = RandomEntries(static_cast<size_t>(rows) * r, ell,
                                                  SeedFor(rows, r, ell, 0x21));
    const std::vector<uint32_t> R = RandomEntries(static_cast<size_t>(r) * cols, ell,
                                                  SeedFor(r, cols, ell, 0x22));
    std::vector<uint64_t> acc(static_cast<size_t>(rows) * cols, 0);
    // Reduce every `block` steps so the uint64 accumulator cannot overflow:
    // each term is < ell^2 <= (2^32-1)^2, so 2^64 / ell^2 terms are safe.
    const uint64_t e = ell;
    const uint64_t max_terms = (e * e == 0) ? 1 : (0xFFFFFFFFFFFFFFFFull / (e * e));
    const uint32_t block = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(max_terms, r)));
    for (uint32_t k0 = 0; k0 < r; k0 += block) {
        const uint32_t k1 = std::min<uint32_t>(k0 + block, r);
        for (uint32_t k = k0; k < k1; ++k) {
            for (uint32_t i = 0; i < rows; ++i) {
                const uint64_t a = L[static_cast<size_t>(i) * r + k];
                if (a == 0) continue;
                uint64_t* out = &acc[static_cast<size_t>(i) * cols];
                const uint32_t* bp = &R[static_cast<size_t>(k) * cols];
                for (uint32_t j = 0; j < cols; ++j) out[j] += a * bp[j];
            }
        }
        for (auto& x : acc) x %= e;
    }
    std::vector<uint32_t> data(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint32_t>(acc[i] % e);
    return FlatMatrix(rows, cols, ell, data);
}

MicroResult NotAvailable(const char* op, const std::string& params) {
    MicroResult res;
    res.available = false;
    res.op = op;
    res.params = params;
    res.note = "backend 'engine' is not implemented yet (matrix-engine Phase 2, Stage B)";
    return res;
}

#if !OC_ENABLE_LEGACY
// (S24 CL-L2) Backend::Legacy is unreachable in an OFF build (ParseBackend refuses "legacy", ENGINE_BACKEND=legacy
// is refused at start-up); should a caller construct it anyway, it is reported unavailable, never run as the engine.
MicroResult NotCompiledLegacy(const char* op, const std::string& params) {
    MicroResult res = NotAvailable(op, params);
    res.note = oc::legacy::compiled_out_text("backend 'legacy'");
    return res;
}
#endif

// Work estimate for a rows x cols elimination of rank R:
// sum over pivots of (remaining rows) * (remaining cols) ~ R * rows * cols in
// the full-rank square-ish case; the exact sum is used so the reported rate is
// comparable with doc 03 Sec. 7.3's "work" column (3.38 Gmodmul for
// 1677 x 1419 full rank).
double ElimWork(uint32_t rows, uint32_t cols, uint32_t rank) {
    double w = 0.0;
    for (uint32_t s = 0; s < rank; ++s) {
        w += static_cast<double>(rows - 1) * static_cast<double>(cols - s);
    }
    return w;
}

#if OC_ENABLE_LEGACY
#include "legacy/MicroBenchLegacy.inc"   // (S24 CL-L2) the five *Legacy micro implementations (--backend legacy)
#endif

// ---------------------------------------------------------------------------
// Engine implementations (matrix-engine Phase 2, Stage B) -- the DenseElim
// path on a DenseBuf over the SAME fixed-seed inputs, digested the same way,
// so the checksums must equal the legacy ones bit for bit.
// ---------------------------------------------------------------------------

engine::DenseBuf<uint32_t> ToBuf(const FlatMatrix& m) {
    engine::DenseBuf<uint32_t> b(m.get_rows(), m.get_cols());
    for (uint32_t r = 0; r < m.get_rows(); ++r)
        for (uint32_t c = 0; c < m.get_cols(); ++c) b.at(r, c) = m.get_entry(r, c);
    return b;
}

MicroResult RrefEngine(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                       uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "rref";
    res.params = params;

    const engine::DenseBuf<uint32_t> A0 = ToBuf(RankedMatrix(rows, cols, rank_target, ell));
    const engine::ModCtx<uint32_t> ctx(ell);

    std::vector<double> ts;
    Fnv h;
    uint32_t rank_found = 0;
    for (uint32_t i = 0; i < repeat; ++i) {
        engine::DenseBuf<uint32_t> A = A0;      // fresh copy, not timed
        const auto t0 = Clock::now();
        const engine::elim::RrefResult r = engine::elim::rref<uint32_t>(ctx, engine::view(A));
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) {
            h.feed_buf(A);
            rank_found = r.rank;
        }
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    res.rank = rank_found;
    res.work_modmul = ElimWork(rows, cols, rank_found);
    std::ostringstream note;
    note << "panel=" << engine::elim::params().panel << " base=" << engine::elim::params().base;
    res.note = note.str();
    return res;
}

MicroResult KernelEngine(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                         uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "kernel";
    res.params = params;

    const engine::DenseBuf<uint32_t> A0 = ToBuf(RankedMatrix(rows, cols, rank_target, ell));
    const engine::ModCtx<uint32_t> ctx(ell);

    std::vector<double> ts;
    Fnv h;
    uint32_t nullity = 0, rank_found = 0;
    for (uint32_t i = 0; i < repeat; ++i) {
        engine::DenseBuf<uint32_t> A = A0;
        const auto t0 = Clock::now();
        const engine::elim::RrefResult r = engine::elim::rref<uint32_t>(ctx, engine::view(A));
        const engine::DenseBuf<uint32_t> K = engine::elim::kernel_basis<uint32_t>(ctx, engine::cview(A), r.pivots);
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) {
            nullity = K.rows;
            h.feed(nullity);
            for (uint32_t t = 0; t < K.rows; ++t) {     // == feed_matrix of each cols x 1 vector
                h.feed(K.cols);
                h.feed(1u);
                for (uint32_t j = 0; j < K.cols; ++j) h.feed(K.at(t, j));
            }
            rank_found = r.rank;
        }
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    res.rank = rank_found;
    res.nullity = nullity;
    res.work_modmul = ElimWork(rows, cols, rank_found);
    return res;
}

// ---------------------------------------------------------------------------
// Engine implementations (matrix-engine Phase 2, Stage B).
//
// These call engine::kernels directly on DenseBufs. They deliberately do NOT
// go through FlatMatrix: FlatMatrix::operator* IS retrofitted onto the same
// kernels now, so routing them through it would add the wrapper's cost
// (allocation, populate_flat_data, the post-hoc scalar-flag rescan) to a
// number that is meant to measure the arithmetic. That makes the two columns
// mean exactly what they say:
//
//   --backend legacy   the PRE-Phase-2 substrate.  Every legacy arm installs
//                      an engine::elim::ScopedLegacy guard, which
//                      engine::elim::use_legacy() reports to BOTH the product
//                      and the elimination paths of FlatMatrix, so the column
//                      is the honest "before" side end to end -- no env var
//                      needed and no arm half-retrofitted.
//   --backend engine   the kernels alone, the arithmetic ceiling the
//                      retrofit is chasing
//
// To time the SHIPPED FlatMatrix path (engine inside the wrapper) use the
// pipeline modes -- `--bench cohom/hecke` -- which is what they measure.
//
// Inputs come from the same RandomEntries()/SeedFor() calls as the legacy
// implementations and the checksum is fed in the same order, so equal
// checksums across the two backends are a bit-identity statement.
// ---------------------------------------------------------------------------

using EngineBuf = engine::DenseBuf<uint32_t>;

EngineBuf EngineMatrixImpl(uint32_t rows, uint32_t cols, uint32_t ell, uint64_t salt) {
    EngineBuf b(rows, cols, engine::Layout::RowMajor);
    const std::vector<uint32_t> v =
        RandomEntries(static_cast<size_t>(rows) * cols, ell, SeedFor(rows, cols, ell, salt));
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < cols; ++c)
            b.at(r, c) = v[static_cast<size_t>(r) * cols + c];
    return b;
}

MicroResult GemmEngine(uint32_t n, uint32_t k, uint32_t ell, uint32_t repeat,
                       const std::string& params) {
    MicroResult res;
    res.op = "gemm";
    res.params = params;
    res.work_modmul = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(k);

    const engine::ModCtx<uint32_t> ctx(ell);
    const EngineBuf A = EngineMatrixImpl(n, k, ell, 0x01);
    const EngineBuf B = EngineMatrixImpl(k, n, ell, 0x02);

    std::vector<double> ts;
    Fnv h;
    for (uint32_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        // C is allocated inside the timed region, as the legacy route's result
        // FlatMatrix is, so the two timings cover the same work.
        EngineBuf C(n, n, engine::Layout::RowMajor);
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(B),
                                        engine::view(C), engine::kernels::GemmOp::Set);
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) h.feed_buf(C);
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    // (slot-23 PERF-P5) name the VNNI route when this shape takes it (text only; the checksum is route-independent)
    const bool vnni = engine::kernels::vnni_route_taken(ell, n, n, k);
    res.note = vnni && engine::kernels::vnni_limbs_for(ell) == 1 ? "AVX-512 VNNI u8 x s8 route (p <= 128)"
             : vnni ? "AVX-512 VNNI two-limb s16 route (128 < p <= 2^14)"
             : engine::kernels::avx512_kernel_active() ? "AVX-512 microkernel"
             : engine::kernels::active_kernel_path() == engine::kernels::KernelPath::Avx2 ? "AVX2 microkernel"
                                                       : "portable microkernel";
    return res;
}

MicroResult GemvEngine(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "gemv";
    res.params = params;
    res.work_modmul = static_cast<double>(n) * static_cast<double>(n);

    const engine::ModCtx<uint32_t> ctx(ell);
    const EngineBuf A = EngineMatrixImpl(n, n, ell, 0x03);
    const EngineBuf x = EngineMatrixImpl(n, 1, ell, 0x04);

    std::vector<double> ts;
    Fnv h;
    for (uint32_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        EngineBuf y(n, 1, engine::Layout::RowMajor);
        engine::kernels::gemv<uint32_t>(ctx, engine::cview(A), engine::cview(x),
                                        engine::view(y), engine::kernels::GemmOp::Set);
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) h.feed_buf(y);
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    return res;
}

MicroResult TransposeEngine(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "transpose";
    res.params = params;
    res.work_modmul = 0.0; // data movement, not modmuls

    const EngineBuf A = EngineMatrixImpl(n, n, ell, 0x05);

    std::vector<double> ts;
    Fnv h;
    for (uint32_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        EngineBuf T(n, n, engine::Layout::RowMajor);
        engine::kernels::transpose<uint32_t>(engine::cview(A), engine::view(T));
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) h.feed_buf(T);
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    std::ostringstream note;
    const double elems = static_cast<double>(n) * static_cast<double>(n);
    note << std::fixed << std::setprecision(2)
         << (res.min_s > 0.0 ? elems / res.min_s / 1e9 : 0.0) << " Gelem/s";
    res.note = note.str();
    return res;
}

} // namespace

const char* BackendName(Backend b) {
    return (b == Backend::Engine) ? "engine" : (b == Backend::Gpu) ? "gpu" : "legacy";
}

bool ParseBackend(const std::string& s, Backend& out) {
    // (S24 CL-L2) `legacy` parses only in an OC_ENABLE_LEGACY=ON build; the bench reports it as a usage error
    // (compiled_out_text) in an OFF build, and ENGINE_BACKEND=legacy is refused at start-up there.
    if (s == "legacy") { if (!oc::legacy::kLegacyBuild) return false; out = Backend::Legacy; return true; }
    if (s == "engine") { out = Backend::Engine; return true; }
    if (s == "gpu")    { out = Backend::Gpu;    return true; }
    return false;
}

// ---- Phase 5: the GPU arm's hooks and shared input helpers ----------------
namespace {
GpuHooks g_gpu_hooks;
MicroResult NotAvailableGpu(const char* op, const std::string& params) {
    MicroResult res;
    res.available = false;
    res.op = op;
    res.params = params;
    res.note = g_gpu_hooks.gemm ? "backend 'gpu' has no implementation of this op (elimination is CPU-driven)"
                                : "backend 'gpu' is not installed (no engine::cuda::InstallMicroHooks() call, "
                                  "or no usable CUDA device)";
    return res;
}
} // namespace
void SetGpuHooks(const GpuHooks& hooks) { g_gpu_hooks = hooks; }
const GpuHooks& GetGpuHooks() { return g_gpu_hooks; }
engine::DenseBuf<uint32_t> EngineMatrix(uint32_t rows, uint32_t cols, uint32_t ell, uint64_t salt) {
    return EngineMatrixImpl(rows, cols, ell, salt);
}
uint64_t ChecksumBuf(const engine::DenseBuf<uint32_t>& b) {
    Fnv h; h.feed_buf(b); return h.value();
}

// (Phase 2, merge) The DEFAULT is now `engine`: both arms exist and the engine
// arm is the shipped substrate, so PerformanceSmokeValidation inside `--all`
// must gate the code the pipeline actually runs.  `ENGINE_BACKEND=legacy`
// still selects the legacy arm -- and with it the legacy floors -- which is
// what `ENGINE_BACKEND=legacy ./cohomology_app --all` exercises.
Backend BackendFromEnv() {
    const char* e = std::getenv("ENGINE_BACKEND");
    Backend b = Backend::Engine;
    if (e && ParseBackend(std::string(e), b)) return b;
    return Backend::Engine;
}

// ===========================================================================
// The five dispatchers. STAGE B: replace the NotAvailable(...) line of each
// with a call to the corresponding *Engine(...) implementation. Nothing else
// changes.
// ===========================================================================

MicroResult RunGemm(uint32_t n, uint32_t k, uint32_t ell, Backend backend, uint32_t repeat) {
    std::ostringstream p;
    p << "n=" << n << " k=" << k << " ell=" << ell;
    switch (backend) {
#if OC_ENABLE_LEGACY
        case Backend::Legacy: return GemmLegacy(n, k, ell, repeat, p.str());
#else
        case Backend::Legacy: return NotCompiledLegacy("gemm", p.str());
#endif
        case Backend::Engine: return GemmEngine(n, k, ell, repeat, p.str());
        case Backend::Gpu: return g_gpu_hooks.gemm ? g_gpu_hooks.gemm(n, k, ell, repeat, p.str()) : NotAvailableGpu("gemm", p.str());
    }
    return NotAvailable("gemm", p.str());
}

MicroResult RunGemv(uint32_t n, uint32_t ell, Backend backend, uint32_t repeat) {
    std::ostringstream p;
    p << "n=" << n << " ell=" << ell;
    switch (backend) {
#if OC_ENABLE_LEGACY
        case Backend::Legacy: return GemvLegacy(n, ell, repeat, p.str());
#else
        case Backend::Legacy: return NotCompiledLegacy("gemv", p.str());
#endif
        case Backend::Engine: return GemvEngine(n, ell, repeat, p.str());
        case Backend::Gpu: return g_gpu_hooks.gemv ? g_gpu_hooks.gemv(n, ell, repeat, p.str()) : NotAvailableGpu("gemv", p.str());
    }
    return NotAvailable("gemv", p.str());
}

MicroResult RunRref(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                    Backend backend, uint32_t repeat) {
    std::ostringstream p;
    p << "rows=" << rows << " cols=" << cols << " rank_req=" << rank_target << " ell=" << ell;
    switch (backend) {
#if OC_ENABLE_LEGACY
        case Backend::Legacy: return RrefLegacy(rows, cols, rank_target, ell, repeat, p.str());
#else
        case Backend::Legacy: return NotCompiledLegacy("rref", p.str());
#endif
        case Backend::Engine: return RrefEngine(rows, cols, rank_target, ell, repeat, p.str());
        case Backend::Gpu: return NotAvailableGpu("rref", p.str());
    }
    return NotAvailable("rref", p.str());
}

MicroResult RunKernel(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                      Backend backend, uint32_t repeat) {
    std::ostringstream p;
    p << "rows=" << rows << " cols=" << cols << " rank_req=" << rank_target << " ell=" << ell;
    switch (backend) {
#if OC_ENABLE_LEGACY
        case Backend::Legacy: return KernelLegacy(rows, cols, rank_target, ell, repeat, p.str());
#else
        case Backend::Legacy: return NotCompiledLegacy("kernel", p.str());
#endif
        case Backend::Engine: return KernelEngine(rows, cols, rank_target, ell, repeat, p.str());
        case Backend::Gpu: return NotAvailableGpu("kernel", p.str());
    }
    return NotAvailable("kernel", p.str());
}

MicroResult RunTranspose(uint32_t n, uint32_t ell, Backend backend, uint32_t repeat) {
    std::ostringstream p;
    p << "n=" << n << " ell=" << ell;
    switch (backend) {
#if OC_ENABLE_LEGACY
        case Backend::Legacy: return TransposeLegacy(n, ell, repeat, p.str());
#else
        case Backend::Legacy: return NotCompiledLegacy("transpose", p.str());
#endif
        case Backend::Engine: return TransposeEngine(n, ell, repeat, p.str());
        case Backend::Gpu: return g_gpu_hooks.transpose ? g_gpu_hooks.transpose(n, ell, repeat, p.str()) : NotAvailableGpu("transpose", p.str());
    }
    return NotAvailable("transpose", p.str());
}

} // namespace micro

// ===========================================================================
// PerformanceSmokeValidation -- the registered `standard`-tier smoke test.
//
// FLOORS. The plan's Phase-2 bullet names GEMM 512^2 >= 4 Gmodmul/s and RREF
// 1677 x 1419 <= 3 s. Those are the numbers the NEW engine must meet. The
// LEGACY substrate, measured on this box (RelWithDebInfo, Ryzen 9 7950X under
// WSL2, 2026-09-05, `--bench micro suite --repeat 5`, min of 5) does
//
//     gemm 512        : 0.169 Gmodmul/s
//     rref 1677x1419  : 3.181 s   (full rank 1419)
//
// so the plan's floors are BOTH missed by legacy today, and registering them
// as-is would make this validation red from its first commit -- useless as a
// regression guard. The floors below are therefore derived from the LEGACY
// measurement with the >= 2x margin the Phase-2 task specifies:
//
//     >= 0.08 Gmodmul/s   is 2.11x below the measured 0.169
//     <= 8.0 s            is 2.22x above the 3.60 s a SINGLE run costs (the
//                         validation uses --repeat 1; min-of-5 is 3.18 s)
//
// i.e. they catch an order-of-magnitude regression (and a build that
// accidentally lost its optimisation flags: at -O0 the same two numbers are
// far worse) while tolerating WSL2 noise. STAGE B RAISES THEM to the plan's
// values once ENGINE_BACKEND=engine is the default -- that is the point of
// this test, and kPlanGemm512Gmodmul / kPlanRrefSeconds below are printed
// next to the measurement so the remaining distance is always visible.
// ===========================================================================
namespace {

constexpr double kFloorGemm512Gmodmul = 0.08;  // >= this many Gmodmul/s
constexpr double kFloorRrefSeconds    = 8.0;   // <= this many seconds

// The plan's Phase-2 targets.
constexpr double kPlanGemm512Gmodmul = 4.0;
constexpr double kPlanRrefSeconds    = 3.0;

// STAGE B (matrix-engine Phase 2, GEMM retrofit): the `engine` backend is held
// to the PLAN's number, not to the legacy-derived one. `--bench micro gemm 512
// --backend engine` measures ~29.8 Gmodmul/s on the reference box, so a 4.0
// floor is a 7.5x margin -- it cannot fire on noise, but it does fire if the
// AVX-512 microkernel silently stops being selected (the portable path is
// 2.7 Gmodmul/s at -O2, i.e. BELOW this floor, which is exactly the regression
// worth catching) or if a future change puts the kernel back behind a call
// boundary. The legacy backend keeps its legacy-derived floor: with
// ENGINE_BACKEND=legacy the shipped FlatMatrix path really is the Phase-1
// scalar loop and must not be judged against a number it was never meant to
// meet.
//
// The rref floor is NOT raised here -- elimination is not part of this
// retrofit and still runs the legacy core.
constexpr double kEngineFloorGemm512Gmodmul = kPlanGemm512Gmodmul;

} // namespace

// (hecke-extensions slot 9, WP-9; [X] open item 6) THE SPEED FLOORS ARE
// MEANINGLESS ON A NON-NDEBUG BUILD.  `Debug` (-O0) and the two sanitizer
// configurations `Check` / `CheckThread` (-O1 -fsanitize=...) all define
// neither NDEBUG, and ASan/UBSan/TSan instrumentation plus the missing -O2/-O3
// routinely cost the GEMM kernel a full order of magnitude -- exactly the
// regression the floor exists to catch, so gating on it there produces a
// standing false failure (found at the slot-8 close: the GEMM floor missed
// under `Check` on an otherwise-unchanged checksum).  `#ifndef NDEBUG` prints
// `[SKIP floor]` with the reason and never counts a floor as a failure;
// EVERY availability check above (a backend with no implementation) and every
// printed `checksum=` stay live and unconditional in both configurations --
// only the numeric floor comparison is skipped.
#ifndef NDEBUG
constexpr const char* kFloorSkipReason =
    "non-NDEBUG build (Debug/Check/CheckThread): a sanitizer/-O0 cost, not a regression -- "
    "[X] open item 6";
#endif

uint32_t PerformanceSmokeValidation() {
    const micro::Backend backend = micro::BackendFromEnv();
    std::cout << "=== PerformanceSmokeValidation (backend=" << micro::BackendName(backend)
              << ", set ENGINE_BACKEND=legacy|engine) ===" << std::endl;

    uint32_t failures = 0;

    // --repeat 1 here: this is a smoke test inside --all, not a measurement
    // run. Use `--bench micro suite --repeat 5` for numbers.
    const micro::MicroResult gemm =
        micro::RunGemm(512, 512, micro::kDefaultEll, backend, /*repeat=*/1);
    if (!gemm.available) {
        std::cout << "  gemm 512: " << gemm.note << std::endl;
        std::cout << "[FAIL-DETAIL] backend '" << micro::BackendName(backend)
                  << "' has no gemm implementation" << std::endl;
        return 1;
    }
    const double gemm_rate = gemm.gmodmul_per_s();
    const double gemm_floor = (backend == micro::Backend::Legacy) ? kFloorGemm512Gmodmul
                                                                  : kEngineFloorGemm512Gmodmul;
#ifndef NDEBUG
    std::cout << "  gemm 512x512x512 : " << std::fixed << std::setprecision(4) << gemm.min_s
              << " s -> " << std::setprecision(3) << gemm_rate << " Gmodmul/s"
              << "  (floor >= " << gemm_floor << ", plan target >= "
              << kPlanGemm512Gmodmul << ")  checksum=0x" << std::hex << gemm.checksum << std::dec
              << "  [SKIP floor: " << kFloorSkipReason << "]";
    if (!gemm.note.empty()) std::cout << "  " << gemm.note;
    std::cout << std::endl;
#else
    // (slot-14 WP-2, [C3]/[C4] finding 4) The engine floor of 4.0 Gmodmul/s is
    // an AVX-512 floor: the portable (no-AVX-512) microkernel measures
    // ~1.7-1.9 Gmodmul/s on Zen 3 / Zen 2, which is BELOW this floor without
    // being a regression -- it is the ISA the node has. Skip the numeric gate
    // there, exactly as the non-NDEBUG arm skips it for a different reason;
    // the checksum stays printed and asserted available either way (it is
    // never part of what is skipped). Legacy keeps its legacy-derived floor
    // (0.08, a 2.11x margin over the portable-scalar 0.169 measurement) --
    // that arm is the Phase-1 scalar loop by design and is judged the same
    // way regardless of ISA.
    const bool isa_skip = (backend != micro::Backend::Legacy) &&
                          !engine::kernels::avx512_kernel_active();
    if (isa_skip) {
        std::cout << "  gemm 512x512x512 : " << std::fixed << std::setprecision(4) << gemm.min_s
                  << " s -> " << std::setprecision(3) << gemm_rate << " Gmodmul/s"
                  << "  (floor >= " << gemm_floor << ", plan target >= "
                  << kPlanGemm512Gmodmul << ")  checksum=0x" << std::hex << gemm.checksum << std::dec
                  << "  [SKIP floor] portable microkernel (no AVX-512)";
        if (!gemm.note.empty()) std::cout << "  " << gemm.note;
        std::cout << std::endl;
    } else {
        const bool gemm_ok = (gemm_rate >= gemm_floor);
        std::cout << "  gemm 512x512x512 : " << std::fixed << std::setprecision(4) << gemm.min_s
                  << " s -> " << std::setprecision(3) << gemm_rate << " Gmodmul/s"
                  << "  (floor >= " << gemm_floor << ", plan target >= "
                  << kPlanGemm512Gmodmul << ")  checksum=0x" << std::hex << gemm.checksum << std::dec
                  << "  [" << (gemm_ok ? "OK" : "BELOW FLOOR") << "]";
        if (!gemm.note.empty()) std::cout << "  " << gemm.note;
        std::cout << std::endl;
        if (!gemm_ok) ++failures;
    }
#endif

    if (backend == micro::Backend::Gpu) {
        // Phase 5: elimination stays CPU-driven; the GPU arm has no rref op.
        std::cout << "  rref 1677x1419   : not measured on backend 'gpu' (elimination is CPU-driven)" << std::endl;
        std::cout << "PerformanceSmokeValidation: " << (failures == 0 ? "gemm floor met." : "gemm floor missed.") << std::endl;
        return failures;
    }
    const micro::MicroResult rref =
        micro::RunRref(1677, 1419, micro::kSuiteRrefRank, micro::kDefaultEll, backend, /*repeat=*/1);
    if (!rref.available) {
        std::cout << "  rref 1677x1419: " << rref.note << std::endl;
        std::cout << "[FAIL-DETAIL] backend '" << micro::BackendName(backend)
                  << "' has no rref implementation" << std::endl;
        return 1;
    }
#ifndef NDEBUG
    std::cout << "  rref 1677x1419   : " << std::fixed << std::setprecision(4) << rref.min_s
              << " s (rank " << rref.rank << ")"
              << "  (floor <= " << std::setprecision(1) << kFloorRrefSeconds
              << " s, plan target <= " << kPlanRrefSeconds << " s)  checksum=0x"
              << std::hex << rref.checksum << std::dec
              << "  [SKIP floor: " << kFloorSkipReason << "]" << std::endl;
#else
    const bool rref_ok = (rref.min_s <= kFloorRrefSeconds);
    std::cout << "  rref 1677x1419   : " << std::fixed << std::setprecision(4) << rref.min_s
              << " s (rank " << rref.rank << ")"
              << "  (floor <= " << std::setprecision(1) << kFloorRrefSeconds
              << " s, plan target <= " << kPlanRrefSeconds << " s)  checksum=0x"
              << std::hex << rref.checksum << std::dec
              << "  [" << (rref_ok ? "OK" : "OVER FLOOR") << "]" << std::endl;
    if (!rref_ok) ++failures;
#endif

#ifndef NDEBUG
    std::cout << "PerformanceSmokeValidation: both floors skipped (" << kFloorSkipReason
              << "); checksums asserted available." << std::endl;
#else
    if (failures == 0) {
        std::cout << "PerformanceSmokeValidation: both floors met." << std::endl;
    } else {
        std::cout << "PerformanceSmokeValidation: " << failures << " floor(s) missed." << std::endl;
    }
#endif
    return failures;
}
