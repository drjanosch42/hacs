// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S9 -- HeckeOperatorFamily. See HeckeOperatorFamily.h for the full
// contract statement ([C, thm:relations], [C, prop:diamond], [C,
// cor:ulrecursion]) and the binding conventions (multiplication order,
// caching, the documented theta_flat() return-type note).

#include "HeckeOperatorFamily.h"

#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../Engine/EngineRuntime.h"
#include "HeckeSettings.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace hecke {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

HeckeOperatorFamily::HeckeOperatorFamily(const EquivariantCohomologyOnK* cohomL,
                                         const SimplicialComplex& K,
                                         const SL2ZTessellationTables& model,
                                         uint32_t ell, uint32_t n_sym)
    : cohomL_(cohomL), K_(&K), model_(&model), ell_(ell), n_sym_(n_sym) {
    // (slot-8 WP-3) [E] sec:ring (b): T_{p^r} / T(p,p) / U_ell family routes are F_ell-only.
    RequireFieldCoefficientRing("HeckeOperatorFamily", cohomL ? cohomL->coefficient_ring_k() : 1u);
    if (!cohomL_) {
        throw std::invalid_argument("HeckeOperatorFamily: cohomL is null.");
    }
    if (!cohomL_->is_cohomology_computed()) {
        throw std::runtime_error(
            "HeckeOperatorFamily: cohomL must be computed (call "
            "compute_cohomology() first).");
    }
    if (cohomL_->get_prime() != ell_) {
        throw std::runtime_error(
            "HeckeOperatorFamily: ell does not match cohomL's prime.");
    }
    if (cohomL_->get_representation_dim() != n_sym_ + 1) {
        throw std::runtime_error(
            "HeckeOperatorFamily: n_sym does not match cohomL's "
            "representation_dim - 1.");
    }
    num_degrees_ = static_cast<uint32_t>(cohomL_->get_H_eq_dimensions().size());
    if (num_degrees_ == 0) {
        throw std::runtime_error("HeckeOperatorFamily: cohomL has no computed degrees.");
    }
}

// ---------------------------------------------------------------------------
// Pipeline cache
// ---------------------------------------------------------------------------

// One process-wide lock for the tau-table (and coset-table) CONSTRUCTIONS of
// every family: HeckeTauTable's build is first-match-wins over a fixed order
// and increments the non-atomic global PredicateStats whose exact counts
// HeckeFpFilterValidation asserts (audit 02 Sec. 6.5, hazards 1-3), so two
// constructions never overlap.  Only compute() runs concurrently.
// (slot-9 WP-7, lever L2) build_pipelines() below now holds it only while it
// primes K's simplex index: hazard 3 is gone (PredicateStats is thread_local
// since slot-8 WP-6) and hazards 1-2 are orders INSIDE one construction; see
// build_pipelines().  Every other holder is unchanged.
//
// (matrix-engine Phase 6) Declared in the header so the campaign driver's
// per-job threads hold the SAME lock; see HeckeOperatorFamily.h.
std::mutex& tau_construction_mutex() {
    static std::mutex m;
    return m;
}

void HeckeOperatorFamily::build_pipelines(std::vector<std::pair<uint64_t, uint32_t>> keys) const {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::pair<uint64_t, uint32_t>> missing;
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        for (const auto& k : keys) {
            if (k.second == 0) {
                throw std::invalid_argument(
                    "HeckeOperatorFamily::get_or_build_pipeline: r == 0 has no coset "
                    "table (HeckeCosetTables requires r >= 1); callers handle r == 0 "
                    "directly as the identity.");
            }
            if (pipeline_cache_.find(k) == pipeline_cache_.end()) missing.push_back(k);
        }
    }
    if (missing.empty()) return;

    // Phase A (serial, ascending key order): tables + operator objects.
    //
    // (slot-9 WP-7, lever L2) The process-wide lock covers ONLY the priming of
    // K's lazily built simplex index -- the one write this phase can make to
    // an object another family may share (two families on one K).  The
    // constructions themselves run outside it: still serially, in ascending
    // key order, start to finish on THIS thread, so two DIFFERENT families
    // (two ext cells of a parallel campaign pass) no longer queue on each
    // other's tau builds (~2.5-3.7 s each at U_10007; the WP-7 report's
    // concurrency table).  Why no value can move:
    //   * a construction is a pure function of (K, model, cosets, ell, n_sym);
    //   * audit 02 Sec. 6.5 hazards 1-2 (first-match-wins tau, the triangle
    //     fill's column / pivot order) are orders INSIDE one construction,
    //     which one thread still runs in its fixed order;
    //   * hazard 3, the non-atomic global PredicateStats, is gone: the
    //     counters are thread_local since slot-8 WP-6 (Predicates.h, "PER
    //     THREAD"), and so is every other mutable static the build reaches
    //     (ExactInt's tier, the mod_nCr factorial tables, FlatMatrix's
    //     ModCtx cache, the kernels' scratch) -- the rest is const or a
    //     thread-safe magic static;
    //   * K's index is complete after the priming below (every degree, and it
    //     does not depend on ell), so the constructions and the concurrent
    //     compute() calls only read it; the model, the cohomology and every
    //     other input are read-only.
    // Every other holder of the lock (HeckeBoundary, the Sturm sweep, the
    // campaign job loop, the console and binding builds) is unchanged.
    std::vector<std::unique_ptr<PipelineEntry>> entries;
    entries.reserve(missing.size());
    {
        {
            std::lock_guard<std::mutex> lk(tau_construction_mutex());
            // Prime the base complex's lazily built simplex index for every
            // degree on this thread, so the constructions and the concurrent
            // compute() calls below only read it.
            for (uint32_t q = 0; q <= K_->get_dimension(); ++q) (void)K_->coboundary_operator_elt<uint32_t>(q, ell_);
        }
        for (const auto& k : missing) {
            auto e = std::make_unique<PipelineEntry>();
            e->cosets = std::make_unique<HeckeCosetTables>(k.first, k.second, ell_, n_sym_);
            e->tau = std::make_unique<HeckeTauTable>(*K_, *model_, *e->cosets, ell_, n_sym_);
            e->op = std::make_unique<HeckeOperatorOnK>(cohomL_, e->cosets.get(), e->tau.get());
            entries.push_back(std::move(e));
        }
    }
    // Phase B (concurrent, axis (d)): the operators -- each enforces the
    // runtime assertions + O12(a) bit-identity (S7) on its own data.
    std::vector<std::function<void()>> tasks;
    for (auto& e : entries) tasks.emplace_back([&e]() { e->op->compute(); });
    engine::tape::run_concurrently(tasks);
    // Phase C: publish in key order.
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        for (size_t i = 0; i < missing.size(); ++i)
            if (pipeline_cache_.find(missing[i]) == pipeline_cache_.end())
                pipeline_cache_.emplace(missing[i], std::move(*entries[i]));
    }
}

HeckeOperatorFamily::PipelineEntry& HeckeOperatorFamily::get_or_build_pipeline(
    uint64_t p, uint32_t r) const {
    if (r == 0) {
        throw std::invalid_argument(
            "HeckeOperatorFamily::get_or_build_pipeline: r == 0 has no coset "
            "table (HeckeCosetTables requires r >= 1); callers handle r == 0 "
            "directly as the identity.");
    }
    const auto key = std::make_pair(p, r);
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        auto it = pipeline_cache_.find(key);
        if (it != pipeline_cache_.end()) return it->second;
    }
    build_pipelines({key});
    std::lock_guard<std::mutex> lk(cache_mutex_);
    return pipeline_cache_.find(key)->second;
}

void HeckeOperatorFamily::prepare(uint64_t p, uint32_t r) {
    std::vector<std::pair<uint64_t, uint32_t>> keys{{p, 1u}};
    for (uint32_t j = 0; 2 * j < r; ++j) keys.emplace_back(p, r - 2 * j);
    build_pipelines(std::move(keys));
}

// (slot-24 G6) See the header.  Each cached pipeline keeps its operator object
// (so every reference T_p() handed out -- to the operator's A_ -- stays valid)
// and loses level (a) and its two tables.  The operator never dereferences its
// table pointers again (HeckeOperatorOnK::release_level_a's contract), and a
// later T_p / prepare of a cached key returns that A_ without touching them.
uint32_t HeckeOperatorFamily::release_level_a() {
    std::lock_guard<std::mutex> lk(cache_mutex_);
    uint32_t released = 0;
    for (auto& kv : pipeline_cache_) {
        PipelineEntry& e = kv.second;
        if (!e.op || e.op->level_a_released()) continue;
        e.op->release_level_a();
        e.tau.reset();      // the tau table reads the coset table: destroyed first
        e.cosets.reset();
        ++released;
    }
    return released;
}

// scalar p^{n_sym+1} mod ell ([C, thm:relations]'s recursion coefficient).
//
// (matrix-engine Phase 3 Stage B) The three modular powers of this file now go
// through engine::ModCtx<uint32_t>::pow. That is not a re-implementation: pow()
// IS ModArithmetic::mod_pow_w (ModCtx.h), i.e. the very function this replaces,
// so the values are bit-identical by construction and the exponent-reduction
// convention (exp %= p-1, base == 0 short-circuit) is unchanged.
uint32_t HeckeOperatorFamily::recursion_scalar(uint64_t p) const {
    const uint32_t p_mod = static_cast<uint32_t>(p % ell_);
    return engine::ModCtx<uint32_t>(ell_).pow(p_mod, n_sym_ + 1);
}

// ---------------------------------------------------------------------------
// T_p (== U_ell when p == ell; invariant I7, no branch)
// ---------------------------------------------------------------------------

const std::vector<Matrix>& HeckeOperatorFamily::T_p(uint64_t p) {
    PipelineEntry& entry = get_or_build_pipeline(p, 1);
    return entry.op->get_H_matrices();
}

// ---------------------------------------------------------------------------
// T_p_pow: the algebra recursion, [C, thm:relations]
// ---------------------------------------------------------------------------

std::vector<Matrix> HeckeOperatorFamily::T_p_pow(uint64_t p, uint32_t r) {
    const auto& dims = cohomL_->get_H_eq_dimensions();
    const uint32_t num_q = static_cast<uint32_t>(dims.size());

    // T_{p^0} = I per degree (0x0 stays 0x0 -- Matrix(0, ell) is the empty
    // identity, and Matrix's *,- operators are well-defined on 0x0 inputs).
    std::vector<Matrix> A0;
    A0.reserve(num_q);
    for (uint32_t d : dims) A0.emplace_back(d, ell_);
    if (r == 0) return A0;

    // T_{p^1} = T_p(p) (a copy of the cached reference).
    const std::vector<Matrix> Tp = T_p(p);
    if (r == 1) return Tp;

    const uint32_t scalar = recursion_scalar(p);

    // Iterative recursion T_{p^{k+1}} = T_p * T_{p^k} - scalar * T_{p^{k-1}},
    // fixed left-to-right order (T_p on the left of every product; see
    // header note on bit-reproducibility).
    std::vector<Matrix> A_km1 = A0; // A_0
    std::vector<Matrix> A_k = Tp;   // A_1
    for (uint32_t k = 1; k < r; ++k) {
        std::vector<Matrix> A_next;
        A_next.reserve(num_q);
        for (uint32_t q = 0; q < num_q; ++q) {
            Matrix term1 = Tp[q] * A_k[q];
            Matrix term2 = A_km1[q] * scalar;
            A_next.push_back(term1 - term2);
        }
        A_km1 = std::move(A_k);
        A_k = std::move(A_next);
    }
    return A_k; // == A_r
}

// ---------------------------------------------------------------------------
// T_p_pow_primitive_direct: the raw PRIMITIVE double-coset operator T'_{p^r}
// of Gamma diag(1, p^r) Gamma (the pipeline HeckeCosetTables(p,r) computes
// directly). For r == 1 this is T_p; for r >= 2 it is NOT the standard
// T_{p^r} (see the header block on the primitive-vs-full distinction and the
// contract-clarification note).
// ---------------------------------------------------------------------------

std::vector<Matrix> HeckeOperatorFamily::T_p_pow_primitive_direct(uint64_t p, uint32_t r) {
    if (r == 0) {
        const auto& dims = cohomL_->get_H_eq_dimensions();
        std::vector<Matrix> result;
        result.reserve(dims.size());
        for (uint32_t d : dims) result.emplace_back(d, ell_); // per-degree identity
        return result;
    }
    PipelineEntry& entry = get_or_build_pipeline(p, r);
    return entry.op->get_H_matrices(); // by value (copy), per the declared interface
}

// ---------------------------------------------------------------------------
// T_p_pow_direct: the INDEPENDENT full-operator route -- assembles the
// standard T_{p^r} from the primitive double-coset operators by the classical
// determinant-p^r decomposition
//     T_{p^r}^full = sum_{j=0}^{floor(r/2)} p^{j n} . T'_{p^{r-2j}},
// with T'_{p^0} := I and the scalar the j-th power of the diamond character
// T(p,p) = p^n I ([C, prop:diamond]). By construction this MATCHES
// T_p_pow(p, r) (the [C, thm:relations] recursion) for every r; the S9 smoke
// and S9V's O9/O10 assert the equality.
// ---------------------------------------------------------------------------

std::vector<Matrix> HeckeOperatorFamily::T_p_pow_direct(uint64_t p, uint32_t r) {
    const auto& dims = cohomL_->get_H_eq_dimensions();
    const uint32_t num_q = static_cast<uint32_t>(dims.size());
    const uint32_t p_mod = static_cast<uint32_t>(p % ell_);
    const engine::ModCtx<uint32_t> ctx(ell_);

    // Accumulate into per-degree ZERO matrices (0x0 stays 0x0).
    std::vector<Matrix> result;
    result.reserve(num_q);
    for (uint32_t d : dims) result.emplace_back(d, d, ell_);

    // (Phase 4, axis (d)) every primitive pipeline this sum needs, built
    // concurrently up front; the summation below then only reads the cache.
    {
        std::vector<std::pair<uint64_t, uint32_t>> keys;
        for (uint32_t j = 0; 2 * j < r; ++j) keys.emplace_back(p, r - 2 * j);
        if (!keys.empty()) build_pipelines(std::move(keys));
    }

    // Fixed summation order: j ascending, 0 <= 2j <= r.
    for (uint32_t j = 0; 2 * j <= r; ++j) {
        const uint32_t m = r - 2 * j; // exponent of the primitive term T'_{p^m}
        const uint32_t scalar = ctx.pow(p_mod, j * n_sym_); // p^{j n} mod ell
        if (m == 0) {
            // T'_{p^0} = I; add scalar * I per degree.
            for (uint32_t q = 0; q < num_q; ++q) {
                Matrix I(dims[q], ell_);
                result[q] = result[q] + (I * scalar);
            }
        } else {
            const std::vector<Matrix> prim = T_p_pow_primitive_direct(p, m);
            for (uint32_t q = 0; q < num_q; ++q) {
                result[q] = result[q] + (prim[q] * scalar);
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// T_n: multiplicativity over the prime factorization (trial division)
// ---------------------------------------------------------------------------

std::vector<Matrix> HeckeOperatorFamily::T_n(uint64_t n) {
    if (n == 0) {
        throw std::invalid_argument("HeckeOperatorFamily::T_n: n == 0 is not a Hecke index.");
    }

    const auto& dims = cohomL_->get_H_eq_dimensions();
    const uint32_t num_q = static_cast<uint32_t>(dims.size());

    // Start from the identity (handles n == 1, the empty product).
    std::vector<Matrix> result;
    result.reserve(num_q);
    for (uint32_t d : dims) result.emplace_back(d, ell_);

    // Trial division, ascending order (n < 2^32 assumed, per the plan).
    uint64_t m = n;
    for (uint64_t d = 2; d * d <= m; ++d) {
        if (m % d != 0) continue;
        uint32_t r = 0;
        while (m % d == 0) {
            m /= d;
            ++r;
        }
        const std::vector<Matrix> factor = T_p_pow(d, r);
        for (uint32_t q = 0; q < num_q; ++q) {
            result[q] = result[q] * factor[q]; // fixed left-to-right, ascending prime order
        }
    }
    if (m > 1) {
        const std::vector<Matrix> factor = T_p_pow(m, 1);
        for (uint32_t q = 0; q < num_q; ++q) {
            result[q] = result[q] * factor[q];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// T_pp: the central/diamond scalar, [C, prop:diamond]
// ---------------------------------------------------------------------------

std::vector<Matrix> HeckeOperatorFamily::T_pp(uint64_t p) {
    const auto& dims = cohomL_->get_H_eq_dimensions();
    const uint32_t p_mod = static_cast<uint32_t>(p % ell_);
    const uint32_t scalar = engine::ModCtx<uint32_t>(ell_).pow(p_mod, n_sym_);

    std::vector<Matrix> result;
    result.reserve(dims.size());
    for (uint32_t d : dims) {
        Matrix I(d, ell_);
        result.push_back(I * scalar);
    }
    return result;
}

// ---------------------------------------------------------------------------
// theta_flat: level-(a) passthrough for T_p
// ---------------------------------------------------------------------------

std::vector<Matrix> HeckeOperatorFamily::theta_flat(uint64_t p, uint32_t q) const {
    PipelineEntry& entry = get_or_build_pipeline(p, 1);
    const std::vector<FlatMatrix>& theta = entry.op->get_Theta_flat_matrices();
    if (q >= theta.size()) {
        throw std::out_of_range("HeckeOperatorFamily::theta_flat: degree q out of range.");
    }
    // See the header note: the declared return type is std::vector<Matrix>
    // even though this passes through a SINGLE FlatMatrix (get_Theta_flat_
    // matrices()[q]); wrapped in a size-1 vector to satisfy the literal
    // signature while preserving the "passthrough" semantics.
    return { theta[q].to_matrix() };
}

// ---------------------------------------------------------------------------
// theta_flat_fm: the same passthrough without the to_matrix() copy (item 7.6)
// ---------------------------------------------------------------------------

const FlatMatrix& HeckeOperatorFamily::theta_flat_fm(uint64_t p, uint32_t q) const {
    PipelineEntry& entry = get_or_build_pipeline(p, 1);
    const std::vector<FlatMatrix>& theta = entry.op->get_Theta_flat_matrices();
    if (q >= theta.size()) {
        throw std::out_of_range("HeckeOperatorFamily::theta_flat_fm: degree q out of range.");
    }
    return theta[q];
}

} // namespace hecke
