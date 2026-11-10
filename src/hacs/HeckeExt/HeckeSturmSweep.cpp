// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1b (slot 3, WP-2) -- see HeckeSturmSweep.h.

#include "HeckeSturmSweep.h"

#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/SubspaceOps.h"
#include "../Util/Rss.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {

namespace {

// --- small shared helpers --------------------------------------------------

std::string JsonBool(bool b) { return b ? "true" : "false"; }

// The D*D row-major coordinate vector of a D x D matrix -- the coordinates
// hecke::algebra_closure / hecke::RrefRowSpan work in.
std::vector<uint32_t> VecOf(const ExtMatrix& M) {
    std::vector<uint32_t> v;
    v.reserve(static_cast<size_t>(M.rows) * M.cols);
    for (uint32_t i = 0; i < M.rows; ++i) {
        for (uint32_t j = 0; j < M.cols; ++j) v.push_back(M.data[i][j]);
    }
    return v;
}

bool SameMatrix(const ExtMatrix& a, const ExtMatrix& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.p != b.p) return false;
    for (uint32_t i = 0; i < a.rows; ++i) {
        for (uint32_t j = 0; j < a.cols; ++j) {
            if (a.data[i][j] != b.data[i][j]) return false;
        }
    }
    return true;
}

bool IsPrime(uint64_t q) {
    if (q < 2) return false;
    if (q % 2 == 0) return q == 2;
    for (uint64_t d = 3; d * d <= q; d += 2) {
        if (q % d == 0) return false;
    }
    return true;
}

// A span over the D*D coordinates of A_S, for the membership test.
hecke::RrefRowSpan SpanOf(const std::vector<ExtMatrix>& basis, uint32_t D, uint32_t ell) {
    hecke::RrefRowSpan span(D * D, ell);
    for (const ExtMatrix& b : basis) span.insert(VecOf(b));
    return span;
}

} // namespace

// ---------------------------------------------------------------------------
// Primes
// ---------------------------------------------------------------------------

std::vector<uint64_t> PrimesUpTo(uint32_t B) {
    std::vector<uint64_t> out;
    for (uint64_t q = 2; q <= static_cast<uint64_t>(B); ++q) {
        if (IsPrime(q)) out.push_back(q);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Stage 1
// ---------------------------------------------------------------------------

const std::vector<uint64_t>& DefaultGoodPrimes() {
    static const std::vector<uint64_t> g = {2, 3, 5, 7};
    return g;
}

bool StageOneOperators::includes_u_ell() const { return u_ell.included; }

std::string StageOneOperators::to_json() const {
    std::ostringstream oss;
    oss << "{\"n\":" << n << ",\"ell\":" << ell << ",\"primes\":[";
    for (size_t i = 0; i < primes.size(); ++i) {
        if (i) oss << ",";
        oss << primes[i];
    }
    oss << "],\"u_ell\":{\"included\":" << JsonBool(u_ell.included) << ",\"reason\":\""
        << to_string(u_ell.reason) << "\",\"predicted_mb\":" << u_ell.predicted_mb
        << ",\"budget_mb\":" << u_ell.budget_mb << ",\"model_version\":\""
        << u_ell.model_version << "\"}}";
    return oss.str();
}

StageOneOperators ComputeStageOneOperators(uint32_t n, uint32_t ell,
                                           const hecke::UEllSetting& setting,
                                           const std::vector<uint64_t>& good_primes) {
    ENGINE_CHECK_MSG(ell >= 2, "ComputeStageOneOperators: ell < 2");
    StageOneOperators out;
    out.n = n;
    out.ell = ell;
    out.u_ell = UEllDecision(n, ell, setting);

    // The good primes, ell removed: at ell <= 7 the good prime ell IS U_ell
    // (the same code path, invariant I7), so the budget decides it there too.
    for (uint64_t q : good_primes) {
        if (q == static_cast<uint64_t>(ell)) continue;
        out.primes.push_back(q);
    }
    if (out.u_ell.included) out.primes.push_back(static_cast<uint64_t>(ell));
    std::sort(out.primes.begin(), out.primes.end());
    out.primes.erase(std::unique(out.primes.begin(), out.primes.end()), out.primes.end());
    return out;
}

StageOneOperators ComputeStageOneOperators(uint32_t n, uint32_t ell,
                                           const hecke::UEllSetting& setting) {
    return ComputeStageOneOperators(n, ell, setting, DefaultGoodPrimes());
}

StageOneOperators ComputeStageOneOperators(uint32_t n, uint32_t ell) {
    return ComputeStageOneOperators(n, ell, hecke::math_settings().u_ell, DefaultGoodPrimes());
}

// ---------------------------------------------------------------------------
// BlockRestrictionSource
// ---------------------------------------------------------------------------

BlockRestrictionSource::BlockRestrictionSource(ExtMatrix basis_columns, Supplier supplier)
    : basis_(std::move(basis_columns)), supplier_(std::move(supplier)) {
    ENGINE_CHECK_MSG(basis_.cols > 0 && basis_.rows > 0,
                     "BlockRestrictionSource: empty block basis");
    ENGINE_CHECK_MSG(static_cast<bool>(supplier_),
                     "BlockRestrictionSource: no supplier");
}

bool BlockRestrictionSource::restricted_operator(uint64_t q, ExtMatrix& out) {
    for (size_t i = 0; i < cached_q_.size(); ++i) {
        if (cached_q_[i] == q) {
            if (!cached_ok_[i]) return false;
            out = cached_[i];
            return true;
        }
    }
    ExtMatrix full;
    const bool ok = supplier_(q, full);
    cached_q_.push_back(q);
    if (!ok) {
        cached_.push_back(ExtMatrix());
        cached_ok_.push_back(0);
        return false;
    }
    ENGINE_CHECK_MSG(full.rows == basis_.rows && full.cols == basis_.rows,
                     "BlockRestrictionSource: the supplied operator does not match the "
                     "ambient dimension of the block basis");
    // Solvability of basis * X = full * basis IS the invariance certificate.
    ExtMatrix restricted = hecke::SolveColumnsExact(basis_, full * basis_,
                                                    "BlockRestrictionSource(restriction)");
    ++solves_;
    cached_.push_back(restricted);
    cached_ok_.push_back(1);
    out = std::move(restricted);
    return true;
}

BlockRestrictionSource::Supplier FamilyOperatorSupplier(hecke::HeckeOperatorFamily& family,
                                                        uint32_t degree,
                                                        uint64_t ell,
                                                        bool u_ell_included) {
    hecke::HeckeOperatorFamily* fam = &family;
    return [fam, degree, ell, u_ell_included](uint64_t q, ExtMatrix& out) -> bool {
        if (q == ell && !u_ell_included) return false;   // u_ell.reason says why
        const std::vector<Matrix>& A = fam->T_p(q);
        if (degree >= A.size()) return false;
        out = A[degree];
        return true;
    };
}

// ---------------------------------------------------------------------------
// (slot 4, WP-2 / E2b) The streamed supplier -- see the header.
// ---------------------------------------------------------------------------

namespace {

// (S23 PERF-P4a repairs) The streamed route reports its own peak ([X] plan
// Sec. 3 WP-2 risk 1: "RSS per prime is recorded") through the one shared
// /proc reader, oc_util::ProcStatusMiB (Util/Rss.h).  0.0 where /proc is
// unavailable (ProcStatusMiB's -1), as before -- a diagnostic, never a gate.
double RssMbHere(const char* key) { return std::max(0.0, oc_util::ProcStatusMiB(key)); }

} // namespace

std::vector<ExtMatrix> StreamedOperatorAllDegrees(
        const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
        const hecke::SL2ZTessellationTables& model, uint32_t ell, uint32_t n_sym, uint64_t q,
        StreamedOperatorStats* stats) {
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::vector<ExtMatrix> all;
    {
        // ONE pipeline alive at a time: everything below is destroyed at
        // the end of this scope, before the function returns.
        const hecke::HeckeOperatorOptions::ScopedThetaEqCertification no_cert(false);
        const hecke::HeckeOperatorOptions::ScopedMemoryLean lean(true);
        std::unique_ptr<hecke::HeckeCosetTables> cosets;
        std::unique_ptr<hecke::HeckeTauTable> tau;
        {
            // The two table CONSTRUCTIONS are first-match-wins over a fixed
            // transporter order and drive the non-atomic PredicateStats
            // counters, so they take the process-wide lock the family holds
            // for its own phase A (Hecke/HeckeOperatorFamily.h).
            const std::lock_guard<std::mutex> guard(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTables(q, /*r=*/1, ell, n_sym));
            tau.reset(new hecke::HeckeTauTable(K, model, *cosets, ell, n_sym));
        }
        if (std::getenv("ENGINE_TRACE_RSS") != nullptr)   // (S23 PERF-P4a) diagnostic only
            std::cout << "      [streamed] p=" << q << " tables built, VmRSS=" << std::fixed << std::setprecision(1)
                      << RssMbHere("VmRSS:") << " MB (heap in use " << hecke::HeckeOperatorOptions::heap_in_use_mib() << " MB)" << std::endl;
        hecke::HeckeOperatorOnK op(&coh, cosets.get(), tau.get());
        op.compute();
        all = op.get_H_matrices();
    }
    // (slot-24 G6, HeckeOperatorOptions (6)(c)) The pipeline's tables and
    // blocks (~2.4 GB at (n, p) = (1000, 103)) were freed above; hand the
    // allocator's free pages back so they do not ratchet the cell's floor
    // from one streamed prime to the next.  Memory only.
    const double rss_before_trim = std::getenv("ENGINE_TRACE_RSS") != nullptr ? RssMbHere("VmRSS:") : 0.0;
    if (hecke::HeckeOperatorOptions::field_diet()) hecke::HeckeOperatorOptions::release_free_heap();
    if (std::getenv("ENGINE_TRACE_RSS") != nullptr)   // diagnostic only
        std::cout << "      [streamed] p=" << q << " pipeline released, VmRSS=" << std::fixed << std::setprecision(1)
                  << rss_before_trim << " MB -> " << RssMbHere("VmRSS:") << " MB (heap in use " << hecke::HeckeOperatorOptions::heap_in_use_mib() << " MB)" << std::endl;
    if (stats != nullptr) {
        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        stats->built += 1;
        stats->wall_s += std::chrono::duration<double>(t1 - t0).count();
        stats->peak_rss_mb = RssMbHere("VmHWM:");
        stats->last_prime = q;
    }
    return all;
}

BlockRestrictionSource::Supplier StreamedOperatorSupplier(
        const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
        const hecke::SL2ZTessellationTables& model, uint32_t ell, uint32_t n_sym,
        uint32_t degree, bool u_ell_included, StreamedOperatorStats* stats) {
    const EquivariantCohomologyOnK* c = &coh;
    const SimplicialComplex* kp = &K;
    const hecke::SL2ZTessellationTables* mp = &model;
    return [c, kp, mp, ell, n_sym, degree, u_ell_included, stats](
                   uint64_t q, ExtMatrix& out) -> bool {
        if (q == ell && !u_ell_included) return false;   // u_ell.reason says why
        std::vector<ExtMatrix> A = StreamedOperatorAllDegrees(*c, *kp, *mp, ell, n_sym, q, stats);
        if (degree >= A.size()) return false;
        out = std::move(A[degree]);
        return true;
    };
}

BlockRestrictionSource::Supplier ReuseBeforeStreamingSupplier(
        std::vector<uint64_t> s0,
        BlockRestrictionSource::Supplier reuse,
        BlockRestrictionSource::Supplier stream) {
    std::sort(s0.begin(), s0.end());
    s0.erase(std::unique(s0.begin(), s0.end()), s0.end());
    return [s0, reuse, stream](uint64_t q, ExtMatrix& out) -> bool {
        const bool cached = std::binary_search(s0.begin(), s0.end(), q);
        if (cached) {
            ENGINE_CHECK_MSG(static_cast<bool>(reuse),
                             "ReuseBeforeStreamingSupplier: no reuse supplier");
            return reuse(q, out);
        }
        ENGINE_CHECK_MSG(static_cast<bool>(stream),
                         "ReuseBeforeStreamingSupplier: no streaming supplier");
        return stream(q, out);
    };
}

// ---------------------------------------------------------------------------
// Enum text
// ---------------------------------------------------------------------------

const char* ToString(SturmSweepOutcome outcome) {
    switch (outcome) {
        case SturmSweepOutcome::InProgress:    return "in_progress";
        case SturmSweepOutcome::NotApplicable: return "not_applicable";
        case SturmSweepOutcome::Passed:        return "passed";
        case SturmSweepOutcome::Failed:        return "failed";
        case SturmSweepOutcome::Unavailable:   return "unavailable";
        case SturmSweepOutcome::Heuristic:     return "heuristic";
    }
    return "in_progress";
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

SturmSweepOptions SturmSweepOptions::from_settings(const hecke::MathSettings& s, uint32_t k) {
    SturmSweepOptions o;
    switch (s.sturm.mode) {
        case hecke::SturmCapMode::Off:
            o.sturm_off = true;
            o.cap = SturmBound::default_cap(k);
            break;
        case hecke::SturmCapMode::Cap:
            o.cap = s.sturm.cap;
            break;
        case hecke::SturmCapMode::Auto:
        default:
            o.cap = SturmBound::default_cap(k);
            break;
    }
    o.mode = (s.sturm_unavailable == hecke::SturmUnavailablePolicy::Heuristic)
                 ? SturmUnavailableMode::Heuristic
                 : SturmUnavailableMode::NonFinal;
    return o;
}

SturmSweepOptions SturmSweepOptions::from_settings(uint32_t k) {
    return from_settings(hecke::math_settings(), k);
}

// ---------------------------------------------------------------------------
// SturmSweepState
// ---------------------------------------------------------------------------

bool SturmSweepState::equals(const SturmSweepState& o, bool ignore_budget_history) const {
    if (n != o.n || k != o.k || ell != o.ell || D != o.D) return false;
    if (cap != o.cap) return false;
    if (!ignore_budget_history && budget_history != o.budget_history) return false;
    if (!bound.equals(o.bound, ignore_budget_history)) return false;
    if (swept != o.swept || member != o.member) return false;
    if (t_q.size() != o.t_q.size()) return false;
    for (size_t i = 0; i < t_q.size(); ++i) {
        if (!SameMatrix(t_q[i], o.t_q[i])) return false;
    }
    return failing_prime == o.failing_prime && u_ell_blocked == o.u_ell_blocked &&
           mode == o.mode && proof_range == o.proof_range && sturm_off == o.sturm_off &&
           outcome == o.outcome;
}

std::string SturmSweepState::to_json() const {
    std::ostringstream oss;
    oss << "{\"kind\":\"sturm_sweep\",\"n\":" << n << ",\"k\":" << k << ",\"ell\":" << ell
        << ",\"D\":" << D << ",\"budget\":" << cap << ",\"budget_history\":[";
    for (size_t i = 0; i < budget_history.size(); ++i) {
        if (i) oss << ",";
        oss << budget_history[i];
    }
    oss << "],\"swept\":[";
    for (size_t i = 0; i < swept.size(); ++i) {
        if (i) oss << ",";
        oss << swept[i];
    }
    oss << "],\"member\":[";
    for (size_t i = 0; i < member.size(); ++i) {
        if (i) oss << ",";
        oss << JsonBool(member[i] != 0);
    }
    oss << "],\"failing_prime\":" << failing_prime
        << ",\"u_ell_blocked\":" << JsonBool(u_ell_blocked)
        << ",\"mode\":\"" << ToString(mode) << "\""
        << ",\"proof_range\":" << JsonBool(proof_range)
        << ",\"sturm_off\":" << JsonBool(sturm_off)
        << ",\"outcome\":\"" << ToString(outcome) << "\""
        << ",\"bound\":" << bound.to_json() << "}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

SturmSweep::SturmSweep(const LocalAlgebra& L, uint32_t n, SturmOperatorSource& src,
                       const SturmSweepOptions& opts) {
    ENGINE_CHECK_MSG(L.ok, "SturmSweep: the block did not certify local");
    ENGINE_CHECK_MSG(L.D > 0, "SturmSweep: empty block");
    ell_ = L.ell;
    D_ = L.D;
    a_basis_ = L.basis;
    cyc_ = L.cyclicity;
    src_ = &src;

    st_.n = n;
    st_.k = n + 2;
    st_.ell = L.ell;
    st_.D = L.D;
    st_.mode = opts.mode;
    st_.proof_range = opts.proof_range;
    st_.sturm_off = opts.sturm_off;
    st_.cap = (opts.cap != 0) ? opts.cap : SturmBound::default_cap(st_.k);
}

uint32_t SturmSweep::B_prime() const {
    return has_B_prime() ? st_.bound.B_prime : 0u;
}

std::vector<uint64_t> SturmSweep::prime_list_() const {
    if (st_.bound.outcome == SturmOutcome::Found) return PrimesUpTo(st_.bound.B_prime);
    if (st_.mode == SturmUnavailableMode::Heuristic) return PrimesUpTo(st_.cap);
    return std::vector<uint64_t>();
}

void SturmSweep::sweep_() {
    st_.swept.clear();
    st_.member.clear();
    st_.t_q.clear();
    st_.failing_prime = 0;
    st_.u_ell_blocked = false;

    const std::vector<uint64_t> primes = prime_list_();
    const hecke::RrefRowSpan span = SpanOf(a_basis_, D_, ell_);

    for (uint64_t q : primes) {
        ExtMatrix t;
        if (!src_->restricted_operator(q, t)) {
            // Today this is exactly a refused U_ell ([X] Sec. 2.2 Stage 4(c)
            // item 2): the sweep cannot be completed, so (c) is unavailable.
            st_.u_ell_blocked = true;
            st_.swept.clear();
            st_.member.clear();
            st_.t_q.clear();
            st_.outcome = SturmSweepOutcome::Unavailable;
            return;
        }
        ENGINE_CHECK_MSG(t.rows == D_ && t.cols == D_ && t.p == ell_,
                         "SturmSweep: the source returned a restriction of the wrong shape");
        const bool in_A = span.contains(VecOf(t));
        st_.swept.push_back(q);
        st_.member.push_back(in_A ? 1u : 0u);
        st_.t_q.push_back(std::move(t));
        if (!in_A && st_.failing_prime == 0) st_.failing_prime = q;
    }

    if (st_.failing_prime != 0) {
        st_.outcome = SturmSweepOutcome::Failed;
        return;
    }
    if (st_.bound.outcome == SturmOutcome::Found) {
        st_.outcome = st_.proof_range ? SturmSweepOutcome::Passed : SturmSweepOutcome::Heuristic;
        return;
    }
    // No B': `heuristic` swept the primes <= cap (all members), `nonfinal`
    // swept nothing.
    st_.outcome = (st_.mode == SturmUnavailableMode::Heuristic) ? SturmSweepOutcome::Heuristic
                                                                : SturmSweepOutcome::Unavailable;
}

SturmSweepOutcome SturmSweep::run() {
    ENGINE_CHECK_MSG(!ran_, "SturmSweep::run called twice; use resume()");
    ran_ = true;
    st_.budget_history.push_back(st_.cap);

    if (SturmBound::dimension_M_k(st_.k) == 0) {
        // [E] v1.4.0 Def. 2.8: n = 0, odd n and k < 4 have no M_k to bound.
        st_.outcome = SturmSweepOutcome::NotApplicable;
        return st_.outcome;
    }
    if (st_.sturm_off) {
        st_.outcome = SturmSweepOutcome::Unavailable;
        return st_.outcome;
    }

    {
        const std::chrono::steady_clock::time_point b0 = std::chrono::steady_clock::now();
        bound_.reset(new SturmBound(st_.k, st_.ell, st_.cap));
        bound_->run();
        bound_wall_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - b0).count();
    }
    st_.bound = bound_->state();
    sweep_();
    return st_.outcome;
}

SturmSweepOutcome SturmSweep::resume(uint32_t new_cap) {
    ENGINE_CHECK_MSG(ran_, "SturmSweep::resume before run()");
    if (new_cap < st_.cap) {
        throw std::invalid_argument("SturmSweep: resume: cap below reached state");
    }
    // A strict no-op on anything a higher cap cannot change: a found bound,
    // `sturm = off`, a `not_applicable` weight, a completed sweep, or a
    // U_ell the budget refused (a cap does not buy it).
    if (st_.outcome == SturmSweepOutcome::NotApplicable || st_.sturm_off ||
        st_.bound.outcome == SturmOutcome::Found || st_.u_ell_blocked) {
        return st_.outcome;
    }

    st_.cap = new_cap;
    st_.budget_history.push_back(new_cap);

    // The bound object is kept alive precisely so that this CONTINUES it from
    // the prime it reached -- it is never re-run from scratch (R3.1 item 2).
    ENGINE_CHECK_MSG(bound_ != nullptr, "SturmSweep::resume: no bound to continue");
    {
        const std::chrono::steady_clock::time_point b0 = std::chrono::steady_clock::now();
        bound_->resume(new_cap);
        bound_wall_s_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - b0).count();
    }
    st_.bound = bound_->state();
    sweep_();
    return st_.outcome;
}

SturmCertificate SturmSweep::certificate() const {
    SturmCertificate c;
    c.n = st_.n;
    c.k = st_.k;
    c.ell = st_.ell;
    c.D = st_.D;
    c.has_B_prime = (st_.bound.outcome == SturmOutcome::Found);
    c.B_prime = c.has_B_prime ? st_.bound.B_prime : 0u;
    c.cap = st_.cap;
    // dim M_k is a PURE function of k, so it is emitted even when the bound
    // never ran (`sturm = off`) -- verify()'s applicability test reads it.
    c.dim_M_k = SturmBound::dimension_M_k(st_.k);
    c.closure_dim = st_.bound.closure_dim;
    c.mode = st_.mode;
    c.proof_range = st_.proof_range;
    c.sturm_off = st_.sturm_off;
    c.sturm_primes = st_.swept;
    c.t_q = st_.t_q;
    c.member = st_.member;
    for (size_t i = 0; i < st_.swept.size(); ++i) {
        if (st_.member[i]) c.generators_certified.push_back(st_.swept[i]);
    }
    c.failing_prime = st_.failing_prime;
    c.u_ell_blocked = st_.u_ell_blocked;
    c.outcome = st_.outcome;
    c.cyclicity = cyc_;
    c.state = st_;

    if (bound_ && st_.outcome != SturmSweepOutcome::NotApplicable && !st_.sturm_off) {
        c.bound = bound_->certificate();
    }
    return c;
}

// ---------------------------------------------------------------------------
// The certificate
// ---------------------------------------------------------------------------

namespace {

// The outcome re-derived from the emitted data alone -- verify()'s clause 9
// and the single place the (outcome, mode, range) map is written down.
SturmSweepOutcome DeriveOutcome(const SturmCertificate& c) {
    if (c.dim_M_k == 0) return SturmSweepOutcome::NotApplicable;
    if (c.sturm_off || c.u_ell_blocked) return SturmSweepOutcome::Unavailable;
    for (size_t i = 0; i < c.member.size(); ++i) {
        if (!c.member[i]) return SturmSweepOutcome::Failed;
    }
    if (c.has_B_prime) {
        return c.proof_range ? SturmSweepOutcome::Passed : SturmSweepOutcome::Heuristic;
    }
    return (c.mode == SturmUnavailableMode::Heuristic) ? SturmSweepOutcome::Heuristic
                                                       : SturmSweepOutcome::Unavailable;
}

const char* const kMessages[] = {
    /* 0 */ "ok",
    /* 1 */ "shape: k != n + 2, or the block dimensions disagree with certificate (a)",
    /* 2 */ "the embedded SturmBoundCertificate does not verify",
    /* 3 */ "the embedded bound describes another (k, ell, cap) or another B'",
    /* 4 */ "certificate (a) does not verify",
    /* 5 */ "the swept prime list is not the one B' / the cap prescribes",
    /* 6 */ "the swept arrays are not index-aligned, or a restriction has the wrong shape",
    /* 7 */ "a stored membership flag disagrees with the re-derived algebra A_S",
    /* 8 */ "generators_certified is not the sublist of members, or failing_prime is wrong",
    /* 9 */ "the outcome disagrees with the one the emitted data implies"
};

} // namespace

const char* SturmCertificate::verify_message(uint32_t code) {
    const uint32_t n = sizeof(kMessages) / sizeof(kMessages[0]);
    return code < n ? kMessages[code] : "unknown code";
}

uint32_t SturmCertificate::verify() const {
    // 1 -- shape.
    if (k != n + 2) return 1;
    if (cyclicity.D != D || cyclicity.ell != ell) return 1;
    if (ell < 2) return 1;

    const bool applicable = (dim_M_k != 0) && !sturm_off;

    // 2, 3 -- the bound.
    if (applicable) {
        if (bound.verify() != 0) return 2;
        if (bound.k != k || bound.ell != ell || bound.cap != cap) return 3;
        if (bound.has_B_prime != has_B_prime) return 3;
        if (has_B_prime && bound.B_prime != B_prime) return 3;
        if (bound.dim_M_k != dim_M_k || bound.closure_dim != closure_dim) return 3;
    } else if (dim_M_k != 0 && sturm_off) {
        // `sturm = off`: no bound is emitted, and nothing may be swept.
        if (!sturm_primes.empty()) return 5;
    }

    // 4 -- certificate (a) re-derives A_S from the stored t_p.
    if (cyclicity.verify() != 0) return 4;

    // 5 -- the prime list.
    std::vector<uint64_t> expect;
    if (applicable && !u_ell_blocked) {
        if (has_B_prime) {
            expect = PrimesUpTo(B_prime);
        } else if (mode == SturmUnavailableMode::Heuristic) {
            expect = PrimesUpTo(cap);
        }
    }
    if (sturm_primes != expect) return 5;

    // 6 -- alignment and shapes.
    if (t_q.size() != sturm_primes.size() || member.size() != sturm_primes.size()) return 6;
    for (const ExtMatrix& t : t_q) {
        if (t.rows != D || t.cols != D || t.p != ell) return 6;
    }

    // 7 -- membership against a RE-DERIVED A_S (never the stored basis).
    {
        uint32_t dim_A = 0;
        const Matrix coords = hecke::algebra_closure(cyclicity.t, D, ell, dim_A);
        hecke::RrefRowSpan span(D * D, ell);
        for (uint32_t i = 0; i < dim_A; ++i) span.insert(coords.data[i]);
        for (size_t i = 0; i < t_q.size(); ++i) {
            const bool in_A = span.contains(VecOf(t_q[i]));
            if (in_A != (member[i] != 0)) return 7;
        }
    }

    // 8 -- generators_certified and failing_prime.
    {
        std::vector<uint64_t> gens;
        uint64_t first_bad = 0;
        for (size_t i = 0; i < sturm_primes.size(); ++i) {
            if (member[i]) {
                gens.push_back(sturm_primes[i]);
            } else if (first_bad == 0) {
                first_bad = sturm_primes[i];
            }
        }
        if (gens != generators_certified) return 8;
        if (first_bad != failing_prime) return 8;
    }

    // 9 -- the outcome.
    if (DeriveOutcome(*this) != outcome) return 9;
    return 0;
}

std::string SturmCertificate::to_json() const {
    std::ostringstream oss;
    oss << "{\"schema\":\"sturm_certificate/v1\",\"n\":" << n << ",\"k\":" << k
        << ",\"ell\":" << ell << ",\"D\":" << D << ",\"B_prime\":";
    if (has_B_prime) {
        oss << B_prime;
    } else {
        oss << "null";
    }
    oss << ",\"cap\":" << cap << ",\"dim_M_k\":" << dim_M_k
        << ",\"closure_dim\":" << closure_dim
        << ",\"mode\":\"" << ToString(mode) << "\""
        << ",\"proof_range\":" << JsonBool(proof_range)
        << ",\"sturm_off\":" << JsonBool(sturm_off)
        << ",\"sturm_primes\":[";
    for (size_t i = 0; i < sturm_primes.size(); ++i) {
        if (i) oss << ",";
        oss << sturm_primes[i];
    }
    oss << "],\"generators_certified\":[";
    for (size_t i = 0; i < generators_certified.size(); ++i) {
        if (i) oss << ",";
        oss << generators_certified[i];
    }
    oss << "],\"failing_prime\":" << failing_prime
        << ",\"u_ell_blocked\":" << JsonBool(u_ell_blocked)
        << ",\"outcome\":\"" << ToString(outcome) << "\""
        << ",\"resumable\":" << resumable_json() << "}";
    return oss.str();
}

std::string SturmCertificate::resumable_json() const {
    std::ostringstream oss;
    oss << "{\"kind\":\"sturm_sweep\",\"budget_setting\":\"sturm.cap\",\"budget\":" << cap
        << ",\"budget_history\":[";
    for (size_t i = 0; i < state.budget_history.size(); ++i) {
        if (i) oss << ",";
        oss << state.budget_history[i];
    }
    oss << "],\"complete\":"
        << JsonBool(outcome != SturmSweepOutcome::InProgress &&
                    !(outcome == SturmSweepOutcome::Unavailable && !has_B_prime && !sturm_off &&
                      !u_ell_blocked && dim_M_k != 0))
        << ",\"sidecar_stem\":\"" << state_sidecar_stem << "\",\"state\":" << state.to_json()
        << "}";
    return oss.str();
}

// ---------------------------------------------------------------------------
// The verdicts
// ---------------------------------------------------------------------------

SturmRouteStatus sturm_route_status(const SturmCertificate& c) {
    switch (c.outcome) {
        case SturmSweepOutcome::Passed:    return SturmRouteStatus::Passed;
        case SturmSweepOutcome::Failed:    return SturmRouteStatus::Failed;
        case SturmSweepOutcome::Heuristic:
            // A heuristic is not a proof: in the DEFAULT configuration (c) is
            // unavailable.  `sweep_finality` is where the selected heuristic
            // mode turns it into the tri-state value.
            return SturmRouteStatus::Unavailable;
        case SturmSweepOutcome::Unavailable:
        case SturmSweepOutcome::NotApplicable:
        case SturmSweepOutcome::InProgress:
        default:
            return SturmRouteStatus::Unavailable;
    }
}

SturmFinality sweep_finality(const SturmCertificate& c) {
    SturmFinality f;
    switch (c.outcome) {
        case SturmSweepOutcome::Passed:
            f.value = SturmFinal::True;
            f.reason.clear();
            return f;
        case SturmSweepOutcome::Heuristic:
            if (c.mode == SturmUnavailableMode::Heuristic) {
                f.value = SturmFinal::Heuristic;
                f.reason = "sturm_heuristic";
            } else {
                f.value = SturmFinal::False;
                f.reason = "sturm_heuristic";
            }
            return f;
        case SturmSweepOutcome::Failed:
            f.value = SturmFinal::False;
            f.reason = "certificate_failed:sturm";
            return f;
        case SturmSweepOutcome::NotApplicable:
        case SturmSweepOutcome::Unavailable:
        case SturmSweepOutcome::InProgress:
        default:
            f.value = SturmFinal::False;
            f.reason = "sturm_unavailable";
            return f;
    }
}

GenerationVerdict generation_verdict_with_sweep(bool cyclic_passed, CommutantOutcome b,
                                                const SturmCertificate& c) {
    GenerationVerdict v = generation_verdict(cyclic_passed, b, sturm_route_status(c));
    // The only place the default map cannot reach: a complete but non-proving
    // sweep under the user-selected heuristic mode ([X] Sec. 1.3.1 R3.2).
    if (v.kind == GenerationKind::Unavailable && c.outcome == SturmSweepOutcome::Heuristic &&
        c.mode == SturmUnavailableMode::Heuristic) {
        v.kind = GenerationKind::Sturm;
        v.final_value = SturmFinal::Heuristic;
        v.reason = "sturm_heuristic";
    }
    return v;
}

} // namespace hecke_ext
