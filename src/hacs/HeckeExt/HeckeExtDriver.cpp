// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2b (slot 4, WP-2) -- see HeckeExtDriver.h.

#include "HeckeExtDriver.h"

#include "HeckeAtlas.h"
#include "HeckeExtReport.h"
#include "HeckeFinalityRoutes.h"
#include "GloverFactors.h"
#include "HeckePredictions.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/DicksonCount.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/SubspaceOps.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../Hecke/HeckeSettings.h"
#include "../Util/Rss.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {

namespace {

using Clock = std::chrono::steady_clock;

// (PERF-VF audit S3) the Stage-1 release floor's validation hook and the release counter.
thread_local uint32_t g_release_min_dim_ceq = kFieldDietReleaseMinDimCeq;
std::atomic<uint64_t> g_stage_one_releases{0};

} // namespace

uint32_t field_diet_release_min_dim_ceq() { return g_release_min_dim_ceq; }
ScopedReleaseMinDimCeq::ScopedReleaseMinDimCeq(uint32_t floor) : saved_(g_release_min_dim_ceq) { g_release_min_dim_ceq = floor; }
ScopedReleaseMinDimCeq::~ScopedReleaseMinDimCeq() { g_release_min_dim_ceq = saved_; }
uint64_t stage_one_releases() { return g_stage_one_releases.load(std::memory_order_relaxed); }

namespace {

// (slot-12 FIN) A FlatMatrix (r^1 from SL2ZBoundaryPair) as an ExtMatrix.
ExtMatrix FlatAsExt(const FlatMatrix& M) {
    ExtMatrix out(M.get_rows(), M.get_cols(), M.get_p());
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) out.data[i][j] = M.get_entry(i, j);
    return out;
}

double Secs(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double>(b - a).count();
}

std::vector<LocalSystemSimplicialComplex::SimplexRelation>
BuildRelationsHere(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    return rels;
}

std::string GroupName(uint32_t q) {
    std::ostringstream os;
    os << "H" << q;
    return os.str();
}

// ---------------------------------------------------------------------------
// [X] Sec. 2.2 Stage 3's "CONSUMED from the locality verdict".
// ---------------------------------------------------------------------------
//
// Rebuilds the `hecke::LocalityVerdict` the decomposition's refinement loop
// already proved, from the block's OWN certified data:
//   * `witness_poly`, `w_hat_exponent` and `character` verbatim;
//   * the witness matrix W -- from `residue_witness` (budget mode's linear
//     witness) or from `witness_matrix` (strict mode's residue-field witness,
//     already restricted to the block);
//   * w_hat = W^(ell^N) by `hecke::FrobeniusPower`, the ONE thing a
//     `PrimaryBlock` does not store (the k = 1 analytic path never forms it).
// Returns a verdict with `local = false` -- i.e. "fall back to the search" --
// whenever any of that is absent, so the fast path can never silently accept
// less than the slow one.
hecke::LocalityVerdict VerdictFromCertifiedBlock(
        const hecke::PrimaryBlock& block,
        const std::vector<std::pair<uint64_t, ExtMatrix>>& ops, uint32_t ell) {
    hecke::LocalityVerdict v;
    if (!block.local_certified) return v;
    if (block.witness_poly.coeffs.empty()) return v;
    if (block.w_hat_exponent == 0) return v;
    if (block.character.size() != ops.size()) return v;
    for (const auto& op : ops) {
        if (block.character.find(op.first) == block.character.end()) return v;
    }

    const uint32_t D = block.dim;
    ExtMatrix W(D, D, ell);
    if (!block.residue_witness.empty()) {
        for (const auto& term : block.residue_witness) {
            const auto it = std::find_if(ops.begin(), ops.end(),
                    [&](const std::pair<uint64_t, ExtMatrix>& o) { return o.first == term.first; });
            if (it == ops.end()) return v;   // a witness prime outside the family
            const ExtMatrix Asub = hecke::SolveColumnsExact(
                    block.basis_columns, it->second * block.basis_columns,
                    "VerdictFromCertifiedBlock(restriction)");
            W = W + Asub * term.second;
        }
    } else if (block.witness_matrix.rows == D && block.witness_matrix.cols == D) {
        W = block.witness_matrix;           // strict mode's matrix witness
    } else {
        return v;
    }

    v.local = true;
    v.failure = hecke::kLocalityOk;
    v.residue_degree = block.residue_degree;
    v.witness_poly = block.witness_poly;
    v.w_hat_exponent = block.w_hat_exponent;
    v.character = block.character;
    v.witness = block.residue_witness;
    v.analytic = false;
    v.attempts = 0;                          // the loop's, not re-counted here
    v.w_hat = hecke::FrobeniusPower(W, ell, block.w_hat_exponent);
    v.witness_matrix = W;
    return v;
}

// ---------------------------------------------------------------------------
// (slot-7 WP-3, stage 2 item 7; [E] sec:schema (d), pr:eiscusp) Predictions.
// ---------------------------------------------------------------------------
uint32_t PowModU32(uint64_t b, uint64_t e, uint32_t m) { return engine::scalar::powmod<uint32_t>(static_cast<uint32_t>(b % m), e, m); }

PredictionsRecord::Kummer KummerEntry(uint32_t k, uint32_t ell) {
    const KummerVerdict v = kummer_verdict(k, ell);
    PredictionsRecord::Kummer e;
    e.k = k;
    e.applicable = v.applicable;
    e.divides = v.divides;
    e.reason = v.reason;
    e.has_reduced = (v.reason == "direct" || v.reason == "kummer_congruence");
    e.k_reduced = e.has_reduced ? v.k_reduced : 0u;
    e.b_over_k_mod_ell = e.has_reduced ? v.b_over_k_mod_ell : 0u;
    return e;
}

// Prediction pr:eiscusp on one entry.  An entry of an H^1 group (`H1`,
// `H1_plus`, `H1_minus`) whose residue field is F_ell and whose anemic system
// on its own primes equals chi_{a, n+1-a}(q) = q^a + q^{n+1-a} (a in
// [0, min(n/2, ell-2)]) is PREDICTED Eisenstein-cuspidal at the weight
// w == n + 2 - 2a (mod ell - 1), 2 < w <= ell - 1, iff ell | num(B_w / w).
// It is CONFIRMED iff the entry's H^1 module is non-split (m X_{4,m} != 0,
// Loewy length >= 2) and refuted otherwise; `null` when no module was
// formed.  `unpredicted` lists the entry's module groups with Loewy length
// >= 2 that no prediction covers (an eisenstein_cuspidal id covers H^1).
void PredictEntry(IdealRecord& id, const LocalAlgebra& L, const std::string& group,
                  uint32_t n, uint32_t ell, std::vector<uint32_t>& weights) {
    id.predicted.clear();
    id.confirmed.clear();
    id.unpredicted.clear();
    const bool h1 = group.size() >= 2 && group[0] == 'H' && group[1] == '1';
    if (h1 && id.local && L.ok && L.witness_poly.degree() == 1 && ell >= 5) {
        const uint32_t period = ell - 1u;
        // chi_{a, n+1-a} = chi_{n+1-a, a}: a runs over a <= (n+1)/2 only, the
        // pr:eiscusp normalisation (a = 0 is cl:harder's weight n + 2).  The
        // mirror a' = n+1-a would name the weight n+2-2a' == -(n-2a) mod
        // (ell-1), which is not the statement's and, at large ell, not even
        // evaluable by the Bernoulli recursion (O(w^2) with w ~ ell).
        const uint32_t a_max = std::min<uint32_t>(n / 2u, ell - 2u);
        std::vector<uint32_t> ws;
        for (uint32_t a = 0; a <= a_max; ++a) {
            const uint64_t b = (static_cast<uint64_t>(n) + 1u + period * 4u - a) % period;
            bool match = true;
            bool any = false;
            for (size_t j = 0; j < L.primes.size() && match; ++j) {
                const uint64_t q = L.primes[j];
                if (q == ell) continue;
                const std::vector<uint32_t>& c = L.character[j].coeffs;
                const uint32_t lam = c.empty() ? 0u : c[0] % ell;
                const uint32_t chi = (PowModU32(q, a, ell) + PowModU32(q, b, ell)) % ell;
                match = (lam == chi);
                any = true;
            }
            if (!match || !any) continue;
            const uint64_t w0 = ((static_cast<uint64_t>(n) + 2u + 2u * period * 4u) - 2u * a) % period;
            uint32_t w = 0;
            if (w0 == 0u) w = period;
            else if (w0 > 2u) w = static_cast<uint32_t>(w0);
            if (w == 0u || !kummer_divides(w, ell)) continue;
            if (std::find(ws.begin(), ws.end(), w) == ws.end()) ws.push_back(w);
        }
        std::sort(ws.begin(), ws.end());
        int8_t conf = -1;
        for (const ModuleRecordEntry& m : id.modules)
            if (m.group == group) conf = (m.structure.loewy.size() >= 2u) ? 1 : 0;
        for (uint32_t w : ws) {
            id.predicted.push_back(std::string(kPredictionEisensteinCuspidal) + ":k=" +
                                   std::to_string(w));
            id.confirmed.push_back(conf);
            weights.push_back(w);
        }
    }
    for (const ModuleRecordEntry& m : id.modules) {
        if (m.structure.loewy.size() < 2u) continue;
        const bool covered = !id.predicted.empty() && m.group.size() >= 2 && m.group[0] == 'H' &&
                             m.group[1] == '1';
        if (!covered) id.unpredicted.push_back(m.group);
    }
    if (hecke::math_settings().predictions == hecke::PredictionsMode::Assert) {
        for (int8_t c : id.confirmed) {
            if (c == 0) {
                throw std::runtime_error("HeckeExtDriver: predictions = assert: a prediction ("
                                         + id.predicted.front() + ") is refuted");
            }
        }
    }
}

void FinishPredictions(PredictionsRecord& pr, uint32_t n, uint32_t ell,
                       std::vector<uint32_t> weights) {
    weights.push_back(n + 2u);
    std::sort(weights.begin(), weights.end());
    weights.erase(std::unique(weights.begin(), weights.end()), weights.end());
    pr.kummer.clear();
    for (uint32_t k : weights) pr.kummer.push_back(KummerEntry(k, ell));
    pr.glover_json = GloverFactors(n, ell).to_json();
}

} // namespace

// ===========================================================================
// The private state.
// ===========================================================================
struct HeckeExtDriver::Impl {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;
    std::unique_ptr<hecke::HeckeOperatorFamily> family;

    // Per degree, index-aligned with `degrees_`.
    std::vector<std::vector<std::pair<uint64_t, ExtMatrix>>> ops;

    // Per ideal (index-aligned with the record's `ideals`).
    std::vector<LocalAlgebra>    algebras;
    std::vector<SturmCertificate> sturm;

    // The streamed A^{(q)}(T_q) on the WHOLE cohomology, memoised per
    // (degree index, q).  Each block restricts it itself, so without this a
    // cell with b blocks would build the same pipeline b times; with it the
    // pipeline is built ONCE and only its d_q x d_q result is kept (the
    // pipeline itself is still destroyed inside the supplier, so "one
    // pipeline alive at a time" is unchanged).
    std::vector<std::vector<std::pair<uint64_t, ExtMatrix>>> streamed_cache;

    // (slot-7 WP-3) A^{(*)}(T_p) on every degree for the anemic pool, keyed by
    // p ascending of first use; a vector, never a hash container.  Survives
    // resume_u_ell(): the pool excludes ell, so no entry depends on the
    // `u_ell` verdict.
    std::vector<std::pair<uint64_t, std::vector<ExtMatrix>>> pool_ops;

    // (slot-12 FIN) Per ideal (index-aligned with the record's `ideals`): the
    // (d1) certificate, `present` iff the route ran on that ideal.
    std::vector<BoundaryFinalityCertificate> boundary_certs;

    // (slot-7 WP-3, items 4-5) The in-process atlas when consulted; one
    // canonical field per r; and, per unsplit ideal, its degree, own-orbit
    // and label -- what the per-sign ideals are matched against.
    std::shared_ptr<const Atlas> atlas;
    std::map<uint32_t, CanonicalField> fields;
    struct UnsplitLabel {
        uint32_t degree = 0;
        bool has = false;
        std::vector<std::vector<std::vector<uint32_t>>> orbit;
        LabelRecord label;
    };
    std::vector<UnsplitLabel> unsplit_labels;

    const CanonicalField& field(uint32_t ell, uint32_t r) {
        auto it = fields.find(r);
        if (it == fields.end()) it = fields.emplace(r, CanonicalField(ell, r)).first;
        return it->second;
    }

    Impl(uint32_t n_sym, uint32_t ell)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, BuildRelationsHere(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {}
};

// ===========================================================================
// Construction
// ===========================================================================
HeckeExtDriver::HeckeExtDriver(uint32_t n_sym, uint32_t ell, const ExtDriverOptions& opts)
    : n_(n_sym), ell_(ell), opts_(opts) {
    // (slot-8 WP-3) [E] sec:ring (b): the extension record is F_ell-only (surface; slot-9 C5
    // brings Z/ell^k record fields).
    hecke::RequireFieldCoefficientRing("HeckeExtDriver");
    ENGINE_CHECK_MSG(ell >= 2, "HeckeExtDriver: ell must be a prime >= 2");
}

HeckeExtDriver::~HeckeExtDriver() = default;

const HeckeExtRecord& HeckeExtDriver::record() const {
    ENGINE_CHECK_MSG(ran_, "HeckeExtDriver::record(): run() has not been called");
    return rec_;
}

const std::vector<std::pair<uint64_t, ExtMatrix>>&
HeckeExtDriver::cell_operators(uint32_t degree) const {
    ENGINE_CHECK_MSG(ran_, "HeckeExtDriver::cell_operators(): run() has not been called");
    for (size_t i = 0; i < degrees_.size(); ++i) {
        if (degrees_[i] == degree) return impl_->ops[i];
    }
    ENGINE_CHECK_MSG(false, "HeckeExtDriver::cell_operators(): degree was not run");
    return impl_->ops.at(0);
}

const SturmCertificate& HeckeExtDriver::sturm_certificate(uint32_t ideal_index) const {
    ENGINE_CHECK_MSG(ran_ && ideal_index < impl_->sturm.size(),
                     "HeckeExtDriver::sturm_certificate(): out of range");
    return impl_->sturm[ideal_index];
}

const LocalAlgebra& HeckeExtDriver::local_algebra(uint32_t ideal_index) const {
    ENGINE_CHECK_MSG(ran_ && ideal_index < impl_->algebras.size(),
                     "HeckeExtDriver::local_algebra(): out of range");
    return impl_->algebras[ideal_index];
}

// ===========================================================================
// Setup -- one triangulation, one local system, ONE cohomology, ONE family.
// ===========================================================================
void HeckeExtDriver::build_stack_() {
    const Clock::time_point t0 = Clock::now();
    impl_.reset(new Impl(n_, ell_));
    impl_->coh.compute_cohomology();
    dims_ = impl_->coh.get_H_eq_dimensions();
    impl_->family.reset(new hecke::HeckeOperatorFamily(&impl_->coh, impl_->K, impl_->model,
                                                       ell_, n_));
    const Clock::time_point t1 = Clock::now();
    t_.setup_s = Secs(t0, t1);
    if (opts_.trace) {
        std::cout << "EXT_DRIVER setup n=" << n_ << " ell=" << ell_
                  << " dims=";
        for (size_t i = 0; i < dims_.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << dims_[i];
        }
        std::cout << " s=" << std::fixed << std::setprecision(4) << t_.setup_s << std::endl;
    }
}

// ===========================================================================
// (slot-7 WP-3, stage 2 item 3) The cell-local proof_range decision.
// ===========================================================================
const std::vector<ExtMatrix>& HeckeExtDriver::pool_operator_(uint64_t p) {
    ENGINE_CHECK_MSG(p != static_cast<uint64_t>(ell_),
                     "HeckeExtDriver: the anemic pool never contains ell");
    for (const auto& e : impl_->pool_ops) {
        if (e.first == p) return e.second;
    }
    std::vector<ExtMatrix> all;
    if (std::find(s1_.primes.begin(), s1_.primes.end(), p) != s1_.primes.end()) {
        all = impl_->family->T_p(p);          // prepared in Stage 1: O(1)
    } else {
        // Streamed, one pipeline alive at a time, like a Sturm prime.
        // Counted in pool_streamed_, NOT streamed_: `streamed()` keeps its
        // meaning (the Sturm sweep's streamed primes), which
        // HeckeExtRamanujanValidation arm 3 pins.
        all = StreamedOperatorAllDegrees(impl_->coh, impl_->K, impl_->model, ell_, n_, p,
                                         &pool_streamed_);
    }
    impl_->pool_ops.push_back({p, std::move(all)});
    return impl_->pool_ops.back().second;
}

bool HeckeExtDriver::cell_local_proof_range_(uint32_t degree, const ExtMatrix& basis_columns) {
    // No H^0: no Dickson class, so no system to equal -- in range.
    if (dims_.empty() || dims_[0] == 0u) return true;

    const Clock::time_point t0 = Clock::now();
    const uint32_t D = basis_columns.cols;
    const uint32_t d0 = dims_[0];
    std::vector<std::pair<uint64_t, ExtMatrix>> joint;
    for (uint64_t p : AnemicPool(ell_)) {
        const std::vector<ExtMatrix>& A = pool_operator_(p);
        ENGINE_CHECK_MSG(degree < A.size() && A.size() > 0,
                         "HeckeExtDriver: a pool operator lacks the degree");
        ENGINE_CHECK_MSG(A[0].rows == d0 && A[0].cols == d0,
                         "HeckeExtDriver: a pool operator's H^0 block has the wrong shape");
        // T_p|_V by the exact solve -- solvability IS the invariance certificate.
        const ExtMatrix t = hecke::SolveColumnsExact(basis_columns, A[degree] * basis_columns,
                                                     "HeckeExtDriver(proof_range restriction)");
        ExtMatrix J(D + d0, D + d0, ell_);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) J.data[i][j] = t.data[i][j];
        for (uint32_t i = 0; i < d0; ++i)
            for (uint32_t j = 0; j < d0; ++j) J.data[D + i][D + j] = A[0].data[i][j];
        joint.push_back({p, std::move(J)});
    }

    hecke::HeckeModuleDecomposition dec(joint, ell_);
    dec.compute();
    ENGINE_CHECK_MSG(dec.certify() == 0,
                     "HeckeExtDriver: the joint pool decomposition does not certify");
    bool in_range = true;
    for (const hecke::PrimaryBlock& b : dec.blocks()) {
        bool meets_v = false, meets_h0 = false;
        for (uint32_t i = 0; i < b.basis_columns.rows; ++i) {
            for (uint32_t j = 0; j < b.basis_columns.cols; ++j) {
                if (b.basis_columns.data[i][j] == 0u) continue;
                (i < D ? meets_v : meets_h0) = true;
            }
        }
        // A primary component V_m (+) H^0_m meeting both summands is a system
        // the block shares with H^0; an uncertified block that meets both is
        // counted as shared (conservative: never a proof it cannot back).
        if (meets_v && meets_h0) in_range = false;
    }
    t_.proof_range_s += Secs(t0, Clock::now());
    return in_range;
}

// ---------------------------------------------------------------------------
// (slot-7 WP-3, stage 2 item 5) The atlas label.
// ---------------------------------------------------------------------------
std::vector<std::vector<std::vector<uint32_t>>> HeckeExtDriver::own_orbit_(const LocalAlgebra& L) {
    const uint32_t r = static_cast<uint32_t>(L.witness_poly.degree());
    const CanonicalField& F = impl_->field(ell_, r);
    const CanonicalSplit sp = SplitInCanonicalField(F, L.witness_poly, L.primes, L.character);
    ENGINE_CHECK_MSG(sp.ok, ("HeckeExtDriver: the canonical split of a local ideal failed: "
                             + sp.failure).c_str());
    std::vector<std::vector<std::vector<uint32_t>>> orbit = sp.orbit;
    std::sort(orbit.begin(), orbit.end());
    orbit.erase(std::unique(orbit.begin(), orbit.end()), orbit.end());
    return orbit;
}

void HeckeExtDriver::attach_split_(IdealRecord& id, const LocalAlgebra& L) {
    ENGINE_CHECK_MSG(std::is_sorted(L.primes.begin(), L.primes.end()),
                     "HeckeExtDriver: the split's primes must be the ascending keys of P");
    const uint32_t r = static_cast<uint32_t>(L.witness_poly.degree());
    const CanonicalField& F = impl_->field(ell_, r);
    const CanonicalSplit sp = SplitInCanonicalField(F, L.witness_poly, L.primes, L.character);
    id.split_present = true;
    id.split_ok = sp.ok;
    id.split_r = sp.r;
    id.split_fragment = sp.to_json();
    ENGINE_CHECK_MSG(sp.ok, ("HeckeExtDriver: the canonical split of a local ideal failed: "
                             + sp.failure).c_str());
}

LabelRecord HeckeExtDriver::atlas_label_(uint32_t degree, const ExtMatrix& basis_columns) {
    ENGINE_CHECK_MSG(impl_->atlas != nullptr, "HeckeExtDriver: no in-process atlas to label from");
    const AtlasStore& st = impl_->atlas->store;
    const std::vector<uint64_t> T = AnemicPool(ell_);
    ENGINE_CHECK_MSG(st.primes == T, "HeckeExtDriver: the atlas pool is not AnemicPool(ell)");

    // The block's anemic system over T, at the cell: T_q|_V for q in T, then
    // one certified primary decomposition of V over T.
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    for (uint64_t p : T) {
        const std::vector<ExtMatrix>& A = pool_operator_(p);
        ENGINE_CHECK_MSG(degree < A.size(), "HeckeExtDriver: a pool operator lacks the degree");
        ops.push_back({p, hecke::SolveColumnsExact(basis_columns, A[degree] * basis_columns,
                                                   "HeckeExtDriver(label restriction)")});
    }
    hecke::HeckeModuleDecomposition dec(ops, ell_);
    dec.compute();
    ENGINE_CHECK_MSG(dec.certify() == 0, "HeckeExtDriver: the label decomposition does not certify");
    ENGINE_CHECK_MSG(dec.blocks().size() == 1 && dec.blocks()[0].local_certified,
                     "HeckeExtDriver: a local ideal is not T-primary although S_sep is in S_0 "
                     "([E] cl:sep) -- no single anemic system to label");
    const hecke::PrimaryBlock& b = dec.blocks()[0];
    std::vector<hecke::FpPoly> chars;
    for (uint64_t p : T) {
        const auto it = b.character.find(p);
        ENGINE_CHECK_MSG(it != b.character.end(), "HeckeExtDriver: a pool prime has no character");
        chars.push_back(it->second);
    }
    const uint32_t r = static_cast<uint32_t>(b.witness_poly.degree());
    const CanonicalField& F = impl_->field(ell_, r);
    const CanonicalSplit sp = SplitInCanonicalField(F, b.witness_poly, T, chars);
    ENGINE_CHECK_MSG(sp.ok, ("HeckeExtDriver: the label's canonical split failed: " + sp.failure).c_str());
    const std::vector<FieldElt> key = AtlasOrbitKey(F, sp.orbit[0]);
    const int64_t id = AtlasFind(st, r, key);
    ENGINE_CHECK_MSG(id >= 0, "HeckeExtDriver: an anemic system of the cell is not in the "
                              "in-process Sigma(ell) ([E] cl:sigma)");
    const AtlasSystem& sys = st.systems[static_cast<size_t>(id)];

    LabelRecord lb;
    lb.computed = true;
    lb.key = key;
    lb.r = r;
    lb.system = sys.id;
    lb.type = sys.type;
    lb.has_chi = sys.has_chi;
    lb.chi_a = sys.chi_a;
    lb.chi_b = sys.chi_b;
    lb.base = sys.base;
    lb.theta_twist = sys.twist;
    return lb;
}

// ===========================================================================
// Stage 1 -- S_0 and the family's operators.
// ===========================================================================
void HeckeExtDriver::stage_one_() {
    const Clock::time_point t0 = Clock::now();

    // (slot-7 WP-3, stage 2 item 4) Stage 2(3) by default: S_sep(ell) from
    // the in-process atlas where one is built; the caller's list if set.
    ssep_ = SSepRecord();
    std::vector<uint64_t> s_sep;
    const bool atlas_here = opts_.consult_atlas && IsInProcessAtlasEll(ell_);
    if (atlas_here) {
        impl_->atlas = InProcessAtlas(ell_);
        ssep_.computed = true;
        ssep_.primes = impl_->atlas->store.s_sep.primes;
    }
    if (opts_.s_sep.has_value()) s_sep = *opts_.s_sep;
    else if (atlas_here) s_sep = ssep_.primes;

    const std::vector<uint64_t> s0 =
        opts_.good_primes.empty() ? DefaultGoodPrimes() : opts_.good_primes;
    if (s_sep.empty()) {
        s1_ = opts_.good_primes.empty()
                  ? ComputeStageOneOperators(n_, ell_)
                  : ComputeStageOneOperators(n_, ell_, hecke::math_settings().u_ell,
                                             opts_.good_primes);
    } else {
        // (slot-6 WP-9) Stage 2(3): S_0 u S_sep.  ComputeStageOneOperators
        // removes ell, sorts and de-duplicates.
        std::vector<uint64_t> primes = s0;
        for (uint64_t q : s_sep) {
            ENGINE_CHECK_MSG(q >= 2 && q != static_cast<uint64_t>(ell_),
                             "HeckeExtDriver: S_sep holds anemic primes only");
            primes.push_back(q);
        }
        s1_ = ComputeStageOneOperators(n_, ell_, hecke::math_settings().u_ell, primes);
    }
    if (ssep_.computed) {
        // What Stage 2(3) actually added: the run's S_sep primes outside S_0.
        for (uint64_t q : s_sep) {
            if (std::find(s0.begin(), s0.end(), q) == s0.end() &&
                std::find(ssep_.primes.begin(), ssep_.primes.end(), q) != ssep_.primes.end()) {
                ssep_.added.push_back(q);
            }
        }
        std::sort(ssep_.added.begin(), ssep_.added.end());
        ssep_.added.erase(std::unique(ssep_.added.begin(), ssep_.added.end()),
                          ssep_.added.end());
    }
    if (u_ell_history_.empty()) u_ell_history_.push_back(s1_.u_ell.budget_mb);

    // The degrees the record is about.
    degrees_.clear();
    if (!opts_.degrees.empty()) {
        degrees_ = opts_.degrees;
        for (uint32_t q : degrees_) {
            ENGINE_CHECK_MSG(q < dims_.size(), "HeckeExtDriver: degree out of range");
        }
    } else {
        for (uint32_t q = 0; q < dims_.size(); ++q) {
            if (dims_[q] > 0) degrees_.push_back(q);
        }
    }

    // The family's A^{(q)}(T_p) for every p in S_0.  prepare() builds the
    // (p, 1) pipelines concurrently on the engine pool; T_p() is then O(1).
    for (uint64_t p : s1_.primes) impl_->family->prepare(p, 1);

    impl_->ops.assign(degrees_.size(), {});
    for (size_t i = 0; i < degrees_.size(); ++i) {
        const uint32_t q = degrees_[i];
        for (uint64_t p : s1_.primes) {
            const std::vector<ExtMatrix>& A = impl_->family->T_p(p);
            ENGINE_CHECK_MSG(q < A.size(), "HeckeExtDriver: the family has no such degree");
            impl_->ops[i].push_back({p, A[q]});
        }
    }
    // (slot-24 G6, HeckeOperatorOptions (6)(b)) From here on the cell reads
    // only the family's A^{(q)} (T_p() again: the anemic pool, the Sturm
    // sweep's reuse, a resume's Stage 1), so its level (a) and tables -- the
    // fused blocks, Theta_eq (dim C_eq^2 per degree) and the tau / coset
    // tables, ~0.65 GB per prime at n = 1000 -- are released.  Memory only.
    //
    // (slot-25 PERF-VF (b2)) A size floor: the release only pays where there is memory to free.
    // Below kFieldDietReleaseMinDimCeq (max_q dim C^q_eq) the level (a) of the cell is small, and
    // the release + malloc_trim(0) + the re-faulted heap of a later repeat cost 3-7 % of a
    // 0.15-0.3 s cell (slot24/CLOSE_H2H_diagnosis.md Sec. 3).  Memory and speed only.
    uint32_t max_dim_C_eq = 0;
    for (uint32_t q = 0; q < dims_.size(); ++q) max_dim_C_eq = std::max(max_dim_C_eq, impl_->coh.C_eq_dimension(q));
    const uint32_t release_floor = field_diet_release_min_dim_ceq();   // kFieldDietReleaseMinDimCeq unless a validation scope
    if (std::getenv("ENGINE_TRACE_RSS") != nullptr)   // diagnostic only
        std::cout << "      [ext] max dim C_eq=" << max_dim_C_eq << " (release floor " << release_floor << ")" << std::endl;
    if (hecke::HeckeOperatorOptions::field_diet() && max_dim_C_eq >= release_floor) {
        const uint32_t released = impl_->family->release_level_a();
        g_stage_one_releases.fetch_add(1, std::memory_order_relaxed);
        hecke::HeckeOperatorOptions::release_free_heap();
        if (std::getenv("ENGINE_TRACE_RSS") != nullptr)   // diagnostic only
            std::cout << "      [ext] family level (a) released: " << released << " pipelines, VmRSS=" << std::fixed
                      << std::setprecision(1) << oc_util::CurrentRssMiB() << " MB" << std::endl;
    }

    const Clock::time_point t1 = Clock::now();
    t_.stage1_s += Secs(t0, t1);
    if (opts_.trace) {
        std::cout << "EXT_DRIVER stage1 primes=";
        for (size_t i = 0; i < s1_.primes.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << s1_.primes[i];
        }
        std::cout << " u_ell=" << (s1_.u_ell.included ? "in" : to_string(s1_.u_ell.reason))
                  << " s=" << std::fixed << std::setprecision(4) << t_.stage1_s << std::endl;
    }
}

// ===========================================================================
// Stages 2-5 and the record.
// ===========================================================================
void HeckeExtDriver::stages_two_to_five_() {
    const hecke::MathSettings settings = hecke::math_settings();

    rec_ = HeckeExtRecord();
    rec_.n_sym = n_;
    rec_.ell = ell_;
    rec_.weight = n_ + 2;
    rec_.settings = settings;
    rec_.bulk.available = true;
    rec_.bulk.num_degrees = static_cast<uint32_t>(dims_.size());
    for (uint32_t q = 0; q < dims_.size(); ++q) {
        rec_.groups.push_back({GroupName(q), true, dims_[q]});
    }
    rec_.dim_dickson = hecke::DicksonInvariantCount(n_, ell_);
    rec_.operators_computed = s1_.primes;
    rec_.anemic_pool = AnemicPool(ell_);   // (slot-7 WP-3) always present
    rec_.s_sep = ssep_;                    // (slot-7 WP-3, item 4)

    // v2: the U_ell admission is a property of (n, ell) and of the budget, so
    // it is the RECORD's object and is emitted once, with ONE `resumable`
    // ([OD] slot-4 close-out 5).  In v1 it sat on every IdealRecord and
    // `resumable_items()` multiplied one refusal by the ideal count.
    rec_.u_ell.present = true;
    rec_.u_ell.included = s1_.u_ell.included;
    rec_.u_ell.reason = to_string(s1_.u_ell.reason);
    rec_.u_ell.predicted_mb = s1_.u_ell.predicted_mb;
    rec_.u_ell.budget_mb = s1_.u_ell.budget_mb;
    rec_.u_ell.model_version = s1_.u_ell.model_version;
    if (!s1_.u_ell.included) {
        // [X] Sec. 1.3.1 R3.3 item 4: the decision is resumable.
        ResumableRecord& r = rec_.u_ell.resumable;
        r.present = true;
        r.kind = kKindUEll;
        r.complete = false;
        std::ostringstream b;
        b << "{\"budget_mb\":" << s1_.u_ell.budget_mb << "}";
        r.budget_json = b.str();
        std::ostringstream reached;
        reached << "{\"predicted_mb\":" << std::fixed << std::setprecision(2)
                << s1_.u_ell.predicted_mb << ",\"model_version\":\""
                << s1_.u_ell.model_version << "\",\"reason\":\""
                << to_string(s1_.u_ell.reason) << "\"}";
        r.reached_json = reached.str();
        std::ostringstream hist;
        hist << "[";
        for (size_t i = 0; i < u_ell_history_.size(); ++i) {
            if (i) hist << ",";
            hist << u_ell_history_[i];
        }
        hist << "]";
        r.budget_history_json = hist.str();
    }
    rec_.certificates_ran.push_back("locality");
    rec_.certificates_ran.push_back("cyclic");
    if (opts_.run_commutant) rec_.certificates_ran.push_back("commutant");
    if (opts_.run_sturm) rec_.certificates_ran.push_back("sturm");

    impl_->algebras.clear();
    impl_->unsplit_labels.clear();
    // (slot-7 WP-3, item 5) Labels only where the atlas was consulted AND the
    // run's operators contain S_sep(ell) -- then every ideal is T-primary.
    bool labels_on = (impl_->atlas != nullptr);
    for (uint64_t q : ssep_.primes) {
        if (std::find(s1_.primes.begin(), s1_.primes.end(), q) == s1_.primes.end()) labels_on = false;
    }
    impl_->sturm.clear();
    impl_->boundary_certs.clear();
    impl_->streamed_cache.assign(degrees_.size(), {});
    cross_group_matches_ = 0;

    uint32_t next_id = 0;
    for (size_t gi = 0; gi < degrees_.size(); ++gi) {
        const uint32_t q = degrees_[gi];
        const std::vector<std::pair<uint64_t, ExtMatrix>>& ops = impl_->ops[gi];
        if (ops.empty() || ops[0].second.rows == 0) continue;

        // --- Stage 2 -----------------------------------------------------
        const Clock::time_point s2a = Clock::now();
        hecke::HeckeModuleDecomposition decomp(ops, ell_);
        decomp.compute();
        const uint32_t cert_code = decomp.certify();
        const Clock::time_point s2b = Clock::now();
        t_.stage2_s += Secs(s2a, s2b);
        ENGINE_CHECK_MSG(cert_code == 0,
                         "HeckeExtDriver: the Stage-2 decomposition does not certify");

        for (const hecke::PrimaryBlock& block : decomp.blocks()) {
            IdealRecord id;
            id.id = next_id++;
            id.dim = block.dim;
            id.residue_degree = block.residue_degree;
            id.residue_degree_certified = block.residue_degree_certified;
            id.local = block.local_certified;
            id.residue_witness = block.residue_witness;
            if (block.phase_reached != hecke::LocalityPhase::Linear) {
                id.has_phase_reached = true;
                id.phase_reached = hecke::to_string(block.phase_reached);
            }

            // --- Stage 3 -------------------------------------------------
            const Clock::time_point s3a = Clock::now();
            LocalAlgebraOptions lopts;
            hecke::LocalityVerdict adopted;
            if (opts_.consume_block_locality) {
                adopted = VerdictFromCertifiedBlock(block, ops, ell_);
                if (adopted.local) lopts.precomputed_locality = &adopted;
            }
            const LocalAlgebra L = ComputeLocalAlgebra(block, ops, ell_, lopts);
            const Clock::time_point s3b = Clock::now();
            t_.stage3_s += Secs(s3a, s3b);

            if (!L.ok) {
                // A block the budget left undecided ([X] R3.3 item 2): emitted
                // flagged, never withheld.
                id.final_inputs.local = false;
                id.final_inputs.locality_complete = false;
                id.final = is_final(id.final_inputs, settings);
                id.certificate = "unavailable";
                rec_.ideals.push_back(id);
                impl_->algebras.push_back(L);
                impl_->sturm.push_back(SturmCertificate());
                impl_->boundary_certs.push_back(BoundaryFinalityCertificate());
                impl_->unsplit_labels.push_back(Impl::UnsplitLabel());
                continue;
            }

            // Stage 2's cross-group evaluation test ([X] Sec. 2.2 Stage 2,
            // [XA M5]).  v1 kept it as a driver diagnostic because [E] Sec. 7
            // had no key for it; v2 REPORTS it, per ambient group, with the
            // witness the test used ([E] sec:schema v1.5.0's second booked
            // repair, slot-5 plan Sec. 2.3 item 3).
            //
            // The entry for this block's OWN group is always present and is
            // the identity match: it is what makes `groups` a list a reader
            // can compare across entries.
            {
                IdealGroupMatch own;
                own.group = GroupName(q);
                own.own = true;
                own.matched = true;
                own.ideal_id = id.id;
                own.witness_poly = L.witness_poly.coeffs;
                own.w_hat_exponent = L.N;
                own.w_hat_exponent_other = L.N;
                id.groups.push_back(own);
            }
            for (size_t pi = 0; pi < impl_->algebras.size(); ++pi) {
                const LocalAlgebra& prev = impl_->algebras[pi];
                if (!prev.ok || prev.primes != L.primes) continue;
                const EvaluationMatch m = SameMaximalIdeal(prev, L.t);
                if (m.same_ideal) ++cross_group_matches_;
                IdealGroupMatch g;
                g.group = rec_.ideals[pi].groups.empty() ? std::string()
                                                         : rec_.ideals[pi].groups[0].group;
                g.own = false;
                g.matched = m.same_ideal;
                g.ideal_id = rec_.ideals[pi].id;
                g.witness_poly = prev.witness_poly.coeffs;
                g.w_hat_exponent = prev.N;
                g.w_hat_exponent_other = m.N_other;
                g.failure = m.failure;
                if (!g.group.empty()) id.groups.push_back(g);
            }

            id.witness_poly = L.witness_poly.coeffs;
            id.coefficient_field.present = true;
            id.coefficient_field.w_hat_exponent = L.N;
            for (size_t i = 0; i < L.primes.size(); ++i) {
                id.coefficient_field.primes.push_back(L.primes[i]);
                id.coefficient_field.P.push_back(L.character[i].coeffs);
                id.tangent_primes.push_back(L.primes[i]);
                id.tangent_active.push_back(L.active[i]);
                id.operator_primes.push_back(L.primes[i]);
                id.t_p.push_back(L.t[i]);
            }
            id.algebra_present = true;
            id.algebra_dim = L.dim_A;
            id.embdim = L.embdim;
            id.nilpotency = L.nu;
            id.gorenstein = L.gorenstein;
            id.hilbert = L.hilbert;

            // (slot-7 WP-3, stage 2 item 5) the atlas label.
            Impl::UnsplitLabel ul;
            ul.degree = q;
            if (labels_on) {
                id.label = atlas_label_(q, block.basis_columns);
                ul.has = true;
                ul.orbit = own_orbit_(L);
                ul.label = id.label;
            }

            // --- Stage 4 -------------------------------------------------
            const Clock::time_point s4a = Clock::now();
            id.certificate_routes.a = L.cyclicity.cyclic ? "cyclic" : "not_cyclic";
            id.final_inputs.cyclic_passed = L.cyclicity.cyclic;

            CommutantOutcome b = CommutantOutcome::InProgress;
            if (opts_.run_commutant && !L.cyclicity.cyclic) {
                CommutantRoute cr(L, /*budget=*/0, /*force_naive=*/false);
                b = cr.run();
                id.certificate_routes.b = ToString(b);
                if (b == CommutantOutcome::OverBudget) {
                    ResumableRecord r;
                    r.present = true;
                    r.kind = kKindCommutant;
                    r.complete = false;
                    id.resumables.push_back(r);
                }
            }
            id.final_inputs.commutant = b;

            SturmCertificate sc;
            SturmSweepOutcome c = SturmSweepOutcome::InProgress;
            if (opts_.run_sturm && !L.cyclicity.cyclic &&
                b != CommutantOutcome::Passed) {
                StreamedOperatorStats before = streamed_;
                BlockRestrictionSource::Supplier reuse = FamilyOperatorSupplier(
                        *impl_->family, q, ell_, s1_.u_ell.included);
                BlockRestrictionSource::Supplier sup = reuse;
                if (opts_.allow_streaming) {
                    BlockRestrictionSource::Supplier raw = StreamedOperatorSupplier(
                            impl_->coh, impl_->K, impl_->model, ell_, n_, q,
                            s1_.u_ell.included, &streamed_);
                    std::vector<std::pair<uint64_t, ExtMatrix>>* memo =
                            &impl_->streamed_cache[gi];
                    BlockRestrictionSource::Supplier cached =
                            [raw, memo](uint64_t qq, ExtMatrix& out) -> bool {
                        for (const auto& e : *memo) {
                            if (e.first == qq) { out = e.second; return true; }
                        }
                        if (!raw(qq, out)) return false;
                        memo->push_back({qq, out});
                        return true;
                    };
                    sup = ReuseBeforeStreamingSupplier(s1_.primes, reuse, cached);
                }
                BlockRestrictionSource src(block.basis_columns, sup);
                SturmSweepOptions so = SturmSweepOptions::from_settings(settings, n_ + 2);
                // (slot-7 WP-3, stage 2 item 3) The range the sweep RUNS with:
                // the caller's, or decided at the cell -- never a flag flipped
                // after the sweep (slot-6 WP-9 finding 4).
                const bool caller = opts_.proof_range.has_value();
                const bool decided =
                        caller ? *opts_.proof_range
                               : cell_local_proof_range_(q, block.basis_columns);
                so.proof_range = decided;
                const Clock::time_point sb0 = Clock::now();
                SturmSweep sw(L, n_, src, so);
                c = sw.run();
                const Clock::time_point sb1 = Clock::now();
                t_.sturm_sweep_s += Secs(sb0, sb1);
                t_.streamed_s += streamed_.wall_s - before.wall_s;
                sc = sw.certificate();
                id.certificate_routes.c =
                        sc.sturm_off ? "off"
                                     : (sc.outcome == SturmSweepOutcome::NotApplicable
                                                ? "not_applicable"
                                                : (sc.has_B_prime ? "found" : "none_at_cap"));
                id.generators_certified = sc.generators_certified;
                id.sturm.present = true;
                id.sturm.has_B_prime = sc.has_B_prime;
                id.sturm.B_prime = sc.B_prime;
                id.sturm.cap = sc.cap;
                id.sturm.closure_dim = sc.closure_dim;
                id.sturm.dim_M_k = sc.dim_M_k;
                id.sturm.mode = ToString(sc.mode);
                id.sturm.outcome = ToString(sc.outcome);
                id.sturm.sturm_primes = sc.sturm_primes;
                // [E] sec:schema (a): caller_input | cell_local | method_undefined
                // (no recorded B' -- then null, and `final` is governed by
                // `sturm_unavailable` as before: the verdict input stays true).
                if (caller) {
                    id.sturm.proof_range_source = kProofRangeCallerInput;
                    id.sturm.has_proof_range = true;
                    id.sturm.proof_range = sc.proof_range;
                } else if (sc.has_B_prime) {
                    id.sturm.proof_range_source = kProofRangeCellLocal;
                    id.sturm.has_proof_range = true;
                    id.sturm.proof_range = sc.proof_range;
                } else {
                    id.sturm.proof_range_source = kProofRangeMethodUndefined;
                    id.sturm.has_proof_range = false;
                    id.sturm.proof_range = true;
                }
                if (rec_.dim_M_k == 0) rec_.dim_M_k = sc.dim_M_k;
                // The bound's own wall, the <= 0.1 s gate of the plan's
                // acceptance (d): it is inside the sweep's first run().
                t_.sturm_bound_s += sw.bound_wall_s();
            }
            id.final_inputs.sturm = c;
            id.final_inputs.sturm_proof_range =
                    id.sturm.present ? id.sturm.proof_range : sc.proof_range;
            id.final_inputs.sturm_has_B_prime = sc.has_B_prime;
            const Clock::time_point s4b = Clock::now();
            t_.stage4_s += Secs(s4a, s4b);

            // The certificate the record NAMES is the route that carried it.
            const GenerationVerdict gv =
                    opts_.run_sturm ? generation_verdict_with_sweep(L.cyclicity.cyclic, b, sc)
                                    : generation_verdict(L.cyclicity.cyclic, b,
                                                         SturmRouteStatus::Unavailable);
            id.certificate = ToString(gv.kind);

            // --- (slot-12 FIN) the finality certificates (d0) / (d1) -------
            // [E] sec:certgen cert:dickson / cert:boundary; run ONLY when
            // (a)-(c) did not pass; a failure falls through to the existing
            // reasons (WP-12 proposal v2 Sec. 5 item 2).
            BoundaryFinalityCertificate bcert;
            {
                const Clock::time_point f0 = Clock::now();
                FinalInputs probe = id.final_inputs;
                probe.local = true;
                probe.locality_complete = true;
                probe.certificates_ok = true;
                const bool abc = is_final(probe, settings).value == Final::True;
                const bool hyp = opts_.run_finality && !abc && ell_ >= 5 && n_ >= 1 && n_ % 2 == 0;
                const bool ell_in_S = s1_.u_ell.included;
                if (hyp && q == 0) {
                    const std::string v = ell_in_S ? DicksonRoute(n_, ell_, L.primes, L.t)
                                                   : std::string(kRouteNotApplicableUEll);
                    ENGINE_CHECK_MSG(IsDicksonRouteValue(v),
                                     ("HeckeExtDriver: certificate (d0) contradicts [C] "
                                      "prop:heckeH0(1): " + v).c_str());
                    id.certificate_routes.d0 = v;
                    id.final_inputs.dickson_passed = (v == kRoutePassed);
                } else if (hyp && q == 1) {
                    ExtMatrix r1;
                    if (ell_in_S) {
                        ensure_pair_();
                        const std::vector<FlatMatrix>& rmaps =
                                pair_->restriction().get_induced_H_eq_maps();
                        r1 = (rmaps.size() > 1 && rmaps[1].get_rows() > 0)
                                     ? FlatAsExt(rmaps[1])
                                     : ExtMatrix(0, block.basis_columns.rows, ell_);
                    }
                    const std::vector<std::pair<uint64_t, ExtMatrix>>& cell_ops = impl_->ops[gi];
                    auto ambient = [&](uint64_t p) -> ExtMatrix {
                        for (const auto& e : cell_ops)
                            if (e.first == p) return e.second;
                        return pool_operator_(p)[q];
                    };
                    bcert = BoundaryRoute(n_, ell_, ell_in_S, s1_.u_ell.budget_mb,
                                          block.basis_columns, r1, L.t,
                                          L.coords, ambient);
                    const uint32_t bv = bcert.verify();
                    ENGINE_CHECK_MSG(bv == 0, ("HeckeExtDriver: certificate (d1) does not verify "
                                               "(code " + std::to_string(bv) + ")").c_str());
                    id.certificate_routes.d1 = bcert.outcome;
                    id.final_inputs.boundary_passed = (bcert.outcome == kRoutePassed);
                    if (bcert.outcome == kRoutePassed ||
                        bcert.outcome.compare(0, 7, kRouteFailedPrefix) == 0) {
                        id.boundary_reps = bcert.reps;
                    }
                    // (slot-13 WP-7, hecke_ext/v5) (d1)'s data, on a passed
                    // ideal only ([E] v1.14.0-pre.1 cert:boundary).
                    if (bcert.outcome == kRoutePassed) {
                        id.boundary_data.present = true;
                        id.boundary_data.rank_r1 = bcert.rank_r1;
                        id.boundary_data.r1 = bcert.r1 * bcert.V;
                        id.boundary_data.t = bcert.reps_restricted;
                    }
                }
                if (id.final_inputs.dickson_passed) id.certificate = "dickson";
                if (id.final_inputs.boundary_passed) id.certificate = "boundary";
                t_.finality_s += Secs(f0, Clock::now());
            }

            // --- Stage 5 -------------------------------------------------
            const Clock::time_point s5a = Clock::now();
            const ModuleStructure ms = ComputeModuleStructure(L);
            const Clock::time_point s5b = Clock::now();
            t_.stage5_s += Secs(s5a, s5b);
            if (ms.ok) id.modules.push_back({GroupName(q), ms});

            // --- the verdict ---------------------------------------------
            id.final_inputs.local = true;
            id.final_inputs.locality_complete = true;
            id.final_inputs.certificates_ok = true;
            id.final = is_final(id.final_inputs, settings);

            rec_.ideals.push_back(id);
            impl_->algebras.push_back(L);
            impl_->sturm.push_back(sc);
            impl_->boundary_certs.push_back(std::move(bcert));
            impl_->unsplit_labels.push_back(ul);
        }
    }

    // (slot-7 WP-3, stage 2 item 6) The canonical split, the record's
    // default report form ([E] sec:schema (c)); timed as canonical_field_s.
    if (settings.split_field_report == hecke::SplitFieldReport::Canonical) {
        const Clock::time_point c0 = Clock::now();
        for (size_t i = 0; i < rec_.ideals.size(); ++i) {
            IdealRecord& id = rec_.ideals[i];
            const LocalAlgebra& L = impl_->algebras[i];
            if (!id.local || !id.coefficient_field.present || !L.ok || L.witness_poly.is_zero())
                continue;
            attach_split_(id, L);
        }
        t_.canonical_field_s += Secs(c0, Clock::now());
    }

    // (slot-7 WP-3, stage 2 item 7) the predictions ([E] sec:schema (d)).
    {
        std::vector<uint32_t> weights;
        for (size_t i = 0; i < rec_.ideals.size(); ++i) {
            IdealRecord& id = rec_.ideals[i];
            const std::string g = id.groups.empty() ? std::string() : id.groups[0].group;
            PredictEntry(id, impl_->algebras[i], g, n_, ell_, weights);
        }
        FinishPredictions(rec_.predictions, n_, ell_, weights);
    }

    const Clock::time_point r0 = Clock::now();
    if (rec_.dim_M_k == 0 && !rec_.ideals.empty()) {
        // Every ideal was cyclic, so no sweep ran and dim M_k was never formed.
        rec_.dim_M_k = 0;
    }
    ENGINE_CHECK_MSG(rec_.verify() == 0, "HeckeExtDriver: the record does not verify");
    const Clock::time_point r1 = Clock::now();
    t_.record_s += Secs(r0, r1);

    if (opts_.trace) {
        std::cout << "EXT_DRIVER stages2-5 ideals=" << rec_.ideals.size()
                  << " cross_group=" << cross_group_matches_
                  << " s2=" << std::fixed << std::setprecision(4) << t_.stage2_s
                  << " s3=" << t_.stage3_s << " s4=" << t_.stage4_s
                  << " s5=" << t_.stage5_s << " bound=" << t_.sturm_bound_s
                  << " streamed=" << t_.streamed_s << std::endl;
    }
}

// ===========================================================================
// The `hecke_ext/v2` attachments (slot-5 WP-1).
//
// Both builders run AFTER stages_two_to_five_() and only ADD to the record:
// nothing below touches a field the unsplit run already wrote, so a run with
// both flags off is byte-identical to a slot-4 one.
// ===========================================================================

const EquivariantMappingCone* HeckeExtDriver::cone() const { return cone_.get(); }
const hecke::SL2ZBoundaryPair* HeckeExtDriver::boundary_pair() const { return pair_.get(); }
const hecke::HeckeInvolution* HeckeExtDriver::involution() const { return involution_.get(); }

const HeckeSignSplit* HeckeExtDriver::sign_split(uint32_t degree) const {
    for (size_t i = 0; i < degrees_.size() && i < splits_.size(); ++i) {
        if (degrees_[i] == degree) return splits_[i].get();
    }
    return nullptr;
}

const HeckeExtRecord& HeckeExtDriver::sign_record(bool plus) const {
    ENGINE_CHECK_MSG(sign_records_built_,
                     "HeckeExtDriver::sign_record(): no per-sign record was built");
    return plus ? plus_record_ : minus_record_;
}

void HeckeExtDriver::attach_cone_action(const hecke::HeckeConeAction& action) {
    ENGINE_CHECK_MSG(ran_, "HeckeExtDriver::attach_cone_action(): run() has not been called");
    ENGINE_CHECK_MSG(rec_.pair.available,
                     "HeckeExtDriver::attach_cone_action(): the cone flags belong inside an "
                     "available `pair` object -- run with `with_cone`");
    ENGINE_CHECK_MSG(action.computed(),
                     "HeckeExtDriver::attach_cone_action(): the action is not computed");
    ENGINE_CHECK_MSG(&action.cone() == cone_.get(),
                     "HeckeExtDriver::attach_cone_action(): the action is about another cone");
    AttachConeActionRecord(rec_.pair, action);
    ENGINE_CHECK_MSG(rec_.verify() == 0,
                     "HeckeExtDriver: the record does not verify after the cone action");
}

// ---------------------------------------------------------------------------
// with_cone -- E4b's boundary complex and E5's pair, from ONE cone.
// ---------------------------------------------------------------------------
// (slot-12 FIN) The boundary pair alone (restriction r^q and the circle's
// cohomology), on the driver's own cohomology: what (d1) needs, built at most
// once and shared with build_cone_().
void HeckeExtDriver::ensure_pair_() {
    if (pair_) return;
    pair_.reset(new hecke::SL2ZBoundaryPair(&impl_->coh, impl_->K));
    if (std::getenv("ENGINE_TRACE_RSS") != nullptr)   // (slot-24 G6) diagnostic only
        std::cout << "      [ext] boundary pair built, VmRSS=" << std::fixed << std::setprecision(1)
                  << oc_util::CurrentRssMiB() << " MB (heap in use " << hecke::HeckeOperatorOptions::heap_in_use_mib()
                  << " MB)" << std::endl;
}

const BoundaryFinalityCertificate* HeckeExtDriver::boundary_certificate(uint32_t ideal_index) const {
    ENGINE_CHECK_MSG(ran_, "HeckeExtDriver::boundary_certificate(): run() has not been called");
    ENGINE_CHECK_MSG(ideal_index < impl_->boundary_certs.size(),
                     "HeckeExtDriver::boundary_certificate(): ideal index out of range");
    const BoundaryFinalityCertificate& c = impl_->boundary_certs[ideal_index];
    return c.present ? &c : nullptr;
}

void HeckeExtDriver::build_cone_() {
    const Clock::time_point t0 = Clock::now();
    // The pair is built on the driver's OWN cohomology, so the boundary side
    // and the restriction are the ones every other object of this record is
    // about.  Both outlive the cone, which holds them by pointer.
    // (slot-12 FIN) (d1) may already have built the pair (restriction only).
    ensure_pair_();
    cone_ = pair_->make_cone();

    rec_.boundary = BoundaryComplexRecord(*cone_);
    rec_.pair = ConeComplexRecord(*cone_);
    for (const GroupRecord& g : BoundaryGroups(*cone_)) rec_.groups.push_back(g);
    for (const GroupRecord& g : ConeGroups(*cone_)) rec_.groups.push_back(g);

    const LesSummary les = ConeLesSummary(*cone_);
    rec_.les_available = les.available;
    rec_.les_exact = les.exact;
    // `hecke_equivariant` is the ACTION's statement, not the cone's: the cone
    // knows nothing about T_p.  It stays false until an action is attached,
    // and a false here is "not claimed", never "refuted" (rule A6).
    rec_.les_hecke_equivariant = les.hecke_equivariant;
    rec_.les_ranks = les.ranks;

    uint32_t shriek = 0;
    if (ConeDimH1Shriek(*cone_, &shriek)) {
        rec_.has_dim_H1_shriek = true;
        rec_.dim_H1_shriek = shriek;
    }
    // (slot-24 G6) Every record object of the cone is attached above.
    if (opts_.release_cone_cochains) {
        cone_->release_cochain_objects();
        hecke::HeckeOperatorOptions::release_free_heap();
        if (std::getenv("ENGINE_TRACE_RSS") != nullptr)   // diagnostic only
            std::cout << "      [ext] cone cochain objects released, VmRSS=" << std::fixed << std::setprecision(1)
                      << oc_util::CurrentRssMiB() << " MB" << std::endl;
    }
    t_.cone_s = Secs(t0, Clock::now());
    if (opts_.trace) {
        std::cout << "EXT_DRIVER cone q_max=" << cone_->q_max()
                  << " dim_Hc=";
        for (uint32_t q = 0; q <= cone_->q_max(); ++q) {
            if (q) std::cout << ",";
            std::cout << cone_->dim_Hc(q);
        }
        std::cout << " les_maps=" << rec_.pair.les_maps.size()
                  << " s=" << std::fixed << std::setprecision(4) << t_.cone_s << std::endl;
    }
}

// ---------------------------------------------------------------------------
// with_sign_split -- E3's `iota` objects and the two per-sign records.
// ---------------------------------------------------------------------------
namespace {

// (slot-23 P4I) ENGINE_TRACE_RSS=1: the iota / sign-split phase's own marks
// (process VmRSS / VmHWM, MB).  Measurement output only.
void IotaPhaseTrace(const std::string& what) {
    if (std::getenv("ENGINE_TRACE_RSS") == nullptr) return;
    std::cout << "      [iota] driver " << what << ": VmRSS=" << std::fixed << std::setprecision(1)
              << oc_util::ProcStatusMiB("VmRSS:") << " MB VmHWM=" << oc_util::ProcStatusMiB("VmHWM:") << " MB"
              << std::endl;
}

ExtMatrix ToExt(const FlatMatrix& M) {
    ExtMatrix out(M.get_rows(), M.get_cols(), M.get_p());
    for (uint32_t i = 0; i < M.get_rows(); ++i) {
        for (uint32_t j = 0; j < M.get_cols(); ++j) out.data[i][j] = M.get_entry(i, j);
    }
    return out;
}

// The A^{(q)}(T_p) restricted to one sign, in the order HeckeSignSplit stored
// them -- the per-sign family Stage 2 runs on.
std::vector<std::pair<uint64_t, ExtMatrix>> SignOperators(const HeckeSignSplit& split,
                                                          bool plus) {
    std::vector<std::pair<uint64_t, ExtMatrix>> out;
    for (size_t i = 0; i < split.num_operators(); ++i) {
        const SignSplitOperator& op = split.op(i);
        if (!op.restricted) continue;
        out.push_back({op.p, plus ? op.A_plus : op.A_minus});
    }
    return out;
}

} // namespace

void HeckeExtDriver::build_sign_split_() {
    const Clock::time_point t0 = Clock::now();
    const hecke::MathSettings settings = hecke::math_settings();

    // The PRODUCTION involution ([C] `def:iota`(iv), iota^* = -iota^*_geo):
    // iota_K is PROPOSED by the exact QuadPoint match and CERTIFIED against
    // [C] `lem:iotamodel` inside BuildIotaSimplicialMapOnModel; compute() then
    // discharges obl:iota(i)-(iii) and ABORTS on failure.
    const hecke::IotaSimplicialMap iota_K =
            hecke::BuildIotaSimplicialMapOnModel(impl_->K, impl_->model);
    involution_.reset(new hecke::HeckeInvolution(&impl_->coh, iota_K, "hecke_ext/v2"));
    involution_->set_exact_relation_data(
            GenerateSL2ZRelationsWithGroupData(impl_->K, n_, ell_));
    IotaPhaseTrace("sign split enter");
    involution_->compute();
    IotaPhaseTrace("involution computed");

    // [C] `obl:iota`(ii)'s RIDER ([OD] 2026-09-22 Q3): the record -- not only
    // the in-memory certificate -- says whether the exact Mat22 comparison
    // happened and on which relation rows.  Attached as soon as compute()
    // returns, i.e. beside `iota` and independently of whether any degree
    // splits, so that a cell whose every degree is `not_applicable` still
    // carries it.
    rec_.iota_exact = IotaExactComparison(*involution_);

    splits_.clear();
    splits_.resize(degrees_.size());

    plus_record_ = HeckeExtRecord();
    minus_record_ = HeckeExtRecord();
    for (int side = 0; side < 2; ++side) {
        HeckeExtRecord& r = (side == 0) ? plus_record_ : minus_record_;
        r.n_sym = n_;
        r.ell = ell_;
        r.weight = n_ + 2;
        r.sign = (side == 0) ? "plus" : "minus";
        r.settings = settings;
        r.bulk = rec_.bulk;
        r.dim_M_k = rec_.dim_M_k;
        r.dim_S_k = rec_.dim_S_k;
        r.dim_dickson = rec_.dim_dickson;
        r.operators_computed = rec_.operators_computed;
        r.anemic_pool = rec_.anemic_pool;
        r.s_sep = rec_.s_sep;
        r.predictions = rec_.predictions;
        r.u_ell = rec_.u_ell;
        r.certificates_ran.push_back("locality");
        r.certificates_ran.push_back("cyclic");
    }

    uint32_t next_plus = 0, next_minus = 0;
    for (size_t gi = 0; gi < degrees_.size(); ++gi) {
        const uint32_t q = degrees_[gi];
        if (dims_[q] == 0u) continue;
        splits_[gi].reset(new HeckeSignSplit(ell_, n_, q,
                                             ToExt(involution_->matrix(q)),
                                             ToExt(involution_->H_plus_basis(q)),
                                             ToExt(involution_->H_minus_basis(q)),
                                             impl_->ops[gi]));
        HeckeSignSplit& split = *splits_[gi];
        split.compute();
        IotaPhaseTrace("sign split q=" + std::to_string(q) + " computed");

        rec_.iota.push_back(IotaRecord(split));
        for (const GroupRecord& g : SignGroups(split)) rec_.groups.push_back(g);

        if (split.certificate().status != SignSplitStatus::Split) continue;
        for (int side = 0; side < 2; ++side) {
            const bool plus = (side == 0);
            HeckeExtRecord& r = plus ? plus_record_ : minus_record_;
            uint32_t& next_id = plus ? next_plus : next_minus;
            const std::vector<std::pair<uint64_t, ExtMatrix>> ops = SignOperators(split, plus);
            const uint32_t dim = plus ? split.dim_plus() : split.dim_minus();
            const std::string group = SignedGroupName(q, plus);
            r.groups.push_back({group, true, dim});
            if (dim == 0u || ops.empty()) continue;

            const hecke::HeckeModuleDecomposition* dec =
                    split.decomposition(plus ? SignSide::Plus : SignSide::Minus);
            if (dec == nullptr) continue;
            for (const hecke::PrimaryBlock& block : dec->blocks()) {
                IdealRecord id;
                id.id = next_id++;
                id.sign = r.sign;
                id.dim = block.dim;
                id.residue_degree = block.residue_degree;
                id.residue_degree_certified = block.residue_degree_certified;
                id.local = block.local_certified;
                id.residue_witness = block.residue_witness;

                // Stage 3 and its route (a).  Routes (b) and (c) are NOT run
                // per sign: certificate (c) is a statement about the Sturm
                // algebra on the unsplit M_k, so a per-sign sweep would be a
                // claim [E] `cert:sturm` does not make.  They record
                // "not_run", which the reader can tell from "failed".
                LocalAlgebraOptions lopts;
                const LocalAlgebra L = ComputeLocalAlgebra(block, ops, ell_, lopts);
                if (!L.ok) {
                    id.final_inputs.local = false;
                    id.final_inputs.locality_complete = false;
                    id.final = is_final(id.final_inputs, settings);
                    r.ideals.push_back(id);
                    continue;
                }
                IdealGroupMatch own;
                own.group = group;
                own.own = true;
                own.matched = true;
                own.ideal_id = id.id;
                own.witness_poly = L.witness_poly.coeffs;
                own.w_hat_exponent = L.N;
                own.w_hat_exponent_other = L.N;
                id.groups.push_back(own);

                id.witness_poly = L.witness_poly.coeffs;
                id.coefficient_field.present = true;
                id.coefficient_field.w_hat_exponent = L.N;
                for (size_t i = 0; i < L.primes.size(); ++i) {
                    id.coefficient_field.primes.push_back(L.primes[i]);
                    id.coefficient_field.P.push_back(L.character[i].coeffs);
                    id.tangent_primes.push_back(L.primes[i]);
                    id.tangent_active.push_back(L.active[i]);
                    id.operator_primes.push_back(L.primes[i]);
                    id.t_p.push_back(L.t[i]);
                }
                id.algebra_present = true;
                id.algebra_dim = L.dim_A;
                id.embdim = L.embdim;
                id.nilpotency = L.nu;
                id.gorenstein = L.gorenstein;
                id.hilbert = L.hilbert;
                // (slot-7 WP-3, item 5) a per-sign ideal carries the label of
                // the unsplit ideal of its degree with the same system.
                if (!impl_->unsplit_labels.empty()) {
                    const std::vector<std::vector<std::vector<uint32_t>>> orb = own_orbit_(L);
                    for (const Impl::UnsplitLabel& u : impl_->unsplit_labels) {
                        if (u.has && u.degree == q && u.orbit == orb) { id.label = u.label; break; }
                    }
                    bool any = false;
                    for (const Impl::UnsplitLabel& u : impl_->unsplit_labels) any = any || u.has;
                    ENGINE_CHECK_MSG(!any || id.label.computed,
                                     "HeckeExtDriver: a per-sign ideal matches no unsplit ideal's "
                                     "system in its degree");
                }
                id.certificate_routes.a = L.cyclicity.cyclic ? "cyclic" : "not_cyclic";
                id.final_inputs.cyclic_passed = L.cyclicity.cyclic;
                id.final_inputs.commutant = CommutantOutcome::InProgress;
                id.final_inputs.sturm = SturmSweepOutcome::InProgress;
                const GenerationVerdict gv = generation_verdict(
                        L.cyclicity.cyclic, CommutantOutcome::InProgress,
                        SturmRouteStatus::Unavailable);
                id.certificate = ToString(gv.kind);

                const ModuleStructure ms = ComputeModuleStructure(L);
                if (ms.ok) id.modules.push_back({group, ms});

                // [E] D6: the Eisenstein line is a statement about the PLUS
                // side of degree 1, and it is the one canonical slot E3 fills.
                if (plus) {
                    id.canonical.push_back(EisensteinCanonicalPosition(split, group));
                }
                if (settings.split_field_report == hecke::SplitFieldReport::Canonical &&
                    !L.witness_poly.is_zero()) {
                    attach_split_(id, L);   // (slot-7 WP-3, item 6)
                }
                {
                    std::vector<uint32_t> ignored;   // (slot-7 WP-3, item 7)
                    PredictEntry(id, L, group, n_, ell_, ignored);
                }

                id.final_inputs.local = true;
                id.final_inputs.locality_complete = true;
                id.final_inputs.certificates_ok = true;
                id.final = is_final(id.final_inputs, settings);
                r.ideals.push_back(id);
            }
        }

        // [X] Sec. 6.2 E3 acceptance (c), as the RECORD can state it:
        // ASSERTED, never recorded.  The per-ideal matching by
        // SameMaximalIdeal is the validation's, which can run the test; what
        // is decidable here is the dimension identity, degree by degree.
        uint32_t unsplit_dim = 0;
        for (const IdealRecord& id : rec_.ideals) {
            for (const IdealGroupMatch& g : id.groups) {
                if (g.own && g.group == GroupName(q)) unsplit_dim += id.dim;
            }
        }
        uint32_t split_dim = 0;
        for (const HeckeExtRecord* r : {&plus_record_, &minus_record_}) {
            for (const IdealRecord& id : r->ideals) {
                for (const IdealGroupMatch& g : id.groups) {
                    if (g.own && (g.group == SignedGroupName(q, true) ||
                                  g.group == SignedGroupName(q, false))) {
                        split_dim += id.dim;
                    }
                }
            }
        }
        ENGINE_CHECK_MSG(unsplit_dim == split_dim,
                         "HeckeExtDriver: the unsplit record is not the sum of the two "
                         "per-sign records ([X] Sec. 6.2 E3 acceptance (c))");
        ENGINE_CHECK_MSG(split.dim_plus() + split.dim_minus() == split.dim_H(),
                         "HeckeExtDriver: dim H^q_+ + dim H^q_- != dim H^q "
                         "([C] prop:iotasplit(1))");
        sign_records_built_ = true;
        IotaPhaseTrace("sign records q=" + std::to_string(q) + " built");
    }

    if (sign_records_built_) {
        ENGINE_CHECK_MSG(plus_record_.verify() == 0 && minus_record_.verify() == 0,
                         "HeckeExtDriver: a per-sign record does not verify");
    }
    t_.iota_s = Secs(t0, Clock::now());
    if (opts_.trace) {
        std::cout << "EXT_DRIVER iota degrees=" << rec_.iota.size()
                  << " plus_ideals=" << plus_record_.ideals.size()
                  << " minus_ideals=" << minus_record_.ideals.size()
                  << " s=" << std::fixed << std::setprecision(4) << t_.iota_s << std::endl;
    }
}

// ---------------------------------------------------------------------------
// with_presentation -- the on-request presentation (slot-6 WP-2, OQ11).
//
// Runs LAST and touches no record: [X] Sec. 2.2 Stage 5's `presentation`
// record key is absent in slot 6 (the key that would switch it on is slot 7's
// re-baseline).  The unsplit side reads the local algebras Stage 3 already
// stored; the per-sign side recomputes each sign block's local algebra from
// the split, walking the split in exactly build_sign_split_()'s order, so the
// i-th presentation is the i-th ideal of the per-sign record -- asserted.
// ---------------------------------------------------------------------------
void HeckeExtDriver::build_presentations_() {
    const Clock::time_point t0 = Clock::now();
    presentations_.clear();
    sign_presentations_[0].clear();
    sign_presentations_[1].clear();

    for (const LocalAlgebra& L : impl_->algebras) presentations_.push_back(ComputePresentation(L));
    ENGINE_CHECK_MSG(presentations_.size() == rec_.ideals.size(),
                     "HeckeExtDriver: one presentation per ideal");

    if (opts_.with_sign_split) {
        for (size_t gi = 0; gi < degrees_.size() && gi < splits_.size(); ++gi) {
            if (!splits_[gi]) continue;
            const HeckeSignSplit& split = *splits_[gi];
            if (split.certificate().status != SignSplitStatus::Split) continue;
            for (int side = 0; side < 2; ++side) {
                const bool plus = (side == 0);
                const std::vector<std::pair<uint64_t, ExtMatrix>> ops = SignOperators(split, plus);
                const uint32_t dim = plus ? split.dim_plus() : split.dim_minus();
                if (dim == 0u || ops.empty()) continue;
                const hecke::HeckeModuleDecomposition* dec =
                        split.decomposition(plus ? SignSide::Plus : SignSide::Minus);
                if (dec == nullptr) continue;
                for (const hecke::PrimaryBlock& block : dec->blocks()) {
                    const LocalAlgebra L =
                            ComputeLocalAlgebra(block, ops, ell_, LocalAlgebraOptions());
                    sign_presentations_[side].push_back(ComputePresentation(L));
                }
            }
        }
        if (sign_records_built_) {
            ENGINE_CHECK_MSG(sign_presentations_[0].size() == plus_record_.ideals.size() &&
                                     sign_presentations_[1].size() == minus_record_.ideals.size(),
                             "HeckeExtDriver: one presentation per per-sign ideal");
        }
    }
    presentations_built_ = true;
    t_.presentation_s = Secs(t0, Clock::now());
    if (opts_.trace) {
        std::cout << "EXT_DRIVER presentation ideals=" << presentations_.size()
                  << " plus=" << sign_presentations_[0].size()
                  << " minus=" << sign_presentations_[1].size()
                  << " s=" << std::fixed << std::setprecision(4) << t_.presentation_s << std::endl;
    }
}

const Presentation& HeckeExtDriver::presentation(uint32_t ideal_index) const {
    ENGINE_CHECK_MSG(presentations_built_ && ideal_index < presentations_.size(),
                     "HeckeExtDriver::presentation(): not built (with_presentation) or out of range");
    return presentations_[ideal_index];
}

size_t HeckeExtDriver::num_sign_presentations(bool plus) const {
    ENGINE_CHECK_MSG(presentations_built_,
                     "HeckeExtDriver::num_sign_presentations(): not built (with_presentation)");
    return sign_presentations_[plus ? 0 : 1].size();
}

const Presentation& HeckeExtDriver::sign_presentation(bool plus, uint32_t ideal_index) const {
    const std::vector<Presentation>& v = sign_presentations_[plus ? 0 : 1];
    ENGINE_CHECK_MSG(presentations_built_ && ideal_index < v.size(),
                     "HeckeExtDriver::sign_presentation(): not built or out of range");
    return v[ideal_index];
}

// ===========================================================================
// run / resume
// ===========================================================================
void HeckeExtDriver::run() {
    ENGINE_CHECK_MSG(!ran_, "HeckeExtDriver::run(): already run; use resume_u_ell()");
    const Clock::time_point t0 = Clock::now();
    build_stack_();
    stage_one_();
    stages_two_to_five_();
    if (opts_.with_cone) build_cone_();
    if (opts_.with_sign_split) build_sign_split_();
    if (opts_.with_cone || opts_.with_sign_split) {
        ENGINE_CHECK_MSG(rec_.verify() == 0,
                         "HeckeExtDriver: the record does not verify after the attachments");
    }
    presentations_for_settings_();
    t_.total_s = Secs(t0, Clock::now());
    ran_ = true;
}

// ---------------------------------------------------------------------------
// (slot-7 WP-2; [X] Sec. 8 stage 1 (2), Sec. 2.2 Stage 5, OQ11) The
// `presentation` settings key.  Off (the default): exactly the slot-6
// behaviour -- the option computes, nothing is attached.  On: compute (as
// the option does) and attach the i-th presentation to the i-th ideal of the
// record, and the per-sign ones to the per-sign records, in the order
// build_presentations_() already asserts; every record re-verifies.  The
// switch is read from the record's OWN settings snapshot (taken in
// stages_two_to_five_), so the record and its presentation objects cannot
// disagree about the setting.
// ---------------------------------------------------------------------------
void HeckeExtDriver::presentations_for_settings_() {
    const bool key_on = (rec_.settings.presentation == hecke::PresentationMode::On);
    if (opts_.with_presentation || key_on) build_presentations_();
    if (!key_on) return;
    for (size_t i = 0; i < rec_.ideals.size(); ++i) {
        rec_.ideals[i].presentation_present = true;
        rec_.ideals[i].presentation_fragment = presentations_[i].to_json();
    }
    ENGINE_CHECK_MSG(rec_.verify() == 0,
                     "HeckeExtDriver: the record does not verify with its presentation objects");
    if (sign_records_built_) {
        HeckeExtRecord* side_rec[2] = {&plus_record_, &minus_record_};
        for (int side = 0; side < 2; ++side) {
            HeckeExtRecord& r = *side_rec[side];
            ENGINE_CHECK_MSG(r.settings.presentation == hecke::PresentationMode::On &&
                                 sign_presentations_[side].size() == r.ideals.size(),
                             "HeckeExtDriver: one presentation per per-sign ideal, under the "
                             "same setting");
            for (size_t i = 0; i < r.ideals.size(); ++i) {
                r.ideals[i].presentation_present = true;
                r.ideals[i].presentation_fragment = sign_presentations_[side][i].to_json();
            }
            ENGINE_CHECK_MSG(r.verify() == 0,
                             "HeckeExtDriver: a per-sign record does not verify with its "
                             "presentation objects");
        }
    }
}

void HeckeExtDriver::resume_u_ell(uint32_t budget_mb, hecke::SettingSource src) {
    ENGINE_CHECK_MSG(ran_, "HeckeExtDriver::resume_u_ell(): run() has not been called");
    // A resume on an INCLUDED verdict is a strict no-op: neither the budget nor
    // the history moves ([X] R3.1 item 2, the SturmSweep / CommutantRoute rule).
    if (s1_.u_ell.included) return;
    const uint32_t reached = u_ell_history_.empty() ? 0u : u_ell_history_.back();
    if (budget_mb <= reached) {
        throw std::runtime_error("resume: u_ell budget below reached state");
    }
    u_ell_history_.push_back(budget_mb);

    // `auto` at the shipped default, an explicit budget otherwise -- so a
    // resume to the default budget IS the default setting and fingerprints to
    // 0, exactly like a fresh `u_ell = auto` run.
    hecke::UEllSetting u;
    u.mode = (budget_mb == hecke::kDefaultUEllBudgetMb) ? hecke::UEllMode::Auto
                                                        : hecke::UEllMode::BudgetMb;
    u.budget_mb = budget_mb;
    // [KI] Sec. 4d: `src` names who is asking (the campaign passes `Spec`),
    // so the record's `settings.u_ell.source` stamps the resume's true
    // origin instead of a hard-coded `console` -- ScopedUEll itself still
    // collapses a resume to the shipped default down to `default`.
    const hecke::ScopedUEll guard(u, src);

    // Stage 1 again on the SAME family (its cache is keyed by (p, r), so the
    // four good primes and the cohomology are reused), then Stages 2-5.
    const Clock::time_point t0 = Clock::now();
    stage_one_();
    stages_two_to_five_();
    // The attachments are about the same cohomology, but they are attached to
    // the record stages_two_to_five_() has just rebuilt, so they are rebuilt
    // with it -- which is what makes "a resume equals a fresh run" true of the
    // WHOLE record and not only of its unsplit part.
    if (opts_.with_cone) {
        cone_.reset();
        pair_.reset();
        build_cone_();
    }
    if (opts_.with_sign_split) {
        involution_.reset();
        splits_.clear();
        sign_records_built_ = false;
        build_sign_split_();
    }
    presentations_for_settings_();
    t_.total_s += Secs(t0, Clock::now());
}

} // namespace hecke_ext
