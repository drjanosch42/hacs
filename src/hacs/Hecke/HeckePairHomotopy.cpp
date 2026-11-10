// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 2, WP-8 (phase E5-1) -- the certified pair homotopy
// eta ([C, def:pairhomotopy], [C, lem:etaexists], [C, obl:eta](i)-(iv)). See
// HeckePairHomotopy.h for the contract references and the design notes.

#include "HeckePairHomotopy.h"

#include "HeckeIntegerSolve.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace hecke {

const char* to_string(EtaVertexPolicy p) {
    switch (p) {
        case EtaVertexPolicy::CanonicalTreePath: return "canonical_tree_path";
        case EtaVertexPolicy::LegacyBfsPath: return "legacy_bfs_path";
    }
    return "?";
}

namespace {

// Parity (+1/-1) of a vertex-slot permutation restricted to slots 0..q
// (mirrors HeckeTauTable / HeckeBoundaryDatum; combinatorial, never
// geometric -- invariant I8).
int8_t perm_parity(const std::array<uint8_t, 3>& perm, uint32_t q) {
    int inversions = 0;
    for (uint32_t i = 0; i + 1 <= q; ++i) {
        for (uint32_t j = i + 1; j <= q; ++j) {
            if (perm[i] > perm[j]) ++inversions;
        }
    }
    return (inversions % 2 == 0) ? int8_t(1) : int8_t(-1);
}

bool lex_less(const Mat22& x, const Mat22& y) {
    if (x.a != y.a) return x.a < y.a;
    if (x.b != y.b) return x.b < y.b;
    if (x.c != y.c) return x.c < y.c;
    return x.d < y.d;
}

std::array<int64_t, 4> mat_key(const Mat22& m) { return { m.a, m.b, m.c, m.d }; }

std::string mat_to_string(const Mat22& m) {
    std::ostringstream oss;
    oss << "[[" << m.a << "," << m.b << "],[" << m.c << "," << m.d << "]]";
    return oss.str();
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

template<class Elt>
HeckePairHomotopyT<Elt>::HeckePairHomotopyT(const HeckeTauTableT<Elt>& tau,
                                            const HeckeBoundaryTauTableT<Elt>& taub,
                                            EtaVertexPolicy policy)
    : tau_(&tau), taub_(&taub), model_(&taub.model()), cosets_(&taub.cosets()), policy_(policy),
      N_(taub.N()), m_(taub.num_cosets()), simplices_(taub.K_simplices()) {
    // [C, obl:eta](i): the tables it refers to are the certified tau and tau_bd
    // of the SAME g, Gamma'' and s_j -- identity of the coset datum and of the
    // model (both objects exist only certified, invariant I8). Checked here so
    // that the construction never reads mismatched tables; re-checked by
    // certify() (codes 20-22).
    if (&tau.cosets() != &taub.cosets() || &tau.model() != &taub.model()) {
        throw std::runtime_error(
            "HeckePairHomotopy: tau and tau_bd are not built on the same coset datum / model ([C, obl:eta](i)).");
    }
    if (tau.num_cosets() != m_) {
        throw std::runtime_error("HeckePairHomotopy: coset count mismatch between tau and tau_bd.");
    }
    // Owner decision OP2 / [C, obl:eta](iv): the pair datum is over Z; a bulk
    // tau with a mod-ell fill anywhere cannot be part of it.
    if (tau.used_mod_ell_solve()) {
        throw std::runtime_error(
            "HeckePairHomotopy: the bulk tau carries a mod-ell fill; the pair datum requires an integral tau (OP2).");
    }
    if (simplices_.empty()) throw std::runtime_error("HeckePairHomotopy: empty model.");
    const uint32_t dim_X = static_cast<uint32_t>(simplices_.size()) - 1;
    // Degrees in which eta is built: a (q+1)-chain of X must exist
    // ([C, open:ell2] rider (i): derived on the finite model).
    const uint32_t num_deg = std::min(taub.num_degrees(), dim_X);
    records_.assign(num_deg, {});
    for (uint32_t q = 0; q < num_deg; ++q) {
        records_[q].assign(model_->boundary_simplices(q).size(), std::vector<ChainRecord>(m_));
    }

    // [C, lem:etaexists], in ascending degree: one construction per
    // Gamma''-orbit, every other slot derived through the (T1) helper.
    for (uint32_t q = 0; q < num_deg; ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            const uint32_t sigma = list[local];
            for (uint32_t j = 1; j <= m_; ++j) {
                if (try_derive(j, q, sigma)) {
                    ++num_derived_;
                    continue;
                }
                if (q == 0) construct_vertex(j, sigma);
                else construct_fill(j, q, sigma);
                ++num_constructed_;
            }
        }
    }

    tally();

    // MANDATORY certification ([C, obl:eta](i)-(iv), invariant I8): a table
    // failing it is REJECTED -- throw, never warn, never skip.
    const uint32_t code = certify();
    if (code != 0) {
        std::ostringstream oss;
        oss << "HeckePairHomotopy: certification FAILED (code " << code << ") for N = " << N_
            << " (m = " << m_ << ", policy = " << hecke::to_string(policy_)
            << ") -- table rejected, computation aborted ([C, obl:eta]).";
        throw std::runtime_error(oss.str());
    }
}

template<class Elt>
void HeckePairHomotopyT<Elt>::tally() {
    bound_B_tab_ = 0;
    num_nonzero_ = 0;
    num_terms_ = 0;
    max_abs_coeff_ = 0;
    for (const auto& per_q : records_) {
        for (const auto& per_sigma : per_q) {
            for (const auto& rec : per_sigma) {
                if (!rec.terms.empty()) ++num_nonzero_;
                for (const auto& term : rec.terms) {
                    ++num_terms_;
                    const uint64_t norm = static_cast<uint64_t>(term.gamma.max_abs());
                    if (norm > bound_B_tab_) bound_B_tab_ = norm;
                    const int64_t ac = (term.coeff < 0) ? -static_cast<int64_t>(term.coeff)
                                                        : static_cast<int64_t>(term.coeff);
                    if (ac > max_abs_coeff_) max_abs_coeff_ = ac;
                }
            }
        }
    }
    const uint64_t four_N_sq = 4u * N_ * N_;
    bound_B_ = (bound_B_tab_ > four_N_sq) ? bound_B_tab_ : four_N_sq;
}

// ============================================================================
// Bookkeeping
// ============================================================================

template<class Elt>
uint32_t HeckePairHomotopyT<Elt>::local_index(uint32_t q, uint32_t sigma) const {
    if (q >= records_.size()) throw std::out_of_range("HeckePairHomotopy: degree out of range.");
    const auto& list = model_->boundary_simplices(q);
    for (uint32_t k = 0; k < list.size(); ++k) {
        if (list[k] == sigma) return k;
    }
    throw std::invalid_argument("HeckePairHomotopy: simplex is not a boundary simplex.");
}

template<class Elt>
void HeckePairHomotopyT<Elt>::check_indices(uint32_t j, uint32_t q, uint32_t sigma) const {
    if (j == 0 || j > m_) {
        throw std::out_of_range("HeckePairHomotopy: coset index j must be in [1, num_cosets()].");
    }
    (void)local_index(q, sigma);
}

template<class Elt>
const typename HeckePairHomotopyT<Elt>::ChainRecord&
HeckePairHomotopyT<Elt>::record(uint32_t j, uint32_t q, uint32_t sigma) const {
    check_indices(j, q, sigma);
    return records_[q][local_index(q, sigma)][j - 1];
}

template<class Elt>
typename HeckePairHomotopyT<Elt>::ChainRecord&
HeckePairHomotopyT<Elt>::record_mut(uint32_t j, uint32_t q, uint32_t sigma) {
    check_indices(j, q, sigma);
    return records_[q][local_index(q, sigma)][j - 1];
}

template<class Elt>
void HeckePairHomotopyT<Elt>::accumulate_canonical(CanonMap& acc, const Mat22& gamma, uint32_t q,
                                                   uint32_t sigma, int64_t coeff) const {
    if (coeff == 0) return;
    const KuSimplexKey key = model_->canonical_key(gamma, q, sigma);
    const std::array<int64_t, 6> k = {
        static_cast<int64_t>(q), static_cast<int64_t>(key.sigma_hat),
        key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d
    };
    acc[k] = checked_add<int64_t>(acc[k], checked_mul<int64_t>(coeff, static_cast<int64_t>(key.sign)));
}

template<class Elt>
std::vector<ChainTerm> HeckePairHomotopyT<Elt>::canon_map_to_terms(const CanonMap& acc, uint32_t q) const {
    std::vector<ChainTerm> out;
    for (const auto& kv : acc) {
        if (kv.first[0] != static_cast<int64_t>(q)) {
            throw std::runtime_error("HeckePairHomotopy: canonical map dimension mismatch.");
        }
        const int64_t c = kv.second;
        if (c == 0) continue;
        if (c > INT32_MAX || c < INT32_MIN) {
            throw HeckeOverflow("HeckePairHomotopy: chain coefficient exceeds int32.");
        }
        out.push_back(ChainTerm{ static_cast<int32_t>(c),
                                 Mat22{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] },
                                 static_cast<uint32_t>(kv.first[1]) });
    }
    return out;
}

template<class Elt>
typename HeckePairHomotopyT<Elt>::CanonMap
HeckePairHomotopyT<Elt>::terms_to_canon_map(const std::vector<ChainTerm>& terms, uint32_t q) const {
    CanonMap acc;
    for (const auto& term : terms) accumulate_canonical(acc, term.gamma, q, term.sigma_idx, term.coeff);
    return acc;
}

template<class Elt>
uint32_t HeckePairHomotopyT<Elt>::face_simplex_index(uint32_t q, uint32_t sigma, uint32_t f) const {
    // d_f(sigma): delete the f-th vertex (increasing vertex order) and look
    // the (q-1)-simplex up in K's list (vertex lists are 0..n in order, so
    // for q == 1 the vertex id is the 0-simplex index).
    const auto& verts = simplices_.at(q).at(sigma);
    if (f > q) throw std::runtime_error("HeckePairHomotopy: face index out of range.");
    std::vector<uint32_t> face;
    for (uint32_t k = 0; k <= q; ++k) {
        if (k != f) face.push_back(verts[k]);
    }
    const auto& list = simplices_.at(q - 1);
    for (uint32_t k = 0; k < list.size(); ++k) {
        if (list[k] == face) return k;
    }
    throw std::runtime_error("HeckePairHomotopy: face not found in the model complex.");
}

template<class Elt>
typename HeckePairHomotopyT<Elt>::CanonMap
HeckePairHomotopyT<Elt>::boundary_of_terms(const std::vector<ChainTerm>& terms, uint32_t q) const {
    // boundary(v_0..v_q) = sum_f (-1)^f (v_0..\hat v_f..v_q), the convention
    // of [C, sec:conventions] (as in HeckeTauTable::boundary_of_terms).
    if (q == 0) throw std::runtime_error("HeckePairHomotopy: boundary_of_terms needs q >= 1.");
    CanonMap acc;
    for (const auto& term : terms) {
        for (uint32_t f = 0; f <= q; ++f) {
            const uint32_t face = face_simplex_index(q, term.sigma_idx, f);
            const int64_t sgn = (f % 2 == 0) ? 1 : -1;
            accumulate_canonical(acc, term.gamma, q - 1, face, sgn * static_cast<int64_t>(term.coeff));
        }
    }
    return acc;
}

template<class Elt>
bool HeckePairHomotopyT<Elt>::canon_maps_equal(const CanonMap& x, const CanonMap& y) {
    for (const auto& kv : x) {
        const auto it = y.find(kv.first);
        const int64_t other = (it == y.end()) ? 0 : it->second;
        if (kv.second != other) return false;
    }
    for (const auto& kv : y) {
        if (x.find(kv.first) == x.end() && kv.second != 0) return false;
    }
    return true;
}

template<class Elt>
typename HeckePairHomotopyT<Elt>::CanonMap
HeckePairHomotopyT<Elt>::defect_map(uint32_t j, uint32_t q, uint32_t sigma) const {
    // (j tau_bd - tau j)(s_j . i(sigma)): the two stored slots. The tau_bd
    // terms name boundary simplices of K, which are simplices of K, so the
    // inclusion j is the identity on ChainTerms; both tables are canonical
    // under the same model, so the difference is formed term by term.
    CanonMap acc;
    for (const auto& term : taub_->tau(j, q, sigma)) {
        accumulate_canonical(acc, term.gamma, q, term.sigma_idx, term.coeff);
    }
    for (const auto& term : tau_->tau(j, q, sigma)) {
        accumulate_canonical(acc, term.gamma, q, term.sigma_idx, -static_cast<int64_t>(term.coeff));
    }
    return acc;
}

template<class Elt>
std::vector<ChainTerm> HeckePairHomotopyT<Elt>::defect(uint32_t j, uint32_t q, uint32_t sigma) const {
    check_indices(j, q, sigma);
    return canon_map_to_terms(defect_map(j, q, sigma), q);
}

template<class Elt>
std::vector<ChainTerm> HeckePairHomotopyT<Elt>::defect_on_translate(const Mat22& gamma, uint32_t q,
                                                                    uint32_t sigma) const {
    uint32_t j = 0;
    const Mat22 conj = cosets_->twisted_conjugate(gamma, j);
    check_indices(j, q, sigma);
    CanonMap acc;
    for (const auto& term : canon_map_to_terms(defect_map(j, q, sigma), q)) {
        accumulate_canonical(acc, conj * term.gamma, q, term.sigma_idx, term.coeff);
    }
    return canon_map_to_terms(acc, q);
}

// ============================================================================
// The single (T1) helper
// ============================================================================

template<class Elt>
std::vector<ChainTerm> HeckePairHomotopyT<Elt>::eval_on_translate(const Records& recs, const Mat22& gamma,
                                                                  uint32_t q, uint32_t sigma) const {
    // gamma = h s_j, j = class_index_of(gamma), h in Gamma_0(N); the exact
    // integral conjugate g^{-1} h g comes from the ONE shared function
    // HeckeCosetTables::twisted_conjugate (tau and tau_bd use the same).
    uint32_t j = 0;
    const Mat22 conj = cosets_->twisted_conjugate(gamma, j);
    check_indices(j, q, sigma);
    const ChainRecord& rec = recs[q][local_index(q, sigma)][j - 1];
    if (!rec.stored) {
        throw std::runtime_error("HeckePairHomotopy: (T1) helper hit an unbuilt slot (construction-order bug).");
    }
    CanonMap acc;
    for (const auto& term : rec.terms) {
        accumulate_canonical(acc, conj * term.gamma, q + 1, term.sigma_idx, term.coeff);
    }
    return canon_map_to_terms(acc, q + 1);
}

template<class Elt>
std::vector<ChainTerm> HeckePairHomotopyT<Elt>::eta_on_translate(const Mat22& gamma, uint32_t q,
                                                                 uint32_t sigma) const {
    return eval_on_translate(records_, gamma, q, sigma);
}

template<class Elt>
typename HeckePairHomotopyT<Elt>::CanonMap
HeckePairHomotopyT<Elt>::eta_of_boundary(const Records& recs, uint32_t j, uint32_t q, uint32_t sigma) const {
    // eta(boundary(s_j . i(sigma))) = sum_f (-1)^f eta(s_j . i(face_f)), each
    // face through the helper (a q-map; the values are q-chains of X).
    CanonMap acc;
    for (uint32_t f = 0; f <= q; ++f) {
        const uint32_t face = face_simplex_index(q, sigma, f);
        const std::vector<ChainTerm> ch = eval_on_translate(recs, cosets_->s(j), q - 1, face);
        const int64_t sgn = (f % 2 == 0) ? 1 : -1;
        for (const auto& term : ch) {
            accumulate_canonical(acc, term.gamma, q, term.sigma_idx, sgn * static_cast<int64_t>(term.coeff));
        }
    }
    return acc;
}

// ============================================================================
// Slot derivation through (T1)
// ============================================================================

template<class Elt>
bool HeckePairHomotopyT<Elt>::try_derive(uint32_t j, uint32_t q, uint32_t sigma) {
    for (uint32_t sigma2 : model_->boundary_simplices(q)) {
        if (!model_->same_groupoid_orbit(q, sigma, sigma2)) continue;
        const auto& ts = model_->transporter(q, sigma, sigma2);
        const auto& perms = model_->transporter_vertex_perms(q, sigma, sigma2);
        for (size_t k = 0; k < ts.size(); ++k) {
            // i(sigma) = t . i(sigma2)  ==>  s_j . i(sigma) = (s_j t) . i(sigma2).
            const Mat22 gamma = cosets_->s(j) * ts[k];
            const uint32_t j2 = cosets_->class_index_of(gamma);
            if (j2 == j && sigma2 == sigma) continue; // self (a constraint, not a derivation)
            const ChainRecord& src = records_[q][local_index(q, sigma2)][j2 - 1];
            if (!src.stored) continue;
            std::vector<ChainTerm> value = eta_on_translate(gamma, q, sigma2);
            const int8_t eps = perm_parity(perms[k], q);
            if (eps < 0) {
                for (auto& term : value) term.coeff = -term.coeff;
            }
            ChainRecord& rec = record_mut(j, q, sigma);
            rec.terms = std::move(value);
            rec.stored = true;
            rec.derived = true;
            rec.regime_Z = src.regime_Z;
            rec.rings = 0;
            return true;
        }
    }
    return false;
}

// ============================================================================
// [C, lem:etaexists](V): vertices -- a path from tau(s_j v) to tau_bd(s_j v)
// ============================================================================

template<class Elt>
void HeckePairHomotopyT<Elt>::append_vertex_path(CanonMap& acc, const Mat22& ga, uint32_t wa, const Mat22& gb,
                                                 uint32_t wb) const {
    // The [C, alg:tau] step-2 device: a 1-chain of X with boundary
    // gamma_b . i(wb) - gamma_a . i(wa). Reduce gamma_a^{-1} gamma_b to a word
    // in T^{+-1}, S by (P2) and concatenate translated, sign-tracked path
    // templates (each certified in S2: the template boundary equals
    // letter . i(w2) - i(w) exactly over Z), so the boundary telescopes. The
    // -I letter is dropped (-I acts trivially on K^u; canonical keys quotient
    // by it). Nothing to add when the two are the same K^u vertex.
    if (ga == gb && wa == wb) return;
    const Mat22 rel = ga.inv_unimodular() * gb;
    const auto word = model_->word_in_T_S(rel);
    std::vector<char> letters;
    for (const auto& lp : word) {
        if (lp.first == '-') continue;
        if (lp.first == 'S') {
            letters.push_back('S');
        } else {
            const char c = (lp.second > 0) ? 'T' : 't';
            const uint64_t reps = uabs64(lp.second);
            for (uint64_t k = 0; k < reps; ++k) letters.push_back(c);
        }
    }
    if (letters.empty()) letters = { 'T', 't' }; // rel = +-I, different vertices: the net-identity detour

    const SL2ZTessellationTables::PathTemplateFamily family =
        (policy_ == EtaVertexPolicy::LegacyBfsPath) ? SL2ZTessellationTables::PathTemplateFamily::LegacyBfs
                                                    : SL2ZTessellationTables::PathTemplateFamily::Canonical;
    Mat22 P = ga; // running prefix gamma_a L_1 ... L_{k-1}
    uint32_t from_w = wa;
    for (const char L : letters) {
        const uint32_t to_w = wb; // deterministic anchor: the target vertex
        const auto& tmpl = model_->path_template(L, from_w, to_w, family);
        for (const auto& t : tmpl) accumulate_canonical(acc, P * t.gamma, 1, t.edge_idx, t.coeff);
        const Mat22 Lm = (L == 'T') ? Mat22::T() : (L == 't') ? Mat22::T().inv_unimodular() : Mat22::S();
        P = P * Lm;
        from_w = to_w;
    }
    if (!(P == gb || P == gb.neg())) {
        throw std::runtime_error("HeckePairHomotopy: vertex path prefix mismatch (internal (P2) error).");
    }
}

template<class Elt>
void HeckePairHomotopyT<Elt>::construct_vertex(uint32_t j, uint32_t v) {
    // [C, lem:etaexists](V): the 0-chain c := (j tau_bd - tau j)(s_j v) has
    // augmentation 0 ((T3bd) and (T3)), hence bounds; ANY 1-chain with
    // boundary c will do. The chain is routed pair by pair: the support of c
    // splits into the vertices B_1, ..., B_k with positive and A_1, ..., A_k
    // with negative coefficient (counted with multiplicity; equal counts
    // because the augmentation is 0), and eta(s_j v) := sum_i path(A_i -> B_i)
    // in the deterministic (canonical-key) order. On the shipped tables both
    // tau and tau_bd send a vertex to a single vertex with coefficient 1
    // ([C, alg:tau] step 1, [C, lem:taubdryexists](1)), so k <= 1 and this is
    // exactly the edge path from tau(s_j v) to tau_bd(s_j v) (empty when they
    // coincide); the pairing is what makes the step valid for ANY datum of
    // [C, def:tau] / [C, def:taubdry], which erratum E5 of [C] asks for.
    const CanonMap c = defect_map(j, 0, v);
    std::vector<std::pair<Mat22, uint32_t>> pos, neg;
    for (const auto& kv : c) {
        if (kv.second == 0) continue;
        const Mat22 g{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] };
        const uint32_t w = static_cast<uint32_t>(kv.first[1]);
        const int64_t reps = (kv.second > 0) ? kv.second : -kv.second;
        for (int64_t r = 0; r < reps; ++r) (kv.second > 0 ? pos : neg).emplace_back(g, w);
    }
    if (pos.size() != neg.size()) {
        throw std::runtime_error(
            "HeckePairHomotopy: (j tau_bd - tau j) has non-zero augmentation on a vertex slot ((T3)/(T3bd) violated).");
    }
    CanonMap acc;
    for (size_t i = 0; i < pos.size(); ++i) {
        append_vertex_path(acc, neg[i].first, neg[i].second, pos[i].first, pos[i].second);
    }
    if (pos.size() > 1) ++num_vertex_pairs_beyond_one_;
    ChainRecord& rec = record_mut(j, 0, v);
    rec.regime_Z = true;
    rec.rings = 0;
    rec.terms = canon_map_to_terms(acc, 1);
    rec.stored = true;
}

// ============================================================================
// [C, lem:etaexists](E): the unique fill (degree-generic, q >= 1)
// ============================================================================

template<class Elt>
bool HeckePairHomotopyT<Elt>::solve_fill(uint32_t q, const CanonMap& z_in, std::vector<ChainTerm>& x_out,
                                         uint32_t& rings_out) const {
    // Solve boundary(x) = z for a (q+1)-chain x of X supported on a finite
    // subcomplex ([C, alg:tau] step 3 with q+1 in place of 2): initialise the
    // neighbourhood N as the translates occurring in supp z, take as columns
    // the canonical (q+1)-keys of N and as rows the canonical q-keys of supp z
    // together with all column faces, solve the integer system, and grow N
    // by one ring of adjacent translates (T, T^{-1}, S) on failure.
    rings_out = 0;
    CanonMap z;
    for (const auto& kv : z_in) {
        if (kv.second != 0) z.insert(kv);
    }
    if (z.empty()) {
        x_out.clear();
        return true;
    }
    const uint32_t dim_X = static_cast<uint32_t>(simplices_.size()) - 1;
    if (q + 1 > dim_X) return false; // C_{q+1}(X) = 0: a non-zero z has no fill

    const auto norm_pm = [](const Mat22& g) { return lex_less(g, g.neg()) ? g : g.neg(); };
    std::set<std::array<int64_t, 4>> translate_keys;
    std::vector<Mat22> translates;
    const auto add_translate = [&](const Mat22& g) {
        const Mat22 n = norm_pm(g);
        if (translate_keys.insert(mat_key(n)).second) translates.push_back(n);
    };
    for (const auto& kv : z) add_translate(Mat22{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] });

    const uint32_t num_cells = static_cast<uint32_t>(simplices_[q + 1].size());
    const Mat22 gens[3] = { Mat22::T(), Mat22::T().inv_unimodular(), Mat22::S() };
    constexpr uint32_t kMaxRings = 12;

    for (uint32_t ring = 0; ring <= kMaxRings; ++ring) {
        rings_out = ring;
        // Columns: canonical (q+1)-keys of the neighbourhood.
        std::map<std::array<int64_t, 6>, size_t> col_index;
        std::vector<std::pair<Mat22, uint32_t>> cols; // (gamma_canon, sigma_hat)
        for (const auto& g : translates) {
            for (uint32_t c = 0; c < num_cells; ++c) {
                const KuSimplexKey key = model_->canonical_key(g, q + 1, c);
                const std::array<int64_t, 6> arr = {
                    static_cast<int64_t>(q + 1), static_cast<int64_t>(key.sigma_hat),
                    key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d
                };
                if (col_index.find(arr) == col_index.end()) {
                    col_index[arr] = cols.size();
                    cols.emplace_back(key.gamma_canon, key.sigma_hat);
                }
            }
        }
        // Rows: canonical q-keys (supp z first, then all column faces).
        std::map<std::array<int64_t, 6>, size_t> row_index;
        std::vector<int64_t> rhs;
        for (const auto& kv : z) {
            row_index[kv.first] = rhs.size();
            rhs.push_back(kv.second);
        }
        std::vector<std::tuple<size_t, size_t, int64_t>> entries; // (row, col, coeff)
        for (size_t c = 0; c < cols.size(); ++c) {
            for (uint32_t f = 0; f <= q + 1; ++f) {
                const uint32_t face = face_simplex_index(q + 1, cols[c].second, f);
                const KuSimplexKey key = model_->canonical_key(cols[c].first, q, face);
                const std::array<int64_t, 6> arr = {
                    static_cast<int64_t>(q), static_cast<int64_t>(key.sigma_hat),
                    key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d
                };
                auto it = row_index.find(arr);
                if (it == row_index.end()) {
                    it = row_index.emplace(arr, rhs.size()).first;
                    rhs.push_back(0);
                }
                const int64_t sgn = (f % 2 == 0) ? 1 : -1;
                entries.emplace_back(it->second, c, sgn * static_cast<int64_t>(key.sign));
            }
        }
        std::vector<std::vector<int64_t>> A(rhs.size(), std::vector<int64_t>(cols.size(), 0));
        for (const auto& en : entries) {
            A[std::get<0>(en)][std::get<1>(en)] =
                checked_add<int64_t>(A[std::get<0>(en)][std::get<1>(en)], std::get<2>(en));
        }
        // Over Z, and ONLY over Z (OP2): an overflow propagates as the tier
        // failure it is; a Z-infeasible system means the neighbourhood is
        // too small (the fill exists and is unique in X), so grow it.
        std::vector<int64_t> y;
        if (solve_integer_system(A, rhs, y)) {
            CanonMap x_map;
            for (size_t c = 0; c < cols.size(); ++c) {
                if (y[c] == 0) continue;
                const std::array<int64_t, 6> arr = {
                    static_cast<int64_t>(q + 1), static_cast<int64_t>(cols[c].second),
                    cols[c].first.a, cols[c].first.b, cols[c].first.c, cols[c].first.d
                };
                x_map[arr] = y[c];
            }
            x_out = canon_map_to_terms(x_map, q + 1);
            return true;
        }
        const size_t old_size = translates.size();
        for (size_t idx = 0; idx < old_size; ++idx) {
            for (const auto& L : gens) add_translate(translates[idx] * L);
        }
        if (translates.size() == old_size) return false;
    }
    return false;
}

template<class Elt>
std::vector<ChainTerm> HeckePairHomotopyT<Elt>::unique_fill(uint32_t q, const std::vector<ChainTerm>& z,
                                                            uint32_t* rings_out) const {
    if (q + 1 >= simplices_.size()) {
        throw std::invalid_argument("HeckePairHomotopy::unique_fill: degree q+1 exceeds dim X.");
    }
    const CanonMap z_map = terms_to_canon_map(z, q);
    std::vector<ChainTerm> x;
    uint32_t rings = 0;
    if (!solve_fill(q, z_map, x, rings)) {
        throw std::runtime_error("HeckePairHomotopy::unique_fill: no fill within the ring budget (z is not a boundary?).");
    }
    if (rings_out) *rings_out = rings;
    return x;
}

template<class Elt>
void HeckePairHomotopyT<Elt>::construct_fill(uint32_t j, uint32_t q, uint32_t s) {
    // z := (j tau_bd - tau j)(s_j sigma) - eta(boundary(s_j sigma)), a q-cycle
    // of X by (T2), (T2bd) and the identity in degree q-1 already certified
    // by construction ([C, lem:etaexists](E)); its fill is unique.
    CanonMap z = defect_map(j, q, s);
    const CanonMap eb = eta_of_boundary(records_, j, q, s);
    for (const auto& kv : eb) z[kv.first] = checked_sub<int64_t>(z[kv.first], kv.second);
    std::vector<ChainTerm> x;
    uint32_t rings = 0;
    if (!solve_fill(q, z, x, rings)) {
        std::ostringstream oss;
        oss << "HeckePairHomotopy: fill unsolvable after max ring growth (j = " << j << ", q = " << q
            << ", sigma = " << s << ") -- internal error (X is acyclic; this must not happen).";
        throw std::runtime_error(oss.str());
    }
    ChainRecord& rec = record_mut(j, q, s);
    rec.terms = std::move(x);
    rec.stored = true;
    rec.regime_Z = true;
    rec.rings = rings;
    ++num_fills_;
    if (rings > max_rings_) max_rings_ = rings;
}

// ============================================================================
// Accessors
// ============================================================================

template<class Elt>
const std::vector<ChainTerm>& HeckePairHomotopyT<Elt>::eta(uint32_t j, uint32_t q, uint32_t sigma) const {
    return record(j, q, sigma).terms;
}

template<class Elt>
bool HeckePairHomotopyT<Elt>::slot_derived(uint32_t j, uint32_t q, uint32_t sigma) const {
    return record(j, q, sigma).derived;
}

template<class Elt>
uint32_t HeckePairHomotopyT<Elt>::num_nonzero_slots(uint32_t q) const {
    if (q >= records_.size()) throw std::out_of_range("HeckePairHomotopy: degree out of range.");
    uint32_t n = 0;
    for (const auto& per_sigma : records_[q]) {
        for (const auto& rec : per_sigma) {
            if (!rec.terms.empty()) ++n;
        }
    }
    return n;
}

template<class Elt>
std::vector<uint32_t> HeckePairHomotopyT<Elt>::degrees() const {
    std::vector<uint32_t> d;
    for (uint32_t q = 0; q < num_degrees(); ++q) d.push_back(q);
    return d;
}

template<class Elt>
uint64_t HeckePairHomotopyT<Elt>::table_fnv64() const {
    engine::FnvHasher h;
    h.feed(N_);
    h.feed(m_);
    for (uint32_t q = 0; q < num_degrees(); ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = records_[q][local][j - 1];
                h.feed(q);
                h.feed(list[local]);
                h.feed(j);
                h.feed(rec.terms.size());
                for (const auto& t : rec.terms) {
                    h.feed(static_cast<uint64_t>(static_cast<int64_t>(t.coeff)));
                    h.feed(static_cast<uint64_t>(t.gamma.a));
                    h.feed(static_cast<uint64_t>(t.gamma.b));
                    h.feed(static_cast<uint64_t>(t.gamma.c));
                    h.feed(static_cast<uint64_t>(t.gamma.d));
                    h.feed(t.sigma_idx);
                }
            }
        }
    }
    return h.value();
}

namespace {

void feed_terms(engine::FnvHasher& h, const std::vector<ChainTerm>& terms) {
    h.feed(terms.size());
    for (const auto& t : terms) {
        h.feed(static_cast<uint64_t>(static_cast<int64_t>(t.coeff)));
        h.feed(static_cast<uint64_t>(t.gamma.a));
        h.feed(static_cast<uint64_t>(t.gamma.b));
        h.feed(static_cast<uint64_t>(t.gamma.c));
        h.feed(static_cast<uint64_t>(t.gamma.d));
        h.feed(t.sigma_idx);
    }
}

} // namespace

template<class Elt>
uint64_t HeckePairHomotopyT<Elt>::tau_table_fnv64() const {
    engine::FnvHasher h;
    h.feed(N_);
    h.feed(m_);
    for (uint32_t q = 0; q < simplices_.size(); ++q) {
        for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
            for (uint32_t j = 1; j <= m_; ++j) {
                h.feed(q);
                h.feed(sigma);
                h.feed(j);
                feed_terms(h, tau_->tau(j, q, sigma));
            }
        }
    }
    return h.value();
}

template<class Elt>
uint64_t HeckePairHomotopyT<Elt>::tau_boundary_table_fnv64() const {
    engine::FnvHasher h;
    h.feed(N_);
    h.feed(m_);
    for (uint32_t q = 0; q < taub_->num_degrees(); ++q) {
        for (uint32_t sigma : model_->boundary_simplices(q)) {
            for (uint32_t j = 1; j <= m_; ++j) {
                h.feed(q);
                h.feed(sigma);
                h.feed(j);
                feed_terms(h, taub_->tau(j, q, sigma));
            }
        }
    }
    return h.value();
}

template<class Elt>
const char* HeckePairHomotopyT<Elt>::solve_regime_tau() const {
    return tau_->used_mod_ell_solve() ? "mod_ell" : "Z";
}

// ============================================================================
// MANDATORY certification ([C, obl:eta](i)-(iv))
// ============================================================================

template<class Elt>
uint32_t HeckePairHomotopyT<Elt>::certify() const {
    return certify_impl(records_);
}

template<class Elt>
uint32_t HeckePairHomotopyT<Elt>::certify_with_override(uint32_t j, uint32_t q, uint32_t sigma,
                                                        const std::vector<ChainTerm>& replacement) const {
    check_indices(j, q, sigma);
    Records copy = records_;
    ChainRecord& rec = copy[q][local_index(q, sigma)][j - 1];
    rec.terms = replacement;
    rec.stored = true;
    return certify_impl(copy);
}

template<class Elt>
uint32_t HeckePairHomotopyT<Elt>::certify_impl(const Records& recs) const {
    const uint32_t num_deg = static_cast<uint32_t>(recs.size());
    const uint32_t dim_X = static_cast<uint32_t>(simplices_.size()) - 1;

    // (i) the tables: identity of the coset datum and model, tau integral,
    // the degree bookkeeping of [C, open:ell2].
    if (&tau_->cosets() != cosets_ || &tau_->model() != model_ || &taub_->cosets() != cosets_ ||
        &taub_->model() != model_) {
        return 20;
    }
    if (tau_->used_mod_ell_solve()) return 21;
    if (num_deg != std::min(taub_->num_degrees(), dim_X)) return 22;

    // (i) structural, per term.
    for (uint32_t q = 0; q < num_deg; ++q) {
        const auto& list = model_->boundary_simplices(q);
        if (recs[q].size() != list.size()) return 10;
        for (uint32_t local = 0; local < list.size(); ++local) {
            if (recs[q][local].size() != m_) return 10;
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = recs[q][local][j - 1];
                if (!rec.stored) return 10;
                for (const auto& term : rec.terms) {
                    if (term.coeff == 0) return 13;
                    if (term.gamma.det() != 1) return 11;
                    if (term.sigma_idx >= simplices_[q + 1].size()) return 14;
                    const KuSimplexKey key = model_->canonical_key(term.gamma, q + 1, term.sigma_idx);
                    if (key.sigma_hat != term.sigma_idx || !(key.gamma_canon == term.gamma) || key.sign != 1) {
                        return 12;
                    }
                }
                // (iv): the recorded regime of every slot is Z.
                if (!rec.regime_Z) return 60;
            }
        }
    }

    // (ii) the homotopy identity [C, eq:pairhomotopy] on every stored slot:
    //   degree 0:  boundary(eta(s_j sigma)) == (j tau_bd - tau j)(s_j sigma);
    //   degree >=1: boundary(eta(s_j sigma)) + eta(boundary(s_j sigma)) == the same,
    // the eta-faces ONLY through the (T1) helper over `recs`, tau_bd through
    // its helper (h = I on a stored slot), tau the stored slot (its helper at
    // h = I), all compared as canonical maps over Z.
    for (uint32_t q = 0; q < num_deg; ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            const uint32_t sigma = list[local];
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = recs[q][local][j - 1];
                CanonMap lhs = boundary_of_terms(rec.terms, q + 1);
                if (q >= 1) {
                    const CanonMap eb = eta_of_boundary(recs, j, q, sigma);
                    for (const auto& kv : eb) lhs[kv.first] = checked_add<int64_t>(lhs[kv.first], kv.second);
                }
                CanonMap rhs;
                for (const auto& term : taub_->tau_on_translate(cosets_->s(j), q, sigma)) {
                    accumulate_canonical(rhs, term.gamma, q, term.sigma_idx, term.coeff);
                }
                for (const auto& term : tau_->tau(j, q, sigma)) {
                    accumulate_canonical(rhs, term.gamma, q, term.sigma_idx, -static_cast<int64_t>(term.coeff));
                }
                if (!canon_maps_equal(lhs, rhs)) return (q == 0) ? 30 : 31;
            }
        }
    }

    // (iii) (T1) bookkeeping over EVERY stored transporter between boundary
    // simplices and every j (the {+-I} self-constraints included).
    for (uint32_t q = 0; q < num_deg; ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            const uint32_t sigma = list[local];
            for (uint32_t sigma2 : list) {
                if (!model_->same_groupoid_orbit(q, sigma, sigma2)) continue;
                const auto& ts = model_->transporter(q, sigma, sigma2);
                const auto& perms = model_->transporter_vertex_perms(q, sigma, sigma2);
                for (size_t k = 0; k < ts.size(); ++k) {
                    for (uint32_t j = 1; j <= m_; ++j) {
                        const Mat22 gamma = cosets_->s(j) * ts[k];
                        const std::vector<ChainTerm> ch = eval_on_translate(recs, gamma, q, sigma2);
                        const int8_t eps = perm_parity(perms[k], q);
                        CanonMap rhs;
                        for (const auto& term : ch) {
                            accumulate_canonical(rhs, term.gamma, q + 1, term.sigma_idx,
                                                 static_cast<int64_t>(eps) * static_cast<int64_t>(term.coeff));
                        }
                        const CanonMap lhs = terms_to_canon_map(recs[q][local][j - 1].terms, q + 1);
                        if (!canon_maps_equal(lhs, rhs)) return 40;
                    }
                }
            }
        }
    }

    // (iii) bounds ([C, lem:bounds](5) recounted on the eta-chains, [C,
    // obl:backend]): full recount, the int64 tier.
    {
        uint64_t recount = 0;
        for (const auto& per_q : recs) {
            for (const auto& per_sigma : per_q) {
                for (const auto& rec : per_sigma) {
                    for (const auto& term : rec.terms) {
                        const uint64_t norm = static_cast<uint64_t>(term.gamma.max_abs());
                        if (norm > recount) recount = norm;
                    }
                }
            }
        }
        if (&recs == &records_ && recount != bound_B_tab_) return 50;
        const uint64_t four_N_sq = 4u * N_ * N_;
        const uint64_t B = (recount > four_N_sq) ? recount : four_N_sq;
        if (&recs == &records_ && B != bound_B_) return 50;
        if (B > 2147483647ull) return 51; // 2 B^2 >= 2^63: int64 tier violated
    }

    return 0;
}

// ============================================================================
// Pretty printing / record
// ============================================================================

template<class Elt>
std::string HeckePairHomotopyT<Elt>::to_string_offset(uint32_t offset) const {
    const std::string ind(offset, ' ');
    std::ostringstream oss;
    oss << ind << "HeckePairHomotopy (certified pair homotopy eta, [C, def:pairhomotopy]/[C, lem:etaexists]):\n";
    oss << ind << "  g = diag(1, " << N_ << "), N = p^nu = " << N_ << " (p = " << cosets_->p_hecke()
        << ", nu = " << cosets_->r() << "), m = " << m_ << " cosets; vertex policy = "
        << hecke::to_string(policy_) << "; solve regime = " << solve_regime()
        << " (integral = " << (integral() ? "true" : "false") << ")\n";
    oss << ind << "  degrees built: 0.." << q_max() << " (q_max derived, truncated = false); B_tab = "
        << bound_B_tab_ << ", B = max(B_tab, 4 N^2) = " << bound_B_ << " (int64 tier 2 B^2 < 2^63: "
        << (bound_B_ <= 2147483647ull ? "OK" : "VIOLATED") << ")\n";
    oss << ind << "  stored chains: " << num_slots() << " (" << num_constructed_ << " constructed, "
        << num_derived_ << " derived via (T1)); nonzero " << num_nonzero_ << " (eta_zero = "
        << (eta_zero() ? "true" : "false") << "); fills " << num_fills_ << ", max rings " << max_rings_
        << ", terms " << num_terms_ << ", max |coeff| " << max_abs_coeff_ << "\n";
    const auto simplex_verts = [&](uint32_t q, uint32_t s) {
        std::string r = "{";
        const auto& verts = simplices_[q][s];
        for (size_t k = 0; k < verts.size(); ++k) {
            r += std::to_string(verts[k]);
            if (k + 1 < verts.size()) r += ",";
        }
        return r + "}";
    };
    oss << ind << "  eta(s_j . i(sigma)) = sum_t n_t * gamma_t . i(sigma_t), sigma_t a (q+1)-simplex, terms canonical:\n";
    for (uint32_t q = 0; q < num_degrees(); ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            const uint32_t sigma = list[local];
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = records_[q][local][j - 1];
                oss << ind << "    [j=" << j << "][q=" << q << "][sigma=" << sigma << " " << simplex_verts(q, sigma)
                    << "]: ";
                if (rec.terms.empty()) {
                    oss << "0";
                } else {
                    for (size_t k = 0; k < rec.terms.size(); ++k) {
                        const auto& t = rec.terms[k];
                        if (k > 0) oss << " + ";
                        oss << "(" << t.coeff << ") * " << mat_to_string(t.gamma) << " . i(" << t.sigma_idx << " "
                            << simplex_verts(q + 1, t.sigma_idx) << ")";
                    }
                }
                if (rec.derived) oss << "   [derived via (T1)]";
                else if (q >= 1) oss << "   [fill, rings " << rec.rings << "]";
                oss << "\n";
            }
        }
    }
    return oss.str();
}

template<class Elt>
std::string HeckePairHomotopyT<Elt>::to_json() const {
    std::ostringstream oss;
    oss << "{\"schema\":\"hecke_pair_homotopy/v1\"";
    oss << ",\"p\":" << cosets_->p_hecke() << ",\"nu\":" << cosets_->r() << ",\"N\":" << N_ << ",\"m\":" << m_;
    oss << ",\"vertex_policy\":\"" << hecke::to_string(policy_) << "\"";
    oss << ",\"solve_regime\":\"" << solve_regime() << "\",\"integral\":" << (integral() ? "true" : "false");
    oss << ",\"regimes\":{\"tau\":\"" << solve_regime_tau() << "\",\"tau_bd\":\"" << solve_regime_tau_boundary()
        << "\",\"eta\":\"" << solve_regime() << "\"}";
    oss << ",\"built_against\":{\"tau_fnv64\":\"0x" << std::hex << tau_table_fnv64() << "\",\"tau_bd_fnv64\":\"0x"
        << tau_boundary_table_fnv64() << std::dec << "\"}";
    oss << ",\"vertex_slots_multi_path\":" << num_vertex_pairs_beyond_one_;
    oss << ",\"q_max\":" << q_max() << ",\"truncated\":" << (truncated() ? "true" : "false")
        << ",\"q_max_source\":\"" << q_max_source() << "\",\"degrees\":[";
    for (uint32_t q = 0; q < num_degrees(); ++q) oss << (q ? "," : "") << q;
    oss << "]";
    oss << ",\"slots\":" << num_slots() << ",\"constructed\":" << num_constructed_ << ",\"derived\":" << num_derived_
        << ",\"nonzero_slots\":" << num_nonzero_ << ",\"eta_zero\":" << (eta_zero() ? "true" : "false")
        << ",\"fills\":" << num_fills_ << ",\"max_rings\":" << max_rings_ << ",\"terms\":" << num_terms_
        << ",\"max_abs_coeff\":" << max_abs_coeff_;
    oss << ",\"B_tab\":" << bound_B_tab_ << ",\"B\":" << bound_B_;
    oss << ",\"table_fnv64\":\"0x" << std::hex << table_fnv64() << std::dec << "\"}";
    return oss.str();
}

// Explicit instantiations (both element widths, as HeckeTauTable).
template class HeckePairHomotopyT<uint32_t>;
template class HeckePairHomotopyT<uint64_t>;

} // namespace hecke
