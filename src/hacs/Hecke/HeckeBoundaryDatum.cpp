// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions WP-4a -- the certified Gamma-level boundary datum tau_bd
// ([C, def:taubdry], [C, lem:taubdryexists], [C, obl:taubdry](i)-(iv)) and
// the B_N / Gamma_inf table ([C, prop:bdrycosets](1)-(3)). See
// HeckeBoundaryDatum.h for the contract references and the design notes.

#include "HeckeBoundaryDatum.h"

#include "HeckeModelDescriptor.h"
#include "../SimplicialCohomology.h" // SimplicialComplex
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hecke {

namespace {

// Parity (+1/-1) of a vertex-slot permutation restricted to slots 0..q
// (mirrors HeckeTauTable; combinatorial, never geometric -- invariant I8).
int8_t perm_parity(const std::array<uint8_t, 3>& perm, uint32_t q) {
    int inversions = 0;
    for (uint32_t i = 0; i + 1 <= q; ++i) {
        for (uint32_t j = i + 1; j <= q; ++j) {
            if (perm[i] > perm[j]) ++inversions;
        }
    }
    return (inversions % 2 == 0) ? int8_t(1) : int8_t(-1);
}

std::string mat_to_string(const Mat22& m) {
    std::ostringstream oss;
    oss << "[[" << m.a << "," << m.b << "],[" << m.c << "," << m.d << "]]";
    return oss.str();
}

std::string cusp_to_string(const CuspKey& c) {
    std::ostringstream oss;
    if (c.c == 0) oss << "inf";
    else oss << "(" << c.a << ":" << c.c << ")";
    return oss.str();
}

// p^e, overflow-checked.
int64_t ipow(int64_t p, uint32_t e) {
    int64_t r = 1;
    for (uint32_t k = 0; k < e; ++k) r = checked_mul<int64_t>(r, p);
    return r;
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

template<class Elt>
HeckeBoundaryTauTableT<Elt>::HeckeBoundaryTauTableT(const SimplicialComplex& K,
                                                    const SL2ZTessellationTables& model,
                                                    const HeckeCosetTablesT<Elt>& cosets,
                                                    TauBoundaryVertexPolicy policy)
    : K_(&K), model_(&model), cosets_(&cosets), policy_(policy), N_(cosets.N()), m_(cosets.num_cosets()),
      simplices_(K.get_simplices()) {
    RequireModel(SL2Z7VertexModel(), &K, "HeckeBoundaryTauTable");
    for (uint32_t v = 0; v < simplices_[0].size(); ++v) {
        if (simplices_[0][v] != std::vector<uint32_t>{ v }) {
            throw std::runtime_error("HeckeBoundaryTauTable: vertex list is not 0..n in order.");
        }
    }
    // g must be diag(1, N) ([C, prop:cosets], [C, prop:bdrycosets]).
    const Mat22& g = cosets_->g();
    if (!(g.a == 1 && g.b == 0 && g.c == 0 && g.d == static_cast<int64_t>(N_))) {
        throw std::runtime_error("HeckeBoundaryTauTable: coset tables do not carry g = diag(1, N).");
    }
    // Degree-generic store over every boundary degree the model reports.
    // (B2) forces a 1-dimensional boundary; a boundary with higher cells is
    // refused rather than silently truncated.
    const uint32_t num_deg = model_->boundary_dim() + 1;
    if (model_->boundary_dim() != 1) {
        throw std::runtime_error(
            "HeckeBoundaryTauTable: the boundary subcomplex must be 1-dimensional ([C, sec:boundarydatum] (B2)).");
    }
    for (uint32_t q = num_deg; q < simplices_.size(); ++q) {
        if (!model_->boundary_simplices(q).empty()) {
            throw std::runtime_error("HeckeBoundaryTauTable: boundary cells above the boundary dimension.");
        }
    }
    records_.assign(num_deg, {});
    for (uint32_t q = 0; q < num_deg; ++q) {
        records_[q].assign(model_->boundary_simplices(q).size(), std::vector<ChainRecord>(m_));
    }

    // [C, prop:bdrycosets](1)-(2): the left cosets of B_N and the bijection
    // with the bulk cosets (gamma_j through the cusp reduction).
    build_boundary_cosets();

    // [C, lem:taubdryexists](1), in ascending degree; one construction per
    // Gamma''-orbit, every other slot derived through the (T1bd) helper.
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
                else if (q == 1) construct_edge(j, sigma);
                else throw std::runtime_error("HeckeBoundaryTauTable: no construction above degree 1 ((B2)).");
                ++num_constructed_;
            }
        }
    }

    // Certified magnitude bound ([C, lem:bounds](5), [C, obl:backend]).
    bound_B_tab_ = 0;
    for (const auto& per_q : records_) {
        for (const auto& per_sigma : per_q) {
            for (const auto& rec : per_sigma) {
                for (const auto& term : rec.terms) {
                    const uint64_t norm = static_cast<uint64_t>(term.gamma.max_abs());
                    if (norm > bound_B_tab_) bound_B_tab_ = norm;
                }
            }
        }
    }
    const uint64_t four_N_sq = 4u * N_ * N_;
    bound_B_ = (bound_B_tab_ > four_N_sq) ? bound_B_tab_ : four_N_sq;

    // MANDATORY certification ([C, obl:taubdry](i)-(iv), invariant I8): a
    // table failing it is REJECTED -- throw, never warn, never skip.
    const uint32_t code = certify();
    if (code != 0) {
        std::ostringstream oss;
        oss << "HeckeBoundaryTauTable: certification FAILED (code " << code
            << ") for N = " << N_ << " (m = " << m_
            << ") -- table rejected, computation aborted ([C, obl:taubdry]).";
        throw std::runtime_error(oss.str());
    }
}

// ============================================================================
// Bookkeeping
// ============================================================================

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::local_index(uint32_t q, uint32_t sigma) const {
    if (q >= records_.size()) {
        throw std::out_of_range("HeckeBoundaryTauTable: degree out of range.");
    }
    const auto& list = model_->boundary_simplices(q);
    for (uint32_t k = 0; k < list.size(); ++k) {
        if (list[k] == sigma) return k;
    }
    throw std::invalid_argument("HeckeBoundaryTauTable: simplex is not a boundary simplex.");
}

template<class Elt>
void HeckeBoundaryTauTableT<Elt>::check_indices(uint32_t j, uint32_t q, uint32_t sigma) const {
    if (j == 0 || j > m_) {
        throw std::out_of_range("HeckeBoundaryTauTable: coset index j must be in [1, num_cosets()].");
    }
    (void)local_index(q, sigma);
}

template<class Elt>
const typename HeckeBoundaryTauTableT<Elt>::ChainRecord&
HeckeBoundaryTauTableT<Elt>::record(uint32_t j, uint32_t q, uint32_t sigma) const {
    check_indices(j, q, sigma);
    return records_[q][local_index(q, sigma)][j - 1];
}

template<class Elt>
typename HeckeBoundaryTauTableT<Elt>::ChainRecord&
HeckeBoundaryTauTableT<Elt>::record_mut(uint32_t j, uint32_t q, uint32_t sigma) {
    check_indices(j, q, sigma);
    return records_[q][local_index(q, sigma)][j - 1];
}

template<class Elt>
void HeckeBoundaryTauTableT<Elt>::accumulate_canonical(CanonMap& acc, const Mat22& gamma, uint32_t q,
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
std::vector<ChainTerm> HeckeBoundaryTauTableT<Elt>::canon_map_to_terms(const CanonMap& acc,
                                                                       uint32_t q) const {
    std::vector<ChainTerm> out;
    for (const auto& kv : acc) {
        if (kv.first[0] != static_cast<int64_t>(q)) {
            throw std::runtime_error("HeckeBoundaryTauTable: canonical map dimension mismatch.");
        }
        const int64_t c = kv.second;
        if (c == 0) continue;
        if (c > INT32_MAX || c < INT32_MIN) {
            throw HeckeOverflow("HeckeBoundaryTauTable: chain coefficient exceeds int32.");
        }
        out.push_back(ChainTerm{ static_cast<int32_t>(c),
                                 Mat22{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] },
                                 static_cast<uint32_t>(kv.first[1]) });
    }
    return out;
}

template<class Elt>
typename HeckeBoundaryTauTableT<Elt>::CanonMap
HeckeBoundaryTauTableT<Elt>::terms_to_canon_map(const std::vector<ChainTerm>& terms, uint32_t q) const {
    CanonMap acc;
    for (const auto& term : terms) {
        accumulate_canonical(acc, term.gamma, q, term.sigma_idx, term.coeff);
    }
    return acc;
}

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::face_simplex_index(uint32_t q, uint32_t sigma, uint32_t f) const {
    // d_f(sigma): delete the f-th vertex (increasing vertex order) and look
    // the (q-1)-simplex up in K's list. Vertex lists are 0..n in order, so
    // for q == 1 the vertex id is the 0-simplex index.
    const auto& verts = simplices_.at(q).at(sigma);
    if (f > q) throw std::runtime_error("HeckeBoundaryTauTable: face index out of range.");
    std::vector<uint32_t> face;
    for (uint32_t k = 0; k <= q; ++k) {
        if (k != f) face.push_back(verts[k]);
    }
    const auto& list = simplices_.at(q - 1);
    for (uint32_t k = 0; k < list.size(); ++k) {
        if (list[k] == face) return k;
    }
    throw std::runtime_error("HeckeBoundaryTauTable: face not found in the model complex.");
}

template<class Elt>
typename HeckeBoundaryTauTableT<Elt>::CanonMap
HeckeBoundaryTauTableT<Elt>::boundary_of_terms(const std::vector<ChainTerm>& terms, uint32_t q) const {
    // boundary(v_0..v_q) = sum_f (-1)^f (v_0..\hat v_f..v_q), the convention
    // of [C, sec:conventions] (as in HeckeTauTable::boundary_of_terms).
    if (q == 0) throw std::runtime_error("HeckeBoundaryTauTable: boundary_of_terms needs q >= 1.");
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
bool HeckeBoundaryTauTableT<Elt>::canon_maps_equal(const CanonMap& x, const CanonMap& y) {
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

// ============================================================================
// The single (T1bd) helper
// ============================================================================

template<class Elt>
std::vector<ChainTerm> HeckeBoundaryTauTableT<Elt>::eval_on_translate(const Records& recs,
                                                                      const Mat22& gamma, uint32_t q,
                                                                      uint32_t sigma) const {
    // gamma = h s_j, j = class_index_of(gamma), h in Gamma_0(N); the exact
    // integral conjugate g^{-1} h g comes from the ONE shared function
    // HeckeCosetTables::twisted_conjugate (the bulk tau uses the same).
    uint32_t j = 0;
    const Mat22 conj = cosets_->twisted_conjugate(gamma, j);
    check_indices(j, q, sigma);
    const ChainRecord& rec = recs[q][local_index(q, sigma)][j - 1];
    if (!rec.stored) {
        throw std::runtime_error(
            "HeckeBoundaryTauTable: (T1bd) helper hit an unbuilt slot (construction-order bug).");
    }
    CanonMap acc;
    for (const auto& term : rec.terms) {
        accumulate_canonical(acc, conj * term.gamma, q, term.sigma_idx, term.coeff);
    }
    return canon_map_to_terms(acc, q);
}

template<class Elt>
std::vector<ChainTerm> HeckeBoundaryTauTableT<Elt>::tau_on_translate(const Mat22& gamma, uint32_t q,
                                                                     uint32_t sigma) const {
    return eval_on_translate(records_, gamma, q, sigma);
}

// ============================================================================
// Components ([C, lem:bdrycomponent])
// ============================================================================

template<class Elt>
CuspKey HeckeBoundaryTauTableT<Elt>::slot_component(uint32_t j) const {
    if (j == 0 || j > m_) {
        throw std::out_of_range("HeckeBoundaryTauTable: coset index j must be in [1, num_cosets()].");
    }
    return CuspKey::of(cosets_->s(j));
}

template<class Elt>
CuspKey HeckeBoundaryTauTableT<Elt>::component_map(const Mat22& g, const CuspKey& c) {
    // g^{-1} c, computed through adj(g) = det(g) g^{-1} = [[d, -b],[-c, a]]:
    // the action on P^1(Q) is projective, so no rational number is formed.
    const Mat22 adj{ g.d, checked_neg<int64_t>(g.b), checked_neg<int64_t>(g.c), g.a };
    return CuspKey::apply(adj, c);
}

template<class Elt>
CuspKey HeckeBoundaryTauTableT<Elt>::chain_component(const std::vector<ChainTerm>& terms, bool& single_out) {
    single_out = false;
    if (terms.empty()) return CuspKey::infinity();
    const CuspKey c = CuspKey::of(terms[0].gamma);
    for (const auto& term : terms) {
        if (CuspKey::of(term.gamma) != c) return CuspKey::infinity();
    }
    single_out = true;
    return c;
}

// ============================================================================
// B_N / Gamma_inf ([C, prop:bdrycosets](1)-(3))
// ============================================================================

template<class Elt>
void HeckeBoundaryTauTableT<Elt>::build_boundary_cosets() {
    const int64_t p = static_cast<int64_t>(cosets_->p_hecke());
    const uint32_t nu = cosets_->r();
    if (ipow(p, nu) != static_cast<int64_t>(N_)) {
        throw std::runtime_error("HeckeBoundaryTauTable: N != p^nu (internal error).");
    }
    // (1): delta = [[p^u, b],[0, p^{nu-u}]], 0 <= b < p^u, primitive: for
    // 0 < u < nu both diagonal entries are divisible by p, so p must not
    // divide b; for u = 0 only b = 0; for u = nu every b mod N occurs.
    bcosets_.clear();
    for (uint32_t u = 0; u <= nu; ++u) {
        const int64_t a = ipow(p, u);
        const int64_t d = ipow(p, nu - u);
        for (int64_t b = 0; b < a; ++b) {
            if (u > 0 && u < nu && (b % p) == 0) continue;
            bcosets_.push_back(BoundaryCoset{ Mat22{ a, b, 0, d }, u, b });
        }
    }
    if (bcosets_.size() != m_) {
        throw std::runtime_error(
            "HeckeBoundaryTauTable: |B_N / Gamma_inf| != m ([C, prop:bdrycosets](1) count).");
    }
    // (2): for each bulk coset element delta_j = s_j^{-1} g the element
    // gamma_j with gamma_j . infinity = delta_j^{-1} . infinity = g^{-1} s_j
    // . infinity; then delta_j gamma_j is upper triangular, i.e. in B_N.
    bcoset_of_bulk_.assign(static_cast<size_t>(m_) + 1, 0);
    gamma_j_.assign(static_cast<size_t>(m_) + 1, Mat22::I());
    std::vector<bool> hit(m_, false);
    for (uint32_t j = 1; j <= m_; ++j) {
        const CuspKey target = component_map(cosets_->g(), slot_component(j));
        const Mat22 gred = CuspKey::reduce_to_infinity(target); // gred . target = infinity
        gamma_j_[j] = gred.inv_unimodular();                     // gamma_j . infinity = target
        const Mat22 dg = cosets_->delta(j) * gamma_j_[j];
        if (dg.c != 0) {
            throw std::runtime_error(
                "HeckeBoundaryTauTable: delta_j gamma_j is not upper triangular (internal error).");
        }
        const uint32_t idx = boundary_coset_index_of(dg);
        if (hit[idx]) {
            throw std::runtime_error(
                "HeckeBoundaryTauTable: j -> delta_j gamma_j Gamma_inf is not injective ([C, prop:bdrycosets](2)).");
        }
        hit[idx] = true;
        bcoset_of_bulk_[j] = idx;
    }
}

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::boundary_coset_index_of(const Mat22& delta_upper) const {
    if (delta_upper.c != 0 || delta_upper.det() != static_cast<int64_t>(N_)) {
        throw std::invalid_argument("HeckeBoundaryTauTable: not an upper-triangular element of det N.");
    }
    // delta Gamma_inf: the sign is absorbed by -I, b is determined mod a
    // (delta T^k = [[a, b + k a],[0, d]]).
    Mat22 d = delta_upper;
    if (d.a < 0) d = d.neg();
    const int64_t a = d.a;
    int64_t b = d.b % a;
    if (b < 0) b += a;
    for (uint32_t k = 0; k < bcosets_.size(); ++k) {
        if (bcosets_[k].delta.a == a && bcosets_[k].b == b && bcosets_[k].delta.d == d.d) return k;
    }
    throw std::invalid_argument("HeckeBoundaryTauTable: element is not in B_N (not primitive?).");
}

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::boundary_coset_of_bulk(uint32_t j) const {
    if (j == 0 || j > m_) throw std::out_of_range("HeckeBoundaryTauTable: coset index out of range.");
    return bcoset_of_bulk_[j];
}

template<class Elt>
const Mat22& HeckeBoundaryTauTableT<Elt>::gamma_j(uint32_t j) const {
    if (j == 0 || j > m_) throw std::out_of_range("HeckeBoundaryTauTable: coset index out of range.");
    return gamma_j_[j];
}

// ============================================================================
// Slot derivation through (T1bd)
// ============================================================================

template<class Elt>
bool HeckeBoundaryTauTableT<Elt>::try_derive(uint32_t j, uint32_t q, uint32_t sigma) {
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
            std::vector<ChainTerm> value = tau_on_translate(gamma, q, sigma2);
            const int8_t eps = perm_parity(perms[k], q);
            if (eps < 0) {
                for (auto& term : value) term.coeff = -term.coeff;
            }
            ChainRecord& rec = record_mut(j, q, sigma);
            rec.terms = std::move(value);
            rec.stored = true;
            rec.derived = true;
            return true;
        }
    }
    return false;
}

// ============================================================================
// [C, lem:taubdryexists](1): vertices, then edges
// ============================================================================

template<class Elt>
void HeckeBoundaryTauTableT<Elt>::construct_vertex(uint32_t j, uint32_t v) {
    // z = (g^{-1} s_j) . pos(v) through adj(g) = diag(N, 1) (projective, no
    // rationals); its cusp is the forced target c' = g^{-1} (s_j . infinity).
    const Mat22 adj_g{ static_cast<int64_t>(N_), 0, 0, 1 };
    const Mat22 M = adj_g * cosets_->s(j);
    QuadPoint z = model_->vertex_position(v).moebius(M);
    const CuspKey target = component_map(cosets_->g(), slot_component(j));
    const Mat22 gred = CuspKey::reduce_to_infinity(target);
    ENGINE_CHECK_MSG(CuspKey::apply(gred, target) == CuspKey::infinity(),
                     "HeckeBoundaryTauTable: cusp reduction did not reach infinity.");
    z = z.moebius(gred); // now on a horocycle around infinity
    ENGINE_CHECK_MSG(z.D == 1, "HeckeBoundaryTauTable: horocycle vertex image is not rational.");
    // The line vertex nearest to Re z (vertical projection onto the
    // horocycle at height Y); ties to the larger real part. T-equivariant,
    // hence independent of which gamma_red was used.
    int64_t P = model_->boundary_line_nearest(z.u, z.w);
    {
        // Tie diagnostic (recorded, gates nothing): is the vertex at P - 1
        // exactly as far from Re z as the one at P?  |z - r_P| vs |z - r_{P-1}|
        // as exact fractions (__int128, the same arithmetic as nearest()).
        using I = __int128;
        int64_t an = 0, ad = 1, bn = 0, bd = 1;
        model_->boundary_line_real_part(P, an, ad);
        model_->boundary_line_real_part(P - 1, bn, bd);
        // |z.u/z.w - an/ad| = |z.u*ad - an*z.w| / (z.w*ad), both denominators > 0.
        I da = static_cast<I>(z.u) * static_cast<I>(ad) - static_cast<I>(an) * static_cast<I>(z.w);
        if (da < 0) da = -da;
        I db = static_cast<I>(z.u) * static_cast<I>(bd) - static_cast<I>(bn) * static_cast<I>(z.w);
        if (db < 0) db = -db;
        if (da * static_cast<I>(bd) == db * static_cast<I>(ad)) ++num_vertex_ties_;
    }
    // [C, lem:taubdryexists](1) admits ANY vertex of the forced target
    // component; the policy selects which one (slot 3, WP-6).  The shift is
    // applied on the standard line, where the component is the whole line, so
    // the result is still a single vertex of the target component with
    // coefficient 1 -- (T3bd) holds by inspection, exactly as for P.
    if (policy_ == TauBoundaryVertexPolicy::NextLineVertex) P = checked_add<int64_t>(P, 1);
    const auto tv = model_->boundary_line_vertex(P); // (T^k, w)
    CanonMap acc;
    accumulate_canonical(acc, gred.inv_unimodular() * tv.first, 0, tv.second, 1);
    ChainRecord& rec = record_mut(j, 0, v);
    rec.terms = canon_map_to_terms(acc, 0);
    rec.stored = true;
}

template<class Elt>
void HeckeBoundaryTauTableT<Elt>::construct_edge(uint32_t j, uint32_t e) {
    const auto& verts = simplices_[1][e]; // {va, vb}, va < vb; boundary (vb) - (va)
    const uint32_t va = verts[0];
    const uint32_t vb = verts[1];
    const std::vector<ChainTerm>& A = record(j, 0, va).terms;
    const std::vector<ChainTerm>& B = record(j, 0, vb).terms;
    if (A.size() != 1 || A[0].coeff != 1 || B.size() != 1 || B[0].coeff != 1) {
        throw std::runtime_error("HeckeBoundaryTauTable: vertex slot is not a single coefficient-1 term.");
    }
    const Mat22& ga = A[0].gamma;
    const Mat22& gb = B[0].gamma;
    const CuspKey ca = CuspKey::of(ga);
    if (ca != CuspKey::of(gb)) {
        throw std::runtime_error(
            "HeckeBoundaryTauTable: the two endpoint images lie in different components (internal error).");
    }
    // Transport the component to the standard line and walk between the two
    // positions: the unique 1-chain of a line with boundary B - A
    // ([C, lem:taubdryexists](1), Z_1 = 0 by (B2)).
    const Mat22 gred = CuspKey::reduce_to_infinity(ca);
    const Mat22 gred_inv = gred.inv_unimodular();
    const int64_t Pa = model_->boundary_line_position(gred * ga, A[0].sigma_idx);
    const int64_t Pb = model_->boundary_line_position(gred * gb, B[0].sigma_idx);
    CanonMap acc;
    const int64_t lo = std::min(Pa, Pb);
    const int64_t hi = std::max(Pa, Pb);
    const int64_t dir = (Pa <= Pb) ? 1 : -1; // walk A -> B
    for (int64_t P = lo; P < hi; ++P) {
        const SL2ZTessellationTables::LineEdge le = model_->boundary_line_edge(P);
        accumulate_canonical(acc, gred_inv * le.gamma, 1, le.edge_idx,
                             dir * static_cast<int64_t>(le.coeff));
    }
    ChainRecord& rec = record_mut(j, 1, e);
    rec.terms = canon_map_to_terms(acc, 1);
    rec.stored = true;
}

// ============================================================================
// Accessors
// ============================================================================

template<class Elt>
const std::vector<ChainTerm>& HeckeBoundaryTauTableT<Elt>::tau(uint32_t j, uint32_t q,
                                                                uint32_t sigma) const {
    return record(j, q, sigma).terms;
}

template<class Elt>
bool HeckeBoundaryTauTableT<Elt>::slot_derived(uint32_t j, uint32_t q, uint32_t sigma) const {
    return record(j, q, sigma).derived;
}

// ============================================================================
// MANDATORY certification ([C, obl:taubdry](i)-(iv))
// ============================================================================

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::certify() const {
    return certify_impl(records_);
}

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::certify_with_override(uint32_t j, uint32_t q, uint32_t sigma,
                                                            const std::vector<ChainTerm>& replacement) const {
    check_indices(j, q, sigma);
    Records copy = records_;
    ChainRecord& rec = copy[q][local_index(q, sigma)][j - 1];
    rec.terms = replacement;
    rec.stored = true;
    return certify_impl(copy);
}

template<class Elt>
uint32_t HeckeBoundaryTauTableT<Elt>::certify_impl(const Records& recs) const {
    const uint32_t num_deg = static_cast<uint32_t>(recs.size());
    const Mat22& g = cosets_->g();

    // (a) structural, (iv) integrality, (i) components.
    for (uint32_t q = 0; q < num_deg; ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = recs[q][local][j - 1];
                if (!rec.stored) return 14;
                for (const auto& term : rec.terms) {
                    if (term.coeff == 0) return 13;
                    if (term.gamma.det() != 1) return 11;
                    if (!model_->is_boundary_simplex(q, term.sigma_idx)) return 16;
                    const KuSimplexKey key = model_->canonical_key(term.gamma, q, term.sigma_idx);
                    if (key.sigma_hat != term.sigma_idx || !(key.gamma_canon == term.gamma) ||
                        key.sign != 1) {
                        return 12;
                    }
                    if (!(term.coeff == 1 || term.coeff == -1)) return 17; // (iv): {0, +-1}
                }
                // (i): (T3bd) in degree 0 -- one simplex, coefficient 1.
                if (q == 0 && (rec.terms.size() != 1 || rec.terms[0].coeff != 1)) return 20;
                // (i): single component, and it is g^{-1} (s_j . infinity),
                // recomputed from g alone ([C, lem:bdrycomponent]).
                if (!rec.terms.empty()) {
                    bool single = false;
                    const CuspKey comp = chain_component(rec.terms, single);
                    if (!single) return 21;
                    if (comp != component_map(g, slot_component(j))) return 22;
                }
            }
        }
    }

    // (ii) (T2bd): boundary(tau_bd(s_j sigma)) == tau_bd(boundary(s_j sigma)),
    // the right side ONLY through the (T1bd) helper.
    for (uint32_t q = 1; q < num_deg; ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            const uint32_t sigma = list[local];
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = recs[q][local][j - 1];
                const CanonMap lhs = boundary_of_terms(rec.terms, q);
                CanonMap rhs;
                for (uint32_t f = 0; f <= q; ++f) {
                    const uint32_t face = face_simplex_index(q, sigma, f);
                    const std::vector<ChainTerm> ch = eval_on_translate(recs, cosets_->s(j), q - 1, face);
                    const int64_t sgn = (f % 2 == 0) ? 1 : -1;
                    for (const auto& term : ch) {
                        accumulate_canonical(rhs, term.gamma, q - 1, term.sigma_idx,
                                             sgn * static_cast<int64_t>(term.coeff));
                    }
                }
                if (!canon_maps_equal(lhs, rhs)) return 30;
            }
        }
    }

    // (iii) (T1bd) bookkeeping over EVERY stored transporter between
    // boundary simplices and every j (the {+-I} self-constraints included).
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
                            accumulate_canonical(rhs, term.gamma, q, term.sigma_idx,
                                                 static_cast<int64_t>(eps) *
                                                     static_cast<int64_t>(term.coeff));
                        }
                        const CanonMap lhs = terms_to_canon_map(recs[q][local][j - 1].terms, q);
                        if (!canon_maps_equal(lhs, rhs)) return 40;
                    }
                }
            }
        }
    }

    // (e) bounds ([C, lem:bounds](5), [C, obl:backend]): full recount, tier.
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
        // certify_with_override may change the attained bound; the recorded
        // bound must be exact on the object's own table.
        if (&recs == &records_ && recount != bound_B_tab_) return 50;
        const uint64_t four_N_sq = 4u * N_ * N_;
        const uint64_t B = (recount > four_N_sq) ? recount : four_N_sq;
        if (&recs == &records_ && B != bound_B_) return 50;
        if (B > 2147483647ull) return 51; // 2 B^2 >= 2^63: int64 tier violated
    }

    // (f) the B_N / Gamma_inf table and the bijection of [C, prop:bdrycosets](2).
    {
        if (bcosets_.size() != m_) return 60;
        const int64_t p = static_cast<int64_t>(cosets_->p_hecke());
        const uint32_t nu = cosets_->r();
        for (size_t k = 0; k < bcosets_.size(); ++k) {
            const BoundaryCoset& bc = bcosets_[k];
            const Mat22& d = bc.delta;
            if (d.c != 0 || d.det() != static_cast<int64_t>(N_)) return 61;
            if (bc.u > nu || d.a != ipow(p, bc.u) || d.d != ipow(p, nu - bc.u)) return 61;
            if (d.b != bc.b || bc.b < 0 || bc.b >= d.a) return 61;
            // primitive: gcd(a, b, d) = 1 <=> not (p | a and p | b and p | d)
            if (bc.u > 0 && bc.u < nu && (bc.b % p) == 0) return 62;
            for (size_t k2 = 0; k2 < k; ++k2) {
                if (bcosets_[k2].delta == d) return 63;
            }
        }
        std::vector<bool> hit(m_, false);
        for (uint32_t j = 1; j <= m_; ++j) {
            const Mat22& gj = gamma_j_[j];
            if (gj.det() != 1) return 64;
            if (CuspKey::of(gj) != component_map(g, slot_component(j))) return 64;
            const Mat22 dg = cosets_->delta(j) * gj;
            if (dg.c != 0) return 65;
            const uint32_t idx = boundary_coset_index_of(dg);
            if (idx != bcoset_of_bulk_[j] || hit[idx]) return 66;
            hit[idx] = true;
        }
        for (bool h : hit) if (!h) return 66;
    }

    return 0;
}

// ============================================================================
// Pretty printing
// ============================================================================

template<class Elt>
std::string HeckeBoundaryTauTableT<Elt>::to_string_offset(uint32_t offset) const {
    const std::string ind(offset, ' ');
    std::ostringstream oss;
    oss << ind << "HeckeBoundaryTauTable (certified boundary g-twisted chain datum, "
                  "[C, def:taubdry]/[C, lem:taubdryexists]):\n";
    oss << ind << "  g = diag(1, " << N_ << "), N = p^nu = " << N_ << " (p = " << cosets_->p_hecke()
        << ", nu = " << cosets_->r() << "), m = " << m_ << " cosets; solve regime = "
        << solve_regime() << " (integral = " << (integral() ? "true" : "false") << ")\n";
    oss << ind << "  B_tab = " << bound_B_tab_ << ", B = max(B_tab, 4 N^2) = " << bound_B_
        << " (int64 tier 2 B^2 < 2^63: " << (bound_B_ <= 2147483647ull ? "OK" : "VIOLATED") << ")\n";
    oss << ind << "  stored chains: " << num_slots() << " (" << num_constructed_ << " constructed, "
        << num_derived_ << " derived via (T1bd)); degrees 0.." << (num_degrees() - 1) << "\n";
    oss << ind << "  B_N / Gamma_inf (left cosets, canonical delta = [[p^u, b],[0, p^(nu-u)]]):\n";
    for (size_t k = 0; k < bcosets_.size(); ++k) {
        oss << ind << "    [" << k << "] u=" << bcosets_[k].u << " b=" << bcosets_[k].b << " delta="
            << mat_to_string(bcosets_[k].delta) << "\n";
    }
    oss << ind << "  bulk j -> boundary coset (delta_j gamma_j Gamma_inf), gamma_j:\n";
    for (uint32_t j = 1; j <= m_; ++j) {
        oss << ind << "    j=" << j << " s_j=" << mat_to_string(cosets_->s(j)) << " component "
            << cusp_to_string(slot_component(j)) << " -> "
            << cusp_to_string(component_map(cosets_->g(), slot_component(j))) << ", gamma_j="
            << mat_to_string(gamma_j_[j]) << ", coset [" << bcoset_of_bulk_[j] << "]\n";
    }
    const auto simplex_verts = [&](uint32_t q, uint32_t s) {
        std::string r = "{";
        const auto& verts = simplices_[q][s];
        for (size_t k = 0; k < verts.size(); ++k) {
            r += std::to_string(verts[k]);
            if (k + 1 < verts.size()) r += ",";
        }
        return r + "}";
    };
    oss << ind << "  tau_bd(s_j . i(sigma)) = sum_k n_k * gamma_k . i(sigma_k), terms canonical:\n";
    for (uint32_t q = 0; q < num_degrees(); ++q) {
        const auto& list = model_->boundary_simplices(q);
        for (uint32_t local = 0; local < list.size(); ++local) {
            const uint32_t sigma = list[local];
            for (uint32_t j = 1; j <= m_; ++j) {
                const ChainRecord& rec = records_[q][local][j - 1];
                oss << ind << "    [j=" << j << "][q=" << q << "][sigma=" << sigma << " "
                    << simplex_verts(q, sigma) << "]: ";
                if (rec.terms.empty()) {
                    oss << "0";
                } else {
                    for (size_t k = 0; k < rec.terms.size(); ++k) {
                        const auto& t = rec.terms[k];
                        if (k > 0) oss << " + ";
                        oss << "(" << t.coeff << ") * " << mat_to_string(t.gamma) << " . i("
                            << t.sigma_idx << " " << simplex_verts(q, t.sigma_idx) << ")";
                    }
                }
                if (rec.derived) oss << "   [derived via (T1bd)]";
                oss << "\n";
            }
        }
    }
    return oss.str();
}

// Explicit instantiations (both element widths, as HeckeTauTable).
const char* to_string(TauBoundaryVertexPolicy p) {
    switch (p) {
        case TauBoundaryVertexPolicy::NearestLineVertex: return "NearestLineVertex";
        case TauBoundaryVertexPolicy::NextLineVertex: return "NextLineVertex";
    }
    return "?";
}

// FNV-1a-64 over the whole stored table in slot order -- the same walk (and
// therefore the same value) as HeckePairHomotopyT::tau_boundary_table_fnv64().
template<class Elt>
uint64_t HeckeBoundaryTauTableT<Elt>::table_fnv64() const {
    engine::FnvHasher h;
    h.feed(N_);
    h.feed(m_);
    for (uint32_t q = 0; q < num_degrees(); ++q) {
        for (uint32_t sigma : model_->boundary_simplices(q)) {
            for (uint32_t j = 1; j <= m_; ++j) {
                h.feed(q);
                h.feed(sigma);
                h.feed(j);
                const std::vector<ChainTerm>& terms = record(j, q, sigma).terms;
                h.feed(terms.size());
                for (const ChainTerm& t : terms) {
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

template class HeckeBoundaryTauTableT<uint32_t>;
template class HeckeBoundaryTauTableT<uint64_t>;

} // namespace hecke
