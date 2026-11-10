// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S4 -- HeckeTauTable: certified g-twisted chain datum ([C, alg:tau]).
// See HeckeTauTable.h for the contract references and the design notes
// (in particular the I8 certification-over-geometry statement and the
// adj(g) projective trick).

#include "HeckeTauTable.h"

#include "HeckeBoundaryDatum.h" // slot-6 WP-4: the tau_bd table of TauVertexPolicy::BoundaryStrict
#include "HeckeIntegerSolve.h" // solve_integer_system (WP-8: the shared integer solver)
#include "HeckeModelDescriptor.h"
#include "HeckeOperatorOnK.h"   // (slot-13 WP-4) HeckeOperatorOptions::u_ell_low_rank
#include "../SimplicialCohomology.h" // SimplicialComplex
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace hecke {

namespace {

// Lexicographic (a,b,c,d) order on exact 2x2 matrices (mirrors S2's
// canonical-key order; used for the +-I translate normalization).
bool lex_less(const Mat22& x, const Mat22& y) {
    if (x.a != y.a) return x.a < y.a;
    if (x.b != y.b) return x.b < y.b;
    if (x.c != y.c) return x.c < y.c;
    return x.d < y.d;
}

// Parity (+1/-1) of a vertex-slot permutation restricted to slots 0..q
// (mirrors S2; combinatorial, never geometric -- invariant I8).
int8_t perm_parity(const std::array<uint8_t, 3>& perm, uint32_t q) {
    int inversions = 0;
    for (uint32_t i = 0; i + 1 <= q; ++i) {
        for (uint32_t j = i + 1; j <= q; ++j) {
            if (perm[i] > perm[j]) ++inversions;
        }
    }
    return (inversions % 2 == 0) ? int8_t(1) : int8_t(-1);
}

unsigned __int128 uabs128(__int128 x) {
    return (x < 0) ? (~static_cast<unsigned __int128>(x) + 1u)
                   : static_cast<unsigned __int128>(x);
}

unsigned __int128 gcd128(unsigned __int128 a, unsigned __int128 b) {
    while (b != 0) {
        unsigned __int128 t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Nearest-integer quotient of u / w (w > 0), deterministic tie rule
// "round the residue up" (the residue r in [0, w) with 2r >= w rounds up)
// -- same convention as S2's (P2) word reduction. Overflow-checked.
template <typename I>
I nearest_int_quotient(I u, I w) {
    I quot = u / w;
    I rem = checked_sub<I>(u, checked_mul<I>(quot, w));
    if (rem < 0) {
        quot = checked_sub<I>(quot, I(1));
        rem = checked_add<I>(rem, w);
    }
    if (checked_mul<I>(I(2), rem) >= w) quot = checked_add<I>(quot, I(1));
    return quot;
}

// The T-shift of the reduction is an int64 Mat22 on BOTH tiers (only the
// point widens, [C, thm:backend]). A wide-tier quotient that does not fit
// int64 would mean the reduction left the regime the transporter bound of
// [C, lem:bounds] covers -- a loud named failure, never a truncation.
inline int64_t narrow_quotient(int64_t q) { return q; }
inline int64_t narrow_quotient(__int128 q) {
    if (q > static_cast<__int128>(INT64_MAX) || q < static_cast<__int128>(INT64_MIN)) {
        throw HeckeOverflow(
            "HeckeTauTable: fundamental-domain T-shift exceeds the int64 transporter regime.");
    }
    return static_cast<int64_t>(q);
}

std::array<int64_t, 4> mat_key(const Mat22& m) { return { m.a, m.b, m.c, m.d }; }

std::string mat_to_string(const Mat22& m) {
    std::ostringstream oss;
    oss << "[[" << m.a << "," << m.b << "],[" << m.c << "," << m.d << "]]";
    return oss.str();
}

// Mod-ell dense Gaussian solve of A x = c (the permitted fallback of
// [C, alg:tau](3)). ModArithmetic::mod_mul is exact for ell < 2^32
// ([C, thm:backend], mod-ell backend box).
template<class Elt>
bool solve_mod_ell_system(const std::vector<std::vector<int64_t>>& A_in,
                          const std::vector<int64_t>& c, Elt ell,
                          std::vector<Elt>& x_out) {
    const size_t R = A_in.size();
    const size_t n = R ? A_in[0].size() : 0;
    const auto red = [&](int64_t v) -> Elt {
        int64_t r = v % static_cast<int64_t>(ell);
        if (r < 0) r += static_cast<int64_t>(ell);
        return static_cast<Elt>(r);
    };
    std::vector<std::vector<Elt>> M(R, std::vector<Elt>(n, 0));
    std::vector<Elt> b(R, 0);
    for (size_t r = 0; r < R; ++r) {
        for (size_t j = 0; j < n; ++j) M[r][j] = red(A_in[r][j]);
        b[r] = red(c[r]);
    }
    std::vector<std::pair<size_t, size_t>> pivots;
    size_t row = 0;
    for (size_t col = 0; col < n && row < R; ++col) {
        size_t prow = SIZE_MAX;
        for (size_t r = row; r < R; ++r) {
            if (M[r][col] != 0) { prow = r; break; }
        }
        if (prow == SIZE_MAX) continue;
        std::swap(M[prow], M[row]);
        std::swap(b[prow], b[row]);
        const Elt inv = ModArithmetic::mod_inv_w(M[row][col], ell);
        for (size_t j = col; j < n; ++j) M[row][j] = ModArithmetic::mod_mul_w(M[row][j], inv, ell);
        b[row] = ModArithmetic::mod_mul_w(b[row], inv, ell);
        for (size_t r = 0; r < R; ++r) {
            if (r == row || M[r][col] == 0) continue;
            const Elt f = M[r][col];
            for (size_t j = col; j < n; ++j) {
                M[r][j] = ModArithmetic::mod_sub_w(M[r][j], ModArithmetic::mod_mul_w(f, M[row][j], ell), ell);
            }
            b[r] = ModArithmetic::mod_sub_w(b[r], ModArithmetic::mod_mul_w(f, b[row], ell), ell);
        }
        pivots.emplace_back(row, col);
        ++row;
    }
    for (size_t r = row; r < R; ++r) {
        if (b[r] != 0) return false; // inconsistent
    }
    x_out.assign(n, 0);
    for (const auto& pv : pivots) x_out[pv.second] = b[pv.first];
    return true;
}

} // namespace

// ============================================================================
// Construction
// ============================================================================

template<class Elt>
HeckeTauTableT<Elt>::HeckeTauTableT(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                             const HeckeCosetTablesT<Elt>& cosets, Elt ell, uint32_t n_sym,
                             TauVertexPolicy policy, TauTableContents contents)
    : K_(&K), model_(&model), cosets_(&cosets), ell_(ell), n_sym_(n_sym),
      policy_(policy), contents_(contents), N_(cosets.N()), m_(cosets.num_cosets()),
      simplices_(K.get_simplices()) {
    if (policy_ == TauVertexPolicy::BoundaryStrict) {
        throw std::invalid_argument(
            "HeckeTauTable: TauVertexPolicy::BoundaryStrict needs the boundary table "
            "(use the constructor taking a HeckeBoundaryTauTable, [C, prop:strictpair]).");
    }
    construct_and_certify();
}

template<class Elt>
HeckeTauTableT<Elt>::HeckeTauTableT(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                             const HeckeCosetTablesT<Elt>& cosets, Elt ell, uint32_t n_sym,
                             const HeckeBoundaryTauTableT<Elt>& taub)
    : K_(&K), model_(&model), cosets_(&cosets), ell_(ell), n_sym_(n_sym),
      policy_(TauVertexPolicy::BoundaryStrict), taub_(&taub), N_(cosets.N()), m_(cosets.num_cosets()),
      simplices_(K.get_simplices()) {
    // [C, prop:strictpair] needs tau_bd on the SAME g, Gamma'' and s_j; the
    // identity is re-asserted by certify() clause (g), code 71.
    if (&taub.cosets() != &cosets || &taub.model() != &model) {
        throw std::invalid_argument(
            "HeckeTauTable (BoundaryStrict): the boundary table is not built on the same coset "
            "tables and model ([C, prop:strictpair]).");
    }
    construct_and_certify();
}

template<class Elt>
void HeckeTauTableT<Elt>::construct_and_certify() {
    if (ell_ < 2) throw std::runtime_error("HeckeTauTable: ell must be >= 2.");
    // (ELL-W audit F3) Pre-flight of clause (e): B >= 4 N^2 a priori, so when even B = 4 N^2 violates the wide clause
    // (code 52: 8 N^3 > 2^63 - 1, i.e. N >= 2^20) the table is refused BEFORE any work, with certify()'s own message.  Only
    // code 52 is pre-flighted: the rollback / ForceNarrow refusal (51) keeps its pre-ELL-W place (after the build).
    {
        const bool allowed = tau_wide_tier_enabled() && current_exact_tier() != ExactTier::ForceNarrow;
        const unsigned __int128 four_N_sq = static_cast<unsigned __int128>(4) * N_ * N_;
        if (allowed && tau_bound_clause(N_, four_N_sq, true) == 52) {
            std::ostringstream oss;
            oss << "HeckeTauTable: certification FAILED (code 52) for p^r = N = " << N_ << ", ell = " << ell_
                << ", n_sym = " << n_sym_
                << " -- refused before the build: 2 N (4 N^2) > 2^63 - 1, outside the wide tier ([C, lem:bounds](7)).";
            throw std::runtime_error(oss.str());
        }
    }
    // Structural checks: this class is specific to the fixed 7-vertex model
    // (matrix-engine Phase 6, audit item 7.18) -- asserted against the MODEL
    // DESCRIPTOR, not the literals 7/11/5 (same shape assertions as
    // SL2ZTessellationTables; the predicate is unchanged).
    RequireModel(SL2Z7VertexModel(), K_, "HeckeTauTable");
    for (uint32_t v = 0; v < 7; ++v) {
        if (simplices_[0][v] != std::vector<uint32_t>{ v }) {
            throw std::runtime_error("HeckeTauTable: vertex list is not 0..6 in order.");
        }
    }
    for (uint32_t e = 0; e < simplices_[1].size(); ++e) {
        const auto& verts = simplices_[1][e];
        if (verts.size() != 2 || verts[0] >= verts[1]) {
            throw std::runtime_error("HeckeTauTable: edge list not sorted/canonical.");
        }
        edge_index_[{ verts[0], verts[1] }] = e;
    }
    // g must be diag(1, N) ([C, prop:cosets]); N fits comfortably in int64.
    const Mat22& g = cosets_->g();
    if (!(g.a == 1 && g.b == 0 && g.c == 0 && g.d == static_cast<int64_t>(N_))) {
        throw std::runtime_error("HeckeTauTable: coset tables do not carry g = diag(1, N).");
    }

    // iota_K on the seven vertices, read off the shipped exact coordinates
    // (z -> -conj(z), i.e. u -> -u on a QuadPoint), as [C, lem:iotamodel]
    // prescribes; used only by the tie-break of choose_vertex(). It must be
    // an involution of the vertex set with exactly one fixed vertex.
    iota_fixed_vertex_ = 7;
    for (uint32_t v = 0; v < 7; ++v) {
        const QuadPoint P = model_->vertex_position(v);
        const QuadPoint img(checked_neg<int64_t>(P.u), P.v, P.w, P.D);
        uint32_t hit = 7;
        for (uint32_t w = 0; w < 7; ++w) {
            if (model_->vertex_position(w) == img) { hit = w; break; }
        }
        ENGINE_CHECK_MSG(hit < 7, "HeckeTauTable: the vertex set is not stable under "
                                  "z -> -conj(z) ([C, lem:iotamodel], hypothesis (I1)).");
        iota_perm_[v] = hit;
        if (hit == v) {
            ENGINE_CHECK_MSG(iota_fixed_vertex_ == 7,
                             "HeckeTauTable: iota_K fixes more than one vertex.");
            iota_fixed_vertex_ = v;
        }
    }
    ENGINE_CHECK_MSG(iota_fixed_vertex_ < 7, "HeckeTauTable: iota_K fixes no vertex.");
    for (uint32_t v = 0; v < 7; ++v) {
        ENGINE_CHECK_MSG(iota_perm_[iota_perm_[v]] == v,
                         "HeckeTauTable: iota_K is not an involution on the vertices.");
    }
    // (S23 G2-F) A ChainsOnly table allocates no xi-term table (xi_ stays empty) and builds no rho pool below.
    const bool full = contents_ == TauTableContents::Full;
    records_.assign(3, {});
    if (full) xi_.assign(3, {});
    for (uint32_t q = 0; q < 3; ++q) {
        records_[q].assign(simplices_[q].size(), std::vector<ChainRecord>(m_));
        if (full) xi_[q].assign(simplices_[q].size(), std::vector<std::vector<XiTermT<Elt>>>(m_));
    }

    // [C, prop:strictpair] step (B) (BoundaryStrict only; a no-op otherwise),
    // then [C, alg:tau] steps 1-3, which skip every slot (B) pre-assigned.
    if (policy_ == TauVertexPolicy::BoundaryStrict) build_strict_boundary();
    build_vertices();
    build_edges();
    build_triangles();

    // Certified magnitude bound ([C, lem:bounds](5), [C, obl:backend]).
    bound_B_tab_ = 0;
    for (uint32_t q = 0; q < 3; ++q) {
        for (const auto& per_sigma : records_[q]) {
            for (const auto& rec : per_sigma) {
                for (const auto& term : rec.terms) {
                    const uint64_t norm = static_cast<uint64_t>(term.gamma.max_abs());
                    if (norm > bound_B_tab_) bound_B_tab_ = norm;
                }
            }
        }
    }
    // (slot-24 ELL-W) 4 N^2 formed in 128 bits (N < 2^32, so 4 N^2 < 2^66 is exact there); B = max(B_tab, 4 N^2) is
    // then saturated into the uint64 member (a B >= 2^64 is refused by clause (e) anyway).  For N <= 2^31 - 1 the
    // value is the old 4u * N_ * N_ bit for bit.
    {
        const unsigned __int128 four_N_sq = static_cast<unsigned __int128>(4) * N_ * N_;
        const unsigned __int128 B = (static_cast<unsigned __int128>(bound_B_tab_) > four_N_sq) ? bound_B_tab_ : four_N_sq;
        bound_B_ = (B >> 64) ? UINT64_MAX : static_cast<uint64_t>(B);
    }
    // (slot-24 ELL-W) The tier this table is certified in ([C, lem:bounds](5), [C, thm:backend]): narrow iff
    // 2 B^2 < 2^63 (B <= 2^31 - 1, the pre-ELL-W criterion verbatim); otherwise the wide tier -- if it may run on this
    // thread (HECKE_TAU_WIDE != "0" and the exact tier is not ForceNarrow).  Recorded here, so that certify() decides
    // by the table's own build, not by the tier in force when a validation re-asserts it.
    wide_allowed_ = tau_wide_tier_enabled() && current_exact_tier() != ExactTier::ForceNarrow;
    wide_tier_ = bound_B_ > 2147483647ull;

    // Eagerly-reduced Xi-term table ([C, eq:xiexplicit]), built once -- on a Full table only (S23 G2-F: the
    // records above are the same on both kinds; nothing below step 3 feeds back into them).
    if (full) build_xi_terms();

    // Step 4: MANDATORY certification ([C, obl:tau], invariant I8). A table
    // failing certification is REJECTED: throw, never warn, never skip.
    const uint32_t code = certify();
    if (code != 0) {
        std::ostringstream oss;
        oss << "HeckeTauTable: certification FAILED (code " << code
            << ") for p^r = N = " << N_ << ", ell = " << ell_ << ", n_sym = " << n_sym_
            << " -- table rejected, computation aborted ([C, obl:tau]).";
        throw std::runtime_error(oss.str());
    }
}

// ============================================================================
// Canonical chain bookkeeping
// ============================================================================

template<class Elt>
void HeckeTauTableT<Elt>::accumulate_canonical(CanonMap& acc, const Mat22& gamma, uint32_t q,
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
std::vector<ChainTerm> HeckeTauTableT<Elt>::canon_map_to_terms(const CanonMap& acc, uint32_t q,
                                                         bool mod_ell) const {
    std::vector<ChainTerm> out;
    for (const auto& kv : acc) {
        if (kv.first[0] != static_cast<int64_t>(q)) {
            throw std::runtime_error("HeckeTauTable: canonical map dimension mismatch.");
        }
        int64_t c = kv.second;
        if (mod_ell) {
            c %= static_cast<int64_t>(ell_);
            if (c < 0) c += static_cast<int64_t>(ell_);
            if (c > static_cast<int64_t>(ell_ / 2)) c -= static_cast<int64_t>(ell_); // balanced
        }
        if (c == 0) continue;
        if (c > INT32_MAX || c < INT32_MIN) {
            throw HeckeOverflow("HeckeTauTable: chain coefficient exceeds int32.");
        }
        out.push_back(ChainTerm{ static_cast<int32_t>(c),
                                 Mat22{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] },
                                 static_cast<uint32_t>(kv.first[1]) });
    }
    return out;
}

template<class Elt>
typename HeckeTauTableT<Elt>::CanonMap HeckeTauTableT<Elt>::terms_to_canon_map(const std::vector<ChainTerm>& terms,
                                                          uint32_t q) const {
    CanonMap acc;
    for (const auto& term : terms) {
        accumulate_canonical(acc, term.gamma, q, term.sigma_idx, term.coeff);
    }
    return acc;
}

template<class Elt>
typename HeckeTauTableT<Elt>::CanonMap HeckeTauTableT<Elt>::boundary_of_terms(const std::vector<ChainTerm>& terms,
                                                         uint32_t q) const {
    // boundary(v_0..v_q) = sum_j (-1)^j (v_0..\hat v_j..v_q), consistent with
    // the cochain convention (d^q f)(tau') = sum_j (-1)^j f(boundary_j tau')
    // of [C, sec:conventions] (S4 risk list).
    if (q == 0 || q > 2) throw std::runtime_error("HeckeTauTable: boundary_of_terms needs q in {1,2}.");
    CanonMap acc;
    for (const auto& term : terms) {
        for (uint32_t j = 0; j <= q; ++j) {
            const uint32_t face = face_simplex_index(q, term.sigma_idx, j);
            const int64_t sgn = (j % 2 == 0) ? 1 : -1;
            accumulate_canonical(acc, term.gamma, q - 1, face,
                                 sgn * static_cast<int64_t>(term.coeff));
        }
    }
    return acc;
}

template<class Elt>
bool HeckeTauTableT<Elt>::canon_maps_equal(const CanonMap& x, const CanonMap& y, bool mod_ell) const {
    const auto zero = [&](int64_t d) -> bool {
        if (!mod_ell) return d == 0;
        return (d % static_cast<int64_t>(ell_)) == 0;
    };
    for (const auto& kv : x) {
        const auto it = y.find(kv.first);
        const int64_t other = (it == y.end()) ? 0 : it->second;
        if (!zero(kv.second - other)) return false;
    }
    for (const auto& kv : y) {
        if (x.find(kv.first) == x.end() && !zero(kv.second)) return false;
    }
    return true;
}

template<class Elt>
uint32_t HeckeTauTableT<Elt>::face_simplex_index(uint32_t q, uint32_t sigma, uint32_t j) const {
    const auto& verts = simplices_.at(q).at(sigma);
    if (j > q) throw std::runtime_error("HeckeTauTable: face index out of range.");
    if (q == 1) {
        // faces of (a,b): j=0 -> (b), j=1 -> (a); vertex list is 0..6 in
        // order, so the vertex id IS the 0-simplex index.
        return (j == 0) ? verts[1] : verts[0];
    }
    // q == 2: remove the j-th vertex, look the edge up.
    std::vector<uint32_t> face;
    for (uint32_t k = 0; k < 3; ++k) {
        if (k != j) face.push_back(verts[k]);
    }
    const auto it = edge_index_.find({ face[0], face[1] });
    if (it == edge_index_.end()) {
        throw std::runtime_error("HeckeTauTable: face edge not found in the model complex.");
    }
    return it->second;
}

template<class Elt>
const typename HeckeTauTableT<Elt>::ChainRecord& HeckeTauTableT<Elt>::record(uint32_t i, uint32_t q,
                                                        uint32_t sigma) const {
    check_indices(i, q, sigma);
    return records_[q][sigma][i - 1];
}

template<class Elt>
typename HeckeTauTableT<Elt>::ChainRecord& HeckeTauTableT<Elt>::record_mut(uint32_t i, uint32_t q, uint32_t sigma) {
    check_indices(i, q, sigma);
    return records_[q][sigma][i - 1];
}

template<class Elt>
void HeckeTauTableT<Elt>::check_indices(uint32_t i, uint32_t q, uint32_t sigma) const {
    if (i == 0 || i > m_) {
        throw std::out_of_range("HeckeTauTable: coset index i must be in [1, num_cosets()].");
    }
    if (q > 2 || sigma >= simplices_[q].size()) {
        throw std::out_of_range("HeckeTauTable: simplex index out of range.");
    }
}

// ============================================================================
// The shared (T1) translation helper -- THE single implementation, used by
// construction (slot derivation) AND certification (boundary RHS and the
// twisted-equivariance bookkeeping). [C, def:tau] (T1) with (P1)/(P4).
// ============================================================================

template<class Elt>
std::vector<ChainTerm> HeckeTauTableT<Elt>::eval_tau_on_translate(const Mat22& gamma, uint32_t q,
                                                            uint32_t sigma,
                                                            bool& mod_ell_out) const {
    // gamma = h * s_i with i = class_index_of(gamma), h in Gamma_0(N) ((P1));
    // then tau(gamma . i(sigma)) = (g^{-1} h g) . tau(s_i . i(sigma)). The
    // split and the exact integral conjugate g^{-1} h g = [[h.a, N h.b],
    // [h.c / N, h.d]] are HeckeCosetTables::twisted_conjugate, the ONE (T1)
    // step shared with the boundary datum (WP-4a); value-preserving
    // refactor of the former inline body, gated by GoldenRegressionValidation.
    uint32_t i = 0;
    const Mat22 conj = cosets_->twisted_conjugate(gamma, i);
    const ChainRecord& rec = record(i, q, sigma);
    if (!rec.stored) {
        throw std::runtime_error(
            "HeckeTauTable: (T1) helper hit an unbuilt slot (construction-order bug).");
    }
    CanonMap acc;
    for (const auto& term : rec.terms) {
        // (slot-24 ELL-W) conj and gamma_t are two stored-class operands: the wide tier forms the product in
        // __int128 after an int64 overflow (byte-identical wherever int64 completes; ExactInt.h).
        accumulate_canonical(acc, mul_tiered(conj, term.gamma), q, term.sigma_idx, term.coeff);
    }
    mod_ell_out = rec.mod_ell;
    return canon_map_to_terms(acc, q, rec.mod_ell);
}

// ============================================================================
// Slot derivation through (T1) (one geometric construction per
// Gamma''-orbit; every Gamma''-related slot is derived -- see the header
// note on the non-injectivity of (i, sigma) -> Gamma''\K^u).
// ============================================================================

template<class Elt>
bool HeckeTauTableT<Elt>::try_derive(uint32_t i, uint32_t q, uint32_t sigma) {
    const uint32_t count = static_cast<uint32_t>(simplices_[q].size());
    for (uint32_t sigma2 = 0; sigma2 < count; ++sigma2) {
        if (!model_->same_groupoid_orbit(q, sigma, sigma2)) continue;
        const auto& ts = model_->transporter(q, sigma, sigma2);
        const auto& perms = model_->transporter_vertex_perms(q, sigma, sigma2);
        for (size_t k = 0; k < ts.size(); ++k) {
            // i(sigma) = t . i(sigma2)  ==>  s_i . i(sigma) = (s_i t) . i(sigma2).
            const Mat22 gamma = cosets_->s(i) * ts[k];
            const uint32_t j = cosets_->class_index_of(gamma);
            if (j == i && sigma2 == sigma) continue; // self (a constraint, not a derivation)
            const ChainRecord& src = records_[q][sigma2][j - 1];
            if (!src.stored) continue;
            bool src_mod = false;
            std::vector<ChainTerm> value = eval_tau_on_translate(gamma, q, sigma2, src_mod);
            // Orientation: i(sigma)_or = eps_t * t . i(sigma2)_or with eps_t
            // the parity of t's vertex correspondence (sign-tracked, S4 risk
            // list; identity in this model but carried generally).
            const int8_t eps = perm_parity(perms[k], q);
            if (eps < 0) {
                for (auto& term : value) term.coeff = -term.coeff;
            }
            ChainRecord& rec = record_mut(i, q, sigma);
            rec.terms = std::move(value);
            rec.mod_ell = src_mod;
            rec.stored = true;
            rec.derived = true;
            return true;
        }
    }
    return false;
}

// ============================================================================
// Step 1: vertices ([C, alg:tau](1))
// ============================================================================

template<class Elt>
template<typename Pt>
Mat22 HeckeTauTableT<Elt>::reduce_to_fundamental_domain(Pt& z) const {
    // Standard reduction: T-shift to |Re| <= 1/2 (block shift by the
    // nearest-integer quotient -- equivalent to iterated T^{-+1}), S while
    // |z| < 1; all branch decisions through the S1 predicates
    // (cmp_abs_re_half / cmp_norm_one), all arithmetic overflow-checked.
    Mat22 gred = Mat22::I();
    const Mat22 S = Mat22::S();
    uint32_t guard = 0;
    while (true) {
        if (++guard > 100000u) {
            throw HeckeInternalError("HeckeTauTable: fundamental-domain reduction failed to terminate.");
        }
        const int64_t quot = narrow_quotient(nearest_int_quotient(z.u, z.w));
        if (quot != 0) {
            const Mat22 Tshift{ 1, checked_neg<int64_t>(quot), 0, 1 };
            z = z.moebius(Tshift);
            gred = Tshift * gred;
        }
        if (z.cmp_norm_one() < 0) { // |z| < 1: apply S and re-normalize Re
            z = z.moebius(S);
            gred = S * gred;
            continue;
        }
        break;
    }
    if (z.cmp_abs_re_half() > 0 || z.cmp_norm_one() < 0) {
        throw std::runtime_error("HeckeTauTable: reduction postcondition violated (internal error).");
    }
    return gred;
}

template<class Elt>
template<typename Pt>
int HeckeTauTableT<Elt>::compare_vertex_distances(const Pt& z, uint32_t a, uint32_t b) const {
    // Exact sign of dist^2(z, P_a) - dist^2(z, P_b) (squared euclidean
    // distance on cleared denominators). For P = (u',v',w',D'):
    //   dist^2(z,P) * (w w')^2 = (u w' - u' w)^2 + D v^2 w'^2 + D' v'^2 w^2
    //                            - 2 v v' w w' sqrt(D D'),
    // rational when D == D' (sqrt(D^2) = D), a sqrt(3) term when {D,D'}={1,3}.
    // The final sign is routed through the S1 predicate sign_lin_sqrt3 after
    // gcd reduction (mixed-D requirement of the S4 spec).
    using I = __int128;
    const auto part = [&](const QuadPoint& P, I& R, I& s3) {  // vertex positions are int64 constants
        const I uwp = checked_sub<I>(checked_mul<I>(static_cast<I>(z.u), static_cast<I>(P.w)),
                                     checked_mul<I>(static_cast<I>(P.u), static_cast<I>(z.w)));
        R = checked_mul<I>(uwp, uwp);
        const I t1 = checked_mul<I>(checked_mul<I>(static_cast<I>(z.D),
                                                   checked_mul<I>(static_cast<I>(z.v), static_cast<I>(z.v))),
                                    checked_mul<I>(static_cast<I>(P.w), static_cast<I>(P.w)));
        const I t2 = checked_mul<I>(checked_mul<I>(static_cast<I>(P.D),
                                                   checked_mul<I>(static_cast<I>(P.v), static_cast<I>(P.v))),
                                    checked_mul<I>(static_cast<I>(z.w), static_cast<I>(z.w)));
        R = checked_add<I>(R, checked_add<I>(t1, t2));
        const I cross = checked_mul<I>(static_cast<I>(2),
                        checked_mul<I>(checked_mul<I>(static_cast<I>(z.v), static_cast<I>(P.v)),
                                       checked_mul<I>(static_cast<I>(z.w), static_cast<I>(P.w))));
        if (z.D == P.D) {
            R = checked_sub<I>(R, checked_mul<I>(cross, static_cast<I>(z.D)));
            s3 = 0;
        } else {
            s3 = checked_sub<I>(0, cross); // coefficient of sqrt(3)
        }
    };
    const QuadPoint Pa = model_->vertex_position(a);
    const QuadPoint Pb = model_->vertex_position(b);
    I Ra = 0, sa = 0, Rb = 0, sb = 0;
    part(Pa, Ra, sa);
    part(Pb, Rb, sb);
    const I wa2 = checked_mul<I>(static_cast<I>(Pa.w), static_cast<I>(Pa.w));
    const I wb2 = checked_mul<I>(static_cast<I>(Pb.w), static_cast<I>(Pb.w));
    I A = checked_sub<I>(checked_mul<I>(Ra, wb2), checked_mul<I>(Rb, wa2));
    I B = checked_sub<I>(checked_mul<I>(sa, wb2), checked_mul<I>(sb, wa2));
    if (A == 0 && B == 0) return 0;
    unsigned __int128 g = gcd128(uabs128(A), uabs128(B));
    if (g > 1) { A /= static_cast<I>(g); B /= static_cast<I>(g); }
    if (A > INT64_MAX || A < INT64_MIN || B > INT64_MAX || B < INT64_MIN) {
        throw HeckeOverflow(
            "HeckeTauTable: nearest-vertex comparison exceeds int64 after gcd reduction.");
    }
    return QuadPoint::sign_lin_sqrt3(static_cast<int64_t>(A), static_cast<int64_t>(B));
}

template<class Elt>
template<typename Pt>
uint32_t HeckeTauTableT<Elt>::choose_vertex(const Pt& z_red) const {
    // Exact position hit first (deterministic for every policy; for slots
    // with a nontrivial Gamma''-stabilizer the reduced point is exactly
    // elliptic and this choice is FORCED by (T1) -- see the header note).
    // [C, alg:tau](1a).
    for (uint32_t v = 0; v < 7; ++v) {
        if (z_red.equals_narrow(model_->vertex_position(v))) return v;
    }
    if (policy_ == TauVertexPolicy::FixedV6) return 6;

    // The exact set S of vertices at minimal euclidean distance from the
    // reduced point ([C, alg:tau](1b)): every comparison is the certified
    // predicate compare_vertex_distances, so S is exact.
    uint32_t best = 0;
    for (uint32_t cand = 1; cand < 7; ++cand) {
        if (compare_vertex_distances(z_red, best, cand) > 0) best = cand;
    }
    std::array<bool, 7> tied{};
    uint32_t tied_count = 0;
    for (uint32_t v = 0; v < 7; ++v) {
        tied[v] = (v == best) || (compare_vertex_distances(z_red, best, v) == 0);
        if (tied[v]) ++tied_count;
    }

    uint32_t chosen = best;
    if (tied_count > 1) {
        if (policy_ == TauVertexPolicy::GeometricNearestLegacy) {
            // Pre-2026-09-16 canonical tau: ties broken by ascending vertex
            // index (`best` is the first index attaining the minimum). NOT
            // iota_K-equivariant ([C, rmk:iotatausym]); kept for the record.
            chosen = best;
        } else {
            // iota_K-equivariant tie-break ([C, alg:tau](1b), owner decision of
            // 2026-09-16; iota_K is the vertex involution of [C, lem:iotamodel],
            // read off the shipped coordinates in the constructor):
            //   * if S contains a full iota_K-orbit {v, iota_K v} (a swapped
            //     pair, or the fixed vertex v6 itself), take v6 -- the unique
            //     iota_K-fixed vertex of K. This is FORCED on the mirror
            //     Re z = 0, where S is iota_K-stable and no member of a swapped
            //     pair can be chosen equivariantly;
            //   * otherwise S meets every orbit in at most one vertex: take the
            //     vertex lying in the orbit of smallest index
            //     ({v0,v3} < {v1,v2} < {v4,v5} < {v6}).
            // Both branches commute with iota_K, so the choice function c
            // satisfies c(iota_K S) = iota_K c(S) -- the hypothesis of the
            // vertex-level symmetry of [C, prop:iotacommutes](3b).
            bool full_orbit = false;
            for (uint32_t v = 0; v < 7; ++v) {
                if (tied[v] && tied[iota_perm_[v]]) { full_orbit = true; break; }
            }
            if (full_orbit) {
                chosen = iota_fixed_vertex_;
            } else {
                // Orbit index := the smaller of the two member indices.
                chosen = 7;
                uint32_t best_orbit = 7;
                for (uint32_t v = 0; v < 7; ++v) {
                    if (!tied[v]) continue;
                    const uint32_t orbit = (iota_perm_[v] < v) ? iota_perm_[v] : v;
                    if (orbit < best_orbit) { best_orbit = orbit; chosen = v; }
                }
                ENGINE_CHECK(chosen < 7);
            }
        }
    }

    return chosen;
}

template<class Elt>
SL2ZTessellationTables::PathTemplateFamily HeckeTauTableT<Elt>::template_family() const {
    // The legacy policy is the pre-2026-09-16 canonical tau in full: legacy
    // tie-break AND legacy BFS templates. Every other policy uses the
    // canonical tree-routed templates ([C, alg:tau](2)).
    return (policy_ == TauVertexPolicy::GeometricNearestLegacy)
               ? SL2ZTessellationTables::PathTemplateFamily::LegacyBfs
               : SL2ZTessellationTables::PathTemplateFamily::Canonical;
}

template<class Elt>
template<typename Pt>
void HeckeTauTableT<Elt>::step1_vertex_on_tier(const Mat22& M, uint32_t v,
                                               Mat22& gred_out, uint32_t& w_out) const {
    Pt z = Pt::widen(model_->vertex_position(v)).moebius(M);
    const Mat22 gred = reduce_to_fundamental_domain(z);
    const uint32_t w = choose_vertex(z);
    gred_out = gred; // written only on success: the Auto retry sees nothing
    w_out = w;
}

template<class Elt>
void HeckeTauTableT<Elt>::construct_vertex(uint32_t i, uint32_t v) {
    // z = (g^{-1} s_i) . pos(v), computed integrally via the projective
    // adj(g) = diag(N, 1) = N g^{-1} trick (NO rationals; Moebius action is
    // projective, det(adj(g) s_i) = N > 0). [C, alg:tau](1).
    //
    // ||z|| reaches ~ ||pos||^2 N^2 and moebius squares it, so the int64 tier
    // of [C, thm:backend] is exhausted at N >= ~4580 (docs/KNOWN_ISSUES.md
    // Sec. 3). escalate_on_overflow attempts int64 first and only then runs
    // the __int128 tier, so nothing changes wherever int64 completes.
    const Mat22 adj_g{ static_cast<int64_t>(N_), 0, 0, 1 };
    const Mat22 M = adj_g * cosets_->s(i);
    Mat22 gred = Mat22::I();
    uint32_t w = 0;
    escalate_on_overflow(
        [&] { step1_vertex_on_tier<QuadPoint>(M, v, gred, w); },
        [&] { step1_vertex_on_tier<QuadPointW>(M, v, gred, w); });
    // tau(s_i . i(v)) := gamma_red^{-1} . i(w)  (satisfies (T3): one
    // coefficient-1 vertex; certification asserts it).
    CanonMap acc;
    accumulate_canonical(acc, gred.inv_unimodular(), 0, w, 1);
    ChainRecord& rec = record_mut(i, 0, v);
    rec.terms = canon_map_to_terms(acc, 0, false);
    rec.mod_ell = false;
    rec.stored = true;
}

template<class Elt>
void HeckeTauTableT<Elt>::build_vertices() {
    for (uint32_t v = 0; v < simplices_[0].size(); ++v) {
        for (uint32_t i = 1; i <= m_; ++i) {
            if (records_[0][v][i - 1].stored) continue; // pre-assigned by step (B)
            if (!try_derive(i, 0, v)) construct_vertex(i, v);
        }
    }
}

// ============================================================================
// Step 2: edges ([C, alg:tau](2))
// ============================================================================

template<class Elt>
void HeckeTauTableT<Elt>::construct_edge(uint32_t i, uint32_t e) {
    const auto& verts = simplices_[1][e]; // {va, vb}, va < vb
    const uint32_t va = verts[0];
    const uint32_t vb = verts[1];
    // Endpoints of tau(boundary(e~)) are the already-stored vertex slots
    // (faces of s_i . i(e) are s_i . i(v_j), themselves stored
    // representatives; the (T1)+(P1) fetch degenerates to a direct lookup).
    const std::vector<ChainTerm>& A = record(i, 0, va).terms;
    const std::vector<ChainTerm>& B = record(i, 0, vb).terms;
    if (A.size() != 1 || A[0].coeff != 1 || B.size() != 1 || B[0].coeff != 1) {
        throw std::runtime_error("HeckeTauTable: vertex slot is not a single coefficient-1 term.");
    }
    const Mat22 ga = A[0].gamma;
    const uint32_t wa = A[0].sigma_idx;
    const Mat22 gb = B[0].gamma;
    const uint32_t wb = B[0].sigma_idx;

    ChainRecord& rec = record_mut(i, 1, e);
    rec.mod_ell = false;
    if (ga == gb && wa == wb) {
        // Same K^u vertex (canonical forms coincide): the empty 1-chain has
        // exactly the required boundary 0.
        rec.terms.clear();
        rec.stored = true;
        return;
    }

    // Connect gamma_a . i(wa) -> gamma_b . i(wb): reduce gamma_a^{-1}
    // gamma_b to a word in T^{+-1}, S by (P2) and concatenate translated,
    // sign-tracked path templates (each certified in S2: the template
    // boundary equals letter . i(w2) - i(w) exactly over Z). The -I word
    // letter is dropped: -I acts trivially on K^u (canonical keys quotient
    // by it).
    // (slot-24 ELL-W) two stored-class operands: the wide tier (ExactInt.h), int64 first.
    const Mat22 rel = mul_tiered(inv_unimodular_tiered(ga), gb);
    const auto word = model_->word_in_T_S(rel);
    std::vector<char> letters;
    for (const auto& lp : word) {
        if (lp.first == '-') continue;
        if (lp.first == 'S') {
            letters.push_back('S');
        } else { // 'T' with exact exponent
            const char c = (lp.second > 0) ? 'T' : 't';
            const uint64_t reps = uabs64(lp.second);
            for (uint64_t k = 0; k < reps; ++k) letters.push_back(c);
        }
    }
    if (letters.empty()) {
        // rel = +-I but different vertices/canonical forms: connect inside
        // the single translate via the T then T^{-1} detour (net letter
        // product I; boundary still telescopes).
        letters = { 'T', 't' };
    }

    CanonMap acc;
    Mat22 P = ga; // running prefix: chain term translations gamma_a * L_1 ... L_{j-1}
    uint32_t from_w = wa;
    for (const char L : letters) {
        const uint32_t to_w = wb; // deterministic anchor: the target vertex
        const auto& tmpl = model_->path_template(L, from_w, to_w, template_family());
        for (const auto& t : tmpl) {
            accumulate_canonical(acc, P * t.gamma, 1, t.edge_idx, t.coeff);
        }
        const Mat22 Lm = (L == 'T') ? Mat22::T()
                       : (L == 't') ? Mat22::T().inv_unimodular()
                                    : Mat22::S();
        P = P * Lm;
        from_w = to_w;
    }
    if (!(P == gb || P == gb.neg())) {
        throw std::runtime_error("HeckeTauTable: edge path prefix mismatch (internal (P2) error).");
    }
    rec.terms = canon_map_to_terms(acc, 1, false);
    rec.stored = true;
}

template<class Elt>
void HeckeTauTableT<Elt>::build_edges() {
    for (uint32_t e = 0; e < simplices_[1].size(); ++e) {
        for (uint32_t i = 1; i <= m_; ++i) {
            if (records_[1][e][i - 1].stored) continue; // pre-assigned by step (B)
            if (!try_derive(i, 1, e)) construct_edge(i, e);
        }
    }
}

// ============================================================================
// Step 3: triangles ([C, alg:tau](3))
// ============================================================================

template<class Elt>
bool HeckeTauTableT<Elt>::solve_triangle_system(const CanonMap& c_map_in, std::vector<ChainTerm>& x_out,
                                          bool& used_mod_ell_out, bool& used_wide_out) const {
    // Strip exact zeros.
    CanonMap c_map;
    for (const auto& kv : c_map_in) {
        if (kv.second != 0) c_map.insert(kv);
    }
    used_mod_ell_out = false;
    used_wide_out = false;
    if (c_map.empty()) {
        x_out.clear();
        return true;
    }

    // Neighborhood N: union of translates gamma . i(K) occurring in the
    // support of c; grown by one ring of adjacent translates (generator
    // moves T, T^{-1}, S) on failure. Terminates: K^u is acyclic and every
    // 1-cycle bounds at a finite stage ([C, alg:tau](3)).
    const auto norm_pm = [](const Mat22& g) { return lex_less(g, g.neg()) ? g : g.neg(); };
    std::set<std::array<int64_t, 4>> translate_keys;
    std::vector<Mat22> translates;
    const auto add_translate = [&](const Mat22& g) {
        const Mat22 n = norm_pm(g);
        if (translate_keys.insert(mat_key(n)).second) translates.push_back(n);
    };
    for (const auto& kv : c_map) {
        add_translate(Mat22{ kv.first[2], kv.first[3], kv.first[4], kv.first[5] });
    }

    const uint32_t num_tris = static_cast<uint32_t>(simplices_[2].size());
    const Mat22 gens[3] = { Mat22::T(), Mat22::T().inv_unimodular(), Mat22::S() };
    constexpr uint32_t kMaxRings = 8;

    for (uint32_t ring = 0; ring <= kMaxRings; ++ring) {
        // Columns: canonical 2-simplex keys of the neighborhood.
        std::map<std::array<int64_t, 6>, size_t> col_index;
        std::vector<std::pair<Mat22, uint32_t>> cols; // (gamma_canon, sigma_hat)
        for (const auto& g : translates) {
            for (uint32_t f = 0; f < num_tris; ++f) {
                const KuSimplexKey key = model_->canonical_key(g, 2, f);
                const std::array<int64_t, 6> arr = {
                    2, static_cast<int64_t>(key.sigma_hat),
                    key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d
                };
                if (col_index.find(arr) == col_index.end()) {
                    col_index[arr] = cols.size();
                    cols.emplace_back(key.gamma_canon, key.sigma_hat);
                }
            }
        }
        // Rows: canonical 1-keys (supp(c) first, then all column faces).
        std::map<std::array<int64_t, 6>, size_t> row_index;
        std::vector<int64_t> rhs;
        for (const auto& kv : c_map) {
            row_index[kv.first] = rhs.size();
            rhs.push_back(kv.second);
        }
        std::vector<std::tuple<size_t, size_t, int64_t>> entries; // (row, col, coeff)
        for (size_t j = 0; j < cols.size(); ++j) {
            for (uint32_t jj = 0; jj <= 2; ++jj) {
                const uint32_t face = face_simplex_index(2, cols[j].second, jj);
                const KuSimplexKey key = model_->canonical_key(cols[j].first, 1, face);
                const std::array<int64_t, 6> arr = {
                    1, static_cast<int64_t>(key.sigma_hat),
                    key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d
                };
                auto it = row_index.find(arr);
                if (it == row_index.end()) {
                    it = row_index.emplace(arr, rhs.size()).first;
                    rhs.push_back(0);
                }
                const int64_t sgn = (jj % 2 == 0) ? 1 : -1;
                entries.emplace_back(it->second, j, sgn * static_cast<int64_t>(key.sign));
            }
        }
        std::vector<std::vector<int64_t>> A(rhs.size(), std::vector<int64_t>(cols.size(), 0));
        for (const auto& en : entries) {
            A[std::get<0>(en)][std::get<1>(en)] =
                checked_add<int64_t>(A[std::get<0>(en)][std::get<1>(en)], std::get<2>(en));
        }

        // (slot-6 WP-4) BoundaryStrict: over Z ONLY ([C, prop:strictpair],
        // [OD] OP2) -- the int64 tier, then the widened __int128 tier after a
        // HeckeOverflow (hecke::escalate_on_overflow, so ScopedExactTier is
        // honoured); a widened-tier overflow propagates (a loud tier failure),
        // and a Z-infeasible system grows the neighbourhood. No mod-ell solve.
        if (policy_ == TauVertexPolicy::BoundaryStrict) {
            std::vector<int64_t> y;
            bool ok = false, wide = false;
            escalate_on_overflow(
                [&] { ok = solve_integer_system(A, rhs, y); },
                [&] { wide = true; ok = solve_integer_system_wide(A, rhs, y); });
            if (ok) {
                CanonMap x_map;
                for (size_t j = 0; j < cols.size(); ++j) {
                    if (y[j] == 0) continue;
                    const std::array<int64_t, 6> arr = {
                        2, static_cast<int64_t>(cols[j].second),
                        cols[j].first.a, cols[j].first.b, cols[j].first.c, cols[j].first.d
                    };
                    x_map[arr] = y[j];
                }
                x_out = canon_map_to_terms(x_map, 2, false);
                used_wide_out = wide;
                return true;
            }
        } else {
        // Over Z first ([C, alg:tau](3) fixes the preference); overflow of
        // the checked int64 elimination is treated as Z-infeasible and falls
        // through to the permitted mod-ell solve.
        try {
            std::vector<int64_t> y;
            if (solve_integer_system(A, rhs, y)) {
                CanonMap x_map;
                for (size_t j = 0; j < cols.size(); ++j) {
                    if (y[j] == 0) continue;
                    const std::array<int64_t, 6> arr = {
                        2, static_cast<int64_t>(cols[j].second),
                        cols[j].first.a, cols[j].first.b, cols[j].first.c, cols[j].first.d
                    };
                    x_map[arr] = y[j];
                }
                x_out = canon_map_to_terms(x_map, 2, false);
                used_mod_ell_out = false;
                return true;
            }
        } catch (const HeckeOverflow&) {
            // fall through to mod-ell
        }
        {
            std::vector<Elt> y;
            if (solve_mod_ell_system(A, rhs, ell_, y)) {
                CanonMap x_map;
                for (size_t j = 0; j < cols.size(); ++j) {
                    if (y[j] == 0) continue;
                    const std::array<int64_t, 6> arr = {
                        2, static_cast<int64_t>(cols[j].second),
                        cols[j].first.a, cols[j].first.b, cols[j].first.c, cols[j].first.d
                    };
                    x_map[arr] = static_cast<int64_t>(y[j]);
                }
                x_out = canon_map_to_terms(x_map, 2, true);
                used_mod_ell_out = true;
                return true;
            }
        }
        } // !BoundaryStrict
        // Grow one ring of adjacent translates.
        const size_t old_size = translates.size();
        for (size_t idx = 0; idx < old_size; ++idx) {
            for (const auto& L : gens) add_translate(translates[idx] * L);
        }
        if (translates.size() == old_size) return false;
    }
    return false;
}

template<class Elt>
void HeckeTauTableT<Elt>::construct_triangle(uint32_t i, uint32_t f) {
    // c := tau(boundary(t~)) = sum_j (-1)^j tau(s_i . i(face_j)), all faces
    // stored representatives (a 1-cycle by step 2 + (T2) in lower degree).
    CanonMap c_map;
    bool face_mod = false;
    for (uint32_t j = 0; j <= 2; ++j) {
        const uint32_t face = face_simplex_index(2, f, j);
        const ChainRecord& fr = record(i, 1, face);
        face_mod = face_mod || fr.mod_ell;
        const int64_t sgn = (j % 2 == 0) ? 1 : -1;
        for (const auto& term : fr.terms) {
            accumulate_canonical(c_map, term.gamma, 1, term.sigma_idx,
                                 sgn * static_cast<int64_t>(term.coeff));
        }
    }
    std::vector<ChainTerm> x;
    bool used_mod = false, used_wide = false;
    if (!solve_triangle_system(c_map, x, used_mod, used_wide)) {
        std::ostringstream oss;
        oss << "HeckeTauTable: triangle fill unsolvable after max ring growth "
            << "(i = " << i << ", triangle = " << f << ") -- internal error "
            << "(K^u is acyclic; this must not happen).";
        throw std::runtime_error(oss.str());
    }
    ChainRecord& rec = record_mut(i, 2, f);
    rec.terms = std::move(x);
    rec.mod_ell = used_mod || face_mod;
    rec.stored = true;
    if (used_mod) {
        any_mod_ell_ = true;
        ++num_mod_ell_solves_;
    } else {
        ++num_z_solves_;
        if (used_wide) ++num_z_wide_solves_;
    }
}

// ============================================================================
// [C, prop:strictpair] step (B) (slot-6 WP-4; BoundaryStrict only)
// ============================================================================

template<class Elt>
void HeckeTauTableT<Elt>::build_strict_boundary() {
    // tau^s(j sigma) := j tau_bd(sigma) for every boundary simplex sigma: a
    // boundary simplex of K IS a simplex of K and tau_bd's terms are already
    // canonical K^u-chains in K's indexing (HeckeBoundaryTauTable::tau), so the
    // chain is copied through the canonical map (a no-op on canonical input,
    // kept so the stored form is this table's own canonical form by
    // construction). (T1)/(T2)/(T3) on these slots are inherited from tau_bd
    // and re-certified by clauses (b)-(d); clause (g) re-asserts the equality.
    ENGINE_CHECK_MSG(taub_ != nullptr, "HeckeTauTable: step (B) without a boundary table.");
    const uint32_t nb = std::min<uint32_t>(taub_->num_degrees(), 3u);
    for (uint32_t q = 0; q < nb; ++q) {
        for (const uint32_t sigma : taub_->boundary_simplices(q)) {
            for (uint32_t i = 1; i <= m_; ++i) {
                ChainRecord& rec = record_mut(i, q, sigma);
                ENGINE_CHECK_MSG(!rec.stored, "HeckeTauTable: a boundary slot was pre-assigned twice.");
                rec.terms = canon_map_to_terms(terms_to_canon_map(taub_->tau(i, q, sigma), q), q, false);
                rec.mod_ell = false;
                rec.stored = true;
                rec.derived = false;
                ++num_strict_bd_slots_;
            }
        }
    }
}

template<class Elt>
uint32_t HeckeTauTableT<Elt>::certify_strict() const {
    if (taub_ == nullptr) return 71;
    if (&taub_->cosets() != cosets_ || &taub_->model() != model_) return 71;
    if (any_mod_ell_ || num_mod_ell_solves_ != 0) return 72;
    const uint32_t nb = std::min<uint32_t>(taub_->num_degrees(), 3u);
    for (uint32_t q = 0; q < nb; ++q) {
        for (const uint32_t sigma : taub_->boundary_simplices(q)) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const ChainRecord& rec = records_[q][sigma][i - 1];
                if (rec.mod_ell) return 72;
                if (!canon_maps_equal(terms_to_canon_map(rec.terms, q),
                                      terms_to_canon_map(taub_->tau(i, q, sigma), q), false)) {
                    return 70;
                }
            }
        }
    }
    return 0;
}

template<class Elt>
void HeckeTauTableT<Elt>::build_triangles() {
    for (uint32_t f = 0; f < simplices_[2].size(); ++f) {
        for (uint32_t i = 1; i <= m_; ++i) {
            if (records_[2][f][i - 1].stored) continue; // pre-assigned by step (B)
            if (!try_derive(i, 2, f)) construct_triangle(i, f);
        }
    }
}

// ============================================================================
// Xi-term table ([C, eq:xiexplicit]), built once
// ============================================================================

template<class Elt>
FlatMatrixT<Elt> HeckeTauTableT<Elt>::rho_g_gamma_of(const Mat22& gamma) const {
    // rho_n((g gamma) mod ell) via ComputeSymmetricPowerMatrixFM (invariant
    // I2). det(g gamma) = N: a semigroup element, NEVER inverted (I7).
    const Mat22 gg = cosets_->g() * gamma;
    std::array<Elt, 4> gm;
    if constexpr (std::is_same_v<Elt, uint32_t>) {
        gm = gg.reduced_mod(ell_);
    } else {
        gm = gg.reduced_mod64(ell_);
    }
    const FlatMatrixT<Elt> fm(2, 2, ell_, std::vector<Elt>{ gm[0], gm[1], gm[2], gm[3] });
    return ComputeSymmetricPowerMatrixFM(fm, n_sym_);
}

template<class Elt>
void HeckeTauTableT<Elt>::build_xi_terms() {
    // rho_n((g gamma) mod ell) depends only on the reduced key, so it is both
    // memoised (as before) and, since 7.3, stored ONCE: the pool holds one
    // matrix per distinct key and each term keeps a handle (pool + index)
    // instead of its own d_V x d_V copy.
    //
    // The loop nest, and therefore the term-visit order and the pool's
    // insertion order, is unchanged -- tau's derivation is first-match-wins and
    // any reordering would silently change Theta (audit 02 Section 6.5).
    auto pool = std::make_shared<std::vector<FlatMatrixT<Elt>>>();
    std::map<std::array<Elt, 4>, uint32_t> cache; // reduced key -> pool slot
    // (slot-13 WP-4) The U_ell lever: read at construction; only where
    // every g gamma is singular mod ell (ell | N).  Same keys, same slots, same
    // visit order -- only the slot's storage differs.
    // (owner decision 2026-09-28, Q8) Default Auto: every table with ell | N
    // factors its pool unless the lever is Off.
    const bool low_rank = HeckeOperatorOptions::u_ell_low_rank_mode() != HeckeOperatorOptions::UEllLowRankMode::Off &&
                          (N_ % static_cast<uint64_t>(ell_) == 0);
    std::vector<lowrank::LowRankT<Elt>> pool_lr;
    const engine::ModCtx<Elt> lr_ctx(ell_);
    // (slot-14 G0) The rank-one closed form (default on; opt-out
    // HeckeOperatorOptions::ScopedRankOneClosedForm(false), HECKE_RANK1_CLOSED_FORM=0):
    // a slot's (U, V) straight from the reduced key in O(d), the same bytes as
    // factoring rho_g_gamma_of(gamma) (lowrank::rank_one_rho_factors; the rank
    // claim ENGINE_CHECKed per key; re-derived densely under !NDEBUG).  The
    // dense witness of every slot stays in certify() clause (f).
    const bool closed_form = low_rank && HeckeOperatorOptions::rank_one_closed_form();
    const std::vector<Elt> binom = closed_form ? lowrank::binomial_row_mod<Elt>(lr_ctx, n_sym_) : std::vector<Elt>();
    auto factored_slot = [&](const Mat22& gamma) -> lowrank::LowRankT<Elt> {
        if (closed_form) {
            const Mat22 gg = cosets_->g() * gamma;
            std::array<Elt, 4> gm;
            if constexpr (std::is_same_v<Elt, uint32_t>) {
                gm = gg.reduced_mod(ell_);
            } else {
                gm = gg.reduced_mod64(ell_);
            }
            lowrank::LowRankT<Elt> f = lowrank::rank_one_rho_factors<Elt>(lr_ctx, gm[0], gm[1], gm[2], gm[3], n_sym_, binom);
#ifndef NDEBUG
            {
                const FlatMatrixT<Elt> dense = rho_g_gamma_of(gamma);
                const lowrank::LowRankT<Elt> ref = lowrank::factor_low_rank<Elt>(
                    lr_ctx, lowrank::dense_rows_of<Elt>(dense), dense.get_rows(), dense.get_cols(),
                    std::min<uint32_t>(1u, dense.get_rows()));
                ENGINE_DEBUG_CHECK_MSG(ref.t == f.t && ref.U == f.U && ref.V == f.V,
                                       "HeckeTauTable (G0): the rank-one closed form differs from the dense factorisation");
            }
#endif
            return f;
        }
        const FlatMatrixT<Elt> dense = rho_g_gamma_of(gamma);
        return lowrank::factor_low_rank<Elt>(lr_ctx, lowrank::dense_rows_of<Elt>(dense), dense.get_rows(),
                                             dense.get_cols(), std::min<uint32_t>(1u, dense.get_rows()));
    };
    // (slot-14 WP-10) The serial phase on the pool (ScopedParallelSetup,
    // default on since 2026-09-28, Q8): the walk below still assigns every key its slot in the serial
    // visit order, but only records the slot's representative gamma (the
    // first term visited with that key -- the one the serial loop evaluates);
    // the rho_n evaluation (and, under the U_ell lever, the factorisation) of
    // each slot then runs as a host task writing that pre-assigned slot.
    const bool par = HeckeOperatorOptions::parallel_setup();
    std::vector<Mat22> slot_gamma;
    std::vector<std::array<Elt, 4>> slot_key;   // (slot-14 G0 repairs, audit m2) per slot, in slot order
    for (uint32_t q = 0; q < 3; ++q) {
        for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const ChainRecord& rec = records_[q][sigma][i - 1];
                std::vector<XiTermT<Elt>>& out = xi_[q][sigma][i - 1];
                out.clear();
                out.reserve(rec.terms.size());
                for (const auto& term : rec.terms) {
                    const Mat22 g_gamma = cosets_->g() * term.gamma;
                    std::array<Elt, 4> key;
                    if constexpr (std::is_same_v<Elt, uint32_t>) {
                        key = g_gamma.reduced_mod(ell_);
                    } else {
                        key = g_gamma.reduced_mod64(ell_);
                    }
                    auto it = cache.find(key);
                    if (it == cache.end()) slot_key.push_back(key);   // the slot about to be created below
                    if (it == cache.end() && (par || low_rank)) slot_gamma.push_back(term.gamma);
                    if (it == cache.end() && par) {
                        pool->push_back(FlatMatrixT<Elt>());   // filled below (a placeholder under the lever)
                        if (low_rank) pool_lr.emplace_back();
                        it = cache.emplace(key, static_cast<uint32_t>(pool->size() - 1)).first;
                    }
                    if (it == cache.end()) {
                        if (low_rank) {
                            pool_lr.push_back(factored_slot(term.gamma));
                            pool->push_back(FlatMatrixT<Elt>());   // placeholder: get() throws on it
                        } else {
                            pool->push_back(rho_g_gamma_of(term.gamma));
                        }
                        it = cache.emplace(key, static_cast<uint32_t>(pool->size() - 1)).first;
                    }
                    out.push_back(XiTermT<Elt>{ term.coeff, term.sigma_idx,
                                                RhoRefT<Elt>(pool, it->second) });
                }
            }
        }
    }
    if (par) {
        std::vector<FlatMatrixT<Elt>>& pool_ref = *pool;
        const uint64_t dd = static_cast<uint64_t>(n_sym_ + 1) * (n_sym_ + 1);
        HeckeOperatorOptions::for_each_index(slot_gamma.size(), true, [&](size_t j) {
            if (low_rank) {
                pool_lr[j] = factored_slot(slot_gamma[j]);
            } else {
                pool_ref[j] = rho_g_gamma_of(slot_gamma[j]);
            }
        }, closed_form ? static_cast<uint64_t>(8) * (n_sym_ + 1) : dd);
    }
    rho_pool_ = std::move(pool);
    rho_pool_lr_ = std::move(pool_lr);
    if (low_rank) slot_gamma_ = std::move(slot_gamma);
    ENGINE_CHECK(slot_key.size() == rho_pool_->size());
    slot_key_ = std::move(slot_key);
}

// ============================================================================
// Accessors
// ============================================================================

template<class Elt>
const std::vector<ChainTerm>& HeckeTauTableT<Elt>::tau(uint32_t i, uint32_t q,
                                                 uint32_t sigma_idx) const {
    return record(i, q, sigma_idx).terms;
}

template<class Elt>
const std::vector<XiTermT<Elt>>& HeckeTauTableT<Elt>::xi_terms(uint32_t i, uint32_t q,
                                                   uint32_t sigma_idx) const {
    require_pool("xi_terms");   // (S23 G2-F) a ChainsOnly table has no xi-term table
    check_indices(i, q, sigma_idx);
    return xi_[q][sigma_idx][i - 1];
}

// ============================================================================
// Step 4: MANDATORY certification ([C, alg:tau](4), [C, obl:tau])
// ============================================================================

template<class Elt>
uint32_t HeckeTauTableT<Elt>::certify() const {
    // (S23 G2-F) On a ChainsOnly table every clause runs as on a Full one except those about the xi-term table and
    // the rho pool, which it does not hold: (a)'s length match (code 15) and (f).
    const bool full = contents_ == TauTableContents::Full;
    // (a) Structural: stored, canonical, unimodular, nonzero.
    for (uint32_t q = 0; q < 3; ++q) {
        for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const ChainRecord& rec = records_[q][sigma][i - 1];
                if (!rec.stored) return 14;
                for (const auto& term : rec.terms) {
                    if (term.coeff == 0) return 13;
                    if (det_tiered(term.gamma) != 1) return 11;   // (slot-24 ELL-W) det of a stored matrix: the wide tier
                    const KuSimplexKey key = model_->canonical_key(term.gamma, q, term.sigma_idx);
                    if (key.sigma_hat != term.sigma_idx ||
                        !(key.gamma_canon == term.gamma) || key.sign != 1) {
                        return 12;
                    }
                }
                if (full && xi_[q][sigma][i - 1].size() != rec.terms.size()) return 15;
            }
        }
    }

    // (b) Augmentation (T3): epsilon(tau(vertex slot)) == 1.
    for (uint32_t v = 0; v < simplices_[0].size(); ++v) {
        for (uint32_t i = 1; i <= m_; ++i) {
            const ChainRecord& rec = records_[0][v][i - 1];
            int64_t total = 0;
            for (const auto& term : rec.terms) {
                total = checked_add<int64_t>(total, term.coeff);
            }
            const int64_t d = total - 1;
            const bool ok = rec.mod_ell ? (d % static_cast<int64_t>(ell_) == 0) : (d == 0);
            if (!ok) return 20;
        }
    }

    // (c) Chain map (T2): boundary(tau(x)) == tau(boundary(x)) for every
    // stored representative, the RHS evaluated ONLY through the shared
    // (T1)/(P1)/(P4) helper + canonical keys ([C, alg:tau](4)); exact over
    // Z, or mod ell where a mod-ell-solved chain is involved.
    for (uint32_t q = 1; q <= 2; ++q) {
        for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const ChainRecord& rec = records_[q][sigma][i - 1];
                const CanonMap lhs = boundary_of_terms(rec.terms, q);
                CanonMap rhs;
                bool mod = rec.mod_ell;
                for (uint32_t j = 0; j <= q; ++j) {
                    const uint32_t face = face_simplex_index(q, sigma, j);
                    bool src_mod = false;
                    const std::vector<ChainTerm> ch =
                        eval_tau_on_translate(cosets_->s(i), q - 1, face, src_mod);
                    mod = mod || src_mod;
                    const int64_t sgn = (j % 2 == 0) ? 1 : -1;
                    for (const auto& term : ch) {
                        accumulate_canonical(rhs, term.gamma, q - 1, term.sigma_idx,
                                             sgn * static_cast<int64_t>(term.coeff));
                    }
                }
                if (!canon_maps_equal(lhs, rhs, mod)) return 30 + (q - 1);
            }
        }
    }

    // (d) Twisted-equivariance bookkeeping ((T1), [C, obl:tau]): for EVERY
    // stored transporter t with i(sigma) = t . i(sigma') and every coset
    // index i, the stored slot (i, sigma) must equal the sign-tracked
    // (T1)-translate of the slot reached through s_i * t. This covers the
    // elliptic stabilizer self-constraints (sigma == sigma', t nontrivial)
    // and every cross-slot Gamma''-identification, independently of which
    // derivation path construction used.
    for (uint32_t q = 0; q < 3; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices_[q].size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t sigma2 = 0; sigma2 < count; ++sigma2) {
                if (!model_->same_groupoid_orbit(q, sigma, sigma2)) continue;
                const auto& ts = model_->transporter(q, sigma, sigma2);
                const auto& perms = model_->transporter_vertex_perms(q, sigma, sigma2);
                for (size_t k = 0; k < ts.size(); ++k) {
                    for (uint32_t i = 1; i <= m_; ++i) {
                        const Mat22 gamma = cosets_->s(i) * ts[k];
                        bool src_mod = false;
                        const std::vector<ChainTerm> ch =
                            eval_tau_on_translate(gamma, q, sigma2, src_mod);
                        const int8_t eps = perm_parity(perms[k], q);
                        CanonMap rhs;
                        for (const auto& term : ch) {
                            accumulate_canonical(rhs, term.gamma, q, term.sigma_idx,
                                                 static_cast<int64_t>(eps) *
                                                     static_cast<int64_t>(term.coeff));
                        }
                        const ChainRecord& rec = records_[q][sigma][i - 1];
                        const CanonMap lhs = terms_to_canon_map(rec.terms, q);
                        if (!canon_maps_equal(lhs, rhs, rec.mod_ell || src_mod)) return 40;
                    }
                }
            }
        }
    }

    // (e) Bounds ([C, lem:bounds](5), [C, thm:backend], [C, obl:backend]):
    // recount B_tab, recompute B = max(B_tab, 4 N^2), assert the int64 tier
    // 2 B^2 < 2^63 (equivalently B <= 2^31 - 1).
    {
        uint64_t recount = 0;
        for (uint32_t q = 0; q < 3; ++q) {
            for (const auto& per_sigma : records_[q]) {
                for (const auto& rec : per_sigma) {
                    for (const auto& term : rec.terms) {
                        const uint64_t norm = static_cast<uint64_t>(term.gamma.max_abs());
                        if (norm > recount) recount = norm;
                    }
                }
            }
        }
        if (recount != bound_B_tab_) return 50;
        const unsigned __int128 four_N_sq = static_cast<unsigned __int128>(4) * N_ * N_;
        const unsigned __int128 B = (static_cast<unsigned __int128>(recount) > four_N_sq) ? recount : four_N_sq;
        if (B != static_cast<unsigned __int128>(bound_B_)) return 50;   // (a saturated member never equals a B >= 2^64)
        {
            // (ELL-W audit F3) the tier assertion is hecke::tau_bound_clause (ExactInt.cpp), a pure function:
            // 2 B^2 < 2^63 (narrow, the pre-ELL-W criterion); else 51 if the wide tier may not run (HECKE_TAU_WIDE=0 or a
            // ForceNarrow build); else the wide clause 2 N B <= 2^63 - 1 ([C] lem:bounds(7), thm:backend(4)) or 52.
            const uint32_t tier_code = tau_bound_clause(N_, B, wide_allowed_ && wide_tier_);
            if (tier_code != 0) return tier_code;
            // The wide clause, spelled out:
            // 2 B^2 >= 2^63: the narrow (int64) tier does not cover the table.  Pre-ELL-W this was the end (code 51);
            // it still is when the wide tier may not run (HECKE_TAU_WIDE=0, or built under ExactTier::ForceNarrow).
            // (slot-24 ELL-W) The wide clause of [C, lem:bounds](5) / [C, thm:backend](4): 2 N B <= 2^63 - 1.
            //   * stored matrices: ||gamma_t|| <= B_tab <= B < 2^61, int64 (the recount above is exact);
            //   * (ii) Euclidean steps <= 2 B < 2^63, and unimodular inverses <= B: int64;
            //   * fixed datum x stored (g gamma, delta_i gamma, s_i gamma; ||datum|| <= N by lem:bounds(2)):
            //     <= 2 N B <= 2^63 - 1 by lem:bounds(1), int64 -- the consumers' products, unchanged code;
            //   * stored x stored and determinants: intermediates <= B^2 < 2^122, formed in __int128 by
            //     mul_tiered / det_tiered (exact), the result narrowed to int64 with a check (abort, never wrap).
            // N >= 2 here, so 2 N B <= 2^63 - 1 implies B < 2^61.  Measured-only (checked, abort beyond int64; [C]
            // lem:bounds(7) last paragraph): (P2) on rel = ga^-1 gb (<= 2 B^2) and the edge prefix P, the (T1) conjugate.
        }
    }

    // (f) Xi-term table matches a recomputation from the exact gamma_k
    // (invariant I2, [C, eq:xiexplicit]).
    //
    // The recomputation is memoised on the reduced key (g gamma_k) mod ell,
    // exactly as build_xi_terms is (item 7.0b): rho_n((g gamma) mod ell) is a
    // pure function of that key, so 125 evaluations collapse to 8. This does
    // NOT weaken the certification: EVERY term is still compared against a
    // matrix recomputed from ITS OWN exact stored gamma_k (so a term pointing
    // at the wrong pool slot is still caught), the comparison is still exact,
    // and it is still mandatory.
    if (!full) {
        // (S23 G2-F) ChainsOnly: no xi-term table, no pool -- nothing for (f) to certify.
    } else if (HeckeOperatorOptions::parallel_setup()) {
        // (slot-14 WP-10) The serial phase on the pool (ScopedParallelSetup,
        // default on since 2026-09-28, Q8).  Serial pass: the structural (coeff, sigma) comparison of
        // every term, and the grouping of the terms by their reduced key in
        // walk order -- per key the representative gamma (the first term's,
        // which is what the serial loop's memo evaluates) and the distinct
        // pool slots the key's terms point at (under the lever: per distinct
        // (key, slot) pair its first term's gamma, exactly the serial loop's
        // lr_verified set).  Then one host task per group recomputes rho_n
        // from the representative's exact gamma and compares every slot of
        // the group, exactly as the serial loop compares every term (a
        // repeated (key, slot) comparison is the same comparison).  Every (f)
        // failure is code 60, so the verdict is the serial one.
        struct Group { Mat22 gamma; std::vector<uint32_t> slots; };
        std::vector<Group> groups;
        std::map<std::array<Elt, 4>, size_t> group_of_key;                        // dense pool
        std::map<std::pair<std::array<Elt, 4>, uint32_t>, size_t> group_of_pair;   // factored pool
        const bool lr = rho_pool_is_low_rank();
        for (uint32_t q = 0; q < 3; ++q) {
            for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
                for (uint32_t i = 1; i <= m_; ++i) {
                    const ChainRecord& rec = records_[q][sigma][i - 1];
                    const std::vector<XiTermT<Elt>>& xt = xi_[q][sigma][i - 1];
                    if (xt.size() != rec.terms.size()) return 60;
                    for (size_t k = 0; k < rec.terms.size(); ++k) {
                        if (xt[k].coeff != rec.terms[k].coeff ||
                            xt[k].sigma_idx != rec.terms[k].sigma_idx) {
                            return 60;
                        }
                        const Mat22 g_gamma = cosets_->g() * rec.terms[k].gamma;
                        std::array<Elt, 4> key;
                        if constexpr (std::is_same_v<Elt, uint32_t>) {
                            key = g_gamma.reduced_mod(ell_);
                        } else {
                            key = g_gamma.reduced_mod64(ell_);
                        }
                        const uint32_t slot = xt[k].rho_g_gamma.index();
                        if (lr) {
                            if (slot >= rho_pool_lr_.size()) return 60;
                            const auto pk = std::make_pair(key, slot);
                            if (group_of_pair.count(pk) != 0) continue;
                            group_of_pair.emplace(pk, groups.size());
                            groups.push_back(Group{ rec.terms[k].gamma, { slot } });
                            continue;
                        }
                        auto g = group_of_key.find(key);
                        if (g == group_of_key.end()) {
                            g = group_of_key.emplace(key, groups.size()).first;
                            groups.push_back(Group{ rec.terms[k].gamma, {} });
                        }
                        std::vector<uint32_t>& sl = groups[g->second].slots;
                        if (std::find(sl.begin(), sl.end(), slot) == sl.end()) sl.push_back(slot);
                    }
                }
            }
        }
        std::vector<char> bad(groups.size(), 0);
        const engine::ModCtx<Elt> lr_ctx(ell_);
        const uint64_t dd = static_cast<uint64_t>(n_sym_ + 1) * (n_sym_ + 1);
        HeckeOperatorOptions::for_each_index(groups.size(), true, [&](size_t j) {
            const Group& G = groups[j];
            const FlatMatrixT<Elt> R = rho_g_gamma_of(G.gamma);
            for (const uint32_t slot : G.slots) {
                if (lr) {
                    const lowrank::LowRankT<Elt>& f = rho_pool_lr_[slot];
                    if (f.rows != R.get_rows() || f.cols != R.get_cols()) { bad[j] = 1; return; }
                    for (uint32_t a = 0; a < f.rows; ++a)
                        for (uint32_t b = 0; b < f.cols; ++b) {
                            Elt acc = 0;
                            for (uint32_t s = 0; s < f.t; ++s)
                                acc = lr_ctx.add(acc, lr_ctx.mul(f.U[static_cast<size_t>(a) * f.t + s],
                                                                 f.V[static_cast<size_t>(s) * f.cols + b]));
                            if (acc != R.get_entry(a, b)) { bad[j] = 1; return; }
                        }
                } else {
                    // xt.rho_g_gamma.get() of a term naming `slot` IS (*rho_pool_)[slot].
                    if (slot >= rho_pool_->size() || !((*rho_pool_)[slot] == R)) { bad[j] = 1; return; }
                }
            }
        }, dd);
        for (const char b : bad) if (b) return 60;
    } else {
        std::map<std::array<Elt, 4>, FlatMatrixT<Elt>> rho_cache;
        // (slot-13 WP-4) Under the lever: (key, slot) pairs already verified;
        // the recomputed dense block is a transient, never cached.
        std::set<std::pair<std::array<Elt, 4>, uint32_t>> lr_verified;
        const engine::ModCtx<Elt> lr_ctx(ell_);
        for (uint32_t q = 0; q < 3; ++q) {
            for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
                for (uint32_t i = 1; i <= m_; ++i) {
                    const ChainRecord& rec = records_[q][sigma][i - 1];
                    const std::vector<XiTermT<Elt>>& xt = xi_[q][sigma][i - 1];
                    if (xt.size() != rec.terms.size()) return 60;
                    for (size_t k = 0; k < rec.terms.size(); ++k) {
                        if (xt[k].coeff != rec.terms[k].coeff ||
                            xt[k].sigma_idx != rec.terms[k].sigma_idx) {
                            return 60;
                        }
                        const Mat22 g_gamma = cosets_->g() * rec.terms[k].gamma;
                        std::array<Elt, 4> key;
                        if constexpr (std::is_same_v<Elt, uint32_t>) {
                            key = g_gamma.reduced_mod(ell_);
                        } else {
                            key = g_gamma.reduced_mod64(ell_);
                        }
                        if (rho_pool_is_low_rank()) {
                            const uint32_t slot = xt[k].rho_g_gamma.index();
                            if (lr_verified.count(std::make_pair(key, slot)) != 0) continue;
                            if (slot >= rho_pool_lr_.size()) return 60;
                            const lowrank::LowRankT<Elt>& lr = rho_pool_lr_[slot];
                            const FlatMatrixT<Elt> R = rho_g_gamma_of(rec.terms[k].gamma);
                            if (lr.rows != R.get_rows() || lr.cols != R.get_cols()) return 60;
                            for (uint32_t a = 0; a < lr.rows; ++a)
                                for (uint32_t b = 0; b < lr.cols; ++b) {
                                    Elt acc = 0;
                                    for (uint32_t s = 0; s < lr.t; ++s)
                                        acc = lr_ctx.add(acc, lr_ctx.mul(lr.U[static_cast<size_t>(a) * lr.t + s],
                                                                         lr.V[static_cast<size_t>(s) * lr.cols + b]));
                                    if (acc != R.get_entry(a, b)) return 60;
                                }
                            lr_verified.insert(std::make_pair(key, slot));
                            continue;
                        }
                        auto it = rho_cache.find(key);
                        if (it == rho_cache.end()) {
                            it = rho_cache.emplace(key, rho_g_gamma_of(rec.terms[k].gamma)).first;
                        }
                        if (!(xt[k].rho_g_gamma.get() == it->second)) return 60;
                    }
                }
            }
        }
    }

    // (g) Strictness (BoundaryStrict only; slot-6 WP-4).
    if (policy_ == TauVertexPolicy::BoundaryStrict) {
        const uint32_t code = certify_strict();
        if (code != 0) return code;
    }

    return 0;
}

// (S23 G2-F) stored_bytes(): the heap payload of the records, the xi-term table and the rho pool (see the header).
template<class Elt>
uint64_t HeckeTauTableT<Elt>::stored_bytes() const {
    uint64_t b = 0;
    b += records_.capacity() * sizeof(std::vector<std::vector<ChainRecord>>);
    for (const auto& per_q : records_) {
        b += per_q.capacity() * sizeof(std::vector<ChainRecord>);
        for (const auto& per_sigma : per_q) {
            b += per_sigma.capacity() * sizeof(ChainRecord);
            for (const auto& rec : per_sigma) b += rec.terms.capacity() * sizeof(ChainTerm);
        }
    }
    b += xi_.capacity() * sizeof(std::vector<std::vector<std::vector<XiTermT<Elt>>>>);
    for (const auto& per_q : xi_) {
        b += per_q.capacity() * sizeof(std::vector<std::vector<XiTermT<Elt>>>);
        for (const auto& per_sigma : per_q) {
            b += per_sigma.capacity() * sizeof(std::vector<XiTermT<Elt>>);
            for (const auto& terms : per_sigma) b += terms.capacity() * sizeof(XiTermT<Elt>);
        }
    }
    if (rho_pool_) {
        b += rho_pool_->capacity() * sizeof(FlatMatrixT<Elt>);
        for (const auto& m : *rho_pool_) b += m.get_flat_data_unsafe().capacity() * sizeof(Elt);
    }
    b += rho_pool_lr_.capacity() * sizeof(lowrank::LowRankT<Elt>);
    for (const auto& f : rho_pool_lr_) b += (f.U.capacity() + f.V.capacity()) * sizeof(Elt);
    b += slot_gamma_.capacity() * sizeof(Mat22) + slot_key_.capacity() * sizeof(std::array<Elt, 4>);
    return b;
}

// ============================================================================
// Pretty printing
// ============================================================================

template<class Elt>
std::string HeckeTauTableT<Elt>::to_string_offset(uint32_t offset) const {
    const std::string ind(offset, ' ');
    std::ostringstream oss;
    uint32_t total_chains = 0;
    for (uint32_t q = 0; q < 3; ++q) {
        total_chains += static_cast<uint32_t>(simplices_[q].size()) * m_;
    }
    oss << ind << "HeckeTauTable (certified g-twisted chain datum, [C, def:tau]/[C, alg:tau]):\n";
    oss << ind << "  g = diag(1, " << N_ << "), N = p_hecke^r = " << N_
        << ", m = " << m_ << " cosets; ell = " << ell_ << ", n_sym = " << n_sym_ << "\n";
    oss << ind << "  vertex policy: "
        << (policy_ == TauVertexPolicy::GeometricNearest ? "GeometricNearest"
            : policy_ == TauVertexPolicy::GeometricNearestLegacy ? "GeometricNearestLegacy"
            : policy_ == TauVertexPolicy::BoundaryStrict ? "BoundaryStrict" : "FixedV6") << "\n";
    oss << ind << "  triangle solves: " << num_z_solves_ << " over Z, "
        << num_mod_ell_solves_ << " mod ell"
        << (any_mod_ell_ ? "  [mod-ell chains certified mod ell]" : "") << "\n";
    if (policy_ == TauVertexPolicy::BoundaryStrict) {
        // (slot-6 WP-4) printed only for the strict-pair table, so every
        // shipped table's text (tau_table_p2.txt) is byte-unchanged.
        oss << ind << "  strict pair ([C, prop:strictpair]): " << num_strict_bd_slots_
            << " boundary slots = j tau_bd; fills over Z: " << num_fills_int64() << " int64, "
            << num_fills_widened() << " widened; solve_regime " << solve_regime() << "\n";
    }
    oss << ind << "  B_tab = " << bound_B_tab_ << ", B = max(B_tab, 4 N^2) = " << bound_B_
        << " (int64 tier 2 B^2 < 2^63: " << (bound_B_ <= 2147483647ull ? "OK" : (wide_tier_ && wide_allowed_ ? "no -- WIDE tier" : "VIOLATED"))
        << ")\n";
    oss << ind << "  stored chains: " << total_chains << " (= " << m_ << " * ("
        << simplices_[0].size() << " + " << simplices_[1].size() << " + "
        << simplices_[2].size() << "))\n";
    oss << ind << "  tau(s_i . i(sigma)) = sum_k n_k * gamma_k . i(sigma_k), terms canonical:\n";
    const auto simplex_verts = [&](uint32_t q, uint32_t s) {
        std::string r = "{";
        const auto& verts = simplices_[q][s];
        for (size_t k = 0; k < verts.size(); ++k) {
            r += std::to_string(verts[k]);
            if (k + 1 < verts.size()) r += ",";
        }
        return r + "}";
    };
    for (uint32_t q = 0; q < 3; ++q) {
        for (uint32_t sigma = 0; sigma < simplices_[q].size(); ++sigma) {
            for (uint32_t i = 1; i <= m_; ++i) {
                const ChainRecord& rec = records_[q][sigma][i - 1];
                oss << ind << "    [i=" << i << "][q=" << q << "][sigma=" << sigma << " "
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
                if (rec.derived) oss << "   [derived via (T1)]";
                if (rec.mod_ell) oss << "   [mod-ell]";
                oss << "\n";
            }
        }
    }
    return oss.str();
}

// Stage S13 explicit instantiations.
template class HeckeTauTableT<uint32_t>;
template class HeckeTauTableT<uint64_t>;

} // namespace hecke
