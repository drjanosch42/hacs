// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * LocalSystemUtils.cpp
 *
 * This file implements miscellaneous utility functions that are helpful for
 * working with simplicial complexes, particularly in the context of local systems
 * and equivariant cohomology.
 *
 * Implemented Functions:
 * - `find_simplex_idx`: A crucial lookup function that, given a set of vertices,
 *   finds the index of the corresponding simplex within the official list of
 *   q-simplices of a SimplicialComplex. This is essential for referencing
 *   simplices when defining relations or constructing maps.
 */

#include "LocalSystemUtils.h"
#include "../ModArithmetic/EngineCheck.h"
#include "Hecke/HeckeSettings.h" // hecke::RequireFieldCoefficientRing (slot-12 WP-1, carry F-1)

// Helper to find index of a simplex in K's list
uint32_t find_simplex_idx(const SimplicialComplex& K, uint32_t q_dim, const std::vector<uint32_t>& simplex_vertices) {
    if (q_dim >= K.get_simplices().size()) {
        std::ostringstream oss;
        oss << "find_simplex_idx: q_dim " << q_dim << " is out of bounds for complex with max dimension data up to " << K.get_simplices().size() - 1 << ".";
        throw std::out_of_range(oss.str());
    }
    std::vector<uint32_t> sorted_vertices = simplex_vertices;
    std::sort(sorted_vertices.begin(), sorted_vertices.end());

    // (matrix-engine Phase 3, 02 Sec. 7.17) O(log N) lookup in the complex's
    // cached ORDERED simplex index (never a hash map -- 02 Sec. 6.5(5) / R10),
    // replacing the O(N) std::find. Same result: first-match-wins on duplicates.
    const int64_t idx = K.find_simplex_index(q_dim, sorted_vertices);
    if (idx < 0) {
        std::ostringstream oss;
        oss << "find_simplex_idx: Simplex {";
        for (size_t i = 0; i < simplex_vertices.size(); ++i) oss << simplex_vertices[i] << (i == simplex_vertices.size() - 1 ? "" : ",");
        oss << "} not found in K's " << q_dim << "-simplices.";
        throw std::runtime_error(oss.str()); // Or return a special value like -1 and check
    }
    return static_cast<uint32_t>(idx);
}

// ---------------------------------------------------------------------------
// hecke-extensions slot 6, WP-1 -- the contragredient fibre builder
// ([C, def:contragredient]); see LocalSystemUtils.h.
// ---------------------------------------------------------------------------

const char* FibreRepresentationName(FibreRepresentation rep) {
    switch (rep) {
    case FibreRepresentation::SymN: return "sym";
    case FibreRepresentation::ContragredientN: return "contragredient";
    case FibreRepresentation::QuotientL: return "quotient_L";
    case FibreRepresentation::HidaLatticeH: return "hida_lattice";   // slot-13 WP-5
    }
    return "unknown";
}

template<class Elt>
FlatMatrixT<Elt> AdjugateMod2x2(const FlatMatrixT<Elt>& A) {
    ENGINE_CHECK_MSG(A.get_rows() == 2 && A.get_cols() == 2, "AdjugateMod2x2: A must be 2x2");
    const Elt p = A.get_p();
    ENGINE_CHECK_MSG(p > 0, "AdjugateMod2x2: modulus must be positive");
    auto neg = [p](Elt x) -> Elt { return x == 0 ? Elt(0) : static_cast<Elt>(p - x); };
    const Elt a = A.get_entry(0, 0), b = A.get_entry(0, 1), c = A.get_entry(1, 0), d = A.get_entry(1, 1);
    return FlatMatrixT<Elt>(2, 2, p, std::vector<Elt>{ d, neg(b), neg(c), a });
}

template<class Elt>
FlatMatrixT<Elt> ComputeContragredientPowerMatrixFM(const FlatMatrixT<Elt>& A, uint32_t n_sym) {
    // rho_n^vee(A) := rho_n(adj A)^T -- literally the definition.
    return ComputeSymmetricPowerMatrixFM(AdjugateMod2x2(A), n_sym).transpose();
}

template<class Elt>
FlatMatrixT<Elt> ComputeFibreMatrixFM(FibreRepresentation rep, const FlatMatrixT<Elt>& A, uint32_t n_sym) {
    if (rep == FibreRepresentation::SymN) return ComputeSymmetricPowerMatrixFM(A, n_sym);
    if (rep == FibreRepresentation::QuotientL) return ComputeQuotientLMatrixFM(A, n_sym);   // slot-11 WP-5
    if (rep == FibreRepresentation::HidaLatticeH)                                            // slot-13 WP-5
        return ComputeSymmetricPowerMatrixFM(A.transpose(), n_sym).transpose();
    ENGINE_CHECK_MSG(rep == FibreRepresentation::ContragredientN, "ComputeFibreMatrixFM: unknown representation");
    return ComputeContragredientPowerMatrixFM(A, n_sym);
}

template FlatMatrixT<uint32_t> AdjugateMod2x2<uint32_t>(const FlatMatrixT<uint32_t>&);
template FlatMatrixT<uint64_t> AdjugateMod2x2<uint64_t>(const FlatMatrixT<uint64_t>&);
template FlatMatrixT<uint32_t> ComputeContragredientPowerMatrixFM<uint32_t>(const FlatMatrixT<uint32_t>&, uint32_t);
template FlatMatrixT<uint64_t> ComputeContragredientPowerMatrixFM<uint64_t>(const FlatMatrixT<uint64_t>&, uint32_t);
template FlatMatrixT<uint32_t> ComputeFibreMatrixFM<uint32_t>(FibreRepresentation, const FlatMatrixT<uint32_t>&, uint32_t);
template FlatMatrixT<uint64_t> ComputeFibreMatrixFM<uint64_t>(FibreRepresentation, const FlatMatrixT<uint64_t>&, uint32_t);

// ---------------------------------------------------------------------------
// hecke-extensions slot 11, WP-5 -- the quotient fibre Q_n; see the header.
// ---------------------------------------------------------------------------

std::vector<uint32_t> QuotientLComplementRows(uint32_t n_sym, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "QuotientLComplementRows: ell must be >= 2");
    std::vector<uint32_t> rows;
    rows.reserve(static_cast<size_t>(ell) + 1);
    for (uint32_t r = 0; r < ell; ++r) rows.push_back(r);
    rows.push_back(n_sym + ell + 1);
    return rows;
}

template<class Elt>
std::vector<Elt> QuotientLReduce(std::vector<Elt>& v, uint32_t n_sym, Elt ell) {
    ENGINE_CHECK_MSG(ell >= 2, "QuotientLReduce: ell must be >= 2");
    const uint64_t e = static_cast<uint64_t>(ell);
    const uint32_t m = n_sym + static_cast<uint32_t>(ell) + 1;
    ENGINE_CHECK_MSG(v.size() == static_cast<size_t>(m) + 1, "QuotientLReduce: v must have n + ell + 2 entries");
    // Pivot rows ell..m-1, top-down: column j = r - ell of L. is u_{j+1} - u_{j+ell};
    // adding v[r] times it clears row r and adds v[r] to row j + 1 = r - ell + 1 < r.
    for (uint32_t r = m - 1; r >= static_cast<uint32_t>(ell); --r) {
        const Elt c = v[r];
        if (c != 0) {
            const uint32_t t = r - static_cast<uint32_t>(ell) + 1;
            v[t] = static_cast<Elt>((static_cast<uint64_t>(v[t]) + c) % e);
            v[r] = 0;
        }
        if (r == static_cast<uint32_t>(ell)) break;
    }
    std::vector<Elt> out;
    out.reserve(static_cast<size_t>(ell) + 1);
    for (uint32_t row : QuotientLComplementRows(n_sym, static_cast<uint32_t>(ell))) out.push_back(v[row]);
    return out;
}

template<class Elt>
FlatMatrixT<Elt> QuotientLProjectionMatrixFM(uint32_t n_sym, Elt ell) {
    const uint32_t m = n_sym + static_cast<uint32_t>(ell) + 1;
    const uint32_t dq = static_cast<uint32_t>(ell) + 1;
    FlatMatrixT<Elt> Pi = FlatMatrixT<Elt>::CreateZero(dq, m + 1, ell);
    for (uint32_t c = 0; c <= m; ++c) {
        std::vector<Elt> v(static_cast<size_t>(m) + 1, 0);
        v[c] = static_cast<Elt>(1u % static_cast<uint64_t>(ell));
        const std::vector<Elt> w = QuotientLReduce(v, n_sym, ell);
        for (uint32_t t = 0; t < dq; ++t)
            if (w[t] != 0) Pi.set_entry(t, c, w[t]);
    }
    return Pi;
}

template<class Elt>
FlatMatrixT<Elt> ComputeQuotientLMatrixFM(const FlatMatrixT<Elt>& A, uint32_t n_sym) {
    // (slot-12 WP-1, carry F-1; [E] sec:ring (b1)) a purely F_ell routine with
    // no notion of a ring: DicksonThetaCell already refuses FIRST on every
    // production path (HeckeDicksonTheta.cpp), so this never fires there --
    // it closes the member for a caller that reaches this fibre builder
    // directly, naming itself rather than staying silent.
    hecke::RequireFieldCoefficientRing("ComputeQuotientLMatrixFM");
    ENGINE_CHECK_MSG(A.get_rows() == 2 && A.get_cols() == 2, "ComputeQuotientLMatrixFM: A must be 2x2");
    const Elt ell = A.get_p();
    ENGINE_CHECK_MSG(ell >= 2 && static_cast<uint64_t>(ell) < (1ull << 31),
                     "ComputeQuotientLMatrixFM: the modulus ell must be in [2, 2^31)");
    const uint32_t m = n_sym + static_cast<uint32_t>(ell) + 1;
    const uint32_t dq = static_cast<uint32_t>(ell) + 1;
    const FlatMatrixT<Elt> R = ComputeSymmetricPowerMatrixFM(A, m);
    const std::vector<uint32_t> rows = QuotientLComplementRows(n_sym, static_cast<uint32_t>(ell));
    FlatMatrixT<Elt> out = FlatMatrixT<Elt>::CreateZero(dq, dq, ell);
    std::vector<Elt> v(static_cast<size_t>(m) + 1);
    for (uint32_t t = 0; t < dq; ++t) {
        for (uint32_t r = 0; r <= m; ++r) v[r] = R.get_entry(r, rows[t]);   // rho_m(A) E, column t
        const std::vector<Elt> w = QuotientLReduce(v, n_sym, ell);
        for (uint32_t s = 0; s < dq; ++s)
            if (w[s] != 0) out.set_entry(s, t, w[s]);
    }
    return out;
}

uint32_t FibreDimension(FibreRepresentation rep, uint32_t n_sym, uint64_t ell) {
    if (rep == FibreRepresentation::QuotientL) return static_cast<uint32_t>(ell) + 1;
    return n_sym + 1;
}

template std::vector<uint32_t> QuotientLReduce<uint32_t>(std::vector<uint32_t>&, uint32_t, uint32_t);
template std::vector<uint64_t> QuotientLReduce<uint64_t>(std::vector<uint64_t>&, uint32_t, uint64_t);
template FlatMatrixT<uint32_t> QuotientLProjectionMatrixFM<uint32_t>(uint32_t, uint32_t);
template FlatMatrixT<uint64_t> QuotientLProjectionMatrixFM<uint64_t>(uint32_t, uint64_t);
template FlatMatrixT<uint32_t> ComputeQuotientLMatrixFM<uint32_t>(const FlatMatrixT<uint32_t>&, uint32_t);
template FlatMatrixT<uint64_t> ComputeQuotientLMatrixFM<uint64_t>(const FlatMatrixT<uint64_t>&, uint32_t);
