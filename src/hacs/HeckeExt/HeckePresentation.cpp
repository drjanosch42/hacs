// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2 (slot 6, WP-2; OQ11).  See HeckePresentation.h for
// the mathematics, the [X] / [E] / [C] citations and the certificate contract.
//
// Like HeckeLocalAlgebra.cpp this TU carries ZERO raw mod_* sites: every scalar
// operation is a MatrixT product, an RrefRowSpan insert / membership test or a
// hecke::AlgebraCoords read-off, each of which owns its arithmetic.

#include "HeckePresentation.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <sstream>
#include <stdexcept>

namespace hecke_ext {

namespace {

using hecke::RrefRowSpan;
using hecke::VecOfMatrix;

ExtMatrix ColumnOf(const std::vector<uint32_t>& v, uint32_t ell) {
    ExtMatrix c(static_cast<uint32_t>(v.size()), 1u, ell);
    for (uint32_t i = 0; i < c.rows; ++i) c.data[i][0] = v[i];
    return c;
}

std::vector<uint32_t> VecOfColumn(const ExtMatrix& c) {
    std::vector<uint32_t> v(c.rows, 0u);
    for (uint32_t i = 0; i < c.rows; ++i) v[i] = c.data[i][0];
    return v;
}

std::vector<uint32_t> Unit(uint32_t n, uint32_t j) {
    std::vector<uint32_t> e(n, 0u);
    e[j] = 1u;
    return e;
}

// The multiplication data of A_S in its stored RREF basis b_0..b_{dA-1}.
struct AlgebraTable {
    uint32_t D = 0, dA = 0, ell = 0;
    ExtMatrix closure;               // dA x D^2, the RREF coordinate rows
    std::vector<uint32_t> pivots;

    // The dA x dA matrix of multiplication by x on A_S: column j holds the
    // coordinates of x * b_j.  Throws (AlgebraCoords) if a product leaves the
    // span, i.e. if the stored basis is not closed under multiplication.
    ExtMatrix mult(const ExtMatrix& x) const {
        ExtMatrix M(dA, dA, ell);
        for (uint32_t j = 0; j < dA; ++j) {
            const ExtMatrix bj = hecke::MatrixOfVec(closure.data[j], D, D, ell);
            const std::vector<uint32_t> c =
                    hecke::AlgebraCoords(closure, pivots, x * bj, ell, "HeckePresentation");
            for (uint32_t k = 0; k < dA; ++k) M.data[k][j] = c[k];
        }
        return M;
    }
};

AlgebraTable BuildTable(const std::vector<ExtMatrix>& a_basis, uint32_t D, uint32_t ell) {
    AlgebraTable T;
    T.D = D;
    T.dA = static_cast<uint32_t>(a_basis.size());
    T.ell = ell;
    T.closure = ExtMatrix(T.dA, D * D, ell);
    for (uint32_t i = 0; i < T.dA; ++i) T.closure.data[i] = VecOfMatrix(a_basis[i]);
    T.pivots = hecke::RrefPivots(T.closure);
    return T;
}

// The action of an A_S-element on A_S^t, as a (t dA) x (t dA) block diagonal
// of t copies of its multiplication matrix.
ExtMatrix OnTuples(const ExtMatrix& mult, uint32_t t, uint32_t ell) {
    const uint32_t dA = mult.rows;
    ExtMatrix M(t * dA, t * dA, ell);
    for (uint32_t c = 0; c < t; ++c) {
        for (uint32_t i = 0; i < dA; ++i) {
            for (uint32_t j = 0; j < dA; ++j) M.data[c * dA + i][c * dA + j] = mult.data[i][j];
        }
    }
    return M;
}

// The evaluation map A_S^t -> V: D x (t dA), column i*dA + j = b_j v_i.
ExtMatrix EvaluationMatrix(const std::vector<ExtMatrix>& a_basis,
                           const std::vector<uint32_t>& gens, uint32_t D, uint32_t ell) {
    const uint32_t dA = static_cast<uint32_t>(a_basis.size());
    const uint32_t t = static_cast<uint32_t>(gens.size());
    ExtMatrix E(D, t * dA, ell);
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t j = 0; j < dA; ++j) {
            for (uint32_t r = 0; r < D; ++r) E.data[r][i * dA + j] = a_basis[j].data[r][gens[i]];
        }
    }
    return E;
}

// m_S V as a row span of F_ell^D (every column of every m_S basis element).
RrefRowSpan ImagesOf(const std::vector<ExtMatrix>& mats, uint32_t D, uint32_t ell) {
    RrefRowSpan span(D, ell);
    for (const ExtMatrix& A : mats) {
        for (uint32_t j = 0; j < D; ++j) {
            std::vector<uint32_t> col(D, 0u);
            for (uint32_t i = 0; i < D; ++i) col[i] = A.data[i][j];
            span.insert(std::move(col));
        }
    }
    return span;
}

// Adds A_S e_j (every column j of every basis element) to `span`.
void InsertOrbitOfUnit(RrefRowSpan& span, const std::vector<ExtMatrix>& a_basis,
                       uint32_t j, uint32_t D) {
    for (const ExtMatrix& b : a_basis) {
        std::vector<uint32_t> v(D, 0u);
        for (uint32_t i = 0; i < D; ++i) v[i] = b.data[i][j];
        span.insert(std::move(v));
    }
}

// The least-index Nakayama lift: e_j is kept iff it is not in
// m_S V + sum_{kept i} A_S e_i.  Returns the kept indices; `span_out` ends as
// that sum (which is V when the lift generates).
std::vector<uint32_t> LeastIndexLift(const std::vector<ExtMatrix>& a_basis,
                                     const std::vector<ExtMatrix>& m_basis,
                                     uint32_t D, uint32_t ell, uint32_t* dim_mV_out,
                                     uint32_t* dim_span_out) {
    RrefRowSpan span = ImagesOf(m_basis, D, ell);
    if (dim_mV_out) *dim_mV_out = span.dim();
    std::vector<uint32_t> kept;
    for (uint32_t j = 0; j < D; ++j) {
        if (span.contains(Unit(D, j))) continue;
        kept.push_back(j);
        InsertOrbitOfUnit(span, a_basis, j, D);
    }
    if (dim_span_out) *dim_span_out = span.dim();
    return kept;
}

// The action matrices on A_S^t of every A_S basis element and of every m_S
// basis element.
struct TupleActions {
    std::vector<ExtMatrix> by_a;   // index-aligned with a_basis
    std::vector<ExtMatrix> by_m;   // index-aligned with m_basis
};

TupleActions BuildTupleActions(const AlgebraTable& T, const std::vector<ExtMatrix>& a_basis,
                               const std::vector<ExtMatrix>& m_basis, uint32_t t) {
    TupleActions out;
    for (const ExtMatrix& b : a_basis) out.by_a.push_back(OnTuples(T.mult(b), t, T.ell));
    for (const ExtMatrix& m : m_basis) out.by_m.push_back(OnTuples(T.mult(m), t, T.ell));
    return out;
}

// The A_S-submodule of A_S^t generated by `vs`, as an F_ell row span.
RrefRowSpan ASpanOf(const std::vector<std::vector<uint32_t>>& vs, const TupleActions& act,
                    uint32_t width, uint32_t ell) {
    RrefRowSpan span(width, ell);
    for (const std::vector<uint32_t>& v : vs) {
        const ExtMatrix c = ColumnOf(v, ell);
        for (const ExtMatrix& L : act.by_a) span.insert(VecOfColumn(L * c));
    }
    return span;
}

// m_S R for R spanned (over F_ell) by `rows`.
RrefRowSpan MTimes(const std::vector<std::vector<uint32_t>>& rows, const TupleActions& act,
                   uint32_t width, uint32_t ell) {
    RrefRowSpan span(width, ell);
    for (const std::vector<uint32_t>& r : rows) {
        const ExtMatrix c = ColumnOf(r, ell);
        for (const ExtMatrix& L : act.by_m) span.insert(VecOfColumn(L * c));
    }
    return span;
}

// R = ker(evaluation), in its canonical RREF F_ell-basis.
RrefRowSpan RelationModule(const ExtMatrix& E, uint32_t ell) {
    const ExtMatrix K = hecke::KernelAsMatrix(E);
    RrefRowSpan span(E.cols, ell);
    for (uint32_t c = 0; c < K.cols; ++c) {
        std::vector<uint32_t> v(K.rows, 0u);
        for (uint32_t i = 0; i < K.rows; ++i) v[i] = K.data[i][c];
        span.insert(std::move(v));
    }
    return span;
}

bool IsZeroVec(const std::vector<uint32_t>& v) {
    for (uint32_t x : v) {
        if (x != 0u) return false;
    }
    return true;
}

} // namespace

// ===========================================================================
// Construction
// ===========================================================================
Presentation ComputePresentation(const LocalAlgebra& L) {
    Presentation P;
    if (!L.ok) return P;

    P.available = true;
    P.ell = L.ell;
    P.D = L.D;
    P.f = L.f;
    P.dim_A = L.dim_A;
    P.algebra = L.cyclicity;

    const std::vector<ExtMatrix>& a = P.algebra.a_basis;
    const std::vector<ExtMatrix>& m = P.algebra.m_basis;
    ENGINE_CHECK_MSG(static_cast<uint32_t>(a.size()) == P.dim_A,
                     "ComputePresentation: the stored A_S basis has the wrong size");

    // --- generators ---------------------------------------------------------
    uint32_t dim_mV = 0, dim_span = 0;
    P.generator_indices = LeastIndexLift(a, m, P.D, P.ell, &dim_mV, &dim_span);
    const uint32_t t = static_cast<uint32_t>(P.generator_indices.size());
    ENGINE_CHECK_MSG(dim_span == P.D, "ComputePresentation: the Nakayama lift does not span V");
    ENGINE_CHECK_MSG(t * P.f == P.D - dim_mV,
                     "ComputePresentation: t * f != dim V / m_S V (Nakayama)");
    ENGINE_CHECK_MSG(t == L.min_generators,
                     "ComputePresentation: t disagrees with the local algebra's min_generators");
    P.min_generators = t;
    P.generators = ExtMatrix(P.D, t, P.ell);
    for (uint32_t i = 0; i < t; ++i) P.generators.data[P.generator_indices[i]][i] = 1u;

    // --- relations ----------------------------------------------------------
    const uint32_t width = t * P.dim_A;
    const ExtMatrix E = EvaluationMatrix(a, P.generator_indices, P.D, P.ell);
    const RrefRowSpan R = RelationModule(E, P.ell);
    P.dim_relation_module = R.dim();
    ENGINE_CHECK_MSG(P.dim_relation_module + P.D == width,
                     "ComputePresentation: dim R != t * dim A_S - D");

    const AlgebraTable T = BuildTable(a, P.D, P.ell);
    const TupleActions act = BuildTupleActions(T, a, m, t);
    RrefRowSpan W = MTimes(R.rows(), act, width, P.ell);
    const uint32_t dim_mR = W.dim();
    for (const std::vector<uint32_t>& r : R.rows()) {
        if (W.contains(r)) continue;
        P.relations.push_back(r);
        const ExtMatrix c = ColumnOf(r, P.ell);
        for (const ExtMatrix& Lb : act.by_a) W.insert(VecOfColumn(Lb * c));
    }
    ENGINE_CHECK_MSG(W.dim() == P.dim_relation_module,
                     "ComputePresentation: the chosen relations do not generate R");
    P.min_relations = static_cast<uint32_t>(P.relations.size());
    ENGINE_CHECK_MSG(P.min_relations * P.f == P.dim_relation_module - dim_mR,
                     "ComputePresentation: min_relations * f != dim R / m_S R (Nakayama)");

    const uint32_t code = P.verify();
    ENGINE_CHECK_MSG(code == 0, "ComputePresentation: the certificate fails (verify() != 0)");
    return P;
}

// ===========================================================================
// The certificate -- re-derives, reads nothing it concludes from.
// ===========================================================================
uint32_t Presentation::verify() const {
    if (!available) return 2;

    // 1: A_S and m_S, re-derived from the stored t_p by their own certificate.
    if (algebra.verify() != 0) return 1;

    // 2: shapes.
    const std::vector<ExtMatrix>& a = algebra.a_basis;
    const std::vector<ExtMatrix>& m = algebra.m_basis;
    if (algebra.ell != ell || algebra.D != D || algebra.f != f) return 2;
    if (D == 0 || f == 0 || dim_A != static_cast<uint32_t>(a.size())) return 2;
    const uint32_t t = min_generators;
    if (generator_indices.size() != t) return 2;
    for (size_t i = 0; i < generator_indices.size(); ++i) {
        if (generator_indices[i] >= D) return 2;
        if (i > 0 && generator_indices[i] <= generator_indices[i - 1]) return 2;
    }
    if (generators.rows != D || generators.cols != t || generators.p != ell) return 2;
    for (uint32_t i = 0; i < t; ++i) {
        for (uint32_t r = 0; r < D; ++r) {
            if (generators.data[r][i] != (r == generator_indices[i] ? 1u : 0u)) return 2;
        }
    }
    const uint32_t width = t * dim_A;
    if (relations.size() != min_relations) return 2;
    for (const std::vector<uint32_t>& r : relations) {
        if (r.size() != width) return 2;
        for (uint32_t x : r) {
            if (x >= ell) return 2;
        }
    }

    // 3: A_S closed under multiplication in the stored basis.
    AlgebraTable T;
    TupleActions act;
    try {
        T = BuildTable(a, D, ell);
        act = BuildTupleActions(T, a, m, t);
    } catch (const std::exception&) {
        return 3;
    }

    // 4: the least-index Nakayama lift, re-derived, and its count as a rank.
    uint32_t dim_mV = 0, dim_span = 0;
    const std::vector<uint32_t> lift = LeastIndexLift(a, m, D, ell, &dim_mV, &dim_span);
    if (lift != generator_indices) return 4;
    if (t * f != D - dim_mV) return 4;

    // 5: the generators span V over A_S.
    const ExtMatrix E = EvaluationMatrix(a, generator_indices, D, ell);
    if (hecke::RankOf(E) != D) return 5;

    // 6: every relation evaluates to 0.
    for (const std::vector<uint32_t>& r : relations) {
        if (!IsZeroVec(VecOfColumn(E * ColumnOf(r, ell)))) return 6;
    }

    // 7: the relations generate R over A_S (R recomputed as a kernel).
    const RrefRowSpan R = RelationModule(E, ell);
    if (R.dim() != dim_relation_module) return 7;
    if (R.dim() + D != width) return 7;
    if (ASpanOf(relations, act, width, ell).dim() != R.dim()) return 7;

    // 8: minimality -- min_relations * f == dim R / m_S R.
    const uint32_t dim_mR = MTimes(R.rows(), act, width, ell).dim();
    if (min_relations * f != R.dim() - dim_mR) return 8;

    return 0;
}

const char* Presentation::verify_message(uint32_t code) {
    switch (code) {
        case 0: return "ok";
        case 1: return "the embedded A_S / m_S certificate does not re-derive";
        case 2: return "shape, index or availability mismatch";
        case 3: return "the stored A_S basis is not closed under multiplication";
        case 4: return "the generators are not the least-index Nakayama lift, or t * f != dim V/m_S V";
        case 5: return "the generators do not span V over A_S";
        case 6: return "a stored relation does not vanish on the generators";
        case 7: return "the stored relations do not generate the relation module";
        case 8: return "min_relations * f != dim R / m_S R (not minimal)";
        default: return "unknown code";
    }
}

std::string Presentation::to_json() const {
    std::ostringstream os;
    os << "{\"schema\":\"hecke_presentation/v1\",\"available\":" << (available ? "true" : "false")
       << ",\"ell\":" << ell << ",\"D\":" << D << ",\"f\":" << f << ",\"dim_A\":" << dim_A
       << ",\"min_generators\":" << min_generators << ",\"min_relations\":" << min_relations
       << ",\"dim_relation_module\":" << dim_relation_module << ",\"generators\":[";
    for (size_t i = 0; i < generator_indices.size(); ++i) {
        if (i) os << ",";
        os << generator_indices[i];
    }
    os << "],\"relations\":[";
    for (size_t k = 0; k < relations.size(); ++k) {
        if (k) os << ",";
        os << "[";
        for (size_t i = 0; i < relations[k].size(); ++i) {
            if (i) os << ",";
            os << relations[k][i];
        }
        os << "]";
    }
    os << "]}";
    return os.str();
}

} // namespace hecke_ext
