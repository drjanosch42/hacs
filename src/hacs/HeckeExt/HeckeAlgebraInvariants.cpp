// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// HeckeAlgebraInvariants.cpp -- see HeckeAlgebraInvariants.h.
// (hecke-extensions slot 9, WP-13 / console phase C5b.)
// ===========================================================================

#include "HeckeAlgebraInvariants.h"

#include "../Hecke/FpPoly.h"
#include "../Hecke/SubspaceOps.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {

namespace {

using M = AlgMatrix;

M Zero(uint32_t r, uint32_t c, uint32_t ell) { return M(r, c, ell); }

M Identity(uint32_t d, uint32_t ell) { return M(d, ell); }

uint32_t RankOf(const M& m) {
    M copy(m);
    return copy.rank();
}

// Columns of `k` (a matrix whose COLUMNS are vectors of length k.rows) as
// vectors.
std::vector<std::vector<uint32_t>> ColumnsOf(const M& k) {
    std::vector<std::vector<uint32_t>> out(k.cols, std::vector<uint32_t>(k.rows, 0u));
    for (uint32_t c = 0; c < k.cols; ++c)
        for (uint32_t r = 0; r < k.rows; ++r) out[c][r] = k.data[r][c];
    return out;
}

// The algebra element sum_i c_i basis[i].
M ElementOf(const std::vector<M>& basis, const std::vector<uint32_t>& c, uint32_t D,
            uint32_t ell) {
    M out = Zero(D, D, ell);
    for (size_t i = 0; i < basis.size(); ++i) {
        if (c[i] == 0) continue;
        out = out + basis[i] * c[i];
    }
    return out;
}

// The canonical (RREF) basis of the span of `elts` as D x D matrices.
std::vector<M> CanonicalSpan(const std::vector<M>& elts, uint32_t D, uint32_t ell,
                             hecke::RrefRowSpan* span_out = nullptr) {
    hecke::RrefRowSpan span(D * D, ell);
    for (const M& e : elts) span.insert(hecke::VecOfMatrix(e));
    std::vector<M> out;
    out.reserve(span.dim());
    for (const std::vector<uint32_t>& row : span.rows())
        out.push_back(hecke::MatrixOfVec(row, D, D, ell));
    if (span_out != nullptr) *span_out = span;
    return out;
}

// Vectors (length `width`) as the columns of a width x n matrix.
M ColumnsMatrix(const std::vector<std::vector<uint32_t>>& vs, uint32_t width, uint32_t ell) {
    M out = Zero(width, static_cast<uint32_t>(vs.size()), ell);
    for (size_t c = 0; c < vs.size(); ++c)
        for (uint32_t r = 0; r < width; ++r) out.data[r][c] = vs[c][r];
    return out;
}

void RequireFamily(const std::vector<M>& gens, uint32_t ell, uint32_t& D) {
    if (gens.empty()) throw std::invalid_argument("algebra invariants: the operator family is empty");
    if (ell < 2) throw std::invalid_argument("algebra invariants: ell must be a prime >= 2");
    D = gens[0].rows;
    if (D == 0) throw std::invalid_argument("algebra invariants: the operators act on a 0-dimensional space");
    for (const M& g : gens) {
        if (g.rows != g.cols) throw std::invalid_argument("algebra invariants: an operator is not square");
        if (g.rows != D) throw std::invalid_argument("algebra invariants: the operators differ in size");
        if (g.p != ell) throw std::invalid_argument("algebra invariants: an operator is not over F_ell");
    }
    for (size_t i = 0; i < gens.size(); ++i)
        for (size_t j = i + 1; j < gens.size(); ++j)
            if (!((gens[i] * gens[j]) == (gens[j] * gens[i])))
                throw std::invalid_argument(
                    "algebra invariants: the operators do not commute (operators " +
                    std::to_string(i) + " and " + std::to_string(j) + ")");
}

uint32_t Lcm(uint32_t a, uint32_t b) {
    uint32_t x = a, y = b;
    while (y != 0) { const uint32_t t = x % y; x = y; y = t; }
    const uint64_t l = static_cast<uint64_t>(a) / x * b;
    ENGINE_CHECK_MSG(l < (1ull << 31), "JordanChevalleySplit: lcm of the factor degrees overflows");
    return static_cast<uint32_t>(l);
}


uint32_t LeastFrobeniusExponent(uint32_t ell, uint32_t D) {
    ENGINE_CHECK_MSG(ell >= 2, "LeastFrobeniusExponent: ell < 2");
    uint32_t e = 1;
    uint64_t pw = ell;
    while (pw < D) {
        pw *= ell;
        ++e;
    }
    return e;
}
} // namespace

AlgebraInvariants ComputeAlgebraInvariants(const std::vector<AlgMatrix>& generators,
                                           uint32_t ell) {
    uint32_t D = 0;
    RequireFamily(generators, ell, D);

    AlgebraInvariants out;
    out.ell = ell;
    out.D = D;

    // A = F_ell[generators]: the canonical closure (rows = vec of a basis).
    uint32_t dimA = 0;
    const M closure = hecke::algebra_closure(generators, D, ell, dimA);
    out.dim_A = dimA;
    out.basis.reserve(dimA);
    for (uint32_t i = 0; i < dimA; ++i)
        out.basis.push_back(hecke::MatrixOfVec(closure.data[i], D, D, ell));

    // rad A = ker Frob^e, ell^e >= D.
    out.frobenius_exponent = LeastFrobeniusExponent(ell, D);
    const M F = hecke::frobenius_matrix(closure, D, ell);          // dimA x dimA
    const M rad_coords = hecke::frobenius_kernel(F, out.frobenius_exponent);   // dimA x r
    std::vector<M> rad_elts;
    for (const std::vector<uint32_t>& c : ColumnsOf(rad_coords))
        rad_elts.push_back(ElementOf(out.basis, c, D, ell));
    hecke::RrefRowSpan rad_span(D * D, ell);
    out.radical = CanonicalSpan(rad_elts, D, ell, &rad_span);
    out.dim_rad = static_cast<uint32_t>(out.radical.size());
    ENGINE_CHECK_MSG(out.dim_rad == rad_coords.cols,
                     "ComputeAlgebraInvariants: the radical's spanning set is dependent");
    ENGINE_CHECK_MSG(out.dim_rad < dimA,
                     "ComputeAlgebraInvariants: the identity lies in the radical");

    // Berlekamp: m = dim{x : x^ell - x in rad} - dim rad = dimA - rank[F - I | R].
    {
        M G = F - Identity(dimA, ell);
        M GR = Zero(dimA, dimA + rad_coords.cols, ell);
        for (uint32_t r = 0; r < dimA; ++r) {
            for (uint32_t c = 0; c < dimA; ++c) GR.data[r][c] = G.data[r][c];
            for (uint32_t c = 0; c < rad_coords.cols; ++c) GR.data[r][dimA + c] = rad_coords.data[r][c];
        }
        const uint32_t rk = RankOf(GR);
        ENGINE_CHECK_MSG(rk <= dimA, "ComputeAlgebraInvariants: rank exceeds the algebra dimension");
        out.components = dimA - rk;
        ENGINE_CHECK_MSG(out.components >= 1,
                         "ComputeAlgebraInvariants: a nonzero algebra with no maximal ideal");
    }
    out.local = (out.components == 1);
    out.residue_degree = out.local ? (dimA - out.dim_rad) : 0u;

    // The powers of the radical (commutative: the products of bases span).
    std::vector<std::vector<M>> powers;   // powers[i] = basis of rad^{i+1}
    if (out.dim_rad > 0) powers.push_back(out.radical);
    while (!powers.empty() && !powers.back().empty()) {
        std::vector<M> prods;
        for (const M& r : out.radical)
            for (const M& s : powers.back()) prods.push_back(r * s);
        std::vector<M> next = CanonicalSpan(prods, D, ell);
        ENGINE_CHECK_MSG(next.size() < powers.back().size(),
                         "ComputeAlgebraInvariants: the radical's powers do not decrease");
        if (next.empty()) break;
        powers.push_back(std::move(next));
    }
    for (const auto& p : powers) out.radical_power_dims.push_back(static_cast<uint32_t>(p.size()));
    out.nilpotency_index = static_cast<uint32_t>(powers.size()) + 1u;

    // rad / rad^2.
    {
        hecke::RrefRowSpan sq(D * D, ell);
        if (powers.size() > 1)
            for (const M& e : powers[1]) sq.insert(hecke::VecOfMatrix(e));
        for (const M& r : out.radical) {
            if (sq.insert(hecke::VecOfMatrix(r))) out.tangent.push_back(r);
        }
        out.dim_tangent = static_cast<uint32_t>(out.tangent.size());
        const uint32_t dim_sq = powers.size() > 1 ? static_cast<uint32_t>(powers[1].size()) : 0u;
        ENGINE_CHECK_MSG(out.dim_tangent + dim_sq == out.dim_rad,
                         "ComputeAlgebraInvariants: dim rad/rad^2 bookkeeping");
    }

    // soc A = Ann_A(rad A): coordinates c with (sum c_i b_i) r = 0 for every r.
    if (out.dim_rad == 0) {
        out.socle = out.basis;
    } else {
        const uint32_t n = D * D;
        M S = Zero(out.dim_rad * n, dimA, ell);
        for (uint32_t j = 0; j < out.dim_rad; ++j) {
            for (uint32_t i = 0; i < dimA; ++i) {
                const std::vector<uint32_t> v = hecke::VecOfMatrix(out.basis[i] * out.radical[j]);
                for (uint32_t k = 0; k < n; ++k) S.data[j * n + k][i] = v[k];
            }
        }
        std::vector<M> soc_elts;
        for (const std::vector<uint32_t>& c : ColumnsOf(hecke::KernelAsMatrix(S)))
            soc_elts.push_back(ElementOf(out.basis, c, D, ell));
        out.socle = CanonicalSpan(soc_elts, D, ell);
    }
    out.dim_socle = static_cast<uint32_t>(out.socle.size());
    out.gorenstein = out.local && out.dim_socle == out.residue_degree;

    // Local: the Hilbert function in units of f.
    if (out.local) {
        const uint32_t f = out.residue_degree;
        std::vector<uint32_t> dims;   // dim rad^i, i = 0 .. nu
        dims.push_back(dimA);
        for (uint32_t d : out.radical_power_dims) dims.push_back(d);
        dims.push_back(0u);
        for (size_t i = 0; i + 1 < dims.size(); ++i) {
            const uint32_t layer = dims[i] - dims[i + 1];
            ENGINE_CHECK_MSG(layer % f == 0,
                             "ComputeAlgebraInvariants: a radical layer is not a residue-field space");
            out.hilbert.push_back(layer / f);
        }
    }
    return out;
}

namespace {

// Validates V and returns its rank-checked copy.
void RequireSubspace(const AlgebraInvariants& A, const M& V, const char* who) {
    if (V.rows != A.D)
        throw std::invalid_argument(std::string(who) + ": the subspace's height is not the operators' size");
    if (V.cols > 0 && RankOf(V) != V.cols)
        throw std::invalid_argument(std::string(who) + ": the subspace's columns are dependent");
    if (V.cols == 0) return;
    for (const M& b : A.basis) {
        try {
            (void)hecke::SolveColumnsExact(V, b * V, who);
        } catch (const std::exception&) {
            throw std::invalid_argument(std::string(who) +
                                        ": the subspace is not stable under the algebra");
        }
    }
}

// The columns of V (D x k) as vectors.
std::vector<std::vector<uint32_t>> Cols(const M& V) { return ColumnsOf(V); }

} // namespace

ModuleSeries LoewySeries(const AlgebraInvariants& A, const AlgMatrix& V) {
    RequireSubspace(A, V, "loewy");
    ModuleSeries out;
    const uint32_t D = A.D, ell = A.ell;
    // rad^0 V = V.
    hecke::RrefRowSpan cur(D, ell);
    for (const auto& c : Cols(V)) cur.insert(c);
    out.dims.push_back(cur.dim());
    while (cur.dim() > 0) {
        hecke::RrefRowSpan next(D, ell);
        const M X = ColumnsMatrix(cur.rows(), D, ell);   // D x dim
        for (const M& r : A.radical) {
            const M RX = r * X;
            for (const auto& c : Cols(RX)) next.insert(c);
        }
        ENGINE_CHECK_MSG(next.dim() < cur.dim(), "LoewySeries: the radical series does not decrease");
        out.dims.push_back(next.dim());
        cur = next;
    }
    for (size_t j = 0; j + 1 < out.dims.size(); ++j) out.layers.push_back(out.dims[j] - out.dims[j + 1]);
    return out;
}

ModuleSeries SocleSeries(const AlgebraInvariants& A, const AlgMatrix& V) {
    RequireSubspace(A, V, "socle_series");
    ModuleSeries out;
    const uint32_t D = A.D, ell = A.ell, k = V.cols;
    out.dims.push_back(0u);
    if (k == 0) return out;
    // soc^j as a D x s matrix of columns; soc^0 = 0.
    M S = Zero(D, 0, ell);
    while (true) {
        // W: the rows spanning the annihilator of span(S) in (F^D)^*.
        M W;
        if (S.cols == 0) {
            W = Identity(D, ell);
        } else {
            M St = Zero(S.cols, D, ell);
            for (uint32_t r = 0; r < D; ++r)
                for (uint32_t c = 0; c < S.cols; ++c) St.data[c][r] = S.data[r][c];
            const M K = hecke::KernelAsMatrix(St);   // D x (D - s): columns w with w.S = 0
            W = Zero(K.cols, D, ell);
            for (uint32_t r = 0; r < D; ++r)
                for (uint32_t c = 0; c < K.cols; ++c) W.data[c][r] = K.data[r][c];
        }
        // Stack W r V over the radical basis; its kernel in F^k is soc^{j+1}.
        std::vector<std::vector<uint32_t>> rows;
        if (W.rows > 0) {
            if (A.radical.empty()) {
                // rad = 0: every x in V satisfies the (empty) condition.
            } else {
                for (const M& r : A.radical) {
                    const M WrV = W * (r * V);
                    for (uint32_t i = 0; i < WrV.rows; ++i) rows.push_back(WrV.data[i]);
                }
            }
        }
        M Y;
        if (rows.empty()) {
            Y = Identity(k, ell);
        } else {
            M stack = Zero(static_cast<uint32_t>(rows.size()), k, ell);
            for (size_t i = 0; i < rows.size(); ++i) stack.data[i] = rows[i];
            Y = hecke::KernelAsMatrix(stack);
        }
        const M next = V * Y;   // D x dim
        const uint32_t dim = next.cols == 0 ? 0u : RankOf(next);
        ENGINE_CHECK_MSG(dim > S.cols || (dim == k && S.cols == k),
                         "SocleSeries: the socle series does not increase");
        out.dims.push_back(dim);
        if (dim == k) break;
        // Canonicalise soc^{j+1}.
        hecke::RrefRowSpan sp(D, ell);
        for (const auto& c : Cols(next)) sp.insert(c);
        S = ColumnsMatrix(sp.rows(), D, ell);
    }
    for (size_t j = 0; j + 1 < out.dims.size(); ++j) out.layers.push_back(out.dims[j + 1] - out.dims[j]);
    return out;
}

JordanChevalley JordanChevalleySplit(const AlgMatrix& t) {
    if (t.rows != t.cols) throw std::invalid_argument("nilpotent_part: the operator is not square");
    const uint32_t D = t.rows;
    const uint32_t ell = t.p;
    if (ell < 2) throw std::invalid_argument("nilpotent_part: ell must be a prime >= 2");
    JordanChevalley out;
    if (D == 0) {
        out.semisimple = t;
        out.nilpotent = t;
        out.lcm_degree = 1;
        out.exponent = 1;
        return out;
    }
    const hecke::FpPoly chi = hecke::CharPolyBerkowitz(t);
    const hecke::FpPolyFactorization fac = hecke::FactorPoly(chi);
    uint32_t L = 1;
    for (const auto& fct : fac.factors) L = Lcm(L, static_cast<uint32_t>(fct.pi.degree()));
    // N: the least multiple of L with ell^N >= D.
    uint32_t N = L;
    {
        uint64_t pw = 1;
        uint32_t e = 0;
        while (pw < D) { pw *= ell; ++e; }
        while (N < e) N += L;
    }
    out.lcm_degree = L;
    out.exponent = N;
    out.semisimple = hecke::FrobeniusPower(t, ell, N);
    out.nilpotent = t - out.semisimple;
    // Re-verify the split.
    ENGINE_CHECK_MSG((out.semisimple * t) == (t * out.semisimple),
                     "JordanChevalleySplit: s does not commute with t");
    ENGINE_CHECK_MSG(hecke::IsNilpotent(out.nilpotent, D),
                     "JordanChevalleySplit: t - s is not nilpotent");
    {
        const hecke::FpPoly mu = hecke::MinPolyKrylov(out.semisimple);
        const hecke::FpPoly g = hecke::FpPoly::gcd(mu, mu.derivative());
        ENGINE_CHECK_MSG(g.degree() == 0, "JordanChevalleySplit: s is not semisimple");
    }
    return out;
}

} // namespace hecke_ext
