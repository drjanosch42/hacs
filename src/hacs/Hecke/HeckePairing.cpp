// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 6, WP-3 (phase E7b) -- see HeckePairing.h.
// [C] = docs/latex/gl2_contract.tex, cited by label.

#include "HeckePairing.h"

#include "../LocalSystemSL2Z.h"
#include "../LocalSystemUtils.h"
#include "../../ModArithmetic/BlockMat.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "HeckeSettings.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hecke {

namespace {

using Elt = uint32_t;

uint32_t coeff_mod(int64_t c, uint32_t ell) {
    const int64_t m = static_cast<int64_t>(ell);
    int64_t r = c % m;
    if (r < 0) r += m;
    return static_cast<uint32_t>(r);
}

FlatMatrix zero(uint32_t r, uint32_t c, uint32_t ell) { return FlatMatrix::CreateZero(r, c, ell); }

FlatMatrix flat_of_multivector(const engine::BlockMat<Elt>& B, uint32_t ell) {
    const engine::DenseBuf<Elt> D = B.to_flat(engine::Layout::RowMajor);
    const uint32_t R = D.rows, C = D.cols;
    if (R == 0 || C == 0) return zero(R, C, ell);
    std::vector<Elt> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(i) * C + j] = D.at(i, j);
    return FlatMatrix(R, C, ell, d);
}

// Plain exact products (the shapes here are a few hundred at most).
FlatMatrix mul(const FlatMatrix& A, const FlatMatrix& B, uint32_t ell) {
    ENGINE_CHECK_MSG(A.get_cols() == B.get_rows(), "HeckePairing: product shapes disagree");
    const uint32_t R = A.get_rows(), K = A.get_cols(), C = B.get_cols();
    if (R == 0 || C == 0) return zero(R, C, ell);
    std::vector<uint64_t> acc(static_cast<size_t>(R) * C, 0);
    std::vector<uint32_t> a(static_cast<size_t>(R) * K), b(static_cast<size_t>(K) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t k = 0; k < K; ++k) a[static_cast<size_t>(i) * K + k] = A.get_entry(i, k);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t j = 0; j < C; ++j) b[static_cast<size_t>(k) * C + j] = B.get_entry(k, j);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            const uint64_t x = a[static_cast<size_t>(i) * K + k];
            if (x == 0) continue;
            for (uint32_t j = 0; j < C; ++j) {
                uint64_t& s = acc[static_cast<size_t>(i) * C + j];
                s = (s + x * b[static_cast<size_t>(k) * C + j]) % ell;
            }
        }
    std::vector<Elt> d(acc.size());
    for (size_t i = 0; i < acc.size(); ++i) d[i] = static_cast<Elt>(acc[i]);
    return FlatMatrix(R, C, ell, d);
}

FlatMatrix tr(const FlatMatrix& A, uint32_t ell) {
    const uint32_t R = A.get_rows(), C = A.get_cols();
    if (R == 0 || C == 0) return zero(C, R, ell);
    std::vector<Elt> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(j) * R + i] = A.get_entry(i, j);
    return FlatMatrix(C, R, ell, d);
}

bool all_zero(const FlatMatrix& A) {
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            if (A.get_entry(i, j) != 0) return false;
    return true;
}

bool same(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols()) return false;
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            if (A.get_entry(i, j) != B.get_entry(i, j)) return false;
    return true;
}

uint32_t rank_of(const FlatMatrix& A) {
    if (A.get_rows() == 0 || A.get_cols() == 0) return 0;
    return A.rank();
}

bool is_perfect(const FlatMatrix& G) {
    return G.get_rows() == G.get_cols() && rank_of(G) == G.get_rows();
}

uint32_t num_simplices(const SimplicialComplex& X, uint32_t q) {
    return q < X.get_simplices().size() ? X.get_num_q_simplices(q) : 0u;
}

uint32_t index_of(const SimplicialComplex& X, uint32_t q, const std::vector<uint32_t>& verts) {
    const int64_t i = X.find_simplex_index(q, verts);
    ENGINE_CHECK_MSG(i >= 0, "HeckePairing: a face of a simplex is not in the complex");
    return static_cast<uint32_t>(i);
}

// Stack two blocks with equal column counts.
FlatMatrix vstack(const FlatMatrix& top, const FlatMatrix& bottom, uint32_t ell) {
    ENGINE_CHECK_MSG(top.get_cols() == bottom.get_cols(), "HeckePairing: vstack column counts disagree");
    const uint32_t R = top.get_rows() + bottom.get_rows(), C = top.get_cols();
    if (R == 0 || C == 0) return zero(R, C, ell);
    std::vector<Elt> d(static_cast<size_t>(R) * C);
    for (uint32_t i = 0; i < top.get_rows(); ++i)
        for (uint32_t j = 0; j < C; ++j) d[static_cast<size_t>(i) * C + j] = top.get_entry(i, j);
    for (uint32_t i = 0; i < bottom.get_rows(); ++i)
        for (uint32_t j = 0; j < C; ++j)
            d[static_cast<size_t>(top.get_rows() + i) * C + j] = bottom.get_entry(i, j);
    return FlatMatrix(R, C, ell, d);
}

// Parity of the permutation sorting `v` into increasing order (+1 / -1).
int32_t sort_parity(std::vector<uint32_t> v) {
    int32_t s = 1;
    for (size_t i = 0; i < v.size(); ++i)
        for (size_t j = 0; j + 1 < v.size() - i; ++j)
            if (v[j] > v[j + 1]) { std::swap(v[j], v[j + 1]); s = -s; }
    return s;
}

FlatMatrix fm22(const Mat22& g, uint32_t ell) {
    const std::array<uint32_t, 4> m = g.reduced_mod(ell);
    return FlatMatrix(2, 2, ell, std::vector<uint32_t>{ m[0], m[1], m[2], m[3] });
}

} // namespace

// ---------------------------------------------------------------------------
// 1. the relative fundamental 2-chain
// ---------------------------------------------------------------------------
ModelFundamentalChain BuildModelFundamentalChain(const SimplicialComplex& K, const SimplicialComplex& K_bd) {
    ENGINE_CHECK_MSG(K.get_simplices().size() == 3, "BuildModelFundamentalChain: K is not two-dimensional");
    ENGINE_CHECK_MSG(K_bd.get_simplices().size() == 2, "BuildModelFundamentalChain: K_bd is not one-dimensional");
    const auto& faces = K.get_simplices()[2];
    const uint32_t F = static_cast<uint32_t>(faces.size());
    const uint32_t E = K.get_num_q_simplices(1);
    ENGINE_CHECK_MSG(F >= 1 && F <= 20, "BuildModelFundamentalChain: the sign search is exhaustive (F <= 20)");

    ModelFundamentalChain out;
    for (uint32_t q = 0; q < 2; ++q)
        for (const auto& s : K_bd.get_simplices()[q]) out.bd_simplex_in_K[q].push_back(index_of(K, q, s));

    // Edge classes in C_1(X)_Gamma: rep[e], sign[e] with [e] = sign[e] [rep[e]].
    std::vector<uint32_t> rep(E);
    std::vector<int32_t> sgn(E, 1);
    for (uint32_t e = 0; e < E; ++e) rep[e] = e;
    for (const SL2ZRelationSeed& seed : SL2ZRelationSeeds_7VertexModel()) {
        if (seed.q_dim != 1) continue;
        const uint32_t lhs = index_of(K, 1, seed.lhs_vertices);
        const uint32_t rhs = index_of(K, 1, seed.rhs_vertices);
        // g . [rhs, increasing] = [image_of_rhs as ordered] = parity * [lhs, increasing].
        std::vector<uint32_t> img = seed.image_of_rhs;
        std::vector<uint32_t> sorted_img = img;
        std::sort(sorted_img.begin(), sorted_img.end());
        ENGINE_CHECK_MSG(sorted_img == seed.lhs_vertices, "BuildModelFundamentalChain: a seed's vertex correspondence "
                                                          "does not land on its left-hand simplex");
        const int32_t parity = sort_parity(img);
        ENGINE_CHECK_MSG(parity == 1, "BuildModelFundamentalChain: a degree-1 relation reverses orientation -- the "
                                      "ordered-complex (Alexander-Whitney) model needs order-preserving relations");
        // Merge lhs into rhs's class (both are singletons or already merged).
        const uint32_t rl = rep[lhs], rr = rep[rhs];
        if (rl == rr) continue;
        const int32_t sl = sgn[lhs] * parity * sgn[rhs];   // [lhs] = parity [rhs]
        for (uint32_t e = 0; e < E; ++e)
            if (rep[e] == rl) { rep[e] = rr; sgn[e] *= sl; }
    }
    std::vector<char> is_bd(E, 0);
    for (uint32_t e : out.bd_simplex_in_K[1]) is_bd[e] = 1;
    for (uint32_t e = 0; e < E; ++e)
        for (uint32_t e2 = 0; e2 < E; ++e2)
            if (e != e2 && rep[e] == rep[e2])
                ENGINE_CHECK_MSG(!is_bd[e] && !is_bd[e2], "BuildModelFundamentalChain: a boundary edge is identified "
                                                          "with another edge");

    // d f for f = [a, b, c]: +[b, c] - [a, c] + [a, b].
    std::vector<std::array<std::pair<uint32_t, int32_t>, 3>> dface(F);
    for (uint32_t f = 0; f < F; ++f) {
        const auto& v = faces[f];
        dface[f] = { std::make_pair(index_of(K, 1, { v[1], v[2] }), +1),
                     std::make_pair(index_of(K, 1, { v[0], v[2] }), -1),
                     std::make_pair(index_of(K, 1, { v[0], v[1] }), +1) };
    }
    std::vector<int32_t> chosen;
    int32_t chosen_h = 0;
    for (uint32_t mask = 0; mask < (1u << F); ++mask) {
        std::vector<int32_t> eps(F);
        for (uint32_t f = 0; f < F; ++f) eps[f] = (mask >> f & 1u) ? -1 : +1;
        std::vector<int64_t> cls(E, 0);
        for (uint32_t f = 0; f < F; ++f)
            for (const auto& t : dface[f]) {
                cls[rep[t.first]] += static_cast<int64_t>(eps[f]) * t.second * sgn[t.first];
            }
        bool ok = true;
        int64_t h = 0;
        for (uint32_t e = 0; e < E && ok; ++e) {
            if (rep[e] != e) continue;
            if (is_bd[e]) {
                if (h == 0) h = cls[e];
                ok = (cls[e] == h) && (h == 1 || h == -1);
            } else {
                ok = (cls[e] == 0);
            }
        }
        if (!ok || h == 0) continue;
        ++out.qualifying_sign_vectors;
        if (h == -1) { chosen = eps; chosen_h = -1; }
    }
    if (out.qualifying_sign_vectors != 2 || chosen.empty())
        throw std::runtime_error("BuildModelFundamentalChain: the cycle condition has " +
                                 std::to_string(out.qualifying_sign_vectors) + " solutions (expected +-eps)");
    out.face_coeff = chosen;
    out.horocycle_coefficient = chosen_h;
    out.boundary_coeff.assign(K_bd.get_num_q_simplices(1), chosen_h);
    return out;
}

// ---------------------------------------------------------------------------
// 2. the cup-product evaluation form and the restriction
// ---------------------------------------------------------------------------
FlatMatrix CupEvaluationForm(const SimplicialComplex& X, uint32_t a, uint32_t b, const std::vector<int32_t>& coeff,
                             const FlatMatrix& fibre_gram, uint32_t ell) {
    const uint32_t d = fibre_gram.get_rows();
    ENGINE_CHECK_MSG(fibre_gram.get_cols() == d, "CupEvaluationForm: the fibre Gram is not square");
    const uint32_t k = a + b;
    const uint32_t Na = num_simplices(X, a), Nb = num_simplices(X, b), Nk = num_simplices(X, k);
    ENGINE_CHECK_MSG(coeff.size() == Nk, "CupEvaluationForm: one chain coefficient per top simplex");
    const uint32_t R = Na * d, C = Nb * d;
    if (R == 0 || C == 0) return zero(R, C, ell);
    std::vector<Elt> q(static_cast<size_t>(R) * C, 0);
    for (uint32_t s = 0; s < Nk; ++s) {
        if (coeff[s] == 0) continue;
        const auto& v = X.get_simplices()[k][s];
        const std::vector<uint32_t> front(v.begin(), v.begin() + a + 1), back(v.begin() + a, v.end());
        const uint32_t fi = index_of(X, a, front), bi = index_of(X, b, back);
        const uint32_t c = coeff_mod(coeff[s], ell);
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                Elt& e = q[static_cast<size_t>(fi * d + i) * C + (bi * d + j)];
                e = static_cast<Elt>((e + static_cast<uint64_t>(c) * fibre_gram.get_entry(i, j)) % ell);
            }
    }
    return FlatMatrix(R, C, ell, q);
}

FlatMatrix RestrictionFlat(const SimplicialComplex& K, const SimplicialComplex& K_bd, uint32_t q, uint32_t d_V,
                           uint32_t ell) {
    const uint32_t Nb = num_simplices(K_bd, q), Nk = num_simplices(K, q);
    const uint32_t R = Nb * d_V, C = Nk * d_V;
    if (R == 0 || C == 0) return zero(R, C, ell);
    std::vector<Elt> m(static_cast<size_t>(R) * C, 0);
    for (uint32_t s = 0; s < Nb; ++s) {
        const uint32_t sk = index_of(K, q, K_bd.get_simplices()[q][s]);
        for (uint32_t i = 0; i < d_V; ++i) m[static_cast<size_t>(s * d_V + i) * C + (sk * d_V + i)] = 1u % ell;
    }
    return FlatMatrix(R, C, ell, m);
}

bool PairingAdjoint(const MatrixT<uint32_t>& A_c, const FlatMatrix& G, const MatrixT<uint32_t>& A_D, uint32_t factor,
                    uint32_t ell, bool* nonvacuous_out) {
    ENGINE_CHECK_MSG(A_c.rows == A_c.cols && A_D.rows == A_D.cols && G.get_rows() == A_c.rows &&
                         G.get_cols() == A_D.rows,
                     "PairingAdjoint: A_c (d_c x d_c), G (d_c x d_D), A_D (d_D x d_D) shapes disagree");
    const uint32_t dc = A_c.rows, dD = A_D.rows;
    bool equal = true, nonvacuous = false;
    for (uint32_t i = 0; i < dc; ++i)
        for (uint32_t j = 0; j < dD; ++j) {
            uint64_t lhs = 0, ga = 0;
            for (uint32_t k = 0; k < dc; ++k) lhs = (lhs + static_cast<uint64_t>(A_c.data[k][i]) * G.get_entry(k, j)) % ell;
            for (uint32_t k = 0; k < dD; ++k) ga = (ga + static_cast<uint64_t>(G.get_entry(i, k)) * A_D.data[k][j]) % ell;
            if (ga != 0) nonvacuous = true;
            if (lhs != (ga * (factor % ell)) % ell) equal = false;
        }
    if (nonvacuous_out) *nonvacuous_out = nonvacuous;
    return equal;
}

// ---------------------------------------------------------------------------
// the certificate
// ---------------------------------------------------------------------------
uint32_t PairingCertificate::verify() const {
    std::vector<std::string> f;
    return verify(f);
}

std::vector<uint32_t> PairingCertificate::failing_items() const {
    std::vector<std::string> f;
    verify(f);
    std::vector<uint32_t> items;
    for (const std::string& s : f) {
        const uint32_t k = static_cast<uint32_t>(s[0] - '0');
        if (std::find(items.begin(), items.end(), k) == items.end()) items.push_back(k);
    }
    return items;
}

uint32_t PairingCertificate::verify(std::vector<std::string>& failures) const {
    failures.clear();
    auto fail = [&](uint32_t item, const std::string& what) {
        failures.push_back(std::to_string(item) + " " + what);
    };
    // 1: the fibre Gram is the identity.
    if (!same(fibre_gram, FlatMatrix::CreateScalar(d_V, 1u % ell, ell))) fail(1, "fibre Gram != I");
    const bool shapes = Q.size() == 3 && Zc.size() == 3 && Bc.size() == 3 && Hc.size() == 3 && ZD.size() == 3 &&
                        BD.size() == 3 && HD.size() == 3 && Qb.size() == 2 && ZbM.size() == 2 && BbM.size() == 2 &&
                        HbM.size() == 2 && ZbD.size() == 2 && BbD.size() == 2 && HbD.size() == 2 && G.size() == 3 &&
                        Gb.size() == 2 && perfect.size() == 3 && boundary_perfect.size() == 2;
    if (!shapes) { fail(4, "misshaped certificate"); return static_cast<uint32_t>(failures.size()); }
    auto conform = [](const FlatMatrix& A, const FlatMatrix& Qm, const FlatMatrix& B) {
        return A.get_rows() == Qm.get_rows() && Qm.get_cols() == B.get_rows();
    };
    // 2: cone-level descent.
    for (uint32_t t = 0; t < 3; ++t) {
        const uint32_t u = 2 - t;
        if (!conform(Zc[t], Q[t], BD[u]) || !conform(Bc[t], Q[t], ZD[u])) { fail(2, "shapes at t=" + std::to_string(t)); continue; }
        if (!all_zero(mul(mul(tr(Zc[t], ell), Q[t], ell), BD[u], ell)))
            fail(2, "P(Z_c, B_D) != 0 at t=" + std::to_string(t));
        if (!all_zero(mul(mul(tr(Bc[t], ell), Q[t], ell), ZD[u], ell)))
            fail(2, "P(B_c, Z_D) != 0 at t=" + std::to_string(t));
    }
    // 3: circle-level descent.
    for (uint32_t s = 0; s < 2; ++s) {
        const uint32_t u = 1 - s;
        if (!conform(ZbM[s], Qb[s], BbD[u]) || !conform(BbM[s], Qb[s], ZbD[u])) { fail(3, "shapes at s=" + std::to_string(s)); continue; }
        if (!all_zero(mul(mul(tr(ZbM[s], ell), Qb[s], ell), BbD[u], ell)))
            fail(3, "P_bd(Z_M, B_D) != 0 at s=" + std::to_string(s));
        if (!all_zero(mul(mul(tr(BbM[s], ell), Qb[s], ell), ZbD[u], ell)))
            fail(3, "P_bd(B_M, Z_D) != 0 at s=" + std::to_string(s));
    }
    // 4: the Grams recomputed.
    for (uint32_t t = 0; t < 3; ++t) {
        if (!conform(Hc[t], Q[t], HD[2 - t])) { fail(4, "Gram shapes at t=" + std::to_string(t)); continue; }
        if (!same(G[t], mul(mul(tr(Hc[t], ell), Q[t], ell), HD[2 - t], ell))) fail(4, "G^" + std::to_string(t) + " != H_c^T Q H_D");
    }
    for (uint32_t s = 0; s < 2; ++s) {
        if (!conform(HbM[s], Qb[s], HbD[1 - s])) { fail(4, "boundary Gram shapes at s=" + std::to_string(s)); continue; }
        if (!same(Gb[s], mul(mul(tr(HbM[s], ell), Qb[s], ell), HbD[1 - s], ell)))
            fail(4, "G_bd^" + std::to_string(s) + " != H_bM^T Qb H_bD");
    }
    // 5: record honesty.
    for (uint32_t t = 0; t < 3; ++t)
        if ((perfect[t] != 0) != is_perfect(G[t])) fail(5, "recorded perfect^" + std::to_string(t) + " is wrong");
    for (uint32_t s = 0; s < 2; ++s)
        if ((boundary_perfect[s] != 0) != is_perfect(Gb[s]))
            fail(5, "recorded boundary_perfect^" + std::to_string(s) + " is wrong");
    return static_cast<uint32_t>(failures.size());
}

// ---------------------------------------------------------------------------
// one cell
// ---------------------------------------------------------------------------
PoincareDualityPairing::PoincareDualityPairing(const ContragredientCell& cell)
    : cell_(&cell), ell_(cell.ell()), n_(cell.n_sym()), d_V_(cell.n_sym() + 1) {
    // (slot-8 WP-3) [E] sec:ring (b): a pairing's perfectness is a field notion.
    RequireFieldCoefficientRing("PoincareDualityPairing");
    const SimplicialComplex& K = cell.K();
    pair_M_ = std::make_unique<SL2ZBoundaryPair>(&cell.coh_M(), K);
    cone_ = pair_M_->make_cone();
    const SimplicialComplex& K_bd = pair_M_->K_bd();
    ENGINE_CHECK_MSG(cone_->q_max() == 2, "PoincareDualityPairing: the cone does not report H^0_c .. H^2_c");

    // The D_n circle model: the Sym^n boundary relation list with rho_n(g) -> rho_n^vee(g).
    const Mat22 T = Mat22::T(), negI = Mat22::I().neg();
    const FlatMatrix rho_T = ComputeSymmetricPowerMatrixFM(fm22(T, ell_), n_);
    const FlatMatrix rho_negI = ComputeSymmetricPowerMatrixFM(fm22(negI, ell_), n_);
    const FlatMatrix dual_T = ComputeContragredientPowerMatrixFM(fm22(T, ell_), n_);
    const FlatMatrix dual_negI = ComputeContragredientPowerMatrixFM(fm22(negI, ell_), n_);
    for (const auto& rel : GenerateSL2ZBoundaryRelationsElt<uint32_t>(K_bd, n_, ell_)) {
        const bool is_T = rel.tag == "cusp_T";
        const bool is_negI = rel.tag.rfind("negI", 0) == 0;
        ENGINE_CHECK_MSG(is_T != is_negI, "PoincareDualityPairing: an unrecognised boundary relation tag");
        ENGINE_CHECK_MSG(same(rel.M, is_T ? rho_T : rho_negI),
                         "PoincareDualityPairing: a boundary relation is not rho_n of its generator");
        rels_bd_D_.emplace_back(rel.q_dim, rel.s_lhs_idx_in_Kq, rel.s_rhs_idx_in_Kq, is_T ? dual_T : dual_negI, rel.tag);
    }
    lssc_bd_D_ = std::make_unique<LocalSystemSimplicialComplex>(K_bd, rels_bd_D_, d_V_, ell_);
    coh_bd_D_ = std::make_unique<EquivariantCohomologyOnK>(lssc_bd_D_.get());
    coh_bd_D_->compute_cohomology();

    chain_ = BuildModelFundamentalChain(K, K_bd);

    PairingCertificate& c = cert_;
    c.ell = ell_; c.n_sym = n_; c.d_V = d_V_;
    c.fibre_gram = PairingGram(n_, ell_);
    const EquivariantCohomologyOnK& cD = cell.coh_D();
    for (uint32_t q = 0; q < 3; ++q) {
        c.ZD.push_back(flat_of_multivector(cD.Z_eq_block(q), ell_));
        c.BD.push_back(flat_of_multivector(cD.B_eq_block(q), ell_));
        c.HD.push_back(flat_of_multivector(cD.H_eq_block(q), ell_));
    }
    for (uint32_t t = 0; t < 3; ++t) {
        const uint32_t u = 2 - t;
        ENGINE_CHECK_MSG(cone_->bulk_part_dim(t) == num_simplices(K, t) * d_V_ &&
                             cone_->boundary_part_dim(t) == (t == 0 ? 0u : num_simplices(K_bd, t - 1) * d_V_),
                         "PoincareDualityPairing: the cone's flat layout is not C^t(K) (+) C^{t-1}(K_bd)");
        const FlatMatrix top = CupEvaluationForm(K, t, u, chain_.face_coeff, c.fibre_gram, ell_);
        FlatMatrix bottom = zero(0, num_simplices(K, u) * d_V_, ell_);
        if (t >= 1) {
            const FlatMatrix Qbd = CupEvaluationForm(K_bd, t - 1, u, chain_.boundary_coeff, c.fibre_gram, ell_);
            bottom = mul(Qbd, RestrictionFlat(K, K_bd, u, d_V_, ell_), ell_);
        }
        c.Q.push_back(vstack(top, bottom, ell_));
        c.Zc.push_back(flat_of_multivector(cone_->Z_block(t), ell_));
        c.Bc.push_back(flat_of_multivector(cone_->B_block(t), ell_));
        c.Hc.push_back(flat_of_multivector(cone_->H_block(t), ell_));
        c.G.push_back(mul(mul(tr(c.Hc[t], ell_), c.Q[t], ell_), c.HD[u], ell_));
        c.perfect.push_back(is_perfect(c.G[t]) ? 1 : 0);
    }
    const EquivariantCohomologyOnK& bM = pair_M_->boundary_cohomology();
    for (uint32_t s = 0; s < 2; ++s) {
        c.ZbM.push_back(flat_of_multivector(bM.Z_eq_block(s), ell_));
        c.BbM.push_back(flat_of_multivector(bM.B_eq_block(s), ell_));
        c.HbM.push_back(flat_of_multivector(bM.H_eq_block(s), ell_));
        c.ZbD.push_back(flat_of_multivector(coh_bd_D_->Z_eq_block(s), ell_));
        c.BbD.push_back(flat_of_multivector(coh_bd_D_->B_eq_block(s), ell_));
        c.HbD.push_back(flat_of_multivector(coh_bd_D_->H_eq_block(s), ell_));
        c.Qb.push_back(CupEvaluationForm(K_bd, s, 1 - s, chain_.boundary_coeff, c.fibre_gram, ell_));
    }
    for (uint32_t s = 0; s < 2; ++s) {
        c.Gb.push_back(mul(mul(tr(c.HbM[s], ell_), c.Qb[s], ell_), c.HbD[1 - s], ell_));
        c.boundary_perfect.push_back(is_perfect(c.Gb[s]) ? 1 : 0);
    }
    std::vector<std::string> failures;
    if (c.verify(failures) != 0) {
        std::ostringstream oss;
        oss << "PoincareDualityPairing (n=" << n_ << ", ell=" << ell_ << "): PairingCertificate fails:";
        for (const std::string& s : failures) oss << " [" << s << "]";
        throw std::runtime_error(oss.str());
    }
}

PoincareDualityPairing::~PoincareDualityPairing() = default;

const FlatMatrix& PoincareDualityPairing::cone_form(uint32_t t) const {
    ENGINE_CHECK_MSG(t < 3, "PoincareDualityPairing::cone_form: t out of range");
    return cert_.Q[t];
}
const FlatMatrix& PoincareDualityPairing::gram(uint32_t t) const {
    ENGINE_CHECK_MSG(t < 3, "PoincareDualityPairing::gram: t out of range");
    return cert_.G[t];
}
bool PoincareDualityPairing::perfect(uint32_t t) const {
    ENGINE_CHECK_MSG(t < 3, "PoincareDualityPairing::perfect: t out of range");
    return cert_.perfect[t] != 0;
}
const FlatMatrix& PoincareDualityPairing::boundary_form(uint32_t s) const {
    ENGINE_CHECK_MSG(s < 2, "PoincareDualityPairing::boundary_form: s out of range");
    return cert_.Qb[s];
}
const FlatMatrix& PoincareDualityPairing::boundary_gram(uint32_t s) const {
    ENGINE_CHECK_MSG(s < 2, "PoincareDualityPairing::boundary_gram: s out of range");
    return cert_.Gb[s];
}
bool PoincareDualityPairing::boundary_perfect(uint32_t s) const {
    ENGINE_CHECK_MSG(s < 2, "PoincareDualityPairing::boundary_perfect: s out of range");
    return cert_.boundary_perfect[s] != 0;
}

uint32_t PoincareDualityPairing::evaluate_direct(uint32_t t, const FlatMatrix& x, const FlatMatrix& y) const {
    ENGINE_CHECK_MSG(t < 3, "PoincareDualityPairing::evaluate_direct: t out of range");
    const SimplicialComplex& K = cell_->K();
    const SimplicialComplex& K_bd = pair_M_->K_bd();
    const uint32_t u = 2 - t, d = d_V_;
    ENGINE_CHECK_MSG(x.get_cols() == 1 && y.get_cols() == 1 && x.get_rows() == cone_->flat_dim(t) &&
                         y.get_rows() == num_simplices(K, u) * d,
                     "PoincareDualityPairing::evaluate_direct: cochain shapes");
    const FlatMatrix& G0 = cert_.fibre_gram;
    auto fibre = [&](uint32_t xoff, uint32_t yoff) {
        uint64_t acc = 0;
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j)
                acc = (acc + static_cast<uint64_t>(x.get_entry(xoff + i, 0)) * G0.get_entry(i, j) % ell_ *
                                 y.get_entry(yoff + j, 0)) % ell_;
        return acc;
    };
    uint64_t total = 0;
    // < c u y, [K] >: c is the bulk part of x.
    const auto& faces = K.get_simplices()[2];
    for (uint32_t f = 0; f < faces.size(); ++f) {
        const auto& v = faces[f];
        const std::vector<uint32_t> front(v.begin(), v.begin() + t + 1), back(v.begin() + t, v.end());
        const uint64_t val = fibre(index_of(K, t, front) * d, index_of(K, u, back) * d);
        total = (total + static_cast<uint64_t>(coeff_mod(chain_.face_coeff[f], ell_)) * val) % ell_;
    }
    // < c_bd u r y, d[K] >: c_bd is the boundary part of x, r y read off K's simplices.
    if (t >= 1) {
        const uint32_t off = cone_->bulk_part_dim(t);
        const auto& edges = K_bd.get_simplices()[1];
        for (uint32_t e = 0; e < edges.size(); ++e) {
            const auto& v = edges[e];
            const std::vector<uint32_t> front(v.begin(), v.begin() + t), back(v.begin() + (t - 1), v.end());
            const uint64_t val = fibre(off + index_of(K_bd, t - 1, front) * d, index_of(K, u, back) * d);
            total = (total + static_cast<uint64_t>(coeff_mod(chain_.boundary_coeff[e], ell_)) * val) % ell_;
        }
    }
    return static_cast<uint32_t>(total);
}

std::vector<DualityGoldenObject> PairingGoldenObjects(const PoincareDualityPairing& pd,
                                                      const std::vector<char>& pd_adjoint_per_t) {
    ENGINE_CHECK_MSG(pd_adjoint_per_t.size() == 3, "PairingGoldenObjects: one adjointness verdict per t = 0..2");
    std::vector<DualityGoldenObject> out;
    for (uint32_t t = 0; t < 3; ++t) {
        const uint64_t perf = pd.perfect(t) ? 1u : 0u;
        out.push_back(DualityGoldenObject{ "pairing_perfect^" + std::to_string(t), 'A', perf, 0,
                                           engine::FnvU64(engine::kFnvOffset64, perf) });
        const uint64_t adj = pd_adjoint_per_t[t] ? 1u : 0u;
        out.push_back(DualityGoldenObject{ "pd_adjoint^" + std::to_string(t), 'A', adj, 0,
                                           engine::FnvU64(engine::kFnvOffset64, adj) });
        const FlatMatrix& G = pd.gram(t);
        uint64_t h = engine::FnvU64(engine::kFnvOffset64, G.get_rows());
        h = engine::FnvU64(h, G.get_cols());
        for (uint32_t r = 0; r < G.get_rows(); ++r)
            for (uint32_t c = 0; c < G.get_cols(); ++c) h = engine::FnvU64(h, static_cast<uint64_t>(G.get_entry(r, c)));
        out.push_back(DualityGoldenObject{ "pairing_matrix^" + std::to_string(t), 'B', G.get_rows(), G.get_cols(), h });
    }
    return out;
}

} // namespace hecke
