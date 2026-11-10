// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 11, WP-5 -- see HeckeDicksonTheta.h.

#include "HeckeDicksonTheta.h"

#include "HeckeDicksonMultiplication.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace hecke_ext {

namespace {

inline uint32_t mulm(uint32_t a, uint32_t b, uint32_t p) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % p);
}
inline uint32_t addm(uint32_t a, uint32_t b, uint32_t p) {
    const uint64_t s = static_cast<uint64_t>(a) + b;
    return static_cast<uint32_t>(s >= p ? s - p : s);
}
inline uint32_t subm(uint32_t a, uint32_t b, uint32_t p) { return a >= b ? a - b : a + p - b; }
inline uint32_t invm(uint32_t a, uint32_t p) {
    // a^(p-2) mod p, p prime, a != 0.
    uint64_t r = 1, b = a % p;
    uint32_t e = p - 2;
    while (e) {
        if (e & 1u) r = (r * b) % p;
        b = (b * b) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

// In-place RREF; returns the pivot columns.
std::vector<uint32_t> Rref(ThetaMat& A) {
    const uint32_t p = A.p;
    std::vector<uint32_t> piv;
    uint32_t r = 0;
    for (uint32_t c = 0; c < A.cols && r < A.rows; ++c) {
        uint32_t k = r;
        while (k < A.rows && A.at(k, c) == 0) ++k;
        if (k == A.rows) continue;
        if (k != r)
            for (uint32_t j = 0; j < A.cols; ++j) std::swap(A.at(k, j), A.at(r, j));
        const uint32_t inv = invm(A.at(r, c), p);
        for (uint32_t j = 0; j < A.cols; ++j) A.at(r, j) = mulm(A.at(r, j), inv, p);
        for (uint32_t i = 0; i < A.rows; ++i) {
            if (i == r || A.at(i, c) == 0) continue;
            const uint32_t f = A.at(i, c);
            for (uint32_t j = 0; j < A.cols; ++j)
                if (A.at(r, j) != 0) A.at(i, j) = subm(A.at(i, j), mulm(f, A.at(r, j), p), p);
        }
        piv.push_back(c);
        ++r;
    }
    return piv;
}

ThetaMat HConcat(const ThetaMat& A, const ThetaMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.rows && A.p == B.p, "HConcat: row counts / moduli differ");
    ThetaMat C(A.rows, A.cols + B.cols, A.p);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = A.at(i, j);
        for (uint32_t j = 0; j < B.cols; ++j) C.at(i, A.cols + j) = B.at(i, j);
    }
    return C;
}

// Solve A x = b exactly; false if b is not in colspan(A).
bool Solve(const ThetaMat& A, const std::vector<uint32_t>& b, std::vector<uint32_t>& x) {
    ENGINE_CHECK_MSG(b.size() == A.rows, "Solve: rhs length != rows");
    ThetaMat Ab(A.rows, A.cols + 1, A.p);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) Ab.at(i, j) = A.at(i, j);
        Ab.at(i, A.cols) = b[i];
    }
    const std::vector<uint32_t> piv = Rref(Ab);
    if (!piv.empty() && piv.back() == A.cols) return false;
    x.assign(A.cols, 0);
    for (uint32_t r = 0; r < piv.size(); ++r) x[piv[r]] = Ab.at(r, A.cols);
    return true;
}

FlatMatrix Two(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t ell) {
    return FlatMatrix(2, 2, ell, std::vector<uint32_t>{ a, b, c, d });
}

bool FlatEqual(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols()) return false;
    for (uint32_t r = 0; r < A.get_rows(); ++r)
        for (uint32_t c = 0; c < A.get_cols(); ++c)
            if (A.get_entry(r, c) != B.get_entry(r, c)) return false;
    return true;
}

bool FlatIsZero(const FlatMatrix& A) {
    for (uint32_t r = 0; r < A.get_rows(); ++r)
        for (uint32_t c = 0; c < A.get_cols(); ++c)
            if (A.get_entry(r, c) != 0) return false;
    return true;
}

ThetaMat FromFlat(const FlatMatrix& F) {
    ThetaMat M(F.get_rows(), F.get_cols(), F.get_p());
    for (uint32_t r = 0; r < M.rows; ++r)
        for (uint32_t c = 0; c < M.cols; ++c) M.at(r, c) = F.get_entry(r, c);
    return M;
}

std::vector<uint32_t> PolyMul(const std::vector<uint32_t>& f, const std::vector<uint32_t>& g, uint32_t p) {
    std::vector<uint32_t> out(f.size() + g.size() - 1, 0);
    for (size_t i = 0; i < f.size(); ++i) {
        if (f[i] == 0) continue;
        for (size_t j = 0; j < g.size(); ++j)
            if (g[j] != 0) out[i + j] = addm(out[i + j], mulm(f[i], g[j], p), p);
    }
    return out;
}

} // namespace

// ===========================================================================
// ThetaMat algebra
// ===========================================================================
bool ThetaMat::is_zero() const {
    for (uint32_t v : a)
        if (v != 0) return false;
    return true;
}

MatrixT<uint32_t> ThetaMat::to_matrix() const {
    MatrixT<uint32_t> M(rows, cols, p);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < cols; ++c) M.data[r][c] = at(r, c);
    return M;
}

uint32_t ThetaRank(const ThetaMat& A) {
    ThetaMat B = A;
    return static_cast<uint32_t>(Rref(B).size());
}

ThetaMat ThetaMul(const ThetaMat& A, const ThetaMat& B) {
    ENGINE_CHECK_MSG(A.cols == B.rows && A.p == B.p, "ThetaMul: shapes / moduli disagree");
    ThetaMat C(A.rows, B.cols, A.p);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t k = 0; k < A.cols; ++k) {
            const uint32_t f = A.at(i, k);
            if (f == 0) continue;
            for (uint32_t j = 0; j < B.cols; ++j)
                if (B.at(k, j) != 0) C.at(i, j) = addm(C.at(i, j), mulm(f, B.at(k, j), A.p), A.p);
        }
    return C;
}

ThetaMat ThetaKernelBasis(const ThetaMat& A) {
    ThetaMat R = A;
    const std::vector<uint32_t> piv = Rref(R);
    std::vector<char> is_piv(A.cols, 0);
    for (uint32_t c : piv) is_piv[c] = 1;
    std::vector<uint32_t> free_cols;
    for (uint32_t c = 0; c < A.cols; ++c)
        if (!is_piv[c]) free_cols.push_back(c);
    ThetaMat K(A.cols, static_cast<uint32_t>(free_cols.size()), A.p);
    for (uint32_t k = 0; k < free_cols.size(); ++k) {
        const uint32_t f = free_cols[k];
        K.at(f, k) = 1u % A.p;
        for (uint32_t r = 0; r < piv.size(); ++r)
            if (R.at(r, f) != 0) K.at(piv[r], k) = subm(0, R.at(r, f), A.p);
    }
    return K;
}

ThetaMat ThetaImageBasis(const ThetaMat& A) {
    ThetaMat R = A;
    const std::vector<uint32_t> piv = Rref(R);
    ThetaMat I(A.rows, static_cast<uint32_t>(piv.size()), A.p);
    for (uint32_t k = 0; k < piv.size(); ++k)
        for (uint32_t r = 0; r < A.rows; ++r) I.at(r, k) = A.at(r, piv[k]);
    return I;
}

bool ThetaSpanContains(const ThetaMat& big, const ThetaMat& small) {
    if (small.cols == 0) return true;
    return ThetaRank(HConcat(big, small)) == ThetaRank(big);
}

namespace {
ExactnessSpot CheckExactnessAt(const std::string& name, const ThetaMat& A, const ThetaMat& B) {
    ENGINE_CHECK_MSG(A.rows == B.cols, "CheckExactnessAt: A's target is not B's source");
    ExactnessSpot s;
    s.name = name;
    s.dim = A.rows;
    const ThetaMat im = ThetaImageBasis(A);
    const ThetaMat ker = ThetaKernelBasis(B);
    s.rank_in = im.cols;
    s.dim_ker_out = ker.cols;
    s.composite_zero = (A.cols == 0 || B.rows == 0) ? true : ThetaMul(B, A).is_zero();
    s.im_in_ker = ThetaSpanContains(ker, im);
    s.ker_in_im = ThetaSpanContains(im, ker);
    return s;
}
} // namespace

std::vector<ExactnessSpot> CheckSequenceExactness(const ThetaMat& iota0, const ThetaMat& pi0, const ThetaMat& delta0,
                                                  const ThetaMat& iota1, const ThetaMat& pi1, const ThetaMat& delta1,
                                                  const std::vector<uint32_t>& dims) {
    ENGINE_CHECK_MSG(dims.size() == 6, "CheckSequenceExactness: six dimensions expected");
    const uint32_t p = iota0.p;
    const ThetaMat zero_in(dims[0], 0, p);   // 0 -> H^0(Sym^n)
    std::vector<ExactnessSpot> out;
    out.push_back(CheckExactnessAt("H^0(Sym^n)", zero_in, iota0));
    out.push_back(CheckExactnessAt("H^0(Sym^m)", iota0, pi0));
    out.push_back(CheckExactnessAt("H^0(Q_n)", pi0, delta0));
    out.push_back(CheckExactnessAt("H^1(Sym^n)", delta0, iota1));
    out.push_back(CheckExactnessAt("H^1(Sym^m)", iota1, pi1));
    out.push_back(CheckExactnessAt("H^1(Q_n)", pi1, delta1));
    return out;
}

// ===========================================================================
// Fibre level
// ===========================================================================
namespace {
int64_t FirstSampleNotPreserving(const FlatMatrix& S, uint32_t m, uint32_t ell, const std::vector<uint32_t>& samples) {
    ENGINE_CHECK_MSG(S.get_rows() == m + 1 && samples.size() % 4 == 0, "FirstSampleNotPreserving: bad input");
    const ThetaMat Sm = FromFlat(S);
    const uint32_t rS = ThetaRank(Sm);
    for (size_t s = 0; s + 3 < samples.size(); s += 4) {
        const FlatMatrix A = Two(samples[s], samples[s + 1], samples[s + 2], samples[s + 3], ell);
        const FlatMatrix img = ComputeSymmetricPowerMatrixFM(A, m) * S;
        if (ThetaRank(HConcat(Sm, FromFlat(img))) != rS) return static_cast<int64_t>(s / 4);
    }
    return -1;
}

FlatMatrix XEllYMultiplicationMatrix(uint32_t n, uint32_t ell) {
    const uint32_t m = n + ell + 1;
    FlatMatrix X = FlatMatrix::CreateZero(m + 1, n + 1, ell);
    for (uint32_t j = 0; j <= n; ++j) X.set_entry(j + 1, j, 1u % ell);   // x^ell y . x^{n-j} y^j = u^{(m)}_{j+1}
    return X;
}
} // namespace

std::vector<uint32_t> DicksonMonomial(uint32_t a, uint32_t b, uint32_t ell) {
    std::vector<uint32_t> Lc(ell + 2, 0);
    Lc[1] = 1u % ell;
    Lc[ell] = subm(0, 1u % ell, ell);
    const uint32_t dD = ell * ell - ell;
    std::vector<uint32_t> Dc(dD + 1, 0);
    for (uint32_t i = 0; i <= ell; ++i) Dc[i * (ell - 1)] = 1u % ell;
    std::vector<uint32_t> f{ 1u % ell };
    for (uint32_t i = 0; i < a; ++i) f = PolyMul(f, Lc, ell);
    for (uint32_t i = 0; i < b; ++i) f = PolyMul(f, Dc, ell);
    return f;
}

std::vector<std::pair<uint32_t, uint32_t>> DicksonExponents(uint32_t n, uint32_t ell) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    const uint32_t dL = ell + 1, dD = ell * ell - ell;
    for (uint32_t a = 0; a * dL <= n; ++a) {
        const uint32_t rest = n - a * dL;
        if (rest % dD == 0) out.emplace_back(a, rest / dD);
    }
    return out;
}

const char* DicksonThetaCertificate::verify_message(uint32_t code) {
    switch (code) {
        case 0: return "ok";
        case 1: return "L. fails its own certificate (CertifyDicksonMultiplication)";
        case 2: return "pi L. != 0, rank pi != ell + 1, or pi E != I";
        case 3: return "a sample g does not preserve L . Sym^n (pi rho_m(g) L. != 0)";
        case 4: return "rho_Q(g) pi != pi rho_m(g) on a sample";
        case 5: return "rho_Q is not multiplicative on a sample pair";
        case 6: return "the negative seam: x^ell y . Sym^n is preserved by every sample";
        case 7: return "a relation matrix of Q_n differs from rho_Q(g_r)";
        case 8: return "a pipeline projector fails P H = I or P B = 0";
        case 9: return "an image cochain violates a relation or is not a cocycle";
        case 10: return "the connecting map: no equivariant lift, or d(lift) not divisible by L.";
        case 11: return "H^2 != 0 on some term";
        case 12: return "a spot of the sequence is not exact";
        case 13: return "the alternating dimension sum is not 0";
        case 14: return "the Dickson monomials do not span H^0";
        default: return "unknown code";
    }
}

// ===========================================================================
// The cell
// ===========================================================================
DicksonThetaCell::DicksonThetaCell(uint32_t n, uint32_t ell, uint32_t num_random_samples)
    : n_(n), ell_(ell), m_(n + ell + 1) {
    hecke::RequireFieldCoefficientRing("DicksonThetaCell");
    // ell < 5 is a mathematical precondition ([E] sec:thetaobj (h), [C] lem:h2torsion /
    // rmk:thetasmallell), not a build capability: a std::logic_error (console exit 3).
    ENGINE_CHECK_MSG(ell >= 5,
                     "DicksonThetaCell requires ell >= 5 ([C] lem:h2torsion; rmk:thetasmallell): at ell in "
                     "{2, 3} the equivariant cochain functor is not exact on the model, so the long sequence "
                     "of 0 -> Sym^n -> Sym^{n+ell+1} -> Q_n -> 0 is not computed there");

    K_ = std::make_unique<SimplicialComplex>(CreateSL2ZFundamentalDomainTriangulation());
    model_ = std::make_unique<hecke::SL2ZTessellationTables>(*K_);
    const std::vector<SL2ZRelationDatum> dn = GenerateSL2ZRelationsWithGroupData(*K_, n_, ell_);
    const std::vector<SL2ZRelationDatum> dm = GenerateSL2ZRelationsWithGroupData(*K_, m_, ell_);
    const std::vector<SL2ZRelationDatum> dq = GenerateSL2ZQuotientLRelationsWithGroupData(*K_, n_, ell_);
    ENGINE_CHECK_MSG(dn.size() == dm.size() && dn.size() == dq.size(), "DicksonThetaCell: relation lists differ in length");
    for (size_t r = 0; r < dn.size(); ++r) {
        ENGINE_CHECK_MSG(dn[r].g_r == dm[r].g_r && dn[r].g_r == dq[r].g_r, "DicksonThetaCell: relation lists differ in g_r");
        rels_[0].push_back(dn[r].rel);
        rels_[1].push_back(dm[r].rel);
        rels_[2].push_back(dq[r].rel);
    }
    const uint32_t dims[3] = { n_ + 1, m_ + 1, ell_ + 1 };
    for (uint32_t t = 0; t < 3; ++t) {
        lssc_[t] = std::make_unique<LocalSystemSimplicialComplex>(*K_, rels_[t], dims[t], ell_);
        coh_[t] = std::make_unique<EquivariantCohomologyOnK>(lssc_[t].get());
        coh_[t]->compute_cohomology();
    }
    L_ = DicksonMultiplicationMatrix(m_, ell_);
    Pi_ = QuotientLProjectionMatrixFM(n_, ell_);
    for (uint32_t q = 0; q + 1 < 3; ++q) D_.push_back(K_->coboundary_operator(q, ell_));

    // Relation matrices of Q_n re-derived as rho_Q(g_r).
    for (size_t r = 0; r < dq.size() && cert_.code == 0; ++r) {
        const std::array<uint32_t, 4> g = dq[r].g_r.reduced_mod(ell_);
        if (!FlatEqual(dq[r].rel.M, ComputeQuotientLMatrixFM(Two(g[0], g[1], g[2], g[3], ell_), n_)))
            fail(7, dq[r].rel.tag);
    }
    if (cert_.code == 0) certify_fibre(num_random_samples);
    if (cert_.code == 0) certify_projectors();
    iota_.resize(3);
    pi_.resize(3);
    for (uint32_t q = 0; q < 3 && cert_.code == 0; ++q) {
        iota_[q] = induced_map(ThetaTerm::SymN, ThetaTerm::SymM, L_, q);
        if (cert_.code == 0) pi_[q] = induced_map(ThetaTerm::SymM, ThetaTerm::Quot, Pi_, q);
    }
    if (cert_.code == 0) compute_delta();
    if (cert_.code == 0) certify_sequence();
    if (cert_.code == 0) certify_h0();
}

void DicksonThetaCell::fail(uint32_t code, const std::string& detail) {
    if (cert_.code != 0) return;
    cert_.code = code;
    cert_.detail = detail;
}

FibreRepresentation DicksonThetaCell::representation(ThetaTerm t) const {
    return t == ThetaTerm::Quot ? FibreRepresentation::QuotientL : FibreRepresentation::SymN;
}

uint32_t DicksonThetaCell::fibre_dim(ThetaTerm t) const {
    return t == ThetaTerm::SymN ? n_ + 1 : (t == ThetaTerm::SymM ? m_ + 1 : ell_ + 1);
}

uint32_t DicksonThetaCell::h(ThetaTerm t, uint32_t q) const { return coh(t).get_H_eq_dimensions().at(q); }

const ThetaMat& DicksonThetaCell::iota(uint32_t q) const {
    ENGINE_CHECK_MSG(q < iota_.size(), "DicksonThetaCell::iota: degree out of range (or the certificate failed)");
    return iota_[q];
}
const ThetaMat& DicksonThetaCell::pi(uint32_t q) const {
    ENGINE_CHECK_MSG(q < pi_.size(), "DicksonThetaCell::pi: degree out of range (or the certificate failed)");
    return pi_[q];
}
const ThetaMat& DicksonThetaCell::delta(uint32_t q) const {
    ENGINE_CHECK_MSG(q < delta_.size(), "DicksonThetaCell::delta: degree out of range (or the certificate failed)");
    return delta_[q];
}

// ---------------------------------------------------------------------------
// Cochain helpers
// ---------------------------------------------------------------------------
bool DicksonThetaCell::is_zero_vec(const std::vector<uint32_t>& y) const {
    for (uint32_t v : y)
        if (v != 0) return false;
    return true;
}

std::vector<uint32_t> DicksonThetaCell::apply_fibre(const FlatMatrix& F, uint32_t q, const std::vector<uint32_t>& y) const {
    const uint32_t r = F.get_rows(), c = F.get_cols(), Nq = N(q);
    ENGINE_CHECK_MSG(y.size() == static_cast<size_t>(Nq) * c, "apply_fibre: cochain length != N_q * cols(F)");
    std::vector<uint32_t> out(static_cast<size_t>(Nq) * r, 0);
    for (uint32_t s = 0; s < Nq; ++s)
        for (uint32_t i = 0; i < r; ++i) {
            uint64_t acc = 0;
            for (uint32_t k = 0; k < c; ++k) {
                const uint32_t f = F.get_entry(i, k);
                if (f) acc = (acc + static_cast<uint64_t>(f) * y[static_cast<size_t>(s) * c + k]) % ell_;
            }
            out[static_cast<size_t>(s) * r + i] = static_cast<uint32_t>(acc);
        }
    return out;
}

std::vector<uint32_t> DicksonThetaCell::coboundary(uint32_t q, uint32_t d, const std::vector<uint32_t>& y) const {
    ENGINE_CHECK_MSG(q < D_.size(), "coboundary: degree out of range");
    const FlatMatrix& D = D_[q];
    const uint32_t Nt = D.get_rows(), Ns = D.get_cols();
    ENGINE_CHECK_MSG(y.size() == static_cast<size_t>(Ns) * d, "coboundary: cochain length != N_q * d");
    std::vector<uint32_t> out(static_cast<size_t>(Nt) * d, 0);
    for (uint32_t t = 0; t < Nt; ++t)
        for (uint32_t s = 0; s < Ns; ++s) {
            const uint32_t f = D.get_entry(t, s);
            if (f == 0) continue;
            for (uint32_t c = 0; c < d; ++c)
                out[static_cast<size_t>(t) * d + c] =
                    addm(out[static_cast<size_t>(t) * d + c], mulm(f, y[static_cast<size_t>(s) * d + c], ell_), ell_);
        }
    return out;
}

bool DicksonThetaCell::satisfies_relations(ThetaTerm t, uint32_t q, const std::vector<uint32_t>& y) const {
    const uint32_t d = fibre_dim(t);
    for (const auto& rel : rels_[static_cast<uint32_t>(t)]) {
        if (rel.q_dim != q) continue;
        const uint32_t sl = rel.s_lhs_idx_in_Kq, sr = rel.s_rhs_idx_in_Kq;
        for (uint32_t i = 0; i < d; ++i) {
            uint64_t acc = 0;
            for (uint32_t k = 0; k < d; ++k) {
                const uint32_t f = rel.M.get_entry(i, k);
                if (f) acc = (acc + static_cast<uint64_t>(f) * y[static_cast<size_t>(sr) * d + k]) % ell_;
            }
            if (y[static_cast<size_t>(sl) * d + i] != acc) return false;
        }
    }
    return true;
}

bool DicksonThetaCell::is_equivariant_cocycle(ThetaTerm t, uint32_t q, const std::vector<uint32_t>& y) const {
    ENGINE_CHECK_MSG(q < 3 && y.size() == static_cast<size_t>(N(q)) * fibre_dim(t),
                     "is_equivariant_cocycle: degree or cochain length out of range");
    if (!satisfies_relations(t, q, y)) return false;
    return q + 1 >= 3 || is_zero_vec(coboundary(q, fibre_dim(t), y));
}

std::vector<uint32_t> DicksonThetaCell::class_of(ThetaTerm t, uint32_t q, const std::vector<uint32_t>& y) const {
    const uint32_t hq = h(t, q);
    std::vector<uint32_t> out(hq, 0);
    if (hq == 0) return out;
    const FlatMatrix& P = coh(t).get_H_eq_projectors_from_C_eq_flat_fm().at(q);
    ENGINE_CHECK_MSG(P.get_rows() == hq && P.get_cols() == y.size(), "class_of: projector shape disagrees");
    for (uint32_t i = 0; i < hq; ++i) {
        uint64_t acc = 0;
        for (uint32_t k = 0; k < P.get_cols(); ++k) {
            const uint32_t f = P.get_entry(i, k);
            if (f && y[k]) acc = (acc + static_cast<uint64_t>(f) * y[k]) % ell_;
        }
        out[i] = static_cast<uint32_t>(acc);
    }
    return out;
}

std::vector<uint32_t> DicksonThetaCell::basis_cochain(ThetaTerm t, uint32_t q, uint32_t j) const {
    const FlatMatrix& B = coh(t).get_H_eq_bases_fm().at(q).at(j);
    std::vector<uint32_t> y(static_cast<size_t>(B.get_rows()) * B.get_cols());
    for (uint32_t s = 0; s < B.get_rows(); ++s)
        for (uint32_t c = 0; c < B.get_cols(); ++c) y[static_cast<size_t>(s) * B.get_cols() + c] = B.get_entry(s, c);
    return y;
}

// ---------------------------------------------------------------------------
// Certificates
// ---------------------------------------------------------------------------
void DicksonThetaCell::certify_fibre(uint32_t num_random_samples) {
    const DicksonMultiplicationCertificate dc = CertifyDicksonMultiplication(m_, ell_);
    cert_.dickson_cert_code = dc.verify();
    if (cert_.dickson_cert_code != 0) { fail(1, DicksonMultiplicationCertificate::verify_message(cert_.dickson_cert_code)); return; }

    // pi L. = 0, rank pi = ell + 1, pi E = I.
    {
        const FlatMatrix PL = Pi_ * L_;
        const std::vector<uint32_t> rows = QuotientLComplementRows(n_, ell_);
        bool ok = FlatIsZero(PL) && ThetaRank(FromFlat(Pi_)) == ell_ + 1;
        for (uint32_t s = 0; ok && s <= ell_; ++s)
            for (uint32_t t = 0; t <= ell_; ++t)
                if (Pi_.get_entry(s, rows[t]) != (s == t ? 1u : 0u)) ok = false;
        if (!ok) { fail(2, "pi / L. / E"); return; }
    }

    // The samples: the six fixed ones of the L. certificate and its random
    // ones, the reductions of U_ell's representatives [[1, j], [0, ell]] and
    // [[ell, 0], [0, 1]], and every relation element g_r.
    std::vector<uint32_t> samples = CertifyDicksonMultiplication(m_, ell_, num_random_samples).samples;
    for (uint32_t j = 0; j < ell_; ++j) {
        const uint32_t s[4] = { 1u, j, 0u, 0u };
        samples.insert(samples.end(), s, s + 4);
    }
    {
        const uint32_t s[4] = { 0u, 0u, 0u, 1u };
        samples.insert(samples.end(), s, s + 4);
    }
    {
        const std::vector<SL2ZRelationDatum> dn = GenerateSL2ZRelationsWithGroupData(*K_, 0, ell_);
        std::vector<std::array<uint32_t, 4>> seen;
        for (const auto& d : dn) {
            const std::array<uint32_t, 4> g = d.g_r.reduced_mod(ell_);
            if (std::find(seen.begin(), seen.end(), g) != seen.end()) continue;
            seen.push_back(g);
            samples.insert(samples.end(), g.begin(), g.end());
        }
    }
    cert_.num_samples = static_cast<uint32_t>(samples.size() / 4);

    std::vector<FlatMatrix> rhoQ;
    for (size_t s = 0; s + 3 < samples.size(); s += 4) {
        const FlatMatrix A = Two(samples[s], samples[s + 1], samples[s + 2], samples[s + 3], ell_);
        const FlatMatrix Rm = ComputeSymmetricPowerMatrixFM(A, m_);
        if (!FlatIsZero(Pi_ * (Rm * L_))) { fail(3, "sample " + std::to_string(s / 4)); return; }
        const FlatMatrix RQ = ComputeQuotientLMatrixFM(A, n_);
        if (!FlatEqual(RQ * Pi_, Pi_ * Rm)) { fail(4, "sample " + std::to_string(s / 4)); return; }
        rhoQ.push_back(RQ);
    }
    for (size_t s = 0; s + 7 < samples.size(); s += 4) {
        const FlatMatrix A = Two(samples[s], samples[s + 1], samples[s + 2], samples[s + 3], ell_);
        const FlatMatrix B = Two(samples[s + 4], samples[s + 5], samples[s + 6], samples[s + 7], ell_);
        if (!FlatEqual(ComputeQuotientLMatrixFM(A * B, n_), rhoQ[s / 4] * rhoQ[s / 4 + 1])) {
            fail(5, "pair " + std::to_string(s / 4));
            return;
        }
    }
    // The negative seam on the same samples.
    cert_.xelly_first_failure = FirstSampleNotPreserving(XEllYMultiplicationMatrix(n_, ell_), m_, ell_, samples);
    if (cert_.xelly_first_failure < 0) fail(6, "x^ell y . Sym^n");
}

void DicksonThetaCell::certify_projectors() {
    for (uint32_t t = 0; t < 3; ++t) {
        const ThetaTerm term = static_cast<ThetaTerm>(t);
        for (uint32_t q = 0; q < 3; ++q) {
            const uint32_t hq = h(term, q);
            if (hq == 0) continue;
            for (uint32_t j = 0; j < hq; ++j) {
                const std::vector<uint32_t> c = class_of(term, q, basis_cochain(term, q, j));
                for (uint32_t i = 0; i < hq; ++i)
                    if (c[i] != (i == j ? 1u : 0u)) { fail(8, "P H != I"); return; }
            }
            const auto& B = coh(term).get_B_eq_bases_fm().at(q);
            for (const FlatMatrix& b : B) {
                std::vector<uint32_t> y(static_cast<size_t>(b.get_rows()) * b.get_cols());
                for (uint32_t s = 0; s < b.get_rows(); ++s)
                    for (uint32_t cc = 0; cc < b.get_cols(); ++cc) y[static_cast<size_t>(s) * b.get_cols() + cc] = b.get_entry(s, cc);
                if (!is_zero_vec(class_of(term, q, y))) { fail(8, "P B != 0"); return; }
            }
        }
    }
}

ThetaMat DicksonThetaCell::induced_map(ThetaTerm src, ThetaTerm dst, const FlatMatrix& F, uint32_t q) {
    const uint32_t hs = h(src, q), hd = h(dst, q);
    ThetaMat M(hd, hs, ell_);
    for (uint32_t j = 0; j < hs; ++j) {
        const std::vector<uint32_t> y = apply_fibre(F, q, basis_cochain(src, q, j));
        ++cert_.relation_checks;
        if (!satisfies_relations(dst, q, y)) { fail(9, "relation"); return M; }
        if (q + 1 < 3) {
            ++cert_.cocycle_checks;
            if (!is_zero_vec(coboundary(q, fibre_dim(dst), y))) { fail(9, "cocycle"); return M; }
        }
        const std::vector<uint32_t> c = class_of(dst, q, y);
        for (uint32_t i = 0; i < hd; ++i) M.at(i, j) = c[i];
    }
    return M;
}

std::vector<uint32_t> DicksonThetaCell::connecting_cochain(uint32_t q, const std::vector<uint32_t>& z) const {
    ENGINE_CHECK_MSG(q < 2, "connecting_cochain: q must be 0 or 1");
    const uint32_t dm = m_ + 1, dq = ell_ + 1, dn = n_ + 1, Nq = N(q), Nq1 = N(q + 1);
    ENGINE_CHECK_MSG(z.size() == static_cast<size_t>(Nq) * dq, "connecting_cochain: z has the wrong length");
    // (1) An equivariant lift: y = C a with pi(C a) = z, C the C^q_eq basis of Sym^m.
    const auto& C = coh(ThetaTerm::SymM).get_C_eq_bases_fm().at(q);
    ThetaMat PiC(Nq * dq, static_cast<uint32_t>(C.size()), ell_);
    std::vector<std::vector<uint32_t>> cvecs(C.size());
    for (uint32_t k = 0; k < C.size(); ++k) {
        std::vector<uint32_t>& ck = cvecs[k];
        ck.resize(static_cast<size_t>(Nq) * dm);
        for (uint32_t s = 0; s < Nq; ++s)
            for (uint32_t c = 0; c < dm; ++c) ck[static_cast<size_t>(s) * dm + c] = C[k].get_entry(s, c);
        const std::vector<uint32_t> pk = apply_fibre(Pi_, q, ck);
        for (uint32_t r = 0; r < PiC.rows; ++r) PiC.at(r, k) = pk[r];
    }
    std::vector<uint32_t> a;
    if (!Solve(PiC, z, a)) throw std::runtime_error("DicksonThetaCell: z has no equivariant lift to Sym^m");
    std::vector<uint32_t> y(static_cast<size_t>(Nq) * dm, 0);
    for (uint32_t k = 0; k < a.size(); ++k) {
        if (a[k] == 0) continue;
        for (size_t i = 0; i < y.size(); ++i) y[i] = addm(y[i], mulm(a[k], cvecs[k][i], ell_), ell_);
    }
    if (!satisfies_relations(ThetaTerm::SymM, q, y) || apply_fibre(Pi_, q, y) != z)
        throw std::runtime_error("DicksonThetaCell: the lift is not an equivariant preimage of z");
    // (2) Its coboundary, then (3) division by L. simplex by simplex:
    //     row j + ell of L. x is -x_j + x_{j+ell-1} (the second term when j + ell - 1 <= n).
    const std::vector<uint32_t> w = coboundary(q, dm, y);
    std::vector<uint32_t> x(static_cast<size_t>(Nq1) * dn, 0);
    for (uint32_t s = 0; s < Nq1; ++s) {
        const uint32_t* ws = &w[static_cast<size_t>(s) * dm];
        uint32_t* xs = &x[static_cast<size_t>(s) * dn];
        for (uint32_t jj = dn; jj-- > 0;) {
            const uint32_t up = (jj + ell_ - 1 <= n_) ? xs[jj + ell_ - 1] : 0u;
            xs[jj] = subm(up, ws[jj + ell_], ell_);
        }
    }
    if (apply_fibre(L_, q + 1, x) != w) throw std::runtime_error("DicksonThetaCell: d(lift) is not divisible by L.");
    if (!satisfies_relations(ThetaTerm::SymN, q + 1, x))
        throw std::runtime_error("DicksonThetaCell: the connecting cochain violates a relation of Sym^n");
    if (q + 1 < 2 && !is_zero_vec(coboundary(q + 1, dn, x)))
        throw std::runtime_error("DicksonThetaCell: the connecting cochain is not a cocycle");
    return x;
}

void DicksonThetaCell::compute_delta() {
    delta_.resize(2);
    for (uint32_t q = 0; q < 2; ++q) {
        const uint32_t hq = h(ThetaTerm::Quot, q), ht = h(ThetaTerm::SymN, q + 1);
        delta_[q] = ThetaMat(ht, hq, ell_);
        for (uint32_t j = 0; j < hq; ++j) {
            std::vector<uint32_t> x;
            try {
                x = connecting_cochain(q, basis_cochain(ThetaTerm::Quot, q, j));
            } catch (const std::runtime_error& e) {
                fail(10, e.what());
                return;
            }
            ++cert_.lifts;
            ++cert_.divisions;
            const std::vector<uint32_t> c = class_of(ThetaTerm::SymN, q + 1, x);
            for (uint32_t i = 0; i < ht; ++i) delta_[q].at(i, j) = c[i];
        }
    }
}

void DicksonThetaCell::certify_sequence() {
    cert_.h2_zero = h(ThetaTerm::SymN, 2) == 0 && h(ThetaTerm::SymM, 2) == 0 && h(ThetaTerm::Quot, 2) == 0;
    const std::vector<uint32_t> dims = { h(ThetaTerm::SymN, 0), h(ThetaTerm::SymM, 0), h(ThetaTerm::Quot, 0),
                                         h(ThetaTerm::SymN, 1), h(ThetaTerm::SymM, 1), h(ThetaTerm::Quot, 1) };
    cert_.spots = CheckSequenceExactness(iota_[0], pi_[0], delta_[0], iota_[1], pi_[1], delta_[1], dims);
    cert_.euler_sum = static_cast<int64_t>(dims[0]) - dims[1] + dims[2] - dims[3] + dims[4] - dims[5];
    if (!cert_.h2_zero) { fail(11, "H^2"); return; }
    for (const ExactnessSpot& s : cert_.spots)
        if (!s.exact()) { fail(12, s.name); return; }
    if (cert_.euler_sum != 0) fail(13, "euler");
}

void DicksonThetaCell::certify_h0() {
    for (uint32_t which = 0; which < 2; ++which) {
        const ThetaTerm t = which == 0 ? ThetaTerm::SymN : ThetaTerm::SymM;
        const uint32_t deg = which == 0 ? n_ : m_;
        const auto ex = DicksonExponents(deg, ell_);
        const uint32_t h0 = h(t, 0), N0 = N(0);
        ThetaMat classes(h0, static_cast<uint32_t>(ex.size()), ell_);
        for (uint32_t k = 0; k < ex.size(); ++k) {
            const std::vector<uint32_t> f = DicksonMonomial(ex[k].first, ex[k].second, ell_);
            ENGINE_CHECK(f.size() == deg + 1);
            std::vector<uint32_t> y(static_cast<size_t>(N0) * (deg + 1));
            for (uint32_t s = 0; s < N0; ++s) std::copy(f.begin(), f.end(), y.begin() + static_cast<size_t>(s) * (deg + 1));
            ++cert_.relation_checks;
            ++cert_.cocycle_checks;
            if (!satisfies_relations(t, 0, y) || !is_zero_vec(coboundary(0, deg + 1, y))) {
                fail(14, "L^a D^b is not a 0-cocycle");
                return;
            }
            const std::vector<uint32_t> c = class_of(t, 0, y);
            for (uint32_t i = 0; i < h0; ++i) classes.at(i, k) = c[i];
        }
        const uint32_t rk = ThetaRank(classes);
        if (which == 0) { cert_.dickson_count_n = static_cast<uint32_t>(ex.size()); cert_.dickson_rank_n = rk; }
        else { cert_.dickson_count_m = static_cast<uint32_t>(ex.size()); cert_.dickson_rank_m = rk; }
        if (rk != ex.size() || rk != h0) { fail(14, which == 0 ? "Sym^n" : "Sym^m"); return; }
    }
}

// ---------------------------------------------------------------------------
// The Hecke triple (WP-6's entry point)
// ---------------------------------------------------------------------------
DicksonThetaCell::HeckeTriple DicksonThetaCell::hecke_triple(uint64_t p) const {
    HeckeTriple ht;
    ht.p = p;
    ht.cosets_n = std::make_unique<hecke::HeckeCosetTables>(p, 1, ell_, n_);
    ht.cosets_m = std::make_unique<hecke::HeckeCosetTables>(p, 1, ell_, m_);
    ht.tau_n = std::make_unique<hecke::HeckeTauTable>(*K_, *model_, *ht.cosets_n, ell_, n_);
    ht.tau_m = std::make_unique<hecke::HeckeTauTable>(*K_, *model_, *ht.cosets_m, ell_, m_);
    ht.op_n = std::make_unique<hecke::HeckeOperatorOnK>(coh_[0].get(), ht.cosets_n.get(), ht.tau_n.get());
    ht.op_m = std::make_unique<hecke::HeckeOperatorOnK>(coh_[1].get(), ht.cosets_m.get(), ht.tau_m.get());
    ht.op_q = std::make_unique<hecke::HeckeOperatorOnK>(coh_[2].get(), ht.cosets_n.get(), ht.tau_n.get(),
                                                        FibreRepresentation::QuotientL);
    ht.op_n->compute();
    ht.op_m->compute();
    ht.op_q->compute();
    return ht;
}

std::string DicksonThetaCell::summary() const {
    std::ostringstream os;
    os << "(n=" << n_ << ", ell=" << ell_ << ", m=" << m_ << ") h0=" << h(ThetaTerm::SymN, 0) << "/"
       << h(ThetaTerm::SymM, 0) << "/" << h(ThetaTerm::Quot, 0) << " h1=" << h(ThetaTerm::SymN, 1) << "/"
       << h(ThetaTerm::SymM, 1) << "/" << h(ThetaTerm::Quot, 1);
    if (delta_.size() == 2 && iota_.size() == 3)
        os << " rk(delta0)=" << ThetaRank(delta_[0]) << " rk(theta1)=" << ThetaRank(iota_[1]);
    os << " cert=" << cert_.code;
    if (cert_.code) os << " (" << DicksonThetaCertificate::verify_message(cert_.code) << ": " << cert_.detail << ")";
    return os.str();
}

} // namespace hecke_ext
