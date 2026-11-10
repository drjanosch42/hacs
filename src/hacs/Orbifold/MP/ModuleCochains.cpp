// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-3) MP cochains with an arbitrary coefficient module -- see ModuleCochains.h.
#include "ModuleCochains.h"

#include <algorithm>
#include <utility>

#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

using integral_model::key_of;
using integral_model::mat_of;

ModuleRep::ModuleRep(std::string name, uint32_t dim, uint32_t ell, uint32_t e, Fn f)
    : name_(std::move(name)), dim_(dim), ell_(ell), e_(e), m_(PowU(ell, e)), f_(std::move(f)),
      memo_(std::make_shared<std::map<GKey, Vec>>()) {
    ENGINE_CHECK_MSG(m_ >= 2 && m_ <= 255, "orbifold_mp::ModuleRep: 2 <= ell^e <= 255");
    ENGINE_CHECK_MSG(dim_ >= 1, "orbifold_mp::ModuleRep: dim >= 1");
}

const Vec& ModuleRep::rho(const Mat22& g) const {
    const GKey k = key_of(g);
    auto it = memo_->find(k);
    if (it == memo_->end()) {
        Vec R = f_(g);
        ENGINE_CHECK_MSG(R.size() == static_cast<size_t>(dim_) * dim_, "orbifold_mp::ModuleRep: rho(g) has the wrong shape");
        for (uint32_t& x : R) x %= m_;
        it = memo_->emplace(k, std::move(R)).first;
    }
    return it->second;
}

ModuleRep SymRep(uint32_t n, uint32_t ell, uint32_t e) {
    const uint32_t d = n + 1;
    const engine::ModCtx<uint32_t> ctx = RingCtx(ell, e);
    return ModuleRep("Sym^" + std::to_string(n), d, ell, e, [ctx, n, d](const Mat22& g) {
        const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx, g, n);
        Vec out(static_cast<size_t>(d) * d);
        for (uint32_t a = 0; a < d; ++a)
            for (uint32_t b = 0; b < d; ++b) out[a * d + b] = R.at(a, b);
        return out;
    });
}

namespace {

// Append one CSR row from (col, value) pairs (unsorted, possibly repeated), reduced mod m.
void PushRow(SparseRows& A, std::vector<std::pair<uint32_t, uint32_t>>& row, uint32_t m) {
    std::sort(row.begin(), row.end());
    uint32_t lastc = UINT32_MAX, acc = 0;
    for (size_t t = 0; t <= row.size(); ++t) {
        if (t < row.size() && row[t].first == lastc) {
            acc = (acc + row[t].second) % m;
            continue;
        }
        if (lastc != UINT32_MAX && acc != 0) {
            A.col.push_back(lastc);
            A.val.push_back(static_cast<uint8_t>(acc));
        }
        if (t < row.size()) {
            lastc = row[t].first;
            acc = row[t].second % m;
        }
    }
    A.start.push_back(A.col.size());
}

}  // namespace

SparseRows CoboundaryRep(const InternedMPSet& S, uint32_t k, const ModuleRep& N) {
    ENGINE_CHECK_MSG(k + 1 <= S.k_max(), "orbifold_mp::CoboundaryRep: S_{k+1} not built");
    const uint32_t d = N.dim(), m = N.m();
    const auto& E = S.elements();
    std::vector<const Vec*> rho(E.size(), nullptr);
    for (uint64_t i : S.nd(k + 1)) {
        const uint32_t g = S.gamma(k + 1, i)[0];
        if (!rho[g]) rho[g] = &N.rho(E[g]);
    }
    SparseRows A;
    ENGINE_CHECK_MSG(S.count_nondegenerate(k) * d < (uint64_t(1) << 32), "orbifold_mp::CoboundaryRep: too many columns");
    A.ncols = static_cast<uint32_t>(S.count_nondegenerate(k) * d);
    const auto& nd = S.nd(k + 1);
    A.start.reserve(nd.size() * d + 1);
    std::vector<std::pair<uint32_t, uint32_t>> row;
    std::vector<uint64_t> fidx(k + 2);
    for (uint64_t r = 0; r < nd.size(); ++r) {
        const uint64_t idx = nd[r];
        for (uint32_t i = 0; i <= k + 1; ++i) fidx[i] = S.nd_index(k, S.face(k + 1, idx, i));
        const Vec& R = *rho[S.gamma(k + 1, idx)[0]];
        for (uint32_t a = 0; a < d; ++a) {
            row.clear();
            for (uint32_t i = 0; i <= k + 1; ++i) {
                if (fidx[i] == InternedMPSet::kNoIndex) continue;   // f := 0 on degenerate faces
                const uint64_t base = fidx[i] * d;
                if (i == 0) {
                    for (uint32_t b = 0; b < d; ++b) {
                        const uint32_t v = R[a * d + b];
                        if (v) row.emplace_back(static_cast<uint32_t>(base + b), v);
                    }
                } else {
                    row.emplace_back(static_cast<uint32_t>(base + a), (i % 2 == 0) ? 1u : m - 1u);
                }
            }
            PushRow(A, row, m);
        }
    }
    return A;
}

SparseRows LabelledPullbackRep(const InternedMPSet& Sf, const InternedMPSet& Sc, const RefinementLabels& L, const StringMap& U,
                               uint32_t q, uint32_t d_out, uint32_t d_in, uint32_t m, const LabelMatrix& Lmat,
                               bool drop_factor) {
    ENGINE_CHECK_MSG(!drop_factor || d_out == d_in, "orbifold_mp::LabelledPullbackRep: dropping the factor needs d_out = d_in");
    std::vector<Vec> lab(L.h.size());
    for (uint32_t t = 0; t < L.h.size(); ++t) {
        if (drop_factor) {
            lab[t].assign(static_cast<size_t>(d_out) * d_in, 0);
            for (uint32_t a = 0; a < d_out; ++a) lab[t][a * d_in + a] = 1 % m;
        } else {
            lab[t] = Lmat(L.h[t]);
            ENGINE_CHECK_MSG(lab[t].size() == static_cast<size_t>(d_out) * d_in, "orbifold_mp::LabelledPullbackRep: label matrix shape");
            for (uint32_t& x : lab[t]) x %= m;
        }
    }
    SparseRows A;
    ENGINE_CHECK_MSG(Sc.count_nondegenerate(q) * d_in < (uint64_t(1) << 32), "orbifold_mp::LabelledPullbackRep: too many columns");
    A.ncols = static_cast<uint32_t>(Sc.count_nondegenerate(q) * d_in);
    for (uint64_t i : Sf.nd(q)) {
        const uint64_t img = U.image.at(q).at(i);
        ENGINE_CHECK_MSG(img != InternedMPSet::kNoIndex, "orbifold_mp::LabelledPullbackRep: a string leaves the target MP set");
        const uint64_t c = Sc.nd_index(q, img);
        const Vec& Lt = lab[Sf.sigma(q, i)[0]];
        for (uint32_t a = 0; a < d_out; ++a) {
            if (c != InternedMPSet::kNoIndex)
                for (uint32_t b = 0; b < d_in; ++b) {
                    const uint32_t v = Lt[a * d_in + b];
                    if (v) {
                        A.col.push_back(static_cast<uint32_t>(c * d_in + b));
                        A.val.push_back(static_cast<uint8_t>(v));
                    }
                }
            A.start.push_back(A.col.size());
        }
    }
    return A;
}

SparseRows CoefficientMap(const InternedMPSet& S, uint32_t q, const Vec& phi, uint32_t d_out, uint32_t d_in, uint32_t m) {
    ENGINE_CHECK_MSG(phi.size() == static_cast<size_t>(d_out) * d_in, "orbifold_mp::CoefficientMap: shape");
    SparseRows A;
    const uint64_t N = S.count_nondegenerate(q);
    ENGINE_CHECK_MSG(N * d_in < (uint64_t(1) << 32), "orbifold_mp::CoefficientMap: too many columns");
    A.ncols = static_cast<uint32_t>(N * d_in);
    for (uint64_t j = 0; j < N; ++j)
        for (uint32_t a = 0; a < d_out; ++a) {
            for (uint32_t b = 0; b < d_in; ++b) {
                const uint32_t v = phi[a * d_in + b] % m;
                if (v) {
                    A.col.push_back(static_cast<uint32_t>(j * d_in + b));
                    A.val.push_back(static_cast<uint8_t>(v));
                }
            }
            A.start.push_back(A.col.size());
        }
    return A;
}

SparseRows PhiPullbackRep(const PhiLift& Phi, const InternedMPSet& S, uint32_t q, uint32_t w_q, const ModuleRep& N) {
    ENGINE_CHECK_MSG(q <= Phi.k_top(), "orbifold_mp::PhiPullbackRep: Phi_q not built");
    const uint32_t d = N.dim(), m = N.m();
    SparseRows A;
    A.ncols = w_q * d;
    std::vector<uint32_t> acc(static_cast<size_t>(w_q) * d);
    const uint64_t cnt = S.count_nondegenerate(q);
    for (uint64_t j = 0; j < cnt; ++j) {
        const integral_model::Chain c = Phi.value(q, j);
        ENGINE_CHECK_MSG(c.size() == w_q, "orbifold_mp::PhiPullbackRep: chain length != w_q");
        for (uint32_t a = 0; a < d; ++a) {
            std::fill(acc.begin(), acc.end(), 0u);
            for (uint32_t cell = 0; cell < w_q; ++cell)
                for (const auto& [g, coef] : c[cell].terms) {
                    int64_t cm = coef % static_cast<int64_t>(m);
                    if (cm < 0) cm += m;
                    if (cm == 0) continue;
                    const Vec& R = N.rho(mat_of(g));
                    for (uint32_t b = 0; b < d; ++b) {
                        uint32_t& v = acc[static_cast<size_t>(cell) * d + b];
                        v = static_cast<uint32_t>((v + static_cast<uint64_t>(cm) * R[a * d + b]) % m);
                    }
                }
            for (uint32_t col = 0; col < acc.size(); ++col)
                if (acc[col]) {
                    A.col.push_back(col);
                    A.val.push_back(static_cast<uint8_t>(acc[col]));
                }
            A.start.push_back(A.col.size());
        }
    }
    return A;
}

namespace {

std::pair<std::vector<int64_t>, uint32_t> LocalIndex(const integral_model::IsotropyModel& CI, uint32_t deg, bool boundary_only) {
    std::vector<int64_t> idx(CI.cells(deg).size(), -1);
    int64_t k = 0;
    for (uint32_t i = 0; i < CI.cells(deg).size(); ++i)
        if (!boundary_only || CI.cells(deg)[i].boundary) idx[i] = k++;
    return {idx, static_cast<uint32_t>(k)};
}

}  // namespace

engine::DenseBuf<uint32_t> CICoboundaryRep(const integral_model::IsotropyModel& CI, uint32_t q, const ModuleRep& N,
                                           bool boundary_only) {
    ENGINE_CHECK_MSG(q <= CI.q_max(), "orbifold_mp::CICoboundaryRep: q > q_max");
    const uint32_t D = N.dim(), m = N.m();
    const auto [src, ncols] = LocalIndex(CI, q, boundary_only);
    const auto [dst, nrows] = LocalIndex(CI, q + 1, boundary_only);
    engine::DenseBuf<uint32_t> M(nrows * D, ncols * D);
    for (const auto& e : CI.differential(q + 1)) {
        // (delta^q f)(e_col) = sum rho(coeff) f(e_row): block (col, row) -- IsotropyModel::coboundary's layout.
        const int64_t bi = dst[e.col], bj = src[e.row];
        if (bi < 0 || bj < 0) {
            ENGINE_CHECK_MSG(bi < 0, "orbifold_mp::CICoboundaryRep: a boundary cochain differential leaves R_d");
            continue;
        }
        for (const auto& [g, nmul] : e.coeff.terms) {
            int64_t r = nmul % static_cast<int64_t>(m);
            if (r < 0) r += m;
            if (r == 0) continue;
            const Vec& R = N.rho(mat_of(g));
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) {
                    uint32_t& v = M.at(static_cast<uint32_t>(bi) * D + i, static_cast<uint32_t>(bj) * D + j);
                    v = static_cast<uint32_t>((v + static_cast<uint64_t>(r) * R[i * D + j]) % m);
                }
        }
    }
    return M;
}

engine::DenseBuf<uint32_t> CIRestrictToBoundary(const integral_model::IsotropyModel& CI, uint32_t q, uint32_t dim) {
    const auto [loc, nb] = LocalIndex(CI, q, true);
    engine::DenseBuf<uint32_t> M(nb * dim, static_cast<uint32_t>(CI.cells(q).size()) * dim);
    for (uint32_t c = 0; c < loc.size(); ++c)
        if (loc[c] >= 0)
            for (uint32_t a = 0; a < dim; ++a) M.at(static_cast<uint32_t>(loc[c]) * dim + a, c * dim + a) = 1;
    return M;
}

engine::DenseBuf<uint32_t> CICoefficientMap(const integral_model::IsotropyModel& CI, uint32_t q, const Vec& phi,
                                            uint32_t d_out, uint32_t d_in, uint32_t m) {
    const uint32_t w = static_cast<uint32_t>(CI.cells(q).size());
    engine::DenseBuf<uint32_t> M(w * d_out, w * d_in);
    for (uint32_t c = 0; c < w; ++c)
        for (uint32_t a = 0; a < d_out; ++a)
            for (uint32_t b = 0; b < d_in; ++b) M.at(c * d_out + a, c * d_in + b) = phi[a * d_in + b] % m;
    return M;
}

SparseRows ConeDifferential(const SparseRows& dC, const SparseRows& phi, const SparseRows* dDm1, uint32_t dimDm1, uint32_t m) {
    const uint32_t dimCq = dC.ncols;
    ENGINE_CHECK_MSG(phi.ncols == dimCq, "orbifold_mp::ConeDifferential: phi^q has the wrong source");
    if (dDm1) ENGINE_CHECK_MSG(dDm1->rows() == phi.rows() && dDm1->ncols == dimDm1, "orbifold_mp::ConeDifferential: d^{q-1}_D shape");
    else ENGINE_CHECK_MSG(dimDm1 == 0, "orbifold_mp::ConeDifferential: D^{-1} = 0");
    SparseRows A;
    A.ncols = dimCq + dimDm1;
    for (uint64_t r = 0; r < dC.rows(); ++r) {
        for (uint64_t t = dC.start[r]; t < dC.start[r + 1]; ++t) {
            A.col.push_back(dC.col[t]);
            A.val.push_back(dC.val[t]);
        }
        A.start.push_back(A.col.size());
    }
    for (uint64_t r = 0; r < phi.rows(); ++r) {
        for (uint64_t t = phi.start[r]; t < phi.start[r + 1]; ++t) {
            A.col.push_back(phi.col[t]);
            A.val.push_back(phi.val[t]);
        }
        if (dDm1)
            for (uint64_t t = dDm1->start[r]; t < dDm1->start[r + 1]; ++t) {
                A.col.push_back(dimCq + dDm1->col[t]);
                A.val.push_back(static_cast<uint8_t>((m - dDm1->val[t]) % m));
            }
        A.start.push_back(A.col.size());
    }
    return A;
}

engine::DenseBuf<uint32_t> DenseCone(const engine::DenseBuf<uint32_t>& dC, const engine::DenseBuf<uint32_t>& phi,
                                     const engine::DenseBuf<uint32_t>* dDm1, uint32_t dimDm1, uint32_t m) {
    ENGINE_CHECK_MSG(phi.cols == dC.cols, "orbifold_mp::DenseCone: phi^q has the wrong source");
    engine::DenseBuf<uint32_t> M(dC.rows + phi.rows, dC.cols + dimDm1);
    for (uint32_t i = 0; i < dC.rows; ++i)
        for (uint32_t j = 0; j < dC.cols; ++j) M.at(i, j) = dC.at(i, j) % m;
    for (uint32_t i = 0; i < phi.rows; ++i) {
        for (uint32_t j = 0; j < phi.cols; ++j) M.at(dC.rows + i, j) = phi.at(i, j) % m;
        if (dDm1)
            for (uint32_t j = 0; j < dimDm1; ++j) M.at(dC.rows + i, dC.cols + j) = (m - dDm1->at(i, j) % m) % m;
    }
    return M;
}

std::vector<int64_t> LengthsOf(const std::vector<SparseRows>& d, const std::vector<uint64_t>& dimC, uint32_t ell, uint32_t e) {
    ENGINE_CHECK_MSG(dimC.size() >= d.size(), "orbifold_mp::LengthsOf: dim C^q missing");
    std::vector<uint64_t> li;
    for (const auto& x : d) li.push_back(x.rows() && x.ncols ? RowSpanLength(x, ell, e).length : 0);
    std::vector<int64_t> H;
    for (uint32_t q = 0; q < d.size(); ++q)
        H.push_back(static_cast<int64_t>(e * dimC[q]) - static_cast<int64_t>(li[q]) - (q ? static_cast<int64_t>(li[q - 1]) : 0));
    return H;
}

RepSide::RepSide(const InternedMPSet& S, const ModuleRep& N, uint32_t q_top, bool lengths)
    : ell(N.ell()), e(N.e()), m(N.m()), dim(N.dim()), have_lengths(lengths) {
    ENGINE_CHECK_MSG(q_top + 1 <= S.k_max(), "orbifold_mp::RepSide: S_{q_top+1} not built");
    for (uint32_t q = 0; q <= q_top + 1; ++q) dimC.push_back(S.count_nondegenerate(q) * dim);
    for (uint32_t q = 0; q <= q_top; ++q) {
        d.push_back(CoboundaryRep(S, q, N));
        len_im.push_back(lengths && d.back().rows() && d.back().ncols ? RowSpanLength(d.back(), ell, e).length : 0);
    }
}

int64_t RepSide::lenH(uint32_t q) const {
    ENGINE_CHECK_MSG(have_lengths, "orbifold_mp::RepSide::lenH: built without lengths");
    return static_cast<int64_t>(e * dimC.at(q)) - static_cast<int64_t>(len_im.at(q)) - (q ? static_cast<int64_t>(len_im.at(q - 1)) : 0);
}

ImageInH RepSide::image(uint32_t q, const std::vector<Vec>& mapped, bool cocycle_check) const {
    ENGINE_CHECK_MSG(dimC.at(q) < (uint64_t(1) << 32), "orbifold_mp::RepSide::image: C^q too large");
    return ImageOfCocycles(mapped, static_cast<uint32_t>(dimC[q]), q ? &d.at(q - 1) : nullptr, cocycle_check ? &d.at(q) : nullptr, ell, e);
}

CIRepSide::CIRepSide(const integral_model::IsotropyModel& CI, const ModuleRep& N, uint32_t q_top, bool boundary_only)
    : ell(N.ell()), e(N.e()), m(N.m()) {
    const engine::ModCtx<uint32_t> ctx = N.ctx();
    for (uint32_t q = 0; q <= q_top; ++q) d.push_back(CICoboundaryRep(CI, q, N, boundary_only));
    for (uint32_t q = 0; q <= q_top; ++q) {
        dimC.push_back(d[q].cols);
        src.push_back(Source(ctx, d[q], q ? &d[q - 1] : nullptr, nullptr));
    }
}

ImageInH CIRepSide::image(uint32_t q, const std::vector<Vec>& mapped) const {
    const SparseRows dq = FromDense(d.at(q), m);
    if (q == 0) return ImageOfCocycles(mapped, dimC[0], nullptr, &dq, ell, e);
    const SparseRows dp = FromDense(d.at(q - 1), m);
    return ImageOfCocycles(mapped, dimC[q], &dp, &dq, ell, e);
}

std::vector<int64_t> DenseLengthsOf(const engine::ModCtx<uint32_t>& ctx, const std::vector<engine::DenseBuf<uint32_t>>& d,
                                    const std::vector<uint32_t>& dimC) {
    std::vector<uint64_t> li;
    for (const auto& x : d) li.push_back(x.rows && x.cols ? DenseLength(ctx, x) : 0);
    std::vector<int64_t> H;
    for (uint32_t q = 0; q < d.size(); ++q)
        H.push_back(static_cast<int64_t>(ctx.k * dimC.at(q)) - static_cast<int64_t>(li[q]) - (q ? static_cast<int64_t>(li[q - 1]) : 0));
    return H;
}

bool CommutesExactly(const SparseRows& phiNext, const SparseRows& dA, const SparseRows& dB, const SparseRows& phi, uint32_t m) {
    return SpEqual(SpMul(phiNext, dA, m), SpMul(dB, phi, m));
}

}  // namespace orbifold_mp
