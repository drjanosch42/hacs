// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-2) Shared harness pieces of the MP-2 validations -- see MP2Common.h.
#include "MP2Common.h"

#include <sstream>

#include "../../../ModArithmetic/EngineCheck.h"

namespace orbifold_mp {

const MvCheckRow& FrozenRow(uint32_t ell, uint32_t n) {
    for (const auto& r : MvCheckOracle())
        if (r.ell == ell && r.n == n) return r;
    ENGINE_CHECK_MSG(false, "orbifold_mp::FrozenRow: no mv_check.out row for this (ell, n)");
    return MvCheckOracle().front();
}

SourceH Source(const engine::ModCtx<uint32_t>& ctx, const engine::DenseBuf<uint32_t>& delta,
               const engine::DenseBuf<uint32_t>* prev, const engine::DenseBuf<uint32_t>* proj) {
    SourceH S;
    const uint32_t n = delta.cols;
    engine::DenseBuf<uint32_t> A(delta.rows + (proj ? n : 0), n);
    for (uint32_t i = 0; i < delta.rows; ++i)
        for (uint32_t j = 0; j < n; ++j) A.at(i, j) = delta.at(i, j);
    if (proj) {
        ENGINE_CHECK_MSG(proj->rows == n && proj->cols == n, "orbifold_mp::Source: projector shape");
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t pij = proj->at(i, j) % ctx.p;
                const uint32_t id = (i == j) ? 1u : 0u;
                A.at(delta.rows + i, j) = static_cast<uint32_t>((id + ctx.p - pij) % ctx.p);
            }
    }
    S.cocycles = KernelGenerators(ctx, A);
    S.len_Z = ctx.is_field() ? S.cocycles.size() : SpanLength(S.cocycles, n, static_cast<uint32_t>(ctx.ell), ctx.k);
    if (prev) {
        ENGINE_CHECK_MSG(prev->rows == n, "orbifold_mp::Source: delta^{q-1} has the wrong row count");
        S.len_B = DenseLength(ctx, *prev);
    }
    ENGINE_CHECK_MSG(S.len_B <= S.len_Z, "orbifold_mp::Source: B^q longer than Z^q (not a complex?)");
    return S;
}

MPSide::MPSide(const InternedMPSet& S, uint32_t n_, uint32_t ell_, uint32_t k_, uint32_t q_top)
    : n(n_), ell(ell_), k(k_), m(PowU(ell_, k_)) {
    ENGINE_CHECK_MSG(q_top + 1 <= S.k_max(), "orbifold_mp::MPSide: S_{q_top+1} not built");
    for (uint32_t q = 0; q <= q_top + 1; ++q) dimC.push_back(S.count_nondegenerate(q) * (n + 1));
    for (uint32_t q = 0; q <= q_top; ++q) {
        d.push_back(Coboundary(S, q, n, m));
        stats.push_back(RowSpanLength(d.back(), ell, k));
        len_im.push_back(stats.back().length);
    }
}

int64_t MPSide::lenH(uint32_t q) const {
    return static_cast<int64_t>(k * dimC.at(q)) - static_cast<int64_t>(len_im.at(q)) -
           (q ? static_cast<int64_t>(len_im.at(q - 1)) : 0);
}

ImageInH MPSide::image(uint32_t q, const std::vector<Vec>& mapped, bool cocycle_check) const {
    ENGINE_CHECK_MSG(dimC.at(q) < (uint64_t(1) << 32), "orbifold_mp::MPSide::image: C^q too large");
    return ImageOfCocycles(mapped, static_cast<uint32_t>(dimC[q]), q ? &d.at(q - 1) : nullptr,
                           cocycle_check ? &d.at(q) : nullptr, ell, k);
}

std::vector<Vec> MapAll(const SparseRows& phi, const std::vector<Vec>& v, uint32_t m) {
    std::vector<Vec> out;
    out.reserve(v.size());
    for (const Vec& x : v) out.push_back(SpMatVec(phi, x, m));
    return out;
}

std::vector<Vec> MapAllDense(const engine::DenseBuf<uint32_t>& phi, const std::vector<Vec>& v, uint32_t m) {
    std::vector<Vec> out;
    out.reserve(v.size());
    for (const Vec& x : v) out.push_back(DenseMatVec(phi, x, m));
    return out;
}

std::string JoinV(const std::vector<int64_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

}  // namespace orbifold_mp
