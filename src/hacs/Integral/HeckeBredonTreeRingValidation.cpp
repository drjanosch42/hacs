// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "HeckeBredonTreeRingValidation.h"

// hecke-extensions slot 9, WP-3 -- see the header.

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/ExactInt.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/ModCtx.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::DenseBuf;
using engine::ModCtx;

// ---------------------------------------------------------------------------
// O7, FROZEN.  Provenance:
//   generator  docs/implementation/integral_line_probe.py, function model_R
//              (slot-9 WP-3; the functions of a500d67 byte-unchanged)
//   blob       f50f3cd3ccfeb7580e85861951f94e19df0a2b4f  (git hash-object of the probe)
//   command    python3 docs/implementation/integral_line_probe.py --model-R <ell> <k> <n,...>
//              ell in {2, 3}: k = 1 and k in {2, 3, 4}, n = 0,2,...,20,1,3,5,7,9;
//              ell = 5: k = 2, n = 6,10,16,30; k = 3, n = 30
//   frozen     2026-09-24
// Row: {ell, k, n, {length_H^0, length_H^1, length_H^2}, {inv_H^0, inv_H^1, inv_H^2}},
// invariant factors as exponents of ell, descending.
// ---------------------------------------------------------------------------
struct O7Cell {
    uint32_t ell, k, n;
    std::array<uint32_t, 3> len;
    std::vector<std::vector<uint32_t>> inv;   // three entries, H^0 .. H^2
};
const char* const kO7Blob = "f50f3cd3ccfeb7580e85861951f94e19df0a2b4f";

const std::vector<O7Cell>& O7Table() {
    static const std::vector<O7Cell> t = {
        {2, 1,  0, {1, 0, 0}, {{1}, {}, {}}},
        {2, 1,  2, {1, 1, 0}, {{1}, {1}, {}}},
        {2, 1,  4, {1, 2, 0}, {{1}, {1,1}, {}}},
        {2, 1,  6, {2, 2, 0}, {{1,1}, {1,1}, {}}},
        {2, 1,  8, {2, 3, 0}, {{1,1}, {1,1,1}, {}}},
        {2, 1, 10, {2, 4, 0}, {{1,1}, {1,1,1,1}, {}}},
        {2, 1, 12, {3, 4, 0}, {{1,1,1}, {1,1,1,1}, {}}},
        {2, 1, 14, {3, 5, 0}, {{1,1,1}, {1,1,1,1,1}, {}}},
        {2, 1, 16, {3, 6, 0}, {{1,1,1}, {1,1,1,1,1,1}, {}}},
        {2, 1, 18, {4, 6, 0}, {{1,1,1,1}, {1,1,1,1,1,1}, {}}},
        {2, 1, 20, {4, 7, 0}, {{1,1,1,1}, {1,1,1,1,1,1,1}, {}}},
        {2, 1,  1, {0, 1, 0}, {{}, {1}, {}}},
        {2, 1,  3, {1, 1, 0}, {{1}, {1}, {}}},
        {2, 1,  5, {1, 2, 0}, {{1}, {1,1}, {}}},
        {2, 1,  7, {1, 3, 0}, {{1}, {1,1,1}, {}}},
        {2, 1,  9, {2, 3, 0}, {{1,1}, {1,1,1}, {}}},
        {3, 1,  0, {1, 0, 0}, {{1}, {}, {}}},
        {3, 1,  2, {0, 1, 0}, {{}, {1}, {}}},
        {3, 1,  4, {1, 1, 0}, {{1}, {1}, {}}},
        {3, 1,  6, {1, 2, 0}, {{1}, {1,1}, {}}},
        {3, 1,  8, {1, 2, 0}, {{1}, {1,1}, {}}},
        {3, 1, 10, {1, 3, 0}, {{1}, {1,1,1}, {}}},
        {3, 1, 12, {2, 3, 0}, {{1,1}, {1,1,1}, {}}},
        {3, 1, 14, {1, 4, 0}, {{1}, {1,1,1,1}, {}}},
        {3, 1, 16, {2, 4, 0}, {{1,1}, {1,1,1,1}, {}}},
        {3, 1, 18, {2, 5, 0}, {{1,1}, {1,1,1,1,1}, {}}},
        {3, 1, 20, {2, 5, 0}, {{1,1}, {1,1,1,1,1}, {}}},
        {3, 1,  1, {0, 0, 0}, {{}, {}, {}}},
        {3, 1,  3, {0, 0, 0}, {{}, {}, {}}},
        {3, 1,  5, {0, 0, 0}, {{}, {}, {}}},
        {3, 1,  7, {0, 0, 0}, {{}, {}, {}}},
        {3, 1,  9, {0, 0, 0}, {{}, {}, {}}},
        {2, 2,  0, {2, 0, 0}, {{2}, {}, {}}},
        {2, 2,  2, {1, 2, 0}, {{1}, {2}, {}}},
        {2, 2,  4, {2, 4, 0}, {{2}, {2,2}, {}}},
        {2, 2,  6, {3, 4, 0}, {{2,1}, {2,2}, {}}},
        {2, 2,  8, {3, 5, 0}, {{2,1}, {2,2,1}, {}}},
        {2, 2, 10, {3, 8, 0}, {{2,1}, {2,2,2,2}, {}}},
        {2, 2, 12, {5, 7, 0}, {{2,2,1}, {2,2,2,1}, {}}},
        {2, 2, 14, {4, 9, 0}, {{2,1,1}, {2,2,2,2,1}, {}}},
        {2, 2, 16, {5, 11, 0}, {{2,2,1}, {2,2,2,2,2,1}, {}}},
        {2, 2, 18, {6, 11, 0}, {{2,2,1,1}, {2,2,2,2,2,1}, {}}},
        {2, 2, 20, {6, 12, 0}, {{2,2,1,1}, {2,2,2,2,2,1,1}, {}}},
        {2, 2,  1, {0, 1, 0}, {{}, {1}, {}}},
        {2, 2,  3, {1, 1, 0}, {{1}, {1}, {}}},
        {2, 2,  5, {1, 2, 0}, {{1}, {1,1}, {}}},
        {2, 2,  7, {1, 3, 0}, {{1}, {1,1,1}, {}}},
        {2, 2,  9, {2, 3, 0}, {{1,1}, {1,1,1}, {}}},
        {2, 3,  0, {3, 0, 0}, {{3}, {}, {}}},
        {2, 3,  2, {1, 3, 0}, {{1}, {3}, {}}},
        {2, 3,  4, {2, 5, 0}, {{2}, {3,2}, {}}},
        {2, 3,  6, {3, 5, 0}, {{2,1}, {3,2}, {}}},
        {2, 3,  8, {4, 7, 0}, {{3,1}, {3,3,1}, {}}},
        {2, 3, 10, {3, 11, 0}, {{2,1}, {3,3,3,2}, {}}},
        {2, 3, 12, {6, 9, 0}, {{3,2,1}, {3,3,2,1}, {}}},
        {2, 3, 14, {4, 12, 0}, {{2,1,1}, {3,3,3,2,1}, {}}},
        {2, 3, 16, {6, 15, 0}, {{3,2,1}, {3,3,3,3,2,1}, {}}},
        {2, 3, 18, {6, 14, 0}, {{2,2,1,1}, {3,3,3,2,2,1}, {}}},
        {2, 3, 20, {7, 16, 0}, {{3,2,1,1}, {3,3,3,3,2,1,1}, {}}},
        {2, 3,  1, {0, 1, 0}, {{}, {1}, {}}},
        {2, 3,  3, {1, 1, 0}, {{1}, {1}, {}}},
        {2, 3,  5, {1, 2, 0}, {{1}, {1,1}, {}}},
        {2, 3,  7, {1, 3, 0}, {{1}, {1,1,1}, {}}},
        {2, 3,  9, {2, 3, 0}, {{1,1}, {1,1,1}, {}}},
        {2, 4,  0, {4, 0, 0}, {{4}, {}, {}}},
        {2, 4,  2, {1, 4, 0}, {{1}, {4}, {}}},
        {2, 4,  4, {2, 6, 0}, {{2}, {4,2}, {}}},
        {2, 4,  6, {3, 6, 0}, {{2,1}, {4,2}, {}}},
        {2, 4,  8, {4, 8, 0}, {{3,1}, {4,3,1}, {}}},
        {2, 4, 10, {3, 14, 0}, {{2,1}, {4,4,4,2}, {}}},
        {2, 4, 12, {6, 10, 0}, {{3,2,1}, {4,3,2,1}, {}}},
        {2, 4, 14, {4, 15, 0}, {{2,1,1}, {4,4,4,2,1}, {}}},
        {2, 4, 16, {7, 19, 0}, {{4,2,1}, {4,4,4,4,2,1}, {}}},
        {2, 4, 18, {6, 17, 0}, {{2,2,1,1}, {4,4,4,2,2,1}, {}}},
        {2, 4, 20, {7, 19, 0}, {{3,2,1,1}, {4,4,4,3,2,1,1}, {}}},
        {2, 4,  1, {0, 1, 0}, {{}, {1}, {}}},
        {2, 4,  3, {1, 1, 0}, {{1}, {1}, {}}},
        {2, 4,  5, {1, 2, 0}, {{1}, {1,1}, {}}},
        {2, 4,  7, {1, 3, 0}, {{1}, {1,1,1}, {}}},
        {2, 4,  9, {2, 3, 0}, {{1,1}, {1,1,1}, {}}},
        {3, 2,  0, {2, 0, 0}, {{2}, {}, {}}},
        {3, 2,  2, {0, 2, 0}, {{}, {2}, {}}},
        {3, 2,  4, {1, 2, 0}, {{1}, {2}, {}}},
        {3, 2,  6, {1, 3, 0}, {{1}, {2,1}, {}}},
        {3, 2,  8, {1, 3, 0}, {{1}, {2,1}, {}}},
        {3, 2, 10, {1, 6, 0}, {{1}, {2,2,2}, {}}},
        {3, 2, 12, {3, 5, 0}, {{2,1}, {2,2,1}, {}}},
        {3, 2, 14, {1, 7, 0}, {{1}, {2,2,2,1}, {}}},
        {3, 2, 16, {2, 7, 0}, {{1,1}, {2,2,2,1}, {}}},
        {3, 2, 18, {3, 9, 0}, {{2,1}, {2,2,2,2,1}, {}}},
        {3, 2, 20, {2, 8, 0}, {{1,1}, {2,2,2,1,1}, {}}},
        {3, 2,  1, {0, 0, 0}, {{}, {}, {}}},
        {3, 2,  3, {0, 0, 0}, {{}, {}, {}}},
        {3, 2,  5, {0, 0, 0}, {{}, {}, {}}},
        {3, 2,  7, {0, 0, 0}, {{}, {}, {}}},
        {3, 2,  9, {0, 0, 0}, {{}, {}, {}}},
        {3, 3,  0, {3, 0, 0}, {{3}, {}, {}}},
        {3, 3,  2, {0, 3, 0}, {{}, {3}, {}}},
        {3, 3,  4, {1, 3, 0}, {{1}, {3}, {}}},
        {3, 3,  6, {1, 4, 0}, {{1}, {3,1}, {}}},
        {3, 3,  8, {1, 4, 0}, {{1}, {3,1}, {}}},
        {3, 3, 10, {1, 9, 0}, {{1}, {3,3,3}, {}}},
        {3, 3, 12, {3, 6, 0}, {{2,1}, {3,2,1}, {}}},
        {3, 3, 14, {1, 10, 0}, {{1}, {3,3,3,1}, {}}},
        {3, 3, 16, {2, 10, 0}, {{1,1}, {3,3,3,1}, {}}},
        {3, 3, 18, {3, 12, 0}, {{2,1}, {3,3,3,2,1}, {}}},
        {3, 3, 20, {2, 11, 0}, {{1,1}, {3,3,3,1,1}, {}}},
        {3, 3,  1, {0, 0, 0}, {{}, {}, {}}},
        {3, 3,  3, {0, 0, 0}, {{}, {}, {}}},
        {3, 3,  5, {0, 0, 0}, {{}, {}, {}}},
        {3, 3,  7, {0, 0, 0}, {{}, {}, {}}},
        {3, 3,  9, {0, 0, 0}, {{}, {}, {}}},
        {3, 4,  0, {4, 0, 0}, {{4}, {}, {}}},
        {3, 4,  2, {0, 4, 0}, {{}, {4}, {}}},
        {3, 4,  4, {1, 4, 0}, {{1}, {4}, {}}},
        {3, 4,  6, {1, 5, 0}, {{1}, {4,1}, {}}},
        {3, 4,  8, {1, 5, 0}, {{1}, {4,1}, {}}},
        {3, 4, 10, {1, 12, 0}, {{1}, {4,4,4}, {}}},
        {3, 4, 12, {3, 7, 0}, {{2,1}, {4,2,1}, {}}},
        {3, 4, 14, {1, 13, 0}, {{1}, {4,4,4,1}, {}}},
        {3, 4, 16, {2, 13, 0}, {{1,1}, {4,4,4,1}, {}}},
        {3, 4, 18, {3, 15, 0}, {{2,1}, {4,4,4,2,1}, {}}},
        {3, 4, 20, {2, 14, 0}, {{1,1}, {4,4,4,1,1}, {}}},
        {3, 4,  1, {0, 0, 0}, {{}, {}, {}}},
        {3, 4,  3, {0, 0, 0}, {{}, {}, {}}},
        {3, 4,  5, {0, 0, 0}, {{}, {}, {}}},
        {3, 4,  7, {0, 0, 0}, {{}, {}, {}}},
        {3, 4,  9, {0, 0, 0}, {{}, {}, {}}},
        {5, 2,  6, {1, 3, 0}, {{1}, {2,1}, {}}},
        {5, 2, 10, {0, 6, 0}, {{}, {2,2,2}, {}}},
        {5, 2, 16, {0, 6, 0}, {{}, {2,2,2}, {}}},
        {5, 2, 30, {2, 12, 0}, {{2}, {2,2,2,2,2,2}, {}}},
        {5, 3, 30, {2, 17, 0}, {{2}, {3,3,3,3,3,2}, {}}},
    };
    return t;
}

uint64_t g_checks = 0;
uint32_t g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) return;
    ++g_fail;
    if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
}
std::string vs(const std::vector<uint32_t>& v) {
    std::ostringstream o; o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}
std::string cell_name(uint32_t n, uint32_t ell, uint32_t k) {
    return "(n=" + std::to_string(n) + ", ell=" + std::to_string(ell) + ", k=" + std::to_string(k) + ")";
}
template<class F>
bool throws_logic_only(F&& f) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild&) { return false; }
    catch (const std::logic_error&) { return true; } catch (...) { return false; }
    return false;
}

std::array<uint32_t, 4> reduced(const hecke::Mat22& g, uint32_t p) {
    const std::array<uint32_t, 4> a = g.reduced_mod(p);
    return {a[0], a[1], a[2], a[3]};
}
bool same_flat(const FlatMatrixT<uint32_t>& A, const FlatMatrixT<uint32_t>& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols()) return false;
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            if (A.get_entry(i, j) != B.get_entry(i, j)) return false;
    return true;
}

// ---- (R) rho_n over Z/ell^k at ell in {2, 3} -------------------------------------------
void rho_arm(const ModCtx<uint32_t>& ctx, uint32_t n, std::map<std::string, uint64_t>& routes) {
    std::vector<hecke::Mat22> gs = {hecke::Mat22::T(), hecke::Mat22::S(), hecke::Mat22::ST(), hecke::Mat22::TS(),
                                    hecke::Mat22::I().neg()};
    for (const hecke::Mat22& g : gs) {
        const auto r = reduced(g, ctx.p);
        RhoRingRoute route = RhoRingRoute::UnitA;
        const FlatMatrixT<uint32_t> fast = ComputeSymmetricPowerMatrixRing<uint32_t>(ctx, r[0], r[1], r[2], r[3], n, &route);
        const FlatMatrixT<uint32_t> orc = ComputeSymmetricPowerMatrixRingOracle<uint32_t>(ctx, r[0], r[1], r[2], r[3], n);
        const FlatMatrixT<uint32_t> via = hecke::RhoOfExactRing<uint32_t>(ctx, g, n);
        ++routes[to_text(route)];
        check(same_flat(fast, orc) && same_flat(via, orc),
              "(R) rho_n(g) == the division-free oracle at " + cell_name(n, ctx.ell, ctx.k));
    }
}

// ---- (T) the in-process tree model over R (consistency, not an oracle) -------------
struct TreeResult { std::array<std::vector<uint32_t>, 3> inv; };
DenseBuf<uint32_t> minus_identity(const ModCtx<uint32_t>& ctx, const FlatMatrixT<uint32_t>& M) {
    const uint32_t d = M.get_rows();
    DenseBuf<uint32_t> out(d, d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) out.at(i, j) = ctx.sub(M.get_entry(i, j), i == j ? 1u : 0u);
    return out;
}
DenseBuf<uint32_t> kernel_columns(const ModCtx<uint32_t>& ctx, const DenseBuf<uint32_t>& A) {
    const engine::elim::HowellResult<uint32_t> h = engine::elim::kernel_module<uint32_t>(ctx, engine::cview(A));
    DenseBuf<uint32_t> out(A.cols, h.rows());
    for (uint32_t c = 0; c < h.rows(); ++c)
        for (uint32_t i = 0; i < A.cols; ++i) out.at(i, c) = h.H.at(c, i);
    return out;
}
TreeResult tree_model(const ModCtx<uint32_t>& ctx, uint32_t n) {
    const uint32_t d = n + 1;
    const DenseBuf<uint32_t> MS = minus_identity(ctx, hecke::RhoOfExactRing<uint32_t>(ctx, hecke::Mat22::S(), n));
    const DenseBuf<uint32_t> MU = minus_identity(ctx, hecke::RhoOfExactRing<uint32_t>(ctx, hecke::Mat22::ST(), n));
    const DenseBuf<uint32_t> MN = minus_identity(ctx, hecke::RhoOfExactRing<uint32_t>(ctx, hecke::Mat22::I().neg(), n));
    DenseBuf<uint32_t> MSU(2 * d, d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) { MSU.at(i, j) = MS.at(i, j); MSU.at(d + i, j) = MU.at(i, j); }
    const DenseBuf<uint32_t> VS = kernel_columns(ctx, MS), VU = kernel_columns(ctx, MU), VN = kernel_columns(ctx, MN);
    const DenseBuf<uint32_t> V0 = kernel_columns(ctx, MSU);
    DenseBuf<uint32_t> VSU(d, VS.cols + VU.cols);
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < VS.cols; ++j) VSU.at(i, j) = VS.at(i, j);
        for (uint32_t j = 0; j < VU.cols; ++j) VSU.at(i, VS.cols + j) = VU.at(i, j);
    }
    // V^S + V^U inside V^{-I} (S^2 = U^3 = -I): (V^{-I} + V^S + V^U) has V^{-I}'s length.
    DenseBuf<uint32_t> ALL(d, VN.cols + VSU.cols);
    for (uint32_t i = 0; i < d; ++i) {
        for (uint32_t j = 0; j < VN.cols; ++j) ALL.at(i, j) = VN.at(i, j);
        for (uint32_t j = 0; j < VSU.cols; ++j) ALL.at(i, VN.cols + j) = VSU.at(i, j);
    }
    auto len_cols = [&ctx](const DenseBuf<uint32_t>& G) -> uint32_t {
        if (G.cols == 0) return 0;
        DenseBuf<uint32_t> Gt(G.cols, G.rows);
        for (uint32_t i = 0; i < G.rows; ++i)
            for (uint32_t j = 0; j < G.cols; ++j) Gt.at(j, i) = G.at(i, j);
        return engine::elim::length<uint32_t>(ctx, engine::cview(Gt));
    };
    check(len_cols(ALL) == len_cols(VN), "(T) V^S + V^U lies inside V^{-I} at " + cell_name(n, ctx.ell, ctx.k));
    TreeResult r;
    r.inv[0] = V0.cols ? engine::elim::invariant_factors<uint32_t>(ctx, engine::cview(V0)) : std::vector<uint32_t>{};
    r.inv[1] = VN.cols ? engine::elim::invariant_factors<uint32_t>(ctx, engine::cview(VN), engine::cview(VSU))
                       : std::vector<uint32_t>{};
    return r;
}

// ---- the pipeline over Z/ell^k (inside the caller's ScopedCoefficientRing) ----------
struct Pipe {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
};
std::unique_ptr<Pipe> ring_pipeline(const ModCtx<uint32_t>& ctx, uint32_t n) {
    auto P = std::make_unique<Pipe>();
    auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(P->K, n, ctx);
    P->lssc = std::make_unique<LocalSystemSimplicialComplex>(P->K, std::move(rels), n + 1, ctx.p);
    P->coh = std::make_unique<EquivariantCohomologyOnK>(P->lssc.get());
    P->coh->compute_cohomology();
    return P;
}
void relation_matrices_arm(const ModCtx<uint32_t>& ctx, uint32_t n) {
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const auto rels = hecke::GenerateSL2ZRelationsRing<uint32_t>(K, n, ctx);
    const auto& seeds = SL2ZRelationSeeds_7VertexModel();
    bool ok = rels.size() >= seeds.size();
    for (size_t r = 0; ok && r < rels.size(); ++r) {
        const hecke::Mat22 g = (r < seeds.size()) ? seeds[r].g_r : hecke::Mat22::I().neg();
        const auto e = reduced(g, ctx.p);
        ok = same_flat(rels[r].M, ComputeSymmetricPowerMatrixRingOracle<uint32_t>(ctx, e[0], e[1], e[2], e[3], n));
    }
    check(ok, "(R) every GenerateSL2ZRelationsRing matrix == the oracle rho_n(g_r) at " + cell_name(n, ctx.ell, ctx.k));
}

} // namespace

uint32_t HeckeBredonTreeRingValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeBredonTreeRingValidation (slot-9 WP-3; E8 at ell in {2, 3}, [C] def:ringsemantics(2), "
                 "prop:bredon(2)-(3))" << std::endl;
    // (P) provenance
    std::cout << "  (P) O7 frozen from docs/implementation/integral_line_probe.py model_R, blob " << kO7Blob
              << ", command `--model-R <ell> <k> <n,...>`, frozen 2026-09-24; " << O7Table().size() << " cells"
              << std::endl;
    std::cout << "  (T) is a CONSISTENCY CHECK, not an oracle ([IB] K6): it shares the Howell / Smith code "
                 "(ENGINE_API.md Sec. 13.6) with the pipeline; its rho_n is RhoOfExactRing "
                 "(ComputeSymmetricPowerMatrixRing), checked against the division-free oracle in (R)" << std::endl;

    std::map<std::string, uint64_t> routes;
    uint32_t a_cells = 0, t_cells = 0, k1_cells = 0, n_cells = 0;
    std::vector<std::string> odd_named;
    for (const O7Cell& c : O7Table()) {
        const auto tc = std::chrono::steady_clock::now();
        if (c.k == 1) {
            // (K1) O7 at k = 1 against the shipped F_ell pipeline (default ring).
            SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
            auto rels = GenerateSL2ZRelationsElt<uint32_t>(K, c.n, c.ell);
            LocalSystemSimplicialComplex lssc(K, std::move(rels), c.n + 1, c.ell);
            EquivariantCohomologyOnK coh(&lssc);
            coh.compute_cohomology();
            const std::vector<uint32_t>& dims = coh.get_H_eq_dimensions();
            bool ok = dims.size() == 3;
            for (uint32_t q = 0; ok && q < 3; ++q)
                ok = dims[q] == c.len[q] && c.inv[q] == std::vector<uint32_t>(c.len[q], 1u);
            check(ok, "(K1) O7 at k = 1 == dim H^q of the F_ell pipeline at " + cell_name(c.n, c.ell, 1));
            ++k1_cells;
            continue;
        }
        const ModCtx<uint32_t> ctx = ModCtx<uint32_t>::prime_power(c.ell, c.k);
        hecke::ScopedCoefficientRing ring(c.k);
        std::unique_ptr<Pipe> P;
        try { P = ring_pipeline(ctx, c.n); }
        catch (const std::exception& e) {
            check(false, "pipeline threw at " + cell_name(c.n, c.ell, c.k) + ": " + e.what());
            continue;
        }
        const EquivariantCohomologyOnK& coh = *P->coh;
        const bool bredon = hecke::IsBredonPrime(c.ell);
        check(coh.is_flat_route() == bredon, "(A) the route is the " + std::string(bredon ? "flat-coordinate" : "C_eq") +
                                                 " one at " + cell_name(c.n, c.ell, c.k));
        std::array<std::vector<uint32_t>, 3> got;
        std::vector<uint32_t> lens;
        try {
            check(coh.num_degrees() == 3, "(A) three degrees at " + cell_name(c.n, c.ell, c.k));
            for (uint32_t q = 0; q < 3; ++q) got[q] = coh.H_invariant_factors(q);
            lens = coh.H_lengths();
        } catch (const std::exception& e) {
            check(false, "(A) an honouring accessor threw at " + cell_name(c.n, c.ell, c.k) + ": " + e.what());
            continue;
        }
        bool ok = lens.size() == 3;
        for (uint32_t q = 0; ok && q < 3; ++q) ok = got[q] == c.inv[q] && lens[q] == c.len[q];
        const std::string arm = bredon ? "(A)" : "(N)";
        check(ok, arm + " the pipeline == O7 at " + cell_name(c.n, c.ell, c.k) + ": got H^0 " + vs(got[0]) + " H^1 " +
                      vs(got[1]) + " H^2 " + vs(got[2]) + ", O7 " + vs(c.inv[0]) + " " + vs(c.inv[1]) + " " + vs(c.inv[2]));
        if (!bredon) { ++n_cells; continue; }
        ++a_cells;
        // the certificates' bookkeeping, re-read: sum e_i = length(Z) - length(B)
        for (uint32_t q = 0; q < 3; ++q) {
            const auto& F = coh.flat_module_degree(q);
            check(F.length_H == F.length_Z - F.length_B && F.length_H == c.len[q],
                  "(A) length(H^q) = length(Z^q) - length(B^q) at " + cell_name(c.n, c.ell, c.k));
        }
        // (R) the rho_n arm on this cell's g's, and the relation list's matrices
        rho_arm(ctx, c.n, routes);
        relation_matrices_arm(ctx, c.n);
        // (T) the in-process tree model
        const TreeResult T = tree_model(ctx, c.n);
        check(T.inv[0] == got[0] && T.inv[1] == got[1] && got[2].empty(),
              "(T) the in-process tree model == the pipeline at " + cell_name(c.n, c.ell, c.k) + ": tree H^0 " +
                  vs(T.inv[0]) + " H^1 " + vs(T.inv[1]));
        ++t_cells;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
        if (c.n % 2 == 1) {
            std::ostringstream o;
            o << "  (A) -I / odd-n cell " << cell_name(c.n, c.ell, c.k) << ": H^0 " << vs(got[0]) << " H^1 "
              << vs(got[1]) << " H^2 " << vs(got[2]) << " (O7 " << vs(c.inv[0]) << " " << vs(c.inv[1]) << " "
              << vs(c.inv[2]) << "), " << s << " s";
            odd_named.push_back(o.str());
        }
    }
    for (const std::string& line : odd_named) std::cout << line << std::endl;

    // (G) the dispatch guards.
    {
        const ModCtx<uint32_t> c5 = ModCtx<uint32_t>::prime_power(5u, 2);
        hecke::ScopedCoefficientRing ring(2);
        auto P5 = ring_pipeline(c5, 6);
        check(!P5->coh->is_flat_route() && throws_logic_only([&] { (void)P5->coh->flat_module_degree(0); }),
              "(G) ell = 5: the C_eq route; flat_module_degree() refuses (logic_error)");
        const ModCtx<uint32_t> c2 = ModCtx<uint32_t>::prime_power(2u, 2);
        auto P2 = ring_pipeline(c2, 4);
        check(throws_logic_only([&] { (void)P2->coh->get_H_eq_dimensions(); }) &&
                  throws_logic_only([&] { (void)P2->coh->get_H_eq_bases_fm(); }),
              "(G) ell = 2 under Z/ell^2: the field accessors refuse (logic_error, no dim)");
        check(throws_logic_only([&] { (void)P2->coh->flat_module_degree(3); }) == false,
              "(G) flat_module_degree(3) is out of range (runtime_error), not a logic_error");
    }

    std::cout << "  rho_n routes at ell in {2, 3}:";
    for (const auto& kv : routes) std::cout << " " << kv.first << "=" << kv.second;
    std::cout << std::endl;
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  cells: (A) " << a_cells << ", (T) " << t_cells << " at ell in {2, 3}, of which " << odd_named.size()
              << " -I / odd-n named above; (K1) " << k1_cells << "; (N) " << n_cells << std::endl;
    std::cout << "HeckeBredonTreeRingValidation: " << (g_fail ? "FAIL" : "PASS") << " (" << g_checks << " checks, "
              << g_fail << " failing, " << s << " s)" << std::endl;
    return g_fail;
}
