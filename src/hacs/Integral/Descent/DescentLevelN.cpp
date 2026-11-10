// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D1: the level-N integral groups by the descent.  See the header; [C] = docs/latex/gl2_contract.tex.
#include "DescentLevelN.h"
#include "DescentTree.h"
#include "DescentParAsm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>

#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNSparse.h"
#include "../../../ModArithmetic/Binary/Ring3k.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {

using engine::DenseBuf;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;

namespace {
double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }
}  // namespace

// ---------------------------------------------------------------------------
// (1) closed forms
// ---------------------------------------------------------------------------
uint32_t SymInvariantDim(uint32_t n, uint32_t order) {
    ENGINE_CHECK_MSG(order == 4 || order == 6, "SymInvariantDim: order 4 (S) or 6 (ST)");
    // tr g^i on Sym^n = sum_k zeta^{i(n - 2k)}, zeta = exp(2 pi i / order); 2 Re zeta^r is an integer for order 4, 6.
    const int c4[4] = {2, 0, -2, 0}, c6[6] = {2, 1, -1, -2, -1, 1};
    int64_t twice = 0;
    for (uint32_t i = 0; i < order; ++i)
        for (uint32_t k = 0; k <= n; ++k) {
            const int64_t e = (int64_t(i) * (int64_t(n) - 2 * int64_t(k))) % int64_t(order);
            const uint32_t r = static_cast<uint32_t>((e + order) % order);
            twice += order == 4 ? c4[r] : c6[r];
        }
    ENGINE_CHECK_MSG(twice >= 0 && twice % (2 * order) == 0, "SymInvariantDim: the character sum is not 2 * order * dim");
    return static_cast<uint32_t>(twice / (2 * order));
}

LevelNCounts LevelNClosedForm(uint32_t N, uint32_t n) {
    ENGINE_CHECK_MSG(N >= 1, "LevelNClosedForm: N >= 1");
    LevelNCounts c;
    c.N = N;
    c.n = n;
    const uint64_t m = hecke::level::IndexGamma0(N);
    ENGINE_CHECK_MSG(m * (uint64_t(n) + 1) < (uint64_t(1) << 31), "LevelNClosedForm: D = m_N (n + 1) must stay below 2^31");
    c.m = static_cast<uint32_t>(m);
    c.d = n + 1;
    c.D = c.m * c.d;
    c.nu2 = hecke::level::Nu2(N);
    c.nu3 = hecke::level::Nu3(N);
    if (n % 2 == 1) return c;   // M_N(-I) = -1: everything is 2-torsion, b = 0
    c.dS = SymInvariantDim(n, 4);
    c.dST = SymInvariantDim(n, 6);
    ENGINE_CHECK((c.m - c.nu2) % 2 == 0 && (c.m - c.nu3) % 3 == 0);
    // prop:levelorbit (1): a free orbit contributes a copy of V, a fixed point V^{F_j} ~ V^{<g>} (conjugate)
    c.dimMS = (c.m - c.nu2) / 2 * c.d + c.nu2 * c.dS;
    c.dimMST = (c.m - c.nu3) / 3 * c.d + c.nu3 * c.dST;
    c.rank_T0 = c.dimMS + c.dimMST;
    c.b0 = n == 0 ? 1u : 0u;
    // prop:levelorbit (3) over Q: b_1 - b_0 = D - dim M^S - dim M^ST
    const int64_t b1 = int64_t(c.b0) + int64_t(c.D) - int64_t(c.rank_T0);
    ENGINE_CHECK_MSG(b1 >= 0, "LevelNClosedForm: negative b_1");
    c.b1 = static_cast<uint32_t>(b1);
    return c;
}

uint32_t DeltaCount(uint32_t ell, uint32_t n) {
    uint32_t cnt = 0;
    const uint64_t a = ell + 1, b = uint64_t(ell) * ell - ell;
    for (uint64_t y = 0; y * b <= n; ++y)
        if ((n - y * b) % a == 0) ++cnt;
    return cnt;
}
uint32_t BetaCount(uint32_t ell, uint32_t n, uint32_t t) {
    uint32_t cnt = 0;
    for (uint64_t j = 0; j * ell <= n; ++j)
        if ((int64_t(n) - 2 * int64_t(j) + int64_t(t)) % int64_t(ell - 1) == 0) ++cnt;
    return cnt;
}
uint32_t PredictedSummands(uint32_t N, uint32_t n, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell), "PredictedSummands: ell >= 5 prime");
    if (n == 0 || n % 2 == 1) return 0;
    return N % ell == 0 ? BetaCount(ell, n) : DeltaCount(ell, n);
}
std::vector<std::pair<uint32_t, uint32_t>> BadPrimesN(uint32_t N, uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> out;
    if (n == 0 || n % 2 == 1) return out;
    for (uint32_t ell = 5; ell <= n + 1; ++ell) {   // prop:badprimesN(2): every ell > n + 1 is torsion-free
        if (!IsPrime(ell)) continue;
        const uint32_t s = PredictedSummands(N, n, ell);
        if (s > 0) out.emplace_back(ell, s);
    }
    return out;
}
uint32_t GoodPrimeN(uint32_t N, uint32_t n) {
    uint32_t g = std::max<uint32_t>(10007, n + 2);
    while (!IsPrime(g) || N % g == 0) ++g;
    return g;
}

// ---------------------------------------------------------------------------
// (2), (3) the sparse cokernel over Z/ell^K
// ---------------------------------------------------------------------------
namespace {

struct SpVec {
    std::vector<uint32_t> col;   // ascending
    std::vector<uint32_t> val;   // nonzero mod q
};

// The generators of M^{<gamma>} (gamma = S, o = 2; gamma = ST, o = 3 on M, n even), orbit by orbit (prop:levelorbit (1)):
// on a free orbit the d vectors sum_{i<o} M^i e_{(j0, a)} (fixed; restriction to block j0 is e_a, so they are a basis of
// the orbit's fixed module: no division), at a fixed point the d columns of (1/o) sum_i B^i (the averaging idempotent).
std::vector<SpVec> FixedGenerators(const hecke::level::Gamma0LevelN& A, const hecke::Mat22& gamma, uint32_t o, uint32_t n,
                                   const U32Ctx& ctx, uint32_t& fixed_points) {
    const hecke::level::MonomialAction M = hecke::level::MonomialOf(A, gamma, n, ctx);
    const uint32_t m = M.m, d = M.d;
    const uint64_t q = ctx.p;
    std::vector<uint32_t> pinv(m + 1, 0);
    for (uint32_t j = 1; j <= m; ++j) pinv[M.perm[j]] = j;
    std::vector<uint8_t> seen(m + 1, 0);
    std::vector<SpVec> out;
    fixed_points = 0;
    for (uint32_t j0 = 1; j0 <= m; ++j0) {
        if (seen[j0]) continue;
        std::vector<uint32_t> orb{j0};
        seen[j0] = 1;
        for (uint32_t j = M.perm[j0]; j != j0; j = M.perm[j]) {
            ENGINE_CHECK_MSG(!seen[j], "FixedGenerators: orbits overlap");
            seen[j] = 1;
            orb.push_back(j);
        }
        if (orb.size() == 1) {
            ++fixed_points;
            ENGINE_CHECK_MSG(std::gcd<uint64_t>(o, q) == 1, "FixedGenerators: a fixed point with the order not a unit");
            const std::vector<uint32_t>& B = M.blocks[M.blk[j0]].dense;
            std::vector<uint64_t> e(static_cast<size_t>(d) * d, 0), P(static_cast<size_t>(d) * d, 0), T;
            for (uint32_t i = 0; i < d; ++i) P[size_t(i) * d + i] = 1;
            for (uint32_t i = 0; i < o; ++i) {
                for (size_t x = 0; x < e.size(); ++x) e[x] = (e[x] + P[x]) % q;
                if (i + 1 == o) break;
                T.assign(e.size(), 0);
                for (uint32_t r = 0; r < d; ++r)
                    for (uint32_t s = 0; s < d; ++s) {
                        const uint64_t brs = B[size_t(r) * d + s];
                        if (!brs) continue;
                        for (uint32_t t = 0; t < d; ++t) T[size_t(r) * d + t] = (T[size_t(r) * d + t] + brs * P[size_t(s) * d + t]) % q;
                    }
                P.swap(T);
            }
            const uint64_t inv_o = ctx.inv(o % ctx.p);
            for (uint32_t c = 0; c < d; ++c) {
                SpVec f;
                for (uint32_t a = 0; a < d; ++a) {
                    const uint32_t v = static_cast<uint32_t>(e[size_t(a) * d + c] * inv_o % q);
                    if (v) { f.col.push_back((j0 - 1) * d + a); f.val.push_back(v); }
                }
                if (!f.col.empty()) out.push_back(std::move(f));
            }
            continue;
        }
        ENGINE_CHECK_MSG(orb.size() == o, "FixedGenerators: an orbit neither free nor a fixed point");
        for (uint32_t a = 0; a < d; ++a) {
            std::map<uint32_t, std::vector<uint64_t>> blocks;
            std::vector<uint64_t> w(d, 0);
            w[a] = 1;
            uint32_t jc = j0;
            for (uint32_t i = 0; i < o; ++i) {
                auto& acc = blocks[jc];
                if (acc.empty()) acc.assign(d, 0);
                for (uint32_t r = 0; r < d; ++r) acc[r] = (acc[r] + w[r]) % q;
                if (i + 1 == o) break;
                // M (vector at block jc) lives at block j = pinv[jc]: (M f)_j = B_j f_{jc}
                const uint32_t j = pinv[jc];
                const std::vector<uint32_t>& B = M.blocks[M.blk[j]].dense;
                std::vector<uint64_t> w2(d, 0);
                for (uint32_t r = 0; r < d; ++r) {
                    uint64_t s = 0;
                    for (uint32_t t = 0; t < d; ++t) s = (s + uint64_t(B[size_t(r) * d + t]) * w[t]) % q;
                    w2[r] = s;
                }
                w.swap(w2);
                jc = j;
            }
            SpVec f;
            for (const auto& [j, v] : blocks)
                for (uint32_t r = 0; r < d; ++r)
                    if (v[r]) { f.col.push_back((j - 1) * d + r); f.val.push_back(static_cast<uint32_t>(v[r])); }
            if (!f.col.empty()) out.push_back(std::move(f));
        }
    }
    return out;
}

std::vector<SpVec> AllGenerators(uint32_t N, uint32_t n, const U32Ctx& ctx, uint32_t& fixS, uint32_t& fixU) {
    const hecke::level::Gamma0LevelN A(N);
    std::vector<SpVec> g = FixedGenerators(A, hecke::Mat22::S(), 2, n, ctx, fixS);
    std::vector<SpVec> u = FixedGenerators(A, hecke::Mat22::ST(), 3, n, ctx, fixU);
    g.insert(g.end(), std::make_move_iterator(u.begin()), std::make_move_iterator(u.end()));
    return g;
}

}  // namespace

SparseElimination EliminateSparseCokernel(uint32_t N, uint32_t n, uint32_t ell, uint32_t K) {
    const auto t0 = clk::now();
    ENGINE_CHECK_MSG(n % 2 == 0, "RunSparseCokernelSmith: n even (M^{-I} = M)");
    ENGINE_CHECK_MSG(IsPrime(ell) && ell >= 3, "RunSparseCokernelSmith: ell >= 3 prime");
    const LevelNCounts c = LevelNClosedForm(N, n);
    ENGINE_CHECK_MSG(ell >= 5 || c.nu3 == 0,
                     "RunSparseCokernelSmith: at ell = 3 only when Gamma_0(N) has no element of order 3 ([C, prop:modeltorsion]; "
                     "[C, prop:descentNthree])");
    if (K == 0) K = ell == 3 ? engine::ring3k::kMaxK : integral_model::DefaultKBudget(ell);
    ENGINE_CHECK_MSG(K >= 2, "RunSparseCokernelSmith: K >= 2");
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const uint64_t q = ctx.p;
    SparseElimination E;
    SparseSmithRun& out = E.run;
    out.ell = ell;
    out.K = K;
    out.D = c.D;
    out.rank_Q = c.rank_T0 - c.b0;   // [C, prop:descentN](7)(b): r = rank T^0 - b_0
    uint32_t fS = 0, fU = 0;
    std::vector<SpVec> gens = AllGenerators(N, n, ctx, fS, fU);
    ENGINE_CHECK_MSG(fS == c.nu2 && fU == c.nu3, "RunSparseCokernelSmith: fixed-point counts != nu_2, nu_3");
    out.gen_seconds = since(t0);
    const auto te = clk::now();
    // generators by their largest column, descending (the shipped SparseCokernel order)
    std::vector<uint32_t> seq(gens.size());
    for (uint32_t i = 0; i < seq.size(); ++i) seq[i] = i;
    std::stable_sort(seq.begin(), seq.end(), [&](uint32_t a, uint32_t b) { return gens[a].col.back() > gens[b].col.back(); });
    const uint32_t D = c.D;
    std::vector<SpVec> rows;                  // pivot rows: tails (no pivot column), pivot entry 1 implicit
    std::vector<int32_t> row_of_col(D, -1);
    std::vector<std::vector<uint32_t>> occ(D);
    std::vector<SpVec> deferred;
    std::vector<uint64_t> acc(D, 0);
    std::vector<uint8_t> touched(D, 0);
    std::vector<uint32_t> tl;
    uint64_t live = 0;
    auto add = [&](uint32_t col, uint64_t v) {
        if (!touched[col]) { touched[col] = 1; tl.push_back(col); }
        acc[col] = (acc[col] + v) % q;
    };
    auto take = [&]() {
        std::sort(tl.begin(), tl.end());
        SpVec v;
        for (uint32_t col : tl) {
            if (acc[col]) { v.col.push_back(col); v.val.push_back(static_cast<uint32_t>(acc[col])); }
            acc[col] = 0;
            touched[col] = 0;
        }
        tl.clear();
        return v;
    };
    auto reduce = [&](const SpVec& g) {
        for (size_t k = 0; k < g.col.size(); ++k) {
            const uint32_t col = g.col[k];
            if (row_of_col[col] >= 0) {
                const SpVec& t = rows[static_cast<size_t>(row_of_col[col])];
                const uint64_t mul = (q - g.val[k]) % q;
                for (size_t i = 0; i < t.col.size(); ++i) add(t.col[i], mul * t.val[i] % q);
                out.merge_ops += t.col.size();
            } else {
                add(col, g.val[k]);
            }
        }
        return take();
    };
    for (uint32_t gi : seq) {
        ++out.generators;
        out.generator_nnz += gens[gi].col.size();
        SpVec v = reduce(gens[gi]);
        SpVec().col.swap(gens[gi].col);
        if (v.col.empty()) { ++out.dependent; continue; }
        size_t pk = v.col.size();
        for (size_t k = 0; k < v.col.size(); ++k)
            if (v.val[k] % ell != 0) { pk = k; break; }
        if (pk == v.col.size()) { deferred.push_back(std::move(v)); continue; }
        const uint32_t piv = v.col[pk];
        const uint64_t iv = ctx.inv(v.val[pk]);
        SpVec tail;
        for (size_t k = 0; k < v.col.size(); ++k)
            if (k != pk) { tail.col.push_back(v.col[k]); tail.val.push_back(static_cast<uint32_t>(v.val[k] * iv % q)); }
        // back substitution into every stored row holding piv
        std::vector<uint32_t> rw;
        rw.swap(occ[piv]);
        std::sort(rw.begin(), rw.end());
        rw.erase(std::unique(rw.begin(), rw.end()), rw.end());
        for (uint32_t r : rw) {
            SpVec& row = rows[r];
            auto it = std::lower_bound(row.col.begin(), row.col.end(), piv);
            if (it == row.col.end() || *it != piv) continue;
            const uint64_t a = row.val[static_cast<size_t>(it - row.col.begin())];
            const uint64_t mul = (q - a) % q;
            SpVec nr;
            size_t i = 0, j = 0;
            while (i < row.col.size() || j < tail.col.size()) {
                if (j == tail.col.size() || (i < row.col.size() && row.col[i] < tail.col[j])) {
                    if (row.col[i] != piv) { nr.col.push_back(row.col[i]); nr.val.push_back(row.val[i]); }
                    ++i;
                } else if (i == row.col.size() || tail.col[j] < row.col[i]) {
                    const uint32_t x = static_cast<uint32_t>(mul * tail.val[j] % q);
                    if (x) { nr.col.push_back(tail.col[j]); nr.val.push_back(x); occ[tail.col[j]].push_back(r); }
                    ++j;
                } else {
                    const uint32_t x = static_cast<uint32_t>((row.val[i] + mul * tail.val[j]) % q);
                    if (x) { nr.col.push_back(row.col[i]); nr.val.push_back(x); }
                    ++i;
                    ++j;
                }
                ++out.merge_ops;
            }
            live = live + nr.col.size() - row.col.size();
            row = std::move(nr);
        }
        const uint32_t id = static_cast<uint32_t>(rows.size());
        for (uint32_t col : tail.col) occ[col].push_back(id);
        live += tail.col.size();
        out.peak_nnz = std::max(out.peak_nnz, live);
        row_of_col[piv] = static_cast<int32_t>(id);
        rows.push_back(std::move(tail));
        ++out.units;
    }
    for (const auto& r : rows) out.final_nnz += r.col.size();
    std::vector<std::vector<uint32_t>>().swap(occ);
    // the deferred rows (every entry divisible by ell), reduced against the final rows (no pivot column in a tail: one pass)
    std::vector<SpVec> res;
    std::map<uint32_t, uint32_t> colmap;
    for (const SpVec& z : deferred) {
        SpVec v = reduce(z);
        if (v.col.empty()) continue;
        for (size_t k = 0; k < v.col.size(); ++k) {
            ENGINE_CHECK_MSG(v.val[k] % ell == 0, "RunSparseCokernelSmith: a deferred row acquired a unit");
            ENGINE_CHECK_MSG(row_of_col[v.col[k]] < 0, "RunSparseCokernelSmith: a deferred row holds a pivot column");
            colmap.emplace(v.col[k], 0);
        }
        res.push_back(std::move(v));
    }
    out.deferred = deferred.size();
    out.elim_seconds = since(te);
    // (PAR-LN) the residual columns numbered densely in ascending order (the colmap order of record); the elimination's rows, occ and
    // work arrays are freed on return, so only the residual reaches the second stage
    uint32_t ci = 0;
    for (auto& kv : colmap) kv.second = ci++;
    out.residual_rows = static_cast<uint32_t>(res.size());
    out.residual_cols = ci;
    E.residual.resize(res.size());
    for (uint32_t i = 0; i < res.size(); ++i) {
        auto& row = E.residual[i];
        row.reserve(res[i].col.size());
        for (size_t k = 0; k < res[i].col.size(); ++k) row.emplace_back(colmap[res[i].col[k]], res[i].val[k]);
        E.residual_nnz += row.size();
        SpVec().col.swap(res[i].col);
        SpVec().val.swap(res[i].val);
    }
    out.seconds = since(t0);
    return E;
}

SparseSmithRun FinishSparseCokernelSmith(SparseElimination&& e, uint64_t residual_cap_bytes) {
    SparseSmithRun out = std::move(e.run);
    const double t_elim = out.seconds;
    const uint32_t ell = out.ell, K = out.K;
    const auto tr = clk::now();
    std::vector<uint32_t> sigma(out.units, 0u);
    const bool any = !e.residual.empty();
    if (any && 4ull * out.residual_rows * out.residual_cols > residual_cap_bytes) {
        // (D1 audit m3) the dense residual is never started above the ceiling: not certified, the descent refuses
        out.residual_refused = true;
        out.residual_seconds = since(tr);
        out.r_K = out.units;
        out.certified = false;
        out.seconds = t_elim + out.residual_seconds;
        return out;
    }
    if (any) {
        DenseBuf<uint32_t> Z(out.residual_rows, out.residual_cols);
        for (uint32_t i = 0; i < e.residual.size(); ++i)
            for (const auto& [col, val] : e.residual[i]) Z.at(i, col) = val;
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>>().swap(e.residual);
        const std::vector<uint32_t> s2 = SmithLevelsEll(ell, K, std::move(Z));
        for (uint32_t x : s2) {
            ENGINE_CHECK_MSG(x >= 1, "RunSparseCokernelSmith: a unit valuation in the residual");
            sigma.push_back(x);
        }
    }
    out.residual_seconds = since(tr);
    out.r_K = static_cast<uint32_t>(sigma.size());
    ENGINE_CHECK_MSG(out.r_K <= out.rank_Q, "RunSparseCokernelSmith: r_K > rank T^0 - b_0");
    out.certified = out.r_K == out.rank_Q;
    if (out.certified)
        for (uint32_t x : sigma)
            if (x >= 1) out.tors1.add_torsion(ell, x);
    out.seconds = t_elim + out.residual_seconds;
    return out;
}

SparseSmithRun RunSparseCokernelSmith(uint32_t N, uint32_t n, uint32_t ell, uint32_t K, uint64_t residual_cap_bytes) {
    return FinishSparseCokernelSmith(EliminateSparseCokernel(N, n, ell, K), residual_cap_bytes);
}

uint64_t PredictSparseElimRss(uint32_t N, uint32_t n) {
    const LevelNCounts c = LevelNClosedForm(N, n);
    const uint64_t D = c.D, d = c.d;
    // the generators: at most 2 D d entries (a free S-orbit's d vectors have 2d entries, a free ST-orbit's 3d, a fixed point's d), 8 B
    // each, plus about 80 B of vector headers and allocator overhead per generator (at most 2 D generators)
    const uint64_t gens = 16 * D * d + 160 * D;
    // the pivot rows at their peak fill: peak_nnz entries, 8 B in the row plus 4 B per occ entry.  peak_nnz is the graph fill of the
    // elimination and has no closed form.  Measured (PARLN_report.md Sec. 3): one ell >= 5 elimination peak_nnz / (D d sqrt(D)) <= 0.5
    // from (35, 30) to (1000, 30); the good prime's shipped SparseCokernel (an item of the same concurrent stage) grows by about
    // 230-280 B x D^1.5 whatever d (1.56 GiB at (1000, 20), 1.72 at (2000, 10), 3.43 at (1000, 30)).  The term takes
    // 12 B x D^1.5 x max(2 d, 50) -- at least a 2x margin on both -- and never more than the dense bound: at most rank T^0 <= D pivot
    // rows of at most D entries each.
    const double emp = double(D) * std::sqrt(double(D)) * std::max(2.0 * double(d), 50.0);
    const uint64_t dense = D * D;
    const uint64_t fill = 12 * std::min<uint64_t>(dense, static_cast<uint64_t>(emp));
    // the monomial actions of S and ST (dense d x d blocks per coset: 2 x 4 D d B), the per-column arrays (acc 8, row_of_col 4,
    // touched 1, occ header 24, row header 48 B), and the coset tables
    const uint64_t rest = 8 * D * d + 96 * D;
    return gens + fill + rest + (64ull << 20);
}

uint64_t PredictSparseResidualRss(uint32_t rows, uint32_t cols) {
    // SmithLevelsEll: the residual A, its reduction Af / the transposed Bt, the Schur complement Sc (each <= rows x cols uint32)
    return 16ull * rows * cols + (32ull << 20);
}

DenseBuf<uint32_t> FixedGeneratorRows(uint32_t N, uint32_t n, const U32Ctx& ctx) {
    ENGINE_CHECK_MSG(n % 2 == 0, "FixedGeneratorRows: n even (M_N^{<-I>} = M_N)");
    const LevelNCounts c = LevelNClosedForm(N, n);
    // (D2 audit F7) the orders 2 (S) and 3 (ST) at the fixed points must be units in ctx
    ENGINE_CHECK_MSG((c.nu2 == 0 || ctx.p % 2 != 0) && (c.nu3 == 0 || ctx.p % 3 != 0),
                     "FixedGeneratorRows: an order 2 or 3 at a fixed point is not a unit in ctx");
    uint32_t fS = 0, fU = 0;
    const std::vector<SpVec> gens = AllGenerators(N, n, ctx, fS, fU);
    ENGINE_CHECK_MSG(fS == c.nu2 && fU == c.nu3, "FixedGeneratorRows: fixed-point counts != nu_2, nu_3");
    DenseBuf<uint32_t> A(static_cast<uint32_t>(gens.size()), c.D);
    for (uint32_t i = 0; i < gens.size(); ++i)
        for (size_t k = 0; k < gens[i].col.size(); ++k) A.at(i, gens[i].col[k]) = gens[i].val[k];
    return A;
}

std::vector<uint32_t> DenseCokernelSmith(uint32_t N, uint32_t n, uint32_t ell, uint32_t K) {
    const LevelNCounts c = LevelNClosedForm(N, n);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    uint32_t fS = 0, fU = 0;
    const std::vector<SpVec> gens = AllGenerators(N, n, ctx, fS, fU);
    DenseBuf<uint32_t> A(static_cast<uint32_t>(gens.size()), c.D);
    for (uint32_t i = 0; i < gens.size(); ++i)
        for (size_t k = 0; k < gens[i].col.size(); ++k) A.at(i, gens[i].col[k]) = gens[i].val[k];
    if (ell == 3) return engine::ring3k::smith(std::move(A), K).sigma;
    return SmithLevelsEll(ell, K, std::move(A));
}

// ---------------------------------------------------------------------------
// the assembly
// ---------------------------------------------------------------------------
LevelNResult AssembleDescentN(uint32_t N, uint32_t n, const LevelNOptions& opt) {
    const auto t0 = clk::now();
    LevelNResult R;
    R.c = LevelNClosedForm(N, n);
    const LevelNCounts& c = R.c;
    const bool tree = DescentTreeEnabled();   // (D5 audit N6) read once per call
    std::ostringstream why;
    std::vector<AbelianGroupType> H(6);
    H[0].free_rank = c.b0;
    H[1].free_rank = c.b1;
    const bool even = n % 2 == 0;

    // (PAR-LN) the independent, single-threaded sparse stages -- the good prime's shipped SparseCokernel, the ell >= 5 eliminations
    // and the ell = 3 shortcut's elimination (nu_3 = 0) -- run concurrently, up to JointWorkers at a time under the joint prediction
    // (rule 19: PredictSparseElimRss each), each into its own slot.  Then everything is consumed in the serial order: (1) the good
    // prime, (2) per prime its dense residual -- whose size is known only now -- under the ceiling rule of record, with the descent
    // Smith's own threads, (3) the shortcut's residual.  An exception is re-thrown where the serial code would have met it first.
    // Lever HECKE_DESCENT_PAR_LN=0: the serial code (each stage in place, one after another).
    R.good_prime = GoodPrimeN(N, n);
    R.bad = BadPrimesN(N, n);
    struct GoodOut {
        uint32_t h0 = 0, h1 = 0;
        double t = 0;
    };
    auto good_run = [&]() {
        GoodOut g;
        const auto t = clk::now();
        const hecke::level::Gamma0LevelN A(N);
        const auto Q = hecke::level::SparseCokernel::Build(A, n, U32Ctx(R.good_prime), false);
        g.h0 = Q.h0();
        g.h1 = Q.h1();
        g.t = since(t);
        return g;
    };
    const bool shortcut3 = even && !opt.skip_three && c.nu3 == 0 && !opt.force_dense_three;
    const bool pre = par::Enabled(par::Lever::EllN);
    const bool pre_ell = pre && opt.ell_route == EllRoute::Sparse;
    GoodOut good;
    std::exception_ptr good_err, three_err;
    std::vector<SparseElimination> el(pre_ell ? R.bad.size() : 0);
    std::vector<std::exception_ptr> el_err(el.size());
    SparseElimination el3;
    if (pre) {
        // the items in the serial order: [good], ell >= 5 ..., [ell = 3]
        std::vector<int> item;   // -1 good, -2 three, i >= 0 the i-th bad prime
        if (even) item.push_back(-1);
        for (size_t i = 0; i < el.size(); ++i) item.push_back(static_cast<int>(i));
        if (shortcut3) item.push_back(-2);
        const uint32_t m = static_cast<uint32_t>(item.size());
        R.ell_items = m;
        R.ell_rss_pred_one = m ? PredictSparseElimRss(N, n) : 0;
        const uint32_t T = par::Threads();
        R.ell_workers = par::JointWorkers(m, R.ell_rss_pred_one, opt.rss_cap_bytes, T);
        if (m > 1) par::NoteLn(R.ell_workers > 1, R.ell_workers < std::min<uint32_t>(T, m));   // audit A3: own counters
        R.ell_concurrent = R.ell_workers > 1;
        R.ell_rss_pred_joint = R.ell_rss_pred_one * R.ell_workers;
        const uint64_t base = CurrentRssBytes();
        RssSampler rs1;
        const auto te = clk::now();
        par::ForEachIndex(m, R.ell_workers, [&](size_t k) {
            const int it = item[k];
            try {
                if (it == -1) good = good_run();
                else if (it == -2) el3 = EliminateSparseCokernel(N, n, 3, 0);
                else el[static_cast<size_t>(it)] = EliminateSparseCokernel(N, n, R.bad[static_cast<size_t>(it)].first, opt.K_ell);
            } catch (...) {
                (it == -1 ? good_err : it == -2 ? three_err : el_err[static_cast<size_t>(it)]) = std::current_exception();
            }
        });
        R.t_ell_elim = since(te);
        const uint64_t peak = rs1.stop();
        R.ell_rss_growth = peak > base ? peak - base : 0;
    }

    // (1) the closed forms against the shipped sparse cokernel at the good prime
    if (even) {
        if (pre && good_err) std::rethrow_exception(good_err);
        if (!pre) good = good_run();
        R.good_h0 = good.h0;
        R.good_h1 = good.h1;
        R.t_good = good.t;
        const int64_t b1x = int64_t(c.b1) + opt.test_b1_offset;
        if (R.good_h0 != c.b0 || int64_t(R.good_h1) != b1x)
            why << " closed form b_0, b_1 = " << c.b0 << ", " << b1x << " != the good prime's " << R.good_h0 << ", " << R.good_h1 << ";";
    }

    // (2) ell >= 5
    {
        RssSampler rs;
        auto t = clk::now();
        // one prime's verdict, consumed in the serial order (ascending ell)
        auto tally = [&](uint32_t ell, uint32_t s, bool cert, const AbelianGroupType& tors) {
            const auto it = tors.torsion.find(ell);
            const uint32_t got = it == tors.torsion.end() ? 0u : static_cast<uint32_t>(it->second.size());
            if (!cert) why << " ell=" << ell << " not certified at the budget K;";
            else if (int64_t(got) != int64_t(s) + opt.test_count_offset)
                why << " ell=" << ell << ": " << got << " summands != prop:badprimesN count " << int64_t(s) + opt.test_count_offset << ";";
            else
                for (uint32_t a : it->second) H[1].add_torsion(ell, a);
        };
        auto sparse_done = [&](SparseSmithRun&& r, uint32_t s) {
            const uint32_t ell = r.ell;
            if (r.residual_refused) why << " ell=" << ell << ": the dense residual is above the RSS ceiling;";
            const bool cert = r.certified;
            const AbelianGroupType tors = r.tors1;
            R.sparse.push_back(std::move(r));
            tally(ell, s, cert, tors);
        };
        if (pre_ell) {
            for (size_t i = 0; i < el.size(); ++i) {
                if (el_err[i]) std::rethrow_exception(el_err[i]);
                R.ell_residual_pred_max = std::max(R.ell_residual_pred_max,
                                                   PredictSparseResidualRss(el[i].run.residual_rows, el[i].run.residual_cols));
                sparse_done(FinishSparseCokernelSmith(std::move(el[i]), opt.rss_cap_bytes), R.bad[i].second);
                SparseElimination().residual.swap(el[i].residual);
            }
            R.t_ell_residual = since(t);
        } else {
            for (const auto& [ell, s] : R.bad) {
                if (opt.ell_route == EllRoute::Sparse) {
                    sparse_done(RunSparseCokernelSmith(N, n, ell, opt.K_ell, opt.rss_cap_bytes), s);
                } else if (424ull * c.D * c.D + (200ull << 20) > opt.rss_cap_bytes) {
                    // (D1 audit m3) the dense direct Smith of C_eq(M_N) (x) Z/ell^K: never started above the ceiling
                    why << " ell=" << ell << ": the direct Smith RSS prediction " << ((424ull * c.D * c.D) >> 20) << " MiB is above the ceiling;";
                    continue;
                } else {
                    const uint32_t K = opt.K_ell ? opt.K_ell : integral_model::DefaultKBudget(ell);
                    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
                    BredonSmithRun r = RunBredonDirectSmith(InducedSymModule(N, n, ctx), ell, K, c.b0);
                    const bool cert = r.certified && r.free_rank1 == c.b1;
                    const AbelianGroupType tors = r.tors1;
                    R.bredon.push_back(std::move(r));
                    tally(ell, s, cert, tors);
                }
            }
        }
        R.t_ell = since(t);
        R.rss_meas_ell = rs.stop();
    }

    std::vector<uint32_t> h(8, 0);
    h[0] = c.b0;
    h[1] = c.b1;
    // (3) ell = 3
    if (even && !opt.skip_three) {
        auto t = clk::now();
        if (c.nu3 == 0 && !opt.force_dense_three) {
            // [C, prop:descentNthree] (pre.17; D1 code audit M1): the nu_3 = 0 shortcut at ell = 3
            R.three_shortcut = true;
            if (pre) {
                if (three_err) std::rethrow_exception(three_err);
                R.three_sparse = FinishSparseCokernelSmith(std::move(el3), opt.rss_cap_bytes);   // (PAR-LN) eliminated in the stage above
            } else {
                R.three_sparse = RunSparseCokernelSmith(N, n, 3, 0, opt.rss_cap_bytes);
            }
            if (!R.three_sparse.certified) why << " the ell = 3 shortcut (nu_3 = 0) not certified;";
            else
                for (const auto& [p, ex] : R.three_sparse.tors1.torsion)
                    for (uint32_t a : ex) H[1].add_torsion(p, a);
        } else {
            // (D5) the tree model Iso^(3)_T unless HECKE_DESCENT_TREE=0 (the dense rollback)
            const bool tree3 = tree;
            R.rss_pred_three = tree3 ? PredictTreeLocalRss(3, c.D, n + 1) : PredictLocalRss(3, c.D, 6);   // dense: D1 audit m6
            if (R.rss_pred_three > opt.rss_cap_bytes) {
                why << (tree3 ? " Iso^(3)_T" : " Iso^(3)") << " RSS prediction " << (R.rss_pred_three >> 20) << " MiB above the ceiling "
                    << (opt.rss_cap_bytes >> 20) << " MiB;";
            } else {
                RssSampler rs;
                const U32Ctx C = U32Ctx::prime_power(3, LocalWidth(3));
                const uint32_t g = R.good_prime;
                if (tree3)
                    R.three = RunTreeLocalPieceN(3, N, n, 6, h, opt.rank_from_betti ? 0u : g);
                else
                    R.three = RunLocalPiece(Iso3(), InducedSymModule(N, n, C), C, 6, h, g,
                                            opt.rank_from_betti ? InducedSymModule(N, n, C) : InducedSymModule(N, n, U32Ctx(g)), -1,
                                            false, opt.rank_from_betti);
                R.rss_meas_three = rs.stop();
                R.ran_three = true;
                if (!R.three.all_certified()) why << (tree3 ? " Iso^(3)_T (tree) not certified;" : " Iso^(3) not certified;");
                else
                    for (uint32_t q = 1; q <= 5; ++q)
                        for (const auto& [p, ex] : R.three.tors[q].torsion)
                            for (uint32_t a : ex) H[q].add_torsion(p, a);
            }
        }
        R.t_three = since(t);
    } else if (opt.skip_three) {
        why << " the 3-adic piece skipped (measurement);";
    }

    // (4) ell = 2
    if (!opt.skip_two) {
        auto t = clk::now();
        // (D5) the tree model Iso^(2)_T unless HECKE_DESCENT_TREE=0 (the dense rollback)
        const bool tree2 = tree;
        R.rss_pred_two = tree2 ? PredictTreeLocalRss(2, c.D, n + 1) : PredictLocalRss(2, c.D, 4);
        if (R.rss_pred_two > opt.rss_cap_bytes) {
            why << (tree2 ? " Iso^(2)_T" : " Iso^(2)") << " RSS prediction " << (R.rss_pred_two >> 20) << " MiB above the ceiling "
                << (opt.rss_cap_bytes >> 20) << " MiB;";
        } else {
            RssSampler rs;
            const U32Ctx C = U32Ctx::prime_power(2, LocalWidth(2));
            const uint32_t g = R.good_prime;
            if (tree2)
                R.two = RunTreeLocalPieceN(2, N, n, 4, h, opt.rank_from_betti ? 0u : g);
            else
                R.two = RunLocalPiece(Iso2(), InducedSymModule(N, n, C), C, 4, h, g,
                                      opt.rank_from_betti ? InducedSymModule(N, n, C) : InducedSymModule(N, n, U32Ctx(g)), -1, false,
                                      opt.rank_from_betti);
            R.rss_meas_two = rs.stop();
            R.ran_two = true;
            if (!R.two.all_certified())
                why << (tree2 ? " Iso^(2)_T (tree)" : " Iso^(2)") << " not certified at 2^30 (no W = 61 route for an Ind module);";
            else {
                for (uint32_t q = 1; q <= 4; ++q)
                    for (const auto& [p, ex] : R.two.tors[q].torsion)
                        for (uint32_t a : ex) H[q].add_torsion(p, a);
                // [C, prop:descentN](5): H^5 = H^3 by cup with res(e), q >= 2
                for (const auto& [p, ex] : R.two.tors[3].torsion)
                    for (uint32_t a : ex) H[5].add_torsion(p, a);
            }
        }
        R.t_two = since(t);
    } else {
        why << " the 2-adic piece skipped (measurement);";
    }
    R.t_total = since(t0);
    const std::string w = why.str();
    if (!w.empty()) {
        R.refused = true;
        R.refusal = "REFUSED N=" + std::to_string(N) + " n=" + std::to_string(n) + ":" + w;
    } else {
        R.H = std::move(H);
    }
    return R;
}

TreeTwoRun RunTreeTwoN(uint32_t N, uint32_t n, uint64_t rss_cap_bytes) {
    TreeTwoRun out;
    const LevelNCounts c = LevelNClosedForm(N, n);
    out.rss_pred = PredictTreeRss(c.D);
    if (out.rss_pred > rss_cap_bytes) return out;   // never started above the ceiling
    RssSampler rs;
    const auto t0 = clk::now();
    const uint32_t D = c.D;
    std::vector<uint32_t> rank = {2 * D - c.b0, 3 * D - c.b1 - (2 * D - c.b0)};
    for (uint32_t q = 2; q < 4; ++q) rank.push_back(3 * D - rank.back());
    const U32Ctx C = U32Ctx::prime_power(2, LocalWidth(2));
    std::vector<AbelianGroupType> tors;
    out.certified = TorsionFromDifferentials(TreeDifferentials(C, InducedSymModule(N, n, C), 4), C, rank, tors);
    out.tors = tors;
    out.seconds = since(t0);
    out.rss_meas = rs.stop();
    out.ran = true;
    return out;
}

}  // namespace integral_descent
