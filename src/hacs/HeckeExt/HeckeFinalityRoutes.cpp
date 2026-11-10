// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 12, FIN -- (d0) "dickson" and (d1) "boundary".
// See HeckeFinalityRoutes.h.
#include "HeckeFinalityRoutes.h"

#include "HeckeAlgebraInvariants.h"
#include "HeckeUEllBudget.h"

#include "../../ModArithmetic/ScalarMod.h"

#include <algorithm>

namespace hecke_ext {

namespace {

bool IsPrimeSmall(uint64_t q) {
    if (q < 2) return false;
    for (uint64_t d = 2; d * d <= q; ++d)
        if (q % d == 0) return false;
    return true;
}

bool AllDigits(const std::string& s) {
    if (s.empty() || s.size() > 19) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}

uint64_t PowMod(uint64_t b, uint64_t e, uint64_t m) { return engine::scalar::powmod<uint64_t>(b, e, m); }

// X as a 1 x D^2 row, row-major -- the layout of LocalAlgebra::coords.
ExtMatrix VecRow(const ExtMatrix& X) {
    ExtMatrix r(1, X.rows * X.cols, X.p);
    for (uint32_t i = 0; i < X.rows; ++i)
        for (uint32_t j = 0; j < X.cols; ++j) r.data[0][i * X.cols + j] = X.data[i][j];
    return r;
}

} // namespace

bool IsDicksonRouteValue(const std::string& v) {
    if (v == kRouteNotRun || v == kRoutePassed || v == kRouteNotApplicableUEll) return true;
    const std::string p = kRouteFailedPrefix;
    return v.size() > p.size() && v.compare(0, p.size(), p) == 0 && AllDigits(v.substr(p.size()));
}

bool IsBoundaryRouteValue(const std::string& v) {
    if (v == kRouteNotRun || v == kRoutePassed || v == kRouteNotApplicableUEll ||
        v == kRouteNotInjective || v == kRouteNotRunBudget) {
        return true;
    }
    const std::string p = kRouteFailedPrefix;
    return v.size() > p.size() && v.compare(0, p.size(), p) == 0 && AllDigits(v.substr(p.size()));
}

uint64_t LeastPrimeInClass(uint32_t ell, uint32_t cls) {
    for (uint64_t q = cls; ; q += ell) {
        if (q != ell && IsPrimeSmall(q)) return q;
    }
}

std::vector<uint64_t> BoundaryRepresentatives(uint32_t ell) {
    std::vector<uint64_t> out;
    for (uint32_t cls = 1; cls < ell; ++cls) out.push_back(LeastPrimeInClass(ell, cls));
    return out;
}

double BoundaryStreamCostMb(uint32_t n, uint32_t ell) {
    double mb = 0.0;
    for (uint64_t p : BoundaryRepresentatives(ell))
        if (p > 31) mb += PredictUEllRssMb(n, p);
    return mb;
}

bool BoundaryStreamWithinBudget(uint32_t n, uint32_t ell, uint32_t u_ell_budget_mb) {
    return BoundaryStreamCostMb(n, ell) <=
           static_cast<double>(kBoundaryStreamBudgetFactor) * static_cast<double>(u_ell_budget_mb);
}

std::vector<uint32_t> DicksonClassesPresent(uint32_t n, uint32_t ell) {
    std::vector<uint32_t> cls;
    const uint64_t dl = ell + 1ull, dd = static_cast<uint64_t>(ell) * (ell - 1ull);
    for (uint64_t a = 0; a * dl <= n; ++a) {
        if ((n - a * dl) % dd != 0) continue;
        const uint32_t c = static_cast<uint32_t>(a % (ell - 1ull));
        if (std::find(cls.begin(), cls.end(), c) == cls.end()) cls.push_back(c);
    }
    std::sort(cls.begin(), cls.end());
    return cls;
}

std::string DicksonRoute(uint32_t n, uint32_t ell, const std::vector<uint64_t>& primes,
                         const std::vector<ExtMatrix>& t) {
    if (primes.size() != t.size()) return "invalid:operator_list";
    if (std::find(primes.begin(), primes.end(), static_cast<uint64_t>(ell)) == primes.end())
        return kRouteNotApplicableUEll;
    // Every T_q|_V (q != ell) must be a scalar lambda_q ([C, prop:heckeH0](1)).
    std::vector<std::pair<uint64_t, uint32_t>> lambda;
    for (size_t i = 0; i < primes.size(); ++i) {
        if (primes[i] == ell) continue;
        const ExtMatrix& M = t[i];
        if (M.rows == 0 || M.rows != M.cols) return "invalid:shape";
        const uint32_t l = M.data[0][0];
        for (uint32_t r = 0; r < M.rows; ++r)
            for (uint32_t c = 0; c < M.cols; ++c)
                if (M.data[r][c] != (r == c ? l : 0u)) return "invalid:not_scalar";
        lambda.push_back({primes[i], l});
    }
    // The present classes carrying the observed S-eigenvalues.
    std::vector<uint32_t> A;
    for (uint32_t a : DicksonClassesPresent(n, ell)) {
        bool ok = true;
        for (const auto& e : lambda) {
            const uint64_t q = e.first % ell;
            const uint64_t v = ((1 + q) % ell) * PowMod(q, a, ell) % ell;
            ok = ok && v == e.second;
        }
        if (ok) A.push_back(a);
    }
    if (A.empty()) return "invalid:no_class";
    if (A.size() == 1) return kRoutePassed;
    return std::string(kRouteFailedPrefix) + std::to_string(A[1]);
}

bool InAlgebraSpan(const ExtMatrix& coords, const ExtMatrix& X) {
    const ExtMatrix x = VecRow(X);
    if (coords.rows == 0) {
        for (uint32_t j = 0; j < x.cols; ++j)
            if (x.data[0][j] != 0u) return false;
        return true;
    }
    if (coords.cols != x.cols) return false;
    ExtMatrix S(coords.rows + 1, coords.cols, coords.p);
    for (uint32_t i = 0; i < coords.rows; ++i) S.data[i] = coords.data[i];
    S.data[coords.rows] = x.data[0];
    return hecke::RankOf(S) == hecke::RankOf(coords);
}

uint32_t BoundaryFinalityCertificate::verify() const {
    if (!present) return 0;
    if (!IsBoundaryRouteValue(outcome) || outcome == kRouteNotRun) return 1;
    if (outcome == kRouteNotApplicableUEll) return 0;
    const uint32_t rk = (r1.rows == 0) ? 0u : hecke::RankOf(r1 * V);
    if (rk != rank_r1) return 2;
    if ((outcome == kRouteNotInjective) != (rk != V.cols)) return 2;
    if (reps != BoundaryRepresentatives(ell)) return 3;
    if (outcome == kRouteNotInjective) return reps_ambient.empty() ? 0u : 4u;
    const bool within = BoundaryStreamWithinBudget(n, ell, budget_mb);
    if ((outcome == kRouteNotRunBudget) == within) return 5;
    if (outcome == kRouteNotRunBudget) return reps_ambient.empty() ? 0u : 4u;
    // Membership by the INDEPENDENT route: the algebra generated by A_S's
    // generators and T_p|_V has A_S's dimension iff T_p|_V is in A_S.
    const uint32_t dimA = ComputeAlgebraInvariants(generators, ell).dim_A;
    bool all_in = true;
    std::string first_fail;
    for (size_t i = 0; i < reps_ambient.size(); ++i) {
        const ExtMatrix AV = reps_ambient[i] * V;
        ExtMatrix t;
        try {
            t = hecke::SolveColumnsExact(V, AV, "BoundaryFinalityCertificate::verify");
        } catch (const std::exception&) {
            return 4;
        }
        std::vector<ExtMatrix> g = generators;
        g.push_back(t);
        const bool in = ComputeAlgebraInvariants(g, ell).dim_A == dimA;
        if (!in && all_in) first_fail = std::string(kRouteFailedPrefix) + std::to_string(reps[i]);
        all_in = all_in && in;
        if (!in) break;
    }
    if (all_in) {
        if (reps_ambient.size() != reps.size() || outcome != kRoutePassed) return 4;
    } else if (outcome != first_fail) {
        return 4;
    }
    return 0;
}

namespace {
bool SameMatrix(const ExtMatrix& a, const ExtMatrix& b) {
    return a.rows == b.rows && a.cols == b.cols && a.data == b.data;
}
bool EntriesBelow(const ExtMatrix& M, uint32_t ell) {
    if (M.data.size() != M.rows) return false;
    for (const auto& row : M.data) {
        if (row.size() != M.cols) return false;
        for (uint32_t x : row)
            if (x >= ell) return false;
    }
    return true;
}
} // namespace

uint32_t BoundaryDataRederive(uint32_t ell, const std::vector<uint64_t>& primes,
                              const std::vector<ExtMatrix>& generators,
                              const std::vector<uint64_t>& reps, uint32_t rank_r1,
                              const ExtMatrix& r1, const std::vector<ExtMatrix>& t) {
    // Shapes and keys: r1 is h x D, every t_p is D x D, t index-aligned with
    // R(ell), the stored generators D x D and index-aligned with S.
    if (primes.size() != generators.size() || generators.empty()) return 1;
    const uint32_t D = generators.front().rows;
    if (D == 0 || r1.cols != D || reps != BoundaryRepresentatives(ell) || t.size() != reps.size()) return 1;
    for (const ExtMatrix& g : generators)
        if (g.rows != D || g.cols != D) return 1;
    for (const ExtMatrix& m : t)
        if (m.rows != D || m.cols != D) return 1;
    if (!EntriesBelow(r1, ell)) return 2;
    for (const ExtMatrix& m : t)
        if (!EntriesBelow(m, ell)) return 2;
    // (H1): r^1|_V injective, re-derived.
    if (hecke::RankOf(r1) != rank_r1) return 3;
    if (rank_r1 != D) return 4;
    // A representative inside S is the stored t_p itself.
    for (size_t i = 0; i < reps.size(); ++i)
        for (size_t j = 0; j < primes.size(); ++j)
            if (primes[j] == reps[i] && !SameMatrix(generators[j], t[i])) return 5;
    // T_infty is commutative: every t_p commutes with A_S's generators.
    for (const ExtMatrix& m : t)
        for (const ExtMatrix& g : generators)
            if (!SameMatrix(m * g, g * m)) return 6;
    // (H2): t_p in A_S -- the algebra generated by the stored generators and
    // t_p has A_S's dimension (the independent closure route, as in
    // BoundaryFinalityCertificate::verify()).
    const uint32_t dimA = ComputeAlgebraInvariants(generators, ell).dim_A;
    for (const ExtMatrix& m : t) {
        std::vector<ExtMatrix> g = generators;
        g.push_back(m);
        if (ComputeAlgebraInvariants(g, ell).dim_A != dimA) return 7;
    }
    return 0;
}

} // namespace hecke_ext
