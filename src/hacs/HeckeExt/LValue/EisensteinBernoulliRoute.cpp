// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LV) The Eisenstein / Bernoulli instance -- see
// EisensteinBernoulliRoute.h for what is computed and why.
#include "EisensteinBernoulliRoute.h"

#include "../HeckeHidaTower.h"                 // hida::CharpolyFree (Berkowitz over Z/ell^e)
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/FpPoly.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeInvolution.h"       // BuildIotaSimplicialMapOnModel
#include "../../Hecke/HeckeOperatorFamily.h"   // tau_construction_mutex
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../Hecke/HeckeOrdinaryPart.h"     // RingMat, FreeRankH1
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseElim.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"
#include "../../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <sstream>

namespace hecke_ext {
namespace lvalue {

using u128 = unsigned __int128;
using Poly64 = hecke::FpPolyT<uint64_t>;

const char* ToString(Sign s) { return s == Sign::Plus ? "plus" : "minus"; }

namespace {

uint64_t PowFit(uint64_t ell, uint32_t k) {
    const uint64_t r = engine::scalar::powfit(ell, k);   // 0 when ell^k >= 2^64
    ENGINE_CHECK_MSG(r != 0 && r < (uint64_t(1) << 62), "HeckeLValue: ell^K must stay below 2^62");
    return r;
}

// ---- free-module matrix arithmetic over Z/ell^K: the shared kit (../ZellK.h) ----
using zk::Lin;
using zk::MinusScalar;
using zk::Mul;
using zk::ReduceTo;
using zk::Zero;
inline FreeMat Id(uint64_t ell, uint32_t K, uint32_t r) { return zk::Identity(ell, K, r); }
inline bool Equal(const FreeMat& A, const FreeMat& B) { return A == B; }   // same ell and K throughout

// rank over F_ell of the columns `cols` (each of length r): the rank of the row list they form.
uint32_t RankModEll(const std::vector<std::vector<uint64_t>>& cols, uint64_t ell) {
    if (cols.empty()) return 0;
    return zk::FieldRank(cols, static_cast<uint32_t>(cols[0].size()), ell);
}

std::vector<uint64_t> Column(const FreeMat& A, uint32_t j) {
    std::vector<uint64_t> c(A.r);
    for (uint32_t i = 0; i < A.r; ++i) c[i] = A.at(i, j);
    return c;
}

// The image of an idempotent P (commuting with ops) as a free direct summand: a basis
// (columns of P independent mod ell, Nakayama) and the ops restricted to it, every
// restricted column re-verified on all r coordinates.
struct Restricted {
    uint32_t s = 0;
    std::vector<FreeMat> ops;
};
Restricted RestrictToImage(const FreeMat& P, const std::vector<FreeMat>& ops) {
    const uint64_t ell = P.ell;
    const uint32_t K = P.K, r = P.r;
    const uint64_t q = PowFit(ell, K);
    std::vector<std::vector<uint64_t>> basis;
    for (uint32_t j = 0; j < r; ++j) {
        std::vector<std::vector<uint64_t>> trial = basis;
        trial.push_back(Column(P, j));
        if (RankModEll(trial, ell) == trial.size()) basis.push_back(trial.back());
    }
    Restricted R;
    R.s = static_cast<uint32_t>(basis.size());
    const uint32_t s = R.s;
    if (s == 0) {
        for (size_t i = 0; i < ops.size(); ++i) R.ops.push_back(Zero(ell, K, 0));
        return R;
    }
    // rows `sel` with B_sel invertible mod ell: greedy on the rows of B
    std::vector<uint32_t> sel;
    {
        std::vector<std::vector<uint64_t>> rows;
        for (uint32_t i = 0; i < r && sel.size() < s; ++i) {
            std::vector<uint64_t> row(s);
            for (uint32_t c = 0; c < s; ++c) row[c] = basis[c][i];
            std::vector<std::vector<uint64_t>> trial = rows;
            trial.push_back(row);
            if (RankModEll(trial, ell) == trial.size()) {
                rows.push_back(row);
                sel.push_back(i);
            }
        }
        ENGINE_CHECK_MSG(sel.size() == s, "HeckeLValue: no invertible minor of the summand basis");
    }
    // inverse of B_sel over Z/ell^K (Gauss-Jordan, unit pivots)
    std::vector<std::vector<uint64_t>> Mx(s, std::vector<uint64_t>(2 * s, 0));
    for (uint32_t a = 0; a < s; ++a) {
        for (uint32_t c = 0; c < s; ++c) Mx[a][c] = basis[c][sel[a]];
        Mx[a][s + a] = 1;
    }
    for (uint32_t c = 0; c < s; ++c) {
        uint32_t piv = c;
        while (piv < s && Mx[piv][c] % ell == 0) ++piv;
        ENGINE_CHECK_MSG(piv < s, "HeckeLValue: the selected minor is not invertible");
        std::swap(Mx[piv], Mx[c]);
        const uint64_t inv = InvModU64(Mx[c][c], q);
        for (uint32_t j = 0; j < 2 * s; ++j) Mx[c][j] = static_cast<uint64_t>(static_cast<u128>(Mx[c][j]) * inv % q);
        for (uint32_t a = 0; a < s; ++a) {
            if (a == c || Mx[a][c] == 0) continue;
            const uint64_t f = Mx[a][c];
            for (uint32_t j = 0; j < 2 * s; ++j)
                Mx[a][j] = static_cast<uint64_t>((Mx[a][j] + static_cast<u128>(q - f) * Mx[c][j]) % q);
        }
    }
    for (const FreeMat& T : ops) {
        FreeMat X = Zero(ell, K, s);
        for (uint32_t b = 0; b < s; ++b) {
            // y = T basis[b]
            std::vector<uint64_t> y(r, 0);
            for (uint32_t i = 0; i < r; ++i) {
                u128 acc = 0;
                for (uint32_t k = 0; k < r; ++k) acc = (acc + static_cast<u128>(T.at(i, k)) * basis[b][k]) % q;
                y[i] = static_cast<uint64_t>(acc);
            }
            std::vector<uint64_t> x(s, 0);
            for (uint32_t a = 0; a < s; ++a) {
                u128 acc = 0;
                for (uint32_t c = 0; c < s; ++c) acc = (acc + static_cast<u128>(Mx[a][s + c]) * y[sel[c]]) % q;
                x[a] = static_cast<uint64_t>(acc);
            }
            for (uint32_t i = 0; i < r; ++i) {
                u128 acc = 0;
                for (uint32_t a = 0; a < s; ++a) acc = (acc + static_cast<u128>(basis[a][i]) * x[a]) % q;
                ENGINE_CHECK_MSG(static_cast<uint64_t>(acc) == y[i],
                                 "HeckeLValue: an operator does not preserve the summand (restriction does not re-verify)");
            }
            for (uint32_t a = 0; a < s; ++a) X.a[static_cast<size_t>(a) * s + b] = x[a];
        }
        R.ops.push_back(X);
    }
    return R;
}

hecke::ordinary::RingMat ToRingMat(const FreeMat& A) {
    hecke::ordinary::RingMat R;
    R.ell = A.ell;
    R.row_exp.assign(A.r, A.K);
    R.a = A.a;
    return R;
}

Poly64 CharpolyModEll(const FreeMat& A) {
    const hida::Poly chi = hida::CharpolyFree(ToRingMat(ReduceTo(A, 1)));
    return Poly64(A.ell, std::vector<uint64_t>(chi.begin(), chi.end()));
}

// multiplicity of the root a of chi (mod ell), and E with E = 1 mod (x-a)^mult, E = 0 mod the cofactor
uint32_t IdempotentPoly(const Poly64& chi, uint64_t a, uint64_t ell, Poly64& E) {
    const Poly64 lin(ell, {(ell - a % ell) % ell, 1});
    Poly64 g = chi, h = Poly64::one(ell);
    uint32_t mult = 0;
    for (;;) {
        Poly64 qq(ell), rr(ell);
        Poly64::divrem(g, lin, qq, rr);
        if (!rr.is_zero()) break;
        g = qq;
        h = h * lin;
        ++mult;
    }
    if (mult == 0) return 0;
    Poly64 u(ell), v(ell);
    const Poly64 d = Poly64::extgcd(h, g, u, v);
    ENGINE_CHECK(d == Poly64::one(ell));
    E = Poly64::mod(v * g, chi);
    return mult;
}

FreeMat EvalPoly(const Poly64& E, const FreeMat& T) { return zk::EvalPoly(E.coeffs, T); }   // Horner (ZellK.h)

// The m_Eis block of one sign: joint idempotent of the system a_p (mod ell), restricted.
SignBlock EisBlock(Sign sign, const std::vector<FreeMat>& side_ops, uint32_t side_rank, const std::vector<uint64_t>& a_sys,
                   uint64_t ell, uint32_t K) {
    SignBlock B;
    B.sign = sign;
    B.side_rank = side_rank;
    if (side_rank == 0) {
        for (size_t i = 0; i < side_ops.size(); ++i) B.ops.push_back(Zero(ell, K, 0));
        return B;
    }
    FreeMat e = Id(ell, K, side_rank);
    bool empty = false;
    for (size_t i = 0; i < side_ops.size(); ++i) {
        Poly64 E(ell);
        if (IdempotentPoly(CharpolyModEll(side_ops[i]), a_sys[i], ell, E) == 0) {
            empty = true;
            break;
        }
        e = Mul(e, EvalPoly(E, side_ops[i]));
    }
    if (empty) {
        for (size_t i = 0; i < side_ops.size(); ++i) B.ops.push_back(Zero(ell, K, 0));
        return B;
    }
    const uint64_t q = PowFit(ell, K);
    for (uint32_t it = 0;; ++it) {
        const FreeMat e2 = Mul(e, e);
        if (Equal(e2, e)) break;
        ENGINE_CHECK_MSG(it < 64, "HeckeLValue: the Newton lift of e_Eis did not converge");
        const FreeMat e3 = Mul(e2, e);
        e = Lin(e2, 3, e3, q - 2);   // 3 e^2 - 2 e^3
    }
    for (const FreeMat& T : side_ops) ENGINE_CHECK_MSG(Equal(Mul(e, T), Mul(T, e)), "HeckeLValue: e_Eis does not commute with T_p");
    const Restricted R = RestrictToImage(e, side_ops);
    B.f = R.s;
    B.ops = R.ops;
    // every T_p - a_p is nilpotent mod ell on the block (it is the block of the system)
    for (size_t i = 0; i < B.ops.size(); ++i) {
        FreeMat X = ReduceTo(MinusScalar(B.ops[i], a_sys[i]), 1);
        FreeMat P = Id(ell, 1, B.f);
        for (uint32_t t = 0; t < B.f; ++t) P = Mul(P, X);
        for (uint64_t x : P.a) ENGINE_CHECK_MSG(x == 0, "HeckeLValue: T_p - a_p is not nilpotent mod ell on the m_Eis block");
    }
    return B;
}

// ---- the algebra reading over Z/ell^K (Howell spans) ---------------------------
using Vec = std::vector<uint64_t>;

engine::DenseBuf<uint64_t> RowsBuf(const std::vector<Vec>& rows, uint32_t N) {
    engine::DenseBuf<uint64_t> A(static_cast<uint32_t>(rows.size()), N);
    for (uint32_t i = 0; i < rows.size(); ++i)
        for (uint32_t j = 0; j < N; ++j) A.at(i, j) = rows[i][j];
    return A;
}
engine::DenseBuf<uint64_t> ColsBuf(const std::vector<Vec>& cols, uint32_t N) {
    engine::DenseBuf<uint64_t> A(N, static_cast<uint32_t>(cols.size()));
    for (uint32_t j = 0; j < cols.size(); ++j)
        for (uint32_t i = 0; i < N; ++i) A.at(i, j) = cols[j][i];
    return A;
}

// Howell rows of the span, and its length.
std::pair<std::vector<Vec>, uint32_t> Span(const engine::ModCtx<uint64_t>& ctx, const std::vector<Vec>& gens, uint32_t N) {
    if (gens.empty()) return {{}, 0};
    const engine::DenseBuf<uint64_t> A = RowsBuf(gens, N);
    const engine::elim::HowellResult<uint64_t> H = engine::elim::howell(ctx, engine::cview(A));
    std::vector<Vec> rows;
    for (uint32_t i = 0; i < H.rows(); ++i) {
        Vec v(N);
        for (uint32_t j = 0; j < N; ++j) v[j] = H.H.at(i, j);
        rows.push_back(v);
    }
    return {rows, engine::elim::length(ctx, H)};
}

std::vector<Vec> Scaled(const std::vector<Vec>& xs, uint64_t f, uint64_t q) {
    std::vector<Vec> out = xs;
    for (Vec& v : out)
        for (uint64_t& x : v) x = static_cast<uint64_t>(static_cast<u128>(x) * f % q);
    return out;
}

Vec Flatten(const FreeMat& A) { return A.a; }
FreeMat Unflatten(const Vec& v, uint64_t ell, uint32_t K, uint32_t r) {
    FreeMat M = Zero(ell, K, r);
    M.a = v;
    return M;
}

// The algebra generated by the ops in End(B_K): Howell generators over Z/ell^K.
std::vector<Vec> AlgebraGenerators(const std::vector<FreeMat>& ops, uint64_t ell, uint32_t K, uint32_t f) {
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, K);
    const uint32_t N = f * f;
    auto sp = Span(ctx, {Flatten(Id(ell, K, f))}, N);
    for (uint32_t it = 0;; ++it) {
        ENGINE_CHECK_MSG(it <= N * K + 2, "HeckeLValue: the algebra closure did not terminate");
        std::vector<Vec> cand = sp.first;
        for (const FreeMat& T : ops)
            for (const Vec& b : sp.first) cand.push_back(Flatten(Mul(T, Unflatten(b, ell, K, f))));
        auto nx = Span(ctx, cand, N);
        if (nx.second == sp.second) break;
        sp = nx;
    }
    return sp.first;
}

uint32_t DimA1(const std::vector<FreeMat>& ops, uint64_t ell, uint32_t K, uint32_t f) {
    if (f == 0) return 0;
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, K);
    const uint64_t q = PowFit(ell, K);
    const std::vector<Vec> gens = AlgebraGenerators(ops, ell, K, f);
    return Span(ctx, Scaled(gens, PowFit(ell, K - 1), q), f * f).second;
}

} // namespace

// ---- (b)-(d) on a cell whose T, iota (free, mod ell^K) are filled ------------------
void FinishEisensteinCell(EisensteinCell& c, const std::vector<uint64_t>& e_values, int64_t expect_plus, int64_t expect_minus) {
    const uint64_t ell = c.ell;
    const uint32_t K = c.K, r = c.rank;
    const uint64_t q = PowFit(ell, K);
    ENGINE_CHECK_MSG(c.T.size() == c.primes.size() && e_values.size() == c.primes.size(), "HeckeLValue: T / e not aligned with S");
    ENGINE_CHECK_MSG(c.iota.r == r && c.iota.K == K, "HeckeLValue: iota has the wrong shape");
    ENGINE_CHECK_MSG(Equal(Mul(c.iota, c.iota), Id(ell, K, r)), "HeckeLValue: iota^* is not an involution on H_K");
    for (const FreeMat& T : c.T)
        ENGINE_CHECK_MSG(Equal(Mul(c.iota, T), Mul(T, c.iota)), "HeckeLValue: iota^* does not commute with T_p ([C, obl:iota](v))");
    for (size_t a = 0; a < c.T.size(); ++a)
        for (size_t b = a + 1; b < c.T.size(); ++b)
            ENGINE_CHECK_MSG(Equal(Mul(c.T[a], c.T[b]), Mul(c.T[b], c.T[a])), "HeckeLValue: the T_p do not commute on H_K");

    // (c) the signs
    const uint64_t inv2 = (q + 1) / 2;
    const FreeMat Pp = Lin(Id(ell, K, r), inv2, c.iota, inv2);
    const FreeMat Pm = Lin(Id(ell, K, r), inv2, c.iota, q - inv2);
    const Restricted Rp = RestrictToImage(Pp, c.T);
    const Restricted Rm = RestrictToImage(Pm, c.T);
    ENGINE_CHECK_MSG(Rp.s + Rm.s == r, "HeckeLValue: H^+ (+) H^- != H");
    if (expect_plus >= 0 && expect_minus >= 0)
        ENGINE_CHECK_MSG(Rp.s == expect_plus && Rm.s == expect_minus,
                         "HeckeLValue: rank H^+ / rank H^- differ from the dimension formula (the global sign of iota^* is off)");

    // (d) the Eisenstein system and its blocks
    c.e_values.clear();
    std::vector<uint64_t> a_sys;
    for (uint64_t e : e_values) {
        c.e_values.push_back(e % q);
        a_sys.push_back(e % ell);
    }
    c.side_plus = Rp.s;
    c.side_minus = Rm.s;
    c.plus = EisBlock(Sign::Plus, Rp.ops, Rp.s, a_sys, ell, K);
    c.minus = EisBlock(Sign::Minus, Rm.ops, Rm.s, a_sys, ell, K);
}

// ---- (a)-(d) -------------------------------------------------------------------
EisensteinCell RunEisensteinCell(uint32_t n, uint64_t ell, const EisensteinOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(n >= 2 && n % 2 == 0, "HeckeLValue: n >= 2 even (odd n vanishes on this model)");
    ENGINE_CHECK_MSG(ell >= 5, "HeckeLValue: ell >= 5 ([E] sec:lvalue)");
    ENGINE_CHECK_MSG(PrimitiveRootModPrime(ell) > 0, "HeckeLValue: ell must be prime");
    const uint32_t k = n + 2;
    ENGINE_CHECK_MSG(k % (ell - 1) != 0, "HeckeLValue: [E] def:lveis needs a = k mod (ell - 1) != 0");
    ENGINE_CHECK_MSG(opt.m_max >= 1, "HeckeLValue: m_max >= 1");
    EisensteinCell c;
    c.n = n;
    c.ell = ell;
    c.K = std::max<uint32_t>(2, opt.m_max + 1);
    const uint32_t K = c.K;
    const uint64_t q = PowFit(ell, K);
    c.primes = opt.primes.empty() ? std::vector<uint64_t>{2, 3, 5, 7, ell} : opt.primes;
    std::sort(c.primes.begin(), c.primes.end());
    c.primes.erase(std::unique(c.primes.begin(), c.primes.end()), c.primes.end());

    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const uint32_t d_k = (k % 12 == 2) ? k / 12 : k / 12 + 1;   // dim M_k
    ENGINE_CHECK_MSG(hecke::ordinary::FreeRankH1(n) == 2 * d_k - 1, "HeckeLValue: rank H^1 != 2 dim M_k - 1");
    const uint32_t r = 2 * d_k - 1;
    // (slot-19 WP-LV2) THE FREE PART AT FINITE PRECISION.  A := H^1(Gamma; Sym^n Z/ell^{K'}) =
    // (H/tors)/ell^{K'} (+) tors(H^1)/ell^{K'} (+) H^2[ell^{K'}] (universal coefficients, split over Z_ell), so for
    // t >= every torsion exponent, ell^t A = (H/tors)/ell^{K' - t}, Hecke-equivariantly and canonically: in the
    // module basis, the exponent-K' indices, every operator's submatrix on them read mod ell^{K' - t}.  Accepted
    // when exactly r = rank H summands have exponent K' and K' - t >= K; else K' is raised.
    uint32_t K_run = K + opt.min_extra_precision;
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    std::unique_ptr<LocalSystemSimplicialComplexT<uint64_t>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<uint64_t>> cohp;
    std::vector<uint32_t> F;
    for (;;) {
        (void)PowFit(ell, K_run);
        cohp.reset();
        lssc.reset();
        ring.reset();
        ring.reset(new hecke::ScopedCoefficientRing(K_run));
        const engine::ModCtx<uint64_t> ctx = hecke::CoefficientRingContext<uint64_t>("HeckeLValue", ell, K_run);
        lssc.reset(new LocalSystemSimplicialComplexT<uint64_t>(*Kx, hecke::GenerateSL2ZRelationsRing<uint64_t>(*Kx, n, ctx), n + 1, ctx.p));
        cohp.reset(new EquivariantCohomologyOnKT<uint64_t>(lssc.get()));
        cohp->compute_cohomology();
        c.exps = cohp->module_degree(1).dec.exponents;
        F.clear();
        uint32_t t = 0;
        for (uint32_t i = 0; i < c.exps.size(); ++i) {
            if (c.exps[i] == K_run) F.push_back(i);
            else t = std::max(t, c.exps[i]);
        }
        const bool ok = F.size() == r && K_run >= K + t;
        if (ok && t == 0) break;
        if (!opt.torsion_aware)
            throw hecke::NotImplementedInThisBuild(
                "HeckeLValue: H^1(Gamma; Sym^" + std::to_string(n) + " Z_" + std::to_string(ell) +
                ") has ell-torsion; [E] def:lveis reads H modulo torsion -- the torsion-aware reading is opt-in "
                "(EisensteinOptions::torsion_aware)");
        if (ok) {
            c.torsion_t = t;
            break;
        }
        ENGINE_CHECK_MSG(K_run < K + opt.max_extra_precision,
                         "HeckeLValue: the free part was not isolated within K + max_extra_precision (torsion exponent too large)");
        K_run = std::max(K_run + 1, K + t);
    }
    c.K_run = K_run;
    c.torsion_summands = static_cast<uint32_t>(c.exps.size()) - r;
    c.rank = r;
    const uint64_t q_run = PowFit(ell, K_run);
    EquivariantCohomologyOnKT<uint64_t>& coh = *cohp;

    for (uint64_t p : c.primes) {
        std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(p, 1, ell, n));
            tau.reset(new hecke::HeckeTauTableT<uint64_t>(*Kx, *model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnKT<uint64_t> opk(&coh, cosets.get(), tau.get());
        opk.compute();
        const engine::DenseBuf<uint64_t>& A = opk.H_matrix_module(1);
        ENGINE_CHECK(A.rows == c.exps.size() && A.cols == c.exps.size());
        FreeMat T = Zero(ell, K, r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) T.a[static_cast<size_t>(i) * r + j] = A.at(F[i], F[j]) % q_run % q;
        c.T.push_back(T);
    }

    // (b) iota^* on H_K: (iota^* f)(sigma_s) = -eps_s rho_n(iota) f(iota_K sigma_s), rho_n(iota) = diag((-1)^{n-j})
    {
        const hecke::IotaSimplicialMap im = hecke::BuildIotaSimplicialMapOnModel(*Kx, *model);
        const auto& H = coh.module_degree(1).H;
        const uint32_t dV = n + 1;
        ENGINE_CHECK_MSG(im.num_degrees() >= 2, "HeckeLValue: iota_K has no degree 1");
        const uint32_t N1 = static_cast<uint32_t>(im.image_index[1].size());
        const uint32_t rt = static_cast<uint32_t>(c.exps.size());
        ENGINE_CHECK_MSG(H.rows == N1 * dV && H.cols == rt, "HeckeLValue: the module basis is not in flat 1-cochain coordinates");
        engine::DenseBuf<uint64_t> Y(H.rows, rt);
        for (uint32_t s = 0; s < N1; ++s) {
            const uint32_t t = im.image_index[1][s];
            const int eps = im.image_sign[1][s];
            for (uint32_t i = 0; i < dV; ++i) {
                const bool rho_minus = ((n - i) % 2u) == 1u;
                const bool neg = (eps > 0) != rho_minus;   // sign = -eps * (-1)^{n-i}
                for (uint32_t j = 0; j < rt; ++j) {
                    const uint64_t x = H.at(t * dV + i, j) % q_run;
                    Y.at(s * dV + i, j) = neg ? (q_run - x) % q_run : x;
                }
            }
        }
        const std::vector<std::vector<uint64_t>> co = coh.H_class_coordinates(1, engine::cview(Y));
        c.iota = Zero(ell, K, r);
        for (uint32_t j = 0; j < r; ++j)
            for (uint32_t i = 0; i < r; ++i) c.iota.a[static_cast<size_t>(i) * r + j] = co[F[j]][F[i]] % q_run % q;
    }
    std::vector<uint64_t> e_values;
    for (uint64_t p : c.primes) e_values.push_back((1 + PowModU64(p, k - 1, q)) % q);
    FinishEisensteinCell(c, e_values, d_k, d_k - 1);
    c.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return c;
}

// ---- (e)-(g) -------------------------------------------------------------------
SignReading ReadSign(const EisensteinCell& cell, Sign sign, uint32_t m, const std::vector<uint64_t>& character) {
    ENGINE_CHECK_MSG(m >= 1 && m <= cell.K, "HeckeLValue: 1 <= m <= K");
    const SignBlock& B = sign == Sign::Plus ? cell.plus : cell.minus;
    const uint64_t ell = cell.ell;
    const uint32_t K = cell.K, f = B.f;
    const uint64_t q = PowFit(ell, K), qm = PowFit(ell, m);
    const std::vector<uint64_t> e = character.empty() ? cell.e_values : character;
    ENGINE_CHECK_MSG(e.size() == cell.primes.size(), "HeckeLValue: the character must be aligned with S");
    SignReading R;
    R.sign = sign;
    R.m = m;
    R.Q.character.name = character.empty() ? "Eisenstein 1+p^{k-1}" : "override";
    R.Q.character.primes = cell.primes;
    R.Q.character.values = e;
    R.Q.character.precision = K;
    R.Q.m = m;
    R.Q.block_rank = f;
    R.Q.ann_term_zero = (sign == Sign::Minus);
    for (size_t i = 0; i < cell.primes.size(); ++i)
        if (cell.primes[i] != ell) R.charpoly_primes.push_back(cell.primes[i]);
    if (f == 0) {
        R.Q.saturated = true;   // rank 0: T(H') = 0
        R.charpoly_valuations.assign(R.charpoly_primes.size(), 0);   // c_{k,p} = 1
        return R;
    }
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, K);
    const uint32_t N = f * f;
    const std::vector<Vec> gens = AlgebraGenerators(B.ops, ell, K, f);
    R.Q.dim_A1 = Span(ctx, Scaled(gens, PowFit(ell, K - 1), q), N).second;
    R.Q.saturated = (R.Q.dim_A1 == f);
    if (R.Q.saturated) {
        R.Q.c_sat = 0;
    } else {
        // End_K / T_K over Z/ell^K: f^2 - f divisors K (a complement of rank f^2 - f) and T's elementary divisors in End
        std::vector<Vec> E;
        for (uint32_t t = 0; t < N; ++t) {
            Vec u(N, 0);
            u[t] = 1;
            E.push_back(u);
        }
        const engine::DenseBuf<uint64_t> G = ColsBuf(E, N), Rl = ColsBuf(gens, N);
        const std::vector<uint32_t> inv = engine::elim::invariant_factors(ctx, engine::cview(G), engine::cview(Rl));
        uint32_t nK = 0, mx = 0;
        for (uint32_t d : inv) {
            if (d >= K) ++nK;
            else mx = std::max(mx, d);
        }
        R.Q.c_sat = nK == N - f ? static_cast<int32_t>(mx) : -1;
    }
    const uint64_t sc = PowFit(ell, K - m);
    const std::vector<Vec> A = Scaled(gens, sc, q);
    std::vector<Vec> J;
    for (size_t i = 0; i < cell.primes.size(); ++i) {
        const FreeMat X = MinusScalar(B.ops[i], e[i]);
        for (const Vec& g : gens) J.push_back(Flatten(Mul(X, Unflatten(g, ell, K, f))));
    }
    J = Scaled(J, sc, q);
    R.Q.length_A = Span(ctx, A, N).second;
    R.Q.length_I = Span(ctx, J, N).second;
    ENGINE_CHECK(R.Q.length_I <= R.Q.length_A);
    R.Q.length = R.Q.length_A - R.Q.length_I;
    if (R.Q.saturated)
        ENGINE_CHECK_MSG(R.Q.length_A == m * f, "HeckeLValue: a saturated algebra must read A_m = T / ell^m, free of rank f");
    if (R.Q.length > 0) {
        const engine::DenseBuf<uint64_t> G = ColsBuf(A, N), Rl = ColsBuf(J, N);
        R.Q.elementary_divisors = engine::elim::invariant_factors(ctx, engine::cview(G), engine::cview(Rl));
        uint32_t sum = 0;
        for (uint32_t d : R.Q.elementary_divisors) sum += d;
        ENGINE_CHECK_MSG(sum == R.Q.length, "HeckeLValue: the elementary divisors of Q do not sum to its length");
    }
    // (g) v(c_{k,p}(e_p)) mod ell^m
    for (size_t i = 0; i < cell.primes.size(); ++i) {
        if (cell.primes[i] == ell) continue;
        const hida::Poly chi = hida::CharpolyFree(ToRingMat(ReduceTo(B.ops[i], m)));
        uint64_t val = 0;
        for (size_t d = chi.size(); d-- > 0;)
            val = static_cast<uint64_t>((static_cast<u128>(val) * (e[i] % qm) + chi[d]) % qm);
        R.charpoly_valuations.push_back(val == 0 ? m : std::min<uint32_t>(m, ValuationU64(val, ell)));
    }
    return R;
}

ModuleFormReading ReadModuleForm(const EisensteinCell& cell) {
    ModuleFormReading R;
    const SignBlock& B = cell.plus;
    R.f_plus = B.f;
    if (B.f == 1) {
        R.applicable = true;
        R.c = 0;
        return R;
    }
    if (B.f != 2) return R;
    const uint64_t ell = cell.ell;
    const uint64_t q = PowFit(ell, cell.K);
    for (size_t i = 0; i < cell.primes.size(); ++i) {
        const uint64_t p = cell.primes[i];
        if (p == ell) continue;
        const FreeMat& T = B.ops[i];
        const uint64_t e = cell.e_values[i];
        const uint64_t t = (T.at(0, 0) + T.at(1, 1) + 2 * (q - e)) % q;   // a_p - e_p
        if (t == 0) continue;
        const FreeMat X = MinusScalar(T, e);
        uint32_t w = cell.K;
        for (uint64_t x : X.a)
            if (x != 0) w = std::min(w, ValuationU64(x, ell));
        const uint32_t vt = ValuationU64(t, ell);
        ENGINE_CHECK_MSG(w <= vt, "HeckeLValue: X_p = (a_p - e_p) pi with pi a projector forces v(X_p) <= v(a_p - e_p)");
        R.primes.push_back(p);
        R.c_per_p.push_back(vt - w);
    }
    if (R.primes.empty()) return R;
    R.applicable = true;
    R.c = R.c_per_p[0];
    for (uint32_t c : R.c_per_p) ENGINE_CHECK_MSG(c == R.c, "HeckeLValue: the lattice index read at two primes differs");
    return R;
}

uint32_t PlantedDimA1(const std::vector<FreeMat>& ops) {
    ENGINE_CHECK_MSG(!ops.empty(), "HeckeLValue: PlantedDimA1 needs at least one operator");
    return DimA1(ops, ops[0].ell, ops[0].K, ops[0].r);
}

// ---- the route -------------------------------------------------------------------
const char* const kCharacterModelCohomology = "H^1(Gamma_0(N);V(chi))";

void EisensteinBernoulliRoute::RequireSupportedDomain(const Domain& d) {
    const bool level_one = d.model == ModelId::SL2ZSevenVertex && d.level == 1 && d.cohomology == "H^1";
    // (slot-19 WP-LV2) the twisted instance cl:lvtwist: the character model at level N = ell (chi = omega^j has conductor ell)
    const bool twisted = d.model == ModelId::CharacterModel && d.ell >= 5 && d.level == d.ell && d.cohomology == kCharacterModelCohomology;
    if (!level_one && !twisted)
        throw hecke::NotImplementedInThisBuild(std::string("EisensteinBernoulliRoute: domain ") + d.to_string() +
                                               " -- implemented: the level-one H^1 on the 7-vertex model (cl:lveis) and the character "
                                               "model H^1(Gamma_0(ell); V(omega^j)) (cl:lvtwist); no other model or level");
}

EisensteinBernoulliRoute::EisensteinBernoulliRoute(std::shared_ptr<const EisensteinCell> cell, uint32_t m)
    : cell_(std::move(cell)), m_(m) {
    ENGINE_CHECK_MSG(cell_ != nullptr, "EisensteinBernoulliRoute: no cell");
    ENGINE_CHECK_MSG(m_ >= 1 && m_ <= cell_->K, "EisensteinBernoulliRoute: 1 <= m <= K");
    const bool twisted = cell_->level != 1;
    d_.model = twisted ? ModelId::CharacterModel : ModelId::SL2ZSevenVertex;
    d_.cohomology = twisted ? kCharacterModelCohomology : "H^1";
    d_.n_sym = cell_->n;
    d_.level = cell_->level;
    d_.ell = cell_->ell;
    d_.m = m_;
    d_.subquotient = cell_->torsion_t ? "H^-_{m_Eis} of H/tors" : "H^-_{m_Eis}";
    RequireSupportedDomain(d_);
    const uint32_t k = cell_->k();
    if (twisted)
        bern_ = TwistedBernoulliOverK(k, cell_->ell, cell_->twist_j, m_);
    else
        bern_ = k <= 20000 ? BernoulliModEllPowerExact(k, cell_->ell, m_) : BernoulliModEllPowerVoronoi(k, cell_->ell, m_);
}

NormalisationRecord EisensteinBernoulliRoute::normalisation() const {
    NormalisationRecord N;
    const uint32_t k = cell_->k();
    N.s0 = "1-k=" + std::to_string(1 - static_cast<int64_t>(k));
    N.sign_at_infinity = '-';
    N.period = "none";
    N.ambiguity = "Z_ell^x";
    if (cell_->level != 1) {
        // [E] sec:lvnorm row (E_chi): L(1-k, chi) = -B_{k,chi}/k, chi = omega^j of conductor ell, chi(ell) = 0
        N.l_function = "L(s,chi)";
        N.twist = "chi=omega^" + std::to_string(cell_->twist_j) + ",conductor=" + std::to_string(cell_->ell) +
                  ",omega(a)==a mod ell (Teichmuller)";
        N.constants.push_back({"L(1-k,chi)", "-B_{k,chi}/k", static_cast<int32_t>(bern_.v), bern_.v_exact});
        return N;
    }
    N.l_function = "zeta";
    N.twist = "chi=1";
    N.euler_factors_removed.push_back({"Euler factor at ell", "1-ell^{k-1}", 0, true});
    N.constants.push_back({"zeta(1-k)", "-B_k/k", static_cast<int32_t>(bern_.v), bern_.v_exact});
    return N;
}

const SignReading& EisensteinBernoulliRoute::reading() const {
    if (!reading_) reading_.reset(new SignReading(ReadSign(*cell_, Sign::Minus, m_)));
    return *reading_;
}

CongruenceModule EisensteinBernoulliRoute::congruence_module() const { return reading().Q; }

IndexStatement EisensteinBernoulliRoute::index_statement() const {
    const SignReading& R = reading();
    IndexStatement I;
    const bool twisted = cell_->level != 1;
    I.label = twisted ? "[E] cl:lvtwist (finite precision, as cl:lveis(3))" : "[E] cl:lveis(3)";
    if (R.Q.saturated)
        I.status = "PROVED-from-(1)(saturated)";
    else if (R.Q.c_sat >= 0)
        I.status = "two-sided-bound(c_sat=" + std::to_string(R.Q.c_sat) + ")";
    else
        I.status = "upper-bound-only(c_sat>=K)";
    I.module_name = "Q_m=A_m/(T_p-e_p:p in S)";
    I.l_value = twisted ? "L(1-k,chi)=-B_{k,chi}/k" : "zeta(1-k)=-B_k/k";
    I.l_valuation = bern_.v;
    I.l_valuation_exact = bern_.v_exact;
    I.expected_hi = std::min(m_, bern_.v);
    if (R.Q.saturated)
        I.expected_lo = I.expected_hi;
    else if (R.Q.c_sat >= 0 && m_ > static_cast<uint32_t>(R.Q.c_sat))
        I.expected_lo = std::min(m_ - static_cast<uint32_t>(R.Q.c_sat), bern_.v);
    else
        I.expected_lo = 0;
    I.found = R.Q.length;
    const bool cyclic = R.Q.elementary_divisors.size() <= 1;
    I.holds = I.expected_lo <= I.found && I.found <= I.expected_hi && (!R.Q.saturated || cyclic);
    return I;
}

std::string LValueLine(const EisensteinBernoulliRoute& r) {
    const SignReading& R = r.reading();
    const IndexStatement I = r.index_statement();
    std::ostringstream os;
    os << "LVALUE route=" << r.name() << " kind=" << ToString(r.kind()) << " n=" << r.domain().n_sym << " ell=" << r.domain().ell
       << " k=" << r.domain().n_sym + 2;
    if (r.domain().level != 1) os << " level=" << r.domain().level << " chi=omega^" << r.cell().twist_j;
    os << " m=" << r.domain().m << " H'=" << r.domain().subquotient;
    if (r.cell().torsion_t) os << " K_run=" << r.cell().K_run << " torsion_t=" << r.cell().torsion_t;
    os << " f=" << R.Q.block_rank << " dim_A1=" << R.Q.dim_A1 << " saturated=" << (R.Q.saturated ? "yes" : "no")
       << " c_sat=" << R.Q.c_sat << " Q=[";
    for (size_t i = 0; i < R.Q.elementary_divisors.size(); ++i) os << (i ? "," : "") << R.Q.elementary_divisors[i];
    os << "] length=" << R.Q.length << " v(B_k/k)=" << r.bernoulli().v << (r.bernoulli().v_exact ? "" : "+")
       << " expected=[" << I.expected_lo << "," << I.expected_hi << "] holds=" << (I.holds ? "yes" : "NO") << " cp_v=[";
    for (size_t i = 0; i < R.charpoly_valuations.size(); ++i)
        os << (i ? "," : "") << R.charpoly_primes[i] << ":" << R.charpoly_valuations[i];
    os << "] N={" << r.normalisation().to_string() << "}";
    return os.str();
}

} // namespace lvalue
} // namespace hecke_ext
