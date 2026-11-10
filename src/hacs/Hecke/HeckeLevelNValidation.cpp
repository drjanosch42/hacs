// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 14, WP-12) HeckeLevelNValidation -- see the header.
// [C] v1.14.0 sec:levelN: def:cosetrepsN, lem:cosetrepsN, def:levelNmodel,
// rmk:levelNcheckable; sec:levelshapiro: prop:levelcorrespondence, prop:levelorbit.
#include "HeckeLevelNValidation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>

#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"   // slot-15 WP-17: the dense presentation route of the (R3x) cross-check
#include "HeckeLevelNR3.h"   // slot-15 WP-9: the (R3) scan reader on the sparse route
#include "HeckeLevelNSparse.h"   // slot-15 WP-17: WP-8's ReadEisensteinR3, SparseCokernel
#include "HeckeOperatorFamily.h"  // hecke::tau_construction_mutex() (kept: review M-4 called it unused, it is not)
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "InducedLocalSystem.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "ValidationSupport.h"

namespace {

using engine::scalar::gcd;
using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::level;
using u128 = unsigned __int128;

constexpr uint32_t kP = 1000003;   // the characteristic-0 surrogate field F_P

std::map<std::string, double>& PhaseTimes() { static std::map<std::string, double> t; return t; }
struct PhaseTimer {
    std::string name;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit PhaseTimer(std::string n) : name(std::move(n)) {}
    ~PhaseTimer() { PhaseTimes()[name] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

struct Checker {
    uint32_t checks = 0, fails = 0;
    std::map<std::string, uint32_t> by_arm;
    void check(bool ok, const std::string& what) {
        ++checks;
        by_arm[what.substr(0, what.find(' '))] += 1;
        if (!ok) {
            ++fails;
            if (fails <= 80) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

uint64_t Mod(int64_t x, uint64_t m) { const int64_t r = x % static_cast<int64_t>(m); return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(m) : r); }

std::string RingText(const engine::ModCtx<uint32_t>& ctx) {
    if (ctx.is_field()) return "F_" + std::to_string(ctx.p);
    return "Z/" + std::to_string(ctx.ell) + "^" + std::to_string(ctx.k);
}

// ---------------------------------------------------------------------------
// The model and the level-1 data.
// ---------------------------------------------------------------------------
struct Model {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
};
Model& TheModel() {
    static Model m;
    static std::once_flag once;
    std::call_once(once, [] {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        m.K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        m.tess.reset(new hecke::SL2ZTessellationTables(*m.K));
    });
    return m;
}

using hecke::vsupport::HeckeDatum;
HeckeDatum MakeDatum(uint64_t p, uint32_t coef, uint32_t n) {
    Model& M = TheModel();
    return hecke::vsupport::MakeHeckeDatum(*M.K, *M.tess, p, 1, coef, n);
}

// ---------------------------------------------------------------------------
// Field linear algebra on HMat (validation-local).
// ---------------------------------------------------------------------------
HMat Pow(const HMat& A, uint64_t e) {
    HMat R = HIdentity(A), B = A;
    while (e) {
        if (e & 1) R = HMul(R, B);
        B = HMul(B, B);
        e >>= 1;
    }
    return R;
}
HMat MakeField(uint32_t rows, uint32_t cols, uint64_t p) {
    HMat M;
    M.rows = rows;
    M.cols = cols;
    M.modulus = p;
    M.ell = p;
    M.field = true;
    M.row_exp.assign(rows, 1);
    M.a.assign(static_cast<size_t>(rows) * cols, 0);
    return M;
}
HMat SubScalar(const HMat& A, uint64_t s) { return HSub(A, HScale(HIdentity(A), s)); }
HMat StackRows(const std::vector<HMat>& Ms) {
    uint32_t r = 0;
    for (const HMat& M : Ms) r += M.rows;
    HMat S = MakeField(r, Ms[0].cols, Ms[0].modulus);
    uint32_t off = 0;
    for (const HMat& M : Ms) {
        for (uint32_t i = 0; i < M.rows; ++i)
            for (uint32_t j = 0; j < M.cols; ++j) S.a[static_cast<size_t>(off + i) * S.cols + j] = M.at(i, j);
        off += M.rows;
    }
    return S;
}
// Kernel basis as the columns of a (cols x k) matrix.
HMat FieldKernel(const HMat& A) {
    const uint64_t p = A.modulus;
    const uint32_t R = A.rows, C = A.cols;
    std::vector<std::vector<uint64_t>> M(R, std::vector<uint64_t>(C));
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < C; ++j) M[i][j] = A.at(i, j) % p;
    std::vector<int> row_of_col(C, -1);
    uint32_t rk = 0;
    for (uint32_t c = 0; c < C && rk < R; ++c) {
        uint32_t piv = rk;
        while (piv < R && M[piv][c] == 0) ++piv;
        if (piv == R) continue;
        std::swap(M[piv], M[rk]);
        const uint64_t iv = powmod<uint64_t>(M[rk][c], p - 2, p);
        for (uint64_t& x : M[rk]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % p);
        for (uint32_t i = 0; i < R; ++i) {
            if (i == rk || M[i][c] == 0) continue;
            const uint64_t f = M[i][c];
            for (uint32_t j = 0; j < C; ++j) M[i][j] = (M[i][j] + p - static_cast<uint64_t>(static_cast<u128>(f) * M[rk][j] % p)) % p;
        }
        row_of_col[c] = static_cast<int>(rk);
        ++rk;
    }
    std::vector<std::vector<uint64_t>> ker;
    for (uint32_t f = 0; f < C; ++f) {
        if (row_of_col[f] >= 0) continue;
        std::vector<uint64_t> v(C, 0);
        v[f] = 1;
        for (uint32_t c = 0; c < C; ++c)
            if (row_of_col[c] >= 0) v[c] = (p - M[row_of_col[c]][f]) % p;
        ker.push_back(v);
    }
    HMat K = MakeField(C, static_cast<uint32_t>(ker.size()), p);
    for (uint32_t j = 0; j < K.cols; ++j)
        for (uint32_t i = 0; i < C; ++i) K.a[static_cast<size_t>(i) * K.cols + j] = ker[j][i];
    return K;
}
uint32_t ColRank(const HMat& A) { return A.cols == 0 || A.rows == 0 ? 0 : HRankModEll(A); }
HMat ConcatCols(const std::vector<HMat>& Ms) {
    uint32_t c = 0;
    for (const HMat& M : Ms) c += M.cols;
    HMat S = MakeField(Ms[0].rows, c, Ms[0].modulus);
    uint32_t off = 0;
    for (const HMat& M : Ms) {
        for (uint32_t i = 0; i < M.rows; ++i)
            for (uint32_t j = 0; j < M.cols; ++j) S.a[static_cast<size_t>(i) * S.cols + off + j] = M.at(i, j);
        off += M.cols;
    }
    return S;
}

using Poly = std::vector<uint64_t>;   // low -> high
Poly PolyMul(const Poly& a, const Poly& b, uint64_t p) {
    Poly c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = (c[i + j] + static_cast<uint64_t>(static_cast<u128>(a[i]) * b[j] % p)) % p;
    return c;
}
// prod over factors (coefficients low -> high, signed) of factor^mult.
Poly FromFactors(const std::vector<std::pair<std::vector<int64_t>, uint32_t>>& fs, uint64_t p) {
    Poly out{1};
    for (const auto& f : fs) {
        Poly q;
        for (int64_t c : f.first) q.push_back(Mod(c, p));
        for (uint32_t k = 0; k < f.second; ++k) out = PolyMul(out, q, p);
    }
    return out;
}
std::string PolyText(const Poly& f) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < f.size(); ++i) os << (i ? "," : "") << f[i];
    os << "]";
    return os.str();
}

// ---------------------------------------------------------------------------
// (T) the composite-N table.
// ---------------------------------------------------------------------------
void ArmTable(Checker& g) {
    PhaseTimer pt("(T) table");
    std::vector<uint64_t> Ns;
    for (uint64_t N = 1; N <= 60; ++N) Ns.push_back(N);
    for (uint64_t N : {77ull, 91ull, 105ull, 120ull, 180ull, 181ull, 210ull, 330ull, 1001ull, 1321ull, 2310ull}) Ns.push_back(N);
    uint32_t mixed_total = 0;
    const std::vector<Mat22> probes{Mat22::T(), Mat22::S(), Mat22::ST(), Mat22::TS(), Mat22{2, 1, 1, 1}, Mat22{5, 2, 7, 3},
                                    Mat22{-3, 7, 2, -5}, Mat22{13, 5, 18, 7}};
    for (uint64_t N : Ns) {
        const Gamma0LevelN L(N);
        const std::string tag = "(T) N=" + std::to_string(N);
        const uint32_t m = L.m();
        g.check(m == IndexGamma0(N), tag + ": m_N = N prod (1 + 1/q) (lem:cosetrepsN (1))");
        // (2): reps in Gamma, bottom row of class j, w_1 = I, case (iii) only if r >= 2.
        bool reps = L.rep(1) == Mat22::I(), cases_ok = true;
        uint32_t mixed = 0, nfix_S = 0, nfix_ST = 0;
        for (uint32_t j = 1; j <= m; ++j) {
            const Mat22 w = L.rep(j);
            reps = reps && w.det() == 1 && L.class_of(w) == j;
            const auto pr = L.canonical_pair(j);
            reps = reps && pr.first < std::max<uint64_t>(N, 1) && pr.second < std::max<uint64_t>(N, 2);
            if (L.rep_case(j) == 3) {
                ++mixed;
                cases_ok = cases_ok && L.factors().size() >= 2 && gcd<uint64_t>(pr.first, N) > 1 && gcd<uint64_t>(pr.second, N) > 1;
            }
            nfix_S += L.act(j, Mat22::S()) == j ? 1 : 0;
            nfix_ST += L.act(j, Mat22::ST()) == j ? 1 : 0;
        }
        mixed_total += mixed;
        g.check(reps, tag + ": every w_j in SL_2(Z), bottom row of class j, w_1 = I (lem:cosetrepsN (2))");
        g.check(cases_ok, tag + ": case (iii) exactly at mixed classes (" + std::to_string(mixed) + ")");
        g.check(nfix_S == Nu2(N) && nfix_ST == Nu3(N),
                tag + ": fixed points of S, ST on the table = nu_2, nu_3 (rmk:levelcost (3)): " + std::to_string(nfix_S) +
                    ", " + std::to_string(nfix_ST));
        // independent P^1(Z/N) test (N <= 60): same class iff c d' - c' d = 0 mod N; count m phi(N).
        if (N <= 60 && N >= 2) {
            bool indep = true;
            uint64_t prim = 0, phi = 0;
            for (uint64_t u = 1; u <= N; ++u) phi += gcd<uint64_t>(u, N) == 1 ? 1 : 0;
            for (uint64_t c = 0; c < N && indep; ++c)
                for (uint64_t d = 0; d < N; ++d) {
                    if (gcd<uint64_t>(gcd<uint64_t>(c, d), N) != 1) continue;
                    ++prim;
                    const uint32_t j = L.index_of_pair(static_cast<int64_t>(c), static_cast<int64_t>(d));
                    const auto pr = L.canonical_pair(j);
                    const uint64_t cross = Mod(static_cast<int64_t>((c * pr.second) % N) - static_cast<int64_t>((d * pr.first) % N), N);
                    if (cross != 0) { indep = false; break; }
                }
            g.check(indep && prim == static_cast<uint64_t>(m) * phi,
                    tag + ": index_of_pair is the P^1(Z/N) class (cross-product test, " + std::to_string(prim) + " pairs)");
        }
        // (4): right action, digitwise, h_j(gamma) in Gamma_0(N); integral delta of det p, (p, N) = 1.
        bool act_ok = true, digit_ok = true, coc_ok = true;
        for (uint32_t j = 1; j <= m; j += (m > 400 ? 7 : 1))
            for (size_t a = 0; a < probes.size(); ++a) {
                const Mat22& x = probes[a];
                const Mat22& y = probes[(a + 3) % probes.size()];
                act_ok = act_ok && L.act(L.act(j, x), y) == L.act(j, x * y) && L.act(j, x) == L.class_of(L.rep(j) * x);
                const std::vector<uint32_t> dj = L.digits(j), dk = L.digits(L.act(j, x));
                for (size_t i = 0; i < dj.size(); ++i) digit_ok = digit_ok && L.digit_act(i, dj[i], x) == dk[i];
                const Mat22 h = L.cocycle(j, x);
                coc_ok = coc_ok && L.contains(h);
            }
        for (int64_t p : {2, 3, 5, 7}) {
            if (N % static_cast<uint64_t>(p) == 0) continue;
            for (const Mat22& dl : {Mat22{1, 0, 0, p}, Mat22{p, 0, 0, 1}, Mat22{1, 1, 0, p}, Mat22{p, 1, 0, 1}})
                for (uint32_t j = 1; j <= m; j += (m > 400 ? 11 : 1)) {
                    if (!L.act_defined(j, dl)) { coc_ok = false; continue; }
                    const Mat22 h = L.cocycle(j, dl);
                    coc_ok = coc_ok && h.det() == p && Mod(h.c, N) == 0;
                }
        }
        // p | N (review M-3): delta = diag(1, p), the U_p admissibility -- (c_j, d_j) delta primitive iff p !| c_j; where
        // defined, h_j(delta) is integral of det p with lower left = 0 mod N (the datum prop:heilbronn's H_{N,p} reads).
        bool bad_ok = true, bad_any = false;
        for (int64_t p : {2, 3, 5, 7}) {
            if (N % static_cast<uint64_t>(p) != 0) continue;
            bad_any = true;
            const Mat22 dl{1, 0, 0, p};
            for (uint32_t j = 1; j <= m; j += (m > 400 ? 11 : 1)) {
                const bool adm = L.canonical_pair(j).first % static_cast<uint64_t>(p) != 0;
                if (L.act_defined(j, dl) != adm) { bad_ok = false; continue; }
                if (!adm) continue;
                const Mat22 h = L.cocycle(j, dl);
                bad_ok = bad_ok && h.det() == p && Mod(h.c, N) == 0;
            }
        }
        if (bad_any)
            g.check(bad_ok, tag + ": p | N, delta = diag(1, p): j . delta defined iff p !| c_j; h_j(delta) integral of det p, "
                                  "lower left = 0 mod N (U_p admissibility, prop:heilbronn)");
        g.check(act_ok, tag + ": j . gamma is a right action, = cls(w_j gamma) (lem:cosetrepsN (4))");
        g.check(digit_ok, tag + ": the action is digitwise (lem:cosetrepsN (4))");
        g.check(coc_ok, tag + ": h_j(gamma) in Gamma_0(N); h_j(delta) integral of det p, lower left = 0 mod N (lem:cosetrepsN (4))");
        // (5): Gamma_0(N p) \ Gamma_0(N).
        if (N <= 400)
            for (uint64_t p : {2ull, 3ull, 5ull}) {
                const Gamma0LevelN Lp(N * p);
                const std::vector<Mat22> r = Gamma0SubTransversalN(L, Lp);
                bool ok = r.size() == (N % p ? p + 1 : p) && r[0] == Mat22::I();
                for (size_t t = 0; t < r.size() && ok; ++t) {
                    ok = ok && L.contains(r[t]);
                    for (size_t u = t + 1; u < r.size() && ok; ++u) ok = ok && !Lp.contains(r[t] * r[u].inv_unimodular());
                }
                g.check(ok, tag + ": def:cosetrepsN (6) at N' = " + std::to_string(N * p) + ": " + std::to_string(r.size()) +
                                " representatives of Gamma_0(N')\\Gamma_0(N) in Gamma_0(N), I first, pairwise distinct (lem:cosetrepsN (5))");
            }
    }
    // (3): prime powers = the shipped table (def:cosetreps).
    for (const auto& pq : std::vector<std::pair<uint64_t, uint32_t>>{{2, 1}, {3, 1}, {5, 1}, {7, 1}, {2, 2}, {2, 3}, {3, 2},
                                                                     {5, 2}, {2, 5}, {3, 3}, {7, 2}, {11, 1}, {13, 1}}) {
        uint64_t N = 1;
        for (uint32_t i = 0; i < pq.second; ++i) N *= pq.first;
        const hecke::HeckeCosetTables C(pq.first, pq.second, kP, 0);
        const Gamma0LevelN L(N);
        bool eq = C.num_cosets() == L.m();
        for (uint32_t j = 1; eq && j <= L.m(); ++j) {
            eq = eq && C.s(j) == L.rep(j);
            for (const Mat22& x : {Mat22::T(), Mat22::S(), Mat22{5, 2, 7, 3}}) eq = eq && C.class_index_of(L.rep(j) * x) == L.act(j, x);
        }
        g.check(eq, "(T) N=" + std::to_string(N) + ": at a prime power the table IS the shipped def:cosetreps table (lem:cosetrepsN (3))");
    }
    g.check(mixed_total > 0, "(T) mixed classes (case (iii)) occur in the grid: " + std::to_string(mixed_total));
    // counts against known values: dim S_2(Gamma_0(N)) = genus, and a few higher weights.
    const std::vector<std::array<int64_t, 3>> known{{11, 2, 1}, {14, 2, 1}, {22, 2, 2}, {23, 2, 2}, {30, 2, 3}, {33, 2, 3}, {35, 2, 3},
                                                    {36, 2, 1}, {37, 2, 2}, {181, 2, 14}, {1, 12, 1}, {1, 24, 2}, {5, 4, 1}, {6, 4, 1},
                                                    {4, 6, 1}, {2, 8, 1}, {11, 4, 2}};
    bool dims = true;
    for (const auto& kv : known) dims = dims && DimCuspForms(static_cast<uint64_t>(kv[0]), static_cast<uint32_t>(kv[1])) == kv[2];
    g.check(dims, "(T) dim S_k(Gamma_0(N)) formula against 17 classical values");
    g.check(CuspCount(12) == 6 && CuspCount(11) == 2 && CuspCount(36) == 12 && CuspCount(30) == 8, "(T) cusp counts c(N)");
}

// ---------------------------------------------------------------------------
// One level-N cell on the model.
// ---------------------------------------------------------------------------
struct NCell {
    uint64_t N = 1;
    uint32_t n = 0;
    engine::ModCtx<uint32_t> ctx;
    uint32_t coef = 0;
    std::unique_ptr<Gamma0LevelN> lvl;
    std::map<uint32_t, HeckeDatum> Tp;
    std::map<uint32_t, std::unique_ptr<Gamma0LevelN>> lvl_Np;
    std::unique_ptr<LocalSystemSimplicialComplex> L;
    std::unique_ptr<EquivariantCohomologyOnK> H;
    double build_s = 0;
    std::string tag() const { return "(N=" + std::to_string(N) + ", n=" + std::to_string(n) + ") over " + RingText(ctx); }
};

std::unique_ptr<NCell> BuildNCell(uint64_t N, uint32_t n, const engine::ModCtx<uint32_t>& ctx, const std::vector<uint32_t>& primes) {
    const auto t0 = std::chrono::steady_clock::now();
    Model& M = TheModel();
    auto c = std::make_unique<NCell>();
    c->N = N;
    c->n = n;
    c->ctx = ctx;
    c->coef = ctx.is_field() ? ctx.p : static_cast<uint32_t>(ctx.ell);
    c->lvl.reset(new Gamma0LevelN(N));
    {
        PhaseTimer pt("build: level-1 coset + tau tables");
        for (uint32_t p : primes) {
            c->Tp[p] = MakeDatum(p, c->coef, n);
            c->lvl_Np[p].reset(new Gamma0LevelN(N * p));
        }
    }
    PhaseTimer pt("build: level-N local system + cohomology");
    const uint32_t D = c->lvl->m() * (n + 1);
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    if (!ctx.is_field()) ring.reset(new hecke::ScopedCoefficientRing(ctx.k));
    c->L.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsLevel(*M.K, *c->lvl, n, ctx), D, ctx.p));
    c->H.reset(new EquivariantCohomologyOnK(c->L.get()));
    c->H->compute_cohomology();
    c->build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return c;
}

// T_p^{(N)} (p !| N) / U_p^{(N)} (p | N) with the pinned r_t of def:cosetrepsN (6).
CorrespondenceSpec SpecHecke(const NCell& c, uint32_t p, bool pinned = true) {
    const HeckeDatum& d = c.Tp.at(p);
    const bool bad = c.N % p == 0;
    CorrespondenceSpec s{c.lvl.get(), c.lvl.get(), d.cosets.get(), d.tau.get(),
                         pinned ? Gamma0SubTransversalN(*c.lvl, *c.lvl_Np.at(p)) : Gamma0SubTransversal(c.N, p),
                         std::string(bad ? "U_" : "T_") + std::to_string(p) + "^(N)"};
    VerifyTransversal(s, bad ? p : p + 1);
    return s;
}

std::map<uint32_t, HMat> ReadOps(const NCell& c, const HFrame& f) {
    PhaseTimer pt("operators on H^1");
    std::map<uint32_t, HMat> ops;
    for (const auto& kv : c.Tp) {
        const CorrespondenceSpec s = SpecHecke(c, kv.first);
        ops[kv.first] = InducedOnH(f, f, BuildLevelCorrespondence(c.ctx, s, *TheModel().K, c.n, 1), s.label);
    }
    return ops;
}

// (D) (LN1), prop:levelorbit (3), H^2 = 0.
void ArmDims(Checker& g, const NCell& c) {
    const auto& dims = c.H->get_H_eq_dimensions();
    const int64_t h0 = dims[0], h1 = dims[1], h2 = dims[2];
    const std::string tag = "(D) " + c.tag();
    g.check(h2 == 0, tag + ": H^2(Gamma_0(N); V) = 0 (6 a unit)");
    if (c.n % 2) {
        g.check(h0 == 0 && h1 == 0, tag + ": odd n -- H^* = 0 (prop:levelorbit (2))");
        return;
    }
    const uint64_t N = c.N;
    const int64_t d = c.n + 1, m = static_cast<int64_t>(IndexGamma0(N)), nu2 = Nu2(N), nu3 = Nu3(N);
    // d_S, d_ST from rho_n(S), rho_n(ST) over the ring's field.
    const engine::ModCtx<uint32_t> fctx(c.ctx.is_field() ? c.ctx.p : static_cast<uint32_t>(c.ctx.ell));
    auto fixdim = [&](const Mat22& x) {
        const FullLevel full;
        const FixedSpace F = OrbitWiseFixedSpace(full, x, x == Mat22::S() ? 2 : 3, c.n, fctx);
        return static_cast<int64_t>(FieldRank(F.span, fctx.p));
    };
    const int64_t dS = fixdim(Mat22::S()), dST = fixdim(Mat22::ST());
    const int64_t orbit = m * d - ((m - nu2) / 2 * d + nu2 * dS) - ((m - nu3) / 3 * d + nu3 * dST);
    const int64_t ln1 = 2 * DimCuspForms(N, c.n + 2) + static_cast<int64_t>(CuspCount(N)) - 2 * (c.n == 0 ? 1 : 0);
    g.check(h1 - h0 == orbit, tag + ": prop:levelorbit (3) -- dim H^1 - dim H^0 = " + std::to_string(orbit) + " (model " +
                                   std::to_string(h1) + " - " + std::to_string(h0) + ")");
    g.check(h1 - h0 == ln1, tag + ": (LN1) dim H^1 - dim H^0 = 2 dim S_" + std::to_string(c.n + 2) + " + c(N) - 2 delta = " +
                                std::to_string(ln1));
    if (c.ctx.is_field() && c.ctx.p == kP)
        g.check(h0 == (c.n == 0 ? 1 : 0), tag + ": dim H^0 = delta_{n,0} in characteristic 0 (surrogate F_P)");
}

// (O) (LN2): orbit-wise = dense fixed spaces; the cokernel dims = the model's.
void ArmOrbit(Checker& g, const NCell& c) {
    if (c.n % 2) return;
    PhaseTimer pt("(O) orbit-wise fixed spaces");
    const std::string tag = "(O) " + c.tag();
    std::vector<FixedSpace> orbit;
    for (const auto& go : std::vector<std::pair<Mat22, uint32_t>>{{Mat22::S(), 2}, {Mat22::ST(), 3}}) {
        const FixedSpace fo = OrbitWiseFixedSpace(*c.lvl, go.first, go.second, c.n, c.ctx);
        const FixedSpace fd = DenseFixedSpace(*c.lvl, go.first, go.second, c.n, c.ctx);
        const MonomialAction mono = MonomialOf(*c.lvl, go.first, c.n, c.ctx);
        bool fixed = true;
        for (const auto& v : fo.span) fixed = fixed && mono.apply(c.ctx, v) == v;
        const std::string gname = go.second == 2 ? "S" : "ST";
        g.check(fixed, tag + ": every orbit-wise vector is fixed by the monomial M_N(" + gname + ")");
        g.check(fo.fixed_points == (go.second == 2 ? Nu2(c.N) : Nu3(c.N)) && fo.solves == fo.fixed_points,
                tag + ": " + gname + ": fixed points = nu, one d x d idempotent per fixed point, none on the " +
                    std::to_string(fo.free_orbits) + " free orbits");
        if (c.ctx.is_field()) {
            const uint32_t ro = FieldRank(fo.span, c.ctx.p), rd = FieldRank(fd.span, c.ctx.p);
            std::vector<std::vector<uint32_t>> all = fo.span;
            all.insert(all.end(), fd.span.begin(), fd.span.end());
            g.check(ro == rd && FieldRank(all, c.ctx.p) == ro,
                    tag + ": (LN2) M_N^<" + gname + "> orbit-wise == dense as subspaces (dim " + std::to_string(ro) + ")");
        }
        orbit.push_back(fo);
    }
    if (c.ctx.is_field()) {
        const CokernelDims cd = CokernelFromFixedSpaces(orbit[0], orbit[1], c.ctx.p);
        const auto& dims = c.H->get_H_eq_dimensions();
        g.check(cd.h1 == dims[1] && cd.h0 == dims[0],
                tag + ": prop:levelorbit (2) -- the orbit-wise cokernel M/Rel_N has dim " + std::to_string(cd.h1) +
                    " = dim H^1 of the model, H^0 = M^S cap M^ST (" + std::to_string(cd.h0) + ")");
    }
}

// (X) cochain maps into C_eq, commuting with d.
std::vector<uint32_t> Coboundary(const NCell& c, uint32_t q, uint32_t D, const std::vector<uint32_t>& X, uint32_t ncols) {
    const SimplicialComplex& K = *TheModel().K;
    const FlatMatrixT<uint32_t> D1 = K.coboundary_operator_elt<uint32_t>(q, c.ctx.p);
    const uint32_t N0 = K.get_num_q_simplices(q), N1 = K.get_num_q_simplices(q + 1);
    std::vector<uint32_t> Y(static_cast<size_t>(N1) * D * ncols, 0);
    for (uint32_t s1 = 0; s1 < N1; ++s1)
        for (uint32_t s0 = 0; s0 < N0; ++s0) {
            const uint32_t inc = D1.get_entry(s1, s0);
            if (!inc) continue;
            for (uint32_t a = 0; a < D; ++a)
                for (uint32_t col = 0; col < ncols; ++col) {
                    uint32_t& y = Y[(static_cast<size_t>(s1) * D + a) * ncols + col];
                    y = c.ctx.add(y, c.ctx.mul(inc, X[(static_cast<size_t>(s0) * D + a) * ncols + col]));
                }
        }
    return Y;
}
void ArmCochain(Checker& g, const NCell& c) {
    PhaseTimer pt("(X) cochain maps");
    const uint32_t D = c.lvl->m() * (c.n + 1);
    for (const auto& kv : c.Tp) {
        const CorrespondenceSpec s = SpecHecke(c, kv.first);
        for (uint32_t q = 0; q < 2; ++q) {
            const HFrame fq(c.H.get(), q);
            uint32_t cq = 0;
            const std::vector<uint32_t> C = fq.ceq_basis(cq);
            if (cq == 0) continue;
            const BlockCochainMap Fq = BuildLevelCorrespondence(c.ctx, s, *TheModel().K, c.n, q);
            const BlockCochainMap Fq1 = BuildLevelCorrespondence(c.ctx, s, *TheModel().K, c.n, q + 1);
            const std::vector<uint32_t> Y = Fq.apply(C, cq);
            bool in_ceq = true;
            for (const auto& rel : c.L->get_user_defined_simplex_relations()) {
                if (rel.q_dim != q || !in_ceq) continue;
                for (uint32_t a = 0; a < D && in_ceq; ++a) {
                    std::vector<std::pair<uint32_t, uint32_t>> row;
                    for (uint32_t b = 0; b < D; ++b) if (rel.M.get_entry(a, b)) row.emplace_back(b, rel.M.get_entry(a, b));
                    for (uint32_t col = 0; col < cq; ++col) {
                        uint32_t acc = Y[(static_cast<size_t>(rel.s_lhs_idx_in_Kq) * D + a) * cq + col];
                        for (const auto& e : row) acc = c.ctx.sub(acc, c.ctx.mul(e.second, Y[(static_cast<size_t>(rel.s_rhs_idx_in_Kq) * D + e.first) * cq + col]));
                        if (acc) { in_ceq = false; break; }
                    }
                }
            }
            const std::string tg = "(X) " + c.tag() + " " + s.label + " q=" + std::to_string(q);
            g.check(in_ceq, tg + ": maps C^q_eq(L_N) into C^q_eq (prop:levelcorrespondence (2))");
            g.check(Fq1.apply(Coboundary(c, q, D, C, cq), cq) == Coboundary(c, q, D, Y, cq), tg + ": F d = d F on C^q_eq");
        }
    }
}

// (H) the Hecke ring at level N on H^1; representative independence.
void ArmHeckeRing(Checker& g, const NCell& c, const HFrame& f, const std::map<uint32_t, HMat>& ops) {
    const std::string tag = "(H) " + c.tag();
    for (auto a = ops.begin(); a != ops.end(); ++a)
        for (auto b = std::next(a); b != ops.end(); ++b)
            g.check(HEqual(HMul(a->second, b->second), HMul(b->second, a->second)),
                    tag + ": [T_" + std::to_string(a->first) + ", T_" + std::to_string(b->first) + "] = 0 on H^1");
    for (const auto& kv : ops) {
        const CorrespondenceSpec s2 = SpecHecke(c, kv.first, false);
        const HMat other = InducedOnH(f, f, BuildLevelCorrespondence(c.ctx, s2, *TheModel().K, c.n, 1), s2.label + " (alt r_t)");
        g.check(HEqual(other, kv.second), tag + ": " + s2.label + " with Gamma0SubTransversal's r_t == with the pinned r_t of "
                                                 "def:cosetrepsN (6), on H^1 (prop:levelcorrespondence (3))");
    }
}

// ---------------------------------------------------------------------------
// (G) the independent route: group 1-cocycles of Gamma_0(N) over F_P, composite N.
// Own P^1(Z/N) table (lexicographically least unit multiple), own lifts, own
// transversals; no tau, no model, no def:cosetrepsN code.
// ---------------------------------------------------------------------------
class GroupRouteN {
public:
    GroupRouteN(uint32_t n, uint64_t N, uint64_t P) : P_(P), N_(N), n_(n), d_(n + 1) {
        std::vector<uint64_t> units;
        for (uint64_t u = 1; u <= N; ++u) if (gcd<uint64_t>(u, N) == 1) units.push_back(u % N);
        if (N == 1) units = {0};
        cls_.assign(static_cast<size_t>(N) * N, 0);
        auto norm = [&](uint64_t c, uint64_t d) {
            std::pair<uint64_t, uint64_t> best{N, N};
            for (uint64_t u : units) best = std::min(best, std::make_pair(u * c % N, u * d % N));
            if (N == 1) best = {0, 0};
            return best;
        };
        std::map<std::pair<uint64_t, uint64_t>, uint32_t> idx;
        auto add = [&](uint64_t c, uint64_t d) {
            const auto k = norm(c, d);
            if (idx.count(k)) return;
            idx[k] = static_cast<uint32_t>(reps_.size() + 1);
            // own lift: least s, t with gcd(c + sN, d + tN) = 1, then Bezout.
            int64_t cc = static_cast<int64_t>(c), dd = static_cast<int64_t>(d);
            bool done = false;
            for (int64_t s = 0; s < 64 && !done; ++s)
                for (int64_t t = 0; t < 64 && !done; ++t) {
                    const int64_t c2 = cc + s * static_cast<int64_t>(N), d2 = dd + t * static_cast<int64_t>(N);
                    if (gcd<uint64_t>(static_cast<uint64_t>(std::llabs(c2)), static_cast<uint64_t>(std::llabs(d2))) != 1) continue;
                    int64_t x0 = 1, y0 = 0, x1 = 0, y1 = 1, a = d2, b = c2;   // a x + b y = gcd
                    while (b) { const int64_t q = a / b; int64_t tmp = a - q * b; a = b; b = tmp; tmp = x0 - q * x1; x0 = x1; x1 = tmp; tmp = y0 - q * y1; y0 = y1; y1 = tmp; }
                    if (a < 0) { x0 = -x0; y0 = -y0; }
                    // x0 d2 + y0 c2 = 1 -> [[x0, -y0], [c2, d2]] has det x0 d2 + y0 c2 = 1
                    reps_.push_back(Mat22{x0, -y0, c2, d2});
                    done = true;
                }
            ENGINE_CHECK_MSG(done, "GroupRouteN: no lift");
        };
        add(0, 1 % std::max<uint64_t>(N, 1));
        for (uint64_t c = 0; c < N; ++c)
            for (uint64_t d = 0; d < N; ++d)
                if (gcd<uint64_t>(gcd<uint64_t>(c, d), N) == 1) add(c, d);
        for (uint64_t c = 0; c < N; ++c)
            for (uint64_t d = 0; d < N; ++d)
                if (gcd<uint64_t>(gcd<uint64_t>(c, d), N) == 1) cls_[c * N + d] = idx.at(norm(c, d));
        if (N == 1) { reps_ = {Mat22::I()}; cls_ = {1}; }
        m_ = static_cast<uint32_t>(reps_.size());
        D_ = m_ * d_;
        for (auto& r : reps_) inv_.push_back(r.inv_unimodular());
        ENGINE_CHECK_MSG(reps_[0] == Mat22::I() || N == 1 || reps_[0].c % static_cast<int64_t>(N) == 0, "GroupRouteN: w_1");
    }
    uint32_t m() const { return m_; }
    uint32_t cls(const Mat22& x) const { return N_ == 1 ? 1 : cls_[Mod(x.c, N_) * N_ + Mod(x.d, N_)]; }
    bool inH(const Mat22& x) const { return x.det() == 1 && Mod(x.c, N_) == 0; }
    Mat22 w(uint32_t j) const { return reps_[j - 1]; }

    const std::vector<uint64_t>& rho(const Mat22& g) {
        const std::array<uint32_t, 4> key = g.reduced_mod(static_cast<uint32_t>(P_));
        auto it = rho_.find(key);
        if (it != rho_.end()) return it->second;
        const FlatMatrixT<uint32_t> A(2, 2, static_cast<uint32_t>(P_), std::vector<uint32_t>{key[0], key[1], key[2], key[3]});
        const FlatMatrixT<uint32_t> R = ComputeSymmetricPowerMatrixFM(A, n_);
        std::vector<uint64_t> v(static_cast<size_t>(d_) * d_);
        for (uint32_t r = 0; r < d_; ++r)
            for (uint32_t c = 0; c < d_; ++c) v[static_cast<size_t>(r) * d_ + c] = R.get_entry(r, c);
        return rho_.emplace(key, std::move(v)).first->second;
    }
    std::vector<uint64_t> RightMul(const std::vector<uint64_t>& R, const Mat22& gamma) {
        std::vector<uint64_t> out(static_cast<size_t>(d_) * D_, 0);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22 x = w(i) * gamma;
            const uint32_t j = cls(x);
            const std::vector<uint64_t>& B = rho(x * inv_[j - 1]);
            for (uint32_t r = 0; r < d_; ++r)
                for (uint32_t k = 0; k < d_; ++k) {
                    const uint64_t a = R[static_cast<size_t>(r) * D_ + (i - 1) * d_ + k];
                    if (a == 0) continue;
                    for (uint32_t c = 0; c < d_; ++c) {
                        uint64_t& o = out[static_cast<size_t>(r) * D_ + (j - 1) * d_ + c];
                        o = (o + a * B[static_cast<size_t>(k) * d_ + c]) % P_;
                    }
                }
        }
        return out;
    }
    std::vector<uint64_t> Dense(const Mat22& gamma) {
        std::vector<uint64_t> out(static_cast<size_t>(D_) * D_, 0);
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22 x = w(i) * gamma;
            const uint32_t j = cls(x);
            const std::vector<uint64_t>& B = rho(x * inv_[j - 1]);
            for (uint32_t r = 0; r < d_; ++r)
                for (uint32_t c = 0; c < d_; ++c) out[static_cast<size_t>((i - 1) * d_ + r) * D_ + (j - 1) * d_ + c] = B[static_cast<size_t>(r) * d_ + c];
        }
        return out;
    }
    struct Letter { bool S; int64_t k; };
    static Mat22 Tpow(int64_t k) { return Mat22{1, k, 0, 1}; }
    std::vector<Letter> Decompose(const Mat22& g0, bool& ok) const {
        std::vector<Letter> L;
        Mat22 x = g0;
        while (x.c != 0) {
            int64_t k = x.a / x.c;
            if ((x.a % x.c != 0) && ((x.a < 0) != (x.c < 0))) --k;
            if (k != 0) { L.push_back({false, k}); x = Tpow(-k) * x; }
            L.push_back({true, 0});
            x = Mat22{0, 1, -1, 0} * x;
        }
        if (x.a == 1) { if (x.b) L.push_back({false, x.b}); }
        else { if (x.b) L.push_back({false, -x.b}); }
        Mat22 prod = Mat22::I();
        for (const Letter& l : L) prod = prod * (l.S ? Mat22::S() : Tpow(l.k));
        ok = (prod == g0) || (prod == g0.neg());
        return L;
    }
    std::vector<uint64_t> SumT(const std::vector<uint64_t>& R, int64_t k) {
        if (k == 0) return std::vector<uint64_t>(R.size(), 0);
        if (k == 1) return R;
        if (k % 2 == 0) {
            std::vector<uint64_t> X = SumT(R, k / 2);
            const std::vector<uint64_t> Y = RightMul(X, Tpow(k / 2));
            for (size_t e = 0; e < X.size(); ++e) X[e] = (X[e] + Y[e]) % P_;
            return X;
        }
        std::vector<uint64_t> X = SumT(R, k - 1);
        const std::vector<uint64_t> Y = RightMul(R, Tpow(k - 1));
        for (size_t e = 0; e < X.size(); ++e) X[e] = (X[e] + Y[e]) % P_;
        return X;
    }
    struct Eval { std::vector<uint64_t> Ea, Eb; };
    const Eval& E(const Mat22& x) {
        const std::array<int64_t, 4> key{x.a, x.b, x.c, x.d};
        auto it = eval_.find(key);
        if (it != eval_.end()) return it->second;
        bool ok = false;
        const std::vector<Letter> L = Decompose(x, ok);
        ENGINE_CHECK_MSG(ok, "GroupRouteN: word decomposition failed");
        std::vector<uint64_t> R(static_cast<size_t>(d_) * D_, 0);
        for (uint32_t r = 0; r < d_; ++r) R[static_cast<size_t>(r) * D_ + r] = 1;
        Eval ev{std::vector<uint64_t>(R.size(), 0), std::vector<uint64_t>(R.size(), 0)};
        auto add_FT = [&](const std::vector<uint64_t>& Q, bool neg) {
            const std::vector<uint64_t> QS = RightMul(Q, Mat22::S());
            for (size_t e = 0; e < Q.size(); ++e) {
                ev.Ea[e] = neg ? (ev.Ea[e] + P_ - Q[e]) % P_ : (ev.Ea[e] + Q[e]) % P_;
                ev.Eb[e] = neg ? (ev.Eb[e] + P_ - QS[e]) % P_ : (ev.Eb[e] + QS[e]) % P_;
            }
        };
        for (const Letter& l : L) {
            if (l.S) {
                for (size_t e = 0; e < R.size(); ++e) ev.Ea[e] = (ev.Ea[e] + R[e]) % P_;
                R = RightMul(R, Mat22::S());
            } else if (l.k > 0) {
                add_FT(SumT(R, l.k), false);
                R = RightMul(R, Tpow(l.k));
            } else {
                const std::vector<uint64_t> Rk = RightMul(R, Tpow(l.k));
                add_FT(SumT(Rk, -l.k), true);
                R = Rk;
            }
        }
        return eval_.emplace(key, std::move(ev)).first->second;
    }
    std::vector<uint64_t> f(const Mat22& x, const std::vector<uint64_t>& z) {
        const Eval& ev = E(x);
        std::vector<uint64_t> out(d_, 0);
        for (uint32_t r = 0; r < d_; ++r) {
            u128 acc = 0;
            for (uint32_t c = 0; c < D_; ++c)
                acc += static_cast<u128>(ev.Ea[static_cast<size_t>(r) * D_ + c]) * z[c] +
                       static_cast<u128>(ev.Eb[static_cast<size_t>(r) * D_ + c]) * z[D_ + c];
            out[r] = static_cast<uint64_t>(acc % P_);
        }
        return out;
    }
    HMat dense_to(const std::vector<uint64_t>& A, uint32_t R, uint32_t C) const {
        HMat M = MakeField(R, C, P_);
        M.a = A;
        return M;
    }
    void Setup(Checker& g, const std::string& tag) {
        const std::vector<uint64_t> MS = Dense(Mat22::S()), MU = Dense(Mat22::ST());
        std::vector<uint64_t> AS(MS), AU(static_cast<size_t>(D_) * D_, 0);
        const HMat MUm = dense_to(MU, D_, D_);
        const HMat MU2 = HMul(MUm, MUm);
        for (uint32_t i = 0; i < D_; ++i) {
            AS[static_cast<size_t>(i) * D_ + i] = (AS[static_cast<size_t>(i) * D_ + i] + 1) % P_;
            for (uint32_t j = 0; j < D_; ++j)
                AU[static_cast<size_t>(i) * D_ + j] = (MU[static_cast<size_t>(i) * D_ + j] + MU2.at(i, j) + (i == j ? 1 : 0)) % P_;
        }
        const HMat kS = FieldKernel(dense_to(AS, D_, D_)), kU = FieldKernel(dense_to(AU, D_, D_));
        for (uint32_t j = 0; j < kS.cols; ++j) { std::vector<uint64_t> z(2 * D_, 0); for (uint32_t i = 0; i < D_; ++i) z[i] = kS.at(i, j); Z1_.push_back(z); }
        for (uint32_t j = 0; j < kU.cols; ++j) { std::vector<uint64_t> z(2 * D_, 0); for (uint32_t i = 0; i < D_; ++i) z[D_ + i] = kU.at(i, j); Z1_.push_back(z); }
        std::vector<uint64_t> B1(static_cast<size_t>(2) * D_ * D_, 0);
        for (uint32_t i = 0; i < D_; ++i)
            for (uint32_t j = 0; j < D_; ++j) {
                B1[static_cast<size_t>(i) * D_ + j] = (MS[static_cast<size_t>(i) * D_ + j] + P_ - (i == j ? 1 : 0)) % P_;
                B1[static_cast<size_t>(D_ + i) * D_ + j] = (MU[static_cast<size_t>(i) * D_ + j] + P_ - (i == j ? 1 : 0)) % P_;
            }
        const uint32_t rankB1 = HRankModEll(dense_to(B1, 2 * D_, D_));
        h0_ = D_ - rankB1;
        h1_ = static_cast<uint32_t>(Z1_.size()) - rankB1;
        for (uint32_t j = 1; j <= m_; ++j)
            for (const Mat22& x : {Mat22::S(), Mat22::T()}) {
                const Mat22 y = w(j) * x;
                const Mat22 s = y * inv_[cls(y) - 1];
                ENGINE_CHECK(inH(s));
                if (s == Mat22::I() || s == Mat22::I().neg()) continue;
                gens_.push_back(s);
            }
        const uint32_t len = static_cast<uint32_t>(gens_.size()) * d_;
        std::vector<std::vector<uint64_t>> cols;
        for (uint32_t r = 0; r < d_; ++r) {
            std::vector<uint64_t> v(len, 0);
            for (size_t s = 0; s < gens_.size(); ++s) {
                const auto& rs = rho(gens_[s]);
                for (uint32_t a = 0; a < d_; ++a) v[s * d_ + a] = (rs[static_cast<size_t>(a) * d_ + r] + P_ - (a == r ? 1 : 0)) % P_;
            }
            cols.push_back(v);
        }
        // B_H basis then the Shapiro restrictions, greedily independent.
        auto rank_of = [&](const std::vector<std::vector<uint64_t>>& cs) {
            HMat T = MakeField(len, static_cast<uint32_t>(cs.size()), P_);
            for (uint32_t j = 0; j < T.cols; ++j)
                for (uint32_t i = 0; i < len; ++i) T.a[static_cast<size_t>(i) * T.cols + j] = cs[j][i];
            return ColRank(T);
        };
        for (const auto& v : cols) {
            bh_.push_back(v);
            if (rank_of(bh_) < bh_.size()) bh_.pop_back();
        }
        std::vector<std::vector<uint64_t>> all = bh_;
        uint32_t base = static_cast<uint32_t>(bh_.size());
        for (const auto& z : Z1_) {
            all.push_back(Stack(z));
            if (rank_of(all) > base) { reps_z_.push_back(z); ++base; }
            else all.pop_back();
        }
        g.check(reps_z_.size() == h1_, "(G) " + tag + ": Shapiro -- dim Z^1/B^1(Gamma; Ind) = " + std::to_string(h1_) +
                                           " = rank of the restriction to Gamma_0(N) mod B_H");
        basis_cols_ = all;
        len_ = len;
    }
    uint32_t h1() const { return h1_; }
    uint32_t h0() const { return h0_; }
    std::vector<uint64_t> Stack(const std::vector<uint64_t>& z) {
        std::vector<uint64_t> v(static_cast<size_t>(gens_.size()) * d_, 0);
        for (size_t s = 0; s < gens_.size(); ++s) {
            const auto fs = f(gens_[s], z);
            std::copy(fs.begin(), fs.end(), v.begin() + s * d_);
        }
        return v;
    }
    // Own transversal of (H cap g H g^{-1}) \ H, g = diag(1, p).
    std::vector<Mat22> Transversal(uint64_t p) const {
        std::vector<Mat22> r;
        for (int64_t t = 0; t < static_cast<int64_t>(p); ++t) r.push_back(Mat22{1, 0, static_cast<int64_t>(N_) * t, 1});
        if (N_ % p != 0)
            for (int64_t x = 1; x < 100000; ++x)
                if ((x * static_cast<int64_t>(p) - 1) % static_cast<int64_t>(N_) == 0) {
                    r.push_back(Mat22{x, (x * static_cast<int64_t>(p) - 1) / static_cast<int64_t>(N_), static_cast<int64_t>(N_), static_cast<int64_t>(p)});
                    break;
                }
        return r;
    }
    HMat Operator(uint64_t p) {
        const Mat22 g{1, 0, 0, static_cast<int64_t>(p)};
        const std::vector<Mat22> r = Transversal(p);
        auto in_Bpp = [&](const Mat22& x) {
            if (!inH(x)) return false;
            const int64_t D = g.det();
            const Mat22 M = g.adj() * x * g;
            if (M.a % D || M.b % D || M.c % D || M.d % D) return false;
            return inH(Mat22{M.a / D, M.b / D, M.c / D, M.d / D});
        };
        const uint32_t nb = static_cast<uint32_t>(bh_.size()), h = static_cast<uint32_t>(reps_z_.size());
        std::vector<std::vector<uint64_t>> rhs;
        for (const auto& z : reps_z_) {
            std::vector<uint64_t> v(len_, 0);
            for (size_t s = 0; s < gens_.size(); ++s)
                for (const Mat22& rt : r) {
                    const Mat22 x = rt * gens_[s];
                    int found = -1;
                    for (size_t u = 0; u < r.size(); ++u)
                        if (in_Bpp(x * r[u].inv_unimodular())) { found = static_cast<int>(u); break; }
                    ENGINE_CHECK_MSG(found >= 0, "GroupRouteN: t . h not found");
                    const Mat22 bt = x * r[found].inv_unimodular();
                    const int64_t D = g.det();
                    const Mat22 M = g.adj() * bt * g;
                    const auto fy = f(Mat22{M.a / D, M.b / D, M.c / D, M.d / D}, z);
                    const auto& R = rho(rt.inv_unimodular() * g);
                    for (uint32_t a = 0; a < d_; ++a) {
                        u128 acc = 0;
                        for (uint32_t c = 0; c < d_; ++c) acc += static_cast<u128>(R[static_cast<size_t>(a) * d_ + c]) * fy[c];
                        v[s * d_ + a] = (v[s * d_ + a] + static_cast<uint64_t>(acc % P_)) % P_;
                    }
                }
            rhs.push_back(v);
        }
        const uint32_t C = static_cast<uint32_t>(basis_cols_.size()), R = static_cast<uint32_t>(rhs.size());
        std::vector<std::vector<uint64_t>> M(len_, std::vector<uint64_t>(C + R, 0));
        for (uint32_t i = 0; i < len_; ++i) {
            for (uint32_t j = 0; j < C; ++j) M[i][j] = basis_cols_[j][i];
            for (uint32_t j = 0; j < R; ++j) M[i][C + j] = rhs[j][i];
        }
        uint32_t rk = 0;
        for (uint32_t c = 0; c < C; ++c) {
            uint32_t piv = rk;
            while (piv < len_ && M[piv][c] == 0) ++piv;
            ENGINE_CHECK_MSG(piv < len_, "GroupRouteN: the basis columns are dependent");
            std::swap(M[piv], M[rk]);
            const uint64_t iv = powmod<uint64_t>(M[rk][c], P_ - 2, P_);
            for (uint64_t& x : M[rk]) x = static_cast<uint64_t>(static_cast<u128>(x) * iv % P_);
            for (uint32_t i = 0; i < len_; ++i) {
                if (i == rk || M[i][c] == 0) continue;
                const uint64_t fct = M[i][c];
                for (uint32_t j = 0; j < C + R; ++j) M[i][j] = (M[i][j] + P_ - static_cast<uint64_t>(static_cast<u128>(fct) * M[rk][j] % P_)) % P_;
            }
            ++rk;
        }
        for (uint32_t i = C; i < len_; ++i)
            for (uint32_t j = 0; j < R; ++j) ENGINE_CHECK_MSG(M[i][C + j] == 0, "GroupRouteN: an image is not a cocycle class");
        HMat Op = MakeField(h, h, P_);
        for (uint32_t i = 0; i < h; ++i)
            for (uint32_t j = 0; j < h; ++j) Op.a[static_cast<size_t>(i) * h + j] = M[nb + i][C + j];
        return Op;
    }

private:
    uint64_t P_, N_;
    uint32_t n_, d_, m_ = 0, D_ = 0;
    std::vector<Mat22> reps_, inv_;
    std::vector<uint32_t> cls_;
    std::map<std::array<uint32_t, 4>, std::vector<uint64_t>> rho_;
    std::map<std::array<int64_t, 4>, Eval> eval_;
    std::vector<std::vector<uint64_t>> Z1_, reps_z_, bh_, basis_cols_;
    std::vector<Mat22> gens_;
    uint32_t h0_ = 0, h1_ = 0, len_ = 0;
};

void ArmGroup(Checker& g, const NCell& c, const std::map<uint32_t, HMat>& ops) {
    PhaseTimer pt("(G) group-cocycle route");
    const std::string tag = c.tag();
    GroupRouteN gr(c.n, c.N, c.ctx.p);
    gr.Setup(g, tag);
    const auto& dims = c.H->get_H_eq_dimensions();
    g.check(gr.h1() == dims[1] && gr.h0() == dims[0],
            "(G) " + tag + ": dim H^0, H^1 group route = model (" + std::to_string(gr.h0()) + ", " + std::to_string(gr.h1()) + ")");
    std::map<uint32_t, HMat> gops;
    for (const auto& kv : ops) {
        gops[kv.first] = gr.Operator(kv.first);
        g.check(HCharpoly(gops[kv.first]) == HCharpoly(kv.second),
                "(G) " + tag + ": charpoly of " + (c.N % kv.first ? "T_" : "U_") + std::to_string(kv.first) +
                    "^(N): group route == eq:levelcorr on the model " + PolyText(HCharpoly(kv.second)));
    }
    if (ops.size() >= 2) {   // a combination separates simultaneous structure
        HMat A = MakeField(ops.begin()->second.rows, ops.begin()->second.cols, c.ctx.p), B = A;
        uint64_t coef = 1;
        for (const auto& kv : ops) {
            A = HAdd(A, HScale(kv.second, coef));
            B = HAdd(B, HScale(gops.at(kv.first), coef));
            coef = coef * 3 + 1;
        }
        g.check(HCharpoly(A) == HCharpoly(B), "(G) " + tag + ": charpoly of sum_p c_p T_p: group route == model");
    }
}

// (K) classical eigenvalues ((LN3)'s list, the tau half), over F_P.
void ArmClassical(Checker& g, const NCell& c, const std::map<uint32_t, HMat>& ops) {
    using F = std::vector<std::pair<std::vector<int64_t>, uint32_t>>;
    std::vector<std::pair<uint32_t, F>> exp;
    if (c.N == 11 && c.n == 0) exp = {{2, {{{2, 1}, 2}, {{-3, 1}, 1}}}, {3, {{{1, 1}, 2}, {{-4, 1}, 1}}}, {11, {{{-1, 1}, 3}}}};
    if (c.N == 5 && c.n == 2) exp = {{2, {{{4, 1}, 2}, {{-9, 1}, 2}}}, {5, {{{5, 1}, 2}, {{-1, 1}, 1}, {{-125, 1}, 1}}}};
    if (c.N == 6 && c.n == 2) exp = {{2, {{{2, 1}, 2}, {{-1, 1}, 2}, {{-8, 1}, 2}}}};
    if (c.N == 35 && c.n == 0) exp = {{2, {{{0, 1}, 2}, {{-3, 1}, 3}, {{-4, 1, 1}, 2}}}};
    for (const auto& e : exp) {
        if (!ops.count(e.first)) continue;
        const Poly want = FromFactors(e.second, c.ctx.p), got = HCharpoly(ops.at(e.first));
        g.check(want == got, "(K) " + c.tag() + ": charpoly of the level-N operator at p = " + std::to_string(e.first) +
                                 " is the classical one " + PolyText(want) + " (got " + PolyText(got) + ")");
    }
}

// ---------------------------------------------------------------------------
// (E) the plug-point regression at N = ell: Gamma0LevelN(ell) against the
// shipped level-ell objects of slot-13 WP-6, entrywise.
// ---------------------------------------------------------------------------
void ArmRegression(Checker& g, uint32_t n, uint32_t ell, const engine::ModCtx<uint32_t>& ctx) {
    PhaseTimer pt("(E) plug-point regression");
    Model& M = TheModel();
    const std::string tag = "(E) (n=" + std::to_string(n) + ", ell=" + std::to_string(ell) + ") over " + RingText(ctx);
    const uint32_t coef = ctx.is_field() ? ctx.p : ell;
    HeckeDatum U = MakeDatum(ell, coef, n), T2 = MakeDatum(2, coef, n);
    const Gamma0Level shipped(*U.cosets);
    const Gamma0LevelN mine(ell);
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    if (!ctx.is_field()) ring.reset(new hecke::ScopedCoefficientRing(ctx.k));
    const uint32_t D = (ell + 1) * (n + 1);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> ref;
    if (ctx.is_field()) ref = hecke::MakeInducedLocalSystemGamma0(*M.K, *U.cosets, ctx.p, n).get_user_defined_simplex_relations();
    else ref = InducedRelationsRing(*M.K, *U.cosets, n, ctx);
    const auto got = InducedRelationsLevel(*M.K, mine, n, ctx);
    bool eq = ref.size() == got.size();
    for (size_t i = 0; eq && i < ref.size(); ++i) {
        eq = ref[i].q_dim == got[i].q_dim && ref[i].s_lhs_idx_in_Kq == got[i].s_lhs_idx_in_Kq &&
             ref[i].s_rhs_idx_in_Kq == got[i].s_rhs_idx_in_Kq && ref[i].tag == got[i].tag;
        for (uint32_t a = 0; eq && a < D; ++a)
            for (uint32_t b = 0; b < D; ++b)
                if (ref[i].M.get_entry(a, b) != got[i].M.get_entry(a, b)) { eq = false; break; }
    }
    g.check(eq, tag + ": InducedRelationsLevel(Gamma0LevelN(ell)) == the shipped level-ell relation list entrywise (" +
                    std::to_string(got.size()) + " relations)");
    // the two cohomologies, and the operators through each level structure.
    LocalSystemSimplicialComplex Lref(*M.K, ref, D, ctx.p), Lmine(*M.K, got, D, ctx.p);
    EquivariantCohomologyOnK Href(&Lref), Hmine(&Lmine);
    Href.compute_cohomology();
    Hmine.compute_cohomology();
    const HFrame fr(&Href, 1), fm(&Hmine, 1);
    std::vector<std::pair<HeckeDatum*, uint32_t>> ds{{&U, ell}, {&T2, 2}};
    for (const auto& dp : ds) {
        const uint32_t p = dp.second;
        const std::vector<Mat22> r_old = Gamma0SubTransversal(ell, p);
        const Gamma0LevelN lp(static_cast<uint64_t>(ell) * p);
        CorrespondenceSpec s_ref{&shipped, &shipped, dp.first->cosets.get(), dp.first->tau.get(), r_old, "ref"};
        CorrespondenceSpec s_same{&mine, &mine, dp.first->cosets.get(), dp.first->tau.get(), r_old, "same r"};
        CorrespondenceSpec s_pin{&mine, &mine, dp.first->cosets.get(), dp.first->tau.get(), Gamma0SubTransversalN(mine, lp), "pinned"};
        VerifyTransversal(s_pin, p == ell ? p : p + 1);
        const std::string op = std::string(p == ell ? "U_" : "T_") + std::to_string(p) + "^(ell)";
        bool agg = true;
        for (uint32_t q = 0; q < 3; ++q)
            agg = agg && BuildLevelCorrespondence(ctx, s_ref, *M.K, n, q).aggregate() == BuildLevelCorrespondence(ctx, s_same, *M.K, n, q).aggregate();
        g.check(agg, tag + ": " + op + " through Gamma0LevelN == through the shipped Gamma0Level, entrywise as cochain maps, q = 0, 1, 2");
        const HMat A = InducedOnH(fr, fr, BuildLevelCorrespondence(ctx, s_ref, *M.K, n, 1), op);
        const HMat B = InducedOnH(fm, fm, BuildLevelCorrespondence(ctx, s_pin, *M.K, n, 1), op);
        g.check(HEqual(A, B), tag + ": " + op + " on H^1 (the pinned r_t on the Gamma0LevelN model) == HeckeHidaLevelEllValidation's "
                                   "object entrywise");
    }
}

// ---------------------------------------------------------------------------
// (R3) Mazur's Eisenstein ideal at weight 2.
// ---------------------------------------------------------------------------
// Meaningful only for N prime and p | N - 1 (the discrete logarithm mod p is then generator-independent; for composite N
// the primitive-root search below would not terminate): refused otherwise (review S-2).
bool MerelCriterion(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 3 && hecke::IsPrimeU64(N) && p >= 2 && hecke::IsPrimeU64(p) && (N - 1) % p == 0,
                     ("MerelCriterion: N prime, p prime and p | N - 1 (LN5); got N=" + std::to_string(N) +
                      " p=" + std::to_string(p)).c_str());
    // primitive root mod N
    uint64_t gen = 2;
    std::vector<uint64_t> fac;
    for (uint64_t x = N - 1, q = 2; x > 1; ++q) {
        if (q * q > x) { fac.push_back(x); break; }
        if (x % q == 0) { fac.push_back(q); while (x % q == 0) x /= q; }
    }
    for (;; ++gen) {
        bool ok = true;
        for (uint64_t q : fac) ok = ok && powmod<uint64_t>(gen, (N - 1) / q, N) != 1;
        if (ok) break;
    }
    std::vector<uint64_t> lg(N, 0);
    uint64_t x = 1;
    for (uint64_t e = 0; e < N - 1; ++e) { lg[x] = e; x = x * gen % N; }
    uint64_t s = 0;
    for (uint64_t i = 1; i <= (N - 1) / 2; ++i) s = (s + i % p * (lg[i] % p)) % p;
    return s == 0;
}

struct R3Cell { uint64_t N, p; uint32_t dimH1, dimW, loewy; bool merel; };

void ArmR3(Checker& g, const R3Cell& r, std::vector<std::string>& lines) {
    PhaseTimer pt("(R3) Eisenstein cells");
    std::vector<uint32_t> qs;
    for (uint32_t q : {2u, 3u, 5u, 7u}) if (q != r.N) qs.push_back(q);
    const auto c = BuildNCell(r.N, 0, engine::ModCtx<uint32_t>(static_cast<uint32_t>(r.p)), qs);
    const std::string tag = "(R3) (N=" + std::to_string(r.N) + ", p=" + std::to_string(r.p) + ")";
    const HFrame f(c->H.get(), 1);
    const std::map<uint32_t, HMat> ops = ReadOps(*c, f);
    const uint32_t h = f.t();
    g.check(h == r.dimH1, tag + ": dim H^1(Gamma_0(N); F_p) = " + std::to_string(h) + " (frozen scan: " + std::to_string(r.dimH1) + ")");
    std::vector<HMat> nil;
    for (const auto& kv : ops) nil.push_back(SubScalar(kv.second, 1 + kv.first));
    std::vector<HMat> gen;
    for (const HMat& Nq : nil) gen.push_back(Pow(Nq, h));
    // Caveats of (LN4)-(LN6) as tested here (review M-8, WP-5 referee F2): W is the JOINT generalised eigenspace for
    // T_q, q in {2, 3, 5, 7} \ {N}, only; e is read off dim W; the Loewy length below is that of W over the subalgebra
    // generated by those T_q (a single q can under-generate: at (151, 5) nilp(T_2 - 3) = 1 against Loewy length 2).
    const HMat W = FieldKernel(StackRows(gen));
    // W invariant, Loewy layers m^i W.
    bool inv = true;
    for (const HMat& Nq : nil) inv = inv && ColRank(ConcatCols({W, HMul(Nq, W)})) == W.cols;
    g.check(inv, tag + ": W_{N,p} is stable under every T_q");
    std::vector<uint32_t> layers{W.cols};
    HMat cur = W;
    while (cur.cols > 0 && ColRank(cur) > 0) {
        std::vector<HMat> imgs;
        for (const HMat& Nq : nil) imgs.push_back(HMul(Nq, cur));
        HMat nxt = ConcatCols(imgs);
        const uint32_t rk = ColRank(nxt);
        if (rk == 0) break;
        // reduce to a basis (greedy column selection)
        HMat basis = MakeField(nxt.rows, 0, nxt.modulus);
        for (uint32_t j = 0; j < nxt.cols; ++j) {
            HMat col = MakeField(nxt.rows, 1, nxt.modulus);
            for (uint32_t i = 0; i < nxt.rows; ++i) col.a[i] = nxt.at(i, j);
            HMat trial = basis.cols ? ConcatCols({basis, col}) : col;
            if (ColRank(trial) > basis.cols) basis = trial;
        }
        layers.push_back(basis.cols);
        cur = basis;
        if (layers.size() > 64) break;
    }
    const uint32_t loewy = static_cast<uint32_t>(layers.size());
    const bool merel = MerelCriterion(r.N, r.p);
    std::ostringstream ls;
    for (size_t i = 0; i < layers.size(); ++i) ls << (i ? "," : "") << layers[i];
    g.check(W.cols == r.dimW && loewy == r.loewy, tag + ": dim W_{N,p} = " + std::to_string(W.cols) + ", Loewy length " +
                                                      std::to_string(loewy) + " (frozen scan: " + std::to_string(r.dimW) + ", " +
                                                      std::to_string(r.loewy) + ")");
    g.check(merel == r.merel, tag + ": Merel's criterion evaluated in C++ = the frozen flag (" + std::to_string(merel) + ")");
    g.check(W.cols % 2 == 1 && loewy == (W.cols + 1) / 2, tag + ": (LN6) Loewy length = (dim W + 1)/2 = e + 1");
    g.check((W.cols >= 5) == merel, tag + ": (LN4)+(LN5) dim W >= 5 iff Merel's criterion");
    std::ostringstream os;
    os << "LEVEL_N_R3 N=" << r.N << " p=" << r.p << " dimH1=" << h << " dimW=" << W.cols << " e=" << (W.cols - 1) / 2
       << " loewy=" << loewy << " layers=" << ls.str() << " merel=" << merel << " build_s=" << c->build_s;
    lines.push_back(os.str());
    std::cout << "  " << os.str() << std::endl;
}

struct R3ScanCell { R3Cell c; bool frozen; };
struct R3ScanTally {
    std::map<uint32_t, uint32_t> e_dist;
    uint32_t merel = 0, ln4_fail = 0, ln45_fail = 0, ln6_fail = 0;
    double max_wall = 0, max_rss = 0;
};

// Slot-15 WP-9: one (R3) cell on the sparse route (ReadEisensteinR3Scan). A frozen cell gates its frozen values
// and (LN4)-(LN6) exactly as ArmR3; a bare reading gates only what the reader certifies (W stable, the layers
// strictly decreasing -- ENGINE_CHECKs inside -- and W non-zero) and reports (LN4)-(LN6) per cell ([C]
// rmk:levelNcheckable: a failing \CONJ item is a finding, not a failure).
void ArmR3Scan(Checker& g, const R3ScanCell& sc, uint32_t qmax, const std::map<std::pair<uint64_t, uint64_t>, std::string>& model,
               R3ScanTally& tally, std::vector<std::string>& lines) {
    const R3Cell& r = sc.c;
    oc_util::ResetPeakRss();
    const auto t0 = std::chrono::steady_clock::now();
    const R3ScanReading R = ReadEisensteinR3Scan(r.N, r.p, R3HeckePrimes(r.N, qmax));
    const bool merel = MerelCriterion(r.N, r.p);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double rss = oc_util::PeakRssMiB();
    const std::string tag = "(R3s) (N=" + std::to_string(r.N) + ", p=" + std::to_string(r.p) + ")";
    std::ostringstream ls, ws, qs;
    for (size_t i = 0; i < R.layers.size(); ++i) ls << (i ? "," : "") << R.layers[i];
    for (size_t i = 0; i < R.w_after.size(); ++i) ws << (i ? "," : "") << R.w_after[i];
    for (size_t i = 0; i < R.qs.size(); ++i) qs << (i ? "," : "") << R.qs[i];
    const bool ln4 = R.dimW % 2 == 1 && R.dimW >= 3;
    const bool ln6 = R.loewy == (R.dimW + 1) / 2;
    const bool ln45 = (R.dimW >= 5) == merel;
    g.check(R.dimW >= 1 && !R.layers.empty() && R.layers.back() >= 1, tag + ": W_{N,p} non-zero (the Eisenstein class), layers read");
    if (sc.frozen) {
        g.check(R.dimH1 == r.dimH1 && R.dimW == r.dimW && R.loewy == r.loewy,
                tag + ": sparse-route dimH1 / dimW / loewy = " + std::to_string(R.dimH1) + " / " + std::to_string(R.dimW) + " / " +
                    std::to_string(R.loewy) + " (frozen: " + std::to_string(r.dimH1) + " / " + std::to_string(r.dimW) + " / " +
                    std::to_string(r.loewy) + ")");
        g.check(merel == r.merel, tag + ": Merel's criterion = the frozen flag");
        g.check(ln4 && ln6, tag + ": (LN6) Loewy length = (dim W + 1)/2 = e + 1");
        g.check(ln45, tag + ": (LN4)+(LN5) dim W >= 5 iff Merel's criterion");
        const auto it = model.find({r.N, r.p});
        if (it != model.end() && qmax == 7)
            g.check(it->second == ls.str(), tag + ": layers on the sparse route = the model route's (" + ls.str() + " vs " + it->second + ")");
    }
    g.check(MerelCriterionLN5(r.N, r.p) == merel, tag + ": MerelCriterionLN5 = MerelCriterion");
    const uint32_t e = R.dimW ? (R.dimW - 1) / 2 : 0;
    ++tally.e_dist[e];
    tally.merel += merel;
    tally.ln4_fail += !ln4;
    tally.ln45_fail += !ln45;
    tally.ln6_fail += !ln6;
    tally.max_wall = std::max(tally.max_wall, wall);
    tally.max_rss = std::max(tally.max_rss, rss);
    std::ostringstream os;
    os << "LEVEL_N_R3_SCAN N=" << r.N << " p=" << r.p << " dimH1=" << R.dimH1 << " dimW=" << R.dimW << " e=" << e
       << " loewy=" << R.loewy << " layers=" << ls.str() << " merel=" << merel << " ln4=" << ln4 << " ln45=" << ln45
       << " ln6=" << ln6 << " qs=" << qs.str() << " w_after=" << ws.str() << " frozen=" << sc.frozen << " nnz=" << R.final_nnz
       << " presentation_s=" << R.presentation_s << " operators_s=" << R.operators_s << " eisenstein_s=" << R.eisenstein_s
       << " wall_s=" << wall << " peak_rss_mib=" << rss;
    lines.push_back(os.str());
    std::cout << "  " << os.str() << std::endl;
}

// Slot-15 WP-17 (WP-9 audit item 1): (R3x), W itself across routes, in the same coordinates. The scan reader's W
// (engine dense kernels on WP-8's sparse presentation) against W computed here by the (R3) arm's own linear algebra
// (Pow / FieldKernel on HMat) on WP-7's DENSE presentation with its dense OnCokernel -- the two presentations share
// their quotient coordinates (the free columns, checked here; arm (S) of HeckeLevelNSparseValidation checks every
// RREF row). Subspace equality: rank W_1 = rank W_2 = rank [W_1 | W_2].
void ArmR3SubspaceCross(Checker& g, uint64_t N, uint64_t p) {
    PhaseTimer pt("(R3x) W across routes");
    const std::string tag = "(R3x) (N=" + std::to_string(N) + ", p=" + std::to_string(p) + ")";
    const engine::ModCtx<uint32_t> ctx(static_cast<uint32_t>(p));
    const Gamma0LevelN lvl(N);
    const std::vector<uint32_t> qs = R3HeckePrimes(N, 7);
    const R3ScanReading R = ReadEisensteinR3Scan(N, p, qs);
    const CokernelPresentation pres = CokernelPresentation::Build(lvl, 0, ctx);
    const SparseCokernel Q = SparseCokernel::Build(lvl, 0, ctx);
    g.check(pres.free_columns() == Q.free_columns() && pres.h1() == R.dimH1,
            tag + ": the dense and the sparse presentation share the quotient coordinates (h1 = " + std::to_string(pres.h1()) + ")");
    const uint32_t h = pres.h1();
    std::vector<HMat> gen;
    for (uint32_t q : qs) gen.push_back(Pow(SubScalar(OnCokernel(pres, HeilbronnOperator(lvl, q, 0, ctx)), 1 + q), h));
    const HMat W2 = FieldKernel(StackRows(gen));
    HMat W1 = MakeField(h, R.dimW, p);
    for (uint32_t i = 0; i < h; ++i)
        for (uint32_t j = 0; j < R.dimW; ++j) W1.a[static_cast<size_t>(i) * R.dimW + j] = R.W[static_cast<size_t>(i) * R.dimW + j];
    const uint32_t r1 = ColRank(W1), r2 = ColRank(W2), r12 = ColRank(ConcatCols({W1, W2}));
    g.check(r1 == R.dimW && r2 == W2.cols && r1 == r2 && r12 == r1,
            tag + ": W_{N,p} of the scan reader = W_{N,p} on the dense presentation as subspaces (dim " + std::to_string(r1) + " / " +
                std::to_string(r2) + ", joint rank " + std::to_string(r12) + ")");
}

// Slot-16 WP-2 (WP-17 section 7 carry): (R3m), the MODEL route against the dense presentation, on three small cells. The
// two H^1 carry different bases (the model's HFrame, the presentation's quotient coordinates), so the comparison is
// basis-free: per q the charpoly of T_q and the ranks of T_q - (1 + q) and of its h-th power (dim W_q), then the joint
// invariants of the whole family -- dim W (kernel of the stacked h-th powers), the charpoly of sum_q c_q T_q and of the
// product of the (T_q - (1+q)) factors, which a separate conjugation of each operator would not preserve.
void ArmR3ModelCross(Checker& g, uint64_t N, uint64_t p) {
    PhaseTimer pt("(R3m) model route vs presentation");
    const std::string tag = "(R3m) (N=" + std::to_string(N) + ", p=" + std::to_string(p) + ")";
    const engine::ModCtx<uint32_t> ctx(static_cast<uint32_t>(p));
    std::vector<uint32_t> qs;
    for (uint32_t q : {2u, 3u, 5u, 7u}) if (q != N) qs.push_back(q);
    const auto c = BuildNCell(N, 0, ctx, qs);
    const HFrame f(c->H.get(), 1);
    const std::map<uint32_t, HMat> mops = ReadOps(*c, f);
    const Gamma0LevelN lvl(N);
    const CokernelPresentation pres = CokernelPresentation::Build(lvl, 0, ctx);
    const uint32_t h = pres.h1();
    g.check(f.t() == h, tag + ": h1 equal on both routes (" + std::to_string(h) + ")");
    std::map<uint32_t, HMat> pops;
    for (uint32_t q : qs) pops[q] = OnCokernel(pres, HeilbronnOperator(lvl, q, 0, ctx));
    bool each = true;
    HMat msum = MakeField(h, h, p), psum = MakeField(h, h, p);
    HMat mprod = HIdentity(msum), pprod = HIdentity(psum);
    std::vector<HMat> mg, pg;
    uint32_t coef = 1;
    for (uint32_t q : qs) {
        const HMat &Tm = mops.at(q), &Tp = pops.at(q);
        const HMat Nm = SubScalar(Tm, 1 + q), Np = SubScalar(Tp, 1 + q);
        each = each && HCharpoly(Tm) == HCharpoly(Tp) && ColRank(Nm) == ColRank(Np) && ColRank(Pow(Nm, h)) == ColRank(Pow(Np, h));
        msum = HAdd(msum, HScale(Tm, coef)); psum = HAdd(psum, HScale(Tp, coef));
        mprod = HMul(mprod, Nm); pprod = HMul(pprod, Np);
        mg.push_back(Pow(Nm, h)); pg.push_back(Pow(Np, h));
        ++coef;
    }
    g.check(each, tag + ": per q, charpoly(T_q), rank(T_q - (1+q)) and dim W_q, model = presentation");
    const HMat Wm = FieldKernel(StackRows(mg)), Wp = FieldKernel(StackRows(pg));
    g.check(Wm.cols == Wp.cols && Wm.cols >= 1, tag + ": dim W_{N,p}, model = presentation (" + std::to_string(Wp.cols) + ")");
    g.check(HCharpoly(msum) == HCharpoly(psum) && ColRank(mprod) == ColRank(pprod) && HCharpoly(HMul(msum, mprod)) == HCharpoly(HMul(psum, pprod)),
            tag + ": joint invariants (charpoly of sum_q q' T_q, rank and charpoly of the product of the T_q - (1+q)), model = presentation");
    // Slot-17 WP-8 (W2-1): the ENTRYWISE form of the same statement -- ONE change of basis P (invertible) with P T^model_q = T^pres_q P
    // for every q at once. The intertwiner space is the kernel of the stacked linear system on the h^2 entries of P; an
    // invertible element is searched among deterministic combinations of its basis (a failure to find one fails the check).
    {
        const size_t hh = static_cast<size_t>(h) * h;
        HMat L = MakeField(static_cast<uint32_t>(qs.size() * hh), static_cast<uint32_t>(hh), p);
        size_t row = 0;
        for (uint32_t q : qs) {
            const HMat &Tm = mops.at(q), &Tp = pops.at(q);
            for (uint32_t i = 0; i < h; ++i)
                for (uint32_t k = 0; k < h; ++k, ++row) {
                    for (uint32_t j = 0; j < h; ++j) {
                        uint64_t& a = L.a[row * hh + static_cast<size_t>(i) * h + j];
                        a = (a + Tm.at(j, k)) % p;
                        uint64_t& b = L.a[row * hh + static_cast<size_t>(j) * h + k];
                        b = (b + p - Tp.at(i, j) % p) % p;
                    }
                }
        }
        const HMat B = FieldKernel(L);   // columns: intertwiners P (row-major h x h)
        bool found = false, entrywise = false;
        uint64_t st = 0x9E3779B97F4A7C15ull ^ (N * 1000003ull + p);
        for (int attempt = 0; attempt < 40 && !found && B.cols > 0; ++attempt) {
            HMat P = MakeField(h, h, p);
            for (uint32_t c = 0; c < B.cols; ++c) {
                st = st * 6364136223846793005ull + 1442695040888963407ull;
                const uint64_t co = (st >> 33) % p;
                for (size_t e = 0; e < hh; ++e) P.a[e] = (P.a[e] + co * B.at(static_cast<uint32_t>(e), c)) % p;
            }
            if (ColRank(P) != h) continue;
            found = true;
            entrywise = true;
            for (uint32_t q : qs) {
                const HMat lhs = HMul(P, mops.at(q)), rhs = HMul(pops.at(q), P);
                entrywise = entrywise && lhs.a == rhs.a;
            }
        }
        g.check(found && entrywise, tag + ": one invertible P with P T^model_q = T^pres_q P entrywise for all q (intertwiner dim " +
                                        std::to_string(B.cols) + ")");
    }
}

// Slot-15 WP-17 (WP-9 audit item 1): the scan reader against WP-8's ReadEisensteinR3 at larger N (same presentation,
// different linear algebra): dimH1, dim W, the layers.
void ArmR3VsWP8(Checker& g, uint64_t N, uint64_t p) {
    PhaseTimer pt("(R3x) scan vs ReadEisensteinR3");
    const std::string tag = "(R3x) (N=" + std::to_string(N) + ", p=" + std::to_string(p) + ")";
    const R3ScanReading S = ReadEisensteinR3Scan(N, p, R3HeckePrimes(N, 7));
    const R3Reading E = ReadEisensteinR3(N, p);
    g.check(S.dimH1 == E.dimH1 && S.dimW == E.dimW && S.loewy == E.loewy && S.layers == E.layers,
            tag + ": ReadEisensteinR3Scan = ReadEisensteinR3 (dimH1 " + std::to_string(S.dimH1) + ", dim W " + std::to_string(S.dimW) +
                ", Loewy " + std::to_string(S.loewy) + ")");
}

// ---- (slot-20 L20-4, ruling O9) HECKE_DUMP_MEREL=<dir>: the owner-gated dump of H_{N,p} for the Lean checker -------
// No read-only arm printed the Heilbronn--Merel data, so this dumper exists, with HECKE_DUMP_TAU's rules: <dir> must be
// an existing directory outside the source tree (else REFUSED, exit 1); unset, the arm is one "[SKIP] MEREL_DUMP" line
// and the validation runs unchanged.  Set, ONLY the dump runs (the validation's own arms do not; EFFECTIVE line): one
// file merel_N<N>_n<n>_p<p>.txt per operator at this validation's own F_P cell (N, n) = (5, 2), p in {2, 5}
// (T_2^{(5)} and U_5^{(5)}), over F_P, P = kP.  Format (MERELDUMP 1): the def:cosetrepsN table (rep j a b c d,
// 1-based), Mer_p in the C++ order (eps a b c d), every term of eq:heilbronn in HeilbronnOperator's order (term dst src,
// the integer matrix adj h_j(eps) = w_{j.eps} adj(eps) w_j^{-1}, then the d x d block row-major as HeilbronnOperator
// stores it), and the dense D x D matrix (row I ...).  Read by lean/tools/merel_dump_to_lean.py ->
// lean/GL2/Checkers/MerelData.lean.  Writes nothing else and changes no value.
// (slot-20 L20-6 repairs, F3-20 MINOR 7 / 8.)  The MEREL_DUMP line carries dense_fnv64 = FNV-1a-64 (Fnv.h) of D, then every
// dense entry row-major, each fed as one 64-bit word; merel_dump_to_lean.py recomputes it from the "row" lines and copies it
// into MerelData.lean's docstring, which ties the committed data to a C++ run.  The EFFECTIVE line says
// validation_arms=skipped: with the variable set, the validation's own arms (D) ... (K) do not run.
bool MerelDumpIsDirectory(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
std::string MerelDumpRealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
    return std::string(buf);
}
bool MerelDumpIsInside(const std::string& inner, const std::string& outer) {
    if (outer.empty() || inner.size() < outer.size()) return false;
    if (inner.compare(0, outer.size(), outer) != 0) return false;
    return inner.size() == outer.size() || inner[outer.size()] == '/';
}

// Returns 0 (dumped, or skipped: *ran = false) or 1 (refused / failed).
uint32_t MerelDumpArm(bool* ran) {
    *ran = false;
    const char* env = std::getenv("HECKE_DUMP_MEREL");
    if (env == nullptr || *env == '\0') {
        std::cout << "  [SKIP] MEREL_DUMP: HECKE_DUMP_MEREL not set (owner-gated dump hook, slot-20 L20-4; when set, it "
                     "replaces this validation's arms)\n";
        return 0;
    }
    *ran = true;
    const std::string dir(env);
    if (!MerelDumpIsDirectory(dir)) {
        std::cout << "HeckeLevelNValidation: REFUSED: HECKE_DUMP_MEREL=" << dir << " is not an existing directory\n";
        return 1;
    }
    const std::string real_dir = MerelDumpRealPath(dir);
#ifdef OC_SOURCE_DIR
    const std::string real_src = MerelDumpRealPath(OC_SOURCE_DIR);
#else
    const std::string real_src;
#endif
    if (real_dir.empty() || MerelDumpIsInside(real_dir, real_src)) {
        std::cout << "HeckeLevelNValidation: REFUSED: HECKE_DUMP_MEREL=" << dir
                  << " lies inside the source tree (" << real_src << "); dump into scratch only\n";
        return 1;
    }
    struct Cell { uint64_t N; uint32_t n; uint64_t p; };
    const std::vector<Cell> cells{{5, 2, 2}, {5, 2, 5}};
    std::cout << "EFFECTIVE HeckeLevelNValidation arm=MEREL_DUMP cells=" << cells.size() << " validation_arms=skipped\n";
    const engine::ModCtx<uint32_t> ctx(kP);
    for (const Cell& c : cells) {
        const Gamma0LevelN lvl(c.N);
        const HeilbronnOperator H(lvl, c.p, c.n, ctx);
        const std::string path = real_dir + "/merel_N" + std::to_string(c.N) + "_n" + std::to_string(c.n) + "_p" +
                                 std::to_string(c.p) + ".txt";
        std::ofstream os(path);
        if (!os) {
            std::cout << "HeckeLevelNValidation: MEREL_DUMP cannot write " << path << "\n";
            return 1;
        }
        const uint32_t d = H.d();
        os << "MERELDUMP 1\n";
        os << "N " << c.N << " n " << c.n << " p " << c.p << " q " << kP << " m " << lvl.m() << "\n";
        for (uint32_t j = 1; j <= lvl.m(); ++j) {
            const Mat22 w = lvl.rep(j);
            os << "rep " << j << ' ' << w.a << ' ' << w.b << ' ' << w.c << ' ' << w.d << "\n";
        }
        for (const Mat22& e : H.merel()) os << "eps " << e.a << ' ' << e.b << ' ' << e.c << ' ' << e.d << "\n";
        // The terms, re-derived pair by pair in HeilbronnOperator's own loop order (so each carries its integer matrix),
        // and checked against H.terms() index by index.
        size_t k = 0;
        for (const Mat22& eps : H.merel()) {
            for (uint32_t j = 1; j <= lvl.m(); ++j) {
                if (!lvl.act_defined(j, eps)) continue;
                const uint32_t i = lvl.act(j, eps);
                ENGINE_CHECK_MSG(k < H.terms().size() && H.terms()[k].dst == i && H.terms()[k].src == j,
                                 "MEREL_DUMP: term order differs from HeilbronnOperator's");
                const Mat22 adj_h = lvl.rep(i) * eps.adj() * lvl.rep(j).inv_unimodular();
                const Block& B = H.blocks()[H.terms()[k].block];
                os << "term " << i << ' ' << j << ' ' << adj_h.a << ' ' << adj_h.b << ' ' << adj_h.c << ' ' << adj_h.d;
                for (uint32_t r = 0; r < d * d; ++r) os << ' ' << B.dense[r];
                os << "\n";
                ++k;
            }
        }
        ENGINE_CHECK_MSG(k == H.terms().size(), "MEREL_DUMP: term count differs from HeilbronnOperator's");
        const std::vector<uint32_t> dense = H.dense();
        const uint32_t D = H.D();
        engine::FnvHasher fnv;   // F3-20 MINOR 7: the provenance hash of the dense matrix
        fnv.feed(D);
        for (uint32_t v : dense) fnv.feed(v);
        for (uint32_t I = 0; I < D; ++I) {
            os << "row " << I;
            for (uint32_t J = 0; J < D; ++J) os << ' ' << dense[static_cast<size_t>(I) * D + J];
            os << "\n";
        }
        os << "end\n";
        os.close();
        if (!os) {
            std::cout << "HeckeLevelNValidation: MEREL_DUMP write failed: " << path << "\n";
            return 1;
        }
        std::cout << "MEREL_DUMP " << path << " N=" << c.N << " n=" << c.n << " p=" << c.p << " q=" << kP
                  << " m=" << lvl.m() << " merel=" << H.merel().size() << " terms=" << H.terms().size()
                  << " dropped=" << H.pairs_dropped() << " D=" << D << " dense_fnv64=0x" << std::hex << std::setw(16)
                  << std::setfill('0') << fnv.value() << std::dec << std::setfill(' ') << "\n";
    }
    return 0;
}

} // namespace

// No catch-all here (review M-5): since S13 WP-16 the harness reports an escaping exception itself as
// "[EXCEPTION] HeckeLevelNValidation: ..." and then [FAIL], and keeps the run going.
uint32_t HeckeLevelNValidation() {
    std::cout << "=== HeckeLevelNValidation (slot-14 WP-12: level N on the model by Shapiro; [C] v1.14.0 sec:levelN "
                 "def:cosetrepsN, def:levelNmodel, (LN1)-(LN6)) ===\n";
    {
        bool dumped = false;
        const uint32_t rc = MerelDumpArm(&dumped);
        if (dumped || rc != 0) return rc;   // HECKE_DUMP_MEREL set: the dump only (slot-20 L20-4)
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Checker g;
    {
        const SimplicialComplex& K = *TheModel().K;
        std::cout << "  model: N_q = " << K.get_num_q_simplices(0) << ", " << K.get_num_q_simplices(1) << ", "
                  << K.get_num_q_simplices(2) << "\n";
    }
    ArmTable(g);
    std::cout << "  [(T) done at " << secs() << " s]" << std::endl;

    // ---- F_P cells: dims, orbit-wise, cochain maps, Hecke ring, group route, classical.
    struct PC { uint64_t N; uint32_t n; std::vector<uint32_t> primes; bool group; };
    const std::vector<PC> cells{
        {4, 0, {3, 5}, true},    {4, 2, {2, 3}, true},   {6, 0, {5, 7}, true},    {6, 2, {2, 5}, true},   {6, 4, {3, 5}, true},
        {8, 2, {2, 3}, true},    {9, 0, {2, 3}, true},   {9, 2, {2, 5}, true},    {10, 0, {3, 5}, true},  {10, 2, {2, 3}, true},
        {12, 0, {5, 7}, true},   {12, 2, {2, 5}, true},  {15, 0, {2, 3}, true},   {15, 2, {2, 5}, true},  {21, 0, {2, 7}, true},
        {21, 2, {2, 3}, true},   {5, 2, {2, 5}, true},   {11, 0, {2, 3, 11}, true}, {35, 0, {2, 5}, true}, {30, 0, {7, 5}, true},
        {12, 4, {5}, false},     {9, 3, {2}, false}};
    for (const PC& pc : cells) {
        const auto c = BuildNCell(pc.N, pc.n, engine::ModCtx<uint32_t>(kP), pc.primes);
        ArmDims(g, *c);
        ArmOrbit(g, *c);
        if (pc.n % 2) continue;
        ArmCochain(g, *c);  // every even-n F_P cell (the former D <= 64 filter dropped, review S-3)
        const HFrame f(c->H.get(), 1);
        const std::map<uint32_t, HMat> ops = ReadOps(*c, f);
        ArmHeckeRing(g, *c, f, ops);
        ArmClassical(g, *c, ops);
        if (pc.group) ArmGroup(g, *c, ops);
        std::cout << "  LEVEL_N N=" << pc.N << " n=" << pc.n << " ring=" << RingText(c->ctx) << " m=" << c->lvl->m()
                  << " h0=" << c->H->get_H_eq_dimensions()[0] << " h1=" << c->H->get_H_eq_dimensions()[1];
        for (const auto& kv : ops) std::cout << " chi(" << (pc.N % kv.first ? "T" : "U") << kv.first << ")=" << PolyText(HCharpoly(kv.second));
        std::cout << " build_s=" << c->build_s << std::endl;
    }
    std::cout << "  [F_P cells done at " << secs() << " s]" << std::endl;

    // ---- F_ell with ell | N and ell !| N: dims and (LN2).
    for (const auto& cc : std::vector<std::array<uint32_t, 3>>{{10, 2, 5}, {21, 2, 7}, {12, 2, 5}, {15, 4, 5}, {14, 0, 7}}) {
        const auto c = BuildNCell(cc[0], cc[1], engine::ModCtx<uint32_t>(cc[2]), {3});
        ArmDims(g, *c);
        ArmOrbit(g, *c);
        const HFrame f(c->H.get(), 1);
        ArmHeckeRing(g, *c, f, ReadOps(*c, f));
    }
    // ---- Z/ell^2: cochain maps and the Hecke ring over the ring (ell | N and ell !| N).
    for (const auto& cc : std::vector<std::array<uint32_t, 3>>{{10, 2, 5}, {6, 2, 5}}) {
        const auto c = BuildNCell(cc[0], cc[1], engine::ModCtx<uint32_t>::prime_power(cc[2], 2), {2, 3});
        std::unique_ptr<hecke::ScopedCoefficientRing> ring(new hecke::ScopedCoefficientRing(2));
        ArmCochain(g, *c);
        ArmOrbit(g, *c);
        const HFrame f(c->H.get(), 1);
        ArmHeckeRing(g, *c, f, ReadOps(*c, f));
    }
    std::cout << "  [F_ell / Z/ell^2 cells done at " << secs() << " s]" << std::endl;

    // ---- (E) the plug-point regression at N = ell.
    ArmRegression(g, 2, 5, engine::ModCtx<uint32_t>(kP));
    ArmRegression(g, 4, 7, engine::ModCtx<uint32_t>(kP));
    ArmRegression(g, 2, 5, engine::ModCtx<uint32_t>(5));
    ArmRegression(g, 2, 5, engine::ModCtx<uint32_t>::prime_power(5, 2));
    std::cout << "  [(E) done at " << secs() << " s]" << std::endl;

    // ---- (R3) Mazur's Eisenstein ideal, weight 2, F_p, p !| N (frozen: docs/implementation/slot14/wp4/eisenstein_scan.tsv).
    std::vector<std::string> lines;
    // Every scan cell with N <= 211 (26 cells), the frozen values of the WP-4 scan.
    std::vector<R3Cell> r3{
        {11, 5, 3, 3, 2, false}, {23, 11, 5, 3, 2, false}, {29, 7, 5, 3, 2, false}, {31, 5, 5, 5, 3, true},
        {41, 5, 7, 3, 2, false}, {43, 7, 7, 3, 2, false}, {53, 13, 9, 3, 2, false}, {61, 5, 9, 3, 2, false},
        {67, 11, 11, 3, 2, false}, {71, 5, 13, 3, 2, false}, {71, 7, 13, 3, 2, false}, {79, 13, 13, 3, 2, false},
        {89, 11, 15, 3, 2, false}, {101, 5, 17, 3, 2, false}, {113, 7, 19, 3, 2, false}, {127, 7, 21, 5, 3, true},
        {131, 5, 23, 5, 3, true}, {131, 13, 23, 3, 2, false}, {151, 5, 25, 3, 2, false}, {157, 13, 25, 3, 2, false},
        {181, 5, 29, 7, 4, true}, {191, 5, 33, 3, 2, false}, {197, 7, 33, 3, 2, false}, {199, 11, 33, 3, 2, false},
        {211, 5, 35, 5, 3, true}, {211, 7, 35, 3, 2, false}};
    // Slot-15 WP-9: the frozen cells on the SPARSE route as well (ReadEisensteinR3Scan, WP-8's presentation and
    // Heilbronn operators): the 26 cells above plus the named (1321, 11), (1571, 5), against the frozen values
    // and, on the 26, against the model route's layers.
    std::vector<R3Cell> frozen_scan = r3;
    frozen_scan.push_back(R3Cell{1321, 11, 219, 7, 4, true});
    frozen_scan.push_back(R3Cell{1571, 5, 263, 7, 4, true});
    std::vector<R3ScanCell> scan;
    // Read-only hook (not an emitter): HECKE_LEVEL_N_R3_CELLS="N:p:dimH1:dimW:loewy:merel,..." REPLACES the list
    // (e.g. the named large cells "1321:11:219:7:4:1,1571:5:263:7:4:1"; predict RSS first). Since slot-15 WP-9 a
    // cell may also be a bare "N:p" (a READING: nothing frozen; (LN4)-(LN6) reported per cell, a disagreement a
    // finding, not a failure), and every hook cell runs on the sparse route (ReadEisensteinR3Scan) unless
    // HECKE_LEVEL_N_R3_ROUTE=model (6-field cells only: the slot-14 tau-route arm). HECKE_LEVEL_N_R3_QMAX=<q>
    // (default 7) widens the Hecke primes of W to every prime q <= QMAX, q != N (sparse route only).
    bool hook = false, model_route = false;
    uint32_t qmax = 7;
    if (const char* rq = std::getenv("HECKE_LEVEL_N_R3_QMAX")) {
        const std::string t(rq);
        ENGINE_CHECK_MSG(!t.empty() && t.size() <= 6 && t.find_first_not_of("0123456789") == std::string::npos && std::stoul(t) >= 2,
                         "HECKE_LEVEL_N_R3_QMAX must be an integer >= 2");
        qmax = static_cast<uint32_t>(std::stoul(t));
    }
    if (const char* rt = std::getenv("HECKE_LEVEL_N_R3_ROUTE")) {
        const std::string t(rt);
        ENGINE_CHECK_MSG(t == "model" || t == "sparse", "HECKE_LEVEL_N_R3_ROUTE must be 'model' or 'sparse'");
        model_route = t == "model";
    }
    if (const char* env = std::getenv("HECKE_LEVEL_N_R3_CELLS")) {
        hook = true;
        r3.clear();
        std::stringstream ss(env);
        std::string item;
        size_t idx = 0;
        while (std::getline(ss, item, ',')) {
            ++idx;
            // Every cell is validated before any runs (review S-2): a bad cell is refused by its position, never hangs.
            const std::string where = "HECKE_LEVEL_N_R3_CELLS: cell " + std::to_string(idx) + " '" + item + "': ";
            std::vector<uint64_t> v;
            std::stringstream is(item);
            std::string tok;
            while (std::getline(is, tok, ':')) {
                const bool digits = !tok.empty() && tok.size() <= 18 && tok.find_first_not_of("0123456789") == std::string::npos;
                ENGINE_CHECK_MSG(digits, (where + "'" + tok + "' is not a non-negative integer").c_str());
                v.push_back(std::stoull(tok));
            }
            ENGINE_CHECK_MSG(v.size() == 6 || v.size() == 2,
                             (where + "expected N:p:dimH1:dimW:loewy:merel (6 fields) or N:p (2 fields, a reading)").c_str());
            ENGINE_CHECK_MSG(v[0] >= 3 && v[0] < (1ull << 31) && hecke::IsPrimeU64(v[0]), (where + "N must be a prime < 2^31").c_str());
            ENGINE_CHECK_MSG(v[1] >= 5 && v[1] < (1ull << 31) && hecke::IsPrimeU64(v[1]),
                             (where + "p must be a prime >= 5 (6 a unit), < 2^31").c_str());
            ENGINE_CHECK_MSG((v[0] - 1) % v[1] == 0, (where + "p must divide N - 1 (Mazur / Merel, (LN4)-(LN5))").c_str());
            if (v.size() == 6) {
                ENGINE_CHECK_MSG(v[5] <= 1, (where + "merel must be 0 or 1").c_str());
                const R3Cell c{v[0], v[1], static_cast<uint32_t>(v[2]), static_cast<uint32_t>(v[3]), static_cast<uint32_t>(v[4]),
                               v[5] != 0};
                if (model_route) r3.push_back(c);
                else scan.push_back(R3ScanCell{c, true});
            } else {
                ENGINE_CHECK_MSG(!model_route, (where + "a bare N:p reading runs on the sparse route only").c_str());
                scan.push_back(R3ScanCell{R3Cell{v[0], v[1], 0, 0, 0, false}, false});
            }
        }
        std::cout << "  EFFECTIVE HeckeLevelNValidation r3_cells=" << (r3.size() + scan.size()) << " (HECKE_LEVEL_N_R3_CELLS)"
                  << " route=" << (model_route ? "model" : "sparse") << " qmax=" << qmax << std::endl;
    }
    for (const R3Cell& r : r3) ArmR3(g, r, lines);
    // The model route's layers per cell (from its LEVEL_N_R3 lines), for the cross-route check of the frozen cells.
    std::map<std::pair<uint64_t, uint64_t>, std::string> model_layers;
    for (const std::string& l : lines) {
        std::stringstream is(l);
        std::string tok;
        uint64_t N = 0, p = 0;
        std::string lay;
        while (is >> tok) {
            if (tok.rfind("N=", 0) == 0) N = std::stoull(tok.substr(2));
            else if (tok.rfind("p=", 0) == 0) p = std::stoull(tok.substr(2));
            else if (tok.rfind("layers=", 0) == 0) lay = tok.substr(7);
        }
        model_layers[{N, p}] = lay;
    }
    if (!hook)
        for (const R3Cell& r : frozen_scan) scan.push_back(R3ScanCell{r, true});
    R3ScanTally tally;
    for (const R3ScanCell& c : scan) ArmR3Scan(g, c, qmax, model_layers, tally, lines);
    if (!scan.empty()) {
        std::ostringstream os;
        os << "LEVEL_N_R3_SCAN_SUMMARY cells=" << scan.size() << " qmax=" << qmax << " e_dist=";
        bool first = true;
        for (const auto& kv : tally.e_dist) { os << (first ? "" : ",") << "e" << kv.first << ":" << kv.second; first = false; }
        os << " merel_cells=" << tally.merel << " ln4_odd_fail=" << tally.ln4_fail << " ln45_disagree=" << tally.ln45_fail
           << " ln6_disagree=" << tally.ln6_fail << " max_wall_s=" << tally.max_wall << " max_peak_rss_mib=" << tally.max_rss;
        lines.push_back(os.str());
        std::cout << "  " << os.str() << std::endl;
    }
    if (!hook) {
        for (const auto& np : std::vector<std::array<uint64_t, 2>>{{31, 5}, {127, 7}, {181, 5}, {211, 5}, {1321, 11}})
            ArmR3SubspaceCross(g, np[0], np[1]);
        for (const auto& np : std::vector<std::array<uint64_t, 2>>{{3001, 5}, {4159, 7}}) ArmR3VsWP8(g, np[0], np[1]);
        for (const auto& np : std::vector<std::array<uint64_t, 2>>{{11, 5}, {31, 5}, {61, 5}}) ArmR3ModelCross(g, np[0], np[1]);
    }
    std::cout << "  [(R3) done at " << secs() << " s]" << std::endl;

    for (const auto& kv : g.by_arm) std::cout << "  arm " << kv.first << ": " << kv.second << " checks\n";
    for (const auto& kv : PhaseTimes()) std::cout << "  phase " << kv.first << ": " << kv.second << " s\n";
    std::cout << "HeckeLevelNValidation: " << g.checks << " checks, " << g.fails << " failing, " << secs() << " s\n";
    return g.fails == 0 ? 0 : 1;
}
