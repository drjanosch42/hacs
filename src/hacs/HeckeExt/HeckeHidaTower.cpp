// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-5) Hida weight towers at level 1 -- see the header.
#include "HeckeHidaTower.h"

#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../Util/CancelPoint.h"   // (NBCONFIRM S5) cooperative cancellation points
#include "ZellK.h"

namespace hecke_ext {
namespace hida {

using hecke::ordinary::RingMat;
using u128 = unsigned __int128;

namespace {

inline uint64_t Pow64(uint64_t b, uint32_t e) { return engine::scalar::pow_wrap(b, e); }
inline uint64_t MulMod(uint64_t a, uint64_t b, uint64_t q) { return engine::scalar::mulmod(a, b, q); }
inline uint64_t SubMod(uint64_t a, uint64_t b, uint64_t q) { return engine::scalar::submod(a, b, q); }
// (slot-12 WP-11, A3) a + b mod q without a uint64 wrap (q may exceed 2^63).
inline uint64_t AddMod(uint64_t a, uint64_t b, uint64_t q) { return engine::scalar::addmod(a % q, b % q, q); }

RingMat FromDense(const engine::DenseBuf<uint64_t>& A, uint64_t ell, const std::vector<uint32_t>& ex) {
    RingMat M;
    M.ell = ell;
    M.row_exp = ex;
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) M.a.push_back(A.at(i, j));
    return hecke::ordinary::Canonical(M);
}

// Column j of A as an element of the module (+) Z/ell^{a_i}.
std::vector<uint64_t> Column(const RingMat& A, uint32_t j) {
    std::vector<uint64_t> v(A.n());
    for (uint32_t i = 0; i < A.n(); ++i) v[i] = A.at(i, j);
    return v;
}
std::vector<uint64_t> Apply(const RingMat& A, const std::vector<uint64_t>& x) {
    const uint32_t g = A.n();
    std::vector<uint64_t> y(g, 0);
    for (uint32_t i = 0; i < g; ++i) {
        const uint64_t q = Pow64(A.ell, A.row_exp[i]);
        u128 s = 0;
        for (uint32_t j = 0; j < g; ++j) s = (s + static_cast<u128>(A.at(i, j) % q) * (x[j] % q)) % q;
        y[i] = static_cast<uint64_t>(s);
    }
    return y;
}

// Solve B gamma = z over F_ell (B: g x r, columns independent); ENGINE_CHECKs solvability.
std::vector<uint64_t> SolveFl(const std::vector<std::vector<uint64_t>>& Bcols, const std::vector<uint64_t>& z, uint64_t ell) {
    const uint32_t r = static_cast<uint32_t>(Bcols.size());
    const uint32_t g = static_cast<uint32_t>(z.size());
    // augmented g x (r+1)
    std::vector<std::vector<uint64_t>> M(g, std::vector<uint64_t>(r + 1));
    for (uint32_t i = 0; i < g; ++i) {
        for (uint32_t k = 0; k < r; ++k) M[i][k] = Bcols[k][i] % ell;
        M[i][r] = z[i] % ell;
    }
    std::vector<uint32_t> pivrow(r, g);
    uint32_t row = 0;
    for (uint32_t k = 0; k < r; ++k) {
        uint32_t p = row;
        while (p < g && M[p][k] == 0) ++p;
        ENGINE_CHECK_MSG(p < g, "HeckeHidaTower: basis columns are dependent mod ell");
        std::swap(M[p], M[row]);
        uint64_t iv = 1, b = M[row][k], e = ell - 2;
        while (e) { if (e & 1) iv = iv * b % ell; b = b * b % ell; e >>= 1; }
        for (uint32_t c = 0; c <= r; ++c) M[row][c] = M[row][c] * iv % ell;
        for (uint32_t i = 0; i < g; ++i) {
            if (i == row || M[i][k] == 0) continue;
            const uint64_t f = M[i][k];
            for (uint32_t c = 0; c <= r; ++c) M[i][c] = (M[i][c] + ell * ell - f * M[row][c] % ell) % ell;
        }
        pivrow[k] = row;
        ++row;
    }
    for (uint32_t i = row; i < g; ++i) ENGINE_CHECK_MSG(M[i][r] == 0, "HeckeHidaTower: vector not in the image mod ell");
    std::vector<uint64_t> gam(r);
    for (uint32_t k = 0; k < r; ++k) gam[k] = M[pivrow[k]][r];
    return gam;
}

// Berkowitz, division-free: ascending coefficients of det(X - A) mod q (the shared kit, ZellK.h).
Poly Berkowitz(const std::vector<std::vector<uint64_t>>& A, uint64_t q) {
    zk::Ring R;   // zk::Charpoly reads R.q only
    R.q = q;
    return zk::Charpoly(R, A.size(), [&](size_t i, size_t j) { return A[i][j]; });
}

} // namespace

const RingMat& ModuleCell::at(uint64_t p) const {
    for (size_t i = 0; i < primes.size(); ++i)
        if (primes[i] == p) return op[i];
    throw std::logic_error("HeckeHidaTower: operator T_" + std::to_string(p) + " was not run");
}

namespace {

// One module run, keeping its objects (the cohomology is needed by arm (c)).
struct CellRun {
    std::unique_ptr<LocalSystemSimplicialComplexT<uint64_t>> lssc;
    std::unique_ptr<EquivariantCohomologyOnKT<uint64_t>> coh;
    ModuleCell cell;
};

// Under the caller's ring scope (hecke::ScopedCoefficientRing(K)).
// (NBCONFIRM S5) RunCellOn's per-operator cancellation point is live only inside ComputeTowerCell (the `hida_tower` verb):
// RunModuleCell also serves hecke_ring_algebra's LevelCache, the HeckeZ local reads and the census lattice, which keep their
// own (unaudited for cancellation) state, so the point stays dark there.  File-local, per thread.
thread_local int t_tower_points = 0;
struct TowerPoints {
    TowerPoints() { ++t_tower_points; }
    ~TowerPoints() { --t_tower_points; }
};

void RunCellOn(CellRun& out, const SimplicialComplex& Kx, const hecke::SL2ZTessellationTables& model, uint32_t n,
               uint64_t ell, uint32_t K, const std::vector<uint64_t>& hecke_primes, FibreRepresentation rep) {
    ENGINE_CHECK_MSG(ell >= 5 && n >= 1 && K >= 2, "HeckeHidaTower: the module route needs ell >= 5, n >= 1, K >= 2 (def:hidatower)");
    ENGINE_CHECK_MSG(!hecke_primes.empty() && hecke_primes[0] == ell, "HeckeHidaTower: the Hecke primes must start with ell");
    ENGINE_CHECK_MSG(rep == FibreRepresentation::SymN || rep == FibreRepresentation::HidaLatticeH,
                     "HeckeHidaTower: the fibre is Sym^n Z^2 or Hida's lattice L_n ([C] def:hidalattice)");
    out.cell.n_sym = n;
    out.cell.ell = ell;
    out.cell.K = K;
    out.cell.primes = hecke_primes;
    const engine::ModCtx<uint64_t> ctx = hecke::CoefficientRingContext<uint64_t>("HeckeHidaTower", ell, K);
    out.lssc.reset(new LocalSystemSimplicialComplexT<uint64_t>(
        Kx,
        rep == FibreRepresentation::HidaLatticeH ? hecke::GenerateSL2ZRelationsRingHida<uint64_t>(Kx, n, ctx)
                                                 : hecke::GenerateSL2ZRelationsRing<uint64_t>(Kx, n, ctx),
        n + 1, ctx.p));
    out.coh.reset(new EquivariantCohomologyOnKT<uint64_t>(out.lssc.get()));
    if (t_tower_points > 0) oc_util::CancelPoint("hida_tower.cohomology");
    out.coh->compute_cohomology();
    out.cell.exps = out.coh->module_degree(1).dec.exponents;
    for (uint64_t p : hecke_primes) {
        if (t_tower_points > 0) oc_util::CancelPoint("hida_tower.operator");   // (NBCONFIRM S5) per Hecke operator, outside the lock
        std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(p, 1, ell, n));
            tau.reset(new hecke::HeckeTauTableT<uint64_t>(Kx, model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnKT<uint64_t> opk(out.coh.get(), cosets.get(), tau.get(), rep);
        opk.compute();
        out.cell.op.push_back(FromDense(opk.H_matrix_module(1), ell, out.cell.exps));
    }
}

void BuildModel(std::unique_ptr<SimplicialComplex>& Kx, std::unique_ptr<hecke::SL2ZTessellationTables>& model) {
    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
    Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
    model.reset(new hecke::SL2ZTessellationTables(*Kx));
}

} // namespace

ModuleCell RunModuleCell(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& hecke_primes,
                         FibreRepresentation rep) {
    ENGINE_CHECK_MSG(ell >= 5 && n >= 1 && K >= 2, "HeckeHidaTower: the module route needs ell >= 5, n >= 1, K >= 2 (def:hidatower)");
    ENGINE_CHECK_MSG(!hecke_primes.empty() && hecke_primes[0] == ell, "HeckeHidaTower: the Hecke primes must start with ell");
    const hecke::ScopedCoefficientRing ring(K);
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    BuildModel(Kx, model);
    CellRun run;
    RunCellOn(run, *Kx, *model, n, ell, K, hecke_primes, rep);
    return run.cell;
}

uint64_t PowMod(uint64_t b, uint64_t e, uint64_t q) { return engine::scalar::powmod<uint64_t>(b, e, q); }

uint64_t EvalMod(const Poly& f, uint64_t x, uint64_t q) {
    uint64_t r = 0;
    for (size_t i = f.size(); i-- > 0;) r = AddMod(MulMod(r, x % q, q), f[i], q);
    return r;
}

Poly ReducePoly(const Poly& f, uint64_t q) {
    Poly g(f.size());
    for (size_t i = 0; i < f.size(); ++i) g[i] = f[i] % q;
    return g;
}

std::string PolyText(const Poly& f) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < f.size(); ++i) os << (i ? "," : "") << f[i];
    os << "]";
    return os.str();
}

Poly CharpolyFree(const RingMat& A) {
    const uint32_t r = A.n();
    uint32_t e = r ? A.row_exp[0] : 1;
    for (uint32_t x : A.row_exp) ENGINE_CHECK_MSG(x == e, "HeckeHidaTower: CharpolyFree needs a free module");
    const uint64_t q = Pow64(A.ell, e);
    std::vector<std::vector<uint64_t>> M(r, std::vector<uint64_t>(r));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) M[i][j] = A.at(i, j) % q;
    return Berkowitz(M, q);
}

uint32_t TopLayerRank(const RingMat& P, uint32_t e) {
    // the rows of P with a_i = e', mod ell, zero-padded to a square matrix for RankModEll
    const uint32_t g = P.n();
    RingMat top;
    top.ell = P.ell;
    top.row_exp.assign(g, 1);
    top.a.assign(static_cast<size_t>(g) * g, 0);
    uint32_t r = 0;
    for (uint32_t i = 0; i < g; ++i) {
        if (P.row_exp[i] != e) continue;
        for (uint32_t j = 0; j < g; ++j) top.a[static_cast<size_t>(r) * g + j] = P.at(i, j) % P.ell;
        ++r;
    }
    return hecke::ordinary::RankModEll(top);
}

FreeImage MakeFreeImage(const RingMat& P, uint32_t e) {
    const uint64_t ell = P.ell;
    const uint32_t g = P.n();
    for (uint32_t x : P.row_exp) ENGINE_CHECK_MSG(x >= 1 && x <= e, "HeckeHidaTower: module exponents must lie in [1, e']");
    FreeImage F;
    F.P = P;
    F.e = e;
    // basis: columns of P independent mod ell (Nakayama on the direct summand P M)
    std::vector<std::vector<uint64_t>> Bbar;
    for (uint32_t j = 0; j < g; ++j) {
        std::vector<uint64_t> col = Column(P, j);
        std::vector<std::vector<uint64_t>> trial = Bbar;
        trial.push_back(col);
        // rank test over F_ell
        std::vector<std::vector<uint64_t>> M(g, std::vector<uint64_t>(trial.size()));
        for (uint32_t i = 0; i < g; ++i)
            for (size_t k = 0; k < trial.size(); ++k) M[i][k] = trial[k][i] % ell;
        uint32_t rk = 0;
        for (size_t k = 0; k < trial.size() && rk < g; ++k) {
            uint32_t p = rk;
            while (p < g && M[p][k] == 0) ++p;
            if (p == g) continue;
            std::swap(M[p], M[rk]);
            uint64_t iv = 1, b = M[rk][k], ex = ell - 2;
            while (ex) { if (ex & 1) iv = iv * b % ell; b = b * b % ell; ex >>= 1; }
            for (uint32_t i = 0; i < g; ++i) {
                if (i == rk || M[i][k] == 0) continue;
                const uint64_t f = M[i][k] * iv % ell;
                for (size_t c = 0; c < trial.size(); ++c) M[i][c] = (M[i][c] + ell * ell - f * M[rk][c] % ell) % ell;
            }
            ++rk;
        }
        if (rk == trial.size()) { Bbar = trial; F.B.push_back(col); }
    }
    const uint32_t r = F.rank();
    ENGINE_CHECK_MSG(r == hecke::ordinary::RankModEll(P), "HeckeHidaTower: basis size != rank_F_ell P");
    // (slot-12 WP-11, A4) the coordinates below are unique -- and the charpoly
    // well defined -- only if P M is free (cor:ordprecision(1), rmk:ordtorsionfree).
    ENGINE_CHECK_MSG(TopLayerRank(P, e) == r,
                     "HeckeHidaTower: P M is not free over Z/ell^{e'} (dim ell^{e'-1} P M != rank_F_ell P; "
                     "[C] cor:ordprecision(1), rmk:ordtorsionfree)");
    return F;
}

// coordinates of x in P M by the ell-adic digit solve
std::vector<uint64_t> FreeImage::coords(const std::vector<uint64_t>& x) const {
    const uint64_t ell = P.ell;
    const uint32_t g = P.n(), r = rank();
    const uint64_t q = Pow64(ell, e);
    std::vector<uint64_t> c(r, 0), z = x;
    uint64_t w = 1;
    for (uint32_t d = 0; d < e; ++d) {
        const std::vector<uint64_t> gam = SolveFl(B, z, ell);
        for (uint32_t k = 0; k < r; ++k) c[k] = AddMod(c[k], MulMod(w, gam[k], q), q);
        // z' = z - sum gam_k b_k  (in the module), then z <- P (z' / ell)
        std::vector<uint64_t> zp(g), u(g);
        for (uint32_t i = 0; i < g; ++i) {
            const uint64_t qi = Pow64(ell, P.row_exp[i]);
            u128 s = z[i] % qi;
            for (uint32_t k = 0; k < r; ++k) s = (s + qi * static_cast<u128>(ell) - static_cast<u128>(gam[k]) * (B[k][i] % qi) % qi) % qi;
            zp[i] = static_cast<uint64_t>(s);
            ENGINE_CHECK_MSG(zp[i] % ell == 0, "HeckeHidaTower: digit remainder not divisible by ell");
            u[i] = zp[i] / ell;
        }
        z = Apply(P, u);
        w *= ell;
    }
    // re-verify: sum c_k b_k == x
    for (uint32_t i = 0; i < g; ++i) {
        const uint64_t qi = Pow64(ell, P.row_exp[i]);
        u128 s = 0;
        for (uint32_t k = 0; k < r; ++k) s = (s + static_cast<u128>(c[k] % qi) * (B[k][i] % qi)) % qi;
        ENGINE_CHECK_MSG(static_cast<uint64_t>(s) == x[i] % qi, "HeckeHidaTower: coordinate solve failed to re-verify");
    }
    return c;
}

Mat MatrixOnImage(const FreeImage& F, const RingMat& T) {
    const uint32_t r = F.rank();
    Mat C(r, std::vector<uint64_t>(r));
    for (uint32_t j = 0; j < r; ++j) {
        const std::vector<uint64_t> cj = F.coords(Apply(T, F.B[j]));
        for (uint32_t i = 0; i < r; ++i) C[i][j] = cj[i];
    }
    return C;
}

Poly CharpolyOnImage(const RingMat& P, const RingMat& T, uint32_t e) {
    const FreeImage F = MakeFreeImage(P, e);
    return Berkowitz(MatrixOnImage(F, T), Pow64(P.ell, e));
}

namespace {

Reading ReadWith(const ModuleCell& c, const std::vector<RingMat>& ops, uint32_t e) {
    Reading rd;
    rd.e_target = e;
    const RingMat& U = ops[0];
    rd.generators = U.n();
    const RingMat P = hecke::ordinary::OrdinaryProjector(U);
    rd.ord_rank = hecke::ordinary::RankModEll(P);
    rd.idempotent = hecke::ordinary::Mul(P, P) == P;
    rd.image_free = TopLayerRank(P, e) == rd.ord_rank;
    rd.commutes = true;
    for (size_t i = 1; i < ops.size(); ++i) {
        const RingMat& T = ops[i];
        rd.commutes = rd.commutes && hecke::ordinary::Mul(P, T) == hecke::ordinary::Mul(T, P);
        rd.primes.push_back(c.primes[i]);
        rd.charpoly.push_back(CharpolyOnImage(P, T, e));
    }
    return rd;
}

} // namespace

Reading ReadOnModule(const ModuleCell& c, uint32_t e) {
    ENGINE_CHECK_MSG(e >= 1 && e <= c.K, "HeckeHidaTower: target precision out of range");
    std::vector<RingMat> ops;
    for (const RingMat& A : c.op) ops.push_back(hecke::ordinary::Reduce(A, e));
    return ReadWith(c, ops, e);
}

Reading ReadOnFreeBlock(const ModuleCell& c, uint32_t e, uint32_t h) {
    std::vector<uint32_t> F;
    for (uint32_t i = 0; i < c.exps.size(); ++i) {
        if (c.exps[i] == c.K) F.push_back(i);
        else ENGINE_CHECK_MSG(c.exps[i] + e <= c.K, "HeckeHidaTower: a torsion summand exceeds ell^{K - e'} (cor:ordprecision(3))");
    }
    ENGINE_CHECK_MSG(F.size() == h, "HeckeHidaTower: the free block does not have rank h (cor:ordprecision(3))");
    const uint64_t q = Pow64(c.ell, e);
    std::vector<RingMat> ops;
    for (const RingMat& A : c.op) {
        RingMat B;
        B.ell = c.ell;
        B.row_exp.assign(F.size(), e);
        for (uint32_t i : F)
            for (uint32_t j : F) B.a.push_back(A.at(i, j) % q);
        ops.push_back(B);
    }
    return ReadWith(c, ops, e);
}

uint32_t PredictedOrdinaryRank(uint32_t n, uint64_t ell) {
    if (ell == 11 && n % 10 == 0) return 3;
    if (ell == 13 && n % 12 == 10) return 3;
    return 1;
}

TowerCell ComputeTowerCell(uint32_t n, uint64_t ell, uint32_t m_max) {
    TowerCell t;
    t.n_sym = n;
    t.ell = ell;
    // (slot-12 WP-11, A10) one pre-pass: the precision-use guard alone, then K = m_max + e_max.
    hecke::ordinary::RequirePrecisionUse(n, ell, m_max);
    const TowerPoints points;   // (NBCONFIRM S5) the cell's cancellation points (stack-local results only)
    oc_util::CancelPoint("hida_tower.prepass");
    t.e_max = hecke::ordinary::TorsionExponentPrepass(n, ell).e_max;
    t.K = m_max + t.e_max;   // cor:ordprecision(2)
    if (t.K < 2)
        throw TowerPreconditionError("m + e_max = K = " + std::to_string(t.K) +
                                     " < 2: the module route of def:hidatower needs K >= 2 (e_max = " + std::to_string(t.e_max) +
                                     " from the torsion pre-pass)");
    oc_util::CancelPoint("hida_tower.module");
    t.h = hecke::ordinary::FreeRankH1(n);
    const ModuleCell mc = RunModuleCell(n, ell, t.K, {ell, 2, 3});
    t.exps = mc.exps;
    for (uint32_t m = 1; m <= m_max; ++m) {
        oc_util::CancelPoint("hida_tower.level");   // per level e'
        t.module_read.push_back(ReadOnModule(mc, m));
        t.free_read.push_back(ReadOnFreeBlock(mc, m, t.h));
    }
    return t;
}

std::string HidaTowerLine(const TowerCell& c, uint32_t m) {
    const Reading& r = c.module_read.at(m - 1);
    const uint64_t period = (c.ell - 1) * Pow64(c.ell, m - 1);
    std::ostringstream os;
    os << "HIDA_TOWER ell=" << c.ell << " class=" << (c.n_sym % period) << " m=" << m << " n=" << c.n_sym
       << " ord_rank=" << r.ord_rank << " charpoly_T2=" << PolyText(r.charpoly.at(0))
       << " charpoly_T3=" << PolyText(r.charpoly.at(1));
    return os.str();
}


// ===========================================================================
// (slot-13 WP-5) The module datum, (Hi8)'s decision procedure, arm (c).
// ===========================================================================
namespace {

uint64_t InvModP(uint64_t a, uint64_t ell) { return engine::scalar::invmod_fermat<uint64_t>(a, ell); }

using zk::FieldRank;   // rank over F_ell and the RREF nullspace basis (ZellK.h)

uint64_t DetModP(Mat A, uint64_t ell) {
    const size_t h = A.size();
    uint64_t det = 1;
    for (auto& r : A) for (auto& x : r) x %= ell;
    for (size_t c = 0; c < h; ++c) {
        size_t piv = h;
        for (size_t i = c; i < h; ++i) if (A[i][c]) { piv = i; break; }
        if (piv == h) return 0;
        if (piv != c) { std::swap(A[piv], A[c]); det = SubMod(0, det, ell); }
        det = MulMod(det, A[c][c], ell);
        const uint64_t iv = InvModP(A[c][c], ell);
        for (size_t i = c + 1; i < h; ++i) {
            if (!A[i][c]) continue;
            const uint64_t f = MulMod(A[i][c], iv, ell);
            for (size_t j = c; j < h; ++j) A[i][j] = SubMod(A[i][j], MulMod(f, A[c][j], ell), ell);
        }
    }
    return det;
}

Mat MatMul(const Mat& A, const Mat& B, uint64_t q) {
    zk::Ring R;   // zk::MatMul reads R.q only
    R.q = q;
    return zk::MatMul(R, A, B);
}

// The linear system P A_p - B_p P = 0 on vec(P) (index i*h + k), (np*h*h) x (h*h), mod q.
std::vector<std::vector<uint64_t>> HomSystem(const std::vector<Mat>& A, const std::vector<Mat>& B, uint32_t h, uint64_t q) {
    std::vector<std::vector<uint64_t>> L;
    for (size_t t = 0; t < A.size(); ++t)
        for (uint32_t i = 0; i < h; ++i)
            for (uint32_t j = 0; j < h; ++j) {
                std::vector<uint64_t> row(static_cast<size_t>(h) * h, 0);
                for (uint32_t k = 0; k < h; ++k) {
                    row[static_cast<size_t>(i) * h + k] = AddMod(row[static_cast<size_t>(i) * h + k], A[t][k][j], q);
                    row[static_cast<size_t>(k) * h + j] = SubMod(row[static_cast<size_t>(k) * h + j], B[t][i][k] % q, q);
                }
                L.push_back(row);
            }
    return L;
}

// Generators of the solution module over Z/ell^m, and its length.
std::vector<std::vector<uint64_t>> HomGenerators(const std::vector<Mat>& A, const std::vector<Mat>& B, uint32_t h,
                                                 uint64_t ell, uint32_t m, uint32_t* length) {
    const uint64_t q = Pow64(ell, m);
    const std::vector<std::vector<uint64_t>> L = HomSystem(A, B, h, q);
    const uint32_t nv = h * h;
    std::vector<std::vector<uint64_t>> gens;
    if (m == 1) {
        const uint32_t rk = FieldRank(L, nv, ell, &gens);
        *length = nv - rk;
        return gens;
    }
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, m);
    engine::DenseBuf<uint64_t> Lb(static_cast<uint32_t>(L.size()), nv);
    for (uint32_t i = 0; i < L.size(); ++i)
        for (uint32_t j = 0; j < nv; ++j) Lb.at(i, j) = L[i][j];
    const auto ker = engine::elim::kernel_module(ctx, engine::cview(Lb));
    *length = engine::elim::length(ctx, ker);
    for (uint32_t r = 0; r < ker.rows(); ++r) {
        std::vector<uint64_t> v(nv);
        for (uint32_t j = 0; j < nv; ++j) v[j] = ker.H.at(r, j);
        gens.push_back(v);
    }
    return gens;
}

std::string MatText(const Mat& A) {
    std::string s = "[";
    for (size_t i = 0; i < A.size(); ++i) {
        s += (i ? ",[" : "[");
        for (size_t j = 0; j < A[i].size(); ++j) s += (j ? "," : "") + std::to_string(A[i][j]);
        s += "]";
    }
    return s + "]";
}

} // namespace

ModuleDatum ReadModuleDatum(const ModuleCell& c, uint32_t m) {
    ENGINE_CHECK_MSG(m >= 1 && m <= c.K, "HeckeHidaTower: ReadModuleDatum: level m out of range");
    ModuleDatum d;
    d.ell = c.ell;
    d.m = m;
    const RingMat U = hecke::ordinary::Reduce(c.op.at(0), m);
    const RingMat P = hecke::ordinary::OrdinaryProjector(U);
    const FreeImage F = MakeFreeImage(P, m);
    d.h = F.rank();
    const uint64_t q = Pow64(c.ell, m);
    for (size_t i = 1; i < c.op.size(); ++i) {
        const RingMat T = hecke::ordinary::Reduce(c.op[i], m);
        ENGINE_CHECK_MSG(hecke::ordinary::Mul(P, T) == hecke::ordinary::Mul(T, P),
                         "HeckeHidaTower: T_p does not commute with the ordinary projector");
        d.primes.push_back(c.primes[i]);
        d.A.push_back(MatrixOnImage(F, T));
        d.charpoly.push_back(Berkowitz(d.A.back(), q));
    }
    return d;
}

std::string IsoDecision::text() const {
    std::ostringstream os;
    os << "iso=" << (isomorphic ? 1 : 0) << " precheck=" << (decided_by_precheck ? 1 : 0) << " len_hom=(" << len_hom_aa
       << "," << len_hom_ab << "," << len_hom_bb << ") dim_hom_bar=" << dim_hom_bar << " grid_points=" << grid_points;
    if (isomorphic) os << " P=" << MatText(certificate);
    return os.str();
}

IsoDecision DecideIsomorphic(const ModuleDatum& a, const ModuleDatum& b) {
    ENGINE_CHECK_MSG(a.ell == b.ell && a.m == b.m && a.primes == b.primes,
                     "HeckeHidaTower: DecideIsomorphic needs two data over one Z/ell^m with one prime set");
    IsoDecision r;
    const uint64_t ell = a.ell;
    const uint32_t m = a.m, h = a.h;
    const uint64_t q = Pow64(ell, m);
    if (a.h != b.h || a.charpoly != b.charpoly) {
        r.decided_by_precheck = true;   // unequal rank or characteristic polynomials: not isomorphic
        return r;
    }
    if (h == 0) {
        r.isomorphic = true;            // the zero module
        return r;
    }
    (void)HomGenerators(a.A, a.A, h, ell, m, &r.len_hom_aa);
    (void)HomGenerators(b.A, b.A, h, ell, m, &r.len_hom_bb);
    const std::vector<std::vector<uint64_t>> gens = HomGenerators(a.A, b.A, h, ell, m, &r.len_hom_ab);
    if (r.len_hom_ab != r.len_hom_aa || r.len_hom_ab != r.len_hom_bb) {
        r.decided_by_precheck = true;   // Hom(M, M') is not End(M): not isomorphic
        return r;
    }
    // Hom-bar: an F_ell-basis among the generators' reductions, with their Z/ell^m lifts.
    std::vector<std::vector<uint64_t>> basis_bar, lifts;
    for (const auto& g : gens) {
        std::vector<std::vector<uint64_t>> trial = basis_bar;
        std::vector<uint64_t> gb(g.size());
        for (size_t i = 0; i < g.size(); ++i) gb[i] = g[i] % ell;
        trial.push_back(gb);
        if (FieldRank(trial, h * h, ell, nullptr) == trial.size()) { basis_bar = trial; lifts.push_back(g); }
    }
    const uint32_t s = static_cast<uint32_t>(basis_bar.size());
    r.dim_hom_bar = s;
    // det on Hom-bar has degree h: the grid {0..h}^s decides it when h < ell; else all of F_ell^s.
    const uint64_t side = (h < ell) ? static_cast<uint64_t>(h) + 1 : ell;
    u128 total = 1;
    for (uint32_t i = 0; i < s; ++i) {
        total *= side;
        ENGINE_CHECK_MSG(total <= (u128(1) << 22),
                         "HeckeHidaTower: (Hi8) decision grid exceeds 2^22 points -- refused by name, never inconclusive");
    }
    std::vector<uint64_t> c(s, 0);
    for (u128 pt = 0; pt < total; ++pt) {
        // odometer
        if (pt > 0) {
            for (uint32_t i = 0; i < s; ++i) {
                if (++c[i] < side) break;
                c[i] = 0;
            }
        }
        ++r.grid_points;
        Mat Pb(h, std::vector<uint64_t>(h, 0));
        for (uint32_t t = 0; t < s; ++t)
            if (c[t])
                for (uint32_t i = 0; i < h; ++i)
                    for (uint32_t k = 0; k < h; ++k)
                        Pb[i][k] = AddMod(Pb[i][k], MulMod(c[t], basis_bar[t][static_cast<size_t>(i) * h + k], ell), ell);
        if (DetModP(Pb, ell) == 0) continue;
        // The certificate over Z/ell^m: the same combination of the lifts.
        Mat P(h, std::vector<uint64_t>(h, 0));
        for (uint32_t t = 0; t < s; ++t)
            if (c[t])
                for (uint32_t i = 0; i < h; ++i)
                    for (uint32_t k = 0; k < h; ++k)
                        P[i][k] = AddMod(P[i][k], MulMod(c[t], lifts[t][static_cast<size_t>(i) * h + k], q), q);
        for (size_t t = 0; t < a.A.size(); ++t)
            ENGINE_CHECK_MSG(MatMul(P, a.A[t], q) == MatMul(b.A[t], P, q),
                             "HeckeHidaTower: (Hi8) certificate fails P A_p = A'_p P over Z/ell^m");
        ENGINE_CHECK_MSG(DetModP(P, ell) != 0, "HeckeHidaTower: (Hi8) certificate is not invertible");
        r.isomorphic = true;
        r.certificate = P;
        return r;
    }
    return r;   // det vanishes on the whole grid: identically zero on Hom-bar, not isomorphic
}

uint64_t ModuleClassDigest(const ModuleDatum& d) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, d.h);
    h = engine::FnvU64(h, d.m);
    for (size_t i = 0; i < d.charpoly.size(); ++i) {
        h = engine::FnvU64(h, d.primes[i]);
        h = engine::FnvU64(h, d.charpoly[i].size());
        for (uint64_t x : d.charpoly[i]) h = engine::FnvU64(h, x);
    }
    return h;
}

LatticePair RunLatticePair(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint64_t>& hecke_primes) {
    const hecke::ScopedCoefficientRing ring(K);
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    BuildModel(Kx, model);
    CellRun rs, rh;
    RunCellOn(rs, *Kx, *model, n, ell, K, hecke_primes, FibreRepresentation::SymN);
    RunCellOn(rh, *Kx, *model, n, ell, K, hecke_primes, FibreRepresentation::HidaLatticeH);
    LatticePair lp;
    lp.n_sym = n;
    lp.ell = ell;
    lp.K = K;
    // D = diag(binom(n, j)) mod ell^K (Pascal's rule).
    const uint64_t Q = Pow64(ell, K);
    const uint32_t d = n + 1;
    std::vector<uint64_t> binom(1, 1 % Q);
    for (uint32_t r = 1; r <= n; ++r) {
        std::vector<uint64_t> nb(r + 1, 0);
        nb[0] = nb[r] = 1 % Q;
        for (uint32_t j = 1; j < r; ++j) nb[j] = AddMod(binom[j - 1], binom[j], Q);
        binom.swap(nb);
    }
    // D applied fibrewise to the Hida-lattice H^1 class representatives.
    const engine::DenseBuf<uint64_t>& Hb = rh.coh->H_module_basis(1);
    engine::DenseBuf<uint64_t> Y(Hb.rows, Hb.cols);
    for (uint32_t row = 0; row < Hb.rows; ++row)
        for (uint32_t col = 0; col < Hb.cols; ++col) Y.at(row, col) = MulMod(binom[row % d], Hb.at(row, col), Q);
    const std::vector<std::vector<uint64_t>> co = rs.coh->H_class_coordinates(1, engine::cview(Y));
    const uint32_t gs = static_cast<uint32_t>(rs.cell.exps.size()), gh = static_cast<uint32_t>(rh.cell.exps.size());
    ENGINE_CHECK(co.size() == gh);
    lp.Dstar.assign(gs, std::vector<uint64_t>(gh, 0));
    for (uint32_t j = 0; j < gh; ++j) {
        ENGINE_CHECK(co[j].size() == gs);
        for (uint32_t i = 0; i < gs; ++i) lp.Dstar[i][j] = co[j][i] % Pow64(ell, rs.cell.exps[i]);
    }
    // Hecke equivariance of D_*: D A^H_p = A_p D, entrywise mod ell^{b_i}.
    bool eq = true;
    for (size_t t = 0; t < hecke_primes.size() && eq; ++t) {
        const RingMat& AS = rs.cell.op[t];
        const RingMat& AH = rh.cell.op[t];
        for (uint32_t i = 0; i < gs && eq; ++i) {
            const uint64_t qi = Pow64(ell, rs.cell.exps[i]);
            for (uint32_t j = 0; j < gh; ++j) {
                u128 l = 0, r = 0;
                for (uint32_t k = 0; k < gh; ++k) l = (l + static_cast<u128>(lp.Dstar[i][k]) * AH.at(k, j)) % qi;
                for (uint32_t k = 0; k < gs; ++k) r = (r + static_cast<u128>(AS.at(i, k)) * lp.Dstar[k][j]) % qi;
                if (l != r) { eq = false; break; }
            }
        }
    }
    lp.Dstar_equivariant = eq;
    lp.sym = std::move(rs.cell);
    lp.hida = std::move(rh.cell);
    return lp;
}

DstarReading ReadDstar(const LatticePair& lp, uint32_t m) {
    ENGINE_CHECK_MSG(m >= 1 && m <= lp.K, "HeckeHidaTower: ReadDstar: level m out of range");
    const uint64_t ell = lp.ell;
    DstarReading rd;
    rd.m = m;
    const RingMat PS = hecke::ordinary::OrdinaryProjector(hecke::ordinary::Reduce(lp.sym.op.at(0), m));
    const RingMat PH = hecke::ordinary::OrdinaryProjector(hecke::ordinary::Reduce(lp.hida.op.at(0), m));
    const FreeImage FS = MakeFreeImage(PS, m), FH = MakeFreeImage(PH, m);
    ENGINE_CHECK_MSG(FS.rank() == FH.rank(), "HeckeHidaTower: the ordinary ranks of the two lattices differ (they are lattice-independent)");
    rd.h = FS.rank();
    const uint32_t gs = PS.n();
    Mat Delta(rd.h, std::vector<uint64_t>(rd.h, 0));
    bool inside = true;
    for (uint32_t k = 0; k < rd.h; ++k) {
        std::vector<uint64_t> y(gs, 0);
        for (uint32_t i = 0; i < gs; ++i) {
            const uint64_t qi = Pow64(ell, PS.row_exp[i]);
            u128 acc = 0;
            for (uint32_t j = 0; j < FH.B[k].size(); ++j) acc = (acc + static_cast<u128>(lp.Dstar[i][j] % qi) * (FH.B[k][j] % qi)) % qi;
            y[i] = static_cast<uint64_t>(acc);
        }
        inside = inside && Apply(PS, y) == y;
        const std::vector<uint64_t> c = FS.coords(y);
        for (uint32_t i = 0; i < rd.h; ++i) Delta[i][k] = c[i];
    }
    rd.image_in_ordinary = inside;
    std::vector<std::vector<uint64_t>> rows(Delta.begin(), Delta.end());
    rd.rank_mod_ell = FieldRank(rows, rd.h, ell, nullptr);
    rd.isomorphism = rd.rank_mod_ell == rd.h;
    return rd;
}

} // namespace hida
} // namespace hecke_ext
