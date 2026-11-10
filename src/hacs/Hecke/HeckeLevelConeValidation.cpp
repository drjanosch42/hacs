// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 23, LES-2) HeckeLevelConeValidation -- see the header, [C] v1.22.0-pre.3 sec:nebles and
// docs/implementation/slot23/LES2_predictions.md (committed before the first run).
#include "HeckeLevelConeValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "HeckeCosetTables.h"
#include "HeckeGammaH.h"
#include "HeckeLevelCone.h"
#include "HeckeLevelEll.h"
#include "HeckeLevelInfRes.h"
#include "HeckeLevelN.h"
#include "HeckeLevelPair.h"
#include "HeckeNebentypus.h"
#include "HeckeOperatorFamily.h"   // hecke::tau_construction_mutex()
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/BlockKernels.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/Timing.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using oc_util::Seconds;
using engine::scalar::gcd;
using engine::scalar::powmod;

using hecke::Mat22;
using namespace hecke::level;
using namespace hecke::level::les;
using BM = engine::BlockMat<uint32_t>;
using Buf = engine::DenseBuf<uint32_t>;
using engine::Layout;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 80) std::cout << "  FAIL: " << what << "\n";
        }
    }
};


// ---- the fixed model and the level-1 Hecke data ----------------------------------------------------------------
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
struct HeckeDatum {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
};
const HeckeDatum& Datum(uint64_t p, uint32_t ell, uint32_t n) {
    static std::map<std::tuple<uint64_t, uint32_t, uint32_t>, std::unique_ptr<HeckeDatum>> cache;
    auto& slot = cache[std::make_tuple(p, ell, n)];
    if (!slot) {
        Model& M = TheModel();
        slot.reset(new HeckeDatum);
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        slot->cosets.reset(new hecke::HeckeCosetTables(p, 1, ell, n));
        slot->tau.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *slot->cosets, ell, n));
    }
    return *slot;
}

// ---- levels and characters ----------------------------------------------------------------------------------------
std::unique_ptr<CosetLevel> G0(uint64_t N) {
    if (N == 1) return std::make_unique<FullLevel>();
    return std::make_unique<Gamma0LevelN>(N);
}
std::unique_ptr<GammaHLevelN> GH(uint64_t N, const std::function<bool(uint64_t)>& pred) {
    std::vector<uint64_t> elems;
    for (uint64_t u = 1; u < N; ++u)
        if (gcd<uint64_t>(u, N) == 1 && pred(u)) elems.push_back(u);
    return std::make_unique<GammaHLevelN>(N, elems);
}
std::unique_ptr<GammaHLevelN> G1(uint64_t N) { return GH(N, [N](uint64_t u) { return u % N == 1 % N; }); }

CharacterDatum Chi(uint64_t N, uint32_t p, const std::function<uint32_t(uint64_t)>& f, const std::string& label) {
    CharacterDatum c;
    c.N = N;
    c.table.assign(N, 0);
    for (uint64_t u = 0; u < N; ++u)
        if (N == 1 || gcd<uint64_t>(u, N) == 1) c.table[u] = f(u) % p;
    c.label = label;
    return c;
}
CharacterDatum Triv(uint64_t N, uint32_t p) { return Chi(N, p, [](uint64_t) { return 1u; }, "1"); }
CharacterDatum Legendre5(uint64_t N, uint32_t p) {
    return Chi(N, p, [p](uint64_t u) { const uint64_t r = u % 5; return (r == 1 || r == 4) ? 1u : p - 1; }, "(./5)");
}
CharacterDatum Chi4mod5(uint64_t N) { return Chi(N, 5, [](uint64_t u) { return static_cast<uint32_t>(u % 5); }, "chi_4"); }
CharacterDatum Omega(uint64_t p, uint32_t i) {      // omega^i on (Z/p)^x into F_p: u -> u^i mod p
    return Chi(p, static_cast<uint32_t>(p), [p, i](uint64_t u) { return static_cast<uint32_t>(powmod<uint64_t>(u, i, p)); },
               "omega^" + std::to_string(i));
}

// ---- dense helpers (row-major) ------------------------------------------------------------------------------------
struct Mat {
    uint32_t r = 0, c = 0;
    std::vector<uint32_t> a;
    Mat() = default;
    Mat(uint32_t rr, uint32_t cc) : r(rr), c(cc), a(static_cast<size_t>(rr) * cc, 0) {}
    uint32_t& at(uint32_t i, uint32_t j) { return a[static_cast<size_t>(i) * c + j]; }
    uint32_t at(uint32_t i, uint32_t j) const { return a[static_cast<size_t>(i) * c + j]; }
};
Mat OfBlock(const BM& M) {
    Mat out(M.rows(), M.cols());
    if (M.rows() == 0 || M.cols() == 0) return out;
    const Buf d = M.to_dense(Layout::RowMajor);
    for (uint32_t i = 0; i < out.r; ++i)
        for (uint32_t j = 0; j < out.c; ++j) out.at(i, j) = d.at(i, j);
    return out;
}
Mat OfFlat(const FlatMatrix& F) {
    Mat out(F.get_rows(), F.get_cols());
    for (uint32_t i = 0; i < out.r; ++i)
        for (uint32_t j = 0; j < out.c; ++j) out.at(i, j) = F.get_entry(i, j);
    return out;
}
// A (block) times X (dense): through the block kernel.
Mat BMul(const BM& A, const Mat& X) {
    ENGINE_CHECK_MSG(A.cols() == X.r, "BMul: shapes");
    Mat Y(A.rows(), X.c);
    if (A.rows() == 0 || X.c == 0 || A.cols() == 0) return Y;
    Buf Xb(X.r, X.c, Layout::ColMajor), Yb(A.rows(), X.c, Layout::ColMajor);
    for (uint32_t i = 0; i < X.r; ++i)
        for (uint32_t j = 0; j < X.c; ++j) Xb.at(i, j) = X.at(i, j);
    engine::blk::gemm_dense<uint32_t>(A, engine::cview(Xb), engine::view(Yb));
    for (uint32_t i = 0; i < Y.r; ++i)
        for (uint32_t j = 0; j < Y.c; ++j) Y.at(i, j) = Yb.at(i, j);
    return Y;
}
Mat Mul(uint32_t p, const Mat& A, const Mat& B) {
    ENGINE_CHECK_MSG(A.c == B.r, "Mul: shapes");
    Mat C(A.r, B.c);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t k = 0; k < A.c; ++k) {
            const uint64_t x = A.at(i, k);
            if (x == 0) continue;
            for (uint32_t j = 0; j < B.c; ++j) C.at(i, j) = static_cast<uint32_t>((C.at(i, j) + x * B.at(k, j)) % p);
        }
    return C;
}
Mat Sub(uint32_t p, const Mat& A, const Mat& B) {
    Mat C(A.r, A.c);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = static_cast<uint32_t>((A.a[i] + p - B.a[i]) % p);
    return C;
}
uint64_t Differ(const Mat& A, const Mat& B) {
    if (A.r != B.r || A.c != B.c) return UINT64_MAX;
    uint64_t d = 0;
    for (size_t i = 0; i < A.a.size(); ++i) d += (A.a[i] != B.a[i]);
    return d;
}
bool IsZero(const Mat& A) { for (uint32_t x : A.a) if (x) return false; return true; }
uint32_t Rank(uint32_t p, const Mat& A) {
    if (A.r == 0 || A.c == 0) return 0;
    Buf b(A.r, A.c, Layout::RowMajor);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) b.at(i, j) = A.at(i, j);
    const engine::ModCtx<uint32_t> ctx(p);
    return engine::elim::rref<uint32_t>(ctx, engine::view(b)).rank;
}
Mat RowsSlice(const Mat& X, uint32_t r0, uint32_t nr) {
    Mat Y(nr, X.c);
    std::copy(X.a.begin() + static_cast<size_t>(r0) * X.c, X.a.begin() + static_cast<size_t>(r0 + nr) * X.c, Y.a.begin());
    return Y;
}
Mat Stack(const Mat& top, const Mat& bot, uint32_t ncols) {
    Mat Y(top.r + bot.r, ncols);
    std::copy(top.a.begin(), top.a.end(), Y.a.begin());
    std::copy(bot.a.begin(), bot.a.end(), Y.a.begin() + top.a.size());
    return Y;
}
Mat ApplyMap(const BlockCochainMap& F, const Mat& X) {
    ENGINE_CHECK_MSG(F.src_dim() == X.r, "ApplyMap: shapes");
    Mat Y(F.dst_dim(), X.c);
    if (X.c == 0 || X.r == 0) return Y;
    Y.a = F.apply(X.a, X.c);
    return Y;
}
// det(X - A) low -> high over F_p.
std::vector<uint64_t> Charpoly(uint32_t p, const Mat& A) {
    if (A.r == 0) return {1};
    HMat H;
    H.rows = A.r; H.cols = A.c; H.modulus = p; H.ell = p; H.field = true;
    H.row_exp.assign(A.r, 1);
    H.a.assign(A.a.begin(), A.a.end());
    return HCharpoly(H);
}
std::vector<uint64_t> PolyMul(uint32_t p, const std::vector<uint64_t>& f, const std::vector<uint64_t>& g) {
    std::vector<uint64_t> h(f.size() + g.size() - 1, 0);
    for (size_t i = 0; i < f.size(); ++i)
        for (size_t j = 0; j < g.size(); ++j) h[i + j] = (h[i + j] + f[i] * g[j]) % p;
    return h;
}

// ---- one cell -------------------------------------------------------------------------------------------------------
struct Pred {
    bool have = false;
    uint32_t h0A = 0, h1A = 0, h0B = 0, h1B = 0, rank1 = 0, ker1 = 0, H1c = 0, H2c = 0;
};
Pred P8(uint32_t h0A, uint32_t h1A, uint32_t h0B, uint32_t h1B, uint32_t rank1, uint32_t ker1, uint32_t H1c, uint32_t H2c) {
    Pred p;
    p.have = true;
    p.h0A = h0A; p.h1A = h1A; p.h0B = h0B; p.h1B = h1B; p.rank1 = rank1; p.ker1 = ker1; p.H1c = H1c; p.H2c = H2c;
    return p;
}
Pred Split(uint32_t h0A, uint32_t h1A, uint32_t h0B, uint32_t h1B) { return P8(h0A, h1A, h0B, h1B, h1A, 0, h0B - h0A, h1B - h1A); }

struct HeckeOp {
    uint64_t p = 0;
    std::function<std::vector<Mat22>(uint64_t)> transB;   // r_t of B''\B (inside B)
    uint32_t index = 0;                                    // p + 1 (T_p) or p (same-level U_p)
    std::string label;
};

struct CellIn {
    std::string tag;
    const CosetLevel* A = nullptr;
    const CosetLevel* B = nullptr;
    CharacterDatum chiA, chiB;
    uint32_t ell = 0, n = 0;
    Pred pred;
    std::vector<HeckeOp> ops;
    bool level_raise = false;          // (VI) U_p with p | N_A, p N_A | N_B (chi = 1)
    uint64_t lr_p = 0, lr_NA = 0, lr_NB = 0;
    int lr_defect_predicted = -1;      // 1: predicted non-zero, -1: recorded only
    bool tags = false;                 // (IV): print and gate the model tags
    bool tagA_group = false, tagB_group = false;
};

struct CellOut {
    std::unique_ptr<RestrictionFibreMap> R;
    std::unique_ptr<LevelModel> MA, MB;
    std::unique_ptr<LevelPairCone> C;
    LevelPairCone::SixTerm st;
    bool ok = false;
};

struct Totals {
    uint32_t cells = 0, cells_ok = 0, hecke_squares = 0, hecke_square_bad_entries = 0, families_bad = 0, charpoly_bad = 0;
    uint32_t zigzag_bad = 0, dalpha_bad = 0, lr_cells = 0, lr_bad_entries = 0;
    double max_cell_s = 0;
    std::string max_cell;
};

// Phi on the cone cochains of stored degree q: (T^A_q (+) T^B_{q-1}).
Mat ApplyPhi(const EquivariantMappingCone& C, const std::vector<BlockCochainMap>& TA, const std::vector<BlockCochainMap>& TB,
             uint32_t q, const Mat& X) {
    const uint32_t bp = C.bulk_part_dim(q), bd = C.boundary_part_dim(q);
    ENGINE_CHECK_MSG(X.r == bp + bd, "ApplyPhi: shapes");
    Mat top(bp, X.c), bot(bd, X.c);
    if (bp > 0) top = ApplyMap(TA.at(q), RowsSlice(X, 0, bp));
    if (bd > 0) bot = ApplyMap(TB.at(q - 1), RowsSlice(X, bp, bd));
    return Stack(top, bot, X.c);
}

// The operator on H^q of a model: P (T H).
Mat OnH(const EquivariantCohomologyOnK& H, const BlockCochainMap& T, uint32_t q) {
    const Mat Hq = OfBlock(H.H_eq_block(q));
    if (Hq.c == 0) return Mat(0, 0);
    return BMul(H.P_block(q), ApplyMap(T, Hq));
}

void HeckeArm(Checker& g, Totals& tot, const CellIn& in, CellOut& out, const HeckeOp& op) {
    const engine::ModCtx<uint32_t> ctx(in.ell);
    const SimplicialComplex& K = *TheModel().K;
    const HeckeDatum& dat = Datum(op.p, in.ell, in.n);
    const std::vector<Mat22> r = op.transB(op.p);
    CorrespondenceSpec sA{in.A, in.A, dat.cosets.get(), dat.tau.get(), r, op.label + "^A"};
    CorrespondenceSpec sB{in.B, in.B, dat.cosets.get(), dat.tau.get(), r, op.label + "^B"};
    VerifyTransversal(sA, op.index);   // the criterion of prop:levelconehecke (1): one transversal for both levels
    VerifyTransversal(sB, op.index);
    std::vector<BlockCochainMap> TA, TB;
    for (uint32_t q = 0; q <= 2; ++q) {
        TA.push_back(hecke::neb::BuildLevelCorrespondenceChi(ctx, sA, K, in.n, q, ScalarOf(in.chiA)));
        TB.push_back(hecke::neb::BuildLevelCorrespondenceChi(ctx, sB, K, in.n, q, ScalarOf(in.chiB)));
    }
    const EquivariantMappingCone& C = out.C->cone();
    const uint32_t p = in.ell;
    // (v) D Phi = Phi D on the stored equivariant cone basis, every stored degree.
    uint64_t bad = 0;
    for (uint32_t q = 0; q <= C.q_max(); ++q) {
        const Mat X = OfBlock(C.Ceq_block(q));
        if (X.c == 0) continue;
        const Mat lhs = BMul(C.D_block(q), ApplyPhi(C, TA, TB, q, X));
        const Mat DX = BMul(C.D_block(q), X);
        const Mat rhs = (C.flat_dim(q + 1) > 0) ? ApplyPhi(C, TA, TB, q + 1, DX) : DX;
        bad += Differ(lhs, rhs);
    }
    ++tot.hecke_squares;
    tot.hecke_square_bad_entries += static_cast<uint32_t>(std::min<uint64_t>(bad, 1u << 30));
    g.check(bad == 0, "(H) " + in.tag + " " + op.label + ": D_cone Phi = Phi D_cone on the stored basis (" + std::to_string(bad) +
                          " differing entries)");
    // the operators on the ten terms, the three families, the alternating characteristic polynomial.
    std::vector<Mat> Tc(C.q_max() + 1), TAH(3), TBH(3);
    for (uint32_t q = 0; q <= C.q_max(); ++q) {
        const Mat Hc = OfBlock(C.H_block(q));
        Tc[q] = Hc.c ? BMul(C.P_block(q), ApplyPhi(C, TA, TB, q, Hc)) : Mat(0, 0);
    }
    for (uint32_t q = 0; q <= 2; ++q) {
        TAH[q] = OnH(*out.MA->H, TA[q], q);
        TBH[q] = OnH(*out.MB->H, TB[q], q);
    }
    bool fam = true;
    for (uint32_t q = 0; q <= 2; ++q) {
        if (C.pi_built(q) && C.pi(q).get_rows() && C.pi(q).get_cols()) {
            const Mat pi = OfFlat(C.pi(q));
            fam = fam && Differ(Mul(p, pi, Tc[q]), Mul(p, TAH[q], pi)) == 0;
        }
        if (C.r_built(q) && C.r(q).get_rows() && C.r(q).get_cols()) {
            const Mat rr = OfFlat(C.r(q));
            fam = fam && Differ(Mul(p, rr, TAH[q]), Mul(p, TBH[q], rr)) == 0;
        }
        if (C.iota_built(q) && C.iota(q).get_rows() && C.iota(q).get_cols()) {
            const Mat io = OfFlat(C.iota(q));
            fam = fam && Differ(Mul(p, io, TBH[q]), Mul(p, Tc[q + 1], io)) == 0;
        }
    }
    if (!fam) ++tot.families_bad;
    g.check(fam, "(H) " + in.tag + " " + op.label + ": pi, res, iota intertwine Phi on cohomology");
    // terms in order H^0_c, H^0(A), H^0(B), H^1_c, ..., H^2(B), H^3_c: prod over even positions = prod over odd positions.
    std::vector<uint64_t> even{1}, odd{1};
    uint32_t pos = 0;
    for (uint32_t q = 0; q <= 3; ++q) {
        auto take = [&](const Mat& M) { (pos % 2 == 0 ? even : odd) = PolyMul(p, pos % 2 == 0 ? even : odd, Charpoly(p, M)); ++pos; };
        take(q <= C.q_max() ? Tc[q] : Mat(0, 0));
        if (q <= 2) { take(TAH[q]); take(TBH[q]); }
    }
    const bool cp = even == odd;
    if (!cp) ++tot.charpoly_bad;
    g.check(cp, "(H) " + in.tag + " " + op.label + ": the alternating product of the characteristic polynomials over eq:levelsixterm is 1");
}

void LevelRaiseArm(Checker& g, Totals& tot, const CellIn& in, CellOut& out) {
    const engine::ModCtx<uint32_t> ctx(in.ell);
    const SimplicialComplex& K = *TheModel().K;
    const uint64_t p = in.lr_p;
    const HeckeDatum& dat = Datum(p, in.ell, in.n);
    const Gamma0LevelN LA(in.lr_NA), LAp(p * in.lr_NA), LB(in.lr_NB), LBp(p * in.lr_NB);
    CorrespondenceSpec sUA{in.A, in.A, dat.cosets.get(), dat.tau.get(), Gamma0SubTransversalN(LA, LAp), "U_" + std::to_string(p) + "^A"};
    CorrespondenceSpec sUB{in.B, in.B, dat.cosets.get(), dat.tau.get(), Gamma0SubTransversalN(LB, LBp), "U_" + std::to_string(p) + "^B"};
    CorrespondenceSpec sDG{in.A, in.B, dat.cosets.get(), dat.tau.get(), {Mat22::I()}, "dg_g"};
    VerifyTransversal(sUA, static_cast<uint32_t>(p));
    VerifyTransversal(sUB, static_cast<uint32_t>(p));
    VerifyTransversal(sDG, 1);   // B'' = B cap g A g^{-1} = B (prop:levelconehecke (3))
    uint64_t bad = 0, square_cochain = 0;
    bool lhs_zero = true;
    Mat UAH, UBH;
    for (uint32_t q = 0; q <= 2; ++q) {
        const BlockCochainMap UA = hecke::neb::BuildLevelCorrespondenceChi(ctx, sUA, K, in.n, q, ScalarOf(in.chiA));
        const BlockCochainMap UB = hecke::neb::BuildLevelCorrespondenceChi(ctx, sUB, K, in.n, q, ScalarOf(in.chiB));
        const BlockCochainMap DG = hecke::neb::BuildLevelCorrespondenceChi(ctx, sDG, K, in.n, q, ScalarOf(in.chiA));
        const Mat X = OfBlock(out.MA->H->C_eq_block(q));
        if (X.c > 0) {
            const uint32_t Nq = K.get_num_q_simplices(q);
            Mat aX(Nq * out.MB->d(), X.c);
            aX.a = ApplyFibre(out.R->alpha(), Nq, X.a, X.c, ctx);
            const Mat lhs = ApplyMap(UB, aX);
            Mat rhs = ApplyMap(DG, X);
            for (uint32_t& v : rhs.a) v = static_cast<uint32_t>(static_cast<uint64_t>(v) * (p % in.ell) % in.ell);
            bad += Differ(lhs, rhs);
            lhs_zero = lhs_zero && IsZero(lhs);
            // (audit R2) the square itself on the stored basis: U^B alpha_* vs alpha_* U^A (the criterion of (1) fails, B <= A'')
            const Mat UAX = ApplyMap(UA, X);
            Mat aUAX(Nq * out.MB->d(), X.c);
            aUAX.a = ApplyFibre(out.R->alpha(), Nq, UAX.a, UAX.c, ctx);
            square_cochain += Differ(lhs, aUAX);
        }
        if (q == 1) {
            UAH = OnH(*out.MA->H, UA, 1);
            UBH = OnH(*out.MB->H, UB, 1);
        }
    }
    ++tot.lr_cells;
    tot.lr_bad_entries += static_cast<uint32_t>(std::min<uint64_t>(bad, 1u << 30));
    g.check(bad == 0, "(VI) " + in.tag + ": U^B_" + std::to_string(p) + " alpha_* = " + std::to_string(p) +
                          " dg_g on C^q_eq(L_A), q = 0, 1, 2 (" + std::to_string(bad) + " differing entries)");
    if (in.ell == p) g.check(lhs_zero, "(VI) " + in.tag + ": U^B_p alpha_* = 0 at ell = p");
    // (audit R2) NEG criterion_fails: at (ell = 7, n = 0) the cochain square U^B alpha_* = alpha_* U^A must fail
    // (prop:levelconehecke (3)(c) makes it fail already on H^1); recorded at the other level-raising cells.
    std::cout << "  LEVEL_RAISE_COCHAIN_SQUARE " << in.tag << ": U^B alpha_* vs alpha_* U^A on C^q_eq(L_A): " << square_cochain
              << " differing entries" << (in.lr_defect_predicted == 1 ? "" : " (recorded)") << "\n";
    if (in.lr_defect_predicted == 1) {
        std::cout << "  NEG criterion_fails " << (square_cochain > 0 ? "caught" : "NOT caught") << " (" << square_cochain
                  << " differing entries)\n";
        g.check(square_cochain > 0, "(NEG) criterion_fails caught at " + in.tag + " (U^B alpha_* != alpha_* U^A, prop:levelconehecke (3)(c))");
    }
    // the H^1 square, RECORDED, not gated (obl:levelconeexact (vi), rule A6; audit R3)
    const EquivariantMappingCone& C = out.C->cone();
    uint32_t defect_rank = 0;
    if (C.r_built(1) && C.r(1).get_rows() && C.r(1).get_cols()) {
        const Mat r1 = OfFlat(C.r(1));
        defect_rank = Rank(in.ell, Sub(in.ell, Mul(in.ell, r1, UAH), Mul(in.ell, UBH, r1)));
    }
    std::cout << "  LEVEL_RAISE_SQUARE " << in.tag << " U_" << p << " on H^1: rank(res U^A - U^B res) = " << defect_rank
              << (in.lr_defect_predicted == 1 ? std::string(" (recorded; the (LES9) prediction non-zero: ") + (defect_rank > 0 ? "met)" : "NOT met)")
                                              : std::string(" (recorded)"))
              << "\n";
}

bool RunCell(Checker& g, Totals& tot, const CellIn& in, CellOut& out) {
    const auto t0 = std::chrono::steady_clock::now();
    ++tot.cells;
    const engine::ModCtx<uint32_t> ctx(in.ell);
    const SimplicialComplex& K = *TheModel().K;
    try {
        out.R.reset(new RestrictionFibreMap(*in.A, in.chiA, *in.B, in.chiB, in.n, ctx));
        out.MA = BuildLevelModel(K, *in.A, in.chiA, in.n, ctx);
        out.MB = BuildLevelModel(K, *in.B, in.chiB, in.n, ctx);
        out.C.reset(new LevelPairCone(*out.MA, *out.MB, out.R->alpha(), in.tag));
        out.C->compute();
    } catch (const std::exception& e) {
        g.check(false, in.tag + ": the cone does not build: " + e.what());
        return false;
    }
    const EquivariantMappingCone& C = out.C->cone();
    std::vector<std::string> failures;
    const uint32_t cert_bad = C.certificate().verify(failures);
    g.check(cert_bad == 0, in.tag + ": the cone's certificate C1-C13 re-derived (" + std::to_string(cert_bad) + " failing items)");
    const LevelPairCone::SixTerm s = out.C->six_term();
    out.st = s;
    // (LES2) lem:levelconeexact
    g.check(C.q_max() == 3 && s.H0c == 0 && s.H3c == 0, in.tag + ": H^0(A,B) = H^3(A,B) = 0");
    g.check(s.h2A == 0 && s.h2B == 0, in.tag + ": H^2(A) = H^2(B) = 0 (prop:bredon (2))");
    g.check(s.rank_res0 == s.h0A, in.tag + ": res^0 injective");
    g.check(s.spots_exact, in.tag + ": every spot of eq:levelsixterm asserted and exact (" + std::to_string(s.spots_asserted) + " / " +
                               std::to_string(s.spots) + ")");
    g.check(s.H1c == s.h0B - s.h0A + s.ker_res1(), in.tag + ": dim H^1(A,B) = h0B - h0A + dim ker res^1");
    g.check(s.H2c == s.h1B - s.rank_res1, in.tag + ": dim H^2(A,B) = h1B - rank res^1");
    // d alpha_* = alpha_* d on the stored equivariant bases of L_A, q = 0, 1 (obl:levelconeexact (ii))
    {
        uint64_t bad = 0;
        const uint32_t dA = out.MA->d(), dB = out.MB->d();
        for (uint32_t q = 0; q <= 1; ++q) {
            const Mat X = OfBlock(out.MA->H->C_eq_block(q));
            if (X.c == 0) continue;
            const uint32_t Nq = K.get_num_q_simplices(q), Nq1 = K.get_num_q_simplices(q + 1);
            const std::vector<uint32_t> l = ApplyCoboundary(K, q, dB, ApplyFibre(out.R->alpha(), Nq, X.a, X.c, ctx), X.c, ctx);
            const std::vector<uint32_t> r = ApplyFibre(out.R->alpha(), Nq1, ApplyCoboundary(K, q, dA, X.a, X.c, ctx), X.c, ctx);
            for (size_t i = 0; i < l.size(); ++i) bad += (l[i] != r[i]);
        }
        if (bad) ++tot.dalpha_bad;
        g.check(bad == 0, in.tag + ": d alpha_* = alpha_* d on C^q_eq(L_A), q = 0, 1 (" + std::to_string(bad) + " differing entries)");
    }
    // the connecting map by the zig-zag: D_cone (z, 0) = (0, alpha z), [alpha z] = res^q [z]  (lem:levelconeexact (1))
    {
        uint64_t bad = 0;
        for (uint32_t q = 0; q <= 2; ++q) {
            const Mat Z = OfBlock(out.MA->H->H_eq_block(q));
            if (Z.c == 0) continue;
            const uint32_t bp = C.bulk_part_dim(q), bd = C.boundary_part_dim(q);
            const Mat Xc = Stack(Z, Mat(bd, Z.c), Z.c);
            ENGINE_CHECK(Z.r == bp);
            const Mat Y = BMul(C.D_block(q), Xc);
            const uint32_t bp1 = C.bulk_part_dim(q + 1), bd1 = C.boundary_part_dim(q + 1);
            bad += IsZero(RowsSlice(Y, 0, bp1)) ? 0 : 1;
            const Mat coords = BMul(out.MB->H->P_block(q), RowsSlice(Y, bp1, bd1));
            const Mat rq = C.r_built(q) ? OfFlat(C.r(q)) : Mat(coords.r, coords.c);
            const uint64_t d = Differ(coords, rq);
            bad += (d == UINT64_MAX) ? 1 : d;
        }
        if (bad) ++tot.zigzag_bad;
        g.check(bad == 0, in.tag + ": the connecting map re-derived by the zig-zag equals +res^q, q = 0, 1, 2");
    }
    // the predicted numbers
    if (in.pred.have) {
        const Pred& p = in.pred;
        const bool ok = s.h0A == p.h0A && s.h1A == p.h1A && s.h0B == p.h0B && s.h1B == p.h1B && s.rank_res1 == p.rank1 &&
                        s.ker_res1() == p.ker1 && s.H1c == p.H1c && s.H2c == p.H2c;
        std::ostringstream os;
        os << in.tag << ": got h0A=" << s.h0A << " h1A=" << s.h1A << " h0B=" << s.h0B << " h1B=" << s.h1B << " rank res1=" << s.rank_res1
           << " ker res1=" << s.ker_res1() << " H1(A,B)=" << s.H1c << " H2(A,B)=" << s.H2c << "; predicted " << p.h0A << " " << p.h1A << " "
           << p.h0B << " " << p.h1B << " " << p.rank1 << " " << p.ker1 << " " << p.H1c << " " << p.H2c;
        g.check(ok, os.str());
    }
    if (in.tags) {
        const ModelTag tA = ModelTagOf(TorsionOf(PermutationOf(*in.A)), in.ell);
        const ModelTag tB = ModelTagOf(TorsionOf(PermutationOf(*in.B)), in.ell);
        g.check((tA == ModelTag::kGroup) == in.tagA_group && (tB == ModelTag::kGroup) == in.tagB_group,
                in.tag + ": model= tags A:" + ModelTagName(tA) + " B:" + ModelTagName(tB) + " (prop:modeltorsion)");
    }
    for (const HeckeOp& op : in.ops) {
        try {
            HeckeArm(g, tot, in, out, op);
        } catch (const std::exception& e) {
            g.check(false, in.tag + " " + op.label + ": the Hecke square does not build: " + e.what());
        }
    }
    if (in.level_raise) {
        try {
            LevelRaiseArm(g, tot, in, out);
        } catch (const std::exception& e) {
            g.check(false, in.tag + ": the level-raising U_p arm does not build: " + e.what());
        }
    }
    const double dt = Seconds(t0);
    if (dt > tot.max_cell_s) { tot.max_cell_s = dt; tot.max_cell = in.tag; }
    ++tot.cells_ok;
    return true;
}

std::string CellTag(const std::string& arm, const CosetLevel& A, const CosetLevel& B, const std::string& chi, uint32_t ell, uint32_t n) {
    std::ostringstream os;
    os << "(" << arm << ") " << B.name() << " < " << A.name() << " chi=" << chi << " ell=" << ell << " n=" << n;
    return os.str();
}

HeckeOp TOp(uint64_t p, const std::function<std::vector<Mat22>(uint64_t)>& tr) {
    HeckeOp o;
    o.p = p; o.transB = tr; o.index = static_cast<uint32_t>(p + 1); o.label = "T_" + std::to_string(p);
    return o;
}

// ---- the arm (IV) table of out_predict.txt l. 136-185 (COMP, the tree complex) ------------------------------------------
struct ArmIVRow {
    uint32_t ell, pair, n;
    bool tagA_group, tagB_group;
    uint32_t h0A, h1A, h0B, h1B, rank1, ker1, H1c, H2c;
};
const ArmIVRow kArmIV[] = {
    {2, 0, 0, false, false, 1, 1, 1, 2, 1, 0, 0, 1},
    {2, 0, 1, false, false, 1, 2, 1, 3, 1, 1, 1, 2},
    {2, 0, 2, false, false, 2, 3, 2, 5, 2, 1, 1, 3},
    {2, 0, 3, false, false, 2, 4, 2, 6, 2, 2, 2, 4},
    {2, 0, 4, false, false, 3, 5, 3, 8, 3, 2, 2, 5},
    {2, 1, 0, false, false, 1, 1, 1, 3, 1, 0, 0, 2},
    {2, 1, 1, false, false, 0, 2, 0, 4, 2, 0, 0, 2},
    {2, 1, 2, false, false, 1, 3, 1, 7, 3, 0, 0, 4},
    {2, 1, 3, false, false, 1, 3, 1, 9, 3, 0, 0, 6},
    {2, 1, 4, false, false, 1, 5, 1, 11, 5, 0, 0, 6},
    {2, 2, 0, false, true, 1, 2, 1, 2, 2, 0, 0, 0},
    {2, 2, 1, false, true, 1, 3, 1, 3, 3, 0, 0, 0},
    {2, 2, 2, false, true, 2, 5, 2, 5, 5, 0, 0, 0},
    {2, 2, 3, false, true, 2, 6, 2, 6, 6, 0, 0, 0},
    {2, 2, 4, false, true, 3, 8, 3, 8, 8, 0, 0, 0},
    {2, 3, 0, false, false, 1, 0, 1, 0, 0, 0, 0, 0},
    {2, 3, 1, false, false, 0, 1, 0, 2, 1, 0, 0, 1},
    {2, 3, 2, false, false, 1, 1, 1, 2, 1, 0, 0, 1},
    {2, 3, 3, false, false, 1, 1, 2, 2, 1, 0, 1, 1},
    {2, 3, 4, false, false, 1, 2, 1, 4, 2, 0, 0, 2},
    {2, 4, 0, false, false, 1, 0, 1, 0, 0, 0, 0, 0},
    {2, 4, 1, false, false, 0, 1, 0, 1, 1, 0, 0, 0},
    {2, 4, 2, false, false, 1, 1, 1, 1, 1, 0, 0, 0},
    {2, 4, 3, false, false, 1, 1, 1, 3, 1, 0, 0, 2},
    {2, 4, 4, false, false, 1, 2, 1, 2, 2, 0, 0, 0},
    {3, 0, 0, true, true, 1, 1, 1, 2, 1, 0, 0, 1},
    {3, 0, 1, true, true, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 0, 2, true, true, 0, 2, 0, 3, 2, 0, 0, 1},
    {3, 0, 3, true, true, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 0, 4, true, true, 1, 3, 1, 6, 3, 0, 0, 3},
    {3, 1, 0, false, true, 1, 1, 1, 3, 1, 0, 0, 2},
    {3, 1, 1, false, true, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 1, 2, false, true, 1, 3, 1, 7, 2, 1, 1, 5},
    {3, 1, 3, false, true, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 1, 4, false, true, 2, 5, 2, 12, 3, 2, 2, 9},
    {3, 2, 0, true, true, 1, 2, 1, 2, 2, 0, 0, 0},
    {3, 2, 1, true, true, 0, 0, 0, 2, 0, 0, 0, 2},
    {3, 2, 2, true, true, 0, 3, 0, 3, 3, 0, 0, 0},
    {3, 2, 3, true, true, 0, 0, 0, 4, 0, 0, 0, 4},
    {3, 2, 4, true, true, 1, 6, 1, 6, 6, 0, 0, 0},
    {3, 3, 0, false, false, 1, 0, 1, 0, 0, 0, 0, 0},
    {3, 3, 1, false, false, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 3, 2, false, false, 0, 1, 0, 1, 1, 0, 0, 0},
    {3, 3, 3, false, false, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 3, 4, false, false, 1, 1, 1, 2, 1, 0, 0, 1},
    {3, 4, 0, false, true, 1, 0, 1, 0, 0, 0, 0, 0},
    {3, 4, 1, false, true, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 4, 2, false, true, 0, 1, 0, 3, 1, 0, 0, 2},
    {3, 4, 3, false, true, 0, 0, 0, 0, 0, 0, 0, 0},
    {3, 4, 4, false, true, 1, 1, 2, 3, 1, 0, 1, 2},
};
static_assert(sizeof(kArmIV) / sizeof(kArmIV[0]) == 50, "arm (IV): the 50 rows of out_predict.txt l. 136-185");

// ---- (slot 23, LES-3) the read-only hook HECKE_LEVEL_CONE_CELLS="A:B:ell:n[:chi],..." (LES0_spec.md section 4,
// LES3_predictions.md section 6): REPLACES the arms by the listed cells; each runs RunCell (certificate, exactness, zig-zag,
// d alpha = alpha d; no predicted numbers, no Hecke operators) and, at ell >= 5, Route B's six numbers, gated equal (LES5).
// Level tokens: G, G0.<N>, G1.<N>, GH.<N>.<u>.<u>..., Gp.<M>, G2, G3; chi: 1 (default), w<e> (u -> (u mod ell)^e, ell | N_A),
// leg<q> (the Legendre symbol mod a prime q | N_A), the same datum on B.  A malformed token is refused by position; nothing runs.
struct HookLevel {
    std::unique_ptr<CosetLevel> lvl;
    uint64_t chiN = 1;          // the character level (1 for Gamma, Gamma(M), Gamma^2, Gamma^3)
    bool gamma0_type = false;   // Gamma_0 / Gamma_H type (a character other than 1 is allowed)
};
bool HookPrime(uint64_t x) {
    if (x < 2) return false;
    for (uint64_t d = 2; d * d <= x; ++d)
        if (x % d == 0) return false;
    return true;
}
std::vector<std::string> HookSplit(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); } else cur += c;
    }
    out.push_back(cur);
    return out;
}
bool HookNumber(const std::string& s, uint64_t& v) {
    if (s.empty() || s.size() > 9) return false;
    v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    return true;
}
bool HookParseLevel(const std::string& s, HookLevel& out, std::string& why) {
    const std::vector<std::string> f = HookSplit(s, '.');
    uint64_t N = 0;
    if (s == "G") { out.lvl = std::make_unique<FullLevel>(); return true; }
    if (s == "G2" || s == "G3") { out.lvl.reset(new CongruenceLevel(CongruenceLevel::Commutator(s == "G2" ? 2 : 3))); return true; }
    if (f.size() == 2 && f[0] == "Gp" && HookNumber(f[1], N) && N >= 2 && N <= 30) {
        out.lvl.reset(new CongruenceLevel(CongruenceLevel::Principal(N)));
        return true;
    }
    if (f.size() == 2 && (f[0] == "G0" || f[0] == "G1") && HookNumber(f[1], N) && N >= 1 && N < 100000) {
        if (N == 1) { out.lvl = std::make_unique<FullLevel>(); return true; }
        out.chiN = N;
        out.gamma0_type = true;
        if (f[0] == "G0") out.lvl = std::make_unique<Gamma0LevelN>(N);
        else out.lvl = G1(N);
        return true;
    }
    if (f.size() >= 2 && f[0] == "GH" && HookNumber(f[1], N) && N >= 2 && N < 100000) {
        std::vector<uint64_t> gens;
        for (size_t i = 2; i < f.size(); ++i) {
            uint64_t u = 0;
            if (!HookNumber(f[i], u) || gcd<uint64_t>(u, N) != 1) { why = "GH generator '" + f[i] + "' is not a unit mod " + std::to_string(N); return false; }
            gens.push_back(u % N);
        }
        if (gens.empty()) gens.push_back(1);
        out.chiN = N;
        out.gamma0_type = true;
        out.lvl = std::make_unique<GammaHLevelN>(N, gens);
        return true;
    }
    why = "level '" + s + "' (expected G, G0.<N>, G1.<N>, GH.<N>.<u>..., Gp.<M>, G2, G3)";
    return false;
}
bool HookParseChi(const std::string& s, uint64_t N, uint32_t ell, CharacterDatum& out, std::string& why) {
    uint64_t v = 0;
    if (s.empty() || s == "1") { out = Triv(N, ell); return true; }
    if (s.size() > 1 && s[0] == 'w' && HookNumber(s.substr(1), v)) {
        if (N % ell != 0) { why = "chi '" + s + "' needs ell | N"; return false; }
        out = Chi(N, ell, [ell, v](uint64_t u) { return static_cast<uint32_t>(powmod<uint64_t>(u % ell, v, ell)); }, s);
        return true;
    }
    if (s.size() > 3 && s.compare(0, 3, "leg") == 0 && HookNumber(s.substr(3), v)) {
        if (!HookPrime(v) || v < 3 || N % v != 0) { why = "chi '" + s + "' needs an odd prime q | N"; return false; }
        out = Chi(N, ell, [ell, v](uint64_t u) { return powmod<uint64_t>(u % v, (v - 1) / 2, v) == 1 ? 1u : ell - 1; }, s);
        return true;
    }
    why = "character '" + s + "' (expected 1, w<e>, leg<q>)";
    return false;
}

uint32_t LevelConeCellsHook(const std::string& spec) {
    Checker g;
    Totals tot;
    struct HookCell { HookLevel A, B; uint32_t ell = 0, n = 0; std::string chi, text; };
    std::vector<HookCell> cells;
    const std::vector<std::string> toks = HookSplit(spec, ',');
    bool refused = false;
    for (size_t k = 0; k < toks.size(); ++k) {
        const std::vector<std::string> f = HookSplit(toks[k], ':');
        HookCell c;
        c.text = toks[k];
        std::string why;
        uint64_t ell = 0, n = 0;
        bool ok = (f.size() == 4 || f.size() == 5);
        if (!ok) why = "expected A:B:ell:n[:chi]";
        if (ok && !(HookNumber(f[2], ell) && HookPrime(ell) && ell < (1u << 16))) { ok = false; why = "ell '" + f[2] + "' is not a prime < 2^16"; }
        if (ok && !(HookNumber(f[3], n) && n <= 200)) { ok = false; why = "n '" + f[3] + "' (0..200)"; }
        if (ok) ok = HookParseLevel(f[0], c.A, why) && HookParseLevel(f[1], c.B, why);
        if (ok) c.chi = f.size() == 5 ? f[4] : "1";
        if (ok && c.chi != "1" && !(c.A.gamma0_type && c.B.gamma0_type)) { ok = false; why = "a character needs Gamma_0 / Gamma_H levels"; }
        if (!ok) {
            refused = true;
            std::cout << "  REFUSED HECKE_LEVEL_CONE_CELLS token " << k << " '" << toks[k] << "': " << why << "\n";
            g.check(false, "HECKE_LEVEL_CONE_CELLS token " + std::to_string(k) + " malformed: " + why);
            continue;
        }
        c.ell = static_cast<uint32_t>(ell);
        c.n = static_cast<uint32_t>(n);
        cells.push_back(std::move(c));
    }
    std::cout << "EFFECTIVE HeckeLevelConeValidation arm=CELLS cells=" << (refused ? 0 : cells.size()) << "\n";
    if (refused) {
        std::cout << "LEVEL_CONE checks=" << g.checks << " failing=" << g.fails << " verdict=FAIL\n";
        return 1;
    }
    for (size_t k = 0; k < cells.size(); ++k) {
        HookCell& c = cells[k];
        const auto t0 = std::chrono::steady_clock::now();
        CellIn in;
        std::string why;
        if (!HookParseChi(c.chi, c.A.chiN, c.ell, in.chiA, why) || !HookParseChi(c.chi, c.B.chiN, c.ell, in.chiB, why)) {
            g.check(false, "HECKE_LEVEL_CONE_CELLS token " + std::to_string(k) + " refused: " + why);
            continue;
        }
        in.A = c.A.lvl.get();
        in.B = c.B.lvl.get();
        in.ell = c.ell;
        in.n = c.n;
        in.tag = CellTag("CELLS " + std::to_string(k), *in.A, *in.B, c.chi, c.ell, c.n);
        CellOut out;
        if (!RunCell(g, tot, in, out)) continue;
        const LevelPairCone::SixTerm& s = out.st;
        std::ostringstream os;
        os << "  " << in.tag << ": Route A h0A=" << s.h0A << " h1A=" << s.h1A << " h0B=" << s.h0B << " h1B=" << s.h1B << " rank res1=" << s.rank_res1
           << " H1(A,B)=" << s.H1c << " H2(A,B)=" << s.H2c << " m_B(n+1)=" << static_cast<uint64_t>(in.B->m()) * (c.n + 1);
        if (c.ell >= 5) {
            try {
                const RouteBSix b = RouteBSixTerm(*in.A, in.chiA, *in.B, in.chiB, c.n, c.ell);
                os << " | Route B " << b.h0A << " " << b.h1A << " " << b.h0B << " " << b.h1B << " " << b.rank_res1 << " " << b.H1c << " " << b.H2c;
                g.check(b.h0A == s.h0A && b.h1A == s.h1A && b.h0B == s.h0B && b.h1B == s.h1B && b.rank_res1 == s.rank_res1 && b.H1c == s.H1c &&
                            b.H2c == s.H2c && b.alpha_escapes == 0,
                        in.tag + ": Route A = Route B (obl:levelconeexact (iv))");
            } catch (const std::exception& e) {
                g.check(false, in.tag + ": Route B threw: " + e.what());
            }
        }
        std::cout << os.str() << " (" << Seconds(t0) << " s, peak_rss=" << oc_util::PeakRssMiB() << " MiB)\n";
    }
    std::cout << "  LEVEL_CONE cells=" << tot.cells << " built=" << tot.cells_ok << " zigzag_bad=" << tot.zigzag_bad << " dalpha_bad=" << tot.dalpha_bad
              << "\n";
    std::cout << "LEVEL_CONE checks=" << g.checks << " failing=" << g.fails << " verdict=" << (g.fails == 0 ? "OK" : "FAIL") << "\n";
    return g.fails == 0 ? 0u : 1u;
}

} // namespace

uint32_t HeckeLevelConeValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeLevelConeValidation: Route A, the level cone Cone(alpha_{B/A}) on the 7-vertex model ([C] sec:nebles, "
                 "obl:levelconeexact (ii), (iii), (v), (vi), (viii), (ix); LES2_predictions.md)\n";
    if (const char* cells_env = std::getenv("HECKE_LEVEL_CONE_CELLS")) return LevelConeCellsHook(cells_env);
    Checker g;
    Totals tot;
    const SimplicialComplex& K = *TheModel().K;
    auto g0tr = [](uint64_t NB) {
        return [NB](uint64_t p) { return Gamma0SubTransversalN(Gamma0LevelN(NB), Gamma0LevelN(NB * p)); };
    };

    // ---- (I) Gamma_0(p) < Gamma, chi = 1 ---------------------------------------------------------------------------
    {
        const FullLevel Gam;
        struct R { uint32_t p, n, h0A, h1A, h0B, h1B; };
        // (h0B, h1B) at ell = 5, p = 5 differ at n = 4, 10 (LES_predictions.md section 2): handled below.
        const R rows[] = {{5, 0, 1, 0, 1, 1}, {5, 2, 0, 1, 0, 4}, {5, 4, 0, 1, 0, 4}, {5, 10, 0, 3, 0, 12},
                          {7, 0, 1, 0, 1, 1}, {7, 2, 0, 1, 0, 4}, {7, 4, 0, 1, 0, 8}, {7, 10, 0, 3, 0, 16},
                          {11, 0, 1, 0, 1, 3}, {11, 2, 0, 1, 0, 6}, {11, 4, 0, 1, 0, 10}, {11, 10, 0, 3, 0, 22}};
        std::map<uint32_t, std::unique_ptr<CosetLevel>> B;
        uint32_t split_ok = 0, cells = 0;
        for (const R& r : rows) {
            if (!B.count(r.p)) B[r.p] = G0(r.p);
            for (uint32_t ell : {5u, 7u, 13u, 101u}) {
                CellIn in;
                in.A = &Gam; in.B = B[r.p].get();
                in.chiA = Triv(1, ell); in.chiB = Triv(r.p, ell);
                in.ell = ell; in.n = r.n;
                in.tag = CellTag("I", Gam, *in.B, "1", ell, r.n);
                uint32_t h0B = r.h0B, h1B = r.h1B;
                if (ell == 5 && r.p == 5 && r.n == 4) { h0B = 1; h1B = 5; }
                if (ell == 5 && r.p == 5 && r.n == 10) { h0B = 1; h1B = 13; }
                in.pred = Split(r.h0A, r.h1A, h0B, h1B);
                if (r.n <= 4)
                    for (uint64_t tp : {2ull, 3ull}) in.ops.push_back(TOp(tp, g0tr(r.p)));
                CellOut out;
                ++cells;
                if (RunCell(g, tot, in, out) && out.st.ker_res1() == 0) ++split_ok;
            }
        }
        g.check(cells == 48 && split_ok == 48, "(I) 48 cells, ker res^1 = 0 at all (prop:levelsplit)");
    }
    // ---- (I') non-split level-one cells ----------------------------------------------------------------------------
    {
        const FullLevel Gam;
        struct R { uint32_t p, ell; uint32_t h1B[3]; };
        const R rows[] = {{19, 5, {3, 10, 18}}, {13, 7, {1, 8, 12}}, {29, 5, {5, 16, 24}}};
        for (const R& r : rows) {
            const auto Bl = G0(r.p);
            for (uint32_t k = 0; k < 3; ++k) {
                const uint32_t n = 2 * k;
                CellIn in;
                in.A = &Gam; in.B = Bl.get();
                in.chiA = Triv(1, r.ell); in.chiB = Triv(r.p, r.ell);
                in.ell = r.ell; in.n = n;
                in.tag = CellTag("I'", Gam, *Bl, "1", r.ell, n);
                const uint32_t h0A = n == 0 ? 1 : 0, h1A = n == 0 ? 0 : 1, h0B = n == 0 ? 1 : 0;
                in.pred = P8(h0A, h1A, h0B, r.h1B[k], h1A, 0, 0, r.h1B[k] - h1A);
                CellOut out;
                RunCell(g, tot, in, out);
            }
        }
    }
    // ---- (II) Gamma_0(25) < Gamma_0(5) ------------------------------------------------------------------------------
    {
        const auto A = G0(5), B = G0(25);
        struct R { uint32_t ell; int chi; uint32_t n; uint32_t v[8]; };   // chi: 0 = 1, 2 = (./5), 4 = chi_4
        const R rows[] = {
            {5, 0, 0, {1, 1, 1, 5, 1, 0, 0, 4}},   {5, 0, 2, {0, 4, 0, 16, 3, 1, 1, 13}}, {5, 0, 4, {1, 5, 1, 25, 5, 0, 0, 20}},
            {5, 2, 0, {0, 2, 0, 6, 1, 1, 1, 5}},   {5, 2, 2, {1, 3, 1, 15, 3, 0, 0, 12}}, {5, 2, 4, {0, 6, 0, 26, 5, 1, 1, 21}},
            {5, 4, 1, {0, 2, 0, 10, 1, 1, 1, 9}},  {5, 4, 3, {1, 5, 1, 21, 5, 0, 0, 16}},
            {7, 0, 0, {1, 1, 1, 5, 1, 0, 0, 4}},   {7, 0, 2, {0, 4, 0, 16, 4, 0, 0, 12}}, {7, 0, 4, {0, 4, 0, 24, 4, 0, 0, 20}},
            {7, 2, 0, {0, 2, 0, 6, 2, 0, 0, 4}},   {7, 2, 2, {0, 2, 0, 14, 2, 0, 0, 12}}, {7, 2, 4, {0, 6, 0, 26, 6, 0, 0, 20}}};
        for (const R& r : rows) {
            CellIn in;
            in.A = A.get(); in.B = B.get();
            if (r.chi == 0) { in.chiA = Triv(5, r.ell); in.chiB = Triv(25, r.ell); }
            if (r.chi == 2) { in.chiA = Legendre5(5, r.ell); in.chiB = Legendre5(25, r.ell); }
            if (r.chi == 4) { in.chiA = Chi4mod5(5); in.chiB = Chi4mod5(25); }
            in.ell = r.ell; in.n = r.n;
            in.tag = CellTag("II", *A, *B, in.chiA.label, r.ell, r.n);
            in.pred = P8(r.v[0], r.v[1], r.v[2], r.v[3], r.v[4], r.v[5], r.v[6], r.v[7]);
            for (uint64_t tp : {2ull, 3ull}) in.ops.push_back(TOp(tp, g0tr(25)));
            if (r.chi == 0 && r.n <= 2) {
                in.level_raise = true;
                in.lr_p = 5; in.lr_NA = 5; in.lr_NB = 25;
                in.lr_defect_predicted = (r.ell == 7 && r.n == 0) ? 1 : -1;
            }
            CellOut out;
            RunCell(g, tot, in, out);
        }
    }
    // ---- (III) Gamma_1(11) |> Gamma_0(11) ---------------------------------------------------------------------------
    {
        const auto A = G0(11);
        const auto B = G1(11);
        const GammaHLevelN* Bh = B.get();
        struct R { uint32_t ell, n; uint32_t v[8]; };
        const R rows[] = {{5, 0, {1, 3, 1, 11, 2, 1, 1, 9}},  {5, 2, {0, 6, 0, 30, 6, 0, 0, 24}},  {5, 4, {0, 10, 0, 50, 10, 0, 0, 40}},
                          {7, 0, {1, 3, 1, 11, 3, 0, 0, 8}},  {7, 2, {0, 6, 0, 30, 6, 0, 0, 24}},  {7, 4, {0, 10, 0, 50, 10, 0, 0, 40}}};
        for (const R& r : rows) {
            CellIn in;
            in.A = A.get(); in.B = B.get();
            in.chiA = Triv(11, r.ell); in.chiB = Triv(11, r.ell);
            in.ell = r.ell; in.n = r.n;
            in.tag = CellTag("III", *A, *B, "1", r.ell, r.n);
            in.pred = P8(r.v[0], r.v[1], r.v[2], r.v[3], r.v[4], r.v[5], r.v[6], r.v[7]);
            if (r.n == 0) {
                auto trH = [Bh](uint64_t p) { return GammaHSubTransversal(*Bh, p); };
                for (uint64_t tp : {2ull, 3ull}) in.ops.push_back(TOp(tp, trH));
                HeckeOp U;   // same-level U_11 (prop:levelconehecke (2)): r_t = [[1, 0], [11 t, 1]] in Gamma_1(11)
                U.p = 11; U.transB = trH; U.index = 11; U.label = "U_11";
                in.ops.push_back(U);
            }
            CellOut out;
            RunCell(g, tot, in, out);
        }
    }
    // ---- (III') Gamma_1(p) |> Gamma_0(p) at ell = p -------------------------------------------------------------------
    {
        struct R { uint32_t p, n, h1B; std::vector<uint32_t> parts; };
        const R rows[] = {{5, 0, 3, {1, 0, 2, 0}}, {5, 2, 7, {4, 0, 3, 0}}, {5, 4, 11, {5, 0, 6, 0}},
                          {7, 0, 5, {1, 0, 2, 0, 2, 0}}, {7, 2, 13, {4, 0, 4, 0, 5, 0}}, {7, 4, 21, {8, 0, 7, 0, 6, 0}}};
        std::map<uint32_t, std::pair<std::unique_ptr<CosetLevel>, std::unique_ptr<GammaHLevelN>>> lv;
        for (const R& r : rows) {
            if (!lv.count(r.p)) lv[r.p] = std::make_pair(G0(r.p), G1(r.p));
            const CosetLevel* A = lv[r.p].first.get();
            const GammaHLevelN* Bh = lv[r.p].second.get();
            CellIn in;
            in.A = A; in.B = Bh;
            in.chiA = Triv(r.p, r.p); in.chiB = Triv(r.p, r.p);
            in.ell = r.p; in.n = r.n;
            in.tag = CellTag("III'", *A, *Bh, "1", r.p, r.n);
            // The committed predictions of this arm (LES_predictions.md section 2 (iii'), LES2_predictions.md section 1) are
            // h1(Gamma_0(p)) = parts[0], h1(Gamma_1(p)), ker res^1 = 0 (split, index p - 1 a unit) and the character sum; h0B = c
            // = floor(n/p) + 1 = 1 is (LES7)'s lem:unipotentinv.  h0A has no committed number: it is printed, and enters only
            // through the split identity H^1(A,B) = h0B - h0A.
            if (r.n <= 2) {
                auto trH = [Bh](uint64_t p) { return GammaHSubTransversal(*Bh, p); };
                for (uint64_t tp : {2ull, 3ull}) in.ops.push_back(TOp(tp, trH));
            }
            CellOut out;
            if (!RunCell(g, tot, in, out)) continue;
            {
                const LevelPairCone::SixTerm& t = out.st;
                std::ostringstream os;
                os << in.tag << ": got h0A=" << t.h0A << " h1A=" << t.h1A << " h0B=" << t.h0B << " h1B=" << t.h1B << " rank res1=" << t.rank_res1
                   << " H1(A,B)=" << t.H1c << " H2(A,B)=" << t.H2c << "; predicted h1A=" << r.parts[0] << " h0B=1 h1B=" << r.h1B << " ker res1=0";
                std::cout << "  " << os.str() << "\n";
                g.check(t.h1A == r.parts[0] && t.h1B == r.h1B && t.h0B == 1 && t.ker_res1() == 0 && t.rank_res1 == t.h1A &&
                            t.H1c == t.h0B - t.h0A && t.H2c == t.h1B - t.h1A,
                        os.str());
            }
            // h^1(Gamma_1(p)) = sum_i h^1(Gamma_0(p), omega^i) (prop:nebdecomp (3), prop:levelsplit): the character models.
            const engine::ModCtx<uint32_t> ctx(r.p);
            std::vector<uint32_t> got;
            uint32_t sum = 0;
            for (uint32_t i = 0; i + 1 < r.p; ++i) {
                const auto M = BuildLevelModel(K, *A, Omega(r.p, i), r.n, ctx);
                got.push_back(M->H->get_H_eq_dimensions()[1]);
                sum += got.back();
            }
            g.check(got == r.parts && sum == out.st.h1B, in.tag + ": h1(Gamma_1(p)) = sum_i h1(Gamma_0(p), omega^i) with the predicted summands");
        }
    }
    // ---- (R3) Gamma_1(5) < Gamma_0(5), chi_4 on A ----------------------------------------------------------------------
    {
        const auto A = G0(5);
        const auto B = G1(5);
        const GammaHLevelN* Bh = B.get();
        for (uint32_t n : {1u, 3u}) {
            CellIn in;
            in.A = A.get(); in.B = B.get();
            in.chiA = Chi4mod5(5); in.chiB = Triv(5, 5);
            in.ell = 5; in.n = n;
            in.tag = CellTag("R3", *A, *B, "chi_4", 5, n);
            in.pred = n == 1 ? P8(0, 2, 1, 5, 2, 0, 1, 3) : P8(1, 5, 1, 9, 5, 0, 0, 4);
            auto trH = [Bh](uint64_t p) { return GammaHSubTransversal(*Bh, p); };
            for (uint64_t tp : {2ull, 3ull}) in.ops.push_back(TOp(tp, trH));
            CellOut out;
            RunCell(g, tot, in, out);
        }
    }
    // ---- (IV) ell in {2, 3}: tags and model dimensions ----------------------------------------------------------------
    {
        const FullLevel Gam;
        const auto G02 = G0(2), G04 = G0(4), G03 = G0(3), G09 = G0(9);
        const auto G14 = G1(4);
        const CongruenceLevel C2 = CongruenceLevel::Commutator(2), C3 = CongruenceLevel::Commutator(3);
        const CosetLevel* As[5] = {G02.get(), G03.get(), G04.get(), &Gam, &Gam};
        const CosetLevel* Bs[5] = {G04.get(), G09.get(), G14.get(), &C2, &C3};
        const uint64_t NA[5] = {2, 3, 4, 1, 1}, NB[5] = {4, 9, 4, 1, 1};
        uint32_t cells = 0;
        for (const ArmIVRow& r : kArmIV) {
            CellIn in;
            in.A = As[r.pair]; in.B = Bs[r.pair];
            in.chiA = Triv(NA[r.pair], r.ell); in.chiB = Triv(NB[r.pair], r.ell);
            in.ell = r.ell; in.n = r.n;
            in.tag = CellTag("IV", *in.A, *in.B, "1", r.ell, r.n);
            in.pred = P8(r.h0A, r.h1A, r.h0B, r.h1B, r.rank1, r.ker1, r.H1c, r.H2c);
            in.tags = true; in.tagA_group = r.tagA_group; in.tagB_group = r.tagB_group;
            CellOut out;
            ++cells;
            RunCell(g, tot, in, out);
        }
        g.check(cells == 50, "(IV) 50 cells");
    }

    // ---- (NEG) obl:levelconeexact (ix) -----------------------------------------------------------------------------------
    {
        const FullLevel Gam;
        const auto B5 = G0(5);
        const uint32_t ell = 7, n = 2;
        const engine::ModCtx<uint32_t> ctx(ell);
        // cone_sign: D'(c, c') = (+dc, dc' + alpha c) has D'^2 != 0 on the stored basis.
        {
            CellIn in;
            in.A = &Gam; in.B = B5.get(); in.chiA = Triv(1, ell); in.chiB = Triv(5, ell); in.ell = ell; in.n = n;
            in.tag = "(NEG cone_sign) " + CellTag("I", Gam, *B5, "1", ell, n);
            CellOut out;
            bool caught = false;
            if (RunCell(g, tot, in, out)) {
                const EquivariantMappingCone& C = out.C->cone();
                for (uint32_t q = 0; q + 1 <= C.q_max() && !caught; ++q) {
                    Mat D0 = OfBlock(C.D_block(q)), D1 = OfBlock(C.D_block(q + 1));
                    auto flip = [&](Mat& D, uint32_t qq) {
                        for (uint32_t i = 0; i < C.bulk_part_dim(qq + 1); ++i)
                            for (uint32_t j = 0; j < C.bulk_part_dim(qq); ++j) D.at(i, j) = (ell - D.at(i, j)) % ell;
                    };
                    flip(D0, q);
                    flip(D1, q + 1);
                    const Mat X = OfBlock(C.Ceq_block(q));
                    if (X.c && D1.r) caught = !IsZero(Mul(ell, D1, Mul(ell, D0, X)));
                }
            }
            std::cout << "  NEG cone_sign " << (caught ? "caught" : "NOT caught") << "\n";
            g.check(caught, "(NEG) cone_sign caught (D'^2 != 0)");
        }
        // res_no_block / res_wrong_scalar: alpha is not a Gamma-map; the cone's certificate refuses it.
        auto neg_alpha = [&](const std::string& name, const CosetLevel& A, const CharacterDatum& cA, const CosetLevel& B,
                             const CharacterDatum& cB, uint32_t l, uint32_t nn, RestrictionOptions opts) {
            const engine::ModCtx<uint32_t> c(l);
            bool caught = false;
            std::string why;
            try {
                RestrictionFibreMap R(A, cA, B, cB, nn, c, opts);
                const auto MA = BuildLevelModel(K, A, cA, nn, c);
                const auto MB = BuildLevelModel(K, B, cB, nn, c);
                // alpha C_eq(L_A) must lie in C_eq(L_B): the boundary part of D_cone (c, 0) = alpha c (C2)
                LevelPairCone Cn(*MA, *MB, R.alpha(), name);
                try {
                    Cn.compute();
                    why = "cone built";
                } catch (const std::exception& e) {
                    caught = true;
                    why = std::string("certificate refuses: ") + std::string(e.what()).substr(0, 120);
                }
            } catch (const std::exception& e) {
                why = std::string("setup threw: ") + e.what();
            }
            std::cout << "  NEG " << name << " " << (caught ? "caught" : "NOT caught") << " (" << why << ")\n";
            g.check(caught, "(NEG) " + name + " caught");
        };
        RestrictionOptions nb;
        nb.neg_identity_block = true;
        neg_alpha("res_no_block", Gam, Triv(1, ell), *B5, Triv(5, ell), ell, n, nb);
        {
            const auto A = G0(5);
            const auto B = G1(5);
            RestrictionOptions ws;
            ws.neg_inverse_scalar = true;
            neg_alpha("res_wrong_scalar", *A, Chi4mod5(5), *B, Triv(5, 5), 5, 1, ws);
        }
        // incompatible_chi: refused by name
        {
            const auto A = G0(5), B = G0(25);
            bool caught = false;
            try {
                RestrictionFibreMap R(*A, Legendre5(5, ell), *B, Triv(25, ell), n, ctx);
            } catch (const LevelPairRefusal& e) {
                caught = e.code() == "les_incompatible_character";
            }
            std::cout << "  NEG incompatible_chi " << (caught ? "caught" : "NOT caught") << " (les_incompatible_character)\n";
            g.check(caught, "(NEG) incompatible_chi refused les_incompatible_character");
        }
        // (audit R1) hecke_two_transversals (formerly tagged "NEG hecke_fixed_tables"; gated, not a negative control,
        // [C, obl:levelconeexact](ix); C1230_pre3_referee F5), re-sited where the transversal enters eq:levelcorr (m_A > 1): T_2 at Gamma_0(25) < Gamma_0(5),
        // chi = 1, ell = 7, n = 2, T^A_2 on A's own transversal against T^A_2 on B's.  Predicted (LES2_predictions.md section 7): equal
        // on C^q_eq(L_A) and the strict square 0 (prop:levelconehecke (4) as re-proved in LES2_audit.md section 3.2); different on the
        // whole flat C^q.  (At m_A = 1 the formula cannot see the transversal: that record is retired.)
        {
            const auto A5 = G0(5), B25 = G0(25);
            CellIn in;
            in.A = A5.get(); in.B = B25.get(); in.chiA = Triv(5, ell); in.chiB = Triv(25, ell); in.ell = ell; in.n = n;
            in.tag = "(hecke_two_transversals) " + CellTag("II", *A5, *B25, "1", ell, n);
            CellOut out;
            if (RunCell(g, tot, in, out)) {
                const HeckeDatum& d2 = Datum(2, ell, n);
                const std::vector<Mat22> rOwn = Gamma0SubTransversalN(Gamma0LevelN(5), Gamma0LevelN(10));
                const std::vector<Mat22> rB = Gamma0SubTransversalN(Gamma0LevelN(25), Gamma0LevelN(50));
                CorrespondenceSpec sAown{A5.get(), A5.get(), d2.cosets.get(), d2.tau.get(), rOwn, "T_2^A(own)"};
                CorrespondenceSpec sAB{A5.get(), A5.get(), d2.cosets.get(), d2.tau.get(), rB, "T_2^A(B's)"};
                CorrespondenceSpec sB{B25.get(), B25.get(), d2.cosets.get(), d2.tau.get(), rB, "T_2^B"};
                VerifyTransversal(sAown, 3);
                VerifyTransversal(sAB, 3);
                VerifyTransversal(sB, 3);
                uint64_t square_own = 0, own_vs_common = 0, own_flat = 0, transversals_differ = 0;
                for (size_t t = 0; t < rOwn.size() && t < rB.size(); ++t)
                    transversals_differ += (rOwn[t].a != rB[t].a || rOwn[t].b != rB[t].b || rOwn[t].c != rB[t].c || rOwn[t].d != rB[t].d);
                for (uint32_t q = 0; q <= 2; ++q) {
                    const Mat X = OfBlock(out.MA->H->C_eq_block(q));
                    const uint32_t Nq = K.get_num_q_simplices(q);
                    const BlockCochainMap TAown = hecke::neb::BuildLevelCorrespondenceChi(ctx, sAown, K, n, q, ScalarOf(in.chiA));
                    const BlockCochainMap TAB = hecke::neb::BuildLevelCorrespondenceChi(ctx, sAB, K, n, q, ScalarOf(in.chiA));
                    const BlockCochainMap TB = hecke::neb::BuildLevelCorrespondenceChi(ctx, sB, K, n, q, ScalarOf(in.chiB));
                    auto alpha_of = [&](const Mat& Y) { Mat Z(Nq * out.MB->d(), Y.c); Z.a = ApplyFibre(out.R->alpha(), Nq, Y.a, Y.c, ctx); return Z; };
                    if (X.c > 0) {
                        square_own += Differ(alpha_of(ApplyMap(TAown, X)), ApplyMap(TB, alpha_of(X)));
                        own_vs_common += Differ(ApplyMap(TAown, X), ApplyMap(TAB, X));
                    }
                    Mat I(Nq * out.MA->d(), Nq * out.MA->d());
                    for (uint32_t i = 0; i < I.r; ++i) I.at(i, i) = 1;
                    own_flat += Differ(ApplyMap(TAown, I), ApplyMap(TAB, I));
                }
                std::cout << "  HECKE_FIXED_TABLES T_2 at " << in.tag << ": transversals differ in " << transversals_differ << " of "
                          << rOwn.size() << " matrices; strict square with A's own transversal: " << square_own
                          << " differing entries; T^A(own) vs T^A(B's) on C_eq(L_A): " << own_vs_common << "; on the whole flat C^q: " << own_flat
                          << " (predicted 0 / 0 / > 0; the flat prediction is REFUTED when this reads 0, see below)\n";
                g.check(transversals_differ > 0, "(gated) hecke_two_transversals: the two transversals of Gamma_0(10) \\ Gamma_0(5) differ");
                g.check(square_own == 0 && own_vs_common == 0,
                        "(gated) hecke_two_transversals: T^A_2 on C_eq(L_A) is transversal-independent and the strict square holds with A's own (prop:levelconehecke (4))");
                // RECORDED, not gated (first repair run, 2026-10-12): the committed prediction "own_flat > 0" (LES2_predictions.md section 7,
                // from LES2_audit.md section 3.2) was REFUTED: own_flat = 0.  eq:levelcorr (BuildLevelCorrespondence) reads r_t only
                // through j = class(r_t v_i) in the level-1 table and a = class_A(g^{-1} r_t v_i s_j^{-1} g), and both depend only on
                // the coset A'' r_t (b in A'' leaves j fixed and multiplies k by g^{-1} b g in A); its block matrices do not contain r_t.
                // So the operator is transversal-independent on the WHOLE flat C^q by construction, at every level; this record cannot
                // discriminate (T1)-independence on C_eq from construction.  LES2_report.md section 3, item 2.
                std::cout << "  HECKE_FIXED_TABLES_FLAT own_flat=" << own_flat << " (recorded; prediction > 0 "
                          << (own_flat > 0 ? "met" : "REFUTED: eq:levelcorr is transversal-independent by construction") << ")\n";
            }
        }
    }

    const double wall = Seconds(t0);
    std::cout << "  LEVEL_CONE cells=" << tot.cells << " built=" << tot.cells_ok << " hecke_squares=" << tot.hecke_squares
              << " hecke_bad_entries=" << tot.hecke_square_bad_entries << " families_bad=" << tot.families_bad
              << " charpoly_bad=" << tot.charpoly_bad << " zigzag_bad=" << tot.zigzag_bad << " dalpha_bad=" << tot.dalpha_bad
              << " level_raise_cells=" << tot.lr_cells << " level_raise_bad_entries=" << tot.lr_bad_entries << "\n";
    std::cout << "  slowest cell " << tot.max_cell << " " << tot.max_cell_s << " s; peak_rss=" << oc_util::PeakRssMiB()
              << " MiB wall=" << wall << " s\n";
    std::cout << "LEVEL_CONE checks=" << g.checks << " failing=" << g.fails << " verdict=" << (g.fails == 0 ? "OK" : "FAIL") << "\n";
    return g.fails == 0 ? 0u : 1u;
}
