// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-10) HeckeHidaHeavyCellProbe -- the heavy Hida
// cells.  Binding text: [C] rmk:hidacheckable2 (Hi11), (Hi12); cor:ordprecision
// (the reading of the ordinary part at precision m); the U_ell memory lever of
// slot-13 WP-4, which cannot change a value (HeckeUEllLowRankValidation held it
// to the dense route entrywise).  S24 CL-R1: the lever is retired, so the probe
// no longer forces it: the default Auto mode puts it in effect exactly where the
// old forcing did (U_ell: ell | N = ell, SymN) and off for T_2, T_3 (ell does not
// divide N), so every route, table and value is the forced one.
//
// INTERLOCK.  Expensive (minutes, GBs), so it must not run inside `--all`,
// `--all-full`, the sanitizer configurations or cross_config.sh: without
// HECKE_HIDA_HEAVY=1 it prints [SKIP] and returns 0 (the HeckeLocalityProbe
// pattern).  Read-only; writes nothing, emits no record and no golden object.
//
// CELLS.  HECKE_HIDA_HEAVY_CELLS="n:ell:m,..." (default: WP-10's ladder and the
// (Hi11) cell over F_691, "100:691:1,200:691:1,400:691:1,700:691:1").  Per cell:
//   * the pre-pass e_max (Hecke/HeckeOrdinaryPart); the reading below is
//     cor:ordprecision's reading H_m = H_K / ell^m H_K only when e_max = 0 (then
//     H_K is free and K = m); a cell with e_max > 0 is refused by name (the
//     heavy cells have e_max = 0: no Dickson invariant in degree n);
//   * m = 1: the shipped FIELD route over F_ell (EquivariantCohomologyOnK,
//     HeckeOperatorOnK), U_ell under the lever (the tau table constructed
//     inside the scope, so its rho pool is factored), T_2 and T_3 on the
//     default route;  m >= 2: the shipped MODULE route over Z/ell^m, U_ell
//     under the lever likewise;
//   * the ordinary summand of H^1 as the image of W = U^{m g} (g = generator
//     count: U^g = 0 mod ell on the non-ordinary summand, so W = 0 mod ell^m
//     there, and U is invertible on the ordinary summand).  Its basis: the
//     columns of W independent mod ell (Nakayama; the image is a free direct
//     summand).  T on it: X with B X = T B, solved on h_ord rows of B that are
//     independent mod ell and then VERIFIED on every row (so T preserves the
//     image).  charpoly by Faddeev-LeVerrier (h_ord < ell).
// GATED (a failure is a failing check): at n <= 40 the reading equals the shipped
// hida::ComputeTowerCell's (an independent projector and charpoly); U T_p = T_p U and T_2 T_3 = T_3 T_2 on
// H^1; the verification B X = T B; h_ord >= 1 and 1 + p^{n+1} a root of
// chi^ord_p mod ell (the Eisenstein half, PROVED by (Hi5)).  REPORTED, NOT GATED:
// (Hi11)'s conjectural part at (700, 691) -- h_ord = 3 and
// chi^ord_p = (X - (1 + p^11))^3 mod 691 -- a failure there is a finding,
// printed as HI11 ... verdict=FAILED, never "fixed".
#include "HeckeHidaHeavyCellProbe.h"
#include "HeckeHidaTower.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorFamily.h"   // tau_construction_mutex()
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeOrdinaryPart.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"

namespace {

using Vec = std::vector<uint64_t>;
using M = std::vector<Vec>;   // square, row-major, entries mod q

struct Checker {
    uint64_t checks = 0, failures = 0;
    void operator()(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

uint64_t PowMod(uint64_t b, uint64_t e, uint64_t q) {
    unsigned __int128 r = 1 % q, x = b % q;
    while (e) {
        if (e & 1) r = r * x % q;
        x = x * x % q;
        e >>= 1;
    }
    return static_cast<uint64_t>(r);
}

// a^{-1} mod q = ell^m, a a unit (Euler: phi(ell^m) = ell^{m-1}(ell - 1)).
uint64_t InvUnit(uint64_t a, uint64_t ell, uint64_t q) {
    ENGINE_CHECK_MSG(a % ell != 0, "HeckeHidaHeavyCellProbe: inverse of a non-unit");
    return PowMod(a, q / ell * (ell - 1) - 1, q);
}

M Mul(const M& A, const M& B, uint64_t q) {
    const size_t n = A.size(), k = B.size(), c = k ? B[0].size() : 0;
    M C(n, Vec(c, 0));
    for (size_t i = 0; i < n; ++i) {
        std::vector<unsigned __int128> acc(c, 0);
        for (size_t t = 0; t < k; ++t) {
            const uint64_t a = A[i][t];
            if (!a) continue;
            for (size_t j = 0; j < c; ++j) acc[j] += static_cast<unsigned __int128>(a) * B[t][j];
        }
        for (size_t j = 0; j < c; ++j) C[i][j] = static_cast<uint64_t>(acc[j] % q);
    }
    return C;
}

M Identity(size_t n) {
    M I(n, Vec(n, 0));
    for (size_t i = 0; i < n; ++i) I[i][i] = 1;
    return I;
}

M PowM(M A, uint64_t e, uint64_t q) {
    M R = Identity(A.size());
    while (e) {
        if (e & 1) R = Mul(R, A, q);
        A = Mul(A, A, q);
        e >>= 1;
    }
    return R;
}

// Indices of vectors (the rows of V) independent mod ell, greedily in order.
std::vector<size_t> IndependentModEll(const M& V, uint64_t ell) {
    std::vector<size_t> pick;
    std::vector<Vec> basis;          // echelon rows mod ell
    std::vector<size_t> lead;
    for (size_t r = 0; r < V.size(); ++r) {
        Vec v(V[r].size());
        for (size_t j = 0; j < v.size(); ++j) v[j] = V[r][j] % ell;
        for (size_t b = 0; b < basis.size(); ++b) {
            const uint64_t f = v[lead[b]];
            if (!f) continue;
            for (size_t j = 0; j < v.size(); ++j) v[j] = (v[j] + (ell - f) * basis[b][j]) % ell;
        }
        size_t l = 0;
        while (l < v.size() && v[l] == 0) ++l;
        if (l == v.size()) continue;
        const uint64_t iv = PowMod(v[l], ell - 2, ell);
        for (auto& x : v) x = x * iv % ell;
        basis.push_back(v);
        lead.push_back(l);
        pick.push_back(r);
    }
    return pick;
}

M Transpose(const M& A) {
    if (A.empty()) return A;
    M T(A[0].size(), Vec(A.size()));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
    return T;
}

// Inverse of a square matrix over Z/q, q = ell^m, invertible mod ell.
M Inverse(M A, uint64_t ell, uint64_t q) {
    const size_t n = A.size();
    M I = Identity(n);
    for (size_t c = 0; c < n; ++c) {
        size_t p = c;
        while (p < n && A[p][c] % ell == 0) ++p;
        ENGINE_CHECK_MSG(p < n, "HeckeHidaHeavyCellProbe: the row block is not invertible mod ell");
        std::swap(A[p], A[c]);
        std::swap(I[p], I[c]);
        const uint64_t iv = InvUnit(A[c][c], ell, q);
        for (size_t j = 0; j < n; ++j) {
            A[c][j] = static_cast<uint64_t>(static_cast<unsigned __int128>(A[c][j]) * iv % q);
            I[c][j] = static_cast<uint64_t>(static_cast<unsigned __int128>(I[c][j]) * iv % q);
        }
        for (size_t i = 0; i < n; ++i) {
            if (i == c || A[i][c] == 0) continue;
            const uint64_t f = q - A[i][c];
            for (size_t j = 0; j < n; ++j) {
                A[i][j] = static_cast<uint64_t>((A[i][j] + static_cast<unsigned __int128>(f) * A[c][j]) % q);
                I[i][j] = static_cast<uint64_t>((I[i][j] + static_cast<unsigned __int128>(f) * I[c][j]) % q);
            }
        }
    }
    return I;
}

// charpoly det(X - A), ascending coefficients, monic; Faddeev-LeVerrier (h < ell).
Vec Charpoly(const M& A, uint64_t ell, uint64_t q) {
    const size_t h = A.size();
    ENGINE_CHECK_MSG(h < ell, "HeckeHidaHeavyCellProbe: Faddeev-LeVerrier needs h < ell");
    Vec c(h + 1, 0);
    c[h] = 1;
    M Mk(h, Vec(h, 0));
    for (size_t k = 1; k <= h; ++k) {
        for (size_t i = 0; i < h; ++i) Mk[i][i] = (Mk[i][i] + c[h - k + 1]) % q;   // M_k = A M_{k-1} + c_{h-k+1} I
        const M AM = Mul(A, Mk, q);
        uint64_t tr = 0;
        for (size_t i = 0; i < h; ++i) tr = (tr + AM[i][i]) % q;
        c[h - k] = static_cast<uint64_t>(static_cast<unsigned __int128>((q - tr) % q) * InvUnit(k, ell, q) % q);
        Mk = AM;
    }
    return c;
}

uint64_t EvalPoly(const Vec& f, uint64_t x, uint64_t q) {
    unsigned __int128 r = 0;
    for (size_t i = f.size(); i-- > 0;) r = (r * (x % q) + f[i]) % q;
    return static_cast<uint64_t>(r);
}

std::string PolyText(const Vec& f) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < f.size(); ++i) os << (i ? "," : "") << f[i];
    os << "]";
    return os.str();
}

bool Equal(const M& A, const M& B) { return A == B; }

struct Ops {
    uint64_t ell = 0, q = 0;
    uint32_t m = 0;
    M U, T2, T3;
    std::string route;
};

// ---- the two routes ---------------------------------------------------------

void BuildModel(std::unique_ptr<SimplicialComplex>& K, std::unique_ptr<hecke::SL2ZTessellationTables>& model) {
    std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
    K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
    model.reset(new hecke::SL2ZTessellationTables(*K));
}

template <class MatT>
M FromField(const MatT& A) {
    M R(A.rows, Vec(A.cols));
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) R[i][j] = A.data[i][j];
    return R;
}

Ops FieldRoute(uint32_t n, uint32_t ell) {
    Ops o;
    o.ell = ell;
    o.q = ell;
    o.m = 1;
    o.route = "field_lever";
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    BuildModel(K, model);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(*K, std::move(rels), n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    for (uint64_t p : {static_cast<uint64_t>(ell), uint64_t{2}, uint64_t{3}}) {
        // (S24 CL-R1) Auto: the lever is in effect iff p == ell (was: ScopedUEllLowRank(p == ell)).
        ENGINE_CHECK_MSG(hecke::HeckeOperatorOptions::u_ell_low_rank_in_effect(p, ell, FibreRepresentation::SymN) == (p == ell),
                         "HeckeHidaHeavyCellProbe: the U_ell lever's Auto mode must be in effect exactly for U_ell");
        std::unique_ptr<hecke::HeckeCosetTables> cosets;
        std::unique_ptr<hecke::HeckeTauTable> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTables(p, 1, ell, n));
            tau.reset(new hecke::HeckeTauTable(*K, *model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnK op(&coh, cosets.get(), tau.get());
        op.compute();
        ENGINE_CHECK_MSG(op.certificates().routes_bit_identical, "HeckeHidaHeavyCellProbe: O12(a) failed");
        M A = FromField(op.get_H_matrices().at(1));
        (p == ell ? o.U : p == 2 ? o.T2 : o.T3) = std::move(A);
    }
    return o;
}

Ops ModuleRoute(uint32_t n, uint64_t ell, uint32_t m) {
    Ops o;
    o.ell = ell;
    o.m = m;
    o.route = "module_lever";
    const hecke::ScopedCoefficientRing ring(m);
    const engine::ModCtx<uint64_t> ctx = hecke::CoefficientRingContext<uint64_t>("HeckeHidaHeavyCellProbe", ell, m);
    o.q = ctx.p;
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    BuildModel(K, model);
    LocalSystemSimplicialComplexT<uint64_t> lssc(*K, hecke::GenerateSL2ZRelationsRing<uint64_t>(*K, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
    coh.compute_cohomology();
    const std::vector<uint32_t> exps = coh.module_degree(1).dec.exponents;
    for (uint32_t e : exps)
        ENGINE_CHECK_MSG(e == m, "HeckeHidaHeavyCellProbe: H^1 over Z/ell^m is not free (e_max > 0 needs K = m + e_max)");
    for (uint64_t p : {ell, uint64_t{2}, uint64_t{3}}) {
        // (S24 CL-R1) Auto: the lever is in effect iff p == ell (was: ScopedUEllLowRank(p == ell)).
        ENGINE_CHECK_MSG(hecke::HeckeOperatorOptions::u_ell_low_rank_in_effect(p, ell, FibreRepresentation::SymN) == (p == ell),
                         "HeckeHidaHeavyCellProbe: the U_ell lever's Auto mode must be in effect exactly for U_ell");
        std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(p, 1, ell, n));
            tau.reset(new hecke::HeckeTauTableT<uint64_t>(*K, *model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnKT<uint64_t> op(&coh, cosets.get(), tau.get());
        op.compute();
        const auto& D = op.H_matrix_module(1);
        M A(D.rows, Vec(D.cols));
        for (uint32_t i = 0; i < D.rows; ++i)
            for (uint32_t j = 0; j < D.cols; ++j) A[i][j] = D.at(i, j) % o.q;
        (p == ell ? o.U : p == 2 ? o.T2 : o.T3) = std::move(A);
    }
    return o;
}

// ---- the reading -------------------------------------------------------------

struct Reading {
    uint32_t g = 0, h = 0;
    Vec chiU, chi2, chi3;
};

Reading Read(const Ops& o, Checker& chk, const std::string& tag) {
    const uint64_t q = o.q, ell = o.ell;
    Reading r;
    r.g = static_cast<uint32_t>(o.U.size());
    chk(Equal(Mul(o.U, o.T2, q), Mul(o.T2, o.U, q)), tag + " U T_2 = T_2 U on H^1");
    chk(Equal(Mul(o.U, o.T3, q), Mul(o.T3, o.U, q)), tag + " U T_3 = T_3 U on H^1");
    chk(Equal(Mul(o.T2, o.T3, q), Mul(o.T3, o.T2, q)), tag + " T_2 T_3 = T_3 T_2 on H^1");
    const M W = PowM(o.U, static_cast<uint64_t>(o.m) * r.g, q);
    const M Wt = Transpose(W);
    const std::vector<size_t> cols = IndependentModEll(Wt, ell);   // columns of W independent mod ell
    r.h = static_cast<uint32_t>(cols.size());
    if (r.h == 0) return r;
    M B(r.g, Vec(r.h));
    for (uint32_t i = 0; i < r.g; ++i)
        for (uint32_t j = 0; j < r.h; ++j) B[i][j] = W[i][cols[j]];
    const std::vector<size_t> rows = IndependentModEll(B, ell);
    ENGINE_CHECK_MSG(rows.size() == r.h, "HeckeHidaHeavyCellProbe: basis rank mismatch mod ell");
    M BS(r.h, Vec(r.h));
    for (uint32_t i = 0; i < r.h; ++i) BS[i] = B[rows[i]];
    const M BSinv = Inverse(BS, ell, q);
    auto restrict = [&](const M& T, const std::string& name) {
        const M TB = Mul(T, B, q);
        M TBS(r.h);
        for (uint32_t i = 0; i < r.h; ++i) TBS[i] = TB[rows[i]];
        const M X = Mul(BSinv, TBS, q);
        chk(Equal(Mul(B, X, q), TB), tag + " " + name + " preserves the ordinary summand (B X = T B on every row)");
        return Charpoly(X, ell, q);
    };
    r.chiU = restrict(o.U, "U");
    r.chi2 = restrict(o.T2, "T_2");
    r.chi3 = restrict(o.T3, "T_3");
    return r;
}

// (slot-14 WP-2, plan item 6: "(Hi11)/(Hi12) gated at every class member")
// The ladder families this probe knows a class-invariance conjecture for:
// (Hi11) at ell = 691, (Hi12) at ell = 2411 ([C] rmk:hidacheckable2). Before
// this package the probe compared against a LITERAL prediction and only at
// the single point (700, 691); WP-C3/WP-C4/WP-C6 instead compared every other
// class member (1390, 2080, 2770 at 691; 1210, 2420 at 2411) against the
// n = 10 reference BY HAND. This closes that gap: any requested cell with
// ell in {691, 2411} and n === 10 (mod ell - 1) gets an automatic HI11/HI12
// line, verdict HELD/FAILED against the n = 10 reference computed in THIS
// process (the m = 1 field route; cheap, ~1 s) -- cached per ell so a run
// naming several class members only pays for the reference once.
bool IsHidaCheckableLadderEll(uint64_t ell) { return ell == 691 || ell == 2411; }
const char* HidaCheckableLadderLabel(uint64_t ell) { return ell == 691 ? "HI11" : "HI12"; }

Vec ReduceModEll(const Vec& v, uint64_t ell) {
    Vec out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] % ell;
    return out;
}

struct Cell { uint32_t n; uint64_t ell; uint32_t m; };

std::vector<Cell> ParseCells(const char* s) {
    std::vector<Cell> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        Cell c{};
        char c1 = 0, c2 = 0;
        std::stringstream is(item);
        is >> c.n >> c1 >> c.ell >> c2 >> c.m;
        ENGINE_CHECK_MSG(!is.fail() && c1 == ':' && c2 == ':' && c.m >= 1 && c.m <= 3 && c.ell >= 5,
                         "HeckeHidaHeavyCellProbe: HECKE_HIDA_HEAVY_CELLS item must be n:ell:m, ell >= 5, 1 <= m <= 3");
        out.push_back(c);
    }
    return out;
}

} // namespace

uint32_t HeckeHidaHeavyCellProbe() {
    std::cout << "================= HeckeHidaHeavyCellProbe =================\n";
    const char* on = std::getenv("HECKE_HIDA_HEAVY");
    if (!(on && std::string(on) == "1")) {
        std::cout << "[SKIP] HeckeHidaHeavyCellProbe: interlocked. To run the heavy Hida cells (slot-13 WP-10):\n"
                     "    HECKE_HIDA_HEAVY=1 [HECKE_HIDA_HEAVY_CELLS=n:ell:m,...] ./hacs HeckeHidaHeavyCellProbe\n"
                     "  default cells 100:691:1,200:691:1,400:691:1,700:691:1 (the (Hi11) cell: ~15 GiB under the lever).\n";
        return 0;
    }
    const char* cl = std::getenv("HECKE_HIDA_HEAVY_CELLS");
    const std::vector<Cell> cells = ParseCells(cl && *cl ? cl : "100:691:1,200:691:1,400:691:1,700:691:1");
    Checker chk;
    // (slot-14 WP-2, item 6) The n = 10 class reference for (Hi11)/(Hi12),
    // computed at most once per ell, lazily, the first time a class member
    // needs it. Its own internal (U T_2 = T_2 U, ...) checks run against a
    // throwaway Checker: the reference is scaffolding for the REPORTED
    // HI11/HI12 verdict below, not itself a gated cell of this run.
    std::map<uint64_t, Reading> class_ref;
    auto GetClassRef = [&](uint64_t ell) -> const Reading* {
        const auto it = class_ref.find(ell);
        if (it != class_ref.end()) return &it->second;
        try {
            if (hecke::ordinary::TorsionExponentPrepass(10, ell).e_max != 0) return nullptr;
            const Ops ro = FieldRoute(10, static_cast<uint32_t>(ell));
            Checker ref_chk;
            const Reading rr = Read(ro, ref_chk, "(10, " + std::to_string(ell) + ") m=1 [class reference]");
            // (slot-14 WP-16, audit M-11) a reference that fails its own checks is not a
            // reference: no HI11/HI12 verdict is reported against it (UNAVAILABLE below).
            if (ref_chk.failures != 0) {
                std::cout << "  (10, " << ell << ") class reference: ref_checks_failing=" << ref_chk.failures
                          << " of " << ref_chk.checks << " -- not used\n";
                return nullptr;
            }
            return &class_ref.emplace(ell, rr).first->second;
        } catch (const std::exception&) {
            return nullptr;
        }
    };
    for (const Cell& c : cells) {
        const std::string tag = "(" + std::to_string(c.n) + ", " + std::to_string(c.ell) + ") m=" + std::to_string(c.m);
        try {
            const bool reset = oc_util::ResetPeakRss();
            const auto t0 = std::chrono::steady_clock::now();
            const uint32_t e_max = hecke::ordinary::TorsionExponentPrepass(c.n, c.ell).e_max;
            if (e_max != 0) {
                chk(false, tag + " e_max = " + std::to_string(e_max) + " > 0: refused (the probe reads H_m on a free H_K only)");
                continue;
            }
            ENGINE_CHECK_MSG(c.n % 2 == 0 && c.n >= 2, "HeckeHidaHeavyCellProbe: n must be even and >= 2");
            const Ops o = c.m == 1 ? FieldRoute(c.n, static_cast<uint32_t>(c.ell)) : ModuleRoute(c.n, c.ell, c.m);
            const double t_ops = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const Reading r = Read(o, chk, tag);
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double peak = oc_util::PeakRssMiB();
            const uint64_t period = (c.ell - 1) * PowMod(c.ell, c.m - 1, UINT64_MAX);
            std::cout << "HIDA_TOWER ell=" << c.ell << " class=" << (c.n % period) << " m=" << c.m << " n=" << c.n
                      << " ord_rank=" << r.h << " charpoly_T2=" << PolyText(r.chi2) << " charpoly_T3=" << PolyText(r.chi3)
                      << " charpoly_U=" << PolyText(r.chiU) << " h1_gens=" << r.g << " e_max=" << e_max
                      << " route=" << o.route << std::fixed << std::setprecision(1) << " ops_s=" << t_ops
                      << " wall_s=" << wall << " peak_rss_mib=" << (reset ? peak : -1.0) << "\n";
            // The Eisenstein half, PROVED ((Hi5)): h_ord >= 1 and 1 + p^{n+1} a root of chi^ord_p mod ell.
            chk(r.h >= 1, tag + " h_ord >= 1 (Hi5)");
            for (uint64_t p : {2ULL, 3ULL}) {
                const Vec& chi = p == 2 ? r.chi2 : r.chi3;
                if (chi.empty()) continue;
                Vec red(chi.size());
                for (size_t i = 0; i < chi.size(); ++i) red[i] = chi[i] % c.ell;
                const uint64_t root = (1 + PowMod(p, c.n + 1, c.ell)) % c.ell;
                chk(EvalPoly(red, root, c.ell) == 0,
                    tag + " 1 + " + std::to_string(p) + "^{n+1} is a root of chi^ord mod ell (Hi5)");
            }
            // Cross-route (gated, n <= 40 only): the shipped slot-12 reading
            // (hida::ComputeTowerCell, module route at K = m + e_max, no lever,
            // lem:ordexponent's projector) agrees on h_ord and chi^ord(T_2), chi^ord(T_3).
            if (c.n <= 40) {
                const hecke_ext::hida::TowerCell t = hecke_ext::hida::ComputeTowerCell(c.n, c.ell, std::max<uint32_t>(2, c.m));   // K >= 2 there
                const auto& rd = t.module_read.at(c.m - 1);
                chk(rd.ord_rank == r.h && rd.charpoly.at(0) == r.chi2 && rd.charpoly.at(1) == r.chi3,
                    tag + " equal to the shipped HeckeHidaTower reading (ord_rank, chi^ord(T_2), chi^ord(T_3))");
                std::cout << "  " << tag << ": cross-route to hida::ComputeTowerCell "
                          << (rd.ord_rank == r.h && rd.charpoly.at(0) == r.chi2 && rd.charpoly.at(1) == r.chi3 ? "equal" : "DIFFERS")
                          << "\n";
            }
            // (Hi11)/(Hi12), REPORTED not gated (slot-14 WP-2, item 6): every
            // OTHER member of the n = 10 class -- n === 10 (mod ell - 1) --
            // at ell in {691, 2411}, against the n = 10 reference computed in
            // this same process (not a literal formula: ell = 2411's class
            // has h_ord = 1, not 3, so no single closed form covers both
            // families). n = 10 itself is the reference and is not compared
            // against itself.
            if (c.n != 10 && IsHidaCheckableLadderEll(c.ell) && (c.n % (c.ell - 1)) == (10 % (c.ell - 1))) {
                const char* label = HidaCheckableLadderLabel(c.ell);
                const Reading* ref = GetClassRef(c.ell);
                if (ref == nullptr) {
                    std::cout << label << " n=" << c.n << " ell=" << c.ell
                              << " verdict=UNAVAILABLE (the n = 10 reference could not be computed, or failed its own checks)\n";
                } else {
                    bool held = r.h == ref->h;
                    for (uint64_t p : {2ULL, 3ULL}) {
                        const Vec& chi = p == 2 ? r.chi2 : r.chi3;
                        const Vec& chi_ref = p == 2 ? ref->chi2 : ref->chi3;
                        if (chi.empty() || chi_ref.empty()) { held = held && chi.empty() == chi_ref.empty(); continue; }
                        const Vec red = ReduceModEll(chi, c.ell);
                        const Vec red_ref = ReduceModEll(chi_ref, c.ell);
                        held = held && red == red_ref;
                        std::cout << "  (" << label << ") p=" << p << ": chi^ord mod " << c.ell << " = "
                                  << PolyText(red) << ", n=10 reference = " << PolyText(red_ref) << "\n";
                    }
                    std::cout << label << " n=" << c.n << " ell=" << c.ell << " h_ord=" << r.h
                              << " predicted_h_ord=" << ref->h << " verdict=" << (held ? "HELD" : "FAILED")
                              << " (CONJ part; the Eisenstein half is gated above)\n";
                }
            }
        } catch (const std::exception& e) {
            chk(false, tag + " threw: " + e.what());
        }
    }
    std::cout << "EFFECTIVE HeckeHidaHeavyCellProbe arm=" << (cl && *cl ? "cells" : "default")
              << " cells=" << cells.size() << "\n";
    std::cout << "HeckeHidaHeavyCellProbe: " << chk.checks << " checks, " << chk.failures << " failing\n";
    return chk.failures == 0 ? 0u : 1u;
}
