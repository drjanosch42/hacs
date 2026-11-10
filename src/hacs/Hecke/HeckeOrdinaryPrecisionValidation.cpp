// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-4) HeckeOrdinaryPrecisionValidation -- see the header.
//
// Arms ([C] v1.12.0-pre.2 sec:hida; slot-12 plan WP-4):
//  (A) the pre-pass on every even n in [2, 200] at ell in {5, 7, 11, 13} (and,
//      since slot-12 WP-11 (A11), every odd n <= 41 there: e_max = 0, no torsion): the
//      two stopping rules agree (inside TorsionExponentPrepass), lambda_1 is the
//      Dickson count (cl:dimH1(2)), e_max = 0 for n < ell + 1, the recorded
//      table of rmk:bocksteinexp(3); and (Hi1) on a K-mode subset: pre-pass
//      e_max and torsion == the certified word-size K-mode's (LadicCell).
//  (B) lambda_e == sum_t min(a_t, e) for the K-mode's invariant factors, every
//      e <= e_max + 1.
//  (C) Hecke on H^0 = ker(J_n mod ell^e): T_p (p = 2, 3) preserves it at every
//      e <= e_max + 1, and at e = 1 acts with the eigenvalue multiplicities of
//      prop:heckeH0(1) (1 + p) p^a on L^a D^b.
//  (D) (Hi2)(i): (U_F')^{N_ord(h, e')} mod ell^{e'} == WP-18c's certified
//      idempotent mod ell^{e'} at (10, 691) and (30, 5), idempotent, commuting
//      with T_2 and U; U^N by factors == U^N by binary powering where N fits
//      u128; (Hi2)(ii): U^{N_ord(h^1, e')} on H_{e'} (torsion included, the
//      shipped module route) is the Fitting projection, commutes with T_2, T_3,
//      and has the ordinary rank of the free-block reading.
//  (E) prop:ordinary(4): the projector at e' reduces to the one at e' - 1.
//  (F) refusals: n = 0; the precision use at ell in {2, 3} (the pre-pass itself
//      holds there: (2, 2) has Z/2, prop:torsionprepass(2)).
//  (G) LadicCell precision_source = Prepass: the same cell record as the
//      default K-mode route at three cells beyond 2^64 (word fields aside).
//  (N) negative controls: an exponent without the ceil(log) term / without
//      ell^{K-1} misses the projector; T_p - 1 misses prop:heckeH0(1);
//      K = e' reads h + 1 "free" summands at (30, 5) (cor:ordprecision(2)-(3):
//      the free quotient needs K = e' + e_max); a perturbed J_n moves e_max.
#include "HeckeOrdinaryPrecisionValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

#include "HeckeOrdinaryPart.h"
#include "LadicCell.h"
#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"
#include "HeckeOperatorOnK.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"

namespace {

using namespace hecke::ordinary;
namespace lc = hecke::ladic_cell;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 40) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

std::string Cell(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + ", " + std::to_string(ell) + ")"; }

uint64_t Pow64(uint64_t ell, uint32_t e) {
    uint64_t a = 1;
    for (uint32_t i = 0; i < e; ++i) a *= ell;
    return a;
}

// Dickson count by a: mult[a] = #{b >= 0 : a(ell+1) + b(ell^2 - ell) = n}.
std::map<uint32_t, uint32_t> DicksonCount(uint32_t n, uint64_t ell) {
    std::map<uint32_t, uint32_t> m;
    for (uint64_t a = 0; a * (ell + 1) <= n; ++a) {
        const uint64_t rest = n - a * (ell + 1);
        if (rest % (ell * ell - ell) == 0) m[static_cast<uint32_t>(a)] += 1;
    }
    return m;
}

template<class F> bool ThrowsLogicWith(F&& f, const std::string& needle) {
    try { f(); } catch (const std::logic_error& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) { return false; }
    return false;
}

// ---------------------------------------------------------------------------
// (A) / (B) / (Hi1)
// ---------------------------------------------------------------------------
void ArmPrepassGrid(Checker& g, std::map<std::string, TorsionPrepass>& keep) {
    uint32_t cells = 0, with_torsion = 0, deep = 0;
    for (uint64_t ell : {5ull, 7ull, 11ull, 13ull})
        for (uint32_t n = 2; n <= 200; n += 2) {
            const TorsionPrepass t = TorsionExponentPrepass(n, ell);
            ++cells;
            const auto dk = DicksonCount(n, ell);
            uint32_t s = 0;
            for (const auto& kv : dk) s += kv.second;
            bool ok = t.lambda.size() == t.e_max + 2 && t.lambda[1] == s && t.torsion.size() == s &&
                      t.rank_full_level == n + 1;
            if (n < ell + 1) ok = ok && t.e_max == 0;
            for (uint32_t e = 0; ok && e <= t.e_max + 1; ++e) {
                uint32_t want = 0;
                for (uint32_t a : t.torsion) want += std::min(a, e);
                ok = t.lambda[e] == want;
            }
            g.check(ok, "(A) " + Cell(n, ell) + ": lambda_1 = Dickson count, lambda_e = sum min(a_t, e), rule consistent");
            if (!t.torsion.empty()) ++with_torsion;
            if (t.e_max >= 2) ++deep;
            keep[Cell(n, ell)] = t;
        }
    // rmk:bocksteinexp(3)'s table (and WP-2's out-of-tree re-derivation).
    const std::vector<std::pair<std::string, std::vector<uint32_t>>> table = {
        {Cell(30, 5), {2}},       {Cell(60, 5), {2, 1}},    {Cell(150, 5), {3, 1, 1}}, {Cell(56, 7), {2}},
        {Cell(90, 5), {2, 1}},    {Cell(100, 5), {2, 1}},   {Cell(120, 5), {2, 1, 1}}, {Cell(12, 5), {1}},
        {Cell(20, 5), {1}},       {Cell(24, 5), {1}},       {Cell(48, 7), {1}},
    };
    for (const auto& r : table)
        g.check(keep.count(r.first) && keep[r.first].torsion == r.second, "(A) " + r.first + ": the torsion exponents of rmk:bocksteinexp(3)");
    // (slot-12 WP-11, A11) odd n: prop:torsionprepass holds for every n >= 1; at
    // odd n and ell >= 5, -I acts by -1 and 2 is a unit, so V^Gamma mod ell^e = 0:
    // lambda_1 = 0 (the Dickson count is 0 too), e_max = 0, no torsion, and the
    // Smith rule still finds full rank n + 1 at E = 2.
    uint32_t odd_cells = 0;
    for (uint64_t ell : {5ull, 7ull, 11ull, 13ull})
        for (uint32_t n = 1; n <= 41; n += 2) {
            const TorsionPrepass t = TorsionExponentPrepass(n, ell);
            ++odd_cells;
            g.check(t.e_max == 0 && t.torsion.empty() && t.lambda.size() == 2 && t.lambda[1] == 0 &&
                        t.rank_full_level == n + 1 && DicksonCount(n, ell).empty(),
                    "(A) " + Cell(n, ell) + " (odd n): lambda_1 = 0, e_max = 0, torsion = {} (-I acts by -1)");
        }
    std::cout << "  (A) odd n <= 41 at ell in {5,7,11,13}: " << odd_cells << " cells, e_max = 0, no torsion\n";
    const TorsionPrepass t691 = TorsionExponentPrepass(10, 691);
    g.check(t691.e_max == 0 && t691.torsion.empty(), "(A) (10, 691): no torsion, e_max = 0");
    std::cout << "  (A) pre-pass on " << cells << " cells (even n <= 200, ell in {5,7,11,13}): " << with_torsion
              << " with ell-torsion, " << deep << " with e_max >= 2\n";
}

void ArmKMode(Checker& g, const std::map<std::string, TorsionPrepass>& keep) {
    struct C { uint32_t n; uint64_t ell; };
    const std::vector<C> cells = {
        {6, 5},   {12, 5},  {16, 5},  {20, 5},  {24, 5},  {30, 5},  {40, 5},  {60, 5},  {90, 5},
        {100, 5}, {120, 5}, {150, 5}, {48, 7},  {56, 7},  {98, 7},  {22, 11}, {60, 11}, {26, 13},
        {84, 13}, {10, 691},
    };
    uint32_t run = 0;
    for (const C& c : cells) {
        const std::string tag = "(Hi1) " + Cell(c.n, c.ell);
        const TorsionPrepass t = c.ell == 691 ? TorsionExponentPrepass(c.n, c.ell) : keep.at(Cell(c.n, c.ell));
        const lc::WordKMode w = lc::CertifiedWordKMode(c.n, c.ell);
        ++run;
        g.check(w.certified, tag + ": the word-size K-mode certifies");
        if (!w.certified) continue;
        g.check(w.e_max == t.e_max, tag + ": pre-pass e_max (" + std::to_string(t.e_max) + ") == K-mode e_max (" +
                                        std::to_string(w.e_max) + ")");
        g.check(w.type[1].torsion == t.torsion, tag + ": pre-pass torsion multiset == the certified type of H^1");
        g.check(w.type[1].free_rank == FreeRankH1(c.n), tag + ": rank H^1 == 2 d_k - 1 (known before the run)");
        g.check(w.type[0].free_rank == 0 && w.type[0].torsion.empty() && w.type[2].torsion.empty(),
                tag + ": H^0(Gamma; V) = 0, H^2 without ell-torsion");
        // (B)
        bool b = true;
        for (uint32_t e = 0; e <= t.e_max + 1; ++e) {
            uint32_t want = 0;
            for (uint32_t a : w.type[1].torsion) want += std::min(a, e);
            b = b && t.lambda[e] == want;
        }
        g.check(b, "(B) " + Cell(c.n, c.ell) + ": lambda_e == sum_t min(a_t, e) for the K-mode's invariant factors");
        // cor:ordprecision(2): K = e' + e_max
        g.check(CertifiedWorkingPrecision(c.n, c.ell, 3) == 3 + w.e_max, tag + ": CertifiedWorkingPrecision = e' + e_max");
    }
    std::cout << "  (Hi1) the pre-pass against the certified word-size K-mode on " << run << " cells\n";
}

// ---------------------------------------------------------------------------
// (C) Hecke on H^0
// ---------------------------------------------------------------------------
bool InKernel(uint32_t n, uint64_t ell, uint32_t e, const std::vector<uint64_t>& v) {
    const uint32_t E = std::max<uint32_t>(2, e);
    const std::vector<uint64_t> J = PrepassMatrix(n, ell, E);
    const uint64_t m = Pow64(ell, e);
    const uint32_t d = n + 1;
    for (uint32_t i = 0; i < 2 * d; ++i) {
        unsigned __int128 s = 0;
        for (uint32_t j = 0; j < d; ++j) s = (s + static_cast<unsigned __int128>(J[static_cast<size_t>(i) * d + j] % m) * (v[j] % m)) % m;
        if (s != 0) return false;
    }
    return true;
}

// The matrix of T_p on the F_ell kernel basis (identity at the free columns).
std::vector<std::vector<uint64_t>> TpMatrixE1(uint32_t n, uint64_t ell, uint64_t p, bool wrong_operator,
                                              const std::vector<std::vector<uint64_t>>& B) {
    // free column of each basis vector: the first index where v == 1 and all other basis vectors are 0
    const uint32_t d = n + 1, s = static_cast<uint32_t>(B.size());
    std::vector<uint32_t> fc(s, d);
    for (uint32_t i = 0; i < s; ++i)
        for (uint32_t c = 0; c < d && fc[i] == d; ++c) {
            if (B[i][c] != 1) continue;
            bool unique = true;
            for (uint32_t j = 0; j < s; ++j) if (j != i && B[j][c] != 0) unique = false;
            if (unique) fc[i] = c;
        }
    std::vector<std::vector<uint64_t>> M(s, std::vector<uint64_t>(s, 0));
    for (uint32_t j = 0; j < s; ++j) {
        std::vector<uint64_t> w = ApplyTpOnH0(n, ell, 1, p, B[j]);
        if (wrong_operator) {
            // the negative control: T_p - 1, a wrong operator that must miss prop:heckeH0(1)
            for (uint32_t c = 0; c < d; ++c) w[c] = (w[c] + ell - B[j][c] % ell) % ell;
        }
        for (uint32_t i = 0; i < s; ++i) M[i][j] = w[fc[i]] % ell;
    }
    return M;
}

uint32_t NullityModEll(std::vector<std::vector<uint64_t>> M, uint64_t ell) {
    RingMat R;
    R.ell = ell;
    const uint32_t s = static_cast<uint32_t>(M.size());
    R.row_exp.assign(s, 1);
    for (uint32_t i = 0; i < s; ++i) for (uint32_t j = 0; j < s; ++j) R.a.push_back(M[i][j] % ell);
    return s - RankModEll(R);
}

bool EigenMultiplicitiesMatch(uint32_t n, uint64_t ell, uint64_t p, const std::vector<std::vector<uint64_t>>& M) {
    std::map<uint64_t, uint32_t> want;   // (1 + p) p^a mod ell -> multiplicity
    for (const auto& kv : DicksonCount(n, ell)) {
        uint64_t c = (1 + p) % ell;
        for (uint32_t i = 0; i < kv.first; ++i) c = c * p % ell;
        want[c] += kv.second;
    }
    uint32_t total = 0;
    for (const auto& kv : want) {
        auto A = M;
        for (size_t i = 0; i < A.size(); ++i) A[i][i] = (A[i][i] + ell - kv.first) % ell;
        const uint32_t nul = NullityModEll(A, ell);
        if (nul != kv.second) return false;
        total += nul;
    }
    return total == M.size();
}

void ArmHeckeH0(Checker& g, const std::map<std::string, TorsionPrepass>& keep) {
    uint32_t cells = 0;
    for (const std::string& key : {Cell(12, 5), Cell(30, 5), Cell(60, 5), Cell(120, 5), Cell(150, 5), Cell(56, 7), Cell(48, 7),
                                  Cell(98, 7), Cell(60, 11), Cell(84, 13)}) {
        const TorsionPrepass& t = keep.at(key);
        const uint32_t n = t.n_sym;
        const uint64_t ell = t.ell;
        ++cells;
        for (uint64_t p : {2ull, 3ull}) {
            // e = 1: eigenvalues
            const auto B = PrepassKernelGenerators(n, ell, 1);
            g.check(B.size() == t.lambda[1], "(C) " + key + ": dim ker(J_n mod ell) == lambda_1");
            bool pres = true;
            for (const auto& v : B) pres = pres && InKernel(n, ell, 1, ApplyTpOnH0(n, ell, 1, p, v));
            g.check(pres, "(C) " + key + " T_" + std::to_string(p) + ": preserves H^0(Gamma; V/ell)");
            const auto M = TpMatrixE1(n, ell, p, false, B);
            g.check(EigenMultiplicitiesMatch(n, ell, p, M),
                    "(C) " + key + " T_" + std::to_string(p) + ": eigenvalues (1+p) p^a with the Dickson multiplicities (prop:heckeH0(1))");
            // (N) a wrong operator (T_p - 1) misses them
            const auto Mbad = TpMatrixE1(n, ell, p, true, B);
            g.check(!EigenMultiplicitiesMatch(n, ell, p, Mbad), "(N) " + key + " T_" + std::to_string(p) + " - 1: the eigenvalue check fails (negative control)");
            // e >= 2: T_p preserves ker(J_n mod ell^e)
            for (uint32_t e = 2; e <= t.e_max + 1; ++e) {
                bool ok = true;
                for (const auto& v : PrepassKernelGenerators(n, ell, e)) ok = ok && InKernel(n, ell, e, ApplyTpOnH0(n, ell, e, p, v));
                g.check(ok, "(C) " + key + " T_" + std::to_string(p) + ": preserves H^0(Gamma; V/ell^" + std::to_string(e) + ")");
            }
        }
    }
    std::cout << "  (C) T_2, T_3 on H^0 = ker(J_n mod ell^e) at " << cells << " cells (prop:torsionprepass(4), prop:heckeH0(1))\n";
}

// ---------------------------------------------------------------------------
// (D) / (E): (Hi2)(i) on the free block of WP-18c
// ---------------------------------------------------------------------------
RingMat FromResidues(const std::vector<lc::Residue>& R, uint32_t r, uint64_t ell, uint32_t e) {
    RingMat M;
    M.ell = ell;
    M.row_exp.assign(r, e);
    for (uint32_t i = 0; i < r * r; ++i) {
        uint64_t v = 0, w = 1;
        for (uint32_t d = 0; d < e; ++d) { v += static_cast<uint64_t>(R[i][d]) * w; w *= ell; }
        M.a.push_back(v);
    }
    return M;
}

void ArmFreeBlock(Checker& g, std::map<std::string, lc::CellRecord>& keep) {
    struct C { uint32_t n; uint64_t ell; uint32_t k; uint32_t emax_prime; };
    for (const C& c : std::vector<C>{{10, 691, 7, 6}, {30, 5, 28, 5}, {20, 5, 28, 5}}) {
        const std::string tag = "(Hi2)(i) " + Cell(c.n, c.ell);
        lc::Request rq;
        rq.n_sym = c.n;
        rq.ell = c.ell;
        rq.k = c.k;
        rq.primes = {2, c.ell};
        rq.width = lc::EngineLadicWidth();
        const lc::CellRecord rec = lc::RunCell(rq);
        keep[Cell(c.n, c.ell)] = rec;
        const lc::HidaOut& h = rec.hida;
        g.check(rec.precision.certified && h.present && h.certified, tag + ": WP-18c's idempotent is certified");
        if (!h.present || rec.hecke.size() != 2) continue;
        const uint32_t r = h.rank;
        const lc::HeckeOut& U = rec.hecke[1];
        const lc::HeckeOut& T2 = rec.hecke[0];
        g.check(U.u_ell && U.rank_free == r && T2.rank_free == r && r == FreeRankH1(c.n),
                tag + ": the free block has rank h = 2 d_k - 1");
        RingMat prev;
        for (uint32_t e = 1; e <= c.emax_prime; ++e) {
            const RingMat Um = FromResidues(U.free_block, r, c.ell, e);
            const RingMat Tm = FromResidues(T2.free_block, r, c.ell, e);
            const RingMat Em = FromResidues(h.e, r, c.ell, e);
            const OrdinaryExponent N = OrdinaryExponentOf(r, e, c.ell);
            const RingMat P = Pow(Um, N);
            const std::string te = tag + " e'=" + std::to_string(e);
            g.check(P == Em, te + ": (U_F')^{N_ord(h, e')} mod ell^e' == WP-18c's e mod ell^e'");
            g.check(Mul(P, P) == P, te + ": idempotent");
            g.check(Mul(P, Tm) == Mul(Tm, P) && Mul(P, Um) == Mul(Um, P), te + ": commutes with T_2 and U");
            g.check(RankModEll(P) == h.rank_e_mod_ell, te + ": rank_F_ell == WP-18c's rank_e");
            // a larger multiple (lem:ordexponent(3), last sentence)
            g.check(Pow(Um, OrdinaryExponentOf(r + 1, e + 1, c.ell)) == P, te + ": N_ord(h+1, e'+1) gives the same projector");
            if (N.fits_u128()) g.check(PowU128(Um, N.value()) == P, te + ": U^N by factors == U^N by binary powering");
            // (N) without the ell^{K-1+c} factor the power is not the projector whenever U has a unipotent part
            // (checked below on synthetic matrices); here: U^1 != P unless U is already idempotent
            if (e >= 2) g.check(Reduce(P, e - 1) == prev, "(E) " + te + ": reduces to the projector at e' - 1 (prop:ordinary(4))");
            prev = P;
        }
        std::cout << "  (Hi2)(i) " << Cell(c.n, c.ell) << " k=" << c.k << ": U_ell^{N_ord} == the Fitting idempotent mod ell^e', e' <= "
                  << c.emax_prime << " (h = " << r << ", rank e = " << h.rank_e_mod_ell << ", chi_W = x^" << h.nilpotent_dim << " p)\n";
    }
}

// (Hi2)(ii): on H_{e'} (torsion included), the shipped module route.
struct ModuleRun {
    std::vector<uint32_t> exps;              // H^1 cyclic exponents
    RingMat U, T2, T3;
};

RingMat FromDense(const engine::DenseBuf<uint64_t>& A, uint64_t ell, const std::vector<uint32_t>& ex) {
    RingMat M;
    M.ell = ell;
    M.row_exp = ex;
    for (uint32_t i = 0; i < A.rows; ++i) for (uint32_t j = 0; j < A.cols; ++j) M.a.push_back(A.at(i, j));
    return Canonical(M);
}

ModuleRun RunModule(uint32_t n, uint64_t ell, uint32_t k) {
    ModuleRun out;
    const hecke::ScopedCoefficientRing ring(k);
    const engine::ModCtx<uint64_t> ctx = hecke::CoefficientRingContext<uint64_t>("HeckeOrdinaryPrecisionValidation", ell, k);
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*K));
    }
    LocalSystemSimplicialComplexT<uint64_t> lssc(*K, hecke::GenerateSL2ZRelationsRing<uint64_t>(*K, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
    coh.compute_cohomology();
    out.exps = coh.module_degree(1).dec.exponents;
    for (uint64_t p : std::vector<uint64_t>{ell, 2, 3}) {
        std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(p, 1, ell, n));
            tau.reset(new hecke::HeckeTauTableT<uint64_t>(*K, *model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnKT<uint64_t> op(&coh, cosets.get(), tau.get());
        op.compute();
        const RingMat M = FromDense(op.H_matrix_module(1), ell, out.exps);
        if (p == ell) out.U = M; else if (p == 2) out.T2 = M; else out.T3 = M;
    }
    return out;
}

void ArmModule(Checker& g, const std::map<std::string, lc::CellRecord>& digit) {
    struct C { uint32_t n; uint64_t ell; uint32_t e; };
    for (const C& c : std::vector<C>{{30, 5, 2}, {30, 5, 3}, {60, 5, 2}, {10, 691, 2}}) {
        const std::string tag = "(Hi2)(ii) " + Cell(c.n, c.ell) + " e'=" + std::to_string(c.e);
        const ModuleRun m = RunModule(c.n, c.ell, c.e);
        const uint32_t g1 = static_cast<uint32_t>(m.exps.size());   // h^1(n): the generator count of H_{e'}
        const TorsionPrepass t = TorsionExponentPrepass(c.n, c.ell);
        g.check(g1 == FreeRankH1(c.n) + t.lambda[1], tag + ": H_{e'} has h^1(n) = h + lambda_1 cyclic summands (rmk:ordexponentuse(2))");
        uint32_t len = 0;
        for (uint32_t x : m.exps) len += x;
        const OrdinaryExponent N = OrdinaryExponentOf(g1, c.e, c.ell);
        const RingMat P = Pow(m.U, N);
        const RingMat I = Identity(c.ell, m.exps);
        g.check(Mul(P, P) == P, tag + ": U^{N_ord(h^1, e')} is idempotent");
        g.check(Mul(P, m.U) == Mul(m.U, P) && Mul(P, m.T2) == Mul(m.T2, P) && Mul(P, m.T3) == Mul(m.T3, P),
                tag + ": commutes with U, T_2, T_3");
        g.check(IsZero(PowU128(Mul(m.U, Sub(I, P)), len)), tag + ": U is nilpotent on (1 - P) H_{e'} (Fitting)");
        g.check(Pow(Mul(m.U, P), N) == P, tag + ": U is invertible on P H_{e'} ((UP)^N == P)");
        // the ordinary part is free: P kills the torsion-looking summands' contribution -- its rank mod ell is h_ord
        auto it = digit.find(Cell(c.n, c.ell));
        if (it != digit.end() && it->second.hida.present)
            g.check(RankModEll(P) == it->second.hida.rank_e_mod_ell,
                    tag + ": ordinary rank on H_{e'} == the free-block reading at K = e' + e_max (cor:ordprecision(4))");
        std::cout << "  " << tag << ": summands " << g1 << " (length " << len << "), ordinary rank " << RankModEll(P) << "\n";
        // (N) cor:ordprecision(2)-(3): the free quotient needs K = e' + e_max
        if (t.e_max >= c.e) {
            uint32_t top = 0;
            for (uint32_t x : m.exps) if (x == c.e) ++top;
            g.check(top > FreeRankH1(c.n), "(N) " + tag + ": at K = e' the torsion masquerades as free (" + std::to_string(top) +
                                               " summands of order ell^e' > h = " + std::to_string(FreeRankH1(c.n)) + ")");
        }
    }
    // cor:ordprecision(3) at K = e' + e_max: exactly h summands of order ell^K.
    for (const auto& pr : std::vector<std::pair<uint32_t, uint32_t>>{{30, 2}, {60, 1}}) {
        const uint32_t n = pr.first;
        const uint32_t K = CertifiedWorkingPrecision(n, 5, pr.second);
        const ModuleRun m = RunModule(n, 5, K);
        uint32_t top = 0;
        bool small = true;
        for (uint32_t x : m.exps) { if (x == K) ++top; else small = small && x <= K - pr.second; }
        g.check(top == FreeRankH1(n) && small, "(Hi2) " + Cell(n, 5) + " K=" + std::to_string(K) +
                                                   ": h summands of order ell^K, the others of order <= ell^{e_max} (cor:ordprecision(3))");
    }
}

// ---------------------------------------------------------------------------
// (N) lem:ordexponent's terms (rmk:ordexponentuse(1)) and a perturbed J_n
// ---------------------------------------------------------------------------
void ArmExponentControls(Checker& g) {
    // Jordan block 1 + E of size 3 over F_5: order 5 = ell^c.
    {
        RingMat J;
        J.ell = 5;
        J.row_exp = {1, 1, 1};
        J.a = {1, 1, 0, 0, 1, 1, 0, 0, 1};
        const RingMat I = Identity(5, J.row_exp);
        g.check(OrdinaryProjector(J) == I, "(N) Jordan block 1+E (g = 3, K = 1, ell = 5): U^{N_ord} = 1 = e_ord(U)");
        OrdinaryExponent bad = OrdinaryExponentOf(3, 1, 5);
        bad.c = 0;   // drop the ceil(log_ell g) term
        g.check(Pow(J, bad) != I, "(N) Jordan block: without the ceil(log_ell g) term the power is NOT the projector");
    }
    // 1 + ell in (Z/ell^K)^x, ell = 5, K = 2: order 5 = ell^{K-1}.
    {
        RingMat u;
        u.ell = 5;
        u.row_exp = {2};
        u.a = {6};
        const RingMat I = Identity(5, u.row_exp);
        g.check(OrdinaryProjector(u) == I, "(N) 1 + ell mod ell^2: U^{N_ord(1, 2)} = 1");
        OrdinaryExponent bad = OrdinaryExponentOf(1, 1, 5);   // K - 1 = 0: drops ell^{K-1}
        g.check(Pow(u, bad) != I, "(N) 1 + ell mod ell^2: without ell^{K-1} the power is NOT the projector");
    }
    // a nilpotent-plus-unit split: U = diag(0-Jordan(2), 2) mod 7 -> e_ord = diag(0, 0, 1)
    {
        RingMat U;
        U.ell = 7;
        U.row_exp = {1, 1, 1};
        U.a = {0, 1, 0, 0, 0, 0, 0, 0, 2};
        RingMat want = U;
        want.a = {0, 0, 0, 0, 0, 0, 0, 0, 1};
        g.check(OrdinaryProjector(U) == want, "(D) U = N (+) 2 over F_7: U^{N_ord} = the Fitting projection diag(0, 0, 1)");
    }
    // N_ord(g, K) >= g K and the log bound (lem:ordexponent(1), rmk:ordexponentuse(3))
    bool ge = true;
    for (uint64_t ell : {5ull, 7ull, 691ull})
        for (uint32_t gg = 1; gg <= 4; ++gg)
            for (uint32_t K = 1; K <= 3; ++K) {
                const OrdinaryExponent N = OrdinaryExponentOf(gg, K, ell);
                if (N.fits_u128()) ge = ge && N.value() >= static_cast<unsigned __int128>(gg) * K;
            }
    g.check(ge, "(D) N_ord(g, K) >= g K wherever it fits u128 (lem:ordexponent(1))");
    // (N) a perturbed J_n: dropping rho(T) (the S-invariants only) moves the invariants at (30, 5).
    {
        const TorsionPrepass t = TorsionExponentPrepass(30, 5);
        const std::vector<uint64_t> J = PrepassMatrix(30, 5, 3);
        const uint32_t d = 31;
        // the S-block alone has a kernel over Q (S has order 4): its rank is < n + 1, so its "e_max" is unbounded
        RingMat S;
        S.ell = 5;
        S.row_exp.assign(d, 1);
        for (uint32_t i = 0; i < d; ++i) for (uint32_t j = 0; j < d; ++j) S.a.push_back(J[static_cast<size_t>(i) * d + j] % 5);
        g.check(d - RankModEll(S) > t.lambda[1], "(N) (30, 5): rho(S) - 1 alone has a larger kernel mod 5 than J_n -- T is needed");
    }
    std::cout << "  (N) exponent terms, Fitting split, perturbed J_n: controls run\n";
}

// ---------------------------------------------------------------------------
// (F) refusals
// ---------------------------------------------------------------------------
void ArmRefusals(Checker& g) {
    g.check(ThrowsLogicWith([] { (void)CertifiedWorkingPrecision(10, 2, 1); }, "prop:torsionprepass(6)"),
            "(F) ell = 2: the precision use is refused naming prop:torsionprepass(6)");
    g.check(ThrowsLogicWith([] { (void)CertifiedWorkingPrecision(10, 3, 1); }, "prop:torsionprepass(6)"),
            "(F) ell = 3: the precision use is refused naming prop:torsionprepass(6)");
    g.check(ThrowsLogicWith([] { (void)TorsionExponentPrepass(0, 5); }, "prop:torsionprepass"),
            "(F) n = 0: the pre-pass is refused naming prop:torsionprepass");
    g.check(ThrowsLogicWith([] { (void)CertifiedWorkingPrecision(0, 5, 1); }, "prop:torsionprepass"),
            "(F) n = 0: the precision use is refused");
    g.check(ThrowsLogicWith([] { (void)CertifiedWorkingPrecision(10, 5, 0); }, "cor:ordprecision"),
            "(F) e' = 0 is refused naming cor:ordprecision");
    // the pre-pass itself holds at every prime (clause (6)): (2, 2) has Z/2 (clause (2)'s hedge, R2)
    const TorsionPrepass t22 = TorsionExponentPrepass(2, 2);
    g.check(t22.torsion == std::vector<uint32_t>({1}) && t22.e_max == 1, "(F) (2, 2): the pre-pass reads Z/2 (prop:torsionprepass(2), R2)");
    bool refused = false;
    try {
        lc::Request rq;
        rq.n_sym = 10;
        rq.ell = 3;
        rq.k = 41;
        rq.primes = {2};
        rq.precision_source = lc::PrecisionSource::Prepass;
        (void)lc::RunCell(rq);
    } catch (const std::exception&) { refused = true; }
    g.check(refused, "(F) LadicCell prepass at ell = 3 is refused");
}

// ---------------------------------------------------------------------------
// (G) LadicCell precision_source = Prepass
// ---------------------------------------------------------------------------
std::string StripWordFields(const std::string& s) {
    static const std::regex re(" word_level=[0-9]+| word=[01]| prepass=1");
    return std::regex_replace(s, re, "");
}

void ArmLadicPrepass(Checker& g, const std::map<std::string, lc::CellRecord>& keep) {
    for (const auto& kv : keep) {
        const lc::CellRecord& a = kv.second;
        lc::Request rq;
        rq.n_sym = a.n_sym;
        rq.ell = a.ell;
        rq.k = a.precision.requested;
        rq.primes = {2, a.ell};
        rq.width = lc::EngineLadicWidth();
        rq.precision_source = lc::PrecisionSource::Prepass;
        const lc::CellRecord b = lc::RunCell(rq);
        const std::string tag = "(G) " + kv.first + " k=" + std::to_string(rq.k);
        g.check(b.precision.prepass && !a.precision.prepass && b.precision.word_level == 0 && a.precision.word_certified,
                tag + ": the pre-pass route ran no word-size run; the default did");
        g.check(b.precision.e_max == a.precision.e_max && b.precision.working == a.precision.working && b.precision.certified,
                tag + ": same e_max and K = k + e_max, certified (digit K-mode + UCT)");
        g.check(b.canonical_text().find(" prepass=1\n") != std::string::npos &&
                    a.canonical_text().find(" prepass=") == std::string::npos,
                tag + ": the pre-pass record says prepass=1, the default record carries no prepass field (A8)");
        g.check(StripWordFields(b.canonical_text()) == StripWordFields(a.canonical_text()),
                tag + ": the cell record is byte-identical to the default route's (word fields aside)");
    }
    std::cout << "  (G) LadicCell precision_source = prepass == the default K-mode route at " << keep.size() << " cells beyond 2^64\n";
}

} // namespace

uint32_t HeckeOrdinaryPrecisionValidation() {
    std::cout << "=== HeckeOrdinaryPrecisionValidation (slot-12 WP-4: the torsion pre-pass and U_ell^N) ===\n";
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::map<std::string, TorsionPrepass> pre;
    std::map<std::string, lc::CellRecord> digit;
    auto guarded = [&](const char* name, const std::function<void()>& f) {
        try { f(); } catch (const std::exception& e) { g.check(false, std::string(name) + ": threw " + e.what()); }
    };
    guarded("(A)", [&] { ArmPrepassGrid(g, pre); });
    guarded("(Hi1)", [&] { ArmKMode(g, pre); });
    guarded("(C)", [&] { ArmHeckeH0(g, pre); });
    guarded("(Hi2)(i)", [&] { ArmFreeBlock(g, digit); });
    guarded("(Hi2)(ii)", [&] { ArmModule(g, digit); });
    guarded("(N)", [&] { ArmExponentControls(g); });
    guarded("(F)", [&] { ArmRefusals(g); });
    guarded("(G)", [&] { ArmLadicPrepass(g, digit); });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeOrdinaryPrecisionValidation: " << g.checks << " checks, " << g.fails << " failing, " << s << " s\n";
    return g.fails;
}
