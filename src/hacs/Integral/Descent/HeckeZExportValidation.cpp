// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZ package EXPORT (slot 25): HeckeZExportValidation.  See the header for the arms.
#include "HeckeZExportValidation.h"

#include "DescentAssembly.h"
#include "DescentHecke.h"
#include "DescentModuleExport.h"
#include "DescentRecord.h"

#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNSparse.h"
#include "../../Hecke/HeckeOperatorFamily.h"
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../HeckeExt/HeckeRingHeckeAlgebra.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace mx = integral_descent::modexport;
namespace h3 = integral_descent::hecke3;
namespace ra = hecke_ext::ringalg;
using clk = std::chrono::steady_clock;

uint32_t g_fail = 0;            // failed checks of the current arm
uint64_t g_checks = 0;
uint32_t g_record_mismatch = 0;  // descent_records_identical
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string Join(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}
uint64_t PowU64(uint64_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) x *= ell;
    return x;
}
std::string Cell(uint32_t N, uint32_t n, uint32_t ell) {
    return "(" + std::to_string(N) + "," + std::to_string(n) + ") ell=" + std::to_string(ell);
}

// ---- the export as the console route's OpModule -------------------------------------------------------------------
ra::OpModule ToOpModule(const mx::ModuleAtK& M) {
    ra::OpModule O;
    O.ell = M.ell;
    O.exps = M.exps;
    for (uint32_t p : M.ps) O.primes.push_back(p);
    for (const auto& a : M.ops) {
        hecke::ordinary::RingMat R;
        R.ell = M.ell;
        R.row_exp = M.exps;
        R.a = a;
        O.op.push_back(hecke::ordinary::Canonical(R));
    }
    return O;
}
// L / ell^L: the free block of every T_p reduced to level L (<= K)
ra::OpModule FreeOpModule(const mx::ModuleAtK& M, uint32_t L) {
    ra::OpModule O;
    O.ell = M.ell;
    O.exps.assign(M.free_rank, L);
    for (uint32_t p : M.ps) O.primes.push_back(p);
    for (size_t o = 0; o < M.ops.size(); ++o) {
        const integral_descent::maps::Mat B = M.free_block(o);
        hecke::ordinary::RingMat R;
        R.ell = M.ell;
        R.row_exp = O.exps;
        for (uint32_t x : B.a) R.a.push_back(x);
        O.op.push_back(hecke::ordinary::Canonical(R));
    }
    return O;
}
// the free summands (exponent K) of a module at K; its characteristic polynomial of op o mod ell^e
std::vector<uint32_t> FreeCharpolyOf(const ra::OpModule& O, uint32_t K, size_t o, uint32_t e) {
    std::vector<uint32_t> idx;
    for (uint32_t i = 0; i < O.rank(); ++i)
        if (O.exps[i] == K) idx.push_back(i);
    integral_descent::maps::Mat B(static_cast<uint32_t>(idx.size()), static_cast<uint32_t>(idx.size()));
    for (uint32_t a = 0; a < idx.size(); ++a)
        for (uint32_t b = 0; b < idx.size(); ++b) B.at(a, b) = static_cast<uint32_t>(O.op[o].at(idx[a], idx[b]));
    if (idx.empty()) return {1};
    return h3::CharpolyModPrimePower(B, static_cast<uint32_t>(O.ell), e);
}
std::vector<uint32_t> ReduceAll(std::vector<uint32_t> v, uint64_t Q) {
    for (auto& x : v) x = static_cast<uint32_t>(x % Q);
    return v;
}
std::string Reading(const ra::AlgebraReading& R) {
    std::ostringstream o;
    o << "len=" << R.length << " dim=" << R.dim_mod_ell << " hilb=";
    for (size_t i = 0; i < R.hilbert.size(); ++i) o << (i ? "," : "") << R.hilbert[i];
    return o.str();
}
// per F_ell-rational joint system of the module: the reading of the local algebra at `shift`
std::map<std::vector<uint64_t>, std::string> SystemReadings(const ra::OpModule& O, uint32_t shift) {
    std::map<std::vector<uint64_t>, std::string> out;
    for (const ra::JointSystem& s : ra::RationalSystems(O)) {
        const ra::Block B = ra::Localize(O, s.a);
        out[s.a] = "sysdim=" + std::to_string(s.dim) + " " + Reading(ra::ReadAlgebra(B.local, s.a, shift));
    }
    return out;
}
std::string Show(const std::map<std::vector<uint64_t>, std::string>& m) {
    std::ostringstream o;
    for (const auto& [a, r] : m) {
        o << " {";
        for (size_t i = 0; i < a.size(); ++i) o << (i ? "," : "") << a[i];
        o << ": " << r << "}";
    }
    return o.str();
}

// ---- descent_records_identical: the export's torsion block against the descent's own q = 1 read ----------------------
void CompareWithDescentRead(const mx::ModuleExport& E) {
    if (!E.ok) return;
    const h3::TorsionEntry D = E.ell <= 3 ? h3::TorsionOnTree(E.ell, E.N, E.n, 1, E.ps) : h3::TorsionOnCokernel(E.ell, E.N, E.n, E.ps);
    const std::string tag = "[records] " + Cell(E.N, E.n, E.ell);
    const uint32_t nj = static_cast<uint32_t>(E.sigma.size());
    // no ell-torsion: the descent's read applies no operator (its invariants list is empty), the sigma lists must agree
    bool same = !D.refused && D.read.sigma == E.sigma && (nj == 0 ? D.inv.empty() || D.inv.size() == E.T.size() : D.inv.size() == E.T.size());
    for (size_t o = 0; same && nj > 0 && o < E.T.size(); ++o) {
        integral_descent::maps::Mat Phi(nj, nj);
        for (uint32_t a = 0; a < nj; ++a)
            for (uint32_t b = 0; b < nj; ++b) Phi.at(a, b) = E.T[o].at(a, b);
        const h3::TorsionInvariants I = h3::Invariants(E.ell, E.sigma, Phi, true, E.K_run - E.sigma_max);
        same = I.ok && D.inv[o].ok && I.canon == D.inv[o].canon;
        if (!same) std::cout << "  " << tag << " p=" << E.ps[o] << " export " << I.canon << " vs descent " << D.inv[o].canon << std::endl;
    }
    if (!check(same, tag + ": the export's torsion block reads the descent's torsion module (sigma, invariants)")) ++g_record_mismatch;
}

mx::ModuleExport Export(uint32_t N, uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps) {
    mx::ModuleExport E = mx::ExportModule(N, n, ell, ps);
    std::cout << "  EXPORT " << Cell(N, n, ell) << " ps=" << Join(ps) << " route=" << E.route << " status=" << E.status()
              << " K_run=" << E.K_run << " K_max=" << E.K_max << " b1=" << E.free_rank << " sigma=" << Join(E.sigma)
              << " sigma1_max=" << E.sigma1_max << " c=" << E.cochain_dim << " seconds=" << std::fixed << std::setprecision(3) << E.seconds
              << " rss_pred_mib=" << (E.rss_pred >> 20) << (E.ok ? "" : " " + E.refusal) << std::endl;
    check(E.ok, "export " + Cell(N, n, ell) + " ok: " + E.refusal);
    // the descent's own q = 1 read builds U32Ctx::prime_power(ell, MaxK(ell)), which needs K >= 2: at ell >= 46341 (K_run = 1) it is
    // not a comparison route (the record never reads such an ell: its bad primes are small); there sigma = [] is certified by rank
    if (E.ok && E.K_run >= 2) CompareWithDescentRead(E);
    else if (E.ok) std::cout << "  [records] " << Cell(N, n, ell) << ": K_run = 1, no descent torsion read to compare (sigma = " << Join(E.sigma) << ")" << std::endl;
    return E;
}

// ---- (a) / (f): the export against the console's Z/ell^K module route ----------------------------------------------
void CompareWithConsole(const mx::ModuleExport& E, uint32_t K) {
    const std::string tag = "(a) " + Cell(E.N, E.n, E.ell) + " K=" + std::to_string(K);
    const mx::ModuleAtK M = E.at(K);
    const ra::OpModule A = ToOpModule(M);
    std::vector<uint64_t> ps64(E.ps.begin(), E.ps.end());
    const ra::OpModule C = ra::RunModule(E.n, E.ell, K, ps64);
    std::vector<uint32_t> ce = C.exps;
    std::sort(ce.rbegin(), ce.rend());
    check(M.invariant_factors() == ce, tag + ": invariant factors " + Join(M.invariant_factors()) + " vs console " + Join(ce));
    uint32_t t = 0;
    for (uint32_t i = 0; i < M.torsion_count; ++i) t = std::max(t, M.exps[i]);
    for (size_t o = 0; t < K && o < E.ps.size(); ++o) {
        const uint64_t Q = PowU64(E.ell, K - t);
        const std::vector<uint32_t> mine = ReduceAll(M.free_charpoly(o), Q);
        const std::vector<uint32_t> theirs = FreeCharpolyOf(C, K, o, K - t);
        check(mine == theirs, tag + " T_" + std::to_string(E.ps[o]) + ": char(free) mod ell^" + std::to_string(K - t) + " " + Join(mine) +
                                  " vs console " + Join(theirs));
    }
    const auto ra0 = SystemReadings(A, 0), rc0 = SystemReadings(C, 0);
    check(ra0 == rc0, tag + ": A_K per system:" + Show(ra0) + " vs console" + Show(rc0));
    if (t > 0 && t < K) {
        const auto ra1 = SystemReadings(A, t), rc1 = SystemReadings(C, t);
        check(ra1 == rc1, tag + ": free-part image (shift " + std::to_string(t) + ") per system:" + Show(ra1) + " vs console" + Show(rc1));
    }
    std::cout << "  " << tag << ": types " << Join(ce) << "; A_K" << Show(ra0) << std::endl;
}

// ---- the coefficient of char_Z mod Q (prop:crtcharpolyN's output rule) ---------------------------------------------
bool CoefModQ(const h3::CharpolyZ& C, uint32_t i, uint64_t Q, uint32_t* out) {
    const h3::CrtCoefficient& c = C.c.at(i);
    if (!c.certified) return false;
    if (c.fits) {
        const int64_t q = static_cast<int64_t>(Q);
        int64_t v = c.value % q;
        if (v < 0) v += q;
        *out = static_cast<uint32_t>(v);
        return true;
    }
    const std::vector<uint32_t> pre(C.primes.begin(), C.primes.begin() + c.primes_used);
    *out = static_cast<uint32_t>(h3::MixedRadixModQ(c.digits, pre, c.negative, Q));
    return true;
}
void CompareWithCharZ(const mx::ModuleExport& E, uint32_t K) {
    for (size_t o = 0; o < E.ps.size(); ++o) {
        const uint32_t p = E.ps[o];
        const std::string tag = "(c) " + Cell(E.N, E.n, E.ell) + " T_" + std::to_string(p) + " K=" + std::to_string(K);
        const h3::CharpolyZ C = h3::CharpolyOverZ(E.N, E.n, p);
        if (!check(!C.refused && C.determined && C.r == E.free_rank, tag + ": char_Z determined, r = b_1")) continue;
        const uint64_t Q = PowU64(E.ell, K);
        std::vector<uint32_t> z(C.r + 1);
        bool ok = true;
        for (uint32_t i = 0; i <= C.r; ++i) ok = ok && CoefModQ(C, i, Q, &z[i]);
        const std::vector<uint32_t> mine = E.at(K).free_charpoly(o);
        check(ok && mine == z, tag + ": char(T_p | free) " + Join(mine) + " vs char_Z mod ell^K " + Join(z));
        std::cout << "  " << tag << ": char_Z = " << C.text() << std::endl;
    }
}

// ---- (d): T_p on H^0(Sym^n Z/ell^K) by the shipped module route (RunModule's recipe in degree 0) --------------------
std::vector<std::vector<uint64_t>> H0Hecke(uint32_t n, uint32_t ell, uint32_t K, const std::vector<uint32_t>& ps, std::vector<uint32_t>& exps) {
    using Elt = uint32_t;
    const hecke::ScopedCoefficientRing ring(K);
    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const engine::ModCtx<Elt> ctx = hecke::CoefficientRingContext<Elt>("HeckeZExportValidation", ell, K);
    LocalSystemSimplicialComplexT<Elt> lssc(*Kx, hecke::GenerateSL2ZRelationsRing<Elt>(*Kx, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<Elt> coh(&lssc);
    coh.compute_cohomology();
    exps = coh.module_degree(0).dec.exponents;
    std::vector<std::vector<uint64_t>> out;
    for (uint32_t p : ps) {
        std::unique_ptr<hecke::HeckeCosetTablesT<Elt>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<Elt>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<Elt>(p, 1, static_cast<Elt>(ell), n));
            tau.reset(new hecke::HeckeTauTableT<Elt>(*Kx, *model, *cosets, static_cast<Elt>(ell), n, hecke::TauVertexPolicy::GeometricNearest,
                                                     hecke::TauTableContents::Full));
        }
        hecke::HeckeOperatorOnKT<Elt> opk(&coh, cosets.get(), tau.get());
        opk.compute();
        const engine::DenseBuf<Elt>& A = opk.H_matrix_module(0);
        std::vector<uint64_t> a;
        for (uint32_t i = 0; i < A.rows; ++i)
            for (uint32_t j = 0; j < A.cols; ++j) a.push_back(A.at(i, j));
        out.push_back(std::move(a));
    }
    return out;
}

// ---- (g): the full module is a commutative algebra of endomorphisms (audit F4: the mixed block at ell in {2, 3}) -----------
// M = (+)_i Z/ell^{e_i}; a matrix A (row i mod ell^{e_i}) is an endomorphism iff A_ik ell^{e_k} = 0 mod ell^{e_i}; then (AB)_ij =
// sum_k A_ik B_kj mod ell^{e_i} is well defined.  Checks every op's endomorphism condition and T_p T_q = T_q T_p entrywise, on the
// whole matrix (torsion block, free block and the block "free generator -> torsion coordinates").  Returns the number of non-zero
// entries of the mixed block (over all ops).
uint64_t CheckModuleAlgebra(const mx::ModuleAtK& M, const std::string& tag) {
    const uint32_t r = M.rank();
    std::vector<uint64_t> q(r);
    for (uint32_t i = 0; i < r; ++i) q[i] = PowU64(M.ell, M.exps[i]);
    uint64_t mixed = 0;
    for (size_t o = 0; o < M.ops.size(); ++o) {
        const auto& A = M.ops[o];
        bool endo = true;
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t k = 0; k < r; ++k) endo = endo && (A[size_t(i) * r + k] % q[i]) * (q[k] % q[i]) % q[i] == 0;
        check(endo, tag + " T_" + std::to_string(M.ps[o]) + ": an endomorphism (A_ik ell^{e_k} = 0 mod ell^{e_i})");
        for (uint32_t i = 0; i < M.torsion_count; ++i)
            for (uint32_t j = M.torsion_count; j < r; ++j) mixed += A[size_t(i) * r + j] != 0;
    }
    for (size_t o = 0; o < M.ops.size(); ++o)
        for (size_t o2 = o + 1; o2 < M.ops.size(); ++o2) {
            const auto& A = M.ops[o];
            const auto& B = M.ops[o2];
            uint32_t bad = 0;
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) {
                    uint64_t ab = 0, ba = 0;
                    for (uint32_t k = 0; k < r; ++k) {
                        ab = (ab + A[size_t(i) * r + k] % q[i] * (B[size_t(k) * r + j] % q[i])) % q[i];
                        ba = (ba + B[size_t(i) * r + k] % q[i] * (A[size_t(k) * r + j] % q[i])) % q[i];
                    }
                    bad += ab != ba;
                }
            check(bad == 0, tag + ": T_" + std::to_string(M.ps[o]) + " T_" + std::to_string(M.ps[o2]) + " = T_" + std::to_string(M.ps[o2]) + " T_" +
                                std::to_string(M.ps[o]) + " on the full module (" + std::to_string(bad) + " differing entries)");
        }
    return mixed;
}

// ---- (c), K = 1: char(T_p | H^1/ell) against level_n's cokernel matrices over F_ell ---------------------------------------
void CompareWithLevelNK1(const mx::ModuleExport& E) {
    const hecke::level::Gamma0LevelN lvl(E.N);
    const engine::ModCtx<uint32_t> f(E.ell);
    const auto Hm = hecke::level::CokernelHeckeMatrices(lvl, E.n, f, E.ps);
    const mx::ModuleAtK M = E.at(1);
    const std::string tag = "(c) " + Cell(E.N, E.n, E.ell);
    for (size_t o = 0; o < E.ps.size(); ++o) {
        const auto it = Hm.find(E.ps[o]);
        if (!check(it != Hm.end() && it->second.rows == M.rank(), tag + " level_n T_" + std::to_string(E.ps[o]) + " dim = rank H^1/ell")) continue;
        std::vector<uint32_t> a;
        for (uint64_t x : it->second.a) a.push_back(static_cast<uint32_t>(x % E.ell));
        const std::vector<uint32_t> lv = h3::CharpolyFp(a, it->second.rows, E.ell);
        const std::vector<uint32_t> mine = M.free_charpoly(o);
        check(lv == mine, tag + " T_" + std::to_string(E.ps[o]) + " K=1: char " + Join(mine) + " vs level_n " + Join(lv));
        std::cout << "  " << tag << " T_" << E.ps[o] << " K=1: char " << Join(mine) << " = level_n" << std::endl;
    }
}

// ---- the arms ---------------------------------------------------------------------------------------------------------
void ArmA() {
    {
        const mx::ModuleExport E = Export(1, 10, 691, {2, 3, 5});
        if (E.ok) CompareWithConsole(E, 2);
    }
    {
        const mx::ModuleExport E = Export(1, 26, 5, {2, 3, 7});
        if (E.ok) CompareWithConsole(E, 2);
    }
}

void ArmB() {
    struct CellB {
        uint32_t N, n;
        std::vector<uint32_t> ells;
        std::vector<uint32_t> ps;
    };
    const std::vector<CellB> cells = {{1, 10, {2, 3}, {2}}, {2, 6, {2, 3, 5}, {3}}, {8, 10, {2, 3}, {3}}};
    for (const CellB& c : cells) {
        integral_descent::record::DescentRecordZ R;
        {
            const hecke::ScopedCoefficientRing z(0);
            integral_descent::record::RecordOptions o;
            o.hecke = false;
            o.pair = false;
            R = integral_descent::record::BuildDescentRecord(c.N, c.n, o);
        }
        if (!check(R.status == "ok" && R.H.size() >= 2, "(b) the descent record's groups at (" + std::to_string(c.N) + "," + std::to_string(c.n) + ")"))
            continue;
        const auto& H1 = R.H[1];
        for (uint32_t ell : c.ells) {
            const std::string tag = "(b) " + Cell(c.N, c.n, ell);
            const mx::ModuleExport E = Export(c.N, c.n, ell, c.ps);
            if (!E.ok) continue;
            std::vector<uint32_t> a;
            const auto it = H1.torsion.find(ell);
            if (it != H1.torsion.end()) a = it->second;
            std::sort(a.begin(), a.end());
            std::vector<uint32_t> s = E.sigma;
            std::sort(s.begin(), s.end());
            check(E.free_rank == H1.free_rank && s == a, tag + ": b_1 = " + std::to_string(E.free_rank) + ", tors " + Join(s) + " vs record H^1 = " +
                                                             H1.to_string());
            std::ostringstream line;
            for (uint32_t K = 1; K <= std::min<uint32_t>(4, E.K_max); ++K) {
                std::vector<uint32_t> pred(H1.free_rank, K);
                for (uint32_t x : a) pred.push_back(std::min(x, K));
                std::sort(pred.rbegin(), pred.rend());
                const std::vector<uint32_t> got = E.at(K).invariant_factors();
                check(got == pred, tag + " K=" + std::to_string(K) + ": H^1 (x) Z/ell^K " + Join(got) + " vs B2(1) from the record " + Join(pred));
                line << " K" << K << "=" << Join(got);
            }
            std::cout << "  " << tag << ": record H^1 = " << H1.to_string() << ";" << line.str() << std::endl;
        }
    }
    // refusals by name (R-E1): U_p (p | N) at ell = 2 and 3, a non-prime ell
    for (uint32_t ell : {2u, 3u}) {
        const mx::ModuleExport E = mx::ExportModule(8, 10, ell, {2});
        check(!E.ok && E.code == mx::ExportRefusal::PDividesN && E.status() == "blocked(export)" && E.refusal.find("[PDividesN]") != std::string::npos,
              "(b) (8,10) ell=" + std::to_string(ell) + " p=2 | N refuses [PDividesN] blocked(export): " + E.refusal);
    }
    {
        const mx::ModuleExport E = mx::ExportModule(1, 10, 4, {2});
        check(!E.ok && E.code == mx::ExportRefusal::BadEll, "(b) ell = 4 refuses [BadEll]: " + E.refusal);
    }
}

void ArmC() {
    {
        const mx::ModuleExport E = Export(2, 6, 2, {3});
        if (E.ok) CompareWithCharZ(E, std::min<uint32_t>(8, E.K_max));
    }
    {
        const mx::ModuleExport E = Export(8, 10, 3, {5});
        if (E.ok) CompareWithCharZ(E, std::min<uint32_t>(6, E.K_max));
    }
    {
        const mx::ModuleExport E = Export(11, 0, 5, {2, 3});
        if (E.ok) {
            CompareWithCharZ(E, 2);
            CompareWithLevelNK1(E);   // K = 1 against level_n's F_5 matrices
        }
    }
    // a word prime with ell^2 >= 2^31 (audit F1): K_run = K_max = 1 on the field F_ell; (11, 0) against level_n and char_Z, (1, 10)
    // against char_Z
    {
        const mx::ModuleExport E = Export(11, 0, 65521, {2, 3});
        if (E.ok) {
            check(E.route == "cokernel" && E.K_run == 1 && E.K_max == 1, "(c) (11,0) ell=65521: K_run = K_max = 1 (" + std::to_string(E.K_run) + ", " +
                                                                              std::to_string(E.K_max) + ")");
            CompareWithCharZ(E, 1);
            CompareWithLevelNK1(E);
        }
    }
    {
        const mx::ModuleExport E = Export(1, 10, 65521, {2, 3});
        if (E.ok) {
            check(E.K_max == 1, "(c) (1,10) ell=65521: K_max = 1");
            CompareWithCharZ(E, 1);
        }
    }
}

void ArmD() {
    const mx::ModuleExport E = Export(1, 30, 5, {2, 3});
    if (!E.ok) return;
    if (!check(E.sigma.size() == 1 && E.sigma[0] == 2, "(d) (1,30) tors_5 H^1 = Z/25")) return;
    std::vector<uint32_t> e0;
    const auto H0 = H0Hecke(30, 5, 2, E.ps, e0);
    if (!check(e0.size() == 1 && e0[0] == 2, "(d) H^0(V/25) = Z/25 (" + Join(e0) + ")")) return;
    for (size_t o = 0; o < E.ps.size(); ++o) {
        const uint64_t tors = E.at(2).ops[o][0];
        const uint64_t h0 = H0[o].at(0) % 25;
        check(tors == h0, "(d) T_" + std::to_string(E.ps[o]) + ": eigenvalue on H^0(V/25) " + std::to_string(h0) + " vs on tors_5 H^1 " + std::to_string(tors));
        std::cout << "  (d) T_" << E.ps[o] << ": H^0(V/25) " << h0 << ", tors_5 H^1 " << tors << std::endl;
    }
}

void ArmE() {
    const mx::ModuleExport E = Export(11, 0, 5, {2});
    if (!E.ok) return;
    const mx::ModuleAtK M = E.at(2);
    if (!check(M.free_rank == 3 && M.torsion_count == 0, "(e) (11,0): L = Z^3, no 5-torsion")) return;
    integral_descent::maps::Mat B = M.free_block(0);
    const integral_descent::maps::ZMod Z(5, 2);
    for (uint32_t i = 0; i < 3; ++i) B.at(i, i) = Z.add(B.at(i, i), 2);
    const integral_descent::maps::Smith S = integral_descent::maps::SmithForm(Z, B, false, false);
    // 02 Sec. 6 (T3 row, "after F1's export: one unit entry of T_2 + 2 mod 25") and Sec. 1 (T1, the same read at 691): the Smith
    // form of c(T_2) = T_2 + 2 on L/25 has the exponents [0] (rank 1: one eigenvalue 0 of multiplicity 2, c(3) = 5), i.e. c(T_2) L =
    // a Z_5 e with v_5(a) = 0, so v_5(exp eta_L) = v_5(c(lambda_Eis)) - v_5(a) = 1 - 0 = 1: eta_L = Z/5 (g_P = 5 is the upper bound).
    // (The 06 card's "(e) ... has exponent [1]" contradicts its own source; EXPORT_report.md records it.)
    std::vector<uint32_t> nz;
    for (uint32_t s : S.sigma) nz.push_back(s);
    check(nz == std::vector<uint32_t>{0}, "(e) Smith of T_2 + 2 on L/25: exponents " + Join(nz) + " (expected [0]: a unit entry, eta_L = Z/5)");
    bool unit = false;
    for (uint32_t x : B.a) unit = unit || x % 5 != 0;
    check(unit, "(e) T_2 + 2 mod 25 has a unit entry (the witness of 02 Sec. 6)");
    std::cout << "  (e) T3 (11,0): Smith(T_2 + 2 | L/25) exponents " << Join(nz) << " => v_5(exp eta_L) = 1 - " << (nz.empty() ? 0 : nz[0])
              << " = " << (nz.empty() ? 1 : 1 - static_cast<int>(nz[0])) << std::endl;
}

void ArmF() {
    const mx::ModuleExport E = Export(1, 30, 5, {2, 3});
    if (!E.ok) return;
    const mx::ModuleAtK M1 = E.at(1);
    check(M1.rank() == 6 && M1.free_rank == 5 && M1.rank() - M1.free_rank == 1,
          "(f) (1,30): dim H^1/5 = " + std::to_string(M1.rank()) + " = dim L/5 (" + std::to_string(M1.free_rank) + ") + dim tors_5/5 (1)");
    // dim A'_1 on L/5 against the console route's free-part image of A_3 at shift 2 (level 3 - 2 = 1), per system
    std::vector<uint64_t> ps64(E.ps.begin(), E.ps.end());
    const ra::OpModule C = ra::RunModule(30, 5, 3, ps64);
    // A'_1 = the image of A'_2 (on L/25) in End(L/5) = A'_2 / A'_2[5] = 5 A'_2: the reading at shift 1 (Localize needs K >= 2)
    const ra::OpModule L = FreeOpModule(E.at(2), 2);
    std::map<std::vector<uint64_t>, uint32_t> mine, theirs;
    uint32_t dim_mine = 0, dim_theirs = 0;
    for (const ra::JointSystem& s : ra::RationalSystems(L)) {
        const uint32_t d = ra::ReadAlgebra(ra::Localize(L, s.a).local, s.a, 1).length;
        mine[s.a] = d;
        dim_mine += d;
    }
    for (const ra::JointSystem& s : ra::RationalSystems(C)) {
        const uint32_t d = ra::ReadAlgebra(ra::Localize(C, s.a).local, s.a, 2).length;
        if (d) theirs[s.a] = d;
        dim_theirs += d;
    }
    check(mine == theirs && dim_mine == dim_theirs, "(f) dim A'_1(L/5) per system: export " + std::to_string(dim_mine) + " vs console (A_3, shift 2) " +
                                                         std::to_string(dim_theirs));
    std::cout << "  (f) (1,30) ell=5: H^1/5 " << Join(M1.invariant_factors()) << ", L/5 rank " << M1.free_rank << ", dim A'_1(L/5) = " << dim_mine
              << " (" << mine.size() << " systems)" << std::endl;
}

void ArmG() {
    // the mixed block at ell in {2, 3}: cells with torsion and a free part, ell | N included ((2,6) and (4,12) at 2, (9,8) at 3)
    struct CellG {
        uint32_t N, n, ell;
        std::vector<uint32_t> ps;
    };
    const std::vector<CellG> cells = {{8, 10, 2, {3, 5, 7}}, {8, 10, 3, {5, 7}}, {2, 6, 2, {3, 5, 7}}, {1, 10, 2, {2, 3, 5}},
                                      {1, 10, 3, {2, 5}},     {4, 12, 2, {3, 5}},  {9, 8, 3, {2, 5, 7}}};
    uint64_t mixed = 0;
    for (const CellG& c : cells) {
        const mx::ModuleExport E = Export(c.N, c.n, c.ell, c.ps);
        if (!E.ok) continue;
        const std::string tag = "(g) " + Cell(c.N, c.n, c.ell);
        uint64_t m = 0;
        for (uint32_t K : {1u, 2u, E.K_max})
            if (K <= E.K_max) m = CheckModuleAlgebra(E.at(K), tag + " K=" + std::to_string(K));
        mixed += m;
        std::cout << "  " << tag << ": " << E.sigma.size() << " torsion + " << E.free_rank << " free generators, K = 1, 2, " << E.K_max
                  << "; mixed-block non-zero entries at K_max " << m << std::endl;
    }
    check(mixed > 0, "(g) the mixed block is exercised (non-zero entries at K_max: " + std::to_string(mixed) + ")");
    // forced refusals (audit F4), reachable by ExportOptions: ModulePrecision at the cokernel route ((1,30) ell = 5, sigma = 2, K_run =
    // 3: d^0 certified, K_max = 1 < sigma_max) and at the tree ((8,10) ell = 2, sigma_max = 3, sigma1_max = 1, K_run = 5: K_max = 0);
    // RssCeiling (rss_cap_bytes = 1, before anything is built) at both routes
    {
        mx::ExportOptions o;
        o.K_run = 3;
        const mx::ModuleExport E = mx::ExportModule(1, 30, 5, {2}, o);
        check(!E.ok && E.code == mx::ExportRefusal::ModulePrecision && E.refusal.find("[ModulePrecision]") != std::string::npos,
              "(g) (1,30) ell=5 K_run=3 refuses [ModulePrecision]: " + E.refusal);
    }
    {
        mx::ExportOptions o;
        o.K_run = 5;
        const mx::ModuleExport E = mx::ExportModule(8, 10, 2, {3}, o);
        check(!E.ok && E.code == mx::ExportRefusal::ModulePrecision && E.status() == "blocked(export)",
              "(g) (8,10) ell=2 K_run=5 refuses [ModulePrecision]: " + E.refusal);
    }
    for (uint32_t ell : {2u, 691u}) {
        mx::ExportOptions o;
        o.rss_cap_bytes = 1;
        const mx::ModuleExport E = mx::ExportModule(8, 10, ell, {3}, o);
        check(!E.ok && E.code == mx::ExportRefusal::RssCeiling && E.refusal.find("[RssCeiling]") != std::string::npos,
              "(g) (8,10) ell=" + std::to_string(ell) + " rss_cap_bytes=1 refuses [RssCeiling]: " + E.refusal);
    }
}

}  // namespace

uint32_t HeckeZExportValidation() {
    std::cout << "HeckeZExportValidation: H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K with T_p (slot-25 EXPORT; basis-free invariants)" << std::endl;
    const std::vector<std::pair<const char*, std::function<void()>>> arms = {
        {"a", ArmA}, {"b", ArmB}, {"c", ArmC}, {"d", ArmD}, {"e", ArmE}, {"f", ArmF}, {"g", ArmG}};
    uint32_t failing = 0;
    bool budget = true;
    // The RSS budget (each arm <= 1 GiB growth) gates only a standalone run (audit F2): inside --all (or after other validations in
    // one process) freed heap stays resident and malloc reuses it without raising VmRSS, so the growth is no bound on the arm's
    // footprint.  Standalone = the process RSS at entry is below 256 MiB.  Seconds are printed, never gated (shared box).
    const uint64_t entry_rss = integral_descent::CurrentRssBytes();
    const bool standalone = entry_rss < (uint64_t(256) << 20);
    for (const auto& [name, fn] : arms) {
        g_fail = 0;
        const uint64_t c0 = g_checks;
        const auto t0 = clk::now();
        const uint64_t rss0 = integral_descent::CurrentRssBytes();   // the arm's budget is its growth over the process's RSS at entry
        integral_descent::RssSampler rs;
        try {
            fn();
        } catch (const std::exception& e) {
            ++g_fail;
            std::cout << "  [EXCEPTION] arm (" << name << "): " << e.what() << std::endl;
        }
        const uint64_t rss = rs.stop();
        const double s = since(t0);
        const uint64_t grow = rss > rss0 ? rss - rss0 : 0;
        const bool within = grow <= (uint64_t(1) << 30);
        budget = budget && within;
        if (g_fail) ++failing;
        std::cout << "ARM (" << name << ") checks=" << (g_checks - c0) << " failed=" << g_fail << " seconds=" << std::fixed << std::setprecision(2) << s
                  << " rss_peak_mib=" << (rss >> 20) << " rss_growth_mib=" << (grow >> 20) << " within_budget=" << (within ? "yes" : "no") << std::endl;
    }
    const bool identical = g_record_mismatch == 0;
    const bool ok = failing == 0 && identical && (budget || !standalone);
    std::cout << "HECKEZ_EXPORT arms=" << arms.size() << " failing=" << failing << " descent_records_identical=" << (identical ? "yes" : "no")
              << " verdict=" << (ok ? "OK" : "FAIL") << " budget="
              << (standalone ? std::string("gated(standalone)") : std::string("informational(in-process)")) << " entry_rss_mib=" << (entry_rss >> 20)
              << " (" << (budget ? "met" : "exceeded") << ": each arm <= 1 GiB RSS growth, gated only standalone; seconds informational)" << std::endl;
    return ok ? 0 : 1;
}
