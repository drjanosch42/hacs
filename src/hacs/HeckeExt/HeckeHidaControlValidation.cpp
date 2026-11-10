// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-5) HeckeHidaControlValidation -- see the header.
//
// Arms ([C] v1.13.0-pre.2 sec:hidafull, rmk:hidacheckable2; slot-13 plan WP-5):
//  (P)   the torsion pre-pass beyond 2^64 (prop:torsionprepass(3) on the
//        engine::ladic digit route, Hecke/HeckeOrdinaryPart): at three cells with
//        e_max > 0 the native route, the digit route at the word-maximal layout
//        and the digit route at m' = 1 (L = e planes: the level induction really
//        runs) give the same lambda_e, e_max and torsion multiset; the digit
//        route's J_n equals the native J_n entrywise; at a level ell^E > 2^64 the
//        default (Auto) route -- digit route there -- reads lambda_E = lambda_{e_max}
//        (stationarity) and the same torsion multiset from the Smith rule; ell >=
//        2^31 beyond 2^64 is refused by name (NotImplementedInThisBuild).
//  (Hi7) chi^ord_{n,p} mod ell^3 constant on Tw_3(5; 10) = {10, 110, 210}
//        (p = 2, 3), read on H_3 (cor:ordprecision(1)) over Z/5^K, K = 3 + e_max,
//        with the free-block reading at e' = 3 compared where e_max > 0 ((Hi2) at
//        e' = 3), the Eisenstein root mod 5^3, and the control n = 30 (one Tw_2
//        class with 10, not one Tw_3 class): differs mod 5^3, agrees mod 5^2.
//  (Hi8) the lattice datum (def:hidalattice) constant on towers, m = 1, 2,
//        P = {2, 3}: on every Tw_m pair of the grid, by the DECISION PROCEDURE of
//        (Hi8) (HeckeHidaTower::DecideIsomorphic: isomorphic with a verified
//        certificate, or not isomorphic -- no third outcome); arm (a) on Sym^n Z^2,
//        arm (b) on Hida's lattice L_n (variant (H), FibreRepresentation::
//        HidaLatticeH on the module route); arm (c) whether D_* : M^H_m(n) ->
//        M_m(n) is an isomorphism, REPORTED per cell (not a prediction) -- gated
//        only where it is proved: D_* Hecke-equivariant on H^1, D_* maps the
//        ordinary part into the ordinary part, and D_* an isomorphism at the
//        Lucas cells n + 1 = c ell^s, 1 <= c <= ell - 1 (rmk:stabconvention(3)).
//        Lattice-independent (proved, rmk:stabconvention(3)): the ordinary rank
//        and every chi^ord agree on the two lattices -- gated.
//  (Hi9) (Hi4) at p in {5, 7}, p != ell, m = 1, 2, on the same pairs; and the
//        Eisenstein root 1 + p^{n+1} of chi^ord(T_p) mod ell^m (Hi5's argument).
//  (N)   controls: pairs of different Tw_1 classes are decided "not isomorphic";
//        a synthetic datum conjugated by an invertible X is decided isomorphic
//        (with a certificate); a Jordan block against a scalar with the same
//        charpoly is decided not isomorphic.
// A \CONJ item that fails is a finding: it is reported (the arm then fails
// loudly, as slot 12's towers), never repaired.
#include "HeckeHidaControlValidation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "HeckeHidaTower.h"
#include "../Hecke/HeckeOrdinaryPart.h"
#include "../Hecke/HeckeSettings.h"   // NotImplementedInThisBuild
#include "../Util/Rss.h"

namespace {

using namespace hecke_ext::hida;
using hecke::ordinary::RingMat;

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

uint64_t Pow64(uint64_t b, uint32_t e) {
    uint64_t a = 1;
    for (uint32_t i = 0; i < e; ++i) a *= b;
    return a;
}

double Secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Run jobs[k]() for every k on up to 12 std::thread workers (a shared cursor;
// each job writes only its own slot, so every check / print is in index order).
// (S13 WP-16) A job's exception is captured per slot and the FIRST in index
// order is rethrown on the calling thread after the join (as
// engine::tape::run_concurrently does) -- never std::terminate on a worker.
void RunConcurrently(size_t count, const std::function<void(size_t)>& job) {
    std::atomic<size_t> cursor(0);
    std::vector<std::exception_ptr> errors(count);
    auto worker = [&]() {
        for (;;) {
            const size_t k = cursor.fetch_add(1);
            if (k >= count) return;
            try { job(k); } catch (...) { errors[k] = std::current_exception(); }
        }
    };
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t conc = std::max<size_t>(1, std::min<size_t>(std::min<size_t>(hw ? hw : 1, 12), count));
    std::vector<std::thread> threads;
    for (size_t t = 1; t < conc; ++t) threads.emplace_back(worker);
    worker();
    for (std::thread& t : threads) t.join();
    for (size_t k = 0; k < count; ++k)
        if (errors[k]) std::rethrow_exception(errors[k]);
}

template<class F> bool ThrowsNotImplemented(F f) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild&) { return true; } catch (...) { return false; }
    return false;
}

// ---------------------------------------------------------------------------
// (P) the pre-pass beyond 2^64
// ---------------------------------------------------------------------------
void ArmPrepass(Checker& g) {
    using namespace hecke::ordinary;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<std::pair<uint32_t, uint64_t>> cells = {{30, 5}, {56, 7}, {60, 5}};
    PrepassOptions native, digit, digit1;
    native.route = PrepassRoute::Native;
    digit.route = PrepassRoute::Ladic;
    digit1.route = PrepassRoute::Ladic;
    digit1.digit_exp_override = 1;
    for (const auto& c : cells) {
        const uint32_t n = c.first;
        const uint64_t ell = c.second;
        const std::string tag = "(P) " + Cell(n, ell);
        const TorsionPrepass tn = TorsionExponentPrepass(n, ell, native);
        const TorsionPrepass td = TorsionExponentPrepass(n, ell, digit);
        const TorsionPrepass t1 = TorsionExponentPrepass(n, ell, digit1);
        const TorsionPrepass ta = TorsionExponentPrepass(n, ell);   // the default (Auto) entry point
        g.check(tn.e_max > 0, tag + ": e_max > 0 (the cell carries torsion)");
        g.check(tn.lambda == td.lambda && tn.lambda == t1.lambda && tn.lambda == ta.lambda,
                tag + ": lambda_e equal on the native route, the digit route (word-maximal m) and the digit route at m' = 1");
        g.check(tn.e_max == td.e_max && tn.e_max == t1.e_max && tn.torsion == td.torsion && tn.torsion == t1.torsion,
                tag + ": e_max and the torsion multiset equal on the three routes");
        // the digit route's J_n == the native J_n, entrywise, at E = max(2, e_max + 1)
        const uint32_t E = std::max<uint32_t>(2, tn.e_max + 1);
        const std::vector<uint64_t> Jn = PrepassMatrix(n, ell, E);
        const std::vector<std::vector<uint32_t>> Jd = PrepassMatrixDigits(n, ell, E);
        bool same = Jn.size() == Jd.size();
        for (size_t i = 0; same && i < Jn.size(); ++i) {
            uint64_t v = 0;
            for (uint32_t k = E; k-- > 0;) v = v * ell + Jd[i][k];
            same = v == Jn[i];
        }
        g.check(same, tag + ": J_n on the digit route (exact binomials, base-ell digits) == the native J_n mod ell^E");
        // beyond 2^64: the first E with ell^E > 2^64, one more for good measure
        uint32_t Ebig = 1;
        {
            unsigned __int128 p = ell;
            while (p <= (static_cast<unsigned __int128>(1) << 64)) { p *= ell; ++Ebig; }
            ++Ebig;
        }
        const uint32_t lam_big = PrepassKernelLength(n, ell, Ebig, PrepassOptions{});
        const uint32_t lam_big1 = PrepassKernelLength(n, ell, Ebig, digit1);
        g.check(lam_big == tn.lambda[tn.e_max] && lam_big1 == lam_big,
                tag + ": lambda_E = lambda_{e_max} at ell^E > 2^64 (E = " + std::to_string(Ebig) +
                    ", the default route = the digit route there; also at m' = 1)");
        const std::vector<uint32_t> tors_big = PrepassTorsionAtLevel(n, ell, Ebig, PrepassOptions{});
        g.check(tors_big == tn.torsion, tag + ": the Smith rule at ell^E > 2^64 reads the same torsion multiset");
        std::cout << "  (P) " << Cell(n, ell) << ": e_max=" << tn.e_max << " torsion=[";
        for (size_t i = 0; i < tn.torsion.size(); ++i) std::cout << (i ? "," : "") << tn.torsion[i];
        std::cout << "] lambda=[";
        for (size_t i = 0; i < tn.lambda.size(); ++i) std::cout << (i ? "," : "") << tn.lambda[i];
        std::cout << "] lambda_" << Ebig << "=" << lam_big << " (" << ell << "^" << Ebig << " > 2^64)\n";
    }
    // Refusals by name: ell >= 2^31 at a level beyond uint64_t (the digit layout's bound).
    g.check(ThrowsNotImplemented([] { (void)hecke::ordinary::PrepassKernelLength(2, 2147483659ull, 3, hecke::ordinary::PrepassOptions{}); }),
            "(P) ell = 2^31 + 11 at ell^3 > 2^64: refused by name (NotImplementedInThisBuild)");
    g.check(ThrowsNotImplemented([] { (void)hecke::ordinary::TorsionExponentPrepass(2, 4294967311ull); }),
            "(P) ell = 2^32 + 15 (ell^2 > 2^64): refused by name (NotImplementedInThisBuild)");
    std::cout << "  (P) " << cells.size() << " cells, " << Secs(t0) << " s\n";
}

// ---------------------------------------------------------------------------
// (Hi7) mod ell^3
// ---------------------------------------------------------------------------
struct Mod3Cell {
    uint32_t n = 0;
    uint64_t ell = 0;
    TowerCell t;           // module_read[0..2] (e' = 1, 2, 3), free_read[2] (e' = 3)
    std::string error;
    double seconds = 0;
};

Mod3Cell ComputeMod3(uint32_t n, uint64_t ell) {
    Mod3Cell c;
    c.n = n;
    c.ell = ell;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        hecke::ordinary::RequirePrecisionUse(n, ell, 3);
        c.t.n_sym = n;
        c.t.ell = ell;
        c.t.e_max = hecke::ordinary::TorsionExponentPrepass(n, ell).e_max;
        c.t.K = 3 + c.t.e_max;   // cor:ordprecision(2)
        c.t.h = hecke::ordinary::FreeRankH1(n);
        const ModuleCell mc = RunModuleCell(n, ell, c.t.K, {ell, 2, 3});
        c.t.exps = mc.exps;
        for (uint32_t m = 1; m <= 3; ++m) c.t.module_read.push_back(ReadOnModule(mc, m));
        c.t.free_read.push_back(ReadOnFreeBlock(mc, 3, c.t.h));
    } catch (const std::exception& e) {
        c.error = e.what();
    }
    c.seconds = Secs(t0);
    return c;
}

// Tw_3 cells `tower` (all one class) and the control `ctrl` (one Tw_2 class with them, not one Tw_3 class).
void CheckHi7(Checker& g, uint64_t ell, const std::vector<uint32_t>& tower, uint32_t ctrl, const std::vector<Mod3Cell>& cells,
              double wall) {
    const uint64_t q3 = Pow64(ell, 3);
    for (const Mod3Cell& c : cells) {
        const std::string tag = "(Hi7) " + Cell(c.n, ell);
        if (!c.error.empty()) { g.check(false, tag + ": threw " + c.error); continue; }
        const Reading& r3 = c.t.module_read[2];
        g.check(r3.idempotent && r3.commutes && r3.image_free, tag + ": U^N idempotent, commuting, P M free over Z/ell^3");
        g.check(r3.ord_rank == PredictedOrdinaryRank(c.n, ell), tag + ": h^1_ord = the (Hi3) table");
        for (size_t i = 0; i < r3.charpoly.size(); ++i) {
            const uint64_t p = r3.primes[i];
            g.check(EvalMod(r3.charpoly[i], (1 + PowMod(p, c.n + 1, q3)) % q3, q3) == 0,
                    tag + ": 1 + p^{n+1} is a root of chi^ord(T_" + std::to_string(p) + ") mod ell^3 (Hi5 at e' = 3)");
        }
        if (c.t.e_max > 0) {
            const Reading& f3 = c.t.free_read[0];
            g.check(f3.ord_rank == r3.ord_rank && f3.charpoly == r3.charpoly,
                    tag + ": the reading on H_3 == the free-block reading at K = 3 + e_max (Hi2 at e' = 3, e_max > 0)");
        }
        std::cout << HidaTowerLine(c.t, 3) << " e_max=" << c.t.e_max << " K=" << c.t.K << " (" << std::fixed
                  << std::setprecision(1) << c.seconds << " s)\n";
    }
    // (Hi7) the Tw_3 class
    for (size_t a = 1; a < tower.size(); ++a) {
        if (!cells[0].error.empty() || !cells[a].error.empty()) continue;
        g.check(cells[0].t.module_read[2].charpoly == cells[a].t.module_read[2].charpoly,
                "(Hi7) ell=" + std::to_string(ell) + " n=" + std::to_string(tower[0]) + " <-> n'=" + std::to_string(tower[a]) +
                    ": chi^ord(T_2), chi^ord(T_3) congruent mod ell^3 (Tw_3)");
    }
    // (N) the control: one Tw_2 class, not one Tw_3 class -- agrees mod ell^2, differs mod ell^3
    if (ctrl && cells.back().error.empty() && cells[0].error.empty()) {
        const Mod3Cell& c = cells.back();
        g.check(c.t.module_read[1].charpoly == cells[0].t.module_read[1].charpoly,
                "(N) (Hi7) control " + Cell(ctrl, ell) + ": congruent to n=" + std::to_string(tower[0]) + " mod ell^2 (one Tw_2 class)");
        g.check(c.t.module_read[2].charpoly != cells[0].t.module_read[2].charpoly,
                "(N) (Hi7) control " + Cell(ctrl, ell) + ": NOT congruent to n=" + std::to_string(tower[0]) + " mod ell^3 (not one Tw_3 class)");
    }
    double cs = 0;
    for (const Mod3Cell& c : cells) cs = std::max(cs, c.seconds);
    std::cout << "  (Hi7) ell=" << ell << ": Tw_3 cells " << tower.size() << (ctrl ? " + 1 control" : "") << ", wall "
              << std::fixed << std::setprecision(1) << wall << " s (slowest cell " << cs << " s)\n";
}

std::vector<uint32_t> Hi7Cells(const std::vector<uint32_t>& tower, uint32_t ctrl) {
    std::vector<uint32_t> ns = tower;
    if (ctrl) ns.push_back(ctrl);
    return ns;
}

// ---------------------------------------------------------------------------
// (Hi8) / (Hi9) and arm (c)
// ---------------------------------------------------------------------------
struct LatticeCell {
    uint32_t n = 0;
    uint64_t ell = 0;
    ModuleDatum sym[2], hida[2];      // m = 1, 2; primes {2, 3, 5, 7} \ {ell}
    DstarReading dstar[2];
    bool equivariant = false;
    std::string error;
    double seconds = 0;
};

std::vector<uint64_t> PrimesFor(uint64_t ell) {
    std::vector<uint64_t> v = {ell};
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull})
        if (p != ell) v.push_back(p);
    return v;
}

LatticeCell ComputeLatticeCell(uint32_t n, uint64_t ell) {
    LatticeCell c;
    c.n = n;
    c.ell = ell;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        // K = 2: the module datum at m <= 2 is read on H_m = H_K / ell^m (cor:ordprecision(1)),
        // which needs no extra precision on either lattice.
        const LatticePair lp = RunLatticePair(n, ell, 2, PrimesFor(ell));
        c.equivariant = lp.Dstar_equivariant;
        for (uint32_t m = 1; m <= 2; ++m) {
            c.sym[m - 1] = ReadModuleDatum(lp.sym, m);
            c.hida[m - 1] = ReadModuleDatum(lp.hida, m);
            c.dstar[m - 1] = ReadDstar(lp, m);
        }
    } catch (const std::exception& e) {
        c.error = e.what();
    }
    c.seconds = Secs(t0);
    return c;
}

// The (T_2, T_3) part of a datum: (Hi8)'s P = {2, 3}.
ModuleDatum Restrict23(const ModuleDatum& d) {
    ModuleDatum r = d;
    r.primes.clear();
    r.A.clear();
    r.charpoly.clear();
    for (size_t i = 0; i < d.primes.size(); ++i)
        if (d.primes[i] == 2 || d.primes[i] == 3) {
            r.primes.push_back(d.primes[i]);
            r.A.push_back(d.A[i]);
            r.charpoly.push_back(d.charpoly[i]);
        }
    return r;
}

const Poly* CharpolyOf(const ModuleDatum& d, uint64_t p) {
    for (size_t i = 0; i < d.primes.size(); ++i)
        if (d.primes[i] == p) return &d.charpoly[i];
    return nullptr;
}

// n + 1 = c ell^s, 1 <= c <= ell - 1: every binom(n, j) prime to ell (Lucas), L_n = V through D.
bool LucasCell(uint32_t n, uint64_t ell) {
    uint64_t x = n + 1;
    while (x % ell == 0) x /= ell;
    return x <= ell - 1;
}

std::string HexU64(uint64_t v) {
    std::ostringstream os;
    os << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return os.str();
}

std::string LatticeLine(const LatticeCell& c, uint32_t m, bool hida) {
    const ModuleDatum& d = hida ? c.hida[m - 1] : c.sym[m - 1];
    const uint64_t period = (c.ell - 1) * Pow64(c.ell, m - 1);
    std::ostringstream os;
    os << "HIDA_TOWER ell=" << c.ell << " class=" << (c.n % period) << " m=" << m << " n=" << c.n
       << " lattice=" << (hida ? "hida" : "sym") << " ord_rank=" << d.h;
    for (size_t i = 0; i < d.primes.size(); ++i) os << " charpoly_T" << d.primes[i] << "=" << PolyText(d.charpoly[i]);
    os << " module_digest=" << HexU64(ModuleClassDigest(Restrict23(d)));
    if (hida) os << " Dstar_iso=" << (c.dstar[m - 1].isomorphism ? 1 : 0) << " Dstar_rank_mod_ell=" << c.dstar[m - 1].rank_mod_ell;
    return os.str();
}

struct PairStats {
    uint32_t pairs = 0, iso_a = 0, iso_b = 0, hi9 = 0, hi9_pairs = 0, grid_decided = 0;
};

void ArmHi8Hi9(Checker& g, const std::map<uint64_t, std::vector<uint32_t>>& grid, const std::vector<LatticeCell>& cells) {
    std::cout << "  (Hi8)/(Hi9) " << cells.size() << " cells (two lattices each)\n";
    uint32_t dstar_iso[2] = {0, 0}, dstar_cells = 0, lucas = 0;
    for (const auto& kv : grid) {
        const uint64_t ell = kv.first;
        std::map<uint32_t, const LatticeCell*> byn;
        double cell_s = 0;
        for (const LatticeCell& c : cells) {
            if (c.ell != ell) continue;
            cell_s = std::max(cell_s, c.seconds);
            const std::string tag = Cell(c.n, ell);
            if (!c.error.empty()) { g.check(false, tag + ": threw " + c.error); continue; }
            byn[c.n] = &c;
            ++dstar_cells;
            const bool luc = LucasCell(c.n, ell);
            if (luc) ++lucas;
            g.check(c.equivariant, "(Hi8)(c) " + tag + ": D_* is Hecke-equivariant on H^1 (D A^H_p = A_p D, every p run)");
            for (uint32_t m = 1; m <= 2; ++m) {
                const ModuleDatum& s = c.sym[m - 1];
                const ModuleDatum& h = c.hida[m - 1];
                const std::string tm = tag + " m=" + std::to_string(m);
                g.check(s.h == PredictedOrdinaryRank(c.n, ell) && h.h == s.h,
                        "(Hi8) " + tm + ": the ordinary rank on both lattices = the (Hi3) table (lattice-independent)");
                g.check(s.charpoly == h.charpoly,
                        "(Hi8) " + tm + ": every chi^ord(T_p) equal on Sym^n and on L_n (lattice-independent, rmk:stabconvention(3))");
                const uint64_t q = Pow64(ell, m);
                for (size_t i = 0; i < s.primes.size(); ++i) {
                    const uint64_t p = s.primes[i];
                    g.check(EvalMod(s.charpoly[i], (1 + PowMod(p, c.n + 1, q)) % q, q) == 0,
                            "(Hi9) " + tm + ": 1 + p^{n+1} is a root of chi^ord(T_" + std::to_string(p) + ") mod ell^m");
                }
                const DstarReading& ds = c.dstar[m - 1];
                g.check(ds.image_in_ordinary, "(Hi8)(c) " + tm + ": D_* maps the ordinary part of L_n into that of Sym^n");
                if (luc)
                    g.check(ds.isomorphism, "(Hi8)(c) " + tm + ": D_* is an isomorphism at a Lucas cell (n + 1 = c ell^s; D invertible)");
                if (ds.isomorphism) ++dstar_iso[m - 1];
                std::cout << LatticeLine(c, m, false) << "\n" << LatticeLine(c, m, true) << "\n";
            }
        }
        // pairs
        PairStats st[2];
        uint32_t across = 0, across_notiso = 0;
        for (const auto& A : byn)
            for (const auto& B : byn) {
                if (B.first <= A.first) continue;
                const uint32_t n = A.first, n2 = B.first;
                const std::string pr = "ell=" + std::to_string(ell) + " n=" + std::to_string(n) + " <-> n'=" + std::to_string(n2);
                if ((n2 - n) % (ell - 1) != 0) {
                    // (N) different Tw_1 classes: decided not isomorphic (at m = 1)
                    ++across;
                    const IsoDecision d = DecideIsomorphic(Restrict23(A.second->sym[0]), Restrict23(B.second->sym[0]));
                    if (!d.isomorphic) ++across_notiso;
                    continue;
                }
                for (uint32_t m = 1; m <= 2; ++m) {
                    if ((n2 - n) % ((ell - 1) * Pow64(ell, m - 1)) != 0) continue;
                    PairStats& S = st[m - 1];
                    ++S.pairs;
                    const IsoDecision da = DecideIsomorphic(Restrict23(A.second->sym[m - 1]), Restrict23(B.second->sym[m - 1]));
                    const IsoDecision db = DecideIsomorphic(Restrict23(A.second->hida[m - 1]), Restrict23(B.second->hida[m - 1]));
                    if (da.isomorphic) ++S.iso_a;
                    if (db.isomorphic) ++S.iso_b;
                    if (!da.decided_by_precheck && da.dim_hom_bar > 0) ++S.grid_decided;
                    g.check(da.isomorphic, "(Hi8)(a) m=" + std::to_string(m) + " " + pr + ": (M_m(n), T_2, T_3) ~ (M_m(n'), T_2, T_3) on Sym^n -- " + da.text());
                    g.check(db.isomorphic, "(Hi8)(b) m=" + std::to_string(m) + " " + pr + ": (M^H_m(n), T_2, T_3) ~ (M^H_m(n'), T_2, T_3) on L_n -- " + db.text());
                    if (A.second->sym[0].h > 1)
                        std::cout << "  (Hi8) m=" << m << " " << pr << " h=" << A.second->sym[0].h << ": (a) " << da.text()
                                  << "; (b) " << db.text() << "\n";
                    for (uint64_t p : {5ull, 7ull}) {
                        if (p == ell) continue;
                        ++S.hi9_pairs;
                        const Poly* fa = CharpolyOf(A.second->sym[m - 1], p);
                        const Poly* fb = CharpolyOf(B.second->sym[m - 1], p);
                        const bool ok = fa && fb && *fa == *fb;
                        if (ok) ++S.hi9;
                        g.check(ok, "(Hi9) m=" + std::to_string(m) + " " + pr + ": chi^ord(T_" + std::to_string(p) + ") congruent mod ell^m");
                    }
                }
            }
        if (across > 0)
            g.check(across_notiso == across, "(N) ell=" + std::to_string(ell) + ": every pair across Tw_1 classes is decided not isomorphic (" +
                                                 std::to_string(across_notiso) + " / " + std::to_string(across) + ")");
        std::cout << "  ell=" << ell << ": " << byn.size() << " cells; (Hi8) Tw_1 pairs " << st[0].pairs << " (iso (a) " << st[0].iso_a
                  << ", (b) " << st[0].iso_b << "), Tw_2 pairs " << st[1].pairs << " (iso (a) " << st[1].iso_a << ", (b) "
                  << st[1].iso_b << "); decided on the grid (not by a pre-check) " << (st[0].grid_decided + st[1].grid_decided)
                  << "; (Hi9) " << (st[0].hi9 + st[1].hi9) << " / " << (st[0].hi9_pairs + st[1].hi9_pairs)
                  << " (pair, p) congruent; across Tw_1 classes " << across_notiso << " / " << across
                  << " not isomorphic; slowest cell " << std::fixed << std::setprecision(1) << cell_s << " s\n";
    }
    std::cout << "  (Hi8)(c) D_* : M^H_m(n) -> M_m(n) an isomorphism on " << dstar_iso[0] << " / " << dstar_cells << " cells at m = 1, "
              << dstar_iso[1] << " / " << dstar_cells << " at m = 2 (reported, not a prediction; gated only at the " << lucas
              << " Lucas cells)\n";
}

// (N) synthetic controls of the decision procedure.
void ArmDecisionControls(Checker& g) {
    const uint64_t ell = 11;
    const uint32_t m = 2;
    const uint64_t q = Pow64(ell, m);
    auto datum = [&](const Mat& A2, const Mat& A3) {
        ModuleDatum d;
        d.ell = ell;
        d.m = m;
        d.h = static_cast<uint32_t>(A2.size());
        d.primes = {2, 3};
        d.A = {A2, A3};
        return d;
    };
    auto finish = [&](ModuleDatum d) {
        d.charpoly.clear();
        for (const Mat& A : d.A) d.charpoly.push_back(CharpolyFree([&] {
            RingMat R;
            R.ell = ell;
            R.row_exp.assign(A.size(), m);
            for (const auto& row : A) for (uint64_t x : row) R.a.push_back(x % q);
            return R;
        }()));
        return d;
    };
    auto mul = [&](const Mat& A, const Mat& B) {
        Mat C(A.size(), std::vector<uint64_t>(A.size(), 0));
        for (size_t i = 0; i < A.size(); ++i)
            for (size_t j = 0; j < A.size(); ++j) {
                unsigned __int128 s = 0;
                for (size_t k = 0; k < A.size(); ++k) s += static_cast<unsigned __int128>(A[i][k]) * B[k][j];
                C[i][j] = static_cast<uint64_t>(s % q);
            }
        return C;
    };
    const Mat A2 = {{3, 1}, {0, 3}}, A3 = {{5, 11}, {0, 5}};
    const Mat X = {{1, 2}, {3, 7}}, Xi = {{7, q - 2}, {q - 3, 1}};   // det X = 1
    const ModuleDatum a = finish(datum(A2, A3));
    const ModuleDatum b = finish(datum(mul(mul(X, A2), Xi), mul(mul(X, A3), Xi)));
    const IsoDecision d1 = DecideIsomorphic(a, b);
    g.check(d1.isomorphic && !d1.certificate.empty(), "(N) a datum and its X-conjugate are decided isomorphic, with a certificate -- " + d1.text());
    const ModuleDatum c = finish(datum({{3, 0}, {0, 3}}, {{5, 0}, {0, 5}}));
    const IsoDecision d2 = DecideIsomorphic(a, c);
    g.check(!d2.isomorphic, "(N) a Jordan block against the scalar with the same charpolys is decided not isomorphic -- " + d2.text());
    // A pair of equal charpolys and equal Hom lengths the grid must decide:
    // T_2 = [[3, ell], [0, 3]] against T_2 = [[3, 0], [0, 3]] + ell E_21 (both = 3 mod ell).
    const ModuleDatum e1 = finish(datum({{3, ell}, {0, 3}}, {{5, 0}, {0, 5}}));
    const ModuleDatum e2 = finish(datum({{3, 0}, {ell, 3}}, {{5, 0}, {0, 5}}));
    const IsoDecision d3 = DecideIsomorphic(e1, e2);
    g.check(d3.isomorphic, "(N) [[3, ell], [0, 3]] ~ [[3, 0], [ell, 3]] over Z/ell^2 (the swap), decided on the grid -- " + d3.text());
    std::cout << "  (N) decision controls: " << d1.text() << " | " << d2.text() << " | " << d3.text() << "\n";
}

uint32_t Run(const char* name, const std::function<void(Checker&)>& body) {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    body(g);
    std::cout << name << ": " << g.checks << " checks, " << g.fails << " failing, " << std::fixed << std::setprecision(1)
              << Secs(t0) << " s, peak RSS (VmHWM) " << oc_util::PeakRssMiB() << " MiB\n";
    return g.fails;
}

} // namespace

// Standard grid ((Hi8) / (Hi9)): Tw_1 at every ell on small n, Tw_2 at 5 (period
// 20), 7 (period 42: 2 <-> 44, 10 <-> 52) and 11 (period 110: 10 <-> 120, Delta's
// family, rank 3); the rank-3 Tw_1 families at 11 (10, 20, 30) and 13 (10, 22, 34).
uint32_t HeckeHidaControlValidation() {
    std::cout << "=== HeckeHidaControlValidation (slot-13 WP-5: [C] sec:hidafull (Hi7)-(Hi9), def:hidalattice (H), "
                 "the pre-pass beyond 2^64) ===\n";
    return Run("HeckeHidaControlValidation", [](Checker& g) {
        auto guarded = [&](const char* nm, const std::function<void()>& f) {
            try { f(); } catch (const std::exception& e) { g.check(false, std::string(nm) + ": threw " + e.what()); }
        };
        guarded("(P)", [&] { ArmPrepass(g); });
        guarded("(N) decision", [&] { ArmDecisionControls(g); });
        std::map<uint64_t, std::vector<uint32_t>> grid;
        for (uint32_t n = 2; n <= 42; n += 2) grid[5].push_back(n);
        for (uint32_t n = 2; n <= 20; n += 2) grid[7].push_back(n);
        for (uint32_t n : {44u, 52u}) grid[7].push_back(n);
        grid[11] = {2, 4, 10, 12, 20, 30, 120};
        grid[13] = {2, 10, 14, 22, 34};
        // ONE concurrent pool over the (Hi7) cells and the lattice cells, heaviest
        // first (the (210, 5) mod-ell^3 cell is the long pole); every result lands
        // in its own slot, so every check and line below is in a fixed order.
        const std::vector<uint32_t> tower7 = {10, 110, 210};
        const uint32_t ctrl7 = 30;
        const std::vector<uint32_t> ns7 = Hi7Cells(tower7, ctrl7);
        struct Job { bool mod3; uint32_t n; uint64_t ell; size_t slot; };
        std::vector<Job> jobs;
        for (size_t k = 0; k < ns7.size(); ++k) jobs.push_back({true, ns7[k], 5, k});
        std::vector<std::pair<uint32_t, uint64_t>> lat;
        for (const auto& kv : grid)
            for (uint32_t n : kv.second) lat.push_back({n, kv.first});
        for (size_t k = 0; k < lat.size(); ++k) jobs.push_back({false, lat[k].first, lat[k].second, k});
        // cost ~ n^3; a mod-ell^3 cell at K = 3 + e_max over one lattice ~ a lattice cell at K = 2 over two
        std::stable_sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.n > b.n; });
        std::vector<Mod3Cell> cells7(ns7.size());
        std::vector<LatticeCell> cellsL(lat.size());
        const auto tp = std::chrono::steady_clock::now();
        RunConcurrently(jobs.size(), [&](size_t k) {
            const Job& j = jobs[k];
            if (j.mod3) cells7[j.slot] = ComputeMod3(j.n, j.ell);
            else cellsL[j.slot] = ComputeLatticeCell(j.n, j.ell);
        });
        const double wall = Secs(tp);
        std::cout << "  " << jobs.size() << " cells ((Hi7) " << ns7.size() << ", lattice " << lat.size()
                  << ") computed concurrently in " << std::fixed << std::setprecision(1) << wall << " s\n";
        guarded("(Hi7)", [&] { CheckHi7(g, 5, tower7, ctrl7, cells7, wall); });
        guarded("(Hi8)/(Hi9)", [&] { ArmHi8Hi9(g, grid, cellsL); });
    });
}
