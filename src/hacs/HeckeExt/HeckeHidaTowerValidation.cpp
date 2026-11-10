// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 12, WP-5) HeckeHidaTowerValidation / HeckeHidaTowerSlowValidation
// -- see the header.
//
// Arms ([C] v1.12.0-pre.2 rmk:hidacheckable; slot-12 plan WP-5), on the grid
// ell in {5, 7, 11, 13}, even n in [2, n_max] (standard 120, slow 200), each cell
// computed ONCE over Z/ell^K, K = 2 + e_max (cor:ordprecision(2), the pre-pass of
// Hecke/HeckeOrdinaryPart), both levels m = 1, 2 read from it:
//  (Hi2) the deferred last sentence of WP-4: the ordinary part read on H_{e'}
//        (cor:ordprecision(1), torsion included) and on the free block at
//        K = e' + e_max (cor:ordprecision(2)-(4)) have the same rank and the SAME
//        characteristic polynomials of T_2, T_3 mod ell^{e'}, e' = 1, 2; U^N is
//        idempotent and commutes with T_2, T_3 on both, and P M is free of rank
//        ord_rank on both (A4).  The comparison has content ONLY at e_max > 0:
//        at e_max = 0, K = 2 and H_K is free, so the two readings are the same
//        matrices -- it is gated on the e_max > 0 cells alone and their count is
//        printed per ell (slot-12 WP-11, A1).
//  (Hi3) h^1_ord(n) (read over F_ell) == the committed table, and == the rank
//        at e' = 2; at ell in {5, 7} it is 1 = the Eisenstein count.
//  (Hi4) chi^ord_{n,p} mod ell^m constant on every class of Tw_m, m = 1, 2,
//        p = 2, 3 (a CONJ item: a failure is a finding and is reported, the
//        arm then fails loudly -- not repaired).
//  (Hi5) 1 + p^{n+1} is a root of chi^ord_{n,p} mod ell^{e'} at every cell; at
//        the primitive root p0 mod ell^2, cells of one Tw_1 class not congruent
//        mod (ell-1) ell have Eisenstein roots equal mod ell and different mod
//        ell^2 -- index arithmetic (rmk:ordinaryeis), ONE check per ell -- and,
//        read off the model (slot-12 WP-11, A2), at every such pair with
//        ord_rank = 1 the computed chi^ord_{A,p0} mod ell^2 does NOT vanish at
//        B's Eisenstein root (and vice versa); at rank > 1 that is reported, not
//        gated (and whether the whole polynomials differ is printed, not
//        predicted).
//  (Det) the concurrent runner's values equal a sequential main-thread
//        recomputation on four cells (one per ell, the smallest n with
//        e_max > 0), every field of both readings (slot-12 WP-11, A7).
//  (Hi6) (10, 2411) over Z/2411^K (K = 2 standard, 3 slow; m < K by reduction):
//        H free of rank 3 (e_max = 0), chi(U) = (X - tau)^2 (X - (1 + 2411^11))
//        with tau(2411) mod 2411^K from the q-expansion (the model confirms the
//        COMP^a residues 0, 150*2411, 9539365011), h^1_ord = 1 at every m,
//        chi^ord_{T_p} = X - (1 + p^11), U nilpotent on (1 - P) with
//        (U(1-P))^{2K} = 0 over Z/2411^K (Cayley-Hamilton on Delta's block).
//  (N)   negative controls: two cells congruent mod (ell-1)/2 but not mod ell-1
//        DO differ mod ell somewhere (the Hi4 comparison has teeth); two cells of
//        one Tw_1 class not in one Tw_2 class DO differ mod ell^2 somewhere; a
//        perturbed tau misses chi(U) at (10, 2411).
#include "HeckeHidaTowerValidation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "HeckeHidaTower.h"
#include "../Hecke/HeckeOrdinaryPart.h"

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

uint64_t PrimitiveRootModEll2(uint64_t ell) { return ell == 7 ? 3 : 2; }

uint64_t OrderMod(uint64_t p, uint64_t q) {
    uint64_t x = p % q, k = 1;
    while (x != 1) { x = x * p % q; ++k; }
    return k;
}

struct Config {
    const char* name;
    std::map<uint64_t, std::vector<uint32_t>> grid;   // ell -> even n
    uint32_t K2411;
};

std::vector<uint32_t> EvenUpTo(uint32_t a, uint32_t b) {
    std::vector<uint32_t> v;
    for (uint32_t n = a; n <= b; n += 2) v.push_back(n);
    return v;
}

// The grid's cells, computed concurrently (std::thread workers on a shared
// atomic cursor; every result lands in its own pre-sized slot, so everything
// checked or printed is in grid order whatever the scheduling).  The coset /
// tau constructions take hecke::tau_construction_mutex() inside RunModuleCell;
// each cell is a pure function of (n, ell), its ring scope thread-local.
struct Slot {
    uint32_t n = 0;
    uint64_t ell = 0;
    TowerCell cell;
    std::string error;
    double seconds = 0;
};

void ComputeAll(std::vector<Slot>& slots) {
    std::atomic<size_t> cursor(0);
    auto worker = [&]() {
        for (;;) {
            const size_t k = cursor.fetch_add(1);
            if (k >= slots.size()) return;
            const auto tc = std::chrono::steady_clock::now();
            try {
                slots[k].cell = ComputeTowerCell(slots[k].n, slots[k].ell, 2);
            } catch (const std::exception& e) {
                slots[k].error = e.what();
            }
            slots[k].seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count();
        }
    };
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t conc = std::max<size_t>(1, std::min<size_t>(std::min<size_t>(hw ? hw : 1, 12), slots.size()));
    std::vector<std::thread> threads;
    for (size_t t = 1; t < conc; ++t) threads.emplace_back(worker);
    worker();
    for (std::thread& t : threads) t.join();
}

void CheckCell(Checker& g, const TowerCell& t) {
    const std::string tag = Cell(t.n_sym, t.ell);
    bool free_img = true;
    for (uint32_t m = 1; m <= 2; ++m) {
        const Reading& a = t.module_read[m - 1];
        const Reading& b = t.free_read[m - 1];
        const std::string te = tag + " e'=" + std::to_string(m);
        free_img = free_img && a.image_free && b.image_free;
        g.check(a.idempotent && a.commutes && b.idempotent && b.commutes,
                "(Hi2) " + te + ": U^N idempotent, commuting with T_2, T_3, on H_{e'} and on the free block");
        // (A1) at e_max = 0 the two readings are the same matrices: compared only at e_max > 0.
        if (t.e_max > 0) {
            g.check(a.ord_rank == b.ord_rank, "(Hi2) " + te + ": ordinary rank on H_{e'} == on the free block (e_max > 0)");
            bool same = a.charpoly.size() == 2 && b.charpoly.size() == 2;
            for (size_t i = 0; same && i < 2; ++i) same = a.charpoly[i] == b.charpoly[i];
            g.check(same, "(Hi2) " + te + ": chi(T_2), chi(T_3) on the ordinary part: H_{e'} == free block mod ell^e' (e_max > 0)");
        }
        for (size_t i = 0; i < a.charpoly.size(); ++i) {
            const uint64_t p = a.primes[i], q = Pow64(t.ell, m);
            g.check(a.charpoly[i].size() == a.ord_rank + 1 && a.charpoly[i].back() == 1 % q,
                    te + ": chi(T_" + std::to_string(p) + ") monic of degree ord_rank");
            const uint64_t eis = (1 + PowMod(p, t.n_sym + 1, q)) % q;
            g.check(EvalMod(a.charpoly[i], eis, q) == 0,
                    "(Hi5) " + te + ": 1 + p^{n+1} is a root of chi^ord(T_" + std::to_string(p) + ") mod ell^e'");
        }
    }
    g.check(free_img, "(Hi2) " + tag + ": P M free of rank ord_rank (ord_torsion = 0) on H_{e'} and the free block, e' = 1, 2 (A4)");
    g.check(t.module_read[0].ord_rank == PredictedOrdinaryRank(t.n_sym, t.ell),
            "(Hi3) " + tag + ": h^1_ord = " + std::to_string(t.module_read[0].ord_rank) + ", table " +
                std::to_string(PredictedOrdinaryRank(t.n_sym, t.ell)));
    g.check(t.module_read[1].ord_rank == t.module_read[0].ord_rank, "(Hi3) " + tag + ": the rank at e' = 2 is the rank over F_ell");
    if (t.ell == 5 || t.ell == 7)
        g.check(t.module_read[0].ord_rank == 1, "(Hi3) " + tag + ": the ordinary part is the Eisenstein line (ell in {5, 7})");
}

// Every field of a tower cell, as text (the (Det) comparison).
std::string ReadingText(const Reading& r) {
    std::string s = "e=" + std::to_string(r.e_target) + " g=" + std::to_string(r.generators) + " rank=" + std::to_string(r.ord_rank) +
                    " idem=" + std::to_string(r.idempotent) + " comm=" + std::to_string(r.commutes) + " free=" + std::to_string(r.image_free);
    for (size_t i = 0; i < r.charpoly.size(); ++i) s += " T" + std::to_string(r.primes.at(i)) + "=" + PolyText(r.charpoly[i]);
    return s;
}
std::string CellText(const TowerCell& t) {
    std::string s = "n=" + std::to_string(t.n_sym) + " ell=" + std::to_string(t.ell) + " e_max=" + std::to_string(t.e_max) +
                    " K=" + std::to_string(t.K) + " h=" + std::to_string(t.h) + " exps=";
    for (uint32_t x : t.exps) s += std::to_string(x) + ",";
    for (const Reading& r : t.module_read) s += " | M " + ReadingText(r);
    for (const Reading& r : t.free_read) s += " | F " + ReadingText(r);
    return s;
}

// charpolys of reading m at a cell, as one comparable key.
std::vector<Poly> Key(const TowerCell& t, uint32_t m) { return t.module_read[m - 1].charpoly; }

void ArmTowers(Checker& g, const Config& cfg) {
    std::vector<Slot> slots;
    // heaviest first (cost grows with n), so the concurrent pass is balanced
    for (const auto& kv : cfg.grid)
        for (uint32_t n : kv.second) {
            Slot sl;
            sl.n = n;
            sl.ell = kv.first;
            slots.push_back(sl);
        }
    std::stable_sort(slots.begin(), slots.end(), [](const Slot& a, const Slot& b) { return a.n > b.n; });
    const auto tall = std::chrono::steady_clock::now();
    ComputeAll(slots);
    std::cout << "  " << slots.size() << " cells computed concurrently in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - tall).count() << " s\n";
    // (Det, slot-12 WP-11 A7) the concurrent values == a sequential main-thread recomputation.
    {
        const auto td = std::chrono::steady_clock::now();
        uint32_t redone = 0;
        for (const auto& kv : cfg.grid) {
            const Slot* pick = nullptr;
            for (const Slot& sl : slots)
                if (sl.ell == kv.first && sl.error.empty() && sl.cell.e_max > 0 && (!pick || sl.n < pick->n)) pick = &sl;
            if (!pick) continue;
            const TowerCell seq = ComputeTowerCell(pick->n, pick->ell, 2);
            ++redone;
            g.check(CellText(seq) == CellText(pick->cell),
                    "(Det) " + Cell(pick->n, pick->ell) + ": the concurrent runner's cell == the sequential recomputation (every field)");
        }
        std::cout << "  (Det) " << redone << " cells recomputed on the main thread, sequentially, in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count() << " s\n";
    }
    for (const auto& kv : cfg.grid) {
        const uint64_t ell = kv.first;
        std::map<uint32_t, TowerCell> cells;
        double cell_s = 0;
        uint32_t nmax = 0;
        for (uint32_t n : kv.second) {
            const Slot* sp = nullptr;
            for (const Slot& sl : slots) if (sl.ell == ell && sl.n == n) sp = &sl;
            const Slot& sl = *sp;
            cell_s += sl.seconds;
            nmax = std::max(nmax, n);
            if (!sl.error.empty()) {
                g.check(false, Cell(n, ell) + ": threw " + sl.error);
                continue;
            }
            cells[n] = sl.cell;
            CheckCell(g, sl.cell);
            std::cout << HidaTowerLine(sl.cell, 1) << "\n" << HidaTowerLine(sl.cell, 2) << "\n";
        }
        // (Hi4) along Tw_1 and Tw_2
        uint32_t others = 0, pairs1 = 0, pairs2 = 0, ctrl1 = 0, ctrl2 = 0, hi5pairs = 0, hi5polydiff = 0;
        uint32_t hi5model = 0, hi5higher = 0, hi5higher_nonzero = 0;
        bool hi5arith = true;
        const uint64_t P1 = ell - 1, P2 = (ell - 1) * ell;
        const uint64_t p0 = PrimitiveRootModEll2(ell);
        g.check(OrderMod(p0, ell * ell) == P2, "(Hi5) " + std::to_string(p0) + " is a primitive root mod " + std::to_string(ell) + "^2");
        for (const auto& A : cells) {
            for (const auto& B : cells) {
                if (B.first <= A.first) continue;
                const uint32_t n = A.first, n2 = B.first;
                const std::string pr = "ell=" + std::to_string(ell) + " n=" + std::to_string(n) + " <-> n'=" + std::to_string(n2);
                if ((n2 - n) % P1 == 0) {
                    ++pairs1;
                    g.check(A.second.module_read[0].ord_rank == B.second.module_read[0].ord_rank, "(Hi3) " + pr + ": equal ordinary rank on Tw_1");
                    g.check(Key(A.second, 1) == Key(B.second, 1), "(Hi4) m=1 " + pr + ": chi^ord(T_2), chi^ord(T_3) congruent mod ell");
                    if ((n2 - n) % P2 == 0) {
                        ++pairs2;
                        g.check(Key(A.second, 2) == Key(B.second, 2), "(Hi4) m=2 " + pr + ": chi^ord(T_2), chi^ord(T_3) congruent mod ell^2");
                    } else {
                        // (Hi5) at the primitive root: the Eisenstein roots agree mod ell, not mod ell^2
                        ++hi5pairs;
                        const uint64_t q2 = ell * ell;
                        const uint64_t ra = (1 + PowMod(p0, n + 1, q2)) % q2, rb = (1 + PowMod(p0, n2 + 1, q2)) % q2;
                        // index arithmetic (rmk:ordinaryeis): folded into ONE check per ell below (A2)
                        hi5arith = hi5arith && ra % ell == rb % ell && ra != rb;
                        const size_t ip = (p0 == 2) ? 0 : 1;
                        // (A2) the model's reading: A's chi^ord(T_p0) mod ell^2 misses B's Eisenstein root, and vice versa
                        const Poly fa = Key(A.second, 2)[ip];
                        const Poly fb = Key(B.second, 2)[ip];
                        const bool miss = EvalMod(fa, rb, q2) != 0 && EvalMod(fb, ra, q2) != 0;
                        if (A.second.module_read[1].ord_rank == 1 && B.second.module_read[1].ord_rank == 1) {
                            ++hi5model;
                            g.check(miss, "(Hi5) " + pr + ": chi^ord(T_" + std::to_string(p0) + ") of each cell does not vanish at the "
                                          "other's Eisenstein root mod ell^2 (model value, ord_rank = 1)");
                        } else {
                            ++hi5higher;
                            if (miss) ++hi5higher_nonzero;
                        }
                        if (Key(A.second, 2)[ip] != Key(B.second, 2)[ip]) ++hi5polydiff;
                        if (Key(A.second, 2) != Key(B.second, 2)) ++ctrl2;
                    }
                } else {
                    ++others;
                    if (Key(A.second, 1) != Key(B.second, 1)) ++ctrl1;
                }
            }
        }
        if (hi5pairs > 0)
            g.check(hi5arith, "(Hi5) ell=" + std::to_string(ell) + ": on all " + std::to_string(hi5pairs) +
                                  " pairs of one Tw_1 class, not one Tw_2 class, the Eisenstein roots 1 + p0^{n+1} agree mod ell "
                                  "and differ mod ell^2 (index arithmetic, rmk:ordinaryeis; one check)");
        if (others > 0)
            g.check(ctrl1 == others, "(N) ell=" + std::to_string(ell) + ": every pair in DIFFERENT Tw_1 classes differs mod ell (" +
                                         std::to_string(ctrl1) + " / " + std::to_string(others) + ")");
        if (hi5pairs > 0)
            g.check(ctrl2 > 0, "(N) ell=" + std::to_string(ell) + ": some pair of one Tw_1 class, not one Tw_2 class, differs mod ell^2");
        uint32_t emax_pos = 0;
        for (const auto& c : cells) if (c.second.e_max > 0) ++emax_pos;
        std::cout << "  ell=" << ell << ": " << cells.size() << " cells (n <= " << nmax << ", " << emax_pos
                  << " with e_max > 0, K = 2 + e_max; (Hi2)'s two-readings comparison gated on those " << emax_pos
                  << "); Tw_1 pairs " << pairs1 << ", Tw_2 pairs " << pairs2
                  << "; (Hi5) pairs " << hi5pairs << " (model check at ord_rank 1 on " << hi5model << "; rank > 1 reported: "
                  << hi5higher_nonzero << " / " << hi5higher << " miss the other's root; whole chi(T_" << p0 << ") differs mod ell^2 on " << hi5polydiff
                  << ", not predicted); controls: " << ctrl1 << "/" << others << " pairs across Tw_1 classes differ mod ell, "
                  << ctrl2 << " across Tw_2 classes differ mod ell^2; cell time " << cell_s << " s\n";
    }
}

void Arm2411(Checker& g, const Config& cfg) {
    const uint64_t ell = 2411;
    const uint32_t n = 10, K = cfg.K2411;
    const auto t0 = std::chrono::steady_clock::now();
    const std::string tag = "(Hi6) (10, 2411) K=" + std::to_string(K);
    const hecke::ordinary::TorsionPrepass pre = hecke::ordinary::TorsionExponentPrepass(n, ell);
    g.check(pre.e_max == 0, tag + ": e_max = 0 (prop:torsionprepass(2), n < ell + 1)");
    const ModuleCell mc = RunModuleCell(n, ell, K, {ell, 2, 3});
    bool free3 = mc.exps.size() == 3;
    for (uint32_t x : mc.exps) free3 = free3 && x == K;
    g.check(free3, tag + ": H_K free of rank h = 3 over Z/2411^K");
    const uint64_t tau_full[3] = {0ull, 361650ull, 9539365011ull};   // tau(2411) mod 2411^m, m = 1, 2, 3 (COMP^a)
    for (uint32_t m = 1; m <= K; ++m) {
        const uint64_t q = Pow64(ell, m);
        const std::string tm = "(Hi6) (10, 2411) m=" + std::to_string(m);
        // chi(U) on H_m (free) == (X - tau)^2 (X - (1 + ell^11))
        const RingMat Um = hecke::ordinary::Reduce(mc.at(ell), m);
        const Poly chiU = CharpolyFree(Um);
        auto expected = [&](uint64_t tau) {
            const uint64_t eis = (1 + PowMod(ell, 11, q)) % q;
            const uint64_t t2 = static_cast<uint64_t>(static_cast<unsigned __int128>(tau) * tau % q);
            // (X^2 - 2 tau X + tau^2)(X - eis)
            const uint64_t a2 = (q - (2 * tau) % q) % q;
            Poly f(4);
            f[3] = 1 % q;
            f[2] = (a2 + q - eis) % q;
            f[1] = (t2 + static_cast<uint64_t>(static_cast<unsigned __int128>(2 * tau % q) * eis % q)) % q;
            f[0] = (q - static_cast<uint64_t>(static_cast<unsigned __int128>(t2) * eis % q)) % q;
            return f;
        };
        const uint64_t tau = tau_full[m - 1] % q;
        g.check(chiU == expected(tau), tm + ": chi(U_2411) = (X - tau(2411))^2 (X - (1 + 2411^11)) mod 2411^m, tau(2411) = " +
                                           std::to_string(tau) + " mod 2411^m (the model confirms the q-expansion residue)");
        g.check(chiU != expected((tau + Pow64(ell, m - 1)) % q), "(N) " + tm + ": a perturbed tau misses chi(U)");
        const Reading rd = ReadOnModule(mc, m);
        g.check(rd.ord_rank == 1, tm + ": h^1_ord(10) = 1, the Eisenstein count");
        g.check(rd.idempotent && rd.commutes, tm + ": U^N idempotent, commuting with T_2, T_3");
        for (size_t i = 0; i < rd.charpoly.size(); ++i) {
            const uint64_t p = rd.primes[i];
            const Poly want = {(q - (1 + PowMod(p, 11, q)) % q) % q, 1 % q};
            g.check(rd.charpoly[i] == want, tm + ": chi^ord(T_" + std::to_string(p) + ") = X - (1 + p^11)");
        }
        // (A1) e_max = 0 here: H is free, the free-block reading is the H_m reading by construction -- not gated.
        std::cout << "HIDA_TOWER ell=2411 class=" << (n % ((ell - 1) * Pow64(ell, m - 1))) << " m=" << m << " n=" << n
                  << " ord_rank=" << rd.ord_rank << " charpoly_T2=" << PolyText(rd.charpoly.at(0))
                  << " charpoly_T3=" << PolyText(rd.charpoly.at(1)) << " charpoly_U=" << PolyText(chiU) << "\n";
    }
    // U nilpotent on (1 - P): (U(1-P))^{2K} = 0 over Z/2411^K
    const RingMat& U = mc.at(ell);
    const RingMat P = hecke::ordinary::OrdinaryProjector(U);
    const RingMat I = hecke::ordinary::Identity(ell, mc.exps);
    const RingMat UN = hecke::ordinary::Mul(U, hecke::ordinary::Sub(I, P));
    g.check(hecke::ordinary::IsZero(hecke::ordinary::PowU128(UN, 2 * K)), tag + ": (U(1-P))^{2K} = 0 (Delta's block non-ordinary, Cayley-Hamilton)");
    g.check(!hecke::ordinary::IsZero(hecke::ordinary::PowU128(UN, 1)), "(N) " + tag + ": U(1-P) itself is not zero");
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  (Hi6) (10, 2411) over Z/2411^m, m <= " << K << ": " << s << " s\n";
}

uint32_t Run(const Config& cfg) {
    std::cout << "=== " << cfg.name << " (slot-12 WP-5: Hida towers at ell in {5, 7, 11, 13}, and (10, 2411)) ===\n";
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    auto guarded = [&](const char* name, const std::function<void()>& f) {
        try { f(); } catch (const std::exception& e) { g.check(false, std::string(name) + ": threw " + e.what()); }
    };
    guarded("towers", [&] { ArmTowers(g, cfg); });
    guarded("(Hi6)", [&] { Arm2411(g, cfg); });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << cfg.name << ": " << g.checks << " checks, " << g.fails << " failing, " << s << " s\n";
    return g.fails;
}

} // namespace

// Standard: even n <= 80 at every ell (Tw_1 on every class, Tw_2 at ell = 5,
// period 20, and ell = 7, period 42), plus the plan's named Tw_2 partners:
// ell = 7 n = 94 (10 <-> 52 <-> 94), ell = 11 n = 112, 120 (period 110:
// 2 <-> 112, 10 <-> 120, Delta's family) and, since slot-12 WP-11 (A15),
// n = 114, 116 at 11 (4 <-> 114, 6 <-> 116).  NO Tw_2 pair at ell = 13 here:
// the first one needs n = 158 (period 156), and n = 158, 166 cost ~ 25 s each
// (measured: the standard wall 14.5 s -> 30.8 s); Tw_2 at 13 is the slow tier's
// (n <= 200: 2..44 <-> 158..200).
uint32_t HeckeHidaTowerValidation() {
    Config c{"HeckeHidaTowerValidation", {}, 2};
    for (uint64_t ell : {5ull, 7ull, 11ull, 13ull}) c.grid[ell] = EvenUpTo(2, 80);
    c.grid[7].push_back(94);
    for (uint32_t n : {112u, 114u, 116u, 120u}) c.grid[11].push_back(n);   // Tw_2 at 11: 2, 4, 6, 10 (slot-12 WP-11, A15)
    return Run(c);
}
// Slow: even n <= 120 at ell in {5, 7, 11} (the plan's standard grid) and even
// n <= 200 at ell = 13 (period 156: 2..44 <-> 158..200); (10, 2411) over Z/2411^3.
uint32_t HeckeHidaTowerSlowValidation() {
    Config c{"HeckeHidaTowerSlowValidation", {}, 3};
    for (uint64_t ell : {5ull, 7ull, 11ull}) c.grid[ell] = EvenUpTo(2, 120);
    c.grid[13] = EvenUpTo(2, 200);
    return Run(c);
}
