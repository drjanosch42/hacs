// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// PAR-ASM: the threaded assembly stages of the integral descent against the serial code of record.  See the header.
#include "IntegralDescentParAsmValidation.h"

#include "DescentAssembly.h"
#include "DescentMaps.h"
#include "DescentPair.h"
#include "DescentParAsm.h"
#include "DescentRecord.h"
#include "DescentTreePair.h"
#include "../Model/TreeCochains.h"
#include "../Model/TreePair.h"
#include "../../Hecke/HeckeSettings.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace ID = integral_descent;
namespace M = integral_descent::maps;
namespace P = integral_descent::pair;
namespace TP = integral_descent::treepair;
namespace par = integral_descent::par;
using M::Mat;
using M::ZMod;
using clk = std::chrono::steady_clock;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}

const std::vector<uint32_t> kThreads = {1, 2, 7, 16};

// every lever off (the serial code of record) or on, scoped
struct Levers {
    par::ScopedLever a, e, m;
    explicit Levers(bool on) : a(par::Lever::Asm, on), e(par::Lever::Ell, on), m(par::Lever::Embed, on) {}
};

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t below(uint64_t n) { return static_cast<uint32_t>(next() % n); }
};

bool SameMat(const Mat& A, const Mat& B) { return A.r == B.r && A.c == B.c && A.a == B.a; }
bool SameVec(const std::vector<Mat>& A, const std::vector<Mat>& B) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i)
        if (!SameMat(A[i], B[i])) return false;
    return true;
}
bool SameZ(const ZMod& a, const ZMod& b) { return a.ell == b.ell && a.k == b.k; }
bool SamePres(const M::Pres& a, const M::Pres& b) {
    return SameZ(a.Z, b.Z) && a.c == b.c && a.tors == b.tors && a.free == b.free && SameMat(a.gensT, b.gensT) && SameMat(a.coord, b.coord) &&
           a.ok == b.ok && a.why == b.why;
}
bool SameMap(const M::MapRes& a, const M::MapRes& b) {
    return SameZ(a.Z, b.Z) && SameMat(a.F, b.F) && a.rankF == b.rankF && a.ker == b.ker && a.im == b.im && a.coker == b.coker &&
           SameMat(a.Lf, b.Lf) && a.rankL == b.rankL && SameZ(a.ZL, b.ZL) && a.ok == b.ok && a.why == b.why;
}
bool SameTypes(const std::vector<M::MapTypes>& a, const std::vector<M::MapTypes>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!(a[i].ker == b[i].ker && a[i].im == b[i].im && a[i].coker == b[i].coker)) return false;
    return true;
}
std::string cell(uint32_t N, uint32_t n) { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }

// ---- (X) the primitives ---------------------------------------------------------------------------------------------------------
void ArmPrimitives() {
    for (uint32_t w : {1u, 2u, 7u, 16u}) {
        std::vector<std::atomic<int>> hits(500);
        par::ForEachIndex(500, w, [&](size_t i) { hits[i].fetch_add(1); });
        bool once = true;
        for (auto& h : hits) once = once && h.load() == 1;
        check(once, "(X) ForEachIndex runs every index exactly once at " + std::to_string(w) + " workers");
        // the lowest throwing index is re-thrown; every index below it ran
        std::vector<std::atomic<int>> ran(64);
        std::string what;
        try {
            par::ForEachIndex(64, w, [&](size_t i) {
                ran[i].store(1);
                if (i == 37 || i == 11 || i == 50) throw std::runtime_error("index " + std::to_string(i));
            });
        } catch (const std::runtime_error& e) {
            what = e.what();
        }
        check(what == "index 11", "(X) the lowest throwing index is re-thrown at " + std::to_string(w) + " workers (got '" + what + "')");
        bool below = true;
        for (size_t i = 0; i <= 11; ++i) below = below && ran[i].load() == 1;
        check(below, "(X) every index below the lowest throwing one ran at " + std::to_string(w) + " workers");
        // rows: disjoint blocks covering [0, rows)
        std::vector<std::atomic<int>> rows(1000);
        {
            par::ScopedThreads st(w);
            par::ForRowBlocks(1000, 1u << 16, [&](uint32_t r0, uint32_t r1) {
                for (uint32_t r = r0; r < r1; ++r) rows[r].fetch_add(1);
            });
        }
        bool cover = true;
        for (auto& r : rows) cover = cover && r.load() == 1;
        check(cover, "(X) ForRowBlocks covers every row once at " + std::to_string(w) + " workers");
    }
    // nested regions run serially on the worker (no oversubscription), and InRegion() says so
    std::atomic<int> nested_parallel{0};
    par::ForEachIndex(4, 4, [&](size_t) {
        if (!par::InRegion()) nested_parallel.fetch_add(100);
        const auto me = std::this_thread::get_id();
        par::ForEachIndex(8, 8, [&](size_t) {
            if (std::this_thread::get_id() != me) nested_parallel.fetch_add(1);
        });
    });
    check(nested_parallel.load() == 0, "(X) a nested ForEachIndex runs on its worker thread only");
    // the mathematical settings (thread-local) reach every worker
    {
        const hecke::ScopedCoefficientRing z(0);
        std::atomic<int> bad{0};
        par::ForEachIndex(16, 8, [&](size_t) {
            if (!hecke::CoefficientRingIsZ()) bad.fetch_add(1);
        });
        check(bad.load() == 0, "(X) the caller's coefficient_ring = Z is in force on every worker");
    }
    // the joint RSS rule
    check(par::JointWorkers(10, 1ull << 30, 40ull << 30, 16) == 10, "(X) JointWorkers: 10 items of 1 GiB under 40 GiB run 10 at once");
    check(par::JointWorkers(40, 1ull << 30, 40ull << 30, 64) == 36, "(X) JointWorkers: at most 0.9 x the ceiling (36 x 1 GiB)");
    check(par::JointWorkers(40, 1ull << 30, 40ull << 30, 8) == 8, "(X) JointWorkers: at most the thread count");
    check(par::JointWorkers(40, 50ull << 30, 40ull << 30, 16) == 1, "(X) JointWorkers: an item above the ceiling runs alone (its own refusal rule applies)");
    check(par::JointWorkers(1, 1, 40ull << 30, 16) == 1, "(X) JointWorkers: one item");
}

// ---- (M) MatMulZ ----------------------------------------------------------------------------------------------------------------
void ArmMatMul() {
    Rng rng(20261027);
    const std::vector<ZMod> rings = {ZMod(2, 30), ZMod(3, 19), ZMod(5, 13), ZMod(691, 3)};
    struct Shape {
        uint32_t r, k, c;
        uint32_t density;   // percent non-zero in A
    };
    const std::vector<Shape> shapes = {{160, 200, 170, 100}, {300, 120, 260, 7}, {1, 4100, 1100, 50}, {257, 129, 513, 30}, {0, 50, 300, 100}};
    for (const ZMod& Z : rings)
        for (const Shape& s : shapes) {
            Mat A(s.r, s.k), B(s.k, s.c);
            for (auto& x : A.a) x = rng.below(100) < s.density ? static_cast<uint32_t>(rng.next() % Z.m) : 0u;
            for (auto& x : B.a) x = static_cast<uint32_t>(rng.next() % Z.m);
            Mat ref;
            {
                Levers off(false);
                ref = P::MatMulZ(Z, A, B);
            }
            for (uint32_t t : kThreads) {
                Levers on(true);
                par::ScopedThreads st(t);
                check(SameMat(P::MatMulZ(Z, A, B), ref), "(M) MatMulZ " + std::to_string(s.r) + "x" + std::to_string(s.k) + "x" + std::to_string(s.c) +
                                                             " mod " + std::to_string(Z.ell) + "^" + std::to_string(Z.k) + " at " + std::to_string(t) + " threads");
            }
        }
}

// ---- (B) the tree pair complex ----------------------------------------------------------------------------------------------------
void ArmTreeComplex() {
    struct C {
        uint32_t ell, N, n, T;
    };
    const std::vector<C> cells = {{2, 23, 2, 5}, {2, 47, 2, 5}, {2, 11, 4, 6}, {3, 23, 2, 5}, {3, 11, 4, 6}, {2, 283, 2, 5}};
    for (const C& c : cells) {
        const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(c.ell, ID::LocalWidth(c.ell));
        const auto TM = integral_model::TreeInducedSymModule(c.N, c.n, ctx);
        P::PairComplex ref;
        {
            Levers off(false);
            ref = TP::BuildTreePairComplex(c.ell, ctx, TM, c.T);
        }
        for (uint32_t t : kThreads) {
            Levers on(true);
            par::ScopedThreads st(t);
            const P::PairComplex X = TP::BuildTreePairComplex(c.ell, ctx, TM, c.T);
            const std::string w = "(B) ell=" + std::to_string(c.ell) + " " + cell(c.N, c.n) + " T=" + std::to_string(c.T) + " at " + std::to_string(t) + " threads";
            check(X.c == ref.c && X.cb == ref.cb && X.T == ref.T && X.route == ref.route, w + ": cochain ranks");
            check(SameVec(X.d, ref.d), w + ": every d^q");
            check(SameVec(X.db, ref.db), w + ": every d_B^q");
            check(SameVec(X.r, ref.r), w + ": every a^{*,q}");
        }
    }
    // a fault: the certificate throws the same message on both routes (the streamed check finds the same first degree)
    for (auto f : {integral_model::TreePairFault::CuspOneMinusT2}) {   // d_B o d_B != 0: detected by the builder's certificate
        const integral_model::ScopedTreePairFault sf(f);
        const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(2, ID::LocalWidth(2));
        const auto TM = integral_model::TreeInducedSymModule(23, 2, ctx);
        std::string w_off = "(none)", w_on = "(none)";
        try {
            Levers off(false);
            TP::BuildTreePairComplex(2, ctx, TM, 5);
        } catch (const std::exception& e) {
            w_off = e.what();
        }
        for (uint32_t t : {1u, 16u}) {
            try {
                Levers on(true);
                par::ScopedThreads st(t);
                TP::BuildTreePairComplex(2, ctx, TM, 5);
                w_on = "(none)";
            } catch (const std::exception& e) {
                w_on = e.what();
            }
            check(w_off != "(none)" && w_on == w_off, "(B) fault " + std::to_string(static_cast<uint32_t>(f)) + ": the same certificate failure at " +
                                                          std::to_string(t) + " threads ('" + w_on + "' vs '" + w_off + "')");
        }
    }
}

// ---- (E) InducedMapEmbed --------------------------------------------------------------------------------------------------------
void ArmEmbed() {
    struct C {
        uint32_t ell, N, n;
    };
    for (const C& c : std::vector<C>{{2, 23, 2}, {3, 23, 2}, {2, 11, 4}}) {
        const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(c.ell, ID::LocalWidth(c.ell));
        const P::PairComplex X = TP::BuildTreePairComplex(c.ell, ctx, integral_model::TreeInducedSymModule(c.N, c.n, ctx), 5);
        const P::PairBetti bt = P::PairBettiOf(c.N, c.n);
        P::PairLocal ref;
        P::PieceReport rr;
        bool okr;
        {
            par::ScopedLever e(par::Lever::Embed, false);
            okr = P::RunPairLes(X, bt, 4, ref, rr);
        }
        const auto e0 = par::GetCounters().embed_maps;
        P::PairLocal L;
        P::PieceReport r;
        bool ok;
        {
            par::ScopedLever e(par::Lever::Embed, true);
            ok = P::RunPairLes(X, bt, 4, L, r);
        }
        const std::string w = "(E) ell=" + std::to_string(c.ell) + " " + cell(c.N, c.n);
        check(okr && ok && rr.why == r.why, w + ": RunPairLes ok on both routes");
        check(par::GetCounters().embed_maps > e0, w + ": the structured maps ran");
        bool pres = L.H.size() == ref.H.size() && L.Hd.size() == ref.Hd.size() && L.Hc.size() == ref.Hc.size();
        for (size_t q = 0; pres && q < L.H.size(); ++q)
            pres = SamePres(L.H[q], ref.H[q]) && SamePres(L.Hd[q], ref.Hd[q]) && SamePres(L.Hc[q], ref.Hc[q]);
        check(pres, w + ": every presentation");
        bool maps = L.res.size() == ref.res.size() && L.j.size() == ref.j.size() && L.proj.size() == ref.proj.size();
        for (size_t q = 0; maps && q < L.res.size(); ++q) maps = SameMap(L.res[q], ref.res[q]) && SameMap(L.proj[q], ref.proj[q]);
        for (size_t q = 0; maps && q < L.j.size(); ++q) maps = SameMap(L.j[q], ref.j[q]);
        check(maps, w + ": every map (F, Lf, kernel, image, cokernel, precisions)");
        check(r.exact_spots == rr.exact_spots && r.checks == rr.checks && r.min_prec == rr.min_prec, w + ": counts and the least precision");
    }
}

// ---- (P) the pair assembly ------------------------------------------------------------------------------------------------------
void ComparePair(const P::PairResult& A, const P::PairResult& B, const std::string& w) {
    check(A.refused == B.refused && A.refusal == B.refusal, w + ": the verdict and refusal text");
    check(A.H == B.H && A.Hd == B.Hd && A.Hc == B.Hc, w + ": H, H(d), H_c");
    check(SameTypes(A.res, B.res) && SameTypes(A.j, B.j) && SameTypes(A.proj, B.proj), w + ": res, iota, pi types");
    check(A.primes == B.primes && A.exact_spots == B.exact_spots && A.checks == B.checks && A.closed_form == B.closed_form, w + ": primes, counts");
    bool pieces = A.pieces.size() == B.pieces.size();
    for (size_t i = 0; pieces && i < A.pieces.size(); ++i) {
        const P::PieceReport &a = A.pieces[i], &b = B.pieces[i];
        pieces = a.ell == b.ell && a.route == b.route && a.K == b.K && a.T == b.T && a.c == b.c && a.cb == b.cb && a.exact_spots == b.exact_spots &&
                 a.checks == b.checks && a.rank_checks == b.rank_checks && a.min_prec == b.min_prec && a.rank_prime == b.rank_prime && a.ok == b.ok &&
                 a.why == b.why && a.matrix_bytes == b.matrix_bytes;
    }
    check(pieces, w + ": every piece report (route, ranks, counts, verdict)");
}
void ArmPair() {
    struct C {
        uint32_t N, n;
    };
    for (const C& c : std::vector<C>{{1, 30}, {11, 4}, {23, 2}}) {
        P::PairResult ref;
        {
            Levers off(false);
            ref = P::AssembleDescentPair(c.N, c.n);
        }
        check(!ref.refused, "(P) " + cell(c.N, c.n) + " serial: not refused (" + ref.refusal + ")");
        check(ref.ell_workers == 1, "(P) " + cell(c.N, c.n) + " serial: one ell >= 5 piece at a time");
        for (uint32_t t : kThreads) {
            Levers on(true);
            par::ScopedThreads st(t);
            const P::PairResult R = P::AssembleDescentPair(c.N, c.n);
            const std::string w = "(P) " + cell(c.N, c.n) + " at " + std::to_string(t) + " threads";
            ComparePair(R, ref, w);
            uint32_t m5 = 0;
            for (uint32_t ell : R.primes) m5 += ell >= 5 ? 1u : 0u;
            check(R.ell_workers == std::min<uint32_t>(t, std::max<uint32_t>(1, m5)) || m5 <= 1, w + ": the ell >= 5 pieces ran " +
                                                                                                    std::to_string(R.ell_workers) + " at a time");
            if (R.ell_workers > 1)
                check(R.ell_rss_growth <= R.ell_rss_pred_joint, w + ": the joint growth " + std::to_string(R.ell_rss_growth >> 20) +
                                                                    " MiB within the joint prediction " + std::to_string(R.ell_rss_pred_joint >> 20) + " MiB (rule 19)");
        }
        // the joint rule acts: a ceiling of three pieces' predictions keeps at most two at a time (0.9 x 3 = 2.7)
        {
            Levers on(true);
            par::ScopedThreads st(16);
            P::PairOptions o;
            o.rss_cap_bytes = std::max<uint64_t>(ref.rss_pred * 10 / 9 + 1, 3 * P::PredictPairRss(3, ref.D, 5));
            const P::PairResult R = P::AssembleDescentPair(c.N, c.n, o);
            ComparePair(R, ref, "(P) " + cell(c.N, c.n) + " under a small ceiling");
            check(R.ell_workers <= std::max<uint64_t>(1, (o.rss_cap_bytes / 10 * 9) / R.ell_rss_pred_one),
                  "(P) " + cell(c.N, c.n) + ": the small ceiling bounds the concurrent pieces (" + std::to_string(R.ell_workers) + ")");
        }
    }
}

// ---- (L) the level-1 groups -----------------------------------------------------------------------------------------------------
void CompareDescent(const ID::DescentResult& A, const ID::DescentResult& B, const std::string& w) {
    check(A.refused == B.refused && A.refusal == B.refusal && A.H == B.H && A.rational_dims == B.rational_dims, w + ": H, the verdict");
    bool bad = A.bad.size() == B.bad.size() && A.dickson == B.dickson;
    for (size_t i = 0; bad && i < A.bad.size(); ++i) {
        const ID::BadPrimeRun &a = A.bad[i], &b = B.bad[i];
        bad = a.ell == b.ell && a.count == b.count && a.K == b.K && a.certified == b.certified && a.K_tried == b.K_tried && a.tors1 == b.tors1 &&
              a.free_rank1 == b.free_rank1 && a.torsion_only_in_H1 == b.torsion_only_in_H1;
    }
    check(bad, w + ": every ell >= 5 run (K, certificate, torsion, free rank), in order");
}
void ArmLevel1() {
    for (uint32_t n : {100u, 202u}) {
        ID::DescentResult ref;
        {
            Levers off(false);
            ref = ID::AssembleDescent(n);
        }
        check(!ref.refused && ref.ell_workers == 1, "(L) n=" + std::to_string(n) + " serial: certified, one prime at a time");
        for (uint32_t t : kThreads) {
            Levers on(true);
            par::ScopedThreads st(t);
            const ID::DescentResult R = ID::AssembleDescent(n);
            const std::string w = "(L) n=" + std::to_string(n) + " at " + std::to_string(t) + " threads";
            CompareDescent(R, ref, w);
            check(R.ell_workers == std::min<uint32_t>(t, static_cast<uint32_t>(R.dickson.size())), w + ": " + std::to_string(R.ell_workers) + " primes at a time");
            check(R.ell_rss_growth <= R.ell_rss_pred_joint, w + ": the joint growth " + std::to_string(R.ell_rss_growth >> 20) + " MiB within the joint prediction " +
                                                                std::to_string(R.ell_rss_pred_joint >> 20) + " MiB (rule 19)");
        }
        {   // the joint rule: a ceiling of 2.5 primes' predictions -> at most 2 at a time; below one prediction -> 1
            Levers on(true);
            par::ScopedThreads st(16);
            ID::DescentOptions o;
            o.rss_cap_bytes = ID::PredictTreeBredonRss(n + 1) * 25 / 9;   // 0.9 x cap = 2.5 x one prediction
            const ID::DescentResult R = ID::AssembleDescent(n, o);
            CompareDescent(R, ref, "(L) n=" + std::to_string(n) + " under a ceiling of 2.5 predictions");
            check(R.ell_workers == 2, "(L) n=" + std::to_string(n) + ": the ceiling of 2.5 predictions runs 2 primes at a time (" + std::to_string(R.ell_workers) + ")");
            o.rss_cap_bytes = ID::PredictTreeBredonRss(n + 1);
            const ID::DescentResult R1 = ID::AssembleDescent(n, o);
            CompareDescent(R1, ref, "(L) n=" + std::to_string(n) + " under a ceiling of one prediction");
            check(R1.ell_workers == 1, "(L) n=" + std::to_string(n) + ": a ceiling of one prediction runs the primes one at a time");
        }
    }
}

// ---- (R) the record -------------------------------------------------------------------------------------------------------------
void ArmRecord() {
    const hecke::ScopedCoefficientRing z(0);
    struct C {
        uint32_t N, n;
    };
    for (const C& c : std::vector<C>{{1, 30}, {11, 4}}) {
        uint64_t ref;
        std::string body;
        {
            Levers off(false);
            const auto R = ID::record::BuildDescentRecord(c.N, c.n);
            ref = R.digest();
            body = R.body();
            check(R.ok(), "(R) " + cell(c.N, c.n) + " serial record ok");
        }
        for (uint32_t t : {1u, 7u, 16u}) {
            Levers on(true);
            par::ScopedThreads st(t);
            const auto R = ID::record::BuildDescentRecord(c.N, c.n);
            check(R.digest() == ref && R.body() == body, "(R) " + cell(c.N, c.n) + " the record body and digest at " + std::to_string(t) + " threads");
        }
    }
}

}  // namespace

uint32_t IntegralDescentParAsmValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "=== IntegralDescentParAsmValidation (PAR-ASM: the threaded assembly stages == the serial code, 1/2/7/16 workers) ===" << std::endl;
    const char* arms[] = {"X", "M", "B", "E", "P", "L", "R"};
    void (*fns[])() = {ArmPrimitives, ArmMatMul, ArmTreeComplex, ArmEmbed, ArmPair, ArmLevel1, ArmRecord};
    for (size_t i = 0; i < 7; ++i) {
        const auto ta = clk::now();
        const uint64_t c0 = g_checks;
        const uint32_t f0 = g_fail;
        try {
            fns[i]();
        } catch (const std::exception& e) {
            check(false, std::string("(") + arms[i] + ") threw: " + e.what());
        }
        std::cout << "  (" << arms[i] << ") checks=" << (g_checks - c0) << " failing=" << (g_fail - f0) << " " << fx(since(ta)) << " s" << std::endl;
    }
    std::cout << par::EffectiveLine() << std::endl;
    std::cout << "IntegralDescentParAsmValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0)) << " s" << std::endl;
    return g_fail == 0 ? 0u : 1u;
}
