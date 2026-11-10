// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, LN23) HeckeLevelNCurveValidation -- see the header and
// docs/implementation/slot22/LN23_spec.md (v0.2, section 10: the predictions this validation gates; v0.3, section 11: the
// LN23-F family certificate, the generation arm (G) and the Sturm arm).
#include "HeckeLevelNCurveValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"
#include "HeckeLevelNR3.h"
#include "HeckeLevelNSparse.h"
#include "../Util/Rss.h"
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

constexpr uint32_t kBigP = 134217757;
// The Hecke primes of the readings: every prime q <= g_qmax, q != N.  Default 13 = the prototype's {2, 3, 5, 7, 11, 13}; the
// read-only arm HECKE_LN23_QMAX widens it (a larger family can only shrink W, and on W_m only lengthen the radical series).
// The readings are those of the family: W_F contains W_m, and on W_m its radical series is at most e + 1 long (the F2 caveat
// of [C] (LN4) / (LN6), rmk:levelNcurvefamily).
uint32_t g_qmax = 13;

// (LN23-F) The family certificate.  The stabilisation re-read widens the family q <= QMAX along this ladder (the rungs above
// QMAX); a family is CERTIFIED when it contains a Mazur good prime (generation, Mazur II (16.6) / (17.1)), its reading
// (dim W and the layers) is unchanged under the next two widenings, and the layers have the curvilinear signature
// d, d - 2, ..., 3, 1 (separation, in practice: a heuristic, not a proof; the proof device is the Sturm family of the STURM
// arm).  The signature is that of W_m = T_m/p + T^0_m/p with T_m/p = F_p[x]/x^{e+1} acting faithfully ((LN6)'s CONJ); it was
// added (spec v0.3.1, section 11.3) after stabilisation alone certified (24473, 2) at q <= 13: 43 / 43,37,28,20,12,4 at 13,
// 19 and 31, separated only above 31.  dim W_F = 1 certifies itself (W_F contains W_m, which contains the Eisenstein line):
// TRIVIAL.  A cell with no certificate below 997 is NONE.
constexpr uint32_t kLadder[] = {13, 19, 31, 47, 71, 101, 151, 211, 307, 457, 691, 997};

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

std::string Cell(uint64_t N, uint64_t p) { return "(N=" + std::to_string(N) + ", p=" + std::to_string(p) + ")"; }

std::string Join(const std::vector<uint32_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}



// 2g + c - 1 = rank of H_1(X_0(N), cusps; Z) (spec L1), from the genus formula.
uint32_t CurveRank(uint64_t N) {
    return static_cast<uint32_t>(2 * DimCuspForms(N, 2) + static_cast<int64_t>(CuspCount(N)) - 1);
}

// h(D), D < 0 a discriminant: the reduced primitive forms (a, b, c), b^2 - 4ac = D, |b| <= a <= c, b >= 0 if |b| = a or
// a = c.  Independent of everything else in the tree (the oracle side of CE Thm 1.1).
uint64_t ClassNumberNeg(int64_t D) {
    ENGINE_CHECK_MSG(D < 0 && (D % 4 == 0 || D % 4 == -3), "ClassNumberNeg: a negative discriminant");
    uint64_t h = 0;
    for (int64_t a = 1; 3 * a * a <= -D; ++a)
        for (int64_t b = -a + 1; b <= a; ++b) {
            const int64_t num = b * b - D;
            if (num % (4 * a) != 0) continue;
            const int64_t c = num / (4 * a);
            if (c < a || (c == a && b < 0)) continue;
            if (gcd<uint64_t>(gcd<uint64_t>(static_cast<uint64_t>(a), static_cast<uint64_t>(b < 0 ? -b : b)), static_cast<uint64_t>(c)) == 1) ++h;
        }
    return h;
}

// CE Thm 1.3(i)'s test: ((N - 1)/3)! is a cube mod N (N = 1 mod 3 prime).
bool FactorialIsCube(uint64_t N) {
    uint64_t f = 1;
    for (uint64_t k = 2; k <= (N - 1) / 3; ++k) f = f * k % N;
    return powmod<uint64_t>(f, (N - 1) / 3, N) == 1;
}

std::vector<uint64_t> PrimesUpTo(uint64_t n) {
    std::vector<uint64_t> out;
    for (uint64_t k = 2; k <= n; ++k)
        if (hecke::IsPrimeU64(k)) out.push_back(k);
    return out;
}

uint64_t EnvNmax(const char* name, uint64_t dflt, uint64_t cap, bool& set) {
    set = false;
    const char* e = std::getenv(name);
    if (e == nullptr) return dflt;
    const std::string t(e);
    ENGINE_CHECK_MSG(!t.empty() && t.size() <= 9 && t.find_first_not_of("0123456789") == std::string::npos,
                     (std::string(name) + " must be a positive integer").c_str());
    const uint64_t v = std::stoull(t);
    ENGINE_CHECK_MSG(v >= 2 && v <= cap, (std::string(name) + " must lie in [2, " + std::to_string(cap) + "]").c_str());
    set = true;
    return v;
}

template <class F>
bool RefusedWith(F f, const std::string& needle, std::string* what = nullptr) {
    try {
        f();
    } catch (const std::logic_error& e) {
        if (what) *what = e.what();
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// ---- (P) the n = 0 generators are the orbit sums ----------------------------------------------------------
void ArmOrbitSums(Checker& g) {
    for (uint64_t N : {11ull, 37ull, 41ull, 61ull, 97ull, 210ull, 2143ull})
        for (uint32_t p : {2u, 3u, 5u, 7u, kBigP}) {
            const Gamma0LevelN lvl(N);
            const engine::ModCtx<uint32_t> ctx(p);
            for (const auto& go : {std::make_pair(Mat22::S(), 2u), std::make_pair(Mat22::ST(), 3u)}) {
                const std::vector<SparseVec> gens = OrbitWiseFixedGenerators(lvl, go.first, go.second, 0, ctx);
                uint64_t covered = 0;
                bool ones = true;
                for (const SparseVec& v : gens) {
                    covered += v.nnz();
                    for (uint32_t x : v.val) ones = ones && x == 1;
                }
                g.check(ones && covered == lvl.m(), "(P) " + Cell(N, p) + " order " + std::to_string(go.second) +
                                                        ": the n = 0 generators are orbit indicators covering P^1 once");
                const FixedSpace F = OrbitWiseFixedSpace(lvl, go.first, go.second, 0, ctx);
                bool dense_ones = F.span.size() == gens.size();
                for (const auto& f : F.span)
                    for (uint32_t x : f) dense_ones = dense_ones && x <= 1;
                g.check(dense_ones && F.orbits == gens.size(),
                        "(P) " + Cell(N, p) + " order " + std::to_string(go.second) + ": the dense fixed space is the same orbit sums");
            }
            if (p >= 5) {
                const SparseCokernel a = SparseCokernel::Build(lvl, 0, ctx, false);
                const SparseCokernel b = SparseCokernel::BuildCurve(lvl, 0, ctx, false);
                g.check(PresentationDigest(a) == PresentationDigest(b) && a.pivots() == b.pivots() && a.free_columns() == b.free_columns(),
                        "(P) " + Cell(N, p) + ": BuildCurve = Build at p >= 5 (the same RREF)");
            }
        }
}

// ---- (Z1) the torsion gate -------------------------------------------------------------------------------
void ArmTorsion(Checker& g, uint64_t nmax, bool widened) {
    std::vector<uint64_t> levels = PrimesUpTo(nmax);
    for (uint64_t N = 4; N <= 100; ++N)
        if (!hecke::IsPrimeU64(N)) levels.push_back(N);
    if (widened)
        std::cout << "  EFFECTIVE HeckeLevelNCurveValidation arm=Z1 cells=" << levels.size() << " nmax=" << nmax
                  << " (HECKE_LN23_Z1_NMAX)" << std::endl;
    uint32_t bad = 0;
    for (uint64_t N : levels) {
        const Gamma0LevelN lvl(N);
        const uint32_t r = CurveRank(N);
        std::ostringstream os;
        os << "LN23_Z1 N=" << N << " rank=" << r;
        bool ok = true;
        for (uint32_t p : {2u, 3u, 5u, 7u, kBigP}) {
            const SparseCokernel Q = SparseCokernel::BuildCurve(lvl, 0, engine::ModCtx<uint32_t>(p), false);
            os << " F" << p << "=" << Q.h1() << "/" << Q.h0();
            ok = ok && Q.h1() == r && Q.h0() == 1;
        }
        g.check(ok, "(Z1) N=" + std::to_string(N) + ": dim = 2g + c - 1 and h0 = 1 at p = 2, 3, 5, 7, 134217757");
        if (!ok) { ++bad; std::cout << "  " << os.str() << std::endl; }
    }
    std::cout << "  LN23_Z1_SUMMARY levels=" << levels.size() << " prime_nmax=" << nmax << " composite_nmax=100 torsion_levels=" << bad
              << std::endl;
}

// ---- (H) H(Rel) in Rel at p = 2, 3 -------------------------------------------------------------------------
void ArmRelStable(Checker& g) {
    for (uint64_t N : {11ull, 17ull, 19ull, 37ull, 41ull, 73ull, 89ull, 137ull, 30ull, 64ull})
        for (uint32_t p : {2u, 3u}) {
            const Gamma0LevelN lvl(N);
            const engine::ModCtx<uint32_t> ctx(p);
            const SparseCokernel Q = SparseCokernel::BuildCurve(lvl, 0, ctx, true);
            for (uint32_t q : R3HeckePrimes(N, g_qmax)) {
                if (N % q == 0) continue;
                const HeilbronnOperator H(lvl, q, 0, ctx);
                g.check(RelEscapesOnGenerators(Q, H) == 0,
                        "(H) " + Cell(N, p) + " q=" + std::to_string(q) + ": H_{N,q}(Rel) in Rel on the curve presentation");
            }
        }
}

// ---- (X) dense = sparse on the curve route; kCurve = kGroup at p >= 5 ---------------------------------------
void ArmCrossRoute(Checker& g) {
    for (uint64_t N : {11ull, 17ull, 19ull, 37ull, 41ull, 43ull, 73ull, 30ull})
        for (uint32_t p : {2u, 3u, 5u, 7u}) {
            if (N % p == 0) continue;
            const Gamma0LevelN lvl(N);
            const engine::ModCtx<uint32_t> ctx(p);
            std::vector<uint32_t> qs;
            for (uint32_t q : R3HeckePrimes(N, g_qmax))
                if (N % q != 0) qs.push_back(q);
            const auto d = CokernelHeckeMatrices(lvl, 0, ctx, qs, CokernelRoute::kDense, nullptr, CokernelObject::kCurve);
            const auto s = CokernelHeckeMatrices(lvl, 0, ctx, qs, CokernelRoute::kSparse, nullptr, CokernelObject::kCurve);
            bool same = d.size() == s.size();
            for (const auto& kv : d) same = same && s.count(kv.first) && s.at(kv.first).a == kv.second.a;
            g.check(same, "(X) " + Cell(N, p) + ": dense = sparse Hecke matrices on the curve route");
            if (p >= 5) {
                const auto grp = CokernelHeckeMatrices(lvl, 0, ctx, qs, CokernelRoute::kSparse);
                bool eq = grp.size() == s.size();
                for (const auto& kv : grp) eq = eq && s.at(kv.first).a == kv.second.a;
                g.check(eq, "(X) " + Cell(N, p) + ": kCurve = kGroup at p >= 5");
            }
        }
}

// ---- one reading, with its family certificate (LN23-F) -----------------------------------------------------------
std::vector<uint32_t> Rungs() {
    std::vector<uint32_t> r{g_qmax};
    for (uint32_t q : kLadder)
        if (q > g_qmax) r.push_back(q);
    return r;
}

bool SameReading(const R3ScanReading& a, const R3ScanReading& b) { return a.dimW == b.dimW && a.layers == b.layers; }

// The layers are d, d - 2, ..., 3, 1 (d = dim W odd).
bool Curvilinear(const R3ScanReading& r) {
    if (r.layers.empty() || r.layers.back() != 1) return false;
    for (size_t i = 0; i < r.layers.size(); ++i)
        if (r.layers[i] + 2 * i != r.dimW) return false;
    return true;
}

struct Reading {
    R3ScanReading R;            // the family q <= QMAX (= ReadEisensteinR3Scan(N, p, R3HeckePrimes(N, QMAX)), checked by (FC0))
    R3ScanReading C;            // the certified family's reading (meaningful iff status != "NONE")
    std::string status = "NONE";   // TRIVIAL (dim W_F = 1) | CERTIFIED (good prime + stable under two widenings + curvilinear) | NONE
    uint32_t q_cert = 0, q_read = 0;   // the certified rung; the largest rung read
    std::string trail;          // rung:dimW/loewy, ...
    double wall = 0, rss = 0;
    bool certified() const { return status != "NONE"; }
    bool cert_at_qmax() const { return certified() && q_cert == g_qmax; }
};

// Reads the ladder on rd (which must be fresh) until the family is certified or the ladder ends.
void Certify(R3FamilyReader& rd, uint64_t N, Reading& r) {
    const std::vector<uint32_t> rungs = Rungs();
    std::vector<R3ScanReading> rs;
    for (size_t k = 0; k < rungs.size(); ++k) {
        rd.Extend(R3HeckePrimes(N, rungs[k]));
        rs.push_back(rd.Reading());
        r.q_read = rungs[k];
        r.trail += (k ? "," : "") + std::to_string(rungs[k]) + ":" + std::to_string(rs[k].dimW) + "/" + std::to_string(rs[k].loewy);
        if (k == 0) r.R = rs[0];
        if (rs[k].dimW == 1) {
            r.status = "TRIVIAL";
            r.C = rs[k];
            r.q_cert = rungs[k];
            return;
        }
        if (k >= 2 && SameReading(rs[k - 2], rs[k - 1]) && SameReading(rs[k - 1], rs[k]) && !rs[k - 2].good.empty() &&
            Curvilinear(rs[k - 2])) {
            r.status = "CERTIFIED";
            r.C = rs[k - 2];
            r.q_cert = rungs[k - 2];
            return;
        }
    }
}

Reading Read(uint64_t N, uint64_t p) {
    oc_util::ResetPeakRss();
    const auto t0 = std::chrono::steady_clock::now();
    Reading r;
    R3FamilyReader rd(N, p);
    Certify(rd, N, r);
    r.wall = Seconds(t0);
    r.rss = oc_util::PeakRssMiB();
    return r;
}

uint32_t FirstOf(const std::vector<uint32_t>& v) { return v.empty() ? 0 : v.front(); }

std::string Line(const char* tag, const Reading& r, const std::string& extra) {
    const R3ScanReading& R = r.R;
    std::ostringstream os;
    os << tag << " N=" << R.N << " p=" << R.p << " object=" << R.object << " dimH1=" << R.dimH1 << " rank=" << CurveRank(R.N)
       << " qmax=" << g_qmax << " dimW=" << R.dimW << " g_p=" << (R.dimW >= 1 ? (R.dimW - 1) / 2 : 0) << " loewy=" << R.loewy
       << " layers=" << Join(R.layers) << " good=" << (R.good.empty() ? "-" : Join(R.good)) << " num_v=" << R.num_condition
       << " exact=" << (R.exact ? 1 : 0) << " cert=" << r.status;
    if (r.certified())
        os << "@" << r.q_cert << " cert_dimW=" << r.C.dimW << " cert_loewy=" << r.C.loewy << " cert_layers=" << Join(r.C.layers)
           << " cert_first_good=" << FirstOf(r.C.good);
    os << " ladder=" << r.trail << extra << " wall_s=" << r.wall << " peak_rss_mib=" << r.rss;
    return os.str();
}

// The reading's structural checks (every cell, on the family q <= QMAX): the curve object, dim = 2g + 1, W odd-dimensional,
// layers strictly decreasing; and (LN23-F) the family certificate exists.
void CheckShape(Checker& g, const Reading& r) {
    const R3ScanReading& R = r.R;
    const std::string c = Cell(R.N, R.p);
    g.check(R.object == (R.p < 5 ? "curve" : "group"), c + ": object label");
    g.check(R.dimH1 == CurveRank(R.N), c + ": dimH1 = 2g + 1 (no 2- or 3-torsion, L1)");
    g.check(R.dimW % 2 == 1, c + ": dim W odd (the Eisenstein line plus a free T^0-module of rank 2, L4)");
    bool dec = !R.layers.empty() && R.layers[0] == R.dimW;
    for (size_t i = 1; i < R.layers.size(); ++i) dec = dec && R.layers[i] < R.layers[i - 1];
    g.check(dec && R.loewy == R.layers.size(), c + ": the radical layers decrease strictly");
    g.check(R.num_condition == 0 ? R.dimW == 1 : R.dimW >= 3,
            c + ": dim W_F = 1 iff p !| num((N-1)/12) (Mazur (9.7) for W_m; a separating family assumed)");
    g.check(r.certified(), "(FC) " + c + ": the family certificate (a Mazur good prime + stable under two widenings + the "
                               "curvilinear signature, or dim W_F = 1) is reached by q <= 997; ladder " + r.trail);
}

struct Oracle {
    uint64_t N, p;
    uint32_t dimH1, dimW, loewy;
    const char* source;
};

void ArmOracles(Checker& g, bool heavy) {
    std::vector<Oracle> cells{
        {41, 2, 7, 7, 4, "CE6"},       {2143, 3, 357, 7, 4, "CE6"},   {73, 2, 11, 3, 2, "Thm1.1"}, {89, 2, 15, 3, 2, "Thm1.1"},
        {137, 2, 23, 7, 4, "Thm1.1"},  {17, 2, 3, 3, 2, "proto"},     {19, 3, 3, 3, 2, "proto"},   {37, 3, 5, 3, 2, "proto"},
        {73, 3, 11, 3, 2, "proto"},
        // (Z6 c) off the cells: p !| num((N-1)/12), T^0_m = 0
        {11, 2, 3, 1, 1, "Z6c"},       {19, 2, 3, 1, 1, "Z6c"},       {43, 2, 7, 1, 1, "Z6c"},     {13, 3, 1, 1, 1, "Z6c"},
        {31, 3, 5, 1, 1, "Z6c"},
        // p >= 5 controls against WP-9 (q <= 13 here, q <= 7 there; equal on these cells, spec section 5)
        {11, 5, 3, 3, 2, "WP9"},       {31, 5, 5, 5, 3, "WP9"},       {181, 5, 29, 7, 4, "WP9"},   {29, 7, 5, 3, 2, "WP9"}};
    if (heavy) {
        cells.push_back(Oracle{21929, 2, 3655, 255, 128, "CE6-heavy"});
        std::cout << "  EFFECTIVE HeckeLevelNCurveValidation arm=HEAVY cells=1 (HECKE_LN23_HEAVY=1: (21929, 2))" << std::endl;
    }
    for (const Oracle& o : cells) {
        const Reading r = Read(o.N, o.p);
        CheckShape(g, r);
        const bool ok = r.certified() && r.C.dimH1 == o.dimH1 && r.C.dimW == o.dimW && r.C.loewy == o.loewy;
        g.check(ok, "(Z3) " + Cell(o.N, o.p) + " [" + o.source + "] on the certified family q <= " + std::to_string(r.q_cert) +
                        " (from q <= QMAX; W_F contains W_m, equality part of the prediction): dimH1/dimW/loewy = " +
                        std::to_string(o.dimH1) + "/" + std::to_string(o.dimW) + "/" + std::to_string(o.loewy) + ", read " +
                        std::to_string(r.C.dimH1) + "/" + std::to_string(r.C.dimW) + "/" + std::to_string(r.C.loewy));
        std::cout << "  " << Line("LN23_CELL", r, std::string(" source=") + o.source + (ok ? " OK" : " DISAGREE")) << std::endl;
    }
}

// ---- (Z4) CE Thm 1.1 ---------------------------------------------------------------------------------------------
void ArmThm11(Checker& g, uint64_t nmax, bool widened) {
    std::vector<uint64_t> Ns;
    for (uint64_t N = 9; N <= nmax; N += 16)
        if (hecke::IsPrimeU64(N)) Ns.push_back(N);
    if (widened)
        std::cout << "  EFFECTIVE HeckeLevelNCurveValidation arm=THM11 cells=" << Ns.size() << " nmax=" << nmax << " (HECKE_LN23_THM11_NMAX)"
                  << std::endl;
    uint32_t agree = 0, agree_qmax = 0, cert_qmax = 0;
    double max_wall = 0, max_rss = 0;
    for (uint64_t N : Ns) {
        const uint64_t h = ClassNumberNeg(-4 * static_cast<int64_t>(N));
        uint32_t m = 0;
        while (((h >> m) & 1) == 0) ++m;
        const uint32_t e = (1u << (m - 1)) - 1;
        const Reading r = Read(N, 2);
        CheckShape(g, r);
        const bool ok = m >= 2 && r.certified() && r.C.exact && r.C.dimW == 2 * e + 1 && r.C.loewy == e + 1;
        const bool ok_qmax = m >= 2 && r.R.exact && r.R.dimW == 2 * e + 1 && r.R.loewy == e + 1;
        agree += ok;
        agree_qmax += ok_qmax;
        cert_qmax += r.cert_at_qmax();
        g.check(ok, "(Z4) N=" + std::to_string(N) + ": CE Thm 1.1 on the certified family q <= " + std::to_string(r.q_cert) +
                        " (from q <= QMAX; W_F contains W_m; equality part of the prediction), h(-4N)=" + std::to_string(h) +
                        ", e_2 = 2^(m-1) - 1 = " + std::to_string(e) + ", read dimW=" + std::to_string(r.C.dimW) +
                        " loewy=" + std::to_string(r.C.loewy) + " (q <= QMAX: " + std::to_string(r.R.dimW) + "/" +
                        std::to_string(r.R.loewy) + ", cert=" + r.status + ")");
        if (!ok || !ok_qmax || !r.cert_at_qmax() || widened)
            std::cout << "  " << Line("LN23_THM11", r, " h=" + std::to_string(h) + " m=" + std::to_string(m) + " e2_pred=" +
                                                           std::to_string(e) + " qmax_agree=" + (ok_qmax ? "1" : "0") +
                                                           (ok ? " OK" : " DISAGREE"))
                      << std::endl;
        max_wall = std::max(max_wall, r.wall);
        max_rss = std::max(max_rss, r.rss);
    }
    std::cout << "  LN23_THM11_SUMMARY nmax=" << nmax << " qmax=" << g_qmax << " cells=" << Ns.size() << " agree=" << agree << " disagree="
              << (Ns.size() - agree) << " agree_at_qmax=" << agree_qmax << " certified_at_qmax=" << cert_qmax << " max_wall_s=" << max_wall << " max_peak_rss_mib=" << max_rss << std::endl;
}

// ---- (Z5) CE Thm 1.2(i) + 1.3(i) -----------------------------------------------------------------------------------
void ArmThm12i(Checker& g, uint64_t nmax, bool widened) {
    std::vector<uint64_t> Ns;
    for (uint64_t N = 10; N <= nmax; ++N)
        if ((N % 27 == 10 || N % 27 == 19) && hecke::IsPrimeU64(N)) Ns.push_back(N);
    if (widened)
        std::cout << "  EFFECTIVE HeckeLevelNCurveValidation arm=THM12I cells=" << Ns.size() << " nmax=" << nmax
                  << " (HECKE_LN23_THM12I_NMAX)" << std::endl;
    uint32_t agree = 0, e1 = 0, agree_qmax = 0, cert_qmax = 0;
    double max_wall = 0, max_rss = 0;
    for (uint64_t N : Ns) {
        const bool cube = FactorialIsCube(N);
        const Reading r = Read(N, 3);
        CheckShape(g, r);
        const bool ok = r.certified() && r.C.exact && (r.C.dimW == 3) == !cube;
        const bool ok_qmax = r.R.exact && (r.R.dimW == 3) == !cube;
        agree += ok;
        agree_qmax += ok_qmax;
        cert_qmax += r.cert_at_qmax();
        e1 += r.certified() && r.C.dimW == 3;
        g.check(ok, "(Z5) N=" + std::to_string(N) + ": CE Thm 1.2(i)+1.3(i) on the certified family q <= " + std::to_string(r.q_cert) +
                        " (from q <= QMAX; W_F contains W_m; equality part of the prediction): e_3 = 1 iff ((N-1)/3)! not a cube "
                        "mod N (cube=" + std::to_string(cube) + ", read dimW=" + std::to_string(r.C.dimW) + ", q <= QMAX: " +
                        std::to_string(r.R.dimW) + ", cert=" + r.status + ")");
        if (!ok || !ok_qmax || !r.cert_at_qmax() || widened || r.C.dimW > 3)
            std::cout << "  " << Line("LN23_THM12I", r, std::string(" fact_cube=") + (cube ? "1" : "0") + (ok ? " OK" : " DISAGREE"))
                      << std::endl;
        max_wall = std::max(max_wall, r.wall);
        max_rss = std::max(max_rss, r.rss);
    }
    std::cout << "  LN23_THM12I_SUMMARY nmax=" << nmax << " qmax=" << g_qmax << " cells=" << Ns.size() << " agree=" << agree << " disagree="
              << (Ns.size() - agree) << " agree_at_qmax=" << agree_qmax << " certified_at_qmax=" << cert_qmax << " e3_eq_1=" << e1 << " max_wall_s=" << max_wall << " max_peak_rss_mib=" << max_rss << std::endl;
}

// ---- (Z6 a, b) the refusals ----------------------------------------------------------------------------------------
void ArmRefusals(Checker& g) {
    const Gamma0LevelN lvl(11);
    for (uint32_t p : {2u, 3u}) {
        const engine::ModCtx<uint32_t> ctx(p);
        const std::string c = " at p = " + std::to_string(p);
        // (a) n > 0 on the curve route, by name
        g.check(RefusedWith([&] { (void)CokernelHeckeMatrices(lvl, 2, ctx, {7u}, CokernelRoute::kSparse, nullptr, CokernelObject::kCurve); },
                            "object=curve"),
                "(Z6a) CokernelHeckeMatrices kCurve, n = 2" + c + ": refused by name");
        g.check(RefusedWith([&] { (void)SparseCokernel::BuildCurve(lvl, 2, ctx, false); }, "object=curve"),
                "(Z6a) SparseCokernel::BuildCurve, n = 2" + c + ": refused by name");
        g.check(RefusedWith([&] { (void)CokernelPresentation::BuildCurve(lvl, 2, ctx); }, "object=curve"),
                "(Z6a) CokernelPresentation::BuildCurve, n = 2" + c + ": refused by name");
        // (b) the averaging path at n > 0, and the default (group) entry points, refused as before
        g.check(RefusedWith([&] { (void)OrbitWiseFixedSpace(lvl, Mat22::ST(), 3, 2, engine::ModCtx<uint32_t>(3)); }, "not a unit"),
                "(Z6b) OrbitWiseFixedSpace, n = 2, p = 3: the averaging path refused");
        g.check(RefusedWith([&] { (void)OrbitWiseFixedGenerators(lvl, Mat22::S(), 2, 2, engine::ModCtx<uint32_t>(2)); }, "order a unit"),
                "(Z6b) OrbitWiseFixedGenerators, n = 2, p = 2: the averaging path refused");
        g.check(RefusedWith([&] { (void)CokernelHeckeMatrices(lvl, 0, ctx, {7u}); }, "p in {2, 3} refused"),
                "(Z6b) CokernelHeckeMatrices (default object), n = 0" + c + ": the slot-15 refusal unchanged");
        g.check(RefusedWith([&] { (void)SparseCokernel::Build(lvl, 0, ctx, false); }, "6 a unit"),
                "(Z6b) SparseCokernel::Build (group), n = 0" + c + ": refused");
        g.check(RefusedWith([&] { (void)CokernelPresentation::Build(lvl, 0, ctx); }, "6 a unit"),
                "(Z6b) CokernelPresentation::Build (group), n = 0" + c + ": refused");
        g.check(RefusedWith([&] { (void)ReadEisensteinR3(41, p); }, "p >= 5"), "(Z6b) ReadEisensteinR3 (WP-8)" + c + ": refused");
        g.check(RefusedWith([&] { (void)ReadEisensteinR3Scan(45, p == 2 ? 2 : 3, {5u, 7u}); }, "N prime"),
                "(Z6b) ReadEisensteinR3Scan at a composite level" + c + ": refused (curve readings at prime N only)");
    }
}

// ---- the read-only hook HECKE_LEVEL_N_R3_SMALLP_CELLS --------------------------------------------------------------
std::vector<std::pair<uint64_t, uint64_t>> ParseSmallP(const char* env) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    std::stringstream ss(env);
    std::string item;
    size_t idx = 0;
    while (std::getline(ss, item, ',')) {
        ++idx;
        const std::string where = "HECKE_LEVEL_N_R3_SMALLP_CELLS: cell " + std::to_string(idx) + " '" + item + "': ";
        std::vector<uint64_t> v;
        std::stringstream is(item);
        std::string tok;
        while (std::getline(is, tok, ':')) {
            const bool digits = !tok.empty() && tok.size() <= 10 && tok.find_first_not_of("0123456789") == std::string::npos;
            ENGINE_CHECK_MSG(digits, (where + "'" + tok + "' is not a non-negative integer").c_str());
            v.push_back(std::stoull(tok));
        }
        ENGINE_CHECK_MSG(v.size() == 2, (where + "expected N:p").c_str());
        ENGINE_CHECK_MSG(v[0] >= 5 && v[0] < (1ull << 20) && hecke::IsPrimeU64(v[0]), (where + "N must be a prime in [5, 2^20)").c_str());
        ENGINE_CHECK_MSG(v[1] == 2 || v[1] == 3, (where + "p must be 2 or 3 (the curve object; p >= 5: HECKE_LEVEL_N_R3_CELLS)").c_str());
        ENGINE_CHECK_MSG(NumConditionValuation(v[0], v[1]) >= 1,
                         (where + "p must divide num((N - 1)/12) (Mazur: T^0_m != 0; at p = 2: N = 1 mod 8, at p = 3: N = 1 mod 9)").c_str());
        out.emplace_back(v[0], v[1]);
    }
    ENGINE_CHECK_MSG(!out.empty(), "HECKE_LEVEL_N_R3_SMALLP_CELLS: the value lists no cell (refused: the arm would run vacuously)");
    return out;
}

void ArmSmallPHook(Checker& g, const std::vector<std::pair<uint64_t, uint64_t>>& cells) {
    std::cout << "  EFFECTIVE HeckeLevelNCurveValidation arm=SMALLP smallp_cells=" << cells.size()
              << " object=curve (HECKE_LEVEL_N_R3_SMALLP_CELLS; every other arm skipped)" << std::endl;
    for (const auto& c : cells) {
        const Reading r = Read(c.first, c.second);
        CheckShape(g, r);
        std::string extra;
        const std::string fam = " on the certified family q <= " + std::to_string(r.q_cert) +
                                " (from q <= QMAX; W_F contains W_m; equality part of the prediction)";
        if (c.second == 2 && r.R.exact) {
            const uint64_t h = ClassNumberNeg(-4 * static_cast<int64_t>(c.first));
            uint32_t m = 0;
            while (((h >> m) & 1) == 0) ++m;
            const uint32_t e = (1u << (m - 1)) - 1;
            const bool ok = r.certified() && r.C.dimW == 2 * e + 1 && r.C.loewy == e + 1;
            const bool ok_qmax = r.R.dimW == 2 * e + 1 && r.R.loewy == e + 1;
            g.check(ok, "(SMALLP) " + Cell(c.first, 2) + ": CE Thm 1.1" + fam);
            extra = " h=" + std::to_string(h) + " e2_pred=" + std::to_string(e) + " qmax_agree=" + (ok_qmax ? "1" : "0") +
                    (ok ? " OK" : " DISAGREE");
        } else if (c.second == 3 && r.R.exact) {
            const bool cube = FactorialIsCube(c.first);
            const bool ok = r.certified() && (r.C.dimW == 3) == !cube;
            const bool ok_qmax = (r.R.dimW == 3) == !cube;
            g.check(ok, "(SMALLP) " + Cell(c.first, 3) + ": CE Thm 1.2(i)+1.3(i)" + fam);
            extra = std::string(" fact_cube=") + (cube ? "1" : "0") + " qmax_agree=" + (ok_qmax ? "1" : "0") + (ok ? " OK" : " DISAGREE");
        } else {
            extra = " not_exact (a reading: g_p, not e_p)";
        }
        std::cout << "  " << Line("LN23_SMALLP", r, extra) << std::endl;
    }
}

// ---- (FC0) the family reader = ReadEisensteinR3Scan, value for value --------------------------------------------------
void ArmReaderIdentity(Checker& g) {
    for (const auto& c : std::vector<std::pair<uint64_t, uint64_t>>{{41, 2}, {137, 2}, {1747, 3}, {2143, 3}, {181, 5}, {31, 5}}) {
        const std::vector<uint32_t> qs = R3HeckePrimes(c.first, g_qmax);
        const R3ScanReading a = ReadEisensteinR3Scan(c.first, c.second, qs);
        R3FamilyReader rd(c.first, c.second);
        // in two steps (the first three primes, then the rest): the incremental restriction is what is being checked
        rd.Extend(std::vector<uint32_t>(qs.begin(), qs.begin() + std::min<size_t>(3, qs.size())));
        rd.Extend(qs);
        const R3ScanReading b = rd.Reading();
        g.check(a.dimH1 == b.dimH1 && a.dimW == b.dimW && a.layers == b.layers && a.W == b.W && a.w_after == b.w_after &&
                    a.qs == b.qs && a.good == b.good && a.object == b.object && a.exact == b.exact,
                "(FC0) " + Cell(c.first, c.second) + ": R3FamilyReader = ReadEisensteinR3Scan (dimH1, W basis, layers, w_after, good)");
    }
    // MazurGoodPrime against hand values (Mazur II.16 Def.): (41, 2): 3, 7, 11 good, 2 (case (b), excluded), 5, 13 not;
    // (1747, 3): 2 good (2 not a cube mod 1747; (2 - 1)/2 = 1/2 a 3-adic unit), 7, 13 not (= 1 mod 3); (2143, 3): 2 not (a cube).
    const bool hand = MazurGoodPrime(41, 2, 3) && MazurGoodPrime(41, 2, 7) && MazurGoodPrime(41, 2, 11) && !MazurGoodPrime(41, 2, 2) &&
                      !MazurGoodPrime(41, 2, 5) && !MazurGoodPrime(41, 2, 13) && MazurGoodPrime(1747, 3, 2) &&
                      MazurGoodPrime(1747, 3, 3) && !MazurGoodPrime(1747, 3, 7) && !MazurGoodPrime(1747, 3, 13) &&
                      !MazurGoodPrime(2143, 3, 2) && MazurGoodPrime(2137, 2, 23) && !MazurGoodPrime(2137, 2, 19) &&
                      !MazurGoodPrime(43, 2, 3);
    g.check(hand, "(FC0) MazurGoodPrime on the hand values (41,2), (1747,3), (2143,3), (2137,2), (43,2) (p !| num: none)");
}

// ---- (G) generation: a single good prime gives the full radical series -----------------------------------------------
// eta_q (q good) generates m T_m (Mazur (16.6) / (17.1)), so on W_m its powers ARE the m-adic layers: the single-prime
// family {q} reads W_{q} containing W_m (= the certified W) with a series at least as long as the certified one.  A
// non-good q has eta_q in m^2 (printed, not gated: W_{q} may contain other components with longer series).
void ArmGeneration(Checker& g) {
    struct Gc { uint64_t N, p; std::vector<uint32_t> qs; };
    const std::vector<Gc> cells{{41, 2, {2, 3, 5, 7, 11, 13}},
                                {2137, 2, {3, 7, 11, 19, 23}},
                                {2143, 3, {2, 3, 5, 7, 11, 13}},
                                {1747, 3, {2, 3, 5, 7, 11, 13}}};
    for (const Gc& c : cells) {
        const Reading r = Read(c.N, c.p);
        g.check(r.certified(), "(G) " + Cell(c.N, c.p) + ": the certified reading exists");
        std::cout << "  " << Line("LN23_GEN_CERT", r, "") << std::endl;
        for (uint32_t q : c.qs) {
            R3FamilyReader rd(c.N, c.p);
            rd.Extend({q});
            const R3ScanReading s = rd.Reading();
            const bool good = MazurGoodPrime(c.N, c.p, q);
            if (good)
                g.check(r.certified() && s.dimW >= r.C.dimW && s.loewy >= r.C.loewy,
                        "(G) " + Cell(c.N, c.p) + " {" + std::to_string(q) + "}: q a Mazur good prime -> eta_q alone reads dim W_{q} >= " +
                            std::to_string(r.C.dimW) + " and Loewy >= " + std::to_string(r.C.loewy) + " (read " + std::to_string(s.dimW) +
                            "/" + std::to_string(s.loewy) + ")");
            std::cout << "  LN23_GEN N=" << c.N << " p=" << c.p << " q=" << q << " good=" << (good ? 1 : 0) << " dimW=" << s.dimW
                      << " loewy=" << s.loewy << " layers=" << Join(s.layers) << " cert_dimW=" << r.C.dimW << " cert_loewy=" << r.C.loewy
                      << std::endl;
        }
    }
    // (G2) Mazur's predicate is sufficient, not necessary, at p = 2 for q = 2: at (2473, 2) (e_2 = 1; 2 a quartic residue,
    // so good in neither of Mazur's senses; no good prime <= 13, the first is 19) the separating prime 13 alone reads
    // W_{13} = W_m (dim 3, eta_13 = 0 on it) and {13, 2} reads 3 / 3,1: eta_2 generates m T_m/2 there, while {13, q} for
    // q = 3, 5, 7, 11 (odd, not good) reads Loewy 1.  An observation (the referee's route ln23ref.py agrees), kept as a
    // regression check; the certificate does not use it (it waits for the good prime 19).
    {
        const uint64_t N = 2473;
        std::string got;
        bool ok = true;
        for (uint32_t q : {2u, 3u, 5u, 7u, 11u}) {
            R3FamilyReader rd(N, 2);
            rd.Extend({13u, q});
            const R3ScanReading s = rd.Reading();
            ok = ok && s.dimW == 3 && s.loewy == (q == 2 ? 2u : 1u) && !MazurGoodPrime(N, 2, q);
            got += " {13," + std::to_string(q) + "}:" + std::to_string(s.dimW) + "/" + std::to_string(s.loewy);
        }
        g.check(ok, "(G2) (N=2473, p=2): eta_2 generates on W_m although 2 is a quartic residue (not Mazur-good); odd non-good q "
                    "do not; read" + got);
        std::cout << "  LN23_GEN2 N=2473 p=2 quartic2=1" << got << std::endl;
    }
}

// ---- STURM (read-only arm HECKE_LN23_STURM=1): the separating family q <= B = floor((N + 1)/6) ------------------------
// [C] rmk:levelNcurvefamily (2): T^0 = Hom_Z(S_2(Gamma_0(N); Z), Z) under (t, f) -> a_1(tf), and a cusp form with a_n = 0 mod p
// for n <= B vanishes mod p (Sturm, Thm 1), so the T_n, n <= B -- polynomials in the T_q, q <= B prime -- span T^0/p and the
// family q <= B separates m from every other maximal ideal of T^0/p; it contains a good prime when e_p >= 2.  Caveats:
// (i) the argument is on T^0: the constant term of the full algebra T/p at p in {2, 3} is not covered (every other maximal
// ideal of T/p comes from T^0/p, which is what separation needs); (ii) Loewy = e + 1 still needs (LN6)'s faithfulness CONJ.
void ArmSturm(Checker& g, const std::vector<std::pair<uint64_t, uint64_t>>& cells) {
    std::cout << "  EFFECTIVE HeckeLevelNCurveValidation arm=STURM cells=" << cells.size()
              << " (HECKE_LN23_STURM=1; every other arm skipped)" << std::endl;
    std::cout << "  STURM caveats: separation by T_n, n <= B = floor((N+1)/6), on T^0 = Hom(S_2(Gamma_0(N); Z), Z) (Sturm Thm 1);"
                 " (i) the constant term of the full algebra at p in {2, 3} is not covered; (ii) Loewy = e + 1 needs (LN6)'s"
                 " faithfulness CONJ" << std::endl;
    uint32_t agree = 0, same = 0;
    double max_wall = 0, max_rss = 0;
    for (const auto& c : cells) {
        const uint64_t N = c.first, p = c.second;
        const uint32_t B = static_cast<uint32_t>((N + 1) / 6);
        oc_util::ResetPeakRss();
        const auto t0 = std::chrono::steady_clock::now();
        Reading r;
        R3FamilyReader rd(N, p);
        Certify(rd, N, r);
        R3ScanReading S;
        if (B >= r.q_read) {
            rd.Extend(R3HeckePrimes(N, B));   // the ladder's primes are a prefix of q <= B: the same family
            S = rd.Reading();
        } else {
            R3FamilyReader fresh(N, p);
            fresh.Extend(R3HeckePrimes(N, B));
            S = fresh.Reading();
        }
        const double wall = Seconds(t0), rss = oc_util::PeakRssMiB();
        max_wall = std::max(max_wall, wall);
        max_rss = std::max(max_rss, rss);
        const std::string cl = Cell(N, p) + " B=" + std::to_string(B);
        const bool eq = r.certified() && SameReading(S, r.C);
        same += eq;
        g.check(eq, "(ST1) " + cl + ": the Sturm family's reading = the certified reading (dim W, layers)");
        g.check(S.dimW == 1 || !S.good.empty(), "(ST2) " + cl + ": the Sturm family contains a Mazur good prime (dim W > 1)");
        bool ok = true;
        std::string pred;
        if (p == 2 && S.exact) {
            const uint64_t h = ClassNumberNeg(-4 * static_cast<int64_t>(N));
            uint32_t m = 0;
            while (((h >> m) & 1) == 0) ++m;
            const uint32_t e = (1u << (m - 1)) - 1;
            ok = S.dimW == 2 * e + 1 && S.loewy == e + 1;
            pred = " e2_pred=" + std::to_string(e);
            g.check(ok, "(ST3) " + cl + ": CE Thm 1.1 on the Sturm family q <= B (W_F = W_m by Sturm on T^0; Loewy: (LN6) CONJ)");
        } else if (p == 3 && S.exact) {
            const bool cube = FactorialIsCube(N);
            ok = (S.dimW == 3) == !cube;
            pred = std::string(" fact_cube=") + (cube ? "1" : "0");
            g.check(ok, "(ST3) " + cl + ": CE Thm 1.2(i)+1.3(i) on the Sturm family q <= B (W_F = W_m by Sturm on T^0)");
        }
        agree += ok;
        std::cout << "  LN23_STURM N=" << N << " p=" << p << " B=" << B << " primes=" << S.qs.size() << " dimH1=" << S.dimH1
                  << " dimW=" << S.dimW << " loewy=" << S.loewy << " layers=" << Join(S.layers) << " good_count=" << S.good.size()
                  << " first_good=" << FirstOf(S.good) << " cert=" << r.status << "@" << r.q_cert << " cert_dimW=" << r.C.dimW
                  << " cert_loewy=" << r.C.loewy << " same_as_cert=" << (eq ? 1 : 0) << pred << (ok ? " OK" : " DISAGREE")
                  << " wall_s=" << wall << " peak_rss_mib=" << rss << std::endl;
    }
    std::cout << "  LN23_STURM_SUMMARY cells=" << cells.size() << " same_as_cert=" << same << " ce_agree=" << agree
              << " max_wall_s=" << max_wall << " max_peak_rss_mib=" << max_rss << std::endl;
}

std::vector<std::pair<uint64_t, uint64_t>> SturmDefaultCells() {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    for (uint64_t N = 9; N <= 2000; N += 16)
        if (hecke::IsPrimeU64(N)) out.emplace_back(N, 2);
    for (uint64_t N = 10; N <= 2500; ++N)
        if ((N % 27 == 10 || N % 27 == 19) && hecke::IsPrimeU64(N)) out.emplace_back(N, 3);
    out.emplace_back(2137, 2);   // generation: no good prime <= 13, the first is 23
    out.emplace_back(2377, 2);   // generation: no good prime <= 13, the first is 19
    out.emplace_back(2953, 2);   // separation: 7 good, yet q <= 13 reads 39; {7, 19} reads 7
    return out;
}

} // namespace

uint32_t HeckeLevelNCurveValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::cout << "HeckeLevelNCurveValidation: LN23, weight 2, the curve object at p = 2, 3 ([C] prop:levelNcurve, rmk:levelNcurvefamily;"
                 " LN23_spec v0.2 section 10, v0.3 section 11: the LN23-F family certificate)"
              << std::endl;
    bool qw = false;
    g_qmax = static_cast<uint32_t>(EnvNmax("HECKE_LN23_QMAX", 13, 997, qw));
    if (qw) std::cout << "  EFFECTIVE HeckeLevelNCurveValidation qmax=" << g_qmax << " (HECKE_LN23_QMAX)" << std::endl;
    const char* st = std::getenv("HECKE_LN23_STURM");
    if (st != nullptr) ENGINE_CHECK_MSG(std::string(st) == "1", "HECKE_LN23_STURM must be 1 (the read-only Sturm arm) or unset");
    if (st != nullptr) {
        const char* env = std::getenv("HECKE_LEVEL_N_R3_SMALLP_CELLS");
        ArmSturm(g, env ? ParseSmallP(env) : SturmDefaultCells());
    } else if (const char* env = std::getenv("HECKE_LEVEL_N_R3_SMALLP_CELLS")) {
        ArmSmallPHook(g, ParseSmallP(env));
    } else {
        bool z1w = false, t11w = false, t12w = false;
        const uint64_t z1 = EnvNmax("HECKE_LN23_Z1_NMAX", 600, 20000, z1w);
        const uint64_t t11 = EnvNmax("HECKE_LN23_THM11_NMAX", 2000, 100000, t11w);
        const uint64_t t12 = EnvNmax("HECKE_LN23_THM12I_NMAX", 2500, 100000, t12w);
        const char* hv = std::getenv("HECKE_LN23_HEAVY");
        const bool heavy = hv != nullptr && std::string(hv) == "1";
        ArmOrbitSums(g);
        std::cout << "  [(P) done at " << Seconds(t0) << " s]" << std::endl;
        ArmTorsion(g, z1, z1w);
        std::cout << "  [(Z1) done at " << Seconds(t0) << " s]" << std::endl;
        ArmRelStable(g);
        ArmCrossRoute(g);
        std::cout << "  [(H), (X) done at " << Seconds(t0) << " s]" << std::endl;
        ArmRefusals(g);
        ArmOracles(g, heavy);
        std::cout << "  [(Z3), (Z6) done at " << Seconds(t0) << " s]" << std::endl;
        ArmReaderIdentity(g);
        ArmGeneration(g);
        std::cout << "  [(FC0), (G) done at " << Seconds(t0) << " s]" << std::endl;
        ArmThm11(g, t11, t11w);
        ArmThm12i(g, t12, t12w);
    }
    std::cout << "HeckeLevelNCurveValidation: " << g.checks << " checks, " << g.fails << " failing, " << Seconds(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
