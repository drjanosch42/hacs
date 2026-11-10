// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LC) see HeckeLatticeModuleValidation.h.
#include "HeckeLatticeModuleValidation.h"

#include "HeckeLatticeModule.h"
#include "../Util/ValidationCheck.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hecke_ext::latmod;

namespace {

oc_util::Checker Check("  [FAIL] ");

uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}

std::vector<uint32_t> Cap(std::vector<uint32_t> v, uint32_t M) {
    for (uint32_t& x : v) x = std::min(x, M);
    std::sort(v.begin(), v.end());
    return v;
}

std::string Br(const std::vector<uint32_t>& v) { return "<" + JoinU32(v) + ">"; }

// ---- the readings of chk:latcmp on one free module with t and U -------------------------------------------
struct Reading {
    uint32_t rank = 0;
    uint32_t mu = 0;                   // dim H / (ell, t) H
    std::vector<uint32_t> st, su;      // Smith valuations of t, U
};
Reading Read(const FreeMat& t, const FreeMat& U) {
    Reading R;
    R.rank = t.r;
    R.mu = t.r - RankModEllJoint({t});
    R.st = SmithValuations(t);
    R.su = SmithValuations(U);
    return R;
}
bool Matches(const Reading& R, uint32_t rank, uint32_t mu, const std::vector<uint32_t>& st, const std::vector<uint32_t>& su, uint32_t M) {
    return R.rank == rank && R.mu == mu && R.st == Cap(st, M) && R.su == Cap(su, M);
}

// ---- R19-1's data (in source, mod 37^6): the planted controls ---------------------------------------------
const std::vector<uint64_t> kP = {956586012ull, 894859171ull, 711659591ull, 1248134135ull, 1ull};
const std::vector<uint64_t> kP1 = {924207201ull, 1404397900ull, 1ull};
const std::vector<uint64_t> kP2 = {1773752287ull, 2409462644ull, 1ull};
const std::vector<uint64_t> kQ37 = {2051474472ull, 713970241ull, 2038601840ull, 2471121741ull};
// chk:latcmp's predicted valuations
const std::vector<uint32_t> kTmT = {0, 0, 0, 2}, kTmU = {0, 0, 6, 6};       // T_m
const std::vector<uint32_t> kMT = {0, 0, 1, 1}, kMU = {0, 1, 5, 6};         // m ~= R
const std::vector<uint32_t> kAllT = {0, 0, 0, 0, 0, 1, 1, 2}, kAllU = {0, 0, 0, 1, 5, 6, 6, 6};

FreeMat Companion(const std::vector<uint64_t>& P, uint64_t ell, uint32_t K) {
    const uint32_t d = static_cast<uint32_t>(P.size() - 1);
    FreeMat C = Identity(ell, K, d);
    std::fill(C.a.begin(), C.a.end(), 0);
    const uint64_t q = C.q();
    for (uint32_t j = 0; j + 1 < d; ++j) C.a[static_cast<size_t>(j + 1) * d + j] = 1;
    for (uint32_t i = 0; i < d; ++i) C.a[static_cast<size_t>(i) * d + (d - 1)] = (q - P[i] % q) % q;
    return C;
}

// t on m = (37, t) T_m, basis 37, t, t^2, t^3, at precision K <= 5 (P_0 / 37 is known mod 37^5).
FreeMat MaxIdealT(uint32_t K) {
    const uint64_t ell = 37;
    FreeMat C = Identity(ell, K, 4);
    std::fill(C.a.begin(), C.a.end(), 0);
    const uint64_t q = C.q();
    C.a[1 * 4 + 0] = 37 % q;                      // t * 37 = 37 t
    C.a[2 * 4 + 1] = 1;                           // t * t = t^2
    C.a[3 * 4 + 2] = 1;                           // t * t^2 = t^3
    const uint64_t p0 = kP[0] / 37;               // exact: 37 | P_0
    C.a[0 * 4 + 3] = (q - p0 % q) % q;
    for (uint32_t i = 1; i < 4; ++i) C.a[static_cast<size_t>(i) * 4 + 3] = (q - kP[i] % q) % q;
    return C;
}

void PlantedControls() {
    std::cout << "(P) planted: R19-1's P, P_1, P_2, Q_37 mod 37^6; the checker on T_m (+) m, T_m (+) T~, T_m^2" << std::endl;
    Check(kP[0] % 37 == 0 && kP[0] % (37 * 37) == 0, "(P) 37^2 | P_0");
    for (uint32_t M = 1; M <= 4; ++M) {
        const FreeMat tTm = ReduceTo(Companion(kP, 37, 6), M);
        const FreeMat tM = ReduceTo(MaxIdealT(5), M);
        const FreeMat tTt = ReduceTo(DirectSum(Companion(kP1, 37, 6), Companion(kP2, 37, 6)), M);
        const FreeMat zero4 = Scale(Identity(37, M, 4), 0);
        Check(EvalPoly(kP, tTm) == zero4 && EvalPoly(kP, tM) == zero4 && EvalPoly(kP, tTt) == zero4, "(P) P(t) = 0 on T_m, m, T~");
        const FreeMat UTm = EvalPoly(kQ37, tTm), UM = EvalPoly(kQ37, tM), UTt = EvalPoly(kQ37, tTt);
        const Reading rTm = Read(tTm, UTm), rM = Read(tM, UM), rTt = Read(tTt, UTt);
        Check(Matches(rTm, 4, 1, kTmT, kTmU, M), "(P) T_m: mu 1, t <0,0,0,2>, U <0,0,6,6> at M=" + std::to_string(M));
        Check(Matches(rM, 4, 2, kMT, kMU, M), "(P) m: mu 2, t <0,0,1,1>, U <0,1,5,6> at M=" + std::to_string(M));
        Check(Matches(rTt, 4, 2, {0, 0, 1, 1}, {1, 1, 5, 5}, M), "(P) T~: mu 2, t <0,0,1,1>, U <1,1,5,5> at M=" + std::to_string(M));
        const Reading good = Read(DirectSum(tTm, tM), DirectSum(UTm, UM));
        const Reading bad1 = Read(DirectSum(tTm, tTt), DirectSum(UTm, UTt));
        const Reading bad2 = Read(DirectSum(tTm, tTm), DirectSum(UTm, UTm));
        Check(Matches(good, 8, 3, kAllT, kAllU, M), "(P) T_m (+) m has chk:latcmp (b), (c) at M=" + std::to_string(M));
        Check(!Matches(bad1, 8, 3, kAllT, kAllU, M) && bad1.mu == 3 && bad1.st == Cap(kAllT, M) && bad1.su != Cap(kAllU, M),
              "(N) T_m (+) T~ is rejected, by U_37 alone, at M=" + std::to_string(M));
        Check(!Matches(bad2, 8, 3, kAllT, kAllU, M) && bad2.mu == 2, "(N) T_m^2 is rejected (mu = 2) at M=" + std::to_string(M));
        std::cout << "  PLANTED M=" << M << " Tm+m: mu=" << good.mu << " t=" << Br(good.st) << " U=" << Br(good.su)
                  << " | Tm+Tt: mu=" << bad1.mu << " U=" << Br(bad1.su) << " | Tm^2: mu=" << bad2.mu << std::endl;
    }
}

// ---- the tracked data file ----------------------------------------------------------------------------------
bool PathExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}
const char* kDataRel = "docs/implementation/slot19/r191/data/universal_eigenform_n1000_mod37e6.txt";
std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string root(OC_SOURCE_DIR);
        if (!root.empty() && PathExists(root + "/" + kDataRel)) return root;
    }
#endif
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
            if (PathExists(dir + "/" + kDataRel)) return dir;
        }
    }
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (PathExists(dir + "/" + kDataRel)) return dir;
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
        }
    }
    return std::string();
}

struct Universal {
    std::vector<uint64_t> P;                          // mod 37^6, low -> high
    std::map<uint64_t, std::vector<uint64_t>> Q;      // n -> c0..c3 mod 37^6
};
bool ReadUniversal(Universal& U) {
    const std::string root = FindRepoRoot();
    if (root.empty()) return false;
    std::ifstream in(root + "/" + kDataRel);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t k = line.find("P = [");
            if (k != std::string::npos) {
                std::string body = line.substr(k + 5, line.find(']', k) - (k + 5));
                std::replace(body.begin(), body.end(), ',', ' ');
                std::istringstream is(body);
                uint64_t x;
                while (is >> x) U.P.push_back(x);
            }
            continue;
        }
        std::istringstream is(line);
        uint64_t n;
        std::vector<uint64_t> c(4);
        if (is >> n >> c[0] >> c[1] >> c[2] >> c[3]) U.Q[n] = c;
    }
    return U.P.size() == 5 && U.Q.size() >= 100;
}

bool IsPrime(uint64_t p) {
    if (p < 2) return false;
    for (uint64_t d = 2; d * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}

std::vector<uint64_t> ModAll(const std::vector<uint64_t>& v, uint64_t q) {
    std::vector<uint64_t> o;
    for (uint64_t x : v) o.push_back(x % q);
    return o;
}

// P(t) == 0 and T_p == Q_p(t) on a part, for the given primes (Q mod ell^6, reduced to the part's precision).
uint32_t CheckPolys(const LatticeBlock& B, const FreePart& part, const FreeMat& t, const std::vector<uint64_t>& P,
                    const std::map<uint64_t, std::vector<uint64_t>>& Q, const std::vector<uint64_t>& primes, const std::string& tag) {
    const uint64_t q = t.q();
    Check(EvalPoly(ModAll(P, q), t) == Scale(Identity(B.ell, t.K, t.r), 0), tag + " P(t) = 0");
    uint32_t ok = 0;
    for (uint64_t p : primes) {
        const auto it = Q.find(p);
        if (it == Q.end()) {
            Check(false, tag + " no Q_" + std::to_string(p) + " in the data");
            continue;
        }
        const bool eq = B.op(part, p) == EvalPoly(ModAll(it->second, q), t);
        Check(eq, tag + " T_" + std::to_string(p) + " = Q_" + std::to_string(p) + "(t)");
        ok += eq ? 1 : 0;
    }
    return ok;
}

std::string Cell(uint32_t n, uint64_t ell) { return "(" + std::to_string(n) + "," + std::to_string(ell) + ")"; }
std::string Sys(const std::vector<uint64_t>& a) {
    std::ostringstream os;
    os << "(";
    for (size_t i = 0; i < a.size(); ++i) os << (i ? "," : "") << a[i];
    os << ")";
    return os.str();
}

// ---- (L) (258, 37) ----------------------------------------------------------------------------------------
bool FullArm() {
    const char* v = std::getenv("HECKE_LATTICE_MODULE_FULL");
    return v != nullptr && std::string(v) == "1";
}

// ---- (L) (258, 37) ----------------------------------------------------------------------------------------
// Default arm: ONE module run at K = 4 with T_2, T_3, T_37 (≈ 100 s); H_M for M = 1..3 is H_4 / 37^M, which is
// chk:latcmp (a) (PROVED: H^2 has no 37-torsion).  Full arm (HECKE_LATTICE_MODULE_FULL=1, read-only): K = 4 with every
// prime p < 100 ((e) in full, ≈ 18 min) and independent runs at K = 2, 3 (T_2, T_37) compared with the reductions.
void Cell258(const Universal& U, bool full) {
    const uint32_t n = 258;
    const uint64_t ell = 37;
    std::vector<uint64_t> e_primes;
    if (full) {
        for (uint64_t p = 2; p < 100; ++p)
            if (IsPrime(p)) e_primes.push_back(p);
    } else {
        e_primes = {2, 3, 37};
    }
    std::cout << "(L) (258, 37), m = (37, T_2 - 29), chk:latcmp (a)-(e), M = 1..4; (e) on " << e_primes.size()
              << (full ? " primes p < 100" : " primes {2, 3, 37} (the full p < 100 list: HECKE_LATTICE_MODULE_FULL=1)") << std::endl;
    const LatticeModule N4 = RunLatticeModule(n, ell, 4, e_primes);
    const LatticeBlock B4 = LocalizeBlock(N4, {2}, {29});
    std::cout << "  RUN K=4 H^1 rank=" << N4.exps.size() << " exps_min=" << *std::min_element(N4.exps.begin(), N4.exps.end())
              << " D=" << B4.D << " primes=" << N4.primes.size() << " seconds=" << N4.seconds << std::endl;
    Check(N4.exps.size() == 43, "(L) rank H^1(Gamma; Sym^258 Z/37^4) = 2 dim M_260 - 1 = 43");
    std::string free_sign;
    for (uint32_t M = 1; M <= 4; ++M) {
        const std::string tag = "(L) " + Cell(n, ell) + " M=" + std::to_string(M);
        // (a)
        Check(B4.D == 8 && B4.block.rank == 8 && B4.block.free(), tag + " (a) H_M free of rank 8");
        Check(B4.plus.rank == 4 && B4.minus.rank == 4 && B4.plus.free() && B4.minus.free(), tag + " (a) each sign free of rank 4");
        if (!(B4.block.free() && B4.plus.free() && B4.minus.free() && B4.block.rank == 8)) break;
        const LatticeBlock B = (M == 4) ? B4 : ReduceBlock(B4, M);
        const FreeMat t = AddScalar(B.op(B.block, 2), -29);
        const Reading all = Read(t, B.op(B.block, 37));
        // (b), (c)
        Check(all.mu == 3, tag + " (b) mu = 3");
        Check(all.st == Cap(kAllT, M), tag + " (c) Smith(t) = <0^5,1,1,2>_M");
        Check(all.su == Cap(kAllU, M), tag + " (c) Smith(U_37) = <0^3,1,5,6,6,6>_M");
        // (d)
        const FreeMat tp = AddScalar(B.op(B.plus, 2), -29), tm = AddScalar(B.op(B.minus, 2), -29);
        const Reading rp = Read(tp, B.op(B.plus, 37)), rm = Read(tm, B.op(B.minus, 37));
        const bool plus_free = Matches(rp, 4, 1, kTmT, kTmU, M) && Matches(rm, 4, 2, kMT, kMU, M);
        const bool minus_free = Matches(rm, 4, 1, kTmT, kTmU, M) && Matches(rp, 4, 2, kMT, kMU, M);
        Check(plus_free != minus_free, tag + " (d) one sign is T_m (mu 1, <0,0,0,2>, <0,0,6,6>), the other m (mu 2, <0,0,1,1>, <0,1,5,6>)");
        const std::string fs = plus_free ? "plus" : (minus_free ? "minus" : "none");
        if (free_sign.empty()) free_sign = fs;
        Check(fs == free_sign, tag + " (d) the free sign is the same at every M");
        // (N) both signs free / both m is rejected
        Check(!(Matches(rp, 4, 1, kTmT, kTmU, M) && Matches(rm, 4, 1, kTmT, kTmU, M)) &&
                  !(Matches(rp, 4, 2, kMT, kMU, M) && Matches(rm, 4, 2, kMT, kMU, M)),
              tag + " (N) the two signs are not alike");
        // (e)
        const uint32_t ok = CheckPolys(B, B.block, t, U.P, U.Q, B.primes, tag + " (e)");
        CheckPolys(B, B.plus, tp, U.P, U.Q, B.primes, tag + " (e) plus");
        CheckPolys(B, B.minus, tm, U.P, U.Q, B.primes, tag + " (e) minus");
        // (N) a perturbed Q_p (c_0 + ell^{M-1}) is rejected
        {
            bool rejected = true;
            for (uint64_t p : {uint64_t(3), uint64_t(37)}) {
                std::vector<uint64_t> bad = U.Q.at(p);
                bad[0] += PowU(ell, M - 1);
                if (B.op(B.block, p) == EvalPoly(ModAll(bad, t.q()), t)) rejected = false;
            }
            Check(rejected, tag + " (N) a perturbed Q_p (c_0 + 37^(M-1), p = 3, 37) is rejected");
        }
        std::cout << "  LATCMP cell=" << Cell(n, ell) << " system=(T_2-29) M=" << M << " rank=" << B.block.rank << " free=1"
                  << " mu=" << all.mu << " smith_t=" << Br(all.st) << " smith_U37=" << Br(all.su)
                  << " plus:mu=" << rp.mu << ",t=" << Br(rp.st) << ",U=" << Br(rp.su)
                  << " minus:mu=" << rm.mu << ",t=" << Br(rm.st) << ",U=" << Br(rm.su)
                  << " free_sign=" << fs << " e_primes_ok=" << ok << "/" << B.primes.size() << std::endl;
    }
    Check(free_sign == "plus", "(L) (258,37) the free summand is the plus sign (reported by chk:latcmp (d); pinned after the slot-19 run)");
    std::cout << "  LATCMP_SIGN cell=(258,37) free_sign=" << free_sign << " (Lambda_m^" << free_sign << " ~= T_m, the other ~= m)" << std::endl;
    if (!full) return;
    // (a) H_M = H_K / 37^M: independent runs at K = 2, 3 read as the K = 4 run reduced
    for (uint32_t K = 2; K <= 3; ++K) {
        const LatticeModule NK = RunLatticeModule(n, ell, K, {2, 37});
        const LatticeBlock BK = LocalizeBlock(NK, {2}, {29});
        std::cout << "  RUN K=" << K << " seconds=" << NK.seconds << std::endl;
        Check(BK.block.free() && BK.block.rank == 8 && BK.plus.free() && BK.minus.free(), "(L) (258,37) the K=" + std::to_string(K) + " run: free of rank 8");
        if (!(BK.block.free() && BK.plus.free() && BK.minus.free())) continue;
        const LatticeBlock R = ReduceBlock(B4, K);
        for (const auto& pr : {std::make_pair(&BK.block, &R.block), std::make_pair(&BK.plus, &R.plus), std::make_pair(&BK.minus, &R.minus)}) {
            const Reading a = Read(AddScalar(BK.op(*pr.first, 2), -29), BK.op(*pr.first, 37));
            const Reading b = Read(AddScalar(R.op(*pr.second, 2), -29), R.op(*pr.second, 37));
            Check(a.mu == b.mu && a.st == b.st && a.su == b.su, "(L) (258,37) the K=" + std::to_string(K) + " run reads as the K=4 run reduced");
        }
        CheckPolys(BK, BK.block, AddScalar(BK.op(BK.block, 2), -29), U.P, U.Q, BK.primes, "(L) (258,37) K=" + std::to_string(K) + " (e)");
    }
}

// ---- (C) ell = 7 controls ---------------------------------------------------------------------------------
// The forms-side P (mod 7^12) and Q_p (mod 7^6) of wplc/l7_pred.py.  The over-order candidates Z_7[t / 7^j] are formed
// from P in source (t = 7^j * companion(P(7^j X) / 7^{dj})) while integral; each sign's reading is matched against them.
// The PREDICTION of wplc/predictions.md ("free": Lambda^+- ~= T_m on both signs) is printed and was REFUTED by the
// slot-19 run on (48,7) plus, (90,7) (3,3,1,0) both signs and (90,7) (1,0,0,0) both signs -- a finding (multiplicity
// one fails mod 7 at these cells), recorded in predictions.md Sec. 3; the gate pins the observed order per sign.
struct Control {
    uint32_t n;
    std::vector<uint64_t> sys;                         // (a_2, a_3, a_5, a_7)
    uint32_t r;                                        // rank per sign
    std::vector<uint32_t> st, su;                      // forms-side Smith valuations of t, U_7 (the "free" prediction)
    std::vector<uint64_t> P;                           // charpoly(t) mod 7^12
    std::map<uint64_t, std::vector<uint64_t>> Q;       // mod 7^6; empty: T_m not monogenic in t
    const char* loewy;
    const char* plus_order;                            // the pinned order per sign ("" = none of the candidates)
    const char* minus_order;
    std::vector<uint32_t> obs_t, obs_U;                // pinned readings where no candidate is pinned (uncapped values)
    uint32_t obs_mu;
};

// t on Z_7[t / 7^j] (basis 1, s, ..., s^{d-1}, s = t / 7^j) at precision 12 - d j, or r == 0 if not integral.
FreeMat OrderT(const std::vector<uint64_t>& P, uint32_t j) {
    const uint64_t ell = 7;
    const uint32_t d = static_cast<uint32_t>(P.size() - 1);
    const uint32_t prec = 12 - d * j;
    std::vector<uint64_t> c(d + 1);
    for (uint32_t i = 0; i <= d; ++i) {
        const uint64_t div = PowU(ell, j * (d - i));
        if (P[i] % div != 0) return FreeMat{};
        c[i] = (P[i] / div) % PowU(ell, prec);
    }
    return Scale(Companion(c, ell, prec), PowU(ell, j));
}

void Controls(bool full) {
    const uint64_t ell = 7;
    const std::vector<uint64_t> sp = {2, 3, 5, 7};
    const std::vector<Control> cs = {
        {48, {1, 0, 0, 0}, 2, {0, 2}, {1, 5}, {7626891797ull, 9624907922ull, 1},
         {{2, {1, 1}}, {3, {109991, 22460}}, {5, {56812, 101284}}, {7, {31262, 36512}}, {11, {47163, 102915}}, {13, {3346, 89925}}},
         "2", "Z_7[t/7]", "T_m", {}, {}, 0},
        {90, {3, 3, 1, 0}, 2, {0, 3}, {2, 10}, {2833507222ull, 12660251653ull, 1},
         {{2, {3, 1}}, {3, {25028, 45431}}, {5, {101550, 68511}}, {7, {21266, 105840}}, {11, {964, 36214}}, {13, {100492, 16900}}},
         "2", "Z_7[t/7]", "Z_7[t/7]", {}, {}, 0},
        {90, {1, 0, 0, 0}, 4, {0, 0, 1, 4}, {0, 4, 8, 12}, {11196991470ull, 7958435548ull, 7642603402ull, 81848571ull, 1}, {},
         "3", "Z_7[t/7]", "", {0, 0, 2, 3}, {1, 4, 4, 4}, 2},
    };
    std::cout << "(C) ell = 7 controls (wplc/predictions.md): prediction 'free' printed; the gate pins the observed order per sign" << std::endl;
    std::map<std::pair<uint32_t, uint32_t>, LatticeModule> mods;
    for (const Control& c : cs) {
        // the candidates Z_7[t / 7^j]; distinct readings at every M are checked below
        std::vector<std::pair<std::string, FreeMat>> cand;
        for (uint32_t j = 0; j <= 3; ++j) {
            const FreeMat tj = OrderT(c.P, j);
            if (tj.r == 0 || 12 < c.r * j + 4) break;
            cand.push_back({j == 0 ? std::string("T_m") : "Z_7[t/" + std::to_string(PowU(ell, j)) + "]", tj});
        }
        Check(cand.size() >= 2, "(C) " + Cell(c.n, ell) + " " + Sys(c.sys) + " at least the candidates T_m and Z_7[t/7]");
        const std::vector<uint32_t> Ks = full ? std::vector<uint32_t>{2, 3, 4} : std::vector<uint32_t>{4};
        for (uint32_t K : Ks) {
            const auto key = std::make_pair(c.n, K);
            if (!mods.count(key)) mods[key] = RunLatticeModule(c.n, ell, K, {2, 3, 5, 7, 11, 13});
            const LatticeBlock BK = LocalizeBlock(mods.at(key), sp, c.sys);
            const bool fr = BK.block.free() && BK.plus.free() && BK.minus.free();
            Check(fr && BK.block.rank == 2 * c.r && BK.plus.rank == c.r && BK.minus.rank == c.r,
                  "(C) " + Cell(c.n, ell) + " " + Sys(c.sys) + " K=" + std::to_string(K) + ": free of rank " + std::to_string(2 * c.r) +
                      ", each sign of rank " + std::to_string(c.r));
            if (!fr || BK.block.rank != 2 * c.r) continue;
            std::vector<uint32_t> Ms;
            if (K == 4) Ms = full ? std::vector<uint32_t>{4} : std::vector<uint32_t>{1, 2, 3, 4};
            else Ms = (K == 2) ? std::vector<uint32_t>{1, 2} : std::vector<uint32_t>{K};
            for (uint32_t M : Ms) {
                const std::string tag = "(C) " + Cell(c.n, ell) + " " + Sys(c.sys) + " M=" + std::to_string(M);
                const LatticeBlock B = (M == K) ? BK : ReduceBlock(BK, M);
                // the candidates' readings at M
                std::vector<Reading> cr;
                for (const auto& cd : cand) {
                    const FreeMat tj = ReduceTo(cd.second, M);
                    cr.push_back(c.Q.empty() ? Read(tj, tj) : Read(tj, EvalPoly(ModAll(c.Q.at(7), tj.q()), tj)));
                }
                for (size_t a = 0; a < cr.size(); ++a)
                    for (size_t b = a + 1; b < cr.size(); ++b)
                        Check(!(cr[a].mu == cr[b].mu && cr[a].st == cr[b].st), tag + " (N) the candidates " + cand[a].first + ", " + cand[b].first + " are told apart");
                std::ostringstream line;
                line << "  LATCMP cell=" << Cell(c.n, ell) << " system=" << Sys(c.sys) << " loewy=" << c.loewy << " M=" << M
                     << " K=" << K << " rank=" << B.block.rank << " free=1";
                bool pred_free = true;
                for (const FreePart* part : {&B.plus, &B.minus}) {
                    const bool is_plus = (part == &B.plus);
                    const char* sg = is_plus ? "plus" : "minus";
                    const FreeMat t = AddScalar(B.op(*part, 2), -static_cast<int64_t>(c.sys[0]));
                    std::vector<FreeMat> nil;
                    for (size_t i = 0; i < sp.size(); ++i) nil.push_back(AddScalar(B.op(*part, sp[i]), -static_cast<int64_t>(c.sys[i])));
                    const uint32_t mu_m = part->rank - RankModEllJoint(nil);
                    const Reading rd = Read(t, B.op(*part, 7));
                    // forced: P(t) = 0, T_p = Q_p(t) ([C, prop:cohforms]: T_m ~= T_m^f)
                    if (!c.Q.empty()) {
                        CheckPolys(B, *part, t, c.P, c.Q, {2, 3, 5, 7, 11, 13}, tag + " " + sg);
                        Check(mu_m == rd.mu, tag + " " + sg + ": m = (7, t) (T_m = Z_7[t]): mu_m == mu_t");
                    } else {
                        Check(EvalPoly(ModAll(c.P, t.q()), t) == Scale(Identity(ell, t.K, t.r), 0), tag + " " + sg + ": P(t) = 0");
                    }
                    // the prediction "free" (printed; refuted where the pinned order is not T_m)
                    const bool free_here = mu_m == 1 && rd.st == Cap(c.st, M) && rd.su == Cap(c.su, M);
                    pred_free = pred_free && free_here;
                    // the matched candidate (t and mu; U as well where Q_7 exists)
                    std::string match;
                    for (size_t a = 0; a < cand.size(); ++a) {
                        const bool m_ok = cr[a].mu == rd.mu && cr[a].st == rd.st && (c.Q.empty() || cr[a].su == rd.su);
                        if (m_ok) match = cand[a].first;
                    }
                    const std::string pinned = is_plus ? c.plus_order : c.minus_order;
                    Check(match == pinned, tag + " " + sg + ": the order is " + (pinned.empty() ? std::string("none of the candidates") : pinned) +
                                               " (read: " + (match.empty() ? std::string("none") : match) + ")");
                    if (pinned.empty()) {
                        Check(rd.mu == c.obs_mu && rd.st == Cap(c.obs_t, M) && rd.su == Cap(c.obs_U, M),
                              tag + " " + sg + ": the pinned observed readings mu " + std::to_string(c.obs_mu) + ", t " + Br(Cap(c.obs_t, M)) + ", U " + Br(Cap(c.obs_U, M)));
                    }
                    line << " " << sg << ":order=" << (match.empty() ? "?" : match) << ",mu_m=" << mu_m << ",mu_t=" << rd.mu
                         << ",t=" << Br(rd.st) << ",U=" << Br(rd.su);
                }
                line << " prediction_free=" << (pred_free ? "held" : "refuted");
                std::cout << line.str() << std::endl;
            }
        }
    }
    // (N) a system that is not an eigenvalue is refused
    bool refused = false;
    try {
        (void)LocalizeBlock(mods.begin()->second, {2}, {0});
    } catch (const std::logic_error&) {
        refused = true;
    }
    Check(refused, "(N) a_2 = 0 (no eigenvalue of T_2 mod 7 at " + Cell(mods.begin()->first.first, ell) + ") is refused");
}

} // namespace

uint32_t HeckeLatticeModuleValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const bool full = FullArm();
    std::cout << "HeckeLatticeModuleValidation: [E] chk:latcmp at (258, 37), M = 1..4; ell = 7 controls; negative controls" << std::endl;
    std::cout << "EFFECTIVE HeckeLatticeModuleValidation arm=" << (full ? "FULL" : "DEFAULT") << " cells=4" << std::endl;
    try {
        PlantedControls();
        Universal U;
        const bool have = ReadUniversal(U);
        Check(have, std::string("R19-1's data file ") + kDataRel + " found and parsed (P and Q_n, n <= 1000)");
        if (have) {
            Check(ModAll(U.P, PowU(37, 6)) == kP, "the file's P equals the in-source P (mod 37^6)");
            Check(U.Q.count(37) && U.Q.at(37) == kQ37, "the file's Q_37 equals the in-source Q_37 (mod 37^6)");
            Controls(full);
            Cell258(U, full);
        }
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeLatticeModuleValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s << " s" << std::endl;
    return Check.fails == 0 ? 0 : 1;
}
