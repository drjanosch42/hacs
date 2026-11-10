// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 16, WP-15 -- HeckeHidaCMValidation (standard).
//
// [U] part 2 (docs/latex/hida_hecke_algebra.tex v0.2.0-pre.2, sec:u-tameN) in code:
// the CM components read on the character model of [C, sec:nebentypus]
// (def:nebmodel, prop:neborbit, rmk:nebheilbronn; Hecke/HeckeNebentypus, slot-16
// WP-10: NebCokernel = H^1(Gamma_0(N); V(chi)) as a cokernel, HeilbronnChi = T_p /
// U_p on it).  chi = chi_K is quadratic, so the [GT, conv:nebentypus] dictionary
// (chi versus chibar) cannot move a value here.  Every cell is (N, chi_K, k, ell),
// n = k - 2, ell >= 5, ell !| N; H^1 is read over F_ell (and over F_q, q = 10^9 + 7,
// as the characteristic-0 proxy: a class killed by T_p over F_q for every inert
// p <= 50 is a class whose integral eigenvalues vanish there).
//
//   (A) arithmetic and the oracles: h_K by reduced forms against the classical
//       table (and the nine h = 1 discriminants below 200), w_K, chi_K against the
//       LMFDB Conrey labels of WP-9's table (and the unique quadratic Conrey label at
//       the other moduli), the CM oracle a_p(theta_{lambda_k}) = Tr(alpha^{k-1})
//       against every split / inert nb6_ap row of WP-9_nb_predictions.tsv (LMFDB),
//       genus theory at d = -15 (p = 2 non-principal).  WP-9's table has LMFDB a_p rows only at k = 3, 5, 7; at the
//       weights carrying (U8) / (U10) (up to 23) the oracle is checked by exact-vs-mod consistency (p <= 100) and an
//       independent evaluation of Tr alpha^{k-1} by powers of alpha in Z[w] at (-3, 19), (-7, 23) (repairs, m2).
//   (C0) characteristic-0 CM classes ([U] thm:u-cmN, prop:u-cmexist): over F_q,
//       dim of the joint kernel of T_p, p <= 50 inert in K, on H^1(Gamma_0(|d_K|);
//       V(chi_K)) is 2 h_K at w_K | k - 1 and 0 otherwise (the h_K Hecke characters
//       of conductor 1 and infinity-type k - 1; 0 when lambda_k is undefined); and
//       (U9) it is 0 at tame level 1 and 2 for EVERY imaginary quadratic K with
//       |d_K| <= 100, including Q(sqrt -47) at (1, 24) and (1, 72) where (U9b)
//       finds residually dihedral blocks.
//   (U8) PROVED: on the CM ladders (the (NB6) cells and k + w_K, k + 2 w_K, ...,
//       d in {-3, -4, -7, -8, -11}, h_K = 1) at every ell <= 31 with ell !| N: the
//       joint generalised eigenspace W of (T_p)_{p <= 37} at the CM system contains
//       the two CM classes (+ the Eisenstein classes when their systems coincide);
//       T_ell's eigenvalue on them is a_ell(theta) mod ell, a unit iff ell splits
//       (ordinary) and 0 iff ell is inert; exactly (X - a_ell)^2 on W when dim W = 2.
//       The coinciding Eisenstein classes are subtracted before every T_ell gate (repairs, M1): one class per
//       Eisenstein system, calibrated by the Eisenstein-multiplicity control on every run where the CM system differs
//       from both Eisenstein systems.  The prime list covers the Sturm bound k [SL_2(Z) : Gamma_0(N)] / 12 (M2).
//       The excess dim W - 2 - #Eis is rmk:u-cmgl2's CM cap GL(2) reading (reported).
//   (U9b) the residually dihedral blocks at tame level 1 and 2: for every imaginary
//       K with |d_K| <= 100 the joint generalised 0-eigenspace of T_p, p <= 50 inert
//       in K, over F_ell, first on p <= 50, confirmed on p <= 100 and re-confirmed (unchanged) on p <= 150; gated:
//       every K found is unramified outside N ell, and d_K = -ell exactly at N = 1 (rmk:u-dihedral(i)); every
//       detection shadowed at p <= 50 only has a confirmed block at the same (N, k, ell); reported with its ordinary
//       dimension; controls pinned: (1, 24, 47) 5-dim ordinary, (1, 72, 47) 10-dim non-ordinary (WP-11's J-7 fixtures).
//   (U10) PROVED (prop:u-cmeis(2), h_K = 1 checked): at (4, chi_-4, 5, 5),
//       (3, chi_-3, 7, 7), (7, chi_-7, 11, 11) and six more ladder cells with
//       (ell - 1) | (k - 1), ell split: a_p(theta) = 1 + chi(p) p^{k-1} =
//       chi(p) + p^{k-1} mod ell (integers, p <= 50) and dim W == 2 Eisenstein + 2 CM + the pinned excess (0 at
//       eight cells, 4 at (3, chi_-3, 13) ell = 13 and (11, chi_-11, 5) ell = 5) at that system;
//       CONJ (reported, a failure is a finding): the local algebra of the T_p on W
//       has Loewy length >= 2.  The negative example h_K = 2: (15, chi_-15, 17, 17)
//       and (15, chi_-15, 19, 19) -- lambda_k is undefined at the non-principal
//       split primes, the two conductor-1 characters split between E_k(1, chi_K)
//       and E_k(chi_-3, chi_5) (WP-14 referee U-1): both systems carry exactly 4 classes.
//   (ENV) HECKE_HIDA_CM_CELLS="N:d:k:ell,..." (read-only, not an emitter; d < 0 the
//       fundamental discriminant of K, |d| | N): replaces the default arms; one
//       HIDA_CM_CELL line per cell (with sturm_bound, p_max, gates=gated|reported) and one EFFECTIVE line; the
//       prime list is extended to the Sturm bound up to 100 (T_p cost grows with p and k: (4, chi_-4, 101) at p <= 50 takes ~110 s), beyond it (U8) / (U10) are reported only (M2); a
//       malformed or empty list is refused loudly (an exception: [EXCEPTION] / [FAIL]).
// ===========================================================================

#include "HeckeHidaCMValidation.h"

#include "HeckeHidaCM.h"

#include "../Hecke/HeckeLevelN.h"
#include "../Hecke/HeckeNebentypus.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace hecke_ext::hidacm;
using hecke::neb::CharacterValues;
using hecke::neb::DirichletCharacter;
using hecke::neb::HeilbronnChi;
using hecke::neb::NebCokernel;
using hecke::neb::NebentypusRefusal;

constexpr uint64_t kQ = 1000000007ull;   // the characteristic-0 proxy field F_q
constexpr uint64_t kB0 = 50;             // T_p, p <= 50, for the characteristic-0 readings
constexpr uint64_t kBd = 100;            // T_p, p <= 100, for the (U9b) residually dihedral reading (p <= 50 let
                                         // Q(sqrt -40), Q(sqrt -84), Q(sqrt -91) shadow Q(sqrt -ell): finite-B artefacts)
constexpr uint64_t kBl = 37;             // T_p, p <= 37, for the mod-ell systems (raised to the Sturm bound where it exceeds 37)
constexpr uint64_t kBd2 = 150;           // (U9b) re-confirmation: a confirmed detection must not change on p <= 150 (audit m3)
constexpr uint64_t kHookPmaxCap = 100;   // (ENV) the prime list is extended to the Sturm bound up to this cap; beyond it the
                                         // cell is reported only (audit M2)

struct Checker {
    uint32_t checks = 0, fails = 0;
    bool quiet = false;   // a reported-only checker (the hook beyond the Sturm cap): tallies, prints REPORTED instead of FAIL
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 100) std::cout << (quiet ? "  REPORTED (not gated): " : "  FAIL: ") << what << "\n";
        }
    }
};

bool IsPrime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p = 2; p * p <= n; ++p)
        if (n % p == 0) return false;
    return true;
}
std::vector<uint64_t> PrimesUpTo(uint64_t B) {
    std::vector<uint64_t> v;
    for (uint64_t p = 2; p <= B; ++p)
        if (IsPrime(p)) v.push_back(p);
    return v;
}
std::string I128(__int128 x) {
    if (x == 0) return "0";
    const bool neg = x < 0;
    unsigned __int128 u = neg ? static_cast<unsigned __int128>(-x) : static_cast<unsigned __int128>(x);
    std::string s;
    while (u) { s.push_back(static_cast<char>('0' + static_cast<int>(u % 10))); u /= 10; }
    if (neg) s.push_back('-');
    return std::string(s.rbegin(), s.rend());
}
std::string VecText(const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}
std::string CellName(uint64_t N, int64_t d, uint32_t k) {
    return "(" + std::to_string(N) + ",chi_" + std::to_string(d) + "," + std::to_string(k) + ")";
}
std::set<uint64_t> PrimeSupport(uint64_t n) {
    std::set<uint64_t> s;
    for (uint64_t p = 2; p * p <= n; ++p)
        while (n % p == 0) { s.insert(p); n /= p; }
    if (n > 1) s.insert(n);
    return s;
}

// The Sturm bound of M_k(Gamma_0(N), chi): floor(k [SL_2(Z) : Gamma_0(N)] / 12), [SL_2(Z) : Gamma_0(N)] = N prod_{p | N}
// (1 + 1/p).  The mod-ell systems below are read on T_p for every prime p <= p_max with p !| N ell; a reading is gated only
// when p_max >= this bound (S16 WP-15 repairs, audit M2).  T_p at p | N and T_ell are not in the joint system, so this is
// the bound for the systems away from N ell, not a proof that the system is determined by them.
uint64_t SturmBound(uint64_t N, uint32_t k) {
    uint64_t idx = N;
    for (uint64_t p : PrimeSupport(N)) idx = idx / p * (p + 1);
    return static_cast<uint64_t>(k) * idx / 12;
}
uint64_t PrimeBound(uint64_t N, uint32_t k) { return std::max<uint64_t>(kBl, SturmBound(N, k)); }

// ---- one cell of the character model over F_r -------------------------------------
struct CellModel {
    uint64_t N = 1, r = 5;
    int64_t d = 0;   // 0: the trivial character (tame level 1, 2)
    uint32_t k = 2;
    std::unique_ptr<Fq> F;
    CharacterValues chi;
    std::unique_ptr<hecke::level::Gamma0LevelN> lvl;
    std::unique_ptr<NebCokernel> K;
    std::string refusal;
    std::map<uint64_t, FqMat> T;
    uint32_t h1() const { return K && !K->parity_zero() ? K->h1() : 0; }
    const FqMat& Op(uint64_t p) {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        const HeilbronnChi H(*lvl, p, k - 2, *F, chi);
        return T[p] = K->OnH1(H);
    }
    uint32_t chi_at(uint64_t p) const { return chi.at(static_cast<int64_t>(p)); }
};
std::unique_ptr<CellModel> MakeCell(uint64_t N, int64_t d, uint32_t k, uint64_t r) {
    std::unique_ptr<CellModel> c(new CellModel);
    c->N = N;
    c->d = d;
    c->k = k;
    c->r = r;
    c->F.reset(new Fq(Fq::Prime(static_cast<uint32_t>(r))));
    const DirichletCharacter chi = d == 0 ? DirichletCharacter::Trivial(N) : KroneckerCharacter(d, N);
    c->chi = hecke::neb::ReduceCharacter(chi, *c->F);
    c->lvl.reset(new hecke::level::Gamma0LevelN(N));
    try {
        c->K.reset(new NebCokernel(*c->lvl, k - 2, *c->F, c->chi));
    } catch (const NebentypusRefusal& e) {
        c->refusal = e.code();
    }
    return c;
}

// (C0): the joint kernel over the cell's field of T_p, p <= B inert in Q(sqrt dK), p !| N, p != r.
// Returns UINT32_MAX when no such p exists.
uint32_t InertKernelDim(CellModel& c, int64_t dK, uint64_t B, bool generalised) {
    if (c.h1() == 0) return 0;
    std::vector<FqMat> ops;
    for (uint64_t p : PrimesUpTo(B))
        if (c.N % p != 0 && p != c.r && Kronecker(dK, p) == -1) ops.push_back(c.Op(p));
    if (ops.empty()) return UINT32_MAX;
    if (generalised) return JointGeneralisedEigenspace(*c.F, ops, std::vector<uint32_t>(ops.size(), 0)).cols;
    const uint32_t h = c.h1();
    FqMat S(static_cast<uint32_t>(ops.size()) * h, h);
    for (size_t i = 0; i < ops.size(); ++i)
        for (uint32_t a = 0; a < h; ++a)
            for (uint32_t b = 0; b < h; ++b) S.at(static_cast<uint32_t>(i) * h + a, b) = ops[i].at(a, b);
    return Kernel(*c.F, S).cols;
}

// ---- (U8) / (U10) readings at one (cell, ell) ------------------------------------------
struct CmReading {
    uint32_t h1 = 0, dimW = 0, eis = 0, gen_al = 0, ordW = 0, ord_h1 = 0;
    // (audit M1) the coinciding Eisenstein systems' T_ell eigenvalues mod ell: E_k(1, chi) has 1 + chi(ell) ell^{k-1},
    // E_k(chi, 1) has chi(ell) + ell^{k-1}; eis_al = how many coinciding ones have the eigenvalue a_ell, eis_ord = how
    // many are ordinary (a unit).  One class per Eisenstein system, calibrated by the (U8) Eisenstein-multiplicity control.
    uint32_t eis_al = 0, eis_ord = 0;
    uint64_t pmax = 0, sturm = 0;
    uint64_t al = 0;
    bool split = false, coinc1 = true, coinc2 = true;
    std::vector<uint32_t> charpolyW;
    FqMat W;
    std::vector<uint64_t> primes;
    std::vector<uint32_t> e1;
};
CmReading ReadCm(CellModel& c, uint64_t B) {
    const Fq& F = *c.F;
    CmReading R;
    R.h1 = c.h1();
    R.pmax = B;
    R.sturm = SturmBound(c.N, c.k);
    std::vector<FqMat> ops;
    std::vector<uint32_t> scal;
    for (uint64_t p : PrimesUpTo(B)) {
        if (c.N % p == 0 || p == c.r) continue;
        const uint32_t a = static_cast<uint32_t>(CmCoefficientMod(c.d, c.k, p, c.r));
        const uint32_t pk1 = F.pow(F.from_int(static_cast<int64_t>(p)), c.k - 1);
        const uint32_t e1 = F.add(1, F.mul(c.chi_at(p), pk1)), e2 = F.add(c.chi_at(p), pk1);
        R.coinc1 = R.coinc1 && a == e1;
        R.coinc2 = R.coinc2 && a == e2;
        R.primes.push_back(p);
        R.e1.push_back(e1);
        ops.push_back(c.Op(p));
        scal.push_back(a);
    }
    R.eis = (R.coinc1 ? 1u : 0u) + (R.coinc2 ? 1u : 0u);
    R.W = JointGeneralisedEigenspace(F, ops, scal);
    R.dimW = R.W.cols;
    const FqMat& Tl = c.Op(c.r);
    const FqMat TlW = Restrict(F, Tl, R.W);
    R.al = CmCoefficientMod(c.d, c.k, c.r, c.r);
    R.split = Kronecker(c.d, c.r) == 1;
    R.gen_al = GeneralisedEigenDim(F, TlW, static_cast<uint32_t>(R.al));
    R.ordW = R.dimW - GeneralisedEigenDim(F, TlW, 0);
    R.ord_h1 = R.h1 - GeneralisedEigenDim(F, Tl, 0);
    R.charpolyW = hecke::neb::FqCharpoly(F, TlW);
    const uint32_t lk1 = F.pow(F.from_int(static_cast<int64_t>(c.r)), c.k - 1);
    const uint32_t eL1 = F.add(1, F.mul(c.chi_at(c.r), lk1)), eL2 = F.add(c.chi_at(c.r), lk1);
    const uint32_t a = static_cast<uint32_t>(R.al);
    R.eis_al = (R.coinc1 && eL1 == a ? 1u : 0u) + (R.coinc2 && eL2 == a ? 1u : 0u);
    R.eis_ord = (R.coinc1 && eL1 != 0 ? 1u : 0u) + (R.coinc2 && eL2 != 0 ? 1u : 0u);
    return R;
}

// The U8 gates on one reading; returns the CM cap GL(2) excess.
uint32_t GateU8(Checker& g, CellModel& c, const CmReading& R, const std::string& tag) {
    const Fq& F = *c.F;
    const std::string at = tag + " " + CellName(c.N, c.d, c.k) + " ell=" + std::to_string(c.r);
    g.check(R.pmax >= R.sturm, at + ": the prime list p <= " + std::to_string(R.pmax) + " covers the Sturm bound " +
                                   std::to_string(R.sturm) + " (audit M2)");
    g.check(R.dimW >= 2 + R.eis, at + ": dim W = " + std::to_string(R.dimW) + " >= 2 CM classes + " + std::to_string(R.eis) +
                                     " Eisenstein class(es) at the CM system");
    // (audit m1) Only the split half is a check: at inert ell the oracle returns 0 by definition (a_p = 0 at inert p), so
    // "inert => a_ell = 0" is tautological here; the cohomological content of the inert case is the gen_al gate below.
    if (R.split)
        g.check(R.al != 0, at + ": ell splits, a_ell(theta) = Tr alpha^{k-1} mod ell = " + std::to_string(R.al) +
                               " is a unit -- prop:u-cmexist (the inert half, a_ell = 0, is the oracle's definition, not checked here)");
    // (audit M1) the Eisenstein classes at a coinciding system are subtracted: they carry T_ell-eigenvalue 1 + chi(ell)
    // ell^{k-1} or chi(ell) + ell^{k-1} mod ell, which can equal a_ell (the U10 cells), so ">= 2" alone was not CM-specific.
    g.check(R.gen_al >= 2 + R.eis_al, at + ": T_ell has the eigenvalue a_ell = " + std::to_string(R.al) + " on " +
                                          std::to_string(R.gen_al) + " >= 2 + " + std::to_string(R.eis_al) +
                                          " (Eisenstein classes at that eigenvalue) classes of W: the CM system is " +
                                          (R.split ? "ordinary" : "non-ordinary"));
    if (R.split) g.check(R.ordW >= 2 + R.eis_ord, at + ": the CM classes lie in the ordinary part (dim W_ord = " + std::to_string(R.ordW) +
                                                      " >= 2 + " + std::to_string(R.eis_ord) + " ordinary Eisenstein classes)");
    else g.check(R.dimW - R.ordW >= 2 + (R.eis - R.eis_ord), at + ": the CM classes lie outside the ordinary part (dim W_nonord = " +
                                                                 std::to_string(R.dimW - R.ordW) + " >= 2 + " +
                                                                 std::to_string(R.eis - R.eis_ord) + " non-ordinary Eisenstein classes)");
    if (R.dimW == 2) {
        const uint32_t a = static_cast<uint32_t>(R.al);
        const std::vector<uint32_t> want{F.mul(a, a), F.neg(F.add(a, a)), 1};
        g.check(R.charpolyW == want, at + ": dim W = 2, so T_ell | W = (X - a_ell)^2 exactly: " + VecText(R.charpolyW));
    }
    return R.dimW >= 2 + R.eis ? R.dimW - 2 - R.eis : 0;
}

void PrintU8(const CellModel& c, const CmReading& R, uint32_t excess) {
    std::cout << "  HIDA_CM_U8 cell=" << CellName(c.N, c.d, c.k) << " ell=" << c.r << " " << (R.split ? "split" : "inert")
              << " h1=" << R.h1 << " W=" << R.dimW << " eis=" << R.eis << " eis_at_a_ell=" << R.eis_al << " cm_gl2_excess=" << excess
              << " a_ell=" << R.al << " gen_a_ell=" << R.gen_al << " sturm=" << R.sturm << " p_max=" << R.pmax
              << " T_ell|W=" << VecText(R.charpolyW) << " ord_h1=" << R.ord_h1 << " ord_W=" << R.ordW
              << " cm=" << (R.split ? "ordinary" : "non-ordinary") << "\n";
}

// (audit M1 (c)) the Eisenstein multiplicity: the joint generalised eigenspaces of the two Eisenstein systems
// E_k(1, chi): 1 + chi(p) p^{k-1} and E_k(chi, 1): chi(p) + p^{k-1} on the primes p <= B, p !| N r.
struct EisDims { uint32_t w1 = 0, w2 = 0; bool same = false; };
EisDims EisensteinDims(CellModel& c, uint64_t B) {
    const Fq& F = *c.F;
    std::vector<FqMat> ops;
    std::vector<uint32_t> s1, s2;
    for (uint64_t p : PrimesUpTo(B)) {
        if (c.N % p == 0 || p == c.r) continue;
        const uint32_t pk1 = F.pow(F.from_int(static_cast<int64_t>(p)), c.k - 1);
        s1.push_back(F.add(1, F.mul(c.chi_at(p), pk1)));
        s2.push_back(F.add(c.chi_at(p), pk1));
        ops.push_back(c.Op(p));
    }
    EisDims e;
    e.same = s1 == s2;
    e.w1 = JointGeneralisedEigenspace(F, ops, s1).cols;
    e.w2 = e.same ? e.w1 : JointGeneralisedEigenspace(F, ops, s2).cols;
    return e;
}

// (audit M1 (c)) the Eisenstein-cusp congruence criterion, independent of the cohomology: a cusp form congruent to
// E_k(1, chi) mod ell (chi primitive of conductor f = |d|, ell !| 2f) exists iff ell | B_{k,chi} / k, the generalised
// Bernoulli number B_{k,chi} = sum_{a=1}^{f} chi(a) sum_{j=0}^{k} C(k, j) B_j a^{k-j} f^{j-1}.  The Bernoulli numbers are
// computed here by their recurrence in exact fractions; B_{k,chi} is evaluated modulo ell^4 after scaling by ell (the
// denominators of the B_j are squarefree, so ell B_j is ell-integral).  Returns v_ell(B_{k,chi} / k), capped at 2.
struct Frac { __int128 n = 0, d = 1; };
__int128 Gcd128(__int128 a, __int128 b) { if (a < 0) a = -a; if (b < 0) b = -b; while (b) { const __int128 t = a % b; a = b; b = t; } return a; }
Frac Reduce(Frac x) { const __int128 g = Gcd128(x.n, x.d); if (g > 1) { x.n /= g; x.d /= g; } if (x.d < 0) { x.n = -x.n; x.d = -x.d; } return x; }
const std::vector<Frac>& Bernoulli() {   // B_0 .. B_32, B_1 = -1/2
    static const std::vector<Frac> B = [] {
        std::vector<Frac> b{Frac{1, 1}};
        for (int m = 1; m <= 32; ++m) {
            Frac acc{0, 1};
            __int128 binom = 1;   // C(m + 1, j)
            for (int j = 0; j < m; ++j) {
                acc = Reduce(Frac{acc.n * b[j].d + binom * b[j].n * acc.d, acc.d * b[j].d});
                binom = binom * (m + 1 - j) / (j + 1);
            }
            b.push_back(Reduce(Frac{-acc.n, acc.d * (m + 1)}));
        }
        return b;
    }();
    return B;
}
uint64_t PowMod(uint64_t a, uint64_t e, uint64_t m) { uint64_t r = 1 % m; a %= m; while (e) { if (e & 1) r = r * a % m; a = a * a % m; e >>= 1; } return r; }
uint64_t InvMod(uint64_t a, uint64_t m) {   // a a unit mod m
    int64_t t = 0, nt = 1, r = static_cast<int64_t>(m), nr = static_cast<int64_t>(a % m);
    while (nr) { const int64_t q = r / nr; int64_t x = t - q * nt; t = nt; nt = x; x = r - q * nr; r = nr; nr = x; }
    ENGINE_CHECK_MSG(r == 1, "InvMod: not a unit");
    return static_cast<uint64_t>(t < 0 ? t + static_cast<int64_t>(m) : t);
}
uint32_t BernoulliChiValuation(int64_t d, uint32_t k, uint64_t ell) {
    ENGINE_CHECK_MSG(k <= 32 && ell >= 5 && static_cast<uint64_t>(-d) % ell != 0, "BernoulliChiValuation: k <= 32, ell >= 5, ell !| d");
    const uint64_t f = static_cast<uint64_t>(-d), M = ell * ell * ell * ell;
    auto modM = [&](__int128 x) { x %= static_cast<__int128>(M); return static_cast<uint64_t>(x < 0 ? x + static_cast<__int128>(M) : x); };
    uint64_t T = 0;   // ell B_{k,chi} mod ell^4
    for (uint64_t a = 1; a <= f; ++a) {
        const int chi = Kronecker(d, a);
        if (chi == 0) continue;
        uint64_t inner = 0;
        __int128 binom = 1;   // C(k, j)
        for (uint32_t j = 0; j <= k; ++j) {
            const Frac& b = Bernoulli()[j];
            uint64_t lb;   // ell B_j mod M
            if (b.n == 0) lb = 0;
            else if (b.d % static_cast<__int128>(ell) == 0) lb = modM(b.n) * InvMod(modM(b.d / static_cast<__int128>(ell)), M) % M;
            else lb = modM(b.n * static_cast<__int128>(ell)) * InvMod(modM(b.d), M) % M;
            const uint64_t fj = j == 0 ? InvMod(f % M, M) : PowMod(f, j - 1, M);
            inner = (inner + modM(binom) * lb % M * PowMod(a, k - j, M) % M * fj) % M;
            binom = binom * (k - j) / (j + 1);
        }
        T = (T + (chi > 0 ? inner : (M - inner) % M)) % M;
    }
    uint32_t vT = 0;
    for (uint64_t x = T; vT < 4 && x % ell == 0; x /= ell) { ++vT; if (x == 0) { vT = 4; break; } }
    uint32_t vk = 0;
    for (uint32_t x = k; x % ell == 0; x /= static_cast<uint32_t>(ell)) ++vk;
    const int v = static_cast<int>(vT) - 1 - static_cast<int>(vk);   // v(B_{k,chi}) = v(T) - 1
    return v <= 0 ? 0u : (v >= 2 ? 2u : static_cast<uint32_t>(v));
}

// ---- WP-9's table -------------------------------------------------------------------
bool PathExists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
const char* kTsv = "/docs/implementation/slot16/WP-9_nb_predictions.tsv";
std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    { const std::string r(OC_SOURCE_DIR); if (!r.empty() && PathExists(r + kTsv)) return r; }
#endif
    char buf[4096];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (PathExists(dir + kTsv)) return dir;
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}
std::vector<std::string> Split(const std::string& s, char c) {
    std::vector<std::string> v;
    std::string cur;
    for (char x : s) { if (x == c) { v.push_back(cur); cur.clear(); } else cur.push_back(x); }
    v.push_back(cur);
    return v;
}
std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
std::string Field(const std::string& s, const std::string& key) {   // "key=VALUE ..." -> VALUE
    const size_t at = s.find(key + "=");
    if (at == std::string::npos) return std::string();
    const size_t b = at + key.size() + 1;
    const size_t e = s.find(' ', b);
    return s.substr(b, e == std::string::npos ? std::string::npos : e - b);
}

// The (NB6) cells: N, d, k, the LMFDB Conrey index.
struct Nb6 { uint64_t N; int64_t d; uint32_t k; uint64_t conrey; };
const std::vector<Nb6>& Nb6Cells() {
    static const std::vector<Nb6> v{{7, -7, 3, 6}, {4, -4, 5, 3}, {3, -3, 7, 2}};
    return v;
}

// ---------------------------------------------------------------------------
// (A) arithmetic and the oracles.
// ---------------------------------------------------------------------------
void ArmArithmetic(Checker& g) {
    // h_K against the classical table (Gauss / Watkins), and the nine h = 1 discriminants below 200.
    const std::vector<std::pair<int64_t, uint32_t>> h_table{{-3, 1}, {-4, 1}, {-7, 1}, {-8, 1}, {-11, 1}, {-19, 1}, {-43, 1}, {-67, 1},
                                                            {-163, 1}, {-15, 2}, {-20, 2}, {-24, 2}, {-35, 2}, {-40, 2}, {-51, 2},
                                                            {-52, 2}, {-23, 3}, {-31, 3}, {-39, 4}, {-56, 4}, {-47, 5}, {-71, 7}};
    for (const auto& dh : h_table)
        g.check(ClassNumber(dh.first) == dh.second, "(A) h(" + std::to_string(dh.first) + ") = " + std::to_string(ClassNumber(dh.first)) +
                                                        " (classical " + std::to_string(dh.second) + ")");
    std::vector<int64_t> h1;
    for (int64_t d : NegativeFundamentalDiscriminants(200))
        if (ClassNumber(d) == 1) h1.push_back(d);
    g.check(h1 == std::vector<int64_t>{-3, -4, -7, -8, -11, -19, -43, -67, -163},
            "(A) the class-number-one discriminants below 200 are the nine of Heegner-Baker-Stark");
    g.check(UnitCount(-3) == 6 && UnitCount(-4) == 4 && UnitCount(-7) == 2 && UnitCount(-15) == 2, "(A) w_K at -3, -4, -7, -15");
    // chi_K = the LMFDB Conrey character of each (NB6) cell (WP-9's conv rows), and the unique quadratic Conrey label elsewhere.
    for (const Nb6& c : Nb6Cells()) {
        const Fq F = Fq::Prime(13);
        const CharacterValues a = hecke::neb::ReduceCharacter(KroneckerCharacter(c.d, c.N), F);
        const CharacterValues b = hecke::neb::ReduceCharacter(DirichletCharacter::Conrey(c.N, c.conrey), F);
        g.check(a.val == b.val, "(A) chi_" + std::to_string(c.d) + " = the Conrey character " + std::to_string(c.N) + "." +
                                    std::to_string(c.conrey) + " (WP-9's lmfdb_conrey_label)");
    }
    for (int64_t d : {-3, -4, -7, -8, -11, -15}) {
        const uint64_t N = static_cast<uint64_t>(-d);
        const DirichletCharacter K = KroneckerCharacter(d, N);
        uint32_t matches = 0;
        uint64_t label = 0;
        for (const DirichletCharacter& c : hecke::neb::AllCharacters(N)) {
            if (c.order() != 2) continue;
            bool eq = true;
            for (uint64_t u : hecke::neb::UnitsMod(N)) eq = eq && (2 * c.exponent_at(static_cast<int64_t>(u)) / static_cast<int64_t>(c.L)) ==
                                                                      K.exponent_at(static_cast<int64_t>(u));
            if (eq) { ++matches; label = c.conrey; }
        }
        g.check(matches == 1 && K.parity() == -1, "(A) chi_" + std::to_string(d) + " is odd and the unique quadratic Conrey character " +
                                                      std::to_string(N) + "." + std::to_string(label));
    }
    // The CM oracle against every split / inert nb6_ap row (LMFDB a_p, p <= 50).
    const std::string root = FindRepoRoot();
    g.check(!root.empty(), "(A) the repository root holding " + std::string(kTsv) + " is located");
    uint32_t rows = 0, held = 0, ramified = 0, cmfield = 0, ramified_expected = 0;
    std::set<uint32_t> weights;
    if (!root.empty()) {
        std::ifstream in(root + kTsv);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            const std::vector<std::string> f = Split(line, '\t');
            if (f.size() < 4) continue;
            const Nb6* cell = nullptr;
            for (const Nb6& c : Nb6Cells())
                if (f[0] == CellName(c.N, c.d, c.k)) cell = &c;
            if (cell == nullptr) continue;
            if (f[1] == "nb6" && f[2] == "cm_field") {
                ++cmfield;
                g.check(f[3].find("disc " + std::to_string(cell->d)) != std::string::npos,
                        "(A) " + f[0] + ": WP-9's cm_field row names disc " + std::to_string(cell->d) + " (" + f[3] + ")");
                continue;
            }
            if (f[1] != "nb6_ap") continue;
            const uint64_t p = std::stoull(f[2].substr(2));
            ramified_expected += (static_cast<uint64_t>(-cell->d) % p == 0) ? 1 : 0;   // derived from the row's p, not its type field
            weights.insert(cell->k);
            const std::string type = Field(f[3], "type");
            const std::string lm = Field(f[3], "a_p(LMFDB)");
            if (type != "split" && type != "inert") { ++ramified; continue; }
            ++rows;
            const __int128 a = CmCoefficientExact(cell->d, cell->k, p);
            const bool ok = Splitting(cell->d, p) == type && I128(a) == lm;
            held += ok ? 1 : 0;
            g.check(ok, "(A) " + f[0] + " p=" + std::to_string(p) + " " + type + ": a_p(theta_{lambda_k}) = " + I128(a) + " = LMFDB " + lm);
        }
    }
    // (audit m5) derived, not hard-coded: every split / inert row held, the ramified rows are exactly those with p | d,
    // one cm_field row per (NB6) cell, and every (NB6) cell contributed rows.
    g.check(rows > 0 && held == rows && ramified == ramified_expected && cmfield == Nb6Cells().size() && weights.size() == Nb6Cells().size(),
            "(A) the nb6_ap tally: " + std::to_string(held) + " / " + std::to_string(rows) + " split/inert rows held, " +
                std::to_string(ramified) + " ramified rows skipped (rows with p | d: " + std::to_string(ramified_expected) + "), " +
                std::to_string(cmfield) + " cm_field rows, " + std::to_string(weights.size()) + " weights (expected one per (NB6) cell)");
    std::cout << "  HIDA_CM_ORACLE nb6_ap_rows=" << rows << " held=" << held << " ramified_skipped=" << ramified
              << " weights=" << weights.size() << " (WP-9's table has LMFDB a_p rows only at k = 3, 5, 7)\n";
    // (audit m2) the oracle at the weights that carry (U8) / (U10) (up to 23), where WP-9's table has no LMFDB row:
    // (i) CmCoefficientExact mod m == CmCoefficientMod at every ladder cell, every prime p <= 100 (p !| d), m in the
    //     (U8) ell list and q; (ii) an independent direct evaluation of Tr alpha^{k-1} by powers of alpha = x + y w in
    //     Z[w] (w^2 = d w - (d^2 - d)/4), alpha found here by its own norm search, at (-3, 19) and (-7, 23).
    {
        uint32_t cons = 0, cons_held = 0;
        for (const auto& dk : std::vector<std::pair<int64_t, uint32_t>>{
                 {-4, 5}, {-4, 9}, {-4, 13}, {-4, 17}, {-3, 7}, {-3, 13}, {-3, 19}, {-7, 3}, {-7, 5}, {-7, 7}, {-7, 11}, {-7, 23},
                 {-8, 3}, {-8, 5}, {-8, 11}, {-11, 3}, {-11, 5}}) {
            for (uint64_t p : PrimesUpTo(100)) {
                if (static_cast<uint64_t>(-dk.first) % p == 0) continue;
                const __int128 a = CmCoefficientExact(dk.first, dk.second, p);
                for (uint64_t m : {5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 1000000007ull}) {
                    __int128 r = a % static_cast<__int128>(m);
                    if (r < 0) r += static_cast<__int128>(m);
                    ++cons;
                    cons_held += static_cast<uint64_t>(r) == CmCoefficientMod(dk.first, dk.second, p, m) ? 1 : 0;
                }
            }
        }
        g.check(cons > 0 && cons_held == cons, "(A) CmCoefficientExact mod m == CmCoefficientMod on " + std::to_string(cons_held) + " / " +
                                                   std::to_string(cons) + " (ladder cell, p <= 100, m) triples");
        uint32_t direct = 0, direct_held = 0;
        for (const auto& dk : std::vector<std::pair<int64_t, uint32_t>>{{-3, 19}, {-7, 23}}) {
            const int64_t d = dk.first, c = (d * d - d) / 4;
            for (uint64_t p : PrimesUpTo(100)) {
                if (Kronecker(d, p) != 1) continue;
                // the norm search: N(x + y w) = x^2 + d x y + c y^2 = p, |y| <= 2 sqrt(p / |d|) + 1, |x| <= 2 sqrt(p) + |d y|
                int64_t X = 0, Y = 0;
                bool found = false;
                const int64_t P = static_cast<int64_t>(p);
                for (int64_t y = 1; y * y * (-d) <= 4 * P + 4 && !found; ++y)
                    for (int64_t x = -(2 * P + (-d) * y); x <= 2 * P + (-d) * y; ++x)
                        if (x * x + d * x * y + c * y * y == P) { X = x; Y = y; found = true; break; }
                ++direct;
                if (!found) continue;
                // alpha^{k-1} = A + B w, (A + B w)(x + y w) = A x - c B y + (A y + B x + d B y) w
                __int128 A = 1, B = 0;
                for (uint32_t j = 0; j + 1 < dk.second; ++j) {
                    const __int128 nA = A * X - static_cast<__int128>(c) * B * Y, nB = A * Y + B * X + static_cast<__int128>(d) * B * Y;
                    A = nA;
                    B = nB;
                }
                const __int128 tr = 2 * A + static_cast<__int128>(d) * B;   // Tr w = d
                direct_held += tr == CmCoefficientExact(d, dk.second, p) ? 1 : 0;
            }
        }
        g.check(direct > 0 && direct_held == direct, "(A) Tr alpha^{k-1} by powers of alpha in Z[w] == the trace recurrence at (-3, 19), (-7, 23), "
                                                     "every split p <= 100: " + std::to_string(direct_held) + " / " + std::to_string(direct));
        std::cout << "  HIDA_CM_ORACLE_HIGHK consistency=" << cons_held << "/" << cons << " direct_Zw=" << direct_held << "/" << direct << "\n";
    }
    // the Bernoulli numbers behind the (U8/Eis) control: B_1, B_12, B_22, B_32 against their classical values, B_odd>1 = 0
    {
        const std::vector<Frac>& B = Bernoulli();
        bool odd0 = true;
        for (size_t j = 3; j < B.size(); j += 2) odd0 = odd0 && B[j].n == 0;
        g.check(B[1].n == -1 && B[1].d == 2 && B[12].n == -691 && B[12].d == 2730 && B[22].n == 854513 && B[22].d == 138 &&
                    B[32].n == static_cast<__int128>(-7709321041217LL) && B[32].d == 510 && odd0,
                "(A) the Bernoulli recurrence: B_1 = -1/2, B_12 = -691/2730, B_22 = 854513/138, B_32 = -7709321041217/510, B_odd = 0");
    }
    // h_K = 1: every split p <= 200 is principal at the h = 1 fields; genus theory at d = -15 (h = 2): a split p is
    // principal iff chi_-3(p) = +1; lambda_k is refused at a non-principal split prime (prop:u-cmeis(2)'s hypothesis).
    for (int64_t d : {-3, -4, -7, -8, -11}) {
        bool all = true;
        for (uint64_t p : PrimesUpTo(200)) {
            int64_t t = 0;
            if (Kronecker(d, p) == 1) all = all && PrincipalPrimeTrace(d, p, t);
        }
        g.check(all, "(A) h_K = 1 at d = " + std::to_string(d) + ": every split p <= 200 is the norm of an element");
    }
    bool genus = true;
    uint32_t nonprincipal = 0;
    for (uint64_t p : PrimesUpTo(200)) {
        if (Kronecker(-15, p) != 1) continue;
        int64_t t = 0;
        const bool pr = PrincipalPrimeTrace(-15, p, t);
        genus = genus && (pr == (Kronecker(-3, p) == 1));
        nonprincipal += pr ? 0 : 1;
    }
    g.check(genus && nonprincipal > 0, "(A) genus theory at d = -15: a split p <= 200 is principal iff chi_-3(p) = 1 (" +
                                           std::to_string(nonprincipal) + " non-principal split primes)");
    bool refused = false;
    try { (void)CmCoefficientMod(-15, 17, 2, 17); } catch (const std::runtime_error& e) { refused = std::string(e.what()).find("cm_nonprincipal") == 0; }
    g.check(refused, "(A) the CM oracle refuses lambda_k at the non-principal split prime 2 of Q(sqrt -15) by name (cm_nonprincipal)");
    refused = false;
    try { (void)CmCoefficientMod(-4, 7, 5, 13); } catch (const std::runtime_error& e) { refused = std::string(e.what()).find("cm_lambda_undefined") == 0; }
    g.check(refused, "(A) the CM oracle refuses lambda_7 on Q(i) by name (w_K = 4 does not divide 6: cm_lambda_undefined)");
}

// ---------------------------------------------------------------------------
// (C0) characteristic-0 CM classes, and (U9) at tame level 1 and 2.
// ---------------------------------------------------------------------------
struct LadderCell { int64_t d; uint32_t k; };
const std::vector<LadderCell>& C0Cells() {
    static const std::vector<LadderCell> v{
        {-4, 3}, {-4, 5}, {-4, 7}, {-4, 9}, {-4, 13}, {-4, 17}, {-3, 3}, {-3, 5}, {-3, 7}, {-3, 9}, {-3, 13}, {-3, 19},
        {-7, 3}, {-7, 5}, {-7, 7}, {-7, 11}, {-7, 23}, {-8, 3}, {-8, 5}, {-8, 11}, {-11, 3}, {-11, 5}, {-15, 3}, {-15, 17}, {-15, 19}};
    return v;
}
void ArmCharZero(Checker& g) {
    uint32_t cells = 0, cm_cells = 0, cm_cells_expected = 0;
    for (const LadderCell& lc : C0Cells()) cm_cells_expected += (lc.k - 1) % UnitCount(lc.d) == 0 ? 1 : 0;
    for (const LadderCell& lc : C0Cells()) {
        const uint64_t N = static_cast<uint64_t>(-lc.d);
        std::unique_ptr<CellModel> c = MakeCell(N, lc.d, lc.k, kQ);
        g.check(c->refusal.empty(), "(C0) " + CellName(N, lc.d, lc.k) + " over F_q: built (" + c->refusal + ")");
        if (!c->refusal.empty()) continue;
        const uint32_t h = ClassNumber(lc.d), w = UnitCount(lc.d);
        const uint32_t want = (lc.k - 1) % w == 0 ? 2 * h : 0;
        const uint32_t got = InertKernelDim(*c, lc.d, kB0, false);
        ++cells;
        cm_cells += want > 0 ? 1 : 0;
        g.check(got == want, "(C0) " + CellName(N, lc.d, lc.k) + ": dim of the joint kernel of T_p (p <= 50 inert) over F_q = " +
                                 std::to_string(got) + " = 2 h_K [w_K | k - 1] = " + std::to_string(want) + " (h1 = " + std::to_string(c->h1()) + ")");
        std::cout << "  HIDA_CM_CHAR0 cell=" << CellName(N, lc.d, lc.k) << " h_K=" << h << " w_K=" << w << " h1=" << c->h1()
                  << " cm_classes=" << got << " predicted=" << want << "\n";
    }
    g.check(cells == C0Cells().size() && cm_cells == cm_cells_expected && cm_cells > 0,
            "(C0) every ladder cell ran (" + std::to_string(cells) + ", of which " + std::to_string(cm_cells) +
                " carry CM; expected " + std::to_string(cm_cells_expected) + " = #{cells with w_K | k - 1})");
    // (U9) tame level 1 and 2: no CM by any K with |d_K| <= 100 (thm:u-cmN (3)).
    const std::vector<int64_t> ds = NegativeFundamentalDiscriminants(100);
    uint32_t tested = 0, nonzero = 0, tested_expected = 0, u9_cells = 0;
    const std::vector<std::pair<uint64_t, std::vector<uint32_t>>> grid{{1, {12, 16, 18, 20, 22, 24, 26, 72}}, {2, {8, 10, 12, 14, 16}}};
    // (audit m5) the expected number of (cell, K) pairs: K with at least one inert p <= 50, p !| N (p = q never occurs)
    for (const auto& Nk : grid)
        for (size_t i = 0; i < Nk.second.size(); ++i) {
            ++u9_cells;
            for (int64_t d : ds) {
                bool any = false;
                for (uint64_t p : PrimesUpTo(kB0)) any = any || (Nk.first % p != 0 && Kronecker(d, p) == -1);
                tested_expected += any ? 1 : 0;
            }
        }
    for (const auto& Nk : grid)
        for (uint32_t k : Nk.second) {
            std::unique_ptr<CellModel> c = MakeCell(Nk.first, 0, k, kQ);
            g.check(c->refusal.empty() && c->h1() > 0, "(U9) (" + std::to_string(Nk.first) + ", 1, " + std::to_string(k) + ") over F_q: h1 > 0");
            if (!c->refusal.empty()) continue;
            uint32_t here = 0;
            for (int64_t d : ds) {
                const uint32_t got = InertKernelDim(*c, d, kB0, false);
                if (got == UINT32_MAX) continue;
                ++tested;
                here += got;
                nonzero += got > 0 ? 1 : 0;
                if (got != 0)
                    g.check(false, "(U9) (" + std::to_string(Nk.first) + ", 1, " + std::to_string(k) + "): CM classes by Q(sqrt " +
                                       std::to_string(d) + ") in characteristic 0 (" + std::to_string(got) + ")");
            }
            std::cout << "  HIDA_CM_U9 N=" << Nk.first << " k=" << k << " h1=" << c->h1() << " fields=" << ds.size()
                      << " cm_classes_total=" << here << "\n";
        }
    std::cout << "  HIDA_CM_U9_SUMMARY cells=" << u9_cells << " fields=" << ds.size() << " pairs_tested=" << tested << " pairs_with_cm=" << nonzero << "\n";
    g.check(nonzero == 0 && tested == tested_expected && tested > 0,
            "(U9) no CM at tame level 1 or 2: " + std::to_string(tested) + " (cell, K) pairs (expected " + std::to_string(tested_expected) +
                "), " + std::to_string(nonzero) + " with a CM class (thm:u-cmN (3))");
}

// ---------------------------------------------------------------------------
// (U9b) residually dihedral blocks at tame level 1 and 2, over F_ell.
// ---------------------------------------------------------------------------
void ArmResiduallyDihedral(Checker& g) {
    const std::vector<int64_t> ds = NegativeFundamentalDiscriminants(100);
    const std::vector<uint64_t> ells{5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    std::vector<std::tuple<uint64_t, uint32_t, std::vector<uint64_t>>> grid{
        {1, 12, ells}, {1, 16, ells}, {1, 18, ells}, {1, 20, ells}, {1, 22, ells}, {1, 24, ells}, {1, 26, ells}, {1, 72, {47}},
        {2, 8, ells}, {2, 10, ells}, {2, 12, ells}, {2, 14, ells}, {2, 16, ells}};
    uint32_t found = 0, found_ord = 0, runs = 0, shadowed = 0, shadow_explained = 0, reconfirmed = 0;
    bool control_ord = false, control_nonord = false;
    for (const auto& cell : grid) {
        const uint64_t N = std::get<0>(cell);
        const uint32_t k = std::get<1>(cell);
        for (uint64_t ell : std::get<2>(cell)) {
            std::unique_ptr<CellModel> c = MakeCell(N, 0, k, ell);
            if (!c->refusal.empty() || c->h1() == 0) continue;
            ++runs;
            const FqMat& Tl = c->Op(ell);
            std::vector<int64_t> confirmed_here, shadowed_here;
            for (int64_t d : ds) {
                // a first pass on p <= 50, confirmed on p <= 100 (the T_p are built lazily, so only where the first pass fires)
                auto inert_space = [&](uint64_t B) {
                    std::vector<FqMat> ops;
                    for (uint64_t p : PrimesUpTo(B))
                        if (N % p != 0 && p != ell && Kronecker(d, p) == -1) ops.push_back(c->Op(p));
                    if (ops.size() < 3) return FqMat(0, 0);
                    return JointGeneralisedEigenspace(*c->F, ops, std::vector<uint32_t>(ops.size(), 0));
                };
                if (inert_space(kB0).cols == 0) continue;
                const FqMat V = inert_space(kBd);
                if (V.cols == 0) { ++shadowed; shadowed_here.push_back(d); continue; }
                const uint32_t ord = V.cols - GeneralisedEigenDim(*c->F, Restrict(*c->F, Tl, V), 0);
                ++found;
                confirmed_here.push_back(d);
                // (audit m3) a confirmed detection must be stable: re-read on p <= 150, the dimension must not move
                const uint32_t v150 = inert_space(kBd2).cols;
                reconfirmed += v150 == V.cols ? 1 : 0;
                g.check(v150 == V.cols, "(U9b) (" + std::to_string(N) + ", 1, " + std::to_string(k) + ") ell=" + std::to_string(ell) +
                                            " d=" + std::to_string(d) + ": the confirmed block is unchanged on p <= 150 (" +
                                            std::to_string(V.cols) + " -> " + std::to_string(v150) + ")");
                if (N == 1)   // rmk:u-dihedral(i): at N = 1, K = Q(sqrt -ell) exactly
                    g.check(d == -static_cast<int64_t>(ell), "(U9b) (1, 1, " + std::to_string(k) + ") ell=" + std::to_string(ell) +
                                                                 ": residually dihedral by Q(sqrt " + std::to_string(d) + "), d_K = -ell");
                found_ord += ord > 0 ? 1 : 0;
                std::set<uint64_t> allowed = PrimeSupport(N);
                allowed.insert(ell);
                bool unram = true;
                for (uint64_t q : PrimeSupport(static_cast<uint64_t>(-d))) unram = unram && allowed.count(q) > 0;
                g.check(unram, "(U9b) (" + std::to_string(N) + ", 1, " + std::to_string(k) + ") ell=" + std::to_string(ell) +
                                   ": residually dihedral by Q(sqrt " + std::to_string(d) + ") -- K must be unramified outside N ell");
                // (audit m3) the controls pinned at their observed dimensions: 5 ordinary at (1, 24, 47), 10 non-ordinary at (1, 72, 47)
                if (N == 1 && k == 24 && ell == 47 && d == -47 && V.cols == 5 && ord == 5) control_ord = true;
                if (N == 1 && k == 72 && ell == 47 && d == -47 && V.cols == 10 && ord == 0) control_nonord = true;
                std::cout << "  HIDA_CM_RESDIH N=" << N << " k=" << k << " ell=" << ell << " d=" << d << " dim=" << V.cols
                          << " ord_dim=" << ord << " cm_in_char0=no (C0/U9)\n";
            }
            // (audit m3) a detection shadowed at p <= 50 only must be explained by a confirmed block at the same (N, k, ell)
            for (int64_t d : shadowed_here) {
                std::string conf;
                for (int64_t e : confirmed_here) conf += (conf.empty() ? "" : ",") + std::to_string(e);
                shadow_explained += confirmed_here.empty() ? 0 : 1;
                g.check(!confirmed_here.empty(), "(U9b) shadowed detection (N, k, ell, d) = (" + std::to_string(N) + ", " + std::to_string(k) +
                                                     ", " + std::to_string(ell) + ", " + std::to_string(d) +
                                                     ") has a confirmed block at the same (N, k, ell)");
                std::cout << "  HIDA_CM_RESDIH_SHADOW N=" << N << " k=" << k << " ell=" << ell << " d=" << d << " confirmed_d=["
                          << conf << "]\n";
            }
        }
    }
    g.check(control_ord, "(U9b) control: (1, 24, 47) has an ordinary residually dihedral block by Q(sqrt -47) (WP-11's J-7 control)");
    g.check(control_nonord, "(U9b) control: (1, 72, 47) has a non-ordinary residually dihedral block of dimension 10 by Q(sqrt -47)");
    std::cout << "  HIDA_CM_RESDIH_SUMMARY runs=" << runs << " found=" << found << " ordinary=" << found_ord << " shadowed_at_p_le_50_only=" << shadowed
              << " shadow_explained=" << shadow_explained << " reconfirmed_p_le_150=" << reconfirmed << "\n";
}

// ---------------------------------------------------------------------------
// (U8) on the CM ladders.
// ---------------------------------------------------------------------------
void ArmU8(Checker& g) {
    const std::vector<uint64_t> ells{5, 7, 11, 13, 17, 19, 23, 29, 31};
    uint32_t runs = 0, split = 0, inert = 0, excess_cells = 0, exact = 0, runs_expected = 0, eis_ctrl = 0, eis_ctrl_held = 0;
    uint32_t eis_cong = 0, eis_ctrl_reported = 0;
    // (audit m5) the expected number of runs, derived from the grid
    for (const LadderCell& lc : C0Cells()) {
        if ((lc.k - 1) % UnitCount(lc.d) != 0 || ClassNumber(lc.d) != 1) continue;
        for (uint64_t ell : ells) runs_expected += static_cast<uint64_t>(-lc.d) % ell != 0 ? 1 : 0;
    }
    for (const LadderCell& lc : C0Cells()) {
        if ((lc.k - 1) % UnitCount(lc.d) != 0 || ClassNumber(lc.d) != 1) continue;
        const uint64_t N = static_cast<uint64_t>(-lc.d);
        for (uint64_t ell : ells) {
            if (N % ell == 0) continue;
            std::unique_ptr<CellModel> c = MakeCell(N, lc.d, lc.k, ell);
            g.check(c->refusal.empty() && c->h1() > 0, "(U8) " + CellName(N, lc.d, lc.k) + " ell=" + std::to_string(ell) + ": built, h1 > 0");
            if (!c->refusal.empty() || c->h1() == 0) continue;
            const CmReading R = ReadCm(*c, PrimeBound(N, lc.k));
            const uint32_t ex = GateU8(g, *c, R, "(U8)");
            PrintU8(*c, R, ex);
            // (audit M1 (c)) the Eisenstein-multiplicity control: where the CM system differs from both Eisenstein systems,
            // each Eisenstein system carries exactly one class of H^1 per system at its eigenvalue vector (two when the
            // two systems coincide mod ell) -- the multiplicity the (U8) / (U10) subtraction of R.eis classes assumes.
            // Gated where n = k - 2 < ell (Sym^n F_ell irreducible, the regime of every (U10) cell): dim W(E) = the number
            // of Eisenstein systems at that vector exactly when ell !| B_{k,chi} / k, and larger by an even number (a
            // cuspidal pair) when ell | B_{k,chi} / k.  At n >= ell the n >= ell torsion can enlarge W: reported, >= gated.
            if (R.eis == 0) {
                const EisDims E = EisensteinDims(*c, R.pmax);
                const uint32_t want = E.same ? 2u : 1u;
                const bool small_n = lc.k - 2 < ell;
                const uint32_t vB = BernoulliChiValuation(lc.d, lc.k, ell);
                const bool both_ge = E.w1 >= want && E.w2 >= want;
                bool ok = both_ge;
                if (small_n) {
                    const bool exact = E.w1 == want && E.w2 == want;
                    const bool cong_shape = E.w1 > want && E.w2 > want && (E.w1 - want) % 2 == 0 && (E.w2 - want) % 2 == 0;
                    ok = vB == 0 ? exact : cong_shape;
                    ++eis_ctrl;
                    eis_ctrl_held += ok ? 1 : 0;
                    eis_cong += vB > 0 ? 1 : 0;
                } else {
                    ++eis_ctrl_reported;
                }
                g.check(ok, "(U8/Eis) " + CellName(N, lc.d, lc.k) + " ell=" + std::to_string(ell) + ": dim W(E(1,chi)) = " +
                                std::to_string(E.w1) + ", dim W(E(chi,1)) = " + std::to_string(E.w2) + ", #Eisenstein systems = " +
                                std::to_string(want) + ", v_ell(B_{k,chi}/k) = " + std::to_string(vB) +
                                (small_n ? (vB == 0 ? ": exactly one class per system (no Eisenstein-cusp congruence)"
                                                    : ": a cuspidal pair on top (ell | B_{k,chi}/k)")
                                         : ": n >= ell, >= only"));
                std::cout << "  HIDA_CM_EISMULT cell=" << CellName(N, lc.d, lc.k) << " ell=" << ell << " W(E(1,chi))=" << E.w1
                          << " W(E(chi,1))=" << E.w2 << " systems_coincide=" << (E.same ? 1 : 0) << " eis_systems=" << want
                          << " v_ell(B_k_chi/k)=" << vB << " regime=" << (small_n ? "n<ell(gated)" : "n>=ell(reported)") << "\n";
            }
            ++runs;
            (R.split ? split : inert) += 1;
            excess_cells += ex > 0 ? 1 : 0;
            exact += R.dimW == 2 ? 1 : 0;
        }
    }
    g.check(runs == runs_expected && split > 0 && inert > 0, "(U8) " + std::to_string(runs) + " (cell, ell) runs (expected " +
                                                                 std::to_string(runs_expected) + "), " + std::to_string(split) +
                                                                 " split, " + std::to_string(inert) + " inert");
    g.check(eis_ctrl > 0 && eis_ctrl_held == eis_ctrl && eis_cong > 0,
            "(U8/Eis) the Eisenstein-multiplicity control held at " + std::to_string(eis_ctrl_held) + " / " + std::to_string(eis_ctrl) +
                " gated runs (n < ell), " + std::to_string(eis_cong) + " of them with ell | B_{k,chi}/k (the criterion is exercised both ways)");
    std::cout << "  HIDA_CM_U8_SUMMARY runs=" << runs << " split=" << split << " inert=" << inert << " exact_dimW2=" << exact
              << " cm_gl2_excess_cells=" << excess_cells << " eis_multiplicity_control=" << eis_ctrl_held << "/" << eis_ctrl
              << " eis_cusp_congruences=" << eis_cong << " eis_control_reported_n_ge_ell=" << eis_ctrl_reported << "\n";
}

// ---------------------------------------------------------------------------
// (U10) CM cap Eisenstein; the h_K = 2 negative example.
// ---------------------------------------------------------------------------
// excess = the pinned non-CM excess of W (audit M1 (b)): 0 at eight cells, 4 at (3, chi_-3, 13) ell = 13 and
// (11, chi_-11, 5) ell = 5 (the rmk:u-cmgl2 readings the report records), so dim W == 2 Eisenstein + 2 CM + excess.
struct U10Cell { uint64_t N; int64_t d; uint32_t k; uint64_t ell; uint32_t excess; };
void ArmU10(Checker& g) {
    const std::vector<U10Cell> cells{{4, -4, 5, 5, 0},   {3, -3, 7, 7, 0},   {7, -7, 11, 11, 0}, {4, -4, 13, 13, 0}, {4, -4, 17, 17, 0},
                                     {3, -3, 13, 13, 4}, {3, -3, 19, 19, 0}, {7, -7, 23, 23, 0}, {8, -8, 11, 11, 0}, {11, -11, 5, 5, 4}};
    uint32_t held = 0, conj_held = 0;
    for (const U10Cell& u : cells) {
        const std::string at = "(U10) " + CellName(u.N, u.d, u.k) + " ell=" + std::to_string(u.ell);
        const uint32_t h = ClassNumber(u.d);
        g.check(h == 1 && (u.k - 1) % UnitCount(u.d) == 0 && (u.k - 1) % (u.ell - 1) == 0 && Kronecker(u.d, u.ell) == 1,
                at + ": h_K = 1 (prop:u-cmeis(2)'s hypothesis), w_K | k - 1, (ell - 1) | (k - 1), ell split");
        // the congruence on integers, p <= 50
        bool cong = true;
        for (uint64_t p : PrimesUpTo(kB0)) {
            if (u.N % p == 0 || p == u.ell) continue;
            const __int128 a = CmCoefficientExact(u.d, u.k, p);
            const int64_t chi = Kronecker(u.d, p);
            __int128 pk = 1;
            for (uint32_t j = 0; j + 1 < u.k; ++j) pk = pk * static_cast<__int128>(p) % static_cast<__int128>(u.ell);
            const __int128 L = static_cast<__int128>(u.ell);
            auto md = [&](__int128 x) { x %= L; return x < 0 ? x + L : x; };
            cong = cong && md(a) == md(1 + chi * pk) && md(a) == md(chi + pk);
        }
        g.check(cong, at + ": a_p(theta) = 1 + chi(p) p^{k-1} = chi(p) + p^{k-1} mod ell at every p <= 50 (exact integers)");
        std::unique_ptr<CellModel> c = MakeCell(u.N, u.d, u.k, u.ell);
        g.check(c->refusal.empty() && c->h1() > 0, at + ": built");
        if (!c->refusal.empty() || c->h1() == 0) continue;
        const CmReading R = ReadCm(*c, PrimeBound(u.N, u.k));
        const uint32_t ex = GateU8(g, *c, R, "(U10/U8)");
        // (audit M1 (b)) pinned, not a lower bound: the two Eisenstein classes (one per system, the (U8/Eis) control) are
        // subtracted, and what is left is exactly the two CM classes plus the pinned excess, so a stray non-CM class
        // cannot stand in for a CM class.
        const bool ok = R.eis == 2 && R.dimW == 2 + 2 + u.excess && ex == u.excess;
        g.check(ok, at + ": both Eisenstein systems coincide with the CM system and dim W = " + std::to_string(R.dimW) + " == 2 Eisenstein + "
                        "2 CM + " + std::to_string(u.excess) + " pinned excess (cm_gl2_excess = " + std::to_string(ex) + ")");
        held += ok ? 1 : 0;
        std::vector<FqMat> nil;
        for (size_t i = 0; i < R.primes.size(); ++i) nil.push_back(Shift(*c->F, Restrict(*c->F, c->Op(R.primes[i]), R.W), R.e1[i]));
        const LoewyReading L = LoewyOfNilpotents(*c->F, nil, R.dimW);
        uint32_t layer_sum = 0;
        for (uint32_t x : L.loewy_layers) layer_sum += x;
        g.check(layer_sum == L.dim_A && L.loewy_layers.size() == L.loewy && !L.loewy_layers.empty() && L.loewy_layers[0] == 1,
                at + ": the Loewy layers " + VecText(L.loewy_layers) + " sum to dim A = " + std::to_string(L.dim_A) + " (A local, A/rad = F)");
        conj_held += L.loewy >= 2 ? 1 : 0;
        std::cout << "  HIDA_CM_U10 cell=" << CellName(u.N, u.d, u.k) << " ell=" << u.ell << " h_K=" << h << " h1=" << R.h1
                  << " W=" << R.dimW << " eis=" << R.eis << " cm_gl2_excess=" << ex << " sturm=" << R.sturm << " p_max=" << R.pmax
                  << " loewy=" << L.loewy << " dim_A=" << L.dim_A << " loewy_layers=" << VecText(L.loewy_layers)
                  << " rad_dims=" << VecText(L.radical_power_dims) << " monomial_degree_dims=" << VecText(L.monomial_degree_dims)
                  << " conj_nonreduced=" << (L.loewy >= 2 ? "held" : "FINDING") << "\n";
    }
    g.check(held == cells.size(), "(U10) held at " + std::to_string(held) + " / " + std::to_string(cells.size()) + " cells");
    std::cout << "  HIDA_CM_U10_SUMMARY cells=" << cells.size() << " proved_held=" << held << " conj_nonreduced_held=" << conj_held << "\n";

    // The negative example, h_K = 2 (WP-14 referee U-1): d = -15, N = 15.
    for (uint64_t ell : {17ull, 19ull}) {
        const uint32_t k = static_cast<uint32_t>(ell);
        const std::string at = "(U10neg) " + CellName(15, -15, k) + " ell=" + std::to_string(ell);
        g.check(ClassNumber(-15) == 2 && Kronecker(-15, ell) == 1 && (k - 1) % (ell - 1) == 0, at + ": h_K = 2, ell split, (ell - 1) | (k - 1)");
        std::unique_ptr<CellModel> c = MakeCell(15, -15, k, ell);
        g.check(c->refusal.empty() && c->h1() > 0, at + ": built");
        if (!c->refusal.empty() || c->h1() == 0) continue;
        const Fq& F = *c->F;
        std::vector<FqMat> ops;
        std::vector<uint32_t> s1, s2;
        uint32_t differ = 0;
        const uint64_t pmax = PrimeBound(15, k), sturm = SturmBound(15, k);   // (audit M2) Sturm 34 at k = 17, 38 at k = 19
        for (uint64_t p : PrimesUpTo(pmax)) {
            if (15 % p == 0 || p == ell) continue;
            const uint32_t pk1 = F.pow(F.from_int(static_cast<int64_t>(p)), k - 1);
            const uint32_t e1 = F.add(1, F.mul(c->chi_at(p), pk1));
            const uint32_t m3 = F.from_int(Kronecker(-3, p)), m5 = F.from_int(Kronecker(5, p));
            const uint32_t e2 = F.add(m3, F.mul(m5, pk1));   // E_k(chi_-3, chi_5): chi_-3 chi_5 = chi_-15
            differ += e1 != e2 ? 1 : 0;
            ops.push_back(c->Op(p));
            s1.push_back(e1);
            s2.push_back(e2);
        }
        const uint32_t w1 = JointGeneralisedEigenspace(F, ops, s1).cols, w2 = JointGeneralisedEigenspace(F, ops, s2).cols;
        g.check(differ > 0, at + ": E_k(1, chi_K) and E_k(chi_-3, chi_5) are distinct systems mod ell (" + std::to_string(differ) + " primes)");
        // (audit m4) == 4, not >= 4: two Eisenstein classes (E_k(psi_1, psi_2) and its swap coincide mod ell at p^{k-1} = 1)
        // plus exactly one CM form (two classes) per system.
        g.check(w1 == 4 && w2 == 4, at + ": the two conductor-1 CM forms split one per Eisenstein system: dim W(E(1,chi_K)) = " +
                                        std::to_string(w1) + " == 4 and dim W(E(chi_-3,chi_5)) = " + std::to_string(w2) +
                                        " == 4 (prop:u-cmeis(1) with mu = chi_-3; the h_K = 1 reading of (2) would put both at E(1, chi_K))");
        std::cout << "  HIDA_CM_U10NEG cell=" << CellName(15, -15, k) << " ell=" << ell << " h_K=2 h1=" << c->h1()
                  << " W(E(1,chi_K))=" << w1 << " W(E(chi_-3,chi_5))=" << w2 << " distinct_primes=" << differ
                  << " sturm=" << sturm << " p_max=" << pmax
                  << " reading=" << (w1 == 4 && w2 == 4 ? "U-1(split between E(1,chi_K) and E(mu,mu chi_K))" : "FINDING") << "\n";
    }
}

// ---------------------------------------------------------------------------
// (ENV) HECKE_HIDA_CM_CELLS.
// ---------------------------------------------------------------------------
struct HookCell { uint64_t N; int64_t d; uint32_t k; uint64_t ell; };
std::vector<HookCell> ParseHook(const std::string& s) {
    if (Trim(s).empty()) throw std::runtime_error("HECKE_HIDA_CM_CELLS is set but empty (expected \"N:d:k:ell,...\")");
    std::vector<HookCell> out;
    size_t pos = 0;
    for (const std::string& raw : Split(s, ',')) {
        ++pos;
        const std::string t = Trim(raw);
        const std::vector<std::string> f = Split(t, ':');
        auto bad = [&](const std::string& why) {
            return std::runtime_error("HECKE_HIDA_CM_CELLS cell " + std::to_string(pos) + " '" + t + "': " + why);
        };
        if (f.size() != 4) throw bad("expected N:d:k:ell");
        HookCell c{};
        try {
            size_t used = 0;
            const long long N = std::stoll(f[0], &used); if (used != f[0].size()) throw bad("N is not an integer");
            const long long d = std::stoll(f[1], &used); if (used != f[1].size()) throw bad("d is not an integer");
            const long long k = std::stoll(f[2], &used); if (used != f[2].size()) throw bad("k is not an integer");
            const long long l = std::stoll(f[3], &used); if (used != f[3].size()) throw bad("ell is not an integer");
            if (N < 3 || N >= (1ll << 20)) throw bad("N must be in [3, 2^20)");
            if (d >= 0 || !IsFundamentalDiscriminant(d)) throw bad("d must be a negative fundamental discriminant");
            if (N % (-d) != 0) throw bad("|d| must divide N (thm:u-cmN (3): CM by K needs |d_K| | N)");
            if (k < 2 || k > 4096) throw bad("k must be in [2, 4096]");
            if (l < 5 || !IsPrime(static_cast<uint64_t>(l)) || l >= (1ll << 20)) throw bad("ell must be a prime in [5, 2^20)");
            if (N % l == 0) throw bad("ell must not divide N (tame level)");
            c = HookCell{static_cast<uint64_t>(N), d, static_cast<uint32_t>(k), static_cast<uint64_t>(l)};
        } catch (const std::invalid_argument&) {
            throw bad("not an integer field");
        } catch (const std::out_of_range&) {
            throw bad("a field is out of range");
        }
        out.push_back(c);
    }
    return out;
}
void ArmHook(Checker& g, const std::vector<HookCell>& cells) {
    std::cout << "  EFFECTIVE HeckeHidaCMValidation arm=CELLS cells=" << cells.size() << " (HECKE_HIDA_CM_CELLS; replaces the default arms)\n";
    uint32_t ran = 0, refused = 0, reported_only = 0;
    for (const HookCell& hc : cells) {
        const std::string at = "(ENV) " + CellName(hc.N, hc.d, hc.k) + " ell=" + std::to_string(hc.ell);
        // (audit M2) the mod-ell systems are read on T_p, p <= p_max; p_max is raised to the Sturm bound when it is at most
        // kHookPmaxCap, and beyond the cap the (U8) / (U10) readings are reported only (never gated on an insufficient
        // prime set: past the bound W can be larger than the system and a false Eisenstein coincidence is possible).
        const uint64_t sturm = SturmBound(hc.N, hc.k);
        const bool gated = sturm <= kHookPmaxCap;
        const uint64_t pmax = gated ? PrimeBound(hc.N, hc.k) : kBl;
        Checker scratch;
        scratch.quiet = true;
        Checker& gm = gated ? g : scratch;
        const uint32_t h = ClassNumber(hc.d), w = UnitCount(hc.d);
        const bool lam = (hc.k - 1) % w == 0;
        std::unique_ptr<CellModel> q = MakeCell(hc.N, hc.d, hc.k, kQ);
        if (!q->refusal.empty()) {
            ++refused;
            std::cout << "  HIDA_CM_CELL cell=" << CellName(hc.N, hc.d, hc.k) << " ell=" << hc.ell << " refused=" << q->refusal << "\n";
            continue;
        }
        const uint32_t cm0 = InertKernelDim(*q, hc.d, kB0, false);
        if (hc.N == static_cast<uint64_t>(-hc.d))
            g.check(cm0 == (lam ? 2 * h : 0), at + ": characteristic-0 CM classes " + std::to_string(cm0) + " = 2 h_K [w_K | k - 1]");
        std::ostringstream os;
        os << "  HIDA_CM_CELL cell=" << CellName(hc.N, hc.d, hc.k) << " ell=" << hc.ell << " " << Splitting(hc.d, hc.ell) << " h_K=" << h
           << " w_K=" << w << " h1=" << q->h1() << " cm_classes_char0=" << cm0 << " sturm_bound=" << sturm << " p_max=" << pmax
           << " gates=" << (gated ? "gated" : "reported");
        if (lam && h == 1 && cm0 >= 2) {
            std::unique_ptr<CellModel> c = MakeCell(hc.N, hc.d, hc.k, hc.ell);
            if (c->refusal.empty() && c->h1() > 0) {
                const CmReading R = ReadCm(*c, pmax);
                const uint32_t ex = GateU8(gm, *c, R, "(ENV/U8)");
                os << " W=" << R.dimW << " eis=" << R.eis;
                if (hc.N == static_cast<uint64_t>(-hc.d)) os << " cm_gl2_excess=" << ex;
                else   // N != |d|: W also holds the CM oldform copies; the excess over the char-0 CM count is reported instead
                    os << " cm_gl2_excess=n/a(oldforms) W_minus_cm0_minus_eis="
                       << (R.dimW >= cm0 + R.eis ? std::to_string(R.dimW - cm0 - R.eis) : std::string("neg"));
                os << " a_ell=" << R.al << " ord_W=" << R.ordW
                   << " cm=" << (R.split ? "ordinary" : "non-ordinary");
                if ((hc.k - 1) % (hc.ell - 1) == 0 && R.split) {
                    if (hc.N == static_cast<uint64_t>(-hc.d))   // (audit M1) the two Eisenstein classes subtracted
                        gm.check(R.eis == 2 && R.dimW >= R.eis + 2, at + ": (U10) both Eisenstein systems coincide and dim W - #Eis = " +
                                                                        std::to_string(R.dimW - std::min(R.dimW, R.eis)) + " >= 2 CM classes");
                    std::vector<FqMat> nil;
                    for (size_t i = 0; i < R.primes.size(); ++i)
                        nil.push_back(Shift(*c->F, Restrict(*c->F, c->Op(R.primes[i]), R.W), R.e1[i]));
                    os << " u10_loewy=" << LoewyOfNilpotents(*c->F, nil, R.dimW).loewy;
                }
            } else {
                os << " ell_refused=" << c->refusal;
            }
        } else {
            os << " u8=" << (lam ? (h == 1 ? "no_cm_classes" : "oracle_unavailable(h_K>1)") : "lambda_k_undefined");
        }
        if (!gated) {
            ++reported_only;
            os << " reported_checks=" << scratch.checks << " reported_fails=" << scratch.fails;
        }
        std::cout << os.str() << "\n";
        ++ran;
    }
    std::cout << "  HIDA_CM_CELLS_SUMMARY cells=" << cells.size() << " ran=" << ran << " refused=" << refused
              << " reported_only_beyond_sturm=" << reported_only << "\n";
}

} // namespace

uint32_t HeckeHidaCMValidation() {
    std::cout << "=== HeckeHidaCMValidation ([U] sec:u-tameN: (U8)-(U10) of rmk:u-pred2 on the character model of "
                 "[C] sec:nebentypus; slot-16 WP-15) ===" << std::endl;
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    Checker g;
    if (const char* env = std::getenv("HECKE_HIDA_CM_CELLS")) {
        ArmHook(g, ParseHook(env));
    } else {
        ArmArithmetic(g);
        std::cout << "  [(A) done at " << secs() << " s]" << std::endl;
        ArmCharZero(g);
        std::cout << "  [(C0)/(U9) done at " << secs() << " s]" << std::endl;
        ArmResiduallyDihedral(g);
        std::cout << "  [(U9b) done at " << secs() << " s]" << std::endl;
        ArmU8(g);
        std::cout << "  [(U8) done at " << secs() << " s]" << std::endl;
        ArmU10(g);
        std::cout << "  [(U10) done at " << secs() << " s]" << std::endl;
    }
    std::cout << "HeckeHidaCMValidation: " << g.checks << " checks, " << g.fails << " failing, " << secs() << " s" << std::endl;
    return g.fails == 0 ? 0u : 1u;
}
