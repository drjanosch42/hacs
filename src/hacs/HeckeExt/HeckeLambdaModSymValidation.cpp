// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeLambdaModSymValidation (standard) -- the gate of P8 (hecke-extensions
// slot 21): ordinary cohomology along the tower as a Lambda-module, truncated
// modulo (ell^K, X^m) -- truncated Lambda-adic modular symbols.  Binding text:
// [U] docs/latex/hida_hecke_algebra.tex v0.5.0-pre.2 part 4 (sec:u-msym).
// Spec: docs/implementation/slot21/P8_spec.md v0.2 (referee F4); the gates
// G1-G16 and the negative controls of its Sec. 4.
//
// Cells (default): (11, 2) and (13, 0) (the Delta family, r = 1), (23, 2) (one
// component of residue degree 2), (37, 32) (the cusp component through m_Eis
// and the others of the class); K = 3, m = 1, 2, 3, B = 30, M_max = 2; N = 5
// consecutive weights k_0 + j (ell - 1) computed mod ell^{K'}, K' = K + 2 = 5,
// then F = 3 further weights.  Plus (5, a), (7, a) (G13), the level-ell arm at
// ell in {11, 13} (G4), the lattice arm (G14).
//
// ORACLES (never this code's own output): P7's forms side (WP-HL's
// ComputeFormsWeight at the weight itself, its component read here by its own
// Fitting split; P7's truncations BuildLambdaQExpansion through SpecialiseAt)
// for G2 / G3; WP-MEAS's BuildMeasure (G6, G7); WP-LIN's HeckeTp (the fast
// Hecke of HeckeLambdaModSym is gated against it); WP-LV3's JointBlockOf (G15);
// the slot-13 level-ell code (G4); the slot-13 lattice pair (G14); the tower
// catalogue docs/matrix-engine/results/hida_towers/hida_towers.tsv (G1); the
// exact Bernoulli numbers (L_k, G11).
//
// Gate kinds: PROVED gates count as [FAIL] (a defect); CONJ / COMP gates as
// [FINDING] (counted, printed, never a failure): G5, G8, G9, G11 at n >= ell - 1,
// G12's w-half.
//
// Env (read-only; prints EFFECTIVE HeckeLambdaModSymValidation arm=<X> cells=<k>):
//   HECKE_LAMBDA_MSYM_ALL=1     adds (103, 24) and F = 5 further weights;
//   HECKE_LAMBDA_MSYM_CELLS="ell:a:k0,..."  replaces the cell list (a malformed token
//                               is refused by position; a list selecting no cell fails).
// Writes nothing, no golden object, no digest of record.
#include "HeckeLambdaModSymValidation.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeLevelEll.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeOrdinaryPart.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "HeckeHidaLambda.h"
#include "HeckeHidaTower.h"
#include "HeckeLambdaModSym.h"
#include "HeckeLambdaQExp.h"
#include "LValue/HidaPairing.h"
#include "LValue/MBlockReader.h"
#include "LValue/ModularSymbolMeasure.h"
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

namespace {

namespace L = hecke_ext::lambda;
namespace MS = hecke_ext::lambda::msym;
namespace bm = hecke_ext::lvalue::birch;
namespace hp = hecke_ext::lvalue::hidapair;
using MS::SMat;
using MS::SVec;
using L::u128;

oc_util::Checker g_check("  [FAIL] "), g_finding("  [FINDING] ");
std::map<std::string, uint32_t> g_by_gate;

void Check(bool ok, const std::string& what) {
    ++g_by_gate[what.substr(0, what.find(' '))];
    g_check(ok, what);
}
void Finding(bool ok, const std::string& what) {
    ++g_by_gate[what.substr(0, what.find(' '))];
    g_finding(ok, what);
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
long PeakRssMiB() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024;
}
bool IsPrime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}
uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}
uint32_t ValU(uint64_t x, uint64_t ell, uint32_t cap) {
    if (x == 0) return cap;
    uint32_t v = 0;
    while (x % ell == 0 && v < cap) {
        x /= ell;
        ++v;
    }
    return v;
}
std::string VecText(const SVec& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}
SMat ToSMat(const L::Mat& A, uint64_t q) {
    SMat B(static_cast<uint32_t>(A.size()), A.empty() ? 0 : static_cast<uint32_t>(A[0].size()));
    for (uint32_t i = 0; i < B.r; ++i)
        for (uint32_t j = 0; j < B.c; ++j) B.at(i, j) = static_cast<uint64_t>(A[i][j] % q);
    return B;
}
SVec Col(const SMat& A, uint32_t j) {
    SVec v(A.r);
    for (uint32_t i = 0; i < A.r; ++i) v[i] = A.at(i, j);
    return v;
}
SMat FromCols(const std::vector<SVec>& cols, uint32_t rows) {
    SMat A(rows, static_cast<uint32_t>(cols.size()));
    for (uint32_t j = 0; j < A.c; ++j)
        for (uint32_t i = 0; i < rows; ++i) A.at(i, j) = cols[j][i];
    return A;
}
std::string CellTag(uint64_t ell, uint32_t a) { return "(" + std::to_string(ell) + "," + std::to_string(a) + ")"; }

// ===========================================================================
// the forms-side oracle: the g(T_{p0})-component of e S_k (WP-HL's forms data at the weight itself), the
// presentation F = charpoly(T_{p0} | V_g) and every operator as a polynomial in T_{p0}, all mod ell^{K'}
// ===========================================================================
struct FormsComp {
    bool ok = false;
    uint32_t r = 0;
    SVec F;
    std::map<uint64_t, SVec> P;   // p -> coefficients; key ell: U
    std::string why;
};

FormsComp FormsCompAt(const L::FormsWeight& w, uint64_t p0, const std::vector<uint64_t>& g, uint32_t Kp) {
    FormsComp fc;
    const bm::Ring R = bm::MakeRing(w.ell, Kp);
    const L::FormsSpace& S = w.S;
    const uint32_t s = S.s;
    if (s == 0) {
        fc.why = "e S_k = 0";
        return fc;
    }
    const SMat t = ToSMat(S.mats.at("T" + std::to_string(p0)), R.q);
    uint64_t e = 1;
    while (e < static_cast<uint64_t>(Kp) * s) e <<= 1;
    const SMat G = MS::MatPow(R, MS::MatPolyOf(R, SVec(g.begin(), g.end()), t), e);
    std::vector<SVec> kg;
    std::vector<uint32_t> ke;
    bm::KernelGenerators(R, G, kg, ke);
    for (uint32_t x : ke)
        if (x != Kp) {
            fc.why = "component not free";
            return fc;
        }
    fc.r = static_cast<uint32_t>(kg.size());
    if (fc.r != g.size() - 1) {
        fc.why = "component rank " + std::to_string(fc.r) + " != deg g";
        return fc;
    }
    const MS::FreeBasis V = MS::MakeFreeBasis(R, kg);
    auto restrict = [&](const SMat& A) {
        SMat C(fc.r, fc.r);
        for (uint32_t j = 0; j < fc.r; ++j) {
            SVec c;
            ENGINE_CHECK_MSG(V.coords(R, bm::MatVec(R, A, V.b[j]), c), "oracle: the forms component is not stable");
            for (uint32_t i = 0; i < fc.r; ++i) C.at(i, j) = c[i];
        }
        return C;
    };
    const SMat tV = restrict(t);
    fc.F = bm::CharPoly(R, tV);
    // a cyclic vector of t on V_g, the basis y_i = t^i v
    SMat Y, Yi;
    bool found = false;
    for (uint32_t j = 0; j < fc.r && !found; ++j) {
        std::vector<SVec> cols{SVec(fc.r, 0)};
        cols[0][j] = 1;
        for (uint32_t i = 1; i < fc.r; ++i) cols.push_back(bm::MatVec(R, tV, cols.back()));
        Y = FromCols(cols, fc.r);
        found = MS::InverseUnit(R, Y, Yi);
    }
    if (!found) {
        fc.why = "T_p0 not cyclic on the forms component";
        return fc;
    }
    const SMat tY = bm::MatMul(R, Yi, bm::MatMul(R, tV, Y));
    for (const std::string& name : S.names) {
        const SMat AY = bm::MatMul(R, Yi, bm::MatMul(R, restrict(ToSMat(S.mats.at(name), R.q)), Y));
        const SVec c = Col(AY, 0);
        if (!MS::MatEq(MS::MatPolyOf(R, c, tY), AY)) {
            fc.why = name + " is not a polynomial in T_p0 on the forms component";
            return fc;
        }
        const uint64_t key = name == "U" ? w.ell : std::stoull(name.substr(1));
        fc.P[key] = c;
    }
    fc.ok = true;
    return fc;
}

// The tower catalogue (oracle of G1's Eichler-Shimura count h^1_ord = 2 d_k + 1).
std::map<std::pair<uint64_t, uint32_t>, uint32_t> TowerCatalogue() {
    std::map<std::pair<uint64_t, uint32_t>, uint32_t> out;
    std::ifstream in(std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results/hida_towers/hida_towers.tsv");
    std::string line;
    std::vector<std::string> head;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, '\t')) f.push_back(x);
        if (head.empty()) {
            head = f;
            continue;
        }
        auto col = [&](const std::string& name) -> std::string {
            for (size_t i = 0; i < head.size() && i < f.size(); ++i)
                if (head[i] == name) return f[i];
            return "";
        };
        if (col("m") != "1" || col("lattice") == "hida") continue;
        const std::string l = col("ell"), n = col("n"), o = col("ord_rank");
        if (l.empty() || n.empty() || o.empty()) continue;
        out[{std::stoull(l), static_cast<uint32_t>(std::stoul(n))}] = static_cast<uint32_t>(std::stoul(o));
    }
    return out;
}

// ===========================================================================
// one cell
// ===========================================================================
struct CellSpec {
    uint64_t ell;
    uint32_t a, k0;
    uint32_t F = 0;   // further weights (0: the arm's default)
};

constexpr uint32_t kK = 3, kKp = 5, kB = 30, kM = 2, kN = 5;

// level-ell targets collected from the cells (G4)
struct LevelTarget {
    uint64_t ell = 0;
    uint32_t k = 0, r = 0;
    uint64_t p0 = 0;
    std::vector<uint64_t> g;
    SVec F, alpha;   // forms side at k, mod ell^{K'}
    L::QElem alphaTrunc;
    std::vector<L::Vec> FTrunc;
    uint32_t specPrec = 0;
    uint32_t centre = 0;
};
std::vector<LevelTarget> g_level_targets;
bool g_level_full = false;   // HECKE_LAMBDA_MSYM_ALL: the level-ell arm at all five weights (k = 32 at 11, k = 24 at 13: + ~80 s), G1 totals at 37

struct SideRun {
    uint32_t comp = 0;
    int sign = 0;
    std::vector<MS::Reading> rd;
    std::vector<FormsComp> fc;
};

void RunCell(const CellSpec& cs, uint32_t Fw, const std::map<std::pair<uint64_t, uint32_t>, uint32_t>& catalogue) {
    const double tc = Now();
    const uint64_t l = cs.ell;
    std::vector<uint64_t> primes;
    for (uint64_t p = 2; p <= kB; ++p)
        if (IsPrime(p) && p != l) primes.push_back(p);
    std::vector<uint32_t> ks, interp, further;
    for (uint32_t j = 0; j < kN + Fw; ++j) {
        const uint32_t k = cs.k0 + j * static_cast<uint32_t>(l - 1);
        ks.push_back(k);
        (j < kN ? interp : further).push_back(k);
    }
    const std::string ctag = CellTag(l, cs.a);
    // ---- the forms side (P7) at every weight
    std::vector<L::FormsWeight> ws;
    for (uint32_t k : ks) ws.push_back(L::ComputeFormsWeight(l, k, kKp, primes));
    const std::vector<L::FormsWeight> wsI(ws.begin(), ws.begin() + kN);
    const bool with_eis = cs.a % (l - 1) != 0;
    std::vector<L::LambdaQExpansion> LQ;
    for (uint32_t m = 1; m <= 3; ++m) LQ.push_back(L::BuildLambdaQExpansion(wsI, cs.a, kK, m, cs.k0, kB, 0, with_eis));
    const uint64_t p0 = LQ[2].p0;
    std::vector<size_t> compIdx;
    for (size_t i = 0; i < LQ[2].comps.size(); ++i)
        if (LQ[2].comps[i].kind != "eisenstein") compIdx.push_back(i);
    {
        std::ostringstream os;
        os << "CELL ell=" << l << " a=" << cs.a << " weights=" << ks.front() << ".." << ks.back() << " (N=" << kN << ", F=" << Fw
           << ") p0=" << p0 << " d=" << ws[0].S.s << " components=";
        for (size_t i : compIdx) os << "[" << LQ[2].comps[i].kind << ",r=" << LQ[2].comps[i].rank << ",g=" << VecText(LQ[2].comps[i].residual) << "]";
        std::cout << os.str() << std::endl;
    }
    for (const auto& w : ws) Check(w.S.s == ws[0].S.s, "G1 " + ctag + " k=" + std::to_string(w.k) + ": the forms ordinary rank d_k is constant in the class");

    // ---- G1 totals: the ordinary ranks of W^{-+} (ell <= 23) and of H^1_{+-} (ell <= 13), against d_k and the catalogue
    std::map<std::tuple<uint32_t, int, uint32_t>, std::shared_ptr<MS::SignSpace>> spaces;
    auto space = [&](uint32_t k, int sign, uint32_t R) {
        const auto key = std::make_tuple(k, sign, R);
        auto it = spaces.find(key);
        if (it == spaces.end()) it = spaces.emplace(key, std::make_shared<MS::SignSpace>(MS::BuildSignSpace(l, k - 2, sign, R, 2))).first;
        return it->second;
    };
    if (l <= 23 || (g_level_full && l <= 37)) {
        for (size_t wi = 0; wi < ks.size(); ++wi) {
            const uint32_t k = ks[wi], n = k - 2, d = ws[wi].S.s;
            const uint32_t om = MS::OrdinaryRankAll(*space(k, -1, kKp)), op = MS::OrdinaryRankAll(*space(k, 1, kKp));
            Check(om == d + 1 && op == d, "G1 " + ctag + " k=" + std::to_string(k) + ": rank e_1 W^{[C]-} = d_k + 1, rank e_1 W^{[C]+} = d_k (" +
                                              std::to_string(om) + ", " + std::to_string(op) + "; d_k=" + std::to_string(d) + ")");
            const auto it = catalogue.find({l, n});
            if (it != catalogue.end())
                Check(it->second == 2 * d + 1, "G1 " + ctag + " k=" + std::to_string(k) + ": the tower catalogue's h^1_ord(n=" +
                                                   std::to_string(n) + ") = " + std::to_string(it->second) + " = 2 d_k + 1");
            if (l <= 13) {
                const auto hr = MS::AmalgamOrdinaryRanks(n, l, kKp);
                Check(hr.first == d + 1 && hr.second == d, "G1 " + ctag + " k=" + std::to_string(k) +
                                                               ": rank e_1 H^1_+ = d_k + 1, rank e_1 H^1_- = d_k on H^1(Gamma; V) (" +
                                                               std::to_string(hr.first) + ", " + std::to_string(hr.second) + ")");
            }
        }
    }

    // ---- (U17)(a): the gluing depth j_k at m_Eis on H^1 (read first: it enters the records of the m_Eis component)
    bool eis_cell = false;
    for (size_t i : compIdx) eis_cell = eis_cell || LQ[2].comps[i].kind == "cusp_eis";
    std::vector<uint32_t> Lks(ks.size(), 0);
    std::vector<MS::GluingReading> glue(ks.size());
    std::vector<int32_t> jrec;
    if (eis_cell)
        for (size_t wi = 0; wi < ks.size(); ++wi) {
            const L::ZMod R6(l, 6);
            Lks[wi] = R6.val(L::BernoulliOverKExact(l, 6, {ks[wi]}).at(ks[wi]), 6);
            glue[wi] = MS::GluingDual(ks[wi] - 2, l, Lks[wi] + 3);
            jrec.push_back(glue[wi].refusal.empty() ? static_cast<int32_t>(glue[wi].j) : -1);
        }

    // ---- per component, per sign, per weight
    // D extra digits on the p_0 sign of m_Eis (the division by ell^d), E = 2 for the reduction W(R + E) -> W(R): ell^{K' + D + E} < 2^62
    uint32_t maxprec = 0;
    for (unsigned __int128 x = l; x < (static_cast<unsigned __int128>(1) << 62); x *= l) ++maxprec;
    const uint32_t D = std::min<uint32_t>(3, maxprec - kKp - 2);
    std::vector<int32_t> jk(ks.size(), -1);
    std::map<std::pair<size_t, int>, SideRun> runs;
    for (size_t ci : compIdx) {
        const L::LambdaComponent& C3 = LQ[2].comps[ci];
        const bool eis = C3.kind == "cusp_eis";
        const uint32_t r = C3.rank;
        for (int sign : {-1, 1}) {
            const double ts = Now();
            const bool p0side = eis && sign < 0;
            const std::string stag = ctag + " comp=" + std::to_string(ci) + "(" + C3.kind + ",r=" + std::to_string(r) + ") sign=" +
                                     (sign > 0 ? "+" : "-");
            SideRun run;
            run.comp = static_cast<uint32_t>(ci);
            run.sign = sign;
            for (size_t wi = 0; wi < ks.size(); ++wi) {
                const uint32_t k = ks[wi];
                MS::ReadingRequest rq;
                rq.ell = l;
                rq.k = k;
                rq.Kp = kKp;
                rq.c_sign = sign;
                rq.p0 = p0;
                rq.g = C3.residual;
                rq.eis = eis;
                rq.D = D;
                rq.Emax = 2;
                rq.M = kM;
                rq.primes = primes;
                rq.space = space(k, sign, kKp + (p0side ? D : 0));
                run.rd.push_back(MS::ReadComponent(rq));
                run.fc.push_back(FormsCompAt(ws[wi], p0, C3.residual, kKp));
            }
            const bm::Ring Rk = bm::MakeRing(l, kKp);
            {
                std::ostringstream os;
                os << "  RANKS " << stag << " (W, g-part, ordinary, E) =";
                for (size_t wi = 0; wi < ks.size(); ++wi)
                    os << " " << ks[wi] << ":(" << run.rd[wi].w_rank << "," << run.rd[wi].g_rank << "," << run.rd[wi].ord_rank << "," << run.rd[wi].E
                       << ")" << (run.rd[wi].refusal.empty() ? "" : "[" + run.rd[wi].refusal + "]");
                std::cout << os.str() << std::endl;
            }
            // ---- per-weight gates
            for (size_t wi = 0; wi < ks.size(); ++wi) {
                const MS::Reading& rd = run.rd[wi];
                const FormsComp& fc = run.fc[wi];
                const uint32_t k = ks[wi], n = k - 2;
                const std::string wt = stag + " k=" + std::to_string(k);
                if (!eis) {
                    // G5 / G15: (BF) over every b mod ell - 1 (a refusal by name is counted, never silent)
                    Finding(rd.refusal != "boundary_component", "G5 " + wt + ": (BF) holds (m != m_b for every b mod ell-1)" +
                                                                       (rd.bf_fail_b >= 0 ? " -- refused at b=" + std::to_string(rd.bf_fail_b) : ""));
                    Check(rd.bf_checked, "G15 " + wt + ": (BF) checked over all ell - 1 residues b, on T_2, T_3, T_5");
                    if (rd.refusal == "boundary_component") continue;
                }
                Check(rd.refusal.empty(), "G1 " + wt + ": the ordinary g-part read (refusal='" + rd.refusal + "')");
                if (!rd.refusal.empty()) continue;
                // G1: rank r (length K' r), free, both signs
                Check(rd.ord_free && rd.r == r && rd.ord_rank == r + (p0side ? 1 : 0) && fc.ok && fc.r == r,
                      "G1 " + wt + ": S^{+-}_k(Z/ell^R)_m free of rank r=" + std::to_string(r) + " (ord_rank=" + std::to_string(rd.ord_rank) +
                          (p0side ? " incl. p_0" : "") + ", g-part " + std::to_string(rd.g_rank) + ", W " + std::to_string(rd.w_rank) +
                          ", E=" + std::to_string(rd.E) + ")");
                // G2: cyclic; charpoly == F(x_k, Y) against the forms data at k (mod ell^{K'}) and P7's truncations
                Check(fc.ok && rd.F == fc.F, "G2 " + wt + ": charpoly(T_p0 | S) == F_k of the forms data mod ell^5 " + VecText(rd.F));
                for (uint32_t m = 1; m <= 3; ++m) {
                    const L::LambdaComponent& Cm = LQ[m - 1].comps[ci];
                    const uint32_t e = L::SpecPrecision(l, kK, m, cs.k0, k);
                    const uint64_t qe = PowU(l, e);
                    bool ok = Cm.residual == C3.residual;
                    const L::Vec Fs = L::SpecialiseAt(L::QElem(Cm.F.begin(), Cm.F.end()), l, cs.k0, k, e);
                    for (uint32_t i = 0; ok && i <= r; ++i) ok = static_cast<uint64_t>(Fs[i] % qe) == rd.F[i] % qe;
                    for (uint64_t p : primes) {
                        const L::Vec ps = L::SpecialiseAt(Cm.ap.at(p), l, cs.k0, k, e);
                        for (uint32_t i = 0; ok && i < r; ++i) ok = static_cast<uint64_t>(ps[i] % qe) == rd.P.at(p)[i] % qe;
                    }
                    const L::Vec as = L::SpecialiseAt(Cm.alpha, l, cs.k0, k, e);
                    for (uint32_t i = 0; ok && i < r; ++i) ok = static_cast<uint64_t>(as[i] % qe) == rd.P.at(l)[i] % qe;
                    Check(ok, "G2 " + wt + " m=" + std::to_string(m) + ": F, P_p (p <= 30) and alpha == P7's truncation at x_k mod ell^" +
                                  std::to_string(e));
                }
                // G3: T_p = P_p(x_k, T_{p0}), U = alpha(x_k, T_{p0}) on the block, against the forms data at k
                {
                    bool ok = fc.ok && rd.P_consistent;
                    for (uint64_t p : primes) ok = ok && rd.P.at(p) == fc.P.at(p);
                    ok = ok && rd.P.at(l) == fc.P.at(l);
                    Check(ok, "G3 " + wt + ": every T_p (p <= 30) and U is P_p(T_p0) / alpha(T_p0) on S, == the forms data mod ell^5" +
                                  (r == 1 ? " (a_p(x_k), alpha=" + std::to_string(rd.P.at(l)[0]) + ")" : ""));
                }
                // G16: U = T_ell on the block over Z/ell^R (k - 1 > R) and ell^{k-2} == 0 mod ell^R
                Check(rd.U_eq_Tell && rd.U_root_ok && rd.lk2_vanishes,
                      "G16 " + wt + ": U = T_ell exactly on the ordinary block, U^2 - T_ell U + ell^{k-1} = 0, ell^{k-2} == 0 mod ell^R");
                // G6: the operator masses are a distribution; r = 1: == BuildMeasure(phi, a_ell, alpha) ball by ball
                Check(rd.dist_failures == 0 && rd.gram_unit,
                      "G6 " + wt + ": the operator-valued masses of every y_i are a distribution (failures=" + std::to_string(rd.dist_failures) +
                          "), Gram of h_k a unit");
                if (r == 1) {
                    const MS::OrdBlock& Bl = *rd.block;
                    const uint32_t off = rd.has_p0 ? 1 : 0;
                    const SVec phiAmb = [&] {
                        SVec v(n + 1, 0);
                        for (uint32_t j = 0; j < Bl.O.rank(); ++j)
                            for (uint32_t t = 0; t <= n; ++t) v[t] = Rk.add(v[t], Rk.mul(rd.phi[j], Bl.O.b[j][t] % Rk.q));
                        return v;
                    }();
                    const uint64_t a_ell = Bl.Tell.at(off, off) % Rk.q, alpha = rd.P.at(l)[0];
                    const bm::SymRho rho(n, Rk);
                    const auto me = bm::BuildMeasure(rho, phiAmb, a_ell, alpha, kM, true);
                    bool ok = true;
                    for (size_t bi = 0; bi < rd.balls.size(); ++bi) ok = ok && me.ball(rd.balls[bi].b, rd.balls[bi].M) == rd.muh[bi][0];
                    if (!p0side) {
                        const MS::OperatorMeasure om(Rk, *rd.tables, MS::Reduce(Bl.U, Rk.q));
                        for (uint32_t m = 0; m <= kM; ++m)
                            for (uint64_t a = 0; a < PowU(l, m); ++a) ok = ok && om.ball(rd.phi, a, m) == me.ball(a, m);
                    }
                    Check(ok, "G6 " + wt + ": r = 1: the operator masses == BuildMeasure(phi, a_ell, alpha) on every " +
                                  (p0side ? "unit ball (mod p_0)" : "ball to level 2"));
                }
                // G7
                Check(rd.p0_unit_mass_nonzero == 0, "G7 " + wt + ": p_0 contributes 0 to every unit ball (U = 1 on p_0)");
                // G15 at m_Eis: the second boundary line (chi_b == chi_0 for 1 <= b <= n/ell) does not occur; the reduction kills E >= d
                if (eis) {
                    bool second = false;
                    for (uint32_t b = 1; b <= n / l; ++b)
                        if (b % (l - 1) == 0 || (b % (l - 1)) == ((n + 1) % (l - 1))) second = true;
                    Check(!second && rd.E >= rd.depth_d && rd.E >= 1,
                          "G15 " + wt + ": no second boundary line at m_Eis (chi_b != chi_0, 1 <= b <= n/ell); W read through Z/ell^{R+" +
                              std::to_string(rd.E) + "} -> Z/ell^R, length R * rank");
                }
            }
            // G15 / step 5: WP-LV3's joint cut (T_2, T_3, T_5) at m_Eis on the p_0 sign agrees with the T_{p0} g-part after e_1
            if (p0side) {
                for (size_t wi = 0; wi < ks.size(); ++wi) {
                    const MS::Reading& rd = run.rd[wi];
                    if (!rd.refusal.empty()) continue;
                    MS::SignSpace& S = *rd.space;
                    const bm::Ring& R = S.ring;
                    const uint32_t n = ks[wi] - 2;
                    auto rows = [&](uint64_t p) {
                        const SMat& A = S.TonW(p);
                        hp::Mat o(A.r, hp::Vec(A.c));
                        for (uint32_t i = 0; i < A.r; ++i)
                            for (uint32_t j = 0; j < A.c; ++j) o[i][j] = A.at(i, j);
                        return o;
                    };
                    const hp::Zq Z(l, S.R);
                    const hp::Mat r2 = rows(2), r3 = rows(3), r5 = rows(5);
                    auto ev = [&](uint64_t p) { return R.add(1, R.pow(p, n + 1)); };
                    const auto jb = hecke_ext::lvalue::mblock::JointBlockOf(Z, r2, ev(2), {{&r3, ev(3)}, {&r5, ev(5)}});
                    std::vector<SVec> amb;
                    for (const auto& c : jb.basis) {
                        SVec v(n + 1, 0);
                        for (size_t j = 0; j < c.size(); ++j)
                            for (uint32_t t = 0; t <= n; ++t) v[t] = R.add(v[t], R.mul(c[j], S.W.b[j][t]));
                        amb.push_back(v);
                    }
                    uint32_t ordr = 0;
                    if (!amb.empty()) {
                        const MS::FreeBasis Bk = MS::MakeFreeBasis(R, amb);
                        SMat A(Bk.rank(), Bk.rank());
                        for (uint32_t j = 0; j < Bk.rank(); ++j) {
                            SVec c;
                            ENGINE_CHECK(Bk.coords(R, S.ApplyT(Bk.b[j], l), c));
                            for (uint32_t i = 0; i < Bk.rank(); ++i) A.at(i, j) = c[i];
                        }
                        ordr = MS::RankModEll(R, MS::MatPow(R, A, 64));   // the ordinary rank: rank of T_ell^N mod ell
                    }
                    Check(ordr == rd.ord_rank, "G15 " + stag + " k=" + std::to_string(ks[wi]) + ": WP-LV3's joint (T_2,T_3,T_5) m_Eis block (rank " +
                                                   std::to_string(jb.basis.size()) + ") has ordinary rank " + std::to_string(ordr) +
                                                   " == the T_p0 g-part's " + std::to_string(rd.ord_rank));
                }
            }
            // ---- the normaliser of record, G8, G10, G12, interpolation, G9, records
            const MS::Reading& centre = run.rd[0];
            if (!centre.refusal.empty()) continue;
            const int b0 = MS::FirstUnitBall(centre);
            Finding(b0 >= 0, "G8 " + stag + ": a unit ball B_0 exists at the centre weight (else no_unit_ball)");
            if (b0 < 0) continue;
            const MS::Ball B0 = centre.balls[static_cast<size_t>(b0)];
            std::vector<std::vector<SVec>> nu(ks.size());
            bool all_unit = true;
            for (size_t wi = 0; wi < ks.size(); ++wi) {
                const MS::Reading& rd = run.rd[wi];
                if (!rd.refusal.empty()) {
                    all_unit = false;
                    continue;
                }
                const bool ok = MS::Normalised(rd, b0, nu[wi]);
                all_unit = all_unit && ok;
                Finding(ok, "G8 " + stag + " k=" + std::to_string(ks[wi]) + ": mu^h_{phi_k}(B_0) a unit, B_0 = " + std::to_string(B0.b) + "+ell^" +
                                std::to_string(B0.M) + " (the same at every weight)");
                // G12 (the p_0 sign of m_Eis): r has every unit mass == 0 mod ell^d (PROVED); w passes the unit check (G8, CONJ)
                if (p0side && rd.depth_d >= 1) {
                    const std::vector<uint64_t> mr = MS::UnitMasses(rd, rd.r_sym);
                    uint32_t vmin = kKp;
                    for (uint64_t x : mr) vmin = std::min(vmin, ValU(x, l, kKp));
                    Check(vmin >= rd.depth_d, "G12 " + stag + " k=" + std::to_string(ks[wi]) + ": the H^1_c-primitive r (depth d=" +
                                                  std::to_string(rd.depth_d) + ") has every unit mass == 0 mod ell^d (min v = " + std::to_string(vmin) + ")");
                    Finding(ok, "G12 " + stag + " k=" + std::to_string(ks[wi]) + ": the reduced symbol w passes the unit check");
                }
            }
            // G10: nu is independent of the choice of phi_k (phi -> c(T_p0) phi, c a random unit of h_k), at the centre and one further weight
            {
                std::mt19937_64 rng(20261005 + 7 * ci + (sign > 0));
                for (size_t wi : {static_cast<size_t>(0), ks.size() - 1}) {
                    const MS::Reading& rd = run.rd[wi];
                    if (!rd.refusal.empty() || nu[wi].empty()) continue;
                    const uint32_t off = rd.has_p0 ? 1 : 0, o = rd.ord_rank;
                    SVec c(r);
                    for (uint32_t i = 0; i < r; ++i) c[i] = rng() % Rk.q;
                    if (c[0] % l == 0) c[0] = Rk.add(c[0], 1);
                    // phi' = c(T_p0) phi in S coordinates (the companion basis -> S basis through phi's Krylov basis)
                    const MS::OrdBlock& Bl = *rd.block;
                    SMat TS(r, r);
                    const SMat Tp0O = MS::Reduce(Bl.T.at(p0), Rk.q);
                    for (uint32_t i = 0; i < r; ++i)
                        for (uint32_t j = 0; j < r; ++j) TS.at(i, j) = Tp0O.at(i + off, j + off);
                    SVec phiS(r);
                    for (uint32_t i = 0; i < r; ++i) phiS[i] = rd.phi[i + off];
                    const SVec phi2 = bm::MatVec(Rk, MS::MatPolyOf(Rk, c, TS), phiS);
                    std::vector<SVec> yO;
                    SVec cur = phi2;
                    for (uint32_t i = 0; i < r; ++i) {
                        SVec x(o, 0);
                        for (uint32_t t = 0; t < r; ++t) x[t + off] = cur[t];
                        yO.push_back(x);
                        cur = bm::MatVec(Rk, TS, cur);
                    }
                    std::vector<std::vector<uint64_t>> ms;
                    for (const SVec& x : yO) ms.push_back(MS::UnitMasses(rd, x));
                    const MS::HRing h{Rk, rd.F};
                    std::vector<SVec> mh(rd.balls.size());
                    bool ok = true;
                    for (size_t bi = 0; bi < rd.balls.size(); ++bi) {
                        SVec mus(r);
                        for (uint32_t i = 0; i < r; ++i) mus[i] = ms[i][bi];
                        ok = ok && MS::HValuedMass(h, mus, mh[bi]);
                    }
                    SVec iv;
                    ok = ok && h.inv(mh[static_cast<size_t>(b0)], iv);
                    for (size_t bi = 0; ok && bi < rd.balls.size(); ++bi) ok = h.mul(mh[bi], iv) == nu[wi][bi];
                    Check(ok, "G10 " + stag + " k=" + std::to_string(ks[wi]) + ": nu_B unchanged under phi -> c(T_p0) phi, c a random unit of h_k");
                }
            }
            // interpolation (thm:u-interp through WP-HL's InterpolateTruncation), G9 (CONJ: U16(b))
            const size_t nb = centre.balls.size();
            if (!all_unit) {
                Finding(false, "G9 " + stag + ": not read (a weight without a unit B_0 or refused)");
                continue;
            }
            std::vector<std::vector<uint64_t>> lam(further.size());
            for (size_t fi = 0; fi < further.size(); ++fi) lam[fi] = MS::NewtonWeights(l, interp, kKp, further[fi]);
            uint32_t mism = 0, vmis = kKp;
            const uint32_t prec9 = std::min<uint32_t>(kKp, kN);
            const uint64_t q9 = PowU(l, prec9);
            for (size_t bi = 0; bi < nb; ++bi)
                for (uint32_t i = 0; i < r; ++i)
                    for (size_t fi = 0; fi < further.size(); ++fi) {
                        uint64_t pr = 0;
                        for (size_t a = 0; a < kN; ++a) pr = Rk.add(pr, Rk.mul(lam[fi][a], nu[a][bi][i]));
                        const uint64_t got = nu[kN + fi][bi][i];
                        if (pr % q9 != got % q9) {
                            ++mism;
                            vmis = std::min(vmis, ValU(Rk.sub(pr, got), l, kKp));
                        }
                    }
            Finding(mism == 0, "G9 " + stag + ": the Newton interpolant of nu_B through the N=5 weights predicts every further weight mod ell^" +
                                   std::to_string(prec9) + " (" + std::to_string(nb) + " balls x r=" + std::to_string(r) + " x " +
                                   std::to_string(further.size()) + " weights; mismatches=" + std::to_string(mism) +
                                   (mism ? ", min v=" + std::to_string(vmis) : "") + ")");
            uint32_t interp_cmp = 0;
            bool interp_ok = true;
            for (uint32_t m = 1; m <= 3; ++m) {
                const MS::TruncationMap TM = MS::MakeTruncationMap(l, interp, kKp, kK, m, cs.k0);
                MS::RecordInput rec;
                rec.ell = l;
                rec.a = cs.a;
                rec.K = kK;
                rec.m = m;
                rec.centre_k = cs.k0;
                rec.N = kN;
                rec.component = static_cast<uint32_t>(ci);
                rec.sign = sign;
                rec.rank = r;
                rec.p0 = p0;
                rec.b0 = b0;
                rec.B0_b = B0.b;
                rec.B0_M = B0.M;
                rec.unit_ok = all_unit;
                rec.weights = interp;
                if (eis) rec.j = jrec;
                std::vector<uint32_t> bad(further.size(), kK + 1);
                for (size_t bi = 0; bi < nb; ++bi) {
                    std::vector<L::Vec> coeffs;
                    for (uint32_t i = 0; i < r; ++i) {
                        std::vector<uint64_t> vals;
                        for (size_t a = 0; a < kN; ++a) vals.push_back(nu[a][bi][i]);
                        L::Vec tr;
                        interp_ok = TM.Apply(vals, tr) && interp_ok;
                        if (centre.balls[bi].M == 1) {
                            // WP-HL's InterpolateTruncation (the enforced precision bound) on every level-1 ball
                            const L::Vec trw = L::InterpolateTruncation(l, interp, L::Vec(vals.begin(), vals.end()), kKp, kK, m, cs.k0);
                            interp_ok = interp_ok && trw == tr;
                            ++interp_cmp;
                        }
                        coeffs.push_back(tr);
                        for (size_t fi = 0; fi < further.size(); ++fi) {
                            const uint32_t e = L::SpecPrecision(l, kK, m, cs.k0, further[fi]);
                            const uint64_t got = L::SpecialiseAt(L::QScalar(tr, 1), l, cs.k0, further[fi], e)[0] % PowU(l, e);
                            if (got != nu[kN + fi][bi][i] % PowU(l, e)) bad[fi] = std::min(bad[fi], ValU(Rk.sub(got, nu[kN + fi][bi][i]), l, e));
                        }
                    }
                    rec.nu.push_back({centre.balls[bi], coeffs});
                }
                for (size_t fi = 0; fi < further.size(); ++fi)
                    rec.further.push_back(std::to_string(further[fi]) + ":" + (bad[fi] > kK ? std::string("ok") : "fail@" + std::to_string(bad[fi])));
                std::cout << MS::RecordLine(rec) << std::endl;
            }
            Check(interp_ok, "INTERP " + stag + ": the truncations of every nu_B are ell-integral (thm:u-interp(1)) and the linear map == WP-HL's "
                             "InterpolateTruncation on all " + std::to_string(interp_cmp) + " level-1 (ball, coefficient, m)");
            runs[{ci, sign}] = run;
            std::cout << "  SIDE " << stag << " B0=" << B0.b << "+ell^" << B0.M << " weights=" << ks.size() << " wall_s=" << Now() - ts << std::endl;
            if (p0side) {
                std::ostringstream os;
                os << "  DEPTH " << stag << " d=[";
                for (size_t wi = 0; wi < ks.size(); ++wi) os << (wi ? "," : "") << run.rd[wi].depth_d;
                os << "]";
                std::cout << os.str() << std::endl;
                for (size_t wi = 0; wi < ks.size(); ++wi) jk[wi] = static_cast<int32_t>(run.rd[wi].depth_d);
            }
        }
        // level-ell targets (G4)
        if (l <= 13)
            for (size_t wi = 0; wi < ks.size(); ++wi) {
                const uint32_t k = ks[wi];
                const bool dflt = (l == 11 && (k == 12 || k == 22)) || (l == 13 && k == 12);
                const bool full = (l == 11 && k == 32) || (l == 13 && k == 24);
                if (!(dflt || (g_level_full && full))) continue;
                const FormsComp fc = FormsCompAt(ws[wi], p0, C3.residual, kKp);
                if (!fc.ok) continue;
                LevelTarget t;
                t.ell = l;
                t.k = k;
                t.r = r;
                t.p0 = p0;
                t.g = C3.residual;
                t.F = fc.F;
                t.alpha = fc.P.at(l);
                t.alphaTrunc = C3.alpha;
                t.FTrunc = C3.F;
                t.specPrec = L::SpecPrecision(l, kK, 3, cs.k0, k);
                t.centre = cs.k0;
                g_level_targets.push_back(t);
            }
    }
    // ---- negative controls (each must be caught)
    {
        const bm::Ring Rk = bm::MakeRing(l, kKp);
        const size_t c0 = compIdx.empty() ? 0 : compIdx[0];
        auto findRun = [&](size_t ci, int sign) -> const SideRun* {
            const auto it = runs.find({ci, sign});
            return it == runs.end() ? nullptr : &it->second;
        };
        if (l == 11) {
            // geometric parity used as the [C] sign: the Eisenstein count of G1 lands on the wrong side
            const uint32_t d = ws[0].S.s;
            const uint32_t wrong = MS::OrdinaryRankAll(*space(ks[0], 1, kKp));   // geometric-odd read as if [C]-minus
            Check(wrong != d + 1, "NEG " + ctag + ": the geometric sign used as the [C] sign is caught (G1: rank " + std::to_string(wrong) +
                                      " != d_k + 1 = " + std::to_string(d + 1) + ")");
            // HeckeTpWrongAdjoint in step 2: G2 fails
            bool caught = false;
            try {
                MS::ReadingRequest rq;
                rq.ell = l;
                rq.k = ks[1];
                rq.Kp = kKp;
                rq.c_sign = -1;
                rq.p0 = p0;
                rq.g = LQ[2].comps[c0].residual;
                rq.Emax = 2;
                rq.M = kM;
                rq.primes = primes;
                rq.wrong_adjoint = true;
                const MS::Reading rd = MS::ReadComponent(rq);
                const FormsComp fc = FormsCompAt(ws[1], p0, rq.g, kKp);
                caught = !rd.refusal.empty() || rd.F != fc.F;
            } catch (const std::exception&) {
                caught = true;
            }
            Check(caught, "NEG " + ctag + ": HeckeTpWrongAdjoint on W is caught (G2: charpoly / block)");
        }
        if (l == 23) {
            // the scalar alpha in place of the operator U at r = 2: the masses are no longer a distribution (G6)
            const SideRun* run = findRun(c0, 1);
            if (run && run->rd[0].refusal.empty()) {
                const MS::Reading& rd = run->rd[0];
                const SMat Us = MS::AddScalarPublic(Rk, SMat(rd.ord_rank, rd.ord_rank), rd.P.at(l)[0]);
                const MS::OperatorMeasure om(Rk, *rd.tables, Us);
                const uint32_t bad = om.distribution_failures(rd.phi);
                Check(bad > 0, "NEG " + ctag + ": the scalar alpha_0 in place of the operator U at r = 2 is caught (G6: " + std::to_string(bad) +
                                   " distribution failures)");
            } else {
                Check(false, "NEG " + ctag + ": the r = 2 reading for the scalar-alpha control is missing");
            }
        }
        if (l == 37 && compIdx.size() >= 2) {
            const size_t c1 = compIdx[1];
            const SideRun* run = findRun(c0, 1);
            if (run && run->rd[0].refusal.empty()) {
                const MS::Reading& rd = run->rd[0];
                // a perturbed a_5 from P7 (G3 against the truncation)
                L::QElem ap5 = LQ[2].comps[c0].ap.at(5);
                const L::ZMod RK(l, kK);
                ap5[0][0] = RK.add(ap5[0][0], l % RK.M);
                const uint64_t got = static_cast<uint64_t>(L::SpecialiseAt(ap5, l, cs.k0, cs.k0, kK)[0]);
                Check(got != rd.P.at(5)[0] % PowU(l, kK), "NEG " + ctag + ": a perturbed a_5 (+ell) from P7 is caught (G3 at the centre weight)");
                // B_0 re-chosen at a weight where nu_{B_0} is a non-unit: the planted non-unit fires G8's detector
                MS::Reading planted = rd;
                const int b0 = MS::FirstUnitBall(rd);
                std::vector<SVec> nu;
                if (b0 >= 0) {
                    for (auto& x : planted.muh[static_cast<size_t>(b0)]) x = Rk.mul(x, l % Rk.q);
                    Check(!MS::Normalised(planted, b0, nu) && MS::FirstUnitBall(planted) != b0,
                          "NEG " + ctag + ": a planted non-unit at B_0 fires the unit check (re-choosing B_0 would hide it)");
                }
                // the wrong component: the residual factor of another component (rank or charpoly fails)
                MS::ReadingRequest rq = rd.req;
                rq.g = LQ[2].comps[c1].residual;
                rq.eis = false;
                rq.space = space(ks[0], 1, kKp);
                const MS::Reading wrong = MS::ReadComponent(rq);
                Check(!wrong.refusal.empty() || wrong.F != run->fc[0].F || wrong.r != rd.r,
                      "NEG " + ctag + ": the wrong component's residual factor is caught (G1/G2)");
            } else {
                Check(false, "NEG " + ctag + ": the centre reading for the controls is missing");
            }
            // the interpolation's precision bound: N = K + m - 2 weights, non-consecutive weights, values mod ell^{K+m-2}: refused
            auto refused = [](const std::function<void()>& f) {
                try {
                    f();
                } catch (const std::logic_error& ex) {
                    return std::string(ex.what()).find("REFUSED") != std::string::npos;
                }
                return false;
            };
            const std::vector<uint32_t> four(interp.begin(), interp.begin() + 4);
            std::vector<uint32_t> gap = interp;
            gap[2] += static_cast<uint32_t>(l - 1);
            const L::Vec v5(5, 1), v4(4, 1);
            Check(refused([&] { MS::MakeTruncationMap(l, four, kKp, kK, 3, cs.k0); }) &&
                      refused([&] { L::InterpolateTruncation(l, four, v4, kKp, kK, 3, cs.k0); }),
                  "NEG " + ctag + ": N = K + m - 2 = 4 weights at m = 3 REFUSED (both maps)");
            Check(refused([&] { MS::MakeTruncationMap(l, gap, kKp, kK, 3, cs.k0); }) &&
                      refused([&] { L::InterpolateTruncation(l, gap, v5, kKp, kK, 3, cs.k0); }),
                  "NEG " + ctag + ": non-consecutive weights REFUSED (both maps)");
            Check(refused([&] { MS::MakeTruncationMap(l, interp, kK + 1, kK, 3, cs.k0); }) &&
                      refused([&] { L::InterpolateTruncation(l, interp, v5, kK + 1, kK, 3, cs.k0); }),
                  "NEG " + ctag + ": values mod ell^{K+m-2} at m = 3 REFUSED (both maps)");
        }
    }
    // ---- G11: (U17)(a) j_k = L_k at m_Eis
    if (eis_cell) {
        for (size_t wi = 0; wi < ks.size(); ++wi) {
            const uint32_t k = ks[wi], n = k - 2, Lk = Lks[wi];
            const MS::GluingReading& gd = glue[wi];
            const bool proved = n < l - 1;
            const std::string tag = "G11 " + ctag + " k=" + std::to_string(k) + ": j_k = L_k = " + std::to_string(Lk) + " on H^1_+ at m_Eis";
            if (gd.refusal.empty()) {
                const std::string txt = tag + " (dual-D route: j=" + std::to_string(gd.j) + ", v(a_2-e_2)=" + std::to_string(gd.v_a_minus_e) +
                                        ", pivot=" + std::to_string(gd.pivot) + ")";
                if (proved)
                    Check(gd.j == Lk, txt);
                else
                    Finding(gd.j == Lk, txt);
            } else {
                Check(false, tag + ": dual-D route refused '" + gd.refusal + "'");
            }
            std::cout << "  GLUING " << ctag << " k=" << k << " L_k=" << Lk << " j_dual=" << gd.j << " (block " << gd.block_rank << ", ordinary "
                      << gd.ord_rank << ") d_V=" << jk[wi] << " wall_s=" << gd.seconds << std::endl;
            if (proved) {
                const MS::GluingReading ga = MS::GluingAmalgam(n, l, Lk + 3);
                Check(ga.refusal.empty() && ga.eis_in_plus, "G11 " + ctag + " k=" + std::to_string(k) +
                                                                ": on H^1(Gamma; V) = V/(V^S+V^U) the ordinary m_Eis block is rank 2 on [C]-plus, 1 on [C]-minus ('" +
                                                                ga.refusal + "')");
                Check(ga.refusal.empty() && ga.j == Lk && ga.j == ga.v_a_minus_e - ga.pivot,
                      "G11 " + ctag + " k=" + std::to_string(k) + ": j_k = L_k on H^1 directly (amalgam route: j=" + std::to_string(ga.j) +
                          ", v(a-e) - pivot = " + std::to_string(ga.v_a_minus_e) + " - " + std::to_string(ga.pivot) + ")");
                Check(ga.shriek_in_cusp_line, "G11 " + ctag + " k=" + std::to_string(k) + ": iota_! of the [C]-plus cusp symbol lies on the cusp line A_1");
            }
        }
    }
    std::cout << "CELL_DONE ell=" << l << " a=" << cs.a << " wall_s=" << Now() - tc << " peak_rss_mib=" << PeakRssMiB() << std::endl;
}


// ===========================================================================
// G13: ell in {5, 7} -- S_k = 0, H^+ = the Eisenstein line, H^- = 0, every class (thm:u-ell13)
// ===========================================================================
void ArmSmallEll() {
    for (uint64_t l : {5ull, 7ull}) {
        std::vector<uint64_t> primes;
        for (uint64_t p = 2; p <= kB; ++p)
            if (IsPrime(p) && p != l) primes.push_back(p);
        for (uint32_t a = 0; a < l - 1; a += 2) {
            uint32_t ka = 12;
            while ((ka + 1000 * (l - 1) - a) % (l - 1)) ++ka;
            for (uint32_t j = 0; j < 3; ++j) {
                const uint32_t k = ka + 2 * j * static_cast<uint32_t>(l - 1), n = k - 2;
                const std::string tag = "G13 " + CellTag(l, a) + " k=" + std::to_string(k);
                const L::FormsWeight fw = L::ComputeFormsWeight(l, k, kKp, primes);
                MS::SignSpace Sm = MS::BuildSignSpace(l, n, -1, kKp, 2), Sp = MS::BuildSignSpace(l, n, 1, kKp, 2);
                const uint32_t om = MS::OrdinaryRankAll(Sm), op = MS::OrdinaryRankAll(Sp);
                // the m_Eis T_2-part of W^{[C]-}: ordinary part = Z p_0
                const uint64_t e2 = (1 + bm::MakeRing(l, 1).pow(2, k - 1)) % l;
                const MS::OrdBlock B = MS::OrdinaryPart(Sm, k, 2, {(l - e2) % l, 1}, true);
                const auto hr = MS::AmalgamOrdinaryRanks(n, l, kKp);
                Check(fw.S.s == 0 && om == 1 && op == 0 && B.has_p0 && B.ord_rank == 1 && hr.first == 1 && hr.second == 0,
                      tag + ": e S_k = 0 (forms), e_1 W^{[C]-} = Z p_0, e_1 W^{[C]+} = 0, e_1 H^1_+ = the Eisenstein line, e_1 H^1_- = 0 (" +
                          std::to_string(fw.S.s) + "; " + std::to_string(om) + "," + std::to_string(op) + "; " + std::to_string(hr.first) + "," +
                          std::to_string(hr.second) + ")");
            }
        }
    }
}

// ===========================================================================
// G4: level ell -- e_ell H^1(Gamma_0(ell); Sym^n (x) Z/ell^{K'})_m has length 2 r K', T_{p0} has charpoly F^2,
// U^{(ell)} = alpha(x_k, T_{p0}) on it (both signs together), on the slot-13 level-ell code
// ===========================================================================
struct LevelModel {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
};
LevelModel& TheLevelModel() {
    static LevelModel m;
    static std::once_flag once;
    std::call_once(once, [] {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        m.K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        m.tess.reset(new hecke::SL2ZTessellationTables(*m.K));
    });
    return m;
}

void ArmLevelEll() {
    using namespace hecke::level;
    for (const LevelTarget& t : g_level_targets) {
        const double t0 = Now();
        const std::string tag = "G4 " + CellTag(t.ell, 0) + " k=" + std::to_string(t.k);
        try {
            const uint32_t n = t.k - 2;
            const engine::ModCtx<uint32_t> ctx = hecke::CoefficientRingContext<uint32_t>("HeckeLambdaModSymValidation", t.ell, kKp);
            LevelModel& M = TheLevelModel();
            std::unique_ptr<hecke::HeckeCosetTables> cU, cT;
            std::unique_ptr<hecke::HeckeTauTable> tU, tT;
            {
                std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
                cU.reset(new hecke::HeckeCosetTables(t.ell, 1, static_cast<uint32_t>(t.ell), n));
                tU.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *cU, static_cast<uint32_t>(t.ell), n));
                cT.reset(new hecke::HeckeCosetTables(t.p0, 1, static_cast<uint32_t>(t.ell), n));
                tT.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *cT, static_cast<uint32_t>(t.ell), n));
            }
            ENGINE_CHECK_MSG(!tU->used_mod_ell_solve(), "G4: the U_ell tau used the mod-ell fallback (not integral)");
            const Gamma0Level lvl(*cU);
            std::unique_ptr<LocalSystemSimplicialComplex> Ll;
            std::unique_ptr<EquivariantCohomologyOnK> Hl;
            {
                const hecke::ScopedCoefficientRing ring(ctx.k);
                const uint32_t Dd = static_cast<uint32_t>((t.ell + 1) * (n + 1));
                Ll.reset(new LocalSystemSimplicialComplex(*M.K, InducedRelationsRing(*M.K, *cU, n, ctx), Dd, ctx.p));
                Hl.reset(new EquivariantCohomologyOnK(Ll.get()));
                Hl->compute_cohomology();
            }
            const HFrame fl(Hl.get(), 1);
            CorrespondenceSpec sU{&lvl, &lvl, cU.get(), tU.get(), Gamma0SubTransversal(t.ell, t.ell), "U_ell^(ell)"};
            VerifyTransversal(sU, static_cast<uint32_t>(t.ell));
            CorrespondenceSpec sT{&lvl, &lvl, cT.get(), tT.get(), Gamma0SubTransversal(t.ell, t.p0), "T_p0^(ell)"};
            VerifyTransversal(sT, static_cast<uint32_t>(t.p0 + 1));
            const HMat U = InducedOnH(fl, fl, BuildLevelCorrespondence(ctx, sU, *M.K, n, 1), "U_ell^(ell)");
            const HMat T = InducedOnH(fl, fl, BuildLevelCorrespondence(ctx, sT, *M.K, n, 1), "T_p0^(ell)");
            auto toRing = [](const HMat& A) {
                hecke::ordinary::RingMat R;
                R.ell = A.ell;
                R.row_exp = A.row_exp;
                R.a = A.a;
                return hecke::ordinary::Canonical(R);
            };
            const hecke::ordinary::RingMat Ur = toRing(U), Tr = toRing(T);
            const hecke::ordinary::RingMat P = hecke::ordinary::OrdinaryProjector(Ur);
            const hecke_ext::hida::FreeImage FI = hecke_ext::hida::MakeFreeImage(P, kKp);
            const bm::Ring R = bm::MakeRing(t.ell, kKp);
            auto toS = [&](const hecke_ext::hida::Mat& A) {
                SMat B(static_cast<uint32_t>(A.size()), static_cast<uint32_t>(A.size()));
                for (uint32_t i = 0; i < B.r; ++i)
                    for (uint32_t j = 0; j < B.c; ++j) B.at(i, j) = A[i][j] % R.q;
                return B;
            };
            const SMat Uo = toS(hecke_ext::hida::MatrixOnImage(FI, Ur)), To = toS(hecke_ext::hida::MatrixOnImage(FI, Tr));
            // the m-part: ker g(T_{p0})^N on the free ordinary module
            const uint32_t h = To.r;
            uint64_t e = 1;
            while (e < static_cast<uint64_t>(kKp) * std::max<uint32_t>(h, 1)) e <<= 1;
            std::vector<SVec> kg;
            std::vector<uint32_t> ke;
            bm::KernelGenerators(R, MS::MatPow(R, MS::MatPolyOf(R, SVec(t.g.begin(), t.g.end()), To), e), kg, ke);
            bool free = true;
            for (uint32_t x : ke) free = free && x == kKp;
            const uint32_t rk = static_cast<uint32_t>(kg.size());
            bool charpoly_ok = false, alpha_ok = false, alpha_trunc_ok = false;
            if (free && rk > 0) {
                const MS::FreeBasis V = MS::MakeFreeBasis(R, kg);
                auto restrict = [&](const SMat& A) {
                    SMat C(rk, rk);
                    for (uint32_t j = 0; j < rk; ++j) {
                        SVec c;
                        ENGINE_CHECK_MSG(V.coords(R, bm::MatVec(R, A, V.b[j]), c), "G4: the m-part is not stable");
                        for (uint32_t i = 0; i < rk; ++i) C.at(i, j) = c[i];
                    }
                    return C;
                };
                const SMat Tm = restrict(To), Um = restrict(Uo);
                // F^2
                SVec F2(2 * t.F.size() - 1, 0);
                for (size_t i = 0; i < t.F.size(); ++i)
                    for (size_t j = 0; j < t.F.size(); ++j) F2[i + j] = R.add(F2[i + j], R.mul(t.F[i], t.F[j]));
                charpoly_ok = bm::CharPoly(R, Tm) == F2;
                alpha_ok = MS::MatEq(MS::MatPolyOf(R, t.alpha, Tm), Um);
                const uint64_t qe = PowU(t.ell, t.specPrec);
                const L::Vec as = L::SpecialiseAt(t.alphaTrunc, t.ell, t.centre, t.k, t.specPrec);
                SVec asv;
                for (u128 x : as) asv.push_back(static_cast<uint64_t>(x % qe));
                alpha_trunc_ok = MS::MatEq(MS::Reduce(MS::MatPolyOf(R, asv, Tm), qe), MS::Reduce(Um, qe));
            }
            Check(free && rk == 2 * t.r, tag + ": e_ell H^1(Gamma_0(ell); Sym^n (x) Z/ell^5)_m free of rank 2r = " + std::to_string(2 * t.r) +
                                             " (read " + std::to_string(rk) + ", e_ell H^1 rank " + std::to_string(h) + ")");
            Check(charpoly_ok, tag + ": charpoly(T_p0^(ell) | m-part) == F(x_k, Y)^2 (the forms data at k) mod ell^5");
            Check(alpha_ok, tag + ": U_ell^(ell) == alpha(x_k, T_p0^(ell)) on the m-part mod ell^5 (the forms data at k)");
            Check(alpha_trunc_ok, tag + ": U_ell^(ell) == P7's truncated alpha(x_k, T_p0) mod ell^" + std::to_string(t.specPrec));
            std::cout << "  LEVEL_ELL ell=" << t.ell << " k=" << t.k << " ordinary_rank=" << h << " m_rank=" << rk << " wall_s=" << Now() - t0
                      << " peak_rss_mib=" << PeakRssMiB() << std::endl;
        } catch (const std::exception& ex) {
            Check(false, tag + " threw: " + ex.what());
        }
    }
}

// ===========================================================================
// G14: lem:u-latticeord -- D_* an isomorphism on e_1 H^1 over Z/ell^3 (Hida's lattice against the contract's),
// and the five d_f = 1 eigenlines of (40, 37) ([E] rmk:lvlatticevalues) non-ordinary
// ===========================================================================
void ArmLattice() {
    struct LC {
        uint32_t n;
        uint64_t ell;
    };
    for (const LC& c : std::vector<LC>{{10, 5}, {12, 5}, {24, 7}, {30, 11}, {40, 37}}) {
        const double t0 = Now();
        const std::string tag = "G14 (" + std::to_string(c.n) + "," + std::to_string(c.ell) + ")";
        try {
            std::vector<uint64_t> primes{c.ell};
            for (uint64_t p : {2ull, 3ull, 5ull, 7ull})
                if (p != c.ell) primes.push_back(p);
            const hecke_ext::hida::LatticePair lp = hecke_ext::hida::RunLatticePair(c.n, c.ell, 3, primes);
            const hecke_ext::hida::DstarReading ds = hecke_ext::hida::ReadDstar(lp, 3);
            Check(lp.Dstar_equivariant && ds.image_in_ordinary && ds.isomorphism,
                  tag + ": D_* : e_1 H^1(Gamma; L_n) -> e_1 H^1(Gamma; Sym^n) an isomorphism over Z/ell^3 (h=" + std::to_string(ds.h) +
                      ", rank mod ell " + std::to_string(ds.rank_mod_ell) + ")");
            std::cout << "  LATTICE n=" << c.n << " ell=" << c.ell << " h=" << ds.h << " iso=" << ds.isomorphism << " wall_s=" << Now() - t0 << std::endl;
        } catch (const std::exception& ex) {
            Check(false, tag + " threw: " + ex.what());
        }
    }
    // (40, 37): the eigenlines with d_f >= 1 (WP-LV3's change of lattice) lie in T_2-blocks whose ordinary part avoids them
    try {
        const double t0 = Now();
        const hp::HidaPairingCell C(40, 37, 6, {2});
        uint32_t lines = 0, nonord = 0;
        std::ostringstream os;
        for (uint64_t a = 0; a < 37; ++a) {
            hecke_ext::lvalue::mblock::MBlockOptions opt;
            opt.hd_torsion = false;
            opt.lattice_change = true;
            opt.primes = {2};
            const auto B = hecke_ext::lvalue::mblock::ReadMBlockCell(C, a, opt);
            for (int s = 0; s < 2; ++s) {
                uint32_t here = 0;
                for (const auto& f : B.sign[s].forms)
                    if (f.have_lattice && f.d_lattice > 0) ++here;
                if (!here) continue;
                lines += here;
                // the T_2-block of W at a (geo-even = [C]-minus, geo-odd = [C]-plus): T_37 nilpotent mod 37 on it
                MS::SignSpace S = MS::BuildSignSpace(37, 40, s == 0 ? -1 : 1, kKp, 2);
                const MS::OrdBlock Bl = MS::OrdinaryPart(S, 42, 2, {(37 - a) % 37, 1}, false);
                const bool zero = Bl.refusal == "ordinary_part_zero";
                if (zero) nonord += here;
                os << " (a=" << a << "," << (s == 0 ? "geo-even" : "geo-odd") << ",lines=" << here << ",g_rank=" << Bl.g_rank
                   << ",ord_rank=" << Bl.ord_rank << ")";
            }
        }
        std::cout << "  LATTICE_NEG (40,37) d_f>=1 eigenlines=" << lines << os.str() << " wall_s=" << Now() - t0 << std::endl;
        Check(lines == 5 && nonord == lines, "G14 (40,37): the " + std::to_string(lines) +
                                                 " eigenlines with d_f >= 1 ([E] rmk:lvlatticevalues) are non-ordinary (T_37 nilpotent on their T_2-blocks)");
    } catch (const std::exception& ex) {
        Check(false, std::string("G14 (40,37) threw: ") + ex.what());
    }
}
}  // namespace

uint32_t HeckeLambdaModSymValidation() {
    const double t_start = Now();
    g_check.reset();
    g_finding.reset();
    g_by_gate.clear();
    g_level_targets.clear();
    g_level_full = false;
    std::vector<CellSpec> cells{{11, 2, 12}, {13, 0, 12}, {23, 2, 24}, {37, 32, 32}};
    uint32_t Fw = 3;
    std::string arm = "DEFAULT";
    if (const char* e = std::getenv("HECKE_LAMBDA_MSYM_ALL"))
        if (std::string(e) == "1") {
            arm = "ALL";
            Fw = 5;
            cells.push_back({103, 24, 24, 3});   // weights to n = 736 (P8_spec Sec. 2)
            g_level_full = true;
        }
    if (const char* e = std::getenv("HECKE_LAMBDA_MSYM_CELLS")) {
        arm = "CELLS";
        cells.clear();
        bool bad = false;
        for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(e)) {
            const size_t pos = t.pos;
            const std::string& tok = t.text;
            const bool fits = t.Fits(3, 3);
            const uint64_t a = fits ? t.fields[0] : 0, b = fits ? t.fields[1] : 0, c = fits ? t.fields[2] : 0;
            if (!fits || !IsPrime(a) || a < 5 || b % 2 || c < 12 ||
                (c + 1000 * (a - 1) - b) % (a - 1) != 0) {
                std::cout << "  HECKE_LAMBDA_MSYM_CELLS: token " << pos << " '" << tok << "' REFUSED (want ell:a:k0, ell >= 5 prime, a even, k0 >= 12 in the class)"
                          << std::endl;
                bad = true;
                continue;
            }
            cells.push_back({a, static_cast<uint32_t>(b), static_cast<uint32_t>(c)});
        }
        Check(!bad && !cells.empty(), "ENV HECKE_LAMBDA_MSYM_CELLS selects at least one cell and every token is well formed");
    }
    std::cout << "EFFECTIVE HeckeLambdaModSymValidation arm=" << arm << " cells=" << cells.size() << " K=" << kK << " Kp=" << kKp
              << " m=1..3 N=" << kN << " F=" << Fw << " B=" << kB << " M=" << kM << std::endl;

    // the fast Hecke of HeckeLambdaModSym against WP-LIN's HeckeTp (the symbol oracle)
    {
        bool ok = true;
        for (uint32_t n : {10u, 30u, 66u}) {
            const bm::Ring R = bm::MakeRing(37, 5), R8 = bm::MakeRing(37, 8);
            for (const bm::Ring* RR : {&R, &R8}) {
                const bm::SymRho rho(n, *RR);
                const MS::SymbolHecke H(n, *RR);
                std::mt19937_64 rng(n);
                SVec v(n + 1);
                for (auto& x : v) x = rng() % RR->q;
                for (uint64_t p : {2ull, 5ull, 29ull, 37ull}) ok = ok && H.Tp(v, p) == bm::HeckeTp(rho, v, p);
                for (const bm::Mat2& A : {bm::Mat2{3, 5, 7, 12}, bm::Mat2{37, 2, 0, 1}, bm::Mat2{0, -1, 1, 0}, bm::Mat2{4, 37, 1, 0}})
                    ok = ok && H.apply(A, v) == rho.apply(A, v);
            }
        }
        Check(ok, "ORACLE the fast symbol Hecke (Taylor-shift rho_n) == WP-LIN's HeckeTp / SymRho::apply (n = 10, 30, 66; mod 37^5, 37^8)");
    }
    const auto catalogue = TowerCatalogue();
    Check(catalogue.size() > 100, "ORACLE the tower catalogue hida_towers.tsv read (" + std::to_string(catalogue.size()) + " (ell, n) cells)");

    for (const CellSpec& cs : cells) {
        try {
            RunCell(cs, cs.F ? cs.F : Fw, catalogue);
        } catch (const std::exception& ex) {
            Check(false, "CELL " + CellTag(cs.ell, cs.a) + " threw: " + ex.what());
        }
    }

    {
        const double ta = Now();
        ArmSmallEll();
        std::cout << "ARM G13 wall_s=" << Now() - ta << std::endl;
        const double tb = Now();
        ArmLevelEll();
        std::cout << "ARM G4 targets=" << g_level_targets.size() << " wall_s=" << Now() - tb << std::endl;
        const double tl = Now();
        ArmLattice();
        std::cout << "ARM G14 wall_s=" << Now() - tl << std::endl;
    }
    std::cout << "HeckeLambdaModSymValidation: checks=" << g_check.checks << " failing=" << g_check.fails << " findings=" << g_finding.checks
              << " finding_failures=" << g_finding.fails << " wall_s=" << Now() - t_start << " peak_rss_mib=" << PeakRssMiB() << std::endl;
    for (const auto& kv : g_by_gate) std::cout << "  gate " << kv.first << ": " << kv.second << std::endl;
    return g_check.fails == 0 ? 0 : 1;
}
