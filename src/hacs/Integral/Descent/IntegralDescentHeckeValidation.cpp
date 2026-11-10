// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D3: IntegralDescentHeckeValidation / IntegralDescentHeckeSlowValidation.  [C] = docs/latex/gl2_contract.tex
// (v1.23.0-pre.31), sec:descenthecke.  The oracle d3_hecke_oracle.tsv is produced by d3_hecke_oracle.py, which imports the C-D4 referee's
// pure-Python modules (group cocycles of Gamma_0(N) over Z) and shares nothing with cohomology_app.  See the header for the arms.
#include "IntegralDescentHeckeValidation.h"

#include "DescentAssembly.h"
#include "DescentHecke.h"
#include "DescentLevelN.h"
#include "DescentMaps.h"
#include "DescentPair.h"
#include "DescentTree.h"

#include "../IntegralCharpolyCrt.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../Hecke/ExactInt.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeOperatorFamily.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace H3 = integral_descent::hecke3;
namespace P = integral_descent::pair;
using integral_descent::maps::Mat;
using integral_descent::maps::ZMod;
using U32Ctx = engine::ModCtx<uint32_t>;
using clk = std::chrono::steady_clock;

uint64_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 80) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string fx(double x, int p = 4) {
    std::ostringstream o;
    o << std::setprecision(p) << x;
    return o.str();
}
std::string mib(uint64_t b) { return std::to_string(b >> 20); }
std::string cellname(uint32_t N, uint32_t n) { return "(N,n)=(" + std::to_string(N) + "," + std::to_string(n) + ")"; }
std::string join(const std::vector<uint32_t>& v, const char* sep = "/") {
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); ++i) o << (i ? sep : "") << v[i];
    return o.str();
}

// ---- the oracle ---------------------------------------------------------------------------------------------------------------
struct Oracle {
    bool found = false;
    std::map<std::array<uint32_t, 3>, std::vector<std::string>> chi;            // (N, n, p) -> coefficients (decimal), high -> low
    std::map<std::array<uint32_t, 5>, std::string> tors;                         // (N, n, q, l, p) -> canonical invariants
    std::vector<std::array<uint32_t, 2>> cells;                                  // in file order
};
std::string src_path(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}
const Oracle& TheOracle() {
    static Oracle O;
    static std::once_flag once;
    std::call_once(once, [] {
        std::ifstream in(src_path("docs/implementation/slot24/d3_hecke_oracle.tsv"));
        if (!in) return;
        O.found = true;
        std::string line;
        std::set<std::array<uint32_t, 2>> seen;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::string tag;
            std::getline(ls, tag, '\t');
            std::vector<std::string> f;
            std::string x;
            while (std::getline(ls, x, '\t')) f.push_back(x);
            if (tag == "CHI" && f.size() == 4) {
                const std::array<uint32_t, 3> k{uint32_t(std::stoul(f[0])), uint32_t(std::stoul(f[1])), uint32_t(std::stoul(f[2]))};
                std::vector<std::string> cs;
                std::istringstream cc(f[3]);
                while (std::getline(cc, x, ',')) cs.push_back(x);
                O.chi[k] = cs;
                if (seen.insert({k[0], k[1]}).second) O.cells.push_back({k[0], k[1]});
            } else if (tag == "TORS" && f.size() == 6) {
                O.tors[{uint32_t(std::stoul(f[0])), uint32_t(std::stoul(f[1])), uint32_t(std::stoul(f[2])), uint32_t(std::stoul(f[3])),
                        uint32_t(std::stoul(f[4]))}] = f[5];
            }
        }
    });
    return O;
}
// a signed decimal mod Q (Q = 0: 2^64), Horner with the word mulmod
uint64_t DecimalModQ(const std::string& s, uint64_t Q) {
    bool neg = !s.empty() && s[0] == '-';
    uint64_t acc = 0;
    for (size_t i = neg ? 1 : 0; i < s.size(); ++i) {
        const uint64_t dgt = static_cast<uint64_t>(s[i] - '0');
        acc = H3::WordMulMod(Q ? acc % Q : acc, 10, Q);
        acc = Q ? (acc + dgt) % Q : acc + dgt;
    }
    if (!neg || acc == 0) return acc;
    return Q ? Q - acc : static_cast<uint64_t>(0) - acc;
}
// compare a CharpolyZ with the oracle row: integers where they fit; the big ones by c mod 2^64 and mod 3^40
bool SameAsOracle(const H3::CharpolyZ& C, const std::vector<std::string>& o, std::string& why) {
    if (C.c.size() != o.size()) {
        why = "degree " + std::to_string(C.r) + " vs the oracle's " + std::to_string(o.size() - 1);
        return false;
    }
    uint64_t Q340 = 1;
    for (int i = 0; i < 40; ++i) Q340 *= 3;
    for (size_t i = 0; i < o.size(); ++i) {
        const auto& x = C.c[i];
        if (!x.certified) {
            why = "c_" + std::to_string(i) + " not certified";
            return false;
        }
        if (x.fits) {
            if (std::to_string(x.value) != o[i]) {
                why = "c_" + std::to_string(i) + " = " + std::to_string(x.value) + " vs " + o[i];
                return false;
            }
        } else if (x.mod_2_64 != DecimalModQ(o[i], 0) || x.mod_3_40 != DecimalModQ(o[i], Q340)) {
            why = "c_" + std::to_string(i) + " (beyond 2^63) differs mod 2^64 / 3^40 from " + o[i];
            return false;
        }
    }
    return true;
}
// CharpolyZ::digest's definition re-stated on the oracle's decimal integers (output rule (5); the D3 audit's m1): r, then per i: i, 1 + negative,
// c_i mod 2^64, c_i mod 3^40
uint64_t OracleDigest(const std::vector<std::string>& o) {
    uint64_t Q340 = 1;
    for (int i = 0; i < 40; ++i) Q340 *= 3;
    engine::FnvHasher h;
    h.feed(o.size() - 1);
    for (size_t i = 0; i < o.size(); ++i) {
        h.feed(i);
        h.feed(1u + (!o[i].empty() && o[i][0] == '-' ? 1u : 0u));
        h.feed(DecimalModQ(o[i], 0));
        h.feed(DecimalModQ(o[i], Q340));
    }
    return h.value();
}

// ---- the shipped level-N route on the 7-vertex model (eq:levelcorr, HFrame): the oracle of arm (S) -----------------------------
struct ShippedModel {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
};
ShippedModel& TheShippedModel() {
    static ShippedModel m;
    static std::once_flag once;
    std::call_once(once, [] {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        m.K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        m.tess.reset(new hecke::SL2ZTessellationTables(*m.K));
    });
    return m;
}
// char T_p on H^1 over F_l' by the dense shipped route: InducedRelationsLevel -> EquivariantCohomologyOnK -> BuildLevelCorrespondence with
// the shipped canonical tau -> InducedOnH (the HeckeLevelNValidation pipeline), high -> low.
std::vector<uint32_t> DenseShippedCharpoly(uint32_t N, uint32_t n, uint32_t p, uint32_t ellp) {
    using namespace hecke::level;
    ShippedModel& M = TheShippedModel();
    const U32Ctx ctx(ellp);
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        cosets.reset(new hecke::HeckeCosetTables(p, 1, ellp, n));
        tau.reset(new hecke::HeckeTauTable(*M.K, *M.tess, *cosets, ellp, n));
    }
    const Gamma0LevelN lvl(N), lvlp(uint64_t(N) * p);
    const uint32_t D = lvl.m() * (n + 1);
    LocalSystemSimplicialComplex L(*M.K, InducedRelationsLevel(*M.K, lvl, n, ctx), D, ctx.p);
    EquivariantCohomologyOnK Hc(&L);
    Hc.compute_cohomology();
    const HFrame f(&Hc, 1);
    CorrespondenceSpec s{&lvl, &lvl, cosets.get(), tau.get(), Gamma0SubTransversalN(lvl, lvlp), "T_p^(N)"};
    VerifyTransversal(s, p + 1);
    const HMat T = InducedOnH(f, f, BuildLevelCorrespondence(ctx, s, *M.K, n, 1), s.label);
    std::vector<uint32_t> A(size_t(T.rows) * T.cols);
    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<uint32_t>(T.a[i] % ellp);
    return H3::CharpolyFp(std::move(A), T.rows, ellp);
}
// char T~_p on H^1(C_I) over F_l' (the new operator; the free part of BuildPres over the field)
std::vector<uint32_t> IsoRouteCharpoly(uint32_t N, uint32_t n, uint32_t p, uint32_t ellp, Mat* Fout = nullptr) {
    // over Z/l'^2 (BuildPres's precision rule needs k >= 2; no l'-torsion, so the free part read mod l' is exact)
    const U32Ctx ctx = U32Ctx::prime_power(ellp, 2);
    const ZMod Z(ellp, 2);
    const P::PairComplex X = P::BuildFreePairN(N, n, ctx, 2);
    const integral_descent::LevelNCounts cnt = integral_descent::LevelNClosedForm(N, n);
    const uint32_t b0 = n % 2 ? 0u : cnt.b0, b1 = n % 2 ? 0u : cnt.b1;
    const integral_descent::maps::Pres H1 = integral_descent::maps::BuildPres(Z, X.d[0], X.c[0] - b0, X.d[1], X.c[1], b1);
    ENGINE_CHECK_MSG(H1.ok && H1.free == b1 && H1.tors.empty(), ("IsoRouteCharpoly: H^1 presentation " + H1.why).c_str());
    const H3::IsoHeckeN T(N, n, p, ctx, 1);
    ENGINE_CHECK_MSG(!T.refused(), T.why().c_str());
    const Mat Td = P::FromDenseBuf(T.dense(1));
    ZMod zo;
    const Mat F = P::InducedMatrix(H1, H1, Td, zo);
    if (Fout) *Fout = F;
    std::vector<uint32_t> a = F.a;
    for (auto& x : a) x %= ellp;
    return H3::CharpolyFp(std::move(a), F.r, ellp);
}
// the integers of a certified CharpolyZ reduced mod l' (fits only)
std::vector<uint32_t> ResidueOf(const H3::CharpolyZ& C, uint32_t ellp) {
    std::vector<uint32_t> out;
    for (const auto& x : C.c) {
        if (!x.fits) return {};
        int64_t v = x.value % static_cast<int64_t>(ellp);
        if (v < 0) v += ellp;
        out.push_back(static_cast<uint32_t>(v));
    }
    return out;
}

// =================================================================================================================================
// (O) the oracle
// =================================================================================================================================
struct OracleStats {
    uint32_t chi = 0, chi_cells = 0, tors_rows = 0, groups = 0, empty_ok = 0, digests = 0, prime_sets = 0;
};
void ArmOracle(OracleStats& st, bool slow) {
    const Oracle& O = TheOracle();
    if (!check(O.found, "(O) the oracle d3_hecke_oracle.tsv is found at " + src_path("docs/implementation/slot24/d3_hecke_oracle.tsv"))) return;
    for (const auto& cell : O.cells) {
        const uint32_t N = cell[0], n = cell[1];
        const uint32_t D = integral_descent::PsiIndex(N) * (n + 1);
        if (!slow && D > 120) continue;
        std::vector<uint32_t> ps;
        for (const auto& kv : O.chi)
            if (kv.first[0] == N && kv.first[1] == n) ps.push_back(kv.first[2]);
        // char_Z, every p
        for (uint32_t p : ps) {
            const H3::CharpolyZ C = H3::CharpolyOverZ(N, n, p);
            std::string why;
            const bool ok = !C.refused && C.determined && SameAsOracle(C, O.chi.at({N, n, p}), why);
            check(ok, "(O) char_Z(T_" + std::to_string(p) + " | H^1/tors) " + cellname(N, n) + " = the oracle: " + (C.refused ? C.refusal : why));
            ++st.chi;
            check(!C.refused && C.digest() == OracleDigest(O.chi.at({N, n, p})) && C.undetermined() == 0,
                  "(O) the output-rule digest of char_Z(T_" + std::to_string(p) + ") " + cellname(N, n) + " = the oracle's integers' digest: " + C.text());
            ++st.digests;
        }
        ++st.chi_cells;
        // torsion: l = 2, 3 at q = 1, 2 on the tree (every cell: an empty group must match an absent row), l >= 5 at q = 1
        std::set<uint32_t> ells = {2, 3}, oracle_ells;
        for (const auto& kv : O.tors)
            if (kv.first[0] == N && kv.first[1] == n) ells.insert(kv.first[3]), oracle_ells.insert(kv.first[3]);
        // (the D3 audit's N2) the record's own torsion primes (RecordTorsionPrimes: 2, 3 for n even, and B(n, N)) contain every oracle torsion
        // prime, and its l >= 5 primes are exactly the oracle's
        {
            const std::vector<uint32_t> rec = H3::RecordTorsionPrimes(N, n);
            const std::set<uint32_t> recs(rec.begin(), rec.end());
            std::set<uint32_t> rec5, or5;
            for (uint32_t l : rec)
                if (l >= 5) rec5.insert(l);
            bool cover = true;
            for (uint32_t l : oracle_ells) {
                cover = cover && recs.count(l) > 0;
                if (l >= 5) or5.insert(l);
            }
            check(cover && rec5 == or5, "(O) the record's torsion primes " + join(rec, ",") + " contain the oracle's, equal at l >= 5 " + cellname(N, n));
            ++st.prime_sets;
        }
        for (uint32_t ell : ells)
            for (uint32_t q : {1u, 2u}) {
                if (ell >= 5 && q == 2) continue;
                const H3::TorsionEntry E = ell <= 3 ? H3::TorsionOnTree(ell, N, n, q, ps) : H3::TorsionOnCokernel(ell, N, n, ps);
                const std::string tag = " l=" + std::to_string(ell) + " q=" + std::to_string(q) + " " + cellname(N, n);
                if (!check(!E.refused && E.read.ok, "(O) the read" + tag + ": " + E.refusal)) continue;
                const bool has = O.tors.count({N, n, q, ell, ps.front()}) > 0;
                if (E.read.sigma.empty()) {
                    check(!has, "(O) no torsion" + tag + " -- and the oracle has none");
                    ++st.empty_ok;
                    continue;
                }
                ++st.groups;
                check(E.read.module_certified, "(O) the module is certified (2 sigma_max <= K)" + tag);
                check(E.commute, "(O) the T_p commute on the group" + tag);
                for (size_t i = 0; i < ps.size(); ++i) {
                    const auto it = O.tors.find({N, n, q, ell, ps[i]});
                    const std::string ours = i < E.inv.size() ? E.inv[i].canon : std::string("(missing)");
                    check(it != O.tors.end() && it->second == ours,
                          "(O) the torsion Hecke module" + tag + " p=" + std::to_string(ps[i]) + ": ours " + ours + " vs the oracle " +
                              (it == O.tors.end() ? std::string("(none)") : it->second));
                    ++st.tors_rows;
                }
            }
    }
    std::cout << "  (O) the oracle: " << st.chi << " char_Z on " << st.chi_cells << " cells equal (" << st.digests << " output-rule digests equal); "
              << st.groups << " torsion groups (" << st.tors_rows << " (group, p) rows) equal in every invariant; " << st.empty_ok
              << " empty groups confirmed; " << st.prime_sets << " torsion-prime sets equal" << std::endl;
}

// =================================================================================================================================
// (S) the residues against the shipped mod-p Hecke
// =================================================================================================================================
void ArmShipped(bool slow) {
    struct C3 {
        uint32_t N, n;
    };
    std::vector<C3> cells = {{1, 10}, {1, 12}, {1, 22}, {11, 0}, {2, 6}, {3, 4}, {5, 4}, {7, 2}, {13, 2}, {6, 2}};
    if (slow) cells.insert(cells.end(), {{1, 40}, {1, 60}, {23, 0}, {37, 2}, {10, 4}, {11, 6}, {4, 10}});
    const std::vector<uint32_t> goods = {10007, 1000003, 1073741789};
    uint32_t n_ok = 0;
    for (const auto& c : cells)
        for (uint32_t p : {2u, 5u, 7u, 13u}) {
            if (c.N % p == 0) continue;
            const H3::CharpolyZ Z = H3::CharpolyOverZ(c.N, c.n, p);
            if (!check(!Z.refused && Z.determined, "(S) char_Z " + cellname(c.N, c.n) + " p=" + std::to_string(p) + ": " + Z.refusal)) continue;
            for (uint32_t ellp : goods) {
                if (ellp <= c.n + 1) continue;
                uint32_t h1 = 0;
                const std::vector<uint32_t> prod = H3::ShippedResidueCharpoly(c.N, c.n, p, ellp, &h1);
                const std::string at = cellname(c.N, c.n) + " p=" + std::to_string(p) + " l'=" + std::to_string(ellp);
                const std::vector<uint32_t> zres = ResidueOf(Z, ellp);
                if (!zres.empty()) check(prod == zres, "(S) the production residue = char_Z mod l' at " + at);
                if (ellp == 10007) {
                    const std::vector<uint32_t> iso = IsoRouteCharpoly(c.N, c.n, p, ellp);
                    check(iso == prod, "(S) T~ on H^1(C_I) over F_l' (the new operator) = the production residue at " + at);
                }
                if (ellp == 10007 || slow) {
                    std::vector<uint32_t> ship;
                    if (c.N == 1) {
                        const integral::FellRun run = integral::RunPipelineModEll(c.n, ellp, {p});
                        const hecke::FpPoly& f = run.charpoly.at(p);
                        ship.assign(f.coeffs.rbegin(), f.coeffs.rend());
                    } else {
                        ship = DenseShippedCharpoly(c.N, c.n, p, ellp);
                    }
                    check(ship == prod, std::string("(S) the shipped F_p pipeline (") + (c.N == 1 ? "tau-table, RunPipelineModEll" : "eq:levelcorr, HFrame") +
                                            ") = the production residue at " + at);
                    n_ok += 2;
                }
                ++n_ok;
            }
        }
    std::cout << "  (S) " << n_ok << " residue comparisons: production (H_{N,p}, cokernel) = char_Z mod l' = the shipped F_p pipeline = T~ on "
              << "H^1(C_I) over F_l'" << std::endl;
    // (C) T_p T_p' = T_p' T_p on H^1 over F_10007 (the free model's operators in one presentation; prop:strictliftN(8))
    uint32_t n_c = 0;
    for (const auto& c : std::vector<C3>{{1, 10}, {1, 22}, {11, 2}, {2, 6}, {5, 4}, {13, 2}}) {
        std::vector<uint32_t> ps;
        for (uint32_t p : {2u, 3u, 5u, 7u})
            if (c.N % p) ps.push_back(p);
        std::vector<Mat> F(ps.size());
        for (size_t i = 0; i < ps.size(); ++i) {
            IsoRouteCharpoly(c.N, c.n, ps[i], 10007, &F[i]);
            for (auto& x : F[i].a) x %= 10007;
        }
        const ZMod Z1(10007, 1);
        for (size_t i = 0; i < ps.size(); ++i)
            for (size_t j = i + 1; j < ps.size(); ++j) {
                check(P::MatMulZ(Z1, F[i], F[j]).a == P::MatMulZ(Z1, F[j], F[i]).a,
                      "(C) T_" + std::to_string(ps[i]) + " T_" + std::to_string(ps[j]) + " = T_" + std::to_string(ps[j]) + " T_" + std::to_string(ps[i]) +
                          " on H^1 over F_10007 " + cellname(c.N, c.n));
                ++n_c;
            }
    }
    std::cout << "  (C) " << n_c << " pairs T_p T_p' = T_p' T_p on H^1 over F_10007 (T~ on the free model)" << std::endl;
}

// =================================================================================================================================
// (X) transport equality; (G) the groups; (E) periodicity
// =================================================================================================================================
void ArmTransport(bool slow) {
    struct C3 {
        uint32_t N, n;
    };
    std::vector<C3> cells = {{1, 4}, {1, 10}, {1, 3}, {2, 2}, {2, 6}, {3, 4}, {4, 2}, {7, 2}, {5, 4}, {2, 1}};
    if (slow) cells.insert(cells.end(), {{1, 16}, {1, 24}, {9, 2}, {13, 2}, {6, 2}, {11, 4}, {3, 6}});
    uint32_t n_eq = 0;
    for (const auto& c : cells) {
        std::vector<uint32_t> ps;
        for (uint32_t p : {2u, 3u, 5u, 7u})
            if (c.N % p && ps.size() < 2) ps.push_back(p);
        for (uint32_t ell : {2u, 3u})
            for (uint32_t q : {1u, 2u, 3u}) {
                if (ell == 3 && c.n % 2) continue;
                const std::string tag = " l=" + std::to_string(ell) + " q=" + std::to_string(q) + " " + cellname(c.N, c.n);
                const H3::TorsionEntry T = H3::TorsionOnTree(ell, c.N, c.n, q, ps);
                const H3::TorsionEntry I = H3::TorsionOnIsoEll(ell, c.N, c.n, q, ps);
                const H3::TorsionEntry F = H3::TorsionOnFreeIso(ell, c.N, c.n, q, ps, ell == 2 ? 30u : 19u);
                if (!check(!T.refused && !I.refused && !F.refused, "(X) three reads" + tag + ": " + T.refusal + " | " + I.refusal + " | " + F.refusal)) continue;
                bool same = T.read.sigma.size() == I.read.sigma.size() && T.read.sigma.size() == F.read.sigma.size();
                if (T.read.sigma.empty()) same = same && T.inv.empty() && I.inv.empty() && F.inv.empty();
                for (size_t i = 0; same && !T.read.sigma.empty() && i < ps.size(); ++i)
                    same = T.inv.size() > i && I.inv.size() > i && F.inv.size() > i && T.inv[i].canon == I.inv[i].canon && T.inv[i].canon == F.inv[i].canon;
                check(same, "(X) T'_{T,p} on the tree = T'^(l)_p on Iso^(l) = T~_p on Iso, every invariant" + tag + ": tree " +
                                (T.inv.empty() ? std::string("-") : T.inv[0].canon) + " | Iso^(l) " + (I.inv.empty() ? std::string("-") : I.inv[0].canon) +
                                " | Iso " + (F.inv.empty() ? std::string("-") : F.inv[0].canon));
                ++n_eq;
            }
        // (G) the tree groups = D5's production groups (RunTreeLocalPieceN, the Betti rank source)
        const integral_descent::LevelNCounts cnt = integral_descent::LevelNClosedForm(c.N, c.n);
        const std::vector<uint32_t> h = {c.n % 2 ? 0u : cnt.b0, c.n % 2 ? 0u : cnt.b1, 0, 0, 0};
        for (uint32_t ell : {2u, 3u}) {
            if (ell == 3 && c.n % 2) continue;
            const integral_descent::LocalPiece LP = integral_descent::RunTreeLocalPieceN(ell, c.N, c.n, 3, h);
            for (uint32_t q : {1u, 2u, 3u}) {
                const H3::TorsionEntry T = H3::TorsionOnTree(ell, c.N, c.n, q, {});
                std::vector<uint32_t> a = T.read.sigma, b;
                std::sort(a.begin(), a.end());
                const auto it = LP.tors[q].torsion.find(ell);
                if (it != LP.tors[q].torsion.end()) b = it->second;
                std::sort(b.begin(), b.end());
                check(LP.all_certified() && a == b, "(G) the tree read's group = D5's production group (RunTreeLocalPieceN) at l=" + std::to_string(ell) +
                                                        " q=" + std::to_string(q) + " " + cellname(c.N, c.n));
            }
        }
    }
    std::cout << "  (X) " << n_eq << " (cell, l, q): the tree, Iso^(l) and Iso reads equal in every invariant; (G) the groups = D5's" << std::endl;
    // (E) periodicity on the tree: H^4 = H^2 with T_p (thm:descentcert(6))
    for (const auto& c : std::vector<C3>{{1, 10}, {2, 6}, {3, 4}, {1, 3}})
        for (uint32_t ell : {2u, 3u}) {
            if (ell == 3 && c.n % 2) continue;
            const std::vector<uint32_t> ps = c.N % 2 ? std::vector<uint32_t>{2} : std::vector<uint32_t>{3};
            const H3::TorsionEntry A = H3::TorsionOnTree(ell, c.N, c.n, 2, ps), B = H3::TorsionOnTree(ell, c.N, c.n, 4, ps);
            check(!A.refused && !B.refused && !A.inv.empty() == !B.inv.empty() && (A.inv.empty() || A.inv[0].canon == B.inv[0].canon),
                  "(E) H^4 = H^2 with T_p on the tree at l=" + std::to_string(ell) + " " + cellname(c.N, c.n));
        }
    std::cout << "  (E) H^4 = H^2 with T_p on the tree (8 cells)" << std::endl;
}

// =================================================================================================================================
// (M) T~ sparse = D4's dense bulk; T~ a cochain map
// =================================================================================================================================
void ArmOperator() {
    uint64_t entries = 0;
    for (const auto& c : std::vector<std::array<uint32_t, 2>>{{1, 10}, {11, 0}, {2, 6}, {3, 2}})
        for (uint32_t p : {2u, 7u, 13u}) {
            if (c[0] % p == 0) continue;
            const U32Ctx ctx(10007);
            const ZMod Z(10007, 1);
            const H3::IsoHeckeN T(c[0], c[1], p, ctx, 4);
            if (!check(!T.refused(), "(M) T~ built " + cellname(c[0], c[1]) + ": " + T.why())) continue;
            const P::ConeHecke Hd = P::BuildConeHecke(c[0], c[1], p, ctx, 4);
            if (!check(!Hd.refused, "(M) D4's operator " + Hd.refusal)) continue;
            const P::PairComplex X = P::BuildFreePairN(c[0], c[1], ctx, 4);
            bool eq = true, cm = true;
            for (uint32_t q = 0; q <= 4; ++q) {
                const engine::DenseBuf<uint32_t> A = T.dense(q);
                eq = eq && A.rows == Hd.bulk[q].rows && A.cols == Hd.bulk[q].cols;
                for (uint32_t i = 0; eq && i < A.rows; ++i)
                    for (uint32_t j = 0; eq && j < A.cols; ++j) eq = A.at(i, j) == Hd.bulk[q].at(i, j);
                entries += uint64_t(A.rows) * A.cols;
                // the sparse apply on random vectors equals the dense matrix
                std::mt19937 rng(q + 7 * p);
                std::vector<uint32_t> v(A.cols * 3), y;
                for (auto& x : v) x = rng() % 10007;
                T.apply(q, v, 3, y);
                for (uint32_t i = 0; i < A.rows && eq; ++i)
                    for (uint32_t t = 0; t < 3; ++t) {
                        uint64_t s = 0;
                        for (uint32_t j = 0; j < A.cols; ++j) s = (s + uint64_t(A.at(i, j)) * v[j * 3 + t]) % 10007;
                        eq = eq && s == y[i * 3 + t];
                    }
                if (q < 4) {
                    const Mat Tq = P::FromDenseBuf(A), Tq1 = P::FromDenseBuf(T.dense(q + 1));
                    cm = cm && P::MatMulZ(Z, X.d[q], Tq).a == P::MatMulZ(Z, Tq1, X.d[q]).a;
                }
            }
            check(eq, "(M) T~ (sparse, DescentHecke) = D4's dense bulk operator entrywise, and its batch apply = the matrix, q <= 4, " + cellname(c[0], c[1]) +
                          " p=" + std::to_string(p));
            check(cm, "(M) T~ is a cochain map on C_I (d T~ = T~ d), q <= 3, " + cellname(c[0], c[1]) + " p=" + std::to_string(p));
        }
    std::cout << "  (M) T~ = D4's bulk operator on " << entries << " entries; T~ a cochain map" << std::endl;
}

// =================================================================================================================================
// (W) the words; (B) the bound, the Eisenstein root, the free quotient
// =================================================================================================================================
void ArmWords() {
    std::mt19937_64 rng(20261026);
    const std::vector<uint32_t> primes = integral::CrtPrimes(6);
    const std::vector<uint32_t> inv = H3::GarnerInverses(primes);
    uint32_t ok = 0;
    for (int t = 0; t < 2000; ++t) {
        // a signed 62-bit integer through 3 primes (P ~ 2^90 > 2^63): residues, digits, sign, value, mod 2^64, mod 3^40
        const int64_t v = static_cast<int64_t>(rng() >> 2) * ((t & 1) ? -1 : 1);
        std::vector<uint32_t> res;
        for (int k = 0; k < 3; ++k) {
            int64_t r = v % static_cast<int64_t>(primes[k]);
            if (r < 0) r += primes[k];
            res.push_back(static_cast<uint32_t>(r));
        }
        const std::vector<uint32_t> pr(primes.begin(), primes.begin() + 3);
        const auto dg = H3::GarnerDigits(res, pr, inv);
        const bool neg = H3::MixedRadixNegative(dg, pr);
        int64_t back = 0;
        const bool fits = H3::MixedRadixToInt64(dg, pr, neg, &back);
        uint64_t Q340 = 1;
        for (int i = 0; i < 40; ++i) Q340 *= 3;
        const uint64_t m64 = H3::MixedRadixModQ(dg, pr, neg, 0);
        const uint64_t m3 = H3::MixedRadixModQ(dg, pr, neg, Q340);
        // the reference residue of the signed v mod 3^40, in uint64 (3^40 > INT64_MAX, so no int64 cast of Q340): r = |v| mod Q340, then Q340 - r for v < 0, r != 0
        const uint64_t av = v < 0 ? static_cast<uint64_t>(0) - static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
        const uint64_t ra = av % Q340;
        const uint64_t r3 = (v < 0 && ra != 0) ? Q340 - ra : ra;
        ok += (neg == (v < 0) && fits && back == v && m64 == static_cast<uint64_t>(v) && m3 == r3);
    }
    check(ok == 2000, "(W) 2000 signed 62-bit integers: Garner digits, the sign test, the int64 value, c mod 2^64 and mod 3^40 (word mulmod), " +
                          std::to_string(ok) + "/2000");
    // beyond 2^63: c = -(x y), x, y < 2^40, through 6 primes; mod 2^64 and mod 3^40 against the word products; |c| >= 2^63 reported as not fitting
    uint32_t big = 0;
    for (int t = 0; t < 500; ++t) {
        const uint64_t x = (rng() >> 24) | (uint64_t(1) << 39), y = (rng() >> 24) | (uint64_t(1) << 39);
        std::vector<uint32_t> res;
        for (uint32_t ell : primes) {
            const uint64_t r = (x % ell) * (y % ell) % ell;
            res.push_back(static_cast<uint32_t>((ell - r) % ell));
        }
        const auto dg = H3::GarnerDigits(res, primes, inv);
        const bool neg = H3::MixedRadixNegative(dg, primes);
        int64_t dummy;
        uint64_t Q340 = 1;
        for (int i = 0; i < 40; ++i) Q340 *= 3;
        const uint64_t want64 = static_cast<uint64_t>(0) - x * y;
        const uint64_t p3 = H3::WordMulMod(x % Q340, y % Q340, Q340);
        const uint64_t want3 = p3 ? Q340 - p3 : 0;
        big += (neg && !H3::MixedRadixToInt64(dg, primes, neg, &dummy) && H3::MixedRadixModQ(dg, primes, neg, 0) == want64 &&
                H3::MixedRadixModQ(dg, primes, neg, Q340) == want3);
    }
    check(big == 500, "(W) 500 negative products beyond 2^63 (6 primes): sign, 'does not fit', c mod 2^64 and mod 3^40 (the negative case), " +
                          std::to_string(big) + "/500");
    // (the D3 audit's m3) at the reach runs' scale: c = +- (product of 140 words of 62 bits, ~ 8680 bits) through 300 primes (P ~ 2^8700): Garner,
    // the sign, mod 2^64 / 3^40 against the word products, and c mod a held-out prime (the held-out check's digits branch)
    uint32_t wide = 0;
    {
        const std::vector<uint32_t> wp = integral::CrtPrimes(301);
        const std::vector<uint32_t> pr(wp.begin(), wp.begin() + 300);
        const uint32_t ho = wp.back();
        const std::vector<uint32_t> winv = H3::GarnerInverses(pr);
        uint64_t Q340 = 1;
        for (int i = 0; i < 40; ++i) Q340 *= 3;
        for (int t = 0; t < 12; ++t) {
            const bool neg = t & 1;
            std::vector<uint64_t> w(140);
            for (auto& x : w) x = (rng() >> 2) | 1u;
            auto prodmod = [&](uint64_t Q) {   // prod w mod Q (Q = 0: 2^64), then the sign
                uint64_t a = Q ? 1 % Q : 1;
                for (uint64_t x : w) a = H3::WordMulMod(a, Q ? x % Q : x, Q);
                if (!neg) return a;
                return Q ? (a ? Q - a : 0) : static_cast<uint64_t>(0) - a;
            };
            std::vector<uint32_t> res;
            for (uint32_t ell : pr) res.push_back(static_cast<uint32_t>(prodmod(ell)));
            const auto dg = H3::GarnerDigits(res, pr, winv);
            const bool sg = H3::MixedRadixNegative(dg, pr);
            int64_t dummy;
            wide += (sg == neg && !H3::MixedRadixToInt64(dg, pr, sg, &dummy) && H3::MixedRadixModQ(dg, pr, sg, 0) == prodmod(0) &&
                     H3::MixedRadixModQ(dg, pr, sg, Q340) == prodmod(Q340) && H3::MixedRadixModQ(dg, pr, sg, ho) == prodmod(ho));
        }
    }
    check(wide == 12, "(W) 12 signed products of 140 words (~ 8680 bits) through 300 primes: sign, mod 2^64, mod 3^40, mod a held-out prime, " +
                          std::to_string(wide) + "/12");
    // the word mulmod against small exact products
    bool mm = true;
    for (int t = 0; t < 1000; ++t) {
        const uint64_t Q = (rng() >> 34) + 3, a = rng() % Q, b = rng() % Q;
        mm = mm && H3::WordMulMod(a, b, Q) == (a * b) % Q;
    }
    check(mm, "(W) the shift-and-add mulmod = the exact product mod Q (Q < 2^30)");
    // the read's Smith engine (levels, left transforms) against D2's maps::SmithForm: the same valuations; L L^{-1} = 1; rows >= s of L A
    // vanish mod ell^k, row i < s of L A is divisible by ell^{sigma_i} with an entry of valuation exactly sigma_i -- on random matrices
    // with planted valuations and on the tree's d^{q-1}
    uint32_t sm = 0;
    auto smith_ok = [&](const ZMod& Z, const Mat& A) {
        const integral_descent::maps::Smith a = H3::SmithLevelsLeft(Z, A), b = integral_descent::maps::SmithForm(Z, A, true, false);
        std::vector<uint32_t> sa = a.sigma, sb = b.sigma;
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        bool ok = sa == sb;
        const Mat LA = P::MatMulZ(Z, a.L, A);
        const uint32_t s = a.rank();
        for (uint32_t i = 0; ok && i < A.r; ++i) {
            bool exact = false;
            for (uint32_t j = 0; ok && j < A.c; ++j) {
                const uint32_t x = LA.at(i, j);
                if (i >= s) ok = x == 0;
                else {
                    ok = Z.val(x) >= a.sigma[i];
                    exact = exact || (x && Z.val(x) == a.sigma[i]);
                }
            }
            if (i < s) ok = ok && exact;
        }
        // L (L^{-1}) = 1: LinvT's row t is column t of L^{-1}
        for (uint32_t i = 0; ok && i < A.r; ++i)
            for (uint32_t t = 0; ok && t < A.r; ++t) {
                uint64_t acc = 0;
                for (uint32_t m = 0; m < A.r; ++m) acc = (acc + uint64_t(a.L.at(i, m)) * a.LinvT.at(t, m)) % Z.m;
                ok = acc == (i == t ? 1u : 0u);
            }
        ++sm;
        return ok;
    };
    bool sall = true;
    for (int t = 0; t < 60; ++t) {
        const uint32_t ell = t % 3 == 0 ? 2 : (t % 3 == 1 ? 3 : 5), k = ell == 2 ? 12 : (ell == 3 ? 7 : 5);
        const ZMod Z(ell, k);
        const uint32_t r = 3 + rng() % 9, c = 3 + rng() % 9;
        Mat A(r, c);
        for (auto& x : A.a) {
            const uint32_t v = rng() % (k + 2);   // a planted valuation (>= k: zero)
            x = v >= k ? 0u : Z.mul(Z.pw(v), static_cast<uint32_t>(rng() % Z.m) | 1u);
        }
        sall = smith_ok(Z, A) && sall;
    }
    for (const auto& c : std::vector<std::array<uint32_t, 3>>{{2, 1, 10}, {2, 2, 6}, {3, 3, 4}, {2, 37, 2}, {3, 1, 12}})
        for (uint32_t q = 1; q <= 3; ++q) {
            const H3::TreeTransport X(c[0], c[1], c[2], q);
            sall = smith_ok(X.Z(), X.differential(q - 1)) && sall;
        }
    check(sall, "(W) SmithLevelsLeft = maps::SmithForm in the valuations, L L^{-1} = 1, L A in the levels' shape (" + std::to_string(sm) + " matrices)");
    std::cout << "  (W) Garner, the sign, mod 2^64 / 3^40 (incl. the negative case) and the word mulmod: 3512 checks (12 at 300 primes)" << std::endl;
}

void ArmBound() {
    // every certified coefficient within its bound; the Eisenstein root (prop:crtcharpolyN(3): attained iff r_N > 0)
    for (const auto& c : std::vector<std::array<uint32_t, 3>>{{1, 22, 13}, {11, 2, 7}, {5, 4, 2}, {1, 30, 5}, {23, 0, 13}}) {
        const H3::CharpolyZ C = H3::CharpolyOverZ(c[0], c[1], c[2]);
        if (!check(!C.refused && C.determined, "(B) char_Z " + cellname(c[0], c[1]) + ": " + C.refusal)) continue;
        bool in = true;
        for (const auto& x : C.c)
            if (x.fits) {
                const uint64_t a = x.value < 0 ? static_cast<uint64_t>(-x.value) : static_cast<uint64_t>(x.value);
                in = in && H3::BitLen(a) <= x.bits;
            }
        check(in, "(B) |c_i| <= 2^{b_{p,i}} on every coefficient " + cellname(c[0], c[1]) + " p=" + std::to_string(c[2]));
        // chi(1 + p^{n+1}) = 0 mod the first prime
        const uint32_t ell = C.primes.front();
        uint64_t e = 1;
        for (uint32_t i = 0; i <= c[1]; ++i) e = e * c[2] % ell;
        e = (e + 1) % ell;
        const std::vector<uint32_t> res = H3::ShippedResidueCharpoly(c[0], c[1], c[2], ell);
        uint64_t v = 0;
        for (uint32_t x : res) v = (v * e + x) % ell;
        check(v == 0, "(B) the Eisenstein eigenvalue 1 + p^{n+1} is a root of chi mod l' " + cellname(c[0], c[1]) + " p=" + std::to_string(c[2]));
        std::cout << "  (B) " << cellname(c[0], c[1]) << " p=" << c[2] << " r=" << C.r << " primes=" << C.primes.size() << " max b_{p,i}="
                  << C.c.back().bits << " chi=" << (C.text().size() > 120 ? C.text().substr(0, 120) + "..." : C.text()) << std::endl;
    }
    // thm:descentcert(4) at l >= 5: Ramanujan's congruence -- T_2 - (1 + 2^11) on Lambda (x) Z_691 at n = 10 has the root 1 + 2^11 (refused, nothing
    // asserted), and at c = tau(2) + 691 = 667 (not a root: chi(667) = (667 - 2049)(667 + 24)^2 = -1382 * 691^2): the exponents are 1, 1, 1
    {
        const H3::FreeQuotient root = H3::FreeQuotientFactors(691, 1, 10, 2, 2049);
        check(!root.ok, "(B) (4): c = 1 + 2^11 is a root of chi -> nothing asserted (refused): " + root.why);
        // the C-D4 author's check (cd4_hecke_check.out, independent): exponents [0, 1, 2] at c = 667 and 2740, [0, 1, 3] at 479530 (needs k >= 4 >
        // the word's 691^3: refused here, as prop:torsionhecke(4) says)
        const H3::FreeQuotient q = H3::FreeQuotientFactors(691, 1, 10, 2, 667), q2 = H3::FreeQuotientFactors(691, 1, 10, 2, 2740),
                               q3 = H3::FreeQuotientFactors(691, 1, 10, 2, 479530);
        check(q.ok && q.exps == std::vector<uint32_t>({1, 2}),
              "(B) (4): T_2 - 667 on Lambda (x) Z_691 at n = 10 has invariant factors 691^{0,1,2} (cd4_hecke_check.out): " + q.why + " got " + join(q.exps, ","));
        check(q2.ok && q2.exps == std::vector<uint32_t>({1, 2}), "(B) (4): T_2 - 2740: 691^{0,1,2}: got " + join(q2.exps, ","));
        check(!q3.ok, "(B) (4): T_2 - 479530 (exponents 0, 1, 3) is refused at k = 3 (k must exceed the largest exponent): " + q3.why);
        std::cout << "  (B) (4) n=10, l=691: T_2 - 2049 refused (a root); T_2 - 667, - 2740: 691^{0," << join(q.exps, ",") << "}; T_2 - 479530 refused at k=3"
                  << std::endl;
    }
}

// =================================================================================================================================
// (K) the precision rule; (N) negative controls; (R) refusals
// =================================================================================================================================
void ArmPrecision() {
    // (1, 4): H^1's 2-part is Z/4 (sigma_max = 2): at K = 2 sigma_max - 1 = 3 the module is refused, P_j and the Fitting types kept
    const Oracle& O = TheOracle();
    const H3::TorsionEntry A = H3::TorsionOnTree(2, 1, 4, 1, {3});
    H3::TorsionEntry B;
    {
        H3::ScopedD3Fault f(H3::D3Fault::ReadBelowModule);
        B = H3::TorsionOnTree(2, 1, 4, 1, {3});
    }
    check(!A.refused && A.read.module_certified && A.read.smax == 2, "(K) the read at K = 30 certifies the module (sigma_max = 2)");
    check(B.read.ok && !B.read.module_certified && B.code == H3::D3Refusal::ModulePrecision && B.read.K == 3,
          "(K) at K = 3 = 2 sigma_max - 1: [ModulePrecision], module_certified = false: " + B.refusal);
    check(!A.inv.empty() && !B.inv.empty() && A.inv[0].layers == B.inv[0].layers,
          "(K) P_j and the Fitting types are the same at K = 3 (they need only a certified form): " + (B.inv.empty() ? std::string() : B.inv[0].layers));
    const auto it = O.tors.find({1, 4, 1, 2, 3});
    check(it != O.tors.end() && !A.inv.empty() && it->second == A.inv[0].canon, "(K) and at K = 30 the whole module = the oracle");
    check(!B.inv.empty() && !B.inv[0].full && B.inv[0].canon.find("ker=") == std::string::npos, "(K) at K = 3 no ker / coker type is reported");
    std::cout << "  (K) (1,4) l=2: K = 30 module certified; K = 3 [ModulePrecision], P_j / Fitting kept (" << (B.inv.empty() ? "" : B.inv[0].layers) << ")" << std::endl;
}

void ArmNegative() {
    // each fault through the production route, with its exact caught pattern
    // the operator faults on a fixed grid of tree reads: per read the refusal code, or "same" / "DIFF" (accepted; DIFF = its invariants differ
    // from the unfaulted read's, i.e. the oracle comparison (O) / (X) catches it); the whole pattern is pinned
    struct G3 {
        uint32_t ell, N, n, q, p;
    };
    const std::vector<G3> grid = {{2, 1, 10, 1, 2}, {2, 1, 10, 2, 2}, {2, 1, 10, 3, 2}, {2, 2, 6, 1, 3}, {2, 2, 6, 2, 3}, {2, 2, 6, 3, 3},
                                  {3, 3, 4, 1, 2}, {3, 3, 4, 2, 2}};
    std::vector<std::string> good;
    for (const G3& g : grid) {
        const H3::TorsionEntry E = H3::TorsionOnTree(g.ell, g.N, g.n, g.q, {g.p});
        good.push_back(E.inv.empty() ? std::string("-") : E.inv[0].canon);
    }
    auto pattern = [&](H3::D3Fault fault) {
        H3::ScopedD3Fault f(fault);
        std::string out;
        for (size_t i = 0; i < grid.size(); ++i) {
            const G3& g = grid[i];
            const H3::TorsionEntry E = H3::TorsionOnTree(g.ell, g.N, g.n, g.q, {g.p});
            const std::string got = E.inv.empty() ? std::string("-") : E.inv[0].canon;
            out += (i ? " " : "") + std::string(E.refused ? H3::D3RefusalName(E.code) : (got == good[i] ? "same" : "DIFF"));
        }
        return out;
    };
    struct Expect {
        H3::D3Fault f;
        const char* name;
        const char* pattern;
    };
    const Expect ex[] = {
        // a "same" read is one where the fault leaves the induced map on that group unchanged (not a defect of the read: the read is exact,
        // and the fault does not move the invariants there); every pattern is pinned exactly
        {H3::D3Fault::DropCoset, "a dropped coset in T~ (p of the p + 1)",
         "NotEndomorphism NotEndomorphism NotEndomorphism NotEndomorphism NotEndomorphism DIFF NotEndomorphism same"},
        {H3::D3Fault::BlockSign, "every second block term of T~ with multiplicity + 1",
         "NotEndomorphism NotEndomorphism NotEndomorphism NotEndomorphism NotEndomorphism DIFF NotEndomorphism same"},
        {H3::D3Fault::PsiDropTerm, "a term of Psi_T dropped", "NotEndomorphism NotEndomorphism same same NotEndomorphism same same same"},
    };
    for (const Expect& e : ex) {
        const std::string pat = pattern(e.f);
        std::cout << "  (N) " << e.name << ": pattern on the 8 tree reads [" << pat << "]" << std::endl;
        check(pat == e.pattern, std::string("(N) ") + e.name + ": the caught pattern is exactly [" + e.pattern + "], got [" + pat + "]");
    }
    {
        H3::ScopedD3Fault f(H3::D3Fault::DropCoset);
        const std::vector<uint32_t> iso = IsoRouteCharpoly(1, 10, 2, 10007);
        const std::vector<uint32_t> prod = H3::ShippedResidueCharpoly(1, 10, 2, 10007);
        check(iso != prod, "(N) a dropped coset: char T~ on H^1(C_I) over F_10007 differs from the shipped residue (arm (S) catches it)");
    }
    {
        H3::ScopedD3Fault f(H3::D3Fault::SinglePrime);
        const H3::CharpolyZ C = H3::CharpolyOverZ(1, 22, 13);
        uint32_t unc = 0, want = 0;
        for (const auto& x : C.c) {
            unc += !x.certified;
            want += x.bits + 1 > 29;   // one prime of bitlen 30 covers 29 bits
        }
        check(!C.refused && !C.determined && unc == want && unc > 0,
              "(N) one residue prime at (1,22) p=13: exactly the " + std::to_string(want) + " coefficients with b_{p,i} + 1 > 29 are reported undetermined "
              "(residues and bound), none lifted (got " + std::to_string(unc) + ")");
    }
    // the held-out prime's mismatch is a defect (the D3 audit's m2): a hecke::HeckeInternalError "[HeldOutMismatch] ...", never a refusal
    auto defect = [](H3::D3Fault fault, uint32_t N, uint32_t n, uint32_t p, std::string& what) {
        H3::ScopedD3Fault f(fault);
        try {
            const H3::CharpolyZ C = H3::CharpolyOverZ(N, n, p);
            what = C.refused ? "refused: " + C.refusal : "accepted: " + C.text();
        } catch (const hecke::HeckeInternalError& e) {
            what = e.what();
            return what.rfind("[HeldOutMismatch]", 0) == 0;
        }
        return false;
    };
    std::string w;
    check(defect(H3::D3Fault::NoEisensteinBound, 1, 22, 13, w),
          "(N) the bound without the Eisenstein term at (1,22) p=13: the held-out prime throws HeckeInternalError [HeldOutMismatch]: " + w);
    check(defect(H3::D3Fault::PerturbedResidue, 1, 10, 2, w), "(N) a perturbed residue (c_1, int64) at (1,10) p=2: [HeldOutMismatch]: " + w);
    {
        // (the D3 audit's m3) the digits branch: c_r beyond 2^63, its residue perturbed at the middle prime of its prefix
        const H3::CharpolyZ C = H3::CharpolyOverZ(1, 22, 13);
        check(!C.refused && C.r > 0 && C.c.back().certified && !C.c.back().fits && C.c.back().primes_used >= 2,
              "(N) at (1,22) p=13 c_r is certified beyond 2^63 on >= 2 primes (the digits branch is the one driven): " + C.text());
        check(defect(H3::D3Fault::PerturbedDigits, 1, 22, 13, w),
              "(N) c_r's residue (beyond 2^63) perturbed at a middle prime of its prefix: the held-out prime's digits branch throws [HeldOutMismatch]: " + w);
    }
    std::cout << "  (N) seven faults through the production route, each caught by its pattern (the dropped coset also by arm (S); the held-out "
                 "prime's int64 and digits branches)" << std::endl;
}

// =================================================================================================================================
// (L) the ker / coker width ladder (ICENSUS-P1): at the level-1 cells where the read at k_0 = min(30, 2 a_max + 3) fell short (the
// ICENSUS probes' [KerCokerPrecision] at l = 2, q = 1) the read is redone at K_max = 30 and now reads; cross-checked independently
// =================================================================================================================================
// the exponents of the cokernel of an r x m matrix over Z/2^K whose cokernel is killed by 2^{K-1} (so every elementary divisor 2^v has
// v < K and is read exactly): plain 2-adic Smith elimination, minimal-valuation pivot; shares nothing with maps::SmithForm / InducedMap
std::vector<uint32_t> CokerExps2(std::vector<std::vector<uint64_t>> A, uint32_t K) {
    const uint64_t mask = (uint64_t(1) << K) - 1;
    const size_t rows = A.size(), cols = rows ? A[0].size() : 0;
    auto val = [](uint64_t x) { return static_cast<uint32_t>(__builtin_ctzll(x)); };
    std::vector<uint32_t> out;
    for (size_t t = 0; t < rows && t < cols; ++t) {
        size_t pi = rows, pj = cols;
        uint32_t best = K;
        for (size_t i = t; i < rows; ++i)
            for (size_t j = t; j < cols; ++j)
                if (A[i][j] && val(A[i][j]) < best) best = val(A[i][j]), pi = i, pj = j;
        if (pi == rows) break;
        std::swap(A[t], A[pi]);
        for (auto& row : A) std::swap(row[t], row[pj]);
        const uint64_t u = A[t][t] >> best;   // odd
        uint64_t inv = 1;
        for (int it = 0; it < 6; ++it) inv = (inv * (2 - u * inv)) & mask;
        for (size_t i = t + 1; i < rows; ++i)
            if (A[i][t]) {
                const uint64_t q = ((A[i][t] >> best) * inv) & mask;
                for (size_t j = t; j < cols; ++j) A[i][j] = (A[i][j] - q * A[t][j]) & mask;
            }
        for (size_t j = t + 1; j < cols; ++j)
            if (A[t][j]) {
                const uint64_t q = ((A[t][j] >> best) * inv) & mask;
                for (size_t i = t; i < rows; ++i) A[i][j] = (A[i][j] - q * A[i][t]) & mask;
            }
        if (best) out.push_back(best);
    }
    std::sort(out.begin(), out.end());
    return out;
}
// "c:e.e.e|c:-|..." -> per c the exponents
std::vector<std::vector<uint32_t>> ParseTypes(const std::string& canon, const std::string& key) {
    std::vector<std::vector<uint32_t>> out;
    const size_t b = canon.find(";" + key + "=");
    if (b == std::string::npos) return out;
    const size_t s = b + key.size() + 2, e = canon.find(';', s);
    std::istringstream in(canon.substr(s, e == std::string::npos ? std::string::npos : e - s));
    std::string tok;
    while (std::getline(in, tok, '|')) {
        const std::string ex = tok.substr(tok.find(':') + 1);
        std::vector<uint32_t> v;
        if (ex != "-") {
            std::istringstream es(ex);
            std::string x;
            while (std::getline(es, x, '.')) v.push_back(static_cast<uint32_t>(std::stoul(x)));
        }
        out.push_back(v);
    }
    return out;
}
void ArmLadder() {
    const std::vector<uint32_t> ns = {48, 64, 80, 200};
    const std::vector<uint32_t> ps = {2, 3, 5};   // DefaultRecordPrimes(1), the record's (and the probes') primes
    uint32_t escalated = 0, reads = 0, c_checked = 0;
    for (uint32_t n : ns) {
        const auto t0 = clk::now();
        const H3::TorsionEntry E = H3::TorsionOnTree(2, 1, n, 1, ps);
        const double secs = since(t0);
        const std::string tag = " l=2 q=1 " + cellname(1, n);
        if (!check(!E.refused && E.inv.size() == ps.size(), "(L) the ker / coker read" + tag + " reads (no refusal): " + E.refusal)) continue;
        std::vector<uint32_t> a = E.read.sigma;
        std::sort(a.begin(), a.end());
        // the groups' 2-primary part (D5's production groups, RunTreeLocalPieceN)
        const integral_descent::LevelNCounts cnt = integral_descent::LevelNClosedForm(1, n);
        const integral_descent::LocalPiece LP = integral_descent::RunTreeLocalPieceN(2, 1, n, 3, {cnt.b0, cnt.b1, 0, 0, 0});
        std::vector<uint32_t> g;
        const auto it = LP.tors[1].torsion.find(2);
        if (it != LP.tors[1].torsion.end()) g = it->second;
        std::sort(g.begin(), g.end());
        check(E.read.module_certified && LP.all_certified() && a == g, "(L) the module = the groups' 2-primary part of H^1" + tag);
        const uint32_t r = static_cast<uint32_t>(E.read.sigma.size());
        const std::vector<uint32_t>& sg = E.read.sigma;
        bool any_esc = false;
        for (size_t o = 0; o < ps.size(); ++o) {
            const H3::TorsionInvariants& I = E.inv[o];
            const std::string ptag = tag + " p=" + std::to_string(ps[o]);
            ++reads;
            any_esc = any_esc || I.kc_escalated;
            escalated += I.kc_escalated;
            check(I.full && I.canon.find(";ker=") != std::string::npos && I.kc_k == (I.kc_escalated ? 30u : std::min(30u, 2 * a.back() + 3)),
                  "(L) ker / cok read" + ptag + " at k = " + std::to_string(I.kc_k) + (I.kc_escalated ? " (escalated)" : ""));
            if (I.kc_escalated) {
                // negative control: the same read forced to the width k_0 refuses, as a precision shortfall
                const H3::TorsionInvariants J = H3::Invariants(2, sg, E.read.Phi[o], true, E.read.Kp, true, 400, std::min(30u, 2 * a.back() + 3));
                check(!J.ok && J.kc_precision && J.why.find("not certified") != std::string::npos,
                      "(L) the read forced to k_0 refuses (a precision shortfall)" + ptag + ": " + J.why);
            } else {
                // the same types at the one width K_max (a certified type does not depend on the width)
                const H3::TorsionInvariants J = H3::Invariants(2, sg, E.read.Phi[o], true, E.read.Kp, true, 400, 30);
                check(J.ok && J.canon == I.canon, "(L) the read forced to k = 30 gives the same canonical text" + ptag);
            }
            const auto Kt = ParseTypes(I.canon, "ker"), Ct = ParseTypes(I.canon, "cok");
            const uint32_t amax = a.back(), E2 = 1u << amax;
            if (!check(Kt.size() == E2 && Ct.size() == E2, "(L) ker / cok types for every c in [0, 2^a_max)" + ptag)) continue;
            for (uint32_t c0 = 0; c0 < E2; ++c0) {
                // the full types, independent of InducedMap: G = sum Z/2^{a_i}, f = phi - c (column j = f(e_j), entries mod 2^{a_i});
                // coker f = coker [F | diag 2^{a_i}]; ker f = (coker f^)^ with f^ the Pontryagin dual, (f^)_{ji} = F_ij 2^{a_j - a_i};
                // both by CokerExps2 over Z/2^{a_max + 1}
                std::vector<std::vector<uint64_t>> Mc(r, std::vector<uint64_t>(2 * r, 0)), Md(r, std::vector<uint64_t>(2 * r, 0));
                bool endo = true;
                for (uint32_t i = 0; i < r; ++i)
                    for (uint32_t j = 0; j < r; ++j) {
                        const uint64_t mi = uint64_t(1) << sg[i];
                        const uint64_t f = (E.read.Phi[o].at(i, j) + (i == j ? mi * 4 - (c0 % mi) : 0)) % mi;
                        Mc[i][j] = f;
                        uint64_t d;
                        if (sg[j] >= sg[i]) d = f << (sg[j] - sg[i]);
                        else {
                            endo = endo && f % (uint64_t(1) << (sg[i] - sg[j])) == 0;   // f well defined on Z/2^{a_j}
                            d = f >> (sg[i] - sg[j]);
                        }
                        Md[j][i] = d % (uint64_t(1) << sg[j]);
                    }
                for (uint32_t i = 0; i < r; ++i) Mc[i][r + i] = Md[i][r + i] = uint64_t(1) << sg[i];
                const std::vector<uint32_t> ck = CokerExps2(Mc, amax + 1), kk = CokerExps2(Md, amax + 1);
                std::vector<uint32_t> Ks = Kt[c0], Cs = Ct[c0];
                std::sort(Ks.begin(), Ks.end());
                std::sort(Cs.begin(), Cs.end());
                check(endo && Ks == kk && Cs == ck,
                      "(L) c=" + std::to_string(c0) + ptag + ": ker type " + join(Ks, ".") + " = the dual's cokernel " + join(kk, ".") +
                          ", cok type " + join(Cs, ".") + " = coker [F | D] " + join(ck, "."));
                ++c_checked;
            }
        }
        check(any_esc, "(L) the cell exercises the ladder (a read at k_0 fell short)" + tag);
        std::cout << "  (L) " << cellname(1, n) << " l=2 q=1: a=" << join(a, ".") << " k_0=" << std::min(30u, 2 * a.back() + 3) << "; per p ";
        for (size_t o = 0; o < ps.size(); ++o)
            std::cout << (o ? ", " : "") << "T_" << ps[o] << " k=" << E.inv[o].kc_k << (E.inv[o].kc_escalated ? "*" : "");
        std::cout << "; " << fx(secs, 3) << " s" << std::endl;
        for (size_t o = 0; o < ps.size(); ++o) std::cout << "  (L)   T_" << ps[o] << ": " << E.inv[o].canon << std::endl;
    }
    std::cout << "  (L) " << reads << " (cell, p) reads, " << escalated << " escalated to k = 30 (* above), " << c_checked
              << " (c, p) ker / cok types against an independent 2-adic Smith form (coker [F | D], the dual for ker)" << std::endl;
}

void ArmRefusals() {
    const H3::CharpolyZ a = H3::CharpolyOverZ(1, 10, 4);
    check(a.refused && a.code == H3::D3Refusal::NotPrime, "(R) T_4: [NotPrime]: " + a.refusal);
    const H3::CharpolyZ b = H3::CharpolyOverZ(11, 0, 11);
    check(b.refused && b.code == H3::D3Refusal::PDividesN, "(R) T_11 at level 11: [PDividesN]: " + b.refusal);
    const H3::TorsionEntry c = H3::TorsionOnTree(2, 2, 2, 1, {2});
    check(c.refused && c.code == H3::D3Refusal::PDividesN, "(R) T_2 at level 2 on the tree: [PDividesN]: " + c.refusal);
    const H3::TorsionEntry d = H3::TorsionOnTree(2, 1, 10, 5, {2});
    check(d.refused && d.code == H3::D3Refusal::DegreeRange, "(R) q = 5: [DegreeRange]: " + d.refusal);
    const H3::TorsionEntry e = H3::TorsionOnTree(2, 1, 10, 1, {2}, 0, 1ull << 20);
    check(e.refused && e.code == H3::D3Refusal::RssCeiling, "(R) a 1 MiB ceiling: [RssCeiling] before building: " + e.refusal);
    const H3::IsoHeckeN f(1, 10, 100003, U32Ctx(10007), 1);
    check(f.refused() && f.code() == H3::D3Refusal::Lift && f.lift_code() == P::PairRefusal::RssCeiling,
          "(R) p = 100003: the strict lift is refused before building -> [Lift] with D4's [RssCeiling]: " + f.why());
    H3::CrtOptions o;
    o.rss_cap_bytes = 1ull << 20;
    const H3::CharpolyZ g = H3::CharpolyOverZ(1, 10, 2, o);
    check(g.refused && g.code == H3::D3Refusal::RssCeiling, "(R) char_Z under a 1 MiB ceiling: [RssCeiling]: " + g.refusal);
    // (the D3 audit's M1) the l >= 5 cokernel read under a 1 MiB ceiling, and at (1, 70000) under the default 40 GiB one (> 36 GiB of 2 D^2 words):
    // refused before FixedGeneratorRows
    const H3::TorsionEntry h = H3::TorsionOnCokernel(691, 1, 10, {2}, 0, false, true, 1ull << 20);
    check(h.refused && h.code == H3::D3Refusal::RssCeiling && h.read.c == 0, "(R) the cokernel read under a 1 MiB ceiling: [RssCeiling] before building: " + h.refusal);
    const H3::TorsionEntry h2 = H3::TorsionOnCokernel(5, 1, 70000, {13});
    check(h2.refused && h2.code == H3::D3Refusal::RssCeiling && h2.rss_pred > (40ull << 30),
          "(R) the cokernel read at (1,70000) l=5: [RssCeiling] under the default ceiling (predicted " + mib(h2.rss_pred) + " MiB): " + h2.refusal);
    // the record marks itself refused (a legitimate limit), and is not refused when nothing is
    {
        H3::HeckeRecordOptions ro;
        ro.charpoly = false;
        ro.rss_cap_bytes = 1ull << 20;
        const H3::HeckeRecord R1 = H3::RunDescentHecke(1, 10, {2}, ro);
        check(R1.refused && R1.refused_entries == R1.tors.size() && R1.refusal.find("[RssCeiling]") != std::string::npos,
              "(R) the record under a 1 MiB ceiling: refused, every entry [RssCeiling] (" + std::to_string(R1.refused_entries) + "/" +
                  std::to_string(R1.tors.size()) + "): " + R1.refusal);
        const H3::HeckeRecord R2 = H3::RunDescentHecke(1, 10, {2});
        check(!R2.refused && R2.refused_entries == 0 && R2.chi.size() == 1 && R2.bad == std::vector<uint32_t>({2, 3}) && R2.tors.size() == 6,
              "(R) the record at (1,10) p=2 under the default ceiling: not refused, torsion primes 2, 3 (B(10, 1) is empty), q = 1 .. 3: " + R2.refusal +
                  " bad=" + join(R2.bad, ","));
    }
    // the classification: named caps refuse; a genuine internal error is not a refusal
    check(H3::ClassifyD3Exception(hecke::HeckeOverflow("x")) == H3::D3Refusal::IntegerCap, "(R) hecke::HeckeOverflow -> [IntegerCap]");
    check(H3::ClassifyD3Exception(std::overflow_error("x")) == H3::D3Refusal::DomainCap, "(R) std::overflow_error -> [DomainCap]");
    check(H3::ClassifyD3Exception(hecke::HeckeInternalError("x")) == H3::D3Refusal::None, "(R) hecke::HeckeInternalError -> not a refusal (propagates)");
    check(H3::ClassifyD3Exception(std::logic_error("x")) == H3::D3Refusal::None, "(R) an ENGINE_CHECK (logic_error) -> not a refusal (propagates)");
    check(H3::ClassifyD3Exception(std::out_of_range("x")) == H3::D3Refusal::None, "(R) out_of_range -> not a refusal (propagates)");
    std::cout << "  (R) refusals by code: NotPrime, PDividesN (x2), DegreeRange, RssCeiling (x4: tree, char_Z, cokernel x2), Lift[RssCeiling]; the "
                 "record's refused flag; the classification (5)" << std::endl;
}

// =================================================================================================================================
// the cost / reach cells (slow; HECKE_DESCENT_HECKE_CELLS)
// =================================================================================================================================
struct ReachCell {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    std::string mode = "all";
    uint32_t max_primes = 0;   // 0: as many as the bound needs
};
std::vector<ReachCell> ParseCells(const char* env) {
    std::vector<ReachCell> out;
    std::istringstream in(env);
    std::string tok;
    while (std::getline(in, tok, ',')) {
        std::vector<std::string> f;
        std::istringstream ts(tok);
        std::string x;
        while (std::getline(ts, x, ':')) f.push_back(x);
        if (f.size() < 2) continue;
        ReachCell c;
        c.N = static_cast<uint32_t>(std::stoul(f[0]));
        c.n = static_cast<uint32_t>(std::stoul(f[1]));
        if (f.size() >= 3 && !f[2].empty()) {
            std::istringstream ps(f[2]);
            while (std::getline(ps, x, '/')) c.ps.push_back(static_cast<uint32_t>(std::stoul(x)));
        }
        if (f.size() >= 4) c.mode = f[3];
        if (f.size() >= 5) c.max_primes = static_cast<uint32_t>(std::stoul(f[4]));
        if (c.ps.empty()) c.ps = {5, 7, 13};
        out.push_back(c);
    }
    return out;
}
void RunReachCell(const ReachCell& c) {
    const uint32_t D = integral_descent::PsiIndex(c.N) * (c.n + 1);
    std::vector<uint32_t> ps;
    for (uint32_t p : c.ps)
        if (c.N % p) ps.push_back(p);
    if (c.mode == "all" || c.mode == "chi")
        for (uint32_t p : ps) {
            const uint64_t base = integral_descent::CurrentRssBytes();
            H3::CrtOptions co;
            co.max_primes = c.max_primes;
            const H3::CharpolyZ C = H3::CharpolyOverZ(c.N, c.n, p, co);
            check(!C.refused, "(C) char_Z " + cellname(c.N, c.n) + " p=" + std::to_string(p) + ": " + C.refusal);
            uint32_t fit = 0, big = 0;
            for (const auto& x : C.c) {
                fit += x.fits;
                big += x.certified && !x.fits;
            }
            std::cout << "DESCENT3_COST N=" << c.N << " n=" << c.n << " D=" << D << " part=chi p=" << p << " r=" << C.r << " primes=" << C.primes.size()
                      << " determined=" << C.determined << " int64=" << fit << " digits=" << big << " held_out=" << (C.held_out_ok ? "ok" : "FAIL")
                      << " residue_s=" << fx(C.residue_seconds) << " garner_s=" << fx(C.garner_seconds) << " total_s=" << fx(C.seconds)
                      << " rss_pred_mib=" << mib(C.rss_pred) << " rss_meas_mib=" << mib(C.rss_meas) << " base_mib=" << mib(base)
                      << " undetermined=" << C.undetermined() << " chi_fnv=0x" << std::hex << std::setw(16) << std::setfill('0') << C.digest()
                      << std::dec << std::setfill(' ') << std::endl;
            if (C.r <= 40) std::cout << "  " << cellname(c.N, c.n) << " p=" << p << ": char_Z = " << C.text() << std::endl;
        }
    if (c.mode == "all" || c.mode == "tors") {
        H3::HeckeRecordOptions o;
        o.charpoly = false;
        o.full_invariants = D <= 2000;
        const H3::HeckeRecord R = H3::RunDescentHecke(c.N, c.n, ps, o);
        for (const auto& E : R.tors) {
            check(!E.refused || E.code == H3::D3Refusal::RssCeiling, "(C) torsion " + cellname(c.N, c.n) + " l=" + std::to_string(E.ell) + " q=" +
                                                                      std::to_string(E.q) + ": " + E.refusal);
            std::cout << "DESCENT3_COST N=" << c.N << " n=" << c.n << " D=" << D << " part=tors l=" << E.ell << " q=" << E.q << " route=" << E.route
                      << " c=" << E.read.c << " summands=" << E.read.sigma.size() << " smax=" << E.read.smax
                      << " module=" << E.read.module_certified << " commute=" << E.commute << " smith_s=" << fx(E.read.smith_seconds)
                      << " apply_s=" << fx(E.read.apply_seconds) << " total_s=" << fx(E.seconds) << " rss_pred_mib=" << mib(E.rss_pred)
                      << (E.refused ? " REFUSED[" + std::string(H3::D3RefusalName(E.code)) + "]" : std::string()) << std::endl;
            if (!E.inv.empty()) std::cout << "  " << cellname(c.N, c.n) << " l=" << E.ell << " q=" << E.q << " p=" << ps.front() << ": " << (E.inv[0].layers.size() > 400 ? E.inv[0].layers.substr(0, 400) + "... (" + std::to_string(E.inv[0].layers.size()) + " chars)" : E.inv[0].layers) << std::endl;
        }
        std::cout << "DESCENT3_COST N=" << c.N << " n=" << c.n << " D=" << D << " part=tors_total s=" << fx(R.t_tors) << " rss_peak_mib=" << mib(R.rss_peak)
                  << std::endl;
    }
}

}  // namespace

uint32_t IntegralDescentHeckeValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    std::cout << "IntegralDescentHeckeValidation: T_p (p !| N) on H^q(Gamma_0(N); Sym^n Z^2) over Z as invariants ([C] sec:descenthecke); "
                 "HECKE_DESCENT_TREE: tree=" << integral_descent::DescentTreeEnabled() << " source=" << integral_descent::DescentTreeSource() << " (the groups' route; Hecke at 2, 3 is always the transport)"
              << std::endl;
    // one try per arm (the D3 audit's N4): an exception fails its arm and the later arms still run
    OracleStats st;
    const std::vector<std::pair<const char*, std::function<void()>>> arms = {
        {"O", [&] { ArmOracle(st, false); }}, {"S", [] { ArmShipped(false); }}, {"X", [] { ArmTransport(false); }}, {"M", [] { ArmOperator(); }},
        {"W", [] { ArmWords(); }},            {"B", [] { ArmBound(); }},        {"K", [] { ArmPrecision(); }},       {"N", [] { ArmNegative(); }},
        {"R", [] { ArmRefusals(); }},        {"L", [] { ArmLadder(); }}};
    for (const auto& [name, arm] : arms) {
        try {
            arm();
        } catch (const std::exception& e) {
            check(false, std::string("(") + name + ") exception: " + e.what());
        }
    }
    std::cout << "IntegralDescentHeckeValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0), 3) << " s" << std::endl;
    return g_fail;
}

uint32_t IntegralDescentHeckeSlowValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = clk::now();
    // one try per arm and per cost cell (the D3 audit's N4)
    auto guarded = [](const std::string& name, const std::function<void()>& f) {
        try {
            f();
        } catch (const std::exception& e) {
            check(false, "(" + name + ") exception: " + e.what());
        }
    };
    const char* env = std::getenv("HECKE_DESCENT_HECKE_CELLS");
    if (env && *env) {
        std::vector<ReachCell> cells;
        guarded("cells", [&] { cells = ParseCells(env); });
        std::cout << "EFFECTIVE IntegralDescentHeckeSlowValidation arm=HECKE_DESCENT_HECKE_CELLS cells=" << cells.size() << std::endl;
        for (const auto& c : cells) guarded("C " + cellname(c.N, c.n), [&] { RunReachCell(c); });
    } else {
        std::cout << "EFFECTIVE IntegralDescentHeckeSlowValidation arm=default cells=10" << std::endl;
        OracleStats st;
        guarded("O", [&] { ArmOracle(st, true); });
        guarded("S", [] { ArmShipped(true); });
        guarded("X", [] { ArmTransport(true); });
        for (const ReachCell& c : std::vector<ReachCell>{{1, 60, {2, 3, 5, 7, 13}, "all"}, {1, 100, {2, 13}, "chi"}, {37, 2, {5, 7, 13}, "all"},
                                                         {11, 10, {2, 7}, "all"}, {101, 0, {5, 7, 13}, "all"}, {23, 4, {5, 7}, "all"},
                                                         {1, 200, {13}, "chi"}, {1009, 0, {5, 7, 13}, "chi"}, {101, 10, {13}, "tors"},
                                                         {211, 10, {13}, "tors"}})
            guarded("C " + cellname(c.N, c.n), [&] { RunReachCell(c); });
    }
    std::cout << "IntegralDescentHeckeSlowValidation: " << g_checks << " checks, " << g_fail << " failing, " << fx(since(t0), 3) << " s" << std::endl;
    return g_fail;
}
