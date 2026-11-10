// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 11, WP-6 -- HeckeDicksonThetaHeckeValidation (standard)
// and HeckeDicksonThetaHeckeCoverageSlowValidation (slow).  The unit is
// HeckeDicksonThetaHecke.h; the predictions these arms test were committed
// before the first run (docs/implementation/slot11/WP-6_report.md Sec. 1,
// slot-11 rule 20).
//
// Arms (standard):
//   (G)   per grid cell -- ell in {5, 7} at even n <= 40, ell in {11, 13} at even
//         n <= 26, plus (120, 5) -- RunThetaHecke with p in {2,3,5,7}\{ell} and
//         U_ell: (T), (E), (U), (H0), (K), (B) of the unit ([C, prop:thetatwist],
//         [C, thm:thetales](2),(3),(5), [C, prop:heckeH0], [C, lem:Qnfunctions](3),
//         [C, lem:heckefibre]); the cochain arm (F) at n <= 20;
//   (I)   the named instances C3-C6 of [C, rmk:thetacheckable], each target
//         block looked up in Sigma(ell) (InProcessAtlas) by its eigenvalues at
//         p <= 13, the matched system's full anemic key checked to be the
//         p-twist of the source's;
//   (C7)  the filtration instances at ell = 11, 13 for n <= 26 (recorded as
//         CONJ-backed observations, asserted as predicted).
// Under HECKE_THETA_IMAGE_ALL=1 (read-only interlock, not an emitter; the
// HECKE_INTEGRAL_CAMPAIGN pattern) additionally:
//   (P5)  every published ext_first cell (N, ell), ell <= 13, N >= ell + 1:
//         RunThetaHecke at n = N - ell - 1, the run's H^1(M_N) blocks matched
//         one-to-one with the record's H1 ideals by witness_poly and the
//         restricted t_p (exact equality, [E] sec:census (e)), one THETA_IMAGE
//         line per H1 ideal ([E] sec:thetaobj (f)), sum_m d_theta = rank theta^1.
//   Slot-12 WP-6: HECKE_THETA_IMAGE_CELLS=<file> (read-only interlock, the
//         census cell-list format of HeckeCensusCellList.h) runs (P5) over
//         the listed cells instead (census edition 2's W2 cells), each
//         record looked up in ext_first/ext, then ext_census_l7l11/ext;
//         cells with ell > 13 or N < ell + 1 are skipped (d_theta = 0 by
//         definition there).  Without it (P5) is edition 1's loop exactly.
//         When both are set HECKE_THETA_IMAGE_CELLS wins (printed); the list's
//         u_ell budget field is not read -- (P5) is record-based (slot-12 WP-11, A12).
// Arms (slow): (G) on a sparse grid n <= 200 at all four ell, C7 at ell = 13
// up to n = 26 (the p^2 tau(p) decision), C9 at ell = 5, the theta-cycles
// theta^{ell-1} from n0 = 10.

#include "HeckeDicksonThetaHecke.h"
#include "HeckeAtlas.h"
#include "HeckeCensusCellList.h"

#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/SubspaceOps.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hecke_ext::DicksonThetaCell;
using hecke_ext::ThetaHeckeOptions;
using hecke_ext::ThetaHeckeRun;
using hecke_ext::ThetaMat;
using hecke_ext::ThetaTerm;

struct Counter {
    uint64_t checks = 0, failures = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "    FAIL: " << what << "\n";
        }
    }
    void absorb(const ThetaHeckeRun& r) {
        for (const auto& c : r.checks) check(c.ok, c.name + (c.detail.empty() ? "" : " [" + c.detail + "]"));
    }
};

uint32_t PowMod(uint64_t b, uint64_t e, uint32_t p) {
    uint64_t r = 1 % p;
    b %= p;
    while (e) {
        if (e & 1u) r = (r * b) % p;
        b = (b * b) % p;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}
uint32_t Mod(int64_t v, uint32_t p) {
    const int64_t r = v % static_cast<int64_t>(p);
    return static_cast<uint32_t>(r < 0 ? r + p : r);
}

// Ramanujan's tau at the primes the arms use.
int64_t Tau(uint64_t p) {
    switch (p) {
        case 2: return -24;
        case 3: return 252;
        case 5: return 4830;
        case 7: return -16744;
        case 11: return 534612;
        case 13: return -577738;
        default: return 0;
    }
}

// The anemic primes <= 13 at ell.
std::vector<uint64_t> Small(uint32_t ell) {
    std::vector<uint64_t> v;
    for (uint64_t p : { 2u, 3u, 5u, 7u, 11u, 13u })
        if (p != ell) v.push_back(p);
    return v;
}

using System = std::vector<uint32_t>;   // eigenvalues at Small(ell)
System SysTau(uint32_t ell, uint32_t twist) {
    System s;
    for (uint64_t p : Small(ell)) s.push_back(static_cast<uint32_t>((static_cast<uint64_t>(Mod(Tau(p), ell)) * PowMod(p, twist, ell)) % ell));
    return s;
}
System SysEis(uint32_t ell, uint32_t a, uint32_t b) {   // p^a + p^b
    System s;
    for (uint64_t p : Small(ell)) s.push_back((PowMod(p, a, ell) + PowMod(p, b, ell)) % ell);
    return s;
}
// 1 + p^{-1} = 1 + p^{ell-2}
System SysEisInv(uint32_t ell) { return SysEis(ell, 0, ell - 2); }

std::string Str(const System& s) {
    std::string o = "(";
    for (size_t i = 0; i < s.size(); ++i) o += (i ? "," : "") + std::to_string(s[i]);
    return o + ")";
}

// Generalised eigenspace of a system on H^1 of a term of a run whose primes
// include Small(ell).
ThetaMat Eig(const ThetaHeckeRun& r, ThetaTerm t, const System& s) {
    std::vector<ThetaMat> ops;
    const auto sm = Small(r.ell);
    for (uint64_t p : sm) ops.push_back(r.op(t, p, 1));
    if (ops[0].rows == 0) return ThetaMat(0, 0, r.ell);
    return hecke_ext::ThetaGeneralisedEigenspace(ops, s);
}

std::vector<uint64_t> PrimesSmallAndU(uint32_t ell) {
    std::vector<uint64_t> v = Small(ell);
    v.push_back(ell);
    std::sort(v.begin(), v.end());
    return v;
}

// ---------------------------------------------------------------------------
// Sigma(ell): the r = 1 systems whose eigenvalues at Small(ell) are `s`.
// ---------------------------------------------------------------------------
struct AtlasHit {
    int64_t id = -1;
    uint32_t count = 0;
    std::string type;
    uint32_t base = 0, twist = 0;
};
AtlasHit AtlasLookup(const hecke_ext::AtlasStore& st, uint32_t ell, const System& s) {
    AtlasHit h;
    const auto sm = Small(ell);
    std::vector<int64_t> idx;
    for (uint64_t p : sm) {
        const auto it = std::find(st.primes.begin(), st.primes.end(), p);
        idx.push_back(it == st.primes.end() ? -1 : static_cast<int64_t>(it - st.primes.begin()));
    }
    for (const auto& sys : st.systems) {
        if (sys.r != 1) continue;
        bool eq = true;
        for (size_t i = 0; i < sm.size() && eq; ++i) {
            if (idx[i] < 0) continue;
            eq = sys.eigenvalues[static_cast<size_t>(idx[i])].size() == 1 && sys.eigenvalues[static_cast<size_t>(idx[i])][0] == s[i];
        }
        if (eq) {
            ++h.count;
            if (h.id < 0) {
                h.id = sys.id;
                h.type = sys.type;
                h.base = sys.base;
                h.twist = sys.twist;
            }
        }
    }
    return h;
}
// The full-pool key of `tgt` equals the p-twist of `src`'s (both r = 1).
bool AtlasIsTwist(const hecke_ext::AtlasStore& st, uint32_t ell, int64_t src, int64_t tgt) {
    if (src < 0 || tgt < 0) return false;
    const auto& a = st.systems.at(static_cast<size_t>(src));
    const auto& b = st.systems.at(static_cast<size_t>(tgt));
    for (size_t i = 0; i < st.primes.size(); ++i)
        if (b.eigenvalues[i][0] != (static_cast<uint64_t>(a.eigenvalues[i][0]) * (st.primes[i] % ell)) % ell) return false;
    return true;
}

// ---------------------------------------------------------------------------
// (G) the grid
// ---------------------------------------------------------------------------
struct Cell {
    uint32_t n, ell;
};

void RunCell(const Cell& g, bool cochain, Counter& ct, bool verbose) {
    const DicksonThetaCell c(g.n, g.ell);
    ThetaHeckeOptions o;
    o.primes = hecke_ext::ThetaDefaultPrimes(g.ell);
    o.cochain_arm = cochain;
    const ThetaHeckeRun r = hecke_ext::RunThetaHecke(c, o);
    ct.absorb(r);
    if (verbose)
        std::cout << "  " << c.summary() << " checks=" << r.checks.size() << " blocks=" << r.blocks_n.size() << "/"
                  << r.blocks_m.size() << " U_rank_H0=" << r.u_rank_h0_n << "/" << r.u_rank_h0_m << "\n";
}

void RunGrid(const std::vector<Cell>& grid, uint32_t cochain_max_n, Counter& ct, bool verbose) {
    for (const Cell& g : grid) RunCell(g, g.n <= cochain_max_n, ct, verbose);
    std::cout << "  (G) " << grid.size() << " cells\n";
}

// ---------------------------------------------------------------------------
// (I) the instances C3-C6
// ---------------------------------------------------------------------------
struct Instance {
    const char* name;
    uint32_t ell, n;
    System src, tgt;
    uint32_t src_dim, tgt_dim;   // generalised eigenspace dims (tgt: of the target system)
    uint32_t rank_delta, h1n, h1m, h1q;
};

void InstanceArm(Counter& ct) {
    std::vector<Instance> list;
    // C3: ell = 5, 4 -> 10, 1 + p (Eisenstein of weight 6) -> p + p^2 = tau(p) mod 5.
    list.push_back({ "C3", 5, 4, SysEis(5, 0, 1), SysEis(5, 1, 2), 1, 2, 0, 1, 3, 2 });
    // C4: ell = 7, 2 -> 10, 1 + p^3 -> p + p^4 = tau(p) mod 7.
    list.push_back({ "C4", 7, 2, SysEis(7, 0, 3), SysEis(7, 1, 4), 1, 2, 0, 1, 3, 2 });
    // C5: ell = 11, 10 -> 22: 1 + p -> p + p^2, tau -> p tau; both target
    // eigenspaces are 2-dimensional ([C] v1.11.0-pre.2 (C5), the referee's m-2).
    list.push_back({ "C5-eis", 11, 10, SysEis(11, 0, 1), SysEis(11, 1, 2), 1, 2, 0, 3, 5, 2 });
    list.push_back({ "C5-delta", 11, 10, SysTau(11, 0), SysTau(11, 1), 2, 2, 0, 3, 5, 2 });
    // C6: ell = 13, 10 -> 24: the Delta pair -> p tau (block dim 2).
    list.push_back({ "C6-delta", 13, 10, SysTau(13, 0), SysTau(13, 1), 2, 2, 1, 3, 3, 1 });
    // The C6 predictions on the killed line and on 1 + p follow below.
    std::map<uint32_t, std::shared_ptr<const hecke_ext::Atlas>> atlases;
    for (const Instance& in : list) {
        const DicksonThetaCell c(in.n, in.ell);
        ThetaHeckeOptions o;
        o.primes = PrimesSmallAndU(in.ell);
        const ThetaHeckeRun r = hecke_ext::RunThetaHecke(c, o);
        ct.absorb(r);
        const std::string tag = std::string("(I) ") + in.name + " (" + std::to_string(in.n) + "->" + std::to_string(c.m()) +
                                ", ell=" + std::to_string(in.ell) + ")";
        ct.check(r.rank_delta == in.rank_delta, tag + " rank delta = " + std::to_string(in.rank_delta));
        ct.check(c.h(ThetaTerm::SymN, 1) == in.h1n && c.h(ThetaTerm::SymM, 1) == in.h1m && c.h(ThetaTerm::Quot, 1) == in.h1q,
                 tag + " dims h1(n), h1(m), h1(Q)");
        const ThetaMat Es = Eig(r, ThetaTerm::SymN, in.src);
        const ThetaMat Et = Eig(r, ThetaTerm::SymM, in.tgt);
        const ThetaMat img = hecke_ext::ThetaMul(c.theta(1), Es);
        const uint32_t rimg = hecke_ext::ThetaRank(img);
        const bool at_least = false;   // [C] pre.2: every target dimension of C3-C6 is exact
        ct.check(Es.cols == in.src_dim, tag + " dim E_src" + Str(in.src) + " = " + std::to_string(in.src_dim) + " (got " + std::to_string(Es.cols) + ")");
        ct.check(at_least ? Et.cols >= in.tgt_dim : Et.cols == in.tgt_dim,
                 tag + " dim E_tgt" + Str(in.tgt) + (at_least ? " >= " : " = ") + std::to_string(in.tgt_dim) + " (got " + std::to_string(Et.cols) + ")");
        ct.check(rimg == in.src_dim, tag + " theta injective on E_src");
        ct.check(hecke_ext::ThetaSpanContains(Et, img), tag + " theta(E_src) inside E_tgt");
        std::printf("  %s: E_src dim %u -> theta image %u inside E_tgt dim %u (h1: %u -> %u, h1(Q) %u, rank delta %u)\n",
                    tag.c_str(), Es.cols, rimg, Et.cols, in.h1n, in.h1m, in.h1q, r.rank_delta);
        // Sigma(ell).
        if (!atlases.count(in.ell)) atlases[in.ell] = hecke_ext::InProcessAtlas(in.ell);
        const auto& st = atlases[in.ell]->store;
        const AtlasHit hs = AtlasLookup(st, in.ell, in.src), ht = AtlasLookup(st, in.ell, in.tgt);
        ct.check(hs.count == 1 && ht.count == 1, tag + " source and target each match ONE system of Sigma(ell) at p <= 13");
        ct.check(AtlasIsTwist(st, in.ell, hs.id, ht.id), tag + " Sigma(ell): target key = p-twist of the source key on the full anemic pool");
        std::printf("    Sigma(%u): source system %lld (%s, base %u, twist %u) -> target system %lld (%s, base %u, twist %u)\n",
                    in.ell, static_cast<long long>(hs.id), hs.type.c_str(), hs.base, hs.twist, static_cast<long long>(ht.id),
                    ht.type.c_str(), ht.base, ht.twist);

        if (std::string(in.name) == "C6-delta") {
            // ker theta = the Eisenstein line 1 + p^{11} = 1 + p^{-1}; the class 1 + p of H^1(M_24) maps onto H^1(Q_10).
            const ThetaMat Eeis = Eig(r, ThetaTerm::SymN, SysEisInv(13));
            const ThetaMat ker = hecke_ext::ThetaKernelBasis(c.theta(1));
            ct.check(Eeis.cols == 1 && ker.cols == 1 && hecke_ext::ThetaSpanContains(Eeis, ker),
                     "(I) C6 theta kills exactly the Eisenstein line (1 + p^{-1})");
            const ThetaMat E1p = Eig(r, ThetaTerm::SymM, SysEis(13, 0, 1));
            ct.check(E1p.cols == 1 && hecke_ext::ThetaRank(hecke_ext::ThetaMul(c.pi(1), E1p)) == 1,
                     "(I) C6 the class 1 + p of H^1(M_24) maps onto H^1(Q_10)");
        }
        if (std::string(in.name) == "C3" || std::string(in.name) == "C4") {
            const ThetaMat E3 = Eig(r, ThetaTerm::SymM, SysEisInv(in.ell));
            ct.check(E3.cols == 1, tag + " the third class of H^1(M_10) has 1 + p^{-1}");
        }
    }
}

// ---------------------------------------------------------------------------
// (C7) filtration: does a system occur in H^1(M_n)?
// ---------------------------------------------------------------------------
uint32_t OccursDim(uint32_t n, uint32_t ell, const System& s) {
    const DicksonThetaCell c(n, ell);
    ThetaHeckeOptions o;
    o.primes = Small(ell);
    o.tau_arm = false;
    o.block_arm = false;
    const ThetaHeckeRun r = hecke_ext::RunThetaHecke(c, o);
    return Eig(r, ThetaTerm::SymN, s).cols;
}

void C7Arm(Counter& ct, bool slow) {
    const System pt11 = SysTau(11, 1);
    for (uint32_t n : { 2u, 12u, 22u }) {
        const uint32_t d = OccursDim(n, 11, pt11);
        std::printf("  (C7) ell=11 n=%u: dim E_{p tau(p)} = %u\n", n, d);
        ct.check(n == 22 ? d > 0 : d == 0, "(C7) ell=11 p tau(p) " + std::string(n == 22 ? "present" : "absent") + " at n=" + std::to_string(n));
    }
    const System pt13 = SysTau(13, 1), p2t13 = SysTau(13, 2);
    for (uint32_t n : { 12u, 24u }) {
        const uint32_t d = OccursDim(n, 13, pt13);
        std::printf("  (C7) ell=13 n=%u: dim E_{p tau(p)} = %u\n", n, d);
        ct.check(n == 24 ? d > 0 : d == 0, "(C7) ell=13 p tau(p) " + std::string(n == 24 ? "present" : "absent") + " at n=" + std::to_string(n));
    }
    if (slow) {
        // p^2 tau(p) at ell = 13: first at n = 14 or n = 26 -- the model decides (recorded).
        uint32_t first = 0;
        for (uint32_t n = 0; n <= 26 && first == 0; n += 2) {
            const uint32_t d = OccursDim(n, 13, p2t13);
            std::printf("  (C7) ell=13 n=%u: dim E_{p^2 tau(p)} = %u\n", n, d);
            if (d > 0) first = n;
        }
        std::printf("  (C7) ell=13: p^2 tau(p) first occurs at n = %u\n", first);
        ct.check(first == 14 || first == 26, "(C7) ell=13 p^2 tau(p) first occurs at n in {14, 26}");
    }
}

// ---------------------------------------------------------------------------
// (C9) ell = 5, targets n = 34 ... 44: the dim >= 4 blocks against their theta sources.
// ---------------------------------------------------------------------------
void C9Arm(Counter& ct) {
    for (uint32_t N = 34; N <= 44; N += 2) {
        const DicksonThetaCell c(N - 6, 5);
        ThetaHeckeOptions o;
        o.primes = hecke_ext::ThetaDefaultPrimes(5);
        const ThetaHeckeRun r = hecke_ext::RunThetaHecke(c, o);
        ct.absorb(r);
        const ThetaMat& th = c.theta(1);
        for (size_t ci = 0; ci < r.blocks_m.size(); ++ci) {
            const auto& C = r.blocks_m[ci];
            if (C.dim < 4) continue;
            const uint32_t dth = hecke_ext::ThetaIntersectionDim(C.basis, th);
            std::string src;
            uint32_t src_total = 0;
            for (size_t bi = 0; bi < r.blocks_n.size(); ++bi) {
                const ThetaMat img = hecke_ext::ThetaMul(th, r.blocks_n[bi].basis);
                if (hecke_ext::ThetaRank(img) > 0 && hecke_ext::ThetaSpanContains(C.basis, img)) {
                    src += " B" + std::to_string(bi) + "(dim " + std::to_string(r.blocks_n[bi].dim) + ")";
                    src_total += r.blocks_n[bi].dim;
                }
            }
            std::printf("  (C9) target n=%u block dim %u: theta-image dim %u, sources at n=%u:%s, rank delta(n-6)=%u\n", N, C.dim,
                        dth, N - 6, src.empty() ? " none" : src.c_str(), r.rank_delta);
            ct.check(C.dim + r.rank_delta >= src_total, "(C9) multiplicity inequality at target n=" + std::to_string(N));
        }
        const bool pred = (N == 36 || N == 44);
        ct.check((r.rank_delta == 1) == pred, "(C9) rank delta(n-6) = 1 exactly at targets 36, 44 (target n=" + std::to_string(N) + ")");
    }
}

// ---------------------------------------------------------------------------
// (P7) theta-cycles theta^{ell-1} on the Delta block from n0 = 10.
// ---------------------------------------------------------------------------
void CycleArm(Counter& ct) {
    for (uint32_t ell : { 5u, 7u, 11u, 13u }) {
        const uint32_t n0 = 10, steps = ell - 1;
        std::vector<std::unique_ptr<DicksonThetaCell>> cells;
        for (uint32_t k = 0; k < steps; ++k) cells.push_back(std::make_unique<DicksonThetaCell>(n0 + k * (ell + 1), ell));
        // The stored bases chain: SymM of cell k is SymN of cell k + 1.
        bool chain = true;
        for (uint32_t k = 0; k + 1 < steps && chain; ++k) {
            const uint32_t h = cells[k]->h(ThetaTerm::SymM, 1);
            chain = h == cells[k + 1]->h(ThetaTerm::SymN, 1);
            for (uint32_t j = 0; j < h && chain; ++j)
                chain = cells[k]->basis_cochain(ThetaTerm::SymM, 1, j) == cells[k + 1]->basis_cochain(ThetaTerm::SymN, 1, j);
        }
        const std::string tag = "(P7) ell=" + std::to_string(ell);
        ct.check(chain, tag + " the stored H^1 bases chain from cell to cell");
        if (!chain) continue;
        ThetaHeckeOptions o;
        o.primes = Small(ell);
        o.tau_arm = false;
        o.block_arm = false;
        const ThetaHeckeRun r0 = hecke_ext::RunThetaHecke(*cells.front(), o);
        const ThetaHeckeRun r1 = hecke_ext::RunThetaHecke(*cells.back(), o);
        const System a = SysTau(ell, 0);
        const ThetaMat Es = Eig(r0, ThetaTerm::SymN, a);
        const ThetaMat Ee = Eig(r1, ThetaTerm::SymM, a);
        ThetaMat v = Es;
        std::string ranks;
        uint32_t predicted_loss_steps = 0;
        for (uint32_t k = 0; k < steps; ++k) {
            v = hecke_ext::ThetaMul(cells[k]->theta(1), v);
            ranks += " " + std::to_string(hecke_ext::ThetaRank(v));
            if (SysTau(ell, k) == SysEisInv(ell) && hecke_ext::ThetaRank(cells[k]->delta(0)) == 1) ++predicted_loss_steps;
        }
        const uint32_t rk = hecke_ext::ThetaRank(v);
        std::printf("  %s: Delta block dim %u at n=%u, theta^%u ranks after each step:%s; end block dim %u at n=%u; loss steps %u\n",
                    tag.c_str(), Es.cols, n0, steps, ranks.c_str(), Ee.cols, cells.back()->m(), predicted_loss_steps);
        ct.check(Es.cols == 2, tag + " the Delta block at n0 = 10 has dim 2");
        ct.check(hecke_ext::ThetaSpanContains(Ee, v), tag + " theta^{ell-1}(E_a) lies in E_a at n0 + ell^2 - 1");
        ct.check(rk + predicted_loss_steps >= Es.cols, tag + " rank >= dim B - (loss steps)");
        if (ell != 5) ct.check(rk == Es.cols, tag + " theta^{ell-1} injective on the Delta block (never Eisenstein)");
        else ct.check(predicted_loss_steps == 1, tag + " exactly one loss step (j = 2, n = 22) predicted");
    }
}

// ---------------------------------------------------------------------------
// (P5) the census column, under HECKE_THETA_IMAGE_ALL=1.
// ---------------------------------------------------------------------------
// A minimal JSON reader (objects, arrays, strings, numbers, literals).
// (S13 WP-12, slot-11 m-6) Strings understand only the escapes \n and \<char>: no \uXXXX, \t or \r.
// Fine for every shipped record (all 384 parse); revisit if a record ever carries one in a string.
struct JV {
    enum K { Null, Bool, Num, Str, Arr, Obj } k = Null;
    std::string s;   // Num: its text; Str: the string
    bool b = false;
    std::vector<JV> a;
    std::vector<std::pair<std::string, JV>> o;
    const JV* get(const std::string& key) const {
        for (const auto& kv : o)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    int64_t i() const { return std::stoll(s); }
};
struct JP {
    const std::string& t;
    size_t p = 0;
    void ws() { while (p < t.size() && std::isspace(static_cast<unsigned char>(t[p]))) ++p; }
    JV val() {
        ws();
        JV v;
        const char c = t.at(p);
        if (c == '{') {
            v.k = JV::Obj;
            ++p;
            ws();
            if (t[p] == '}') { ++p; return v; }
            for (;;) {
                ws();
                JV key = val();
                ws();
                ++p;   // ':'
                JV x = val();
                v.o.emplace_back(key.s, std::move(x));
                ws();
                if (t[p] == ',') { ++p; continue; }
                ++p;   // '}'
                return v;
            }
        }
        if (c == '[') {
            v.k = JV::Arr;
            ++p;
            ws();
            if (t[p] == ']') { ++p; return v; }
            for (;;) {
                v.a.push_back(val());
                ws();
                if (t[p] == ',') { ++p; continue; }
                ++p;
                return v;
            }
        }
        if (c == '"') {
            v.k = JV::Str;
            ++p;
            while (t[p] != '"') {
                if (t[p] == '\\') { ++p; v.s += t[p] == 'n' ? '\n' : t[p]; ++p; continue; }
                v.s += t[p++];
            }
            ++p;
            return v;
        }
        if (t.compare(p, 4, "true") == 0) { v.k = JV::Bool; v.b = true; p += 4; return v; }
        if (t.compare(p, 5, "false") == 0) { v.k = JV::Bool; p += 5; return v; }
        if (t.compare(p, 4, "null") == 0) { p += 4; return v; }
        v.k = JV::Num;
        while (p < t.size() && (std::isdigit(static_cast<unsigned char>(t[p])) || t[p] == '-' || t[p] == '+' || t[p] == '.' || t[p] == 'e' || t[p] == 'E')) v.s += t[p++];
        return v;
    }
};

std::string ExtFirstDir() { return std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results/ext_first/ext"; }

std::string RecordPathIn(const std::string& dir, uint32_t N, uint32_t ell) {
    const std::string prefix = "hecke_ext_ell" + std::to_string(ell) + "_n" + std::to_string(N) + "_x";
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        const std::string f = e.path().filename().string();
        if (f.rfind(prefix, 0) == 0) return e.path().string();
    }
    return "";
}

std::string RecordPath(uint32_t N, uint32_t ell) { return RecordPathIn(ExtFirstDir(), N, ell); }

// Slot-12 WP-6: ext_first first, then the W2 set ext_census_l7l11.
std::string RecordPathW2(uint32_t N, uint32_t ell) {
    const std::string a = RecordPathIn(ExtFirstDir(), N, ell);
    if (!a.empty()) return a;
    return RecordPathIn(std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results/ext_census_l7l11/ext", N, ell);
}

void CensusArm(Counter& ct, uint32_t& lines, const std::vector<std::pair<uint32_t, uint32_t>>* listed = nullptr) {
    std::vector<std::pair<uint32_t, uint32_t>> cells;   // (N, ell)
    if (listed == nullptr) {
        for (uint32_t ell : { 5u, 7u, 11u, 13u })
            for (uint32_t N = ell + 1; N <= 200; N += 2) cells.push_back({ N, ell });
    } else {
        for (const auto& c : *listed)
            if (c.second <= 13 && c.first >= c.second + 1) cells.push_back(c);
    }
    for (const auto& cell : cells) {
        const uint32_t N = cell.first, ell = cell.second;
        {
            const std::string path = listed == nullptr ? RecordPath(N, ell) : RecordPathW2(N, ell);
            const std::string tag = "(P5) (" + std::to_string(N) + "," + std::to_string(ell) + ")";
            ct.check(!path.empty(), tag + " record present");
            if (path.empty()) continue;
            std::ifstream in(path);
            std::stringstream ss;
            ss << in.rdbuf();
            const std::string text = ss.str();
            JP parser{ text };
            const JV rec = parser.val();
            std::vector<uint64_t> primes;
            for (const JV& x : rec.get("operators_computed")->a) primes.push_back(static_cast<uint64_t>(x.i()));

            const DicksonThetaCell c(N - ell - 1, ell);
            ThetaHeckeOptions o;
            o.primes = hecke_ext::ThetaDefaultPrimes(ell);
            const ThetaHeckeRun r = hecke_ext::RunThetaHecke(c, o);
            ct.absorb(r);
            ct.check(primes == o.primes, tag + " the record's operators_computed is the run's prime set");
            if (primes != o.primes) continue;

            // The driver's Stage-2 decomposition of H^1(M_N), on the same operators in the same order.
            const uint32_t h1 = c.h(ThetaTerm::SymM, 1);
            std::vector<hecke::PrimaryBlock> blocks;
            std::vector<std::pair<uint64_t, MatrixT<uint32_t>>> ops;
            if (h1 > 0) {
                for (uint64_t p : primes) ops.push_back({ p, r.op(ThetaTerm::SymM, p, 1).to_matrix() });
                hecke::HeckeModuleDecomposition dec(ops, ell);
                dec.compute();
                blocks = dec.blocks();
            }
            // The record's H1 ideals.
            struct RI { uint32_t id, dim; std::vector<uint32_t> w; std::map<uint64_t, std::vector<std::vector<uint32_t>>> t; };
            std::vector<RI> ris;
            for (const JV& id : rec.get("ideals")->a) {
                std::string own;
                for (const JV& g : id.get("groups")->a)
                    if (g.get("own") && g.get("own")->b) own = g.get("group")->s;
                if (own != "H1") continue;
                RI x;
                x.id = static_cast<uint32_t>(id.get("id")->i());
                x.dim = static_cast<uint32_t>(id.get("dim")->i());
                if (id.get("witness_poly") && id.get("witness_poly")->k == JV::Arr)
                    for (const JV& v : id.get("witness_poly")->a) x.w.push_back(static_cast<uint32_t>(v.i()));
                if (id.get("t_p") && id.get("t_p")->k == JV::Obj)
                    for (const auto& kv : id.get("t_p")->o) {
                        std::vector<std::vector<uint32_t>> M;
                        for (const JV& row : kv.second.a) {
                            std::vector<uint32_t> rr;
                            for (const JV& e : row.a) rr.push_back(static_cast<uint32_t>(e.i()));
                            M.push_back(rr);
                        }
                        x.t[static_cast<uint64_t>(std::stoull(kv.first))] = M;
                    }
                ris.push_back(x);
            }
            ct.check(ris.size() == blocks.size(), tag + " as many run blocks as record H1 ideals (" + std::to_string(blocks.size()) + " vs " + std::to_string(ris.size()) + ")");
            // Match one-to-one.
            std::vector<int> match(blocks.size(), -1);
            std::vector<int> used(ris.size(), 0);
            for (size_t bi = 0; bi < blocks.size(); ++bi) {
                const auto& b = blocks[bi];
                std::map<uint64_t, std::vector<std::vector<uint32_t>>> tb;
                for (const auto& op : ops) {
                    const MatrixT<uint32_t> tp = hecke::SolveColumnsExact(b.basis_columns, op.second * b.basis_columns, "THETA_IMAGE restriction");
                    tb[op.first] = tp.data;
                }
                for (size_t ri = 0; ri < ris.size(); ++ri) {
                    if (used[ri] || ris[ri].dim != b.dim || ris[ri].w != b.witness_poly.coeffs || ris[ri].t != tb) continue;
                    match[bi] = static_cast<int>(ri);
                    used[ri] = 1;
                    break;
                }
                ct.check(match[bi] >= 0, tag + " run block " + std::to_string(bi) + " matches a record H1 ideal (witness_poly + t_p)");
            }
            const ThetaMat& th = c.theta(1);
            uint32_t sum = 0;
            for (size_t bi = 0; bi < blocks.size(); ++bi) {
                if (match[bi] < 0) continue;
                const RI& x = ris[static_cast<size_t>(match[bi])];
                const uint32_t dth = hecke_ext::ThetaIntersectionDim(hecke_ext::ThetaFromMatrix(blocks[bi].basis_columns), th);
                sum += dth;
                std::string w;
                for (size_t k = 0; k < x.w.size(); ++k) w += (k ? "," : "") + std::to_string(x.w[k]);
                std::printf("THETA_IMAGE n=%u ell=%u ideal=%u witness=%s dim_E=%u theta_image_dim=%u rank_theta_cell=%u\n", N, ell, x.id,
                            w.c_str(), x.dim, dth, r.rank_theta1);
                ++lines;
            }
            ct.check(sum == r.rank_theta1, tag + " sum_m d_theta(m) = rank theta^1 (" + std::to_string(sum) + " vs " + std::to_string(r.rank_theta1) + ")");
        }
    }
}

uint32_t Finish(const char* name, const Counter& ct, std::chrono::steady_clock::time_point t0) {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %llu checks, %llu failing (%.1f s)\n", name, static_cast<unsigned long long>(ct.checks),
                static_cast<unsigned long long>(ct.failures), s);
    return ct.failures == 0 ? 0u : 1u;
}

std::vector<Cell> StandardGrid() {
    std::vector<Cell> g;
    for (uint32_t ell : { 5u, 7u })
        for (uint32_t n = 0; n <= 40; n += 2) g.push_back({ n, ell });
    for (uint32_t ell : { 11u, 13u })
        for (uint32_t n = 0; n <= 26; n += 2) g.push_back({ n, ell });
    g.push_back({ 120, 5 });
    return g;
}

std::vector<Cell> SlowGrid() {
    std::vector<Cell> g;
    for (uint32_t ell : { 5u, 7u, 11u, 13u })
        for (uint32_t n = 42; n <= 200; n += 14) g.push_back({ n + (n % 2), ell });
    return g;
}

bool EnvOn(const char* v) {
    const char* e = std::getenv(v);
    return e && std::string(e) == "1";
}

} // namespace

uint32_t HeckeDicksonThetaHeckeValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Counter ct;
    RunGrid(StandardGrid(), 20, ct, EnvOn("HECKE_THETA_VERBOSE"));
    InstanceArm(ct);
    C7Arm(ct, false);
    const char* cells_file = std::getenv("HECKE_THETA_IMAGE_CELLS");
    if (cells_file != nullptr && std::string(cells_file).size() > 0) {
        std::vector<hecke_ext::CensusCell> list;
        std::string err;
        const bool ok = hecke_ext::ReadCensusCellList(cells_file, list, err);
        std::printf("  (P5) HECKE_THETA_IMAGE_CELLS=%s: %s\n", cells_file,
                    ok ? (std::to_string(list.size()) + " listed cell(s)").c_str() : ("REFUSED (" + err + ")").c_str());
        ct.check(ok, "(P5) the cell list parses (" + err + ")");
        if (EnvOn("HECKE_THETA_IMAGE_ALL"))
            std::printf("  (P5) HECKE_THETA_IMAGE_ALL=1 ignored: HECKE_THETA_IMAGE_CELLS takes precedence\n");
        uint32_t lines = 0;
        if (ok) {
            std::vector<std::pair<uint32_t, uint32_t>> cells;
            for (const auto& c : list) cells.push_back({ c.n, c.ell });
            CensusArm(ct, lines, &cells);
        }
        std::printf("  (P5) THETA_IMAGE lines: %u\n", lines);
        std::printf("EFFECTIVE HeckeDicksonThetaHeckeValidation arm=P5 cells=%zu\n", list.size());
    } else if (EnvOn("HECKE_THETA_IMAGE_ALL")) {
        uint32_t lines = 0;
        CensusArm(ct, lines);
        std::printf("  (P5) THETA_IMAGE lines: %u\n", lines);
        std::printf("EFFECTIVE HeckeDicksonThetaHeckeValidation arm=P5 cells=%u\n", lines);
    } else {
        std::printf("  (P5) the census column: [SKIP] (set HECKE_THETA_IMAGE_ALL=1; read-only, not an emitter)\n");
    }
    return Finish("HeckeDicksonThetaHeckeValidation", ct, t0);
}

uint32_t HeckeDicksonThetaHeckeCoverageSlowValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Counter ct;
    RunGrid(SlowGrid(), 0, ct, EnvOn("HECKE_THETA_VERBOSE"));
    C7Arm(ct, true);
    C9Arm(ct);
    CycleArm(ct);
    return Finish("HeckeDicksonThetaHeckeCoverageSlowValidation", ct, t0);
}
