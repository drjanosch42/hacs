// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZUpValidation -- slot-26 Z1 (UP-Z (a)); see the header and UPZ_design.md Sec. 5.

#include "HeckeZUpValidation.h"

#include "HeckeZLocal.h"
#include "HeckeZRecord.h"
#include "HeckeZUTag.h"
#include "HeckeZUp.h"
#include "Descent/DescentModuleExport.h"
#include "Descent/DescentModuleExportZ.h"
#include "Descent/IsoTreeCarrierZ.h"
#include "../HeckeExt/HeckeZRecordSource.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/FlatMatrix.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace mx = integral_descent::modexport;
namespace loc = heckez::local;
using heckez::record::HeckeStructureRecord;

std::string SourcePath(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

struct Tally {
    uint32_t checks = 0, failing = 0;
    bool check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [MISMATCH] " << what << "\n";
        }
        return ok;
    }
};

// HZ-DS edition 1: key -> record_fnv1a64 (docs/integral_census/results/hecke_ed1/rows.tsv)
std::map<std::string, std::string> Ed1Hashes() {
    std::map<std::string, std::string> m;
    std::ifstream in(SourcePath("docs/integral_census/results/hecke_ed1/rows.tsv"));
    std::string line;
    int col = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        if (col < 0) {
            for (size_t i = 0; i < f.size(); ++i)
                if (f[i] == "record_fnv1a64") col = static_cast<int>(i);
            continue;
        }
        if (col >= 0 && static_cast<size_t>(col) < f.size()) m[f[0]] = f[col];
    }
    return m;
}

// U_p on L (x) Z/ell^K (the export's free block at K = K_max), cokernel route, ExportOptions::allow_bad_primes
struct UpBlock {
    bool ok = false;
    std::string why;
    uint32_t K = 0, D = 0;
    loc::Mat T;
    std::vector<uint32_t> all;   // the whole module matrix (torsion + free), for the route comparisons
};
UpBlock ExportUp(uint32_t N, uint32_t n, uint32_t ell, uint32_t p) {
    UpBlock U;
    mx::ExportOptions eo;
    eo.allow_bad_primes = true;
    const mx::ModuleExport E = mx::ExportModule(N, n, ell, {p}, eo);
    if (!E.ok) {
        U.why = E.refusal;
        return U;
    }
    U.ok = true;
    U.K = E.K_max;
    const mx::ModuleAtK A = E.at(E.K_max);
    const auto F = A.free_block(0);
    U.D = F.r;
    U.T.assign(F.a.begin(), F.a.end());
    U.all = E.T[0].a;
    return U;
}
// chi (low -> high, canonical hex) mod ell^K
loc::Poly ChiMod(const loc::Zlk& R, const std::vector<std::string>& chi) {
    loc::Poly c;
    for (const auto& h : chi) c.push_back(loc::HexResidue(R, h));
    return c;
}
loc::Poly PolyMul(const loc::Zlk& R, const loc::Poly& a, const loc::Poly& b) {
    loc::Poly c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = R.add(c[i + j], R.mul(a[i], b[j]));
    return c;
}
uint64_t PowMod(uint64_t b, uint32_t e, const loc::Zlk& R) {
    uint64_t r = 1 % R.q;
    for (uint32_t i = 0; i < e; ++i) r = R.mul(r, b % R.q);
    return r;
}

struct Cell {
    uint32_t N, n;
};

}  // namespace

uint32_t HeckeZUpValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeZUpValidation: U_p, p | N, over Z_(ell), every ell (slot-26 Z1, UP-Z (a), ell >= 5 on the cokernel route; slot-26 Z2, "
                 "UP-Z (b), ell in {2, 3} on the tree route; [C, prop:heilbronn](2), `prop:strictliftN` (9), [C, def:heckeZ]; UPZ_design.md route (i))\n";
    heckez::up::PrintDescentEffectiveOnce();
    Tally T;
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    heckez::RationalSettings rs;
    rs.threads = 4;

    // ---- (R1) the default refuses p | N byte for byte; the option opens the cokernel route only ---------------------------------------
    {
        const mx::ModuleExport D0 = mx::ExportModule(11, 0, 5, {11});
        T.check(!D0.ok && D0.code == mx::ExportRefusal::PDividesN, "(R1) the default export refuses p | N (PDividesN)");
        mx::ExportOptions eo;
        eo.allow_bad_primes = true;
        const mx::ModuleExport D3 = mx::ExportModule(2, 6, 3, {2}, eo);
        T.check(D3.ok, "(R1) (Z2) the tree route (ell = 3) serves p | N with the option: " + D3.refusal);
        const mx::ModuleExport D3d = mx::ExportModule(2, 6, 3, {2});
        T.check(!D3d.ok && D3d.code == mx::ExportRefusal::PDividesN, "(R1) the default tree route still refuses p | N (PDividesN)");
    }

    // ---- (a) the export's U_p (Heilbronn on M_N / Rel_N over Z/ell^K) against HZ-UTAG's chi_Q(U_p) (word primes, level_n) -----------
    const std::vector<Cell> cells_a = {{11, 0}, {2, 6}, {8, 10}, {3, 4}, {5, 4}, {23, 0}};
    uint32_t umodl_ok = 0, umodl_all = 0, wp_ok = 0, wp_all = 0;
    bool t4 = false;
    std::map<std::pair<uint32_t, uint32_t>, heckez::UTagResult> utag;   // (N, p)
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, UpBlock> blocks;   // (N, ell, p)
    for (const Cell& c : cells_a) {
        for (uint32_t p : heckez::up::PrimesDividing(c.N)) {
            const heckez::UTagResult u = heckez::UTagStep(c.N, c.n, p, {}, rs);
            utag[{c.N, p}] = u;
            // (audit S6) the residues RationalStepFull reads (RunAtPrime on the operator list, here {U_p}) at one word prime q: chi of
            // that matrix = HZ-UTAG's chi_Q(U_p) mod q -- the direct tie of the full rational step's U_p to the oracle
            {
                const uint32_t q = heckez::WordPrimes(1, c.N)[0];
                const heckez::PrimeRun run = heckez::RunAtPrime(c.N, c.n, {p}, q);
                const std::vector<uint64_t> cw = heckez::CharpolyOfCombination(run, {1});
                const loc::Poly cu = ChiMod(loc::Zlk::Make(q, 1), u.chi);
                ++wp_all;
                const bool eq = T.check(cw == cu, "(a') (" + std::to_string(c.N) + "," + std::to_string(c.n) + ") U_" + std::to_string(p) +
                                                      " at the word prime " + std::to_string(q) + ": chi(RunAtPrime's U_p) = chi_Q(U_p) mod q");
                wp_ok += eq ? 1 : 0;
            }
            for (uint32_t ell : {2u, 3u, 5u, 7u}) {
                ++umodl_all;
                const UpBlock U = ExportUp(c.N, c.n, ell, p);
                const std::string tag = "(a) (" + std::to_string(c.N) + "," + std::to_string(c.n) + ") U_" + std::to_string(p) + " at ell = " +
                                        std::to_string(ell);
                if (!T.check(U.ok, tag + ": the export serves U_p (" + U.why + ")")) continue;
                const loc::Zlk R = loc::Zlk::Make(ell, U.K);
                const loc::Poly chiE = loc::Charpoly(R, U.T, U.D);
                const loc::Poly chiU = ChiMod(R, u.chi);
                const bool eq = T.check(U.D == u.b1 && chiE == chiU, tag + ": chi(U_p | L / ell^K) = chi_Q(U_p) mod ell^K (K = " +
                                                                        std::to_string(U.K) + ", D = " + std::to_string(U.D) + ")");
                umodl_ok += eq ? 1 : 0;
                if (c.N == 2 && c.n == 6 && eq && ell >= 5) t4 = true;
                std::cout << "  UMODL (" << c.N << "," << c.n << ") U_" << p << " ell=" << ell << " K=" << U.K << " D=" << U.D
                          << (eq ? " equal" : " DIFFER") << "\n";
                blocks[{c.N, ell, p}] = U;
            }
        }
    }

    // ---- (c) a_p^2 = p^{k-2} on the p-new part: N = p prime, S_k(SL_2(Z)) = 0, so H^1 = new cusp (+) Eisenstein (U_p in {1, p^{k-1}})
    uint32_t ap2_ok = 0, ap2_all = 0;
    for (const Cell& c : std::vector<Cell>{{11, 0}, {23, 0}, {5, 4}}) {
        const uint32_t p = c.N, k = c.n + 2;
        for (uint32_t ell : {5u, 7u}) {
            if (c.N % ell == 0) continue;
            const auto it = blocks.find({c.N, ell, p});
            if (it == blocks.end() || !it->second.ok) continue;
            ++ap2_all;
            const UpBlock& U = it->second;
            const loc::Zlk R = loc::Zlk::Make(ell, U.K);
            const loc::Poly eis = PolyMul(R, {R.neg(1 % R.q), 1}, {R.neg(PowMod(p, k - 1, R)), 1});
            const loc::Poly g = PolyMul(R, {R.neg(PowMod(p, k - 2, R)), 0, 1}, eis);
            const bool zero = loc::MatIsZero(loc::MatPolyEval(R, g, U.T, U.D));
            // over Z: chi_Q(U_p) (HZ-UTAG) splits completely, every root r in {1, p^{k-1}} or r^2 = p^{k-2}, and the roots with r^2 =
            // p^{k-2} number b1 - e_k = 2 dim S_k (every cusp form is p-new).  (Not "the Eisenstein factor alone does not kill U_p": at
            // (11, 0) a_11 = +1 is also the Eisenstein root.)
            const heckez::UTagResult& u = utag[{c.N, p}];
            int64_t pk1 = 1, pk2 = 1;
            for (uint32_t i = 0; i + 1 < k; ++i) pk1 *= p;
            for (uint32_t i = 0; i + 2 < k; ++i) pk2 *= p;
            uint32_t cusp_mult = 0;
            bool roots_ok = u.split_degree == u.b1 && u.m0 == 0;
            for (const auto& rt : u.roots) {
                if (rt.r * rt.r == pk2) cusp_mult += rt.mult;
                roots_ok = roots_ok && (rt.r == 1 || rt.r == pk1 || rt.r * rt.r == pk2);
            }
            // at k = 2 the root 1 is both the Eisenstein value and a possible cusp value: then only the total is checked
            const int64_t cusp_dim2 = static_cast<int64_t>(u.b1) - u.e_k;
            const bool count_ok = k == 2 ? static_cast<int64_t>(u.split_degree) == cusp_dim2 + u.e_k : cusp_mult == cusp_dim2;
            const bool ok = T.check(zero && roots_ok && count_ok, "(c) (" + std::to_string(c.N) + "," + std::to_string(c.n) + ") ell = " +
                                    std::to_string(ell) + ": (U_p^2 - p^{k-2})(U_p - 1)(U_p - p^{k-1}) = 0 on L / ell^K, and chi_Q(U_p) splits "
                                    "with every root in {1, p^{k-1}} or of square p^{k-2} (2 dim S_k of them)");
            std::cout << "  AP2 (" << c.N << "," << c.n << ") ell=" << ell << " zero=" << zero << " roots_ok=" << roots_ok << " count_ok=" << count_ok << " roots=";
            for (const auto& rt : u.roots) std::cout << rt.r << "^" << rt.mult << ";";
            std::cout << "\n";
            ap2_ok += ok ? 1 : 0;
        }
    }

    // ---- (d) route identity: threads 1 / 4 / 16 and every rho-fill ISA body (entrywise), the U_p word-prime routes, the record -------
    bool threads_eq = true, isa_eq = true;
    std::string isa_list;
    {
        const std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> rc = {{8, 10, 7, 2}, {11, 0, 5, 11}, {2, 6, 7, 2}};
        for (const auto& [N, n, ell, p] : rc) {
            std::vector<uint32_t> ref;
            for (uint32_t th : {1u, 4u, 16u}) {
                const mx::ScopedExportThreads st(th);
                const UpBlock U = ExportUp(N, n, ell, p);
                if (!U.ok) { threads_eq = false; continue; }
                if (th == 1) ref = U.all;
                else threads_eq = threads_eq && U.all == ref;
            }
            const struct { ::RhoFillPath path; const char* name; } paths[] = {
                {::RhoFillPath::Scalar, "scalar"}, {::RhoFillPath::Avx2, "avx2"}, {::RhoFillPath::Avx512, "avx512"}};
            for (const auto& ph : paths) {
                if (!::rho_fill_path_available(ph.path)) continue;
                if (N == 8 && isa_list.find(ph.name) == std::string::npos) isa_list += std::string(isa_list.empty() ? "" : ",") + ph.name;
                const ::ScopedRhoFillPath sp(ph.path);
                const UpBlock U = ExportUp(N, n, ell, p);
                isa_eq = isa_eq && U.ok && U.all == ref;
            }
        }
        T.check(threads_eq, "(d) the export's U_p at threads 1 / 4 / 16: entrywise equal");
        T.check(isa_eq, "(d) the export's U_p under every available rho-fill path (" + isa_list + "): entrywise equal");
        // HZ-UTAG's word-prime routes of U_p (term table vs count-once / dense-per-prime) entrywise
        for (const auto& [N, n, p] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{11, 0, 11}, {8, 10, 2}}) {
            const std::vector<uint32_t> ells = heckez::WordPrimes(3, N);
            heckez::UTagRoute used = heckez::UTagRoute::kAuto;
            const auto a = heckez::UTagRuns(N, n, p, ells, 4, heckez::UTagRoute::kAuto, &used);
            const auto b = heckez::UTagRuns(N, n, p, ells, 1, heckez::UTagRoute::kTermTable);
            bool eq = a.size() == b.size();
            for (size_t i = 0; eq && i < a.size(); ++i) eq = a[i].mats == b[i].mats && a[i].h1 == b[i].h1;
            T.check(eq, "(d) U_" + std::to_string(p) + " at (" + std::to_string(N) + "," + std::to_string(n) + "): word-prime route " +
                            heckez::UTagRouteName(used) + " (4 threads) = term_table (1 thread) entrywise");
            threads_eq = threads_eq && eq;
        }
    }

    // ---- (e) tampered controls: a perturbed U_p entry and a perturbed chi coefficient are caught by (a) ---------------------------------
    bool tamper = false;
    {
        const auto it = blocks.find({11, 7, 11});
        if (it != blocks.end() && it->second.ok) {
            UpBlock U = it->second;
            const loc::Zlk R = loc::Zlk::Make(7, U.K);
            const loc::Poly chiU = ChiMod(R, utag[{11, 11}].chi);
            U.T[0] = R.add(U.T[0], 1);
            const bool c1 = loc::Charpoly(R, U.T, U.D) != chiU;
            loc::Poly chi2 = chiU;
            chi2[0] = R.add(chi2[0], 1);
            const bool c2 = loc::Charpoly(R, it->second.T, U.D) != chi2;
            tamper = c1 && c2;
        }
        T.check(tamper, "(e) a perturbed U_p entry and a perturbed chi_Q(U_p) coefficient are both caught by the (a) comparison");
    }

    // ---- (g) the free-model U_p over Z (IsoHeckeN with allow_bad_primes, `prop:strictliftN` (9)) in EXPORT-Z's basis B against the
    //      Heilbronn U_p in the same B (ExportInBasis at ell = 5, 7, cokernel route): F mod ell^K == Heilbronn's matrix, entrywise; a
    //      perturbed F entry is caught.  Cells without 2- or 3-torsion (the tree route refuses p | N until Z2; that refusal is checked).
    uint32_t iso_ok = 0, iso_all = 0;
    bool iso_tamper = false, iso_default_refuses = false;
    {
        const integral_descent::hecke3::IsoHeckeN d0(11, 0, 11, integral_descent::hecke3::U32Ctx(1000003), 1);
        iso_default_refuses = d0.refused() && d0.code() == integral_descent::hecke3::D3Refusal::PDividesN;
        T.check(iso_default_refuses, "(g) IsoHeckeN refuses p | N by default (PDividesN)");
        for (const auto& [N, n, p] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{11, 0, 11}, {23, 0, 23}, {2, 6, 2}, {8, 10, 2}, {5, 4, 5}}) {
            mx::ModuleZ2Options zo;
            zo.allow_bad_primes = true;
            zo.threads = 4;
            const integral_descent::freez::IsoTreeCarrierZ car(N, n);
            const mx::FreeBlockRun run = mx::RunFreeBlock(car, {p}, zo);
            if (!T.check(run.refusal.empty(), "(g) (" + std::to_string(N) + "," + std::to_string(n) + ") the free block of U_" + std::to_string(p) +
                                                  " over Z is served (" + run.refusal + ")"))
                continue;
            // (the whole object [[tau, M], [0, F]] needs the exports at 2 and 3, refused for p | N until Z2: F is compared on its own)
            const uint32_t r = run.FB.B.cols;
            auto red = [](const nf::BigInt& x, uint64_t m) {   // x mod m in [0, m), from the decimal text
                const std::string s = x.to_string();
                const bool neg = !s.empty() && s[0] == '-';
                unsigned __int128 v = 0;
                for (size_t i = neg ? 1 : 0; i < s.size(); ++i) v = (v * 10 + static_cast<unsigned>(s[i] - '0')) % m;
                const uint64_t u = static_cast<uint64_t>(v);
                return neg && u ? m - u : u;
            };
            for (uint32_t ell : {2u, 3u, 5u, 7u}) {
                if (N % ell == 0 && ell >= 5) continue;
                ++iso_all;
                mx::ExportOptions eo;
                eo.allow_bad_primes = true;
                const mx::ModuleExport E = mx::ExportInBasis(N, n, ell, {p}, run.FB.B, eo);
                if (!T.check(E.ok && E.basis_ok, "(g) Heilbronn U_p in B at ell = " + std::to_string(ell) + " (" + E.refusal + E.basis_why + ")")) continue;
                std::vector<uint64_t> H(size_t(r) * r);
                for (uint32_t i = 0; i < r; ++i)
                    for (uint32_t j = 0; j < r; ++j) H[size_t(i) * r + j] = E.F_basis[0].at(i, j);
                uint64_t m = 1;
                for (uint32_t i = 0; i < E.K_max; ++i) m *= ell;
                std::vector<uint64_t> Fm(size_t(r) * r);
                for (size_t i = 0; i < Fm.size(); ++i) Fm[i] = red(run.G.F[0][i], m);
                const bool eq = T.check(Fm == H, "(g) (" + std::to_string(N) + "," + std::to_string(n) + ") ell = " + std::to_string(ell) +
                                                     ": F(U_p over Z, Iso) mod ell^" + std::to_string(E.K_max) + " = Heilbronn U_p in B");
                iso_ok += eq ? 1 : 0;
                std::cout << "  ISO (" << N << "," << n << ") U_" << p << " ell=" << ell << " K=" << E.K_max << " r=" << r << " proof="
                          << run.G.proof[0] << (eq ? " equal" : " DIFFER") << "\n";
                if (eq && !iso_tamper && !Fm.empty()) {
                    std::vector<uint64_t> Ft = Fm;
                    Ft[0] = (Ft[0] + 1) % m;
                    iso_tamper = Ft != H;
                }
            }
        }
        // (Z2) a cell with 2- and 3-torsion: the EXPORT-Z object of U_2, tau and M included, is now served ([[tau, M], [0, F]] by the tree exports)
        mx::ModuleZ2Options zo;
        zo.allow_bad_primes = true;
        zo.step1.export_opt.allow_bad_primes = true;
        const mx::ModuleZ Zt = mx::ExportModuleZ2(2, 6, {2}, zo);
        T.check(Zt.ok && Zt.free_block == "exact", "(g) (2,6): the EXPORT-Z object of U_2 is served (tree route, Z2): free_block=" + Zt.free_block + " " +
                                                      Zt.refusal.substr(0, 120));
        // the Iso-tree-carrier default keeps refusing without the option
        mx::ModuleZ2Options zd;
        const mx::ModuleZ Zd = mx::ExportModuleZ2(2, 6, {2}, zd);
        T.check(!Zd.ok, "(g) (2,6): without the option the object is still refused");
        T.check(iso_tamper, "(g) a perturbed F entry is caught by the Iso-vs-Heilbronn comparison");
    }

    // ---- (h) (Z2) the IsoHeckeN routes at p | N entrywise (materialised, streamed, count at n = 0), threads 1 / 4 / 16, on the tree export
    bool route_eq = true;
    uint32_t route_cells = 0;
    {
        namespace H3 = integral_descent::hecke3;
        for (const auto& [N, n, ell, p] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>{
                 {11, 0, 2, 11}, {23, 0, 3, 23}, {2, 6, 2, 2}, {8, 10, 3, 2}, {6, 0, 2, 3}}) {
            std::vector<uint32_t> ref;
            bool have_ref = false;   // (audit S4) an empty first matrix must not make the comparison vacuous
            std::vector<H3::IsoRoute> routes = {H3::IsoRoute::kMaterialised, H3::IsoRoute::kStreamed};
            if (n == 0) routes.push_back(H3::IsoRoute::kCountDense);
            for (H3::IsoRoute r : routes) {
                for (uint32_t th : {1u, 4u, 16u}) {
                    mx::ExportOptions eo;
                    eo.allow_bad_primes = true;
                    eo.iso_route = r;
                    const mx::ScopedExportThreads st(th);
                    const mx::ModuleExport E = mx::ExportModule(N, n, ell, {p}, eo);
                    if (!T.check(E.ok, "(h) (" + std::to_string(N) + "," + std::to_string(n) + ") ell=" + std::to_string(ell) + " route " +
                                           H3::IsoRouteName(r) + " serves U_p: " + E.refusal))
                        continue;
                    if (!have_ref) {
                        ref = E.T[0].a;
                        have_ref = true;
                    } else {
                        route_eq = route_eq && E.T[0].a == ref;
                    }
                }
            }
            T.check(have_ref && !ref.empty(), "(h) (" + std::to_string(N) + "," + std::to_string(n) + ") ell=" + std::to_string(ell) + ": the compared export matrix is non-empty");
            // the operator itself, entrywise, every route against the materialised table (the dense matrices on C^1_Iso)
            const H3::U32Ctx ctx = ell == 2 ? H3::U32Ctx::prime_power(2, 30) : H3::U32Ctx::prime_power(3, 19);
            const H3::IsoHeckeN Tm(N, n, p, ctx, 1, false, true, H3::IsoRoute::kMaterialised);
            const auto Dm = Tm.dense(1);
            for (H3::IsoRoute r : routes) {
                if (r == H3::IsoRoute::kMaterialised) continue;
                const H3::IsoHeckeN Tr(N, n, p, ctx, 1, false, true, r);
                const auto Dr = Tr.dense(1);
                bool eq = Dr.rows == Dm.rows && Dr.cols == Dm.cols;
                for (uint32_t i = 0; eq && i < Dm.rows; ++i)
                    for (uint32_t j = 0; eq && j < Dm.cols; ++j) eq = Dm.at(i, j) == Dr.at(i, j);
                route_eq = route_eq && eq;
                T.check(Tr.terms(1) == Tm.terms(1), "(h) the term counts agree between routes");
            }
            ++route_cells;
            std::cout << "  ISOROUTE (" << N << "," << n << ") ell=" << ell << " U_" << p << " terms=" << Tm.terms(1) << " equal=" << route_eq << "\n";
        }
        // the auto route of a small cell agrees with the forced ones (n = 0: count; n > 0: materialised), and the count route refuses n > 0 at the check
        T.check(route_eq, "(h) (Z2) materialised = streamed = count (n = 0), threads 1 / 4 / 16, entrywise");
    }

    // ---- (e') (Z2) a dropped coset of the free-model U_p (D3's validation-only fault) must not pass at ell in {2, 3}: refused by the read
    //      (not a cocycle / not an endomorphism) or caught by chi against HZ-UTAG
    bool drop_caught = true;
    {
        namespace H3 = integral_descent::hecke3;
        for (const auto& [N, n, ell, p] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>{{11, 0, 2, 11}, {23, 0, 3, 23}, {2, 6, 3, 2}}) {
            const H3::ScopedD3Fault f(H3::D3Fault::DropCoset);
            const UpBlock U = ExportUp(N, n, ell, p);
            // (audit S4) only the refusals the fault is meant to provoke count as caught: an RSS or lift refusal says nothing about the coset
            bool caught = !U.ok && (U.why.find("[NotEndomorphism]") != std::string::npos || U.why.find("[NotCocycle]") != std::string::npos);
            std::string how = U.ok ? "chi" : U.why.substr(0, 60);
            if (U.ok) {
                const loc::Zlk R = loc::Zlk::Make(ell, U.K);
                caught = loc::Charpoly(R, U.T, U.D) != ChiMod(R, utag[{N, p}].chi);
            }
            drop_caught = drop_caught && caught;
            T.check(caught, "(e') (" + std::to_string(N) + "," + std::to_string(n) + ") ell=" + std::to_string(ell) + ": a dropped coset of U_" + std::to_string(p) + " is caught");
            std::cout << "  DROPCOSET (" << N << "," << n << ") ell=" << ell << " caught_by=" << how << "\n";
        }
    }

    // ---- (N2, [C] 1.26.0-pre.3 confirm) every blocked( / refused( / open( word of every closed enumeration has a StatusTable class
    {
        uint32_t words = 0, unclassed = 0;
        for (const std::string& v : heckez::record::EnumVocabNames())
            for (const std::string& w : heckez::record::EnumValues(v)) {
                if (w.find("blocked(") == std::string::npos && w.find("refused(") == std::string::npos && w.find("open(") == std::string::npos)
                    continue;
                if (w.rfind("not_run(", 0) == 0 || w.rfind("omitted(", 0) == 0 || w.rfind("absent(", 0) == 0) continue;
                ++words;
                if (heckez::record::ClassOfStatus(w) == heckez::record::StatusClass::Unknown) {
                    ++unclassed;
                    std::cout << "  [VOCAB] " << v << ": '" << w << "' has no StatusTable() class\n";
                }
            }
        T.check(unclassed == 0 && words > 0, "(N2) every status-like word of the closed enumerations (" + std::to_string(words) +
                                                 ") has a StatusTable() class");
        std::cout << "  VOCAB status_like_words=" << words << " unclassed=" << unclassed << "\n";
    }

    // ---- (f) the recompute subset: 15 golden level-1 records + 3 anemic records hash-equal to HZ-DS ed. 1; the full records' scan ------
    const auto ed1 = Ed1Hashes();
    // (slot-27 EXW, design M3 likewise) the arm pins the ed. 1 hashes, i.e. that UP-Z moves no anemic / level-1 record: it runs with EXW's seam
    // empty, since (8, 10) anemic is one of the 22 pending records EXW moves (pinned by HeckeZRecordValidation and HeckeZExportWordValidation (E))
    heckez::record::RecordSources src = heckez::record::DefaultRecordSources();
    src.local_read_free = nullptr;
    heckez::record::RecordSettings S1;
    S1.threads = 4;
    uint32_t rec_ok = 0, rec_all = 0;
    bool fp_ok = true;
    auto one = [&](uint32_t N, uint32_t n, const char* alg) {
        ++rec_all;
        const HeckeStructureRecord R = heckez::record::BuildHeckeStructureRecord(N, n, alg, S1, src);
        fp_ok = fp_ok && R.settings_fingerprint == "0x8ffc29730c4f9672";
        const std::string key = "N" + std::to_string(N) + "_n" + std::to_string(n) + "_" + alg;
        const auto it = ed1.find(key);
        const bool eq = it != ed1.end() && it->second == R.record_fnv1a64;
        rec_ok += eq ? 1 : 0;
        T.check(eq, "(f) " + key + " record_fnv1a64 " + R.record_fnv1a64 + " = ed. 1 " + (it == ed1.end() ? "<none>" : it->second));
    };
    for (uint32_t n = 2; n <= 30; n += 2) one(1, n, "full");
    one(2, 6, "anemic");
    one(8, 10, "anemic");
    one(11, 0, "anemic");
    uint32_t full_scan_ok = 0, full_all = 0;
    std::string rec26;
    for (const Cell& c : std::vector<Cell>{{2, 6}, {8, 10}, {11, 0}, {23, 0}, {5, 4}}) {
        ++full_all;
        const HeckeStructureRecord R = heckez::record::BuildHeckeStructureRecord(c.N, c.n, "full", S1, src);
        const heckez::record::ScanResult sc = heckez::record::ScanRecord(R);
        for (const auto& pr : sc.problems) std::cout << "  [SCAN] (" << c.N << "," << c.n << ",full) " << pr << "\n";
        const heckez::record::Field* ops = R.find("G1.theta_ops");
        const heckez::record::Field* ctr = R.find("G10.contract");
        const bool ok = sc.values_outside_lists == 0 && sc.exact_under_bound == 0 && sc.overall_rule_ok && ops && ctr &&
                        R.settings_fingerprint == "0x8ffc29730c4f9672" && ctr->value == heckez::up::kFullContract;
        T.check(ok, "(f) (" + std::to_string(c.N) + "," + std::to_string(c.n) + ",full): scan clean, G1.theta_ops and G10.contract written");
        // (Z2, up::FullG3Rule; [C] 1.26.0-pre.9 item 8 (d')(ii)) G3.C_T.lower = not_run on every full record at N > 1, every per-tag and
        // aggregate status final; generation at ell in {2, 3} no longer pending (no blocked(U_p) outside G1.components / maximal_ideals: those
        // are HZ-EISUP's); the non-reduced branch at (8, 10): refused(non_reduced), theta / C2 not_run, no disc field
        auto val = [&](const char* p) { const heckez::record::Field* f = R.find(p); return f ? f->value : std::string("<absent>"); };
        bool no_blocked_outside_eisup = true;
        for (const auto& fd : R.fields)
            if (fd.value == "blocked(U_p)" && fd.path != "G1.components" && fd.path.find(".maximal_ideals") == std::string::npos) {
                no_blocked_outside_eisup = false;
                std::cout << "  [BLOCKED] " << fd.path << "\n";
            }
        T.check(no_blocked_outside_eisup && val("G3.C_T.lower") == "not_run", "(f) (Z2) no blocked(U_p) outside G1.components / maximal_ideals; G3.C_T.lower = not_run");
        const std::string ct = val("G3.C_T.status");
        T.check(ct == "not_run" || ct == "upper_bound_certified", "(f) (Z2) G3.C_T.status is final (" + ct + ")");
        if (c.N == 2)   // ell = 2, 3: the read ran on the tree route (no blocked word), generation certified or open(sandwich)
            T.check(val("G2.ell=2.generation.status") != "blocked(U_p)" && val("G2.ell=3.generation.status") != "blocked(U_p)", "(f) (2,6,full): generation at 2, 3 is read");
        if (c.N == 8) {
            T.check(val("G1.C1.status") == "refused(non_reduced)" && val("G1.theta") == "not_run" && val("G1.C2_status") == "not_run" &&
                        val("G1.components") == "not_run" && val("G1.disc.digest") == "<absent>" && val("G1.C1.deg_f") == "<absent>" &&
                        val("G3.C_T.status") == "not_run" && val("G2.ell=2.generation.status") == "open(sandwich)" && val("G2.ell=3.generation.status") == "open(sandwich)",
                    "(f) (8,10,full): the non-reduced branch's words (refused(non_reduced), not_run, no disc field, G3 not_run, open(sandwich) at 2, 3)");
            T.check(R.status == "final", "(f) (8,10,full): no pending field left, the record is final (status = " + R.status + ")");
        }
        full_scan_ok += ok ? 1 : 0;
        std::cout << "  RECORD " << heckez::record::StatusSignature(R) << "\n";
        std::cout << "  HASH " << R.id << " record_fnv1a64=" << R.record_fnv1a64 << " status=" << R.status << " theta_ops="
                  << (ops ? ops->value : "-") << " secs=" << R.secs << "\n";
        if (c.N == 2) rec26 = R.record_fnv1a64;
    }
    // (Z2 fix round, audit S1; [C] 1.26.0-pre.10 W-1) the G3 rule where the tag step did not run: blocked(U_p), never the false-final not_run;
    // no record reaches this branch as built, so the rule itself is tested
    {
        namespace up = heckez::up;
        const up::FullG3Words miss = up::FullG3Rule(false, false, false, false);
        T.check(miss.tag_status == "blocked(U_p)" && miss.ct_status == "blocked(U_p)" && miss.ct_lower == "blocked(U_p)", "(f) (W-1) tag step not run: all three words blocked(U_p)");
        const up::FullG3Words other = up::FullG3Rule(true, true, true, false);
        T.check(other.tag_status == "upper_bound_certified" && other.ct_status == "blocked(U_p)" && other.ct_lower == "blocked(U_p)", "(f) (W-1) a tag ran, another did not: its word stands, the aggregate words blocked(U_p)");
        const up::FullG3Words ran = up::FullG3Rule(false, false, true, true);
        T.check(ran.tag_status == "not_run" && ran.ct_status == "not_run" && ran.ct_lower == "not_run", "(f) (W-1) every tag ran, none bounds: the final not_run");
    }
    // the record itself at threads 1 / 16 (the full path end to end)
    for (uint32_t th : {1u, 16u}) {
        heckez::record::RecordSettings St = S1;
        St.threads = th;
        const HeckeStructureRecord R = heckez::record::BuildHeckeStructureRecord(2, 6, "full", St, src);
        const bool eq = R.record_fnv1a64 == rec26;
        threads_eq = threads_eq && eq;
        T.check(eq, "(d) (2,6,full) record at " + std::to_string(th) + " threads = at 4 threads");
    }
    T.check(fp_ok, "the settings fingerprint is 0x8ffc29730c4f9672 (route (i): no caps line, no schema step)");

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok = T.failing == 0;
    std::cout << "HECKEZ_UP cells=" << cells_a.size() << " mismatched=" << T.failing << " umodl=" << umodl_ok << "/" << umodl_all
              << " wordprime=" << wp_ok << "/" << wp_all << " t4_chiU2=" << (t4 ? "ok" : "FAIL") << " newform_ap2=" << ap2_ok << "/" << ap2_all
              << " threads=1,4,16:" << (threads_eq ? "equal" : "DIFFER") << " isa=" << isa_list << ":" << (isa_eq ? "equal" : "DIFFER")
              << " tamper=" << (tamper ? "caught" : "MISSED") << " recompute=" << rec_ok << "/" << rec_all << " full_scan=" << full_scan_ok
              << "/" << full_all << " iso=" << iso_ok << "/" << iso_all << " isoroute=" << route_cells << ":" << (route_eq ? "equal" : "DIFFER") << " dropcoset=" << (drop_caught ? "caught" : "MISSED") << " checks=" << T.checks << " secs=" << secs << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return T.failing;
}
