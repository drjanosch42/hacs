// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZTorValidation -- slot-25 HZ-TOR (see the header).

#include "HeckeZTorValidation.h"

#include "HeckeZTor.h"
#include "Descent/DescentModuleExport.h"
#include "Descent/DescentModuleExportZ.h"
#include "Descent/DescentRecord.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/Rss.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using namespace heckez::tor;

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

double Since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string Cell(uint32_t N, uint32_t n, uint64_t ell) {
    return "(" + std::to_string(N) + "," + std::to_string(n) + ")@" + std::to_string(ell);
}

uint32_t g_cells = 0, g_exact_J = 0;

void Show(const TorRead& T) {
    std::cout << "  READ " << T.text() << " secs=" << T.secs << "\n";
    ++g_cells;
    g_exact_J += T.exact_J;
    for (const auto& b : T.blocks)
        if (b.J.rfind("exact", 0) == 0) ++g_exact_J;
}

uint32_t CountKind(const TorRead& T, const std::string& kind) {
    uint32_t c = 0;
    for (const auto& b : T.blocks) c += (b.kind == kind);
    return c;
}
const BlockRead* FirstCommon(const TorRead& T) {
    for (const auto& b : T.blocks)
        if (b.kind == "common") return &b;
    return nullptr;
}
bool AllJZero(const TorRead& T) {
    for (const auto& b : T.blocks)
        if (b.J != "zero") return false;
    return true;
}
bool SectionsChecked(const TorRead& T) {
    for (const auto& b : T.blocks) {
        if (b.split == "split(T')" && !b.section_checked) return false;
        if (b.split == "non_split" && !b.witness_checked) return false;
    }
    return true;
}
const Separation* SepAt(const TorRead& T, uint32_t p) {
    for (const auto& s : T.sep)
        if (s.p == p) return &s;
    return nullptr;
}

// ---- the Lean-kernel tables (K5) -------------------------------------------------------------------------------------------------
const char* kLeanRel[2] = {"docs/implementation/slot22/lv/lean_dims_l2.tsv", "docs/implementation/slot22/lv/lean_dims_l3.tsv"};

bool Exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}
std::string FindRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && Exists(r + "/" + kLeanRel[0])) return r;
    }
#endif
    char buf[4096];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (Exists(dir + "/" + kLeanRel[0])) return dir;
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}
// (ell, n) -> h0..h3 for rows with lean_dims = 1
std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> LoadLean(const std::string& root, Tally& T) {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> out;
    for (const char* rel : kLeanRel) {
        std::ifstream in(root + "/" + rel);
        T.check(in.good(), std::string("(C) open ") + rel);
        std::string line;
        std::vector<std::string> head;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> f;
            std::string cur;
            for (char c : line) {
                if (c == '\t') { f.push_back(cur); cur.clear(); } else cur += c;
            }
            f.push_back(cur);
            if (head.empty()) { head = f; continue; }
            auto col = [&](const std::string& name) -> std::string {
                for (size_t i = 0; i < head.size() && i < f.size(); ++i)
                    if (head[i] == name) return f[i];
                return "";
            };
            if (col("lean_dims") != "1") continue;
            const uint32_t ell = static_cast<uint32_t>(std::stoul(col("ell"))), n = static_cast<uint32_t>(std::stoul(col("n")));
            std::vector<uint32_t> h;
            for (const char* k : {"h0", "h1", "h2", "h3"}) h.push_back(static_cast<uint32_t>(std::stoul(col(k))));
            out[{ell, n}] = h;
        }
    }
    return out;
}

// ---- a toy extension ---------------------------------------------------------------------------------------------------------------
ExtensionInput Toy(uint64_t ell, uint32_t K, std::vector<uint32_t> sigma, uint32_t r, const std::vector<Mat>& A) {
    ExtensionInput X;
    X.N = 0;   // a toy (no cell)
    X.n = 0;
    X.ell = ell;
    X.K = K;
    X.sigma = std::move(sigma);
    X.r = r;
    for (size_t o = 0; o < A.size(); ++o) X.ps.push_back(static_cast<uint32_t>(101 + o));
    X.A = A;
    return X;
}

// ---- the cell list of (A), (B), (D), (E), (F): one unit per (cell, ell) --------------------------------------------------------
struct Unit {
    uint32_t N, n;
    uint64_t ell;
    std::vector<uint32_t> P;
};
std::vector<Unit> Units() {
    std::vector<Unit> u = {
        {1, 18, 5, {2, 3, 7}}, {1, 20, 5, {2, 3, 7}}, {1, 26, 5, {2, 3, 7, 11}}, {1, 30, 5, {2, 3, 7}}, {1, 8, 2, {2, 3, 5, 7}},
        {2, 6, 5, {3, 5, 7}},  {2, 6, 2, {3, 5, 7}},  {2, 6, 3, {3, 5, 7}},      {8, 10, 3, {3, 5, 7}}, {8, 10, 2, {3, 5, 7}}};
    for (uint32_t n : {8u, 10u, 12u, 16u, 18u, 22u, 26u, 30u})
        for (uint64_t ell : {2ull, 3ull})
            if (!(n == 8 && ell == 2)) u.push_back({1, n, ell, {2, 3, 5, 7}});
    return u;
}
std::vector<TorRead> RunUnits(const std::vector<Unit>& U, uint32_t threads) {
    std::vector<TorRead> out(U.size());
    std::vector<std::thread> pool;
    std::vector<std::string> err(U.size());
    for (uint32_t w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            for (size_t i = w; i < U.size(); i += threads) {
                try {
                    out[i] = ReadTor(U[i].N, U[i].n, U[i].ell, U[i].P);
                } catch (const std::exception& e) {
                    err[i] = e.what();
                }
            }
        });
    for (auto& t : pool) t.join();
    for (size_t i = 0; i < U.size(); ++i)
        if (!err[i].empty()) throw std::runtime_error("HeckeZTorValidation unit " + Cell(U[i].N, U[i].n, U[i].ell) + ": " + err[i]);
    return out;
}
uint64_t Digest(const std::vector<TorRead>& R) {
    uint64_t h = engine::kFnvOffset64;
    for (const auto& T : R) h = engine::FnvBytes(h, T.text() + "\n");
    return h;
}


// ---- (J) a change of presentation: g'_i = sum_k G_ki g_k + sum_j h0_ji tau_j (G unimodular on L, h0 in Hom(L, tors)) -------------
struct Lcg {
    uint64_t x;
    uint64_t next() { x = x * 6364136223846793005ull + 1442695040888963407ull; return x >> 17; }
};
ExtensionInput Represent(const ExtensionInput& X, uint64_t seed) {
    const Zlk R = Zlk::Make(X.ell, X.K);
    const uint32_t t = X.t(), r = X.r, D = X.D();
    Lcg g{seed};
    Mat G = heckez::local::MatIdentity(r), Gi = heckez::local::MatIdentity(r);
    for (uint32_t s = 0; s < 3 * r + 2 && r > 1; ++s) {
        const uint32_t i = static_cast<uint32_t>(g.next() % r), j = static_cast<uint32_t>(g.next() % r);
        if (i == j) {
            uint64_t u = g.next() % R.q;
            if (u % X.ell == 0) u += 1;
            const uint64_t ui = R.inv_unit(u);
            for (uint32_t k = 0; k < r; ++k) G[size_t(k) * r + i] = R.mul(G[size_t(k) * r + i], u);
            for (uint32_t k = 0; k < r; ++k) Gi[size_t(i) * r + k] = R.mul(Gi[size_t(i) * r + k], ui);
        } else {
            const uint64_t c = g.next() % R.q;
            for (uint32_t k = 0; k < r; ++k) G[size_t(k) * r + i] = R.add(G[size_t(k) * r + i], R.mul(c, G[size_t(k) * r + j]));
            for (uint32_t k = 0; k < r; ++k) Gi[size_t(j) * r + k] = R.sub(Gi[size_t(j) * r + k], R.mul(c, Gi[size_t(i) * r + k]));
        }
    }
    Mat h0(size_t(t) * r);
    for (uint32_t j = 0; j < t; ++j)
        for (uint32_t i = 0; i < r; ++i) h0[size_t(j) * r + i] = g.next() % R.pw(X.sigma[j]);
    // S = [[I, h0], [0, G]], S^{-1} = [[I, -h0 G^{-1}], [0, G^{-1}]]
    Mat S(size_t(D) * D, 0), Si(size_t(D) * D, 0);
    for (uint32_t i = 0; i < t; ++i) S[size_t(i) * D + i] = Si[size_t(i) * D + i] = 1;
    for (uint32_t j = 0; j < t; ++j)
        for (uint32_t i = 0; i < r; ++i) {
            S[size_t(j) * D + t + i] = h0[size_t(j) * r + i];
            uint64_t a = 0;
            for (uint32_t k = 0; k < r; ++k) a = R.add(a, R.mul(h0[size_t(j) * r + k], Gi[size_t(k) * r + i]));
            Si[size_t(j) * D + t + i] = R.neg(a);
        }
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t k = 0; k < r; ++k) {
            S[size_t(t + i) * D + t + k] = G[size_t(i) * r + k];
            Si[size_t(t + i) * D + t + k] = Gi[size_t(i) * r + k];
        }
    auto canon = [&](Mat A) {
        for (uint32_t i = 0; i < D; ++i) {
            const uint64_t m = i < t ? R.pw(X.sigma[i]) : 0;
            for (uint32_t j = 0; j < D; ++j) A[size_t(i) * D + j] = m ? A[size_t(i) * D + j] % m : A[size_t(i) * D + j] % R.q;
        }
        return A;
    };
    ExtensionInput Y = X;
    for (auto& A : Y.A) A = canon(heckez::local::MatMul(R, canon(heckez::local::MatMul(R, Si, A, D)), S, D));
    return Y;
}
std::string Verdict(const TorRead& T) {
    std::vector<std::string> b;
    for (const auto& x : T.blocks)
        b.push_back(x.kind + "/" + std::to_string(x.r_m) + "/" + std::to_string(x.t_m) + "/" + x.split + "/" + std::to_string(x.hom_length) +
                    "/" + x.J + "/" + x.coincidence);
    std::sort(b.begin(), b.end());
    std::string s = T.split + " jprime=" + std::to_string(T.jprime_length) + " A=" + std::to_string(T.algebra_length);
    for (const auto& x : b) s += " " + x;
    return s;
}
}  // namespace

uint32_t HeckeZTorValidation() {
    std::cout << "============ HeckeZTorValidation (slot-25 HZ-TOR) ============\n";
    const auto t_all = std::chrono::steady_clock::now();
    const uint64_t rss0 = oc_util::CurrentRssMiB();
    g_cells = 0;
    g_exact_J = 0;
    Tally T;

    const std::vector<Unit> U = Units();
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<TorRead> R1 = RunUnits(U, 1);
    const double secs1 = Since(t0);
    std::map<std::string, const TorRead*> by;
    for (size_t i = 0; i < U.size(); ++i) {
        by[Cell(U[i].N, U[i].n, U[i].ell)] = &R1[i];
        Show(R1[i]);
        T.check(R1[i].jprime_length == 0, "j' = 0 at " + Cell(U[i].N, U[i].n, U[i].ell) + " (audit M2)");
        T.check(R1[i].status == "ok", "status ok (a unit with torsion in the model; no_torsion is a loss) at " + Cell(U[i].N, U[i].n, U[i].ell) + ": " + R1[i].status);
        T.check(SectionsChecked(R1[i]), "every section / witness checked by substitution at " + Cell(U[i].N, U[i].n, U[i].ell));
    }

    // ---- (A) (1,18), (1,20) at 5 ----------------------------------------------------------------------------------------------
    for (uint32_t n : {18u, 20u}) {
        const TorRead& X = *by[Cell(1, n, 5)];
        const std::string tag = "(A) " + Cell(1, n, 5);
        T.check(X.t_ell == "added" && std::count(X.P.begin(), X.P.end(), 5u) == 1, tag + ": T_5 in the read set (" + X.t_ell + ")");
        T.check(X.sigma == std::vector<uint32_t>{1} && X.r == 3, tag + ": tors_5 = Z/5, rank L = 3");
        T.check(X.split == "split(separated)" && CountKind(X, "common") == 0 && AllJZero(X), tag + ": split by separation, J_5 zero");
        // 03 Sec. 3.1: the per-operator test is weaker than the joint one -- no single T_p separates here (every T_p alone agrees
        // with a free residue: the Eisenstein one at p != 5, the cusp one at T_5), the joint residues do
        bool none_alone = !X.sep.empty();
        for (const auto& s : X.sep) none_alone = none_alone && !s.separated;
        T.check(none_alone, tag + ": no single T_p separates (the joint test is needed, 03 Sec. 3.1)");
        T.check(X.jprime_length == 0, tag + ": J' = 0");
        // the control without T_5: the prime-to-5 algebra sees a common point (03 Sec. 6.4: Hom = 5)
        const auto E = integral_descent::modexport::ExportModule(1, n, 5, {2, 3, 7});
        if (T.check(E.ok, tag + ": the export without T_5")) {
            const TorRead Y = ReadExtension(FromModuleAtK(1, n, E.at(1)));
            Show(Y);
            const BlockRead* c = FirstCommon(Y);
            T.check(c && c->hom_length == 1 && c->split == "split(T')" && Y.split == "split(T')",
                    tag + ": without T_5 one common T'-block, |Hom_T'| = 5, split for T'");
        }
    }

    // ---- (B) (1,26), (1,30) at 5 ----------------------------------------------------------------------------------------------
    for (uint32_t n : {26u, 30u}) {
        const TorRead& X = *by[Cell(1, n, 5)];
        const std::string tag = "(B) " + Cell(1, n, 5);
        T.check(X.sigma == std::vector<uint32_t>{n == 26 ? 1u : 2u} && X.r == 5, tag + ": tors_5 and rank L = 5");
        const BlockRead* c = FirstCommon(X);
        T.check(CountKind(X, "common") == 1 && c != nullptr, tag + ": exactly one common block");
        if (c) {
            T.check(c->split == "split(T')" && c->section_checked, tag + ": split for T' (section checked)");
            T.check(c->hom_length == 2 && c->J == "bounds(<=5^2)", tag + ": J_5 bounds, |J_5| <= 25 (" + c->J + ")");
            T.check(c->coincidence == "not_certified(T'_block)", tag + ": coincidence not certified (split for T')");
            T.check(c->lean == "in_cap", tag + ": K7 cap");
        }
        T.check(X.split == "split(T')", tag + ": per-ell word split(T')");
        T.check(X.jprime_length == 0, tag + ": J_5 cap T' = 0 (j' = 1 at K_J = " + std::to_string(X.K_J) + ")");
    }

    // ---- (C) K5: B2(1) against the Lean-kernel tables ------------------------------------------------------------------------
    {
        t0 = std::chrono::steady_clock::now();
        const std::string root = FindRoot();
        uint32_t match = 0, mismatch = 0;
        if (T.check(!root.empty(), "(C) the Lean tables found (OC_SOURCE_DIR / cwd)")) {
            const auto lean = LoadLean(root, T);
            for (uint32_t n = 8; n <= 60; n += 2) {
                integral_descent::record::DescentRecordZ Rd;
                {
                    const hecke::ScopedCoefficientRing z(0);
                    integral_descent::record::RecordOptions o;
                    o.hecke = false;
                    o.pair = false;
                    Rd = integral_descent::record::BuildDescentRecord(1, n, o);
                }
                if (!T.check(Rd.status == "ok" && Rd.H.size() >= 4, "(C) the record's groups at (1," + std::to_string(n) + ")")) continue;
                for (uint32_t ell : {2u, 3u}) {
                    std::vector<uint32_t> b, t;
                    for (uint32_t q = 0; q < 4; ++q) {
                        b.push_back(Rd.H[q].free_rank);
                        const auto it = Rd.H[q].torsion.find(ell);
                        t.push_back(it == Rd.H[q].torsion.end() ? 0u : static_cast<uint32_t>(it->second.size()));
                    }
                    const auto h = UctPredict(b, t);
                    const auto it = lean.find({ell, n});
                    if (!T.check(it != lean.end(), "(C) a Lean row at ell = " + std::to_string(ell) + ", n = " + std::to_string(n))) {
                        ++mismatch;
                        continue;
                    }
                    if (h == it->second) ++match; else {
                        ++mismatch;
                        T.check(false, "(C) K5 at (1," + std::to_string(n) + ")@" + std::to_string(ell));
                    }
                }
            }
        }
        std::cout << "  K5 cells=27 primes=2 match=" << match << " mismatch=" << mismatch << " secs=" << Since(t0) << "\n";
        T.check(match == 54 && mismatch == 0, "(C) K5 54/54");
    }

    // ---- (D) K6 at (1,8), ell = 2 -------------------------------------------------------------------------------------------
    {
        const TorRead& X = *by[Cell(1, 8, 2)];
        const std::string tag = "(D) " + Cell(1, 8, 2);
        const Separation* s = SepAt(X, 2);
        const uint32_t t = static_cast<uint32_t>(X.sigma.size());
        std::vector<uint64_t> xt(t + 1, 0);
        xt[t] = 1;
        T.check(s && s->chi_L == std::vector<uint64_t>{1, 1} && s->chi_T == xt, tag + ": chi(T_2 | L) = x + 1, chi(T_2 | tors) = x^t");
        T.check(s && s->separated && s->bezout_checked, tag + ": separated at T_2, Bezout pair checked");
        T.check(X.split == "split(separated)" && AllJZero(X) && CountKind(X, "torsion_only") == 1, tag + ": torsion-only point, split, J zero");
    }

    // ---- (E) B8 / Cor. B8' -------------------------------------------------------------------------------------------------------
    uint32_t e_cells = 0;
    for (uint32_t n : {8u, 10u, 12u, 16u, 18u, 22u, 26u, 30u})
        for (uint64_t ell : {2ull, 3ull}) {
            const TorRead& X = *by[Cell(1, n, ell)];
            const std::string tag = "(E) " + Cell(1, n, ell);
            if (!T.check(X.status == "ok", tag + ": ell-torsion present (a silent skip here is a loss)")) continue;
            ++e_cells;
            T.check(X.b8_applies && X.b8_scalar, tag + ": T_p == 1 + p on H^1 / ell, p != ell (B8(1))");
            const bool Sk0 = (n == 8 || n == 12);
            T.check((CountKind(X, "torsion_only") == 1) == Sk0, tag + ": torsion-only point iff S_k = 0 (Cor. B8')");
            T.check((CountKind(X, "common") == 1) == !Sk0, tag + ": a common point iff S_k != 0");
            for (const auto& b : X.blocks)
                if (b.t_m > 0) {
                    // the T_ell residue of a torsion block is x (B8(3): T_ell nilpotent on tors)
                    size_t oi = 0;
                    while (oi < X.P.size() && X.P[oi] != ell) ++oi;
                    T.check(oi < X.P.size() && b.residue[oi] == std::vector<uint64_t>{0, 1}, tag + ": the torsion's T_ell residue is x");
                    if (b.kind == "common") T.check(b.coincidence == "certified(B8)", tag + ": coincidence certified(B8)");
                }
        }

    T.check(e_cells >= 16, "(E) counted " + std::to_string(e_cells) + " cells (floor 16)");

    // ---- (F) T4, T5 on the export ----------------------------------------------------------------------------------------------
    // (N, n, ell): r_m of the common block, |Hom| exponent, lean word
    struct FPin { uint32_t N, n; uint64_t ell; uint32_t rm, hom; const char* lean; };
    for (const FPin& c : {FPin{2, 6, 5, 2, 2, "in_cap"}, FPin{2, 6, 3, 2, 2, "in_cap"}, FPin{8, 10, 3, 12, 12, "absent(cap)"}}) {
        const TorRead& X = *by[Cell(c.N, c.n, c.ell)];
        const std::string tag = "(F) " + Cell(c.N, c.n, c.ell);
        T.check(X.status == "ok" && X.sigma == std::vector<uint32_t>{1}, tag + ": tors = Z/" + std::to_string(c.ell));
        const BlockRead* b = FirstCommon(X);
        T.check(CountKind(X, "common") == 1 && b, tag + ": one common T'-block");
        if (b) {
            T.check(b->split == "split(T')" && b->section_checked, tag + ": the split test (split for T')");
            T.check(b->r_m == c.rm && b->hom_length == c.hom &&
                        b->J == "bounds(<=" + std::to_string(c.ell) + "^" + std::to_string(c.hom) + ")" && b->lean == c.lean,
                    tag + ": r_m, |J| <= ell^hom, K7 word (" + b->J + ", " + b->lean + ")");
        }
        T.check(X.jprime_length == 0, tag + ": J cap T' = 0");
    }
    {
        // T4 at 2: one anemic point (T_2 not in P: U_2 is not exported), split for T'; T5 at 2: NON-SPLIT for T (B12(2)), so B14
        // certifies a common maximal ideal of T at 2; the verdict is the same at K = e, e + 1, e + 2 (B12(1))
        const TorRead& A2 = *by[Cell(2, 6, 2)];
        const BlockRead* a = FirstCommon(A2);
        T.check(A2.t_ell == "omitted(ell|N:U_ell_not_exported)" && A2.blocks.size() == 1 && a && a->r_m == 4 && a->split == "split(T')" &&
                    a->hom_length == 24,
                "(F) (2,6)@2: one anemic block, r = 4, split for T', |Hom| = 2^24");
        {
            // why |Hom| is everything there: T' acts on H^1 / 2^6 by the scalars 1 + p^7 (the eigenvalues 1 + p^7 and a_p of the
            // newform agree modulo 2^7, 2^9, 2^8 for p = 3, 5, 7 -- see (O5) -- and the torsion carries the same scalar)
            const auto E = integral_descent::modexport::ExportModule(2, 6, 2, {3, 5, 7});
            bool scal = E.ok && E.K_max >= 6;
            if (scal) {
                const auto M = E.at(6);
                const uint32_t D = M.rank();
                for (size_t o = 0; o < M.ps.size(); ++o) {
                    uint64_t lam = 1;
                    for (int i = 0; i < 7; ++i) lam *= M.ps[o];
                    lam += 1;
                    for (uint32_t i = 0; i < D; ++i) {
                        const uint64_t m = 1ull << M.exps[i];
                        for (uint32_t j = 0; j < D; ++j) scal = scal && M.ops[o][size_t(i) * D + j] == (i == j ? lam % m : 0);
                    }
                }
            }
            T.check(scal, "(F) (2,6)@2: every T_p (p = 3, 5, 7) is the scalar 1 + p^7 on H^1 / 2^6");
        }
        const TorRead& B2 = *by[Cell(8, 10, 2)];
        const BlockRead* b = FirstCommon(B2);
        T.check(b && b->hom_length == 227, "(F) (8,10)@2: |Hom_T'(L/2^3, tors)| = 2^227 (independent model, audit M2)");
        T.check(B2.split == "non_split" && b && b->witness_checked && b->coincidence == "certified(B14)" && b->r_m == 22 &&
                    B2.sigma == std::vector<uint32_t>({1, 1, 1, 2, 3, 3}),
                "(F) (8,10)@2: non-split for T (witness checked), B14 coincidence");
        const auto E = integral_descent::modexport::ExportModule(8, 10, 2, {3, 5, 7});
        if (T.check(E.ok && E.K_max >= 5, "(F) (8,10)@2: the export at K >= e + 2")) {
            for (uint32_t Kr : {4u, 5u}) {
                TorOptions o;
                o.K_read = Kr;
                o.jprime = false;
                const TorRead Y = ReadExtension(FromModuleAtK(8, 10, E.at(Kr)), o);
                T.check(Y.K == Kr && Y.split == "non_split" && SectionsChecked(Y), "(F) (8,10)@2 at K = " + std::to_string(Kr) + ": non-split (B12(1))");
            }
        }
        // the same consistency on a split(T') cell: (1,30)@5 (e = 2) at K = 3, 4
        const auto E5 = integral_descent::modexport::ExportModule(1, 30, 5, {2, 3, 7, 5});
        if (T.check(E5.ok && E5.K_max >= 4, "(F) (1,30)@5: the export at K >= e + 2")) {
            for (uint32_t Kr : {3u, 4u}) {
                TorOptions o;
                o.K_read = Kr;
                o.jprime = false;
                const TorRead Y = ReadExtension(FromModuleAtK(1, 30, E5.at(Kr)), o);
                const BlockRead* c = FirstCommon(Y);
                T.check(Y.split == "split(T')" && c && c->hom_length == 2, "(F) (1,30)@5 at K = " + std::to_string(Kr) + ": split for T', |Hom| = 25 (B12(1))");
            }
        }
    }

    // ---- (O5) the T4 (2,6) 2-adic at-root read, side read for the referee (Z0 referee O5; HZ-LOC Sec. 4) -------------------------
    // On L / 2^K, K = 20 (HZ-LOC read K = 9): chi(T_p | L) = (x - lambda_p)^2 (x - a_p)^2 with lambda_p = 1 + p^7 and a_p the
    // eigenvalues of the newform of S_8(Gamma_0(2)) (a_3 = 12, a_5 = -210, a_7 = 1016); the square-free annihilator c_p = x - a_p
    // of the cuspidal part; Smith valuations of c_p(T_p) on L / 2^20 against v_2 c_p(lambda_p).  a_i = v for both i means
    // c_p(T_p) L = 2^v L_E: the 2-part of eta_L is trivial, read by three operators independently of HZ-RAT's theta.
    {
        using namespace heckez::local;
        const uint32_t K = 20;
        const auto E = integral_descent::modexport::ExportModule(2, 6, 2, {3, 5, 7});
        if (T.check(E.ok && E.K_max >= K && E.free_rank == 4, "(O5) the export (2,6)@2 at K = 20")) {
            const auto M = E.at(K);
            const Zlk Rz = Zlk::Make(2, K);
            const uint32_t t = M.torsion_count, r = M.free_rank, D = M.rank();
            const int64_t ap[3] = {12, -210, 1016};
            std::ostringstream line;
            line << "  O5 (2,6)@2 K=" << K;
            for (size_t o = 0; o < 3; ++o) {
                const uint64_t p = M.ps[o];
                Mat B(size_t(r) * r);
                for (uint32_t i = 0; i < r; ++i)
                    for (uint32_t j = 0; j < r; ++j) B[size_t(i) * r + j] = M.ops[o][size_t(t + i) * D + t + j];
                uint64_t lam = 1;
                for (int i = 0; i < 7; ++i) lam *= p;
                lam += 1;
                const uint64_t a = static_cast<uint64_t>((ap[o] % int64_t(Rz.q) + int64_t(Rz.q)) % int64_t(Rz.q));
                // expected chi = ((x - lam)(x - a))^2
                const uint64_t s1 = Rz.add(lam % Rz.q, a), s0 = Rz.mul(lam % Rz.q, a);
                const Poly q2 = {s0, Rz.neg(s1), 1};
                Poly want(5, 0);
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j) want[i + j] = Rz.add(want[i + j], Rz.mul(q2[i], q2[j]));
                const Poly chi = Charpoly(Rz, B, r);
                T.check(chi == want, "(O5) chi(T_" + std::to_string(p) + " | L) = (x - lambda)^2 (x - a_p)^2 mod 2^20");
                const auto val = SmithValuations(Rz, MatMinusScalar(Rz, B, a, r), r, r);
                const uint32_t v = Rz.val(Rz.sub(lam % Rz.q, a));
                line << " T_" << p << ":v=" << v << ",a=[";
                for (size_t i = 0; i < val.size(); ++i) line << (i ? "," : "") << val[i];
                line << "]";
                T.check(val.size() == 2 && val[0] == v && val[1] == v, "(O5) T_" + std::to_string(p) + ": a = [v, v], the 2-part of eta_L trivial");
                const uint32_t want_v = o == 0 ? 7u : (o == 1 ? 9u : 8u);
                T.check(v == want_v, "(O5) T_" + std::to_string(p) + ": v_2 c(lambda) = " + std::to_string(want_v));
            }
            std::cout << line.str() << " eta_2=trivial\n";
        }
    }

    // ---- (G) toy extensions ------------------------------------------------------------------------------------------------------
    {
        // B12(3): H = Z_l^2 + F_l, x(a, b, c) = (0, l b, a mod l); generators [tau; g1; g2], column convention
        const uint64_t l = 5;
        const Mat Ax = {0, 1, 0, /**/ 0, 0, 0, /**/ 0, 0, l};
        const ExtensionInput X = Toy(l, 3, {1}, 2, {Ax});
        const TorRead Y = ReadExtension(X);
        Show(Y);
        const BlockRead* b = FirstCommon(Y);
        T.check(Y.split == "non_split" && b && b->witness_checked && b->coincidence == "certified(B14)",
                "(G) the B12(3) counterexample: non-split, witness checked, B14");
        T.check(b && b->J == "bounds(<=5^2)", "(G) B12(3): J = 0 there, the bound reads <= 25 (a bound, never exact)");
        // tamper: beta = 0 -> split
        Mat A0 = Ax;
        A0[1] = 0;
        const TorRead Y0 = ReadExtension(Toy(l, 3, {1}, 2, {A0}));
        Show(Y0);
        T.check(Y0.split == "split(T')", "(G) tamper beta = 0: split");
        // mixed exponents at 3: tors = Z/3 + Z/9, free rank 1; T g = tau_2, T tau = 0, B = 0: non-split
        const Mat Am = {0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 0};
        const TorRead Ym = ReadExtension(Toy(3, 2, {1, 2}, 1, {Am}));
        Show(Ym);
        T.check(Ym.split == "non_split" && Ym.e == 2 && SectionsChecked(Ym), "(G) mixed exponents Z/3 + Z/9: non-split, witness checked");
        // and with T g = 3 tau_2 = an ell-multiple: still non-split (beta in 3 tors != the image of h B - C h = 0)
        Mat Am3 = Am;
        Am3[5] = 3;
        const TorRead Ym3 = ReadExtension(Toy(3, 2, {1, 2}, 1, {Am3}));
        Show(Ym3);
        T.check(Ym3.split == "non_split" && SectionsChecked(Ym3), "(G) T g = 3 tau_2: non-split");
        // B with a unit eigenvalue gap: T g = g + tau_2, T tau = 0 (C = 0, B = 1): separated, split
        const Mat As = {0, 0, 0, /**/ 0, 0, 1, /**/ 0, 0, 1};
        const TorRead Ys = ReadExtension(Toy(3, 2, {1, 2}, 1, {As}));
        Show(Ys);
        T.check(Ys.split == "split(separated)" && AllJZero(Ys), "(G) separated toy: split(separated), J zero");
        const Separation* s = SepAt(Ys, 101);
        T.check(s && s->separated && s->bezout_checked, "(G) separated toy: Bezout pair checked");
        // the solver: a random-looking consistent system and a perturbed one, cross-checked by substitution
        {
            const Zlk Rz = Zlk::Make(5, 3);
            const std::vector<uint64_t> M = {5, 10, 0, 1, 25, 3, 0, 0, 50};
            const std::vector<uint64_t> x = {7, 3, 11};
            std::vector<uint64_t> bb(3, 0);
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j) bb[i] = (bb[i] + M[i * 3 + j] * x[j]) % 125;
            const LinSolve S = SolveLinear(Rz, M, 3, 3, bb);
            bool sub = S.solvable;
            for (size_t i = 0; i < 3 && sub; ++i) {
                uint64_t a = 0;
                for (size_t j = 0; j < 3; ++j) a = (a + M[i * 3 + j] * S.x[j]) % 125;
                sub = (a == bb[i]);
            }
            T.check(sub, "(G) SolveLinear: a consistent system solved (substitution)");
            std::vector<uint64_t> bad = bb;
            bad[2] = (bad[2] + 1) % 125;   // the third row is 50 x_3: an odd right-hand side is not in the image
            const LinSolve S2 = SolveLinear(Rz, M, 3, 3, bad);
            bool wit = !S2.solvable && S2.y.size() == 3;
            if (wit) {
                for (size_t j = 0; j < 3; ++j) {
                    uint64_t a = 0;
                    for (size_t i = 0; i < 3; ++i) a = (a + S2.y[i] * M[i * 3 + j]) % 125;
                    wit = wit && a == 0;
                }
                uint64_t a = 0;
                for (size_t i = 0; i < 3; ++i) a = (a + S2.y[i] * bad[i]) % 125;
                wit = wit && a != 0;
            }
            T.check(wit, "(G) SolveLinear: a perturbed system refused with a dual witness (y M = 0, y b != 0)");
        }
    }

    // ---- (H) the EXPORT-Z interface -------------------------------------------------------------------------------------------
    for (const auto& c : std::vector<std::pair<uint32_t, uint32_t>>{{2, 6}, {8, 10}}) {
        const std::vector<uint32_t> ps = {3, 5, 7};
        const auto Z = integral_descent::modexport::ExportModuleZ(c.first, c.second, ps);
        const std::string tag = "(H) (" + std::to_string(c.first) + "," + std::to_string(c.second) + ")";
        if (!T.check(Z.ok, tag + ": ExportModuleZ ok")) continue;
        bool all_eq = true, any_ns = false, any_ts = false;
        for (size_t l = 0; l < Z.local.size(); ++l) {
            if (Z.local[l].K == 0) continue;
            TorOptions o;
            const TorRead A = SplitFromModuleZ(Z, l, o);
            const auto E = integral_descent::modexport::ExportModule(c.first, c.second, Z.local[l].ell, ps);
            uint32_t e = 0;
            for (uint32_t s : E.sigma) e = std::max(e, s);
            const TorRead B = ReadExtension(FromModuleAtK(c.first, c.second, E.at(e)), o);
            std::cout << "  " << tag << " ell=" << Z.local[l].ell << " module_Z split=" << A.split << " digest=0x" << std::hex << A.digest()
                      << " export digest=0x" << B.digest() << std::dec << "\n";
            all_eq = all_eq && A.text() == B.text();
            any_ns = any_ns || A.split == "non_split";
            any_ts = any_ts || A.split == "split(T')";
        }
        T.check(all_eq, tag + ": SplitFromModuleZ == the export read at every torsion prime");
        const std::string w = ModuleZSplitWord(Z);
        std::cout << "  " << tag << " ModuleZ.split word = " << w << "\n";
        T.check(w == (any_ns ? "non_split" : (any_ts ? "split(T')" : "split(separated)")), tag + ": ModuleZSplitWord folds the per-ell verdicts");
    }

    // ---- (J) presentation invariance: the verdicts do not depend on the generators of L (G) nor on the lift (h0) ---------------
    {
        struct JC { uint32_t N, n; uint64_t ell; std::vector<uint32_t> P; };
        uint32_t pairs = 0, equal = 0;
        for (const JC& c : {JC{8, 10, 2, {3, 5, 7}}, JC{1, 30, 5, {2, 3, 7, 5}}, JC{1, 26, 2, {2, 3, 5, 7}}, JC{1, 18, 2, {2, 3, 5, 7}},
                            JC{8, 10, 3, {3, 5, 7}}}) {
            const auto E = integral_descent::modexport::ExportModule(c.N, c.n, static_cast<uint32_t>(c.ell), c.P);
            if (!T.check(E.ok, "(J) the export at " + Cell(c.N, c.n, c.ell))) continue;
            uint32_t e = 0;
            for (uint32_t x : E.sigma) e = std::max(e, x);
            const uint32_t KJ = std::min(E.K_max, e + 4);
            const ExtensionInput X = FromModuleAtK(c.N, c.n, E.at(KJ));
            TorOptions o;
            const TorRead A0 = ReadExtension(X, o);
            for (uint64_t seed : {1ull, 2ull, 3ull}) {
                const ExtensionInput Y = Represent(X, seed);
                T.check(Y.A != X.A, "(J) " + Cell(c.N, c.n, c.ell) + " seed " + std::to_string(seed) + ": the presentation moved (not vacuous)");
                const TorRead A1 = ReadExtension(Y, o);
                ++pairs;
                const bool same = Verdict(A0) == Verdict(A1) && SectionsChecked(A1);
                equal += same;
                T.check(same, "(J) " + Cell(c.N, c.n, c.ell) + " seed " + std::to_string(seed) + ": " + Verdict(A0) + " vs " + Verdict(A1));
            }
        }
        std::cout << "  PRESENTATION pairs=" << pairs << " equal=" << equal << "\n";
    }

    // ---- (I) schedule independence ------------------------------------------------------------------------------------------
    {
        t0 = std::chrono::steady_clock::now();
        const std::vector<TorRead> R4 = RunUnits(U, 4);
        const double secs4 = Since(t0);
        const uint64_t d1 = Digest(R1), d4 = Digest(R4);
        std::cout << "  THREADS units=" << U.size() << " digest1=0x" << std::hex << d1 << " digest4=0x" << d4 << std::dec
                  << " secs1=" << secs1 << " secs4=" << secs4 << "\n";
        T.check(d1 == d4, "(I) the unit list on 1 and 4 threads: digest equal");
    }

    const double secs = Since(t_all);
    const uint64_t rss1 = oc_util::CurrentRssMiB();
    std::cout << "  checks=" << T.checks << " secs=" << secs << " rss_mib=" << rss1 << " rss_growth_mib=" << (rss1 > rss0 ? rss1 - rss0 : 0) << "\n";
    const bool ok = T.failing == 0 && g_exact_J == 0;
    std::cout << "HECKEZ_TOR cells=" << g_cells << " mismatched=" << T.failing << " exact_J=" << g_exact_J
              << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
