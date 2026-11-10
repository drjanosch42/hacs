// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZ package EXPORT-Z step 1 (slot 25): HeckeZExportZValidation.  See the header for the arms.
#include "HeckeZExportZValidation.h"

#include "DescentAssembly.h"
#include "DescentHecke.h"
#include "DescentLevelN.h"
#include "DescentModuleExport.h"
#include "DescentModuleExportZ.h"
#include "DescentRecord.h"

#include "../../Hecke/CoefficientRing.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNHeilbronn.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace mx = integral_descent::modexport;
namespace h3 = integral_descent::hecke3;
namespace rec = integral_descent::record;
using integral_descent::maps::Mat;
using integral_descent::maps::ZMod;
using clk = std::chrono::steady_clock;

uint32_t g_fail = 0;
uint64_t g_checks = 0, g_skips = 0;
// audit F3: a skipped block is counted and printed with its reason (the per-arm check floors catch a shrinking arm)
void Skip(const std::string& what, const std::string& why) {
    ++g_skips;
    std::cout << "  SKIP " << what << ": " << why << std::endl;
}
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string Join(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}
uint64_t Pow(uint64_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) x *= ell;
    return x;
}
uint64_t ModSub(uint64_t a, uint64_t b, uint64_t m) { return (a % m + m - b % m) % m; }

// ---- the cells -------------------------------------------------------------------------------------------------------
struct Cell {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    rec::DescentRecordZ R;
    mx::ModuleZ Z;
    std::string tag() const { return "(" + std::to_string(N) + "," + std::to_string(n) + ")"; }
};
std::vector<Cell>& Cells() {
    static std::vector<Cell> cells;
    if (!cells.empty()) return cells;
    const std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>> list = {
        {{1, 10}, {2, 3}}, {{2, 6}, {3, 5}}, {{8, 10}, {3, 5}}, {{9, 8}, {2, 5}}, {{11, 0}, {2, 3}}, {{11, 2}, {2, 3}},
        // audit F4: Z/25 (K = 2 on the cokernel route) and Z/29; (Z/5)^2 at ell = 5 dividing N (t = 2 on the cokernel route) and Z/11
        {{1, 30}, {2, 3}}, {{5, 12}, {2, 3}}};
    for (const auto& [Nn, ps] : list) {
        Cell c;
        c.N = Nn.first;
        c.n = Nn.second;
        c.ps = ps;
        {
            const hecke::ScopedCoefficientRing z(0);
            rec::RecordOptions o;
            o.ps = ps;
            c.R = rec::BuildDescentRecord(c.N, c.n, o);
        }
        c.Z = mx::ExportModuleZ(c.N, c.n, c.ps);
        std::cout << "  CELL " << c.tag() << " ps=" << Join(ps) << " status=" << c.Z.status() << " r=" << c.Z.free_rank << " d=[";
        for (size_t i = 0; i < c.Z.d.size(); ++i) std::cout << (i ? "," : "") << c.Z.d[i];
        std::cout << "] torsion_primes=" << Join(c.Z.torsion_primes) << " largest_torsion_prime=" << c.Z.largest_torsion_prime()
                  << " record=" << c.R.status << " H^1=" << (c.R.H.size() > 1 ? c.R.H[1].to_string() : "?") << " seconds=" << std::fixed
                  << std::setprecision(3) << c.Z.seconds << (c.Z.ok ? "" : " " + c.Z.refusal) << std::endl;
        cells.push_back(std::move(c));
    }
    return cells;
}

// ---- matrices with rows mod ell^{e_i} ----------------------------------------------------------------------------------
// (A B)_ij mod m_i over Z (entries < 2^31, at most a few hundred terms: no overflow with a reduction per term)
std::vector<uint64_t> MulRows(const std::vector<uint64_t>& A, const std::vector<uint64_t>& B, uint32_t k, const std::vector<uint64_t>& m) {
    std::vector<uint64_t> C(size_t(k) * k, 0);
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < k; ++j) {
            uint64_t s = 0;
            for (uint32_t l = 0; l < k; ++l) s = (s + (A[size_t(i) * k + l] % m[i]) * (B[size_t(l) * k + j] % m[i])) % m[i];
            C[size_t(i) * k + j] = s;
        }
    return C;
}
// the full ell-local block [[tau_ell, M_ell], [0, F_local]] of op o (rows mod ell^{exps_i}) from the ModuleZ's public pieces
struct LocalBlock {
    uint32_t t = 0, r = 0, k = 0;
    std::vector<uint64_t> A, mod;
};
LocalBlock Full(const mx::ModuleZ& Z, size_t o, size_t l) {
    const mx::LocalPartZ& P = Z.local[l];
    LocalBlock B;
    B.t = P.t;
    B.r = Z.free_rank;
    B.k = B.t + B.r;
    B.A.assign(size_t(B.k) * B.k, 0);
    const uint32_t T = Z.torsion_rank();
    for (uint32_t i = 0; i < B.t; ++i)
        for (uint32_t j = 0; j < B.t; ++j) B.A[size_t(i) * B.k + j] = Z.tau[o][size_t(P.t0 + i) * T + P.t0 + j];
    const auto M = Z.M_block(o, l), F = Z.F_local(o, l);
    for (uint32_t i = 0; i < B.t; ++i)
        for (uint32_t j = 0; j < B.r; ++j) B.A[size_t(i) * B.k + B.t + j] = M[size_t(i) * B.r + j];
    for (uint32_t i = 0; i < B.r; ++i)
        for (uint32_t j = 0; j < B.r; ++j) B.A[size_t(B.t + i) * B.k + B.t + j] = F[size_t(i) * B.r + j];
    for (uint32_t i = 0; i < B.t; ++i) B.mod.push_back(Z.d[P.t0 + i]);
    for (uint32_t i = 0; i < B.r; ++i) B.mod.push_back(Pow(P.ell, P.K));
    return B;
}

// ---- the cochain model (the operators re-built from the shipped constructors, as ExportModule builds them) ---------------------
struct Model {
    ZMod Z;                                   // the export's width K_run
    Mat d0;                                   // C^1 <- C^0 (tree) | the presentation M_N <- generators (cokernel)
    std::vector<h3::BatchOp> ops;
    std::unique_ptr<h3::TreeTransport> X;
    std::vector<std::unique_ptr<h3::IsoHeckeN>> iso;
    std::unique_ptr<hecke::level::Gamma0LevelN> lvl;
    std::vector<std::unique_ptr<hecke::level::HeilbronnOperator>> H;
    std::unique_ptr<engine::ModCtx<uint32_t>> ctx;
};
std::unique_ptr<Model> BuildModel(const mx::ModuleExport& E) {
    auto M = std::make_unique<Model>();
    if (E.ell <= 3) {
        M->X = std::make_unique<h3::TreeTransport>(E.ell, E.N, E.n, 1, E.K_run);
        M->Z = M->X->Z();
        M->d0 = M->X->differential(0);
        for (uint32_t p : E.ps) {
            M->iso.push_back(std::make_unique<h3::IsoHeckeN>(E.N, E.n, p, M->X->ctx(), 1));
            ENGINE_CHECK_MSG(!M->iso.back()->refused(), "oracle: IsoHeckeN refused where the export did not");
            const h3::IsoHeckeN* tp = M->iso.back().get();
            const h3::TreeTransport* X = M->X.get();
            M->ops.push_back([X, tp](const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& Y) { X->apply_hecke(*tp, 1, U, k, Y); });
        }
    } else {
        M->Z = ZMod(E.ell, E.K_run);
        M->ctx = std::make_unique<engine::ModCtx<uint32_t>>(E.K_run == 1 ? engine::ModCtx<uint32_t>(E.ell)
                                                                          : engine::ModCtx<uint32_t>::prime_power(E.ell, E.K_run));
        const uint32_t D = integral_descent::LevelNClosedForm(E.N, E.n).D;
        ENGINE_CHECK_MSG(E.n % 2 == 0, "oracle: n even on the cokernel route");
        const engine::DenseBuf<uint32_t> G = integral_descent::FixedGeneratorRows(E.N, E.n, *M->ctx);
        M->d0 = Mat(D, G.rows);
        for (uint32_t i = 0; i < G.rows; ++i)
            for (uint32_t j = 0; j < D; ++j) M->d0.at(j, i) = G.at(i, j);
        M->lvl = std::make_unique<hecke::level::Gamma0LevelN>(E.N);
        for (uint32_t p : E.ps) {
            M->H.push_back(std::make_unique<hecke::level::HeilbronnOperator>(*M->lvl, p, E.n, *M->ctx));
            const hecke::level::HeilbronnOperator* hp = M->H.back().get();
            M->ops.push_back([hp, D](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
                Y.assign(size_t(D) * k, 0);
                std::vector<uint32_t> col(D);
                for (uint32_t t = 0; t < k; ++t) {
                    for (uint32_t r = 0; r < D; ++r) col[r] = X[size_t(r) * k + t];
                    const std::vector<uint32_t> y = hp->apply(col);
                    for (uint32_t r = 0; r < D; ++r) Y[size_t(r) * k + t] = y[r];
                }
            });
        }
    }
    ENGINE_CHECK_MSG(M->d0.r == E.cochain_dim, "oracle: the model's cochain rank differs from the export's");
    return M;
}

// Is every column of W (c x k, entries mod ell^K) in im(d0) mod ell^K?  A fresh Smith form of [d0 | W] over Z/ell^K: im(d0) is
// contained in im([d0 | W]), with equality iff the two have the same Smith levels (the images are finite of equal length).
bool InImage(const Mat& d0, const std::vector<uint32_t>& W, uint32_t k, uint32_t ell, uint32_t K) {
    const ZMod ZK(ell, K);
    Mat A(d0.r, d0.c), B(d0.r, d0.c + k);
    for (uint32_t i = 0; i < d0.r; ++i) {
        for (uint32_t j = 0; j < d0.c; ++j) A.at(i, j) = B.at(i, j) = ZK.red(d0.at(i, j));
        for (uint32_t j = 0; j < k; ++j) B.at(i, d0.c + j) = ZK.red(W[size_t(i) * k + j]);
    }
    std::vector<uint32_t> sa = integral_descent::maps::SmithForm(ZK, A, false, false).sigma;
    std::vector<uint32_t> sb = integral_descent::maps::SmithForm(ZK, B, false, false).sigma;
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    return sa == sb;
}

// (C) / (E): does the matrix A' (rows mod `mod`) present op o on the generators X' (the export's cocycles, free columns moved by h)?
// Residual W = T X' - X' A' over Z/ell^K, then InImage.
bool Presents(const Model& Mo, size_t o, const std::vector<uint32_t>& Xp, uint32_t c, const LocalBlock& B, uint32_t ell, uint32_t K) {
    std::vector<uint32_t> Y;
    Mo.ops[o](Xp, B.k, Y);
    ENGINE_CHECK(Y.size() == size_t(c) * B.k);
    const uint64_t QK = Pow(ell, K);
    std::vector<uint32_t> W(size_t(c) * B.k, 0);
    for (uint32_t r = 0; r < c; ++r)
        for (uint32_t j = 0; j < B.k; ++j) {
            uint64_t s = Mo.Z.red(Y[size_t(r) * B.k + j]) % QK;
            for (uint32_t i = 0; i < B.k; ++i) {
                const uint64_t a = B.A[size_t(i) * B.k + j];
                if (a) s = ModSub(s, (a % QK) * (Xp[size_t(r) * B.k + i] % QK) % QK, QK);
            }
            W[size_t(r) * B.k + j] = static_cast<uint32_t>(s);
        }
    return InImage(Mo.d0, W, B.k, ell, K);
}

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
uint64_t Rand() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}

// ---- (A) algebra -----------------------------------------------------------------------------------------------------
void ArmA() {
    for (const Cell& c : Cells()) {
        if (!check(c.Z.ok, "(A) " + c.tag() + " ok: " + c.Z.refusal)) continue;
        const mx::ModuleZ& Z = c.Z;
        const uint32_t T = Z.torsion_rank();
        for (size_t o = 0; o < Z.ps.size(); ++o) {
            const std::string tp = "(A) " + c.tag() + " T_" + std::to_string(Z.ps[o]);
            bool endo = true, diag = true;
            for (uint32_t i = 0; i < T; ++i)
                for (uint32_t j = 0; j < T; ++j) {
                    const uint64_t v = Z.tau[o][size_t(i) * T + j];
                    if (Z.tor_ell[i] != Z.tor_ell[j]) diag = diag && v == 0;
                    else endo = endo && v < Z.d[i] && (v * (Z.d[j] % Z.d[i])) % Z.d[i] == 0;
                }
            check(endo, tp + ": tau carries the relations d_j g_j into the relations (tau_ij d_j = 0 mod d_i)");
            check(diag, tp + ": tau is block-diagonal over ell");
            for (size_t q = o + 1; q < Z.ps.size(); ++q) {
                const std::string pq = tp + " T_" + std::to_string(Z.ps[q]);
                check(MulRows(Z.tau[o], Z.tau[q], T, Z.d) == MulRows(Z.tau[q], Z.tau[o], T, Z.d), pq + ": tau_p tau_q = tau_q tau_p");
                for (size_t l = 0; l < Z.local.size(); ++l) {
                    if (Z.local[l].t == 0) continue;
                    const LocalBlock A = Full(Z, o, l), B = Full(Z, q, l);
                    check(MulRows(A.A, B.A, A.k, A.mod) == MulRows(B.A, A.A, A.k, A.mod),
                          pq + " ell=" + std::to_string(Z.local[l].ell) + ": the full local block [[tau, M], [0, F_local]] commutes");
                }
            }
        }
        // the retraction against a re-built d^0
        for (const mx::LocalPartZ& P : Z.local) {
            if (P.t == 0) continue;
            const mx::ModuleExport& E = *P.E;
            const std::string tl = "(A) " + c.tag() + " ell=" + std::to_string(P.ell);
            check(mx::RetractionFnv(E) == P.retraction_fnv, tl + ": the retraction FNV is reproducible");
            const auto Mo = BuildModel(E);
            const uint32_t cdim = E.cochain_dim, k = E.rank();
            bool kills_b = true, unit = true;
            for (uint32_t a = 0; a < P.t; ++a) {
                const uint64_t m = Z.d[P.t0 + a];
                for (uint32_t col = 0; col < Mo->d0.c && kills_b; ++col) {
                    uint64_t s = 0;
                    for (uint32_t r = 0; r < cdim; ++r) s = (s + uint64_t(E.retraction.at(a, r)) * (Mo->d0.at(r, col) % m)) % m;
                    kills_b = s == 0;
                }
                for (uint32_t j = 0; j < k; ++j) {
                    uint64_t s = 0;
                    for (uint32_t r = 0; r < cdim; ++r) s = (s + uint64_t(E.retraction.at(a, r)) * (E.cocycles.at(r, j) % m)) % m;
                    unit = unit && s == (j == a ? 1u % m : 0u);
                }
            }
            check(kills_b, tl + ": pi_ell d^0 = 0 (the retraction vanishes on B^1)");
            check(unit, tl + ": pi_ell(g_b) = delta_ab on the torsion generators and 0 on the local free generators");
        }
    }
}

// ---- (B) the record's basis-free invariants ------------------------------------------------------------------------------
// a decimal or "0x.." / "-0x.." integer string reduced mod Q (false: "?" or another form)
bool ReduceString(const std::string& s, uint64_t Q, uint64_t& out) {
    if (s.empty() || s == "?") return false;
    const bool neg = s[0] == '-';
    size_t i = neg ? 1 : 0;
    uint64_t base = 10, v = 0;
    if (s.compare(i, 2, "0x") == 0) {
        base = 16;
        i += 2;
    }
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        const char ch = s[i];
        uint64_t dg;
        if (ch >= '0' && ch <= '9') dg = uint64_t(ch - '0');
        else if (base == 16 && ch >= 'a' && ch <= 'f') dg = uint64_t(ch - 'a' + 10);
        else return false;
        v = (v * base + dg) % Q;
    }
    out = neg ? (Q - v) % Q : v;
    return true;
}
void ArmB() {
    for (const Cell& c : Cells()) {
        if (!c.Z.ok) {
            Skip("(B) " + c.tag(), "module_Z blocked (arm (A) fails it): " + c.Z.refusal);
            continue;
        }
        if (!check(c.R.ok() && c.R.H.size() > 1, "(B) " + c.tag() + " record ok")) continue;
        const auto& H1 = c.R.H[1];
        check(H1.free_rank == c.Z.free_rank, "(B) " + c.tag() + ": r = " + std::to_string(c.Z.free_rank) + " = the record's rank H^1 " +
                                                 std::to_string(H1.free_rank));
        std::map<uint32_t, std::vector<uint32_t>> mine;
        for (uint32_t i = 0; i < c.Z.torsion_rank(); ++i) mine[c.Z.tor_ell[i]].push_back(c.Z.tor_sigma[i]);
        std::map<uint32_t, std::vector<uint32_t>> theirs;
        for (const auto& [p, e] : H1.torsion) theirs[static_cast<uint32_t>(p)] = std::vector<uint32_t>(e.begin(), e.end());
        for (auto* m : {&mine, &theirs})
            for (auto& [p, e] : *m) std::sort(e.begin(), e.end());
        check(mine == theirs, "(B) " + c.tag() + ": the d_i are the torsion of the record's H^1 = " + H1.to_string());
        for (size_t l = 0; l < c.Z.local.size(); ++l) {
            const mx::LocalPartZ& P = c.Z.local[l];
            if (P.t == 0) continue;
            for (size_t o = 0; o < c.Z.ps.size(); ++o) {
                const rec::CharpolyEntry* ce = nullptr;
                for (const auto& e : c.R.charpoly)
                    if (e.p == c.Z.ps[o]) ce = &e;
                if (!ce || ce->refused || !ce->determined) {
                    Skip("(B) " + c.tag() + " ell=" + std::to_string(P.ell) + " T_" + std::to_string(c.Z.ps[o]),
                         !ce ? "the record has no char_Z entry" : ce->refused ? "the record's char_Z is refused" : "the record's char_Z is undetermined");
                    continue;
                }
                const uint64_t Q = Pow(P.ell, P.K);
                const std::vector<uint32_t> loc = P.at.free_charpoly(o);   // high -> low mod ell^K
                std::vector<uint64_t> rz;
                bool parsed = true;
                for (size_t i = 0; i < ce->coefficients.size(); ++i) {   // the record's coefficients are high -> low
                    uint64_t v = 0;
                    const std::string& s = ce->coefficients[i];
                    parsed = parsed && (ReduceString(s, Q, v) || (i < ce->exact.size() && ReduceString(ce->exact[i], Q, v)));
                    rz.push_back(v);
                }
                std::vector<uint64_t> l64(loc.begin(), loc.end());
                check(parsed && l64 == rz, "(B) " + c.tag() + " ell=" + std::to_string(P.ell) + " T_" + std::to_string(c.Z.ps[o]) +
                                               ": char(F_local) = char_Z(T_p | H^1/tors) mod ell^" + std::to_string(P.K));
            }
        }
    }
}

// ---- (C) / (E) the cochain-level presentation and splitting invariance -------------------------------------------------------
void ArmCE(bool move) {
    const char* arm = move ? "(E)" : "(C)";
    for (const Cell& c : Cells()) {
        if (!c.Z.ok) {
            Skip(std::string(arm) + " " + c.tag(), "module_Z blocked (arm (A) fails it): " + c.Z.refusal);
            continue;
        }
        for (size_t l = 0; l < c.Z.local.size(); ++l) {
            const mx::LocalPartZ& P = c.Z.local[l];
            if (P.t == 0) continue;
            const mx::ModuleExport& E = *P.E;
            const auto Mo = BuildModel(E);
            const uint32_t cdim = E.cochain_dim, t = P.t, r = c.Z.free_rank, k = t + r;
            const std::string tl = std::string(arm) + " " + c.tag() + " ell=" + std::to_string(P.ell) + " K=" + std::to_string(P.K);
            for (uint32_t seed = 0; seed < (move ? 2u : 1u); ++seed) {
                // h in Hom(L, T_ell): t x r, h_ij in [0, d_i)
                std::vector<uint64_t> h(size_t(t) * r, 0);
                if (move)
                    for (uint32_t i = 0; i < t; ++i)
                        for (uint32_t j = 0; j < r; ++j) h[size_t(i) * r + j] = Rand() % c.Z.d[P.t0 + i];
                std::vector<uint32_t> Xp(size_t(cdim) * k);
                for (uint32_t row = 0; row < cdim; ++row) {
                    for (uint32_t j = 0; j < k; ++j) Xp[size_t(row) * k + j] = E.cocycles.at(row, j);
                    for (uint32_t j = 0; j < r; ++j) {
                        uint64_t s = Xp[size_t(row) * k + t + j];
                        for (uint32_t i = 0; i < t; ++i) s += h[size_t(i) * r + j] * E.cocycles.at(row, i) % Mo->Z.m;
                        Xp[size_t(row) * k + t + j] = Mo->Z.red(s % Mo->Z.m);
                    }
                }
                for (size_t o = 0; o < c.Z.ps.size(); ++o) {
                    const std::string to = tl + " T_" + std::to_string(c.Z.ps[o]) + (move ? " seed=" + std::to_string(seed) : "");
                    LocalBlock B = Full(c.Z, o, l);
                    const LocalBlock B0 = B;
                    bool moved_differs = false;
                    if (move) {
                        // M' = M + tau h - h F_local (row i mod d_i)
                        for (uint32_t i = 0; i < t; ++i) {
                            const uint64_t m = B.mod[i];
                            for (uint32_t j = 0; j < r; ++j) {
                                uint64_t s = B.A[size_t(i) * k + t + j] % m;
                                for (uint32_t a = 0; a < t; ++a) s = (s + (B.A[size_t(i) * k + a] % m) * (h[size_t(a) * r + j] % m)) % m;
                                for (uint32_t b = 0; b < r; ++b) s = ModSub(s, (h[size_t(i) * r + b] % m) * (B.A[size_t(t + b) * k + t + j] % m) % m, m);
                                if (s != B.A[size_t(i) * k + t + j]) moved_differs = true;
                                B.A[size_t(i) * k + t + j] = s;
                            }
                        }
                    }
                    check(Presents(*Mo, o, Xp, cdim, B, P.ell, P.K), to + ": [[tau, M" + (move ? "+tau h-h F" : "") +
                                                                       "], [0, F_local]] presents T_p on the generators in H^1(C / ell^K)");
                    // teeth: a perturbed entry must fail
                    LocalBlock Bad = B;
                    if (r > 0) Bad.A[size_t(0) * k + t] = (Bad.A[size_t(0) * k + t] + 1) % Bad.mod[0];
                    else Bad.A[0] = (Bad.A[0] + 1) % Bad.mod[0];
                    check(!Presents(*Mo, o, Xp, cdim, Bad, P.ell, P.K), to + ": a perturbed " + (r > 0 ? "M" : "tau") + " entry is rejected");
                    if (move && moved_differs)
                        check(!Presents(*Mo, o, Xp, cdim, B0, P.ell, P.K), to + ": the unmoved M is rejected in the moved generators");
                }
            }
        }
    }
}
void ArmC() { ArmCE(false); }
void ArmE() { ArmCE(true); }

// ---- (D) the record's torsion canon ---------------------------------------------------------------------------------------
void ArmD() {
    for (const Cell& c : Cells()) {
        if (!c.Z.ok) {
            Skip("(D) " + c.tag(), "module_Z blocked (arm (A) fails it): " + c.Z.refusal);
            continue;
        }
        const uint32_t T = c.Z.torsion_rank();
        for (const mx::LocalPartZ& P : c.Z.local) {
            const rec::TorsionModule* tm = nullptr;
            for (const auto& m : c.R.torsion)
                if (m.q == 1 && m.ell == P.ell) tm = &m;
            const std::string tl = "(D) " + c.tag() + " ell=" + std::to_string(P.ell);
            if (P.t == 0) {
                check(!tm || tm->exponents.empty(), tl + ": no ell-torsion, and none in the record");
                continue;
            }
            if (!check(tm && !tm->refused, tl + ": the record has the torsion module")) continue;
            check(tm->exponents == P.E->sigma, tl + ": exponents " + Join(P.E->sigma) + " = the record's " + Join(tm->exponents));
            if (!tm->module_certified) {
                Skip(tl, "the record's torsion module is not certified (no canon to compare)");
                continue;
            }
            for (size_t o = 0; o < c.Z.ps.size(); ++o) {
                Mat Phi(P.t, P.t);
                for (uint32_t a = 0; a < P.t; ++a)
                    for (uint32_t b = 0; b < P.t; ++b) Phi.at(a, b) = static_cast<uint32_t>(c.Z.tau[o][size_t(P.t0 + a) * T + P.t0 + b]);
                const h3::TorsionInvariants I = h3::Invariants(P.ell, P.E->sigma, Phi, true, P.E->K_run - P.E->sigma_max);
                std::string theirs = "?";
                for (const auto& [p, s] : tm->invariants)
                    if (p == c.Z.ps[o]) theirs = s;
                check(I.ok && I.canon == theirs, tl + " T_" + std::to_string(c.Z.ps[o]) + ": tau's block reads " + I.canon +
                                                     " = the record's torsion canon " + theirs);
            }
        }
    }
}

// ---- (G) hand checks, the step-1 status words, the refusal pass-through ---------------------------------------------------------
const Cell* Find(uint32_t N, uint32_t n) {
    for (const Cell& c : Cells())
        if (c.N == N && c.n == n) return &c;
    return nullptr;
}
std::map<uint32_t, std::vector<uint32_t>> TorsionByEll(const mx::ModuleZ& Z) {
    std::map<uint32_t, std::vector<uint32_t>> m;
    for (uint32_t i = 0; i < Z.torsion_rank(); ++i) m[Z.tor_ell[i]].push_back(Z.tor_sigma[i]);
    for (auto& [p, e] : m) std::sort(e.begin(), e.end());
    return m;
}
void ArmG() {
    for (const Cell& c : Cells()) {
        check(c.Z.free_block == "pending(step 2)" && c.Z.splitting == "export_retraction/v1" && c.Z.split == "unknown" &&
                  c.Z.M_basis == "local" && c.Z.entry_encoding == "word",
              "(G) " + c.tag() + ": the step-1 words (free_block pending(step 2), export_retraction/v1, split unknown, M_basis local, word)");
        check(c.Z.tau.size() == c.Z.ps.size(), "(G) " + c.tag() + ": one tau per Hecke prime");
    }
    if (const Cell* c = Find(11, 0)) {
        check(c->Z.ok && c->Z.torsion_rank() == 0 && c->Z.free_rank == 3 && c->Z.largest_torsion_prime() == 0,
              "(G) (11,0): T = 0 (n = 0: H^1 = Hom(Gamma^ab, Z)), r = 2 g + cusps - 1 = 3");
    }
    if (const Cell* c = Find(1, 10)) {
        const auto m = TorsionByEll(c->Z);
        check(c->Z.ok && c->Z.free_rank == 3 && m == std::map<uint32_t, std::vector<uint32_t>>{{2, {1, 2}}, {3, {1}}},
              "(G) (1,10): T = Z/2 + Z/4 + Z/3, r = 3 (dim M_12 = 2: 2 dim S_12 + 1)");
        bool ch = false;
        for (const auto& e : c->R.charpoly)
            if (e.p == 2) ch = e.coefficients == std::vector<std::string>{"1", "-2001", "-97776", "-1180224"};
        check(ch, "(G) (1,10): char_Z(T_2) = (x - 2049)(x + 24)^2 = x^3 - 2001 x^2 - 97776 x - 1180224");
    }
    if (const Cell* c = Find(8, 10)) {
        const auto m = TorsionByEll(c->Z);
        check(c->Z.ok && c->Z.free_rank == 22 && m == std::map<uint32_t, std::vector<uint32_t>>{{2, {1, 1, 1, 2, 3, 3}}, {3, {1}}},
              "(G) (8,10): T = (Z/2)^3 + Z/4 + (Z/8)^2 + Z/3, r = 22 (EXPORT_report.md arm (b))");
    }
    // audit F4 (both verified against the auditor's independent model, audexportz1/cmp.py: 3530 and 1103 checks, 0 mismatches)
    if (const Cell* c = Find(1, 30)) {
        const auto m = TorsionByEll(c->Z);
        check(c->Z.ok && c->Z.free_rank == 5 && c->Z.largest_torsion_prime() == 29 &&
                  m == std::map<uint32_t, std::vector<uint32_t>>{{2, {1, 1, 1, 2, 2, 2}}, {3, {1, 1, 2}}, {5, {2}}, {29, {1}}},
              "(G) (1,30): T = (Z/2)^3 + (Z/4)^3 + (Z/3)^2 + Z/9 + Z/25 + Z/29, r = 5 (Z/25: K = 2 on the cokernel route)");
    }
    if (const Cell* c = Find(5, 12)) {
        const auto m = TorsionByEll(c->Z);
        check(c->Z.ok && c->Z.free_rank == 12 &&
                  m == std::map<uint32_t, std::vector<uint32_t>>{{2, {1, 2, 3}}, {3, {1, 2}}, {5, {1, 1}}, {11, {1}}},
              "(G) (5,12): T = Z/2 + Z/4 + Z/8 + Z/3 + Z/9 + (Z/5)^2 + Z/11, r = 12 ((Z/5)^2: t = 2 at ell = 5 dividing N)");
    }
    {
        mx::ModuleZOptions o;
        o.export_opt.rss_cap_bytes = 1;
        const mx::ModuleZ Z = mx::ExportModuleZ(1, 10, {2, 3}, o);
        check(!Z.ok && Z.status() == "blocked(export)" && Z.code == mx::ExportRefusal::RssCeiling &&
                  Z.refusal.find("[RssCeiling]") != std::string::npos && Z.tau.empty() && Z.d.empty() && Z.free_block == "pending(step 2)",
              "(G) (1,10) rss_cap_bytes = 1: blocked(export) [RssCeiling], no tau served, free_block still pending: " + Z.refusal);
    }
    // audit F1: a refusal at the SECOND torsion prime (ell = 3) after an accepted export at ell = 2 leaves nothing of the first:
    // no d, tau, local, and free_rank = 0 (the console prints these fields as "blocked")
    {
        const std::vector<uint32_t> ps = {2, 3};
        const mx::ExportProvider stub = [&ps](uint32_t ell) -> std::shared_ptr<const mx::ModuleExport> {
            mx::ExportOptions o;
            if (ell == 3) o.rss_cap_bytes = 1;
            return std::make_shared<const mx::ModuleExport>(mx::ExportModule(1, 10, ell, ps, o));
        };
        const mx::ModuleZ Z = mx::ExportModuleZ(1, 10, ps, {}, stub);
        check(!Z.ok && Z.status() == "blocked(export)" && Z.code == mx::ExportRefusal::RssCeiling && Z.refusal.rfind("ell = 3: ", 0) == 0 &&
                  Z.free_rank == 0 && Z.d.empty() && Z.tau.empty() && Z.local.empty() && Z.tor_ell.empty() && Z.largest_torsion_prime() == 0,
              "(G) (1,10) accepted at ell = 2, refused at ell = 3: blocked(export), free_rank reset to 0, no torsion data kept: " + Z.refusal);
    }
}

// ---- (S) the step-2 entry point: T_p in a CALLER-SUPPLIED basis B (ExportOptions::basis) ----------------------------------------
// F_B with T_p b_j = sum_i F_ij b_i mod T, mod ell^{K_max}.  Oracles: char(F_B) = the record's char_Z mod ell^{K_max} (basis-free);
// F_B(T_2) F_B(T_3) = F_B(T_3) F_B(T_2); B = the export's own free generators gives its free block; B G (G unit upper-triangular)
// gives G^{-1} F G; a singular or misshaped B is refused by name.  One random INTEGRAL B (entries in [-3, 3]) at two word primes.
std::vector<uint64_t> MatVec(const Mat& A) { return std::vector<uint64_t>(A.a.begin(), A.a.end()); }
std::vector<uint64_t> MulMod(const Mat& A, const Mat& B, uint64_t Q) {
    std::vector<uint64_t> C(size_t(A.r) * B.c, 0);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < B.c; ++j) {
            uint64_t s = 0;
            for (uint32_t l = 0; l < A.c; ++l) s = (s + uint64_t(A.at(i, l)) % Q * (B.at(l, j) % Q)) % Q;
            C[size_t(i) * B.c + j] = s;
        }
    return C;
}
std::vector<uint64_t> FB691, FB65521;   // F_B(T_2) of the single calls, for the batch
// char(F) mod q^K = the record's char_Z(T_p | H^1/tors) mod q^K (false if the record's entry is missing or undetermined)
bool CharMatches(const Cell& c, uint32_t p, const Mat& F, uint32_t q, uint32_t K) {
    const std::vector<uint32_t> ch = h3::CharpolyModPrimePower(F, q, K);
    const uint64_t Q = Pow(q, K);
    const rec::CharpolyEntry* ce = nullptr;
    for (const auto& e : c.R.charpoly)
        if (e.p == p) ce = &e;
    if (!ce || !ce->determined || ce->refused) return false;
    std::vector<uint64_t> rz;
    for (size_t i = 0; i < ce->coefficients.size(); ++i) {   // high -> low
        uint64_t v = 0;
        if (!(ReduceString(ce->coefficients[i], Q, v) || (i < ce->exact.size() && ReduceString(ce->exact[i], Q, v)))) return false;
        rz.push_back(v);
    }
    return std::vector<uint64_t>(ch.begin(), ch.end()) == rz;
}
// audit F5: a caller-supplied basis B = Own G + Tors h + d^0 w (G a random unit, L U with unit diagonals and random entries; h, w
// random) at a cell WITH ell-torsion: the read must drop the torsion and coboundary parts, and F_B = G^{-1} F_own G mod ell^{K_max}
// (K_max < K_run is asserted, so the precision drop is exercised); char(F_B) = char_Z mod ell^{K_max}.
void BasisMoved(uint32_t N, uint32_t n, uint32_t ell, const char* route) {
    const Cell* c = Find(N, n);
    const std::string tl = "(S) " + std::string(c ? c->tag() : "?") + " ell=" + std::to_string(ell) + " " + route;
    if (!check(c != nullptr && c->Z.ok, tl + ": the cell")) return;
    const mx::LocalPartZ* P = nullptr;
    for (const mx::LocalPartZ& x : c->Z.local)
        if (x.ell == ell) P = &x;
    if (!check(P != nullptr && P->t > 0 && P->E->route == route, tl + ": a local part with ell-torsion on this route")) return;
    const mx::ModuleExport& E0 = *P->E;
    const auto Mo = BuildModel(E0);
    const uint32_t cd = E0.cochain_dim, t = P->t, r = E0.free_rank;
    const uint64_t m = Mo->Z.m, Qmax = Pow(ell, E0.K_max);
    check(E0.K_max < E0.K_run, tl + ": K_max = " + std::to_string(E0.K_max) + " < K_run = " + std::to_string(E0.K_run) + " (the drop is exercised)");
    Mat Lw(r, r), Up(r, r), G(r, r);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            Lw.at(i, j) = i == j ? 1u : (i > j ? static_cast<uint32_t>(Rand() % m) : 0u);
            Up.at(i, j) = i == j ? 1u : (i < j ? static_cast<uint32_t>(Rand() % m) : 0u);
        }
    {
        const auto g = MulMod(Lw, Up, m);
        for (size_t i = 0; i < g.size(); ++i) G.a[i] = static_cast<uint32_t>(g[i]);
    }
    const uint32_t g0 = Mo->d0.c;
    std::vector<uint64_t> h(size_t(t) * r), w(size_t(g0) * r);
    for (auto& x : h) x = Rand() % m;
    for (auto& x : w) x = Rand() % m;
    Mat B(cd, r);
    for (uint32_t i = 0; i < cd; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            uint64_t s = 0;
            for (uint32_t l = 0; l < r; ++l) s = (s + uint64_t(E0.cocycles.at(i, t + l)) % m * G.at(l, j)) % m;
            for (uint32_t a = 0; a < t; ++a) s = (s + uint64_t(E0.cocycles.at(i, a)) % m * h[size_t(a) * r + j]) % m;
            for (uint32_t g = 0; g < g0; ++g) {
                const uint64_t d = Mo->d0.at(i, g) % m;
                if (d) s = (s + d * w[size_t(g) * r + j]) % m;
            }
            B.at(i, j) = static_cast<uint32_t>(s);
        }
    mx::ExportOptions o;
    o.basis = &B;
    const mx::ModuleExport E = mx::ExportModule(N, n, ell, c->ps, o);
    if (!check(E.ok && E.basis_ok && E.F_basis.size() == c->ps.size() && E.K_max == E0.K_max,
               tl + ": B = Own G + Tors h + d^0 w is a basis: " + E.basis_why))
        return;
    for (size_t k = 0; k < c->ps.size(); ++k) {
        Mat Fo(r, r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) Fo.at(i, j) = static_cast<uint32_t>(E0.T[k].at(t + i, t + j) % Qmax);
        const std::string tk = tl + " T_" + std::to_string(c->ps[k]);
        check(MulMod(G, E.F_basis[k], Qmax) == MulMod(Fo, G, Qmax), tk + ": G F_B = F_own G mod ell^K_max (F_B = G^{-1} F_own G)");
        check(CharMatches(*c, c->ps[k], E.F_basis[k], ell, E0.K_max), tk + ": char(F_B) = char_Z mod ell^K_max");
    }
}

void ArmS() {
    const Cell* c = Find(1, 10);
    if (!check(c != nullptr, "(S) the cell (1,10)")) return;
    const std::vector<uint32_t> ps = c->ps;
    Mat Bint;   // the integral basis: built once from the carrier's dimension (the same Z-cochains at every q)
    for (uint32_t q : {691u, 65521u}) {
        const std::string tq = "(S) (1,10) q=" + std::to_string(q);
        const mx::ModuleExport E0 = mx::ExportModule(1, 10, q, ps);
        if (!check(E0.ok && E0.route == "cokernel" && E0.free_rank == 3, tq + ": the export (cokernel carrier M_N, r = 3)")) continue;
        const uint32_t cd = E0.cochain_dim, r = E0.free_rank, t = static_cast<uint32_t>(E0.sigma.size());
        const uint64_t Q = Pow(q, E0.K_max);
        if (Bint.r == 0) {
            Bint = Mat(cd, r);
            for (auto& x : Bint.a) x = static_cast<uint32_t>(Rand() % 7);   // 0..6 ~ -3..3 after the shift below
        }
        ENGINE_CHECK(Bint.r == cd);
        Mat B(cd, r);
        for (uint32_t i = 0; i < cd; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                const int64_t v = int64_t(Bint.at(i, j)) - 3;
                const int64_t Qrun = int64_t(Pow(q, E0.K_run));
                B.at(i, j) = static_cast<uint32_t>((v % Qrun + Qrun) % Qrun);
            }
        mx::ExportOptions o;
        o.basis = &B;
        const mx::ModuleExport E = mx::ExportModule(1, 10, q, ps, o);
        if (!check(E.ok && E.basis_given && E.basis_ok && E.F_basis.size() == ps.size(), tq + ": a random integral B is a basis: " + E.basis_why)) continue;
        for (size_t k = 0; k < ps.size(); ++k) {
            const std::vector<uint32_t> ch = h3::CharpolyModPrimePower(E.F_basis[k], q, E0.K_max);
            const rec::CharpolyEntry* ce = nullptr;
            for (const auto& e : c->R.charpoly)
                if (e.p == ps[k]) ce = &e;
            std::vector<uint64_t> rz;
            bool parsed = ce && ce->determined;
            for (size_t i = 0; ce && i < ce->coefficients.size(); ++i) {   // high -> low
                uint64_t v = 0;
                parsed = parsed && (ReduceString(ce->coefficients[i], Q, v) || (i < ce->exact.size() && ReduceString(ce->exact[i], Q, v)));
                rz.push_back(v);
            }
            check(parsed && std::vector<uint64_t>(ch.begin(), ch.end()) == rz,
                  tq + " T_" + std::to_string(ps[k]) + ": char(T_p in B) = char_Z(T_p | H^1/tors) mod q^" + std::to_string(E0.K_max));
        }
        check(MulMod(E.F_basis[0], E.F_basis[1], Q) == MulMod(E.F_basis[1], E.F_basis[0], Q), tq + ": T_2, T_3 in B commute");
        (q == 691 ? FB691 : FB65521) = MatVec(E.F_basis[0]);
        // B = the export's own free generators: the free block, coordinates the identity
        Mat Own(cd, r);
        for (uint32_t i = 0; i < cd; ++i)
            for (uint32_t j = 0; j < r; ++j) Own.at(i, j) = E0.cocycles.at(i, t + j);
        mx::ExportOptions o2;
        o2.basis = &Own;
        const mx::ModuleExport E2 = mx::ExportModule(1, 10, q, ps, o2);
        bool same = E2.ok && E2.basis_ok;
        for (size_t k = 0; same && k < ps.size(); ++k) {
            Mat Fe(r, r);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) Fe.at(i, j) = static_cast<uint32_t>(E0.T[k].at(t + i, t + j) % Q);
            same = MatVec(E2.F_basis[k]) == MatVec(Fe);
        }
        check(same, tq + ": B = the export's free generators gives the export's free block");
        // B G, G unit upper-triangular: G F_{BG} = F_B G
        Mat G(r, r);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = i; j < r; ++j) G.at(i, j) = i == j ? 1u : static_cast<uint32_t>(Rand() % Q);
        Mat BG(cd, r);
        const auto bg = MulMod(B, G, Pow(q, E0.K_run));
        for (size_t i = 0; i < bg.size(); ++i) BG.a[i] = static_cast<uint32_t>(bg[i]);
        mx::ExportOptions o3;
        o3.basis = &BG;
        const mx::ModuleExport E3 = mx::ExportModule(1, 10, q, ps, o3);
        bool conj = E3.ok && E3.basis_ok;
        for (size_t k = 0; conj && k < ps.size(); ++k) conj = MulMod(G, E3.F_basis[k], Q) == MulMod(E.F_basis[k], G, Q);
        check(conj, tq + ": B -> B G conjugates: G F_{BG} = F_B G");
        // refusals by name
        Mat Sing = B;
        for (uint32_t i = 0; i < cd; ++i) Sing.at(i, r - 1) = Sing.at(i, 0);
        mx::ExportOptions o4;
        o4.basis = &Sing;
        const mx::ModuleExport E4 = mx::ExportModule(1, 10, q, ps, o4);
        check(E4.ok && !E4.basis_ok && E4.F_basis.empty() && E4.basis_why.find("singular") != std::string::npos,
              tq + ": a repeated column is refused (singular): " + E4.basis_why);
        Mat Bad(cd, r + 1);
        mx::ExportOptions o5;
        o5.basis = &Bad;
        const mx::ModuleExport E5 = mx::ExportModule(1, 10, q, ps, o5);
        check(E5.ok && !E5.basis_ok && E5.basis_why.find("must be") != std::string::npos, tq + ": a misshaped B is refused: " + E5.basis_why);
    }
    // the batch over primes (the step-2 hook): the same integral B at 691, 65521, 7, 1009 (7: a q^K_run ring with K_max > 1), one and
    // two threads bit-identical, and equal to the single calls above at 691 / 65521
    if (Bint.r > 0) {
        std::vector<int64_t> Bz(Bint.a.size());
        for (size_t i = 0; i < Bz.size(); ++i) Bz[i] = int64_t(Bint.a[i]) - 3;
        const std::vector<uint32_t> qs = {691, 65521, 7, 1009, 3};
        const auto b1 = mx::MatricesInBasisBatch(1, 10, ps, qs, Bz, Bint.r, Bint.c, 1);
        const auto b2 = mx::MatricesInBasisBatch(1, 10, ps, qs, Bz, Bint.r, Bint.c, 2);
        bool same = b1.size() == b2.size();
        for (size_t s = 0; same && s < b1.size(); ++s) {
            same = b1[s].ok == b2[s].ok && b1[s].K == b2[s].K && b1[s].why == b2[s].why && b1[s].F.size() == b2[s].F.size();
            for (size_t k = 0; same && k < b1[s].F.size(); ++k) same = b1[s].F[k].a == b2[s].F[k].a;
        }
        check(same, "(S) batch: 1 and 2 threads bit-identical (ok, K, why, F)");
        uint32_t okc = 0;
        for (size_t s = 0; s + 1 < qs.size(); ++s) {
            const auto& e = b1[s];
            if (!check(e.ok && e.F.size() == ps.size(), "(S) batch q=" + std::to_string(e.q) + " ok: " + e.why)) continue;
            ++okc;
            const uint64_t Q = Pow(e.q, e.K);
            for (size_t k = 0; k < ps.size(); ++k) {
                const std::vector<uint32_t> ch = h3::CharpolyModPrimePower(e.F[k], e.q, e.K);
                const rec::CharpolyEntry* ce = nullptr;
                for (const auto& x : c->R.charpoly)
                    if (x.p == ps[k]) ce = &x;
                std::vector<uint64_t> rz;
                bool parsed = ce && ce->determined;
                for (size_t i = 0; ce && i < ce->coefficients.size(); ++i) {   // high -> low
                    uint64_t v = 0;
                    parsed = parsed && (ReduceString(ce->coefficients[i], Q, v) || (i < ce->exact.size() && ReduceString(ce->exact[i], Q, v)));
                    rz.push_back(v);
                }
                check(parsed && std::vector<uint64_t>(ch.begin(), ch.end()) == rz,
                      "(S) batch q=" + std::to_string(e.q) + " T_" + std::to_string(ps[k]) + ": char = char_Z mod q^" + std::to_string(e.K));
            }
            if (e.q == 691 || e.q == 65521) check(MatVec(e.F[0]) == (e.q == 691 ? FB691 : FB65521), "(S) batch q=" + std::to_string(e.q) + ": = the single call");
        }
        check(!b1.back().ok && b1.back().why.find("q >= 5") != std::string::npos, "(S) batch q=3 refused by name: " + b1.back().why);
        std::cout << "  (S) batch: " << okc << " of " << qs.size() - 1 << " primes served, threads 1 = 2" << std::endl;
        // audit F9: q | N is refused by name, before any export runs
        const auto bN = mx::MatricesInBasisBatch(5, 12, {2, 3}, {5}, Bz, Bint.r, Bint.c, 1);
        check(bN.size() == 1 && !bN[0].ok && bN[0].why.find("divides N") != std::string::npos, "(S) batch q=5 at N=5 refused by name: " +
                                                                                                  (bN.empty() ? std::string("?") : bN[0].why));
        // audit F2: a defect in a lane is reported (rethrown after the join, the first in qs order), never std::terminate
        for (uint32_t th : {1u, 2u}) {
            std::string got;
            try {
                mx::MatricesInBasisBatch(1, 10, ps, {691, 65521, 7, 1009}, Bz, Bint.r, Bint.c, th, [](uint32_t q) {
                    if (q == 65521) throw std::logic_error("injected defect at q = 65521");
                    if (q == 7) throw std::bad_alloc();
                });
                got = "no exception";
            } catch (const std::logic_error& e) {
                got = e.what();
            } catch (const std::exception& e) {
                got = std::string("another exception: ") + e.what();
            }
            check(got == "injected defect at q = 65521", "(S) batch threads=" + std::to_string(th) +
                                                              ": an injected lane defect is rethrown after the join (first in qs order): " + got);
        }
    }
    // the tree route takes a basis too (its carrier is ell-local: C^1_T in J-coordinates): own generators at ell = 2, (8,10)
    if (const Cell* c8 = Find(8, 10)) {
        for (const mx::LocalPartZ& P : c8->Z.local) {
            if (P.ell != 2) continue;
            const mx::ModuleExport& E0 = *P.E;
            const uint32_t t = static_cast<uint32_t>(E0.sigma.size()), r = E0.free_rank;
            Mat Own(E0.cochain_dim, r);
            for (uint32_t i = 0; i < E0.cochain_dim; ++i)
                for (uint32_t j = 0; j < r; ++j) Own.at(i, j) = E0.cocycles.at(i, t + j);
            mx::ExportOptions o;
            o.basis = &Own;
            const mx::ModuleExport E = mx::ExportModule(8, 10, 2, c8->ps, o);
            const uint64_t Q = Pow(2, E0.K_max);
            bool same = E.ok && E.basis_ok;
            for (size_t k = 0; same && k < c8->ps.size(); ++k)
                for (uint32_t i = 0; same && i < r; ++i)
                    for (uint32_t j = 0; same && j < r; ++j) same = E.F_basis[k].at(i, j) == E0.T[k].at(t + i, t + j) % Q;
            check(same, "(S) (8,10) ell=2 tree: B = the export's free generators gives its free block mod 2^K_max: " + E.basis_why);
        }
    }
    BasisMoved(9, 8, 7, "cokernel");   // Z/7, q = 7 a torsion prime of the export
    BasisMoved(8, 10, 2, "tree");      // (Z/2)^3 + Z/4 + (Z/8)^2, a random basis on the tree route
}

}  // namespace

uint32_t HeckeZExportZValidation() {
    std::cout << "HeckeZExportZValidation: H^1(Gamma_0(N); Sym^n Z^2) over Z, tau and M (slot-25 EXPORT-Z step 1; F pending step 2)" << std::endl;
    g_rng = 0x9e3779b97f4a7c15ull;
    const auto t0 = clk::now();
    try {
        Cells();
    } catch (const std::exception& e) {
        std::cout << "  [EXCEPTION] building the cells: " << e.what() << std::endl;
        std::cout << "HECKEZ_EXPORTZ arms=7 failing=7 verdict=FAIL" << std::endl;
        return 1;
    }
    std::cout << "  cells built in " << std::fixed << std::setprecision(2) << since(t0) << " s" << std::endl;
    // audit F3: per-arm check floors (the cells are fixed and deterministic: the counts of record, slot-25 EXPORT-Z audit fixes).
    // An arm that shrinks below its floor (a skipped block, a record regression) FAILS; skips print their reason (SKIP lines).
    struct ArmDef {
        const char* name;
        std::function<void()> fn;
        uint64_t floor;
    };
    const std::vector<ArmDef> arms = {{"A", ArmA, 124}, {"B", ArmB, 62}, {"C", ArmC, 76}, {"D", ArmD, 79},
                                      {"E", ArmE, 213}, {"G", ArmG, 24}, {"S", ArmS, 55}};
    uint32_t failing = 0;
    for (const auto& [name, fn, floor] : arms) {
        g_fail = 0;
        g_skips = 0;
        const uint64_t c0 = g_checks;
        const auto ta = clk::now();
        try {
            fn();
        } catch (const std::exception& e) {
            ++g_fail;
            std::cout << "  [EXCEPTION] arm (" << name << "): " << e.what() << std::endl;
        }
        const uint64_t n_checks = g_checks - c0;
        if (n_checks < floor) {
            ++g_fail;
            std::cout << "  [FAIL] arm (" << name << "): " << n_checks << " checks, below the floor " << floor << std::endl;
        }
        if (g_fail) ++failing;
        std::cout << "ARM (" << name << ") checks=" << n_checks << " floor=" << floor << " skipped=" << g_skips << " failed=" << g_fail
                  << " seconds=" << std::fixed << std::setprecision(2) << since(ta) << std::endl;
    }
    uint32_t largest = 0;
    for (const Cell& c : Cells()) largest = std::max(largest, c.Z.largest_torsion_prime());
    const bool ok = failing == 0;
    std::cout << "HECKEZ_EXPORTZ arms=" << arms.size() << " failing=" << failing << " cells=" << Cells().size() << " checks=" << g_checks
              << " largest_torsion_prime=" << largest << " free_block=pending(step 2) verdict=" << (ok ? "OK" : "FAIL") << " seconds=" << std::fixed
              << std::setprecision(2) << since(t0) << std::endl;
    return ok ? 0 : 1;
}
