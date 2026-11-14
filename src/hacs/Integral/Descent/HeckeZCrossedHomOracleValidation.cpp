// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-26 card Z3b: HeckeZCrossedHomOracleValidation (standard).  See the header and docs/implementation/slot26/Z3b_design.md v0.2.
#include "HeckeZCrossedHomOracleValidation.h"

#include "DescentFreeBasisZ.h"
#include "DescentModuleExport.h"
#include "DescentModuleExportZ.h"
#include "HeckeZCrossedHomOracle.h"
#include "IsoTreeCarrierZ.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace mx = integral_descent::modexport;
namespace fz = integral_descent::freez;
namespace xh = integral_descent::xhom;
using nf::BigInt;
using hecke::Mat22;
using clk = std::chrono::steady_clock;
using Vec = std::vector<uint64_t>;

uint32_t g_fail = 0;
uint64_t g_checks = 0, g_teeth = 0, g_caught = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 40) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
// a tooth: `fired` must be true (the targeted check FAILED on the planted error)
void tooth(bool fired, const std::string& what) {
    ++g_teeth;
    if (fired) ++g_caught;
    check(fired, "tooth not caught: " + what);
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
uint64_t VmHwmKb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmHWM:", 0) == 0) return std::strtoull(line.c_str() + 6, nullptr, 10);
    return 0;
}
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
};

// ---- residue helpers (row-major, k columns) --------------------------------------------------------------------------------------
Vec PackedMod(const fz::PackedZ& P, uint64_t Q) {
    Vec X(size_t(P.rows) * P.cols, 0);
    for (uint32_t j = 0; j < P.cols; ++j)
        for (uint64_t e = P.colptr[j]; e < P.colptr[j + 1]; ++e) X[size_t(P.row[e]) * P.cols + j] = P.mod_u32(e, static_cast<uint32_t>(Q));
    return X;
}
Vec SparseMul(const fz::SparseZ& A, uint64_t Q, const Vec& X, uint32_t k) {
    ENGINE_CHECK_MSG(X.size() == size_t(A.cols) * k, "SparseMul: shape");
    Vec Y(size_t(A.rows) * k, 0);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint64_t e = A.rowptr[i]; e < A.rowptr[i + 1]; ++e) {
            const uint64_t a = A.val[e].mod_u32(static_cast<uint32_t>(Q));
            if (!a) continue;
            for (uint32_t c = 0; c < k; ++c)
                Y[size_t(i) * k + c] = static_cast<uint64_t>((Y[size_t(i) * k + c] + (unsigned __int128)a * X[size_t(A.col[e]) * k + c]) % Q);
        }
    return Y;
}
// A^T X (A: rows x a, X: rows x k) mod Q
Vec TransMul(const Vec& A, uint32_t rows, uint32_t a, const Vec& X, uint32_t k, uint64_t Q) {
    Vec Y(size_t(a) * k, 0);
    for (uint32_t i = 0; i < a; ++i)
        for (uint32_t c = 0; c < k; ++c) {
            unsigned __int128 s = 0;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint64_t x = A[size_t(r) * a + i];
                if (x) s = (s + (unsigned __int128)x * X[size_t(r) * k + c]) % Q;
            }
            Y[size_t(i) * k + c] = static_cast<uint64_t>(s);
        }
    return Y;
}
Vec Column(const Vec& X, uint32_t rows, uint32_t k, uint32_t j) {
    Vec c(rows);
    for (uint32_t i = 0; i < rows; ++i) c[i] = X[size_t(i) * k + j];
    return c;
}
bool AllZero(const Vec& v) {
    for (uint64_t x : v)
        if (x) return false;
    return true;
}

// ---- the cells -----------------------------------------------------------------------------------------------------------------
struct ModData {
    uint64_t Q = 0;
    uint32_t ell = 0;                 // 0: a word prime q1 / q2
    std::unique_ptr<xh::CrossedModel> X;
    Vec Bq;                           // B mod Q (3D x r)
    std::vector<Vec> y, yI;           // per op: y_j, and the rho(delta^iota) tooth (q1 only)
    Vec yW;                           // op 0 with the changed representative system (q1 and the units)
    Vec Fq;                           // per op F mod Q, concatenated (r x r each)
};
struct Unit {
    size_t l = 0, mod = 0;
    uint32_t ell = 0, t = 0, t0 = 0, K_run = 0, nu = 0;
    uint64_t Q = 0;
    std::vector<Vec> piY;             // per op: t x r
    Vec piB, piCob, piG, piYW, piCoc; // t x r, t x 2, t x nu, t x r, t x t
    std::vector<Vec> piYG;            // per i: t x 1 (column i % r of op 0 shifted by g_i)
    std::vector<Vec> piTG;            // (slot-27 Z3b-F, tau arm) per op: t x nu, pi(R T^x g_i)
    Vec piTGG;                        // (slot-27 Z3b-F, tau tooth) t x nu, pi(R (T^x g_i + g_i)) at op 0
    std::vector<Vec> Fd;              // per op: F mod ell^{sigma_max} (r x r)
};
struct Cell {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    std::unique_ptr<fz::IsoTreeCarrierZ> car;
    mx::FreeBlockRun run;
    mx::ExportCache cache;
    mx::ModuleZ Z;
    uint32_t r = 0, D = 0;
    uint64_t q1 = 0, q2 = 0;
    std::vector<ModData> mods;        // 0: q1, 1: q2, then one per unit
    std::vector<Unit> units;
    xh::HeckeStats hs;
    double t_export = 0, t_oracle = 0;
    bool prepared = false;
    std::string tag() const { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }
};

std::vector<Mat22> AltReps(uint32_t p, uint32_t N) {
    std::vector<Mat22> r = xh::StandardReps(p);
    for (size_t j = 0; j < r.size(); ++j) {
        const Mat22 b = j % 2 == 0 ? Mat22{1, int64_t(j) + 1, 0, 1} : Mat22{1, 0, int64_t(N), 1};
        r[j] = r[j] * b;
    }
    return r;
}

void BuildCell(Cell& c) {
    const auto t0 = clk::now();
    c.car = std::make_unique<fz::IsoTreeCarrierZ>(c.N, c.n);
    mx::ModuleZ2Options o;
    c.run = mx::RunFreeBlock(*c.car, c.ps, o);
    c.Z = mx::AssembleModuleZ2(c.N, c.n, c.ps, c.run, o, &c.cache);
    ENGINE_CHECK_MSG(c.Z.ok && c.Z.free_block == "exact", ("XHOM: the export of " + c.tag() + " is not exact: " + c.Z.refusal + c.Z.free_refusal).c_str());
    c.r = c.Z.free_rank;
    c.D = c.car->D();
    ENGINE_CHECK_MSG(c.run.FB.B.cols == c.r && c.run.FB.Phi_cert.cols == c.r && c.run.FB.B.rows == 3 * c.D, "XHOM: B / Phi_cert shapes");
    std::vector<uint64_t> qs;
    for (uint32_t q : fz::CachedCrtPrimes(16))
        if ((6ull * c.N) % q != 0 && q > c.n + 1 && qs.size() < 2) qs.push_back(q);
    ENGINE_CHECK_MSG(qs.size() == 2, "XHOM: two word primes");
    c.q1 = qs[0];
    c.q2 = qs[1];
    for (size_t l = 0; l < c.Z.local.size(); ++l) {
        const mx::LocalPartZ& P = c.Z.local[l];
        if (P.t == 0) continue;
        Unit u;
        u.l = l;
        u.ell = P.ell;
        u.t = P.t;
        u.t0 = P.t0;
        u.K_run = P.E->K_run;
        uint64_t Q = 1;
        for (uint32_t i = 0; i < u.K_run; ++i) Q *= P.ell;
        u.Q = Q;
        ENGINE_CHECK_MSG(P.E->sigma_max <= u.K_run && Q < (1ull << 32), "XHOM: sigma_max <= K_run, ell^K_run < 2^32");
        c.units.push_back(std::move(u));
    }
    c.t_export = since(t0);
}

// ---- (K) conventions, before any Hecke step --------------------------------------------------------------------------------------
void ArmK(std::vector<Cell>& cells) {
    for (Cell& c : cells) {
        const std::string t = "(K) " + c.tag();
        const uint32_t D = c.D, r = c.r, n = c.n;
        const xh::CrossedModel X(c.N, n, c.q1);
        const uint32_t d = X.d, m = X.m;
        check(X.m == c.car->m() && X.D == D, t + ": the model's index and dim agree with the carrier's");
        const hecke::level::Gamma0LevelN A(c.N);
        check(A.rep(1) == (Mat22{1, 0, 0, 1}), t + ": w_1 = I (the Shapiro base class j0 = 1)");
        check(c.car->b0() == 0, t + ": H^0 = 0 (b_0 = 0: the hypothesis of the (I) arm and of the integral comparison)");
        // exact blocks of the oracle's own rho_n vs the carrier's d^0, and vs RhoNZ
        const fz::SparseZ& d0 = c.car->d(0);
        std::vector<BigInt> dense(size_t(3) * D * 2 * D, BigInt(0)), expect(size_t(3) * D * 2 * D, BigInt(0));
        for (uint32_t i = 0; i < d0.rows; ++i)
            for (uint64_t e = d0.rowptr[i]; e < d0.rowptr[i + 1]; ++e) dense[size_t(i) * 2 * D + d0.col[e]] = d0.val[e];
        bool rho_ok = true;
        for (uint8_t x : {uint8_t(xh::kS), uint8_t(xh::kU)}) {
            const uint32_t blk = x == xh::kS ? 0 : 1;
            for (uint32_t j = 0; j < m; ++j) {
                const std::vector<BigInt> R = xh::SymRhoExact(X.h[x][j], n);
                rho_ok = rho_ok && R == fz::RhoNZ(X.h[x][j], n);
                for (uint32_t s = 0; s < d; ++s)
                    for (uint32_t u = 0; u < d; ++u) {
                        BigInt& E = expect[size_t(blk * D + j * d + s) * 2 * D + blk * D + X.col[x][j] * d + u];
                        E = E - R[size_t(s) * d + u];
                    }
            }
            for (uint32_t i = 0; i < D; ++i) {
                BigInt& E = expect[size_t(blk * D + i) * 2 * D + blk * D + i];
                E = E + BigInt(1);
            }
        }
        for (uint32_t i = 0; i < D; ++i) {
            expect[size_t(2 * D + i) * 2 * D + i] = BigInt(1);
            expect[size_t(2 * D + i) * 2 * D + D + i] = BigInt(-1);
        }
        auto rows_eq = [&](uint32_t r0, uint32_t r1) {
            for (size_t e = size_t(r0) * 2 * D; e < size_t(r1) * 2 * D; ++e)
                if (dense[e] != expect[e]) return false;
            return true;
        };
        check(rows_eq(0, D), t + ": d^0's S-rows = (1 - rho_{M_N}(S) | 0), from the oracle's rho_n and coset table, exactly");
        check(rows_eq(D, 2 * D), t + ": d^0's U-rows = (0 | 1 - rho_{M_N}(U)), exactly");
        check(rows_eq(2 * D, 3 * D), t + ": d^0's edge rows = (1 | -1), exactly");
        check(rho_ok, t + ": the oracle's rho_n = RhoNZ on the 2m blocks rho_n(h_j(S)), rho_n(h_j(U))");
        // kappa / kappa' identities mod q1 on seeded random data
        const uint64_t Q = c.q1;
        Rng rng{0x5eed0000ull + c.N * 1000ull + n};
        Vec x(D), y(D), mm(D), xy(2 * D), mmm(2 * D);
        for (uint32_t i = 0; i < D; ++i) {
            x[i] = rng.next() % Q;
            y[i] = rng.next() % Q;
            mm[i] = rng.next() % Q;
        }
        for (uint32_t i = 0; i < D; ++i) {
            xy[i] = x[i];
            xy[D + i] = y[i];
            mmm[i] = mm[i];
            mmm[D + i] = mm[i];
        }
        auto delta = [&](const Vec& v, Vec& dS, Vec& dU) {
            Vec Sv, Uv;
            X.act(xh::kS, v, 1, Sv);
            X.act(xh::kU, v, 1, Uv);
            dS.resize(D);
            dU.resize(D);
            for (uint32_t i = 0; i < D; ++i) {
                dS[i] = (v[i] + Q - Sv[i]) % Q;
                dU[i] = (v[i] + Q - Uv[i]) % Q;
            }
        };
        {
            Vec fS, fU, dS, dU;
            X.kappa(SparseMul(d0, Q, xy, 1), 1, fS, fU);
            delta(x, dS, dU);
            check(fS == dS && fU == dU, t + ": kappa(d^0(x, y)) = delta x (random x, y mod q1)");
            delta(mm, dS, dU);
            check(X.kappa_prime(dS, dU, 1) == SparseMul(d0, Q, mmm, 1), t + ": kappa'(delta m) = d^0(m, m) (random m mod q1)");
            Vec gS, gU;
            X.kappa(X.kappa_prime(x, y, 1), 1, gS, gU);
            check(gS == x && gU == y, t + ": kappa kappa' = id");
        }
        // d^1 kappa' kappa b = 0 EXACTLY (BigInt), every column of B
        {
            const fz::SparseZ& d1 = c.car->d(1);
            std::vector<std::vector<BigInt>> RU(m);
            for (uint32_t j = 0; j < m; ++j) RU[j] = xh::SymRhoExact(X.h[xh::kU][j], n);
            bool ok = true;
            for (uint32_t j = 0; ok && j < r; ++j) {
                const std::vector<BigInt> b = c.run.FB.B.col_dense(j);
                std::vector<BigInt> v(3 * D, BigInt(0));
                for (uint32_t i = 0; i < 2 * D; ++i) v[i] = b[i];
                for (uint32_t jj = 0; jj < m; ++jj)   // + z - U z on the U-cell
                    for (uint32_t s = 0; s < d; ++s) {
                        BigInt acc = b[2 * D + jj * d + s];
                        for (uint32_t u = 0; u < d; ++u)
                            if (!RU[jj][size_t(s) * d + u].is_zero()) acc -= RU[jj][size_t(s) * d + u] * b[2 * D + X.col[xh::kU][jj] * d + u];
                        v[D + jj * d + s] += acc;
                    }
                for (uint32_t i = 0; ok && i < d1.rows; ++i) {
                    BigInt s(0);
                    for (uint64_t e = d1.rowptr[i]; e < d1.rowptr[i + 1]; ++e) s += d1.val[e] * v[d1.col[e]];
                    ok = s.is_zero();
                }
            }
            check(ok, t + ": d^1 kappa' kappa (b_j) = 0 exactly, every column (kappa lands in crossed homomorphisms)");
        }
        // f(-I) = (1 + S) f_S == 0 on kappa(b) (audit N11)
        {
            const Vec Bq = PackedMod(c.run.FB.B, Q);
            Vec fS, fU, SfS;
            X.kappa(Bq, r, fS, fU);
            X.act(xh::kS, fS, r, SfS);
            bool z = true;
            for (size_t e = 0; e < fS.size(); ++e) z = z && (fS[e] + SfS[e]) % Q == 0;
            check(z, t + ": f(-I) = (1 + S) f_S == 0 on kappa(B) mod q1");
        }
        // words multiply back (ENGINE_CHECK inside Word): the 4m h_j(x) and 32 seeded random products of letters
        {
            uint64_t words = 0;
            for (uint8_t x2 = 0; x2 < 4; ++x2)
                for (uint32_t j = 0; j < m; ++j) {
                    xh::Word(X.h[x2][j]);
                    ++words;
                }
            for (int i = 0; i < 32; ++i) {
                Mat22 g{1, 0, 0, 1};
                for (int l = 0; l < 12; ++l) g = g * xh::LetterMat(static_cast<uint8_t>(rng.next() % 4));
                xh::Word(g);
                ++words;
            }
            // the real test is the ENGINE_CHECK inside xh::Word (it throws); this check records that none threw (audit F5)
            check(words == 4ull * m + 32, t + ": " + std::to_string(words) + " Euclid words multiply back (no exception from Word)");
        }
        // Phi_cert^T d^0 == 0 (the hypothesis of (F); audit N6) mod q1, q2 and ell^{K_run} at every unit
        std::vector<uint64_t> mods = {c.q1, c.q2};
        for (const Unit& u : c.units) mods.push_back(u.Q);
        for (uint64_t M : mods) {
            const Vec Ph = PackedMod(c.run.FB.Phi_cert, M);
            // (Phi^T d^0)[a][col] = sum_i Phi[i][a] d0[i][col]
            Vec acc(size_t(r) * 2 * D, 0);
            for (uint32_t i = 0; i < d0.rows; ++i)
                for (uint64_t e = d0.rowptr[i]; e < d0.rowptr[i + 1]; ++e) {
                    const uint64_t v = d0.val[e].mod_u32(static_cast<uint32_t>(M));
                    if (!v) continue;
                    for (uint32_t a = 0; a < r; ++a) {
                        const uint64_t p = Ph[size_t(i) * r + a];
                        if (p) acc[size_t(a) * 2 * D + d0.col[e]] = static_cast<uint64_t>((acc[size_t(a) * 2 * D + d0.col[e]] + (unsigned __int128)p * v) % M);
                    }
                }
            check(AllZero(acc), t + ": Phi_cert^T d^0 == 0 mod " + std::to_string(M));
        }
    }
}

// ---- the oracle (after (K)) ------------------------------------------------------------------------------------------------------
void FillMod(Cell& c, ModData& md, bool alt) {
    md.X = std::make_unique<xh::CrossedModel>(c.N, c.n, md.Q);
    md.Bq = PackedMod(c.run.FB.B, md.Q);
    Vec fS, fU;
    md.X->kappa(md.Bq, c.r, fS, fU);
    for (uint32_t p : c.ps) md.y.push_back(xh::HeckeCross(*md.X, fS, fU, c.r, p, xh::StandardReps(p), false, &c.hs));
    if (md.ell == 0 && md.Q == c.q1)
        for (uint32_t p : c.ps) md.yI.push_back(xh::HeckeCross(*md.X, fS, fU, c.r, p, xh::StandardReps(p), true, &c.hs));
    if (alt) md.yW = xh::HeckeCross(*md.X, fS, fU, c.r, c.ps[0], AltReps(c.ps[0], c.N), false, &c.hs);
    const uint32_t r = c.r;
    for (size_t o = 0; o < c.ps.size(); ++o)
        for (size_t e = 0; e < size_t(r) * r; ++e) md.Fq.push_back(c.Z.global.F[o][e].mod_u32(static_cast<uint32_t>(md.Q)));
}

// the kernel of A mod ell (A: rows x cols residues), a basis as columns
std::vector<Vec> KernelModEll(Vec A, uint32_t rows, uint32_t cols, uint64_t ell) {
    std::vector<int64_t> piv_of_col(cols, -1);
    uint32_t rk = 0;
    for (uint32_t cc = 0; cc < cols && rk < rows; ++cc) {
        uint32_t pr = rk;
        while (pr < rows && A[size_t(pr) * cols + cc] == 0) ++pr;
        if (pr == rows) continue;
        if (pr != rk)
            for (uint32_t j = 0; j < cols; ++j) std::swap(A[size_t(pr) * cols + j], A[size_t(rk) * cols + j]);
        // inverse of the pivot mod ell (ell prime)
        uint64_t a = A[size_t(rk) * cols + cc], inv = 1, e = ell - 2, b = a;
        while (e) {
            if (e & 1) inv = inv * b % ell;
            b = b * b % ell;
            e >>= 1;
        }
        for (uint32_t j = 0; j < cols; ++j) A[size_t(rk) * cols + j] = A[size_t(rk) * cols + j] * inv % ell;
        for (uint32_t i = 0; i < rows; ++i) {
            if (i == rk) continue;
            const uint64_t f = A[size_t(i) * cols + cc];
            if (!f) continue;
            for (uint32_t j = 0; j < cols; ++j) A[size_t(i) * cols + j] = (A[size_t(i) * cols + j] + (ell - f) * A[size_t(rk) * cols + j]) % ell;
        }
        piv_of_col[cc] = rk++;
    }
    std::vector<Vec> K;
    for (uint32_t fcol = 0; fcol < cols; ++fcol) {
        if (piv_of_col[fcol] >= 0) continue;
        Vec v(cols, 0);
        v[fcol] = 1;
        for (uint32_t cc = 0; cc < cols; ++cc)
            if (piv_of_col[cc] >= 0) v[cc] = (ell - A[size_t(piv_of_col[cc]) * cols + fcol]) % ell;
        K.push_back(std::move(v));
    }
    return K;
}
uint32_t RankModEll(Vec A, uint32_t rows, uint32_t cols, uint64_t ell) {
    return cols - static_cast<uint32_t>(KernelModEll(std::move(A), rows, cols, ell).size());
}

// pi(R X) for the columns X (3D x k residues mod ell^{K_run}): t x k, row a mod d_a
Vec PiOfRoute(const Cell& c, const Unit& u, const mx::Mat& RB, uint32_t c0, uint32_t k) {
    const mx::ModuleExport& E = *c.Z.local[u.l].E;
    ENGINE_CHECK_MSG(E.retraction.r == u.t && E.retraction.c == RB.r, "XHOM: the retraction's shape");
    Vec out(size_t(u.t) * k, 0);
    for (uint32_t a = 0; a < u.t; ++a) {
        const uint64_t da = c.Z.d[u.t0 + a];
        for (uint32_t j = 0; j < k; ++j) {
            unsigned __int128 s = 0;
            for (uint32_t x = 0; x < RB.r; ++x) {
                const uint64_t ra = E.retraction.at(a, x);
                if (ra) s = (s + (unsigned __int128)(ra % da) * (RB.at(x, c0 + j) % da)) % da;
            }
            out[size_t(a) * k + j] = static_cast<uint64_t>(s);
        }
    }
    return out;
}

void Prepare(Cell& c) {
    const auto t0 = clk::now();
    const uint32_t D = c.D, r = c.r;
    c.mods.clear();
    for (uint64_t q : {c.q1, c.q2}) {
        ModData md;
        md.Q = q;
        FillMod(c, md, q == c.q1);
        c.mods.push_back(std::move(md));
    }
    const fz::SparseZ& d0 = c.car->d(0);
    for (Unit& u : c.units) {
        ModData md;
        md.Q = u.Q;
        md.ell = u.ell;
        FillMod(c, md, true);
        u.mod = c.mods.size();
        c.mods.push_back(std::move(md));
        const ModData& M = c.mods[u.mod];
        const uint64_t Q = u.Q;
        // (I) the carrier's ell-torsion cocycles g_i = d^0(m_i) / ell, m_i a basis of ker(d^0 mod ell)
        Vec A0(size_t(d0.rows) * d0.cols, 0);
        for (uint32_t i = 0; i < d0.rows; ++i)
            for (uint64_t e = d0.rowptr[i]; e < d0.rowptr[i + 1]; ++e) A0[size_t(i) * d0.cols + d0.col[e]] = d0.val[e].mod_u32(u.ell);
        const std::vector<Vec> ker = KernelModEll(std::move(A0), d0.rows, d0.cols, u.ell);
        u.nu = static_cast<uint32_t>(ker.size());
        std::vector<Vec> G;
        for (const Vec& mv : ker) {
            Vec g(d0.rows);
            for (uint32_t i = 0; i < d0.rows; ++i) {
                BigInt s(0);
                for (uint64_t e = d0.rowptr[i]; e < d0.rowptr[i + 1]; ++e)
                    if (mv[d0.col[e]]) s += d0.val[e] * BigInt(int64_t(mv[d0.col[e]]));
                ENGINE_CHECK_MSG(s.mod_u32(u.ell) == 0, "XHOM: d^0(m) is not divisible by ell for m in ker(d^0 mod ell)");
                g[i] = s.div_exact_u32(u.ell).mod_u32(static_cast<uint32_t>(Q));
            }
            G.push_back(std::move(g));
        }
        // one RouteBasis call per (cell, ell) (audit N8): y per op | B | d^0 m1, d^0 m2 | g_i | y_{i%r} + g_i | yW
        std::vector<Vec> cols;
        const uint32_t P = static_cast<uint32_t>(c.ps.size());
        for (uint32_t o = 0; o < P; ++o)
            for (uint32_t j = 0; j < r; ++j) cols.push_back(Column(M.y[o], 3 * D, r, j));
        for (uint32_t j = 0; j < r; ++j) cols.push_back(Column(M.Bq, 3 * D, r, j));
        Rng rng{0xc0b0000ull + u.ell * 7919ull + c.N * 1000ull + c.n};
        for (int s = 0; s < 2; ++s) {
            Vec m0(2 * D);
            for (auto& v : m0) v = rng.next() % Q;
            cols.push_back(SparseMul(d0, Q, m0, 1));
        }
        for (const Vec& g : G) cols.push_back(g);
        for (uint32_t i = 0; i < u.nu; ++i) {
            Vec v = Column(M.y[0], 3 * D, r, i % std::max<uint32_t>(r, 1));
            for (uint32_t e = 0; e < 3 * D; ++e) v[e] = (v[e] + G[i][e]) % Q;
            cols.push_back(std::move(v));
        }
        for (uint32_t j = 0; j < r; ++j) cols.push_back(Column(M.yW, 3 * D, r, j));
        // (slot-27 Z3b-F, design Q4 follow-up) T^x g_i per op: kappa(g_i) -> [C, eq:barcorrected] via Shapiro -> kappa', no T~_Iso;
        // then, at op 0, T^x g_i + g_i (the tau arm's cochain-level tooth).  Appended after yW, so every earlier column is unchanged.
        if (u.nu > 0) {
            Vec Gm(size_t(3) * D * u.nu, 0);
            for (uint32_t i = 0; i < u.nu; ++i)
                for (uint32_t e = 0; e < 3 * D; ++e) Gm[size_t(e) * u.nu + i] = G[i][e];
            Vec gS, gU;
            M.X->kappa(Gm, u.nu, gS, gU);
            std::vector<Vec> TG;
            for (uint32_t o = 0; o < P; ++o) TG.push_back(xh::HeckeCross(*M.X, gS, gU, u.nu, c.ps[o], xh::StandardReps(c.ps[o]), false, nullptr));
            for (uint32_t o = 0; o < P; ++o)
                for (uint32_t i = 0; i < u.nu; ++i) cols.push_back(Column(TG[o], 3 * D, u.nu, i));
            for (uint32_t i = 0; i < u.nu; ++i) {
                Vec v = Column(TG[0], 3 * D, u.nu, i);
                for (uint32_t e = 0; e < 3 * D; ++e) v[e] = (v[e] + G[i][e]) % Q;
                cols.push_back(std::move(v));
            }
        }
        fz::PackedZ Pk(3 * D);
        for (const Vec& v : cols) {
            std::vector<std::pair<uint32_t, BigInt>> ent;
            for (uint32_t e = 0; e < 3 * D; ++e)
                if (v[e]) ent.emplace_back(e, BigInt::from_u64(v[e]));
            Pk.append_col(ent);
        }
        Pk.cols = static_cast<uint32_t>(cols.size());
        uint32_t K_run = 0;
        const mx::Mat RB = mx::RouteBasis(c.N, c.n, Pk, u.ell, K_run);
        ENGINE_CHECK_MSG(K_run == u.K_run && RB.c == cols.size(), "XHOM: RouteBasis' width / columns");
        uint32_t c0 = 0;
        for (uint32_t o = 0; o < P; ++o, c0 += r) u.piY.push_back(PiOfRoute(c, u, RB, c0, r));
        u.piB = PiOfRoute(c, u, RB, c0, r);
        c0 += r;
        u.piCob = PiOfRoute(c, u, RB, c0, 2);
        c0 += 2;
        u.piG = PiOfRoute(c, u, RB, c0, u.nu);
        c0 += u.nu;
        for (uint32_t i = 0; i < u.nu; ++i, ++c0) u.piYG.push_back(PiOfRoute(c, u, RB, c0, 1));
        u.piYW = PiOfRoute(c, u, RB, c0, r);
        c0 += r;
        if (u.nu > 0) {
            for (uint32_t o = 0; o < P; ++o, c0 += u.nu) u.piTG.push_back(PiOfRoute(c, u, RB, c0, u.nu));
            u.piTGG = PiOfRoute(c, u, RB, c0, u.nu);
            c0 += u.nu;
        }
        ENGINE_CHECK_MSG(c0 == RB.c, "XHOM: the RouteBasis columns are all read");
        // pi of the export's own torsion generators (sanity, audit N9)
        const mx::ModuleExport& E = *c.Z.local[u.l].E;
        mx::Mat CC(E.cocycles.r, u.t);
        for (uint32_t x = 0; x < E.cocycles.r; ++x)
            for (uint32_t a = 0; a < u.t; ++a) CC.at(x, a) = E.cocycles.at(x, a);
        u.piCoc = PiOfRoute(c, u, CC, 0, u.t);
        uint64_t Qs = 1;
        for (uint32_t i = 0; i < c.Z.local[u.l].K; ++i) Qs *= u.ell;
        for (uint32_t o = 0; o < P; ++o) {
            Vec F(size_t(r) * r);
            for (size_t e = 0; e < F.size(); ++e) F[e] = c.Z.global.F[o][e].mod_u32(static_cast<uint32_t>(Qs));
            u.Fd.push_back(std::move(F));
        }
    }
    c.t_oracle = since(t0);
    c.prepared = true;
    std::cout << std::fixed << std::setprecision(3) << "XHOM cell=" << c.tag() << " D=" << D << " r=" << r << " t=" << c.Z.torsion_rank() << " units=" << c.units.size()
              << " q1=" << c.q1 << " moduli=" << c.mods.size() << " words=" << c.hs.words << " letters=" << c.hs.letters << " evals=" << c.hs.evals
              << " memo_hits=" << c.hs.memo_hits << " t_export=" << c.t_export << " t_oracle=" << c.t_oracle << " hwm_kb=" << VmHwmKb() << std::endl;
}

// (M): piY == M + pi_B F (row a mod d_a), F exact reduced
bool OracleHolds(const Cell& c, const Unit& u, size_t o, const Vec& piY, const Vec* Mover = nullptr) {
    const mx::LocalPartZ& P = c.Z.local[u.l];
    const uint32_t r = c.r;
    const Vec& M = Mover ? *Mover : P.M_global[o];
    for (uint32_t i = 0; i < u.t; ++i) {
        const uint64_t di = c.Z.d[u.t0 + i];
        for (uint32_t j = 0; j < r; ++j) {
            unsigned __int128 s = M[size_t(i) * r + j] % di;
            for (uint32_t k = 0; k < r; ++k) s += (unsigned __int128)(P.pi_B[size_t(i) * r + k] % di) * (u.Fd[o][size_t(k) * r + j] % di);
            if (static_cast<uint64_t>(s % di) != piY[size_t(i) * r + j] % di) return false;
        }
    }
    return true;
}
// (F): Phi_cert^T y == G F mod Q
bool FreeHolds(const Cell& c, const ModData& md, size_t o, const Vec& y) {
    const uint32_t r = c.r, rows = 3 * c.D;
    const uint64_t Q = md.Q;
    const Vec Ph = PackedMod(c.run.FB.Phi_cert, Q);
    const Vec G = TransMul(Ph, rows, r, md.Bq, r, Q), L = TransMul(Ph, rows, r, y, r, Q);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            unsigned __int128 s = 0;
            for (uint32_t k = 0; k < r; ++k) s = (s + (unsigned __int128)G[size_t(i) * r + k] * md.Fq[o * size_t(r) * r + size_t(k) * r + j]) % Q;
            if (static_cast<uint64_t>(s) != L[size_t(i) * r + j]) return false;
        }
    return true;
}
std::string ModTag(const ModData& md) { return md.ell ? std::to_string(md.ell) + "^K_run" : "q=" + std::to_string(md.Q); }

void ArmF(std::vector<Cell>& cells) {
    for (Cell& c : cells)
        for (const ModData& md : c.mods)
            for (size_t o = 0; o < c.ps.size(); ++o) {
                const std::string t = "(F) " + c.tag() + " T_" + std::to_string(c.ps[o]) + " mod " + ModTag(md);
                check(AllZero(SparseMul(c.car->d(1), md.Q, md.y[o], c.r)), t + ": d^1 y == 0 (T^x f is a crossed homomorphism)");
                check(FreeHolds(c, md, o, md.y[o]), t + ": Phi_cert^T y_j == G F_.j (the free part, T_p = [AgA] vs the proven F)");
            }
}

uint32_t g_units = 0, g_units23 = 0, g_nonzero23 = 0, g_nonzeroM23 = 0;
void ArmM(std::vector<Cell>& cells, bool pinned) {
    g_units = g_units23 = g_nonzero23 = g_nonzeroM23 = 0;
    for (Cell& c : cells)
        for (const Unit& u : c.units) {
            ++g_units;
            bool nz = false, nzM = false;
            for (size_t o = 0; o < c.ps.size(); ++o) {
                nz = nz || !AllZero(u.piY[o]);
                // (slot-27 Z3b-F, audit F6) M itself non-zero (row i mod d_i): the units that exercise the M block, not only pi_B F
                const Vec& Mo = c.Z.local[u.l].M_global[o];
                for (uint32_t i = 0; i < u.t && !nzM; ++i)
                    for (uint32_t j = 0; j < c.r && !nzM; ++j) nzM = Mo[size_t(i) * c.r + j] % c.Z.d[u.t0 + i] != 0;
                check(OracleHolds(c, u, o, u.piY[o]), "(M) " + c.tag() + " ell=" + std::to_string(u.ell) + " T_" + std::to_string(c.ps[o]) +
                                                          ": pi(R y_j) == M_.j + pi_B F_.j (rows mod d_a; T_p = [AgA] by Shapiro, no T~_Iso)");
            }
            if (u.ell <= 3) {
                ++g_units23;
                if (nz) ++g_nonzero23;
                if (nzM) ++g_nonzeroM23;
            }
        }
    if (pinned) {
        check(g_units == 28, "(M) tor_units = 28 (" + std::to_string(g_units) + ")");
        check(g_units23 == 15, "(M) units at ell in {2, 3} = 15 (" + std::to_string(g_units23) + ")");
        check(g_nonzero23 >= 4, "(M) nonzero_23 >= 4 (" + std::to_string(g_nonzero23) + "): the comparison is not vacuous at 2, 3");
    }
}

void ArmPi(std::vector<Cell>& cells) {
    for (Cell& c : cells)
        for (const Unit& u : c.units) {
            const std::string t = "(Pi) " + c.tag() + " ell=" + std::to_string(u.ell);
            check(AllZero(Column(u.piCob, u.t, 2, 0)), t + ": pi(R d^0 m1) == 0");
            check(AllZero(Column(u.piCob, u.t, 2, 1)), t + ": pi(R d^0 m2) == 0");
            bool eq = true;
            const mx::LocalPartZ& P = c.Z.local[u.l];
            for (uint32_t a = 0; a < u.t; ++a)
                for (uint32_t j = 0; j < c.r; ++j) eq = eq && u.piB[size_t(a) * c.r + j] == P.pi_B[size_t(a) * c.r + j] % c.Z.d[u.t0 + a];
            check(eq, t + ": pi(R b_i) = the exported pi_B column i, every i (audit N1)");
            bool id = true;
            for (uint32_t a = 0; a < u.t; ++a)
                for (uint32_t b = 0; b < u.t; ++b) id = id && u.piCoc[size_t(a) * u.t + b] == (a == b ? 1u % c.Z.d[u.t0 + a] : 0u);
            check(id, t + ": pi(cocycles[:, a]) = e_a (sanity: L0 L0^{-1} = 1, not evidence about classes, audit N9)");
        }
}

void ArmI(std::vector<Cell>& cells) {
    for (Cell& c : cells)
        for (const Unit& u : c.units) {
            const std::string t = "(I) " + c.tag() + " ell=" + std::to_string(u.ell);
            check(u.nu == u.t, t + ": dim ker(d^0 mod ell) = " + std::to_string(u.nu) + " = t_ell = " + std::to_string(u.t) + " (H^1[ell] = H^0(M/ell))");
            bool div = true;
            Vec S(size_t(u.t) * u.nu, 0);
            for (uint32_t a = 0; a < u.t; ++a) {
                uint64_t s1 = 1;
                for (uint32_t i = 1; i < c.Z.tor_sigma[u.t0 + a]; ++i) s1 *= u.ell;
                for (uint32_t i = 0; i < u.nu; ++i) {
                    const uint64_t v = u.piG[size_t(a) * u.nu + i];
                    div = div && v % s1 == 0;
                    S[size_t(a) * u.nu + i] = (v / s1) % u.ell;
                }
            }
            check(div, t + ": every pi(R g_i)_a is divisible by ell^{sigma_a - 1} (g_i is ell-torsion in the route)");
            const uint32_t rk = RankModEll(S, u.t, u.nu, u.ell);
            check(rk == u.t, t + ": the socle matrix (pi(R g_i)_a / ell^{sigma_a - 1}) mod ell has rank " + std::to_string(rk) + " = t_ell (R is injective on the ell-torsion)");
        }
}

void ArmW(std::vector<Cell>& cells) {
    for (Cell& c : cells) {
        const ModData& q1 = c.mods[0];
        const std::string t = "(W) " + c.tag() + " T_" + std::to_string(c.ps[0]);
        check(q1.yW != q1.y[0], t + ": the changed representatives delta_j b_j change y as a cochain (non-vacuous)");
        check(FreeHolds(c, q1, 0, q1.yW), t + ": ... and (F) still holds mod q1");
        for (const Unit& u : c.units) check(OracleHolds(c, u, 0, u.piYW), t + " ell=" + std::to_string(u.ell) + ": ... and (M) still holds");
    }
}

void ArmT(std::vector<Cell>& cells) {
    for (Cell& c : cells) {
        const uint32_t r = c.r;
        for (size_t o = 0; o < c.ps.size(); ++o)
            tooth(!FreeHolds(c, c.mods[0], o, c.mods[0].yI[o]), "(T) " + c.tag() + " T_" + std::to_string(c.ps[o]) + ": rho_n(delta^iota) fails (F) mod q1");
        for (const Unit& u : c.units) {
            const std::string t = "(T) " + c.tag() + " ell=" + std::to_string(u.ell);
            const mx::LocalPartZ& P = c.Z.local[u.l];
            if (u.ell <= 3) {
                Vec M1 = P.M_global[0];
                M1[0] = (M1[0] + 1) % c.Z.d[u.t0];
                tooth(!OracleHolds(c, u, 0, u.piY[0], &M1), t + ": M + 1 at (0, 0) fails (M)");
                for (uint32_t i = 0; i < u.nu; ++i) {
                    Vec piS = u.piY[0];
                    const uint32_t j = i % r;
                    for (uint32_t a = 0; a < u.t; ++a) piS[size_t(a) * r + j] = u.piYG[i][a];
                    tooth(!OracleHolds(c, u, 0, piS), t + ": y_" + std::to_string(j) + " + g_" + std::to_string(i) + " (a carrier ell-torsion cocycle through R) fails (M)");
                }
            }
            // the pre-fix block M_local C, wherever it differs from M
            for (size_t o = 0; o < c.ps.size(); ++o) {
                const Vec Ml = c.Z.M_block(o, u.l);
                Vec MC(size_t(u.t) * r, 0);
                for (uint32_t i = 0; i < u.t; ++i) {
                    const uint64_t di = c.Z.d[u.t0 + i];
                    for (uint32_t j = 0; j < r; ++j) {
                        unsigned __int128 s = 0;
                        for (uint32_t k = 0; k < r; ++k) s += (unsigned __int128)(Ml[size_t(i) * r + k] % di) * (P.C[size_t(k) * r + j] % di);
                        MC[size_t(i) * r + j] = static_cast<uint64_t>(s % di);
                    }
                }
                bool differs = false;
                for (uint32_t i = 0; i < u.t; ++i)
                    for (uint32_t j = 0; j < r; ++j) differs = differs || MC[size_t(i) * r + j] != P.M_global[o][size_t(i) * r + j] % c.Z.d[u.t0 + i];
                if (differs) tooth(!OracleHolds(c, u, o, u.piY[o], &MC), t + " T_" + std::to_string(c.ps[o]) + ": the pre-fix block M_local C fails (M)");
            }
        }
    }
}

// ---- (Tau) the slot-27 Z3b-F follow-up (design Q4): pi(R T^x g_i) == tau pi(R g_i) on the ell-socle --------------------------------
// g_i (of (I)) are exact integral carrier cocycles spanning H^1[ell]; T^x g_i is computed by eq:barcorrected via Shapiro (no T~_Iso), so
// the check is an exact test of tau's action on the socle with no precision argument.  Teeth: tau + 1 at every diagonal entry (a, a) of
// the ell-block (caught because row a of the socle matrix of (I) is non-zero mod ell: rank t), and T^x g_i + g_i pushed through R.
uint32_t g_tau_nonzero = 0;
// tau|_ell (t x t, row a mod d_a) applied to the socle columns piG; == piTG (t x nu)?  tau_ell: the ell-block, optional override
bool TauHolds(const Cell& c, const Unit& u, const Vec& tauL, const Vec& piTG) {
    for (uint32_t a = 0; a < u.t; ++a) {
        const uint64_t da = c.Z.d[u.t0 + a];
        for (uint32_t i = 0; i < u.nu; ++i) {
            unsigned __int128 s = 0;
            for (uint32_t b = 0; b < u.t; ++b) s += (unsigned __int128)(tauL[size_t(a) * u.t + b] % da) * (u.piG[size_t(b) * u.nu + i] % da);
            if (static_cast<uint64_t>(s % da) != piTG[size_t(a) * u.nu + i] % da) return false;
        }
    }
    return true;
}
Vec TauBlock(const Cell& c, const Unit& u, size_t o) {
    const uint32_t T = c.Z.torsion_rank();
    ENGINE_CHECK_MSG(c.Z.tau.size() == c.ps.size() && c.Z.tau[o].size() == size_t(T) * T, "XHOM: tau's shape");
    Vec L(size_t(u.t) * u.t);
    for (uint32_t a = 0; a < u.t; ++a)
        for (uint32_t b = 0; b < u.t; ++b) L[size_t(a) * u.t + b] = c.Z.tau[o][size_t(u.t0 + a) * T + u.t0 + b];
    return L;
}
void ArmTau(std::vector<Cell>& cells) {
    g_tau_nonzero = 0;
    for (Cell& c : cells)
        for (const Unit& u : c.units) {
            const std::string t = "(Tau) " + c.tag() + " ell=" + std::to_string(u.ell);
            if (!check(u.nu == u.t && u.piTG.size() == c.ps.size(), t + ": the socle basis g_1 .. g_t and T^x g_i per op are present")) continue;
            bool nz = false;
            for (size_t o = 0; o < c.ps.size(); ++o) {
                const Vec L = TauBlock(c, u, o);
                nz = nz || !AllZero(u.piTG[o]);
                check(TauHolds(c, u, L, u.piTG[o]), t + " T_" + std::to_string(c.ps[o]) +
                                                        ": pi(R T^x g_i) == tau pi(R g_i) on the ell-socle, every i (T_p = [AgA] by Shapiro, no T~_Iso)");
                for (uint32_t a = 0; a < u.t; ++a) {
                    Vec L1 = L;
                    L1[size_t(a) * u.t + a] = (L1[size_t(a) * u.t + a] + 1) % c.Z.d[u.t0 + a];
                    tooth(!TauHolds(c, u, L1, u.piTG[o]), t + " T_" + std::to_string(c.ps[o]) + ": tau + 1 at (" + std::to_string(a) + ", " +
                                                              std::to_string(a) + ") fails (Tau)");
                }
            }
            tooth(!TauHolds(c, u, TauBlock(c, u, 0), u.piTGG), t + " T_" + std::to_string(c.ps[0]) + ": T^x g_i + g_i through R fails (Tau)");
            if (nz) ++g_tau_nonzero;
        }
}

struct ArmDef {
    const char* name;
    std::function<void()> fn;
    uint64_t floor;
};

std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>> ParseCells(const char* s) {
    std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        std::stringstream is(item);
        std::string a, b, c;
        ENGINE_CHECK_MSG(std::getline(is, a, ':') && std::getline(is, b, ':') && std::getline(is, c), "HECKE_XHOM_CELLS: N:n:p1/p2,...");
        std::vector<uint32_t> ps;
        std::stringstream pp(c);
        std::string p;
        while (std::getline(pp, p, '/')) ps.push_back(static_cast<uint32_t>(std::stoul(p)));
        const uint32_t N = static_cast<uint32_t>(std::stoul(a)), n = static_cast<uint32_t>(std::stoul(b));
        ENGINE_CHECK_MSG(N >= 1 && n % 2 == 0 && !ps.empty(), "HECKE_XHOM_CELLS: N >= 1, n even, ops given");
        for (uint32_t q : ps) {
            bool prime = q >= 2;
            for (uint32_t d = 2; prime && static_cast<uint64_t>(d) * d <= q; ++d) prime = q % d != 0;
            if (!prime) throw std::logic_error("HECKE_XHOM_CELLS: every op p must be a prime (got p = " + std::to_string(q) + ")");
            if (N % q == 0)
                throw std::logic_error("HECKE_XHOM_CELLS: every op p must satisfy p !| N (got p = " + std::to_string(q) + ", N = " + std::to_string(N) + ")");
        }
        out.push_back({{N, n}, ps});
    }
    return out;
}

uint32_t Run() {
    const auto t0 = clk::now();
    // reset every global, so a second in-process invocation never prints the previous run's counts (audit F7)
    g_fail = 0;
    g_checks = g_teeth = g_caught = 0;
    g_units = g_units23 = g_nonzero23 = g_nonzeroM23 = 0;
    g_tau_nonzero = 0;
    const char* env = std::getenv("HECKE_XHOM_CELLS");
    const bool pinned = !(env && *env);
    std::vector<Cell> cells;
    std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>> list;
    if (pinned) {
        list = {{{1, 10}, {2, 3}}, {{2, 6}, {3, 5}}, {{9, 8}, {2, 5}}, {{11, 2}, {2, 3}},
                {{1, 30}, {2, 3}}, {{5, 12}, {2, 3}}, {{1, 60}, {2, 3}}, {{4, 12}, {3, 5}}};
    } else {
        // the EFFECTIVE line is printed on a refused hook too (audit F2)
        try {
            list = ParseCells(env);
        } catch (const std::exception& e) {
            std::cout << "EFFECTIVE HeckeZCrossedHomOracleValidation arm=cells cells=0 refused (HECKE_XHOM_CELLS=" << env << ")" << std::endl;
            std::cout << "  [EXCEPTION] HECKE_XHOM_CELLS refused: " << e.what() << std::endl;
            std::cout << "HECKEZ_XHOM_ORACLE arms=0 failing=1 verdict=FAIL" << std::endl;
            return 1;
        }
        std::cout << "EFFECTIVE HeckeZCrossedHomOracleValidation arm=cells cells=" << list.size() << " (HECKE_XHOM_CELLS=" << env << ")" << std::endl;
    }
    try {
        for (const auto& [Nn, ps] : list) {
            Cell c;
            c.N = Nn.first;
            c.n = Nn.second;
            c.ps = ps;
            BuildCell(c);
            cells.push_back(std::move(c));
        }
    } catch (const std::exception& e) {
        std::cout << "  [EXCEPTION] building the cells: " << e.what() << std::endl;
        std::cout << "HECKEZ_XHOM_ORACLE arms=0 failing=1 verdict=FAIL" << std::endl;
        return 1;
    }
    std::cout << "  cells built (exports) in " << std::fixed << std::setprecision(2) << since(t0) << " s" << std::endl;
    bool k_ok = true, prep_ok = true;
    const std::vector<ArmDef> arms = {
        {"K", [&] { ArmK(cells); }, 96},
        {"F", [&] { ArmF(cells); }, 120},
        {"M", [&] { ArmM(cells, pinned); }, 56},
        {"Pi", [&] { ArmPi(cells); }, 112},
        {"I", [&] { ArmI(cells); }, 84},
        {"W", [&] { ArmW(cells); }, 40},
        {"T", [&] { ArmT(cells); }, 40},
        {"Tau", [&] { ArmTau(cells); }, 200},
    };
    uint32_t failing = 0;
    for (const auto& [name, fn, floor0] : arms) {
        const uint64_t floor = pinned ? floor0 : 0;
        g_fail = 0;
        const uint64_t c0 = g_checks, teeth0 = g_teeth, caught0 = g_caught;
        const auto ta = clk::now();
        const bool first = std::string(name) == "K";
        if (!first && !(k_ok && prep_ok)) {
            ++g_fail;
            std::cout << "  [FAIL] arm (" << name << "): not run (" << (k_ok ? "the oracle's preparation threw" : "(K) failed: no Hecke step") << ")" << std::endl;
        } else {
            try {
                fn();
            } catch (const std::exception& e) {
                ++g_fail;
                std::cout << "  [EXCEPTION] arm (" << name << "): " << e.what() << std::endl;
            }
            if (first) {
                k_ok = g_fail == 0;
                if (k_ok) {
                    try {
                        for (Cell& c : cells) Prepare(c);
                    } catch (const std::exception& e) {
                        prep_ok = false;
                        std::cout << "  [EXCEPTION] the oracle's preparation: " << e.what() << std::endl;
                    }
                }
            }
        }
        const uint64_t nc = g_checks - c0;
        if (nc < floor) {
            ++g_fail;
            std::cout << "  [FAIL] arm (" << name << "): " << nc << " checks, below the floor " << floor << std::endl;
        }
        if (g_fail) ++failing;
        // (slot-27 Z3b-F, audit F7) the always-zero skipped= field is dropped; caught= is per arm
        std::cout << "ARM (" << name << ") checks=" << nc << " floor=" << floor << " failed=" << g_fail
                  << (g_teeth > teeth0 ? " caught=" + std::to_string(g_caught - caught0) + "/" + std::to_string(g_teeth - teeth0) : std::string())
                  << " seconds=" << std::fixed << std::setprecision(2) << since(ta) << std::endl;
    }
    const bool ok = failing == 0;
    std::cout << "HECKEZ_XHOM_ORACLE arms=" << arms.size() << " failing=" << failing << " cells=" << cells.size() << " checks=" << g_checks
              << " tor_units=" << g_units << " units_23=" << g_units23 << " nonzero_23=" << g_nonzero23 << " nonzero_M_23=" << g_nonzeroM23
              << " tau_nonzero=" << g_tau_nonzero << " teeth=" << g_caught << "/" << g_teeth
              << " hecke=barcorrected/shapiro verdict=" << (ok ? "OK" : "FAIL") << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
    return ok ? 0 : 1;
}

}  // namespace

uint32_t HeckeZCrossedHomOracleValidation() {
    std::cout << "HeckeZCrossedHomOracleValidation: an M oracle at ell = 2, 3 without T~_Iso -- crossed homomorphisms, [C, eq:barcorrected] via Shapiro "
                 "(slot-26 Z3b)"
              << std::endl;
    return Run();
}
