// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZ package EXPORT (slot 25): H^1(Gamma_0(N); Sym^n Z^2) (x) Z/ell^K with its T_p as one object.  See DescentModuleExport.h.
#include "DescentModuleExport.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "DescentAssembly.h"
#include "DescentLevelN.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNHeilbronn.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {
namespace modexport {

namespace {
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
uint64_t PowU64(uint32_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) {
        ENGINE_CHECK_MSG(x <= (uint64_t(1) << 32), "modexport: ell^e overflow");
        x *= ell;
    }
    return x;
}
// the largest K with ell^K < 2^31 (the word bound of maps::ZMod; DescentHeckeTorsion's MaxK)
uint32_t MaxK(uint32_t ell) {
    uint32_t K = 0;
    uint64_t x = 1;
    while (x * ell < (uint64_t(1) << 31)) {
        x *= ell;
        ++K;
    }
    return K;
}
// an operator applied in column chunks of at most `maxk` (DescentHeckeTorsion's Chunked)
hecke3::BatchOp Chunked(hecke3::BatchOp op, uint32_t rows, uint32_t maxk) {
    return [op, rows, maxk](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
        if (k <= maxk) return op(X, k, Y);
        Y.assign(size_t(rows) * k, 0);
        std::vector<uint32_t> Xc, Yc;
        for (uint32_t c0 = 0; c0 < k; c0 += maxk) {
            const uint32_t kc = std::min(maxk, k - c0);
            Xc.assign(size_t(rows) * kc, 0);
            for (uint32_t r = 0; r < rows; ++r)
                std::copy(X.begin() + size_t(r) * k + c0, X.begin() + size_t(r) * k + c0 + kc, Xc.begin() + size_t(r) * kc);
            op(Xc, kc, Yc);
            for (uint32_t r = 0; r < rows; ++r)
                std::copy(Yc.begin() + size_t(r) * kc, Yc.begin() + size_t(r + 1) * kc, Y.begin() + size_t(r) * k + c0);
        }
    };
}
void Refuse(ModuleExport& E, ExportRefusal c, const std::string& why) {
    E.ok = false;
    E.code = c;
    E.refusal = std::string("REFUSED [") + ExportRefusalName(c) + "] " + why;
}
ExportRefusal FromD3(hecke3::D3Refusal r) {
    switch (r) {
        case hecke3::D3Refusal::IntegerCap: return ExportRefusal::IntegerCap;
        case hecke3::D3Refusal::DomainCap: return ExportRefusal::DomainCap;
        case hecke3::D3Refusal::RssCeiling: return ExportRefusal::RssCeiling;
        default: return ExportRefusal::Lift;
    }
}
// sum_r w[r] y[r * k + t] over Z/ell^K for every column t (acc mod m; ell = 2: 64-bit wrap is exact mod 2^K, masked)
void RowTimes(const ZMod& Z, const uint32_t* w, uint32_t c, const std::vector<uint32_t>& Y, uint32_t k, std::vector<uint64_t>& acc) {
    acc.assign(k, 0);
    for (uint32_t r = 0; r < c; ++r) {
        const uint32_t f = w[r];
        if (!f) continue;
        const uint32_t* yr = Y.data() + size_t(r) * k;
        if (Z.two) {
            for (uint32_t t = 0; t < k; ++t) acc[t] += uint64_t(f) * yr[t];
        } else {
            for (uint32_t t = 0; t < k; ++t)
                if (yr[t]) acc[t] = (acc[t] + uint64_t(f) * yr[t]) % Z.m;
        }
    }
    if (Z.two)
        for (auto& a : acc) a &= Z.mask;
}
}  // namespace

uint32_t RouteMaxK(uint32_t ell) { return MaxK(ell); }

const char* ExportRefusalName(ExportRefusal r) {
    switch (r) {
        case ExportRefusal::None: return "None";
        case ExportRefusal::BadEll: return "BadEll";
        case ExportRefusal::NotPrime: return "NotPrime";
        case ExportRefusal::PDividesN: return "PDividesN";
        case ExportRefusal::Lift: return "Lift";
        case ExportRefusal::RssCeiling: return "RssCeiling";
        case ExportRefusal::GroupUncertified: return "GroupUncertified";
        case ExportRefusal::KernelUncertified: return "KernelUncertified";
        case ExportRefusal::ModulePrecision: return "ModulePrecision";
        case ExportRefusal::NotEndomorphism: return "NotEndomorphism";
        case ExportRefusal::NotCocycle: return "NotCocycle";
        case ExportRefusal::IntegerCap: return "IntegerCap";
        case ExportRefusal::DomainCap: return "DomainCap";
    }
    return "?";
}

// =================================================================================================================================
// the module at K
// =================================================================================================================================
std::vector<uint32_t> ModuleAtK::invariant_factors() const {
    std::vector<uint32_t> e = exps;
    std::sort(e.rbegin(), e.rend());
    return e;
}

Mat ModuleAtK::free_block(size_t o) const {
    ENGINE_CHECK_MSG(o < ops.size(), "ModuleAtK::free_block: operator index");
    const uint32_t r = rank(), t0 = torsion_count, f = free_rank;
    Mat B(f, f);
    for (uint32_t a = 0; a < f; ++a)
        for (uint32_t b = 0; b < f; ++b) B.at(a, b) = static_cast<uint32_t>(ops[o][size_t(t0 + a) * r + (t0 + b)]);
    return B;
}

std::vector<uint32_t> ModuleAtK::free_charpoly(size_t o) const {
    if (free_rank == 0) return {1};
    return hecke3::CharpolyModPrimePower(free_block(o), ell, K);
}

ModuleAtK ModuleExport::at(uint32_t K) const {
    ENGINE_CHECK_MSG(ok, "ModuleExport::at: the export was refused");
    ENGINE_CHECK_MSG(K >= 1 && K <= K_max, "ModuleExport::at: 1 <= K <= K_max");
    ModuleAtK M;
    M.ell = ell;
    M.K = K;
    M.ps = ps;
    M.torsion_count = static_cast<uint32_t>(sigma.size());
    M.free_rank = free_rank;
    for (uint32_t s : sigma) M.exps.push_back(std::min(s, K));
    for (uint32_t t = 0; t < free_rank; ++t) M.exps.push_back(K);
    const uint32_t r = M.rank();
    for (const Mat& A : T) {
        ENGINE_CHECK(A.r == r && A.c == r);
        std::vector<uint64_t> op(size_t(r) * r);
        for (uint32_t i = 0; i < r; ++i) {
            const uint64_t mod = PowU64(ell, M.exps[i]);
            for (uint32_t j = 0; j < r; ++j) op[size_t(i) * r + j] = A.at(i, j) % mod;
        }
        M.ops.push_back(std::move(op));
    }
    return M;
}

// =================================================================================================================================
// the read
// =================================================================================================================================
void ReadModule(ModuleExport& E, const ZMod& Z, const Mat& d0, uint32_t rank_Q0, const Mat* d1, uint32_t rank_Q1, uint32_t b1,
                const std::vector<hecke3::BatchOp>& ops, bool keep_cocycles, const Mat* basis) {
    E.K_run = Z.k;
    E.rank_Q0 = rank_Q0;
    E.rank_Q1 = d1 ? rank_Q1 : 0;
    E.b1 = b1;
    const uint32_t c = d0.r;
    E.cochain_dim = c;
    // (1) d^0
    const maps::Smith S0 = hecke3::SmithLevelsLeft(Z, d0);
    const uint32_t s0 = S0.rank();
    if (s0 != rank_Q0) {
        Refuse(E, ExportRefusal::GroupUncertified, "d^0 mod " + std::to_string(Z.ell) + "^" + std::to_string(Z.k) + ": r_K = " +
                                                       std::to_string(s0) + " != rank_Q = " + std::to_string(rank_Q0) + " (prop:descentmaps(2)(g))");
        return;
    }
    std::vector<uint32_t> J;
    E.sigma.clear();
    E.sigma_max = 0;
    for (uint32_t t = 0; t < s0; ++t)
        if (S0.sigma[t] >= 1) {
            J.push_back(t);
            E.sigma.push_back(S0.sigma[t]);
            E.sigma_max = std::max(E.sigma_max, S0.sigma[t]);
        }
    const uint32_t nj = static_cast<uint32_t>(J.size());
    const uint32_t f = c - s0;
    if (keep_cocycles) {   // (slot-25 EXPORT-Z) the retraction: rows J of L0, row a mod ell^{sigma_a}
        E.retraction = Mat(nj, c);
        for (uint32_t a = 0; a < nj; ++a) {
            const uint64_t mod = PowU64(Z.ell, E.sigma[a]);
            for (uint32_t r = 0; r < c; ++r) E.retraction.at(a, r) = static_cast<uint32_t>(S0.L.at(J[a], r) % mod);
        }
    }
    // (2) the kernel of d^1 on F (columns s0 .. c-1 of L0^{-1}): Kb (f x b) its basis coordinates
    maps::Smith S1;
    uint32_t s1 = 0;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> nx;   // the sparse rows of d^1
    if (d1) {
        ENGINE_CHECK_MSG(d1->c == c, "modexport::ReadModule: d^1 must start at C^1");
        nx.resize(d1->r);
        for (uint32_t i = 0; i < d1->r; ++i)
            for (uint32_t j = 0; j < c; ++j)
                if (const uint32_t v = Z.red(d1->at(i, j))) nx[i].push_back({j, v});
        Mat MT(f, d1->r);   // (d^1 G)^T: entry (t, i) = sum_r d^1(i, r) L0^{-1}(r, s0 + t)
        for (uint32_t i = 0; i < d1->r; ++i)
            for (uint32_t t = 0; t < f; ++t) {
                const uint32_t* g = S0.LinvT.row(s0 + t);
                uint64_t s = 0;
                if (Z.two) {
                    for (const auto& [j, v] : nx[i]) s += uint64_t(v) * g[j];
                } else {
                    for (const auto& [j, v] : nx[i]) s = (s + uint64_t(v) * g[j]) % Z.m;
                }
                MT.at(t, i) = Z.red(s);
            }
        S1 = hecke3::SmithLevelsLeft(Z, MT);
        s1 = S1.rank();
        if (s1 != rank_Q1) {
            Refuse(E, ExportRefusal::KernelUncertified, "d^1 on the free part of coker d^0 mod " + std::to_string(Z.ell) + "^" +
                                                            std::to_string(Z.k) + ": r_K = " + std::to_string(s1) + " != rank_Q d^1 = " +
                                                            std::to_string(rank_Q1));
            return;
        }
        for (uint32_t t = 0; t < s1; ++t) E.sigma1_max = std::max(E.sigma1_max, S1.sigma[t]);
        if (f - s1 != b1) {
            Refuse(E, ExportRefusal::KernelUncertified, "the kernel of d^1 on F has rank " + std::to_string(f - s1) + " != b_1 = " +
                                                            std::to_string(b1));
            return;
        }
    } else if (f != b1) {
        Refuse(E, ExportRefusal::GroupUncertified, "the free part of the cokernel has rank " + std::to_string(f) + " != b_1 = " + std::to_string(b1));
        return;
    }
    // precision
    const int64_t Kmax = int64_t(Z.k) - E.sigma_max - 2 * int64_t(E.sigma1_max);
    if (Kmax < 1 || Kmax < int64_t(E.sigma_max)) {
        Refuse(E, ExportRefusal::ModulePrecision, "K_run = " + std::to_string(Z.k) + ", sigma_max = " + std::to_string(E.sigma_max) +
                                                      ", sigma1_max = " + std::to_string(E.sigma1_max) +
                                                      ": K_max = K_run - sigma_max - 2 sigma1_max < max(1, sigma_max)");
        return;
    }
    E.K_max = static_cast<uint32_t>(Kmax);
    const uint64_t modKmax = PowU64(Z.ell, E.K_max);
    const ZMod ZKp(Z.ell, Z.k - E.sigma_max);
    // the generators X (c x rank): the torsion columns J of L0^{-1}, then G Kb
    const uint32_t b = b1, k = nj + b;
    E.free_rank = b;
    std::vector<uint32_t> X(size_t(c) * k, 0);
    for (uint32_t a = 0; a < nj; ++a)
        for (uint32_t r = 0; r < c; ++r) X[size_t(r) * k + a] = S0.LinvT.at(J[a], r);
    for (uint32_t q = 0; q < b; ++q) {
        // coefficient vector in F: d1 ? row (s1 + q) of L1 : the unit vector e_q
        for (uint32_t t = 0; t < f; ++t) {
            const uint32_t w = d1 ? S1.L.at(s1 + q, t) : (t == q ? 1u : 0u);
            if (!w) continue;
            const uint32_t* g = S0.LinvT.row(s0 + t);
            for (uint32_t r = 0; r < c; ++r)
                if (g[r]) X[size_t(r) * k + nj + q] = Z.add(X[size_t(r) * k + nj + q], Z.mul(w, g[r]));
        }
    }
    // the generators are cocycles (the tree only; on the cokernel presentation H^1 = coker d^0 has no d^1; audit F5): a free
    // generator G L1^T e_{s1+q} has d^1 x = 0 mod ell^{K_run} exactly (rows >= s1 of L1 (d^1 G)^T = Sigma1 R1^{-1} vanish); a
    // torsion generator g_a (column J_a of L0^{-1}) has ell^{sigma_a} g_a = d^0 R0 e_{J_a}, so ell^{sigma_a} d^1 g_a = 0 mod
    // ell^{K_run}: d^1 g_a = 0 mod ell^{K_run - sigma_a}, checked mod ell^{K_run - sigma_max}.  Both moduli are multiples of
    // ell^{K_max}: every served cocycle (mod ell^K, K <= K_max) is a cocycle mod ell^K.
    if (d1) {
        const uint64_t modTors = PowU64(Z.ell, Z.k - E.sigma_max), modFree = PowU64(Z.ell, Z.k);
        for (uint32_t i = 0; i < nx.size(); ++i)
            for (uint32_t j = 0; j < k; ++j) {
                uint64_t s = 0;
                for (const auto& [col, v] : nx[i]) s = (s + uint64_t(v) * X[size_t(col) * k + j]) % Z.m;
                if (s % (j < nj ? modTors : modFree) != 0) {
                    Refuse(E, ExportRefusal::NotCocycle, "d^1 g_" + std::to_string(j) + " != 0 mod ell^" +
                                                         std::to_string(j < nj ? Z.k - E.sigma_max : Z.k) + " (row " + std::to_string(i) +
                                                         "): the generator is not a cocycle");
                    return;
                }
            }
    }
    if (keep_cocycles) {
        E.cocycles = Mat(c, k);
        E.cocycles.a = X;
    }
    // (slot-25 EXPORT-Z) the free coordinates (b x m, mod ell^{K_max}) of m cochains Yb (c x m): the rows s0.. of L0, then (the tree)
    // L1^{-T}, keeping the kernel rows; false (and `why`) when a column leaves the kernel of d^1 mod ell^{K_max}
    auto FreeCoords = [&](const std::vector<uint32_t>& Yb, uint32_t m, Mat& out, std::string& why) -> bool {
        std::vector<uint64_t> Ub(size_t(f) * m, 0), ac;
        for (uint32_t t = 0; t < f; ++t) {
            RowTimes(Z, S0.L.row(s0 + t), c, Yb, m, ac);
            for (uint32_t j = 0; j < m; ++j) Ub[size_t(t) * m + j] = ac[j];
        }
        out = Mat(b, m);
        for (uint32_t j = 0; j < m; ++j)
            for (uint32_t t = 0; t < f; ++t) {
                uint64_t y;
                if (d1) {
                    const uint32_t* li = S1.LinvT.row(t);
                    uint64_t s = 0;
                    for (uint32_t r = 0; r < f; ++r) {
                        if (!li[r]) continue;
                        if (Z.two) s += uint64_t(li[r]) * Ub[size_t(r) * m + j];
                        else s = (s + uint64_t(li[r]) * Ub[size_t(r) * m + j]) % Z.m;
                    }
                    y = Z.red(s);
                    if (t < s1) {
                        if (y % modKmax != 0) {
                            why = "column " + std::to_string(j) + " leaves the kernel of d^1 mod ell^{K_max} (not a cocycle)";
                            return false;
                        }
                        continue;
                    }
                } else {
                    y = Ub[size_t(t) * m + j];
                }
                out.at(d1 ? t - s1 : t, j) = static_cast<uint32_t>(y % modKmax);
            }
        return true;
    };
    std::vector<uint32_t> XB;
    Mat CBinv;   // (basis_coords)^{-1} mod ell^{K_max}
    E.basis_given = basis != nullptr;
    E.basis_ok = false;
    E.F_basis.clear();
    if (basis) {
        const ZMod ZK(Z.ell, E.K_max);
        if (basis->r != c || basis->c != b) {
            E.basis_why = "B must be cochain_dim x b_1 = " + std::to_string(c) + " x " + std::to_string(b) + ", got " + std::to_string(basis->r) +
                          " x " + std::to_string(basis->c);
        } else {
            XB.assign(size_t(c) * b, 0);
            for (uint32_t r = 0; r < c; ++r)
                for (uint32_t j = 0; j < b; ++j) XB[size_t(r) * b + j] = Z.red(basis->at(r, j));
            if (FreeCoords(XB, b, E.basis_coords, E.basis_why)) {
                // Gauss-Jordan over Z/ell^{K_max} with unit pivots: B is a basis of L / ell^{K_max} iff its coordinate matrix is a unit
                Mat A = E.basis_coords;
                CBinv = Mat(b, b);
                for (uint32_t i = 0; i < b; ++i) CBinv.at(i, i) = ZK.red(1);
                bool unit = true;
                for (uint32_t col = 0; col < b && unit; ++col) {
                    uint32_t piv = b;
                    for (uint32_t i = col; i < b; ++i)
                        if (A.at(i, col) % Z.ell != 0) {
                            piv = i;
                            break;
                        }
                    if (piv == b) {
                        unit = false;
                        break;
                    }
                    for (uint32_t j = 0; j < b; ++j) {
                        std::swap(A.at(col, j), A.at(piv, j));
                        std::swap(CBinv.at(col, j), CBinv.at(piv, j));
                    }
                    const uint32_t iv = ZK.inv(A.at(col, col));
                    for (uint32_t j = 0; j < b; ++j) {
                        A.at(col, j) = ZK.mul(A.at(col, j), iv);
                        CBinv.at(col, j) = ZK.mul(CBinv.at(col, j), iv);
                    }
                    for (uint32_t i = 0; i < b; ++i) {
                        if (i == col || A.at(i, col) == 0) continue;
                        const uint32_t fct = A.at(i, col);
                        for (uint32_t j = 0; j < b; ++j) {
                            A.at(i, j) = ZK.sub(A.at(i, j), ZK.mul(fct, A.at(col, j)));
                            CBinv.at(i, j) = ZK.sub(CBinv.at(i, j), ZK.mul(fct, CBinv.at(col, j)));
                        }
                    }
                }
                if (unit) E.basis_ok = true;
                else E.basis_why = "B is not a basis of L / ell^{K_max}: its coordinate matrix is singular mod ell";
                // (EXZ2-P3 fix round 1, audit F2) pi_ell(B): the torsion coordinates of the classes [b_j] (rows J of L0, row a mod
                // ell^{sigma_a}, the retraction's rows)
                if (unit && keep_cocycles) {
                    E.pi_B = Mat(nj, b);
                    std::vector<uint64_t> ac;
                    for (uint32_t a = 0; a < nj; ++a) {
                        RowTimes(Z, S0.L.row(J[a]), c, XB, b, ac);
                        const uint64_t mod = PowU64(Z.ell, E.sigma[a]);
                        for (uint32_t j = 0; j < b; ++j) E.pi_B.at(a, j) = static_cast<uint32_t>(ac[j] % mod);
                    }
                }
            }
        }
    }
    // (3) the operators
    E.T.clear();
    std::vector<uint64_t> acc;
    for (size_t o = 0; o < ops.size(); ++o) {
        std::vector<uint32_t> Y;
        if (k > 0) ops[o](X, k, Y);
        else Y.clear();
        ENGINE_CHECK_MSG(Y.size() == size_t(c) * k, "modexport::ReadModule: the operator returned the wrong shape");
        for (auto& y : Y) y = Z.red(y);
        Mat A(k, k);
        // torsion rows: mod ell^{sigma_a}
        for (uint32_t a = 0; a < nj; ++a) {
            RowTimes(Z, S0.L.row(J[a]), c, Y, k, acc);
            const uint64_t mod = PowU64(Z.ell, E.sigma[a]);
            for (uint32_t j = 0; j < k; ++j) A.at(a, j) = static_cast<uint32_t>(acc[j] % mod);
            for (uint32_t j = 0; j < nj; ++j)
                if (A.at(a, j) * PowU64(Z.ell, E.sigma[j]) % mod != 0) {
                    Refuse(E, ExportRefusal::NotEndomorphism, "T_p (operator " + std::to_string(o) + ") does not carry the relation ell^{sigma_j} g_j "
                                                              "into the relations (entry (" + std::to_string(a) + "," + std::to_string(j) + "))");
                    return;
                }
        }
        // the F coordinates u (f x k)
        std::vector<uint64_t> U(size_t(f) * k, 0);
        for (uint32_t t = 0; t < f; ++t) {
            RowTimes(Z, S0.L.row(s0 + t), c, Y, k, acc);
            for (uint32_t j = 0; j < k; ++j) U[size_t(t) * k + j] = acc[j];
        }
        // torsion goes to torsion
        for (uint32_t t = 0; t < f; ++t)
            for (uint32_t j = 0; j < nj; ++j)
                if (ZKp.red(U[size_t(t) * k + j]) != 0) {
                    Refuse(E, ExportRefusal::NotEndomorphism, "the image of torsion generator " + std::to_string(j) + " has a non-zero free "
                                                              "coordinate mod ell^{K_run - sigma_max}");
                    return;
                }
        // free rows: the kernel coordinates y = L1^{-T} u (the cokernel route: y = u)
        for (uint32_t j = nj; j < k; ++j) {
            for (uint32_t t = 0; t < f; ++t) {
                uint64_t y;
                if (d1) {
                    const uint32_t* li = S1.LinvT.row(t);
                    uint64_t s = 0;
                    for (uint32_t r = 0; r < f; ++r) {
                        if (!li[r]) continue;
                        if (Z.two) s += uint64_t(li[r]) * U[size_t(r) * k + j];
                        else s = (s + uint64_t(li[r]) * U[size_t(r) * k + j]) % Z.m;
                    }
                    y = Z.red(s);
                } else {
                    y = U[size_t(t) * k + j];
                }
                if (d1 && t < s1) {
                    if (y % modKmax != 0) {
                        Refuse(E, ExportRefusal::NotCocycle, "the image of free generator " + std::to_string(j - nj) + " leaves the kernel of d^1 "
                                                             "mod ell^{K_max} (row " + std::to_string(t) + ")");
                        return;
                    }
                    continue;
                }
                const uint32_t row = nj + (d1 ? t - s1 : t);
                A.at(row, j) = static_cast<uint32_t>(y % modKmax);
            }
        }
        // cocycles: d^1 T g = 0 mod ell^{K_max}, every generator
        if (d1)
            for (uint32_t i = 0; i < nx.size(); ++i)
                for (uint32_t j = 0; j < k; ++j) {
                    uint64_t s = 0;
                    for (const auto& [col, v] : nx[i]) s = (s + uint64_t(v) * Y[size_t(col) * k + j]) % Z.m;
                    if (s % modKmax != 0) {
                        Refuse(E, ExportRefusal::NotCocycle, "d^1 T_p g_" + std::to_string(j) + " != 0 mod ell^{K_max} (row " + std::to_string(i) +
                                                             "): the operator is not a cochain map");
                        return;
                    }
                }
        E.T.push_back(std::move(A));
        // (slot-25 EXPORT-Z) T_p in the caller's basis: F = (coords B)^{-1} coords(T_p B) mod ell^{K_max}
        if (E.basis_ok) {
            std::vector<uint32_t> YB;
            if (b > 0) ops[o](XB, b, YB);
            ENGINE_CHECK_MSG(YB.size() == size_t(c) * b, "modexport::ReadModule: the operator returned the wrong shape (basis)");
            for (auto& y : YB) y = Z.red(y);
            Mat CT;
            std::string why;
            if (!FreeCoords(YB, b, CT, why)) {
                E.basis_ok = false;
                E.F_basis.clear();
                E.basis_why = "T_" + std::to_string(E.ps[o]) + " B: " + why;   // names p (audit F9), not the op index
            } else {
                const ZMod ZK(Z.ell, E.K_max);
                Mat Fb(b, b);
                for (uint32_t i = 0; i < b; ++i)
                    for (uint32_t j = 0; j < b; ++j) {
                        uint32_t s = 0;
                        for (uint32_t l = 0; l < b; ++l) s = ZK.add(s, ZK.mul(CBinv.at(i, l), ZK.red(CT.at(l, j))));
                        Fb.at(i, j) = s;
                    }
                E.F_basis.push_back(std::move(Fb));
                // (EXZ2-P3 fix round 1, audit F2/F4) pi_ell(T_p B): the torsion coordinates of the images of the basis, read exactly as
                // the torsion rows of T above (rows J of L0 on the route's own T_p B)
                if (keep_cocycles) {
                    Mat PT(nj, b);
                    std::vector<uint64_t> ac;
                    for (uint32_t a = 0; a < nj; ++a) {
                        RowTimes(Z, S0.L.row(J[a]), c, YB, b, ac);
                        const uint64_t mod = PowU64(Z.ell, E.sigma[a]);
                        for (uint32_t j = 0; j < b; ++j) PT.at(a, j) = static_cast<uint32_t>(ac[j] % mod);
                    }
                    E.pi_TB.push_back(std::move(PT));
                }
            }
        }
    }
    if (!E.basis_ok) {
        E.pi_B = Mat();
        E.pi_TB.clear();
    }
    E.ok = true;
}

uint64_t PredictExportRss(uint32_t ell, uint32_t N, uint32_t n) {
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    if (ell <= 3) {
        // the tree read of H^1 (its Smith form of d^0 with both left transforms) plus the transform of (d^1 | F)^T and the
        // generator block: bounded by a second tree read of H^2's size
        return hecke3::PredictTreeReadRss(ell, N, n, 1) + hecke3::PredictTreeReadRss(ell, N, n, 2) / 2;
    }
    return hecke3::PredictCokernelReadRss(N, n) + 4ull * uint64_t(cnt.D) * (cnt.b1 + 64);
}

namespace {
thread_local uint32_t t_export_threads = 1;
}  // namespace
uint32_t ExportThreads() { return t_export_threads; }
ScopedExportThreads::ScopedExportThreads(uint32_t t) : prev(t_export_threads) { t_export_threads = std::max<uint32_t>(t, 1); }
ScopedExportThreads::~ScopedExportThreads() { t_export_threads = prev; }

void ApplyHeilbronnColumns(const hecke::level::HeilbronnOperator& H, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint64_t>& acc,
                           uint32_t threads) {
    const uint64_t q = H.ctx().p;
    const uint32_t d = H.d();
    const size_t D = H.D();
    ENGINE_CHECK_MSG(q >= 2 && q < (1ull << 31), "ApplyHeilbronnColumns: a word modulus < 2^31");
    ENGINE_CHECK_MSG(X.size() == D * k && acc.size() == D * k, "ApplyHeilbronnColumns: D x k operands");
    const auto& terms = H.terms();
    const auto& blocks = H.blocks();
    // column blocks of 8 .. 64 columns, about 4 per thread (k is the module's generator count, often ~ b1: 64-wide blocks would idle)
    const uint32_t Tw = std::max<uint32_t>(threads, 1);
    const uint32_t kCols = std::min<uint32_t>(64, std::max<uint32_t>(8, (k + 4 * Tw - 1) / (4 * Tw)));
    const uint32_t nblk = (k + kCols - 1) / kCols;
    auto work_block = [&](uint32_t cb) {
        const uint32_t t0 = cb * kCols, t1 = std::min(k, t0 + kCols), w = t1 - t0;
        for (const hecke::level::HeilbronnTerm& tm : terms) {
            const std::vector<uint32_t>& B = blocks[tm.block].dense;
            const uint32_t* xs = &X[size_t(tm.src - 1) * d * k + t0];
            uint64_t* ys = &acc[size_t(tm.dst - 1) * d * k + t0];
            for (uint32_t r = 0; r < d; ++r) {
                uint64_t* yr = ys + size_t(r) * k;
                for (uint32_t c = 0; c < d; ++c) {
                    const uint64_t b = B[size_t(r) * d + c] % q;
                    if (!b) continue;
                    const uint64_t bp = (b << 32) / q;   // Shoup: x b - floor(x bp / 2^32) q lies in [0, 2q) for x < 2^32
                    const uint32_t* xr = xs + size_t(c) * k;
                    for (uint32_t t = 0; t < w; ++t) {
                        const uint64_t x = xr[t];
                        uint64_t m = x * b - ((x * bp) >> 32) * q;
                        if (m >= q) m -= q;
                        uint64_t y = yr[t] + m;
                        yr[t] = y >= q ? y - q : y;
                    }
                }
            }
        }
    };
    const uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, nblk));
    if (T == 1) {
        for (uint32_t cb = 0; cb < nblk; ++cb) work_block(cb);
        return;
    }
    std::atomic<uint32_t> next(0);
    auto work = [&]() {
        for (;;) {
            const uint32_t cb = next.fetch_add(1);
            if (cb >= nblk) break;
            work_block(cb);
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t i = 1; i < T; ++i) {
        try {
            pool.emplace_back(work);
        } catch (const std::system_error&) {
            break;
        }
    }
    work();
    for (auto& th : pool) th.join();
}

std::vector<uint32_t> HeilbronnDense(const hecke::level::Gamma0LevelN& lvl, uint64_t p, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                     const std::vector<hecke::Mat22>& mer, size_t slice, uint32_t threads) {
    const uint64_t q = ctx.p;
    const uint32_t d = n_sym + 1;
    const size_t D = size_t(lvl.m()) * d;
    ENGINE_CHECK_MSG(q >= 2 && q < (1ull << 31) && slice >= 1, "HeilbronnDense: a word modulus < 2^31, slices of >= 1 matrix");
    const size_t nsl = (mer.size() + slice - 1) / slice;
    // one accumulator per worker (the sum of the partial operators is exact mod q, so the split changes nothing); one only when
    // the copies would not fit 512 MiB
    uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, static_cast<uint32_t>(std::max<size_t>(nsl, 1))));
    while (T > 1 && uint64_t(T) * D * D * 4 > (512ull << 20)) --T;
    std::vector<std::vector<uint32_t>> part(T, std::vector<uint32_t>());
    std::atomic<size_t> next(0);
    std::vector<std::exception_ptr> errs(T);
    auto work = [&](uint32_t w) {
        try {
            std::vector<uint32_t>& Mw = part[w];
            Mw.assign(D * D, 0);
            for (;;) {
                const size_t si = next.fetch_add(1);
                if (si >= nsl) break;
                const size_t a = si * slice, b = std::min(mer.size(), a + slice);
                const hecke::level::HeilbronnOperator Hc(lvl, p, n_sym, ctx, std::vector<hecke::Mat22>(mer.begin() + a, mer.begin() + b));
                const auto& blocks = Hc.blocks();
                for (const hecke::level::HeilbronnTerm& tm : Hc.terms()) {
                    const std::vector<uint32_t>& B = blocks[tm.block].dense;
                    for (uint32_t r = 0; r < d; ++r) {
                        uint32_t* row = &Mw[(size_t(tm.dst - 1) * d + r) * D + size_t(tm.src - 1) * d];
                        for (uint32_t c = 0; c < d; ++c) {
                            const uint64_t v = uint64_t(row[c]) + B[size_t(r) * d + c] % q;
                            row[c] = static_cast<uint32_t>(v >= q ? v - q : v);
                        }
                    }
                }
            }
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) {
        try {
            pool.emplace_back(work, w);
        } catch (const std::system_error&) {
            break;
        }
    }
    work(0);
    for (auto& th : pool) th.join();
    for (auto& e : errs) if (e) std::rethrow_exception(e);
    std::vector<uint32_t> M = std::move(part[0]);
    for (uint32_t w = 1; w < T; ++w) {
        if (part[w].empty()) continue;
        for (size_t i = 0; i < M.size(); ++i) {
            const uint64_t v = uint64_t(M[i]) + part[w][i];
            M[i] = static_cast<uint32_t>(v >= q ? v - q : v);
        }
    }
    return M;
}

void DenseApplyColumns(const std::vector<uint32_t>& M, size_t D, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint64_t>& acc,
                       uint64_t q, uint32_t threads) {
    ENGINE_CHECK_MSG(q >= 2 && q < (1ull << 31), "DenseApplyColumns: a word modulus < 2^31");
    ENGINE_CHECK_MSG(M.size() == D * D && X.size() == D * k && acc.size() == D * k, "DenseApplyColumns: shapes");
    std::atomic<size_t> next(0);
    constexpr size_t kRows = 16;
    auto work = [&]() {
        for (;;) {
            const size_t r0 = next.fetch_add(kRows);
            if (r0 >= D) break;
            for (size_t r = r0; r < std::min(D, r0 + kRows); ++r) {
                uint64_t* yr = &acc[r * k];
                for (size_t c = 0; c < D; ++c) {
                    const uint64_t b = M[r * D + c];
                    if (!b) continue;
                    const uint64_t bp = (b << 32) / q;   // Shoup
                    const uint32_t* xr = &X[c * k];
                    for (uint32_t t = 0; t < k; ++t) {
                        const uint64_t x = xr[t];
                        uint64_t m = x * b - ((x * bp) >> 32) * q;
                        if (m >= q) m -= q;
                        const uint64_t y = yr[t] + m;
                        yr[t] = y >= q ? y - q : y;
                    }
                }
            }
        }
    };
    const uint32_t T = std::max<uint32_t>(1, threads);
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) {
        try {
            pool.emplace_back(work);
        } catch (const std::system_error&) {
            break;
        }
    }
    work();
    for (auto& th : pool) th.join();
}

ModuleExport ExportModule(uint32_t N, uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps, const ExportOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    ModuleExport E;
    E.N = N;
    E.n = n;
    E.ell = ell;
    E.ps = ps;
    ENGINE_CHECK_MSG(N >= 1, "ExportModule: N >= 1");
    if (ell < 2 || !IsPrime(ell) || (ell >= 5 && MaxK(ell) == 0)) {
        Refuse(E, ExportRefusal::BadEll, "ell = " + std::to_string(ell) + " (a prime with ell < 2^31)");
        return E;
    }
    for (uint32_t p : ps) {
        if (p < 2 || !IsPrime(p)) {
            Refuse(E, ExportRefusal::NotPrime, "p = " + std::to_string(p));
            return E;
        }
        if (N % p == 0 && !opt.allow_bad_primes) {   // (S26 Z1 / Z2) U_p on request, on both routes (the tree: [C, prop:strictliftN](9))
            Refuse(E, ExportRefusal::PDividesN, "p = " + std::to_string(p) + " divides N = " + std::to_string(N) +
                                                    " (U_p, p | N, is deferred: no cochain-level lift is exported)");
            return E;
        }
    }
    E.rss_pred = PredictExportRss(ell, N, n);
    if (E.rss_pred > opt.rss_cap_bytes) {
        Refuse(E, ExportRefusal::RssCeiling, "rule 19: the export predicts " + std::to_string(E.rss_pred >> 20) + " MiB above the ceiling " +
                                                 std::to_string(opt.rss_cap_bytes >> 20) + " MiB (refused before building)");
        return E;
    }
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    const uint32_t b1 = n % 2 ? 0u : cnt.b1;
    RssSampler rs;
    try {
        if (ell <= 3) {
            E.route = "tree";
            E.carrier = "C^1_T (J-coordinates, the ell-local tree model)";
            const hecke3::TreeTransport X(ell, N, n, 1, opt.K_run);
            const Mat d0 = X.differential(0), d1 = X.differential(1);
            std::vector<std::unique_ptr<hecke3::IsoHeckeN>> T;
            for (uint32_t p : ps) {
                // (S26 Z2) p | N: the free-model U_p of [C, prop:strictliftN](9), transported to the tree ((5)); route (materialised / streamed /
                // count) by size, value-neutral (HeckeZUpValidation).  p !| N: the shipped call, byte for byte.
                T.push_back(N % p == 0 ? std::make_unique<hecke3::IsoHeckeN>(N, n, p, X.ctx(), 1, false, opt.allow_bad_primes, opt.iso_route)
                                       : std::make_unique<hecke3::IsoHeckeN>(N, n, p, X.ctx(), 1));
                if (T.back()->refused()) {
                    Refuse(E, FromD3(T.back()->code()), "the cochain-level lift of T_" + std::to_string(p) + " at ell = " + std::to_string(ell) +
                                                            ": " + T.back()->why());
                    E.rss_meas = rs.stop();
                    E.seconds = since(t0);
                    return E;
                }
            }
            std::vector<hecke3::BatchOp> ops;
            for (const auto& t : T) {
                const hecke3::IsoHeckeN* tp = t.get();
                ops.push_back(Chunked([&X, tp](const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& Y) { X.apply_hecke(*tp, 1, U, k, Y, ExportThreads()); },
                                      X.rank(1), 128));
            }
            ReadModule(E, X.Z(), d0, X.rank_Q(0), &d1, X.rank_Q(1), b1, ops, opt.keep_cocycles, opt.basis);
        } else {
            E.route = "cokernel";
            E.carrier = "M_N (the cokernel presentation M_N / Rel_N, prop:heilbronn)";
            const uint32_t K = opt.K_run ? opt.K_run : MaxK(ell);
            ENGINE_CHECK_MSG(K >= 1 && K <= MaxK(ell), "ExportModule: K_run with ell^K_run < 2^31");
            const ZMod Z(ell, K);
            // K = 1 (every ell with ell^2 >= 2^31, i.e. ell >= 46341, or an explicit K_run = 1): the field context F_ell
            // (ModCtx::prime_power demands k >= 2; audit F1)
            const engine::ModCtx<uint32_t> ctx = K == 1 ? engine::ModCtx<uint32_t>(ell) : engine::ModCtx<uint32_t>::prime_power(ell, K);
            const uint32_t D = cnt.D;
            Mat A(D, 0);
            if (n % 2 == 0) {
                const engine::DenseBuf<uint32_t> G = FixedGeneratorRows(N, n, ctx);   // generators x D (rows generating M^S + M^ST)
                ENGINE_CHECK(G.cols == D);
                A = Mat(D, G.rows);
                for (uint32_t i = 0; i < G.rows; ++i)
                    for (uint32_t j = 0; j < D; ++j) A.at(j, i) = G.at(i, j);
            } else {
                // n odd: -I acts by -1, so Rel_N = M_N over Z_(ell) (2 is a unit): H^1 = 0; the identity presents it
                A = Mat(D, D);
                for (uint32_t i = 0; i < D; ++i) A.at(i, i) = 1;
            }
            const hecke::level::Gamma0LevelN lvl(N);
            std::vector<std::unique_ptr<hecke::level::HeilbronnOperator>> H;
            std::vector<hecke3::BatchOp> ops;
            if (n % 2 == 1)   // nothing to apply to (no generator); one (never called) operator per p keeps T's shape
                for (size_t i = 0; i < ps.size(); ++i)
                    ops.push_back([](const std::vector<uint32_t>&, uint32_t, std::vector<uint32_t>& Y) { Y.clear(); });
            else
                for (uint32_t p : ps) {
                    // (slot-25 HZ-RAT-PERF (c)) eq:heilbronn has |Mer_p| m_N terms of about kTermBytes each (the term, its rho_n block,
                    // the memo node; measured 1 - 4.3 kB per unit of ell psi(N)(n + 1) at the T_ell read).  Three value-neutral routes,
                    // the same operator entrywise (exact residues mod ell^K):
                    //  * dense: when |Mer_p| > m_N (applying the terms costs |Mer_p| m_N d^2 per column, the D x D matrix D^2 = m_N^2 d^2)
                    //    and D^2 words fit kDenseBudget: the D x D matrix accumulated slice by slice of Mer_p (partial operators, never
                    //    the whole term table), then one D x D times D x k product;
                    //  * materialised: the term table within kExportTermBudget, applied to all columns at once (as before, batched);
                    //  * streamed: beyond it, slices of Mer_p applied one at a time.
                    const uint64_t kTermBytes = 448 + 12ull * (n + 1) * (n + 1);
                    constexpr uint64_t kExportTermBudget = 256ull << 20, kSliceBytes = 32ull << 20, kDenseBudget = 512ull << 20;
                    auto M = std::make_shared<const std::vector<hecke::Mat22>>(hecke::level::MerelSet(p));
                    const uint64_t per_mat = uint64_t(lvl.m()) * kTermBytes;
                    const bool fits = uint64_t(M->size()) * per_mat <= kExportTermBudget;
                    const size_t slice = fits ? std::max<size_t>(1, M->size())
                                              : static_cast<size_t>(std::max<uint64_t>(1, kSliceBytes / per_mat));
                    const bool dense = M->size() > lvl.m() && uint64_t(D) * D * 4 <= kDenseBudget;
                    if (dense) {
                        // (HZ-RAT-PERF fix F8) the D x D matrix is built once, on the op's first call, and kept for the later ones
                        // (ReadModule calls each op twice when basis_ok: X, then X B); its slices of Mer_p are cut to give every
                        // export thread work (about 4 slices per thread, never larger than the term-budget slice) -- the sum of the
                        // partial operators is exact mod ell^K, so neither the split nor the thread count changes an entry
                        auto Md = std::make_shared<std::vector<uint32_t>>();
                        ops.push_back([M, Md, slice, p, n, ctx, D, &lvl](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
                            const uint32_t T = ExportThreads();
                            if (Md->empty()) {
                                const size_t want = (M->size() + 4 * size_t(T) - 1) / (4 * size_t(T));
                                const size_t sl = T > 1 ? std::max<size_t>(1, std::min(slice, want)) : slice;
                                *Md = HeilbronnDense(lvl, p, n, ctx, *M, sl, T);
                            }
                            std::vector<uint64_t> acc(size_t(D) * k, 0);
                            DenseApplyColumns(*Md, D, X, k, acc, ctx.p, T);
                            Y.assign(acc.begin(), acc.end());
                        });
                    } else if (fits) {
                        H.push_back(std::make_unique<hecke::level::HeilbronnOperator>(lvl, p, n, ctx));
                        const hecke::level::HeilbronnOperator* hp = H.back().get();
                        ops.push_back([hp, D](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
                            std::vector<uint64_t> acc(size_t(D) * k, 0);
                            ApplyHeilbronnColumns(*hp, X, k, acc, ExportThreads());
                            Y.assign(acc.begin(), acc.end());
                        });
                    } else {
                        ops.push_back([M, slice, p, n, ctx, D, &lvl](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
                            std::vector<uint64_t> acc(size_t(D) * k, 0);
                            for (size_t a = 0; a < M->size(); a += slice) {
                                const size_t b = std::min(M->size(), a + slice);
                                const hecke::level::HeilbronnOperator Hc(lvl, p, n, ctx,
                                                                         std::vector<hecke::Mat22>(M->begin() + a, M->begin() + b));
                                ApplyHeilbronnColumns(Hc, X, k, acc, ExportThreads());
                            }
                            Y.assign(acc.begin(), acc.end());
                        });
                    }
                }
            // rank_Q of the presentation = dim Rel = D - b_1 (prop:levelorbit (2))
            ReadModule(E, Z, A, D - b1, nullptr, 0, b1, ops, opt.keep_cocycles, opt.basis);
        }
    } catch (const std::exception& e) {
        const hecke3::D3Refusal r = hecke3::ClassifyD3Exception(e);
        if (r == hecke3::D3Refusal::None) throw;   // a defect propagates
        Refuse(E, FromD3(r), e.what());
    }
    E.rss_meas = rs.stop();
    E.seconds = since(t0);
    return E;
}

}  // namespace modexport
}  // namespace integral_descent
