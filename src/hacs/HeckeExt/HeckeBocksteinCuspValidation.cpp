// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-11) HeckeBocksteinCuspValidation -- see the header.
//
// Claim (H4') of docs/implementation/slot13/WP-11_mixed_finality_proposal.md
// Sec. 3.1 (\VERIFY; referee finding m6): r^1 is injective on im beta^0.  By
// naturality of the Bockstein, r^1 o beta^0 = beta^0_bd o r^0, with r^0 the
// inclusion H^0(Gamma; M_n) = F_ell[L, D]_n (Dickson; [C, prop:heckeH0]) into
// H^0(Gamma_inf; M_n) = M_n^T, and beta^0_bd(v) = ((T - 1) v~ / ell mod ell) in
// H^1(Gamma_inf; M_n) = M_n / (T - 1) M_n, T : f(x, y) -> f(x, x + y), v~ the
// integral lift.  So everything is a computation in Sym^n(Z/ell^2), independent
// of the bulk (a C++ port of the referee's bockstein_cusp.py).
//
// Per cell (n, ell) with H^0 != 0 and per class a mod (ell - 1) of the Dickson
// monomials L^a D^b of degree n: the lifts are T-invariant mod ell; a
// Frobenius-power monomial (ell | a and ell | b, i.e. (L^{a/ell} D^{b/ell})^ell)
// has (T - 1) v~ = 0 mod ell^2, so it lies in ker beta^0 (bulk and boundary alike:
// gamma(g^ell) = (g + ell h)^ell = g^ell mod ell^2); and -- the gated claim --
//      rank(beta^0_bd o r^0 | class) = #monomials - #Frobenius powers,
// per class and over the whole cell.  Since rank(beta^0_bd o r^0) = rank(r^1
// beta^0) <= rank beta^0 <= #monomials - #Frobenius powers (the monomials are a
// basis of H^0), equality certifies at that cell that r^1 is injective on
// im beta^0 and that ker beta^0 is spanned by the Frobenius powers ([C,
// rmk:bocksteinexp](2)).  Grid: ell in {5, 7} with n <= 300 and ell in {11, 13}
// with n <= 240, n even (the referee's grid).  Pipeline-free: no cochain
// complex, no engine.  Read-only.
#include "HeckeBocksteinCuspValidation.h"
#include "../Util/ValidationCheck.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

oc_util::Checker Check("  [FAIL] ");

using Poly = std::vector<uint64_t>;   // coefficient of x^{deg - j} y^j at index j, mod ell^2

Poly Mul(const Poly& a, const Poly& b, uint64_t m) {
    Poly out(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < b.size(); ++j) out[i + j] = (out[i + j] + a[i] * b[j]) % m;
    }
    return out;
}

uint64_t PowMod(uint64_t b, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    for (b %= m; e; e >>= 1, b = b * b % m)
        if (e & 1) r = r * b % m;
    return r;
}

// Rank over F_ell of the rows (dense elimination; the matrices here are tiny).
uint32_t RankRows(std::vector<std::vector<uint64_t>> M, uint64_t ell) {
    uint32_t rk = 0;
    const size_t cols = M.empty() ? 0 : M[0].size();
    for (size_t c = 0; c < cols && rk < M.size(); ++c) {
        size_t piv = rk;
        while (piv < M.size() && M[piv][c] % ell == 0) ++piv;
        if (piv == M.size()) continue;
        std::swap(M[rk], M[piv]);
        const uint64_t inv = PowMod(M[rk][c], ell - 2, ell);
        for (auto& x : M[rk]) x = x * inv % ell;
        for (size_t r = 0; r < M.size(); ++r)
            if (r != rk && M[r][c] % ell) {
                const uint64_t f = M[r][c] % ell;
                for (size_t k = 0; k < cols; ++k) M[r][k] = (M[r][k] + (ell - f) * M[rk][k]) % ell;
            }
        ++rk;
    }
    return rk;
}

struct Totals {
    uint32_t cells = 0, classes = 0, monomials = 0, frobenius = 0, deficient_classes = 0;
};

void RunEll(uint64_t ell, uint32_t nmax, const std::map<uint32_t, uint32_t>& rank_pins) {
    const uint64_t m2 = ell * ell;
    // Binomials mod ell^2 up to nmax (Pascal).
    std::vector<std::vector<uint64_t>> C(nmax + 1);
    for (uint32_t j = 0; j <= nmax; ++j) {
        C[j].assign(j + 1, 1);
        for (uint32_t i = 1; i < j; ++i) C[j][i] = (C[j - 1][i - 1] + C[j - 1][i]) % m2;
    }
    // L = x^ell y - x y^ell; D = (x^{ell^2} y - x y^{ell^2}) / L by exact division.
    Poly L(ell + 2, 0);
    L[1] = 1;
    L[ell] = m2 - 1;
    Poly num(m2 + 2, 0);
    num[1] = 1;
    num[m2] = m2 - 1;
    Poly Dq(num.size() - L.size() + 1, 0);
    for (size_t k = 0; k < Dq.size(); ++k) {
        const uint64_t c = num[k + 1];   // L's lowest coefficient (index 1) is 1
        Dq[k] = c;
        for (size_t j = 0; j < L.size(); ++j) num[k + j] = (num[k + j] + (m2 - c) * L[j] % m2) % m2;
    }
    bool exact = true;
    for (uint64_t x : num) exact = exact && x == 0;
    Check(exact && Dq.size() == m2 - ell + 1, "ell=" + std::to_string(ell) + ": D = N / L exactly, degree ell^2 - ell");
    const uint32_t dL = static_cast<uint32_t>(ell + 1), dD = static_cast<uint32_t>(m2 - ell);
    std::vector<Poly> Lp(1, Poly{1}), Dp(1, Poly{1});
    while ((Lp.size()) * dL <= nmax) Lp.push_back(Mul(Lp.back(), L, m2));
    while ((Dp.size()) * dD <= nmax) Dp.push_back(Mul(Dp.back(), Dq, m2));

    Totals T;
    for (uint32_t n = 2; n <= nmax; n += 2) {
        std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> cls;   // a mod (ell-1) -> (a, b)
        for (uint32_t a = 0; a * dL <= n; ++a)
            if ((n - a * dL) % dD == 0) cls[a % (ell - 1)].push_back({a, (n - a * dL) / dD});
        if (cls.empty()) continue;
        ++T.cells;
        const std::string cell = "(" + std::to_string(n) + ", " + std::to_string(ell) + ")";
        // im(T - 1) mod ell, echelon keyed by the highest nonzero index, leading coefficient 1.
        std::vector<std::vector<uint64_t>> piv(n + 1);
        for (uint32_t j = 1; j <= n; ++j) {
            std::vector<uint64_t> v(n + 1, 0);
            for (uint32_t i = 0; i < j; ++i) v[i] = C[j][i] % ell;
            for (int32_t idx = static_cast<int32_t>(j) - 1; idx >= 0; --idx) {
                if (v[idx] == 0) continue;
                if (!piv[idx].empty()) {
                    const uint64_t f = v[idx];
                    for (int32_t k = 0; k <= idx; ++k) v[k] = (v[k] + (ell - f) * piv[idx][k]) % ell;
                    continue;
                }
                const uint64_t inv = PowMod(v[idx], ell - 2, ell);
                for (int32_t k = 0; k <= idx; ++k) v[k] = v[k] * inv % ell;
                v.resize(idx + 1);
                piv[idx] = v;
                break;
            }
        }
        std::vector<std::vector<uint64_t>> all_rows;
        uint32_t all_frob = 0, all_mon = 0, cell_rank_expected = 0;
        for (const auto& kv : cls) {
            std::vector<std::vector<uint64_t>> rows;
            uint32_t frob = 0;
            for (const auto& ab : kv.second) {
                const Poly v = Mul(Lp[ab.first], Dp[ab.second], m2);
                bool inv_mod_ell = v.size() == n + 1, zero_mod_ell2 = true;
                std::vector<uint64_t> img(n + 1, 0);
                for (uint32_t i = 0; i <= n && inv_mod_ell; ++i) {
                    uint64_t w = 0;
                    for (uint32_t j = i + 1; j <= n; ++j) w = (w + C[j][i] * v[j]) % m2;
                    inv_mod_ell = inv_mod_ell && w % ell == 0;
                    zero_mod_ell2 = zero_mod_ell2 && w == 0;
                    img[i] = (w / ell) % ell;
                }
                const std::string mon = cell + " L^" + std::to_string(ab.first) + " D^" + std::to_string(ab.second);
                Check(inv_mod_ell, mon + ": the lift is T-invariant mod ell");
                const bool is_frob = ab.first % ell == 0 && ab.second % ell == 0;
                if (is_frob) {
                    ++frob;
                    Check(zero_mod_ell2, mon + ": a Frobenius power has (T - 1) v~ = 0 mod ell^2 (in ker beta^0)");
                }
                // Reduce modulo im(T - 1): canonical coordinates in the coinvariants.
                for (int32_t idx = static_cast<int32_t>(n); idx >= 0; --idx)
                    if (img[idx] && !piv[idx].empty()) {
                        const uint64_t f = img[idx];
                        for (int32_t k = 0; k <= idx; ++k) img[k] = (img[k] + (ell - f) * piv[idx][k]) % ell;
                    }
                rows.push_back(img);
                all_rows.push_back(img);
            }
            const uint32_t nm = static_cast<uint32_t>(kv.second.size());
            const uint32_t rk = RankRows(rows, ell);
            ++T.classes;
            T.monomials += nm;
            T.frobenius += frob;
            all_frob += frob;
            all_mon += nm;
            cell_rank_expected += nm - frob;
            T.deficient_classes += rk != nm - frob;
            Check(rk == nm - frob, cell + " class a=" + std::to_string(kv.first) + ": rank(beta_bd o r^0) = " +
                                       std::to_string(rk) + " == #monomials - #Frobenius = " +
                                       std::to_string(nm - frob));
        }
        const uint32_t rk_all = RankRows(all_rows, ell);
        Check(rk_all == all_mon - all_frob, cell + ": rank(beta_bd o r^0) over the cell = " + std::to_string(rk_all) +
                                                " == " + std::to_string(all_mon - all_frob));
        const auto pin = rank_pins.find(n);
        if (pin != rank_pins.end())
            Check(rk_all == pin->second, cell + " (pinned): rank beta^0 = " + std::to_string(pin->second));
        (void)cell_rank_expected;
    }
    std::cout << "BOCKSTEIN_CUSP ell=" << ell << " n<=" << nmax << " dickson_cells=" << T.cells
              << " classes=" << T.classes << " monomials=" << T.monomials << " frobenius_powers=" << T.frobenius
              << " deficient_classes=" << T.deficient_classes << "\n";
}

}  // namespace

uint32_t HeckeBocksteinCuspValidation() {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeBocksteinCuspValidation (slot-13 WP-11, referee m6): r^1 injective on im beta^0, "
                 "pipeline-free over Sym^n(Z/ell^2)\n";
    // Pins: rank beta^0 at (26, 5) = 1, (30, 5) = 0 (L^5, the ell^2-torsion of [C] rmk:bocksteinexp),
    // (300, 5) = 4 (six monomials, L^50 and D^15 Frobenius powers); (56, 7) = 0 (L^7).
    RunEll(5, 300, {{26, 1}, {30, 0}, {300, 4}});
    RunEll(7, 300, {{56, 0}});
    RunEll(11, 240, {});
    RunEll(13, 240, {});
    std::cout << "HeckeBocksteinCuspValidation: " << Check.checks << " checks, " << Check.fails << " failing, "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    return Check.fails == 0 ? 0u : 1u;
}
