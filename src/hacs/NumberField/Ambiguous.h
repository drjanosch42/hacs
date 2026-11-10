// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Ambiguous.h -- Chevalley's ambiguous class number formula for L = F(N^{1/p}) / F,
// F = Q(zeta_p), N prime = 1 mod p, p regular (NF-PRANK, slot 22; NF-PRANK_spec.md S1-S4,
// sec. 4.2). STL only. The C++ twin of the prototype docs/implementation/slot22/nfprank/chevalley.py
// (frozen oracle chevalley_cells.tsv, 1240 cells):
//   t      = (p - 1) + [N^{p-1} != 1 mod p^2]     (the primes over N, plus lambda when ramified)
//   r*     = F_p-rank of the tame symbols (u, N)_{v_j}, u in {zeta, xi_2 .. xi_{(p-1)/2}},
//            v_j : zeta -> r^j, r of order p in F_N^x (the units span E_F / E_F^p for p
//            regular, Washington GTM 83 Thm 8.2)
//   chev   = t - 1 - r*   (|Cl(L)^G|_p = p^chev)
//   nonnorm_chars: the Delta-eigencharacters omega^i of E_F / E_F^p that are not norms
//            (1 for zeta_p, 2k for the cyclotomic unit components); their number must equal r*
//            (the equivariant check of sec. 4.2).
#include <cstdint>
#include <string>
#include <vector>

namespace nf {

// p regular <=> p does not divide B_2, B_4, ..., B_{p-3} (Kummer.h's ell_divides_bernoulli).
bool p_is_regular(uint32_t p);

struct ChevalleyData {
    uint32_t N = 0, p = 0;
    uint32_t t = 0;
    bool lambda_ramified = false;
    uint32_t rstar = 0;
    uint32_t chev_exponent = 0;
    std::vector<uint32_t> nonnorm_chars;    // i of omega^i
    bool product_formula_ok = true;         // lambda unramified: symbol sums vanish
    bool equivariant_ok = true;             // #nonnorm_chars == rstar
    std::vector<std::vector<uint32_t>> rows;
    std::string refusal;                    // e.g. "p divides h_F (p irregular)"
};
ChevalleyData chevalley_data(uint32_t N, uint32_t p);

}  // namespace nf
