// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Ambiguous.cpp -- see Ambiguous.h (NF-PRANK, slot 22).
#include "Ambiguous.h"

#include <unordered_map>

#include "Kummer.h"
#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

bool p_is_regular(uint32_t p) {
    for (uint32_t k = 2; k + 3 <= p; k += 2)
        if (ell_divides_bernoulli(p, k)) return false;
    return true;
}

namespace {

uint32_t rank_mod_p(std::vector<std::vector<uint32_t>> rows, uint32_t p) {
    if (rows.empty()) return 0;
    const std::size_t nc = rows[0].size();
    uint32_t rk = 0;
    for (std::size_t col = 0; col < nc && rk < rows.size(); ++col) {
        std::size_t piv = rk;
        while (piv < rows.size() && rows[piv][col] % p == 0) ++piv;
        if (piv == rows.size()) continue;
        std::swap(rows[rk], rows[piv]);
        const uint64_t inv = invmod32(rows[rk][col] % p, p);
        for (auto& x : rows[rk]) x = static_cast<uint32_t>(x % p * inv % p);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i == rk || rows[i][col] % p == 0) continue;
            const uint64_t c = rows[i][col] % p;
            for (std::size_t j = 0; j < nc; ++j) rows[i][j] = static_cast<uint32_t>((rows[i][j] % p + p - c * rows[rk][j] % p) % p);
        }
        ++rk;
    }
    return rk;
}

}  // namespace

ChevalleyData chevalley_data(uint32_t N, uint32_t p) {
    ChevalleyData C;
    C.N = N;
    C.p = p;
    if (!is_prime_u32(p) || p < 5 || !is_prime_u32(N) || N % p != 1) {
        C.refusal = "cell outside the domain (N prime = 1 mod p, p >= 5 prime)";
        return C;
    }
    if (!p_is_regular(p)) {
        C.refusal = "p divides h_F (p irregular)";
        return C;
    }
    const uint32_t g = primitive_root_prime(N);
    const uint32_t r = powmod32(g, (N - 1) / p, N);
    std::unordered_map<uint32_t, uint32_t> dlog;
    uint64_t x = 1;
    for (uint32_t k = 0; k < p; ++k, x = x * r % N) dlog[static_cast<uint32_t>(x)] = k;
    auto sym = [&](uint64_t y) {
        const auto it = dlog.find(powmod32(y % N, (N - 1) / p, N));
        NF_CHECK(it != dlog.end());
        return it->second;
    };
    const uint64_t pp = static_cast<uint64_t>(p) * p;
    C.lambda_ramified = powmod32(N, p - 1, static_cast<uint32_t>(pp)) != 1;
    C.t = (p - 1) + (C.lambda_ramified ? 1 : 0);
    std::vector<uint32_t> row0;
    for (uint32_t j = 1; j < p; ++j) row0.push_back(sym(powmod32(r, j, N)));
    C.rows.push_back(row0);
    for (uint32_t a = 2; a <= (p - 1) / 2; ++a) {
        std::vector<uint32_t> row;
        for (uint32_t j = 1; j < p; ++j) {
            const uint64_t num = (1 + N - powmod32(r, static_cast<uint64_t>(a) * j, N)) % N;
            const uint64_t den = (1 + N - powmod32(r, j, N)) % N;
            row.push_back(sym(num * invmod32(den, N) % N));
        }
        C.rows.push_back(row);
    }
    if (!C.lambda_ramified) {
        for (const auto& row : C.rows) {
            uint64_t s = 0;
            for (uint32_t v : row) s += v;
            if (s % p) C.product_formula_ok = false;
        }
    }
    C.rstar = rank_mod_p(C.rows, p);
    if (C.rows[0][0] % p) C.nonnorm_chars.push_back(1);
    std::vector<uint32_t> s(p);
    for (uint32_t a = 1; a < p; ++a) s[a] = sym((1 + N - powmod32(r, a, N)) % N);
    for (uint32_t k = 1; k <= (p - 3) / 2; ++k) {
        const uint32_t m = (p - 1 - 2 * k) % (p - 1);
        uint64_t T = 0;
        for (uint32_t a = 1; a < p; ++a) T = (T + static_cast<uint64_t>(powmod32(a, m, p)) * s[a]) % p;
        if (T) C.nonnorm_chars.push_back(2 * k);
    }
    C.equivariant_ok = C.nonnorm_chars.size() == C.rstar;
    C.chev_exponent = C.t - 1 - C.rstar;
    return C;
}

}  // namespace nf
