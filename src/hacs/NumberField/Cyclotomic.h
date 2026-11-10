// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Cyclotomic.h -- Z[zeta_{ell^k}] exactly (BigInt coefficients) and modulo M
// (uint32_t coefficients), the cyclotomic units u_c, Galois action sigma_t and
// the valuation v_pi at lambda = (1 - zeta) (slot-19 WP-NF; [E] sec:lvnf;
// the arithmetic of R18-2's cycarith.py / layer2.py, now in C++).
//
// Elements are coefficient vectors of length n = phi(ell^k) on the power basis
// 1, z, ..., z^{n-1} of Z[z]/Phi_{ell^k}(z), z = zeta_{ell^k}.
// Phi_{ell^k}(z) = sum_{j<ell} z^{j s}, s = ell^{k-1}.
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "BigInt.h"

namespace nf {

// f_c = c^{-i} mod m, lifted to (-m/2, m/2), for c in (Z/m)^x, 2 <= c < m
// (c = 1 is skipped: u_1 = 1). m = ell^k; i is taken mod phi(m).
std::map<uint32_t, int64_t> kummer_exponents(uint32_t ell, unsigned k, uint64_t i);

// (Z/M)[z]/Phi_{ell^k}. When M = ell^P, v_pi is available (precision pi^{P e}).
class CycloModRing {
public:
    using Elt = std::vector<uint32_t>;

    CycloModRing(uint32_t ell, unsigned k, uint32_t M);

    uint32_t ell() const { return ell_; }
    unsigned k() const { return k_; }
    uint32_t m() const { return m_; }  // ell^k
    uint32_t n() const { return n_; }  // phi(ell^k) = e, the ramification index
    uint32_t M() const { return M_; }
    unsigned precision() const { return prec_; }  // P with M = ell^P (0 if M is not a power of ell)
    uint64_t products() const { return products_; }

    Elt zero() const { return Elt(n_, 0); }
    Elt one() const;
    Elt pi() const;  // 1 - z
    // Reduce a dense list in z of any length (entries mod M) modulo z^m - 1, then Phi.
    Elt reduce(const std::vector<uint32_t>& dense) const;
    Elt from_signed(const std::vector<int64_t>& coeffs) const;  // any length, reduced
    Elt from_bigint(const std::vector<BigInt>& coeffs) const;   // length n

    Elt add(const Elt& a, const Elt& b) const;
    Elt sub(const Elt& a, const Elt& b) const;
    Elt mul(const Elt& a, const Elt& b) const;
    Elt sqr(const Elt& a) const;
    Elt pow(const Elt& a, uint64_t e) const;
    // prod Y_i^{a_i} with shared squarings (Straus); zero exponents skipped.
    Elt multiexp(const std::vector<std::pair<const Elt*, uint64_t>>& terms) const;
    Elt sigma(const Elt& a, uint64_t t) const;  // z -> z^t, t prime to ell

    // u_c = (1 - z^c)/(1 - z) = 1 + z + ... + z^{c-1}, c prime to ell (c taken mod m).
    Elt unit_u(uint64_t c) const;
    // u_c^{-1} = (1 - z^{c d})/(1 - z^c) = 1 + z^c + ... + z^{c(d-1)}, d = c^{-1} mod m.
    Elt unit_u_inv(uint64_t c) const;
    // eta = prod u_c^{f_c} (product form, never expanded exactly).
    Elt product_unit(const std::map<uint32_t, int64_t>& f) const;

    // v_pi(a) and the leading F_ell-coefficient: a = lead * pi^v + O(pi^{v+1}) mod ell.
    // Requires M = ell^P; returns {P * n, 0} when a == 0 mod M (meaning ">= P n").
    std::pair<unsigned, uint32_t> vpi_lead(const Elt& a) const;
    unsigned vpi(const Elt& a) const { return vpi_lead(a).first; }
    std::pair<unsigned, uint32_t> vone_lead(const Elt& a) const;  // of a - 1
    unsigned vone(const Elt& a) const { return vone_lead(a).first; }

    // a evaluated at z = r modulo q (q prime, r of order m mod q).
    static uint32_t eval_at(const Elt& a, uint32_t r, uint32_t q);

private:
    uint32_t ell_;
    unsigned k_;
    uint32_t m_, n_, s_, M_;
    unsigned prec_;
    std::vector<uint32_t> pascal_;  // ell x ell binomials mod ell (only when prec_ > 0)
    mutable uint64_t products_ = 0;
};

// Z[zeta_{ell^k}] exactly, BigInt coefficients.
class CycloZ {
public:
    using Elt = std::vector<BigInt>;

    CycloZ(uint32_t ell, unsigned k);
    uint32_t n() const { return n_; }
    uint32_t m() const { return m_; }
    Elt one() const;
    Elt reduce(std::vector<BigInt> dense) const;
    Elt mul(const Elt& a, const Elt& b) const;
    Elt pow(const Elt& a, uint64_t e) const;
    Elt multiexp(const std::vector<std::pair<const Elt*, uint64_t>>& terms) const;
    Elt unit_u(uint64_t c) const;
    Elt unit_u_inv(uint64_t c) const;
    Elt product_unit(const std::map<uint32_t, int64_t>& f) const;
    Elt sigma(const Elt& a, uint64_t t) const;
    // v_pi exactly (k = 1 only): a(1 - pi) = sum b_j pi^j, v = min ((ell-1) v_ell(b_j) + j). -1 if a == 0.
    long vpi_exact(const Elt& a) const;

private:
    uint32_t ell_;
    unsigned k_;
    uint32_t m_, n_, s_;
};

// eval of an exact element at z = r mod q.
uint32_t eval_at(const CycloZ::Elt& a, uint32_t r, uint32_t q);

// The product form prod_c ((1 - r^c)/(1 - r))^{f_c} mod q, r of order m (exponents reduced mod q - 1).
uint32_t eval_product_unit(const std::map<uint32_t, int64_t>& f, uint32_t r, uint32_t q);

// The first `count` primes q = 1 mod m, and an element of exact order m mod q.
std::vector<uint32_t> split_primes(uint32_t m, unsigned count);
uint32_t root_of_order(uint32_t q, uint32_t m);

}  // namespace nf
