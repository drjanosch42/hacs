// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// PureField.h -- the pure field K = Q(N^{1/p}) of NF-PRANK (hecke-extensions slot 22;
// docs/implementation/slot22/NF-PRANK_spec.md sec. 4.1, 4.3, 5 items 1-3). STL only, on
// nf::BigInt / nf::BigRational (check_numberfield_includes.sh rule 1); no CAS.
//
//   * polynomials over F_q (q < 2^32 prime): gcd, powmod, distinct-degree factorisation and
//     Cantor--Zassenhaus equal-degree splitting (brute force for q = 2), the distinct
//     irreducible factors of a NON-squarefree f mod q (gcd with x^{q^d} - x, which is
//     squarefree) with multiplicities (spec sec. 5 item 1);
//   * Dedekind's criterion at one prime (input: the factorisation of f mod q) and the
//     Round-2 (Pohst--Zassenhaus) q-maximal overorder of Z[alpha]: q-radical, ring of
//     multipliers, iterate until stable (Cohen GTM 138 sec. 6.1, locator unverified) -- spec
//     sec. 5 item 2;
//   * the field data of K: d_K = disc(f) / [O_K : Z[alpha]]^2, signature, Minkowski bound;
//   * the prime ideals of O_K of norm <= a bound and the valuation of an element of Z[alpha]
//     at them (spec sec. 5 item 3). For q not dividing the index, Kummer--Dedekind:
//     (q, h(alpha)), h | f mod q irreducible; v(beta) read from gcd(beta(x), f(x)) mod q,
//     accepted only when exactly one prime over q divides beta (spec sec. 4.3 rule). For
//     q = p with p^2 | N - 1 (Z[alpha] not p-maximal): N is a p-th power u^p in Z_p, so
//     x^p - N = (x - u) u^{p-1} Phi_p(x/u) over Q_p and p O_K = P_1 P_2^{p-1}, both of degree
//     1; v_{P_1}(g(alpha)) = v_p(g(u)) (u to precision p^K by Hensel) and
//     v_{P_2} = v_p(N(beta)) - v_{P_1}. The Round-2 discriminant v_p(d_K) = p - 2 is checked
//     against this shape (tame, e = p - 1).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "BigInt.h"
#include "BigRational.h"
#include "Polred.h"

namespace nf {

// ---------------------------------------------------------------------------
// Polynomials over F_q, coefficients in [0, q), constant term first, trimmed.
using FqPoly = std::vector<uint32_t>;

FqPoly fq_trim(FqPoly a);
FqPoly fq_from_int(const std::vector<int64_t>& c, uint32_t q);
FqPoly fq_from_bigint(const IntPoly& c, uint32_t q);
FqPoly fq_mul(const FqPoly& a, const FqPoly& b, uint32_t q);
FqPoly fq_sub(const FqPoly& a, const FqPoly& b, uint32_t q);
void fq_divrem(const FqPoly& a, const FqPoly& b, uint32_t q, FqPoly& quo, FqPoly& rem);
FqPoly fq_mod(const FqPoly& a, const FqPoly& b, uint32_t q);
FqPoly fq_monic(const FqPoly& a, uint32_t q);
FqPoly fq_gcd(FqPoly a, FqPoly b, uint32_t q);  // monic (empty if both zero)
FqPoly fq_powmod(const FqPoly& base, uint64_t e, const FqPoly& m, uint32_t q);
uint32_t fq_eval(const FqPoly& a, uint32_t x, uint32_t q);

// Monic irreducible factors of a SQUAREFREE f over F_q of degree <= maxdeg (each once),
// sorted by (degree, coefficients). Deterministic (seeded by q).
std::vector<FqPoly> fq_factor_squarefree(const FqPoly& f, uint32_t q, unsigned maxdeg);
// All distinct monic irreducible factors of any nonzero f over F_q with their
// multiplicities (f need not be squarefree).
std::vector<std::pair<FqPoly, unsigned>> fq_factor_with_multiplicity(const FqPoly& f, uint32_t q);

// ---------------------------------------------------------------------------
// Dedekind's criterion: Z[alpha] (f monic in Z[x]) is q-maximal iff
// gcd(F mod q, g mod q, h mod q) = 1, with g = prod of the distinct irreducible factors
// of f mod q (lifted), h = (f mod q) / g (lifted), F = (f - g h) / q.
bool dedekind_maximal_at(const IntPoly& f, uint32_t q);

// Round 2 at q: a Z-basis (rows, power-basis coordinates) of the q-maximal overorder
// O_q of Z[alpha]; index_q = [O_q : Z[alpha]] (a power of q).
struct Round2Result {
    std::vector<std::vector<BigRational>> basis;  // n x n, rows
    BigInt index;                                 // [O_q : Z[alpha]]
    unsigned iterations = 0;                      // enlargements made
};
Round2Result round2_at(const IntPoly& f, uint32_t q);

// ---------------------------------------------------------------------------
// The pure field K = Q(N^{1/p}), N prime, p odd prime, N = 1 mod p.
struct PureFieldData {
    uint32_t N = 0, p = 0;
    IntPoly f;                     // x^p - N
    BigInt disc_f;                 // disc(f)
    bool dedekind_p = false;       // Z[alpha] p-maximal by Dedekind
    bool dedekind_N = false;       // Z[alpha] N-maximal by Dedekind
    bool order_maximal = false;    // the order used is O_K (Round 2 run where Dedekind fails, unless forced)
    bool forced_zalpha = false;    // G4 (c): Round 2 suppressed
    BigInt index;                  // [O_K : Z[alpha]] (1 if forced)
    unsigned round2_iterations = 0;
    BigInt dK;                     // disc(f) / index^2
    uint32_t r1 = 0, r2 = 0;
    long double minkowski = 0;     // (4/pi)^{r2} n!/n^n sqrt|d_K|
    bool p_squared = false;        // p^2 | N - 1
    uint64_t p_adic_root = 0;      // u mod p^K (p_squared only), u^p = N in Z_p
    uint64_t p_adic_mod = 0;       // p^K
    std::string refusal;           // non-empty: the field step refused
};
PureFieldData pure_field(uint32_t N, uint32_t p, bool force_zalpha = false);

// A prime ideal of O_K. kind 0: Kummer--Dedekind (q, h(alpha)); kind 1: P_1 over p
// (unramified, alpha -> u); kind 2: P_2 over p (e = p - 1); for q = N and for q = p with
// p || N - 1 the unique (totally ramified) prime is kind 0 with h = x - root.
struct PrimeIdeal {
    uint32_t q = 0;
    unsigned deg = 1;
    unsigned e = 1;
    int kind = 0;
    FqPoly h;            // monic factor of f mod q (kind 0)
    uint32_t root = 0;   // h = x - root when deg = 1 (also the sieve root for kinds 1, 2)
    uint64_t norm = 0;   // q^deg
};

// The prime ideals of O_K of norm <= bound, ordered by (norm, q, h); complete[q] = true when
// every prime over q has norm <= bound.
struct FactorBase {
    std::vector<PrimeIdeal> ideals;
    std::map<uint32_t, bool> all_over_q;              // q -> every prime over q is in the list
    std::map<uint32_t, std::vector<uint32_t>> by_q;   // q -> indices
    std::map<uint64_t, uint32_t> deg1;                // (q << 32 | root) -> index (kind 0, deg 1)
    int find(uint32_t q, const FqPoly& h) const;      // kind-0 ideal with this monic h, or -1
};
FactorBase pure_factor_base(const PureFieldData& K, uint64_t bound);

// Valuations of beta = g(alpha) (g in Z[x], content 1, deg g < p) at the prime ideals of the
// factor base, given |N(beta)| fully factored as prod q^k (all q in the factor base's
// primes). Returns false if some prime over some q is outside the base or the valuation is
// ambiguous (two primes over q divide beta), as in the spec's sec. 4.3 rule.
bool pure_valuations(const PureFieldData& K, const FactorBase& FB, const std::vector<int64_t>& g,
                     const std::vector<std::pair<uint32_t, unsigned>>& norm_fact,
                     std::vector<std::pair<uint32_t, int64_t>>& out);

// |N_{K/Q}(g(alpha))| for g of degree < p (exact; det of the multiplication matrix).
BigInt pure_norm(const PureFieldData& K, const std::vector<int64_t>& g);
long double bigint_to_ld(const BigInt& x);

}  // namespace nf
