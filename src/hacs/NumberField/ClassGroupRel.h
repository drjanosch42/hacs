// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// ClassGroupRel.h -- the p-rank of Cl(K), K = Q(N^{1/p}), by relations (NF-PRANK, slot 22;
// docs/implementation/slot22/NF-PRANK_spec.md sec. 3, 4.3-4.6, 5 items 4-7). STL only, no CAS.
//
// Route (p = 5; the relation arm refuses other p in slot 22):
//   1. the field (PureField.h): d_K with Round 2 where Dedekind fails, the Minkowski bound M_K;
//   2. the factor base S = every prime ideal of norm <= floor(M_K) + 1 (S7: S generates Cl(K); the +1 guards the floating floor);
//   3. relations among the "dense" ideals S_0 (norm <= B_0, B_0 = M_K by default): a line sieve on
//      a + b alpha (norm a^p + N b^p, homogeneous), a box of a + b alpha + c alpha^2 for the
//      degree >= 2 ideals, and (q) = prod P^e when every prime over q lies in S_0;
//   4. every ideal of S \ S_0 is DESCENDED: one element of the ideal (2-d lattice for degree 1,
//      3-d LLL lattice for degree 2) whose norm is q^deg times a cofactor whose prime ideals all
//      come earlier in the (norm, q, h) order, or (q) itself when the ideal is the last prime over q.
//      The descent rows are lower-triangular with unit diagonal on S \ S_0, hence
//      Z^S / <R_0, D> = Z^{S_0} / R_0, which surjects onto Cl(K);
//   5. Z^{S_0}/R_0 by sparse elimination on +-1 pivots and a dense BigInt Smith form of the rest:
//      group_bound (the invariant factors; a 0 = free summand = not enough relations), rank_upper
//      = its p-rank -- CERTIFIED when S is Minkowski-complete and every descent succeeded;
//   6. rank_lower = 1 from the genus field (S6): the character chi(Q) = (NQ mod N)^{(N-1)/p} is
//      non-trivial on S and must kill every relation and every descent row (genus_ok);
//   7. units = kernel vectors of the elimination; each row carries its log vector at the
//      r_1 + r_2 = 3 places and its local p-th-power characters at test primes q > M_K
//      (homomorphisms, so no history is stored); R' = the regulator of the found unit group E'
//      (2-d log lattice); "p does not divide [E:E']" CERTIFIED by the F_p-rank (= 2) of those
//      characters on E';
//   8. the analytic class number formula with the Euler product truncated at X (NOT certified):
//      h_times_index = hR / R' where hR = w sqrt|d_K| / (2^{r_1} (2 pi)^{r_2}) Res. Note: since
//      R' = [E:E'] R this quotient is h / [E:E'] (the spec's and the prototype's name
//      "h * [E:E']" is kept for the field; with [E:E'] = 1 both agree).
// Certification levels as spec sec. 4.6 (REFUSED carries the reason by name).
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "Ambiguous.h"
#include "BigInt.h"
#include "PureField.h"

namespace nf {

enum class PRankLevel { CERTIFIED_EXACT, UPPER_CERTIFIED_LOWER_ANALYTIC, GRH, REFUSED };
const char* prank_level_name(PRankLevel l);

struct PRankOptions {
    int64_t sieve_A = 10000;         // |a| <= A
    uint32_t sieve_B = 200;          // 1 <= b <= B
    int box3_ab = 25;                // a + b alpha + c alpha^2, |a|, |b| <= box3_ab
    int box3_c = 15;                 //                          1 <= c <= box3_c
    uint64_t dense_bound = 0;        // B_0 (0 = M_K, no descent)
    uint64_t euler_X = 10000000;     // Euler product to X
    bool analytic = true;            // units, regulator, index certificate, Euler product
    bool force_zalpha = false;       // G4 (c): suppress Round 2
    int drop_ideal = -1;             // G4 (a): remove the ideal with this factor-base index from S
    bool corrupt_relation = false;   // G4 (b): corrupt one valuation of one relation
    uint64_t max_minkowski = 5000000;
    unsigned index_test_primes = 12;
    unsigned max_widen = 3;          // sieve_B doubled at most this often if Z^{S_0}/R_0 is not finite
};

struct PRankResult {
    uint32_t N = 0, p = 0;
    // field
    BigInt dK, index;
    uint32_t r1 = 0, r2 = 0;
    bool order_maximal = false, dedekind_p = false, dedekind_N = false;
    unsigned round2_iterations = 0;
    long double minkowski = 0;
    // Chevalley data over F = Q(zeta_p)
    ChevalleyData chev;
    // relation side
    std::size_t factor_base = 0;                                  // |S| (every prime of norm <= M_K)
    std::vector<std::pair<uint32_t, unsigned>> deg2plus;          // (q, deg) of the degree >= 2 ideals of S
    std::size_t dense_ideals = 0;                                 // |S_0|
    uint64_t dense_bound = 0;                                     // B_0
    uint32_t sieve_B_used = 0;
    std::size_t candidates = 0, relations_2term = 0, relations_3term = 0, relations_q = 0, relations = 0;
    std::size_t descended = 0, descended_by_q = 0, descent_failed = 0;
    std::string free_columns;                                     // diagnostic: columns without a pivot (free part)
    std::size_t extra_rows = 0;                                   // lattice-element relations added for weakly covered dense ideals
    std::vector<BigInt> group_bound;                              // invariant factors != 1 of Z^{S_0}/R_0 (0 = free)
    uint32_t rank_Fp_rel = 0;                                     // F_p-rank of R_0
    uint32_t rank_upper = 0;
    bool minkowski_complete = false;
    uint32_t rank_lower = 0;
    bool genus_ok = false, genus_nontrivial = false;
    std::size_t genus_bad = 0;
    // analytic side
    std::size_t kernel_vectors = 0, rejected_numerics = 0;
    double regulator_found = 0;
    uint32_t index_rank = 0;
    bool p_coprime_unit_index = false;
    double euler_res = 0, hR_estimate = 0, h_times_index = 0;
    uint64_t euler_X = 0;
    PRankLevel level = PRankLevel::REFUSED;
    std::string refusal;
    double secs = 0;
};

PRankResult nf_prank(uint32_t N, uint32_t p, const PRankOptions& opt = PRankOptions());

// Exposed helpers (tested by the validation).
bool is_prime_u64(uint64_t n);                                         // deterministic Miller--Rabin
bool factor_u64(uint64_t n, std::vector<std::pair<uint64_t, unsigned>>& out);  // Pollard--Brent rho; false on failure
// Res_{s=1} zeta_K ~ prod_{q <= X} (1 - 1/q) / prod_{Q | q} (1 - NQ^{-1}) for K = Q(N^{1/p}) (closed-form splitting; truncated, not certified).
double pure_euler_residue(uint32_t N, uint32_t p, uint64_t X);

}  // namespace nf
