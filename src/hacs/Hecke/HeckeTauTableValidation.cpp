// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S4V -- INDEPENDENT validation of Stage S4 (HeckeTauTable).
// See HeckeTauTableValidation.h for the independence statement and the full
// list of checks.
//
// -----------------------------------------------------------------------
// Independence discipline (read before modifying): this file NEVER calls
// HeckeTauTable::certify() and NEVER touches HeckeTauTable's private
// eval_tau_on_translate()/boundary_of_terms() (inaccessible anyway -- they
// are private members). Every piece of mathematics used to CHECK the table
// (the boundary operator, the (T1) twisted-equivariance translation, the
// canonical-chain bookkeeping) is re-derived here from
// docs/latex/hecke_action.tex ([C]) and re-implemented from scratch against
// only the public accessors listed in HeckeTauTableValidation.h.
// -----------------------------------------------------------------------

#include "HeckeTauTableValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "ExactInt.h"
#include "../LocalSystemSL2Z.h"      // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h" // SimplicialComplex
#include "../../ModArithmetic/FlatMatrix.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "ValidationSupport.h"

namespace {

using hecke::ChainTerm;
using hecke::HeckeCosetTables;
using hecke::HeckeTauTable;
using hecke::KuSimplexKey;
using hecke::Mat22;
using hecke::SL2ZTessellationTables;
using hecke::TauVertexPolicy;
using hecke::XiTerm;

// =========================================================================
// Section 0: generic simplicial-face bookkeeping, independent of
// HeckeTauTable's private face_simplex_index(). Works purely off the public
// SimplicialComplex::get_simplices() table (vertex lists, sorted increasing
// per [C, sec:scope] orientation convention).
// =========================================================================

using SimplexTable = std::vector<std::vector<std::vector<uint32_t>>>;

uint32_t find_simplex_index(const SimplexTable& simplices, uint32_t q, std::vector<uint32_t> verts) {
    return hecke::vsupport::find_simplex_index(simplices, q, std::move(verts),
                                               "HeckeTauTableValidation: simplex not found for the given vertex set.");
}

// d_j(sigma): delete the j-th vertex (in sigma's increasing vertex order)
// and return the index of the resulting (q-1)-simplex in K's simplex list.
uint32_t face_index(const SimplexTable& simplices, uint32_t q, uint32_t sigma, uint32_t j) {
    std::vector<uint32_t> verts = simplices.at(q).at(sigma);
    if (j >= verts.size()) {
        throw std::runtime_error("HeckeTauTableValidation: face index out of range.");
    }
    verts.erase(verts.begin() + static_cast<std::ptrdiff_t>(j));
    return find_simplex_index(simplices, q - 1, verts);
}

// =========================================================================
// Section 1: our own canonical-chain accumulator. A term coeff * (gamma .
// i(sigma)) of a K^u q-chain is folded, via the PUBLIC
// SL2ZTessellationTables::canonical_key(gamma, q, sigma), into the identity
//   gamma . i(sigma) == key.sign * (key.gamma_canon . i(key.sigma_hat))
// (this is exactly what canonical_key's contract states: candidates
// gamma' = gamma * g over g in S(sigma, sigma_hat), with S(sigma,sigma_hat)
// providing i(sigma) = g * i(sigma_hat) up to the vertex-permutation sign),
// so coeff * sign accumulates at the KEY (q, sigma_hat, gamma_canon) --
// independent of which (gamma, sigma) representation produced the term.
// This is OUR OWN accumulation scheme (not HeckeTauTable's private
// accumulate_canonical()), built only from the public canonical_key() call.
// =========================================================================

using CanonKey = std::array<int64_t, 6>; // {q, sigma_hat, a, b, c, d}
using ChainAcc = std::map<CanonKey, int64_t>;

void add_term(ChainAcc& acc, const SL2ZTessellationTables& model, uint32_t q,
              const Mat22& gamma, uint32_t sigma_idx, int64_t coeff) {
    if (coeff == 0) return;
    const KuSimplexKey key = model.canonical_key(gamma, q, sigma_idx);
    const int64_t signed_coeff = coeff * static_cast<int64_t>(key.sign);
    const CanonKey k{ static_cast<int64_t>(key.q), static_cast<int64_t>(key.sigma_hat),
                       key.gamma_canon.a, key.gamma_canon.b, key.gamma_canon.c, key.gamma_canon.d };
    acc[k] += signed_coeff;
}

void prune_zero(ChainAcc& acc) {
    for (auto it = acc.begin(); it != acc.end(); ) {
        if (it->second == 0) it = acc.erase(it); else ++it;
    }
}

void reduce_mod_ell(ChainAcc& acc, uint32_t ell) {
    for (auto& kv : acc) {
        int64_t v = kv.second % static_cast<int64_t>(ell);
        if (v < 0) v += static_cast<int64_t>(ell);
        kv.second = v;
    }
    prune_zero(acc);
}

// Exact-over-Z comparison unless mod_ell is set ([C, alg:tau](4): "over Z,
// or F_ell if step 3 used mod-ell solving" -- used_mod_ell_solve() is the
// table's own record of which regime applies).
bool chains_equal(ChainAcc a, ChainAcc b, bool mod_ell, uint32_t ell) {
    if (mod_ell) {
        reduce_mod_ell(a, ell);
        reduce_mod_ell(b, ell);
    } else {
        prune_zero(a);
        prune_zero(b);
    }
    return a == b;
}

std::string chain_acc_to_string(const ChainAcc& acc) {
    std::ostringstream oss;
    if (acc.empty()) return "0";
    bool first = true;
    for (const auto& kv : acc) {
        if (!first) oss << " + ";
        first = false;
        oss << kv.second << "*(q=" << kv.first[0] << ",sigma_hat=" << kv.first[1] << ",gamma=[["
            << kv.first[2] << "," << kv.first[3] << "],[" << kv.first[4] << "," << kv.first[5]
            << "]])";
    }
    return oss.str();
}

std::string context_str(uint64_t p, uint32_t r, uint32_t ell, uint32_t n_sym,
                         const char* policy_name) {
    std::ostringstream oss;
    oss << "p=" << p << ",r=" << r << ",ell=" << ell << ",n=" << n_sym
        << ",policy=" << policy_name;
    return oss.str();
}

// =========================================================================
// Oracle (i): chain-map identity d(tau(x)) == tau(d(x)).
// =========================================================================

uint32_t check_chain_map_identity(const SimplexTable& simplices, const SL2ZTessellationTables& model,
                                   const HeckeCosetTables& cosets, const HeckeTauTable& table,
                                   uint32_t ell, uint64_t p, uint32_t r, uint32_t n_sym,
                                   const char* policy_name, uint32_t& checked_out) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const bool mod_ell = table.used_mod_ell_solve();

    for (uint32_t q = 1; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                ++checked_out;
                const std::vector<ChainTerm>& terms = table.tau(i, q, sigma);

                // LHS: OUR boundary operator applied to the returned chain
                // tau(s_i . i(sigma)) = sum_k coeff_k * (gamma_k . i(sigma_k)).
                // d(gamma_k . i(sigma_k)) = sum_j (-1)^j gamma_k . i(d_j sigma_k)
                // (gamma acts simplicially, i is simplicial; [C, sec:conventions]
                // sign convention Sigma_j (-1)^j).
                ChainAcc lhs;
                for (const auto& t : terms) {
                    for (uint32_t j = 0; j <= q; ++j) {
                        const uint32_t fidx = face_index(simplices, q, t.sigma_idx, j);
                        const int64_t sj = (j % 2 == 0) ? 1 : -1;
                        add_term(lhs, model, q - 1, t.gamma, fidx,
                                 static_cast<int64_t>(t.coeff) * sj);
                    }
                }

                // RHS: OUR evaluation of tau(d x), x = s_i . i(sigma):
                //   d x = sum_j (-1)^j (s_i . i(d_j sigma)).
                // Each face term s_i . i(sigma') is, BY CONSTRUCTION of this
                // API (tau() is documented to accept ANY sigma_idx indexing
                // K's q-simplex list, not only an orbit representative --
                // every K-simplex is directly addressable at the SAME coset
                // index i), fetched directly via tau(i, q-1, sigma'). If that
                // were ever NOT the case (a future implementation storing
                // only orbit representatives) the correct fallback is the
                // general (T1) translator independently implemented and
                // exercised below in check_twisted_equivariance(); it is not
                // needed here because the direct fetch always succeeds for
                // this table (asserted structurally, not merely assumed --
                // tau() throws std::out_of_range on any bad index, so a
                // structural violation would surface as an exception here).
                ChainAcc rhs;
                for (uint32_t j = 0; j <= q; ++j) {
                    const uint32_t fidx = face_index(simplices, q, sigma, j);
                    const int64_t sj = (j % 2 == 0) ? 1 : -1;
                    const std::vector<ChainTerm>& face_terms = table.tau(i, q - 1, fidx);
                    for (const auto& ft : face_terms) {
                        add_term(rhs, model, q - 1, ft.gamma, ft.sigma_idx,
                                 static_cast<int64_t>(ft.coeff) * sj);
                    }
                }

                if (!chains_equal(lhs, rhs, mod_ell, ell)) {
                    ChainAcc lhs_disp = lhs, rhs_disp = rhs;
                    if (mod_ell) {
                        reduce_mod_ell(lhs_disp, ell);
                        reduce_mod_ell(rhs_disp, ell);
                    } else {
                        prune_zero(lhs_disp);
                        prune_zero(rhs_disp);
                    }
                    std::cout << "HeckeTauTableValidation: (i) CHAIN-MAP IDENTITY FAILED ["
                              << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                              << ",q=" << q << ",sigma=" << sigma
                              << (mod_ell ? "] (compared mod ell)" : "]") << ":\n"
                              << "    d(tau(x)) = " << chain_acc_to_string(lhs_disp) << "\n"
                              << "    tau(d(x)) = " << chain_acc_to_string(rhs_disp) << std::endl;
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// =========================================================================
// Oracle (ii): augmentation epsilon(tau(v-tilde)) == 1 for every vertex slot.
// Never involves the mod-ell fallback (only triangle fills, step 3, can use
// it): exact over Z always.
// =========================================================================

uint32_t check_augmentation(const SimplexTable& simplices, const HeckeCosetTables& cosets,
                             const HeckeTauTable& table, uint64_t p, uint32_t r, uint32_t ell,
                             uint32_t n_sym, const char* policy_name, uint32_t& checked_out) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const uint32_t nverts = static_cast<uint32_t>(simplices.at(0).size());
    for (uint32_t v = 0; v < nverts; ++v) {
        for (uint32_t i = 1; i <= m; ++i) {
            ++checked_out;
            const std::vector<ChainTerm>& terms = table.tau(i, 0, v);
            int64_t eps = 0;
            for (const auto& t : terms) eps += t.coeff;
            if (eps != 1) {
                std::cout << "HeckeTauTableValidation: (ii) AUGMENTATION FAILED ["
                          << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                          << ",v=" << v << "): epsilon(tau(v)) = " << eps << " != 1." << std::endl;
                ++failures;
            }
        }
    }
    return failures;
}

// =========================================================================
// Oracle (iii): det(gamma_k) == 1; det(g . gamma_k) == N.
// =========================================================================

uint32_t check_determinants(const SimplexTable& simplices, const HeckeCosetTables& cosets,
                             const HeckeTauTable& table, uint64_t p, uint32_t r, uint32_t ell,
                             uint32_t n_sym, const char* policy_name, uint32_t& checked_out) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const Mat22& g = cosets.g();
    const int64_t N = static_cast<int64_t>(cosets.N());
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                for (const auto& t : table.tau(i, q, sigma)) {
                    ++checked_out;
                    if (t.gamma.det() != 1) {
                        std::cout << "HeckeTauTableValidation: (iii) det(gamma_k) != 1 ["
                                  << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                                  << ",q=" << q << ",sigma=" << sigma << "): det = " << t.gamma.det()
                                  << std::endl;
                        ++failures;
                    }
                    const Mat22 g_gamma = g * t.gamma;
                    if (g_gamma.det() != N) {
                        std::cout << "HeckeTauTableValidation: (iii) det(g*gamma_k) != N ["
                                  << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                                  << ",q=" << q << ",sigma=" << sigma << "): det = " << g_gamma.det()
                                  << ", N = " << N << std::endl;
                        ++failures;
                    }
                }
            }
        }
    }
    return failures;
}

// =========================================================================
// Oracle (iv): max||gamma_k|| <= certified_bound_B(); 2*B^2 < 2^63.
// =========================================================================

uint32_t check_bound(const SimplexTable& simplices, const HeckeCosetTables& cosets,
                      const HeckeTauTable& table, uint64_t p, uint32_t r, uint32_t ell,
                      uint32_t n_sym, const char* policy_name) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    uint64_t recomputed_max = 0;
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                for (const auto& t : table.tau(i, q, sigma)) {
                    const uint64_t a = static_cast<uint64_t>(t.gamma.max_abs());
                    if (a > recomputed_max) recomputed_max = a;
                }
            }
        }
    }
    const uint64_t B = table.certified_bound_B();
    if (recomputed_max > B) {
        std::cout << "HeckeTauTableValidation: (iv) BOUND VIOLATION [" << context_str(p, r, ell, n_sym, policy_name)
                  << "]: recomputed max||gamma_k|| = " << recomputed_max
                  << " > certified_bound_B() = " << B << std::endl;
        ++failures;
    }
    // 2*B^2 < 2^63, computed exactly in 128-bit to avoid the very overflow
    // this check exists to rule out.
    const unsigned __int128 lhs128 = static_cast<unsigned __int128>(2) *
                                      static_cast<unsigned __int128>(B) *
                                      static_cast<unsigned __int128>(B);
    const unsigned __int128 rhs128 = static_cast<unsigned __int128>(1) << 63;
    if (!(lhs128 < rhs128)) {
        std::cout << "HeckeTauTableValidation: (iv) BACKEND TIER VIOLATION ["
                  << context_str(p, r, ell, n_sym, policy_name) << "]: 2*B^2 >= 2^63 for B = " << B
                  << std::endl;
        ++failures;
    }
    return failures;
}

// =========================================================================
// Oracle (v): xi_terms(i,q,sigma) == rho_n((g*gamma_k) mod ell), recomputed
// independently via the public ComputeSymmetricPowerMatrixFM utility
// (ModArithmetic; NOT part of Stage S4 -- the same utility S4 itself must
// use per invariant I2, but calling it ourselves on our own operands is a
// genuine independent recomputation, not a reuse of S4's xi_terms builder).
// =========================================================================

FlatMatrix rho_n_of_exact(const Mat22& m, uint32_t ell, uint32_t n_sym) {
    const std::array<uint32_t, 4> mod = m.reduced_mod(ell);
    const FlatMatrix fm(2, 2, ell, std::vector<uint32_t>{ mod[0], mod[1], mod[2], mod[3] });
    return ComputeSymmetricPowerMatrixFM(fm, n_sym);
}

uint32_t check_xi_terms(const SimplexTable& simplices, const HeckeCosetTables& cosets,
                         const HeckeTauTable& table, uint32_t ell, uint64_t p, uint32_t r,
                         uint32_t n_sym, const char* policy_name, uint32_t& checked_out) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const Mat22& g = cosets.g();
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t count = static_cast<uint32_t>(simplices.at(q).size());
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                const std::vector<ChainTerm>& terms = table.tau(i, q, sigma);
                const std::vector<XiTerm>& xis = table.xi_terms(i, q, sigma);
                ++checked_out;
                if (xis.size() != terms.size()) {
                    std::cout << "HeckeTauTableValidation: (v) xi_terms SIZE MISMATCH ["
                              << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                              << ",q=" << q << ",sigma=" << sigma << "): terms=" << terms.size()
                              << " xis=" << xis.size() << std::endl;
                    ++failures;
                    continue;
                }
                for (size_t k = 0; k < terms.size(); ++k) {
                    if (xis[k].coeff != terms[k].coeff || xis[k].sigma_idx != terms[k].sigma_idx) {
                        std::cout << "HeckeTauTableValidation: (v) xi_terms[" << k
                                  << "] (coeff,sigma_idx) MISMATCH ["
                                  << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                                  << ",q=" << q << ",sigma=" << sigma << "): tau=(" << terms[k].coeff
                                  << "," << terms[k].sigma_idx << ") xi=(" << xis[k].coeff << ","
                                  << xis[k].sigma_idx << ")" << std::endl;
                        ++failures;
                    }
                    const Mat22 g_gamma = g * terms[k].gamma;
                    const FlatMatrix expected = rho_n_of_exact(g_gamma, ell, n_sym);
                    // (slot-14 WP-10) through the pool's block accessor (a factored
                    // pool -- the U_ell lever, default since 2026-09-28 -- evaluates it).
                    if (!(expected == table.rho_pool_block(xis[k].rho_g_gamma.index()).get())) {
                        std::cout << "HeckeTauTableValidation: (v) xi_terms[" << k
                                  << "] rho_g_gamma MISMATCH ["
                                  << context_str(p, r, ell, n_sym, policy_name) << "] at (i=" << i
                                  << ",q=" << q << ",sigma=" << sigma << ")." << std::endl;
                        ++failures;
                    }
                }
            }
        }
    }
    return failures;
}

// =========================================================================
// ADDITIONAL cross-check (beyond items i-vi): genuine independent (T1)
// twisted-equivariance translation, exercised via the model's own 8-row
// relation/generator table -- transcribed VERBATIM from
// docs/latex/hecke_action.tex, \S "The arithmetic group datum and the
// 7-vertex model" (\label{sec:model}), NOT read from LocalSystemSL2Z.cpp:
//
//   q  s_lhs   s_rhs   g_r
//   0  {3}     {0}     T
//   0  {5}     {4}     T
//   1  {3,5}   {0,4}   T
//   0  {4}     {5}     S
//   0  {6}     {6}     S   (stabilizer of i)
//   1  {4,6}   {5,6}   S
//   0  {4}     {4}     ST  (stabilizer of rho^2)
//   0  {5}     {5}     TS  (stabilizer of rho)
//
// generator dictionary: rows 1-3 -> T, rows 4-6 -> S, row 7 -> ST, row 8 -> TS
// (matching HeckeCosetTables's documented gen index convention
// 0:T, 1:S, 2:ST, 3:TS).
//
// For each row and each coset index i: i(s_lhs) = g_r * i(s_rhs) means
//   s_i . i(s_lhs) = s_i * g_r . i(s_rhs) = h_i(g_r) * s_{i.g_r} . i(s_rhs)
//                   = h_i(g_r) . x',   x' := s_{i.g_r} . i(s_rhs).
// By (T1) [C, def:tau]: tau(h . x') = (g^{-1} h g) . tau(x') for h in
// Gamma'' = Gamma_0(N). With g = diag(1,N) and h = [[a,b],[c,d]] in
// Gamma_0(N) (c == 0 mod N by construction of the cocycle), the exact
// integral conjugate is g^{-1} h g = [[a, N*b],[c/N, d]] (never a rational
// matrix). This file re-derives that formula directly from [C, def:tau] and
// applies it to the STORED tau(i.g_r, q, sigma_rhs) chain, then compares the
// result against the DIRECTLY stored tau(i, q, sigma_lhs) chain -- a
// genuine two-different-slots cross-check of twisted equivariance, using
// only HeckeCosetTables::perm/cocycle/N/g and the exact Mat22 layer.
// =========================================================================

struct RelationRow {
    uint32_t q;
    std::vector<uint32_t> lhs_v;
    std::vector<uint32_t> rhs_v;
    uint32_t gen; // 0:T 1:S 2:ST 3:TS
    const char* tag;
};

const std::vector<RelationRow>& relation_table() {
    static const std::vector<RelationRow> rows = {
        { 0, { 3 }, { 0 }, 0, "T_v3_v0" },
        { 0, { 5 }, { 4 }, 0, "T_v5_v4" },
        { 1, { 3, 5 }, { 0, 4 }, 0, "T_e(3,5)_e(0,4)" },
        { 0, { 4 }, { 5 }, 1, "S_v4_v5" },
        { 0, { 6 }, { 6 }, 1, "S_v6_stabilizer" },
        { 1, { 4, 6 }, { 5, 6 }, 1, "S_e(4,6)_e(5,6)" },
        { 0, { 4 }, { 4 }, 2, "ST_v4_stabilizer" },
        { 0, { 5 }, { 5 }, 3, "TS_v5_stabilizer" },
    };
    return rows;
}

uint32_t check_twisted_equivariance(const SimplexTable& simplices, const SL2ZTessellationTables& model,
                                     const HeckeCosetTables& cosets, const HeckeTauTable& table,
                                     uint32_t ell, uint64_t p, uint32_t r, uint32_t n_sym,
                                     const char* policy_name, uint32_t& checked_out) {
    uint32_t failures = 0;
    const uint32_t m = cosets.num_cosets();
    const int64_t N = static_cast<int64_t>(cosets.N());
    const bool mod_ell = table.used_mod_ell_solve();

    for (const auto& row : relation_table()) {
        const uint32_t sigma_lhs = find_simplex_index(simplices, row.q, row.lhs_v);
        const uint32_t sigma_rhs = find_simplex_index(simplices, row.q, row.rhs_v);
        for (uint32_t i = 1; i <= m; ++i) {
            ++checked_out;
            const uint32_t j = cosets.perm(row.gen, i);       // i . g_r
            const Mat22 h = cosets.cocycle(row.gen, i);        // h_i(g_r) in Gamma_0(N)

            if (h.c % N != 0) {
                std::cout << "HeckeTauTableValidation: (T1x) cocycle h_i(g_r) not in Gamma_0(N) ["
                          << context_str(p, r, ell, n_sym, policy_name) << "] row=" << row.tag
                          << " i=" << i << ": h.c=" << h.c << " N=" << N << std::endl;
                ++failures;
                continue;
            }
            const int64_t Nb = hecke::checked_mul<int64_t>(h.b, N);
            const Mat22 conj{ h.a, Nb, h.c / N, h.d }; // g^{-1} h g, exact

            ChainAcc predicted;
            for (const auto& t : table.tau(j, row.q, sigma_rhs)) {
                const Mat22 new_gamma = conj * t.gamma;
                add_term(predicted, model, row.q, new_gamma, t.sigma_idx, t.coeff);
            }
            ChainAcc actual;
            for (const auto& t : table.tau(i, row.q, sigma_lhs)) {
                add_term(actual, model, row.q, t.gamma, t.sigma_idx, t.coeff);
            }

            if (!chains_equal(predicted, actual, mod_ell, ell)) {
                ChainAcc pred_disp = predicted, act_disp = actual;
                if (mod_ell) {
                    reduce_mod_ell(pred_disp, ell);
                    reduce_mod_ell(act_disp, ell);
                } else {
                    prune_zero(pred_disp);
                    prune_zero(act_disp);
                }
                std::cout << "HeckeTauTableValidation: (T1x) TWISTED-EQUIVARIANCE CROSS-CHECK "
                             "FAILED [" << context_str(p, r, ell, n_sym, policy_name)
                          << "] row=" << row.tag << " i=" << i << " (j=i.g_r=" << j << "):\n"
                          << "    (g^-1 h g).tau(j,rhs) = " << chain_acc_to_string(pred_disp) << "\n"
                          << "    tau(i,lhs) directly    = " << chain_acc_to_string(act_disp)
                          << std::endl;
                ++failures;
            }
        }
    }
    return failures;
}

// =========================================================================
// Grid point driver.
// =========================================================================

struct GridPoint {
    uint64_t p;
    uint32_t r;
    uint32_t ell;
    uint32_t n_sym;
};

uint32_t run_grid_point(const SimplicialComplex& K, const SL2ZTessellationTables& model,
                         const GridPoint& gp, TauVertexPolicy policy, const char* policy_name,
                         uint32_t& checked_out) {
    uint32_t failures = 0;
    const HeckeCosetTables cosets(gp.p, gp.r, gp.ell, gp.n_sym);
    const HeckeTauTable table(K, model, cosets, gp.ell, gp.n_sym, policy);
    const SimplexTable& simplices = K.get_simplices();

    failures += check_chain_map_identity(simplices, model, cosets, table, gp.ell, gp.p, gp.r,
                                          gp.n_sym, policy_name, checked_out);
    failures += check_augmentation(simplices, cosets, table, gp.p, gp.r, gp.ell, gp.n_sym,
                                    policy_name, checked_out);
    failures += check_determinants(simplices, cosets, table, gp.p, gp.r, gp.ell, gp.n_sym,
                                    policy_name, checked_out);
    failures += check_bound(simplices, cosets, table, gp.p, gp.r, gp.ell, gp.n_sym, policy_name);
    failures += check_xi_terms(simplices, cosets, table, gp.ell, gp.p, gp.r, gp.n_sym, policy_name,
                                checked_out);
    failures += check_twisted_equivariance(simplices, model, cosets, table, gp.ell, gp.p, gp.r,
                                            gp.n_sym, policy_name, checked_out);
    return failures;
}

} // namespace

uint32_t HeckeTauTableValidation() {
    uint32_t failures = 0;
    uint32_t combinations = 0;
    uint32_t total_checked = 0;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const SL2ZTessellationTables model(K);

        // Main grid ([C, obl:tau]/[C, obl:backend] DoD; the task's minimum
        // grid): (p,r) in {(2,1),(3,1),(5,1),(2,2)} x ell in {5,691} x
        // n in {0,2,10}. (5,1) x ell=5 IS the bad case p=ell=5; (2,2) is the
        // r>=2 point. Every point is run under BOTH TauVertexPolicy variants
        // (oracle (vi)).
        std::vector<GridPoint> grid;
        const std::vector<std::pair<uint64_t, uint32_t>> prs = {
            { 2, 1 }, { 3, 1 }, { 5, 1 }, { 2, 2 }
        };
        const std::vector<uint32_t> ells = { 5, 691 };
        const std::vector<uint32_t> ns = { 0, 2, 10 };
        for (const auto& pr : prs) {
            for (uint32_t ell : ells) {
                for (uint32_t n : ns) {
                    grid.push_back(GridPoint{ pr.first, pr.second, ell, n });
                }
            }
        }
        // A couple of large-ell points (optional per the task; kept small in
        // count to bound runtime -- p = 2 has the fewest cosets).
        grid.push_back(GridPoint{ 2, 1, 134217757u, 0 });
        grid.push_back(GridPoint{ 2, 1, 134217757u, 10 });

        const std::vector<std::pair<TauVertexPolicy, const char*>> policies = {
            { TauVertexPolicy::GeometricNearest, "GeometricNearest" },
            { TauVertexPolicy::FixedV6, "FixedV6" },
        };

        for (const auto& gp : grid) {
            for (const auto& pol : policies) {
                ++combinations;
                try {
                    uint32_t checked = 0;
                    const uint32_t f = run_grid_point(K, model, gp, pol.first, pol.second, checked);
                    total_checked += checked;
                    if (f != 0) {
                        std::cout << "HeckeTauTableValidation: " << f
                                  << " discrepancy(ies) at grid point p=" << gp.p << " r=" << gp.r
                                  << " ell=" << gp.ell << " n=" << gp.n_sym
                                  << " policy=" << pol.second << "." << std::endl;
                    }
                    failures += f;
                } catch (const std::exception& e) {
                    std::cout << "HeckeTauTableValidation: EXCEPTION at grid point p=" << gp.p
                              << " r=" << gp.r << " ell=" << gp.ell << " n=" << gp.n_sym
                              << " policy=" << pol.second << ": " << e.what() << std::endl;
                    ++failures;
                }
            }
        }

        std::cout << "HeckeTauTableValidation: " << combinations << " (grid point,policy) "
                  << "combinations exercised across " << grid.size() << " grid points x "
                  << policies.size() << " policies; " << total_checked
                  << " individual oracle assertions checked; " << failures
                  << " discrepancy(ies) found." << std::endl;

        if (failures != 0) return failures;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeTauTableValidation: EXCEPTION during setup: " << e.what() << std::endl;
        return 1000;
    }
}
