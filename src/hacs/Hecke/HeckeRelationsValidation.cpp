// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S9V -- see HeckeRelationsValidation.h for the file-level
// independence statement, the full O9/O10 oracle list, and the A-F1 gating
// rules this file must honor.

#include "HeckeRelationsValidation.h"

#include "HeckeOperatorFamily.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Pipeline plumbing (mechanical; NOT part of the independently-derived
// expected values -- see HeckeRelationsValidation.h). Mirrors the public
// construction recipe documented in HeckeOperatorFamily.h / LocalSystemSL2Z.h:
// build the 7-vertex model K, its S2 tessellation tables, the base local
// system L = Sym^n(F_ell^2) via the single-source-of-truth relation
// generator, compute its cohomology, then build a HeckeOperatorFamily on top.
// ---------------------------------------------------------------------

std::vector<LocalSystemSimplicialComplex::SimplexRelation> build_relations(
    const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    return rels;
}

// Owns the full stack for one (ell, n_sym) grid point. K / model / coh are
// declared (hence constructed, and destroyed in reverse order) before
// `family`, satisfying HeckeOperatorFamily's documented lifetime requirement
// that cohomL/K/model outlive it and every reference/value it returns.
struct Stack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;
    std::unique_ptr<hecke::HeckeOperatorFamily> family;

    Stack(uint32_t ell, uint32_t n_sym)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, build_relations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {
        coh.compute_cohomology();
        family = std::make_unique<hecke::HeckeOperatorFamily>(&coh, K, model, ell, n_sym);
    }
};

// scalar * I_d over F_ell (d may be 0, giving the 0x0 "identity").
Matrix scalar_identity(uint32_t d, uint32_t ell, uint32_t scalar) {
    Matrix I(d, ell); // Matrix(rows, prime) ctor: identity matrix.
    return I * scalar;
}

bool well_typed(const Matrix& M, uint32_t d, uint32_t ell) {
    return M.rows == d && M.cols == d && M.p == ell;
}

std::string mat_str(const Matrix& M) { return M.to_string(); }

// ---------------------------------------------------------------------
// O9 -- internal-consistency Hecke-algebra identities, contract-backed for
// ALL ell >= 5 ([C, obl:algebra]). See HeckeRelationsValidation.h for the
// derivation of each expected identity.
// ---------------------------------------------------------------------

uint32_t run_O9_case(uint32_t ell, uint32_t n_sym) {
    Stack st(ell, n_sym);
    hecke::HeckeOperatorFamily& family = *st.family;
    const auto& dims = st.coh.get_H_eq_dimensions();
    const uint32_t num_q = static_cast<uint32_t>(dims.size());

    const std::vector<Matrix>& A2 = family.T_p(2);
    const std::vector<Matrix>& A3 = family.T_p(3);
    const std::vector<Matrix> T4_full_direct = family.T_p_pow_direct(2, 2);      // FULL T_4, independent route
    const std::vector<Matrix> T4_primitive = family.T_p_pow_primitive_direct(2, 2); // raw primitive T'_4
    const std::vector<Matrix> T6 = family.T_n(6);
    const std::vector<Matrix> Tpp2 = family.T_pp(2);
    const std::vector<Matrix> Tpp3 = family.T_pp(3);

    const uint32_t scalar_2_np1 = ModArithmetic::mod_pow(2u % ell, n_sym + 1, ell); // 2^{n+1} mod ell
    const uint32_t scalar_2_n = ModArithmetic::mod_pow(2u % ell, n_sym, ell);       // 2^n mod ell
    const uint32_t scalar_3_n = ModArithmetic::mod_pow(3u % ell, n_sym, ell);       // 3^n mod ell

    if (A2.size() != num_q || A3.size() != num_q) {
        std::cout << "HeckeRelationsValidation: O9 (ell=" << ell << ", n=" << n_sym
                  << "): degree-count mismatch between T_p(2)/T_p(3) and H_eq_dimensions." << std::endl;
        return 1;
    }

    for (uint32_t q = 0; q < num_q; ++q) {
        const uint32_t d = dims[q];

        // (1) Commutativity: [A(T_2), A(T_3)] = 0.
        const Matrix lhs1 = A2[q] * A3[q];
        const Matrix rhs1 = A3[q] * A2[q];
        if (lhs1 != rhs1) {
            std::cout << "HeckeRelationsValidation: O9.1 (commutativity) FAILED at (ell,n,q)=("
                      << ell << "," << n_sym << "," << q << ")." << std::endl;
            std::cout << "  A(T_2)*A(T_3):\n" << mat_str(lhs1) << std::endl;
            std::cout << "  A(T_3)*A(T_2):\n" << mat_str(rhs1) << std::endl;
            return 10;
        }

        // (2) Recursion vs independent direct (both FULL): T_p_pow_direct(2,2) == A(T_2)^2 - 2^{n+1} I.
        const Matrix expected_T4 = A2[q] * A2[q] - scalar_identity(d, ell, scalar_2_np1);
        if (T4_full_direct[q] != expected_T4) {
            std::cout << "HeckeRelationsValidation: O9.2 (T_4 = T_2^2 - 2^{n+1} I, FULL operator, "
                         "independent route) FAILED at (ell,n,q)=("
                      << ell << "," << n_sym << "," << q << ")." << std::endl;
            std::cout << "  T_p_pow_direct(2,2) [FULL, independent]:\n" << mat_str(T4_full_direct[q]) << std::endl;
            std::cout << "  A(T_2)^2 - 2^{n+1} I [expected]:\n" << mat_str(expected_T4) << std::endl;
            return 20;
        }

        // (3) Multiplicativity: T_n(6) == A(T_2) A(T_3).
        const Matrix expected_T6 = A2[q] * A3[q];
        if (T6[q] != expected_T6) {
            std::cout << "HeckeRelationsValidation: O9.3 (T_6 = T_2 T_3) FAILED at (ell,n,q)=("
                      << ell << "," << n_sym << "," << q << ")." << std::endl;
            std::cout << "  T_n(6):\n" << mat_str(T6[q]) << std::endl;
            std::cout << "  A(T_2)*A(T_3) [expected]:\n" << mat_str(expected_T6) << std::endl;
            return 30;
        }

        // (4) Diamond: T_pp(p) == p^n I, p in {2,3}.
        const Matrix expected_Tpp2 = scalar_identity(d, ell, scalar_2_n);
        if (Tpp2[q] != expected_Tpp2) {
            std::cout << "HeckeRelationsValidation: O9.4 (T(2,2) = 2^n I) FAILED at (ell,n,q)=("
                      << ell << "," << n_sym << "," << q << ")." << std::endl;
            std::cout << "  T_pp(2):\n" << mat_str(Tpp2[q]) << std::endl;
            std::cout << "  2^n I [expected]:\n" << mat_str(expected_Tpp2) << std::endl;
            return 40;
        }
        const Matrix expected_Tpp3 = scalar_identity(d, ell, scalar_3_n);
        if (Tpp3[q] != expected_Tpp3) {
            std::cout << "HeckeRelationsValidation: O9.4 (T(3,3) = 3^n I) FAILED at (ell,n,q)=("
                      << ell << "," << n_sym << "," << q << ")." << std::endl;
            std::cout << "  T_pp(3):\n" << mat_str(Tpp3[q]) << std::endl;
            std::cout << "  3^n I [expected]:\n" << mat_str(expected_Tpp3) << std::endl;
            return 41;
        }

        // (5) Decomposition witness: T_p_pow_direct(2,2) == T_p_pow_primitive_direct(2,2) + 2^n I.
        const Matrix expected_decomp = T4_primitive[q] + scalar_identity(d, ell, scalar_2_n);
        if (T4_full_direct[q] != expected_decomp) {
            std::cout << "HeckeRelationsValidation: O9.5 (decomposition witness T_4^full = T'_4 + 2^n I) "
                         "FAILED at (ell,n,q)=("
                      << ell << "," << n_sym << "," << q << ")." << std::endl;
            std::cout << "  T_p_pow_direct(2,2) [full]:\n" << mat_str(T4_full_direct[q]) << std::endl;
            std::cout << "  T_p_pow_primitive_direct(2,2) + 2^n I [expected]:\n" << mat_str(expected_decomp) << std::endl;
            return 50;
        }
    }

    std::cout << "HeckeRelationsValidation: O9 PASSED at (ell,n)=(" << ell << "," << n_sym
              << ") for all q=0.." << (num_q == 0 ? 0 : num_q - 1)
              << " [commutativity, T_4=T_2^2-2^{n+1}I (full, independent), T_6=T_2 T_3, "
                 "T(p,p)=p^n I (p=2,3), decomposition witness]." << std::endl;
    return 0;
}

// ---------------------------------------------------------------------
// O10, good prime p = ell = 5 (contract-backed, ell >= 5): HARD asserts.
// ---------------------------------------------------------------------

uint32_t run_O10_good_prime_case(uint32_t n_sym) {
    const uint32_t ell = 5;
    Stack st(ell, n_sym); // if this throws, A-F1's "pipeline completes" claim is FALSE at ell=5 -- a real bug.
    hecke::HeckeOperatorFamily& family = *st.family;
    const auto& dims = st.coh.get_H_eq_dimensions();
    const uint32_t num_q = static_cast<uint32_t>(dims.size());

    const std::vector<Matrix>& U5 = family.T_p(5); // U_5 = T_p(5), well-defined (no throw reaching here).
    const std::vector<Matrix> U5_rec2 = family.T_p_pow(5, 2);
    const std::vector<Matrix> U5_direct2 = family.T_p_pow_direct(5, 2); // INDEPENDENT (fresh N=25 pipeline)

    if (U5.size() != num_q || U5_rec2.size() != num_q || U5_direct2.size() != num_q) {
        std::cout << "HeckeRelationsValidation: O10 (ell=5, n=" << n_sym
                  << "): degree-count mismatch." << std::endl;
        return 1;
    }

    for (uint32_t q = 0; q < num_q; ++q) {
        const Matrix U5_sq = U5[q] * U5[q];

        // Recursion route: trivially true by construction (recursion scalar
        // 5^{n+1} mod 5 == 0 whenever p == ell), but asserted as a basic
        // sanity/plumbing check.
        if (U5_rec2[q] != U5_sq) {
            std::cout << "HeckeRelationsValidation: O10 (ell=5, n=" << n_sym
                      << ") recursion sanity check T_p_pow(5,2) == T_p(5)^2 FAILED at q=" << q << "." << std::endl;
            std::cout << "  T_p_pow(5,2):\n" << mat_str(U5_rec2[q]) << std::endl;
            std::cout << "  T_p(5)^2 [expected]:\n" << mat_str(U5_sq) << std::endl;
            return 10;
        }

        // The GENUINE independent check (cor:ulrecursion, U_25 = U_5^2 via
        // a pipeline run entirely disjoint from U_5's own).
        if (U5_direct2[q] != U5_sq) {
            std::cout << "HeckeRelationsValidation: O10 (ell=5, n=" << n_sym
                      << ") HARD FAILURE: independent double-coset route T_p_pow_direct(5,2) != "
                         "T_p(5)^2 (U_25 != U_5^2 via [C, cor:ulrecursion]) at q=" << q << "." << std::endl;
            std::cout << "  T_p_pow_direct(5,2) [independent, N=25 pipeline]:\n" << mat_str(U5_direct2[q]) << std::endl;
            std::cout << "  T_p(5)^2 [expected]:\n" << mat_str(U5_sq) << std::endl;
            return 20;
        }
    }

    // Rank monotonicity / stabilization: rank(U_5^r) non-increasing in r,
    // stationary (constant) by r = d_q.
    uint32_t max_d = 0;
    for (uint32_t d : dims) max_d = std::max(max_d, d);
    const uint32_t R = max_d + 3; // a few steps past the worst-case stabilization point

    std::vector<std::vector<uint32_t>> ranks(num_q);
    for (uint32_t r = 1; r <= R; ++r) {
        const std::vector<Matrix> Ur = family.T_p_pow(5, r);
        for (uint32_t q = 0; q < num_q; ++q) {
            Matrix copy = Ur[q]; // rank() is non-const (caches RREF).
            ranks[q].push_back(copy.rank());
        }
    }

    for (uint32_t q = 0; q < num_q; ++q) {
        const uint32_t d = dims[q];
        for (size_t i = 1; i < ranks[q].size(); ++i) {
            if (ranks[q][i] > ranks[q][i - 1]) {
                std::cout << "HeckeRelationsValidation: O10 (ell=5, n=" << n_sym
                          << ") HARD FAILURE: rank(U_5^r) is NOT non-increasing at q=" << q
                          << " (r=" << i << " -> rank " << ranks[q][i] << " > r=" << i
                          << " -> rank " << ranks[q][i - 1] << ")." << std::endl;
                return 30;
            }
        }
        if (d >= 1) {
            const uint32_t stable_val = ranks[q][d - 1]; // rank at r = d
            for (size_t i = d - 1; i < ranks[q].size(); ++i) {
                if (ranks[q][i] != stable_val) {
                    std::cout << "HeckeRelationsValidation: O10 (ell=5, n=" << n_sym
                              << ") HARD FAILURE: rank(U_5^r) not stationary by r=d_q=" << d
                              << " at q=" << q << " (rank at r=" << d << " is " << stable_val
                              << ", rank at r=" << (i + 1) << " is " << ranks[q][i] << ")." << std::endl;
                    return 31;
                }
            }
        }
    }

    std::cout << "HeckeRelationsValidation: O10 (ell=5) PASSED at n=" << n_sym
              << " (pipeline completes; T_p_pow(5,2)==T_p(5)^2 and, independently, "
                 "T_p_pow_direct(5,2)==T_p(5)^2 [U_25=U_5^2]; rank(U_5^r) non-increasing "
                 "and stationary by r=d_q, all q)." << std::endl;
    return 0;
}

// ---------------------------------------------------------------------
// O10, bad-gate primes p = ell in {2,3} (NOT contract-backed, A-F1): the
// pipeline-completion / well-typedness claims are HARD-asserted; the
// algebra-relation observations are RECORDED (printed) only.
// ---------------------------------------------------------------------

uint32_t run_O10_bad_gate_case(uint32_t ell, uint32_t n_sym) {
    try {
        Stack st(ell, n_sym);
        hecke::HeckeOperatorFamily& family = *st.family;
        const auto& dims = st.coh.get_H_eq_dimensions();
        const uint32_t num_q = static_cast<uint32_t>(dims.size());

        const std::vector<Matrix>& U = family.T_p(ell); // U_ell = T_p(ell); well-defined iff this doesn't throw.
        const std::vector<Matrix> U_rec2 = family.T_p_pow(ell, 2);
        const std::vector<Matrix> U_direct2 = family.T_p_pow_direct(ell, 2);

        if (U.size() != num_q || U_rec2.size() != num_q || U_direct2.size() != num_q) {
            std::cout << "HeckeRelationsValidation: O10 (ell=" << ell << ", n=" << n_sym
                      << ") HARD FAILURE: degree-count mismatch." << std::endl;
            return 1;
        }

        // HARD assertion: well-typedness (square, correct dimension, correct modulus).
        for (uint32_t q = 0; q < num_q; ++q) {
            const uint32_t d = dims[q];
            if (!well_typed(U[q], d, ell) || !well_typed(U_rec2[q], d, ell) || !well_typed(U_direct2[q], d, ell)) {
                std::cout << "HeckeRelationsValidation: O10 (ell=" << ell << ", n=" << n_sym
                          << ") HARD FAILURE: ill-typed matrix at q=" << q << " (expected " << d
                          << "x" << d << " mod " << ell << ")." << std::endl;
                std::cout << "  T_p(" << ell << "): " << U[q].rows << "x" << U[q].cols << " mod " << U[q].p << std::endl;
                std::cout << "  T_p_pow(" << ell << ",2): " << U_rec2[q].rows << "x" << U_rec2[q].cols
                          << " mod " << U_rec2[q].p << std::endl;
                std::cout << "  T_p_pow_direct(" << ell << ",2): " << U_direct2[q].rows << "x" << U_direct2[q].cols
                          << " mod " << U_direct2[q].p << std::endl;
                return 2;
            }
        }

        // RECORD (do not assert, per A-F1): U_{ell^2} == U_ell^2, via both the
        // recursion route (tautologically true given the recursion's own
        // scalar-== 0 construction when p == ell -- a plumbing check, not a
        // test of the algebra) and the independent double-coset route
        // (the substantive, NOT contract-backed at this ell, comparison).
        std::cout << "HeckeRelationsValidation: [A-F1 RECORD ONLY, NOT asserted -- ell=" << ell
                  << " not contract-backed for the Hecke recursion] n=" << n_sym << ":" << std::endl;
        for (uint32_t q = 0; q < num_q; ++q) {
            const Matrix U_sq = U[q] * U[q];
            const bool eq_rec = (U_rec2[q] == U_sq);
            const bool eq_direct = (U_direct2[q] == U_sq);
            std::cout << "  q=" << q << " (dim=" << dims[q] << "): T_p_pow(" << ell << ",2)==U_" << ell
                      << "^2 ? " << (eq_rec ? "true" : "false") << " (recursion route, tautological); "
                      << "T_p_pow_direct(" << ell << ",2)==U_" << ell << "^2 ? "
                      << (eq_direct ? "true" : "false") << " (independent double-coset route, substantive)."
                      << std::endl;
        }

        // RECORD (do not assert, per A-F1): rank(U_ell^r) sequence and
        // whether it is non-increasing / stabilizes.
        uint32_t max_d = 0;
        for (uint32_t d : dims) max_d = std::max(max_d, d);
        const uint32_t R = max_d + 3;
        for (uint32_t q = 0; q < num_q; ++q) {
            std::vector<uint32_t> ranks;
            ranks.reserve(R);
            for (uint32_t r = 1; r <= R; ++r) {
                Matrix Ur = family.T_p_pow(ell, r)[q];
                ranks.push_back(Ur.rank());
            }
            bool monotone = true;
            for (size_t i = 1; i < ranks.size(); ++i) {
                if (ranks[i] > ranks[i - 1]) monotone = false;
            }
            std::cout << "  q=" << q << ": rank(U_" << ell << "^r), r=1.." << R << " = [";
            for (size_t i = 0; i < ranks.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << ranks[i];
            }
            std::cout << "]; non-increasing? " << (monotone ? "true" : "false") << std::endl;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeRelationsValidation: O10 (ell=" << ell << ", n=" << n_sym
                  << ") HARD FAILURE: pipeline did NOT complete (A-F1 still requires completion "
                     "even though the algebra relations are not contract-backed at this ell): "
                  << e.what() << std::endl;
        return 3;
    }
}

} // namespace

uint32_t HeckeRelationsValidation() {
    try {
        // ---- O9: internal-consistency Hecke-algebra identities (n in {0,2,10}, ell in {13,101}, all q). ----
        for (const uint32_t ell : {13u, 101u}) {
            for (const uint32_t n_sym : {0u, 2u, 10u}) {
                const uint32_t code = run_O9_case(ell, n_sym);
                if (code != 0) {
                    std::cout << "HeckeRelationsValidation: FAILED (O9) at (ell,n)=(" << ell << ","
                              << n_sym << "), code=" << code << "." << std::endl;
                    return 1000 + code;
                }
            }
        }

        // ---- O10, p = ell = 5 (contract-backed, ell >= 5): hard asserts. ----
        for (const uint32_t n_sym : {0u, 2u, 10u}) {
            const uint32_t code = run_O10_good_prime_case(n_sym);
            if (code != 0) {
                std::cout << "HeckeRelationsValidation: FAILED (O10, ell=5) at n=" << n_sym
                          << ", code=" << code << "." << std::endl;
                return 2000 + code;
            }
        }

        // ---- O10, p = ell in {2,3} (NOT contract-backed, A-F1): hard-assert
        // ---- only completion/well-typedness; record everything else. ----
        for (const uint32_t ell : {2u, 3u}) {
            for (const uint32_t n_sym : {0u, 2u, 10u}) {
                const uint32_t code = run_O10_bad_gate_case(ell, n_sym);
                if (code != 0) {
                    std::cout << "HeckeRelationsValidation: FAILED (O10, ell=" << ell << ") at n="
                              << n_sym << ", code=" << code << "." << std::endl;
                    return 3000 + code;
                }
            }
        }

        std::cout << "HeckeRelationsValidation: PASSED -- O9 (commutativity, T_4=T_2^2-2^{n+1}I "
                     "[FULL operator via the independent double-coset route], T_6=T_2 T_3, "
                     "T(p,p)=p^n I, decomposition witness) for n in {0,2,10}, ell in {13,101}, all "
                     "q; O10 hard-asserted at p=ell=5 (pipeline completes, U_25=U_5^2 via the "
                     "recursion AND independently via the fresh N=25 double-coset pipeline, "
                     "rank(U_5^r) non-increasing and stationary by r=d_q); O10 recorded-only "
                     "(per A-F1) at p=ell in {2,3} (pipeline completion and well-typedness "
                     "hard-asserted; the U_{ell^2}=U_ell^2 identity and rank(U_ell^r) monotonicity "
                     "are printed, not asserted)." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeRelationsValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
