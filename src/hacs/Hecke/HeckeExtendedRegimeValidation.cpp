// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S13 -- extended coefficient-regime validation (ell < 2^64). See the
// header for the statement of what this validation asserts and the two verified
// primes it uses. It reuses the SAME public construction recipe as
// HeckeEndToEndValidation, but instantiated over the uint64_t element width
// (Matrix64 / FlatMatrix64 / EquivariantCohomologyOnK64 / HeckeCosetTables64 /
// HeckeTauTable64 / HeckeOperatorOnK64 / FpPoly64), so the whole pipeline runs
// on the __int128-backed 64-bit ModArithmetic backend.

#include "HeckeExtendedRegimeValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "HeckeOperatorOnK.h"
#include "FpPoly.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Deterministic Miller-Rabin, exact for all n < 2^64 (witness set
// {2,3,5,7,11,13,17,19,23,29,31,37}). Used to re-verify primality of the two
// chosen coefficient primes in-test (no embedded trust).
bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (n % p == 0) return n == p;
    }
    uint64_t d = n - 1;
    int r = 0;
    while ((d & 1) == 0) { d >>= 1; ++r; }
    auto mulmod = [](uint64_t a, uint64_t b, uint64_t m) -> uint64_t {
        return (uint64_t)(((unsigned __int128)a * b) % m);
    };
    auto powmod = [&](uint64_t a, uint64_t e, uint64_t m) -> uint64_t {
        uint64_t res = 1 % m;
        a %= m;
        while (e) { if (e & 1) res = mulmod(res, a, m); a = mulmod(a, a, m); e >>= 1; }
        return res;
    };
    for (uint64_t a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        if (a % n == 0) continue;
        uint64_t x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int i = 0; i < r - 1; ++i) {
            x = mulmod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// (x mod ell) canonical in [0, ell) for a signed integer x.
uint64_t mod_of_int(int64_t x, uint64_t ell) {
    int64_t m = x % static_cast<int64_t>(ell);
    if (m < 0) m += static_cast<int64_t>(ell);
    return static_cast<uint64_t>(m);
}

// (X - root) over F_ell, root a signed integer.
hecke::FpPoly64 XMinusInt(uint64_t ell, int64_t root) {
    return hecke::FpPoly64(ell, std::vector<uint64_t>{ mod_of_int(-root, ell), 1ull });
}

// Eisenstein eigenvalue (1 + p^{n+1}) mod ell.
uint64_t eis_mod(uint64_t p, uint32_t n, uint64_t ell) {
    const uint64_t pn1 = ModArithmetic::mod_pow64(p % ell, n + 1u, ell);
    return ModArithmetic::mod_add64(1u % ell, pn1, ell);
}

// dim S_k for even k >= 4 (= dim M_k - 1).
uint32_t dim_Sk(uint32_t k) {
    if (k < 4) return 0;
    const uint32_t mk = (k % 12 == 2) ? (k / 12) : (k / 12 + 1);
    return mk - 1;
}

uint32_t expected_dim_H1(uint32_t n) {
    if (n == 0) return 0;
    return 2u * dim_Sk(n + 2u) + 1u;
}

struct Fails {
    uint32_t count = 0;
    void fail(const std::string& msg) {
        ++count;
        std::cout << "  *** FAIL: " << msg << std::endl;
    }
};

// One (ell, n) cell: build cohomology, assert dims, then the Hecke operators
// for p in {2,3} with O1 / Eisenstein / charpoly assertions.
void run_cell(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
              uint64_t ell, uint32_t n, Fails& F) {
    std::ostringstream tag;
    tag << "ell=" << ell << " n=" << n;
    try {
        // --- base cohomology L = Sym^n(F_ell^2) over the uint64 backend ------
        auto rels = GenerateSL2ZRelationsElt<uint64_t>(K, n, ell);
        LocalSystemSimplicialComplex64 lssc(K, rels, n + 1, ell);
        EquivariantCohomologyOnK64 coh(&lssc);
        coh.compute_cohomology();
        const auto& dims = coh.get_H_eq_dimensions();
        const uint32_t h0 = dims.size() > 0 ? dims[0] : 0;
        const uint32_t h1 = dims.size() > 1 ? dims[1] : 0;
        const uint32_t h2 = dims.size() > 2 ? dims[2] : 0;

        const uint32_t exp_h0 = (n == 0) ? 1u : 0u;
        const uint32_t exp_h1 = expected_dim_H1(n);
        if (h0 != exp_h0 || h1 != exp_h1 || h2 != 0u) {
            std::ostringstream oss;
            oss << tag.str() << ": dim mismatch, got (" << h0 << "," << h1 << "," << h2
                << ") expected (" << exp_h0 << "," << exp_h1 << ",0)";
            F.fail(oss.str());
        } else {
            std::cout << "  [dims] " << tag.str() << ": (H^0,H^1,H^2)=(" << h0 << ","
                      << h1 << "," << h2 << ") OK" << std::endl;
        }

        for (uint64_t p : {2ull, 3ull}) {
            // Building tau CERTIFIES (obl:tau); compute() asserts the two-route
            // bit-identity (O12a) and (i)/(ii)/(iii). A throw is a failure.
            hecke::HeckeCosetTables64 cosets(p, /*r=*/1, ell, n);
            hecke::HeckeTauTable64 tau(K, model, cosets, ell, n);
            hecke::HeckeOperatorOnK64 op(&coh, &cosets, &tau);
            op.compute();
            const auto& A = op.get_H_matrices();

            if (n == 0) {
                // O1: A^{(0)}(T_p) = (p+1) mod ell; A^{(1)}=A^{(2)}=0x0.
                const uint64_t expected = (p + 1ull) % ell;
                if (A.size() < 1 || A[0].rows != 1 || A[0].cols != 1 || A[0].data[0][0] != expected) {
                    std::ostringstream oss;
                    oss << tag.str() << " p=" << p << ": O1 A^{(0)}(T_p) expected (" << expected
                        << "), got " << (A.size() && A[0].rows == 1 ? std::to_string(A[0].data[0][0]) : std::string("<shape>"));
                    F.fail(oss.str());
                } else {
                    std::cout << "  [O1] " << tag.str() << " p=" << p << ": A^{(0)}(T_p)=(" << expected << ") OK" << std::endl;
                }
                continue;
            }

            // A^{(1)} is the arithmetic heart.
            if (A.size() < 2 || A[1].rows != exp_h1) {
                std::ostringstream oss;
                oss << tag.str() << " p=" << p << ": A^{(1)} has wrong shape";
                F.fail(oss.str());
                continue;
            }

            if (exp_h1 == 1) {
                // Pure-Eisenstein weight (n=2, weight 4): T_p eigenvalue = 1 + p^{n+1}.
                const uint64_t expected = eis_mod(p, n, ell);
                if (A[1].data[0][0] != expected) {
                    std::ostringstream oss;
                    oss << tag.str() << " p=" << p << ": A^{(1)}(T_p) Eisenstein expected ("
                        << expected << "), got (" << A[1].data[0][0] << ")";
                    F.fail(oss.str());
                } else {
                    std::cout << "  [Eis] " << tag.str() << " p=" << p << ": A^{(1)}(T_p)=(" << expected << ") OK" << std::endl;
                }
                continue;
            }

            // n=10 (weight 12): charpoly identities (integral, reduced mod ell).
            const hecke::FpPoly64 computed = hecke::CharPolyBerkowitz(A[1]);
            int64_t ap = 0;
            switch (p) { case 2: ap = -24; break; case 3: ap = 252; break; default: ap = 0; }
            const hecke::FpPoly64 cusp = hecke::FpPoly64::pow(XMinusInt(ell, ap), 2);
            const hecke::FpPoly64 eis = XMinusInt(ell, static_cast<int64_t>(eis_mod(p, n, ell)));
            const hecke::FpPoly64 expected = cusp * eis;
            if (computed != expected) {
                std::ostringstream oss;
                oss << tag.str() << " p=" << p << ": charpoly mismatch\n"
                    << "        computed = " << computed.to_string() << "\n"
                    << "        expected = " << expected.to_string();
                F.fail(oss.str());
            } else {
                std::cout << "  [charpoly] " << tag.str() << " p=" << p
                          << ": charpoly(A^{(1)}(T_" << p << ")) = " << expected.to_string() << " OK" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        F.fail(tag.str() + ": EXCEPTION: " + e.what());
    }
}

} // namespace

uint32_t HeckeExtendedRegimeValidation() {
    try {
        std::cout << "============ HeckeExtendedRegimeValidation (ell < 2^64, S13) ============\n";
        Fails F;

        // Two verified coefficient primes, re-checked in-test.
        const uint64_t ell1 = 8589934627ull;          // ~2^34, prime
        const uint64_t ell2 = 2305843009213693951ull; // 2^61 - 1 (Mersenne), prime
        for (uint64_t ell : {ell1, ell2}) {
            if (!is_prime_u64(ell)) {
                F.fail("chosen coefficient prime " + std::to_string(ell) + " failed the in-test Miller-Rabin primality check");
            }
        }
        std::cout << "  primes: ell_1 = " << ell1 << " (~2^34), ell_2 = " << ell2
                  << " (2^61-1); both certified prime in-test.\n";

        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        for (uint64_t ell : {ell1, ell2}) {
            for (uint32_t n : {0u, 2u, 10u}) {
                run_cell(K, model, ell, n, F);
            }
        }

        std::cout << "\nHeckeExtendedRegimeValidation: aggregate failing case(s) = " << F.count << "\n";
        std::cout << "  (two-route bit-identity O12a + tau certification obl:tau exercised at 64-bit ell "
                     "inside every op.compute()/HeckeTauTable ctor above.)\n";
        return F.count;
    } catch (const std::exception& e) {
        std::cout << "HeckeExtendedRegimeValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
