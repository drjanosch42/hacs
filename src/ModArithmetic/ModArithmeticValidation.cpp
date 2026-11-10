// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h" // If using precompiled headers
#include "ModArithmeticValidation.h"
#include "ModArithmetic.h"

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

// Reference implementation for modular exponentiation to verify mod_pow
uint32_t reference_mod_pow(uint64_t base, uint64_t exp, uint64_t p) {
    uint64_t res = 1;
    base %= p;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % p;
        base = (base * base) % p;
        exp /= 2;
    }
    return static_cast<uint32_t>(res);
}

// ---------------------------------------------------------------------------
// Stage S13 -- independent __int128 references for the 64-bit primitives
// ([C, thm:backend]). These are the trusted oracles the mod_*64 functions are
// checked against; they use __int128 directly (I5: NOT arbitrary precision).
// ---------------------------------------------------------------------------
static uint64_t ref_mul128(uint64_t a, uint64_t b, uint64_t p) {
    unsigned __int128 prod = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(prod % (unsigned __int128)p);
}
static uint64_t ref_pow128(uint64_t base, uint64_t exp, uint64_t p) {
    unsigned __int128 res = 1 % p;
    base %= p;
    while (exp > 0) {
        if (exp & 1ull) res = (res * base) % p;
        base = (uint64_t)(((unsigned __int128)base * base) % p);
        exp >>= 1;
    }
    return (uint64_t)res;
}

uint32_t mod_arithmetic_validation(int verbosity) {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;

    // <<< CHANGED: Create a list of primes to test against.
    const std::vector<uint32_t> primes_to_test = {
        23,          // Small prime
        65537,       // Large prime > 2^16
        1048583,     // Larger prime
        42992777,    // Larger prime
        1745880347   // Very large prime < 2^31
    };

    auto run_test_section = [&](const std::string& section_name) {
        if (verbosity > 0) std::cout << "\n--- Testing " << section_name << " ---" << std::endl;
    };

    auto run_test = [&](const std::string& name, bool condition) {
        total_tests++;
        if (condition) {
            passed_tests++;
            if (verbosity > 1) std::cout << "  [PASS] " << name << std::endl;
        }
        else {
            std::cout << "  [FAIL] " << name << std::endl;
        }
        return condition;
    };

    auto run_test_throws = [&](const std::string& name, auto lambda_expr) {
        total_tests++;
        bool thrown_correctly = false;
        try {
            lambda_expr();
        }
        catch (const std::runtime_error& e) {
            thrown_correctly = true;
            if (verbosity > 1) std::cout << "  [PASS] " << name << " (threw as expected: " << e.what() << ")" << std::endl;
        }
        catch (...) {
            // caught something else
        }

        if (thrown_correctly) {
            passed_tests++;
        }
        else {
            std::cout << "  [FAIL] " << name << " (did not throw std::runtime_error as expected)" << std::endl;
        }
        return thrown_correctly;
    };

    if (verbosity > 0) std::cout << "Starting mod_arithmetic_validation with multiple primes." << std::endl;

    // <<< CHANGED: Loop over all specified primes.
    for (const uint32_t p : primes_to_test) {
        if (verbosity > 0) std::cout << "\n\n=== Running tests for prime p = " << p << " ===" << std::endl;

        try {
            run_test_section("Modular Addition");
            {
                run_test("mod_add (no wrap)", ModArithmetic::mod_add(10, 5, p) == 15);
                run_test("mod_add (with wrap)", ModArithmetic::mod_add(p - 10, 20, p) == 10);
            }

            run_test_section("Modular Subtraction");
            {
                run_test("mod_sub (no wrap)", ModArithmetic::mod_sub(15, 5, p) == 10);
                run_test("mod_sub (with wrap)", ModArithmetic::mod_sub(5, 15, p) == (p - 10));
            }

            run_test_section("Modular Negation (mod_minus)");
            {
                run_test("mod_minus (non-zero)", ModArithmetic::mod_minus(12345, p) == (p - 12345));
                run_test("mod_minus (zero)", ModArithmetic::mod_minus(0, p) == 0);
            }

            run_test_section("Modular Multiplication (mod_mul)");
            {
                run_test("mod_mul (small numbers)", ModArithmetic::mod_mul(5, 6, p) == (30 % p));

                // This test remains excellent as it tests numbers close to the modulus.
                uint32_t a = p - 7;
                uint32_t b = p - 7;
                uint32_t expected = 49 % p;
                uint32_t actual = ModArithmetic::mod_mul(a, b, p);
                bool overflow_test_passed = (actual == expected);
                if (!overflow_test_passed && verbosity > 1) {
                    std::cout << "      mod_mul overflow test detail: " << a << " * " << b << " % " << p << " | Expected: " << expected << ", Got: " << actual << std::endl;
                }
                run_test("mod_mul (large numbers near p)", overflow_test_passed);
                run_test("mod_mul (by zero)", ModArithmetic::mod_mul(12345, 0, p) == 0);
            }

            run_test_section("Modular Inverse (mod_inv)");
            {
                uint32_t n = 12345 % p;
                uint32_t inv_val = ModArithmetic::mod_inv(n, p);
                run_test("mod_inv (check correctness)", ModArithmetic::mod_mul(12345, inv_val, p) == 1);
                run_test("mod_inv (of 1)", ModArithmetic::mod_inv(1, p) == 1);
                run_test_throws("mod_inv (of 0)", [&]() { ModArithmetic::mod_inv(0, p); });
            }

            run_test_section("Modular Exponentiation (mod_pow)");
            {
                uint32_t n = 12345 % p;
                run_test("mod_pow (general)", ModArithmetic::mod_pow(123, 456, p) == reference_mod_pow(123, 456, p));
                run_test("mod_pow (exp=0)", ModArithmetic::mod_pow(123, 0, p) == 1);
                run_test("mod_pow (base=0)", ModArithmetic::mod_pow(0, 123, p) == 0);
                run_test("mod_pow (base=0, exp=0)", ModArithmetic::mod_pow(0, 0, p) == 1); // by convention
                // Fermat's Little Theorem check (a^(p-1) = 1 mod p)
                run_test("mod_pow (Fermat's Little Theorem)", ModArithmetic::mod_pow(n, p - 1, p) == 1);
            }

            run_test_section("Binomial Coefficient (mod_nCr)");
            {
                // Note: The logic for these tests only works if n < p, or for specific cases of Lucas's.
                // We will test with values smaller than the smallest prime (23).
                run_test("mod_nCr (10, 3)", ModArithmetic::mod_nCr(10, 3, p) == 120 % p);
                run_test("mod_nCr (r > n)", ModArithmetic::mod_nCr(5, 10, p) == 0);
                run_test("mod_nCr (r=0)", ModArithmetic::mod_nCr(20, 0, p) == 1);
                run_test("mod_nCr (r=n)", ModArithmetic::mod_nCr(20, 20, p) == 1);
                // Lucas's Theorem case that will work for any p
                // C(p+1, 2) = C(1,0)*C(1,2) mod p = 1 * 0 = 0.
                run_test("mod_nCr (Lucas's Theorem, C(p+1, 2))", ModArithmetic::mod_nCr(p + 1, 2, p) == 0);
            }

            // Wilson's Theorem
            run_test_section("Wilson's Theorem");
            {
                // This test is computationally expensive and will be skipped for large primes.
                if (p > 1048583) {
                    if (verbosity > 1) std::cout << "  [SKIP] Wilson's Theorem for p=" << p << " (computationally expensive)" << std::endl;
                }
                else {
                    auto wilson_test_lambda = [&]() -> bool {
                        // Calculate (p-1)! mod p
                        uint64_t factorial = 1; // Use 64-bit for accumulator safety, though mod_mul keeps it small
                        for (uint32_t i = 2; i < p; ++i) {
                            factorial = ModArithmetic::mod_mul(static_cast<uint32_t>(factorial), i, p);
                        }
                        // Check if (p-1)! == p-1 (which is -1 mod p)
                        return factorial == (p - 1);
                    };
                    run_test("Verifying (p-1)! = -1 mod p", wilson_test_lambda());
                }
            }

        }
        catch (const std::exception& e) {
            std::cerr << "UNHANDLED EXCEPTION in mod_arithmetic_validation for p=" << p << ": " << e.what() << std::endl;
            // Mark as failure if an exception happens
            total_tests++;
        }
    } // <<< END of prime loop

    // =======================================================================
    // Phase 0 (matrix-engine) -- 32-bit boundary regime 2^31 <= ell < 2^32.
    //
    // Guards defect #2 (mod_inv accumulated in int32_t) and defect #33 (mod_add
    // dropped the uint32_t carry).  Before the fix BOTH returned wrong values
    // with no exception for every prime >= 2^31, e.g.
    //   mod_add(p-1, p-1, 4294967291) == 4294967284  (correct: 4294967289)
    // and mod_inv was garbage from the very first step (r = p overflows int32).
    //
    // Every uint32_t primitive is checked here against a slow reference built
    // from uint64_t / __int128 arithmetic -- an oracle that shares no code with
    // the implementation.
    // =======================================================================
    if (verbosity > 0) std::cout << "\n\n=== Running 32-bit boundary tests (2^31 <= p < 2^32) ===" << std::endl;
    try {
        // Slow, obviously-correct references (widened arithmetic, no shared code).
        auto ref_add32 = [](uint32_t a, uint32_t b, uint32_t p) -> uint32_t {
            return (uint32_t)(((uint64_t)a + (uint64_t)b) % (uint64_t)p);
        };
        auto ref_sub32 = [](uint32_t a, uint32_t b, uint32_t p) -> uint32_t {
            return (uint32_t)(((uint64_t)a + (uint64_t)p - (uint64_t)b) % (uint64_t)p);
        };
        auto ref_mul32 = [](uint32_t a, uint32_t b, uint32_t p) -> uint32_t {
            return (uint32_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)p);
        };
        auto ref_pow32 = [](uint32_t base, uint32_t e, uint32_t p) -> uint32_t {
            uint64_t res = 1, b = (uint64_t)base % p;
            while (e > 0) { if (e & 1u) res = (res * b) % p; b = (b * b) % p; e >>= 1; }
            return (uint32_t)res;
        };
        // Exact binomial in __int128, then reduced -- independent of Lucas and
        // of the Pascal recurrence used inside mod_nCr_small.
        auto ref_nCr_exact32 = [](uint32_t n, uint32_t r, uint32_t p) -> uint32_t {
            if (r > n) return 0;
            std::vector<unsigned __int128> row(n + 1, 0);
            row[0] = 1;
            for (uint32_t i = 1; i <= n; ++i)
                for (uint32_t j = i; j > 0; --j)
                    row[j] += row[j - 1];
            return (uint32_t)(row[r] % (unsigned __int128)p);
        };

        const std::vector<uint32_t> boundary_primes = {
            2147483647u, // 2^31 - 1 (Mersenne)
            2147483659u, // 2^31 + 11 -- the defect-#2 witness named in the plan
            4294967291u  // largest prime below 2^32
        };

        // Deterministic xorshift32 stream (reproducible, no <random> dependency).
        uint32_t rs = 0x1234567u;
        auto nextr32 = [&rs]() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; };

        for (const uint32_t P : boundary_primes) {
            run_test_section("boundary prime p = " + std::to_string(P));

            // --- Named single-value regressions for the two defects. ---
            run_test("mod_add(p-1, p-1) == p-2 (defect #33 carry)",
                     ModArithmetic::mod_add(P - 1, P - 1, P) == P - 2);
            run_test("mod_add(p-1, p-1) == reference",
                     ModArithmetic::mod_add(P - 1, P - 1, P) == ref_add32(P - 1, P - 1, P));
            run_test("mod_add(p-1, 1) == 0",
                     ModArithmetic::mod_add(P - 1, 1, P) == 0);
            run_test("mod_inv(p-1) == p-1 (since (-1)^2 = 1)",
                     ModArithmetic::mod_inv(P - 1, P) == P - 1);
            run_test("mod_inv(1) == 1", ModArithmetic::mod_inv(1, P) == 1);
            run_test("mod_inv(2) * 2 == 1",
                     ref_mul32(ModArithmetic::mod_inv(2, P), 2, P) == 1);
            run_test("mod_sub(0, 1) == p-1", ModArithmetic::mod_sub(0, 1, P) == P - 1);
            run_test("mod_minus(1) == p-1", ModArithmetic::mod_minus(1, P) == P - 1);
            run_test("mod_mul(p-1, p-1) == 1",
                     ModArithmetic::mod_mul(P - 1, P - 1, P) == 1);

            // --- Randomised sweep of every primitive against the references. ---
            bool add_ok = true, sub_ok = true, mul_ok = true, inv_ok = true;
            bool minus_ok = true, pow_ok = true, w64_ok = true;
            // Corner residues first, then pseudo-random ones.
            std::vector<uint32_t> corners = {0u, 1u, 2u, P / 2, P / 2 + 1, P - 2, P - 1};
            for (int i = 0; i < 40000; ++i) {
                uint32_t a, b;
                if (i < (int)(corners.size() * corners.size())) {
                    a = corners[i / corners.size()];
                    b = corners[i % corners.size()];
                } else {
                    a = (uint32_t)(nextr32() % P);
                    b = (uint32_t)(nextr32() % P);
                }
                if (ModArithmetic::mod_add(a, b, P) != ref_add32(a, b, P)) add_ok = false;
                if (ModArithmetic::mod_sub(a, b, P) != ref_sub32(a, b, P)) sub_ok = false;
                if (ModArithmetic::mod_mul(a, b, P) != ref_mul32(a, b, P)) mul_ok = false;
                if (ModArithmetic::mod_minus(a, P) != ref_sub32(0, a, P)) minus_ok = false;
                if (a != 0) {
                    const uint32_t ia = ModArithmetic::mod_inv(a, P);
                    if (ia >= P || ref_mul32(a, ia, P) != 1) inv_ok = false;
                    // Cross-check against the (independent) 64-bit backend.
                    if ((uint64_t)ia != ModArithmetic::mod_inv64((uint64_t)a, (uint64_t)P)) w64_ok = false;
                }
                if ((uint64_t)ModArithmetic::mod_add(a, b, P) !=
                    ModArithmetic::mod_add64((uint64_t)a, (uint64_t)b, (uint64_t)P)) w64_ok = false;
                if ((uint64_t)ModArithmetic::mod_sub(a, b, P) !=
                    ModArithmetic::mod_sub64((uint64_t)a, (uint64_t)b, (uint64_t)P)) w64_ok = false;
                if ((uint64_t)ModArithmetic::mod_mul(a, b, P) !=
                    ModArithmetic::mod_mul64((uint64_t)a, (uint64_t)b, (uint64_t)P)) w64_ok = false;
            }
            for (int i = 0; i < 2000; ++i) {
                const uint32_t a = 1u + (uint32_t)(nextr32() % (P - 1));
                const uint32_t e = (uint32_t)(nextr32() % (P - 1)); // < p-1: internal reduction is a no-op
                if (ModArithmetic::mod_pow(a, e, P) != ref_pow32(a, e, P)) pow_ok = false;
                // Fermat, in the form a * a^(p-2) == 1 (mod_pow reduces the
                // exponent mod p-1, so a^(p-1) would be a trivial no-op check).
                if (ref_mul32(a, ModArithmetic::mod_pow(a, P - 2, P), P) != 1) pow_ok = false;
            }
            run_test("mod_add vs uint64 reference (" + std::to_string(P) + ")", add_ok);
            run_test("mod_sub vs uint64 reference (" + std::to_string(P) + ")", sub_ok);
            run_test("mod_mul vs uint64 reference (" + std::to_string(P) + ")", mul_ok);
            run_test("mod_minus vs uint64 reference (" + std::to_string(P) + ")", minus_ok);
            run_test("mod_inv correctness a*a^-1 == 1 (" + std::to_string(P) + ")", inv_ok);
            run_test("mod_pow vs uint64 reference + Fermat (" + std::to_string(P) + ")", pow_ok);
            run_test("32-bit ops agree with 64-bit backend (" + std::to_string(P) + ")", w64_ok);
            run_test_throws("mod_inv (of 0) throws at p=" + std::to_string(P),
                            [&]() { ModArithmetic::mod_inv(0, P); });

            // --- mod_nCr / mod_nCr_small (they route through mod_add). ---
            // n <= 56 keeps the exact binomial inside __int128 while the reduced
            // Pascal entries are pseudo-random in [0, p), so the mod_add carry
            // path is exercised on essentially every step.
            bool ncr_ok = true;
            for (uint32_t n = 0; n <= 56; ++n)
                for (uint32_t r = 0; r <= n; ++r)
                    if (ModArithmetic::mod_nCr(n, r, P) != ref_nCr_exact32(n, r, P)) ncr_ok = false;
            run_test("mod_nCr vs exact __int128 binomial, n <= 56 (" + std::to_string(P) + ")", ncr_ok);

            bool ncr_edge_ok = true;
            ncr_edge_ok &= (ModArithmetic::mod_nCr(5, 10, P) == 0);
            ncr_edge_ok &= (ModArithmetic::mod_nCr(20, 0, P) == 1);
            ncr_edge_ok &= (ModArithmetic::mod_nCr(20, 20, P) == 1);
            ncr_edge_ok &= (ModArithmetic::mod_nCr(P + 1, 2, P) == 0);   // Lucas: C(1,0)*C(1,2) = 0
            ncr_edge_ok &= (ModArithmetic::mod_nCr(P + 1, 1, P) == 1);   // Lucas: C(1,0)*C(1,1) = 1
            ncr_edge_ok &= (ModArithmetic::mod_nCr(P + 3, P + 1, P) == 3); // Lucas: C(1,1)*C(3,1) = 3
            run_test("mod_nCr Lucas edge cases (" + std::to_string(P) + ")", ncr_edge_ok);

            // 64-bit twin must agree on the same inputs.
            bool ncr64_ok = true;
            for (uint32_t n = 0; n <= 40; ++n)
                for (uint32_t r = 0; r <= 40; ++r)
                    if ((uint64_t)ModArithmetic::mod_nCr(n, r, P) !=
                        ModArithmetic::mod_nCr64((uint64_t)n, (uint64_t)r, (uint64_t)P)) ncr64_ok = false;
            run_test("mod_nCr agrees with mod_nCr64 (" + std::to_string(P) + ")", ncr64_ok);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in mod_arithmetic_validation 32-bit boundary section: " << e.what() << std::endl;
        total_tests++;
    }

    // =======================================================================
    // Stage S13 -- 64-bit backend (ell < 2^64) validation ([C, thm:backend]).
    //   (a) agreement of the mod_*64 ops with the uint32_t ops on small inputs;
    //   (b) correctness of the mod_*64 ops near 2^63/2^64 against the __int128
    //       reference above.
    // =======================================================================
    if (verbosity > 0) std::cout << "\n\n=== Running 64-bit backend (ell < 2^64) tests ===" << std::endl;
    try {
        // (a) Agreement with the 32-bit ops on the small primes / small inputs.
        run_test_section("64-bit: agreement with 32-bit ops on small inputs");
        for (const uint32_t p : primes_to_test) {
            const uint64_t p64 = p;
            bool agree = true;
            const uint64_t a_list[] = {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(7), uint64_t(12345), uint64_t(p) - 1, uint64_t(p) / 2};
            const uint64_t b_list[] = {uint64_t(0), uint64_t(1), uint64_t(3), uint64_t(9999), uint64_t(p) - 1, uint64_t(p) / 3};
            for (uint64_t a : a_list) {
                for (uint64_t b : b_list) {
                    const uint32_t a32 = (uint32_t)(a % p), b32 = (uint32_t)(b % p);
                    if (ModArithmetic::mod_mul64(a % p, b % p, p64) != ModArithmetic::mod_mul(a32, b32, p)) agree = false;
                    if (ModArithmetic::mod_add64(a % p, b % p, p64) != ModArithmetic::mod_add(a32, b32, p)) agree = false;
                    if (ModArithmetic::mod_sub64(a % p, b % p, p64) != ModArithmetic::mod_sub(a32, b32, p)) agree = false;
                }
                if (a % p != 0 && ModArithmetic::mod_inv64(a % p, p64) != ModArithmetic::mod_inv((uint32_t)(a % p), p)) agree = false;
                if (ModArithmetic::mod_minus64(a % p, p64) != ModArithmetic::mod_minus((uint32_t)(a % p), p)) agree = false;
                if (ModArithmetic::mod_pow64(a % p, 456ull, p64) != ModArithmetic::mod_pow((uint32_t)(a % p), 456u, p)) agree = false;
            }
            for (uint64_t n = 0; n <= 25; ++n)
                for (uint64_t r = 0; r <= 25; ++r)
                    if (ModArithmetic::mod_nCr64(n, r, p64) != ModArithmetic::mod_nCr((uint32_t)n, (uint32_t)r, p)) agree = false;
            run_test("64-bit ops agree with 32-bit ops for p=" + std::to_string(p), agree);
        }

        // (b) Correctness near 2^63/2^64 against the __int128 reference.
        run_test_section("64-bit: correctness near 2^63/2^64 vs __int128 reference");
        // Two verified large primes (deterministic Miller-Rabin unnecessary --
        // these are standard, well-documented primes): the Mersenne prime
        // 2^61-1, and a ~2^63 prime.
        const std::vector<uint64_t> big_primes = {
            2305843009213693951ull, // 2^61 - 1 (Mersenne)
            9223372036854775783ull  // largest prime below 2^63
        };
        // xorshift64 deterministic stream so this is reproducible.
        uint64_t s = 0x9E3779B97F4A7C15ull;
        auto nextr = [&s]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
        for (uint64_t P : big_primes) {
            bool mul_ok = true, add_ok = true, sub_ok = true, inv_ok = true, pow_ok = true, fermat_ok = true;
            for (int i = 0; i < 200000; ++i) {
                const uint64_t a = nextr() % P;
                const uint64_t b = nextr() % P;
                if (ModArithmetic::mod_mul64(a, b, P) != ref_mul128(a, b, P)) mul_ok = false;
                // add/sub against __int128 (mod (a+b), (a-b)).
                if (ModArithmetic::mod_add64(a, b, P) != (uint64_t)(((unsigned __int128)a + b) % P)) add_ok = false;
                const uint64_t exp_sub = (uint64_t)(((unsigned __int128)a + P - b) % P);
                if (ModArithmetic::mod_sub64(a, b, P) != exp_sub) sub_ok = false;
                if (a != 0) {
                    const uint64_t inv = ModArithmetic::mod_inv64(a, P);
                    if (ref_mul128(a, inv, P) != 1) inv_ok = false;
                }
            }
            // Exponentiation + Fermat on a handful of bases (mod_pow64 reduces
            // exp mod P-1, so only meaningful for the a^(P-1)=1 check and vs ref).
            for (int i = 0; i < 2000; ++i) {
                const uint64_t a = 1 + nextr() % (P - 1);
                const uint64_t e = nextr() % (P - 1); // < P-1 so the internal reduction is a no-op
                if (ModArithmetic::mod_pow64(a, e, P) != ref_pow128(a, e, P)) pow_ok = false;
                if (ModArithmetic::mod_pow64(a, P - 1, P) != 1) fermat_ok = false;
            }
            run_test("mod_mul64 vs __int128 near " + std::to_string(P), mul_ok);
            run_test("mod_add64 vs __int128 near " + std::to_string(P), add_ok);
            run_test("mod_sub64 vs __int128 near " + std::to_string(P), sub_ok);
            run_test("mod_inv64 correctness near " + std::to_string(P), inv_ok);
            run_test("mod_pow64 vs __int128 near " + std::to_string(P), pow_ok);
            run_test("mod_pow64 Fermat near " + std::to_string(P), fermat_ok);
        }
        // mod_nCr64 Lucas spot-checks at a large prime.
        {
            const uint64_t P = big_primes[0];
            bool ncr_ok = true;
            ncr_ok &= (ModArithmetic::mod_nCr64(10, 3, P) == 120);
            ncr_ok &= (ModArithmetic::mod_nCr64(5, 10, P) == 0);
            ncr_ok &= (ModArithmetic::mod_nCr64(20, 0, P) == 1);
            ncr_ok &= (ModArithmetic::mod_nCr64(20, 20, P) == 1);
            ncr_ok &= (ModArithmetic::mod_nCr64(P + 1, 2, P) == 0); // Lucas: C(1,0)*C(1,2)=0
            run_test("mod_nCr64 Lucas spot-checks near 2^61", ncr_ok);
        }
        run_test_throws("mod_inv64 (of 0) throws", [&]() { ModArithmetic::mod_inv64(0, big_primes[0]); });

        // ===================================================================
        // matrix-engine Phase 1, item 7.0: mod_nCr_small no longer rebuilds a
        // Pascal row per call; for n < p it multiplies tabulated factorials by
        // tabulated inverse factorials. That is the SAME value -- these tests
        // pin it against the pre-Phase-1 Pascal DP, reproduced verbatim below,
        // wrapped in the same Lucas loop mod_nCr uses. The grid deliberately
        // includes n >= ell (so Lucas actually recurses over several base-ell
        // digits) and interleaves moduli, so a stale (n, ell) table cache would
        // show up immediately.
        // ===================================================================
        {
            // Verbatim pre-Phase-1 mod_nCr_small: Pascal DP with a fresh
            // heap row per call.
            auto dp_nCr_small = [](uint32_t n, uint32_t r, uint32_t p) -> uint32_t {
                if (r > n) return 0;
                if (r == 0 || r == n) return 1;
                if (r > n / 2) r = n - r;
                std::vector<uint32_t> C(r + 1, 0);
                C[0] = 1;
                for (uint32_t i = 1; i <= n; ++i)
                    for (uint32_t j = std::min(i, r); j > 0; --j)
                        C[j] = ModArithmetic::mod_add(C[j], C[j - 1], p);
                return C[r];
            };
            // Verbatim pre-Phase-1 mod_nCr: Lucas over the DP helper.
            auto dp_nCr = [&dp_nCr_small](uint32_t n, uint32_t r, uint32_t p) -> uint32_t {
                if (r > n) return 0;
                if (r == 0) return 1;
                uint32_t res = 1;
                while (r > 0) {
                    res = ModArithmetic::mod_mul(res, dp_nCr_small(n % p, r % p, p), p);
                    n /= p;
                    r /= p;
                }
                return res;
            };

            // Small moduli => n >= ell for most of the grid (multi-digit
            // Lucas); large moduli => the pure n < ell factorial-table path.
            const uint32_t ncr_primes[] = { 2u, 3u, 5u, 7u, 13u, 101u, 691u,
                                            134217757u, 2147483647u, 4294967291u };
            bool tab_ok = true;
            uint32_t tab_first_n = 0, tab_first_r = 0, tab_first_p = 0;
            for (uint32_t P : ncr_primes) {
                for (uint32_t n = 0; n <= 96 && tab_ok; ++n) {
                    for (uint32_t r = 0; r <= n + 2; ++r) {
                        if (ModArithmetic::mod_nCr(n, r, P) != dp_nCr(n, r, P)) {
                            tab_ok = false; tab_first_n = n; tab_first_r = r; tab_first_p = P;
                            break;
                        }
                    }
                }
            }
            run_test("mod_nCr (tabulated) == pre-Phase-1 Pascal DP, n <= 96 x 10 primes",
                     tab_ok);
            if (!tab_ok && verbosity >= 0) {
                std::cout << "      first mismatch: C(" << tab_first_n << "," << tab_first_r
                          << ") mod " << tab_first_p << std::endl;
            }

            // Large n across several base-ell digits, and the (n, ell) cache
            // interleaved between two moduli within one loop.
            bool lucas_ok = true;
            for (uint32_t P : { 2u, 3u, 5u, 7u, 691u }) {
                for (uint32_t n : { 1000u, 4096u, 65535u, 1000000u, 4294967290u }) {
                    for (uint32_t r : { 0u, 1u, 2u, 17u, 999u, 65534u }) {
                        if (r > n) continue;
                        if (ModArithmetic::mod_nCr(n, r, P) != dp_nCr(n, r, P)) lucas_ok = false;
                        // Interleave a different modulus to exercise the reset.
                        if (ModArithmetic::mod_nCr(n % 131, r % 131, 131u)
                                != dp_nCr(n % 131, r % 131, 131u)) lucas_ok = false;
                    }
                }
            }
            run_test("mod_nCr (tabulated) == Pascal DP for n >> ell, moduli interleaved",
                     lucas_ok);

            // 64-bit twin: same DP oracle, uint64_t path.
            auto dp_nCr64 = [](uint64_t n, uint64_t r, uint64_t p) -> uint64_t {
                auto small = [](uint64_t nn, uint64_t rr, uint64_t pp) -> uint64_t {
                    if (rr > nn) return 0;
                    if (rr == 0 || rr == nn) return 1;
                    if (rr > nn / 2) rr = nn - rr;
                    std::vector<uint64_t> C(rr + 1, 0);
                    C[0] = 1;
                    for (uint64_t i = 1; i <= nn; ++i)
                        for (uint64_t j = std::min<uint64_t>(i, rr); j > 0; --j)
                            C[j] = ModArithmetic::mod_add64(C[j], C[j - 1], pp);
                    return C[rr];
                };
                if (r > n) return 0;
                if (r == 0) return 1;
                uint64_t res = 1;
                while (r > 0) { res = ModArithmetic::mod_mul64(res, small(n % p, r % p, p), p); n /= p; r /= p; }
                return res;
            };
            bool tab64_ok = true;
            for (uint64_t P : { uint64_t(2), uint64_t(5), uint64_t(691),
                                uint64_t(134217757), big_primes[0] }) {
                for (uint64_t n = 0; n <= 90 && tab64_ok; ++n)
                    for (uint64_t r = 0; r <= n; ++r)
                        if (ModArithmetic::mod_nCr64(n, r, P) != dp_nCr64(n, r, P)) { tab64_ok = false; break; }
            }
            run_test("mod_nCr64 (tabulated) == pre-Phase-1 Pascal DP, n <= 90", tab64_ok);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in mod_arithmetic_validation 64-bit section: " << e.what() << std::endl;
        total_tests++;
    }


    if (verbosity >= 0) {
        std::cout << "\n\nmod_arithmetic_validation Summary:" << std::endl;
        std::cout << "  Tests run:    " << total_tests << std::endl;
        std::cout << "  Tests passed: " << passed_tests << std::endl;
        std::cout << "  Tests failed: " << (total_tests - passed_tests) << std::endl;
        if (total_tests == 0) std::cout << "  WARNING: No tests were run!" << std::endl;
    }

    return (total_tests > 0 && total_tests == passed_tests) ? 0 : 1;
}
