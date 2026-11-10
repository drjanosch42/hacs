// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S10 -- Definition-of-Done smoke validation. See the header for the
// full statement of what is checked.

#include "HeckeDecompositionSmokeValidation.h"

#include "FpPoly.h"
#include "HeckeModuleDecomposition.h"

#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using hecke::CanonicalLess;
using hecke::CertifyIrreducible;
using hecke::CharPolyBerkowitz;
using hecke::FactorPoly;
using hecke::FpPoly;
using hecke::FpPolyFactorization;
using hecke::HeckeModuleDecomposition;
using hecke::MinPolyKrylov;
using hecke::PrimaryBlock;

// Deterministic PRNG (fixed seed below) for the pseudorandom test matrices.
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t next_mod(uint32_t m) { return static_cast<uint32_t>(next() % m); }
};

constexpr uint64_t kValidationSeed = 0x510DEC0DE5EED042ull;

// Companion matrix of a monic q, LOCALLY re-implemented (the validation does
// not reach into HeckeModuleDecomposition's internals).
Matrix LocalCompanion(const FpPoly& q) {
    const uint32_t k = static_cast<uint32_t>(q.degree());
    Matrix C(k, k, q.ell);
    for (uint32_t i = 1; i < k; ++i) C.data[i][i - 1] = 1;
    for (uint32_t i = 0; i < k; ++i) C.data[i][k - 1] = ModArithmetic::mod_minus(q.coeffs[i], q.ell);
    return C;
}

// Random monic irreducible of degree f, CERTIFIED irreducible at construction
// time by the deterministic gcd/Frobenius test -- so the constructed truth is
// itself certified, independent of the factorization search.
FpPoly RandomCertifiedIrreducible(uint32_t ell, uint32_t f, SplitMix64& rng) {
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) {
        std::vector<uint32_t> c(f + 1, 0);
        for (uint32_t i = 0; i < f; ++i) c[i] = rng.next_mod(ell);
        c[f] = 1;
        if (f >= 2 && c[0] == 0) continue; // divisible by X, reducible
        FpPoly pi(ell, std::move(c));
        if (CertifyIrreducible(pi) == 0) return pi;
    }
    throw std::runtime_error("RandomCertifiedIrreducible: budget exhausted");
}

Matrix RandomInvertible(uint32_t ell, uint32_t d, SplitMix64& rng) {
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) {
        Matrix M(d, d, ell);
        for (uint32_t i = 0; i < d; ++i) {
            for (uint32_t j = 0; j < d; ++j) M.data[i][j] = rng.next_mod(ell);
        }
        Matrix copy = M;
        if (copy.is_invertible()) return M;
    }
    throw std::runtime_error("RandomInvertible: budget exhausted");
}

Matrix BlockDiag(const std::vector<Matrix>& parts, uint32_t ell) {
    uint32_t d = 0;
    for (const Matrix& part : parts) d += part.rows;
    Matrix B(d, d, ell);
    uint32_t off = 0;
    for (const Matrix& part : parts) {
        for (uint32_t i = 0; i < part.rows; ++i) {
            for (uint32_t j = 0; j < part.cols; ++j) B.data[off + i][off + j] = part.data[i][j];
        }
        off += part.rows;
    }
    return B;
}

struct TruthBlock {
    FpPoly pi;
    uint32_t multiplicity = 0;               // m = sum of chain sizes
    std::vector<uint32_t> partition;         // chain sizes, descending
};

std::string TruthToString(const std::vector<TruthBlock>& truth) {
    std::string s;
    for (const TruthBlock& t : truth) {
        s += "    pi = " + t.pi.to_string() + ", m = " + std::to_string(t.multiplicity) +
             ", partition = (";
        for (size_t i = 0; i < t.partition.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(t.partition[i]);
        }
        s += ")\n";
    }
    return s;
}

bool CheckFactorization(const FpPoly& f,
                        std::vector<std::pair<FpPoly, uint32_t>> expected,
                        const char* label) {
    FpPolyFactorization fac;
    try {
        fac = FactorPoly(f);
    } catch (const std::exception& e) {
        std::cout << "S10 smoke [" << label << "]: FactorPoly threw: " << e.what() << std::endl;
        return false;
    }
    std::sort(expected.begin(), expected.end(),
              [](const std::pair<FpPoly, uint32_t>& a, const std::pair<FpPoly, uint32_t>& b) {
                  return CanonicalLess(a.first, b.first);
              });
    bool ok = (fac.factors.size() == expected.size());
    if (ok) {
        for (size_t i = 0; i < expected.size(); ++i) {
            if (fac.factors[i].pi != expected[i].first ||
                fac.factors[i].multiplicity != expected[i].second) {
                ok = false;
                break;
            }
        }
    }
    if (!ok) {
        std::cout << "S10 smoke [" << label << "]: factor multiset mismatch for "
                  << f.to_string() << std::endl;
        for (const auto& fj : fac.factors) {
            std::cout << "    got: (" << fj.pi.to_string() << ")^" << fj.multiplicity << std::endl;
        }
    }
    return ok;
}

// --- Part (0): FpPoly corner batteries -------------------------------------

uint32_t FpPolyCornerChecks() {
    // Berkowitz: charpoly(companion(q)) == q, including ell = 2 with d = 5 > ell
    // (the division-free requirement) and ell = 5 with d = 5.
    {
        const std::vector<std::pair<uint32_t, std::vector<uint32_t>>> cases = {
            {2u, {1, 0, 1, 1, 0, 1}},   // deg 5 over F_2 (ell <= d)
            {2u, {1, 1, 0, 0, 1}},      // deg 4 over F_2
            {5u, {2, 3, 0, 1, 4, 1}},   // deg 5 over F_5 (ell == d)
            {13u, {7, 1, 12, 1}},       // deg 3 over F_13
            {134217757u, {5, 0, 0, 1}}, // deg 3 over the large prime
        };
        for (const auto& cs : cases) {
            FpPoly q(cs.first, cs.second);
            if (CharPolyBerkowitz(LocalCompanion(q)) != q) {
                std::cout << "S10 smoke [berkowitz]: charpoly(companion(q)) != q for "
                          << q.to_string() << std::endl;
                return 10;
            }
        }
    }
    // Inseparable corner (f' = 0, ell-th-root path): X^5 - 1 = (X - 1)^5 over F_5.
    if (!CheckFactorization(FpPoly(5, {4, 0, 0, 0, 0, 1}),
                            {{FpPoly(5, {4, 1}), 5u}}, "inseparable F_5")) {
        return 11;
    }
    // Inseparable + repeated over F_2: X^4 + X^2 = X^2 (X + 1)^2 (derivative 0 mod 2).
    if (!CheckFactorization(FpPoly(2, {0, 0, 1, 0, 1}),
                            {{FpPoly(2, {0, 1}), 2u}, {FpPoly(2, {1, 1}), 2u}},
                            "inseparable F_2")) {
        return 12;
    }
    // ell = 2 equal-degree (trace-map variant must split the two degree-3
    // irreducibles): (X^2+X+1)(X^3+X+1)(X^3+X^2+1).
    {
        FpPoly a(2, {1, 1, 1});
        FpPoly b(2, {1, 1, 0, 1});
        FpPoly c(2, {1, 0, 1, 1});
        if (!CheckFactorization(a * b * c, {{a, 1u}, {b, 1u}, {c, 1u}}, "trace-map F_2")) {
            return 13;
        }
    }
    // Equal-degree split over an odd prime: two distinct irreducible quadratics mod 5.
    {
        FpPoly a(5, {1, 1, 1});  // X^2+X+1, irreducible mod 5 (disc = -3 = 2, non-residue)
        FpPoly b(5, {2, 0, 1});  // X^2+2, irreducible mod 5 (-2 = 3, non-residue)
        if (!CheckFactorization(a * b, {{a, 1u}, {b, 1u}}, "equal-degree F_5")) {
            return 14;
        }
    }
    // extgcd Bezout identity: coprime pair over F_13.
    {
        FpPoly a(13, {1, 2, 0, 1});
        FpPoly b(13, {3, 1, 1});
        FpPoly u, v;
        FpPoly g = FpPoly::extgcd(a, b, u, v);
        if (g != u * a + v * b || g.degree() != 0 || !g.is_monic()) {
            std::cout << "S10 smoke [extgcd]: Bezout identity failed" << std::endl;
            return 15;
        }
    }
    // powmod consistency: X^{ell^2} mod f via frobenius ladder == via general powmod
    // over the large prime (exercises the O(log ell) squaring ladder).
    {
        FpPoly f(134217757, {3, 1, 0, 0, 1});
        FpPoly h1 = FpPoly::frobenius_powmod(1, f);
        FpPoly h2 = FpPoly::powmod(FpPoly::X(f.ell), 134217757ull, f);
        if (h1 != h2) {
            std::cout << "S10 smoke [powmod]: frobenius vs general powmod mismatch" << std::endl;
            return 16;
        }
    }
    return 0;
}

// --- Part (i): 200 constructed-truth reconstructions ------------------------

uint32_t ReconstructionBattery() {
    const std::vector<uint32_t> primes = {5u, 13u, 101u, 134217757u};
    const uint32_t kTestsPerPrime = 50; // 200 total
    const uint64_t p_label = 2;         // formal Hecke-prime label for the single operator
    SplitMix64 rng(kValidationSeed);

    uint32_t test_index = 0;
    for (uint32_t ell : primes) {
        for (uint32_t t = 0; t < kTestsPerPrime; ++t, ++test_index) {
            // ---- Construct the known block-diagonal truth. ----
            const uint32_t d = 1 + rng.next_mod(8);
            std::vector<std::pair<FpPoly, uint32_t>> chain; // (pi, s) companion blocks
            uint32_t remaining = d;
            while (remaining > 0) {
                const uint32_t f = 1 + rng.next_mod(std::min<uint32_t>(3, remaining));
                const uint32_t s = 1 + rng.next_mod(remaining / f);
                chain.emplace_back(RandomCertifiedIrreducible(ell, f, rng), s);
                remaining -= f * s;
            }
            std::vector<Matrix> parts;
            for (const auto& cs : chain) parts.push_back(LocalCompanion(FpPoly::pow(cs.first, cs.second)));
            const Matrix B = BlockDiag(parts, ell);
            const Matrix U0 = RandomInvertible(ell, d, rng);
            Matrix U0copy = U0;
            const Matrix A = U0 * B * U0copy.inverse();

            // Truth invariants: group the chain by pi (duplicate pi's across
            // chain entries MERGE into one primary block -- this deliberately
            // exercises multi-part partitions).
            std::vector<TruthBlock> truth;
            for (const auto& cs : chain) {
                TruthBlock* found = nullptr;
                for (TruthBlock& tb : truth) {
                    if (tb.pi == cs.first) { found = &tb; break; }
                }
                if (found == nullptr) {
                    truth.push_back(TruthBlock{cs.first, 0, {}});
                    found = &truth.back();
                }
                found->multiplicity += cs.second;
                found->partition.push_back(cs.second);
            }
            for (TruthBlock& tb : truth) {
                std::sort(tb.partition.begin(), tb.partition.end(), std::greater<uint32_t>());
            }
            std::sort(truth.begin(), truth.end(),
                      [](const TruthBlock& a, const TruthBlock& b) {
                          return CanonicalLess(a.pi, b.pi);
                      });

            // ---- Decompose and compare. ----
            HeckeModuleDecomposition dec({{p_label, A}}, ell);
            std::string failure;
            try {
                dec.compute();
                const std::vector<PrimaryBlock>& blocks = dec.blocks();
                if (blocks.size() != truth.size()) {
                    failure = "block count mismatch";
                } else {
                    for (size_t j = 0; j < truth.size() && failure.empty(); ++j) {
                        const TruthBlock& tb = truth[j];
                        const PrimaryBlock& rb = blocks[j];
                        const uint32_t f = static_cast<uint32_t>(tb.pi.degree());
                        const uint32_t e = tb.partition[0];
                        if (rb.dim != tb.multiplicity * f) failure = "block dim mismatch";
                        else if (rb.residue_degree != f) failure = "residue degree mismatch";
                        else if (rb.char_poly.at(p_label) != FpPoly::pow(tb.pi, tb.multiplicity))
                            failure = "char poly mismatch";
                        else if (rb.min_poly.at(p_label) != FpPoly::pow(tb.pi, e))
                            failure = "min poly mismatch";
                        else if (rb.jordan_partition.at(p_label) != tb.partition)
                            failure = "Jordan partition mismatch";
                    }
                }
                if (failure.empty() && dec.certify() != 0) {
                    failure = "certify() != 0 (code " + std::to_string(dec.certify()) + ")";
                }
                if (failure.empty()) {
                    // conjugated(p) EXACTLY block-diagonal -- re-verified here
                    // from change_of_basis(), independently of the class's check.
                    const Matrix C = dec.conjugated(p_label);
                    const Matrix& U = dec.change_of_basis();
                    Matrix Ucopy = U;
                    const Matrix Uinv = Ucopy.inverse();
                    if (!(Uinv * A * U == C)) failure = "conjugated != U^{-1} A U";
                    if (failure.empty()) {
                        uint32_t off = 0;
                        for (const PrimaryBlock& rb : dec.blocks()) {
                            for (uint32_t i = 0; i < d && failure.empty(); ++i) {
                                const bool in_block = (i >= off && i < off + rb.dim);
                                for (uint32_t jj = off; jj < off + rb.dim; ++jj) {
                                    if (!in_block && C.data[i][jj] != 0) {
                                        failure = "conjugated not exactly block-diagonal";
                                        break;
                                    }
                                }
                            }
                            off += rb.dim;
                        }
                        if (failure.empty() && off != d) failure = "block sizes do not tile d";
                    }
                    if (failure.empty() && !(U * C * Uinv == A)) {
                        failure = "reconstruction U C U^{-1} != A";
                    }
                }
                if (failure.empty()) {
                    // Independent cross-check: minpoly(A) (Krylov route) equals
                    // prod pi^{e} over the CONSTRUCTED truth.
                    FpPoly expected_min = FpPoly::one(ell);
                    for (const TruthBlock& tb : truth) {
                        expected_min = expected_min * FpPoly::pow(tb.pi, tb.partition[0]);
                    }
                    if (MinPolyKrylov(A) != expected_min) failure = "MinPolyKrylov mismatch";
                }
                if (failure.empty()) {
                    const std::string js = dec.to_json();
                    if (js.find("\"certified\":true") == std::string::npos) {
                        failure = "to_json does not report certified:true";
                    }
                }
            } catch (const std::exception& e) {
                failure = std::string("exception: ") + e.what();
            }

            if (!failure.empty()) {
                std::cout << "S10 smoke [reconstruction]: FAILED test " << test_index
                          << " (ell = " << ell << ", d = " << d
                          << ", master seed = " << kValidationSeed << "): " << failure << "\n";
                std::cout << "  constructed truth:\n" << TruthToString(truth);
                std::cout << "  recovered:\n" << dec.to_string_offset(4);
                return 20;
            }
        }
    }
    return 0;
}

// --- Part (ii): companion of (X^2+X+1)^2 over F_5 ---------------------------

uint32_t IrreducibleSquareCase() {
    const uint32_t ell = 5;
    FpPoly pi(ell, {1, 1, 1});              // X^2+X+1, irreducible mod 5
    if (CertifyIrreducible(pi) != 0) {
        std::cout << "S10 smoke [(X^2+X+1)^2/F_5]: X^2+X+1 not certified irreducible mod 5"
                  << std::endl;
        return 30;
    }
    const FpPoly q = FpPoly::pow(pi, 2);    // X^4 + 2X^3 + 3X^2 + 2X + 1
    const Matrix A = LocalCompanion(q);

    HeckeModuleDecomposition dec({{2ull, A}}, ell);
    try {
        dec.compute();
    } catch (const std::exception& e) {
        std::cout << "S10 smoke [(X^2+X+1)^2/F_5]: compute() threw: " << e.what() << std::endl;
        return 31;
    }
    const std::vector<PrimaryBlock>& blocks = dec.blocks();
    if (blocks.size() != 1 || blocks[0].dim != 4 || blocks[0].residue_degree != 2 ||
        blocks[0].jordan_partition.at(2) != std::vector<uint32_t>{2} ||
        blocks[0].min_poly.at(2) != q || blocks[0].char_poly.at(2) != q ||
        dec.certify() != 0) {
        std::cout << "S10 smoke [(X^2+X+1)^2/F_5]: expected one block, dim 4, residue degree 2, "
                     "partition (2); got:\n"
                  << dec.to_string_offset(4);
        return 32;
    }
    return 0;
}

} // namespace

uint32_t HeckeDecompositionSmokeValidation() {
    uint32_t rc = FpPolyCornerChecks();
    if (rc != 0) return rc;
    rc = IrreducibleSquareCase();
    if (rc != 0) return rc;
    rc = ReconstructionBattery();
    if (rc != 0) return rc;
    std::cout << "HeckeDecompositionSmokeValidation: FpPoly corners, the (X^2+X+1)^2/F_5 "
                 "companion case, and 200 constructed-truth reconstructions over "
                 "ell in {5, 13, 101, 134217757} all passed (all decompositions certified)."
              << std::endl;
    return 0;
}
