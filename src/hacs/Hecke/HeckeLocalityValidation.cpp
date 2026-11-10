// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E0a -- HeckeLocalityValidation / HeckeLocalityProbe.
// See HeckeLocalityValidation.h for the arm-by-arm statement.
//
// INDEPENDENCE.  Every truth here is CONSTRUCTED (companion matrices of a
// certified irreducible pi, Kronecker products, seeded conjugations) and every
// expectation is re-derived from the construction, never read out of
// HeckeModuleDecomposition.cpp.  The one exception is deliberate: the FROZEN
// PRE-FIX ACCEPTANCE PREDICATE of arm (1) is a byte-for-byte copy of the
// shipped residue-degree witness test as it stood before E0a, kept here so that
// the arm can prove the defect existed (the BarCocycleSolverValidation
// pattern).

#include "HeckeLocalityValidation.h"

#include "FpPoly.h"
#include "HeckeModuleDecomposition.h"
#include "HeckeSettings.h"
#include "HeckeOperatorFamily.h"
#include "SL2ZTessellationTables.h"
#include "SubspaceOps.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Golden/GoldenHarness.h"

#include "../../ModArithmetic/Matrix.h"
#include "../../Engine/EngineRuntime.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <chrono>
#include <thread>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

uint32_t g_fail = 0;
uint32_t g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  *** FAIL: " << what << "\n";
    }
}

// Captures std::cout for the duration of its scope, so that the deliberate
// NOTE the bounded locality exit prints ([X] Sec. 3.1 step 3: never silent)
// can be ASSERTED ON instead of scrolling past 48 times.
class CoutCapture {
    std::ostringstream buf_;
    std::streambuf* saved_;
public:
    CoutCapture() : saved_(std::cout.rdbuf(buf_.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(saved_); }
    std::string text() const { return buf_.str(); }
    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;
};

// --------------------------------------------------------------------------
// Constructed-truth toolkit (independent of HeckeModuleDecomposition.cpp).
// --------------------------------------------------------------------------

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t mod(uint32_t m) { return static_cast<uint32_t>(next() % m); }
};

// The first monic irreducible of degree f over F_ell in the lexicographic
// sweep X^f + sum c_i X^i, c decoded from a counter base ell (low -> high).
hecke::FpPoly FirstIrreducible(uint32_t ell, uint32_t f) {
    for (uint64_t k = 0; k < 1000000ull; ++k) {
        std::vector<uint32_t> c(f + 1, 0);
        uint64_t t = k;
        for (uint32_t i = 0; i < f; ++i) { c[i] = static_cast<uint32_t>(t % ell); t /= ell; }
        if (t != 0) break;
        c[f] = 1;
        const hecke::FpPoly pi(ell, c);
        if (pi.degree() == static_cast<int64_t>(f) && hecke::CertifyIrreducible(pi) == 0) return pi;
    }
    throw std::runtime_error("FirstIrreducible: none found");
}

// Companion matrix of a monic q (subdiagonal 1's, last column -q_i).
Matrix Companion(const hecke::FpPoly& q) {
    const uint32_t k = static_cast<uint32_t>(q.degree());
    Matrix C(k, k, q.ell);
    for (uint32_t i = 1; i < k; ++i) C.data[i][i - 1] = 1;
    for (uint32_t i = 0; i < k; ++i) {
        C.data[i][k - 1] = (q.coeffs[i] == 0) ? 0u : (q.ell - q.coeffs[i]);
    }
    return C;
}

Matrix Zero(uint32_t n, uint32_t ell) { return Matrix(n, n, ell); }

Matrix BlockDiag(const std::vector<Matrix>& parts, uint32_t ell) {
    uint32_t n = 0;
    for (const Matrix& M : parts) n += M.rows;
    Matrix out(n, n, ell);
    uint32_t off = 0;
    for (const Matrix& M : parts) {
        for (uint32_t i = 0; i < M.rows; ++i)
            for (uint32_t j = 0; j < M.cols; ++j) out.data[off + i][off + j] = M.data[i][j];
        off += M.rows;
    }
    return out;
}

// [[A, I], [0, A]] -- the "alpha + eps" extension of A by F_ell[eps]/(eps^2).
Matrix NilExtend(const Matrix& A) {
    const uint32_t f = A.rows;
    Matrix out(2 * f, 2 * f, A.p);
    for (uint32_t i = 0; i < f; ++i)
        for (uint32_t j = 0; j < f; ++j) {
            out.data[i][j] = A.data[i][j];
            out.data[f + i][f + j] = A.data[i][j];
        }
    for (uint32_t i = 0; i < f; ++i) out.data[i][f + i] = 1;
    return out;
}

// S (m x m over F_ell) tensor I_f, plus I_m tensor A: realises the F_{ell^f}-
// linear operator "alpha + S" on F_{ell^f}^m as an (m f) x (m f) F_ell matrix.
Matrix AlphaPlus(const Matrix& S, const Matrix& A) {
    const uint32_t m = S.rows, f = A.rows, ell = A.p;
    Matrix out(m * f, m * f, ell);
    for (uint32_t a = 0; a < m; ++a)
        for (uint32_t b = 0; b < m; ++b)
            for (uint32_t i = 0; i < f; ++i)
                for (uint32_t j = 0; j < f; ++j) {
                    uint32_t v = 0;
                    if (a == b) v = A.data[i][j];
                    if (S.data[a][b] != 0 && i == j) {
                        v = static_cast<uint32_t>((static_cast<uint64_t>(v) + S.data[a][b]) % ell);
                    }
                    out.data[a * f + i][b * f + j] = v;
                }
    return out;
}

Matrix RandomInvertible(uint32_t n, uint32_t ell, uint64_t seed) {
    Rng rng(seed);
    for (uint32_t tries = 0; tries < 256; ++tries) {
        Matrix U(n, n, ell);
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t j = 0; j < n; ++j) U.data[i][j] = rng.mod(ell);
        Matrix copy = U;
        if (copy.is_invertible()) return U;
    }
    throw std::runtime_error("RandomInvertible: no invertible matrix found");
}

// --------------------------------------------------------------------------
// The FROZEN PRE-FIX acceptance predicate.
//
// Byte-for-byte the residue-degree witness test of HeckeModuleDecomposition's
// ComputeMultiOp as it stood BEFORE E0a: f := lcm_j deg(pi_j), and a block is
// "certified" as soon as some witness w (a single operator with deg pi_p == f,
// else a deterministic-seed combination) has a char-poly whose squarefree
// decomposition is a SINGLE irreducible factor of degree exactly f.  It says
// nothing about locality -- which is the defect.
// --------------------------------------------------------------------------
bool PreFixAccepts(const hecke::PrimaryBlock& block,
                   const std::vector<std::pair<uint64_t, Matrix>>& ops, uint32_t ell) {
    std::map<uint64_t, Matrix> Asub;
    std::map<uint64_t, uint32_t> deg;
    uint32_t f = 1;
    for (const auto& op : ops) {
        Matrix A = hecke::SolveColumnsExact(block.basis_columns, op.second * block.basis_columns,
                                            "PreFixAccepts");
        const hecke::FpPoly cp = hecke::CharPolyBerkowitz(A);
        const hecke::FpPolyFactorization fac = hecke::FactorPoly(cp);
        if (fac.factors.size() != 1) return false;
        const uint32_t fj = static_cast<uint32_t>(fac.factors[0].pi.degree());
        deg[op.first] = fj;
        // lcm
        uint32_t a = f, b = fj;
        while (b) { const uint32_t t = a % b; a = b; b = t; }
        f = (f / a) * fj;
        Asub.emplace(op.first, std::move(A));
    }
    for (const auto& op : ops) {
        if (deg.at(op.first) == f) return true;  // the trivial single-operator witness
    }
    hecke::SplitMix64 rng(0x510C0FFEE5EED042ull ^ (0x9E3779B97F4A7C15ull * (block.dim + 1u)) ^ ell);
    std::vector<std::pair<uint64_t, uint32_t>> combo;
    for (const auto& op : ops) combo.emplace_back(op.first, 1u);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        if (attempt > 0) {
            for (auto& term : combo) term.second = (ell > 1) ? (1u + rng.next_mod(ell - 1)) : 0u;
        }
        Matrix W(block.dim, block.dim, ell);
        for (const auto& term : combo) W = W + Asub.at(term.first) * term.second;
        try {
            const hecke::FpPoly cpW = hecke::CharPolyBerkowitz(W);
            const auto sfd = hecke::SquarefreeDecomposition(cpW);
            if (sfd.size() == 1 && static_cast<uint32_t>(sfd[0].first.degree()) == f &&
                hecke::CertifyIrreducible(sfd[0].first) == 0) {
                return true;
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    return false;
}

std::string PolyStr(const hecke::FpPoly& P) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < P.coeffs.size(); ++i) oss << (i ? "," : "") << P.coeffs[i];
    oss << "]";
    return oss.str();
}

// --------------------------------------------------------------------------
// Arm 1 -- the [E] Claim 3.1 non-local families.
//
// E0a proved the DEFECT: the shipped refinement emitted ONE block, the frozen
// pre-fix predicate accepted it, and the read-only CertifyLocality rejected it.
// E0b (slot 2 WP-1) puts the certificate INSIDE the loop, so the same families
// are now SPLIT.  This arm therefore runs each family three ways:
//
//  (1a) under a forced budget of ONE candidate, where the loop consumes only
//       the trivial witness t_1 (single irreducible char-poly, so it cannot
//       split) and exits BOUNDED: exactly the pre-fix block, but honestly
//       flagged `local_certified = false` with a resumable state.  The frozen
//       pre-fix predicate still ACCEPTS it and CertifyLocality still REJECTS
//       it -- so the defect and its detection are both still exhibited.
//  (1b) at the default budget, where compute() SPLITS the block into the two
//       maximal ideals, certifies both, and certify() returns 0.
//  (1c) resume: run(budget = 1) followed by resume_locality(64) equals a
//       fresh run(64) entrywise, modulo `budget_history` ([X] Sec. 1.3.1
//       R3.1 items 3-4).
// --------------------------------------------------------------------------
void ArmNonLocal() {
    std::cout << "-- arm 1: [E] Claim 3.1 non-local families (bounded exit at budget 1; "
                 "SPLIT at the default budget; resume-invariance)\n";
    uint32_t cases = 0, bounded = 0, resumes = 0;
    for (uint32_t ell : {5u, 13u, 691u}) {
        for (uint32_t f : {2u, 3u}) {
            const hecke::FpPoly pi = FirstIrreducible(ell, f);
            const Matrix C = Companion(pi);
            // r = X^ell mod pi, so r(C) represents multiplication by alpha^ell.
            const hecke::FpPoly r = hecke::FpPoly::powmod(hecke::FpPoly::X(ell), ell, pi);
            const Matrix Cfrob = hecke::EvalPolyAtMatrix(r, C);
            for (uint32_t m1 : {1u, 2u}) {
                for (uint32_t m2 : {1u, 2u}) {
                    for (bool nil : {false, true}) {
                        // t_1 = (alpha)^{m1} (+) (alpha^ell)^{m2};
                        // t_2 = (alpha)^{m1} (+) (alpha)^{m2}.
                        std::vector<Matrix> p1, p2;
                        for (uint32_t i = 0; i < m1; ++i) {
                            p1.push_back(nil ? NilExtend(C) : C);
                            p2.push_back(nil ? NilExtend(C) : C);
                        }
                        for (uint32_t i = 0; i < m2; ++i) {
                            p1.push_back(nil ? NilExtend(Cfrob) : Cfrob);
                            p2.push_back(nil ? NilExtend(C) : C);
                        }
                        Matrix t1 = BlockDiag(p1, ell), t2 = BlockDiag(p2, ell);
                        const uint32_t n = t1.rows;
                        Matrix U = RandomInvertible(n, ell, 0xC1A13100ull + 7u * ell + f);
                        Matrix Ucopy = U;
                        const Matrix Uinv = Ucopy.inverse();
                        t1 = U * t1 * Uinv;
                        t2 = U * t2 * Uinv;
                        Check((t1 * t2 - t2 * t1).is_zero(),
                              "arm1: constructed operators commute");

                        std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t1}, {3, t2}};
                        const std::string tag = "ell=" + std::to_string(ell) + " f=" +
                                                std::to_string(f) + " m=(" + std::to_string(m1) +
                                                "," + std::to_string(m2) + ") nil=" +
                                                (nil ? "1" : "0");

                        // ---- (1a) the BOUNDED EXIT at budget 1. ----
                        // Only on a small, fixed subset: the NOTE the loop
                        // prints is the point of the arm, and 48 copies of it
                        // would bury every other line.
                        const bool bounded_subset = (m1 == 1 && m2 == 1 && !nil);
                        if (bounded_subset) {
                            std::string note;
                            hecke::HeckeModuleDecomposition D1(ops, ell);
                            {
                                hecke::ScopedLocality one(hecke::LocalityMode::Budget, 1);
                                CoutCapture cap;
                                D1.compute();
                                note = cap.text();
                            }
                            Check(D1.blocks().size() == 1,
                                  "arm1(a) " + tag + ": budget 1 consumes only the trivial "
                                  "witness, so the block is NOT split (got " +
                                  std::to_string(D1.blocks().size()) + ")");
                            if (D1.blocks().size() == 1) {
                                const hecke::PrimaryBlock& b = D1.blocks()[0];
                                Check(b.residue_degree == f,
                                      "arm1(a) " + tag + ": residue degree f");
                                Check(b.residue_degree_certified,
                                      "arm1(a) " + tag + ": the RESIDUE DEGREE is still "
                                      "certified (unchanged pre-E0b semantics)");
                                Check(!b.local_certified,
                                      "arm1(a) " + tag + ": LOCALITY is NOT claimed");
                                Check(b.witness_poly.coeffs.empty() && b.character.empty(),
                                      "arm1(a) " + tag + ": a non-local block carries no "
                                      "certificate data");
                                Check(PreFixAccepts(b, ops, ell),
                                      "arm1(a) " + tag + ": the FROZEN pre-fix predicate "
                                      "ACCEPTS the block (the defect)");
                                const hecke::LocalityVerdict v =
                                    hecke::CertifyLocality(b, ops, ell);
                                Check(!v.local,
                                      "arm1(a) " + tag + ": CertifyLocality REJECTS it (code " +
                                          std::to_string(v.failure) + ")");
                                Check(v.failure == hecke::kLocalityCharacterUnsolvable ||
                                          v.failure == hecke::kLocalityNotNilpotent ||
                                          v.failure == hecke::kLocalityEmptyIntersection,
                                      "arm1(a) " + tag + ": the rejection is clause (3) of "
                                      "[E] Cert. 5.2 (code " + std::to_string(v.failure) + ")");
                            }
                            Check(note.find("LOCALITY of a dim-") != std::string::npos &&
                                      note.find("is NOT certified") != std::string::npos,
                                  "arm1(a) " + tag + ": the bounded exit is announced, never "
                                  "silent ([X] Sec. 3.1 step 3)");
                            const hecke::LocalityState& s1 = D1.locality_state();
                            Check(!s1.complete && s1.blocks.size() == 1 &&
                                      !s1.blocks[0].local && s1.blocks[0].attempts == 1 &&
                                      s1.budget == 1 &&
                                      s1.budget_history == std::vector<uint32_t>{1u},
                                  "arm1(a) " + tag + ": the LocalityState is incomplete, at "
                                  "attempt 1, with budget_history [1]");

                            // ---- (1c) RESUME to the default budget. ----
                            D1.resume_locality(hecke::kDefaultLocalityBudget);
                            hecke::HeckeModuleDecomposition Dfresh(ops, ell);
                            Dfresh.compute();
                            const hecke::LocalityState& sr = D1.locality_state();
                            const hecke::LocalityState& sf = Dfresh.locality_state();
                            Check(sr.equals(sf, /*ignore_budget_history=*/true),
                                  "arm1(c) " + tag + ": resume(64) state == fresh run(64) "
                                  "state, entrywise, modulo budget_history");
                            Check(sr.budget_history ==
                                          (std::vector<uint32_t>{1u, hecke::kDefaultLocalityBudget}) &&
                                      sf.budget_history ==
                                          (std::vector<uint32_t>{hecke::kDefaultLocalityBudget}),
                                  "arm1(c) " + tag + ": the budget histories are [1, 64] and "
                                  "[64] -- the one member that MUST differ (R3.1 item 4)");
                            Check(D1.blocks().size() == Dfresh.blocks().size() &&
                                      D1.change_of_basis() == Dfresh.change_of_basis(),
                                  "arm1(c) " + tag + ": the resumed decomposition equals the "
                                  "fresh one (blocks and change of basis)");
                            Check(D1.certify() == 0,
                                  "arm1(c) " + tag + ": the resumed decomposition certifies");
                            bool refused = false;
                            try {
                                D1.resume_locality(1);
                            } catch (const std::exception& e) {
                                refused = std::string(e.what()).find(
                                              "budget below reached state") != std::string::npos;
                            }
                            Check(refused,
                                  "arm1(c) " + tag + ": a resume below the reached budget is "
                                  "refused with the named error");
                            const std::vector<uint32_t> before = D1.locality_state().budget_history;
                            D1.resume_locality(hecke::kDefaultLocalityBudget + 8u);
                            Check(D1.locality_state().budget_history == before,
                                  "arm1(c) " + tag + ": a resume on a COMPLETE state is a "
                                  "strict no-op (the history does not move)");
                            ++bounded;
                            ++resumes;
                        }

                        // ---- (1b) the SPLIT at the default budget. ----
                        hecke::HeckeModuleDecomposition D(ops, ell);
                        D.compute();
                        Check(D.blocks().size() == 2,
                              "arm1(b) " + tag + ": compute() SPLITS the non-local block into "
                              "the two maximal ideals (got " +
                                  std::to_string(D.blocks().size()) + ")");
                        if (D.blocks().size() != 2) continue;
                        uint32_t dim_sum = 0;
                        for (const hecke::PrimaryBlock& b : D.blocks()) {
                            dim_sum += b.dim;
                            Check(b.local_certified,
                                  "arm1(b) " + tag + ": every sub-block is certified LOCAL");
                            Check(b.residue_degree_certified,
                                  "arm1(b) " + tag + ": every sub-block's residue degree is "
                                  "certified");
                            Check(static_cast<uint32_t>(b.witness_poly.degree()) ==
                                      b.residue_degree,
                                  "arm1(b) " + tag + ": pi_w has degree f on every sub-block");
                            Check(b.character.size() == 2,
                                  "arm1(b) " + tag + ": a character per operator");
                            const hecke::LocalityVerdict v = hecke::CertifyLocality(b, ops, ell);
                            Check(v.local && v.witness_poly == b.witness_poly &&
                                      v.w_hat_exponent == b.w_hat_exponent &&
                                      v.character == b.character,
                                  "arm1(b) " + tag + ": the loop's certificate == the "
                                  "read-only verdict, field for field");
                        }
                        Check(dim_sum == n, "arm1(b) " + tag + ": the sub-block dims sum to d");
                        Check(D.certify() == 0,
                              "arm1(b) " + tag + ": certify() == 0 (codes 25-29 included; got " +
                                  std::to_string(D.certify()) + ")");
                        Check(D.locality_state().complete && D.locality_state().splits >= 1,
                              "arm1(b) " + tag + ": the LocalityState is complete and records "
                              "the split");
                        // Determinism: a second run gives the same partition.
                        hecke::HeckeModuleDecomposition D2(ops, ell);
                        D2.compute();
                        Check(D2.change_of_basis() == D.change_of_basis(),
                              "arm1(b) " + tag + ": the split is deterministic");
                        ++cases;
                    }
                }
            }
        }
    }
    std::cout << "   " << cases << " non-local families split, " << bounded
              << " bounded exits, " << resumes << " resumes\n";
}
// --------------------------------------------------------------------------
// Arm 2 -- genuinely local blocks.
// --------------------------------------------------------------------------
void ArmLocal() {
    std::cout << "-- arm 2: genuinely local blocks are ACCEPTED\n";
    uint32_t cases = 0;
    for (uint32_t ell : {5u, 13u, 691u}) {
        for (uint32_t f : {1u, 2u, 3u}) {
            const hecke::FpPoly pi = FirstIrreducible(ell, f);
            const Matrix C = Companion(pi);
            const hecke::FpPoly Pexp = hecke::FpPoly::mod(hecke::FpPoly::X(ell), pi);
            const std::string tag = "ell=" + std::to_string(ell) + " f=" + std::to_string(f);

            // (a) A = F_{ell^f}[eps]/(eps^2), two operators inside F_ell[t].
            {
                const Matrix t1 = NilExtend(C);
                const Matrix t2 = t1 * t1 + t1;  // a polynomial in t1: same local algebra
                std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t1}, {3, t2}};
                hecke::HeckeModuleDecomposition D(ops, ell);
                D.compute();
                Check(D.blocks().size() == 1, "arm2(a) " + tag + ": one block");
                if (D.blocks().size() == 1) {
                    const hecke::LocalityVerdict v = hecke::CertifyLocality(D.blocks()[0], ops, ell);
                    Check(v.local, "arm2(a) " + tag + ": LOCAL (code " +
                                       std::to_string(v.failure) + " -- " + v.note + ")");
                    Check(v.residue_degree == f, "arm2(a) " + tag + ": residue degree f");
                    if (v.local) {
                        Check(v.witness_poly == pi || v.witness_poly.degree() ==
                                                          static_cast<int64_t>(f),
                              "arm2(a) " + tag + ": pi_w has degree f");
                        Check(v.character.count(2) == 1 && v.character.at(2) == Pexp,
                              "arm2(a) " + tag + ": P_2 = X mod pi_w (got " +
                                  (v.character.count(2) ? PolyStr(v.character.at(2)) : "-") + ")");
                        Check(v.character.count(3) == 1,
                              "arm2(a) " + tag + ": P_3 present");
                    }
                    // (E0b) compute() now carries the certificate itself.
                    const hecke::PrimaryBlock& b0 = D.blocks()[0];
                    Check(b0.local_certified && b0.witness_poly == v.witness_poly &&
                              b0.w_hat_exponent == v.w_hat_exponent &&
                              b0.character == v.character,
                          "arm2(a) " + tag + ": the loop's stored certificate == the "
                          "read-only verdict");
                    Check(D.certify() == 0,
                          "arm2(a) " + tag + ": certify() == 0 (codes 25-29 included)");
                    ++cases;
                }
            }
            // (b) A = F_{ell^f}[x, y]/(x, y)^2, presented by t_1 = alpha + x,
            //     t_2 = alpha + y -- a genuinely 2-generator local algebra.
            {
                Matrix X = Zero(3, ell), Y = Zero(3, ell);
                X.data[1][0] = 1;  // 1 |-> x, x |-> 0, y |-> 0
                Y.data[2][0] = 1;  // 1 |-> y, x |-> 0, y |-> 0
                const Matrix t1 = AlphaPlus(X, C);
                const Matrix t2 = AlphaPlus(Y, C);
                Check((t1 * t2 - t2 * t1).is_zero(), "arm2(b) " + tag + ": operators commute");
                std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t1}, {3, t2}};
                hecke::HeckeModuleDecomposition D(ops, ell);
                D.compute();
                Check(D.blocks().size() == 1, "arm2(b) " + tag + ": one block");
                if (D.blocks().size() == 1) {
                    const hecke::LocalityVerdict v = hecke::CertifyLocality(D.blocks()[0], ops, ell);
                    Check(v.local, "arm2(b) " + tag + ": LOCAL (code " +
                                       std::to_string(v.failure) + " -- " + v.note + ")");
                    Check(v.residue_degree == f, "arm2(b) " + tag + ": residue degree f");
                    if (v.local) {
                        Check(v.character.count(2) == 1 && v.character.at(2) == Pexp &&
                                  v.character.count(3) == 1 && v.character.at(3) == Pexp,
                              "arm2(b) " + tag + ": both characters are X mod pi_w");
                    }
                    const hecke::PrimaryBlock& b0 = D.blocks()[0];
                    Check(b0.local_certified && b0.witness_poly == v.witness_poly &&
                              b0.w_hat_exponent == v.w_hat_exponent &&
                              b0.character == v.character,
                          "arm2(b) " + tag + ": the loop's stored certificate == the "
                          "read-only verdict");
                    Check(D.certify() == 0,
                          "arm2(b) " + tag + ": certify() == 0 (codes 25-29 included)");
                    ++cases;
                }
            }
        }
    }
    std::cout << "   " << cases << " local algebras\n";
}

// --------------------------------------------------------------------------
// Arm 3 -- the [E] Claim 3.3 trap, and arm 4 -- analytic == matrix path.
// --------------------------------------------------------------------------
void ArmTrapAndAnalytic() {
    std::cout << "-- arm 3: the [E] Claim 3.3 trap (t = alpha + eps, single operator)\n";
    uint32_t traps = 0, twins = 0;
    for (uint32_t ell : {2u, 5u, 13u, 691u}) {
        for (uint32_t f : {1u, 2u, 3u}) {
            const hecke::FpPoly pi = FirstIrreducible(ell, f);
            const Matrix C = Companion(pi);
            const Matrix t = NilExtend(C);       // alpha + eps on F_{ell^f}[eps]/(eps^2)
            const uint32_t D = t.rows;           // 2f
            const std::string tag = "ell=" + std::to_string(ell) + " f=" + std::to_string(f);

            std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t}};
            hecke::HeckeModuleDecomposition Dec(ops, ell);
            Dec.compute();
            Check(Dec.blocks().size() == 1, "arm3 " + tag + ": one block");
            if (Dec.blocks().size() != 1) continue;
            const hecke::PrimaryBlock& b = Dec.blocks()[0];
            Check(b.min_poly.at(2) == hecke::FpPoly::pow(pi, 2),
                  "arm3 " + tag + ": min_V(w) = pi^2 (the trivial witness IS accepted)");
            Check(b.jordan_partition.at(2).size() == 1 && b.jordan_partition.at(2)[0] == 2,
                  "arm3 " + tag + ": Jordan partition (2) over F_{ell^f}");
            // (E0b) The k = 1 ANALYTIC path stored on the block: local for
            // free, pi_w = pi, character = X mod pi_w, and w_hat NEVER formed.
            Check(b.local_certified && b.witness_poly == pi,
                  "arm3 " + tag + ": the k = 1 block is certified local with pi_w = pi");
            Check(b.character.size() == 1 &&
                      b.character.at(2) == hecke::FpPoly::mod(hecke::FpPoly::X(ell), pi),
                  "arm3 " + tag + ": the analytic character is X mod pi_w");
            Check(Dec.certify() == 0,
                  "arm3 " + tag + ": certify() == 0 on the analytic path (codes 25-29 "
                  "re-checked arithmetically, no matrix power)");

            // The general matrix path (the analytic shortcut turned off).
            hecke::LocalityOptions forced;
            forced.force_matrix_path = true;
            const hecke::LocalityVerdict m = hecke::CertifyLocality(b, ops, ell, forced);
            Check(m.local, "arm3 " + tag + ": the trap block IS local (code " +
                               std::to_string(m.failure) + " -- " + m.note + ")");
            if (!m.local) continue;
            Check(!m.analytic, "arm3 " + tag + ": the matrix path was taken");
            Check(hecke::EvalPolyAtMatrix(m.witness_poly, m.w_hat).is_zero(),
                  "arm3 " + tag + ": pi_w(w_hat) = 0, i.e. w_hat = alpha");
            Check(!(m.w_hat - m.witness_matrix).is_zero(),
                  "arm3 " + tag + ": w_hat != w (the Teichmueller lift really moved)");
            const hecke::FpPoly Pexp = hecke::FpPoly::mod(hecke::FpPoly::X(ell), pi);
            Check(m.character.at(2) == Pexp,
                  "arm3 " + tag + ": P_t = X mod pi_w (got " + PolyStr(m.character.at(2)) + ")");
            const Matrix n_t = m.witness_matrix - hecke::EvalPolyAtMatrix(m.character.at(2), m.w_hat);
            Check(!n_t.is_zero(), "arm3 " + tag + ": n_t = t - P_t(w_hat) = eps != 0");
            Check(hecke::IsNilpotent(n_t, D), "arm3 " + tag + ": n_t is nilpotent");
            // NEGATIVE CONTROL: the v1.0 recipe evaluated P_t at w, not at
            // w_hat.  For f >= 2 the canonical character is P_t = X, so it
            // returns t - t = 0 -- it would have reported a SEMISIMPLE block
            // and lost the eps.  For f = 1 the character is the CONSTANT alpha,
            // P_t(w) = P_t(w_hat) = alpha*I, so the trap does not arise; the
            // control then asserts exactly that (the two recipes agree), which
            // is what makes f >= 2 the interesting case.
            const Matrix n_v10 =
                m.witness_matrix - hecke::EvalPolyAtMatrix(m.character.at(2), m.witness_matrix);
            if (f >= 2) {
                Check(n_v10.is_zero(),
                      "arm3 " + tag + ": the v1.0 recipe t - P_t(w) returns 0 (the defect)");
            } else {
                Check((n_v10 - n_t).is_zero(),
                      "arm3 " + tag + ": at f = 1 the character is a constant, so the v1.0 "
                                      "recipe coincides with the certificate (no trap)");
            }
            ++traps;

            // Arm 4: the k = 1 ANALYTIC path must agree with the matrix path.
            const hecke::LocalityVerdict a = hecke::CertifyLocality(b, ops, ell);
            Check(a.analytic, "arm4 " + tag + ": the analytic path was taken for k = 1");
            Check(a.w_hat.rows == 0, "arm4 " + tag + ": the analytic path never forms w_hat");
            Check(a.local == m.local && a.residue_degree == m.residue_degree &&
                      a.witness_poly == m.witness_poly && a.w_hat_exponent == m.w_hat_exponent &&
                      a.character == m.character && a.witness == m.witness,
                  "arm4 " + tag + ": analytic verdict == matrix verdict");
            Check(b.local_certified == a.local && b.witness_poly == a.witness_poly &&
                      b.w_hat_exponent == a.w_hat_exponent && b.character == a.character,
                  "arm4 " + tag + ": and == what compute() stored on the block");
            ++twins;

            // Arm 4 again on a purely semisimple single-operator block, and on
            // a height-3 one, so the comparison is not only on e = 2.
            for (uint32_t e : {1u, 3u}) {
                const Matrix te = Companion(hecke::FpPoly::pow(pi, e));
                std::vector<std::pair<uint64_t, Matrix>> ope = {{5, te}};
                hecke::HeckeModuleDecomposition De(ope, ell);
                De.compute();
                if (De.blocks().size() != 1) { Check(false, "arm4 " + tag + ": one block"); continue; }
                const hecke::LocalityVerdict va = hecke::CertifyLocality(De.blocks()[0], ope, ell);
                const hecke::LocalityVerdict vm =
                    hecke::CertifyLocality(De.blocks()[0], ope, ell, forced);
                Check(va.local && vm.local, "arm4 " + tag + " e=" + std::to_string(e) + ": local");
                Check(va.local == vm.local && va.residue_degree == vm.residue_degree &&
                          va.witness_poly == vm.witness_poly &&
                          va.w_hat_exponent == vm.w_hat_exponent && va.character == vm.character,
                      "arm4 " + tag + " e=" + std::to_string(e) + ": analytic == matrix");
                const hecke::PrimaryBlock& be = De.blocks()[0];
                Check(be.local_certified && be.witness_poly == va.witness_poly &&
                          be.w_hat_exponent == va.w_hat_exponent && be.character == va.character,
                      "arm4 " + tag + " e=" + std::to_string(e) +
                          ": == what compute() stored");
                Check(De.certify() == 0,
                      "arm4 " + tag + " e=" + std::to_string(e) + ": certify() == 0");
                ++twins;
            }
        }
    }
    std::cout << "   " << traps << " trap cases, " << twins << " analytic/matrix twins\n";
}

// --------------------------------------------------------------------------
// Arm 6 -- the mathematical settings layer ([X] Sec. 1.2 / Sec. 1.5, [OD] R1):
// parsing and validation, the canonical text and its fingerprint, the
// defaults that the golden objects pin, the named strict-mode error, and the
// thread-local propagation onto engine::tape::run_concurrently() tasks.
// --------------------------------------------------------------------------
void ArmSettings() {
    std::cout << "-- arm 6: HeckeSettings -- values, fingerprint, strict mode, propagation\n";
    using hecke::LocalityMode;
    using hecke::LocalitySetting;
    using hecke::MathSettings;

    // (a) Defaults.  The golden harness, --bench and every golden-comparing
    // validation arm run here, and the fingerprint of the defaults is 0.
    {
        const MathSettings& d = MathSettings::defaults();
        Check(d.is_default(), "arm6(a): the default MathSettings is default");
        Check(d.locality.mode == LocalityMode::Budget &&
                  d.locality.budget == hecke::kDefaultLocalityBudget,
              "arm6(a): locality defaults to budget=64 (the shipped "
              "kResidueWitnessMaxAttempts; OQ16 / slot-1 confirmation 7)");
        Check(d.canonical_text().empty() && d.fingerprint() == 0,
              "arm6(a): the all-defaults canonical text is empty and the fingerprint is 0");
        Check(hecke::math_settings().is_default(),
              "arm6(a): the thread-local settings in force are the defaults");
    }

    // (b) Parsing and validation -- the ONE validation point.
    {
        Check(hecke::ParseLocality("budget=64") == LocalitySetting(),
              "arm6(b): budget=64 parses to the default");
        Check(hecke::ParseLocality("budget=7").budget == 7,
              "arm6(b): budget=7 parses");
        Check(hecke::ParseLocality("strict").mode == LocalityMode::Strict,
              "arm6(b): strict parses and is a LEGAL value");
        for (const char* bad : {"budget=0", "budget=", "budget=x", "budget", "", "Strict",
                                "budget=-1", "budget=12345678901"}) {
            bool threw = false;
            try {
                hecke::ParseLocality(bad);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            Check(threw, std::string("arm6(b): locality = \"") + bad + "\" is rejected");
        }
        MathSettings s;
        Check(!s.set("no_such_key", "x", hecke::SettingSource::Spec),
              "arm6(b): an unknown key returns false (the caller reports a spec error)");
        Check(s.is_default(), "arm6(b): ... and leaves the settings untouched");
        Check(s.set("locality", "budget=9", hecke::SettingSource::Spec) &&
                  s.locality.budget == 9 &&
                  s.locality_source == hecke::SettingSource::Spec,
              "arm6(b): set() records the value AND its source");
        // Slot 2 registered exactly `locality`; slot-3 WP-1 (E1b-0) APPENDED
        // `u_ell`.  What this arm pins is the REGISTRY ORDER, because the
        // fingerprint is a function of it: a key is appended, never inserted,
        // once a value has been published under it.  (`u_ell`'s own arms are
        // HeckeUEllBudgetValidation's.)
        Check(MathSettings::known_key("locality") && MathSettings::known_key("u_ell"),
              "arm6(b): both registered keys are known");
        Check(!MathSettings::known_key("no_such_key"),
              "arm6(b): an unregistered key is not known");
        // Slot-3 WP-2 appended `sturm` and `sturm_unavailable`.  A key is
        // APPENDED, never inserted, so what this arm pins is the PREFIX --
        // which is exactly the property the fingerprint depends on.  The
        // later keys have their own arms (HeckeUEllBudgetValidation,
        // HeckeExtSturmSweepValidation).
        Check(MathSettings::keys().size() >= 2 && MathSettings::keys()[0] == "locality" &&
                  MathSettings::keys()[1] == "u_ell",
              "arm6(b): ... and the registry starts [locality, u_ell], in that order");
    }

    // (c) The canonical text and the fingerprint.
    {
        MathSettings a, b;
        a.set("locality", "budget=7", hecke::SettingSource::Spec);
        b.set("locality", "budget=7", hecke::SettingSource::Console);
        Check(a.canonical_text() == "locality = budget=7\n",
              "arm6(c): the canonical text lists only the NON-DEFAULT entries");
        Check(a.fingerprint() != 0 && a.fingerprint() == b.fingerprint(),
              "arm6(c): the fingerprint is a function of the VALUES, not of their source");
        MathSettings c;
        c.set("locality", "budget=8", hecke::SettingSource::Spec);
        Check(c.fingerprint() != a.fingerprint(),
              "arm6(c): a different value is a different fingerprint");
        MathSettings d;
        d.set("locality", "budget=64", hecke::SettingSource::Console);
        Check(d.fingerprint() == 0,
              "arm6(c): explicitly setting a key to its default is still the default "
              "fingerprint 0 (so a golden mismatch can never be explained by a setting)");
        MathSettings st;
        st.set("locality", "strict", hecke::SettingSource::Console);
        Check(st.fingerprint() != 0 && st.canonical_text() == "locality = strict\n",
              "arm6(c): strict is a non-default value and changes the fingerprint");
        Check(a.to_json().find("\"value\":\"budget=7\"") != std::string::npos &&
                  a.to_json().find("\"source\":\"spec\"") != std::string::npos,
              "arm6(c): to_json() carries the value and its source");
    }

    // (d) STRICT MODE is a legal value that the consumer now HONOURS
    // (slot 3 WP-3, E0b-S): it is never silently degraded to budget mode, and
    // since WP-3 it is never refused either.  The named build-capability
    // error survives for the next value a build cannot honour; no settings
    // value raises it today.  Arm 7 is where strict mode's mathematics is
    // asserted -- this is only the settings-layer half.
    {
        const uint32_t ell = 13;
        const hecke::FpPoly pi = FirstIrreducible(ell, 2);
        const Matrix C = Companion(pi);
        const Matrix t = NilExtend(C);
        for (bool multi : {false, true}) {
            std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t}};
            if (multi) ops.emplace_back(3, t * t + t);
            hecke::HeckeModuleDecomposition D(ops, ell);
            std::string msg;
            bool threw = false;
            {
                hecke::ScopedLocality strict(LocalityMode::Strict,
                                             hecke::kDefaultLocalityBudget);
                Check(hecke::math_settings().locality.mode == LocalityMode::Strict,
                      "arm6(d): the scoped override installs strict mode");
                try {
                    hecke::ResetLocalitySteps();
                    D.compute();
                } catch (const std::exception& e) {
                    threw = true;
                    msg = e.what();
                }
            }
            Check(!threw,
                  std::string("arm6(d) k=") + (multi ? "2" : "1") +
                      ": locality = strict COMPUTES (it is implemented since slot 3 WP-3; "
                      "got: " + msg + ")");
            Check(!threw && D.blocks().size() >= 1 && D.certify() == 0,
                  std::string("arm6(d) k=") + (multi ? "2" : "1") +
                      ": ... and its decomposition certifies");
            Check(hecke::math_settings().locality.mode == LocalityMode::Budget,
                  "arm6(d): the scope restored the default on the way out");
            hecke::HeckeModuleDecomposition D2(ops, ell);
            D2.compute();
            Check(D2.blocks().size() >= 1,
                  "arm6(d): the same family computes normally under the default budget mode");
            Check(!threw && D.to_json() == D2.to_json(),
                  std::string("arm6(d) k=") + (multi ? "2" : "1") +
                      ": the two modes agree BYTE FOR BYTE where budget mode certifies");
        }
        // The named error type is still the one a build-capability refusal
        // uses, and its tag is unchanged (nothing raises it today).
        Check(std::string(hecke::kNotImplementedInThisBuildTag) ==
                  "not_implemented_in_this_build:",
              "arm6(d): the build-capability error tag is unchanged");
        bool tagged = false;
        try {
            throw hecke::NotImplementedInThisBuild("probe");
        } catch (const hecke::NotImplementedInThisBuild& e) {
            tagged = std::string(e.what()).compare(
                         0, std::string(hecke::kNotImplementedInThisBuildTag).size(),
                         hecke::kNotImplementedInThisBuildTag) == 0;
        }
        Check(tagged, "arm6(d): ... and its message still begins with that tag");
    }

    // (e) PROPAGATION.  run_concurrently() installs the LAUNCHING thread's
    // settings on every task thread, and two pipelines running concurrently
    // under different `locality` cannot see each other's.
    {
        std::vector<uint32_t> seen(4, 0);
        {
            hecke::ScopedLocality seven(LocalityMode::Budget, 7);
            std::vector<std::function<void()>> tasks;
            for (int i = 0; i < 2; ++i) {
                tasks.push_back([i, &seen] {
                    seen[static_cast<size_t>(i)] = hecke::math_settings().locality.budget;
                });
            }
            engine::tape::run_concurrently(tasks);
        }
        Check(seen[0] == 7 && seen[1] == 7,
              "arm6(e): a task thread sees the launching thread's locality (got " +
                  std::to_string(seen[0]) + ", " + std::to_string(seen[1]) + ")");
        Check(hecke::math_settings().locality.budget == hecke::kDefaultLocalityBudget,
              "arm6(e): the launching thread is back at the default");

        uint32_t a_inner = 0, a_outer = 0, b_inner = 0, b_outer = 0;
        std::thread ta([&a_inner, &a_outer] {
            hecke::ScopedLocality three(LocalityMode::Budget, 3);
            std::vector<std::function<void()>> t = {[&a_inner] {
                a_inner = hecke::math_settings().locality.budget;
            }};
            engine::tape::run_concurrently(t);
            a_outer = hecke::math_settings().locality.budget;
        });
        std::thread tb([&b_inner, &b_outer] {
            hecke::ScopedLocality nine(LocalityMode::Budget, 9);
            std::vector<std::function<void()>> t = {[&b_inner] {
                b_inner = hecke::math_settings().locality.budget;
            }};
            engine::tape::run_concurrently(t);
            b_outer = hecke::math_settings().locality.budget;
        });
        ta.join();
        tb.join();
        Check(a_inner == 3 && a_outer == 3 && b_inner == 9 && b_outer == 9,
              "arm6(e): two concurrent pipelines under different locality settings do not "
              "leak into each other (got " + std::to_string(a_inner) + "/" +
                  std::to_string(a_outer) + " and " + std::to_string(b_inner) + "/" +
                  std::to_string(b_outer) + ")");
        Check(hecke::math_settings().locality.budget == hecke::kDefaultLocalityBudget,
              "arm6(e): and neither reaches the main thread");
    }
    std::cout << "   settings layer: defaults, parsing, fingerprint, strict error, "
                 "propagation\n";
}


// --------------------------------------------------------------------------
// Arm 7 -- STRICT LOCALITY MODE (hecke-extensions slot 3 WP-3, phase E0b-S;
// [X] Sec. 3.1 "Strict mode" steps 1-6, [E] Cert. `sec:localcert` "Strict
// mode" (1)-(3), [X] Sec. 3.2 arms (i)-(iv)).
//
// The four claims, in the order the plan states them:
//   (a) the (2, 30) COVERING EXAMPLE -- a local block on which EVERY linear
//       witness lies in a proper subfield, so budget mode correctly reports
//       `local = false` with `residue_degree_lower_bound = 30` and strict mode
//       certifies `local = true`, f = 30, `phase_reached = residue_search`;
//   (b) the SPLIT PATH -- on an [E] Claim 3.1 non-local family with the linear
//       family FORCED to fail (budget 1), strict mode reaches the same final
//       partition the budget loop reaches through its witnesses, by the
//       Berlekamp subalgebra (`dim B = 2`), with `phase_reached = berlekamp`;
//   (c) GRID IDENTITY -- strict mode over every golden-grid decomposition
//       gives the same four hashes and `phase_reached = linear` on every
//       block, i.e. strict and budget mode are byte-identical wherever budget
//       mode certifies;
//   (d) INTERRUPTION -- a cancellation raised by the test hook between two
//       witnesses leaves a LocalityState from which `resume_locality()`
//       completes with the same partition and certificate as an uninterrupted
//       run, entrywise, modulo `budget_history`.
// plus the settings fingerprint, the deterministic lexicographic fallback, a
// four-way tamper arm on certify()'s codes 25 / 26 / 30, and a concurrent
// budget-vs-strict propagation arm (which is what the CheckThread run
// exercises).
//
// INDEPENDENCE.  The (2, 30) datum is CONSTRUCTED here from a degree-30
// irreducible P found and CERTIFIED by FpPoly (never taken from a table), and
// every claimed degree is re-derived by MinPolyKrylov on the multiplication
// matrices, never read out of HeckeModuleDecomposition.cpp.
// --------------------------------------------------------------------------

// Multiplication-by-`a` on F_ell[X]/(P), in the basis 1, X, ..., X^{n-1}
// (column j = the coefficients of a * X^j mod P).
Matrix MultMatrix(const hecke::FpPoly& a, const hecke::FpPoly& P) {
    const uint32_t ell = P.ell;
    const uint32_t n = static_cast<uint32_t>(P.degree());
    Matrix M(n, n, ell);
    hecke::FpPoly xj = hecke::FpPoly::one(ell);
    for (uint32_t j = 0; j < n; ++j) {
        const hecke::FpPoly prod = hecke::FpPoly::mod(a * xj, P);
        for (uint32_t i = 0; i < n && i < prod.coeffs.size(); ++i) M.data[i][j] = prod.coeffs[i];
        xj = hecke::FpPoly::mod(xj * hecke::FpPoly::X(ell), P);
    }
    return M;
}

// The (2, 30) covering datum of [E] Cert. `sec:localcert` and slot-3 plan
// Sec. 3 WP-3, built WITHOUT A SEARCH exactly as the plan's audit prescribes:
//   P  -- a degree-30 irreducible over F_2, irreducibility CERTIFIED;
//   u  -- X mod P, advanced to the next element in the fixed order if needed;
//   x = u^((2^30-1)/31), y = u^((2^30-1)/7), z = u^((2^30-1)/3), each != 1,
//       hence of order 31 / 7 / 3, hence generators of F_32 / F_8 / F_4;
//   a = x + y, b = x + z, c = a + b = y + z.
// The subfield-lattice argument in the plan then gives deg min(a) = 15,
// deg min(b) = 10, deg min(c) = 6 and F_2(a, b) = F_{2^30}; this function
// only BUILDS the datum, and ArmStrict asserts every one of those numbers.
struct Covering230 {
    hecke::FpPoly P, a, b, c;
    Matrix t1, t2;      // multiplication by a and by b, 30 x 30 over F_2
    uint32_t u_index = 0;
};

Covering230 BuildCovering230() {
    const uint32_t ell = 2, n = 30;
    Covering230 out;
    out.P = FirstIrreducible(ell, n);
    const uint64_t order = (1ull << 30) - 1ull;     // |F_{2^30}^*|
    const hecke::FpPoly one = hecke::FpPoly::one(ell);
    for (uint64_t k = 2; k < 4096; ++k) {
        // u_k: coefficient vector = the base-2 digits of k, low -> high.
        std::vector<uint32_t> cf;
        for (uint64_t t = k; t != 0; t >>= 1) cf.push_back(static_cast<uint32_t>(t & 1u));
        const hecke::FpPoly u(ell, cf);
        if (u.degree() >= static_cast<int64_t>(n)) continue;
        const hecke::FpPoly x = hecke::FpPoly::powmod(u, order / 31ull, out.P);
        const hecke::FpPoly y = hecke::FpPoly::powmod(u, order / 7ull, out.P);
        const hecke::FpPoly z = hecke::FpPoly::powmod(u, order / 3ull, out.P);
        if (x == one || y == one || z == one) continue;
        out.u_index = static_cast<uint32_t>(k);
        out.a = x + y;
        out.b = x + z;
        out.c = out.a + out.b;
        out.t1 = MultMatrix(out.a, out.P);
        out.t2 = MultMatrix(out.b, out.P);
        return out;
    }
    throw std::runtime_error("BuildCovering230: no admissible u in the fixed order");
}

// The final partition, as the list of block bases in block order.
std::vector<Matrix> PartitionOf(const hecke::HeckeModuleDecomposition& D) {
    std::vector<Matrix> out;
    for (const hecke::PrimaryBlock& b : D.blocks()) out.push_back(b.basis_columns);
    return out;
}

bool SamePartitionOrdered(const std::vector<Matrix>& A, const std::vector<Matrix>& B) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) {
        if (!(A[i] == B[i])) return false;
    }
    return true;
}

bool SamePartitionUnordered(const std::vector<Matrix>& A, const std::vector<Matrix>& B) {
    if (A.size() != B.size()) return false;
    std::vector<char> used(B.size(), 0);
    for (size_t i = 0; i < A.size(); ++i) {
        bool hit = false;
        for (size_t j = 0; j < B.size() && !hit; ++j) {
            if (!used[j] && A[i] == B[j]) { used[j] = 1; hit = true; }
        }
        if (!hit) return false;
    }
    return true;
}

void ArmStrict() {
    std::cout << "-- arm 7: STRICT locality mode ([X] Sec. 3.1 steps 1-6; the (2, 30) covering "
                 "example, the split path, grid identity, interruption)\n";
    using hecke::LocalityMode;
    using hecke::LocalityPhase;

    // ======================================================================
    // (a) THE (2, 30) COVERING EXAMPLE.
    // ======================================================================
    const Covering230 cov = BuildCovering230();
    {
        Check(hecke::CertifyIrreducible(cov.P) == 0 && cov.P.degree() == 30,
              "arm7(a): P is a CERTIFIED irreducible of degree 30 over F_2 (never a table "
              "constant)");
        const Matrix tc = MultMatrix(cov.c, cov.P);
        const uint32_t da = static_cast<uint32_t>(hecke::MinPolyKrylov(cov.t1).degree());
        const uint32_t db = static_cast<uint32_t>(hecke::MinPolyKrylov(cov.t2).degree());
        const uint32_t dc = static_cast<uint32_t>(hecke::MinPolyKrylov(tc).degree());
        Check(da == 15, "arm7(a): deg min(a) = 15 (a = x + y with x in F_32, y in F_8; got " +
                            std::to_string(da) + ")");
        Check(db == 10, "arm7(a): deg min(b) = 10 (b = x + z with x in F_32, z in F_4; got " +
                            std::to_string(db) + ")");
        Check(dc == 6, "arm7(a): deg min(a + b) = 6 (c = y + z with y in F_8, z in F_4; got " +
                           std::to_string(dc) + ")");
        Check((cov.t1 * cov.t2 - cov.t2 * cov.t1).is_zero(),
              "arm7(a): the two multiplication operators commute");
        // THE POINT OF THE EXAMPLE: every non-zero linear witness over F_2 is
        // a, b or a + b, of degrees 15, 10, 6 -- all proper subfields -- while
        // the algebra they generate is all of F_{2^30}.
        uint32_t dimA = 0;
        (void)hecke::algebra_closure({cov.t1, cov.t2}, 30, 2, dimA);
        Check(dimA == 30,
              "arm7(a): dim F_2[a, b] = 30, i.e. F_2(a, b) = F_{2^30} -- the example is "
              "pointless unless the closure is everything (got " + std::to_string(dimA) + ")");
        Check(da < 30 && db < 30 && dc < 30,
              "arm7(a): NO linear witness has degree 30 -- which is exactly why the bounded "
              "linear family cannot certify this block");
    }

    const std::vector<std::pair<uint64_t, Matrix>> ops230 = {{2, cov.t1}, {3, cov.t2}};
    {
        // --- budget mode: the CORRECT budget-mode verdict, pinned so that a
        // later "fix" cannot silently change it. ---
        hecke::HeckeModuleDecomposition Db(ops230, 2);
        {
            CoutCapture cap;
            Db.compute();
        }
        Check(Db.blocks().size() == 1,
              "arm7(a): budget mode emits ONE block (no linear witness splits it; got " +
                  std::to_string(Db.blocks().size()) + ")");
        if (Db.blocks().size() == 1) {
            const hecke::PrimaryBlock& b = Db.blocks()[0];
            Check(!b.local_certified, "arm7(a): budget mode reports local = false");
            Check(b.residue_degree == 30,
                  "arm7(a): budget mode reports residue_degree_lower_bound = 30 = lcm(15, 10) "
                  "(got " + std::to_string(b.residue_degree) + ")");
            Check(!b.residue_degree_certified,
                  "arm7(a): budget mode certifies no residue degree either");
            Check(b.witness_matrix.rows == 0,
                  "arm7(a): budget mode never produces a matrix witness");
            Check(Db.certify() == 0, "arm7(a): budget mode's decomposition certifies");
            Check(!Db.locality_state().complete && !Db.locality_state().interrupted,
                  "arm7(a): the budget-mode state is incomplete and NOT interrupted");
            Check(Db.locality_state().phase_reached == LocalityPhase::Linear,
                  "arm7(a): budget mode never leaves the linear phase");
        }

        // --- strict mode: local = true, f = 30, via the residue search. ---
        hecke::HeckeModuleDecomposition Ds(ops230, 2);
        const auto t0 = std::chrono::steady_clock::now();
        {
            hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
            hecke::ResetLocalitySteps();
            Ds.compute();
        }
        const double secs = std::chrono::duration_cast<std::chrono::duration<double>>(
                                std::chrono::steady_clock::now() - t0).count();
        std::cout << "   STRICT_230 dim=30 ell=2 seconds=" << secs << "\n";
        Check(secs <= 10.0,
              "arm7(a): the (2, 30) case finishes in <= 10 s on RelWithDebInfo (acceptance (c); "
              "got " + std::to_string(secs) + " s)");
        Check(Ds.blocks().size() == 1,
              "arm7(a): strict mode emits ONE block (dim B = 1, so it never splits; got " +
                  std::to_string(Ds.blocks().size()) + ")");
        if (Ds.blocks().size() == 1) {
            const hecke::PrimaryBlock& b = Ds.blocks()[0];
            Check(b.local_certified, "arm7(a): STRICT MODE CERTIFIES local = true");
            Check(b.residue_degree == 30 && b.residue_degree_certified,
                  "arm7(a): f = 30, certified");
            Check(static_cast<uint32_t>(b.witness_poly.degree()) == 30 &&
                      hecke::CertifyIrreducible(b.witness_poly) == 0,
                  "arm7(a): pi_w is irreducible of degree 30");
            Check(b.witness_matrix.rows == 30 && b.witness_matrix.cols == 30,
                  "arm7(a): the witness is reported as a MATRIX (it is a polynomial in the "
                  "t_p, not an F_2-combination of them)");
            Check(b.residue_witness.empty(),
                  "arm7(a): ... and NOT as a linear witness");
            Check(b.phase_reached == LocalityPhase::ResidueSearch,
                  "arm7(a): phase_reached = residue_search");
            Check(b.character.size() == 2, "arm7(a): a character P_p per operator");
            // The nilpotency clause, re-derived here from the reported data
            // alone (V = F_{2^30} is a field, so every n_p must be ZERO).
            const Matrix w_hat = hecke::FrobeniusPower(b.witness_matrix, 2, b.w_hat_exponent);
            Check(hecke::EvalPolyAtMatrix(b.witness_poly, w_hat).is_zero(),
                  "arm7(a): pi_w(w_hat) = 0, re-derived");
            bool nil_ok = true;
            for (const auto& op : ops230) {
                const Matrix n_p =
                    op.second - hecke::EvalPolyAtMatrix(b.character.at(op.first), w_hat);
                if (!n_p.is_zero()) nil_ok = false;
            }
            Check(nil_ok,
                  "arm7(a): every n_p = t_p - P_p(w_hat) is ZERO -- V is a field, so the "
                  "nilpotents of [E] Cert. 5.2 clause (3) vanish identically");
            Check(Ds.certify() == 0,
                  "arm7(a): certify() accepts the strict certificate (matrix witness, code 30 "
                  "commutation included)");
        }
        Check(Ds.locality_state().complete && !Ds.locality_state().interrupted,
              "arm7(a): the strict state is COMPLETE");
        Check(Ds.locality_state().phase_reached == LocalityPhase::ResidueSearch,
              "arm7(a): the state's high-water phase is residue_search");
        Check(Ds.locality_state().mode == LocalityMode::Strict,
              "arm7(a): the state records the mode it ran in");

        // --- the lexicographic fallback is REACHABLE and deterministic, but
        // only through the test hook: on (2, 30) itself the seeded phase
        // succeeds (expected trials ~ 1). ---
        hecke::HeckeModuleDecomposition Dl(ops230, 2);
        {
            hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
            hecke::SetStrictResidueSeededTrials(0);   // force the fallback
            hecke::ResetLocalitySteps();
            Dl.compute();
            hecke::SetStrictResidueSeededTrials(hecke::kStrictSeededFromBudget);
        }
        Check(Dl.blocks().size() == 1 && Dl.blocks()[0].local_certified && Dl.certify() == 0,
              "arm7(a): the DETERMINISTIC lexicographic fallback also certifies the block");
        if (Dl.blocks().size() == 1 && Ds.blocks().size() == 1) {
            Check(!(Dl.blocks()[0].witness_matrix == Ds.blocks()[0].witness_matrix),
                  "arm7(a): ... with a DIFFERENT witness, so the arm really exercised the "
                  "fallback and not the seeded phase");
        }
    }

    // ======================================================================
    // (b) THE SPLIT PATH: dim B = 2 on an [E] Claim 3.1 family whose linear
    //     family is FORCED to fail (budget 1).  f = 3 > ell = 2 and f = 3 >
    //     ell is the "f, f' > ell" separation regime of the plan's (d).
    // ======================================================================
    {
        uint32_t split_cases = 0, ordered = 0, separation_cases = 0;
        for (uint32_t ell : {2u, 3u, 5u}) {
            for (uint32_t f : {2u, 3u, 5u}) {
                // f > ell is the SEPARATION REGIME of [E] Cert. `sec:localcert`
                // ("they separate two non-Galois-conjugate characters of
                // residue degrees f, f' whenever min(f, f') <= ell"): there
                // the linear family carries no guarantee, and the Berlekamp
                // subalgebra is the only route that always separates.
                if (f > ell) ++separation_cases;
                const hecke::FpPoly pi = FirstIrreducible(ell, f);
                const Matrix C = Companion(pi);
                const hecke::FpPoly r =
                    hecke::FpPoly::powmod(hecke::FpPoly::X(ell), ell, pi);
                const Matrix Cfrob = hecke::EvalPolyAtMatrix(r, C);
                // t_1 = alpha (+) alpha^ell, t_2 = alpha (+) alpha: two
                // distinct maximal ideals with Galois-conjugate characters.
                Matrix t1 = BlockDiag({C, Cfrob}, ell);
                Matrix t2 = BlockDiag({C, C}, ell);
                Matrix U = RandomInvertible(t1.rows, ell, 0x5719A11Cull + 11u * ell + f);
                Matrix Ucopy = U;
                const Matrix Uinv = Ucopy.inverse();
                t1 = U * t1 * Uinv;
                t2 = U * t2 * Uinv;
                const std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t1}, {3, t2}};
                const std::string tag = "ell=" + std::to_string(ell) + " f=" + std::to_string(f);

                // The reference: budget mode at the default budget, which
                // splits through one of its own witnesses.
                hecke::HeckeModuleDecomposition Db(ops, ell);
                { CoutCapture cap; Db.compute(); }
                // Strict mode with the linear family FORCED to fail.
                hecke::HeckeModuleDecomposition Ds(ops, ell);
                {
                    hecke::ScopedLocality strict(LocalityMode::Strict, 1);
                    hecke::ResetLocalitySteps();
                    CoutCapture cap;
                    Ds.compute();
                }
                ++split_cases;
                Check(Ds.blocks().size() == 2,
                      "arm7(b) " + tag + ": the Berlekamp subalgebra (dim B = 2) splits the "
                      "block into two (got " + std::to_string(Ds.blocks().size()) + ")");
                Check(Ds.locality_state().phase_reached == LocalityPhase::Berlekamp,
                      "arm7(b) " + tag + ": phase_reached = berlekamp");
                Check(Ds.locality_state().complete && Ds.certify() == 0,
                      "arm7(b) " + tag + ": both sub-blocks are certified local and certify() "
                      "returns 0");
                Check(Db.blocks().size() == Ds.blocks().size(),
                      "arm7(b) " + tag + ": the same number of blocks as the budget loop");
                const std::vector<Matrix> pb = PartitionOf(Db), ps = PartitionOf(Ds);
                const bool ord = SamePartitionOrdered(pb, ps);
                if (ord) ++ordered;
                Check(SamePartitionUnordered(pb, ps),
                      "arm7(b) " + tag + ": the FINAL PARTITION is the budget loop's, block "
                      "bases entrywise (the maximal ideals are intrinsic; only the order in "
                      "which the splitting element's factors list them can differ)");
                for (const hecke::PrimaryBlock& b : Ds.blocks()) {
                    Check(b.local_certified && b.phase_reached == LocalityPhase::Linear,
                          "arm7(b) " + tag + ": each SUB-block is decided in the linear phase "
                          "again (the strict route is entered per block, not once)");
                }
            }
        }
        Check(separation_cases >= 3,
              "arm7(b): at least three cells are in the SEPARATION REGIME f, f' > ell, where "
              "[E] Cert. `sec:localcert` gives the linear family no guarantee (got " +
                  std::to_string(separation_cases) + ")");
        std::cout << "   split path: " << split_cases << " cases (" << separation_cases
                  << " with f > ell), " << ordered
                  << " with the block ORDER identical to the budget loop's -- the order is the "
                     "CanonicalLess order of the SPLITTING ELEMENT's factors, and the two modes "
                     "split by different elements, so it is deterministic per mode but not "
                     "mode-independent (WP-3 report, finding 2)\n";
    }

    // (c) GRID IDENTITY runs in HeckeLocalitySlowValidation (ArmStrictGrid below; S24 VAL-IMPL,
    // VALSTREAM_audit_1030.md row 9): it repeats the whole arm-5 sweep under `locality = strict`.

    // ======================================================================
    // (d) INTERRUPTION + resume ([X] Sec. 1.3.1 R3.3 item 5, R3.1 item 3).
    // ======================================================================
    {
        // The reference: an uninterrupted strict run of (2, 30).
        hecke::HeckeModuleDecomposition Dref(ops230, 2);
        {
            hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
            hecke::ResetLocalitySteps();
            Dref.compute();
        }
        // 3  -- a boundary inside the LINEAR phase; 65 -- one inside the
        // RESIDUE SEARCH (the linear family of (2, 30) consumes exactly 64
        // candidates before the strict route starts).
        for (uint32_t limit : {3u, 65u}) {
            const std::string tag = "limit=" + std::to_string(limit);
            hecke::HeckeModuleDecomposition Di(ops230, 2);
            {
                hecke::ScopedLocality strict(LocalityMode::Strict,
                                             hecke::kDefaultLocalityBudget);
                hecke::ResetLocalitySteps();
                hecke::SetLocalityStepLimit(limit);
                CoutCapture cap;
                Di.compute();
                hecke::SetLocalityStepLimit(0);
                Check(cap.text().find("INTERRUPTED") != std::string::npos,
                      "arm7(d) " + tag + ": the interrupted exit is announced, never silent");
            }
            Check(Di.locality_state().interrupted && !Di.locality_state().complete,
                  "arm7(d) " + tag + ": the state is interrupted and incomplete");
            Check(!Di.blocks().empty() && !Di.blocks()[0].local_certified,
                  "arm7(d) " + tag + ": the undecided block is emitted local = false -- the "
                  "ONLY way strict mode reports one");
            Check(Di.locality_state().budget_history == std::vector<uint32_t>{
                      hecke::kDefaultLocalityBudget},
                  "arm7(d) " + tag + ": budget_history = [64] after the interrupted run");
            {
                hecke::ScopedLocality strict(LocalityMode::Strict,
                                             hecke::kDefaultLocalityBudget);
                hecke::ResetLocalitySteps();
                Di.resume_locality(hecke::kDefaultLocalityBudget);
            }
            Check(Di.locality_state().complete && !Di.locality_state().interrupted,
                  "arm7(d) " + tag + ": the resume COMPLETES and clears `interrupted`");
            Check(Di.locality_state().equals(Dref.locality_state(),
                                             /*ignore_budget_history=*/true),
                  "arm7(d) " + tag + ": the resumed state equals an uninterrupted run's, "
                  "ENTRYWISE, modulo budget_history (R3.1 item 3)");
            Check(Di.locality_state().budget_history ==
                      (std::vector<uint32_t>{hecke::kDefaultLocalityBudget,
                                             hecke::kDefaultLocalityBudget}),
                  "arm7(d) " + tag + ": ... and budget_history is the one member that DIFFERS "
                  "([64, 64] against [64])");
            Check(Di.blocks().size() == Dref.blocks().size() && Di.certify() == 0,
                  "arm7(d) " + tag + ": the resumed decomposition certifies");
            if (!Di.blocks().empty() && !Dref.blocks().empty()) {
                Check(Di.blocks()[0].witness_matrix == Dref.blocks()[0].witness_matrix &&
                          Di.blocks()[0].witness_poly == Dref.blocks()[0].witness_poly &&
                          Di.blocks()[0].character == Dref.blocks()[0].character,
                      "arm7(d) " + tag + ": the same witness and the same certificate");
            }
        }
        // A resume on a COMPLETE state is a strict no-op, and a lower budget
        // is refused with the named error (R3.1 item 2) -- re-asserted here
        // because strict mode is the first mode that resumes at an UNCHANGED
        // budget.
        hecke::HeckeModuleDecomposition Dn(ops230, 2);
        {
            hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
            hecke::ResetLocalitySteps();
            Dn.compute();
        }
        const std::vector<uint32_t> hist_before = Dn.locality_state().budget_history;
        Dn.resume_locality(hecke::kDefaultLocalityBudget);
        Check(Dn.locality_state().budget_history == hist_before,
              "arm7(d): a resume on a COMPLETE strict state is a strict no-op");
        bool refused = false;
        std::string msg;
        try {
            Dn.resume_locality(1);
        } catch (const std::exception& e) {
            refused = true;
            msg = e.what();
        }
        Check(refused && msg.find("budget below reached state") != std::string::npos,
              "arm7(d): a resume below the reached budget is refused with the NAMED error");
    }

    // ======================================================================
    // (e) THE FINGERPRINT, and agreement where budget mode certifies.
    // ======================================================================
    {
        hecke::MathSettings sb, ss;
        ss.set("locality", "strict", hecke::SettingSource::Console);
        Check(sb.fingerprint() == 0 && ss.fingerprint() != 0,
              "arm7(e): `locality = strict` is a NON-DEFAULT mathematical setting -- it changes "
              "the record's fingerprint, so a published strict result can never be confused "
              "with a golden one");
        Check(ss.canonical_text() == "locality = strict\n",
              "arm7(e): ... with the canonical text the fingerprint is taken over");
        hecke::MathSettings s7;
        s7.set("locality", "budget=7", hecke::SettingSource::Console);
        Check(ss.fingerprint() != s7.fingerprint(),
              "arm7(e): strict and budget=7 are different fingerprints");

        // On a family the linear phase decides, the two modes agree on the
        // whole decomposition text -- the in-process statement of the grid
        // arm above.
        const uint32_t ell = 13;
        const hecke::FpPoly pi = FirstIrreducible(ell, 2);
        const Matrix C = Companion(pi);
        const Matrix t = NilExtend(C);
        const std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t}, {3, t * t + t}};
        hecke::HeckeModuleDecomposition Db(ops, ell), Ds(ops, ell);
        Db.compute();
        {
            hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
            hecke::ResetLocalitySteps();
            Ds.compute();
        }
        Check(Db.to_json() == Ds.to_json(),
              "arm7(e): where budget mode certifies, strict mode's whole decomposition JSON is "
              "IDENTICAL (the state's mode is not part of it)");
        Check(Ds.locality_state().phase_reached == LocalityPhase::Linear,
              "arm7(e): ... because strict mode never leaves the linear phase there");
    }

    // ======================================================================
    // (f) THE TAMPER ARM.  Each corruption of the REPORTED data fails exactly
    //     the certificate clause it breaks, and certify() checks the residue
    //     degree (codes 20-24) before locality (25-30), so a corruption that
    //     also breaks the residue claim is caught there first -- which is
    //     itself the pinned behaviour.
    // ======================================================================
    {
        const struct { const char* what; uint32_t code; int kind; } cases[] = {
            {"the matrix witness is dropped (no witness at all)", 20, 0},
            {"BOTH a matrix and a linear witness are reported, and the linear one is "
             "re-derived first", 23, 1},
            {"BOTH are reported with the residue claim withdrawn", 25, 2},
            {"the matrix witness is replaced by a NON-COMMUTING matrix", 30, 3},
            {"w_hat_exponent is no longer a multiple of f", 26, 4},
        };
        for (const auto& tc : cases) {
            hecke::HeckeModuleDecomposition D(ops230, 2);
            {
                hecke::ScopedLocality strict(LocalityMode::Strict,
                                             hecke::kDefaultLocalityBudget);
                hecke::ResetLocalitySteps();
                D.compute();
            }
            Check(D.certify() == 0, std::string("arm7(f): the untampered strict certificate "
                                                "passes before `") + tc.what + "`");
            hecke::PrimaryBlock& b = D.blocks_for_tamper_test()[0];
            switch (tc.kind) {
                case 0: b.witness_matrix = Matrix(0, 0, 2); break;
                case 1: b.residue_witness = { {2, 1u} }; break;
                case 2:
                    b.residue_witness = { {2, 1u} };
                    b.residue_degree_certified = false;
                    break;
                case 3: {
                    Matrix bad = b.witness_matrix;
                    bad.data[0][1] = bad.data[0][1] ^ 1u;   // ell = 2
                    Check(!((bad * cov.t1) == (cov.t1 * bad)),
                          "arm7(f): ... the tampered witness really fails to commute");
                    b.witness_matrix = bad;
                    b.residue_degree_certified = false;     // reach the LOCALITY clauses
                    break;
                }
                case 4: b.w_hat_exponent = b.w_hat_exponent + 1u; break;
                default: break;
            }
            const uint32_t code = D.certify();
            Check(code == tc.code,
                  std::string("arm7(f): ") + tc.what + " -> certify() = " +
                      std::to_string(tc.code) + " (got " + std::to_string(code) + ")");
        }
    }

    // ======================================================================
    // (g) CONCURRENT budget-vs-strict propagation -- the arm the CheckThread
    //     run exercises.  The family is one the LINEAR phase decides in both
    //     modes, so neither thread writes to std::cout (two threads swapping
    //     std::cout's streambuf, or writing through it at once, is a race in
    //     the HARNESS, not in the code under test).  What is asserted is that
    //     the strict setting reaches the concurrently running pipeline, does
    //     not leak into the other one, and does not change a value there.
    // ======================================================================
    {
        const uint32_t ell = 13;
        const hecke::FpPoly pi = FirstIrreducible(ell, 2);
        const Matrix C = Companion(pi);
        const Matrix t = NilExtend(C);
        const std::vector<std::pair<uint64_t, Matrix>> ops = {{2, t}, {3, t * t + t}};
        LocalityMode mb = LocalityMode::Strict, ms = LocalityMode::Budget;
        std::string jb, js;
        uint32_t cb = 1, cs = 1;
        std::thread tb([&] {
            hecke::HeckeModuleDecomposition D(ops, ell);
            D.compute();
            mb = D.locality_state().mode;
            jb = D.to_json();
            cb = D.certify();
        });
        std::thread ts([&] {
            hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
            hecke::ResetLocalitySteps();
            hecke::HeckeModuleDecomposition D(ops, ell);
            D.compute();
            ms = D.locality_state().mode;
            js = D.to_json();
            cs = D.certify();
        });
        tb.join();
        ts.join();
        Check(mb == LocalityMode::Budget && ms == LocalityMode::Strict,
              "arm7(g): two concurrently running pipelines each see THEIR OWN `locality` "
              "(budget / strict) and neither leaks into the other");
        Check(cb == 0 && cs == 0 && jb == js,
              "arm7(g): ... and, where budget mode certifies, they compute the same bytes");
        Check(hecke::math_settings().locality.mode == LocalityMode::Budget,
              "arm7(g): and neither reaches the main thread");
    }
    std::cout << "   strict mode: (2, 30) covering example, Berlekamp split, grid identity, "
                 "interruption + resume, fingerprint, tamper, propagation\n";
}

// Arm 7 (c), the GRID IDENTITY of strict mode (moved out of ArmStrict unchanged by S24 VAL-IMPL; run by
// HeckeLocalitySlowValidation, so `--all-full` runs it once, as before).
void ArmStrictGrid() {
    std::cout << "-- arm 7 (c): STRICT locality mode, grid identity over every golden-grid decomposition\n";
    using hecke::LocalityMode;
    // ======================================================================
    // (c) GRID IDENTITY: strict mode over every golden-grid decomposition.
    // ======================================================================
    {
        hecke::ScopedLocality strict(LocalityMode::Strict, hecke::kDefaultLocalityBudget);
        const GoldenLocalitySweepResult R = GoldenLocalitySweep(/*quiet=*/true);
        std::cout << "LOCALITY_GRID_STRICT decompositions=" << R.decompositions
                  << " blocks=" << R.blocks << " nonlocal=" << R.nonlocal_blocks
                  << " witness_changes=" << R.witness_changes
                  << " loop_disagreements=" << R.loop_disagreements
                  << " nonlinear_phase=" << R.nonlinear_phase_blocks
                  << " hashes=" << R.hashes_checked << " failing=" << R.hashes_failing << "\n";
        for (const std::string& s : R.hash_failures) std::cout << "  *** " << s << "\n";
        Check(R.blocks > 0, "arm7(c): the strict sweep visited at least one block");
        Check(R.hashes_failing == 0,
              "arm7(c): under `locality = strict` ALL FOUR golden decomposition hashes still "
              "match GoldenReference.inc -- strict and budget mode are byte-identical wherever "
              "budget mode certifies");
        Check(R.nonlocal_blocks == 0 && R.loop_disagreements == 0,
              "arm7(c): every golden block is local under strict mode too");
        Check(R.nonlinear_phase_blocks == 0,
              "arm7(c): phase_reached = linear on EVERY golden block -- the terminating route "
              "of steps 1-3 is entered on no golden cell, which is what makes the two modes "
              "byte-identical there");
    }

}

}  // namespace

// ==========================================================================
// Registered entry points.
// ==========================================================================

uint32_t HeckeLocalityValidation() {
    g_fail = 0;
    g_checks = 0;
    try {
        std::cout << "================= HeckeLocalityValidation =================\n";
        ArmNonLocal();
        ArmLocal();
        ArmTrapAndAnalytic();
        ArmSettings();

        // --- arm 5: the grid arm ([X] Sec. 3.3). -------------------------
        std::cout << "-- arm 5: the grid arm -- every decomposition the golden harness builds\n";
        const GoldenLocalitySweepResult R = GoldenLocalitySweep(/*quiet=*/true);
        std::cout << "LOCALITY_GRID decompositions=" << R.decompositions
                  << " blocks=" << R.blocks << " analytic=" << R.analytic_blocks
                  << " nonlocal=" << R.nonlocal_blocks
                  << " witness_changes=" << R.witness_changes
                  << " loop_disagreements=" << R.loop_disagreements
                  << " hashes=" << R.hashes_checked << " failing=" << R.hashes_failing << " [";
        for (size_t i = 0; i < R.findings.size(); ++i) {
            std::cout << (i ? "; " : "") << R.findings[i];
        }
        std::cout << "]\n";
        for (const std::string& s : R.hash_failures) std::cout << "  *** " << s << "\n";
        Check(R.blocks > 0, "arm5: the sweep visited at least one block");
        Check(R.nonlocal_blocks == 0,
              "arm5: every block of every golden decomposition is LOCAL (see the cell list "
              "above -- a non-local cell is a FINDING for the owner, not something to work "
              "around: it decides whether E0b may move a decomposition hash)");
        Check(R.witness_changes == 0,
              "arm5: the certified witness is the shipped one on every block (so E0b's "
              "refinement loop takes no new branch)");
        Check(R.loop_disagreements == 0,
              "arm5: compute()'s OWN locality certificate -- the one the E0b refinement loop "
              "now enforces -- equals the independent read-only verdict on every block");
        Check(R.hashes_failing == 0,
              "arm5: all four golden decomposition hashes match GoldenReference.inc");
        Check(R.nonlinear_phase_blocks == 0,
              "arm5: every golden block's verdict is reached in the LINEAR witness phase");

        // --- arm 7 (slot 3 WP-3): strict mode.  It runs LAST because its
        // grid arm re-runs the sweep above under `locality = strict`. ------
        ArmStrict();

        std::cout << "HeckeLocalityValidation: " << g_checks << " checks, " << g_fail
                  << " failing\n";
        return g_fail;
    } catch (const std::exception& e) {
        std::cout << "HeckeLocalityValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return g_fail + 1;
    }
}

uint32_t HeckeLocalitySlowValidation() {
    g_fail = 0;
    g_checks = 0;
    try {
        std::cout << "================= HeckeLocalitySlowValidation =================\n";
        ArmStrictGrid();
        std::cout << "HeckeLocalitySlowValidation: " << g_checks << " checks, " << g_fail
                  << " failing\n";
        return g_fail;
    } catch (const std::exception& e) {
        std::cout << "HeckeLocalitySlowValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return g_fail + 1;
    }
}

uint32_t HeckeLocalityProbe() {
    try {
        std::cout << "================= HeckeLocalityProbe =================\n";
        // Interlock, the GoldenEmit pattern: RECORDED, NOT ASSERTED, and
        // expensive, so it must not run inside `--all`, `--all-full`, the
        // sanitizer configurations or cross_config.sh.
        const char* on = std::getenv("HECKE_LOCALITY_PROBE");
        if (!(on && std::string(on) == "1")) {
            std::cout << "[SKIP] HeckeLocalityProbe: interlocked. To run the (160, 691) probe:\n"
                         "    HECKE_LOCALITY_PROBE=1 ./hacs HeckeLocalityProbe\n"
                         "  Its verdict is RECORDED (slot-1 report / [X] Sec. 9.3), never "
                         "asserted here.\n";
            return 0;
        }
        const uint32_t n_sym = 160, ell = 691;
        const std::vector<uint64_t> primes = {2, 3, 5};

        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        LocalSystemSimplicialComplex lssc(K, rels, n_sym + 1, ell);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        const auto& dims = coh.get_H_eq_dimensions();
        std::cout << "  dims = (";
        for (size_t i = 0; i < dims.size(); ++i) std::cout << (i ? "," : "") << dims[i];
        std::cout << ")\n";
        if (dims.size() < 2 || dims[1] == 0) {
            std::cout << "LOCALITY_PROBE n=160 ell=691 blocks=0 verdict=empty-H1\n";
            return 0;
        }

        hecke::HeckeOperatorFamily fam(&coh, K, model, ell, n_sym);
        std::vector<std::pair<uint64_t, Matrix>> ops;
        for (uint64_t p : primes) ops.emplace_back(p, fam.T_p(p)[1]);

        // Per-operator block counts first (what the SHIPPED per-operator
        // refinement already separates), then the simultaneous one.
        for (const auto& op : ops) {
            std::vector<std::pair<uint64_t, Matrix>> one = {op};
            hecke::HeckeModuleDecomposition D1(one, ell);
            D1.compute();
            std::cout << "  T_" << op.first << " alone: " << D1.blocks().size() << " block(s), dims";
            for (const auto& b : D1.blocks()) std::cout << " " << b.dim << ":" << b.residue_degree;
            std::cout << "\n";
        }

        hecke::HeckeModuleDecomposition D(ops, ell);
        D.compute();
        uint32_t nonlocal = 0;
        std::ostringstream cells;
        for (size_t i = 0; i < D.blocks().size(); ++i) {
            const hecke::PrimaryBlock& b = D.blocks()[i];
            const hecke::LocalityVerdict v = hecke::CertifyLocality(b, ops, ell);
            cells << (i ? "; " : "") << "dim " << b.dim << ":f " << b.residue_degree << " -> "
                  << (v.local ? "local" : ("NONLOCAL code " + std::to_string(v.failure)));
            if (!v.local) ++nonlocal;
        }
        std::cout << "LOCALITY_PROBE n=" << n_sym << " ell=" << ell
                  << " blocks=" << D.blocks().size()
                  << " verdict=" << (nonlocal == 0 ? "all-local" : ("nonlocal:" +
                                                                   std::to_string(nonlocal)))
                  << " certify=" << D.certify() << " [" << cells.str() << "]\n";
        std::cout << "  (RECORDED, not asserted -- [Q17]; E2's HeckeExtSeparationValidation "
                     "asserts it.)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "HeckeLocalityProbe: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 0;  // recorded, not asserted
    }
}
