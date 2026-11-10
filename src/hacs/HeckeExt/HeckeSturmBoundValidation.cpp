// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1s (WP-3) -- HeckeSturmBoundValidation.
// See the header for the independence statement.  Contract references:
//   [E]  docs/latex/hecke_extensions.tex v1.3.2  -- `cl:sturm`, `def:bprime`,
//        `cl:bprime`, `cl:sturmmodl`, `cert:sturm`, Claim 1.8, Claim 8.1
//   [X]  docs/implementation/hecke_extensions_implementation_plan.md v1.5 --
//        Sec. 1.3 (SturmBoundCertificate), Sec. 1.3.1 R3 (resumable state),
//        Sec. 1.5 (sturm.cap / sturm.unavailable), Sec. 2.2 Stage 4(c), Sec. 5
//   [SP] docs/implementation/hecke_extensions_slot1_plan.md Sec. 3, WP-3

#include "HeckeSturmBoundValidation.h"

#include "HeckeSturmBound.h"

#include "../../ModArithmetic/ModArithmetic.h"
#include "../Util/ValidationCheck.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hecke_ext::SturmBound;
using hecke_ext::SturmBoundCertificate;
using hecke_ext::SturmFinal;
using hecke_ext::SturmFinality;
using hecke_ext::SturmOutcome;
using hecke_ext::SturmState;
using hecke_ext::SturmUnavailableMode;

oc_util::Checker Check("  FAIL: ", false, 40);

std::string Str(uint64_t v) { return std::to_string(v); }

// --- independent oracles ---------------------------------------------------

// EXACT integer sigma_j(m), j in {3, 5}, by a naive divisor loop.  m <= 40 and
// j <= 5, so 40^5 * 40 fits comfortably in uint64_t.
uint64_t SigmaExact(uint32_t j, uint32_t m) {
    uint64_t s = 0;
    for (uint32_t d = 1; d <= m; ++d) {
        if (m % d != 0u) continue;
        uint64_t t = 1;
        for (uint32_t i = 0; i < j; ++i) t *= d;
        s += t;
    }
    return s;
}

uint32_t ModOfSigned(int64_t v, uint32_t ell) {
    int64_t r = v % static_cast<int64_t>(ell);
    if (r < 0) r += static_cast<int64_t>(ell);
    return static_cast<uint32_t>(r);
}

// dim M_k for level 1 as #{(a, b) in Z_{>=0}^2 : 4a + 6b = k} -- an oracle
// independent of the floor(k/12) closed form.
uint32_t DimMkByMonomialCount(uint32_t k) {
    if (k % 2u != 0u) return 0u;
    uint32_t count = 0;
    for (uint32_t b = 0; 6u * b <= k; ++b) {
        if ((k - 6u * b) % 4u == 0u) ++count;
    }
    return count;
}

// tau(1..7), the published Ramanujan values.
const int64_t kTau[8] = {0, 1, -24, 252, -1472, 4830, -6048, -16744};

struct Pin {
    uint32_t k;
    uint32_t ell;
    uint32_t expected_B_prime;
    const char* note;
};

// [E] `cl:bprime`'s proof, [E] Claim 1.8's ten Kummer cells, and [X] Sec. 2.2
// Stage 4(c)'s v1.2 recomputation.
const Pin kPins[] = {
    // B' = 2
    {12,     691,      2, "cl:bprime; Kummer"},
    {34,     131,      2, "cl:bprime (F_131[T_2] is already everything)"},
    {16,    3617,      2, "Kummer"},
    {18,   43867,      2, "Kummer"},
    {20,     283,      2, "Kummer"},
    {20,     617,      2, "Kummer"},
    {22,     131,      2, "Kummer"},
    {22,     593,      2, "Kummer"},
    {24,     103,      2, "Kummer"},
    {24, 2294797,      2, "Kummer"},
    {26,  657931,      2, "Kummer"},
    {162,    691,      2, "the (160, 691) row"},
    {200,    691,      2, "the (198, 691) row"},
    // B' = 5
    {24,       5,      5, "cl:bprime; the primes <= B(k) do NOT generate"},
    {30,       5,      5, "cl:bprime"},
    {32,       5,      5, "cl:bprime"},
    {36,       5,      5, "cl:bprime"},
    {60,       5,      5, "the (58, 5) row; B' = ell here"},
    // B' = 7
    {28,       7,      7, "cl:bprime; the primes <= B(k) do NOT generate"},
    // B' = 3
    {36,       7,      3, "v1.2 recomputation"},
};

// --- arms ------------------------------------------------------------------

void ArmQExpansions() {
    const uint32_t kEll[] = {5u, 7u, 131u, 691u, 134217757u};
    const uint32_t L = 32u;
    for (uint32_t ell : kEll) {
        const std::vector<uint32_t> e4 = hecke_ext::EisensteinSeries(4u, L, ell);
        const std::vector<uint32_t> e6 = hecke_ext::EisensteinSeries(6u, L, ell);
        const std::vector<uint32_t> dl = hecke_ext::DeltaSeries(L, ell);

        Check(e4[0] == 1u % ell, "E_4 constant term at ell=" + Str(ell));
        Check(e6[0] == 1u % ell, "E_6 constant term at ell=" + Str(ell));
        Check(dl[0] == 0u,       "Delta constant term at ell=" + Str(ell));

        for (uint32_t m = 1; m < L; ++m) {
            const uint32_t want4 = ModOfSigned(240 * static_cast<int64_t>(SigmaExact(3u, m)), ell);
            const uint32_t want6 = ModOfSigned(-504 * static_cast<int64_t>(SigmaExact(5u, m)), ell);
            Check(e4[m] == want4, "E_4 coefficient " + Str(m) + " at ell=" + Str(ell));
            Check(e6[m] == want6, "E_6 coefficient " + Str(m) + " at ell=" + Str(ell));
        }
        for (uint32_t m = 1; m <= 7u; ++m) {
            Check(dl[m] == ModOfSigned(kTau[m], ell),
                  "tau(" + Str(m) + ") at ell=" + Str(ell));
        }

        // The independent identity E_4^3 - E_6^2 = 1728 Delta (usable whenever
        // 1728 is nonzero mod ell, i.e. for every ell >= 5).
        if (ell >= 5u) {
            std::vector<uint32_t> e4cube(L, 0u), e6sq(L, 0u);
            // school-book truncated products, deliberately not the unit's
            for (uint32_t i = 0; i < L; ++i) {
                for (uint32_t j = 0; i + j < L; ++j) {
                    e6sq[i + j] = ModArithmetic::mod_add(
                        e6sq[i + j], ModArithmetic::mod_mul(e6[i], e6[j], ell), ell);
                }
            }
            std::vector<uint32_t> e4sq(L, 0u);
            for (uint32_t i = 0; i < L; ++i) {
                for (uint32_t j = 0; i + j < L; ++j) {
                    e4sq[i + j] = ModArithmetic::mod_add(
                        e4sq[i + j], ModArithmetic::mod_mul(e4[i], e4[j], ell), ell);
                }
            }
            for (uint32_t i = 0; i < L; ++i) {
                for (uint32_t j = 0; i + j < L; ++j) {
                    e4cube[i + j] = ModArithmetic::mod_add(
                        e4cube[i + j], ModArithmetic::mod_mul(e4sq[i], e4[j], ell), ell);
                }
            }
            bool ok = true;
            for (uint32_t m = 0; m < L; ++m) {
                const uint32_t lhs = ModArithmetic::mod_sub(e4cube[m], e6sq[m], ell);
                const uint32_t rhs = ModArithmetic::mod_mul(1728u % ell, dl[m], ell);
                if (lhs != rhs) { ok = false; break; }
            }
            Check(ok, "E_4^3 - E_6^2 == 1728 Delta at ell=" + Str(ell));
        }
    }
}

void ArmDimensions() {
    for (uint32_t k = 0; k <= 300u; ++k) {
        const uint32_t want = DimMkByMonomialCount(k);
        Check(SturmBound::dimension_M_k(k) == want, "dim M_" + Str(k));
    }
    // B(k) and the default cap ([X] Sec. 1.5, confirmation 6).
    for (uint32_t k = 4; k <= 300u; k += 2u) {
        Check(SturmBound::weight_bound(k) == k / 12u + 1u, "B(" + Str(k) + ")");
        Check(SturmBound::default_cap(k) == 12u * (k / 12u + 1u), "default cap at k=" + Str(k));
    }
}

void ArmMillerEchelon() {
    const uint32_t kK[]   = {4u, 12u, 24u, 26u, 34u, 36u, 60u, 162u};
    const uint32_t kEll[] = {2u, 3u, 5u, 7u, 691u, 134217757u};
    for (uint32_t k : kK) {
        const uint32_t d = SturmBound::dimension_M_k(k);
        for (uint32_t ell : kEll) {
            const uint32_t L = 3u * d + 4u;
            const std::vector<std::vector<uint32_t>> basis = hecke_ext::EchelonMillerBasis(k, ell, L);
            Check(static_cast<uint32_t>(basis.size()) == d,
                  "Miller basis size at (k,ell)=(" + Str(k) + "," + Str(ell) + ")");
            bool echelon = true;
            for (uint32_t i = 0; i < d && echelon; ++i) {
                for (uint32_t m = 0; m < d; ++m) {
                    const uint32_t want = (i == m) ? (1u % ell) : 0u;
                    if (basis[i][m] != want) { echelon = false; break; }
                }
            }
            Check(echelon, "Miller basis is echelon at (k,ell)=(" + Str(k) + "," + Str(ell) + ")");

            // Stability in the precision: the coefficients do not depend on L.
            const std::vector<std::vector<uint32_t>> wider =
                hecke_ext::EchelonMillerBasis(k, ell, L + 17u);
            bool stable = true;
            for (uint32_t i = 0; i < d && stable; ++i) {
                for (uint32_t m = 0; m < L; ++m) {
                    if (basis[i][m] != wider[i][m]) { stable = false; break; }
                }
            }
            Check(stable, "Miller basis stable in L at (k,ell)=(" + Str(k) + "," + Str(ell) + ")");
        }
    }
}

// [E] Claim 8.1 seen from the modular-form side: on M_12(F_691), T_q has the
// eigenvalues sigma_11(q) (Eisenstein) and tau(q) (Delta), which are congruent
// mod 691 -- so T_q is scalar + a nonzero nilpotent for at least one q.
void ArmT_q_on_M12_691() {
    const uint32_t ell = 691u, k = 12u;
    const uint32_t d = SturmBound::dimension_M_k(k);
    Check(d == 2u, "dim M_12 == 2");
    const std::vector<std::vector<uint32_t>> miller = hecke_ext::EchelonMillerBasis(k, ell, 64u);
    bool saw_nonsemisimple = false;
    for (uint32_t q : {2u, 3u, 5u, 7u}) {
        const MatrixT<uint32_t> T = hecke_ext::HeckeOperatorOnMk(q, k, ell, miller);
        // sigma_11(q) = 1 + q^11 for prime q.
        uint64_t qp = 1;
        for (int i = 0; i < 11; ++i) qp *= q;
        const uint32_t sig11 = ModOfSigned(static_cast<int64_t>(1 + qp % 691ull), ell);
        const uint32_t tau   = ModOfSigned(kTau[q], ell);
        Check(sig11 == tau, "tau(q) == sigma_11(q) mod 691 for q=" + Str(q));

        const uint32_t trace = ModArithmetic::mod_add(T.data[0][0], T.data[1][1], ell);
        const uint32_t det   = ModArithmetic::mod_sub(
            ModArithmetic::mod_mul(T.data[0][0], T.data[1][1], ell),
            ModArithmetic::mod_mul(T.data[0][1], T.data[1][0], ell), ell);
        Check(trace == ModArithmetic::mod_add(sig11, tau, ell),
              "trace T_" + Str(q) + " on M_12(F_691)");
        Check(det == ModArithmetic::mod_mul(sig11, tau, ell),
              "det T_" + Str(q) + " on M_12(F_691)");

        // N = T_q - sigma_11(q) I is nilpotent; it is nonzero for some q.
        MatrixT<uint32_t> N = T;
        N.data[0][0] = ModArithmetic::mod_sub(N.data[0][0], sig11, ell);
        N.data[1][1] = ModArithmetic::mod_sub(N.data[1][1], sig11, ell);
        const MatrixT<uint32_t> N2 = N * N;
        Check(N2.is_zero(), "(T_q - sigma_11(q) I)^2 == 0 at q=" + Str(q));
        if (!N.is_zero()) saw_nonsemisimple = true;
    }
    Check(saw_nonsemisimple, "T_q is non-semisimple on M_12(F_691) for some q <= 7");
}

// PERF1 audit R1: the incremental closure's basis equals the from-scratch one entrywise
// (RREF uniqueness made executable): closure_basis == algebra_closure(T_q, from the identity).
void CheckClosureFromScratch(const SturmBound& sb, const std::string& at) {
    const SturmState& st = sb.state();
    uint32_t dim = 0;
    const auto scratch = hecke::algebra_closure(st.tq_cache, st.dim_M_k, st.ell, dim);
    bool same = (scratch.rows == st.closure_basis.size()) && dim == st.closure_dim;
    for (uint32_t i = 0; same && i < scratch.rows; ++i) same = (scratch.data[i] == st.closure_basis[i]);
    Check(same, "closure_basis == algebra_closure(T_q) from scratch, entrywise, at " + at);
}

double RunPin(const Pin& pin) {
    const auto t0 = std::chrono::steady_clock::now();
    SturmBound sb(pin.k, pin.ell);
    const SturmOutcome outcome = sb.run();
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    const std::string at = "(k,ell)=(" + Str(pin.k) + "," + Str(pin.ell) + ")";
    Check(outcome == SturmOutcome::Found, "B' found at " + at);
    if (outcome != SturmOutcome::Found) return secs;
    Check(sb.B_prime() == pin.expected_B_prime,
          "B' == " + Str(pin.expected_B_prime) + " at " + at + " [" + pin.note +
          "], got " + Str(sb.B_prime()));
    Check(sb.closure_dim() == sb.dim_M_k(), "closure fills M_k at " + at);
    CheckClosureFromScratch(sb, at);
    Check(sb.cap() == SturmBound::default_cap(pin.k), "default cap in force at " + at);

    const SturmBoundCertificate cert = sb.certificate();
    const uint32_t code = cert.verify();
    Check(code == 0u, "certificate verifies at " + at + " (code " + Str(code) + ": " +
                          SturmBoundCertificate::verify_message(code) + ")");

    // A tampered certificate must fail: (i) a dropped T_q, (ii) closure_dim--.
    if (cert.primes.size() >= 2u) {
        SturmBoundCertificate dropped = cert;
        dropped.primes.pop_back();
        dropped.T_q.pop_back();
        Check(dropped.verify() != 0u, "a dropped T_q fails verify() at " + at);
    }
    {
        SturmBoundCertificate bent = cert;
        bent.closure_dim = (bent.closure_dim == 0u) ? 1u : bent.closure_dim - 1u;
        Check(bent.verify() != 0u, "a decremented closure_dim fails verify() at " + at);
    }
    return secs;
}

void ArmPinnedBounds() {
    double worst = 0.0;
    uint32_t worst_k = 0, worst_ell = 0;
    for (const Pin& pin : kPins) {
        const double secs = RunPin(pin);
        if (secs > worst) { worst = secs; worst_k = pin.k; worst_ell = pin.ell; }
    }
    // The k = 258 end of [X] Sec. 6.2 E1 acceptance (d) ("the Sturm bound
    // <= 0.1 s at n <= 256"), recorded with the rest.
    {
        const auto t0 = std::chrono::steady_clock::now();
        SturmBound sb(258u, 691u);
        sb.run();
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "  STURM k=258 ell=691 dim_M_k=" << sb.dim_M_k()
                  << " outcome=" << hecke_ext::ToString(sb.outcome())
                  << " B'=" << sb.B_prime() << " time=" << secs << " s (recorded)\n";
        if (secs > worst) { worst = secs; worst_k = 258u; worst_ell = 691u; }
    }
    std::cout << "  STURM slowest pinned case: (k,ell)=(" << worst_k << "," << worst_ell
              << ") at " << worst << " s\n";
    // A coarse floor in the style of PerformanceSmokeValidation: the acceptance
    // target is 0.1 s per case, asserted here with 10x headroom so a loaded or
    // sanitised box cannot fail on timing alone.
    Check(worst <= 1.0, "the slowest Sturm bound is under the coarse 1.0 s floor");
}

// [E] `cl:bprime`: at (24, 5) and (28, 7) the F_ell-algebra generated by
// T_2, T_3 has dimension 2 < 3 = dim M_k -- the primes <= B(k) do not generate.
void ArmProperSubalgebra() {
    const uint32_t cells[2][2] = {{24u, 5u}, {28u, 7u}};
    for (const auto& cell : cells) {
        const uint32_t k = cell[0], ell = cell[1];
        const uint32_t d = SturmBound::dimension_M_k(k);
        Check(d == 3u, "dim M_" + Str(k) + " == 3");
        const std::vector<std::vector<uint32_t>> miller =
            hecke_ext::EchelonMillerBasis(k, ell, 3u * d + 2u);
        std::vector<MatrixT<uint32_t>> gens;
        gens.push_back(hecke_ext::HeckeOperatorOnMk(2u, k, ell, miller));
        gens.push_back(hecke_ext::HeckeOperatorOnMk(3u, k, ell, miller));
        uint32_t dim = 0;
        (void)hecke::algebra_closure(gens, d, ell, dim);
        Check(dim == 2u, "<T_2, T_3> has dimension 2 at (k,ell)=(" + Str(k) + "," + Str(ell) +
                             "), got " + Str(dim));
        Check(dim < d, "<T_2, T_3> is a PROPER subalgebra at (k,ell)=(" + Str(k) + "," + Str(ell) + ")");
    }
}

// The R3 arm: the forced cap, the resume, the refusal, the no-op and the
// budget history ([X] Sec. 1.3.1 R3.4, verbatim).
void ArmResumableState() {
    const uint32_t k = 24u, ell = 5u;
    const uint32_t full_cap = SturmBound::default_cap(k);  // 12 * B(24) = 36
    Check(full_cap == 36u, "default cap at k=24 is 36");

    // (1) `none` under a forced cap = 3 -- recorded, not thrown.
    SturmBound cut(k, ell, 3u);
    SturmOutcome o1 = SturmOutcome::InProgress;
    bool threw = false;
    try {
        o1 = cut.run();
    } catch (const std::exception&) {
        threw = true;
    }
    Check(!threw, "a cap hit is recorded, not thrown");
    Check(o1 == SturmOutcome::NoneAtCap, "outcome == none_at_cap under cap = 3 at (24, 5)");
    Check(cut.closure_dim() < cut.dim_M_k(), "the closure is still proper at cap = 3");
    CheckClosureFromScratch(cut, "(24, 5) cap = 3");
    Check(cut.state().primes_done == std::vector<uint32_t>({2u, 3u}),
          "the cap-3 state processed exactly the primes 2, 3");
    Check(cut.state().budget_history == std::vector<uint32_t>({3u}),
          "budget_history after run(cap=3)");
    {
        const SturmBoundCertificate c = cut.certificate();
        const uint32_t code = c.verify();
        Check(code == 0u, "the none_at_cap certificate verifies (code " + Str(code) + ")");
        Check(!c.has_B_prime, "the none_at_cap certificate carries no B'");
        SturmBoundCertificate bent = c;
        bent.closure_dim += 1u;
        Check(bent.verify() != 0u, "a bumped closure_dim fails verify() on the none certificate");
    }

    // (2) a resume BELOW the reached state is refused with the named error.
    {
        bool refused = false;
        std::string msg;
        try {
            cut.resume(2u);
        } catch (const std::logic_error& e) {
            refused = true;
            msg = e.what();
        }
        Check(refused, "resume(cap = 2) below B_reached = 3 is refused");
        Check(msg.find("resume: budget below reached state") != std::string::npos,
              "the refusal carries the named error, got: " + msg);
    }

    // (3) resume to the default cap reaches B' = 5.
    const SturmOutcome o2 = cut.resume(full_cap);
    Check(o2 == SturmOutcome::Found, "resume(cap = 36) finds B'");
    Check(cut.B_prime() == 5u, "resume reaches B' = 5 at (24, 5)");
    Check(cut.state().budget_history == std::vector<uint32_t>({3u, 36u}),
          "budget_history after run(3) + resume(36) is [3, 36]");
    Check(cut.cap() == full_cap, "the cap in force after the resume is the new one");
    CheckClosureFromScratch(cut, "(24, 5) resumed");

    // (4) RESUME INVARIANCE: identical to a fresh run at the same cap.
    SturmBound fresh(k, ell, full_cap);
    const SturmOutcome o3 = fresh.run();
    Check(o3 == o2, "resumed and fresh outcomes agree");
    CheckClosureFromScratch(fresh, "(24, 5) fresh");
    Check(fresh.B_prime() == cut.B_prime(), "resumed and fresh B' agree");
    Check(fresh.state().equals(cut.state(), /*ignore_budget_history=*/true),
          "the resumed SturmState equals the fresh one entrywise (budget history aside)");
    Check(!fresh.state().equals(cut.state(), /*ignore_budget_history=*/false),
          "the budget histories differ, as R3.1 item 4 requires ([36] vs [3, 36])");

    const SturmBoundCertificate c_resumed = cut.certificate();
    const SturmBoundCertificate c_fresh   = fresh.certificate();
    Check(c_resumed.k == c_fresh.k && c_resumed.ell == c_fresh.ell &&
              c_resumed.dim_M_k == c_fresh.dim_M_k && c_resumed.cap == c_fresh.cap &&
              c_resumed.has_B_prime == c_fresh.has_B_prime &&
              c_resumed.B_prime == c_fresh.B_prime &&
              c_resumed.closure_dim == c_fresh.closure_dim &&
              c_resumed.primes == c_fresh.primes,
          "the resumed and fresh certificate headers agree");
    Check(c_resumed.T_q.size() == c_fresh.T_q.size(), "the certificates carry the same T_q count");
    bool matrices_equal = c_resumed.T_q.size() == c_fresh.T_q.size();
    for (size_t i = 0; matrices_equal && i < c_fresh.T_q.size(); ++i) {
        if (c_resumed.T_q[i].data != c_fresh.T_q[i].data) matrices_equal = false;
    }
    Check(matrices_equal, "the resumed and fresh T_q matrices are entrywise equal");
    Check(c_resumed.verify() == 0u && c_fresh.verify() == 0u,
          "both certificates verify");

    // (5) a resume on a COMPLETE state is a strict no-op.
    const SturmState before = cut.state();
    const SturmOutcome o4 = cut.resume(full_cap + 24u);
    Check(o4 == SturmOutcome::Found, "resume on a complete state returns the same outcome");
    Check(cut.state().equals(before, /*ignore_budget_history=*/false),
          "resume on a complete state changes nothing -- not even the budget history");
    const SturmBoundCertificate c_after = cut.certificate();
    Check(c_after.cap == c_resumed.cap && c_after.B_prime == c_resumed.B_prime &&
              c_after.primes == c_resumed.primes,
          "resume on a complete state returns the same certificate");

    // (6) the JSON carries the R3.1 "resumable" object.
    const std::string j = c_resumed.to_json();
    Check(j.find("\"resumable\":") != std::string::npos, "to_json carries a resumable object");
    Check(j.find("\"kind\":\"sturm_bound\"") != std::string::npos, "the resumable object names its kind");
    Check(j.find("\"complete\":true") != std::string::npos, "the resumable object is complete");
    Check(j.find("\"budget_history\":[3,36]") != std::string::npos,
          "the resumable object records the budget history [3, 36]");
}

// The two `sturm.unavailable` modes ([X] Sec. 1.5 / Sec. 1.3.1 R3.2,
// confirmation 6).  WP-3 owns the pipeline-free half: the map
// (outcome, mode) -> (final, reason).  E1b owns the heuristic SWEEP.
void ArmUnavailableModes() {
    SturmBound found(24u, 5u);
    found.run();
    Check(found.outcome() == SturmOutcome::Found, "modes arm: the found case is found");
    for (SturmUnavailableMode mode : {SturmUnavailableMode::NonFinal, SturmUnavailableMode::Heuristic}) {
        const SturmFinality f = hecke_ext::sturm_finality(found, mode);
        Check(f.value == SturmFinal::True,
              std::string("a found B' does not obstruct finality in mode ") + hecke_ext::ToString(mode));
        Check(f.reason.empty(), "a found B' carries no reason");
    }

    SturmBound none(24u, 5u, 3u);
    none.run();
    Check(none.outcome() == SturmOutcome::NoneAtCap, "modes arm: the capped case is none_at_cap");
    {
        const SturmFinality f = hecke_ext::sturm_finality(none, SturmUnavailableMode::NonFinal);
        Check(f.value == SturmFinal::False, "default mode: none_at_cap => final = false");
        Check(f.reason == "sturm_unavailable", "default mode: reason == sturm_unavailable");
    }
    {
        const SturmFinality f = hecke_ext::sturm_finality(none, SturmUnavailableMode::Heuristic);
        Check(f.value == SturmFinal::Heuristic, "heuristic mode: none_at_cap => final = heuristic");
        Check(f.reason == "sturm_heuristic", "heuristic mode: reason == sturm_heuristic");
    }
    Check(std::string(hecke_ext::ToString(SturmUnavailableMode::NonFinal)) == "nonfinal",
          "the default mode is named nonfinal");
    Check(std::string(hecke_ext::ToString(SturmFinal::Heuristic)) == "heuristic",
          "the tri-state value heuristic is named");

    // The cap is a SETTING: an explicit cap overrides the default.
    SturmBound capped(60u, 5u, 4u);
    capped.run();
    Check(capped.cap() == 4u, "an explicit cap overrides 12*B(k)");
    Check(capped.outcome() == SturmOutcome::NoneAtCap,
          "B'(60, 5) = 5 is out of reach under a forced cap of 4");
    Check(capped.resume(SturmBound::default_cap(60u)) == SturmOutcome::Found,
          "resuming the forced-cap (60, 5) case to the default cap finds B'");
    Check(capped.B_prime() == 5u, "B'(60, 5) == 5 after the resume");
}

} // namespace

uint32_t HeckeSturmBoundValidation(void) {
    std::cout << "HeckeSturmBoundValidation: the computed Sturm bound B'(k, ell) "
                 "([E] def:bprime / cl:bprime), its certificate and its resumable state "
                 "([X] Sec. 1.3.1 R3, instance 1).\n";
    Check.reset();

    try {
        ArmQExpansions();
        ArmDimensions();
        ArmMillerEchelon();
        ArmT_q_on_M12_691();
        ArmPinnedBounds();
        ArmProperSubalgebra();
        ArmResumableState();
        ArmUnavailableModes();
    } catch (const std::exception& e) {
        std::cout << "  FAIL: unexpected exception: " << e.what() << "\n";
        ++Check.fails;
    }

    std::cout << "HeckeSturmBoundValidation: " << Check.checks << " checks, " << Check.fails
              << " failing.\n";
    if (Check.fails == 0u) {
        std::cout << "HeckeSturmBoundValidation: PASSED.\n";
        return 0u;
    }
    std::cout << "HeckeSturmBoundValidation: FAILED.\n";
    return Check.fails;
}
