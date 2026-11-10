// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZUTagValidation -- slot-25 HZ-UTAG (see the header).

#include "HeckeZUTagValidation.h"

#include "HeckeZUTag.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Tally {
    uint32_t checks = 0, failing = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [MISMATCH] " << what << "\n";
        }
    }
};

// the expected polynomial prod (x - r)^m (low to high), as canonical signed hex, from int64 products (the toy sizes fit)
std::vector<std::string> ExpectedHex(const std::vector<std::vector<int64_t>>& factors) {
    std::vector<int64_t> p{1};
    for (const auto& f : factors) {
        std::vector<int64_t> q(p.size() + f.size() - 1, 0);
        for (size_t i = 0; i < p.size(); ++i)
            for (size_t j = 0; j < f.size(); ++j) q[i + j] += p[i] * f[j];
        p = q;
    }
    std::vector<std::string> out;
    for (int64_t c : p) {
        std::ostringstream o;
        if (c < 0) o << "-";
        o << "0x" << std::hex << static_cast<uint64_t>(c < 0 ? -c : c);
        out.push_back(o.str());
    }
    return out;
}

void Print(const heckez::UTagResult& R, double secs) {
    std::cout << "  CELL (" << R.N << "," << R.n << ") U_" << R.p << " " << R.text() << "    primes=" << R.primes_used
              << " secs=" << secs << "\n";
}

bool HasBound(const heckez::UTagResult& R, const std::string& a, const std::string& b, uint64_t bound) {
    for (const auto& x : R.bounds)
        if (x.a == a && x.b == b && x.bound == bound) return true;
    return false;
}

}  // namespace

uint32_t HeckeZUTagValidation() {
    std::cout << "============ HeckeZUTagValidation (slot-25 HZ-UTAG) ============\n";
    Tally T;
    uint32_t cells = 0;
    try {
        // ---- T4 (2, 6), U_2
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto R = heckez::UTagStep(2, 6, 2, {3, 5, 7, 11});
            Print(R, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            ++cells;
            T.check(R.b1 == 4 && R.d == 3 && R.d_an == 2 && R.e_k == 2, "T4: b1 = 4, d = 3, d^an = 2, e_k = 2");
            T.check(R.chi == ExpectedHex({{-1, 1}, {-128, 1}, {8, 1}, {8, 1}}), "T4: chi_Q(U_2) = (x - 1)(x - 128)(x + 8)^2");
            T.check(R.m0 == 0 && R.split_degree == 4 && R.distinct == 3, "T4: split over Z, 3 distinct roots = d ((C1) for U_2)");
            T.check(R.upper_bound_certified && !R.non_reduced_certified, "T4: tags = upper_bound_certified");
            T.check(R.gP_an == "2^7*17", "T4: g_P^an = 2^7 * 17");
            T.check(HasBound(R, "eis(1)", "cusp(-8)", 1), "T4: C(f, E^(1)) | 1 (they do not meet)");
            T.check(HasBound(R, "eis(128)", "cusp(-8)", 136), "T4: C(f, E^(2^7)) | 2^3 * 17");
            T.check(HasBound(R, "eis(1)", "eis(128)", 127), "T4: C(E^(1), E^(2^7)) | 127");
        }
        // ---- T5 (8, 10), U_2
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto R = heckez::UTagStep(8, 10, 2, {});
            Print(R, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            ++cells;
            std::vector<std::vector<int64_t>> fs(16, std::vector<int64_t>{0, 1});
            fs.push_back({-1, 1});
            fs.push_back({-2048, 1});
            fs.push_back({2048, 24, 1});
            fs.push_back({2048, 24, 1});
            T.check(R.b1 == 22 && R.d == 13 && R.d_an == 6, "T5: b1 = 22, d = 13, d^an = 6");
            T.check(R.chi == ExpectedHex(fs), "T5: chi_Q(U_2) = x^16 (x - 1)(x - 2048)(x^2 + 24 x + 2048)^2");
            T.check(R.m0 == 16, "T5: the root 0 with multiplicity 16");
            T.check(R.non_reduced_certified && !R.upper_bound_certified, "T5: tags = non_reduced_certified (22 - 16 < 13 - 6)");
        }
        // ---- the scope guard of non_reduced_certified (audit F1): N = 12 is not a power of 2, so the tag is never set
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto R = heckez::UTagStep(12, 2, 2, {});
            Print(R, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            ++cells;
            T.check(!R.n_is_p_power && !R.non_reduced_certified && R.text().find("nr_scope=N_not_p_power") != std::string::npos,
                    "F1: (12, 2) U_2: N not a power of 2, non_reduced_certified not set, the text says why");
            T.check(R.m0 == 6 && !R.non_reduced_criterion, "F1: (12, 2): b1 - m0 = 6 = d - d^an (margin 0, the audit's count)");
        }
        // (24, 2): here the raw inequality HOLDS (24 - 18 = 6 < 16 - 5 = 11) and the guard alone withholds the tag
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto R = heckez::UTagStep(24, 2, 2, {});
            Print(R, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            ++cells;
            T.check(R.b1 == 24 && R.m0 == 18 && R.d == 16 && R.d_an == 5 && R.non_reduced_criterion && !R.n_is_p_power &&
                        !R.non_reduced_certified && R.tags().empty(),
                    "F1: (24, 2) U_2: criterion 6 < 11 holds, N = 24 not a power of 2: the guard withholds non_reduced_certified");
        }
        // ---- (HZ-RAT-PERF fix round 1, audit F1) arm RT: the U_p routes against the shipped term route (RunAtPrime), entrywise, at
        // three word primes; the new routes on 4 workers, the term route on 1
        {
            struct RtCell { uint32_t N, n, p; };
            const std::vector<RtCell> rt = {{11, 0, 11}, {37, 0, 37}, {26, 0, 13}, {26, 0, 2}, {2, 6, 2}, {3, 4, 3}, {26, 12, 13}, {8, 10, 2}};
            bool saw_count = false, saw_dense = false;
            for (const auto& c : rt) {
                const std::vector<uint32_t> ells = heckez::WordPrimes(3, c.N);
                heckez::UTagRoute used = heckez::UTagRoute::kAuto;
                const auto t0 = std::chrono::steady_clock::now();
                const auto A = heckez::UTagRuns(c.N, c.n, c.p, ells, 4, heckez::UTagRoute::kAuto, &used);
                const double ta = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const auto B = heckez::UTagRuns(c.N, c.n, c.p, ells, 1, heckez::UTagRoute::kTermTable);
                bool eq = A.size() == B.size();
                for (size_t i = 0; eq && i < A.size(); ++i) eq = A[i].ell == B[i].ell && A[i].h1 == B[i].h1 && A[i].mats == B[i].mats;
                saw_count = saw_count || used == heckez::UTagRoute::kCountOnce;
                saw_dense = saw_dense || used == heckez::UTagRoute::kDensePerPrime;
                std::cout << "  ARM RT (" << c.N << "," << c.n << ") U_" << c.p << " route=" << heckez::UTagRouteName(used)
                          << " b1=" << (A.empty() ? 0u : A[0].h1) << " equal=" << (eq ? "yes" : "no") << " secs=" << ta << "\n";
                T.check(eq, "RT: (" + std::to_string(c.N) + "," + std::to_string(c.n) + ") U_" + std::to_string(c.p) +
                                " route " + heckez::UTagRouteName(used) + " = the term route at 3 word primes");
            }
            T.check(saw_count && saw_dense, "RT: both new routes (count_once, dense_per_prime) exercised");
            // (re-audit N1) every cell above has |Mer_p| m_N < 2^20 terms: BuildCounts runs ONE slice on ONE worker.  The multi-slice,
            // multi-worker partial-count sum (the path of (431, 0) .. (2089, 0) in production) on (37, 0) U_37 with the validation-only
            // slice size of 64 terms (one Merel matrix per slice; the 2^20 default would need (431, 0), ~17 s): = the term route
            {
                const std::vector<uint32_t> ells = heckez::WordPrimes(3, 37);
                heckez::UTagRoute used = heckez::UTagRoute::kAuto;
                heckez::UTagCountInfo info;
                const auto A = heckez::UTagRuns(37, 0, 37, ells, 4, heckez::UTagRoute::kAuto, &used, 64, &info);
                const auto B = heckez::UTagRuns(37, 0, 37, ells, 1, heckez::UTagRoute::kTermTable);
                heckez::UTagCountInfo info1;
                const auto A1 = heckez::UTagRuns(37, 0, 37, ells, 1, heckez::UTagRoute::kAuto, nullptr, 0, &info1);
                bool eq = A.size() == B.size() && A1.size() == B.size();
                for (size_t i = 0; eq && i < A.size(); ++i) eq = A[i].ell == B[i].ell && A[i].h1 == B[i].h1 && A[i].mats == B[i].mats && A1[i].mats == B[i].mats;
                std::cout << "  ARM RT-MULTI (37,0) U_37 route=" << heckez::UTagRouteName(used) << " slices=" << info.slices << " workers=" << info.workers
                          << " (default: slices=" << info1.slices << " workers=" << info1.workers << ") equal=" << (eq ? "yes" : "no") << "\n";
                T.check(used == heckez::UTagRoute::kCountOnce && info.slices >= 3 && info.workers >= 2 && info1.slices == 1 && eq,
                        "RT (re-audit N1): (37, 0) U_37 count_once summed over >= 3 slices on >= 2 workers = the term route at 3 word primes");
            }
        }
        // ---- (fix round 1, audit F7) the cap is honoured below 2048: (2, 6) U_2 needs 3 word primes (66 bits)
        {
            const auto def = heckez::UTagStep(2, 6, 2, {});
            heckez::RationalSettings s2, s3;
            s2.word_prime_cap = 2;
            s3.word_prime_cap = 3;
            bool refused = false;
            try {
                (void)heckez::UTagStep(2, 6, 2, {}, s2);
            } catch (const heckez::WordPrimeCapRefusal& e) {
                refused = std::string(e.what()).compare(0, 23, "refused(word_prime_cap)") == 0;
            }
            const auto at3 = heckez::UTagStep(2, 6, 2, {}, s3);
            std::cout << "  ARM CAP (2,6) U_2 primes=" << def.primes_used << " cap2=" << (refused ? "refused" : "not_refused")
                      << " cap3_equal=" << (at3.digest() == def.digest() ? "yes" : "no") << "\n";
            T.check(def.primes_used == 3 && refused && at3.digest() == def.digest(),
                    "CAP: (2, 6) U_2 at cap 2 is refused(word_prime_cap), at cap 3 (= the need) the default's text");
        }
        // ---- (fix round 1) threads: (37, 0) U_37 (count_once) and T4 (dense_per_prime) at 1 and 4 threads, the same text
        {
            heckez::RationalSettings s1, s4;
            s1.threads = 1;
            s4.threads = 4;
            const auto a1 = heckez::UTagStep(37, 0, 37, {}, s1), a4 = heckez::UTagStep(37, 0, 37, {}, s4);
            const auto b1 = heckez::UTagStep(2, 6, 2, {3, 5, 7, 11}, s1), b4 = heckez::UTagStep(2, 6, 2, {3, 5, 7, 11}, s4);
            ++cells;
            Print(a4, 0);
            T.check(a1.digest() == a4.digest() && b1.digest() == b4.digest(), "THREADS: (37, 0) U_37 and (2, 6) U_2 equal at 1 and 4 threads");
        }
    } catch (const std::exception& e) {
        std::cout << "HeckeZUTagValidation: EXCEPTION: " << e.what() << std::endl;
        ++T.failing;
    }
    std::cout << "HECKEZ_UTAG cells=" << cells << " mismatched=" << T.failing << " verdict=" << (T.failing == 0 ? "OK" : "FAIL") << "\n";
    std::cout << "HeckeZUTagValidation: " << T.checks << " checks, " << T.failing << " failing\n";
    return T.failing;
}
