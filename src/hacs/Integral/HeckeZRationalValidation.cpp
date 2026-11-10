// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZRationalValidation -- slot-25 HZ-RAT (see the header).

#include "HeckeZRationalValidation.h"

#include "HeckeZCharpoly.h"
#include "HeckeZFactorFinder.h"
#include "HeckeZRational.h"
#include "IntegralCharpolyCrt.h"
#include "Descent/DescentHecke.h"
#include "Descent/DescentModuleExport.h"
#include "../Hecke/HeckeLevelN.h"
#include "../Hecke/HeckeLevelNHeilbronn.h"
#include "../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <set>
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

const heckez::RationalResult Run(uint32_t N, uint32_t n, const std::vector<uint32_t>& P, uint32_t threads, double* secs) {
    heckez::RationalSettings s;
    s.threads = threads;
    const auto t0 = std::chrono::steady_clock::now();
    heckez::RationalResult R = heckez::RationalStep(N, n, P, s);
    if (secs) *secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

void Print(const heckez::RationalResult& R, double secs) {
    std::cout << "  CELL (" << R.N << "," << R.n << ") " << R.algebra << " b1=" << R.b1 << " d=" << R.d << " d_an=" << R.d_an
              << " C1=" << R.c1_status << " theta_index=" << R.theta_index << " C2=" << R.c2_status
              << " disc_gcd=" << heckez::FactoredText(R.disc_gcd) << " gP=" << heckez::FactoredText(R.gP) << " (" << R.gP_status
              << ") digest=0x" << std::hex << R.digest() << std::dec << " secs=" << secs << "\n";
    std::cout << "    S_cand=[";
    for (size_t i = 0; i < R.S_cand.size(); ++i)
        std::cout << (i ? "," : "") << R.S_cand[i].ell << ":" << R.S_cand[i].v_disc << ":" << R.S_cand[i].reason;
    std::cout << "] gP_running=[";
    for (size_t i = 0; i < R.gP_running.size(); ++i) std::cout << (i ? ", " : "") << "T_" << R.gP_ops[i] << ":" << R.gP_running[i];
    std::cout << "]\n";
    for (uint32_t ti : R.discriminants_used) {
        const auto& t = R.thetas[ti];
        std::cout << "    theta[" << ti << "] primes=" << t.primes_used << " disc=" << heckez::FactoredText(t.disc);
        if (R.eis_rational) std::cout << " c(lambda)=" << heckez::FactoredText(t.eis_res);
        std::cout << "\n";
    }
}

bool HasFactor(const heckez::Factored& f, uint64_t q, uint32_t e, const std::string& proof) {
    for (const auto& pp : f.factors)
        if (pp.q == q && pp.e == e && pp.proof == proof) return true;
    return false;
}

}  // namespace

uint32_t HeckeZRationalValidation() {
    std::cout << "============ HeckeZRationalValidation (slot-25 HZ-RAT) ============\n";
    Tally T;
    uint32_t cells = 0, tampered_rejected = 0;
    bool threads_equal = true;
    try {
        // ---- T1 (1, 10)
        uint64_t dig_t1 = 0;
        {
            double secs = 0;
            const auto R = Run(1, 10, {2, 3, 5, 7}, 4, &secs);
            Print(R, secs);
            ++cells;
            dig_t1 = R.digest();
            T.check(R.b1 == 3 && R.d == 2 && R.d_an == 2 && R.algebra == "full", "T1: b1 = 3, d = d^an = 2");
            T.check(R.c1_status == "certified" && R.theta_index == 0, "T1: (C1) with theta = T_2");
            T.check(R.thetas.size() == 4 && R.discriminants_used.size() == 4, "T1: every T_p primitive");
            const auto& t2 = R.thetas[0];
            T.check(t2.f == std::vector<std::string>{"-0xc018", "-0x7e9", "0x1"}, "T1: f_{T_2} = x^2 - 2025 x - 49176");
            T.check(heckez::FactoredText(t2.disc) == "3^2*691^2" && t2.disc.value_hex == "0x419271", "T1: disc f_{T_2} = 3^2 691^2");
            T.check(heckez::FactoredText(t2.eis_res) == "3*691" && t2.lambda_hex == "0x801", "T1: c_2(2049) = 2073 = 3 * 691");
            T.check(heckez::FactoredText(R.thetas[1].eis_res) == "2^8*691", "T1: c_3(lambda_3) = 176896 = 2^8 691");
            T.check(heckez::FactoredText(R.disc_gcd) == "691^2" && R.c2_status == "complete", "T1: gcd of the discriminants 691^2");
            T.check(R.S_cand.size() == 1 && R.S_cand[0].ell == 691 && R.S_cand[0].v_disc == 2 && R.S_cand[0].reason == "disc_square",
                    "T1: S_cand = {691} (disc_square)");
            T.check(R.gP_running.size() == 4 && R.gP_running[0] == "3*691" && R.gP_running[1] == "691" && R.gP_running[3] == "691",
                    "T1: running g_P 3*691 -> 691");
            T.check(heckez::FactoredText(R.gP) == "691" && R.gP_status == "upper_bound_certified", "T1: g_P = 691");
            T.check(R.cusp_part == std::vector<std::string>{"0x18", "0x1"}, "T1: the cusp part x + 24 (tau(2) = -24)");
        }
        // ---- T2 (1, 22)
        uint64_t dig_t2 = 0;
        {
            double secs = 0;
            const auto R = Run(1, 22, {2, 3, 5, 7}, 4, &secs);
            Print(R, secs);
            ++cells;
            dig_t2 = R.digest();
            T.check(R.b1 == 5 && R.d == 3, "T2: b1 = 5, d = 3");
            T.check(R.c1_status == "certified" && R.theta_index == 0, "T2: (C1) with theta = T_2");
            T.check(R.gP_running.size() == 4 && R.gP_running[0] == "3^5*5^2*7^2*103*2294797" &&
                        R.gP_running[1] == "5^2*103*2294797" && R.gP_running[2] == "103*2294797" && R.gP_running[3] == "103*2294797",
                    "T2: the running g_P drops to 103 * 2294797 exactly at T_5");
            T.check(HasFactor(R.gP, 2294797, 1, "pratt") && HasFactor(R.gP, 103, 1, "small"), "T2: 2294797 by a verified Pratt certificate");
            T.check(R.gP_status == "upper_bound_certified", "T2: g_P upper_bound_certified");
            T.check(R.cusp_part == std::vector<std::string>{"-0x1385400", "-0x438", "0x1"}, "T2: cusp part x^2 - 1080 x - 20468736");
            // audit F2 + F3: the pool to a fixed point removes 2294797 from theta[3]'s c(lambda), its 57-bit cofactor 79249298311832863
            // gets a Pratt certificate, and theta[3]'s disc (cofactor = that prime squared) is then complete by a Pratt-certified root
            const auto& t3 = R.thetas[3];
            T.check(t3.disc.complete() && HasFactor(t3.disc, 79249298311832863ull, 2, "pratt") &&
                        HasFactor(t3.eis_res, 79249298311832863ull, 1, "pratt"),
                    "T2: theta[3] disc complete, 79249298311832863^2 (Pratt), the pool at its fixed point");
        }
        // ---- T5 (8, 10) anemic
        uint64_t dig_t5 = 0;
        {
            double secs = 0;
            const auto R = Run(8, 10, {3, 5, 7, 11, 13}, 4, &secs);
            Print(R, secs);
            ++cells;
            dig_t5 = R.digest();
            T.check(R.algebra == "anemic" && R.b1 == 22 && R.d == 13 && R.d_an == 6 && R.d_target == 6, "T5: b1 = 22, d = 13, d^an = 6");
            T.check(R.c1_status == "certified" && R.theta_index == 0, "T5: anemic (C1) with theta = T_3");
            const std::set<uint64_t> allowed = {2, 3, 5, 7, 691};
            bool sub = true;
            for (const auto& e : R.S_cand)
                if (e.reason == "disc_square" && !allowed.count(e.ell)) sub = false;
            T.check(sub && R.c2_status == "complete", "T5: S_cand (square part) subset {2, 3, 5, 7, 691}, complete");
        }
        // ---- W44 (1, 42)
        {
            double secs = 0;
            const auto R = Run(1, 42, {2, 3, 5, 7}, 4, &secs);
            Print(R, secs);
            ++cells;
            T.check(R.d == 4 && R.b1 == 7 && R.c1_status == "certified", "W44: d = 4, (C1)");
            bool probable67 = false;
            for (const auto& pp : R.disc_gcd.factors)
                if (pp.proof == "probable" && pp.bits == 67) probable67 = true;
            T.check(probable67 && R.c2_status == "partial(unfactored)", "W44: the 67-bit prime is `probable`, C2 partial(unfactored)");
        }
        // ---- negative controls
        {
            heckez::RationalSettings s;
            const uint64_t q = 2294797, c = 103ull * q;
            heckez::PrattCert pc;
            const bool found = heckez::finder::FindPratt(q, s.pratt_witness_cap, s.pratt_nodes, &pc);
            T.check(found && heckez::VerifyPratt(pc, s.pratt_nodes), "control: the finder's Pratt certificate of 2294797 verifies");
            // the production verifier (CertifyWordClaim, behind SplitWordCofactor) on the true claim, then on forged ones
            auto claim = [&](const std::vector<std::pair<uint64_t, unsigned>>& cl, std::vector<heckez::PrimePower>* got, uint64_t* rest) {
                got->clear();
                heckez::CertifyWordClaim(c, cl, s, *got, rest);
            };
            std::vector<heckez::PrimePower> got;
            uint64_t rest = 0;
            claim({{103, 1}, {q, 1}}, &got, &rest);
            T.check(got.size() == 2 && rest == 1 && got[1].q == q && got[1].proof == "pratt",
                    "control: the true factorisation of 103 * 2294797 certifies through CertifyWordClaim");
            // (1a) a wrong exponent (103^2 claimed): the whole claim is discarded
            claim({{103, 2}, {q, 1}}, &got, &rest);
            if (got.empty() && rest == c) ++tampered_rejected;
            else std::cout << "  [MISMATCH] wrong-exponent claim accepted\n";
            // (1b) a tampered factor: 2294797 replaced by 2294799
            // (103 is a true factor and stays certified; the false prime is not certified and the true 2294797 is left as the rest)
            claim({{103, 1}, {2294799, 1}}, &got, &rest);
            if (got.size() == 1 && got[0].q == 103 && rest == q) ++tampered_rejected;
            else std::cout << "  [MISMATCH] tampered factor accepted\n";
            // (2) a tampered Pratt certificate: the witness replaced by a square (order divides (q - 1) / 2)
            heckez::PrattCert pc_w = pc;
            pc_w.witness = 4;
            if (!heckez::VerifyPratt(pc_w, s.pratt_nodes)) ++tampered_rejected;
            else std::cout << "  [MISMATCH] tampered Pratt certificate accepted\n";
            // (3) a composite with a well-formed forged certificate: 236364091 = 103 * 2294797 claimed prime, q - 1 = 2*3*5*17*463459
            // exactly (all small), the witness 4511 a Fermat liar (4511^{q-1} = 1 mod q): rejected only by the Lucas step
            heckez::PrattCert forged;
            forged.q = c;
            forged.witness = 4511;
            forged.factors = {{2, 1}, {3, 1}, {5, 1}, {17, 1}, {463459, 1}};
            T.check(heckez::PowModU64(4511, c - 1, c) == 1, "control: 4511 is a Fermat liar for 103 * 2294797 (the forgery is sharp)");
            if (!heckez::VerifyPratt(forged, s.pratt_nodes)) ++tampered_rejected;
            else std::cout << "  [MISMATCH] forged certificate of a composite accepted\n";
            // (4) the composite claimed prime through the production verifier
            {
                std::vector<heckez::PrimePower> g2;
                uint64_t r2 = 0;
                heckez::CertifyWordClaim(c, {{c, 1}}, s, g2, &r2);
                if (g2.empty() && r2 == c) ++tampered_rejected;
                else std::cout << "  [MISMATCH] composite claimed prime accepted\n";
            }
            std::cout << "  CONTROLS tampered_rejected=" << tampered_rejected << " (witness of 2294797: " << pc.witness << ")\n";
        }
        // ==== slot-25 HZ-RAT-PERF arms (value-neutral speed and memory) ====
        // (CP) the charpoly mod a word prime: the Hessenberg route (O(b^3), Shoup / Barrett) against the Berkowitz route of
        // hecke::level::HCharpoly (O(b^4), the route before HZ-RAT-PERF), coefficient for coefficient, on random dense, sparse,
        // nilpotent, permutation, rank-one and block matrices, mod several primes (small ones force zero pivots and swaps)
        {
            std::mt19937_64 rng(0x5eed2511ull);
            uint32_t cases = 0, agree = 0;
            const std::vector<uint64_t> ells = {5, 7, 11, 1009, 65521, 1073741789ull, 2147483647ull};
            for (uint64_t ell : ells)
                for (uint32_t b : {0u, 1u, 2u, 3u, 5u, 8u, 17u, 40u})
                    for (uint32_t kind = 0; kind < 6; ++kind) {
                        heckez::PrimeRun r;
                        r.ell = static_cast<uint32_t>(ell);
                        r.h1 = b;
                        std::vector<uint64_t> A(size_t(b) * b, 0);
                        for (uint32_t i = 0; i < b; ++i)
                            for (uint32_t j = 0; j < b; ++j) {
                                uint64_t v = rng() % ell;
                                switch (kind) {
                                    case 0: break;                                   // dense
                                    case 1: if (rng() % 10) v = 0; break;            // sparse
                                    case 2: if (j <= i) v = 0; break;                // nilpotent (strictly upper)
                                    case 3: v = (j == (i + 1) % b) ? 1 : 0; break;   // a cyclic permutation (zero subdiagonals)
                                    case 4: break;                                   // rank one, below
                                    case 5: if ((i < b / 2) != (j < b / 2)) v = 0; else if (i % 3 == 0) v = 0; break;   // blocks
                                }
                                A[size_t(i) * b + j] = v;
                            }
                        if (kind == 4 && b) {
                            std::vector<uint64_t> u(b), w(b);
                            for (auto& x : u) x = rng() % ell;
                            for (auto& x : w) x = rng() % ell;
                            for (uint32_t i = 0; i < b; ++i)
                                for (uint32_t j = 0; j < b; ++j)
                                    A[size_t(i) * b + j] = static_cast<uint64_t>(static_cast<unsigned __int128>(u[i]) * w[j] % ell);
                        }
                        r.mats = {A, A};
                        for (const std::vector<uint32_t>& c : {std::vector<uint32_t>{1, 0}, std::vector<uint32_t>{2, 3}}) {
                            if (ell == 2147483647ull && b > 0) {
                                // beyond the word primes (< 2^30) but inside the kernel's range (< 2^31): Hessenberg vs Berkowitz direct
                                ++cases;
                                if (heckez::CharpolyHessenberg(A, b, ell) == heckez::CharpolyOfCombinationBerkowitz(r, {1, 0})) ++agree;
                                break;
                            }
                            ++cases;
                            if (heckez::CharpolyOfCombination(r, c) == heckez::CharpolyOfCombinationBerkowitz(r, c)) ++agree;
                        }
                    }
            // the real Hecke matrices of the shipped cokernel route: (1, 22), (11, 10), (8, 10) at three word primes, units and a combination
            for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{{1, 22}, {11, 10}, {8, 10}}) {
                const std::vector<uint32_t> PP = cell.first == 1 ? std::vector<uint32_t>{2, 3, 5} : std::vector<uint32_t>{3, 5, 7};
                for (uint32_t ell : heckez::WordPrimes(3, cell.first)) {
                    const heckez::PrimeRun r = heckez::RunAtPrime(cell.first, cell.second, PP, ell);
                    for (const std::vector<uint32_t>& c : {std::vector<uint32_t>{1, 0, 0}, std::vector<uint32_t>{0, 0, 1}, std::vector<uint32_t>{1, 2, 3}}) {
                        ++cases;
                        if (heckez::CharpolyOfCombination(r, c) == heckez::CharpolyOfCombinationBerkowitz(r, c)) ++agree;
                    }
                }
            }
            // timing at b = 128, one word prime (the speed-up behind (b1); Berkowitz is O(b^4))
            double t_new = 0, t_old = 0;
            {
                heckez::PrimeRun r;
                r.ell = 1073741789u;
                r.h1 = 128;
                std::vector<uint64_t> A(128 * 128);
                for (auto& x : A) x = rng() % r.ell;
                r.mats = {A};
                auto t0 = std::chrono::steady_clock::now();
                const auto a = heckez::CharpolyOfCombination(r, {1});
                t_new = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                t0 = std::chrono::steady_clock::now();
                const auto b = heckez::CharpolyOfCombinationBerkowitz(r, {1});
                t_old = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                ++cases;
                if (a == b) ++agree;
            }
            std::cout << "  ARM CP charpoly Hessenberg vs Berkowitz: " << agree << "/" << cases << " agree; b=128: new " << t_new << " s, old "
                      << t_old << " s (x" << (t_new > 0 ? t_old / t_new : 0) << ")\n";
            T.check(agree == cases && cases >= 600, "CP: the Hessenberg charpoly equals Berkowitz's on every case");
        }
        // (GA) Garner's digits: the one-pass Barrett route (parallel over the later indices) against the descent's two-pass route
        {
            std::mt19937_64 rng(0x6a2e25ull);
            uint32_t cases = 0, agree = 0;
            for (size_t K : {size_t(1), size_t(2), size_t(7), size_t(513), size_t(1500), size_t(6000)}) {
                const std::vector<uint32_t> pr = heckez::WordPrimes(static_cast<uint32_t>(K), 26);
                std::vector<uint32_t> res(K);
                for (size_t i = 0; i < K; ++i) res[i] = static_cast<uint32_t>(rng() % pr[i]);
                const std::vector<uint32_t> ref = integral_descent::hecke3::GarnerDigits(res, pr, integral_descent::hecke3::GarnerInverses(pr));
                for (uint32_t th : {1u, 4u}) {
                    ++cases;
                    if (heckez::GarnerDigitsFast(pr, res, th) == ref) ++agree;
                }
            }
            std::cout << "  ARM GA Garner one-pass vs two-pass: " << agree << "/" << cases << " agree\n";
            T.check(agree == cases, "GA: the one-pass Garner digits equal the descent's at 1 and 4 threads");
        }
        // (EX) the export's Hecke application (HZ-RAT-PERF (c)): every column at once (ApplyHeilbronnColumns, 1 and 4 threads) against
        // HeilbronnOperator::apply column by column, and the partial operators over slices of Mer_p summed against the full operator;
        // at N = 26, n = 4, T_97 over Z/7^10 (a non-field ring, two column blocks)
        {
            const hecke::level::Gamma0LevelN lvl(26);
            const engine::ModCtx<uint32_t> ctx = engine::ModCtx<uint32_t>::prime_power(7, 10);
            const hecke::level::HeilbronnOperator H(lvl, 97, 4, ctx);
            const uint32_t D = H.D(), k = 70;
            std::mt19937_64 rng(0xe8ull);
            std::vector<uint32_t> X(size_t(D) * k);
            for (auto& x : X) x = static_cast<uint32_t>(rng() % ctx.p);
            std::vector<uint64_t> ref(size_t(D) * k);
            std::vector<uint32_t> col(D);
            for (uint32_t t = 0; t < k; ++t) {
                for (uint32_t r = 0; r < D; ++r) col[r] = X[size_t(r) * k + t];
                const std::vector<uint32_t> y = H.apply(col);
                for (uint32_t r = 0; r < D; ++r) ref[size_t(r) * k + t] = y[r];
            }
            bool ok = true;
            for (uint32_t th : {1u, 4u}) {
                std::vector<uint64_t> acc(size_t(D) * k, 0);
                integral_descent::modexport::ApplyHeilbronnColumns(H, X, k, acc, th);
                ok = ok && acc == ref;
            }
            const std::vector<hecke::Mat22> mer = hecke::level::MerelSet(97);
            std::vector<uint64_t> acc(size_t(D) * k, 0);
            for (size_t a = 0; a < mer.size(); a += 37) {
                const hecke::level::HeilbronnOperator Hc(lvl, 97, 4, ctx,
                                                         std::vector<hecke::Mat22>(mer.begin() + a, mer.begin() + std::min(mer.size(), a + 37)));
                integral_descent::modexport::ApplyHeilbronnColumns(Hc, X, k, acc, 4);
            }
            const bool sliced = acc == ref;
            // the dense route: the D x D matrix from slices (37 matrices, 1 and 4 workers) = HeilbronnOperator::dense(); its product = apply
            bool dense_ok = true;
            for (uint32_t th : {1u, 4u}) {
                const std::vector<uint32_t> Md = integral_descent::modexport::HeilbronnDense(lvl, 97, 4, ctx, mer, 37, th);
                dense_ok = dense_ok && Md == H.dense();
                std::vector<uint64_t> accd(size_t(D) * k, 0);
                integral_descent::modexport::DenseApplyColumns(Md, D, X, k, accd, ctx.p, th);
                dense_ok = dense_ok && accd == ref;
            }
            T.check(dense_ok, "EX: the dense matrix accumulated from slices equals dense(), its product equals apply (1 and 4 workers)");
            std::cout << "  ARM EX columns-at-once = column-wise: " << (ok ? "yes" : "no") << "; slices of Mer_97 (" << mer.size()
                      << " matrices) summed = full: " << (sliced ? "yes" : "no") << "; dense from slices = dense(): " << (dense_ok ? "yes" : "no") << "\n";
            T.check(ok, "EX: ApplyHeilbronnColumns equals HeilbronnOperator::apply (1 and 4 threads)");
            T.check(sliced, "EX: the partial operators over slices of Mer_p sum to the full operator");
        }
        // (WP) the word-prime list: grown in place, prefix-stable (= WordPrimes(4096, N)); the cap in the unhashed tail only
        {
            bool prefix_ok = true;
            for (uint32_t N : {1u, 26u, 107u, 2089u, 30030u}) {
                std::vector<uint32_t> v;
                for (size_t c : {size_t(1), size_t(100), size_t(1000), size_t(4096)}) heckez::ExtendWordPrimes(v, c, N);
                prefix_ok = prefix_ok && v == heckez::WordPrimes(4096, N);
            }
            T.check(prefix_ok, "WP: the grown word-prime list equals WordPrimes(4096, N) (N = 1, 26, 107, 2089, 30030)");
            const std::string ct = heckez::RationalSettings{}.canonical_text();
            const std::string tail = ";word_primes=CrtPrimes(<2^30,sized_from_bound,cap=2^21)";
            const size_t at = ct.find(";word_primes=");
            T.check(ct.size() >= tail.size() && ct.compare(ct.size() - tail.size(), tail.size(), tail) == 0 &&
                        ct.find(';', at + 1) == std::string::npos,
                    "WP: the cap is the last key of the canonical text (the record's caps text erases it)");
        }
        // (RF) beyond the cap: the named refusal refused(word_prime_cap), not an ENGINE_CHECK; at a cap that suffices the same values
        {
            auto strip = [](std::string t) { return t.substr(0, t.find("settings=")); };
            heckez::RationalSettings s8;
            s8.word_prime_cap = 8;
            s8.threads = 4;
            bool refused = false;
            try {
                (void)heckez::RationalStep(1, 22, {2, 3, 5, 7}, s8);
            } catch (const heckez::WordPrimeCapRefusal& e) {
                refused = std::string(e.what()).compare(0, 23, "refused(word_prime_cap)") == 0;
            }
            T.check(refused, "RF: (1, 22) at cap 8 ends in refused(word_prime_cap)");
            heckez::RationalSettings s64 = s8;
            s64.word_prime_cap = 200;
            const auto R64 = heckez::RationalStep(1, 22, {2, 3, 5, 7}, s64);
            const auto Rd = Run(1, 22, {2, 3, 5, 7}, 4, nullptr);
            T.check(strip(R64.text()) == strip(Rd.text()), "RF: at a sufficient cap (200) the values equal the default's");
            std::cout << "  ARM RF refused_at_cap8=" << (refused ? "yes" : "no") << " primes_used(theta0)="
                      << (Rd.theta_index >= 0 ? Rd.thetas[size_t(Rd.theta_index)].primes_used : 0) << "\n";
        }
        // (CHI) the descent's chi_Z(T_p) route for a unit theta (no per-prime matrices) against the matrix route: chi_Z is rebuilt here
        // independently, by CRT from the cokernel route's charpolys with the bound |c_i| <= binom(b1, i) (1 + p^{n+1})^i; same text.
        // A tampered coefficient must be caught by the probe comparison (a defect: std::logic_error).
        {
            uint32_t agree = 0, cells_chi = 0, caught = 0;
            for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{{1, 22}, {1, 34}, {11, 10}, {8, 10}}) {
                const uint32_t N = cell.first, n = cell.second;
                std::vector<uint32_t> PP;
                for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u})
                    if (N % p != 0 && PP.size() < 3) PP.push_back(p);
                const auto Rm = Run(N, n, PP, 4, nullptr);
                const uint32_t b1 = Rm.b1;
                heckez::RationalInputs in;
                for (size_t o = 0; o < PP.size(); ++o) {
                    const uint64_t bits = b1 + uint64_t(b1) * ((n + 1) * heckez::BitLen64(PP[o]) + 1) + 2;
                    std::vector<uint32_t> pr = heckez::WordPrimes(static_cast<uint32_t>(bits / 29 + 2), N);
                    std::vector<std::vector<uint32_t>> res(b1 + 1, std::vector<uint32_t>(pr.size()));
                    for (size_t i = 0; i < pr.size(); ++i) {
                        const heckez::PrimeRun r = heckez::RunAtPrime(N, n, PP, pr[i]);
                        std::vector<uint32_t> unit(PP.size(), 0);
                        unit[o] = 1;
                        const auto cp = heckez::CharpolyOfCombination(r, unit);
                        for (uint32_t j = 0; j <= b1; ++j) res[j][i] = static_cast<uint32_t>(cp[j]);
                    }
                    for (uint32_t j = 0; j <= b1; ++j) in.chi_z[PP[o]].push_back(heckez::ZFromResidues(pr, res[j], bits).hex());
                }
                heckez::RationalSettings sc;
                for (uint32_t th : {1u, 4u}) {
                    sc.threads = th;
                    const auto Rc = heckez::RationalStep(N, n, PP, sc, &in);
                    ++cells_chi;
                    if (Rc.text() == Rm.text()) ++agree;
                    else std::cout << "  [MISMATCH] CHI (" << N << "," << n << ") threads " << th << "\n";
                }
                heckez::RationalInputs bad = in;
                std::string& c0 = bad.chi_z[PP[0]][0];
                c0 = c0 == "0x1" ? std::string("0x2") : std::string("0x1");   // chi(0) changed: differs mod every word prime
                try {
                    (void)heckez::RationalStep(N, n, PP, sc, &bad);
                } catch (const std::logic_error&) {
                    ++caught;
                }
            }
            std::cout << "  ARM CHI chi_Z route = matrix route: " << agree << "/" << cells_chi << "; tampered chi_Z caught " << caught << "/4\n";
            T.check(agree == cells_chi && cells_chi == 8, "CHI: the chi_Z route gives the matrix route's text at 1 and 4 threads");
            T.check(caught == 4, "CHI: a tampered chi_Z is caught at the probe primes (a defect, never a value)");
        }
        // ---- thread independence
        // (the cells above ran at 4 threads; T1 and T2 again at 1 and 16; HZ-RAT-PERF: T5 (8, 10) anemic too -- the parallel good-prime
        // walk at level N)
        for (uint32_t th : {1u, 16u}) {
            const uint64_t c = Run(8, 10, {3, 5, 7, 11, 13}, th, nullptr).digest();
            std::cout << "  THREADS " << th << ": T5 0x" << std::hex << c << std::dec << "\n";
            if (c != dig_t5) threads_equal = false;
        }
        for (uint32_t th : {1u, 16u}) {
            const uint64_t a = Run(1, 10, {2, 3, 5, 7}, th, nullptr).digest();
            const uint64_t b = Run(1, 22, {2, 3, 5, 7}, th, nullptr).digest();
            std::cout << "  THREADS " << th << ": T1 0x" << std::hex << a << " T2 0x" << b << std::dec << "\n";
            if (a != dig_t1 || b != dig_t2) threads_equal = false;
        }
        T.check(threads_equal, "threads: digests equal at 1, 4, 16");
        T.check(tampered_rejected == 5, "controls: all five forged inputs rejected");
    } catch (const std::exception& e) {
        std::cout << "HeckeZRationalValidation: EXCEPTION: " << e.what() << std::endl;
        ++T.failing;
    }
    // ---- the read-only arm HECKEZ_RAT_CELLS="N:n,N:n,..." (reports only; checks nothing, changes no value)
    if (const char* env = std::getenv("HECKEZ_RAT_CELLS")) {
        uint32_t k = 0, refused = 0;
        std::stringstream ss(env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t c = item.find(':');
            if (c == std::string::npos) { ++refused; std::cout << "  ARM refused: '" << item << "' (N:n expected)\n"; continue; }
            try {
                const uint32_t N = static_cast<uint32_t>(std::stoul(item.substr(0, c)));
                const uint32_t n = static_cast<uint32_t>(std::stoul(item.substr(c + 1)));
                std::vector<uint32_t> P;
                for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u})
                    if (N % p != 0 && P.size() < 4) P.push_back(p);
                double secs = 0;
                const auto R = Run(N, n, P, 4, &secs);
                ++k;
                uint32_t probable = 0;
                for (const auto& pp : R.disc_gcd.factors) if (pp.proof == "probable") ++probable;
                std::cout << "  ARM (" << N << "," << n << ") d_target=" << R.d_target << " C1=" << R.c1_status << " C2=" << R.c2_status
                          << " disc_gcd=" << heckez::FactoredText(R.disc_gcd) << " cofactor_bits=" << R.disc_gcd.cofactor_bits
                          << " probable=" << probable << " gP=" << heckez::FactoredText(R.gP) << " gP_status=" << R.gP_status
                          << " secs=" << secs << "\n";
            } catch (const std::exception& e) {
                ++refused;
                std::cout << "  ARM refused (" << item << "): " << e.what() << "\n";
            }
        }
        std::cout << "EFFECTIVE HeckeZRationalValidation arm=cells cells=" << k << " refused=" << refused << "\n";
    }
    const bool ok = T.failing == 0;
    std::cout << "HECKEZ_RAT cells=" << cells << " mismatched=" << T.failing << " tampered_rejected=" << tampered_rejected
              << " threads_equal=" << (threads_equal ? "yes" : "no") << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    std::cout << "HeckeZRationalValidation: " << T.checks << " checks, " << T.failing << " failing\n";
    return T.failing;
}
