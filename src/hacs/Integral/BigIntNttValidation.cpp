// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// BigIntNttValidation -- see the header (slot-27 BIGNTT M1).
#include "BigIntNttValidation.h"

#include "HeckeZFactorFinder.h"

#include "../NumberField/BigInt.h"
#include "../NumberField/BigNtt.h"
#include "../NumberField/Ntt64.h"
#include "../NumberField/Parallel.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using nf::BigInt;
using Mag = BigInt::Mag;
namespace bn = nf::bigntt;

struct Tally {
    uint64_t checks = 0, mismatched = 0;
    void eq(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++mismatched;
            std::cout << "  MISMATCH " << what << "\n";
        }
    }
};

Mag RandomMag(std::mt19937_64& rng, std::size_t limbs) {
    Mag m(limbs);
    for (auto& x : m) x = static_cast<uint32_t>(rng());
    if (limbs) m.back() |= 1u;   // exactly `limbs` limbs
    return m;
}

BigInt FromMag(const Mag& m) { return BigInt::from_magnitude(m, false); }

// The finder's PowModBig as it was before slot 27 (the reference of arm B): (r b) % m, (b b) % m, e halved by divmod.
BigInt PowModBigOld(BigInt b, BigInt e, const BigInt& m) {
    BigInt r(1);
    b = b % m;
    const BigInt two(2);
    while (!e.is_zero()) {
        BigInt q, rem;
        BigInt::divmod(e, two, q, rem);
        if (!rem.is_zero()) r = (r * b) % m;
        b = (b * b) % m;
        e = q;
    }
    return r;
}

BigInt Mersenne(unsigned p) { return BigInt::pow(BigInt(2), p) - BigInt(1); }

// ---- arm P: products ------------------------------------------------------------------------------------
void ArmP(Tally& T, std::mt19937_64& rng) {
    // random balanced and unbalanced shapes, 2^8 .. 2^20 bits
    const std::size_t sizes[] = {8, 9, 31, 32, 33, 64, 100, 255, 256, 257, 512, 1000, 2048, 4096, 8191, 16384, 32768};
    for (std::size_t na : sizes) {
        for (std::size_t nb : {na, na / 3 + 1, std::size_t(1), std::size_t(17)}) {
            const Mag a = RandomMag(rng, na), b = RandomMag(rng, nb);
            const Mag p = bn::mul_ntt(a, b, 1);
            const Mag ref = (na <= 2048 && nb <= 2048) ? BigInt::mul_schoolbook(a, b) : BigInt::mul_mag(a, b);
            T.eq(p == ref, "P random " + std::to_string(na) + "x" + std::to_string(nb));
            if (na <= 512) T.eq(BigInt::mul_mag(a, b) == ref, "P karatsuba " + std::to_string(na));
        }
    }
    // every length class f 2^a at and around its boundary: nc = 2 (na + nb) - 1 chunks = L - 1, L, L + 1
    for (std::size_t L : {std::size_t(64), std::size_t(96), std::size_t(160), std::size_t(1024), std::size_t(1536),
                          std::size_t(2560), std::size_t(32768), std::size_t(49152), std::size_t(81920)}) {
        for (long d : {-3L, -1L, 1L}) {
            const long tot = (static_cast<long>(L) + d + 1) / 2;   // na + nb
            if (tot < 2) continue;
            const std::size_t na = static_cast<std::size_t>(tot) / 2 + 1, nb = static_cast<std::size_t>(tot) - na;
            if (nb == 0) continue;
            const Mag a = RandomMag(rng, na), b = RandomMag(rng, nb);
            const Mag ref = (na <= 2048) ? BigInt::mul_schoolbook(a, b) : BigInt::mul_mag(a, b);
            T.eq(bn::mul_ntt(a, b, 1) == ref, "P class L=" + std::to_string(L) + " d=" + std::to_string(d));
        }
    }
    // adversarial: all limbs 0xffffffff (the coefficient bound attained), squares, 2^k +- 1, a single non-zero chunk
    for (std::size_t n : {std::size_t(1), std::size_t(2), std::size_t(1023), std::size_t(1024), std::size_t(16384)}) {
        const Mag f(n, 0xffffffffu);
        const Mag ref = n <= 2048 ? BigInt::mul_schoolbook(f, f) : BigInt::mul_mag(f, f);
        T.eq(bn::mul_ntt(f, f, 1) == ref, "P all-ones mul " + std::to_string(n));
        T.eq(bn::sqr_ntt(f, 1) == ref, "P all-ones sqr " + std::to_string(n));
        const Mag g(n + 5, 0xffffffffu);
        T.eq(bn::mul_ntt(f, g, 1) == (n <= 2048 ? BigInt::mul_schoolbook(f, g) : BigInt::mul_mag(f, g)),
             "P all-ones unbalanced " + std::to_string(n));
    }
    for (unsigned k : {31u, 32u, 33u, 4096u, 65537u}) {
        const BigInt pk = BigInt::pow(BigInt(2), k);
        for (int s : {-1, 1}) {
            const BigInt x = pk + BigInt(s);
            const BigInt y = pk - BigInt(s);
            T.eq(FromMag(bn::mul_ntt(x.magnitude(), y.magnitude(), 1)) == x * y, "P 2^k+-1 k=" + std::to_string(k));
        }
    }
    {
        Mag a(3000, 0), b(2000, 0);
        a[1499] = 0x00010000u;   // one non-zero chunk
        b[0] = 0xffff0000u;
        T.eq(bn::mul_ntt(a, b, 1) == BigInt::mul_schoolbook(a, b), "P single chunk");
        T.eq(bn::mul_ntt(a, Mag(), 1).empty(), "P zero operand");
    }
}

// ---- arm T: threads, the pool, the nested fallback, exceptions -------------------------------------------
// M1b (code audit m4): every threaded check runs at a length where effective_threads really grants more than one
// thread (checked, and printed), radix-5 and radix-3 lengths among them; a barrier pass proves the pool runs the
// ranges concurrently.
void ArmT(Tally& T, std::mt19937_64& rng) {
    const std::uint64_t w0 = nf::parallel_pool_worker_ranges();
    auto eff = [&](std::size_t L, unsigned th, const std::string& what) {
        const unsigned e = bn::effective_threads(L, th);
        std::cout << "  arm T  " << what << " L=" << L << " threads=" << th << " effective=" << e << "\n";
        T.eq(e > 1, "T effective threads > 1: " + what + " L=" + std::to_string(L) + " threads=" + std::to_string(th));
    };
    // n -> L = ntt64_length(4 n + 25): 5000 -> 5 * 2^12, 22000 -> 3 * 2^15, 40000 -> 5 * 2^15
    for (std::size_t n : {std::size_t(5000), std::size_t(22000), std::size_t(40000)}) {
        const Mag a = RandomMag(rng, n), b = RandomMag(rng, n + 13);
        const Mag p1 = bn::mul_ntt(a, b, 1), s1 = bn::sqr_ntt(a, 1);
        for (unsigned th : {4u, 16u}) {
            eff(nf::ntt64_length(2 * a.size() + 2 * b.size() - 1), th, "product n=" + std::to_string(n));
            T.eq(bn::mul_ntt(a, b, th) == p1, "T product n=" + std::to_string(n) + " threads=" + std::to_string(th));
            T.eq(bn::sqr_ntt(a, th) == s1, "T square n=" + std::to_string(n) + " threads=" + std::to_string(th));
        }
    }
    {   // Barrett step and a powmod at threads 1 / 4 / 16: k -> L_ = ntt64_length(4 k + 3): 5000 -> 5 * 2^12, 6000 -> 3 * 2^13
        for (std::size_t k : {std::size_t(5000), std::size_t(6000)}) {
            const BigInt m = FromMag(RandomMag(rng, k));
            const BigInt x = FromMag(RandomMag(rng, k - 1)), y = FromMag(RandomMag(rng, k - 2));
            const BigInt e = BigInt::from_u64(rng());
            const bn::BarrettModulus M1(m, 2, 1);
            const BigInt r1 = M1.mulmod(x, y), s1 = bn::powmod(x, e, M1);
            for (unsigned th : {4u, 16u}) {
                const bn::BarrettModulus Mt(m, 2, th);
                eff(nf::ntt64_length(4 * k + 3), th, "barrett k=" + std::to_string(k));
                T.eq(Mt.threads() > 1, "T barrett object threads > 1 k=" + std::to_string(k));
                T.eq(Mt.mulmod(x, y) == r1, "T barrett k=" + std::to_string(k) + " threads=" + std::to_string(th));
                T.eq(bn::powmod(x, e, Mt) == s1, "T powmod k=" + std::to_string(k) + " threads=" + std::to_string(th));
            }
        }
    }
    {   // concurrency: four ranges that each wait until all four have started complete only if they run at once
        std::atomic<unsigned> started{0};
        std::atomic<bool> timed_out{false};
        const std::uint64_t wb = nf::parallel_pool_worker_ranges();
        nf::parallel_ranges(4, 4, [&](std::size_t, std::size_t, unsigned) {
            started.fetch_add(1);
            const auto t0 = std::chrono::steady_clock::now();
            while (started.load() < 4) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) {
                    timed_out = true;
                    break;
                }
                std::this_thread::yield();
            }
        });
        const std::uint64_t by_workers = nf::parallel_pool_worker_ranges() - wb;
        T.eq(!timed_out.load() && started.load() == 4, "T barrier: 4 ranges concurrently");
        T.eq(by_workers >= 3, "T barrier: >= 3 ranges on pool workers (got " + std::to_string(by_workers) + ")");
    }
    {   // nested: products inside pool tasks fall back to per-call threads; same bytes
        const Mag a = RandomMag(rng, 9000), b = RandomMag(rng, 7000);
        const Mag ref = bn::mul_ntt(a, b, 1);
        std::vector<Mag> got(4);
        nf::parallel_ranges(4, 4, [&](std::size_t lo, std::size_t hi, unsigned) {
            for (std::size_t i = lo; i < hi; ++i) got[i] = bn::mul_ntt(a, b, 4);
        });
        bool ok = true;
        for (const auto& g : got) ok = ok && g == ref;
        T.eq(ok, "T nested fallback");
    }
    {   // the pool's exception path: the first exception reaches the caller, the pool stays usable
        bool caught = false;
        try {
            nf::parallel_ranges(64, 8, [&](std::size_t lo, std::size_t, unsigned) {
                if (lo >= 32) throw std::logic_error("BigIntNttValidation: deliberate");
            });
        } catch (const std::logic_error&) {
            caught = true;
        }
        T.eq(caught, "T exception through the pool");
        std::vector<uint64_t> v(1000, 0);
        nf::parallel_ranges(v.size(), 8, [&](std::size_t lo, std::size_t hi, unsigned) {
            for (std::size_t i = lo; i < hi; ++i) v[i] = i * i;
        });
        bool ok = true;
        for (std::size_t i = 0; i < v.size(); ++i) ok = ok && v[i] == i * i;
        T.eq(ok, "T pool usable after an exception");
    }
    std::cout << "  arm T  pool_worker_ranges=" << nf::parallel_pool_worker_ranges() - w0
              << " (ranges run by pool workers, not the caller)\n";
}

// ---- arm B: Barrett primitive and short-exponent powmods ---------------------------------------------------
void ArmB(Tally& T, std::mt19937_64& rng) {
    std::vector<BigInt> mods;
    for (std::size_t k : {std::size_t(2), std::size_t(3), std::size_t(8), std::size_t(33), std::size_t(64),
                          std::size_t(257), std::size_t(1024), std::size_t(2048)})
        mods.push_back(FromMag(RandomMag(rng, k)));
    for (unsigned k : {64u, 255u, 256u, 4096u, 65536u}) {
        const BigInt pk = BigInt::pow(BigInt(2), k);
        mods.push_back(pk + BigInt(1));
        mods.push_back(pk - BigInt(1));
        mods.push_back(pk + BigInt(12345));
        mods.push_back(pk - BigInt(999));
    }
    for (unsigned k : {2u, 3u, 1100u}) mods.push_back(BigInt::pow(BigInt(2), 32 * (k - 1)));   // b^{k-1} (code audit m1)
    for (const BigInt& m : mods) {
        const std::string tag = " bits=" + std::to_string(m.bit_length());
        const bn::BarrettModulus M(m, 2, 1);
        T.eq(M.ntt_route(), "B forced route" + tag);
        const bn::BarrettModulus Old(m, std::numeric_limits<std::size_t>::max(), 1);
        T.eq(!Old.ntt_route(), "B old route" + tag);
        const BigInt mm1 = m - BigInt(1);
        std::vector<BigInt> xs = {BigInt(0), BigInt(1), mm1, m - BigInt(2)};
        for (int i = 0; i < 4; ++i) xs.push_back(FromMag(RandomMag(rng, m.limbs())).floor_mod(m));
        for (const BigInt& x : xs) {
            T.eq(M.sqrmod(x) == (x * x) % m, "B sqrmod" + tag);
            for (const BigInt& y : {mm1, xs.back()})
                T.eq(M.mulmod(x, y) == (x * y) % m, "B mulmod" + tag);
        }
        {   // negative operands (code audit m2): powmod reduces by floor_mod on both routes; mulmod / sqrmod refuse
            const BigInt ref = BigInt(-243).floor_mod(m);
            T.eq(bn::powmod(BigInt(-3), BigInt(5), M) == ref, "B powmod base -3" + tag);
            T.eq(bn::powmod(BigInt(-3), BigInt(5), Old) == ref, "B powmod base -3 old route" + tag);
            for (const bn::BarrettModulus* P : {&M, &Old}) {
                bool threw = false;
                try {
                    (void)P->mulmod(BigInt(-1), BigInt(1));
                } catch (const std::logic_error&) {
                    threw = true;
                }
                T.eq(threw, std::string("B mulmod refuses a negative operand") + (P == &M ? "" : " old route") + tag);
            }
        }
        if (m.limbs() <= 2048) {
            const BigInt base = xs.back();
            for (uint64_t e : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(65537), rng(), ~uint64_t(0)}) {
                const BigInt E = BigInt::from_u64(e);
                T.eq(bn::powmod(base, E, M) == PowModBigOld(base, E, m), "B powmod" + tag);
                T.eq(bn::powmod(BigInt(2), E, M) == bn::powmod(BigInt(2), E, Old), "B powmod base 2" + tag);
            }
        }
    }
}

// ---- arm MR: the finder's Miller--Rabin on known outcomes ----------------------------------------------------
void ArmMR(Tally& T) {
    using heckez::finder::MillerRabinBigForTest;
    for (unsigned p : {521u, 607u, 1279u, 2203u, 2281u}) {
        const BigInt M = Mersenne(p);
        T.eq(MillerRabinBigForTest(M, 12, 2, 1), "MR Mersenne prime p=" + std::to_string(p));
        if (p <= 1279) T.eq(MillerRabinBigForTest(M, 12, std::numeric_limits<std::size_t>::max(), 1),
                            "MR Mersenne prime old route p=" + std::to_string(p));
    }
    T.eq(MillerRabinBigForTest(Mersenne(4423), 2, 2, 4), "MR Mersenne prime p=4423 (2 bases, 4 threads)");
    {   // composites: a product of two primes, a square (the isqrt branch of BigProbablePrime)
        const BigInt c = Mersenne(521) * Mersenne(607);
        T.eq(!MillerRabinBigForTest(c, 12, 2, 1), "MR composite M521 M607");
        T.eq(!MillerRabinBigForTest(Mersenne(523), 12, 2, 1), "MR composite 2^523 - 1");
        const BigInt sq = Mersenne(1279) * Mersenne(1279);
        const heckez::finder::BigProbable bp = heckez::finder::BigProbablePrime(sq.to_string(), 12, 2, 2);
        T.eq(bp.found && bp.e == 2 && bp.root_decimal == Mersenne(1279).to_string(), "MR isqrt branch M1279^2");
        const heckez::finder::BigProbable bq = heckez::finder::BigProbablePrime(c.to_string(), 12, 1, 2);
        T.eq(!bq.found, "MR finder composite M521 M607");
    }
    {   // psi_12: the least strong pseudoprime to the first 12 prime bases (2 .. 37); composite, `probable` on both routes
        const BigInt psi = BigInt::from_string("318665857834031151167461");
        T.eq(psi == BigInt::from_string("399165290221") * BigInt::from_string("798330580441"), "MR psi12 factorisation");
        T.eq(MillerRabinBigForTest(psi, 12, 2, 1), "MR psi12 probable (NTT route)");
        T.eq(MillerRabinBigForTest(psi, 12, std::numeric_limits<std::size_t>::max(), 1), "MR psi12 probable (old route)");
        T.eq(!MillerRabinBigForTest(psi, 13, 2, 1), "MR psi12 caught by base 41");
    }
    {   // one large composite (M4423 M2281, 6.7 kbit), one base, forced route, 4 threads
        const BigInt c = Mersenne(4423) * Mersenne(2281);
        T.eq(!MillerRabinBigForTest(c, 1, 2, 4), "MR 6.7-kbit composite, one base (forced)");
    }
}

}  // namespace

uint32_t BigIntNttValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t p0 = bn::ntt_products(), b0 = bn::barrett_steps();
    std::mt19937_64 rng(0x5a27b16e7701ULL);
    Tally T;
    unsigned arms = 0;
    {
        const auto ta = std::chrono::steady_clock::now();
        ArmP(T, rng);
        ++arms;
        std::cout << "  arm P  checks=" << T.checks << " mismatched=" << T.mismatched << " secs="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count() << "\n";
    }
    {
        const auto ta = std::chrono::steady_clock::now();
        ArmT(T, rng);
        ++arms;
        std::cout << "  arm T  checks=" << T.checks << " mismatched=" << T.mismatched << " secs="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count() << "\n";
    }
    {
        const auto ta = std::chrono::steady_clock::now();
        ArmB(T, rng);
        ++arms;
        std::cout << "  arm B  checks=" << T.checks << " mismatched=" << T.mismatched << " secs="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count() << "\n";
    }
    {
        const auto ta = std::chrono::steady_clock::now();
        ArmMR(T);
        ++arms;
        std::cout << "  arm MR checks=" << T.checks << " mismatched=" << T.mismatched << " secs="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count() << "\n";
    }
    const uint64_t products = bn::ntt_products() - p0, steps = bn::barrett_steps() - b0;
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "EFFECTIVE BigIntNttValidation ntt_products=" << products << " barrett_steps=" << steps
              << " min_limbs_default=" << heckez::finder::BigNttMinLimbs() << "\n";
    const bool ok = T.mismatched == 0 && products > 0 && steps > 0;
    std::cout << "BIGNTT_VALIDATION arms=" << arms << " checks=" << T.checks << " products=" << products
              << " mismatched=" << T.mismatched << " routes_equal=" << (T.mismatched == 0 ? "yes" : "no")
              << " secs=" << secs << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
