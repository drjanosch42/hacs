// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZFactorFinder -- the untrusted factor finder of HZ-RAT (see the header). Proposals only.

#include "HeckeZFactorFinder.h"

#include "../NumberField/BigInt.h"
#include "../NumberField/BigNtt.h"
#include "../NumberField/ClassGroupRel.h"
#include "../NumberField/RibetField.h"

#include <cstdlib>
#include <map>
#include <memory>

namespace heckez {
namespace finder {

bool FindWordFactors(uint64_t c, std::vector<std::pair<uint64_t, unsigned>>& out) {
    out.clear();
    if (c <= 1) return c == 1;
    return nf::factor_u64(c, out);
}

namespace {
bool FindPrattRec(uint64_t q, uint32_t witness_cap, uint32_t node_cap, uint32_t* nodes, PrattCert* out) {
    if (++*nodes > node_cap) return false;
    out->q = q;
    out->factors.clear();
    out->sub.clear();
    out->witness = 0;
    if (q < 3 || !nf::is_prime_u64(q)) return false;
    std::vector<std::pair<uint64_t, unsigned>> fs;
    if (!nf::factor_u64(q - 1, fs)) return false;
    for (const auto& f : fs) out->factors.emplace_back(f.first, static_cast<uint32_t>(f.second));
    for (uint64_t a = 2; a < 2 + witness_cap && a < q; ++a) {
        if (PowModU64(a, q - 1, q) != 1) return false;   // q is not prime after all
        bool ok = true;
        for (const auto& f : fs)
            if (PowModU64(a, (q - 1) / f.first, q) == 1) { ok = false; break; }
        if (ok) { out->witness = a; break; }
    }
    if (out->witness == 0) return false;
    for (const auto& f : fs) {
        if (f.first <= kSmallBound) continue;
        PrattCert sub;
        if (!FindPrattRec(f.first, witness_cap, node_cap, nodes, &sub)) return false;
        out->sub.push_back(std::move(sub));
    }
    return true;
}

// x^e mod n for the Miller--Rabin test. Slot-27 BIGNTT M1: the x y mod n primitive is nf::bigntt::BarrettModulus
// (an NTT product with Barrett's reduction above the route threshold; BigInt's own (x y) % n below it). The control
// flow -- right to left over the bits of e, r = r b on a set bit, then b = b^2 -- and so every value is today's.
nf::BigInt PowModBig(const nf::BigInt& b, const nf::BigInt& e, const nf::bigntt::BarrettModulus& M) {
    return nf::bigntt::powmod(b, e, M);
}

bool MillerRabinBig(const nf::BigInt& n, uint32_t rounds, std::size_t min_limbs, unsigned threads) {
    static const uint32_t kBases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
    if (n < nf::BigInt(2)) return false;
    nf::BigInt d = n - nf::BigInt(1);
    uint32_t s = 0;
    const nf::BigInt two(2);
    for (;;) {
        nf::BigInt q, r;
        nf::BigInt::divmod(d, two, q, r);
        if (!r.is_zero()) break;
        d = q;
        ++s;
    }
    const nf::BigInt nm1 = n - nf::BigInt(1);
    std::unique_ptr<nf::bigntt::BarrettModulus> M;   // built on the first base that needs a powmod
    for (uint32_t i = 0; i < rounds && i < sizeof(kBases) / sizeof(kBases[0]); ++i) {
        const nf::BigInt a(static_cast<int64_t>(kBases[i]));
        if (n.mod_u32(kBases[i]) == 0) return n == a;
        if (!M) M.reset(new nf::bigntt::BarrettModulus(n, min_limbs, threads));
        nf::BigInt x = PowModBig(a, d, *M);
        if (x == nf::BigInt(1) || x == nm1) continue;
        bool comp = true;
        for (uint32_t j = 1; j < s; ++j) {
            x = M->sqrmod(x);
            if (x == nm1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}
}  // namespace

bool FindPratt(uint64_t q, uint32_t witness_cap, uint32_t node_cap, PrattCert* out) {
    uint32_t nodes = 0;
    return FindPrattRec(q, witness_cap, node_cap, &nodes, out);
}

std::size_t BigNttMinLimbs() {
    static const std::size_t v = [] {
        const char* e = std::getenv("ENGINE_BIGNTT_MIN_LIMBS");   // a route knob: it cannot change a value
        if (!e || !*e) return nf::bigntt::kDefaultMinLimbs;
        char* end = nullptr;
        const unsigned long long x = std::strtoull(e, &end, 10);
        return (end && *end == '\0' && x > 0) ? static_cast<std::size_t>(x) : nf::bigntt::kDefaultMinLimbs;
    }();
    return v;
}

bool MillerRabinBigForTest(const nf::BigInt& n, uint32_t rounds, std::size_t min_limbs, unsigned threads) {
    return MillerRabinBig(n, rounds, min_limbs, threads);
}

BigProbable BigProbablePrime(const std::string& value_decimal, uint32_t rounds, unsigned threads, std::size_t min_limbs) {
    BigProbable r;
    if (min_limbs == 0) min_limbs = BigNttMinLimbs();
    const nf::BigInt v = nf::BigInt::from_string(value_decimal);
    if (v < nf::BigInt(2)) return r;
    if (MillerRabinBig(v, rounds, min_limbs, threads)) {
        r.found = true;
        r.root_decimal = v.to_string();
        r.e = 1;
        return r;
    }
    const nf::BigInt s = nf::bigint_isqrt(v);
    if (s * s == v && MillerRabinBig(s, rounds, min_limbs, threads)) {
        r.found = true;
        r.root_decimal = s.to_string();
        r.e = 2;
    }
    return r;
}

}  // namespace finder
}  // namespace heckez
