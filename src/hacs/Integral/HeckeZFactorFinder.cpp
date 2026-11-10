// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZFactorFinder -- the untrusted factor finder of HZ-RAT (see the header). Proposals only.

#include "HeckeZFactorFinder.h"

#include "../NumberField/BigInt.h"
#include "../NumberField/ClassGroupRel.h"
#include "../NumberField/RibetField.h"

#include <map>

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

nf::BigInt PowModBig(nf::BigInt b, nf::BigInt e, const nf::BigInt& m) {
    nf::BigInt r(1);
    b = b % m;
    const nf::BigInt two(2);
    while (!e.is_zero()) {
        nf::BigInt q, rem;
        nf::BigInt::divmod(e, two, q, rem);
        if (!rem.is_zero()) r = (r * b) % m;
        b = (b * b) % m;
        e = q;
    }
    return r;
}

bool MillerRabinBig(const nf::BigInt& n, uint32_t rounds) {
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
    for (uint32_t i = 0; i < rounds && i < sizeof(kBases) / sizeof(kBases[0]); ++i) {
        const nf::BigInt a(static_cast<int64_t>(kBases[i]));
        if (n.mod_u32(kBases[i]) == 0) return n == a;
        nf::BigInt x = PowModBig(a, d, n);
        if (x == nf::BigInt(1) || x == nm1) continue;
        bool comp = true;
        for (uint32_t j = 1; j < s; ++j) {
            x = (x * x) % n;
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

BigProbable BigProbablePrime(const std::string& value_decimal, uint32_t rounds) {
    BigProbable r;
    const nf::BigInt v = nf::BigInt::from_string(value_decimal);
    if (v < nf::BigInt(2)) return r;
    if (MillerRabinBig(v, rounds)) {
        r.found = true;
        r.root_decimal = v.to_string();
        r.e = 1;
        return r;
    }
    const nf::BigInt s = nf::bigint_isqrt(v);
    if (s * s == v && MillerRabinBig(s, rounds)) {
        r.found = true;
        r.root_decimal = s.to_string();
        r.e = 2;
    }
    return r;
}

}  // namespace finder
}  // namespace heckez
