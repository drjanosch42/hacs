// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZRecordSource -- slot-25 HZ-REC: the shipped sources of the record builder (see the header).

#include "HeckeZRecordSource.h"

#include "HeckeHidaEisLambda.h"
#include "HeckeHidaLambda.h"
#include "HeckeZLocalSource.h"

#include <map>
#include <stdexcept>

namespace heckez {
namespace record {

RecordSources DefaultRecordSources() {
    RecordSources s;
    s.local_read = [](uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                      const local::ReadOptions& opt, uint32_t max_steps) {
        const local::CellLocalRead C = local::ReadCellLocal(N, n, ell, K, P, b1, opt, max_steps);
        LocalReadOut o;
        o.M = C.M;
        o.L = C.L;
        o.read = C.read;
        o.escalation = C.escalation;
        return o;
    };
    // (slot-25 H1) heckez.G8 = hida: the forms side of one level-1 weight (HeckeHidaEisLambda.h)
    s.hida_block = [](uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint32_t>& ops, const std::vector<uint32_t>& theta) {
        return hecke_ext::eislambda::HidaBlockAt(ell, k, W, ops, theta);
    };
    s.v_bernoulli = [](uint64_t ell, uint32_t k, uint32_t prec) -> int32_t {
        // B_k / k mod ell^prec (the Brent-Harvey route of HeckeHidaLambda), level 1, ell >= 5, (ell - 1) !| k; -1 when not served.
        // (re-audit R2) served iff the three word caps of HeckeZRecord.h hold (they are in the settings text, heckez.second_route);
        // they are decided here, before the route runs, so a refusal of BernoulliOverK past them is a defect, not a cap.
        if (ell < 5 || k % (ell - 1) == 0 || prec == 0) return -1;
        using u128 = unsigned __int128;
        // ell^m < 2^bits (bits <= 128), exact
        auto pow_below = [ell](uint32_t m, uint32_t bits) {
            const u128 lim = bits >= 128 ? ~static_cast<u128>(0) : (static_cast<u128>(1) << bits) - 1;
            u128 q = 1;
            for (uint32_t i = 0; i < m; ++i) {
                if (q > lim / ell) return false;
                q *= ell;
            }
            return true;
        };
        // ell^m + ell <= 2^bits (bits < 128): the loop guard `Q < 2^bits / ell` of ZMod / BernoulliOverK, step by step
        auto guard_ok = [ell](uint32_t m, uint32_t bits) {
            const u128 lim = (static_cast<u128>(1) << bits) / ell;
            u128 q = 1;
            for (uint32_t i = 0; i < m; ++i) {
                if (!(q < lim)) return false;
                q *= ell;
            }
            return true;
        };
        uint32_t V = 0;   // v_ell(4^(k/2) - 1) = v_ell(2^k - 1)
        {
            u128 r = 1;
            const u128 two = 2;
            // 2^k - 1 mod ell^j for increasing j while it vanishes (ell^j < 2^64 suffices: V is tiny for ell >= 5)
            uint64_t mod = ell;
            for (;;) {
                r = 1;
                for (uint32_t i = 0; i < k; ++i) r = (r * two) % mod;
                if ((r + mod - 1) % mod != 0) break;
                ++V;
                if (mod > (~0ull) / ell) break;
                mod *= ell;
            }
        }
        if (!pow_below(prec + 3, record::kSecondRouteBits) || !guard_ok(prec, record::kSecondRouteRingBits) ||
            !guard_ok(prec + V, record::kSecondRouteTangentBits))
            return -1;
        std::map<uint32_t, hecke_ext::lambda::u128> m;
        try {
            m = hecke_ext::lambda::BernoulliOverK(ell, prec, {k});
        } catch (const std::exception& e) {
            throw std::logic_error(std::string("HeckeZRecordSource: BernoulliOverK refused inside the stated caps (heckez.second_route): ") +
                                   e.what());
        }
        const auto it = m.find(k);
        if (it == m.end()) return -1;
        unsigned __int128 x = it->second;
        if (x == 0) return -1;   // v >= prec
        int32_t v = 0;
        while (x % ell == 0) {
            x /= ell;
            ++v;
        }
        return v;
    };
    return s;
}

}  // namespace record
}  // namespace heckez
