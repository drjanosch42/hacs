// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZRecordSource -- slot-25 HZ-REC: the shipped sources of the record builder (see the header).

#include "HeckeZRecordSource.h"

#include "HeckeHidaEisLambda.h"
#include "HeckeHidaLambda.h"
#include "HeckeZLocalSource.h"

#include "../Integral/Descent/DescentModuleExportZ.h"
#include "../Integral/Descent/IsoTreeCarrierZ.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace heckez {
namespace record {

namespace {
namespace mx = integral_descent::modexport;
namespace fz = integral_descent::freez;

// The cached free block of one cell (EXW).
struct FreeCache {
    std::mutex mu;
    bool have_key = false;
    uint32_t N = 0, n = 0;
    bool abp = false;
    std::unique_ptr<fz::IsoTreeCarrierZ> car;
    bool have_block = false;
    mx::FreeBlockZ G;                        // the block (have_block)
    std::vector<uint32_t> ops;               // its operators (= G.F order)
    // (EXW fix, M-EXW1) the route refusals ("REFUSED [<Name>] ..") keyed by the SORTED operator set they were refused for: a set is
    // never re-run, and a refusal never blocks a request the cached block (or another set) serves
    std::map<std::vector<uint32_t>, std::string> refused;
    std::set<uint32_t> terms_refused;        // operators refused [Terms] (design m2)
    std::string terms_text;
};

std::string RefusalNameOf(const std::string& text) {
    if (text.compare(0, 9, "REFUSED [") != 0) return "Unnamed";
    const size_t e = text.find(']', 9);
    return e == std::string::npos ? std::string("Unnamed") : text.substr(9, e - 9);
}

LocalReadOut BlockedFree(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, const std::string& text) {
    LocalReadOut o;
    o.M.N = N;
    o.M.n = n;
    o.M.ell = ell;
    o.M.K = K;
    o.M.P = P;
    o.M.source = "free_block_z";
    o.M.word = "u64";
    o.M.ok = false;
    o.M.status = "blocked(free_block)";
    o.M.detail = text;
    return o;
}
}  // namespace

std::function<LocalReadOut(uint32_t, uint32_t, uint64_t, uint32_t, const std::vector<uint32_t>&, uint32_t, const local::ReadOptions&,
                           uint32_t)>
FreeLocalRead(const FreeSourceOptions& fo) {
    auto cache = std::make_shared<FreeCache>();
    return [cache, fo](uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                       const local::ReadOptions& opt, uint32_t max_steps) -> LocalReadOut {
        std::lock_guard<std::mutex> lk(cache->mu);
        if (fo.stats) ++fo.stats->calls;
        FreeCache& C = *cache;
        if (!C.have_key || C.N != N || C.n != n || C.abp != opt.allow_bad_primes) {
            C.have_key = true;
            C.N = N;
            C.n = n;
            C.abp = opt.allow_bad_primes;
            C.car.reset();
            C.have_block = false;
            C.G = mx::FreeBlockZ{};
            C.ops.clear();
            C.refused.clear();
            C.terms_refused.clear();
            C.terms_text.clear();
        }
        std::string t_ell;
        const std::vector<uint32_t> Pr = local::ReadPrimes(N, ell, P, &t_ell);
        auto has = [](const std::vector<uint32_t>& v, uint32_t p) { return std::find(v.begin(), v.end(), p) != v.end(); };
        auto key_of = [](std::vector<uint32_t> v) {
            std::sort(v.begin(), v.end());
            return v;
        };
        auto blocked = [&](const std::string& text) {
            if (fo.stats) fo.stats->refusal = text;
            return BlockedFree(N, n, ell, K, P, text);
        };
        // the read set needs an operator refused [Terms] earlier: refused by name (the pending word stays)
        auto terms_blocked = [&]() {
            for (uint32_t p : Pr)
                if (C.terms_refused.count(p)) return true;
            return false;
        };
        if (terms_blocked()) return blocked(C.terms_text);
        bool have_all = C.have_block;
        for (uint32_t p : Pr) have_all = have_all && has(C.ops, p);
        // (M-EXW1) a request the cached block serves is served: no earlier refusal on the cell is consulted
        if (!have_all) {
            if (!C.car) C.car = std::make_unique<fz::IsoTreeCarrierZ>(N, n);
            mx::ModuleZ2Options mo;
            mo.threads = fo.threads ? fo.threads : opt.free_threads;
            mo.rss_cap_bytes = fo.rss_cap_bytes;
            mo.rss_lanes = fo.rss_lanes;   // (M-EXW2) the decision at a fixed lane count
            ENGINE_CHECK_MSG(mo.rss_lanes > 0, "FreeLocalRead: rss_lanes must be fixed (> 0) on the record's free route");
            mo.allow_bad_primes = opt.allow_bad_primes;
            auto run_once = [&](const std::vector<uint32_t>& ops, std::string* name, std::string* text, bool* terms) -> bool {
                *terms = false;
                for (uint32_t p : ops)
                    if (has(fo.force_terms, p)) {
                        *terms = true;
                        *name = "Terms";
                        *text = "REFUSED [Terms] T_" + std::to_string(p) + " (forced by the validation)";
                        return false;
                    }
                const auto t0 = std::chrono::steady_clock::now();
                const mx::FreeBlockRun run = mx::RunFreeBlock(*C.car, ops, mo);
                if (fo.stats) {
                    ++fo.stats->runs;
                    fo.stats->rss_pred = run.CR.rss_pred;
                    fo.stats->secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                }
                if (run.refusal.empty()) {
                    ENGINE_CHECK_MSG(run.G.F.size() == ops.size(), "FreeLocalRead: the free block lacks an operator");
                    C.G = run.G;
                    C.ops = ops;
                    C.have_block = true;
                    if (fo.stats) {
                        fo.stats->ops = C.ops;
                        fo.stats->F_fnv = C.G.F_fnv;
                    }
                    return true;
                }
                *text = run.refusal;
                *name = run.CR.refused ? std::string(fz::FreeBlockRefusalName(run.CR.code)) : RefusalNameOf(run.refusal);
                // (design m1) a defect on a carrier EXPORT-Z step 2 certified, never a word
                ENGINE_CHECK_MSG(*name != "Uncertified" && *name != "SingularG" && *name != "BoundViolated",
                                 std::string("FreeLocalRead: the free block over Z of (" + std::to_string(N) + ", " +
                                             std::to_string(n) + ") is refused by a defect: " + run.refusal)
                                     .c_str());
                *terms = run.CR.refused && run.CR.code == fz::FreeBlockRefusal::Terms;
                return false;
            };
            std::string last;   // the refusal of the last set tried
            // one operator set: true iff it is served now (the cached block is then that set); a refused set is cached by its key
            auto attempt = [&](const std::vector<uint32_t>& want) -> bool {
                const auto hit = C.refused.find(key_of(want));
                if (hit != C.refused.end()) {
                    last = hit->second;
                    return false;
                }
                std::string name, text;
                bool terms = false;
                if (run_once(want, &name, &text, &terms)) return true;
                // (design m2) a [Terms] refusal with appended T_ell's: once more without them
                std::vector<uint32_t> base, extra;
                for (uint32_t p : want) (has(P, p) || has(C.ops, p) ? base : extra).push_back(p);
                if (terms && !extra.empty()) {
                    const auto hb = C.refused.find(key_of(base));
                    std::string name2, text2;
                    bool terms2 = false;
                    if (hb == C.refused.end()) {
                        if (run_once(base, &name2, &text2, &terms2)) {
                            for (uint32_t p : extra) C.terms_refused.insert(p);
                            C.terms_text = text;
                        } else {
                            C.refused[key_of(base)] = text2;
                        }
                    }
                }
                C.refused[key_of(want)] = text;
                last = text;
                return false;
            };
            // the union (the cached block keeps serving the earlier requests), then the read set alone
            std::vector<uint32_t> uni = C.have_block ? C.ops : std::vector<uint32_t>{};
            for (uint32_t p : Pr)
                if (!has(uni, p)) uni.push_back(p);
            bool served = attempt(uni);
            if (!served && !terms_blocked() && key_of(uni) != key_of(Pr)) served = attempt(Pr);
            if (terms_blocked()) return blocked(C.terms_text);
            if (!served) return blocked(last);
        }
        local::FreeSource fs;
        fs.G = &C.G;
        fs.ops = C.ops;
        const local::CellLocalRead R = local::ReadCellLocal(N, n, ell, K, P, b1, opt, max_steps, nullptr, &fs);
        LocalReadOut o;
        o.M = R.M;
        o.L = R.L;
        o.read = R.read;
        o.escalation = R.escalation;
        return o;
    };
}

RecordSources DefaultRecordSources() {
    RecordSources s;
    s.local_read_free = FreeLocalRead();   // (slot-27 EXW) called only where rule (alpha) fires
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
