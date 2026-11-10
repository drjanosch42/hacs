// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZLocalParValidation -- slot-26 Z9 HZ-LOC-PAR (see the header).

#include "HeckeZLocalParValidation.h"

#include "HeckeZLocal.h"
#include "HeckeZLocalValidation.h"
#include "../../Engine/EngineRuntime.h"
#include "../Util/Rss.h"

#include <memory>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace heckez::local;

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

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

const uint32_t kThreads[] = {1, 4, 16};

// (S) a family of vectors with planted leading zeros, valuations and in-span combinations
std::vector<std::vector<uint64_t>> Family(const Zlk& R, size_t m, size_t count, uint64_t seed) {
    Rng g{seed};
    std::vector<std::vector<uint64_t>> out;
    for (size_t i = 0; i < count; ++i) {
        std::vector<uint64_t> v(m, 0);
        if (i >= 4 && g.below(4) == 0) {
            // a combination of two earlier vectors (often in the span already)
            const auto& a = out[g.below(out.size())];
            const auto& b = out[g.below(out.size())];
            const uint64_t ca = R.red(g.next()), cb = R.red(g.next());
            for (size_t t = 0; t < m; ++t) v[t] = R.add(R.mul(ca, a[t]), R.mul(cb, b[t]));
        } else {
            const size_t lead = g.below(4) == 0 ? 0 : static_cast<size_t>(g.below(m / 2));
            const uint64_t s = R.pw(static_cast<uint32_t>(g.below(R.K)));
            const uint64_t density = 1 + g.below(4);
            for (size_t t = lead; t < m; ++t)
                if (g.below(density) == 0) v[t] = R.mul(R.red(g.next()), s);
        }
        out.push_back(std::move(v));
    }
    return out;
}

struct SpanRun {
    std::vector<bool> grew;
    std::vector<std::vector<uint64_t>> rows;
    std::vector<bool> member;
    uint32_t length = 0;
};

SpanRun BuildSpan(const Zlk& R, size_t m, const std::vector<std::vector<uint64_t>>& vs,
                  const std::vector<std::vector<uint64_t>>& probes) {
    SpanRun r;
    Span S(R, m);
    for (const auto& v : vs) r.grew.push_back(S.insert(v));
    r.rows = S.rows();
    for (const auto& p : probes) r.member.push_back(S.contains(p));
    r.length = S.length();
    return r;
}

bool SameSpanRun(const SpanRun& a, const SpanRun& b) {
    return a.grew == b.grew && a.rows == b.rows && a.member == b.member && a.length == b.length;
}

// every row of A lies in B
bool Within(const Span& A, const Span& B) {
    for (const auto& row : A.rows())
        if (!B.contains(row)) return false;
    return true;
}

std::vector<Mat> RandomGens(const Zlk& R, size_t D, size_t G, uint64_t seed) {
    Rng g{seed};
    std::vector<Mat> gens;
    for (size_t k = 0; k < G; ++k) {
        Mat A(D * D, 0);
        for (auto& x : A)
            if (g.below(3) == 0) x = R.mul(R.red(g.next()), R.pw(static_cast<uint32_t>(g.below(R.K))));
        gens.push_back(std::move(A));
    }
    // commuting: the second generator is a polynomial in the first
    if (G >= 2) gens[1] = MatAdd(R, MatMul(R, gens[0], gens[0], D), MatScale(R, gens[0], 3));
    return gens;
}

// the gate line and DIGEST of one HeckeZLocalValidation run, its stdout captured
struct CellsRun { std::string digest; bool ok = false; uint32_t cells = 0; uint32_t rc = 0; };
CellsRun RunCells() {
    std::ostringstream cap;
    std::streambuf* saved = std::cout.rdbuf(cap.rdbuf());
    CellsRun r;
    try {
        r.rc = HeckeZLocalValidation();
    } catch (...) {
        std::cout.rdbuf(saved);
        throw;
    }
    std::cout.rdbuf(saved);
    std::istringstream in(cap.str());
    std::string line;
    while (std::getline(in, line)) {
        const size_t d = line.find("DIGEST 0x");
        if (d != std::string::npos) r.digest = line.substr(d + 7, line.find(' ', d + 7) - (d + 7));
        if (line.rfind("HECKEZ_LOCAL cells=", 0) == 0) {
            r.ok = line.find("verdict=OK") != std::string::npos;
            r.cells = static_cast<uint32_t>(std::stoul(line.substr(19)));
        }
    }
    return r;
}

double Since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

uint32_t HeckeZLocalParValidation() {
    Tally T;
    const auto t_all = std::chrono::steady_clock::now();
    bool threads_equal = true, routes_equal = true;
    uint32_t cells = 0;
    // (Z9 fix1, SF-2) a pool of < 2 workers would leave reduce_par / parallel MatMul unexercised: install a 4-worker scoped pool
    std::unique_ptr<engine::tape::ThreadPool> own_pool;
    std::unique_ptr<engine::tape::ScopedConsumerPool> own_scope;
    const uint32_t ambient_pool = engine::tape::consumer_pool().size();
    if (ambient_pool < 2) {
        own_pool = std::make_unique<engine::tape::ThreadPool>(4);
        own_scope = std::make_unique<engine::tape::ScopedConsumerPool>(own_pool.get());
    }
    const uint32_t pool = engine::tape::consumer_pool().size();
    const ParCallCounts calls0 = ParCalls();
    std::cout << "  ambient_pool=" << ambient_pool << (own_pool ? " (scoped 4-worker pool installed)" : "") << "\n";
    std::cout << "  pool=" << pool << " default_route=" << DefaultParRouteName() << "\n";
    try {
        // ---- (S) spans
        const std::vector<std::pair<uint64_t, uint32_t>> rings = {{3, 2}, {2, 5}, {7, 3}, {691, 2}, {2294797, 2}, {1798482437, 2}};
        uint64_t seed = 0x5a39;
        uint32_t s_cases = 0;
        for (const auto& rk : rings) {
            const Zlk R = Zlk::Make(rk.first, rk.second);
            for (size_t m : {size_t(300), size_t(2000)}) {
                const auto vs = Family(R, m, 70, ++seed);
                const auto probes = Family(R, m, 40, ++seed);
                SpanRun ser;
                {
                    ScopedParRoute sr(ParRoute{false, 0, false});
                    ser = BuildSpan(R, m, vs, probes);
                }
                for (uint32_t th : kThreads) {
                    ScopedParRoute sr(ParRoute{true, th, true});
                    const SpanRun par = BuildSpan(R, m, vs, probes);
                    const bool same = SameSpanRun(ser, par);
                    routes_equal = routes_equal && same;
                    T.check(same, "(S) span route identity ell=" + std::to_string(rk.first) + " K=" + std::to_string(rk.second) +
                                      " m=" + std::to_string(m) + " threads=" + std::to_string(th));
                }
                ++s_cases;
            }
        }
        std::cout << "  (S) span cases=" << s_cases << " (x " << (sizeof(kThreads) / sizeof(kThreads[0])) << " thread counts)\n";

        // ---- (C) closures, small D (serial vs par, forced)
        uint32_t c_cases = 0;
        for (const auto& rk : std::vector<std::pair<uint64_t, uint32_t>>{{3, 2}, {2, 4}, {691, 2}, {2294797, 2}}) {
            const Zlk R = Zlk::Make(rk.first, rk.second);
            for (size_t G : {size_t(1), size_t(2), size_t(3)}) {
                const size_t D = 16 + 4 * G;
                const auto gens = RandomGens(R, D, G, ++seed);
                ScopedParRoute s0(ParRoute{false, 0, false});
                const Span ser = AlgebraClosure(R, gens, D);
                std::vector<std::vector<uint64_t>> rows1;
                for (uint32_t th : kThreads) {
                    ScopedParRoute sr(ParRoute{true, th, true});
                    const Span par = AlgebraClosure(R, gens, D);
                    const bool same = par.length() == ser.length() && par.length_by_smith() == ser.length_by_smith() &&
                                      par.generators() == ser.generators() && Within(par, ser) && Within(ser, par);
                    routes_equal = routes_equal && same;
                    const std::string tag = "(C) closure ell=" + std::to_string(rk.first) + " K=" + std::to_string(rk.second) +
                                            " D=" + std::to_string(D) + " G=" + std::to_string(G) + " threads=" + std::to_string(th);
                    T.check(same, tag + ": same span as the serial route");
                    if (th == kThreads[0]) rows1 = par.rows();
                    else {
                        const bool teq = par.rows() == rows1;
                        threads_equal = threads_equal && teq;
                        T.check(teq, tag + ": row list equal to threads=1");
                    }
                }
                ++c_cases;
            }
        }
        // one natural D = 128 closure, above the size thresholds (not forced): threads 1 vs 16
        {
            const Zlk R = Zlk::Make(3, 2);
            const size_t D = 128;
            const auto gens = RandomGens(R, D, 2, ++seed);
            std::vector<std::vector<uint64_t>> r1;
            uint32_t len1 = 0;
            for (uint32_t th : {1u, 16u}) {
                ScopedParRoute sr(ParRoute{true, th, false});
                const auto t0 = std::chrono::steady_clock::now();
                const Span par = AlgebraClosure(R, gens, D);
                std::cout << "  (C) D=128 closure threads=" << th << " length=" << par.length() << " rows=" << par.size()
                          << " secs=" << Since(t0) << "\n";
                if (th == 1) {
                    r1 = par.rows();
                    len1 = par.length();
                } else {
                    const bool teq = par.rows() == r1 && par.length() == len1;
                    threads_equal = threads_equal && teq;
                    T.check(teq, "(C) D=128 closure: threads=16 row list equal to threads=1");
                }
            }
            ++c_cases;
        }
        std::cout << "  (C) closure cases=" << c_cases << "\n";

        // ---- (R) the HZ-LOC cells
        CellsRun ser;
        {
            ScopedParRoute sr(ParRoute{false, 0, false});
            ser = RunCells();
        }
        T.check(ser.ok && ser.rc == 0 && !ser.digest.empty(), "(R) the serial route's HECKEZ_LOCAL gate reads verdict=OK");
        cells = ser.cells;
        std::cout << "  (R) serial cells=" << ser.cells << " digest=" << ser.digest << "\n";
        for (uint32_t th : kThreads) {
            ScopedParRoute sr(ParRoute{true, th, true});
            const CellsRun par = RunCells();
            std::cout << "  (R) par threads=" << th << " cells=" << par.cells << " digest=" << par.digest << "\n";
            T.check(par.ok && par.rc == 0, "(R) par threads=" + std::to_string(th) + ": HECKEZ_LOCAL verdict=OK");
            const bool same = par.digest == ser.digest && par.cells == ser.cells;
            routes_equal = routes_equal && same;
            T.check(same, "(R) par threads=" + std::to_string(th) + ": DIGEST equal to the serial route's");
        }
    } catch (const std::exception& e) {
        std::cout << "HeckeZLocalParValidation: EXCEPTION: " << e.what() << std::endl;
        ++T.failing;
    }
    std::cout << "  secs=" << Since(t_all) << " rss_mib=" << oc_util::CurrentRssMiB() << "\n";
    const ParCallCounts calls1 = ParCalls();
    const uint64_t d_reduce = calls1.reduce_par - calls0.reduce_par, d_matmul = calls1.matmul_par - calls0.matmul_par;
    T.check(pool < 2 || (d_reduce > 0 && d_matmul > 0), "the parallel path ran (par_reduce_calls and par_matmul_calls > 0)");
    std::cout << "HECKEZ_LOCAL_PAR threads_max=" << pool << " par_reduce_calls=" << d_reduce << " par_matmul_calls=" << d_matmul << "\n";
    std::cout << "HECKEZ_LOCAL_PAR cells=" << cells << " threads_equal=" << (threads_equal ? "yes" : "no")
              << " routes_equal=" << (routes_equal ? "yes" : "no") << " default_route=" << DefaultParRouteName()
              << " verdict=" << (T.failing == 0 ? "OK" : "FAIL") << "\n";
    std::cout << "HeckeZLocalParValidation: " << T.checks << " checks, " << T.failing << " failing\n";
    return T.failing;
}
