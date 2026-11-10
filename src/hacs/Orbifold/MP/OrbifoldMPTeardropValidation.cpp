// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, MP-T) OrbifoldMPTeardropValidation -- see the header.
// A port of docs/implementation/slot22/mp/teardrop.py (frozen output mp/out_checks.txt, run `teardrop.py 4`).
// Gate values: docs/implementation/slot22/MP_predictions.md P3.
#include "OrbifoldMPTeardropValidation.h"

#include "SparseLength.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using orbifold_mp::SparseRows;

struct Tally {
    uint32_t checks = 0, failing = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [FAIL] " << what << "\n";
        }
    }
};

constexpr uint32_t kKmax = 4;            // S_0 .. S_4
constexpr uint32_t kTop = 6;             // top simplices a..f = 0..5
constexpr uint8_t A = 0, C = 2;          // the cone simplices a, b, c are 0, 1, 2

// A string of S_k: chain of top simplices (k+1 entries) with labels g_1..g_k in C_n (k entries).
struct Str {
    uint8_t k = 0;
    std::array<uint8_t, kKmax + 1> ch{};
    std::array<uint8_t, kKmax> g{};
};

bool InCone(const Str& s) {
    for (uint32_t i = 0; i <= s.k; ++i)
        if (s.ch[i] > C) return false;
    return true;
}

// The five vertex stars t,u,v,w,x as bitmasks over {a..f}: abde, bcef, acdf, abc, def.
constexpr uint32_t kStars[5] = {0b011011, 0b110110, 0b101101, 0b000111, 0b111000};

bool ChainOk(const Str& s) {
    uint32_t m = 0;
    for (uint32_t i = 0; i <= s.k; ++i) m |= 1u << s.ch[i];
    for (uint32_t st : kStars)
        if ((m & ~st) == 0) return true;
    return false;
}

// The mu-maps and the group: labels in Z/n; teardrop.py gives a consecutive pair (a,c) coming from a chain
// through v the exponent n-1 and (c,a) the exponent 1.
struct Model {
    uint32_t n = 2;
    uint32_t mu_ac = 1;
    uint32_t mu_ca = 1;
};

Model Real(uint32_t n) { return Model{n, (n - 1) % n, 1 % n}; }
Model Swapped(uint32_t n) { return Model{n, 1 % n, (n - 1) % n}; }   // NC1b (an isomorphic twin, not a failure)
// NC2: the exponent n = (n+1)-1 that an order-(n+1) cone point would give, read in C_n, i.e. (mu_ac, mu_ca) = (1, rho):
// a second non-inverse pair (it bites through the same cocycle condition as NC1; it does not model a C_{n+1} vertex group).
Model WrongOrder(uint32_t n) { return Model{n, n % n, 1 % n}; }
// NC4 (audit): the trivial twisting (1, 1) satisfies d o d = 0 (0 + 0 = 0 mod n) but is not MP's complex of groups;
// it must fail the dimension gate (it gives H^1 = 1, H^2 = 2), so the gate, not only d o d, separates the genuine mu.
Model Trivial(uint32_t n) { return Model{n, 0, 0}; }

// teardrop.py labels_after.
Str LabelsAfter(const Str& old_s, Str nw, const Model& M) {
    if (InCone(nw)) {
        if (InCone(old_s)) return nw;
        for (uint32_t i = 0; i < nw.k; ++i) {
            const uint8_t x = nw.ch[i], y = nw.ch[i + 1];
            if (x == A && y == C) nw.g[i] = static_cast<uint8_t>(M.mu_ac);
            else if (x == C && y == A) nw.g[i] = static_cast<uint8_t>(M.mu_ca);
            else nw.g[i] = 0;
        }
        return nw;
    }
    for (uint32_t i = 0; i < nw.k; ++i) nw.g[i] = 0;
    return nw;
}

// The k+1 faces d_0 .. d_k of a string of S_k, k >= 1 (teardrop.py faces).
std::vector<Str> Faces(const Str& s, const Model& M) {
    const uint32_t k = s.k, n = M.n;
    std::vector<Str> out;
    {   // d_0: drop the first top simplex and the first label
        Str t;
        t.k = static_cast<uint8_t>(k - 1);
        for (uint32_t i = 0; i < k; ++i) t.ch[i] = s.ch[i + 1];
        for (uint32_t i = 0; i + 1 < k; ++i) t.g[i] = s.g[i + 1];
        out.push_back(LabelsAfter(s, t, M));
    }
    for (uint32_t i = 1; i < k; ++i) {   // inner faces: drop ch[i], multiply g_i g_{i+1}
        Str t;
        t.k = static_cast<uint8_t>(k - 1);
        uint32_t w = 0;
        for (uint32_t j = 0; j <= k; ++j)
            if (j != i) t.ch[w++] = s.ch[j];
        for (uint32_t j = 0; j + 1 < i; ++j) t.g[j] = s.g[j];
        t.g[i - 1] = static_cast<uint8_t>((s.g[i - 1] + s.g[i]) % n);
        for (uint32_t j = i + 1; j < k; ++j) t.g[j - 1] = s.g[j];
        out.push_back(LabelsAfter(s, t, M));
    }
    {   // d_k
        Str t;
        t.k = static_cast<uint8_t>(k - 1);
        for (uint32_t i = 0; i < k; ++i) t.ch[i] = s.ch[i];
        for (uint32_t i = 0; i + 1 < k; ++i) t.g[i] = s.g[i];
        out.push_back(LabelsAfter(s, t, M));
    }
    return out;
}

uint64_t Key(const Str& s, uint32_t n) {
    uint64_t c = 0;
    for (uint32_t i = 0; i <= s.k; ++i) c = c * kTop + s.ch[i];
    for (uint32_t i = 0; i < s.k; ++i) c = c * n + s.g[i];
    return c;
}

bool Degenerate(const Str& s) {
    for (uint32_t i = 0; i < s.k; ++i)
        if (s.ch[i] == s.ch[i + 1] && s.g[i] == 0) return true;
    return false;
}

// All strings of S_k (degenerate ones included), in teardrop.py's order.
std::vector<Str> Simplices(uint32_t k, uint32_t n) {
    std::vector<Str> out;
    uint32_t nch = 1;
    for (uint32_t i = 0; i <= k; ++i) nch *= kTop;
    for (uint32_t c = 0; c < nch; ++c) {
        Str s;
        s.k = static_cast<uint8_t>(k);
        uint32_t x = c;
        for (int i = static_cast<int>(k); i >= 0; --i) {
            s.ch[i] = static_cast<uint8_t>(x % kTop);
            x /= kTop;
        }
        if (!ChainOk(s)) continue;
        if (InCone(s)) {
            uint32_t ng = 1;
            for (uint32_t i = 0; i < k; ++i) ng *= n;
            for (uint32_t gc = 0; gc < ng; ++gc) {
                Str t = s;
                uint32_t y = gc;
                for (int i = static_cast<int>(k) - 1; i >= 0; --i) {
                    t.g[i] = static_cast<uint8_t>(y % n);
                    y /= n;
                }
                out.push_back(t);
            }
        } else {
            out.push_back(s);   // trivial labels
        }
    }
    return out;
}

struct Complex {
    std::vector<std::vector<Str>> nd;   // non-degenerate strings, S_0 .. S_4
    std::vector<std::unordered_map<uint64_t, uint32_t>> idx;
};

Complex Build(uint32_t n) {
    Complex X;
    for (uint32_t k = 0; k <= kKmax; ++k) {
        std::vector<Str> v;
        for (const Str& s : Simplices(k, n))
            if (!Degenerate(s)) v.push_back(s);
        std::unordered_map<uint64_t, uint32_t> m;
        for (uint32_t i = 0; i < v.size(); ++i) m[Key(v[i], n)] = i;
        X.nd.push_back(std::move(v));
        X.idx.push_back(std::move(m));
    }
    return X;
}

// rank of d^k : C^k -> C^{k+1} over F_ell (rows indexed by S_{k+1}, as in teardrop.py).
uint64_t Rank(const Complex& X, const Model& M, uint32_t k, uint32_t ell) {
    SparseRows R;
    R.ncols = static_cast<uint32_t>(X.nd[k].size());
    std::vector<std::pair<uint32_t, uint32_t>> row;
    for (const Str& s : X.nd[k + 1]) {
        row.clear();
        const std::vector<Str> F = Faces(s, M);
        for (uint32_t i = 0; i < F.size(); ++i) {
            const auto it = X.idx[k].find(Key(F[i], M.n));
            if (it == X.idx[k].end()) continue;   // degenerate face: f := 0
            row.emplace_back(it->second, (i % 2 == 0) ? 1u : ell - 1u);
        }
        std::sort(row.begin(), row.end());
        uint32_t last = UINT32_MAX, acc = 0;
        for (size_t t = 0; t <= row.size(); ++t) {
            if (t < row.size() && row[t].first == last) {
                acc = (acc + row[t].second) % ell;
                continue;
            }
            if (last != UINT32_MAX && acc) {
                R.col.push_back(last);
                R.val.push_back(static_cast<uint8_t>(acc));
            }
            if (t < row.size()) {
                last = row[t].first;
                acc = row[t].second % ell;
            }
        }
        R.start.push_back(R.col.size());
    }
    return orbifold_mp::RowSpanLength(R, ell, 1).length;
}

// dim H^0 .. H^3.
std::vector<int64_t> Dims(const Complex& X, const Model& M, uint32_t ell) {
    std::vector<uint64_t> r;
    for (uint32_t k = 0; k < kKmax; ++k) r.push_back(Rank(X, M, k, ell));
    std::vector<int64_t> d;
    for (uint32_t q = 0; q < kKmax; ++q)
        d.push_back(static_cast<int64_t>(X.nd[q].size()) - static_cast<int64_t>(r[q]) -
                    (q ? static_cast<int64_t>(r[q - 1]) : 0));
    return d;
}

// Exhaustive d o d = 0 on every non-degenerate string of S_2 .. S_4: the formal sum
// sum_{i,j} (-1)^{i+j} d_i d_j s over ALL strings (degenerate included) must vanish.
void DdCheck(const Complex& X, const Model& M, uint64_t* bad, uint64_t* total) {
    *bad = 0;
    *total = 0;
    for (uint32_t k = 2; k <= kKmax; ++k)
        for (const Str& s : X.nd[k]) {
            ++*total;
            std::map<uint64_t, int> tot;
            const std::vector<Str> F = Faces(s, M);
            for (uint32_t i = 0; i < F.size(); ++i) {
                const std::vector<Str> FF = Faces(F[i], M);
                for (uint32_t j = 0; j < FF.size(); ++j) tot[Key(FF[j], M.n)] += ((i + j) % 2 == 0) ? 1 : -1;
            }
            for (const auto& kv : tot)
                if (kv.second != 0) {
                    ++*bad;
                    break;
                }
        }
}

}  // namespace

uint32_t OrbifoldMPTeardropValidation() {
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();
    Tally T;

    // P3: |S_0..S_4| (non-degenerate), per n.
    const std::map<uint32_t, std::vector<uint64_t>> sizes = {
        {2, {6, 39, 177, 705, 2877}}, {3, {6, 48, 294, 1866, 13290}}, {4, {6, 57, 465, 4323, 44925}}};
    // P3: dim H^0..H^3 per (n, ell): A, 0, A, H^3(C_n; F_ell).
    auto expect = [](uint32_t n, uint32_t ell) { return std::vector<int64_t>{1, 0, 1, (n % ell == 0) ? 1 : 0}; };

    uint32_t cells = 0, agree = 0;
    for (uint32_t n : {2u, 3u, 4u}) {
        const Complex X = Build(n);
        std::vector<uint64_t> got;
        for (const auto& v : X.nd) got.push_back(v.size());
        for (uint32_t k = 0; k <= kKmax; ++k)
            T.check(got[k] == sizes.at(n)[k], "|S_" + std::to_string(k) + "| nondegenerate, Tear-" + std::to_string(n));
        const Model M = Real(n);
        uint64_t bad = 0, tot = 0;
        DdCheck(X, M, &bad, &tot);
        T.check(bad == 0 && tot == got[2] + got[3] + got[4],
                "d o d = 0 exhaustively on S_2..S_4, Tear-" + std::to_string(n) + " (bad=" + std::to_string(bad) + ")");
        std::cout << "  Tear-" << n << " |S_0..S_4|=[" << got[0] << "," << got[1] << "," << got[2] << "," << got[3] << ","
                  << got[4] << "] d^2=0 exhaustive over " << tot << " strings (bad=" << bad << ")\n";
        for (uint32_t ell : {2u, 3u, 5u}) {
            const std::vector<int64_t> d = Dims(X, M, ell);
            const std::vector<int64_t> e = expect(n, ell);
            ++cells;
            bool ok = true;
            for (uint32_t q = 0; q < 4; ++q) {
                const bool qok = d[q] == e[q];
                ok = ok && qok;
                T.check(qok, "dim H^" + std::to_string(q) + " Tear-" + std::to_string(n) + " ell=" + std::to_string(ell));
            }
            agree += ok ? 1u : 0u;
            std::cout << "  Tear-" << n << " ell=" << ell << " dim H^0..3=[" << d[0] << "," << d[1] << "," << d[2] << ","
                      << d[3] << "] expected=[" << e[0] << "," << e[1] << "," << e[2] << "," << e[3] << "] "
                      << (ok ? "OK" : "MISMATCH") << "\n";
        }
        // NC1 (bites): the two mu-maps must be inverse on the two orientations of the pair (a, c): d o d = 0 holds iff
        // mu_ac + mu_ca = 0 mod n (all n^2 exponent pairs were scanned when this was written).  Both exponents
        // equal to 1 violates it for n >= 3 (at n = 2, 1 + 1 = 0 is the genuine mu, so NC1 is skipped).
        // NC1b (does NOT bite, by design): the SWAPPED pair (rho <-> rho^{n-1}) is the genuine mu composed with the
        // automorphism g -> g^{-1} of C_n; it satisfies d o d = 0 with the same dimensions.  MP_predictions.md P3's
        // "swapped mu breaks d o d = 0" is therefore false as worded (deviation note in MPT_report.md).
        if (n >= 3) {
            DdCheck(X, Model{n, 1, 1}, &bad, &tot);
            T.check(bad > 0, "NC1: mu_ac = mu_ca = rho (not inverse) breaks d o d = 0, Tear-" + std::to_string(n));
            std::cout << "  NC1 Tear-" << n << " mu_ac=mu_ca=1: d^2 != 0 on " << bad << " of " << tot << " strings\n";
            DdCheck(X, Swapped(n), &bad, &tot);
            const uint32_t pl = (n % 2 == 0) ? 2u : n;   // n in {3, 4}: the prime dividing n
            const std::vector<int64_t> ds = Dims(X, Swapped(n), pl);
            T.check(bad == 0 && ds == expect(n, pl),
                    "NC1b: swapped mu is the g -> g^{-1} twin (d o d = 0, same dims), Tear-" + std::to_string(n));
            std::cout << "  NC1b Tear-" << n << " swapped mu: d^2 bad=" << bad << " dim H^0..3 at ell=" << pl << " = [" << ds[0]
                      << "," << ds[1] << "," << ds[2] << "," << ds[3] << "]\n";
        }
        DdCheck(X, WrongOrder(n), &bad, &tot);
        T.check(bad > 0, "NC2: mu from isotropy order n+1 breaks d o d = 0, Tear-" + std::to_string(n));
        std::cout << "  NC2 Tear-" << n << " mu from order " << n + 1 << ": d^2 != 0 on " << bad << " of " << tot
                  << " strings\n";
        {   // NC4: consistent (d o d = 0) but wrong twisting; the dims gate must bite.
            const uint32_t pl = (n % 2 == 0) ? 2u : n;
            DdCheck(X, Trivial(n), &bad, &tot);
            const std::vector<int64_t> dt = Dims(X, Trivial(n), pl);
            T.check(bad == 0 && dt != expect(n, pl),
                    "NC4: trivial mu keeps d o d = 0 but fails the dims gate, Tear-" + std::to_string(n));
            std::cout << "  NC4 Tear-" << n << " trivial mu: d^2 bad=" << bad << " dim H^0..3 at ell=" << pl << " = [" << dt[0]
                      << "," << dt[1] << "," << dt[2] << "," << dt[3] << "]\n";
        }
    }
    std::cout << "  ORACLE teardrop.py cells=" << cells << " agree=" << agree << "\n";
    T.check(cells == 9 && agree == 9, "ORACLE teardrop.py: 9 cells agree");

    // NC3: trivial cone isotropy (n = 1) has H^3 = 0 at ell = 2, where Tear-2 has H^3 = 1.
    {
        const Complex X = Build(1);
        const std::vector<int64_t> d = Dims(X, Real(1), 2);
        T.check(d[3] == 0 && d[3] != expect(2, 2)[3], "NC3: trivial isotropy (n=1) has H^3 = 0 at ell = 2 (Tear-2 has 1)");
        std::cout << "  NC3 Tear-1 ell=2 dim H^0..3=[" << d[0] << "," << d[1] << "," << d[2] << "," << d[3] << "]\n";
    }

    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "ORBIFOLD_MP OrbifoldMPTeardropValidation checks=" << T.checks << " failing=" << T.failing
              << " wall_s=" << wall << " model=mp\n";
    return T.failing ? 1 : 0;
}
