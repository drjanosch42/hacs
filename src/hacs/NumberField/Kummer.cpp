// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Kummer.cpp -- see Kummer.h (slot-19 WP-NF). A C++ port of R18-2's layer1.py /
// layer1_big.py and of R19-3's eqlayer2.py (the equivariant Z/ell^2 layer).
#include "Kummer.h"

#include <algorithm>
#include <sstream>

#include "ModUtil.h"
#include "NfCheck.h"

namespace nf {

bool ell_divides_bernoulli(uint32_t ell, uint32_t i) {
    NF_CHECK(i % (ell - 1) != 0);
    const uint32_t m = static_cast<uint32_t>(ipow_u64(ell, 2));
    uint64_t s = 0;
    for (uint32_t a = 1; a < ell; ++a) s = (s + powmod32(a, i, m)) % m;
    NF_CHECK(s % ell == 0);
    return (s / ell) % ell == 0;
}

ExactEta exact_eta_layer1(uint32_t ell, uint32_t i) {
    ExactEta r;
    r.ell = ell;
    r.i = i;
    r.exponents = kummer_exponents(ell, 1, i);
    CycloZ R(ell, 1);
    r.coeffs = R.product_unit(r.exponents);
    BigInt mx(0);
    for (const auto& c : r.coeffs) {
        if (c.abs() > mx) mx = c.abs();
    }
    r.max_digits = mx.decimal_digits();
    return r;
}

namespace {

uint64_t teich64(uint64_t x, uint32_t ell, uint64_t mod) {
    uint64_t r = 1 % mod, b = x % ell, e = ipow_u64(ell, 6);
    while (e) {
        if (e & 1u) r = mulmod64(r, b, mod);
        e >>= 1;
        if (e) b = mulmod64(b, b, mod);
    }
    return r;
}

uint64_t powmod64(uint64_t a, uint64_t e, uint64_t mod) {
    uint64_t r = 1 % mod;
    a %= mod;
    while (e) {
        if (e & 1u) r = mulmod64(r, a, mod);
        e >>= 1;
        if (e) a = mulmod64(a, a, mod);
    }
    return r;
}

}  // namespace

Layer1Result kummer_layer1(uint32_t ell, uint32_t i, unsigned prec, unsigned n_split) {
    NF_CHECK(prec >= 2 && i % 2 == 0 && i >= 2 && i + 2 < ell + 1);
    Layer1Result r;
    r.ell = ell;
    r.i = i;
    const uint64_t M64 = ipow_u64(ell, prec);
    NF_CHECK_MSG(M64 <= 0xffffffffull, "kummer_layer1: ell^prec must fit in 32 bits");
    CycloModRing R(ell, 1, static_cast<uint32_t>(M64));
    const auto f = kummer_exponents(ell, 1, i);
    const auto eta = R.product_unit(f);
    r.v_pi = R.vone(R.pow(eta, ell - 1));
    r.unramified = r.v_pi >= ell;
    r.lambda_split = r.v_pi >= ell + 1;
    r.split_primes = split_primes(ell, n_split);
    r.not_lth_power = true;
    r.eigen_ok = true;
    r.unit_ok = true;
    for (uint32_t q : r.split_primes) {
        const uint32_t rt = root_of_order(q, ell);
        const uint32_t h = (q - 1) / ell;
        const uint32_t base = powmod32(eval_product_unit(f, rt, q), h, q);
        if (base == 1) r.not_lth_power = false;
        const uint32_t tmax = ell <= 101 ? ell - 1 : 12;
        for (uint32_t t = 2; t <= tmax; ++t) {
            const uint32_t lhs = powmod32(eval_product_unit(f, powmod32(rt, t, q), q), h, q);
            if (lhs != powmod32(base, powmod32(t, i, ell), q)) r.eigen_ok = false;
        }
        if (ell <= 101) {
            uint64_t nm = 1;
            for (uint32_t t = 1; t < ell; ++t) nm = nm * eval_product_unit(f, powmod32(rt, t, q), q) % q;
            if (nm != 1 && nm != q - 1) r.unit_ok = false;
        }
    }
    r.ell_divides_Bi = ell_divides_bernoulli(ell, i);
    r.oracle_agrees = (r.v_pi > i) == r.ell_divides_Bi;
    r.products = R.products();
    return r;
}

unsigned layer1_vpi_of(uint32_t ell, const std::vector<BigInt>& eta, unsigned prec) {
    const uint64_t M64 = ipow_u64(ell, prec);
    NF_CHECK(M64 <= 0xffffffffull);
    CycloModRing R(ell, 1, static_cast<uint32_t>(M64));
    return R.vone(R.pow(R.from_bigint(eta), ell - 1));
}

std::string Layer2Result::summary() const {
    std::ostringstream os;
    os << "ell=" << ell << " w=" << w << " lifts=" << rows.size() << " unramified_kappas=[";
    for (std::size_t i = 0; i < unramified_kappas.size(); ++i) os << (i ? "," : "") << unramified_kappas[i];
    os << "] levels=" << L1 << "," << L2 << " thresholds=" << thr1 << "," << thr2 << " v_x0=" << v_x0
       << " grouping_identity=" << (grouping_identity ? 1 : 0) << " products=" << products;
    return os.str();
}

Layer2Result kummer_layer2_equivariant(uint32_t ell, uint32_t w, const Layer2Options& opt) {
    NF_CHECK(is_prime_u32(ell) && ell >= 5 && w % 2 == 0 && w >= 2 && w < ell - 1);
    Layer2Result res;
    res.ell = ell;
    res.w = w;
    const uint32_t m = static_cast<uint32_t>(ipow_u64(ell, 2));
    const uint32_t n = ell * (ell - 1);
    const uint32_t e = n;
    const uint64_t M64 = ipow_u64(ell, 3);
    NF_CHECK_MSG(M64 <= 0xffffffffull, "layer 2: ell^3 must fit in 32 bits");
    const uint64_t M4 = ipow_u64(ell, 4);
    CycloModRing R(ell, 2, static_cast<uint32_t>(M64));
    using Elt = CycloModRing::Elt;

    // primitive root mod ell^2, Delta = <h>, the CRT factors of Delta
    const std::vector<uint32_t> primes = prime_factors_u32(ell - 1);
    uint32_t g = 0;
    for (uint32_t c = 2; c < m && !g; ++c) {
        if (c % ell == 0) continue;
        bool ok = powmod32(c, ell - 1, m) != 1;
        for (uint32_t p : primes) {
            if (powmod32(c, (ell - 1) / p, ell) == 1) ok = false;
        }
        if (ok) g = c;
    }
    NF_CHECK(g != 0);
    res.g = g;
    const uint32_t h = powmod32(g, ell, m);
    struct Stage {
        uint32_t q, hq;
        uint64_t dlog;
    };
    std::vector<Stage> stages;
    for (uint32_t p : primes) {
        uint32_t q = 1;
        while ((ell - 1) % (q * p) == 0) q *= p;
        res.delta_factors.push_back(q);
        stages.push_back({q, powmod32(h, (ell - 1) / q, m), static_cast<uint64_t>(ell) * ((ell - 1) / q)});
    }
    const uint32_t gam = 1 + ell;

    // (1) the group-ring identity Theta_kappa == E_w (S - kappa ell T) mod ell^2
    if (opt.grouping_identity) {
        std::vector<uint32_t> powg(n), dlog(m, 0);
        uint64_t x = 1;
        for (uint32_t k = 0; k < n; ++k) {
            dlog[x] = k;
            powg[k] = static_cast<uint32_t>(x);
            x = x * g % m;
        }
        std::vector<uint64_t> V(n, 0);
        V[0] = 1;
        for (const Stage& st : stages) {
            const uint64_t chi = powmod32(st.hq, (ell - 1 - w % (ell - 1)) % (ell - 1), m);
            std::vector<uint64_t> Rr(n, 0);
            uint64_t a = 1;
            for (uint32_t t = 0; t < st.q; ++t) {
                const uint64_t sh = st.dlog * t % n;
                for (uint32_t i = 0; i < n; ++i) Rr[i] = (Rr[i] + a * V[(i + n - sh) % n]) % m;
                a = a * chi % m;
            }
            V.swap(Rr);
        }
        const uint64_t sg = dlog[gam];
        std::vector<uint64_t> S(n, 0), T(n, 0);
        for (uint32_t j = 0; j < ell; ++j) {
            const uint64_t sh = sg * j % n;
            for (uint32_t i = 0; i < n; ++i) {
                const uint64_t rv = V[(i + n - sh) % n];
                S[i] = (S[i] + rv) % m;
                T[i] = (T[i] + j * rv) % m;
            }
        }
        bool ok = true;
        for (uint32_t t = 0; t < ell && ok; ++t) {
            const uint64_t kap = (w + static_cast<uint64_t>(ell - 1) * t) % n;
            const uint64_t kl = kap * ell % m;
            for (uint32_t k = 0; k < n; ++k) {
                const uint64_t tgt = powmod32(powg[k], (n - kap % n) % n, m);
                const uint64_t got = (S[k] + m - kl * T[k] % m) % m;
                if (tgt != got) {
                    ok = false;
                    break;
                }
            }
        }
        res.grouping_identity = ok;
    }

    auto char_coeffs = [&](uint32_t hq, uint32_t q, uint64_t mod, uint64_t scale) {
        const uint64_t chi = powmod64(teich64(hq, ell, mod), (ell - 1 - w % (ell - 1)) % (ell - 1), mod);
        std::vector<uint64_t> a(q);
        uint64_t c = 1 % mod;
        for (uint32_t t = 0; t < q; ++t) {
            a[t] = mulmod64(c, scale, mod);
            c = mulmod64(c, chi, mod);
        }
        return a;
    };

    // (2) U = E_w'(pi): a non-trivial stage first (its pi^{sum a_t} is an ell^2-th power and is dropped)
    std::vector<Stage> order = stages;
    std::stable_sort(order.begin(), order.end(), [&](const Stage& a, const Stage& b) {
        const bool ta = powmod32(a.hq, w, m) == 1, tb = powmod32(b.hq, w, m) == 1;
        if (ta != tb) return !ta;
        return a.q < b.q;
    });
    Elt X;
    for (std::size_t idx = 0; idx < order.size(); ++idx) {
        const Stage& st = order[idx];
        const std::vector<uint64_t> a = char_coeffs(st.hq, st.q, m, 1);
        std::vector<Elt> bases;
        std::vector<std::pair<const Elt*, uint64_t>> terms;
        if (idx == 0) {
            NF_CHECK(powmod32(st.hq, w, m) != 1);
            uint64_t sum = 0;
            for (uint64_t v : a) sum = (sum + v) % m;
            NF_CHECK(sum == 0);
            bases.reserve(st.q);
            for (uint32_t t = 1; t < st.q; ++t) bases.push_back(R.unit_u(powmod32(st.hq, t, m)));
            for (uint32_t t = 1; t < st.q; ++t) terms.emplace_back(&bases[t - 1], a[t]);
        } else {
            bases.reserve(st.q);
            for (uint32_t t = 0; t < st.q; ++t) bases.push_back(R.sigma(X, powmod32(st.hq, t, m)));
            for (uint32_t t = 0; t < st.q; ++t) terms.emplace_back(&bases[t], a[t]);
        }
        X = R.multiexp(terms);
    }
    const Elt U = X;

    // (3) the omega^w projection (exponents mod ell^4), stage by stage in the factor order
    const uint64_t inv_lm1 = [&] {
        // (ell - 1)^{-1} mod ell^4 by Euler: phi(ell^4) = ell^3 (ell - 1)
        return powmod64(ell - 1, ipow_u64(ell, 3) * (ell - 1) - 1, M4);
    }();
    auto project = [&](Elt Y) {
        for (std::size_t idx = 0; idx < stages.size(); ++idx) {
            const Stage& st = stages[idx];
            const std::vector<uint64_t> a = char_coeffs(st.hq, st.q, M4, idx == 0 ? inv_lm1 : 1);
            std::vector<Elt> bases;
            bases.reserve(st.q);
            for (uint32_t t = 0; t < st.q; ++t) bases.push_back(R.sigma(Y, powmod32(st.hq, t, m)));
            std::vector<std::pair<const Elt*, uint64_t>> terms;
            for (uint32_t t = 0; t < st.q; ++t) terms.emplace_back(&bases[t], a[t]);
            Y = R.multiexp(terms);
        }
        return Y;
    };
    const Elt WU = project(R.pow(U, ell - 1));
    const Elt x0 = project(R.add(R.one(), R.pow(R.pi(), w)));
    const Elt y1 = R.pow(x0, ell);
    const Elt y2 = R.pow(y1, ell);
    res.v_x0 = R.vone(x0);
    res.L1 = R.vone(y1);
    res.L2 = R.vone(y2);
    res.thr1 = w + (ell - 1) + e;
    res.thr2 = w + (ell - 1) + 2 * e;

    // (4) A = S(W_U), B = T(W_U) by doubling over Gamma_1 = <gamma>, gamma = 1 + ell
    Elt P = R.one(), Q = R.one();
    uint32_t mm = 0;
    {
        std::string bits;
        for (uint32_t x = ell; x; x >>= 1) bits.insert(bits.begin(), static_cast<char>('0' + (x & 1u)));
        for (char bit : bits) {
            if (mm) {
                const uint32_t gm = powmod32(gam, mm, m);
                const Elt gP = R.sigma(P, gm);
                Q = R.mul(R.mul(Q, R.sigma(Q, gm)), R.pow(gP, mm));
                P = R.mul(P, gP);
                mm *= 2;
            }
            if (bit == '1') {
                const Elt gX = R.sigma(WU, powmod32(gam, mm, m));
                if (mm) {
                    Q = R.mul(Q, R.pow(gX, mm));
                    P = R.mul(P, gX);
                } else {
                    P = gX;
                }
                ++mm;
            }
        }
    }
    NF_CHECK(mm == ell);
    const Elt& A = P;
    const Elt C = R.pow(Q, ell);

    // (5) the ell lifts
    struct PowTable {
        std::vector<Elt> p2;  // y^{2^k}
        uint32_t lead = 0;
    };
    auto make_table = [&](const Elt& y) {
        PowTable tb;
        Elt cur = y;
        unsigned bl = 0;
        for (uint32_t x = ell; x; x >>= 1) ++bl;
        for (unsigned k = 0; k < bl; ++k) {
            tb.p2.push_back(cur);
            cur = R.sqr(cur);
        }
        tb.lead = R.vone_lead(y).second;
        return tb;
    };
    const PowTable T1 = make_table(y1), T2 = make_table(y2);
    auto times_pow = [&](Elt x, const PowTable& tb, uint32_t a) {
        for (std::size_t k = 0; k < tb.p2.size(); ++k) {
            if ((a >> k) & 1u) x = R.mul(x, tb.p2[k]);
        }
        return x;
    };
    struct Greedy {
        unsigned v, best;
        uint32_t a;
    };
    auto greedy = [&](const Elt& wt, const PowTable& tb, unsigned L) -> Greedy {
        const auto vl = R.vone_lead(wt);
        if (vl.first != L) return {vl.first, vl.first, 0};
        NF_CHECK(tb.lead != 0);
        const uint32_t a = static_cast<uint32_t>(
            (static_cast<uint64_t>(ell - vl.second % ell) % ell) * invmod32(tb.lead, ell) % ell);
        return {vl.first, R.vone(times_pow(wt, tb, a)), a};
    };
    auto brute = [&](const Elt& wt, const Elt& y) {
        unsigned best = 0;
        Elt cur = wt;
        for (uint32_t a = 0; a < ell; ++a) {
            best = std::max(best, R.vone(cur));
            cur = R.mul(cur, y);
        }
        return best;
    };

    Elt cur = R.mul(A, R.pow(C, (ell - w % ell) % ell));
    for (uint32_t t = 0; t < ell; ++t) {
        Layer2Row row;
        row.t = t;
        row.kappa = (w + static_cast<uint64_t>(ell - 1) * t) % n;
        NF_CHECK(row.kappa % ell == (w + ell - t % ell) % ell);
        row.u = static_cast<uint32_t>((1 + static_cast<uint64_t>(ell) * ell - row.kappa % ell) % ell);
        const Greedy g1 = greedy(cur, T1, res.L1);
        const Greedy g2 = greedy(cur, T2, res.L2);
        row.v_w = g1.v;
        row.best_lth = g1.best;
        row.best_l2th = g2.best;
        row.a_l2 = g2.a;
        row.local_lth_power = g1.best >= res.thr1;
        row.unramified = g2.best >= res.thr2;
        if (std::find(opt.brute_ts.begin(), opt.brute_ts.end(), t) != opt.brute_ts.end()) {
            row.brute_checked = true;
            row.brute_lth = brute(cur, y1);
            row.brute_l2th = brute(cur, y2);
            row.greedy_eq_brute = row.brute_lth == row.best_lth && row.brute_l2th == row.best_l2th;
        }
        if (row.unramified) res.unramified_kappas.push_back(row.kappa);
        res.rows.push_back(row);
        cur = R.mul(cur, C);
    }

    // (6) global tests on the canonical generator of each unramified lift
    if (opt.global_tests) {
        for (uint64_t kap : res.unramified_kappas) {
            Layer2Global gl;
            gl.kappa = kap;
            const auto f = kummer_exponents(ell, 2, kap);
            for (const auto& kv : f) {
                gl.sum_f += kv.second;
                gl.sumabs_f += kv.second < 0 ? -kv.second : kv.second;
            }
            gl.split_primes = split_primes(m, 3);
            gl.not_global_lth_power = true;
            gl.eigen_eps_kappa = true;
            for (uint32_t q : gl.split_primes) {
                const uint32_t rt = root_of_order(q, m);
                const uint32_t s = eval_product_unit(f, rt, q);
                if (powmod32(s, (q - 1) / ell, q) == 1) gl.not_global_lth_power = false;
                const uint32_t base = powmod32(s, (q - 1) / m, q);
                for (uint32_t b : {2u, 3u, ell + 1}) {
                    const uint32_t lhs = powmod32(eval_product_unit(f, powmod32(rt, b, q), q), (q - 1) / m, q);
                    if (lhs != powmod32(base, powmod32(b, kap, m), q)) gl.eigen_eps_kappa = false;
                }
            }
            res.global.push_back(gl);
        }
    }
    res.products = R.products();
    return res;
}

}  // namespace nf
