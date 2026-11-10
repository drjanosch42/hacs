// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 16, WP-5) The saturation-defect Tor term -- see
// HeckeSaturationDefect.h and docs/implementation/slot16/WP-5_tor_note.md.
#include "HeckeSaturationDefect.h"

#include "HeckeAlgebraInvariants.h"
#include "HeckeFormsHeckeAlgebra.h"
#include "../Hecke/CoefficientRing.h"     // hecke::IsPrimeU64
#include "../Hecke/HeckeOrdinaryPart.h"
#include "../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../Engine/EngineRuntime.h"   // (S24 SATPERF) engine::tape::ScopedHostElimHook (allow-listed, check_heckeext_includes.sh)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <utility>

namespace hecke_ext {
namespace satdefect {

using ringalg::OpModule;
using ringalg::AlgebraReading;
using hecke::ordinary::u128;
using Vec = std::vector<uint64_t>;

namespace {

using engine::scalar::powfit;

std::string JoinU(const std::vector<uint32_t>& v, const char* sep = ",") {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) s << (i ? sep : "") << v[i];
    return s.str();
}
std::string JoinU64(const std::vector<uint64_t>& v) {
    std::ostringstream s;
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str();
}

// ---- F_ell linear algebra (ell < 2^32: products fit uint64_t) ----------------
uint64_t InvMod(uint64_t a, uint64_t p) {
    ENGINE_CHECK_MSG(a % p != 0, "HeckeSaturationDefect: inverse of 0 mod ell");
    return engine::scalar::invmod_fermat<uint64_t>(a, p);
}

// An incrementally maintained reduced row-echelon basis of a subspace of F_p^w.
struct FSpan {
    uint64_t p = 0;
    uint32_t w = 0;
    std::vector<Vec> rows;
    std::vector<uint32_t> piv;
    FSpan(uint64_t p_, uint32_t w_) : p(p_), w(w_) {}
    Vec reduce(Vec v) const {
        for (size_t r = 0; r < rows.size(); ++r) {
            const uint64_t c = v[piv[r]];
            if (!c) continue;
            for (uint32_t j = 0; j < w; ++j) v[j] = (v[j] + (p - c) * rows[r][j]) % p;
        }
        return v;
    }
    bool contains(const Vec& v) const {
        const Vec r = reduce(v);
        for (uint64_t x : r) if (x) return false;
        return true;
    }
    bool add(const Vec& v0) {
        Vec v = reduce(v0);
        uint32_t lead = w;
        for (uint32_t j = 0; j < w; ++j) if (v[j]) { lead = j; break; }
        if (lead == w) return false;
        const uint64_t inv = InvMod(v[lead], p);
        for (uint64_t& x : v) x = x * inv % p;
        for (size_t r = 0; r < rows.size(); ++r) {
            const uint64_t c = rows[r][lead];
            if (!c) continue;
            for (uint32_t j = 0; j < w; ++j) rows[r][j] = (rows[r][j] + (p - c) * v[j]) % p;
        }
        // keep the rows ordered by pivot
        size_t at = 0;
        while (at < piv.size() && piv[at] < lead) ++at;
        rows.insert(rows.begin() + static_cast<long>(at), v);
        piv.insert(piv.begin() + static_cast<long>(at), lead);
        return true;
    }
    uint32_t dim() const { return static_cast<uint32_t>(rows.size()); }
};

// A basis of {x in F_p^t : R x = 0}, R given by its rows (each of length t).
std::vector<Vec> Nullspace(const std::vector<Vec>& R, uint32_t t, uint64_t p) {
    FSpan s(p, t);
    for (const Vec& r : R) s.add(r);
    std::vector<bool> is_piv(t, false);
    for (uint32_t c : s.piv) is_piv[c] = true;
    std::vector<Vec> out;
    for (uint32_t f = 0; f < t; ++f) {
        if (is_piv[f]) continue;
        Vec x(t, 0);
        x[f] = 1;
        for (size_t r = 0; r < s.rows.size(); ++r) x[s.piv[r]] = (p - s.rows[r][f]) % p;
        out.push_back(x);
    }
    return out;
}

// ---- End((Z/ell^K)^D): D x D row-major mod q ---------------------------------
struct EndK {
    uint64_t ell = 0, q = 0;
    uint32_t K = 0, D = 0;
    Vec mul(const Vec& A, const Vec& B) const {
        Vec C(static_cast<size_t>(D) * D, 0);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) {
                u128 acc = 0;
                for (uint32_t k = 0; k < D; ++k) {
                    acc += static_cast<u128>(A[static_cast<size_t>(i) * D + k]) * B[static_cast<size_t>(k) * D + j] % q;
                }
                C[static_cast<size_t>(i) * D + j] = static_cast<uint64_t>(acc % q);
            }
        return C;
    }
    Vec identity() const {
        Vec I(static_cast<size_t>(D) * D, 0);
        for (uint32_t i = 0; i < D; ++i) I[static_cast<size_t>(i) * D + i] = 1;
        return I;
    }
    Vec minus_scalar(Vec A, uint64_t c) const {
        for (uint32_t i = 0; i < D; ++i) {
            uint64_t& x = A[static_cast<size_t>(i) * D + i];
            x = static_cast<uint64_t>((static_cast<u128>(x) + q - c % q) % q);
        }
        return A;
    }
    // The subgroup generated by xs inside (Z/q)^{D^2}: its Howell rows and length.
    std::pair<std::vector<Vec>, uint32_t> span(const std::vector<Vec>& xs) const {
        const uint32_t cols = D * D;
        engine::DenseBuf<uint64_t> G(static_cast<uint32_t>(xs.size()), cols);
        for (uint32_t i = 0; i < xs.size(); ++i)
            for (uint32_t j = 0; j < cols; ++j) G.at(i, j) = xs[i][j];
        const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, K);
        const engine::elim::HowellResult<uint64_t> H = engine::elim::howell(ctx, engine::cview(G));
        std::vector<Vec> rows;
        for (uint32_t i = 0; i < H.rows(); ++i) {
            Vec r(cols);
            for (uint32_t j = 0; j < cols; ++j) r[j] = H.H.at(i, j);
            rows.push_back(r);
        }
        return {rows, engine::elim::length(ctx, H)};
    }
};

std::string Monomial(const std::vector<uint32_t>& alpha, const std::vector<uint64_t>& primes, const std::vector<uint64_t>& sys) {
    std::ostringstream s;
    bool any = false;
    for (size_t i = 0; i < alpha.size(); ++i) {
        if (!alpha[i]) continue;
        if (any) s << "*";
        any = true;
        if (sys[i] == 0) s << "T" << primes[i];
        else s << "(T" << primes[i] << "-" << sys[i] << ")";
        if (alpha[i] > 1) s << "^" << alpha[i];
    }
    if (!any) s << "1";
    return s.str();
}

// Every exponent vector of total degree d over v variables, lexicographically descending.
void Compositions(uint32_t v, uint32_t d, std::vector<uint32_t>& cur, std::vector<std::vector<uint32_t>>& out) {
    if (cur.size() + 1 == v) {
        cur.push_back(d);
        out.push_back(cur);
        cur.pop_back();
        return;
    }
    for (uint32_t a = d + 1; a-- > 0;) {
        cur.push_back(a);
        Compositions(v, d - a, cur, out);
        cur.pop_back();
    }
}

std::vector<uint32_t> StructureFromEpsImpl(const std::vector<uint32_t>& eps) {
    // eps[j-1] = length G[ell^j]; the number of summands of exponent >= j is eps_j - eps_{j-1}.
    std::vector<uint32_t> ge;   // ge[j-1] = #summands with exponent >= j
    uint32_t prev = 0;
    for (uint32_t e : eps) {
        ENGINE_CHECK(e >= prev);
        ge.push_back(e - prev);
        prev = e;
    }
    std::vector<uint32_t> out;
    for (size_t j = ge.size(); j-- > 0;) {
        const uint32_t next = (j + 1 < ge.size()) ? ge[j + 1] : 0;
        ENGINE_CHECK_MSG(ge[j] >= next, "HeckeSaturationDefect: the eps-filtration is not that of a finite Z_ell-module");
        for (uint32_t c = 0; c < ge[j] - next; ++c) out.push_back(static_cast<uint32_t>(j + 1));
    }
    return out;   // descending
}

} // namespace

// ---------------------------------------------------------------------------
DefectReading ReadDefect(const OpModule& B, const std::vector<uint64_t>& system) {
    ENGINE_CHECK_MSG(!B.is_digit(), "HeckeSaturationDefect: a digit module (ell^K >= 2^64) is refused");
    ENGINE_CHECK_MSG(B.rank() > 0, "HeckeSaturationDefect: an empty block");
    ENGINE_CHECK(system.size() == B.primes.size() && B.op.size() == B.primes.size());
    DefectReading R;
    const uint64_t ell = B.ell;
    ENGINE_CHECK_MSG(ell >= 5 && ell < (uint64_t(1) << 31), "HeckeSaturationDefect: 5 <= ell < 2^31");
    EndK E;
    E.ell = ell;
    E.D = B.rank();
    E.K = B.exps[0];
    for (uint32_t x : B.exps)
        ENGINE_CHECK_MSG(x == E.K, "HeckeSaturationDefect: the block is not free over Z/ell^K (a torsion block is refused)");
    ENGINE_CHECK_MSG(E.K >= 2, "HeckeSaturationDefect: K >= 2");
    E.q = powfit(ell, E.K);
    ENGINE_CHECK_MSG(E.q != 0, "HeckeSaturationDefect: ell^K >= 2^64");
    R.K = E.K;
    const uint32_t D = E.D, m = D * D;
    std::vector<Vec> T;
    for (const auto& op : B.op) {
        Vec a = op.a;
        for (uint64_t& x : a) x %= E.q;
        T.push_back(a);
    }
    const Vec I = E.identity();
    // A_K = Z[T_p] in End(B): the closure of {1} under the T_p.
    auto sp = E.span({I});
    std::vector<Vec> basis = sp.first;
    uint32_t L = sp.second;
    for (uint32_t it = 0;; ++it) {
        ENGINE_CHECK_MSG(it <= m * E.K + 2, "HeckeSaturationDefect: the algebra closure did not terminate");
        std::vector<Vec> cand = basis;
        for (const Vec& Tp : T)
            for (const Vec& b : basis) cand.push_back(E.mul(Tp, b));
        auto nx = E.span(cand);
        if (nx.second == L) break;
        basis = nx.first;
        L = nx.second;
    }
    R.length_A = L;
    // The cyclic decomposition of A_K (columns in (Z/q)^{D^2}).
    const engine::ModCtx<uint64_t> ctx = engine::ModCtx<uint64_t>::prime_power(ell, E.K);
    engine::DenseBuf<uint64_t> G(m, static_cast<uint32_t>(basis.size())), Rl(m, 0);
    for (uint32_t c = 0; c < basis.size(); ++c)
        for (uint32_t i = 0; i < m; ++i) G.at(i, c) = basis[c][i];
    const engine::elim::CyclicDecomposition<uint64_t> dec = engine::elim::cyclic_decomposition(ctx, engine::cview(G), engine::cview(Rl));
    const uint32_t t = dec.rows();
    R.dim_A = t;
    R.A_exponents = dec.exponents;
    {
        uint32_t s = 0;
        for (uint32_t e : dec.exponents) s += e;
        ENGINE_CHECK_MSG(s == L, "HeckeSaturationDefect: the cyclic summands of A_K do not sum to its length");
    }
    std::vector<Vec> b(t, Vec(m));
    for (uint32_t s = 0; s < t; ++s)
        for (uint32_t i = 0; i < m; ++i) b[s][i] = dec.basis.at(i, s);
    // coords mod ell of a batch of elements of A_K
    auto coords = [&](const std::vector<Vec>& ys) {
        engine::DenseBuf<uint64_t> Y(m, static_cast<uint32_t>(ys.size()));
        for (uint32_t c = 0; c < ys.size(); ++c)
            for (uint32_t i = 0; i < m; ++i) Y.at(i, c) = ys[c][i];
        const std::vector<std::vector<uint64_t>> co = engine::elim::cyclic_coordinates(ctx, dec, engine::cview(Y));
        std::vector<Vec> out;
        for (const auto& v : co) {
            Vec w(t);
            for (uint32_t s = 0; s < t; ++s) w[s] = v[s] % ell;
            out.push_back(w);
        }
        return out;
    };
    // The regular representation of A / ell A: Lb[i] columns = coords(b_i b_j).
    std::vector<std::vector<Vec>> Lb(t);
    for (uint32_t i = 0; i < t; ++i) {
        std::vector<Vec> prods;
        for (uint32_t j = 0; j < t; ++j) prods.push_back(E.mul(b[i], b[j]));
        Lb[i] = coords(prods);
    }
    auto mul = [&](const Vec& u, const Vec& v) {
        Vec w(t, 0);
        for (uint32_t i = 0; i < t; ++i) {
            if (!u[i]) continue;
            for (uint32_t j = 0; j < t; ++j) {
                if (!v[j]) continue;
                const uint64_t c = u[i] * v[j] % ell;
                for (uint32_t k = 0; k < t; ++k) w[k] = (w[k] + c * Lb[i][j][k]) % ell;
            }
        }
        return w;
    };
    std::vector<Vec> xs;
    for (size_t pi = 0; pi < T.size(); ++pi) xs.push_back(E.minus_scalar(T[pi], system[pi] % ell));
    std::vector<Vec> xb = coords(xs);
    const Vec unit = coords({I})[0];
    // commutativity, exactly in A / ell (a consistency check of the regular representation)
    for (uint32_t i = 0; i < t; ++i)
        for (uint32_t j = i + 1; j < t; ++j) {
            Vec ei(t, 0), ej(t, 0);
            ei[i] = 1;
            ej[j] = 1;
            ENGINE_CHECK_MSG(mul(ei, ej) == mul(ej, ei), "HeckeSaturationDefect: A / ell A is not commutative");
        }
    auto ideal_step = [&](const std::vector<Vec>& W) {   // span{x_p w}
        FSpan s(ell, t);
        for (const Vec& x : xb)
            for (const Vec& w : W) s.add(mul(x, w));
        return s;
    };
    std::vector<Vec> all;
    for (uint32_t j = 0; j < t; ++j) {
        Vec e(t, 0);
        e[j] = 1;
        all.push_back(e);
    }
    // the m-adic powers of A / ell A
    std::vector<FSpan> mpow;   // mpow[k] = m^{k+1}
    {
        std::vector<Vec> W = all;
        uint32_t prev = t;
        for (uint32_t it = 0;; ++it) {
            ENGINE_CHECK_MSG(it <= t + 1, "HeckeSaturationDefect: m is not nilpotent on A / ell A (not local at the system)");
            FSpan s = ideal_step(W);
            ENGINE_CHECK_MSG(s.dim() < prev, "HeckeSaturationDefect: A / ell A is not local at the system");
            R.hilbert.push_back(prev - s.dim());
            prev = s.dim();
            W = s.rows;
            mpow.push_back(s);
            if (s.dim() == 0) break;
        }
    }
    const FSpan& mm = mpow[0];
    // the reduction map A / ell A -> End(B / ell): column s = b_s mod ell
    std::vector<Vec> Prow(m, Vec(t));
    for (uint32_t s = 0; s < t; ++s)
        for (uint32_t i = 0; i < m; ++i) Prow[i][s] = b[s][i] % ell;
    const std::vector<Vec> Ibasis = Nullspace(Prow, t, ell);
    R.dim_I = static_cast<uint32_t>(Ibasis.size());
    R.dim_image = t - R.dim_I;
    FSpan Ispan(ell, t);
    for (const Vec& w : Ibasis) Ispan.add(w);
    // I is an ideal (x_p I in I)
    for (const Vec& x : xb)
        for (const Vec& w : Ibasis) ENGINE_CHECK_MSG(Ispan.contains(mul(x, w)), "HeckeSaturationDefect: the kernel is not an ideal");
    R.I_in_m = true;
    for (const Vec& w : Ibasis) R.I_in_m = R.I_in_m && mm.contains(w);
    R.I_is_m = R.I_in_m && R.dim_I == mm.dim();
    // (A / ell) / I at m: dims of m^k + I
    {
        uint32_t prev = t;
        for (size_t k = 0; k < mpow.size() && prev > R.dim_I; ++k) {
            FSpan s = Ispan;
            for (const Vec& r : mpow[k].rows) s.add(r);
            R.quotient_hilbert.push_back(prev - s.dim());
            prev = s.dim();
        }
        while (!R.quotient_hilbert.empty() && R.quotient_hilbert.back() == 0) R.quotient_hilbert.pop_back();
    }
    // the radical layers of I
    FSpan mI(ell, t);
    {
        std::vector<Vec> W = Ibasis;
        uint32_t prev = R.dim_I;
        bool first = true;
        for (uint32_t it = 0; prev > 0; ++it) {
            ENGINE_CHECK(it <= t + 1);
            FSpan s = ideal_step(W);
            R.I_layers.push_back(prev - s.dim());
            if (first) { mI = s; first = false; }
            prev = s.dim();
            W = s.rows;
        }
        R.I_loewy = static_cast<uint32_t>(R.I_layers.size());
        R.min_generators = R.I_layers.empty() ? 0 : R.I_layers[0];
        R.principal = R.min_generators <= 1;
    }
    // soc(I): w = sum c_k w_k with x_p w = 0 for all p
    if (R.dim_I) {
        std::vector<Vec> rows;
        for (const Vec& x : xb) {
            std::vector<Vec> cols;
            for (const Vec& w : Ibasis) cols.push_back(mul(x, w));
            for (uint32_t i = 0; i < t; ++i) {
                Vec r(R.dim_I);
                for (uint32_t k = 0; k < R.dim_I; ++k) r[k] = cols[k][i];
                rows.push_back(r);
            }
        }
        R.I_socle = static_cast<uint32_t>(Nullspace(rows, R.dim_I, ell).size());
    }
    // Ann(I): u with u w_k = 0 for all k
    {
        std::vector<Vec> rows;
        for (const Vec& w : Ibasis) {
            std::vector<Vec> cols;
            for (const Vec& e : all) cols.push_back(mul(w, e));
            for (uint32_t i = 0; i < t; ++i) {
                Vec r(t);
                for (uint32_t j = 0; j < t; ++j) r[j] = cols[j][i];
                rows.push_back(r);
            }
        }
        R.I_ann = static_cast<uint32_t>(Nullspace(rows, t, ell).size());
    }
    // A monomial basis of A / ell in degree order, and the generators as polynomials.
    const uint32_t v = static_cast<uint32_t>(xb.size());
    std::vector<std::vector<uint32_t>> mono;
    std::vector<Vec> mclass;
    {
        std::map<std::vector<uint32_t>, Vec> cls;
        cls[std::vector<uint32_t>(v, 0)] = unit;
        FSpan chosen(ell, t);
        const uint32_t maxdeg = static_cast<uint32_t>(R.hilbert.size());
        for (uint32_t d = 0; d < maxdeg && chosen.dim() < t; ++d) {
            std::vector<std::vector<uint32_t>> deg;
            std::vector<uint32_t> cur;
            Compositions(v, d, cur, deg);
            for (const auto& a : deg) {
                if (!cls.count(a)) {
                    size_t p0 = 0;
                    while (a[p0] == 0) ++p0;
                    std::vector<uint32_t> prev = a;
                    --prev[p0];
                    cls[a] = mul(xb[p0], cls.at(prev));
                }
                if (chosen.add(cls.at(a))) {
                    mono.push_back(a);
                    mclass.push_back(cls.at(a));
                }
            }
        }
        ENGINE_CHECK_MSG(chosen.dim() == t, "HeckeSaturationDefect: the monomials in T_p - a_p do not span A / ell A");
    }
    for (const auto& a : mono) R.monomial_basis.push_back(Monomial(a, B.primes, system));
    if (R.dim_I) {
        // monomial coordinates: solve sum c_i mclass_i = w via the augmented span
        auto mono_coords = [&](const Vec& w) {
            // Gaussian elimination on [mclass | w] (t x (t+1))
            std::vector<Vec> M(t, Vec(t + 1));
            for (uint32_t i = 0; i < t; ++i) {
                for (uint32_t j = 0; j < t; ++j) M[i][j] = mclass[j][i];
                M[i][t] = w[i];
            }
            for (uint32_t c = 0, r = 0; c < t; ++c) {
                uint32_t pr = r;
                while (pr < t && !M[pr][c]) ++pr;
                ENGINE_CHECK(pr < t);
                std::swap(M[pr], M[r]);
                const uint64_t inv = InvMod(M[r][c], ell);
                for (uint64_t& x : M[r]) x = x * inv % ell;
                for (uint32_t i = 0; i < t; ++i) {
                    if (i == r || !M[i][c]) continue;
                    const uint64_t f = M[i][c];
                    for (uint32_t j = 0; j <= t; ++j) M[i][j] = (M[i][j] + (ell - f) * M[r][j]) % ell;
                }
                ++r;
            }
            Vec c(t);
            for (uint32_t i = 0; i < t; ++i) c[i] = M[i][t];
            return c;
        };
        // RREF of I in monomial coordinates (pivots on the lowest-degree monomials first)
        FSpan Imono(ell, t);
        for (const Vec& w : Ibasis) Imono.add(mono_coords(w));
        FSpan gen = mI;
        for (const Vec& c : Imono.rows) {
            Vec w(t, 0);
            for (uint32_t i = 0; i < t; ++i)
                for (uint32_t k = 0; k < t; ++k) w[k] = (w[k] + c[i] * mclass[i][k]) % ell;
            if (!gen.add(w)) continue;
            std::ostringstream s;
            bool any = false;
            for (uint32_t i = 0; i < t; ++i) {
                if (!c[i]) continue;
                if (any) s << "+";
                any = true;
                if (c[i] != 1) s << c[i] << "*";
                s << R.monomial_basis[i];
            }
            R.generators.push_back(s.str());
        }
        ENGINE_CHECK_MSG(R.generators.size() == R.min_generators, "HeckeSaturationDefect: generator count != dim I / m I");
    }
    return R;
}

// ---------------------------------------------------------------------------
namespace {

struct FormsCtx {
    bool built = false;
    std::vector<uint64_t> P;
    forms::CuspSpace M;
    std::vector<ExtMatrix> TM;
    bool M_gen = false;
};

std::vector<uint64_t> PrimesUpTo(uint32_t b) {
    std::vector<uint64_t> out;
    for (uint32_t p = 2; p <= b; ++p) {
        bool pr = true;
        for (uint32_t q = 2; q * q <= p; ++q)
            if (p % q == 0) pr = false;
        if (pr) out.push_back(p);
    }
    return out;
}

void BuildForms(FormsCtx& F, uint32_t n, uint64_t ell) {
    if (F.built) return;
    const uint32_t k = n + 2;
    // The generators of HeckeFormsHeckeAlgebraValidation arm (T): every prime <= max(31, k/12), and ell.
    F.P = PrimesUpTo(std::max(31u, k / 12));
    if (std::find(F.P.begin(), F.P.end(), ell) == F.P.end()) F.P.push_back(ell);
    const uint32_t pmax = static_cast<uint32_t>(F.P.back());
    const uint32_t N = 2 * (k / 12 + 1) * pmax + 2;
    F.M = forms::BuildModularSpace(k, static_cast<uint32_t>(ell), N);
    for (uint64_t p : F.P) F.TM.push_back(forms::HeckeOnCusp(F.M, static_cast<uint32_t>(p)));
    F.M_gen = ComputeAlgebraInvariants(F.TM, static_cast<uint32_t>(ell)).dim_A == F.M.basis.size();
    F.built = true;
}

ExtMatrix BlockDiag(const ExtMatrix& a, const ExtMatrix& b) {
    ExtMatrix c(a.rows + b.rows, a.cols + b.cols, a.p);
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j) c.data[i][j] = a.data[i][j];
    for (uint32_t i = 0; i < b.rows; ++i)
        for (uint32_t j = 0; j < b.cols; ++j) c.data[a.rows + i][a.cols + j] = b.data[i][j];
    return c;
}

FormsCross RunForms(FormsCtx& F, uint32_t n, uint64_t ell, const std::vector<uint64_t>& S, const std::vector<uint64_t>& sys,
                    const OpModule& Bred, uint32_t dim_A1) {
    FormsCross o;
    BuildForms(F, n, ell);
    o.M_gen = F.M_gen;
    std::vector<uint32_t> sys32;
    for (uint64_t a : sys) sys32.push_back(static_cast<uint32_t>(a % ell));
    const forms::FormsBlock B = forms::LocaliseAtSystem(F.M, F.P, F.TM, S, sys32);
    o.ran = true;
    o.dim_V = B.dim;
    if (B.dim == 0) return o;
    o.T_dim = B.A.dim_A;
    const uint32_t D = Bred.rank();
    std::vector<ExtMatrix> tT, joint;
    for (size_t j = 0; j < S.size(); ++j) {
        const size_t idx = static_cast<size_t>(std::find(F.P.begin(), F.P.end(), S[j]) - F.P.begin());
        ENGINE_CHECK(idx < F.P.size());
        ExtMatrix c(D, D, static_cast<uint32_t>(ell));
        for (uint32_t r = 0; r < D; ++r)
            for (uint32_t s = 0; s < D; ++s) c.data[r][s] = static_cast<uint32_t>(Bred.op[j].a[static_cast<size_t>(r) * D + s] % ell);
        tT.push_back(B.t[idx]);
        joint.push_back(BlockDiag(B.t[idx], c));
    }
    o.Tc = ComputeAlgebraInvariants(tT, static_cast<uint32_t>(ell)).dim_A;
    o.joint = ComputeAlgebraInvariants(joint, static_cast<uint32_t>(ell)).dim_A;
    o.S_gen = o.Tc == o.T_dim;
    // (slot-16 WP-5 audit repair, finding 4) J -> A_1 is onto (A_1 is the algebra of the same operators on
    // B / ell, a quotient of the joint algebra), so joint >= dim A_1; a violation is a defect, not a 0 kernel.
    ENGINE_CHECK_MSG(o.joint >= dim_A1, "HeckeSaturationDefect: the joint algebra is smaller than A_1 (J -> A_1 must be onto)");
    o.kernel = o.joint - dim_A1;
    return o;
}

} // namespace

SatCell ComputeSaturationDefectCell(uint32_t n, uint64_t ell, const SatOptions& opts) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(ell >= 5 && n >= 2 && n % 2 == 0, "HeckeSaturationDefect: n >= 2 even, ell >= 5");
    // (slot-16 WP-5 audit repair, finding 3) a composite ell used to run into confusing module-route failures
    ENGINE_CHECK_MSG(hecke::IsPrimeU64(ell), ("HeckeSaturationDefect: ell = " + std::to_string(ell) + " is not prime").c_str());
    ENGINE_CHECK_MSG(opts.kmax >= 2, "HeckeSaturationDefect: KMAX >= 2");
    // (slot-24 ELL-W audit F6) the forms cross-route (run per block at k = n + 2 >= 12, AFTER the module run) needs
    // ell < 2^16 (forms::BuildCuspSpace): refuse here, before the module run, instead of after it (measured: 75 min lost at
    // (300, 100003)).  Value-neutral: such a cell could never complete; opts.forms = false still runs it without the forms side.
    // (slot-24 FNTT) the range is now forms::FormsSpaceSupported at the route's precision: any ell < 2^16, and above it the
    // FNTT route's exactness bounds (e.g. ell = 100003 at n <= 464); still refused before the module run otherwise.
    // (slot-24 FNTT3) ... or the three-prime route's (Fntt3Applicable; e.g. (1000, 100003)) unless HECKE_FORMS_NTT3=0.
    if (opts.forms && n + 2 >= 12 &&
        (ell >= forms::kFormsMaxEllExclusive ||
         !forms::FormsSpaceSupported(static_cast<uint32_t>(ell), forms::SatFormsPrecision(n + 2, static_cast<uint32_t>(ell)))))
        throw hecke::NotImplementedInThisBuild("HeckeSaturationDefect: ell = " + std::to_string(ell) + " at n = " + std::to_string(n) +
                                               ": the forms cross-route cannot build M_k (forms::FormsSpaceSupported: ell >= 2^16 "
                                               "needs the FNTT route within 2N - 1 <= 2^25 and N (ell-1)^2 < P1 P2, or the three-prime "
                                               "route, HECKE_FORMS_NTT3 not 0, within nb B (ell-1)^2 < P1 P2 P3); refused before "
                                               "the module run");
    SatCell C;
    C.n = n;
    C.ell = ell;
    C.kmax = opts.kmax;
    std::vector<uint64_t> primes = opts.primes.empty() ? std::vector<uint64_t>{2, 3, 5, 7, ell} : opts.primes;
    std::sort(primes.begin(), primes.end());
    primes.erase(std::unique(primes.begin(), primes.end()), primes.end());
    C.primes = primes;
    // (S24 SATPERF) the stage clocks (SatCell::stage_seconds; printed by the validation, never part of a record)
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto since = [](std::chrono::steady_clock::time_point a) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    };
    const bool perf = ringalg::SatPerfEnabled();
    if (std::getenv("HECKE_SAT_PERF") != nullptr) {
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) std::cout << ringalg::SatPerfEffectiveLine(false) << std::endl;
    }
    auto tm = now();
    // [C, prop:torsionprepass]; (S24 SATPERF) the one run of the cell is handed to every certificate below (the slot-18
    // WP-5 (e) pattern of ComputeRingLoewyCell); the rollback re-runs it per certificate, as before
    const hecke::ordinary::TorsionPrepass pp = hecke::ordinary::TorsionExponentPrepass(n, ell);
    C.e_max = pp.e_max;
    C.stage_seconds["prepass"] += since(tm);
    C.K_cls = std::max<uint32_t>(2, C.e_max + 1);
    const uint32_t kneed = std::max(C.kmax, C.K_cls);
    if (powfit(ell, kneed) == 0)
        throw hecke::NotImplementedInThisBuild("HeckeSaturationDefect: ell^K >= 2^64 at (n, ell, K) = (" + std::to_string(n) + ", " +
                                               std::to_string(ell) + ", " + std::to_string(kneed) +
                                               "): this package runs the native uint64_t module route only");
    // (slot-23 G1) the level cache: one module run per cell, the levels below it derived ([C, lem:precisionreduction],
    // [C, obl:precisionreuse]); HECKE_PRECISION_REUSE=0 is the per-K route (one run per level read), the oracle.
    ringalg::LevelCache levels(n, ell, primes, ringalg::ModuleRoute::Native, C.K_cls, kneed);
    auto get = [&](uint32_t K) -> const OpModule& { return levels.get(K); };
    // (S24 SATPERF) the per-level memos (HeckeRingHeckeAlgebra.h, ModuleDerived): get(K) -- which may RUN a module -- is
    // called before the memo exists, outside the post-module hook below, so the module run sees exactly the hooks it saw
    std::map<uint32_t, std::unique_ptr<ringalg::ModuleDerived>> memo;
    auto D = [&](uint32_t K) -> ringalg::ModuleDerived& {
        const OpModule& M = get(K);
        std::unique_ptr<ringalg::ModuleDerived>& p = memo[K];
        if (!p) p.reset(new ringalg::ModuleDerived(M));
        return *p;
    };
    // the post-module stages' End(N) products and eliminations split over the engine pool (bit-identical by construction:
    // ENGINE_API Sec. 3 / D4, and EndCtx::mul_fast's disjoint rows); a no-op at one thread or with ENGINE_HOST_ELIM_HOOK=0
    auto post_hook = [&]() -> std::unique_ptr<engine::tape::ScopedHostElimHook> {
        return perf ? std::unique_ptr<engine::tape::ScopedHostElimHook>(new engine::tape::ScopedHostElimHook()) : nullptr;
    };
    auto localize = [&](uint32_t K, const std::vector<uint64_t>& a) -> ringalg::Block {
        if (!perf) return ringalg::Localize(get(K), a);
        ringalg::ModuleDerived& d = D(K);
        const auto h = post_hook();
        return d.block(a);
    };
    FormsCtx F;
    const OpModule& N2 = get(2);
    std::vector<ringalg::JointSystem> systems;
    tm = now();
    if (perf) {
        ringalg::ModuleDerived& d2 = D(2);
        const auto h = post_hook();
        systems = d2.systems();
    } else {
        systems = ringalg::RationalSystems(N2);
    }
    C.stage_seconds["systems"] += since(tm);
    for (const ringalg::JointSystem& js : systems) {
        SatBlock b;
        b.system = js.a;
        tm = now();
        const ringalg::Block Bc = localize(C.K_cls, js.a);
        C.stage_seconds["localize"] += since(tm);
        b.D = Bc.local.rank();
        ENGINE_CHECK_MSG(b.D == js.dim, "HeckeSaturationDefect: the block at K_cls has dim_F H^1(F_ell)_m generators");
        for (uint32_t x : Bc.local.exps) {
            if (x == C.K_cls) ++b.free_rank;
            else b.torsion.push_back(x);
        }
        std::sort(b.torsion.rbegin(), b.torsion.rend());
        b.rank_T = (b.free_rank + 1) / 2;
        const OpModule Bred = ringalg::ReduceModEll(Bc.local);
        const AlgebraReading A1 = ringalg::ReadAlgebra(Bred, js.a);
        b.A1_hilbert = A1.hilbert;
        b.dim_A1 = A1.dim_mod_ell;
        if (!b.torsion.empty()) {
            b.status = "refused:torsion_block";
            b.verdict = "torsion_block";
            C.blocks.push_back(b);
            continue;
        }
        ringalg::CheckWithinRankT(A1, b.rank_T);
        b.eps1_len = b.rank_T - b.dim_A1;
        b.eps.push_back(b.eps1_len);
        ringalg::LoewyBlock claim;   // the route's claim, for ringalg::CertifyStable
        claim.system = js.a;
        claim.D = b.D;
        claim.free_rank = b.free_rank;
        claim.rank_T = b.rank_T;
        claim.A1 = A1;
        bool stable = false, final = false;
        if (b.dim_A1 == 1 && b.rank_T == 1) {
            claim.verdict = "reduced_rank1";
            stable = final = true;
            b.at = ReadDefect(Bc.local, js.a);
            b.K_stab = 0;
        } else {
            for (uint32_t K = 2; K <= kneed && !final; ++K) {
                tm = now();
                const ringalg::Block Bk = (K == C.K_cls) ? Bc : localize(K, js.a);
                C.stage_seconds["localize"] += since(tm);
                ENGINE_CHECK(Bk.local.rank() == b.D);
                for (uint32_t x : Bk.local.exps) ENGINE_CHECK_MSG(x == K, "HeckeSaturationDefect: a free block has a torsion summand");
                const AlgebraReading RK = ringalg::ReadAlgebra(Bk.local, js.a);
                ringalg::CheckWithinRankT(RK, b.rank_T);
                b.ak_len.push_back(RK.length);
                const int64_t epsK = static_cast<int64_t>(K) * b.rank_T - RK.length;
                ENGINE_CHECK_MSG(epsK >= static_cast<int64_t>(b.eps.back()), "HeckeSaturationDefect: eps^sat decreased ([C, prop:satdefect](2))");
                b.eps.push_back(static_cast<uint32_t>(epsK));
                if (!stable) {
                    ringalg::LoewyStep st;
                    st.K = K;
                    st.full = RK;
                    claim.history.push_back(st);
                    if (RK.dim_mod_ell == b.rank_T) {
                        stable = true;
                        b.K_stab = K;
                        claim.K_stab = K;
                        claim.verdict = "stable";
                        claim.eps = static_cast<int64_t>(K) * RK.dim_mod_ell - RK.length;
                        b.at = ReadDefect(Bk.local, js.a);
                    }
                }
                if (stable && b.eps[b.eps.size() - 1] == b.eps[b.eps.size() - 2]) final = true;
            }
        }
        if (!stable) {
            b.status = "refused:not_stable_at_KMAX";
            b.verdict = "not_stable_at_KMAX";
            b.sat_length = b.eps.back();
            C.blocks.push_back(b);
            continue;
        }
        b.status = "ok";
        b.verdict = claim.verdict;
        if (opts.certify) {
            // [C, obl:precisionreuse](vii): unchanged on the modules read; the run level each came from is recorded beside it
            const uint32_t KK = claim.verdict == "reduced_rank1" ? C.K_cls : b.K_stab;
            tm = now();
            if (perf) {
                ringalg::ModuleDerived& dc = D(C.K_cls);
                ringalg::ModuleDerived& dk = D(KK);
                const auto h = post_hook();
                (void)ringalg::CertifyStable(n, dc, dk, claim, C.K_cls, &pp);
            } else {
                (void)ringalg::CertifyStable(n, get(C.K_cls), get(KK), claim, C.K_cls);
            }
            C.stage_seconds["cert"] += since(tm);
            b.cert_from_cls = levels.source(C.K_cls);
            b.cert_from_K = levels.source(KK);
            b.stable_cert = "certified";
        } else {
            b.stable_cert = "reading";
        }
        b.sat_final = final;
        b.sat_length = b.eps.back();
        if (final) {
            std::vector<uint32_t> e = b.eps;
            e.pop_back();   // the plateau's repeated value
            b.sat_structure = StructureFromEpsImpl(e);
        }
        // the forms side is built for k = n + 2 >= 12 (HeckeFormsHeckeAlgebra's range); below it, not run
        tm = now();
        if (opts.forms && n + 2 >= 12) b.forms = RunForms(F, n, ell, primes, js.a, Bred, b.dim_A1);
        C.stage_seconds["forms"] += since(tm);
        C.blocks.push_back(b);
    }
    C.K_run = levels.levels_read();   // (vi) Lv_read: the per-K route's list, byte-stable
    C.Lv_run = levels.levels_run();
    C.level_mode = levels.mode();
    C.module_seconds = levels.module_seconds();
    C.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return C;
}

std::vector<uint32_t> SaturationStructure(const std::vector<uint32_t>& eps) { return StructureFromEpsImpl(eps); }

CensusColumns Columns(const SatBlock& b) {
    CensusColumns c;
    if (b.status != "ok") {
        const std::string why = b.status.substr(b.status.find(':') + 1);
        c.sat_defect_dim = c.sat_defect_loewy = c.sat_length = "not_computed:" + why;
        return c;
    }
    c.sat_defect_dim = std::to_string(b.at.dim_I);
    c.sat_defect_loewy = std::to_string(b.at.I_loewy);
    c.sat_length = (b.sat_final ? "" : ">=") + std::to_string(b.sat_length);
    return c;
}

std::string SatDefectLine(const SatCell& c, const SatBlock& b) {
    std::ostringstream os;
    os << "SAT_DEFECT n=" << c.n << " ell=" << c.ell << " primes={" << JoinU64(c.primes) << "} system=(" << JoinU64(b.system)
       << ") D=" << b.D << " free_rank=" << b.free_rank << " torsion=[" << JoinU(b.torsion) << "] rank_T=" << b.rank_T
       << " A1_hilbert=[" << JoinU(b.A1_hilbert) << "] status=" << b.status << " verdict=" << b.verdict;
    if (b.status == "ok") {
        const DefectReading& r = b.at;
        os << " K_stab=" << b.K_stab << " stable_cert=" << b.stable_cert << " T_hilbert=[" << JoinU(r.hilbert) << "] dim_I=" << r.dim_I
           << " I_layers=[" << JoinU(r.I_layers) << "] I_loewy=" << r.I_loewy << " I_socle=" << r.I_socle << " I_ann=" << r.I_ann
           << " principal=" << (r.dim_I == 0 ? "zero" : (r.principal ? "yes" : "no")) << " I_is_m=" << (r.I_is_m ? 1 : 0)
           << " generators=[";
        for (size_t i = 0; i < r.generators.size(); ++i) os << (i ? ";" : "") << r.generators[i];
        os << "] eps1_len=" << b.eps1_len << " tor_routes=" << (r.dim_I == b.eps1_len ? "agree" : "DISAGREE") << " eps=["
           << JoinU(b.eps) << "] sat_length=" << (b.sat_final ? "" : ">=") << b.sat_length << " sat_structure=["
           << JoinU(b.sat_structure) << "]";
        if (b.forms.ran)
            os << " forms=(dim_V=" << b.forms.dim_V << ",T=" << b.forms.T_dim << ",Tc=" << b.forms.Tc << ",joint=" << b.forms.joint
               << ",S_gen=" << (b.forms.S_gen ? 1 : 0) << ",M_gen=" << (b.forms.M_gen ? 1 : 0) << ",kernel=" << b.forms.kernel << ")";
    } else {
        os << " eps=[" << JoinU(b.eps) << "]";
    }
    const CensusColumns cc = Columns(b);
    os << " sat_defect_dim=" << cc.sat_defect_dim << " sat_defect_loewy=" << cc.sat_defect_loewy << " sat_length_col=" << cc.sat_length;
    return os.str();
}

} // namespace satdefect
} // namespace hecke_ext
