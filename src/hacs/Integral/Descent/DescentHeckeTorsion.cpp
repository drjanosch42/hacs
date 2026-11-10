// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D3, parts (D)-(F): the torsion read of [C] prop:torsionhecke(3) with its choice-free invariants (1), the local
// pieces (the tree at 2 and 3, the cokernel presentation at ell >= 5; the free model and Iso^(ell) as oracles), the free quotient
// (4), and the record of thm:descentcert(2), (3).  See DescentHecke.h.
#include "DescentHecke.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <random>
#include <sstream>

#include "DescentLevelN.h"
#include "../Model/IsotropyModelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeLevelNHeilbronn.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../Util/CancelPoint.h"   // (NBCONFIRM S5) cooperative cancellation points

namespace integral_descent {
namespace hecke3 {

namespace {
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
uint32_t PowU(uint32_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) x *= ell;
    ENGINE_CHECK_MSG(x < (uint64_t(1) << 32), "PowU: overflow");
    return static_cast<uint32_t>(x);
}
// the largest K with ell^K < 2^31
uint32_t MaxK(uint32_t ell) {
    uint32_t K = 0;
    uint64_t x = 1;
    while (x * ell < (uint64_t(1) << 31)) {
        x *= ell;
        ++K;
    }
    return K;
}
std::string FmtExps(std::vector<uint32_t> e) {
    if (e.empty()) return "-";
    std::sort(e.begin(), e.end());
    std::ostringstream o;
    for (size_t i = 0; i < e.size(); ++i) o << (i ? "." : "") << e[i];
    return o.str();
}
std::vector<uint32_t> TorsExps(const AbelianGroupType& G, uint32_t ell) {
    std::vector<uint32_t> out;
    const auto it = G.torsion.find(ell);
    if (it != G.torsion.end()) out = it->second;
    std::sort(out.begin(), out.end());
    return out;
}
// an operator applied in column chunks of at most `maxk`
BatchOp Chunked(BatchOp op, uint32_t rows, uint32_t maxk) {
    return [op, rows, maxk](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
        if (k <= maxk) return op(X, k, Y);
        Y.assign(size_t(rows) * k, 0);
        std::vector<uint32_t> Xc, Yc;
        for (uint32_t c0 = 0; c0 < k; c0 += maxk) {
            const uint32_t kc = std::min(maxk, k - c0);
            Xc.assign(size_t(rows) * kc, 0);
            for (uint32_t r = 0; r < rows; ++r)
                std::copy(X.begin() + size_t(r) * k + c0, X.begin() + size_t(r) * k + c0 + kc, Xc.begin() + size_t(r) * kc);
            op(Xc, kc, Yc);
            for (uint32_t r = 0; r < rows; ++r)
                std::copy(Yc.begin() + size_t(r) * kc, Yc.begin() + size_t(r + 1) * kc, Y.begin() + size_t(r) * k + c0);
        }
    };
}
Mat ToMat(const std::vector<uint32_t>& v, uint32_t r, uint32_t c) {
    Mat M(r, c);
    M.a = v;
    return M;
}
}  // namespace

// =================================================================================================================================
// (D) the read
// =================================================================================================================================
// A Smith form with LEFT transforms by valuation levels (the ring2k / ring3k scheme with L tracked): at level s every column of the
// active block is scanned once for an entry of valuation exactly s (all active entries are divisible by ell^s); such an entry is a
// pivot u ell^s: rows below are cleared by row operations (applied to L and, inversely, to L^{-1}T), the rest of the pivot row by the
// (untracked) column operations of R -- which change no other row, the pivot column being zero below; when no entry of valuation s
// is left the level is raised.  The same output contract as maps::SmithForm(Z, A, true, false): L A R = Sigma with pivots u_t ell^{sigma_t}
// (sigma ascending), every sigma_t < k; only the pivot search differs (O(rows) per column, not O(rows x cols) per pivot).
maps::Smith SmithLevelsLeft(const ZMod& Z, Mat A) {
    maps::Smith S;
    const uint32_t r = A.r, c = A.c;
    S.L = Mat(r, r);
    S.LinvT = Mat(r, r);
    for (uint32_t i = 0; i < r; ++i) S.L.at(i, i) = S.LinvT.at(i, i) = 1;
    for (auto& x : A.a) x = Z.red(x);
    auto rowsub = [&](uint32_t* a, const uint32_t* b, uint32_t f, uint32_t n) {   // a -= f b
        if (Z.two) {
            for (uint32_t j = 0; j < n; ++j) a[j] = (a[j] - f * b[j]) & Z.mask;
        } else {
            for (uint32_t j = 0; j < n; ++j)
                if (b[j]) a[j] = Z.sub(a[j], static_cast<uint32_t>(uint64_t(f) * b[j] % Z.m));
        }
    };
    std::vector<uint8_t> done(c, 0);
    uint32_t t = 0;
    for (uint32_t s = 0; s < Z.k && t < r; ++s) {
        const uint32_t ps = Z.pw(s);   // ell^s (ell^0 = 1)
        for (uint32_t j = 0; j < c && t < r; ++j) {
            if (done[j]) continue;
            uint32_t pi = r;
            for (uint32_t i = t; i < r; ++i) {
                const uint32_t x = A.at(i, j);
                if (x && Z.val(x) == s) {
                    pi = i;
                    break;
                }
            }
            if (pi == r) continue;
            if (pi != t) {
                std::swap_ranges(A.row(t), A.row(t) + c, A.row(pi));
                std::swap_ranges(S.L.row(t), S.L.row(t) + r, S.L.row(pi));
                std::swap_ranges(S.LinvT.row(t), S.LinvT.row(t) + r, S.LinvT.row(pi));
            }
            const uint32_t uinv = Z.inv(s ? A.at(t, j) / ps : A.at(t, j));
            for (uint32_t i = t + 1; i < r; ++i) {
                const uint32_t x = A.at(i, j);
                if (!x) continue;
                const uint32_t f = Z.mul(s ? x / ps : x, uinv);   // x is divisible by ell^s (the level invariant)
                rowsub(A.row(i), A.row(t), f, c);
                rowsub(S.L.row(i), S.L.row(t), f, r);
                rowsub(S.LinvT.row(t), S.LinvT.row(i), static_cast<uint32_t>((Z.m - f) % Z.m), r);   // L^{-1}: col t += f col i
                ENGINE_DEBUG_CHECK(A.at(i, j) == 0);
            }
            // the column operations of R: clear the pivot row outside the pivot column (no other row changes)
            const uint32_t pv = A.at(t, j);
            uint32_t* prow = A.row(t);
            std::fill(prow, prow + c, 0u);
            prow[j] = pv;
            done[j] = 1;
            S.sigma.push_back(s);
            ++t;
        }
    }
    return S;
}

TorsionRead ReadTorsion(const ZMod& Z0, const Mat& d, uint32_t rank_Q, const Mat* dnext, const std::vector<BatchOp>& ops, bool with_free) {
    TorsionRead R;
    R.ell = Z0.ell;
    R.K = Z0.k;
    R.c = d.r;
    R.rank_Q = rank_Q;
    const auto t0 = std::chrono::steady_clock::now();
    maps::Smith S = SmithLevelsLeft(Z0, d);
    R.smith_seconds = since(t0);
    R.s = S.rank();
    if (R.s != rank_Q) {
        R.code = D3Refusal::GroupUncertified;
        R.why = "REFUSED [GroupUncertified] d^{q-1} mod " + std::to_string(Z0.ell) + "^" + std::to_string(Z0.k) + ": r_K = " + std::to_string(R.s) +
                " != rank_Q = " + std::to_string(rank_Q);
        return R;
    }
    std::vector<uint32_t> J;
    for (uint32_t t = 0; t < R.s; ++t)
        if (S.sigma[t] >= 1) {
            J.push_back(t);
            R.sigma.push_back(S.sigma[t]);
            R.smax = std::max(R.smax, S.sigma[t]);
        }
    ZMod Z = Z0;
    if (ActiveD3Fault() == D3Fault::ReadBelowModule && R.smax >= 2) {
        // VALIDATION ONLY: the same certified form read at K = 2 sigma_max - 1 (still > sigma_max: certified, but below the module)
        Z = ZMod(Z0.ell, 2 * R.smax - 1);
        for (auto& x : S.L.a) x = Z.red(x);
        for (auto& x : S.LinvT.a) x = Z.red(x);
        R.K = Z.k;
    }
    R.Kp = R.K - R.smax;
    R.module_certified = 2 * R.smax <= R.K;
    const uint32_t c = d.r, nj = static_cast<uint32_t>(J.size());
    const uint32_t nfree = with_free ? c - R.s : 0u;
    const uint32_t k = nj + nfree;
    R.ok = true;
    if (!R.module_certified) {
        R.code = D3Refusal::ModulePrecision;
        R.why = "REFUSED [ModulePrecision] 2 sigma_max = " + std::to_string(2 * R.smax) + " > K = " + std::to_string(R.K) +
                " (prop:torsionhecke(3)): the module and (1)(c), (d) are not certified; P_j and the Fitting types are";
    }
    if (k == 0 || ops.empty()) return R;
    // the generators: columns J (and s .. c-1) of L^{-1}
    std::vector<uint32_t> X(size_t(c) * k, 0);
    for (uint32_t t = 0; t < k; ++t) {
        const uint32_t col = t < nj ? J[t] : R.s + (t - nj);
        for (uint32_t r = 0; r < c; ++r) X[size_t(r) * k + t] = S.LinvT.at(col, r);
    }
    const ZMod ZKp(Z.ell, R.Kp);
    // the sparse rows of the next differential (the cocycle check)
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> nx;
    if (dnext) {
        ENGINE_CHECK_MSG(dnext->c == c, "ReadTorsion: the next differential must start at C^q");
        nx.resize(dnext->r);
        for (uint32_t i = 0; i < dnext->r; ++i)
            for (uint32_t j = 0; j < c; ++j)
                if (const uint32_t v = Z.red(dnext->at(i, j))) nx[i].push_back({j, v});
    }
    // the free-coordinate test: exact (every row >= s of L) when cheap, else 48 random combinations of those rows
    const bool exact_free = uint64_t(c) * (c - R.s) * k <= 40000000000ull;
    std::vector<std::vector<uint32_t>> combos;
    if (!exact_free && R.s < c) {
        std::mt19937_64 rng(0x5eed0d3ull + c);
        for (int t = 0; t < 48; ++t) {
            std::vector<uint32_t> w(c, 0);
            for (uint32_t i = R.s; i < c; ++i) {
                const uint32_t f = static_cast<uint32_t>(rng() % Z.m);
                if (!f) continue;
                const uint32_t* Li = S.L.row(i);
                for (uint32_t j = 0; j < c; ++j)
                    if (Li[j]) w[j] = Z.add(w[j], Z.mul(f, Li[j]));
            }
            combos.push_back(std::move(w));
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (size_t o = 0; o < ops.size(); ++o) {
        std::vector<uint32_t> Y;
        ops[o](X, k, Y);
        ENGINE_CHECK_MSG(Y.size() == size_t(c) * k, "ReadTorsion: the operator returned the wrong shape");
        for (auto& y : Y) y = Z.red(y);
        // coordinates (L Y)_i, i in J (and the free rows)
        Mat Phi(nj, nj), Fr(nfree, nfree);
        auto coord = [&](uint32_t i, std::vector<uint64_t>& acc) {
            acc.assign(k, 0);
            const uint32_t* Li = S.L.row(i);
            for (uint32_t r = 0; r < c; ++r) {
                const uint32_t f = Li[r];
                if (!f) continue;
                const uint32_t* yr = Y.data() + size_t(r) * k;
                if (Z.two) {
                    // 2^K: 64-bit wrap-around is exact mod 2^64, hence mod 2^K (masked below)
                    for (uint32_t t = 0; t < k; ++t) acc[t] += uint64_t(f) * yr[t];
                } else {
                    for (uint32_t t = 0; t < k; ++t)
                        if (yr[t]) acc[t] = (acc[t] + uint64_t(f) * yr[t]) % Z.m;
                }
            }
            if (Z.two)
                for (auto& a : acc) a &= Z.mask;
        };
        std::vector<uint64_t> acc;
        for (uint32_t a = 0; a < nj; ++a) {
            coord(J[a], acc);
            const uint32_t mod = PowU(Z.ell, R.sigma[a]);
            for (uint32_t b = 0; b < nj; ++b) Phi.at(a, b) = static_cast<uint32_t>(acc[b] % mod);
            // relations: the image of a torsion generator has no component on the free rows -- checked below; and
            // Phi_ab ell^{sigma_b} = 0 mod ell^{sigma_a} (an endomorphism), when the module is certified
            if (R.module_certified)
                for (uint32_t b = 0; b < nj; ++b)
                    if (uint64_t(Phi.at(a, b)) * PowU(Z.ell, R.sigma[b]) % mod != 0) {
                        R.code = D3Refusal::NotEndomorphism;
                        R.why = "REFUSED [NotEndomorphism] the relation ell^{sigma_j} x_j is not carried into the relations (entry (" +
                                std::to_string(a) + "," + std::to_string(b) + "))";
                        R.ok = false;
                        return R;
                    }
        }
        for (uint32_t a = 0; a < nfree; ++a) {
            coord(R.s + a, acc);
            for (uint32_t b = 0; b < nfree; ++b) Fr.at(a, b) = ZKp.red(static_cast<uint32_t>(acc[nj + b]));
        }
        // torsion goes to torsion: zero free coordinates (rows >= s of L) on the J columns, mod ell^{K'}
        bool tors_ok = true;
        if (R.s < c && nj > 0) {
            if (exact_free) {
                for (uint32_t i = R.s; i < c && tors_ok; ++i) {
                    coord(i, acc);
                    for (uint32_t t = 0; t < nj && tors_ok; ++t) tors_ok = ZKp.red(static_cast<uint32_t>(acc[t])) == 0;
                }
            } else {
                for (const auto& w : combos) {
                    for (uint32_t t = 0; t < nj && tors_ok; ++t) {
                        uint64_t s = 0;
                        for (uint32_t r = 0; r < c; ++r)
                            if (w[r]) s = (s + uint64_t(w[r]) * Y[size_t(r) * k + t]) % Z.m;
                        tors_ok = ZKp.red(static_cast<uint32_t>(s)) == 0;
                    }
                    if (!tors_ok) break;
                }
            }
        }
        if (!tors_ok) {
            R.code = D3Refusal::NotEndomorphism;
            R.why = "REFUSED [NotEndomorphism] the image of a torsion generator has a non-zero free coordinate mod ell^{K'} (" +
                    std::string(exact_free ? "exact test" : "48 random combinations of the free rows") + ")";
            R.ok = false;
            return R;
        }
        // cocycles: d^q Y = 0 mod ell^{K'} on the J columns
        if (dnext) {
            for (uint32_t i = 0; i < nx.size(); ++i)
                for (uint32_t t = 0; t < nj; ++t) {
                    uint64_t s = 0;
                    for (const auto& [j, v] : nx[i]) s = (s + uint64_t(v) * Y[size_t(j) * k + t]) % Z.m;
                    if (ZKp.red(static_cast<uint32_t>(s)) != 0) {
                        R.code = D3Refusal::NotCocycle;
                        R.why = "REFUSED [NotCocycle] d^q F x~_j != 0 mod ell^{K'} (row " + std::to_string(i) + ", generator " + std::to_string(t) +
                                "): the operator is not a cochain map";
                        R.ok = false;
                        return R;
                    }
                }
        }
        R.Phi.push_back(std::move(Phi));
        if (with_free) R.Free.push_back(std::move(Fr));
    }
    R.apply_seconds = since(t1);
    return R;
}

// =================================================================================================================================
// the invariants (prop:torsionhecke(1))
// =================================================================================================================================
std::vector<uint32_t> CharpolyModPrimePower(const Mat& A, uint32_t ell, uint32_t e) {
    const uint64_t M = PowU(ell, e);
    const uint32_t n = A.r;
    ENGINE_CHECK_MSG(A.r == A.c, "CharpolyModPrimePower: square");
    auto at = [&](uint32_t i, uint32_t j) { return uint64_t(A.at(i, j)) % M; };
    std::vector<uint64_t> pc{1};   // Berkowitz, high -> low
    for (uint32_t r = 1; r <= n; ++r) {
        const uint32_t m = r - 1;
        std::vector<uint64_t> t(r + 1, 0);
        t[0] = 1;
        t[1] = (M - at(m, m)) % M;
        std::vector<uint64_t> v(m);
        for (uint32_t i = 0; i < m; ++i) v[i] = at(i, m);
        for (uint32_t kk = 2; kk <= r; ++kk) {
            uint64_t s = 0;
            for (uint32_t i = 0; i < m; ++i) s = (s + at(m, i) * v[i]) % M;
            t[kk] = (M - s) % M;
            std::vector<uint64_t> nv(m, 0);
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < m; ++j) nv[i] = (nv[i] + at(i, j) * v[j]) % M;
            v.swap(nv);
        }
        std::vector<uint64_t> np(r + 1, 0);
        for (uint32_t i = 0; i <= r; ++i)
            for (uint32_t j = 0; j < r && j <= i; ++j) np[i] = (np[i] + t[i - j] * pc[j]) % M;
        pc.swap(np);
    }
    std::vector<uint32_t> out(pc.size());
    for (size_t i = 0; i < pc.size(); ++i) out[i] = static_cast<uint32_t>(pc[i]);
    return out;
}

bool CommuteOnTorsion(uint32_t ell, const std::vector<uint32_t>& sigma, const Mat& A, const Mat& B) {
    const uint32_t r = static_cast<uint32_t>(sigma.size());
    for (uint32_t i = 0; i < r; ++i) {
        const uint64_t mod = PowU(ell, sigma[i]);
        for (uint32_t k = 0; k < r; ++k) {
            uint64_t ab = 0, ba = 0;
            for (uint32_t j = 0; j < r; ++j) {
                ab = (ab + uint64_t(A.at(i, j)) * B.at(j, k)) % mod;
                ba = (ba + uint64_t(B.at(i, j)) * A.at(j, k)) % mod;
            }
            if (ab != ba) return false;
        }
    }
    return true;
}

TorsionInvariants Invariants(uint32_t ell, const std::vector<uint32_t>& sigma, const Mat& Phi, bool module_certified, uint32_t Kp, bool full,
                             uint32_t max_full_summands, uint32_t kc_k) {
    TorsionInvariants T;
    const uint32_t r = static_cast<uint32_t>(sigma.size());
    T.a = sigma;
    std::sort(T.a.begin(), T.a.end());
    if (r == 0) {
        T.ok = true;
        T.canon = "a=-";
        T.layers = "P=;fit=";
        return T;
    }
    const uint32_t amax = T.a.back();
    std::map<uint32_t, std::vector<uint32_t>> layer;   // j -> indices (Smith order)
    for (uint32_t i = 0; i < r; ++i) layer[sigma[i]].push_back(i);
    std::ostringstream P, B;
    std::map<uint32_t, std::vector<uint32_t>> Pj;
    bool first = true;
    for (const auto& [j, idx] : layer) {
        const uint32_t m = static_cast<uint32_t>(idx.size());
        std::vector<uint32_t> blk(size_t(m) * m);
        for (uint32_t a = 0; a < m; ++a)
            for (uint32_t b = 0; b < m; ++b) blk[size_t(a) * m + b] = Phi.at(idx[a], idx[b]) % ell;
        Pj[j] = CharpolyFp(blk, m, ell);
        P << (first ? "" : "|") << j << ":";
        for (size_t t = 0; t < Pj[j].size(); ++t) P << (t ? "," : "") << Pj[j][t];
        // (1)(e): e_j = min(j, d_j) (and K' when the module is not certified)
        uint32_t e = j;
        for (const auto& kv : layer)
            if (kv.first != j) e = std::min(e, kv.first > j ? kv.first - j : j - kv.first);
        if (!module_certified) e = std::min(e, Kp);
        // e = 1: the block's polynomial mod ell is P_j itself (Hessenberg, O(m^3)); e >= 2: division-free Berkowitz over Z/ell^e (O(m^4)),
        // up to m = kMaxBerkowitz -- above it the block polynomial is reported as not computed ("j:e:-"), never guessed
        constexpr uint32_t kMaxBerkowitz = 300;
        B << (first ? "" : "|") << j << ":" << e << ":";
        if (e == 1) {
            for (size_t t = 0; t < Pj[j].size(); ++t) B << (t ? "," : "") << Pj[j][t];
        } else if (m > kMaxBerkowitz) {
            B << "-";
        } else {
            Mat Bm(m, m);
            for (uint32_t a = 0; a < m; ++a)
                for (uint32_t b = 0; b < m; ++b) Bm.at(a, b) = Phi.at(idx[a], idx[b]);
            const std::vector<uint32_t> bp = CharpolyModPrimePower(Bm, ell, e);
            for (size_t t = 0; t < bp.size(); ++t) B << (t ? "," : "") << bp[t];
        }
        first = false;
    }
    // (1)(b): the Fitting types from the layers: G_[c] has mult_{x - c}(P_j) summands of order ell^j
    std::ostringstream F;
    for (uint32_t c0 = 0; c0 < ell; ++c0) {
        std::vector<uint32_t> ex;
        for (const auto& [j, poly] : Pj) {
            std::vector<uint32_t> pp = poly;   // high -> low
            uint32_t mult = 0;
            while (pp.size() > 1) {
                std::vector<uint32_t> qq;
                uint64_t accv = 0;
                for (uint32_t coef : pp) {
                    accv = (accv * c0 + coef) % ell;
                    qq.push_back(static_cast<uint32_t>(accv));
                }
                const uint32_t rem = qq.back();
                qq.pop_back();
                if (rem) break;
                ++mult;
                pp = qq;
            }
            for (uint32_t t = 0; t < mult; ++t) ex.push_back(j);
        }
        F << (c0 ? "|" : "") << c0 << ":" << FmtExps(ex);
    }
    T.layers = "P=" + P.str() + ";fit=" + F.str();
    std::ostringstream C;
    C << "a=" << FmtExps(T.a) << ";P=" << P.str() << ";B=" << B.str();
    // (1)(c): ker / coker (phi - c), c = 0 .. ell^{amax} - 1, through D2's InducedMap on the presentation (+)_i Z/ell^{a_i}
    const uint32_t kmax = MaxK(ell);
    T.full = module_certified && full && r <= max_full_summands && kmax >= amax + 2;
    if (T.full) {
        // (ICENSUS-P1) the working precision of the read: k_0 = min(K_max, 2 a_max + 3) first; on a precision shortfall (a rank not
        // certified or a precision exhausted inside InducedMap, MapRes::precision) the whole read is redone at k = K_max (30 at 2, 19 at
        // 3, ...: ell^K < 2^31, the word bound of maps::ZMod), and only a shortfall there refuses; a failed safeguard is never escalated.  Every read certifies itself (rank_K = rank_Q, the precision checks), so the width changes
        // whether the types are read, never what is read: a certified ker / coker type is choice-free and k-independent.
        auto read_at = [&](uint32_t kk, std::string& kt, std::string& ct, std::string& why, bool& prec) {
            const ZMod Zk(ell, kk);
            maps::Pres X = maps::ZeroPres(Zk, r);
            X.tors = sigma;
            X.gensT = maps::Mat(r, r);
            X.coord = maps::Mat(r, r);
            for (uint32_t i = 0; i < r; ++i) X.gensT.at(i, i) = X.coord.at(i, i) = 1;
            const uint32_t E = PowU(ell, amax);
            std::ostringstream Kt, Ct;
            for (uint32_t c0 = 0; c0 < E; ++c0) {
                maps::Mat f(r, r);
                for (uint32_t i = 0; i < r; ++i)
                    for (uint32_t j = 0; j < r; ++j) f.at(i, j) = Zk.red(Phi.at(i, j));
                for (uint32_t i = 0; i < r; ++i) f.at(i, i) = Zk.sub(f.at(i, i), Zk.red(c0));
                const maps::MapRes M = maps::InducedMap(X, X, f, 0);
                if (!M.ok) {
                    why = "ker / coker of phi - " + std::to_string(c0) + ": " + M.why + " (k = " + std::to_string(kk) + ")";
                    prec = M.precision;
                    return false;
                }
                Kt << (c0 ? "|" : "") << c0 << ":" << FmtExps(TorsExps(M.ker, ell));
                Ct << (c0 ? "|" : "") << c0 << ":" << FmtExps(TorsExps(M.coker, ell));
            }
            kt = Kt.str();
            ct = Ct.str();
            return true;
        };
        ENGINE_CHECK_MSG(kc_k == 0 || (kc_k >= amax + 2 && kc_k <= kmax), "Invariants: a forced ker / coker width in [a_max + 2, K_max]");
        const uint32_t k0 = kc_k ? kc_k : std::min(kmax, 2 * amax + 3);
        std::string Kt, Ct, why0, why1;
        bool prec0 = false, prec1 = false;
        T.kc_k = k0;
        if (!read_at(k0, Kt, Ct, why0, prec0)) {
            const bool stop = kc_k || k0 >= kmax || !prec0;
            if (stop || !read_at(kmax, Kt, Ct, why1, prec1)) {
                T.ok = false;
                T.kc_precision = stop ? prec0 : prec1;
                T.why = stop ? why0 : why1 + "; first " + why0;
                if (!stop) T.kc_k = kmax;
                return T;
            }
            T.kc_k = kmax;
            T.kc_escalated = true;
        }
        C << ";ker=" << Kt << ";cok=" << Ct;
    }
    C << ";fit=" << F.str();
    T.canon = C.str();
    T.ok = true;
    return T;
}

// =================================================================================================================================
// (E) the pieces
// =================================================================================================================================
namespace {
bool PrimesAdmissible(uint32_t N, const std::vector<uint32_t>& ps, TorsionEntry& E) {
    for (uint32_t p : ps) {
        if (p < 2 || !IsPrime(p)) {
            E.refused = true;
            E.code = D3Refusal::NotPrime;
            E.refusal = "REFUSED [NotPrime] p = " + std::to_string(p);
            return false;
        }
        if (N % p == 0) {
            E.refused = true;
            E.code = D3Refusal::PDividesN;
            E.refusal = "REFUSED [PDividesN] p = " + std::to_string(p) + " divides N = " + std::to_string(N) + " (U_p deferred, [R-Q4])";
            return false;
        }
    }
    return true;
}
void Finish(TorsionEntry& E, uint32_t ell, bool full) {
    if (!E.read.ok) {
        E.refused = true;
        E.code = E.read.code;
        E.refusal = E.read.why;
        return;
    }
    if (E.read.code == D3Refusal::ModulePrecision) {
        E.code = D3Refusal::ModulePrecision;   // partial: P_j and Fitting types kept
        E.refusal = E.read.why;
    }
    for (size_t o = 0; o < E.read.Phi.size(); ++o)
        E.inv.push_back(Invariants(ell, E.read.sigma, E.read.Phi[o], E.read.module_certified, E.read.Kp, full));
    // (D6, [C] v1.23.0-pre.37 sec:recordschema item 7 (j)) Invariants() fails only in the ker / coker read of (1)(c), on a module
    // whose read certified it with T_p an endomorphism: a precision shortfall at the last width of the ladder is [KerCokerPrecision];
    // (ICENSUS-P1, [C] v1.23.0-pre.39) a failed safeguard inside that read (relations not carried, I off L, the rank input) is
    // [NotEndomorphism], as the read's own checks above
    for (const auto& iv : E.inv)
        if (!iv.ok) {
            E.refused = true;
            E.code = iv.kc_precision ? D3Refusal::KerCokerPrecision : D3Refusal::NotEndomorphism;
            E.refusal = std::string("REFUSED [") + D3RefusalName(E.code) + "] " + iv.why;
            return;
        }
    if (E.read.module_certified)
        for (size_t a = 0; a < E.read.Phi.size(); ++a)
            for (size_t b = a + 1; b < E.read.Phi.size(); ++b)
                E.commute = E.commute && CommuteOnTorsion(ell, E.read.sigma, E.read.Phi[a], E.read.Phi[b]);
}
bool OperatorsOrRefuse(std::vector<std::unique_ptr<IsoHeckeN>>& T, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const U32Ctx& ctx,
                       uint32_t q, TorsionEntry& E) {
    for (uint32_t p : ps) {
        T.push_back(std::make_unique<IsoHeckeN>(N, n, p, ctx, q));
        if (T.back()->refused()) {
            E.refused = true;
            E.code = T.back()->code();
            E.refusal = T.back()->why();
            return false;
        }
    }
    return true;
}
}  // namespace

TorsionEntry TorsionOnTree(uint32_t ell, uint32_t N, uint32_t n, uint32_t q, const std::vector<uint32_t>& ps, uint32_t K, uint64_t rss_cap,
                           bool full) {
    const auto t0 = std::chrono::steady_clock::now();
    TorsionEntry E;
    E.q = q;
    E.ell = ell;
    E.route = "tree";
    E.ps = ps;
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "TorsionOnTree: ell in {2, 3}");
    if (!PrimesAdmissible(N, ps, E)) return E;
    if (q == 0 || q > 4) {
        E.refused = true;
        E.code = D3Refusal::DegreeRange;
        E.refusal = "REFUSED [DegreeRange] q = " + std::to_string(q) + " (the tree read serves 1 <= q <= 4; q_max = 3 determines every degree)";
        return E;
    }
    E.rss_pred = PredictTreeReadRss(ell, N, n, q);
    if (E.rss_pred > rss_cap) {
        E.refused = true;
        E.code = D3Refusal::RssCeiling;
        E.refusal = "REFUSED [RssCeiling] rule 19: the tree read predicts " + std::to_string(E.rss_pred >> 20) + " MiB above the ceiling " +
                    std::to_string(rss_cap >> 20) + " MiB (refused before building)";
        return E;
    }
    const TreeTransport X(ell, N, n, q, K);
    const Mat d = X.differential(q - 1), dn = X.differential(q);
    std::vector<std::unique_ptr<IsoHeckeN>> T;
    if (!OperatorsOrRefuse(T, N, n, ps, X.ctx(), q, E)) return E;
    std::vector<BatchOp> ops;
    for (const auto& t : T) {
        const IsoHeckeN* tp = t.get();
        ops.push_back(Chunked([&X, tp, q](const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& Y) { X.apply_hecke(*tp, q, U, k, Y); },
                              X.rank(q), 128));
    }
    E.read = ReadTorsion(X.Z(), d, X.rank_Q(q - 1), &dn, ops);
    Finish(E, ell, full);
    E.seconds = since(t0);
    return E;
}

uint64_t PredictCokernelReadRss(uint32_t N, uint32_t n) {
    const LevelNCounts c = LevelNClosedForm(N, n);
    const uint64_t D = c.D, d = n + 1, m = c.m;
    const uint64_t G = d * ((m + c.nu2) / 2 + (m + 2ull * c.nu3) / 3);
    return 4ull * (3ull * D * G + 2ull * D * D) + (256ull << 20);
}

TorsionEntry TorsionOnCokernel(uint32_t ell, uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, uint32_t K, bool with_free, bool full,
                               uint64_t rss_cap) {
    const auto t0 = std::chrono::steady_clock::now();
    TorsionEntry E;
    E.q = 1;
    E.ell = ell;
    E.route = "cokernel";
    E.ps = ps;
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime(ell), "TorsionOnCokernel: ell >= 5 prime (6 a unit, prop:heilbronn)");
    if (!PrimesAdmissible(N, ps, E)) return E;
    if (n % 2 == 1) {
        E.read.ok = true;   // n odd: 2 kills everything (prop:badprimesN(2)); no ell-torsion
        return E;
    }
    // rule 19 (the D3 audit's M1): refused before anything is built
    E.rss_pred = PredictCokernelReadRss(N, n);
    if (E.rss_pred > rss_cap) {
        E.refused = true;
        E.code = D3Refusal::RssCeiling;
        E.refusal = "REFUSED [RssCeiling] rule 19: the cokernel read at l = " + std::to_string(ell) + " predicts " + std::to_string(E.rss_pred >> 20) +
                    " MiB above the ceiling " + std::to_string(rss_cap >> 20) + " MiB (refused before building)";
        return E;
    }
    const uint32_t KK = K ? K : MaxK(ell);
    const ZMod Z(ell, KK);
    const U32Ctx ctx = U32Ctx::prime_power(ell, KK);
    const engine::DenseBuf<uint32_t> G = FixedGeneratorRows(N, n, ctx);   // generators x D (rows generating M^S + M^ST)
    const uint32_t D = G.cols;
    Mat A(D, G.rows);
    for (uint32_t i = 0; i < G.rows; ++i)
        for (uint32_t j = 0; j < D; ++j) A.at(j, i) = G.at(i, j);
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    ENGINE_CHECK_MSG(G.rows <= uint64_t(n + 1) * ((cnt.m + cnt.nu2) / 2 + (cnt.m + 2ull * cnt.nu3) / 3),
                     "TorsionOnCokernel: more generators than PredictCokernelReadRss's bound");
    const hecke::level::Gamma0LevelN lvl(N);
    std::vector<std::unique_ptr<hecke::level::HeilbronnOperator>> H;
    for (uint32_t p : ps) H.push_back(std::make_unique<hecke::level::HeilbronnOperator>(lvl, p, n, ctx));
    std::vector<BatchOp> ops;
    for (const auto& h : H) {
        const hecke::level::HeilbronnOperator* hp = h.get();
        ops.push_back([hp, D](const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y) {
            Y.assign(size_t(D) * k, 0);
            std::vector<uint32_t> col(D);
            for (uint32_t t = 0; t < k; ++t) {
                for (uint32_t r = 0; r < D; ++r) col[r] = X[size_t(r) * k + t];
                const std::vector<uint32_t> y = hp->apply(col);
                for (uint32_t r = 0; r < D; ++r) Y[size_t(r) * k + t] = y[r];
            }
        });
    }
    // rank_Q of the presentation = dim Rel = D - b_1 (prop:levelorbit(2))
    E.read = ReadTorsion(Z, A, D - cnt.b1, nullptr, ops, with_free);
    Finish(E, ell, full);
    E.seconds = since(t0);
    return E;
}

TorsionEntry TorsionOnFreeIso(uint32_t ell, uint32_t N, uint32_t n, uint32_t q, const std::vector<uint32_t>& ps, uint32_t K) {
    const auto t0 = std::chrono::steady_clock::now();
    TorsionEntry E;
    E.q = q;
    E.ell = ell;
    E.route = "iso";
    E.ps = ps;
    if (!PrimesAdmissible(N, ps, E)) return E;
    const ZMod Z(ell, K);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const pair::PairComplex X = pair::BuildFreePairN(N, n, ctx, q + 1);
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    const uint32_t b0 = n % 2 ? 0u : cnt.b0, b1 = n % 2 ? 0u : cnt.b1;
    uint32_t prev = 0;
    for (uint32_t t = 0; t < q; ++t) prev = X.c[t] - (t == 0 ? b0 : (t == 1 ? b1 : 0u)) - prev;
    std::vector<std::unique_ptr<IsoHeckeN>> T;
    if (!OperatorsOrRefuse(T, N, n, ps, ctx, q, E)) return E;
    std::vector<BatchOp> ops;
    for (const auto& t : T) {
        const IsoHeckeN* tp = t.get();
        ops.push_back([tp, q](const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& Y) { tp->apply(q, U, k, Y); });
    }
    E.read = ReadTorsion(Z, X.d[q - 1], prev, &X.d[q], ops);
    Finish(E, ell, true);
    E.seconds = since(t0);
    return E;
}

TorsionEntry TorsionOnIsoEll(uint32_t ell, uint32_t N, uint32_t n, uint32_t q, const std::vector<uint32_t>& ps) {
    const auto t0 = std::chrono::steady_clock::now();
    TorsionEntry E;
    E.q = q;
    E.ell = ell;
    E.route = ell == 2 ? "Iso2" : "Iso3";
    E.ps = ps;
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "TorsionOnIsoEll: ell in {2, 3}");
    if (!PrimesAdmissible(N, ps, E)) return E;
    const uint32_t K = LocalWidth(ell);
    const ZMod Z(ell, K);
    const U32Ctx ctx = U32Ctx::prime_power(ell, K);
    const integral_model::LocalModel& L = ell == 2 ? Iso2() : Iso3();
    const integral_model::LocalModel& F = pair::FreeIsoModel();
    static std::mutex mu;
    static std::map<uint32_t, std::pair<integral_model::ComparisonMap, integral_model::ComparisonMap>> cmps;
    const integral_model::ComparisonMap* Phi;
    const integral_model::ComparisonMap* Psi;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cmps.find(ell);
        if (it == cmps.end())
            it = cmps.emplace(ell, std::make_pair(integral_model::StrictComparison(F, L), integral_model::StrictComparison(L, F))).first;
        Phi = &it->second.first;
        Psi = &it->second.second;
    }
    const auto M = InducedSymModule(N, n, ctx);
    const pair::PairComplex X = pair::BuildModelPair(L, ctx, M, q + 1, true);
    const Mat PhiS = pair::FromDenseBuf(integral_model::Pullback(ctx, M, q, F, L, *Phi));   // C_ell full -> C_I
    const Mat PsiS = pair::FromDenseBuf(integral_model::Pullback(ctx, M, q, L, F, *Psi));   // C_I -> C_ell full
    const Mat Pin = pair::MatMulZ(Z, PhiS, X.emb[q]);                                       // C_ell (J) -> C_I
    const Mat Pout = pair::MatMulZ(Z, X.coords[q], PsiS);                                   // C_I -> C_ell (J)
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    const uint32_t b0 = n % 2 ? 0u : cnt.b0, b1 = n % 2 ? 0u : cnt.b1;
    uint32_t prev = 0;
    for (uint32_t t = 0; t < q; ++t) prev = X.c[t] - (t == 0 ? b0 : (t == 1 ? b1 : 0u)) - prev;
    std::vector<std::unique_ptr<IsoHeckeN>> T;
    if (!OperatorsOrRefuse(T, N, n, ps, ctx, q, E)) return E;
    std::vector<BatchOp> ops;
    for (const auto& t : T) {
        const IsoHeckeN* tp = t.get();
        ops.push_back([tp, q, &Pin, &Pout, &Z](const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& Y) {
            const Mat W = pair::MatMulZ(Z, Pin, ToMat(U, Pin.c, k));
            std::vector<uint32_t> W2;
            tp->apply(q, W.a, k, W2);
            Y = pair::MatMulZ(Z, Pout, ToMat(W2, Pout.c, k)).a;
        });
    }
    E.read = ReadTorsion(Z, X.d[q - 1], prev, &X.d[q], ops);
    Finish(E, ell, true);
    E.seconds = since(t0);
    return E;
}

FreeQuotient FreeQuotientFactors(uint32_t ell, uint32_t N, uint32_t n, uint32_t p, int64_t c, uint32_t K) {
    FreeQuotient Q;
    const TorsionEntry E = TorsionOnCokernel(ell, N, n, {p}, K, true, false);
    if (E.refused || E.read.Free.empty()) {
        Q.why = E.refused ? E.refusal : "no free part";
        return Q;
    }
    const ZMod Zp(ell, E.read.Kp);
    Mat X = E.read.Free[0];
    const uint32_t r = X.r;
    const int64_t m = static_cast<int64_t>(Zp.m);
    int64_t cr = c % m;
    if (cr < 0) cr += m;
    for (uint32_t i = 0; i < r; ++i) X.at(i, i) = Zp.sub(X.at(i, i), static_cast<uint32_t>(cr));
    const maps::Smith S = maps::SmithForm(Zp, X, false, false);
    if (S.rank() != r) {
        Q.why = "REFUSED: the Smith form of T_p - c mod " + std::to_string(ell) + "^" + std::to_string(Zp.k) + " has " + std::to_string(S.rank()) +
                " < r = " + std::to_string(r) + " pivots (c a root of chi, or k too small; prop:torsionhecke(4): nothing asserted)";
        return Q;
    }
    for (uint32_t s : S.sigma)
        if (s) Q.exps.push_back(s);
    std::sort(Q.exps.begin(), Q.exps.end());
    Q.ok = true;
    return Q;
}

// =================================================================================================================================
// (F) the record
// =================================================================================================================================
std::vector<uint32_t> RecordTorsionPrimes(uint32_t N, uint32_t n) {
    std::vector<uint32_t> ells = {2};
    if (n % 2 == 0) {
        ells.push_back(3);
        if (n > 0)
            for (const auto& [ell, s] : BadPrimesN(N, n)) ells.push_back(ell);
    }
    return ells;
}

HeckeRecord RunDescentHecke(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const HeckeRecordOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    HeckeRecord H;
    H.N = N;
    H.n = n;
    H.ps = ps;
    RssSampler rs;
    if (opt.charpoly) {
        const auto t1 = std::chrono::steady_clock::now();
        for (uint32_t p : ps) {
            oc_util::CancelPoint("descent.charpoly");   // (NBCONFIRM S5) per Hecke prime, on the calling thread
            H.chi.push_back(CharpolyOverZ(N, n, p, opt.crt));
        }
        H.t_chi = since(t1);
    }
    if (opt.torsion) {
        const auto t2 = std::chrono::steady_clock::now();
        for (uint32_t ell : RecordTorsionPrimes(N, n)) {
            if (!opt.only_ells.empty() && std::find(opt.only_ells.begin(), opt.only_ells.end(), ell) == opt.only_ells.end()) continue;
            H.bad.push_back(ell);
            if (ell <= 3) {
                for (uint32_t q = 1; q <= opt.q_max; ++q) {
                    oc_util::CancelPoint("descent.torsion");   // (NBCONFIRM S5) per torsion read
                    H.tors.push_back(TorsionOnTree(ell, N, n, q, ps, 0, opt.rss_cap_bytes, opt.full_invariants));
                }
            } else {
                oc_util::CancelPoint("descent.torsion");
                H.tors.push_back(TorsionOnCokernel(ell, N, n, ps, 0, false, opt.full_invariants, opt.rss_cap_bytes));
            }
        }
        H.t_tors = since(t2);
    }
    H.rss_peak = rs.stop();
    // the record marks itself refused when an entry was (a legitimate limit; a defect has propagated by now, the D3 audit's m2)
    auto note = [&H](const std::string& why) {
        if (!H.refused_entries++) H.refusal = why;
    };
    for (const auto& C : H.chi)
        if (C.refused) note(C.refusal);
    for (const auto& E : H.tors)
        if (E.refused) note(E.refusal);
    H.refused = H.refused_entries > 0;
    if (H.refused_entries > 1) H.refusal += " (and " + std::to_string(H.refused_entries - 1) + " more refused entries)";
    H.t_total = since(t0);
    return H;
}

}  // namespace hecke3
}  // namespace integral_descent
