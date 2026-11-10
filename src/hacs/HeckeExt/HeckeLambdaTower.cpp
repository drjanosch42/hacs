// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, P10) The real Hida tower -- see HeckeLambdaTower.h and docs/implementation/slot22/P10_spec.md.
#include "HeckeLambdaTower.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

#include "../Hecke/CoefficientRing.h"   // hecke::IsPrimeU64

#include "../Hecke/HeckeGammaH.h"
#include "../Hecke/HeckeLevelN.h"
#include "../Hecke/HeckeLevelNSparse.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ScalarMod.h"

namespace hecke_ext {
namespace tower {

using hecke::Mat22;
using hecke::level::GammaHLevelN;
using hecke::level::SparseVec;
using namespace hecke::level::tower;

namespace {

double Since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
inline uint64_t IPow(uint64_t b, uint32_t e) { return engine::scalar::pow_wrap(b, e); }

// Gaussian elimination mod ell on a copy: rows of `rows` (each of length n); returns the pivot list (column per row).
struct EchelonModEll {
    uint32_t ell = 0, n = 0;
    std::vector<std::vector<uint32_t>> basis;   // reduced vectors, basis[k][piv[k]] = 1
    std::vector<uint32_t> piv;
    // reduce x against the basis; true (and the vector added) if it is independent
    bool add(std::vector<uint32_t> x) {
        for (size_t k = 0; k < basis.size(); ++k) {
            const uint32_t f = x[piv[k]];
            if (!f) continue;
            const uint32_t mf = ell - f;
            const std::vector<uint32_t>& b = basis[k];
            for (uint32_t i = 0; i < n; ++i) if (b[i]) x[i] = (x[i] + mf * b[i]) % ell;
        }
        uint32_t p = n;
        for (uint32_t i = 0; i < n; ++i) if (x[i]) { p = i; break; }
        if (p == n) return false;
        uint32_t inv = 1;
        for (uint32_t t = 1; t < ell; ++t) if (t * x[p] % ell == 1) { inv = t; break; }
        for (uint32_t i = 0; i < n; ++i) x[i] = x[i] * inv % ell;
        basis.push_back(std::move(x));
        piv.push_back(p);
        return true;
    }
};

std::vector<uint32_t> ColModEll(const DM& A, uint32_t j, uint32_t ell) {
    std::vector<uint32_t> v(A.r);
    for (uint32_t i = 0; i < A.r; ++i) v[i] = A.at(i, j) % ell;
    return v;
}

// col-major block (h x s) <-> DM
DM BlockToDM(const std::vector<uint32_t>& B, uint32_t h, uint32_t s) {
    DM M(h, s);
    for (uint32_t k = 0; k < s; ++k)
        for (uint32_t i = 0; i < h; ++i) M.at(i, k) = B[static_cast<size_t>(k) * h + i];
    return M;
}
std::vector<uint32_t> DMToBlock(const DM& M) {
    std::vector<uint32_t> B(static_cast<size_t>(M.r) * M.c);
    for (uint32_t k = 0; k < M.c; ++k)
        for (uint32_t i = 0; i < M.r; ++i) B[static_cast<size_t>(k) * M.r + i] = M.at(i, k);
    return B;
}
DM ApplyOp(const QuotientOperator& X, const DM& B, const ZK& R) {
    std::vector<uint32_t> Y;
    X.apply_block(R, DMToBlock(B), B.c, Y);
    return BlockToDM(Y, B.r, B.c);
}

// The binomial expansion helper: (1 + X)^i coefficients mod Q, i < L.
std::vector<std::vector<uint32_t>> Pascal(uint32_t L, const ZK& R) {
    std::vector<std::vector<uint32_t>> P(L, std::vector<uint32_t>(L, 0));
    for (uint32_t i = 0; i < L; ++i) {
        P[i][0] = static_cast<uint32_t>(1 % R.Q);
        for (uint32_t j = 1; j <= i; ++j) P[i][j] = R.add(P[i - 1][j - 1], j < i ? P[i - 1][j] : 0);
    }
    return P;
}

// The certified block Krylov iteration of U (spec section 3 step 4).
struct KrylovOut {
    uint32_t T = 0, rank = 0;
    std::vector<uint32_t> ranks;
    bool stable = false, free = false, ustable = false, uinv = false;
    DM block;               // the final block U^T B
    Subspace E;
    DM UonE;
};

KrylovOut Krylov(const QuotientOperator& U, uint32_t h, const ZK& R, std::vector<uint32_t> B, uint32_t s, uint32_t T0,
                 int64_t fixed_T) {
    KrylovOut out;
    std::vector<uint32_t> Y;
    auto run = [&](uint32_t n) {
        for (uint32_t i = 0; i < n; ++i) { U.apply_block(R, B, s, Y); std::swap(B, Y); }
        out.T += n;
    };
    auto rank_now = [&]() { return RankModEll(BlockToDM(B, h, s), R.ell); };
    if (fixed_T >= 0) {
        run(static_cast<uint32_t>(fixed_T));
        out.ranks.push_back(rank_now());
        out.stable = false;
    } else {
        run(T0);
        out.ranks.push_back(rank_now());
        run(T0);
        out.ranks.push_back(rank_now());
        for (;;) {
            run(out.T);   // doubling
            out.ranks.push_back(rank_now());
            const size_t k = out.ranks.size();
            if (out.ranks[k - 1] == out.ranks[k - 2] && out.ranks[k - 2] == out.ranks[k - 3]) { out.stable = true; break; }
            if (out.T >= 64ull * T0)
                throw std::logic_error("krylov_unstable: the F_ell rank of the Krylov block did not stabilise by T = 64 T0 = " +
                                       std::to_string(64ull * T0));
        }
    }
    out.block = BlockToDM(B, h, s);
    const std::vector<uint32_t> ind = IndepColsModEll(out.block, R.ell);
    out.rank = static_cast<uint32_t>(ind.size());
    if (ind.empty()) return out;
    out.E = Subspace(Cols(out.block, ind), R);
    // E free: every column of the block lies in the span over Z/ell^K
    out.free = true;
    std::vector<uint32_t> c;
    for (uint32_t k = 0; k < s && out.free; ++k) {
        std::vector<uint32_t> col(h);
        for (uint32_t i = 0; i < h; ++i) col[i] = out.block.at(i, k);
        out.free = out.E.coords(col, c);
    }
    out.ustable = out.E.restrict(ApplyOp(U, out.E.basis(), R), out.UonE);
    out.uinv = out.ustable && RankModEll(out.UonE, R.ell) == out.rank;
    return out;
}

} // namespace

// ============================================================================
// dense linear algebra
// ============================================================================
DM Ident(uint32_t n, const ZK& R) {
    DM I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.at(i, i) = static_cast<uint32_t>(1 % R.Q);
    return I;
}

DM MatMul(const DM& A, const DM& B, const ZK& R) {
    ENGINE_CHECK_MSG(A.c == B.r, "MatMul: shapes");
    DM C(A.r, B.c);
    std::vector<uint64_t> acc(B.c);
    for (uint32_t i = 0; i < A.r; ++i) {
        std::fill(acc.begin(), acc.end(), 0);
        for (uint32_t k = 0; k < A.c; ++k) {
            const uint64_t x = A.at(i, k);
            if (!x) continue;
            const uint32_t* b = &B.a[static_cast<size_t>(k) * B.c];
            for (uint32_t j = 0; j < B.c; ++j) {
                acc[j] += x * b[j];
                if (acc[j] >= (1ull << 62)) acc[j] %= R.Q;
            }
        }
        for (uint32_t j = 0; j < B.c; ++j) C.at(i, j) = static_cast<uint32_t>(acc[j] % R.Q);
    }
    return C;
}

DM MatSub(const DM& A, const DM& B, const ZK& R) {
    ENGINE_CHECK_MSG(A.r == B.r && A.c == B.c, "MatSub: shapes");
    DM C(A.r, A.c);
    for (size_t i = 0; i < A.a.size(); ++i) C.a[i] = R.sub(A.a[i], B.a[i]);
    return C;
}

DM MatPow(DM A, uint64_t e, const ZK& R) {
    DM P = Ident(A.r, R);
    while (e) {
        if (e & 1) P = MatMul(P, A, R);
        e >>= 1;
        if (e) A = MatMul(A, A, R);
    }
    return P;
}

DM ScalarSub(const DM& A, uint32_t lambda, const ZK& R) {
    DM C = A;
    for (uint32_t i = 0; i < A.r; ++i) C.at(i, i) = R.sub(C.at(i, i), lambda);
    return C;
}

uint32_t RankModEll(const DM& A, uint32_t ell) { return static_cast<uint32_t>(IndepColsModEll(A, ell).size()); }

std::vector<uint32_t> IndepColsModEll(const DM& A, uint32_t ell) {
    EchelonModEll E;
    E.ell = ell;
    E.n = A.r;
    std::vector<uint32_t> out;
    for (uint32_t j = 0; j < A.c; ++j) {
        if (E.basis.size() == A.r) break;
        if (E.add(ColModEll(A, j, ell))) out.push_back(j);
    }
    return out;
}

std::vector<uint32_t> PivotRowsModEll(const DM& A, uint32_t ell) {
    DM T(A.c, A.r);
    for (uint32_t i = 0; i < A.r; ++i)
        for (uint32_t j = 0; j < A.c; ++j) T.at(j, i) = A.at(i, j);
    return IndepColsModEll(T, ell);
}

DM Inverse(const DM& A, const ZK& R) {
    ENGINE_CHECK_MSG(A.r == A.c, "Inverse: square");
    const uint32_t n = A.r;
    DM M = A, I = Ident(n, R);
    for (uint32_t c = 0; c < n; ++c) {
        uint32_t p = n;
        for (uint32_t i = c; i < n; ++i) if (R.unit(M.at(i, c))) { p = i; break; }
        ENGINE_CHECK_MSG(p < n, "Inverse: the matrix is not invertible mod ell");
        if (p != c) for (uint32_t j = 0; j < n; ++j) { std::swap(M.at(p, j), M.at(c, j)); std::swap(I.at(p, j), I.at(c, j)); }
        const uint32_t inv = R.inv(M.at(c, c));
        for (uint32_t j = 0; j < n; ++j) { M.at(c, j) = R.mul(M.at(c, j), inv); I.at(c, j) = R.mul(I.at(c, j), inv); }
        for (uint32_t i = 0; i < n; ++i) {
            if (i == c) continue;
            const uint32_t f = M.at(i, c);
            if (!f) continue;
            const uint32_t mf = R.neg(f);
            for (uint32_t j = 0; j < n; ++j) {
                M.at(i, j) = R.add(M.at(i, j), R.mul(mf, M.at(c, j)));
                I.at(i, j) = R.add(I.at(i, j), R.mul(mf, I.at(c, j)));
            }
        }
    }
    return I;
}

DM Cols(const DM& A, const std::vector<uint32_t>& cols) {
    DM B(A.r, static_cast<uint32_t>(cols.size()));
    for (uint32_t i = 0; i < A.r; ++i)
        for (size_t k = 0; k < cols.size(); ++k) B.at(i, static_cast<uint32_t>(k)) = A.at(i, cols[k]);
    return B;
}

DM Rows(const DM& A, const std::vector<uint32_t>& rows) {
    DM B(static_cast<uint32_t>(rows.size()), A.c);
    for (size_t k = 0; k < rows.size(); ++k)
        for (uint32_t j = 0; j < A.c; ++j) B.at(static_cast<uint32_t>(k), j) = A.at(rows[k], j);
    return B;
}

std::vector<uint32_t> Charpoly(const DM& A, const ZK& R) {
    ENGINE_CHECK_MSG(A.r == A.c, "Charpoly: square");
    const uint32_t n = A.r;
    if (n == 0) return {static_cast<uint32_t>(1 % R.Q)};
    // Berkowitz: vect = coefficients of det(xI - A_r), high -> low
    std::vector<uint32_t> vect{static_cast<uint32_t>(1 % R.Q), R.neg(A.at(0, 0))};
    for (uint32_t r = 1; r < n; ++r) {
        // Q = [1, -a, -R C, -R A_r C, ..., -R A_r^{r-1} C], R = A[r][0..r), C = A[0..r)[r]
        std::vector<uint32_t> q(r + 2, 0);
        q[0] = static_cast<uint32_t>(1 % R.Q);
        q[1] = R.neg(A.at(r, r));
        std::vector<uint32_t> v(r), w(r);
        for (uint32_t i = 0; i < r; ++i) v[i] = A.at(i, r);
        for (uint32_t k = 0; k < r; ++k) {
            uint64_t s = 0;
            for (uint32_t i = 0; i < r; ++i) s = (s + static_cast<uint64_t>(A.at(r, i)) * v[i]) % R.Q;
            q[k + 2] = R.neg(static_cast<uint32_t>(s));
            if (k + 1 < r) {
                for (uint32_t i = 0; i < r; ++i) {
                    uint64_t t = 0;
                    for (uint32_t j = 0; j < r; ++j) t = (t + static_cast<uint64_t>(A.at(i, j)) * v[j]) % R.Q;
                    w[i] = static_cast<uint32_t>(t);
                }
                std::swap(v, w);
            }
        }
        std::vector<uint32_t> nv(r + 2, 0);
        for (uint32_t i = 0; i < r + 2; ++i) {
            uint64_t s = 0;
            for (uint32_t j = 0; j <= std::min(i, r); ++j) s = (s + static_cast<uint64_t>(q[i - j]) * vect[j]) % R.Q;
            nv[i] = static_cast<uint32_t>(s);
        }
        vect = std::move(nv);
    }
    std::reverse(vect.begin(), vect.end());
    return vect;
}

KerIm KernelImage(const DM& A, const ZK& R) {
    ENGINE_CHECK_MSG(A.r == A.c, "KernelImage: square");
    const uint32_t n = A.r;
    KerIm out;
    DM M = A;
    std::vector<uint32_t> pivc;
    uint32_t row = 0;
    for (uint32_t c = 0; c < n && row < n; ++c) {
        uint32_t p = n;
        for (uint32_t i = row; i < n; ++i) if (R.unit(M.at(i, c))) { p = i; break; }
        if (p == n) continue;
        if (p != row) for (uint32_t j = 0; j < n; ++j) std::swap(M.at(p, j), M.at(row, j));
        const uint32_t inv = R.inv(M.at(row, c));
        for (uint32_t j = 0; j < n; ++j) M.at(row, j) = R.mul(M.at(row, j), inv);
        for (uint32_t i = 0; i < n; ++i) {
            if (i == row) continue;
            const uint32_t f = M.at(i, c);
            if (!f) continue;
            const uint32_t mf = R.neg(f);
            for (uint32_t j = 0; j < n; ++j) M.at(i, j) = R.add(M.at(i, j), R.mul(mf, M.at(row, j)));
        }
        pivc.push_back(c);
        ++row;
    }
    for (uint32_t i = row; i < n; ++i)
        for (uint32_t j = 0; j < n; ++j)
            if (M.at(i, j)) return out;   // a non-unit residual: not split
    std::vector<uint8_t> isp(n, 0);
    for (uint32_t c : pivc) isp[c] = 1;
    std::vector<uint32_t> freec;
    for (uint32_t c = 0; c < n; ++c) if (!isp[c]) freec.push_back(c);
    out.ker = DM(n, static_cast<uint32_t>(freec.size()));
    for (size_t k = 0; k < freec.size(); ++k) {
        out.ker.at(freec[k], static_cast<uint32_t>(k)) = static_cast<uint32_t>(1 % R.Q);
        for (size_t i = 0; i < pivc.size(); ++i) out.ker.at(pivc[i], static_cast<uint32_t>(k)) = R.neg(M.at(static_cast<uint32_t>(i), freec[k]));
    }
    out.im = Cols(A, pivc);
    out.ok = true;
    return out;
}

Subspace::Subspace(const DM& basis, const ZK& R) : R_(R), B_(basis) {
    P_ = PivotRowsModEll(basis, R.ell);
    ENGINE_CHECK_MSG(P_.size() == basis.c, "Subspace: the basis is not independent mod ell");
    Ainv_ = Inverse(Rows(basis, P_), R);
}

bool Subspace::coords(const std::vector<uint32_t>& x, std::vector<uint32_t>& c) const {
    const uint32_t k = B_.c;
    c.assign(k, 0);
    for (uint32_t i = 0; i < k; ++i) {
        uint64_t s = 0;
        for (uint32_t j = 0; j < k; ++j) s = (s + static_cast<uint64_t>(Ainv_.at(i, j)) * x[P_[j]]) % R_.Q;
        c[i] = static_cast<uint32_t>(s);
    }
    for (uint32_t r = 0; r < B_.r; ++r) {
        uint64_t s = 0;
        for (uint32_t j = 0; j < k; ++j) s = (s + static_cast<uint64_t>(B_.at(r, j)) * c[j]) % R_.Q;
        if (s != x[r] % R_.Q) return false;
    }
    return true;
}

bool Subspace::restrict(const DM& XB, DM& out) const {
    out = DM(B_.c, XB.c);
    std::vector<uint32_t> x(B_.r), c;
    for (uint32_t j = 0; j < XB.c; ++j) {
        for (uint32_t i = 0; i < B_.r; ++i) x[i] = XB.at(i, j);
        if (!coords(x, c)) return false;
        for (uint32_t i = 0; i < B_.c; ++i) out.at(i, j) = c[i];
    }
    return true;
}

DM Subspace::canonical() const { return MatMul(B_, Ainv_, R_); }

// ============================================================================
// group-ring helpers
// ============================================================================
std::string TowerCell::text() const {
    return "(" + std::to_string(ell) + "," + std::to_string(a) + "," + std::to_string(r) + ")";
}

uint32_t LogU(uint64_t d, uint32_t ell, uint32_t r) {
    const ZK Rr(ell, r);
    const uint32_t om = Rr.teich(d);
    const uint32_t x = Rr.mul(static_cast<uint32_t>(d % Rr.Q), Rr.inv(om));   // <d>_ell mod ell^r
    const uint64_t L = IPow(ell, r - 1);
    uint32_t cur = static_cast<uint32_t>(1 % Rr.Q);
    for (uint64_t s = 0; s < L; ++s) {
        if (cur == x) return static_cast<uint32_t>(s);
        cur = Rr.mul(cur, static_cast<uint32_t>((1 + ell) % Rr.Q));
    }
    throw std::logic_error("LogU: <d> is not a power of u mod ell^r");
}

std::vector<uint32_t> Weierstrass(const GR& c, const ZK& R) {
    const uint32_t L = static_cast<uint32_t>(c.size());
    const auto P = Pascal(L, R);
    const uint32_t u2inv = R.inv(R.mul(1 + R.ell, 1 + R.ell));
    std::vector<uint32_t> A(L, 0);
    uint32_t w = static_cast<uint32_t>(1 % R.Q);
    for (uint32_t i = 0; i < L; ++i) {
        const uint32_t ci = R.mul(c[i], w);
        for (uint32_t j = 0; j <= i; ++j) A[j] = R.add(A[j], R.mul(ci, P[i][j]));
        w = R.mul(w, u2inv);
    }
    return A;
}

uint32_t FibreAt(const GR& c, uint32_t ell, int64_t k, uint32_t prec) {
    ENGINE_CHECK_MSG(k >= 2, "FibreAt: k >= 2");
    const ZK Rp(ell, prec);
    const uint32_t x = Rp.pow(static_cast<uint32_t>((1 + ell) % Rp.Q), static_cast<uint64_t>(k - 2));
    uint64_t s = 0, w = 1 % Rp.Q;
    for (size_t i = 0; i < c.size(); ++i) {
        s = (s + (c[i] % Rp.Q) * w) % Rp.Q;
        w = w * x % Rp.Q;
    }
    return static_cast<uint32_t>(s);
}

GR ReduceTo(const GR& c, uint32_t Lp, const ZK& R) {
    GR out(Lp, 0);
    for (size_t i = 0; i < c.size(); ++i) out[i % Lp] = R.add(out[i % Lp], c[i]);
    return out;
}

GR EisensteinScalar(uint32_t ell, uint32_t a, uint32_t r, uint64_t p, const ZK& R, bool untwisted) {
    const uint32_t L = static_cast<uint32_t>(IPow(ell, r - 1));
    const uint32_t s = LogU(p, ell, r) % L;
    GR c(L, 0);
    c[0] = static_cast<uint32_t>(1 % R.Q);
    const uint32_t om = R.teich(p);
    const uint64_t e2 = static_cast<uint64_t>((static_cast<int64_t>(a) - 2) % static_cast<int64_t>(ell - 1) + (ell - 1)) % (ell - 1);
    if (!untwisted) {
        c[s] = R.add(c[s], R.mul(static_cast<uint32_t>(p % R.Q), R.pow(om, e2)));
    } else {
        const uint64_t e1 = (e2 + 1) % (ell - 1);
        const uint32_t br = R.mul(static_cast<uint32_t>(p % R.Q), R.inv(om));   // <p>_ell in Z/ell^K
        c[s] = R.add(c[s], R.mul(R.pow(om, e1), R.inv(br)));
    }
    return c;
}

const TowerPart* TowerReading::part(const std::string& kind) const {
    for (const TowerPart& p : parts) if (p.kind == kind) return &p;
    return nullptr;
}

// ============================================================================
// BuildTower
// ============================================================================
TowerReading BuildTower(const TowerCell& cell, const TowerOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    TowerReading t;
    t.cell = cell;
    const uint32_t ell = cell.ell, r = cell.r, K = cell.K;
    ENGINE_CHECK_MSG(ell >= 5 && hecke::IsPrimeU64(ell), "nebentypus_small_ell: the tower needs ell >= 5 prime");
    ENGINE_CHECK_MSG(r >= 1 && r <= 3, "BuildTower: r in [1, 3] (r >= 4 is out of scope)");
    ENGINE_CHECK_MSG(cell.a % 2 == 0 && cell.a < ell - 1, "BuildTower: a an even class in [0, ell - 1)");
    const ZK R(ell, K);
    const uint64_t N = IPow(ell, r);
    t.N = N;
    t.L = static_cast<uint32_t>(IPow(ell, r - 1));
    const GammaHLevelN G = GammaHLevelN::PrimeToPart(N, ell);
    ENGINE_CHECK_MSG(G.index_H() == t.L && G.contains_minus_one(), "BuildTower: H_r = mu_{ell-1} has index ell^{r-1}, -1 in H_r");
    t.m = G.m();
    t.D = t.m;
    t.teich_gen = static_cast<uint32_t>(G.generators().empty() ? 1 : G.generators()[0]);
    // G1 at r = 1: def:cosetrepsN's table at N = ell prime, transcribed (p10c_orient.py), and Gamma0LevelN's
    if (r == 1) {
        const hecke::level::Gamma0LevelN lvl0(N);
        for (uint32_t J = 1; J <= t.m; ++J) {
            const Mat22 w = G.rep(J);
            const Mat22 want = J <= ell ? Mat22{1, 0, static_cast<int64_t>(J) - 1, 1} : Mat22{0, -1, 1, 0};
            if (!(w == want) || !(w == lvl0.rep(J))) t.table_r1_ok = false;
        }
    }
    const int64_t e = opt.have_tame_override ? opt.tame_override : static_cast<int64_t>(cell.a) - 2;
    const std::vector<uint32_t> chi = TameCharacterTable(N, R, e);
    const MonomialChi MS = MonomialChiOf(G, Mat22::S(), chi), MU = MonomialChiOf(G, Mat22::ST(), chi);
    std::vector<SparseVec> gS, gU;
    RelGeneratorsChi(MS, MU, R, 2, 3, gS, gU);
    std::vector<SparseVec> gens = gS;
    gens.insert(gens.end(), gU.begin(), gU.end());
    t.setup_s = Since(t0);
    const ChainCokernel Q(t.D, gens, R, opt.byte_cap);
    t.elim = Q.stats();
    t.elim_s = t.elim.seconds;
    t.h = Q.h();
    if (opt.fl_crosscheck) {
        auto modl = [&](const std::vector<SparseVec>& in) {
            std::vector<SparseVec> out;
            for (const SparseVec& v : in) {
                SparseVec w;
                for (size_t k = 0; k < v.col.size(); ++k)
                    if (v.val[k] % ell) { w.col.push_back(v.col[k]); w.val.push_back(v.val[k] % ell); }
                if (!w.col.empty()) out.push_back(std::move(w));
            }
            return out;
        };
        const hecke::level::SparseCokernel F(t.D, modl(gS), modl(gU), ell);
        t.fl_pivots_ok = F.pivots() == Q.pivots();
        if (t.fl_pivots_ok) {
            for (uint32_t c : Q.pivots()) {
                const SparseVec& ft = F.tail_of(c);
                const SparseVec& qt = Q.tail(c);
                SparseVec qm;
                for (size_t k = 0; k < qt.col.size(); ++k)
                    if (qt.val[k] % ell) { qm.col.push_back(qt.col[k]); qm.val.push_back(qt.val[k] % ell); }
                SparseVec fm;
                for (size_t k = 0; k < ft.col.size(); ++k) {
                    fm.col.push_back(static_cast<uint32_t>(Q.free_index(ft.col[k])));
                    fm.val.push_back(ft.val[k] % ell);
                }
                if (qm.col != fm.col || qm.val != fm.val) { t.fl_tails_ok = false; break; }
            }
        } else {
            t.fl_tails_ok = false;
        }
    }
    // ---- operators ----
    const auto t1 = std::chrono::steady_clock::now();
    t.primes = opt.primes;
    if (t.primes.empty())
        for (uint64_t p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull})
            if (p != ell) t.primes.push_back(p);
    for (uint64_t p : t.primes) t.s_of_p.push_back(LogU(p, ell, r));
    std::map<std::string, std::unique_ptr<ChiOperator>> ops;
    for (uint64_t p : t.primes) ops["T" + std::to_string(p)] = std::make_unique<ChiHeilbronn>(G, p, chi, R);
    ops["U"] = std::make_unique<ChiHeilbronn>(G, ell, chi, R);
    ops["g"] = std::make_unique<ChiDiamond>(G, 1 + ell, chi, R);
    for (uint64_t p : t.primes) ops["<" + std::to_string(p) + ">"] = std::make_unique<ChiDiamond>(G, p % N, chi, R);
    for (uint64_t hgen : G.generators()) ops["<h" + std::to_string(hgen) + ">"] = std::make_unique<ChiDiamond>(G, hgen, chi, R);
    std::map<std::string, QuotientOperator> q;
    for (const auto& kv : ops) {
        if (opt.rel_escapes) t.rel_escapes[kv.first] = RelEscapesChi(Q, gens, *kv.second);
        q[kv.first] = OnQuotient(Q, *kv.second);
        t.nnz[kv.first] = q[kv.first].nnz();
    }
    // G3: commutation and the diamond relations on random vectors (fixed seed)
    {
        std::mt19937_64 rng(opt.seed ^ 0x6733ull);
        bool comm = true, dp = true, dh = true;
        std::vector<uint32_t> x(t.h), y1, y2, z1, z2;
        auto apply = [&](const std::string& n, const std::vector<uint32_t>& in, std::vector<uint32_t>& out) { q.at(n).apply(R, in, out); };
        for (int rep = 0; rep < 3; ++rep) {
            for (auto& v : x) v = static_cast<uint32_t>(rng() % R.Q);
            const std::vector<std::pair<std::string, std::string>> pairs = {
                {"U", "T2"}, {"U", "T3"}, {"T2", "T3"}, {"g", "U"}, {"g", "T2"}, {"g", "T3"}};
            for (const auto& pr : pairs) {
                if (!q.count(pr.first) || !q.count(pr.second)) continue;
                apply(pr.second, x, y1); apply(pr.first, y1, z1);
                apply(pr.first, x, y2); apply(pr.second, y2, z2);
                if (z1 != z2) comm = false;
            }
            for (size_t i = 0; i < t.primes.size(); ++i) {
                apply("<" + std::to_string(t.primes[i]) + ">", x, y1);
                y2 = x;
                for (uint32_t k = 0; k < t.s_of_p[i]; ++k) { apply("g", y2, z2); y2 = z2; }
                if (y1 != y2) dp = false;
            }
            for (uint64_t hgen : G.generators()) {
                apply("<h" + std::to_string(hgen) + ">", x, y1);
                if (y1 != x) dh = false;
            }
        }
        t.commute_ok = comm;
        t.diamond_p_ok = dp;
        t.diamond_H_ok = dh;
    }
    t.ops_s = Since(t1);
    // ---- G5: the ordinary part ----
    const auto t2 = std::chrono::steady_clock::now();
    const uint32_t Rp = opt.R_pred ? opt.R_pred : 1;
    const uint32_t s = std::min<uint32_t>(t.h, Rp + 16);
    t.krylov_s = s;
    const uint32_t phiL = t.L == 1 ? 1 : t.L - t.L / ell;
    const uint32_t T0 = K * phiL + 2 * r;
    std::vector<uint32_t> B(static_cast<size_t>(t.h) * s);
    {
        std::mt19937_64 rng(opt.seed);
        for (auto& v : B) v = static_cast<uint32_t>(rng() % R.Q);
    }
    KrylovOut kr = Krylov(q.at("U"), t.h, R, std::move(B), s, T0, opt.krylov_fixed_T);
    t.krylov_s_wall = Since(t2);
    t.krylov_T = kr.T;
    t.krylov_ranks = kr.ranks;
    t.krylov_stable = kr.stable;
    t.rank = kr.rank;
    t.E_free = kr.free;
    t.E_Ustable = kr.ustable;
    t.U_invertible = kr.uinv;
    if (opt.krylov_fixed_T >= 0 || !(kr.free && kr.ustable && kr.uinv) || kr.rank == 0) {
        t.seconds = Since(t0);
        t.peak_rss_mib = oc_util::PeakRssMiB();
        return t;
    }
    const Subspace& E = kr.E;
    // G5': the deterministic ordinary part at r <= 2
    if (opt.deterministic) {
        t.g5prime_run = true;
        DM Ud(t.h, t.h);
        const QuotientOperator& qu = q.at("U");
        for (uint32_t f = 0; f < t.h; ++f)
            for (uint64_t k = qu.start[f]; k < qu.start[f + 1]; ++k) Ud.at(qu.ent[k].first, f) = qu.ent[k].second;
        uint64_t T = 1;
        while (T < static_cast<uint64_t>(K) * t.h) T *= 2;
        const DM e = MatPow(Ud, T, R);
        const std::vector<uint32_t> ind = IndepColsModEll(e, ell);
        bool ok = ind.size() == E.dim();
        if (ok) {
            const Subspace F(Cols(e, ind), R);
            ok = F.pivot_rows() == E.pivot_rows() && F.canonical() == E.canonical();
        }
        t.g5prime_ok = ok;
    }
    // operators on E
    for (const auto& kv : q) {
        DM M;
        if (!E.restrict(ApplyOp(kv.second, E.basis(), R), M)) { t.onE.clear(); t.seconds = Since(t0); return t; }
        t.onE[kv.first] = M;
    }
    if (t.rank <= 64) {
        for (const char* n : {"T2", "T3", "U"}) if (t.onE.count(n)) t.charpolyE[n] = Charpoly(t.onE.at(n), R);
        for (const char* n : {"T2", "T3"}) {
            if (!t.onE.count(n)) continue;
            std::vector<uint32_t> ev;
            DM A = t.onE.at(n);
            for (uint32_t x = 0; x < ell; ++x) {
                const DM P = MatPow(ScalarSub(A, x, R), t.rank, R);
                const uint32_t mult = t.rank - RankModEll(P, ell);
                for (uint32_t k = 0; k < mult; ++k) ev.push_back(x);
            }
            t.eigsE[n] = ev;
        }
    }
    // ---- the split by the residual Eisenstein value of T_2 (BF') ----
    const uint64_t eexp = static_cast<uint64_t>(((e % static_cast<int64_t>(ell - 1)) + (ell - 1)) % (ell - 1));
    auto eis_res = [&](uint64_t p) {   // 1 + omega^e(p) p mod ell
        uint64_t w = 1;
        for (uint64_t k = 0; k < eexp; ++k) w = w * (p % ell) % ell;
        return static_cast<uint32_t>((1 + w * (p % ell)) % ell);
    };
    const DM& A2 = t.onE.at("T2");
    uint64_t NT = 1;
    while (NT < static_cast<uint64_t>(K) * t.rank) NT *= 2;
    const DM Bsplit = MatPow(ScalarSub(A2, eis_res(2), R), NT, R);
    const KerIm ki = KernelImage(Bsplit, R);
    if (!ki.ok) {
        t.split_ok = false;
        t.seconds = Since(t0);
        return t;
    }
    {
        DM both(t.rank, ki.ker.c + ki.im.c);
        for (uint32_t i = 0; i < t.rank; ++i) {
            for (uint32_t j = 0; j < ki.ker.c; ++j) both.at(i, j) = ki.ker.at(i, j);
            for (uint32_t j = 0; j < ki.im.c; ++j) both.at(i, ki.ker.c + j) = ki.im.at(i, j);
        }
        t.split_ok = both.c == t.rank && RankModEll(both, ell) == t.rank;
    }
    const std::vector<std::pair<std::string, const DM*>> partdefs = {{"eis", &ki.ker}, {"cusp", &ki.im}};
    for (const auto& pd : partdefs) {
        if (pd.second->c == 0) continue;
        TowerPart P;
        P.kind = pd.first;
        P.rank = pd.second->c;
        const Subspace S(*pd.second, R);
        std::map<std::string, DM> onP;
        bool ok = true;
        for (const auto& kv : t.onE) {
            DM M;
            if (!S.restrict(MatMul(kv.second, S.basis(), R), M)) { ok = false; break; }
            onP[kv.first] = M;
        }
        if (!ok) { t.split_ok = false; continue; }
        const uint32_t k = P.rank;
        // G6: generators over (Z/ell^K)[G_r] (Nakayama)
        const DM& gp = onP.at("g");
        const DM gm1 = ScalarSub(gp, 1 % static_cast<uint32_t>(R.Q), R);
        EchelonModEll ech;
        ech.ell = ell;
        ech.n = k;
        for (uint32_t j = 0; j < k; ++j) ech.add(ColModEll(gm1, j, ell));
        const uint32_t rk = static_cast<uint32_t>(ech.basis.size());
        P.lambda_rank = k - rk;
        std::vector<uint32_t> gensP;
        for (uint32_t i = 0; i < k && gensP.size() < P.lambda_rank; ++i) {
            std::vector<uint32_t> ei(k, 0);
            ei[i] = 1;
            if (ech.add(ei)) gensP.push_back(i);
        }
        const uint32_t L = t.L;
        P.basis_ok = P.lambda_rank * L == k && gensP.size() == P.lambda_rank;
        if (P.basis_ok) {
            DM Bm(k, k);
            for (size_t gi = 0; gi < gensP.size(); ++gi) {
                std::vector<uint32_t> v(k, 0);
                v[gensP[gi]] = static_cast<uint32_t>(1 % R.Q);
                for (uint32_t j = 0; j < L; ++j) {
                    for (uint32_t i = 0; i < k; ++i) Bm.at(i, static_cast<uint32_t>(gi) * L + j) = v[i];
                    std::vector<uint32_t> w(k, 0);
                    for (uint32_t i = 0; i < k; ++i) {
                        uint64_t sacc = 0;
                        for (uint32_t l = 0; l < k; ++l) sacc = (sacc + static_cast<uint64_t>(gp.at(i, l)) * v[l]) % R.Q;
                        w[i] = static_cast<uint32_t>(sacc);
                    }
                    v = w;
                }
            }
            P.basis_ok = RankModEll(Bm, ell) == k;
            if (P.basis_ok) {
                const DM Binv = Inverse(Bm, R);
                P.scalar_ok = true;
                const uint32_t ng = P.lambda_rank;
                for (const auto& kv : onP) {
                    const DM C = MatMul(Binv, MatMul(kv.second, Bm, R), R);
                    // c_{k1,k2}(g): C[k1 L + i][k2 L] ; consistency: C[k1 L + i][k2 L + j] = c_{k1,k2}[(i - j) mod L]
                    GR c11(L);
                    for (uint32_t i = 0; i < L; ++i) c11[i] = C.at(i, 0);
                    bool sc = true;
                    for (uint32_t k1 = 0; k1 < ng && sc; ++k1)
                        for (uint32_t k2 = 0; k2 < ng && sc; ++k2)
                            for (uint32_t i = 0; i < L && sc; ++i)
                                for (uint32_t j = 0; j < L && sc; ++j) {
                                    const uint32_t want = k1 == k2 ? c11[(i + L - j) % L] : 0;
                                    if (C.at(k1 * L + i, k2 * L + j) != want) sc = false;
                                }
                    if (!sc) P.scalar_ok = false;
                    if (kv.first == "U" || (kv.first.size() >= 2 && kv.first[0] == 'T')) {
                        P.scalars[kv.first] = c11;
                        P.weierstrass[kv.first] = Weierstrass(c11, R);
                    }
                }
                for (uint64_t p : {2ull, 3ull, 5ull, 7ull}) {
                    if (p == ell) { P.residual.push_back(UINT32_MAX); continue; }
                    const GR& c = P.scalars.at("T" + std::to_string(p));
                    uint64_t aug = 0;
                    for (uint32_t x : c) aug += x;
                    P.residual.push_back(static_cast<uint32_t>(aug % ell));
                }
            }
        }
        if (k <= 64)
            for (const char* n : {"T2", "T3", "U"}) if (onP.count(n)) P.charpoly[n] = Charpoly(onP.at(n), R);
        t.parts.push_back(std::move(P));
    }
    t.seconds = Since(t0);
    t.peak_rss_mib = oc_util::PeakRssMiB();
    return t;
}

// ============================================================================
// the Gamma_1 oracle
// ============================================================================
Gamma1Oracle BuildGamma1Oracle(uint32_t ell, uint32_t r, uint32_t K, const std::vector<uint32_t>& classes,
                               const std::map<uint32_t, uint32_t>& R_pred) {
    const auto t0 = std::chrono::steady_clock::now();
    Gamma1Oracle o;
    o.ell = ell;
    o.r = r;
    o.K = K;
    const ZK R(ell, K);
    const uint64_t N = IPow(ell, r);
    const GammaHLevelN G(N, {1});
    ENGINE_CHECK_MSG(G.H_elements().size() == 1 && !G.contains_minus_one(), "Gamma1 oracle: H = {1}");
    o.m = G.m();
    const std::vector<uint32_t> triv = TameCharacterTable(N, R, 0, true);
    const MonomialChi MS = MonomialChiOf(G, Mat22::S(), triv), MU = MonomialChiOf(G, Mat22::ST(), triv);
    std::vector<SparseVec> gS, gU;
    RelGeneratorsChi(MS, MU, R, 4, 6, gS, gU);
    std::vector<SparseVec> gens = gS;
    gens.insert(gens.end(), gU.begin(), gU.end());
    const ChainCokernel Q(o.m, gens, R);
    o.h = Q.h();
    const QuotientOperator qU = OnQuotient(Q, ChiHeilbronn(G, ell, triv, R));
    const QuotientOperator qT2 = OnQuotient(Q, ChiHeilbronn(G, 2, triv, R));
    const QuotientOperator qT3 = OnQuotient(Q, ChiHeilbronn(G, 3, triv, R));
    // H_r = mu_{ell-1} mod ell^r: the Teichmueller lifts of 1 .. ell - 1
    const ZK Rr(ell, r);
    std::vector<uint64_t> Hr;
    for (uint64_t x = 1; x < ell; ++x) Hr.push_back(Rr.teich(x));
    std::vector<QuotientOperator> dia;
    for (uint64_t hh : Hr) dia.push_back(OnQuotient(Q, ChiDiamond(G, hh, triv, R)));
    std::vector<uint32_t> tr(Hr.size(), 0);
    for (size_t i = 0; i < Hr.size(); ++i) {
        uint64_t sacc = 0;
        for (uint32_t f = 0; f < o.h; ++f)
            for (uint64_t k = dia[i].start[f]; k < dia[i].start[f + 1]; ++k)
                if (dia[i].ent[k].first == f) sacc += dia[i].ent[k].second;
        tr[i] = static_cast<uint32_t>(sacc % R.Q);
    }
    const uint32_t inv_lm1 = R.inv(ell - 1);
    for (uint32_t a : classes) {
        Gamma1OracleClass C;
        C.a = a;
        const std::vector<uint32_t> chi = TameCharacterTable(N, R, static_cast<int64_t>(a) - 2);
        std::vector<uint32_t> w(Hr.size());   // chi(h)^{-1} / (ell - 1)
        for (size_t i = 0; i < Hr.size(); ++i) w[i] = R.mul(inv_lm1, R.inv(chi[Hr[i]]));
        uint64_t te = 0;
        for (size_t i = 0; i < Hr.size(); ++i) te = (te + static_cast<uint64_t>(w[i]) * tr[i]) % R.Q;
        C.trace_e = static_cast<uint32_t>(te);
        const auto it = R_pred.find(a);
        const uint32_t s = std::min<uint32_t>(o.h, (it == R_pred.end() ? 1 : it->second) + 16);
        std::vector<uint32_t> B(static_cast<size_t>(o.h) * s), Y, acc(static_cast<size_t>(o.h) * s, 0);
        std::mt19937_64 rng(0x6731ull + a);
        for (auto& v : B) v = static_cast<uint32_t>(rng() % R.Q);
        for (size_t i = 0; i < Hr.size(); ++i) {
            dia[i].apply_block(R, B, s, Y);
            for (size_t k = 0; k < acc.size(); ++k) acc[k] = R.add(acc[k], R.mul(w[i], Y[k]));
        }
        const uint32_t L = static_cast<uint32_t>(IPow(ell, r - 1));
        const uint32_t phiL = L == 1 ? 1 : L - L / ell;
        KrylovOut kr = Krylov(qU, o.h, R, std::move(acc), s, K * phiL + 2 * r, -1);
        C.rank = kr.rank;
        C.certified = kr.free && kr.ustable && kr.uinv;
        if (C.certified) {
            // in the chi-eigenspace: <h> = chi(h) on E for every h in H_r
            for (size_t i = 0; i < Hr.size() && C.certified; ++i) {
                DM M;
                if (!kr.E.restrict(ApplyOp(dia[i], kr.E.basis(), R), M)) { C.certified = false; break; }
                const DM want = ScalarSub(DM(M.r, M.c), R.neg(chi[Hr[i]]), R);
                if (!(M == want)) C.certified = false;
            }
            DM M;
            if (C.certified && kr.rank <= 64) {
                if (kr.E.restrict(ApplyOp(qT2, kr.E.basis(), R), M)) C.charpoly["T2"] = Charpoly(M, R);
                if (kr.E.restrict(ApplyOp(qT3, kr.E.basis(), R), M)) C.charpoly["T3"] = Charpoly(M, R);
                C.charpoly["U"] = Charpoly(kr.UonE, R);
            }
        }
        o.classes.push_back(std::move(C));
    }
    o.seconds = Since(t0);
    return o;
}

// ============================================================================
// the record
// ============================================================================
std::vector<std::string> TowerRecordLines(const TowerReading& t, const std::map<std::string, std::string>& extra) {
    std::vector<std::string> out;
    auto vec = [](const std::vector<uint32_t>& v) {
        std::ostringstream os;
        for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
        return os.str();
    };
    for (const TowerPart& P : t.parts) {
        std::ostringstream os;
        os << "HECKE_LAMBDA_TOWER ell=" << t.cell.ell << " a=" << t.cell.a << " r=" << t.cell.r << " K=" << t.cell.K
           << " component=" << P.kind << " rank=" << P.rank << " h=" << t.h << " krylov_T=" << t.krylov_T << " A_p=[";
        bool first = true;
        for (uint64_t p : t.primes) {
            const auto it = P.weierstrass.find("T" + std::to_string(p));
            if (it == P.weierstrass.end()) continue;
            os << (first ? "" : ";") << p << ":" << vec(it->second);
            first = false;
        }
        os << "] alpha=[";
        const auto ia = P.weierstrass.find("U");
        if (ia != P.weierstrass.end()) os << vec(ia->second);
        os << "]";
        for (const auto& kv : extra) os << " " << kv.first << "=" << kv.second;
        const std::string line = os.str();
        const uint64_t d = engine::FnvBytes(engine::kFnvOffset64, line);
        std::ostringstream hx;
        hx << std::hex << d;
        out.push_back(line + " digest=0x" + hx.str());
    }
    return out;
}

// (slot 23, G-EVT23) the default-arm set-up of HeckeLambdaTowerValidation -- moved verbatim (DofA, RPred, the options).
uint32_t TowerDofA(uint32_t ell, uint32_t a) { return (ell == 11 && a == 2) || (ell == 13 && a == 0) ? 1 : 0; }
uint32_t TowerRankPred(uint32_t ell, uint32_t a, uint32_t r) {
    uint64_t q = 1;
    for (uint32_t e = 1; e < r; ++e) q *= ell;
    return static_cast<uint32_t>(q) * (2 * TowerDofA(ell, a) + 1);
}
TowerOptions DefaultTowerOptions(const TowerCell& cell) {
    TowerOptions o;
    o.R_pred = TowerRankPred(cell.ell, cell.a, cell.r);
    o.deterministic = cell.r <= 2;
    return o;
}

} // namespace tower
} // namespace hecke_ext
