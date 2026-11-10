// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZTor -- slot-25 HZ-TOR (see the header).

#include "HeckeZTor.h"

#include "Descent/DescentModuleExport.h"
#include "Descent/DescentModuleExportZ.h"
#include "../Hecke/FpPoly.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <sstream>

namespace heckez {
namespace tor {

namespace {

using heckez::local::Poly;
using FP = hecke::FpPoly64;

uint64_t PowU(uint64_t b, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) x *= b;
    return x;
}

std::string VecText(const std::vector<uint64_t>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}
std::string VecText32(const std::vector<uint32_t>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]";
    return o.str();
}

// ---- the full module H^1 / ell^K: D x D endomorphisms, row i mod ell^{exp_i} ---------------------------------------------------
struct Full {
    const ExtensionInput* X = nullptr;
    Zlk R;                         // Z/ell^K
    std::vector<uint64_t> mod;     // ell^{exp_i}
    uint32_t D = 0;
    explicit Full(const ExtensionInput& x) : X(&x), R(Zlk::Make(x.ell, x.K)), D(x.D()) {
        for (uint32_t i = 0; i < D; ++i) mod.push_back(PowU(x.ell, x.exp(i)));
    }
    Mat canon(Mat A) const {
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) A[size_t(i) * D + j] %= mod[i];
        return A;
    }
    Mat mul(const Mat& A, const Mat& B) const { return canon(heckez::local::MatMul(R, A, B, D)); }
    Mat eye() const { return heckez::local::MatIdentity(D); }
    Mat lin(const Mat& A, uint64_t a, const Mat& B, uint64_t b) const {   // a A + b B
        Mat C(A.size());
        for (size_t k = 0; k < A.size(); ++k) C[k] = R.add(R.mul(a % R.q, A[k]), R.mul(b % R.q, B[k]));
        return canon(C);
    }
    // the injective encoding (+)_i Z/ell^{exp_i} -> Z/ell^K, entry (i, j) times ell^{K - exp_i}
    std::vector<uint64_t> enc(const Mat& A) const {
        std::vector<uint64_t> v(A.size());
        for (uint32_t i = 0; i < D; ++i) {
            const uint64_t s = R.pw(X->K - X->exp(i));
            for (uint32_t j = 0; j < D; ++j) v[size_t(i) * D + j] = R.mul(A[size_t(i) * D + j] % mod[i], s);
        }
        return v;
    }
};

Mat ModEll(const Mat& A, uint64_t ell) {
    Mat B(A);
    for (auto& x : B) x %= ell;
    return B;
}

uint32_t RankModEll(uint64_t ell, const Mat& A, size_t r, size_t c) {
    if (r == 0 || c == 0) return 0;
    const Zlk F = Zlk::Make(ell, 1);
    return static_cast<uint32_t>(heckez::local::SmithValuations(F, ModEll(A, ell), r, c).size());
}

FP ToFP(uint64_t ell, const Poly& p) { return FP(ell, p); }
std::vector<uint64_t> FromFP(const FP& f) { return f.coeffs; }

Poly CharpolyModEll(uint64_t ell, const Mat& A, size_t D) {
    if (D == 0) return Poly{1};
    const Zlk F = Zlk::Make(ell, 1);
    return heckez::local::Charpoly(F, ModEll(A, ell), D);
}

Mat Sub(const Mat& A, size_t D, size_t r0, size_t c0, size_t nr, size_t nc) {
    Mat B(nr * nc);
    for (size_t i = 0; i < nr; ++i)
        for (size_t j = 0; j < nc; ++j) B[i * nc + j] = A[(r0 + i) * D + c0 + j];
    return B;
}

// p(A) on the full module, p with coefficients in [0, ell) (lifted)
Mat EvalFull(const Full& F, const std::vector<uint64_t>& p, const Mat& A) {
    Mat Y(F.D * size_t(F.D), 0);
    for (size_t k = p.size(); k-- > 0;) {
        Y = F.mul(Y, A);
        for (uint32_t i = 0; i < F.D; ++i) Y[size_t(i) * F.D + i] = F.R.add(Y[size_t(i) * F.D + i], p[k] % F.R.q);
        Y = F.canon(Y);
    }
    return Y;
}

// Newton lift of an idempotent mod ell to Z/ell^K (E <- 3E^2 - 2E^3)
Mat LiftIdempotent(const Full& F, Mat E) {
    for (uint32_t it = 0; it < 2 * 64; ++it) {
        const Mat E2 = F.mul(E, E);
        if (E2 == E) return E;
        const Mat E3 = F.mul(E2, E);
        E = F.lin(E2, 3, E3, F.R.q - 2);
    }
    ENGINE_CHECK_MSG(false, "HeckeZTor: the idempotent did not converge");
    return E;
}

// a basis of im(E_L) (E_L idempotent on the free (Z/ell^K)^r): columns of E_L independent mod ell; and the left inverse on the
// image (rows Rsel with V[Rsel, :] invertible mod ell)
struct ImageBasis {
    uint32_t r_m = 0;
    Mat V;                   // r x r_m
    std::vector<size_t> rows;
    Mat Winv;                // r_m x r_m, the inverse of V[rows, :]
};

Mat InvertUnitPivot(const Zlk& R, Mat A, size_t n) {
    Mat I = heckez::local::MatIdentity(n);
    for (size_t k = 0; k < n; ++k) {
        size_t piv = n;
        for (size_t i = k; i < n; ++i)
            if (A[i * n + k] % R.ell != 0) { piv = i; break; }
        ENGINE_CHECK_MSG(piv < n, "HeckeZTor: block basis not invertible mod ell");
        if (piv != k)
            for (size_t j = 0; j < n; ++j) { std::swap(A[k * n + j], A[piv * n + j]); std::swap(I[k * n + j], I[piv * n + j]); }
        const uint64_t u = R.inv_unit(A[k * n + k]);
        for (size_t j = 0; j < n; ++j) { A[k * n + j] = R.mul(A[k * n + j], u); I[k * n + j] = R.mul(I[k * n + j], u); }
        for (size_t i = 0; i < n; ++i) {
            if (i == k || A[i * n + k] == 0) continue;
            const uint64_t f = A[i * n + k];
            for (size_t j = 0; j < n; ++j) {
                A[i * n + j] = R.sub(A[i * n + j], R.mul(f, A[k * n + j]));
                I[i * n + j] = R.sub(I[i * n + j], R.mul(f, I[k * n + j]));
            }
        }
    }
    return I;
}

ImageBasis BasisOfImage(const Zlk& R, const Mat& EL, size_t r) {
    ImageBasis B;
    // greedy: column j joins if independent mod ell of the chosen ones
    std::vector<size_t> cols;
    Mat acc;   // chosen columns mod ell, as rows (for the rank test)
    for (size_t j = 0; j < r; ++j) {
        Mat trial = acc;
        for (size_t i = 0; i < r; ++i) trial.push_back(EL[i * r + j] % R.ell);
        const size_t rows = cols.size() + 1;
        if (RankModEll(R.ell, trial, rows, r) == rows) { acc = trial; cols.push_back(j); }
    }
    B.r_m = static_cast<uint32_t>(cols.size());
    B.V.assign(r * size_t(B.r_m), 0);
    for (size_t i = 0; i < r; ++i)
        for (size_t c = 0; c < cols.size(); ++c) B.V[i * B.r_m + c] = EL[i * r + cols[c]];
    // rows: greedy on the rows of V mod ell
    Mat racc;
    for (size_t i = 0; i < r && B.rows.size() < B.r_m; ++i) {
        Mat trial = racc;
        for (size_t c = 0; c < B.r_m; ++c) trial.push_back(B.V[i * B.r_m + c] % R.ell);
        const size_t rows = B.rows.size() + 1;
        if (RankModEll(R.ell, trial, rows, B.r_m) == rows) { racc = trial; B.rows.push_back(i); }
    }
    ENGINE_CHECK(B.rows.size() == B.r_m);
    Mat W(size_t(B.r_m) * B.r_m);
    for (size_t a = 0; a < B.r_m; ++a)
        for (size_t c = 0; c < B.r_m; ++c) W[a * B.r_m + c] = B.V[B.rows[a] * B.r_m + c];
    B.Winv = B.r_m ? InvertUnitPivot(R, W, B.r_m) : Mat{};
    return B;
}

// coordinates of w (in im E_L) in the basis V; ENGINE_CHECK that V a = w
std::vector<uint64_t> Coords(const Zlk& R, const ImageBasis& B, const std::vector<uint64_t>& w, size_t r) {
    std::vector<uint64_t> a(B.r_m, 0);
    for (size_t i = 0; i < B.r_m; ++i) {
        uint64_t s = 0;
        for (size_t k = 0; k < B.r_m; ++k) s = R.add(s, R.mul(B.Winv[i * B.r_m + k], w[B.rows[k]]));
        a[i] = s;
    }
    for (size_t i = 0; i < r; ++i) {
        uint64_t s = 0;
        for (size_t k = 0; k < B.r_m; ++k) s = R.add(s, R.mul(B.V[i * B.r_m + k], a[k]));
        ENGINE_CHECK_MSG(s == w[i] % R.q, "HeckeZTor: a vector off the block image");
    }
    return a;
}

std::string PolyText(const std::vector<uint64_t>& p) { return VecText(p); }

}  // namespace

uint32_t ExtensionInput::e() const {
    uint32_t m = 0;
    for (uint32_t s : sigma) m = std::max(m, s);
    return m;
}

ExtensionInput FromModuleAtK(uint32_t N, uint32_t n, const integral_descent::modexport::ModuleAtK& M) {
    ExtensionInput X;
    X.N = N;
    X.n = n;
    X.ell = M.ell;
    X.K = M.K;
    X.r = M.free_rank;
    X.ps = M.ps;
    for (uint32_t i = 0; i < M.torsion_count; ++i) X.sigma.push_back(M.exps[i]);
    X.A = M.ops;
    return X;
}

ExtensionInput FromModuleZ(const integral_descent::modexport::ModuleZ& Z, size_t l) {
    ENGINE_CHECK_MSG(Z.ok && l < Z.local.size(), "HeckeZTor::FromModuleZ: index");
    // (audit M1, closed by EXZ2-P3) M and F come from ONE basis-tagged accessor, ModuleZ::free_view: the local pair (step 1) or the
    // global one (step 2: M_global = M_local C_ell, F mod ell^K); the verdicts are invariants of the presentation (arm J here, arm (G)
    // of HeckeZExportZ2Validation)
    const auto& P = Z.local[l];
    ExtensionInput X;
    X.N = Z.N;
    X.n = Z.n;
    X.ell = P.ell;
    X.K = P.K;
    X.r = Z.free_rank;
    X.ps = Z.ps;
    for (uint32_t i = 0; i < P.t; ++i) X.sigma.push_back(Z.tor_sigma[P.t0 + i]);
    const uint32_t t = P.t, D = t + X.r, tg = Z.torsion_rank();
    for (size_t o = 0; o < Z.ps.size(); ++o) {
        Mat A(size_t(D) * D, 0);
        const integral_descent::modexport::FreeView v = Z.free_view(o, l);
        const auto& M = v.M_block;
        const auto& F = v.F_block;
        for (uint32_t i = 0; i < t; ++i)
            for (uint32_t j = 0; j < t; ++j) A[size_t(i) * D + j] = Z.tau[o][size_t(P.t0 + i) * tg + P.t0 + j];
        for (uint32_t i = 0; i < t; ++i)
            for (uint32_t j = 0; j < X.r; ++j) A[size_t(i) * D + t + j] = M[size_t(i) * X.r + j];
        for (uint32_t i = 0; i < X.r; ++i)
            for (uint32_t j = 0; j < X.r; ++j) A[size_t(t + i) * D + t + j] = F[size_t(i) * X.r + j];
        X.A.push_back(std::move(A));
    }
    return X;
}

ExtensionInput Reduce(const ExtensionInput& X, uint32_t K) {
    ENGINE_CHECK_MSG(K >= 1 && K <= X.K, "HeckeZTor::Reduce: 1 <= K <= X.K");
    ExtensionInput Y = X;
    Y.K = K;
    for (auto& s : Y.sigma) s = std::min(s, K);
    const uint32_t D = Y.D();
    for (auto& A : Y.A)
        for (uint32_t i = 0; i < D; ++i) {
            const uint64_t m = PowU(Y.ell, Y.exp(i));
            for (uint32_t j = 0; j < D; ++j) A[size_t(i) * D + j] %= m;
        }
    return Y;
}

std::vector<uint32_t> UctPredict(const std::vector<uint32_t>& b, const std::vector<uint32_t>& t) {
    ENGINE_CHECK(b.size() >= 4 && t.size() >= 4);
    std::vector<uint32_t> h(4);
    for (uint32_t q = 0; q < 4; ++q) h[q] = b[q] + t[q] + (q + 1 < 4 ? t[q + 1] : t[2]);
    return h;
}

// ---- Smith with transforms over Z/ell^E ----------------------------------------------------------------------------------------
LinSolve SolveLinear(const Zlk& R, const std::vector<uint64_t>& M, size_t rows, size_t cols, const std::vector<uint64_t>& b) {
    ENGINE_CHECK(M.size() == rows * cols && b.size() == rows);
    std::vector<uint64_t> W(M);
    for (auto& x : W) x %= R.q;
    std::vector<uint64_t> U = heckez::local::MatIdentity(rows);
    std::vector<uint64_t> V = heckez::local::MatIdentity(cols);
    std::vector<uint32_t> s;
    const size_t m = std::min(rows, cols);
    for (size_t k = 0; k < m; ++k) {
        size_t pi = rows, pj = cols;
        uint32_t best = R.K;
        for (size_t i = k; i < rows && best > 0; ++i)
            for (size_t j = k; j < cols; ++j) {
                const uint64_t x = W[i * cols + j];
                if (x == 0) continue;
                const uint32_t v = R.val(x);
                if (v < best) { best = v; pi = i; pj = j; if (v == 0) break; }
            }
        if (pi == rows) break;
        if (pi != k) {
            for (size_t j = 0; j < cols; ++j) std::swap(W[k * cols + j], W[pi * cols + j]);
            for (size_t j = 0; j < rows; ++j) std::swap(U[k * rows + j], U[pi * rows + j]);
        }
        if (pj != k) {
            for (size_t i = 0; i < rows; ++i) std::swap(W[i * cols + k], W[i * cols + pj]);
            for (size_t i = 0; i < cols; ++i) std::swap(V[i * cols + k], V[i * cols + pj]);
        }
        const uint64_t piv = W[k * cols + k];
        const uint64_t lv = R.pw(best);
        const uint64_t unit = R.inv_unit(piv / lv);
        for (size_t j = 0; j < cols; ++j) W[k * cols + j] = R.mul(W[k * cols + j], unit);
        for (size_t j = 0; j < rows; ++j) U[k * rows + j] = R.mul(U[k * rows + j], unit);
        ENGINE_CHECK(W[k * cols + k] == lv);
        for (size_t i = 0; i < rows; ++i) {
            if (i == k) continue;
            const uint64_t x = W[i * cols + k];
            if (x == 0) continue;
            const uint64_t c = x / lv;   // exact: v(x) >= best
            for (size_t j = 0; j < cols; ++j) W[i * cols + j] = R.sub(W[i * cols + j], R.mul(c, W[k * cols + j]));
            for (size_t j = 0; j < rows; ++j) U[i * rows + j] = R.sub(U[i * rows + j], R.mul(c, U[k * rows + j]));
        }
        for (size_t j = 0; j < cols; ++j) {
            if (j == k) continue;
            const uint64_t x = W[k * cols + j];
            if (x == 0) continue;
            const uint64_t c = x / lv;
            for (size_t i = 0; i < rows; ++i) W[i * cols + j] = R.sub(W[i * cols + j], R.mul(c, W[i * cols + k]));
            for (size_t i = 0; i < cols; ++i) V[i * cols + j] = R.sub(V[i * cols + j], R.mul(c, V[i * cols + k]));
        }
        s.push_back(best);
    }
    const size_t rk = s.size();
    LinSolve out;
    for (size_t k = 0; k < rk; ++k) out.kernel_length += s[k];
    out.kernel_length += static_cast<uint32_t>((cols - rk) * R.K);
    // c = U b
    std::vector<uint64_t> c(rows, 0);
    for (size_t i = 0; i < rows; ++i) {
        uint64_t acc = 0;
        for (size_t j = 0; j < rows; ++j) acc = R.add(acc, R.mul(U[i * rows + j], b[j] % R.q));
        c[i] = acc;
    }
    size_t bad = rows;
    for (size_t k = 0; k < rows && bad == rows; ++k) {
        const uint32_t sk = k < rk ? s[k] : R.K;
        if (c[k] != 0 && R.val(c[k]) < sk) bad = k;
    }
    if (bad == rows) {
        out.solvable = true;
        std::vector<uint64_t> z(cols, 0);
        for (size_t k = 0; k < rk; ++k) z[k] = (c[k] / R.pw(s[k])) % R.q;
        out.x.assign(cols, 0);
        for (size_t i = 0; i < cols; ++i) {
            uint64_t acc = 0;
            for (size_t k = 0; k < cols; ++k) acc = R.add(acc, R.mul(V[i * cols + k], z[k]));
            out.x[i] = acc;
        }
    } else {
        const uint32_t sk = bad < rk ? s[bad] : R.K;
        const uint64_t f = (bad < rk) ? R.pw(R.K - sk) : 1;
        out.y.assign(rows, 0);
        for (size_t j = 0; j < rows; ++j) out.y[j] = R.mul(f, U[bad * rows + j]);
    }
    return out;
}

namespace {

bool CheckSolution(const Zlk& R, const std::vector<uint64_t>& M, size_t rows, size_t cols, const std::vector<uint64_t>& x,
                   const std::vector<uint64_t>& b) {
    for (size_t i = 0; i < rows; ++i) {
        uint64_t acc = 0;
        for (size_t j = 0; j < cols; ++j) acc = R.add(acc, R.mul(M[i * cols + j] % R.q, x[j] % R.q));
        if (acc != b[i] % R.q) return false;
    }
    return true;
}
bool CheckWitness(const Zlk& R, const std::vector<uint64_t>& M, size_t rows, size_t cols, const std::vector<uint64_t>& y,
                  const std::vector<uint64_t>& b) {
    for (size_t j = 0; j < cols; ++j) {
        uint64_t acc = 0;
        for (size_t i = 0; i < rows; ++i) acc = R.add(acc, R.mul(y[i] % R.q, M[i * cols + j] % R.q));
        if (acc != 0) return false;
    }
    uint64_t acc = 0;
    for (size_t i = 0; i < rows; ++i) acc = R.add(acc, R.mul(y[i] % R.q, b[i] % R.q));
    return acc != 0;
}

struct Block {
    Mat E;
    std::vector<std::vector<uint64_t>> residue;
};

}  // namespace

std::string BlockRead::text() const {
    std::ostringstream o;
    o << "block{res=";
    for (size_t i = 0; i < residue.size(); ++i) o << (i ? "|" : "") << PolyText(residue[i]);
    o << " r=" << r_m << " t=" << t_m << " len_tors=" << len_tors << " kind=" << kind << " split=" << split
      << " hom=" << hom_length << " J=" << J << " coinc=" << coincidence << " lean=" << lean;
    if (!h.empty()) o << " h=" << VecText(h);
    if (!y.empty()) o << " y=" << VecText(y);
    o << "}";
    return o.str();
}

std::string TorRead::text() const {
    std::ostringstream o;
    o << "HZTOR N=" << N << " n=" << n << " ell=" << ell << " P=" << VecText32(P) << " t_ell=" << t_ell << " status=" << status;
    if (!detail.empty()) o << " detail=" << detail;
    o << " theory=" << theory << " e=" << e << " K=" << K << " sigma=" << VecText32(sigma) << " r=" << r << " split=" << split
      << " b8=" << (b8_applies ? (b8_scalar ? "scalar" : "NOT_SCALAR") : "n/a");
    for (const auto& s : sep)
        o << " sep(T_" << s.p << ")={chiL=" << PolyText(s.chi_L) << " chiT=" << PolyText(s.chi_T) << " gcd=" << PolyText(s.gcd)
          << (s.separated ? " u=" + PolyText(s.u) + " v=" + PolyText(s.v) + (s.bezout_checked ? " ok" : " BAD") : "") << "}";
    for (const auto& b : blocks) o << " " << b.text();
    o << " K_J=" << K_J << " jprime=" << jprime_length << " A_len=" << algebra_length << " exact_J=" << exact_J;
    return o.str();
}

uint64_t TorRead::digest() const {
    const std::string s = text();
    return engine::FnvBytes(engine::kFnvOffset64, s);
}

TorRead ReadExtension(const ExtensionInput& X0, const TorOptions& opt, const ExtensionInput* XJ) {
    const auto t0 = std::chrono::steady_clock::now();
    TorRead T;
    T.N = X0.N;
    T.n = X0.n;
    T.ell = X0.ell;
    T.P = X0.ps;
    T.r = X0.r;
    T.e = X0.e();
    T.sigma = X0.sigma;
    T.status = "ok";
    if (T.e == 0) {
        T.status = "no_torsion";
        T.split = "no_torsion";
        return T;
    }
    ENGINE_CHECK_MSG(X0.K >= T.e, "HeckeZTor: the extension must be read at K >= sigma_max");
    // everything at K_s = e (B12(1)), or at the caller's K_read in [e, X0.K] (the B12(1) consistency arm)
    const uint32_t Ks = opt.K_read > T.e ? std::min(opt.K_read, X0.K) : T.e;
    const ExtensionInput X = Reduce(X0, Ks);
    T.K = X.K;
    const Full F(X);
    const uint32_t D = X.D(), t = X.t(), r = X.r;
    const uint64_t ell = X.ell;
    const Zlk& R = F.R;
    std::vector<Mat> A;
    for (const auto& a : X.A) A.push_back(F.canon(a));
    // structural checks: the torsion submodule is stable (free rows, torsion columns = 0), the operators commute
    for (size_t o = 0; o < A.size(); ++o) {
        for (uint32_t i = t; i < D; ++i)
            for (uint32_t j = 0; j < t; ++j)
                ENGINE_CHECK_MSG(A[o][size_t(i) * D + j] == 0, "HeckeZTor: a torsion image with a free coordinate");
        for (size_t o2 = 0; o2 < o; ++o2)
            ENGINE_CHECK_MSG(F.mul(A[o], A[o2]) == F.mul(A[o2], A[o]), "HeckeZTor: the operators do not commute");
    }

    // ---- K6, per operator -----------------------------------------------------------------------------------------------------
    for (size_t o = 0; o < A.size(); ++o) {
        Separation S;
        S.p = X.ps[o];
        S.chi_L = CharpolyModEll(ell, Sub(A[o], D, t, t, r, r), r);
        S.chi_T = CharpolyModEll(ell, Sub(A[o], D, 0, 0, t, t), t);
        FP u(ell), v(ell);
        const FP g = FP::extgcd(ToFP(ell, S.chi_L), ToFP(ell, S.chi_T), u, v);
        S.gcd = FromFP(g);
        S.separated = (g.degree() == 0);
        if (S.separated) {
            S.u = FromFP(u);
            S.v = FromFP(v);
            const FP one = u * ToFP(ell, S.chi_L) + v * ToFP(ell, S.chi_T);
            S.bezout_checked = (one == FP::one(ell));
        }
        T.sep.push_back(S);
    }

    // ---- B8 (level 1, ell in {2, 3}): T_p == 1 + p on H^1 / ell for every p != ell ---------------------------------------------
    T.b8_applies = (X.N == 1 && (ell == 2 || ell == 3));
    bool tell_in_P = false;
    if (T.b8_applies) {
        T.b8_scalar = true;
        for (size_t o = 0; o < A.size(); ++o) {
            if (X.ps[o] == ell) { tell_in_P = true; continue; }
            const Mat Abar = ModEll(A[o], ell);
            for (uint32_t i = 0; i < D; ++i)
                for (uint32_t j = 0; j < D; ++j) {
                    const uint64_t want = (i == j) ? (1 + X.ps[o]) % ell : 0;
                    if (Abar[size_t(i) * D + j] != want) T.b8_scalar = false;
                }
        }
    }

    // ---- the joint blocks ------------------------------------------------------------------------------------------------------
    std::vector<Block> blocks{Block{F.eye(), {}}};
    for (size_t o = 0; o < A.size(); ++o) {
        const Poly f = CharpolyModEll(ell, A[o], D);
        const FP fF = ToFP(ell, f);
        const auto fac = hecke::FactorPoly(fF);
        std::vector<std::pair<std::vector<uint64_t>, Mat>> idem;
        for (const auto& pf : fac.factors) {
            const FP pm = FP::pow(pf.pi, pf.multiplicity);
            const FP w = FP::divide_exact(fF, pm);
            FP a(ell), b(ell);
            const FP g = FP::extgcd(pm, w, a, b);
            ENGINE_CHECK(g == FP::one(ell));
            const FP e0 = FP::mod(b * w, fF);
            Mat E = EvalFull(F, e0.coeffs, A[o]);
            E = LiftIdempotent(F, E);
            idem.emplace_back(pf.pi.coeffs, E);
        }
        std::vector<Block> next;
        for (const auto& B : blocks)
            for (const auto& ie : idem) {
                Mat E = F.mul(B.E, ie.second);
                if (RankModEll(ell, E, D, D) == 0) continue;
                Block nb{E, B.residue};
                nb.residue.push_back(ie.first);
                next.push_back(std::move(nb));
            }
        blocks.swap(next);
    }
    {
        uint32_t tot = 0;
        for (const auto& B : blocks) tot += RankModEll(ell, B.E, D, D);
        ENGINE_CHECK_MSG(tot == D, "HeckeZTor: the blocks do not sum to the module");
    }

    // ---- per block: the split test and the Hom bound ---------------------------------------------------------------------------
    const uint32_t e = T.e;
    const Zlk Re = Zlk::Make(ell, Ks);   // = R (K = K_s)
    bool any_nonsplit = false, any_tsplit = false;
    for (const auto& B : blocks) {
        BlockRead br;
        br.residue = B.residue;
        const Mat ET = Sub(B.E, D, 0, 0, t, t);
        const Mat EL = Sub(B.E, D, t, t, r, r);
        br.r_m = RankModEll(ell, EL, r, r);
        br.t_m = RankModEll(ell, ET, t, t);
        {
            heckez::local::Span S(R, t);
            for (uint32_t j = 0; j < t; ++j) {
                std::vector<uint64_t> col(t);
                for (uint32_t i = 0; i < t; ++i) col[i] = R.mul(ET[size_t(i) * t + j] % F.mod[i], R.pw(X.K - X.exp(i)));
                S.insert(col);
            }
            br.len_tors = S.length();
        }
        br.lean = (br.r_m <= 6 && br.t_m <= 6) ? "in_cap" : "absent(cap)";
        if (br.r_m == 0 || br.t_m == 0) {
            br.kind = br.r_m == 0 ? "torsion_only" : "free_only";
            br.split = "split(separated)";
            br.J = "zero";
            br.coincidence = "none(separated)";
            T.blocks.push_back(br);
            continue;
        }
        br.kind = "common";
        const ImageBasis IB = BasisOfImage(R, EL, r);
        ENGINE_CHECK(IB.r_m == br.r_m);
        const uint32_t rm = br.r_m;
        const size_t nu = size_t(t) * rm, neq = A.size() * nu;
        std::vector<uint64_t> M(neq * nu, 0), rhs(neq, 0);
        for (size_t o = 0; o < A.size(); ++o) {
            const Mat Bp = Sub(A[o], D, t, t, r, r), Cp = Sub(A[o], D, 0, 0, t, t), bp = Sub(A[o], D, 0, t, t, r);
            // B' = coords of B V; beta' = E_T beta V
            Mat BV(size_t(r) * rm, 0);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t c = 0; c < rm; ++c) {
                    uint64_t s = 0;
                    for (uint32_t k = 0; k < r; ++k) s = R.add(s, R.mul(Bp[size_t(i) * r + k], IB.V[size_t(k) * rm + c]));
                    BV[size_t(i) * rm + c] = s;
                }
            Mat Bq(size_t(rm) * rm, 0);
            for (uint32_t c = 0; c < rm; ++c) {
                std::vector<uint64_t> w(r);
                for (uint32_t i = 0; i < r; ++i) w[i] = BV[size_t(i) * rm + c];
                const auto a = Coords(R, IB, w, r);
                for (uint32_t i = 0; i < rm; ++i) Bq[size_t(i) * rm + c] = a[i];
            }
            Mat bq(size_t(t) * rm, 0);
            for (uint32_t j = 0; j < t; ++j)
                for (uint32_t c = 0; c < rm; ++c) {
                    uint64_t s = 0;
                    for (uint32_t k = 0; k < t; ++k)
                        for (uint32_t i = 0; i < r; ++i)
                            s = R.add(s, R.mul(R.mul(ET[size_t(j) * t + k], bp[size_t(k) * r + i]), IB.V[size_t(i) * rm + c]));
                    bq[size_t(j) * rm + c] = s % F.mod[j];
                }
            // equations (j, i): sum_k h_jk B'_ki - sum_k C_jk h_ki = beta'_ji  (mod ell^{sigma_j}), scaled by ell^{e - sigma_j}
            for (uint32_t j = 0; j < t; ++j) {
                const uint64_t sc = Re.pw(Ks - X.sigma[j]);
                for (uint32_t i = 0; i < rm; ++i) {
                    const size_t row = o * nu + size_t(j) * rm + i;
                    for (uint32_t k = 0; k < rm; ++k) {
                        const size_t col = size_t(j) * rm + k;
                        M[row * nu + col] = Re.add(M[row * nu + col], Re.mul(sc, Bq[size_t(k) * rm + i] % Re.q));
                    }
                    for (uint32_t k = 0; k < t; ++k) {
                        const size_t col = size_t(k) * rm + i;
                        M[row * nu + col] = Re.sub(M[row * nu + col], Re.mul(sc, Cp[size_t(j) * t + k] % Re.q));
                    }
                    rhs[row] = Re.mul(sc, bq[size_t(j) * rm + i] % Re.q);
                }
            }
        }
        const LinSolve hom = SolveLinear(Re, M, neq, nu, std::vector<uint64_t>(neq, 0));
        uint32_t amb = 0;
        for (uint32_t j = 0; j < t; ++j) amb += rm * (Ks - X.sigma[j]);
        ENGINE_CHECK(hom.kernel_length >= amb);
        br.hom_length = hom.kernel_length - amb;
        const LinSolve sp = SolveLinear(Re, M, neq, nu, rhs);
        if (sp.solvable) {
            br.split = "split(T')";
            br.section_checked = CheckSolution(Re, M, neq, nu, sp.x, rhs);
            br.h = sp.x;
            for (uint32_t j = 0; j < t; ++j)
                for (uint32_t i = 0; i < rm; ++i) br.h[size_t(j) * rm + i] %= F.mod[j];
            any_tsplit = true;
        } else {
            br.split = "non_split";
            br.witness_checked = CheckWitness(Re, M, neq, nu, sp.y, rhs);
            br.y = sp.y;
            any_nonsplit = true;
        }
        br.J = br.hom_length == 0 ? "zero" : "bounds(<=" + std::to_string(ell) + "^" + std::to_string(br.hom_length) + ")";
        if (T.b8_applies && T.b8_scalar && tell_in_P)
            br.coincidence = "certified(B8)";
        else if (br.split == "non_split")
            br.coincidence = "certified(B14)";
        else
            br.coincidence = "not_certified(T'_block)";
        T.blocks.push_back(br);
    }
    std::sort(T.blocks.begin(), T.blocks.end(), [](const BlockRead& a, const BlockRead& b) { return a.text() < b.text(); });
    T.split = any_nonsplit ? "non_split" : (any_tsplit ? "split(T')" : "split(separated)");

    // ---- the J' read at K_J ----------------------------------------------------------------------------------------------------
    if (opt.jprime) {
        const ExtensionInput& XJsrc = XJ ? *XJ : X0;
        const uint32_t KJ = std::min(XJsrc.K, e + opt.K_J_extra);
        const ExtensionInput Y = Reduce(XJsrc, KJ);
        T.K_J = KJ;
        const Full G(Y);
        const size_t m = size_t(D) * D;
        heckez::local::Span Alg(G.R, m);
        std::vector<Mat> queue{G.eye()};
        Alg.insert(G.enc(queue[0]));
        std::vector<Mat> AY;
        for (const auto& a : Y.A) AY.push_back(G.canon(a));
        for (size_t qi = 0; qi < queue.size(); ++qi)
            for (const auto& Ap : AY) {
                Mat W = G.mul(queue[qi], Ap);
                if (Alg.insert(G.enc(W))) queue.push_back(std::move(W));
            }
        T.algebra_length = Alg.length();
        heckez::local::Span Gr(G.R, m);
        for (auto row : Alg.rows()) {
            for (uint32_t i = 0; i < t; ++i)
                for (uint32_t j = t; j < D; ++j) row[size_t(i) * D + j] = 0;
            Gr.insert(row);
        }
        T.jprime_length = static_cast<int32_t>(T.algebra_length - Gr.length());
    }
    T.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return T;
}

TorRead ReadTor(uint32_t N, uint32_t n, uint64_t ell, const std::vector<uint32_t>& P, const TorOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    TorRead T;
    T.N = N;
    T.n = n;
    T.ell = ell;
    T.P = heckez::local::ReadPrimes(N, ell, P, &T.t_ell);
    const auto E = integral_descent::modexport::ExportModule(N, n, static_cast<uint32_t>(ell), T.P);
    if (!E.ok) {
        T.status = "blocked(export)";
        T.split = "blocked(export)";
        T.detail = E.refusal;
        return T;
    }
    T.K_max = E.K_max;
    uint32_t e = 0;
    for (uint32_t s : E.sigma) e = std::max(e, s);
    if (e == 0) {
        T.status = "no_torsion";
        T.split = "no_torsion";
        T.r = E.free_rank;
        return T;
    }
    if (E.K_max < e) {
        T.status = "open(precision)";
        T.split = "open(precision)";
        return T;
    }
    const ExtensionInput X = FromModuleAtK(N, n, E.at(e));
    const uint32_t KJ = std::min(E.K_max, e + opt.K_J_extra);
    const ExtensionInput XJ = FromModuleAtK(N, n, E.at(KJ));
    TorRead R = ReadExtension(X, opt, &XJ);
    R.t_ell = T.t_ell;
    R.K_max = T.K_max;
    R.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

TorRead SplitFromModuleZ(const integral_descent::modexport::ModuleZ& Z, size_t l, const TorOptions& opt) {
    ENGINE_CHECK_MSG(Z.ok && l < Z.local.size(), "HeckeZTor::SplitFromModuleZ: index");
    if (Z.local[l].K == 0) {
        TorRead T;
        T.N = Z.N;
        T.n = Z.n;
        T.ell = Z.local[l].ell;
        T.P = Z.ps;
        T.status = "no_torsion";
        T.split = "no_torsion";
        return T;
    }
    return ReadExtension(FromModuleZ(Z, l), opt);
}

std::string ModuleZSplitWord(const integral_descent::modexport::ModuleZ& Z, const TorOptions& opt) {
    if (!Z.ok) return "blocked(export)";
    if (Z.torsion_rank() == 0) return "no_torsion";
    bool ns = false, ts = false;
    for (size_t l = 0; l < Z.local.size(); ++l) {
        if (Z.local[l].K == 0) continue;
        const TorRead T = SplitFromModuleZ(Z, l, opt);
        if (T.split == "non_split") ns = true;
        if (T.split == "split(T')") ts = true;
    }
    return ns ? "non_split" : (ts ? "split(T')" : "split(separated)");
}

}  // namespace tor
}  // namespace heckez
