// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZFreeBasisSatValidation: EXPORT-Z step 2, package EXZ2-P1b (HeckeZFreeBasisSatValidation.h).

#include "HeckeZFreeBasisSatValidation.h"

#include "DescentAssembly.h"
#include "DescentFreeBasisZ.h"
#include "IsoTreeCarrierZ.h"

#include "../../../ModArithmetic/EngineCheck.h"

#include <chrono>
#include <cstdio>
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim: the (M) arm's sums baseline (fix round 1)
#endif
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fz = integral_descent::freez;
using fz::BigInt;
using clk = std::chrono::steady_clock;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
void TrimHeap() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}
std::string Cell(uint32_t N, uint32_t n) { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }

using DenseCols = std::vector<std::vector<BigInt>>;
std::vector<std::vector<BigInt>> DenseOf(const fz::SparseZ& A) {
    std::vector<std::vector<BigInt>> M(A.rows, std::vector<BigInt>(A.cols, BigInt(0)));
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) M[i][A.col[k]] = A.val[k];
    return M;
}
// the validation's own dense pairing and Bareiss determinant
BigInt OwnDet(const DenseCols& Phi, const DenseCols& B) {
    const size_t r = B.size();
    std::vector<BigInt> G(r * r, BigInt(0));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < r; ++j) {
            BigInt s(0);
            for (size_t t = 0; t < B[j].size(); ++t) s += Phi[i][t] * B[j][t];
            G[i * r + j] = s;
        }
    if (r == 0) return BigInt(1);
    BigInt prev(1);
    int sign = 1;
    for (size_t k = 0; k + 1 < r; ++k) {
        if (G[k * r + k].is_zero()) {
            size_t p = k + 1;
            while (p < r && G[p * r + k].is_zero()) ++p;
            if (p == r) return BigInt(0);
            for (size_t j = 0; j < r; ++j) std::swap(G[k * r + j], G[p * r + j]);
            sign = -sign;
        }
        for (size_t i = k + 1; i < r; ++i)
            for (size_t j = k + 1; j < r; ++j) G[i * r + j] = (G[i * r + j] * G[k * r + k] - G[i * r + k] * G[k * r + j]) / prev;
        prev = G[k * r + k];
    }
    return sign < 0 ? -G[r * r - 1] : G[r * r - 1];
}
bool OwnCocycles(const std::vector<std::vector<BigInt>>& D1, const DenseCols& B) {
    for (const auto& b : B)
        for (const auto& row : D1) {
            BigInt s(0);
            for (size_t t = 0; t < row.size(); ++t) s += row[t] * b[t];
            if (!s.is_zero()) return false;
        }
    return true;
}
bool OwnKills(const std::vector<std::vector<BigInt>>& D0, const DenseCols& Phi) {
    const size_t n0 = D0.empty() ? 0 : D0[0].size();
    for (const auto& v : Phi)
        for (size_t c = 0; c < n0; ++c) {
            BigInt s(0);
            for (size_t t = 0; t < D0.size(); ++t) s += v[t] * D0[t][c];
            if (!s.is_zero()) return false;
        }
    return true;
}

// ---- (F) the integer kernel by unimodular row reduction of [A^T | I] -------------------------------------------------------------
// A: rows x cols dense; returns a Z-basis of {x in Z^cols : A x = 0} (saturated: the transform is unimodular)
DenseCols IntKernel(const std::vector<std::vector<BigInt>>& A, size_t cols) {
    const size_t rows = A.size();
    std::vector<std::vector<BigInt>> M(cols, std::vector<BigInt>(rows + cols, BigInt(0)));
    for (size_t t = 0; t < cols; ++t) {
        for (size_t i = 0; i < rows; ++i) M[t][i] = A[i][t];
        M[t][rows + t] = BigInt(1);
    }
    size_t top = 0;
    for (size_t c = 0; c < rows && top < cols; ++c) {
        while (true) {
            // the smallest non-zero |entry| in column c among rows top..
            size_t best = cols;
            for (size_t t = top; t < cols; ++t)
                if (!M[t][c].is_zero() && (best == cols || M[t][c].abs() < M[best][c].abs())) best = t;
            if (best == cols) break;
            std::swap(M[top], M[best]);
            bool more = false;
            for (size_t t = top + 1; t < cols; ++t) {
                if (M[t][c].is_zero()) continue;
                const BigInt q = M[t][c] / M[top][c];
                for (size_t j = 0; j < rows + cols; ++j)
                    if (!M[top][j].is_zero()) M[t][j] -= q * M[top][j];
                if (!M[t][c].is_zero()) more = true;
            }
            if (!more) {
                ++top;
                break;
            }
        }
    }
    DenseCols K;
    for (size_t t = top; t < cols; ++t) {
        bool zero = true;
        for (size_t i = 0; i < rows; ++i) zero = zero && M[t][i].is_zero();
        if (!zero) continue;
        K.push_back(std::vector<BigInt>(M[t].begin() + static_cast<std::ptrdiff_t>(rows), M[t].end()));
    }
    return K;
}
// coordinates a with Kz a = v (Kz a Z-basis of a saturated lattice containing v), by integer elimination on the transposed system
bool Coords(const DenseCols& Kz, const std::vector<BigInt>& v, std::vector<BigInt>& a) {
    // the lattice basis Kz: find z coordinate rows forming a unimodular-free system; solve by fraction-free Gauss over Q (Cramer-free)
    const size_t z = Kz.size(), n = v.size();
    // augmented rows: [Kz^T rows as columns | v]: solve sum_j a_j Kz[j][i] = v[i] for all i
    std::vector<std::vector<BigInt>> num(n, std::vector<BigInt>(z + 1));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < z; ++j) num[i][j] = Kz[j][i];
        num[i][z] = v[i];
    }
    // rational Gauss-Jordan with a common denominator per row (fraction-free row ops)
    size_t row = 0;
    std::vector<size_t> pc;
    for (size_t c = 0; c < z && row < n; ++c) {
        size_t p = row;
        while (p < n && num[p][c].is_zero()) ++p;
        if (p == n) return false;
        std::swap(num[row], num[p]);
        for (size_t i = 0; i < n; ++i) {
            if (i == row || num[i][c].is_zero()) continue;
            const BigInt f = num[i][c], g = num[row][c];
            for (size_t j = 0; j <= z; ++j) num[i][j] = num[i][j] * g - num[row][j] * f;
            BigInt h(0);
            for (const auto& x : num[i]) h = BigInt::gcd(h, x);
            if (!h.is_zero() && h != BigInt(1))
                for (auto& x : num[i]) x = x / h;
        }
        pc.push_back(c);
        ++row;
    }
    for (size_t i = z; i < n; ++i)
        if (!num[i][z].is_zero()) return false;   // v not in the span
    a.assign(z, BigInt(0));
    for (size_t t = 0; t < z; ++t) {
        BigInt q, rem;
        BigInt::divmod(num[t][z], num[t][t], q, rem);
        if (!rem.is_zero()) return false;   // not integral
        a[t] = q;
    }
    return true;
}
// det[coords(B) | B^1_sat] in Z^1 coordinates
BigInt OracleIndex(const fz::IsoTreeCarrierZ& car, const DenseCols& B, std::string* why) {
    const auto D1 = DenseOf(car.d(1));
    const auto D0 = DenseOf(car.d(0));
    const size_t nv = car.dim(1), n0 = car.dim(0);
    const DenseCols Kz = IntKernel(D1, nv);   // Z^1
    const size_t z = Kz.size();
    if (z != car.dim(0) - car.b0() + car.b1()) {
        if (why) *why = "dim Z^1";
        return BigInt(0);
    }
    // coordinates of the coboundaries (columns of d^0) and of B
    std::vector<std::vector<BigInt>> Lgen;   // rows = generators of Lambda = coords(B^1)
    for (size_t c = 0; c < n0; ++c) {
        std::vector<BigInt> col(nv), a;
        for (size_t i = 0; i < nv; ++i) col[i] = D0[i][c];
        if (!Coords(Kz, col, a)) {
            if (why) *why = "a coboundary outside Z^1";
            return BigInt(0);
        }
        Lgen.push_back(a);
    }
    // F: integer vectors y with Lgen y = 0 (the orthogonal complement), then B^1_sat = ker_Z(F)
    const DenseCols Y = IntKernel(Lgen, z);
    std::vector<std::vector<BigInt>> F(Y.begin(), Y.end());   // rows
    const DenseCols Sat = IntKernel(F, z);
    if (Sat.size() + B.size() != z) {
        if (why) *why = "rank of B^1_sat";
        return BigInt(0);
    }
    DenseCols cols;
    for (const auto& b : B) {
        std::vector<BigInt> a;
        if (!Coords(Kz, b, a)) {
            if (why) *why = "a B column outside Z^1";
            return BigInt(0);
        }
        cols.push_back(a);
    }
    for (const auto& s : Sat) cols.push_back(s);
    // det of the z x z matrix with these columns: OwnDet(I, cols) with Phi = the identity rows
    DenseCols I(z, std::vector<BigInt>(z, BigInt(0)));
    for (size_t i = 0; i < z; ++i) I[i][i] = BigInt(1);
    return OwnDet(I, cols);
}

// ============================================================================================================================
// (M) measurement, first
// ============================================================================================================================
uint32_t ArmMeasure() {
    std::cout << "ARM (M) measurement (fresh process)" << std::endl;
    const uint64_t c0 = g_checks;
    std::vector<std::pair<uint32_t, uint32_t>> cells = {{2, 6}};
    std::string arm = "default";
    if (const char* e = std::getenv("HECKE_EXZ2_CELLS")) {
        if (!*e) {
            std::cout << "HeckeZFreeBasisSatValidation: REFUSED: HECKE_EXZ2_CELLS: empty value (unset it for the default)\n";
            return 1;
        }
        cells.clear();
        arm = "cells";
        std::stringstream ss(e);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            unsigned N = 0, n = 0;
            const int got = std::sscanf(tok.c_str(), "%u:%u", &N, &n);
            if (got < 2 || N == 0 || n % 2 || n > 400) {
                std::cout << "HeckeZFreeBasisSatValidation: REFUSED: HECKE_EXZ2_CELLS: bad entry '" << tok << "' (N:n[:p1/p2], n even <= 400)\n";
                return 1;
            }
            cells.push_back({N, n});
        }
    }
    std::cout << "EFFECTIVE HeckeZFreeBasisSatValidation arm=" << arm << " cells=" << cells.size() << std::endl;
    for (const auto& [N, n] : cells) {
        const std::string tag = Cell(N, n);
        const uint64_t rss0 = integral_descent::CurrentRssBytes();
        integral_descent::RssSampler samp;
        const auto t0 = clk::now();
        const fz::IsoTreeCarrierZ car(N, n);
        const double t_car = since(t0);
        const uint64_t pred = fz::PredictPairsRss(car);
        const uint64_t pred_p1a = fz::PredictPairsRss(car, fz::kPairsK, false, 0);
        const uint64_t rss_car = integral_descent::CurrentRssBytes();
        // the pairs alone first, in this fresh process (the calibration of record for PredictPairsRss, P1a audit F3 / F4)
        uint64_t pairs_growth = 0, sums_growth = 0, sums_pred = 0;
        {
            integral_descent::RssSampler sp;
            const fz::PairsResult Pq = fz::BuildPairs(car);
            const uint64_t pk = sp.stop();
            pairs_growth = pk > rss_car ? pk - rss_car : 0;
            check(!Pq.refused, tag + ": the pairs alone");
            // (fix round 1, F1 (b)) the sums alone on these pairs, from a trimmed heap: the calibration of PredictSumsRss
            TrimHeap();
            sums_pred = fz::PredictSumsRss(car, Pq.pairs);
            const uint64_t rs0 = integral_descent::CurrentRssBytes();
            integral_descent::RssSampler ss;
            const fz::SumsResult S1 = fz::LatticeSums(car, Pq.pairs);
            const uint64_t sk = ss.stop();
            sums_growth = sk > rs0 ? sk - rs0 : 0;
            check(!S1.refused && S1.rss_pred == sums_pred, tag + ": the sums alone (" + S1.refusal + ")");
            check(sums_pred >= sums_growth, tag + ": PredictSumsRss " + std::to_string(sums_pred >> 10) + " KiB >= the sampled sums growth " +
                                                std::to_string(sums_growth >> 10) + " KiB");
        }
        TrimHeap();
        const uint64_t rss_before = integral_descent::CurrentRssBytes();
        fz::FreeBasisOptions fo;
        fo.verbose = true;
        const fz::FreeBasisResult R = fz::BuildFreeBasis(car, fo);
        const uint64_t peak = samp.stop();
        if (!check(!R.refused, tag + ": the free basis is built (" + R.refusal + ")")) continue;
        check(R.cert.ok && R.sat.done, tag + ": the certificate det(Phi^T B) = +-1");
        std::ostringstream o;
        o << "EXZ2_P1B cell=" << N << ":" << n << " D=" << car.D() << " dimC1=" << car.dim(1) << " r=" << car.b1() << " K=" << R.K_used
          << " rule=" << R.basis_rule << " " << fz::Q0Key(R.pairs);
        for (const fz::Pair& P : R.pairs.pairs)
            o << " | s=" << P.s << " rot=" << P.rot << (P.reversed ? "R" : "") << " steps=(" << P.dixB.steps << "," << P.dixPhi.steps << ") bits=("
              << P.bits_B << "," << P.bits_Phi << ") nnz=(" << P.B.nnz() << "," << P.Phi.nnz() << ") fnv=0x" << std::hex << P.fnv_B << "/0x"
              << P.fnv_Phi << std::dec;
        std::cout << o.str() << std::endl;
        std::cout << "EXZ2_P1B_SUMS cell=" << N << ":" << n << " delta_bits=" << R.sums0.delta.bit_length() << " delta=" << fz::FactorString(R.sums0.delta)
                  << " U_bits=" << R.sums0.bits_U0 << "->" << R.sums0.bits_U << " V_bits=" << R.sums0.bits_V0 << "->" << R.sums0.bits_V
                  << " pred_bits_x_g_d=" << R.sums0.pred_bits_x << "/" << R.sums0.pred_bits_g << "/" << R.sums0.pred_bits_d
                  << " peak_bits=" << R.sums0.bits_U_peak << "/" << R.sums0.bits_V_peak << " inline=" << (R.sums0.inline_B ? 1 : 0) << "/"
                  << (R.sums0.inline_Phi ? 1 : 0)
                  << " rel=" << R.sums0.relations_B << "/" << R.sums0.relations_Phi << " lll=" << (R.sums0.lll_B ? 1 : 0) << "/" << (R.sums0.lll_Phi ? 1 : 0) << " bits_Bsum=" << R.sums0.B.max_bits()
                  << " bits_Phisum=" << R.sums0.Phi.max_bits() << " nnz_Bsum=" << R.sums0.B.nnz() << " nnz_Phisum=" << R.sums0.Phi.nnz()
                  << std::fixed << std::setprecision(3) << " t_sums=" << R.sums0.seconds << std::endl;
        for (const fz::SatStep& s : R.sat.log)
            std::cout << "EXZ2_P1B_SAT cell=" << N << ":" << n << " side=" << s.side << " piece=" << s.piece << " mode=" << s.mode << " k=" << s.k
                      << " rounds=" << s.rounds << " cols=" << s.cols << " smith_pivots=" << s.smith_pivots << " smith_ops=" << s.smith_ops
                      << " smith_nnz=" << s.smith_nnz << std::fixed << std::setprecision(3) << " t=" << s.seconds << std::endl;
        std::cout << "EXZ2_P1B_CERT cell=" << N << ":" << n << " saturation=" << R.saturation << " rule=" << R.basis_rule << " " << fz::Q0Key(R.pairs)
                  << " rank_witness_q=" << R.cert.rank_q << " rank_d1=" << R.cert.rank_d1 << " rank_d0=" << R.cert.rank_d0 << " tries="
                  << R.cert.rank_tries << " det=" << R.cert.sign << " primes=" << R.cert.primes
                  << " hadamard_bits=" << R.cert.hadamard_bits << " iterations=" << R.sat.iterations << " splits=" << R.sat.splits
                  << " bits_B=" << R.bits_B << " bits_Phi=" << R.bits_Phi << " nnz_B=" << R.B.nnz() << " nnz_Phi=" << R.Phi_cert.nnz() << std::hex
                  << " basis_fnv=0x" << R.fnv_B << " phi_fnv=0x" << R.fnv_Phi << std::dec << std::fixed << std::setprecision(3)
                  << " t_carrier=" << t_car << " t_pairs=" << R.t_pairs << " t_sums=" << R.t_sums << " t_sat=" << R.t_sat << " t_cert=" << R.t_cert
                  << " rss_pred_pairs_MiB=" << (pred >> 20) << " rss_pred_pairs_p1a_layout_MiB=" << (pred_p1a >> 20)
                  << " rss_pairs_growth_MiB=" << (pairs_growth >> 20) << " rss_pairs_growth_KiB=" << (pairs_growth >> 10)
                  << " rss_pred_sums_MiB=" << (sums_pred >> 20) << " rss_pred_sums_KiB=" << (sums_pred >> 10) << " rss_sums_growth_MiB="
                  << (sums_growth >> 20) << " rss_sums_growth_KiB=" << (sums_growth >> 10)
                  << " rss_growth_freebasis_MiB=" << ((peak > rss_before ? peak - rss_before : 0) >> 20)
                  << " rss_growth_MiB=" << ((peak > rss0 ? peak - rss0 : 0) >> 20) << " rss_peak_MiB=" << (peak >> 20) << std::endl;
    }
    std::cout << "ARM (M) checks=" << g_checks - c0 << std::endl;
    return 0;
}

// ============================================================================================================================
// (S) sums, (B) certificate
// ============================================================================================================================
void ArmSumsCert() {
    std::cout << "ARM (S) the sums, (B) the certificate" << std::endl;
    const auto t0 = clk::now();
    const uint64_t c0 = g_checks;
    struct C {
        uint32_t N, n, delta_bits;
    };
    // the P1a audit's alt2 Delta_Sigma bits (EXZ2P1a_audit.md Sec. 3.2; an independent Python run of the same rule on the same complex)
    for (const C& cl : std::vector<C>{{1, 10, 1}, {2, 6, 2}, {11, 2, 1}, {1, 30, 42}, {5, 12, 29}, {9, 8, 18}}) {
        const std::string tag = Cell(cl.N, cl.n);
        const fz::IsoTreeCarrierZ car(cl.N, cl.n);
        const fz::FreeBasisResult R = fz::BuildFreeBasis(car);
        if (!check(!R.refused, tag + ": built (" + R.refusal + ")")) continue;
        const auto D1 = DenseOf(car.d(1));
        const auto D0 = DenseOf(car.d(0));
        const DenseCols Bs = R.sums0.B.dense(), Ps = R.sums0.Phi.dense();
        check(OwnCocycles(D1, Bs) && OwnKills(D0, Ps), tag + ": (S) B_Sigma cocycles, Phi_Sigma kills d^0 (dense exact)");
        const BigInt dS = OwnDet(Ps, Bs);
        check(dS.abs() == R.sums0.delta, tag + ": (S) |det G_Sigma| (Bareiss) = Delta_Sigma (the HNF diagonals)");
        check(R.sums0.delta.bit_length() == cl.delta_bits,
              tag + ": (S) |Delta_Sigma| bits = the audit's alt2 figure " + std::to_string(cl.delta_bits) + " (got " +
                  std::to_string(R.sums0.delta.bit_length()) + ", " + fz::FactorString(R.sums0.delta) + ")");
        const DenseCols B = R.B.dense(), Phi = R.Phi_cert.dense();
        check(OwnCocycles(D1, B), tag + ": (B) d^1 B = 0 (dense exact, own code)");
        check(OwnKills(D0, Phi), tag + ": (B) Phi_cert^T d^0 = 0 (dense exact, own code)");
        const BigInt det = OwnDet(Phi, B);
        check(det == BigInt(1) || det == BigInt(-1), tag + ": (B) det(Phi_cert^T B) = +-1 (own Bareiss)");
        check(R.cert.ok && det == BigInt(R.cert.sign), tag + ": (B) the library certificate agrees (sign)");
        check(R.fnv_B == fz::MatrixFnv(B, car.dim(1)), tag + ": basis_fnv = FNV of the canonical text");
        std::cout << "  SAT " << tag << " r=" << car.b1() << " delta_sigma=" << fz::FactorString(R.sums0.delta) << " iterations=" << R.sat.iterations
                  << " steps=" << R.sat.log.size() << " splits=" << R.sat.splits << " bits_B=" << R.bits_B << " bits_Phi=" << R.bits_Phi
                  << " basis_fnv=0x" << std::hex << R.fnv_B << std::dec << std::fixed << std::setprecision(3) << " t_pairs=" << R.t_pairs
                  << " t_sums=" << R.t_sums << " t_sat=" << R.t_sat << " t_cert=" << R.t_cert << " U_bits=" << R.sums0.bits_U0 << "->"
                  << R.sums0.bits_U << " V_bits=" << R.sums0.bits_V0 << "->" << R.sums0.bits_V << " bits_Bsum=" << R.sums0.B.max_bits()
                  << " bits_Phisum=" << R.sums0.Phi.max_bits() << std::endl;
    }
    std::cout << "ARM (S)+(B) checks=" << g_checks - c0 << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
}

// ============================================================================================================================
// (K) the escalation orders
// ============================================================================================================================
// fix round 1 (audit F1, F3, F5, F11): the inline reduction, dixon_batch, the ExtendPairs ceiling, the K-escalation loop
void ArmEscalationFix(const fz::IsoTreeCarrierZ& c810, const fz::PairsResult& R5) {
    // F1: inline structural reduction = post-hoc structural reduction, bit for bit, on (8, 10) and (9, 8) with the structural path forced
    {
        const fz::IsoTreeCarrierZ c98(9, 8);
        const fz::PairsResult P98 = fz::BuildPairs(c98);
        const std::vector<fz::Pair> p810(R5.pairs.begin(), R5.pairs.begin() + 3);
        for (int w = 0; w < 2; ++w) {
            const fz::IsoTreeCarrierZ& car = w ? c98 : c810;
            const std::vector<fz::Pair>& pp = w ? P98.pairs : p810;
            const std::string tag = w ? "(9, 8)" : "(8, 10)";
            fz::FreeBasisOptions so;
            so.lll_max_relations = 0;
            fz::testing::FreeBasisHooks post;
            post.sums_posthoc_only = true;
            const fz::SumsResult Si = fz::LatticeSums(car, pp, so);
            const fz::SumsResult Sp = fz::LatticeSums(car, pp, so, &post);
            check(Si.inline_B && Si.inline_Phi && !Sp.inline_B && !Sp.inline_Phi, tag + ": the inline / post-hoc routes ran (F1)");
            check(Si.delta == Sp.delta && fz::PackedFnv(Si.B) == fz::PackedFnv(Sp.B) && fz::PackedFnv(Si.Phi) == fz::PackedFnv(Sp.Phi),
                  tag + ": the inline structural reduction is bit-identical to the post-hoc one (B_Sigma, Phi_Sigma FNV, Delta_Sigma)");
            check(Si.bits_V_peak <= Sp.bits_V0 && Si.bits_U_peak <= Sp.bits_U0,
                  tag + ": the inline transform peak is below the unreduced HNF transform (" + std::to_string(Si.bits_V_peak) + " <= " +
                      std::to_string(Sp.bits_V0) + ")");
            std::cout << "  INLINE " << tag << " peak_bits=" << Si.bits_U_peak << "/" << Si.bits_V_peak << " posthoc_bits=" << Sp.bits_U0 << "/"
                      << Sp.bits_V0 << " reduced=" << Si.bits_U << "/" << Si.bits_V << " fnv=0x" << std::hex << fz::PackedFnv(Si.B) << "/0x"
                      << fz::PackedFnv(Si.Phi) << std::dec << std::fixed << std::setprecision(3) << " t_inline=" << Si.seconds
                      << " t_posthoc=" << Sp.seconds << std::endl;
        }
        // F11: dixon_batch does not change the pairs (schedule independence)
        bool same = true;
        for (uint32_t b : {1u, 8u, 32u}) {
            fz::PairsOptions o;
            o.dixon_batch = b;
            const fz::PairsResult Rb = fz::BuildPairs(c98, o);
            same = same && !Rb.refused && Rb.pairs.size() == P98.pairs.size();
            for (size_t s = 0; s < P98.pairs.size() && same; ++s)
                same = Rb.pairs[s].fnv_B == P98.pairs[s].fnv_B && Rb.pairs[s].fnv_Phi == P98.pairs[s].fnv_Phi;
        }
        check(same, "(9, 8): dixon_batch = 1, 8, 32 give identical pairs (F11)");
    }
    // F5: ExtendPairs above the ceiling is a refusal, nothing built
    {
        fz::PairsResult R3 = fz::BuildPairs(c810);
        fz::PairsOptions o;
        o.rss_cap_bytes = R3.rss_pred;   // = PredictPairsRss(K = 3): K = 4 is above it
        fz::ExtendPairs(c810, R3, o);
        check(R3.refused && R3.refusal.find("REFUSED [RssCeiling] the K-escalation") == 0 && R3.pairs.size() == 3,
              "(8, 10): ExtendPairs above the ceiling: " + R3.refusal);
    }
    // F3: the K-escalation loop of BuildFreeBasis, executed (validation hooks: cofactor limit 2^6, trial division below 8)
    {
        const fz::IsoTreeCarrierZ c130(1, 30);
        fz::testing::FreeBasisHooks hk;
        hk.cofactor_limit_bits = 6;
        hk.trial_bound = 8;
        const fz::FreeBasisResult R = fz::BuildFreeBasis(c130, {}, &hk);
        const bool mw = R.sat.multiword_passes > 0;
        const BigInt det = R.refused ? BigInt(0) : OwnDet(R.Phi_cert.dense(), R.B.dense());
        check(!R.refused && R.K_used == fz::kPairsKMax && mw && R.sat.splits > 0 && R.cert.ok && R.saturation == "det_pm1" &&
                  (det == BigInt(1) || det == BigInt(-1)),
              "(1, 30): the K-escalation loop 3 -> 5, multi-word, splits, certified (own Bareiss) (" + R.refusal + ")");
        std::cout << "  ESCALATION (1, 30) K_used=" << R.K_used << " iterations=" << R.sat.iterations << " steps=" << R.sat.log.size()
                  << " splits=" << R.sat.splits << " multiword=" << (mw ? 1 : 0) << " bits_B=" << R.bits_B << " det=" << det.to_string()
                  << " basis_fnv=0x" << std::hex << R.fnv_B << std::dec << std::endl;
        // F5: the same escalation with the ceiling at the K = 3 prediction: a refusal, not an exception
        // the pairs' ceiling at the K = 3 prediction (K = 4 is above it), the sums' ceiling at its default 3 GiB
        fz::FreeBasisOptions co;
        co.pairs.rss_cap_bytes = fz::PredictPairsRss(c130, fz::kPairsK, true, co.pairs.dixon_batch);
        co.sums_rss_cap_bytes = 3ull << 30;
        check(fz::PredictPairsRss(c130, fz::kPairsK + 1, true, co.pairs.dixon_batch) > co.pairs.rss_cap_bytes,
              "(1, 30): PredictPairsRss(K = 4) > PredictPairsRss(K = 3) (the refusal below is not vacuous)");
        const fz::FreeBasisResult Rc = fz::BuildFreeBasis(c130, co, &hk);
        check(Rc.refused && Rc.refusal.find("REFUSED [RssCeiling] the K-escalation to K = 4") == 0 && Rc.saturation.empty(),
              "(1, 30): a ceiling refusal during the escalation is a refusal: " + Rc.refusal);
    }
}

void ArmEscalation() {
    std::cout << "ARM (K) the K-escalation orders s = 3, 4 (iso_dixon/v3)" << std::endl;
    const uint64_t c0 = g_checks;
    const fz::IsoTreeCarrierZ car(8, 10);
    const uint32_t D = car.D(), nv = car.dim(1);
    std::vector<std::vector<uint32_t>> ord;
    for (uint32_t s = 0; s < fz::kPairsKMax; ++s) ord.push_back(fz::PairOrder(s, D, nv));
    bool distinct = true;
    for (size_t a = 0; a < ord.size(); ++a)
        for (size_t b = a + 1; b < ord.size(); ++b) distinct = distinct && ord[a] != ord[b];
    check(distinct, "(8, 10): pi_0 .. pi_4 pairwise distinct");
    bool unaligned = true;
    for (uint32_t s = 1; s < fz::kPairsKMax; ++s)
        for (uint32_t m = 0; m < 3; ++m) {
            std::vector<uint32_t> rot(nv);
            for (uint32_t i = 0; i < nv; ++i) rot[i] = (i + m * D) % nv;
            unaligned = unaligned && ord[s] != rot;
        }
    check(unaligned, "(8, 10): no pi_s (s >= 1) is a rotation of pi_0 by a multiple of D (not cell-aligned)");
    check(ord[3].front() == (D / 2 + nv - 1) % nv && ord[4].front() == D / 4, "(8, 10): pi_3 = reversed rot(D/2), pi_4 = rot(D/4)");
    fz::PairsOptions o5;
    o5.K = fz::kPairsKMax;
    const fz::PairsResult R5 = fz::BuildPairs(car, o5);
    check(!R5.refused && R5.pairs.size() == 5, "(8, 10): K = 5 pairs built");
    bool valid = true;
    for (const fz::Pair& P : R5.pairs) {
        valid = valid && P.P.size() == car.b1() && fz::ExactCocycles(car, P.B) && fz::ExactKillsD0(car, P.Phi);
        valid = valid && !fz::DetBareiss(fz::PairingMatrix(P.Phi, P.B), car.b1()).is_zero();
        std::cout << "  PAIR (8, 10) s=" << P.s << " rot=" << P.rot << (P.reversed ? " reversed" : "") << " steps=(" << P.dixB.steps << ","
                  << P.dixPhi.steps << ") bits=(" << P.bits_B << "," << P.bits_Phi << ") fnvB=0x" << std::hex << P.fnv_B << std::dec << std::endl;
    }
    check(valid, "(8, 10): every K = 5 pair is valid (cocycles, kills d^0, det != 0)");
    // ExtendPairs from K = 3 gives the same pairs 3, 4
    fz::PairsResult R3 = fz::BuildPairs(car);
    fz::ExtendPairs(car, R3);
    fz::ExtendPairs(car, R3);
    bool same = R3.pairs.size() == 5;
    for (size_t s = 0; s < 5 && same; ++s) same = R3.pairs[s].fnv_B == R5.pairs[s].fnv_B && R3.pairs[s].fnv_Phi == R5.pairs[s].fnv_Phi;
    check(same, "(8, 10): ExtendPairs (3 -> 5) = BuildPairs(K = 5)");
    // Delta_Sigma does not depend on the size reduction (the sum lattices only): skipped here to keep the standard budget
    fz::FreeBasisOptions nored;
    nored.size_reduce = false;
    const fz::SumsResult S3 = fz::LatticeSums(car, std::vector<fz::Pair>(R5.pairs.begin(), R5.pairs.begin() + 3), nored);
    const fz::SumsResult S5 = fz::LatticeSums(car, R5.pairs, nored);
    check(S3.delta.bit_length() == 33, "(8, 10): |Delta_Sigma(K = 3)| bits = the audit's alt2 figure 33 (" + fz::FactorString(S3.delta) + ")");
    check(S3.delta.floor_mod(S5.delta).is_zero(), "(8, 10): Delta_Sigma(K = 5) | Delta_Sigma(K = 3)");
    std::cout << "  SUMS (8, 10) K=3 " << fz::FactorString(S3.delta) << " K=5 " << fz::FactorString(S5.delta) << std::endl;
    // the structural reduction (the path above lll_max_relations), forced on K = 3: the same Delta, exact cocycles / functionals
    {
        fz::FreeBasisOptions so;
        so.lll_max_relations = 0;
        const fz::SumsResult St = fz::LatticeSums(car, std::vector<fz::Pair>(R5.pairs.begin(), R5.pairs.begin() + 3), so);
        check(St.delta == S3.delta && !St.lll_B && !St.lll_Phi && fz::ExactCocycles(car, St.B) && fz::ExactKillsD0(car, St.Phi),
              "(8, 10): the structural size reduction keeps Delta_Sigma and exactness");
        std::cout << "  SUMS (8, 10) structural: U_bits=" << St.bits_U0 << "->" << St.bits_U << " V_bits=" << St.bits_V0 << "->" << St.bits_V
                  << " bits_Bsum=" << St.B.max_bits() << " bits_Phisum=" << St.Phi.max_bits() << std::endl;
    }
    // F10: a q0 that divides 6N or drops a rank moves to the next prime (recorded), it does not throw.  A small q0 whose ranks agree may
    // still give another greedy profile than the rational one (a prime can only delay pivots): then the pairs are valid but not the
    // rule's -- printed as "profile=other" (finding of EXZ2-P1b; at q0 ~ 2^30 the probability is negligible, see the report).
    {
        const fz::IsoTreeCarrierZ c10(1, 10);
        const fz::PairsResult R0 = fz::BuildPairs(c10);
        uint32_t drops = 0;
        for (uint32_t q : {3u, 5u, 7u, 11u, 691u}) {
            fz::PairsOptions o;
            o.q0 = q;
            const fz::PairsResult Rq = fz::BuildPairs(c10, o);
            bool same = !Rq.refused && Rq.pairs.size() == R0.pairs.size();
            for (size_t s = 0; s < R0.pairs.size() && same; ++s)
                same = Rq.pairs[s].fnv_B == R0.pairs[s].fnv_B && Rq.pairs[s].fnv_Phi == R0.pairs[s].fnv_Phi;
            bool valid = !Rq.refused;
            for (const fz::Pair& P : Rq.pairs) valid = valid && fz::ExactCocycles(c10, P.B) && fz::ExactKillsD0(c10, P.Phi) && P.P.size() == c10.b1();
            const bool skipped = !Rq.q0_skipped.empty() && Rq.q0_skipped.front() == q;
            if (skipped && q != 3) ++drops;
            // (fix round 1, F7) the profile is confirmed at a second prime: every requested q0 now gives the rule's pairs (P1b: q0 = 5, 7
            // gave valid pairs with another profile, "profile=other")
            check(valid && (q != 3 || skipped) && same,
                  "(1, 10): q0 = " + std::to_string(q) + " -> q0 = " + std::to_string(Rq.q0) + ": valid pairs, the rule's (profile confirmed)");
            std::cout << "  Q0 (1, 10) requested=" << q << " " << fz::Q0Key(Rq) << " profile=" << (same ? "rational" : "other") << std::endl;
        }
        // no small prime prime to 6N drops a rank on (1, 10) (ranks drop only at the torsion primes): the F10 path forced at the pinned q0
        fz::testing::PairsHooks hk;
        hk.rank_drop_q0 = fz::kQ0;
        const fz::PairsResult Rd = fz::BuildPairs(c10, {}, &hk);
        bool same = !Rd.refused && Rd.pairs.size() == R0.pairs.size() && Rd.q0 != fz::kQ0 && !Rd.q0_skipped.empty() && Rd.q0_skipped[0] == fz::kQ0;
        for (size_t s = 0; s < R0.pairs.size() && same; ++s) same = Rd.pairs[s].fnv_B == R0.pairs[s].fnv_B && Rd.pairs[s].fnv_Phi == R0.pairs[s].fnv_Phi;
        check(same, "(1, 10): a rank drop at q0 moves to the next CrtPrimes prime (" + std::to_string(Rd.q0) + "), recorded, same pairs (F10)");
        (void)drops;
    }
    ArmEscalationFix(car, R5);
    std::cout << "ARM (K) checks=" << g_checks - c0 << std::endl;
}

// ============================================================================================================================
// (E) basis teeth, (F) the global oracle
// ============================================================================================================================
fz::PackedZ ScaleCol(const fz::PackedZ& X, uint32_t j, const BigInt& f) {
    fz::PackedZ Y(X.rows);
    for (uint32_t c = 0; c < X.cols; ++c) {
        auto e = X.col_sparse(c);
        if (c == j)
            for (auto& kv : e) kv.second = kv.second * f;
        Y.append_col(e);
    }
    return Y;
}
// X M on the first three columns, M upper triangular (columns: m_00 e_0; m_01 e_0 + m_11 e_1; m_02 e_0 + m_12 e_1 + m_22 e_2)
fz::PackedZ TriCombine(const fz::PackedZ& X, const int64_t m[3][3]) {
    const std::vector<std::vector<BigInt>> D = X.dense();
    fz::PackedZ Y(X.rows);
    for (uint32_t c = 0; c < X.cols; ++c) {
        if (c >= 3) {
            Y.append_col(X.col_sparse(c));
            continue;
        }
        std::vector<BigInt> v(X.rows, BigInt(0));
        for (uint32_t k = 0; k <= c; ++k)
            for (uint32_t i = 0; i < X.rows; ++i) v[i] += D[k][i] * BigInt(m[k][c]);
        Y.append_dense_col(v);
    }
    return Y;
}
fz::PackedZ DropLast(const fz::PackedZ& X) {
    fz::PackedZ Y(X.rows);
    for (uint32_t c = 0; c + 1 < X.cols; ++c) Y.append_col(X.col_sparse(c));
    return Y;
}
// fix round 1 (audit F2, F4, F6): the gcd split, the side Phi, a non-diagonal defect, the stall, the rank witness
void ArmTeethFix(const fz::IsoTreeCarrierZ& car, const fz::FreeBasisResult& R) {
    const BigInt p1 = BigInt::from_string("1048583"), p2 = BigInt::from_string("2147483659");
    const BigInt P1 = BigInt::from_string("34359738421"), P2 = BigInt::from_string("34359738451");
    struct T {
        const char* name;
        int kind;   // 0: B cols 0, 1 scaled by f0, f1; 1: Phi col 0 scaled by f0; 2: B M (triangular)
        BigInt f0, f1;
        const char* mode;
        bool want_split;
        char side;
    };
    const std::vector<T> teeth = {
        {"b1*p1,b2*p2", 0, p1, p2, "composite64", true, 'B'},
        {"b1*P1,b2*P2", 0, P1, P2, "multiword", true, 'B'},
        {"phi1*p2", 1, p2, BigInt(1), "composite64", false, 'P'},
        {"phi1*P1*P2", 1, P1 * P2, BigInt(1), "multiword", false, 'P'},
        {"phi1*7", 1, BigInt(7), BigInt(1), "local", false, 'P'},
        {"B*M(8,9,7)", 2, BigInt(1), BigInt(1), "local", false, 'B'},
    };
    const int64_t M[3][3] = {{8, 1, 2}, {0, 9, 3}, {0, 0, 7}};
    for (const T& t : teeth) {
        fz::PackedZ B = R.B, Phi = R.Phi_cert;
        if (t.kind == 0) B = ScaleCol(ScaleCol(R.B, 0, t.f0), 1, t.f1);
        if (t.kind == 1) Phi = ScaleCol(R.Phi_cert, 0, t.f0);
        if (t.kind == 2) B = TriCombine(R.B, M);
        const BigInt d = OwnDet(Phi.dense(), B.dense()).abs();
        const BigInt want = t.kind == 2 ? BigInt(504) : t.f0 * t.f1;
        check(d == want && !fz::CertifyBasis(car, B, Phi).ok, "(9, 8): tooth " + std::string(t.name) + ": |det| = " + d.to_string() + ", fails");
        const fz::SaturationResult s = fz::Saturate(car, B, Phi, d, fz::FreeBasisOptions{}, false);
        // a multi-word pass that ends in a gcd split is not logged as a step: it counts in multiword_passes
        bool mode = std::string(t.mode) == "multiword" && s.multiword_passes > 0, side_ok = true, side_hit = false;
        for (const auto& st : s.log) {
            mode = mode || st.mode == t.mode;
            if (st.cols) {
                side_hit = side_hit || st.side == t.side;
                side_ok = side_ok && st.side == t.side;
            }
        }
        const fz::Certificate c2 = fz::CertifyBasis(car, B, Phi);
        const BigInt d2 = OwnDet(Phi.dense(), B.dense());
        check(s.done && c2.ok && (d2 == BigInt(1) || d2 == BigInt(-1)) && mode,
              "(9, 8): tooth " + std::string(t.name) + " repaired in " + t.mode + " mode (certificate, own Bareiss)");
        check(side_hit && side_ok, "(9, 8): tooth " + std::string(t.name) + ": the replaced columns are on side " + std::string(1, t.side));
        check(t.want_split ? s.splits >= 1 : s.splits == 0,
              "(9, 8): tooth " + std::string(t.name) + ": splits = " + std::to_string(s.splits) + (t.want_split ? " >= 1" : " = 0"));
        std::cout << "  TOOTH (9, 8) " << t.name << " steps=" << s.log.size() << " splits=" << s.splits << " multiword_passes=" << s.multiword_passes
                  << " iterations=" << s.iterations;
        for (const auto& st : s.log) std::cout << " [" << st.side << " " << st.piece << " " << st.mode << " k=" << st.k << " cols=" << st.cols << "]";
        std::cout << std::endl;
    }
    // F4: Delta given too large (49 for 7 b_1): repaired, then the next iteration replaces nothing -- refused at once
    {
        fz::PackedZ B = ScaleCol(R.B, 0, BigInt(7)), Phi = R.Phi_cert;
        const fz::SaturationResult s = fz::Saturate(car, B, Phi, BigInt(49), fz::FreeBasisOptions{}, false);
        check(s.refused && s.stalled && s.iterations == 2 && s.refusal.find("REFUSED [SaturationStall]") == 0 && fz::CertifyBasis(car, B, Phi).ok,
              "(9, 8): a stalled iteration is refused at once (" + s.refusal + ", iterations " + std::to_string(s.iterations) + ")");
    }
    // F6: the rank premise -- the first r - 1 columns of a certified pair are not certified (rank witness)
    {
        const fz::Certificate c1 = fz::CertifyBasis(car, DropLast(R.B), DropLast(R.Phi_cert));
        check(!c1.rank_ok && !c1.ok && c1.rank_q != 0, "(9, 8): r - 1 columns fail the rank witness (rank_q d^1 " + std::to_string(c1.rank_d1) +
                                                          " + rank_q d^0 " + std::to_string(c1.rank_d0) + " + r - 1 < dim C^1)");
        check(R.cert.rank_ok && R.cert.rank_q != fz::kQ0 && R.cert.rank_tries == 1,
              "(9, 8): the certified basis has its rank witness at the first prime != q0 (q_w = " + std::to_string(R.cert.rank_q) + ")");
    }
}
void ArmTeeth() {
    std::cout << "ARM (E) basis teeth, (F) the global oracle" << std::endl;
    const uint64_t c0 = g_checks;
    // (E) on (9, 8): 2, 3 local; 7 local; 2147483659 composite (> 2^20); a 70-bit composite: multi-word
    {
        const fz::IsoTreeCarrierZ car(9, 8);
        const fz::FreeBasisResult R = fz::BuildFreeBasis(car);
        if (check(!R.refused && R.cert.ok, "(9, 8): the certified basis")) {
            struct T {
                const char* f;
                const char* mode;
            };
            const BigInt big = BigInt::from_string("34359738421") * BigInt::from_string("34359738451");   // two primes > 2^35: a 71-bit cofactor
            for (const T& t : std::vector<T>{{"7", "local"}, {"2", "local"}, {"2147483659", "composite64"}, {"", "multiword"}}) {
                const BigInt f = *t.f ? BigInt::from_string(t.f) : big;
                fz::PackedZ B = ScaleCol(R.B, 0, f), Phi = R.Phi_cert;
                const fz::Certificate c1 = fz::CertifyBasis(car, B, Phi);
                const BigInt d = OwnDet(Phi.dense(), B.dense());
                check(!c1.ok && d.abs() == f, "(9, 8): " + std::string(t.mode) + " tooth f b_1: the certificate fails, |det| = f");
                fz::FreeBasisOptions o;
                const fz::SaturationResult s = fz::Saturate(car, B, Phi, f, o, false);
                bool used = false;
                for (const auto& st : s.log) used = used || st.mode == t.mode;
                const fz::Certificate c2 = fz::CertifyBasis(car, B, Phi);
                check(s.done && c2.ok && used, "(9, 8): f b_1 repaired by Saturate in " + std::string(t.mode) + " mode (det = +-1)");
                // (fix round 1, F2 / F8) a B-only defect: no column replaced on side Phi, and no side-Phi pass once the piece is gone
                bool phi_pass = false;
                for (const auto& st : s.log) phi_pass = phi_pass || st.side == 'P';
                check(!phi_pass && s.splits == 0, "(9, 8): " + std::string(t.mode) + " tooth: side B only, no side-Phi pass (F8), no split");
                std::cout << "  TOOTH (9, 8) f=" << (*t.f ? t.f : "c70b") << " steps=" << s.log.size() << " splits=" << s.splits << " repaired="
                          << (c2.ok ? 1 : 0) << std::endl;
            }
            ArmTeethFix(car, R);
            // multi-word disabled: the resource refusal
            {
                fz::PackedZ B = ScaleCol(R.B, 0, big), Phi = R.Phi_cert;
                fz::FreeBasisOptions o;
                o.multiword = false;
                const fz::SaturationResult s = fz::Saturate(car, B, Phi, big, o, false);
                check(s.refused && s.refusal.find("REFUSED [SaturationCofactor]") == 0, "(9, 8): multiword off, K = K_max: " + s.refusal);
            }
            // B U, U unimodular (b_0 += 3 b_1): the certificate holds
            {
                fz::PackedZ BU(R.B.rows);
                auto c0v = R.B.col_dense(0), c1v = R.B.col_dense(1);
                for (size_t i = 0; i < c0v.size(); ++i) c0v[i] += BigInt(3) * c1v[i];
                BU.append_dense_col(c0v);
                for (uint32_t j = 1; j < R.B.cols; ++j) BU.append_col(R.B.col_sparse(j));
                check(fz::CertifyBasis(car, BU, R.Phi_cert).ok, "(9, 8): B U (U unimodular) keeps det = +-1");
            }
        }
    }
    // (F) the global oracle
    for (const auto& [N, n] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 10}, {2, 6}, {11, 2}}) {
        const std::string tag = Cell(N, n);
        const fz::IsoTreeCarrierZ car(N, n);
        const fz::FreeBasisResult R = fz::BuildFreeBasis(car);
        if (!check(!R.refused && R.cert.ok, tag + ": certified")) continue;
        std::string why;
        const DenseCols B = R.B.dense();
        const BigInt idx = OracleIndex(car, B, &why);
        check(idx == BigInt(1) || idx == BigInt(-1), tag + ": (F) the oracle: [Z^1 : B^1_sat + Z B] = 1 (" + why + ", det " + idx.to_string() + ")");
        DenseCols B7 = B;
        for (auto& x : B7[0]) x = x * BigInt(7);
        const BigInt idx7 = OracleIndex(car, B7, &why);
        check(idx7.abs() == BigInt(7), tag + ": (F) the oracle sees 7 b_1 (index 7, got " + idx7.to_string() + ")");
    }
    std::cout << "ARM (E)+(F) checks=" << g_checks - c0 << std::endl;
}

}  // namespace

uint32_t HeckeZFreeBasisSatValidation() {
    std::cout << "HeckeZFreeBasisSatValidation: EXPORT-Z step 2 P1b -- lattice sums, saturation, the det = +-1 certificate (" << fz::kBasisRule << ")"
              << std::endl;
    const auto t0 = clk::now();
    g_fail = 0;
    g_checks = 0;
    const uint32_t refused = ArmMeasure();
    ArmSumsCert();
    ArmEscalation();
    ArmTeeth();
    const bool ok = g_fail == 0 && refused == 0;
    std::cout << "HECKEZ_FREEBASIS_SAT arms=6 failing=" << g_fail << " checks=" << g_checks << " rule=" << fz::kBasisRule
              << " saturation=det_pm1 verdict=" << (ok ? "OK" : "FAIL") << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
    return ok ? 0 : 1;
}
