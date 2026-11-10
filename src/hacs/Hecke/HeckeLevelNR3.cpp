// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-15 WP-9 -- (R3) at scale. See HeckeLevelNR3.h.
#include "HeckeLevelNR3.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

#include "CoefficientRing.h"   // hecke::IsPrimeU64
#include "HeckeLevelNSparse.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../Util/Timing.h"

namespace hecke {
namespace level {

namespace {

using oc_util::Seconds;

using Buf = engine::DenseBuf<uint32_t>;
using Ctx = engine::ModCtx<uint32_t>;

// C = A * B (A: m x k, B: k x n); k == 0 gives the zero matrix.
Buf Mul(const Ctx& ctx, const Buf& A, const Buf& B) {
    ENGINE_CHECK(A.cols == B.rows);
    Buf C(A.rows, B.cols);
    if (A.rows && B.cols && A.cols) engine::kernels::gemm(ctx, engine::cview(A), engine::cview(B), engine::view(C));
    return C;
}

// A basis (as columns, w x k) of the generalised kernel of the square X (w x w): ker X^{2^j} once
// rank X^{2^j} == rank X^{2^{j+1}} (the kernel chain of powers is monotone, so it is then stable).
Buf GeneralisedKernel(const Ctx& ctx, const Buf& X) {
    const uint32_t w = X.rows;
    ENGINE_CHECK(X.cols == w);
    if (w == 0) return Buf(0, 0);
    Buf P = X;
    Buf R = P;
    engine::elim::RrefResult rr = engine::elim::rref(ctx, engine::view(R));
    // The chain rank X^{2^j} is strictly decreasing until it stabilises, so it stabilises after at most
    // log2(w) + 1 < 33 squarings; exhausting the 40-step guard means a broken invariant and throws (WP-17, WP-9
    // audit item 4 -- before, the loop exited silently with a kernel that need not be the generalised one).
    bool stable = rr.rank == 0;
    for (uint32_t guard = 0; guard < 40 && !stable; ++guard) {
        Buf P2 = Mul(ctx, P, P);
        Buf R2 = P2;
        engine::elim::RrefResult rr2 = engine::elim::rref(ctx, engine::view(R2));
        if (rr2.rank == rr.rank) { stable = true; break; }
        P = std::move(P2);
        R = std::move(R2);
        rr = std::move(rr2);
        stable = rr.rank == 0;
    }
    ENGINE_CHECK_MSG(stable, "ReadEisensteinR3Scan: GeneralisedKernel: the rank chain of X^{2^j} did not stabilise in 40 squarings");
    const Buf K = engine::elim::kernel_basis(ctx, engine::cview(R), rr.pivots);   // (w - rank) x w, rows
    Buf G(w, K.rows);
    for (uint32_t j = 0; j < K.rows; ++j)
        for (uint32_t i = 0; i < w; ++i) G.at(i, j) = K.at(j, i);
    return G;
}

// Coordinates Z (w x c) of the columns of Y (n x c) in the column basis B (n x w) of full rank;
// throws (ENGINE_CHECK semantics) when a column of Y leaves span B.
Buf CoordsIn(const Ctx& ctx, const Buf& B, const Buf& Y) {
    Buf Z(B.cols, Y.cols);
    if (B.cols == 0 || Y.cols == 0) return Z;
    std::vector<std::vector<uint32_t>> s;
    try {
        s = engine::elim::solve_multi_rhs(ctx, engine::cview(B), engine::cview(Y));
    } catch (const std::runtime_error& e) {
        ENGINE_CHECK_MSG(false, (std::string("ReadEisensteinR3Scan: an image leaves W (W not stable): ") + e.what()).c_str());
    }
    ENGINE_CHECK(s.size() == Y.cols);
    for (uint32_t t = 0; t < Y.cols; ++t) {
        ENGINE_CHECK(s[t].size() == B.cols);
        for (uint32_t j = 0; j < B.cols; ++j) Z.at(j, t) = s[t][j];
    }
    return Z;
}

} // namespace

std::vector<uint32_t> R3HeckePrimes(uint64_t N, uint32_t qmax) {
    std::vector<uint32_t> qs;
    for (uint32_t q = 2; q <= qmax; ++q)
        if (hecke::IsPrimeU64(q) && q != N) qs.push_back(q);
    return qs;
}

bool MerelCriterionLN5(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 3 && N < (1ull << 31) && hecke::IsPrimeU64(N) && p >= 2 && hecke::IsPrimeU64(p) && (N - 1) % p == 0,
                     ("MerelCriterionLN5: N prime < 2^31, p prime and p | N - 1 (LN5); got N=" + std::to_string(N) +
                      " p=" + std::to_string(p)).c_str());
    // x = prod_{k <= (N-1)/2} k^k mod N; Merel's sum_k k log k == 0 mod p  <=>  x is a p-th power  <=>
    // x^{(N-1)/p} == 1 ((Z/N)^x cyclic of order N - 1, p | N - 1).  x is a unit (every k < N).
    uint64_t x = 1;
    for (uint64_t k = 1; k <= (N - 1) / 2; ++k) x = static_cast<uint64_t>(static_cast<unsigned __int128>(x) * engine::scalar::powmod<uint64_t>(k, k, N) % N);
    return engine::scalar::powmod<uint64_t>(x, (N - 1) / p, N) == 1;
}

uint32_t NumConditionValuation(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 2 && p >= 2 && hecke::IsPrimeU64(p), "NumConditionValuation: N >= 2, p prime");
    uint64_t num = N - 1, a = N - 1, b = 12;
    while (b) { const uint64_t t = a % b; a = b; b = t; }
    num /= a;   // (N - 1)/gcd(N - 1, 12)
    uint32_t v = 0;
    while (num % p == 0) { num /= p; ++v; }
    return v;
}

R3ScanReading ReadEisensteinR3Scan(uint64_t N, uint64_t p, const std::vector<uint32_t>& qs_in) {
    ENGINE_CHECK_MSG(N >= 2 && N < (1ull << 31), "ReadEisensteinR3Scan: 2 <= N < 2^31");
    // (slot-22 LN23) p in {2, 3}: the curve object (n = 0, the orbit-sum presentation), at prime levels only.
    const bool curve = p == 2 || p == 3;
    if (curve)
        ENGINE_CHECK_MSG(hecke::IsPrimeU64(N) && N % p != 0,
                         "ReadEisensteinR3Scan: p in {2, 3} reads the curve object (LN23) at N prime, p !| N only");
    else
        ENGINE_CHECK_MSG(p >= 5 && p < (1ull << 31) && hecke::IsPrimeU64(p) && N % p != 0,
                         "ReadEisensteinR3Scan: p >= 5 prime (6 a unit), p !| N");
    R3ScanReading R;
    R.N = N;
    R.p = p;
    R.object = curve ? "curve" : "group";
    R.num_condition = NumConditionValuation(N, p);
    R.exact = R.num_condition == 1;
    for (uint32_t q : qs_in) {
        ENGINE_CHECK_MSG(q >= 2 && hecke::IsPrimeU64(q), "ReadEisensteinR3Scan: every q prime");
        if (q != N) R.qs.push_back(q);
    }
    ENGINE_CHECK_MSG(!R.qs.empty(), "ReadEisensteinR3Scan: no Hecke prime q != N");
    for (uint32_t q : R.qs)
        if (MazurGoodPrime(N, p, q)) R.good.push_back(q);
    const Ctx ctx(static_cast<uint32_t>(p));
    const uint32_t P = static_cast<uint32_t>(p);

    auto t0 = std::chrono::steady_clock::now();
    const Gamma0LevelN lvl(N);
    const SparseCokernel Q = curve ? SparseCokernel::BuildCurve(lvl, 0, ctx, false) : SparseCokernel::Build(lvl, 0, ctx, false);
    R.presentation_s = Seconds(t0);
    R.final_nnz = Q.stats().final_nnz;
    R.dimH1 = Q.h1();
    const uint32_t h = Q.h1();

    // X_q = [H_{N,q}] - (1 + q) on H^1 (the same matrices as ReadEisensteinR3's).
    t0 = std::chrono::steady_clock::now();
    std::vector<Buf> X;
    for (uint32_t q : R.qs) {
        const HMat M = OnCokernelSparse(Q, HeilbronnOperator(lvl, q, 0, ctx));
        ENGINE_CHECK(M.rows == h && M.cols == h);
        Buf A(h, h);
        for (uint32_t i = 0; i < h; ++i)
            for (uint32_t j = 0; j < h; ++j) A.at(i, j) = static_cast<uint32_t>(M.at(i, j) % P);
        const uint32_t lam = static_cast<uint32_t>((1 + q) % p);
        for (uint32_t i = 0; i < h; ++i) A.at(i, i) = (A.at(i, i) + P - lam) % P;
        X.push_back(std::move(A));
    }
    R.operators_s = Seconds(t0);

    // W: the joint generalised eigenspace, one operator at a time, each restricted to the running W.
    t0 = std::chrono::steady_clock::now();
    Buf Wb = GeneralisedKernel(ctx, X[0]);   // h x w
    R.w_after.push_back(Wb.cols);
    for (size_t k = 1; k < X.size() && Wb.cols > 0; ++k) {
        const Buf Z = CoordsIn(ctx, Wb, Mul(ctx, X[k], Wb));   // w x w
        const Buf G = GeneralisedKernel(ctx, Z);               // w x k'
        Wb = Mul(ctx, Wb, G);
        R.w_after.push_back(Wb.cols);
    }
    const uint32_t w = Wb.cols;
    R.dimW = w;
    R.W.assign(static_cast<size_t>(h) * w, 0);
    for (uint32_t i = 0; i < h; ++i)
        for (uint32_t j = 0; j < w; ++j) R.W[static_cast<size_t>(i) * w + j] = Wb.at(i, j);
    std::vector<Buf> Zs;   // every X_q restricted to W (W stable: checked by CoordsIn)
    for (const Buf& Xq : X) Zs.push_back(CoordsIn(ctx, Wb, Mul(ctx, Xq, Wb)));

    // The layers m^i W: the span of the images of the previous layer under every Z_q.
    if (w > 0) {
        R.layers.push_back(w);
        Buf cur(w, w);
        for (uint32_t i = 0; i < w; ++i) cur.at(i, i) = 1;
        for (int guard = 0; guard < 256 && cur.cols > 0; ++guard) {
            const uint32_t c = cur.cols, cc = c * static_cast<uint32_t>(Zs.size());
            Buf T(cc, w);   // the image vectors as rows
            for (size_t z = 0; z < Zs.size(); ++z) {
                const Buf Im = Mul(ctx, Zs[z], cur);   // w x c
                for (uint32_t j = 0; j < c; ++j)
                    for (uint32_t i = 0; i < w; ++i) T.at(static_cast<uint32_t>(z * c + j), i) = Im.at(i, j);
            }
            const engine::elim::RrefResult rr = engine::elim::rref(ctx, engine::view(T));
            if (rr.rank == 0) break;
            ENGINE_CHECK_MSG(rr.rank < c, "ReadEisensteinR3Scan: the radical layers must strictly decrease (nilpotent action)");
            R.layers.push_back(rr.rank);
            Buf nxt(w, rr.rank);
            for (uint32_t j = 0; j < rr.rank; ++j)
                for (uint32_t i = 0; i < w; ++i) nxt.at(i, j) = T.at(j, i);
            cur = std::move(nxt);
        }
    }
    R.loewy = static_cast<uint32_t>(R.layers.size());
    R.eisenstein_s = Seconds(t0);
    return R;
}

bool MazurGoodPrime(uint64_t N, uint64_t p, uint64_t q) {
    ENGINE_CHECK_MSG(N >= 2 && N < (1ull << 62) && hecke::IsPrimeU64(N) && hecke::IsPrimeU64(p) && hecke::IsPrimeU64(q),
                     "MazurGoodPrime: N, p, q prime");
    if (q == N || (p == 2 && q == 2)) return false;   // q = p = 2: case (b), not a generator (Prop. (17.1))
    if (NumConditionValuation(N, p) == 0) return false;   // p !| n = num((N - 1)/12): no Eisenstein prime above p (9.7)
    if (engine::scalar::powmod<uint64_t>(q % N, (N - 1) / p, N) == 1) return false;   // (i): q a p-th power mod N
    // (ii) (q - 1)/2 !== 0 mod p: p = 2 -> q = 3 mod 4 (q odd here); p odd -> q != 1 mod p (q = 2: 1/2, a unit).
    return p == 2 ? q % 4 == 3 : q % p != 1;
}

// ---- R3FamilyReader ---------------------------------------------------------------------------------------------
struct R3FamilyReader::Impl {
    Ctx ctx;
    Gamma0LevelN lvl;
    SparseCokernel Q;
    R3ScanReading R;          // N, p, object, num_condition, exact, dimH1, qs, good, w_after
    bool started = false;
    Buf Wb;                   // h x w, the running W
    std::vector<Buf> Zs;      // every X_q restricted to the running W (w x w), in the order of R.qs
    Impl(uint64_t N, uint64_t p, bool curve)
        : ctx(static_cast<uint32_t>(p)), lvl(N),
          Q(curve ? SparseCokernel::BuildCurve(lvl, 0, ctx, false) : SparseCokernel::Build(lvl, 0, ctx, false)) {}
};

R3FamilyReader::R3FamilyReader(uint64_t N, uint64_t p) {
    ENGINE_CHECK_MSG(N >= 2 && N < (1ull << 31), "R3FamilyReader: 2 <= N < 2^31");
    const bool curve = p == 2 || p == 3;
    if (curve)
        ENGINE_CHECK_MSG(hecke::IsPrimeU64(N) && N % p != 0, "R3FamilyReader: p in {2, 3} reads the curve object (LN23) at N prime, p !| N only");
    else
        ENGINE_CHECK_MSG(p >= 5 && p < (1ull << 31) && hecke::IsPrimeU64(p) && N % p != 0, "R3FamilyReader: p >= 5 prime (6 a unit), p !| N");
    const auto t0 = std::chrono::steady_clock::now();
    impl_ = std::make_unique<Impl>(N, p, curve);
    R3ScanReading& R = impl_->R;
    R.N = N;
    R.p = p;
    R.object = curve ? "curve" : "group";
    R.num_condition = NumConditionValuation(N, p);
    R.exact = R.num_condition == 1;
    R.presentation_s = Seconds(t0);
    R.final_nnz = impl_->Q.stats().final_nnz;
    R.dimH1 = impl_->Q.h1();
}

R3FamilyReader::~R3FamilyReader() = default;

uint32_t R3FamilyReader::dimH1() const { return impl_->R.dimH1; }
uint32_t R3FamilyReader::dimW() const { return impl_->started ? impl_->Wb.cols : impl_->R.dimH1; }

void R3FamilyReader::Extend(const std::vector<uint32_t>& qs) {
    Impl& I = *impl_;
    R3ScanReading& R = I.R;
    const uint32_t h = R.dimH1, P = static_cast<uint32_t>(R.p);
    for (uint32_t q : qs) {
        ENGINE_CHECK_MSG(q >= 2 && hecke::IsPrimeU64(q), "R3FamilyReader: every q prime");
        if (q == R.N) continue;
        bool have = false;
        for (uint32_t r : R.qs) have = have || r == q;
        if (have) continue;
        R.qs.push_back(q);
        if (MazurGoodPrime(R.N, R.p, q)) R.good.push_back(q);
        // X_q = [H_{N,q}] - (1 + q) on H^1 (ReadEisensteinR3Scan's matrix).
        auto t0 = std::chrono::steady_clock::now();
        Buf X(h, h);
        {
            const HMat M = OnCokernelSparse(I.Q, HeilbronnOperator(I.lvl, q, 0, I.ctx));
            ENGINE_CHECK(M.rows == h && M.cols == h);
            for (uint32_t i = 0; i < h; ++i)
                for (uint32_t j = 0; j < h; ++j) X.at(i, j) = static_cast<uint32_t>(M.at(i, j) % P);
        }
        const uint32_t lam = static_cast<uint32_t>((1 + q) % R.p);
        for (uint32_t i = 0; i < h; ++i) X.at(i, i) = (X.at(i, i) + P - lam) % P;
        R.operators_s += Seconds(t0);
        t0 = std::chrono::steady_clock::now();
        if (!I.started) {
            I.Wb = GeneralisedKernel(I.ctx, X);   // h x w
            I.started = true;
            I.Zs.push_back(CoordsIn(I.ctx, I.Wb, Mul(I.ctx, X, I.Wb)));
        } else {
            Buf Z = CoordsIn(I.ctx, I.Wb, Mul(I.ctx, X, I.Wb));   // w x w
            const Buf G = GeneralisedKernel(I.ctx, Z);           // w x w'
            I.Wb = Mul(I.ctx, I.Wb, G);
            // Restrict every operator to the new W = old W * G (G of full column rank, W' stable): coordinates in G.
            for (Buf& Zq : I.Zs) Zq = CoordsIn(I.ctx, G, Mul(I.ctx, Zq, G));
            I.Zs.push_back(CoordsIn(I.ctx, G, Mul(I.ctx, Z, G)));
        }
        R.w_after.push_back(I.Wb.cols);
        R.eisenstein_s += Seconds(t0);
    }
}

R3ScanReading R3FamilyReader::Reading() const {
    const Impl& I = *impl_;
    ENGINE_CHECK_MSG(I.started, "R3FamilyReader::Reading: no Hecke prime q != N added");
    const auto t0 = std::chrono::steady_clock::now();
    R3ScanReading R = I.R;
    const uint32_t h = R.dimH1, w = I.Wb.cols;
    R.dimW = w;
    R.W.assign(static_cast<size_t>(h) * w, 0);
    for (uint32_t i = 0; i < h; ++i)
        for (uint32_t j = 0; j < w; ++j) R.W[static_cast<size_t>(i) * w + j] = I.Wb.at(i, j);
    // The layers m^i W, exactly as ReadEisensteinR3Scan.
    if (w > 0) {
        R.layers.push_back(w);
        Buf cur(w, w);
        for (uint32_t i = 0; i < w; ++i) cur.at(i, i) = 1;
        for (int guard = 0; guard < 256 && cur.cols > 0; ++guard) {
            const uint32_t c = cur.cols, cc = c * static_cast<uint32_t>(I.Zs.size());
            Buf T(cc, w);
            for (size_t z = 0; z < I.Zs.size(); ++z) {
                const Buf Im = Mul(I.ctx, I.Zs[z], cur);
                for (uint32_t j = 0; j < c; ++j)
                    for (uint32_t i = 0; i < w; ++i) T.at(static_cast<uint32_t>(z * c + j), i) = Im.at(i, j);
            }
            const engine::elim::RrefResult rr = engine::elim::rref(I.ctx, engine::view(T));
            if (rr.rank == 0) break;
            ENGINE_CHECK_MSG(rr.rank < c, "R3FamilyReader: the radical layers must strictly decrease (nilpotent action)");
            R.layers.push_back(rr.rank);
            Buf nxt(w, rr.rank);
            for (uint32_t j = 0; j < rr.rank; ++j)
                for (uint32_t i = 0; i < w; ++i) nxt.at(i, j) = T.at(j, i);
            cur = std::move(nxt);
        }
    }
    R.loewy = static_cast<uint32_t>(R.layers.size());
    R.eisenstein_s += Seconds(t0);
    return R;
}

} // namespace level
} // namespace hecke
